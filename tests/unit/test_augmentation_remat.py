"""The tabulated augmentation scan must not stack its residuals.

`MEMORY-AUDIT.md` A1. The tabulated route rebuilds ``Q_ij(G)`` a chunk at a time
inside a ``lax.scan``, which is what keeps the dense table off the *forward*
working set (P73). Under ``jax.grad`` that saving evaporates unless the body is
rematted: ``build(gcart_chunk)`` is a known value while the other factor is not,
so transposing the contraction needs ``Q`` -- and because ``build``'s argument
comes from a ``dynamic_slice`` on the scan index it is **not** loop-invariant, so
partial evaluation stacks it into a ``(nchunks, nh, nh, chunk)`` residual. That
is the dense table reborn on the tape.

**It had an exact floor, which is why this is worth a test rather than a
comment.** :func:`~defumat.pseudo.augmentation.build_augmentation` takes this
route only when ``nh^2 ngm x 16 > DEFUMAT_AUG_MAX_BYTES`` (2 GiB), and the
stacked residual equals that product to within ``npad/ngm``. So the regression is
not "a bit more memory": on every cell that takes the path the tape would be
**at least 2 GiB by construction**. Measured on ``bismuthene-soc-small``, the
force tape is 2.32 GiB without the remat and 0.99 GiB with it.

**Why a jaxpr test and not a memory one.** A peak-RSS assertion on this would be
a flake; the property that matters is structural and is visible in the traced
program, where it can be asserted exactly. And a ``@jax.checkpoint`` decorator is
precisely the kind of line a later refactor removes without noticing, because
nothing else in the package is rematted at all -- ``grep -rn "jax.checkpoint"``
returned one hit before A1, and it was a comment.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat.pseudo.augmentation import _tabulated_charge, _tabulated_integrals

pytestmark = [pytest.mark.unit]

#: Toy shapes. Nothing here is physical -- what is asserted is the *shape* of the
#: traced program, which does not depend on the numbers.
NH, NAT, NPAD, CHUNK = 3, 2, 20, 5
NCHUNKS = NPAD // CHUNK


def _build(gcart):
    """Stands in for the radial-to-G transform: a function of ``gcart`` alone.

    That is the whole of the mechanism -- it makes ``Q`` a *known* value inside
    the body, which is what lets partial evaluation stack it as a residual.
    """
    radius = jnp.linalg.norm(gcart, axis=-1)
    decay = jnp.arange(1, NH * NH + 1, dtype=radius.dtype).reshape(NH, NH)
    return jnp.exp(-radius[None, None, :] * decay[..., None]) + 0j


def _stacked_leading(jaxpr_text, leading=NCHUNKS):
    """Shapes in the jaxpr whose leading axis is the number of chunks."""
    return re.findall(
        r"[a-z]+:[a-z]+\d+\[" + str(leading) + r",([0-9,]+)\]", jaxpr_text
    )


def _inputs():
    random = np.random.RandomState(0)
    return dict(
        gcart=jnp.asarray(random.randn(NPAD, 3)),
        mask=jnp.ones(NPAD),
        phases=jnp.asarray(random.randn(NAT, NPAD) + 0j),
        becsum=jnp.asarray(random.randn(NAT, NH, NH)),
        potential=jnp.asarray(random.randn(NPAD) + 0j),
    )


def test_the_charge_scan_does_not_stack_the_augmentation_table():
    """No ``(nchunks, nh, nh, chunk)`` residual survives into the backward pass.

    Measured before the remat: the jaxpr carried ``(4, 3, 3, 5)`` -- the table --
    and ``(4, 2, 5)``, a copy of the atom phases. Both are gone with it.
    """
    data = _inputs()

    def energy(becsum):
        charge = _tabulated_charge(
            _build, data["gcart"], data["mask"], data["phases"], becsum,
            CHUNK, NPAD,
        )
        return jnp.sum(jnp.abs(charge) ** 2)

    text = str(jax.make_jaxpr(jax.grad(energy))(data["becsum"]))
    stacked = _stacked_leading(text)
    table = f"{NH},{NH},{CHUNK}"
    assert table not in stacked, (
        f"the dense Q_ij(G) table is stacked as a scan residual ({NCHUNKS}, "
        f"{table}); the body's @jax.checkpoint has been lost. This is >= 2 GiB "
        f"of tape by construction on any cell that takes this route "
        f"(MEMORY-AUDIT.md A1)"
    )
    assert f"{NAT},{CHUNK}" not in stacked, "the atom phases are stacked too"


def test_the_integral_scan_does_not_stack_the_augmentation_table():
    """The same, for the route every reverse-mode consumer of ``newd`` reaches.

    Its accumulator was always the scan's *carry* and so never the problem; the
    residual is, and it is the same table.
    """
    data = _inputs()

    def energy(potential):
        integrals = _tabulated_integrals(
            _build, data["gcart"], data["mask"], potential, data["phases"],
            1.0, CHUNK, NH,
        )
        return jnp.sum(integrals ** 2)

    text = str(jax.make_jaxpr(jax.grad(energy))(data["potential"]))
    assert f"{NH},{NH},{CHUNK}" not in _stacked_leading(text), (
        "the dense Q_ij(G) table is stacked as a scan residual in "
        "_tabulated_integrals (MEMORY-AUDIT.md A1)"
    )


def test_the_remat_does_not_move_the_gradient():
    """The guard on the guard: remat must be a memory trade and nothing else.

    A test that only checked the jaxpr would pass if the body were replaced by
    something cheaper *and wrong*, so the value is pinned too. Against a
    hand-written unrematted copy of the same body the gradient is bit-identical;
    through the full force and stress of a displaced ultrasoft cell it agrees to
    one ulp, because remat reorders the backward pass (`PERFORMANCE.md`).
    """
    data = _inputs()

    def unrematted(build, gcart, mask, phases, becsum, chunk, ngm):
        nchunks = mask.shape[0] // chunk
        nat = phases.shape[0]

        def body(carry, index):
            start = index * chunk
            gcart_chunk = jax.lax.dynamic_slice(gcart, (start, 0), (chunk, 3))
            phase_chunk = jax.lax.dynamic_slice(phases, (0, start), (nat, chunk))
            mask_chunk = jax.lax.dynamic_slice(mask, (start,), (chunk,))
            weighted = jnp.einsum("aij,ac->ijc", becsum, phase_chunk)
            block = jnp.einsum("ijc,ijc->c", build(gcart_chunk), weighted)
            return carry, block * mask_chunk

        _, blocks = jax.lax.scan(body, None, jnp.arange(nchunks))
        return blocks.reshape(-1)[:ngm]

    def make(fn):
        return jax.grad(lambda becsum: jnp.sum(jnp.abs(fn(
            _build, data["gcart"], data["mask"], data["phases"], becsum,
            CHUNK, NPAD,
        )) ** 2))

    rematted = np.asarray(make(_tabulated_charge)(data["becsum"]))
    plain = np.asarray(make(unrematted)(data["becsum"]))
    assert np.array_equal(rematted, plain)
