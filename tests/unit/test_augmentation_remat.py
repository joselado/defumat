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

from defumat.pseudo.augmentation import (
    _assemble_qgm, _tabulated_charge, _tabulated_integrals,
)

pytestmark = [pytest.mark.unit]

#: Toy shapes. Nothing here is physical -- what is asserted is the *shape* of the
#: traced program, which does not depend on the numbers. Since the scan
#: contracts in the radial basis it builds no ``(nh, nh, chunk)`` block at all;
#: what it builds per chunk, and what must not be stacked, is the radial table
#: ``(nbeta, nbeta, nl, chunk)`` and the harmonics ``(chunk, nl^2)``.
NH, NAT, NPAD, CHUNK = 3, 2, 30, 5
NBETA, NL = 4, 2
NCHUNKS = NPAD // CHUNK
BETA_OF = jnp.asarray([0, 1, 3])
COEFFICIENTS = jnp.asarray(np.random.RandomState(1).randn(NL * NL, NH, NH))


def _factors(gcart):
    """Stands in for the harmonics and the radial transform: functions of ``gcart`` alone.

    That is the whole of the mechanism -- it makes them *known* values inside
    the body, which is what lets partial evaluation stack them as residuals.
    """
    radius = jnp.linalg.norm(gcart, axis=-1)
    decay = jnp.arange(1, NBETA * NBETA * NL + 1, dtype=radius.dtype)
    radial = jnp.exp(-radius[None, :] * decay[:, None]).reshape(NBETA, NBETA, NL, -1)
    ylm = jnp.cos(gcart[:, :1] * jnp.arange(1, NL * NL + 1, dtype=radius.dtype))
    return ylm, radial


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


def _charge(fn=_tabulated_charge):
    data = _inputs()
    return lambda becsum: fn(
        _factors, data["gcart"], data["mask"], data["phases"], becsum,
        COEFFICIENTS, BETA_OF, NBETA, NL, CHUNK, NPAD)


def test_the_charge_scan_does_not_stack_its_factors():
    """No per-chunk radial table, harmonics or phases survive into the backward pass.

    Before the scan contracted in the radial basis the residual was the block
    of ``Q_ij(G)`` itself, ``(nchunks, nh, nh, chunk)``; now, measured on the
    same body without its remat (:func:`_unrematted_charge`), it is each ``L``'s
    slice of the radial table, ``(6, 4, 4, 5)``, and ``(6, 2, 5)`` for the
    phases, as before. Both are gone with the remat.
    """
    charge = _charge()
    text = str(jax.make_jaxpr(jax.grad(
        lambda b: jnp.sum(jnp.abs(charge(b)) ** 2)))(_inputs()["becsum"]))
    stacked = _stacked_leading(text)
    for shape, what in ((f"{NBETA},{NBETA},{NL},{CHUNK}", "the radial table"),
                        (f"{NBETA},{NBETA},{CHUNK}", "one L of the radial table"),
                        (f"{CHUNK},{NL * NL}", "the harmonics"),
                        (f"{NAT},{CHUNK}", "the atom phases")):
        assert shape not in stacked, (
            f"{what} is stacked as a scan residual ({NCHUNKS}, {shape}); the "
            f"body's @jax.checkpoint has been lost (MEMORY-AUDIT.md A1)")


def test_the_integral_scan_does_not_stack_its_factors():
    """The same, for the route every reverse-mode consumer of ``newd`` reaches.

    Its accumulator was always the scan's *carry* and so never the problem; the
    residual is.
    """
    data = _inputs()

    def energy(potential):
        integrals = _tabulated_integrals(
            _factors, data["gcart"], data["mask"], potential, data["phases"],
            1.0, CHUNK, COEFFICIENTS, BETA_OF, NBETA, NL,
        )
        return jnp.sum(integrals ** 2)

    text = str(jax.make_jaxpr(jax.grad(energy))(data["potential"]))
    stacked = _stacked_leading(text)
    assert f"{NBETA},{NBETA},{NL},{CHUNK}" not in stacked, (
        "the radial table is stacked as a scan residual in _tabulated_integrals "
        "(MEMORY-AUDIT.md A1)")
    assert f"{NBETA},{NBETA},{CHUNK}" not in stacked
    assert f"{CHUNK},{NL * NL}" not in stacked


def _block_form(data, becsum, potential):
    """The charge and the integrals through ``Q_ij(G)`` itself, ``_assemble_qgm``'s block."""
    ylm, radial = _factors(data["gcart"])
    q = _assemble_qgm(COEFFICIENTS, ylm, radial, BETA_OF, NL)  # (nh, nh, npad)
    weighted = jnp.einsum("aij,ac->ijc", becsum + 0j, data["phases"])
    charge = jnp.einsum("ijc,ijc->c", q, weighted) * data["mask"]
    shifted = potential[None, :] * jnp.conj(data["phases"]) * data["mask"]
    integrals = jnp.einsum("ijc,ac->aij", jnp.conj(q), shifted)
    return charge, integrals


def test_the_radial_basis_contraction_is_the_block_contraction():
    """Contracting ``becsum`` with ``ap`` first is the same sum as forming ``Q_ij(G)``.

    Against ``_assemble_qgm``'s block on the same factors, with ``beta_of``
    mapping three channels onto two radial functions and both ``L`` present:
    the charge, and the integrals with and without their real part.
    """
    data = _inputs()
    becsum = data["becsum"]
    charge, integrals = _block_form(data, becsum, data["potential"])
    scale = float(jnp.max(jnp.abs(charge)))
    assert float(jnp.max(jnp.abs(_charge()(becsum) - charge))) < 1e-14 * scale
    for real in (True, False):
        factored = _tabulated_integrals(
            _factors, data["gcart"], data["mask"], data["potential"],
            data["phases"], 1.0, CHUNK, COEFFICIENTS, BETA_OF, NBETA, NL,
            real=real)
        expected = jnp.real(integrals) if real else integrals
        assert float(jnp.max(jnp.abs(factored - expected))) < (
            1e-13 * float(jnp.max(jnp.abs(integrals))))


def test_the_remat_does_not_move_the_gradient():
    """The guard on the guard: remat must be a memory trade and nothing else.

    A test that only checked the jaxpr would pass if the body were replaced by
    something cheaper *and wrong*, so the value is pinned too: against the same
    contraction with no remat the gradient is bit-identical.
    """
    data = _inputs()
    rematted = np.asarray(jax.grad(
        lambda b: jnp.sum(jnp.abs(_charge()(b)) ** 2))(data["becsum"]))
    unrematted_charge = _charge(_unrematted_charge)
    plain = np.asarray(jax.grad(
        lambda b: jnp.sum(jnp.abs(unrematted_charge(b)) ** 2))(data["becsum"]))
    assert np.array_equal(rematted, plain)


def _unrematted_charge(factors, gcart, mask, phases, becsum, coefficients, beta_of,
                       nbeta, nl, chunk, ngm):
    """:func:`_tabulated_charge` with its body written out and no ``jax.checkpoint``."""
    from defumat.pseudo.augmentation import _beta_basis

    nchunks = mask.shape[0] // chunk
    nat = phases.shape[0]
    basis = _beta_basis(coefficients, becsum.astype(phases.dtype), beta_of, nbeta, nl)

    def body(carry, index):
        start = index * chunk
        gcart_chunk = jax.lax.dynamic_slice(gcart, (start, 0), (chunk, 3))
        phase_chunk = jax.lax.dynamic_slice(phases, (0, start), (nat, chunk))
        mask_chunk = jax.lax.dynamic_slice(mask, (start,), (chunk,))
        ylm, radial = factors(gcart_chunk)
        per_atom = jnp.zeros((nat, chunk), dtype=phase_chunk.dtype)
        for l in range(nl):
            block = slice(l * l, (l + 1) ** 2)
            radial_l = jnp.einsum("amnk,nkc->amc", basis[:, block],
                                  radial[:, :, l].astype(basis.dtype))
            per_atom = per_atom + jnp.einsum(
                "amc,cm->ac", radial_l, ylm[:, block].astype(basis.dtype))
        return carry, jnp.einsum("ac,ac->c", per_atom, phase_chunk) * mask_chunk

    _, blocks = jax.lax.scan(body, None, jnp.arange(nchunks))
    return blocks.reshape(-1)[:ngm]
