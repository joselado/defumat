"""The per-atom structure factor must not survive into the backward pass.

`MEMORY-AUDIT.md` A5. Both sites compute ``e^{-i G . tau_a}`` for every atom,
and ``exp``'s VJP is ``ans * t`` -- it saves its own output. So the ``(nat, ngm)``
complex array is a residual of **every** derivative that moves an atom or strains
the cell, which is every force, every stress, every phonon column and every Born
charge. On a 45-atom slab with a 3.5-million-vector dense set that is 2.55 GB per
site.

**The two sites are not the same case, and only a measurement separates them.**

* :func:`~defumat.pseudo.potentials._structure_factors_at` contracts the per-atom
  array to ``(ntyp, ngm)`` on the very next line, so the residual is larger than
  the result by ``nat/ntyp`` and is plainly waste.
* :func:`~defumat.pseudo.augmentation._atom_phases` *returns* the per-atom array
  and it is kept as a field on the augmentation charge, which is also an input to
  the already-rematted augmentation scan. The expectation was that rematting it
  would therefore be a no-op. It is not -- the recomputed value has a short live
  range at each use where the stored one spanned the whole pass -- and that is
  worth pinning precisely because the argument for it is not obvious from the
  source.

**Why residuals and not a jaxpr grep.** A ``make_jaxpr`` of the gradient shows
the per-atom array either way: it is in the forward computation regardless, and
the text cannot say whether it *crosses* into the backward pass. ``jax.linearize``
can -- the linear function it returns closes over exactly the residuals, so the
constvars of its jaxpr are the tape and nothing else is.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat.pseudo.augmentation import _atom_phases
from defumat.pseudo.potentials import _structure_factors_at

pytestmark = [pytest.mark.unit]

#: Toy shapes; nothing here is physical. What is asserted is which arrays cross
#: from the forward pass into the backward one, which does not depend on the
#: numbers. ``NGM`` is large enough that the per-atom array is unambiguously the
#: biggest thing in the tape.
NGM, NAT, NTYP = 512, 6, 2


def _tape_bytes(energy, x):
    """Bytes closed over by the linearized function: the residuals, exactly."""
    _, linear = jax.linearize(energy, x)
    jaxpr = jax.make_jaxpr(linear)(jnp.zeros_like(x))
    return sum(
        int(np.prod(var.aval.shape)) * var.aval.dtype.itemsize
        for var in jaxpr.jaxpr.constvars
    )


def _inputs():
    random = np.random.RandomState(0)
    types = np.array([0] * (NAT // 2) + [1] * (NAT - NAT // 2))
    return (
        jnp.asarray(random.randn(NGM, 3)),
        jnp.asarray(random.randn(NAT, 3)),
        jnp.asarray(np.equal(types[None, :], np.arange(NTYP)[:, None]).astype(float)),
        jnp.asarray(random.randn(NAT)),
    )


#: One ``(nat, ngm)`` complex array. The residual set is asserted to be *below*
#: this, which is a statement no arrangement of the smaller arrays can satisfy by
#: accident: the next largest thing in either tape is ``(ngm, 3)`` real, a factor
#: of ``2 nat / 3`` smaller.
PER_ATOM_BYTES = NAT * NGM * 16


def test_the_species_structure_factor_does_not_tape_its_per_atom_array():
    """``vloc`` and ``rho_core``'s path, reached by every force and stress."""
    g, positions, membership, _ = _inputs()

    def energy(tau):
        return jnp.sum(jnp.abs(_structure_factors_at(g, tau, membership)) ** 2)

    tape = _tape_bytes(energy, positions)
    assert tape < PER_ATOM_BYTES, (
        f"the (ngm, nat) structure factor is saved for the backward pass: the "
        f"tape is {tape / 2**20:.2f} MB against {PER_ATOM_BYTES / 2**20:.2f} MB "
        f"for one copy of it. _structure_factors_at's @jax.checkpoint has been "
        f"lost -- that is 2.55 GB on a 45-atom slab (MEMORY-AUDIT.md A5)"
    )


def test_the_atom_phases_do_not_tape_their_own_output():
    """The augmentation charge's half, which is also its `at_positions` cost."""
    gcart, positions, _, becsum = _inputs()

    def energy(tau):
        phases = _atom_phases(gcart, tau)
        # Stands in for the contraction the augmentation scan body does: the
        # phases enter it as an *input*, which is why this one needed measuring
        # rather than arguing.
        return jnp.sum(jnp.abs(jnp.einsum("ag,a->g", phases, becsum)) ** 2)

    tape = _tape_bytes(energy, positions)
    assert tape < PER_ATOM_BYTES, (
        f"the (nat, ngm) atom phases are saved for the backward pass: the tape "
        f"is {tape / 2**20:.2f} MB against {PER_ATOM_BYTES / 2**20:.2f} MB for "
        f"one copy. _atom_phases' @jax.checkpoint has been lost "
        f"(MEMORY-AUDIT.md A5)"
    )


def test_the_remat_does_not_move_either_gradient():
    """The guard on the guard: a memory trade and nothing else.

    Both are pure functions of their inputs re-executed, so here the gradient is
    **bit-identical** against an unrematted copy of the same expression. Through
    a whole force it is round-off instead, because remat reorders the backward
    pass -- 6.9e-17 on a force component of 1.4e-2 (`PERFORMANCE.md`).
    """
    g, positions, membership, becsum = _inputs()

    def plain_factors(g, positions, membership):
        phases = jnp.exp(-1j * (g @ positions.T))
        return membership @ phases.T

    def plain_phases(gcart, positions):
        return jnp.exp(-1j * (positions @ gcart.T))

    def factors_energy(fn):
        return jax.grad(lambda tau: jnp.sum(jnp.abs(fn(g, tau, membership)) ** 2))

    def phases_energy(fn):
        return jax.grad(lambda tau: jnp.sum(jnp.abs(
            jnp.einsum("ag,a->g", fn(g, tau), becsum)
        ) ** 2))

    assert np.array_equal(
        np.asarray(factors_energy(_structure_factors_at)(positions)),
        np.asarray(factors_energy(plain_factors)(positions)),
    )
    assert np.array_equal(
        np.asarray(phases_energy(_atom_phases)(positions)),
        np.asarray(phases_energy(plain_phases)(positions)),
    )
