"""A Hamiltonian holds the one plane-wave count the eigensolver reads, not the per-k list.

``OPEN.md`` Part XXIII item 9, half (a). The Davidson subspace is capped at
``npol * min_k npw`` and that minimum is the only thing any consumer reads off
the per-k counts, but the operators held the whole tuple as a static field. A
static field is part of the pytree's structure, so two spheres with the same
padded width and the same smallest sphere -- two wavevectors of a spin-spiral
scan, typically -- gave two treedefs, and every compiled unit that takes the
Hamiltonian (the Davidson solve, the Rayleigh-Ritz start) compiled again at
every wavevector for a value it never read. The field now holds the minimum.
"""

import jax
import jax.numpy as jnp
import pytest

from defumat.hamiltonian.noncollinear import SpinorHamiltonian
from defumat.hamiltonian.operator import Hamiltonian
from defumat.pseudo.projectors import Projectors

pytestmark = pytest.mark.unit

NPWX = 6


def _projectors(rows):
    return Projectors(stored=jnp.zeros((rows, NPWX, 0), dtype=complex),
                      dij=jnp.zeros((0, 0)), atom_of_channel=())


def _collinear(npw):
    return Hamiltonian(
        kinetic=jnp.zeros((2, NPWX)), potential=jnp.zeros((2, 2, 2)),
        fft_index=jnp.zeros((2, NPWX), dtype=int), mask=jnp.ones((2, NPWX), dtype=bool),
        projectors=_projectors(2), grid=(2, 2, 2), npw=npw,
    )


def _spiral(npw):
    # A spiral's arrays carry both components' rows, ``2 nk`` of them.
    return SpinorHamiltonian(
        kinetic=jnp.zeros((4, NPWX)), potential=jnp.zeros((1, 2, 2, 2)),
        fft_index=jnp.zeros((4, NPWX), dtype=int), mask=jnp.ones((4, NPWX), dtype=bool),
        projectors=_projectors(4), deeq=jnp.zeros((2, 2, 0, 0), dtype=complex),
        grid=(2, 2, 2), spiral=True, npw=npw,
    )


def test_spheres_with_the_same_smallest_count_share_one_treedef():
    a, b = _collinear((5, 6)), _collinear((6, 5))
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    assert a.npw == b.npw == 5 and a.space == 5
    # and the field still decides the cap, so a different minimum is a
    # different program, as it has to be
    assert (jax.tree_util.tree_structure(_collinear((6, 6)))
            != jax.tree_util.tree_structure(a))

    s, t = _spiral((4, 6, 5, 6)), _spiral((6, 4, 6, 6))
    assert jax.tree_util.tree_structure(s) == jax.tree_util.tree_structure(t)
    assert s.space == 2 * 4


def test_no_counts_leaves_the_cap_at_the_padded_width():
    assert _collinear(None).npw is None and _collinear(None).space == NPWX
    # an empty list is what a chunk of a force pass carries; nothing reads it
    assert _collinear(()).npw is None
    assert _spiral(None).space == 2 * NPWX
