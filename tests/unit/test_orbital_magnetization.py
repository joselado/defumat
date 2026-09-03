"""P64: the orbital magnetization's machinery, on models where it is exact.

Four statements, and none of them needs a plane wave. The zone sums
:func:`~defumat.topology.orbital_magnetization.orbital_magnetization_sums`
returns carry no lattice and no unit, so a tight-binding model reaches every
line of the assembly:

* the **curvature** sum is the Chern number, and must agree with the
  Fukui-Hatsugai-Suzuki construction -- which shares nothing with it beyond the
  states, being a product of link determinants where this is a covariant finite
  difference. It is what pins the sign and the factor of ``4 pi`` in
  ``dM/dmu``;
* a **time-reversal symmetric** model has no orbital magnetization at all, and
  the two terms vanish separately rather than cancelling;
* the answer is invariant under a **unitary mixing of the occupied manifold**
  -- rule D4, which the dual-state construction satisfies by being covariant
  rather than by any special handling of degeneracies;
* and it is invariant under a **rigid shift of the zero of energy** provided
  the chemical potential shifts with it, which is the one check that pins
  ``dm_dmu``'s own factor of two against the two terms it sits beside.

The Haldane model at ``mass = 0`` is particle-hole symmetric, so its ``M_LC``
and ``M_IC`` are exactly opposite and its total vanishes at ``mu = 0``: an
analytic statement, and the reason the shift test below is where a nonzero
total comes from.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.topology.invariants import ModelSource, chern_number
from defumat.topology.mesh import volume_mesh
from defumat.topology.orbital_magnetization import (
    orbital_magnetization,
    orbital_magnetization_sums,
)
from defumat.topology.states import ModelStates
from tests.models import haldane, random_gauge

pytestmark = pytest.mark.unit


def _states(hamiltonian, divisions, nocc=1):
    mesh = volume_mesh(divisions)
    return mesh, ModelStates.solve(hamiltonian, mesh.flat(), nocc=nocc)


def _shifted(hamiltonian, delta):
    """The same model with its zero of energy moved by ``delta``."""
    def shifted(k):
        matrix = hamiltonian(k)
        return matrix + delta * jnp.eye(matrix.shape[0], dtype=matrix.dtype)

    return shifted


def _doubled_haldane(gap: float = 0.3):
    """Two Haldane copies, the second raised, so the manifold is two bands.

    Both copies are Chern insulators with the same winding, and the raised one
    stays below the unraised one's empty band (``3 sqrt(3) t2 / 2 = 0.52``
    against ``gap = 0.3``), so the two lowest bands are an isolated manifold
    with time reversal broken. What it is for is the *unitary* gauge test:
    with one occupied band a gauge is only a phase.
    """
    block = haldane(t2=0.2, mass=0.0)

    def hamiltonian(k):
        small = block(k)
        out = jnp.zeros((4, 4), dtype=complex)
        out = out.at[:2, :2].set(small)
        return out.at[2:, 2:].set(small + gap * jnp.eye(2, dtype=complex))

    return hamiltonian


# -- the curvature sum is the Chern number ------------------------------------


@pytest.mark.parametrize("n, tolerance", [(5, 5e-2), (9, 1e-2), (15, 3e-3)])
def test_the_curvature_sum_converges_onto_the_chern_number(n, tolerance):
    """``S_curv = -4 pi N_l C_l``, against the exact-integer construction.

    The Fukui-Hatsugai-Suzuki Chern number is ``-1`` on any mesh; this
    discretisation is a Riemann sum of a covariant derivative and converges to
    it, which is the check that the dual states really are ``d u/dk``. The
    tolerance shrinks with the mesh because the error does, and that is the
    content: a wrong factor would be mesh-independent.
    """
    model = haldane(t2=0.2, mass=0.0)
    exact = chern_number(ModelSource(hamiltonian=model, nocc=1), shape=(n, n))
    mesh, states = _states(model, (n, n, 1))
    sums = orbital_magnetization_sums(states, mesh)
    implied = -sums["curvature"][2] / (4.0 * np.pi)
    assert exact.chern_number == pytest.approx(-1.0, abs=1e-12)
    assert implied == pytest.approx(exact.chern_number, abs=tolerance)


def test_a_flat_direction_carries_no_derivative():
    """One division along a direction means no dispersion to differentiate.

    A two-dimensional model is run on an ``(n, n, 1)`` mesh, so the components
    that pair the third direction with another must be *identically* zero
    rather than the difference of two gathers of the same k-point.
    """
    mesh, states = _states(haldane(t2=0.2, mass=0.0), (7, 7, 1))
    sums = orbital_magnetization_sums(states, mesh)
    assert sums["flat_directions"] == (2,)
    for name in ("lc", "ic", "curvature"):
        assert sums[name][0] == 0.0
        assert sums[name][1] == 0.0
    assert abs(sums["curvature"][2]) > 1.0


def test_two_divisions_are_refused_rather_than_aliased():
    """With two divisions a point's two neighbours are the same k-point."""
    mesh = volume_mesh((2, 2, 1))
    states = ModelStates.solve(haldane(), mesh.flat(), nocc=1)
    with pytest.raises(ValueError, match="two divisions"):
        orbital_magnetization_sums(states, mesh)


# -- the nulls -----------------------------------------------------------------


def test_a_time_reversal_symmetric_model_has_none():
    """``t2 = 0`` leaves a real Hamiltonian, and both terms vanish separately.

    Not only their sum: a cancellation between two large numbers would pass a
    total-only test and would mean the split is wrong.
    """
    mesh, states = _states(haldane(t2=0.0, mass=1.0), (7, 7, 1))
    sums = orbital_magnetization_sums(states, mesh)
    for name in ("lc", "ic", "curvature"):
        assert np.abs(sums[name]).max() < 1e-12


def test_the_particle_hole_symmetric_model_has_no_total_at_mu_zero():
    """Haldane at ``mass = 0``: ``E_+ = -E_-``, so ``M_LC = -M_IC`` exactly.

    An analytic statement about a two-band model with one occupied band, where
    the difference of dual states is proportional to the empty state: the two
    terms then carry ``E_empty`` and ``E_occupied``, which are opposite. It
    checks the *relative* normalisation of the two terms, which nothing else
    here does.
    """
    mesh, states = _states(haldane(t2=0.2, mass=0.0), (9, 9, 1))
    sums = orbital_magnetization_sums(states, mesh)
    assert abs(sums["lc"][2]) > 1.0
    assert sums["lc"][2] == pytest.approx(-sums["ic"][2], rel=1e-10)


# -- gauge invariance ----------------------------------------------------------


def test_a_random_phase_per_state_changes_nothing():
    """The freedom an eigensolver has for a nondegenerate band."""
    mesh, states = _states(haldane(t2=0.2, mass=0.0), (7, 7, 1))
    reference = orbital_magnetization_sums(states, mesh)
    gauge = random_gauge(7, (states.nk, states.nbnd))
    rotated = ModelStates(
        coefficients=jnp.asarray(np.asarray(states.coefficients) * gauge),
        hamiltonian=states.hamiltonian,
        points=states.points,
    )
    sums = orbital_magnetization_sums(rotated, mesh)
    for name in ("lc", "ic", "curvature"):
        assert sums[name] == pytest.approx(reference[name], rel=1e-10, abs=1e-12)


def test_a_unitary_mixing_of_the_manifold_changes_nothing():
    """Rule D4: the answer is a property of the subspace, not of its basis.

    Two occupied bands, mixed at every k-point by an independent random unitary
    -- the freedom a degenerate eigensolver has, and what a per-band ``E_n``
    expression would not survive.
    """
    model = _doubled_haldane()
    mesh, states = _states(model, (7, 7, 1), nocc=2)
    reference = orbital_magnetization_sums(states, mesh)
    mixing = random_gauge(11, (states.nk, states.nbnd), unitary=True)
    rotated = ModelStates(
        coefficients=jnp.einsum(
            "kmn,kna->kma", jnp.asarray(mixing), states.coefficients
        ),
        hamiltonian=states.hamiltonian,
        points=states.points,
    )
    sums = orbital_magnetization_sums(rotated, mesh)
    assert abs(reference["lc"][2]) > 1.0
    for name in ("lc", "ic", "curvature"):
        assert sums[name] == pytest.approx(reference[name], rel=1e-9, abs=1e-11)


# -- the chemical potential ----------------------------------------------------


def test_moving_the_zero_of_energy_moves_nothing_physical():
    """``M(H + delta, mu = delta) = M(H, mu = 0)``.

    The one statement that pins ``dm_dmu`` -- its sign, and its factor of two
    against the two terms it is added to. A rigid shift of the Hamiltonian adds
    ``delta`` to ``M_LC`` and to ``M_IC`` alike (both carry one power of an
    energy), and the ``-2 mu`` term has to take it back exactly. On a Chern
    insulator, where ``dM/dmu`` is not zero, this is a real cancellation and
    not an identity between zeros.
    """
    class _Cell:
        """The smallest thing the wrapper needs: a volume and ``bg``."""

        volume = 1.0
        bg = np.eye(3)

    delta = 0.37
    mesh = volume_mesh((9, 9, 1))
    plain = ModelStates.solve(haldane(t2=0.2, mass=0.0), mesh.flat(), nocc=1)
    shifted = ModelStates.solve(
        _shifted(haldane(t2=0.2, mass=0.0), delta), mesh.flat(), nocc=1
    )

    first = orbital_magnetization(plain, mesh, _Cell())
    second = orbital_magnetization(shifted, mesh, _Cell(), mu=delta)

    assert abs(first.dm_dmu[2]) > 1e-3
    assert second.lc[2] != pytest.approx(first.lc[2], rel=1e-6)
    assert second.total[2] == pytest.approx(first.total[2], abs=1e-10)
    assert second.chern[2] == pytest.approx(first.chern[2], rel=1e-12)


def test_the_chern_vector_comes_out_of_the_same_sums():
    """A Chern insulator's ``dM/dmu`` is its Chern number, up to the prefactor.

    ``dM/dmu = -2 (Omega/(4 (2 pi)^3)) sum_l b_l/N_l S^curv_l`` and
    ``S^curv_l = -4 pi N_l C_l``, so on a cubic cell the two are related by
    ``Omega |b|/(2 (2 pi)^3) * 4 pi``. Checking it here is what makes ``chern``
    on the result an assertion rather than a label.
    """
    class _Cell:
        volume = 2.0
        bg = 3.0 * np.eye(3)

    mesh = volume_mesh((9, 9, 1))
    states = ModelStates.solve(haldane(t2=0.2, mass=0.0), mesh.flat(), nocc=1)
    result = orbital_magnetization(states, mesh, _Cell())
    expected = (
        -2.0 * 2.0 / (4.0 * (2.0 * np.pi) ** 3) * 3.0
        * (-4.0 * np.pi * result.chern[2])
    )
    assert result.dm_dmu[2] == pytest.approx(expected, rel=1e-12)
    assert result.chern[2] == pytest.approx(-1.0, abs=2e-2)
