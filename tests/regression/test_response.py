"""P24: linear response by autodiff -- the velocity operator and what stands on it.

The stages of the phase, each checked against something that shares no
machinery with it:

* **the velocity operator** (:mod:`pypresso.response.velocity`) against a
  central difference of the band structure. ``dH/dk`` comes from one ``jvp`` of
  ``H(k)`` at a frozen sphere and the reference comes from diagonalising at
  ``k +- h``, so the only thing the two have in common is the Hamiltonian
  itself.

The finite-difference reference has one failure mode worth knowing about, and it
is the reference's rather than the operator's: eigenvalues come back **sorted**,
so a step that straddles a band crossing compares two different bands and gives
a difference of order the band width. It is why the step below is small and why
the k-point is a generic one -- at a symmetry point every band is degenerate and
a diagonal expectation value is basis-dependent anyway (rule D4).
"""

from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.response import VelocityOperator
from pypresso.scf import run_scf
from pypresso.system import build_system
from pypresso.system.kpoints import KPoints
from pypresso.workflows.nscf import fixed_density_states

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"

#: A generic k-point in units of ``2 pi/alat``: no symmetry operation fixes it,
#: so no band is degenerate and every diagonal velocity is well defined.
GENERIC_K = np.array([[0.13, 0.27, 0.41]])

#: Central-difference step in 1/bohr. At ``1e-4`` the truncation error is a few
#: parts in ``1e-7`` of a velocity of order 1 Ry bohr, which is what the
#: comparison below is measuring.
STEP = 1.0e-4

#: How far the operator and the difference may disagree. This is the finite
#: difference's own error, not the operator's -- see the step size above.
VELOCITY_RY_BOHR = 5e-6


@lru_cache(maxsize=None)
def _converged(case: str):
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudo_dir = Path(__file__).resolve().parents[1] / "data" / "pseudo"
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    result = run_scf(system, pseudos, conv_thr=1e-10, max_iterations=80)
    return system, pseudos, result


def _states_at(case: str, coords, nbnd: int = 8):
    """Diagonalise at ``coords`` on the converged density."""
    system, pseudos, result = _converged(case)
    kpoints = KPoints(coords=jnp.asarray(coords), weights=jnp.ones(len(coords)))
    calculation, _, eigenvalues, psi = fixed_density_states(
        system, pseudos, result.density, kpoints=kpoints, nbnd=nbnd,
        conv_thr=1e-12,
    )
    return calculation, np.asarray(eigenvalues), psi


@pytest.mark.parametrize("case", ["si2-nc-force", "si2-us"])
def test_band_velocity_matches_a_finite_difference(case):
    """``<psi|dH/dk - eps dS/dk|psi>`` against ``(eps(k+h) - eps(k-h))/2h``.

    Both cases run, and the ultrasoft one is the point of the pair: ``S(k)``
    is built from the same ``vkb(k)`` the nonlocal potential is, so it carries a
    velocity of its own, and an operator that dropped it would pass on
    ``si2-nc-force`` and fail here.
    """
    system, _, result = _converged(case)
    tpiba = float(system.cell.tpiba)

    calculation, eigenvalues, psi = _states_at(case, GENERIC_K)
    operator = VelocityOperator(
        calculation, calculation.potential(result.density).v_scf
    )
    velocities = operator.band_velocities(psi, eigenvalues).velocities

    reference = np.zeros_like(velocities)
    for axis in range(3):
        step = np.zeros(3)
        # ``coords`` is in units of 2 pi/alat and the step is in 1/bohr.
        step[axis] = STEP / tpiba
        _, plus, _ = _states_at(case, GENERIC_K + step)
        _, minus, _ = _states_at(case, GENERIC_K - step)
        reference[..., axis] = (plus - minus) / (2.0 * STEP)

    assert np.abs(velocities - reference).max() < VELOCITY_RY_BOHR


def test_the_overlap_carries_a_velocity_only_when_it_is_not_the_identity():
    """``dS/dk`` is exactly zero for a norm-conserving dataset and is not for USPP.

    The pair is what makes the ultrasoft comparison above mean something: if
    ``dS/dk`` were being dropped, this test says by how much the band velocity
    would then be wrong.
    """
    directions = jnp.eye(3)

    def largest_ds(case):
        _, _, result = _converged(case)
        calculation, _, psi = _states_at(case, GENERIC_K)
        operator = VelocityOperator(
            calculation, calculation.potential(result.density).v_scf
        )
        return max(
            float(jnp.abs(operator.apply_s(psi, directions[axis])).max())
            for axis in range(3)
        )

    assert largest_ds("si2-nc-force") == 0.0
    assert largest_ds("si2-us") > 1e-3
