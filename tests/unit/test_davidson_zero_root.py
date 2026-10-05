"""A Cholesky subspace solve that is finite and wrong is caught and re-solved.

The defect: a non-self-consistent Davidson solve at ``k + q`` on a four-atom
aluminium cell (``tests/data/qe/al4-metal-k5-nosym.in``), from a random start at
``ethr = 1e-13``, returned at one k-point of 125 a lowest root of -1.08 Ry,
1.07 Ry below the true lowest state, with an eigenvector of zero norm, so every
true state moved up one slot and the top one was lost. A metallic phonon at
``q`` built on those states was 3 to 4 cm^-1 out, and nothing said so.

The mechanism, read off the captured solves of that call. A stubborn root (the
top band, at ``ethr = 1e-13``) keeps the solve going for 61 steps, and the
expansion vectors, normalised residuals of nearly converged roots, go linearly
dependent: the subspace overlap's smallest eigenvalue falls to 4.9e-16 against a
largest of order one. It stays **positive**, so the Cholesky factor is finite
and the existing guard, which watches for a non-finite factor, passes. The
reduction ``L^-1 H L^-H`` then turns the round-off in ``H`` along that
near-null direction into an eigenvalue of order one, here below the spectrum,
and the root taken is that direction, coefficients of norm 4e7 whose
plane-wave vector is close to zero. The collapse that followed stored the zero
vector under a unit overlap (it assumes the Ritz vectors are S-orthonormal), so
from then on the phantom was a converged root with a zero residual.

What separates a broken solve from a clean one is the S-orthonormality of the
coefficients it returns: ``max |V^H S V - 1|`` was 3e-15 or less on every clean
solve of that call and 1.4e-2 to 6.8e-2 on the broken ones. The matrices here
are the four solves 50 to 53 of that call, captured on the CPU: 50 is the last
clean one, 51 is the first broken one (its lowest root is still right, a higher
one is not), and 52 and 53 carry the spurious lowest root at -0.60 and -1.08 Ry.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.solvers import davidson
from defumat.solvers.subspace import (
    generalised_eigh, orthonormality_defect, orthonormality_tolerance,
)

pytestmark = pytest.mark.unit

CAPTURED = Path(__file__).resolve().parents[1] / "data" / "davidson" / "al4-kq-zero-root.npz"
NBND = 10
#: The true lowest state at that k-point, in Ry, from the canonical route on
#: the same matrices and from the full solve at the equivalent grid point.
LOWEST = -0.013882


def _captured(i):
    z = np.load(CAPTURED)
    return tuple(jnp.asarray(z[f"{name}{i}"]) for name in ("h", "s", "parked"))


def test_the_last_clean_solve_reads_clean_on_both_routes():
    h, s, parked = _captured(50)
    tolerance = orthonormality_tolerance(s.dtype)
    for robust in (False, True):
        values, vectors = generalised_eigh(h, s, robust=robust, parked=parked)
        assert float(orthonormality_defect(s, vectors[:, :NBND])) < 1e-12
        assert float(orthonormality_defect(s, vectors[:, :NBND])) < tolerance
        assert abs(float(values[0]) - LOWEST) < 1e-6


@pytest.mark.parametrize("i", [51, 52, 53])
def test_a_finite_cholesky_answer_with_a_spurious_root_is_flagged(i):
    """The factor is finite, so the old guard passed; the defect does not."""
    h, s, parked = _captured(i)
    values, vectors = generalised_eigh(h, s, robust=False, parked=parked)
    assert np.all(np.isfinite(np.asarray(values)))
    assert np.all(np.isfinite(np.asarray(vectors)))
    assert float(orthonormality_defect(s, vectors[:, :NBND])) > orthonormality_tolerance(s.dtype)


@pytest.mark.parametrize("i", [51, 52, 53])
def test_the_canonical_route_drops_the_near_null_direction_and_cholesky_does_not(i):
    """Where the spurious root lands is round-off; that Cholesky is wrong is not.

    On the CPU that captured them, solves 52 and 53 put it at -0.60 and -1.08
    Ry, below the spectrum, but its value is round-off divided by an overlap
    eigenvalue of 4.9e-16, so another LAPACK, or cuSOLVER, may put it anywhere.
    What holds on any arithmetic is that the Cholesky answer is wrong somewhere:
    its coefficients are not S-orthonormal, or its roots are not the canonical
    route's.
    """
    h, s, parked = _captured(i)
    values, vectors = generalised_eigh(h, s, robust=True, parked=parked)
    assert float(orthonormality_defect(s, vectors[:, :NBND])) < 1e-12
    assert abs(float(values[0]) - LOWEST) < 1e-6

    cholesky, coefficients = generalised_eigh(h, s, robust=False, parked=parked)
    defect = float(orthonormality_defect(s, coefficients[:, :NBND]))
    shift = float(np.max(np.abs(np.asarray(cholesky[:NBND]) - np.asarray(values[:NBND]))))
    assert defect > orthonormality_tolerance(s.dtype) or shift > 1e-6, (
        "the captured solve no longer shows the defect"
    )


class _Skewed:
    """``generalised_eigh`` whose Cholesky route returns its first root shrunk.

    A stand-in for the broken solve that does not depend on round-off: the
    first Ritz vector comes back at a thousandth of its norm, which is the
    shape of the defect (a root whose plane-wave vector is close to zero)
    without its cause.
    """

    def __init__(self, original):
        self.original = original

    def __call__(self, h, s, robust=None, parked=None):
        values, vectors = self.original(h, s, robust=robust, parked=parked)
        if robust is False:
            vectors = vectors.at[:, 0].multiply(1.0e-3)
        return values, vectors


@pytest.fixture
def hamiltonian(pseudo_dir):
    import dataclasses

    from defumat.io.pwin import read_pw_input
    from defumat.pseudo import read_upf
    from defumat.scf.driver import Calculation
    from defumat.scf.potential import v_of_rho
    from defumat.system import build_system

    benchmark = Path(__file__).resolve().parents[2] / "benchmarks" / "si-1k.in"
    system = build_system(read_pw_input(benchmark))
    system = dataclasses.replace(system, occupations="smearing", degauss=0.02,
                                 smearing="mv")
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file)
                    for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    potential = v_of_rho(calculation.starting_density(), calculation.basis.dense,
                         system.cell)
    return calculation.hamiltonian(potential.v_scf)[0]


def test_a_solve_with_a_root_that_is_not_s_orthonormal_is_re_solved(hamiltonian, monkeypatch):
    """The wiring: the defect stops the loop and the k-point is re-solved.

    With every Cholesky solve returning a first root that is not S-normalised,
    the fast pass must come back non-finite at that k-point, and the canonical
    retry must return what an unpatched solve returns.
    """
    nbnd = 8
    davidson.davidson_eigensolver_all.clear_cache()
    reference = davidson.davidson_eigensolver_all(hamiltonian, nbnd, None, ethr=1e-10)

    monkeypatch.setattr(davidson, "generalised_eigh", _Skewed(davidson.generalised_eigh))
    davidson.davidson_eigensolver_all.clear_cache()
    try:
        with pytest.warns(UserWarning, match="not S-orthonormal"):
            values, vectors = davidson.davidson_eigensolver_all(
                hamiltonian, nbnd, None, ethr=1e-10)
    finally:
        monkeypatch.undo()
        davidson.davidson_eigensolver_all.clear_cache()

    norms = np.linalg.norm(np.asarray(vectors), axis=-1)
    assert np.max(np.abs(norms - 1.0)) < 1e-10
    assert np.max(np.abs(np.asarray(values) - np.asarray(reference[0]))) < 1e-8
