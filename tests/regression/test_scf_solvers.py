"""The SCF as a root-find: the residual, its Jacobian action, and the solver.

Three things are checked, in the order they can go wrong:

1. **The Jacobian action.** ``ScfResidual.jvp`` differentiates one SCF step with
   ``jax.jvp``; ``jvp_finite_difference`` central-differences the same residual.
   They share no machinery, so agreement means both are right -- the same
   argument the autodiff forces and QE's transcribed ones stand on (P15).
   Where they *disagree* is itself recorded here, because it is the finding that
   decides the default backend (see ``test_autodiff_jvp_needs_a_warm_start``).
2. **The fixed point.** A different solver reaching a different answer is a bug
   in one of them, so Newton-Krylov's converged energy is compared against the
   mixing loop's on the same input, to the tolerance the QE comparison uses.
3. **The preconditioner.** Kerker changes how the mixer gets there and must not
   change where it arrives.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.scf import Calculation, run_scf
from pypresso.scf.driver import default_nbnd
from pypresso.scf.residual import make_residual
from pypresso.system import build_system
from tests.tolerances import TOTAL_ENERGY_RY

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"

#: Silicon, two atoms, one k-point -- the smallest thing that exercises the
#: whole path, and gapped, so the occupations are constant and the Jacobian is
#: the density response alone.
SILICON = BENCHMARKS / "si-1k.in"

#: bcc iron from a *small* starting moment: two self-consistent solutions, one
#: of them unstable. See ``test_newton_krylov_reaches_an_unstable_solution``.
IRON = BENCHMARKS / "fe-unstable.in"
IRON_NONMAGNETIC = BENCHMARKS / "fe-unstable-nonmagnetic.in"

#: An aluminium slab with vacuum: metallic, so the Fermi level moves and its
#: ``custom_jvp`` is on the path, and *inhomogeneously screened*, which is the
#: regime this solver was built for. ``PLAN.md`` P22 has the measurement.
SLAB = BENCHMARKS / "al-slab.in"


@lru_cache(maxsize=None)
def _setup(path, ethr=1.0e-11):
    system = build_system(read_pw_input(path))
    pseudos = tuple(
        read_upf(Path(__file__).parents[1] / "data" / "pseudo" / s.pseudo_file)
        for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    nbnd = system.nbnd or default_nbnd(
        calculation.nelec, system.occupations,
        *((calculation.nelup, calculation.neldw) if system.nspin == 2 else (None, None)),
        noncolin=system.noncolin,
    )
    return system, pseudos, calculation, make_residual(calculation, nbnd, ethr)


@pytest.mark.regression
@pytest.mark.parametrize("case", [SILICON, SLAB], ids=["silicon", "al-slab"])
def test_jacobian_action_matches_finite_differences(case):
    """``J v`` two ways, from a self-consistent point.

    The comparison is made at a *converged* density, which is where a Newton
    solver spends the steps that decide whether it converges at all, and where
    Davidson's warm start is the one the solver actually uses.
    """
    system, pseudos, calculation, residual = _setup(case)
    converged = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-10)
    x = residual.pack(converged.density, calculation.starting_becsum())
    _, psi = residual.residual(x, converged.wavefunctions)

    rng = np.random.default_rng(0)
    direction = rng.normal(size=x.shape)
    direction /= np.linalg.norm(direction)

    autodiff, _ = residual.jvp(x, direction, psi)
    difference = residual.jvp_finite_difference(x, direction, psi)
    relative = np.linalg.norm(autodiff - difference) / np.linalg.norm(difference)
    # Loose on purpose, and the looseness is the point: the autodiff Jacobian is
    # the Jacobian of Davidson *as implemented*, and from a converged start
    # Davidson exits in one or two steps, so its tangent is a one-step
    # approximation to the eigenvector response rather than the response itself.
    # A percent is enough for a Krylov solver, which needs a direction; it is
    # not the exact response, which is what the Sternheimer rule of P22c is for.
    assert relative < 0.05


@pytest.mark.regression
def test_autodiff_jvp_needs_a_warm_start():
    """From a *cold* start the two backends disagree completely, and that is a
    statement about the method rather than a tolerance to be tuned.

    With no starting wavefunctions Davidson runs from the pseudo-atomic
    orbitals, and differentiating that trajectory gives the derivative of a
    different map -- one that happens to land in the same place. The finite
    difference is immune because it relies on Davidson *converging*, not on how
    it got there. This is why ``finite-difference`` is the default backend, and
    it is asserted rather than commented so that a future exact-response rule
    (P22c) has to come and delete it.
    """
    system, pseudos, calculation, residual = _setup(SLAB)
    x = residual.pack(calculation.starting_density(), calculation.starting_becsum())
    rng = np.random.default_rng(0)
    direction = rng.normal(size=x.shape)
    direction /= np.linalg.norm(direction)

    autodiff, _ = residual.jvp(x, direction, None)
    difference = residual.jvp_finite_difference(x, direction, None)
    relative = np.linalg.norm(autodiff - difference) / np.linalg.norm(difference)
    assert relative > 0.5


@pytest.mark.regression
@pytest.mark.slow
@pytest.mark.parametrize("case", [SILICON, SLAB], ids=["silicon", "al-slab"])
def test_newton_krylov_finds_the_mixer_s_fixed_point(case):
    system, pseudos, _, _ = _setup(case)
    mixed = run_scf(system, pseudos, calculation=Calculation(system, pseudos), conv_thr=1e-9)
    newton = run_scf(
        system, pseudos, calculation=Calculation(system, pseudos), conv_thr=1e-9,
        scf_solver="newton-krylov",
        scf_solver_options={"forcing": 0.5, "warmup": 6},
    )
    assert newton.converged and mixed.converged
    assert newton.total_energy == pytest.approx(mixed.total_energy, abs=TOTAL_ENERGY_RY)
    assert newton.solver is not None and newton.solver.steps > 0


@pytest.mark.regression
@pytest.mark.parametrize("case", [SILICON, SLAB], ids=["silicon", "al-slab"])
def test_kerker_changes_the_path_and_not_the_answer(case):
    system, pseudos, _, _ = _setup(case)
    plain = run_scf(system, pseudos, calculation=Calculation(system, pseudos), conv_thr=1e-9)
    kerker = run_scf(system, pseudos, calculation=Calculation(system, pseudos),
                     conv_thr=1e-9, mixing_mode="kerker")
    assert kerker.converged
    assert kerker.total_energy == pytest.approx(plain.total_energy, abs=TOTAL_ENERGY_RY)


@pytest.mark.regression
@pytest.mark.parametrize("case", [SILICON, SLAB], ids=["silicon", "al-slab"])
def test_weights_agree_with_the_driver(case):
    """``residual._weights`` is a second copy of the driver's occupation
    dispatch, kept because ``Calculation.occupations`` syncs the Fermi level to
    the host and cannot be traced. A second copy is a second thing to get wrong,
    so it is pinned against the original rather than trusted."""
    from pypresso.scf.residual import _weights

    system, pseudos, calculation, _ = _setup(case)
    converged = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-9)
    eigenvalues = np.asarray(converged.eigenvalues_by_spin)
    expected, _ = calculation.occupations(eigenvalues)
    assert np.allclose(_weights(calculation, eigenvalues), np.asarray(expected), atol=1e-14)


@pytest.mark.unit
def test_unsupported_occupations_are_refused_by_name():
    """A residual solver that silently substituted a scheme it cannot
    differentiate would converge to the wrong physics, so it raises instead."""
    from pypresso.scf.residual import _weights

    class _Calculation:
        system = type("S", (), {"occupations": "tetrahedra"})()

    with pytest.raises(NotImplementedError, match="tetrahedron"):
        _weights(_Calculation(), np.zeros((1, 1, 4)))


@pytest.mark.regression
@pytest.mark.slow
def test_newton_krylov_reaches_an_unstable_solution():
    """The one thing the residual solver does that no mixer can.

    Iron has two self-consistent solutions here: the ferromagnetic ground state
    and a non-magnetic one, which is a *saddle* of the energy in the
    magnetization direction. Damped mixing is a discrete relaxation dynamics, so
    from a small starting moment it runs downhill into the stable one. Newton is
    stability-blind -- it converges on whichever root it started nearest -- and
    from the same guess it collapses the moment instead.

    **The validation is free and independent**: that ``nspin = 2`` non-magnetic
    energy has to equal a plain ``nspin = 1`` run's on the same cell, and no part
    of the two calculations is shared. The difference between the two roots is
    iron's magnetic stabilisation energy, which is the number such a reference
    state exists to give.
    """
    system, pseudos, _, _ = _setup(IRON)
    mixed = run_scf(system, pseudos, calculation=Calculation(system, pseudos),
                    conv_thr=1e-8, max_iterations=200)
    newton = run_scf(
        system, pseudos, calculation=Calculation(system, pseudos), conv_thr=1e-8,
        max_iterations=200, scf_solver="newton-krylov",
        # ``kerker`` is load-bearing and not a tuning choice: see
        # ``test_an_inexact_newton_is_only_as_stability_blind_as_its_inner_solve``.
        scf_solver_options={"forcing": 0.5, "gmres_maxiter": 8,
                            "max_iterations": 12, "kerker": True},
    )
    reference_system, reference_pseudos, _, _ = _setup(IRON_NONMAGNETIC)
    reference = run_scf(
        reference_system, reference_pseudos, conv_thr=1e-8, max_iterations=200,
        calculation=Calculation(reference_system, reference_pseudos),
    )

    assert mixed.converged and newton.converged and reference.converged
    # Mixing found the ground state...
    assert abs(float(mixed.magnetization)) > 3.0
    # ...and Newton found the saddle, which is a different solution of the same
    # equations, not a failure to converge: its own residual is below conv_thr.
    assert abs(float(newton.magnetization)) < 1e-2
    assert newton.total_energy == pytest.approx(reference.total_energy, abs=1e-6)
    assert mixed.total_energy < newton.total_energy - 0.02  # the stabilisation energy


@pytest.mark.regression
@pytest.mark.slow
def test_an_inexact_newton_is_only_as_stability_blind_as_its_inner_solve():
    """Turning the preconditioner off changes *which root* is found.

    This is the trap, and it is silent: both runs converge, both report an
    accuracy below ``conv_thr``, and they land on different physics. A Newton
    method is stability-blind only to the extent that its inner solve actually
    delivers the Newton direction; with a badly conditioned Krylov system the
    inexact step degrades towards a damped-mixing step, and a damped-mixing step
    flows to the *stable* fixed point. So the preconditioner here is not a
    tuning knob for speed -- it decides the answer.
    """
    system, pseudos, _, _ = _setup(IRON)
    options = {"forcing": 0.5, "gmres_maxiter": 8, "max_iterations": 12}
    unpreconditioned = run_scf(
        system, pseudos, calculation=Calculation(system, pseudos), conv_thr=1e-8,
        max_iterations=200, scf_solver="newton-krylov",
        scf_solver_options={**options, "kerker": False},
    )
    assert unpreconditioned.converged
    assert abs(float(unpreconditioned.magnetization)) > 3.0
