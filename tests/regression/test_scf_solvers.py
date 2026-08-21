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

import jax.numpy as jnp
import numpy as np
import pytest

from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.scf import Calculation, run_scf
from pypresso.scf.driver import default_nbnd
from pypresso.scf.residual import make_residual
from pypresso.hubbard import uniform_ns
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

#: How far to kick the symmetric iron solution, in the atomic magnetization's
#: shape. The middle of a *measured* window, not a tuned number: below 0.08
#: mixing comes back too, above 0.12 Newton stops coming back -- see
#: :func:`test_newton_krylov_reaches_an_unstable_solution`.
KICK = 0.10
IRON_NONMAGNETIC = BENCHMARKS / "fe-unstable-nonmagnetic.in"

#: fcc nickel with U = 3 eV: the DFT+U counterpart of ``IRON``, and the case
#: that needs a custom starting occupation matrix to set up at all.
NICKEL_U = BENCHMARKS / "ni-u-unstable.in"
NICKEL_U_NONMAGNETIC = BENCHMARKS / "ni-u-nonmagnetic.in"

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

    **The state is perturbed away from the symmetric root and handed to both
    solvers**, rather than both being started from the atomic superposition.
    That is the protocol
    :func:`test_a_hubbard_saddle_is_unstable_to_mixing_and_stable_to_newton`
    uses, and it is the one ``PLAN.md`` P22 says is robust: from far away,
    *which* root Newton-Krylov lands on depends on the inner-solve accuracy in
    no systematic way, and no setting makes it otherwise.

    **Why it had to be rewritten, and what the rewrite measured.** An earlier
    version started both solvers from the atomic superposition, and it was
    passing on a knife edge: changing how ``|psi|^2`` is evaluated --
    ``Re(conj(psi) psi)`` rather than ``abs(psi)**2``, the same number to **3.5
    eps** (:func:`pypresso.scf.density.band_density`) -- was enough to send
    Newton to the ferromagnet instead. Rebuilding it around a perturbed root
    then turned up something about the *physics* that P22 had got slightly
    wrong. The symmetric solution of this cell is **not a saddle in the linear
    sense**: it is metastable, with a finite basin, and the kick has to clear it.
    Measured here, with the kick in the atomic magnetization's own shape (a
    uniform ``1 +- eps`` scaling of the two channels is the wrong direction and
    decays by a factor of 300):

    ======  ====================  ====================
    kick    mixing ends at        Newton ends at
    ======  ====================  ====================
    0.05    ``7e-6`` (comes back)  --
    0.08    ``+3.4052``            ``-0.0003``
    0.10    ``+3.4052``            ``+0.0007``
    0.12    ``+3.4053``            ``+0.0009``
    0.15    ``+3.4053``            ``+3.4052``
    0.20    ``+3.4052``            ``-3.4052``
    0.50    ``+3.4051``            ``+3.4052``
    ======  ====================  ====================

    So the demonstration lives in a **window**, and both of its edges are
    physical rather than numerical: below ~0.08 the kick is inside the
    symmetric root's basin and mixing returns to it too, and above ~0.12 the
    perturbed state is nearer the ferromagnet than the symmetric root and Newton
    converges on *that* -- which is what "converges on whichever root it started
    nearest" means, said quantitatively. Three consecutive points in between
    behave identically, which is what makes 0.10 a reproducible choice rather
    than another knife edge.

    "A root no mixer can hold" is still exactly true of the Hubbard saddle
    below, whose 2% kick runs away. For iron the honest statement is a root no
    mixer *reaches*, which Newton returns to from outside its basin.
    """
    system, pseudos, calculation, _ = _setup(IRON)

    # The saddle itself: a spin-symmetric density, which the SCF map preserves
    # exactly -- nothing in it breaks spin symmetry on its own.
    rho = np.asarray(calculation.starting_density())
    symmetric = jnp.asarray(
        np.repeat(rho.mean(axis=0, keepdims=True), rho.shape[0], axis=0)
    )
    saddle = run_scf(system, pseudos, calculation=Calculation(system, pseudos),
                     conv_thr=1e-8, max_iterations=200,
                     starting_density=symmetric)
    assert saddle.converged
    assert abs(float(saddle.magnetization)) < 1e-6

    # ...and it is the non-magnetic solution, by an independent nspin = 1 run.
    reference_system, reference_pseudos, _, _ = _setup(IRON_NONMAGNETIC)
    reference = run_scf(
        reference_system, reference_pseudos, conv_thr=1e-8, max_iterations=200,
        calculation=Calculation(reference_system, reference_pseudos),
    )
    assert reference.converged
    assert saddle.total_energy == pytest.approx(reference.total_energy, abs=1e-6)

    # Kick it along the magnetization direction and hand the *same* perturbed
    # state to both solvers. Both the **shape** and the **size** of the kick
    # matter, and measuring them is what this test had to do to become
    # reproducible -- see the docstring.
    atomic = np.asarray(calculation.starting_density())
    shape = atomic[0] - atomic[1]
    shape = shape / np.abs(shape).max()
    density = np.asarray(saddle.density)
    kick = KICK * shape
    kicked = jnp.asarray(np.stack([density[0] + 0.5 * kick, density[1] - 0.5 * kick]))
    ran_away = run_scf(system, pseudos, calculation=Calculation(system, pseudos),
                       conv_thr=1e-8, max_iterations=200,
                       starting_density=kicked)
    came_back = run_scf(
        system, pseudos, calculation=Calculation(system, pseudos), conv_thr=1e-8,
        max_iterations=200, scf_solver="newton-krylov",
        # ``kerker`` is load-bearing and not a tuning choice: see
        # ``test_an_inexact_newton_is_only_as_stability_blind_as_its_inner_solve``.
        scf_solver_options={"forcing": 0.5, "gmres_maxiter": 8,
                            "max_iterations": 12, "kerker": True},
        starting_density=kicked,
    )

    assert ran_away.converged and came_back.converged
    # Mixing amplified the kick into the ferromagnetic ground state -- which is
    # *what makes the symmetric solution a saddle*...
    assert abs(float(ran_away.magnetization)) > 3.0
    # ...and Newton put it back, which no mixer can be made to do.
    assert abs(float(came_back.magnetization)) < 1e-2
    assert came_back.total_energy == pytest.approx(reference.total_energy, abs=1e-6)
    # The gap between the two roots is iron's magnetic stabilisation energy.
    assert ran_away.total_energy < came_back.total_energy - 0.02


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


# --- DFT+U ------------------------------------------------------------------
#
# ``ns`` joins the packed state exactly as ``mix_rho.f90`` puts it in
# ``mix_type``: it is not a function of the density, because the Hubbard
# potential is built from it before the Hamiltonian exists. Two things are
# checked -- that the solver still lands on the mixer's fixed point with a U in
# play, and that it reaches a solution the mixer cannot.


@pytest.mark.regression
@pytest.mark.slow
def test_newton_krylov_matches_the_mixer_with_a_hubbard_u():
    """The packing check: ``ns`` in the state must not move the answer."""
    system, pseudos, _, _ = _setup(NICKEL_U)
    mixed = run_scf(system, pseudos, calculation=Calculation(system, pseudos),
                    conv_thr=1e-8, max_iterations=150)
    newton = run_scf(
        system, pseudos, calculation=Calculation(system, pseudos), conv_thr=1e-8,
        max_iterations=150, scf_solver="newton-krylov",
        scf_solver_options={"forcing": 0.5, "gmres_maxiter": 8, "max_iterations": 12},
    )
    assert mixed.converged and newton.converged
    assert newton.total_energy == pytest.approx(mixed.total_energy, abs=TOTAL_ENERGY_RY)
    # ``ns`` is compared in ``ns_ddot``, the metric the SCF itself converges on,
    # rather than elementwise. An elementwise tolerance here would be a number
    # invented for the test: ``conv_thr`` bounds ``U/2 sum |dns|^2``, so at
    # 1e-8 with U = 3 eV the individual entries are only pinned to about 3e-4,
    # and the two solvers do differ at 1e-4. Judging them by the convergence
    # criterion is both the right comparison and one that scales with it.
    _, _, calculation, _ = _setup(NICKEL_U)
    assert float(calculation.ns_accuracy(newton.ns - mixed.ns)) < 10.0 * 1e-8


@pytest.mark.regression
@pytest.mark.slow
def test_a_hubbard_saddle_is_unstable_to_mixing_and_stable_to_newton():
    """The definition of an unstable fixed point, tested as such.

    Perturb the non-magnetic solution along the magnetization direction and the
    two solvers do opposite things: damped mixing runs away to the ferromagnet,
    which is *what makes it a saddle*, and Newton comes back to it. This is
    pyqula's own check on the solutions it found (``scftk/vjinteraction_jax.py``)
    and it is stronger than merely landing on an unusual answer, because the
    same perturbed input is handed to both.

    **It needs the custom starting occupation matrix**, and that is the point of
    ``uniform_ns``: ``init_ns`` reads Hund's rule off ``starting_magnetization``,
    so the default start is already deep in the ferromagnetic basin whatever
    that number is, and the perturbation has to be applied to ``ns`` as well as
    to the density -- a kick in the density alone is undone by the Hubbard
    potential the polarised ``ns`` is still generating.
    """
    system, pseudos, calculation, _ = _setup(NICKEL_U)

    # The saddle: spin-symmetric density and an orbitally uniform ns, which the
    # SCF map preserves exactly.
    rho = np.asarray(calculation.starting_density())
    symmetric = jnp.asarray(np.repeat(rho.mean(axis=0, keepdims=True), rho.shape[0], axis=0))
    start = dict(starting_density=symmetric,
                 starting_becsum=calculation.starting_becsum(),
                 starting_ns=uniform_ns(calculation.hubbard, calculation.nspin))
    saddle = run_scf(system, pseudos, calculation=Calculation(system, pseudos),
                     conv_thr=1e-9, max_iterations=150, **start)
    assert saddle.converged
    assert abs(float(saddle.magnetization)) < 1e-6

    # ...and it is the non-magnetic solution, by an independent nspin = 1 run.
    reference_system, reference_pseudos, _, _ = _setup(NICKEL_U_NONMAGNETIC)
    reference = run_scf(reference_system, reference_pseudos, conv_thr=1e-9,
                        max_iterations=150,
                        calculation=Calculation(reference_system, reference_pseudos))
    assert saddle.total_energy == pytest.approx(reference.total_energy, abs=1e-6)

    # Kick it 2% along the magnetization direction, in both parts of the state.
    epsilon = 0.02
    density = np.asarray(saddle.density)
    density = density * np.array([1.0 + epsilon, 1.0 - epsilon])[:, None, None, None]
    ns = np.asarray(saddle.ns) * np.array([1.0 + epsilon, 1.0 - epsilon])[:, None, None, None]
    kicked = dict(starting_density=jnp.asarray(density),
                  starting_becsum=calculation.starting_becsum(),
                  starting_ns=jnp.asarray(ns))

    ran_away = run_scf(system, pseudos, calculation=Calculation(system, pseudos),
                       conv_thr=1e-9, max_iterations=150, **kicked)
    came_back = run_scf(
        system, pseudos, calculation=Calculation(system, pseudos), conv_thr=1e-9,
        max_iterations=150, scf_solver="newton-krylov",
        scf_solver_options={"forcing": 0.1, "gmres_maxiter": 30, "max_iterations": 15},
        **kicked,
    )
    assert ran_away.converged and came_back.converged
    # Mixing amplified the kick into the ferromagnetic solution...
    assert abs(float(ran_away.magnetization)) > 1.0
    # ...and Newton put it back.
    assert abs(float(came_back.magnetization)) < 1e-4
    assert came_back.total_energy == pytest.approx(saddle.total_energy, abs=1e-7)


@pytest.mark.regression
def test_starting_ns_is_checked_rather_than_broadcast():
    """A wrong-shaped occupation matrix is a mistake worth a message: the shape
    has one slot per *correlated* atom and zero-padded manifolds, so it is not
    something to infer from the structure."""
    system, pseudos, calculation, _ = _setup(NICKEL_U)
    with pytest.raises(ValueError, match="nslot"):
        run_scf(system, pseudos, calculation=Calculation(system, pseudos),
                max_iterations=1, starting_ns=np.zeros((2, 1, 3, 3)))


@pytest.mark.regression
def test_starting_ns_without_a_u_is_refused():
    system, pseudos, calculation, _ = _setup(SILICON)
    with pytest.raises(ValueError, match="no Hubbard U"):
        run_scf(system, pseudos, calculation=Calculation(system, pseudos),
                max_iterations=1, starting_ns=np.zeros((1, 1, 5, 5)))
