"""P21: ``dE/dq`` for a spin spiral, and the relaxation that follows it.

``pw.x`` has no spin spiral, so there is no gradient to compare against either
and the validation is P19's again: identities, and finite differences of a
quantity that is already validated. Four of them, and each one isolates a
different thing:

1. **The functional is the total energy.**
   :func:`~pypresso.forces.spiral.spiral_energy`, evaluated at the converged
   state and the wavevector it converged at, must reproduce the SCF's own total
   energy to round-off. It is written out in full for exactly this reason: it is
   the only check on the terms the gradient does *not* see.
2. **The gradient is the derivative of that functional.** A central difference
   of :func:`~pypresso.forces.spiral.spiral_energy` at frozen state against
   ``jax.grad`` of it. This tests the automatic differentiation and nothing
   else, and it holds to the finite difference's own truncation error.
3. **The gradient is the derivative of the converged energy.** The one that
   matters, and the one that tests *stationarity*: a central difference of a
   re-converged SCF, at a **frozen plane-wave sphere**, against the gradient at
   the midpoint. Freezing the sphere is not a convenience -- see below.
4. **Symmetry pins two wavevectors exactly.** ``E(q)`` is even in ``q``, so
   ``q = 0`` and the zone boundary ``q = b3/2`` are stationary points whatever
   the physics is. The gradient there must be zero to round-off, and it is the
   sharpest test in the file: nothing about it is a tolerance judgement.
5. **Moving the atom changes nothing.** Every other test in the file has its
   atom at the origin, where the structure factor ``e^{-i(k +- q/2 + G).tau}``
   is identically one for *every* ``q`` -- so none of them exercises that half
   of ``vkb``'s ``q`` dependence at all. Translating the crystal is a symmetry
   of a spiral (a lattice translation combined with the matching spin rotation),
   so ``E(q)`` and ``dE/dq`` must both be unchanged by it, and the same three
   checks are repeated with the atom somewhere general.

**Why test 3 freezes the sphere.** ``E(q)`` computed the way a scan computes it
-- rebuilding the spheres at every point -- is not smooth: it jumps wherever a
plane wave crosses ``|k +- q/2 + G|^2 = ecutwfc``, which on this cell at
``ecutwfc = 25`` moves up to 16 of 1540 plane waves over a step of 0.02. That is
the Pulay error of a finite basis and it is *measured* here rather than assumed:
against a sphere-rebuilding difference the gradient disagrees by 8.3e-4 at
``ecutwfc = 25`` and by 8.3e-6 at 60. Not smoothly -- 40 Ry gives 5.8e-4, which
is not between them, because the number counts the plane waves that happen to
cross inside one window of ``q`` rather than truncating a series. So the
frozen-sphere comparison is the test of the physics, and the other one is the
measurement of what the discretisation costs.
"""

from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from pypresso.forces.energy import state_from_result
from pypresso.forces.spiral import compute_spiral_gradient, spiral_energy
from pypresso.io.pwin import parse_pw_input
from pypresso.pseudo import read_upf
from pypresso.scf import run_scf
from pypresso.scf.driver import Calculation
from pypresso.system import build_system
from pypresso.workflows.spiral import relax_spiral_q
from tests.conftest import GENERATED

pytestmark = [pytest.mark.regression, pytest.mark.slow]

#: A stationary point of ``E(q)`` fixed by ``E(-q) = E(q)`` is stationary in
#: exact arithmetic, so what is left is the k-sum's round-off.
SYMMETRY_ZERO = 1e-7


def _pseudos(system, pseudo_dir: Path):
    return tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)


@lru_cache(maxsize=None)
def _converged(q3: float, pseudo_dir: Path, ecutwfc: float = 25.0):
    """An SCF at one spiral wavevector, and the calculation it ran on.

    The calculation is returned as well as the result because the gradient needs
    both: the state is frozen, and the calculation is what says where it is
    frozen *at*.
    """
    text = (GENERATED / "h-chain-spiral.in").read_text()
    text = text.replace("spiral_q(3) = 0.25", f"spiral_q(3) = {q3}")
    text = text.replace("ecutwfc = 25.0", f"ecutwfc = {ecutwfc}")
    system = build_system(parse_pw_input(text))
    pseudos = _pseudos(system, pseudo_dir)
    calculation = Calculation(system, pseudos)
    result = run_scf(
        system, pseudos, calculation=calculation,
        conv_thr=1e-12, mixing_beta=0.3, max_iterations=300,
    )
    assert result.converged
    return calculation, result, pseudos


def test_the_differentiated_functional_is_the_total_energy(pseudo_dir):
    """Identity 1: ``spiral_energy`` at the converged state *is* ``etot``.

    The gradient only ever sees the kinetic and nonlocal terms -- everything
    else is independent of ``q`` at frozen coefficients -- so this is the only
    thing that checks the rest of the functional at all. It is an identity
    between two evaluations of the same expression and holds to round-off.
    """
    calculation, result, _ = _converged(0.3, pseudo_dir)
    gradient = compute_spiral_gradient(calculation, result)
    assert gradient.total_energy == pytest.approx(result.total_energy, abs=1e-12)


def test_the_gradient_differentiates_the_functional(pseudo_dir):
    """Identity 2: ``grad`` against a central difference of the same function.

    Frozen state *and* frozen basis on both sides, so the only thing that can
    differ is the differentiation. What is left is the finite difference's own
    ``delta^2 E'''/6``.
    """
    calculation, result, _ = _converged(0.3, pseudo_dir)
    state = state_from_result(result)
    gradient = compute_spiral_gradient(calculation, result)

    delta = 1.0e-5
    q = np.asarray(calculation.system.spiral_q, dtype=float)
    for axis in range(3):
        plus, minus = q.copy(), q.copy()
        plus[axis] += delta
        minus[axis] -= delta
        difference = (
            float(spiral_energy(calculation, jnp.asarray(plus), state))
            - float(spiral_energy(calculation, jnp.asarray(minus), state))
        ) / (2.0 * delta)
        assert difference == pytest.approx(gradient.gradient[axis], abs=1e-7)


def test_the_gradient_is_the_slope_of_the_converged_energy(pseudo_dir):
    """Identity 3: the stationarity test, at a frozen plane-wave sphere.

    Each side of the difference is a *separate SCF* -- the wavefunctions, the
    density, the eigenvalues and the occupations are all reconverged at the
    displaced wavevector -- and the gradient is taken at the midpoint with the
    state frozen. That the two agree is the whole claim of the method: at the
    fixed point the total derivative equals the partial one.

    The sphere is held fixed on both sides (``rebuild_basis = False``) so that
    what is compared is a differentiable function of ``q``; the module docstring
    says what the rebuilding one costs and
    :func:`test_the_basis_set_jumps_shrink_with_the_cutoff` measures it.
    """
    calculation, result, pseudos = _converged(0.3, pseudo_dir)
    gradient = compute_spiral_gradient(calculation, result)

    delta = 0.02
    energies = []
    for step in (+delta, -delta):
        q = np.asarray(calculation.system.spiral_q, dtype=float)
        q[2] += step
        moved = calculation.at_spiral_q(jnp.asarray(q), rebuild_basis=False)
        displaced = run_scf(
            moved.system, pseudos, calculation=moved,
            conv_thr=1e-12, mixing_beta=0.3, max_iterations=300,
        )
        assert displaced.converged
        energies.append(displaced.total_energy)

    difference = (energies[0] - energies[1]) / (2.0 * delta)
    # ``delta^2 E'''/6`` on this surface, measured: 5.2e-5 at delta = 0.02 and
    # falling by four when delta is halved, which is what makes it truncation
    # rather than a missing term.
    assert difference == pytest.approx(gradient.gradient[2], abs=1e-4)


@pytest.mark.parametrize("q3", [0.0, 0.5])
def test_symmetry_fixes_two_wavevectors_exactly(q3, pseudo_dir):
    """Identity 4: ``E(-q) = E(q)`` makes ``q = 0`` and ``q = b3/2`` stationary.

    No tolerance judgement anywhere: the gradient is zero in exact arithmetic
    whatever the electrons do, so a nonzero one is a bug in the shifted spheres,
    the ``+-q/2`` split or the chain rule and nothing else. It also catches the
    two transverse components, which the finite-difference tests exercise only
    weakly.
    """
    calculation, result, _ = _converged(q3, pseudo_dir)
    gradient = compute_spiral_gradient(calculation, result)
    assert np.abs(gradient.gradient).max() < SYMMETRY_ZERO


def test_the_basis_set_jumps_shrink_with_the_cutoff(pseudo_dir):
    """The jumps in ``E(q)`` are jumps, and a converged basis does not have them.

    The difference the *scan* would take -- new spheres at every point -- against
    the gradient, at two cutoffs. It is a bad test of the gradient and a good
    test of the claim the module docstring makes about why: 8.3e-4 at
    ``ecutwfc = 25`` and 8.3e-6 at 60, so the disagreement is basis-set
    incompleteness rather than a term that is missing from the derivative.

    **Not a convergence rate**, and the assertions are one-sided for that reason:
    the error is the sum of the jumps that fall inside this particular window of
    ``q``, not a truncated series, so it is erratic in the cutoff -- 40 Ry gives
    5.8e-4, which is *not* between the two. What shrinks monotonically is a
    single jump, since it is the size of a coefficient at the cutoff.
    """
    delta, discrepancies = 0.02, []
    for ecutwfc in (25.0, 60.0):
        calculation, result, _ = _converged(0.3, pseudo_dir, ecutwfc)
        gradient = compute_spiral_gradient(calculation, result)
        energies = [
            _converged(0.3 + step, pseudo_dir, ecutwfc)[1].total_energy
            for step in (+delta, -delta)
        ]
        difference = (energies[0] - energies[1]) / (2.0 * delta)
        discrepancies.append(abs(difference - gradient.gradient[2]))

    assert discrepancies[0] > 1e-4  # the jumps are there at a low cutoff
    assert discrepancies[1] < 1e-4  # and are gone by 60 Ry


def test_relaxation_finds_the_antiferromagnet(pseudo_dir):
    """The chain's ground state is ``q = b3/2``, and the relaxation walks to it.

    An unambiguous target: this hydrogen chain is a nearest-neighbour
    antiferromagnet, so its ``E(q)`` minimum is the zone boundary, where the
    moment reverses from cell to cell -- and it is a point the relaxation has to
    *reach* rather than one it starts at, since ``E(q)`` being even makes the
    zone boundary stationary and a run started there would report convergence
    without moving.

    The search is restricted to the chain axis (``free = (0, 0, 1)``). The other
    two components are zero by symmetry and their gradient is zero to 1e-9
    there; letting them move would only let the basis-set noise of the module
    docstring push them off the axis.
    """
    text = (GENERATED / "h-chain-spiral.in").read_text()
    text = text.replace("spiral_q(3) = 0.25", "spiral_q(3) = 0.30")
    system = build_system(parse_pw_input(text))
    pseudos = _pseudos(system, pseudo_dir)

    relaxed = relax_spiral_q(
        system, pseudos, mixing_beta=0.3, free=(0, 0, 1), nstep=20,
    )

    assert relaxed.converged
    assert relaxed.wavevector[2] == pytest.approx(0.5, abs=0.01)
    assert np.abs(relaxed.wavevector[:2]).max() == 0.0  # frozen, so exactly zero
    # It went downhill, and it got there in a handful of steps rather than by
    # crawling: the surface is smooth and BFGS's second step already uses a
    # curvature estimate.
    assert relaxed.total_energy < relaxed.steps[0].total_energy
    assert relaxed.nsteps <= 12


def test_a_gradient_is_refused_where_it_would_be_silently_wrong(pseudo_dir):
    """The two refusals, by name.

    A calculation that is not a spiral has no ``q`` to differentiate with
    respect to; one with a magnetic field is stationary for a functional that is
    not the one being differentiated (:mod:`pypresso.scf.fields`), and that term
    would be missing without anything looking wrong.
    """
    text = (GENERATED / "h-atom-lsda.in").read_text().replace(
        "    nspin = 2", "    noncolin = .true.\n    nosym = .true."
    )
    system = build_system(parse_pw_input(text))
    calculation = Calculation(system, _pseudos(system, pseudo_dir))
    with pytest.raises(ValueError, match="needs a spin spiral"):
        compute_spiral_gradient(calculation, None)

    spiral, result, _ = _converged(0.3, pseudo_dir)
    with_field = object.__new__(type(spiral))
    with_field.__dict__.update(spiral.__dict__)
    with_field.magnetic_field = "a field, of whatever kind"
    with pytest.raises(NotImplementedError, match="magnetic field"):
        compute_spiral_gradient(with_field, result)


@lru_cache(maxsize=None)
def _converged_at(tau: tuple, q3: float, pseudo_dir: Path):
    """The same chain with its atom somewhere other than the origin."""
    text = (GENERATED / "h-chain-spiral.in").read_text()
    text = text.replace("spiral_q(3) = 0.25", f"spiral_q(3) = {q3}")
    text = text.replace(" H 0.0 0.0 0.0", f" H {tau[0]} {tau[1]} {tau[2]}")
    system = build_system(parse_pw_input(text))
    pseudos = _pseudos(system, pseudo_dir)
    calculation = Calculation(system, pseudos)
    result = run_scf(
        system, pseudos, calculation=calculation,
        conv_thr=1e-12, mixing_beta=0.3, max_iterations=300,
    )
    assert result.converged
    return calculation, result, pseudos


def test_moving_the_atom_does_not_move_the_gradient(pseudo_dir):
    """Identity 5: the structure factor's half of ``vkb(k +- q/2)``.

    With the atom at the origin the structure factor is one whatever ``q`` is,
    so every other test in this file is blind to it. Here the atom is at a
    general crystal position and the three checks are repeated:

    * ``E(q)`` and ``dE/dq`` are what they were at the origin. Translating the
      crystal is a symmetry of a spiral -- a lattice translation together with
      the spin rotation the theorem pairs it with -- so this is exact up to the
      FFT grid, which samples a shifted atom slightly differently.
    * ``jax.grad`` still equals a finite difference of the functional, now with
      a nontrivial phase in the chain.
    * and so does a finite difference of a *re-converged* SCF, which is the
      stationarity test with the term that was missing from it.

    The structure factor's contribution turns out to **cancel** for a
    scalar-relativistic spiral: its ``q`` derivative brings down ``-/+ i tau/2``
    on the two components, and ``dvan_so`` is diagonal in spin space when there
    is no spin-orbit coupling, so the two halves of each diagonal term subtract.
    That is a fact about the physics and not a licence to leave it untested --
    the whole point of the test is that the zero is *produced* rather than
    assumed, and it would stop being zero the moment ``D`` gained an off-diagonal
    block.
    """
    tau = (0.13, 0.07, 0.25)
    origin_calculation, origin_result, _ = _converged(0.3, pseudo_dir)
    origin = compute_spiral_gradient(origin_calculation, origin_result)

    calculation, result, pseudos = _converged_at(tau, 0.3, pseudo_dir)
    assert np.abs(np.asarray(calculation.system.structure.positions)).max() > 1.0
    gradient = compute_spiral_gradient(calculation, result)

    assert result.total_energy == pytest.approx(origin_result.total_energy, abs=1e-10)
    assert gradient.gradient == pytest.approx(origin.gradient, abs=1e-6)

    # ...and the two finite differences, exactly as at the origin.
    state = state_from_result(result)
    q = np.asarray(calculation.system.spiral_q, dtype=float)
    delta = 1.0e-5
    plus, minus = q.copy(), q.copy()
    plus[2] += delta
    minus[2] -= delta
    plumbing = (
        float(spiral_energy(calculation, jnp.asarray(plus), state))
        - float(spiral_energy(calculation, jnp.asarray(minus), state))
    ) / (2.0 * delta)
    assert plumbing == pytest.approx(gradient.gradient[2], abs=1e-7)

    delta = 0.02
    energies = []
    for step in (+delta, -delta):
        displaced_q = q.copy()
        displaced_q[2] += step
        moved = calculation.at_spiral_q(jnp.asarray(displaced_q), rebuild_basis=False)
        displaced = run_scf(
            moved.system, pseudos, calculation=moved,
            conv_thr=1e-12, mixing_beta=0.3, max_iterations=300,
        )
        assert displaced.converged
        energies.append(displaced.total_energy)
    stationarity = (energies[0] - energies[1]) / (2.0 * delta)
    assert stationarity == pytest.approx(gradient.gradient[2], abs=1e-4)


def test_a_moved_calculation_does_not_reuse_a_stale_gradient(pseudo_dir):
    """``at_positions``, ``at_kpoints`` and ``at_spiral_q`` all drop the cache.

    The compiled ``dE/dq`` closes over the calculation it was built from -- its
    plane-wave sphere, its local potential, its Ewald sum, its projector
    positions -- and all three of those methods go through ``copy.copy``, which
    would carry it onto the moved object. It would then be *evaluated*, at the
    old geometry or the old cutoff, and give a plausible wrong number rather
    than an error. Cheaper to assert than to debug.
    """
    calculation, result, _ = _converged(0.3, pseudo_dir)
    compute_spiral_gradient(calculation, result)
    assert "_spiral_gradient" in calculation.__dict__

    shifted = np.asarray(calculation.system.structure.positions) + 0.1
    assert "_spiral_gradient" not in calculation.at_positions(jnp.asarray(shifted)).__dict__
    assert "_spiral_gradient" not in calculation.at_spiral_q([0.0, 0.0, 0.35]).__dict__


def test_integrating_the_gradient_reproduces_the_scan(pseudo_dir):
    """``E(q)`` accumulated from ``dE/dq`` against ``E(q)`` differenced from ``E``.

    The two are the same curve and the gap between them is the finding. It is
    *not* quadrature error: refining the ``q`` path shrinks the trapezoid rule's
    own ``h^2`` and leaves the gap where it was, which is what makes it a
    property of the basis rather than of the sum. Measured here at
    ``ecutwfc = 25``, 7 points against 13.

    What the integrated curve buys is smoothness and nothing else, so the two
    other things a reader might hope for are asserted *not* to happen anywhere:
    it is the same k-mesh (the gradient is the exact derivative of the same
    fixed-mesh energy, which
    :func:`test_the_gradient_is_the_slope_of_the_converged_energy` is the proof
    of) and the same number of SCF runs.

    The endpoints are the free, sharp part: ``E(-q) = E(q)`` makes ``q = 0`` and
    ``q = b3/2`` stationary whatever the electrons do, so a scan that ends on
    them must report zero gradients there to round-off.
    """
    from pypresso.workflows.spiral import run_spiral_scan

    text = (GENERATED / "h-chain-spiral.in").read_text()
    system = build_system(parse_pw_input(text))
    pseudos = _pseudos(system, pseudo_dir)

    gaps = []
    for npoints in (7, 13):
        q = np.zeros((npoints, 3))
        q[:, 2] = np.linspace(0.0, 0.5, npoints)
        scan = run_spiral_scan(
            system, pseudos, q, gradients=True,
            conv_thr=1e-12, mixing_beta=0.3, max_iterations=300,
        )
        assert all(scan.converged)
        assert scan.gradients.shape == (npoints, 3)

        # The two stationary wavevectors the path ends on.
        assert np.abs(scan.gradients[0]).max() < SYMMETRY_ZERO
        assert np.abs(scan.gradients[-1]).max() < SYMMETRY_ZERO
        # Both curves are measured from the first point, which is where the
        # integral starts and where the energies are subtracted.
        assert scan.integrated[0] == 0.0

        # The antiferromagnet is the ground state and both routes must say so:
        # a sign error in the accumulation would put the minimum at q = 0.
        assert scan.integrated[-1] == pytest.approx(scan.relative[-1], abs=0.5)
        assert scan.integrated[-1] < -2.0
        gaps.append(float(np.abs(scan.integrated - scan.relative).max()))

    # Refining the path halves the step and so quarters the trapezoid rule's
    # error (0.051 -> 0.016 mRy on this cell, measured against a spline
    # quadrature of the same gradients). The gap does not follow it down --
    # 0.139 against 0.138 -- so what is left is the basis-set noise the
    # energies carry and the gradients do not.
    assert gaps[0] == pytest.approx(gaps[1], rel=0.1)
    assert 0.05 < gaps[0] < 0.5
