"""``E(q)``: a spin-spiral energy surface, and the exchange constants in it.

One SCF run per wavevector, and the point of collecting them is that ``E(q)``
*is* the magnon dispersion of a Heisenberg model. Mapping a classical Heisenberg
Hamiltonian ``H = - sum_ij J_ij e_i . e_j`` onto a flat spiral of unit moments
gives

    E(q) - E(0) = - m^2 sum_R J(R) [cos(q . R) - 1] = m^2 [J(0) - J(q)],

so a scan over ``q`` is the Fourier transform of the exchange constants
(Sandratskii's frozen-magnon method), and the curvature at ``q = 0`` is the spin
stiffness. That is what makes a spiral worth computing rather than a curiosity:
a handful of SCF runs give the parameters of a spin model that a supercell
calculation would need one cell per period to reach.

**Every point is an independent SCF and they share almost everything.** The
cell, the atoms, the pseudopotentials, the dense G set, the local potential and
the Ewald sum do not depend on ``q``; only the plane-wave spheres, ``|k+G|^2``,
the stick layout and ``vkb`` do. :meth:`~pypresso.scf.driver.Calculation.at_spiral_q`
rebuilds exactly those and shares the rest, the way ``at_kpoints`` does for a
k-list (P16 measured that sharing at 29.8x on a large cell).

**Reading the result.** The energies are per unit cell and include no
contribution from the field or constraint machinery (:mod:`pypresso.scf.fields`),
so differences between points are directly the magnetic energy. ``E(q)`` is even
in ``q`` and periodic under ``q -> q + 2G``; it is periodic under ``q -> q + G``
only on a k-grid invariant under a shift by ``G/2``, which is what
:mod:`pypresso.system.spiral` documents and what an even Monkhorst-Pack grid
gives.

**Relaxing ``q`` rather than scanning it.** A scan is the right tool when the
whole surface is wanted -- the exchange constants are a fit to it -- and the
wrong one when only its minimum is. :func:`relax_spiral_q` is the minimum, and
it is the ionic relaxation of :mod:`pypresso.workflows.relax` with one
substitution: the coordinate is ``q`` instead of the atomic positions, the
gradient is :func:`~pypresso.forces.spiral.compute_spiral_gradient` instead of
the force, and the optimizer is the same transcribed BFGS working in the
*reciprocal* cell's metric instead of the direct one. The physical content of
the answer is the pitch the magnet chooses: an incommensurate ground state comes
out as a ``q`` that is not a simple fraction, which no supercell calculation can
represent at all.

Two differences from moving atoms are worth stating, because they set what the
relaxation can be asked for:

* **A step of ``q`` rebuilds the plane-wave spheres**, since they are centred on
  ``k +- q/2``. The wavefunctions therefore cannot be carried from one step to
  the next -- they are coefficients on a basis that no longer exists -- but the
  density can, because in the rotated frame it is a lattice-periodic function on
  a grid that does not move. That is the warm start used here, and it is
  ``update_pot.f90``'s ``pot_extrapolation = 'file'`` rather than its atomic
  extrapolation: nothing has moved for an atomic superposition to follow.
* **The surface has a floor of basis-set noise.** ``E(q)`` jumps by a
  Pulay-sized amount wherever a plane wave crosses the cutoff, and the gradient
  -- taken at a frozen sphere -- does not see those jumps
  (:mod:`pypresso.forces.spiral`). Driving ``|dE/dq|`` below that scale is
  chasing the discretisation, so the thresholds default to a size a converged
  cutoff can actually deliver and the honest way to tighten them is to raise
  ``ecutwfc``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from pypresso.forces.spiral import SpiralGradient, compute_spiral_gradient
from pypresso.pseudo.upf import Pseudopotential
from pypresso.relax import get_ion_dynamics
from pypresso.relax.bfgs import BFGSSettings
from pypresso.scf.driver import Calculation, SCFResult, run_scf
from pypresso.system.builder import System

__all__ = ["SpiralScan", "run_spiral_scan", "heisenberg_exchange",
           "SpiralRelaxResult", "relax_spiral_q"]

#: The trust radius of an *ionic* step is a length in bohr and QE's defaults say
#: what a reasonable one is. A step in ``q`` has no such absolute scale: the
#: useful range of ``q`` is the Brillouin zone, whose size is the cell's and
#: differs by orders of magnitude between a molecule in a large box and a dense
#: metal. So the three radii are fractions of the zone's **linear size**, taken
#: as the cube root of its volume -- a single number that is right for a cubic
#: cell and does not collapse for an anisotropic one, where the shortest
#: reciprocal vector may be the one along a vacuum direction the physics never
#: uses. Once that scale is known they are ordinary
#: :class:`~pypresso.relax.bfgs.BFGSSettings`.
TRUST_RADIUS_MAX_FRACTION = 0.25
TRUST_RADIUS_INI_FRACTION = 0.08
TRUST_RADIUS_MIN_FRACTION = 1.0e-4

#: ``upscale``, as in ``move_ions.f90``: how much tighter than the input
#: ``conv_thr`` the SCF may become as the relaxation closes in. A gradient is a
#: derivative and needs a better density than an energy does.
UPSCALE = 100.0


@dataclass
class SpiralScan:
    """``E(q)`` over a list of spiral wavevectors."""

    #: ``(nq, 3)`` in lattice coordinates, as they were given.
    wavevectors: np.ndarray
    #: ``(nq,)`` total energies in Ry, per unit cell.
    energies: np.ndarray
    #: ``(nq, 3)`` the rotated-frame moment of each converged state, in Bohr
    #: magnetons -- the amplitude of the moment that turns, not a net moment.
    moments: np.ndarray
    converged: tuple
    results: tuple = field(default_factory=tuple)

    @property
    def relative(self) -> np.ndarray:
        """Energies measured from the first point, in mRy."""
        return 1.0e3 * (self.energies - self.energies[0])


def run_spiral_scan(
    system: System,
    pseudos: tuple[Pseudopotential, ...],
    wavevectors,
    keep_results: bool = False,
    **scf_options,
) -> SpiralScan:
    """One SCF per wavevector, sharing everything that does not depend on ``q``.

    Args:
        system: a spiral system -- ``noncolin``, ``nosym``, and a ``spiral_q``
            that this scan overrides point by point.
        wavevectors: ``(nq, 3)`` in lattice coordinates.
        keep_results: hold every :class:`~pypresso.scf.driver.SCFResult`. Off by default: each one
            carries its wavefunctions, which is the largest array in the run.
    """
    wavevectors = np.asarray(wavevectors, dtype=float).reshape(-1, 3)
    if not system.spiral:
        raise ValueError(
            "run_spiral_scan needs a system with spiral_q set: it is what makes "
            "the run noncollinear, symmetry-free and two-sphered"
        )

    base = Calculation(system, pseudos, k_batch=scf_options.pop("k_batch", "default"))
    energies, moments, converged, results = [], [], [], []
    for q in wavevectors:
        calculation = base.at_spiral_q(q)
        result = run_scf(
            calculation.system, pseudos, calculation=calculation, **scf_options
        )
        energies.append(result.total_energy)
        moments.append(result.magnetization_vector or (0.0, 0.0, 0.0))
        converged.append(bool(result.converged))
        if keep_results:
            results.append(result)

    return SpiralScan(
        wavevectors=wavevectors,
        energies=np.array(energies),
        moments=np.array(moments),
        converged=tuple(converged),
        results=tuple(results),
    )


def heisenberg_exchange(scan: SpiralScan, cell, shells) -> np.ndarray:
    """Fit ``E(q) - E(0) = m^2 sum_R J(R) [1 - cos(q . R)]`` for the ``J(R)``.

    Args:
        shells: ``(nshell, 3)`` lattice vectors in *crystal* coordinates, one
            per neighbour shell to fit. The moment is taken from the scan's own
            converged states, so the ``J`` come out in Ry per pair of unit
            vectors -- the convention in which ``H = -sum_ij J_ij e_i . e_j``.

    A least-squares fit rather than an inversion: the number of ``q`` points is
    usually larger than the number of shells, and the residual is the honest
    statement of how well a Heisenberg model describes the surface. A large one
    means the moments' *magnitude* is changing with ``q``, which is exactly what
    the ``moments`` column of the scan is there to show.
    """
    q = np.asarray(scan.wavevectors, dtype=float)
    shells = np.asarray(shells, dtype=float).reshape(-1, 3)
    # q is in lattice (reciprocal) coordinates and R in crystal coordinates, so
    # q . R is 2 pi times their dot product -- no metric needed, which is the
    # convenience those two conventions exist for.
    phase = 1.0 - np.cos(2.0 * np.pi * (q @ shells.T))
    amplitude = np.linalg.norm(np.asarray(scan.moments), axis=1)
    magnitude = float(np.mean(amplitude[amplitude > 0.0])) if np.any(amplitude) else 1.0
    energies = scan.energies - scan.energies[0]
    solution, *_ = np.linalg.lstsq(phase * magnitude**2, energies, rcond=None)
    return solution


@dataclass
class SpiralRelaxStep:
    """One step of a ``q`` relaxation: where it was, and what was found there."""

    index: int
    #: The wavevector, in lattice coordinates.
    wavevector: np.ndarray
    total_energy: float  # Ry
    #: ``max |dE/dq|`` in Ry per ``2 pi / alat``, over the free components.
    max_gradient: float
    scf_iterations: int
    conv_thr: float
    energy_error: float | None = None
    gradient_error: float | None = None


@dataclass
class SpiralRelaxResult:
    """The relaxed spiral, and the path taken to it."""

    converged: bool
    #: The final wavevector in lattice coordinates -- the pitch of the ground
    #: state, and the number the whole calculation exists to produce.
    wavevector: np.ndarray
    #: The final system, with ``spiral_q`` at that wavevector, so it can be
    #: handed straight to another calculation.
    system: System
    #: The last SCF, at the final wavevector.
    scf: SCFResult
    #: The gradient there.
    gradient: SpiralGradient
    steps: list = field(default_factory=list)
    #: Set when the optimizer gave up -- its line search stopped making
    #: progress -- rather than converging. On a spiral surface this most often
    #: means the steps have reached the size at which the basis-set jumps in
    #: ``E(q)`` are as large as the energy differences being resolved.
    optimizer_failed: bool = False

    @property
    def total_energy(self) -> float:
        return self.scf.total_energy

    @property
    def moment(self) -> tuple | None:
        """The rotated-frame moment at the relaxed wavevector, in Bohr magnetons."""
        return self.scf.magnetization_vector

    @property
    def nsteps(self) -> int:
        return len(self.steps)


def relax_spiral_q(
    system: System,
    pseudos: tuple[Pseudopotential, ...],
    *,
    nbnd: int | None = None,
    conv_thr: float = 1.0e-8,
    etot_conv_thr: float = 1.0e-5,
    grad_conv_thr: float = 1.0e-4,
    nstep: int = 30,
    free=(1, 1, 1),
    ion_dynamics: str | None = None,
    calculation: Calculation | None = None,
    diagonalization: str | None = None,
    mixing_mode: str = "anderson",
    mixing_beta: float = 0.7,
    k_batch: int | None | str = "default",
    warm_start: bool = True,
    verbose: bool = False,
    **scf_options,
) -> SpiralRelaxResult:
    """Move ``q`` downhill until ``dE/dq`` vanishes: the ground-state spiral.

    The counterpart of :func:`~pypresso.workflows.relax.run_relax` on the other
    coordinate. Converge the electrons at fixed ``q``, take ``dE/dq``, ask BFGS
    where to go, rebuild the spheres there, repeat.

    Args:
        system: a spiral system. Its ``spiral_q`` is the starting guess, and it
            matters: ``E(q)`` is even in ``q``, so ``q = 0`` and the zone
            boundary are stationary points *by symmetry* and a relaxation
            started at either one will report convergence without moving. Start
            between them.
        conv_thr: the SCF's starting threshold. Tighter than a plain SCF's
            default by two decades, for the reason ``move_ions.f90`` gives: a
            gradient is more sensitive to the density than an energy is.
        etot_conv_thr: in Ry, and ``grad_conv_thr`` in Ry per ``2 pi / alat``.
            Both must be satisfied, as in a ``pw.x`` relaxation -- and of the
            two it is the **energy** one that is loose here, at 1e-5 Ry rather
            than the 1e-4 an ionic relaxation uses over an energy scale a
            hundred times larger. It has to be: near the minimum the energy
            differences being resolved are of the size of the basis-set jumps
            described above (measured on the hydrogen chain at ``ecutwfc = 40``:
            3e-6 Ry between two wavevectors 0.006 apart, where the physics is
            8e-7), so a tighter one is asking the line search to follow
            discretisation noise. The gradient threshold is the one that carries
            the physics, and it is the one to tighten -- after raising the
            cutoff.
        free: which components of ``q`` may move, as ``if_pos`` does for an
            atom -- and, like ``if_pos``, **cartesian**: it multiplies
            ``dE/dq`` in units of ``2 pi / alat``, not the lattice coordinates
            ``spiral_q`` is written in. The two coincide whenever the reciprocal
            vectors are orthogonal, which covers the case this is for: a search
            along one symmetry direction, ``(0, 0, 1)`` for a chain or a
            tetragonal or hexagonal axis, where the other two components are
            zero by symmetry and letting them move only lets basis-set noise
            push them off it. Freezing a single *lattice* component of an
            oblique cell is not what this does and needs a projector rather than
            a mask.
        warm_start: hand each step the previous one's converged density. The
            wavefunctions cannot travel -- a new ``q`` is a new plane-wave
            sphere -- but the density is a lattice-periodic function on a grid
            that does not move, so it costs nothing and saves several SCF
            iterations a step.
    """
    if not system.spiral:
        raise ValueError(
            "relax_spiral_q needs a system with spiral_q set: it is the "
            "coordinate being relaxed, and it is what makes the run "
            "noncollinear, symmetry-free and two-sphered"
        )
    calculation = calculation or Calculation(
        system, pseudos, diagonalization=diagonalization, k_batch=k_batch
    )
    cell = system.cell
    # The optimizer's "cartesian" is ``q`` in units of ``2 pi / alat`` and its
    # "crystal" is ``q`` in lattice coordinates, so the lattice it is handed is
    # the *reciprocal* one and its metric ``b_i . b_j`` is what measures the
    # length of a step. That is the whole of the substitution: everything else
    # in BFGS -- the trust radius, the Wolfe line search, the damped Hessian
    # update -- is the same arithmetic on a single three-component coordinate.
    bg = np.asarray(cell.bg_2pi_alat, dtype=float)
    scale = float(abs(np.linalg.det(bg))) ** (1.0 / 3.0)
    settings = BFGSSettings(
        trust_radius_max=TRUST_RADIUS_MAX_FRACTION * scale,
        trust_radius_ini=TRUST_RADIUS_INI_FRACTION * scale,
        trust_radius_min=TRUST_RADIUS_MIN_FRACTION * scale,
    )
    optimizer = get_ion_dynamics(ion_dynamics)(
        at=bg, energy_thr=etot_conv_thr, grad_thr=grad_conv_thr, settings=settings,
    )

    free = np.asarray(free, dtype=float).reshape(1, 3)
    starting_threshold = threshold = conv_thr
    steps: list[SpiralRelaxStep] = []
    density = None
    converged = False
    q_crystal = np.asarray(system.spiral_q, dtype=float)

    for index in range(1, nstep + 1):
        result = run_scf(
            calculation.system,
            pseudos,
            nbnd=nbnd,
            conv_thr=threshold,
            calculation=calculation,
            mixing_mode=mixing_mode,
            mixing_beta=mixing_beta,
            starting_density=density,
            verbose=verbose,
            **scf_options,
        )
        gradient = compute_spiral_gradient(calculation, result)
        q_crystal = np.asarray(calculation.system.spiral_q, dtype=float)
        max_gradient = float(np.abs(gradient.force * free).max())

        if index == 1:
            # BFGS's first step is a Newton step through an inverse Hessian it
            # has no information for, and for the atoms QE's guess -- the
            # inverse metric, i.e. a curvature of 1 Ry/bohr^2 -- happens to be
            # the size of a chemical bond. On this surface it is out by two
            # orders of magnitude (the energy scale of magnetism is milli-Ry
            # over a coordinate of order one), so the first step comes out a
            # hundredth of the trust radius and the relaxation crawls until the
            # updates have rebuilt the curvature from scratch. Scaling the guess
            # so that the first Newton step is exactly ``trust_radius_ini`` long
            # says the honest thing instead: with no curvature information,
            # take a steepest-descent step of the length the trust radius
            # allows. Every step after this one uses a measured curvature.
            settings.hessian_scale = _first_step_scale(
                optimizer, gradient.gradient * free.reshape(3), settings
            )

        moved, converged = optimizer.step(
            (q_crystal @ bg).reshape(1, 3),
            result.total_energy,
            gradient.force.reshape(1, 3) * free,
        )
        steps.append(SpiralRelaxStep(
            index=index,
            wavevector=q_crystal,
            total_energy=result.total_energy,
            max_gradient=max_gradient,
            scf_iterations=result.iterations,
            conv_thr=threshold,
            energy_error=getattr(optimizer, "energy_error", None),
            gradient_error=getattr(optimizer, "gradient_error", None),
        ))
        if verbose:
            print(f"spiral step {index:3d}   q = "
                  f"({q_crystal[0]:8.5f}, {q_crystal[1]:8.5f}, {q_crystal[2]:8.5f})"
                  f"   E = {result.total_energy:16.8f} Ry"
                  f"   max |dE/dq| = {max_gradient:.6f}"
                  f"   dE = {optimizer.energy_error:.2e}")
        if converged:
            break

        if getattr(optimizer, "step_accepted", False):
            threshold = max(
                starting_threshold / UPSCALE,
                starting_threshold * min(
                    1.0,
                    optimizer.energy_error / (etot_conv_thr * UPSCALE),
                    optimizer.gradient_error / (grad_conv_thr * UPSCALE),
                ),
            )

        # The spheres are rebuilt here -- ``rebuild_basis`` defaults to True --
        # because this is a real move of ``q`` and the frozen sphere the
        # gradient was taken on belongs to the point that was just left.
        density = result.density if warm_start else None
        calculation = calculation.at_spiral_q(optimizer.to_crystal(moved).reshape(3))

    # ``q_crystal`` and not the calculation's own ``spiral_q``: the last thing
    # the loop does when it has *not* converged is move, so the two differ
    # exactly when the step budget ran out -- and the wavevector reported has to
    # be the one ``scf`` and ``gradient`` were evaluated at.
    return SpiralRelaxResult(
        converged=converged and not optimizer.failed,
        wavevector=q_crystal,
        system=dataclasses.replace(
            calculation.system, spiral_q=tuple(float(v) for v in q_crystal)
        ),
        scf=result,
        gradient=gradient,
        steps=steps,
        optimizer_failed=bool(optimizer.failed),
    )


def _first_step_scale(optimizer, gradient, settings: BFGSSettings) -> float:
    """The multiple of the inverse metric that makes step one the trust radius.

    ``direction = -s M^-1 g`` and its length in the cell metric is linear in
    ``s``, so the scale that makes it ``trust_radius_ini`` is one division. A
    vanishing gradient means the run has converged before it started and the
    scale is irrelevant, so it falls back to QE's.
    """
    step = (optimizer.inverse_metric @ np.asarray(gradient)).reshape(1, 3)
    length = optimizer._norm(step)
    if length < 1.0e-30:
        return 1.0
    return settings.trust_radius_ini / length
