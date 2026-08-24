"""Structural relaxation: move the atoms until the forces vanish.

``PW/src/run_pwscf.f90``'s outer loop, which is short because everything hard is
somewhere else: converge the electrons at fixed positions, compute the forces,
ask the optimizer where to go next, and move. What this module owns is the four
things that loop has to get right.

**1. The setup is done once.** The FFT grid, the G-vector sphere, the k-points
and the symmetry group are all chosen for the starting geometry and are *not*
re-derived as the atoms move (``setup.f90`` runs once, whatever the ion dynamics
does). Two of them would otherwise change mid-relaxation: the FFT dimensions
must be a multiple of the denominators of the fractional translations, so a step
that breaks a symmetry would silently switch to a different grid -- and the
exchange-correlation energy is evaluated pointwise on that grid, so the energy
being minimised would jump by ~1e-6 Ry for a reason that is not physics. The
symmetry group is *checked* each step instead (``checkallsym``), which is what
:func:`~pypresso.system.symmetry.check_symmetry` does here.

**2. The next SCF starts from the last one's density.** ``update_pot.f90``
extrapolates; the default (`pot_extrapolation = 'atomic'`) moves the atomic
superposition to the new positions and carries the *difference* between the
converged density and the old atomic superposition along with it. That
difference is the part that took an SCF to find, and it barely changes when an
atom moves a hundredth of a bohr. It is worth several iterations a step and it
is exact in the limit that matters -- at convergence the starting guess is
irrelevant to the answer.

**3. The SCF threshold follows the relaxation.** ``move_ions.f90`` tightens
``conv_thr`` as the forces get small (``upscale``), because a force is a
derivative and needs a better-converged density than an energy does, but only
once the geometry is close enough for that to be worth paying for.

**4. Forces on frozen coordinates are zeroed, not omitted.** ``if_pos`` multiplies
the force after it is reported, so a frozen atom still has a force to look at and
simply is not allowed to follow it.

Variable-cell relaxation (``calculation = 'vc-relax'``) is
:mod:`pypresso.workflows.vc_relax`, which is this loop with nine more
coordinates. It does not break point 1, and the reason is worth reading there:
QE keeps the same G-vectors for the whole relaxation too (``scale_h.f90``
re-expresses their Miller indices against the new reciprocal cell and changes
nothing else) and then runs one *further* SCF, from scratch, at the relaxed
cell. Two runs, each with its setup done once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from pypresso.forces import compute_forces
from pypresso.relax import get_ion_dynamics
from pypresso.relax.bfgs import BFGSSettings
from pypresso.scf.driver import Calculation, SCFResult, run_scf
from pypresso.system.builder import System
from pypresso.system.symmetry import check_symmetry
from pypresso.units import BOHR_TO_ANGSTROM

__all__ = ["RelaxResult", "run_relax"]

#: ``upscale`` in ``Modules/read_namelists.f90``: how much tighter than the
#: input ``conv_thr`` the SCF is allowed to become as the relaxation converges.
UPSCALE = 100.0


@dataclass
class RelaxStep:
    """One ionic step: where the atoms were, and what was found there."""

    index: int
    positions: np.ndarray  # (nat, 3) cartesian bohr
    total_energy: float  # Ry
    max_force: float  # Ry/bohr, over the coordinates free to move
    scf_iterations: int
    conv_thr: float
    energy_error: float | None = None
    gradient_error: float | None = None


@dataclass
class RelaxResult:
    """The relaxed structure and the path taken to it."""

    converged: bool
    #: The final geometry, as a :class:`~pypresso.system.builder.System` -- the
    #: same object the run started from with the positions moved, so it can be
    #: handed straight to another calculation.
    system: System
    #: The last SCF, at the final geometry.
    scf: SCFResult
    #: The forces there, in Ry/bohr.
    forces: np.ndarray
    steps: list = field(default_factory=list)
    #: Set when the optimizer gave up (its line search stopped making progress)
    #: rather than converging.
    optimizer_failed: bool = False

    @property
    def positions(self) -> np.ndarray:
        return np.asarray(self.system.structure.positions)

    @property
    def positions_angstrom(self) -> np.ndarray:
        return self.positions * BOHR_TO_ANGSTROM

    @property
    def total_energy(self) -> float:
        return self.scf.total_energy

    @property
    def nsteps(self) -> int:
        return len(self.steps)


def run_relax(
    system: System,
    pseudos: tuple,
    *,
    nbnd: int | None = None,
    conv_thr: float = 1.0e-6,
    etot_conv_thr: float | None = None,
    forc_conv_thr: float | None = None,
    nstep: int | None = None,
    ion_dynamics: str | None = None,
    force_method: str | None = None,
    calculation: Calculation | None = None,
    diagonalization: str | None = None,
    mixing_mode: str = "anderson",
    mixing_beta: float = 0.7,
    k_batch: int | None | str = "default",
    density_extrapolation: str = "atomic",
    verbose: bool = False,
    **scf_options,
) -> RelaxResult:
    """Relax the atomic positions at fixed cell.

    The thresholds mean what they mean in a ``pw.x`` input: ``etot_conv_thr``
    (1e-4 Ry) and ``forc_conv_thr`` (1e-3 Ry/bohr) must *both* be satisfied,
    ``conv_thr`` is the SCF's starting threshold, and ``nstep`` caps the ionic
    steps.

    **They come from the input file unless given here.** ``None`` -- the
    default -- reads :attr:`System.relax`, which carries what ``&control`` and
    ``&ions`` said or QE's defaults if they said nothing
    (:class:`~pypresso.relax.settings.RelaxSettings`). Before that existed
    these arguments defaulted to QE's numbers directly and a file asking for
    anything else was parsed and ignored, so the two codes stopped at different
    points on the same curve and both reported success (`PLAN.md` P28b).
    """
    settings = system.relax
    etot_conv_thr = (
        settings.etot_conv_thr if etot_conv_thr is None else etot_conv_thr
    )
    forc_conv_thr = (
        settings.forc_conv_thr if forc_conv_thr is None else forc_conv_thr
    )
    nstep = settings.nstep if nstep is None else nstep
    ion_dynamics = settings.ion_dynamics if ion_dynamics is None else ion_dynamics
    calculation = calculation or Calculation(
        system, pseudos, diagonalization=diagonalization, k_batch=k_batch
    )
    optimizer = get_ion_dynamics(ion_dynamics)(
        at=np.asarray(system.cell.at),
        energy_thr=etot_conv_thr,
        grad_thr=forc_conv_thr,
        settings=BFGSSettings(),
    )

    upscale = settings.upscale
    free = system.structure.free
    starting_threshold = conv_thr
    threshold = conv_thr
    steps: list[RelaxStep] = []
    density = becsum = None
    converged = False

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
            starting_becsum=becsum,
            verbose=verbose,
            **scf_options,
        )
        forces = compute_forces(calculation, result, method=force_method)
        positions = np.asarray(calculation.system.structure.positions)

        moved, converged = optimizer.step(
            positions, result.total_energy, forces.forces * free
        )
        steps.append(RelaxStep(
            index=index,
            positions=positions,
            total_energy=result.total_energy,
            max_force=forces.max_force,
            scf_iterations=result.iterations,
            conv_thr=threshold,
            energy_error=getattr(optimizer, "energy_error", None),
            gradient_error=getattr(optimizer, "gradient_error", None),
        ))
        if verbose:
            print(f"ionic step {index:3d}   E = {result.total_energy:16.8f} Ry"
                  f"   max |F| = {forces.max_force:.6f} Ry/bohr"
                  f"   dE = {optimizer.energy_error:.2e}")
        if converged:
            break

        # ``move_ions``: a better-converged density is only worth paying for
        # once the geometry is close, and then it is worth a lot -- a force is a
        # derivative and is more sensitive to the density than the energy is.
        if getattr(optimizer, "step_accepted", False):
            threshold = max(
                starting_threshold / upscale,
                starting_threshold * min(
                    1.0,
                    optimizer.energy_error / (etot_conv_thr * upscale),
                    optimizer.gradient_error / (forc_conv_thr * upscale),
                ),
            )

        previous = calculation
        calculation = calculation.at_positions(jnp.asarray(moved))
        if not check_symmetry(
            calculation.system.cell, calculation.system.structure, calculation.symmetries
        ):
            raise RuntimeError(
                "the ionic step broke a symmetry the run was set up with "
                "(checkallsym): the FFT grid and the k-point set were chosen "
                "for that group and are no longer valid. Symmetrised forces "
                "cannot do this in exact arithmetic, so this is a bug or a "
                "structure whose symmetry was mis-detected"
            )
        density, becsum = _extrapolate(
            previous, calculation, result, density_extrapolation
        )

    return RelaxResult(
        converged=converged and not optimizer.failed,
        system=calculation.system,
        scf=result,
        forces=forces.forces,
        steps=steps,
        optimizer_failed=bool(optimizer.failed),
    )


def _extrapolate(previous: Calculation, moved: Calculation, result, scheme: str):
    """The starting density for the next geometry (``update_pot.f90``).

    ``'atomic'`` -- QE's ``pot_extrapolation = 'atomic'`` -- writes the density
    as *(superposition of atomic charges) + (what the SCF added to it)*, moves
    the first part with the atoms and keeps the second. ``'none'`` starts from
    the atomic superposition, which is what a fresh run does, and ``'previous'``
    reuses the converged density unchanged.

    ``becsum`` travels with the density and not separately. The SCF mixes the
    two as one state -- for PAW the one-centre potential is built from
    ``becsum`` before the Hamiltonian exists -- so handing over an extrapolated
    density with an *atomic* ``becsum`` would start the next geometry from two
    different states at once. The previous geometry's converged ``becsum`` is
    the right partner for the extrapolated density, and recomputing it from the
    wavefunctions costs one projection pass.
    """
    if scheme == "none":
        return None, None
    if scheme not in ("atomic", "previous"):
        raise ValueError(
            f"unknown density_extrapolation {scheme!r}; "
            "expected 'atomic', 'previous' or 'none'"
        )

    if scheme == "previous":
        density = result.density
    else:
        deformation = result.density - previous.starting_density()
        density = moved.starting_density() + deformation

    becsum = None
    if previous.is_ultrasoft:
        weights = result.occupations if result.nspin == 2 else result.occupations[None]
        becsum = previous.becsum(result.wavefunctions, jnp.asarray(weights))
    return density, becsum
