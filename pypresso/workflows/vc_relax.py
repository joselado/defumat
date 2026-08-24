"""Variable-cell relaxation: move the atoms *and* the cell until both are done.

``calculation = 'vc-relax'``. The atoms follow the force and the cell follows
the stress, in one BFGS over ``3 nat + 9`` coordinates
(:mod:`pypresso.relax.bfgs`), minimising the **enthalpy** ``E + P Omega``. What
this module owns is the outer loop and the two things that make a moving cell
different from moving atoms.

**1. A vc-relax is two runs, and that is what resolves the objection to it.**
`PLAN.md` refused this phase on the grounds that a moving cell invalidates the
rule that the FFT grid and the symmetry group are fixed once for the whole run.
It does not, because Quantum ESPRESSO does not move them either.
``scale_h.f90`` re-expresses the *same* G-vectors -- the same Miller indices,
the same sphere membership, the same FFT dimensions, the same k-points in
crystal coordinates -- against the new reciprocal cell, and that is the whole
of what a step does to the basis
(:meth:`~pypresso.scf.driver.Calculation.at_cell`). So the relaxation is one
run under the fixed-setup rule from beginning to end. Then, when it converges,
``reset_gvectors`` throws the setup away and runs **one more SCF from scratch**
at the relaxed geometry -- "Final scf calculation at the relaxed structure. The
G-vectors are recalculated for the final unit cell. Results may differ from
those at the preceding step." That second run is a second setup, and it too
obeys the rule.

**The difference between the two energies is not noise, it is the Pulay error**
of relaxing in a basis that was chosen for a different cell, and it is the
number to read before believing a relaxed volume. It is reported as
:attr:`VCRelaxResult.pulay_error` rather than left for the user to notice, and
it is the reason a vc-relax wants a higher ``ecutwfc`` than an SCF of the same
crystal: the basis has to be good enough that its *derivative* with respect to
the cell is small, not merely its value. ``treinit_gvectors`` is QE's escape
hatch and is here too -- rebuild everything on every accepted step, paying a
full setup per step to make the error zero.

**2. The relaxed crystal has the stress of the applied pressure, not zero.**
The stationary point of ``E + P Omega`` is ``sigma = P I``
(:mod:`pypresso.relax.cell`), which is what ``press_conv_thr`` measures and why
a third convergence threshold joins ``etot_conv_thr`` and ``forc_conv_thr``:
all three have to be satisfied at once, as in ``bfgs_module.f90``.

Everything else is :mod:`pypresso.workflows.relax`'s and is shared with it --
the density extrapolation between geometries, the SCF threshold that tightens
as the relaxation converges, and ``if_pos`` zeroing the force on a frozen
coordinate. The one addition is that the symmetry check grows a second half:
:func:`~pypresso.system.symmetry.check_symmetry` is blind to a deformation of
the cell because it works in crystal coordinates, so
:func:`~pypresso.system.symmetry.check_lattice_symmetry` checks the metric.

**Refused by name.** ``cell_dynamics`` other than ``'bfgs'`` -- QE's ``'damp-w'``
and ``'damp-pr'`` are Wentzcovitch/Parrinello-Rahman damped dynamics with a
fictitious cell mass, a different optimizer rather than a different setting --
and the ``cell_dofree`` values that impose a constraint beyond their mask
(:mod:`pypresso.relax.cell`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from pypresso.forces import compute_forces
from pypresso.relax import get_ion_dynamics
from pypresso.relax.bfgs import BFGSSettings
from pypresso.relax.cell import cell_dofree_mask
from pypresso.scf.driver import Calculation, SCFResult, run_scf
from pypresso.stress import compute_stress
from pypresso.system.builder import System
from pypresso.system.symmetry import check_lattice_symmetry, check_symmetry
from pypresso.units import BOHR_TO_ANGSTROM, RY_TO_KBAR
from pypresso.workflows.relax import RelaxStep, _extrapolate

__all__ = ["VCRelaxResult", "VCRelaxStep", "run_vc_relax"]


@dataclass
class VCRelaxStep(RelaxStep):
    """One ionic step, with what the cell was doing at it."""

    #: ``(3, 3)`` lattice vectors as rows, in bohr, before the step.
    cell: np.ndarray | None = None
    #: The cell volume there, in bohr^3.
    volume: float = 0.0
    #: ``E + P Omega``, in Ry -- the quantity actually being minimised.
    enthalpy: float = 0.0
    #: ``max |P I - sigma|`` in Ry/bohr^3: how far the stress is from the
    #: target pressure. QE prints this in kbar as the "cell gradient error".
    cell_error: float = np.inf


@dataclass
class VCRelaxResult:
    """The relaxed cell and structure, and the path taken to them."""

    converged: bool
    #: The final geometry -- cell *and* positions -- as a rebuilt
    #: :class:`~pypresso.system.builder.System`, with its own k-points, so it
    #: can be handed straight to another calculation.
    system: System
    #: The **final** SCF: a fresh run at the relaxed geometry with the
    #: G-vectors recalculated for it, which is the energy to quote. When
    #: ``final_scf=False`` this is the last SCF of the relaxation instead.
    scf: SCFResult
    #: The last SCF *of the relaxation*, in the basis the relaxation used.
    relaxation_scf: SCFResult
    forces: np.ndarray
    #: The stress at the relaxed geometry, Ry/bohr^3, from the final SCF.
    stress: np.ndarray
    #: The target pressure, in Ry/bohr^3.
    pressure: float = 0.0
    steps: list = field(default_factory=list)
    optimizer_failed: bool = False
    #: Whether the final SCF ran at all.
    final_scf: bool = True

    @property
    def cell(self) -> np.ndarray:
        return np.asarray(self.system.cell.at)

    @property
    def volume(self) -> float:
        return float(self.system.cell.volume)

    @property
    def positions(self) -> np.ndarray:
        return np.asarray(self.system.structure.positions)

    @property
    def positions_crystal(self) -> np.ndarray:
        return np.asarray(self.system.structure.positions_crystal(self.system.cell))

    @property
    def positions_angstrom(self) -> np.ndarray:
        return self.positions * BOHR_TO_ANGSTROM

    @property
    def total_energy(self) -> float:
        return self.scf.total_energy

    @property
    def enthalpy(self) -> float:
        return self.total_energy + self.pressure * self.volume

    @property
    def pulay_error(self) -> float:
        """``E(final, own basis) - E(last step, relaxation's basis)``, in Ry.

        The price of having relaxed in a basis chosen for the starting cell --
        QE's "results may differ from those at the preceding step", as a number.
        Zero with ``treinit_gvectors``, and the figure to look at before
        believing a relaxed volume from a low cutoff.
        """
        if not self.final_scf:
            return 0.0
        return self.scf.total_energy - self.relaxation_scf.total_energy

    @property
    def nsteps(self) -> int:
        return len(self.steps)


def run_vc_relax(
    system: System,
    pseudos: tuple,
    *,
    press: float | None = None,
    press_conv_thr: float | None = None,
    cell_dofree: str | None = None,
    cell_dynamics: str | None = None,
    treinit_gvectors: bool | None = None,
    final_scf: bool = True,
    nbnd: int | None = None,
    conv_thr: float = 1.0e-6,
    etot_conv_thr: float | None = None,
    forc_conv_thr: float | None = None,
    nstep: int | None = None,
    ion_dynamics: str | None = None,
    force_method: str | None = None,
    stress_method: str | None = None,
    calculation: Calculation | None = None,
    diagonalization: str | None = None,
    mixing_mode: str = "anderson",
    mixing_beta: float = 0.7,
    k_batch: int | None | str = "default",
    density_extrapolation: str = "atomic",
    verbose: bool = False,
    **scf_options,
) -> VCRelaxResult:
    """Relax the atomic positions and the cell at fixed applied pressure.

    Args:
        press: the applied pressure in **kbar**, as ``pw.x`` reads it. Converted
            to Ry/bohr^3 here and nowhere else.
        press_conv_thr: how close the stress must come to ``press``, in kbar.
        cell_dofree: which components of the cell may move
            (:mod:`pypresso.relax.cell`).
        treinit_gvectors: rebuild the FFT grid, the G-sphere, the symmetry group
            and the k-points on every accepted step instead of once. Removes the
            Pulay error and costs a full setup per step; the density cannot be
            carried across such a step, so each one starts from the atomic
            superposition, as ``reset_gvectors`` does.
        final_scf: run the extra SCF at the relaxed geometry with the basis
            rebuilt for it. On by default because it is the energy QE reports
            and the only one that is variational in its own cell.

    The other thresholds mean what they mean in a ``pw.x`` input, and all three
    of ``etot_conv_thr``, ``forc_conv_thr`` and ``press_conv_thr`` must be
    satisfied together. Every one of these arguments defaults to ``None``,
    which reads the value off :attr:`System.relax` -- what the input's
    ``&control``, ``&ions`` and ``&cell`` said, or QE's default where they said
    nothing (:class:`~pypresso.relax.settings.RelaxSettings`).
    """
    settings = system.relax
    press = settings.press if press is None else press
    press_conv_thr = (
        settings.press_conv_thr if press_conv_thr is None else press_conv_thr
    )
    cell_dofree = settings.cell_dofree if cell_dofree is None else cell_dofree
    cell_dynamics = (
        settings.cell_dynamics if cell_dynamics is None else cell_dynamics
    )
    treinit_gvectors = (
        settings.treinit_gvectors if treinit_gvectors is None else treinit_gvectors
    )
    etot_conv_thr = (
        settings.etot_conv_thr if etot_conv_thr is None else etot_conv_thr
    )
    forc_conv_thr = (
        settings.forc_conv_thr if forc_conv_thr is None else forc_conv_thr
    )
    nstep = settings.nstep if nstep is None else nstep
    ion_dynamics = settings.ion_dynamics if ion_dynamics is None else ion_dynamics
    _check_cell_dynamics(cell_dynamics)
    pressure = press / RY_TO_KBAR
    base = calculation or Calculation(
        system, pseudos, diagonalization=diagonalization, k_batch=k_batch
    )
    optimizer = get_ion_dynamics(ion_dynamics)(
        at=np.asarray(base.system.cell.at),
        energy_thr=etot_conv_thr,
        grad_thr=forc_conv_thr,
        settings=BFGSSettings(),
        variable_cell=True,
        pressure=pressure,
        cell_thr=press_conv_thr / RY_TO_KBAR,
        cell_mask=cell_dofree_mask(cell_dofree),
    )

    current = base
    free = system.structure.free
    starting_threshold = threshold = conv_thr
    steps: list[VCRelaxStep] = []
    density = becsum = None
    converged = False

    for index in range(1, nstep + 1):
        result = run_scf(
            current.system, pseudos, nbnd=nbnd, conv_thr=threshold,
            calculation=current, mixing_mode=mixing_mode, mixing_beta=mixing_beta,
            starting_density=density, starting_becsum=becsum, verbose=verbose,
            **scf_options,
        )
        forces = compute_forces(current, result, method=force_method)
        stress = compute_stress(current, result, method=stress_method)
        positions = np.asarray(current.system.structure.positions)
        cell = np.asarray(current.system.cell.at)
        volume = float(current.system.cell.volume)

        moved, converged = optimizer.step(
            positions, result.total_energy, forces.forces * free,
            stress=stress.tensor,
        )
        steps.append(VCRelaxStep(
            index=index,
            positions=positions,
            total_energy=result.total_energy,
            max_force=forces.max_force,
            scf_iterations=result.iterations,
            conv_thr=threshold,
            energy_error=optimizer.energy_error,
            gradient_error=optimizer.gradient_error,
            cell=cell,
            volume=volume,
            enthalpy=result.total_energy + pressure * volume,
            cell_error=optimizer.cell_error,
        ))
        if verbose:
            print(
                f"vc step {index:3d}   H = {steps[-1].enthalpy:16.8f} Ry"
                f"   V = {volume:10.4f}   max |F| = {forces.max_force:.6f}"
                f"   |P I - sigma| = {optimizer.cell_error * RY_TO_KBAR:.3f} kbar"
            )
        if converged:
            break

        if getattr(optimizer, "step_accepted", False):
            threshold = max(
                starting_threshold / settings.upscale,
                starting_threshold * min(
                    1.0,
                    optimizer.energy_error / (etot_conv_thr * settings.upscale),
                    optimizer.gradient_error / (forc_conv_thr * settings.upscale),
                ),
            )

        previous = current
        current, density, becsum = _advance(
            base, previous, np.asarray(optimizer.at), moved, result,
            pseudos, treinit_gvectors, density_extrapolation,
            diagonalization, k_batch,
        )

    relaxed = current.system
    relaxation_scf = result
    if final_scf and not treinit_gvectors:
        # ``reset_gvectors``: a whole new run at the relaxed geometry, with
        # nothing carried over -- not the density, not the wavefunctions, not
        # the grids. Anything carried would be carried in the old basis.
        relaxed = system.with_cell(current.system.cell.at,
                                   current.system.structure.positions)
        final = Calculation(relaxed, pseudos, diagonalization=diagonalization,
                            k_batch=k_batch)
        result = run_scf(relaxed, pseudos, nbnd=nbnd, conv_thr=conv_thr,
                         calculation=final, mixing_mode=mixing_mode,
                         mixing_beta=mixing_beta, verbose=verbose, **scf_options)
        forces = compute_forces(final, result, method=force_method)
        stress = compute_stress(final, result, method=stress_method)

    return VCRelaxResult(
        converged=converged and not optimizer.failed,
        system=relaxed,
        scf=result,
        relaxation_scf=relaxation_scf,
        forces=forces.forces,
        stress=stress.tensor,
        pressure=pressure,
        steps=steps,
        optimizer_failed=bool(optimizer.failed),
        final_scf=bool(final_scf and not treinit_gvectors),
    )


def _advance(
    base, previous, at, positions, result, pseudos, treinit_gvectors,
    density_extrapolation, diagonalization, k_batch,
):
    """The next step's calculation, and the density to start it from."""
    if treinit_gvectors:
        # A new grid means a density on a different mesh and a sphere with
        # different Miller indices, so nothing crosses: ``reset_gvectors`` sets
        # ``starting_pot = 'atomic'`` and starts over.
        system = previous.system.with_cell(at, positions)
        _check_geometry(system, previous)
        return (
            Calculation(system, pseudos, diagonalization=diagonalization,
                        k_batch=k_batch),
            None, None,
        )

    # Cumulative from the *starting* calculation rather than chained from the
    # last one: the deformation is then the one relating the current cell to
    # the cell the G-vectors were enumerated for, which is what ``scale_h``
    # applies, and no rounding accumulates along the trajectory.
    current = base.at_cell(jnp.asarray(at)).at_positions(jnp.asarray(positions))
    _check_geometry(current.system, previous)
    density, becsum = _extrapolate(
        previous, current, result, density_extrapolation
    )
    return current, density, becsum


def _check_geometry(system: System, previous) -> None:
    """``checkallsym``, in both halves -- the structure and the lattice."""
    if not check_lattice_symmetry(system.cell, previous.symmetries):
        raise RuntimeError(
            "the cell step broke a symmetry the run was set up with: the "
            "deformed lattice no longer admits every rotation of the group the "
            "FFT grid and the k-point set were chosen for. A stress "
            "symmetrised over that group cannot do this in exact arithmetic, "
            "so this is a bug or a cell whose symmetry was mis-detected"
        )
    if not check_symmetry(system.cell, system.structure, previous.symmetries):
        raise RuntimeError(
            "the ionic step broke a symmetry the run was set up with "
            "(checkallsym): the FFT grid and the k-point set were chosen for "
            "that group and are no longer valid"
        )


#: ``cell_dynamics`` values ``pw.x`` accepts, with what each would need here.
_CELL_DYNAMICS = {
    "bfgs": None,
    "none": None,
    "sd": "steepest descent on the cell (Modules/cell_base.f90, calc = 'sd')",
    "damp-pr": (
        "Parrinello-Rahman damped dynamics with a fictitious cell mass "
        "(wmass), which is a different optimizer and not a setting of this one"
    ),
    "damp-w": (
        "Wentzcovitch damped dynamics with a fictitious cell mass (wmass), "
        "which is a different optimizer and not a setting of this one"
    ),
    "pr": "Parrinello-Rahman variable-cell molecular dynamics",
    "w": "Wentzcovitch variable-cell molecular dynamics",
}


def _check_cell_dynamics(cell_dynamics: str | None) -> None:
    if cell_dynamics is None:
        return
    name = str(cell_dynamics).strip().strip("'\"").lower()
    if name not in _CELL_DYNAMICS:
        raise ValueError(
            f"unknown cell_dynamics {name!r}; expected one of "
            f"{', '.join(sorted(_CELL_DYNAMICS))}"
        )
    if _CELL_DYNAMICS[name] is not None:
        raise NotImplementedError(
            f"cell_dynamics = {name!r} is not implemented -- it needs "
            f"{_CELL_DYNAMICS[name]}. It is refused rather than run as BFGS, "
            "which would relax the same crystal by a different path and report "
            "success under a name that did not happen"
        )
