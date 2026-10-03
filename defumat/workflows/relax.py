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
:func:`~defumat.system.symmetry.check_symmetry` does here.

**2. The next SCF starts from the last one's density.** ``update_pot.f90``
extrapolates; the default (`pot_extrapolation = 'atomic'`) moves the atomic
superposition to the new positions and carries the *difference* between the
converged density and the old atomic superposition along with it. That
difference is the part that took an SCF to find, and it barely changes when an
atom moves a hundredth of a bohr. It is worth several iterations a step and it
is exact in the limit that matters -- at convergence the starting guess is
irrelevant to the answer.

The previous step's **wavefunctions** cross too, and that is not an
extrapolation. ``update_pot`` extrapolates them only when ``wfc_order > 0``
(``update_pot.f90:293``), whose default is 0 (``input.f90:1043``), so ``pw.x``
starts every later step from the previous geometry's converged states as they
are, and ``run_pwscf.f90:331-334`` diagonalises the first iteration of each such
step at ``ethr = 1e-6`` rather than at the 1e-2 a start from atomic orbitals
gets. Here the states are the span of the first Rayleigh-Ritz of the next
step's SCF and :data:`LATER_STEP_ETHR` is its ``diago_thr_init``. It is sound
because the cell is fixed: :meth:`~defumat.scf.driver.Calculation.at_positions`
keeps the plane-wave sphere, so a coefficient means the same plane wave at both
geometries.

Two differences from ``pw.x`` are deliberate. The states are first rotated in
the new Hamiltonian, the Rayleigh-Ritz every start goes through, where
``c_bands`` passes ``lrot = (iter == 1)`` to ``cegterg``, which then skips its
first subspace diagonalisation and takes the old states' diagonal as their
energies (``cegterg.f90:266``). And they cross with ``pw.x``'s
``atomic+random`` factor on every coefficient (:func:`_randomized`), because a
converged state keeps the symmetry of the geometry it came from and a step
that moves a state of another symmetry into the occupied manifold otherwise
leaves the next SCF in the wrong one, which was measured here and is the
reason for the factor. The states are not carried into a resumed step, which
starts from the checkpoint's density alone as before, nor under k-point pools,
nor into a residual solver, which starts its own.

**3. The SCF threshold follows the relaxation.** ``move_ions.f90`` tightens
``conv_thr`` as the forces get small (``upscale``), because a force is a
derivative and needs a better-converged density than an energy does, but only
once the geometry is close enough for that to be worth paying for.

**4. Forces on frozen coordinates are zeroed, not omitted.** ``if_pos`` multiplies
the force after it is reported, so a frozen atom still has a force to look at and
simply is not allowed to follow it.

Variable-cell relaxation (``calculation = 'vc-relax'``) is
:mod:`defumat.workflows.vc_relax`, which is this loop with nine more
coordinates. It does not break point 1, and the reason is worth reading there:
QE keeps the same G-vectors for the whole relaxation too (``scale_h.f90``
re-expresses their Miller indices against the new reciprocal cell and changes
nothing else) and then runs one *further* SCF, from scratch, at the relaxed
cell. Two runs, each with its setup done once.
"""

from __future__ import annotations

import dataclasses

import json
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from defumat.forces import compute_forces
from defumat.parallel import current_pools, stop_latched
from defumat.relax import get_ion_dynamics
from defumat.relax.bfgs import BFGSSettings
from defumat.scf.driver import Calculation, SCFResult, run_scf
from defumat.system.builder import System
from defumat.system.symmetry import check_symmetry
from defumat.scf.checkpoint import (
    load_optimizer, load_state, save_optimizer, save_state,
)
from defumat.units import BOHR_TO_ANGSTROM

__all__ = ["RelaxResult", "run_relax", "site_magnetization", "site_moment_report"]

#: ``upscale`` in ``Modules/read_namelists.f90``: how much tighter than the
#: input ``conv_thr`` the SCF is allowed to become as the relaxation converges.
UPSCALE = 100.0

#: ``ethr = 1.0D-6`` in ``run_pwscf.f90:331-334``: the first iteration's
#: diagonalisation threshold for every ionic step after the first, which starts
#: from the previous step's states rather than from atomic orbitals.
LATER_STEP_ETHR = 1.0e-6

#: ``0.05_DP`` in ``init_wfc``'s ``atomic+random`` (``wfcinit.f90:356``): the
#: size of the random factor on each coefficient of a carried state, and
#: :func:`_randomized` says why it is there and why it is not smaller.
CARRIED_NOISE = 0.05

#: The key the factor is drawn from. Any fixed value: what matters is that the
#: same run draws the same numbers. It is folded with each block's index and
#: never split the way ``starting_vectors``' own ``PRNGKey(0)`` is, so the two
#: draw different numbers although both start from 0.
CARRIED_SEED = 0


def _carries_states(scf_options: dict, pools) -> bool:
    """Whether a step's converged states can start the next step's SCF.

    Only the mixing loop reads ``starting_wavefunctions``: a residual solver
    starts its own, and the span would sit unread under its whole run. Under
    k-point pools the states are each pool's share of the set, which
    :meth:`~defumat.scf.driver.Calculation.starting_wavefunctions` can address
    but no relaxation here has been checked with, so a pooled relaxation starts
    each step from atomic orbitals as before.
    """
    from defumat.scf.solvers import get_scf_solver

    return (pools.size == 1
            and get_scf_solver(scf_options.get("scf_solver", "mixing")) is None)


def _perturb_block(states, index):
    """One ``(nbnd, ndim)`` block times ``1 + a rr1 exp(2 pi i rr2)``, keyed by its index."""
    first, second = jax.random.split(
        jax.random.fold_in(jax.random.PRNGKey(CARRIED_SEED), index))
    real = jnp.finfo(states.dtype).dtype
    size = jax.random.uniform(first, states.shape, dtype=real)
    angle = (2.0 * jnp.pi) * jax.random.uniform(second, states.shape, dtype=real)
    return states * (1.0 + CARRIED_NOISE * size
                     * jax.lax.complex(jnp.cos(angle), jnp.sin(angle)))


_perturb_one = jax.jit(_perturb_block)


@jax.jit
def _perturb_all(states):
    """Every block of a whole store at once, reshaped inside the trace, so free."""
    shape = states.shape
    blocks = int(np.prod(shape[:-2], dtype=int))
    flat = states.reshape((blocks,) + shape[-2:])
    return jax.vmap(_perturb_block)(flat, jnp.arange(blocks)).reshape(shape)


def _randomized(states):
    """The carried states with ``init_wfc``'s ``atomic+random`` factor on every coefficient.

    **Why the carried states are perturbed at all.** A converged state has the
    symmetry of the Hamiltonian it came from, exactly, whether or not the run
    uses that symmetry, and the Davidson iteration keeps it: every correction
    it adds is a preconditioned residual of one of the ``nbnd`` vectors it
    holds, in the same symmetry sector. So when an ionic step moves a state of
    one sector below a state of another across the Fermi level, a start from
    the old states cannot reach it, and the SCF converges, quietly, to the
    wrong occupied manifold. Measured on two-atom silicon with ``nosym`` and a
    ``2 2 2`` grid (``tests/unit/test_checkpoint.py``'s relaxation, ``nbnd = 4``,
    whose Hamiltonian keeps symmetries the run does not use: two atoms always
    have an inversion centre between them, and this displacement also keeps a
    mirror): at the second geometry the bare carried start converged to
    -15.5689 Ry with the fourth band at Gamma at 0.6345 Ry, where the atomic
    start gives -15.5954 Ry and 0.5353 and ``pw.x`` the same energy, and the
    relaxation then took 14 ionic steps instead of ``pw.x``'s 6. ``pw.x`` keeps
    ``evc`` bare and still reached the right state there, in 9.4 Davidson
    steps at that iteration; what lets it escape is not established, and this
    does not rely on it.

    The remedy is ``pw.x``'s own for the same loss in a start from atomic
    orbitals (``INPUT_PW.txt``, ``startingwfc``: "Prevents the 'loss' of states"):
    multiply every coefficient by ``1 + 0.05 rr1 exp(2 pi i rr2)`` with ``rr1``
    and ``rr2`` uniform on ``[0, 1)`` (``wfcinit.f90:353-356``). It gives every
    vector a component in every sector, and costs Davidson steps in the first
    iteration, since the states are no longer converged. At 0.01 the probe
    above found the missing state only at the fourth SCF iteration, in 11
    iterations against 4, and at 0.001 not at all, so the amplitude is
    ``pw.x``'s and is not tuned below it.

    The random numbers come from a fixed key folded with the block's index
    (channel and k-point), so a run is reproducible and a store held on the host
    gets the same numbers, block by block, as one on the device. The perturbed
    copy is made while the original is still held, the brief second set the
    hand-over costs; the original is released when this returns.
    """
    if isinstance(states, np.ndarray):
        # A store parked on the host: perturbed a block at a time, so the whole
        # set never crosses to the device, which is what parking it was for.
        shape = states.shape
        blocks = int(np.prod(shape[:-2], dtype=int))
        perturbed = np.empty_like(states)
        source = states.reshape((blocks,) + shape[-2:])
        target = perturbed.reshape((blocks,) + shape[-2:])
        for index in range(blocks):
            target[index] = np.asarray(_perturb_one(jnp.asarray(source[index]), index))
        return perturbed
    return _perturb_all(states)


def _next_step_options(scf_options: dict, carried: list) -> dict:
    """The inner SCF's options for a step, with ``diago_thr_init`` when states cross.

    ``run_pwscf.f90`` sets the threshold for every later step whatever
    ``diago_thr_init`` the input gave the first, so this one replaces a
    caller's.
    """
    if not carried:
        return scf_options
    return {**scf_options, "diago_thr_init": LATER_STEP_ETHR}


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
    #: ``report_mag`` at this step's converged density: the charge ``(nat,)``
    #: in electrons and the moment ``(nat, 1)`` collinear or ``(nat, 3)``
    #: noncollinear, in Bohr magnetons. ``None`` for a run with no
    #: magnetization.
    #:
    #: **A relaxation can unwind a texture and then converge cleanly**, exactly
    #: as an SCF can, and for the same reason: every quantity the path records
    #: -- the energy, the force, the cell -- is zero-sum over the sites, so a
    #: compensated magnet whose moments have collapsed looks like a compensated
    #: magnet that still has them. :attr:`SCFResult.site_moments` answers the
    #: question at the *last* geometry only, which is the one step that cannot
    #: show where it went.
    site_charges: np.ndarray | None = None
    site_moments: np.ndarray | None = None


def site_magnetization(result) -> tuple:
    """``(site_charges, site_moments)`` off an :class:`SCFResult`, as arrays.

    ``(None, None)`` when the run carried no magnetization. One helper for the
    three relaxation drivers, which record the same pair at each of their own
    kinds of step.
    """
    if getattr(result, "site_moments", None) is None:
        return None, None
    return np.asarray(result.site_charges), np.asarray(result.site_moments)


def site_moment_report(site_moments) -> str:
    """``   |m|_site = a..b``, or ``""`` -- the SCF console line's own suffix.

    The smallest and the largest site moment, in the format
    :func:`~defumat.scf.driver.run_scf` prints per iteration, so that a
    relaxation's log line and its inner SCF's read the same way. The *pair* is
    what a partial collapse shows in: one site losing its moment moves the
    minimum and leaves the maximum where it was.
    """
    if site_moments is None:
        return ""
    lengths = np.linalg.norm(np.asarray(site_moments), axis=1)
    return f"   |m|_site = {lengths.min():.4f}..{lengths.max():.4f}"


@dataclass
class RelaxResult:
    """The relaxed structure and the path taken to it."""

    converged: bool
    #: The final geometry, as a :class:`~defumat.system.builder.System` -- the
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

    def plot(self, ax=None, **kwargs):
        """Draw the relaxation's progress, and return the axes.

        Two curves against the ionic step: the total energy relative to the
        final one, in mRy, and the largest force still on any atom, in
        Ry/bohr on a log axis. They answer the two questions asked of a
        relaxation that has finished -- how far downhill it went, and whether
        the force actually reached the threshold or the optimizer simply
        stopped.

        matplotlib is imported here rather than at module scope: it is not a
        dependency of any calculation, and a headless run should not need it.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        steps = [step.index for step in self.steps]
        energies = np.array([step.total_energy for step in self.steps])
        forces = np.array([step.max_force for step in self.steps])
        kwargs.setdefault("marker", "o")
        ax.plot(steps, 1.0e3 * (energies - energies[-1]),
                color="C0", **kwargs)
        ax.set_xlabel("ionic step")
        ax.set_ylabel(r"$E - E_{\rm final}$   [mRy]", color="C0")
        ax.tick_params(axis="y", labelcolor="C0")
        twin = ax.twinx()
        twin.semilogy(steps, forces, marker="s", ls="--", lw=1.0, color="C3")
        twin.set_ylabel(r"max $|F|$   [Ry/bohr]", color="C3")
        twin.tick_params(axis="y", labelcolor="C3")
        return ax


#: The ``&electrons`` options a relaxation's *inner* SCF wants, and which
#: ``Calculator._defaults_for`` cannot pass unless the entry point names them.
#:
#: Every one of these is adopted from the input file's own ``&electrons``
#: namelist by :func:`~defumat.calculator.electrons_defaults`, and every one was
#: then discarded on the way into a relaxation: ``_defaults_for`` filters
#: strictly by *named parameter*, and a ``**scf_options`` in the signature is
#: not permission to pass everything (there are response entry points that would
#: raise on ``nbnd``). So the three relaxation drivers named ``conv_thr`` and
#: the two mixing knobs, and silently dropped the rest -- every SCF inside a
#: relaxation ran at the default 100 iterations, with loose empty states and no
#: fixed-``ns`` warm-up, whatever the input asked for. On a DFT+U relaxation the
#: dropped ``mixing_fixed_ns`` decides which minimum of the +U functional the
#: run lands in, so the relaxed *geometry* can differ with nothing saying the
#: request was ignored.
#:
#: They are declared ``None`` rather than repeating
#: :func:`~defumat.scf.driver.run_scf`'s own defaults, so that "not given" stays
#: distinguishable from "given the default" and the two cannot drift apart.
SCF_LOOP_OPTIONS = (
    "max_iterations", "david", "diago_full_acc", "mixing_fixed_ns",
    "mixing_ndim", "scf_solver", "scf_solver_options",
)


def _scf_loop_options(scf_options: dict, **named) -> dict:
    """``scf_options`` plus whichever of :data:`SCF_LOOP_OPTIONS` was given.

    An option left at ``None`` is absent from the result, so ``run_scf`` keeps
    deciding it -- the same rule :func:`~defumat.calculator.electrons_defaults`
    follows one layer up.
    """
    given = {name: value for name, value in named.items() if value is not None}
    return {**given, **scf_options}


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
    mixing_beta: float | None = None,
    k_batch: int | None | str = "default",
    density_extrapolation: str = "atomic",
    checkpoint_dir=None,
    resume: bool = True,
    on_step=None,
    verbose: bool = False,
    # See :data:`SCF_LOOP_OPTIONS`. Named rather than left to ``scf_options``
    # so that the facade can forward them.
    max_iterations: int | None = None,
    david: int | None = None,
    diago_full_acc: bool | None = None,
    mixing_fixed_ns: int | None = None,
    mixing_ndim: int | None = None,
    scf_solver: str | None = None,
    scf_solver_options: dict | None = None,
    **scf_options,
) -> RelaxResult:
    """Relax the atomic positions at fixed cell.

    The thresholds mean what they mean in a ``pw.x`` input: ``etot_conv_thr``
    (1e-4 Ry) and ``forc_conv_thr`` (1e-3 Ry/bohr) must *both* be satisfied,
    ``conv_thr`` is the SCF's starting threshold, and ``nstep`` caps the ionic
    steps.

    ``checkpoint_dir`` makes the relaxation restartable. After every ionic step
    the converged state and the **optimizer's own history** are written there,
    and a run started against a directory that already holds them picks up where
    it stopped (``resume=False`` starts over and overwrites). The history is the
    half that is easy to omit and is most of the value: BFGS earns its
    convergence rate from the inverse Hessian and the trust radius, so a resume
    from positions alone takes its next step as if it were the first.

    ``on_step`` is called as ``on_step(step, result, forces)`` after each ionic
    step, for a caller that wants to write its own record beside the checkpoint.

    ``max_iterations``, ``david``, ``diago_full_acc``, ``mixing_fixed_ns``,
    ``mixing_ndim``,
    ``scf_solver`` and ``scf_solver_options`` are the **inner SCF's** options and
    are named here so that :class:`~defumat.calculator.Calculator` can forward
    them -- see :data:`SCF_LOOP_OPTIONS` for why naming them is what it takes.
    ``max_iterations`` is the electronic loop; ``nstep`` is the ionic one.

    **They come from the input file unless given here.** ``None`` -- the
    default -- reads :attr:`System.relax`, which carries what ``&control`` and
    ``&ions`` said or QE's defaults if they said nothing
    (:class:`~defumat.relax.settings.RelaxSettings`). Before that existed
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

    # Checkpoint layout: one state file and one optimizer file, both rewritten
    # in place each step. Two files rather than one because the state is tens of
    # gigabytes and the optimizer is kilobytes, and because a resume wants to
    # know which of the two it is missing.
    checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
    state_path = optimizer_path = step_path = None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state_path = checkpoint_dir / "scf_state.npz"
        optimizer_path = checkpoint_dir / "optimizer.npz"
        step_path = checkpoint_dir / "relax_step.json"

    first_step = 1
    resumed_state = None
    if (checkpoint_dir is not None and resume and step_path.is_file()
            and optimizer_path.is_file()):
        record = json.loads(step_path.read_text())
        load_optimizer(optimizer, optimizer_path)
        moved = np.asarray(record["next_positions"])
        calculation = calculation.at_positions(jnp.asarray(moved))
        first_step = int(record["index"]) + 1
        threshold_resume = float(record["conv_thr"])
        if state_path.is_file():
            # The state belongs to the *previous* geometry, so it is carried as
            # a starting density rather than as an answer -- which is what the
            # loop does with the extrapolated density anyway.
            resumed_state = load_state(state_path, system=calculation.system)
        if verbose:
            print(f"resuming from {checkpoint_dir} at ionic step {first_step}")

    upscale = settings.upscale
    free = system.structure.free
    starting_threshold = conv_thr
    scf_options = _scf_loop_options(
        scf_options, max_iterations=max_iterations, david=david,
        diago_full_acc=diago_full_acc, mixing_fixed_ns=mixing_fixed_ns,
        mixing_ndim=mixing_ndim,
        scf_solver=scf_solver, scf_solver_options=scf_solver_options,
    )
    threshold = conv_thr
    steps: list[RelaxStep] = []
    density = becsum = None
    converged = False
    pools = current_pools()
    if pools.size > 1 and pools.rank != 0:
        verbose = False

    if resumed_state is not None:
        density, becsum = resumed_state.density, resumed_state.becsum
        threshold = threshold_resume

    # The previous step's converged states, as the one entry of a list, or
    # empty: the first step (and a resumed one) starts from atomic orbitals.
    carried = []
    carry = _carries_states(scf_options, pools)
    for index in range(first_step, nstep + 1):
        # **The previous step's mixed state must not be live under this one's
        # SCF.** ``result`` holds the wavefunctions, and rebinding it on the
        # next statement is not enough: the old object stays referenced for the
        # whole of the ``run_scf`` call that is meant to replace it, so two
        # steps' wavefunctions coexist at the peak. Sized at 7.7 GB together
        # with ``previous`` below on a 45-atom slab, which is what makes a
        # relaxation there 40.0 GB where its own SCF is 32.3 on a 39 GB machine
        # (`MEMORY-AUDIT.md` A2). Dropping it costs nothing -- the body rebinds
        # it immediately, ``RelaxStep`` keeps no reference to it, and the
        # post-loop ``RelaxResult`` reads whatever the last iteration left.
        #
        # **The carried states are the exception, and only until they are
        # read.** They are popped into the call, so that ``run_scf``'s
        # parameter is the one reference left, and ``run_scf`` drops that after
        # the first Rayleigh-Ritz, before its first Davidson call. So the
        # previous step's set is alive from here through the first potential to
        # the end of that Rayleigh-Ritz, where it stands in for the atomic
        # orbitals, a set at least as large (``natomwfc`` vectors topped up to
        # ``nbnd``), and the Davidson peak is not raised. The perturbed copy
        # :func:`_randomized` makes is the one moment two sets are alive, and
        # it is over before ``run_scf`` is entered.
        options = _next_step_options(scf_options, carried)
        result = None
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
            starting_wavefunctions=_randomized(carried.pop()) if carried else None,
            verbose=verbose,
            **options,
        )
        if pools.size > 1 and stop_latched():
            # A SIGTERM reached the pools during this step's SCF, which stopped
            # every pool at the same iteration (``parallel.poll_stop``). Its
            # state is not converged, so no force is taken from it and no step:
            # the checkpoint of the previous step stays the one a resume runs.
            if verbose:
                print(f"ionic step {index:3d}   stopped by a SIGTERM before its SCF "
                      f"converged; the last completed step is what a resume "
                      f"continues from")
            break
        forces = compute_forces(calculation, result, method=force_method)
        positions = np.asarray(calculation.system.structure.positions)
        energy, force_values = result.total_energy, forces.forces
        if pools.size > 1:
            # Every pool takes the optimizer's step; rank 0's energy and forces
            # are everyone's, so no pool can move the atoms differently.
            energy, force_values = pools.broadcast((np.asarray(energy),
                                                    np.asarray(force_values)))
            energy = float(energy)

        moved, converged = optimizer.step(
            positions, energy, force_values * free
        )
        charges, moments = site_magnetization(result)
        steps.append(RelaxStep(
            index=index,
            positions=positions,
            total_energy=result.total_energy,
            max_force=forces.max_force,
            scf_iterations=result.iterations,
            conv_thr=threshold,
            energy_error=getattr(optimizer, "energy_error", None),
            gradient_error=getattr(optimizer, "gradient_error", None),
            site_charges=charges,
            site_moments=moments,
        ))
        if verbose:
            print(f"ionic step {index:3d}   E = {result.total_energy:16.8f} Ry"
                  f"   max |F| = {forces.max_force:.6f} Ry/bohr"
                  f"   dE = {optimizer.energy_error:.2e}"
                  f"{site_moment_report(moments)}")
        if on_step is not None:
            on_step(steps[-1], result, forces)
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

        if checkpoint_dir is not None and pools.rank == 0:
            # Written after the step is decided and before it is taken, so what
            # a resume finds is a geometry to *run*, together with the state and
            # the history that produced it. The optimizer goes last: it is the
            # file whose presence says the record is complete.
            #
            # **Under k-point pools rank 0 writes it, without the states.** A
            # resume seeds the next SCF from the density and ``becsum`` alone
            # (above), every pool reads the same file, and a pool's store is a
            # share of the set that no single file here could hold whole; the
            # optimizer and the step are rank 0's, which every pool followed.
            save_state(dataclasses.replace(result, wavefunctions=None)
                       if pools.size > 1 else result, state_path)
            step_path.write_text(json.dumps({
                "index": index,
                "next_positions": np.asarray(moved).tolist(),
                "conv_thr": float(threshold),
                "total_energy": float(result.total_energy),
                "max_force": float(forces.max_force),
            }))
            save_optimizer(optimizer, optimizer_path)
        if checkpoint_dir is not None and pools.size > 1:
            # No pool takes the next step before rank 0's record of this one is
            # on disk, so a job stopped from any rank finds it complete.
            pools.broadcast_flag(True)

        previous = calculation
        calculation = calculation.at_positions(jnp.asarray(moved))
        # **The group to check is the one the run applies, not the one the
        # crystal has.** ``Calculation.symmetries`` is the full detected group
        # whatever the input said, with ``use_symmetry`` beside it as the switch
        # (``scf/driver.py``), and under ``nosym`` nothing this check protects
        # depends on it: ``build_basis`` takes ``fft_fact = (1, 1, 1)`` and the
        # k-set is not reduced. Checking the unused group there refuses a
        # perfectly valid step -- which is what an adsorbate on a surface does
        # at its first displacement, the whole reason such a run sets ``nosym``.
        if calculation.use_symmetry and not check_symmetry(
            calculation.system.cell, calculation.system.structure, calculation.symmetries
        ):
            raise RuntimeError(
                "the ionic step broke a symmetry the run was set up with "
                "(checkallsym): the FFT grid and the k-point set were chosen "
                "for that group and are no longer valid. Symmetrised forces "
                "cannot do this in exact arithmetic, so this is a bug or a "
                "structure whose symmetry was mis-detected. A run that means to "
                "break symmetry wants nosym = .true., which makes this check "
                "vacuous rather than merely quiet"
            )
        density, becsum = _extrapolate(
            previous, calculation, result, density_extrapolation
        )
        if carry and result.wavefunctions is not None:
            carried = [result.wavefunctions]
        # **After** the extrapolation and not before: that call is the last
        # reader of the old ``Calculation``, through ``starting_density()`` and
        # ``becsum(...)``. What it returns is bare arrays closing over nothing,
        # so the step's projectors, augmentation tables and G sets go here.
        del previous

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
