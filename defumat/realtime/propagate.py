"""The driver: occupied states propagated under ``H(k + kappa(t))`` at a frozen potential.

The time-dependent Kohn-Sham equation for the periodic part of each occupied
state in a uniform field, in the velocity gauge, is

    i d|u_nk>/dt = H(k + kappa(t)) |u_nk>,        kappa(t) = A(t)/c,

with ``H(k)`` the ordinary Bloch Hamiltonian on the plane-wave sphere that was
built for ``k`` and is never rebuilt. For a norm-conserving dataset that is the
whole coupling to the field: the kinetic energy becomes ``|k+G+kappa|^2`` and
the projectors are evaluated at ``k+G+kappa``, since the gauge factor
``exp(-i kappa.R)`` cancels between the two projectors of one atom. Nothing is
expanded in the field and no empty state is needed (``HARMONICS-NEXT.md``,
"The physics").

**The current is the derivative of the band energy with respect to kappa at
frozen states**,

    J(t) = -(1/Omega) sum_nk w_nk <u_nk| dH/dk |u_nk>   at  k + kappa(t),

in Hartree atomic units (a velocity is ``dH_Ha/dk``, half the Rydberg one, and
an electron carries charge -1), with ``w`` the occupation times the k-weight,
summing to the electron count. It is ``jax.grad`` of the kinetic and nonlocal
band energy, the only two terms that carry ``kappa``, so it needs no Fourier
transform; the derivative of ``|k+G+kappa|^2`` is the diamagnetic current and
the derivative of the projectors is the commutator ``[r, V_NL]``, so neither is
added by hand. A transcription of Elk's ``-(1/c) A N`` beside it would count
the diamagnetic term twice.

**The potential is frozen** at the one it is handed, the ground state's, which
is the independent-particle response and is what makes the k-points
independent. So the loop is k outside and time inside: a chunk of k-points is
propagated through the whole pulse, only its weighted share of ``J(t)`` is
kept, and the peak is one chunk whatever the mesh. Inside a chunk the steps
run as a ``lax.scan`` over a block of steps, with ``kappa`` at the midpoints
and at the ends as the scanned arrays, compiled once for every block and every
chunk (:func:`~defumat.eager.compiled_function`); the projectors at ``k +
kappa`` come from the table of :mod:`defumat.realtime.radial` rather than from
the radial transform, which measured five times the cost of a Hamiltonian
application at 12 Ry.

**Units.** The user's times are Hartree atomic units (Elk's ``dtimes``), the
Hamiltonian is in Ry, and the step exponentiates ``H_Ry dt_Ry`` with
``dt_Ry = dt / 2``, the internal time unit being ``hbar/Ry`` = 48.4 as. That is
the one conversion.

**Memory.** Nothing is differentiated in reverse through time, so there is
no tape: the current is one small gradient in ``kappa`` per step, and the
perturbative orders are forward mode. The resident set is the states,
``16 nk nbnd npwx`` bytes for the whole mesh (``2^n`` times that inside
:mod:`defumat.realtime.orders` at order ``n``), and what is in flight is one
k-chunk: the four Taylor terms and their sum, ``5 x 16 nk_chunk nbnd npwx``,
``vkb`` at ``k + kappa`` for the chunk, ``16 nk_chunk npwx nkb``, one FFT box
per band in flight, and the table's Clenshaw recurrence on ``(nk_chunk, npwx,
nbeta)``, which the current's gradient keeps once per term (32 of them). The
current is accumulated over chunks on the host as ``(nt, 3)``, never per
k-point.

**The step** is the fourth-order Taylor expansion of
``exp(-i H(t + dt/2) dt)`` (:mod:`defumat.realtime.propagators`), with the
centre of the spectrum subtracted. The spectrum is bounded rigorously before
the run (:func:`spectral_bounds`) and a step past the propagator's stability
bound is refused by name, since a Taylor instability is an exponential that
looks like physics for the first thousand steps.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from defumat.batching import k_chunks, map_k
from defumat.eager import compiled_function
from defumat.realtime.propagators import get_propagator
from defumat.realtime.radial import radial_table
from defumat.units import AU_SEC

__all__ = ["RealTimeResult", "propagate", "require_a_realtime_regime",
           "spectral_bounds", "time_grid", "largest_stable_step"]


@dataclass
class RealTimeResult:
    """What :func:`propagate` returns. Hartree atomic units throughout.

    Attributes:
        times: ``(nt,)``, Hartree atomic units of time (24.189 as), from the
            start of the run; ``J`` and ``kappa`` are on these.
        kappa: ``(nt, 3)``, 1/bohr, the field as a crystal-momentum shift.
        efield: ``(nt, 3)``, ``E = -dkappa/dt``. Zero for a kick, whose field is
            a delta at ``t = 0`` of strength ``-kappa``.
        current: ``(nt, 3)``, the macroscopic electric current density,
            ``-(1/Omega) sum w <dH/dk>``, in ``e E_h / (hbar a_0^2)``.
        energy_times: the times the energy was recorded at, the end of each
            block of steps and the start.
        energy: ``(n,)``, the band energy ``sum w <u|H(k+kappa)|u>`` in Hartree,
            at a frozen potential the energy the field has done work against.
        norm_drift: ``max |<u|u> - 1|`` over every state at the end.
        excited: electrons that have left the subspace of the carried states,
            ``sum w (1 - sum_m |<u_m(0)|u_n(T)>|^2)``. For an insulator that
            carries its occupied bands only, this is the number of electrons per
            cell promoted across the gap, and it is a statement about the
            ground state's bands only when ``kappa`` is back at zero at the end.
        volume: the cell volume in bohr^3.
        nelec: ``sum w``, the electrons carried.
        dt: the step, Hartree atomic units.
        effective_cutoff: ``(sqrt(ecutwfc) - kappa_max)^2`` in Ry, the cutoff of
            the frozen sphere along the field at the largest shift
            (``HARMONICS-NEXT.md``, "The frozen sphere under a large shift").
    """

    times: np.ndarray
    kappa: np.ndarray
    efield: np.ndarray
    current: np.ndarray
    energy_times: np.ndarray
    energy: np.ndarray
    norm_drift: float
    excited: float
    volume: float
    nelec: float
    dt: float
    effective_cutoff: float
    propagator: str = "taylor4"
    pulse: object = None
    #: ``(lower, upper)`` bound on the spectrum of ``H(k + kappa)`` in Ry, and
    #: ``dt_Ry`` times its half-width, which the propagator's bound refused above.
    spectrum: tuple = ()
    step_radius: float = float("nan")
    #: Which k-points were propagated and with what weight: the whole mesh, or
    #: the wedge of the field's little group with its operations.
    symmetry_operations: int = 1
    extras: dict = field(default_factory=dict)

    @property
    def times_fs(self) -> np.ndarray:
        """The time axis in femtoseconds."""
        return np.asarray(self.times) * AU_SEC * 1.0e15

    @property
    def energy_gained(self) -> float:
        """``E(T) - E(0)`` in Hartree per cell."""
        return float(self.energy[-1] - self.energy[0])

    @property
    def work(self) -> float:
        """``Omega int J.E dt`` in Hartree per cell, the work the field did.

        For a unitary evolution under the Hamiltonian the energy is measured
        with this equals :attr:`energy_gained`; the trapezoid rule on the
        recorded grid is the only error, of order ``dt^2``. A kick has no
        ``E`` on the grid and its work is the jump at ``t = 0``, which this
        does not see.
        """
        power = np.sum(np.asarray(self.current) * np.asarray(self.efield), axis=-1)
        return float(self.volume * np.trapezoid(power, self.times))


def require_a_realtime_regime(calculation) -> None:
    """Refuse, by name, every regime whose missing term is known (``HARMONICS-NEXT.md``)."""
    if calculation.augmentation is not None or calculation.is_paw:
        raise NotImplementedError(
            "real-time propagation with an ultrasoft or PAW dataset is not "
            "implemented: with the projectors at k + kappa(t) the overlap S moves "
            "with time, and the equation of motion gains a term, "
            "i S dpsi/dt = (H + P) psi with P built from the augmentation dipole "
            "and the projectors' kappa derivative, which is derived and not "
            "sourced (HARMONICS-NEXT.md, 'What is refused'). Use a "
            "norm-conserving dataset")
    if calculation.spiral:
        raise NotImplementedError(
            "real-time propagation of a spin spiral is not implemented: the two "
            "spinor components sit on spheres at k + q/2 and k - q/2")
    if calculation.gamma_only:
        raise NotImplementedError(
            "real-time propagation with gamma-only storage is not implemented: "
            "H(k + kappa) is not real at kappa != 0, so half of each (G, -G) "
            "pair does not describe the state; run the cell with an explicit "
            "k = 0 (K_POINTS automatic 1 1 1 0 0 0)")
    if calculation.nspin != 1:
        raise NotImplementedError(
            "real-time propagation is implemented for nspin = 1 only: a collinear "
            "or a spinor run is two channels or one spinor Hamiltonian with no "
            "new physics, and neither has a linear-response reference here to be "
            "checked against (optical_conductivity and chi_0 both refuse "
            "collinear spin), so both are refused for lack of a number")
    if calculation.is_hubbard:
        raise NotImplementedError(
            "real-time propagation with DFT+U is not implemented: at a frozen "
            "occupation matrix the Hubbard projectors move with k + kappa as "
            "the beta functions do, and that mode is refused until its linear "
            "check against the optical conductivity is run")
    if getattr(calculation, "magnetic_field", None) is not None:
        raise NotImplementedError(
            "real-time propagation of a ground state converged under a magnetic "
            "field or a constrained moment is not implemented, for the reason the "
            "response stack gives: the potential is rebuilt from the input field, "
            "which reducebf and the fixed-spin-moment scheme make wrong")


def time_grid(pulse, dt: float, duration: float | None, start: float | None = None):
    """``(times, kappa_at_times, kappa_at_midpoints, efield)`` on the run's grid.

    ``times`` has ``nsteps + 1`` entries, ``t_n = start + n dt``; the midpoints are
    ``t_n + dt/2`` for ``n < nsteps``. ``duration`` defaults to the pulse's own.
    """
    if start is None:
        start = float(pulse.start)
    if duration is None:
        duration = pulse.natural_duration
        if duration is None:
            raise ValueError(
                "this pulse has no length of its own (a kick, a ramp): pass duration")
    nsteps = int(math.ceil(float(duration) / float(dt) - 1e-9))
    times = start + dt * np.arange(nsteps + 1)
    midpoints = times[:-1] + 0.5 * dt
    return times, pulse.kappa(times), pulse.kappa(midpoints), pulse.efield(times)


class _Chunk(eqx.Module):
    """``H(k + kappa)`` for a few k-points, with the projectors from the table.

    Built once per k-chunk on the host and **passed as an argument** to the
    compiled block, never closed over: a program kept by
    :func:`~defumat.eager.compiled_function` replays the arrays it closed over
    when it was traced, so a block closing over one chunk and called on the
    next returns the first chunk's current (measured in review, wrong by 1.2e-4
    with nothing to say so). As a pytree argument every chunk of one shape
    shares one program and one trace. ``at_kcart`` would do the same arithmetic
    with the radial transform and a copy of the whole calculation per step.
    """

    k0: jnp.ndarray
    gcart: jnp.ndarray
    mask: jnp.ndarray
    core: object
    template: object
    table: object
    positions: jnp.ndarray
    nk: int = eqx.field(static=True)

    @classmethod
    def build(cls, calculation, rows, terms, table, kcart):
        row = calculation.at_rows(rows)
        cell = calculation.system.cell
        indices = row.basis.planewaves.indices
        return cls(
            k0=jnp.asarray(np.asarray(kcart)[rows]),
            gcart=calculation.basis.smooth.cartesian(cell)[indices],
            mask=row.basis.planewaves.mask,
            core=row.projector_core,
            # ``npw`` is the eigensolver's cap and nothing else reads it; it is
            # static and is each chunk's own smallest sphere, so leaving it would
            # give every chunk its own tree and send each block through a retrace
            template=dataclasses.replace(row.hamiltonian_from(terms)[0], npw=None),
            table=table,
            positions=jnp.asarray(calculation.system.structure.positions),
            nk=len(rows),
        )

    def moved(self, kappa):
        """``(|k+G+kappa|^2, projectors at k+G+kappa)`` for every k-point of the chunk."""
        kg = self.gcart + (self.k0 + kappa)[:, None, :]
        kinetic = jnp.where(self.mask, jnp.sum(kg * kg, axis=-1), 0.0)
        columns = self.table.columns(kg).astype(self.core.columns.dtype)
        core = eqx.tree_at(lambda c: (c.columns, c.kg), self.core, (columns, kg))
        return kinetic.astype(self.template.kinetic.dtype), core.at_positions(self.positions)

    def hamiltonian(self, kappa):
        kinetic, projectors = self.moved(kappa)
        return dataclasses.replace(self.template, kinetic=kinetic, projectors=projectors)

    def kappa_energy(self, kappa, psi, weights):
        """The kinetic and nonlocal band energy at ``k + kappa``, Ry; ``psi`` is ``(nk, nbnd, npwx)``."""
        kinetic, projectors = self.moved(kappa)
        density = jnp.real(jnp.conj(psi) * psi)
        kin = jnp.einsum("kg,kng,kn->", kinetic, density, weights)
        becp = jnp.einsum("kgi,kng->kni", jnp.conj(projectors.vkb), psi)
        coefficients = self.template.coefficients
        nonlocal_ = jnp.real(jnp.einsum("kni,ij,knj,kn->", jnp.conj(becp),
                                        coefficients.astype(becp.dtype), becp, weights))
        return kin + nonlocal_

    def energy(self, kappa, psi, weights):
        """``sum w <u|H(k+kappa)|u>`` in Ry."""
        ham = self.hamiltonian(kappa)
        hpsi = map_k(lambda ik: ham.apply(psi[ik], ik), jnp.arange(self.nk), batch=None)
        return jnp.real(jnp.einsum("kng,kng,kn->", jnp.conj(psi), hpsi, weights))


def _band_energies(chunk: "_Chunk", states, terms):
    """``<u_n|H(k)|u_n>`` in Ry on the first chunk's rows, ``(nk, nbnd)``, zero elsewhere."""
    ham = chunk.hamiltonian(jnp.zeros(3))
    rows = min(chunk.nk, states.shape[0])
    values = np.zeros(states.shape[:2])
    for ik in range(rows):
        hpsi = ham.apply(states[ik], ik)
        values[ik] = np.real(np.einsum("ng,ng->n", np.conj(np.asarray(states[ik])),
                                       np.asarray(hpsi)))
    # the other rows take the first chunk's mean, which is what a centre needs
    mean = values[:rows].mean(axis=0)
    values[rows:] = mean
    return values


def spectral_bounds(calculation, chunk: _Chunk, terms, kappa_max: float):
    """``(lower, upper)`` in Ry bounding the spectrum of ``H(k + kappa)`` for ``|kappa| <= kappa_max``.

    Weyl's inequality on the three terms: the kinetic energy lies in
    ``[0, max (|k+G| + kappa_max)^2]``, the local potential between its
    extremes on the smooth grid, and the nonlocal term between the extreme
    eigenvalues of ``G^(1/2) D G^(1/2)`` with ``G = vkb^dagger vkb`` the Gram
    matrix of the projectors on the sphere, taken at ``kappa = 0`` and widened by
    ten per cent for the shift. Rigorous for the first two and for the third at
    the k-points of the first chunk.
    """
    norms = np.linalg.norm(np.asarray(chunk.gcart + chunk.k0[:, None, :]), axis=-1)
    norms = np.where(np.asarray(chunk.mask), norms, 0.0)
    kinetic_max = float((norms.max() + kappa_max) ** 2)
    potential = np.asarray(terms.potentials[0])
    vmin, vmax = float(potential.min()), float(potential.max())
    _, projectors = chunk.moved(jnp.zeros(3))
    vkb = np.asarray(projectors.vkb)
    nl_min, nl_max = 0.0, 0.0
    coefficients = np.asarray(chunk.template.coefficients)
    for ik in range(vkb.shape[0]):
        gram = vkb[ik].conj().T @ vkb[ik]
        values, vectors = np.linalg.eigh(gram)
        root = (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.conj().T
        eig = np.linalg.eigvalsh(root @ coefficients @ root)
        nl_min, nl_max = min(nl_min, float(eig.min())), max(nl_max, float(eig.max()))
    return vmin + 1.1 * nl_min, kinetic_max + vmax + 1.1 * nl_max


def largest_stable_step(calculation, states, weights, v_scf, kappa_max: float = 0.0,
                        propagator: str = "taylor4", k_batch="default", kcart=None) -> float:
    """The largest ``dt`` in Hartree atomic units the propagator is stable for here.

    ``4 bound / (upper - lower)`` from :func:`spectral_bounds` on the first
    k-chunk, which is what :func:`propagate` refuses a step against. A caller
    that has a period to divide (:func:`~defumat.workflows.realtime.run_harmonic_orders`)
    reads it to choose its steps rather than be refused.
    """
    require_a_realtime_regime(calculation)
    _, bound = get_propagator(propagator)
    cell = calculation.system.cell
    if kcart is None:
        kcart = getattr(calculation, "_kcart", None)
    if kcart is None:
        kcart = calculation.system.kpoints.cartesian(cell)
    kcart = np.asarray(kcart)
    terms = calculation.local_terms(v_scf)
    planewaves = calculation.basis.planewaves
    gnorm = np.linalg.norm(
        np.asarray(calculation.basis.smooth.cartesian(cell))[np.asarray(planewaves.indices)]
        + kcart[:, None, :], axis=-1)
    radius = float(np.max(np.where(np.asarray(planewaves.mask), gnorm, 0.0)))
    table = radial_table(calculation.pseudos, float(cell.volume),
                         (radius + kappa_max + 0.5) ** 2)
    batch = calculation.k_batch if k_batch == "default" else k_batch
    rows, _ = next(iter(k_chunks(states.shape[0], batch)))
    chunk = _Chunk.build(calculation, rows, terms, table, kcart)
    lower, upper = spectral_bounds(calculation, chunk, terms, kappa_max)
    # dt_Ry (upper - lower)/2 <= bound, and dt = 2 dt_Ry in Hartree time
    return 4.0 * bound / (upper - lower)


def _checkpoint_signature(times, kpoints_rows, nbnd):
    return np.asarray([len(times), float(times[0]), float(times[-1]), kpoints_rows, nbnd])


@dataclass
class _Setup:
    """What every propagation needs before its first step, built once."""

    terms: object
    table: object
    chunks: list
    first: object
    step_fn: object
    bound: float
    lower: float
    upper: float
    centre: float
    dt_ry: float
    volume: float
    effective_cutoff: float
    real: object
    kcart: np.ndarray


def _prepare(calculation, states, weights, v_scf, kappa_max, dt, propagator,
             k_batch, kcart) -> _Setup:
    """The table, the k-chunks, the spectral bounds and the centre; refuses an unstable step."""
    require_a_realtime_regime(calculation)
    step_fn, bound = get_propagator(propagator)
    cell = calculation.system.cell
    volume = float(cell.volume)
    if kcart is None:
        kcart = getattr(calculation, "_kcart", None)
    if kcart is None:
        kcart = calculation.system.kpoints.cartesian(cell)
    kcart = np.asarray(kcart)
    nk = states.shape[0]

    terms = calculation.local_terms(v_scf)
    planewaves = calculation.basis.planewaves
    gnorm = np.linalg.norm(
        np.asarray(calculation.basis.smooth.cartesian(cell))[np.asarray(planewaves.indices)]
        + kcart[:, None, :], axis=-1)
    radius = float(np.max(np.where(np.asarray(planewaves.mask), gnorm, 0.0)))
    table = radial_table(calculation.pseudos, volume, (radius + kappa_max + 0.5) ** 2)
    effective_cutoff = max(0.0, radius - kappa_max) ** 2

    batch = calculation.k_batch if k_batch == "default" else k_batch
    chunks = list(k_chunks(nk, batch))
    first = _Chunk.build(calculation, chunks[0][0], terms, table, kcart)
    lower, upper = spectral_bounds(calculation, first, terms, kappa_max)
    half = 0.5 * (upper - lower)
    dt_ry = 0.5 * float(dt)
    if dt_ry * half > bound:
        raise ValueError(
            f"the time step dt = {dt} (Hartree a.u.) is past the stability bound of "
            f"the {propagator!r} propagator: the spectrum of H(k + kappa) lies in "
            f"[{lower:.2f}, {upper:.2f}] Ry, so dt_Ry times its half-width is "
            f"{dt_ry * half:.3f} against {bound:.3f}. A Taylor instability grows "
            f"exponentially and reads as physics for the first thousand steps; "
            f"use dt <= {2.0 * bound / half:.4f}")
    # **The centre is put where the occupied states are**, as near as stability
    # allows, and not at the middle of the spectrum. The step's error on a
    # component at energy e goes as (dt (e - centre))^5 in phase and ^6 in norm,
    # and the states live at the bottom of a one-sided spectrum: centred at the
    # middle (2.3 Ry on two-atom silicon at 12 Ry) the occupied components sat
    # 2 to 3 Ry from it and the norm drifted 1.6e-5 in 200 steps of 0.1, against
    # 3.6e-9 with the centre on the band energies, which leaves only the small
    # high-energy tail far from it. Any centre within ``bound/dt`` of both ends
    # of the spectrum is stable.
    occupied = float(np.sum(weights * np.asarray(_band_energies(first, states, terms))))
    occupied /= max(float(weights.sum()), 1e-300)
    reach = bound / dt_ry
    centre = min(max(occupied, upper - reach), lower + reach)
    return _Setup(terms=terms, table=table, chunks=chunks, first=first, step_fn=step_fn,
                  bound=bound, lower=lower, upper=upper, centre=centre, dt_ry=dt_ry,
                  volume=volume, effective_cutoff=effective_cutoff,
                  real=cell.precision.real, kcart=kcart)


def _block_function(step_fn, centre):
    """``block(chunk, w, psi, kappa_mid, kappa_end, steps) -> (psi, gradients)``, one ``lax.scan``.

    Each step applies the propagator at ``H(k + kappa_mid)`` and records the
    ``kappa`` gradient of the kinetic and nonlocal band energy at ``kappa_end``
    on the stepped states, ``(nsteps, 3)`` in Ry bohr. A step of length zero is
    the identity, which is how a block is padded to one shape. Everything with
    a k index arrives as an argument (``chunk``, ``w``), so one kept program
    serves every chunk; only the propagator and the centre, the same for the
    whole run, are closed over.
    """
    def block(chunk, w, psi, kmid, kend, steps):
        def body(state, x):
            k_mid, k_end, step = x
            ham = chunk.hamiltonian(k_mid)
            moved = map_k(
                lambda ik: step_fn(lambda v: ham.apply(v, ik), state[ik], step, centre),
                jnp.arange(chunk.nk), batch=None)
            gradient = jax.grad(chunk.kappa_energy)(k_end, moved, w)
            return moved, gradient
        return jax.lax.scan(body, psi, (kmid, kend, steps))
    return block


def _energy_of(chunk, w, psi, kappa):
    return chunk.energy(kappa, psi, w)


def _gradient_of(chunk, w, psi, kappa):
    return jax.grad(chunk.kappa_energy)(kappa, psi, w)


def _chunk_weights(weights, rows, live, real):
    """The chunk's weights, zero on the rows that only pad it to one shape."""
    w = np.where(np.arange(len(rows))[:, None] < live, weights[rows], 0.0)
    return jnp.asarray(w, dtype=real)


def _padded_grid(kappa_mid, kappa_end, nsteps, block_steps, dt_ry):
    """``(kappa_mid, kappa_end, steps, nblocks)`` padded to whole blocks with zero steps."""
    nblocks = int(math.ceil(nsteps / block_steps))
    pad = nblocks * block_steps - nsteps

    def padded_array(a):
        return np.concatenate([a, np.repeat(a[-1:], pad, axis=0)]) if pad else a

    steps = np.concatenate([np.full(nsteps, dt_ry), np.zeros(pad)])
    return padded_array(kappa_mid), padded_array(kappa_end), steps, nblocks


def propagate(calculation, states, weights, v_scf, pulse, *, dt: float,
              duration: float | None = None, start: float | None = None,
              propagator: str = "taylor4", k_batch="default",
              block_steps: int = 400, kcart=None, checkpoint=None,
              symmetrise=None) -> RealTimeResult:
    """Propagate ``states`` under ``pulse`` at the frozen potential ``v_scf``.

    Args:
        calculation: the :class:`~defumat.scf.driver.Calculation` the states
            belong to, on the k-set to be propagated.
        states: ``(nk, nbnd, npwx)``, the occupied states at ``t = start``.
        weights: ``(nk, nbnd)``, occupation times k-weight, held fixed.
        v_scf: the potential to freeze, the ground state's.
        pulse: a :class:`~defumat.realtime.pulse.Pulse`.
        dt: the step in Hartree atomic units of time (Elk's ``dtimes``).
        duration: the length of the run; the pulse's own when not given.
        start: when the run starts; the pulse's own when not given.
        propagator: a name in :mod:`defumat.realtime.propagators`.
        k_batch: how many k-points are propagated together; the calculation's
            dial by default.
        block_steps: steps per compiled block, the unit ``J`` comes back in.
        kcart: ``(nk, 3)`` in 1/bohr where the states' k-points are, when the
            calculation's arrays are not at ``system.kpoints``.
        checkpoint: a path; after each k-chunk the accumulated current and
            energy are written there, and a run that finds a file with the same
            time grid and k-set resumes after the chunks it records.
        symmetrise: ``(nsym, 3, 3)`` cartesian rotations to average the current
            over, a polar vector, when the k-set is the wedge of their group.
    """
    states = jnp.asarray(states)
    weights = np.asarray(weights, dtype=float)
    nk, nbnd, _ = states.shape
    times, kappa_t, kappa_mid, efield = time_grid(pulse, dt, duration, start)
    nsteps = len(times) - 1
    kappa_max = float(np.max(np.linalg.norm(np.concatenate([kappa_t, kappa_mid]), axis=-1)))
    setup = _prepare(calculation, states, weights, v_scf, kappa_max, dt, propagator,
                     k_batch, kcart)
    terms, table, chunks, first = setup.terms, setup.table, setup.chunks, setup.first
    step_fn, centre, dt_ry, volume = setup.step_fn, setup.centre, setup.dt_ry, setup.volume
    kcart, real = setup.kcart, setup.real
    lower, upper = setup.lower, setup.upper
    effective_cutoff = setup.effective_cutoff
    half = 0.5 * (upper - lower)

    kappa_mid_p, kappa_end_p, dts, nblocks = _padded_grid(
        kappa_mid, kappa_t[1:], nsteps, block_steps, dt_ry)

    current = np.zeros((nsteps + 1, 3))
    energy_index = [0] + [min((b + 1) * block_steps, nsteps) for b in range(nblocks)]
    energy = np.zeros(len(energy_index))
    norm_drift, excited, done = 0.0, 0.0, 0
    signature = _checkpoint_signature(times, nk, nbnd)
    if checkpoint is not None and Path(checkpoint).exists():
        saved = np.load(checkpoint)
        if np.array_equal(saved["signature"], signature):
            current, energy = saved["current"], saved["energy"]
            norm_drift, excited, done = (float(saved["norm_drift"]),
                                         float(saved["excited"]), int(saved["done"]))

    run = measure = slope = None
    for index, (rows, live) in enumerate(chunks):
        if index < done:
            continue
        chunk = first if index == 0 else _Chunk.build(calculation, rows, terms, table, kcart)
        w = _chunk_weights(weights, rows, live, real)
        psi = states[jnp.asarray(rows)]
        initial = psi
        kappa0 = jnp.asarray(kappa_t[0], dtype=real)
        if run is None:
            args = (chunk, w, psi, jnp.asarray(kappa_mid_p[:block_steps], dtype=real),
                    jnp.asarray(kappa_end_p[:block_steps], dtype=real),
                    jnp.asarray(dts[:block_steps], dtype=real))
            run = compiled_function(_block_function(step_fn, centre), *args)
            measure = compiled_function(_energy_of, chunk, w, psi, kappa0)
            slope = compiled_function(_gradient_of, chunk, w, psi, kappa0)

        current[0] += np.asarray(slope(chunk, w, psi, kappa0))
        energy[0] += float(measure(chunk, w, psi, kappa0))
        for b in range(nblocks):
            sl = slice(b * block_steps, (b + 1) * block_steps)
            psi, gradients = run(chunk, w, psi, jnp.asarray(kappa_mid_p[sl], dtype=real),
                                 jnp.asarray(kappa_end_p[sl], dtype=real),
                                 jnp.asarray(dts[sl], dtype=real))
            stop = min((b + 1) * block_steps, nsteps)
            # the padded steps of the last block are the identity and their
            # current is a repeat; it is dropped here
            current[b * block_steps + 1:stop + 1] += np.asarray(gradients)[:stop - b * block_steps]
            energy[b + 1] += float(measure(chunk, w, psi, jnp.asarray(kappa_t[stop], dtype=real)))

        norms = np.real(np.einsum("kng,kng->kn", np.conj(np.asarray(psi)), np.asarray(psi)))
        norm_drift = max(norm_drift, float(np.abs(norms[:live] - 1.0).max()))
        overlap = np.einsum("kmg,kng->kmn", np.conj(np.asarray(initial)), np.asarray(psi))
        kept = np.sum(np.abs(overlap) ** 2, axis=1)  # (nk, nbnd)
        excited += float(np.sum(np.asarray(w) * (1.0 - kept)))
        if checkpoint is not None:
            np.savez(checkpoint, signature=signature, current=current, energy=energy,
                     norm_drift=norm_drift, excited=excited, done=index + 1)

    # Ry bohr of band-energy slope to the current density in Hartree units: the
    # velocity is dH_Ha/dk, half the Rydberg one, and the charge is -1.
    current = -current / (2.0 * volume)
    if symmetrise is not None:
        rotations = np.asarray(symmetrise, dtype=float)
        current = np.einsum("sab,tb->ta", rotations, current) / len(rotations)
    return RealTimeResult(
        times=times,
        kappa=kappa_t, efield=efield, current=current,
        energy_times=times[np.asarray(energy_index)], energy=0.5 * energy,
        norm_drift=norm_drift, excited=excited, volume=volume,
        nelec=float(weights.sum()), dt=float(dt), effective_cutoff=effective_cutoff,
        propagator=propagator, pulse=pulse, spectrum=(lower, upper),
        step_radius=dt_ry * half,
        symmetry_operations=1 if symmetrise is None else len(symmetrise),
    )
