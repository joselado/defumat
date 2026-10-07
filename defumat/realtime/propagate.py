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
from defumat.hamiltonian.noncollinear import SpinorHamiltonian
from defumat.realtime.propagators import get_propagator
from defumat.realtime.pulse import Kick, Sum
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
    #: ``(lower, upper)`` edges of the spectrum of ``H(k + kappa)`` in Ry, and
    #: ``dt_Ry`` times the distance from the centre to the farther one, which the
    #: propagator's bound refused above.
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
    """Refuse, by name, every regime whose missing term is known (``HARMONICS-NEXT.md``).

    Collinear spin, spinors (spin-orbit coupling and noncollinear magnetism)
    and ultrasoft and PAW datasets run (``PLAN.md`` P139 to P141): the first is
    two independent channels at a frozen potential, the second one Hamiltonian
    on ``2 npwx`` whose coupling to the field is the scalar one, and the third
    the generalised equation of motion of :meth:`_Chunk.generator`.
    """
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


def _times_i(x):
    """``i x`` by exchanging the parts, so the dtype is the array's own (``config.py``)."""
    return jax.lax.complex(-x.imag, x.real)


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

    **One Hamiltonian problem, whatever the spin regime.** ``channel`` picks the
    collinear channel of an ``nspin = 2`` run (each is its own band structure
    at a frozen potential); a spinor run has one channel, and its template is
    the :class:`~defumat.hamiltonian.noncollinear.SpinorHamiltonian`, whose
    states are ``2 npwx`` long with both components on the same sphere and the
    same projectors, coupled by the 2x2 ``D`` and, for an augmented dataset,
    the 2x2 ``qq_so``. Every operator below works on components
    ``(..., npol, npwx)`` with ``D`` and ``qq`` as ``(npol, npol, nkb, nkb)``,
    so the collinear case is the spinor one at ``npol = 1``.

    **An augmented dataset** (ultrasoft or PAW) adds an overlap that moves with
    ``kappa`` and the position operator's augmentation tail ``X``
    (:meth:`position`), whose dipole is :attr:`dipole`; ``None`` for a
    norm-conserving one, where neither exists.
    """

    k0: jnp.ndarray
    gcart: jnp.ndarray
    mask: jnp.ndarray
    core: object
    template: object
    table: object
    positions: jnp.ndarray
    nk: int = eqx.field(static=True)
    channel: int = eqx.field(static=True, default=0)
    #: ``dpqq`` as ``(3, nkb, nkb)``, or ``dpqq_so`` as ``(3, 2, 2, nkb, nkb)``
    #: for a spinor (:func:`~defumat.response.efield._augmentation_dipole`).
    dipole: jnp.ndarray | None = None

    @classmethod
    def build(cls, calculation, rows, terms, table, kcart, channel: int = 0, dipole=None):
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
            template=dataclasses.replace(row.hamiltonian_from(terms)[channel], npw=None),
            table=table,
            positions=jnp.asarray(calculation.system.structure.positions),
            nk=len(rows),
            channel=int(channel),
            dipole=dipole,
        )

    # -- the shape of a state ---------------------------------------------

    @property
    def spinor(self) -> bool:
        """Whether the states are two-component spinors, ``2 npwx`` long."""
        return isinstance(self.template, SpinorHamiltonian)

    @property
    def npol(self) -> int:
        return 2 if self.spinor else 1

    @property
    def augmented(self) -> bool:
        """Whether there is an overlap ``S`` other than the identity."""
        return self.template.has_overlap

    def _components(self, x):
        """``(nk, n, npol * npwx)`` as ``(nk, n, npol, npwx)``, the padding zeroed."""
        npwx = self.mask.shape[-1]
        components = x.reshape(x.shape[:-1] + (self.npol, npwx))
        return jnp.where(self.mask[:, None, None, :], components, 0.0)

    @staticmethod
    def _join(components):
        return components.reshape(components.shape[:-2] + (-1,))

    def _d(self):
        """``D`` as ``(npol, npol, nkb, nkb)``: ``deeq_nc`` for a spinor."""
        if self.spinor:
            return self.template.deeq
        return self.template.coefficients[None, None]

    def _q(self):
        """``qq`` as ``(npol, npol, nkb, nkb)``: ``qq_so`` for a spinor; ``None`` without one."""
        if self.spinor:
            return self.template.qq
        qq = self.template.projectors.qq
        return None if qq is None else qq[None, None]

    @staticmethod
    def _project(vkb, components):
        """``<beta_i|x^a>``, ``(nk, n, npol, nkb)``."""
        return jnp.einsum("kgi,knag->knai", jnp.conj(vkb), components)

    @staticmethod
    def _expand(vkb, coefficients):
        """``sum_i |beta_i> c^a_i``, ``(nk, n, npol, npwx)``."""
        return jnp.einsum("kgi,knai->knag", vkb, coefficients)

    @staticmethod
    def _contract(matrix, becp):
        """``sum_bj M^{ab}_ij becp^b_j`` for ``M`` of shape ``(npol, npol, nkb, nkb)``."""
        return jnp.einsum("abij,knbj->knai", matrix.astype(becp.dtype), becp)

    # -- the operators at k + kappa ----------------------------------------

    def moved(self, kappa):
        """``(|k+G+kappa|^2, projectors at k+G+kappa)`` for every k-point of the chunk."""
        kg = self.gcart + (self.k0 + kappa)[:, None, :]
        kinetic = jnp.where(self.mask, jnp.sum(kg * kg, axis=-1), 0.0)
        columns = self.table.columns(kg).astype(self.core.columns.dtype)
        core = eqx.tree_at(lambda c: (c.columns, c.kg), self.core, (columns, kg))
        # ``qq`` rides on the projector set, and the collinear operator's ``S``
        # reads it from there: rebuilt without it, an ultrasoft chunk would
        # propagate with the identity for an overlap and say nothing.
        qq = self.template.projectors.qq
        return kinetic.astype(self.template.kinetic.dtype), core.at_positions(self.positions, qq=qq)

    def projector_derivative(self, kappa, direction):
        """``d vkb/dk`` along ``direction`` about each projector's own atom, ``(nk, npwx, nkb)``.

        The derivative of the table's columns, with the structure factor of
        the atom left where it is, which is what
        :meth:`~defumat.response.velocity.VelocityOperator.projectors` returns
        from the radial transform.
        """
        kg = self.gcart + (self.k0 + kappa)[:, None, :]
        tangent = jnp.broadcast_to(jnp.asarray(direction, dtype=kg.dtype), kg.shape)
        columns = jax.jvp(self.table.columns, (kg,), (tangent,))[1]
        core = eqx.tree_at(lambda c: (c.columns, c.kg), self.core,
                           (columns.astype(self.core.columns.dtype), kg))
        return core.at_positions(self.positions).vkb

    def hamiltonian(self, kappa):
        kinetic, projectors = self.moved(kappa)
        return dataclasses.replace(self.template, kinetic=kinetic, projectors=projectors)

    def kinetic_nonlocal(self, kappa, x):
        """``(T + V_NL)(k + kappa) x`` for ``x`` of shape ``(nk, n, ndim)``, no transform."""
        kinetic, projectors = self.moved(kappa)
        vkb = projectors.vkb
        components = self._components(x)
        becp = self._project(vkb, components)
        out = (kinetic[:, None, None, :].astype(components.dtype) * components
               + self._expand(vkb, self._contract(self._d(), becp)))
        return self._join(jnp.where(self.mask[:, None, None, :], out, 0.0))

    def band_form(self, kappa, c1, c2, weights):
        """``sum_kn w <c1|T + V_NL|c2>`` at ``k + kappa``, complex, Ry; no transform."""
        kinetic, projectors = self.moved(kappa)
        vkb = projectors.vkb
        a, b = self._components(c1), self._components(c2)
        w = weights.astype(a.dtype)
        kin = jnp.einsum("kn,knag,kg,knag->", w, jnp.conj(a), kinetic.astype(a.dtype), b)
        nonlocal_ = jnp.einsum("kn,knai,knai->", w, jnp.conj(self._project(vkb, a)),
                               self._contract(self._d(), self._project(vkb, b)))
        return kin + nonlocal_

    def kappa_energy(self, kappa, psi, weights):
        """The kinetic and nonlocal band energy at ``k + kappa``, Ry; ``psi`` is ``(nk, nbnd, ndim)``."""
        return jnp.real(self.band_form(kappa, psi, psi, weights))

    def overlap(self, kappa, x, projectors=None):
        """``S(k + kappa) x``; ``x`` itself for a norm-conserving dataset."""
        if not self.augmented:
            return x
        if projectors is None:
            _, projectors = self.moved(kappa)
        vkb = projectors.vkb
        components = self._components(x)
        out = components + self._expand(vkb, self._contract(self._q(), self._project(vkb, components)))
        return self._join(jnp.where(self.mask[:, None, None, :], out, 0.0))

    def inverse_overlap(self, kappa, x, projectors=None):
        """``S(k + kappa)^-1 x`` by Woodbury, ``sm1_psi.f90``: an ``npol nkb`` solve per k-point.

        ``S = 1 + B Q B^dag`` gives ``S^-1 = 1 - B Q (1 + B^dag B Q)^-1 B^dag``,
        with ``B`` the projectors on each spinor component and ``Q`` the
        ``qq`` (``qq_so``) matrix of the channels.
        """
        if not self.augmented:
            return x
        if projectors is None:
            _, projectors = self.moved(kappa)
        vkb = projectors.vkb
        components = self._components(x)
        becp = self._project(vkb, components)  # (nk, n, npol, nkb)
        nk, n, npol, nkb = becp.shape
        q = self._q().astype(becp.dtype)
        # (npol nkb)^2 per k-point, the spin index outside the channel one
        q_flat = jnp.transpose(q, (0, 2, 1, 3)).reshape(npol * nkb, npol * nkb)
        gram = jnp.einsum("kgi,kgj->kij", jnp.conj(vkb), vkb)
        gram_flat = jnp.einsum("ab,kij->kaibj", jnp.eye(npol, dtype=gram.dtype),
                               gram).reshape(nk, npol * nkb, npol * nkb)
        matrix = jnp.eye(npol * nkb, dtype=becp.dtype)[None] + gram_flat @ q_flat[None]
        solved = jnp.linalg.solve(matrix, jnp.swapaxes(becp.reshape(nk, n, npol * nkb), 1, 2))
        coefficients = jnp.swapaxes(q_flat[None] @ solved, 1, 2).reshape(nk, n, npol, nkb)
        out = components - self._expand(vkb, coefficients)
        return self._join(jnp.where(self.mask[:, None, None, :], out, 0.0))

    def position(self, kappa, x, direction, projectors=None):
        """``e.X(k + kappa) x``, the augmentation's share of the position operator.

        ``X = r~ - S r = sum_ij |b_i> [dpqq_ij <b_j| + i q_ij <db_j/dk|]``, the
        position operator of the augmented density less ``S`` times the bare
        one, with ``<b_j|(r - R)`` written as ``-i`` times the projector's
        derivative about its own atom. It is ``i`` times the connection
        :meth:`~defumat.response.velocity.VelocityOperator.augmentation_connection`
        and the term ``adddvepsi_us.f90`` adds; ``X - X^dag = i dS/dk``. Zero
        for a norm-conserving dataset.
        """
        if not self.augmented:
            return jnp.zeros_like(x)
        if projectors is None:
            _, projectors = self.moved(kappa)
        vkb = projectors.vkb
        direction = jnp.asarray(direction)
        derivative = self.projector_derivative(kappa, direction)
        components = self._components(x)
        becp = self._project(vkb, components)
        moving = self._project(derivative, components)
        along = jnp.einsum("a,a...->...", direction.astype(self.dipole.dtype), self.dipole)
        if not self.spinor:
            along = along[None, None]
        coefficients = (self._contract(along, becp)
                        + _times_i(self._contract(self._q(), moving)))
        out = self._expand(vkb, coefficients)
        return self._join(jnp.where(self.mask[:, None, None, :], out, 0.0))

    def energy(self, kappa, psi, weights):
        """``sum w <u|H(k+kappa)|u>`` in Ry."""
        ham = self.hamiltonian(kappa)
        hpsi = map_k(lambda ik: ham.apply(psi[ik], ik), jnp.arange(self.nk), batch=None)
        return jnp.real(jnp.einsum("kng,kng,kn->", jnp.conj(psi), hpsi, weights))

    def applied(self, kappa, psi, ham=None):
        """``H(k + kappa) psi`` for the chunk's states, ``(nk, n, ndim)``, one transform a band."""
        if ham is None:
            ham = self.hamiltonian(kappa)
        return map_k(lambda ik: ham.apply(psi[ik], ik), jnp.arange(self.nk), batch=None)

    def generator(self, kappa, rate):
        """``x -> S^-1 [H - kappadot . X] x`` at ``k + kappa``, on ``(nk, n, ndim)``.

        The right-hand side of ``i S dphi/dt = (H - kappadot . X) phi``, the
        velocity gauge's equation of motion for an augmented dataset: the gauge
        factor ``e^{i kappa.r}`` that takes every operator from ``k`` to
        ``k + kappa`` meets the time derivative through ``S``, and leaves
        ``-kappadot . (r~ - S r)`` where ``r~`` is the position operator of the
        augmented density (:meth:`position`). ``rate`` is ``dkappa/dt`` in the
        Rydberg time unit. For a norm-conserving dataset this is ``H``.
        """
        ham = self.hamiltonian(kappa)

        def apply(x):
            out = self.applied(kappa, x, ham)
            if not self.augmented:
                return out
            out = out - self.position(kappa, x, rate, ham.projectors)
            return self.inverse_overlap(kappa, out, ham.projectors)
        return apply

    def current(self, kappa, psi, weights):
        """``sum w <phi|J_a|phi>`` up to ``-1/Omega``, ``(3,)`` in Ry bohr.

        ``<dH/dk_a>`` at frozen states, which is ``jax.grad`` of the kinetic
        and nonlocal band energy, and, for an augmented dataset,
        ``- 2 Im <phi| H S^-1 X_a |phi>``: the velocity of the states
        ``T|phi>`` the augmented density is made of, whose matrix element
        between two eigenstates is
        :meth:`~defumat.response.velocity.VelocityOperator.generalised_matrix_elements`'s
        ``<n|dH - e_m dS|m> + (e_m - e_n) K_nm``. The energy the field does work
        against obeys ``dE/dt = kappadot . current`` along the motion of
        :meth:`generator`, which is the same statement.
        """
        slope = jax.grad(self.kappa_energy)(kappa, psi, weights)
        if not self.augmented:
            return slope
        ham = self.hamiltonian(kappa)
        chi = self.inverse_overlap(kappa, self.applied(kappa, psi, ham), ham.projectors)
        w = weights.astype(psi.dtype)
        tails = []
        for axis in range(3):
            unit = jnp.zeros(3, dtype=kappa.dtype).at[axis].set(1.0)
            moved = self.position(kappa, psi, unit, ham.projectors)
            tails.append(-2.0 * jnp.imag(jnp.einsum("kn,kng,kng->", w, jnp.conj(chi), moved)))
        return slope + jnp.stack(tails).astype(slope.dtype)

    def jump(self, kappa_from, kappa_to, psi, steps: int = 1):
        """The states after ``kappa`` jumps from ``kappa_from`` to ``kappa_to``.

        A step in ``A`` is a delta in the field, under which the all-electron
        state does not move, so the augmented states follow
        ``dphi/dkappa = i S^-1 X phi`` along the jump (``T(kappa) phi`` held
        fixed, ``T^-1 dT/dk = -i S^-1 X``); taken here by the classical
        fourth-order Runge-Kutta rule in ``steps`` steps, which is exact to
        fourth order in the jump, and so for every perturbative order this
        code takes. The identity for a norm-conserving dataset, whose states
        are the all-electron ones.
        """
        if not self.augmented:
            return psi
        delta = (kappa_to - kappa_from) / steps

        def rate(kappa, x):
            return _times_i(self.inverse_overlap(kappa, self.position(kappa, x, delta)))

        for n in range(steps):
            kappa = kappa_from + n * delta
            k1 = rate(kappa, psi)
            k2 = rate(kappa + 0.5 * delta, psi + 0.5 * k1)
            k3 = rate(kappa + 0.5 * delta, psi + 0.5 * k2)
            k4 = rate(kappa + delta, psi + k3)
            psi = psi + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return psi

    def norms(self, kappa, psi):
        """``<u|S(k + kappa)|u>`` for every state, ``(nk, n)``."""
        return jnp.real(jnp.einsum("kng,kng->kn", jnp.conj(psi), self.overlap(kappa, psi)))


def _energies_of(chunk, psi):
    """``<u_n|H(k)|u_n>`` in Ry for every k-point of a chunk, ``(nk, nbnd)``.

    The carried states are ``S``-normalised, so for an augmented dataset this is
    their generalised eigenvalue as it is the ordinary one otherwise.
    """
    ham = chunk.hamiltonian(jnp.zeros(3, dtype=chunk.k0.dtype))
    hpsi = map_k(lambda ik: ham.apply(psi[ik], ik), jnp.arange(chunk.nk), batch=None)
    return jnp.real(jnp.einsum("kng,kng->kn", jnp.conj(psi), hpsi))


def _flat_blocks(matrix):
    """``(npol, npol, nkb, nkb)`` as the ``(npol nkb)^2`` matrix, spin outside the channel."""
    matrix = np.asarray(matrix)
    npol, nkb = matrix.shape[0], matrix.shape[-1]
    return matrix.transpose(0, 2, 1, 3).reshape(npol * nkb, npol * nkb)


def _chunk_spectrum(chunk, psi, terms, kappa_max):
    """``(energies, kinetic_max, nl_min, nl_max, s_min)`` of one chunk, on the host.

    The carried states' energies at ``kappa = 0``, the largest ``|k+G| + kappa_max``
    squared over the chunk's spheres, the extreme eigenvalues of
    ``G^(1/2) D G^(1/2)``, ``G = vkb^dagger vkb``, over its k-points at
    ``kappa = 0``, with ``D`` the 2x2 block matrix of a spinor run on the
    two components' copies of ``G``, and the smallest eigenvalue of ``S``,
    ``1 + min eig G^(1/2) qq G^(1/2)`` (one without an overlap).
    """
    norms = np.linalg.norm(np.asarray(chunk.gcart + chunk.k0[:, None, :]), axis=-1)
    norms = np.where(np.asarray(chunk.mask), norms, 0.0)
    kinetic_max = float((norms.max() + kappa_max) ** 2)
    _, projectors = chunk.moved(jnp.zeros(3, dtype=chunk.k0.dtype))
    vkb = np.asarray(projectors.vkb)
    coefficients = _flat_blocks(chunk._d())
    q = chunk._q()
    overlap = None if q is None else _flat_blocks(q)
    npol = chunk.npol
    nl_min, nl_max, s_min = 0.0, 0.0, 1.0
    for ik in range(vkb.shape[0]):
        gram = vkb[ik].conj().T @ vkb[ik]
        values, vectors = np.linalg.eigh(gram)
        root = (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.conj().T
        root = np.kron(np.eye(npol), root)
        eig = np.linalg.eigvalsh(root @ coefficients @ root)
        nl_min, nl_max = min(nl_min, float(eig.min())), max(nl_max, float(eig.max()))
        if overlap is not None:
            s_min = min(s_min, 1.0 + float(np.linalg.eigvalsh(root @ overlap @ root).min()))
    energies = np.asarray(compiled_function(_energies_of, chunk, psi)(chunk, psi))
    return energies, kinetic_max, nl_min, nl_max, s_min


def _potential_maximum(potential) -> float:
    """The largest eigenvalue of the local potential anywhere on the grid.

    One component for a collinear channel; for a spinor's ``(v, m_x, m_y, m_z)``
    the 2x2 matrix ``v + m.sigma`` at each point, whose larger eigenvalue is
    ``v + |m|``.
    """
    potential = np.asarray(potential)
    if potential.ndim == 4 and potential.shape[0] == 4:
        return float((potential[0] + np.linalg.norm(potential[1:], axis=0)).max())
    return float(potential.max())


def spectral_bounds(calculation, states, weights, terms, table, kcart, chunks,
                    kappa_max: float, channel: int = 0, dipole=None):
    """``(lower, upper, centre, carried)`` in Ry, over **every** k-point of the run.

    ``upper`` is Weyl's inequality on the three terms of ``H(k + kappa)`` for
    ``|kappa| <= kappa_max``: the largest kinetic energy ``(|k+G| + kappa_max)^2``
    on any sphere, the largest value of the local potential on the smooth grid,
    and the largest eigenvalue of the nonlocal term ``G^(1/2) D G^(1/2)`` at
    ``kappa = 0``, widened by ten per cent for the shift. With an overlap the
    spectrum is the generalised one, ``H x = e S x``, so a positive edge is
    divided by the smallest eigenvalue of ``S`` (again ten per cent wider).
    ``lower`` is the lowest
    carried energy less the largest drop the kinetic energy can take under the
    shift, ``2 |k+G| kappa_max + kappa_max^2``, and one Rydberg: the carried states
    are the bottom of each k-point's spectrum, which Weyl's bound on the same
    three terms (the local potential's minimum, about -14 Ry on silicon) places
    ten times too low. It is an estimate rather than a bound, and the run checks
    it as it goes, by refusing to continue if a norm grows. ``centre`` is the
    carried states' weighted mean energy and ``carried`` their ``(min, max)``.

    All of it over every chunk: built on the first chunk alone, as it once was,
    the centre and the refusal depended on the chunk size, which moved the
    current by 3.9e-7 of its size between ``k_batch`` 1 and 8 on two-atom
    silicon (found in review).
    """
    energies = np.zeros(states.shape[:2])
    kinetic_max, nl_min, nl_max, s_min = 0.0, 0.0, 0.0, 1.0
    radius = 0.0
    for rows, live in chunks:
        chunk = _Chunk.build(calculation, rows, terms, table, kcart, channel, dipole)
        values, kin, low, high, smallest = _chunk_spectrum(
            chunk, jnp.asarray(states[rows]), terms, kappa_max)
        energies[rows[:live]] = values[:live]
        kinetic_max = max(kinetic_max, kin)
        radius = max(radius, math.sqrt(kin) - kappa_max)
        nl_min, nl_max = min(nl_min, low), max(nl_max, high)
        s_min = min(s_min, smallest)
    potential = terms.potentials[0 if len(terms.potentials) == 1 else channel]
    upper = kinetic_max + _potential_maximum(potential) + 1.1 * nl_max
    if s_min < 1.0 and upper > 0.0:
        upper = upper / (0.9 * s_min)
    drop = 2.0 * radius * kappa_max + kappa_max**2
    lower = float(energies.min()) - drop - 1.0
    total = max(float(np.sum(weights)), 1e-300)
    centre = float(np.sum(weights * energies)) / total
    return lower, upper, centre, (float(energies.min()), float(energies.max()))


def _reach(lower, upper, centre):
    """How far the spectrum reaches from the centre, the half-width the step must be stable for."""
    return max(upper - centre, centre - lower)


def largest_stable_step(calculation, states, weights, v_scf, kappa_max: float = 0.0,
                        propagator: str = "taylor4", k_batch="default", kcart=None,
                        ddd_paw=None) -> float:
    """The largest ``dt`` in Hartree atomic units the propagator is stable for here.

    ``2 bound / reach`` with ``reach`` the distance from the centre of the step
    (the carried energies) to the farther edge of the spectrum
    (:func:`spectral_bounds`), which is what :func:`propagate` refuses a step
    against; the smallest over the channels of an ``nspin = 2`` run. A caller
    that has a period to divide
    (:func:`~defumat.workflows.realtime.run_harmonic_orders`) reads it to choose
    its steps rather than be refused.
    """
    steps = []
    for channel, (own, carried) in enumerate(channel_arrays(calculation, states, weights)):
        setup = _prepare(calculation, own, carried, v_scf, kappa_max, None, propagator,
                         k_batch, kcart, channel=channel, ddd_paw=ddd_paw)
        steps.append(2.0 * setup.bound / _reach(setup.lower, setup.upper, setup.centre))
    return min(steps)


def _batch(calculation, k_batch):
    """The k-chunk to walk: the calculation's own for ``'default'`` and ``'fit'``.

    ``'fit'`` is sized from the card where a calculation is built, so a run
    handed one reads the calculation's resolved chunk rather than resolving the
    string again, which :func:`~defumat.batching.resolve_k_batch` refuses.
    """
    from defumat.batching import resolve_k_batch

    if isinstance(k_batch, str) and k_batch in ("default", "fit"):
        return calculation.k_batch
    return resolve_k_batch(k_batch)


def _checkpoint_signature(*parts) -> str:
    """A digest of everything that decides the result of a run.

    The time grid and the field sampled on it (so a pulse of another amplitude
    or shape at the same length is another run), the propagator, the block
    length, the chunk boundaries, the centre of the step, and the bytes of the
    states, the weights and the k-points. A file whose digest differs is
    ignored: a resume that found the same grid and k-count used to return a run
    at another amplitude bit for bit (found in review, 43 per cent off).
    """
    import hashlib

    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            digest.update(str(part.shape).encode())
            digest.update(np.ascontiguousarray(part).tobytes())
        else:
            digest.update(repr(part).encode())
    return digest.hexdigest()


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
    carried: tuple
    dt_ry: float
    volume: float
    effective_cutoff: float
    real: object
    kcart: np.ndarray
    calculation: object = None
    channel: int = 0
    dipole: object = None

    def chunk(self, index: int, rows):
        """The ``index``-th k-chunk's :class:`_Chunk`, the first one already built."""
        if index == 0:
            return self.first
        return _Chunk.build(self.calculation, rows, self.terms, self.table, self.kcart,
                            self.channel, self.dipole)


def _prepare(calculation, states, weights, v_scf, kappa_max, dt, propagator,
             k_batch, kcart, bounds: bool = True, channel: int = 0,
             ddd_paw=None) -> _Setup:
    """The table, the k-chunks, the spectrum and the centre; refuses an unstable step.

    ``dt = None`` skips the refusal, for :func:`largest_stable_step`;
    ``bounds = False`` skips the spectrum as well, a pass over every k-point
    that the frequency-domain hierarchy, which takes no time step, has no use
    for. ``channel`` is the collinear spin channel the states belong to and
    ``ddd_paw`` PAW's one-centre ``D``, frozen with the potential.
    """
    from defumat.response.efield import _augmentation_dipole

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

    terms = calculation.local_terms(v_scf, ddd_paw)
    dipole = _augmentation_dipole(calculation)
    if dipole is not None:
        dipole = dipole.astype(cell.precision.complex)
    planewaves = calculation.basis.planewaves
    gnorm = np.linalg.norm(
        np.asarray(calculation.basis.smooth.cartesian(cell))[np.asarray(planewaves.indices)]
        + kcart[:, None, :], axis=-1)
    radius = float(np.max(np.where(np.asarray(planewaves.mask), gnorm, 0.0)))
    table = radial_table(calculation.pseudos, volume, (radius + kappa_max + 0.5) ** 2)
    # The table runs in the dtype policy's real type: built in float64 from the
    # transform, it ran the recurrence in double precision inside a
    # single-precision step (found in review), which a card pays 1/70 for.
    table = eqx.tree_at(lambda t: (t.coefficients, t.s_max), table,
                        (table.coefficients.astype(cell.precision.real),
                         table.s_max.astype(cell.precision.real)))
    effective_cutoff = max(0.0, radius - kappa_max) ** 2

    chunks = list(k_chunks(nk, _batch(calculation, k_batch)))
    first = _Chunk.build(calculation, chunks[0][0], terms, table, kcart, channel, dipole)
    if bounds:
        lower, upper, centre, carried = spectral_bounds(
            calculation, states, weights, terms, table, kcart, chunks, kappa_max,
            channel, dipole)
    else:
        lower = upper = centre = float("nan")
        carried = (float("nan"), float("nan"))
    # **The centre is the carried states' mean energy and is never moved.** The
    # step's error on a component at energy e goes as (dt (e - centre))^5 in
    # phase and ^6 in norm, and the states live at the bottom of a one-sided
    # spectrum. Centred at the middle of the spectrum the occupied components
    # sat 2 to 3 Ry from it and the norm drifted 1.6e-5 in 200 steps of 0.1 on
    # silicon at 12 Ry, against 3.6e-9 on the band energies. And a centre moved
    # off the bands to keep a long step stable is worse than a refusal: on AlAs
    # at 12 Ry a step of 0.305, inside the old half-width bound, put it 9.8 Ry
    # above the bands, damped every occupied state by 0.925 to 0.954 a step and
    # took the first-order current from 9.5e-9 to 2e-20 in 800 steps (found by
    # the second-order comparison against get_shg); on silicon a clamp between
    # the two bounds lost 4.6 per cent of the norm over a run and 2 to 3 per cent
    # of chi^(3). So the step is refused unless the farther edge of the spectrum
    # is within reach of the centre itself.
    dt_ry = 0.0 if dt is None else 0.5 * float(dt)
    reach = _reach(lower, upper, centre)
    if dt is not None and dt_ry * reach > bound:
        raise ValueError(
            f"the time step dt = {dt} (Hartree a.u.) is past the stability bound of "
            f"the {propagator!r} propagator: the spectrum of H(k + kappa) reaches "
            f"from {lower:.2f} to {upper:.2f} Ry and the step is centred on the "
            f"carried states at {centre:.3f} Ry, so dt_Ry times the farther "
            f"distance is {dt_ry * reach:.3f} against {bound:.3f}. A Taylor "
            f"instability grows exponentially and reads as physics for the first "
            f"thousand steps; use dt <= {2.0 * bound / reach:.4f}")
    return _Setup(terms=terms, table=table, chunks=chunks, first=first, step_fn=step_fn,
                  bound=bound, lower=lower, upper=upper, centre=centre, carried=carried,
                  dt_ry=dt_ry, volume=volume, effective_cutoff=effective_cutoff,
                  real=cell.precision.real, kcart=kcart, calculation=calculation,
                  channel=channel, dipole=dipole)


def _check_growth(norms, live, where: str):
    """Refuse to go on if a norm has grown: a Taylor step past its region is unitary nowhere.

    ``norms`` are ``<u|S|u>`` at the current ``kappa`` (:meth:`_Chunk.norms`),
    ``<u|u>`` for a norm-conserving dataset.
    """
    norms = np.asarray(norms)[:live]
    if norms.max() > 1.0 + 1e-6:
        raise FloatingPointError(
            f"a state's norm grew to {norms.max():.8f} {where}: the step is unstable "
            "for part of the spectrum the estimate of its lower edge missed. Halve dt")
    return norms


def _block_function(step_fn, centre):
    """``block(chunk, w, psi, kappa_mid, kappa_end, rate, steps) -> (psi, currents)``, one ``lax.scan``.

    Each step applies the propagator at ``H(k + kappa_mid)`` and records the
    current at ``kappa_end`` on the stepped states (:meth:`_Chunk.current`,
    ``(nsteps, 3)`` in Ry bohr). For an augmented dataset the step's generator
    is :meth:`_Chunk.generator`, which needs ``rate``, ``dkappa/dt`` at the
    midpoint in the Rydberg time unit; a norm-conserving one never reads it. A
    step of length zero is the identity, which is how a block is padded to one
    shape. Everything with a k index arrives as an argument (``chunk``, ``w``),
    so one kept program serves every chunk; only the propagator and the
    centre, the same for the whole run, are closed over.
    """
    # an array rather than a float, so that it is a constant of the kept
    # program and not a literal in it: a run from other states, or a second
    # run, then reuses the program
    centre = jnp.asarray(centre)

    def block(chunk, w, psi, kmid, kend, rate, steps):
        def body(state, x):
            k_mid, k_end, k_rate, step = x
            if chunk.augmented:
                moved = step_fn(chunk.generator(k_mid, k_rate), state, step, centre)
            else:
                ham = chunk.hamiltonian(k_mid)
                moved = map_k(
                    lambda ik: step_fn(lambda v: ham.apply(v, ik), state[ik], step, centre),
                    jnp.arange(chunk.nk), batch=None)
            return moved, chunk.current(k_end, moved, w)
        return jax.lax.scan(body, psi, (kmid, kend, rate, steps))
    return block


def _energy_of(chunk, w, psi, kappa):
    return chunk.energy(kappa, psi, w)


def _gradient_of(chunk, w, psi, kappa):
    return chunk.current(kappa, psi, w)


def _norms_of(chunk, psi, kappa):
    return chunk.norms(kappa, psi)


def _overlapped_of(chunk, psi, kappa):
    return chunk.overlap(kappa, psi)


def _jumped(chunk, psi, kappa_from, kappa_to):
    return chunk.jump(kappa_from, kappa_to, psi)


def kappa_rate(kappa_t, dt_ry: float) -> np.ndarray:
    """``dkappa/dt`` at the midpoints, ``(kappa(t + dt) - kappa(t))/dt`` in the Rydberg time unit.

    The difference on the run's own grid rather than the pulse's field, which
    is unit-free and second order at the midpoint as the step is; the
    augmented equation of motion reads it (:meth:`_Chunk.generator`). A wrong
    factor here shows as a drift of ``<phi|S|phi>``, since the term's
    anti-Hermitian part has to cancel ``dS/dt`` exactly.
    """
    kappa_t = np.asarray(kappa_t, dtype=float)
    if dt_ry == 0.0:
        return np.zeros_like(kappa_t[1:])
    return (kappa_t[1:] - kappa_t[:-1]) / float(dt_ry)


def require_a_smooth_field(calculation, pulse, start: float) -> None:
    """Refuse a step in ``kappa`` after the start for an augmented dataset.

    A step is a delta in the field that the time grid cannot hold, and the
    augmented states have to cross it with :meth:`_Chunk.jump`; the one at the
    start is crossed that way, a later one would be missed. A norm-conserving
    run crosses any step unchanged, since its states are the all-electron ones.
    """
    if getattr(calculation, "augmentation", None) is None:
        return
    pulses = pulse.pulses if isinstance(pulse, Sum) else (pulse,)
    for part in pulses:
        if isinstance(part, Kick) and float(part.time) > float(start) + 1e-12:
            raise NotImplementedError(
                "a kick after the start of the run is not implemented for an "
                "ultrasoft or PAW dataset: its step in kappa is a delta in the field "
                "that the augmented states cross by a transformation of their own "
                "(the jump at the start does it), and the time grid cannot hold a "
                "delta. Start the run at the kick")


def _chunk_weights(weights, rows, live, real):
    """The chunk's weights, zero on the rows that only pad it to one shape."""
    w = np.where(np.arange(len(rows))[:, None] < live, weights[rows], 0.0)
    return jnp.asarray(w, dtype=real)


def _padded_grid(kappa_mid, kappa_end, nsteps, block_steps, dt_ry):
    """``(kappa_mid, kappa_end, steps, nblocks)`` padded to whole blocks with zero steps."""
    nblocks = int(math.ceil(nsteps / block_steps))
    steps = np.concatenate([np.full(nsteps, dt_ry), np.zeros(nblocks * block_steps - nsteps)])
    return (_padded(kappa_mid, nsteps, block_steps), _padded(kappa_end, nsteps, block_steps),
            steps, nblocks)


def _padded(a, nsteps, block_steps):
    """``a`` padded to whole blocks by repeating its last row."""
    pad = int(math.ceil(nsteps / block_steps)) * block_steps - nsteps
    return np.concatenate([a, np.repeat(a[-1:], pad, axis=0)]) if pad else a


def _warn_damping(setup, nsteps: int, tolerance: float = 1e-4) -> None:
    """Warn when the step will lose more than ``tolerance`` of a carried state's norm.

    The fourth-order Taylor step multiplies a component at ``y = dt_Ry (e -
    centre)`` by ``|R(iy)|`` with ``1 - |R|^2 = y^6/72 - y^8/576``, so the carried
    band farthest from the centre loses about ``nsteps y^6/144`` of its norm over
    the run. It is a loss of accuracy and not an instability, and it reads as a
    response that fades.
    """
    import warnings

    low, high = setup.carried
    y = setup.dt_ry * max(abs(high - setup.centre), abs(setup.centre - low))
    loss = nsteps * (y**6 / 72.0 - y**8 / 576.0) / 2.0
    if loss > tolerance:
        warnings.warn(
            f"the step loses about {loss:.1e} of the norm of the carried state "
            f"farthest from the centre over {nsteps} steps (dt_Ry (e - centre) = "
            f"{y:.3f}); the response fades by as much. Halve dt", RuntimeWarning,
            stacklevel=3)


def channel_arrays(calculation, states, weights):
    """``[(states, weights)]``, one pair per Hamiltonian problem, each cut to its carried bands.

    ``states`` is ``(nk, nbnd, ndim)`` for a run with one channel (``nspin = 1``
    or a spinor) or ``(nchannel, nk, nbnd, ndim)`` with ``weights``
    ``(nchannel, nk, nbnd)``; a collinear ``nspin = 2`` run has two channels,
    whose band counts differ when ``tot_magnetization`` fixes the moment, and
    the shorter one arrives padded with bands of zero weight. Those are cut
    here, so a channel propagates the bands it carries and no others.
    """
    states = np.asarray(states)
    weights = np.asarray(weights, dtype=float)
    channels = 1 if getattr(calculation, "noncolin", False) else int(calculation.nspin)
    if states.ndim == 3:
        states, weights = states[None], weights[None]
    if states.shape[0] != channels:
        raise ValueError(
            f"{states.shape[0]} channels of states for a calculation with {channels}: "
            "an nspin = 2 run takes (2, nk, nbnd, npwx), one per spin channel")
    pairs = []
    for c in range(channels):
        carrying = np.flatnonzero(np.any(weights[c] != 0.0, axis=0))
        keep = int(carrying[-1]) + 1 if len(carrying) else 0
        pairs.append((states[c][:, :keep], weights[c][:, :keep]))
    return pairs


def _combined(results: list) -> "RealTimeResult":
    """The channels' runs as one: the currents and energies add, the drifts are the worst."""
    if len(results) == 1:
        return results[0]
    first = results[0]
    return dataclasses.replace(
        first,
        current=sum(np.asarray(r.current) for r in results),
        energy=sum(np.asarray(r.energy) for r in results),
        norm_drift=max(r.norm_drift for r in results),
        excited=sum(r.excited for r in results),
        nelec=sum(r.nelec for r in results),
        spectrum=(min(r.spectrum[0] for r in results), max(r.spectrum[1] for r in results)),
        step_radius=max(r.step_radius for r in results),
        effective_cutoff=min(r.effective_cutoff for r in results),
        extras={**first.extras, "channels": results},
    )


def propagate(calculation, states, weights, v_scf, pulse, *, dt: float,
              duration: float | None = None, start: float | None = None,
              propagator: str = "taylor4", k_batch="default",
              block_steps: int = 400, kcart=None, checkpoint=None,
              symmetrise=None, potential: str = "frozen", corrector: int = 1,
              density_symmetry=None, ddd_paw=None) -> RealTimeResult:
    """Propagate ``states`` under ``pulse`` at the frozen potential ``v_scf``.

    Args:
        calculation: the :class:`~defumat.scf.driver.Calculation` the states
            belong to, on the k-set to be propagated.
        states: ``(nk, nbnd, ndim)``, the occupied states at ``t = start``; for
            a collinear ``nspin = 2`` run ``(2, nk, nbnd, npwx)``, one block per
            channel, padded with zero weights (:func:`channel_arrays`).
        weights: ``(nk, nbnd)`` (``(2, nk, nbnd)``), occupation times k-weight,
            held fixed.
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
        potential: ``'frozen'``, or ``'hartree'`` or ``'hxc'`` for the potential
            updated in time (:mod:`defumat.realtime.selfconsistent`), where the
            k-points are no longer independent and the whole set is resident.
        corrector: with the potential updated, how many times the midpoint
            potential is corrected after it is extrapolated.
        density_symmetry: with the potential updated on a reduced k-set, the
            :class:`~defumat.system.symmetry.Symmetries` of the field's little
            group, which completes the density.
        ddd_paw: PAW's one-centre ``D`` of the ground state, frozen with the
            potential (``Calculation.onecenter``).
    """
    if potential != "frozen":
        from defumat.realtime.selfconsistent import (
            propagate_self_consistent, require_a_potential_mode)

        require_a_potential_mode(calculation, potential, symmetrise, density_symmetry)
        if checkpoint is not None:
            raise NotImplementedError(
                "a checkpoint of a run with the potential updated is not implemented: "
                "the k-points are not independent, so the run would have to save every "
                "state and two potentials rather than the current of the chunks done")
        return propagate_self_consistent(
            calculation, states, weights, v_scf, pulse, dt=dt, duration=duration,
            start=start, propagator=propagator, k_batch=k_batch,
            block_steps=block_steps, kcart=kcart, symmetrise=symmetrise,
            density_symmetry=density_symmetry, potential=potential,
            corrector=corrector)
    pairs = channel_arrays(calculation, states, weights)
    results = []
    for channel, (own, carried) in enumerate(pairs):
        path = checkpoint
        if checkpoint is not None and len(pairs) > 1:
            path = Path(f"{checkpoint}.channel{channel}")
        results.append(_propagate_channel(
            calculation, own, carried, v_scf, pulse, dt=dt, duration=duration,
            start=start, propagator=propagator, k_batch=k_batch,
            block_steps=block_steps, kcart=kcart, checkpoint=path,
            symmetrise=symmetrise, channel=channel, ddd_paw=ddd_paw))
    return _combined(results)


def _propagate_channel(calculation, states, weights, v_scf, pulse, *, dt, duration, start,
                       propagator, k_batch, block_steps, kcart, checkpoint, symmetrise,
                       channel, ddd_paw) -> RealTimeResult:
    """:func:`propagate` at a frozen potential for one channel's states, ``(nk, nbnd, ndim)``."""
    # The states stay where they are, a host array in the frozen mode, and go to
    # the device one chunk at a time: the peak is one chunk whatever the mesh.
    states = np.asarray(states)
    weights = np.asarray(weights, dtype=float)
    times, kappa_t, kappa_mid, efield = time_grid(pulse, dt, duration, start)
    require_a_smooth_field(calculation, pulse, times[0])
    nsteps = len(times) - 1
    kappa_max = float(np.max(np.linalg.norm(np.concatenate([kappa_t, kappa_mid]), axis=-1)))
    setup = _prepare(calculation, states, weights, v_scf, kappa_max, dt, propagator,
                     k_batch, kcart, channel=channel, ddd_paw=ddd_paw)
    chunks, step_fn, centre = setup.chunks, setup.step_fn, setup.centre
    dt_ry, volume, kcart, real = setup.dt_ry, setup.volume, setup.kcart, setup.real
    lower, upper = setup.lower, setup.upper
    effective_cutoff = setup.effective_cutoff
    _warn_damping(setup, nsteps)

    kappa_mid_p, kappa_end_p, dts, nblocks = _padded_grid(
        kappa_mid, kappa_t[1:], nsteps, block_steps, dt_ry)
    rate_p = _padded(kappa_rate(kappa_t, dt_ry), nsteps, block_steps)

    current = np.zeros((nsteps + 1, 3))
    energy_index = [0] + [min((b + 1) * block_steps, nsteps) for b in range(nblocks)]
    energy = np.zeros(len(energy_index))
    norm_drift, excited, done = 0.0, 0.0, 0
    signature = _checkpoint_signature(
        np.asarray(times), np.asarray(kappa_t), np.asarray(kappa_mid), propagator,
        block_steps, np.concatenate([rows for rows, _ in chunks]), centre,
        states, weights, kcart, channel)
    if checkpoint is not None and Path(checkpoint).exists():
        saved = np.load(checkpoint)
        if str(saved["signature"]) == signature:
            current, energy = saved["current"], saved["energy"]
            norm_drift, excited, done = (float(saved["norm_drift"]),
                                         float(saved["excited"]), int(saved["done"]))

    run = measure = slope = norms_of = overlapped = jump = None
    zero = jnp.zeros(3, dtype=real)
    for index, (rows, live) in enumerate(chunks):
        if index < done:
            continue
        chunk = setup.chunk(index, rows)
        w = _chunk_weights(weights, rows, live, real)
        psi = jnp.asarray(states[rows])
        initial = states[rows]
        kappa0 = jnp.asarray(kappa_t[0], dtype=real)
        if run is None:
            args = (chunk, w, psi, jnp.asarray(kappa_mid_p[:block_steps], dtype=real),
                    jnp.asarray(kappa_end_p[:block_steps], dtype=real),
                    jnp.asarray(rate_p[:block_steps], dtype=real),
                    jnp.asarray(dts[:block_steps], dtype=real))
            run = compiled_function(_block_function(step_fn, centre), *args)
            measure = compiled_function(_energy_of, chunk, w, psi, kappa0)
            slope = compiled_function(_gradient_of, chunk, w, psi, kappa0)
            norms_of = compiled_function(_norms_of, chunk, psi, kappa0)
            if chunk.augmented:
                overlapped = compiled_function(_overlapped_of, chunk, psi, kappa0)
                jump = compiled_function(_jumped, chunk, psi, zero, kappa0)
        if jump is not None:
            # the step in kappa a run starts with, from the ground state's zero
            psi = jump(chunk, psi, zero, kappa0)

        current[0] += np.asarray(slope(chunk, w, psi, kappa0))
        energy[0] += float(measure(chunk, w, psi, kappa0))
        for b in range(nblocks):
            sl = slice(b * block_steps, (b + 1) * block_steps)
            psi, gradients = run(chunk, w, psi, jnp.asarray(kappa_mid_p[sl], dtype=real),
                                 jnp.asarray(kappa_end_p[sl], dtype=real),
                                 jnp.asarray(rate_p[sl], dtype=real),
                                 jnp.asarray(dts[sl], dtype=real))
            stop = min((b + 1) * block_steps, nsteps)
            # the padded steps of the last block are the identity and their
            # current is a repeat; it is dropped here
            current[b * block_steps + 1:stop + 1] += np.asarray(gradients)[:stop - b * block_steps]
            kappa_stop = jnp.asarray(kappa_t[stop], dtype=real)
            energy[b + 1] += float(measure(chunk, w, psi, kappa_stop))
            norms = _check_growth(norms_of(chunk, psi, kappa_stop), live,
                                  f"after step {stop} of k-chunk {index}")

        norm_drift = max(norm_drift, float(np.abs(norms - 1.0).max()))
        final = psi if overlapped is None else overlapped(
            chunk, psi, jnp.asarray(kappa_t[-1], dtype=real))
        overlap = np.einsum("kmg,kng->kmn", np.conj(np.asarray(initial)), np.asarray(final))
        kept = np.sum(np.abs(overlap) ** 2, axis=1)  # (nk, nbnd)
        excited += float(np.sum(np.asarray(w) * (1.0 - kept)))
        if checkpoint is not None:
            np.savez(checkpoint, signature=np.asarray(signature), current=current,
                     energy=energy, norm_drift=norm_drift, excited=excited, done=index + 1)

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
        step_radius=dt_ry * _reach(lower, upper, centre),
        symmetry_operations=1 if symmetrise is None else len(symmetrise),
    )
