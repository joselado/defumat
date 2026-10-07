"""The propagation with the Hartree and exchange-correlation potentials updated in time.

The time-dependent Kohn-Sham equation of :mod:`defumat.realtime.propagate`,
with the local potential rebuilt from the density at every step:

    i d|u_nk>/dt = [T(k + kappa) + V_NL(k + kappa) + vltot + v(t)] |u_nk>,
    v(t) = v_scf + U[rho(t)] - U[rho0],     rho(t) = sum_nk w |u_nk(r)|^2,

with ``U`` the Hartree potential (``potential = 'hartree'``, local fields with
the exchange-correlation potential frozen, the Dyson route's ``rpa``) or the
Hartree plus exchange-correlation potential of the ground state's functional
(``'hxc'``, the adiabatic functional: the ALDA for an LDA run, the Dyson
route's ``alda``). ``rho0`` is the density of the propagated states at the
start (``HARMONICS-NEXT.md``, "Updating the potential in time").

**The difference form is required, not a convenience.** The states come from a
fixed-density solve at the ground state's density on the k-set the run
propagates, and their own density is that density only on the ground state's
grid and to its ``conv_thr``: measured in review on two-atom silicon at
``conv_thr = 1e-10``, ``max |V[rho0] - v_scf|`` is 1.2e-6 Ry on the SCF's own
4x4x4 grid and 1.0e-2 Ry on a 6x6x6 one. Written as ``v(t) = V[rho(t)]`` the
start would not be a fixed point and a run with no field would evolve; as the
difference it is one by construction, and the energy the dynamics conserves is

    E = E_kinNL(kappa) + int (vltot + v_scf - U[rho0]) rho + E_U[rho],

whose derivative in ``rho`` is ``vltot + v(t)``, with ``E_U`` the Hartree
energy or the Hartree and exchange-correlation energies (``E_xc`` of
``rho + rho_core``). With ``U = 0`` it is the band energy of the frozen route.

**The applied field is the whole macroscopic field.** The Hartree potential
has no ``G = 0`` term (``potential.hartree``), so no induced macroscopic field
is fed back, the current over the applied field is the macroscopic
conductivity, and ``1 + 4 pi i sigma / z`` from a kick is ``eps_M`` with local
fields, ``1/[eps^-1]_00``, Elk's ``tddft`` with ``tafindt`` off. The ``G = 0``
part of the exchange-correlation difference is uniform, since ``rho(t)`` keeps
its charge, and is a global phase.

**The loop is inverted.** At a frozen potential the k-points are independent
and the driver walks k outside and time inside. Here every step needs the
density of every k-point, so time is outside and the whole state set is
resident; the k-chunks of the calculation's ``k_batch`` are a ``lax.map``
inside each step, so what is in flight is one chunk's Taylor terms and its
projectors at ``k + kappa``, which are built inside that map (built for the
whole mesh at once they are 16 nk npwx nkb bytes a step). The chunks' arrays
with a k index are stacked on a leading axis and passed to the kept block as
an argument, never closed over (``propagate.py``, ``_Chunk``); the
calculation's grids and the potential's constants are the same for the whole
run and are closed over.

**The step is the midpoint rule with a predictor-corrector for the potential.**
The field is already at the midpoint; the potential there is extrapolated,
``3/2 v(t) - 1/2 v(t - dt)``, the states stepped, the density and potential
rebuilt at ``t + dt``, and with ``corrector = 1`` (the default) the step taken
again from ``t`` at ``(v(t) + v(t + dt))/2``. Both are second order in ``dt``;
the extrapolation misses ``v(t + dt/2)`` by ``-(3/8) dt^2 v''`` and the corrector
by ``+(1/8) dt^2 v''``, so the corrector buys a factor of three in the constant
for twice the cost of a step. Elk's ``tddft.f90`` steps with the potential of
``t``, which is first order.

**The density is symmetrised with the field's little group and nothing else.**
On the wedge of that group (:func:`~defumat.workflows.realtime.field_little_group`)
each k-point stands for its star, since ``S kappa = kappa`` makes
``u_Sk(t) = S u_k(t)``, and the whole mesh's density is the group average of
the wedge's. The calculation's own symmetrisation is the crystal's full group,
and the first-order density of a field along ``x`` is odd under silicon's
inversion, so averaging it over that group removes the whole local-field
effect and ``'hxc'`` would read exactly as ``'frozen'``; which is why
:meth:`~defumat.scf.driver.Calculation.density` and ``finish_density`` are not
called here.

**Memory.** The states of the whole mesh are resident, ``16 nk nbnd npwx``
bytes, twice inside a step (the states at ``t`` and the stepped ones), and
``2^n`` times that at order ``n`` of :func:`propagate_orders_self_consistent`;
beside them, every chunk's k-dependent arrays (kinetic energies, FFT indices
and masks; the projectors at ``kappa = 0`` are dropped, since every step builds
its own), three dense potentials and the density.
"""

from __future__ import annotations

import dataclasses
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from defumat.basis.fft import g_to_r, r_to_g
from defumat.basis.interpolate import to_dense
from defumat.batching import map_k
from defumat.eager import compiled_function

__all__ = ["POTENTIALS", "require_a_potential_mode", "propagate_self_consistent",
           "propagate_orders_self_consistent"]

#: The potential modes, by name: frozen at the ground state's, the Hartree
#: potential updated, the Hartree and exchange-correlation potentials updated.
POTENTIALS = ("frozen", "hartree", "hxc")


def require_a_potential_mode(calculation, potential: str, symmetrise, density_symmetry):
    """Refuse, by name, a mode or a regime the update does not cover."""
    if potential not in POTENTIALS:
        raise ValueError(f"potential must be one of {POTENTIALS}, not {potential!r}")
    if potential == "frozen":
        return
    if getattr(calculation, "augmentation", None) is not None:
        raise NotImplementedError(
            "the potential updated in time is not implemented for an ultrasoft or "
            "PAW dataset: rho(t) needs the augmentation charge from becsum at "
            "k + kappa(t), D(t) has to be rebuilt by newd from v(t) at every step, "
            "and PAW's one-centre D(t) from the one-centre densities. "
            "potential = 'frozen' runs")
    if potential == "hxc" and calculation.functional.is_meta:
        raise NotImplementedError(
            "the exchange-correlation potential updated in time is not implemented "
            "for a meta-GGA: a potential-only one (tb09, bj06) has no energy for the "
            "dynamics to conserve, and either kind needs tau(t) propagated beside "
            "rho(t). potential = 'hartree' keeps the ground state's v_xc and runs")
    if symmetrise is not None and density_symmetry is None:
        raise ValueError(
            "a k-set reduced by symmetry needs the group its density is completed "
            "with when the potential is updated: pass density_symmetry, the field's "
            "little group (workflows.realtime.field_symmetries). The crystal's own "
            "group would average the response away")


class _KPart(eqx.Module):
    """The arrays of one k-chunk that carry a k index, stacked over chunks.

    A :class:`~defumat.realtime.propagate._Chunk` without the table, the
    positions and the local potential, which are the same for every chunk and
    are supplied at the step; the template's potential is a scalar placeholder
    so that the stack does not hold one copy of the grid per chunk.
    """

    k0: jnp.ndarray
    gcart: jnp.ndarray
    mask: jnp.ndarray
    core: object
    template: object
    nk: int = eqx.field(static=True)
    channel: int = eqx.field(static=True, default=0)

    @classmethod
    def of(cls, chunk):
        """The chunk's k-indexed arrays, with what every step rebuilds left out.

        The step builds the projectors and their columns at ``k + kappa`` from
        the table (:meth:`~defumat.realtime.propagate._Chunk.moved`), so the ones
        at ``kappa = 0`` the chunk was built with, ``vkb`` and the real columns
        with their ``k + G``, are never read again; kept for the whole mesh they
        were ``16 nk npwx nkb + 8 nk npwx (ncs + 3)`` bytes resident beside states
        that are ``16 nk npwx nbnd``, several times their size.
        """
        placeholder = jnp.zeros((), dtype=chunk.template.potential.dtype)
        projectors = chunk.template.projectors
        projectors = dataclasses.replace(
            projectors, stored=jnp.zeros((chunk.nk, 1, projectors.dij.shape[0]),
                                         dtype=projectors.dtype))
        template = dataclasses.replace(chunk.template, potential=placeholder,
                                       potential_wave=placeholder, projectors=projectors)
        core = chunk.core
        core = eqx.tree_at(lambda c: (c.columns, c.kg), core,
                           (jnp.zeros((chunk.nk, 1, core.columns.shape[-1]),
                                      dtype=core.columns.dtype),
                            jnp.zeros((chunk.nk, 1, 3), dtype=core.kg.dtype)))
        return cls(k0=chunk.k0, gcart=chunk.gcart, mask=chunk.mask, core=core,
                   template=template, nk=chunk.nk, channel=chunk.channel)

    def chunk(self, table, positions, terms=None):
        from defumat.realtime.propagate import _Chunk

        template = self.template
        if terms is not None:
            # a collinear channel's own potential; a spinor's one, whose
            # components are the charge's and the magnetization's
            index = self.channel if len(terms.potentials) > 1 else 0
            template = dataclasses.replace(template, potential=terms.potentials[index],
                                           potential_wave=terms.waves[index])
        return _Chunk(k0=self.k0, gcart=self.gcart, mask=self.mask, core=self.core,
                      template=template, table=table, positions=positions, nk=self.nk,
                      channel=self.channel)


def _plain_norms(states):
    """``<u|u>`` of host states ``(nk, nbnd, ndim)``: the potential is updated for norm-conserving runs only."""
    states = np.asarray(states)
    return np.real(np.einsum("kng,kng->kn", np.conj(states), states))


def _stack(trees):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *trees)


class _Run:
    """What one self-consistent run needs, built once: the stacked chunks and the update.

    One entry per channel in ``parts``, ``w`` and ``psi``: a collinear
    ``nspin = 2`` run has two, each with its own number of carried bands, and
    every other run one. ``density(psi)`` takes the stacked states, one
    ``(nchunks, kb, nbnd, ndim)`` per channel, and returns the density on the
    dense grid, ``(nspin_mag, n1, n2, n3)`` -- the two channels' ``(up, down)``,
    or a spinor's ``(n, m_x, m_y, m_z)`` -- symmetrised with the little group's
    maps when there are any; ``potential(rho)`` returns ``(v(t), E_U[rho])``;
    ``local_energy(rho)`` is the part of the conserved energy that is not the
    band's kinetic and nonlocal energy.
    """

    def __init__(self, calculation, setups, pairs, v_scf, potential: str,
                 density_symmetry):
        from defumat.realtime.propagate import _chunk_weights
        from defumat.scf.driver import _symmetrize, _symmetrize_noncollinear
        from defumat.scf.potential import as_potential_components, hartree
        from defumat.system.symmetry import cartesian_rotations, symmetry_maps

        real = setups[0].real
        self.calculation = calculation
        self.mode = potential
        self.chunks = setups[0].chunks
        self.table = setups[0].table
        self.positions = jnp.asarray(calculation.system.structure.positions)
        parts, ws, psis, flats = [], [], [], []
        for setup, (states, weights) in zip(setups, pairs):
            built = [setup.chunk(i, rows) for i, (rows, _) in enumerate(self.chunks)]
            parts.append(_stack([_KPart.of(c) for c in built]))
            ws.append(_stack([_chunk_weights(weights, rows, live, real)
                              for rows, live in self.chunks]))
            psis.append(_stack([jnp.asarray(states[rows]) for rows, _ in self.chunks]))
            flats.append(jnp.asarray(weights, dtype=real))
        self.parts, self.w, self.psi = tuple(parts), tuple(ws), tuple(psis)
        kb = len(self.chunks[0][0])
        self.order = np.asarray([i * kb + j for i, (_, live) in enumerate(self.chunks)
                                 for j in range(live)])
        self.nk = pairs[0][0].shape[0]
        self.weights = tuple(flats)

        dense, smooth = calculation.basis.dense, calculation.basis.smooth
        cell = calculation.system.cell
        nspin_mag = calculation.nspin_mag
        maps = (None if density_symmetry is None
                else symmetry_maps(dense, density_symmetry))
        rotations = None
        if maps is not None and nspin_mag == 4:
            # an axial vector's rotation, ``det(R) R``; the field's little group
            # has no operation that needs time reversal (``field_symmetries``)
            cartesian = np.asarray(cartesian_rotations(cell, density_symmetry))
            rotations = jnp.asarray(np.sign(np.linalg.det(cartesian))[:, None, None] * cartesian)
        order, flat_weights = self.order, self.weights

        def density(psi):
            channels = []
            for states, weights in zip(psi, flat_weights):
                whole = states.reshape((-1,) + states.shape[2:])[order]
                channels.append(calculation.smooth_density(whole[None], weights[None]))
            rho = to_dense(jnp.concatenate(channels), smooth, dense)
            if maps is not None:
                if rotations is not None:
                    rho = _symmetrize_noncollinear(rho, dense.fft_index, dense.grid, maps,
                                                   rotations)
                else:
                    rho = _symmetrize(rho, dense.fft_index, dense.grid, maps)
            return rho

        if potential == "hxc":
            def field(rho):
                result = calculation.potential(rho)
                return result.v_scf, result.ehart + result.etxc
        else:
            def field(rho):
                total = rho[0] if calculation.noncolin or rho.shape[0] == 1 else rho.sum(axis=0)
                vg, energy = hartree(r_to_g(total, dense.fft_index), dense, cell)
                v = jnp.real(g_to_r(vg, dense.fft_index, dense.grid))
                return as_potential_components(v, nspin_mag), energy

        self.density = density
        self.rho0 = compiled_function(density, self.psi)(self.psi)
        u0, _ = field(self.rho0)
        v_scf = jnp.asarray(v_scf)
        self.v_scf = v_scf
        constant = as_potential_components(jnp.asarray(calculation.vltot), nspin_mag) + (v_scf - u0)
        volume = float(cell.volume)

        def potential_of(rho):
            u, energy = field(rho)
            return v_scf + u - u0, energy

        def local_energy(rho):
            _, energy = field(rho)
            return volume / rho[0].size * jnp.sum(constant * rho) + energy

        self.potential = potential_of
        self.local_energy = local_energy
        self.local_terms = calculation.local_terms

    def flat(self, psi):
        """The stacked states of each channel as ``(nk, nbnd, ndim)`` on the host, the padding dropped."""
        out = []
        for states in psi:
            states = np.asarray(states)
            out.append(states.reshape((-1,) + states.shape[2:])[self.order])
        return out


def _advance(step_fn, centres, parts, table, positions, psi, terms, kappa, step):
    """Every channel's and chunk's states one step on, at the local terms ``terms`` and ``kappa``."""
    moved = []
    for part_set, states, centre in zip(parts, psi, centres):
        def one(args, centre=centre):
            part, p = args
            ham = part.chunk(table, positions, terms).hamiltonian(kappa)
            return map_k(lambda ik: step_fn(lambda v: ham.apply(v, ik), p[ik], step, centre),
                         jnp.arange(part.nk), batch=None)
        moved.append(jax.lax.map(one, (part_set, states)))
    return tuple(moved)


def _slope(parts, table, positions, w, psi, kappa):
    """The current up to ``-1/Omega``, summed over channels and chunks, Ry bohr."""
    total = 0.0
    for part_set, wc, states in zip(parts, w, psi):
        def one(args):
            part, wk, p = args
            return part.chunk(table, positions).current(kappa, p, wk)
        total = total + jnp.sum(jax.lax.map(one, (part_set, wc, states)), axis=0)
    return total


def _band(parts, table, positions, w, psi, kappa):
    total = 0.0
    for part_set, wc, states in zip(parts, w, psi):
        def one(args):
            part, wk, p = args
            return part.chunk(table, positions).kappa_energy(kappa, p, wk)
        total = total + jnp.sum(jax.lax.map(one, (part_set, wc, states)))
    return total


def _block_function(run: _Run, step_fn, centres, corrector: int):
    """``block(parts, table, positions, w, carry, kmid, kend, steps) -> (carry, gradients)``.

    ``carry`` is ``(psi, v_prev, v_now)``: the stacked states at ``t`` and the
    potential at ``t - dt`` and ``t``. A step of length zero leaves the states
    and ``v_now`` where they are, which is how a block is padded.
    """
    density, potential_of, local_terms = run.density, run.potential, run.local_terms
    # constants of the program, not literals in it, one per channel
    centres = tuple(jnp.asarray(c) for c in centres)

    def block(parts, table, positions, w, carry, kmid, kend, steps):
        def body(state, x):
            psi, v_prev, v_now = state
            k_mid, k_end, step = x

            def advance(v_mid):
                return _advance(step_fn, centres, parts, table, positions, psi,
                                local_terms(v_mid), k_mid, step)

            moved = advance(1.5 * v_now - 0.5 * v_prev)
            v_next, _ = potential_of(density(moved))
            for _ in range(corrector):
                moved = advance(0.5 * (v_now + v_next))
                v_next, _ = potential_of(density(moved))
            gradient = _slope(parts, table, positions, w, moved, k_end)
            return (moved, v_now, v_next), gradient
        return jax.lax.scan(body, carry, (kmid, kend, steps))
    return block


def _energy_function(run: _Run):
    density, local_energy = run.density, run.local_energy

    def energy(parts, table, positions, w, psi, kappa):
        return _band(parts, table, positions, w, psi, kappa) + local_energy(density(psi))
    return energy


def _setups(calculation, states, weights, v_scf, kappa_max, dt, propagator, k_batch, kcart):
    """One :func:`~defumat.realtime.propagate._prepare` per channel, and the channels' states."""
    from defumat.realtime.propagate import _prepare, channel_arrays

    pairs = channel_arrays(calculation, states, weights)
    setups = [_prepare(calculation, own, carried, v_scf, kappa_max, dt, propagator,
                       k_batch, kcart, channel=channel)
              for channel, (own, carried) in enumerate(pairs)]
    return pairs, setups


def propagate_self_consistent(calculation, states, weights, v_scf, pulse, *, dt: float,
                              duration=None, start=None, propagator: str = "taylor4",
                              k_batch="default", block_steps: int = 400, kcart=None,
                              symmetrise=None, density_symmetry=None,
                              potential: str = "hxc", corrector: int = 1):
    """:func:`~defumat.realtime.propagate.propagate` with the potential updated in time.

    ``potential`` is ``'hartree'`` or ``'hxc'`` (module docstring);
    ``density_symmetry`` the :class:`~defumat.system.symmetry.Symmetries` of the
    field's little group when the k-set is its wedge, which ``symmetrise`` (its
    cartesian rotations, for the current) then says; ``corrector`` how many
    times the midpoint potential is corrected after its extrapolation. The
    energy on the result is the conserved functional of the module docstring.
    The states are :func:`~defumat.realtime.propagate.propagate`'s, one block
    per channel of a collinear run, which the density couples.
    """
    from defumat.realtime.propagate import (
        RealTimeResult, _check_growth, _padded_grid, _reach, _warn_damping, time_grid)

    require_a_potential_mode(calculation, potential, symmetrise, density_symmetry)
    times, kappa_t, kappa_mid, efield = time_grid(pulse, dt, duration, start)
    nsteps = len(times) - 1
    kappa_max = float(np.max(np.linalg.norm(np.concatenate([kappa_t, kappa_mid]), axis=-1)))
    pairs, setups = _setups(calculation, states, weights, v_scf, kappa_max, dt, propagator,
                            k_batch, kcart)
    for setup in setups:
        _warn_damping(setup, nsteps)
    real = setups[0].real
    run = _Run(calculation, setups, pairs, v_scf, potential, density_symmetry)
    kappa_mid_p, kappa_end_p, dts, nblocks = _padded_grid(
        kappa_mid, kappa_t[1:], nsteps, block_steps, setups[0].dt_ry)

    fixed = (run.parts, run.table, run.positions, run.w)
    carry = (run.psi, run.v_scf, run.v_scf)
    kappa0 = jnp.asarray(kappa_t[0], dtype=real)
    block = compiled_function(
        _block_function(run, setups[0].step_fn, [s.centre for s in setups], int(corrector)),
        *fixed, carry,
        jnp.asarray(kappa_mid_p[:block_steps], dtype=real),
        jnp.asarray(kappa_end_p[:block_steps], dtype=real),
        jnp.asarray(dts[:block_steps], dtype=real))
    measure = compiled_function(_energy_function(run), *fixed, run.psi, kappa0)
    slope = compiled_function(_slope, *fixed, run.psi, kappa0)

    current = np.zeros((nsteps + 1, 3))
    energy_index = [0] + [min((b + 1) * block_steps, nsteps) for b in range(nblocks)]
    energy = np.zeros(len(energy_index))
    current[0] = np.asarray(slope(*fixed, run.psi, kappa0))
    energy[0] = float(measure(*fixed, run.psi, kappa0))
    for b in range(nblocks):
        sl = slice(b * block_steps, (b + 1) * block_steps)
        carry, gradients = block(*fixed, carry, jnp.asarray(kappa_mid_p[sl], dtype=real),
                                 jnp.asarray(kappa_end_p[sl], dtype=real),
                                 jnp.asarray(dts[sl], dtype=real))
        stop = min((b + 1) * block_steps, nsteps)
        current[b * block_steps + 1:stop + 1] = np.asarray(gradients)[:stop - b * block_steps]
        energy[b + 1] = float(measure(*fixed, carry[0],
                                      jnp.asarray(kappa_t[stop], dtype=real)))
        final = run.flat(carry[0])
        norms = [_check_growth(_plain_norms(f), run.nk, f"after step {stop}") for f in final]

    excited = 0.0
    for (own, carried), f in zip(pairs, final):
        overlap = np.einsum("kmg,kng->kmn", np.conj(own), f)
        kept = np.sum(np.abs(overlap) ** 2, axis=1)
        excited += float(np.sum(carried * (1.0 - kept)))
    volume = setups[0].volume
    current = -current / (2.0 * volume)
    if symmetrise is not None:
        rotations = np.asarray(symmetrise, dtype=float)
        current = np.einsum("sab,tb->ta", rotations, current) / len(rotations)
    lower = min(s.lower for s in setups)
    upper = max(s.upper for s in setups)
    return RealTimeResult(
        times=times, kappa=kappa_t, efield=efield, current=current,
        energy_times=times[np.asarray(energy_index)], energy=0.5 * energy,
        norm_drift=max(float(np.abs(n - 1.0).max()) for n in norms), excited=excited,
        volume=volume, nelec=float(sum(np.sum(c) for _, c in pairs)), dt=float(dt),
        effective_cutoff=min(s.effective_cutoff for s in setups),
        propagator=propagator, pulse=pulse, spectrum=(lower, upper),
        step_radius=max(s.dt_ry * _reach(s.lower, s.upper, s.centre) for s in setups),
        symmetry_operations=1 if symmetrise is None else len(symmetrise),
        extras={"potential": potential, "corrector": int(corrector),
                "potential_change": float(np.abs(np.asarray(carry[2] - run.v_scf)).max())},
    )


def propagate_orders_self_consistent(calculation, states, weights, v_scf, shape, *,
                                     dt: float, order: int = 3, duration=None, start=None,
                                     propagator: str = "taylor4", k_batch="default",
                                     block_steps: int = 400, kcart=None, symmetrise=None,
                                     density_symmetry=None, potential: str = "hxc",
                                     corrector: int = 1):
    """:func:`~defumat.realtime.orders.propagate_orders` with the potential updated in time.

    The tower of nested ``jvp`` carries the states and the two potentials of
    the step, so the derivative of the density goes through the potential as
    it does in the Sternheimer stack; order ``n`` carries ``2^n`` copies of the
    whole mesh's states.
    """
    from defumat.realtime.orders import OrdersResult, _derivative, _lift, _tower
    from defumat.realtime.propagate import (
        _check_growth, _padded_grid, _warn_damping, time_grid)

    require_a_potential_mode(calculation, potential, symmetrise, density_symmetry)
    times, a_t, a_mid, _ = time_grid(shape, dt, duration, start)
    nsteps = len(times) - 1
    pairs, setups = _setups(calculation, states, weights, v_scf, 0.0, dt, propagator,
                            k_batch, kcart)
    for setup in setups:
        _warn_damping(setup, nsteps)
    real = setups[0].real
    run = _Run(calculation, setups, pairs, v_scf, potential, density_symmetry)
    a_mid_p, a_end_p, dts, nblocks = _padded_grid(a_mid, a_t[1:], nsteps, block_steps,
                                                  setups[0].dt_ry)
    base = _block_function(run, setups[0].step_fn, [s.centre for s in setups], int(corrector))
    depth = int(order)

    def lifted(parts, table, positions, w, tower, lam, amid, aend, steps):
        def f(carry, l):
            return base(parts, table, positions, w, carry, l * amid, l * aend, steps)
        return _lift(f, depth)(tower, lam)

    def initial(parts, table, positions, w, tower, lam, a0):
        def f(carry, l):
            return carry, _slope(parts, table, positions, w, carry[0], l * a0)
        return _lift(f, depth)(tower, lam)[1]

    fixed = (run.parts, run.table, run.positions, run.w)
    lam = jnp.zeros((), dtype=real)
    a0 = jnp.asarray(a_t[0], dtype=real)
    tower = _tower((run.psi, run.v_scf, run.v_scf), depth)
    run_block = compiled_function(
        lifted, *fixed, tower, lam, jnp.asarray(a_mid_p[:block_steps], dtype=real),
        jnp.asarray(a_end_p[:block_steps], dtype=real),
        jnp.asarray(dts[:block_steps], dtype=real))
    start_run = compiled_function(initial, *fixed, tower, lam, a0)

    raw = np.zeros((depth + 1, nsteps + 1, 3))
    out = start_run(*fixed, tower, lam, a0)
    for k in range(depth + 1):
        raw[k, 0] = np.asarray(_derivative(out, k, depth))
    for b in range(nblocks):
        sl = slice(b * block_steps, (b + 1) * block_steps)
        tower, out = run_block(*fixed, tower, lam, jnp.asarray(a_mid_p[sl], dtype=real),
                               jnp.asarray(a_end_p[sl], dtype=real),
                               jnp.asarray(dts[sl], dtype=real))
        stop = min((b + 1) * block_steps, nsteps)
        for k in range(depth + 1):
            raw[k, b * block_steps + 1:stop + 1] = np.asarray(
                _derivative(out, k, depth))[:stop - b * block_steps]
        norms = [_check_growth(_plain_norms(f), run.nk, f"after step {stop}")
                 for f in run.flat(_derivative(tower, 0, depth)[0])]

    factorials = np.asarray([math.factorial(k) for k in range(depth + 1)], dtype=float)
    currents = -raw / (2.0 * setups[0].volume) / factorials[:, None, None]
    if symmetrise is not None:
        rotations = np.asarray(symmetrise, dtype=float)
        currents = np.einsum("sab,ntb->nta", rotations, currents) / len(rotations)
    return OrdersResult(times=times, currents=currents, shape=shape, dt=float(dt),
                        volume=setups[0].volume,
                        norm_drift=max(float(np.abs(n - 1.0).max()) for n in norms))
