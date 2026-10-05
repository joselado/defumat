"""The dynamical matrix at ``Gamma``, a k-chunk at a time.

``GPU-MEMORY-NEXT.md`` item 2, its second piece after the field response
(:mod:`defumat.response.chunked`). :func:`~defumat.response.phonon.
dynamical_matrix` holds, for its ``P = 3 len(atoms)`` perturbations, a bare
perturbation ``(dH/du - eps dS/du)|psi>`` and a first-order state ``dpsi`` each --
and for an ultrasoft or PAW dataset the occupied block ``ort`` as well -- every one
a ``(nspin, nk, nocc, ndim)`` array, all whole-k and all on the device; its
assembly then differentiates the force once more with the whole k axis on one
tape. Here every such array lives in host memory and every pass is one chunk's.

**The walks**, each a module-level ``jax.jit`` taking the chunk's arrays and the
unit tangent ``e_p`` as arguments, so one program serves every chunk and every
perturbation:

* **setup**, once: per chunk and perturbation, the bare perturbation (one ``jvp``
  of ``H(x) - eps S(x)`` through ``at_positions`` on the chunk's row-subset
  calculation), and for ultrasoft or PAW the overlap's derivative
  ``S'_mn = <psi_m|dS/du|psi_n>`` and the tangents of the two raw sums over k the
  mixed state is built from -- ``becsum`` and the smooth density -- along the
  atom's motion at frozen states and along the occupied block. **``ort`` is not
  stored**: it is ``-1/2 psi S'``, rebuilt from the chunk's states and the small
  ``(nocc, nocc)`` matrix wherever it is needed. The two halves of
  ``drhous``/``becsumort`` are finished once from the summed tangents and
  symmetrised as the whole route symmetrises them;
* **the response**, once per self-consistent iteration: the field's own
  ``respond`` pass (:func:`defumat.response.chunked._passes`), which has nothing
  in it that is about a field, with ``P`` perturbations in place of three. The
  loop around it is :func:`~defumat.response.phonon.screening_loop`, shared with
  the whole route, and acts on whole-grid objects only. A metal's ``ldos`` is
  walked once rather than rebuilt per mode and iteration;
* **the assembly**, per perturbation (so that ``on_row`` still fires as each row
  is finished): for a norm-conserving dataset the energy is ``E_glob(x, rho) +
  sum_c E_c(x, psi_c)`` with ``rho`` an independent array, so each column is a
  ``jvp`` of the global terms' gradient plus a sum over chunks of the separable
  part's two ``jvp`` -- the frozen Hessian at ``wg``, the state response at
  ``wk`` (``PLAN.md`` P28). For ultrasoft and PAW the mixed state couples the
  chunks, and the column is :mod:`defumat.response.chunked`'s Born split with the
  coordinate itself moving: a forward walk for the raw sums' tangents, one
  ``jvp`` of the global terms' gradient at the whole sums, and a pull-back walk.

Against the whole-k route on the same converged state
(``tests/regression/test_streamed_phonons.py``): the matrix to round-off on the
norm-conserving wedge, a norm-conserving metal, an ultrasoft polar cell, a PAW
wedge, half-sphere storage and an ``atoms=`` subset.

**The phonon at ``q``** (:class:`StreamedDisplacementsAtQ`) is the same three
stages on two spheres: the ``k + q`` states are written into a host store a
chunk at a time (``stream_states``), each chunk has two row-subset calculations
built from the calculation's own fields, the solve pass is the two-sphere
solver's, the response density's sum over k is added over chunks and finished
once, the electronic half of the matrix is one Gram product of the two host
stores, and the frozen half is the Gamma passes with zero tangents before the
Ewald term is moved to ``q``. Against the whole route: 2.1e-14 end to end at one
k-point a chunk, and 1.9e-14 at a chunk of 3 with one ``k + q`` store handed to
both.

**The ``P`` dense-grid fields the loop carries** (``dvscf``, the response,
the induced potential, ``drhous`` and the core term, six to seven grids per
perturbation at an iteration's peak) are kept in host memory too, one mode at a
time on the device, **wherever the run has no symmetry** (``host_fields``):
there the displacement average is the identity and nothing in
:func:`~defumat.response.phonon.screening_loop` needs the modes together, and
the mixer's history was in host memory already. With symmetry they stay on the
device, because ``symmetrize_atom_displacement`` acts on the stack as one object.

**What it does not change**: a response handed in through ``response=``, a
strained calculation and a k-point pool, which take the whole-k route or are
refused before this is reached. The phonon at ``q``'s loop
(:func:`~defumat.response.phononq.screening_loop_at_q`) keeps its ``3 nat``
complex grids in host memory always, since it runs without symmetry.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from defumat.basis.interpolate import to_dense
from defumat.batching import k_chunks
from defumat.forces.chunked import _rows_of, _separable, row_leaves, with_rows
from defumat.forces.energy import (
    _constraint_energy, _kinetic_energy, _projector_energies, hoisted, with_hoisted,
)
from defumat.response.chunked import _add, _passes as _field_passes
from defumat.response.sternheimer import SternheimerSolver, local_perturbation, scalars_at
from defumat.response.velocity import over_kpoints
from defumat.scf.potential import total_charge

__all__ = ["StreamedDisplacements"]


class StreamedDisplacements:
    """The displacement response's stores in host memory, and the walks over them.

    The four methods :func:`~defumat.response.phonon.screening_loop` drives
    (:meth:`respond`, :meth:`fermi_level_shift`, :meth:`shift_states` and the
    counters), plus :meth:`prepare` before the loop and :meth:`force_constants`
    after it. ``solver`` is the whole set's
    :class:`~defumat.response.sternheimer.SternheimerSolver`, built on the host
    store: it supplies the eigenvalues, the weights, the masks and the level
    shift, and its states stay where they are. ``wavefunctions`` is the whole
    store, every band, which the mixed state and the frozen-state functional are
    built from; ``weights`` is ``wg`` over every band.
    """

    def __init__(self, calculation, solver, v_scf, positions, atoms,
                 wavefunctions, eigenvalues, weights, density, becsum):
        self.calculation = calculation
        self.solver = solver
        self.v_scf = v_scf
        self.ddd_paw = solver.ddd_paw
        self.hamiltonians = solver.hamiltonians
        self.positions = jnp.asarray(positions)
        self.atoms = tuple(atoms)
        self.rows = len(self.atoms)
        #: ``(row, atom, cart)`` in the order the loop stacks the modes.
        self.modes = [(row, atom, cart) for row, atom in enumerate(self.atoms)
                      for cart in range(3)]
        self.wavefunctions = wavefunctions
        eigenvalues = np.asarray(eigenvalues)
        self.eigenvalues = eigenvalues[None] if eigenvalues.ndim == 2 else eigenvalues
        self.weights = np.asarray(weights)
        self.density = jnp.asarray(density)
        self.becsum = becsum
        self.ultrasoft = bool(calculation.is_ultrasoft)
        self.chunks = list(k_chunks(calculation.system.kpoints.nk,
                                    calculation.k_batch))
        key = (solver.nocc, solver.occupied_counts, solver.smearing)
        self.field_passes = _field_passes(calculation, key)
        self.passes = _phonon_passes(calculation, key)
        self.big = hoisted(calculation)
        self.scalars = solver.scalars()
        shape = (self.rows, 3) + tuple(solver.psi.shape)
        dtype = np.dtype(solver.psi.dtype)
        self.bare = np.zeros(shape, dtype)
        self.dpsi = np.zeros(shape, dtype)
        nspin, nk, nocc = solver.psi.shape[:3]
        #: ``S'_p``, ``(rows, 3, nspin, nk, nocc, nocc)``: what ``ort`` and the
        #: multipliers' gauge term are rebuilt from.
        self.overlaps = (np.zeros((self.rows, 3, nspin, nk, nocc, nocc), dtype)
                         if self.ultrasoft else None)
        self.multipliers = None
        self.drhous = None
        self.becsumort = None
        self.iterations = 0
        self.solves = 0
        self._levels = None
        #: Whether :func:`~defumat.response.phonon.screening_loop` keeps the
        #: ``P`` dense grids in host memory and puts one mode at a time on the
        #: device: wherever the run has no symmetry, so that the displacement
        #: average is the identity and nothing needs the modes together.
        self.host_fields = calculation._symmetry_maps is None

    # -- the chunk's arguments -------------------------------------------------

    def _on_device(self, field):
        """One mode's field for a pass: uploaded from the host store in host
        mode, where the device holds no stack of them."""
        return jax.device_put(field) if isinstance(field, np.ndarray) else field

    def _tangent(self, atom: int, cart: int):
        """The unit displacement, built on the host so that no index is a
        constant of anything compiled."""
        tangent = np.zeros(self.positions.shape)
        tangent[atom, cart] = 1.0
        return jnp.asarray(tangent, dtype=self.positions.dtype)

    def _arguments(self, rows, live, threshold=None):
        """The leading arguments of the solver passes, for one chunk, with
        ``threshold`` in the traced scalars when a schedule sets one."""
        return (self.big, row_leaves(self.calculation, rows), self.hamiltonians,
                _rows_of(self.solver.psi, rows),
                self.solver.chunk_arrays(rows, live),
                scalars_at(self.scalars, threshold), self.v_scf, self.ddd_paw)

    def _states(self, rows, live):
        """Every band of one chunk, with ``wg`` and the eigenvalues; padding zeroed."""
        weights = np.array(self.weights[:, rows])
        weights[:, live:] = 0.0
        return (_rows_of(self.wavefunctions, rows), jnp.asarray(weights),
                jnp.asarray(self.eigenvalues[:, rows]))

    def _state_weights(self, arrays, weights):
        """``wg``, or a metal's ``wk`` -- :func:`~defumat.response.phonon._state_weights`
        on one chunk, from the chunk's density weights so the padding is zero."""
        if self.solver.smearing is None:
            return weights
        return jnp.broadcast_to(arrays["density_weights"][:, :, :1], weights.shape)

    # -- walk 1: the bare perturbations and what S's motion adds ---------------

    def prepare(self) -> None:
        """The bare perturbations into the host store, and ``drhous``/``becsumort``."""
        from defumat.response.phonon import (
            _add_becsum, _stack_modes, _symmetrize_modes, symmetrize_becsum_modes,
        )

        calculation = self.calculation
        moved_sums = [None] * len(self.modes)
        ort_sums = [None] * len(self.modes)
        for rows, live in self.chunks:
            arguments = self._arguments(rows, live)
            written = rows[:live]
            if self.ultrasoft:
                psi, weights, _ = self._states(rows, live)
            for index, (row, atom, cart) in enumerate(self.modes):
                tangent = self._tangent(atom, cart)
                bare = self.passes["bare"](
                    arguments[0], arguments[1], arguments[3],
                    arguments[4]["eigenvalues"], self.v_scf, self.ddd_paw,
                    self.positions, tangent)
                self.bare[row, cart][:, written] = np.asarray(bare)[:, :live]
                if not self.ultrasoft:
                    continue
                overlap, moved, ort = self.passes["mixed"](
                    arguments[0], arguments[1], self.positions, tangent, psi,
                    weights)
                self.overlaps[row, cart][:, written] = np.asarray(overlap)[:, :live]
                moved_sums[index] = _add(moved_sums[index], moved)
                ort_sums[index] = _add(ort_sums[index], ort)
        if not self.ultrasoft:
            return

        # ``non_variational_response``'s two halves, finished from the summed
        # tangents: the atom's motion at frozen states moves ``becsum`` through
        # the projectors and the augmentation charge itself, at the converged
        # ``becsum``; the occupied block moves both sums and nothing else.
        rowset = row_leaves(calculation, self.chunks[0][0])
        shape = (self.rows, 3)
        moved_density = np.empty(shape, dtype=object)
        moved_becsum = np.empty(shape, dtype=object)
        ort_density = np.empty(shape, dtype=object)
        ort_becsum = np.empty(shape, dtype=object)
        keep = np.asarray if self.host_fields else (lambda field: field)
        for index, (row, atom, cart) in enumerate(self.modes):
            d_becsum, _ = moved_sums[index]
            moved_density[row, cart] = keep(self.passes["moved_density"](
                self.big, rowset, self.positions, self._tangent(atom, cart),
                self.density, tuple(self.becsum), d_becsum))
            moved_becsum[row, cart] = d_becsum
            o_becsum, o_smooth = ort_sums[index]
            ort_density[row, cart] = keep(
                self.solver.finish_density(o_smooth, o_becsum))
            ort_becsum[row, cart] = o_becsum
        if self.host_fields:
            # No symmetry, so the average is the identity: stacked on the host.
            self.drhous = (_stack_host(moved_density)
                           + _stack_host(ort_density))
        else:
            moved_stacked = _stack_modes(
                _symmetrize_modes(calculation, moved_density))
            self.drhous = moved_stacked + _stack_modes(
                _symmetrize_modes(calculation, ort_density))
        self.becsumort = _add_becsum(
            symmetrize_becsum_modes(calculation, moved_becsum),
            symmetrize_becsum_modes(calculation, ort_becsum))

    # -- walk 2: the solves, once per iteration --------------------------------

    def _coefficients(self, dvscf, onecentre, include_induced):
        """``(dv, int3 + d(ddd_paw))`` per mode: k-independent, so once per mode
        and iteration rather than per chunk; ``None`` coefficients on a
        norm-conserving dataset."""
        zeros = np.zeros_like if self.host_fields else jnp.zeros_like
        fields, coefficients = [], []
        for row, _, cart in self.modes:
            dv = dvscf[row, cart] if include_induced else zeros(dvscf[row, cart])
            dddd = None if onecentre is None else (
                onecentre[row, cart] if include_induced
                else jnp.zeros_like(onecentre[row, cart]))
            fields.append(dv)
            coefficients.append(
                self.solver.perturbed_coefficients(self._on_device(dv), dddd)
                if self.ultrasoft else None)
        return fields, coefficients

    def respond(self, dvscf, onecentre, include_induced: bool, threshold=None):
        """One iteration's solves: the finished, unsymmetrised response density
        per mode, and for PAW the raw ``becsum`` response. Each chunk starts
        from its previous solution in the host store (zero before the first
        pass); ``threshold`` is this pass's CG threshold."""
        solver = self.solver
        fields, coefficients = self._coefficients(dvscf, onecentre, include_induced)
        parts = [None] * len(self.modes)
        worst = [0] * len(self.modes)
        add = _add_host if self.host_fields else _add
        for rows, live in self.chunks:
            arguments = self._arguments(rows, live, threshold)
            written = rows[:live]
            for index, (row, _, cart) in enumerate(self.modes):
                dpsi, steps, _, chunk_parts = self.field_passes["respond"](
                    *arguments, _rows_of(self.bare[row, cart], rows),
                    self._on_device(fields[index]), coefficients[index],
                    _rows_of(self.dpsi[row, cart], rows))
                self.dpsi[row, cart][:, written] = np.asarray(dpsi)[:, :live]
                # In host mode each mode's share of the two raw sums is added on
                # the host, in the same order, so the ``P`` accumulators are not
                # held on the device across the chunks.
                parts[index] = add(parts[index], chunk_parts)
                worst[index] = max(worst[index], int(np.max(np.asarray(steps))))
        self.iterations += sum(worst)
        self.solves += len(modes)
        if self.host_fields:
            response = [np.asarray(solver.finish_density(
                *jax.tree_util.tree_map(jax.device_put, part))) for part in parts]
        else:
            response = [solver.finish_density(*part) for part in parts]
        becsum_response = ([part[1] for part in parts]
                           if onecentre is not None else [])
        return response, becsum_response

    def fermi_level_shift(self, drho):
        """``ef_shift`` with ``ldos`` and ``dos_ef`` walked once, not per call."""
        if self._levels is None:
            total, dos_ef = None, 0.0
            for rows, live in self.chunks:
                parts, weight = self.passes["ldos"](*self._arguments(rows, live))
                total = _add(total, parts)
                dos_ef += float(weight)
            self._levels = (self.solver.finish_density(*total), dos_ef)
        corrected, shift = self.solver.fermi_level_shift(
            self._on_device(drho), levels=self._levels)
        if self.host_fields:
            corrected = np.asarray(corrected)
        return corrected, shift

    def shift_states(self, shifts) -> None:
        """``ef_shift_wfc`` on the host store, chunk by chunk, ``shifts`` in mode order."""
        for rows, live in self.chunks:
            arguments = self._arguments(rows, live)
            written = rows[:live]
            for index, (row, _, cart) in enumerate(self.modes):
                shifted = self.passes["shift"](
                    *arguments, _rows_of(self.dpsi[row, cart], rows),
                    jnp.asarray(shifts[index]))
                self.dpsi[row, cart][:, written] = np.asarray(shifted)[:, :live]

    # -- after the loop: the multipliers and the matrix ------------------------

    def _multipliers(self, extras) -> None:
        """``dLambda`` per mode into the host store -- :func:`~defumat.response.
        phonon.multiplier_response` on each chunk, at the converged ``dV_scf``."""
        fields, coefficients = self._coefficients(
            extras["dvscf"], extras["onecentre"], True)
        nspin, nk, nbnd = self.weights.shape
        self.multipliers = np.zeros(
            (self.rows, 3, nspin, nk, nbnd, nbnd), np.dtype(self.solver.psi.dtype))
        for rows, live in self.chunks:
            arguments = self._arguments(rows, live)
            _, weights, _ = self._states(rows, live)
            written = rows[:live]
            for index, (row, _, cart) in enumerate(self.modes):
                dlambda = self.passes["multipliers"](
                    *arguments, weights, _rows_of(self.bare[row, cart], rows),
                    self._on_device(fields[index]), coefficients[index],
                    _rows_of(self.overlaps[row, cart], rows))
                self.multipliers[row, cart][:, written] = (
                    np.asarray(dlambda)[:, :live])

    def force_constants(self, drho, extras, on_row=None) -> np.ndarray:
        """``(rows, 3, nat, 3)``: :func:`~defumat.response.phonon._force_constants`
        walked, one perturbation at a time."""
        nat = self.positions.shape[0]
        matrix = np.zeros((self.rows, 3, nat, 3))
        rowset = row_leaves(self.calculation, self.chunks[0][0])
        if self.ultrasoft:
            self._multipliers(extras)
            raw = self._raw_sums()
        for row, atom, cart in self.modes:
            tangent = self._tangent(atom, cart)
            if self.ultrasoft:
                dbecsum = extras.get("dbecsum")
                column = self._ultrasoft_column(
                    row, cart, tangent, rowset, raw,
                    self._on_device(drho[row, cart]),
                    None if dbecsum is None else dbecsum[row, cart])
            else:
                column = np.asarray(self.passes["global_nc"](
                    self.big, rowset, self.positions, tangent, self.density,
                    self._on_device(drho[row, cart])))
                for rows, live in self.chunks:
                    psi, weights, eigenvalues = self._states(rows, live)
                    arrays = self.solver.chunk_arrays(rows, live)
                    column = column + np.asarray(self.passes["pull_nc"](
                        self.big, row_leaves(self.calculation, rows),
                        self.positions, tangent, psi, weights,
                        self._state_weights(arrays, weights), eigenvalues,
                        _rows_of(self.dpsi[row, cart], rows)))
            matrix[row, cart] = column
            if on_row is not None:
                on_row(atom, cart, matrix[row, cart])
        return matrix

    def _raw_sums(self):
        """``(becsum, rho_smooth)`` raw over k, and the offsets that make them the
        converged mixed state in value (:func:`~defumat.response.born._raw_mixed_state`)."""
        calculation = self.calculation
        total = None
        for rows, live in self.chunks:
            psi, weights, _ = self._states(rows, live)
            total = _add(total, self.passes["sums"](
                self.big, row_leaves(calculation, rows), self.positions, psi,
                weights))
        raw_becsum, raw_smooth = total
        becsum_offset = tuple(
            None if value is None else jnp.asarray(value) - part
            for value, part in zip(self.becsum, raw_becsum))
        parts = tuple(None if part is None else part + offset
                      for part, offset in zip(raw_becsum, becsum_offset))
        smooth, dense = calculation.basis.smooth, calculation.basis.dense
        density_offset = self.density - calculation.augmented(
            to_dense(raw_smooth, smooth, dense), parts)
        return raw_becsum, raw_smooth, becsum_offset, density_offset

    def _ultrasoft_column(self, row, cart, tangent, rowset, raw, drho, dbecsum):
        """One column by the Born split with the coordinate moving: forward walk,
        the global terms' ``jvp``, pull-back walk."""
        raw_becsum, raw_smooth, becsum_offset, density_offset = raw
        tangents = None
        for rows, live in self.chunks:
            psi, weights, _ = self._states(rows, live)
            tangents = _add(tangents, self.passes["forward"](
                self.big, row_leaves(self.calculation, rows), self.positions,
                tangent, psi, weights, _rows_of(self.dpsi[row, cart], rows),
                _rows_of(self.overlaps[row, cart], rows)))
        d_becsum, d_smooth = tangents
        # An ultrasoft dataset's assembly has no ``becsum`` correction of its own
        # (``parts = 0`` in the whole route); handing the raw tangent in as the
        # symmetrised one makes the difference exactly zero.
        (_, g_b, g_rho), (column, dg_b, dg_rho) = self.passes["global_us"](
            self.big, rowset, self.positions, tangent, raw_becsum, raw_smooth,
            becsum_offset, density_offset, d_becsum, d_smooth, drho,
            d_becsum if dbecsum is None else tuple(dbecsum))
        column = np.asarray(column)
        for rows, live in self.chunks:
            psi, weights, eigenvalues = self._states(rows, live)
            # **The padded rows' multipliers are zero, not a repeat.** Every
            # other term of a padded row is killed by its zero weight, but the
            # constraint ``Tr[Lambda (<psi|S|psi> - 1)]`` is weighted by the
            # multipliers themselves, and ``dLambda`` read back by global row
            # index would hand the repeat its original's: 1.3e-2 on ultrasoft
            # AlAs's force constants of 0.22, in chunks of 3 over 8 k-points.
            multipliers = np.array(self.multipliers[row, cart][:, rows])
            multipliers[:, live:] = 0.0
            column = column + np.asarray(self.passes["pull_us"](
                self.big, row_leaves(self.calculation, rows), self.positions,
                tangent, psi, weights, eigenvalues, g_b, g_rho,
                _rows_of(self.dpsi[row, cart], rows),
                _rows_of(self.overlaps[row, cart], rows),
                jax.device_put(multipliers), dg_b, dg_rho))
        return column


def _add_host(total, part):
    """:func:`~defumat.response.chunked._add` into a host accumulator."""
    part = jax.tree_util.tree_map(np.asarray, part)
    return part if total is None else jax.tree_util.tree_map(np.add, total, part)


def _stack_host(per_mode) -> np.ndarray:
    """``(rows, 3)`` object array of host grids -> one stacked host array."""
    rows, ncart = per_mode.shape
    return np.stack([np.stack([per_mode[row, cart] for cart in range(ncart)])
                     for row in range(rows)])


def _phonon_passes(calculation, key) -> dict:
    """The displacement response's compiled passes, built once and cached on it.

    Keyed as :func:`defumat.response.chunked._passes` is, in the same cache, so
    a moved or strained copy builds its own. ``key`` is the solver's static
    configuration: the occupied-block width, the counts per channel and the
    smearing.
    """
    cached = calculation.__dict__.get("_streamed_response")
    if cached is None or cached[0] is not calculation:
        cached = (calculation, {})
        calculation._streamed_response = cached
    key = ("phonon",) + tuple(key)
    if key in cached[1]:
        return cached[1][key]
    from defumat.response.born import _multiplier_response
    from defumat.response.phonon import _ground_state_multipliers

    nocc, counts, smearing = key[1:]
    batch = calculation.k_batch

    def local(big, rowset):
        return with_rows(with_hoisted(calculation, big), rowset)

    def solver_on(big, rowset, hamiltonians, psi, arrays, scalars, v_scf, ddd_paw):
        sub = local(big, rowset)
        return SternheimerSolver.on_chunk(
            sub, sub.restricted_hamiltonians(hamiltonians), psi, arrays, scalars,
            nocc=nocc, occupied_counts=counts, smearing=smearing, v_scf=v_scf,
            ddd_paw=ddd_paw,
        )

    def embed(psi, block):
        """A tangent over the first bands, widened to every band with zeros."""
        return jnp.zeros_like(psi).at[:, :, :block.shape[2]].set(block)

    def orthogonal(psi, overlap):
        """``ort = -1/2 sum_m psi_m S'_mn`` (:func:`~defumat.response.phonon.
        orthogonality_states`), widened to every band."""
        return embed(psi, -0.5 * jnp.einsum("skmg,skmn->skng", psi[:, :, :nocc],
                                            overlap))

    def sums(moved, states, weights):
        rows = jnp.arange(states.shape[1])
        return (moved.becsum(states, weights, rows=rows, symmetrize=False),
                moved.smooth_density(states, weights, rows=rows))

    def bare(big, rowset, psi, eigenvalues, v_scf, ddd_paw, x, dx):
        """``(dH/du - eps dS/du)|psi>`` on one chunk -- ``_bare_displacements``."""
        sub = local(big, rowset)

        def h_psi(position):
            moved = sub.at_positions(position)
            applied = []
            for spin, hamiltonian in enumerate(moved.hamiltonian(v_scf, ddd_paw)):
                values = over_kpoints(hamiltonian, psi[spin], batch)
                if hamiltonian.has_overlap:
                    overlap = over_kpoints(hamiltonian, psi[spin], batch,
                                           overlap=True)
                    values = values - eigenvalues[spin][..., None] * overlap
                applied.append(values)
            return jnp.stack(applied)

        return jax.jvp(h_psi, (x,), (dx,))[1]

    def mixed(big, rowset, x, dx, psi, weights):
        """``S'`` on one chunk, and the raw sums' tangents at frozen states and
        along ``ort`` -- ``overlap_derivatives`` and ``non_variational_response``."""
        sub = local(big, rowset)
        occupied = psi[:, :, :nocc]

        def overlap_matrix(position):
            moved = sub.at_positions(position)
            vkb = moved.projectors_at(jnp.arange(psi.shape[1]))
            qq = moved.projectors.qq.astype(psi.dtype)
            becp = jnp.einsum("kgc,skng->sknc", vkb.conj(), occupied)
            return jnp.einsum("skmi,ij,sknj->skmn", becp.conj(), qq, becp)

        overlap = jax.jvp(overlap_matrix, (x,), (dx,))[1]
        _, moved_tangent = jax.jvp(
            lambda position: sums(sub.at_positions(position), psi, weights),
            (x,), (dx,))
        _, ort_tangent = jax.jvp(lambda states: sums(sub, states, weights),
                                 (psi,), (orthogonal(psi, overlap),))
        return overlap, moved_tangent, ort_tangent

    def moved_density(big, rowset, x, dx, template, becsum_, d_becsum):
        """The augmentation charge's change with its atom at the converged
        ``becsum``, plus ``becsum``'s own change through it."""
        sub = local(big, rowset)
        zero = jnp.zeros_like(template)
        return jax.jvp(
            lambda position, parts: sub.at_positions(position).augmented(zero, parts),
            (x, becsum_), (dx, d_becsum))[1]

    def ldos(big, rowset, hamiltonians, psi, arrays, scalars, v_scf, ddd_paw):
        """One chunk's share of ``localdos``: the density parts at the smeared
        delta's weights, and their sum."""
        solver = solver_on(big, rowset, hamiltonians, psi, arrays, scalars,
                           v_scf, ddd_paw)
        weights = solver._delta_weights()
        return solver.density_parts(solver.psi, weights), jnp.sum(weights)

    def shift(big, rowset, hamiltonians, psi, arrays, scalars, v_scf, ddd_paw,
              dpsi, value):
        solver = solver_on(big, rowset, hamiltonians, psi, arrays, scalars,
                           v_scf, ddd_paw)
        return solver.fermi_level_shift_states(dpsi, value)

    def multipliers(big, rowset, hamiltonians, psi, arrays, scalars, v_scf,
                    ddd_paw, weights, bare_c, dv, coefficients, overlap):
        """``dLambda`` on one chunk -- ``multiplier_response``."""
        solver = solver_on(big, rowset, hamiltonians, psi, arrays, scalars,
                           v_scf, ddd_paw)
        induced = local_perturbation(solver.calculation, dv, v_scf, ddd_paw,
                                     coefficients=coefficients)

        def perturbation(states, ik, spin):
            return bare_c[spin][ik] + induced(states, ik, spin)

        matrix = _multiplier_response(solver, perturbation, weights,
                                      weights.shape[-1], nocc)
        eps = solver.eigenvalues[..., :nocc]
        gaps = eps[..., None, :] - eps[..., :, None]
        correction = 0.5 * gaps * overlap * weights[:, :, None, :nocc]
        return matrix.at[:, :, :nocc, :nocc].add(correction)

    def global_energy(moved, rho):
        """The norm-conserving functional's terms that read the density alone."""
        potential = moved.potential(rho)
        volume = moved.system.cell.volume
        local_ = volume / rho[0].size * jnp.sum(moved.vltot * total_charge(rho))
        return (local_ + potential.ehart + potential.etxc + moved.ewald
                + moved.dispersion)

    def global_nc(big, rowset, x, dx, rho, drho):
        """``jvp_(x, rho) [grad_x E_glob]`` along ``(e_p, drho_p)``."""
        sub = local(big, rowset)
        gradient = jax.grad(
            lambda position, density: global_energy(sub.at_positions(position),
                                                    density))
        return jax.jvp(gradient, (x, rho), (dx, drho))[1]

    def pull_nc(big, rowset, x, dx, psi, weights, state_weights, eigenvalues, dpsi):
        """The separable part's frozen Hessian at ``wg`` and state response at
        ``wk`` on one chunk -- ``_force_constants``' two ``jvp``."""
        sub = local(big, rowset)
        rows = jnp.arange(psi.shape[1])

        def gradient(position, states, w):
            return jax.grad(lambda p: _separable(
                sub.at_positions(p), states, w, eigenvalues, rows)[0])(position)

        _, hessian = jax.jvp(lambda p: gradient(p, psi, weights), (x,), (dx,))
        _, response = jax.jvp(lambda states: gradient(x, states, state_weights),
                              (psi,), (embed(psi, dpsi),))
        return hessian + response

    def sums_pass(big, rowset, x, psi, weights):
        return sums(local(big, rowset).at_positions(x), psi, weights)

    def forward(big, rowset, x, dx, psi, weights, dpsi, overlap):
        """The raw sums' tangents on one chunk along ``(e_p, dpsi + ort)``."""
        sub = local(big, rowset)
        states = embed(psi, dpsi) + orthogonal(psi, overlap)
        return jax.jvp(
            lambda position, s: sums(sub.at_positions(position), s, weights),
            (x, psi), (dx, states))[1]

    def global_us(big, rowset, x, dx, becsum_, rho_smooth, becsum_offset,
                  density_offset, d_becsum, d_smooth, drho, dbecsum):
        """``(grad, d grad)`` of the terms that are a function of the whole sums.

        The density's tangent is made the symmetrised response ``drho`` and
        ``becsum``'s the symmetrised ``dbecsum``: ``s`` and ``bs`` carry the
        differences from the raw tangents with zero primal, as in
        :func:`~defumat.response.phonon._force_constants`, including its
        subtraction of the augmentation charge's share of ``bs``.
        """
        sub = local(big, rowset)
        smooth, dense = sub.basis.smooth, sub.basis.dense

        def mixed_state(position, b, rho_s, bs):
            moved = sub.at_positions(position)
            parts = tuple(
                None if part is None else part + offset + extra
                for part, offset, extra in zip(b, becsum_offset, bs))
            return moved, parts, moved.augmented(to_dense(rho_s, smooth, dense),
                                                 parts)

        def whole(position, b, rho_s, s, bs):
            moved, parts, rho = mixed_state(position, b, rho_s, bs)
            rho = rho + density_offset + s
            potential = moved.potential(rho)
            epaw, _ = moved.onecenter(parts)
            volume = moved.system.cell.volume
            local_ = volume / rho[0].size * jnp.sum(
                moved.vltot * total_charge(rho))
            return (local_ + potential.ehart + potential.etxc + epaw
                    + moved.ewald + moved.dispersion)

        zero_bs = jax.tree_util.tree_map(jnp.zeros_like, d_becsum)
        _, raw = jax.jvp(lambda p, b, r: mixed_state(p, b, r, zero_bs)[2],
                         (x, becsum_, rho_smooth), (dx, d_becsum, d_smooth))
        bs = tuple(None if a is None else a - b for a, b in zip(dbecsum, d_becsum))
        s = drho - raw - sub.augmented(jnp.zeros_like(drho), bs)
        gradient = jax.grad(whole, argnums=(0, 1, 2))
        return jax.jvp(gradient,
                       (x, becsum_, rho_smooth, jnp.zeros_like(s), zero_bs),
                       (dx, d_becsum, d_smooth, s, bs))

    def separable_us(moved, psi, weights, eigenvalues, multipliers):
        rows = jnp.arange(psi.shape[1])
        vkb = moved.projectors_at(rows)
        kinetic = _kinetic_energy(psi, moved.kinetic[rows], weights)
        nonlocal_, _ = _projector_energies(
            psi, vkb, moved.projectors.dij, moved.projectors.qq, weights,
            eigenvalues)
        norm = _constraint_energy(psi, vkb, moved.projectors.qq, multipliers)
        return kinetic + nonlocal_ - norm

    def pull_us(big, rowset, x, dx, psi, weights, eigenvalues, g_b, g_rho, dpsi,
                overlap, dlambda, dg_b, dg_rho):
        """The ``jvp`` of one chunk's pull-back ``d/dx [E_c + g . sums_c]`` along
        ``(e_p, dpsi + ort, dLambda, dg_b, dg_rho)``."""
        sub = local(big, rowset)
        ground = _ground_state_multipliers(weights, eigenvalues, psi.dtype)
        states = embed(psi, dpsi) + orthogonal(psi, overlap)

        def slope(position, s, lambdas, g_b, g_rho):
            def energy(p):
                moved = sub.at_positions(p)
                b, rho = sums(moved, s, weights)
                coupled = jnp.sum(g_rho * rho) + sum(
                    jnp.sum(g * part) for g, part in zip(g_b, b)
                    if part is not None)
                return separable_us(moved, s, weights, eigenvalues,
                                    lambdas) + coupled
            return jax.grad(energy)(position)

        return jax.jvp(slope, (x, psi, ground, g_b, g_rho),
                       (dx, states, dlambda, dg_b, dg_rho))[1]

    passes = {name: jax.jit(fn) for name, fn in (
        ("bare", bare), ("mixed", mixed), ("moved_density", moved_density),
        ("ldos", ldos), ("shift", shift), ("multipliers", multipliers),
        ("global_nc", global_nc), ("pull_nc", pull_nc), ("sums", sums_pass),
        ("forward", forward), ("global_us", global_us), ("pull_us", pull_us))}
    cached[1][key] = passes
    return passes


class StreamedDisplacementsAtQ:
    """The phonon at ``q``'s stores in host memory, and the walks over them.

    :func:`~defumat.response.phononq.dynamical_matrix_at_q`'s route when the
    store streams, with the same three stages the whole route has: the bare
    perturbations (:meth:`prepare`), the solves the self-consistent loop drives
    (:meth:`respond`, through :func:`~defumat.response.phononq.
    screening_loop_at_q`), and the two halves of the matrix. Every pass is one
    k-chunk's, on **two** row-subset calculations: the chunk's k-points and the
    same rows of the ``k + q`` list, which has the same length and order. Both
    are built from the calculation's own hoisted fields with the row leaves of
    their own list, so the passes close over the calculation alone and a second
    ``q`` reuses them.

    ``solver`` is the whole set's solver at ``k``, built on the host store;
    ``states_kq`` the ``k + q`` states, a host store from
    :func:`~defumat.scf.streaming.stream_states`.
    """

    def __init__(self, calculation, calculation_kq, solver, hamiltonians_kq,
                 eigenvalues_kq, states_kq, q_cart, positions):
        self.calculation = calculation
        self.calculation_kq = calculation_kq
        self.solver = solver
        self.hamiltonians = solver.hamiltonians
        # **Without the smallest sphere's plane-wave count**, ``min_k npw`` of
        # the ``k + q`` spheres, which is static and bounds only the Davidson
        # subspace (``Hamiltonian.npw``): it can differ from one ``q`` to the
        # next, and kept it would compile the solve pass again at every ``q`` of
        # a dispersion for nothing the solve reads.
        self.hamiltonians_kq = tuple(dataclasses.replace(h, npw=None)
                                     for h in hamiltonians_kq)
        keep = solver.psi.shape[2]
        self.keep = keep
        self.states_kq = states_kq
        self.eigenvalues_kq = np.asarray(eigenvalues_kq)[:, :, :keep]
        self.q_cart = jnp.asarray(np.asarray(q_cart, dtype=float))
        self.positions = jnp.asarray(positions)
        self.nat = int(self.positions.shape[0])
        self.modes = [(atom, cart) for atom in range(self.nat) for cart in range(3)]
        self.chunks = list(k_chunks(calculation.system.kpoints.nk,
                                    calculation.k_batch))
        self.passes = _phonon_q_passes(calculation, (keep, solver.occupied_counts))
        self.gamma_passes = _phonon_passes(
            calculation, (solver.nocc, solver.occupied_counts, solver.smearing))
        self.big = hoisted(calculation)
        self.scalars = solver.scalars()
        # ``D`` per channel, k-independent: the whole route reads it off the
        # ``k + q`` Hamiltonians, and so does this.
        self.dij = tuple(h.coefficients for h in self.hamiltonians_kq)
        #: The loop's ``3 nat`` complex grids in host memory: always, since the
        #: phonon at ``q`` runs without symmetry (``require_a_two_sphere_regime``).
        self.host_fields = True
        nspin, nk = solver.psi.shape[:2]
        ndim = np.shape(states_kq)[-1]
        shape = (self.nat, 3, nspin, nk, keep, ndim)
        dtype = np.dtype(solver.psi.dtype)
        self.bare = np.zeros(shape, dtype)
        self.dpsi = np.zeros(shape, dtype)
        self.iterations = 0
        self.solves = 0

    def _tangent(self, atom: int, cart: int):
        tangent = np.zeros(self.positions.shape)
        tangent[atom, cart] = 1.0
        return jnp.asarray(tangent, dtype=self.positions.dtype)

    def _arguments(self, rows, live, threshold=None):
        """The leading arguments of the solve pass, for one chunk, with
        ``threshold`` in the traced scalars when a schedule sets one."""
        return (self.big, row_leaves(self.calculation, rows),
                row_leaves(self.calculation_kq, rows), self.hamiltonians,
                self.hamiltonians_kq, _rows_of(self.solver.psi, rows),
                _rows_of(self.states_kq[:, :, :self.keep], rows),
                jnp.asarray(self.eigenvalues_kq[:, rows]),
                self.solver.chunk_arrays(rows, live),
                scalars_at(self.scalars, threshold))

    def prepare(self) -> None:
        """``dV_bare_q/du |psi_k>`` per mode, on the ``k + q`` sphere, into the host store."""
        for rows, live in self.chunks:
            leaves = row_leaves(self.calculation, rows)
            leaves_kq = row_leaves(self.calculation_kq, rows)
            psi = _rows_of(self.solver.psi, rows)
            written = rows[:live]
            for atom, cart in self.modes:
                bare = self.passes["bare"](
                    self.big, leaves, leaves_kq, psi, self.positions,
                    self._tangent(atom, cart), self.q_cart, self.dij)
                self.bare[atom, cart][:, written] = np.asarray(bare)[:, :live]

    def respond(self, dvscf, include_induced: bool, threshold=None, modes=None):
        """One iteration's solves: the complex response density per mode,
        finished. Each chunk starts from its previous solution in the host
        store (zero before the first pass). ``modes`` is the ``(atom, cart)``
        pairs still being solved, every one when ``None``; ``threshold`` this
        pass's CG threshold, one value or one per mode."""
        from defumat.response.phononq import _levels

        modes = self.modes if modes is None else list(modes)
        levels = _levels(threshold, len(modes))
        totals = [None] * len(modes)
        worst = [0] * len(modes)
        for rows, live in self.chunks:
            arguments = self._arguments(rows, live)
            written = rows[:live]
            for index, (atom, cart) in enumerate(modes):
                dv = (dvscf[atom, cart] if include_induced
                      else np.zeros_like(dvscf[atom, cart]))
                dpsi, steps, total = self.passes["respond"](
                    *arguments[:-1], scalars_at(self.scalars, levels[index]),
                    _rows_of(self.bare[atom, cart], rows),
                    jax.device_put(dv), _rows_of(self.dpsi[atom, cart], rows))
                self.dpsi[atom, cart][:, written] = np.asarray(dpsi)[:, :live]
                # Each mode's share of the sum over k is added on the host, in
                # the same order, so the ``P`` accumulators stay off the device.
                total = np.asarray(total)
                totals[index] = total if totals[index] is None else totals[index] + total
                worst[index] = max(worst[index], int(np.max(np.asarray(steps))))
        self.iterations += sum(worst)
        self.solves += len(modes)
        return [np.asarray(self._finish(jax.device_put(total))) for total in totals]

    def _finish(self, total):
        """The whole route's own finish (normalise, lift to the dense grid), on
        the summed chunks: it reads nothing but the calculation's grids."""
        from defumat.response.phononq import TwoSphereSolver

        holder = TwoSphereSolver.__new__(TwoSphereSolver)
        holder.calculation = self.calculation
        return TwoSphereSolver.finish_response_at_q(holder, total)

    def frozen_force_constants(self, density) -> np.ndarray:
        """``(3 nat, 3 nat)``: the frozen Hessian at ``Gamma`` on the occupied block,
        walked -- :func:`~defumat.response.phononq.frozen_force_constants` before
        its Ewald swap. The Gamma route's two assembly passes with zero tangents."""
        density = jnp.asarray(density)
        zero = jnp.zeros_like(density)
        rowset = row_leaves(self.calculation, self.chunks[0][0])
        matrix = np.zeros((self.nat, 3, self.nat, 3))
        for atom, cart in self.modes:
            tangent = self._tangent(atom, cart)
            column = np.asarray(self.gamma_passes["global_nc"](
                self.big, rowset, self.positions, tangent, density, zero))
            for rows, live in self.chunks:
                arrays = self.solver.chunk_arrays(rows, live)
                psi = _rows_of(self.solver.psi, rows)
                column = column + np.asarray(self.gamma_passes["pull_nc"](
                    self.big, row_leaves(self.calculation, rows), self.positions,
                    tangent, psi, arrays["weights"], arrays["weights"],
                    arrays["eigenvalues"], jnp.zeros_like(psi)))
            matrix[atom, cart] = column
        return matrix.reshape(3 * self.nat, 3 * self.nat)

    def response_force_constants(self) -> np.ndarray:
        """``2 sum_kn w <dpsi_i| dV_bare_j |psi>`` from the two host stores --
        :func:`~defumat.response.phononq.response_force_constants` as one Gram
        product. The stores hold real rows only, so there is no padding here."""
        modes = 3 * self.nat
        weights = np.asarray(self.solver.weights)
        left = (np.conj(self.dpsi) * weights[None, None, ..., None]).reshape(modes, -1)
        return 2.0 * left @ self.bare.reshape(modes, -1).T


def _phonon_q_passes(calculation, key) -> dict:
    """The phonon at ``q``'s compiled passes, built once and cached on the calculation.

    ``key`` is the solved block's width and the counts per channel. ``q`` is an
    argument, and the ``k + q`` sphere arrives as row leaves, so nothing here
    closes over one ``q``'s calculation.
    """
    cached = calculation.__dict__.get("_streamed_response")
    if cached is None or cached[0] is not calculation:
        cached = (calculation, {})
        calculation._streamed_response = cached
    key = ("phonon-q",) + tuple(key)
    if key in cached[1]:
        return cached[1][key]
    from defumat.response.phononq import (
        TwoSphereSolver, applied_at_q, induced_perturbation_at_q,
    )

    keep, counts = key[1:]

    def local(big, rowset):
        return with_rows(with_hoisted(calculation, big), rowset)

    def bare(big, rowset, rowset_kq, psi, x, dx, q_cart, dij):
        sub, sub_kq = local(big, rowset), local(big, rowset_kq)
        return jax.jvp(
            lambda position: applied_at_q(sub, sub_kq, psi, position, q_cart, dij),
            (x,), (dx,))[1]

    def respond(big, rowset, rowset_kq, hamiltonians, hamiltonians_kq, psi,
                psi_kq, eigenvalues_kq, arrays, scalars, bare_c, dv, start_c):
        """``dpsi`` on one chunk, the worst iteration count per channel, and the
        chunk's share of the response density's sum over k. ``start_c`` is the
        chunk's previous solution, the CG's first iterate."""
        sub, sub_kq = local(big, rowset), local(big, rowset_kq)
        base = SternheimerSolver.on_chunk(
            sub, sub.restricted_hamiltonians(hamiltonians), psi, arrays, scalars,
            nocc=keep, occupied_counts=counts, smearing=None)
        two = TwoSphereSolver(base, sub_kq.restricted_hamiltonians(hamiltonians_kq),
                              psi_kq, eigenvalues_kq, sub_kq)
        induced = induced_perturbation_at_q(sub, sub_kq, dv)

        def perturbation(states, ik, spin):
            return bare_c[spin][ik] + induced(states, ik, spin)

        dpsi, steps, _ = two.solve_arrays(perturbation, start=start_c)
        return dpsi, steps, two.response_parts_at_q(dpsi)

    passes = {name: jax.jit(fn) for name, fn in (
        ("bare", bare), ("respond", respond))}
    cached[1][key] = passes
    return passes
