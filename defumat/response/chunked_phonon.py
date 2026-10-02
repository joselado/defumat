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

**What it does not change**: the ``P`` dense-grid fields the loop carries
(``dvscf``, the induced potential, the symmetrised response, ``drhous`` and the
core term), which are ``5 P nspin_mag`` grids on the device and are the next
lever for a subset of a large cell; and a response handed in through
``response=``, a strained calculation and a k-point pool, which take the whole-k
route or are refused before this is reached.
"""

from __future__ import annotations

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
from defumat.response.sternheimer import SternheimerSolver, local_perturbation
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

    # -- the chunk's arguments -------------------------------------------------

    def _tangent(self, atom: int, cart: int):
        """The unit displacement, built on the host so that no index is a
        constant of anything compiled."""
        tangent = np.zeros(self.positions.shape)
        tangent[atom, cart] = 1.0
        return jnp.asarray(tangent, dtype=self.positions.dtype)

    def _arguments(self, rows, live):
        """The leading arguments of the solver passes, for one chunk."""
        return (self.big, row_leaves(self.calculation, rows), self.hamiltonians,
                _rows_of(self.solver.psi, rows),
                self.solver.chunk_arrays(rows, live), self.scalars, self.v_scf,
                self.ddd_paw)

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
        for index, (row, atom, cart) in enumerate(self.modes):
            d_becsum, _ = moved_sums[index]
            moved_density[row, cart] = self.passes["moved_density"](
                self.big, rowset, self.positions, self._tangent(atom, cart),
                self.density, tuple(self.becsum), d_becsum)
            moved_becsum[row, cart] = d_becsum
            o_becsum, o_smooth = ort_sums[index]
            ort_density[row, cart] = self.solver.finish_density(o_smooth, o_becsum)
            ort_becsum[row, cart] = o_becsum
        moved_stacked = _stack_modes(_symmetrize_modes(calculation, moved_density))
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
        fields, coefficients = [], []
        for row, _, cart in self.modes:
            dv = dvscf[row, cart] if include_induced else jnp.zeros_like(dvscf[row, cart])
            dddd = None if onecentre is None else (
                onecentre[row, cart] if include_induced
                else jnp.zeros_like(onecentre[row, cart]))
            fields.append(dv)
            coefficients.append(self.solver.perturbed_coefficients(dv, dddd)
                                if self.ultrasoft else None)
        return fields, coefficients

    def respond(self, dvscf, onecentre, include_induced: bool):
        """One iteration's solves: the finished, unsymmetrised response density
        per mode, and for PAW the raw ``becsum`` response."""
        solver = self.solver
        fields, coefficients = self._coefficients(dvscf, onecentre, include_induced)
        parts = [None] * len(self.modes)
        worst = [0] * len(self.modes)
        for rows, live in self.chunks:
            arguments = self._arguments(rows, live)
            written = rows[:live]
            for index, (row, _, cart) in enumerate(self.modes):
                dpsi, steps, _, chunk_parts = self.field_passes["respond"](
                    *arguments, _rows_of(self.bare[row, cart], rows),
                    fields[index], coefficients[index])
                self.dpsi[row, cart][:, written] = np.asarray(dpsi)[:, :live]
                parts[index] = _add(parts[index], chunk_parts)
                worst[index] = max(worst[index], int(np.max(np.asarray(steps))))
        self.iterations += sum(worst)
        self.solves += len(self.modes)
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
        return self.solver.fermi_level_shift(drho, levels=self._levels)

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
                    fields[index], coefficients[index],
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
                    row, cart, tangent, rowset, raw, drho[row, cart],
                    None if dbecsum is None else dbecsum[row, cart])
            else:
                column = np.asarray(self.passes["global_nc"](
                    self.big, rowset, self.positions, tangent, self.density,
                    drho[row, cart]))
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
