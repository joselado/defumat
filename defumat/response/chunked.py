"""The response to an electric field, a k-chunk at a time.

``GPU-MEMORY-NEXT.md`` item 2. The field response
(:func:`~defumat.response.efield.dielectric_tensor`) holds, besides the ground
state's occupied block, three bare perturbations ``P_c r_a|psi>`` and three
first-order wavefunctions ``dpsi_a`` for the whole of the loop -- each a
``(nspin, nk, nocc, ndim)`` array -- and it used to put a streamed store (a numpy
array in host memory, memory mode on a card) back on the device whole to get
them. That is seven times the store the SCF had just taken off the card, and
the CG's working set over the whole k axis on top.

Here every one of those arrays lives in host memory and the loop walks
:func:`~defumat.batching.k_chunks`, which is ``solve_e.f90``'s own loop over
k-points with ``get_buffer``/``save_buffer`` around each one:

* **the bare walk**, once: per chunk and per direction, the commutator
  ``[H - eps S, r_a]|psi>`` from the velocity operator's ``jvp`` over the
  chunk's own k-points (``at_rows``, so only that chunk's projector core is
  rebuilt), the projected solve, and on an ultrasoft or PAW dataset
  ``adddvepsi_us``'s tail;
* **the response walk**, once per self-consistent iteration: per chunk and per
  direction, the Sternheimer solve against bare plus induced, and the chunk's
  share of the response density as the ``jvp`` of the two raw sums over k the
  density is built from -- ``sum_band``'s smooth density and the unsymmetrised
  ``becsum``. Those are added over chunks and finished once (lifted to the dense
  grid and augmented), which is **exact** because every piece is linear: the
  tangent of a sum is the sum of the tangents, and ``to_dense`` and the
  augmentation charge are linear in what they are handed. Measured on ultrasoft
  silicon against the whole-set ``response_density``: 4.4e-16 on a density of
  1.23. The directional symmetrisation, the screening kernel and the mixing then
  act on whole-grid objects with no k index, exactly as before;
* **the assembly**, ``dielec.f90``'s contraction, on the host from the two stores.

**Every index is the chunk's own.** The solver, the Hamiltonians, the induced
perturbation and the velocity operator all run on ``Calculation.at_rows`` of the
chunk, so the ``ik`` that indexes a chunk's states is the one that indexes its
FFT tables and projectors; the Hamiltonians are the whole set's with their
k-indexed fields sliced (``restricted_hamiltonians``), so ``newd`` is not redone
per chunk, and ``int3`` is computed once per direction and iteration for the
same reason. **Two numbers are the whole set's and must be**: the level shift
``alpha_pv`` and the gap check at the occupied cut (``SternheimerSolver.
scalars``). The padded rows of a short last chunk carry zero weight and are
never written back.

**One program per pass, not one per chunk.** The two walks are module-level
``jax.jit`` functions taking the chunk's arrays as arguments -- the chunked
force's arrangement (:mod:`defumat.forces.chunked`), whose ``row_leaves`` and
``with_rows`` this reuses -- cached on the calculation. Building the solve
afresh per chunk through :func:`~defumat.eager.compiled` traced it each time:
0.11 s of tracing against 0.04 s of work per k-point on ultrasoft silicon, which
on a 216-point mesh is a quarter of an hour.

**The Born charges are walked the same way** (:meth:`StreamedField.
born_charges`, :func:`_born_passes`). They are a ``jvp`` of the force along the
field's response (:mod:`defumat.response.born`), and the force is already split
into a sum over chunks and a function of the whole sums
(:mod:`defumat.forces.chunked`); the split carries one ``jvp`` through each of
its passes. Against the whole-k route: 3.5e-13 on ultrasoft AlAs, the polar cell
where the coupling of a chunk's projector occupations to the whole-cell energy
matters -- dropping it moves ``Z*`` by 39 -- and 8.8e-15 and 4.4e-15 on
norm-conserving and PAW silicon, and 1.6e-13 on ultrasoft AlAs's wedge, where the
full-zone shift between the walks is not zero.

**So is the piezoelectric tensor** (:meth:`StreamedField.piezoelectric`): the
same ``jvp`` of the *stress* along the same response, which is the Born split
with ``at_strain`` where it has ``at_positions`` (``_born_passes(kind="strain")``)
and no frozen polarization term. The whole route held the k axis on one
forward-over-reverse tape, 31.5 GiB of temporaries on ultrasoft AlAs at 64
k-points; walked, the tape is one chunk's.

**What still goes whole, by name.** A caller asking for the internals without
``streamed_internals`` (the third derivatives, which read ``bare``, ``dpsi`` and
the solver as device arrays), a calculation carrying a strained ``_kcart``, and a
k-point pool's store take the whole-k route or are refused before this is
reached.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from defumat.basis.interpolate import to_dense
from defumat.batching import k_chunks, resolve_field_batch
from defumat.forces.chunked import _move, _rows_of, row_leaves, with_rows
from defumat.forces.energy import (
    _constraint_energy, _kinetic_energy, _projector_energies,
    _spinor_constraint_energy, _spinor_projector_energies, hoisted, with_hoisted,
)
from defumat.scf.potential import total_charge
from defumat.response.sternheimer import SternheimerSolver, local_perturbation, scalars_at
from defumat.response.velocity import VelocityOperator

__all__ = ["StreamedField"]


def _add(total, part):
    return part if total is None else jax.tree_util.tree_map(jnp.add, total, part)


def _passes(calculation, key) -> dict:
    """The two compiled walks for ``calculation``, built once and cached on it.

    Keyed on the calculation's identity, as :func:`~defumat.forces.chunked.
    _compiled` keys the stress: the passes close over everything about the
    calculation that is not an argument (the structure, the cell, the
    symmetry), so a moved or strained copy -- which ``copy.copy`` gives the same
    ``__dict__`` entry -- builds its own rather than reusing these. ``key`` is
    the solver's static configuration: the occupied-block width, the counts per
    channel and the smearing.
    """
    cached = calculation.__dict__.get("_streamed_response")
    if cached is None or cached[0] is not calculation:
        cached = (calculation, {})
        calculation._streamed_response = cached
    if key in cached[1]:
        return cached[1][key]
    from defumat.response.efield import _solve_stored, ultrasoft_position

    nocc, counts, smearing = key

    def solver_on(big, rowset, hamiltonians, psi, arrays, scalars, v_scf, ddd_paw):
        sub = with_rows(with_hoisted(calculation, big), rowset)
        return SternheimerSolver.on_chunk(
            sub, sub.restricted_hamiltonians(hamiltonians), psi, arrays, scalars,
            nocc=nocc, occupied_counts=counts, smearing=smearing, v_scf=v_scf,
            ddd_paw=ddd_paw,
        )

    def bare(big, rowset, hamiltonians, psi, arrays, scalars, v_scf, ddd_paw,
             kcart, direction, dipole):
        """``(P_c^+ r_a|psi>, P_c r_a|psi>, d(vkb)/dk_a)`` on one chunk."""
        solver = solver_on(big, rowset, hamiltonians, psi, arrays, scalars,
                           v_scf, ddd_paw)
        velocity = VelocityOperator(solver.calculation, v_scf, ddd_paw, kcart=kcart)
        derivative, overlap = velocity.both(solver.psi, direction)
        commutator = -1j * (derivative - solver.eigenvalues[..., None] * overlap)
        position = _solve_stored(solver, commutator)
        if dipole is None:
            return position, position, None
        projector_velocity = velocity.projectors(direction)
        return ultrasoft_position(
            solver.calculation, solver.hamiltonians, solver.psi, position, dipole,
            projector_velocity,
        ), position, projector_velocity

    def respond(big, rowset, hamiltonians, psi, arrays, scalars, v_scf, ddd_paw,
                bare_c, dv, coefficients, start_c):
        """``dpsi`` on one chunk, its worst iteration count and residual per
        channel, and the tangent of the chunk's raw ``(rho_smooth, becsum)``.
        ``start_c`` is the chunk's previous solution, the CG's first iterate."""
        solver = solver_on(big, rowset, hamiltonians, psi, arrays, scalars,
                           v_scf, ddd_paw)
        induced = local_perturbation(solver.calculation, dv, v_scf, ddd_paw,
                                     coefficients=coefficients)

        def perturbation(states, ik, spin):
            return bare_c[spin][ik] + induced(states, ik, spin)

        dpsi, steps, residual = solver.solve_arrays(perturbation, start=start_c)
        _, parts = jax.jvp(solver.density_parts, (solver.psi,), (dpsi,))
        return dpsi, steps, residual, parts

    def respond_many(big, rowset, hamiltonians, psi, arrays, scalars, v_scf,
                     ddd_paw, bare_c, dv, coefficients, start_c):
        """:func:`respond` for the three directions in one CG loop per k-point:
        ``bare_c``, ``dv`` and ``start_c`` carry a leading direction axis and
        ``coefficients`` is a tuple of three (or ``None``). The right-hand
        sides and the density tangents are built a direction at a time; only
        the solve is batched (:meth:`SternheimerSolver.solve_arrays_many`)."""
        solver = solver_on(big, rowset, hamiltonians, psi, arrays, scalars,
                           v_scf, ddd_paw)
        induced = [
            local_perturbation(solver.calculation, dv[axis], v_scf, ddd_paw,
                               coefficients=None if coefficients is None
                               else coefficients[axis])
            for axis in range(3)
        ]

        def perturbation(states, ik, spin):
            return jnp.stack([bare_c[axis][spin][ik] + induced[axis](states, ik, spin)
                              for axis in range(3)])

        dpsi, steps, residual = solver.solve_arrays_many(perturbation, start=start_c)
        parts = [jax.jvp(solver.density_parts, (solver.psi,), (dpsi[axis],))[1]
                 for axis in range(3)]
        return dpsi, steps, residual, parts

    passes = {"bare": jax.jit(bare), "respond": jax.jit(respond),
              "respond_many": jax.jit(respond_many)}
    cached[1][key] = passes
    return passes


def _stacked_rows(store, rows):
    """One chunk of a ``(3, nspin, nk, ...)`` store, the k axis the third."""
    return jnp.stack([_rows_of(store[axis], rows) for axis in range(store.shape[0])])


class StreamedField:
    """The field response's three stores in host memory, and the walks over them.

    The same four steps :func:`~defumat.response.efield.dielectric_tensor`'s
    whole-k route takes -- :meth:`prepare` the bare perturbations,
    :meth:`respond` once per iteration, :meth:`assemble` the tensor and
    :meth:`born_charges` -- so the self-consistent loop around them is one loop
    for both routes.

    ``solver`` is the whole set's :class:`~defumat.response.sternheimer.
    SternheimerSolver`, built on the host store: it supplies the eigenvalues,
    the weights, the masks and the level shift, and its states stay in host
    memory.
    """

    def __init__(self, calculation, solver, v_scf, dipole, keep_commutators: bool):
        self.calculation = calculation
        self.solver = solver
        self.v_scf = v_scf
        self.ddd_paw = solver.ddd_paw
        self.dipole = dipole
        self.hamiltonians = solver.hamiltonians
        self.chunks = list(k_chunks(calculation.system.kpoints.nk,
                                    calculation.k_batch))
        batch = calculation.k_batch
        self.batched_chunks = list(k_chunks(
            calculation.system.kpoints.nk,
            None if batch is None else max(1, batch // 3)))
        self.kcart = np.asarray(
            calculation.system.kpoints.cartesian(calculation.system.cell))
        self.passes = _passes(calculation, (solver.nocc, solver.occupied_counts,
                                            solver.smearing))
        self.big = hoisted(calculation)
        self.scalars = solver.scalars()
        shape = (3,) + tuple(solver.psi.shape)
        dtype = np.dtype(solver.psi.dtype)
        self.bare = np.zeros(shape, dtype)
        self.dpsi = np.zeros(shape, dtype)
        # ``P_c r|psi>`` before ``S`` and the augmentation dipole -- QE's
        # ``iucom``, which the Born charges read. The same array as ``bare`` on
        # a norm-conserving dataset, so it is stored only when they differ.
        self.commutators = (np.zeros(shape, dtype)
                            if keep_commutators and dipole is not None else None)
        self.iterations = 0
        self.solves = 0

    def _arguments(self, rows, live, threshold=None):
        """The leading arguments of both passes, for one chunk. ``threshold``
        replaces the solver's in the traced scalars, so a schedule reaches the
        compiled pass without recompiling it."""
        return (self.big, row_leaves(self.calculation, rows), self.hamiltonians,
                _rows_of(self.solver.psi, rows),
                self.solver.chunk_arrays(rows, live),
                scalars_at(self.scalars, threshold), self.v_scf, self.ddd_paw)

    def prepare(self) -> None:
        """``P_c^+ r_a|psi>`` for the three directions, into the host store."""
        for rows, live in self.chunks:
            arguments = self._arguments(rows, live)
            kcart = jnp.asarray(self.kcart[rows])
            written = rows[:live]
            for axis in range(3):
                position, commutator, _ = self.passes["bare"](
                    *arguments, kcart, jnp.asarray(np.eye(3)[axis]),
                    None if self.dipole is None else self.dipole[axis],
                )
                self.bare[axis][:, written] = np.asarray(position)[:, :live]
                if self.commutators is not None:
                    self.commutators[axis][:, written] = (
                        np.asarray(commutator)[:, :live])

    def respond(self, dvscf, onecentre, include_induced: bool, threshold=None):
        """One iteration's three solves: ``(drho, dbecsum)`` per direction.

        ``drho`` is finished (on the dense grid, augmented) and not
        symmetrised; ``dbecsum`` is the raw response, and is returned only for
        PAW, whose one-centre potential is built from it -- the same two
        objects the whole-k route's ``response_density`` and
        ``response_becsum`` return. Each chunk's solve starts from its previous
        solution in the host store, which is zero before the first pass, so one
        program serves every pass at the cost of one more ``(nspin, live, nocc,
        npwx)`` upload per solve; ``threshold`` is this pass's CG threshold.
        """
        solver = self.solver
        fields, coefficients = [], []
        for axis in range(3):
            dv = dvscf[axis] if include_induced else jnp.zeros_like(dvscf[axis])
            dddd = None if onecentre is None else (
                onecentre[axis] if include_induced
                else jnp.zeros_like(onecentre[axis]))
            fields.append(dv)
            # ``int3``: k-independent, so once per direction rather than per
            # chunk; ``None`` on a norm-conserving dataset.
            coefficients.append(solver.perturbed_coefficients(dv, dddd)
                                if self.calculation.is_ultrasoft else None)

        if resolve_field_batch():
            return self._respond_batched(fields, coefficients, onecentre, threshold)
        parts = [None, None, None]
        worst = [0, 0, 0]
        for rows, live in self.chunks:
            arguments = self._arguments(rows, live, threshold)
            written = rows[:live]
            for axis in range(3):
                dpsi, steps, _, chunk_parts = self.passes["respond"](
                    *arguments, _rows_of(self.bare[axis], rows), fields[axis],
                    coefficients[axis], _rows_of(self.dpsi[axis], rows),
                )
                self.dpsi[axis][:, written] = np.asarray(dpsi)[:, :live]
                parts[axis] = _add(parts[axis], chunk_parts)
                worst[axis] = max(worst[axis], int(np.max(np.asarray(steps))))
        self.iterations += sum(worst)
        self.solves += 3

        response = [solver.finish_density(*parts[axis]) for axis in range(3)]
        becsum_response = ([parts[axis][1] for axis in range(3)]
                           if onecentre is not None else [])
        return response, becsum_response

    def _respond_batched(self, fields, coefficients, onecentre, threshold):
        """:meth:`respond` with the three directions in one CG loop per k-point.

        Three directions' CG state are in flight where one was, so the k-chunk
        is a third of the SCF's (:attr:`batched_chunks`), which keeps the
        response's peak where the serial loop has it.
        """
        solver = self.solver
        stacked_fields = jnp.stack([jnp.asarray(f) for f in fields])
        stacked_coefficients = (None if coefficients[0] is None
                                else tuple(coefficients))
        parts = [None, None, None]
        worst = np.zeros(3, dtype=int)
        for rows, live in self.batched_chunks:
            arguments = self._arguments(rows, live, threshold)
            written = rows[:live]
            dpsi, steps, _, chunk_parts = self.passes["respond_many"](
                *arguments, _stacked_rows(self.bare, rows), stacked_fields,
                stacked_coefficients, _stacked_rows(self.dpsi, rows),
            )
            dpsi = np.asarray(dpsi)
            for axis in range(3):
                self.dpsi[axis][:, written] = dpsi[axis][:, :live]
                parts[axis] = _add(parts[axis], chunk_parts[axis])
            worst = np.maximum(worst, np.max(np.asarray(steps), axis=1))
        self.iterations += int(np.sum(worst))
        self.solves += 3

        response = [solver.finish_density(*parts[axis]) for axis in range(3)]
        becsum_response = ([parts[axis][1] for axis in range(3)]
                           if onecentre is not None else [])
        return response, becsum_response

    def overlaps(self) -> np.ndarray:
        """``sum_kn w Re <P_c r_i psi|dpsi_j>``, ``(3, 3)``, on the host."""
        weights = np.asarray(self.solver.weights)
        totals = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                overlap = np.einsum("skng,skng->skn", self.bare[i].conj(),
                                    self.dpsi[j])
                totals[i, j] = float(np.sum(weights * overlap.real))
        return totals

    def born_charges(self, dvscf, onecentre, wavefunctions, eigenvalues, weights,
                     density, becsum):
        """``Z*`` as :func:`~defumat.response.born.born_effective_charges` builds
        it, with the k axis walked rather than taped: see :func:`_born_passes`.

        ``wavefunctions`` is the whole store, every band, in host memory;
        ``weights`` and ``eigenvalues`` every band too, as the frozen-state
        functional wants them. Two walks over the chunks and one step between
        them:

        1. per chunk: the multipliers' response ``dLambda`` and the transcribed
           ``add_for_charges`` term per direction, the frozen polarization's
           derivative once, and the forward ``jvp`` of the two raw sums over k
           the mixed state is built from;
        2. on the whole sums: the offsets that make the raw state the converged
           one, the full-zone correction of the field's response
           (``_full_zone_field_response``), and the ``jvp`` of the global terms'
           gradient;
        3. per chunk: the ``jvp`` of each chunk's pull-back, with the states, the
           multipliers and the global cotangent all moving.
        """
        from defumat.response.born import frozen_polarization

        calculation = self.calculation
        solver = self.solver
        positions = jnp.asarray(calculation.system.structure.positions)
        natoms = positions.shape[0]
        weights = np.asarray(weights)
        eigenvalues = np.asarray(eigenvalues)
        if eigenvalues.ndim == 2:
            eigenvalues = eigenvalues[None]
        passes = _born_passes(calculation, (solver.nocc, solver.occupied_counts,
                                            solver.smearing))
        ultrasoft = calculation.is_ultrasoft
        fields, coefficients = [], []
        for axis in range(3):
            dddd = None if onecentre is None else onecentre[axis]
            fields.append(dvscf[axis])
            coefficients.append(solver.perturbed_coefficients(dvscf[axis], dddd)
                                if ultrasoft else None)

        def chunk_states(rows, live):
            psi = _rows_of(wavefunctions, rows)
            w = np.array(weights[:, rows])
            w[:, live:] = 0.0
            return psi, jnp.asarray(w), jnp.asarray(eigenvalues[:, rows])

        # 1. The first walk.
        frozen = np.asarray(jax.jacfwd(
            lambda pos: frozen_polarization(calculation, pos, None, None, None)
        )(positions))
        constraint = np.zeros((3, natoms, 3), dtype=complex)
        multipliers = {}
        raw = tangents = None
        for rows, live in self.chunks:
            arguments = self._arguments(rows, live)
            psi, w, eps = chunk_states(rows, live)
            if self.dipole is not None:
                frozen = frozen + np.asarray(passes["frozen"](
                    arguments[0], arguments[1], positions, psi, w,
                    jnp.asarray(self.kcart[rows]), self.v_scf, self.ddd_paw,
                    self.dipole))
            chunk_tangents = []
            for axis in range(3):
                dlambda, sandwich = passes["extras"](
                    *arguments, positions, w, _rows_of(self.bare[axis], rows),
                    fields[axis], coefficients[axis],
                    _rows_of(self.commutators[axis] if self.commutators is not None
                             else self.bare[axis], rows),
                )
                # In host memory between the walks: ``nbnd^2`` per k-point and
                # direction, which grows with the mesh (2.65 MB on the card at
                # 216 k-points of eight-atom silicon, measured above the SCF's).
                multipliers[(rows[0], axis)] = np.asarray(dlambda)
                constraint[axis] += np.asarray(sandwich)
                sums, derivative = passes["forward"](
                    arguments[0], arguments[1], positions, psi, w,
                    _rows_of(self.dpsi[axis], rows))
                if axis == 0:
                    raw = _add(raw, sums)
                chunk_tangents.append(derivative)
            tangents = (chunk_tangents if tangents is None
                        else [_add(a, b) for a, b in zip(tangents, chunk_tangents)])

        # 2. The whole sums: the offsets and the full-zone shifts
        #    (:func:`_full_zone_shifts`).
        shifts, becsum_shifts, offsets = _full_zone_shifts(
            calculation, raw, tangents, density, becsum)
        raw_becsum, raw_smooth, becsum_offset, density_offset = offsets
        rowset = row_leaves(calculation, self.chunks[0][0])
        globals_ = [
            passes["global"](self.big, rowset, positions, raw_becsum, raw_smooth,
                             becsum_offset, density_offset, tangents[axis][0],
                             tangents[axis][1], shifts[axis], becsum_shifts[axis])
            for axis in range(3)
        ]

        # 3. The second walk: each chunk's pull-back, differentiated.
        columns = [np.asarray(slope) for (_, (slope, _, _)) in globals_]
        for rows, live in self.chunks:
            psi, w, eps = chunk_states(rows, live)
            rowset = row_leaves(calculation, rows)
            for axis in range(3):
                (_, g_b, g_rho), (_, dg_b, dg_rho) = globals_[axis]
                columns[axis] = columns[axis] + np.asarray(passes["pull"](
                    self.big, rowset, positions, psi, w, eps, g_b, g_rho,
                    _rows_of(self.dpsi[axis], rows),
                    jax.device_put(multipliers[(rows[0], axis)]),
                    dg_b, dg_rho))

        charges = np.zeros((natoms, 3, 3))
        for axis in range(3):
            charges[:, axis, :] = (frozen[axis] - columns[axis]
                                   - np.real(constraint[axis]))
        return calculation.symmetrize_atom_tensor(charges)

    def piezoelectric(self, dvscf, onecentre, wavefunctions, eigenvalues, weights,
                      density, becsum) -> np.ndarray:
        """``(3, 3, 3)``: :func:`~defumat.response.piezo.clamped_ion_piezoelectric`'s
        columns, walked, before its ``-1/Omega`` and its rank-3 average.

        :meth:`born_charges` with the strain as the coordinate: the same three
        steps on :func:`_born_passes` at ``kind = "strain"``, and no frozen
        polarization term. ``column[i]`` is the ``jvp`` of ``dE/d(eps)`` along
        the field's response ``i`` plus ``add_for_charges`` in the strain
        coordinate (:func:`~defumat.response.piezo.constraint_strain_term`).

        **A norm-conserving dataset takes the whole route's functional, not the
        augmented one**: no multipliers' tangent, no constraint term and no
        full-zone shift, exactly as :func:`~defumat.response.piezo.
        clamped_ion_piezoelectric` skips them. With ``S = 1`` the first two are
        contracted with a ``dS/d(eps)`` that is identically zero, and the shift
        is linear in a per-k tangent there, so the rank-3 average completes it;
        so the two functionals have the same column, and mirroring the whole
        route is what makes the comparison against it an identity.
        """
        calculation = self.calculation
        solver = self.solver
        strain = jnp.zeros((3, 3))
        weights = np.asarray(weights)
        eigenvalues = np.asarray(eigenvalues)
        if eigenvalues.ndim == 2:
            eigenvalues = eigenvalues[None]
        augmented = bool(calculation.is_ultrasoft)
        passes = _born_passes(calculation, (solver.nocc, solver.occupied_counts,
                                            solver.smearing), kind="strain")
        fields, coefficients = [], []
        if augmented:
            for axis in range(3):
                dddd = None if onecentre is None else onecentre[axis]
                fields.append(dvscf[axis])
                coefficients.append(solver.perturbed_coefficients(dvscf[axis], dddd))

        def chunk_states(rows, live):
            psi = _rows_of(wavefunctions, rows)
            w = np.array(weights[:, rows])
            w[:, live:] = 0.0
            return psi, jnp.asarray(w), jnp.asarray(eigenvalues[:, rows])

        # 1. The first walk: per chunk the multipliers' response and the strain
        #    sandwich (augmented only), and the forward ``jvp`` of the raw sums.
        constraint = np.zeros((3, 3, 3), dtype=complex)
        multipliers = {}
        raw = tangents = None
        for rows, live in self.chunks:
            arguments = self._arguments(rows, live)
            psi, w, _ = chunk_states(rows, live)
            chunk_tangents = []
            for axis in range(3):
                if augmented:
                    dlambda, sandwich = passes["extras"](
                        *arguments, strain, w, _rows_of(self.bare[axis], rows),
                        fields[axis], coefficients[axis],
                        _rows_of(self.commutators[axis]
                                 if self.commutators is not None
                                 else self.bare[axis], rows),
                    )
                    # Per chunk, so a padded row's ``dLambda`` is the zero its
                    # zero weight makes it, not a repeat of its original's.
                    multipliers[(rows[0], axis)] = np.asarray(dlambda)
                    # The whole route differentiates along the six symmetric
                    # tangents ``(E_ab + E_ba)/2``; the nine columns of the
                    # Jacobian give the same numbers symmetrised.
                    sandwich = np.asarray(sandwich)
                    constraint[axis] += 0.5 * (sandwich + sandwich.T)
                sums, derivative = passes["forward"](
                    arguments[0], arguments[1], strain, psi, w,
                    _rows_of(self.dpsi[axis], rows))
                if axis == 0:
                    raw = _add(raw, sums)
                chunk_tangents.append(derivative)
            tangents = (chunk_tangents if tangents is None
                        else [_add(a, b) for a, b in zip(tangents, chunk_tangents)])

        # 2. The whole sums.
        shifts, becsum_shifts, offsets = _full_zone_shifts(
            calculation, raw, tangents, density, becsum)
        raw_becsum, raw_smooth, becsum_offset, density_offset = offsets
        if not augmented:
            shifts = jnp.zeros_like(shifts)
            becsum_shifts = [jax.tree_util.tree_map(jnp.zeros_like, b)
                             for b in becsum_shifts]
        rowset = row_leaves(calculation, self.chunks[0][0])
        globals_ = [
            passes["global"](self.big, rowset, strain, raw_becsum, raw_smooth,
                             becsum_offset, density_offset, tangents[axis][0],
                             tangents[axis][1], shifts[axis], becsum_shifts[axis])
            for axis in range(3)
        ]

        # 3. The second walk: each chunk's pull-back, differentiated.
        columns = [np.asarray(slope) for (_, (slope, _, _)) in globals_]
        nbnd = weights.shape[-1]
        for rows, live in self.chunks:
            psi, w, eps = chunk_states(rows, live)
            rowset = row_leaves(calculation, rows)
            for axis in range(3):
                (_, g_b, g_rho), (_, dg_b, dg_rho) = globals_[axis]
                dlambda = (multipliers[(rows[0], axis)] if augmented else
                           np.zeros((w.shape[0], len(rows), nbnd, nbnd),
                                    np.dtype(psi.dtype)))
                columns[axis] = columns[axis] + np.asarray(passes["pull"](
                    self.big, rowset, strain, psi, w, eps, g_b, g_rho,
                    _rows_of(self.dpsi[axis], rows), jax.device_put(dlambda),
                    dg_b, dg_rho))
        return np.stack([columns[axis] + np.real(constraint[axis])
                         for axis in range(3)])


def _full_zone_shifts(calculation, raw, tangents, density, becsum):
    """The step between the two walks: ``(shifts, becsum_shifts, offsets)``.

    ``raw`` is the forward walk's ``(becsum, rho_smooth)`` summed over k and
    ``tangents`` its three tangents along the field's responses. The offsets make
    the raw, unsymmetrised mixed state the converged, symmetrised one in *value*
    (``_raw_mixed_state``); the shifts make the field's response the full-zone
    one in *tangent* (``_full_zone_field_response``), with the augmentation
    charge's share of the ``becsum`` shift taken out of the density's because the
    density is built from ``becsum`` -- ``augmented`` is linear in it, so that
    share is ``augmented(0, shift)`` and needs no states. ``offsets`` is
    ``(raw_becsum, raw_smooth, becsum_offset, density_offset)``.
    """
    from defumat.response.born import _full_zone_becsum_response

    raw_becsum, raw_smooth = raw
    becsum_offset = tuple(
        None if value is None else jnp.asarray(value) - part
        for value, part in zip(becsum, raw_becsum))
    parts = tuple(None if part is None else part + offset
                  for part, offset in zip(raw_becsum, becsum_offset))
    smooth, dense = calculation.basis.smooth, calculation.basis.dense
    density_offset = jnp.asarray(density) - calculation.augmented(
        to_dense(raw_smooth, smooth, dense), parts)
    responses = jnp.stack([
        calculation.augmented(to_dense(d_smooth, smooth, dense), d_becsum)
        for d_becsum, d_smooth in tangents])
    becsum_shifts = _full_zone_becsum_response(
        calculation, [d_becsum for d_becsum, _ in tangents])
    through_becsum = jnp.stack([
        calculation.augmented(jnp.zeros_like(responses[0]), shifts)
        for shifts in becsum_shifts])
    shifts = (calculation.symmetrize_directional(responses) - responses
              - through_becsum)
    return shifts, becsum_shifts, (raw_becsum, raw_smooth, becsum_offset,
                                   density_offset)


def _born_passes(calculation, key, kind: str = "positions") -> dict:
    """The Born charges' compiled passes: ``forces/chunked.py``'s split, one ``jvp`` up.

    ``kind`` is the coordinate the energy is differentiated in, as in
    :func:`~defumat.forces.chunked.chunked_gradient`: ``"positions"`` for the
    Born charges, ``"strain"`` for the piezoelectric tensor, which is the same
    ``jvp`` of the *stress* along the same response
    (:meth:`StreamedField.piezoelectric`). Every pass moves the chunk's
    calculation through the one mover, so the strain reaches what it reaches in
    the chunked stress: the cell, the volume in the raw smooth density, the
    radial transforms, Ewald and the local and core terms.

    ``Z*[:, i, :]`` is, besides two transcribed terms, the ``jvp`` of the force
    -- ``grad`` of the frozen-state energy in the positions ``x`` -- along the
    field's response: the states move by ``dpsi_i``, the multipliers by
    ``dLambda_i``, and the mixed state by the full-zone shifts
    (:func:`~defumat.response.born.born_effective_charges`). The force is split
    as the chunked force splits it,

        E(x) = sum_c E_c(x, psi_c, Lambda_c) + E_glob(x, b, rho_s; shift, bshift)
        dE/dx = g_x + sum_c d/dx [E_c + g_b . b_c(x) + g_rho . rho_c],

    with ``b`` and ``rho_s`` the raw sums over k and ``(g_x, g_b, g_rho)`` the
    gradient of ``E_glob`` at them, and its ``jvp`` along the response is then
    term by term:

    * **forward**: per chunk, ``(b_c, rho_c)`` and their tangents along
      ``dpsi_c``, added over chunks;
    * **global**: the ``jvp`` of ``grad E_glob`` at the whole sums, along
      ``(0, db, drho, shift, bshift)`` -- the shifts carry only a tangent, their
      value is zero;
    * **pull**: per chunk, the ``jvp`` of ``d/dx [E_c + g_b . b_c + g_rho .
      rho_c]`` along ``(dpsi_c, dLambda_c, dg_b, dg_rho)``.

    The column is ``dg_x + sum_c d(pull_c)``. ``E_c`` differs from the chunked
    force's separable part in one term: with a *matrix* multiplier the
    constraint is ``Tr[Lambda (<psi|S|psi> - 1)]`` and replaces both the overlap
    and the norm terms (``energy_at``). ``E_glob`` differs in having no
    symmetrisation: the raw sums plus the constant offsets that make their value
    the converged state's, plus the two shifts. The ``g_b . db_c/dx`` cross term
    is the ultrasoft ``int dvscf drho_us`` piece -- zero for a norm-conserving
    dataset and on silicon's symmetric residue, so the polar ultrasoft cell is
    the one that would see it dropped.
    """
    cached = calculation.__dict__.get("_streamed_response")
    if cached is None or cached[0] is not calculation:
        cached = (calculation, {})
        calculation._streamed_response = cached
    key = ("born", kind) + tuple(key)
    if key in cached[1]:
        return cached[1][key]
    from defumat.response.born import (
        _ground_state_multipliers, _multiplier_response, _position_operator,
        constraint_sandwich_at, frozen_polarization,
    )

    nocc, counts, smearing = key[2:]

    def local(big, rowset):
        return with_rows(with_hoisted(calculation, big), rowset)

    def moved_to(big, rowset, x):
        return _move(local(big, rowset), kind, x)

    def sums(moved, psi, weights):
        rows = jnp.arange(psi.shape[1])
        return (moved.becsum(psi, weights, rows=rows, symmetrize=False),
                moved.smooth_density(psi, weights, rows=rows))

    def separable(moved, psi, weights, eigenvalues, multipliers):
        rows = jnp.arange(psi.shape[1])
        vkb = moved.projectors_at(rows)
        if moved.noncolin:
            kinetic = _kinetic_energy(psi, moved.state_kinetic[rows], weights)
            nonlocal_, _ = _spinor_projector_energies(
                psi, vkb, moved.dvan_so, moved.qq_so, weights, eigenvalues)
            norm = _spinor_constraint_energy(psi, vkb, moved.qq_so, multipliers)
        else:
            kinetic = _kinetic_energy(psi, moved.kinetic[rows], weights)
            nonlocal_, _ = _projector_energies(
                psi, vkb, moved.projectors.dij, moved.projectors.qq, weights,
                eigenvalues)
            norm = _constraint_energy(psi, vkb, moved.projectors.qq, multipliers)
        return kinetic + nonlocal_ - norm

    def embed(psi, dpsi):
        return jnp.zeros_like(psi).at[:, :, :nocc].set(dpsi)

    def forward(big, rowset, x, psi, weights, dpsi):
        """``((b_c, rho_c), (db_c, drho_c))`` along the field's response."""
        moved = moved_to(big, rowset, x)
        return jax.jvp(lambda states: sums(moved, states, weights), (psi,),
                       (embed(psi, dpsi),))

    def extras(big, rowset, hamiltonians, psi, arrays, scalars, v_scf, ddd_paw,
               x, weights, bare_c, dv, coefficients, commutator):
        """``dLambda`` and the transcribed ``add_for_charges`` term on one chunk."""
        sub = local(big, rowset)
        solver = SternheimerSolver.on_chunk(
            sub, sub.restricted_hamiltonians(hamiltonians), psi, arrays, scalars,
            nocc=nocc, occupied_counts=counts, smearing=smearing, v_scf=v_scf,
            ddd_paw=ddd_paw)
        induced = local_perturbation(sub, dv, v_scf, ddd_paw,
                                     coefficients=coefficients)

        def perturbation(states, ik, spin):
            return bare_c[spin][ik] + induced(states, ik, spin)

        # ``nbnd`` is every band's, the width the multipliers have in the
        # frozen-state functional; ``weights`` carries every band.
        dlambda = _multiplier_response(solver, perturbation, weights,
                                       weights.shape[-1], nocc)
        if not sub.is_ultrasoft:
            return dlambda, jnp.zeros(x.shape, dtype=psi.dtype)
        return dlambda, _jacobian_by_columns(
            lambda y: constraint_sandwich_at(_move(sub, kind, y), solver.psi,
                                             weights[:, :, :nocc], commutator), x)

    def frozen(big, rowset, x, psi, weights, kcart, v_scf, ddd_paw, dipole):
        """``d(Omega P)/du`` of one chunk's augmentation charge, ``(3, nat, 3)``."""
        sub = local(big, rowset)
        velocity = VelocityOperator(sub, v_scf, ddd_paw, kcart=kcart)
        derivatives = [velocity.projectors(direction) for direction in jnp.eye(3)]
        operator = _position_operator(sub, derivatives, dipole)
        return _jacobian_by_columns(
            lambda pos: frozen_polarization(sub, pos, psi, weights, operator)
            - frozen_polarization(sub, pos, psi, weights, None), x)

    def global_(big, rowset, x, becsum_, rho_smooth, becsum_offset,
                density_offset, d_becsum, d_smooth, shift, becsum_shift):
        """``(grad, d grad)`` of the terms that are a function of the whole sums."""
        def whole(x, b, rho_s, s, bs):
            moved = moved_to(big, rowset, x)
            parts = tuple(
                None if part is None else part + offset + extra
                for part, offset, extra in zip(b, becsum_offset, bs))
            smooth, dense = moved.basis.smooth, moved.basis.dense
            rho = (moved.augmented(to_dense(rho_s, smooth, dense), parts)
                   + density_offset + s)
            potential = moved.potential(rho)
            epaw, _ = moved.onecenter(parts)
            volume = moved.system.cell.volume
            local_ = volume / rho[0].size * jnp.sum(
                moved.vltot * total_charge(rho))
            return (local_ + potential.ehart + potential.etxc + epaw
                    + moved.ewald + moved.dispersion)

        gradient = jax.grad(whole, argnums=(0, 1, 2))
        zero_bs = jax.tree_util.tree_map(jnp.zeros_like, becsum_shift)
        return jax.jvp(
            lambda b, rho_s, s, bs: gradient(x, b, rho_s, s, bs),
            (becsum_, rho_smooth, jnp.zeros_like(shift), zero_bs),
            (d_becsum, d_smooth, shift, becsum_shift),
        )

    def pull(big, rowset, x, psi, weights, eigenvalues, g_b, g_rho, dpsi,
             dlambda, dg_b, dg_rho):
        """The ``jvp`` of one chunk's pull-back ``d/dx [E_c + g . sums_c]``."""
        ground = _ground_state_multipliers(weights, eigenvalues, psi.dtype)

        def slope(states, multipliers, g_b, g_rho):
            def energy(position):
                moved = moved_to(big, rowset, position)
                b, rho = sums(moved, states, weights)
                coupled = jnp.sum(g_rho * rho) + sum(
                    jnp.sum(g * part) for g, part in zip(g_b, b)
                    if part is not None)
                return separable(moved, states, weights, eigenvalues,
                                 multipliers) + coupled
            return jax.grad(energy)(x)

        _, derivative = jax.jvp(slope, (psi, ground, g_b, g_rho),
                                (embed(psi, dpsi), dlambda, dg_b, dg_rho))
        return derivative

    passes = {name: jax.jit(fn) for name, fn in (
        ("forward", forward), ("extras", extras), ("frozen", frozen),
        ("global", global_), ("pull", pull))}
    cached[1][key] = passes
    return passes


def _jacobian_by_columns(function, x):
    """``d f/dx`` one coordinate at a time, shaped ``f.shape + x.shape``.

    ``jax.jacfwd`` is a ``vmap`` of ``jvp`` over every coordinate at once, so
    for a derivative in the ``3 nat`` atomic positions it holds ``3 nat``
    tangent copies of everything the function builds -- here the chunk's moved
    projectors. Measured on the card, ultrasoft eight-atom silicon, a chunk of 27
    k-points: the frozen polarization's pass 3289 MB of temporaries and the
    ``add_for_charges`` pass 2185, against 925 MB for the whole SCF. Walking the
    unit tangents under ``lax.map`` holds one at a time; the columns are the
    same ``jvp`` either way. A complex ``f`` of a real ``x`` comes out complex.
    """
    basis = jnp.eye(x.size, dtype=x.dtype).reshape((x.size,) + x.shape)
    columns = jax.lax.map(lambda tangent: jax.jvp(function, (x,), (tangent,))[1],
                          basis)
    return jnp.moveaxis(columns, 0, -1).reshape(columns.shape[1:] + x.shape)
