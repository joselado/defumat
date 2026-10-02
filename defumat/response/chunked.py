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

**What still goes whole, by name.** The Born charges
(:mod:`defumat.response.born`) are a ``jvp`` of the whole-k force gradient, so
:meth:`StreamedField.born_charges` uploads the stores and runs that route as it
stands: the default call's peak is unchanged until it is split the way the force
is. A caller asking for the internals (the third derivatives, which read
``bare``, ``dpsi`` and the solver as device arrays), a calculation carrying a
strained ``_kcart``, and a k-point pool's store take the whole-k route or are
refused before this is reached.
"""

from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import numpy as np

from defumat.batching import k_chunks
from defumat.forces.chunked import _rows_of, row_leaves, with_rows
from defumat.forces.energy import hoisted, with_hoisted
from defumat.response.sternheimer import SternheimerSolver, local_perturbation
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
                bare_c, dv, coefficients):
        """``dpsi`` on one chunk, its worst iteration count and residual per
        channel, and the tangent of the chunk's raw ``(rho_smooth, becsum)``."""
        solver = solver_on(big, rowset, hamiltonians, psi, arrays, scalars,
                           v_scf, ddd_paw)
        induced = local_perturbation(solver.calculation, dv, v_scf, ddd_paw,
                                     coefficients=coefficients)

        def perturbation(states, ik, spin):
            return bare_c[spin][ik] + induced(states, ik, spin)

        dpsi, steps, residual = solver.solve_arrays(perturbation)
        _, parts = jax.jvp(solver.density_parts, (solver.psi,), (dpsi,))
        return dpsi, steps, residual, parts

    passes = {"bare": jax.jit(bare), "respond": jax.jit(respond)}
    cached[1][key] = passes
    return passes


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

    def _arguments(self, rows, live):
        """The leading arguments of both passes, for one chunk."""
        return (self.big, row_leaves(self.calculation, rows), self.hamiltonians,
                _rows_of(self.solver.psi, rows),
                self.solver.chunk_arrays(rows, live), self.scalars, self.v_scf,
                self.ddd_paw)

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

    def respond(self, dvscf, onecentre, include_induced: bool):
        """One iteration's three solves: ``(drho, dbecsum)`` per direction.

        ``drho`` is finished (on the dense grid, augmented) and not
        symmetrised; ``dbecsum`` is the raw response, and is returned only for
        PAW, whose one-centre potential is built from it -- the same two
        objects the whole-k route's ``response_density`` and
        ``response_becsum`` return.
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

        parts = [None, None, None]
        worst = [0, 0, 0]
        for rows, live in self.chunks:
            arguments = self._arguments(rows, live)
            written = rows[:live]
            for axis in range(3):
                dpsi, steps, _, chunk_parts = self.passes["respond"](
                    *arguments, _rows_of(self.bare[axis], rows), fields[axis],
                    coefficients[axis],
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
        """The Born charges by the whole-k route, from the stores uploaded.

        :func:`~defumat.response.born.born_effective_charges` differentiates
        the whole-k force gradient, so it needs the solver's states, the three
        ``dpsi``, the three perturbations and the commutators on the device; the
        projectors' ``d(vkb)/dk`` is rebuilt whole. The peak of this stage is
        the one the whole-k route has.
        """
        from defumat.response.born import born_effective_charges
        from defumat.response.efield import _bare_plus_induced

        solver = copy.copy(self.solver)
        solver.psi = jnp.asarray(self.solver.psi)
        bare = [jnp.asarray(self.bare[axis]) for axis in range(3)]
        dpsi = [jnp.asarray(self.dpsi[axis]) for axis in range(3)]
        commutators = (bare if self.commutators is None
                       else [jnp.asarray(self.commutators[axis]) for axis in range(3)])
        projector_velocities = None
        if self.dipole is not None:
            velocity = VelocityOperator(self.calculation, self.v_scf, self.ddd_paw)
            projector_velocities = [velocity.projectors(direction)
                                    for direction in np.eye(3)]
        perturbations = [
            _bare_plus_induced(solver, bare[axis], dvscf[axis],
                               None if onecentre is None else onecentre[axis], True)
            for axis in range(3)
        ]
        return born_effective_charges(
            self.calculation, solver, jnp.asarray(wavefunctions), eigenvalues,
            jnp.asarray(weights), jnp.asarray(density), becsum, dpsi,
            perturbations, commutators, projector_velocities=projector_velocities,
        )
