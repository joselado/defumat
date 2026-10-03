"""The third derivatives of the energy, a k-chunk at a time.

``GPU-MEMORY-NEXT.md`` item 2, its last piece. Electrostriction's
``d(eps)/d(strain)`` (:func:`~defumat.response.electrostriction.
susceptibility_strain_derivative`) and the Raman tensors' ``d(eps)/d(tau)``
(:func:`~defumat.response.nonlinear.susceptibility_displacement_derivative`)
are, per geometry tangent ``t``, one ``jvp`` of

    eps(x, psi, rho, b) = 1 - 4 FPI F(x, psi, rho, b; u) / Omega(x)

along ``(t, dpsi_t, drho_t, db_t)``, with ``F`` the variational second-order
energy (:func:`~defumat.response.electrostriction._second_order_energy_at`) and
``db_t`` three further Sternheimer solves
(:func:`~defumat.response.electrostriction._position_response`). Both read the
field's response ``u``, its bare perturbation ``b`` and the occupied states as
whole arrays. Here they are the field's host stores and every pass is one
chunk's.

**The derivative is forward mode only, and that is what makes the split
simple.** ``F`` is a sum over k of per-k terms that read two whole-cell objects,
plus terms of the whole sums:

    F = sum_c F_c(x, psi_c, b_c, u_c; v, D) + F_glob(x, rho, P, R_1, R_2, R_3)

with ``v = moved.potential(rho).v_scf``, ``D`` PAW's one-centre coefficients of
``P``, ``P`` the raw ``becsum`` of the occupied states, and ``R_i`` the raw
density sums along ``P_c u_i`` (``drho`` before it is finished and averaged).
``F_c`` is the band, multiplier and source terms of the chunk's k-points;
``F_glob`` the screening and one-centre terms. A ``jvp`` of a sum is the sum of
the ``jvp``, so per tangent:

* **forward walk**: per chunk ``P_c`` and ``R_c`` and their tangents along
  ``(t, dpsi_c)`` (``u`` has none), added;
* **global step**: ``v``, ``D``, the volume and ``F_glob`` with their tangents
  at the whole sums -- the stop-gradient average of ``drho`` acts on the
  finished whole-grid field there, exactly as in the whole route;
* **chunk walk**: per chunk ``db_c`` -- the residual's ``jvp`` with ``(v, D)``
  handed in as primal and tangent, three solves on the chunk's solver, and for
  an augmented dataset the tail's ``jvp`` -- then ``(F_c, dF_c)`` along
  ``(t, dpsi_c, db_c; dv, dD)``.

No cotangent walk is needed, unlike the Born charges' split, because nothing
here is a gradient.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from defumat.basis.interpolate import to_dense
from defumat.batching import k_chunks
from defumat.forces.chunked import _move, _rows_of, row_leaves, with_rows
from defumat.forces.energy import hoisted, with_hoisted
from defumat.response.chunked import _add
from defumat.response.sternheimer import SternheimerSolver, paw_response
from defumat.response.velocity import VelocityOperator, over_kpoints
from defumat.units import FPI

__all__ = ["walked_susceptibility_derivative"]


def walked_susceptibility_derivative(calculation, field, rho, kind, x, tangents,
                                     project=False, verbose=False) -> list:
    """``d(eps_ij)/dt`` per tangent, ``(3, 3)`` each, walked.

    Args:
        field: the :class:`~defumat.response.chunked.StreamedField` of the
            field response, whose ``solver`` (built on the host store of the
            re-diagonalised states), ``bare``, ``dpsi`` and ``commutators`` are
            the ``psi``, ``b``, ``u`` and ``stored`` of the whole route.
        rho: the converged density.
        kind: ``"strain"`` or ``"positions"``, the geometry coordinate.
        x: the coordinate's value, zero strain or the positions.
        tangents: ``(dx, dpsi, ort, drho)`` per column, ``dpsi`` and ``ort``
            ``(nspin, nk, nocc, ndim)`` host stores or device arrays, ``ort``
            ``None`` for a norm-conserving dataset. The state tangent is
            ``dpsi + ort``.
        project: apply ``P_c`` to ``dpsi`` first, with the unmoved metric, as
            :func:`~defumat.response.nonlinear.susceptibility_displacement_derivative`
            does; ``ort`` is added after the projection, as there.
    """
    solver = field.solver
    passes = _third_passes(calculation, (solver.nocc, solver.occupied_counts,
                                         solver.smearing), kind, project)
    big = hoisted(calculation)
    rho = jnp.asarray(rho)
    stored = field.commutators if field.commutators is not None else field.bare
    rowset0 = row_leaves(calculation, field.chunks[0][0])
    columns = []
    for dx, dpsi, ort, drho in tangents:
        dx = jnp.asarray(dx)
        drho = jnp.asarray(drho)

        def tangent_rows(rows):
            occupied = _rows_of(dpsi, rows)
            block = (jnp.zeros_like(occupied) if ort is None
                     else _rows_of(ort, rows))
            return occupied, block

        # 1. The forward walk.
        sums = dsums = None
        for rows, live in field.chunks:
            arguments = field._arguments(rows, live)
            value, tangent = passes["forward"](
                *arguments, x, dx, *tangent_rows(rows),
                _stacked_rows(field.dpsi, rows))
            sums, dsums = _add(sums, value), _add(dsums, tangent)
        # 2. The global step.
        (globals_, dglobals) = passes["global"](
            big, rowset0, x, dx, rho, drho, sums, dsums)
        f_glob, v, ddd, volume = globals_
        df_glob, dv, dddd, dvolume = dglobals
        # 3. The chunk walk.
        energy, denergy = np.asarray(f_glob), np.asarray(df_glob)
        for rows, live in field.chunks:
            arguments = field._arguments(rows, live)
            f_c, df_c = passes["chunk"](
                *arguments, x, dx, *tangent_rows(rows),
                _stacked_rows(field.bare, rows), _stacked_rows(field.dpsi, rows),
                _stacked_rows(stored, rows), v, dv, ddd, dddd, field.dipole)
            energy = energy + np.asarray(f_c)
            denergy = denergy + np.asarray(df_c)
        volume, dvolume = float(volume), float(dvolume)
        column = -4.0 * FPI * (denergy / volume - energy * dvolume / volume**2)
        columns.append(column)
        if verbose:
            print(f"  walked column: max = {np.abs(column).max():.6f}")
    return columns


def _third_passes(calculation, key, kind, project) -> dict:
    """The compiled passes, built once and cached on the calculation."""
    from defumat.response.efield import _solve_stored, ultrasoft_position
    from defumat.response.electrostriction import _apply_overlap, _project_conduction

    cached = calculation.__dict__.get("_streamed_response")
    if cached is None or cached[0] is not calculation:
        cached = (calculation, {})
        calculation._streamed_response = cached
    full = ("third", kind, bool(project)) + tuple(key)
    if full in cached[1]:
        return cached[1][full]
    nocc, counts, smearing = key
    batch = calculation.k_batch

    def local(big, rowset):
        return with_rows(with_hoisted(calculation, big), rowset)

    def solver_on(big, rowset, hamiltonians, psi, arrays, scalars, v_scf, ddd_paw):
        sub = local(big, rowset)
        return SternheimerSolver.on_chunk(
            sub, sub.restricted_hamiltonians(hamiltonians), psi, arrays, scalars,
            nocc=nocc, occupied_counts=counts, smearing=smearing, v_scf=v_scf,
            ddd_paw=ddd_paw)

    def raw(moved, states, weights):
        """``(rho_smooth, becsum)`` raw at the moved geometry -- ``sum_band``
        and ``mixed_becsum``, unsymmetrised."""
        rows = jnp.arange(states.shape[1])
        becsum_ = (moved.becsum(states, weights, rows=rows, symmetrize=False)
                   if moved.is_ultrasoft else ())
        return moved.smooth_density(states, weights, rows=rows), becsum_

    def mode_states(solver, dpsi, ort):
        """The state tangent ``P_c dpsi + ort``, or ``dpsi + ort`` unprojected."""
        if project:
            dpsi = _project_conduction(solver.psi, dpsi[None], solver.hamiltonians,
                                       batch)[0]
        return dpsi + ort

    def forward(big, rowset, hamiltonians, psi, arrays, scalars, v_scf, ddd_paw,
                x, dx, dpsi, ort, u):
        """``(P_c, R_c)`` and their tangents along ``(dx, dpsi + ort)``."""
        solver = solver_on(big, rowset, hamiltonians, psi, arrays, scalars,
                           v_scf, ddd_paw)
        weights = solver.weights
        tangent = mode_states(solver, dpsi, ort)
        sub = local(big, rowset)

        def sums(y, states):
            moved = _move(sub, kind, y)
            # ``S`` alone is read off these Hamiltonians; the potential they
            # carry is the ground state's and does not enter it.
            moved_h = moved.hamiltonian(v_scf, ddd_paw)
            pcu = _project_conduction(states, u, moved_h, batch)
            responses = tuple(
                jax.jvp(lambda s: raw(moved, s, weights), (states,), (pcu[axis],))[1]
                for axis in range(3))
            return raw(moved, states, weights)[1], responses

        return jax.jvp(sums, (x, solver.psi), (dx, tangent))

    def global_(big, rowset, x, dx, rho, drho, sums, dsums):
        """``(F_glob, v, D, Omega)`` and their tangents at the whole sums."""
        sub = local(big, rowset)
        smooth, dense = sub.basis.smooth, sub.basis.dense

        def whole(y, r, parts_and_responses):
            parts, responses = parts_and_responses
            moved = _move(sub, kind, y)
            v = moved.potential(r).v_scf
            ddd = None if moved.paw is None else moved.onecenter(parts)[1]
            fields = jnp.stack([
                moved.augmented(to_dense(smooth_part, smooth, dense), becsum_part)
                for smooth_part, becsum_part in responses])
            frozen = jax.lax.stop_gradient(fields)
            fields = sub.symmetrize_directional(frozen) + (fields - frozen)
            kernel = jnp.stack([
                jax.jvp(lambda q: moved.potential(q).v_scf, (r,), (fields[axis],))[1]
                for axis in range(3)])
            measure = moved.system.cell.volume / r.size * r.shape[0]
            onecentre = None
            if moved.paw is not None:
                dbecsum = [becsum_part for _, becsum_part in responses]
                blocks = [jnp.stack([
                    moved.augmentation.block_matrix(
                        tuple(None if b is None else b[spin] for b in dbecsum[axis]))
                    for spin in range(moved.nspin_mag)]) for axis in range(3)]
                induced = [paw_response(moved, dbecsum[axis], parts)
                           for axis in range(3)]
            rows_ = []
            for i in range(3):
                row = []
                for j in range(3):
                    value = 0.5 * measure * jnp.sum(fields[i] * kernel[j])
                    if moved.paw is not None:
                        value = value + 0.5 * jnp.sum(blocks[i] * induced[j])
                    row.append(value)
                rows_.append(jnp.stack(row))
            return jnp.stack(rows_), v, ddd, moved.system.cell.volume

        return jax.jvp(whole, (x, rho, sums), (dx, drho, dsums))

    def chunk(big, rowset, hamiltonians, psi, arrays, scalars, v_scf, ddd_paw,
              x, dx, dpsi, ort, b, u, stored, v, dv, ddd, dddd, dipole):
        """``db_c`` and then ``(F_c, dF_c)`` along ``(dx, dpsi + ort, db; dv, dD)``."""
        solver = solver_on(big, rowset, hamiltonians, psi, arrays, scalars,
                           v_scf, ddd_paw)
        weights = solver.weights
        tangent = mode_states(solver, dpsi, ort)
        sub = local(big, rowset)
        directions = jnp.eye(3)

        def operators(y, vv, dd):
            moved = _move(sub, kind, y)
            return (moved, VelocityOperator(moved, vv, dd),
                    moved.hamiltonian(vv, dd))

        def apply_h(hams, block):
            return jnp.stack([over_kpoints(hams[spin], block[spin], batch)
                              for spin in range(block.shape[0])])

        def apply_s(hams, block):
            return jnp.stack([over_kpoints(hams[spin], block[spin], batch,
                                           overlap=True)
                              for spin in range(block.shape[0])])

        # ``_position_response``'s residual, with ``(v, D)`` handed in.
        def residual(y, states, vv, dd):
            moved, velocity, hams = operators(y, vv, dd)
            lambdas = jnp.einsum("skmg,skng->skmn", jnp.conj(states),
                                 apply_h(hams, states))
            s_states = apply_s(hams, states)
            out = []
            for axis in range(3):
                derivative, overlap = velocity.both(states, directions[axis])
                commutator = -1j * (
                    derivative - jnp.einsum("skmn,skmg->skng", lambdas, overlap))
                overlaps = jnp.einsum("skmg,skng->skmn", jnp.conj(states),
                                      commutator)
                projected = commutator - jnp.einsum("skmn,skmg->skng", overlaps,
                                                    s_states)
                applied = apply_h(hams, stored[axis]) - jnp.einsum(
                    "skmn,skmg->skng", lambdas, apply_s(hams, stored[axis]))
                out.append(projected - applied)
            return jnp.stack(out)

        _, rhs = jax.jvp(residual, (x, solver.psi, v, ddd),
                         (dx, tangent, dv, dddd))
        solution = jnp.stack([_solve_stored(solver, rhs[axis]) for axis in range(3)])
        if dipole is None:
            db = solution
        else:
            # ``dpqq`` is the datasets' own (``int r Q_ij(r)``, one block per
            # atom of a species), the same at every geometry, so the field's
            # is handed in; the whole route rebuilds it at the moved cell, to
            # the same values and a zero tangent.
            def tail(y, states, vv, dd, position):
                moved, velocity, hams = operators(y, vv, dd)
                return jnp.stack([
                    ultrasoft_position(moved, hams, states, position[axis],
                                       dipole[axis],
                                       velocity.projectors(directions[axis]))
                    for axis in range(3)])

            _, db = jax.jvp(tail, (x, solver.psi, v, ddd, stored),
                            (dx, tangent, dv, dddd, solution))

        # ``F_c``: the band, multiplier and source terms of the chunk.
        def separable(y, states, bb, vv, dd):
            moved = _move(sub, kind, y)
            hams = moved.hamiltonian(vv, dd)
            lambdas = jnp.einsum("skmg,skng->skmn", jnp.conj(states),
                                 apply_h(hams, states))
            overlapped = _apply_overlap(states, hams, batch)
            overlaps = jnp.einsum("skmg,askng->askmn", jnp.conj(states), bb)
            pcb = bb - jnp.einsum("askmn,skmg->askng", overlaps, overlapped)
            pcu = _project_conduction(states, u, hams, batch)
            hu = jnp.stack([apply_h(hams, pcu[axis]) for axis in range(3)])
            su = _apply_overlap(pcu, hams, batch)
            rows_ = []
            for i in range(3):
                row = []
                for j in range(3):
                    band = jnp.sum(weights * jnp.real(
                        jnp.einsum("skng,skng->skn", jnp.conj(pcu[i]), hu[j])))
                    multiplier = jnp.sum(weights * jnp.real(jnp.einsum(
                        "skmn,skng,skmg->skn", lambdas, jnp.conj(pcu[i]), su[j])))
                    source = jnp.sum(weights * jnp.real(
                        jnp.einsum("skng,skng->skn", jnp.conj(pcu[i]), pcb[j])
                        + jnp.einsum("skng,skng->skn", jnp.conj(pcu[j]), pcb[i])))
                    row.append(band - multiplier + source)
                rows_.append(jnp.stack(row))
            return jnp.stack(rows_)

        return jax.jvp(separable, (x, solver.psi, b, v, ddd),
                       (dx, tangent, db, dv, dddd))

    passes = {name: jax.jit(fn) for name, fn in (
        ("forward", forward), ("global", global_), ("chunk", chunk))}
    cached[1][full] = passes
    return passes


def _stacked_rows(store, rows):
    """One chunk of a ``(3, nspin, nk, ...)`` store, the k axis the third."""
    return jnp.stack([_rows_of(store[axis], rows) for axis in range(store.shape[0])])

