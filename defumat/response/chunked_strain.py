"""The strain response, a k-chunk at a time.

``GPU-MEMORY-NEXT.md`` item 2, after the field response
(:mod:`defumat.response.chunked`) and the phonons
(:mod:`defumat.response.chunked_phonon`). :func:`~defumat.response.strain.
strain_response` holds, for each of the six independent strains, a bare
perturbation ``(dH/d(eps) - eps dS/d(eps))|psi>`` and a first-order state
``dpsi``, and for an ultrasoft or PAW dataset the occupied block ``ort`` and the
overlap derivative ``S'`` as well -- every one a ``(nspin, nk, nocc, ndim)``
array, whole-k and on the device. Here they live in host memory and every pass is
one chunk's, with the Gamma phonon's three stages and the six strains as its
modes:

* **prepare**, once: per chunk and strain, the bare perturbation (one ``jvp``
  through ``at_strain`` on the chunk's row-subset calculation, with the potential
  rebuilt from the frozen density inside the derivative, which is
  :func:`~defumat.response.strain._bare_strains`' rule), and for ultrasoft or PAW
  ``S'``; and for every dataset the tangents of the two raw sums over k the
  frozen-state density response is built from -- along the strain at frozen
  states, which carries the volume's ``1/Omega``, and along ``ort``. The potential
  and its strain tangent are k-independent, so they are taken once per strain
  and handed to every chunk as a primal and a tangent. The two halves of the
  frozen-state response are finished once from the summed tangents: the strained
  augmentation charge at the summed ``becsum`` for the first, the unstrained one
  for the second;
* **the response**, once per self-consistent iteration: the field's own
  ``respond`` pass, with six perturbations in place of three, driven by
  :func:`~defumat.response.strain._self_consistent_response`'s loop, which acts
  on whole-grid objects only and is shared with the whole route;
* **the eigenvalue response**, after the loop, per chunk.

Against the whole-k route on the same converged state
(``tests/regression/test_streamed_strain.py``): see that file.

**What it does not change**: the ``(3, 3)`` dense-grid fields the loop carries
(``dvscf``, the response, the induced potential, the frozen-state response),
which have no k index; and the consumers, which read ``dpsi`` and ``ort`` from
the host stores this leaves on :class:`~defumat.response.strain.StrainResponse`
one strain at a time -- the elastic constants and the third derivatives still
assemble each strain's column with the whole k axis on one tape.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from defumat.basis.interpolate import to_dense
from defumat.batching import k_chunks
from defumat.forces.chunked import _rows_of, row_leaves, with_rows
from defumat.forces.energy import hoisted, with_hoisted
from defumat.response.chunked import _add, _passes as _field_passes
from defumat.response.sternheimer import SternheimerSolver, local_perturbation
from defumat.response.velocity import over_kpoints

__all__ = ["StreamedStrains"]

#: The six independent strains, in the order the whole route solves them.
PAIRS = tuple((a, b) for a in range(3) for b in range(a, 3))


class StreamedStrains:
    """The strain response's stores in host memory, and the walks over them.

    ``solver`` is the whole set's :class:`~defumat.response.sternheimer.
    SternheimerSolver`, built on the host store (or on the device array, when
    the route walks because of memory mode on a card): it supplies the
    eigenvalues, the weights, the masks and the level shift, and its states stay
    where they are.
    """

    def __init__(self, calculation, solver, density):
        self.calculation = calculation
        self.solver = solver
        self.v_scf = solver.v_scf
        self.ddd_paw = solver.ddd_paw
        self.hamiltonians = solver.hamiltonians
        self.density = jnp.asarray(density)
        self.ultrasoft = bool(calculation.is_ultrasoft)
        self.chunks = list(k_chunks(calculation.system.kpoints.nk,
                                    calculation.k_batch))
        key = (solver.nocc, solver.occupied_counts, solver.smearing)
        self.field_passes = _field_passes(calculation, key)
        self.passes = _strain_passes(calculation, key)
        self.big = hoisted(calculation)
        self.scalars = solver.scalars()
        shape = (len(PAIRS),) + tuple(solver.psi.shape)
        dtype = np.dtype(solver.psi.dtype)
        self.bare = np.zeros(shape, dtype)
        self.dpsi = np.zeros(shape, dtype)
        nspin, nk, nocc = solver.psi.shape[:3]
        #: ``S'`` per strain, ``(6, nspin, nk, nocc, nocc)``: what ``ort`` is
        #: rebuilt from.
        self.overlaps = (np.zeros((len(PAIRS), nspin, nk, nocc, nocc), dtype)
                         if self.ultrasoft else None)
        self.iterations = 0
        self.solves = 0

    # -- the chunk's arguments -------------------------------------------------

    @staticmethod
    def _tangent(pair):
        """``strain_tangent(a, b)``, built on the host so no index is a constant."""
        tangent = np.zeros((3, 3))
        a, b = pair
        tangent[a, b] += 0.5
        tangent[b, a] += 0.5
        return jnp.asarray(tangent)

    def _arguments(self, rows, live):
        """The leading arguments of the solver passes, for one chunk."""
        return (self.big, row_leaves(self.calculation, rows), self.hamiltonians,
                _rows_of(self.solver.psi, rows),
                self.solver.chunk_arrays(rows, live), self.scalars, self.v_scf,
                self.ddd_paw)

    # -- walk 1: the bare perturbations and the frozen-state response ----------

    def prepare(self):
        """The bare perturbations and ``S'`` into the host stores; returns
        :func:`~defumat.response.strain._frozen_density_response`'s
        ``(total, moved, becsum_total, becsum_moved)``."""
        calculation = self.calculation
        strain = jnp.zeros((3, 3))
        rowset = row_leaves(calculation, self.chunks[0][0])
        potentials = [self.passes["potential"](self.big, rowset, self.density,
                                               strain, self._tangent(pair))
                      for pair in PAIRS]
        raw = None
        moved_sums = [None] * len(PAIRS)
        ort_sums = [None] * len(PAIRS)
        for rows, live in self.chunks:
            arguments = self._arguments(rows, live)
            leaves, psi, arrays = arguments[1], arguments[3], arguments[4]
            written = rows[:live]
            for index, pair in enumerate(PAIRS):
                tangent = self._tangent(pair)
                v_scf, dv_scf = potentials[index]
                bare = self.passes["bare"](
                    self.big, leaves, psi, arrays["eigenvalues"], v_scf, dv_scf,
                    self.ddd_paw, strain, tangent)
                self.bare[index][:, written] = np.asarray(bare)[:, :live]
                overlap = None
                if self.ultrasoft:
                    overlap = self.passes["overlap"](self.big, leaves, psi,
                                                     strain, tangent)
                    self.overlaps[index][:, written] = (
                        np.asarray(overlap)[:, :live])
                sums, moved, ort = self.passes["mixed"](
                    self.big, leaves, strain, tangent, psi, arrays["weights"],
                    overlap)
                if index == 0:
                    raw = _add(raw, sums)
                moved_sums[index] = _add(moved_sums[index], moved)
                ort_sums[index] = _add(ort_sums[index], ort)

        raw_becsum, raw_smooth = raw
        total = np.empty((3, 3), dtype=object)
        moved_part = np.empty((3, 3), dtype=object)
        becsum_total = np.empty((3, 3), dtype=object)
        becsum_moved = np.empty((3, 3), dtype=object)
        for index, (a, b) in enumerate(PAIRS):
            d_becsum, d_smooth = moved_sums[index]
            rho_m = self.passes["finish_moved"](
                self.big, rowset, strain, self._tangent((a, b)), raw_smooth,
                raw_becsum, d_smooth, d_becsum)
            if self.ultrasoft:
                o_becsum, o_smooth = ort_sums[index]
                rho_t = rho_m + self.solver.finish_density(o_smooth, o_becsum)
                bec_t = tuple(None if x is None else x + y
                              for x, y in zip(d_becsum, o_becsum))
            else:
                rho_t, bec_t = rho_m, d_becsum
            total[a, b] = total[b, a] = rho_t
            moved_part[a, b] = moved_part[b, a] = rho_m
            becsum_total[a, b] = becsum_total[b, a] = bec_t
            becsum_moved[a, b] = becsum_moved[b, a] = d_becsum

        def stacked(fields):
            return jnp.stack([jnp.stack([fields[a, b] for b in range(3)])
                              for a in range(3)])

        return stacked(total), stacked(moved_part), becsum_total, becsum_moved

    # -- walk 2: the solves, once per iteration --------------------------------

    def respond(self, dvscf, onecentre, include_induced: bool, frozen_becsum=None):
        """One iteration's six solves: the finished, unsymmetrised response
        density and (PAW) the raw ``becsum`` response with the frozen-state part
        added, as ``(3, 3)`` object arrays -- the whole route's ``respond``."""
        solver = self.solver
        fields, coefficients = [], []
        for a, b in PAIRS:
            dv = dvscf[a, b] if include_induced else jnp.zeros_like(dvscf[a, b])
            dddd = None if onecentre is None else (
                onecentre[a, b] if include_induced
                else jnp.zeros_like(onecentre[a, b]))
            fields.append(dv)
            coefficients.append(solver.perturbed_coefficients(dv, dddd)
                                if self.ultrasoft else None)

        parts = [None] * len(PAIRS)
        worst = [0] * len(PAIRS)
        for rows, live in self.chunks:
            arguments = self._arguments(rows, live)
            written = rows[:live]
            for index in range(len(PAIRS)):
                dpsi, steps, _, chunk_parts = self.field_passes["respond"](
                    *arguments, _rows_of(self.bare[index], rows), fields[index],
                    coefficients[index])
                self.dpsi[index][:, written] = np.asarray(dpsi)[:, :live]
                parts[index] = _add(parts[index], chunk_parts)
                worst[index] = max(worst[index], int(np.max(np.asarray(steps))))
        self.iterations += sum(worst)
        self.solves += len(PAIRS)

        response = np.empty((3, 3), dtype=object)
        becsum_response = np.empty((3, 3), dtype=object)
        for index, (a, b) in enumerate(PAIRS):
            smooth, becsum_part = parts[index]
            response[a, b] = response[b, a] = solver.finish_density(
                smooth, becsum_part)
            if onecentre is not None:
                if frozen_becsum is not None:
                    becsum_part = tuple(
                        None if x is None else x + y
                        for x, y in zip(becsum_part, frozen_becsum[a, b]))
                becsum_response[a, b] = becsum_response[b, a] = becsum_part
        return response, becsum_response

    # -- after the loop --------------------------------------------------------

    def eigenvalue_response(self, dvscf) -> np.ndarray:
        """``(3, 3, nspin, nk, nocc)``: ``<psi_n|dH_bare + dV_scf|psi_n>``, walked."""
        solver = self.solver
        out = np.empty((3, 3) + tuple(solver.psi.shape[:3]))
        for index, (a, b) in enumerate(PAIRS):
            coefficients = (solver.perturbed_coefficients(dvscf[a, b])
                            if self.ultrasoft else None)
            values = np.empty(solver.psi.shape[:3])
            for rows, live in self.chunks:
                chunk = self.passes["eigenvalues"](
                    *self._arguments(rows, live), _rows_of(self.bare[index], rows),
                    dvscf[a, b], coefficients)
                values[:, rows[:live]] = np.asarray(chunk)[:, :live]
            out[a, b] = out[b, a] = values
        return out

    def first_order_states(self) -> np.ndarray:
        """``dpsi`` as the whole route's ``(3, 3)`` object array, host stores."""
        out = np.empty((3, 3), dtype=object)
        for index, (a, b) in enumerate(PAIRS):
            out[a, b] = out[b, a] = self.dpsi[index]
        return out

    def overlap_derivatives(self):
        """``S'`` and ``ort = -1/2 psi S'`` as ``(3, 3)`` object arrays of host
        arrays, or ``(None, None)`` for a norm-conserving dataset."""
        if not self.ultrasoft:
            return None, None
        psi = np.asarray(self.solver.psi)
        derivatives = np.empty((3, 3), dtype=object)
        ort = np.empty((3, 3), dtype=object)
        for index, (a, b) in enumerate(PAIRS):
            derivatives[a, b] = derivatives[b, a] = self.overlaps[index]
            ort[a, b] = ort[b, a] = -0.5 * np.einsum(
                "skmg,skmn->skng", psi, self.overlaps[index])
        return derivatives, ort


def _strain_passes(calculation, key) -> dict:
    """The strain response's compiled passes, built once and cached on it.

    Keyed as :func:`defumat.response.chunked._passes` is, in the same cache, so
    a moved or strained copy builds its own. ``key`` is the solver's static
    configuration: the occupied-block width, the counts per channel and the
    smearing.
    """
    cached = calculation.__dict__.get("_streamed_response")
    if cached is None or cached[0] is not calculation:
        cached = (calculation, {})
        calculation._streamed_response = cached
    key = ("strain",) + tuple(key)
    if key in cached[1]:
        return cached[1][key]

    nocc, counts, smearing = key[1:]
    batch = calculation.k_batch

    def local(big, rowset):
        return with_rows(with_hoisted(calculation, big), rowset)

    def potential(big, rowset, density, x, dx):
        """``v_scf`` at the strained cell from the frozen density, and its tangent."""
        sub = local(big, rowset)
        return jax.jvp(lambda s: sub.at_strain(s).potential(density).v_scf,
                       (x,), (dx,))

    def bare(big, rowset, psi, eigenvalues, v_scf, dv_scf, ddd_paw, x, dx):
        """``(dH/d(eps) - eps dS/d(eps))|psi>`` on one chunk -- ``_bare_strains``,
        with the potential's strain tangent handed in rather than rebuilt."""
        sub = local(big, rowset)

        def h_psi(strain, v):
            moved = sub.at_strain(strain)
            applied = []
            for spin, hamiltonian in enumerate(moved.hamiltonian(v, ddd_paw)):
                values = over_kpoints(hamiltonian, psi[spin], batch)
                if hamiltonian.has_overlap:
                    overlap = over_kpoints(hamiltonian, psi[spin], batch,
                                           overlap=True)
                    values = values - eigenvalues[spin][..., None] * overlap
                applied.append(values)
            return jnp.stack(applied)

        return jax.jvp(h_psi, (x, v_scf), (dx, dv_scf))[1]

    def overlap(big, rowset, psi, x, dx):
        """``S'_mn = <psi_m|dS/d(eps)|psi_n>`` on one chunk -- ``overlap_derivatives``."""
        sub = local(big, rowset)

        def matrix(strain):
            moved = sub.at_strain(strain)
            vkb = moved.projectors_at(jnp.arange(psi.shape[1]))
            qq = moved.projectors.qq.astype(psi.dtype)
            becp = jnp.einsum("kgc,skng->sknc", vkb.conj(), psi)
            return jnp.einsum("skmi,ij,sknj->skmn", becp.conj(), qq, becp)

        return jax.jvp(matrix, (x,), (dx,))[1]

    def sums(moved, states, weights):
        rows = jnp.arange(states.shape[1])
        becsum_ = (moved.becsum(states, weights, rows=rows, symmetrize=False)
                   if moved.is_ultrasoft else ())
        return becsum_, moved.smooth_density(states, weights, rows=rows)

    def mixed(big, rowset, x, dx, psi, weights, overlap):
        """The raw sums, and their tangents along the strain at frozen states
        and along ``ort`` -- ``_frozen_density_response``'s two ``jvp``."""
        sub = local(big, rowset)
        value, moved_tangent = jax.jvp(
            lambda strain: sums(sub.at_strain(strain), psi, weights), (x,), (dx,))
        if overlap is None:
            return value, moved_tangent, None
        ort = -0.5 * jnp.einsum("skmg,skmn->skng", psi, overlap)
        _, ort_tangent = jax.jvp(lambda states: sums(sub, states, weights),
                                 (psi,), (ort,))
        return value, moved_tangent, ort_tangent

    def finish_moved(big, rowset, x, dx, rho_smooth, becsum_, d_smooth, d_becsum):
        """The moved half finished: the strained augmentation charge at the summed
        ``becsum``, along the summed tangents."""
        sub = local(big, rowset)
        smooth, dense = sub.basis.smooth, sub.basis.dense
        return jax.jvp(
            lambda strain, r, parts: sub.at_strain(strain).augmented(
                to_dense(r, smooth, dense), parts),
            (x, rho_smooth, becsum_), (dx, d_smooth, d_becsum))[1]

    def eigenvalues(big, rowset, hamiltonians, psi, arrays, scalars, v_scf,
                    ddd_paw, bare_c, dv, coefficients):
        """``<psi_n|bare + induced|psi_n>`` on one chunk -- ``_eigenvalue_response``."""
        sub = local(big, rowset)
        solver = SternheimerSolver.on_chunk(
            sub, sub.restricted_hamiltonians(hamiltonians), psi, arrays, scalars,
            nocc=nocc, occupied_counts=counts, smearing=smearing, v_scf=v_scf,
            ddd_paw=ddd_paw)
        induced = local_perturbation(sub, dv, v_scf, ddd_paw,
                                     coefficients=coefficients)
        states = solver.psi
        out = []
        for spin in range(states.shape[0]):
            def one_k(ik, spin=spin):
                total = bare_c[spin][ik] + induced(states[spin][ik], ik, spin)
                return jnp.real(jnp.einsum("ng,ng->n", jnp.conj(states[spin][ik]),
                                           total))
            out.append(jax.vmap(one_k)(jnp.arange(states.shape[1])))
        return jnp.stack(out)

    passes = {name: jax.jit(fn) for name, fn in (
        ("potential", potential), ("bare", bare), ("overlap", overlap),
        ("mixed", mixed), ("finish_moved", finish_moved),
        ("eigenvalues", eigenvalues))}
    cached[1][key] = passes
    return passes
