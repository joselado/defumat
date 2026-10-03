"""The frozen-state force and stress, a k-chunk at a time.

``GPU-MEMORY-NEXT.md`` item 3. The force and the stress are ``jax.grad`` of the
frozen energy (:func:`~defumat.forces.energy.energy_at`), and a single ``grad``
puts the whole k axis on one tape: every k-point's states, projectors and
per-k tables are live for the backward pass at once, which is the SCF's
whole-set working set coming back after memory mode took it away -- and a
streamed store (a numpy array in host memory) has to cross to the device whole
to be differentiated at all. ``MEMORY-AUDIT.md`` §8 records why the k dial is
inert under reverse mode: a ``lax.map`` stacks every chunk's residuals for the
backward pass.

So the energy is split the way :func:`~defumat.forces.spiral.
_split_energy_and_gradient` splits ``dE/dq`` on an augmented spiral:

    E(x) = sum_c E_c(x) + E_glob(x, b(x), rho_s(x), ns(x))

with ``x`` the coordinate -- the positions, or a strain. ``E_c`` is one chunk's
**separable** part: the kinetic energy, the nonlocal and overlap quadratic
forms and the orthonormality term, each a plain sum over k. ``b``, ``rho_s`` and
``ns`` are the raw sums over k that the rest of the energy is a *nonlinear*
function of -- ``becsum``, ``sum_band``'s smooth-grid density and DFT+U's
occupation matrix -- and ``E_glob`` is everything built from them: the local,
Hartree and exchange-correlation energies (with the augmentation charge and the
core charge, both moving with ``x``), PAW's one-centre energy, the Hubbard
energy, Ewald and dispersion. Then

    dE/dx = dE_glob/dx|_(b, rho, ns)
          + sum_c [ dE_c/dx + (dE_glob/db) . db_c/dx + (dE_glob/drho) . drho_c/dx
                    + (dE_glob/dns) . dns_c/dx ]

and the three passes below are its terms: a forward walk for ``b``, ``rho_s``
and ``ns``; one ``value_and_grad`` of ``E_glob`` in all four arguments **at the
whole sums**; and a second walk in which each chunk's ``(E_c, b_c, rho_c,
ns_c)`` is pulled back with cotangent ``(1, g_b, g_rho, g_ns)``. Each walk's
tape is one chunk's, and the global tape has no k axis. It is exact: the
*value* inside every quadratic term is the whole sum and only its derivative is
taken chunk by chunk -- the wedge-sum rule of ``CLAUDE.md`` one level down.
Accumulating per-chunk gradients of ``E_c + E_glob(b_c)`` instead would drop
every cross term between two chunks.

What it costs: the separable part and the three sums are evaluated twice (once
in the forward walk, once under each chunk's ``vjp``), and one chunk's
dispatch per chunk. What it removes is the k axis from every tape.

Every chunk is padded to the same shape with a repeat of one of its own rows at
**zero weight** (:func:`~defumat.batching.k_chunks`), so every chunk shares one
compilation and the padding contributes nothing: every piece is linear in the
weights. The compiled functions take the calculation's large fields as
arguments (:data:`~defumat.forces.energy.HOISTED_FIELDS`), as the single-pass
gradients do, so no per-k table becomes a constant of the executable, and the
arrays the geometry moves as well (:data:`~defumat.forces.energy.
GEOMETRY_FIELDS`), so a relaxation's later steps reuse them, the variable-cell
one included.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from defumat.batching import k_chunks
from defumat.forces.energy import (
    FrozenState, _kinetic_energy, _norms, _projector_energies,
    _spinor_projector_energies, geometry_compiled, hoisted, reject_potential_only,
    reject_spinor_spiral,
)
from defumat.hubbard.energy import hubbard_energy
from defumat.scf.potential import total_charge
from defumat.parallel import PoolStore, current_pools
from defumat.scf.streaming import is_host_store, rows_to_device

__all__ = ["chunked_gradient", "walks_chunks", "wants_chunks"]


def wants_chunks(calculation, state: FrozenState) -> bool:
    """Whether the force and stress should walk the k axis rather than tape it whole.

    Where the state lives in host memory (a streamed store), and in memory
    mode wherever the chunk is smaller than the k-set. Speed mode, the CPU
    default every validated number was taken in, keeps the single pass.
    """
    return walks_chunks(calculation, state.wavefunctions)


def walks_chunks(calculation, wavefunctions) -> bool:
    """:func:`wants_chunks` for a consumer holding the states rather than a :class:`FrozenState`.

    The one rule for every derivative or response that can walk the k axis a
    chunk at a time (:mod:`defumat.response.chunked` reads it too), so the
    force, the stress and the field response cannot disagree about when the
    store is too large to take whole.
    """
    nk = calculation.system.kpoints.nk
    if is_host_store(wavefunctions) or isinstance(wavefunctions, PoolStore):
        # A k-point pool's share is walked as the pool's own rows.
        return True
    batch = calculation.k_batch
    return (calculation.memory_mode == "memory" and batch is not None
            and batch < nk)


def _move(calculation, kind: str, x):
    if kind == "positions":
        return calculation.at_positions(x)
    return calculation.at_strain(x)


def _separable(moved, psi, weights, eigenvalues, rows):
    """``(E_c, b_c, rho_c, ns_c)``: one chunk's separable energy and its raw sums.

    The same expressions :func:`~defumat.forces.energy.energy_at` writes, on the
    chunk's rows of every per-k table. ``b_c`` is ``()`` on a norm-conserving
    run and ``ns_c`` is ``None`` without DFT+U, which keeps the pytree the
    global function is differentiated in stable.
    """
    gamma_only = bool(getattr(moved, "gamma_only", False))
    vkb = moved.projectors_at(rows)
    if moved.noncolin:
        kinetic = _kinetic_energy(psi, moved.state_kinetic[rows], weights)
        nonlocal_, overlap = _spinor_projector_energies(
            psi, vkb, moved.dvan_so, moved.qq_so, weights, eigenvalues)
    else:
        kinetic = _kinetic_energy(psi, moved.kinetic[rows], weights, gamma_only)
        nonlocal_, overlap = _projector_energies(
            psi, vkb, moved.projectors.dij, moved.projectors.qq, weights,
            eigenvalues, gamma_only)
    norm = jnp.sum(weights * eigenvalues * (_norms(psi, gamma_only) - 1.0))
    energy = kinetic + nonlocal_ - overlap - norm
    becsum_ = moved.becsum(psi, weights, rows=rows, symmetrize=False)
    rho = moved.smooth_density(psi, weights, rows=rows)
    ns = (moved.occupation_matrix(psi, weights, rows=rows, symmetrize=False)
          if moved.is_hubbard else None)
    return energy, becsum_, rho, ns


def _global(moved, becsum_, rho_smooth, ns):
    """Everything in the energy that is a nonlinear function of the whole sums."""
    finished = moved.finish_becsum(becsum_) if becsum_ else becsum_
    rho = moved.finish_density(rho_smooth, finished)
    potential = moved.potential(rho)
    epaw, _ = moved.onecenter(finished)
    volume = moved.system.cell.volume
    local = volume / rho[0].size * jnp.sum(moved.vltot * total_charge(rho))
    energy = (local + potential.ehart + potential.etxc + epaw + moved.ewald
              + moved.dispersion)
    if ns is not None:
        energy = energy + hubbard_energy(
            moved.finish_occupation_matrix(ns), moved.hubbard_coefficients)
    return energy


def _rows_of(array, rows):
    """One chunk of a ``(nspin, nk, ...)`` state array on the device.

    A host array crosses as a view of its rows wherever they are a run, which
    is every chunk but a padded last one (:func:`~defumat.scf.streaming.
    rows_to_device`); a device array is indexed where it is.
    """
    if isinstance(array, np.ndarray):
        return rows_to_device(array, rows)
    return jnp.asarray(array)[:, jnp.asarray(rows)]


def _chunk(state: FrozenState, rows, live, positions=None, weights=None):
    """``psi``, the weights with the padding zeroed, and the eigenvalues of one chunk.

    ``rows`` are global k indices; ``positions``, for a k-point pool's
    :class:`~defumat.parallel.PoolStore`, are where those rows sit in the
    pool's own store. ``weights`` is ``state.weights`` already on the host,
    which a walk converts once rather than once per chunk.
    """
    weights = np.array((state.weights if weights is None else weights)[:, rows])
    weights[:, live:] = 0.0
    store = state.wavefunctions
    psi = (_rows_of(store.array, positions) if isinstance(store, PoolStore)
           else _rows_of(store, rows))
    return (psi, jnp.asarray(weights),
            jnp.asarray(np.asarray(state.eigenvalues)[:, rows]))


def _add(total, part):
    return part if total is None else jax.tree_util.tree_map(jnp.add, total, part)


def _compiled(calculation, kind: str) -> tuple:
    """``(passes, geometry)``: the three compiled passes for ``calculation``, and their geometry.

    **Built once a run and inherited by every geometry**, the single-pass
    gradients' rule (:func:`~defumat.forces.energy.geometry_compiled`). The
    passes take the coordinate ``x``, the large fields
    (:data:`~defumat.forces.energy.HOISTED_FIELDS`), the arrays every geometry
    carries (:data:`~defumat.forces.energy.GEOMETRY_FIELDS`) and one chunk's
    rows as arguments, so a calculation moved by ``at_positions`` or
    ``at_cell`` reuses them whenever the key of what they still close over
    matches. The force's passes used to be inherited only through
    ``at_positions`` and the stress's keyed on the calculation itself, so a
    variable-cell relaxation compiled both again at every step.
    """
    store, geometry = geometry_compiled(calculation, "_chunked_gradient",
                                        lambda key: {"key": key})
    if kind in store:
        return store[kind], geometry
    key = store["key"]

    def local(big, geometry, rowset):
        """The calculation on one chunk's k-points, from traced leaves."""
        return with_rows(key.rebuild(geometry, big), rowset)

    def sums(x, big, geometry, rowset, psi, weights, eigenvalues):
        moved = _move(local(big, geometry, rowset), kind, x)
        return _separable(moved, psi, weights, eigenvalues, _all(psi))[1:]

    def whole(x, big, geometry, rowset, becsum_, rho, ns):
        moved = _move(local(big, geometry, rowset), kind, x)
        return _global(moved, becsum_, rho, ns)

    def pull(x, big, geometry, rowset, psi, weights, eigenvalues, cotangent):
        (energy, *parts), back = jax.vjp(
            lambda x: _separable(_move(local(big, geometry, rowset), kind, x),
                                 psi, weights, eigenvalues, _all(psi)),
            x,
        )
        (slope,) = back((jnp.ones_like(energy), *cotangent))
        return energy, slope

    passes = {
        "sums": jax.jit(sums),
        "global": jax.jit(jax.value_and_grad(whole, argnums=(0, 4, 5, 6))),
        "pull": jax.jit(pull),
    }
    store[kind] = passes
    return passes, geometry


def chunked_gradient(calculation, state: FrozenState, kind: str, x,
                     k_batch: int | None = None, pools=None):
    """``(E, dE/dx)`` at ``x``, the k axis walked ``k_batch`` at a time.

    ``k_batch`` defaults to the calculation's own chunk.

    **Under k-point pools** (a :class:`~defumat.parallel.PoolStore` state) each
    pool walks its own rows: the forward walk's raw sums are all-reduced
    before the global terms are evaluated at them, every pool evaluates those
    terms identically, the pull-back walk's separable energy and gradient are
    all-reduced, and the global gradient is added once, after the reduction --
    so no k-independent term is counted once per pool.

    ``kind`` is ``"positions"`` (``x`` the ``(nat, 3)`` positions, for the
    force) or ``"strain"`` (``x`` the ``(3, 3)`` strain, for the stress). The
    energy is :func:`~defumat.forces.energy.energy_at`'s at the moved
    calculation, to round-off -- only the order of the sums over k differs.
    """
    # The refusals :func:`~defumat.forces.energy.energy_at` makes, since this
    # route does not pass through it.
    reject_potential_only(calculation)
    if calculation.noncolin:
        reject_spinor_spiral(calculation)
    passes, geometry = _compiled(calculation, kind)
    big = hoisted(calculation)
    nk = calculation.system.kpoints.nk
    batch = (calculation.k_batch if k_batch is None else k_batch) or nk
    store = state.wavefunctions
    pooled = isinstance(store, PoolStore)
    if pooled:
        pools = current_pools() if pools is None else pools
        chunks = [(store.rows[positions], live, positions)
                  for positions, live in k_chunks(len(store.rows), batch)]
    else:
        chunks = [(rows, live, None) for rows, live in k_chunks(nk, batch)]
    # On the host once for both walks, where each chunk used to fetch its own.
    host_weights = np.asarray(state.weights)

    becsum_ = rho = ns = None
    for rows, live, positions in chunks:
        psi, weights, eigenvalues = _chunk(state, rows, live, positions,
                                           host_weights)
        part = passes["sums"](x, big, geometry, row_leaves(calculation, rows), psi,
                              weights, eigenvalues)
        becsum_, rho, ns = (_add(becsum_, part[0]), _add(rho, part[1]),
                            _add(ns, part[2]))
    if pooled:
        becsum_, rho, ns = pools.allreduce_sum((becsum_, rho, ns))

    # The global terms read nothing with a k index; any one chunk's rows stand
    # in, so that the moved calculation's per-k rebuilds are one chunk's.
    e_glob, (g_x, g_b, g_rho, g_ns) = passes["global"](
        x, big, geometry, row_leaves(calculation, chunks[0][0]), becsum_, rho, ns)
    separable = 0.0
    pulled = jnp.zeros_like(g_x)
    for rows, live, positions in chunks:
        psi, weights, eigenvalues = _chunk(state, rows, live, positions,
                                           host_weights)
        value, slope = passes["pull"](x, big, geometry, row_leaves(calculation, rows),
                                      psi, weights, eigenvalues,
                                      (g_b, g_rho, g_ns))
        separable = separable + value
        pulled = pulled + slope
    if pooled:
        separable, pulled = pools.allreduce_sum((jnp.asarray(separable), pulled))
    return e_glob + state.entropy + separable, g_x + pulled


def _tangent_compiled(calculation, kind: str) -> tuple:
    """``(passes, geometry)`` of :func:`chunked_gradient_tangent`, cached beside
    :func:`_compiled`'s under the same key."""
    _compiled(calculation, kind)
    store, geometry = geometry_compiled(calculation, "_chunked_gradient",
                                        lambda key: {"key": key})
    name = "tangent-" + kind
    if name in store:
        return store[name], geometry
    key = store["key"]

    def local(big, geometry, rowset):
        return with_rows(key.rebuild(geometry, big), rowset)

    def embed(psi, dpsi):
        """A tangent over the first bands, widened to every band with zeros."""
        return jnp.zeros_like(psi).at[:, :, :dpsi.shape[2]].set(dpsi)

    def forward(x, dx, big, geometry, rowset, psi, weights, eigenvalues, dpsi):
        """The chunk's raw sums and their tangent along ``(dx, dpsi)``."""
        return jax.jvp(
            lambda y, states: _separable(_move(local(big, geometry, rowset), kind, y),
                                         states, weights, eigenvalues,
                                         _all(states))[1:],
            (x, psi), (dx, embed(psi, dpsi)))

    def global_(x, dx, big, geometry, rowset, sums, dsums):
        """``(grad, d grad)`` of the whole-cell terms in ``(x, b, rho, ns)``."""
        gradient = jax.grad(
            lambda y, b, rho, ns: _global(_move(local(big, geometry, rowset), kind, y),
                                          b, rho, ns),
            argnums=(0, 1, 2, 3))
        return jax.jvp(gradient, (x,) + tuple(sums), (dx,) + tuple(dsums))

    def pull(x, dx, big, geometry, rowset, psi, weights, eigenvalues, dpsi, cotangent,
             dcotangent):
        """The ``jvp`` of the chunk's pull-back ``d/dx [E_c + g . sums_c]``
        along ``(dx, dpsi, dg)``."""
        def slope(y, states, g):
            g_b, g_rho, g_ns = g

            def energy(z):
                value, b, rho, ns = _separable(
                    _move(local(big, geometry, rowset), kind, z), states, weights,
                    eigenvalues, _all(states))
                coupled = jnp.sum(g_rho * rho) + sum(
                    jnp.sum(gb * part) for gb, part in zip(g_b, b)
                    if part is not None)
                if ns is not None:
                    coupled = coupled + jnp.sum(g_ns * ns)
                return value + coupled
            return jax.grad(energy)(y)

        return jax.jvp(slope, (x, psi, cotangent),
                       (dx, embed(psi, dpsi), dcotangent))[1]

    passes = {"forward": jax.jit(forward), "global": jax.jit(global_),
              "pull": jax.jit(pull)}
    store[name] = passes
    return passes, geometry


def chunked_gradient_tangent(calculation, state: FrozenState, kind: str, x, dx,
                             dstates, k_batch: int | None = None):
    """``d/dt [dE/dx](x + t dx, psi + t dstates)``, the k axis walked.

    :func:`chunked_gradient`'s split differentiated once more: a forward walk
    for the raw sums and their tangent, one ``jvp`` of the whole-cell terms'
    gradient at the whole sums, and a walk whose ``jvp`` of each chunk's
    pull-back carries the coordinate, the states and the global cotangent. It is
    what the elastic constants are with ``kind = "strain"`` and the strain
    response's ``dpsi`` as ``dstates`` (:func:`~defumat.response.elastic.
    elastic_constants`), with the energy's diagonal constraint, as there.

    ``dstates`` is ``(nspin, nk, nocc, ndim)``, the first ``nocc`` bands' tangent
    (a host store or a device array); the rest of the bands have none.
    """
    reject_potential_only(calculation)
    passes, geometry = _tangent_compiled(calculation, kind)
    big = hoisted(calculation)
    nk = calculation.system.kpoints.nk
    batch = (calculation.k_batch if k_batch is None else k_batch) or nk
    chunks = list(k_chunks(nk, batch))
    host_weights = np.asarray(state.weights)
    sums = dsums = None
    for rows, live in chunks:
        psi, weights, eigenvalues = _chunk(state, rows, live, weights=host_weights)
        value, tangent = passes["forward"](
            x, dx, big, geometry, row_leaves(calculation, rows), psi, weights,
            eigenvalues, _rows_of(dstates, rows))
        sums, dsums = _add(sums, value), _add(dsums, tangent)
    gradient, dgradient = passes["global"](
        x, dx, big, geometry, row_leaves(calculation, chunks[0][0]), sums, dsums)
    column = dgradient[0]
    for rows, live in chunks:
        psi, weights, eigenvalues = _chunk(state, rows, live, weights=host_weights)
        column = column + passes["pull"](
            x, dx, big, geometry, row_leaves(calculation, rows), psi, weights,
            eigenvalues, _rows_of(dstates, rows), tuple(gradient[1:]),
            tuple(dgradient[1:]))
    return column


#: The :class:`~defumat.scf.driver.Calculation` attributes that carry a k
#: index -- what :meth:`~defumat.scf.driver.Calculation.at_rows` slices. Inside
#: the compiled passes they are replaced by one chunk's rows, passed as traced
#: leaves, so that ``at_positions`` and ``at_strain`` rebuild the projector
#: core, ``|k+G|^2`` and ``wfcU`` for the chunk's k-points only. Without it
#: every chunk's backward pass rebuilt every k-point's core -- measured on the
#: GTX 1060, eight-atom silicon at 64 k-points: the chunked stress 16x slower
#: than the single pass and its peak still growing with the mesh.
#: ``system`` stands for its k-point list alone: the rest of it (the cell, the
#: structure's static bookkeeping) must stay what the passes closed over.
ROW_FIELDS = ("system", "basis", "basis_kpoints", "_kcrystal", "kinetic",
              "fft_index", "fft_index_minus", "kplusg", "sticks",
              "projector_core", "projectors", "wfcU")


def row_leaves(calculation, rows) -> tuple:
    """One chunk's k-indexed attributes, from ``calculation.at_rows(rows)``.

    The plane-wave spheres' ``npw`` is host bookkeeping kept as a *static*
    field; nothing in the passes reads it and a per-chunk value would compile
    the passes once per chunk, so it is dropped here.
    """
    import dataclasses

    sub = calculation.at_rows(rows)
    planewaves = dataclasses.replace(sub.basis.planewaves, npw=())
    sub.basis = eqx.tree_at(lambda basis: basis.planewaves, sub.basis, planewaves)
    sub._kcrystal = jnp.asarray(sub._kcrystal)
    return tuple(sub.system.kpoints if name == "system" else getattr(sub, name)
                 for name in ROW_FIELDS)


def with_rows(calculation, leaves):
    """``calculation`` carrying one chunk's k-indexed attributes (:func:`row_leaves`)."""
    import copy

    here = copy.copy(calculation)
    for name, value in zip(ROW_FIELDS, leaves):
        if name == "system":
            value = eqx.tree_at(lambda system: system.kpoints, here.system, value)
        setattr(here, name, value)
    # The chunk's projectors are its own rows, stored whole (``at_rows``); a
    # k-point pool's row list indexes the whole set and must not follow them.
    here.projector_rows = None
    return here


def _all(psi):
    """Every k-point of a chunk, as the rows of its own row-subset calculation."""
    return jnp.arange(psi.shape[1])
