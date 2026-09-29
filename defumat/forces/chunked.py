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
gradients do, so no per-k table becomes a constant of the executable.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from defumat.batching import k_chunks
from defumat.forces.energy import (
    FrozenState, _kinetic_energy, _norms, _projector_energies,
    _spinor_projector_energies, hoisted, reject_potential_only,
    reject_spinor_spiral, with_hoisted,
)
from defumat.hubbard.energy import hubbard_energy
from defumat.scf.potential import total_charge
from defumat.scf.streaming import is_host_store

__all__ = ["chunked_gradient", "wants_chunks"]


def wants_chunks(calculation, state: FrozenState) -> bool:
    """Whether the force and stress should walk the k axis rather than tape it whole.

    Where the state lives in host memory (a streamed store), and in memory
    mode wherever the chunk is smaller than the k-set. Speed mode, the CPU
    default every validated number was taken in, keeps the single pass.
    """
    nk = calculation.system.kpoints.nk
    if is_host_store(state.wavefunctions):
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
    """One chunk of a ``(nspin, nk, ...)`` state array on the device."""
    if isinstance(array, np.ndarray):
        return jax.device_put(np.ascontiguousarray(array[:, rows]))
    return jnp.asarray(array)[:, jnp.asarray(rows)]


def _chunk(state: FrozenState, rows, live):
    """``psi``, the weights with the padding zeroed, and the eigenvalues of one chunk."""
    weights = np.array(state.weights[:, rows])
    weights[:, live:] = 0.0
    return (_rows_of(state.wavefunctions, rows), jnp.asarray(weights),
            jnp.asarray(np.asarray(state.eigenvalues)[:, rows]))


def _add(total, part):
    return part if total is None else jax.tree_util.tree_map(jnp.add, total, part)


def _compiled(calculation, kind: str) -> dict:
    """The three compiled passes for ``calculation``, built once and cached on it.

    **The force's passes are inherited, the stress's are not**, which is the
    single-pass gradients' own rule. The force depends on the geometry only
    through ``x`` -- everything position-dependent is rebuilt inside, and
    :func:`~defumat.forces.energy.with_hoisted` brings the moved calculation's
    large arrays -- so a calculation moved by ``at_positions``, which copies
    the instance dict, reuses the compiled passes at every step of a
    relaxation instead of retracing them. The movers that change what the
    passes close over (``at_strain``, ``at_kcart``, ``at_kpoints``) drop the
    entry. The stress's passes strain the cell they closed over, so they are
    keyed on the calculation itself.
    """
    cached = calculation.__dict__.get("_chunked_gradient")
    entries = {} if cached is None else {
        name: passes for name, passes in cached[1].items()
        if name == "positions" or cached[0] is calculation}
    if kind in entries:
        return entries[kind]
    cached = (calculation, entries)
    calculation._chunked_gradient = cached

    def local(big, rowset):
        """The calculation on one chunk's k-points, from traced leaves."""
        return with_rows(with_hoisted(calculation, big), rowset)

    def sums(x, big, rowset, psi, weights, eigenvalues):
        moved = _move(local(big, rowset), kind, x)
        return _separable(moved, psi, weights, eigenvalues, _all(psi))[1:]

    def whole(x, big, rowset, becsum_, rho, ns):
        moved = _move(local(big, rowset), kind, x)
        return _global(moved, becsum_, rho, ns)

    def pull(x, big, rowset, psi, weights, eigenvalues, cotangent):
        (energy, *parts), back = jax.vjp(
            lambda x: _separable(_move(local(big, rowset), kind, x),
                                 psi, weights, eigenvalues, _all(psi)),
            x,
        )
        (slope,) = back((jnp.ones_like(energy), *cotangent))
        return energy, slope

    passes = {
        "sums": jax.jit(sums),
        "global": jax.jit(jax.value_and_grad(whole, argnums=(0, 3, 4, 5))),
        "pull": jax.jit(pull),
    }
    cached[1][kind] = passes
    return passes


def chunked_gradient(calculation, state: FrozenState, kind: str, x,
                     k_batch: int | None = None):
    """``(E, dE/dx)`` at ``x``, the k axis walked ``k_batch`` at a time.

    ``k_batch`` defaults to the calculation's own chunk.

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
    passes = _compiled(calculation, kind)
    big = hoisted(calculation)
    nk = calculation.system.kpoints.nk
    batch = (calculation.k_batch if k_batch is None else k_batch) or nk
    chunks = list(k_chunks(nk, batch))

    becsum_ = rho = ns = None
    for rows, live in chunks:
        psi, weights, eigenvalues = _chunk(state, rows, live)
        part = passes["sums"](x, big, row_leaves(calculation, rows), psi,
                              weights, eigenvalues)
        becsum_, rho, ns = (_add(becsum_, part[0]), _add(rho, part[1]),
                            _add(ns, part[2]))

    # The global terms read nothing with a k index; any one chunk's rows stand
    # in, so that the moved calculation's per-k rebuilds are one chunk's.
    e_glob, (g_x, g_b, g_rho, g_ns) = passes["global"](
        x, big, row_leaves(calculation, chunks[0][0]), becsum_, rho, ns)
    energy = e_glob + state.entropy
    gradient = g_x
    for rows, live in chunks:
        psi, weights, eigenvalues = _chunk(state, rows, live)
        value, slope = passes["pull"](x, big, row_leaves(calculation, rows),
                                      psi, weights, eigenvalues,
                                      (g_b, g_rho, g_ns))
        energy = energy + value
        gradient = gradient + slope
    return energy, gradient


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
    return here


def _all(psi):
    """Every k-point of a chunk, as the rows of its own row-subset calculation."""
    return jnp.arange(psi.shape[1])
