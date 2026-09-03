"""The orbital magnetization of a crystal, by the modern theory.

A magnet's moment has two parts. The spin part is an integral of the spin
density over the cell and every code prints it. The **orbital** part is not an
integral of anything local: the current a Bloch state carries circulates
through the whole crystal, so the naive ``r x j`` integral depends on where the
cell is cut and is not a property of the material at all -- the same difficulty
the electric polarization has, and with the same resolution. What is well
defined is a k-space quantity (Thonhauser, Ceresoli, Vanderbilt and Resta, PRL
95, 137205 (2005); Ceresoli, Thonhauser, Vanderbilt and Resta, PRB 74, 024408
(2006); Malashevich, Souza, Coh and Vanderbilt, NJP 12, 053032 (2010)):

    M = (e / 2 hbar c) Im sum_n int [dk] <d_k u_n| x (H_k + E_nk - 2 mu)|d_k u_n>

-- the sum of a *local circulation*, the current going round inside each cell,
and an *itinerant circulation*, the centre of mass of a wavepacket drifting
along the boundary. Neither is separately measurable and their sum is.

``pw.x`` computes it (``PW/src/orbm_kubo.f90``, reached by ``lorbm`` in a
non-self-consistent run over a uniform grid) and this is a transcription of
that routine, which is itself the discretisation the papers above derive. Two
things about it are worth stating before the formulae.

**The chemical potential is not in what either code prints.** ``orbm_kubo``
imports ``ef`` and never uses it, so what it reports is ``M(mu = 0)``. That is
not an approximation: the ``-2 mu`` term integrates to ``mu`` times the Chern
vector, which vanishes for any crystal whose occupied manifold is topologically
trivial, and is otherwise a statement about where the zero of energy is. It is
carried here as :attr:`OrbitalMagnetization.dm_dmu` -- computed, reported, and
added only if a caller passes a ``mu`` -- because the same sum is also how a
Chern number falls out of this machinery for free.

**QE's two printed terms are not the paper's LC/IC split.** The paper writes
``LC = Im<d u|(H - E)|d u>`` and ``IC = 2 Im<d u|(E - mu)|d u>``; the Fortran
prints ``Im<d u|H|d u>`` and ``Im<d u|E|d u>`` (the second in its
gauge-covariant matrix form). The two splits differ by ``Im<d u|E|d u>`` term
by term and agree in the sum, which is the number that means anything. This
module follows the Fortran, so that its two terms can be compared with
``pw.x``'s two rather than only their total.

The discretisation
------------------

Everything is built from the **dual states** of the neighbouring manifolds,

    |w^{d,s}_m> = sum_n (M^{-1})_{nm} |u_{k + s b_d, n}>,
    M_{mn} = <u_{k,m} | u_{k + s b_d, n}>,

which satisfy ``<u_{k,l}|w_m> = delta_{lm}``: the covariant finite difference,
and the reason rule D4 is satisfied here without any special handling. The
textbook expression carries a per-band ``E_n``, which is not gauge invariant
inside a degenerate multiplet; the dual construction replaces it by the
occupied-block matrix ``<u_n|H|u_m>`` and every quantity below is invariant
under any unitary mixing of the manifold. It also means nothing is ever divided
by ``E_n - E_m``.

With ``D_d = w^{d,+} - w^{d,-}`` (an unnormalised central difference, twice the
derivative times the step ``1/N_d``) and ``(i, j) = (l + 1, l + 2) mod 3``:

    S^LC_l   = Im sum_n <D_i n| H_k |D_j n>
    S^IC_l   = Im tr( <u|H|u> G^{ij} ),   G^{ij}_{mn} = <D_i m|D_j n>
    S^curv_l = Im tr( G^{ij} )

summed over the k-points of the mesh. Those three sums are what
:func:`orbital_magnetization_sums` returns -- dimensionless, with no lattice in
them -- and :func:`orbital_magnetization` turns them into Bohr magnetons per
cell with

    M_a = (Omega / (4 (2 pi)^3)) sum_l (b_l)_a / N_l * S_l.

The vector direction being ``b_l`` while the derivatives are along the *other*
two directions is not a slip: ``d/dk`` in reduced coordinates carries the direct
lattice vectors, and ``a_i x a_j = Omega b_l / (2 pi)``, so the cross product of
the two derivative directions points along ``b_l`` whatever the lattice is.

Refused by name
---------------

**Ultrasoft and PAW**, which ``setup.f90:130`` refuses too: the overlaps between
neighbouring manifolds would need ``q_ij(b)`` (which
:mod:`defumat.topology.augmentation` has) *and* the dual states would then have
to be dual in the ``S`` metric, with ``H`` contracted against them accordingly
-- terms nothing norm-conserving can validate. **A non-isolated manifold**, i.e.
a metal: the dual construction inverts the overlap of two manifolds and there is
no manifold to speak of if the band count changes across the zone. And a mesh
with **two** divisions along a direction that carries a derivative, where a
point's two neighbours are the same k-point and the difference is an alias
rather than a derivative. **One** division is different and is allowed: the
derivative along that direction is set to zero, which is what a slab with a
single k-point across the vacuum means.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from defumat.batching import map_k, resolve_k_batch

__all__ = [
    "OrbitalMagnetization",
    "orbital_magnetization",
    "orbital_magnetization_sums",
]

#: ``(l, i, j)``: the derivative directions each Cartesian assembly pairs, in
#: the cyclic order that makes the result a cross product.
CYCLIC: tuple[tuple[int, int, int], ...] = ((0, 1, 2), (1, 2, 0), (2, 0, 1))


@dataclass(frozen=True)
class OrbitalMagnetization:
    """The two Kubo terms, in Bohr magnetons per cell.

    ``lc`` and ``ic`` are ``pw.x``'s ``M_LC`` and ``M_IC``; ``total`` adds them
    and applies the chemical potential's term, which is zero unless the manifold
    carries a Chern vector.
    """

    lc: np.ndarray  # (3,) mu_B/cell
    ic: np.ndarray  # (3,)
    #: ``dM/dmu`` in mu_B/cell per Ry -- the ``-2 mu`` term's coefficient.
    #: Proportional to the Chern vector and therefore zero for an ordinary
    #: insulator, which is why ``pw.x`` can leave ``mu`` out of what it prints.
    dm_dmu: np.ndarray  # (3,)
    #: The chemical potential the total is quoted at, in Ry. Zero is ``pw.x``'s
    #: convention and the default.
    mu: float = 0.0
    #: The Chern vector implied by the same discretisation: ``C_l`` is the Chern
    #: number of the plane normal to crystal direction ``l``. An integer for an
    #: isolated manifold, up to the mesh's own error -- and a *different*
    #: discretisation from :func:`~defumat.topology.invariants.chern_number`,
    #: which is exact on any mesh. Comparing the two is the check that the
    #: dual-state construction here is the derivative it claims to be.
    chern: np.ndarray = field(default_factory=lambda: np.zeros(3))
    #: The three raw zone sums, before any lattice or unit factor: what a model
    #: check compares and what the prefactor is applied to.
    sums: dict = field(default_factory=dict)
    #: Crystal directions whose derivative was set to zero because the mesh has
    #: a single division along them.
    flat_directions: tuple[int, ...] = ()
    #: The smallest ``|det M|`` over every neighbour overlap of the mesh, as a
    #: diagnostic: the dual states are that matrix inverted, so a small value
    #: means the mesh is too coarse to follow the manifold and the answer is
    #: noise. It is the true minimum, not an average -- one bad overlap is what
    #: there is to catch.
    smallest_determinant: float = 1.0

    @property
    def total(self) -> np.ndarray:
        """``M_LC + M_IC + mu dM/dmu``, in Bohr magnetons per cell."""
        return self.lc + self.ic + self.mu * self.dm_dmu


def orbital_magnetization_sums(states, mesh, *, k_batch="default") -> dict:
    """The three zone sums of the module docstring, ``(3,)`` each.

    ``states`` is the occupied manifold at **every** point of ``mesh``, in the
    mesh's own order, and must answer
    :meth:`~defumat.topology.states.StateSet.transport_plan` and
    :meth:`~defumat.topology.states.StateSet.hamiltonian_matvec`. Nothing here
    knows about a lattice or a unit: it is the discrete expression and no more,
    which is what lets the same function be checked on a tight-binding model
    where the answer is an integer.

    **Memory.** The gather plans are ``(nk, dim)`` integers per neighbour, six
    of them -- megabytes beside the states themselves, which are the whole mesh
    at once here rather than one row at a time. A derivative needs both
    neighbours of every point, so there is no streaming order that keeps fewer
    than a plane of them resident, and a plane is what the caller would have to
    hold anyway.
    """
    divisions = tuple(int(n) for n in mesh.divisions)
    flat = tuple(d for d, n in enumerate(divisions) if n == 1)
    aliased = [d for d, n in enumerate(divisions) if n == 2]
    if aliased:
        raise ValueError(
            f"the mesh has two divisions along direction(s) {aliased}: a point's "
            "two neighbours there are the same k-point, so the central difference "
            "is an alias of the derivative rather than the derivative. Use one "
            "division (the derivative is then taken as zero) or at least three"
        )

    nk = mesh.nk
    if states.nk != nk:
        raise ValueError(
            f"the state set has {states.nk} k-points and the mesh {nk}; an "
            "orbital magnetization is assembled at every point of its own mesh"
        )

    matvec = states.hamiltonian_matvec()
    vectors = jnp.asarray(states.coefficients)
    live = [d for d in range(3) if d not in flat]

    plans, targets = {}, {}
    for direction in live:
        for sign in (1, -1):
            pairs = [mesh.neighbour(i, direction, sign) for i in range(nk)]
            index, phase = states.transport_plan(
                [(i, j, shift) for i, (j, shift) in enumerate(pairs)]
            )
            plans[direction, sign] = (
                jnp.asarray(np.asarray(index, dtype=np.int32)), phase,
            )
            targets[direction, sign] = jnp.asarray([j for j, _ in pairs])

    dim = vectors.shape[-1]
    nbnd = vectors.shape[-2]

    def body(ik):
        u = vectors[ik]
        hu = matvec(u, ik)
        huu = jnp.einsum("na,ma->nm", u.conj(), hu)

        difference, logdet = {}, []
        for direction in live:
            duals = []
            for sign in (1, -1):
                index, phase = plans[direction, sign]
                neighbour = vectors[targets[direction, sign][ik]]
                transported = neighbour[:, index[ik]] * phase[ik]
                overlap = jnp.einsum("ma,na->mn", u.conj(), transported)
                _, magnitude = jnp.linalg.slogdet(overlap)
                logdet.append(magnitude)
                # ``solve(M^T, C)`` is ``sum_n (M^{-1})_{nm} C_n`` -- the dual
                # basis of the manifold at ``k``, band ``m`` in row ``m``.
                duals.append(jnp.linalg.solve(overlap.T, transported))
            difference[direction] = duals[0] - duals[1]
        zero = jnp.zeros((nbnd, dim), dtype=vectors.dtype)
        for direction in flat:
            difference[direction] = zero

        applied = {
            direction: matvec(difference[direction], ik) if direction in live else zero
            for direction in range(3)
        }

        lc, ic, curvature = [], [], []
        for _, first, second in CYCLIC:
            di, dj = difference[first], difference[second]
            gram = jnp.einsum("ma,na->mn", di.conj(), dj)
            lc.append(jnp.imag(jnp.sum(di.conj() * applied[second])))
            ic.append(jnp.imag(jnp.trace(huu @ gram)))
            curvature.append(jnp.imag(jnp.trace(gram)))
        return {
            "lc": jnp.stack(lc),
            "ic": jnp.stack(ic),
            "curvature": jnp.stack(curvature),
            "logdet": jnp.min(jnp.stack(logdet)) if logdet else jnp.asarray(0.0),
        }

    # ``map_k`` rather than ``sum_k``, and the reason is the diagnostic: what
    # each k-point contributes is ten floats, so stacking the whole axis costs
    # nothing and keeps the smallest determinant a *minimum* -- an accumulator
    # can only sum, and a mesh is bad because of its worst overlap rather than
    # because of its average one.
    per_k = map_k(body, jnp.arange(nk), batch=resolve_k_batch(k_batch))
    return {
        "lc": np.asarray(per_k["lc"]).sum(axis=0),
        "ic": np.asarray(per_k["ic"]).sum(axis=0),
        "curvature": np.asarray(per_k["curvature"]).sum(axis=0),
        "determinant": float(np.exp(np.asarray(per_k["logdet"]).min())),
        "flat_directions": flat,
    }


def orbital_magnetization(
    states, mesh, cell, *, mu: float = 0.0, degeneracy: int = 1,
    k_batch="default",
) -> OrbitalMagnetization:
    """The zone sums turned into Bohr magnetons per cell.

    ``degeneracy`` is how many electrons a band of ``states`` holds: one for a
    spinor calculation, which is the only kind that has an orbital magnetization
    at all, and two for a spin-degenerate scalar one -- where the answer is zero
    by time reversal whatever the factor is. ``pw.x`` applies no such factor;
    the difference therefore only ever multiplies zero, and it is applied here
    because a factor that is right for the wrong reason is how the ``degspin``
    trap of :mod:`defumat.response.conductivity` got in twice.
    """
    sums = orbital_magnetization_sums(states, mesh, k_batch=k_batch)
    bg = np.asarray(cell.bg, dtype=float)  # rows b_1, b_2, b_3 in 1/bohr
    volume = float(cell.volume)
    divisions = np.asarray(mesh.divisions, dtype=float)

    # Omega / (4 (2 pi)^3): the 4 is the two central differences, each left
    # unnormalised, and the (2 pi)^3 / Omega is the Brillouin zone's volume.
    # Rydberg energies and a Bohr magneton of 1/2 in atomic units cancel, which
    # is what ``orbm_kubo``'s comment says and why nothing converts here.
    prefactor = degeneracy * volume / (4.0 * (2.0 * np.pi) ** 3)
    weights = prefactor * bg / divisions[:, None]  # (3, 3): l -> vector

    lc = np.einsum("l,la->a", sums["lc"], weights)
    ic = np.einsum("l,la->a", sums["ic"], weights)
    dm_dmu = -2.0 * np.einsum("l,la->a", sums["curvature"], weights)

    # ``S^curv_l = -4 pi N_l C_l``. The difference ``D`` is ``2/N`` times the
    # derivative and the curvature is ``-2 Im<d_i u|d_j u>`` (the sign
    # :func:`~defumat.topology.links.berry_phase` fixes), so one plane's sum is
    # ``-4 pi C_l``; the mesh holds ``N_l`` copies of that plane and they add.
    chern = -np.asarray(sums["curvature"]) / (4.0 * np.pi * divisions)

    return OrbitalMagnetization(
        lc=lc,
        ic=ic,
        dm_dmu=dm_dmu,
        mu=float(mu),
        chern=chern,
        sums={k: sums[k] for k in ("lc", "ic", "curvature")},
        flat_directions=tuple(sums["flat_directions"]),
        smallest_determinant=float(sums["determinant"]),
    )
