"""The ultrasoft augmentation charge, in reciprocal space.

An ultrasoft pseudopotential relaxes norm conservation: the pseudo orbital
inside the core no longer carries the same charge as the all-electron one, so
``sum_G |c_G|^2`` is not the number of electrons and ``|psi(r)|^2`` is not the
density. Both are repaired by the same object, the augmentation charge

    Q_ij^a(r) = phi_i^AE*(r-tau_a) phi_j^AE(r-tau_a)
              - phi_i^PS*(r-tau_a) phi_j^PS(r-tau_a)

which the pseudopotential file tabulates radially. Three quantities follow from
it, and they are exactly the three things this module computes:

* ``qq_ij = int Q_ij(r) dr`` makes the overlap operator
  ``S = 1 + sum |beta_i> qq_ij <beta_j|`` -- the eigenproblem becomes generalised
  (``upflib/init_us_1.f90``, ``compute_qqr``);
* ``rho_aug(G) = sum_a sum_ij becsum_ij^a Q_ij(G) e^{-i G tau_a}`` is added to
  the density every time it is rebuilt (``PW/src/addusdens.f90``);
* ``D_ij^a = D_ij^(0) + int V_eff(r) Q_ij^a(r) dr`` replaces the file's fixed
  ``D_ij`` and has to be rebuilt whenever the potential changes
  (``PW/src/newd_acc.f90``).

The reciprocal-space form is ``upflib/qvan2.f90``:

    Q_ij(G) = sum_LM (-i)^L ap(LM, lm_i, lm_j) Y_LM(G) Q^L_{n_i n_j}(|G|)

with the coupling coefficients from :mod:`pypresso.pseudo.coupling` and the
radial transforms

    Q^L_nm(q) = 4 pi / Omega int_0^{r_kkbeta} dr j_L(q r) [r^2 Q^L_nm(r)]

evaluated directly at each ``|G|``, as everything else in
:mod:`pypresso.pseudo.formfactors` is, rather than interpolated from QE's
``dq = 0.01`` table -- same reason: it keeps the augmentation charge a
differentiable function of the cell.

**The integration range is ``kkbeta``, not the 10-bohr mesh** the local potential
and the atomic charge use. Q is identically zero outside the augmentation
sphere, so extending the range only adds the tabulated noise beyond it; QE
integrates to ``kkbeta`` and so does this. The check that it is right is that
``Omega * Q_ij(G=0)`` reproduces the file's own ``PP_Q`` values.
"""

from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from pypresso.basis.gvectors import GVectors
from pypresso.pseudo.coupling import harmonic_products
from pypresso.pseudo.harmonics import real_spherical_harmonics
from pypresso.pseudo.projectors import projector_channels
from pypresso.pseudo.radial import simpson_weights, spherical_bessel
from pypresso.pseudo.upf import Pseudopotential
from pypresso.system.cell import Cell
from pypresso.system.structure import Structure
from pypresso.units import FPI

__all__ = ["AugmentationCharge", "build_augmentation", "radial_augmentation_transforms"]


class AugmentationCharge(eqx.Module):
    """``Q_ij(G)`` for every species, and what the SCF loop does with it.

    ``qgm[t]`` is ``(nh_t, nh_t, ngm)`` on the **dense** G-vector set -- the
    augmentation charge is sharp, and representing it is the entire reason the
    dense grid exists. ``phases`` is ``(nat, ngm)``, the structure factor
    ``e^{-i G . tau_a}`` of each atom.
    """

    qgm: tuple  # per species, (nh, nh, ngm) complex
    qq: tuple  # per species, (nh, nh) real -- Omega * Q_ij(G=0)
    phases: jnp.ndarray  # (nat, ngm) complex
    volume: jnp.ndarray  # bohr^3
    species_atoms: tuple = eqx.field(static=True)  # atom indices, per species
    channel_offsets: tuple = eqx.field(static=True)  # first channel of each atom
    nkb: int = eqx.field(static=True)

    @property
    def ntyp(self) -> int:
        return len(self.qgm)

    def charge(self, becsum: tuple) -> jnp.ndarray:
        """``rho_aug(G)`` on the dense grid, from the per-atom ``becsum``.

        ``becsum[t]`` is ``(nat_t, nh_t, nh_t)``. Summing the atoms into the
        structure factor *before* contracting with ``Q_ij(G)`` is what
        ``addusdens_g`` does with its ``DGEMM`` over ``nab``, and it is the
        difference between one ``(nh, nh, ngm)`` intermediate and ``nat`` of
        them.
        """
        total = None
        for t, (q, atoms) in enumerate(zip(self.qgm, self.species_atoms)):
            if q.shape[0] == 0 or not atoms:
                continue
            contribution = _species_charge(q, becsum[t], self.phases[jnp.asarray(atoms)])
            total = contribution if total is None else total + contribution
        if total is None:
            return jnp.zeros(self.phases.shape[-1], dtype=self.phases.dtype)
        return total

    def integrals(self, potential_g: jnp.ndarray) -> tuple:
        """``int V(r) Q_ij^a(r) dr`` for every atom -- ``newd``'s contribution.

        Returns one ``(nat_t, nh_t, nh_t)`` real array per species, in the same
        layout ``charge`` consumes. ``potential_g`` is the *total* local
        potential (``vltot + v_scf``) on the dense G set, which is what
        ``newq_acc`` transforms.
        """
        result = []
        for q, atoms in zip(self.qgm, self.species_atoms):
            if q.shape[0] == 0 or not atoms:
                result.append(jnp.zeros((len(atoms), q.shape[0], q.shape[0])))
                continue
            result.append(
                _species_integrals(
                    q, potential_g, self.phases[jnp.asarray(atoms)], self.volume
                )
            )
        return tuple(result)

    def at_positions(self, positions: jnp.ndarray, gcart: jnp.ndarray):
        """The same augmentation charge with the atoms somewhere else.

        ``Q_ij(G)`` is a property of the species; only the structure factor
        moves, so a new geometry costs one complex exponential per atom.
        """
        phases = _atom_phases(gcart, positions).astype(self.phases.dtype)
        return eqx.tree_at(lambda a: a.phases, self, phases)

    def block_matrix(self, blocks: tuple) -> jnp.ndarray:
        """Per-atom ``(nh, nh)`` blocks -> the ``(nkb, nkb)`` matrix ``H`` uses.

        The nonlocal term is block diagonal over atoms; the Hamiltonian stores
        it as one dense matrix because ``nkb`` is small next to ``npw`` and a
        single ``einsum`` is worth more than the zeros are worth avoiding.
        """
        matrix = jnp.zeros((self.nkb, self.nkb), dtype=blocks[0].dtype)
        for block, atoms in zip(blocks, self.species_atoms):
            for n, atom in enumerate(atoms):
                start = self.channel_offsets[atom]
                stop = start + block.shape[-1]
                matrix = matrix.at[start:stop, start:stop].set(block[n])
        return matrix


@jax.jit
def _species_charge(qgm, becsum, phases):
    """``sum_a sum_ij becsum_ij^a Q_ij(G) e^{-i G tau_a}`` for one species."""
    weighted = jnp.einsum("aij,ag->ijg", becsum.astype(phases.dtype), phases)
    return jnp.einsum("ijg,ijg->g", qgm, weighted)


@jax.jit
def _species_integrals(qgm, potential_g, phases, volume):
    """``Omega * Re sum_G conj(Q_ij(G)) V(G) e^{+i G tau_a}``."""
    shifted = potential_g[None, :] * jnp.conj(phases)  # (nat, ngm)
    return volume * jnp.real(jnp.einsum("ijg,ag->aij", jnp.conj(qgm), shifted))


def radial_augmentation_transforms(
    pseudo: Pseudopotential, q, omega: float, nl: int
) -> jnp.ndarray:
    """``Q^L_{nm}(q)``, shaped ``(nbeta, nbeta, nl, nq)``.

    The triangle rule and parity decide which ``(n, m, L)`` are stored at all:
    ``|l_n - l_m| <= L <= l_n + l_m`` with ``L + l_n + l_m`` even, exactly the
    condition ``init_tab_qrad`` applies. The rest stay zero.
    """
    augmentation = pseudo.augmentation
    if augmentation is None or augmentation.qfuncl is None:
        raise NotImplementedError(
            f"{pseudo.element}: this pseudopotential's augmentation charge is stored "
            "in the pre-2.0 qfcoef form, which is not implemented"
        )

    kkbeta = pseudo.kkbeta
    r = jnp.asarray(pseudo.r[:kkbeta])
    weights = simpson_weights(jnp.asarray(pseudo.rab[:kkbeta]))
    q = jnp.atleast_1d(jnp.asarray(q))

    nbeta = pseudo.nbeta
    ls = [projector.l for projector in pseudo.projectors]
    prefactor = FPI / omega

    rows = []
    for l in range(nl):
        pairs, functions = [], []
        for nb in range(nbeta):
            for mb in range(nbeta):
                allowed = (
                    abs(ls[nb] - ls[mb]) <= l <= ls[nb] + ls[mb]
                    and (l + ls[nb] + ls[mb]) % 2 == 0
                    and l < augmentation.qfuncl.shape[2]
                )
                if allowed:
                    pairs.append((nb, mb))
                    functions.append(augmentation.qfuncl[nb, mb, l, :kkbeta])
        table = jnp.zeros((nbeta, nbeta) + q.shape)
        if pairs:
            values = _qrad_kernel(q, r, weights, jnp.asarray(np.stack(functions)), prefactor, l)
            index = np.asarray(pairs)
            table = table.at[index[:, 0], index[:, 1]].set(values)
        rows.append(table)

    return jnp.stack(rows, axis=2)


@partial(jax.jit, static_argnames=("l",))
def _qrad_kernel(q, r, weights, functions, prefactor, l):
    """``4 pi / Omega int dr j_l(q r) [r^2 Q^l(r)]`` for a stack of ``Q``."""
    argument = q[:, None] * r[None, :]
    bessel = spherical_bessel(l, argument)  # (nq, mesh)
    return prefactor * jnp.einsum("fm,qm,m->fq", functions, bessel, weights)


def build_augmentation(
    pseudos: tuple[Pseudopotential, ...],
    structure: Structure,
    cell: Cell,
    gvectors: GVectors,
) -> AugmentationCharge | None:
    """Assemble ``Q_ij(G)`` for every ultrasoft species. ``None`` if there are none.

    ``gvectors`` must be the **dense** set.
    """
    if not any(pseudos[t].is_ultrasoft for t in structure.types):
        return None

    if gvectors.gamma_only:
        # Gamma-only storage keeps one G of each (G, -G) pair, so every sum over
        # the sphere needs QE's doubling (``fact = 2`` in ``newq_acc``, with the
        # G = 0 term counted once). None of that is written here, and nothing in
        # the reference set would catch it, so the combination is refused rather
        # than silently computed with half the augmentation charge.
        raise NotImplementedError(
            "gamma-only storage with an ultrasoft or PAW pseudopotential is not "
            "implemented; run with an explicit k-point at Gamma instead"
        )

    lmax = max(p.lmax for p in pseudos)
    ap = harmonic_products(lmax)  # ((2lmax+1)^2, (lmax+1)^2, (lmax+1)^2)
    nl = 2 * lmax + 1

    gcart = gvectors.cartesian(cell)
    gmod = jnp.sqrt(jnp.sum(gcart**2, axis=1))
    ylm = real_spherical_harmonics(gcart, 2 * lmax)  # (ngm, (2lmax+1)^2)
    volume = float(cell.volume)

    qgm, qq = [], []
    for pseudo in pseudos:
        channels = projector_channels(pseudo)
        if not pseudo.is_ultrasoft or not channels:
            qgm.append(jnp.zeros((0, 0, gvectors.ngm), dtype=cell.precision.complex))
            qq.append(jnp.zeros((0, 0)))
            continue

        nl_species = min(nl, pseudo.augmentation.nqlc) if pseudo.augmentation else nl
        radial = radial_augmentation_transforms(pseudo, gmod, volume, nl_species)

        beta_of = np.array([nb for nb, _, _ in channels])
        lm_of = np.array([lm for _, _, lm in channels])
        # ap restricted to the (lm_i, lm_j) actually present, as qvan2's
        # lpl/lpx lists restrict the sum.
        coefficients = jnp.asarray(ap[:, lm_of[:, None], lm_of[None, :]])

        values = _assemble_qgm(
            coefficients, ylm, radial, jnp.asarray(beta_of), nl_species
        )
        qgm.append(values.astype(cell.precision.complex))
        qq.append(volume * jnp.real(values[:, :, 0]))

    phases = _atom_phases(gcart, structure.positions)

    types = np.asarray(structure.types)
    species_atoms = tuple(
        tuple(int(a) for a in np.flatnonzero(types == t)) for t in range(structure.ntyp)
    )
    sizes = [len(projector_channels(pseudos[t])) for t in types]
    offsets = tuple(int(o) for o in np.cumsum([0] + sizes)[:-1])

    return AugmentationCharge(
        qgm=tuple(qgm),
        qq=tuple(qq),
        phases=phases.astype(cell.precision.complex),
        volume=jnp.asarray(volume),
        species_atoms=species_atoms,
        channel_offsets=offsets,
        nkb=int(sum(sizes)),
    )


@partial(jax.jit, static_argnames=("nl",))
def _assemble_qgm(coefficients, ylm, radial, beta_of, nl):
    """``sum_LM (-i)^L ap(LM,i,j) Y_LM(G) Q^L_{n_i n_j}(|G|)``.

    Accumulated one ``L`` at a time. The alternative -- one contraction over all
    ``LM`` at once -- needs the radial table broadcast to ``(nh, nh, nlm, ngm)``,
    which is the same arithmetic through several times the memory.
    """
    # (nbeta, nbeta, nl, ngm) -> (nh, nh, nl, ngm), one row per projector channel
    channel_radial = radial[beta_of[:, None], beta_of[None, :]]

    total = None
    for l in range(nl):
        block = slice(l * l, (l + 1) ** 2)
        angular = jnp.einsum("mij,gm->ijg", coefficients[block], ylm[:, block])
        # (-i)^L: real for even L, imaginary for odd, which is qvan2's sig/ind.
        phase = (-1j) ** l
        term = phase * angular * channel_radial[:, :, l, :]
        total = term if total is None else total + term
    return total


@jax.jit
def _atom_phases(gcart, positions):
    """``e^{-i G . tau_a}`` for every atom, ``(nat, ngm)``."""
    return jnp.exp(-1j * (positions @ gcart.T))
