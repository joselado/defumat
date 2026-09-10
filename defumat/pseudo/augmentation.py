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

with the coupling coefficients from :mod:`defumat.pseudo.coupling` and the
radial transforms

    Q^L_nm(q) = 4 pi / Omega int_0^{r_kkbeta} dr j_L(q r) [r^2 Q^L_nm(r)]

evaluated directly at each ``|G|``, as everything else in
:mod:`defumat.pseudo.formfactors` is, rather than interpolated from QE's
``dq = 0.01`` table -- same reason: it keeps the augmentation charge a
differentiable function of the cell.

**The integration range is ``kkbeta``, not the 10-bohr mesh** the local potential
and the atomic charge use. Q is identically zero outside the augmentation
sphere, so extending the range only adds the tabulated noise beyond it; QE
integrates to ``kkbeta`` and so does this. The check that it is right is that
``Omega * Q_ij(G=0)`` reproduces the file's own ``PP_Q`` values.
"""

from __future__ import annotations

import hashlib
import os
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from defumat.basis.gvectors import GVectors, modulus
from defumat.pseudo.coupling import harmonic_products
from defumat.pseudo.harmonics import real_spherical_harmonics
from defumat.pseudo.projectors import projector_channels
from defumat.pseudo.radial import simpson_weights, spherical_bessel
from defumat.pseudo.upf import Pseudopotential
from defumat.system.cell import Cell
from defumat.system.structure import Structure
from defumat.units import FPI

__all__ = ["AugmentationCharge", "TabulatedAugmentation", "augmentation_dipole",
           "build_augmentation", "radial_augmentation_transforms"]


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


def augmentation_dipole(pseudo: Pseudopotential) -> np.ndarray:
    """``dpqq``: the augmentation charge's dipole, ``(3, nh, nh)`` in bohr.

    ``PW/src/compute_qdipol.f90``.

        dpqq^a_ij = int Q_ij(r) r_a dr

    about the atom's own centre, in cartesian components. It is zero for a
    norm-conserving species and is what makes the *position* operator of an
    ultrasoft calculation differ from ``r``: the charge an ultrasoft state
    carries is not all in ``|psi|^2``, and the part inside the augmentation
    sphere has a dipole of its own.

    **Only ``L = 1`` contributes**, which is the whole reason this is a
    thirty-line function rather than a transform: ``r_a`` is an ``l = 1``
    harmonic, so the angular integral kills every multipole of ``Q_ij`` but one,
    and what is left is a single radial moment ``int r^3 Q^{L=1}_{nm}(r) dr``
    (``qfuncl`` already carrying the ``r^2``) times the harmonic product
    ``ap[LM, lm_i, lm_j]`` the augmentation charge is built from anyway.

    **It is not a derivative of** :func:`defumat.topology.augmentation.
    augmentation_at_q`, though it is one mathematically -- ``dpqq^a =
    i d/dq_a [int Q(r) e^{-i q.r} dr]`` at ``q = 0``. That form factor is
    written as a radial function of ``|q|`` times a harmonic of ``q/|q|``, and
    at the origin those two are ``0`` and ``infinity``: the product is smooth and
    the factorisation is not, so a ``jvp`` there is ``NaN``. The closed form
    below is the same number without the coordinate singularity, which is the
    trap :func:`defumat.basis.gvectors.modulus` documents, met once more.
    """
    channels = projector_channels(pseudo)
    nh = len(channels)
    dipole = np.zeros((3, nh, nh))
    if not pseudo.is_ultrasoft or nh == 0:
        return dipole
    augmentation = pseudo.augmentation
    if augmentation is None or augmentation.qfuncl is None:
        raise NotImplementedError(
            f"{pseudo.element}: the augmentation charge is stored in the pre-2.0 "
            "qfcoef form, which is not implemented"
        )
    if augmentation.qfuncl.shape[2] <= 1:
        return dipole  # no L = 1 channel: every dipole vanishes by parity

    # ``int r^3 Q^{L=1}_{nm}(r) dr``, per pair of radial projectors. The triangle
    # rule and parity decide which pairs have an L = 1 channel at all.
    kkbeta = pseudo.kkbeta
    radius = np.asarray(pseudo.r[:kkbeta])
    weights = np.asarray(simpson_weights(jnp.asarray(pseudo.rab[:kkbeta])))
    angular = [projector.l for projector in pseudo.projectors]
    moment = np.zeros((pseudo.nbeta, pseudo.nbeta))
    for nb in range(pseudo.nbeta):
        for mb in range(pseudo.nbeta):
            l_n, l_m = angular[nb], angular[mb]
            if abs(l_n - l_m) <= 1 <= l_n + l_m and (1 + l_n + l_m) % 2 == 0:
                function = np.asarray(augmentation.qfuncl[nb, mb, 1, :kkbeta])
                moment[nb, mb] = float(np.sum(weights * radius * function))

    ap = harmonic_products(pseudo.lmax)
    beta_of = np.array([nb for nb, _, _ in channels])
    lm_of = np.array([lm for _, _, lm in channels])
    # QE's ``lp`` in 1-based ``lm``: 3 for x, 4 for y, 2 for z -- one less here,
    # and ``fact`` changes sign for z because ``Y_10`` is the one with no
    # ``(x + iy)`` in it. ``compute_qdipol``'s ``fact = -sqrt(4 pi / 3)``.
    factor = -np.sqrt(FPI / 3.0)
    for axis, (lp, sign) in enumerate(((2, 1.0), (3, 1.0), (1, -1.0))):
        coefficients = ap[lp][np.ix_(lm_of, lm_of)]
        dipole[axis] = sign * factor * coefficients * moment[np.ix_(beta_of, beta_of)]
    return dipole


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
    """``4 pi / Omega int dr j_l(q r) [r^2 Q^l(r)]`` for a stack of ``Q``.

    **This is where a stress evaluation's memory goes.** The intermediate is
    ``(ngm, kkbeta)`` -- 36257 by ~1100 on eight-atom ultrasoft silicon, so 300
    MB, with the temporaries inside ``spherical_bessel`` on top and one of them
    per ``L``. Evaluated forward they are transient; differentiated in
    **reverse** mode, as the stress differentiates them (P11), they are all live
    at once, and the peak working set goes to 11 GB against the SCF's 0.9.
    ``jax.checkpoint`` here was tried and measured to be worth nothing -- the
    intermediates are spread across the radial kernels rather than concentrated
    in this one -- so what is recorded is the measurement and the fix that has
    not been written: a ``custom_jvp`` carrying ``dF/d|G|`` in closed form, so
    that the transform tapes a vector of length ``ngm`` instead of a matrix.
    See `PERFORMANCE.md`.
    """
    argument = q[:, None] * r[None, :]
    bessel = spherical_bessel(l, argument)  # (nq, mesh)
    return prefactor * jnp.einsum("fm,qm,m->fq", functions, bessel, weights)



#: QE's interpolation step in ``|q|`` (``upflib/qrad_mod.f90:22``). ``q`` is in
#: sqrt(Ry), since ``q^2`` is an energy in Rydberg atomic units.
AUG_DQ = 0.01

#: How much ``Q_ij(G)`` may occupy before the run stops storing it and rebuilds
#: it from a table instead. Below this it is built once and kept, which is what
#: every validated number in this project was measured with and is the faster
#: of the two on a small cell. ``DEFUMAT_AUG_MAX_BYTES`` overrides it; ``off``
#: means never tabulate.
AUG_MAX_BYTES = 2 * 1024**3

#: QE's ``cell_factor``: how far past ``sqrt(ecutrho)`` the table reaches, so
#: that a strained cell -- whose G vectors move while the G *set* does not --
#: still lands on it. QE's own default for a variable cell
#: (``PW/src/input.f90:1284``), and the table is kilobytes, so the margin is
#: free.
AUG_CELL_FACTOR = 2.0


def _aug_max_bytes() -> int:
    value = os.environ.get("DEFUMAT_AUG_MAX_BYTES")
    if value is None:
        return AUG_MAX_BYTES
    if value.strip().lower() in ("off", "none", "inf"):
        return 1 << 62
    return int(float(value))


def _aug_chunk(nh_max: int, ngm: int) -> int:
    """How many G vectors ``Q_ij(G)`` is rebuilt for at a time.

    The intermediate this decides is ``(nh, nh, chunk)`` complex, so the
    default is chosen to put *that* near 256 MB rather than fixed at a count:
    one chunk is eleven times more memory for a fully-relativistic nickel
    dataset (``nh = 34``) than for silicon's (``nh = 8``).
    ``DEFUMAT_AUG_CHUNK`` overrides it.

    Like every other batching dial here it is a loop bound over an exact sum,
    and must not be visible in a result beyond round-off.
    """
    value = os.environ.get("DEFUMAT_AUG_CHUNK")
    if value is not None:
        return max(1, min(int(value), ngm))
    target = 256 * 1024**2 // max(1, nh_max * nh_max * 16)
    return int(min(1 << max(10, int(np.floor(np.log2(max(target, 1024))))), ngm))


def _qrad_table(pseudo: Pseudopotential, qmax: float, omega, nl: int) -> jnp.ndarray:
    """``tab_qrad``: ``Q^L_nm`` on the ``(i - 1) dq`` grid. ``init_tab_qrad``.

    ``nqx = INT(qmax/dq + 4)`` is QE's own sizing (``qrad_mod.f90:86``), and
    the four extra points are the room the forward-biased stencil in
    :func:`_interpolate_qrad` needs at the top of the range rather than a
    safety margin.
    """
    nqx = int(qmax / AUG_DQ + 4)
    knots = jnp.arange(nqx) * AUG_DQ
    return radial_augmentation_transforms(pseudo, knots, omega, nl)


def _interpolate_qrad(table: jnp.ndarray, qmod: jnp.ndarray) -> jnp.ndarray:
    """``qvan2``'s four-point Lagrange, at every ``|G|`` at once.

    ``upflib/qvan2.f90:143-158``, transcribed rather than replaced by a spline:
    every committed ``pw.x`` reference this code is checked against was
    produced with *this* stencil, so a smoother one would move the comparison
    by its own interpolation error and leave the disagreement unreadable.

    The stencil is forward-biased -- the four knots are ``i0 .. i0+3`` with the
    evaluation point in the **first** of the three intervals, not the middle
    one -- which is why ``init_tab_qrad`` adds four points and not two.

    **Off the top of the table is NaN, never a clamped extrapolation.** A
    gather in JAX clamps its indices silently, and a clamped ``Q^L(q)`` is
    smooth, plausible and wrong: it would surface as a wrong stress under a
    strain large enough to push ``|G|`` past ``qmax``, with nothing anywhere to
    say so.
    """
    nqx = table.shape[-1]
    qm = qmod / AUG_DQ
    i0 = jnp.floor(qm).astype(jnp.int32)
    px = qm - i0
    ux, vx, wx = 1.0 - px, 2.0 - px, 3.0 - px
    uvx = ux * vx / 6.0
    pwx = px * wx * 0.5

    total = None
    for step, weight in enumerate((uvx * wx, pwx * vx, -pwx * ux, px * uvx)):
        # Summed one stencil point at a time: gathering all four first would
        # hold four (nbeta, nbeta, nl, nq) arrays at once for no reason.
        term = table[..., jnp.clip(i0 + step, 0, nqx - 1)] * weight
        total = term if total is None else total + term
    return jnp.where(i0 + 3 < nqx, total, jnp.nan)


class TabulatedAugmentation(AugmentationCharge):
    """``Q_ij(G)`` rebuilt from a table, for a cell too large to store it on.

    The base class's materialised ``(nh, nh, ngm)`` array is the largest object
    in the whole calculation on a vacuum-padded slab: 65 GB for one
    fully-relativistic nickel dataset on a 45-atom NiBr2 cell at
    ``ecutrho = 360``, against the 8.4 MB QE spends on the same physics. QE
    never materialises it. ``init_tab_qrad`` tabulates the *radial* transforms
    on a grid in ``|q|`` and ``qvan2`` interpolates per ``(ij)`` pair inside
    ``addusdens`` and ``newd``, every iteration.

    This does the same, with QE's loop over ``(ij)`` pairs replaced by a scan
    over blocks of G -- the axis a plane-wave code in JAX can afford to walk.

    **The trade is smaller than it looks, and it was measured rather than
    assumed.** The rebuild happens twice an iteration where the stored array is
    built once, and the two contractions do cost what that implies: on
    ``benchmarks/si8-us-1k.in``, single core, ``charge`` goes from 0.054 s to
    0.174 s and ``integrals`` from 0.033 s to 0.153 s, both about 4.5 times.
    But they are a small part of an SCF iteration, and the table is *cheaper*
    to build than the stored array is -- the radial transform runs on 2533
    knots instead of 36257 G vectors -- so the whole run comes out level: 5.56
    s against 5.42 s, the same six iterations, and a total energy 1.6e-9 Ry
    apart on -91.01 Ry. What changes by 38 times is only the memory.

    It is still not the default, because "level on one cell" is not "never
    slower", and because the stored path is what every validated number in this
    project was measured with.

    ``gcart`` and ``phases`` are stored **padded** to a whole number of chunks,
    with ``mask`` zero on the padding. A padded G is the origin, where
    ``Q_ij(G)`` is emphatically not zero, so the padding is killed in the
    contraction rather than left to vanish on its own.
    """

    tables: tuple  # per species, (nbeta, nbeta, nl, nqx) -- QE's tab_qrad
    coefficients: tuple  # per species, (nlm, nh, nh) -- ap, restricted
    beta_of: tuple  # per species, (nh,) -- which radial projector a channel is
    gcart: jnp.ndarray  # (npad, 3) cartesian G, padded; carries the cell
    mask: jnp.ndarray  # (npad,) 1.0 on a real G, 0.0 on the padding
    ngm: int = eqx.field(static=True)
    chunk: int = eqx.field(static=True)
    lmax2: int = eqx.field(static=True)  # the ylm order, 2 lmax
    nl_species: tuple = eqx.field(static=True)

    @property
    def ntyp(self) -> int:
        return len(self.tables)

    def _builder(self, t: int):
        """``Q_ij(G)`` for species ``t``, as a function of a block of G."""
        table, coefficients = self.tables[t], self.coefficients[t]
        beta_of, nl = self.beta_of[t], self.nl_species[t]

        def build(gcart_chunk):
            ylm = real_spherical_harmonics(gcart_chunk, self.lmax2)
            radial = _interpolate_qrad(table, modulus(gcart_chunk))
            return _assemble_qgm(coefficients, ylm, radial, beta_of, nl)

        return build

    def charge(self, becsum: tuple) -> jnp.ndarray:
        total = None
        for t, atoms in enumerate(self.species_atoms):
            if self.tables[t] is None or not atoms:
                continue
            contribution = _tabulated_charge(
                self._builder(t), self.gcart, self.mask,
                self.phases[jnp.asarray(atoms)],
                becsum[t].astype(self.phases.dtype), self.chunk, self.ngm,
            )
            total = contribution if total is None else total + contribution
        if total is None:
            return jnp.zeros(self.ngm, dtype=self.phases.dtype)
        return total

    def integrals(self, potential_g: jnp.ndarray) -> tuple:
        padded = jnp.pad(potential_g, (0, self.mask.shape[0] - self.ngm))
        result = []
        for t, atoms in enumerate(self.species_atoms):
            nh = 0 if self.tables[t] is None else self.beta_of[t].shape[0]
            if self.tables[t] is None or not atoms:
                result.append(jnp.zeros((len(atoms), nh, nh)))
                continue
            result.append(
                _tabulated_integrals(
                    self._builder(t), self.gcart, self.mask, padded,
                    self.phases[jnp.asarray(atoms)], self.volume, self.chunk, nh,
                )
            )
        return tuple(result)

    def at_positions(self, positions: jnp.ndarray, gcart: jnp.ndarray):
        """As the base class, except the G set is padded with the phases."""
        padded = jnp.pad(gcart, ((0, self.mask.shape[0] - self.ngm), (0, 0)))
        phases = _atom_phases(padded, positions).astype(self.phases.dtype)
        return eqx.tree_at(lambda a: (a.phases, a.gcart), self, (phases, padded))


def _tabulated_charge(build, gcart, mask, phases, becsum, chunk, ngm):
    """``rho_aug(G)`` for one species, scanning over blocks of G."""
    nchunks = mask.shape[0] // chunk
    nat = phases.shape[0]

    def body(carry, index):
        start = index * chunk
        gcart_chunk = jax.lax.dynamic_slice(gcart, (start, 0), (chunk, 3))
        phase_chunk = jax.lax.dynamic_slice(phases, (0, start), (nat, chunk))
        mask_chunk = jax.lax.dynamic_slice(mask, (start,), (chunk,))
        weighted = jnp.einsum("aij,ac->ijc", becsum, phase_chunk)
        block = jnp.einsum("ijc,ijc->c", build(gcart_chunk), weighted)
        return carry, block * mask_chunk

    _, blocks = jax.lax.scan(body, None, jnp.arange(nchunks))
    return blocks.reshape(-1)[:ngm]


def _tabulated_integrals(build, gcart, mask, potential_g, phases, volume, chunk, nh):
    """``int V(r) Q_ij^a(r) dr`` for one species, scanning over blocks of G.

    A reduction over G rather than a map along it, so the accumulator is the
    scan's *carry*: ``(nat, nh, nh)`` whatever the chunk is.
    """
    nchunks = mask.shape[0] // chunk
    nat = phases.shape[0]

    def body(carry, index):
        start = index * chunk
        gcart_chunk = jax.lax.dynamic_slice(gcart, (start, 0), (chunk, 3))
        phase_chunk = jax.lax.dynamic_slice(phases, (0, start), (nat, chunk))
        mask_chunk = jax.lax.dynamic_slice(mask, (start,), (chunk,))
        potential_chunk = jax.lax.dynamic_slice(potential_g, (start,), (chunk,))
        shifted = potential_chunk[None, :] * jnp.conj(phase_chunk) * mask_chunk
        block = jnp.einsum("ijc,ac->aij", jnp.conj(build(gcart_chunk)), shifted)
        return carry + jnp.real(block), None

    total, _ = jax.lax.scan(
        body, jnp.zeros((nat, nh, nh)), jnp.arange(nchunks)
    )
    return volume * total


def _dataset_key(pseudo: Pseudopotential, nl_species: int) -> tuple:
    """A fingerprint of everything ``Q_ij(G)`` is built from, bar the G set.

    Two *species* that name the same UPF file are the same dataset, and
    ``Q_ij(G)`` depends on the dataset and on the G set -- never on which label
    an atom carries. Writing one species per magnetic site is the standard way
    to write a noncollinear input, because ``angle1``/``angle2`` are per
    species, so a fifteen-site helix arrives here as fifteen identical
    datasets; without this key each of them builds and holds its own
    ``(nh, nh, ngm)`` array. On a 45-atom NiBr2 slab that is 65 GB per Ni
    species.

    The key is the *content* rather than the file name: the path is a hint that
    can be absent (a hand-built :class:`Pseudopotential` has none) and can lie
    (two objects read from one path, one of them since modified). Hashing the
    radial data is a few milliseconds against an array measured in tens of GB.
    """
    augmentation = pseudo.augmentation
    kkbeta = pseudo.kkbeta

    def digest(array) -> bytes:
        return hashlib.blake2b(
            np.ascontiguousarray(np.asarray(array)).tobytes(), digest_size=16
        ).digest()

    qfuncl = (
        augmentation.qfuncl[:, :, :nl_species, :kkbeta]
        if augmentation is not None and augmentation.qfuncl is not None
        else np.zeros(0)
    )
    return (
        nl_species,
        kkbeta,
        tuple(projector_channels(pseudo)),
        tuple(projector.l for projector in pseudo.projectors),
        digest(pseudo.r[:kkbeta]),
        digest(pseudo.rab[:kkbeta]),
        digest(qfuncl),
    )



def _nl_of(pseudo: Pseudopotential, nl: int) -> int:
    """How many multipoles this dataset's augmentation charge actually has."""
    if pseudo.augmentation is None:
        return nl
    return min(nl, pseudo.augmentation.nqlc)


def _build_tabulated_augmentation(
    pseudos, structure, cell, gvectors, ap, lmax, nl, nh_max, cell_factor
):
    """:class:`TabulatedAugmentation` -- the branch for a cell too large to store.

    ``qmax`` is QE's, from ``memory_report.f90:173``: the table reaches
    ``cell_factor`` times ``sqrt(ecutrho)`` because a strained cell moves its G
    vectors while the G *set* stays as it was, and an evaluation past the end
    of the table is NaN rather than a clamp.
    """
    volume = cell.volume
    qmax = float(np.sqrt(gvectors.ecut)) * cell_factor
    chunk = _aug_chunk(nh_max, gvectors.ngm)
    npad = -(-gvectors.ngm // chunk) * chunk

    gcart = gvectors.cartesian(cell)
    gcart = jnp.pad(gcart, ((0, npad - gvectors.ngm), (0, 0)))
    mask = jnp.concatenate([
        jnp.ones(gvectors.ngm, dtype=cell.precision.real),
        jnp.zeros(npad - gvectors.ngm, dtype=cell.precision.real),
    ])

    tables, coefficients, beta_of, nl_species, qq = [], [], [], [], []
    built: dict = {}
    for pseudo in pseudos:
        channels = projector_channels(pseudo)
        if not pseudo.is_ultrasoft or not channels:
            tables.append(None)
            coefficients.append(None)
            beta_of.append(jnp.zeros(0, dtype=jnp.int32))
            nl_species.append(0)
            qq.append(jnp.zeros((0, 0)))
            continue

        nl_t = _nl_of(pseudo, nl)
        key = _dataset_key(pseudo, nl_t)
        if key not in built:
            lm_of = np.array([lm for _, _, lm in channels])
            built[key] = (
                _qrad_table(pseudo, qmax, volume, nl_t),
                jnp.asarray(ap[:, lm_of[:, None], lm_of[None, :]]),
                jnp.asarray(np.array([nb for nb, _, _ in channels])),
            )
        table, coefficient, betas = built[key]
        tables.append(table)
        coefficients.append(coefficient)
        beta_of.append(betas)
        nl_species.append(nl_t)

        # ``Omega * Q_ij(G = 0)``. Taken through the table and the stencil
        # rather than from the radial transform directly, so that the file's
        # own ``PP_Q`` check reaches the interpolation as well: at ``q = 0``
        # the four weights are (1, 0, 0, 0), so this reads the first knot.
        at_origin = _assemble_qgm(
            coefficient,
            real_spherical_harmonics(gcart[:1], 2 * lmax),
            _interpolate_qrad(table, modulus(gcart[:1])),
            betas, nl_t,
        )
        qq.append(volume * jnp.real(at_origin[:, :, 0]))

    phases = _atom_phases(gcart, structure.positions).astype(cell.precision.complex)
    types = np.asarray(structure.types)
    species_atoms = tuple(
        tuple(int(a) for a in np.flatnonzero(types == t)) for t in range(structure.ntyp)
    )
    sizes = [len(projector_channels(pseudos[t])) for t in types]

    return TabulatedAugmentation(
        qgm=(),
        qq=tuple(qq),
        phases=phases,
        volume=jnp.asarray(volume),
        species_atoms=species_atoms,
        channel_offsets=tuple(int(o) for o in np.cumsum([0] + sizes)[:-1]),
        nkb=int(sum(sizes)),
        tables=tuple(tables),
        coefficients=tuple(coefficients),
        beta_of=tuple(beta_of),
        gcart=gcart,
        mask=mask,
        ngm=int(gvectors.ngm),
        chunk=int(chunk),
        lmax2=int(2 * lmax),
        nl_species=tuple(nl_species),
    )


def build_augmentation(
    pseudos: tuple[Pseudopotential, ...],
    structure: Structure,
    cell: Cell,
    gvectors: GVectors,
    max_bytes: int | None = None,
    cell_factor: float = AUG_CELL_FACTOR,
) -> AugmentationCharge | None:
    """Assemble ``Q_ij(G)`` for every ultrasoft species. ``None`` if there are none.

    ``gvectors`` must be the **dense** set.

    **Two storage schemes, and the size of the cell picks one.** Below
    ``max_bytes`` (:data:`AUG_MAX_BYTES`, or ``DEFUMAT_AUG_MAX_BYTES``) the
    array is built on the whole G sphere and kept for the run, which is what
    every validated number here was measured with. Above it,
    :class:`TabulatedAugmentation` keeps QE's radial table instead and rebuilds
    ``Q_ij(G)`` a block of G at a time -- 8.4 MB where the stored array is 65
    GB on a 45-atom NiBr2 slab, at the cost of rebuilding it twice an
    iteration. The two agree to interpolation error, which
    ``tests/regression/test_uspp.py`` measures rather than assumes.
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

    # Which scheme, decided on what the stored array would cost -- per distinct
    # *dataset*, since that is what is built below.
    stored_bytes, nh_max = 0, 0
    seen = set()
    for pseudo in pseudos:
        channels = projector_channels(pseudo)
        if not pseudo.is_ultrasoft or not channels:
            continue
        key = _dataset_key(pseudo, _nl_of(pseudo, nl))
        if key in seen:
            continue
        seen.add(key)
        stored_bytes += len(channels) ** 2 * gvectors.ngm * 16
        nh_max = max(nh_max, len(channels))
    if stored_bytes > (_aug_max_bytes() if max_bytes is None else max_bytes):
        return _build_tabulated_augmentation(
            pseudos, structure, cell, gvectors, ap, lmax, nl, nh_max, cell_factor
        )

    gcart = gvectors.cartesian(cell)
    gmod = modulus(gcart)  # guarded at G = 0; see gvectors.modulus
    ylm = real_spherical_harmonics(gcart, 2 * lmax)  # (ngm, (2lmax+1)^2)
    volume = cell.volume

    qgm, qq = [], []
    built: dict = {}  # dataset fingerprint -> (Q_ij(G), qq); see _dataset_key
    for pseudo in pseudos:
        channels = projector_channels(pseudo)
        if not pseudo.is_ultrasoft or not channels:
            qgm.append(jnp.zeros((0, 0, gvectors.ngm), dtype=cell.precision.complex))
            qq.append(jnp.zeros((0, 0)))
            continue

        nl_species = min(nl, pseudo.augmentation.nqlc) if pseudo.augmentation else nl

        key = _dataset_key(pseudo, nl_species)
        if key in built:
            # The same dataset under a second species label. Q_ij(G) is a
            # property of the *file* and of the G set, not of the label, and
            # both are unchanged -- so the two species share one array.
            shared = built[key]
            qgm.append(shared[0])
            qq.append(shared[1])
            continue

        radial = radial_augmentation_transforms(pseudo, gmod, volume, nl_species)

        beta_of = np.array([nb for nb, _, _ in channels])
        lm_of = np.array([lm for _, _, lm in channels])
        # ap restricted to the (lm_i, lm_j) actually present, as qvan2's
        # lpl/lpx lists restrict the sum.
        coefficients = jnp.asarray(ap[:, lm_of[:, None], lm_of[None, :]])

        values = _assemble_qgm(
            coefficients, ylm, radial, jnp.asarray(beta_of), nl_species
        )
        entry = (values.astype(cell.precision.complex), volume * jnp.real(values[:, :, 0]))
        built[key] = entry
        qgm.append(entry[0])
        qq.append(entry[1])

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
    total = None
    for l in range(nl):
        block = slice(l * l, (l + 1) ** 2)
        angular = jnp.einsum("mij,gm->ijg", coefficients[block], ylm[:, block])
        # (-i)^L: real for even L, imaginary for odd, which is qvan2's sig/ind.
        phase = (-1j) ** l
        # The (nbeta, nbeta) -> (nh, nh) expansion is done **inside** the L loop
        # and never for every L at once. One row per projector channel is a
        # factor (nh/nbeta)^2 more rows -- 11.6 for a fully-relativistic Ni
        # dataset, nh = 34 against nbeta = 10 -- and the whole table at once is
        # a further nl on top. Same arithmetic, one L's worth of memory.
        term = phase * angular * radial[beta_of[:, None], beta_of[None, :], l]
        total = term if total is None else total + term
    return total


@jax.jit
def _atom_phases(gcart, positions):
    """``e^{-i G . tau_a}`` for every atom, ``(nat, ngm)``."""
    return jnp.exp(-1j * (positions @ gcart.T))
