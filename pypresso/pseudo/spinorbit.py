"""Spin-orbit coupling: the ``j``-resolved projectors as a spinor operator.

A fully-relativistic pseudopotential is generated from the Dirac equation, so
each channel with ``l > 0`` comes in two, ``j = l - 1/2`` and ``j = l + 1/2``,
with different radial projectors and different ``D_ij``. The difference between
them *is* the spin-orbit coupling: average the pair and what is left is the
scalar-relativistic potential every collinear calculation uses.

The nonlocal potential is naturally written in the ``|l, j, m_j>`` basis, where
it is diagonal, but the plane-wave code works with real spherical harmonics
times a spin index. The bridge is the matrix QE calls ``fcoef``,

    F^{s1 s2}_{ih kh} = sum_m  U_{m(ih) mi}  C^{s1}_{l j m}
                               conj(U_{m(kh) mk}) C^{s2}_{l j m}

with ``U`` (``rot_ylm``) the unitary taking real spherical harmonics to complex
ones and ``C`` (``spinor``) the Clebsch-Gordan coefficient of the spin-angle
function. It is a projector onto one ``(l, j)`` shell, resolved into the two
spin components, and every spin-orbit quantity in the code is that same matrix
sandwiching a scalar one:

    D^{s1 s2}_ij = D^(0)_ij F^{s1 s2}_ij                    (``dvan_so``)
    q^{s1 s2}    = sum_s F^{s1 s}  q  F^{s s2}              (``transform_qq_so``)
    D^{s1 s2}    = D^{s1 s2}_so + sum_s F^{s1 s} D F^{s s2} (``newd_so``)

Transcribed from ``upflib/init_us_1.f90``, ``upflib/spinor.f90``,
``upflib/sph_ind.f90`` and ``upflib/upf_spinorb.f90``.

**The trap.** ``init_us_1`` builds ``fcoef`` for every pair with matching
``(l, j)``, uses it to build ``dvan_so``, and *then* zeroes the entries whose
two radial projectors differ. Everything downstream -- ``transform_qq_so``,
``newd_so``, ``add_becsum_so`` -- consumes the **zeroed** array and depends on
it to kill the cross-radial terms, having no test of its own. One array used
everywhere gives a correct ``dvan_so`` and a silently wrong ``qq_so``,
``deeq_nc`` and ``becsum``, so the two are kept apart here and named for what
they are.

This is all host-side setup on ``(nh, nh)`` matrices -- a few tens of channels
per species, once per run -- so it is NumPy, not JAX.
"""

from __future__ import annotations

import numpy as np

from pypresso.pseudo.projectors import projector_channels
from pypresso.pseudo.upf import Pseudopotential

__all__ = [
    "rot_ylm",
    "spinor",
    "sph_ind",
    "SpinOrbitCoupling",
    "build_spin_orbit",
    "pauli_blocks",
    "becsum_transform",
]

#: ``lmaxx`` in ``upflib/upf_params.f90``. ``rot_ylm`` is built once at this
#: fixed ``l`` and indexed relative to its centre row, which is why one matrix
#: serves every angular momentum.
LMAXX = 3

_SQRT2 = np.sqrt(2.0)


def rot_ylm(lmaxx: int = LMAXX) -> np.ndarray:
    """The unitary taking QE's real spherical harmonics to complex ones.

    ``rot_ylm[lmaxx + m, i]`` is the coefficient of the ``i``-th real harmonic
    of a shell in the complex ``Y_{l m}``, with ``i`` counting QE's ordering
    (``m = 0``, then the cosine/sine pair of each ``|m|``; see
    :mod:`pypresso.pseudo.harmonics`). Rows are indexed by ``m`` relative to the
    centre, which is what lets a single matrix built at ``l = lmaxx`` serve every
    ``l``: the relation between the two conventions has the same form in each
    shell.

    ``init_us_1.f90`` builds it inline; it is a function here because it is
    pure, small, and worth testing on its own (it must be unitary).
    """
    n = 2 * lmaxx + 1
    u = np.zeros((n, n), dtype=complex)
    u[lmaxx, 0] = 1.0  # m = 0 is already real
    for m in range(1, lmaxx + 1):
        cosine, sine = 2 * m - 1, 2 * m  # columns of the (cos, sin) pair
        sign = (-1.0) ** m
        u[lmaxx - m, cosine] = sign / _SQRT2
        u[lmaxx - m, sine] = -1j * sign / _SQRT2
        u[lmaxx + m, cosine] = 1.0 / _SQRT2
        u[lmaxx + m, sine] = 1j / _SQRT2
    return u


def spinor(l: int, j: float, m: int, spin: int) -> float:
    """The Clebsch-Gordan coefficient of a spin-angle function (``spinor.f90``).

    The spin-angle function ``Omega_{l j m_j}`` with ``m_j = m + 1/2`` is a
    two-component object whose upper component multiplies ``Y_{l m}`` and whose
    lower multiplies ``Y_{l m+1}``; this returns those two coefficients for
    ``spin = 0`` and ``spin = 1``.

    ``m`` runs from ``-l-1`` to ``l``, one value more than ``2l+1``: the two
    ``j`` shells of an ``l`` hold ``2l`` and ``2l+2`` states between them, and
    the extra value is where the shorter one has a zero.
    """
    if spin not in (0, 1):
        raise ValueError(f"spin must be 0 or 1, got {spin}")
    if m < -l - 1 or m > l:
        raise ValueError(f"m = {m} out of range for l = {l}")

    denominator = 1.0 / (2.0 * l + 1.0)
    if abs(j - l - 0.5) < 1.0e-8:
        return np.sqrt((l + m + 1.0) * denominator) if spin == 0 else np.sqrt((l - m) * denominator)
    if abs(j - l + 0.5) < 1.0e-8:
        if m < -l + 1:
            return 0.0
        if spin == 0:
            return np.sqrt((l - m + 1.0) * denominator)
        return -np.sqrt((l + m) * denominator)
    raise ValueError(f"j = {j} is not compatible with l = {l}")


def sph_ind(l: int, j: float, m: int, spin: int) -> int:
    """Which ``Y_{l m'}`` the given component of a spin-angle function uses.

    ``sph_ind.f90``. The companion to :func:`spinor`: that gives the
    coefficient, this gives the magnetic quantum number it multiplies. Out-of-
    range results are folded to ``m' = 0``, where :func:`spinor` has returned
    zero anyway.
    """
    if spin not in (0, 1):
        raise ValueError(f"spin must be 0 or 1, got {spin}")
    if m < -l - 1 or m > l:
        raise ValueError(f"m = {m} out of range for l = {l}")

    if abs(j - l - 0.5) < 1.0e-8:
        index = m if spin == 0 else m + 1
    elif abs(j - l + 0.5) < 1.0e-8:
        if m < -l + 1:
            index = 0
        else:
            index = m - 1 if spin == 0 else m
    else:
        raise ValueError(f"j = {j} is not compatible with l = {l}")
    return 0 if index < -l or index > l else index


def _channel_table(pseudo: Pseudopotential):
    """``(indv, nhtol, nhtolm, nhtoj)`` for one species, as QE's ``init_us_1``.

    Each radial projector contributes ``2l+1`` channels, so these four arrays
    say, for every channel, which radial function it came from, its ``l``, its
    column among the spherical harmonics, and its ``j``.
    """
    channels = projector_channels(pseudo)
    indv = np.array([nb for nb, _, _ in channels], dtype=int)
    nhtol = np.array([l for _, l, _ in channels], dtype=int)
    nhtolm = np.array([lm for _, _, lm in channels], dtype=int)
    nhtoj = np.array(
        [
            -1.0 if pseudo.projectors[nb].j is None else pseudo.projectors[nb].j
            for nb, _, _ in channels
        ]
    )
    return indv, nhtol, nhtolm, nhtoj


def _fcoef(pseudo: Pseudopotential) -> np.ndarray:
    """``fcoef`` before the cross-radial entries are zeroed, ``(nh, nh, 2, 2)``.

    The sum over ``m`` in ``init_us_1`` runs over ``-l-1 .. l``, and both factors
    pick their spherical harmonic through :func:`sph_ind` -- so a term survives
    only when the two channels' real harmonics are the ones the same ``m_j``
    couples. That is what makes the result a projector onto a single ``(l, j)``
    shell rather than a general matrix.
    """
    indv, nhtol, nhtolm, nhtoj = _channel_table(pseudo)
    nh = len(indv)
    u = rot_ylm()
    coefficients = np.zeros((nh, nh, 2, 2), dtype=complex)

    for ih in range(nh):
        li, ji = int(nhtol[ih]), float(nhtoj[ih])
        mi = int(nhtolm[ih] - li * li)  # 0-based column within the shell
        for kh in range(nh):
            lk, jk = int(nhtol[kh]), float(nhtoj[kh])
            if li != lk or abs(ji - jk) >= 1.0e-7:
                continue
            mk = int(nhtolm[kh] - lk * lk)
            for is1 in range(2):
                for is2 in range(2):
                    total = 0.0 + 0.0j
                    for m in range(-li - 1, li + 1):
                        m0 = sph_ind(li, ji, m, is1) + LMAXX
                        m1 = sph_ind(lk, jk, m, is2) + LMAXX
                        total += (
                            u[m0, mi]
                            * spinor(li, ji, m, is1)
                            * np.conj(u[m1, mk])
                            * spinor(lk, jk, m, is2)
                        )
                    coefficients[ih, kh, is1, is2] = total
    return coefficients


def pauli_blocks(components: np.ndarray) -> np.ndarray:
    """``(n, mx, my, mz)`` -> the 2x2 spin matrix ``n I + m . sigma``.

    ``newd_nc_acc``'s recombination, and the same one that turns the four
    components of a noncollinear potential into what multiplies a spinor:

        [[ n + mz     ,  mx - i my ],
         [ mx + i my  ,  n - mz    ]]

    Args:
        components: ``(nspin_mag, ...)`` with ``nspin_mag`` 1 or 4. One
            component is a scalar, which is the nonmagnetic case and gives a
            multiple of the identity with no off-diagonal terms at all.

    Returns ``(2, 2, ...)`` complex.
    """
    components = np.asarray(components)
    n = components[0]
    zero = np.zeros_like(n, dtype=complex)
    if components.shape[0] == 1:
        return np.array([[n + zero, zero], [zero, n + zero]])
    mx, my, mz = components[1], components[2], components[3]
    return np.array([
        [n + mz, mx - 1j * my],
        [mx + 1j * my, n - mz],
    ])



def spin_trace(blocks: np.ndarray) -> np.ndarray:
    """The spin-independent half of a ``(nh, nh, 2, 2)`` operator.

    ``delta_{s s'} (M[..., 0, 0] + M[..., 1, 1]) / 2`` -- and the remainder,
    ``M - spin_trace(M)``, **is** the spin-orbit coupling. That decomposition is
    what ``soc_scale`` interpolates along, and it is worth saying why it is this
    one rather than the obvious alternative.

    The obvious alternative is to give the two ``j`` channels of a shell one
    ``j``-averaged ``D``, which is ``average_pp.f90``'s arithmetic, and to
    expect Clebsch-Gordan completeness to collapse ``sum_j D_j fcoef^(l,j)``
    onto ``delta_{s s'} delta_{m m'}``. **It does not, and the reason is the
    one thing** ``average_pp`` **does that looks cosmetic**: completeness needs
    the two shells to share a radial function, and a fully-relativistic file
    gives them *different* ``beta``, so ``average_pp`` merges the pair into a
    single refitted projector with ``sqrt(D)`` weights. Averaging ``D`` alone
    leaves ``D_bar [P_{l-1/2}(beta_a) + P_{l+1/2}(beta_b)]``, a sum of two
    shell projectors on two different radial functions, which is **not**
    spin-diagonal -- measured on ``Co.rel-pbe-nd-rrkjus``, its spin off-diagonal
    block comes out at 2.2, *larger* than the coupled operator's own 1.1.

    The spin trace needs no such argument. It is spin-diagonal and spin-
    *independent* by construction, so a Hamiltonian built from it cannot
    produce a magnetic anisotropy whatever the radial functions are; it is
    exact for a scalar-relativistic species, where ``dvan_so`` is already
    ``delta_{s s'} D``; and it requires nothing of the dataset, which is what
    lets ``soc_scale`` work on ultrasoft and PAW where ``average_pp`` refuses.
    What it is *not* is QE's scalar-relativistic dataset: that is a different
    pseudopotential, and reproducing it is the two-file route's job
    (:mod:`pypresso.workflows.anisotropy`), not this knob's.
    """
    blocks = np.asarray(blocks)
    mean = 0.5 * (blocks[..., 0, 0] + blocks[..., 1, 1])
    out = np.zeros_like(blocks)
    out[..., 0, 0] = mean
    out[..., 1, 1] = mean
    return out


class SpinOrbitCoupling:
    """One species' spin-orbit machinery: ``fcoef`` and what is built from it.

    Held per species rather than per atom because every atom of a species shares
    it -- the coefficients depend on the pseudopotential and on nothing about
    where the atom sits.
    """

    def __init__(self, pseudo: Pseudopotential, soc_scale: float = 1.0):
        self.pseudo = pseudo
        self.nh = pseudo.nh
        self.has_so = pseudo.has_so
        self.soc_scale = float(soc_scale)
        if self.soc_scale not in (0.0, 1.0):
            raise ValueError(
                f"soc_scale = {self.soc_scale}: only 0 and 1 are implemented, "
                "and the reason is the overlap rather than the potential. "
                "qq_so's coupling-free end is its spin trace, which is "
                "spin-independent -- so at soc_scale = 0 the anisotropy "
                "vanishes identically whatever else that operator is -- but a "
                "blend of it with the true qq_so partway is not the overlap of "
                "any set of projectors, and S = 1 + sum |beta> qq_so <beta| "
                "stops being a usable metric. Measured on the cobalt slab, "
                "where the answer is under a meV: -132 meV at 0.25 and -102 "
                "at 0.5, against exactly 0.000000 at 0 and a value validated "
                "against pw.x at 1. Elk's socscf takes any value because it "
                "scales a genuinely additive sigma.L term in an all-electron "
                "second-variational Hamiltonian; a pseudopotential has no such "
                "term to scale"
            )

        if not self.has_so:
            # A scalar-relativistic species inside a spin-orbit calculation is
            # legal and common (light atoms in a heavy compound). Its nonlocal
            # potential is spin-diagonal, which is exactly ``fcoef`` being the
            # identity on each spin block -- so nothing below needs a branch.
            identity = np.eye(self.nh)
            self.fcoef = np.zeros((self.nh, self.nh, 2, 2), dtype=complex)
            self.fcoef[:, :, 0, 0] = identity
            self.fcoef[:, :, 1, 1] = identity
            self.dvan_so = self._diagonal_blocks(self._expanded_dij())
            self.dvan_scalar = self.dvan_so
            return

        # ``init_us_1``, in its order: the full coefficients build ``dvan_so``,
        # and only then are the cross-radial entries dropped. See the module
        # docstring -- swapping these two lines is a silent error.
        full = _fcoef(pseudo)
        indv, _, _, _ = _channel_table(pseudo)

        # ``socscf`` (Elk manual 5.118, ``gensocfr.f90``), which exists there
        # for exactly this: "to enhance the effect of spin-orbit coupling in
        # order to accurately determine the magnetic anisotropy energy".
        #
        # **What is scaled is the spin-traceless half**, because a
        # pseudopotential has no additive ``xi L.S`` term to scale: the
        # coupling lives in the spin structure of ``dvan_so``, which is built
        # from spin-*independent* radial data, so everything spin-dependent in
        # it is the coupling and :func:`spin_trace` removes exactly that. The
        # same rule applies to :meth:`qq_so` and the **opposite** one to
        # ``newd_so``'s sandwich, whose input carries the exchange field --
        # see :func:`pypresso.scf.driver._newd_noncollinear`.
        #
        # Two other decompositions were tried and each fails the identity that
        # ``soc_scale = 0`` must give **exactly zero** anisotropy. They are
        # written down because each looked like the tidy answer. Giving the two
        # ``j`` channels one ``j``-averaged ``D`` -- ``average_pp``'s own
        # arithmetic -- leaves a spin off-diagonal block of **2.2**, *larger*
        # than the coupled operator's 1.1, because Clebsch-Gordan completeness
        # needs the two shells to share a radial function and a relativistic
        # file gives them different ``beta`` (which is why ``average_pp``
        # refits them, and why it refuses ultrasoft and PAW). Blending
        # ``fcoef`` itself towards the identity is tidier still and keeps the
        # overlap a congruence, but its identity limit drops ``dion``'s
        # off-diagonal radial couplings within a shell, which an ultrasoft
        # dataset needs: **1082 meV** on the cobalt slab where the answer is
        # zero. Every one of these failures is silent -- covariance still holds
        # algebraically at zero, so a wrong operator shows up as a large number
        # rather than as an error.
        coupled = self._expanded_dij()[:, :, None, None] * full
        self.dvan_scalar = spin_trace(coupled)
        self.dvan_so = (
            coupled if self.soc_scale == 1.0
            else self.dvan_scalar + self.soc_scale * (coupled - self.dvan_scalar)
        )

        same_radial = indv[:, None] == indv[None, :]
        self.fcoef = np.where(same_radial[:, :, None, None], full, 0.0)

    def _expanded_dij(self) -> np.ndarray:
        """``D^(0)`` in the channel basis, ``(nh, nh)``.

        For a fully-relativistic species this is ``dion(vi, vj)`` with **no**
        ``lm`` selection rule: ``init_us_1``'s spin-orbit branch multiplies by
        ``fcoef``, which already vanishes unless the two channels belong to the
        same ``(l, j)`` shell and the same ``m_j`` couples them. Imposing the
        collinear ``lm_i == lm_j`` rule here as well would throw away exactly the
        off-diagonal terms the spin-orbit coupling consists of.
        """
        pseudo = self.pseudo
        indv, nhtol, nhtolm, _ = _channel_table(pseudo)
        if pseudo.dij is None:
            return np.zeros((self.nh, self.nh))
        block = pseudo.dij[np.ix_(indv, indv)]
        if self.has_so:
            return block
        return np.where(nhtolm[:, None] == nhtolm[None, :], block, 0.0)

    @staticmethod
    def _diagonal_blocks(matrix: np.ndarray) -> np.ndarray:
        """A spin-independent ``(nh, nh)`` matrix as ``(nh, nh, 2, 2)`` blocks."""
        blocks = np.zeros(matrix.shape + (2, 2), dtype=complex)
        blocks[:, :, 0, 0] = matrix
        blocks[:, :, 1, 1] = matrix
        return blocks

    def qq_so(self, qq: np.ndarray) -> np.ndarray:
        """``transform_qq_so``: the augmentation integrals as spin blocks.

        ``qq_so[kh, lh, s1, s2] = sum_s sum_{ih,jh} F[kh,ih,s1,s] qq[ih,jh]
        F[jh,lh,s,s2]`` (``upflib/upf_spinorb.f90``). For a species that is not
        fully relativistic ``fcoef`` is the identity on each diagonal block, so
        this reproduces ``qq`` on blocks ``(0,0)`` and ``(1,1)`` and zero
        elsewhere -- which is what the Fortran's ``else`` branch writes out by
        hand.

        **``soc_scale`` scales this by the spin trace and not to** ``qq``, and
        that is the opposite of the rule ``_newd_noncollinear`` follows for the
        other ``fcoef`` sandwich. It was measured rather than reasoned out.
        Interpolating towards ``_diagonal_blocks(qq)`` -- the ``fcoef =
        identity`` limit, which *looks* like the scalar-relativistic overlap
        and is what the Fortran's ``else`` branch writes -- fails the identity
        that ``soc_scale = 0`` must give **zero** anisotropy, by 169 meV on
        hexagonal cobalt, where the spin trace gives 0.000000. The reason is
        that ``S = 1 + sum |beta> qq_so <beta|`` is a *metric*: a
        fully-relativistic dataset's scalar ``qq`` is never used on its own by
        any code -- ``transform_qq_so`` always dresses it -- and undressed it
        is not the overlap of these ``beta`` at all, so the generalised
        eigenproblem stops being well posed and the iterative solver lands
        somewhere different for each direction. Covariance still holds
        *algebraically* at ``soc_scale = 0``, which is why the failure shows up
        as a large number rather than as an error.
        """
        full = self._sandwich({(0, 0): qq, (1, 1): qq})
        if self.soc_scale == 1.0:
            return full
        scalar = spin_trace(full)
        return scalar + self.soc_scale * (full - scalar)

    def _sandwich(self, blocks: dict) -> np.ndarray:
        """``sum_{s t} F^{s1 s} M^{s t} F^{t s2}`` for a block-structured ``M``.

        ``blocks`` maps a spin pair to the ``(nh, nh)`` matrix in that block;
        absent pairs are zero. This one contraction is ``transform_qq_so``, the
        integral half of ``newd_so``, and -- transposed -- ``add_becsum_so``.
        Writing it once is not only economy: the two ``fcoef`` factors carry
        their indices in *different* orders in the Fortran
        (``fcoef(ih,kh,is1,s)`` against ``fcoef(lh,jh,s,is2)``), and one tested
        contraction is how that stays right.
        """
        f = self.fcoef  # (nh, nh, 2, 2), indexed [ih, kh, s1, s2]
        result = np.zeros((self.nh, self.nh, 2, 2), dtype=complex)
        for is1 in range(2):
            for is2 in range(2):
                total = np.zeros((self.nh, self.nh), dtype=complex)
                for s in range(2):
                    for t in range(2):
                        matrix = blocks.get((s, t))
                        if matrix is None:
                            continue
                        total = total + f[:, :, is1, s] @ matrix @ f[:, :, t, is2]
                result[:, :, is1, is2] = total
        return result

    def deeq_so(self, deeq_components: np.ndarray) -> np.ndarray:
        """``newd_so``: ``D^(0)_so`` plus the transformed potential integrals.

        Args:
            deeq_components: ``(nspin_mag, nh, nh)`` real -- ``int V_s Q_ij``
                for each component of the potential, as ``newq`` produces them.

        Returns ``(nh, nh, 2, 2)`` complex, the coefficients the spinor
        Hamiltonian's nonlocal term uses.

        The nonmagnetic case is the Fortran's ``.NOT. domag`` branch, where only
        the charge component exists and the sandwich reduces to
        ``sum_s F[s1,s] D F[s,s2]``.
        """
        blocks = pauli_blocks(np.asarray(deeq_components))
        mapping = {(s, t): blocks[s, t] for s in range(2) for t in range(2)}
        return self.dvan_so + self._sandwich(mapping)

    def becsum_so(self, becsum_nc: np.ndarray, nspin_mag: int) -> np.ndarray:
        """``add_becsum_so``: spinor projector occupations -> ``(nspin_mag, nh, nh)``.

        Args:
            becsum_nc: ``(nh, 2, nh, 2)`` complex, ``sum_b w_b <psi_b|beta_i s1>
                <beta_j s2|psi_b>``.

        The Fortran guards its sums on the two channels sharing ``l``, ``j`` and
        radial index; every one of those is already enforced by ``fcoef`` being
        zero otherwise, so the guards are dropped here rather than duplicated.

        The result is the **symmetric** matrix this code stores, not QE's packed
        upper triangle with the off-diagonal entries doubled -- the two carry the
        same information and the conversion is where a factor of two goes
        missing, so it is done in one place (see
        :func:`pypresso.scf.density.becsum`).
        """
        f = self.fcoef
        components = []
        # The four Pauli traces, in QE's (charge, x, y, z) order. Each is the
        # same sandwich with a different pairing of the two spin sums, which is
        # what the identity and the three Pauli matrices amount to here.
        weights = _PAULI if nspin_mag == 4 else _PAULI[:1]
        for sigma in weights:
            total = np.zeros((self.nh, self.nh), dtype=complex)
            for is1 in range(2):
                for is2 in range(2):
                    for s in range(2):
                        for t in range(2):
                            if sigma[s, t] == 0.0:
                                continue
                            # sum_{kh,lh} F[kh,ih,is1,s] B[kh,is1,lh,is2] F[jh,lh,t,is2]
                            total = total + sigma[s, t] * (
                                f[:, :, is1, s].T
                                @ becsum_nc[:, is1, :, is2]
                                @ f[:, :, t, is2].T
                            )
            components.append(np.real(total))
        stacked = np.stack(components)
        return 0.5 * (stacked + np.swapaxes(stacked, -1, -2))


#: The identity and the three Pauli matrices, in QE's ``(n, m_x, m_y, m_z)``
#: order. ``rho_s = sum_{s s'} sigma_{s' s} rho_{s s'}`` is the trace that turns
#: a spin-density matrix into those four real components.
_PAULI = np.array([
    [[1.0, 0.0], [0.0, 1.0]],
    [[0.0, 1.0], [1.0, 0.0]],
    [[0.0, -1.0j], [1.0j, 0.0]],
    [[1.0, 0.0], [0.0, -1.0]],
])


def build_spin_orbit(pseudos: tuple[Pseudopotential, ...], soc_scale: float = 1.0) -> tuple:
    """One :class:`SpinOrbitCoupling` per species.

    ``soc_scale`` is Elk's ``socscf`` and applies to every species alike: it is
    a property of the *calculation*, not of a dataset.
    """
    return tuple(SpinOrbitCoupling(pseudo, soc_scale) for pseudo in pseudos)


# --- the same transforms, on device --------------------------------------------
#
# ``fcoef`` is host-side setup, but ``becsum`` is rebuilt from the wavefunctions
# every SCF iteration and has to stay inside the compiled region. The two live
# together so that the contraction is written once and the index order cannot
# drift between the setup version and the traced one.


def becsum_transform(fcoef, becsum_nc, nspin_mag: int):
    """``add_becsum_so`` for a whole species at once, in JAX.

    Args:
        fcoef: ``(nh, nh, 2, 2)`` complex -- the zeroed coefficients.
        becsum_nc: ``(nat, nh, 2, nh, 2)`` complex, from
            :func:`pypresso.scf.density.spinor_becsum`.

    Returns ``(nspin_mag, nat, nh, nh)`` real: the projector occupations in the
    representation the augmentation charge and the one-centre terms use.

    The four output components are the Pauli traces of the transformed
    spin-density matrix; the Fortran writes them out one at a time with the
    signs of ``I``, ``sigma_x``, ``sigma_y``, ``sigma_z`` inline, which is what
    :data:`_PAULI` collects.
    """
    import jax.numpy as jnp

    sigma = jnp.asarray(_PAULI[:1] if nspin_mag == 1 else _PAULI, dtype=fcoef.dtype)
    # F[kh, ih, is1, s] B[na, kh, is1, lh, is2] F[jh, lh, t, is2], summed over
    # kh, lh, is1, is2 and the (s, t) pair the Pauli matrix selects.
    transformed = jnp.einsum(
        "cst,kias,nkalb,jltb->cnij", sigma, fcoef, becsum_nc, fcoef, optimize=True
    )
    real = jnp.real(transformed)
    # QE stores the packed upper triangle with the off-diagonal entries doubled;
    # this code stores the full symmetric matrix, and the two agree exactly when
    # the transpose is folded in here rather than left to the caller.
    return 0.5 * (real + jnp.swapaxes(real, -1, -2))
