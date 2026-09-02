"""The occupation matrix decomposed into tensor moments.

A Hubbard occupation matrix on a spinor is a ``2(2l+1)`` Hermitian matrix, and
counting its entries says nothing about what is in it. The **coupled 3-index
tensor moments** are an orthonormal, complete basis of that space in which each
element is a physical multipole:

    Gamma^{kpr}_t = sqrt(2r+1) sum_{x,y} ( k  r  p ; -x  t  -y ) Gamma^{kp}_{xy}

with ``k`` the rank in the orbital index (``0 .. 2l``), ``p`` the rank in spin
(``0`` or ``1``), ``r`` their coupling (``|k-p| .. k+p``) and ``t`` its
projection. Any Hermitian ``N`` is then ``N = sum w^{kpr}_t Gamma^{kpr}_t`` with
**real** ``w``, and the first few carry names:

===========  ============================================================
``k p r``    what it is
===========  ============================================================
``0 0 0``    the shell's charge
``0 1 1``    its spin moment, three components in ``t``
``1 1 0``    proportional to ``L . S`` -- the spin-orbit energy of the shell
``2 0 0``    the quadrupole of the charge; ``k >= 2`` with ``p = 1`` are the
             multipoles an orbitally ordered magnet carries and a collinear
             occupation matrix cannot
===========  ============================================================

Elk's ``tm3todm.f90``/``dmtotm3.f90``/``tm2todm.f90`` (task 400, ``tmwrite``),
after Bultmark, Cricchio, Granas and Nordstrom, Phys. Rev. B **80**, 035121
(2009) and van der Laan and Thole, J. Phys.: Condens. Matter **7**, 9947 (1995).
``pw.x`` has nothing of the kind.

**The decomposition is a projection and needs no linear solve**, because the
basis is orthonormal under ``Tr(Gamma^† N)``: Elk builds each ``Gamma`` by
running the composition on a unit vector and takes a trace, and so does
:func:`decompose`. What that buys is the constraint below --
:mod:`defumat.hubbard.ftm` fixes one ``w`` and lets the SCF find the state that
has it -- and what it costs is a general Wigner 3-j symbol, which the package
did not have: :func:`wigner3j` is the Racah formula in **doubled** arguments, so
that the half-integer spin symbols ``wigner3jf`` needs are the same function.

**Elk's spin index convention is not assumed, it is measured.** Whether index 0
of the spin pair is "up" decides the *sign* of every ``p = 1`` moment and
nothing else, and the transcription below is checked against the physical
statement instead -- ``w^{011}`` must reproduce the magnetization the density
already carries. See ``tests/unit/test_tensor_moments.py``.
"""

from __future__ import annotations

from functools import lru_cache
from math import factorial, sqrt

import numpy as np

__all__ = [
    "MomentLabel",
    "compose",
    "decompose",
    "moment_labels",
    "moment_matrices",
    "wigner3j",
]


def wigner3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    """The Wigner 3-j symbol, with **every argument doubled**.

    ``wigner3j(2, 4, 2, 0, 0, 0)`` is ``(1 2 1; 0 0 0)``. Doubling is what lets
    the half-integer spin symbols share one implementation with the integer
    orbital ones -- Elk keeps two routines, ``wigner3j`` and ``wigner3jf``, whose
    docstring says the formula is identical.

    The Racah expression, with the factorials exact and only the square root in
    floating point. Everything here is a small integer.
    """
    if m1 + m2 + m3 != 0:
        return 0.0
    if any((j + m) % 2 for j, m in ((j1, m1), (j2, m2), (j3, m3))):
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        return 0.0
    if (j1 + j2 + j3) % 2:
        return 0.0

    # Halves, all integers once the tests above pass.
    a, b, c = (j1 + j2 - j3) // 2, (j1 - j2 + j3) // 2, (-j1 + j2 + j3) // 2
    delta = factorial(a) * factorial(b) * factorial(c) / factorial((j1 + j2 + j3) // 2 + 1)
    norm = 1.0
    for j, m in ((j1, m1), (j2, m2), (j3, m3)):
        norm *= factorial((j + m) // 2) * factorial((j - m) // 2)

    total = 0.0
    lower = max(0, (j2 - j3 - m1) // 2, (j1 - j3 + m2) // 2)
    upper = min(a, (j1 - m1) // 2, (j2 + m2) // 2)
    for k in range(lower, upper + 1):
        denominator = (
            factorial(k)
            * factorial(a - k)
            * factorial((j1 - m1) // 2 - k)
            * factorial((j2 + m2) // 2 - k)
            * factorial((j3 - j2 + m1) // 2 + k)
            * factorial((j3 - j1 - m2) // 2 + k)
        )
        total += (-1) ** k / denominator
    sign = (-1) ** ((j1 - j2 - m3) // 2)
    return sign * sqrt(delta * norm) * total


class MomentLabel(tuple):
    """``(k, p, r, t)``, with the names the first few carry."""

    __slots__ = ()

    NAMES = {
        (0, 0, 0): "charge",
        (0, 1, 1): "spin moment",
        (1, 1, 0): "L . S",
        (2, 0, 0): "charge quadrupole",
    }

    @property
    def k(self) -> int:
        return self[0]

    @property
    def p(self) -> int:
        return self[1]

    @property
    def r(self) -> int:
        return self[2]

    @property
    def t(self) -> int:
        return self[3]

    @property
    def name(self) -> str:
        return self.NAMES.get(self[:3], f"k={self[0]} p={self[1]} r={self[2]}")

    def __str__(self) -> str:
        return f"w^({self.k}{self.p}{self.r})_{self.t}"


def moment_labels(l: int) -> tuple:
    """Every ``(k, p, r, t)`` of a shell, in Elk's order.

    ``(2l+1)^2 * 4`` of them, which is the dimension of the space of
    ``2(2l+1)`` Hermitian matrices -- so the basis is complete by counting as
    well as by construction.
    """
    labels = []
    for k in range(0, 2 * l + 1):
        for p in (0, 1):
            for r in range(abs(k - p), k + p + 1):
                for t in range(-r, r + 1):
                    labels.append(MomentLabel((k, p, r, t)))
    return tuple(labels)


@lru_cache(maxsize=None)
def _orbital_matrices(l: int, k: int) -> np.ndarray:
    """``dlm[x, m1, m2]``, ``tm2todm.f90``'s angular momentum matrices."""
    width = 2 * l + 1
    out = np.zeros((2 * k + 1, width, width))
    root = sqrt(2 * k + 1)
    for xi, x in enumerate(range(-k, k + 1)):
        for i2, m2 in enumerate(range(-l, l + 1)):
            sign = root if (l - m2) % 2 == 0 else -root
            for i1, m1 in enumerate(range(-l, l + 1)):
                out[xi, i1, i2] = sign * wigner3j(
                    2 * l, 2 * k, 2 * l, -2 * m2, 2 * x, 2 * m1
                )
    return out


@lru_cache(maxsize=None)
def _spin_matrices(p: int) -> np.ndarray:
    """``dsp[y, s1, s2]``, ``tm2todm.f90``'s spin matrices.

    The half-integer symbols, which is why :func:`wigner3j` takes doubled
    arguments. Elk's index convention is transcribed rather than reasoned out;
    what it decides is the sign of every ``p = 1`` moment, and that is checked
    against the magnetization the density already carries.
    """
    out = np.zeros((2 * p + 1, 2, 2))
    root = sqrt(2 * p + 1)
    for yi, y in enumerate(range(-p, p + 1)):
        for j in (1, 2):
            sign = root if j == 1 else -root
            for i in (1, 2):
                out[yi, i - 1, j - 1] = sign * wigner3j(
                    1, 2 * p, 1, 2 * j - 3, 2 * y, 3 - 2 * i
                )
    return out


@lru_cache(maxsize=None)
def moment_matrices(l: int, real_harmonics: bool = True) -> np.ndarray:
    """``(nmoments, 2, 2, 2l+1, 2l+1)`` complex: the basis ``Gamma^{kpr}_t``.

    **In the real harmonic basis by default, and that is not Elk's.** The 3-j
    construction below is written in complex ``Y_lm``, which is where Elk's
    ``dmatmt`` lives; the occupation matrix here is measured on the *real*
    harmonic projectors QE uses, so the basis is conjugated with ``rot_ylm``
    before it is returned -- the same matrix, the same direction and the same
    one-line contraction :func:`defumat.projwfc.angular_momentum.orbital_matrices`
    already uses for ``L``. Decomposing a real-harmonic ``ns`` in the complex
    basis is not an error anything downstream would report: the moments stay
    orthonormal and complete, and the ones that mean something get mixed into
    the ones that do not.

    Laid out the way :mod:`defumat.hubbard.occupations` lays out ``ns`` -- the
    two spin indices leading -- so a decomposition is one contraction.

    Built as ``tm3todm`` builds it: couple ``(k, p) -> r`` with a 3-j symbol,
    take the Kronecker product of the orbital and spin matrices, and split the
    real matrix into its symmetric and antisymmetric parts to make it Hermitian
    (Elk's ``dmrtoz``), which is what makes the ``w`` real.
    """
    labels = moment_labels(l)
    width = 2 * l + 1
    out = np.zeros((len(labels), 2, 2, width, width), dtype=complex)
    for index, label in enumerate(labels):
        k, p, r, t = label
        coupling = sqrt(2 * r + 1)
        real = np.zeros((2, 2, width, width))
        orbital, spin = _orbital_matrices(l, k), _spin_matrices(p)
        for xi, x in enumerate(range(-k, k + 1)):
            for yi, y in enumerate(range(-p, p + 1)):
                weight = coupling * wigner3j(
                    2 * k, 2 * r, 2 * p, -2 * x, 2 * t, -2 * y
                )
                if weight == 0.0:
                    continue
                real += weight * np.einsum(
                    "ij,ab->ijab", spin[yi], orbital[xi]
                )
        # ``dmrtoz``: the symmetric part is real and the antisymmetric part is
        # the imaginary one, which is what makes ``Gamma`` Hermitian on the
        # combined ``(m, spin)`` index rather than on either alone.
        flat = np.transpose(real, (0, 2, 1, 3)).reshape(2 * width, 2 * width)
        hermitian = 0.5 * (flat + flat.T) + 0.5j * (flat - flat.T)
        out[index] = np.transpose(
            hermitian.reshape(2, width, 2, width), (0, 2, 1, 3)
        )
    if real_harmonics and l > 0:
        from defumat.pseudo.spinorbit import LMAXX, rot_ylm

        u = rot_ylm(LMAXX)
        a = u[LMAXX - l:LMAXX + l + 1, :width]
        out = np.einsum("ba,nstbc,cd->nstad", a, out, a.conj())
    return out


def decompose(ns: np.ndarray, l: int) -> np.ndarray:
    """``w^{kpr}_t`` of one slot's occupation matrix, in :func:`moment_labels` order.

    ``ns`` is ``(4, ldim, ldim)`` -- the spin pairs packed as ``2 s1 + s2``, the
    layout :mod:`defumat.hubbard.occupations` produces. The basis is orthonormal
    under ``Tr(Gamma N)``, so this is a projection.
    """
    width = 2 * l + 1
    blocks = np.asarray(ns).reshape(2, 2, width, width)[:, :, :width, :width]
    basis = moment_matrices(l)
    return np.real(np.einsum("nstab,stab->n", np.conj(basis), blocks))


def compose(w: np.ndarray, l: int) -> np.ndarray:
    """``(4, ldim, ldim)``: the occupation matrix a set of moments describes."""
    basis = moment_matrices(l)
    blocks = np.einsum("n,nstab->stab", np.asarray(w, dtype=float), basis)
    return blocks.reshape((4,) + blocks.shape[2:])
