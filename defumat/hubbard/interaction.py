"""The Coulomb interaction matrix of the full (Liechtenstein) functional.

The simplified functional of :mod:`defumat.hubbard.energy` replaces the whole
electron-electron interaction inside the correlated shell by one number,
``U_eff = U - J0``. Liechtenstein, Anisimov and Zaanen (Phys. Rev. B **52**,
R5467 (1995)) keep the matrix instead:

    vee(m1, m2, m3, m4) = <m1 m2| 1/|r - r'| |m3 m4>
                        = sum_k  F^k  4 pi/(2k+1)
                                 sum_{q=-k}^{k} a_{kq}(m1, m3) a_{kq}(m2, m4)

with ``a_{kq}(m, m')`` the expansion of a product of two real spherical
harmonics of the shell's ``l`` back into harmonics, and ``F^k`` (``k = 0, 2,
... 2l``) the Slater integrals of the radial part. Only ``F^0 = U`` survives for
an ``s`` shell; a ``d`` shell has ``F^0, F^2, F^4`` and an ``f`` shell one more.

**The angular part is already here.** ``PW/src/plus_u_full.f90``'s
``hubbard_matrix`` calls ``aainit_1``, which is ``upflib/uspp.f90``'s ``aainit``
with different dimensions -- and that is
:func:`defumat.pseudo.coupling.harmonic_products`, written for the augmentation
charge. So the matrix is a contraction of a table this package already builds
and validates, in the *real* harmonic basis the Hubbard projectors are measured
in. Elk's ``genveedu.f90`` contracts Wigner 3-j symbols in the *complex* basis
instead, because Elk's density matrix lives there; transcribing that would mean
carrying ``rot_ylm`` through the whole functional for nothing.

**`F^4/F^2` is fixed in a routine that is not the one building the matrix, and
reading only the latter is silently wrong.** ``hubbard_matrix`` takes the ``d``
shell as ``F^2 = 5 J + 31.5 B``, ``F^4 = 9 J - 31.5 B`` with ``B`` Racah's
parameter -- so ``B = 0`` there would mean ``F^4/F^2 = 1.8``, where atomic
spectroscopy says 0.625. QE never evaluates that: ``PW/src/ldaU.f90``'s
``init_hubbard`` substitutes ``B = 0.114774114774 J`` whenever ``B`` is exactly
zero, and that number is precisely the one giving ``F^2 = 8.61538 J``,
``F^4 = 5.38462 J``, ``F^4/F^2 = 0.625`` -- the same ratio Elk hardcodes as
``r1`` in ``genfdu.f90``. The substitution lives in :func:`default_racah` and is
applied where the setup is built, as QE applies it. Both codes also agree on
``J = (F^2 + F^4)/14``, which is the identity :func:`exchange_from_slater`
inverts and which is what the pair should be tested against.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from defumat.pseudo.coupling import harmonic_products

__all__ = [
    "coulomb_matrix",
    "racah_to_slater",
    "default_racah",
    "exchange_from_slater",
    "slater_integrals",
]

#: ``ldaU.f90:421-427``: what ``init_hubbard`` puts in ``Hubbard_J(2:3)`` when
#: the input leaves it at zero, as multiples of ``J``. Index by ``l``.
_RACAH_DEFAULTS = {
    2: (0.114774114774,),
    3: (0.002268, 0.0438),
}


def default_racah(l: int, j: float, given: tuple[float, ...]) -> tuple[float, ...]:
    """``init_hubbard``'s substitution for a Racah parameter left at zero.

    ``given`` is ``(B,)`` for ``l = 2`` and ``(E2, E3)`` for ``l = 3``; ``()``
    for ``s`` and ``p``, which have no such parameter. Each entry that is
    *exactly* zero is replaced by its multiple of ``J`` -- QE tests ``== 0.d0``
    rather than a tolerance, so a deliberate 1e-12 is kept and a plain 0.0 is
    not, and that is copied rather than improved on.
    """
    ratios = _RACAH_DEFAULTS.get(l, ())
    values = list(given) + [0.0] * (len(ratios) - len(given))
    return tuple(
        ratio * j if value == 0.0 else value
        for value, ratio in zip(values[: len(ratios)], ratios)
    )


def slater_integrals(l: int, u: float, j: float, racah=()) -> np.ndarray:
    """``F[0:7]`` in Ry, ``plus_u_full.f90:69-88``.

    ``racah`` is what :func:`default_racah` returned: ``(B,)`` for ``d``,
    ``(E2, E3)`` for ``f``. Odd ``k`` are zero and are carried anyway so that
    the array can be indexed by ``k`` directly, as the Fortran's is.
    """
    f = np.zeros(7)
    f[0] = u
    if l == 0:
        return f
    if l == 1:
        f[2] = 5.0 * j
        return f
    if l == 2:
        (b,) = racah
        f[2] = 5.0 * j + 31.5 * b
        f[4] = 9.0 * j - 31.5 * b
        return f
    if l == 3:
        e2, e3 = racah
        f[2] = 225.0 / 54.0 * j + 32175.0 / 42.0 * e2 + 2475.0 / 42.0 * e3
        f[4] = 11.0 * j - 141570.0 / 77.0 * e2 + 4356.0 / 77.0 * e3
        f[6] = 7361.64 / 594.0 * j + 36808.2 / 66.0 * e2 - 11154.0e-2 * e3
        return f
    raise NotImplementedError(
        f"the full DFT+U functional is not implemented for l = {l}; QE's "
        "hubbard_matrix stops at l = 3 as well"
    )


def racah_to_slater(l: int, e) -> np.ndarray:
    """``F[0:7]`` in Ry from the Racah parameters ``E^0..E^l``.

    Elk's ``inpdftu = 3``, ``genfdu.f90:55-127``, with the relations of Condon
    and Shortley's *The Theory of Atomic Spectra* (1935). This is a **change of
    coordinates**, not new physics: ``(E^0, E^1, E^2)`` spans exactly the space
    ``(U, J, B)`` spans for a ``d`` shell, and the same for ``f``. It is here
    because the conversion is cheap and easy to get wrong, and because it is the
    third of Elk's five ``inpdftu`` values -- **no card syntax reaches it**, for
    the reason :data:`defumat.hubbard.manifold.SLATER_SOURCES` gives.
    """
    e = np.asarray(e, dtype=float)
    f = np.zeros(7)
    if l == 0:
        f[0] = e[0]
        return f
    if l == 1:
        f[0] = e[0] + (5.0 / 3.0) * e[1]
        f[2] = (25.0 / 3.0) * e[1]
        return f
    if l == 2:
        a = np.array([
            [1.0, 1.4, 0.0],
            [0.0, 0.1428571428571428, 1.285714285714286],
            [0.0, 2.8571428571428571e-2, -0.1428571428571428],
        ])
        v = a @ e[0:3]
        f[0], f[2], f[4] = v[0], 49.0 * v[1], 441.0 * v[2]
        return f
    if l == 3:
        f[0] = e[0] + (9.0 / 7.0) * e[1]
        a = np.array([
            [2.3809523809523808e-2, 3.404761904761904, 0.2619047619047619],
            [1.2987012987012984e-2, -1.688311688311688, 5.1948051948051951e-2],
            [2.1645021645021645e-3, 7.5757575757575760e-2, -1.5151515151515152e-2],
        ])
        v = a @ e[1:4]
        f[2], f[4], f[6] = 225.0 * v[0], 1089.0 * v[1], (184041.0 / 25.0) * v[2]
        return f
    raise NotImplementedError(f"no Racah relations for l = {l}")


def exchange_from_slater(l: int, f) -> float:
    """``J`` recovered from the Slater integrals (Elk's ``genfdu.f90:150-166``).

    The inverse of the ``J`` half of :func:`slater_integrals` and independent of
    the Racah parameters, so ``J -> F^k -> J`` is a round trip that tests the
    coefficients without needing another code.
    """
    f = np.asarray(f, dtype=float)
    if l == 0:
        return 0.0
    if l == 1:
        return f[2] / 5.0
    if l == 2:
        return (f[2] + f[4]) / 14.0
    if l == 3:
        return (2.0 / 45.0) * f[2] + (1.0 / 33.0) * f[4] + (50.0 / 1287.0) * f[6]
    raise NotImplementedError(f"no J relation for l = {l}")


@lru_cache(maxsize=None)
def _angular(l: int) -> tuple[np.ndarray, ...]:
    """``A_k[q, m, m']`` for each even ``k``, already scaled by ``4 pi/(2k+1)``.

    Cached on ``l`` because it depends on nothing else -- the same table serves
    every atom of every species with that manifold, and rebuilding it inverts a
    matrix of spherical harmonics each time.
    """
    ap = harmonic_products(l)
    offset, width = l * l, 2 * l + 1
    block = ap[:, offset : offset + width, offset : offset + width]
    return tuple(
        block[k * k : k * k + 2 * k + 1] * np.sqrt(4.0 * np.pi / (2.0 * k + 1.0))
        for k in range(0, 2 * l + 1, 2)
    )


def coulomb_matrix(l: int, f) -> np.ndarray:
    """``vee[m1, m2, m3, m4]``, shape ``(2l+1,) * 4``, in Ry.

    ``hubbard_matrix``'s quadruple loop, written as one contraction per ``k``.
    The ``4 pi/(2k+1)`` is split symmetrically between the two factors in
    :func:`_angular`, which is the same number and keeps the table's own
    normalisation visible.
    """
    f = np.asarray(f, dtype=float)
    width = 2 * l + 1
    vee = np.zeros((width, width, width, width))
    for index, a in enumerate(_angular(l)):
        k = 2 * index
        vee += f[k] * np.einsum("qac,qbd->abcd", a, a)
    return vee
