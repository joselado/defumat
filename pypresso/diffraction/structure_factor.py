"""X-ray and magnetic structure factors -- the Fourier coefficients of the
density and of the magnetization, on the reflections a diffractometer measures.

``PLAN.md`` P61; Elk's tasks 195 and 196 (``sfacrho.f90``, ``sfacmag.f90``).
Quantum ESPRESSO computes neither.

.. math::

    F(\\mathbf{H}) = \\int_\\Omega \\rho(\\mathbf{r})\\,
                     e^{i \\mathbf{H} \\cdot \\mathbf{r}}\\, d^3 r,
    \\qquad
    F_j(\\mathbf{H}) = \\int_\\Omega m_j(\\mathbf{r})\\,
                       e^{i \\mathbf{H} \\cdot \\mathbf{r}}\\, d^3 r

for every reciprocal lattice vector with ``|H| <= hmax``. In a plane-wave code
that integral **is** an array that already exists: the density's own Fourier
coefficient, up to the crystallographers' two conventions, which are the
positive phase and no ``1/Omega``. Elk's ``zftrf`` returns the physicists'
``(1/Omega) int rho e^{-iH.r}`` and ``sfacrho`` prints ``Omega`` times its
complex *conjugate*; :func:`structure_factors_of_field` takes the
positive-phase transform directly instead, which is the same number for a
density and is also right for a complex field. ``F(0)`` is then the number of
electrons in the cell and ``F`` is in electrons; the magnetic one is in Bohr
magnetons.

**Three things are traps here and each is silent.**

*The grid holds more coefficients than the calculation has.* A discrete
transform of the density on the dense FFT box returns a coefficient for every
frequency out to the corner of the box, and only those inside the ``ecutrho``
sphere mean anything -- past it they are the aliased tail of ``|psi|^2`` rather
than a Fourier component of the density. Nothing raises: the numbers come back
small and smooth and wrong. The caller is checked against ``sqrt(ecutrho)`` in
:func:`~pypresso.workflows.sfac.run_structure_factors`.

*A star may only be collapsed with the symmorphic operations.* ``rho(Rr) =
rho(r)`` gives ``F(RH) = F(H)`` for a point operation, but an operation with a
fractional translation ``f`` gives ``F(RH) = e^{-i 2 pi H.f} F(H)`` -- equal in
modulus, different in phase, and diamond has exactly such operations. Elk tests
``tv0symc`` for the same reason, and :func:`symmorphic_rotations` is that test.

*The magnetization is an axial vector, so its star members are not equal at
all*: ``F_mag(RH) = det(R) R F_mag(H)``. The reduced set carries one
representative per star, as Elk's does, and the relation is how a member is
recovered -- not by copying the representative.

**What this is a structure factor of.** A pseudopotential density is
valence-only, so ``F(H)`` here is the *valence* structure factor and is not
what a diffraction experiment measures, which includes the core. The gap is
large and systematic -- silicon's ``F(000)`` is 8 here and 28 all-electron --
and it is not a defect to be tuned away but the thing to state. It also has an
exception that makes the quantity sharp rather than approximate: in a
**forbidden reflection** the spherical part of every atom cancels by symmetry,
so the core contributes nothing and what is left is the aspherical bonding
density, which is exactly the part a pseudopotential keeps. Silicon's (222) is
the classic one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "HVectors",
    "StructureFactors",
    "h_vectors",
    "symmorphic_rotations",
    "conventional_transform",
    "structure_factors_of_field",
]

#: Two Miller indices count as the same reflection below this.
_TOLERANCE = 1.0e-8


@dataclass(frozen=True)
class HVectors:
    """The reflections a structure factor is reported on.

    ``miller`` are integer coefficients on the *primitive* reciprocal basis,
    which is what indexes the density's transform; ``indices`` are the same
    vectors after the output transformation (Elk's ``vhmat``), which is what
    turns an fcc primitive triple into the conventional ``(h k l)`` a
    crystallographer would recognise. ``multiplicity`` counts how many members
    of the reflection's star were collapsed onto it.
    """

    miller: np.ndarray  # (nh, 3), integers on the primitive reciprocal basis
    cartesian: np.ndarray  # (nh, 3), 1/bohr
    length: np.ndarray  # (nh,), 1/bohr
    multiplicity: np.ndarray  # (nh,), integers
    indices: np.ndarray  # (nh, 3), the transformed (h k l) for output

    def __len__(self) -> int:
        return int(self.miller.shape[0])

    def find(self, hkl) -> int:
        """The row carrying the transformed index ``hkl``, or ``-1``.

        Looks in :attr:`indices`, i.e. in whatever setting the transform put
        the reflections into, because that is the label a reflection is known
        by. Only the representative of each star is stored, so an equivalent
        index of a reduced set will not be found -- ask for the one the table
        prints.
        """
        target = np.asarray(hkl, dtype=float)
        matches = np.flatnonzero(
            np.all(np.abs(self.indices - target) < 1.0e-6, axis=1))
        return int(matches[0]) if matches.size else -1


@dataclass(frozen=True)
class StructureFactors:
    """``F(H)`` for the charge and, when there is one, for the magnetization.

    :attr:`charge` is in electrons per cell and :attr:`magnetization` in Bohr
    magnetons per cell, with one column per magnetization component: a single
    one (``m_z``) for a collinear run and three for a noncollinear one. Both
    are complex; a centrosymmetric crystal with the inversion centre at the
    origin makes them real.
    """

    vectors: HVectors
    charge: np.ndarray  # (nh,) complex, electrons
    magnetization: np.ndarray | None = None  # (nh, ncomp) complex, mu_B
    #: The energy window the density was rebuilt in (Elk's ``wsfac``), in Ry,
    #: or ``None`` when every occupied state contributed.
    window: tuple[float, float] | None = None
    #: Whether the frozen core charge of a nonlinear core correction was added
    #: to the valence density before transforming.
    core_included: bool = False

    def __len__(self) -> int:
        return len(self.vectors)

    def find(self, hkl) -> int:
        return self.vectors.find(hkl)

    def of(self, hkl) -> complex:
        """The charge structure factor of one reflection, by its index."""
        row = self.find(hkl)
        if row < 0:
            raise KeyError(
                f"no reflection {tuple(hkl)} in this set: it is beyond hmax, or "
                "it is not the representative its star was collapsed onto "
                "(print .table() to see which one is)"
            )
        return complex(self.charge[row])

    def table(self, limit: int | None = None) -> str:
        """Elk's ``SFACRHO.OUT`` table, as a string.

        One line per reflection: its index, the multiplicity of its star,
        ``|H|`` in 1/bohr, and the real part, imaginary part and modulus of
        ``F``.
        """
        header = (f"{'h':>6}{'k':>6}{'l':>6}{'mult':>6}{'|H|':>12}"
                  f"{'Re F':>14}{'Im F':>14}{'|F|':>14}")
        lines = [header, "-" * len(header)]
        rows = range(len(self) if limit is None else min(limit, len(self)))
        for row in rows:
            index = self.vectors.indices[row]
            integral = np.all(np.abs(index - np.rint(index)) < 1.0e-6)
            label = ("".join(f"{int(round(v)):>6d}" for v in index) if integral
                     else "".join(f"{v:>6.2f}" for v in index))
            value = self.charge[row]
            lines.append(
                f"{label}{self.vectors.multiplicity[row]:>6d}"
                f"{self.vectors.length[row]:>12.6f}"
                f"{value.real:>14.6f}{value.imag:>14.6f}{abs(value):>14.6f}"
            )
        return "\n".join(lines)


def symmorphic_rotations(symmetries) -> np.ndarray:
    """The rotations a reflection's star may be collapsed with.

    Elk's ``genhvec`` keeps the operations with ``tv0symc`` (no fractional
    translation) and ``lspnsymc == 1`` (no spin rotation), and both conditions
    are about the *phase* of the coefficient rather than its modulus: a
    fractional translation ``f`` sends ``F(H)`` to ``e^{-i 2 pi H.f} F(H)``,
    and an operation that is a symmetry only together with time reversal sends
    the magnetization to minus itself. Collapsing either kind averages numbers
    that are not equal.

    Returns ``(nsym, 3, 3)`` integer matrices, always including the identity.
    """
    if symmetries is None:
        return np.eye(3, dtype=int)[None, :, :]
    rotations = symmetries.rotation_array()
    translations = symmetries.translation_array()
    reversed_ = symmetries.t_rev_array()
    keep = []
    for index in range(len(rotations)):
        translation = translations[index]
        folded = np.minimum(np.abs(translation), np.abs(1.0 - translation))
        if np.any(folded > 1.0e-6) or reversed_[index]:
            continue
        keep.append(rotations[index])
    if not keep:
        return np.eye(3, dtype=int)[None, :, :]
    return np.array(keep, dtype=int)


def conventional_transform(cell) -> np.ndarray:
    """The matrix taking primitive Miller indices to conventional cubic ones.

    Elk's ``vhmat``, which it leaves to the user to write down. A primitive fcc
    or bcc cell indexes its reflections on the primitive reciprocal basis,
    where the reflections a crystallographer names -- silicon's (111), the
    forbidden (222) -- do not appear at all; they are indices on the
    conventional cubic axes of length ``alat``. ``H' = M H`` with
    ``M = (alat / 2 pi) B^T``, ``B`` the reciprocal basis in 1/bohr, is that
    change of setting, and it is the identity for a cell whose own axes are the
    conventional ones.

    It is a *labelling* and nothing else: no structure factor changes, and for
    a lattice whose conventional cell is not cubic the transform is not this
    one and should be passed explicitly.
    """
    from pypresso.units import TPI

    return (float(cell.alat) / TPI) * np.asarray(cell.bg, dtype=float).T


def h_vectors(cell, hmax: float, symmetries=None, reduce: bool = True,
              transform: np.ndarray | None = None) -> HVectors:
    """Every reciprocal lattice vector with ``|H| <= hmax``, sorted by length.

    Elk's ``genhvec``. The set is enumerated over the box of Miller indices the
    cutoff can reach, sorted by ``|H|``, and -- with ``reduce`` -- collapsed
    onto one representative per star of the symmorphic point operations, each
    carrying the multiplicity of the star it stands for.

    Args:
        cell: the crystal's :class:`~pypresso.system.cell.Cell`.
        hmax: the cutoff on ``|H|`` in 1/bohr (Elk's ``hmaxvr``, default 6).
        symmetries: the crystal's group; ``None`` reduces nothing.
        reduce: Elk's ``reduceh``.
        transform: the output index transformation, Elk's ``vhmat``; ``None``
            leaves the primitive indices alone.
    """
    if hmax <= 0.0:
        raise ValueError(f"hmax must be positive, got {hmax}")
    at = np.asarray(cell.at, dtype=float)
    bg = np.asarray(cell.bg, dtype=float)
    # |H . a_i| <= |H| |a_i| and H . a_i = 2 pi m_i, so this bound is exact.
    bounds = np.floor(hmax * np.linalg.norm(at, axis=1) / (2.0 * np.pi) + 1.0e-9)
    ranges = [np.arange(-int(b), int(b) + 1) for b in bounds]
    grid = np.meshgrid(*ranges, indexing="ij")
    miller = np.stack([axis.ravel() for axis in grid], axis=1).astype(int)
    cartesian = miller @ bg
    length = np.linalg.norm(cartesian, axis=1)
    inside = length <= hmax + 1.0e-9
    miller, cartesian, length = miller[inside], cartesian[inside], length[inside]
    order = np.argsort(length, kind="stable")
    miller, cartesian, length = miller[order], cartesian[order], length[order]

    rotations = symmorphic_rotations(symmetries) if reduce else None
    if rotations is not None and len(rotations) > 1:
        miller, length, multiplicity = _collapse_stars(miller, length, rotations)
        cartesian = miller @ bg
    else:
        multiplicity = np.ones(len(miller), dtype=int)

    indices = (miller.astype(float) if transform is None
               else miller.astype(float) @ np.asarray(transform, dtype=float).T)
    return HVectors(miller=miller, cartesian=cartesian, length=length,
                    multiplicity=multiplicity, indices=indices)


def _collapse_stars(miller, length, rotations):
    """One representative per star, with the star's size, in Elk's order.

    ``genhvec`` walks the length-sorted list and, for each vector, looks for an
    image of it among the representatives kept so far; the first vector of a
    star is the one that stands for it. Miller indices transform as ``m' = R m``
    (:class:`~pypresso.system.symmetry.Symmetries`), so the image is a matrix
    product on the integer triple and the search is exact -- no tolerance is
    involved and none should be.
    """
    kept: dict[tuple[int, int, int], int] = {}
    representatives: list[np.ndarray] = []
    lengths: list[float] = []
    multiplicity: list[int] = []
    for row in range(len(miller)):
        vector = miller[row]
        images = np.einsum("sij,j->si", rotations, vector)
        found = -1
        for image in images:
            index = kept.get((int(image[0]), int(image[1]), int(image[2])), -1)
            if index >= 0:
                found = index
                break
        if found >= 0:
            multiplicity[found] += 1
            continue
        kept[(int(vector[0]), int(vector[1]), int(vector[2]))] = len(representatives)
        representatives.append(vector)
        lengths.append(float(length[row]))
        multiplicity.append(1)
    return (np.array(representatives, dtype=int), np.array(lengths, dtype=float),
            np.array(multiplicity, dtype=int))


def structure_factors_of_field(field, cell, miller, method: str = "fft"):
    """``F(H) = int f(r) e^{iH.r} d3r`` for a real field on the FFT grid.

    Args:
        field: ``(..., n1, n2, n3)`` real, sampled on the dense grid. Any
            leading axes are carried through, which is what lets the three
            components of a magnetization go in one call.
        cell: for the volume, which is the whole of the normalisation.
        miller: ``(nh, 3)`` integer indices of the reflections.
        method: ``"fft"`` transforms the whole box once and gathers; ``"direct"``
            is the definition, an explicit sum over grid points per reflection.
            They agree to round-off and the second is ``O(nh N)``; it is here
            because it shares no index arithmetic with the first, and index
            arithmetic is what a Fourier convention gets wrong.

    Returns ``(..., nh)`` complex, in whatever units ``field`` carried per
    bohr^3 -- electrons for a density, Bohr magnetons for a magnetization.
    """
    field = np.asarray(field)
    miller = np.asarray(miller, dtype=int)
    n1, n2, n3 = field.shape[-3:]
    reach = np.array([n1 // 2, n2 // 2, n3 // 2])
    if np.any(np.abs(miller) > reach):
        raise ValueError(
            "a reflection asks for a frequency the FFT grid does not carry: "
            f"|h| up to {np.abs(miller).max(axis=0)} against a grid of "
            f"({n1}, {n2}, {n3}). Lower hmax or raise ecutrho"
        )
    volume = float(cell.volume)
    if method == "direct":
        # The definition, summed over the grid: F = (Omega/N) sum_r rho(r)
        # e^{i2pi (h,k,l).s}, with s the fractional coordinate of the point.
        axes = [np.arange(n) / n for n in (n1, n2, n3)]
        phases = np.exp(2.0j * np.pi * (
            miller[:, 0, None, None, None] * axes[0][None, :, None, None]
            + miller[:, 1, None, None, None] * axes[1][None, None, :, None]
            + miller[:, 2, None, None, None] * axes[2][None, None, None, :]))
        weight = volume / (n1 * n2 * n3)
        return weight * np.einsum("...xyz,hxyz->...h", field, phases)
    if method != "fft":
        raise ValueError(f"unknown method {method!r}: use 'fft' or 'direct'")
    # The crystallographers' transform carries the *positive* phase and no
    # ``1/Omega``, which is why ``sfacrho.f90`` prints ``Omega * conj(zftrf)``:
    # ``zftrf`` is the physicists' ``(1/Omega) int rho e^{-iH.r}``. Conjugating
    # the coefficient is the same thing only because a density is real -- the
    # inverse transform *is* the positive-phase one, and is right either way.
    box = np.fft.ifftn(field, axes=(-3, -2, -1))
    gathered = box[..., miller[:, 0] % n1, miller[:, 1] % n2, miller[:, 2] % n3]
    return volume * gathered
