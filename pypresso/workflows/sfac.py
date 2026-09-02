"""``run_structure_factors``: X-ray and magnetic structure factors of a
converged density.

``PLAN.md`` P61, Elk's tasks 195/196. The physics is in
:mod:`pypresso.diffraction.structure_factor`; what is here is the part that is
specific to a plane-wave run -- which density is transformed, how far out the
grid can be believed, and Elk's ``wsfac`` energy window, which is what turns a
structure factor into a probe of bonding rather than of the whole density.

**The window rebuilds the density; it does not filter the answer.** Elk's
``sfacinit`` zeroes the occupation of every state outside ``wsfac`` and calls
``rhomag`` again, so the transform is of a different density rather than of a
piece of the same one. Here that is
:meth:`~pypresso.scf.driver.Calculation.density` with masked weights, which
carries the augmentation charge and the symmetrisation along with it: an
ultrasoft ``becsum`` is rebuilt from the same masked weights, so the
augmentation follows the window instead of being left behind at its
full-occupation value. A window and ``core = True`` together are legal and are
an odd object rather than an error: a *chosen range* of valence states with the
*whole* frozen core added back.

**On a magnetic run the reduction is the crystal's own group**, which for
``nspin_mag = 4`` is the magnetic one, and the operations that need time
reversal are dropped from it along with the non-symmorphic ones (Elk tests
``lspnsymc == 1`` for the same reason). That makes the reported representative
right; it does not make the star flat, because a magnetization is an axial
vector and its members are related by ``det(R) R``. Use ``reduce = False``
where every member is wanted.
"""

from __future__ import annotations

import numpy as np

from pypresso.basis.builder import build_basis
from pypresso.diffraction.structure_factor import (
    StructureFactors,
    conventional_transform,
    h_vectors,
    structure_factors_of_field,
)

__all__ = ["run_structure_factors"]


def run_structure_factors(
    system,
    pseudos,
    result,
    *,
    hmax: float = 6.0,
    reduce: bool = True,
    transform="conventional",
    window: tuple[float, float] | None = None,
    core: bool = False,
    method: str = "fft",
    k_batch="default",
) -> StructureFactors:
    """``F(H)`` for the charge and the magnetization of a converged run.

    Args:
        system: the run's :class:`~pypresso.system.builder.System`.
        pseudos: its pseudopotentials.
        result: the :class:`~pypresso.scf.driver.SCFResult` to transform. Only
            its density is needed unless ``window`` is set, which needs the
            wavefunctions and eigenvalues too.
        hmax: the cutoff on ``|H|`` in 1/bohr -- Elk's ``hmaxvr``, default 6.
            That default is past ``sqrt(ecutrho)`` for any run below
            ``ecutwfc = 9``, which the guard below refuses rather than
            quietly truncating: a low-cutoff cell has to ask for less.
        reduce: collapse each star onto one representative with a multiplicity
            (Elk's ``reduceh``).
        transform: the output index transformation, Elk's ``vhmat``.
            ``"conventional"`` reports the conventional cubic ``(h k l)`` of a
            primitive cubic Bravais lattice, which is the setting the
            reflections are named in; ``None`` leaves the primitive indices, or
            pass a 3x3 matrix.
        window: ``(lo, hi)`` in Ry, Elk's ``wsfac``: rebuild the density from
            the states inside it only. The eigenvalues are absolute, as QE
            prints them, so a window about the Fermi level is written with it.
        core: add the frozen core charge of a nonlinear core correction to the
            valence density before transforming. It is a *pseudised* partial
            core rather than the true one, so it moves the low-``|H|``
            reflections towards an all-electron value and cannot reach it.
        method: ``"fft"`` or the definition, ``"direct"``.

    Returns a :class:`~pypresso.diffraction.structure_factor.StructureFactors`.
    """
    density = np.asarray(result.density if window is None else
                         _windowed_density(system, pseudos, result, window, k_batch))
    if density.ndim != 4:
        raise ValueError(
            f"the density must be (nspin_mag, n1, n2, n3), got {density.shape}")

    cell = system.cell
    _require_a_representable_cutoff(system, hmax)
    matrix = (conventional_transform(cell) if transform == "conventional"
              else transform)
    vectors = h_vectors(cell, hmax, symmetries=system.symmetry_group(),
                        reduce=reduce, transform=matrix)

    charge = density[0] if density.shape[0] != 2 else density[0] + density[1]
    if core:
        charge = charge + _core_density(system, pseudos)
    factors = structure_factors_of_field(charge, cell, vectors.miller, method=method)

    magnetization = _magnetization_field(density)
    magnetic = (None if magnetization is None else
                structure_factors_of_field(magnetization, cell, vectors.miller,
                                           method=method).T)
    return StructureFactors(
        vectors=vectors,
        charge=np.asarray(factors),
        magnetization=None if magnetic is None else np.asarray(magnetic),
        window=None if window is None else (float(window[0]), float(window[1])),
        core_included=bool(core),
    )


def _magnetization_field(density):
    """``m(r)`` from a density carrying its channel axis, or ``None``.

    Two regimes and one number apiece: a collinear run stores the two channels,
    whose difference is the single component ``m_z``; a noncollinear one stores
    ``(n, m_x, m_y, m_z)`` already. A nonmagnetic run has no magnetization at
    all -- and that includes a *spin-orbit* run without one, where ``nspin`` is
    4 and ``nspin_mag`` is 1.
    """
    if density.shape[0] == 2:
        return (density[0] - density[1])[None, ...]
    if density.shape[0] == 4:
        return density[1:]
    return None


def _core_density(system, pseudos):
    """The nonlinear core correction on the dense grid, in electrons/bohr^3."""
    from pypresso.basis.fft import g_to_r
    from pypresso.pseudo.potentials import core_charge

    dense = build_basis(system).dense
    charge_g = core_charge(pseudos, system.structure, system.cell, dense)
    if charge_g is None:
        raise ValueError(
            "core = True, but no pseudopotential in this run has a nonlinear "
            "core correction: there is no core charge to add. The valence "
            "structure factor is what this dataset can produce"
        )
    return np.real(np.asarray(g_to_r(charge_g, dense.fft_index, dense.grid)))


def _require_a_representable_cutoff(system, hmax: float) -> None:
    """``hmax`` must stay inside the sphere the density was built on.

    A transform of the density on the dense grid returns a coefficient for
    every frequency the box carries, and the box reaches past the ``ecutrho``
    sphere into its corners. Those coefficients are not Fourier components of
    the density: they are what aliasing left there, and they are small, smooth
    and entirely plausible. This is the one guard that cannot be replaced by a
    check on the answer.
    """
    reach = float(np.sqrt(system.ecutrho))
    if hmax > reach + 1.0e-9:
        raise ValueError(
            f"hmax = {hmax} 1/bohr is beyond the density's own cutoff, "
            f"sqrt(ecutrho) = {reach:.3f} 1/bohr: past it the grid carries "
            "aliasing rather than Fourier coefficients of the density. Raise "
            "ecutrho if the reflections are wanted"
        )


def _windowed_density(system, pseudos, result, window, k_batch):
    """The density of the states inside ``window`` -- Elk's ``wsfac``."""
    from pypresso.scf.driver import Calculation

    if result.wavefunctions is None:
        raise ValueError(
            "an energy window needs the wavefunctions, and this result carries "
            "none: run the SCF without discarding them"
        )
    lo, hi = float(window[0]), float(window[1])
    if hi <= lo:
        raise ValueError(f"the window must be (lo, hi) with hi > lo, got {window}")

    eigenvalues = np.asarray(result.eigenvalues)
    weights = np.asarray(result.occupations)
    if eigenvalues.ndim == 2:  # the squeezed nspin = 1 (or spinor) shape
        eigenvalues = eigenvalues[None, ...]
        weights = weights[None, ...]
    inside = (eigenvalues >= lo) & (eigenvalues <= hi)
    if not inside.any():
        raise ValueError(
            f"no state has an eigenvalue in [{lo}, {hi}] Ry: the window is "
            "empty and the density would be zero"
        )
    calculation = Calculation(system, pseudos, k_batch=k_batch)
    return calculation.density(result.wavefunctions, weights * inside)
