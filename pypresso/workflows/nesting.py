"""``run_nesting``: the Fermi-surface nesting function from a converged density.

``PLAN.md`` P52. One fixed-density run on a dense grid, then the correlation of
:mod:`pypresso.response.nesting`. The physics is in that module; what is here
is the part that is specific to a plane-wave run with symmetry:

**the eigenvalues have to live on the complete grid, and a wedge is unfolded
rather than re-diagonalised.** ``N(q)`` sums ``g(k) g(k + q)`` with ``k + q``
folded back onto the grid, so every one of the ``n1 n2 n3`` points needs a
value; but ``eps_n(Rk) = eps_n(k)`` exactly, so the irreducible wedge carries
all of them. :func:`~pypresso.system.kpoints.grid_equivalence` is the map --
``tetra.f90``'s ``equiv``, Elk's ``ivkik`` -- and using it turns a 24x24x24
nesting function on an fcc metal from 13824 diagonalisations into 413.

**The group it is unfolded with must be the group it was reduced with.** Those
are two different questions on a ``nosym``, ``noinv`` or magnetic run, and
getting them apart maps grid points onto the wrong representatives without
raising anything: what comes back is a smooth, positive, plausible ``N(q)``
built from somebody else's bands. Both sides go through
:func:`~pypresso.workflows.nscf.grid_symmetry` for that reason, and the check
that it works is the ``reduce = False`` route, which diagonalises the complete
grid and must agree to the eigensolver's own scatter.
"""

from __future__ import annotations

import numpy as np

from pypresso.response.nesting import (
    NestingFunction,
    nesting_from_eigenvalues,
    require_a_fermi_surface,
)
from pypresso.system.kpoints import KPoints, grid_equivalence
from pypresso.workflows.nscf import denser_grid, grid_symmetry, run_nscf

__all__ = ["run_nesting"]


def run_nesting(
    system,
    pseudos,
    density,
    *,
    grid: tuple[int, int, int] | None = None,
    shift: tuple[int, int, int] | None = None,
    nbnd: int | None = None,
    degauss: float | None = None,
    smearing: str | None = None,
    method: str = "fft",
    reduce: bool = True,
    fermi_energy: float | None = None,
    conv_thr: float = 1.0e-8,
    k_batch="default",
    ns=None,
    tau=None,
    becsum: tuple = (),
    field=None,
    field_scale: float | None = None,
) -> NestingFunction:
    """``N(q)`` for a converged metal.

    Args:
        system: the converged :class:`~pypresso.system.builder.System`.
        pseudos: its pseudopotentials.
        density: the converged density (``SCFResult.density``).
        grid: the Monkhorst-Pack divisions the Fermi surface is resolved on.
            It is the convergence parameter of the whole quantity and wants to
            be much denser than the density needed -- ``g(k)`` is a delta
            function on a surface. Defaults to the run's own grid, which is
            almost never enough.
        shift: the k-grid's shift; defaults to the run's. **The q-grid is
            always unshifted** whatever this is, because ``q`` is a difference
            of two k-points and the shift cancels in it.
        degauss: the delta's width in Ry. Defaults to the run's own
            ``degauss``, which is the width its Fermi level was found with.
        smearing: the delta's shape, a name from
            :data:`~pypresso.scf.occupations.SMEARING_ORDER`. Defaults to
            ``"gaussian"`` **even when the run used Methfessel-Paxton or cold
            smearing**: those go negative on the wings, and a product of two
            weights that can each be negative is a ``N(q)`` with no sign. Pass
            the run's own name to have it anyway.
        method: ``"fft"`` (the cyclic cross-correlation, one transform) or
            ``"direct"`` (``nesting.f90``'s double loop). They agree to
            round-off and the second is there to say so.
        reduce: unfold a symmetry-reduced wedge onto the complete grid instead
            of diagonalising all of it. ``False`` is the check on the map, and
            costs the size of the point group.
        fermi_energy: in Ry. Defaults to the level the *dense* run finds for
            itself, which is the right one -- a Fermi surface resolved on one
            grid and a level found on another do not have to be consistent, and
            ``nesting.f90`` reads the stored ground-state level instead
            (``readefm``).

    Returns:
        :class:`~pypresso.response.nesting.NestingFunction`.
    """
    from pypresso.scf.driver import Calculation

    calculation = Calculation(system, pseudos, k_batch=k_batch)
    require_a_fermi_surface(calculation)

    divisions = grid if grid is not None else system.kpoints.grid
    if divisions is None:
        raise ValueError(
            "the nesting function needs Monkhorst-Pack divisions and this run "
            "has none -- its k-points came from an explicit K_POINTS list, "
            "which carries no grid. Pass grid=(n1, n2, n3); it wants to be "
            "much denser than the density needed in any case"
        )
    divisions = tuple(int(n) for n in divisions)
    if any(n <= 0 for n in divisions):
        raise ValueError(f"the nesting grid must be positive, got {divisions}")
    if shift is None:
        shift = system.kpoints.shift or (0, 0, 0)
    shift = tuple(int(s) for s in shift)

    rotations, time_reversal, t_rev = grid_symmetry(system)
    if reduce and rotations is not None:
        kpoints = denser_grid(system, divisions, shift)
        equivalent = grid_equivalence(
            divisions, shift, rotations, time_reversal=time_reversal, t_rev=t_rev
        )
    else:
        # The complete grid, in ``monkhorst_pack`` order: ``equivalent`` is then
        # the identity and the unfold below is a no-op, which is the point --
        # the two routes differ only in how many diagonalisations they pay for.
        from pypresso.system.kpoints import for_spin

        kpoints = for_spin(
            KPoints.automatic(divisions, shift, system.cell,
                              precision=system.kpoints.precision),
            system.nspin,
        )
        equivalent = np.arange(int(np.prod(divisions)))

    nscf = run_nscf(system, pseudos, density, kpoints, nbnd, conv_thr, k_batch,
                    ns=ns, tau=tau, becsum=becsum, field=field,
                    field_scale=field_scale)

    level = fermi_energy
    if level is None:
        level = nscf.fermi_energy if nscf.fermi_energy is not None else nscf.homo
    if level is None:
        raise NotImplementedError(
            "the nesting function needs a Fermi level and the fixed-density "
            "run produced none"
        )

    width = degauss if degauss is not None else system.degauss
    if not width:
        raise ValueError(
            "the nesting function needs a delta width: the run carries no "
            "degauss, so pass one explicitly (in Ry)"
        )

    # ``(nspin, nk_irr, nbnd)`` -> ``(nspin, nk_full, nbnd)``. The unfold is the
    # whole of the symmetry saving and is exact: ``eps_n(Rk) = eps_n(k)``.
    eigenvalues = np.asarray(nscf.eigenvalues_by_spin)[:, equivalent, :]

    return nesting_from_eigenvalues(
        eigenvalues,
        divisions,
        level,
        float(width),
        smearing=smearing or "gaussian",
        degeneracy=2.0 if calculation.nspin == 1 else 1.0,
        method=method,
        cell_volume=float(system.cell.volume),
        nspin=calculation.nspin,
    )
