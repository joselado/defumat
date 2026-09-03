"""The orbital magnetization of a converged calculation.

    ``run_orbital_magnetization``   M_orb in Bohr magnetons per cell

The physics is in :mod:`defumat.topology.orbital_magnetization`; this is the
fixed-density workflow around it, and it is the same shape as
:func:`~defumat.workflows.topology.run_berry_curvature`: an SCF produces a
density, the potential built from it is frozen, and the Hamiltonian is
diagonalised on a mesh. It shares that module's
:class:`~defumat.workflows.topology.DFTSource` rather than restating it, and
with it every refusal a fixed-density run already makes -- PAW without its
``becsum``, DFT+U without its ``ns``, a converged magnetic field that is not
the input one, a potential-only meta-GGA.

**The mesh is the whole zone and is built here, not reduced.** ``pw.x`` does
the same for ``lorbm`` (``kpoint_grid_efield`` and ``nosym = .TRUE.`` in
``setup.f90``), and the reason is not convenience: the quantity is an axial
vector assembled from k-derivatives, so a wedge would have to be unfolded with
the ``det(R)`` rule before the derivative rather than after it.

**What makes it nonzero.** Time reversal must be broken, so the calculation has
to be magnetic; and spin and orbital motion must be coupled, so it needs
spin-orbit coupling. Either alone gives zero -- a collinear magnet without
spin-orbit coupling has a real Hamiltonian in each channel and its orbital
moment is quenched, which is the same 1.7e-16 :mod:`defumat.projwfc` measures
for ``<L>``. Both are checked here rather than left to produce a confident zero.

**Memory.** The whole mesh's occupied states are resident:
``nk nbnd npol npwx * 16`` bytes, plus six ``(nk, npol npwx)`` integer gather
plans. A derivative needs both neighbours of every point, so unlike a Wilson
loop there is no streaming order that holds less than a plane; on the committed
iodine reference (27 k-points, 7 bands, 8829 plane waves) it is 53 MB.
"""

from __future__ import annotations

import jax.numpy as jnp

from defumat.system.builder import System
from defumat.topology.mesh import volume_mesh
from defumat.topology.orbital_magnetization import (
    OrbitalMagnetization,
    orbital_magnetization,
)
from defumat.workflows.topology import _source

__all__ = ["run_orbital_magnetization"]


def run_orbital_magnetization(
    system: System,
    pseudos,
    density: jnp.ndarray,
    divisions: tuple[int, int, int] | None = None,
    shift: tuple[int, int, int] = (0, 0, 0),
    nocc: int | None = None,
    nbnd: int | None = None,
    mu: float = 0.0,
    conv_thr: float = 1.0e-8,
    k_batch: int | None | str = "default",
    becsum: tuple = (),
    ns: jnp.ndarray | None = None,
    field=None,
    field_scale: float | None = None,
    gap_tol: float = 1.0e-4,
) -> OrbitalMagnetization:
    """``M_orb`` by the modern theory, in Bohr magnetons per cell.

    Args:
        divisions: the uniform grid ``(nk1, nk2, nk3)``. Defaults to the grid
            the calculation's own k-points were generated on. A direction with
            one division contributes no derivative, which is right for a slab
            and is reported in ``flat_directions``; two is refused, the two
            neighbours being the same k-point.
        shift: QE's ``k1, k2, k3`` half-step offsets.
        nocc: the occupied manifold. Defaults to the electron count, one
            electron per spinor band.
        mu: the chemical potential in Ry the total is quoted at. ``pw.x``
            prints at zero and so does this by default; the term it multiplies
            is reported as ``dm_dmu`` whatever is passed.
        gap_tol: the smallest gap above the manifold that is still trusted.

    Returns:
        :class:`~defumat.topology.orbital_magnetization.OrbitalMagnetization`
        -- ``lc`` and ``ic`` as ``pw.x`` splits them, ``total``, and the
        diagnostics.
    """
    if system.nspin == 2:
        raise NotImplementedError(
            "an orbital magnetization of a collinear spin-polarized run is not "
            "implemented: without spin-orbit coupling each channel's "
            "Hamiltonian is time-reversal symmetric and the answer is zero, "
            "and with it the two channels are not separate Hamiltonians at all. "
            "Run the noncollinear spinor path (noncolin = .true., "
            "lspinorb = .true.), which is what carries this quantity"
        )
    if not system.noncolin:
        raise ValueError(
            "an orbital magnetization needs spin-orbit coupling and a broken "
            "time reversal: without the coupling, spin and orbital motion are "
            "independent, a global spin rotation is free and the orbital moment "
            "is quenched identically. Run with noncolin = .true., "
            "lspinorb = .true. and a fully-relativistic pseudopotential"
        )
    if getattr(system, "spiral_q", None) is not None:
        raise NotImplementedError(
            "an orbital magnetization of a spin spiral is not implemented: the "
            "two spinor components live on different plane-wave spheres, so the "
            "overlap between neighbouring manifolds is not a single gather"
        )

    if any(p.is_ultrasoft or p.is_paw for p in pseudos):
        raise NotImplementedError(
            "an orbital magnetization of an ultrasoft or PAW calculation is not "
            "implemented, and pw.x refuses the same combination "
            "(setup.f90: 'Orbital Magnetization not implemented with "
            "USPP/PAW'). The overlap between two neighbouring manifolds would "
            "need q_ij(b), and the dual states built from it would have to be "
            "dual in the S metric -- terms no norm-conserving case can check"
        )

    source = _source(system, pseudos, density, nocc, nbnd, conv_thr, k_batch,
                     becsum=becsum, ns=ns, field=field, field_scale=field_scale)
    source.gap_tol = gap_tol
    mesh = volume_mesh(_divisions(system, divisions), shift)
    states = source.states(mesh.flat(), keep_hamiltonian=True)
    return orbital_magnetization(
        states, mesh, system.cell, mu=mu, k_batch=k_batch,
    )


def _divisions(system: System, divisions) -> tuple[int, int, int]:
    """The grid asked for, or the one the calculation's k-points came from."""
    if divisions is not None:
        return tuple(int(n) for n in divisions)
    grid = getattr(system.kpoints, "grid", None)
    if grid is None:
        raise ValueError(
            "this calculation's k-points were not generated on a uniform grid, "
            "so there is no default mesh for an orbital magnetization: pass "
            "divisions = (nk1, nk2, nk3)"
        )
    return tuple(int(n) for n in grid)
