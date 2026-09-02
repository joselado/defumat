"""Non-self-consistent diagonalisation at a fixed density.

``PW/src/non_scf.f90``. The density comes from a converged SCF run, the
potential built from it is frozen, and the Hamiltonian is diagonalised wherever
the caller asks -- which is the same operation whether those k-points form a
band path (:mod:`defumat.workflows.bands`) or a denser uniform grid for a
density of states (:mod:`defumat.workflows.dos`). This module is that shared
core; the two workflows differ only in what they do with the eigenvalues
afterwards.

A denser grid is the whole reason a DOS is an NSCF run rather than a by-product
of the SCF: the density converges on a coarse grid, but a density of states
resolved to a few tens of meV needs an order of magnitude more k-points, and
paying for those inside the SCF loop would be waste.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from defumat.pseudo.upf import Pseudopotential
from defumat.scf.driver import Calculation, default_nbnd
from defumat.solvers.davidson import ETHR_MIN
from defumat.system.builder import System
from defumat.system.cell import Cell
from defumat.system.kpoints import KPoints, for_spin as kpoints_for_spin
from defumat.units import RY_TO_EV

__all__ = [
    "NSCFResult",
    "fixed_density_bands",
    "fixed_density_states",
    "run_nscf",
    "denser_grid",
    "grid_symmetry",
]


@dataclass
class NSCFResult:
    """Eigenvalues on a fixed density, with whatever occupation statistic applies.

    ``eigenvalues`` and ``occupations`` are ``(nk, nbnd)`` for an unpolarized run
    and ``(2, nk, nbnd)`` for LSDA -- the spin axis is squeezed away when there
    is only one channel, the same convention
    :class:`~defumat.scf.driver.SCFResult` uses, so that everything written
    against the unpolarized shape keeps working and a polarized result cannot be
    mistaken for one. :attr:`eigenvalues_by_spin` always has the axis.
    """

    kpoints: KPoints
    eigenvalues: np.ndarray  # (nk, nbnd) or (2, nk, nbnd), Ry
    occupations: np.ndarray | None = None  # same shape, QE's wg
    fermi_energy: float | None = None  # Ry
    homo: float | None = None  # Ry
    lumo: float | None = None  # Ry
    nspin: int = 1
    #: Only when ``tot_magnetization`` constrained the channels separately.
    fermi_energy_up: float | None = None
    fermi_energy_down: float | None = None

    @property
    def eigenvalues_ev(self) -> np.ndarray:
        return self.eigenvalues * RY_TO_EV

    @property
    def eigenvalues_by_spin(self) -> np.ndarray:
        """``(nspin, nk, nbnd)`` whatever ``nspin`` is."""
        return self.eigenvalues if self.nspin == 2 else self.eigenvalues[None]

    @property
    def occupations_by_spin(self) -> np.ndarray:
        return self.occupations if self.nspin == 2 else self.occupations[None]


def fixed_density_states(
    system: System,
    pseudos: tuple[Pseudopotential, ...],
    density: jnp.ndarray,
    kpoints: KPoints | None = None,
    nbnd: int | None = None,
    conv_thr: float = 1.0e-6,
    k_batch: int | None | str = "default",
    ns: jnp.ndarray | None = None,
    tau: jnp.ndarray | None = None,
    becsum: tuple = (),
    field=None,
    field_scale: float | None = None,
):
    """Diagonalise once at every k-point of ``system`` with ``density`` fixed.

    Returns ``(calculation, system, eigenvalues, wavefunctions)``: the caller
    usually needs the :class:`~defumat.scf.driver.Calculation` too, because the
    electron count and the symmetry group live on it and both are needed to turn
    eigenvalues into occupations. :func:`fixed_density_bands` is this without the
    wavefunctions, which is all a band structure or a density of states wants;
    a projection onto atomic orbitals is what needs them kept.

    The potential is built once from the given density and never updated -- that
    is the whole content of "non self-consistent".

    ``ns`` is the converged Hubbard occupation matrix (``SCFResult.ns``), needed
    for the same reason PAW's ``becsum`` is below: it is a property of the
    *wavefunctions*, so it cannot be rebuilt from the density this is handed,
    and the Hubbard potential is built from it.

    ``field`` and ``field_scale`` are the pair ``SCFResult.magnetic_field`` and
    ``SCFResult.field_scale``, and they are state rather than input for the same
    reason. A fresh :class:`~defumat.scf.driver.Calculation` rebuilds the field
    from the *input*'s ``B_field`` / ``constrained_magnetization``, and two
    schemes make that the wrong field: Elk's ``reducebf`` (manual 5.104) exists
    so a symmetry-breaking field can drive the SCF off the unpolarized solution
    and then be multiplied down towards zero -- after ~25 iterations at 0.9 it
    is 7% of its input value and the converged state is very nearly a field-free
    one -- and the fixed-spin-moment scheme drives its field from the moment's
    error, so its converged value is not in any input at all. Re-applying the
    input field to a band structure or a DOS afterwards shifts every eigenvalue
    by a Zeeman term the ground state does not have. **This has no ``pw.x``
    counterpart**: ``reducebf`` is Elk's, and QE's ``i_cons = 3`` field is
    likewise a converged quantity rather than a namelist variable.
    """
    if kpoints is not None:
        # **``for_spin`` at the boundary, and this is the one that gets missed.**
        # Every ``KPoints`` constructor applies the unpolarized spin degeneracy
        # unconditionally, because a k-set can be built long before it is known
        # which regime will use it; a spinor band holds *one* electron rather
        # than two. So a mesh a caller built with ``KPoints.automatic`` and
        # handed in here carries weights summing to 2 where a ``nspin = 2`` or
        # ``nspin = 4`` run needs 1, and nothing about that looks like an error
        # -- the electron count is still met and the Fermi level simply lands
        # somewhere else. ``denser_grid`` below has always done this for the
        # density of states; doing it here covers every caller that builds its
        # own set (:mod:`defumat.response.conductivity`,
        # :func:`~defumat.response.velocity.band_velocities`,
        # :mod:`defumat.response.effmass`). It is idempotent, so a set that
        # has already been normalised passes through untouched.
        system = eqx.tree_at(
            lambda s: s.kpoints, system, kpoints_for_spin(kpoints, system.nspin)
        )

    calculation = Calculation(system, pseudos, k_batch=k_batch)
    nbnd = nbnd or system.nbnd or default_nbnd(
        calculation.nelec,
        system.occupations,
        *((calculation.nelup, calculation.neldw) if system.nspin == 2 else (None, None)),
        noncolin=system.noncolin,
    )

    if calculation.is_paw and not becsum:
        # A PAW Hamiltonian's nonlocal coefficients are D^(0) + int V Q + ddd_paw,
        # and only the first two can be rebuilt from the density: ddd_paw comes
        # from ``becsum``, which is a property of the *wavefunctions* and is not
        # recoverable from the density this function is handed. Building the
        # Hamiltonian without it converges perfectly well and gives eigenvalues
        # that are wrong by tenths of an eV -- the failure mode this codebase
        # refuses rather than risks. ``SCFResult.becsum`` carries it, so passing
        # it is the fix, and the refusal now only catches *not* passing it.
        raise NotImplementedError(
            "a fixed-density run with a PAW pseudopotential needs the converged "
            "becsum as well as the density: pass becsum = scf_result.becsum. It "
            "cannot be rebuilt from the density, and leaving it out is wrong by "
            "tenths of an eV"
        )

    hubbard_terms = None
    if calculation.is_hubbard:
        if ns is None:
            raise ValueError(
                "a fixed-density run with a Hubbard U needs the converged "
                "occupation matrix as well as the density: pass ns = "
                "scf_result.ns. It cannot be rebuilt from the density, and "
                "leaving the term out gives eigenvalues that look plausible "
                "and are wrong by the whole Hubbard shift"
            )
        _, _, hubbard_terms = calculation.hubbard_terms(jnp.asarray(ns))

    if calculation.functional.is_meta and tau is None:
        # The same argument the PAW branch above makes, for the same kind of
        # quantity. ``tau`` is a property of the *occupied states over the whole
        # Brillouin zone* and this function is handed a k-set that is usually a
        # different one -- a band path has no occupations at all. It cannot be
        # rebuilt here and leaving it out is not an approximation but a
        # different functional.
        raise NotImplementedError(
            f"a fixed-density run under {calculation.functional.name} needs the "
            "converged kinetic energy density as well as the density: pass "
            "tau = scf_result.tau. It is a property of the occupied states over "
            "the whole zone and cannot be rebuilt from a band path"
        )
    if calculation.magnetic_field is not None and field is None:
        # The same argument as ``becsum`` and ``ns``, one field along: what the
        # run converged under is not what the input asked for. It is refused
        # rather than approximated because the difference is a rigid Zeeman
        # shift of every eigenvalue, which looks exactly like a band structure.
        raise ValueError(
            "a fixed-density run of a calculation with a magnetic field or a "
            "constrained moment needs the field the SCF ended with: pass "
            "field = scf_result.magnetic_field and field_scale = "
            "scf_result.field_scale. Rebuilding it from the input re-applies a "
            "field that reducebf or the fixed-spin-moment scheme had already "
            "changed, which shifts every eigenvalue and still looks like a band "
            "structure"
        )
    potential = calculation.potential(
        density,
        1.0 if field_scale is None else float(field_scale),
        field,
        tau=tau,
    )
    _, ddd_paw = (
        calculation.onecenter(becsum, None if tau is None else potential.meta_c)
        if becsum else (None, None)
    )
    hamiltonians = calculation.hamiltonian(potential.v_scf, ddd_paw, hubbard_terms)

    # There is no SCF here to tighten the threshold over, so ``setup.f90`` picks
    # one up front from the accuracy of the density the bands are computed in.
    ethr = max(ETHR_MIN, 0.1 * min(1.0e-2, conv_thr / max(1.0, calculation.nelec)))
    eigenvalues, wavefunctions = calculation.diagonalize(hamiltonians, nbnd, None, ethr)
    return calculation, system, np.asarray(eigenvalues), wavefunctions


def fixed_density_bands(*args, **kwargs):
    """:func:`fixed_density_states` without the wavefunctions.

    The wavefunctions are ``(nspin, nk, nbnd, npwx)`` complex -- the largest
    array a run holds -- so the caller that does not need them says so by
    calling this, and the buffer is free as soon as this returns.
    """
    calculation, system, eigenvalues, _ = fixed_density_states(*args, **kwargs)
    return calculation, system, eigenvalues


def run_nscf(
    system: System,
    pseudos: tuple[Pseudopotential, ...],
    density: jnp.ndarray,
    kpoints: KPoints | None = None,
    nbnd: int | None = None,
    conv_thr: float = 1.0e-6,
    k_batch: int | None | str = "default",
    ns: jnp.ndarray | None = None,
    tau: jnp.ndarray | None = None,
    becsum: tuple = (),
    field=None,
    field_scale: float | None = None,
) -> NSCFResult:
    """A full NSCF run: diagonalise, then occupy by the system's own scheme.

    ``system.occupations`` decides how -- fixed, smeared or tetrahedron -- so an
    input asking for ``occupations='tetrahedra'`` gets a tetrahedron Fermi level
    here exactly as it would from an SCF run, which is what makes the DOS of a
    metal consistent with the calculation that produced its density.
    """
    calculation, system, eigenvalues = fixed_density_bands(
        system, pseudos, density, kpoints, nbnd, conv_thr, k_batch, ns, tau,
        becsum, field, field_scale,
    )
    wg, levels = calculation.occupations(jnp.asarray(eigenvalues))
    nspin = calculation.nspin
    return NSCFResult(
        kpoints=system.kpoints,
        eigenvalues=eigenvalues if nspin == 2 else eigenvalues[0],
        occupations=np.asarray(wg if nspin == 2 else wg[0]),
        fermi_energy=levels.get("fermi_energy"),
        homo=levels.get("homo"),
        lumo=levels.get("lumo"),
        nspin=nspin,
        fermi_energy_up=levels.get("fermi_energy_up"),
        fermi_energy_down=levels.get("fermi_energy_down"),
    )


def denser_grid(
    system: System,
    grid: tuple[int, int, int],
    shift: tuple[int, int, int] | None = None,
    cell: Cell | None = None,
    rotations: np.ndarray | None = None,
) -> KPoints:
    """The same crystal's irreducible wedge on a finer Monkhorst-Pack grid.

    Convenience for "SCF on the input's grid, DOS on a denser one": the symmetry
    used to reduce it must be the crystal's, so it is taken from the system
    rather than rediscovered, and the shift defaults to the input's own.

    The weights go through :func:`defumat.system.kpoints.for_spin`, because
    every constructor applies the spin degeneracy unconditionally and an LSDA
    run wants it halved. Skipping that step counts every electron twice on the
    denser grid, which does not fail -- it moves the Fermi level and integrates
    to the right electron count at the wrong energy.

    **The symmetry is the run's, not the crystal's**, and that is a different
    thing three ways. This used to call ``find_symmetries`` and reduce with
    whatever the lattice and the basis allow, which reduces a grid the SCF ran
    unreduced:

    * ``nosym`` means ``nsym = 1`` (``setup.f90``), and a **spin spiral is
      required to be nosym** -- the spin space group is not written, so the
      crystal's operations are not the spiral's. Reducing with them folds
      k-points the run deliberately kept apart;
    * a **magnetic** noncollinear run has the smaller group
      ``magnetic_symmetries`` returns, the magnetization being an axial vector,
      and has no ``-k = k``: ``time_reversal`` is off (``magnetic_sym`` in
      ``setup.f90``);
    * ``noinv`` turns time reversal off on its own.

    These are the four lines of :meth:`defumat.system.builder.System.
    _recelled_kpoints`, and they are duplicated rather than shared because that
    method rebuilds *this* system's grid where this builds a denser one. An
    explicit ``rotations`` keeps its own meaning: the caller owns it, and it is
    used as given.
    """
    cell = cell if cell is not None else system.cell
    own_rotations, time_reversal, t_rev = grid_symmetry(system)
    if rotations is None:
        rotations = own_rotations
    else:
        # An explicit ``rotations`` keeps its own meaning: the caller owns it,
        # and it carries no ``t_rev`` of its own.
        t_rev = None
    if shift is None:
        shift = system.kpoints.shift or (0, 0, 0)
    return kpoints_for_spin(
        KPoints.automatic(
            tuple(int(n) for n in grid),
            tuple(int(s) for s in shift),
            cell,
            precision=system.kpoints.precision,
            rotations=rotations,
            time_reversal=time_reversal,
            t_rev=t_rev,
        ),
        system.nspin,
    )


def grid_symmetry(system: System):
    """``(rotations, time_reversal, t_rev)`` a denser grid of *this run* is reduced with.

    The four lines above, factored out for the one caller that needs them
    twice: :func:`~defumat.workflows.nesting.run_nesting` reduces a grid with
    :func:`denser_grid` and then has to **unfold** it again with
    :func:`~defumat.system.kpoints.grid_equivalence`, which walks the same
    orbits. If the two disagree about the group, every point of the complete
    grid is mapped to the wrong representative and nothing raises -- the
    eigenvalues are simply somebody else's. That is the P28a family of bug and
    the reason this is one function rather than two copies.

    ``rotations`` is ``None`` for a ``nosym`` run, which is what ``KPoints.
    automatic`` reads as "return the complete grid".
    """
    magnetic = system.nspin == 4 and system.domag
    symmetries = system.symmetry_group()
    rotations = None if system.nosym else symmetries.rotation_array()
    t_rev = None if system.nosym else symmetries.t_rev_array()
    return rotations, (not system.noinv and not magnetic), t_rev
