"""Topological invariants of a converged calculation.

The shape of the other fixed-density workflows (:mod:`defumat.workflows.bands`,
:mod:`defumat.workflows.dos`): an SCF run produces a density, the potential
built from it is frozen, and the Hamiltonian is diagonalised wherever the
quantity asks for it. What is different here is *where* it asks, and how often:
a Chern number wants a closed mesh over a plane, a Wilson loop wants a whole
half-zone one loop at a time, and the Fu-Kane parity criterion wants four or
eight points and nothing else.

    ``run_berry_curvature``   Omega(k) and the Chern number on a plane
    ``run_z2``                the 2D Z2 of one plane
    ``run_z2_3d``             the four indices (nu0; nu1 nu2 nu3)

**Spin-orbit coupling is a precondition, not an option.** Without it every band
is doubly degenerate in spin, the two copies carry opposite Chern numbers, the
Wannier centres cross in pairs and every Z2 is trivially zero -- a number that
is correct, uninformative, and indistinguishable from a real answer. So a Z2 run
on a calculation with ``noncolin = .false.`` is refused. A Chern number is a
different matter: it needs *broken* time reversal, which this code can only
reach through a magnetization, so a nonmagnetic run's Chern number is zero by
symmetry and computing it is a check of the machinery rather than physics.

**Memory** (the standing rule, and this is the workflow where it bites). The
states of one loop are ``nloop * nocc * npol * npwx * 16`` bytes and are dropped
before the next loop is built, so ``npump`` costs time and not space. On the
committed bismuthene reference -- ``npwx = 2705``, ``npol = 2``, ``nocc = 30`` --
that is 2.6 MB per k-point and 63 MB for a 24-point loop. What dominates instead
is the setup each call rebuilds: the augmentation charge ``Q_ij(G)`` on the
dense grid is ``nh^2 * ngm`` complex, 1.1 GB for that cell, which is why the
source holds **one** :class:`~defumat.scf.driver.Calculation` at a time and why
a plane mesh is walked rather than materialised.
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
from defumat.system.kpoints import KPoints
from defumat.topology.berry import BerryCurvature
from defumat.topology.invariants import chern_number as _chern_number
from defumat.topology.invariants import z2_invariant, z2_invariant_3d
from defumat.topology.parity import inversion_centre
from defumat.topology.states import build_plane_wave_states

__all__ = ["DFTSource", "run_berry_curvature", "run_z2", "run_z2_3d"]


@dataclass
class DFTSource:
    """A state source: Kohn-Sham states anywhere in the zone, density fixed.

    Each call freezes the potential from ``density`` at the k-points asked for
    and diagonalises. The setup those k-points do *not* affect -- both G-vector
    sets, the local potential, the augmentation charge, the Ewald sum, the
    symmetry group, the radial tables -- is built once and shared between calls
    through :meth:`~defumat.scf.driver.Calculation.at_kpoints`; only the
    plane-wave spheres, ``|k+G|^2``, the stick layout and ``vkb(k)`` are rebuilt.

    That is what makes streaming the cheap option rather than the expensive one.
    Rebuilding a whole ``Calculation`` per call cost ~1 GB and seconds each time,
    which on any mesh worth taking is more than the states of the entire mesh
    cost to hold -- so the caller was forced to choose between a streaming loop
    that was ruinous in time and a resident mesh that was ruinous in memory.
    Sharing the k-independent setup removes the choice: streaming now costs one
    diagonalisation per call and nothing else.

    ``nocc`` is the number of occupied bands and is **explicit**: every quantity
    here is a property of a gapped manifold, so which bands are in it has to be
    stated rather than inferred from an occupation that a smearing has blurred.
    ``nbnd`` adds empty bands for the eigensolver's benefit only.
    """

    system: System
    pseudos: tuple[Pseudopotential, ...]
    density: jnp.ndarray
    nocc: int
    #: The converged ``becsum`` (``SCFResult.becsum``), needed by a PAW dataset
    #: and ignored by any other. A PAW Hamiltonian's nonlocal coefficients are
    #: ``D^(0) + int V Q + ddd_paw``, and only the first two can be rebuilt from
    #: the density: ``ddd_paw`` comes from ``becsum``, which is a property of the
    #: *wavefunctions*. Leaving it out is the same tenths-of-an-eV error
    #: :func:`~defumat.workflows.nscf.fixed_density_states` refuses, and here it
    #: would move the invariant only by moving a band across the manifold's gap
    #: -- silently, since the integer would still be an integer.
    becsum: tuple = ()
    #: The converged Hubbard occupation matrix (``SCFResult.ns``), needed for
    #: the same reason ``becsum`` is: ``v_ns`` is built from it and it is a
    #: property of the *wavefunctions*, so it cannot be rebuilt from the
    #: density. Without it the manifold is diagonalised on the **un-Hubbard**
    #: Hamiltonian -- and then :meth:`_check_gap` certifies it isolated using
    #: those eigenvalues, so the invariant is a confident integer taken off the
    #: wrong bands.
    ns: jnp.ndarray | None = None
    #: ``SCFResult.magnetic_field`` and ``SCFResult.field_scale``, needed for
    #: the same reason ``becsum`` and ``ns`` are: what the run *converged*
    #: under is not what the input asked for. Elk's ``reducebf`` scales a
    #: symmetry-breaking field down as the SCF runs, and the fixed-spin-moment
    #: scheme drives its field from the moment's error, so rebuilding the field
    #: from the input re-applies one the SCF had already changed -- a rigid
    #: Zeeman shift of every eigenvalue, which is invisible in an invariant
    #: because the answer is still an integer.
    #: :func:`~defumat.workflows.nscf.fixed_density_states` has refused this
    #: since the 2026-08-29 sweep and this class did not, which made it the same
    #: defect one layer over.
    field: object = None
    field_scale: float | None = None
    nbnd: int | None = None
    conv_thr: float = 1.0e-8
    k_batch: int | None | str = "default"
    #: Smallest direct gap above the manifold, in Ry, that is still trusted.
    #: A Wilson loop or a parity product taken across a gap the mesh cannot see
    #: returns a confident integer that is simply wrong -- the failure mode the
    #: reference implementation records twice (``elkpy``: graphene with a narrow
    #: anticrossing, bulk Bi2Se3 on a coarse loop). Checking it costs nothing:
    #: the eigenvalues are already computed, since the eigensolver is asked for
    #: empty bands anyway.
    gap_tol: float = 1.0e-4

    def __post_init__(self):
        if self.system.nspin == 2:
            raise NotImplementedError(
                "a collinear spin-polarized calculation has two independent "
                "Hamiltonians and therefore two independent sets of invariants; "
                "which is meant is not defined here. Run the noncollinear "
                "spinor path instead, where there is one Hamiltonian"
            )

    def _base(self) -> Calculation:
        """The k-independent setup, built once and kept.

        Cached on the instance rather than in ``__post_init__`` so that
        constructing a source stays free, and so that a source that is never
        asked for states never pays for one. It is built at the system's own
        k-points -- whatever they are, they are replaced by
        :meth:`~defumat.scf.driver.Calculation.at_kpoints` before anything
        k-dependent is used.
        """
        if getattr(self, "_calculation", None) is None:
            object.__setattr__(
                self, "_calculation",
                Calculation(self.system, self.pseudos, k_batch=self.k_batch),
            )
        return self._calculation

    def _ddd_paw(self):
        """PAW's one-centre coefficients, built once and kept.

        A function of ``becsum`` alone, so it is k-independent and belongs with
        the rest of the shared setup rather than inside the per-call
        diagonalisation: a streaming Wilson loop calls :meth:`states` once per
        k-point, and rebuilding the radial one-centre terms there would put a
        Gauss-Legendre quadrature over every sphere in the inner loop.
        """
        if not self.becsum:
            return None
        if getattr(self, "_ddd", None) is None:
            _, ddd = self._base().onecenter(self.becsum)
            object.__setattr__(self, "_ddd", ddd)
        return self._ddd

    def states(self, points, keep_projectors: bool = False,
               keep_velocity: bool = False, keep_hamiltonian: bool = False):
        """Occupied states at the given crystal k-points, in the given order.

        ``keep_hamiltonian`` keeps the fixed-density Hamiltonian itself, which
        an orbital magnetization applies between two k-derivatives. It costs
        nothing -- the object was built to diagonalise with and is otherwise
        dropped.

        ``keep_velocity`` additionally builds the
        :class:`~defumat.response.velocity.VelocityOperator` of *this*
        fixed-density Hamiltonian and keeps the whole diagonalised band set --
        what the ``kubo`` Berry curvature is a sum over. It costs the empty
        bands' coefficients (the manifold alone is what everything else here
        needs) and nothing else: the operator holds the frozen potential and
        the shared setup by reference and does no work until it is applied.
        """
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        kpoints = KPoints.from_crystal(
            points,
            np.full(len(points), 1.0 / len(points)),
            self.system.cell,
            precision=self.system.kpoints.precision,
        )
        calculation = self._base().at_kpoints(kpoints)
        system = calculation.system
        if calculation.is_paw and not self.becsum:
            # The same refusal ``workflows.nscf`` makes, and for the same
            # reason: ``ddd_paw`` is a function of ``becsum`` and ``becsum`` is
            # a property of the wavefunctions, so it cannot be rebuilt from the
            # density this source is handed. It is passed rather than derived,
            # and the refusal now only catches *not* passing it.
            raise NotImplementedError(
                "a fixed-density run with a PAW pseudopotential needs the "
                "converged becsum as well as the density: pass "
                "becsum = scf_result.becsum. It cannot be rebuilt from the "
                "density, and leaving it out is wrong by tenths of an eV"
            )
        hubbard_terms = None
        if calculation.is_hubbard:
            if self.ns is None:
                # ``workflows.nscf``'s refusal again, and it had no counterpart
                # here at all: ``hamiltonian`` was called with two positional
                # arguments into a signature whose third is ``hubbard=None``,
                # so a DFT+U crystal was diagonalised without its Hubbard term
                # and the invariant came off eigenvalues wrong by the whole
                # Hubbard shift.
                raise ValueError(
                    "a topological invariant of a DFT+U calculation needs the "
                    "converged occupation matrix as well as the density: pass "
                    "ns = scf_result.ns. It cannot be rebuilt from the density, "
                    "and leaving the term out gives eigenvalues that look "
                    "plausible, are wrong by the whole Hubbard shift, and still "
                    "produce an integer"
                )
            _, _, hubbard_terms = calculation.hubbard_terms(jnp.asarray(self.ns))
        if calculation.functional.is_meta:
            # Refused at the boundary rather than threaded. ``tau`` could be
            # carried the way ``ns`` and ``becsum`` are, and then a Tran-Blaha
            # invariant would run -- unvalidated, on a functional whose total
            # energy is not the value of anything it minimised. The refusal is
            # the honest state of it; what it replaces is ``v_of_rho`` raising
            # about a missing keyword deep inside the potential builder.
            raise NotImplementedError(
                f"a topological invariant under {calculation.functional.name} "
                "is not implemented: the functional is a potential rather than "
                "the derivative of an energy, and the combination is "
                "unvalidated. The band structure and the density of states are "
                "unaffected"
            )
        nbnd = self.nbnd or max(
            self.nocc + 2,
            default_nbnd(
                calculation.nelec, system.occupations, None, None,
                noncolin=system.noncolin,
            ),
        )
        if calculation.magnetic_field is not None and self.field is None:
            # ``fixed_density_states``' refusal, word for word, because it is
            # the same mistake: the converged field is state, not input.
            raise ValueError(
                "a fixed-density run of a calculation with a magnetic field or "
                "a constrained moment needs the field the SCF ended with: pass "
                "field = scf_result.magnetic_field and field_scale = "
                "scf_result.field_scale. Rebuilding it from the input "
                "re-applies a field that reducebf or the fixed-spin-moment "
                "scheme had already changed, which shifts every eigenvalue and "
                "still leaves an invariant looking like an integer"
            )
        potential = calculation.potential(
            self.density,
            1.0 if self.field_scale is None else float(self.field_scale),
            self.field,
        )
        hamiltonians = calculation.hamiltonian(
            potential.v_scf, self._ddd_paw(), hubbard_terms
        )
        ethr = max(
            ETHR_MIN,
            0.1 * min(1.0e-2, self.conv_thr / max(1.0, calculation.nelec)),
        )
        eigenvalues, wavefunctions = calculation.diagonalize(
            hamiltonians, nbnd, None, ethr
        )
        self._check_gap(np.asarray(eigenvalues[0]), points)
        velocity = None
        if keep_velocity:
            from defumat.response.velocity import VelocityOperator

            # The same three things ``calculation.hamiltonian`` was just given,
            # because ``dH/dk`` is the derivative of *that* Hamiltonian and of
            # no other: the frozen potential, PAW's one-centre coefficients and
            # the Hubbard occupation matrix.
            velocity = VelocityOperator(
                calculation, potential.v_scf, self._ddd_paw(), self.ns
            )
        return build_plane_wave_states(
            calculation,
            wavefunctions[0],
            nbnd=self.nocc,
            keep_projectors=keep_projectors,
            energies=eigenvalues[0],
            velocity=velocity,
            hamiltonian=hamiltonians[0] if keep_hamiltonian else None,
        )

    def _check_gap(self, eigenvalues: np.ndarray, points: np.ndarray) -> None:
        """Refuse a manifold that is not separated from the bands above it.

        ``(nk, nbnd)`` eigenvalues; the manifold is the lowest ``nocc``. Every
        invariant here is a property of an *isolated* set of bands, and the one
        way to get a wrong integer that still looks like an integer is to take
        it across a touching. Since the eigensolver has already produced the
        first empty band, this costs a subtraction.
        """
        if eigenvalues.shape[1] <= self.nocc:
            return
        gaps = eigenvalues[:, self.nocc] - eigenvalues[:, self.nocc - 1]
        worst = int(np.argmin(gaps))
        if gaps[worst] < self.gap_tol:
            raise ValueError(
                f"the occupied manifold is not isolated: bands {self.nocc} and "
                f"{self.nocc + 1} are {gaps[worst]:.3e} Ry apart at k = "
                f"{np.round(points[worst], 4).tolist()} (crystal), below the "
                f"gap_tol of {self.gap_tol:.1e} Ry. A topological invariant of a "
                "manifold that touches the one above it is not defined; choose "
                "a different band count, or raise gap_tol if the touching is "
                "known to be avoided"
            )


def _source(system, pseudos, density, nocc, nbnd, conv_thr, k_batch,
            becsum=(), ns=None, field=None, field_scale=None) -> DFTSource:
    if nocc is None:
        nocc = _occupied_bands(system, pseudos)
    return DFTSource(
        system=system,
        pseudos=tuple(pseudos),
        density=density,
        nocc=int(nocc),
        becsum=tuple(becsum or ()),
        ns=ns,
        field=field,
        field_scale=field_scale,
        nbnd=nbnd,
        conv_thr=conv_thr,
        k_batch=k_batch,
    )


def _occupied_bands(system: System, pseudos) -> int:
    """How many bands a band insulator fills: one electron each, or two.

    A spinor band holds one electron and a spin-degenerate band two, which is
    the same rule :meth:`defumat.workflows.bands.BandStructure.gap` uses.
    Getting it wrong halves or doubles the manifold, and every invariant here is
    a property of the manifold.
    """
    nelec = sum(
        float(pseudos[t].z_valence) for t in np.asarray(system.structure.types)
    )
    per_band = 1 if system.noncolin else 2
    count = nelec / per_band
    if abs(count - round(count)) > 1e-8:
        raise ValueError(
            f"{nelec} electrons do not fill a whole number of bands; a "
            "topological invariant needs a gapped manifold, so the occupied "
            "band count must be given explicitly"
        )
    return int(round(count))


def _require_spinors(system: System, what: str) -> None:
    if not system.noncolin:
        raise ValueError(
            f"{what} needs spin-orbit coupling: without it the bands are "
            "spin-degenerate, the two copies contribute opposite windings and "
            "the invariant is zero for a reason that has nothing to do with the "
            "band structure. Run with noncolin = .true. and lspinorb = .true. "
            "and a fully-relativistic pseudopotential"
        )


def run_berry_curvature(
    system: System,
    pseudos,
    density: jnp.ndarray,
    shape=(12, 12),
    axis: int = 2,
    offset: float = 0.0,
    nocc: int | None = None,
    nbnd: int | None = None,
    method: str | None = None,
    conv_thr: float = 1.0e-8,
    k_batch: int | None | str = "default",
    becsum: tuple = (),
    ns: jnp.ndarray | None = None,
    field=None,
    field_scale: float | None = None,
    **kwargs,
) -> BerryCurvature:
    """Berry curvature and the Chern number on one plane of the zone.

    Args:
        shape: the plaquette mesh, ``(n1, n2)`` points along the two crystal
            directions that span the plane.
        axis: the crystal direction held fixed at ``offset``.
        nocc: how many bands are occupied. Defaults to the electron count
            divided by one or two according to whether a band is a spinor.
        method: ``"fhs"`` (default) or ``"kubo"``; see
            :mod:`defumat.topology.berry`. ``"kubo"`` gives the pointwise
            curvature and reports the truncation of its sum over empty states;
            ``"fhs"`` is the only one a Chern *number* should be read from.
        kwargs: forwarded to the curvature method -- ``degeneracy_tol`` for
            ``"kubo"``.
    """
    source = _source(system, pseudos, density, nocc, nbnd, conv_thr, k_batch,
                     becsum=becsum, ns=ns,
                     field=field, field_scale=field_scale)
    return _chern_number(
        source, shape=shape, axis=axis, offset=offset, method=method,
        k_batch=k_batch, **kwargs,
    )


def run_z2(
    system: System,
    pseudos,
    density: jnp.ndarray,
    axis: int = 2,
    offset: float = 0.0,
    nocc: int | None = None,
    nbnd: int | None = None,
    method: str | None = None,
    nloop: int = 24,
    npump: int = 13,
    stream: bool = True,
    conv_thr: float = 1.0e-8,
    k_batch: int | None | str = "default",
    becsum: tuple = (),
    ns: jnp.ndarray | None = None,
    field=None,
    field_scale: float | None = None,
):
    """The 2D Z2 invariant of one plane of the zone.

    ``method="wilson"`` (the default) sweeps the Wannier charge centres over the
    half zone and needs only time-reversal symmetry. ``method="parity"`` takes
    the Fu-Kane product over the four TRIM of the plane, which costs four
    diagonalisations instead of ``nloop * npump`` and needs an inversion centre
    -- found here from the space group, and refused if there is none.

    Running both on a crystal that has an inversion centre is the check worth
    making: they share no machinery beyond the state set.
    """
    _require_spinors(system, "the Z2 invariant")
    source = _source(system, pseudos, density, nocc, nbnd, conv_thr, k_batch,
                     becsum=becsum, ns=ns,
                     field=field, field_scale=field_scale)
    kwargs = dict(axis=axis, offset=offset)
    if (method or "wilson").lower() == "parity":
        kwargs.update(dimension=2, centre=_centre(system))
    else:
        kwargs.update(nloop=nloop, npump=npump, k_batch=k_batch, stream=stream)
    return z2_invariant(source, method=method, **kwargs)


def run_z2_3d(
    system: System,
    pseudos,
    density: jnp.ndarray,
    nocc: int | None = None,
    nbnd: int | None = None,
    method: str | None = None,
    nloop: int = 24,
    npump: int = 13,
    stream: bool = True,
    conv_thr: float = 1.0e-8,
    k_batch: int | None | str = "default",
    becsum: tuple = (),
    ns: jnp.ndarray | None = None,
    field=None,
    field_scale: float | None = None,
):
    """The four three-dimensional indices ``(nu0; nu1 nu2 nu3)``.

    By ``wilson``, six planes; by ``parity``, eight k-points. The weak indices
    are components on the *primitive* reciprocal basis of this cell and mean
    nothing without it -- they are not invariant under a change of cell.
    """
    _require_spinors(system, "the Z2 invariants")
    source = _source(system, pseudos, density, nocc, nbnd, conv_thr, k_batch,
                     becsum=becsum, ns=ns,
                     field=field, field_scale=field_scale)
    kwargs = {}
    if (method or "wilson").lower() == "parity":
        kwargs["centre"] = _centre(system)
    else:
        kwargs["stream"] = stream
    return z2_invariant_3d(
        source, method=method, nloop=nloop, npump=npump, k_batch=k_batch, **kwargs
    )


def _centre(system: System) -> np.ndarray:
    """The crystal's inversion centre, from its space group."""
    from defumat.system.symmetry import find_symmetries

    return inversion_centre(find_symmetries(system.cell, system.structure))
