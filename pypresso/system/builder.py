"""Turn a parsed ``pw.x`` input into the objects the rest of the code uses.

This is where QE's input conventions are interpreted -- which cell parameters
win when several are given, what a card's units mean, which defaults apply --
and it is deliberately the *only* place that knows them. Everything downstream
takes a :class:`System` and never looks at an input file again.

Defaults come from ``Modules/input_parameters.f90``; the rules for combining
``ibrav``/``celldm``/``A,B,C``/``CELL_PARAMETERS`` come from ``PW/src/input.f90``.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import numpy as np

from pypresso.config import DEFAULT_PRECISION, Precision
from pypresso.io.pwin import PwInput, fortran_float, read_pw_input
from pypresso.system.cell import Cell, celldm_from_abc
from pypresso.system.kpoints import (
    KPoints,
    expand_to_subgroup,
    for_spin as kpoints_for_spin,
)
from pypresso.system.structure import Species, Structure
from pypresso.system.symmetry import (
    Symmetries,
    find_symmetries,
    lattice_point_group,
    magnetic_symmetries,
)
from pypresso.units import ANGSTROM_TO_BOHR, RY_TO_EV
from pypresso.vdw.registry import canonical_vdw_corr

__all__ = ["System", "build_system", "system_from_file", "local_moments"]


class System(eqx.Module):
    """A cell, its atoms, and the k-points to sample -- the output of setup.

    Everything a phase beyond P1 needs about *what* is being calculated, with
    none of the input-file syntax left in it.
    """

    cell: Cell
    structure: Structure
    kpoints: KPoints
    ecutwfc: float = eqx.field(static=True)
    ecutrho: float = eqx.field(static=True)
    #: 1 for an unpolarized calculation, 2 for collinear LSDA, 4 for
    #: noncollinear. Static, because it is an array *rank* everywhere
    #: downstream, not a value.
    #:
    #: ``nspin`` alone does not fix any array shape once it is 4: QE keeps
    #: *three* numbers and so does this code (``set_spin_vars`` in
    #: ``Modules/noncol.f90``). :attr:`npol` is how many spinor components a
    #: wavefunction has, :attr:`nspin_mag` how many components the density and
    #: the potential have, and ``nspin`` only says which of the three regimes is
    #: in force. Collapsing them is the mistake that makes a nonmagnetic
    #: spin-orbit run allocate -- and symmetrise -- a magnetization it does not
    #: have.
    nspin: int = eqx.field(static=True, default=1)
    calculation: str = eqx.field(static=True, default="scf")
    nbnd: int | None = eqx.field(static=True, default=None)
    occupations: str = eqx.field(static=True, default="fixed")
    #: ``input_dft``: an exchange-correlation functional that overrides the one
    #: the pseudopotentials were generated with. ``None`` -- the normal case --
    #: means the pseudopotentials decide.
    input_dft: str | None = eqx.field(static=True, default=None)
    smearing: str = eqx.field(static=True, default="gaussian")
    degauss: float = eqx.field(static=True, default=0.0)
    #: Occupations read from an OCCUPATIONS card, for occupations='from_input'.
    #: One row per spin channel when nspin = 2 (``f_inp(:, isk(ik))``).
    input_occupations: tuple[float, ...] | None = eqx.field(static=True, default=None)
    #: ``starting_magnetization(i)``, per species, in [-1, 1]. It splits the
    #: superposition of atomic charges the SCF starts from -- and it is the only
    #: thing that does, so an LSDA run left at zero converges to the unpolarized
    #: solution whenever that is a stationary point, which for a symmetric
    #: crystal it always is.
    starting_magnetization: tuple[float, ...] = eqx.field(static=True, default=())
    #: ``tot_magnetization``: constrain ``N_up - N_dw`` instead of letting the
    #: two channels share one Fermi level. ``None`` -- QE's -10000 sentinel --
    #: means unconstrained.
    tot_magnetization: float | None = eqx.field(static=True, default=None)
    #: ``nosym``: use no symmetry at all. Not an optimisation switch -- an input
    #: whose occupations break the crystal's symmetry (an atom with one of its
    #: three p channels filled) needs it, and symmetrising anyway converges to a
    #: different state.
    nosym: bool = eqx.field(static=True, default=False)
    #: ``tstress`` (``&control``): compute the stress tensor once the SCF has
    #: converged. It is a property of the *run* rather than of the system, and
    #: it lives here for the same reason :attr:`calculation` does -- this is the
    #: object an input file becomes, and a switch nothing carries is a switch
    #: nothing can honour. ``run_scf`` reads it when its own ``tstress``
    #: argument is left at ``None`` (P11).
    tstress: bool = eqx.field(static=True, default=False)
    #: ``lspinorb``: use the ``j``-resolved projectors of a fully-relativistic
    #: pseudopotential, which is what puts spin-orbit coupling in the
    #: Hamiltonian. Requires ``noncolin`` -- QE refuses the combination too,
    #: because the spin-orbit term does not commute with ``S_z`` and there is no
    #: collinear Hamiltonian for it to enter.
    lspinorb: bool = eqx.field(static=True, default=False)
    #: ``angle1``/``angle2`` in degrees, per species: the polar and azimuthal
    #: angles of that species' starting magnetization. Only meaningful when
    #: ``noncolin`` -- a collinear run has nothing to point.
    angle1: tuple[float, ...] = eqx.field(static=True, default=())
    angle2: tuple[float, ...] = eqx.field(static=True, default=())
    #: ``constrained_magnetization`` and its ``lambda``/``fixed_magnetization``
    #: (``input.f90``'s ``i_cons``), plus the fields put in by hand. See
    #: :mod:`pypresso.scf.fields`.
    constrained_magnetization: str = eqx.field(static=True, default="none")
    constraint_lambda: float = eqx.field(static=True, default=1.0)
    fixed_magnetization: tuple[float, float, float] = eqx.field(
        static=True, default=(0.0, 0.0, 0.0)
    )
    #: ``B_field``: a uniform field over the cell, in Ry.
    b_field: tuple[float, float, float] = eqx.field(static=True, default=(0.0, 0.0, 0.0))
    #: A pypresso extension with no pw.x counterpart -- Elk's ``bfcmt``: one
    #: field per atom, applied inside that atom's sphere. ``()`` for none.
    atomic_b_field: tuple = eqx.field(static=True, default=())
    #: Elk's ``reducebf``: multiply the external fields by this after every SCF
    #: iteration, so a symmetry-breaking field is gone by convergence.
    reducebf: float = eqx.field(static=True, default=1.0)
    #: Which fixed-spin-moment update rule to use -- a pypresso extension with
    #: no pw.x counterpart, since the scheme itself has none. See
    #: :data:`pypresso.scf.fields.FSM_UPDATES`.
    fsm_update: str = eqx.field(static=True, default="secant")
    #: ``r_m`` per species in bohr, or ``()`` for the radius ``make_pointlists``
    #: derives. A pypresso extension only in being an input at all.
    integration_radii: tuple = eqx.field(static=True, default=())
    #: Which local-weight scheme the atom-resolved quantities use; see
    #: :mod:`pypresso.scf.locals`.
    local_weights: str = eqx.field(static=True, default="qe")
    #: ``spiral_q``: the wavevector of a spin spiral, in **lattice**
    #: coordinates, as Elk's ``vqlss`` is (P19). ``None`` -- the normal case --
    #: is no spiral. See :mod:`pypresso.system.spiral`.
    spiral_q: tuple | None = eqx.field(static=True, default=None)
    #: The ``HUBBARD`` card and its namelist companions (P20), or ``None`` for a
    #: run with no Hubbard correction. Static: it decides array shapes (how many
    #: projectors, how wide the occupation matrix) and which code paths run.
    hubbard: object = eqx.field(static=True, default=None)
    #: ``vdw_corr``: which van der Waals correction to add, canonicalised by
    #: :func:`~pypresso.vdw.registry.canonical_vdw_corr`. ``'none'`` -- the
    #: default -- is no correction. Only ``'grimme-d2'`` is implemented; the
    #: others are refused by name, since QE's ``set_vdw_corr`` merely warns and
    #: runs on without one (:mod:`pypresso.vdw`).
    vdw_corr: str = eqx.field(static=True, default="none")
    #: ``london_s6``: D2's global scaling factor.
    london_s6: float = eqx.field(static=True, default=0.75)
    #: ``london_rcut``: D2's real-space cutoff in bohr. The sum's truncation
    #: error falls only as ``1/rcut^3``, which is why QE's default is as large
    #: as it is (:mod:`pypresso.vdw.grimme`).
    london_rcut: float = eqx.field(static=True, default=200.0)
    #: ``london_c6`` and ``london_rvdw`` per species, overriding the tabulated
    #: values. ``()`` -- or QE's -1 sentinel in an entry -- means "use the
    #: table".
    london_c6: tuple[float, ...] = eqx.field(static=True, default=())
    london_rvdw: tuple[float, ...] = eqx.field(static=True, default=())

    @property
    def spiral(self) -> bool:
        return self.spiral_q is not None

    @property
    def lsda(self) -> bool:
        return self.nspin == 2

    @property
    def noncolin(self) -> bool:
        return self.nspin == 4

    @property
    def npol(self) -> int:
        """How many spinor components a wavefunction has: 2 noncollinear, 1 not.

        This is an array *dimension* of every wavefunction-shaped quantity, and
        the reason a noncollinear Hamiltonian is one operator on a space of
        ``npol * npwx`` rather than ``nspin`` operators on ``npwx``.
        """
        return 2 if self.noncolin else 1

    @property
    def domag(self) -> bool:
        """Whether the run carries a magnetization at all (``setup.f90``).

        For a noncollinear calculation this is decided by ``starting_magnetization``
        being nonzero *somewhere* and by nothing else: a spin-orbit run on a
        nonmagnetic crystal has spinor wavefunctions and a scalar density, and
        QE says so in the comment above the assignment -- "set the domag
        variable to make a spin-orbit calculation with zero magnetization".

        It is a property of the input rather than of the converged state, which
        is what makes it static: the magnetization cannot appear during the SCF
        if nothing in the starting guess breaks the symmetry.
        """
        if not self.noncolin:
            return False
        return any(abs(m) > 1.0e-6 for m in self.starting_magnetization)

    @property
    def nspin_mag(self) -> int:
        """Components of the density and the potential: 1, 2 or 4.

        4 only for a *magnetic* noncollinear run -- ``(n, m_x, m_y, m_z)``. A
        nonmagnetic spin-orbit run has one, exactly as an unpolarized run does,
        and every routine that builds, mixes, symmetrises or integrates a
        density then runs unchanged.
        """
        if self.noncolin:
            return 4 if self.domag else 1
        return self.nspin

    @property
    def local_moments(self) -> np.ndarray:
        """``m_loc``: each atom's starting moment, cartesian ``(nat, 3)``.

        The magnetic symmetry group is decided from this and from nothing else
        (:func:`local_moments`), so it lives on :class:`System` -- the driver and
        the k-point reduction must not each compute their own version of the
        rule, or they can disagree about which operations exist.
        """
        return local_moments(
            self.structure, self.nspin, self.starting_magnetization,
            self.angle1, self.angle2,
        )

    def with_spin(
        self,
        nspin: int | None = None,
        *,
        lspinorb: bool | None = None,
        starting_magnetization=None,
        angle1=None,
        angle2=None,
        nbnd: int | None = None,
    ) -> "System":
        """The same crystal in another spin regime, with its k-points rebuilt.

        What an input file would say differently if it asked for a collinear or
        a noncollinear run of this cell -- ``nspin``, ``lspinorb``,
        ``starting_magnetization``, ``angle1``/``angle2`` -- with everything the
        builder derives from them derived again. It exists because
        ``dataclasses.replace`` is *not* enough and the two ways it is wrong are
        both silent:

        * **The weights.** ``setup.f90`` multiplies them by ``degspin`` only in
          the unpolarized branch, so an unpolarized set handed to an ``nspin =
          2`` run counts every electron twice and the Fermi level comes out
          somewhere else (:func:`~pypresso.system.kpoints.for_spin`).
        * **The k-set itself.** A *magnetic* noncollinear run has a smaller
          symmetry group and no ``-k = k``, so it needs k-points the collinear
          run never had -- 22 where the input listed 11, in QE's own
          ``pw_noncolin`` benchmark. An automatic grid is reduced again with the
          target's group, and an explicit list goes through ``irreducible_BZ``'s
          expansion, exactly as :func:`build_system` does it.

        ``nbnd`` is doubled crossing into ``nspin = 4`` and halved coming back,
        because a spinor band holds one electron where a collinear band holds
        two; pass it explicitly to override. A band path is left alone -- its
        weights mean nothing and it is not what an SCF runs on.

        Everything else is carried over unchanged, including the fields and the
        constraints, so a regime change does not quietly drop them. The result
        goes through the same consistency checks the input does: ``lspinorb``
        without ``noncolin`` is refused here as it is there.
        """
        nspin = int(self.nspin if nspin is None else nspin)
        if nspin not in (1, 2, 4):
            raise ValueError(f"nspin = {nspin}: expected 1, 2 or 4")
        lspinorb = bool(self.lspinorb if lspinorb is None else lspinorb)
        if lspinorb and nspin != 4:
            raise ValueError(
                "lspinorb = .true. needs nspin = 4: the spin-orbit term couples "
                "the two spin channels, so it has no collinear form"
            )
        magnetization = tuple(
            self.starting_magnetization if starting_magnetization is None
            else np.asarray(starting_magnetization, dtype=float).ravel().tolist()
        )
        angle1 = tuple(self.angle1 if angle1 is None
                       else np.asarray(angle1, dtype=float).ravel().tolist())
        angle2 = tuple(self.angle2 if angle2 is None
                       else np.asarray(angle2, dtype=float).ravel().tolist())
        if nspin == 1 and any(abs(m) > 1.0e-6 for m in magnetization):
            raise ValueError(
                "nspin = 1 with a nonzero starting_magnetization: an unpolarized "
                "density has nothing for it to split, so it would be ignored "
                "rather than honoured"
            )
        if self.spiral and nspin != 4:
            raise ValueError(
                "a spin spiral is a noncollinear calculation by construction "
                "(the two spinor components live at k +- q/2); drop spiral_q "
                "before leaving nspin = 4"
            )
        if nbnd is None and self.nbnd is not None:
            # ``setup.f90``: a spinor band holds one electron, a collinear band
            # two, so the same physical states need twice as many.
            factor = (2 if nspin == 4 else 1) / (2 if self.nspin == 4 else 1)
            nbnd = int(round(self.nbnd * factor))

        # ``dataclasses.replace`` rather than ``eqx.tree_at``: most of what
        # changes here is a *static* field, which is not in the pytree at all.
        return dataclasses.replace(
            self,
            nspin=nspin,
            lspinorb=lspinorb,
            starting_magnetization=magnetization,
            angle1=angle1,
            angle2=angle2,
            nbnd=nbnd,
            kpoints=self._respin_kpoints(nspin, magnetization, angle1, angle2),
        )

    def _respin_kpoints(self, nspin, magnetization, angle1, angle2) -> KPoints:
        """The target regime's k-point set, reduced with *its* symmetry group."""
        kpoints = self.kpoints
        if kpoints.path_length is not None or kpoints.gamma_only:
            return kpoints_for_spin(kpoints, nspin)

        moments = local_moments(self.structure, nspin, magnetization, angle1, angle2)
        magnetic = nspin == 4 and bool(np.any(np.abs(moments) > 1.0e-6))
        symmetries = find_symmetries(self.cell, self.structure)
        if magnetic:
            symmetries = magnetic_symmetries(
                self.cell, self.structure, symmetries, moments
            )
        rotations = None if self.nosym else symmetries.rotation_array()
        t_rev = None if self.nosym else symmetries.t_rev_array()

        if kpoints.grid is not None:
            rebuilt = KPoints.automatic(
                kpoints.grid, kpoints.shift or (0, 0, 0), self.cell,
                precision=kpoints.precision, rotations=rotations,
                time_reversal=not magnetic, t_rev=t_rev,
            )
            return kpoints_for_spin(rebuilt, nspin)

        # An explicit list is the wedge of the *lattice's* group by QE's
        # convention, so the same completion ``irreducible_BZ`` performs on it
        # applies again with the target's -- and does nothing at all when the
        # target's group is the one it was already reduced with.
        points = np.asarray(kpoints.crystal(self.cell))
        weights = np.asarray(kpoints.weights)
        if rotations is not None and len(rotations):
            points, weights = expand_to_subgroup(
                points, weights,
                np.array(lattice_point_group(np.asarray(self.cell.at))),
                rotations, time_reversal=not magnetic, t_rev=t_rev,
            )
        rebuilt = KPoints.from_crystal(
            points, weights, self.cell, precision=kpoints.precision
        )
        return kpoints_for_spin(rebuilt, nspin)

    def symmetry_group(self, nosym: bool | None = None) -> Symmetries:
        """The space group this run symmetrises with -- magnetic if it is.

        ``find_sym`` is called with ``magnetic_sym = noncolin .AND. domag``
        (``setup.f90``), and that is the only place the distinction is made.
        """
        if nosym is None:
            nosym = self.nosym
        symmetries = find_symmetries(self.cell, self.structure)
        if self.nspin_mag == 4:
            symmetries = magnetic_symmetries(
                self.cell, self.structure, symmetries, self.local_moments
            )
        return symmetries


def system_from_file(path, precision: Precision = DEFAULT_PRECISION) -> System:
    return build_system(read_pw_input(path), precision=precision)


def build_system(pwin: PwInput, precision: Precision = DEFAULT_PRECISION) -> System:
    cell = _build_cell(pwin, precision)
    structure = _build_structure(pwin, cell, precision)

    # ``noncolin`` and ``nspin`` say the same thing twice in a pw.x input, and
    # ``input.f90`` resolves it in one direction: noncolin wins and sets
    # nspin = 4. An input that says both consistently is common; one that says
    # nspin = 2 *and* noncolin is not, and is refused rather than silently
    # resolved, since which one the author meant is not recoverable.
    noncolin = _logical(pwin.get("system", "noncolin", False))
    lspinorb = _logical(pwin.get("system", "lspinorb", False))
    nspin = int(pwin.get("system", "nspin", 1))
    if noncolin:
        if nspin == 2:
            raise ValueError(
                "noncolin = .true. together with nspin = 2: a noncollinear "
                "calculation is nspin = 4; drop one of the two"
            )
        nspin = 4
    if nspin not in (1, 2, 4):
        raise ValueError(f"nspin = {nspin}: expected 1, 2 or 4")
    if lspinorb and nspin != 4:
        # QE's own check (``input.f90``). Spin-orbit does not commute with S_z,
        # so there is no collinear Hamiltonian for it to enter -- asking for it
        # without noncolin is an input error, not a request to be approximated.
        raise ValueError(
            "lspinorb = .true. needs noncolin = .true.: the spin-orbit term "
            "couples the two spin channels, so it has no collinear form"
        )
    starting_magnetization = tuple(
        pwin.indexed("system", "starting_magnetization", structure.ntyp)
    )
    # Imported here rather than at module scope: ``pypresso.scf`` imports the
    # driver, which imports this module, and the cycle is real.
    from pypresso.scf.fields import CONSTRAINTS

    constrained_magnetization = str(
        pwin.get("system", "constrained_magnetization", "none")
    ).lower()
    if constrained_magnetization not in CONSTRAINTS:
        raise ValueError(
            f"constrained_magnetization = {constrained_magnetization!r} is not "
            f"implemented; available: {sorted(CONSTRAINTS)}"
        )
    if constrained_magnetization != "none" and nspin == 1:
        # QE's own check: there is no magnetization to constrain.
        raise ValueError(
            "constrained_magnetization requires nspin = 2 or noncolin = .true."
        )
    if any(float(v) != 0.0 for v in pwin.indexed("system", "b_field", 3)) and (
        constrained_magnetization != "none"
    ):
        # ``input.f90:1614``: QE refuses the combination rather than deciding
        # which of the two fields wins.
        raise ValueError(
            "a nonzero B_field together with constrained_magnetization: QE "
            "refuses this and so does pypresso; use one or the other"
        )
    angle1 = tuple(pwin.indexed("system", "angle1", structure.ntyp))
    angle2 = tuple(pwin.indexed("system", "angle2", structure.ntyp))

    # An automatic k-grid is reduced to its irreducible wedge here, which is
    # where QE does it too (``setup.f90``, after the symmetry analysis and
    # before anything is sized from the k-point count). It needs the crystal's
    # symmetries, hence the ordering: cell, then structure, then the spin
    # variables -- because a magnetic noncollinear run has a *smaller* group and
    # no ``-k = k`` (``magnetic_sym`` in ``setup.f90``) -- then symmetry, then
    # k-points.
    nosym = _logical(pwin.get("system", "nosym", False))
    # ``tstress`` and nothing else: there is no ``tprnstress`` in QE 7.5
    # (``INPUT_PW.txt`` lists one stress switch and a grep of the tree finds no
    # other spelling), so accepting an alias would be inventing input syntax.
    # ``input.f90``'s own rule is ``tstress_ = lmovecell .OR. (tstress .AND.
    # lscf)``: a variable-cell run turns it on whatever the input says, and a
    # non-self-consistent one turns it off. Neither branch is reachable here --
    # ``vc-relax`` is not implemented (P11) and an NSCF run does not come
    # through ``run_scf`` -- so what is read is the input's own value.
    tstress = _logical(pwin.get("control", "tstress", False))
    spiral_q = _spiral_q(pwin, nspin, lspinorb, nosym)
    moments = local_moments(structure, nspin, starting_magnetization, angle1, angle2)
    magnetic = nspin == 4 and bool(np.any(np.abs(moments) > 1.0e-6))
    symmetries = find_symmetries(cell, structure)
    if magnetic:
        symmetries = magnetic_symmetries(cell, structure, symmetries, moments)
    rotations = None if nosym else symmetries.rotation_array()
    kpoints = _build_kpoints(
        pwin, cell, precision, rotations,
        time_reversal=not magnetic,
        t_rev=None if nosym else symmetries.t_rev_array(),
        lattice_rotations=None if nosym else np.array(lattice_point_group(np.asarray(cell.at))),
    )

    # ``setup.f90``'s ``degspin``, applied in the one place that knows the rule
    # -- a k-set built later, for a denser DOS grid, has to go through the same
    # function or it counts every electron twice.
    kpoints = kpoints_for_spin(kpoints, nspin)

    ecutwfc = pwin.get("system", "ecutwfc")
    if ecutwfc is None:
        raise ValueError(f"{pwin.path or 'input'}: ecutwfc is required")
    # QE's default: the density cutoff is 4*ecutwfc, exact for norm-conserving
    # pseudopotentials (the density has twice the wavefunction's G range).
    ecutrho = pwin.get("system", "ecutrho") or 4.0 * float(ecutwfc)

    # ``vdw_corr``, with ``input.f90``'s obsolescent alias: ``london = .true.``
    # is the same thing as ``vdw_corr = 'grimme-d2'``, and QE still honours it.
    # It is read *after* ``vdw_corr`` and only when that was left at its default,
    # so an input that says both does not have the older spelling win.
    vdw_corr = canonical_vdw_corr(pwin.get("system", "vdw_corr", "none"))
    if vdw_corr == "none" and _logical(pwin.get("system", "london", False)):
        vdw_corr = "grimme-d2"
    london_c6 = tuple(pwin.indexed("system", "london_c6", structure.ntyp, default=-1.0))
    london_rvdw = tuple(
        pwin.indexed("system", "london_rvdw", structure.ntyp, default=-1.0)
    )

    return System(
        cell=cell,
        structure=structure,
        kpoints=kpoints,
        ecutwfc=float(ecutwfc),
        ecutrho=float(ecutrho),
        nspin=nspin,
        calculation=str(pwin.get("control", "calculation", "scf")).lower(),
        nbnd=pwin.get("system", "nbnd"),
        occupations=str(pwin.get("system", "occupations", "fixed")).lower(),
        input_dft=pwin.get("system", "input_dft"),
        smearing=str(pwin.get("system", "smearing", "gaussian")).lower(),
        degauss=float(pwin.get("system", "degauss", 0.0)),
        input_occupations=_input_occupations(pwin),
        starting_magnetization=starting_magnetization,
        tot_magnetization=_tot_magnetization(pwin),
        nosym=nosym,
        tstress=tstress,
        lspinorb=lspinorb,
        angle1=angle1,
        angle2=angle2,
        constrained_magnetization=constrained_magnetization,
        constraint_lambda=float(pwin.get("system", "lambda", 1.0)),
        fixed_magnetization=tuple(
            float(v) for v in pwin.indexed("system", "fixed_magnetization", 3)
        ),
        b_field=tuple(float(v) for v in pwin.indexed("system", "b_field", 3)),
        atomic_b_field=_atomic_b_field(pwin, structure.nat),
        reducebf=float(pwin.get("system", "reducebf", 1.0)),
        fsm_update=_fsm_update(pwin),
        integration_radii=tuple(
            float(v) for v in pwin.indexed("system", "r_m", structure.ntyp)
        ) if pwin.get("system", "r_m") is not None else (),
        local_weights=str(pwin.get("system", "local_weights", "qe")).lower(),
        hubbard=_hubbard(pwin, structure),
        spiral_q=spiral_q,
        vdw_corr=vdw_corr,
        london_s6=float(pwin.get("system", "london_s6", 0.75)),
        london_rcut=float(pwin.get("system", "london_rcut", 200.0)),
        london_c6=london_c6,
        london_rvdw=london_rvdw,
    )


def _logical(value) -> bool:
    """A Fortran logical that may already have been parsed to a bool."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in (".true.", ".t.", "true", "t")


def _tot_magnetization(pwin: PwInput) -> float | None:
    """``tot_magnetization``, with QE's sentinel turned into ``None``.

    ``input_parameters.f90`` defaults it to -10000 and ``set_nelup_neldw`` tests
    ``< -9999`` -- the flag for "not given" is the *value*, and the switch it
    controls (``two_fermi_energies``) changes the physics, so the sentinel is
    resolved once here rather than compared for again downstream.
    """
    value = pwin.get("system", "tot_magnetization")
    if value is None or float(value) < -9999.0:
        return None
    return float(value)


def _input_occupations(pwin: PwInput) -> tuple[float, ...] | None:
    """The OCCUPATIONS card, flattened. Present only for occupations='from_input'."""
    card = pwin.card("OCCUPATIONS")
    if card is None:
        return None
    return tuple(value for row in card.floats() for value in row)


def _build_cell(pwin: PwInput, precision: Precision) -> Cell:
    ibrav = pwin.get("system", "ibrav")
    if ibrav is None:
        if pwin.get("system", "space_group") is not None:
            # ``input.f90``'s ``sup_spacegroup`` derives ``ibrav`` from the space
            # group and expands the Wyckoff letters into a full basis, so such an
            # input is legal QE and simply omits ``ibrav``. Saying "ibrav is
            # required" of it names the wrong thing -- the gap is the Wyckoff
            # input PLAN.md's P6 still lists as outstanding, and
            # ``Structure.from_card_units`` refuses ``crystal_sg`` a step later
            # for the same reason.
            raise NotImplementedError(
                f"{pwin.path or 'input'}: space_group = "
                f"{pwin.get('system', 'space_group')} selects Wyckoff input, "
                "which is not implemented (Modules/space_group.f90's "
                "sup_spacegroup derives ibrav from the group and expands the "
                "Wyckoff letters into a basis). Give ibrav and the full "
                "ATOMIC_POSITIONS instead"
            )
        raise ValueError(f"{pwin.path or 'input'}: ibrav is required")
    ibrav = int(ibrav)

    celldm = np.array(pwin.indexed("system", "celldm", 6))
    a = pwin.get("system", "a")
    if celldm[0] == 0.0 and a is not None:
        # The crystallographic alternative to celldm: A,B,C in angstrom plus cosines.
        celldm = celldm_from_abc(
            ibrav,
            float(a),
            float(pwin.get("system", "b", 0.0)),
            float(pwin.get("system", "c", 0.0)),
            float(pwin.get("system", "cosab", 0.0)),
            float(pwin.get("system", "cosac", 0.0)),
            float(pwin.get("system", "cosbc", 0.0)),
        )

    if ibrav != 0:
        return Cell.from_ibrav(ibrav, celldm, precision=precision)

    card = pwin.require_card("CELL_PARAMETERS")
    vectors = np.array(card.floats(), dtype=float)
    if vectors.shape != (3, 3):
        raise ValueError("CELL_PARAMETERS must give three vectors of three components")

    units = card.option
    if units is None:
        # Historical default, kept because old inputs rely on it: alat when a
        # lattice parameter was supplied, bohr otherwise.
        units = "alat" if celldm[0] != 0.0 else "bohr"

    if units == "alat":
        if celldm[0] == 0.0:
            raise ValueError("CELL_PARAMETERS alat needs celldm(1) or A")
        return Cell.from_vectors(vectors * celldm[0], alat=float(celldm[0]), precision=precision)
    if units == "bohr":
        return Cell.from_vectors(vectors, alat=celldm[0] or None, precision=precision)
    if units == "angstrom":
        vectors = vectors * ANGSTROM_TO_BOHR
        return Cell.from_vectors(vectors, alat=celldm[0] or None, precision=precision)
    raise ValueError(f"unknown CELL_PARAMETERS units {units!r}")


def _build_structure(pwin: PwInput, cell: Cell, precision: Precision) -> Structure:
    species_card = pwin.require_card("ATOMIC_SPECIES")
    species, index_of = [], {}
    for line in species_card.lines:
        name, mass, pseudo = line.split()[:3]
        index_of[name] = len(species)
        species.append(Species(name=name, mass=fortran_float(mass), pseudo_file=pseudo))

    positions_card = pwin.require_card("ATOMIC_POSITIONS")
    names, coordinates, if_pos = [], [], []
    for line in positions_card.lines:
        tokens = line.split()
        names.append(tokens[0])
        coordinates.append([fortran_float(t) for t in tokens[1:4]])
        # Trailing 0/1 flags -- ``if_pos``, which freezes a coordinate during a
        # relaxation. Absent means free.
        flags = tokens[4:7]
        if_pos.append([int(fortran_float(f)) for f in flags] if len(flags) == 3 else [1, 1, 1])

    unknown = set(names) - set(index_of)
    if unknown:
        raise ValueError(f"ATOMIC_POSITIONS names a species not in ATOMIC_SPECIES: {sorted(unknown)}")

    nat = pwin.get("system", "nat")
    if nat is not None and int(nat) != len(names):
        raise ValueError(f"nat={nat} but ATOMIC_POSITIONS lists {len(names)} atoms")

    return Structure.from_card_units(
        positions=coordinates,
        types=[index_of[name] for name in names],
        species=species,
        units=positions_card.option,
        cell=cell,
        precision=precision,
        if_pos=if_pos,
    )


def local_moments(
    structure: Structure,
    nspin: int,
    starting_magnetization,
    angle1,
    angle2,
) -> np.ndarray:
    """``m_loc``: the starting moment of every atom, cartesian -- ``setup.f90``.

    It is what decides the magnetic symmetry group, so it is computed here, once,
    from the input and never from the converged state -- exactly as ``domag`` is
    (see :attr:`System.domag`). ``angle1``/``angle2`` are the polar and azimuthal
    angles in degrees, and a collinear run has no angles: its moment is along
    ``z`` by construction.
    """
    types = np.asarray(structure.types, dtype=int)
    ntyp = structure.ntyp
    magnitudes = np.zeros(ntyp)
    given = np.asarray(starting_magnetization, dtype=float)
    magnitudes[: len(given)] = given
    if nspin != 4:
        moments = np.zeros((len(types), 3))
        moments[:, 2] = magnitudes[types]
        return moments

    theta = np.zeros(ntyp)
    phi = np.zeros(ntyp)
    for target, values in ((theta, angle1), (phi, angle2)):
        values = np.asarray(values, dtype=float)
        target[: len(values)] = np.deg2rad(values)
    directions = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
        axis=1,
    )
    return magnitudes[types, None] * directions[types]


def _spiral_q(pwin: PwInput, nspin: int, lspinorb: bool, nosym: bool) -> tuple | None:
    """``spiral_q``, and the three things a spiral cannot be combined with.

    A pypresso extension -- ``pw.x`` has no spin spiral at all -- with Elk's
    name for the quantity (``vqlss``) and Elk's units (lattice coordinates).
    """
    if "spiral_q" not in pwin.namelists.get("system", {}):
        return None
    q = tuple(float(v) for v in pwin.indexed("system", "spiral_q", 3))

    if nspin != 4:
        # The two spinor components live on different plane-wave spheres, so
        # there is no collinear or unpolarized form of the calculation.
        raise ValueError(
            "spiral_q needs noncolin = .true.: a spin spiral is a two-component "
            "spinor with a different G-sphere per component"
        )
    if lspinorb:
        # Permanently: spin-orbit coupling ties spin to the lattice, so the
        # combined translation-plus-spin-rotation the generalized Bloch theorem
        # rests on is not a symmetry. Elk refuses the same combination
        # (``init0.f90`` sets ``spinorb = .false.`` when ``spinsprl``).
        raise ValueError(
            "spiral_q with lspinorb = .true. is not a calculation: spin-orbit "
            "coupling breaks the generalized Bloch theorem a spiral rests on"
        )
    if not nosym:
        # The spin space group is not written; see pypresso.system.spiral.
        raise ValueError(
            "a spin spiral needs nosym = .true.: only the operations with "
            "S^T q = q survive at all, they act on the rotated-frame "
            "magnetization with a spin rotation of their own (the spin space "
            "group, which is not implemented), and time reversal sends q to -q"
        )
    return q



def _fsm_update(pwin) -> str:
    """``fsm_update``, refused by name rather than silently substituted."""
    from pypresso.scf.fields import DEFAULT_FSM_UPDATE, FSM_UPDATES

    name = str(pwin.get("system", "fsm_update", DEFAULT_FSM_UPDATE)).strip().lower()
    if name not in FSM_UPDATES:
        raise ValueError(
            f"fsm_update = {name!r}: implemented are {', '.join(FSM_UPDATES)}"
        )
    return name


def _atomic_b_field(pwin: PwInput, nat: int) -> tuple:
    """The ``LOCAL_MAGNETIC_FIELDS`` card: one field per atom, cartesian, in Ry.

    Elk's ``bfcmt``. ``()`` when the card is absent, which is the normal case.
    """
    card = pwin.card("LOCAL_MAGNETIC_FIELDS")
    if card is None:
        return ()
    rows = [line.split() for line in card.lines if line.strip()]
    if len(rows) != nat:
        raise ValueError(
            f"LOCAL_MAGNETIC_FIELDS lists {len(rows)} atoms but the cell has {nat}"
        )
    return tuple(tuple(fortran_float(v) for v in row[:3]) for row in rows)


def _hubbard(pwin: PwInput, structure):
    """The ``HUBBARD`` card, with ``hubbard_occ`` and ``starting_ns_eigenvalue``.

    ``CARD: HUBBARD`` in ``INPUT_PW.txt``. Each line is a parameter name, a
    ``label-manifold`` specification and a value:

        HUBBARD (ortho-atomic)
          U  Fe1-3d 4.3
          J0 Fe1-3d 1.0

    **The card's energies are in eV** and are converted to Ry here, which is the
    only place in the code that sees an eV (rule R6). The parameters QE accepts
    but this does not -- ``J``, ``B``, ``E2``, ``E3`` (the full Liechtenstein
    formulation) and ``V`` (the intersite term) -- are refused by name rather
    than ignored, because ignoring one silently runs a different functional.
    """
    from pypresso.hubbard.manifold import HubbardInput, parse_manifold

    card = pwin.card("HUBBARD")
    if card is None:
        return None
    projectors = (card.option or "atomic").lower()

    accepted = {"u", "j0", "alpha", "beta"}
    refused = {
        "j": "the full (Liechtenstein) formulation, lda_plus_u_kind = 1",
        "b": "the full (Liechtenstein) formulation, lda_plus_u_kind = 1",
        "e2": "the full (Liechtenstein) formulation, lda_plus_u_kind = 1",
        "e3": "the full (Liechtenstein) formulation, lda_plus_u_kind = 1",
        "v": "the intersite Hubbard V, lda_plus_u_kind = 2",
    }
    entries: dict[str, dict] = {}
    for line in card.lines:
        fields = line.split()
        if not fields:
            continue
        name = fields[0].lower()
        if name in refused:
            raise NotImplementedError(
                f"HUBBARD parameter {fields[0]!r} selects {refused[name]}, which "
                "is not implemented; only U, J0, ALPHA and BETA (the simplified "
                "rotationally-invariant functional, lda_plus_u_kind = 0) are"
            )
        if name not in accepted:
            raise ValueError(f"unknown HUBBARD parameter {fields[0]!r}")
        if len(fields) < 3:
            raise ValueError(f"malformed HUBBARD line {line!r}")
        if len(fields) > 3:
            raise NotImplementedError(
                f"HUBBARD line {line!r} carries orbital indices, which select "
                "the orbital-resolved formulation; that is not implemented"
            )
        label, n, l = parse_manifold(fields[1])
        entry = entries.setdefault(label, {"n": n, "l": l})
        if (entry["n"], entry["l"]) != (n, l):
            raise NotImplementedError(
                f"{label} is given two Hubbard manifolds "
                f"({entry['n']}{'spdf'[entry['l']]} and {n}{'spdf'[l]}); the "
                "second (background) channel is not implemented"
            )
        entry[name] = fortran_float(fields[2]) / RY_TO_EV

    names = [species.name for species in structure.species]
    occupations = []
    raw = pwin.get("system", "hubbard_occ")
    if raw is not None and not isinstance(raw, dict):
        raw = {(1, 1): raw}
    if isinstance(raw, dict):
        for index, value in raw.items():
            kind = index[0] - 1
            channel = index[1] if len(index) > 1 else 1
            if channel != 1:
                raise NotImplementedError(
                    f"hubbard_occ(..., {channel}) sets a background channel's "
                    "occupation; background channels are not implemented"
                )
            if not 0 <= kind < len(names):
                raise ValueError(f"hubbard_occ{index} out of range")
            occupations.append((names[kind], float(value)))

    starting_ns = []
    raw = pwin.get("system", "starting_ns_eigenvalue")
    if isinstance(raw, dict):
        for index, value in raw.items():
            m, spin, kind = (list(index) + [1, 1])[:3]
            starting_ns.append((kind - 1, spin - 1, m - 1, float(value)))

    return HubbardInput(
        projectors=projectors,
        parameters=tuple(
            (
                label, entry["n"], entry["l"],
                entry.get("u", 0.0), entry.get("j0", 0.0),
                entry.get("alpha", 0.0), entry.get("beta", 0.0),
            )
            for label, entry in entries.items()
        ),
        occupations=tuple(occupations),
        starting_ns=tuple(starting_ns),
    )


def _build_kpoints(
    pwin: PwInput,
    cell: Cell,
    precision: Precision,
    rotations=None,
    time_reversal: bool = True,
    t_rev=None,
    lattice_rotations=None,
) -> KPoints:
    card = pwin.card("K_POINTS")
    if card is None:
        return KPoints.gamma(precision=precision)

    option = (card.option or "tpiba").lower()

    if option == "gamma":
        return KPoints.gamma(precision=precision)

    if option == "automatic":
        values = [int(fortran_float(v)) for v in card.lines[0].split()[:6]]
        return KPoints.automatic(
            tuple(values[:3]), tuple(values[3:6]), cell,
            precision=precision, rotations=rotations,
            time_reversal=time_reversal, t_rev=t_rev,
        )

    # All remaining forms start with a count, then one line per k-point.
    rows = np.array([[fortran_float(t) for t in line.split()[:4]] for line in card.lines[1:]])
    declared = int(card.lines[0].split()[0])
    if len(rows) != declared:
        raise ValueError(f"K_POINTS declares {declared} points but lists {len(rows)}")

    points, fourth = rows[:, :3], rows[:, 3]

    if option in ("tpiba", "crystal"):
        # ``irreducible_BZ`` runs on an explicit list too: QE takes it to be the
        # wedge of the *lattice's* point group and completes it wherever the
        # crystal has fewer operations. That is a no-op for most inputs and is
        # exactly what a magnetic noncollinear run needs -- see
        # :func:`pypresso.system.kpoints.expand_to_subgroup`.
        crystal_points = (
            np.asarray(points, dtype=float) if option == "crystal"
            else np.asarray(cell.k_to_crystal(np.asarray(points, dtype=float)))
        )
        if rotations is not None and lattice_rotations is not None and len(rotations):
            crystal_points, fourth = expand_to_subgroup(
                crystal_points, fourth, lattice_rotations, rotations,
                time_reversal=time_reversal, t_rev=t_rev,
            )
        return KPoints.from_crystal(crystal_points, fourth, cell, precision=precision)

    if option in ("tpiba_b", "crystal_b"):
        return KPoints.band_path(
            points, fourth, cell, crystal=option.startswith("crystal"), precision=precision
        )

    if option in ("tpiba_c", "crystal_c"):
        raise NotImplementedError(
            "K_POINTS *_c (three points defining a plane) is not needed before the band phase"
        )

    raise ValueError(f"unknown K_POINTS option {option!r}")
