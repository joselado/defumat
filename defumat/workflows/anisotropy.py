"""Magnetocrystalline anisotropy by the force theorem.

The energy it costs to point a magnet's moment one way rather than another.
That difference is what makes a hard magnet hard, what pins a spiral into a
plane, and what a recording medium is; it is also tiny -- tenths of a meV per
atom against total energies of hundreds of Ry -- so computing it as a
difference of two self-consistent total energies asks two SCF runs to agree to
a part in 10^9.

**The force theorem does not ask them to.** Converge the magnet *without*
spin-orbit coupling, rotate the converged density so its magnetization points
along ``n``, and diagonalise **once** with spin-orbit coupling switched on. At
frozen density every term of the total energy except the band energy is a
functional of ``rho`` alone and is therefore *identical* between two
directions, so

    E(n_1) - E(n_2) = eband(n_1) - eband(n_2),    eband = sum_ik w_ik f_ik eps_ik

exactly, and the whole calculation is one diagonalisation per direction.
Jansen, PRB 38, 8022 (1988); Daalderop, Kelly and Schuurmans, PRB 41, 11919
(1990); the QE implementation follows Blonski and Hafner's usage in PRB 90,
205409 (2014), which is the paper ``PP/examples/ForceTheorem_example`` cites.

**What is *not* enough, and it is the reason this is a diagonalisation.** The
obvious cheaper thing -- freeze the wavefunctions too and take the one-shot
expectation value ``<psi|H_SOC|psi>`` -- gives almost none of the anisotropy.
Spin-orbit coupling enters at first order as ``xi <L> . n``, and the orbital
moment of a scalar-relativistic collinear state is quenched: this package
measures it at 1.7e-16 (:mod:`defumat.projwfc.angular_momentum`). The
anisotropy is second order in the coupling, and what supplies it is the
*repulsion between levels* that the diagonalisation performs and an expectation
value does not. :func:`frozen_expectation` computes the first-order term
anyway, because a number measured is worth more than an argument about why it
should be small, and the number is **not** the argument's zero: on tetragonal
cobalt it carries **1.79e-3 meV** of anisotropy, with the diagonalisation's
sign, where the force theorem on the same density gives **0.552 meV** of
free-energy anisotropy -- a factor of 300 between the two orders. The part that
couples to ``<L>`` does vanish, to 1e-6 meV; what survives is the exchange
field's augmentation seen through the ``j``-resolved projectors, which a
quenched state does not remove (the function's docstring has the split).

**How QE spreads it over three runs, and what each contributes.**

===========================================  ==========================================
``pw.x``, ``nspin = 2``, scalar-relativistic  the SCF. Writes the density.
``pw.x``, ``nscf``, ``noncolin``,             ``potinit.f90:96`` reads *only* ``rho``
``lspinorb``, ``lforcet``,                    (``read_rhog``, not ``read_scf``), then
``angle1``/``angle2``                         ``nc_magnetization_from_lsda`` rotates
                                              ``m_z`` onto ``angle1(1)``/``angle2(1)``.
                                              ``print_ks_energies.f90:90`` prints
                                              ``eband``.
``projwfc.x``, ``lforcet``, ``ef_0``          ``force_theorem`` (``projwfc.f90:541``)
                                              decomposes ``sum wg (eps - ef_0)`` over
                                              atomic orbitals.
===========================================  ==========================================

Here it is one call, because there is no file to hand between processes -- but
the *physics* is transcribed rather than reinvented, including the two things
that make it work at all and are easy to get wrong. **The two legs use two
different pseudopotential files**: a scalar-relativistic dataset for the SCF and
the fully-relativistic dataset *of the same generation* for the one-shot
(``Co.pbe-nd-rrkjus.UPF`` and ``Co.rel-pbe-nd-rrkjus.UPF`` in QE's example).
That is what makes an ultrasoft anisotropy possible at all: the alternative --
one fully-relativistic file with its ``j`` channels averaged back for the SCF
leg -- is QE's ``average_pp``, which refuses ultrasoft and PAW outright
(``average_pp.f90:34``). Only the density crosses between the two, and a
density does not know which file made it.

**And the k-set must not move between directions.** A noncollinear magnetic run
reduces its grid with the *magnetic* symmetry group, which depends on where the
moment points -- so two directions asked for naively arrive on two different
wedges and their band energies differ by the k-sampling rather than by the
physics. QE's own example sets ``nosym = .true.`` for exactly this reason, and
so does the refusal in :func:`run_force_theorem`.

**PAW needs one file rather than two, and that is the whole of it.** A PAW
Hamiltonian's one-centre coefficients ``ddd_paw`` are a functional of
``becsum`` exactly as the grid potential is a functional of ``rho``, so what
the theorem freezes on this dataset is the *pair*: the potential has two
representations and both are held fixed. ``becsum`` cannot cross between the
two files of the route above -- a scalar-relativistic dataset has one projector
per ``(n, l)`` channel and its fully-relativistic partner one per ``(n, l, j)``,
so the arrays are indexed by different things -- and that is why ``pw.x``
refuses the path outright (``potinit.f90:98``), ``average_pp`` having no way to
average a PAW dataset's ``j`` channels back into a scalar one. The way round it
is Elk's dial rather than QE's file pair: run the **self-consistent** leg on the
fully-relativistic dataset with ``soc_scale = 0`` and the one-shot leg on the
same file with the coupling on, so that both legs share a projector set by
construction and ``becsum`` crosses by shape. It is rotated onto the requested
direction with the density, off the axis the *density* defines.

**Refused by name.** A Hubbard ``U``, whose ``ns`` is not in the handoff;
a potential-only meta-GGA, whose ``tau`` is not; a converged magnetic field or
constrained moment, whose energy is outside the reported total; and a spin
spiral, which has no spin-orbit coupling to switch on.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from defumat.pseudo.projectors import projector_channels
from defumat.pseudo.upf import Pseudopotential
from defumat.scf.continuation import (
    direction_from_angles,
    nc_magnetization_from_lsda,
)
from defumat.system.builder import System, local_moments
from defumat.system.kpoints import KPoints
from defumat.units import RY_TO_EV
from defumat.workflows.nscf import fixed_density_states

__all__ = [
    "becsum_fits",
    "becsum_for_leg",
    "ForceTheorem",
    "MagneticAnisotropy",
    "run_force_theorem",
    "run_anisotropy",
    "frozen_expectation",
    "MagneticTorque",
    "run_torque",
    "OrientationTorque",
    "run_orientation_torque",
    "rotation_from_euler",
    "euler_from_rotation",
    "OrientationStep",
    "RelaxedOrientation",
    "relax_orientation",
    "RelaxedDirection",
    "RelaxedAnisotropy",
    "run_relaxed_direction",
    "run_relaxed_anisotropy",
    "cardinal_directions",
    "sphere_cover",
]

#: How far a requested direction may sit from the system's own ``angle1``/
#: ``angle2`` axis and still count as the same one, so that a symmetry-reduced
#: k-set is not silently reused for a different magnetic group.
DIRECTION_TOL = 1.0e-8


@dataclass
class ForceTheorem:
    """One direction's one-shot band energy, and what it was built from."""

    #: Cartesian unit vector the magnetization was rotated onto.
    direction: tuple
    #: ``sum_ik w f eps`` in Ry -- ``pw.x``'s ``eband`` on the ``lforcet`` path.
    band_energy: float
    fermi_energy: float | None = None
    #: The smearing's ``-TS``, in Ry. Carried but **not** added to
    #: :attr:`band_energy`, which is QE's convention -- ``print_ks_energies``
    #: prints the bare sum. :attr:`free_energy` is the sum with it.
    entropy: float = 0.0
    eigenvalues: np.ndarray | None = None
    occupations: np.ndarray | None = None
    kpoints: KPoints | None = None
    #: Per-orbital decomposition, when it was asked for.
    projected: "ProjectedBandEnergy | None" = None

    @property
    def band_energy_ev(self) -> float:
        return self.band_energy * RY_TO_EV

    @property
    def free_energy(self) -> float:
        """``eband + (-TS)``, the quantity that is variational for a metal."""
        return self.band_energy + self.entropy


@dataclass
class ProjectedBandEnergy:
    """``force_theorem``'s decomposition of the band energy over atomic orbitals.

    ``eband_proj[nwfc] = sum_ik w_ik (eps_ik - ef_0) |<phi_nwfc|S|psi_ik>|^2``
    (``projwfc.f90:576-582``). Differencing it between two directions gives the
    orbital-resolved anisotropy, which is what says *which* orbitals supply it.
    """

    #: ``(natomwfc,)`` in Ry.
    by_orbital: np.ndarray
    #: One ``(channel, spin)`` pair per entry, the channel being a
    #: :class:`~defumat.projwfc.channels.AtomicChannel` and the spin the string
    #: ``"up"`` or ``"down"``, so there are ``2 * len(channels)`` of them rather
    #: than one per orbital.
    labels: tuple
    #: The reference level subtracted from every eigenvalue, in Ry.
    ef_0: float
    #: ``(nat,)`` in Ry -- :attr:`by_orbital` summed over each atom's orbitals.
    by_atom: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def total(self) -> float:
        """``eband_proj_tot``: the projected sum, which is not ``eband``.

        The gap between the two is the spilling -- what the atomic-orbital
        basis does not span -- so comparing them is this decomposition's own
        diagnostic, exactly as ``projwfc.f90`` prints both on one line.
        """
        return float(np.sum(self.by_orbital))


@dataclass
class MagneticAnisotropy:
    """A set of directions and the band energy of each."""

    directions: tuple
    results: tuple
    #: Which entry of :attr:`results` the energies are measured from.
    reference: int = 0

    @property
    def band_energies(self) -> np.ndarray:
        """``(ndir,)`` in Ry."""
        return np.array([r.band_energy for r in self.results])

    @property
    def energies(self) -> np.ndarray:
        """Band energies relative to :attr:`reference`, in Ry."""
        return self.band_energies - self.band_energies[self.reference]

    @property
    def energies_mev(self) -> np.ndarray:
        return self.energies * RY_TO_EV * 1000.0

    @property
    def free_energies(self) -> np.ndarray:
        """``(ndir,)`` in Ry: the band energy **plus the smearing's ``-TS``**.

        For an insulator this is :attr:`band_energies`. For a **smeared metal
        it is a different curve**, and the difference is not small: the free
        energy is what a variational argument is about, so it is the free
        energy whose derivative is the torque
        (:func:`run_torque`), while ``sum w eps`` -- the quantity ``pw.x``
        prints on the ``lforcet`` path and :attr:`band_energies` carries -- has
        an extra ``sum (dw/dtheta) eps`` in its derivative. On tetragonal
        cobalt at ``degauss = 0.02`` Ry the entropy supplies **55 per cent** of
        the band energy's slope, so the two anisotropies are 1.235 and 0.551
        meV. They converge onto each other as the smearing goes to zero; at a
        production smearing they must be quoted apart.
        """
        return np.array([r.free_energy for r in self.results])

    @property
    def free_anisotropy_mev(self) -> float:
        """The spread of :attr:`free_energies`, in meV."""
        energies = self.free_energies
        return float(np.max(energies) - np.min(energies)) * RY_TO_EV * 1000.0

    @property
    def anisotropy(self) -> float:
        """Hardest minus easiest, in Ry -- the MAE proper."""
        energies = self.band_energies
        return float(np.max(energies) - np.min(energies))

    @property
    def anisotropy_mev(self) -> float:
        return self.anisotropy * RY_TO_EV * 1000.0

    @property
    def easy_axis(self) -> tuple:
        return self.directions[int(np.argmin(self.band_energies))]

    @property
    def hard_axis(self) -> tuple:
        return self.directions[int(np.argmax(self.band_energies))]

    def difference(self, i: int, j: int) -> float:
        """``eband(i) - eband(j)`` in Ry, the sign QE's README states."""
        return float(self.band_energies[i] - self.band_energies[j])


def becsum_for_leg(becsum, source, pseudos) -> tuple:
    """The first leg's ``becsum`` when it belongs to this leg's dataset.

    The decision the front door takes, written here so that the front door
    stays one line and the reasoning stays with the physics. On the one-file
    route, where the two legs differ by ``soc_scale`` alone, ``becsum`` is
    indexed by the projectors this leg has and crosses. On the two-file route
    it is not, and handing over nothing is right: an ultrasoft dataset needs no
    ``becsum`` at all and a PAW one then refuses by name, with the one-file
    route in the message.
    """
    return tuple(becsum or ()) if becsum_fits(becsum, pseudos, source=source) else ()


def _files(pseudos) -> tuple:
    """The file each species names, which is what identifies a dataset."""
    import os

    return tuple(
        os.path.basename(str(getattr(pseudo, "path", "") or "")) or pseudo.element
        for pseudo in pseudos
    )


def becsum_fits(becsum, pseudos, source=None) -> bool:
    """Whether ``becsum`` is indexed by *these* projectors.

    What the front door asks before it hands the first leg's ``becsum`` on. On
    the two-file route -- a scalar-relativistic dataset for the self-consistent
    leg and its fully-relativistic partner for the one-shot -- the answer is no
    and the right thing to do is to hand over nothing, an ultrasoft dataset
    needing no ``becsum`` and a PAW one refusing by name with the one-file
    route in the message. On the one-file route, ``soc_scale = 0`` and then
    ``1``, the answer is yes.

    ``source`` is the first leg's own pseudopotentials, and it is the
    discriminator that cannot be fooled where the shape can. **A matching
    ``nh`` does not mean matching projectors**: it separates the two files of
    the route above (18 against 34 on both the platinum and the cobalt pairs
    committed here), and it does *not* separate two datasets of the same
    generation, where ``Si.pbe-n-rrkjus_psl.0.1`` and ``Si.pbe-n-kjpaw_psl.0.1``
    both have ``nh = 8`` and ``Ni.rel-pbe-spn-rrkjus_psl.1.0.0`` and its
    ``kjpaw`` partner both have 34. Those are different radial projectors with
    the same count, so a ``becsum`` would cross silently. Comparing the file
    each species names is what says the two legs are one dataset.
    """
    becsum = tuple(becsum or ())
    if not becsum or len(becsum) != len(pseudos):
        return False
    if source is not None and _files(source) != _files(pseudos):
        return False
    for pseudo, values in zip(pseudos, becsum):
        if values is None:
            continue
        shape = tuple(np.shape(values))
        nh = len(projector_channels(pseudo))
        if len(shape) != 4 or shape[-2:] != (nh, nh) or shape[0] not in (2, 4):
            return False
    return True


def _checked_becsum(becsum, pseudos) -> tuple:
    """The first leg's ``becsum``, checked against *this* leg's projector set.

    The check is the whole reason a PAW force theorem can be run at all, so it
    is made here rather than left to fail somewhere inside the Hamiltonian.
    ``becsum`` is indexed by projector pairs, ``(nspin_mag, natom, nh, nh)``
    per species, and ``nh`` is a property of the pseudopotential file: a
    scalar-relativistic dataset has one projector per ``(n, l)`` channel and
    its fully-relativistic partner has one per ``(n, l, j)``, so the two
    numbers differ and the arrays do not describe the same object. A mismatch
    is a swapped file rather than a mistake in the shapes, and saying which
    file is what makes the error fixable.

    **The check is necessary and not sufficient, and the caller owns the rest.**
    It separates the two files of the two-file route -- ``nh`` is 18 against 34
    on both committed pairs -- and it cannot separate two datasets of the *same*
    generation, an ultrasoft and a PAW silicon both having ``nh = 8``. Passing
    a ``becsum`` from a run on a different file with the same projector count
    would cross silently, so the rule is one file for both legs and the front
    door checks that by name (:func:`becsum_fits`).
    """
    becsum = tuple(becsum or ())
    if not becsum:
        return ()
    if len(becsum) != len(pseudos):
        raise ValueError(
            f"the force theorem was handed becsum for {len(becsum)} species "
            f"and this leg has {len(pseudos)}: it has to come from a run on "
            "the same structure"
        )
    for pseudo, values in zip(pseudos, becsum):
        if values is None:
            continue
        shape = tuple(np.shape(values))
        nh = len(projector_channels(pseudo))
        if len(shape) != 4 or shape[-2:] != (nh, nh):
            raise ValueError(
                f"the becsum handed over for {pseudo.element} has shape "
                f"{shape} where this leg's dataset has {nh} projector "
                f"channels and needs (nspin_mag, natom, {nh}, {nh}). That is "
                "a different pseudopotential file, not a different spin "
                "regime: run the self-consistent leg on *this* file with "
                "soc_scale = 0 rather than on its scalar-relativistic partner"
            )
        if shape[0] not in (2, 4):
            raise ValueError(
                f"the becsum handed over for {pseudo.element} has "
                f"nspin_mag = {shape[0]}: the force theorem rotates a magnetic "
                "state onto a direction, so the first leg has to be magnetic"
            )
    return becsum


def _refuse_system(
    system: System, pseudos, require_spin_orbit: bool = True, becsum: tuple = ()
) -> None:
    """Everything this handoff cannot carry, from the input alone.

    **All of it has to be decided here rather than from the assembled
    Calculation**, and that is not a stylistic choice:
    :func:`~defumat.workflows.nscf.fixed_density_states` makes its own
    PAW/Hubbard/meta-GGA/field refusals *before* it returns, so a check written
    after it is unreachable code. Its messages are the right ones for an
    ordinary NSCF run and the wrong ones here -- "pass becsum =
    scf_result.becsum" cannot be followed when the ``becsum`` in question
    belongs to a run that used a **different pseudopotential file** and has a
    different projector count.

    Each entry is a quantity that is a property of the *wavefunctions* rather
    than of the density, so it does not cross with ``rho``: QE's own handoff is
    ``read_rhog`` and nothing else (``potinit.f90:96``).
    """
    if any(pseudo.is_paw for pseudo in pseudos) and not becsum:
        raise NotImplementedError(
            "the force theorem with a PAW dataset needs the converged becsum "
            "beside the density: a PAW Hamiltonian's one-centre coefficients "
            "ddd_paw are built from becsum, which is a property of the "
            "wavefunctions and cannot be rebuilt from the density. Pass "
            "becsum = scf_result.becsum from a first leg that ran *this* "
            "dataset with soc_scale = 0 -- the matched scalar-relativistic "
            "file has a different number of projectors, so its becsum does "
            "not fit this Hamiltonian at all, and that is why QE refuses the "
            "path outright (potinit.f90:98) rather than for a reason that "
            "could be plumbed around. An ultrasoft dataset needs none of "
            "this, its augmentation charge being inside the density that "
            "crosses"
        )
    if system.hubbard:
        raise NotImplementedError(
            "the force theorem with a Hubbard U: ns is a property of the "
            "wavefunctions and is not in the handoff, and QE's average_pp "
            "refuses lda_plus_u on the same path (average_pp.f90:34)"
        )
    if system.input_dft and system.input_dft.strip().lower() in ("tb09", "bj06"):
        raise NotImplementedError(
            f"the force theorem under {system.input_dft}: tau is a property of "
            "the occupied states and is not in the handoff. A potential-only "
            "meta-GGA has no total energy for the theorem to be about either"
        )
    if (
        system.constrained_magnetization != "none"
        or any(system.b_field)
        or any(any(v) for v in system.atomic_b_field or ())
    ):
        raise NotImplementedError(
            "the force theorem with a magnetic field or a constrained moment: "
            "the field's energy is deliberately outside the reported total "
            "(defumat/scf/fields.py), so the band energies of two directions "
            "differ by a Zeeman term that no total energy accounts for"
        )
    if system.spiral:
        raise NotImplementedError(
            "the force theorem for a spin spiral: a spiral refuses spin-orbit "
            "coupling permanently (it breaks the generalized Bloch theorem), "
            "so there is no coupling to switch on"
        )
    if not system.noncolin:
        raise ValueError(
            f"the force theorem's one-shot leg is a noncollinear run and this "
            f"system has nspin = {system.nspin}: build it from the nscf input, "
            "with noncolin = .true. and lspinorb = .true."
        )
    if require_spin_orbit and not system.lspinorb:
        raise ValueError(
            "the force theorem's one-shot leg needs lspinorb = .true. and a "
            "fully-relativistic dataset: without spin-orbit coupling the band "
            "energy does not depend on the direction at all and every "
            "direction comes out equal. Pass require_spin_orbit = False to run "
            "it anyway, which is the control for exactly that identity"
        )


def run_force_theorem(
    system: System,
    pseudos: tuple[Pseudopotential, ...],
    density: jnp.ndarray,
    direction=None,
    nbnd: int | None = None,
    conv_thr: float = 1.0e-10,
    k_batch: int | None | str = "default",
    ef_0: float | None = None,
    projected: bool = False,
    require_spin_orbit: bool = True,
    soc_scale: float | None = None,
    becsum: tuple = (),
) -> ForceTheorem:
    """One direction: rotate the density onto ``n``, diagonalise once, sum.

    ``system`` and ``pseudos`` are the *one-shot* leg's -- noncollinear, with
    spin-orbit coupling and a fully-relativistic dataset -- and ``density`` is
    the **collinear** density a previous scalar-relativistic SCF converged
    (``SCFResult.density``, ``(2, n1, n2, n3)``). That mismatch is the point:
    the two legs are two different pseudopotential files and only the density
    crosses, exactly as ``pw.x`` does it across two invocations.

    ``direction`` defaults to the system's own ``angle1(1)``/``angle2(1)``,
    which is where ``nc_magnetization_from_lsda`` reads it. Passing it
    explicitly is how a scan over directions avoids rebuilding the system --
    and is refused unless the run is ``nosym``, because a magnetic
    noncollinear run's k-set is reduced with a group that depends on where the
    moment points.

    ``conv_thr`` defaults far tighter than an NSCF run's usual 1e-6: the whole
    answer is a difference of band-energy sums in the fifth decimal of an eV,
    and QE's own example sets ``diago_thr_init = 1.d-14`` for the same reason.

    ``soc_scale`` is Elk's ``socscf`` (:meth:`~defumat.system.builder.System.
    with_soc_scale`), restricted to **0 or 1**: it switches the spin-orbit part
    of ``dvan_so`` and ``qq_so`` off or on while keeping the same dataset and,
    crucially, the same k-points. ``0`` gives an anisotropy of exactly zero,
    which is the same identity ``require_spin_orbit = False`` gives but on
    **one** file rather than on a matched scalar/relativistic pair.

    ``becsum`` is the first leg's converged projector occupations
    (``SCFResult.becsum``) and is what a **PAW** dataset needs beside the
    density, its one-centre coefficients ``ddd_paw`` being a functional of
    ``becsum`` exactly as the grid potential is of ``rho``. Freezing the pair
    is what the theorem asks for on this dataset: the frozen object is the
    whole potential, and on a PAW dataset the potential has two representations
    rather than one. It only crosses from a leg that used **this** file, so the
    route is one fully-relativistic dataset run twice -- ``soc_scale = 0`` for
    the self-consistent leg and ``1`` for the one-shot -- rather than the
    matched scalar/relativistic pair an ultrasoft run uses, whose two files
    have different numbers of projectors. It is rotated onto ``direction``
    with the density and with the axis the *density* defines, so that the
    one-centre field and the grid field point the same way.

    ``require_spin_orbit = False`` runs the same assembly on a
    *scalar-relativistic* one-shot leg, where the answer is known in advance:
    the Hamiltonian without spin-orbit coupling commutes with a global spin
    rotation, so every direction must give the **same** band energy exactly.
    That is the phase's strongest internal check and the one that found its
    only real bug (:func:`_with_quantization_axis`), so it is an option rather
    than something only a test can reach by calling internals.
    """
    if soc_scale is not None:
        system = system.with_soc_scale(soc_scale)
    becsum = _checked_becsum(becsum, pseudos)
    _refuse_system(system, pseudos, require_spin_orbit, becsum)
    if direction is None:
        direction = _reference_axis(system)
    else:
        own = np.asarray(_reference_axis(system))
        wanted = np.asarray(direction, dtype=float)
        wanted = wanted / np.sqrt(np.sum(wanted**2))
        if not system.nosym and np.sum(np.abs(own - wanted)) > DIRECTION_TOL:
            raise ValueError(
                "a force-theorem direction other than the system's own "
                f"angle1/angle2 ({tuple(np.round(own, 6))}) needs nosym = "
                ".true.: a magnetic noncollinear run reduces its k-grid with "
                "the magnetic symmetry group, which depends on where the "
                "moment points, so two directions would be sampled on two "
                "different wedges and their band energies would differ by the "
                "k-sampling rather than by the physics. QE's own force-theorem "
                "example sets nosym for this reason"
            )

    direction = np.asarray(direction, dtype=float)
    direction = tuple(float(x) for x in direction / np.sqrt(np.sum(direction**2)))
    system = _with_quantization_axis(system, direction)

    rotated = nc_magnetization_from_lsda(density, direction)
    # The one-centre occupations follow the density onto the same axis, and
    # they are handed the *density* to read the old axis off: a species' own
    # ``becsum`` knows which way that species points, which is the wrong
    # question, an antiferromagnet's two sublattices pointing opposite ways
    # while the cell has one frame to rotate.
    rotated_becsum = tuple(
        None if values is None
        else nc_magnetization_from_lsda(values, direction, axis_from=density)
        for values in becsum
    )

    calculation, system, eigenvalues, wavefunctions = fixed_density_states(
        system, pseudos, rotated, nbnd=nbnd, conv_thr=conv_thr, k_batch=k_batch,
        becsum=rotated_becsum,
    )

    wg, levels = calculation.occupations(jnp.asarray(eigenvalues))
    wg = np.asarray(wg)
    # ``print_ks_energies.f90:91-96``: the bare sum over bands and k-points,
    # with no entropy term and no reference level. The weights already carry
    # the k-point weight, which is QE's ``wg``.
    band_energy = float(np.sum(wg * np.asarray(eigenvalues)))

    result = ForceTheorem(
        direction=direction,
        band_energy=band_energy,
        fermi_energy=levels.get("fermi_energy"),
        entropy=float(levels.get("smearing", 0.0)),
        eigenvalues=np.asarray(eigenvalues),
        occupations=wg,
        kpoints=system.kpoints,
    )
    if projected:
        if ef_0 is None:
            # ``projwfc.x`` makes ``ef_0`` an explicit input because the
            # decomposition is meant to be read with **one** reference level
            # shared by every direction. Defaulting to this run's own Fermi
            # level is right for a single direction and is what a scan overrides.
            ef_0 = result.fermi_energy
            if ef_0 is None:
                ef_0 = 0.0 if result.eigenvalues is None else float(
                    np.max(np.asarray(eigenvalues)[np.asarray(wg) > 0.0])
                )
        result.projected = _project_band_energy(
            calculation, system, eigenvalues, wg, wavefunctions, ef_0,
        )
    return result


def sphere_cover(n: int) -> tuple:
    """``n`` directions covering the unit sphere nearly optimally.

    Elk's ``sphcover.f90``, the golden-section formula
    ``theta_k = acos(1 - (k - 1/2) dz)``, ``phi_k = (k - 1) dphi`` with
    ``dz = 2/n`` and ``dphi = pi (1 - sqrt 5)`` -- what ``gentpmae`` generates
    for ``npmae >= 4``. Returned as cartesian unit vectors rather than as
    ``(theta, phi)``, because that is what the rotation takes.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"sphere_cover wants n >= 1, got {n}")
    dz = 2.0 / n
    dphi = np.pi * (1.0 - np.sqrt(5.0))
    k = np.arange(n)
    z = 1.0 - dz / 2.0 - k * dz
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.mod(k * dphi, 2.0 * np.pi)
    return tuple(
        (float(np.sin(t) * np.cos(f)), float(np.sin(t) * np.sin(f)), float(np.cos(t)))
        for t, f in zip(theta, phi)
    )


def cardinal_directions(system: System, n: int = 1) -> tuple:
    """The symmetry-inequivalent lattice directions ``n_1 a_1 + n_2 a_2 + n_3 a_3``.

    ``gentpmae``'s ``npmae`` in ``-4:-1``, with ``|npmae| = n``: every integer
    combination with ``|n_i| <= n`` except the origin, reduced by the crystal's
    point group so that two directions related by a symmetry -- which must have
    the same band energy -- are not both computed.

    The group is the *crystal's*, taken from the system rather than
    rediscovered, and it is the group of the **lattice and the basis**, not the
    magnetic group: what is being reduced here is a set of candidate moment
    directions, and the magnetic group of one of them is a statement about that
    candidate rather than about the crystal.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"cardinal_directions wants n >= 1, got {n}")
    at = np.asarray(system.cell.at, dtype=float)
    rotations = system.symmetry_group().rotation_array()

    seen: list[np.ndarray] = []
    directions: list[tuple] = []
    span = range(-n, n + 1)
    for i1 in span:
        for i2 in span:
            for i3 in span:
                if i1 == i2 == i3 == 0:
                    continue
                lattice = np.array([i1, i2, i3], dtype=float)
                # ``gentpmae`` compares the *lattice* coordinates under the
                # integer rotation matrices, then converts once at the end --
                # and the matrix is the **transpose**. ``Symmetries``' own class
                # docstring fixes the convention: ``rotations[s]`` is ``M`` with
                # ``S a_i = sum_j M_ij a_j``, so a *direct*-lattice coordinate
                # vector goes to ``M^T n`` while a Miller index goes to ``M m``
                # (``kpoints.py`` rotates ``k`` as ``xkg @ rotation.T``, which is
                # the same statement). ``cartesian`` two lines below is
                # ``at.T @ lattice``, a direct-lattice vector, so this is the
                # first of the two. ``{M}`` and ``{M^T}`` are both groups of the
                # same order, so the reduction still partitioned the candidates
                # into orbits and nothing raised -- they were the orbits of a
                # different action, and they coincide only where the matrices
                # are signed permutations, which is cubic, tetragonal and
                # orthorhombic.
                images = [rotation.T @ lattice for rotation in rotations]
                if any(
                    any(np.sum(np.abs(image - other)) < 1.0e-6 for other in seen)
                    for image in images
                ):
                    continue
                seen.append(lattice)
                cartesian = at.T @ lattice
                norm = float(np.sqrt(np.sum(cartesian**2)))
                directions.append(tuple(float(x) for x in cartesian / norm))
    return tuple(directions)


def run_anisotropy(
    system: System,
    pseudos: tuple[Pseudopotential, ...],
    density: jnp.ndarray,
    directions=None,
    nbnd: int | None = None,
    conv_thr: float = 1.0e-10,
    k_batch: int | None | str = "default",
    projected: bool = False,
    soc_scale: float | None = None,
    becsum: tuple = (),
) -> MagneticAnisotropy:
    """The band energy of every direction in ``directions``, and their spread.

    ``directions`` is a sequence of cartesian vectors (normalised here), an
    integer ``n`` asking for :func:`sphere_cover`, or one of the strings
    ``"cardinal"``, ``"xz"`` and ``"xyz"`` -- ``gentpmae``'s conventions. It
    defaults to ``"xyz"``, the three cartesian axes, which is the smallest set
    that separates a uniaxial anisotropy from a cubic one.

    Every direction is diagonalised on the **same** k-set from the **same**
    density, so what differs between two entries is the spin-orbit coupling and
    nothing else -- which is the whole reason the difference of two numbers of
    order 100 Ry can be trusted in its eighth decimal.
    """
    directions = _direction_set(system, directions)

    results = tuple(
        run_force_theorem(
            system, pseudos, density, direction=direction, nbnd=nbnd,
            conv_thr=conv_thr, k_batch=k_batch, projected=projected,
            soc_scale=soc_scale, becsum=becsum,
        )
        for direction in directions
    )
    return MagneticAnisotropy(
        directions=tuple(r.direction for r in results), results=results
    )


def _project_band_energy(calculation, system, eigenvalues, wg, wavefunctions, ef_0):
    """``force_theorem``'s decomposition, ``projwfc.f90:565-584``.

    ``eband_proj[nwfc] = sum_ik w_ik (eps_ik - ef_0) |<phi_nwfc|S|psi_ik>|^2``.

    **The basis is the one line of this worth stating**, because QE has two and
    picks the *other* one here: ``force_theorem`` projects on
    ``atomic_wfc_nc_updown`` (``projwfc.f90:898``, whose own comment reads "to
    project on real harmonics, not on spinors"), which is a real spherical
    harmonic times a pure up or down spinor -- **not** the ``j``-resolved
    ``atomic_wfc_nc_proj`` that a spin-orbit projected DOS uses. That is the
    whole point of the decomposition: it says which ``l``, ``m`` and *spin*
    supplies the anisotropy, and ``m`` and the spin are not good labels in the
    ``j`` basis. :meth:`~defumat.scf.driver.Calculation._as_spinors` is that
    set.

    The overlap operator is the spinor one with ``qq_so``
    (:meth:`~defumat.scf.driver.Calculation._spinor_overlap`), which is what
    makes this work on the fully-relativistic ultrasoft dataset the reference
    case uses.

    The column *order* is this package's and not QE's -- ``_as_spinors`` stacks
    every up column then every down column, where ``atomic_wfc_nc_updown``
    interleaves them shell by shell -- so a comparison against ``filproj`` goes
    through the labels rather than through the index.
    """
    from defumat.hubbard.projectors import _apply_transform, lowdin_transform
    from defumat.projwfc.channels import projection_channels
    from defumat.pseudo.atomic import atomic_wavefunctions

    channels = projection_channels(calculation.pseudos, system.structure)
    if not channels:
        raise ValueError(
            "none of the pseudopotentials carries an atomic orbital to project "
            "on -- projwave refuses the same way"
        )

    atomic = atomic_wavefunctions(
        calculation.pseudos, system.structure, system.cell,
        calculation.basis.smooth, calculation.basis.planewaves,
        calculation.basis_kpoints,
    )
    spinors = calculation._as_spinors(atomic)  # (nk, 2 n, 2 npwx)

    psi = jnp.asarray(wavefunctions)[0]  # (nk, nbnd, 2 npwx)
    eigenvalues = np.asarray(eigenvalues)[0]
    wg = np.asarray(wg)[0]
    weight = wg * (eigenvalues - float(ef_0))  # (nk, nbnd)

    total = np.zeros(spinors.shape[1])
    for ik in range(spinors.shape[0]):
        phi = spinors[ik]
        sphi = calculation._spinor_overlap(phi, ik)
        overlap = jnp.conj(phi) @ sphi.T
        transform = lowdin_transform(overlap)
        projectors = _apply_transform(transform, jnp.transpose(sphi, (1, 0)))
        proj0 = jnp.einsum("gi,bg->ib", jnp.conj(projectors), psi[ik])
        # ``lsym`` is refused with ``lforcet`` (``projwfc.f90:152``) and the run
        # is ``nosym`` anyway, so there is no ``sym_proj_k`` average here.
        total += np.asarray(jnp.abs(proj0) ** 2 @ jnp.asarray(weight[ik]))

    count = len(channels)
    labels = tuple(
        (channel, spin)
        for spin in ("up", "down")
        for channel in channels
    )
    by_atom = np.zeros(system.structure.nat)
    for (channel, _), value in zip(labels, total):
        by_atom[channel.atom] += value
    assert len(labels) == total.size == 2 * count
    return ProjectedBandEnergy(
        by_orbital=total, labels=labels, ef_0=float(ef_0), by_atom=by_atom
    )


def angles_from_direction(direction) -> tuple:
    """``(angle1, angle2)`` in degrees from a cartesian unit vector.

    The inverse of :func:`~defumat.scf.continuation.direction_from_angles`.
    """
    x, y, z = (float(v) for v in direction)
    return float(np.degrees(np.arccos(np.clip(z, -1.0, 1.0)))), float(
        np.degrees(np.arctan2(y, x))
    )


def _reference_axis(system: System) -> tuple:
    """The direction a system's magnetization is said to point along.

    **Where the first magnetic atom's starting moment points, sign included**,
    and the same rule whether the moments come from a ``STARTING_MOMENTS`` card
    or from the per-species ``starting_magnetization``/``angle1``/``angle2``:
    both are folded into one ``(nat, 3)`` array by
    :func:`~defumat.system.builder.local_moments`, and the axis is its first row
    whose length passes ``setup.f90``'s ``domag`` threshold of 1e-6. Every
    default direction in this module is this vector, so a call that names no
    direction is exactly the identity rotation, and a named direction means
    "turn the texture until the first magnetic atom points there".

    Until 2026-09-23 the two routes disagreed. With no card the axis was
    species one's angles, whatever the sign of its ``starting_magnetization``
    and whether or not it carried a moment at all, so an oxide listed with O
    first had the axis ``z`` from O's default angles while the metal's moments
    lay along ``x``, and a direction named along ``x`` rotated them by 90
    degrees onto ``-z``. With a card it was the first nonzero row, sign
    included. The signed rule is the card's, extended to the species.

    The angles are read at ``nspin = 4`` whatever this system's regime, since
    the axis is a statement about the noncollinear run the texture is turned
    in, and :func:`~defumat.system.builder.local_moments` puts a collinear
    run's moments on ``z`` by construction. A cell with no moment anywhere
    falls back to species one's angles, the only direction it states.

    The sign is a convention and nothing measured depends on it: turning every
    moment over is time reversal, which leaves a band energy and its
    derivative in the angle unchanged.
    """
    rows = local_moments(
        system.structure, 4, system.starting_magnetization,
        system.angle1, system.angle2, per_atom=system.starting_moments,
    )
    rows = np.asarray(rows, dtype=float).reshape(-1, 3)
    norms = np.sqrt(np.sum(rows**2, axis=1))
    nonzero = np.flatnonzero(norms > 1.0e-6)
    if len(nonzero):
        row = rows[nonzero[0]] / norms[nonzero[0]]
        return tuple(float(x) for x in row)
    return direction_from_angles(system.angle1[0], system.angle2[0])


def _rotation_taking(source, target) -> np.ndarray:
    """The proper rotation that takes the unit vector ``source`` onto ``target``.

    The one about ``source x target`` (Rodrigues), which is the smallest; for
    ``target = -source`` there is no preferred axis and the rotation by ``pi``
    about the coordinate axis least aligned with ``source`` is taken, so the
    choice is deterministic. Host-side: the axis of a noncollinear run is
    static input, not something a gradient passes through.
    """
    a = np.asarray(source, dtype=float)
    b = np.asarray(target, dtype=float)
    a = a / np.sqrt(np.sum(a**2))
    b = b / np.sqrt(np.sum(b**2))
    axis = np.cross(a, b)
    sine = float(np.sqrt(np.sum(axis**2)))
    cosine = float(a @ b)
    if sine < 1.0e-12:
        if cosine > 0.0:
            return np.eye(3)
        helper = np.zeros(3)
        helper[int(np.argmin(np.abs(a)))] = 1.0
        axis = np.cross(a, helper)
        axis = axis / np.sqrt(np.sum(axis**2))
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    axis = axis / sine
    cross = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + sine * cross + (1.0 - cosine) * (cross @ cross)


def _clean(vector) -> np.ndarray:
    """Round-off below 1e-14 set to zero, so that a moment rotated into a
    plane lies in it exactly and ``fixed_quantization_axis`` sees parallel
    rows as parallel."""
    vector = np.asarray(vector, dtype=float)
    return np.where(np.abs(vector) < 1.0e-14, 0.0, vector)


def _with_quantization_axis(system: System, direction) -> System:
    """The same run with its magnetic texture turned rigidly onto ``direction``.

    **A rigid rotation of the whole texture, not one pair of angles for every
    species.** The rotation is the one taking :func:`_reference_axis` onto
    ``direction``, and it is applied to every species' ``angle1``/``angle2``
    and to every row of a ``STARTING_MOMENTS`` card, so the angles between
    moments are kept: an antiferromagnet stays antiparallel and a canted cell
    keeps its cant. Until 2026-09-23 this wrote ``(angle1,) * ntyp``, which
    turned ``fe2-afm-soc.in``'s ``(+0.5, 0, 0), (-0.5, 0, 0)`` into
    ``(0.5, 0, 0), (0.5, 0, 0)`` -- a ferromagnet -- on every rotation, and
    left a ``STARTING_MOMENTS`` card untouched, where it overrides the angles
    and so kept both the seed and the fixed axis on the old direction. This is
    the operation :func:`~defumat.scf.continuation.nc_magnetization_from_lsda`
    performs on the density, which maps signed ``m_z`` onto ``+/- direction``.

    **Not cosmetic, and this is the trap of the whole phase.** A
    gradient-corrected noncollinear run does not evaluate the functional on
    ``|m|``: it takes the *signed* projection ``s = sign(m . u_x)`` on a fixed
    axis (``compute_ux``, :func:`~defumat.scf.potential.
    fixed_quantization_axis`) so that "up" stays up across a node where ``m``
    changes sign -- because ``|m|`` has a **kink** there and a GGA reads the
    gradient of that kink. The axis is built from ``starting_magnetization``
    and ``angle1``/``angle2``, so rotating the *density* and leaving the
    *system* alone leaves the two disagreeing: for a moment turned into the
    ``xy`` plane, ``m . z`` is zero everywhere, ``s`` sticks at ``+1``, and
    what the functional differentiates is ``|m|`` with every one of its kinks.

    Measured on a one-atom cubic Co cell, where every direction must give the
    same band energy by symmetry: **36.8 meV** between ``x`` and ``z`` with the
    axis left behind, and 1e-14 with it carried -- and it is *not* a
    spin-orbit effect at all, since it survives switching the coupling off
    entirely, which is what identified it. The ``abs`` trap of ``PLAN.md``
    P28a, in a fifth place.

    Rebuilding the k-points is what :meth:`~defumat.system.builder.System.
    with_spin` would otherwise cost here, and it is exactly what must not
    happen between two directions. **The argument that it is safe is
    ``nosym``**, and until 2026-09-21 that argument was made in this docstring
    and enforced at two of the four call sites. :func:`run_force_theorem`
    checks it and the relaxed path checks it; :func:`run_torque` and
    :func:`frozen_expectation` did not, and both rotate the axis essentially
    always -- ``run_torque``'s direction is ``cos(angle) first + sin(angle)
    second`` at a default angle of ``pi/4``. So the check is here now, where the
    rebuild is, and a fifth call site cannot miss it.

    ``with_spin`` routes into ``System._respin_kpoints``, which for a magnetic
    noncollinear run takes the *magnetic* group of the **new** angles, sets
    ``time_reversal = False`` and rebuilds ``KPoints.automatic`` on it. Two
    directions then arrive on two different wedges, which is the k-sampling
    rather than the physics; and a torque differentiates
    ``sum_k w_k <psi|H(theta)|psi>`` in ``theta``, whose ``dH/dtheta`` carries
    an axial vector perpendicular to the direction the group was built around
    and therefore not invariant under it -- a response on a reduced k-set, with
    nothing symmetrising it.
    """
    own = np.asarray(_reference_axis(system), dtype=float)
    wanted = np.asarray(direction, dtype=float)
    wanted = wanted / np.sqrt(np.sum(wanted**2))
    if np.sum(np.abs(own - wanted)) <= DIRECTION_TOL:
        # Already pointing there -- which is the ordinary case, the direction
        # having defaulted to the system's own axis. Returned untouched so
        # that a single-direction run never rebuilds its k-points at all.
        return system
    _require_nosym_to_turn(system, own)
    return _turn(system, _rotation_taking(own, wanted), wanted)


def _with_rotation(system: System, rotation) -> System:
    """The same run with its magnetic texture turned rigidly by ``rotation``.

    :func:`_with_quantization_axis` for a rotation given whole rather than as
    the smallest one taking the reference axis onto a direction: the two agree
    whenever the rotation is that one, and this one also carries a turn about
    the reference axis, which a texture that is not collinear feels. Every
    reason that function's docstring gives applies here unchanged, the
    quantization axis and the ``nosym`` requirement above all.
    """
    rotation = np.asarray(rotation, dtype=float)
    if np.max(np.abs(rotation - np.eye(3))) <= DIRECTION_TOL:
        return system
    own = np.asarray(_reference_axis(system), dtype=float)
    _require_nosym_to_turn(system, own)
    wanted = rotation @ own
    return _turn(system, rotation, wanted / np.sqrt(np.sum(wanted**2)))


def _require_nosym_to_turn(system: System, own) -> None:
    """The refusal :func:`_with_quantization_axis` explains, in one place."""
    if not system.nosym:
        own = tuple(np.round(own, 6))
        raise ValueError(
            f"turning the quantization axis away from the system's own "
            f"angle1/angle2 ({own}) needs nosym = .true.: a magnetic "
            "noncollinear run reduces its k-grid with the magnetic symmetry "
            "group, which depends on where the moment points, so the rotated "
            "run would be sampled on a different wedge from the one it is "
            "compared against -- and a torque is a derivative of a wedge sum "
            "with respect to a direction the wedge is not symmetric in. QE's "
            "own force-theorem example sets nosym for this reason"
        )


def _turn(system: System, rotation, wanted) -> System:
    """Every species' angles and the ``STARTING_MOMENTS`` card, turned by ``rotation``.

    ``wanted`` is where the reference axis lands, ``rotation @ own``: a species
    along that axis, or against it, is put on ``+/- wanted`` exactly rather than
    on its rotated image, which differs from it by round-off.
    """
    # Every species, magnetic or not: a species with no moment has angles that
    # do nothing, and rotating them too keeps the rule one line long. The
    # magnitudes, signs included, stay on ``starting_magnetization``.
    count = max(len(system.angle1), len(system.angle2),
                len(system.starting_magnetization))
    theta = list(system.angle1) + [0.0] * (count - len(system.angle1))
    phi = list(system.angle2) + [0.0] * (count - len(system.angle2))
    angle1, angle2 = [], []
    for t in range(count):
        turned = _clean(rotation @ np.asarray(direction_from_angles(theta[t], phi[t])))
        # A species along the reference axis, or against it, lands on
        # ``+/- wanted`` itself rather than on its rotated image, so that a
        # collinear cell -- every committed Co reference -- gets exactly the
        # angles ``angles_from_direction(wanted)`` it got before the rotation
        # was written, and not a copy of them one Rodrigues product away.
        if np.sqrt(np.sum(np.cross(turned, wanted) ** 2)) < 1.0e-12:
            turned = np.sign(float(turned @ wanted)) * wanted
        first, second = angles_from_direction(turned)
        angle1.append(first)
        angle2.append(second)
    if len(system.starting_moments):
        rows = np.asarray(system.starting_moments, dtype=float).reshape(-1, 3)
        # ``with_moments`` first, ``with_spin`` second: each rebuilds the
        # k-points from the state it is handed, and the card overrides the
        # angles in ``local_moments``, so the second rebuild sees both.
        system = system.with_moments(_clean(rows @ rotation.T))
    return system.with_spin(angle1=tuple(angle1), angle2=tuple(angle2))


def frozen_expectation(
    system: System,
    pseudos: tuple[Pseudopotential, ...],
    density: jnp.ndarray,
    direction=None,
    nbnd: int | None = None,
    conv_thr: float = 1.0e-10,
    k_batch: int | None | str = "default",
) -> float:
    """The first-order term the force theorem exists because it cannot use.

    Take the states of the **coupling-free** Hamiltonian (``soc_scale = 0``)
    with the magnetization already rotated onto ``direction``, freeze them, and
    evaluate the spin-orbit term's expectation value once:

        E1(n) = sum_occ w [ <psi| dV_NL |psi> - eps <psi| dS |psi> ],

    with ``dV_NL`` and ``dS`` the coupled Hamiltonian's nonlocal coefficients
    and overlap minus the reduced ones, at the same frozen potential. On an
    ultrasoft dataset ``dV_NL`` has two pieces: the spin-*traceless* half of
    ``dvan_so`` (:func:`~defumat.pseudo.spinorbit.spin_trace`), and
    ``newd_so``'s sandwich of the augmentation integrals against its spin
    trace, ``F B F - T(B)`` (``scf/driver.py:_newd_noncollinear`` at 1 against
    0), taken on the **total** local potential the states were computed in.
    ``dS`` is the traceless half of ``qq_so``, and the ``eps dS`` piece is there
    because an ultrasoft eigenproblem is generalised and the metric is perturbed
    too. ``soc_scale`` blends all three linearly, so ``E1`` is exactly the
    derivative of the free energy with respect to ``soc_scale`` at zero, by
    Hellmann-Feynman at frozen occupations. A norm-conserving dataset has the
    first piece alone.

    **This is the calculation the force theorem is often assumed to be, and it
    is small rather than zero.** Measured at ``conv_thr = 1e-10``
    (``PLAN.md`` P120): on tetragonal cobalt (``co-tetragonal-anisotropy-*.in``)
    it is +1.1408e-2 meV along ``x`` and +0.9623e-2 along ``z``, so 1.79e-3 meV
    of first-order anisotropy with the easy axis along ``c``, where
    :func:`run_force_theorem` on the same density gives 0.552 meV of free-energy
    anisotropy (1.235 in the band energy). On the cubic smoke cell it is
    +1.2605e-2 meV in every direction, isotropic to 1.1e-7 meV as symmetry
    requires.

    **Which part survives the quenched orbital moment, measured by splitting
    the sandwich's input.** The bare ``dvan_so`` and the overlap together, and
    the charge component of ``newd_so`` on its own, are each at 1e-6 meV and
    nearly cancel; the exchange components carry all of the rest. The
    reason is time reversal. An operator that is time-even as a whole has a
    spin-dependent part ``sigma . O`` with ``O`` time-odd in the orbitals, like
    ``L``, and a collinear state without spin-orbit coupling has real orbitals,
    so ``<O>`` vanishes: that is the argument, and it covers the first three.
    The exchange field's term is time-odd as a whole, so ``O`` is time-even,
    ranks 0 and 2, and real orbitals do not annul it; with the field and the
    spin both along ``n`` the result is a quadratic form ``n . K . n``, whose
    trace is the cubic cell's isotropic shift and whose traceless part is the
    tetragonal cell's uniaxial term. By the completeness argument in
    :func:`~defumat.pseudo.spinorbit.spin_trace` the sandwich would equal its
    trace if the two ``j`` shells shared a radial function, so the term is the
    dataset's ``j`` splitting seen by the exchange field. That is an argument
    fitted to two cells, not a third measurement.

    It is a function rather than a footnote because a measured number is a
    stronger statement than an argument about its size -- the argument said
    zero, and it held for three of the four terms -- and because the
    ``soc_scale`` knob is what makes it well posed: a pseudopotential has no
    additive ``xi L.S`` operator to take the expectation value *of*, only the
    parts of its coefficients that a spin trace does not keep. PAW is refused
    with the rest of :func:`_refuse_system`: its one-centre terms carry
    ``soc_scale`` too (``build_paw``'s small component) and are not in ``E1``.
    """
    from defumat.forces.energy import _spinor_projector_energies
    from defumat.scf.potential import as_potential_components

    _refuse_system(system, pseudos)
    if direction is None:
        direction = _reference_axis(system)
    direction = np.asarray(direction, dtype=float)
    direction = tuple(float(x) for x in direction / np.sqrt(np.sum(direction**2)))
    system = _with_quantization_axis(system, direction).with_soc_scale(0.0)

    rotated = nc_magnetization_from_lsda(density, direction)
    calculation, system, eigenvalues, wavefunctions = fixed_density_states(
        system, pseudos, rotated, nbnd=nbnd, conv_thr=conv_thr, k_batch=k_batch,
    )
    wg, _ = calculation.occupations(jnp.asarray(eigenvalues))

    # The potential the states were computed in, and it is the **total** local
    # potential, the one ``Calculation.hamiltonian`` hands ``coefficients``
    # (``set_vrs``): ``v_scf`` alone leaves ``vltot`` out of the charge
    # component of ``newd_so``'s integrals and moves the cubic cell's spread
    # from 1e-7 meV to 1.2e-5.
    total = calculation.potential(rotated).v_scf + as_potential_components(
        calculation.vltot, calculation.nspin_mag
    )
    delta_d, delta_qq = _first_order_operator(calculation, total)
    nonlocal_, overlap = _spinor_projector_energies(
        jnp.asarray(wavefunctions), calculation.projectors.vkb,
        delta_d, delta_qq, jnp.asarray(wg), jnp.asarray(eigenvalues),
    )
    return float(nonlocal_ - overlap)


def _first_order_operator(calculation, total) -> tuple:
    """``(dV_NL, dS)``: the coupled spinor coefficients minus the reduced ones.

    ``calculation`` is the ``soc_scale = 0`` one and ``total`` the local
    potential its Hamiltonian was built at. ``dV_NL`` is ``dvan_so``'s
    traceless half plus, on an augmented dataset, ``newd_so``'s sandwich of
    ``int V_c Q_ij`` against its spin trace; ``dS`` is ``qq_so``'s traceless
    half, and ``None`` without augmentation. The two together are the whole of
    what ``soc_scale`` changes in an ultrasoft Hamiltonian at a frozen
    potential, which ``test_anisotropy.py`` holds entry by entry against a
    coupled ``Calculation``'s own ``coefficients`` and ``qq_so`` -- the check a
    missing term fails, and ``newd_so``'s was missing until P120.
    """
    from defumat.pseudo.spinorbit import build_spin_orbit
    from defumat.scf.driver import _newd_noncollinear, _spin_block_diagonal

    pseudos = calculation.pseudos
    types = calculation.system.structure.types
    coupled = build_spin_orbit(pseudos, 1.0)
    free = build_spin_orbit(pseudos, 0.0)
    delta_d = jnp.asarray(
        _spin_block_diagonal([coupled[t].dvan_so for t in types])
    ) - jnp.asarray(_spin_block_diagonal([free[t].dvan_so for t in types]))
    if calculation.qq_so is None:
        return delta_d, None

    augmentation = calculation.augmentation

    def species_qq(t: int) -> np.ndarray:
        nh = pseudos[t].nh
        values = np.asarray(augmentation.qq[t])
        return values if values.shape == (nh, nh) else np.zeros((nh, nh))

    delta_qq = jnp.asarray(
        _spin_block_diagonal([coupled[t].qq_so(species_qq(t)) for t in types])
    ) - jnp.asarray(
        _spin_block_diagonal([free[t].qq_so(species_qq(t)) for t in types])
    )
    # ``fcoef`` is the same zeroed array at both scales; taking it from the
    # coupled set rather than from ``calculation`` is what lets that test tell
    # the two apart if it ever stops being true.
    components, _ = calculation._noncollinear_components(total)
    fcoef = jnp.asarray(_spin_block_diagonal([coupled[t].fcoef for t in types]))
    bare = jnp.zeros_like(fcoef)
    delta_d = delta_d + (
        _newd_noncollinear(components, bare, fcoef, 1.0)
        - _newd_noncollinear(components, bare, fcoef, 0.0)
    )
    return delta_d, delta_qq


@dataclass
class MagneticTorque:
    """``-dE/dtheta`` at one angle, and the anisotropy constant it implies."""

    #: Radians from the plane's first axis.
    angle: float
    #: ``-dF/dtheta`` in Ry per radian, ``F`` the **free** energy (the band
    #: energy plus the smearing's ``-TS``). That is what a Hellmann-Feynman
    #: derivative at frozen occupations gives, and for a smeared metal it is
    #: *not* the derivative of ``sum w eps`` --
    #: :attr:`MagneticAnisotropy.free_energies` says how far apart they are.
    torque: float
    #: The plane the moment was turned in, as the pair of unit vectors.
    plane: tuple
    #: ``sum w eps`` at this angle, in Ry -- what the gradient was taken of.
    band_energy: float
    #: The same sum rebuilt as ``sum w <psi|H|psi>``. Equal to
    #: :attr:`band_energy` to the eigensolver's residual, and the check that
    #: the quadratic form the gradient runs on is the right one.
    band_energy_check: float
    fermi_energy: float | None = None

    @property
    def torque_mev(self) -> float:
        return self.torque * RY_TO_EV * 1000.0

    @property
    def anisotropy_constant(self) -> float:
        """``K1`` in Ry, for a uniaxial magnet measured at 45 degrees.

        ``E = K1 sin^2(theta)`` gives ``-dE/dtheta = -K1 sin(2 theta)``, so at
        ``pi/4`` the torque *is* ``-K1``. Away from 45 degrees this divides by
        ``sin(2 theta)``, which is the same statement and is why 45 is the
        angle the method is always quoted at: it is where the division is by
        one, and where the fourth-order term ``K2 sin^4`` contributes least to
        the ratio.
        """
        return -self.torque / np.sin(2.0 * self.angle)

    @property
    def anisotropy_constant_mev(self) -> float:
        return self.anisotropy_constant * RY_TO_EV * 1000.0

    @property
    def residual(self) -> float:
        """How far the two band energies are apart, in Ry."""
        return abs(self.band_energy - self.band_energy_check)


def run_torque(
    system: System,
    pseudos: tuple[Pseudopotential, ...],
    density: jnp.ndarray,
    angle: float = np.pi / 4.0,
    plane=((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    nbnd: int | None = None,
    conv_thr: float = 1.0e-10,
    k_batch: int | None | str = "default",
    soc_scale: float | None = None,
) -> MagneticTorque:
    """The magnetic torque, and through it the anisotropy from **one** angle.

    :func:`run_anisotropy` takes the anisotropy as a difference of two band
    energies, which is 1e-5 Ry out of 1e2 -- seven digits of cancellation. This
    takes it as a *derivative* instead, evaluated once, where nothing cancels
    (:mod:`defumat.forces.torque`). For a uniaxial magnet
    ``E = K1 sin^2(theta)``, so the torque at ``pi/4`` is ``-K1`` and
    :attr:`MagneticTorque.anisotropy_constant` reads it off.

    ``plane`` is the orthonormal pair the moment turns in; the default turns it
    from ``z`` towards ``x``, so ``angle = 0`` is along ``z``. ``density`` is
    the **collinear** density of a scalar-relativistic run, exactly as
    :func:`run_force_theorem` takes it, and everything that function refuses is
    refused here for the same reasons.
    """
    from defumat.forces.torque import band_energy_at_angle, torque_at_angle

    if soc_scale is not None:
        system = system.with_soc_scale(soc_scale)
    _refuse_system(system, pseudos)

    first, second = (np.asarray(v, dtype=float) for v in plane)
    first, second = first / np.linalg.norm(first), second / np.linalg.norm(second)
    if abs(float(first @ second)) > 1.0e-8:
        raise ValueError(
            f"the two axes of the rotation plane are not orthogonal "
            f"(dot product {float(first @ second):.3e}); the angle would not "
            "parameterise a rotation and its derivative would not be a torque"
        )
    angle = float(angle)
    direction = np.cos(angle) * first + np.sin(angle) * second

    # The quantization axis follows the moment, exactly as it must for
    # ``run_force_theorem`` -- and here it is also what makes the derivative
    # clean, since a *static* axis is a constant the gradient passes through.
    system = _with_quantization_axis(system, tuple(direction))
    rotated = nc_magnetization_from_lsda(density, tuple(direction))
    calculation, system, eigenvalues, wavefunctions = fixed_density_states(
        system, pseudos, rotated, nbnd=nbnd, conv_thr=conv_thr, k_batch=k_batch,
    )
    wg, levels = calculation.occupations(jnp.asarray(eigenvalues))

    plane_pair = (tuple(float(v) for v in first), tuple(float(v) for v in second))
    check = float(band_energy_at_angle(
        calculation, wavefunctions, wg, density, plane_pair, angle
    ))
    # ``k_batch`` reaches the *derivative* too, not only the NSCF above it: the
    # backward pass holds one real-space block per k-point in flight, which is
    # where this costs more than the run that produced the states.
    value = torque_at_angle(
        calculation, wavefunctions, wg, density, plane_pair, angle,
        k_batch=k_batch,
    )
    return MagneticTorque(
        angle=angle,
        torque=value,
        plane=plane_pair,
        band_energy=float(np.sum(np.asarray(wg) * np.asarray(eigenvalues))),
        band_energy_check=check,
        fermi_energy=levels.get("fermi_energy"),
    )


# ---------------------------------------------------------------------------
# The torque on a whole texture: one rotation of every spin, three generators
# ---------------------------------------------------------------------------


def rotation_from_euler(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """``Rz(alpha) Ry(beta) Rz(gamma)``, the ZYZ convention, angles in radians.

    An active rotation: it turns the texture, and ``beta`` is the angle the
    reference axis ``z`` is tilted through. Euler angles are how an orientation
    is given and read, not what anything is differentiated in: at ``beta = 0``
    the angles ``alpha`` and ``gamma`` are the same rotation, and that is an
    easy axis along ``z``, where a uniaxial answer usually sits.
    """
    def about_z(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    c, s = np.cos(beta), np.sin(beta)
    about_y = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return about_z(alpha) @ about_y @ about_z(gamma)


def euler_from_rotation(rotation) -> tuple:
    """The ZYZ angles of :func:`rotation_from_euler`, in radians.

    ``beta`` in ``[0, pi]``. Where ``beta`` is 0 or ``pi`` only ``alpha +/-
    gamma`` is defined, and ``gamma = 0`` is returned.
    """
    r = np.asarray(rotation, dtype=float)
    beta = float(np.arccos(np.clip(r[2, 2], -1.0, 1.0)))
    if np.sin(beta) > 1.0e-12:
        alpha = float(np.arctan2(r[1, 2], r[0, 2]))
        gamma = float(np.arctan2(r[2, 1], -r[2, 0]))
    else:
        alpha, gamma = float(np.arctan2(r[1, 0], r[0, 0])), 0.0
        if r[2, 2] < 0.0:
            alpha = float(np.arctan2(-r[1, 0], -r[0, 0]))
    return alpha, beta, gamma


def _checked_rotation(rotation) -> np.ndarray:
    """A proper rotation matrix, or a refusal naming what it is instead."""
    if rotation is None:
        return np.eye(3)
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(f"a rotation is a 3x3 matrix, got shape {rotation.shape}")
    error = float(np.max(np.abs(rotation @ rotation.T - np.eye(3))))
    if error > 1.0e-10:
        raise ValueError(
            f"the rotation is not orthogonal (|R R^T - 1| = {error:.3e}); use "
            "rotation_from_euler to build one"
        )
    if np.linalg.det(rotation) < 0.0:
        raise ValueError(
            "the rotation has determinant -1: an improper rotation is not a "
            "turn of the texture, and the magnetization is an axial vector, so "
            "it would not act on it the way the matrix says. Reversing every "
            "moment is time reversal, which leaves the energy unchanged"
        )
    return rotation


def _reference_texture(density, own) -> jnp.ndarray:
    """The four-channel texture at the reference orientation, the one ``R = 1`` means.

    A collinear source, two channels or four, has its moment laid along the
    system's own axis ``own`` with its sign, exactly as the force theorem lays
    it, so that a collinear run here is P58's and P60's rotation. A
    four-channel source whose moments are **not** collinear is already a
    texture in the orientation it converged in, and is returned as it is:
    there is no axis to read off it, and :func:`rotate_texture` needs none.
    """
    density = jnp.asarray(density)
    if density.shape[0] == 4 and not _is_collinear(density):
        return density
    return nc_magnetization_from_lsda(density, own)


def _is_collinear(density) -> bool:
    """Whether a four-channel magnetization lies along one axis.

    ``_collinear_axis``'s test, the two smaller eigenvalues of
    ``M_ab = integral of m_a m_b`` against the largest, asked as a question
    rather than raised as a refusal.
    """
    from defumat.scf.continuation import TRANSVERSE_TOL

    moment = np.asarray(jnp.real(jnp.asarray(density)[1:4])).reshape(3, -1)
    values = np.linalg.eigvalsh(moment @ moment.T)
    return float(values[0] + values[1]) <= TRANSVERSE_TOL * float(values[2])


@dataclass
class OrientationTorque:
    """``-dF/dw`` for a rigid rotation of the whole texture, at one orientation."""

    #: ``R0``, the orientation the states were diagonalised at, relative to the
    #: system's own texture.
    rotation: np.ndarray
    #: Where the reference axis (the first magnetic atom's moment) points.
    direction: tuple
    #: ``(3,)`` in Ry per radian, cartesian: component ``a`` is minus the
    #: derivative of the free energy for a turn about ``e_a``.
    torque: np.ndarray
    #: ``sum w eps`` of the states, in Ry.
    band_energy: float
    #: ``sum w <psi|H|psi>`` rebuilt from the rotated texture at ``w = 0``.
    band_energy_check: float
    #: The smearing's ``-TS``, in Ry.
    entropy: float = 0.0
    fermi_energy: float | None = None

    @property
    def torque_mev(self) -> np.ndarray:
        return np.asarray(self.torque) * RY_TO_EV * 1000.0

    @property
    def gradient(self) -> np.ndarray:
        """``dF/dw``, which is what an optimizer steps against."""
        return -np.asarray(self.torque)

    @property
    def free_energy(self) -> float:
        """``sum w eps - TS``, the energy this torque is the derivative of."""
        return self.band_energy + self.entropy

    @property
    def along_moment(self) -> float:
        """The component about :attr:`direction`, zero for a collinear texture."""
        return float(np.asarray(self.torque) @ np.asarray(self.direction))

    @property
    def euler_angles(self) -> tuple:
        """:attr:`rotation` as ZYZ angles, radians."""
        return euler_from_rotation(self.rotation)

    @property
    def residual(self) -> float:
        """``|sum w <psi|H|psi> - sum w eps|``, in Ry: the assembly's own check."""
        return abs(self.band_energy_check - self.band_energy)


def run_orientation_torque(
    system: System,
    pseudos: tuple[Pseudopotential, ...],
    density: jnp.ndarray,
    rotation=None,
    nbnd: int | None = None,
    conv_thr: float = 1.0e-10,
    k_batch: int | None | str = "default",
    soc_scale: float | None = None,
) -> OrientationTorque:
    """The torque on the texture for a rigid rotation of every spin, three components.

    :func:`run_torque` turns a collinear moment in one plane and returns one
    number. This turns the whole texture by ``rotation`` (``R0``, the identity
    by default, meaning the system's own orientation; :func:`rotation_from_euler`
    builds one from angles), diagonalises once with the coupling at that
    orientation, and returns ``-dF/dw`` for the three generators about it
    (:func:`defumat.forces.torque.orientation_torque`). For a collinear magnet
    turning in the plane ``(e1, e2)`` it contains :func:`run_torque`'s number as
    ``torque . (e1 x e2)``, and its component along the moment is zero.

    ``density`` is the source's, converged **without** spin-orbit coupling, as
    :func:`run_force_theorem` takes it, and everything that function refuses is
    refused here for the same reasons. It may be a four-component density whose
    moments are not collinear -- a commensurate spiral in a supercell, a canted
    antiferromagnet -- which is turned as it is (:func:`_reference_texture`),
    with the system's ``angle1``/``angle2`` describing it, since those are what
    ``_with_rotation`` turns with it.
    """
    from defumat.forces.torque import (
        band_energy_at_rotation,
        orientation_torque,
        rotate_texture,
    )

    if soc_scale is not None:
        system = system.with_soc_scale(soc_scale)
    _refuse_system(system, pseudos)
    rotation = _checked_rotation(rotation)

    own = _reference_axis(system)
    texture = _reference_texture(density, own)
    turned = _with_rotation(system, rotation)
    calculation, turned, eigenvalues, wavefunctions = fixed_density_states(
        turned, pseudos, rotate_texture(texture, rotation), nbnd=nbnd,
        conv_thr=conv_thr, k_batch=k_batch,
    )
    wg, levels = calculation.occupations(jnp.asarray(eigenvalues))

    check = float(band_energy_at_rotation(
        calculation, wavefunctions, wg, texture, rotation, np.zeros(3)
    ))
    value = orientation_torque(
        calculation, wavefunctions, wg, texture, rotation, k_batch=k_batch,
    )
    direction = rotation @ np.asarray(own, dtype=float)
    return OrientationTorque(
        rotation=rotation,
        direction=tuple(float(x) for x in direction / np.linalg.norm(direction)),
        torque=np.asarray(value, dtype=float),
        band_energy=float(np.sum(np.asarray(wg) * np.asarray(eigenvalues))),
        band_energy_check=check,
        entropy=float(levels.get("smearing", 0.0)),
        fermi_energy=levels.get("fermi_energy"),
    )


#: Trust radii of the orientation relaxation, in **radians**: the rotation
#: vector is the coordinate, so a length in it is an angle. The first step is a
#: steepest-descent step of ``INI`` (about 11 degrees), no step turns the
#: texture by more than ``MAX`` (about 29 degrees), and ``MIN`` is where QE's
#: line search gives up. Chosen as angles rather than inherited from the
#: atoms' bohr, which have no meaning here.
ORIENTATION_TRUST_INI = 0.2
ORIENTATION_TRUST_MAX = 0.5
ORIENTATION_TRUST_MIN = 1.0e-4

#: How far the chart ``R(x) = exp([x]x) R_start`` is followed before it is
#: re-centred on the current orientation, in radians. The exponential map is a
#: good chart up to ``|x| < pi``, and its Jacobian degenerates as ``|x|``
#: approaches it, so the optimizer is restarted well before.
ORIENTATION_CHART_LIMIT = 2.0


def _exp_rotation(vector) -> np.ndarray:
    """``exp([x]x)``, Rodrigues' formula, on the host (nothing differentiates it)."""
    vector = np.asarray(vector, dtype=float)
    angle = float(np.linalg.norm(vector))
    if angle < 1.0e-14:
        return np.eye(3)
    axis = vector / angle
    cross = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * cross @ cross


def _left_jacobian(vector) -> np.ndarray:
    """``J`` with ``exp([x + d]x) = exp([J d]x) exp([x]x)`` to first order in ``d``.

    So ``dF/dx = J^T g`` for a gradient ``g`` taken, as :func:`run_orientation_torque`
    takes it, for a turn applied on the left of the current orientation.
    ``J = 1 + (1 - cos t)/t^2 [x]x + (t - sin t)/t^3 [x]x^2`` with ``t = |x|``,
    and its series below ``t = 1e-4``, where the two quotients lose digits.
    """
    vector = np.asarray(vector, dtype=float)
    angle = float(np.linalg.norm(vector))
    cross = np.array([
        [0.0, -vector[2], vector[1]],
        [vector[2], 0.0, -vector[0]],
        [-vector[1], vector[0], 0.0],
    ])
    if angle < 1.0e-4:
        first, second = 0.5 - angle**2 / 24.0, 1.0 / 6.0 - angle**2 / 120.0
    else:
        first = (1.0 - np.cos(angle)) / angle**2
        second = (angle - np.sin(angle)) / angle**3
    return np.eye(3) + first * cross + second * cross @ cross


@dataclass
class OrientationStep:
    """One point of an orientation relaxation."""

    index: int
    rotation: np.ndarray
    #: ``sum w eps - TS`` of the one-shot, Ry: what the optimizer minimises.
    free_energy: float
    #: ``(3,)`` Ry per radian, cartesian.
    torque: np.ndarray
    #: Where the reference axis points at this orientation.
    direction: tuple
    #: The largest free component of the gradient in the chart, Ry per radian.
    max_gradient: float


@dataclass
class RelaxedOrientation:
    """The orientation a texture's free energy is stationary at, and how it got there."""

    converged: bool
    #: The last orientation evaluated, relative to the system's own texture.
    rotation: np.ndarray
    #: The one-shot at :attr:`rotation`.
    torque: OrientationTorque
    steps: list
    optimizer_failed: bool = False
    #: ``(3, 3)`` Ry per radian^2 from a central difference of the torque about
    #: the final orientation, when asked for; the symmetrised Hessian of ``F`` in
    #: the space frame. Positive on the live generators at a minimum.
    curvature: np.ndarray | None = None

    @property
    def direction(self) -> tuple:
        """Where the reference axis points: the easy axis, for a collinear magnet."""
        return self.torque.direction

    @property
    def euler_angles(self) -> tuple:
        return euler_from_rotation(self.rotation)

    @property
    def free_energies(self) -> np.ndarray:
        return np.array([step.free_energy for step in self.steps])

    @property
    def curvature_eigenvalues(self) -> np.ndarray | None:
        """The curvature's eigenvalues, Ry per radian^2: all non-negative at a minimum.

        A collinear texture has one zero among them, the turn about its own
        moment, which moves nothing.
        """
        if self.curvature is None:
            return None
        return np.linalg.eigvalsh(self.curvature)


def relax_orientation(
    system: System,
    pseudos: tuple[Pseudopotential, ...],
    density: jnp.ndarray,
    rotation=None,
    *,
    nbnd: int | None = None,
    conv_thr: float = 1.0e-10,
    etot_conv_thr: float = 1.0e-9,
    grad_conv_thr: float = 1.0e-8,
    nstep: int = 30,
    free=(1, 1, 1),
    curvature: bool = False,
    curvature_step: float = 0.02,
    ion_dynamics: str | None = None,
    k_batch: int | None | str = "default",
    soc_scale: float | None = None,
    verbose: bool = False,
) -> RelaxedOrientation:
    """Turn the whole texture until the torque on it vanishes: the easy orientation.

    ``ORIENTATION-NEXT.md`` Route A, the relaxation. Each step is
    :func:`run_orientation_torque` at the current orientation, one
    diagonalisation with the coupling at the frozen source density turned
    rigidly, and the free energy and torque it returns go to the same BFGS the
    atoms use, as :func:`~defumat.workflows.spiral.relax_spiral_q` hands it the
    spiral wavevector. The coordinate is the rotation vector ``x`` of the chart
    ``R(x) = exp([x]x) R_start``, the metric is the identity, so a length is an
    angle in radians, and the gradient in the chart is ``J(x)^T`` times the
    torque's (:func:`_left_jacobian`). The chart is re-centred on the current
    orientation, losing the Hessian, if ``|x|`` passes
    :data:`ORIENTATION_CHART_LIMIT`.

    Args:
        rotation: the starting orientation, relative to the system's own
            texture (:func:`rotation_from_euler` builds one). **Start off every
            symmetry element**: an orientation a symmetry fixes has no torque
            whether it is an easy axis or a hard one, and a relaxation started
            there reports convergence without moving. A start whose torque is
            already below ``grad_conv_thr`` is warned about for that reason.
        etot_conv_thr, grad_conv_thr: in Ry and Ry per radian, and both must
            hold, as in a ``pw.x`` relaxation. The gradient threshold's floor is
            the torque's own noise, set by the eigensolver's floor
            ``ETHR_MIN = 1e-13`` at about 1e-10 Ry per radian on tetragonal
            cobalt (P122); 1e-8 is an angle of about 1e-4 rad against a
            curvature of ``2 K1 = 8e-5`` Ry per radian^2.
        free: a cartesian mask on the chart's gradient, as ``if_pos`` is on a
            force. A collinear texture needs none: the turn about its own
            moment has no torque, so the optimizer never moves along it.
        curvature: add a central difference of the torque about the final
            orientation, six more one-shots, so that a minimum is told from a
            saddle (:attr:`RelaxedOrientation.curvature_eigenvalues`).

    The energy minimised is the free energy ``sum w eps - TS`` and not the band
    energy, because the torque is the free energy's derivative and a line
    search on the other would see an energy inconsistent with its gradient
    (P60 measured the entropy at 55 per cent of the band energy's slope on
    tetragonal cobalt at ``degauss = 0.02``).
    """
    from defumat.relax.bfgs import BFGSSettings
    from defumat.relax.registry import get_ion_dynamics
    from defumat.workflows.spiral import _first_step_scale

    start = _checked_rotation(rotation)
    free = np.asarray(free, dtype=float).reshape(3)

    def fresh_optimizer():
        settings = BFGSSettings(
            trust_radius_max=ORIENTATION_TRUST_MAX,
            trust_radius_ini=ORIENTATION_TRUST_INI,
            trust_radius_min=ORIENTATION_TRUST_MIN,
        )
        optimizer = get_ion_dynamics(ion_dynamics)(
            at=np.eye(3), energy_thr=etot_conv_thr, grad_thr=grad_conv_thr,
            settings=settings,
        )
        return optimizer, settings

    def one_shot(orientation):
        return run_orientation_torque(
            system, pseudos, density, rotation=orientation, nbnd=nbnd,
            conv_thr=conv_thr, k_batch=k_batch, soc_scale=soc_scale,
        )

    optimizer, settings = fresh_optimizer()
    chart = np.zeros(3)
    first_in_chart = True
    steps: list[OrientationStep] = []
    converged = False
    for index in range(1, nstep + 1):
        orientation = _exp_rotation(chart) @ start
        result = one_shot(orientation)
        gradient = (_left_jacobian(chart).T @ result.gradient) * free
        max_gradient = float(np.max(np.abs(gradient)))
        if index == 1 and max_gradient < grad_conv_thr:
            warnings.warn(
                f"the starting orientation already has a torque below "
                f"grad_conv_thr ({max_gradient:.2e} Ry/rad): an orientation a "
                "symmetry of the crystal fixes has no torque whether it is an "
                "easy axis or a hard one, so this reports convergence without "
                "having searched. Start off every symmetry element, or pass "
                "curvature=True to see which it is",
                stacklevel=2,
            )
        if first_in_chart:
            # No curvature is known yet, so the first step is steepest descent
            # of the trust radius's length, as ``relax_spiral_q`` makes it.
            settings.hessian_scale = _first_step_scale(optimizer, gradient, settings)
            first_in_chart = False
        moved, converged = optimizer.step(
            chart.reshape(1, 3), result.free_energy, -gradient.reshape(1, 3),
        )
        steps.append(OrientationStep(
            index=index,
            rotation=orientation,
            free_energy=result.free_energy,
            torque=np.asarray(result.torque),
            direction=result.direction,
            max_gradient=max_gradient,
        ))
        if verbose:
            d = result.direction
            print(f"orientation step {index:3d}   axis = ({d[0]:8.5f}, "
                  f"{d[1]:8.5f}, {d[2]:8.5f})   F = {result.free_energy:18.10f} Ry"
                  f"   max |dF/dx| = {max_gradient:.3e}", flush=True)
        if converged or optimizer.failed:
            break
        chart = np.asarray(optimizer.to_crystal(moved), dtype=float).reshape(3)
        if np.linalg.norm(chart) > ORIENTATION_CHART_LIMIT:
            start = _exp_rotation(chart) @ start
            chart = np.zeros(3)
            optimizer, settings = fresh_optimizer()
            first_in_chart = True

    hessian = None
    if curvature:
        columns = []
        for axis in np.eye(3):
            plus = one_shot(_exp_rotation(curvature_step * axis) @ orientation)
            minus = one_shot(_exp_rotation(-curvature_step * axis) @ orientation)
            columns.append((plus.gradient - minus.gradient) / (2.0 * curvature_step))
        hessian = np.asarray(columns).T
        hessian = 0.5 * (hessian + hessian.T)

    return RelaxedOrientation(
        converged=bool(converged and not optimizer.failed),
        rotation=orientation,
        torque=result,
        steps=steps,
        optimizer_failed=bool(optimizer.failed),
        curvature=hessian,
    )


# ---------------------------------------------------------------------------
# The relaxed anisotropy: two self-consistent totals instead of two band sums
# ---------------------------------------------------------------------------

#: How far a converged moment may sit from the direction it was asked for, in
#: degrees, before :func:`run_relaxed_anisotropy` says so. **Nothing holds the
#: moment in a relaxed run** -- the whole point is that the density is free --
#: so a direction that is not a stationary point of the anisotropy energy will
#: drift towards one that is, and the two totals then belong to two states
#: neither of which is the one asked for. A cardinal axis of a cubic or uniaxial
#: crystal is stationary by symmetry and does not drift; an oblique direction
#: generally does. 1 degree is far above the wander of a converged run and far
#: below the ~30 degrees a genuine collapse gives.
RELAXED_DRIFT_TOL = 1.0


@dataclass
class RelaxedDirection:
    """One direction converged self-consistently, and where it ended up."""

    #: Cartesian unit vector the moment was started along.
    direction: tuple
    #: The **total** energy, in Ry -- not a band-energy sum.
    total_energy: float
    converged: bool
    iterations: int
    accuracy: float
    #: The converged cell moment as a cartesian vector, in Bohr magnetons.
    moment: tuple
    #: Angle between :attr:`moment` and :attr:`direction`, in degrees.
    drift: float
    #: The magnitude of :attr:`moment`, in Bohr magnetons.
    moment_length: float


@dataclass
class RelaxedAnisotropy:
    """A set of directions, each with its own self-consistent total energy."""

    directions: tuple
    results: tuple
    reference: int = 0

    @property
    def total_energies(self) -> np.ndarray:
        return np.array([r.total_energy for r in self.results])

    @property
    def energies(self) -> np.ndarray:
        """Total energies relative to :attr:`reference`, in Ry."""
        return self.total_energies - self.total_energies[self.reference]

    @property
    def energies_mev(self) -> np.ndarray:
        return self.energies * RY_TO_EV * 1000.0

    @property
    def anisotropy(self) -> float:
        """Hardest minus easiest, in Ry."""
        energies = self.total_energies
        return float(np.max(energies) - np.min(energies))

    @property
    def anisotropy_mev(self) -> float:
        return self.anisotropy * RY_TO_EV * 1000.0

    @property
    def easy_axis(self) -> tuple:
        return self.directions[int(np.argmin(self.total_energies))]

    @property
    def hard_axis(self) -> tuple:
        return self.directions[int(np.argmax(self.total_energies))]

    @property
    def converged(self) -> bool:
        """Whether **every** direction's SCF converged."""
        return all(r.converged for r in self.results)

    @property
    def drifts(self) -> np.ndarray:
        """``(ndir,)`` in degrees: how far each moment left its direction."""
        return np.array([r.drift for r in self.results])

    @property
    def moment_lengths(self) -> np.ndarray:
        return np.array([r.moment_length for r in self.results])

    def difference(self, i: int, j: int) -> float:
        """``E(i) - E(j)`` in Ry."""
        return float(self.total_energies[i] - self.total_energies[j])


def _refuse_relaxed(system: System, pseudos, require_spin_orbit: bool = True) -> None:
    """What a *self-consistent* anisotropy cannot do, which is less than the theorem.

    **The differences from :func:`_refuse_system` are the point of this
    function existing**, so they are named rather than left to a diff:

    * **PAW is allowed.** The force theorem refuses it because its handoff is a
      density and a PAW Hamiltonian needs ``ddd_paw``, which is built from
      ``becsum`` -- a property of the wavefunctions of the *other* leg's run,
      with a different pseudopotential file and a different projector count.
      There is no handoff here. Each direction is one ordinary self-consistent
      run that builds its own ``becsum`` from its own states, so the dataset
      never changes hands and the refusal does not apply. This is the whole
      reason the relaxed route is worth having beyond its second-order term:
      it reaches a regime the theorem cannot, and
      ``Ni.rel-pbe-spn-kjpaw_psl.1.0.0.UPF`` is a fully-relativistic PAW
      dataset for a magnetic element.
    * **A Hubbard U is allowed**, for the same reason: ``ns`` is converged here
      rather than carried across.
    * **A magnetic field or a constrained moment is still refused**, and this
      one is *not* inherited -- it is the same argument arriving at the same
      answer. The field's energy is deliberately outside the reported total
      (``defumat/scf/fields.py``), so two directions' totals differ by a Zeeman
      term that neither total accounts for, and the anisotropy is contaminated
      by whatever the constraint cost. A held moment also makes the drift check
      below meaningless, since the constraint is what is holding it.
    * **A potential-only meta-GGA is still refused**, and here the reason is
      sharper than the theorem's: ``tb09`` and ``bj06`` have no energy
      functional at all, so the total energy this quantity is a difference of
      is not the value of anything the run minimised (``PLAN.md`` P30-P32).
    * **A spin spiral is still refused**: it has no spin-orbit coupling to be
      anisotropic about.
    """
    if system.input_dft and system.input_dft.strip().lower() in ("tb09", "bj06"):
        raise NotImplementedError(
            f"a relaxed magnetic anisotropy under {system.input_dft}: a "
            "potential-only meta-GGA has no energy functional, so its printed "
            "total is not the value of anything the SCF minimised and a "
            "difference of two of them is not an energy difference. The force "
            "theorem refuses it too, for the different reason that tau is not "
            "in its handoff"
        )
    if (
        system.constrained_magnetization != "none"
        or any(system.b_field)
        or any(any(v) for v in system.atomic_b_field or ())
    ):
        raise NotImplementedError(
            "a relaxed magnetic anisotropy with a magnetic field or a "
            "constrained moment: the field's energy is deliberately outside the "
            "reported total (defumat/scf/fields.py), so two directions' totals "
            "differ by a Zeeman term no total accounts for. It also defeats the "
            "drift check, since the constraint is what holds the moment where it "
            "was put rather than the anisotropy"
        )
    if system.spiral:
        raise NotImplementedError(
            "a relaxed magnetic anisotropy for a spin spiral: a spiral refuses "
            "spin-orbit coupling permanently (it breaks the generalized Bloch "
            "theorem), so there is no coupling for the energy to depend on a "
            "direction through"
        )
    if not system.noncolin:
        raise ValueError(
            f"a relaxed magnetic anisotropy is a noncollinear run and this "
            f"system has nspin = {system.nspin}: set noncolin = .true. and "
            "lspinorb = .true. with a fully-relativistic dataset"
        )
    if require_spin_orbit and not system.lspinorb:
        raise ValueError(
            "a relaxed magnetic anisotropy needs lspinorb = .true. and a "
            "fully-relativistic dataset: without spin-orbit coupling the "
            "Hamiltonian commutes with a global spin rotation and every "
            "direction has exactly the same total energy. Use "
            "run_relaxed_anisotropy(..., require_spin_orbit=False) to run it "
            "anyway, which is the control for that identity"
        )
    if not system.nosym:
        raise ValueError(
            "a relaxed magnetic anisotropy needs nosym = .true.: a magnetic "
            "noncollinear run reduces its k-grid with the magnetic symmetry "
            "group, which depends on where the moment points, so two "
            "directions would be sampled on two different wedges and their "
            "total energies would differ by the k-sampling rather than by the "
            "physics. The force theorem requires it for the same reason, and "
            "here it is worse: a total energy carries the Ewald and Hartree "
            "terms as well, so the contamination is not confined to the bands"
        )


def run_relaxed_direction(
    system: System,
    pseudos: tuple[Pseudopotential, ...],
    direction=None,
    require_spin_orbit: bool = True,
    soc_scale: float | None = None,
    **options,
) -> RelaxedDirection:
    """One direction, converged self-consistently, with its drift measured.

    Where :func:`run_force_theorem` freezes a density and diagonalises once,
    this runs a whole SCF with the magnetization started along ``direction``
    and reports the **total** energy. The difference between the two routes is
    the energy the density gains by relaxing in the spin-orbit field, which is
    second order in the coupling where the anisotropy itself is second order --
    so it is not negligible by any general argument, and the size of it is a
    measurement rather than an assumption (:func:`run_relaxed_anisotropy`
    reports both when asked).

    ``options`` are :func:`~defumat.scf.driver.run_scf`'s. ``conv_thr``
    defaults to 1e-10 rather than the SCF's usual 1e-6 for the same reason the
    theorem's does: the answer is a difference of two numbers of order 100 Ry
    in their eighth decimal, and a looser threshold leaves more noise than
    signal.
    """
    if soc_scale is not None:
        system = system.with_soc_scale(soc_scale)
    # ``require_spin_orbit = False`` lifts **only** the spin-orbit clause. Every
    # other refusal still holds, and ``nosym`` especially: without it the
    # identity this control checks would be broken by the k-set rather than by
    # the physics, and the control would fail for a reason unrelated to what it
    # is controlling.
    _refuse_relaxed(system, pseudos, require_spin_orbit)

    if direction is None:
        direction = _reference_axis(system)
    direction = np.asarray(direction, dtype=float)
    direction = tuple(float(x) for x in direction / np.sqrt(np.sum(direction**2)))
    # The same rebuild the force theorem needs, and for the same reason: a GGA
    # reads its quantization axis off ``angle1``/``angle2``, so rotating where
    # the moment is started without rotating the axis leaves the functional
    # differentiating ``|m|`` through its own kinks
    # (:func:`_with_quantization_axis` has the 36.8 meV that cost).
    system = _with_quantization_axis(system, direction)

    options.setdefault("conv_thr", 1.0e-10)
    from defumat.scf.driver import run_scf

    scf = run_scf(system, pseudos, **options)

    moment = np.asarray(scf.magnetization_vector, dtype=float).reshape(3)
    length = float(np.linalg.norm(moment))
    if length > 0.0:
        cosine = float(np.dot(moment / length, np.asarray(direction)))
        drift = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    else:
        drift = float("nan")
    return RelaxedDirection(
        direction=direction,
        total_energy=float(scf.total_energy),
        converged=bool(scf.converged),
        iterations=int(scf.iterations),
        accuracy=float(scf.accuracy),
        moment=tuple(float(x) for x in moment),
        drift=drift,
        moment_length=length,
    )


def run_relaxed_anisotropy(
    system: System,
    pseudos: tuple[Pseudopotential, ...],
    directions=None,
    require_spin_orbit: bool = True,
    soc_scale: float | None = None,
    warn_on_drift: bool = True,
    **options,
) -> RelaxedAnisotropy:
    """The magnetocrystalline anisotropy as a difference of **total** energies.

    One full self-consistent noncollinear run per direction, against
    :func:`run_anisotropy`'s one diagonalisation per direction at a frozen
    density. The two answer slightly different questions and both are worth
    having:

    * the **force theorem** freezes the density converged without spin-orbit
      coupling and asks what the bands cost when the coupling is switched on.
      Every other term of the total energy is a functional of ``rho`` alone and
      cancels exactly between two directions, so the whole answer is one band
      sum and the noise of two nearly-equal 100 Ry totals never appears;
    * the **relaxed** anisotropy lets the density respond to the spin-orbit
      field in each direction. The extra energy that buys is variational, so it
      is negative in both directions and the anisotropy is the difference of
      two such gains -- a second-order effect on a second-order quantity, which
      is small but is not zero and is not bounded by any argument this code
      can make. Measuring it is the point.

    ``directions`` follows :func:`run_anisotropy`: a sequence of cartesian
    vectors, an integer for :func:`sphere_cover`, or ``"cardinal"``/``"xz"``/
    ``"xyz"``, defaulting to ``"xyz"``.

    **Nothing holds the moment**, which is the price of letting the density
    relax, so each direction reports where its moment actually ended
    (:attr:`RelaxedDirection.drift`) and a drift past
    :data:`RELAXED_DRIFT_TOL` is warned about by name. A cardinal axis of a
    cubic or uniaxial crystal is stationary by symmetry and stays put; an
    oblique direction on a strongly anisotropic magnet need not, and an
    anisotropy assembled from directions that drifted is a difference between
    two states that are not the ones asked for.

    ``options`` go to :func:`~defumat.scf.driver.run_scf`. ``conv_thr``
    defaults to 1e-10.
    """
    directions = _direction_set(system, directions)
    results = tuple(
        run_relaxed_direction(
            system, pseudos, direction=direction,
            require_spin_orbit=require_spin_orbit, soc_scale=soc_scale,
            **options,
        )
        for direction in directions
    )
    anisotropy = RelaxedAnisotropy(
        directions=tuple(r.direction for r in results), results=results
    )
    unconverged = [r for r in results if not r.converged]
    if unconverged:
        warnings.warn(
            f"{len(unconverged)} of {len(results)} directions did not converge "
            f"(worst accuracy {max(r.accuracy for r in unconverged):.3e} Ry). A "
            "relaxed anisotropy is a difference of total energies in their "
            "eighth decimal, so an unconverged direction does not give a "
            "slightly wrong anisotropy -- it gives one dominated by where the "
            "SCF happened to stop. Raise max_iterations or loosen conv_thr "
            "deliberately, and read RelaxedAnisotropy.converged before the "
            "number",
            RuntimeWarning,
            stacklevel=2,
        )
    drifted = [r for r in results if np.isfinite(r.drift)
               and r.drift > RELAXED_DRIFT_TOL]
    if warn_on_drift and drifted:
        worst = max(drifted, key=lambda r: r.drift)
        warnings.warn(
            f"{len(drifted)} of {len(results)} directions drifted more than "
            f"{RELAXED_DRIFT_TOL} degrees from where the moment was started, "
            f"the worst by {worst.drift:.2f} degrees (direction "
            f"{tuple(np.round(worst.direction, 4))}, moment "
            f"{tuple(np.round(worst.moment, 4))}). Nothing holds the moment in "
            "a relaxed run, so that direction's total energy belongs to a "
            "state other than the one asked for and the anisotropy built from "
            "it is not the anisotropy of those directions. Cardinal axes of a "
            "cubic or uniaxial crystal are stationary by symmetry and should "
            "not drift; for an oblique direction, use the force theorem (which "
            "cannot drift, its density being frozen) or hold the moment with "
            "constrained_magnetization and accept that the constraint's energy "
            "is outside the total",
            RuntimeWarning,
            stacklevel=2,
        )
    return anisotropy


def _direction_set(system: System, directions):
    """``run_anisotropy``'s ``directions`` argument, resolved to vectors.

    Shared by the frozen and relaxed routes so that the two cannot drift apart
    in what ``"cardinal"`` means -- an anisotropy compared between them must be
    over the same set or the comparison is of two different quantities.
    """
    if directions is None:
        directions = "xyz"
    if isinstance(directions, str):
        if directions == "cardinal":
            return cardinal_directions(system)
        if directions == "xz":
            return ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        if directions == "xyz":
            return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        raise ValueError(
            f"directions = {directions!r}: expected a sequence of vectors, "
            "an integer, or one of 'cardinal', 'xz', 'xyz'"
        )
    if isinstance(directions, (int, np.integer)):
        return sphere_cover(int(directions))
    return tuple(directions)
