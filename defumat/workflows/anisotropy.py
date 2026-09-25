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
expectation value ``<psi|H_SOC|psi>`` -- gives **zero anisotropy**, not a small
one. Spin-orbit coupling enters at first order as ``xi <L> . n``, and the
orbital moment of a scalar-relativistic collinear state is quenched: this
package measures it at 1.7e-16 (:mod:`defumat.projwfc.angular_momentum`). The
anisotropy is second order in the coupling, and what supplies it is the
*repulsion between levels* that the diagonalisation performs and an expectation
value does not. :func:`frozen_expectation` computes that vanishing first-order
term anyway, because a number measured to be zero is worth more than an
argument about why it should be: **+/-0.000001 meV** on a one-atom cobalt cell,
direction-independent to **1.9e-6 meV**, where the force theorem on the same
density gives **0.597 meV**.

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
    rotation = _rotation_taking(own, wanted)
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

    with ``dV_NL`` and ``dS`` the spin-*traceless* halves of ``dvan_so`` and
    ``qq_so`` (:func:`~defumat.pseudo.spinorbit.spin_trace`). The ``eps dS``
    piece is there because an ultrasoft eigenproblem is generalised and the
    metric is perturbed too, so first-order perturbation theory carries it.

    **One term of the first-order operator is not in it.** On an ultrasoft or
    PAW dataset the coupled Hamiltonian at a frozen potential also differs from
    the reduced one by ``newd_so``'s sandwich against its spin trace,
    ``F B F - T(B)`` (``scf/driver.py:_newd_noncollinear`` at 1 against 0), and
    that difference is not evaluated here. The number below is on the ultrasoft
    ``Co.rel-pbe-nd-rrkjus``, so it is the expectation value of the other two
    terms; the quenched-moment argument says the third vanishes at first order
    as well, and that has not been measured (``AUDIT-2026-09-25.md``).

    **This is the calculation the force theorem is often assumed to be, and it
    returns essentially zero for every direction** -- +/-0.000001 meV on a
    one-atom cobalt cell, direction-independent to 1.9e-6 meV, where the force
    theorem on the same density gives 0.597 meV.** Spin-orbit coupling enters
    at first order as ``xi <L> . n``, and the orbital moment of a
    scalar-relativistic collinear state is quenched -- 1.7e-16 as this package
    measures it (:mod:`defumat.projwfc.angular_momentum`). A magnetic
    anisotropy is second order in the coupling, and what supplies it is the
    *repulsion between levels* that a diagonalisation performs and an
    expectation value does not. Hence :func:`run_force_theorem`.

    It is a function rather than a footnote because a number that has been
    measured to be zero is a stronger statement than an argument that it should
    be, and because the ``soc_scale`` knob is what makes it well posed: a
    pseudopotential has no additive ``xi L.S`` operator to take the expectation
    value *of*, only the part of ``dvan_so`` that a spin trace does not keep.
    """
    from defumat.forces.energy import _spinor_projector_energies
    from defumat.pseudo.spinorbit import build_spin_orbit
    from defumat.scf.driver import _spin_block_diagonal

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

    types = system.structure.types
    coupled = build_spin_orbit(pseudos, 1.0)
    free = build_spin_orbit(pseudos, 0.0)
    delta_dvan = jnp.asarray(
        _spin_block_diagonal([coupled[t].dvan_so for t in types])
    ) - jnp.asarray(_spin_block_diagonal([free[t].dvan_so for t in types]))

    delta_qq = None
    if calculation.qq_so is not None:
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

    nonlocal_, overlap = _spinor_projector_energies(
        jnp.asarray(wavefunctions), calculation.projectors.vkb,
        delta_dvan, delta_qq, jnp.asarray(wg), jnp.asarray(eigenvalues),
    )
    return float(nonlocal_ - overlap)


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
