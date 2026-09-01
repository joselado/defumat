"""Continuing a converged SCF into a *different spin regime*.

An unpolarized run, a collinear one and a noncollinear one are three
descriptions of the same electrons, and the expensive part of all three -- the
charge density -- is very nearly the same object. This module maps the converged
state of one onto the starting state of another, so that a magnetic calculation
starts from the converged non-magnetic charge instead of from a superposition of
atomic charges, and so that spin-orbit coupling can be switched on and off
without going back to the atoms in between.

**The mixed state is a triple, and all three are carried.** ``run_scf`` mixes
``(rho, becsum, ns)`` together, and its own docstring says that giving one
without the others starts the run from two states at once. So does this: a
promotion converts the density, the ``becsum`` of every ultrasoft/PAW species and
the Hubbard occupation matrix *by the same rule*, and where the rule needs a
decision -- whether to carry a magnetization or to seed a new one -- the decision
is taken once, from the density, and applied to the rest.

**The representation, and why one intermediate serves every direction.** The
three regimes differ only in how they write the same pair ``(n(r), m(r))``:

============  =====================  =========================================
``nspin_mag``  the array                what it means
============  =====================  =========================================
1             ``[n]``                 no magnetization at all
2             ``[n_up, n_dw]``        ``n = n_up + n_dw``, ``m = n_up - n_dw``
                                      along ``z`` by construction
4             ``[n, m_x, m_y, m_z]``  a magnetization *vector* field
============  =====================  =========================================

Every promotion here is therefore "decompose into ``(n, m)``, decide what ``m``
should be, recompose" -- :func:`spin_components` and
:func:`from_spin_components`. There is no separate 1->2, 2->4 and 4->2 code path
to keep consistent, and a demotion is the same function read the other way --
with one asymmetry, since a vector magnetization has an axis to *find* before it
can be written down as a scalar (:func:`_collinear_axis`).

**The trap that makes a promotion useless, and it is silent.** Nothing in the
SCF breaks spin symmetry on its own (``starting_density``'s docstring says so
for the atomic start, and it is just as true here): promote a converged
unpolarized density to two identical channels and the run converges straight
back to the unpolarized solution, having found a stationary point rather than
the magnetic one. **The magnetization has to be put in by hand**, exactly as
``starting_magnetization`` puts it into a fresh run -- and that is what
``magnetization="auto"`` does: it carries the source's magnetization when it has
one, and otherwise seeds the target's atomic one on top of the converged charge.
The charge is what took the iterations; the seed is what takes the run off the
symmetric solution.

**What is *not* transferred, and why.** The eigenvalues, the Fermi level and the
occupations are all rebuilt from the first diagonalisation, so a continuation is
not a restart of the loop's internal state -- it is a starting guess, and the
result it converges to is the target regime's own self-consistent solution. The
wavefunctions are transferred where they can be (:func:`promote_wavefunctions`),
and where they cannot -- a noncollinear target whose *magnetic* symmetry group
reduces the k-mesh differently -- they are dropped with a warning and the run
falls back to the pseudo-atomic orbitals. Dropping them costs a few Davidson
steps; carrying wavefunctions belonging to different k-points would be wrong.

**What ``pw.x`` has of this, checked in the Fortran rather than assumed.**
``startingpot = 'file'`` (``PW/src/potinit.f90``) reads a density whose ``nspin``
need not match the run's, and ``read_rhog`` (``Modules/io_base.f90``) handles the
mismatch with ``infomsg('read_rhog', 'some spin components not found')`` and a
zero fill -- so QE's own continuation from an unpolarized file into an LSDA run
starts *unpolarized*, and converges back to the non-magnetic solution without
saying anything more about it. The collinear-to-noncollinear rotation does exist
-- ``nc_magnetization_from_lsda`` in the same file -- but only on the ``lforcet``
path (the force-theorem magnetocrystalline-anisotropy calculation), and it uses
``angle1(1)``/``angle2(1)``, **species one's angles, for the whole cell**. That
restriction is kept here rather than papered over: a target whose species point
different ways is refused, with the message naming the way out.
``PW/src/update_pot.f90`` is the same idea across an ionic step.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

__all__ = [
    "ContinuedState",
    "continued_state",
    "spin_components",
    "from_spin_components",
    "promote_density",
    "promote_becsum",
    "promote_ns",
    "promote_wavefunctions",
    "nc_magnetization_from_lsda",
]

#: Below this many Bohr magnetons per cell, a source magnetization counts as
#: absent and ``magnetization="auto"`` seeds the target's atomic one instead.
#: It is loose on purpose: a run that converged to ``1e-4 mu_B`` converged to
#: the *unpolarized* solution and its residue is numerical noise, so carrying it
#: would start the target run on the symmetric solution it is trying to leave.
MAGNETIZATION_TOL = 1.0e-4

#: How far two species' starting directions may differ and still count as one
#: axis (the dot product of the unit vectors, against 1).
DIRECTION_TOL = 1.0e-6

#: How much of a noncollinear magnetization may lie off its dominant axis and
#: still be written down as a collinear one. Measured as the two smaller
#: eigenvalues of ``int m_a m_b dr`` against the largest, so it is a fraction
#: of ``int |m|^2`` and not of the moment.
TRANSVERSE_TOL = 1.0e-6

#: What ``magnetization=`` accepts.
MAGNETIZATION_MODES = ("auto", "carry", "seed", "none")


def spin_components(values, nspin_mag: int):
    """``(n, m)`` from an array whose leading axis is the spin one.

    ``values`` is any ``(nspin_mag, ...)`` quantity written the way the density
    is -- the density itself, or one species' ``becsum``. The magnetization
    comes back as a ``(3, ...)`` vector in every case, with the collinear one
    placed on ``z``, which is the convention that makes 2 and 4 the same
    representation and the promotion between them a rotation rather than a
    reinterpretation.

    Returns ``(charge, magnetization)``, the magnetization being ``None`` when
    there is none to have.
    """
    values = jnp.asarray(values)
    if nspin_mag == 1:
        return values[0], None
    if nspin_mag == 2:
        charge = values[0] + values[1]
        moment = values[0] - values[1]
        zero = jnp.zeros_like(moment)
        return charge, jnp.stack([zero, zero, moment])
    if nspin_mag == 4:
        return values[0], values[1:4]
    raise ValueError(f"nspin_mag = {nspin_mag}: expected 1, 2 or 4")


def from_spin_components(charge, magnetization, nspin_mag: int):
    """The inverse of :func:`spin_components`.

    A collinear target takes the ``z`` component of the magnetization and
    nothing else -- the caller is responsible for having rotated the axis onto
    ``z`` first, because dropping ``m_x`` and ``m_y`` silently is exactly the
    kind of quiet reinterpretation this module exists to avoid.
    """
    charge = jnp.asarray(charge)
    if nspin_mag == 1:
        return charge[None]
    if magnetization is None:
        magnetization = jnp.zeros((3,) + charge.shape, dtype=charge.dtype)
    magnetization = jnp.asarray(magnetization)
    if nspin_mag == 2:
        moment = magnetization[2]
        return jnp.stack([charge + moment, charge - moment]) / 2.0
    if nspin_mag == 4:
        return jnp.concatenate([charge[None], magnetization])
    raise ValueError(f"nspin_mag = {nspin_mag}: expected 1, 2 or 4")


@dataclass(frozen=True)
class _SpinTransfer:
    """The one decision a promotion makes, applied to every part of the state.

    Built once from the density -- which is the only part big enough to say
    reliably whether the source is magnetic -- and then reused for ``becsum``,
    so the two cannot disagree about how polarized the starting state is. That
    is the same requirement ``_becsum_split`` states for the atomic start: "the
    two starting guesses have to agree about how polarized the atom is or the
    first iteration contradicts itself".
    """

    source: int
    target: int
    #: ``"carry"``, ``"seed"`` or ``"none"`` -- what actually happens to the
    #: magnetization, which is not always what the caller asked for: ``"auto"``
    #: has been resolved by here, and a target with ``nspin_mag = 1`` has
    #: nowhere to put one whatever was asked.
    mode: str
    #: Unit vector the collinear ``z`` axis is rotated onto, or ``None`` for no
    #: rotation (which includes every direction that is already ``z``).
    direction: tuple | None = None
    #: The reverse: the axis a *vector* magnetization is projected onto on its
    #: way down to a collinear one, or ``None`` when it is already ``z``.
    project: tuple | None = None

    @property
    def seeded(self) -> bool:
        return self.mode == "seed"

    def apply(self, values, seed=None):
        """Promote one ``(nspin_mag, ...)`` array, given the target's seed."""
        charge, moment = spin_components(values, self.source)
        if self.target == 1 or self.mode == "none":
            return from_spin_components(charge, None, self.target)
        if self.mode == "seed":
            if seed is None:
                moment = None
            else:
                _, moment = spin_components(seed, self.target)
        elif self.direction is not None:
            # A collinear source: the whole magnetization is ``m_z``, and the
            # target's ``angle1``/``angle2`` say where it should point.
            moment = _axis(self.direction, moment.ndim) * moment[2]
        elif self.project is not None:
            # The other way: a vector magnetization that happens to be
            # collinear, laid back down on ``z``, which is where the collinear
            # representation keeps it. ``m . n`` and not ``|m|``, so that an
            # antiferromagnet keeps its signs.
            along = jnp.sum(_axis(self.project, moment.ndim) * moment, axis=0)
            zero = jnp.zeros_like(along)
            moment = jnp.stack([zero, zero, along])
        return from_spin_components(charge, moment, self.target)


def _axis(direction, ndim: int):
    """A unit vector shaped to broadcast against a ``(3, ...)`` field."""
    values = jnp.asarray(np.asarray(direction, dtype=float))
    return values.reshape((3,) + (1,) * (ndim - 1))


def _collinear_axis(density) -> tuple | None:
    """The axis a noncollinear magnetization lies along, or a refusal.

    The collinear representation has one scalar field and a fixed axis, so a
    ``nspin_mag = 4`` state can only be written in it if it is *collinear* --
    and QE's own promotion in the other direction (``nc_magnetization_from_lsda``)
    is exactly this rotation read backwards.

    The axis is the dominant eigenvector of ``M_ab = int m_a m_b dr`` rather
    than of ``int m dr``, and the difference is not pedantry: an
    antiferromagnet's *signed* integral is zero and would leave the axis
    undefined, while the second moment is blind to the sign and finds it. The
    other two eigenvalues measure how much magnetization is off that axis,
    which is the refusal test.

    Returns ``None`` when the axis is already ``z``, so that the common case
    costs no arithmetic downstream.
    """
    _, moment = spin_components(density, 4)
    moment = np.asarray(moment).reshape(3, -1)
    second = moment @ moment.T
    values, vectors = np.linalg.eigh(second)
    off_axis = float(values[0] + values[1])
    if off_axis > TRANSVERSE_TOL * float(values[2]):
        raise ValueError(
            "the source magnetization is genuinely noncollinear "
            f"(a fraction {off_axis / max(float(values[2]), 1e-300):.2e} of it "
            "lies off its dominant axis) and a collinear calculation has one "
            "scalar magnetization on a fixed axis to write it in. Pass "
            "magnetization='seed' to keep the converged charge and start the "
            "collinear run from its own starting_magnetization, or "
            "'none' to start it unpolarized"
        )
    axis = vectors[:, 2]
    # The eigenvector's sign is arbitrary and flips which channel is "up",
    # which is a global spin flip -- harmless physically and confusing to read.
    # Fix it by the signed integral, and where that vanishes (an
    # antiferromagnet) by the point carrying the most magnetization.
    signed = float(axis @ moment.sum(axis=1))
    if abs(signed) < 1.0e-8:
        signed = float(axis @ moment[:, int(np.argmax(np.sum(moment**2, axis=0)))])
    if signed < 0.0:
        axis = -axis
    if abs(axis[2] - 1.0) < DIRECTION_TOL:
        return None
    return tuple(float(x) for x in axis)


def _absolute_magnetization(density, nspin_mag: int, cell) -> float:
    """``int |m(r)| dr`` in Bohr magnetons -- QE's absolute magnetization.

    The test of whether a source is magnetic at all. The *signed* integral is
    the wrong one to test with: an antiferromagnet integrates to zero and is as
    magnetic as a state gets.
    """
    _, moment = spin_components(density, nspin_mag)
    if moment is None:
        return 0.0
    size = jnp.sqrt(jnp.sum(moment**2, axis=0))
    return float(jnp.sum(size) * cell.volume / size.size)


def _common_direction(calculation) -> tuple | None:
    """The one axis every magnetic species of the target points along.

    ``None`` when it is ``z`` (so no rotation is needed) and a ``ValueError``
    when the species disagree: a collinear source carries one scalar field and
    there is no way to make it point two ways at once. The message names the
    escape hatch, because there is one -- ``magnetization="seed"`` keeps the
    converged *charge* and takes the magnetization from the atomic
    superposition, which does honour per-species angles.
    """
    magnitudes = np.asarray(calculation.starting_magnetization, dtype=float)
    directions = np.asarray(calculation.magnetization_directions, dtype=float)
    magnetic = np.abs(magnitudes) > 1.0e-6
    if not np.any(magnetic):
        return None
    axes = directions[magnetic]
    signs = np.sign(magnitudes[magnetic])[:, None]
    # A species with a *negative* starting magnetization points the other way
    # along the same axis, and its collinear counterpart already carries that
    # sign in ``m(r)``; what has to agree is the axis, not the direction on it.
    reference = axes[0] * signs[0]
    if not np.all(np.abs(np.abs(axes @ reference) - 1.0) < DIRECTION_TOL):
        raise ValueError(
            "the target's species point their moments along different axes "
            f"(angle1 = {tuple(calculation.system.angle1)}, angle2 = "
            f"{tuple(calculation.system.angle2)}), and a collinear source "
            "carries one scalar magnetization that cannot point two ways at "
            "once. Pass magnetization='seed' to keep the converged charge and "
            "take the magnetization from the atomic superposition, which does "
            "honour the angles"
        )
    if abs(reference[2] - 1.0) < DIRECTION_TOL:
        return None
    return tuple(float(x) for x in reference)


def _transfer(result, calculation, magnetization: str) -> _SpinTransfer:
    """Decide, once, how the magnetization crosses the regime change."""
    mode = str(magnetization).lower()
    if mode not in MAGNETIZATION_MODES:
        raise ValueError(
            f"magnetization = {magnetization!r}: expected one of "
            f"{MAGNETIZATION_MODES}"
        )
    source, target = int(result.nspin_mag), int(calculation.nspin_mag)
    if target == 1 or mode == "none":
        return _SpinTransfer(source, target, mode="none")

    present = _absolute_magnetization(
        result.density, source, calculation.system.cell
    )
    if mode == "carry" and present <= MAGNETIZATION_TOL:
        raise ValueError(
            f"magnetization='carry' but the source run has none to carry "
            f"(int |m| = {present:.2e} mu_B). Its density is a stationary "
            "point of the target functional too, so the run would converge "
            "straight back to it -- pass magnetization='auto' or 'seed' to put "
            "the target's starting_magnetization in on top of the converged "
            "charge"
        )
    seeded = mode == "seed" or (mode == "auto" and present <= MAGNETIZATION_TOL)

    direction = project = None
    if not seeded and source == 2 and target == 4:
        direction = _common_direction(calculation)
    if not seeded and source == 4 and target == 2:
        project = _collinear_axis(result.density)
    return _SpinTransfer(
        source, target, mode="seed" if seeded else "carry",
        direction=direction, project=project,
    )


def _check_grid(result, calculation) -> None:
    """The two runs must be the same *system*, differing only in its spin."""
    grid = tuple(calculation.basis.dense.grid)
    shape = tuple(np.shape(result.density))[1:]
    if shape != grid:
        raise ValueError(
            f"the source density is on a {shape} grid and the target run uses "
            f"{grid}. A continuation carries a density from one calculation to "
            "another, so the cell, the atoms and both cutoffs have to match"
        )
    system = getattr(result, "system", None)
    if system is not None:
        source_electrons = sum(
            calculation.pseudos[t].z_valence for t in system.structure.types
        )
        if abs(source_electrons - calculation.nelec) > 1.0e-8:
            raise ValueError(
                f"the source run has {source_electrons} electrons and the "
                f"target {calculation.nelec}: a continuation cannot change the "
                "number of electrons, only the spin regime"
            )


def promote_density(result, calculation, transfer: _SpinTransfer):
    """The target run's starting density, from the source's converged one."""
    seed = calculation.starting_density() if transfer.seeded else None
    return transfer.apply(result.density, seed)


def promote_becsum(result, calculation, transfer: _SpinTransfer) -> tuple:
    """``becsum`` for every ultrasoft/PAW species, promoted the same way.

    Falls back to the *target's* atomic ``becsum`` (``PAW_atomic_becsum``) --
    per species, so one swapped dataset does not throw away the rest -- when the
    source carried none, or when the two runs' projector counts disagree. The
    second case is a swapped pseudopotential rather than a change of spin
    regime, which is what switching spin-orbit coupling on means for an
    ultrasoft or PAW dataset, and the atoms of the target's own file are a
    better guess than a reshaped ``becsum`` from someone else's.
    """
    if not calculation.is_ultrasoft:
        return ()
    source = tuple(getattr(result, "becsum", ()) or ())
    seeds = calculation.starting_becsum()
    if not source:
        return seeds
    if len(source) != len(seeds):
        warnings.warn(
            "the source run has becsum for a different number of species than "
            "the target; starting from the target's atomic becsum instead",
            RuntimeWarning,
            stacklevel=3,
        )
        return seeds
    out = []
    for values, seed in zip(source, seeds):
        if values is None or seed is None:
            out.append(seed)
            continue
        if np.shape(values)[1:] != np.shape(seed)[1:]:
            warnings.warn(
                f"the source becsum has shape {tuple(np.shape(values))[1:]} "
                f"per channel where the target needs "
                f"{tuple(np.shape(seed))[1:]} -- a different pseudopotential, "
                "not a different spin regime; starting from the target's "
                "atomic becsum for this species",
                RuntimeWarning,
                stacklevel=3,
            )
            out.append(seed)
            continue
        out.append(transfer.apply(values, seed))
    return tuple(out)


def promote_ns(result, calculation):
    """The Hubbard occupation matrix, ``(nspin, nslot, ldmx, ldmx)``.

    ``ns`` is per *channel* for every ``nspin`` -- ``new_ns`` halves it in the
    unpolarized case -- so 1 -> 2 is the same matrix in both channels and 2 -> 1
    is their average, with no factor anywhere. There is no noncollinear form:
    ``ns_nc`` is refused by name in :mod:`pypresso.hubbard`, so a Hubbard run
    cannot cross into ``nspin = 4`` at all.
    """
    if not calculation.is_hubbard:
        return None
    ns = getattr(result, "ns", None)
    if ns is None:
        return None
    if calculation.nspin == 4:
        raise NotImplementedError(
            "a Hubbard U in a noncollinear calculation needs ns_nc, which is "
            "refused by name (PLAN.md P20); drop the HUBBARD card or stay "
            "collinear"
        )
    ns = jnp.asarray(ns)
    source, target = ns.shape[0], calculation.nspin
    if ns.shape[1:] != (calculation.hubbard.nslot, calculation.hubbard.ldmx,
                        calculation.hubbard.ldmx):
        warnings.warn(
            f"the source ns has shape {tuple(ns.shape)} where the target needs "
            f"(*, {calculation.hubbard.nslot}, {calculation.hubbard.ldmx}, "
            f"{calculation.hubbard.ldmx}); starting from init_ns instead",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
    if source == target:
        return ns
    if source == 1 and target == 2:
        # The same occupation in both channels: an unpolarized ``ns`` is
        # already half the total, so this is a copy and not a halving.
        return jnp.concatenate([ns, ns])
    if source == 2 and target == 1:
        return jnp.mean(ns, axis=0, keepdims=True)
    raise NotImplementedError(
        f"no ns promotion from nspin = {source} to nspin = {target}"
    )


def promote_wavefunctions(result, calculation):
    """The converged states as a *span* for the target's first Rayleigh-Ritz.

    What comes back is not the target's wavefunctions -- it is a set of vectors
    to diagonalise the target Hamiltonian inside, exactly as ``wfcinit`` uses
    the pseudo-atomic orbitals. That is what makes the transfer safe: the span
    need not be orthonormal in the target's overlap operator (with spin-orbit
    coupling ``S`` mixes the components and it is not), need not have the right
    number of vectors, and need not be ordered.

    * **1 -> 2** the same states seed both channels, which is what the atomic
      start does too -- what splits the channels is the Hamiltonian they are
      then diagonalised inside.
    * **2 -> 4** the two channels become the two components of ``2 nbnd``
      spinors, ``(psi_up, 0)`` and ``(0, psi_dw)``. This is the same
      construction :meth:`~pypresso.scf.Calculation._as_spinors` performs on the
      atomic orbitals, applied to states that are already self-consistent, and
      the doubled count is exactly the ``nbnd`` a noncollinear run asks for.
    * **4 -> 4** (switching spin-orbit coupling on or off) the spinors carry
      over untouched: the two runs differ in ``D_ij`` and in nothing that a
      wavefunction is stored against.

    ``None`` -- fall back to the atomic orbitals -- whenever the two runs do not
    share a plane-wave basis or a k-point set, which a *magnetic* noncollinear
    target usually does not: its symmetry group is smaller, so ``irreducible_BZ``
    hands it more k-points than the collinear run had.
    """
    psi = getattr(result, "wavefunctions", None)
    if psi is None:
        return None
    psi = jnp.asarray(psi)
    npwx = calculation.basis.npwx
    # The number of *states*, which for a spiral is not the number of entries in
    # the basis list -- that one is doubled, ``k + q/2`` beside ``k - q/2``.
    nk = calculation.system.kpoints.nk
    source_npol = 2 if int(result.nspin) == 4 else 1
    if psi.shape[1] != nk or psi.shape[-1] != source_npol * npwx:
        return _dropped(
            f"the source run has {psi.shape[1]} k-points and {psi.shape[-1]} "
            f"coefficients per state where the target has {nk} and "
            f"{calculation.npol * npwx}"
        )
    if not _same_kpoints(result, calculation):
        return _dropped(
            "the two runs' k-point sets differ -- a magnetic noncollinear run "
            "has a smaller symmetry group, so irreducible_BZ gives it k-points "
            "the collinear run never had"
        )

    if calculation.npol == source_npol:
        # Same state layout on both sides. One channel per Hamiltonian when the
        # counts agree, and the first channel shared when they do not.
        if psi.shape[0] == _channels(calculation):
            return psi
        return psi[0]
    if source_npol == 2:
        return _dropped(
            "a spinor cannot be split back into two collinear channels: its "
            "components are not separately normalised, and with spin-orbit "
            "coupling they are not separately eigenstates either"
        )
    if calculation.spiral:
        return _dropped(
            "the target is a spin spiral, whose two components live on "
            "different plane-wave spheres (k +- q/2), so a scalar state does "
            "not embed in it"
        )
    # 1 -> 4 and 2 -> 4: the two channels become the two components.
    up = psi[0]
    down = psi[-1]
    zero_up, zero_down = jnp.zeros_like(up), jnp.zeros_like(down)
    return jnp.concatenate(
        [
            jnp.concatenate([up, zero_up], axis=-1),
            jnp.concatenate([zero_down, down], axis=-1),
        ],
        axis=1,
    )


def _channels(calculation) -> int:
    """How many Hamiltonians one iteration solves: 2 for LSDA, 1 otherwise.

    A noncollinear run has *one*, on a space twice as large, which is why this
    is not ``nspin`` (``Calculation.hamiltonian`` returns a one-element tuple
    there) and not ``nspin_mag`` either.
    """
    return 1 if calculation.noncolin else int(calculation.nspin)


def _same_kpoints(result, calculation) -> bool:
    """Whether the two runs sample the same points, not merely as many."""
    system = getattr(result, "system", None)
    if system is None:
        # Nothing to compare against: the shapes matched, which is the most
        # that can be checked, and a mismatch of coordinates at equal counts
        # would need two different symmetry groups of the same order.
        return True
    source = np.asarray(system.kpoints.coords)
    target = np.asarray(calculation.system.kpoints.coords)
    return source.shape == target.shape and bool(
        np.allclose(source, target, atol=1.0e-8)
    )


def _dropped(reason: str):
    warnings.warn(
        f"the converged wavefunctions are not being carried over: {reason}. "
        "The run starts from the pseudo-atomic orbitals instead, which costs a "
        "few Davidson steps and changes nothing else.",
        RuntimeWarning,
        stacklevel=3,
    )
    return None


@dataclass(frozen=True)
class ContinuedState:
    """A converged state expressed in another regime's variables.

    The three mixed quantities plus the span for the first diagonalisation --
    everything ``run_scf`` needs to start where another run stopped.
    """

    density: jnp.ndarray
    becsum: tuple = ()
    ns: jnp.ndarray | None = None
    wavefunctions: jnp.ndarray | None = None
    #: ``(source nspin_mag, target nspin_mag)`` -- what a caller needs in order
    #: to say what it just did.
    regimes: tuple = (1, 1)
    #: What happened to the magnetization: ``"carry"``, ``"seed"`` or
    #: ``"none"``. Resolved, so ``"auto"`` never appears here.
    magnetization: str = "none"

    @property
    def seeded(self) -> bool:
        """Whether the magnetization came from ``starting_magnetization``."""
        return self.magnetization == "seed"

    @property
    def description(self) -> str:
        source, target = self.regimes
        if target == 1:
            how = "dropped (the target carries none)"
        elif self.magnetization == "none":
            how = "left at zero"
        elif self.seeded:
            how = "seeded from starting_magnetization"
        else:
            how = "carried"
        return (
            f"nspin_mag {source} -> {target}, magnetization {how}, "
            f"wavefunctions {'carried' if self.wavefunctions is not None else 'dropped'}"
        )


def continued_state(
    result,
    calculation,
    magnetization: str = "auto",
    wavefunctions: bool = True,
) -> ContinuedState:
    """Express a converged :class:`~pypresso.scf.SCFResult` in ``calculation``'s regime.

    Args:
        result: what the previous ``run_scf`` returned.
        calculation: the :class:`~pypresso.scf.Calculation` the next run will
            use -- it is what says which regime is being promoted *to*, and it
            carries the atomic seeds the promotion may need.
        magnetization: ``"auto"`` carries the source's magnetization when it has
            one and seeds the target's ``starting_magnetization`` when it does
            not; ``"carry"`` insists on the former and raises rather than
            silently starting on a symmetric solution; ``"seed"`` insists on the
            latter, which is how a *different* magnetic state is reached from
            the same converged charge; ``"none"`` starts unpolarized, which is
            the only way to hand the target a state it can converge away from
            without the seed choosing the answer.
        wavefunctions: whether to carry the converged states over as the span
            for the first Rayleigh-Ritz. They are dropped with a warning
            whenever the two runs do not share a basis and a k-set.
    """
    _check_grid(result, calculation)
    transfer = _transfer(result, calculation, magnetization)
    span = None
    if wavefunctions:
        span = promote_wavefunctions(result, calculation)
    return ContinuedState(
        density=promote_density(result, calculation, transfer),
        becsum=promote_becsum(result, calculation, transfer),
        ns=promote_ns(result, calculation),
        wavefunctions=span,
        regimes=(transfer.source, transfer.target),
        magnetization=transfer.mode,
    )


def nc_magnetization_from_lsda(density, direction):
    """``potinit.f90``'s rotation of a collinear density onto an arbitrary axis.

    The force theorem's whole handoff (:mod:`pypresso.workflows.anisotropy`).
    QE writes it in G space on ``rho%of_g`` and in place::

        rho(:,4) = rho(:,2)*cos(theta)
        rho(:,2) = rho(:,2)*sin(theta)
        rho(:,3) = rho(:,2)*sin(phi)
        rho(:,2) = rho(:,2)*cos(phi)

    -- four lines whose second and third read the value the line above just
    wrote, which is correct and worth reading twice. Here it is the same
    rotation written through :func:`spin_components`: the charge is untouched
    and the collinear ``m_z`` is laid along ``direction``.

    **It is a rotation of the density and of nothing else.** ``potinit`` reaches
    it through ``read_rhog`` rather than ``read_scf``, so no ``becsum``, no
    ``ns`` and no ``tau`` cross with it -- which is exactly why QE refuses PAW
    on this path (``potinit.f90:98``): a PAW Hamiltonian needs ``ddd_paw``, and
    ``becsum`` is not in the handoff. An ultrasoft dataset is fine, because its
    augmentation charge is already inside the density being handed over.

    ``density`` is ``(2, ...)`` or ``(4, ...)``; ``direction`` is a unit vector.
    A ``(4, ...)`` density is rotated *from its own axis*, so promoting an
    already-noncollinear collinear-along-z state is the same operation.
    """
    direction = np.asarray(direction, dtype=float)
    norm = float(np.sqrt(np.sum(direction**2)))
    if norm < DIRECTION_TOL:
        raise ValueError("direction must be a non-zero vector")
    direction = direction / norm

    density = jnp.asarray(density)
    nspin_mag = density.shape[0]
    if nspin_mag not in (2, 4):
        raise ValueError(
            f"nc_magnetization_from_lsda wants a magnetic density, got "
            f"nspin_mag = {nspin_mag}"
        )
    charge, moment = spin_components(density, nspin_mag)
    if nspin_mag == 4:
        # Already a vector field: rotate it off *its* axis rather than off z,
        # so that this is idempotent on a state that is already along
        # ``direction`` and so that a second call cannot silently re-tilt one.
        along = _collinear_axis(density)
        scalar = jnp.sum(_axis(along or (0.0, 0.0, 1.0), moment.ndim) * moment, axis=0)
    else:
        scalar = moment[2]
    return from_spin_components(charge, _axis(direction, scalar.ndim + 1) * scalar, 4)


def direction_from_angles(angle1: float, angle2: float) -> tuple:
    """QE's ``(angle1, angle2)`` in degrees as a cartesian unit vector.

    ``theta`` from ``z`` and ``phi`` from ``x`` in the ``xy`` plane, the
    convention ``INPUT_PW.txt`` states for ``angle1``/``angle2`` and
    ``local_moments`` already uses -- repeated here because the force theorem
    takes its direction from *species one's* angles for the whole cell
    (``nc_magnetization_from_lsda``), where ``local_moments`` takes each
    species' own.
    """
    theta = np.radians(float(angle1))
    phi = np.radians(float(angle2))
    return (
        float(np.sin(theta) * np.cos(phi)),
        float(np.sin(theta) * np.sin(phi)),
        float(np.cos(theta)),
    )
