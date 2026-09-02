"""Raman and infrared spectra: the tensors projected on the phonon modes.

``PLAN.md`` P36. Everything under this module was computed by an earlier phase
and none of it is what a spectroscopist reads. The Raman tensors P35 gives are
**per atom** -- ``d(eps_ij)/d(tau_(a,c))``, a rank-3 object with an atom label --
and the Born effective charges P24b gives are per atom too, while what an
experiment resolves is a **mode**: a frequency, an intensity, and a
depolarisation ratio. The projection between the two is the phonon
eigendisplacement, so this module is the contraction

    R^(nu)_ij = sum_(a,c) d(chi_ij)/d(tau_(a,c)) z_(a,c)^(nu),
    p^(nu)_i  = sum_(a,c) Z*_(a,i,c) z_(a,c)^(nu),

followed by the two rotational invariants of ``R`` that a powder average leaves
-- ``alpha = tr(R)/3`` and ``beta^2`` -- and the standard combinations
``45 alpha^2 + 7 beta^2`` (the activity) and ``3 beta^2/(45 alpha^2 + 4 beta^2)``
(the depolarisation ratio). Those are Placzek's, and the form used here is the
one of Porezag and Pederson, `Phys. Rev. B 54, 7830 (1996)
<https://doi.org/10.1103/PhysRevB.54.7830>`_, which is the reference
``dynmat_sub.f90`` itself cites.

**This is arithmetic, and it is transcribed rather than derived**, because QE
has the routine and -- unusually for anything third-order here -- that routine
is *not* the one that regressed. ``LR_Modules/dynmat_sub.f90``'s ``RamanIR``,
reached by ``dynmat.x``, is pure post-processing: it reads ``dchi_dtau``,
``zstar`` and ``eps0`` from a dynamical-matrix file and does the contraction
above. Nothing in it touches the third-derivative code path that
``PLAN.md`` P35 establishes is broken in the vendored 7.5 build, so it is a
usable reference in the way ``ph.x``'s ``lraman`` branch is not.
:func:`~defumat.io.dynmat.write_dynamical_matrix` writes that file and the
regression test runs the vendored ``dynmat.x`` on it.

**What a degenerate multiplet does to these numbers is the trap.** The
eigensolver fixes the basis of a degenerate manifold arbitrarily (rule D4), and
``alpha``, ``beta^2`` and the depolarisation ratio of a *single* mode inside one
are not invariant under the orthogonal mixing that leaves the manifold alone.
The **sum** over a manifold is -- both invariants are quadratic forms in ``R``
-- so :meth:`VibrationalSpectrum.by_manifold` is the form of the result that can
be compared against another code, and per-mode numbers inside a multiplet are
reported but must not be. Silicon's Raman-active ``T_2g`` triplet happens to be
per-mode invariant as well (its tensors are traceless and its ``beta^2`` is the
same for every member), which is an accident of that mode and not a licence.

**The long-range electric field is here as of P55**, and it is what the two
paragraphs that used to end this docstring named. A polar crystal's ``Gamma``
dynamical matrix is not analytic: an optical mode that moves the ions against
each other builds a macroscopic field, the field depends on the *direction* the
zone centre is approached from, and the matrix acquires the rank-one term

    D_(a i, b j) += (4 pi e^2/Omega) (q.Z*_a)_i (q.Z*_b)_j / (q.eps.q)

(:func:`nonanal`, ``rigid.f90``). It raises the longitudinal mode and leaves the
transverse ones alone, which is the LO-TO splitting, and it is zero in a
non-polar crystal because ``Z*`` is. Contracting the same two ingredients the
other way gives the **ionic permittivity** (:func:`polar_mode_permittivity`,
``dynmat_sub.f90``, after Fennie and Rabe, `Phys. Rev. B 68, 184111 (2003)
<https://doi.org/10.1103/PhysRevB.68.184111>`_): each polar mode contributes
``4 pi e^2 p_i p_j/(Omega omega^2)`` to the static dielectric constant, where
``p`` is the mode dipole the infrared activity is already built from. So
``eps_0`` and ``eps_infinity`` are two ends of the same contraction, and the two
of them together with the frequencies satisfy the Lyddane-Sachs-Teller relation,
which is the check that neither piece can pass alone (§"how it is validated" in
``tests/regression/test_spectra.py``).

**Only the permittivity's constant is written down rather than transcribed.**
``polar_mode_permittivity`` reaches it through a chain of SI constants and a
conversion to THz; in Rydberg units it is ``4 pi e^2/Omega`` with the mode
dipole in ``e/sqrt(Ry mass)``, and the two agree to **1.6e-12**, which is QE's
own constants' round-off. That agreement is a test
(``test_the_ionic_permittivity_constant_is_qes``), because a wrong constant here
is a plausible number rather than a broken one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from defumat.units import (
    AMU_TO_RY, BOHR_TO_ANGSTROM, E2, FPI, RY_TO_CMM1, RY_TO_THZ,
)

__all__ = [
    "VibrationalSpectrum",
    "eigendisplacements",
    "mode_activities",
    "degenerate_manifolds",
    "nonanal",
    "loto_modes",
    "neutral_born_charges",
    "polar_mode_permittivity",
    "vibrational_spectrum",
]

#: The elementary charge in Debye/Angstrom -- ``dynmat_sub.f90``'s literal.
#: ``1 e = 4.80324e-10 esu = 4.80324 Debye/A``, and it is written out here
#: rather than derived from :mod:`defumat.units` because it is what fixes the
#: infrared activities' unit against QE's, to the digit QE prints.
ELECTRON_DEBYE_PER_ANGSTROM = 4.80324

#: The gap in cm^-1 below which two neighbouring frequencies are one multiplet.
#: Loose by four orders and deliberately so: a degeneracy the *symmetry* imposes
#: comes out at **2e-13 cm^-1** on silicon's optical triplet and 2e-11 on its
#: acoustic one, where the nearest thing it has to be told apart from is 500
#: cm^-1 away. Anything between those two scales would do; 1.0 is chosen because
#: it is also below any splitting worth reporting as physics.
DEGENERACY_TOLERANCE = 1.0

#: ``omega^2`` in Ry^2 below which a mode is left out of the ionic
#: permittivity. ``polar_mode_permittivity``'s ``eps8``, verbatim -- it is 11
#: cm^-1, so what it drops is the acoustic branch, whose mode dipole the
#: translational sum rule silences and whose ``1/omega^2`` would otherwise
#: divide a residue by a residue. An **imaginary** mode is dropped by the same
#: test, since its ``omega^2`` is negative: a crystal that is unstable has no
#: static dielectric constant to report.
HARD_MODE_OMEGA2 = 1.0e-8


@dataclass
class VibrationalSpectrum:
    """The per-mode Raman and infrared activities of a crystal at ``Gamma``.

    Every array has ``3 nat`` entries, ordered by frequency as
    :class:`~defumat.response.phonon.Phonons` orders them.
    """

    #: cm^-1, an imaginary frequency carried as a negative number (``dyndia``).
    frequencies: np.ndarray
    #: The same frequencies in THz, which is the other column ``dynmat.x`` prints.
    frequencies_thz: np.ndarray
    #: ``(3 nat, 3, 3)`` -- the Raman tensor of each mode, in ``A^2`` per square
    #: root of a Rydberg mass unit. ``None`` when no Raman tensors were given.
    mode_raman: np.ndarray | None
    #: ``(3 nat,)`` -- ``tr(R)/3``, the isotropic invariant.
    alpha: np.ndarray | None
    #: ``(3 nat,)`` -- the anisotropy invariant of Placzek's average.
    beta2: np.ndarray | None
    #: ``(3 nat,)`` -- ``(45 alpha^2 + 7 beta^2)`` in ``A^4/amu``, which is the
    #: powder Raman activity and the column ``dynmat.x`` labels ``Raman``.
    raman_activity: np.ndarray | None
    #: ``(3 nat,)`` -- ``3 beta^2/(45 alpha^2 + 4 beta^2)``, between 0 and 3/4.
    #: ``0`` where the activity is zero, since the ratio is then ``0/0``.
    depolarisation: np.ndarray | None
    #: ``(3 nat, 3)`` -- the mode dipole ``sum Z* z``, in ``e/sqrt(Ry mass)``.
    mode_dipole: np.ndarray | None
    #: ``(3 nat,)`` -- the infrared activity in ``(D/A)^2/amu``. ``None`` when
    #: no Born charges were given.
    infrared: np.ndarray | None
    #: ``(3, 3)`` -- the electronic dielectric tensor the tensors sit on.
    epsilon: np.ndarray
    #: ``(3, 3)`` -- the electronic polarizability ``Omega chi/(4 pi)`` in
    #: ``A^3``, which is the table ``dynmat.x`` prints above the modes.
    polarizability: np.ndarray
    #: The Clausius-Mossotti factor ``3/(2 + tr(eps)/3)``. QE prints it and
    #: applies it to nothing; it is the correction to a *molecular*
    #: polarizability and is carried for the same reason.
    clausius_mossotti: float
    #: ``(3 nat,)`` -- which multiplet each mode belongs to, from
    #: :func:`degenerate_manifolds`.
    manifold: np.ndarray
    #: ``(3 nat, 3)`` -- the mode effective charge ``Z~*``, which is
    #: :attr:`mode_dipole` in the units ``dynmat.x`` prints it
    #: (``e bohr/sqrt(2)``, its ``Z~*_x`` columns). ``None`` without ``Z*``.
    mode_effective_charges: np.ndarray | None = None
    #: ``(3, 3)`` -- the polar modes' contribution to the static dielectric
    #: tensor, ``4 pi e^2/Omega sum_nu p p/omega^2``. Zero in a non-polar
    #: crystal and ``None`` without ``Z*``.
    ionic_permittivity: np.ndarray | None = None
    #: ``(3, 3)`` -- ``epsilon + ionic_permittivity``: the **static** dielectric
    #: tensor, which is what a capacitance measures where :attr:`epsilon` is
    #: what an optical one does. ``None`` without ``Z*``.
    static_permittivity: np.ndarray | None = None
    #: The cartesian direction the zone centre was approached from, or ``None``
    #: for the analytic matrix. When it is set the frequencies are **LO**, and
    #: then :attr:`static_permittivity` is not the static dielectric constant:
    #: the sum over modes has to run over the transverse ones, which is
    #: precisely what a direction removes. ``dynmat.x`` prints "BEWARE" and the
    #: same number; :func:`vibrational_spectrum` computes the permittivity from
    #: the analytic modes instead, so this warning applies only to
    #: :func:`mode_activities` called by hand.
    loto_direction: tuple | None = None
    #: ``(3 nat,)`` in cm^-1 -- the frequencies of the **analytic** matrix,
    #: kept beside :attr:`frequencies` when a direction was given so that the
    #: splitting is a subtraction rather than a second calculation. ``None``
    #: when there was no direction, where :attr:`frequencies` are already these.
    transverse_frequencies: np.ndarray | None = None

    def by_manifold(self):
        """The activities summed over each degenerate multiplet.

        **This is the comparable form and the per-mode arrays are not**, inside
        a multiplet: the eigensolver's basis there is arbitrary and both
        invariants are quadratic in ``R``, so their sum over the multiplet is
        invariant and the individual terms are not (module docstring).

        Returns a list of ``(frequency, raman_activity, infrared)`` with one
        entry per multiplet, the activities summed and the frequency averaged.
        """
        out = []
        for group in range(int(self.manifold.max()) + 1):
            members = self.manifold == group
            out.append((
                float(self.frequencies[members].mean()),
                None if self.raman_activity is None
                else float(self.raman_activity[members].sum()),
                None if self.infrared is None
                else float(self.infrared[members].sum()),
            ))
        return out

    def table(self) -> str:
        """The mode table, in the columns ``dynmat.x`` prints them.

        A column that was not computed is **left out** rather than printed as
        zero, which is ``RamanIR``'s own convention (its ``noraman`` branch
        drops the two Raman columns and their header with them). Remember that
        the per-mode entries of a degenerate multiplet are not comparable
        against another code -- :meth:`by_manifold` is (class docstring).
        """
        head = "# mode   [cm-1]    [THz]"
        if self.infrared is not None:
            head += "      IR    "
        if self.raman_activity is not None:
            head += "      Raman   depol.fact"
        lines = [head]
        for mode, (freq, thz) in enumerate(
            zip(self.frequencies, self.frequencies_thz)
        ):
            row = f"{mode + 1:5d}{freq:10.2f}{thz:10.4f}"
            if self.infrared is not None:
                row += f"{self.infrared[mode]:10.4f}"
            if self.raman_activity is not None:
                row += f"{self.raman_activity[mode]:15.4f}"
                row += f"{self.depolarisation[mode]:10.4f}"
            lines.append(row)
        return "\n".join(lines)

    def plot(self, ax=None, kind: str = "raman", width: float = 8.0,
             grid=None, acoustic_cutoff: float = 20.0, **kwargs):
        """Draw the spectrum as a Lorentzian-broadened curve, and return the axes.

        A vibrational spectrum is read as a curve and computed as a set of
        sticks, and the twenty lines that turn one into the other are twenty
        lines of a script that are not about physics.

        ``kind`` is ``'raman'`` or ``'infrared'``. ``width`` is the Lorentzian
        half-width in cm^-1, which is an instrumental parameter rather than a
        physical one and is why it is exposed. Modes below ``acoustic_cutoff``
        are dropped: they are the acoustic branch, whose activity the sum rule
        silences, and a residual stick at 2 cm^-1 is numerical noise magnified
        by the normalisation.

        matplotlib is imported here rather than at module scope: it is not a
        dependency of any calculation, and a headless run should not need it.
        """
        import matplotlib.pyplot as plt

        activities = {"raman": self.raman_activity, "infrared": self.infrared}
        if kind not in activities:
            raise ValueError(
                f"kind must be one of {sorted(activities)}, not {kind!r}")
        activity = activities[kind]
        if activity is None:
            raise ValueError(
                f"this spectrum has no {kind} activity: it was built without "
                + ("Raman tensors" if kind == "raman" else "Born charges")
            )
        if ax is None:
            _, ax = plt.subplots()
        frequencies = np.asarray(self.frequencies)
        if grid is None:
            top = float(np.max(frequencies)) * 1.15 + 5.0 * width
            grid = np.linspace(0.0, top, 1400)
        grid = np.asarray(grid)
        curve = np.zeros_like(grid)
        for frequency, value in zip(frequencies, np.asarray(activity)):
            if frequency < acoustic_cutoff:
                continue
            curve += value * width**2 / ((grid - frequency) ** 2 + width**2)
        peak = float(np.max(curve))
        if peak > 0.0:
            curve = curve / peak
        kwargs.setdefault("lw", 1.6)
        colour = kwargs.setdefault("color", "C0" if kind == "raman" else "C3")
        ax.fill_between(grid, curve, color=colour, alpha=0.25)
        ax.plot(grid, curve, **kwargs)
        ax.set_xlabel(r"wavenumber   [cm$^{-1}$]")
        ax.set_ylabel(f"{kind} intensity   [normalised]")
        ax.set_xlim(float(grid[0]), float(grid[-1]))
        ax.set_ylim(bottom=0.0)
        return ax


def eigendisplacements(eigenvectors: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """``z``: the mass-weighted eigenvectors turned into displacements.

    ``dyndiag`` in ``PHonon/PH/rigid.f90``. The dynamical matrix is diagonalised
    mass-weighted, so its eigenvectors ``u`` are orthonormal and *not* what an
    atom does; the displacement pattern is ``z = u/sqrt(M)``, normalised by
    ``<z|M|z> = 1``. ``M`` is in **Rydberg** mass units, which is why
    :data:`~defumat.units.AMU_TO_RY` appears here and in
    :func:`~defumat.response.phonon._diagonalize` and nowhere between them.

    Args:
        eigenvectors: ``(3 nat, 3 nat)``, mode index **last**, as
            :attr:`~defumat.response.phonon.Phonons.eigenvectors` stores them.
        masses: ``(nat,)`` in amu.

    Returns ``(3 nat, nat, 3)`` indexed ``[mode, atom, cartesian]``.
    """
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    masses = np.asarray(masses, dtype=float)
    scale = 1.0 / np.sqrt(np.repeat(masses, 3) * AMU_TO_RY)
    nat = masses.size
    return (eigenvectors * scale[:, None]).T.reshape(3 * nat, nat, 3)


def degenerate_manifolds(
    frequencies: np.ndarray, tolerance: float = DEGENERACY_TOLERANCE
) -> np.ndarray:
    """Label each mode with the multiplet it belongs to.

    Frequencies arrive sorted, so a multiplet is a run of consecutive entries
    within ``tolerance`` cm^-1 of the one before. Crude, and right for the only
    thing it is used for: telling a symmetry-imposed degeneracy (exact, and
    split here only by the linear solves' residue) from two distinct modes.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    labels = np.zeros(frequencies.size, dtype=int)
    for mode in range(1, frequencies.size):
        same = abs(frequencies[mode] - frequencies[mode - 1]) <= tolerance
        labels[mode] = labels[mode - 1] + (0 if same else 1)
    return labels


def nonanal(
    force_constants: np.ndarray,
    born: np.ndarray,
    epsilon: np.ndarray,
    direction,
    volume: float,
) -> np.ndarray:
    """The non-analytic term: ``rigid.f90``'s ``nonanal``, verbatim.

    At ``q = 0`` exactly, a polar crystal's dynamical matrix is missing the
    macroscopic electric field an optical mode builds. The field is longitudinal
    and its size depends on the direction ``q`` from which the zone centre is
    approached, so the ``Gamma`` matrix has a rank-one term that is a function
    of a *direction* rather than of a point -- which is why a phonon dispersion
    is discontinuous at ``Gamma`` and why this is applied here rather than
    inside the second derivative that produced the matrix::

        D_(a i, b j) += (4 pi e^2/Omega) (q . Z*_a)_i (q . Z*_b)_j / (q . eps . q)

    Both factors are dot products of ``q`` with a Born charge, so the term
    vanishes identically wherever ``Z*`` does and a non-polar crystal is
    untouched. It is homogeneous of degree zero in ``q``: only the direction
    matters, and the vector need not be normalised.

    Args:
        force_constants: ``(nat, 3, nat, 3)`` in Ry/bohr^2 --
            :attr:`~defumat.response.phonon.Phonons.matrix`.
        born: ``(nat, 3, 3)`` indexed ``[atom, field, displacement]``, in ``e``.
        epsilon: ``(3, 3)``, the **electronic** dielectric tensor.
        direction: three cartesian components, any length.
        volume: the cell volume in bohr^3.

    Returns the corrected ``(nat, 3, nat, 3)`` matrix. Raises ``ValueError``
    for a direction with no length, where ``nonanal`` writes "a direction for q
    was not specified" and returns the matrix unchanged -- a refusal rather than
    a silent no-op, since a caller that asked for LO modes and got TO ones back
    has no way to tell.
    """
    q = np.asarray(direction, dtype=float).reshape(3)
    epsilon = np.asarray(epsilon, dtype=float)
    qeq = float(q @ epsilon @ q)
    if qeq < 1.0e-8:
        raise ValueError(
            "the non-analytic term needs a direction to approach q = 0 from, "
            f"and q . eps . q = {qeq:.3e} for q = {tuple(q)}"
        )
    born = np.asarray(born, dtype=float)
    # ``zag(i) = sum_alpha q(alpha) zeu(alpha, i, na)``: the sum is over the
    # **field** label of ``Z*`` and the surviving one is the displacement, which
    # is this code's ``[atom, field, displacement]`` in that order.
    zag = np.einsum("a,kai->ki", q, born)
    term = FPI * E2 * np.einsum("ki,lj->kilj", zag, zag) / (qeq * volume)
    return np.asarray(force_constants, dtype=float) + term


def neutral_born_charges(born: np.ndarray) -> np.ndarray:
    """``Z*`` with the charge-neutrality violation shared out over the atoms.

    ``set_asr``'s ``asr = 'simple'`` branch (``LR_Modules/dynmat_sub.f90``),
    which is one line: ``zeu -= sum_atoms zeu/nat``, component by component.

    **It is not cosmetic, and the LO-TO splitting is where that shows.** The
    exact ``sum_a Z*_a = 0`` says a rigid translation of the crystal builds no
    field; a calculated ``Z*`` misses it by the basis-set error (AlAs at
    ``ecutwfc = 10``: -1.257 against charges of 1.925 and -3.181), and
    :func:`nonanal` then gives the cell a net charge and lifts a
    **longitudinal acoustic** mode -- 1.8 to 33.8 cm^-1 on that cell, where the
    acoustic modes must stay at zero whatever the field does. The
    Lyddane-Sachs-Teller relation is what measures it: violated by 1.6e-3 with
    the raw charges and by **5.0e-11** with these.

    ``dynmat.x`` applies the same correction under every ``asr`` but ``'no'``.
    """
    born = np.asarray(born, dtype=float)
    return born - born.mean(axis=0)


def loto_modes(
    force_constants: np.ndarray,
    masses: np.ndarray,
    born: np.ndarray,
    epsilon: np.ndarray,
    direction,
    volume: float,
):
    """The LO frequencies and displacement patterns along ``direction``.

    :func:`nonanal` followed by the same mass-weighted diagonalisation the
    analytic matrix gets, which is ``dynmat.x``'s order of operations. Returns
    ``(frequencies, eigenvectors)`` shaped as
    :class:`~defumat.response.phonon.Phonons` carries them -- cm^-1 signed, and
    the mode index last.
    """
    from defumat.response.phonon import _diagonalize

    matrix = nonanal(force_constants, born, epsilon, direction, volume)
    return _diagonalize(matrix, np.asarray(masses, dtype=float))


def polar_mode_permittivity(
    frequencies: np.ndarray,
    mode_dipole: np.ndarray,
    epsilon: np.ndarray,
    volume: float,
):
    """The ionic contribution to the static dielectric tensor.

    ``LR_Modules/dynmat_sub.f90``'s ``polar_mode_permittivity``, after Fennie
    and Rabe, `Phys. Rev. B 68, 184111 (2003)
    <https://doi.org/10.1103/PhysRevB.68.184111>`_. Every polar mode is an
    oscillator that a static field drives, so it adds

        d(eps)_ij = (4 pi e^2/Omega) p_i p_j / omega^2

    to the electronic dielectric constant, where ``p`` is the mode dipole
    ``sum_(a,c) Z*_(a,i,c) z_(a,c)`` that the infrared activity is already the
    square of. A soft mode contributes divergently, which is the physics of a
    ferroelectric and the reason :data:`HARD_MODE_OMEGA2` is a threshold rather
    than a filter on the acoustic branch by index.

    **The constant is derived here and transcribed in QE**, which is the one
    departure in this module. ``polar_mode_permittivity`` reaches it through
    ``e^2/(eps_0 a_0^3 amu)`` and a conversion of ``omega`` to THz; in Rydberg
    units the same quantity is ``4 pi e^2/Omega`` with ``omega`` in Ry and the
    dipole in ``e/sqrt(Ry mass)``, since the Rydberg mass unit is what
    :func:`eigendisplacements` already normalises against. The two agree to
    1.6e-12 and there is a test that says so.

    Args:
        frequencies: ``(3 nat,)`` in cm^-1, signed.
        mode_dipole: ``(3 nat, 3)`` in ``e/sqrt(Ry mass)`` --
            :attr:`VibrationalSpectrum.mode_dipole`.
        epsilon: ``(3, 3)``, the electronic dielectric tensor.
        volume: the cell volume in bohr^3.

    Returns ``(ionic, static)``, both ``(3, 3)``.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    mode_dipole = np.asarray(mode_dipole, dtype=float)
    omega2 = np.sign(frequencies) * (frequencies / RY_TO_CMM1) ** 2
    hard = omega2 > HARD_MODE_OMEGA2
    weight = np.where(hard, 1.0 / np.where(hard, omega2, 1.0), 0.0)
    ionic = FPI * E2 / volume * np.einsum(
        "n,ni,nj->ij", weight, mode_dipole, mode_dipole
    )
    return ionic, np.asarray(epsilon, dtype=float) + ionic


def mode_activities(
    frequencies: np.ndarray,
    eigenvectors: np.ndarray,
    masses: np.ndarray,
    epsilon: np.ndarray,
    volume: float,
    raman: np.ndarray | None = None,
    born: np.ndarray | None = None,
    tolerance: float = DEGENERACY_TOLERANCE,
    loto_direction=None,
) -> VibrationalSpectrum:
    """``RamanIR``: the whole of the assembly, and nothing that solves anything.

    A transcription of ``LR_Modules/dynmat_sub.f90``'s ``RamanIR``, kept a pure
    function of arrays so that it is testable without a self-consistent run
    behind it -- which is what lets the ``dynmat.x`` comparison be a comparison
    of *this* against the Fortran rather than of two whole calculations.

    Args:
        frequencies: ``(3 nat,)`` in cm^-1, signed
            (:attr:`~defumat.response.phonon.Phonons.frequencies`).
        eigenvectors: ``(3 nat, 3 nat)``, mode index last.
        masses: ``(nat,)`` in amu.
        epsilon: ``(3, 3)`` electronic dielectric tensor.
        volume: the cell volume in bohr^3.
        raman: ``(nat, 3, 3, 3)`` indexed ``[atom, cart, i, j]`` in **inverse
            bohr** -- :attr:`~defumat.response.nonlinear.RamanTensors.raman`,
            not the Angstrom-squared form, which this converts to itself.
        born: ``(nat, 3, 3)`` indexed ``[atom, field, displacement]`` -- P24b's
            Born effective charges, in units of ``e``.
        loto_direction: recorded on the result and used for nothing else. The
            frequencies and eigenvectors handed in are whatever the caller
            diagonalised; :func:`loto_modes` is what makes them LO ones, and
            passing the direction here is how the result says which they are.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    epsilon = np.asarray(epsilon, dtype=float)
    z = eigendisplacements(eigenvectors, masses)

    # The electronic polarizability table ``dynmat.x`` prints, in A^3.
    chi = epsilon - np.eye(3)
    polarizability = chi * BOHR_TO_ANGSTROM**3 * volume / FPI
    clausius_mossotti = 3.0 / (2.0 + np.trace(epsilon) / 3.0)

    mode_dipole = infrared = mode_effective_charges = None
    ionic_permittivity = static_permittivity = None
    if born is not None:
        # ``polar(i) = sum_(a,j) zstar(i,j,a) z(a,j)``: QE's ``zstar`` is
        # ``(field, displacement, atom)``, which is this code's ``[a, i, j]``
        # with the same two meanings in the same order.
        mode_dipole = np.einsum("aij,naj->ni", np.asarray(born, dtype=float), z)
        # ``irfac``, and the factor of two beside it, verbatim: QE writes
        # ``2*(...)*irfac`` with ``irfac = 4.80324^2/2*amu_ry``, so the twos
        # cancel and are kept apart because that is where the convention lives.
        irfac = ELECTRON_DEBYE_PER_ANGSTROM**2 / 2.0 * AMU_TO_RY
        infrared = 2.0 * np.sum(mode_dipole**2, axis=1) * irfac
        # ``meffc``, which is the same contraction in QE's printed units:
        # ``z`` carries ``1/sqrt(amu_ry M)`` and ``meffc`` puts one of the two
        # back. Nothing downstream uses it -- it is a column ``dynmat.x``
        # prints under ``lplasma`` and therefore a comparable number.
        mode_effective_charges = mode_dipole * np.sqrt(AMU_TO_RY)
        ionic_permittivity, static_permittivity = polar_mode_permittivity(
            frequencies, mode_dipole, epsilon, volume
        )

    mode_raman = alpha = beta2 = activity = depolarisation = None
    if raman is not None:
        # ``d(chi)/d(tau)`` in A^2, which is what ``write_ramtns`` puts on file
        # and what ``RamanIR`` expects: ``Omega/(4 pi)`` times the derivative of
        # ``eps``, with bohr^2 converted.
        dchi = np.asarray(raman, dtype=float) * volume / FPI * BOHR_TO_ANGSTROM**2
        mode_raman = np.einsum("acij,nac->nij", dchi, z)

        alpha = np.trace(mode_raman, axis1=1, axis2=2) / 3.0
        diagonal = np.diagonal(mode_raman, axis1=1, axis2=2)
        beta2 = 0.5 * (
            (diagonal[:, 0] - diagonal[:, 1]) ** 2
            + (diagonal[:, 0] - diagonal[:, 2]) ** 2
            + (diagonal[:, 1] - diagonal[:, 2]) ** 2
            + 6.0 * (
                mode_raman[:, 0, 1] ** 2
                + mode_raman[:, 0, 2] ** 2
                + mode_raman[:, 1, 2] ** 2
            )
        )
        activity = (45.0 * alpha**2 + 7.0 * beta2) * AMU_TO_RY
        # ``0/0`` at an inactive mode, which every acoustic mode is: the ratio
        # is reported as zero rather than as a nan, and QE prints nothing there
        # at all (``noraman`` drops the column).
        denominator = 45.0 * alpha**2 + 4.0 * beta2
        active = denominator > 0.0
        depolarisation = np.where(
            active, 3.0 * beta2 / np.where(active, denominator, 1.0), 0.0
        )

    return VibrationalSpectrum(
        frequencies=frequencies,
        frequencies_thz=frequencies / RY_TO_CMM1 * RY_TO_THZ,
        mode_raman=mode_raman,
        alpha=alpha,
        beta2=beta2,
        raman_activity=activity,
        depolarisation=depolarisation,
        mode_dipole=mode_dipole,
        infrared=infrared,
        epsilon=epsilon,
        polarizability=polarizability,
        clausius_mossotti=float(clausius_mossotti),
        manifold=degenerate_manifolds(frequencies, tolerance),
        mode_effective_charges=mode_effective_charges,
        ionic_permittivity=ionic_permittivity,
        static_permittivity=static_permittivity,
        loto_direction=None if loto_direction is None
        else tuple(float(value) for value in loto_direction),
    )


def vibrational_spectrum(
    calculation,
    result,
    raman: object | None = None,
    phonons: object | None = None,
    infrared: bool = True,
    loto_direction=None,
    neutralize: bool = False,
    verbose: bool = False,
    **response_options,
):
    """The ``Gamma`` spectrum of a converged insulator, tensors and all.

    Three objects go into a spectrum and this solves whichever were not handed
    in: the Raman tensors (P35), the dynamical matrix (P25) and the Born
    effective charges (P24b). **The first two share their expensive half** --
    the displacement response -- so they are solved once here and threaded,
    which is what :func:`~defumat.response.nonlinear.raman_tensors`'s
    ``keep_internals`` and :func:`~defumat.response.phonon.dynamical_matrix`'s
    ``response`` exist for. The Born charges come from the field response the
    Raman tensors already need, so ``infrared`` costs nothing beyond an assembly.

    Args:
        calculation: the :class:`~defumat.scf.driver.Calculation` the run used.
            A symmetry-reduced k-set is fine -- the rank-3 average that makes it
            so is P36's, and the wedge reproduces the closed grid to 1e-13.
        result: the converged :class:`~defumat.scf.driver.SCFResult`.
        raman: :class:`~defumat.response.nonlinear.RamanTensors` computed
            earlier. Pass one with ``keep_internals=True`` to save the
            dynamical matrix its own solve; without it the phonons are solved
            from scratch and this says so under ``verbose``.
        phonons: :class:`~defumat.response.phonon.Phonons` computed earlier.
        infrared: also assemble the infrared activities, which needs the Born
            effective charges.
        loto_direction: three cartesian components, any length. Given one, the
            long-range electric field of :func:`nonanal` is added before the
            modes are diagonalised, so the frequencies are the **LO** ones for
            a zone centre approached from that direction and the Raman and
            infrared activities are the LO modes'. Needs the Born charges, so
            it needs ``infrared=True``.
        neutralize: impose ``sum_a Z*_a = 0`` before anything is contracted
            (:func:`neutral_born_charges`). Off by default, which is
            ``dynmat.x``'s ``asr = 'no'``; **on is what a physical LO-TO
            splitting needs**, because the violation a finite basis leaves
            charges the cell and lifts a longitudinal acoustic mode.
        response_options: passed to both responses.

    Returns a :class:`VibrationalSpectrum`. Its permittivity is always built
    from the **analytic** modes, whatever ``loto_direction`` says: the sum over
    oscillators that screens a static field runs over the transverse modes, and
    a direction is exactly what removes them. ``dynmat.x`` computes it from
    whichever modes it has and prints "BEWARE" instead.
    """
    # Imported here rather than at the top: this module is what
    # ``defumat.response`` exposes last, and those three import it back through
    # the package. The assembly above needs none of them.
    import jax.numpy as jnp

    from defumat.response.electrostriction import refined_states
    from defumat.response.nonlinear import raman_tensors
    from defumat.response.phonon import dynamical_matrix

    if raman is None:
        raman = raman_tensors(
            calculation, result, born_charges=infrared, keep_internals=True,
            verbose=verbose, **response_options,
        )
    if phonons is None:
        eigenvalues, psi = refined_states(calculation, result)
        if verbose and raman.displacement is None:
            print("  no displacement response to reuse: solving it again")
        phonons = dynamical_matrix(
            calculation, psi, eigenvalues, jnp.asarray(result.density),
            response=raman.displacement, verbose=verbose, **response_options,
        )

    born = getattr(raman.field, "born_charges", None) if infrared else None
    if infrared and born is None:
        raise ValueError(
            "infrared activities need the Born effective charges and the field "
            "response was run without them: pass a raman= computed with "
            "born_charges=True, or infrared=False"
        )
    if loto_direction is not None and born is None:
        raise ValueError(
            "the LO-TO splitting is built from the Born effective charges and "
            "the electronic dielectric tensor: it needs infrared=True"
        )

    masses = calculation.system.structure.masses
    volume = float(calculation.system.cell.volume)
    born = None if born is None else np.asarray(born, dtype=float)
    if neutralize and born is not None:
        violation = np.abs(born.sum(axis=0)).max()
        born = neutral_born_charges(born)
        if verbose:
            print(f"  charge neutrality imposed on Z*: {violation:.3e} shared "
                  f"over {born.shape[0]} atoms")
    frequencies, eigenvectors = phonons.frequencies, phonons.eigenvectors
    if loto_direction is not None:
        frequencies, eigenvectors = loto_modes(
            phonons.matrix, masses, born, raman.epsilon, loto_direction, volume,
        )
        if verbose:
            splitting = float(np.max(frequencies) - np.max(phonons.frequencies))
            print(f"  LO-TO splitting along {tuple(loto_direction)}: "
                  f"{splitting:.2f} cm^-1")

    spectrum = mode_activities(
        frequencies=frequencies,
        eigenvectors=eigenvectors,
        masses=masses,
        epsilon=raman.epsilon,
        volume=volume,
        raman=raman.raman,
        born=born,
        loto_direction=loto_direction,
    )
    if loto_direction is not None and born is not None:
        # The static dielectric constant is a sum over the **transverse**
        # modes, so it is rebuilt from the analytic ones rather than from the
        # LO set just diagonalised (see this function's own docstring).
        analytic = mode_activities(
            frequencies=phonons.frequencies,
            eigenvectors=phonons.eigenvectors,
            masses=masses,
            epsilon=raman.epsilon,
            volume=volume,
            born=born,
        )
        spectrum.ionic_permittivity = analytic.ionic_permittivity
        spectrum.static_permittivity = analytic.static_permittivity
        spectrum.transverse_frequencies = analytic.frequencies
    return spectrum
