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
:func:`~pypresso.io.dynmat.write_dynamical_matrix` writes that file and the
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

**What is not here.** The **non-analytic term** that splits LO from TO
(``rigid.f90``'s ``nonanal``) is omitted: the ``Gamma`` dynamical matrix P25
computes is the analytic one, and the correction needs a direction of approach
``q -> 0`` as well as ``Z*`` and ``eps``, both of which are available. So an
optical triplet comes out unsplit here, and the ``dynmat.x`` comparison is run
with its ``q`` left at zero for the same reason. Adding it is arithmetic of the
same kind as this module's; it is named rather than done.

Ionic (mode-resolved) contributions to the static dielectric constant --
``polar_mode_permittivity`` -- are omitted for the same reason and would be the
same few lines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pypresso.units import AMU_TO_RY, BOHR_TO_ANGSTROM, FPI, RY_TO_CMM1, RY_TO_THZ

__all__ = [
    "VibrationalSpectrum",
    "eigendisplacements",
    "mode_activities",
    "degenerate_manifolds",
    "vibrational_spectrum",
]

#: The elementary charge in Debye/Angstrom -- ``dynmat_sub.f90``'s literal.
#: ``1 e = 4.80324e-10 esu = 4.80324 Debye/A``, and it is written out here
#: rather than derived from :mod:`pypresso.units` because it is what fixes the
#: infrared activities' unit against QE's, to the digit QE prints.
ELECTRON_DEBYE_PER_ANGSTROM = 4.80324

#: The gap in cm^-1 below which two neighbouring frequencies are one multiplet.
#: Loose by four orders and deliberately so: a degeneracy the *symmetry* imposes
#: comes out at **2e-13 cm^-1** on silicon's optical triplet and 2e-11 on its
#: acoustic one, where the nearest thing it has to be told apart from is 500
#: cm^-1 away. Anything between those two scales would do; 1.0 is chosen because
#: it is also below any splitting worth reporting as physics.
DEGENERACY_TOLERANCE = 1.0


@dataclass
class VibrationalSpectrum:
    """The per-mode Raman and infrared activities of a crystal at ``Gamma``.

    Every array has ``3 nat`` entries, ordered by frequency as
    :class:`~pypresso.response.phonon.Phonons` orders them.
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
    :data:`~pypresso.units.AMU_TO_RY` appears here and in
    :func:`~pypresso.response.phonon._diagonalize` and nowhere between them.

    Args:
        eigenvectors: ``(3 nat, 3 nat)``, mode index **last**, as
            :attr:`~pypresso.response.phonon.Phonons.eigenvectors` stores them.
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


def mode_activities(
    frequencies: np.ndarray,
    eigenvectors: np.ndarray,
    masses: np.ndarray,
    epsilon: np.ndarray,
    volume: float,
    raman: np.ndarray | None = None,
    born: np.ndarray | None = None,
    tolerance: float = DEGENERACY_TOLERANCE,
) -> VibrationalSpectrum:
    """``RamanIR``: the whole of the assembly, and nothing that solves anything.

    A transcription of ``LR_Modules/dynmat_sub.f90``'s ``RamanIR``, kept a pure
    function of arrays so that it is testable without a self-consistent run
    behind it -- which is what lets the ``dynmat.x`` comparison be a comparison
    of *this* against the Fortran rather than of two whole calculations.

    Args:
        frequencies: ``(3 nat,)`` in cm^-1, signed
            (:attr:`~pypresso.response.phonon.Phonons.frequencies`).
        eigenvectors: ``(3 nat, 3 nat)``, mode index last.
        masses: ``(nat,)`` in amu.
        epsilon: ``(3, 3)`` electronic dielectric tensor.
        volume: the cell volume in bohr^3.
        raman: ``(nat, 3, 3, 3)`` indexed ``[atom, cart, i, j]`` in **inverse
            bohr** -- :attr:`~pypresso.response.nonlinear.RamanTensors.raman`,
            not the Angstrom-squared form, which this converts to itself.
        born: ``(nat, 3, 3)`` indexed ``[atom, field, displacement]`` -- P24b's
            Born effective charges, in units of ``e``.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    epsilon = np.asarray(epsilon, dtype=float)
    z = eigendisplacements(eigenvectors, masses)

    # The electronic polarizability table ``dynmat.x`` prints, in A^3.
    chi = epsilon - np.eye(3)
    polarizability = chi * BOHR_TO_ANGSTROM**3 * volume / FPI
    clausius_mossotti = 3.0 / (2.0 + np.trace(epsilon) / 3.0)

    mode_dipole = infrared = None
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
    )


def vibrational_spectrum(
    calculation,
    result,
    raman: object | None = None,
    phonons: object | None = None,
    infrared: bool = True,
    verbose: bool = False,
    **response_options,
):
    """The ``Gamma`` spectrum of a converged insulator, tensors and all.

    Three objects go into a spectrum and this solves whichever were not handed
    in: the Raman tensors (P35), the dynamical matrix (P25) and the Born
    effective charges (P24b). **The first two share their expensive half** --
    the displacement response -- so they are solved once here and threaded,
    which is what :func:`~pypresso.response.nonlinear.raman_tensors`'s
    ``keep_internals`` and :func:`~pypresso.response.phonon.dynamical_matrix`'s
    ``response`` exist for. The Born charges come from the field response the
    Raman tensors already need, so ``infrared`` costs nothing beyond an assembly.

    Args:
        calculation: the :class:`~pypresso.scf.driver.Calculation` the run used.
            A symmetry-reduced k-set is fine -- the rank-3 average that makes it
            so is P36's, and the wedge reproduces the closed grid to 1e-13.
        result: the converged :class:`~pypresso.scf.driver.SCFResult`.
        raman: :class:`~pypresso.response.nonlinear.RamanTensors` computed
            earlier. Pass one with ``keep_internals=True`` to save the
            dynamical matrix its own solve; without it the phonons are solved
            from scratch and this says so under ``verbose``.
        phonons: :class:`~pypresso.response.phonon.Phonons` computed earlier.
        infrared: also assemble the infrared activities, which needs the Born
            effective charges.
        response_options: passed to both responses.

    Returns a :class:`VibrationalSpectrum`.
    """
    # Imported here rather than at the top: this module is what
    # ``pypresso.response`` exposes last, and those three import it back through
    # the package. The assembly above needs none of them.
    import jax.numpy as jnp

    from pypresso.response.electrostriction import refined_states
    from pypresso.response.nonlinear import raman_tensors
    from pypresso.response.phonon import dynamical_matrix

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
    return mode_activities(
        frequencies=phonons.frequencies,
        eigenvectors=phonons.eigenvectors,
        masses=calculation.system.structure.masses,
        epsilon=raman.epsilon,
        volume=float(calculation.system.cell.volume),
        raman=raman.raman,
        born=born,
    )
