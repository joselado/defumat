"""The optical conductivity tensor, the Kerr angle and the anomalous Hall effect.

``PLAN.md`` P51, and the fourth entry taken from ``ELK-FEATURES.md``: Elk's
tasks 121 (``dielectric.f90``) and 122 (``moke.f90``). What it computes is the
whole complex tensor

    sigma_ab(omega) = sigma_ab^inter(omega) + sigma_ab^intra(omega),

its *antisymmetric* part -- which is the part that exists only when time
reversal is broken -- and the three things that part is read for: the complex
magneto-optical Kerr angle, the anomalous Hall conductivity, and the
magneto-optical spectrum a Kerr measurement is fitted against.

**Where the physics is.** ``sigma_xx`` is an absorption spectrum and P37
already produces one, so on its own it would be a second implementation. What
is new is the **antisymmetric** part ``sigma_xy``, which is the response that
turns linearly polarized light reflected off a magnet and whose static limit is
the intrinsic anomalous Hall conductivity. It is a property of the *occupied
manifold's geometry* rather than of any single band, which is what connects this
module to :mod:`pypresso.topology.kubo`.

**When it vanishes, stated carefully**, because the loose version of this is
wrong and is the version usually quoted. Time reversal forces it to zero, so a
**nonmagnetic** crystal has none whatever its spin-orbit coupling. A magnetic
crystal *without* spin-orbit coupling has none either -- but only when its
moments are **collinear or coplanar**, where a spin rotation composed with
complex conjugation is an antiunitary symmetry that survives. A **noncoplanar**
magnet has an anomalous Hall effect with no spin-orbit coupling at all: the
scalar spin chirality plays its part, which is the topological Hall effect. So
the pair "magnetism and spin-orbit coupling" is the usual route and not a
theorem, and the surviving antiunitary symmetry forces ``Omega(k) = -Omega(-k)``
rather than ``Omega(k) = 0`` -- which means the cancellation is only exact on a
k-set closed under ``k -> -k``. A shifted grid is not one.

**What the two established codes have.** ``pw.x``'s post-processing
``epsilon.x`` does form the whole tensor -- ``PP/src/epsilon.f90``'s
``offdiag_calc`` -- so this is not a quantity QE lacks outright. It computes no
conductivity and no Kerr angle (it stops at ``epsilon_ab``), it refuses
ultrasoft datasets by name (``grid_build``: "USPP are not implemented"), and it
builds the dipole from **momentum matrix elements**: ``dipole_calc`` accumulates
``<psi_1| G |psi_2>`` and nothing else. That is the one line ``CLAUDE.md``
already records as not to be transcribed, because ``[H, r] != p`` when the
pseudopotential is nonlocal. Here it is ``dH/dk`` from
:class:`~pypresso.response.velocity.VelocityOperator` -- one ``jvp`` of
``H(k)`` at a frozen sphere -- exactly as in P37's head, and it is the only
load-bearing autodiff in the module. Everything else is a transcription of
``dielectric.f90`` and says so.

The expression
--------------

``dielectric.f90`` writes it as (Physica Scripta **T109**, 170 (2004))

    sigma_ij(w) = (i/Omega) sum_k sum_{n,m} t_nm
                  [ z_nm / (w - e_mn + i eta)
                  + conj(z_nm) / (w + e_mn + i eta) ],

    z_nm = <n|v_i|m> <m|v_j|n>,   t_nm = W_n (1 - f_m) / e_mn,

with ``e_mn = e_m - e_n``, ``W_n`` the k-weighted occupation (QE's ``wg``,
summing to ``nelec``) and ``f_m = W_m / w_k`` the fractional filling in
``[0, 1]``. Both orderings of every pair are summed, so an occupied-empty pair
contributes its resonant and antiresonant terms and an empty-occupied one
contributes nothing.

**The units cancel, and it is worth writing down because it looks as though
they cannot.** Elk is a Hartree-unit code and this one is Rydberg, so three
conversions enter at once: an energy doubles, ``dH/dk`` doubles with it, and a
momentum matrix element in Hartree units is ``<n|dH/dk|m> / 2``. The
expression carries ``1/e_mn`` and ``1/(w - e_mn)`` -- two factors of two --
against the squared matrix element's ``1/4``, and the product is exactly one.
So the formula above evaluated with **Rydberg** energies and **Ry bohr**
velocity matrix elements returns ``sigma`` in **Hartree atomic units**,
``e^2/(hbar a_0)``, with no conversion factor anywhere in the sum. That is a
coincidence of this expression's homogeneity and not a general rule; P50's
factor of two is the warning about assuming it elsewhere.

**The intraband term is a separate object** and Elk keeps it separate too. It
is a Drude conductivity built from the plasma frequency,

    wp_ab^2 = (4 pi / Omega) sum_k w_k sum_n v_a^nn v_b^nn delta(e_n - E_F),
    sigma_ab^intra(w) = wp_ab^2 / (4 pi (gamma - i w)),

with ``delta`` the same smeared delta the SCF's occupations came from
(``w0gauss``, QE's ``Modules/w0gauss.f90``) -- so a run with ``occupations =
'fixed'`` has no intraband term at all, which is the statement that an insulator
does not conduct at zero frequency. ``gamma`` is a phenomenological relaxation
rate and is a knob, not a computed quantity; Elk's ``dielectric.f90`` silently
reuses ``swidth`` for it and this module makes it an argument that defaults the
same way.

Three routes to the static limit, and why there are three
---------------------------------------------------------

The anomalous Hall conductivity is ``sigma_xy(w -> 0)``, and taking that limit
in the expression above analytically -- ``eta -> 0`` and ``w -> 0`` -- collapses
the two resolvents onto a single ``1/e_mn^2``:

    sigma_xy = (2/Omega) sum_k sum_{n,m} W_n (1 - f_m) Im(z_nm) / e_mn^2
             = -(1/Omega) sum_k sum_n W_n Omega_n^{xy}(k),

which is the Brillouin-zone integral of the **Berry curvature** over the
occupied manifold -- the quantity :mod:`pypresso.topology.kubo` computes by an
independently written assembly that is anchored against the Fukui-Hatsugai-Suzuki
lattice flux (P47). So ``method="curvature"`` is that second route, and it is
what the ``w = 0`` value of the frequency sum is checked against. It is the
arrangement P50 used, and P50 is why it is worth the trouble: two routes are
what caught a factor of two that every symmetry check passed.

**The two routes are one limit taken in two orders, and for a metal the orders
do not commute.** Collapsing the resolvents took ``eta -> 0``
*before* ``w -> 0``; the frequency sum at ``w = 0`` takes them the other way
round, keeping ``eta`` finite. Where every gap is large compared with ``eta``
the two limits commute and the numbers coincide. At a **Fermi surface** they do
not: a metal has occupied-empty pairs with arbitrarily small ``e_mn``, the
curvature route weights them ``1/e_mn^2`` and the frequency route regularises
them at ``eta``, and the difference is not a correction. It is the reason an
intrinsic anomalous Hall conductivity is famously mesh-hungry -- the integrand
is concentrated on near-degeneracies -- and the honest thing is to quote the
curvature route, which is the quantity's definition, together with its
k-convergence. The frequency route's ``w = 0`` value is what a spectrum
extrapolates to at finite scattering, and is a different number.

**What is refused, and each for its own reason.** An **ultrasoft or PAW**
dataset, for :mod:`pypresso.topology.kubo`'s reason: with a moving ``S`` the
current operator acquires ``e_n dS/dk`` off the diagonal, the term is
identically zero for a norm-conserving dataset, and nothing validated here can
see whether its convention is right. A **spin spiral**, whose two spinor
components live on different spheres. And a **symmetry-reduced k-set**: the
antisymmetric part of ``sigma`` is an axial vector and a wedge does not sum to
the cell's, exactly as P48b's angular momenta are refused on one -- the escape
is the whole unshifted grid, which is closed under the point group.

**Cost and peak.** One NSCF with empty states, then ``(3, nk, nbnd, nbnd)``
velocity matrix elements -- three ``jvp`` calls over the k axis, which is the
whole expense -- and a frequency sum whose working set is ``nw x nbnd^2``
complex per k-point, accumulated through :func:`~pypresso.batching.sum_k`. For
sixty-four k-points, forty bands and five hundred frequencies that is 13 MB per
chunk against 5 MB of matrix elements: the frequency axis is free and the band
count is what has to be watched, since it enters squared and is also the
truncation the f-sum rule measures.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from pypresso.batching import resolve_k_batch, sum_k
from pypresso.response.velocity import VelocityOperator
from pypresso.scf.occupations import smearing_order, w0gauss
from pypresso.units import AU_TO_S_PER_CM, FPI, RY_TO_EV

__all__ = [
    "OpticalConductivity",
    "optical_conductivity",
    "require_a_conductivity_regime",
]


@dataclass(frozen=True)
class OpticalConductivity:
    """What :func:`optical_conductivity` returns.

    Attributes:
        frequencies: ``(nw,)`` in Ry -- the photon energies, ``hbar omega``.
        sigma: ``(nw, 3, 3)`` complex, in Hartree atomic units
            ``e^2/(hbar a_0)``. The interband and intraband terms summed.
        interband: ``(nw, 3, 3)``, the Kubo-Greenwood part alone.
        intraband: ``(nw, 3, 3)``, the Drude part alone -- identically zero
            for a run with fixed occupations.
        plasma: ``(3, 3)`` in Ry, ``hbar wp_ab`` -- the square root of the
            tensor above, signed by nothing since it is a square root of an
            absolute value, exactly as ``dielectric.f90`` takes it.
        volume: the cell volume in bohr^3, which every prefactor divides by.
        broadening: ``eta`` in Ry.
        relaxation: ``gamma`` in Ry, the Drude rate.
        nbnd: how many bands the sum ran over. It is the convergence parameter
            and there is no refusal that catches it being too small, which is
            what :attr:`fsum` is for.
        nelec: the electron count the f-sum rule is measured against.
        band_cut_gap: the smallest ``eps_(nbnd+1) - eps_nbnd`` over the k-set,
            in Ry -- how far the band sum's last state is from the first one it
            did **not** include. It cannot be computed from the band set the
            sum runs over, since that set stops one state short of it, so
            :func:`~pypresso.workflows.conductivity.run_conductivity`
            diagonalises one extra band to measure it and the assembly takes it
            as an argument; it is ``nan`` when nobody supplied one. **Read it
            before believing a quantity symmetry says should vanish.**
            Truncating *inside* a degenerate multiplet keeps some of its
            members and drops others, and the cancellation those members were
            making between them does not happen. Measured on silicon, whose
            antisymmetric ``sigma`` is zero by time reversal, the residue
            against the gap at the cut is 4.0e-13 at ``nbnd = 20`` (gap
            **2.83e-2** Ry, the one band count here that stops at a real gap),
            and 1.0e-5, 2.3e-6 and 8.3e-7 at ``nbnd = 12``, 24 and 32, whose
            gaps are 1.2e-13, 3.2e-13 and 8.9e-13. Six orders of magnitude. It
            is **necessary and not sufficient**: ``nbnd = 36`` cuts a
            degeneracy (6.0e-13) and escapes with 4.0e-13 anyway, because its
            two sides happen to contribute equally. A large gap here is a
            guarantee; a small one is a warning.
    """

    frequencies: np.ndarray
    sigma: np.ndarray
    interband: np.ndarray
    intraband: np.ndarray
    plasma: np.ndarray
    volume: float
    broadening: float
    relaxation: float
    nbnd: int
    nelec: float
    band_cut_gap: float = float("nan")

    # -- what a reader wants it in ----------------------------------------

    @property
    def frequencies_ev(self) -> np.ndarray:
        """The photon energy axis in eV."""
        return np.asarray(self.frequencies) * RY_TO_EV

    @property
    def sigma_s_per_cm(self) -> np.ndarray:
        """``sigma`` in S/cm, the unit an anomalous Hall conductivity is quoted in."""
        return np.asarray(self.sigma) * AU_TO_S_PER_CM

    @property
    def plasma_ev(self) -> np.ndarray:
        """``hbar wp_ab`` in eV. Aluminium's ``xx`` entry is about 15.8."""
        return np.asarray(self.plasma) * RY_TO_EV

    @property
    def dielectric(self) -> np.ndarray:
        """``eps_ab(w) = delta_ab + 4 pi i sigma_ab / w``, ``(nw, 3, 3)``.

        The independent-particle dielectric tensor **without local fields** --
        which is exactly what P37's ``run_absorption`` produces in RPA when the
        response sphere is the head alone, and the agreement between them is
        this module's check on its own prefactor.

        ``w`` carries the broadening, so ``w = 0`` is finite rather than a
        division by zero; what it is not is the static dielectric constant of a
        metal, which diverges and should.
        """
        omega = _hartree(np.asarray(self.frequencies)) + 1j * _hartree(self.broadening)
        eye = np.eye(3)[None]
        return eye + FPI * 1j * np.asarray(self.sigma) / omega[:, None, None]

    @property
    def hall_conductivity(self) -> np.ndarray:
        """The static antisymmetric part ``sigma_ab(w -> 0)`` in S/cm, ``(3, 3)``.

        Read as an axial vector, ``sigma_yz``, ``sigma_zx``, ``sigma_xy`` are
        its ``x``, ``y``, ``z`` components. The antisymmetric part is taken so
        that the (much larger) symmetric part cannot leak into it through
        round-off.

        **Which of the two static numbers this is depends on ``method``, and
        for a metal they are different quantities.** The *intrinsic* anomalous
        Hall conductivity is the ``eta -> 0`` limit taken first, which is
        ``method = "curvature"``; what ``method = "frequency"`` returns here is
        the conductivity at zero frequency and *finite* scattering ``eta``,
        which is the number a measured spectrum extrapolates to. For an
        insulator the two coincide, because every gap is large compared with
        ``eta``.

        It is the frequency sum's **first point**, so a caller who supplied a
        grid that does not start at zero is asking for a different quantity;
        that is refused rather than returned, because the number looks the same
        either way.
        """
        if abs(float(np.asarray(self.frequencies)[0])) > 1e-12:
            raise ValueError(
                "the Hall conductivity is the static limit and this "
                f"conductivity's frequency grid starts at "
                f"{float(np.asarray(self.frequencies)[0]):g} Ry, not at zero. "
                "Ask for a grid that includes omega = 0 (the default does), or "
                "read the antisymmetric part of `sigma` at the frequency you "
                "want"
            )
        sigma = np.asarray(self.sigma)[0]
        return 0.5 * (sigma - sigma.T) * AU_TO_S_PER_CM

    @property
    def kerr(self) -> np.ndarray:
        """The complex Kerr angle ``theta_K + i eta_K`` in **degrees**, ``(nw,)``.

        ``moke.f90``'s expression for a polar Kerr geometry with the
        magnetization along ``z``,

            theta_K + i eta_K = -sigma_xy / (sigma_xx sqrt(1 + 4 pi i sigma_xx / w)),

        which is pure post-processing -- all of the work is the tensor. It is
        zero at ``w = 0`` by construction (``moke.f90`` sets it so), because
        the expression's denominator is not defined there.
        """
        sigma = np.asarray(self.sigma)
        omega = _hartree(np.asarray(self.frequencies))
        sxx, sxy = sigma[:, 0, 0], sigma[:, 0, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            root = sxx * np.sqrt(1.0 + FPI * 1j * sxx / omega)
            angle = np.where(np.abs(root) > 1.0e-8, -sxy / np.where(
                np.abs(root) > 1.0e-8, root, 1.0), 0.0)
        return np.where(omega > 0.0, angle, 0.0) * 180.0 / np.pi

    @property
    def fsum(self) -> float:
        """``(2/pi) int Re sigma_xx dw / n_e`` -- the spectral weight, and it is not 1.

        The Thomas-Reiche-Kuhn sum rule in its conductivity form is usually
        quoted as ``int_0^inf Re sigma_aa dw = pi n_e / 2`` with
        ``n_e = nelec/Omega``, and this property is that integral divided by
        that. **It does not come out at one and it should not**, for two
        reasons that are worth keeping apart because only one of them is an
        error.

        *The exact statement* for a velocity ``dH/dk`` is

            int_0^inf Re sigma_aa dw
                = (pi/2 Omega) sum_k sum_n W_n [<n|d2H/dk_a^2|n> - d2eps_n/dk_a^2],

        which follows from the ``k.p`` identity
        ``d2eps_n/dk^2 = <n|d2H/dk^2|n> + 2 sum_m |v_nm|^2/(eps_n - eps_m)``
        once the occupied-occupied pairs cancel. Two things then separate it
        from ``pi n_e/2``:

        - ``<n|d2H/dk^2|n>`` is ``1/m = 1`` only for a **local** Hamiltonian.
          A nonlocal pseudopotential has a second ``k``-derivative of its own,
          and on ``si2-nosym.in`` it is measurably not one: **0.9432** in
          Hartree units, averaged over the occupied states.
        - ``sum_k w_k d2eps_n/dk^2`` integrates to **zero** over the zone by
          periodicity, but only in the limit. On a coarse grid it does not, and
          that is a k-convergence error nothing else in this package sees.

        Both were measured, and the second is much the larger on a small grid:
        on silicon at ``nbnd = 40`` this ratio is **1.185** on a 4x4x4 grid,
        **0.985** on 6x6x6 and **0.934** on 8x8x8, against a ``<d2H/dk^2>``
        that moves only from 0.9423 to 0.9432 over the same three. So it
        converges onto the diamagnetic weight to **1 per cent**, and that
        convergence -- not the value one -- is what makes this an absolute
        check on the assembly's prefactor: the volume, the electron count, the
        spin degeneracy and the Rydberg-to-Hartree cancellation all enter it and
        none of them is fitted.

        Two things make it read low and are the caller's to fix: the band sum
        stops where the eigensolver did, and the integral stops at the last
        frequency asked for. Ask for a window well past the largest
        occupied-empty gap before reading this at all.
        """
        omega = _hartree(np.asarray(self.frequencies))
        real = np.real(np.asarray(self.sigma)[:, 0, 0])
        integral = float(np.trapezoid(real, omega))
        density = self.nelec / self.volume
        return integral / (np.pi * density / 2.0)


def _hartree(x):
    """Ry to Hartree. ``sigma`` is in Hartree atomic units, so ``w`` must be too."""
    return np.asarray(x) / 2.0


# -- what it refuses -----------------------------------------------------------


def require_a_conductivity_regime(calculation) -> None:
    """Three refusals, each with the missing term named.

    They are the same three :mod:`pypresso.topology.kubo` makes, for the same
    reasons and in the same order, because this module and that one are the
    same velocity matrix elements contracted differently.
    """
    if calculation.is_ultrasoft or calculation.is_paw:
        raise NotImplementedError(
            "the optical conductivity with an ultrasoft or PAW "
            "pseudopotential is not implemented: the current operator of a "
            "generalised eigenproblem carries <psi_n|dS/dk_a|psi_m> beside "
            "dH/dk, and that term is identically zero for a norm-conserving "
            "dataset -- so nothing validated here can see whether its "
            "convention is right. It is the refusal pypresso.topology.kubo "
            "makes for the same matrix element. Use a norm-conserving dataset"
        )
    if getattr(calculation, "spiral", False):
        raise NotImplementedError(
            "the optical conductivity of a spin spiral is not implemented: "
            "the two spinor components live on spheres centred at k + q/2 and "
            "k - q/2, so <n|dH/dk|m> is not a single contraction over one "
            "plane-wave set"
        )
    if _kpoints_are_reduced(calculation):
        raise NotImplementedError(
            "the optical conductivity needs the whole k-grid, not a "
            "symmetry-reduced wedge: the antisymmetric part of sigma_ab is an "
            "axial vector -- it is the Berry curvature integral in the static "
            "limit -- and a wedge does not sum to the cell's, which is the "
            "refusal P48b's angular momenta make. Run with nosym = .true. and "
            "noinv = .true. on an *unshifted* grid, which is closed under the "
            "point group"
        )


def _kpoints_are_reduced(calculation) -> bool:
    """Whether the k-set is a wedge rather than the whole grid.

    Read off the weights, as :func:`~pypresso.tddft.chi0._kpoints_are_reduced`
    does and for its reason: an unreduced grid gives every k-point the same
    weight and a reduction is exactly what makes them differ, which is the test
    that survives an explicit ``K_POINTS`` list -- how every closed-grid case in
    ``tests/data/qe`` is written.
    """
    weights = np.asarray(calculation.system.kpoints.weights)
    if weights.size <= 1:
        return False
    return bool(np.ptp(weights) > 1.0e-8 * np.abs(weights).max())


# -- the assembly --------------------------------------------------------------


def optical_conductivity(
    calculation,
    wavefunctions,
    eigenvalues,
    v_scf,
    *,
    fermi_energy: float,
    frequencies=None,
    window: float = 1.5,
    nw: int = 300,
    broadening: float = 0.01,
    relaxation: float | None = None,
    intraband: bool = True,
    scissor: float = 0.0,
    method: str = "frequency",
    ddd_paw=None,
    ns=None,
    band_cut_gap: float = float("nan"),
    k_batch="default",
) -> OpticalConductivity:
    """``sigma_ab(omega)`` from a fixed-density run, in Hartree atomic units.

    The pure assembly, taking the states rather than a result object -- the
    same shape :func:`~pypresso.tddft.chi0.independent_response` has and for
    the same reason: the sum runs over empty states, so what it consumes is an
    NSCF's output and not a ground state's.
    :func:`~pypresso.workflows.conductivity.run_conductivity` is the entry
    point that produces one.

    Args:
        calculation: the :class:`~pypresso.scf.driver.Calculation` the states
            belong to, on the **whole** k-grid.
        wavefunctions: ``(nspin, nk, nbnd, ndim)``. ``ndim`` is ``2 npwx`` for
            a spinor, which is the regime a magneto-optical spectrum wants.
        eigenvalues: ``(nspin, nk, nbnd)`` or the squeezed ``(nk, nbnd)``, Ry.
        v_scf: the converged potential ``dH/dk`` is built at.
        fermi_energy: in Ry, for the Drude term's delta function.
        frequencies: ``(nw,)`` in Ry. Defaults to a uniform grid from zero to
            ``window``, which is what makes ``sigma[0]`` the static limit.
        window: the top of that default grid, in Ry.
        nw: how many points it has.
        broadening: ``eta`` in Ry, the Lorentzian half-width every transition
            is given. Elk's ``swidth``.
        relaxation: ``gamma`` in Ry for the Drude term; defaults to
            ``broadening``, which is ``dielectric.f90``'s own choice.
        intraband: whether to add the Drude term. It is zero anyway for a run
            with fixed occupations, which has no Fermi surface to build it on.
        scissor: a rigid shift (Ry) of the empty states, applied to the
            eigenvalues **and** to the matrix elements they renormalise --
            P37's convention, and one approximation rather than two.
        method: ``"frequency"`` for the resolvent sum above, or
            ``"curvature"`` for the analytic ``w -> 0`` limit alone, which is
            the Berry-curvature integral and returns a tensor at ``w = 0``
            only. The second exists to check the first.
        ddd_paw: PAW's one-centre coefficients, for the Hamiltonian ``dH/dk``
            differentiates. Refused upstream, and carried for the day it is not.
        ns: the Hubbard occupations, likewise.
        band_cut_gap: the gap between the last band kept and the first dropped,
            in Ry, which the caller measures because this function's band set
            stops one state short of it. See
            :attr:`OpticalConductivity.band_cut_gap`.
        k_batch: the batching dial.

    Returns:
        An :class:`OpticalConductivity`. Nothing is symmetrised: the refusal
        above means there is nothing to put back.
    """
    require_a_conductivity_regime(calculation)
    if method not in ("frequency", "curvature"):
        raise ValueError(
            f"method must be 'frequency' or 'curvature', not {method!r}"
        )

    eigenvalues = jnp.asarray(eigenvalues)
    if eigenvalues.ndim == 2:
        eigenvalues = eigenvalues[None]
    wavefunctions = jnp.asarray(wavefunctions)
    if wavefunctions.ndim == 3:
        wavefunctions = wavefunctions[None]
    if eigenvalues.shape[0] != 1:
        raise NotImplementedError(
            "the optical conductivity of a collinear spin-polarized run is "
            "not implemented: the two channels are two independent band "
            "structures whose conductivities add, which is a loop this "
            "assembly does not have. A spinor run (noncolin = .true.) is the "
            "regime a magneto-optical spectrum wants anyway, since sigma_xy "
            "needs spin-orbit coupling and nspin = 2 has none"
        )

    nbnd = int(eigenvalues.shape[-1])
    precision = calculation.system.cell.precision
    volume = float(calculation.system.cell.volume)

    # The scissors shift moves the empty states, and the matrix elements have
    # to move with them (``getpmat.f90``). With a metal there is no clean
    # valence/conduction split, so it is applied above the Fermi level.
    shifted = eigenvalues
    if scissor:
        above = eigenvalues > precision.as_real(fermi_energy)
        shifted = jnp.where(above, eigenvalues + precision.as_real(scissor),
                            eigenvalues)

    velocity = VelocityOperator(calculation, v_scf, ddd_paw, ns)
    elements = velocity.matrix_elements(wavefunctions)  # (3, nspin, nk, nb, nb)
    elements = jnp.moveaxis(elements[:, 0], 0, 1)  # (nk, 3, nbnd, nbnd)

    if scissor:
        elements = _renormalise(elements, eigenvalues[0], shifted[0])

    wg, _ = calculation.occupations(shifted)
    wg = jnp.asarray(wg)[0]  # (nk, nbnd), summing to nelec
    wk = jnp.asarray(calculation.system.kpoints.weights)  # (nk,)
    filling = wg / wk[:, None]  # in [0, 1]

    if frequencies is None:
        frequencies = np.linspace(0.0, float(window), int(nw))
    frequencies = np.asarray(frequencies, dtype=float)
    if method == "curvature":
        frequencies = np.zeros(1)

    batch = resolve_k_batch(k_batch)
    eta = precision.as_real(broadening)
    zomega = (jnp.asarray(frequencies) + 1j * eta).astype(precision.complex)

    if method == "frequency":
        def one_k(arrays):
            element, energy, weight, fill = arrays
            return _resolvent_sum(element, energy, weight, fill, zomega)
    else:
        def one_k(arrays):
            element, energy, weight, fill = arrays
            return _curvature_sum(element, energy, weight, fill)

    inter = sum_k(one_k, (elements, shifted[0], wg, filling), batch=batch)
    inter = np.asarray(inter) * (1.0 / volume)

    plasma, intra = _drude(
        calculation, fermi_energy, elements, shifted[0], wk, frequencies,
        volume, broadening if relaxation is None else relaxation,
        enabled=intraband and method == "frequency",
    )

    return OpticalConductivity(
        frequencies=frequencies,
        sigma=inter + intra,
        interband=inter,
        intraband=intra,
        plasma=plasma,
        volume=volume,
        broadening=float(broadening),
        relaxation=float(broadening if relaxation is None else relaxation),
        nbnd=nbnd,
        nelec=float(calculation.nelec),
        band_cut_gap=float(band_cut_gap),
    )


def _renormalise(elements, energies, shifted):
    """``getpmat.f90``'s rescaling of a matrix element under a scissors shift.

    The velocity matrix element between two states is ``<n|dH/dk|m>`` and the
    quantity the sum actually wants is ``<n|r|m> ~ <n|dH/dk|m>/(e_m - e_n)``,
    so moving the eigenvalues without moving the numerator silently changes the
    dipole. The ratio of the unshifted gap to the shifted one restores it.
    """
    gap = energies[:, :, None] - energies[:, None, :]
    moved = shifted[:, :, None] - shifted[:, None, :]
    safe = jnp.where(jnp.abs(moved) > 1.0e-12, moved, 1.0)
    ratio = jnp.where(jnp.abs(moved) > 1.0e-12, gap / safe, 1.0)
    return elements * ratio[:, None, :, :]


#: Below this gap (Ry) a pair of states is dropped from the interband sum.
#: ``dielectric.f90`` uses 1e-8 and so does this; the pair is a genuine
#: singularity of ``1/e_mn`` and its physical content -- an intraband
#: transition -- is the Drude term instead.
DEGENERACY_TOL = 1.0e-8


def _pair_weights(energies, wg, filling):
    """``t_nm = W_n (1 - f_m) / e_mn`` and the gap, with the degenerate pairs cut.

    ``W_n`` is the k-weighted occupation (QE's ``wg``) and ``f_m`` the
    fractional filling in ``[0, 1]``; ``dielectric.f90``'s ``t1``, whose
    asymmetry is deliberate -- the *bra* carries the whole weight and the ket
    only the Pauli blocking factor.
    """
    gap = energies[:, None] - energies[None, :]  # e_n - e_m
    gap = -gap  # e_mn = e_m - e_n
    finite = jnp.abs(gap) > DEGENERACY_TOL
    safe = jnp.where(finite, gap, 1.0)
    t = jnp.where(finite, wg[:, None] * (1.0 - filling[None, :]) / safe, 0.0)
    return t, gap


def _resolvent_sum(element, energies, wg, filling, zomega):
    """One k-point's interband ``sigma_ij(w)``, ``(nw, 3, 3)``.

    ``element`` is ``(3, nbnd, nbnd)``. The pair axis is flattened so the
    frequency dependence is one matrix product: ``(nw, pairs)`` against
    ``(pairs, 3, 3)``, which is why the frequency grid is nearly free.
    """
    t, gap = _pair_weights(energies, wg, filling)
    # z_nm = <n|v_i|m> <m|v_j|n>, the outer product in the cartesian labels.
    z = jnp.einsum("inm,jmn->nmij", element, element)
    nb = energies.shape[0]
    z = z.reshape(nb * nb, 3, 3)
    t = t.reshape(-1)
    gap = gap.reshape(-1)

    resonant = t[None, :] / (zomega[:, None] - gap[None, :])
    antires = t[None, :] / (zomega[:, None] + gap[None, :])
    return 1j * (jnp.einsum("wp,pij->wij", resonant, z)
                 + jnp.einsum("wp,pij->wij", antires, jnp.conj(z)))


def _curvature_sum(element, energies, wg, filling):
    """The analytic ``w -> 0, eta -> 0`` limit, ``(1, 3, 3)``.

    Both resolvents collapse onto ``1/e_mn`` with opposite signs, leaving
    ``(z - conj(z))/e_mn = 2i Im(z)/e_mn``, so the whole static tensor is

        sigma_ij = (2/Omega) sum W_n (1 - f_m) Im(z_nm) / e_mn^2,

    which is antisymmetric in ``ij`` by construction -- the symmetric part of a
    static conductivity is zero for an insulator and is the Drude term for a
    metal. It is the occupied manifold's Berry curvature integral, and this is
    the route :mod:`pypresso.topology.kubo` reaches by a different assembly.
    """
    t, gap = _pair_weights(energies, wg, filling)
    z = jnp.einsum("inm,jmn->nmij", element, element)
    finite = jnp.abs(gap) > DEGENERACY_TOL
    safe = jnp.where(finite, gap, 1.0)
    weight = jnp.where(finite, t / safe, 0.0)
    return (2.0 * jnp.einsum("nm,nmij->ij", weight, jnp.imag(z)))[None]


def _drude(calculation, fermi_energy, elements, energies, wk, frequencies,
           volume, relaxation, *, enabled: bool):
    """The plasma frequency tensor and the Drude conductivity it generates.

    ``dielectric.f90``'s intraband branch. The delta function is the smearing
    the SCF itself used, so a fixed-occupation run has no Fermi surface here
    and the whole term is zero -- which is the statement that an insulator has
    no free carriers, not an approximation.
    """
    nw = len(frequencies)
    zero = np.zeros((nw, 3, 3), dtype=complex)
    scheme = str(getattr(calculation.system, "occupations", "fixed")).lower()
    degauss = float(getattr(calculation.system, "degauss", 0.0) or 0.0)
    if enabled and scheme.startswith("tetrahedra"):
        # **Refused rather than dropped**, because dropping it is invisible.
        # The delta function here is the smearing's ``w0gauss``, and a
        # tetrahedron run has none -- it integrates the true step function,
        # which is why it has no ``-TS`` term either. Falling through would
        # return an insulator's conductivity for a metal, with a plasma
        # frequency of exactly zero and nothing saying why.
        raise NotImplementedError(
            "the intraband (Drude) term of a tetrahedron run is not "
            "implemented: its weight is a Fermi-surface delta function and "
            "the tetrahedron method has no smeared delta to supply one "
            "(tetra.f90 integrates the step function itself). Run with "
            "occupations = 'smearing', or pass intraband = False and read the "
            "interband part alone"
        )
    if not enabled or degauss <= 0.0 or "smearing" not in scheme:
        return np.zeros((3, 3)), zero

    ngauss = smearing_order(getattr(calculation.system, "smearing", "gaussian"))
    x = (energies - float(fermi_energy)) / degauss
    delta = w0gauss(x, ngauss) / degauss  # (nk, nbnd), 1/Ry

    # **The multiplet block, not the diagonal.** ``dielectric.f90`` writes the
    # Drude weight as ``sum_n v_a^nn v_b^nn``, which is what the semiclassical
    # picture asks for and what is **not invariant** under the unitary rotation
    # a degenerate eigensolver is free in (design rule D4): inside a multiplet
    # the diagonal of an operator moves with the basis and only the block's
    # trace does not, and a Fermi surface is where a metal keeps its
    # degeneracies. Summing ``v_a^nm v_b^mn`` over the pairs the interband term
    # *excludes* as degenerate is invariant, reduces to the diagonal wherever
    # nothing is degenerate, and makes the two halves of the tensor
    # complementary rather than overlapping.
    #
    # **It is a rule, not a repair, and the measurement says so**: on fcc
    # nickel with spin-orbit coupling the two forms give 0.5971 eV each,
    # because an exact degeneracy has to sit within 1e-8 Ry of the Fermi level
    # to be caught at all and a 4x4x4 mesh puts none there. The invariant form is what is
    # written because the diagonal one has no reason to keep agreeing on a
    # denser mesh or a more symmetric metal, and the failure would be silent.
    gap = energies[:, :, None] - energies[:, None, :]
    multiplet = (jnp.abs(gap) < DEGENERACY_TOL).astype(delta.dtype)
    plasma2 = FPI / volume * jnp.real(jnp.einsum(
        "k,kn,knm,kanm,kbmn->ab", wk, delta, multiplet, elements, elements
    ))
    # In Hartree units: the matrix element halves twice and the delta doubles,
    # so ``4 pi/Omega sum w v_a v_b delta`` in Rydberg quantities is
    # ``wp^2`` in Hartree^2 up to one factor of two.
    plasma2 = np.asarray(plasma2) / 2.0
    plasma = np.sqrt(np.abs(plasma2)) * 2.0  # back to Ry, which is what is reported

    gamma = _hartree(relaxation)
    omega = _hartree(np.asarray(frequencies))
    drude = plasma2[None] / (FPI * (gamma - 1j * omega)[:, None, None])
    return plasma, drude
