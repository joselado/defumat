"""The shift current: the bulk photovoltaic effect of a non-centrosymmetric crystal.

Shine light on a crystal with no inversion centre and it carries a direct
current, with no junction and no built-in field. The intrinsic part of that --
the *shift current* -- is a second-order optical response,

    J^a = 2 sigma^abc(0; w, -w) E_b(w) E_c(-w),

and what makes it a quantum-geometric quantity rather than an ordinary
absorption is that the photoexcited electron is born displaced: the real-space
shift between the valence and conduction Wannier centres is what the current
counts. It is written here following Sipe and Shkrebtii (PRB **61**, 5337
(2000)) in the form Ibanez-Azpiroz, Tsirkin and Souza give it (PRB **97**,
245143 (2018), arXiv:1804.04030 -- "IATS18" below, the reference Wannier90's
``berry_get_sc_klist`` cites), their Eq. (8):

    sigma^abc(w) = -i pi e^3 / (4 hbar^2) int [dk] sum_nm f_nm
                   (I^abc_mn + I^acb_mn) [delta(w_mn - w) + delta(w_nm - w)],

    I^abc_mn = r^b_mn r^{c;a}_nm.

**The whole difficulty is one object**, the *generalised derivative*
``r^{c;a}_nm`` -- the derivative of the interband dipole with respect to ``k``,
made covariant so that it does not depend on the arbitrary phase the
eigensolver hands each band:

    r^{c;a}_nm = d_a r^c_nm - i (A^a_nn - A^a_mm) r^c_nm.

Neither piece of that is separately computable here. ``A^a_nn``, the diagonal
Berry connection, is gauge dependent and not a function of ``H`` at one ``k``
at all, and ``d_a r^c_nm`` needs the phases of two different k-points related.
The escape is the **sum rule** -- IATS18's Eq. (32), which is Aversa and Sipe's
(PRB **52**, 14636 (1995)) -- which expresses the whole covariant object in
quantities that live at a single ``k`` and carry no gauge freedom at all:

    r^{c;a}_nm = (i/w_nm) [ (v^c_nm D^a_nm + v^a_nm D^c_nm)/w_nm - w^{ca}_nm
                            + sum_{p != n,m} ( v^c_np v^a_pm / w_pm
                                             - v^a_np v^c_pm / w_np ) ],

with ``w_nm = e_n - e_m``, ``D^a_nm = v^a_nn - v^a_mm`` the band-velocity
difference, and

    v^a_nm = <n| dH/dk_a |m>,     w^{ab}_nm = <n| d^2H/dk_a dk_b |m>.

**Every one of those is a derivative of code that already exists.** ``v`` is
:meth:`~defumat.response.velocity.VelocityOperator.matrix_elements`, one
``jvp`` of ``H(k)`` at a frozen sphere (P24), and ``w`` is
:meth:`~defumat.response.velocity.VelocityOperator.second_matrix_elements`,
that ``jvp`` differentiated by a second one. Nothing is transcribed and no
radial form factor is a table lookup, which is rule D2 being cashed in one
quantity further along than P51 took it.

Two things about that are worth stating rather than leaving to be noticed.

**The momentum matrix element is the wrong operator and this code never forms
one.** ``[H, r] = p`` only for a *local* Hamiltonian; with a nonlocal
pseudopotential ``v = dH/dk`` and ``p`` differ by the projector term, which is
why ``PP/src/epsilon.f90`` and Elk's ``getpmat`` are the one line ``CLAUDE.md``
records as not to be transcribed. The same applies one order up: ``w^{ab}``
here is the true second derivative of ``H``, where the textbook derivations
(Hughes and Sipe, and Elk's ``nonlinopt.f90`` after them) substitute the
free-electron effective-mass sum rule ``w^{ab}_nm -> delta_ab delta_nm / m``
plus a triple sum over states. That substitution is exact in an all-electron
basis and wrong here, and the difference between the two is a *measurement* of
the nonlocal term rather than a check of anything.

**The intermediate sum truncates, the truncation is reported, and it is the
largest approximation in this module.** ``p`` runs over the bands the
eigensolver was asked for and stops -- IATS18 avoid that with ``k.p`` theory
inside a closed Wannier subspace, and there is no closed subspace here.
:attr:`ShiftCurrent.truncation` is the shift in the spectrum when the highest
empty band is dropped, in the manner of P37's ``static_residual`` and P47's
``BerryCurvature.truncation``: a diagnostic to read, never a knob to tune until
a test passes.

**How large it is, measured rather than assumed.** On AlAs at a generic
k-point, held on a frozen sphere of 158 plane waves so that the band set can be
run all the way to completeness, the valence-to-conduction block of
``r^{c;a}`` agrees with a parallel-transport finite difference to **6e-2 at 20
bands, 4.8e-2 at 80, 4.3e-2 at 120 -- and 1.8e-4 at 158**, which is the
complete basis. The sum rule is an *identity* once ``p`` runs over the whole
space, and that is what the last number is; the slow approach to it is the tail
of a sum whose terms fall off like ``1/E_p``. So a shift current at a
production band count carries a few per cent from this and no symmetry check
sees it.

**The route that would remove it is identified rather than fitted.** Writing
``|u_m^{;a}>`` for the covariant k-derivative of band ``m``, the two
intermediate sums are exactly

    sum_{p != n,m} (...) = -<u_n| dH/dk_c |u_m^{;a}>
                           - <u_n^{;a}| dH/dk_c |u_m>
                           - g^a_nm D^c_nm,

which contains no sum over states at all -- the same collapse P24 makes
everywhere else, a Sternheimer solve in place of a spectral sum. What stops it
being written today is that ``|u_m^{;a}>`` is a *per-band* derivative: the
operator to invert is ``H - e_m`` on the complement of one band, which is
**indefinite** whenever any state lies below ``e_m``, and the projected CG of
:mod:`defumat.response.sternheimer` needs a positive definite one. It is P48's
objection to obtaining an individual band's ``|dpsi/dk>`` that way, and lifting
it needs an indefinite solver, which is a machine this package does not have.

**The exclusion of ``p = n, m`` collapses into a commutator**, which is the one
piece of algebra here that is not in either reference. Writing ``g^a = v^a /
w_nm`` elementwise -- zero on the diagonal, and zero wherever the pair is
degenerate -- the two sums with their exclusions are exactly

    sum_{p != n,m} (...) = [v^c, g^a]_nm - g^a_nm D^c_nm,

because ``1/w_pm`` already kills ``p = m`` in the first sum and ``1/w_np``
kills ``p = n`` in the second, leaving one excluded term each which is
precisely what the ``D^c`` piece subtracts. So the whole intermediate sum is
two matrix products, and the assembly is ``O(nbnd^3)`` per k-point rather than
a Python loop over triples.

Scope
-----

Norm-conserving, ``nspin = 1`` or a spinor run, an **insulator** with fixed
occupations, on a k-grid closed under the point group. Refused by name, each
for its own missing term, in :func:`require_a_shift_current_regime`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from defumat.batching import resolve_k_batch, sum_k
from defumat.response.velocity import VelocityOperator
from defumat.scf.occupations import w0gauss

__all__ = [
    "ShiftCurrent",
    "dipole_matrix",
    "generalized_derivative",
    "shift_current",
    "shift_integrand",
    "require_a_shift_current_regime",
]

#: Below this energy (Ry) a pair of bands counts as degenerate and every
#: ``1/w_nm`` it appears in is dropped. The pair is a genuine singularity of
#: the dipole and rule D4 says its individual matrix elements are not defined
#: anyway -- only the multiplet's sum is.
#:
#: **It is a floor, not the default.** :func:`shift_current` uses the
#: *broadening* instead, which is Elk's rule in ``nonlinopt.f90``
#: (``if (abs(t1) > swidth)``) and not ``dielectric.f90``'s 1e-8 that
#: :mod:`defumat.response.conductivity` follows. The difference is that a
#: linear conductivity carries ``1/w_nm`` once and this carries it three times.
#: Measured on AlAs: at the ``W`` points of a 4x4x4 fcc mesh a band pair sits
#: **9.18e-5 Ry** apart -- far above 1e-8, far below any resolvable splitting
#: -- and ``|r^{c;a}|`` there is **9.2e6 bohr^2** against a mesh median of
#: 4.4e3, which put a spike in the spectrum at 6.84 eV that moved by two orders
#: of magnitude between a 4x4x4, a 6x6x6 and an 8x8x8 mesh. **The tensor was
#: exactly ``-43m`` throughout**, to five figures, which is `NONLINEAR.md` §5's
#: warning arriving once more: the symmetry checks do not see this.
DEGENERACY_TOL = 1.0e-8

#: ``pi e^3 / (4 hbar E_Ry)`` in A/V^2 -- the SI prefactor of IATS18's Eq. (8)
#: once the dipoles are in bohr, the volume in bohr^3 (the two cancel) and the
#: smeared delta in 1/Ry. Derived rather than looked up: the ``-i(I + I')``
#: of Eq. (8) summed over *ordered* pairs equals ``Im(I + I')`` summed the same
#: way, exactly and with no factor lost, because the reversed pair is the
#: complex conjugate with the opposite ``f_nm``.
#:
#: **What validates it, and what does not.** Everything else in this module is
#: checked by an identity that is blind to an overall constant: the
#: complete-basis agreement tests ``generalised_derivative``, and the ``-43m``
#: form, the silicon zero and the below-gap vanishing would all survive a
#: ``sigma`` wrong by a factor of two. That is P50's trap in this module's
#: coordinates -- "the wrong one is exactly zincblende, exactly symmetric,
#: vanishes on silicon and is twice too large". The **spin sum** is anchored by
#: running a cell with no magnetization as ``nspin = 1`` and as a spinor and
#: requiring the same answer, which is the check P52 used for the same factor.
#:
#: **The overall scale is anchored against the published literature, and the
#: ordering is what makes that sharp.** AlAs's first peak converges to
#: **35.0 uA/V^2 at 4.17 eV**: 33.9 at ``ecutwfc = 16``, then 33.7, 34.8 and 35.0
#: at 22 with 14, 22 and 30 bands. First-principles calculations across the
#: fourteen III-V and II-VI zincblende semiconductors span **14 uA/V^2** (CdSe,
#: the smallest) to **83** (AlSb, the largest), and find the *aluminium*
#: compounds the strongest responders of the family and the II-VI compounds the
#: weakest (Opt. Quantum Electron. **58**, 10.1007/s11082-026-08937-7). A factor
#: of two either way breaks that ordering rather than merely moving the number:
#: 17 would put AlAs at the very bottom of the family, below the cadmium
#: chalcogenides, and 70 would put it beside AlSb, the heaviest and
#: narrowest-gap member and the one the trend says should be largest.
#:
#: **Two things about that sweep are worth keeping.** The number to compare is
#: the peak *below about 6 eV*, the range published spectra cover; the global
#: maximum of a wider window sits at 8.69 eV, where a band count of this size is
#: least trustworthy, and it is not a quantity anyone quotes. And
#: :attr:`ShiftCurrent.truncation` turned out to be **predictive of the real
#: error rather than merely indicative**: it falls 3.5% -> 1.0% over that band
#: sweep while the value moves 3.9%, so reading it is worth what it claims.
#:
#: **The convention is declared because the literature has two of them.** What is
#: implemented is IATS18's Eq. (1),
#: ``j^a = 2 sigma^abc(0; w, -w) Re[E_b(w) E_c(-w)]``, which is also the
#: convention of arXiv:2308.09641. Cook, Fregoso, de Juan, Coh and Moore
#: (arXiv:1507.08677) write ``J = sigma E E`` and report numbers twice as large
#: for the same physics, so a comparison against a paper that does not state
#: its normalisation is worth nothing to better than a factor of two. The
#: comparison above is against the first convention.
SIGMA_SI = 1.4049e-5


def _safe_ratio(numerator, denominator, tol: float):
    """``numerator / denominator`` where the denominator is resolvable, else 0."""
    finite = jnp.abs(denominator) > tol
    return jnp.where(finite, numerator / jnp.where(finite, denominator, 1.0), 0.0)


def dipole_matrix(energies, velocity, tol: float = DEGENERACY_TOL):
    """``r^a_nm = -i v^a_nm / w_nm``, ``(3, nk, nb, nb)``, in bohr.

    The interband Berry connection ``A^a_nm = i <u_n|d_a u_m>``, which for
    ``n != m`` is exactly ``-i v^a_nm / w_nm`` by first-order perturbation
    theory on ``H|m> = e_m|m>``. **Exactly**, not to within a truncation: it
    needs no sum over states, which is why a plane-wave code has no counterpart
    to Wannier90's ``AA_R``. That array exists only to undo the Wannier gauge,
    and there is no Wannier gauge here.

    The diagonal is zero by construction -- it is the *intraband* connection,
    which is gauge dependent and appears nowhere in a gauge-invariant answer --
    and so is every degenerate pair.
    """
    energies = jnp.asarray(energies)
    gap = energies[:, :, None] - energies[:, None, :]  # w_nm, (nk, nb, nb)
    return -1j * _safe_ratio(jnp.asarray(velocity), gap[None], tol)


def generalized_derivative(energies, velocity, second, tol: float = DEGENERACY_TOL):
    """``r^{c;a}_nm``, ``(3, 3, nk, nb, nb)`` indexed ``[a, c, k, n, m]``, in bohr^2.

    IATS18's Eq. (32) -- the module docstring's sum rule, with the intermediate
    sum written as the commutator that its ``p != n, m`` exclusions collapse
    into. ``a`` is the direction the derivative is taken along and ``c`` the
    cartesian component of the dipole being differentiated; the object is *not*
    symmetric in the two.

    Args:
        energies: ``(nk, nb)`` in Ry.
        velocity: ``(3, nk, nb, nb)``, ``<n|dH/dk_a|m>`` in Ry bohr.
        second: ``(3, 3, nk, nb, nb)``, ``<n|d^2H/dk_a dk_b|m>`` in Ry bohr^2.
    """
    energies = jnp.asarray(energies)
    velocity = jnp.asarray(velocity)
    second = jnp.asarray(second)

    gap = energies[:, :, None] - energies[:, None, :]  # w_nm
    g = _safe_ratio(velocity, gap[None], tol)  # v^a_nm / w_nm, = i r^a_nm
    diagonal = jnp.real(jnp.diagonal(velocity, axis1=-2, axis2=-1))  # (3, nk, nb)
    delta = diagonal[:, :, :, None] - diagonal[:, :, None, :]  # D^a_nm

    rows = []
    for a in range(3):
        columns = []
        for c in range(3):
            # The two intermediate sums, exclusions and all.
            commutator = (
                velocity[c] @ g[a] - g[a] @ velocity[c]
            ) - g[a] * delta[c]
            bracket = (
                _safe_ratio(
                    velocity[c] * delta[a] + velocity[a] * delta[c], gap, tol
                )
                - second[a, c]
                + commutator
            )
            columns.append(1j * _safe_ratio(bracket, gap, tol))
        rows.append(jnp.stack(columns))
    return jnp.stack(rows)


def shift_integrand(energies, velocity, second, tol: float = DEGENERACY_TOL):
    """``Im[r^b_mn r^{c;a}_nm + r^c_mn r^{b;a}_nm]``, ``(3, 3, 3, nk, nb, nb)``.

    Indexed ``[a, b, c, k, n, m]``: the current direction ``a`` and the two
    field directions ``b`` and ``c``, in which it is symmetric by construction.
    This is the whole k-resolved content of the shift current -- everything
    after it is occupations, a delta function and a Brillouin-zone sum -- and
    it is the object the validation compares, because a spectrum folds a
    tensor, a resonance condition and a smearing into one number.
    """
    r = dipole_matrix(energies, velocity, tol)  # (3, nk, nb, nb)
    gen = generalized_derivative(energies, velocity, second, tol)  # (3, 3, ...)
    r_mn = jnp.swapaxes(r, -1, -2)  # r^b_mn
    return jnp.stack([
        jnp.stack([
            jnp.stack([
                jnp.imag(r_mn[b] * gen[a, c] + r_mn[c] * gen[a, b])
                for c in range(3)
            ])
            for b in range(3)
        ])
        for a in range(3)
    ])


# -- the spectrum --------------------------------------------------------------


@dataclass
class ShiftCurrent:
    """What :func:`shift_current` returns.

    Attributes:
        frequencies: ``(nw,)`` in Ry -- the photon energies ``hbar omega``.
        sigma: ``(nw, 3, 3, 3)`` in A/V^2, indexed ``[w, a, b, c]``: the
            current along ``a`` from fields along ``b`` and ``c``, in which it
            is symmetric by construction. Real -- a shift current is a
            *rectified* response and has no phase.
        volume: the cell volume in bohr^3.
        broadening: the resonance delta's width in Ry.
        nbnd: how many bands the sums ran over -- both the ``n, m`` pair sum
            and, more importantly, the intermediate ``p`` sum of the
            generalised derivative. It is the convergence parameter.
        truncation: the largest change in :attr:`sigma` when the top
            **quarter** of the bands is dropped, relative to ``max|sigma|``.
            **Read it before believing a number.**

            A quarter and not one band, which is where this diagnostic differs
            from P37's ``static_residual`` and P47's ``BerryCurvature.
            truncation``. Those sums converge band by band and dropping the
            last one measures the tail. This one does not: the intermediate
            sum of the generalised derivative is an identity only over a
            *complete* basis, so its error is the whole tail rather than its
            last term, and dropping one band of sixteen on AlAs reports
            **5.3e-5** where the matrix elements themselves are 4 to 6 per cent
            away from a finite difference. One band under-reports by three
            orders of magnitude, and a diagnostic that does that is worse than
            none.
        band_cut_gap: ``min_k (e_(nbnd+1) - e_nbnd)`` in Ry when the caller
            diagonalised one extra band to measure it, else ``nan``. The same
            warning :class:`~defumat.response.conductivity.
            OpticalConductivity` carries: cutting inside a degenerate multiplet
            keeps some members and drops others.
    """

    frequencies: np.ndarray
    sigma: np.ndarray
    volume: float
    broadening: float
    nbnd: int
    truncation: float = float("nan")
    band_cut_gap: float = float("nan")

    @property
    def frequencies_ev(self) -> np.ndarray:
        """The photon energy axis in eV."""
        from defumat.units import RY_TO_EV

        return np.asarray(self.frequencies) * RY_TO_EV

    def component(self, a: int, b: int, c: int) -> np.ndarray:
        """``sigma^abc(w)`` as a function of frequency, ``(nw,)`` in A/V^2."""
        return np.asarray(self.sigma)[:, a, b, c]

    def __repr__(self) -> str:  # pragma: no cover - display only
        peak = float(np.max(np.abs(self.sigma))) if np.size(self.sigma) else 0.0
        return (
            f"ShiftCurrent(nw={len(self.frequencies)}, nbnd={self.nbnd}, "
            f"peak={peak:.3e} A/V^2, truncation={self.truncation:.2e})"
        )


def _sigma_at_k(energies, velocity, second, filling, weight, frequencies,
                broadening, ngauss, tol):
    """One k-point's contribution, ``(nw, 3, 3, 3)``, unnormalised by volume."""
    e = energies[None]
    integrand = shift_integrand(e, velocity[:, None], second[:, :, None], tol)
    integrand = integrand[..., 0, :, :]  # (3, 3, 3, nb, nb)

    occupancy = filling[:, None] - filling[None, :]  # f_n - f_m
    gap = energies[:, None] - energies[None, :]  # e_n - e_m
    # Both resonances, exactly as ``berry_get_sc_klist`` accumulates them: the
    # pair sum runs over *ordered* pairs, so each physical transition is met
    # once from each side and the two deltas keep the spectrum even in the sign
    # of the gap.
    argument = (gap[None] - frequencies[:, None, None]) / broadening
    delta = w0gauss(argument, ngauss) / broadening
    argument = (-gap[None] - frequencies[:, None, None]) / broadening
    delta = delta + w0gauss(argument, ngauss) / broadening

    weighted = occupancy[None] * delta  # (nw, nb, nb)
    return weight * jnp.einsum("abcnm,wnm->wabc", integrand, weighted)


def shift_current(
    calculation,
    wavefunctions,
    eigenvalues,
    v_scf,
    *,
    frequencies=None,
    window: float = 1.0,
    nw: int = 200,
    broadening: float = 0.01,
    smearing: str = "gaussian",
    band_cut_gap: float = float("nan"),
    k_batch: int | None | str = "default",
    degeneracy_tol: float | None = None,
) -> ShiftCurrent:
    """``sigma^abc(0; w, -w)``, the shift-current tensor, in A/V^2.

    Args:
        calculation: the fixed-density :class:`~defumat.scf.driver.
            Calculation` the states came from.
        wavefunctions: ``(nspin, nk, nbnd, ndim)`` -- **every** diagonalised
            band, not the occupied manifold: the intermediate sum needs them.
        eigenvalues: ``(nspin, nk, nbnd)`` in Ry.
        v_scf: the self-consistent potential the Hamiltonian was built with.
        frequencies: an explicit photon-energy axis in Ry; ``None`` builds a
            uniform one of ``nw`` points up to ``window``.
        broadening: the resonance delta's width in Ry.
        smearing: which smeared delta. **Gaussian whatever the run used**, for
            P52's reason: Methfessel-Paxton and cold smearing go negative on
            the wings, and a spectral weight that changes sign is not a
            resonance condition.
        degeneracy_tol: how close in Ry two bands may be before every
            ``1/w_nm`` between them is dropped. ``None`` -- the default -- is
            ``broadening``, which is Elk's rule and the one
            :data:`DEGENERACY_TOL` records the measurement for. A splitting the
            broadening cannot resolve is a multiplet as far as the spectrum is
            concerned, and a multiplet's individual matrix elements are not
            defined (rule D4).

    The frequency axis carries ``hbar omega`` in Ry, so a visible-light photon
    is around 0.15-0.25.
    """
    require_a_shift_current_regime(calculation)

    eigenvalues = jnp.asarray(eigenvalues)
    wavefunctions = jnp.asarray(wavefunctions)
    if eigenvalues.ndim == 2:
        eigenvalues, wavefunctions = eigenvalues[None], wavefunctions[None]
    if eigenvalues.shape[0] != 1:
        raise NotImplementedError(
            "the shift current of a collinear spin-polarized run is not "
            "implemented: the two channels are two band structures whose "
            "currents add, which is a loop this assembly does not have"
        )

    precision = calculation.system.cell.precision
    volume = float(calculation.system.cell.volume)
    nbnd = int(eigenvalues.shape[-1])

    velocity = VelocityOperator(calculation, v_scf)
    v = velocity.matrix_elements(wavefunctions)[:, 0]  # (3, nk, nb, nb)
    w = velocity.second_matrix_elements(wavefunctions)[:, :, 0]  # (3,3,nk,nb,nb)

    wg, _ = calculation.occupations(eigenvalues)
    wg = jnp.asarray(wg)[0]
    wk = jnp.asarray(calculation.system.kpoints.weights)
    filling = wg / wk[:, None]  # in [0, 1] per spin channel

    if frequencies is None:
        frequencies = np.linspace(0.0, float(window), int(nw))
    frequencies = jnp.asarray(np.asarray(frequencies, dtype=float))

    from defumat.scf.occupations import smearing_order

    ngauss = smearing_order(smearing)
    eta = precision.as_real(broadening)
    if degeneracy_tol is None:
        degeneracy_tol = max(float(broadening), DEGENERACY_TOL)
    batch = resolve_k_batch(k_batch)

    def total(bands: int):
        def one_k(arrays):
            e_k, v_k, w_k, f_k, wk_k = arrays
            return _sigma_at_k(
                e_k[:bands], v_k[:, :bands, :bands],
                w_k[:, :, :bands, :bands], f_k[:bands], wk_k,
                frequencies, eta, ngauss, degeneracy_tol,
            )

        arrays = (
            eigenvalues[0],
            jnp.moveaxis(v, 0, 1),
            jnp.moveaxis(w, 2, 0),
            filling,
            wk,
        )
        return np.asarray(sum_k(one_k, arrays, batch=batch)) * (SIGMA_SI / volume)

    sigma = total(nbnd)
    scale = float(np.max(np.abs(sigma)))
    truncation = float("nan")
    dropped = max(1, nbnd // 4)
    if nbnd - dropped > 1 and scale > 0.0:
        truncation = float(np.max(np.abs(total(nbnd - dropped) - sigma)) / scale)

    return ShiftCurrent(
        frequencies=np.asarray(frequencies),
        sigma=sigma,
        volume=volume,
        broadening=float(broadening),
        nbnd=nbnd,
        truncation=truncation,
        band_cut_gap=float(band_cut_gap),
    )



# -- the refusals --------------------------------------------------------------


def require_a_shift_current_regime(calculation) -> None:
    """Four refusals, each with the term it is missing named.

    The first three are :func:`~defumat.response.conductivity.
    require_a_conductivity_regime`'s, because this module is the same velocity
    matrix elements one derivative further along and inherits every reason. The
    fourth is its own.
    """
    if calculation.is_ultrasoft or calculation.is_paw:
        raise NotImplementedError(
            "the shift current with an ultrasoft or PAW pseudopotential is "
            "not implemented: the dipole of a generalised eigenproblem carries "
            "<psi_n|dS/dk_a|psi_m> beside dH/dk, and its second derivative "
            "carries d^2S/dk_a dk_b as well. Both are identically zero for a "
            "norm-conserving dataset, so nothing validated here can see "
            "whether their convention is right -- the refusal "
            "defumat.topology.kubo and defumat.response.conductivity both "
            "make for the first of the two. Use a norm-conserving dataset"
        )
    if getattr(calculation, "spiral", False):
        raise NotImplementedError(
            "the shift current of a spin spiral is not implemented: the two "
            "spinor components live on spheres centred at k + q/2 and k - q/2, "
            "so <n|dH/dk|m> is not a single contraction over one plane-wave set"
        )
    if _kpoints_are_reduced(calculation):
        raise NotImplementedError(
            "the shift current needs the whole k-grid, not a symmetry-reduced "
            "wedge: sigma^abc is a polar rank-3 tensor and a wedge sum is not "
            "the cell's until it is averaged over the point group, which this "
            "assembly does not do (defumat.system.symmetry."
            "symmetrize_cartesian_tensor would, and lifting this is a "
            "separate piece of work). Run with nosym = .true. and "
            "noinv = .true. on an *unshifted* grid, which is closed under the "
            "point group where a shifted one is not"
        )
    if calculation.is_hubbard:
        raise NotImplementedError(
            "the shift current of a DFT+U calculation is not implemented: the "
            "Hubbard term is another separable operator built from wfcU(k+G), "
            "so it carries a velocity and a second velocity of its own, and "
            "both would have to be threaded through with the converged ns. "
            "Nothing here has been validated against a Hubbard dH/dk^2. It is "
            "refused rather than left to surface as VelocityOperator's request "
            "for an ns this entry point has no way to pass it"
        )
    scheme = str(getattr(calculation.system, "occupations", "fixed")).lower()
    if scheme != "fixed":
        raise NotImplementedError(
            f"the shift current of a metal is not implemented (occupations = "
            f"{scheme!r}): the expression above is an interband sum weighted by "
            "f_n - f_m over pairs separated by a real gap, and a partially "
            "filled band contributes pairs whose energy difference goes to "
            "zero, where every 1/w_nm in the generalised derivative diverges. "
            "The Fermi-surface terms that replace them (an injection current, "
            "and the intraband part of the shift) are a different assembly. "
            "Use occupations = 'fixed'"
        )


def _kpoints_are_reduced(calculation) -> bool:
    """Whether the k-set is a wedge rather than the whole grid.

    The same test :func:`~defumat.response.conductivity._kpoints_are_reduced`
    makes and for its reason: an unreduced grid gives every k-point the same
    weight and a reduction is exactly what makes them differ.
    """
    weights = np.asarray(calculation.system.kpoints.weights)
    if weights.size <= 1:
        return False
    return bool(np.ptp(weights) > 1.0e-8 * np.abs(weights).max())
