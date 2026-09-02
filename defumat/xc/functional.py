"""Composing an exchange-correlation functional, behind a name registry.

QE does not have "a functional"; it has four slots -- local exchange, local
correlation, and a gradient correction to each -- filled independently, and a
name like ``PBE`` is shorthand for one particular filling
(``XClib/qe_dft_list.f90``). This module reproduces that structure, because it
is also what UPF files carry: a pseudopotential's header names the four terms it
was generated with, and a calculation has to use the same ones.

    >>> get_functional("PBE").is_gradient
    True
    >>> get_functional(" SLA  PZ   NOGX NOGC").name
    'PZ'

Adding a functional is a new component in :mod:`defumat.xc.lda` or
:mod:`defumat.xc.gga` plus one line in a table here (rule R4).

**Only the energy is written down anywhere.** Both potentials come from
``jax.grad``:

* the local part gives ``v_xc = d(rho e_xc)/d rho``;
* the gradient part gives ``v1 = d e/d rho`` and ``v2 = 2 d e/d sigma``, with
  ``sigma = |grad rho|^2``. The factor of two is the chain rule through
  ``sigma``, and ``v2`` in this convention is exactly QE's -- the quantity its
  ``gradcorr`` multiplies ``grad rho`` by before taking the divergence.

**The thresholds are part of the functional, not an implementation detail.**
XClib evaluates nothing where ``rho <= 1e-6`` or ``|grad rho|^2 <= 1e-10`` in
the GGA drivers -- both the energy *and* both potentials are set to zero there,
and the cut is at a density six orders of magnitude larger than the LDA one.
That is not a rounding-level choice: a plane-wave density has low-density
regions covering much of the cell, and evaluating the gradient terms there
instead of zeroing them moves the total energy in the sixth decimal. The gating
is reproduced here, with the masked points evaluated on substitute values so
that ``grad`` sees no singular arithmetic through the ``where`` (the same
two-sided sanitisation the Perdew-Zunger branches use).
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from defumat.xc.gga import (
    PBE_KAPPA,
    PBE_MU,
    PBESOL_BETA,
    PBESOL_MU,
    REVPBE_KAPPA,
    no_gradient_correlation,
    no_gradient_correlation_spin,
    no_gradient_exchange,
    pbe_correlation,
    pbe_correlation_spin,
    pbe_exchange,
)
from defumat.xc.lda import (
    RHO_THRESHOLD,
    no_correlation,
    no_correlation_spin,
    no_exchange,
    pw_correlation,
    pw_correlation_spin,
    pz_correlation,
    pz_correlation_spin,
    slater_exchange,
)
from defumat.xc.mgga import tb09_coefficient, tb09_potential

__all__ = ["Functional", "get_functional", "resolve_functional", "FUNCTIONALS",
           "EXCHANGE", "CORRELATION", "GRADIENT_EXCHANGE", "GRADIENT_CORRELATION",
           "CORRELATION_SPIN", "GRADIENT_CORRELATION_SPIN", "META",
           "META_FUNCTIONALS",
           "RHO_THRESHOLD_GGA", "SIGMA_THRESHOLD_GGA", "SMALL_SPIN_GGA"]

#: ``rho_threshold_gga`` and ``grho_threshold_gga`` of
#: ``XClib/dft_setting_params.f90``.
RHO_THRESHOLD_GGA = 1.0e-6
SIGMA_THRESHOLD_GGA = 1.0e-10

#: ``gcx_spin``'s own gate, which is **not** ``RHO_THRESHOLD_GGA``: the
#: spin-polarized exchange driver cuts at ``small = 1e-10`` and tests it against
#: ``rho_sigma`` and against ``sqrt(|grho2_sigma|)`` -- the gradient's *modulus*,
#: not its square. Three differences from the unpolarized gate in one line, and
#: P13's experience is that this class of detail is worth ~1e-6 Ry.
SMALL_SPIN_GGA = 1.0e-10

#: What the masked-out points are evaluated at instead. Any finite, harmless
#: pair does; these are QE's own (``rho_trash``, ``grho2_trash``).
_RHO_TRASH = 0.5
_SIGMA_TRASH = 0.2


def _revpbe_exchange(rho, sigma):
    """``REVX``: PBE exchange with Zhang and Yang's larger ``kappa``."""
    return pbe_exchange(rho, sigma, REVPBE_KAPPA, PBE_MU)


def _pbesol_exchange(rho, sigma):
    """``PSX``: PBE exchange with the gradient expansion's ``mu = 10/81``."""
    return pbe_exchange(rho, sigma, PBE_KAPPA, PBESOL_MU)


def _pbesol_correlation(rho, sigma):
    """``PSC``: PBE correlation with PBEsol's smaller ``beta``."""
    return pbe_correlation(rho, sigma, PBESOL_BETA)


def _pbesol_correlation_spin(rho, zeta, sigma):
    """``PSC``, spin-polarized."""
    return pbe_correlation_spin(rho, zeta, sigma, PBESOL_BETA)


#: The four component tables, keyed by the names XClib and UPF headers use.
#: ``PBE`` appears in both gradient tables: it is the legacy spelling of ``PBX``
#: and ``PBC`` that older UPF files carry (``igcx = 14`` and ``igcc = 9``, which
#: ``set_dft_from_name`` maps back to 3 and 4 under a "TO BE REMOVED" comment
#: that has outlived several releases). Every ``*.pbe-*.UPF`` in the test set
#: spells it that way, so reading it is not optional.
EXCHANGE: dict[str, Callable] = {
    "NOX": no_exchange,
    "SLA": slater_exchange,
}

CORRELATION: dict[str, Callable] = {
    "NOC": no_correlation,
    "PZ": pz_correlation,
    "PW": pw_correlation,
}

GRADIENT_EXCHANGE: dict[str, Callable] = {
    "NOGX": no_gradient_exchange,
    "PBX": pbe_exchange,
    "PBE": pbe_exchange,
    "REVX": _revpbe_exchange,
    "PSX": _pbesol_exchange,
}

GRADIENT_CORRELATION: dict[str, Callable] = {
    "NOGC": no_gradient_correlation,
    "PBC": pbe_correlation,
    "PBE": pbe_correlation,
    "PSC": _pbesol_correlation,
}

#: The spin-polarized partners of the two correlation tables. Exchange has no
#: such table on purpose: the spin-scaling relation derives the polarized
#: exchange from *any* unpolarized slot exactly, and QE uses it that way
#: (``gcx_spin`` doubles each channel and calls the unpolarized routine). A
#: correlation slot with no entry here makes ``nspin = 2`` refuse rather than
#: silently run the unpolarized fit, which would converge to a plausible wrong
#: number -- the same convention P13 used for an unimplemented functional.
CORRELATION_SPIN: dict[str, Callable] = {
    "NOC": no_correlation_spin,
    "PZ": pz_correlation_spin,
    "PW": pw_correlation_spin,
}

GRADIENT_CORRELATION_SPIN: dict[str, Callable] = {
    "NOGC": no_gradient_correlation_spin,
    "PBC": pbe_correlation_spin,
    "PBE": pbe_correlation_spin,
    "PSC": _pbesol_correlation_spin,
}

#: Composite names, as ``input_dft`` or a UPF header may give them, spelled as
#: the four slots they stand for. From ``dft_full`` in ``qe_dft_list.f90``.
FUNCTIONALS: dict[str, tuple[str, str, str, str]] = {
    "PZ": ("SLA", "PZ", "NOGX", "NOGC"),
    "LDA": ("SLA", "PZ", "NOGX", "NOGC"),
    "PW": ("SLA", "PW", "NOGX", "NOGC"),
    "PBE": ("SLA", "PW", "PBX", "PBC"),
    "REVPBE": ("SLA", "PW", "REVX", "PBC"),
    "PBESOL": ("SLA", "PW", "PSX", "PSC"),
    "PBC": ("SLA", "PW", "NOGX", "PBC"),
}

#: The **potential-only** meta-GGA exchange slot, which the other four do not
#: reach: a name here replaces exchange entirely with a potential that is not
#: the derivative of any energy (:mod:`defumat.xc.mgga`). The value is the
#: fixed Tran-Blaha coefficient ``c``, or ``None`` where ``c`` is a cell average
#: of the density and therefore cannot be a constant.
#:
#: ``BJ06`` is listed for a reason beyond completeness: it is what ``pw.x``
#: actually runs when asked for ``tb09``. QE hands libxc its default parameter
#: list and libxc's default is ``c = 1``, so QE's Tran-Blaha is Becke-Johnson.
#: Having both here makes that difference measurable rather than a footnote.
META: dict[str, float | None] = {
    "TB09": None,
    "BJ06": 1.0,
}

#: The meta names as fillings of the other four slots. Exchange is ``NOX``
#: because the meta potential *is* the exchange -- adding Slater on top would
#: double it -- and correlation is the local Perdew-Wang one, which is what
#: Tran and Blaha pair the potential with ("the correlation potential is the
#: LDA one"). QE's libxc route pairs it with TPSS correlation instead
#: (``imetac = 231`` in ``dft_setting_routines.f90``), which is a meta-GGA
#: correlation and would need a ``tau`` derivative in the Hamiltonian; that is
#: a different functional and is not offered.
META_FUNCTIONALS: dict[str, tuple[tuple[str, str, str, str], str]] = {
    "TB09": (("NOX", "PW", "NOGX", "NOGC"), "TB09"),
    "MBJ": (("NOX", "PW", "NOGX", "NOGC"), "TB09"),
    "BJ06": (("NOX", "PW", "NOGX", "NOGC"), "BJ06"),
    "BJ": (("NOX", "PW", "NOGX", "NOGC"), "BJ06"),
}

#: The legacy spellings, resolved to the term they mean, per slot. QE does the
#: same fixup by index (``IF (igcx == 14) igcx = 3``); doing it here is what
#: makes an old ``SLA PW PBE PBE`` header come out named ``PBE`` rather than as
#: an unrecognisable filling of the slots.
_CANONICAL_TERM = {(2, "PBE"): "PBX", (3, "PBE"): "PBC"}

#: Canonical name for a filling of the four slots -- the inverse of the table
#: above, used so that a functional read from a UPF header prints as ``PBE``
#: rather than as the four terms it was spelled with.
_SHORTNAMES = {slots: name for name, slots in reversed(list(FUNCTIONALS.items()))}


class Functional(eqx.Module):
    """One filling of QE's four exchange-correlation slots.

    Every field is static, so the object is part of a ``jit`` signature rather
    than an argument: which functional is in use changes the compiled code, and
    nothing about it varies between calls.
    """

    name: str = eqx.field(static=True)
    exchange: Callable = eqx.field(static=True)
    correlation: Callable = eqx.field(static=True)
    gradient_exchange: Callable = eqx.field(static=True)
    gradient_correlation: Callable = eqx.field(static=True)
    #: The spin-polarized correlation partners, or ``None`` when the slot has
    #: none implemented. Checked when a calculation is set up, not when it is
    #: run, so an unsupported combination is refused before any work is done.
    correlation_spin: Callable | None = eqx.field(static=True, default=None)
    gradient_correlation_spin: Callable | None = eqx.field(static=True, default=None)
    #: The potential-only meta-GGA exchange slot: a key of :data:`META`, or
    #: ``None``. Static like the rest -- it changes the compiled code, and a
    #: calculation does not change functional halfway through.
    meta: str | None = eqx.field(static=True, default=None)
    #: An explicit Tran-Blaha ``c``, overriding the cell average. WIEN2k and
    #: VASP both expose the same knob, and it is not only a convenience: ``c``
    #: is an average of ``|grad rho|/rho`` over the cell, and a *pseudopotential*
    #: density has no core region, which is where that ratio is largest. A
    #: norm-conserving silicon gives ``c = 1.033`` where an all-electron
    #: calculation gives ``1.12``, so being able to impose the all-electron
    #: value is how the two are compared on equal footing. Reached from an input
    #: file as ``mbj_c`` in ``&system``. Named ``imposed_c`` and not ``meta_c``
    #: because :meth:`meta_c` is the method that *evaluates* the coefficient,
    #: and an ``equinox`` field of the same name shadows it silently.
    imposed_c: float | None = eqx.field(static=True, default=None)

    @property
    def is_meta(self) -> bool:
        """Whether exchange is a potential rather than an energy derivative.

        The single most consequential property in this class, because it makes
        the *total energy* meaningless: with no ``E_x`` there is nothing for
        the reported total to be the value of, and every quantity that is a
        derivative of the total -- forces, stress, the dynamical matrix, the
        response -- is a derivative of an expression the SCF did not minimise.
        Those are refused by name where they are computed, not here.
        """
        return self.meta is not None

    @property
    def meta_coefficient(self) -> float | None:
        """The fixed ``c``, or ``None`` when it is a cell average."""
        if self.meta is None:
            return None
        return META[self.meta] if self.imposed_c is None else self.imposed_c

    def with_meta_coefficient(self, c: float | None) -> "Functional":
        """The same functional with ``c`` imposed rather than averaged."""
        if c is None:
            return self
        if not self.is_meta:
            raise ValueError(
                f"mbj_c sets the Tran-Blaha coefficient and {self.name} is not a "
                "meta-GGA; it is read only by input_dft = 'tb09'"
            )
        return Functional(
            name=self.name,
            exchange=self.exchange,
            correlation=self.correlation,
            gradient_exchange=self.gradient_exchange,
            gradient_correlation=self.gradient_correlation,
            correlation_spin=self.correlation_spin,
            gradient_correlation_spin=self.gradient_correlation_spin,
            meta=self.meta,
            imposed_c=float(c),
        )

    def meta_exchange_potential(self, rho, sigma, laplacian, tau, c):
        """``v_x`` for one spin channel, Ry -- see :func:`defumat.xc.mgga.tb09_potential`.

        Args:
            rho: ``rho_sigma``, the channel density.
            sigma: ``|grad rho_sigma|^2``.
            laplacian: ``lap rho_sigma``.
            tau: ``tau_sigma`` in **Hartree**.
            c: the Tran-Blaha coefficient.
        """
        return tb09_potential(rho, sigma, laplacian, tau, c)

    def meta_c(self, rho, grad_rho):
        """``c`` for this functional: the cell average, or the fixed value."""
        fixed = self.meta_coefficient
        if fixed is not None:
            return jnp.asarray(fixed)
        return tb09_coefficient(rho, grad_rho)

    @property
    def is_gradient(self) -> bool:
        """Whether anything depends on ``grad rho`` (QE's ``dft_is_gradient``)."""
        return (
            self.gradient_exchange is not no_gradient_exchange
            or self.gradient_correlation is not no_gradient_correlation
        )

    # --- the local part -------------------------------------------------------

    def energy_density(self, rho: jnp.ndarray) -> jnp.ndarray:
        """``e_xc`` per electron, Ry."""
        return self.exchange(rho) + self.correlation(rho)

    def potential(self, rho: jnp.ndarray) -> jnp.ndarray:
        """``v_xc = d(rho e_xc)/d rho``, Ry, by differentiation.

        Taken at ``|rho|``, which is where ``xc_lda`` evaluates it: a truncated
        plane-wave density is slightly negative in vacuum and QE treats those
        points as low-density rather than as empty. The energy that goes with
        this potential keeps the *signed* density as its prefactor -- see
        :func:`defumat.scf.potential.exchange_correlation` -- so the pair is
        QE's, not the naive derivative of one expression.

        Zero wherever the density is below the vacuum threshold: those points do
        not contribute to the energy, and letting the derivative act there would
        put spurious structure into empty space.
        """
        rho = jnp.abs(jnp.asarray(rho))
        potential = jax.grad(lambda r: jnp.sum(r * self.energy_density(r)))(rho)
        return jnp.where(rho > RHO_THRESHOLD, potential, 0.0)

    def potential_and_energy_density(self, rho: jnp.ndarray):
        """``(v_xc, e_xc)`` from one pass, not two.

        :meth:`potential` differentiates ``sum(rho e_xc(rho))`` and
        :meth:`energy_density` evaluates ``e_xc`` -- the same transcendentals
        (a cube root, a log and a square root per grid point) twice over. The
        forward value of the differentiated expression already contains
        ``e_xc``, so ``value_and_grad`` with an auxiliary output returns both
        for the cost of the derivative alone.

        The two are the same numbers the separate methods give, exactly: the
        local functional is a function of ``|rho|`` alone
        (:func:`defumat.xc.lda.wigner_seitz_radius`), so evaluating it at the
        absolute value the potential needs also gives the energy density the
        *signed* density is then multiplied by.
        """
        rho = jnp.abs(jnp.asarray(rho))

        def total(r):
            energy_density = self.energy_density(r)
            return jnp.sum(r * energy_density), energy_density

        (_, energy_density), potential = jax.value_and_grad(total, has_aux=True)(rho)
        return jnp.where(rho > RHO_THRESHOLD, potential, 0.0), energy_density

    # --- the gradient correction ---------------------------------------------

    def gradient_energy(self, rho: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        """The gradient correction's energy **per unit volume**, Ry/bohr^3.

        QE's ``sx + sc``. Per volume rather than per electron because that is
        how XClib's GGA routines return it and how ``gradcorr`` integrates it;
        the local slots above are per electron, also following QE.
        """
        active, safe_rho, safe_sigma = _sanitise(rho, sigma)
        return jnp.where(active, self._gradient_energy(safe_rho, safe_sigma), 0.0)

    def gradient_potentials(self, rho: jnp.ndarray, sigma: jnp.ndarray):
        """``(v1, v2)``: QE's ``v1x + v1c`` and ``v2x + v2c``.

        ``v1`` is what adds to the local potential; ``v2`` is what multiplies
        ``grad rho`` to form the vector field whose divergence is subtracted.
        """
        active, safe_rho, safe_sigma = _sanitise(rho, sigma)

        def total(r, s):
            return jnp.sum(jnp.where(active, self._gradient_energy(r, s), 0.0))

        v1, dsigma = jax.grad(total, argnums=(0, 1))(safe_rho, safe_sigma)
        return v1, 2.0 * dsigma

    def _gradient_energy(self, rho, sigma):
        return self.gradient_exchange(rho, sigma) + self.gradient_correlation(rho, sigma)

    # --- the spin-polarized versions ------------------------------------------
    #
    # These are kept beside the unpolarized ones rather than replacing them with
    # a general ``(nspin, ...)`` implementation, and deliberately so: an
    # ``nspin = 1`` calculation must go on executing exactly the operations it
    # executed before, because the agreements those numbers are held to are at
    # 1e-9 Ry and "the arithmetic is equivalent" is not the same claim as "the
    # arithmetic is identical".

    @property
    def supports_spin(self) -> bool:
        """Whether both correlation slots have a spin-polarized partner."""
        return (
            self.correlation_spin is not None
            and self.gradient_correlation_spin is not None
        )

    def require_spin(self) -> None:
        """Refuse ``nspin = 2`` for a functional with no polarized correlation."""
        if not self.supports_spin:
            raise NotImplementedError(
                f"the correlation slots of {self.name} have no spin-polarized "
                "form implemented; nspin = 2 is refused rather than run with "
                "the unpolarized fit, which would converge to a wrong number"
            )

    def spin_energy_density(self, rho: jnp.ndarray) -> jnp.ndarray:
        """``e_xc`` per electron for a ``(2, ...)`` pair of channel densities.

        ``xc_lsda`` is handed the *total* density and ``zeta = m / |rho|`` with
        ``zeta`` clipped to [-1, 1], and evaluates everything at ``|rho|``; the
        signed total is only used afterwards, as the prefactor of ``etxc``. That
        split is reproduced here, so what this returns is a function of the
        sanitised pair alone and the caller supplies the sign.
        """
        return self._spin_energy_density(*_spin_channels(rho)[1:3])

    def spin_potential(self, rho: jnp.ndarray) -> jnp.ndarray:
        """``(v_up, v_dw)``: ``d(rho e_xc)/d rho_sigma``, Ry, by differentiation.

        QE derives these by hand -- ``slater_spin`` returns two potentials,
        ``pz_spin`` and ``pw_spin`` three terms each including an explicit
        ``df/dzeta`` -- and their correctness depends on agreeing with an energy
        written in another file. Here the same two numbers are the gradient of
        that energy with respect to the two channel densities.
        """
        active, arho, zeta = _spin_channels(rho)
        channels = jnp.stack([0.5 * arho * (1.0 + zeta), 0.5 * arho * (1.0 - zeta)])

        def total(pair):
            density = pair[0] + pair[1]
            polarization = (pair[0] - pair[1]) / jnp.maximum(density, RHO_THRESHOLD)
            return jnp.sum(density * self._spin_energy_density(density, polarization))

        return jnp.where(active, jax.grad(total)(channels), 0.0)

    def spin_potential_and_energy_density(self, rho: jnp.ndarray):
        """``(v_xc, e_xc)`` for the polarized case, from one pass.

        The unpolarized :meth:`potential_and_energy_density`'s reasoning, with
        the pair of channels in place of the single density.
        """
        active, arho, zeta = _spin_channels(rho)
        channels = jnp.stack([0.5 * arho * (1.0 + zeta), 0.5 * arho * (1.0 - zeta)])

        def total(pair):
            density = pair[0] + pair[1]
            polarization = (pair[0] - pair[1]) / jnp.maximum(density, RHO_THRESHOLD)
            energy_density = self._spin_energy_density(density, polarization)
            return jnp.sum(density * energy_density), energy_density

        (_, energy_density), potential = jax.value_and_grad(total, has_aux=True)(channels)
        return jnp.where(active, potential, 0.0), energy_density

    def _spin_energy_density(self, arho, zeta):
        """``e_x + e_c`` per electron at a positive density and its polarization.

        Exchange comes from the unpolarized slot by the spin-scaling relation
        ``E_x[n_up, n_dw] = (E_x[2 n_up] + E_x[2 n_dw]) / 2``, which is exact and
        is how ``gcx_spin`` gets it too. Correlation comes from the slot's own
        polarized parameterisation, because there is no such relation for it.
        """
        up = 0.5 * arho * (1.0 + zeta)
        down = 0.5 * arho * (1.0 - zeta)
        exchange = (
            up * self.exchange(2.0 * up) + down * self.exchange(2.0 * down)
        ) / jnp.maximum(arho, RHO_THRESHOLD)
        return exchange + self.correlation_spin(arho, zeta)

    def spin_gradient_energy(self, rho: jnp.ndarray, grad: jnp.ndarray) -> jnp.ndarray:
        """The gradient correction's energy per unit volume, ``(...)``.

        Args:
            rho: ``(2, ...)`` channel densities, core charge already folded in.
            grad: ``(2, 3, ...)`` their gradients.

        ``sx + sc`` of ``xc_gcx``. The two halves are gated differently and the
        difference is QE's, not an approximation: exchange is per channel, on
        ``rho_sigma`` and ``|grad rho_sigma|`` against 1e-10, while correlation
        is on the total density and ``|grad rho_total|`` against 1e-6 -- and it
        is the *total* density's gradient that correlation depends on, not the
        two channels' separately.
        """
        return self._spin_exchange_energy(rho, grad) + self._spin_correlation_energy(
            rho, grad
        )

    def spin_gradient_terms(self, rho: jnp.ndarray, grad: jnp.ndarray):
        """``(v1, h)``: the local part of the potential and the vector field.

        ``v1`` is ``(2, ...)`` and ``h`` is ``(2, 3, ...)``, with
        ``v_sigma = v1_sigma - div h_sigma``. QE assembles ``h`` from three
        hand-derived pieces per channel -- ``(v2x + v2c) grad rho_sigma +
        v2c_ud grad rho_-sigma`` -- and the cross term ``v2c_ud`` exists only
        because correlation depends on the total gradient. Differentiating with
        respect to the gradient *field* rather than to ``|grad rho|^2`` produces
        all three at once and cannot get their pairing wrong.
        """

        def total(density, gradient):
            return jnp.sum(self.spin_gradient_energy(density, gradient))

        return jax.grad(total, argnums=(0, 1))(rho, grad)

    def _spin_exchange_energy(self, rho, grad):
        """``sx``: the gradient correction to exchange, by spin scaling."""
        sigma = jnp.sum(grad * grad, axis=1)  # (2, ...) -- per channel
        # ``rho_up + rho_dw <= small`` zeroes both channels at once; each channel
        # is then separately replaced by the trash pair where it is too small or
        # its gradient is. ``rnull`` in ``gcx_spin`` is exactly this mask.
        both = (rho[0] + rho[1]) > SMALL_SPIN_GGA
        active = (
            both
            & (rho > SMALL_SPIN_GGA)
            & (jnp.sqrt(jnp.abs(sigma)) > SMALL_SPIN_GGA)
        )
        safe_rho = jnp.where(active, rho, _RHO_TRASH)
        safe_sigma = jnp.where(active, sigma, _SIGMA_TRASH)
        # The scaling: double the density, quadruple |grad rho|^2, halve the sum.
        contribution = self.gradient_exchange(2.0 * safe_rho, 4.0 * safe_sigma)
        return 0.5 * jnp.sum(jnp.where(active, contribution, 0.0), axis=0)

    def _spin_correlation_energy(self, rho, grad):
        """``sc``: the gradient correction to correlation, on the total density."""
        density = rho[0] + rho[1]
        gradient = grad[0] + grad[1]
        sigma = jnp.sum(gradient * gradient, axis=0)
        safe_density = jnp.maximum(jnp.abs(density), RHO_THRESHOLD_GGA)
        # ``gcc_spin`` clamps |zeta| to 1 - rho_threshold_gga before testing it,
        # so a polarization that rounds to exactly 1 is kept rather than cut.
        zeta = jnp.clip(
            (rho[0] - rho[1]) / safe_density,
            -(1.0 - RHO_THRESHOLD_GGA),
            1.0 - RHO_THRESHOLD_GGA,
        )
        active = (density > RHO_THRESHOLD_GGA) & (
            jnp.sqrt(jnp.abs(sigma)) > RHO_THRESHOLD_GGA
        )
        safe_sigma = jnp.where(active, sigma, _SIGMA_TRASH)
        contribution = self.gradient_correlation_spin(
            jnp.where(active, density, _RHO_TRASH), zeta, safe_sigma
        )
        return jnp.where(active, contribution, 0.0)


def _spin_channels(rho):
    """``(active, |rho_total|, zeta)`` from a ``(2, ...)`` pair of densities.

    ``xc_wrapper_lda_lsda.f90``: ``zeta = rho(2) / |rho(1)|`` wherever
    ``|rho(1)| > rho_threshold_lda``, and ``xc_lsda`` then clips it to [-1, 1].
    A plane-wave magnetization can exceed the density it is divided by where
    both are near zero, so the clip is doing real work rather than guarding
    against round-off.
    """
    rho = jnp.asarray(rho)
    density = rho[0] + rho[1]
    absolute = jnp.abs(density)
    active = absolute > RHO_THRESHOLD
    safe = jnp.maximum(absolute, RHO_THRESHOLD)
    zeta = jnp.clip((rho[0] - rho[1]) / safe, -1.0, 1.0)
    return active, safe, zeta


def _sanitise(rho, sigma):
    """QE's gate, and inputs the masked branch can be differentiated at.

    The test is on the *signed* density and the functional then sees its
    absolute value, both as in ``qe_drivers_gga.f90``.
    """
    rho = jnp.asarray(rho)
    sigma = jnp.asarray(sigma)
    active = (rho > RHO_THRESHOLD_GGA) & (sigma > SIGMA_THRESHOLD_GGA)
    return (
        active,
        jnp.where(active, jnp.abs(rho), _RHO_TRASH),
        jnp.where(active, sigma, _SIGMA_TRASH),
    )


@lru_cache(maxsize=None)
def get_functional(name: str) -> Functional:
    """The functional a name stands for, in any of the spellings QE accepts.

    Accepted: a composite name (``PBE``, ``PZ``, ``PBESOL``); the four slots
    separated by spaces, dashes or plus signs (``SLA PW PBX PBC``,
    ``SLA-PW-PBX-PBC``), which is what UPF headers carry; and any subset of the
    slots, with the rest left empty.

    An unrecognised term is an error rather than a silently empty slot. That is
    deliberate: the failure mode it prevents is a PBE pseudopotential run with
    an LDA functional, which converges perfectly well to a total energy that is
    wrong by tenths of a Rydberg and matches no reference.
    """
    tokens = [token for token in name.replace("-", " ").replace("+", " ").split() if token]
    if not tokens:
        raise ValueError("empty exchange-correlation functional name")

    upper = [token.upper() for token in tokens]
    if len(upper) == 1 and upper[0] in META_FUNCTIONALS:
        # Checked before the four-slot table: a meta name stands for a filling
        # of those slots *and* for the potential that replaces exchange, and
        # dropping the second half would leave a bare Perdew-Wang correlation
        # run that converges and means nothing.
        return _from_slots(*META_FUNCTIONALS[upper[0]])
    if len(upper) == 1 and upper[0] in FUNCTIONALS:
        return _from_slots(FUNCTIONALS[upper[0]])

    slots = ["NOX", "NOC", "NOGX", "NOGC"]
    for token in upper:
        tables = (EXCHANGE, CORRELATION, GRADIENT_EXCHANGE, GRADIENT_CORRELATION)
        found = [index for index, table in enumerate(tables) if token in table]
        if not found:
            raise ValueError(
                f"unknown exchange-correlation term {token!r} in {name!r}; "
                f"implemented terms are {sorted(set().union(*tables))} and the "
                f"composite names {sorted(set(FUNCTIONALS) | set(META_FUNCTIONALS))}"
            )
        for index in found:
            slots[index] = _CANONICAL_TERM.get((index, token), token)

    return _from_slots(tuple(slots))


def _from_slots(slots: tuple[str, str, str, str], meta: str | None = None) -> Functional:
    exchange, correlation, gradient_exchange, gradient_correlation = slots
    return Functional(
        name=meta or _SHORTNAMES.get(slots, "-".join(slots)),
        meta=meta,
        exchange=EXCHANGE[exchange],
        correlation=CORRELATION[correlation],
        gradient_exchange=GRADIENT_EXCHANGE[gradient_exchange],
        gradient_correlation=GRADIENT_CORRELATION[gradient_correlation],
        correlation_spin=CORRELATION_SPIN.get(correlation),
        gradient_correlation_spin=GRADIENT_CORRELATION_SPIN.get(gradient_correlation),
    )


def resolve_functional(pseudo_dfts, input_dft: str | None = None) -> Functional:
    """Which functional a calculation runs, from its pseudopotentials and input.

    ``PW/src/input.f90`` and ``upflib/read_upf_new.f90``: every pseudopotential
    carries the functional it was generated with, all of them must agree, and
    ``input_dft`` in the ``&system`` namelist overrides the lot. Overriding is a
    legitimate thing to do -- QE's own ``pw_dft`` tests run GGA functionals on
    an LDA-generated silicon pseudopotential to compare functionals on equal
    footing -- but it is inconsistent by construction, so QE announces it and so
    does this.

    Args:
        pseudo_dfts: the ``functional`` string of each pseudopotential.
        input_dft: an explicit override, or ``None``.
    """
    if input_dft is not None and input_dft.strip():
        # Parsed before the pseudopotentials' own strings are looked at: an
        # override is exactly the situation in which one of those may name a
        # functional this code does not implement, and refusing it then would
        # refuse the run the override was meant to make possible.
        requested = get_functional(input_dft)
        _announce_override(requested, pseudo_dfts)
        return requested

    from_pseudos = [get_functional(dft) for dft in pseudo_dfts if dft and dft.strip()]

    if not from_pseudos:
        raise ValueError(
            "no exchange-correlation functional: the pseudopotentials name none "
            "and the input does not set input_dft"
        )

    names = {functional.name for functional in from_pseudos}
    if len(names) > 1:
        raise ValueError(
            f"the pseudopotentials were generated with different functionals "
            f"({sorted(names)}); QE refuses this and so does defumat. Set "
            f"input_dft explicitly to run them together anyway"
        )
    return from_pseudos[0]


def _announce_override(requested: Functional, pseudo_dfts) -> None:
    """Warn when ``input_dft`` disagrees with what the pseudopotentials expect.

    ``set_dft_from_name`` prints the same warning. It is only a warning because
    the combination is useful and QE's own test suite relies on it, but it is
    never silent: a total energy computed this way is not comparable with a
    published one for the same functional.
    """
    for dft in pseudo_dfts:
        if not dft or not dft.strip():
            continue
        try:
            generated_with = get_functional(dft)
        except ValueError:
            continue
        if generated_with.name != requested.name:
            warnings.warn(
                f"input_dft asks for {requested.name} but the pseudopotentials were "
                f"generated with {generated_with.name}; running them together is "
                "inconsistent, as it is in QE",
                stacklevel=3,
            )
            return


def local_spin_frame(charge: jnp.ndarray, magnetization: jnp.ndarray):
    """The ``(up, down)`` pair along the local spin axis of a noncollinear density.

    ``v_xc``'s ``nspin == 4`` branch and ``PAW_xc_potential``'s do the same
    thing, on the plane-wave grid and on the radial sphere, and this is the part
    they share: at every point the magnetization picks out an axis, and the
    density resolved on it is

        rho_up = (n + |m|) / 2,   rho_down = (n - |m|) / 2.

    ``|m|`` is clamped to ``|n|`` because a magnetization larger than the charge
    is not a physical state and would make one channel negative -- the same
    clamp ``xc_lsda`` applies to ``zeta``.

    Returns ``(channels, modulus, direction)``: the ``(2, ...)`` pair, the
    clamped modulus, and the unit vector the splitting is attached to (zero
    where there is no magnetization to define one, as QE zeroes the vector part
    of the potential there rather than picking a direction out of rounding
    error).
    """
    modulus = jnp.sqrt(jnp.sum(magnetization**2, axis=0))
    clamped = jnp.minimum(modulus, jnp.abs(charge))
    channels = jnp.stack([(charge + clamped) / 2.0, (charge - clamped) / 2.0])
    safe = jnp.where(modulus > 0.0, modulus, 1.0)
    direction = jnp.where(modulus > VANISHING_MAGNETIZATION, magnetization / safe, 0.0)
    return channels, modulus, direction


#: Below this magnetization the local spin axis is undefined, so the potential's
#: vector part is set to zero rather than to a direction picked out of rounding
#: error. ``vanishing_mag`` in ``PW/src/v_of_rho.f90``, and ``eps12`` in
#: ``paw_onecenter.f90``.
VANISHING_MAGNETIZATION = 1.0e-20
