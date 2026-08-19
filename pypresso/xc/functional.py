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

Adding a functional is a new component in :mod:`pypresso.xc.lda` or
:mod:`pypresso.xc.gga` plus one line in a table here (rule R4).

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

from pypresso.xc.gga import (
    PBE_KAPPA,
    PBE_MU,
    PBESOL_BETA,
    PBESOL_MU,
    REVPBE_KAPPA,
    no_gradient_correlation,
    no_gradient_exchange,
    pbe_correlation,
    pbe_exchange,
)
from pypresso.xc.lda import (
    RHO_THRESHOLD,
    no_correlation,
    no_exchange,
    pw_correlation,
    pz_correlation,
    slater_exchange,
)

__all__ = ["Functional", "get_functional", "resolve_functional", "FUNCTIONALS",
           "EXCHANGE", "CORRELATION", "GRADIENT_EXCHANGE", "GRADIENT_CORRELATION",
           "RHO_THRESHOLD_GGA", "SIGMA_THRESHOLD_GGA"]

#: ``rho_threshold_gga`` and ``grho_threshold_gga`` of
#: ``XClib/dft_setting_params.f90``.
RHO_THRESHOLD_GGA = 1.0e-6
SIGMA_THRESHOLD_GGA = 1.0e-10

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
        :func:`pypresso.scf.potential.exchange_correlation` -- so the pair is
        QE's, not the naive derivative of one expression.

        Zero wherever the density is below the vacuum threshold: those points do
        not contribute to the energy, and letting the derivative act there would
        put spurious structure into empty space.
        """
        rho = jnp.abs(jnp.asarray(rho))
        potential = jax.grad(lambda r: jnp.sum(r * self.energy_density(r)))(rho)
        return jnp.where(rho > RHO_THRESHOLD, potential, 0.0)

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
                f"composite names {sorted(FUNCTIONALS)}"
            )
        for index in found:
            slots[index] = _CANONICAL_TERM.get((index, token), token)

    return _from_slots(tuple(slots))


def _from_slots(slots: tuple[str, str, str, str]) -> Functional:
    exchange, correlation, gradient_exchange, gradient_correlation = slots
    return Functional(
        name=_SHORTNAMES.get(slots, "-".join(slots)),
        exchange=EXCHANGE[exchange],
        correlation=CORRELATION[correlation],
        gradient_exchange=GRADIENT_EXCHANGE[gradient_exchange],
        gradient_correlation=GRADIENT_CORRELATION[gradient_correlation],
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
            f"({sorted(names)}); QE refuses this and so does pypresso. Set "
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
