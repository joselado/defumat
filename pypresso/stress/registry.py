"""Which expression the stress comes from: a name, as every pluggable piece is.

The same two-implementation arrangement :mod:`pypresso.forces.registry`
describes, one coordinate over:

``autodiff``
    ``-(1/Omega) grad`` of the total energy with respect to a strain applied to
    the cell (:mod:`pypresso.stress.autodiff`). The default, because
    differentiating the energy rather than re-deriving its derivative is the
    reason this code is written in JAX.
``analytic``
    Quantum ESPRESSO's hand-derived contributions, transcribed term by term
    (:mod:`pypresso.stress.analytic`). It is the reference the first one is
    checked against, and it shares no machinery with it.

**The analytic route does not offer a total.** ``stres_us`` -- the projectors'
own strain derivative, 632 lines of Fortran with two auxiliary generators
(``gen_us_dj``, ``gen_us_dy``) behind it -- is not transcribed, and neither is
``addusstress``. Every real system has a nonlocal pseudopotential, so a sum of
the terms that *are* written would be wrong by a large and entirely plausible
amount. :func:`~pypresso.stress.compute_stress` therefore refuses
``method = 'analytic'`` by name and the terms are reached through
:func:`~pypresso.stress.analytic_terms`, which is what the cross-check test
compares against the autodiff decomposition.
"""

from __future__ import annotations

__all__ = ["register_stress_method", "get_stress_method", "stress_methods",
           "DEFAULT_STRESS_METHOD"]

DEFAULT_STRESS_METHOD = "autodiff"

_METHODS: dict = {}


def register_stress_method(name: str, function) -> None:
    _METHODS[name.lower()] = function


def get_stress_method(name: str | None):
    """The named stress implementation, or the default when ``name`` is ``None``."""
    key = (name or DEFAULT_STRESS_METHOD).lower()
    if key not in _METHODS:
        raise ValueError(
            f"unknown stress method {name!r}; available: {sorted(_METHODS)}"
        )
    return _METHODS[key]


def stress_methods() -> tuple[str, ...]:
    return tuple(sorted(_METHODS))
