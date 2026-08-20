"""Which expression the forces come from: a name, as every pluggable piece is.

Two implementations, and they must agree to round-off:

``autodiff``
    ``-grad`` of the total energy (:mod:`pypresso.forces.autodiff`). The
    default, because differentiating the energy rather than re-deriving its
    derivative is the reason this code is written in JAX.
``analytic``
    QE's six hand-derived contributions, transcribed term by term
    (:mod:`pypresso.forces.analytic`). It is the reference the first one is
    checked against, and it is the only one that has the ``force_corr``
    correction for a density that is not quite converged.
"""

from __future__ import annotations

__all__ = ["register_force_method", "get_force_method", "force_methods",
           "DEFAULT_FORCE_METHOD"]

DEFAULT_FORCE_METHOD = "autodiff"

_METHODS: dict = {}


def register_force_method(name: str, function) -> None:
    _METHODS[name.lower()] = function


def get_force_method(name: str | None):
    """The named force implementation, or the default when ``name`` is ``None``."""
    key = (name or DEFAULT_FORCE_METHOD).lower()
    if key not in _METHODS:
        raise ValueError(
            f"unknown force method {name!r}; available: {sorted(_METHODS)}"
        )
    return _METHODS[key]


def force_methods() -> tuple[str, ...]:
    return tuple(sorted(_METHODS))
