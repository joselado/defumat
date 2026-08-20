"""Which scheme moves the ions: a name, as every pluggable piece is.

``ion_dynamics`` in a ``pw.x`` ``&IONS`` namelist. Only ``bfgs`` -- QE's default
and the only one a ``calculation = 'relax'`` run uses unless told otherwise -- is
implemented; ``damp``, ``fire`` and the molecular-dynamics schemes of
``PW/src/dynamics_module.f90`` are additions of a file and a registration, which
is the point of the registry.
"""

from __future__ import annotations

__all__ = ["register_ion_dynamics", "get_ion_dynamics", "ion_dynamics_schemes",
           "DEFAULT_ION_DYNAMICS"]

DEFAULT_ION_DYNAMICS = "bfgs"

_SCHEMES: dict = {}


def register_ion_dynamics(name: str, factory) -> None:
    _SCHEMES[name.lower()] = factory


def get_ion_dynamics(name: str | None):
    """The named optimizer's factory, or the default when ``name`` is ``None``."""
    key = (name or DEFAULT_ION_DYNAMICS).lower()
    if key not in _SCHEMES:
        raise NotImplementedError(
            f"ion_dynamics = {name!r} is not implemented; available: "
            f"{sorted(_SCHEMES)}"
        )
    return _SCHEMES[key]


def ion_dynamics_schemes() -> tuple[str, ...]:
    return tuple(sorted(_SCHEMES))
