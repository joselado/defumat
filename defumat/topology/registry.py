"""Which construction each topological quantity comes from: a name, as
every pluggable piece in this codebase is (rule R4).

Two registries, because there are two independent choices.

**Curvature** (:mod:`defumat.topology.berry`)

``fhs``
    The lattice field strength of Fukui, Hatsugai and Suzuki -- overlaps between
    neighbouring k-points, gauge invariant and integer-quantised. The default,
    and the only one from which a Chern number should be read.
``kubo``
    The sum-over-states expression with the velocity operator from ``jacfwd`` of
    ``H(k)``. Pointwise and smooth; not quantised, and singular at a degeneracy.

**Z2** (:mod:`defumat.topology.wilson`, :mod:`defumat.topology.parity`)

``wilson``
    Wannier-charge-centre flow across half the zone, counted by the largest-gap
    method (Yu, Qi, Bernevig, Fang and Dai, PRB 84, 075119 (2011); Soluyanov and
    Vanderbilt, PRB 83, 235401 (2011)). The default: it needs nothing of the
    crystal but time-reversal symmetry.
``parity``
    The Fu-Kane product of parity eigenvalues at the time-reversal-invariant
    momenta (Fu and Kane, PRB 76, 045302 (2007)). Exact and enormously cheaper
    -- eight k-points against a whole half-zone mesh -- but it needs an
    inversion centre, so it is a cross-check on the first rather than a
    replacement for it.

The two are independent derivations sharing no machinery beyond the state set,
which is what makes their agreement on an inversion-symmetric crystal a real
check and not a tautology.
"""

from __future__ import annotations

__all__ = [
    "register_curvature_method",
    "get_curvature_method",
    "curvature_methods",
    "register_z2_method",
    "get_z2_method",
    "z2_methods",
    "DEFAULT_CURVATURE_METHOD",
    "DEFAULT_Z2_METHOD",
]

DEFAULT_CURVATURE_METHOD = "fhs"
DEFAULT_Z2_METHOD = "wilson"

_CURVATURE: dict = {}
_Z2: dict = {}


def register_curvature_method(name: str, function) -> None:
    _CURVATURE[name.lower()] = function


def get_curvature_method(name: str | None):
    key = (name or DEFAULT_CURVATURE_METHOD).lower()
    if key not in _CURVATURE:
        raise ValueError(
            f"unknown Berry curvature method {name!r}; available: {sorted(_CURVATURE)}"
        )
    return _CURVATURE[key]


def curvature_methods() -> tuple[str, ...]:
    return tuple(sorted(_CURVATURE))


def register_z2_method(name: str, function) -> None:
    _Z2[name.lower()] = function


def get_z2_method(name: str | None):
    key = (name or DEFAULT_Z2_METHOD).lower()
    if key not in _Z2:
        raise ValueError(f"unknown Z2 method {name!r}; available: {sorted(_Z2)}")
    return _Z2[key]


def z2_methods() -> tuple[str, ...]:
    return tuple(sorted(_Z2))
