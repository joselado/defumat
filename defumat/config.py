"""Runtime configuration: precision policy and solver-independent options.

The precision policy exists because the GPU target is hardware where float64 is
expensive, so single precision has to stay a usable mode. The rule that makes
that possible is: **no dtype literals in compute code**. Every array is created
with a dtype taken from a :class:`Precision` instance that is threaded through
construction, never with a hardcoded ``jnp.complex128`` or a bare ``1.0j``.

Validation against Quantum ESPRESSO always runs in float64 -- single precision
cannot reproduce QE to 1e-6 Ry and is a performance mode only.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np


class Precision(eqx.Module):
    """The real/complex dtype pair used to build every array.

    Attributes are static: changing precision should retrace, not silently
    reinterpret a traced value.
    """

    real: np.dtype = eqx.field(static=True)
    complex: np.dtype = eqx.field(static=True)
    name: str = eqx.field(static=True)

    @property
    def eps(self) -> float:
        """Machine epsilon of the real dtype, for tolerance-setting."""
        return float(np.finfo(self.real).eps)

    def as_real(self, x):
        return jnp.asarray(x, dtype=self.real)

    def as_complex(self, x):
        return jnp.asarray(x, dtype=self.complex)

    def zeros(self, shape, *, complex_: bool = False):
        return jnp.zeros(shape, dtype=self.complex if complex_ else self.real)


DOUBLE = Precision(real=np.dtype(np.float64), complex=np.dtype(np.complex128), name="double")
SINGLE = Precision(real=np.dtype(np.float32), complex=np.dtype(np.complex64), name="single")

#: Default for every calculation. Correctness claims are only ever made here.
DEFAULT_PRECISION = DOUBLE

#: The precision of every small dense solve in a subspace -- Davidson's projected
#: problem and the Rayleigh-Ritz of the start -- whatever the bands run in. Those
#: are ``m x m`` with ``m`` at most four times the band count, so double costs
#: nothing there, and single would not do: the overlap floor below which the
#: canonical route drops a direction (``solvers.subspace.OVERLAP_FLOOR``, 1e-12)
#: is five orders under float32's epsilon.
SUBSPACE_PRECISION = DOUBLE


def resolve_band_precision(requested="default") -> Precision:
    """The precision a calculation's band side runs in: the argument, ``DEFUMAT_BAND_PRECISION``, double.

    The band side is ``H|psi>``, the Davidson work arrays and the wavefunction
    store, where the cost and the memory of a run are; the grid side -- the
    density, the potential, the mixer, the energies -- stays in the cell's
    precision whatever this says (:class:`~defumat.scf.driver.Calculation`).
    """
    requested = _band_request(requested)
    if isinstance(requested, Precision):
        return requested
    # 'mixed' starts in single; :func:`band_precision_is_mixed` says it switches
    return precision_by_name("single" if requested == "mixed" else requested)


def band_precision_is_mixed(requested="default") -> bool:
    """Whether ``requested`` is ``'mixed'``: single to begin with, double to finish.

    The SCF switches the band side to the cell's precision at the first
    iteration whose diagonalisation threshold single cannot deliver
    (:func:`~defumat.scf.driver.run_scf`), with the mixer's history and the
    threshold schedule carried across, so the state it converges to is a
    double one.
    """
    return _band_request(requested) == "mixed"


def _band_request(requested):
    if isinstance(requested, Precision):
        return requested
    if requested is None or requested == "default":
        from defumat._envcompat import environ_get

        requested = (environ_get("DEFUMAT_BAND_PRECISION", "") or "").strip() or "double"
    return str(requested).strip().lower()


def subspace_dtype(dtype):
    """The dtype a subspace matrix of ``dtype`` is solved in: :data:`SUBSPACE_PRECISION`'s."""
    if jnp.issubdtype(dtype, jnp.complexfloating):
        return SUBSPACE_PRECISION.complex
    return SUBSPACE_PRECISION.real


def precision_by_name(name: str) -> Precision:
    """Look up a precision policy by the name used in input files."""
    table = {"double": DOUBLE, "float64": DOUBLE, "single": SINGLE, "float32": SINGLE}
    try:
        return table[name.lower()]
    except KeyError as exc:  # pragma: no cover - trivial
        raise ValueError(f"unknown precision {name!r}; expected one of {sorted(table)}") from exc
