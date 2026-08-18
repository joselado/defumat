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


def precision_by_name(name: str) -> Precision:
    """Look up a precision policy by the name used in input files."""
    table = {"double": DOUBLE, "float64": DOUBLE, "single": SINGLE, "float32": SINGLE}
    try:
        return table[name.lower()]
    except KeyError as exc:  # pragma: no cover - trivial
        raise ValueError(f"unknown precision {name!r}; expected one of {sorted(table)}") from exc
