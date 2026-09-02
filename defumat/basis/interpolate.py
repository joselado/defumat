"""Moving a field between the dense and the smooth FFT grid.

``FFTXlib/src/fft_interpolate.f90`` (QE's ``fft_interpolate_real``): transform
to G space on the grid the field is on, copy the coefficients the *other* grid
can hold, and transform back there. Because the smooth G list is a prefix of the
dense one (see :mod:`defumat.basis.builder`), "copy the coefficients the other
grid can hold" is a slice going down and a zero-pad going up -- no index
matching, no interpolation in the numerical-analysis sense at all.

Which direction is used where follows QE:

* **dense -> smooth** for the local potential. ``set_vrs`` builds ``vrs`` on the
  dense grid and ``interpolate`` hands ``vloc_psi`` a smooth-grid copy. The high
  ``G`` components dropped are ones no product of two wavefunctions can see, so
  nothing is lost that ``H|psi>`` could have used.
* **smooth -> dense** for the density. ``sum_band`` accumulates ``|psi(r)|^2``
  on the smooth grid and interpolates it up before the augmentation charge --
  which needs the dense grid, that being the whole reason there are two -- is
  added on top.

When there is no double grid the two GVectors are the same object and both
functions are the identity, which is the norm-conserving path and stays free.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from defumat.basis.fft import g_to_r, r_to_g
from defumat.basis.gvectors import GVectors

__all__ = ["to_smooth", "to_dense", "restrict_g", "extend_g"]


def restrict_g(coefficients: jnp.ndarray, ngms: int) -> jnp.ndarray:
    """Dense-grid G coefficients -> smooth-grid ones: keep the first ``ngms``."""
    return coefficients[..., :ngms]


def extend_g(coefficients: jnp.ndarray, ngm: int) -> jnp.ndarray:
    """Smooth-grid G coefficients -> dense-grid ones: zero beyond ``ngms``."""
    pad = [(0, 0)] * (coefficients.ndim - 1) + [(0, ngm - coefficients.shape[-1])]
    return jnp.pad(coefficients, pad)


def to_smooth(field: jnp.ndarray, dense: GVectors, smooth: GVectors) -> jnp.ndarray:
    """A real field on the dense grid, resampled onto the smooth one."""
    if smooth is dense or (smooth.grid == dense.grid and smooth.ngm == dense.ngm):
        return field
    return _to_smooth(field, dense.fft_index, smooth.fft_index, smooth.grid, smooth.ngm)


def to_dense(field: jnp.ndarray, smooth: GVectors, dense: GVectors) -> jnp.ndarray:
    """A real field on the smooth grid, resampled onto the dense one."""
    if smooth is dense or (smooth.grid == dense.grid and smooth.ngm == dense.ngm):
        return field
    return _to_dense(field, smooth.fft_index, dense.fft_index, dense.grid, dense.ngm)


@partial(jax.jit, static_argnames=("grid", "ngms"))
def _to_smooth(field, dense_index, smooth_index, grid, ngms):
    coefficients = restrict_g(r_to_g(field, dense_index), ngms)
    return jnp.real(g_to_r(coefficients, smooth_index, grid))


@partial(jax.jit, static_argnames=("grid", "ngm"))
def _to_dense(field, smooth_index, dense_index, grid, ngm):
    coefficients = extend_g(r_to_g(field, smooth_index), ngm)
    return jnp.real(g_to_r(coefficients, dense_index, grid))
