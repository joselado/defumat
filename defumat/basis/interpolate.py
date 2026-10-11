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

__all__ = ["to_smooth", "to_dense", "restrict_g", "extend_g", "pair_to_smooth",
           "pair_to_dense"]


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


# --- a complex pair on a displaced sphere --------------------------------------
#
# A spin spiral's transverse magnetization ``m'_+ = m'_x + i m'_y`` (and its
# potential ``W = V_x + i V_y``) is band-limited to a sphere displaced from the
# resident one, ``|G - q| < G_cut`` (:func:`~defumat.basis.gvectors.displaced_gvectors`):
# ``2 conj(u_up) u_dn`` has its components at ``G2 - G1`` with
# ``|(k - q/2 + G2) - (k + q/2 + G1)| <= 2 G_w``, so it sits on the displaced
# smooth sphere exactly, and the resident cut would drop the shell of it the two
# spheres do not share. The two real components each have support on that sphere
# *and* its mirror, so the cut is taken on the complex combination, and the
# pair is taken apart again after the transform. ``pair`` is ``(2, n1, n2, n3)``.


def pair_to_dense(pair: jnp.ndarray, smooth: GVectors, dense: GVectors) -> jnp.ndarray:
    """:func:`to_dense` for a real pair ``(f_x, f_y)`` cut as ``f_x + i f_y`` on ``smooth``."""
    if smooth is dense or (smooth.grid == dense.grid and smooth.ngm == dense.ngm):
        return pair
    return _pair_to_dense(pair, smooth.fft_index, dense.fft_index, dense.grid, dense.ngm)


def pair_to_smooth(pair: jnp.ndarray, dense: GVectors, smooth: GVectors) -> jnp.ndarray:
    """:func:`to_smooth` for a real pair ``(f_x, f_y)`` cut as ``f_x + i f_y`` on ``smooth``."""
    if smooth is dense or (smooth.grid == dense.grid and smooth.ngm == dense.ngm):
        return pair
    return _pair_to_smooth(pair, dense.fft_index, smooth.fft_index, smooth.grid, smooth.ngm)


@partial(jax.jit, static_argnames=("grid", "ngm"))
def _pair_to_dense(pair, smooth_index, dense_index, grid, ngm):
    coefficients = extend_g(r_to_g(jax.lax.complex(pair[0], pair[1]), smooth_index), ngm)
    field = g_to_r(coefficients, dense_index, grid)
    return jnp.stack([jnp.real(field), jnp.imag(field)])


@partial(jax.jit, static_argnames=("grid", "ngms"))
def _pair_to_smooth(pair, dense_index, smooth_index, grid, ngms):
    coefficients = restrict_g(r_to_g(jax.lax.complex(pair[0], pair[1]), dense_index), ngms)
    field = g_to_r(coefficients, smooth_index, grid)
    return jnp.stack([jnp.real(field), jnp.imag(field)])
