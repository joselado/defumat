"""Gradient and divergence of a field on the grid, taken in G space.

A plane-wave code never differentiates on the real-space grid: a field expanded
in plane waves has an exact derivative, ``grad f = sum_G i G f_G exp(iGr)``, and
taking it costs one transform per direction instead of a finite-difference
stencil's truncation error. ``Modules/gradutils.f90`` (``fft_gradient_g2r`` and
``fft_graddot``) is the reference; the only difference here is that QE's ``g``
is in units of ``2 pi / alat`` and is multiplied by ``tpiba``, while
:meth:`GVectors.cartesian` already returns 1/bohr.

The pair exists for the gradient-corrected exchange-correlation functionals,
which need ``grad rho`` to evaluate and the divergence of a vector field to turn
the result back into a potential (see :mod:`pypresso.scf.potential`). Both are
differentiable and both are just multiplications in G space, so a strain or a
``k`` derivative flows through them.
"""

from __future__ import annotations

import jax.numpy as jnp

from pypresso.basis.fft import g_to_r, r_to_g
from pypresso.basis.gvectors import GVectors
from pypresso.system.cell import Cell

__all__ = ["gradient", "divergence", "laplacian"]


def gradient(field_g: jnp.ndarray, gvectors: GVectors, cell: Cell) -> jnp.ndarray:
    """``grad f`` on the real-space grid, from the field's G components.

    Args:
        field_g: ``(ngm,)`` coefficients on the G-vector sphere.

    Returns ``(3, n1, n2, n3)``, real. The three directions go through the
    transform as one batched call rather than a Python loop over ``ipol``: they
    share the scatter into the box, and on GPU three separate transforms of one
    array each would be three kernel launches where one does.
    """
    _reject_gamma_only(gvectors)
    g = gvectors.cartesian(cell)  # (ngm, 3), 1/bohr
    components = 1j * g.T * field_g[None, :]  # (3, ngm)
    return jnp.real(g_to_r(components, gvectors.fft_index, gvectors.grid))


def divergence(field_r: jnp.ndarray, gvectors: GVectors, cell: Cell) -> jnp.ndarray:
    """``div h`` on the real-space grid, for a real vector field ``h``.

    Args:
        field_r: ``(3, n1, n2, n3)``, real.

    The field is taken back to G space first, so what is differentiated is its
    projection onto the sphere -- which is what makes ``div grad`` on this pair
    exactly ``-|G|^2`` and keeps the gradient correction consistent with the
    density it was built from.
    """
    _reject_gamma_only(gvectors)
    g = gvectors.cartesian(cell)  # (ngm, 3)
    components = r_to_g(field_r, gvectors.fft_index)  # (3, ngm)
    divergence_g = jnp.sum(1j * g.T * components, axis=0)
    return jnp.real(g_to_r(divergence_g, gvectors.fft_index, gvectors.grid))


def laplacian(field_g: jnp.ndarray, gvectors: GVectors, cell: Cell) -> jnp.ndarray:
    """``lap f`` on the real-space grid, from the field's G components.

    ``-|G|^2 f_G``, one transform -- against the three a gradient costs and the
    four a divergence of a gradient would. QE has no counterpart, and the reason
    is worth recording: ``XClib/xc_wrapper_mgga.f90`` declares its Laplacian
    argument "not used in QE" and passes zeros to every libxc meta-GGA call, so
    a functional that needs one (Becke-Roussel, and Tran-Blaha on top of it) is
    evaluated there without it. In a plane-wave basis it is the cheapest
    derivative there is, which is why :mod:`pypresso.xc.mgga` can have it.
    """
    _reject_gamma_only(gvectors)
    g2 = jnp.sum(gvectors.cartesian(cell) ** 2, axis=1)  # (ngm,), 1/bohr^2
    return jnp.real(g_to_r(-g2 * field_g, gvectors.fft_index, gvectors.grid))


def _reject_gamma_only(gvectors: GVectors) -> None:
    """Half-sphere storage needs the conjugate half put back before an FFT.

    ``fft_gradient_g2r`` fills both ``nl`` and ``nlm`` when the descriptor is
    gamma-only, precisely because ``i G f_G`` over half the sphere is not the
    transform of a real field. None of that is written here, so the combination
    is refused rather than silently computed -- the same choice the augmentation
    charge makes.
    """
    if gvectors.gamma_only:
        raise NotImplementedError(
            "gamma-only storage with a gradient-corrected functional is not "
            "implemented; run with an explicit k-point at Gamma instead"
        )
