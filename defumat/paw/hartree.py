"""The radial Poisson equation, solved the way Quantum ESPRESSO solves it.

Given the ``l``-th multipole of a density, ``r^2 rho_l(r)``, the Hartree
potential it produces is the solution of

    d^2/dr^2 (r V_l) - l(l+1)/r^2 (r V_l) = -4 pi r rho_l

and ``upflib/radial_grids.f90``'s ``hartree`` integrates it with a Numerov
scheme on the logarithmic mesh, closing the linear system with a small-``r``
series expansion at one end and the ``r^-(l+1)`` asymptote at the other. The
result is a symmetric positive-definite tridiagonal solve.

**Transcribed rather than replaced**, and this is the "mirror QE" rule doing
real work. The obvious alternative -- evaluating the closed form

    V_l(r) = 4 pi/(2l+1) [ r^-(l+1) int_0^r s^l (s^2 rho_l) ds
                         + r^l int_r^inf (s^2 rho_l) s^-(l+1) ds ]

by cumulative quadrature -- is correct for the continuous problem and would be
shorter. But the PAW one-centre energy has to agree with QE's to about 1e-8
*relative*, and two different discretisations of the same integral do not agree
to 1e-8; the same discretisation does, trivially. The transcription is also
differentiable and batches over ``lm`` under ``vmap``, which is all the JAX
rules ask of it.

The one place the Fortran needs decoding is its small-``r`` handling: it writes
the series coefficients into an array at an offset that depends on ``l``, so
that ``c2`` and ``c3`` pick up different polynomial coefficients for ``l = 0``
and ``l = 1`` and vanish for ``l >= 2``. That is spelled out in
:func:`_series_coefficients` rather than reproduced by index arithmetic.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax.lax.linalg import tridiagonal_solve

__all__ = ["radial_hartree", "RadialMesh"]


class RadialMesh:
    """The logarithmic mesh a radial solve runs on, with its derived arrays.

    Host-side and immutable: built once per species from the UPF file, then
    handed to the compiled solver as plain arrays.
    """

    __slots__ = ("r", "r2", "sqr", "rab", "dx", "mesh")

    def __init__(self, r, rab, dx: float):
        self.r = jnp.asarray(r)
        self.r2 = self.r**2
        self.sqr = jnp.sqrt(self.r)
        self.rab = jnp.asarray(rab)
        self.dx = float(dx)
        self.mesh = int(self.r.shape[0])
        if self.dx <= 0.0:
            raise ValueError(
                "the radial mesh has no logarithmic step (PP_MESH has no dx); "
                "the PAW radial Poisson solver needs one"
            )


def _series_coefficients(l: int, f, r, r2, nst: int):
    """``c2`` and ``c3``: the ``r^2`` and ``r^3`` terms of ``V_l`` near zero.

    ``hartree`` fits a cubic through the first four points of
    ``-(2l+1) f(r) / r^nst`` and then reads two of its coefficients off at an
    ``l``-dependent offset -- ``e(nk1)`` with ``nk1 = l + 1``, against ``e(1)``
    and ``e(2)``. Written out, that offset means:

    * ``l = 0``: the constant and linear coefficients of the fit;
    * ``l = 1``: zero and the constant coefficient;
    * ``l >= 2``: both zero, because the fit is written past the two slots that
      are read and they were left at zero.
    """
    if l >= 2:
        return 0.0, 0.0

    k21 = 2 * l + 1
    values = -k21 * f[:4] / r[:4] ** nst
    b = _cubic_through_four_points(values, r[:4], r2[:4])
    if l == 0:
        return b[0] / 6.0, b[1] / 12.0
    return 0.0, b[0] / 18.0


def _cubic_through_four_points(f, r, r2):
    """``upflib``'s ``series``: the cubic through four unevenly spaced points."""
    dr21, dr31, dr32 = r[1] - r[0], r[2] - r[0], r[2] - r[1]
    dr41, dr42, dr43 = r[3] - r[0], r[3] - r[1], r[3] - r[2]
    df21 = (f[1] - f[0]) / dr21
    df32 = (f[2] - f[1]) / dr32
    df43 = (f[3] - f[2]) / dr43
    ddf42 = (df43 - df32) / dr42
    ddf31 = (df32 - df21) / dr31

    b3 = (ddf42 - ddf31) / dr41
    b2 = ddf31 - b3 * (r[0] + r[1] + r[2])
    b1 = df21 - b2 * (r[1] + r[0]) - b3 * (r2[0] + r2[1] + r[0] * r[1])
    b0 = f[0] - r[0] * (b1 + r[0] * (b2 + r[0] * b3))
    return b0, b1, b2, b3


@partial(jax.jit, static_argnames=("l", "nst"))
def radial_hartree(f, r, r2, sqr, dx: float, l: int, nst: int):
    """Solve the radial Poisson equation for one multipole.

    Args:
        f: ``(mesh,)`` the source, already carrying its prefactor -- QE passes
            ``4 pi e^2 / (2l+1)`` times ``r^2 rho_l(r)``.
        r, r2, sqr: the mesh, its square, and its square root.
        dx: the logarithmic step.
        l: the multipole.
        nst: the power of ``r`` that ``f`` vanishes with at the origin. QE calls
            this routine with ``nst = 2l + 2``.

    Returns ``(mesh,)`` the potential, in whatever units ``f`` carried.
    """
    mesh = r.shape[0]
    k21 = 2 * l + 1
    c2, c3 = _series_coefficients(l, f, r, r2, nst)

    ch = dx * dx / 12.0
    xkh2 = ch * (l + 0.5) ** 2
    ei = 1.0 - xkh2

    # The Numerov right-hand side: a three-point stencil on k21 * ch * sqrt(r) f.
    source = k21 * ch * sqr * f
    rhs = source[:-2] + 10.0 * source[1:-1] + source[2:]  # rows 2..mesh-1

    # ... and the boundary rows, which fold the two closures into the matrix:
    # the r -> 0 series on the left and the r^-(l+1) decay on the right.
    f1 = (sqr[0] / sqr[1]) ** k21
    fn = (sqr[mesh - 2] / sqr[mesh - 1]) ** k21
    diagonal = jnp.full((mesh - 2,), 2.0 + 10.0 * xkh2)
    diagonal = diagonal.at[0].add(-ei * f1).at[-1].add(-ei * fn)
    rhs = rhs.at[0].add(
        -ei * sqr[0] ** k21 * (c2 * (r2[1] - r2[0]) + c3 * (r[1] ** 3 - r[0] ** 3))
    )

    off = jnp.full((mesh - 2,), xkh2 - 1.0)
    lower = off.at[0].set(0.0)
    upper = off.at[-1].set(0.0)
    interior = tridiagonal_solve(lower, diagonal, upper, rhs[:, None])[:, 0]

    # ... and the two points the solve left out.
    c0 = interior[0] / sqr[1] ** k21 - c2 * r2[1] - c3 * r[1] * r2[1]
    first = sqr[0] ** k21 * (c0 + c2 * r2[0] + c3 * r[0] ** 3)
    last = interior[-1] * fn

    potential = jnp.concatenate(
        [jnp.atleast_1d(first), interior, jnp.atleast_1d(last)]
    )
    return potential / sqr
