"""Link variables: the gauge-invariant content of an overlap matrix.

Every quantity in this subpackage reduces an overlap matrix ``M`` between two
neighbouring k-points to one of two things:

``link_phase(M) = det M / |det M|``
    a single complex number of unit modulus. The determinant is what makes it
    blind to any unitary mixing *inside* the occupied manifold -- which is the
    only freedom a degenerate eigensolver leaves -- and dividing by the modulus
    is Fukui-Hatsugai-Suzuki's normalisation, which matters not for the phase
    but for keeping products of many links from under- or overflowing.

``unitarize(M) = U V^H`` from ``M = U S V^H``
    the closest unitary matrix, which is what a Wilson loop multiplies. A
    *product* of overlap matrices is not the same object as a product of their
    determinants: the Wannier centres are the phases of the eigenvalues of the
    product, so the matrix structure has to survive, and each factor must be
    unitary for the product's eigenvalues to lie on the unit circle at all.
    (Soluyanov and Vanderbilt, PRB 83, 235401 (2011); the polar decomposition is
    the standard "closest unitary" of Marzari-Vanderbilt parallel transport.)

Both fail loudly on a singular ``M``. A vanishing determinant means the occupied
manifolds at two neighbouring k-points are orthogonal, which on any usable mesh
means the mesh is too coarse to follow the band, or the manifold is not gapped.
Clipping it would return a number rather than an error, and that number would be
an integer that is simply wrong.
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = ["link_phase", "unitarize", "berry_phase"]

#: Below this, a singular value or a determinant is treated as zero and raises.
SINGULAR = 1.0e-8


def link_phase(matrix: jnp.ndarray) -> jnp.ndarray:
    """``det M / |det M|``, over a trailing ``(n, n)`` block.

    Computed through ``slogdet``, whose sign output *is* the normalised
    determinant for a complex matrix and which does not overflow for a large
    manifold -- a 30-band determinant of overlaps each a little under one is
    ``1e-10`` or smaller, and the product of a whole loop of them is not
    representable.
    """
    sign, logabs = jnp.linalg.slogdet(matrix)
    return sign


def unitarize(matrix: jnp.ndarray) -> jnp.ndarray:
    """The closest unitary matrix to ``M``, by polar decomposition."""
    u, _, vh = jnp.linalg.svd(matrix, full_matrices=False)
    return u @ vh


def berry_phase(product: jnp.ndarray) -> jnp.ndarray:
    """The Berry phase of a closed loop's link product: ``-arg``.

    **The sign is the whole content of this function**, and it is set in exactly
    one place so that curvature, Chern numbers and everything derived from them
    cannot disagree about it.

    Write ``U_mu(k) = <u_k|u_{k+mu}> / |...| = exp(-i A_mu(k) delta)`` with
    ``A_mu = i <u|d_mu u>`` the Berry connection in the standard convention
    (Xiao, Chang and Niu, Rev. Mod. Phys. 82, 1959 (2010)). The plaquette
    product is then

        w = U_1(k) U_2(k+1) U_1(k+2)^{-1} U_2(k)^{-1}
          = exp(-i delta^2 [d_1 A_2 - d_2 A_1]) = exp(-i delta^2 Omega_12),

    so ``Omega_12 = -arg(w) / delta^2``. The negation is not cosmetic: omitting
    it returns ``-Omega`` and ``-C``, which is self-consistent, passes every
    gauge-invariance test, and disagrees with the literature. The absolute pin is
    an analytic one -- the spin-1/2 coherent state ``|n(theta, phi)>``, whose
    curvature is ``-sin(theta)/2`` -- and it is in
    ``tests/unit/test_topology_curvature.py``.

    (``pyqula`` stores its overlaps conjugated and therefore reports ``-Omega``
    relative to this; ``elkpy`` uses the same convention as here. A sign
    disagreement with one of them is expected and is not a bug in either.)
    """
    return -jnp.angle(product)
