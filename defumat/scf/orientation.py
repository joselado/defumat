"""Turning the moments inside the SCF loop: ``ORIENTATION-NEXT.md`` Route C.

**Why a run with spin-orbit coupling needs this.** The orientation of a magnetic
texture is an electronic degree of freedom, so an SCF with the coupling is free
to turn the moments itself, and at its fixed point the energy is stationary under
every rotation. But the rotation is a soft mode: the lean of the output
magnetization off the field it was solved in is the anisotropy over the
exchange, of order 1e-5 rad, and a mixer takes a fraction ``beta`` of it per
iteration. Measured on tetragonal cobalt started 45 degrees off ``c`` (``PLAN.md``
P122): 100 iterations, the angle wandering between 36 and 54 degrees, ``dr2``
between 1e-8 and 2e-5, and the torque steady at -3.3e-5 Ry per radian the whole
time.

**What is done instead.** Every iteration has diagonalised ``H[rho_in]`` with the
coupling, which is the force theorem's calculation with ``rho_in`` frozen, so the
torque on a rigid rotation of ``rho_in`` is ``integral of B[rho_in] x m_out``
(``SCFResult.orientation_torque``), the gradient of the Harris-Foulkes energy in
the orientation. After the mix the input density is turned by
``w = -H^-1 G``, ``G`` being minus that torque: the component of the residual
along the three global spin rotations divided by their curvature, where the mixer
multiplies it by ``beta``. Those three are the only modes this is needed for,
because they are exactly the zero modes the coupling alone gaps; the angles
inside a texture are held by exchange and converge as they always did.

**The curvature** starts from nothing: the first step is ``first_step`` radians
along ``-G``, and a 3x3 BFGS in the rotation vector builds ``H`` from the torques
after it, skipping a pair whose ``s . y`` is not positive, since early on the
density's own convergence changes ``G`` as much as the rotation does. Every step
is bounded by a trust angle and none is taken while ``dr2`` is above ``start``.

**The trust angle adapts**, which is what a nearly flat direction needs. On PAW
nickel the moment went in-plane, where the anisotropy is nearly flat, BFGS's
inverse Hessian was large along it, and a torque of a few 1e-6 asked for more
than the bound every time: two full 5.7-degree steps took ``dr2`` from 2.4e-6 to
1.7e-4, where it sat for twenty iterations. So the bound halves whenever the
torque has not fallen since the last step, down to ``trust / 64``, and grows back
by half as much again while it keeps falling, up to ``trust``.

**Without the coupling nothing happens by construction**, not because ``G`` is
small: it is zero only for exact eigenstates, and dividing the solver's noise by
a curvature that is itself zero would make steps out of nothing. The driver
refuses the option there.
"""

from __future__ import annotations

import numpy as np

__all__ = ["OrientationStepper", "rotate_spinors", "spin_turned",
           "spinor_rotation", "rotation_matrix"]


def rotation_matrix(omega) -> np.ndarray:
    """``exp([w]x)`` by Rodrigues' formula, on the host."""
    omega = np.asarray(omega, dtype=float)
    angle = float(np.linalg.norm(omega))
    if angle < 1.0e-14:
        return np.eye(3)
    axis = omega / angle
    cross = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * cross @ cross


def spinor_rotation(omega) -> np.ndarray:
    """``exp(-i sigma . w / 2)``, the spin-1/2 rotation that turns ``m`` by ``exp([w]x)``."""
    omega = np.asarray(omega, dtype=float)
    angle = float(np.linalg.norm(omega))
    if angle < 1.0e-14:
        return np.eye(2, dtype=complex)
    nx, ny, nz = omega / angle
    c, s = np.cos(0.5 * angle), np.sin(0.5 * angle)
    return np.array([
        [c - 1j * s * nz, -1j * s * (nx - 1j * ny)],
        [-1j * s * (nx + 1j * ny), c + 1j * s * nz],
    ])


def rotate_spinors(psi, omega):
    """Turn every spinor ``(..., 2 npwx)`` by ``spinor_rotation(omega)``.

    A warm start for the next diagonalisation after the density has been
    turned: the states solved in the old frame, carried into the new one. With
    the coupling they are not the new frame's eigenstates, which is why they
    are only a start.
    """
    u = spinor_rotation(omega)
    psi = np.asarray(psi)
    npwx = psi.shape[-1] // 2
    up, down = psi[..., :npwx], psi[..., npwx:]
    u = u.astype(psi.dtype)
    return np.concatenate([u[0, 0] * up + u[0, 1] * down,
                           u[1, 0] * up + u[1, 1] * down], axis=-1)


def spin_turned(psi, omega):
    """``(1 - i sigma . w / 2) psi``, traceable in ``w``: the spin turn to first order.

    Exact in value and first derivative at ``w = 0``, which is where the torque
    reads it, and with no norm of ``w`` in it (``forces/torque.py`` says why that
    matters at the origin). :func:`rotate_spinors` is the exact turn of a finite
    step, on the host.
    """
    import jax.numpy as jnp

    psi = jnp.asarray(psi)
    omega = jnp.asarray(omega)
    npwx = psi.shape[-1] // 2
    up, down = psi[..., :npwx], psi[..., npwx:]
    wx, wy, wz = omega[0], omega[1], omega[2]
    half = -0.5j
    return jnp.concatenate([
        up + half * (wz * up + (wx - 1j * wy) * down),
        down + half * ((wx + 1j * wy) * up - wz * down),
    ], axis=-1)


class OrientationStepper:
    """The rotation step of Route C, one call per iteration.

    ``gradient`` is ``dE/dw``, minus :attr:`SCFResult.orientation_torque`, in Ry
    per radian; :meth:`propose` returns the rotation vector to turn the input by,
    or ``None``. :attr:`total` is the rotation applied so far.
    """

    def __init__(self, trust: float = 0.1, first_step: float = 0.05,
                 start: float = 1.0e-5):
        self.trust = float(trust)
        self.first_step = float(first_step)
        self.start = float(start)
        self.inverse_hessian = None
        self.previous_gradient = None
        self.previous_step = None
        self.total = np.eye(3)
        self.steps = 0
        #: The current bound on a step, radians; ``trust`` is its ceiling.
        self.radius = self.trust

    def propose(self, gradient, accuracy: float):
        gradient = np.asarray(gradient, dtype=float).reshape(3)
        if accuracy > self.start:
            # The last step is kept across the iterations the density needs to
            # recover from it: the torque once it has recovered, against the
            # torque before the step, is the secant with the density relaxed,
            # which is the curvature that matters, and the trust update needs
            # the same pair.
            return None
        if self.previous_step is not None:
            if np.linalg.norm(gradient) >= np.linalg.norm(self.previous_gradient):
                self.radius = max(self.radius / 2.0, self.trust / 64.0)
            else:
                self.radius = min(self.radius * 1.5, self.trust)
            s = self.previous_step
            y = gradient - self.previous_gradient
            sy = float(s @ y)
            if sy > 1.0e-12 * float(np.linalg.norm(s) * np.linalg.norm(y)) and sy > 0.0:
                if self.inverse_hessian is None:
                    self.inverse_hessian = (sy / float(y @ y)) * np.eye(3)
                rho = 1.0 / sy
                identity = np.eye(3)
                left = identity - rho * np.outer(s, y)
                self.inverse_hessian = (left @ self.inverse_hessian @ left.T
                                        + rho * np.outer(s, s))
        norm = float(np.linalg.norm(gradient))
        if norm == 0.0:
            return None
        if self.inverse_hessian is None:
            step = -self.first_step * gradient / norm
        else:
            step = -self.inverse_hessian @ gradient
        length = float(np.linalg.norm(step))
        if length > self.radius:
            step *= self.radius / length
        self.previous_gradient = gradient
        self.previous_step = step
        self.total = rotation_matrix(step) @ self.total
        self.steps += 1
        return step
