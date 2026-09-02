"""``E(q)`` accumulated from ``dE/dq`` instead of differenced from ``E``.

The quantities themselves are P21's and are validated in
``tests/regression/test_spiral_relaxation.py``; what is left for a unit test is
the accumulation, which is a line integral and is exercised here against cases
whose answer is written down. Two of them, because the two things that can go
wrong are different:

* **A curved path with a known primitive.** A cosine ``E(q)`` -- the shape a
  frozen-magnon curve actually has -- integrated from its own derivative, where
  what is being tested is the trapezoid rule and the reference point.
* **A constant gradient along a bent path.** The trapezoid rule is *exact* for a
  constant integrand, so the answer is ``c . (q_n - q_0)`` whatever route the
  path takes between them, and any metric wrongly inserted into the contraction
  shows up immediately. A scan along a zone boundary bends, so this is not a
  hypothetical.
"""

import numpy as np
import pytest

from defumat.workflows.spiral import SpiralScan

pytestmark = pytest.mark.unit


def _scan(q, energies, gradients):
    return SpiralScan(
        wavevectors=np.asarray(q, dtype=float),
        energies=np.asarray(energies, dtype=float),
        moments=np.zeros((len(q), 3)),
        converged=tuple([True] * len(q)),
        gradients=None if gradients is None else np.asarray(gradients, dtype=float),
    )


def test_the_integral_of_a_cosine_is_the_cosine():
    """``E = -A cos(2 pi q)`` recovered from ``dE/dq = 2 pi A sin(2 pi q)``.

    The tolerance is the trapezoid rule's own ``h^2`` and nothing else, so it is
    checked to shrink fourfold under a halved step rather than merely to be
    small -- a reference-point error would pass a loose bound at both sizes.
    """
    amplitude = 2.0e-3
    errors = []
    for n in (13, 25):
        q = np.zeros((n, 3))
        q[:, 2] = np.linspace(0.0, 0.5, n)
        energies = -amplitude * np.cos(2.0 * np.pi * q[:, 2])
        gradients = np.zeros((n, 3))
        gradients[:, 2] = 2.0 * np.pi * amplitude * np.sin(2.0 * np.pi * q[:, 2])
        scan = _scan(q, energies, gradients)
        assert scan.integrated[0] == 0.0
        errors.append(np.abs(scan.integrated - scan.relative).max())

    assert errors[0] < 1.0e-2 * abs(scan.relative[-1])
    assert errors[1] == pytest.approx(errors[0] / 4.0, rel=0.1)


def test_a_constant_gradient_integrates_along_a_bent_path():
    """The line integral is path independent, and the trapezoid rule is exact.

    A constant ``dE/dq = c`` is the gradient of ``E = c . q``, so the answer at
    every point is ``c . (q - q_0)`` however the path bends between them. This
    is the check that the contraction is with the *step* rather than with a
    single varying component, and that no metric has been inserted: both
    factors are in lattice coordinates, where none is needed.
    """
    q = np.array([[0.0, 0.0, 0.0],
                  [0.1, 0.0, 0.0],
                  [0.1, 0.2, 0.0],
                  [0.1, 0.2, 0.5],
                  [0.0, 0.2, 0.5]], dtype=float)
    c = np.array([0.3, -0.7, 1.1])
    scan = _scan(q, np.zeros(len(q)), np.tile(c, (len(q), 1)))
    expected = 1.0e3 * (q - q[0]) @ c
    assert scan.integrated == pytest.approx(expected, abs=1e-12)


def test_a_scan_without_gradients_says_so():
    """The property refuses rather than returning a curve built from nothing."""
    q = np.zeros((3, 3))
    q[:, 2] = [0.0, 0.25, 0.5]
    scan = _scan(q, [-1.0, -1.1, -1.2], None)
    with pytest.raises(ValueError, match="gradients=True"):
        scan.integrated
