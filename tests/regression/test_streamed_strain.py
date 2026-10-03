"""The strain response walked a k-chunk at a time against the whole-k route.

``GPU-MEMORY-NEXT.md`` item 2 (:mod:`defumat.response.chunked_strain`). The same
re-diagonalised states are handed to :func:`~defumat.response.strain.
strain_response` once as a device array (the whole route) and once as a host
store (the walk), and every field of the result is compared: the response
density with the volume's frozen-state part in it, the converged induced
potential, the first-order states, the eigenvalue response, and for an augmented
dataset the overlap derivatives, the occupied block ``ort`` and the moved halves
the consumers subtract. Cells:

* **norm-conserving silicon, closed grid** (8 k-points in chunks of 3, the last
  padded): the frozen-state response is the volume's alone;
* **norm-conserving silicon on its wedge**: the rank-2 average of the response;
* **ultrasoft and PAW silicon, closed grids**: the deforming augmentation
  charge, ``S'``, ``ort``, and for PAW the one-centre response in the loop.

The elastic constants walked (:func:`~defumat.forces.chunked.
chunked_gradient_tangent`) are held to the single pass on the norm-conserving
closed grid, the one regime :func:`~defumat.response.elastic.elastic_constants`
admits, each route on its own strain response.
"""

import warnings
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat import Calculator
from defumat.response.elastic import elastic_constants
from defumat.response.electrostriction import refined_states
from defumat.response.strain import strain_response

pytestmark = [pytest.mark.regression, pytest.mark.slow]

#: ``(case, k_batch)``: a chunk that does not divide the k-set wherever it can.
CASES = [
    ("si-electrostriction", 3),   # 8 k-points: 3, 3, 2 + 1 pad
    ("si-strain-wedge", 2),
    ("si-us-nosym", 3),
    ("si-paw-nosym", 3),
]

TOLERANCE = 1e-11


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """``jax.clear_caches()`` between tests, for ``CLAUDE.md``'s reason."""
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _refined(case: str, k_batch: int):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_file(
            f"tests/data/qe/{case}.in", pseudo_dir="tests/data/pseudo",
            k_batch=k_batch, conv_thr=1e-12, announce=False,
        )
        calculation = calculator.calculation
        result = calculator.get_scf()
        eigenvalues, psi = refined_states(calculation, result)
        return calculation, result, eigenvalues, psi


@lru_cache(maxsize=2)
def _response(case, k_batch, walked: bool):
    calculation, result, eigenvalues, psi = _refined(case, k_batch)
    return strain_response(calculation, np.asarray(psi) if walked else psi,
                           eigenvalues, jnp.asarray(result.density),
                           result.becsum)


def _close(a, b, what):
    a, b = np.asarray(a), np.asarray(b)
    scale = max(float(np.abs(b).max()), 1.0)
    assert np.abs(a - b).max() < TOLERANCE * scale, (
        f"{what}: {np.abs(a - b).max():.3e} on a scale of {scale:.3e}")


@pytest.mark.parametrize("case, k_batch", CASES)
def test_the_walked_strain_response_is_the_whole_k_response(case, k_batch):
    whole = _response(case, k_batch, False)
    walked = _response(case, k_batch, True)
    assert isinstance(walked.dpsi[0, 0], np.ndarray)

    _close(walked.drho, whole.drho, "drho")
    _close(walked.dvscf, whole.dvscf, "dvscf")
    _close(walked.deigenvalues, whole.deigenvalues, "deigenvalues")
    for a in range(3):
        for b in range(3):
            _close(walked.dpsi[a, b], whole.dpsi[a, b], f"dpsi[{a}, {b}]")
    assert (walked.ort is None) == (whole.ort is None)
    if whole.ort is not None:
        for a in range(3):
            for b in range(3):
                _close(walked.ort[a, b], whole.ort[a, b], f"ort[{a}, {b}]")
                _close(walked.overlap_derivatives[a, b],
                       whole.overlap_derivatives[a, b], f"S'[{a}, {b}]")
    _close(walked.moved_drho, whole.moved_drho, "moved_drho")
    assert len(walked.history) == len(whole.history)
    assert walked.converged and whole.converged
    # The same work, not the same count: which side of the CG threshold one
    # solve lands on is rounding, and the two routes round differently. On
    # PAW silicon master's two routes agreed at the default radial chunk and
    # read 20.233 / 20.2 and 20.283 / 20.217 at 700 and 2000 values
    # (2026-10-03), one or two iterations over thirty solves.
    assert abs(walked.average_iterations - whole.average_iterations) <= (
        0.01 * whole.average_iterations)
    # Not a response of zeros agreeing with another.
    assert float(np.abs(np.asarray(whole.drho)).max()) > 1e-2


def test_the_walked_elastic_constants_are_the_single_pass():
    case, k_batch = CASES[0]
    calculation, _, eigenvalues, psi = _refined(case, k_batch)
    whole = elastic_constants(calculation, psi, eigenvalues,
                              None, _response(case, k_batch, False))
    walked = elastic_constants(calculation, np.asarray(psi), eigenvalues,
                               None, _response(case, k_batch, True))
    # In GPa, where C_11 is about 200.
    np.testing.assert_allclose(walked.voigt, whole.voigt, rtol=0, atol=1e-8)
    assert whole.voigt[0, 0] > 100.0
