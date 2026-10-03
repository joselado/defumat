"""The third derivatives walked a k-chunk at a time against the whole-k route.

``GPU-MEMORY-NEXT.md`` item 2 (:mod:`defumat.response.chunked_third`). The
same re-diagonalised states give the field response twice, once whole and once
walked with its stores in host memory, and the same tangents (the whole route's
strain or displacement response) are handed to the whole route's third
derivative and to the walked one, so what is compared is the third derivative
alone:

* **electrostriction's** ``d(eps)/d(strain)`` on norm-conserving silicon (a
  closed grid, 8 k-points in chunks of 3) and ultrasoft silicon (the occupied
  block ``ort`` and ``adddvepsi_us``'s tail in ``db``);
* the end-to-end :func:`~defumat.response.electrostriction.electrostriction`
  on a host store against the whole route on the device, where the two
  re-diagonalisations are different programs and agree only to their
  threshold;
* the end-to-end **Raman tensors** the same way, on norm-conserving,
  ultrasoft and PAW silicon: the walked displacement response (the Gamma
  phonon's stages without the assembly), ``ort`` rebuilt from the overlap
  derivatives and added after the projection, and the third derivative along
  the positions;
* the end-to-end **vibrational spectrum**, whose dynamical matrix is assembled
  from the walked displacement response the Raman tensors hand back, rather
  than solved again or put on the device whole.
"""

import dataclasses
import warnings
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat import Calculator
from defumat.response.chunked_third import walked_susceptibility_derivative
from defumat.response.efield import dielectric_tensor
from defumat.response.electrostriction import (
    VOIGT, electrostriction, field_blocks, refined_states,
    susceptibility_strain_derivative,
)
from defumat.response.nonlinear import raman_tensors
from defumat.response.spectra import vibrational_spectrum
from defumat.response.strain import strain_response, strain_tangent

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = [("si-electrostriction", 3), ("si-us-nosym", 3)]

TOLERANCE = 1e-10


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """``jax.clear_caches()`` between tests, for ``CLAUDE.md``'s reason."""
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _converged(case: str, k_batch: int):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_file(
            f"tests/data/qe/{case}.in", pseudo_dir="tests/data/pseudo",
            k_batch=k_batch, conv_thr=1e-12, announce=False,
        )
        return calculator.calculation, calculator.get_scf()


#: The CG held at a fixed threshold, as it was before 2026-10-03, so that the two
#: routes differ only by the order of their sums: at the scheduled default they
#: still stop at the same pass but a band's CG can stop a step apart near the looser
#: threshold, which on D22 read 1.5e-6 relative on the electrostriction and the Raman tensors and 3.5e-6 on a
#: vibrational spectrum's activities.
ROUTE = {"threshold": 1.0e-12}


def _strain_derivatives(case, k_batch):
    calculation, result = _converged(case, k_batch)
    eigenvalues, psi = refined_states(calculation, result)
    density = jnp.asarray(result.density)
    strain = strain_response(calculation, psi, eigenvalues, density, result.becsum,
                             **ROUTE)

    whole_field = dielectric_tensor(calculation, psi, eigenvalues, density,
                                    result.becsum, born_charges=False,
                                    keep_internals=True, **ROUTE)
    _, solver, _, b, u, stored = field_blocks(whole_field)
    whole = susceptibility_strain_derivative(calculation, solver, density, b, u,
                                             strain, stored=stored)

    walked_field = dielectric_tensor(calculation, np.asarray(psi), eigenvalues,
                                     density, result.becsum, born_charges=False,
                                     **ROUTE,
                                     keep_internals=True, streamed_internals=True)
    tangents = [(strain_tangent(k, l), strain.dpsi[k, l],
                 None if strain.ort is None else strain.ort[k, l],
                 strain.drho[k, l]) for (k, l) in VOIGT]
    columns = walked_susceptibility_derivative(
        calculation, walked_field.internals["field"], density, "strain",
        jnp.zeros((3, 3)), tangents)
    walked = np.zeros((3, 3, 3, 3))
    for (k, l), column in zip(VOIGT, columns):
        walked[:, :, k, l] = walked[:, :, l, k] = column
    return whole, walked


@pytest.mark.parametrize("case, k_batch", CASES)
def test_the_walked_strain_derivative_is_the_whole_k_one(case, k_batch):
    whole, walked = _strain_derivatives(case, k_batch)
    scale = float(np.abs(whole).max())
    assert scale > 1.0, "a tensor of zeros agreeing with another"
    assert float(np.abs(walked - whole).max()) < TOLERANCE * scale


def test_the_walked_electrostriction_is_the_whole_route():
    """End to end: the route is chosen by the store, and the two
    re-diagonalisations agree only to their threshold."""
    calculation, result = _converged(*CASES[0])
    whole = electrostriction(calculation, result, **ROUTE)
    hosted = dataclasses.replace(result, wavefunctions=np.asarray(result.wavefunctions))
    walked = electrostriction(calculation, hosted, **ROUTE)
    scale = float(np.abs(whole.depsilon_dstrain).max())
    assert float(np.abs(walked.depsilon_dstrain - whole.depsilon_dstrain).max()) < 1e-7 * scale
    np.testing.assert_allclose(walked.elastic.voigt, whole.elastic.voigt, rtol=0,
                               atol=1e-6)


@pytest.mark.parametrize("case, k_batch",
                         [("si-electrostriction", 3), ("si-us-nosym", 3),
                          ("si-paw-nosym", 3)])
def test_the_walked_raman_tensors_are_the_whole_route(case, k_batch):
    calculation, result = _converged(case, k_batch)
    whole = raman_tensors(calculation, result, **ROUTE)
    hosted = dataclasses.replace(result, wavefunctions=np.asarray(result.wavefunctions))
    walked = raman_tensors(calculation, hosted, **ROUTE)
    scale = float(np.abs(whole.raman).max())
    assert scale > 1e-2, "a tensor of zeros agreeing with another"
    assert float(np.abs(walked.raman - whole.raman).max()) < 1e-7 * scale
    assert walked.converged and whole.converged


def _multiplet_sums(frequencies, values, tolerance=1e-3):
    """``values`` summed over each degenerate group of frequencies (rule D4)."""
    order = np.argsort(frequencies)
    groups, sums = [], []
    for index in order:
        if groups and abs(frequencies[index] - groups[-1]) < tolerance:
            sums[-1] += values[index]
        else:
            groups.append(frequencies[index])
            sums.append(values[index])
    return np.array(groups), np.array(sums)


@pytest.mark.parametrize("case, k_batch", [("si-us-nosym", 3)])
def test_the_walked_vibrational_spectrum_is_the_whole_route(case, k_batch):
    calculation, result = _converged(case, k_batch)
    whole = vibrational_spectrum(calculation, result, **ROUTE)
    hosted = dataclasses.replace(result, wavefunctions=np.asarray(result.wavefunctions))
    walked = vibrational_spectrum(calculation, hosted, **ROUTE)
    np.testing.assert_allclose(np.sort(walked.frequencies),
                               np.sort(whole.frequencies), rtol=0, atol=1e-6)
    _, whole_sums = _multiplet_sums(whole.frequencies, whole.raman_activity)
    _, walked_sums = _multiplet_sums(walked.frequencies, walked.raman_activity)
    scale = float(np.abs(whole_sums).max())
    assert scale > 0.0
    assert float(np.abs(walked_sums - whole_sums).max()) < 1e-7 * scale
