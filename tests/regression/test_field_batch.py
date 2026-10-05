"""The three field directions solved as one CG loop against one at a time.

``DEFUMAT_FIELD_BATCH`` (:func:`~defumat.batching.resolve_field_batch`) makes
the field response ``vmap`` its Sternheimer CG over the three directions
(:meth:`~defumat.response.sternheimer.SternheimerSolver.solve_many`), the
default on a card, where it divides the launches and the host reads of the
loop condition by three. Each band of each direction keeps its own convergence
flag, so the claim is the serial answer to round-off and the same CG counts,
on both routes (whole k and walked).

AlAs is the cell that discriminates for the Born charges: polar and
ultrasoft, so their antisymmetric part is a charge rather than a residue, and
``int3`` reaches every direction's right-hand side.
"""

import logging
import warnings
from functools import lru_cache

import jax
import numpy as np
import pytest

from defumat import Calculator
from defumat.response.efield import dielectric_tensor

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = [
    ("alas-epsilon-us-unshifted-nosym", 3),   # walked: chunks of 3 // 3 = 1
    ("si-epsilon", 4),
]

#: Measured on D22's CPU (2026-10-05) through the facade: epsilon 1.1e-14 and
#: Z* 2.2e-15 apart on ultrasoft AlAs, 3.6e-15 and 2.1e-15 on silicon, every
#: CG count equal.
TOLERANCE = 1e-11


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


def _response(calculation, result, wavefunctions):
    return dielectric_tensor(
        calculation, wavefunctions, result.eigenvalues, result.density,
        result.becsum, born_charges=True, threshold=1.0e-12,
    )


@pytest.mark.parametrize("streamed", [False, True], ids=["whole", "walked"])
@pytest.mark.parametrize("case, k_batch", CASES)
def test_the_batched_directions_are_the_serial_ones(case, k_batch, streamed,
                                                    monkeypatch):
    calculation, result = _converged(case, k_batch)
    store = (np.asarray(result.wavefunctions) if streamed
             else result.wavefunctions)
    monkeypatch.setenv("DEFUMAT_FIELD_BATCH", "0")
    serial = _response(calculation, result, store)
    monkeypatch.setenv("DEFUMAT_FIELD_BATCH", "1")
    batched = _response(calculation, result, store)

    np.testing.assert_allclose(batched.epsilon, serial.epsilon, rtol=0,
                               atol=TOLERANCE)
    np.testing.assert_allclose(batched.born_charges, serial.born_charges,
                               rtol=0, atol=TOLERANCE)
    assert len(batched.history) == len(serial.history)
    assert batched.converged and serial.converged
    assert batched.average_iterations == serial.average_iterations


def test_a_second_whole_k_born_call_compiles_nothing(monkeypatch):
    """The whole-k route's ultrasoft Born charges compile nothing on a second call.

    Measured before the fix (2026-10-05, D22's CPU, through the facade): 24
    programs on every call, ``efield.ultrasoft_position``'s eager ``map_k`` (3)
    and two top-level derivatives in ``born_effective_charges`` with a ``map_k``
    inside, the ``jacfwd`` of the frozen polarization (3) and the ``jvp`` of
    the constraint sandwich, once per atom and direction (18). The walked route
    had none, which is why the card did not show it.
    """
    monkeypatch.setenv("DEFUMAT_FIELD_BATCH", "0")
    calculation, result = _converged(*CASES[0])
    _response(calculation, result, result.wavefunctions)

    compiles = []

    class Count(logging.Handler):
        def emit(self, record):
            if "ompiling" in record.getMessage():
                compiles.append(record.getMessage())

    handler = Count()
    logger = logging.getLogger("jax")
    logger.addHandler(handler)
    level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        with jax.log_compiles(True):
            _response(calculation, result, result.wavefunctions)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
    assert compiles == []
