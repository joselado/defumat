"""The field response walked a k-chunk at a time against the whole-k route.

``GPU-MEMORY-NEXT.md`` item 2 (:mod:`defumat.response.chunked`). A streamed SCF
store -- a numpy array in host memory -- is handed to
:func:`~defumat.response.efield.dielectric_tensor` without being put on the
device, and the response then walks the k axis: the bare perturbations, the
Sternheimer solves and the response density per chunk, every sum over k added
over chunks and finished once. The claim is that this is the whole-k route's
answer to round-off, since only the order of the sums over k changes, and the
check is the same converged state through both routes.

The three cells each add something the others do not reach:

* **ultrasoft AlAs on its closed ``nosym`` grid** (8 k-points in chunks of 3, so
  the last chunk is padded with a repeat of its first row at zero weight): the
  chunk's projectors in ``adddvepsi_us``, ``int3`` handed to every chunk, and a
  polar crystal whose Born charge is a charge rather than a residue;
* **norm-conserving silicon on its wedge**: the directional symmetrisation acts
  on whole sums, after the walk;
* **PAW silicon on its wedge**: the ``becsum`` response, which the one-centre
  potential is built from and which is symmetrised as a polar vector on the
  whole sum.

The Born charges are walked too: the force's split
(:mod:`defumat.forces.chunked`) with one ``jvp`` through each pass. **AlAs is the
cell that discriminates there**, being polar and ultrasoft: the term coupling a
chunk's projector occupations to the whole-cell energy, ``g_b . db_c/dx``, is
zero for a norm-conserving dataset and only a symmetric residue on silicon, and
dropping it from the pull-back moves AlAs's ``Z*`` by 39 where the two routes
agree to 3.5e-13.
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

#: ``(case, k_batch)``: a chunk that does not divide the k-set wherever the
#: cell allows it, so the padded row is exercised.
CASES = [
    ("alas-epsilon-us-unshifted-nosym", 3),   # 8 k-points: chunks 3, 3, 2 + 1 pad
    ("si-epsilon", 4),                         # 10 k-points: 4, 4, 2 + 2 pad
    ("si-epsilon-paw", 3),
    # The wedge of the first cell, 3 k-points in chunks 2, 1 + 1 pad: the one
    # case here where the full-zone shift of the field's response is not zero
    # and is not cancelled by symmetry (a polar crystal, an augmented dataset),
    # so the step between the Born walks is exercised where it can be wrong.
    ("alas-epsilon-us-unshifted", 2),
]

#: Measured on the four cells: the dielectric constant 5e-14, 2.7e-15, 5.3e-15
#: and 5.7e-14 apart; the Born charges 3.5e-13, 8.8e-15, 4.4e-15 and 1.6e-13;
#: the induced density 8e-15, 0 and 1.2e-15 on the first three.
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


def _response(calculation, result, wavefunctions, born_charges=True):
    # The CG held at a fixed threshold, as it was before 2026-10-03: at the
    # scheduled default the two routes still stop at the same pass, and agree to
    # 2.4e-11 on an epsilon of 40.3 (6e-13 relative, AlAs on D22), which is a
    # band's CG stopping one step apart near a looser threshold rather than
    # anything the chunking does.
    return dielectric_tensor(
        calculation, wavefunctions, result.eigenvalues, result.density,
        result.becsum, born_charges=born_charges, threshold=1.0e-12,
    )


@pytest.mark.parametrize("case, k_batch", CASES)
def test_the_streamed_response_is_the_whole_k_response(case, k_batch):
    calculation, result = _converged(case, k_batch)
    whole = _response(calculation, result, result.wavefunctions)
    streamed = _response(calculation, result, np.asarray(result.wavefunctions))

    np.testing.assert_allclose(streamed.epsilon, whole.epsilon, rtol=0,
                               atol=TOLERANCE)
    np.testing.assert_allclose(streamed.born_charges, whole.born_charges,
                               rtol=0, atol=TOLERANCE)
    np.testing.assert_allclose(streamed.induced_density, whole.induced_density,
                               rtol=0, atol=TOLERANCE)
    assert len(streamed.history) == len(whole.history)
    assert streamed.converged and whole.converged
    assert streamed.average_iterations == whole.average_iterations


def test_a_second_streamed_call_compiles_nothing():
    """The walks are compiled once per structure, not once per chunk or call.

    Both halves, the dielectric loop and the Born charges' split: measured, 108
    programs on a fresh process's first call and none on the second, and 137 on
    the second when ``jax.clear_caches()`` comes between, which is the check that
    this counter fires.

    ``CLAUDE.md``'s check for the eager-closure trap: a second call, counted on
    the ``jax`` logger. Building each chunk's solve afresh through
    :func:`~defumat.eager.compiled` traced it every time instead -- 0.11 s per
    chunk on ultrasoft silicon, a quarter of an hour on a 216-point mesh.
    """
    calculation, result = _converged(*CASES[0])
    store = np.asarray(result.wavefunctions)
    _response(calculation, result, store)

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
            _response(calculation, result, store)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
    assert compiles == []
