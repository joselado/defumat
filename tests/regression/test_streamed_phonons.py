"""The dynamical matrix walked a k-chunk at a time against the whole-k route.

``GPU-MEMORY-NEXT.md`` item 2 (:mod:`defumat.response.chunked_phonon`). A
streamed SCF store -- a numpy array in host memory -- is handed to
:func:`~defumat.response.phonon.dynamical_matrix` without being put on the
device, and the bare perturbations, the overlap's derivatives, the solves, the
multipliers and the assembly then walk the k axis. The claim is that this is the
whole-k route's matrix to round-off, and the check is the same converged state
through both routes, on cells chosen so that each regime the whole route accepts
is reached by one of them:

* **norm-conserving silicon on its wedge** (10 k-points in chunks of 4, the last
  padded with repeats at zero weight): the symmetrisation of the response and of
  the assembled matrix act on whole sums, after the walks;
* **two-atom aluminium, a metal**: the Fermi level's shift with ``ldos`` walked
  once, ``ef_shift_wfc`` on the host store, and P28's split between the frozen
  Hessian at ``wg`` and the state response at ``wk`` -- which needs the padded
  rows' ``wk`` to be zero as well;
* **ultrasoft AlAs on its closed grid**, the arsenic atom's three
  displacements: the polar cell, where a chunk's projector occupations couple to
  the whole-cell energy, and where the multipliers' tangent is the term the
  padding broke -- read back by global row index, the padded repeat got its
  original's ``dLambda``, and since the constraint is weighted by the multipliers
  rather than by the occupations that was 1.3e-2 on force constants of 0.22;
* **PAW silicon on its wedge**: ``dbecsum``, the one-centre loop, and the
  becsum correction the assembly carries apart from the density's;
* **half-sphere storage**, whole and with one atom displaced: the ``G = 0``
  halving in every separable term, and the rows of a subset rather than of the
  cell.
"""

import logging
import warnings
from functools import lru_cache

import jax
import numpy as np
import pytest

from defumat import Calculator
from defumat.response.phonon import dynamical_matrix

pytestmark = [pytest.mark.regression, pytest.mark.slow]

#: ``(case, k_batch, atoms)``.
CASES = [
    ("si-epsilon", 4, None),                            # 10 k: 4, 4, 2 + 2 pad
    ("al2-metal", 3, None),                             # 32 k: ... 2 + 1 pad
    ("alas-epsilon-us-unshifted-nosym", 3, (1,)),       # 8 k: 3, 3, 2 + 1 pad
    ("si-epsilon-paw", 3, None),                        # 10 k: 3, 3, 3, 1 + 2 pad
    ("gamma", 1, None),
    ("gamma", 1, (0,)),
]

#: Measured, the matrix: 5.1e-15 on silicon's force constants of 0.28, 4.3e-15
#: on aluminium's 0.048, 2.3e-15 on AlAs's 0.22, and 2.2e-16 on the half sphere;
#: the induced densities 1e-14 or below.
TOLERANCE = 1e-11

#: Two-atom silicon at Gamma, stored on the half sphere (``K_POINTS gamma``).
GAMMA = """&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.20, nat = 2, ntyp = 1, ecutwfc = 12.0,
  nosym = .true.
/
&electrons
  conv_thr = 1.0d-12
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS gamma
"""


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """``jax.clear_caches()`` between tests, for ``CLAUDE.md``'s reason."""
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _converged(case: str, k_batch: int):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if case == "gamma":
            calculator = Calculator.from_text(
                GAMMA, "tests/data/pseudo", k_batch=k_batch, announce=False)
            assert calculator.calculation.gamma_only
        else:
            calculator = Calculator.from_file(
                f"tests/data/qe/{case}.in", pseudo_dir="tests/data/pseudo",
                k_batch=k_batch, conv_thr=1e-12, announce=False,
            )
        return calculator.calculation, calculator.get_scf()


def _phonons(calculation, result, wavefunctions, atoms, on_row=None):
    return dynamical_matrix(
        calculation, wavefunctions, result.eigenvalues, result.density,
        result.becsum, atoms=atoms, on_row=on_row,
    )


@pytest.mark.parametrize("case, k_batch, atoms", CASES,
                         ids=[f"{c}-{a}" for c, _, a in CASES])
def test_the_streamed_phonon_is_the_whole_k_phonon(case, k_batch, atoms):
    calculation, result = _converged(case, k_batch)
    rows = {"whole": [], "streamed": []}
    whole = _phonons(calculation, result, result.wavefunctions, atoms,
                     lambda a, c, r: rows["whole"].append((a, c)))
    streamed = _phonons(calculation, result, np.asarray(result.wavefunctions),
                        atoms, lambda a, c, r: rows["streamed"].append((a, c)))

    np.testing.assert_allclose(streamed.matrix, whole.matrix, rtol=0,
                               atol=TOLERANCE)
    np.testing.assert_allclose(streamed.induced_density, whole.induced_density,
                               rtol=0, atol=TOLERANCE)
    assert len(streamed.history) == len(whole.history)
    assert streamed.converged and whole.converged
    assert streamed.average_iterations == whole.average_iterations
    # ``on_row`` still fires a row at a time, in the whole route's order.
    assert rows["streamed"] == rows["whole"]


def test_a_second_streamed_call_compiles_nothing():
    """The walks are compiled once per structure, not per chunk, mode or call.

    The ultrasoft cell, so that every pass is reached -- the overlap's
    derivatives, the multipliers and the three assembly walks -- and the unit
    tangent of each displacement is an argument rather than a constant, which
    a second call with the same three tangents would not catch but a fourth
    program per pass would show here as a compilation on the first atom of a
    later call. ``CLAUDE.md``'s check for the eager-closure trap.
    """
    calculation, result = _converged("alas-epsilon-us-unshifted-nosym", 3)
    store = np.asarray(result.wavefunctions)
    _phonons(calculation, result, store, (1,))

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
            _phonons(calculation, result, store, (0,))
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
    assert compiles == []
