"""The piezoelectric tensor walked a k-chunk at a time against the whole-k route.

``GPU-MEMORY-NEXT.md`` item 2. The differentiated route
(:func:`~defumat.response.piezo.clamped_ion_piezoelectric`) is one ``jvp`` of the
stress along each field response, and with the whole k axis on one tape it was
recorded at 31.5 GiB of temporaries on ultrasoft AlAs at 64 k-points
(``PERFORMANCE.md``). Walked
(:meth:`~defumat.response.chunked.StreamedField.piezoelectric`), it is the
streamed Born charges' split with ``at_strain`` where they have
``at_positions``. The claim is that it is the whole route's tensor to round-off,
and the check is the same re-diagonalised states handed to
:func:`~defumat.response.piezo._piezoelectric_from_states` once as a device array
(the whole route) and once as a host store (the walk), on cells chosen so each
term the augmented assembly carries is reached:

* **norm-conserving AlAs on its wedge**: no multipliers, no constraint term and
  no shift, only the symmetrisation of the field's response and of the tensor;
* **ultrasoft AlAs on its closed grid** in chunks of 3 over 8 k-points: the
  multipliers' tangent, ``add_for_charges`` in the strain coordinate, and a
  padded last chunk;
* **ultrasoft AlAs on its wedge**: the full-zone shift of the field's response,
  worth 1.05e-3 C/m^2 there (``piezo.clamped_ion_piezoelectric``), so the
  comparison is not blind to it -- which the falsifier below checks;
* **PAW AlAs on its wedge**: the one-centre energy in the global terms.
"""

import logging
import warnings
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat import Calculator
from defumat.response.electrostriction import refined_states
from defumat.response.piezo import _piezoelectric_from_states

pytestmark = [pytest.mark.regression, pytest.mark.slow]

#: ``(case, k_batch)``: a chunk that does not divide the k-set wherever it can.
CASES = [
    ("alas-raman-wedge", 3),
    ("alas-piezo-tiny", 3),           # 8 k-points: 3, 3, 2 + 1 pad
    ("alas-piezo-tiny-wedge", 2),
    ("alas-piezo-tiny-paw-wedge", 2),
]

#: In e/bohr^2, where the cells' e_14 is of order 1e-2 (0.8 C/m^2).
TOLERANCE = 1e-12


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """``jax.clear_caches()`` between tests, for ``CLAUDE.md``'s reason."""
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _refined(case: str, k_batch: int):
    """The ground state and its re-diagonalised states, on QE's ``k + G = 0``
    convention as ``test_piezoelectric.py`` runs it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_file(
            f"tests/data/qe/{case}.in", pseudo_dir="tests/data/pseudo",
            k_batch=k_batch, conv_thr=1e-12, announce=False,
            origin_tangent=False,
        )
        calculation = calculator.calculation
        result = calculator.get_scf()
        eigenvalues, psi = refined_states(calculation, result)
        return calculation, result, eigenvalues, psi


#: The CG held at a fixed threshold, as it was before 2026-10-03, so that the two
#: routes differ only by the order of their sums: at the scheduled default they
#: still stop at the same pass but a band's CG can stop a step apart near the looser
#: threshold, which on D22 read beyond the 1e-11 bounds below on both AlAs cells.
ROUTE = {"threshold": 1.0e-12}


def _tensor(case, k_batch, states, **options):
    calculation, result, eigenvalues, _ = _refined(case, k_batch)
    e, field = _piezoelectric_from_states(
        calculation, states, eigenvalues, jnp.asarray(result.density),
        result.becsum, **{**ROUTE, **options})
    return np.asarray(e), field


@pytest.mark.parametrize("case, k_batch", CASES)
def test_the_walked_tensor_is_the_whole_k_tensor(case, k_batch):
    _, _, _, psi = _refined(case, k_batch)
    whole, whole_field = _tensor(case, k_batch, psi)
    walked, walked_field = _tensor(case, k_batch, np.asarray(psi))
    assert "field" in walked_field.internals and "field" not in whole_field.internals

    np.testing.assert_allclose(walked, whole, rtol=0, atol=TOLERANCE)
    np.testing.assert_allclose(walked_field.epsilon, whole_field.epsilon,
                               rtol=0, atol=1e-11)
    # Not a tensor of zeros agreeing with another: zincblende's one component.
    assert abs(whole[0, 1, 2]) > 1e-3


def test_the_wedge_comparison_sees_the_full_zone_shift(monkeypatch):
    """The falsifier: with the walk's shift zeroed the wedge moves by its 1e-3.

    On a closed grid the shift is identically zero, so an agreement there says
    nothing about it; this is the case where it is the term that would be
    wrong, and the test that the comparison above can see it.
    """
    from defumat.response import chunked

    case, k_batch = "alas-piezo-tiny-wedge", 2
    _, _, _, psi = _refined(case, k_batch)
    whole, _ = _tensor(case, k_batch, psi)
    original = chunked._full_zone_shifts

    def unshifted(*args, **kwargs):
        shifts, becsum_shifts, rest = original(*args, **kwargs)
        return (jnp.zeros_like(shifts),
                [jax.tree_util.tree_map(jnp.zeros_like, b) for b in becsum_shifts],
                rest)

    monkeypatch.setattr(chunked, "_full_zone_shifts", unshifted)
    walked, _ = _tensor(case, k_batch, np.asarray(psi))
    assert np.abs(walked - whole).max() > 1e-6


def test_a_second_walked_call_compiles_nothing():
    """``CLAUDE.md``'s check for the eager-closure trap, on the walk's passes."""
    case, k_batch = "alas-piezo-tiny", 3
    _, _, _, psi = _refined(case, k_batch)
    store = np.asarray(psi)
    _tensor(case, k_batch, store)

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
            _tensor(case, k_batch, store)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
    assert compiles == [], compiles[:5]
