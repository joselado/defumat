"""Anderson's ``mix()`` reads its history through views and sums in place, and no bit moves.

``OPEN.md`` Part XXIII item 23. The fitted part of every history entry used to
be cut out through a boolean mask on every call, a whole copy per entry even
when nothing was excluded, and the combination ``total = total + c * d``
allocated a whole vector per term per sum. Both are gone where they can go
without changing a number: the fitted part is a view whenever it is one
contiguous block (nothing excluded, or ``becsum`` at either end of the packed
vector), and the two sums run in place in three buffers, term by term in the
order they always had.

Two claims, asserted separately:

* the mixed vector and the Gram matrix are the old expressions' to the last bit,
  on random (not integer) data, for every kind of ``exclude`` and for a float32
  history, so the view-against-copy dot and the in-place sums are both checked
  against what they replaced;
* the peak memory of one call at the default depth of eight is about four
  vectors, where the old code reached twelve (measured with ``tracemalloc``,
  which numpy reports its data buffers to). This is the half that fails on the
  old code; the first half passes on both, by design.
"""

import tracemalloc

import numpy as np
import pytest

from defumat.scf.mixing import AndersonMixer

pytestmark = pytest.mark.unit

SIZE = 100_000
HISTORY = 8
EXCLUDES = {
    "none": None,
    "tail": slice(80_000, SIZE),
    "head": slice(0, 10_000),
    "middle": slice(40_000, 50_000),
}


def _states(rng, dtype, calls):
    """A converging sequence of input and output densities, with exact zeros in it."""
    rho = rng.standard_normal(SIZE).astype(dtype)
    rho[::97] = 0.0
    for step in range(calls):
        residual = (rng.standard_normal(SIZE) * 0.5 ** step).astype(dtype)
        residual[::89] = 0.0
        yield rho, (rho + residual).astype(dtype)
        rho = (rho + 0.3 * residual).astype(dtype)


def _old_mix(mixer):
    """The combination exactly as ``mix`` wrote it before item 23, from the mixer's state.

    Called after ``mix`` has extended the history and the Gram matrix, so the
    coefficients are recomputed from the same cache by the same trimming rule,
    and then summed the old way: ``0.0 + c * d`` and so on, a new array per
    term, and the step added to a new array at the end.
    """
    gram, norms = mixer._gram, mixer._norms
    n = len(norms)
    keep = n
    while keep > 1:
        if np.linalg.cond(mixer._build_overlap(gram, norms, keep)) < mixer.condition_limit:
            break
        keep -= 1
    overlap = mixer._build_overlap(gram, norms, keep)
    used = slice(n - keep, n)
    rhs = np.zeros(keep + 1)
    rhs[keep] = 1.0
    coefficients = np.linalg.solve(overlap, rhs)[:keep] / norms[used]
    mixed_density, mixed_residual = 0.0, 0.0
    for c, d, r in zip(coefficients, mixer._densities[used], mixer._residuals[used]):
        mixed_density = mixed_density + c * d
        mixed_residual = mixed_residual + c * r
    return mixed_density + mixer.step(mixed_residual, mixed_density)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("name", list(EXCLUDES))
def test_the_mixed_vector_is_the_old_expression_bit_for_bit(name, dtype):
    """Twelve calls through an eight-deep history, compared with ``tobytes``.

    The Gram matrix is compared too, against the old rebuild over masked
    *copies*: the new code dots views where the fitted part is one block, and
    the claim that a view and a copy of the same numbers give the same dot is
    the BLAS's, so it is checked here on numbers that are not small integers.
    """
    exclude = EXCLUDES[name]
    mixer = AndersonMixer(beta=0.4, history=HISTORY)
    mask = np.ones(SIZE, dtype=bool)
    if exclude is not None:
        mask[exclude] = False
    for rho_in, rho_out in _states(np.random.default_rng(23), dtype, 12):
        mixed = mixer.mix(rho_in, rho_out, exclude=exclude)
        if len(mixer._residuals) == 1:
            continue
        expected = _old_mix(mixer)
        assert mixed.dtype == expected.dtype
        assert mixed.tobytes() == np.asarray(expected).tobytes()
        copies = [r[mask] for r in mixer._residuals]
        rebuilt = np.array([[float(a @ b) for b in copies] for a in copies])
        assert mixer._gram.tobytes() == rebuilt.tobytes()


@pytest.mark.parametrize("name", ["none", "tail"])
def test_one_call_peaks_at_four_vectors_not_twelve(name):
    """``tracemalloc``'s peak over one ``mix()`` at a full history, in vectors.

    What one call must allocate: the new residual it stores, the two running
    sums, one product buffer and the step, and the boolean mask (an eighth of
    a vector) its Gram cache is keyed on. Before item 23 the peak was 12.1
    vectors with nothing excluded and 10.5 with ``becsum`` as the tail: the
    eight fitted copies, live together, and the products of the combination.
    """
    exclude = EXCLUDES[name]
    mixer = AndersonMixer(beta=0.4, history=HISTORY)
    vector = SIZE * np.dtype(np.float64).itemsize
    peaks = []
    for rho_in, rho_out in _states(np.random.default_rng(5), np.float64, 11):
        tracemalloc.start()
        try:
            start = tracemalloc.get_traced_memory()[0]
            mixer.mix(rho_in, rho_out, exclude=exclude)
            peaks.append((tracemalloc.get_traced_memory()[1] - start) / vector)
        finally:
            tracemalloc.stop()
    assert len(mixer._residuals) == HISTORY
    assert max(peaks[-3:]) < 5.0, peaks
