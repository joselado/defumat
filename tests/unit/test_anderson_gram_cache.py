"""The Anderson mixer keeps its Gram matrix between calls, and no number moves.

``OPEN.md`` M3: ``AndersonMixer.mix`` rebuilt ``r_i . r_j`` over the whole
history on every call, ``n^2 + n`` host dots of which ``n`` were new. It now
keeps the matrix and the norms beside the history, adds one row and column per
call and drops the leading one when the history rolls. The claim is that this
changes the cost and nothing else, so everything here is asserted with ``==``:

* the kept matrix is, at every step, the one the old expression rebuilds from
  the current history, entry for entry, through rolls, a change of
  ``exclude``, a trimmed solve, every reset and a checkpoint;
* the mixed density is the one a mixer with the cache switched off returns;
* an entry is carried rather than recomputed, which is the one assertion a
  mixer that rebuilt the matrix and then stored it would fail.

**The densities have small-integer entries, and that is what makes ``==``
honest.** numpy's dot goes to the BLAS (MKL on the workstation), which does
not promise the same last bit for one dot of two vectors held at two different
addresses; the cache computes a pair once, on the fitted copy made in the call
it arrived in, and the rebuild here recomputes it on a fresh copy. With integer
entries every dot is an integer far below ``2**24``, so it has one value in any
summation order, in float32 as in float64, and the comparison is about which
vectors were dotted rather than about how the BLAS sums them.
"""

import numpy as np
import pytest

from defumat.scf.checkpoint import load_mixer, save_mixer
from defumat.scf.mixing import AndersonMixer, get_mixer

pytestmark = pytest.mark.unit

SIZE = 40
#: In the middle of the packed vector, where the driver's ``_mix`` puts
#: ``becsum`` (after the density, before ``ns`` and ``tau``), so the fitted
#: part is two segments rather than a prefix.
BLOCK = slice(24, 32)
HISTORY = 4


def _fitted(exclude):
    fitted = np.ones(SIZE, dtype=bool)
    if exclude is not None:
        fitted[exclude] = False
    return fitted


def _rebuilt(mixer, exclude):
    """The Gram matrix and the norms exactly as ``mix`` computed them before M3."""
    fit = [r[_fitted(exclude)] for r in mixer._residuals]
    gram = np.array([[float(a @ b) for b in fit] for a in fit])
    norms = np.array([float(np.sqrt(r @ r)) for r in fit])
    return gram, norms


def _densities(rng, dtype=np.float64):
    """An input density in -8..8 and an output one a residual in -8..8 away."""
    rho_in = rng.integers(-8, 9, SIZE).astype(dtype)
    return rho_in, rho_in + rng.integers(-8, 9, SIZE).astype(dtype)


def _uncached(**settings):
    """The reference: the same mixer, rebuilding its Gram matrix on every call."""
    mixer = AndersonMixer(**settings)
    mixer._cache_gram = False
    return mixer


def _assert_the_cache_is_the_rebuild(mixer, exclude):
    n = len(mixer._residuals)
    gram, norms = _rebuilt(mixer, exclude)
    assert mixer._gram.shape == (n, n) and mixer._norms.shape == (n,)
    np.testing.assert_array_equal(mixer._gram, gram)
    np.testing.assert_array_equal(mixer._norms, norms)
    np.testing.assert_array_equal(mixer._gram, mixer._gram.T)


def _advance(cached, reference, rng, steps, exclude=BLOCK, dtype=np.float64):
    """Mix ``steps`` densities through both mixers, checking the cache at each."""
    for _ in range(steps):
        rho_in, rho_out = _densities(rng, dtype)
        mixed = cached.mix(rho_in, rho_out, exclude=exclude)
        np.testing.assert_array_equal(
            mixed, reference.mix(rho_in, rho_out, exclude=exclude)
        )
        _assert_the_cache_is_the_rebuild(cached, exclude)
        assert reference._gram is None


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize(
    "condition_limit", [1.0e12, 1.0], ids=["full-fit", "trimmed-to-newest"]
)
def test_the_kept_matrix_is_the_rebuilt_one_as_the_history_rolls(condition_limit, dtype):
    """Eighteen steps through a four-deep history, so it turns over four times.

    ``exclude`` is switched off for three steps in the middle and back on, and
    each switch is a different inner product, which must rebuild the matrix
    rather than extend it. ``condition_limit = 1.0`` trims every solve to the
    newest entry, since no condition number is below one, and the matrix must
    still cover the whole history, because the trimming is per solve. float32
    is here because the norm is kept beside the matrix rather than read off its
    diagonal, and in float32 ``sqrt`` of the dot and of its float64 copy differ.
    """
    rng = np.random.default_rng(20260923)
    settings = dict(beta=0.3, history=HISTORY, condition_limit=condition_limit)
    cached, reference = AndersonMixer(**settings), _uncached(**settings)
    schedule = [BLOCK] * 8 + [None] * 3 + [BLOCK] * 7
    previous, previous_exclude = None, None
    for step, exclude in enumerate(schedule):
        rolls = len(cached._residuals) == HISTORY
        _advance(cached, reference, rng, 1, exclude=exclude, dtype=dtype)
        assert len(cached._residuals) == min(step + 1, HISTORY)
        if rolls and exclude == previous_exclude:
            # The oldest entry left and the rest moved up one, as they were.
            np.testing.assert_array_equal(cached._gram[:-1, :-1], previous[1:, 1:])
        previous, previous_exclude = cached._gram.copy(), exclude


def test_every_restart_of_the_history_restarts_the_matrix(tmp_path):
    """Each way the history is emptied or replaced, in one sequence.

    ``reset()`` itself, which the response mixer's wrapper also calls; a zero
    residual, which ``mix`` answers by resetting itself; a checkpoint, across
    which the matrix travels with the history it describes; and a checkpoint
    written before the matrix was kept, which restores the history alone and
    leaves the matrix to be rebuilt. After each, the next steps must agree with
    a reference that never stopped.
    """
    rng = np.random.default_rng(7)
    settings = dict(beta=0.3, history=HISTORY)
    cached, reference = AndersonMixer(**settings), _uncached(**settings)
    _advance(cached, reference, rng, 6)

    cached.reset()
    reference.reset()
    assert cached._gram is None and cached._norms is None and cached._fit_mask is None
    _advance(cached, reference, rng, 6)

    rho = rng.integers(-8, 9, SIZE).astype(np.float64)
    np.testing.assert_array_equal(
        cached.mix(rho, rho, exclude=BLOCK), reference.mix(rho, rho, exclude=BLOCK)
    )
    assert cached._residuals == [] and cached._gram is None
    _advance(cached, reference, rng, 6)

    path = tmp_path / "scf_mixer.npz"
    save_mixer(cached, path)
    cached = load_mixer(get_mixer("anderson", **settings), path)
    _assert_the_cache_is_the_rebuild(cached, BLOCK)
    _advance(cached, reference, rng, 6)

    save_mixer(cached, path)
    cached = load_mixer(get_mixer("anderson", **settings), path)
    cached._gram = cached._norms = cached._fit_mask = None
    _advance(cached, reference, rng, 6)


def test_an_entry_is_computed_once_and_then_carried():
    """The one assertion that separates keeping the matrix from rebuilding it.

    Everything above would also pass for a mixer that rebuilt the matrix on
    every call and stored the result, since the rebuild is what the kept matrix
    must equal. So one entry is marked, moved by one half, which no dot of two
    integer vectors can be, and one more call rolls the history: a kept matrix
    carries the mark up one row and one column, and a rebuilt one erases it.
    """
    rng = np.random.default_rng(11)
    mixer = AndersonMixer(beta=0.3, history=HISTORY)
    for _ in range(HISTORY):
        mixer.mix(*_densities(rng), exclude=BLOCK)
    true = mixer._gram[1, 2]
    mixer._gram[1, 2] = mixer._gram[2, 1] = true + 0.5
    mixer.mix(*_densities(rng), exclude=BLOCK)
    assert mixer._gram[0, 1] == mixer._gram[1, 0] == true + 0.5
    assert _rebuilt(mixer, BLOCK)[0][0, 1] == true
