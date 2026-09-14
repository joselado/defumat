"""Whether ``<k+G|beta>`` is held for every k-point or rebuilt one at a time.

``vkb`` is ``(nk, npwx, nkb)`` and resident for the whole run -- 13.96 GB on the
45-atom NiBr2 slab against a 12.10 GB wavefunction set. QE holds one k-point's
(``init_us_2`` inside ``c_bands.f90``'s ``k_loop``) and ``rebuild`` is that.

**The trap this file is written against is the same one the wavefunction store
had**: every correctness assertion here would pass byte for byte if the dial did
nothing at all, because both routes evaluate the same expression. So two tests
check that the branch *executed* -- that a lazy set holds no whole-k array, and
that the dtype read does not quietly materialise one.
"""

import warnings

import jax
import numpy as np
import pytest

from defumat.batching import PROJECTOR_STORES, resolve_projectors
from defumat.calculator import Calculator

pytestmark = pytest.mark.unit

CELL = "tests/data/qe/h-atom-lsda.in"


def _calculation(which):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_file(
            CELL, pseudo_dir="tests/data/pseudo", announce=False)
        return calculator.calculation.__class__(
            calculator.system, calculator.pseudos, projectors=which,
        )


# --------------------------------------------------------------------------
# the dial
# --------------------------------------------------------------------------

def test_the_default_is_what_every_validated_number_was_measured_with():
    assert resolve_projectors() == "store"
    assert resolve_projectors("default") == "store"
    assert set(PROJECTOR_STORES) == {"store", "rebuild"}


def test_an_explicit_setting_beats_the_environment(monkeypatch):
    monkeypatch.setenv("DEFUMAT_PROJECTORS", "rebuild")
    assert resolve_projectors() == "rebuild"
    assert resolve_projectors("store") == "store"


def test_a_storage_that_is_not_one_is_refused():
    with pytest.raises(ValueError, match="projectors must be one of"):
        resolve_projectors("lazy-ish")


def test_an_unreadable_environment_variable_warns_and_falls_back(monkeypatch):
    monkeypatch.setenv("DEFUMAT_PROJECTORS", "rebiuld")
    with pytest.warns(RuntimeWarning, match="DEFUMAT_PROJECTORS"):
        assert resolve_projectors() == "store"


# --------------------------------------------------------------------------
# that the branch ran -- without these the file tests nothing
# --------------------------------------------------------------------------

def test_a_lazy_set_holds_no_whole_k_array():
    """The point of the dial, asserted on the object rather than on a number."""
    lazy = _calculation("rebuild").projectors
    stored = _calculation("store").projectors
    assert lazy.is_lazy and lazy.stored is None
    assert not stored.is_lazy and stored.stored is not None
    # And the core it keeps instead is the smaller array: one column per
    # species channel, where vkb has one per atom channel.
    assert lazy.core.columns.shape[-1] <= stored.stored.shape[-1]


def test_reading_the_dtype_does_not_materialise_the_array():
    """``projectors.vkb.dtype`` would build the whole array to read five bytes.

    That is the silent version of this feature not working: correct, and with
    the array it exists to avoid back in memory. The dtype comes off the core.
    """
    lazy = _calculation("rebuild").projectors
    built = []
    real = type(lazy).vkb.fget
    monkey = property(lambda self: (built.append(1), real(self))[1])
    try:
        type(lazy).vkb = monkey
        assert lazy.dtype == _calculation("store").projectors.stored.dtype
        assert built == [], "reading .dtype materialised the whole-k vkb"
    finally:
        type(lazy).vkb = property(real)


# --------------------------------------------------------------------------
# and that it changes no number
# --------------------------------------------------------------------------

def test_the_two_routes_give_the_same_projectors():
    """Same expression, same order, so this is equality and not a tolerance."""
    lazy, stored = _calculation("rebuild"), _calculation("store")
    assert np.array_equal(np.asarray(lazy.projectors.vkb),
                          np.asarray(stored.projectors.stored))
    for ik in range(stored.projectors.nk):
        assert np.array_equal(np.asarray(lazy.projectors.at_k(ik)),
                              np.asarray(stored.projectors.at_k(ik)))


def test_at_k_survives_a_traced_index():
    """``map_k`` walks the axis with ``jnp.arange``, so ``ik`` is a tracer."""
    lazy = _calculation("rebuild").projectors
    gathered = jax.jit(lambda i: lazy.at_k(i))(0)
    assert np.array_equal(np.asarray(gathered), np.asarray(lazy.at_k(0)))


def test_the_dial_reaches_the_calculation_through_the_facade():
    """It did not, and the verbose line is what caught it.

    ``projectors`` is a ``SETUP_OPTIONS`` member, so a per-call value is
    adopted and the ``Calculation`` rebuilt -- but the rebuild read every other
    setup option out of ``defaults`` and not this one, so ``get_scf(projectors=
    'rebuild')`` ran happily and stored the whole array anyway. A dial that
    does not reach the code is indistinguishable from a dial that does not
    help, which is the shape ``DEFUMAT_BAND_BATCH`` already failed in once.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_file(
            CELL, pseudo_dir="tests/data/pseudo", announce=False)
        calculator.get_scf(projectors="rebuild", max_iterations=1)
        assert calculator.calculation.projector_storage == "rebuild"
        assert calculator.calculation.projectors.is_lazy


def _scf(path, which, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_file(
            path, pseudo_dir="tests/data/pseudo", announce=False)
        return calculator.get_scf(projectors=which, **kw)


@pytest.mark.parametrize("path", [
    CELL,
    # **The noncollinear branch is a separate set of call sites**, not the same
    # ones with a bigger array: `noncollinear.py` reaches the projectors in six
    # more places than the scalar operator does -- the two `deeq_nc` diagonals,
    # the two dense-matrix builders, and a spiral's `stack([at_k(up),
    # at_k(down)])`, which is the only caller that wants two k-rows for one
    # physical k-point. A collinear pass says nothing about any of them.
    pytest.param("tests/data/qe/h2-texture-120.in", marks=pytest.mark.slow),
])
def test_the_dial_is_not_visible_in_any_scf_number(path):
    """Bit for bit: the dial changes what is stored, never what is computed.

    Not a tolerance. Both routes evaluate the same expression in the same
    order -- `_apply_phases` with an ellipsis where the k axis was -- so a
    difference in the last digit would mean one of them is not that expression.
    """
    on_store = _scf(path, "store", conv_thr=1e-10, max_iterations=4)
    on_rebuild = _scf(path, "rebuild", conv_thr=1e-10, max_iterations=4)
    assert on_rebuild.total_energy == on_store.total_energy
    assert on_rebuild.iterations == on_store.iterations
    assert np.array_equal(np.asarray(on_rebuild.eigenvalues),
                          np.asarray(on_store.eigenvalues))


# --------------------------------------------------------------------------
# the estimate has to describe the run that will happen
# --------------------------------------------------------------------------

def test_the_size_report_follows_the_dial():
    """`MEMORY-AUDIT` D11: sizing a rebuild run as a stored one overstates the
    floor by the largest single line in it.

    The floor line is the first thing anyone reads in a cluster log, and on the
    45-atom NiBr2 slab `vkb` is 13.00 GiB of a 42.60 GiB floor -- so a report
    that ignores the dial is wrong by more than any other term it carries.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_file(
            CELL, pseudo_dir="tests/data/pseudo", announce=False)
        stored = calculator.estimate(projectors="store")
        rebuilt = calculator.estimate(projectors="rebuild")

    assert stored.projectors == "store" and rebuilt.projectors == "rebuild"
    assert "projectors = store" in stored.report()
    assert "projectors = rebuild" in rebuilt.report()

    # The stored route carries one line per *atom* channel; the rebuilt route
    # carries the core it is built from plus one chunk, and no whole-k vkb.
    assert "projectors vkb (nk,npwx,nkb)" in stored.arrays
    assert "projectors vkb (nk,npwx,nkb)" not in rebuilt.arrays
    assert "projector core columns (nk,npwx,ncs)" in rebuilt.arrays


def test_the_rebuilt_floor_is_never_the_stored_one():
    """A dial the model does not see is a dial the model reports wrongly.

    On a many-k cell the difference is most of `vkb`; on a single-k one the
    rebuilt route is the *larger* of the two, because there is nothing to save
    and the chunk is the whole array. Both directions are the model working.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_file(
            CELL, pseudo_dir="tests/data/pseudo", announce=False)
        stored = calculator.estimate(projectors="store")
        rebuilt = calculator.estimate(projectors="rebuild")
    assert stored.total_bytes != rebuilt.total_bytes
