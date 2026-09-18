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


def test_the_dial_survives_into_a_derived_calculator():
    """It did not: adopting it wrote a key the constructor then rejected.

    ``projectors`` is a ``SETUP_OPTIONS`` member and was in no set
    ``Calculator.__init__`` accepts, so ``get_scf(projectors='rebuild')``
    succeeded and left ``defaults['projectors']``, and the next
    ``with_positions``, ``with_cell``, ``with_spin``, ``with_moments`` or
    ``relaxed`` raised ``TypeError`` out of ``_derived`` -- worst in
    ``relaxed()``, where the whole relaxation is paid for first and the
    calculator can then never produce a derived one at all.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_file(
            CELL, pseudo_dir="tests/data/pseudo", announce=False)
        calculator.get_scf(projectors="rebuild", max_iterations=1)
        derived = calculator.with_positions(calculator.system.structure.positions)
    assert derived.defaults["projectors"] == "rebuild"
    assert derived.calculation.projector_storage == "rebuild"


def test_the_dial_is_not_part_of_the_scf_cache_key():
    """It says which ``Calculation`` exists, not which run was made over it.

    The first repair put ``projectors`` out of ``_defaults_for``'s forwarding,
    which is right -- ``run_pdos`` has a parameter of that name meaning
    something else -- and that alone made a plain ``get_scf()`` after a
    ``get_scf(projectors='rebuild')`` miss its own cache and run the whole SCF
    a second time: the first call's ``options`` carried the name and the second
    call had nothing to supply it. Stripping it from the key is the fix rather
    than restoring the forwarding, because ``_adopt`` already drops the cache
    when a setup option changes.
    """
    import defumat.calculator as facade

    calls = {"n": 0}
    real = facade.run_scf

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_file(
            CELL, pseudo_dir="tests/data/pseudo", announce=False)
        facade.run_scf = counting
        try:
            calculator.get_scf(projectors="rebuild", max_iterations=1)
            calculator.get_scf(max_iterations=1)
        finally:
            facade.run_scf = real
    assert calls["n"] == 1


def test_the_dial_can_be_set_for_a_calculator_s_whole_life():
    """The other face of the same hole: the constructor refused it."""
    calculator = Calculator.from_file(
        CELL, pseudo_dir="tests/data/pseudo", announce=False, projectors="rebuild")
    assert calculator.calculation.projector_storage == "rebuild"


def test_the_dial_does_not_reach_the_pdos_projector_scheme():
    """Same word, different meaning, and forwarding it crossed the two.

    ``run_pdos``'s ``projectors`` is ``'ortho-atomic'`` against ``'atomic'``
    -- a physics choice -- where this one is where ``vkb`` lives. With the
    memory dial in ``defaults`` and ``_defaults_for`` filtering by named
    parameter alone, ``get_pdos`` was handed ``projectors='rebuild'`` as the
    scheme, and ``atomic_projections`` raised ``unknown projector set``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_file(
            CELL, pseudo_dir="tests/data/pseudo", announce=False)
        calculator.get_scf(projectors="rebuild", max_iterations=1)
    from defumat.workflows.pdos import run_pdos
    forwarded = calculator._defaults_for(run_pdos, {})
    assert "projectors" not in forwarded


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
    # carries one rebuilt chunk instead, and no whole-k vkb.
    assert "projectors vkb (nk,npwx,nkb)" in stored.arrays
    assert "projectors vkb (nk,npwx,nkb)" not in rebuilt.arrays
    assert "projectors rebuilt, one chunk (npwx,nkb)" in rebuilt.arrays


def test_the_core_is_charged_to_both_routes():
    """`driver.py` builds `projector_core` before the dial is even resolved.

    This has now been got wrong twice, in opposite directions, from the same
    picture -- that the core arrives *with* the rebuilt route. A13 first
    counted it as a new cost of `rebuild`; the model then counted it as a cost
    of `rebuild` alone, which understates the stored floor and makes the
    modelled saving `vkb - columns - kg - chunk` where the measured resident
    saving is `vkb` flat.

    The slab's A/B is the arbiter: the delta is 13.96 GB at six brackets with
    no residual, and `columns + kg` there is 0.6-0.7 GB, which would have
    shown. So the two core lines belong to both branches.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_file(
            CELL, pseudo_dir="tests/data/pseudo", announce=False)
        stored = calculator.estimate(projectors="store")
        rebuilt = calculator.estimate(projectors="rebuild")

    for name in ("projector core columns (nk,npwx,ncs)",
                 "projector core kg (nk,npwx,3)"):
        assert name in stored.arrays, f"{name} missing from the stored route"
        assert name in rebuilt.arrays, f"{name} missing from the rebuilt route"
        assert stored.arrays[name] == rebuilt.arrays[name]

    # And therefore the modelled difference is exactly what the dial chooses
    # between -- the whole-k array against one chunk -- and nothing else.
    difference = stored.total_bytes - rebuilt.total_bytes
    assert difference == (stored.arrays["projectors vkb (nk,npwx,nkb)"]
                          - rebuilt.arrays["projectors rebuilt, one chunk (npwx,nkb)"])


def test_a_single_k_point_run_saves_exactly_nothing():
    """`vkb` *is* the chunk when there is one k-point, so the dial is neutral.

    Reported as a small *loss* once, which was the double-counted core: the
    stored floor was missing `columns + kg` and so read lower than it is. The
    honest statement is that the dial buys nothing at `nk = 1` and costs
    nothing either, and starts paying as soon as `nk > k_batch`.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_file(
            CELL, pseudo_dir="tests/data/pseudo", announce=False)
        stored = calculator.estimate(projectors="store", k_batch=1)
        rebuilt = calculator.estimate(projectors="rebuild", k_batch=1)
    if stored.nk == 1:
        assert stored.total_bytes == rebuilt.total_bytes
    else:
        assert rebuilt.total_bytes < stored.total_bytes
