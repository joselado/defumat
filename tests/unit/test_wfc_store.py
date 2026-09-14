"""Where the wavefunction store lives, and proof that the host branch ran.

The store is ``nspin nk nbnd npwx npol`` complex -- 11.3 GB on a 45-atom NiBr2
slab at six k-points and 45 GB at twenty-four -- and it sat on the accelerator
for the whole SCF because it is the next iteration's starting guess. QE does not
keep it there: ``c_bands.f90``'s ``get_buffer``/``save_buffer`` pair reads one
k-point's ``evc`` in and writes it back, and ``io_level`` says whether the buffer
is memory or disk. :func:`park_wavefunctions`/:func:`fetch_wavefunctions` are
that buffer.

**The trap this file is written against is a null that reads as a pass.** Every
correctness assertion below -- same energy, same wavefunctions, same iteration
count -- would pass byte for byte if parking did nothing at all. So two of the
tests check that the branch *executed*: that a parked array is a different
buffer from the one it came from, and that the driver parked the store once per
iteration and got a host-resident array back. Without those the file would be
testing a branch that never runs.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat.batching import (
    WFC_STORES, fetch_wavefunctions, park_wavefunctions, resolve_wfc_store,
)
from defumat.calculator import Calculator

pytestmark = pytest.mark.unit

CELL = "tests/data/qe/h-atom-lsda.in"


def _host_memory_available() -> bool:
    try:
        return "pinned_host" in {m.kind for m in jax.devices()[0].addressable_memories()}
    except Exception:
        return False


# --------------------------------------------------------------------------
# the dial
# --------------------------------------------------------------------------

def test_the_platform_decides_when_nothing_else_does(monkeypatch):
    """CPU keeps it on the device, an accelerator parks it.

    The two dials in this module fall the *other* way from the chunk sizes, and
    for the reason that makes them a pair: on a CPU the host is the device, so
    parking is a memcpy of the whole store per iteration that buys nothing.
    """
    from defumat import batching

    monkeypatch.delenv("DEFUMAT_WFC_STORE", raising=False)
    monkeypatch.setattr(batching, "_backend", lambda: "cpu")
    assert resolve_wfc_store() == "device"
    monkeypatch.setattr(batching, "_backend", lambda: "gpu")
    assert resolve_wfc_store() == "host"


def test_an_explicit_setting_beats_the_environment_beats_the_platform(monkeypatch):
    """The precedence `k_batch` already has, and for the same reason."""
    from defumat import batching

    monkeypatch.setattr(batching, "_backend", lambda: "cpu")
    monkeypatch.setenv("DEFUMAT_WFC_STORE", "host")
    assert resolve_wfc_store() == "host", "the environment must beat the platform"
    assert resolve_wfc_store("device") == "device", "an argument must beat both"


def test_a_setting_that_is_not_a_place_is_refused():
    with pytest.raises(ValueError, match="wfc_store must be one of"):
        resolve_wfc_store("disk")


def test_an_unreadable_environment_variable_warns_and_falls_back(monkeypatch):
    """Ignored settings are the quietest kind of wrong, so it says so."""
    monkeypatch.setenv("DEFUMAT_WFC_STORE", "somewhere")
    with pytest.warns(RuntimeWarning, match="DEFUMAT_WFC_STORE"):
        assert resolve_wfc_store() in WFC_STORES


def test_none_is_not_a_place(monkeypatch):
    """Unlike the chunk dials, where ``None`` means the whole axis.

    ``resolve_k_batch(None)`` is a meaningful request and cannot double as
    "nothing was said". Here it can, and does.
    """
    from defumat import batching

    monkeypatch.delenv("DEFUMAT_WFC_STORE", raising=False)
    monkeypatch.setattr(batching, "_backend", lambda: "cpu")
    assert resolve_wfc_store(None) == "device"


# --------------------------------------------------------------------------
# the buffer itself, and that it is a buffer
# --------------------------------------------------------------------------

def test_parking_is_a_real_move_and_not_a_relabelling():
    """**The test that stops every other one in this file being a null.**

    A correctness check cannot tell parking from a no-op: the answers are the
    same either way, by construction. The buffer pointer can. On this CPU
    backend ``pinned_host`` is a genuine second allocation, which is what makes
    the rest of the file a test of the branch that ships on an accelerator.
    """
    if not _host_memory_available():
        pytest.skip("this backend has no pinned-host memory to park in")
    psi = jnp.arange(256, dtype=jnp.complex128).reshape(2, 8, 16)
    parked = park_wavefunctions(psi, "host")
    assert parked.sharding.memory_kind == "pinned_host"
    assert parked.unsafe_buffer_pointer() != psi.unsafe_buffer_pointer()
    back = fetch_wavefunctions(parked)
    assert back.sharding.memory_kind == "device"
    assert np.array_equal(np.asarray(back), np.asarray(psi))


def test_the_device_setting_and_a_missing_store_are_both_free():
    """Called unconditionally by the driver, so both must be exactly nothing."""
    psi = jnp.ones((2, 3))
    assert park_wavefunctions(psi, "device") is psi
    assert park_wavefunctions(None, "host") is None
    assert fetch_wavefunctions(psi) is psi
    assert fetch_wavefunctions(None) is None


def test_a_transfer_inside_a_traced_path_is_refused():
    """It *works* inside ``jit``, which is exactly why it is guarded.

    ``jax.grad`` raises on its own -- a memory kind is part of the aval, so a
    parked array's cotangent has a type the primal does not. A plain ``jit``
    does not: it happily compiles a transfer into the middle of a kernel, which
    is slower than doing nothing and says nothing.
    """
    with pytest.raises(TypeError, match="cannot happen inside jit or grad"):
        jax.jit(lambda a: park_wavefunctions(a, "host"))(jnp.ones(4))
    with pytest.raises(TypeError, match="cannot happen inside jit or grad"):
        jax.jit(fetch_wavefunctions)(jnp.ones(4))


# --------------------------------------------------------------------------
# the driver
# --------------------------------------------------------------------------

def _scf(where, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_file(
            CELL, pseudo_dir="tests/data/pseudo", announce=False)
        return calculator.get_scf(wfc_store=where, conv_thr=1e-11, **options)


def test_the_dial_is_not_visible_in_any_number():
    """Bit for bit, not close: nothing about a transfer is arithmetic.

    The chunk dials change the *order* contributions are added in and so move
    the last digit; this one moves no contribution at all, so the standard here
    is equality and not a tolerance.
    """
    on_device, on_host = _scf("device"), _scf("host")
    assert on_host.total_energy == on_device.total_energy
    assert on_host.iterations == on_device.iterations
    assert np.array_equal(np.asarray(on_host.wavefunctions),
                          np.asarray(on_device.wavefunctions))
    assert np.array_equal(np.asarray(on_host.eigenvalues),
                          np.asarray(on_device.eigenvalues))


def test_the_driver_really_parks_the_store(monkeypatch):
    """Counted, and the parked object's memory kind asserted.

    The test above passes whether or not anything was parked. This one fails if
    the driver stops calling the buffer, if the dial stops reaching it, or if
    the call is made somewhere the store is ``None``.
    """
    if not _host_memory_available():
        pytest.skip("this backend has no pinned-host memory to park in")
    from defumat.batching import park_wavefunctions as real
    from defumat.scf import driver

    kinds = []

    def spy(psi, where):
        parked = real(psi, where)
        if psi is not None:
            kinds.append(getattr(parked.sharding, "memory_kind", None))
        return parked

    monkeypatch.setattr(driver, "park_wavefunctions", spy)
    result = _scf("host", max_iterations=3)

    assert len(kinds) >= 3, f"the store was parked {len(kinds)} times in 3 iterations"
    assert set(kinds) == {"pinned_host"}, kinds
    # ... and what leaves the driver is an ordinary device array, so nothing
    # downstream has to know a buffer existed.
    assert result.wavefunctions.sharding.memory_kind == "device"


def test_the_solver_is_never_handed_a_parked_store(monkeypatch):
    """The fetch is unconditional, so a store the dial did not park still works.

    A resume, or a caller passing its own span, can reach the diagonalisation
    with a store this loop never touched. `diagonalize` would accept a host
    array -- JAX inserts the transfer -- and that is the failure to avoid: it
    runs, and the transfer lands inside the compiled unit instead of in front
    of it.
    """
    if not _host_memory_available():
        pytest.skip("this backend has no pinned-host memory to park in")
    from defumat.scf.driver import Calculation

    seen = []
    real = Calculation.diagonalize

    def spy(self, hamiltonians, nbnd, psi0=None, *args, **kw):
        seen.append(None if psi0 is None
                    else getattr(getattr(psi0, "sharding", None), "memory_kind", None))
        return real(self, hamiltonians, nbnd, psi0, *args, **kw)

    monkeypatch.setattr(Calculation, "diagonalize", spy)
    _scf("host", max_iterations=3)

    assert seen, "the solver was never called"
    assert all(kind in (None, "device") for kind in seen), seen


def test_the_resolved_mode_is_printed_and_a_typo_does_not_read_as_a_pin(
        monkeypatch, capsys):
    """The log says what is in force, not what was asked for.

    ``_wfc_store_default`` warns and falls back on a value it does not
    recognise, and nothing else printed the answer -- so in a cluster job's log
    a misspelt ``DEFUMAT_WFC_STORE`` looked exactly like a working pin. The
    line under test is the one that tells them apart, and the assertion is that
    it agrees with :func:`resolve_wfc_store` rather than with the request.
    """
    _scf("device", max_iterations=1, verbose=True)
    printed = capsys.readouterr().out
    assert "wfc_store = device" in printed
    assert "k_batch = " in printed and "band_batch = " in printed

    # The typo. The run falls back to the platform default; the log must say
    # the default, and must not echo what was asked for.
    monkeypatch.setenv("DEFUMAT_WFC_STORE", "hsot")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _scf("default", max_iterations=1, verbose=True)
        fallback = resolve_wfc_store("default")
    printed = capsys.readouterr().out
    assert f"wfc_store = {fallback}" in printed
    assert "hsot" not in printed
