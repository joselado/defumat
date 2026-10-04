"""A streamed store crosses to the device through ``device_put``, at its own size.

``OPEN.md`` Part XXIII item 24, first half. Where the store streams,
``fixed_density_states`` returns the states as a host numpy array, and the
conductivity, SHG and shift-current workflows, the TDDFT and magnon responses and
the two torque energies put it on the device whole with ``jnp.asarray``, which on
a card peaks at twice the array's size where ``jax.device_put`` of a contiguous
array peaks at once (``GPU-MEMORY-NEXT.md``, 2026-09-29). They now go through
:func:`defumat.batching.upload`. The memory is measured on a card; what is
checked here is that the helper does the right thing with each kind of input, and
that the two torque energies, which still take a store whole, use it. The five
sum-over-states assemblies no longer take it whole at all (item 24's second
half): they walk the k axis a chunk at a time, and their workflows hand the band
slice on where it is.
"""

import importlib
import inspect
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat.batching import upload

pytestmark = pytest.mark.unit


def test_a_strided_host_slice_crosses_once_and_contiguous(monkeypatch):
    rng = np.random.default_rng(0)
    store = rng.normal(size=(1, 3, 6, 5)) + 1j * rng.normal(size=(1, 3, 6, 5))
    view = store[..., :4, :]                  # a band slice, as the workflows take
    assert not view.flags.c_contiguous

    handed = []
    real = jax.device_put

    def spy(value, *args, **kwargs):
        handed.append(value)
        return real(value, *args, **kwargs)

    monkeypatch.setattr(jax, "device_put", spy)
    on_device = upload(view)
    monkeypatch.undo()

    assert len(handed) == 1 and handed[0].flags.c_contiguous
    assert isinstance(on_device, jax.Array)
    assert on_device.dtype == view.dtype
    assert np.array_equal(np.asarray(on_device), view)


def test_a_device_array_and_a_tracer_are_left_alone(monkeypatch):
    already = jnp.arange(6.0).reshape(2, 3)
    monkeypatch.setattr(jax, "device_put", lambda *a, **k: pytest.fail(
        "a device array was sent through device_put"))
    assert upload(already) is already
    # Inside a trace the store is a tracer, which ``np.ascontiguousarray``
    # cannot take; it must pass through as ``jnp.asarray`` passes it.
    assert np.array_equal(np.asarray(jax.jit(lambda x: upload(x) * 2.0)(already)),
                          2.0 * np.asarray(already))


#: The sites that still take a store whole, as ``(module, function)``: the two
#: torque energies, whose chunked derivative walks indices into a device store.
#: Imported inside the test rather than here, so that a module that fails to
#: import is one failing case and not a collection error that stops the whole
#: run.
UPLOADERS = [
    ("defumat.forces.torque", "_band_energy"),
    ("defumat.forces.torque", "_chunked_value_and_grad"),
]

#: The five sum-over-states assemblies, which walk the k axis and take one
#: chunk's rows to the device at a time (item 24's second half,
#: :mod:`defumat.response.walk`), and the three workflows in front of them,
#: which hand the band slice on where it is rather than uploading it.
WALKERS = [
    ("defumat.response.conductivity", "optical_conductivity"),
    ("defumat.response.shg", "second_harmonic"),
    ("defumat.response.photocurrent", "shift_current"),
    ("defumat.tddft.chi0", "independent_response"),
    ("defumat.tddft.spinchi0", "transverse_response"),
]
HANDERS = [
    ("defumat.workflows.conductivity", "run_conductivity"),
    ("defumat.workflows.shg", "run_shg"),
    ("defumat.workflows.photocurrent", "run_shift_current"),
]

#: Putting the whole store on the device, by either spelling.
WHOLE = r"(?:jnp\.asarray|upload)\((?:wavefunctions|states)\b"


@pytest.mark.parametrize("module, name", UPLOADERS, ids=[name for _, name in UPLOADERS])
def test_each_whole_store_site_uploads_its_states_through_the_helper(module, name):
    function = getattr(importlib.import_module(module), name)
    source = inspect.getsource(function)
    assert "upload(" in source, f"{function.__qualname__} does not use upload"
    whole = re.findall(r"jnp\.asarray\((?:wavefunctions|states)\b", source)
    assert not whole, (
        f"{function.__qualname__} still puts the store on the device with "
        f"jnp.asarray: {whole}"
    )


@pytest.mark.parametrize("module, name", WALKERS, ids=[name for _, name in WALKERS])
def test_each_assembly_walks_the_store_a_chunk_at_a_time(module, name):
    function = getattr(importlib.import_module(module), name)
    source = inspect.getsource(function)
    assert "walk(" in source and "store_rows(" in source, (
        f"{function.__qualname__} does not walk the k axis")
    assert not re.findall(WHOLE, source), (
        f"{function.__qualname__} still puts the whole store on the device")


@pytest.mark.parametrize("module, name", HANDERS, ids=[name for _, name in HANDERS])
def test_each_workflow_hands_the_store_on_where_it_is(module, name):
    function = getattr(importlib.import_module(module), name)
    source = inspect.getsource(function)
    assert not re.findall(WHOLE, source), (
        f"{function.__qualname__} puts the whole store on the device before "
        "the assembly walks it")
