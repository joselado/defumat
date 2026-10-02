"""A closure called eagerly is compiled once per structure, and never wrongly reused.

:mod:`defumat.eager` keys a compiled program on the printed jaxpr of a closure
with its arrays hoisted out, so the property that matters is two-sided: calls
that differ only in their arrays share one program, and calls that differ in
anything the printed form does not show (a nested constant, a callback) do not.
Each refusal here is fed a case that must trip it, since a guard that cannot fire
reads exactly like one that passes.

The response-stack counts are measured on the one-k-point silicon cell; the
AlAs and phonon-at-q figures behind the change are in ``PERFORMANCE.md``, "The
response stack compiled its k loops again at every iteration".
"""

import contextlib
import logging
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from defumat import eager
from defumat.calculator import Calculator
from defumat.eager import compiled

pytestmark = pytest.mark.unit

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"


@contextlib.contextmanager
def counting_compiles():
    """Every XLA compilation inside the block, by name, read off the ``jax`` logger."""
    names = []

    class Grab(logging.Handler):
        def emit(self, record):
            found = re.search(r"Finished XLA compilation of (.+?) in", record.getMessage())
            if found:
                names.append(found.group(1))

    handler, logger = Grab(), logging.getLogger("jax")
    before = jax.config.jax_log_compiles
    jax.config.update("jax_log_compiles", True)
    logger.addHandler(handler)
    try:
        yield names
    finally:
        logger.removeHandler(handler)
        jax.config.update("jax_log_compiles", before)


@pytest.fixture(autouse=True)
def fresh_cache():
    eager.clear()
    yield
    eager.clear()


def _loop(arr):
    return compiled(lambda xs: lax.map(lambda x: x * arr + jnp.sin(x), xs),
                    jnp.arange(4.0))


def test_closures_over_different_arrays_share_one_program():
    a, b = jnp.arange(4.0), jnp.linspace(1.0, 2.0, 4)
    first = _loop(a)
    with counting_compiles() as names:
        second = _loop(b)
    assert names == []
    assert len(eager._PROGRAMS) == 1
    for arr, got in ((a, first), (b, second)):
        expected = lax.map(lambda x: x * arr + jnp.sin(x), jnp.arange(4.0))
        np.testing.assert_array_equal(got, expected)


def test_a_different_structure_is_a_different_program():
    _loop(jnp.arange(4.0))
    compiled(lambda xs: lax.map(lambda x: x + 1.0, xs), jnp.arange(4.0))
    compiled(lambda xs: lax.map(lambda x: x + 1.0 + 2**-50, xs), jnp.arange(4.0))
    assert len(eager._PROGRAMS) == 3


def test_a_nested_constant_is_part_of_the_key():
    """A ``jit`` closing over an array keeps it unprinted inside its own jaxpr."""
    def nested(arr):
        inner = jax.jit(lambda y: y + arr)
        return compiled(lambda xs: lax.map(inner, xs), jnp.arange(4.0))

    a, b = jnp.arange(4.0), jnp.ones(4)
    got_a = nested(a)
    got_b = nested(b)
    assert len(eager._PROGRAMS) == 2
    np.testing.assert_array_equal(got_a, jnp.arange(4.0)[:, None] + a)
    np.testing.assert_array_equal(got_b, jnp.arange(4.0)[:, None] + b)
    nested(a)
    assert len(eager._PROGRAMS) == 2


def test_a_nested_constant_past_the_limit_is_not_kept(monkeypatch):
    monkeypatch.setattr(eager, "NESTED_LIMIT", 8)
    arr = jnp.arange(4.0)
    inner = jax.jit(lambda y: y + arr)
    got = compiled(lambda xs: lax.map(inner, xs), jnp.arange(4.0))
    assert len(eager._PROGRAMS) == 0
    np.testing.assert_array_equal(got, jnp.arange(4.0)[:, None] + arr)


def test_a_callback_is_not_kept():
    def host(x):
        return np.asarray(x) * 2.0

    def body(x):
        return jax.pure_callback(host, jax.ShapeDtypeStruct(x.shape, x.dtype), x)

    got = compiled(lambda xs: lax.map(body, xs), jnp.arange(4.0))
    assert len(eager._PROGRAMS) == 0
    np.testing.assert_array_equal(got, 2.0 * jnp.arange(4.0))


def test_under_a_transformation_it_is_the_plain_call():
    arr = jnp.arange(4.0)
    primal, tangent = jax.jvp(lambda s: _loop(arr * s), (1.0,), (1.0,))
    assert len(eager._PROGRAMS) == 0
    np.testing.assert_allclose(tangent, jnp.arange(4.0)[:, None] * arr)
    np.testing.assert_allclose(jax.jit(lambda s: _loop(arr * s))(2.0),
                               _loop(2.0 * arr))


@pytest.fixture(scope="module")
def silicon(pseudo_dir):
    calculator = Calculator.from_file(BENCHMARKS / "si-1k.in", pseudo_dir,
                                      announce=False)
    calculator.get_scf()
    return calculator


def test_a_second_band_velocity_call_compiles_nothing(silicon):
    first = silicon.get_band_velocities()
    with counting_compiles() as names:
        second = silicon.get_band_velocities()
    assert names == []
    np.testing.assert_array_equal(first.velocities_by_spin, second.velocities_by_spin)


@pytest.mark.slow
def test_a_second_dielectric_tensor_call_compiles_nothing(silicon):
    """Before :mod:`defumat.eager` a second call compiled 127 programs here."""
    first = silicon.get_dielectric_tensor()
    with counting_compiles() as names:
        second = silicon.get_dielectric_tensor()
    assert names == []
    np.testing.assert_allclose(second.epsilon, first.epsilon, rtol=0, atol=1e-12)


def test_a_full_cache_says_so_once(monkeypatch):
    monkeypatch.setattr(eager, "CACHE_SIZE", 1)
    _loop(jnp.arange(4.0))
    with pytest.warns(RuntimeWarning, match="dropped the oldest"):
        compiled(lambda xs: lax.map(lambda x: x + 1.0, xs), jnp.arange(4.0))
    assert len(eager._PROGRAMS) == 1
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compiled(lambda xs: lax.map(lambda x: x - 1.0, xs), jnp.arange(4.0))


def test_host_arithmetic_on_constants_is_evaluated_while_tracing():
    """``augmentation_dipole`` reads ``np.asarray`` of a ``jnp`` result on constants."""
    table = np.arange(4.0)

    def fn(x):
        weights = np.asarray(jnp.cumsum(jnp.asarray(table)))
        return x * float(weights.sum())

    got = compiled(fn, jnp.asarray(2.0))
    assert len(eager._PROGRAMS) == 1
    np.testing.assert_allclose(got, 2.0 * 10.0)


def test_a_value_read_off_an_argument_falls_back_to_the_plain_call():
    """``map_axis`` with one entry calls its body on concrete values, which a trace cannot."""
    from defumat.batching import map_k

    def body(x):
        return x * 2.0 if float(x) > 0 else x

    got = compiled(lambda xs: map_k(body, xs, batch=1), jnp.ones(1))
    assert len(eager._PROGRAMS) == 0
    np.testing.assert_allclose(got, [2.0])
