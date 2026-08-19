"""The precision policy is the mechanism that keeps single precision viable.

These tests exist to catch the failure mode the policy is meant to prevent:
somewhere down the line a dtype gets hardcoded and the float32 path silently
becomes float64 (or worse, x64 gets disabled and float64 silently becomes
float32).
"""

import os
import subprocess
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pypresso
from pypresso.config import DOUBLE, SINGLE, precision_by_name

pytestmark = pytest.mark.unit


def test_x64_is_enabled_on_import():
    assert jax.config.jax_enable_x64 is True
    assert jnp.zeros(1).dtype == jnp.float64


@pytest.mark.parametrize(
    ("precision", "real", "complex_"),
    [(DOUBLE, np.float64, np.complex128), (SINGLE, np.float32, np.complex64)],
)
def test_policy_controls_dtypes(precision, real, complex_):
    assert precision.zeros(3).dtype == real
    assert precision.zeros(3, complex_=True).dtype == complex_
    assert precision.as_real([1, 2]).dtype == real
    assert precision.as_complex([1, 2]).dtype == complex_


def test_single_precision_is_actually_single():
    """Guards against x64 quietly promoting the float32 path back to float64."""
    x = SINGLE.as_complex(np.arange(4))
    assert (x * SINGLE.as_real(2.0)).dtype == np.complex64
    assert jnp.fft.fftn(x).dtype == np.complex64


def test_precision_is_a_static_pytree_field():
    """A Precision must not become a traced leaf: it carries no arrays."""
    leaves = jax.tree_util.tree_leaves(DOUBLE)
    assert leaves == []


def test_precision_by_name():
    assert precision_by_name("double") is DOUBLE
    assert precision_by_name("float32") is SINGLE
    with pytest.raises(ValueError, match="unknown precision"):
        precision_by_name("quadruple")


def test_equinox_module_survives_jit_and_grad():
    """The OO pattern the codebase is built on has to work end to end."""

    class Toy(eqx.Module):
        values: jnp.ndarray
        scale: float = eqx.field(static=True)

        def energy(self):
            return self.scale * jnp.vdot(self.values, self.values).real

    toy = Toy(values=DOUBLE.as_complex(np.arange(3)), scale=2.0)
    assert eqx.filter_jit(lambda t: t.energy())(toy) == pytest.approx(10.0)

    grad = jax.grad(lambda v: Toy(values=v, scale=2.0).energy())(DOUBLE.as_real(np.arange(3.0)))
    assert grad == pytest.approx([0.0, 4.0, 8.0])
    assert grad.dtype == np.float64


def test_package_exports():
    assert pypresso.__version__
    assert pypresso.DOUBLE is DOUBLE


# --- the persistent compilation cache -----------------------------------------
#
# Enabled on import, because compilation rather than arithmetic is what a short
# run spends its time on. Tested through a subprocess: the setting is read once,
# before JAX has created anything, so it cannot be exercised by re-importing.


def _cache_dir_in_a_fresh_process(value: str | None) -> str:
    """What ``jax_compilation_cache_dir`` ends up as, with the env var set."""
    environment = dict(os.environ)
    environment.pop("PYPRESSO_CACHE_DIR", None)
    if value is not None:
        environment["PYPRESSO_CACHE_DIR"] = value

    script = (
        "import sys; sys.path.insert(0, '.');"
        "import pypresso, jax;"
        "print(jax.config.jax_compilation_cache_dir or '')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        env=environment, cwd=Path(__file__).resolve().parents[2],
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout.strip()


@pytest.mark.parametrize("setting", ["off", "0", "none", ""])
def test_the_cache_can_be_turned_off(setting):
    assert _cache_dir_in_a_fresh_process(setting) == ""


def test_the_cache_honours_an_explicit_directory(tmp_path):
    directory = tmp_path / "somewhere"
    assert _cache_dir_in_a_fresh_process(str(directory)) == str(directory)
    assert directory.is_dir()


def _cores_in_a_fresh_process(value: str | None) -> int:
    """How many CPUs the process is left with, with the env var set."""
    environment = dict(os.environ)
    environment.pop("PYPRESSO_THREADS", None)
    if value is not None:
        environment["PYPRESSO_THREADS"] = value

    script = (
        "import sys, os; sys.path.insert(0, '.');"
        "import pypresso;"
        "print(len(os.sched_getaffinity(0)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        env=environment, cwd=Path(__file__).resolve().parents[2],
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return int(result.stdout.strip())


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="Linux only")
def test_the_thread_pool_is_capped_by_default():
    """XLA sizes its CPU pool from the affinity mask, and its default is slower.

    See ``pypresso._limit_thread_pool``: on this workload fourteen threads are
    almost twice as slow as four, so the package narrows the mask on import.
    """
    available = len(os.sched_getaffinity(0))
    expected = min(pypresso.DEFAULT_THREADS, available)
    assert _cores_in_a_fresh_process(None) == expected


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="Linux only")
def test_the_cap_can_be_set_or_turned_off():
    available = len(os.sched_getaffinity(0))
    assert _cores_in_a_fresh_process("1") == 1
    assert _cores_in_a_fresh_process("off") == available


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="Linux only")
def test_the_cap_never_widens_an_existing_restriction():
    """An outer taskset, or a scheduler's allocation, must be respected."""
    available = sorted(os.sched_getaffinity(0))
    if len(available) < 2:
        pytest.skip("needs more than one CPU to narrow")
    script = (
        "import sys, os; sys.path.insert(0, '.');"
        f"os.sched_setaffinity(0, {{{available[0]}}});"
        "import pypresso;"
        "print(len(os.sched_getaffinity(0)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert int(result.stdout.strip()) == 1


def test_the_cache_is_on_by_default():
    assert _cache_dir_in_a_fresh_process(None) != ""
