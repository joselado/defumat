"""defumat -- plane-wave DFT in Python and JAX.

Importing this package does three things before any array can be created:
enables JAX's 64-bit mode, points XLA at a persistent compilation cache, and
stops XLA from oversubscribing the machine with threads.

The 64-bit switch only *permits* 64-bit; the dtype every array is actually built
with still comes from a :class:`defumat.config.Precision` policy.

The cache is here because compilation, not arithmetic, is what a short run
spends its time on: a silicon SCF compiles for seconds and then computes for
tenths of a second, and without a cache it recompiles the identical kernels
every time the process starts. Quantum ESPRESSO has no equivalent cost, so this
is not an optimisation so much as removing an artefact of the implementation
language. See ``PERFORMANCE.md``.
"""

import os as _os
import warnings as _warnings
from pathlib import Path as _Path

from ._envcompat import environ_get as _environ_get

import jax as _jax

_jax.config.update("jax_enable_x64", True)

#: How many CPUs to leave visible to XLA. Four, because that is what the
#: measurements say; see :func:`_limit_thread_pool`.
DEFAULT_THREADS = 4


def _limit_thread_pool() -> None:
    """Keep XLA's CPU thread pool small, which on this workload makes it faster.

    XLA sizes its CPU thread pool from the process's *affinity mask*, and its
    default -- every core on the machine -- is the worst choice measured here.
    A plane-wave SCF is a long chain of FFTs and small matrix products with
    little parallelism inside any one operation, so past a handful of threads
    the pool spends more time synchronising than computing. On this 14-core
    machine, per SCF iteration of the 1131-plane-wave silicon benchmark:

        1 core   38 ms      4 cores   33 ms      14 cores   63 ms

    and the same shape holds for a 3215-plane-wave cell, so this is not an
    artefact of the small cases. Nothing else responds: ``OMP_NUM_THREADS`` moves
    it by a few percent, because it is not what sizes the pool.

    The real parallel axis for this code is k-points, and that is a
    ``jax.sharding`` question rather than a thread-pool one (``PLAN.md`` §5).

    ``DEFUMAT_THREADS`` overrides the count; ``0`` or ``off`` leaves the machine
    alone. The mask is only ever *narrowed* -- an outer ``taskset`` or a cluster
    scheduler's allocation is respected, never widened.
    """
    setting = _environ_get("DEFUMAT_THREADS", "").strip().lower()
    if setting in ("0", "off", "none", "false"):
        return
    try:
        wanted = int(setting) if setting else DEFAULT_THREADS
    except ValueError:
        _warnings.warn(f"ignoring DEFUMAT_THREADS={setting!r}: not a number",
                       RuntimeWarning, stacklevel=2)
        return

    affinity = getattr(_os, "sched_getaffinity", None)
    if affinity is None:  # pragma: no cover - not Linux
        return
    try:
        available = sorted(affinity(0))
        if 0 < wanted < len(available):
            _os.sched_setaffinity(0, set(available[:wanted]))
    except OSError as error:  # pragma: no cover - depends on the scheduler
        _warnings.warn(f"could not limit the CPU affinity: {error}",
                       RuntimeWarning, stacklevel=2)


_limit_thread_pool()


def _enable_compilation_cache() -> None:
    """Point XLA at a persistent cache, unless the user has said not to.

    ``DEFUMAT_CACHE_DIR`` overrides the location; setting it empty, or to
    ``0``/``off``/``none``, disables the cache entirely. Failures here are
    warnings and never exceptions: a read-only or full home directory must not
    stop a calculation from running.
    """
    setting = _environ_get("DEFUMAT_CACHE_DIR")
    if setting is not None and setting.strip().lower() in ("", "0", "off", "none", "false"):
        return

    directory = _Path(setting) if setting else _Path.home() / ".cache" / "defumat" / "jax"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        _jax.config.update("jax_compilation_cache_dir", str(directory))
        # Both defaults would silently cache nothing here: JAX only caches a
        # kernel that took over a second to compile, and ours take about fifty
        # milliseconds each -- there are simply a great many of them.
        _jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        _jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    except OSError as error:  # pragma: no cover - depends on the filesystem
        _warnings.warn(
            f"could not enable the JAX compilation cache at {directory}: {error}. "
            "Calculations will run, but every process will recompile from scratch.",
            RuntimeWarning,
            stacklevel=2,
        )


_enable_compilation_cache()

from defumat import config, units  # noqa: E402
from defumat.config import DOUBLE, SINGLE, Precision  # noqa: E402

__version__ = "0.0.1"
__all__ = ["config", "units", "Precision", "DOUBLE", "SINGLE", "DEFAULT_THREADS",
           "Calculator", "__version__"]


def __getattr__(name):
    """Resolve :class:`~defumat.calculator.Calculator` on first use.

    ``from defumat import Calculator`` is meant to be the only import a script
    needs, and importing it eagerly here would pull the whole package -- the
    SCF driver, the response layer, every workflow -- into any process that
    merely wanted ``defumat.units``. PEP 562 lets the short spelling cost
    nothing until it is actually asked for.
    """
    if name == "Calculator":
        from defumat.calculator import Calculator

        return Calculator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
