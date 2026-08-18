"""pypresso -- plane-wave DFT in Python and JAX.

Importing this package does two things before any array can be created: enables
JAX's 64-bit mode, and points XLA at a persistent compilation cache.

The 64-bit switch only *permits* 64-bit; the dtype every array is actually built
with still comes from a :class:`pypresso.config.Precision` policy.

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

import jax as _jax

_jax.config.update("jax_enable_x64", True)


def _enable_compilation_cache() -> None:
    """Point XLA at a persistent cache, unless the user has said not to.

    ``PYPRESSO_CACHE_DIR`` overrides the location; setting it empty, or to
    ``0``/``off``/``none``, disables the cache entirely. Failures here are
    warnings and never exceptions: a read-only or full home directory must not
    stop a calculation from running.
    """
    setting = _os.environ.get("PYPRESSO_CACHE_DIR")
    if setting is not None and setting.strip().lower() in ("", "0", "off", "none", "false"):
        return

    directory = _Path(setting) if setting else _Path.home() / ".cache" / "pypresso" / "jax"
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

from pypresso import config, units  # noqa: E402
from pypresso.config import DOUBLE, SINGLE, Precision  # noqa: E402

__version__ = "0.0.1"
__all__ = ["config", "units", "Precision", "DOUBLE", "SINGLE", "__version__"]
