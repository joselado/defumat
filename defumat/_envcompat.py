"""The pre-rename ``PYPRESSO_*`` environment variables, still read.

The package was called ``pypresso`` until 2026-09. A script or a cluster job
file that still exports ``PYPRESSO_K_BATCH`` keeps working and is told once,
per variable, that the name has moved. Remove this module after 0.1.

Only the variables a *user* sets are covered. ``DEFUMAT_PINNED`` is not: it is
a marker ``tools/compare_qe.py`` writes and reads within one process, so it has
no old spelling in anybody's shell.
"""

from __future__ import annotations

import os
import warnings

_warned: set[str] = set()


def environ_get(name: str, default: str | None = None) -> str | None:
    """``name``, falling back to its ``PYPRESSO_`` spelling with one warning."""
    value = os.environ.get(name)
    if value is not None:
        return value

    legacy = name.replace("DEFUMAT_", "PYPRESSO_", 1)
    value = os.environ.get(legacy)
    if value is None:
        return default

    if legacy not in _warned:
        _warned.add(legacy)
        warnings.warn(
            f"{legacy} is the old name for {name} and is read only until 0.1; "
            "rename it in whatever sets it.",
            DeprecationWarning,
            stacklevel=2,
        )
    return value
