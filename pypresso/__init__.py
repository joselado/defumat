"""pypresso -- plane-wave DFT in Python and JAX.

Importing this package enables JAX's 64-bit mode before any array can be
created. That only *permits* 64-bit; the dtype every array is actually built
with still comes from a :class:`pypresso.config.Precision` policy.
"""

import jax as _jax

_jax.config.update("jax_enable_x64", True)

from pypresso import config, units  # noqa: E402
from pypresso.config import DOUBLE, SINGLE, Precision  # noqa: E402

__version__ = "0.0.1"
__all__ = ["config", "units", "Precision", "DOUBLE", "SINGLE", "__version__"]
