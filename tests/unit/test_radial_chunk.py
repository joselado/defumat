"""The radial transforms' chunk is sized from the mesh, and chunking changes nothing.

:func:`~defumat.pseudo.formfactors.radial_chunk` bounds one ``(chunk, mesh)``
integrand at :data:`~defumat.pseudo.formfactors.RADIAL_CHUNK_BYTES` rather than
at a fixed 4096 values, because the strain's derivatives hold many such arrays at
once (``PERFORMANCE.md``, "The radial transforms' chunk, sized from the mesh").
Two things are checked: the arithmetic, and that a transform walked in chunks
across a padded boundary is the same transform taken in pieces small enough to
need no chunking.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.pseudo.formfactors import (
    CHUNK, MIN_CHUNK, core_charge_of_g, local_potential_of_g, radial_chunk,
)
from defumat.pseudo.upf import read_upf

pytestmark = [pytest.mark.unit]

PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


def test_the_chunk_follows_the_mesh_between_its_bounds(monkeypatch):
    monkeypatch.delenv("DEFUMAT_RADIAL_CHUNK", raising=False)
    assert radial_chunk(841) == 1246
    assert radial_chunk(995) == 1053
    assert radial_chunk(10) == CHUNK
    assert radial_chunk(10**7) == MIN_CHUNK
    monkeypatch.setenv("DEFUMAT_RADIAL_CHUNK", "300")
    assert radial_chunk(841) == 300


@pytest.mark.parametrize("transform", [local_potential_of_g, core_charge_of_g])
def test_a_chunked_transform_is_the_transform_in_pieces(transform):
    """Across the scan's padded boundary, against pieces below one chunk."""
    pseudo = read_upf(PSEUDO / "As.pbe-n-rrkjus_psl.1.0.0.UPF")
    if transform is core_charge_of_g and pseudo.rho_core is None:
        pytest.skip("no core charge in this dataset")
    chunk = radial_chunk(int(pseudo.msh))
    q = jnp.linspace(0.0, 12.0, 2 * chunk + 37)
    walked = np.asarray(transform(pseudo, q, 300.0))
    pieces = np.concatenate([
        np.asarray(transform(pseudo, q[start:start + 200], 300.0))
        for start in range(0, q.shape[0], 200)
    ])
    scale = float(np.abs(pieces).max())
    assert float(np.abs(walked - pieces).max()) <= 1e-13 * scale
