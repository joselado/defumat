"""Which route the field response takes: the whole k axis, or a k-chunk at a time.

:func:`defumat.response.efield._streams` is a rule about memory rather than about
physics -- the two routes agree to round-off (``tests/regression/
test_streamed_response.py``) -- so what is checked here is the rule itself, on a
stand-in calculation that carries only what the rule reads. The card branch is
reached by making the platform read as a GPU: ``k_batch = 'fit'`` sizes the
chunk for the SCF, and on ultrasoft eight-atom silicon at 27 k-points the whole-k
route's Born charges took an RTX A2000 to 6825.4 MB where the SCF peaked at
925.2, so in memory mode on a card the response walks chunks even when the chunk
is the whole mesh.
"""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import defumat.batching as batching
from defumat.response.efield import _streams

pytestmark = pytest.mark.unit


def _calculation(memory_mode, k_batch, nk=27, **extra):
    return SimpleNamespace(
        system=SimpleNamespace(kpoints=SimpleNamespace(nk=nk)),
        memory_mode=memory_mode, k_batch=k_batch, **extra)


STORE = np.zeros((1, 27, 2, 3), dtype=complex)


def test_on_a_cpu_the_route_is_the_force_rule():
    """A host store, or memory mode with a chunk smaller than the mesh."""
    assert _streams(_calculation("speed", None), STORE, False)
    assert _streams(_calculation("memory", 4), jnp.asarray(STORE), False)
    assert not _streams(_calculation("memory", None), jnp.asarray(STORE), False)
    assert not _streams(_calculation("speed", None), jnp.asarray(STORE), False)


def test_on_a_card_memory_mode_walks_chunks_whatever_the_chunk(monkeypatch):
    monkeypatch.setattr(batching, "_backend", lambda: "gpu")
    device = jnp.asarray(STORE)
    assert _streams(_calculation("memory", None), device, False)
    assert _streams(_calculation("memory", 27), device, False)
    # Speed mode keeps the whole-k route it asked for.
    assert not _streams(_calculation("speed", None), device, False)
    # And the three exits to the whole-k route still hold on a card.
    assert not _streams(_calculation("memory", None), device, True)
    assert not _streams(_calculation("memory", None, _kcart=np.zeros((27, 3))),
                        device, False)
