"""The laboratory sphere a spin spiral keeps its transverse pair on (``OPEN.md`` Part XXI item 1).

:func:`~defumat.basis.gvectors.displaced_gvectors` cuts ``{G : |G + shift| <= G_cut}``
on the resident box, with the smooth sphere as its prefix. These are host-side
checks of the set itself; whether the spiral then equals its unfolded supercell
is ``tests/regression/test_spin_spirals_augmented.py``.
"""

from functools import lru_cache

import numpy as np
import pytest

from defumat.basis.builder import build_basis
from defumat.basis.gvectors import GVectors, displaced_gvectors
from defumat.io.pwin import parse_pw_input
from defumat.system import build_system
from defumat.system.spiral import spiral_cartesian
from tests.conftest import GENERATED


@lru_cache(maxsize=2)
def _cobalt(ecutrho: float):
    """The cobalt helix's cell, which has a separate smooth grid at 160 Ry."""
    text = (GENERATED / "co-helix4-spiral.in").read_text().replace(
        "ecutrho = 160.0", f"ecutrho = {ecutrho}")
    system = build_system(parse_pw_input(text))
    basis = build_basis(system)
    q = np.asarray(spiral_cartesian(system.spiral_q, system.cell)) * float(system.cell.tpiba)
    return system, basis, q


def test_at_no_shift_the_sets_are_the_resident_ones_index_for_index():
    """``shift = 0`` is the resident dense set and its smooth prefix, Miller index for Miller index.

    The ``q = 0`` spiral's energy test is a 1e-10 tolerance, which a
    reordering inside a shell would pass; this is the identity itself.
    """
    system, basis, _ = _cobalt(160.0)
    assert basis.doublegrid
    dense, smooth = displaced_gvectors(system.cell, basis.dense, np.zeros(3),
                                       smooth=basis.smooth)
    assert np.array_equal(np.asarray(dense.miller), np.asarray(basis.dense.miller))
    assert np.array_equal(np.asarray(smooth.miller), np.asarray(basis.smooth.miller))
    assert dense.grid == basis.dense.grid and smooth.grid == basis.smooth.grid


@pytest.mark.parametrize("ecutrho", [80.0, 160.0])
def test_the_displaced_set_is_the_laboratory_sphere(ecutrho):
    """Every G with ``|G - q|^2 <= ecutrho`` and no other, ``G = 0`` first, the smooth sphere its prefix.

    Counted against a brute-force enumeration over a box three times as wide,
    so a member the enumeration's range missed would show as a count short.
    The resident and the displaced counts differ (cobalt at 160 Ry: 4325
    against 4344), which is the shell the change is about.
    """
    system, basis, q = _cobalt(ecutrho)
    dense, smooth = displaced_gvectors(system.cell, basis.dense, -q, smooth=basis.smooth)
    bg = np.asarray(system.cell.bg)

    ranges = [np.arange(-n, n + 1) for n in basis.dense.grid]
    every = np.stack([a.ravel() for a in np.meshgrid(*ranges, indexing="ij")], axis=1)
    lab = np.sum((every @ bg - q) ** 2, axis=1)
    expected = {tuple(m) for m in every[lab <= basis.dense.ecut]}
    got = [tuple(int(v) for v in m) for m in np.asarray(dense.miller)]
    assert set(got) == expected and len(got) == len(expected)
    assert got[0] == (0, 0, 0)
    assert dense.ngm != basis.dense.ngm

    g2 = np.sum((np.asarray(dense.miller) @ bg - q) ** 2, axis=1)
    inner = g2 <= basis.smooth.ecut
    assert inner[:smooth.ngm].all() and not inner[smooth.ngm:].any()
    assert np.array_equal(np.asarray(smooth.miller), np.asarray(dense.miller)[:smooth.ngm])
    # Members past ``(n - 1) // 2`` are held at the wrapped index, and the
    # point of the check is that no two of them share one.
    assert len(np.unique(np.asarray(dense.fft_index))) == dense.ngm
    assert len(np.unique(np.asarray(smooth.fft_index))) == smooth.ngm


def test_a_box_that_would_fold_two_members_together_is_refused():
    """The guard fires on a box one plane narrower than the sphere spans.

    ``pw.x``'s FFT sizes leave room on every axis of every cell measured (the
    cobalt cell from 80 to 320 Ry at half a reciprocal vector along each axis,
    the oxygen chain at ``q3 = 1/4``, ``0.3`` and ``1/2``, silicon at ``0.3``
    and ``1/2``), so no real input reaches the refusal and the box is narrowed
    by hand: ``2 max|m_3|`` points along ``c`` for a sphere spanning
    ``2 max|m_3| + 1``. One plane wider, the same sphere is held.
    """
    system, basis, q = _cobalt(80.0)
    miller = np.asarray(basis.dense.miller)
    span = int(2 * np.abs(miller[:, 2]).max() + 1)
    held = GVectors(miller=basis.dense.miller, grid=basis.dense.grid[:2] + (span,),
                    ecut=basis.dense.ecut)
    narrow = GVectors(miller=basis.dense.miller, grid=basis.dense.grid[:2] + (span - 1,),
                      ecut=basis.dense.ecut)
    displaced_gvectors(system.cell, held, np.zeros(3))
    with pytest.raises(NotImplementedError, match="share a box index"):
        displaced_gvectors(system.cell, narrow, -q)


def test_g_space_mixing_is_refused_on_a_magnetic_spiral(pseudo_dir):
    """``mixing_space = 'g'`` keeps every channel on the resident sphere, so a spiral refuses it."""
    from defumat.pseudo import read_upf
    from defumat.scf import run_scf

    system = build_system(parse_pw_input((GENERATED / "h-chain-spiral.in").read_text()))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    with pytest.raises(NotImplementedError, match="mixing_space = 'g' on a magnetic spin spiral"):
        run_scf(system, pseudos, mixing_space="g", max_iterations=1)
