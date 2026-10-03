"""The two warnings the second-order optical entry points give, checked without an SCF.

Both are about a residue that no symmetry check of the assembly can see,
because each one is a property of the *input* to the sum rather than of the
sum: the tensor comes out of a correct assembly carrying a part that is not
the crystal's.

**The grid.** A ``nosym`` run chooses its FFT grid with no regard for the
crystal's fractional translations, which is ``pw.x``'s own rule
(``setup.f90`` sets ``fft_fact = 1`` under ``nosym``). On diamond silicon, whose
inversion carries a quarter-lattice translation, the 15^3 grid ``ecutwfc = 12``
asks for is not mapped onto itself by the operations that were dropped, the
exchange-correlation potential evaluated pointwise on it breaks them, and a
second-order tensor that inversion forbids picks it up. Measured with
``get_shg(nbnd=8)`` on the whole unshifted 2x2x2 mesh: **0.7157 pm/V** on that
grid, 0.0018 on a commensurate 20^3 one (``ecutwfc = 16``), and 0.00074 with
symmetry kept for the SCF and the whole mesh passed as ``kpoints=``.

**The cut.** A band set cut inside a degenerate multiplet keeps an arbitrary
part of it, since any rotation inside the multiplet is an equally good set of
eigenvectors. On the same silicon cell at ``nbnd = 12`` (a cut 2e-14 eV wide)
the tensor inversion forbids reads **1095 pm/V**.

Each case drives the real entry point on a stand-in calculation whose
``VelocityOperator`` is replaced by a sentinel, so the run stops where the
work would start and what is checked is what the entry point said before it.
"""

from __future__ import annotations

import types
import warnings
from pathlib import Path

import equinox as eqx
import numpy as np
import pytest

import defumat.response.photocurrent as photocurrent
import defumat.response.shg as shg
from defumat.basis.builder import build_basis
from defumat.io.pwin import parse_pw_input, read_pw_input
from defumat.system.builder import build_system
from defumat.system.kpoints import KPoints

pytestmark = pytest.mark.unit

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"

#: The two entry points, with the module whose ``VelocityOperator`` each reads.
ENTRIES = [
    pytest.param(shg.second_harmonic, shg, id="second-harmonic"),
    pytest.param(photocurrent.shift_current, photocurrent, id="shift-current"),
]


class _Stop(Exception):
    """Raised where the entry point would build its velocity matrix elements."""


def _whole_mesh(system, n: int) -> KPoints:
    """The complete unshifted ``n x n x n`` grid, which no symmetry reduced."""
    axis = np.arange(n) / n
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    return KPoints.from_crystal(
        points, np.full(len(points), 1.0 / len(points)), system.cell,
        precision=system.kpoints.precision,
    )


def _silicon_nosym():
    """``si2-nosym.in`` as committed: ``nosym`` and a 15^3 dense grid."""
    return build_system(read_pw_input(CASES / "si2-nosym.in"))


def _silicon_with_symmetry():
    """The same cell and cutoff with symmetry kept, on the whole 4x4x4 mesh.

    Kept symmetry makes the SCF's own k-set a wedge, which is refused, so the
    mesh is swapped for the whole grid the way ``run_shg(kpoints=...)`` swaps
    it. This is the remedy the grid warning names.
    """
    text = (CASES / "si2-nosym.in").read_text().replace("nosym = .true.", "")
    system = build_system(parse_pw_input(text))
    assert not system.nosym
    return eqx.tree_at(lambda s: s.kpoints, system, _whole_mesh(system, 4))


def _alas_nosym():
    """``alas-raman.in``: ``nosym``, but zincblende has no fractional translation."""
    return build_system(read_pw_input(CASES / "alas-raman.in"))


def _stand_in(system):
    """What the two entry points read off a calculation before the real work."""
    grid = build_basis(system).dense.grid
    return types.SimpleNamespace(
        system=system,
        basis=types.SimpleNamespace(dense=types.SimpleNamespace(grid=grid)),
        is_ultrasoft=False, is_paw=False, is_hubbard=False, spiral=False,
    )


def _call(entry, module, monkeypatch, system, **options):
    """Run ``entry`` up to its velocity operator, and return what it warned."""
    def stop(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(module, "VelocityOperator", stop)
    nk, nb = system.kpoints.nk, 4
    states = np.zeros((1, nk, nb, 8), dtype=complex)
    energies = np.zeros((1, nk, nb))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(_Stop):
            entry(_stand_in(system), states, energies, None, **options)
    return [w for w in caught if issubclass(w.category, RuntimeWarning)]


def _about_the_grid(caught):
    return [w for w in caught if "fractional translations" in str(w.message)]


def _about_the_cut(caught):
    return [w for w in caught if "multiplet" in str(w.message)]


# --- the grid -----------------------------------------------------------------


@pytest.mark.parametrize("entry, module", ENTRIES)
def test_a_nosym_grid_the_translations_do_not_map_onto_itself_is_named(
    entry, module, monkeypatch
):
    """Silicon under ``nosym``: a 15^3 grid against a quarter-lattice translation."""
    system = _silicon_nosym()
    assert system.nosym
    caught = _about_the_grid(_call(entry, module, monkeypatch, system))
    assert len(caught) == 1
    message = str(caught[0].message)
    # The grid and the factor it misses are in the text, and so is the remedy.
    assert "(15, 15, 15)" in message and "(4, 4, 4)" in message
    assert "kpoints=" in message


@pytest.mark.parametrize("entry, module", ENTRIES)
@pytest.mark.parametrize(
    "build",
    [
        # Symmetry kept: the grid is chosen commensurate (``fft_factors``).
        pytest.param(_silicon_with_symmetry, id="silicon-with-symmetry"),
        # ``nosym``, but no operation of zincblende carries a translation.
        pytest.param(_alas_nosym, id="alas-nosym"),
    ],
)
def test_a_grid_that_breaks_nothing_is_not_warned_about(
    entry, module, build, monkeypatch
):
    caught = _about_the_grid(_call(entry, module, monkeypatch, build()))
    assert caught == []


# --- the cut ------------------------------------------------------------------


@pytest.mark.parametrize("entry, module", ENTRIES)
def test_a_cut_inside_a_multiplet_is_named(entry, module, monkeypatch):
    """A ``band_cut_gap`` below the module's degeneracy tolerance."""
    gap = 0.1 * photocurrent.DEGENERACY_TOL
    caught = _about_the_cut(
        _call(entry, module, monkeypatch, _alas_nosym(), band_cut_gap=gap)
    )
    assert len(caught) == 1
    assert "band_cut_gap" in str(caught[0].message)


@pytest.mark.parametrize("entry, module", ENTRIES)
@pytest.mark.parametrize(
    "gap",
    [
        pytest.param(10.0 * photocurrent.DEGENERACY_TOL, id="just-above-the-tolerance"),
        # A real gap at the cut, the 6.66e-2 Ry of AlAs at 14 bands on 4x4x4.
        pytest.param(6.66e-2, id="a-real-gap"),
        # What the assembly carries when nobody measured the cut.
        pytest.param(float("nan"), id="not-measured"),
    ],
)
def test_a_cut_at_a_gap_is_not_warned_about(entry, module, gap, monkeypatch):
    caught = _about_the_cut(
        _call(entry, module, monkeypatch, _alas_nosym(), band_cut_gap=gap)
    )
    assert caught == []
