"""The ultracell's three guards: the cut, the reference grid, the occupations.

Nothing here runs an SCF or a diagonalisation. Each guard is a statement about
a number the loop would otherwise have computed wrongly without saying so, and
each is decided before the frozen solve -- which is why each can be tested on
the host with a stand-in for the ground state.

* **The multiplet cut** is the gap *across* the ``nbnd`` truncation,
  ``eps[nbnd] - eps[nbnd-1]``. It used to be the spacing of the last two
  retained bands, which is a different number and fails in both directions on
  silicon's ``Gamma`` spectrum.
* **The reference grid** must be the unshifted, unreduced ``supercell * kgrid``
  Monkhorst-Pack grid the folded set ``k0 + Q`` is, or the tiled density is not
  a fixed point and a different Brillouin-zone sampling reads as a modulation
  (0.635 Ry per cell on silicon, ``4 2 2`` folded onto ``2 1 1``).
* **The occupations** have a fixed and a smeared branch and nothing else, and a
  tetrahedron run used to fall into the smeared one at ``degauss = 0``.
"""

from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import numpy as np
import pytest

from defumat import Calculator
from defumat.basis.builder import build_basis
from defumat.system.kpoints import KPoints
from defumat.ultracell.driver import (
    require_an_ultracell_regime,
    require_the_folded_grid,
    run_ultracell,
)
from defumat.ultracell.hamiltonian import multiplet_cut

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests" / "data" / "qe"

#: Silicon's lowest eight levels at ``Gamma`` in Ry, rounded: ``Gamma_1``, the
#: ``Gamma_25'`` triplet at the valence band top, the ``Gamma_15`` triplet and
#: ``Gamma_2'``. Four bands hold the electrons.
GAMMA = np.array([-0.394, 0.499, 0.499, 0.499, 0.672, 0.672, 0.672, 0.758])


@pytest.fixture(scope="module")
def silicon(pseudo_dir):
    """``si-ultracell.in``: two atoms, nosym, ``K_POINTS automatic 8 2 2 0 0 0``."""
    return Calculator.from_file(DATA / "si-ultracell.in", pseudo_dir=pseudo_dir)


def _converged():
    # Everything run_ultracell reads off a ground state before the frozen
    # solve, and nothing after it: a guard that fires is reached, and one that
    # does not would fail on the missing density rather than run an SCF.
    return SimpleNamespace(converged=True, magnetic_field=None)


# -- the cut ------------------------------------------------------------------

def test_a_cut_through_a_triplet_reads_as_degenerate():
    """``nbnd = 5`` and ``6`` split ``Gamma_15``: the gap across the cut is zero.

    The retained-band spacing at ``nbnd = 5`` is the 0.17 Ry gap below the
    triplet, which is what the first form of the check reported. ``nbnd = 7``
    closes the triplet and cuts in the gap below ``Gamma_2'``.
    """
    assert multiplet_cut(GAMMA, 5) == pytest.approx(0.0, abs=1e-12)
    assert multiplet_cut(GAMMA, 6) == pytest.approx(0.0, abs=1e-12)
    assert multiplet_cut(GAMMA, 7) == pytest.approx(0.758 - 0.672, abs=1e-12)


def test_a_cut_in_a_gap_reads_as_the_gap():
    """``nbnd = 4`` cuts between two triplets, in the 0.173 Ry gap.

    The retained-band spacing there is inside ``Gamma_25'`` and is zero, which
    is the other way round from the truth.
    """
    assert multiplet_cut(GAMMA, 4) == pytest.approx(0.672 - 0.499, abs=1e-12)
    assert multiplet_cut(GAMMA, 1) == pytest.approx(0.499 + 0.394, abs=1e-12)


def test_the_cut_is_the_tightest_over_the_k_set():
    """The minimum over every leading axis -- channels, ``k0``, ``Q``."""
    shifted = GAMMA + np.array([0, 0, 0, 0, 0.05, 0.05, 0.05, 0])
    stack = np.stack([GAMMA, shifted])[None, :, None, :]
    assert multiplet_cut(stack, 4) == pytest.approx(0.672 - 0.499, abs=1e-12)
    assert multiplet_cut(stack[:, 1:], 4) == pytest.approx(0.722 - 0.499, abs=1e-12)


def test_the_cut_needs_the_band_beyond_it():
    """Without band ``nbnd + 1`` there is no gap across the cut to measure."""
    with pytest.raises(ValueError, match="one band more"):
        multiplet_cut(GAMMA[:5], 5)


# -- the reference grid -------------------------------------------------------

def test_a_mismatched_reference_grid_is_refused(silicon):
    """``8 2 2`` converged, ``(2, 1, 1) x (1, 1, 1)`` folded: a different zone."""
    with pytest.raises(ValueError, match=r"supercell \* kgrid"):
        run_ultracell(silicon.system, silicon.pseudos, _converged(),
                      (2, 1, 1), (1, 1, 1), nbnd=8)
    with pytest.raises(ValueError, match=r"supercell \* kgrid"):
        run_ultracell(silicon.system, silicon.pseudos, _converged(),
                      (4, 1, 1), (1, 2, 2), nbnd=8)


@pytest.mark.parametrize("supercell,kgrid", [((8, 1, 1), (1, 2, 2)),
                                             ((4, 1, 1), (2, 2, 2)),
                                             ((2, 2, 1), (4, 1, 2)),
                                             ((1, 1, 1), (8, 2, 2))])
def test_every_factorisation_of_the_reference_grid_passes(silicon, supercell,
                                                          kgrid):
    require_the_folded_grid(silicon.system, supercell, kgrid)


def test_a_shifted_or_reduced_grid_is_refused(silicon):
    """Same ``8 2 2`` count, different points or weights standing for others."""
    system, cell = silicon.system, silicon.system.cell
    shifted = eqx.tree_at(lambda s: s.kpoints, system,
                          KPoints.automatic((8, 2, 2), (1, 1, 1), cell))
    with pytest.raises(ValueError, match="not that set"):
        require_the_folded_grid(shifted, (8, 1, 1), (1, 2, 2))

    whole = np.asarray(system.kpoints.crystal(cell))
    wedge = eqx.tree_at(lambda s: s.kpoints, system, KPoints.from_crystal(
        whole[: len(whole) // 2], np.full(len(whole) // 2, 2.0), cell,
        grid=(8, 2, 2), shift=(0, 0, 0), reduced=True))
    with pytest.raises(ValueError, match="symmetry-reduced wedge"):
        require_the_folded_grid(wedge, (8, 1, 1), (1, 2, 2))


def test_an_explicit_list_is_judged_by_its_points(silicon):
    """The folded grid written out, reordered and moved by ``G``, passes."""
    system, cell = silicon.system, silicon.system.cell
    j = np.stack(np.meshgrid(*[np.arange(n) for n in (8, 2, 2)], indexing="ij"),
                 axis=-1).reshape(-1, 3)
    points = j / np.array([8.0, 2.0, 2.0])
    points = points[::-1] - np.array([1.0, 0.0, 1.0])
    listed = eqx.tree_at(lambda s: s.kpoints, system, KPoints.from_crystal(
        points, np.ones(len(points)), cell))
    require_the_folded_grid(listed, (8, 1, 1), (1, 2, 2))

    # The right points at unequal weights: the weight spread is what
    # ``is_reduced`` reads as a wedge, so it is that refusal which fires, and it
    # is the only uniformity test the guard needs.
    uneven = eqx.tree_at(lambda s: s.kpoints, system, KPoints.from_crystal(
        points, np.linspace(1.0, 2.0, len(points)), cell))
    with pytest.raises(ValueError, match="symmetry-reduced wedge"):
        require_the_folded_grid(uneven, (8, 1, 1), (1, 2, 2))


# -- the occupations ----------------------------------------------------------

class _WithOccupations:
    """The real system with ``occupations`` read as something else."""

    def __init__(self, system, occupations):
        self._system, self.occupations = system, occupations

    def __getattr__(self, name):
        return getattr(self._system, name)


@pytest.mark.parametrize("occupations", ["tetrahedra", "tetrahedra_opt",
                                         "tetrahedra-lin"])
def test_tetrahedra_are_refused(silicon, occupations):
    system = _WithOccupations(silicon.system, occupations)
    with pytest.raises(NotImplementedError, match="tetrahedron"):
        require_an_ultracell_regime(system, silicon.pseudos,
                                    build_basis(silicon.system))


def test_occupations_from_input_are_refused(silicon):
    system = _WithOccupations(silicon.system, "from_input")
    with pytest.raises(NotImplementedError, match="from_input"):
        require_an_ultracell_regime(system, silicon.pseudos,
                                    build_basis(silicon.system))


def test_a_parsed_tetrahedron_input_is_refused(tmp_path, pseudo_dir):
    """The same refusal through the input file, not only through a stand-in."""
    text = (DATA / "si-ultracell.in").read_text().replace(
        "nosym = .true., noinv = .true.",
        "nosym = .true., noinv = .true., occupations = 'tetrahedra'")
    path = tmp_path / "si-ultracell-tetra.in"
    path.write_text(text)
    calculator = Calculator.from_file(path, pseudo_dir=pseudo_dir)
    assert calculator.system.occupations == "tetrahedra"
    with pytest.raises(NotImplementedError, match="tetrahedron"):
        run_ultracell(calculator.system, calculator.pseudos, _converged(),
                      (8, 1, 1), (1, 2, 2), nbnd=8)


@pytest.mark.parametrize("occupations", ["fixed", "smearing", None])
def test_the_two_implemented_schemes_pass(silicon, occupations):
    system = _WithOccupations(silicon.system, occupations)
    require_an_ultracell_regime(system, silicon.pseudos,
                                build_basis(silicon.system))
