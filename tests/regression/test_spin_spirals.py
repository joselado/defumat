"""P19: spin spirals, validated against calculations that are not spirals.

There is no reference implementation to compare against -- ``pw.x`` has no spin
spiral, and a grep of the vendored tree finds nothing -- so this is P16's
situation: the validation has to come from identities. Three of them reduce a
spiral to a calculation this repository has already validated, and they are the
whole case that the machinery is right:

1. **``q = 0``** is not a spiral at all: the two components' spheres coincide,
   and the answer must be the ordinary noncollinear one -- which P17 in turn
   ties to the collinear LSDA path validated in P9.
2. **``q = b3/2``** is the antiferromagnet of the cell doubled along ``z``,
   computed here by the *collinear* path in that doubled cell.
3. **``q = b3/4``** is the 90-degree noncollinear supercell of the cell
   quadrupled along ``z``.

The last two are the sharp ones: they pin the two shifted spheres, the sign of
the ``q/2`` split, the cross term between components on *different* spheres and
the rotated-frame potential all at once, and nothing else does.

**What is compared is the energy without the Ewald term.** Two cells of
different size do not compute the same Ewald sum to better than QE's own
``upperbound`` tolerance of 1e-7 Ry (``ewald.f90``, transcribed in
:func:`defumat.scf.ewald.ewald_alpha`), and that difference -- about 8e-8 Ry
here -- would otherwise swamp the identity, which holds to **1e-10**. The Ewald
sums are checked against each other separately, at the tolerance they deserve.

**Both sides must land in the same state**, which for a metal with a soft
magnetic surface is a real condition rather than a formality: at ``degauss =
0.02`` the spiral and its supercell converge to different minima and disagree in
the fourth decimal. See the header of ``tests/data/qe/h-chain-spiral.in``. The
same caveat is why ``E(q + 2G) = E(q)`` is *not* asserted here even though it is
exact: at ``q + 2G`` the magnetic solution lives in a gauge whose transverse
magnetization winds twice across the cell, the uniform starting guess has little
overlap with it, and the SCF converges to the nonmagnetic minimum instead --
measured, on this chain, as a moment of 5e-6 where the same physics at ``q``
gives 0.54.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from defumat.io.pwin import parse_pw_input, read_pw_input
from defumat.pseudo import read_upf
from defumat.scf import run_scf
from defumat.system import build_system
from tests.conftest import GENERATED

pytestmark = [pytest.mark.regression, pytest.mark.slow]

#: Two calculations of the same state by different routes. Both numbers come
#: from this code, so the tolerance is a round-off one.
IDENTITY_RY = 1e-9


def _pseudos(system, pseudo_dir: Path):
    return tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)


@lru_cache(maxsize=None)
def _run(text: str, pseudo_dir: Path):
    """SCF on an input given as text, so a test can vary ``spiral_q`` in place."""
    pwin = parse_pw_input(text)
    system = build_system(pwin)
    return system, run_scf(
        system,
        _pseudos(system, pseudo_dir),
        conv_thr=float(pwin.get("electrons", "conv_thr") or 1e-10),
        mixing_beta=float(pwin.get("electrons", "mixing_beta") or 0.7),
        max_iterations=200,
    )


def _spiral_input(q3: float, grid: str | None = None) -> str:
    text = (GENERATED / "h-chain-spiral.in").read_text()
    text = text.replace("spiral_q(3) = 0.25", f"spiral_q(3) = {q3}")
    if grid is not None:
        text = text.replace(" 1 1 4 0 0 0", grid)
    return text


def _electronic(result) -> float:
    """The total energy without the Ewald term, per unit cell of the input."""
    return result.total_energy - result.energy_terms["ewald"]


def test_a_zero_spiral_is_an_ordinary_noncollinear_run(pseudo_dir):
    """``q = 0``: the two spheres coincide and nothing should change.

    The cheapest of the three identities and the one that isolates the *plumbing*
    -- the doubled k-list, the per-component projectors, the per-component FFT --
    from the physics, since at ``q = 0`` all of it must collapse back to what
    P17 already validated.
    """
    text = (GENERATED / "h-atom-lsda.in").read_text()
    noncollinear = text.replace(
        "    nspin = 2",
        "    noncolin = .true.\n    nosym = .true.\n    angle1(1) = 90.0",
    )
    spiral = noncollinear.replace(
        "    nosym = .true.",
        "    nosym = .true.\n    spiral_q(1) = 0.0, spiral_q(2) = 0.0, spiral_q(3) = 0.0",
    )

    plain_system, plain = _run(noncollinear, pseudo_dir)
    system, result = _run(spiral, pseudo_dir)

    assert system.spiral and not plain_system.spiral
    assert result.converged and plain.converged
    assert result.total_energy == pytest.approx(plain.total_energy, abs=1e-10)


def test_half_a_reciprocal_vector_is_the_antiferromagnet(pseudo_dir):
    """``q = b3/2`` against the collinear LSDA antiferromagnet of the doubled cell."""
    system, spiral = _run(_spiral_input(0.5), pseudo_dir)
    doubled_system, doubled = _run(
        (GENERATED / "h-chain-afm.in").read_text(), pseudo_dir
    )

    assert spiral.converged and doubled.converged
    assert doubled_system.nspin == 2  # the reference is not a spiral at all
    assert _electronic(spiral) == pytest.approx(_electronic(doubled) / 2.0, abs=IDENTITY_RY)
    # The Ewald sums are the same physical lattice computed in two cells, and
    # agree only to QE's own truncation tolerance.
    assert spiral.energy_terms["ewald"] == pytest.approx(
        doubled.energy_terms["ewald"] / 2.0, abs=1e-6
    )


def test_a_quarter_turn_is_the_ninety_degree_supercell(pseudo_dir):
    """``q = b3/4`` against a four-cell noncollinear supercell.

    The strongest of the three: ``q`` is not commensurate with the k-grid, so
    both components' spheres are centred *off* the grid, and the supercell it is
    compared against is a genuinely noncollinear calculation with four different
    moment directions.
    """
    system, spiral = _run(_spiral_input(0.25), pseudo_dir)
    supercell_system, supercell = _run(
        (GENERATED / "h-chain-90deg.in").read_text(), pseudo_dir
    )

    assert spiral.converged and supercell.converged
    assert supercell_system.nspin_mag == 4 and not supercell_system.spiral
    assert _electronic(spiral) == pytest.approx(
        _electronic(supercell) / 4.0, abs=IDENTITY_RY
    )


@pytest.mark.parametrize("q3", [0.25, 0.5])
def test_the_energy_is_even_in_q(q3, pseudo_dir):
    """``E(-q) = E(q)``: the two spirals are mirror images, so this is exact."""
    _, forward = _run(_spiral_input(q3), pseudo_dir)
    _, backward = _run(_spiral_input(-q3), pseudo_dir)
    assert forward.converged and backward.converged
    assert backward.total_energy == pytest.approx(forward.total_energy, abs=1e-11)


@pytest.mark.parametrize("q3", [0.0, 0.25])
def test_q_is_periodic_in_the_reciprocal_lattice(q3, pseudo_dir):
    """``E(q + G) = E(q)`` -- but only on a k-grid invariant under a ``G/2`` shift.

    Adding ``G`` to ``q`` moves the up component's sphere by ``+G/2`` and the
    down's by ``-G/2``, which is the same calculation with every ``k`` shifted
    by ``G/2``. The *sum over the Brillouin zone* is unchanged only if the k-set
    is invariant under that shift, which an even Monkhorst-Pack grid is and an
    odd one is not -- measured: 2e-9 on the 1x1x4 grid this input uses, and
    2e-3 on a 1x1x3 grid.

    **The moment does not come back the same, and that is the point.** The two
    calculations differ by relabelling ``G`` in each component, which multiplies
    the rotated-frame transverse magnetization by a lattice-periodic phase --
    a gauge transformation. Its *modulus* is unchanged pointwise, so the LDA
    energy is; its integral over the cell is not, and here it is 0.540 against
    0.205 for the identical energy. Asserting both is what makes this a test of
    the gauge structure rather than of two runs having happened to agree.
    """
    _, base = _run(_spiral_input(q3), pseudo_dir)
    _, shifted = _run(_spiral_input(q3 + 1.0), pseudo_dir)

    assert base.converged and shifted.converged
    assert shifted.total_energy == pytest.approx(base.total_energy, abs=1e-8)
    if q3:
        # The gauge phase is trivial at q = 0, where there is nothing to wind.
        assert not np.isclose(
            np.linalg.norm(shifted.magnetization_vector),
            np.linalg.norm(base.magnetization_vector),
            atol=1e-3,
        )


def test_the_scan_workflow_reproduces_single_runs(pseudo_dir):
    """``run_spiral_scan`` shares everything that does not depend on ``q``.

    The check is that sharing changes nothing: the scan's energies must be the
    ones separate runs give, to round-off, and its ``at_spiral_q`` must leave the
    dense G set and the local potential as the *same objects*.
    """
    from defumat.scf.driver import Calculation
    from defumat.workflows.spiral import run_spiral_scan

    text = _spiral_input(0.25)
    system = build_system(parse_pw_input(text))
    pseudos = _pseudos(system, pseudo_dir)

    scan = run_spiral_scan(
        system, pseudos, [[0.0, 0.0, 0.25], [0.0, 0.0, 0.5]],
        conv_thr=1e-11, mixing_beta=0.3, max_iterations=200,
    )
    assert all(scan.converged)

    _, quarter = _run(_spiral_input(0.25), pseudo_dir)
    _, half = _run(_spiral_input(0.5), pseudo_dir)
    assert scan.energies[0] == pytest.approx(quarter.total_energy, abs=1e-10)
    assert scan.energies[1] == pytest.approx(half.total_energy, abs=1e-10)

    base = Calculation(system, pseudos)
    moved = base.at_spiral_q([0.0, 0.0, 0.5])
    assert moved.basis.dense is base.basis.dense
    assert moved.vltot is base.vltot
    assert moved.basis.planewaves is not base.basis.planewaves


def test_what_a_spiral_refuses(pseudo_dir, qe_testsuite):
    """The three combinations that are refused, and why each one is.

    Spin-orbit coupling: permanently, because it breaks the generalized Bloch
    theorem. Symmetry: until the spin space group is written. Ultrasoft and PAW:
    until the augmentation charge between the two components -- ``q_ij(q)``, not
    ``qq`` -- is threaded through.
    """
    text = _spiral_input(0.25)

    with pytest.raises(ValueError, match="lspinorb"):
        build_system(parse_pw_input(text.replace(
            "    nosym = .true.", "    nosym = .true.\n    lspinorb = .true."
        )))

    with pytest.raises(ValueError, match="nosym"):
        build_system(parse_pw_input(text.replace("    nosym = .true.", "")))

    with pytest.raises(ValueError, match="noncolin"):
        build_system(parse_pw_input(text.replace("    noncolin = .true.", "")))

    # ... and an ultrasoft dataset is refused when the calculation is built,
    # which is where the augmentation charge first exists.
    from defumat.scf.driver import Calculation

    ultrasoft = read_pw_input(qe_testsuite / "pw_noncolin" / "noncolin.in")
    marker = ultrasoft.namelists["system"]
    marker["nosym"] = ".true."
    marker["spiral_q"] = {(3,): 0.5}
    system = build_system(ultrasoft)
    with pytest.raises(NotImplementedError, match="q_ij"):
        Calculation(system, _pseudos(system, pseudo_dir))
