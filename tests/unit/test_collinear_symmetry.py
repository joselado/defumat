"""``sgam_at_collin``: the magnetic symmetry filter for ``nspin = 2``.

The filter this file is about was absent, and its absence is a *silent* wrong
answer rather than a missing feature: a one-species antiferromagnet kept the
operation carrying one sublattice onto the other, the density symmetriser
averaged the two, and the run converged cleanly to the nonmagnetic state with a
perfectly good total energy. So the tests come in two kinds -- the guard must
fire on the case it exists for, and it must **not** fire anywhere else, because
a filter that is too aggressive gives a group that is too small and a k-wedge to
match, which is a wrong answer of the opposite sign.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from defumat.pseudo.upf import read_upf
from defumat.scf.driver import run_scf
from defumat.system.builder import system_from_file
from defumat.system.symmetry import (
    atom_mapping,
    collinear_symmetries,
    find_symmetries,
)

QE = Path(__file__).resolve().parents[1] / "data" / "qe"


def _swaps(system, symmetries) -> int:
    """How many operations send atom 0 onto atom 1."""
    mapping = np.asarray(atom_mapping(system.cell, system.structure, symmetries))
    return int(sum(1 for row in mapping if row[0] == 1))


def test_the_sublattice_swap_is_in_the_group_before_the_filter():
    """The premise, checked rather than assumed.

    If the crystal group did not contain the swap there would be nothing to
    filter and every assertion below would pass for the wrong reason. Eight of
    the sixteen operations swap the two atoms, and every one of them has a zero
    fractional translation -- so ``is_supercell``, which disables fractional
    translations and is what saves a cell whose sublattices are related by a
    pure translation, cannot help here.
    """
    system = system_from_file(QE / "h2-mirror-afm.in")
    full = find_symmetries(system.cell, system.structure)
    assert full.nsym == 16
    assert _swaps(system, full) == 8
    translations = np.asarray(full.translation_array())
    mapping = np.asarray(atom_mapping(system.cell, system.structure, full))
    swapping = [s for s in range(full.nsym) if mapping[s][0] == 1]
    assert np.allclose(translations[swapping], 0.0), "these are point operations"


def test_a_one_species_antiferromagnet_keeps_no_sublattice_swap():
    """The guard fires: the filter removes exactly the swapping operations."""
    system = system_from_file(QE / "h2-mirror-afm.in")
    filtered = system.symmetry_group()
    assert filtered.nsym == 8, "half the group swaps the sublattices"
    assert _swaps(system, filtered) == 0
    # A subgroup, not a different group: nothing new may appear.
    full = find_symmetries(system.cell, system.structure)
    assert set(filtered.rotations) <= set(full.rotations)


def test_the_filter_leaves_a_ferromagnet_alone():
    """Every operation preserves a pattern that is the same on every site."""
    system = system_from_file(QE / "h-atom-lsda.in")
    full = find_symmetries(system.cell, system.structure)
    assert system.symmetry_group().nsym == full.nsym


def test_the_moment_is_a_scalar_and_is_not_rotated():
    """The trap this filter is easiest to get wrong by.

    Without spin-orbit coupling a spatial operation does not turn the spin, so
    the collinear filter compares a **number** per site. Reusing the
    noncollinear filter -- which rotates an axial vector and carries
    ``det(R)`` -- would cut a ``C2x`` or a mirror on a z-ferromagnet, giving a
    group that is too small. Two atoms with the *same* moment must keep the
    whole crystal group even though half of it reverses a z axial vector.
    """
    system = system_from_file(QE / "h2-mirror-afm.in")
    full = find_symmetries(system.cell, system.structure)
    ferro = np.tile(np.array([0.0, 0.0, 0.6]), (2, 1))
    kept = collinear_symmetries(system.cell, system.structure, full, ferro)
    assert kept.nsym == full.nsym

    rotations = np.asarray(full.rotation_array())
    reversing = sum(1 for r in rotations if np.linalg.det(r) * r[2, 2] < 0)
    assert reversing > 0, "this cell has operations an axial filter would cut"


def test_no_moment_means_no_filtering():
    """A collinear run that is not magnetic must pay nothing for this."""
    system = system_from_file(QE / "h2-mirror-afm.in")
    full = find_symmetries(system.cell, system.structure)
    kept = collinear_symmetries(
        system.cell, system.structure, full, np.zeros((2, 3))
    )
    assert kept.rotations == full.rotations


#: The committed inputs that **are** one-species compensated magnets, so the
#: filter is *supposed* to cut them. Kept as a list rather than a prefix match
#: because the sweep below turns it into a second positive assertion: every name
#: here must lose operations, and every name not here must keep all of them.
#: ``o2-paw-afm.in`` earned its place by breaking the sweep when it was added --
#: which is the sweep working, since a new case silently joining the "unchanged"
#: side is exactly how a coverage claim goes quietly false.
DELIBERATELY_CUT = {
    "h2-mirror-afm.in",
    "h2-mirror-afm-nosym.in",
    "o2-paw-afm.in",
}


def test_the_filter_changes_nothing_on_the_committed_collinear_inputs():
    """A silent regression here would move numbers validated against ``pw.x``.

    Apart from the handful in :data:`DELIBERATELY_CUT`, none of the committed
    collinear cases is a one-species compensated magnet -- precisely because
    that spelling did not work -- so the filter must be a no-op on all of them.
    """
    expected_to_cut = []
    unchanged, cut, unparsed = [], [], []
    for path in sorted(QE.glob("*.in")):
        if path.name in DELIBERATELY_CUT:
            expected_to_cut.append(path)
            continue
        try:
            system = system_from_file(path)
        except (ValueError, NotImplementedError, FileNotFoundError, KeyError) as why:
            # A handful of committed inputs are refusal cases and are *meant*
            # not to build. Anything else is a parse failure that would remove
            # an input from this sweep silently, which is the failure mode this
            # test exists to prevent one level up.
            unparsed.append(f"{path.name}: {type(why).__name__}")
            continue
        if system.nspin != 2:
            continue
        full = find_symmetries(system.cell, system.structure)
        (cut if system.symmetry_group().nsym != full.nsym else unchanged).append(
            path.name
        )
    # A floor, not a truthiness check: the sweep silently shrinking is exactly
    # how a coverage assertion stops covering anything.
    assert len(unchanged) >= 20, (
        f"only {len(unchanged)} collinear inputs were examined (expected at "
        f"least 20); inputs that did not build: {unparsed}"
    )
    assert not cut, f"the filter cut operations on {cut}"

    # The other half: the exemptions are exemptions and not a quiet allowlist.
    # ``nosym`` inputs have a group of one and nothing to cut, so they are only
    # required not to be *larger* than the filter would leave them.
    for path in expected_to_cut:
        system = system_from_file(path)
        full = find_symmetries(system.cell, system.structure)
        if system.nosym:
            continue
        assert system.symmetry_group().nsym < full.nsym, (
            f"{path.name} is listed as a case the filter cuts and it did not"
        )


@pytest.mark.slow
def test_the_symmetrised_run_now_reaches_the_antiferromagnet(pseudo_dir):
    """The number the fix is worth, against the spelling that always worked.

    ``nosym = .true.`` was the only way to converge this state. The two runs
    must now agree -- on the energy, on the site moments and on the iteration
    count -- while the symmetric one still *uses* the eight operations that
    survive rather than falling back to the identity.
    """
    results = {}
    for name in ("h2-mirror-afm", "h2-mirror-afm-nosym"):
        system = system_from_file(QE / f"{name}.in")
        pseudos = tuple(
            read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
        )
        results[name] = run_scf(system, pseudos, conv_thr=1e-8, verbose=False)

    symmetric, free = results["h2-mirror-afm"], results["h2-mirror-afm-nosym"]
    assert symmetric.total_energy == pytest.approx(free.total_energy, abs=1e-8)
    assert np.asarray(symmetric.site_moments) == pytest.approx(
        np.asarray(free.site_moments), abs=1e-5
    )
    # The state is the antiferromagnet and not the nonmagnetic collapse: the
    # cell total is zero for both of those and the site moments are not.
    moments = np.asarray(symmetric.site_moments)[:, 0]
    assert symmetric.magnetization == pytest.approx(0.0, abs=1e-4)
    assert abs(moments[0]) > 0.2 and moments[0] == pytest.approx(-moments[1], abs=1e-5)
    assert symmetric.iterations == free.iterations
    assert system_from_file(QE / "h2-mirror-afm.in").symmetry_group().nsym == 8
