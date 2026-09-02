"""``Calculation.at_kpoints``: a new k-list must not change anything else.

The counterpart of ``at_positions`` on the other axis. Almost nothing in a
calculation depends on which k-points were asked for -- both G-vector sets, the
FFT dimensions, the local potential, the augmentation charge, the Ewald sum, the
symmetry group and the radial tables are properties of the cell and the atoms --
so a k-list can be replaced by rebuilding only what carries a ``k`` index.

Two things are checked, and they fail differently. That the result is *identical*
to a calculation built from scratch at those k-points is correctness: anything
stale carried across would show up as a wrong ``vkb(k)`` or a wrong ``|k+G|^2``,
and a Berry phase built on it would come back a plausible non-integer rather than
an error. That the shared parts are the *same objects* is the point of the
method: it is what makes streaming a k-mesh cost one diagonalisation per row
instead of a gigabyte and seventy seconds (see :mod:`defumat.workflows.topology`).
"""

import numpy as np
import pytest

import equinox as eqx

from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.scf.driver import Calculation
from defumat.system import build_system
from defumat.system.kpoints import KPoints

pytestmark = pytest.mark.unit

#: A k-list unrelated to the input's own, including a point on the zone boundary
#: and one at the origin -- the two places a plane-wave sphere changes size.
POINTS = np.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])


def _calculation(qe_testsuite, pseudo_dir):
    system = build_system(read_pw_input(qe_testsuite / "pw_scf" / "scf.in"))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    return system, pseudos


def _kpoints(system):
    return KPoints.from_crystal(
        POINTS, np.full(len(POINTS), 1.0 / len(POINTS)), system.cell,
        precision=system.kpoints.precision,
    )


def test_it_is_the_calculation_that_would_have_been_built(qe_testsuite, pseudo_dir):
    """Every k-dependent array, against one built from scratch at those points."""
    system, pseudos = _calculation(qe_testsuite, pseudo_dir)
    kpoints = _kpoints(system)

    shared = Calculation(system, pseudos).at_kpoints(kpoints)
    fresh = Calculation(eqx.tree_at(lambda s: s.kpoints, system, kpoints), pseudos)

    for name, a, b in [
        ("kinetic", shared.kinetic, fresh.kinetic),
        ("fft_index", shared.fft_index, fresh.fft_index),
        ("mask", shared.basis.planewaves.mask, fresh.basis.planewaves.mask),
        ("vkb", shared.projectors.vkb, fresh.projectors.vkb),
        ("sticks", shared.sticks.index, fresh.sticks.index),
        ("stick columns", shared.sticks.columns, fresh.sticks.columns),
    ]:
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape, f"{name}: {a.shape} != {b.shape}"
        # Identical, not merely close: the same arithmetic on the same inputs.
        np.testing.assert_array_equal(a, b, err_msg=name)


def test_the_k_independent_setup_is_untouched(qe_testsuite, pseudo_dir):
    """The other half of the statement: what a k-list does *not* affect."""
    system, pseudos = _calculation(qe_testsuite, pseudo_dir)
    fresh = Calculation(eqx.tree_at(lambda s: s.kpoints, system, _kpoints(system)), pseudos)
    base = Calculation(system, pseudos)
    moved = base.at_kpoints(_kpoints(system))

    np.testing.assert_array_equal(np.asarray(moved.vltot), np.asarray(fresh.vltot))
    assert float(moved.ewald) == pytest.approx(float(fresh.ewald), abs=1e-12)
    assert moved.basis.dense.grid == fresh.basis.dense.grid
    assert moved.basis.dense.ngm == fresh.basis.dense.ngm


def test_the_shared_parts_are_shared_not_copied(qe_testsuite, pseudo_dir):
    """Object identity, because that is what the memory claim rests on.

    Equality would pass just as well on a full rebuild, which is the thing this
    method exists to avoid -- so the assertion has to be ``is``.
    """
    system, pseudos = _calculation(qe_testsuite, pseudo_dir)
    base = Calculation(system, pseudos)
    one = base.at_kpoints(_kpoints(system))
    two = base.at_kpoints(_kpoints(system))

    assert one.basis.dense is base.basis.dense
    assert one.basis.smooth is base.basis.smooth
    assert one.basis.dense is two.basis.dense
    assert one.vltot is base.vltot
    assert one.symmetries is base.symmetries
    assert one.augmentation is base.augmentation  # None here, shared when not


def test_the_original_is_not_mutated(qe_testsuite, pseudo_dir):
    """``at_kpoints`` returns a new calculation; the one it came from is intact."""
    system, pseudos = _calculation(qe_testsuite, pseudo_dir)
    base = Calculation(system, pseudos)
    before_nk = base.kinetic.shape[0]
    before_kinetic = np.asarray(base.kinetic).copy()

    moved = base.at_kpoints(_kpoints(system))

    assert moved.kinetic.shape[0] == len(POINTS)
    assert base.kinetic.shape[0] == before_nk
    np.testing.assert_array_equal(np.asarray(base.kinetic), before_kinetic)
