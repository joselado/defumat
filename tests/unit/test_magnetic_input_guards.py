"""Five magnetic inputs that ran and returned the wrong physics without saying so.

Each is a host-side check on :func:`build_system`; nothing here runs an SCF.

* ``constrained_magnetization = 'total direction'`` at ``nspin = 2``: the polar
  angle of a one-component moment is 0 or 180 degrees whatever the density
  does, so the penalty is a constant. ``add_bfield.f90:177-180`` stops on it.
* ``tot_magnetization`` at ``nspin = 1`` or with ``noncolin``: stored and read
  by nothing, since the two Fermi levels exist only for ``nspin = 2``.
  ``input.f90:778-782`` stops on it.
* A uniform ``B_field`` off the moment's axis left the magnetic group whole, so
  the density was symmetrised with rotations that turn the field.
* A staggered ``LOCAL_MAGNETIC_FIELDS`` card at ``nspin = 2`` left the operations
  swapping the two sites, so the staggered polarisation it drives was averaged
  back to zero.
* A partly given ``r_m`` left the other species with a sphere of radius zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from defumat.io.pwin import parse_pw_input
from defumat.system.builder import build_system
from defumat.system.kpoints import KPoints
from defumat.system.symmetry import (
    atom_mapping,
    cartesian_rotations,
    find_symmetries,
)

pytestmark = pytest.mark.unit


def _build(text: str):
    return build_system(parse_pw_input(text))


# -- 'total direction' needs a vector moment ------------------------------------

_H2_LSDA = """
&control
  calculation = 'scf'
/
&system
  ibrav = 6, celldm(1) = 12.0, celldm(3) = 0.83333333333333,
  nat = 2, ntyp = 1, ecutwfc = 25.0,
  occupations = 'smearing', smearing = 'gaussian', degauss = 0.10
  nspin = 2
  {extra}
/
&electrons
/
ATOMIC_SPECIES
 H  1.008  H.pz-vbc.UPF
ATOMIC_POSITIONS (crystal)
 H 0.0 0.0 0.2
 H 0.0 0.0 0.8
K_POINTS {{automatic}}
 2 2 4 0 0 0
{cards}"""


def _h2(extra="starting_magnetization(1) = 0.0", cards=""):
    return _H2_LSDA.format(extra=extra, cards=cards)


def test_total_direction_is_refused_for_a_collinear_run():
    """``i_cons = 6`` constrains the polar angle of the total moment.

    A collinear total has one component, so ``m_z/|m|`` is +-1 and the
    penalty has no gradient: the run would converge to the unconstrained state
    and report the constraint as held. ``pw.x`` stops in ``add_bfield.f90``.
    """
    text = _h2(
        "starting_magnetization(1) = 0.5\n"
        "  constrained_magnetization = 'total direction', lambda = 1.0\n"
        "  fixed_magnetization(3) = 30.0"
    )
    with pytest.raises(ValueError, match="'total direction' requires noncolin"):
        _build(text)


# -- tot_magnetization needs two collinear channels ------------------------------

_SILICON = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.2, nat = 2, ntyp = 1, ecutwfc = 12.0
  occupations = 'smearing', degauss = 0.02
  {extra}
/
&electrons
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS crystal
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""


@pytest.mark.parametrize(
    "extra",
    [
        "tot_magnetization = 1.0",
        "nspin = 1, tot_magnetization = 1.0",
        "noncolin = .true., starting_magnetization(1) = 0.3, "
        "tot_magnetization = 1.0",
    ],
    ids=["nspin unset", "nspin = 1", "noncolin"],
)
def test_tot_magnetization_needs_nspin_2(extra):
    """A fixed ``N_up - N_down`` without an up and a down channel to fill.

    ``Calculation.two_fermi_energies`` is false for all three, so the value was
    carried on the :class:`System` and the run was the unconstrained one.
    """
    with pytest.raises(ValueError, match="tot_magnetization requires nspin = 2"):
        _build(_SILICON.format(extra=extra))


def test_tot_magnetization_is_still_accepted_at_nspin_2():
    """The negative: the collinear run it is meant for builds and keeps it."""
    system = _build(_SILICON.format(
        extra="nspin = 2, starting_magnetization(1) = 0.1, tot_magnetization = 0.0"
    ))
    assert system.nspin == 2
    assert system.tot_magnetization == 0.0


# -- a uniform B_field is an axial vector on every atom --------------------------

_FE_BCC = """
&control
  calculation = 'scf'
/
&system
  ibrav = 3, celldm(1) = 5.42, nat = 1, ntyp = 1, ecutwfc = 25.0,
  occupations = 'smearing', smearing = 'mv', degauss = 0.02
  noncolin = .true., starting_magnetization(1) = 0.5
  {extra}
/
&electrons
/
ATOMIC_SPECIES
 Fe 55.845 Fe.pz-nd-rrkjus.UPF
ATOMIC_POSITIONS alat
 Fe 0.0 0.0 0.0
K_POINTS automatic
 4 4 4 0 0 0
"""


def _field_images(system, field):
    """``det(R) R B`` per kept operation, with the ``t_rev`` sign taken off."""
    group = system.symmetry_group()
    rotations = cartesian_rotations(system.cell, group)
    signs = np.where(np.asarray(group.t_rev_array()) == 1, -1.0, 1.0)
    determinants = np.sign(np.linalg.det(rotations))
    return group, (determinants * signs)[:, None] * (rotations @ field)


def test_a_uniform_field_off_the_moment_cuts_the_magnetic_group():
    """bcc Fe along z with ``B_field(1) = 0.01``: the field must be a symmetry too.

    Every kept operation has to map the field onto itself, with the same time
    reversal it applies to the moment. Before the fix the group was the
    z-ferromagnet's 16, ``C4z`` among them, which carries ``B_x`` onto ``B_y``.
    """
    plain = _build(_FE_BCC.format(extra=""))
    assert plain.symmetry_group().nsym == 16, "the premise: a z-ferromagnet's group"
    field = np.array([0.01, 0.0, 0.0])
    tilted = _build(_FE_BCC.format(extra="B_field(1) = 0.01"))
    assert tilted.nspin_mag == 4

    group, images = _field_images(tilted, field)
    np.testing.assert_allclose(images, np.tile(field, (group.nsym, 1)), atol=1e-12)
    assert group.nsym < plain.symmetry_group().nsym
    # A subgroup, nothing new.
    assert set(group.rotations) <= set(plain.symmetry_group().rotations)

    # The build reduced its k-set with the same group the density is
    # symmetrised with: a magnetic noncollinear run has no -k = k.
    expected = KPoints.automatic(
        (4, 4, 4), (0, 0, 0), tilted.cell,
        rotations=group.rotation_array(), time_reversal=False,
        t_rev=group.t_rev_array(),
    )
    assert len(np.asarray(tilted.kpoints.weights)) == len(
        np.asarray(expected.weights))


def test_with_b_field_rebuilds_the_k_set_for_the_smaller_group():
    """A field put on after the build must sample the zone the field's group does.

    ``dataclasses.replace(system, b_field=...)`` keeps the field-free wedge
    while :meth:`System.symmetry_group` now filters with the field, so the two
    would disagree; :meth:`System.with_b_field` rebuilds the k-set, and must
    land on exactly what building the input with the field gives.
    """
    plain = _build(_FE_BCC.format(extra=""))
    moved = plain.with_b_field((0.01, 0.0, 0.0))
    built = _build(_FE_BCC.format(extra="B_field(1) = 0.01"))
    assert moved.symmetry_group().rotations == built.symmetry_group().rotations
    np.testing.assert_allclose(
        np.asarray(moved.kpoints.weights), np.asarray(built.kpoints.weights)
    )
    assert len(np.asarray(moved.kpoints.weights)) > len(
        np.asarray(plain.kpoints.weights))


def test_a_field_along_the_moment_leaves_the_group_alone():
    """The negative: a field parallel to ``m`` breaks nothing ``m`` did not."""
    plain = _build(_FE_BCC.format(extra=""))
    parallel = _build(_FE_BCC.format(extra="B_field(3) = 0.01"))
    assert parallel.symmetry_group().nsym == plain.symmetry_group().nsym


# -- a staggered LOCAL_MAGNETIC_FIELDS card at nspin = 2 -------------------------

def _card(bz1, bz2, fmt=".8f"):
    return (f"LOCAL_MAGNETIC_FIELDS\n 0.0 0.0 {bz1:{fmt}}\n"
            f" 0.0 0.0 {bz2:{fmt}}\n")


def _swaps(system, group) -> int:
    mapping = np.asarray(atom_mapping(system.cell, system.structure, group))
    return int(sum(1 for row in mapping if row[0] == 1))


def test_a_staggered_local_field_removes_the_sublattice_swap():
    """Two H atoms, no starting moment, fields ``+-0.05`` along z.

    An operation that carries the ``+b`` site onto the ``-b`` one is not a
    symmetry of the Hamiltonian, and symmetrising with it sets the staggered
    polarisation the field drives to zero. Before the fix all 16 operations were
    kept and 8 of them swapped the atoms.
    """
    bare = _build(_h2())
    full = find_symmetries(bare.cell, bare.structure)
    assert full.nsym == 16
    assert _swaps(bare, full) == 8, "the premise: the swap is there"

    system = _build(_h2(cards=_card(0.05, -0.05)))
    group = system.symmetry_group()
    assert _swaps(system, group) == 0
    assert group.nsym == 8
    assert set(group.rotations) <= set(full.rotations)

    # The build's k-set is reduced with the same group.
    expected = KPoints.automatic(
        (2, 2, 4), (0, 0, 0), system.cell,
        rotations=group.rotation_array(), time_reversal=True,
    )
    assert len(np.asarray(system.kpoints.weights)) == len(
        np.asarray(expected.weights))


def test_a_seed_sized_staggered_field_cuts_the_same_operations():
    """The filter reads a pattern, so a 1e-8 Ry seed field is as good as 0.05."""
    system = _build(_h2(cards=_card(1.0e-8, -1.0e-8, fmt=".16e")))
    assert _swaps(system, system.symmetry_group()) == 0


def test_a_uniform_local_field_keeps_the_whole_group():
    """The negative: the same field on both sites labels nothing."""
    plain = _build(_h2())
    system = _build(_h2(cards=_card(0.05, 0.05)))
    assert system.symmetry_group().nsym == plain.symmetry_group().nsym == 16


def test_a_field_can_restore_nothing_the_moments_removed():
    """Moments and fields are tested together: both patterns must survive.

    An antiferromagnet under a *uniform* local field, which alone would keep
    the whole group, still loses exactly the swap: a field can only remove
    operations, never put back one the moments removed.
    """
    moments = "STARTING_MOMENTS\n 0.0 0.0 0.6\n 0.0 0.0 -0.6\n"
    afm = _build(_h2("starting_magnetization(1) = 0.6", cards=moments))
    both = _build(_h2("starting_magnetization(1) = 0.6",
                      cards=moments + _card(0.05, 0.05)))
    assert _swaps(both, both.symmetry_group()) == 0
    assert both.symmetry_group().nsym == afm.symmetry_group().nsym


# -- a partly given r_m ----------------------------------------------------------

_TWO_SPECIES = """
&control
  calculation = 'scf'
/
&system
  ibrav = 1, celldm(1) = 8.0, nat = 2, ntyp = 2, ecutwfc = 20.0,
  occupations = 'smearing', degauss = 0.02
  noncolin = .true., starting_magnetization(1) = 0.3,
  starting_magnetization(2) = 0.3
  {extra}
/
&electrons
/
ATOMIC_SPECIES
 H  1.008  H.pz-vbc.UPF
 He 4.003  He.pz-vbc.UPF
ATOMIC_POSITIONS crystal
 H  0.0 0.0 0.0
 He 0.5 0.5 0.5
K_POINTS gamma
"""


def test_an_unset_r_m_takes_the_default_radius_not_zero():
    """``make_pointlists.f90:150-154`` resets ``r_m < 1e-8`` to its default.

    ``pwin.indexed`` fills the species the input leaves out with 0.0, and a
    sphere of radius zero integrates nothing under ``'qe'`` weights and is a
    ``0/0`` under ``'smooth'``. The given radius must survive untouched.
    """
    from defumat.scf.locals import build_local_regions, default_radii

    plain = _build(_TWO_SPECIES.format(extra=""))
    default = default_radii(plain.cell, plain.structure)
    assert plain.integration_radii == ()

    system = _build(_TWO_SPECIES.format(extra="r_m(1) = 1.5"))
    radii = system.integration_radii
    assert radii[0] == pytest.approx(1.5)
    assert radii[1] == pytest.approx(float(default[1]))
    assert radii[1] > 0.5

    # What that buys: the He sphere holds a region, and the smooth weights are
    # finite everywhere.
    grid = (24, 24, 24)
    qe = build_local_regions(system.cell, system.structure, grid, radii=radii)
    owned = int(np.sum(np.asarray(qe.owner) == 1))
    assert owned > 10
    smooth = build_local_regions(
        system.cell, system.structure, grid, radii=radii, scheme="smooth"
    )
    assert np.all(np.isfinite(np.asarray(smooth.weights)))
