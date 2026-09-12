"""A per-atom magnetic texture, and the symmetry group that could not see it.

``m_loc`` decides the magnetic symmetry group and is built from
``starting_magnetization``/``angle1``/``angle2``, which are **per species**. A
texture -- a helix, a cycloid, a skyrmion -- has one direction per *atom*, so QE's
input cannot state one and the group is built from a ferromagnet instead: it keeps
operations the texture does not have, and ``sym_rho``'s ``nspin = 4`` branch then
averages the texture away while the charge converges and ``dr2`` reports nothing.

Measured on a 45-atom NiBr2 helix: one distinct Ni direction in ``m_loc`` against
15 in the ``LOCAL_MAGNETIC_FIELDS`` card, ``nsym = 4`` kept, and a
24-degree-per-site cycloid collapsed to collinear between iterations 3 and 6 --
at Elk's own converged per-atom fields, biasing the other way.

Two things are asserted here: the refusal fires on that shape, and the
``STARTING_MOMENTS`` card removes the cause rather than the symptom by putting
the texture into ``m_loc``.
"""

import numpy as np
import pytest

from defumat.io.pwin import parse_pw_input
from defumat.system.builder import build_system, _distinct_directions

pytestmark = pytest.mark.unit

#: A four-atom chain with room for a texture along it. Simple cubic so the group
#: is large and the magnetic filter has something to cut down.
HEAD = (
    "&system\n ibrav=1, celldm(1)=12.0, nat=4, ntyp=1, ecutwfc=12.0,\n"
    " noncolin = .true., starting_magnetization(1) = 0.5,\n{extra}/\n"
    "ATOMIC_SPECIES\n Ni 58.69 Ni.pbe-nd-rrkjus.UPF\n"
    "ATOMIC_POSITIONS alat\n"
    " Ni 0.00 0.0 0.0\n Ni 0.25 0.0 0.0\n Ni 0.50 0.0 0.0\n Ni 0.75 0.0 0.0\n"
    "K_POINTS gamma\n"
)

#: A 90-degree-per-site cycloid in the xy plane: four distinct directions.
CYCLOID = [(np.cos(t), np.sin(t), 0.0) for t in np.deg2rad([0, 90, 180, 270])]


def _card(name, rows, scale=1.0):
    body = "\n".join(f" {scale * x:.8f} {scale * y:.8f} {scale * z:.8f}"
                     for x, y, z in rows)
    return f"{name}\n{body}\n"


def _input(extra="", cards=""):
    return HEAD.format(extra=extra) + cards


def test_distinct_directions_counts_antiparallel_separately():
    """Two sublattices are two directions, and a zero moment is none.

    Antiparallel has to count as distinct: an operation mapping one sublattice
    onto the other is a symmetry only *with* time reversal, which is the whole
    distinction ``magnetic_symmetries`` makes.
    """
    assert _distinct_directions([(0.0, -0.02, 0.0)] * 15) == 1
    assert _distinct_directions(CYCLOID) == 4
    assert _distinct_directions([(1, 0, 0), (-1, 0, 0)]) == 2
    assert _distinct_directions([(0, 0, 0), (0, 0, 0)]) == 0
    # Length carries no direction.
    assert _distinct_directions([(1, 0, 0), (7.5, 0, 0)]) == 1


def test_a_textured_field_cuts_the_symmetry_group():
    """The NiBr2 shape, and the whole point: the field is in the filter.

    ``m_loc`` has one direction here and the card has four, so a group filtered
    by the moments alone keeps operations the texture does not have and
    ``sym_rho`` averages the texture away. Elk's ``findsym.f90`` rotates
    ``bfcmt0`` and compares it exactly as it does the moments, and this asserts
    that we do too -- the group must be strictly smaller than the one the
    moments alone would give.
    """
    ferro = build_system(parse_pw_input(_input()))
    textured = build_system(parse_pw_input(
        _input(cards=_card("LOCAL_MAGNETIC_FIELDS", CYCLOID, scale=1.0e-3))
    ))
    assert _distinct_directions(textured.local_moments) == 1
    assert _distinct_directions(textured.atomic_b_field) == 4
    assert textured.symmetry_group().nsym < ferro.symmetry_group().nsym


def test_an_infinitesimal_field_cuts_the_group_just_as_a_large_one_does():
    """The filter must not have a scale, because a seed field has no natural one.

    An infinitesimal symmetry-breaking field is the *standard* way to start a
    texture -- small enough not to bias the energy, large enough to pick the
    state -- and it is what the NiBr2 helix used. Two rules used to disagree
    about how small was too small: ``is_magnetic`` counts a field from 1e-12,
    deliberately, while the filter dropped one below 1e-5. In between, a run
    turned magnetic, switched time reversal off, and then kept the *full*
    crystal group, which averages away exactly the texture the field was
    applied to create. Seven orders of magnitude wide.

    Measured before the fix: 1e-5 cut the group from 8 to 2 and 1e-6 left it at
    8. The card is written at ``%.16e`` here rather than the ``%.8f`` the other
    tests use, because at 1e-11 that format rounds every component to zero and
    the test would be measuring its own printf.
    """
    def wide_card(rows, scale):
        body = "\n".join(
            f" {scale * x:.16e} {scale * y:.16e} {scale * z:.16e}" for x, y, z in rows
        )
        return f"LOCAL_MAGNETIC_FIELDS\n{body}\n"

    ferro = build_system(parse_pw_input(_input())).symmetry_group().nsym
    large = build_system(parse_pw_input(
        _input(cards=wide_card(CYCLOID, 1.0e-3))
    )).symmetry_group().nsym
    assert large < ferro, "the premise: a large field cuts the group"

    # Every decade from the old threshold down to is_magnetic's own floor.
    for scale in (1.0e-6, 1.0e-9, 1.0e-12):
        system = build_system(parse_pw_input(_input(cards=wide_card(CYCLOID, scale))))
        assert system.nspin_mag == 4, f"{scale:g} should still be a magnetic run"
        assert system.symmetry_group().nsym == large, (
            f"a field of {scale:g} Ry made the run magnetic and then contributed "
            f"nothing to the filter"
        )

    # And below that floor neither rule counts it, which is the consistency the
    # fix is: a field too small to make a run magnetic cannot cut its group.
    tiny = build_system(parse_pw_input(_input(cards=wide_card(CYCLOID, 1.0e-14))))
    assert tiny.symmetry_group().nsym == ferro


def test_a_parallel_field_leaves_the_group_alone():
    """The negative: a field that asks for no texture takes nothing away.

    This is what says the cut is about the *texture* and not about the card
    being present, and it is what keeps an ordinary ferromagnet with a per-atom
    field running at full symmetry.
    """
    rows = [(0.0, 0.0, 1.0)] * 4
    ferro = build_system(parse_pw_input(_input()))
    system = build_system(parse_pw_input(
        _input(cards=_card("LOCAL_MAGNETIC_FIELDS", rows, scale=1.0e-3))
    ))
    assert _distinct_directions(system.local_moments) == 1
    assert system.symmetry_group().nsym == ferro.symmetry_group().nsym


def test_the_three_symmetry_sites_agree():
    """``System.axial_fields`` exists so the builder and the property cannot differ.

    The k-point reduction, ``symmetry_group`` and the build all decide the same
    group, and a run whose k-set was reduced with one group and whose density is
    symmetrised with another is wrong in a way no single number reveals.
    """
    system = build_system(parse_pw_input(
        _input(cards=_card("LOCAL_MAGNETIC_FIELDS", CYCLOID, scale=1.0e-3))
    ))
    fields = system.axial_fields
    assert isinstance(fields, tuple) and len(fields) == 2
    np.testing.assert_allclose(np.asarray(fields[0]), np.asarray(system.local_moments))
    np.testing.assert_allclose(
        np.asarray(fields[1]), np.asarray(system.atomic_b_field, dtype=float)
    )


def test_nosym_still_wins():
    """``nosym`` is the blunt fix and must keep working whatever the filter does."""
    system = build_system(parse_pw_input(
        _input(extra=" nosym = .true., noinv = .true.,\n",
               cards=_card("LOCAL_MAGNETIC_FIELDS", CYCLOID, scale=1.0e-3))
    ))
    assert system.nosym is True
    assert system.symmetry_group(nosym=True).nsym == 1


def test_starting_moments_puts_the_texture_into_m_loc():
    """The other fix, and the one that keeps the symmetry machinery working.

    The card is the only way the texture reaches ``m_loc``, so this asserts both
    halves: ``local_moments`` gains the four directions, and the magnetic group
    shrinks because the operations a ferromagnet had are not a cycloid's.
    """
    cards = (_card("LOCAL_MAGNETIC_FIELDS", CYCLOID, scale=1.0e-3)
             + _card("STARTING_MOMENTS", CYCLOID, scale=0.5))
    system = build_system(parse_pw_input(_input(cards=cards)))

    assert _distinct_directions(system.starting_moments) == 4
    assert _distinct_directions(system.local_moments) == 4
    np.testing.assert_allclose(
        np.asarray(system.local_moments),
        0.5 * np.asarray(CYCLOID),
        atol=1.0e-12,
    )
    # The per-species route would have given a ferromagnet here, and a
    # ferromagnet's group is strictly larger than a cycloid's.
    ferro = build_system(parse_pw_input(_input()))
    assert system.symmetry_group().nsym < ferro.symmetry_group().nsym


def test_starting_moments_off_the_axis_needs_noncolin():
    """The same rule ``B_field`` and ``LOCAL_MAGNETIC_FIELDS`` are held to."""
    collinear = (
        "&system\n ibrav=1, celldm(1)=12.0, nat=4, ntyp=1, ecutwfc=12.0,\n"
        " nspin = 2, starting_magnetization(1) = 0.5,\n"
        " occupations = 'smearing', degauss = 0.01,\n/\n"
        "ATOMIC_SPECIES\n Ni 58.69 Ni.pbe-nd-rrkjus.UPF\n"
        "ATOMIC_POSITIONS alat\n"
        " Ni 0.00 0.0 0.0\n Ni 0.25 0.0 0.0\n Ni 0.50 0.0 0.0\n Ni 0.75 0.0 0.0\n"
        "K_POINTS gamma\n"
    ) + _card("STARTING_MOMENTS", CYCLOID, scale=0.5)
    with pytest.raises(ValueError, match="noncolin"):
        build_system(parse_pw_input(collinear))


def test_a_wrong_length_card_is_refused():
    with pytest.raises(ValueError, match="STARTING_MOMENTS lists"):
        build_system(parse_pw_input(
            _input(cards=_card("STARTING_MOMENTS", CYCLOID[:2], scale=0.5))
        ))


# --- the constraint that can hold a texture, and the one that cannot ---------

import jax.numpy as jnp

from defumat.scf.fields import MagneticField, constraint_targets
from defumat.scf.locals import LocalRegions
from defumat.system.cell import Cell

CGRID = (4, 1, 1)
CELL = Cell.from_ibrav(1, [8.0, 0, 0, 0, 0, 0])


def _one_hot_regions(nat=4):
    """One grid point per atom, so a local moment is a density value times a scale."""
    weights = np.zeros((nat,) + CGRID)
    for a in range(nat):
        weights[a, a, 0, 0] = 1.0
    return LocalRegions(weights=jnp.asarray(weights), radii=(1.0,) * nat,
                        grid=weights.shape[1:], nat=nat, scheme="qe")


def _density_with_moments(moments):
    """A ``(4, ...)`` density whose per-atom local moments are ``moments``."""
    scale = CELL.volume / int(np.prod(CGRID))
    rho = np.zeros((4,) + CGRID)
    rho[0] = 1.0
    for a, m in enumerate(np.asarray(moments, dtype=float)):
        rho[1:, a, 0, 0] = m / scale
    return jnp.asarray(rho)


def _field(constraint, targets, penalty=1.0):
    return MagneticField(
        regions=_one_hot_regions(),
        uniform=jnp.zeros(3),
        atomic=None,
        targets=jnp.asarray(targets),
        penalty=penalty,
        constraint=constraint,
    )


#: The cycloid, and a collinear state lying in the *same* plane. Both have every
#: moment perpendicular to z, which is the whole trap.
COLLINEAR_IN_PLANE = [(1.0, 0.0, 0.0)] * 4


def test_atomic_direction_cannot_tell_a_cycloid_from_a_collinear_state():
    """QE's ``i_cons = 2`` is identically satisfied by both, at zero penalty.

    This is the check firing on the case it has to catch: ``atomic direction``
    constrains ``m_z/|m|`` alone, so for any texture in a plane containing the
    origin-to-z axis the penalty is the same constant whatever the azimuths do.
    A constraint that returns a clean zero for the state it is meant to exclude
    reads as satisfied, which is why this is asserted rather than assumed.
    """
    targets = np.zeros((4, 1))  # cos(theta) = 0: every moment in the xy plane
    field = _field("atomic direction", targets)

    helix = field.constraint_energy(_density_with_moments(CYCLOID), CELL)
    flat = field.constraint_energy(_density_with_moments(COLLINEAR_IN_PLANE), CELL)

    assert float(helix) == pytest.approx(0.0, abs=1e-20)
    assert float(flat) == pytest.approx(0.0, abs=1e-20)


def test_atomic_texture_separates_them():
    """The mode that constrains the full unit vector does tell them apart."""
    targets = constraint_targets(
        "atomic texture", [0] * 4, (0.5,), (), (), (), 1, True, per_atom=CYCLOID,
    )
    np.testing.assert_allclose(
        np.linalg.norm(targets, axis=-1), 1.0, atol=1e-12
    )
    field = _field("atomic texture", targets)

    helix = float(field.constraint_energy(_density_with_moments(CYCLOID), CELL))
    flat = float(field.constraint_energy(
        _density_with_moments(COLLINEAR_IN_PLANE), CELL))

    # Zero where every moment points where it was asked to ...
    assert helix == pytest.approx(0.0, abs=1e-12)
    # ... and the collinear state pays for three of its four sites: the cycloid
    # asks for 0, 90, 180 and 270 degrees, so cos = 1, 0, -1, 0 and the penalty
    # is sum(1 - cos) = 4.
    assert flat == pytest.approx(4.0, rel=1e-10)


def test_atomic_texture_is_insensitive_to_moment_length():
    """It is a *direction* constraint, so scaling every moment changes nothing."""
    targets = constraint_targets(
        "atomic texture", [0] * 4, (0.5,), (), (), (), 1, True, per_atom=CYCLOID,
    )
    field = _field("atomic texture", targets)
    tilted = [(np.cos(t), np.sin(t), 0.0) for t in np.deg2rad([10, 100, 190, 280])]
    small = float(field.constraint_energy(
        _density_with_moments(0.01 * np.asarray(tilted)), CELL))
    large = float(field.constraint_energy(
        _density_with_moments(7.5 * np.asarray(tilted)), CELL))
    assert small == pytest.approx(large, rel=1e-9)
    # A uniform 10-degree error on four sites.
    assert small == pytest.approx(4 * (1 - np.cos(np.deg2rad(10))), rel=1e-9)


def test_atomic_texture_gradient_is_finite_at_a_vanishing_moment():
    """The double-``where`` safe divide, which is the P70 trap in miniature.

    A site whose moment goes to zero has no direction, and masking the *result*
    of ``m/|m|`` while leaving the argument singular puts ``0 * inf`` in the
    tangent. The energy and its gradient both have to stay finite.
    """
    import jax

    targets = constraint_targets(
        "atomic texture", [0] * 4, (0.5,), (), (), (), 1, True, per_atom=CYCLOID,
    )
    field = _field("atomic texture", targets)
    moments = np.asarray(CYCLOID, dtype=float)
    moments[2] = 0.0  # one site with nothing to point
    density = _density_with_moments(moments)

    energy = field.constraint_energy(density, CELL)
    gradient = jax.grad(lambda r: field.constraint_energy(r, CELL))(density)
    assert np.isfinite(float(energy))
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_atomic_texture_without_a_card_is_refused():
    with pytest.raises(ValueError, match="STARTING_MOMENTS card"):
        constraint_targets(
            "atomic texture", [0] * 4, (0.5,), (), (), (), 1, True,
        )


def test_atomic_texture_refuses_a_zero_target():
    with pytest.raises(ValueError, match="zero row"):
        constraint_targets(
            "atomic texture", [0] * 4, (0.5,), (), (), (), 1, True,
            per_atom=[(1, 0, 0), (0, 0, 0), (0, 1, 0), (0, 0, 1)],
        )


def test_atomic_direction_gradient_is_finite_at_a_vanishing_moment():
    """The same safe divide, on QE's own scheme, where it was latent.

    ``_polar_cosine`` masked the *result* of ``m_z/|m|`` and left ``sqrt`` seeing
    a zero, whose derivative is infinite -- so a run with a ligand carrying no
    induced moment put a NaN into the constraint's contribution to the potential
    the first time anyone differentiated it. The energy was finite throughout,
    which is why it survived: this is the trap where the value is right and the
    derivative is not.
    """
    import jax

    field = _field("atomic direction", np.zeros((4, 1)))
    moments = np.asarray(CYCLOID, dtype=float)
    moments[1] = 0.0
    density = _density_with_moments(moments)

    energy = field.constraint_energy(density, CELL)
    gradient = jax.grad(lambda r: field.constraint_energy(r, CELL))(density)
    assert np.isfinite(float(energy))
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_atomic_texture_without_a_card_is_refused_at_the_input():
    """Before anything is allocated, not when the targets are built.

    An input that cannot work should stop at the boundary; the same refusal in
    ``constraint_targets`` is for a caller that reaches it directly.
    """
    with pytest.raises(ValueError, match="STARTING_MOMENTS card"):
        build_system(parse_pw_input(_input(
            extra=" constrained_magnetization = 'atomic texture', lambda = 0.5,\n"
        )))


def test_atomic_texture_runs_with_a_card():
    """And the same input with the card builds."""
    system = build_system(parse_pw_input(_input(
        extra=" constrained_magnetization = 'atomic texture', lambda = 0.5,\n",
        cards=_card("STARTING_MOMENTS", CYCLOID, scale=0.5),
    )))
    assert system.constrained_magnetization == "atomic texture"
    assert _distinct_directions(system.local_moments) == 4


# --- ...and the card has to start the moments it names -----------------------
#
# The group was the half that was fixed. The card was also documented as
# overriding ``starting_magnetization``/``angle1``/``angle2``, and it reached
# only the *constraint* field: ``domag`` read the per-species array, and so did
# the starting density. Two consequences, and the second survives fixing the
# first.


#: The same chain with **no** ``starting_magnetization`` at all, so the card is
#: the only thing that can say the run is magnetic.
BARE = (
    "&system\n ibrav=1, celldm(1)=12.0, nat=4, ntyp=1, ecutwfc=12.0,\n"
    " noncolin = .true.,\n/\n"
    "ATOMIC_SPECIES\n Ni 58.69 Ni.pbe-nd-rrkjus.UPF\n"
    "ATOMIC_POSITIONS alat\n"
    " Ni 0.00 0.0 0.0\n Ni 0.25 0.0 0.0\n Ni 0.50 0.0 0.0\n Ni 0.75 0.0 0.0\n"
    "K_POINTS gamma\n"
)


def test_a_texture_given_only_by_the_card_is_a_magnetic_run():
    """``domag`` read ``starting_magnetization``, the per-*species* array.

    An input whose texture is given **only** through ``STARTING_MOMENTS``
    therefore got ``domag = False``, hence ``nspin_mag = 1``: it converged, it
    reported a total energy, and it never had a magnetization. Nothing in the
    output said so, because a nonmagnetic spin-orbit run is a legitimate thing
    to ask for and looks exactly like this.
    """
    system = build_system(parse_pw_input(
        BARE + _card("STARTING_MOMENTS", CYCLOID, scale=0.5)
    ))
    assert system.nspin == 4
    assert system.domag
    assert system.nspin_mag == 4
    assert _distinct_directions(system.local_moments) == 4

    # The control: the same cell with neither is correctly *not* magnetic.
    plain = build_system(parse_pw_input(BARE))
    assert plain.nspin == 4 and not plain.domag and plain.nspin_mag == 1


def test_one_rule_decides_whether_a_run_is_magnetic():
    """``domag``, the k-point reduction and the symmetry group had two rules
    between them: one read ``starting_magnetization``, the other read
    ``local_moments`` and the ``LOCAL_MAGNETIC_FIELDS`` card.

    The consequence is trap 4 through bookkeeping: the SCF can symmetrise the
    density with a larger group than the one its k-set was reduced with.
    """
    from defumat.system.builder import is_magnetic

    system = build_system(parse_pw_input(
        BARE + _card("LOCAL_MAGNETIC_FIELDS", CYCLOID, scale=1.0e-3)
    ))
    # A per-atom field alone is enough: it is an applied constraint, not a
    # guess, so asking for one is asking for the magnetic branch.
    assert system.domag and system.nspin_mag == 4
    assert is_magnetic(4, np.zeros((4, 3)), np.asarray(CYCLOID) * 1e-3)
    assert not is_magnetic(4, np.zeros((4, 3)), ())
    assert not is_magnetic(2, np.asarray(CYCLOID), ())  # collinear: never 4


def test_the_starting_density_carries_the_cards_texture(pseudo_dir):
    """The consequence that survives fixing ``domag``.

    With ``starting_magnetization`` also set the run *is* magnetic, and the
    starting density was still built from the per-*type* magnitudes and angles
    -- so a card documented as a starting magnetic texture seeded no texture. It
    started the per-species ferromagnet and then penalised it toward the
    texture, which is a different calculation and a slower one.

    Asserted on the starting density itself rather than on a converged state:
    an SCF would say the same thing four minutes later.
    """
    from defumat.calculator import Calculator

    cards = _card("STARTING_MOMENTS", CYCLOID, scale=0.5)
    calc = Calculator.from_text(
        _input(extra=" angle1(1) = 0.0,\n", cards=cards), pseudo_dir,
        announce=False,
    )
    density = np.asarray(calc.calculation.starting_density())
    assert density.shape[0] == 4  # (n, m_x, m_y, m_z)

    # Integrate each moment component over a sphere around each atom. The
    # cycloid puts atom 0 along +x and atom 2 along -x, so the two must come out
    # with opposite sign and the cell total must vanish.
    grid = density.shape[1:]
    axes = [np.arange(n) / n for n in grid]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    positions = np.asarray(calc.system.structure.positions)
    crystal = positions @ np.linalg.inv(np.asarray(calc.system.cell.at))

    def near(atom):
        delta = mesh - crystal[atom][None, None, None, :]
        delta -= np.round(delta)
        return np.sum(delta**2, axis=-1) < (0.10) ** 2

    mx = density[1]
    lobes = [float(mx[near(a)].sum()) for a in range(4)]
    assert lobes[0] > 0.0 and lobes[2] < 0.0, lobes
    assert abs(lobes[0] + lobes[2]) < 0.05 * abs(lobes[0])
    # ... and the whole cell is unpolarized, as a cycloid is.
    assert abs(float(mx.sum())) < 1e-6 * float(density[0].sum())
