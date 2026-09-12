"""P18: the field and constraint potentials, against QE's hand-derived ones.

:mod:`defumat.scf.fields` writes down the *energy* of a field or a penalty and
takes the potential from ``jax.grad``. QE writes the potential out by hand
instead -- ``add_bfield.f90``, five expressions, one of them three lines of
quotient rule -- and the two must agree exactly, because they are the same
derivative.

That agreement is the whole point of the arrangement, so it is asserted here for
every scheme rather than left to the one benchmark that happens to exercise one
of them. The Fortran expressions are transcribed literally below, including the
signs, and nothing in them is shared with the implementation under test.

The densities are synthetic: a random ``(nspin_mag, ...)`` array on a small grid
with random per-atom weights. Nothing about this test needs a physical density,
and using one would only make the comparison harder to read.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.scf.fields import MagneticField
from defumat.scf.locals import LocalRegions
from defumat.system.cell import Cell

pytestmark = pytest.mark.unit

GRID = (6, 5, 4)
NAT = 2


@pytest.fixture(scope="module")
def cell() -> Cell:
    return Cell.from_ibrav(1, [7.0, 0, 0, 0, 0, 0])


@pytest.fixture(scope="module")
def density():
    """A random noncollinear density, ``(4, n1, n2, n3)``."""
    rng = np.random.default_rng(20260820)
    rho = rng.normal(size=(4,) + GRID)
    rho[0] = np.abs(rho[0]) + 0.5
    return jnp.asarray(rho * 0.1)


@pytest.fixture(scope="module")
def regions():
    """Random per-atom weights, in ``[0, 1]`` and not summing to anything."""
    rng = np.random.default_rng(31337)
    weights = rng.uniform(size=(NAT,) + GRID)
    return LocalRegions(weights=jnp.asarray(weights), radii=(1.0,),
                        grid=GRID, nat=NAT, scheme="qe")


def _moments(field: MagneticField, density, cell):
    return np.asarray(field.local_moments(density, cell))


def _potential(field: MagneticField, density, cell, scale: float = 1.0):
    v, _, _ = field.potential(density, cell, scale)
    return np.asarray(v)


def _weighted(regions, values, grid_scale):
    """``sum_a w_a(r) values[a, ipol]`` as a ``(3, ...)`` grid array."""
    return np.einsum("anmk,ac->cnmk", np.asarray(regions.dense_weights()), values)


def test_uniform_field_is_minus_b(density, cell):
    """``i_cons = 4``: ``v(:, ipol+1) -= bfield(ipol)``, and nothing else moves."""
    b = jnp.asarray([0.03, -0.02, 0.05])
    field = MagneticField(
        regions=None, uniform=b, atomic=None, targets=None, penalty=0.0,
    )
    v = _potential(field, density, cell)

    assert v[0] == pytest.approx(np.zeros(GRID), abs=1e-14)
    for ipol in range(3):
        assert v[ipol + 1] == pytest.approx(np.full(GRID, -float(b[ipol])), abs=1e-12)

    # ... and the energy is the Zeeman one, -B . M.
    _, e_field, e_constraint = field.potential(density, cell)
    moment = np.asarray(field.total_moment(density, cell))
    assert float(e_field) == pytest.approx(-float(np.dot(np.asarray(b), moment)))
    assert float(e_constraint) == 0.0


def test_reducebf_scales_the_field_and_not_the_penalty(density, cell, regions):
    """Elk 5.104: the external field is multiplied down, the constraint is not."""
    field = MagneticField(
        regions=regions,
        uniform=jnp.asarray([0.0, 0.0, 0.02]),
        atomic=None,
        targets=jnp.asarray([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        penalty=0.4,
        constraint="atomic",
        reducebf=0.5,
    )
    full = _potential(field, density, cell, 1.0)
    half = _potential(field, density, cell, 0.5)
    penalty_only = _potential(field, density, cell, 0.0)

    # v(scale) is affine in the scale: the field part scales, the penalty does not.
    assert half == pytest.approx(0.5 * (full + penalty_only), abs=1e-12)


def test_atomic_constraint_matches_add_bfield(density, cell, regions):
    """``i_cons = 1``: ``v += 2 lambda factlist(ir) (m_loc - mcons)``."""
    targets = np.array([[0.10, -0.05, 0.20], [0.00, 0.30, -0.10]])
    penalty = 0.37
    field = MagneticField(
        regions=regions, uniform=jnp.zeros(3), atomic=None,
        targets=jnp.asarray(targets), penalty=penalty, constraint="atomic",
    )

    m_loc = _moments(field, density, cell)
    m2 = m_loc - targets  # add_bfield's m2(ipol, na)
    expected = 2.0 * penalty * _weighted(regions, m2, cell)

    v = _potential(field, density, cell)
    assert v[0] == pytest.approx(np.zeros(GRID), abs=1e-14)
    assert v[1:] == pytest.approx(expected, abs=1e-10)

    # etcon = lambda * sum_a sum_ipol m2^2
    _, _, e_constraint = field.potential(density, cell)
    assert float(e_constraint) == pytest.approx(penalty * float(np.sum(m2**2)))


def test_atomic_direction_constraint_matches_add_bfield(density, cell, regions):
    """``i_cons = 2``: the polar-angle penalty, whose derivative QE writes out.

    Transcribed from ``add_bfield.f90``::

        xx   = m_loc(3)/ma - mcons(3)
        m2(1) = -xx*m_loc(1)*m_loc(3) / ma^3
        m2(2) = -xx*m_loc(2)*m_loc(3) / ma^3
        m2(3) =  xx*(-m_loc(3)^2 / ma^3 + 1/ma)
    """
    cosines = np.array([[0.3], [-0.6]])
    penalty = 0.21
    field = MagneticField(
        regions=regions, uniform=jnp.zeros(3), atomic=None,
        targets=jnp.asarray(cosines), penalty=penalty,
        constraint="atomic direction",
    )

    m_loc = _moments(field, density, cell)
    m2 = np.zeros_like(m_loc)
    total = 0.0
    for a in range(NAT):
        ma = np.linalg.norm(m_loc[a])
        xx = m_loc[a, 2] / ma - cosines[a, 0]
        m2[a, 0] = -xx * m_loc[a, 0] * m_loc[a, 2] / ma**3
        m2[a, 1] = -xx * m_loc[a, 1] * m_loc[a, 2] / ma**3
        m2[a, 2] = xx * (-m_loc[a, 2] ** 2 / ma**3 + 1.0 / ma)
        total += penalty * (m_loc[a, 2] / ma - cosines[a, 0]) ** 2
    expected = 2.0 * penalty * _weighted(regions, m2, cell)

    v = _potential(field, density, cell)
    assert v[1:] == pytest.approx(expected, abs=1e-10)
    _, _, e_constraint = field.potential(density, cell)
    assert float(e_constraint) == pytest.approx(total)


def test_total_constraint_matches_add_bfield(density, cell):
    """``i_cons = 3``: ``bfield = -2 lambda (M - mcons)`` and ``v -= bfield``."""
    target = np.array([0.4, 0.0, -0.2])
    penalty = 0.13
    field = MagneticField(
        regions=None, uniform=jnp.zeros(3), atomic=None,
        targets=jnp.asarray(target), penalty=penalty, constraint="total",
    )
    moment = np.asarray(field.total_moment(density, cell))
    bfield = -2.0 * penalty * (moment - target)

    v = _potential(field, density, cell)
    for ipol in range(3):
        assert v[ipol + 1] == pytest.approx(np.full(GRID, -bfield[ipol]), abs=1e-10)


def test_total_direction_constraint_matches_add_bfield(density, cell):
    """``i_cons = 6``: ``E = lambda (arccos(m_z/|m|) - theta)^2``, QE's ``fact1``."""
    theta_degrees = 35.0
    penalty = 0.29
    field = MagneticField(
        regions=None, uniform=jnp.zeros(3), atomic=None,
        targets=jnp.asarray([theta_degrees]), penalty=penalty,
        constraint="total direction",
    )
    m = np.asarray(field.total_moment(density, cell))
    ma = np.linalg.norm(m)
    mperp = np.hypot(m[0], m[1])
    xx = np.arccos(m[2] / ma) - np.deg2rad(theta_degrees)
    fact1 = np.array([
        m[0] / mperp * m[2] / ma**2,
        m[1] / mperp * m[2] / ma**2,
        -np.sqrt(1.0 - (m[2] / ma) ** 2) / ma,
    ])
    bfield = 2.0 * penalty * xx * fact1

    v = _potential(field, density, cell)
    for ipol in range(3):
        assert v[ipol + 1] == pytest.approx(np.full(GRID, bfield[ipol]), abs=1e-9)
    _, _, e_constraint = field.potential(density, cell)
    assert float(e_constraint) == pytest.approx(penalty * xx**2)


@pytest.fixture(scope="module")
def axial_density():
    """A noncollinear density whose total moment lies **exactly** on ``z``.

    This is not a contrived state: it is where a run seeded from
    ``starting_magnetization`` with no ``angle1``/``angle2`` starts, so a polar
    angle constraint meets it on its first iteration.
    """
    rng = np.random.default_rng(20260912)
    rho = rng.normal(size=(4,) + GRID)
    rho[0] = np.abs(rho[0]) + 0.5
    rho = rho * 0.1
    # Bit-exactly zero, because the boundary is reached rather than approached
    # (`CLAUDE.md`, the clamp trap). Cancelling a random field to its own mean
    # instead leaves 6e-16, which is the *realistic* case and is covered by the
    # threshold rather than by the exact zero -- both are asserted below.
    rho[1] = 0.0
    rho[2] = 0.0
    rho[3] = np.abs(rho[3]) + 0.2
    return jnp.asarray(rho)


def test_the_polar_angle_constraint_is_finite_on_the_axis(axial_density, cell):
    """``i_cons = 6`` with the moment along ``z``: QE guards it and so must this.

    ``arccos'`` diverges at ``+-1`` while ``d|m_perp|/dm_x`` vanishes there, so
    the chain is ``0 * inf`` and **every** component of the potential comes back
    NaN. ``add_bfield.f90:184-192`` zeroes the transverse factors below
    ``mperp < 1.D-14``; here the same thing falls out of masking the argument
    ``atan2`` is taken at.
    """
    field = MagneticField(
        regions=None, uniform=jnp.zeros(3), atomic=None,
        targets=jnp.asarray([40.0]), penalty=0.31,
        constraint="total direction",
    )
    m = np.asarray(field.total_moment(axial_density, cell))
    assert m[0] == 0.0 and m[1] == 0.0, "the fixture is not on the axis"
    assert m[2] > 0.1

    v = _potential(field, axial_density, cell)
    assert np.all(np.isfinite(v)), "the polar-angle potential is not finite on the axis"

    # QE's fact1(1:2) = 0 and fact1(3) = -mperp/|m|^2 = 0, so the only thing
    # left is the 1e-14 escape along x.
    assert np.allclose(v[2], 0.0, atol=1e-30)
    assert np.allclose(v[3], 0.0, atol=1e-30)
    error = 0.0 - np.deg2rad(40.0)
    escape = 2.0 * 0.31 * error * 1.0e-14
    assert v[1] == pytest.approx(np.full(GRID, escape), rel=1e-9)
    assert escape < 0.0, "the escape must push the moment off the axis, not onto it"


def test_the_polar_angle_constraint_is_finite_just_off_the_axis(axial_density, cell):
    """The realistic case: a transverse moment of round-off rather than of zero.

    Cancelling a random transverse channel against its own mean leaves ~1e-15,
    not 0. That is **also** a NaN in the unguarded form, and for a second
    reason: ``m_z/|m|`` rounds to bit-exactly 1.0 at that separation, so
    ``jnp.clip`` sits on its boundary and hands each argument half the tangent
    -- the clamp trap, on top of the diverging ``arccos'``. Which of the two
    fires is rounding, which is why the guard is a *threshold* at QE's 1e-14
    rather than a test for zero.
    """
    rho = np.asarray(axial_density).copy()
    rng = np.random.default_rng(4242)
    rho[1] = rng.normal(size=GRID) * 0.1
    rho[1] -= rho[1].mean()
    field = MagneticField(
        regions=None, uniform=jnp.zeros(3), atomic=None,
        targets=jnp.asarray([40.0]), penalty=0.31,
        constraint="total direction",
    )
    m = np.asarray(field.total_moment(jnp.asarray(rho), cell))
    mperp = np.hypot(m[0], m[1])
    assert 0.0 < mperp < 1.0e-14, f"the fixture is not in the guarded band ({mperp:.3e})"

    v = _potential(field, jnp.asarray(rho), cell)
    assert np.all(np.isfinite(v))
    assert np.max(np.abs(v[1:])) < 1e-13, "the guard is not covering the sub-threshold band"


def test_the_polar_angle_derivative_is_qes_fact1_off_the_axis(density, cell):
    """The guarded form must not change the answer anywhere it was already right.

    Written as ``atan2(|m_perp|, m_z)`` where QE writes ``arccos(m_z/|m|)``:
    the same angle, and the derivative has to be the same three numbers.
    """
    field = MagneticField(
        regions=None, uniform=jnp.zeros(3), atomic=None,
        targets=jnp.asarray([35.0]), penalty=0.29,
        constraint="total direction",
    )
    m = np.asarray(field.total_moment(density, cell))
    ma = np.linalg.norm(m)
    mperp = np.hypot(m[0], m[1])
    assert mperp > 1e-6, "the fixture is on the axis; this test is the other branch"
    fact1 = np.array([
        m[0] / mperp * m[2] / ma**2,
        m[1] / mperp * m[2] / ma**2,
        -np.sqrt(1.0 - (m[2] / ma) ** 2) / ma,
    ])
    xx = np.arccos(m[2] / ma) - np.deg2rad(35.0)

    v = _potential(field, density, cell)
    for ipol in range(3):
        assert v[ipol + 1] == pytest.approx(
            np.full(GRID, 2.0 * 0.29 * xx * fact1[ipol]), abs=1e-12
        )


def test_every_atom_resolved_constraint_gets_its_spheres(regions, density, cell):
    """The set that decides whether the spheres are built must cover all three.

    ``atomic texture`` was left out of a hand-written tuple in the driver, so a
    run asking for it built ``regions = None`` and died on
    ``None.integrate`` inside the **first potential build** -- a crash rather
    than a wrong number, but only reachable for an input that did not also
    happen to carry a ``LOCAL_MAGNETIC_FIELDS`` card, which is why every
    committed test missed it.
    """
    from defumat.scf.fields import ATOM_RESOLVED

    for constraint in sorted(ATOM_RESOLVED):
        width = 1 if constraint == "atomic direction" else 3
        targets = jnp.asarray(np.tile(
            [0.0, 0.0, 1.0][:width] if width == 3 else [0.5], (NAT, 1)))
        field = MagneticField(
            regions=regions, uniform=jnp.zeros(3), atomic=None,
            targets=targets, penalty=0.2, constraint=constraint,
        )
        assert np.isfinite(float(field.constraint_energy(density, cell)))

        without = MagneticField(
            regions=None, uniform=jnp.zeros(3), atomic=None,
            targets=targets, penalty=0.2, constraint=constraint,
        )
        with pytest.raises(ValueError, match="ATOM_RESOLVED"):
            without.constraint_energy(density, cell)


def test_the_driver_builds_spheres_for_a_bare_atomic_texture_run():
    """The regression itself, from the input file down.

    No ``LOCAL_MAGNETIC_FIELDS``: the spheres have to be built because the
    *constraint* is atom-resolved. Reaching ``constraint_energy`` is the point
    -- constructing the ``Calculation`` succeeded on the broken code too.
    """
    from defumat import Calculator
    from defumat.scf.driver import Calculation

    calculator = Calculator.from_file(
        "tests/data/qe/h2-texture-120.in", pseudo_dir="tests/data/pseudo")
    lines = Path("tests/data/qe/h2-texture-120.in").read_text().splitlines()
    assert not [ln for ln in lines
                if ln.strip().upper().startswith("LOCAL_MAGNETIC_FIELDS")]

    calculation = Calculation(calculator.system, calculator.pseudos)
    field = calculation.magnetic_field
    assert field.constraint == "atomic texture"
    assert field.atomic is None, "the input must not carry a per-atom field"
    assert field.regions is not None

    energy = float(field.constraint_energy(
        calculation.starting_density(), calculation.system.cell))
    assert np.isfinite(energy) and energy >= 0.0


def test_collinear_field_splits_the_two_channels(cell):
    """``add_bfield``'s collinear branch: ``v(:,1) -= B``, ``v(:,2) += B``.

    A collinear density is ``(up, down)`` and its magnetization is the
    difference, so the *same* Zeeman energy gives opposite potentials in the two
    channels -- which is what the chain rule does here and what QE writes by
    hand.
    """
    rng = np.random.default_rng(7)
    rho = jnp.asarray(np.abs(rng.normal(size=(2,) + GRID)) + 0.2)
    b = 0.017
    field = MagneticField(
        regions=None, uniform=jnp.asarray([b]), atomic=None, targets=None,
        penalty=0.0,
    )
    v, e_field, _ = field.potential(rho, cell)
    v = np.asarray(v)

    assert v[0] == pytest.approx(np.full(GRID, -b), abs=1e-12)
    assert v[1] == pytest.approx(np.full(GRID, +b), abs=1e-12)

    scale = float(cell.volume) / int(np.prod(GRID))
    moment = scale * float(jnp.sum(rho[0] - rho[1]))
    assert float(e_field) == pytest.approx(-b * moment)


def test_local_field_acts_only_inside_its_sphere(density, cell, regions):
    """Elk's ``bfcmt``: one field per atom, felt where that atom's weight is."""
    atomic = np.array([[0.0, 0.0, 0.05], [0.0, 0.0, -0.05]])
    field = MagneticField(
        regions=regions, uniform=jnp.zeros(3), atomic=jnp.asarray(atomic),
        targets=None, penalty=0.0,
    )
    expected = -_weighted(regions, atomic, cell)
    v = _potential(field, density, cell)
    assert v[1:] == pytest.approx(expected, abs=1e-12)


def test_the_potential_is_the_gradient_of_the_energy(density, cell, regions):
    """The arrangement itself: ``v`` is ``dE/drho`` and the pairing with deband works.

    A finite difference along a random direction, which is what would catch a
    missing quadrature weight -- the factor ``omega/N`` that turns a gradient
    with respect to grid values into a potential.
    """
    field = MagneticField(
        regions=regions,
        uniform=jnp.asarray([0.01, 0.0, -0.02]),
        atomic=None,
        targets=jnp.asarray([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]),
        penalty=0.3,
        constraint="atomic",
    )
    rng = np.random.default_rng(11)
    direction = jnp.asarray(rng.normal(size=density.shape) * 1e-3)

    def energy(rho):
        return field.energy(rho, cell)

    step = 1e-5
    finite = (float(energy(density + step * direction))
              - float(energy(density - step * direction))) / (2 * step)
    v, _, _ = field.potential(density, cell)
    scale = float(cell.volume) / density[0].size
    analytic = scale * float(jnp.sum(v * direction))
    assert finite == pytest.approx(analytic, rel=1e-6)


def test_b_field_survives_the_input_file():
    """``B_field`` in a namelist reaches ``System.b_field``.

    It did not, for the whole of P18: ``build_system`` asked the parser for
    ``"B_field"`` while the parser lowercases every namelist key, so the lookup
    returned zeros and a run that asked for a uniform field quietly got none.
    Nothing raised, the SCF converged, and the answer was the *unconstrained*
    one -- which is the same failure mode the fixed-spin-moment docstring warns
    about for a sign error, arrived at from the other direction.

    Every test of the field machinery until now built :class:`MagneticField` in
    Python, so the input path had no coverage at all. This is that path.
    """
    from defumat.io.pwin import parse_pw_input
    from defumat.system import build_system

    source = """ &control
    calculation = 'scf'
 /
 &system
    ibrav = 3, celldm(1) = 5.217, nat = 1, ntyp = 1, ecutwfc = 12.0
    nspin = 2
    starting_magnetization(1) = 0.5
    B_field(3) = -0.02
    occupations = 'smearing', degauss = 0.02
 /
 &electrons
 /
ATOMIC_SPECIES
 Fe 55.847 Fe.pz-nd-rrkjus.UPF
ATOMIC_POSITIONS (alat)
 Fe 0.0 0.0 0.0
K_POINTS gamma
"""
    assert build_system(parse_pw_input(source)).b_field == (0.0, 0.0, -0.02)


def test_b_field_with_a_constraint_is_refused():
    """``input.f90:1614``'s refusal, which the same typo had disabled.

    QE will not decide which of an external field and a constraint wins, and
    neither does this -- but the check read the same misspelled key, so the
    combination it exists to reject went through.

    **Noncollinear, and it has to be**: ``constrained_magnetization = 'total'``
    is ``i_cons = 3``, which ``input.f90`` allows only for ``nspin = 4``
    (``fixed_magnetization`` is a vector). The case was written collinear and
    reached this refusal only because that one was missing -- QE stops on the
    same input one check earlier. See
    ``tests/unit/test_sibling_refusals.py``.
    """
    import pytest as _pytest

    from defumat.io.pwin import parse_pw_input
    from defumat.system import build_system

    source = """ &control
    calculation = 'scf'
 /
 &system
    ibrav = 3, celldm(1) = 5.217, nat = 1, ntyp = 1, ecutwfc = 12.0
    noncolin = .true.
    starting_magnetization(1) = 0.5
    B_field(3) = -0.02
    constrained_magnetization = 'total'
    lambda = 0.1
    fixed_magnetization(3) = 2.0
 /
 &electrons
 /
ATOMIC_SPECIES
 Fe 55.847 Fe.pz-nd-rrkjus.UPF
ATOMIC_POSITIONS (alat)
 Fe 0.0 0.0 0.0
K_POINTS gamma
"""
    with _pytest.raises(ValueError, match="B_field"):
        build_system(parse_pw_input(source))


def test_a_field_that_did_not_fade_is_reported_and_a_faded_one_is_not(regions):
    """``reducebf`` decays the field *after* the convergence test.

    So nothing requires the field to be small before the loop breaks, and the
    state that is then reported is the ground state of a functional carrying a
    Zeeman term whose energy is excluded from the total by convention. The
    guard has to fire on that and stay quiet on a run that did fade, or it is
    noise and gets switched off.

    Measured on the hydrogen atom of
    ``test_a_local_field_breaks_the_symmetry_it_should``: ``reducebf = 0.5``
    stops with 6.1e-06 Ry of field and a total right to 5e-14 Ry, and
    ``reducebf = 0.99`` stops after **six** iterations with 9.5e-02 Ry still
    applied and a total 4.7e-05 Ry out. The threshold sits between them; the
    table is in ``FADED_FIELD``.
    """
    import warnings

    from defumat.scf.driver import _warn_if_the_field_did_not_fade

    def field_with(reducebf):
        return MagneticField(
            regions=regions,
            uniform=jnp.asarray([0.0, 0.0, 0.10]),
            atomic=None, targets=None, penalty=0.0,
            constraint="none", reducebf=reducebf, fsm_update="elk",
        )

    def warnings_from(reducebf, scale, converged=True):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_if_the_field_did_not_fade(
                field_with(reducebf), scale, converged, -1.0e-3
            )
        return [w for w in caught if "field still on" in str(w.message)]

    # 0.951 * 0.10 = 9.5e-2 Ry still applied: the measured 0.99 case.
    fired = warnings_from(0.99, 0.951)
    assert len(fired) == 1
    message = str(fired[0].message)
    # Both numbers a reader needs to act, named rather than implied.
    assert "field_scale" in message and "9.510e-01" in message
    assert "9.510e-02 Ry" in message

    # 6.1e-05 * 0.10 = 6.1e-06 Ry: the measured 0.5 case, whose total is right
    # to 5e-14 Ry. A guard that fires here is a guard nobody leaves on.
    assert warnings_from(0.5, 6.104e-05) == []

    # A field held at full strength is a deliberate calculation, and
    # field_energy and magnetic_field already say so on the result.
    assert warnings_from(1.0, 1.0) == []

    # An unconverged run has worse problems and says so elsewhere.
    assert warnings_from(0.99, 0.951, converged=False) == []


def test_the_two_halves_of_dr2_add_up_to_dr2(density, cell):
    """``scf_accuracy`` is a sum of two very differently weighted terms.

    The charge half is a Hartree energy, weighted by ``1/G^2``; the
    magnetization half has a constant weight and keeps its ``G = 0`` component.
    Reporting only their sum hides which one is still moving, and on a magnetic
    cell they are far enough apart at ``G_min`` that a ``dr2`` under
    ``conv_thr`` bounds the moment much more weakly than it bounds the charge.
    """
    from defumat.basis.gvectors import generate_gvectors
    from defumat.scf.potential import scf_accuracy, scf_accuracy_terms

    gvectors = generate_gvectors(cell, ecut=8.0, grid=GRID)
    residual = jnp.asarray(np.random.default_rng(7).normal(size=(4,) + GRID) * 1e-3)
    charge, magnetic = scf_accuracy_terms(residual, gvectors, cell)
    assert float(charge) > 0.0 and float(magnetic) > 0.0
    assert float(charge + magnetic) == pytest.approx(
        float(scf_accuracy(residual, gvectors, cell)), rel=1e-14
    )

    # nspin = 1 has no magnetization to be inaccurate about.
    charge1, magnetic1 = scf_accuracy_terms(residual[:1], gvectors, cell)
    assert float(magnetic1) == 0.0
    assert float(charge1) == pytest.approx(
        float(scf_accuracy(residual[:1], gvectors, cell)), rel=1e-14
    )


def _build(text: str):
    from defumat.io.pwin import parse_pw_input
    from defumat.system.builder import build_system
    return build_system(parse_pw_input(text))


_FSM_SOC = """
 &control
    calculation = 'scf'
 /
 &system
    ibrav = 3, celldm(1) = 5.217, nat = 1, ntyp = 1,
    ecutwfc = 25.0, ecutrho = 200.0,
    occupations = 'smearing', smearing = 'gaussian', degauss = 0.05
    noncolin = .true.
    nosym = .true.
    lspinorb = %s
    starting_magnetization(1) = 0.5
    constrained_magnetization = 'fsm'
    lambda = 0.02
    fixed_magnetization(3) = 2.0
 /
 &electrons
    conv_thr = 1.0d-8
 /
ATOMIC_SPECIES
 Fe 55.847 Fe.rel-pbe-spn-rrkjus_psl.0.2.1.UPF
ATOMIC_POSITIONS (alat)
 Fe 0.0 0.0 0.0
K_POINTS gamma
"""


def test_the_fixed_spin_moment_search_refuses_spin_orbit_coupling():
    """Its secant models ``dm_a/dB_b`` as diagonal, and SOC is what breaks that.

    ``_secant_step`` measures a susceptibility per cartesian component and
    inverts it the same way, so it is three independent one-dimensional
    searches. Spin-orbit coupling ties the moment to the lattice: pushing along
    x moves it along z too. The scheme still converges sometimes, which is
    exactly why it needs a refusal rather than a warning -- nothing in the
    output says which time it was.
    """
    _build(_FSM_SOC % ".false.")  # the collinear-axis case is still allowed
    with pytest.raises(ValueError, match="dm_a/dB_b is diagonal"):
        _build(_FSM_SOC % ".true.")


_REDUCEBF = """
 &control
    calculation = 'scf'
 /
 &system
    ibrav = 1, celldm(1) = 10.0, nat = 1, ntyp = 1, ecutwfc = 15.0,
    occupations = 'smearing', smearing = 'gaussian', degauss = 0.02
    nspin = 2
    nosym = .true.
    starting_magnetization(1) = 0.5
    reducebf = %s
 /
 &electrons
    conv_thr = 1.0d-6
 /
ATOMIC_SPECIES
 H 1.008 H.pz-vbc.UPF
ATOMIC_POSITIONS crystal
 H 0.0 0.0 0.0
K_POINTS gamma
"""


@pytest.mark.parametrize("value", ["0.5", "0.9", "1.0"])
def test_reducebf_is_accepted_inside_elks_range(value):
    assert _build(_REDUCEBF % value).reducebf == float(value)


@pytest.mark.parametrize("value", ["0.0", "0.49", "1.01", "2.0", "-1.0"])
def test_reducebf_outside_elks_range_is_refused(value):
    """Elk stops outside ``[0.5, 1]`` and this took anything at all.

    The range is not arbitrary: above 1 the symmetry-breaking field *grows*
    every iteration, and below 0.5 it is gone before the density has responded
    to it, so the run is the unmagnetised one with a slower start.
    """
    with pytest.raises(ValueError, match=r"outside \[0.5, 1\]"):
        _build(_REDUCEBF % value)
