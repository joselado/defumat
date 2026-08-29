"""P18: the field and constraint potentials, against QE's hand-derived ones.

:mod:`pypresso.scf.fields` writes down the *energy* of a field or a penalty and
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

import jax.numpy as jnp
import numpy as np
import pytest

from pypresso.scf.fields import MagneticField
from pypresso.scf.locals import LocalRegions
from pypresso.system.cell import Cell

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
    return LocalRegions(weights=jnp.asarray(weights), radii=(1.0,), scheme="qe")


def _moments(field: MagneticField, density, cell):
    return np.asarray(field.local_moments(density, cell))


def _potential(field: MagneticField, density, cell, scale: float = 1.0):
    v, _, _ = field.potential(density, cell, scale)
    return np.asarray(v)


def _weighted(regions, values, grid_scale):
    """``sum_a w_a(r) values[a, ipol]`` as a ``(3, ...)`` grid array."""
    return np.einsum("anmk,ac->cnmk", np.asarray(regions.weights), values)


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
    from pypresso.io.pwin import parse_pw_input
    from pypresso.system import build_system

    source = """ &control
    calculation = 'scf'
 /
 &system
    ibrav = 3, celldm(1) = 5.217, nat = 1, ntyp = 1, ecutwfc = 12.0
    nspin = 2
    starting_magnetization(1) = 0.5
    B_field(3) = -0.02
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

    from pypresso.io.pwin import parse_pw_input
    from pypresso.system import build_system

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
