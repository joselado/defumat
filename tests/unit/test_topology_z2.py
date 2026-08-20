"""Z2 invariants: the Wilson-loop route, the parity route, and their agreement.

The structure of this file follows what can go wrong.

* The **arithmetic** -- unitarisation, the largest-gap reference line, the
  three-sine crossing test, the assembly of the four 3D indices -- is checked on
  hand-made inputs, where the answer is a statement about the formula and not
  about any physics.
* **Gauge invariance** is checked with a full random unitary at every point of
  the loop, which is the freedom a degenerate eigensolver has.
* The **physics** is checked on models whose invariants are known in closed
  form: the doubled Qi-Wu-Zhang model (where ``nu = C_up mod 2`` ties the Z2 to
  a Chern number computed by completely different code), the Kane-Mele
  transition at ``m = 3 sqrt(3) lambda``, and the three-dimensional lattice
  Dirac model, whose four indices can be written down from the sign of
  ``m + sum cos k_i`` at the eight TRIM.
* Finally the two routes are required to **agree** wherever both apply. They
  share nothing but the state set: one sweeps a half-zone mesh and counts
  crossings, the other multiplies eight numbers.

The doubled-QWZ values ``nu = 1, 1, 0, 0`` at ``m = -0.5, 1.0, 3.0, 4.5`` are
``elkpy``'s (``tests/test_wilson_gauge_invariance.py``), reproduced here to the
integer.
"""

import numpy as np
import pytest

from pypresso.topology import (
    ArrayStates,
    ModelSource,
    chern_number,
    combine_3d,
    z2_invariant,
    z2_invariant_3d,
)
from pypresso.topology.wilson import (
    _orientation,
    largest_gap_center,
    wilson_loop,
    z2_from_centers,
)
from tests.models import (
    DOUBLED_QWZ_INVERSION,
    WILSON_FERMION_INVERSION,
    doubled_qwz,
    kane_mele,
    kane_mele_critical_mass,
    qwz,
    random_gauge,
    wilson_fermion_3d,
    wilson_fermion_indices,
)

pytestmark = pytest.mark.unit


# --- the arithmetic ---------------------------------------------------------

def test_largest_gap_sits_in_the_long_arc():
    center = largest_gap_center([-0.5, 0.2])
    assert not (-0.5 < center < 0.2)
    # Equidistant from both, going the long way round the circle.
    def arc(a, b):
        return abs(((a - b + np.pi) % (2 * np.pi)) - np.pi)
    assert arc(center, -0.5) == pytest.approx(arc(center, 0.2), abs=1e-12)


def test_crossing_parity_of_a_flow_that_does_not_move_is_trivial():
    centers = np.tile(np.array([0.1, 2.0, -2.0, 1.2]), (5, 1))
    z2, _ = z2_from_centers(centers)
    assert z2 == 0


def test_an_odd_number_of_charge_centres_is_refused():
    """The crossing count is only a parity, and only for an even manifold."""
    with pytest.raises(ValueError, match="odd"):
        z2_from_centers(np.tile(np.array([0.1, 2.0, -2.0]), (5, 1)))


def test_orientation_signs_are_the_reference_ones():
    """The three-sine directed area, pinned against ``elkpy``'s own values.

    ``_orientation(a, b, c)`` is negative when ``c`` lies inside the arc swept
    from ``a`` to ``b`` and positive when it does not, and a crossing is counted
    on the negative sign. Getting it backwards inverts every Z2.
    """
    assert _orientation(0.0, 1.0, 0.5) < 0
    assert _orientation(0.0, 1.0, 1.5) > 0
    assert _orientation(0.0, 1.0, -0.5) > 0


def test_a_kramers_partner_switch_is_the_nontrivial_flow():
    """The textbook picture, as arithmetic on the charge centres alone.

    Two centres leaving a TRI plane together at ``0`` and arriving at the next
    one at ``+pi`` and ``-pi`` -- the same point of the circle -- have swapped
    partners, and that is the Z2-nontrivial flow. Stopping short of ``pi``
    returns them to where they started as a *pair*, which is the trivial one.
    The reference line is nowhere in the input; the counting has to find it.
    """
    switching = np.linspace(0.0, np.pi - 1e-6, 41)
    assert z2_from_centers(np.column_stack([switching, -switching]))[0] == 1

    returning = np.linspace(0.0, 1.0, 41)
    assert z2_from_centers(np.column_stack([returning, -returning]))[0] == 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_wannier_centers_are_gauge_invariant(seed):
    """A different unitary at every k-point of the loop must change nothing.

    The Wilson loop's eigenvalues are invariant under
    ``F_i -> g_i^H F_i g_{i+1}`` around a closed loop, because the ``g`` cancel
    cyclically -- which is why no gauge fixing is needed anywhere in this
    subpackage.
    """
    # A generic pumping value, not a time-reversal-invariant one: there the
    # charge centres are pinned to 0 or pi, and one sitting exactly on the
    # branch cut comes back as +pi or -pi according to round-off. That is the
    # same point of the circle and no invariant sees it, but it makes a
    # component-by-component comparison meaningless.
    points = np.zeros((8, 3))
    points[:, 0] = np.arange(8) / 8
    points[:, 1] = 0.137
    source = ModelSource(hamiltonian=doubled_qwz(1.0), nocc=2)
    states = source.states(points)
    plain = wilson_loop(states, np.array([1, 0, 0]))

    gauge = random_gauge(seed, (8, 2), unitary=True)
    rotated = np.einsum("kmn,kna->kma", gauge, np.asarray(states.coefficients))
    turned = wilson_loop(ArrayStates(coefficients=rotated), np.array([1, 0, 0]))
    assert np.allclose(plain, turned, atol=1e-10)


def test_three_dimensional_assembly_and_its_consistency_check():
    trivial = {(a, o): 0 for a in range(3) for o in (0.0, 0.5)}
    strong = dict(trivial)
    for axis in range(3):
        strong[(axis, 0.0)] = 1
    result = combine_3d(strong)
    assert (result.nu0, result.nu) == (1, (0, 0, 0))
    assert result.nu0_by_axis == (1, 1, 1)

    weak = dict(trivial)
    weak[(1, 0.0)] = weak[(1, 0.5)] = 1
    assert combine_3d(weak).nu == (0, 1, 0)

    broken = dict(trivial)
    broken[(0, 0.0)] = 1
    with pytest.raises(ValueError, match="disagrees across axes"):
        combine_3d(broken)


def test_all_six_planes_are_required():
    with pytest.raises(ValueError, match="six planes"):
        combine_3d({(0, 0.0): 0})


# --- the physics ------------------------------------------------------------

@pytest.mark.parametrize(
    "mass, expected", [(-0.5, 1), (1.0, 1), (3.0, 0), (4.5, 0)]
)
def test_doubled_qwz_z2_matches_the_spin_chern_number(mass, expected):
    """``nu = C_up mod 2``, with the two sides computed by unrelated code.

    The Chern number comes from the plaquette sum over a single spin sector and
    the Z2 from a Wannier-centre sweep over both. ``elkpy`` pins the same four
    numbers.
    """
    chern = chern_number(
        ModelSource(hamiltonian=qwz(mass), nocc=1), shape=(24, 24)
    ).chern_number
    assert abs(chern - round(chern)) < 1e-12
    assert int(round(abs(chern))) % 2 == expected

    source = ModelSource(
        hamiltonian=doubled_qwz(mass), nocc=2, inversion=DOUBLED_QWZ_INVERSION
    )
    assert z2_invariant(source, nloop=41, npump=21).z2 == expected


@pytest.mark.parametrize("mass, expected", [(-0.5, 1), (1.0, 1), (3.0, 0), (4.5, 0)])
def test_wilson_and_parity_agree_on_an_inversion_symmetric_model(mass, expected):
    """The cross-check the whole design is arranged to make possible."""
    source = ModelSource(
        hamiltonian=doubled_qwz(mass), nocc=2, inversion=DOUBLED_QWZ_INVERSION
    )
    wilson = z2_invariant(source, nloop=31, npump=16).z2
    parity = z2_invariant(source, method="parity", dimension=2).z2
    assert wilson == parity == expected


@pytest.mark.parametrize("nloop, npump", [(21, 11), (31, 16), (41, 21)])
def test_z2_does_not_move_under_mesh_refinement(nloop, npump):
    source = ModelSource(hamiltonian=doubled_qwz(1.0), nocc=2)
    assert z2_invariant(source, nloop=nloop, npump=npump).z2 == 1


def test_kane_mele_transition_is_at_the_analytic_critical_mass():
    """``m_c = 3 sqrt(3) lambda`` -- ``pyqula``'s ``examples/2d/z2_transition``.

    A sublattice imbalance breaks inversion, so there is no parity route here
    and the Wilson loop is on its own. That is the point of the case.
    """
    soc = 0.05
    critical = kane_mele_critical_mass(soc)
    below = ModelSource(hamiltonian=kane_mele(soc=soc, mass=0.5 * critical), nocc=2)
    above = ModelSource(hamiltonian=kane_mele(soc=soc, mass=1.5 * critical), nocc=2)
    assert z2_invariant(below, nloop=31, npump=16).z2 == 1
    assert z2_invariant(above, nloop=31, npump=16).z2 == 0


@pytest.mark.parametrize("mass", [-4.0, -2.0, 0.0, 2.0])
def test_three_dimensional_indices_match_the_closed_form(mass):
    """All four phases of the lattice Dirac model, by both routes.

    ``(0;000)``, ``(1;000)``, ``(0;111)`` and ``(1;111)`` in turn -- including
    the weak indices, which the strong one alone would not distinguish.
    """
    source = ModelSource(
        hamiltonian=wilson_fermion_3d(mass), nocc=2, inversion=WILSON_FERMION_INVERSION
    )
    expected = wilson_fermion_indices(mass)
    parity = z2_invariant_3d(source, method="parity")
    assert (parity.nu0, parity.nu) == expected

    wilson = z2_invariant_3d(source, method="wilson", nloop=16, npump=9)
    assert (wilson.nu0, wilson.nu) == expected
    assert len(set(wilson.nu0_by_axis)) == 1


def test_parity_refuses_a_model_without_an_inversion_centre():
    source = ModelSource(hamiltonian=kane_mele(mass=0.1), nocc=2)
    with pytest.raises(ValueError, match="no inversion representation"):
        z2_invariant(source, method="parity", dimension=2)


def test_an_unknown_z2_method_is_refused_by_name():
    source = ModelSource(hamiltonian=doubled_qwz(1.0), nocc=2)
    with pytest.raises(ValueError, match="unknown Z2 method"):
        z2_invariant(source, method="pfaffian")
