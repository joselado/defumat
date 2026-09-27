"""P123: spin-orbit coupling to first order on a spin spiral, against a supercell.

There is no reference code for the quantity -- ``pw.x`` has no spiral, and Elk
switches spin-orbit coupling off on one (``init0.f90:108``) -- so the checks are
identities and a second route to the same physics, all on the nickel chain with an
iodine beside each bond (``tests/data/qe/nii-chain-spiral.in``), a cell with no
inversion centre:

1. **the symmetry's own zero**: the cell's mirror in the plane of its atoms
   leaves only ``V_y``, so ``V_x`` and ``V_z`` are the floor, and ``V_y`` is not;
2. **``V(-q) = -V(q)``**, two independent SCFs, since the spiral at ``(q, n)`` is
   the texture at ``(-q, -n)``;
3. **the four-cell supercell at the unfolded density**, where the two spinor
   components share one sphere and ``dD``'s whole spin structure is contracted
   with no mask: its first-order energy, turned to the spiral's axis, is four
   times ``n . V``. This is the check the mask has to pass, since without it the
   two layouts disagree by the transverse cross terms;
4. **the force theorem on the same supercell at a scaled coupling**,
   ``H0 + lambda dD`` diagonalised for both senses of the spiral: the odd part
   of the free energy over ``lambda`` tends to ``V_y`` as ``lambda -> 0``. The
   remainder is *linear* in ``lambda``, not quadratic: the transverse part of
   the coupling enters second order as a quadratic form, and its
   antisymmetric piece (the ``L_-`` transition to ``k - q`` against the ``L_+``
   one to ``k + q``) changes sign with ``q``, so there is a second-order odd
   part, and on this cell it is larger than the first-order one. So the check
   extrapolates ``O / lambda`` through three couplings to zero rather than
   reading it at one. The coupling is scaled through ``soc_scale``, which a
   norm-conserving dataset admits between 0 and 1 because there it reaches
   ``dvan_so`` alone and linearly (``tests/unit/test_spiral_soc.py`` holds the
   blend against ``H(0) + s dD`` entry by entry). At the full coupling the same odd part is ten times ``V_y`` on
   this cell, iodine's coupling being far from small, which is why the check
   is the limit and not the value at 1.

The supercell is never converged on its own. Its SCF would have nothing holding
the texture a spiral (the four-cell cobalt helix limit-cycles for that reason,
``co-helix4-spiral.in``), and the unfolded spiral density is its stationary
density exactly, the generalized Bloch theorem being exact at a commensurate
``q``.
"""

from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat import Calculator
from defumat.forces.energy import _spinor_projector_energies
from defumat.forces.torque import rotate_texture
from defumat.scf.driver import Calculation
from defumat.workflows.anisotropy import _first_order_operator
from defumat.workflows.nscf import fixed_density_states
from defumat.workflows.spiral import unfold_spiral_density
from tests.conftest import GENERATED

pytestmark = [pytest.mark.regression, pytest.mark.slow]

SPIRAL = GENERATED / "nii-chain-spiral.in"
SUPERCELL = GENERATED / "nii-chain-4cell.in"

#: ``V(-q) + V(q)`` against ``|V_y|``, two SCFs at the input's ``conv_thr = 1e-9``.
#: Measured 1.3e-5 (1.54e-5 meV on 1.218).
ODD_TOLERANCE = 1.0e-4
#: The spiral against the supercell at one density, Ry per cell. Both come from
#: this code at ``conv_thr = 1e-10`` on the diagonalisation, and the floor is the
#: eigensolver's: measured 9.0e-12 here and 1.8e-11 on the same chain at 40 Ry.
IDENTITY_RY = 1.0e-10
#: The scaled-coupling limit against ``V_y``, relative. Measured 2.3e-5: the
#: quadratic through 0.05, 0.1 and 0.2 lands at -8.95158e-5 Ry per cell against
#: -8.95137e-5, and its slope, the second-order odd part, is +1.09e-5, a tenth of
#: ``V_y`` and of the opposite sign.
LIMIT_TOLERANCE = 1.0e-3


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _spiral(q3: float, pseudo_dir: Path):
    text = SPIRAL.read_text().replace("spiral_q(3) = 0.25", f"spiral_q(3) = {q3}")
    calculator = Calculator.from_text(text, pseudo_dir, announce=False)
    scf = calculator.get_scf()
    assert scf.converged
    return calculator, np.asarray(scf.density), calculator.get_spiral_spin_orbit_energy()


def _about_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


#: The spiral's axis ``z`` turned onto ``+y`` and onto ``-y``.
TO_PLUS_Y, TO_MINUS_Y = _about_x(-np.pi / 2), _about_x(np.pi / 2)


def _supercell_leg(pseudo_dir, rotation, coupling):
    """The supercell at the unfolded, turned density: free energy, and ``E1`` at 0.

    ``coupling`` is ``lambda`` in ``H0 + lambda dD``, which is ``soc_scale``.
    """
    spiral, density, _ = _spiral(0.25, pseudo_dir)
    supercell = Calculator.from_file(SUPERCELL, pseudo_dir, announce=False)
    system = supercell.system.with_soc_scale(coupling)
    calculation = Calculation(system, supercell.pseudos)
    unit = Calculation(spiral.system, spiral.pseudos).basis.dense.grid
    # The identity needs the potential evaluated on the same points, which is
    # the supercell's grid being the unit cell's repeated; see the input.
    assert calculation.basis.dense.grid == (unit[0], unit[1], 4 * unit[2])
    lab = unfold_spiral_density(density, spiral.system.spiral_q, (1, 1, 4),
                                calculation.basis.dense.grid)
    delta, _ = _first_order_operator(calculation, None)
    calculation, _, eigenvalues, states = fixed_density_states(
        system, supercell.pseudos, rotate_texture(lab, rotation),
        conv_thr=1.0e-10, calculation=calculation,
    )
    weights, levels = calculation.occupations(jnp.asarray(eigenvalues))
    free = float(np.sum(np.asarray(weights) * np.asarray(eigenvalues))
                 + levels.get("smearing", 0.0))
    first = None
    if coupling == 0.0:
        first, _ = _spinor_projector_energies(
            jnp.asarray(states), calculation.projectors.vkb, delta, None,
            weights, jnp.asarray(eigenvalues))
        first = float(first)
    return free, first


def test_the_vector_lies_along_the_one_axis_the_mirror_allows(pseudo_dir):
    _, _, first = _spiral(0.25, pseudo_dir)
    vx, vy, vz = first.vector
    assert abs(vy) > 1.0e-7
    assert abs(vx) < 1.0e-3 * abs(vy) and abs(vz) < 1.0e-3 * abs(vy)
    assert first.trace == 0.0


def test_the_first_order_energy_is_odd_in_q(pseudo_dir):
    """Two SCFs, one per sign of ``q``: ``V(-q) = -V(q)``."""
    _, _, plus = _spiral(0.25, pseudo_dir)
    _, _, minus = _spiral(-0.25, pseudo_dir)
    np.testing.assert_allclose(minus.vector, -plus.vector,
                               atol=ODD_TOLERANCE * abs(plus.vector[1]))


def test_the_spiral_is_the_supercell_at_the_unfolded_density(pseudo_dir):
    """Four cells, both components on one sphere, the full ``dD`` contracted."""
    _, _, first = _spiral(0.25, pseudo_dir)
    _, along_y = _supercell_leg(pseudo_dir, TO_PLUS_Y, 0.0)
    assert abs(along_y / 4.0 - first.energy((0, 1, 0))) < IDENTITY_RY
    _, along_z = _supercell_leg(pseudo_dir, np.eye(3), 0.0)
    assert abs(along_z) / 4.0 < 1.0e-3 * abs(first.vector[1])


def test_the_force_theorem_at_a_scaled_coupling_tends_to_it(pseudo_dir):
    """``O(lambda) / lambda -> V_y``, extrapolated through three couplings.

    ``O`` is half the difference of the free energies of the two senses of the
    spiral, per cell. A quadratic in ``lambda`` through ``O / lambda`` at 0.05,
    0.1 and 0.2 is extrapolated to zero, and the slope it has is the
    second-order odd part, which is asserted to be there: a slope of zero would
    say the fit had nothing to extrapolate across. On this cell it is a tenth of
    ``V_y``; on the same chain at 40 Ry, where ``V_y`` nearly cancels over the
    k-points, it is seventeen times.
    """
    _, _, first = _spiral(0.25, pseudo_dir)
    couplings = (0.05, 0.1, 0.2)
    ratios = []
    for coupling in couplings:
        plus, _ = _supercell_leg(pseudo_dir, TO_PLUS_Y, coupling)
        minus, _ = _supercell_leg(pseudo_dir, TO_MINUS_Y, coupling)
        ratios.append((plus - minus) / 8.0 / coupling)
    curvature, slope, limit = np.polyfit(couplings, ratios, 2)
    assert limit == pytest.approx(first.vector[1], rel=LIMIT_TOLERANCE)
    assert abs(slope) > 0.01 * abs(first.vector[1])
