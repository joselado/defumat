"""Second derivatives on a cell that is a **supercell**, which is its own regime.

Both bugs pinned here were invisible on every other cell committed to this
repository, and both were found by running the four-atom conventional cubic cell
of fcc aluminium (``al4-metal.in``) where every earlier case had one or two
atoms in a primitive cell. Neither needs a Quantum ESPRESSO reference to detect:
each is an identity the code must satisfy against itself.

**Why a supercell is different, and it is not that it is bigger.** Its atoms sit
at exact fractions of the cell, so two things become *exact* that are otherwise
only approximate. The structure factor vanishes identically on a whole set of
G-vectors -- the extinction rule -- rather than cancelling to round-off; and the
atom permutations of the point group acquire cycles longer than a transposition.
The first breaks ``abs``; the second breaks anything that pairs an atom label
with a spatial one.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pypresso.basis.fft import g_to_r
from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.scf import Calculation
from pypresso.system import build_system
from pypresso.system.symmetry import atom_mapping, symmetrize_atom_pair_tensor

pytestmark = pytest.mark.unit

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: The cells, and what each one's atom permutations look like. Only the first
#: has a permutation that is not an involution, which is the whole point.
SUPERCELL = "al4-metal"
PRIMITIVE = ("al2-metal", "si-epsilon")


def _build(case):
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    return system, Calculation(system, pseudos)


def test_the_supercell_is_the_only_case_with_a_non_involutive_permutation():
    """The premise of both tests below, asserted rather than assumed.

    If a future cell makes one of the primitive cases non-involutive, the two
    identities below start testing something there too and this test says so by
    failing -- which is better than the tests quietly becoming stronger without
    anyone noticing what they now cover.
    """
    system, calculation = _build(SUPERCELL)
    mapping = atom_mapping(system.cell, system.structure, calculation.symmetries)
    assert not np.array_equal(mapping, np.argsort(mapping, axis=1)), (
        "al4 must have an operation whose atom permutation is not an involution"
    )
    for case in PRIMITIVE:
        system, calculation = _build(case)
        mapping = atom_mapping(system.cell, system.structure, calculation.symmetries)
        assert np.array_equal(mapping, np.argsort(mapping, axis=1)), (
            f"{case} is involutive, which is why it cannot see the mapping direction"
        )


def _invariant_displacement_field(calculation, system):
    """``drho_{a,i}(r) = -d/dx_i g(r - tau_a)``: invariant by construction.

    It is what a rigid displacement of a rigid charge cloud on each atom
    produces to first order, so the *set* over ``(a, i)`` carries exactly the
    symmetry a response does and averaging it over the group must return it
    unchanged.
    """
    gvectors = calculation.basis.dense
    g = np.asarray(gvectors.cartesian(system.cell))
    form = np.exp(-(g**2).sum(axis=1) * 0.5)
    tau = np.asarray(system.structure.positions)
    fields = []
    for atom in range(len(tau)):
        phase = np.exp(-1j * (g @ tau[atom]))
        for cart in range(3):
            component = -1j * g[:, cart] * form * phase
            fields.append(np.real(np.asarray(
                g_to_r(jnp.asarray(component), gvectors.fft_index, gvectors.grid)
            )))
    return np.asarray(fields).reshape((len(tau), 3) + tuple(gvectors.grid))


@pytest.mark.parametrize("case", (SUPERCELL,) + PRIMITIVE)
def test_symdvscf_returns_an_invariant_response_unchanged(case):
    """``symmetrize_atom_displacement`` must be the identity on an invariant field.

    Worth **0.33** on al4 when the average ran over ``irt`` instead of its
    inverse -- an operation carries a displacement of atom ``a`` to atom
    ``irt[s,a]``, so labelling the result by the atom it lands on puts ``S^-1``
    under the sum. On the two primitive cells the two directions coincide and
    this passed throughout.
    """
    system, calculation = _build(case)
    field = _invariant_displacement_field(calculation, system)
    averaged = np.asarray(
        calculation.symmetrize_atom_displacement(jnp.asarray(field)[:, :, None])
    )[:, :, 0]
    assert np.abs(averaged - field).max() / np.abs(field).max() < 1e-12


@pytest.mark.parametrize("case", (SUPERCELL,) + PRIMITIVE)
def test_the_response_symmetrisations_are_projectors(case):
    """``P(P(x)) = P(x)``, which is what a group average *is*.

    The cheapest statement of the same fact as the test above, and the one that
    needs no invariant object to compare against: an average over a set that is
    not the group's action on this quantity is not idempotent. It scored 0.756
    on al4 before the fix.
    """
    system, calculation = _build(case)
    rng = np.random.default_rng(0)
    shape = ((system.structure.nat, 3, 1) + tuple(calculation.basis.dense.grid))
    once = np.asarray(calculation.symmetrize_atom_displacement(
        jnp.asarray(rng.standard_normal(shape))))
    twice = np.asarray(calculation.symmetrize_atom_displacement(jnp.asarray(once)))
    assert np.abs(twice - once).max() / np.abs(once).max() < 1e-12


def test_symdynph_gq_keeps_the_forward_mapping():
    """The companion symmetrisation does **not** share the direction.

    ``symmetrize_atom_pair_tensor`` carries two atom labels and no spatial
    argument, and there ``irt`` is right: a force-constant matrix built from a
    pair potential is invariant by construction, and averaging it is a no-op
    with ``irt`` and wrong by 0.29 with the inverse. The distinction is what
    fixes the direction in each case, so it is asserted rather than left to
    the reader.
    """
    system, calculation = _build(SUPERCELL)
    at = np.asarray(system.cell.at_alat) * system.cell.alat
    tau = np.asarray(system.structure.positions)
    nat = len(tau)
    matrix = np.zeros((nat, 3, nat, 3))
    shifts = [i * at[0] + j * at[1] + k * at[2]
              for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    for a in range(nat):
        for b in range(nat):
            for shift in shifts:
                d = tau[b] + shift - tau[a]
                r = float(np.linalg.norm(d))
                if r < 1e-8 or r > 12.0:
                    continue
                n = d / r
                outer = np.outer(n, n)
                block = np.exp(-r) * outer + (-np.exp(-r) / r) * (np.eye(3) - outer)
                matrix[a, :, b, :] -= block
                matrix[a, :, a, :] += block

    mapping = atom_mapping(system.cell, system.structure, calculation.symmetries)
    forward = symmetrize_atom_pair_tensor(
        matrix, system.cell, calculation.symmetries, mapping)
    inverse = symmetrize_atom_pair_tensor(
        matrix, system.cell, calculation.symmetries, np.argsort(mapping, axis=1))
    scale = np.abs(matrix).max()
    assert np.abs(forward - matrix).max() / scale < 1e-12
    assert np.abs(inverse - matrix).max() / scale > 1e-3, (
        "the inverse mapping must be visibly wrong here, or this cell proves nothing"
    )


def test_the_structure_factor_vanishes_exactly_on_a_supercell():
    """The premise of the Ewald test below: `|rho(G)| == 0.0`, not 4e-16.

    A supercell puts its atoms at exact fractions, so every phase is exactly
    +-1 and the extinction rule holds in floating point. That is what turns
    ``abs``'s ``0/0`` from a measure-zero accident into something a whole class
    of cells meets every time.
    """
    system, calculation = _build(SUPERCELL)
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    g = np.asarray(calculation.basis.dense.cartesian(system.cell))
    charges = np.asarray([p.z_valence for p in pseudos])[np.asarray(system.structure.types)]
    rho = (charges * np.exp(1j * (g @ np.asarray(system.structure.positions).T))).sum(axis=1)
    assert (np.abs(rho) == 0.0).sum() > 50

    for case in PRIMITIVE:
        system, calculation = _build(case)
        pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
        g = np.asarray(calculation.basis.dense.cartesian(system.cell))
        charges = np.asarray([p.z_valence for p in pseudos])[np.asarray(system.structure.types)]
        rho = (charges * np.exp(1j * (g @ np.asarray(system.structure.positions).T))).sum(axis=1)
        assert (np.abs(rho) == 0.0).sum() == 0, f"{case} was expected never to reach zero"


@pytest.mark.parametrize("case", (SUPERCELL,) + PRIMITIVE)
def test_the_ewald_second_derivative_converges_to_its_finite_difference(case):
    """``jvp(grad E_ewald)`` against a finite difference of the same gradient.

    No SCF, no response, no symmetrisation: one pure function of the positions,
    differentiated two ways. The error must fall as ``h^2``, and with
    ``abs(rho)**2`` in the reciprocal sum it did not fall at all on al4 -- a
    fixed 3.0e-4 Ry/bohr^2 at every step from 1e-2 to 1e-5, because ``abs``'s
    derivative is ``0/0`` exactly where the structure factor vanishes. Halving
    the step must quarter the error, which is the statement that the two agree.
    """
    system, calculation = _build(case)
    positions = jnp.asarray(system.structure.positions)
    ewald = lambda p: calculation.at_positions(p).ewald
    gradient = jax.grad(ewald)
    tangent = jnp.zeros_like(positions).at[0, 0].set(1.0)
    _, exact = jax.jvp(gradient, (positions,), (tangent,))
    exact = np.asarray(exact)

    errors = []
    for step in (1e-2, 1e-3):
        difference = np.asarray(
            (gradient(positions + tangent * step) - gradient(positions - tangent * step))
            / (2 * step)
        )
        errors.append(np.abs(exact - difference).max())
    # h^2: a tenfold smaller step is a hundredfold smaller error. Allowing a
    # factor of three of slack still fails a discrepancy that does not converge.
    assert errors[1] < errors[0] / 30.0, f"errors {errors} do not fall as h^2"
    assert errors[1] < 1e-6


# ---------------------------------------------------------------------------
# ``nosym`` reaches the symmetrisation, not only the k-point reduction.
# ---------------------------------------------------------------------------

def test_nosym_switches_the_symmetrisation_off_and_not_just_the_kpoints():
    """``calculation.symmetries`` is the crystal's group; it is not the switch.

    The group is deliberately kept whole even under ``nosym`` -- the FFT box is
    sized from the fractional translations whatever the input says -- so
    :attr:`Calculation.use_symmetry` is what a consumer must read, and the
    ``symmetrize_*`` methods are where that is read once instead of at every
    call site. Three places in ``response/`` reached for the group instead and
    symmetrised a ``nosym`` run: the force constants (``symdynph_gq``) and the
    Born charges twice over (``symtensor``). The stress and the forces wrote the
    guard out by hand and got it right, which is how the asymmetry survived.

    ``al2-metal.in`` is the case it mattered on: it carries ``nosym = .true.``
    and its documentation says its response is not symmetrised, while its twelve
    operations were being applied.
    """
    system, calculation = _build("al2-metal")
    assert system.nosym, "this input is the nosym one"
    assert calculation.symmetries.nsym > 1, "the group is kept whole regardless"
    assert not calculation.use_symmetry

    rng = np.random.default_rng(0)
    nat = system.structure.nat
    pair = rng.standard_normal((nat, 3, nat, 3))
    single = rng.standard_normal((nat, 3, 3))
    field = rng.standard_normal((nat, 3, 1) + tuple(calculation.basis.dense.grid))
    assert np.array_equal(calculation.symmetrize_atom_pair_tensor(pair), pair)
    assert np.array_equal(calculation.symmetrize_atom_tensor(single), single)
    assert np.array_equal(
        np.asarray(calculation.symmetrize_atom_displacement(jnp.asarray(field))), field
    )


def test_a_symmetric_run_still_symmetrises():
    """The other half of the switch, so the test above cannot pass vacuously."""
    system, calculation = _build(SUPERCELL)
    assert not system.nosym
    assert calculation.use_symmetry

    rng = np.random.default_rng(0)
    nat = system.structure.nat
    pair = rng.standard_normal((nat, 3, nat, 3))
    averaged = calculation.symmetrize_atom_pair_tensor(pair)
    assert np.abs(averaged - pair).max() > 1e-3, "48 operations must do something"
