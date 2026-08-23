"""P25: the dynamical matrix at ``Gamma``, and the four things it is checked against.

The matrix itself is one ``jvp`` of the gradient the force already is, so almost
nothing here is a transcription and almost everything can be wrong silently. The
checks are therefore chosen to share as little machinery with it as possible:

* **A rigid translation of the crystal is a translation of its density.** Sum
  the response densities of all the atoms along one direction and the answer must
  be ``-d(rho)/dx``, obtained by differentiating the converged density on the
  grid. It holds only for the *screened* response -- the bare one is 52% off --
  so it tests the linear solve, the screening kernel and the symmetrisation
  together against something none of them touches.
* **Finite-differenced forces.** Displace an atom, re-converge, and difference
  the forces: the only check that reaches the *response* half of the second
  derivative, since differentiating the force with respect to the positions
  alone would reproduce the frozen Hessian and nothing else. Run on the
  **unshifted** grid with ``nosym``, because that is the grid that is closed
  under the point group -- a wedge is the irreducible set of the *undisplaced*
  crystal and is not a valid sample for a displaced one.
* **The wedge against the whole grid.** The same unshifted sample, once reduced
  to its wedge with the response symmetrised and once whole with the
  symmetrisation idle. This is the only check ``symmetrize_atom_displacement``
  and ``symmetrize_atom_pair_tensor`` have, exactly as it was the only one
  ``symmetrize_directional`` had in P24: QE computes the wedge route alone.
* **``ph.x``**, on the ten-point wedge of ``si-epsilon.in`` -- the *vendored*
  binary, regenerated, for the reason ``test_response.py`` gives.

The acoustic modes are the diagnostic rather than a target. Translating the
crystal costs no energy, so three frequencies are zero exactly; what comes out
instead is the finite basis's own error, and ``ph.x`` does not impose the sum
rule either. It is a difference of large numbers -- ``D_00 + D_01`` against
``D_00`` -- so it magnifies whatever is left far more than the optical mode
does, and that is what makes it worth asserting on.
"""

from functools import lru_cache
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from pypresso.basis.fft import g_to_r, r_to_g
from pypresso.forces import compute_forces
from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.response.phonon import _require_one_spin_channel, dynamical_matrix
from pypresso.scf import Calculation, run_scf
from pypresso.system import build_system

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: What the **vendored** ``ph.x`` prints for ``si-epsilon.in``, from the same
#: run that gave the dielectric constant: three acoustic modes at 2.045258 and
#: the triply degenerate optical mode at 510.151844 cm^-1. Committed as
#: ``reference.out.ph-si-epsilon``.
QE_OPTICAL = 510.151844
QE_ACOUSTIC = 2.045258

#: How far the optical mode may sit from ``ph.x``'s, in cm^-1. The measured
#: difference is 0.05, which is 1e-4 relative -- the same floor the dielectric
#: constant has at 4.3e-5, and for the same reason: QE interpolates every radial
#: form factor from a ``dq = 0.01`` table where this code integrates it directly
#: (``tests/tolerances.py``).
OPTICAL_TOLERANCE = 0.2

#: The acoustic residue is **not** required to match ``ph.x``'s. Both are the
#: same basis-set error made slightly differently, and at 1e-4 of the force
#: constants there is nothing in either to reproduce; what is asserted is that
#: it is small on the scale of the optical mode.
ACOUSTIC_CEILING = 8.0

#: Cubic silicon: nothing here imposes the crystal class, so a departure from it
#: measures the rotation convention and the atom permutation rather than
#: confirming them.
CUBIC_TOLERANCE = 1e-9

#: How far a finite-differenced force constant may sit from the analytic one,
#: in Ry/bohr^2. Measured: 2.55e-5 at a step of 1e-2 bohr and 2.14e-5 at 3e-3,
#: against force constants of 0.2865 -- 7.5e-5 relative, and *improving* with
#: the step, so what is left is the floor rather than truncation. That floor is
#: the same 1e-4 relative as everything else here.
FD_TOLERANCE = 5e-5

#: What the **vendored** ``ph.x`` prints for ``al2-metal.in`` -- two-atom fcc
#: aluminium, `marzari-vanderbilt` smearing, committed as
#: ``reference.out.ph-al2-metal``. Three acoustic modes at the basis residue and
#: three the cell doubling folds in from the zone boundary. This is the metal
#: reference P24c measured its *refusal* against and P28 lifted it with.
QE_AL2 = (1.108857, 1.827469, 1.924700, 146.710511, 146.714378, 311.035401)

#: How far aluminium's three real modes may sit from ``ph.x``'s, in cm^-1.
#: Measured: **0.0031** on the worst of them, an order tighter than silicon's
#: 0.049 -- the folded pair at 146.711240 against 146.710511 and 146.714378,
#: and the zone-centre mode at 311.033545 against 311.035401. The looser
#: tolerance is kept because the floor is the same one (QE's ``dq = 0.01``
#: form-factor table against direct integration) and there is no reason for
#: this cell to hold it more tightly than silicon on another day.
AL2_OPTICAL_TOLERANCE = 0.05

#: ``max|sum_b D_(a i)(b j)|`` in Ry/bohr^2, against on-site force constants of
#: 0.0476. This is the number the refusal was about: the ``wg``-weighted
#: assembly gave the acoustic modes 155.7 cm^-1 here, half the optical
#: spectrum, and the split one gives **1.06e-5** -- 2.2e-4 relative, which is
#: silicon's own basis residue on this cell. Asserted rather than the acoustic
#: frequencies themselves, because it is the quantity that separates a weight
#: error from the basis error every code makes.
AL2_SUM_RULE = 5e-5

#: The wedge and the whole grid must agree to arithmetic. Measured: 2.7e-14 on
#: the matrix, which is what an exact group average of an exactly closed grid
#: gives and is why this number is not a tolerance so much as an assertion that
#: the two routes are the same calculation.
WEDGE_TOLERANCE = 1e-10


def _build(case: str):
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    return system, pseudos, Calculation(system, pseudos)


@lru_cache(maxsize=None)
def _phonons(case: str):
    """The converged ground state and the force constants of one input."""
    system, pseudos, calculation = _build(case)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-12,
                     max_iterations=100)
    phonons = dynamical_matrix(
        calculation, result.wavefunctions, result.eigenvalues, result.density,
        result.becsum,
    )
    return calculation, result, phonons


# ---------------------------------------------------------------------------
# Against ph.x.
# ---------------------------------------------------------------------------

def test_the_gamma_phonon_matches_quantum_espresso():
    """Silicon's optical mode against the vendored ``ph.x``.

    Everything on this side comes from differentiating. The bare perturbation
    ``dvqpsi_us`` is one ``jvp`` through ``at_positions``; the screening kernel
    ``dv_of_drho`` is one ``jvp`` of ``v_of_rho``; the response density is one
    ``jvp`` of the density builder; and ``dynmat0`` and ``drhodv`` -- the frozen
    second derivative and the electronic response -- are the two halves of a
    single ``jvp`` of the gradient that already gives the force. What is
    transcribed is the linear solve, the two symmetrisations and ``dyndia``.
    """
    _, _, phonons = _phonons("si-epsilon")
    assert phonons.converged
    optical = phonons.frequencies[3:]
    assert np.allclose(optical, optical[0], atol=1e-6), "the optical mode is triple"
    assert optical[0] == pytest.approx(QE_OPTICAL, abs=OPTICAL_TOLERANCE)


def test_the_acoustic_modes_are_the_basis_sets_own_error():
    """Three frequencies are zero by translation invariance, and are not imposed.

    ``ph.x`` prints 2.045258 cm^-1 here and this code prints about 4; neither
    number is physics. What both measure is that the plane-wave basis does not
    follow the atoms, so the energy depends slightly on where they sit relative
    to it. The assertion is that the residue is small next to the optical mode
    and that nothing has imposed it: :func:`dynamical_matrix` defaults to
    ``acoustic_sum_rule = False`` precisely so this stays visible.
    """
    _, _, phonons = _phonons("si-epsilon")
    assert phonons.acoustic_residue < ACOUSTIC_CEILING
    assert phonons.acoustic_residue > 0.0, "the sum rule must not have been imposed"
    # Both residues are the same size and neither is small next to the other:
    # 4.09 here against QE_ACOUSTIC's 2.05, which is 1.3e-4 of the force
    # constants against 6.5e-5. Asserting one against the other would be
    # asserting a property of two truncated basis sets.
    assert phonons.acoustic_residue < 20 * QE_ACOUSTIC


def test_the_acoustic_sum_rule_can_be_imposed():
    """With ``asr`` on, the three acoustic frequencies are zero to round-off."""
    calculation, result, _ = _phonons("si-epsilon")
    phonons = dynamical_matrix(
        calculation, result.wavefunctions, result.eigenvalues, result.density,
        result.becsum, acoustic_sum_rule=True,
    )
    # Not smaller than 1e-4 cm^-1, and the reason is the square root: a
    # frequency is sqrt(eigenvalue), so a matrix entry left at round-off --
    # 1e-16 Ry/bohr^2, which is what the sum rule leaves -- surfaces as about
    # 1e-5 cm^-1. Measured: 6.5e-6.
    assert phonons.acoustic_residue < 1e-4
    assert phonons.frequencies[3] == pytest.approx(QE_OPTICAL, abs=OPTICAL_TOLERANCE)


def test_the_force_constants_are_symmetric_and_cubic():
    """Two measurements, neither of them a construction.

    The matrix is symmetric in ``(a i) <-> (b j)`` exactly, and
    :attr:`Phonons.asymmetry` is what is left after the group average -- which
    reports the linear solves rather than the wedge, since the average is
    applied first. And the on-site block of a diamond site is isotropic by its
    ``Td`` symmetry, which the group average makes *possible* rather than
    imposes: it projects onto what the crystal's operations allow, and a wrong
    rotation convention or a wrong atom permutation would survive it.
    """
    _, _, phonons = _phonons("si-epsilon")
    assert phonons.asymmetry < 1e-6
    for atom in range(2):
        block = phonons.matrix[atom, :, atom, :]
        diagonal = np.diag(block)
        assert np.allclose(diagonal, diagonal[0], atol=CUBIC_TOLERANCE)
        assert np.abs(block - np.diag(diagonal)).max() < CUBIC_TOLERANCE


# ---------------------------------------------------------------------------
# A metal (P28): the same machinery with the electronic half reweighted.
# ---------------------------------------------------------------------------

def test_the_gamma_phonon_of_a_metal_matches_quantum_espresso():
    """Two-atom aluminium against the vendored ``ph.x``.

    The Sternheimer solve was already metallic one phase before this
    (``PLAN.md`` P24c: ``orthogonalize``'s smearing branch, ``ef_shift``); what
    P28 adds is the **weight the assembled matrix contracts ``dpsi`` with**. The
    frozen energy weights its states by ``wg = wk f``, and a metal's ``dpsi``
    already carries ``f`` from the smeared right-hand side, so a single ``jvp``
    counted the occupation twice. QE never does: ``dynmat_us.f90`` reads ``wg``
    for the frozen Hessian and ``drhodvnl.f90`` reads ``2 wk`` for the
    electronic term, in two routines. Splitting the ``jvp`` accordingly is the
    whole of the change (:func:`pypresso.response.phonon._state_weights`).

    Three of the six modes are the primitive cell's at the zone-boundary point
    the doubling folds in; the other three are acoustic and are the diagnostic
    below rather than a target.
    """
    _, _, phonons = _phonons("al2-metal")
    assert phonons.converged
    optical = phonons.frequencies[3:]
    # The folded pair is degenerate and nothing here imposes it -- the cell is
    # a doubling and the degeneracy is the zone-boundary point's, so this is a
    # statement about the assembly and not about the input.
    assert optical[1] - optical[0] == pytest.approx(0.0, abs=1e-6)
    for computed, reference in zip(optical, QE_AL2[3:]):
        assert computed == pytest.approx(reference, abs=AL2_OPTICAL_TOLERANCE)


def test_a_metals_acoustic_sum_rule_is_the_basis_residue_and_not_a_weight_error():
    """``sum_b D_(a i)(b j) = 0``, and it is what the refusal was measured on.

    Translating the crystal costs nothing, so this sum is zero exactly and what
    survives is the finite basis. It is the sharpest diagnostic there is for the
    weight convention, because a factor of ``f`` on the electronic half leaves
    the *optical* modes looking plausible while the acoustic ones absorb the
    error: with the ``wg``-weighted assembly they came out at **155.7 cm^-1**
    against ``ph.x``'s 1.9, from a run that converged to
    ``|ddv_scf|^2 = 8.7e-17`` and returned a symmetric, cubic matrix. Nothing
    but this identity and the reference said so.

    The frequencies are not compared to ``ph.x``'s 1.1/1.8/1.9 digit for digit:
    both are the same basis-set error made slightly differently, exactly as on
    silicon, where this code prints 4.09 against ``ph.x``'s 2.05.
    """
    _, _, phonons = _phonons("al2-metal")
    residue = float(np.abs(phonons.matrix.sum(axis=2)).max())
    onsite = float(np.abs(phonons.matrix[0, :, 0, :]).max())
    assert residue < AL2_SUM_RULE, f"{residue} against on-site {onsite}"
    assert phonons.acoustic_residue < ACOUSTIC_CEILING


def test_a_metals_rigid_translation_reproduces_the_density_gradient():
    """The metal's response density, against a quantity built no other way.

    Same identity as the insulator's below and the same reason for it, run here
    because a metal's response density is the one carrying ``ef_shift``: a
    displacement at ``q = 0`` moves charge in and out of the cell, so the Fermi
    level moves with it and ``def ldos`` has to be in ``drho`` before it screens
    anything. If that correction were wrong or missing, the sum over atoms would
    not be a pure translation of the ground-state density.
    """
    calculation, result, phonons = _phonons("al2-metal")
    gvectors = calculation.basis.dense
    rho_g = r_to_g(jnp.asarray(result.density)[0], gvectors.fft_index)
    cartesian = gvectors.cartesian(calculation.system.cell)

    for axis in range(3):
        exact = -jnp.real(
            g_to_r(1j * cartesian[:, axis] * rho_g, gvectors.fft_index, gvectors.grid)
        )
        translated = phonons.induced_density[:, axis, 0].sum(axis=0)
        scale = float(jnp.abs(exact).max())
        assert float(jnp.abs(translated - exact).max()) / scale < 1e-3


# ---------------------------------------------------------------------------
# The rigid translation, which shares nothing with the assembly.
# ---------------------------------------------------------------------------

def test_a_rigid_translation_reproduces_the_density_gradient():
    """``sum_a drho_(a i) = -d(rho)/dr_i``, because translating is a symmetry.

    The identity needs the *screened* response: with the bare perturbation alone
    the two sides differ by 52%. It therefore tests the Sternheimer solve, the
    kernel and the symmetrisation at once, against a reference obtained by
    differentiating the converged density in G-space -- which shares no code
    with any of them.
    """
    calculation, result, phonons = _phonons("si-epsilon")
    gvectors = calculation.basis.dense
    rho = jnp.asarray(result.density)
    rho_g = r_to_g(rho[0], gvectors.fft_index)
    cartesian = gvectors.cartesian(calculation.system.cell)

    for axis in range(3):
        exact = -jnp.real(
            g_to_r(1j * cartesian[:, axis] * rho_g, gvectors.fft_index, gvectors.grid)
        )
        translated = phonons.induced_density[:, axis, 0].sum(axis=0)
        scale = float(jnp.abs(exact).max())
        assert float(jnp.abs(translated - exact).max()) / scale < 1e-3


# ---------------------------------------------------------------------------
# Finite-differenced forces: the only check that reaches the response term.
# ---------------------------------------------------------------------------

#: The step, in bohr. Two of them, because a single one cannot tell truncation
#: from noise -- the U-curve P24a measured for the Sternheimer solve is the same
#: shape here, with the SCF's own convergence putting the floor under it.
FD_STEPS = (1.0e-2, 3.0e-3)


@pytest.mark.parametrize("step", FD_STEPS)
def test_the_force_constants_reproduce_finite_differenced_forces(step):
    """Displace, re-converge, difference the forces.

    ``jacfwd`` of the force with respect to the positions alone would give the
    frozen Hessian back and check nothing: this is the reference that reaches
    the electronic response, because a re-converged SCF has let the
    wavefunctions relax. Whole columns are compared, not a trace.

    The k-sample is the unshifted grid with ``nosym``: it is closed under the
    point group, so it is a valid sample for the *displaced* structures too. The
    ten-point wedge of ``si-epsilon.in`` is not -- it is the irreducible set of
    the undisplaced crystal, and re-converging a displaced structure on it would
    compare against a different Brillouin-zone sum.
    """
    calculation, result, phonons = _phonons("si-epsilon-unshifted-nosym")
    system, pseudos, _ = _build("si-epsilon-unshifted-nosym")
    reference = np.asarray(system.structure.positions)

    def force_at(displacement):
        moved = eqx.tree_at(
            lambda s: s.structure.positions, system,
            jnp.asarray(reference + displacement),
        )
        calc = Calculation(moved, pseudos)
        converged = run_scf(moved, pseudos, calculation=calc, conv_thr=1e-12,
                            max_iterations=100)
        return np.asarray(compute_forces(calc, converged).forces)

    for atom, cart in [(0, 0), (0, 2)]:
        displacement = np.zeros_like(reference)
        displacement[atom, cart] = step
        difference = -(force_at(displacement) - force_at(-displacement)) / (2 * step)
        assert np.abs(difference - phonons.matrix[atom, cart]).max() < FD_TOLERANCE


# ---------------------------------------------------------------------------
# The wedge against the whole grid: the symmetrisation's only check.
# ---------------------------------------------------------------------------

def test_the_wedge_and_the_whole_grid_give_the_same_matrix():
    """The same k-sample, reduced and whole, with the symmetrisation on and off.

    QE computes the wedge route alone, so this is the only check the two new
    symmetrisations have -- ``symmetrize_atom_displacement`` inside the loop and
    ``symmetrize_atom_pair_tensor`` on the assembled matrix. Diamond silicon is
    the right cell for it because half its operations exchange the two
    sublattices, so the atom permutation is doing work rather than being the
    identity.

    It runs on the **unshifted** grid, which is closed under the point group. A
    shifted one is not (2304 of 3072 images land off it), which is what
    ``require_a_symmetrisable_response`` refuses and what P24 measured.
    """
    _, _, wedge = _phonons("si-epsilon-unshifted")
    _, _, whole = _phonons("si-epsilon-unshifted-nosym")
    assert np.abs(wedge.matrix - whole.matrix).max() < WEDGE_TOLERANCE
    assert np.abs(wedge.frequencies - whole.frequencies).max() < 1e-3


# ---------------------------------------------------------------------------
# What is refused, and by name.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case,expected", [
    ("si-epsilon-us", "ultrasoft or PAW"),
    ("si-epsilon-paw", "ultrasoft or PAW"),
])
def test_ultrasoft_and_paw_are_refused(case, expected):
    """Not a missing routine but a term missing from the formula.

    The identity the assembly rests on holds because the frozen energy is
    stationary in the states at *fixed* Lagrange multipliers, and those
    multipliers sit on the constraint ``<psi|S(u)|psi> - 1``. When ``S`` moves
    with the atoms the second derivative acquires a term in ``deps/du`` that is
    identically zero for a norm-conserving dataset; and the augmentation charge
    moves at frozen ``becsum`` besides. The measured cost of ignoring both is in
    ``PLAN.md`` P25 -- and, as with ``zstar_eu_us``, it is invisible in the
    answer, which comes out looking like an ordinary phonon spectrum.
    """
    system, pseudos, calculation = _build(case)
    with pytest.raises(NotImplementedError, match=expected):
        dynamical_matrix(calculation, None, jnp.zeros((1, 1, 1)), None)


def test_a_spin_polarized_calculation_is_refused():
    """``nspin = 2``, and the reason is a shared occupied-band count.

    ``nocc`` here is a single ``nelec / 2`` applied to both channels, which is
    right for an unpolarized insulator and wrong for a magnetic one -- and wrong
    invisibly, since the shapes still fit and the solve still converges. The same
    line is in ``dielectric_tensor`` and is not refused there; ``PLAN.md`` P25
    names that as a gap in P24 rather than a decision.

    The guard is exercised directly rather than through
    :func:`dynamical_matrix`, because every ``nspin = 2`` input committed here is
    a *metal*, and what fires first through the front door is
    ``require_a_sternheimer_regime``'s own ``nspin = 2`` refusal -- the same fact
    stated one layer down, where every response entry point inherits it. (Before
    P28 it was the metal refusal that fired first; that one is gone.) Reaching
    *this* message needs a magnetic insulator, which is the same thing the fix
    needs, and neither is here.
    """
    _, _, calculation = _build("h-atom-lsda")
    assert calculation.nspin == 2
    with pytest.raises(NotImplementedError, match="spin-polarized"):
        _require_one_spin_channel(calculation)
