"""The magnetocrystalline anisotropy by the force theorem (P58).

Four rungs, and only the last one reaches Fortran -- which is why the first
three are here at all. Each is a statement the assembly cannot satisfy by
accident:

1. **Switch the coupling off and every direction must agree.** Without
   spin-orbit coupling the Hamiltonian is invariant under a *global* spin
   rotation, so a band energy cannot depend on where the moment points -- not
   approximately, exactly. This is the check that found the phase's one real
   bug: the density was being rotated and the *quantization axis* QE's
   ``compute_ux`` builds was not, so the gradient correction differentiated
   ``|m|`` through its own nodes. It was worth **36.8 meV** on a cell whose
   answer is zero, it survived switching spin-orbit coupling off entirely
   (which is what identified it), and no spin-orbit test could have seen it.

2. **A cubic crystal has no anisotropy between its cubic axes.** Nothing in
   the code imposes that; it comes out of the k-sum.

3. **The first-order term is zero**, which is the reason this phase is a
   diagonalisation and not an expectation value.

4. **QE's own committed force-theorem example**, ``PP/examples/
   ForceTheorem_example`` -- a 3-layer Co(0001) slab, PRB 90, 205409 (2014),
   whose reference output carries ``eband`` to thirteen digits for two
   directions. That case is the phase's external anchor and the reason the
   pseudopotential pair ``Co.pbe-nd-rrkjus`` / ``Co.rel-pbe-nd-rrkjus`` is
   committed.
"""

from functools import lru_cache

import jax
import numpy as np
import pytest

from pypresso.calculator import Calculator
from pypresso.scf.continuation import (
    direction_from_angles,
    nc_magnetization_from_lsda,
)
from pypresso.units import RY_TO_EV
from pypresso.workflows.anisotropy import (
    angles_from_direction,
    frozen_expectation,
    run_torque,
    cardinal_directions,
    run_anisotropy,
    run_force_theorem,
    sphere_cover,
)
from tests.conftest import GENERATED

pytestmark = [pytest.mark.regression]

#: ``pw.x`` with ``lforcet``, from the committed reference outputs of
#: ``PP/examples/ForceTheorem_example`` (``eband, Ef (eV)`` in ``par.out`` and
#: ``per.out``). ``par`` is ``angle1 = 90``, in the slab plane; ``per`` is
#: ``angle1 = 0``, along the surface normal.
QE_EBAND = {"par": -75.5059287216436, "per": -75.5062821245659}
QE_FERMI = {"par": -0.454721315571854, "per": -0.457778043909281}


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """Every cell here compiles the whole SCF stack afresh and XLA keeps it.

    The rule of ``CLAUDE.md``'s memory section, applied because this file
    sweeps several cells: the results stay cached below, only the executables
    are dropped.
    """
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _smoke_pair():
    """A one-atom cubic Co cell, and the two datasets of the same generation.

    Cubic and one atom on purpose: every direction is then related to every
    other by a symmetry, so the answer to rungs 1 and 2 is *zero* and is known
    without computing anything.
    """
    scalar = Calculator.from_text(
        _SMOKE_SR, pseudo_dir=GENERATED.parent / "pseudo", conv_thr=1.0e-10
    )
    return scalar, scalar.get_scf()


@lru_cache(maxsize=2)
def _tetragonal():
    """The one-atom tetragonal cobalt cell: cheap, and *not* cubic.

    A cubic cell's anisotropy vanishes by symmetry, which is the right check
    for the assembly and useless for the torque -- a zero derivative of a zero
    curve says nothing. Stretching ``c/a`` to 1.30 gives a genuine uniaxial
    ``K1`` in one atom and eighteen k-points.
    """
    directory = GENERATED.parent / "pseudo"
    return (
        Calculator.from_file(GENERATED / "co-tetragonal-anisotropy-sr.in",
                             pseudo_dir=directory, announce=False),
        Calculator.from_file(GENERATED / "co-tetragonal-anisotropy-soc.in",
                             pseudo_dir=directory, announce=False),
    )


_SMOKE_SR = """
&control
   calculation='scf'
/
&system
   ibrav = 1, celldm(1) = 5.0, nat = 1, ntyp = 1,
   nspin = 2, ecutwfc = 20.0, ecutrho = 160.0,
   occupations = 'smearing', smearing = 'mv', degauss = 0.02,
   starting_magnetization(1) = 0.5,
/
&electrons
   conv_thr = 1.0e-10
/
ATOMIC_SPECIES
Co 58.933 Co.pbe-nd-rrkjus.UPF
ATOMIC_POSITIONS crystal
Co 0.0 0.0 0.0
K_POINTS automatic
2 2 2 0 0 0
"""

_SMOKE_SOC = _SMOKE_SR.replace(
    "calculation='scf'", "calculation='nscf'"
).replace(
    "   nspin = 2,",
    "   noncolin = .true., lspinorb = .true., lforcet = .true., nosym = .true.,",
).replace("Co.pbe-nd-rrkjus.UPF", "Co.rel-pbe-nd-rrkjus.UPF")

#: The same one-shot leg with the coupling *off*: a scalar-relativistic dataset
#: in a noncollinear run, which is legal and is the control for rung 1.
_SMOKE_NOSOC = _SMOKE_SOC.replace("lspinorb = .true., ", "").replace(
    "Co.rel-pbe-nd-rrkjus.UPF", "Co.pbe-nd-rrkjus.UPF"
)


# ----------------------------------------------------------------------
# the rotation itself
# ----------------------------------------------------------------------

def test_the_rotation_keeps_the_charge_and_the_moment():
    """``nc_magnetization_from_lsda`` moves the moment and nothing else."""
    rng = np.random.default_rng(0)
    density = rng.random((2, 6, 6, 6)) + 0.5
    charge = density[0] + density[1]
    moment = np.abs(density[0] - density[1])

    for angle1, angle2 in [(0, 0), (90, 0), (90, 90), (45, 30), (137, 201)]:
        direction = direction_from_angles(angle1, angle2)
        rotated = np.asarray(nc_magnetization_from_lsda(density, direction))
        assert rotated.shape == (4, 6, 6, 6)
        np.testing.assert_allclose(rotated[0], charge, atol=1e-14)
        np.testing.assert_allclose(
            np.linalg.norm(rotated[1:4], axis=0), moment, atol=1e-14
        )


def test_the_angles_and_the_direction_are_inverses():
    for direction in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1), (0, -1, 0),
                      (-1, 2, -3)]:
        wanted = np.asarray(direction, dtype=float)
        wanted = wanted / np.linalg.norm(wanted)
        angle1, angle2 = angles_from_direction(wanted)
        np.testing.assert_allclose(
            direction_from_angles(angle1, angle2), wanted, atol=1e-14
        )


def test_sphere_cover_is_unit_vectors_spread_over_the_sphere():
    """Elk's ``sphcover``: unit length, and an even spread in ``z``."""
    for n in (1, 2, 7, 40):
        points = np.asarray(sphere_cover(n))
        assert points.shape == (n, 3)
        np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0, atol=1e-14)
    # ``dz = 2/n`` between consecutive points is the formula's own statement.
    z = np.asarray(sphere_cover(20))[:, 2]
    np.testing.assert_allclose(np.diff(z), -2.0 / 20, atol=1e-12)


def test_cardinal_directions_are_reduced_by_the_crystal_group():
    """``gentpmae``'s ``npmae < 0``: a cube keeps far fewer than its 26 rays."""
    scalar, _ = _smoke_pair()
    directions = cardinal_directions(scalar.system, 1)
    points = np.asarray(directions)
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0, atol=1e-12)
    # 26 non-zero integer rays with |n_i| <= 1; a cubic group leaves the three
    # inequivalent families <100>, <110>, <111>.
    assert 1 <= len(directions) <= 26
    assert len(directions) < 26


# ----------------------------------------------------------------------
# rung 1: without the coupling there is no anisotropy at all
# ----------------------------------------------------------------------

@pytest.mark.slow
def test_without_spin_orbit_the_band_energy_does_not_know_the_direction():
    """The identity that caught the quantization-axis bug.

    The Hamiltonian without spin-orbit coupling commutes with a global spin
    rotation, so this is exact rather than approximate -- and it is checked
    with the *coupling off* precisely so that a failure cannot be blamed on the
    spin-orbit term.
    """
    scalar, scf = _smoke_pair()
    nosoc = Calculator.from_text(_SMOKE_NOSOC, pseudo_dir=GENERATED.parent / "pseudo")
    assert not nosoc.system.lspinorb and nosoc.system.noncolin

    energies = [
        run_force_theorem(
            nosoc.system, nosoc.pseudos, scf.density, direction=direction,
            require_spin_orbit=False,
        ).band_energy
        for direction in [(0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 1, 1), (0, 1, 1)]
    ]
    spread = (max(energies) - min(energies)) * RY_TO_EV
    assert spread == pytest.approx(0.0, abs=1.0e-10), (
        f"the band energy moved by {spread:.3e} eV under a rotation the "
        "Hamiltonian is invariant under"
    )


@pytest.mark.slow
def test_soc_scale_zero_gives_exactly_no_anisotropy():
    """The same identity as above, on **one** dataset instead of a matched pair.

    ``soc_scale = 0`` switches the coupling off inside the fully-relativistic
    file, so this is the control the two-file route cannot run: same
    projectors, same overlap, same everything but the coupling. It is the
    check that caught the phase's second bug -- ``newd_so``'s ``fcoef``
    sandwich left unscaled, which is invisible in a bulk cell and worth
    -6.7 meV on a slab, because ``dvan_so`` carries the coupling too and
    switching only *it* off still looks like it worked.
    """
    scalar, scf = _smoke_pair()
    soc = Calculator.from_text(_SMOKE_SOC, pseudo_dir=GENERATED.parent / "pseudo")
    result = run_anisotropy(
        soc.system, soc.pseudos, scf.density,
        directions=[(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)], soc_scale=0.0,
    )
    # 1e-7 meV is 1e-10 eV: the identity is exact and what is left is the
    # eigensolver's own residual over four directions.
    assert result.anisotropy_mev == pytest.approx(0.0, abs=1.0e-7)


def test_an_intermediate_soc_scale_is_refused():
    scalar, _ = _smoke_pair()
    soc = Calculator.from_text(_SMOKE_SOC, pseudo_dir=GENERATED.parent / "pseudo")
    with pytest.raises(ValueError, match="only 0 and 1"):
        soc.system.with_soc_scale(0.5)


@pytest.mark.slow
def test_the_first_order_term_is_zero_and_the_diagonalisation_is_not():
    """Why this is a diagonalisation and not an expectation value.

    Freezing the wavefunctions as well as the density and taking the
    spin-orbit term's expectation value once -- the calculation the force
    theorem is often assumed to be -- gives **no anisotropy at all**, because
    the coupling enters at first order as ``xi <L> . n`` and the orbital
    moment of a scalar-relativistic collinear state is quenched (P48 measured
    it at 1.7e-16). The anisotropy is second order, and what supplies it is
    the repulsion between levels that a diagonalisation performs.

    Both numbers come from the same cell and the same density, so the ratio
    between them is the statement: 1.9e-6 meV of first-order spread against
    0.597 meV from the force theorem, a factor of 3e5.
    """
    scalar, scf = _smoke_pair()
    soc = Calculator.from_text(_SMOKE_SOC, pseudo_dir=GENERATED.parent / "pseudo")
    first = [
        frozen_expectation(soc.system, soc.pseudos, scf.density, direction=d)
        for d in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    ]
    spread = (max(first) - min(first)) * RY_TO_EV * 1000
    assert spread == pytest.approx(0.0, abs=1.0e-4), (
        f"the first-order term acquired a direction dependence of {spread:.3e} meV"
    )
    # ... and it is the *term itself* that vanishes, not only its anisotropy,
    # which is the quenched orbital moment showing through.
    assert max(abs(e) for e in first) * RY_TO_EV * 1000 < 1.0e-3


@pytest.mark.slow
def test_a_cubic_crystal_has_no_anisotropy_between_its_axes():
    """Rung 2, and nothing in the assembly imposes it."""
    scalar, scf = _smoke_pair()
    soc = Calculator.from_text(_SMOKE_SOC, pseudo_dir=GENERATED.parent / "pseudo")
    result = run_anisotropy(soc.system, soc.pseudos, scf.density, directions="xyz")
    assert result.anisotropy_mev == pytest.approx(0.0, abs=1.0e-6)


# ----------------------------------------------------------------------
# rung 4: QE's own force-theorem example
# ----------------------------------------------------------------------

@pytest.mark.slow
def test_co_slab_reproduces_pw_x_on_the_force_theorem():
    """``PP/examples/ForceTheorem_example``: a 3-layer Co(0001) slab.

    Two ``pw.x`` runs with ``lforcet``, whose reference outputs carry ``eband``
    to thirteen digits, and whose difference -- the MAE -- is 0.3534 meV for
    the three-atom cell. QE's own README warns that its k-grid is too coarse
    for the *physics*, which does not matter here: what is being reproduced is
    the number ``pw.x`` prints on that grid.

    **About 25 minutes**, nearly all of it the SCF, and it does not reach
    ``conv_thr``. QE converges this slab in 24 iterations with
    ``mixing_mode = 'local-TF'`` (``approx_screening2``), which is not
    implemented here and which exists for exactly this shape of system: plain
    Anderson at ``beta = 0.7`` *diverges* on it (to +335 Ry), and Kerker at
    ``beta = 0.3`` converges linearly and is still an order short of 1e-10 at
    250 iterations. So the assertions below are about what survives that.

    **What is asserted is the pair, not the two numbers.** Each leg's ``eband``
    sits about 0.25 eV above QE's, because the density is the one the mixer
    reached rather than the one QE reached, and a density error shifts every
    eigenvalue. The two statements that are the actual finding are that the
    *same* shift applies to both legs -- which is what a frozen-density theorem
    promises -- and that the MAE, where it cancels, is QE's.
    """
    scalar = Calculator.from_file(
        GENERATED / "co-slab-forcetheorem-sr.in",
        pseudo_dir=GENERATED.parent / "pseudo",
        mixing_mode="kerker", mixing_beta=0.3, max_iterations=250,
    )
    scf = scalar.get_scf()
    # Not ``scf.converged``: see the docstring. What *is* asserted is that it
    # found QE's magnetic solution, which is the thing a different mixer could
    # plausibly have got wrong -- ``pw.x`` prints 5.26 and 5.84 Bohr magnetons.
    assert scf.accuracy < 1.0e-5
    moment = float(np.sum(np.asarray(scf.density)[0] - np.asarray(scf.density)[1])
                   * scalar.system.cell.volume / np.asarray(scf.density)[0].size)
    assert moment == pytest.approx(5.26, abs=0.02)

    energies = {}
    for name in ("par", "per"):
        leg = Calculator.from_file(
            GENERATED / f"co-slab-forcetheorem-{name}.in",
            pseudo_dir=GENERATED.parent / "pseudo",
        )
        assert leg.system.lforcet and leg.system.lspinorb and leg.system.nosym
        result = run_force_theorem(leg.system, leg.pseudos, scf.density)
        energies[name] = result.band_energy * RY_TO_EV

    # The two legs are offset from QE by the same amount, which is the
    # frozen-density theorem's own promise and what makes the difference mean
    # anything. Measured identical to four significant figures (+0.2481 eV).
    offsets = {name: energies[name] - QE_EBAND[name] for name in energies}
    assert offsets["par"] == pytest.approx(offsets["per"], abs=1.0e-3), (
        f"the two legs are shifted differently: {offsets}"
    )

    mae = energies["par"] - energies["per"]
    reference = QE_EBAND["par"] - QE_EBAND["per"]
    assert mae * 1000 == pytest.approx(reference * 1000, abs=0.02), (
        f"MAE {mae * 1000:.4f} meV against pw.x's {reference * 1000:.4f} meV"
    )


# ----------------------------------------------------------------------
# the refusals
# ----------------------------------------------------------------------

@pytest.mark.slow
def test_a_direction_other_than_the_system_s_own_needs_nosym():
    """Two directions on two different magnetic wedges is the silent failure."""
    scalar, scf = _smoke_pair()
    text = _SMOKE_SOC.replace("nosym = .true., ", "")
    soc = Calculator.from_text(text, pseudo_dir=GENERATED.parent / "pseudo")
    with pytest.raises(ValueError, match="nosym"):
        run_force_theorem(soc.system, soc.pseudos, scf.density, direction=(1, 0, 0))


def test_a_scalar_relativistic_one_shot_leg_is_refused_by_default():
    """It is legal only as the control above, so it has to be asked for."""
    scalar, scf = _smoke_pair()
    nosoc = Calculator.from_text(_SMOKE_NOSOC, pseudo_dir=GENERATED.parent / "pseudo")
    with pytest.raises(ValueError, match="lspinorb"):
        run_force_theorem(nosoc.system, nosoc.pseudos, scf.density)


def test_a_collinear_one_shot_leg_is_refused():
    scalar, _ = _smoke_pair()
    with pytest.raises(ValueError, match="noncollinear"):
        run_force_theorem(scalar.system, scalar.pseudos,
                          np.zeros((2, 4, 4, 4)), direction=(0, 0, 1))


def test_a_paw_dataset_is_refused_with_qe_s_own_reason():
    """``potinit.f90:98``, and the refusal has to fire *before* the NSCF's.

    ``fixed_density_states`` refuses PAW too, and its advice -- pass
    ``becsum = scf_result.becsum`` -- cannot be followed here: that ``becsum``
    belongs to a run with a different pseudopotential file and a different
    number of projectors. So the message has to come from this workflow, which
    means the check has to run before any array is built.
    """
    scalar, _ = _smoke_pair()
    text = _SMOKE_SOC.replace(
        "Co.rel-pbe-nd-rrkjus.UPF", "Pt.rel-pbe-n-kjpaw_psl.0.1.UPF"
    ).replace("Co 58.933", "Pt 195.08").replace("Co 0.0 0.0 0.0", "Pt 0.0 0.0 0.0")
    paw = Calculator.from_text(text, pseudo_dir=GENERATED.parent / "pseudo")
    with pytest.raises(NotImplementedError, match="PAW"):
        run_force_theorem(paw.system, paw.pseudos, np.zeros((2, 24, 24, 24)))


@pytest.mark.slow
def test_the_decomposition_recovers_the_band_energy_up_to_the_spilling():
    """``force_theorem``'s two printed totals, and the gap between them.

    ``projwfc.f90:636`` writes ``eband_tot`` and ``eband_proj_tot`` on one
    line, which is this decomposition's own diagnostic: the atomic-orbital
    basis does not span the occupied manifold, and what it misses is the
    spilling. The electron count is asserted beside it because a projection
    that has lost a ``degspin`` factor still looks entirely reasonable
    (``PLAN.md`` P51).
    """
    scalar, scf = _smoke_pair()
    soc = Calculator.from_text(_SMOKE_SOC, pseudo_dir=GENERATED.parent / "pseudo")
    result = run_force_theorem(
        soc.system, soc.pseudos, scf.density, direction=(0, 0, 1), projected=True
    )
    # A spinor band holds one electron; the weights must still sum to nelec.
    assert result.occupations.sum() == pytest.approx(9.0, abs=1e-9)

    shifted = float(
        np.sum(result.occupations * (result.eigenvalues - result.projected.ef_0))
    )
    assert result.projected.total == pytest.approx(shifted, rel=2.0e-3)
    # Every orbital belongs to the one atom, so the per-atom sum is the total.
    assert result.projected.by_atom.sum() == pytest.approx(
        result.projected.total, abs=1e-12
    )
    # With the moment along ``z`` and the coupling on, a cubic site keeps
    # ``p_x`` and ``p_y`` degenerate and need **not** keep ``p_z`` with them --
    # which is the anisotropy itself, seen orbital by orbital. QE's real
    # harmonics for ``l = 1`` are ordered ``(z, x, y)``
    # (:data:`pypresso.projwfc.channels.M_LABELS`), so it is the last two that
    # must agree.
    p_up = {
        channel.m: value for (channel, spin), value in
        zip(result.projected.labels, result.projected.by_orbital)
        if channel.l == 1 and spin == "up" and channel.wfc == 2
    }
    assert set(p_up) == {0, 1, 2}
    assert p_up[1] == pytest.approx(p_up[2], rel=1e-8)


# ----------------------------------------------------------------------
# the torque: the anisotropy as a derivative rather than a difference
# ----------------------------------------------------------------------

@pytest.mark.slow
def test_the_torque_reproduces_the_free_energy_difference():
    """One angle against two, and they share almost no machinery.

    ``run_anisotropy`` takes the anisotropy as ``E(n_1) - E(n_2)``: two
    independent diagonalisations differenced, with seven digits of
    cancellation. ``run_torque`` takes it as ``-dF/dtheta`` at a single angle,
    where nothing cancels. For ``E = K1 sin^2(theta)`` the torque at 45 degrees
    *is* ``-K1``, so the two must give the same constant -- and on tetragonal
    cobalt they agree to **2.4e-5 meV**.

    **The comparison is against the FREE energy and that is the whole point of
    this test.** A Hellmann-Feynman derivative at frozen occupations is the
    derivative of ``F = sum w eps - TS``, not of ``sum w eps``: the band energy
    carries an extra ``sum (dw/dtheta) eps`` that the entropy cancels. On this
    cell at ``degauss = 0.02`` Ry that term is **55 per cent** of the answer, so
    comparing against ``anisotropy_mev`` instead would look like a factor-of-two
    bug in the gradient and is not one.
    """
    scalar, spinor = _tetragonal()
    scf = scalar.get_scf()
    energies = run_anisotropy(spinor.system, spinor.pseudos, scf.density,
                              directions="xz")
    torque = run_torque(spinor.system, spinor.pseudos, scf.density)

    assert torque.anisotropy_constant_mev == pytest.approx(
        energies.free_anisotropy_mev, abs=2.0e-3
    )
    # ... and it is emphatically not the band-energy difference here.
    assert abs(energies.anisotropy_mev - energies.free_anisotropy_mev) > 0.5


@pytest.mark.slow
def test_the_torque_is_the_gradient_of_the_energy_it_claims_to_be():
    """Two checks that need no second method at all.

    ``sum w <psi|H|psi>`` must reproduce ``sum w eps`` at the angle the states
    came from -- one line that catches a wrong contraction, a lost weight or a
    mis-shaped spinor -- and the analytic gradient must reproduce a central
    difference of that same functional.
    """
    import jax.numpy as jnp
    from pypresso.forces.torque import band_energy_at_angle, torque_at_angle

    scalar, spinor = _tetragonal()
    scf = scalar.get_scf()
    result = run_torque(spinor.system, spinor.pseudos, scf.density)
    assert result.residual * RY_TO_EV * 1000 < 1.0e-6, (
        "sum w <psi|H|psi> does not reproduce sum w eps"
    )

    # The gradient against a central difference of its own functional.
    from pypresso.scf.continuation import nc_magnetization_from_lsda
    from pypresso.workflows.anisotropy import _with_quantization_axis
    from pypresso.workflows.nscf import fixed_density_states

    angle, plane = result.angle, result.plane
    direction = (np.cos(angle) * np.asarray(plane[0])
                 + np.sin(angle) * np.asarray(plane[1]))
    system = _with_quantization_axis(spinor.system, tuple(direction))
    rotated = nc_magnetization_from_lsda(scf.density, tuple(direction))
    calculation, system, eigenvalues, states = fixed_density_states(
        system, spinor.pseudos, rotated, conv_thr=1.0e-10)
    weights, _ = calculation.occupations(jnp.asarray(eigenvalues))

    step = 1.0e-3
    def energy(value):
        return float(band_energy_at_angle(
            calculation, states, weights, scf.density, plane, value))
    difference = (energy(angle + step) - energy(angle - step)) / (2 * step)
    analytic = -torque_at_angle(
        calculation, states, weights, scf.density, plane, angle)
    assert analytic == pytest.approx(difference, rel=1.0e-5)


def test_the_rotation_plane_must_be_orthogonal():
    scalar, spinor = _tetragonal()
    with pytest.raises(ValueError, match="orthogonal"):
        run_torque(spinor.system, spinor.pseudos, np.zeros((2, 4, 4, 4)),
                   plane=((0, 0, 1), (0, 0.3, 1)))
