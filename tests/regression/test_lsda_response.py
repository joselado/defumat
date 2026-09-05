"""The Sternheimer response of a spin-polarized system (``nspin = 2``).

``GAPS.md`` section 3 called this "the single widest guard": one refusal in
:func:`~defumat.response.sternheimer.require_a_sternheimer_regime` blocked
*every* response quantity for *every* spin-polarized system, and the reason it
gave was an occupied-band count. The count is per spin channel now
(:func:`~defumat.response.sternheimer.occupied_counts`), which is what QE gets
for free by doubling ``nks`` in LSDA -- there each channel is a separate
k-point and ``setup_nbnd_occ`` writes its own ``nbnd_occ(ik)``.

**The anchors here are QE-free**, in the order ``NEXT-SESSION.md`` asks for:

1. ``chi_0`` against a central difference of the density, for a spin-polarized
   *metal* (``h-chain-afm``) and for a spin-polarized *insulator*
   (``o2-fixed-lsda``). Both use a **spin-dependent probe potential** -- a
   different amplitude and sign in each channel -- because ``chi_0`` is
   block-diagonal in spin and a probe that is the same in both channels would
   not tell the two blocks apart. This is P24c's harness with a spin axis.
2. The identity that catches the one bug this change is most likely to have:
   the same cell run as ``nspin = 1`` and as ``nspin = 2`` with no
   magnetization must give the *same* dielectric constant. The spin sum is
   where a factor of two would hide, and the kernel (Hartree plus ``f_xc``) is
   where the channels couple, so the identity is run end to end on
   ``epsilon_infinity`` rather than on ``chi_0``.
3. What is still refused, by name, with the missing piece in the message.

**The insulator cell is not the obvious one, and it fails silently.**
``o-atom-fixed-lsda`` -- the oxygen *atom* -- cannot be used: at
``tot_magnetization = 2`` its minority channel holds two electrons, so its
filling cuts the triply degenerate 2p shell (the two levels are 1.4e-14 Ry
apart). What that does was **measured rather than assumed**, and the result is
why the refusal exists: the CG converges normally -- 42 iterations to a
residual of 5e-12 against a 1e-11 threshold -- and ``chi_0`` then disagrees
with a central difference of the density by **100 per cent** (1.24, 1.02, 1.01
and 1.01 relative, for probes at Miller (1,0,0), (0,0,1), (1,1,0) and (1,1,1)).
The difference re-selects which member of the shell falls below the cut,
because the perturbation splits it at first order; the solve keeps the member
the eigensolver handed it, which is an arbitrary one. Nothing inside the solve
can see that, so it is refused by name
(:data:`~defumat.response.sternheimer.DEGENERATE_CUT_RY`).

The oxygen *molecule* is the case that works: twelve electrons and
``tot_magnetization = 2`` give seven up and five down, and both are closed
shells with a real gap (0.438 Ry up, 0.517 Ry down, measured).
"""

from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.response.sternheimer import (
    make_sternheimer,
    occupied_counts,
    require_a_sternheimer_regime,
)
from defumat.scf import run_scf
from defumat.system import build_system

pytestmark = [pytest.mark.regression]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: Miller index of the probe potential, as ``test_response.py`` uses it.
PROBE_MILLER = (1, 0, 0)

#: How far ``chi_0 dV`` may sit from a central difference of the density, as a
#: fraction of the largest response. Measured in this session, and it is the
#: difference's own truncation rather than the solve's -- see the constants
#: below for the step each was measured at.
CHI0_RELATIVE = 1e-5

#: The step for the metallic hydrogen chain. Measured: the error falls as
#: ``h^2`` through 3e-4, so this is truncation.
METAL_CHI0_STEP = 3.0e-4

#: The step for the O2 insulator, and the buffer bands its reference
#: diagonalises and throws away -- Davidson converges its topmost requested root
#: worst, and without the buffer it is the *reference* that fails.
INSULATOR_CHI0_STEP = 1.0e-4
REFERENCE_BUFFER = 4

#: How far the ``nspin = 2`` dielectric constant of a cell with no magnetization
#: may sit from the ``nspin = 1`` one. Both SCFs are run to ``conv_thr = 1e-12``
#: and land on the same fixed point to about that; anything larger than
#: round-off here is a factor in the spin sum.
IDENTITY_TOLERANCE = 1e-8


def _probe_potential(calculation, amplitudes):
    """``a_s cos(2 pi G.r)`` on the dense grid, ``(nspin, n1, n2, n3)``.

    ``amplitudes`` is one number per spin channel. Giving the channels
    *different* amplitudes is the point: ``chi_0`` is block-diagonal in spin
    (the independent-particle response of one channel cannot see the other's
    potential), so a probe equal in both channels leaves the two blocks
    indistinguishable and a swapped or shared count would pass.
    """
    grid = calculation.basis.dense.grid
    axes = [np.arange(n) / n for n in grid]
    positions = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    field = np.cos(2.0 * np.pi * (positions @ np.asarray(PROBE_MILLER)))
    return jnp.asarray(np.stack([a * field for a in amplitudes]))


def _converged(name, **kwargs):
    from defumat.scf import Calculation

    system = build_system(read_pw_input(CASES / f"{name}.in"))
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation, **kwargs)
    assert result.converged
    return system, pseudos, calculation, result


@lru_cache(maxsize=None)
def _hydrogen_chain():
    """The antiferromagnetic hydrogen chain of ``h-chain-afm.in``.

    A **metal**: gaussian smearing, one Fermi level shared by the channels,
    ``nosym``. The smearing branch of the solve never takes the occupied slice
    at all -- every band stays in the block and the occupation rides inside
    ``dpsi`` -- so what this case tests is that the branch works with a spin
    axis, which is exactly the half of the old refusal that never bound it.
    """
    return _converged("h-chain-afm", conv_thr=1e-12, max_iterations=200)


@lru_cache(maxsize=None)
def _oxygen_molecule():
    """The triplet O2 of ``o2-fixed-lsda.in`` -- seven bands up, five down."""
    return _converged("o2-fixed-lsda", conv_thr=1e-12, max_iterations=200)


@pytest.mark.slow
def test_the_occupied_counts_are_per_channel():
    """``(7, 5)`` and not ``(6, 6)``, and the mask that comes with it.

    ``nelec / 2`` is 6 here, which is neither channel's filling. The whole
    refusal this file lifts was that one number; the assertion is that the
    solver now carries the pair, keeps ``max`` of them as the block depth (rule
    R2: the shape is static and the deficient channel is masked, not resized),
    and that the ground state's own weights agree -- the down channel's bands 5
    and 6 are in the array and carry no charge.
    """
    _, _, calculation, result = _oxygen_molecule()
    assert occupied_counts(calculation) == (7, 5)

    solver = make_sternheimer(calculation, result, spin_polarized=True)
    assert solver.occupied_counts == (7, 5)
    assert solver.nocc == 7
    mask = np.asarray(solver.projector_mask)
    assert mask.shape == (2, 1, 7)
    assert mask[0].all()
    assert mask[1, :, :5].all() and not mask[1, :, 5:].any()
    # The occupations the SCF built its density from say the same thing.
    assert np.allclose(np.asarray(solver.weights)[1, :, 5:], 0.0)
    assert np.asarray(solver.weights)[1, :, 4] > 0.0


@pytest.mark.slow
def test_chi0_matches_a_finite_difference_for_a_spin_polarized_metal():
    """P24c's measurement with a spin axis, on the AFM hydrogen chain.

    The reference re-occupies at the **same** Fermi level rather than
    re-converging it, for the reason the unpolarized version gives: the
    Sternheimer response of a metal is the response at fixed ``ef``, and the
    level's own motion is a separate correction (``ef_shift``). The level is
    shared between the channels here, which is the case this change ships --
    ``tot_magnetization`` with a smearing gives two levels and is refused below.

    The probe is **twice as strong in the up channel and opposite in sign in the
    down one**, so the two blocks are told apart.
    """
    from defumat.basis.interpolate import to_dense
    from defumat.scf.density import sum_band
    from defumat.scf.occupations import smearing_order, wgauss

    system, _, calculation, result = _hydrogen_chain()
    solver = make_sternheimer(
        calculation, result, metals=True, spin_polarized=True
    )
    assert solver.smearing is not None
    assert solver.nspin == 2
    dv = _probe_potential(calculation, (1.0, -0.5))

    solution = solver.solve(solver.perturbation(dv))
    assert solution.converged
    drho = np.asarray(solver.response_density(solution.dpsi))
    assert drho.shape[0] == 2
    # The response is genuinely different in the two channels, which is what
    # makes the comparison below a two-block test rather than one repeated.
    assert not np.allclose(drho[0], drho[1])

    smooth, dense = calculation.basis.smooth, calculation.basis.dense
    v_scf = calculation.potential(result.density).v_scf
    ngauss = smearing_order(system.smearing)
    kweights = jnp.asarray(system.kpoints.weights)
    nbnd = np.asarray(result.wavefunctions).shape[2]

    def density_at(scale):
        hamiltonians = calculation.hamiltonian(v_scf + scale * dv, None)
        eigenvalues, psi = calculation.diagonalize(hamiltonians, nbnd, None, 1e-13)
        occupation = wgauss(
            (result.fermi_energy - eigenvalues) / system.degauss, ngauss
        )
        rho = sum_band(
            psi, calculation.fft_index, smooth.grid,
            occupation * kweights[None, :, None], system.cell,
            calculation.k_batch,
        )
        return np.asarray(to_dense(rho, smooth, dense))

    step = METAL_CHI0_STEP
    reference = (density_at(step) - density_at(-step)) / (2.0 * step)
    relative = np.abs(drho - reference).max() / np.abs(drho).max()
    print(f"\nmetal nspin=2 chi_0 relative error: {relative:.3e}")
    assert relative < CHI0_RELATIVE


@pytest.mark.slow
def test_chi0_matches_a_finite_difference_for_a_spin_polarized_insulator():
    """The sliced branch, where the two channels are filled to different depths.

    This is the measurement the old refusal was a placeholder for. If the
    occupied count were shared -- six bands in each channel, ``nelec / 2`` --
    then ``P_c^+`` would project the up channel's response onto a manifold one
    band too small and the down channel's onto one a band too large, and the
    density response would be wrong with no shape error and no failed
    convergence. Here it is compared against a central difference of the density
    itself, which shares the Hamiltonian with the solve and nothing else.

    The dataset is **ultrasoft**, so the reference carries ``becsum`` and the
    augmentation charge, and the perturbation carries ``int3`` -- all of them
    with the spin axis they have had since P12 and none of them exercised in a
    response before.
    """
    from defumat.basis.interpolate import to_dense
    from defumat.scf.density import becsum as becsum_of, sum_band

    system, _, calculation, result = _oxygen_molecule()
    solver = make_sternheimer(calculation, result, spin_polarized=True)
    assert solver.smearing is None, "this cell must be the sliced branch"
    dv = _probe_potential(calculation, (1.0, -0.5))

    solution = solver.solve(solver.perturbation(dv))
    assert solution.converged
    drho = np.asarray(solver.response_density(solution.dpsi))
    assert not np.allclose(drho[0], drho[1])

    smooth, dense = calculation.basis.smooth, calculation.basis.dense
    v_scf = calculation.potential(result.density).v_scf
    _, ddd_paw = calculation.onecenter(result.becsum)
    nbnd = np.asarray(result.wavefunctions).shape[2]
    weights = solver.weights

    def density_at(scale):
        hamiltonians = calculation.hamiltonian(v_scf + scale * dv, ddd_paw)
        _, psi = calculation.diagonalize(
            hamiltonians, nbnd + REFERENCE_BUFFER, None, 1e-13
        )
        psi = psi[:, :, : solver.nocc]
        becsum_ = becsum_of(
            psi, calculation.projectors.vkb, weights,
            calculation.species_channels, calculation.k_batch,
        ) if calculation.is_ultrasoft else ()
        rho = sum_band(
            psi, calculation.fft_index, smooth.grid, weights,
            system.cell, calculation.k_batch,
        )
        return np.asarray(
            calculation.augmented(to_dense(rho, smooth, dense), becsum_)
        )

    step = INSULATOR_CHI0_STEP
    reference = (density_at(step) - density_at(-step)) / (2.0 * step)
    relative = np.abs(drho - reference).max() / np.abs(drho).max()
    print(f"\ninsulator nspin=2 chi_0 relative error: {relative:.3e}")
    assert relative < CHI0_RELATIVE


@lru_cache(maxsize=None)
def _silicon_dielectric(nspin: int):
    """``epsilon_infinity`` of ``si-epsilon.in`` at ``nspin = 1`` or ``2``.

    The ``nspin = 2`` run is the *same input file* with two lines added --
    ``nspin = 2`` and ``tot_magnetization = 0`` -- rather than a second
    committed input, so that the identity below cannot be weakened by the two
    sides drifting apart. ``occupations = 'fixed'`` with LSDA needs an integer
    ``tot_magnetization`` (``input.f90:784-800``), and zero is one; it fills
    four bands in each channel, which is ``nelec / 2`` twice.
    """
    from defumat.response.efield import dielectric_tensor
    from defumat.scf import Calculation

    parsed = read_pw_input(CASES / "si-epsilon.in")
    if nspin == 2:
        parsed.namelists["system"]["nspin"] = 2
        parsed.namelists["system"]["tot_magnetization"] = 0.0
        parsed.namelists["system"]["starting_magnetization"] = {(1,): 0.0}
    system = build_system(parsed)
    assert system.nspin == nspin
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-12,
                     max_iterations=80)
    assert result.converged
    response = dielectric_tensor(
        calculation, result.wavefunctions, result.eigenvalues, result.density,
        result.becsum, born_charges=False,
    )
    assert response.converged
    return result, response


@pytest.mark.slow
def test_the_polarized_dielectric_constant_reduces_to_the_unpolarized_one():
    """The identity that catches a factor of two in the spin sum.

    Silicon has no magnetization, so an ``nspin = 2`` run of it is the
    unpolarized one written twice -- two identical channels, each carrying half
    the k-point weight. Every stage of the response must therefore return the
    *same* number: the occupied count (four bands per channel, where ``nelec /
    2`` is also four), the spin sum in ``dielec.f90``'s assembly, and the
    screening kernel, which is where the two channels talk to each other and
    where a Hartree term counted once per channel would double.

    Run end to end on ``epsilon_infinity`` rather than on ``chi_0`` for exactly
    that last reason: ``chi_0`` is block-diagonal and would not see it.
    """
    result1, response1 = _silicon_dielectric(1)
    result2, response2 = _silicon_dielectric(2)

    # First the ground states, so that a disagreement below is the response's.
    assert float(result2.total_energy) == pytest.approx(
        float(result1.total_energy), abs=1e-9
    )
    difference = abs(float(response2.isotropic) - float(response1.isotropic))
    print(f"\nepsilon nspin=1 {response1.isotropic:.9f} "
          f"nspin=2 {response2.isotropic:.9f} diff {difference:.3e}")
    assert difference < IDENTITY_TOLERANCE
    assert np.allclose(
        np.asarray(response2.epsilon), np.asarray(response1.epsilon),
        atol=IDENTITY_TOLERANCE,
    )


@pytest.mark.slow
def test_a_magnetic_insulators_dielectric_constant_matches_ph_x():
    """``epsilon_infinity`` of a magnetic insulator, against the vendored ``ph.x``.

    **This case used to be a refusal and the refusal was reasoning about the
    wrong code.** ``dv_of_drho`` is one ``jvp`` of ``v_of_rho``, so for
    ``nspin = 2`` it is the second derivative of the LSDA exchange-correlation
    energy in the two channel densities -- and that is infinite wherever a
    channel density reaches zero, which a plane-wave magnetization does in
    vacuum. Measured on this cell: **1504 of 91125 grid points** have
    ``|m| >= |n|``, and ``dv_of_drho`` had exactly 1504 ``NaN``. What was
    concluded from that was that ``defumat.xc``'s spin branch had to be made
    twice differentiable at a fully polarized point.

    It does not: **QE does not compute a kernel there either.** ``dmxc_lsda``
    *defines* it to be zero, on both of its branches -- the analytic one
    pre-zeroes the whole ``dmuxc`` block and ``CYCLE``s past ``|zeta| >= 1``
    (``XClib/qe_drivers_d_lda_lsda.f90:234-237, :255, :257``), the numerical one
    clamps ``zeta`` to ``1 - 2e-6`` and sets ``rhotot = dr = 0``
    (``:341, :344-346``). So the missing piece was a convention, not an
    analysis.

    **Why the recorded attempts failed is worth keeping**: clipping the density
    to ``1 - eps`` at 1e-12, 1e-10 and 1e-8 left all 1504 points ``NaN``,
    because that masks the *value* and leaves the primal singular -- the tangent
    is still ``0 * inf``. Masking the **argument** the derivative is taken at is
    what works, and at a regular point the masked and unmasked expressions are
    bit-identical, so nothing already validated moves.

    ``ph.x`` computes this happily -- ``phq_readin.f90:546`` refuses an electric
    field only for *noncollinear* magnetism and ``:957`` only for a smeared or
    tetrahedron metal -- and ``reference.out.ph-o2-fixed-lsda`` is its output.
    """
    from defumat.response.efield import dielectric_tensor

    _, _, calculation, result = _oxygen_molecule()
    response = dielectric_tensor(
        calculation, result.wavefunctions, result.eigenvalues,
        result.density, result.becsum, born_charges=False, max_iterations=40,
    )
    assert response.converged
    epsilon = np.asarray(response.epsilon)
    # ``reference.out.ph-o2-fixed-lsda``, the "Dielectric constant in cartesian
    # axis" block. The molecule lies along z, so xx = yy and zz is the odd one.
    assert np.allclose(np.diag(epsilon), [1.110915996, 1.110915996, 1.198004867],
                       atol=2e-5)
    off = epsilon - np.diag(np.diag(epsilon))
    assert np.abs(off).max() < 1e-10


@pytest.mark.slow
def test_the_lsda_born_charges_match_ph_x():
    """``Z*`` for ``nspin = 2``, against ``reference.out.ph-o2-fixed-lsda``.

    **The bug this closes could not be seen by anything already validated, and
    that is the point of the case.** The solver keeps ``max(occupied_counts)``
    bands in every channel because the shape has to be static, so triplet O2's
    counts of ``(7, 5)`` leave the down channel carrying its two empty pi*_g
    bands. In ``_multiplier_response`` the weight was applied to the *column* of
    ``dLambda_mn`` and nothing to the *row*, so those two bands entered as rows
    with their full interband position matrix element -- ``P_c`` does not remove
    a state that is conduction in its own channel -- and were contracted against
    ``<psi_n|dS/du|pi*>``.

    Three things had to be true at once for it to survive: the term it feeds is
    ``dS/du``, which is **zero for a norm-conserving dataset**; the mask is all
    ones wherever the two channels are occupied to the **same depth**, which is
    every ``nspin = 1`` run and every LSDA run at ``tot_magnetization = 0``; and
    a **metal** never reaches here at all. QE cannot write the bug down --
    ``zstar_eu_us.f90:224-231`` runs both band loops over ``nbnd_occ(ikk)``, the
    spin-resolved k-point's own count.

    Worth 0.0918 on ``Z*_xx`` of an answer of 0.1337, and 0.0100 on ``zz`` of
    0.2004. The transverse component was out by a factor of three and ``zz`` by
    5 per cent, for a reason the numbers show: the genuine ``x`` occupied block
    holds only the augmentation dipole and a weakly screened ``dV_scf``
    (``eps_xx = 1.11``), so the leak dominated it, where along the molecular
    axis the genuine block is eight times larger.

    Compared against the **raw** block -- QE prints ``Z*`` both with and without
    the acoustic sum rule, and the ASR-applied one is ~0 by construction on a
    homonuclear molecule and would agree with anything.
    """
    from defumat.response.efield import dielectric_tensor

    _, _, calculation, result = _oxygen_molecule()
    response = dielectric_tensor(
        calculation, result.wavefunctions, result.eigenvalues,
        result.density, result.becsum, born_charges=True, max_iterations=40,
    )
    charges = np.asarray(response.born_charges)
    # "Effective charges (d Force / dE) in cartesian axis without acoustic sum
    # rule applied (asr)", reference.out.ph-o2-fixed-lsda:181-192.
    reference = np.array([
        [[0.13367, 0.0, 0.0], [0.0, 0.13367, 0.0], [0.0, 0.0, 0.20023]],
        [[0.13366, 0.0, 0.0], [0.0, 0.13366, 0.0], [0.0, 0.0, 0.20025]],
    ])
    assert np.abs(charges - reference).max() < 5e-4
    # The molecule is homonuclear, so the two atoms carry the same charge and
    # ph.x's own spread between them (1-2e-5) is the floor this can be held to.
    assert np.abs(charges[0] - charges[1]).max() < 5e-5


# -- what is still refused, and why -------------------------------------------


def _calculation(name, **overrides):
    from defumat.scf import Calculation

    parsed = read_pw_input(CASES / f"{name}.in")
    parsed.namelists["system"].update(overrides)
    system = build_system(parsed)
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    return Calculation(system, pseudos)


def test_a_filling_that_cuts_a_degenerate_multiplet_is_refused():
    """The oxygen *atom*: ``neldw = 2`` cuts the triply degenerate 2p shell.

    Not a missing term and not a tolerance. Which member of the multiplet the
    eigensolver returned is arbitrary, so ``P_c`` is built for an arbitrary
    subspace of a degenerate shell and the answer is a property of the
    eigensolver rather than of the density -- the same multivaluedness the
    *residual* solver is diagnosed for, one layer up.

    **And it converges**, which is the measurement that turned this from a
    guess into a refusal: with the check bypassed the CG finishes in 42
    iterations at a residual of 5e-12, and the ``chi_0`` it returns is 100 per
    cent away from a central difference of the density (1.24, 1.02, 1.01, 1.01
    for four different probes). A stalling solve would have announced itself;
    this one does not.

    The refusal is raised by the solver rather than by the guard, because it is
    a property of the converged spectrum and not of the input.
    """
    _, _, calculation, result = _converged(
        "o-atom-fixed-lsda", conv_thr=1e-10, max_iterations=200
    )
    assert occupied_counts(calculation) == (4, 2)
    with pytest.raises(NotImplementedError, match="degenerate multiplet"):
        make_sternheimer(calculation, result, spin_polarized=True)


def test_two_fermi_levels_are_refused_by_name():
    """``tot_magnetization`` with a *smearing*: the descoped case.

    Deliberately not shipped. ``Smearing`` carries a single scalar ``ef`` and
    ``smearing_of`` reads ``result.fermi_energy``, which for a constrained
    magnetization is the **mean** of the two levels -- a number QE prints only
    so that the field is not NaN. Every weight in ``orthogonalize``'s smearing
    branch would then be evaluated at a level neither channel has. Giving
    ``Smearing.ef`` a spin axis is the missing piece and it is named in the
    message.
    """
    calculation = _calculation("h-chain-afm", tot_magnetization=0.0)
    assert calculation.two_fermi_energies
    with pytest.raises(NotImplementedError, match="tot_magnetization"):
        require_a_sternheimer_regime(
            calculation, metals=True, spin_polarized=True
        )
    # And the same run without the constraint is not refused.
    shared = _calculation("h-chain-afm")
    assert not shared.two_fermi_energies
    require_a_sternheimer_regime(shared, metals=True, spin_polarized=True)


def test_an_unflagged_quantity_still_refuses_nspin_two():
    """The guard is opt-in, exactly as ``metals`` is.

    Lifting the refusal for everything at once would have silently opened the
    third derivatives of :mod:`defumat.response.electrostriction` and
    :mod:`defumat.response.nonlinear`, neither of which has ever been run with
    a spin axis and neither of which has a reference here. The flag keeps them
    refusing, and the message says what is now true: the *solve* is
    spin-polarized and the *assembly* is what is missing.
    """
    calculation = _calculation("h-chain-afm")
    with pytest.raises(NotImplementedError, match="not implemented for nspin = 2"):
        require_a_sternheimer_regime(calculation, metals=True)


def test_a_potential_only_functional_is_refused_by_name():
    """The free half of this pass (``GAPS.md`` section 3, part (a)).

    A meta-GGA whose *potential* is written down and whose energy does not exist
    reached the response stack and surfaced as ``v_of_rho`` asking for a ``tau``
    nobody had passed. ``reject_potential_only`` is the refusal the stress, the
    dynamical matrix and the elastic constants already make, reused rather than
    restated, and it is now made at the top of the Sternheimer guard so every
    response entry point inherits it.
    """
    calculation = _calculation("si-epsilon", input_dft="tb09")
    assert calculation.functional.is_meta
    with pytest.raises(NotImplementedError, match="meta-GGA|Tran|potential"):
        require_a_sternheimer_regime(calculation)


def test_the_dynamical_matrix_and_the_strain_response_still_refuse_nspin_two():
    """Both refusals stay, and both messages changed.

    Their old reason -- "the occupied-band count here is one number for both
    channels" -- is gone: they derive that pair from
    :func:`~defumat.response.sternheimer.occupied_counts` now. What is left is
    the second-derivative assembly above the solve, which is a term rather than
    a count, and the messages name it.
    """
    from defumat.response.phonon import _require_one_spin_channel as phonon_guard
    from defumat.response.strain import _require_one_spin_channel as strain_guard

    calculation = _calculation("h-chain-afm")
    with pytest.raises(NotImplementedError, match="second-derivative assembly"):
        phonon_guard(calculation)
    with pytest.raises(NotImplementedError, match="assembly above it"):
        strain_guard(calculation)
