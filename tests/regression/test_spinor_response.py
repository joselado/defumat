"""The Sternheimer response of a noncollinear system (``noncolin = .true.``).

One refusal in :func:`~defumat.response.sternheimer.require_a_sternheimer_regime`
blocked *every* response quantity for *every* spinor calculation, and the reason
it gave was that ``incdrhoscf_nc`` and ``set_int3_nc`` are "a second
implementation rather than a spin axis on this one". Taken at face value that
overstated it. The solve never had a spin axis to lose: a spinor is **one**
vector of length ``2 npwx`` and
:class:`~defumat.hamiltonian.noncollinear.SpinorHamiltonian` applies to it, so
the operator, the projector, the preconditioner and the CG were already the
right shape -- ``sternheimer.py`` even read ``degeneracy = 1 if noncolin`` from
behind the blanket refusal.

Two things were genuinely missing, and both are on the *outside* of the solve:

* the **density** it feeds back, which was the collinear ``sum_band`` and is now
  ``spinor_sum_band``, so an induced magnetization exists at all;
* the **perturbation**, which is a 2x2 matrix ``dv_0 I + dm . sigma`` at each
  point of the grid rather than one potential per channel.

``set_int3_nc`` is real and stays refused: for an ultrasoft or PAW dataset
``dD_ij`` is a 2x2 matrix in spin space, where a norm-conserving dataset has no
``dD`` at all.

**The anchors here are QE-free.**

1. ``chi_0`` against a central difference of the density on the 90-degree
   hydrogen cycloid, with **four** probes. A charge-only probe is not enough:
   the cross terms -- a charge perturbation inducing a magnetization, a Zeeman
   one inducing a charge -- are exactly what the spinor density's off-diagonal
   ``conj(up) down`` carries, and a probe in the charge channel alone would pass
   with that term wrong.
2. The identity that catches a factor of two in the spin sum: the same cell run
   as ``nspin = 1`` and as a **spinor with no magnetization** must give the same
   ``chi_0``. A spinor band holds one electron where a scalar band holds two,
   and that bookkeeping reaches the k-point weights, the occupied-band count and
   the density.
3. A wedge symmetrised against the whole zone, which is what tests the one piece
   of new physics here: the three magnetization channels of an induced density
   are an **axial** vector and the perturbation direction is a **polar** one, so
   the average carries two different rotations
   (:func:`~defumat.system.symmetry.symmetrize_spin_vector_density`).
4. What is still refused, by name.
"""

from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.response.sternheimer import make_sternheimer
from defumat.scf import run_scf
from defumat.system import build_system

pytestmark = [pytest.mark.regression]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: How far ``chi_0 dV`` may sit from a central difference of the density, as a
#: fraction of the largest response. The same bound the collinear cases take
#: (``test_response.py``, ``test_lsda_response.py``), and it is the difference's
#: own truncation rather than the solve's: measured 7.4e-6, 7.7e-6, 1.2e-6 and
#: 2.1e-6 for the four probes below at :data:`CHI0_STEP`.
CHI0_RELATIVE = 1e-5

#: The step for the metallic hydrogen cycloid, as ``test_lsda_response.py``
#: measured it for the collinear chain: the error falls as ``h^2`` through 3e-4,
#: so this is truncation.
CHI0_STEP = 3.0e-4

#: Miller index of the probe potential, as ``test_response.py`` uses it.
PROBE_MILLER = (1, 0, 0)


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


@lru_cache(maxsize=2)
def _cycloid():
    """``h-chain-90deg``: four hydrogens, each moment 90 degrees from the last.

    A **metal** (gaussian smearing) with ``nosym``, four species so that each
    atom carries its own angle, and ``nspin_mag = 4``. The texture is planar,
    which is what makes the cross terms below readable: the moments lie in the
    ``xy`` plane, so a charge perturbation induces magnetization *in that plane*
    and none along ``z``.

    ``maxsize = 2`` and never ``None``: the converged state holds the
    wavefunctions, and an unbounded cache of them is the accumulation that has
    killed test files on this machine.
    """
    return _converged("h-chain-90deg", conv_thr=1e-12, max_iterations=200)


def _probe(calculation, amplitudes):
    """``a_s cos(2 pi G.r)`` on the dense grid, one amplitude per component.

    The components are ``(n, m_x, m_y, m_z)``, so an amplitude tuple selects
    which block of ``chi_0`` the probe reaches.
    """
    grid = calculation.basis.dense.grid
    axes = [np.arange(n) / n for n in grid]
    positions = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    field = np.cos(2.0 * np.pi * (positions @ np.asarray(PROBE_MILLER)))
    return jnp.asarray(np.stack([a * field for a in amplitudes]))


def _finite_difference(system, calculation, result, dv, step):
    """``(rho(+h) - rho(-h)) / 2h`` at a frozen Fermi level.

    The reference re-occupies at the **same** Fermi level rather than
    re-converging it, for the reason the unpolarized and collinear versions
    give: the Sternheimer response of a metal is the response at fixed ``ef``
    and the level's own motion is a separate correction (``ef_shift``).
    """
    from defumat.basis.interpolate import to_dense
    from defumat.scf.density import spinor_sum_band
    from defumat.scf.occupations import smearing_order, wgauss

    smooth, dense = calculation.basis.smooth, calculation.basis.dense
    v_scf = calculation.potential(result.density).v_scf
    ngauss = smearing_order(system.smearing)
    kweights = jnp.asarray(system.kpoints.weights)
    nbnd = np.asarray(result.wavefunctions).shape[-2]

    def density_at(scale):
        hamiltonians = calculation.hamiltonian(v_scf + scale * dv, None)
        eigenvalues, psi = calculation.diagonalize(hamiltonians, nbnd, None, 1e-13)
        occupation = wgauss(
            (result.fermi_energy - eigenvalues) / system.degauss, ngauss
        )
        weights = occupation * kweights[None, :, None]
        rho = spinor_sum_band(
            psi[0], calculation.state_fft_index, smooth.grid, weights[0],
            system.cell, calculation.nspin_mag, calculation.k_batch,
        )
        return np.asarray(to_dense(rho, smooth, dense))

    return (density_at(step) - density_at(-step)) / (2.0 * step)


@pytest.mark.slow
def test_the_spinor_density_weights_are_the_ground_state_s():
    """The reference below rebuilds the SCF's own density, to round-off.

    Not decoration: **every** ``KPoints`` constructor applies the unpolarized
    ``degspin`` and a spinor band holds one electron, so a weight convention
    that is out by two is the most likely single error in this whole file and it
    would show up as a factor of two in ``chi_0`` that no shape check and no
    convergence test can see. Rebuilding the converged density from the same
    weights the finite difference uses pins it before anything is differentiated.

    Measured: **5.3e-23** against a density whose largest value is 0.143.
    """
    from defumat.basis.interpolate import to_dense
    from defumat.scf.density import spinor_sum_band
    from defumat.scf.occupations import smearing_order, wgauss

    system, _, calculation, result = _cycloid()
    smooth, dense = calculation.basis.smooth, calculation.basis.dense
    eigenvalues = jnp.asarray(result.eigenvalues)
    if eigenvalues.ndim == 2:
        eigenvalues = eigenvalues[None]
    psi = jnp.asarray(result.wavefunctions)
    if psi.ndim == 3:
        psi = psi[None]
    occupation = wgauss(
        (result.fermi_energy - eigenvalues) / system.degauss,
        smearing_order(system.smearing),
    )
    weights = occupation * jnp.asarray(system.kpoints.weights)[None, :, None]
    rho = spinor_sum_band(
        psi[0], calculation.state_fft_index, smooth.grid, weights[0],
        system.cell, calculation.nspin_mag, calculation.k_batch,
    )
    rebuilt = np.asarray(to_dense(rho, smooth, dense))
    reference = np.asarray(result.density)
    assert np.abs(rebuilt - reference).max() < 1e-18


@pytest.mark.slow
@pytest.mark.parametrize(
    "label, amplitudes",
    [
        ("all four channels", (1.0, -0.5, 0.3, 0.7)),
        ("charge only", (1.0, 0.0, 0.0, 0.0)),
        ("Zeeman m_z only", (0.0, 0.0, 0.0, 1.0)),
        ("Zeeman m_x only", (0.0, 1.0, 0.0, 0.0)),
    ],
)
def test_chi0_matches_a_finite_difference_for_a_spinor(label, amplitudes):
    """P24c's measurement with a spin *matrix* in place of a spin axis.

    Measured: **7.4e-6**, **7.7e-6**, **1.2e-6** and **2.1e-6** relative for the
    four probes, against a bound of 1e-5 that is the difference's own truncation.
    """
    system, _, calculation, result = _cycloid()
    assert calculation.noncolin and calculation.nspin_mag == 4
    solver = make_sternheimer(
        calculation, result, metals=True, noncollinear=True
    )
    assert solver.nspin == 1, "a spinor is one Hamiltonian on a doubled space"

    dv = _probe(calculation, amplitudes)
    solution = solver.solve(solver.perturbation(dv))
    assert solution.converged
    drho = np.asarray(solver.response_density(solution.dpsi))
    assert drho.shape[0] == 4

    reference = _finite_difference(system, calculation, result, dv, CHI0_STEP)
    relative = np.abs(drho - reference).max() / np.abs(drho).max()
    print(f"\n{label}: chi_0 relative error {relative:.3e}")
    assert relative < CHI0_RELATIVE


@pytest.mark.slow
def test_a_charge_probe_induces_a_magnetization_in_the_plane_of_the_texture():
    """The cross term, and the physics that says which components it reaches.

    ``chi_0`` is *not* block-diagonal in spin once the ground state carries a
    texture: perturbing the charge alone moves the magnetization, because the
    two spinor components share the states being mixed. Which components move is
    fixed by the texture -- this one is a **planar** cycloid, all four moments in
    the ``xy`` plane -- so the induced ``m_x`` and ``m_y`` are large and ``m_z``
    is zero to the level of the solve.

    Measured: ``m_x`` and ``m_y`` at **0.175** each and ``m_z`` at **7.3e-6**,
    against an induced charge of 0.732. A collinear ``sum_band`` behind the
    solve would give zero for all three, which is the error this catches.
    """
    _, _, calculation, result = _cycloid()
    solver = make_sternheimer(
        calculation, result, metals=True, noncollinear=True
    )
    drho = np.asarray(solver.chi0(_probe(calculation, (1.0, 0.0, 0.0, 0.0))))
    charge = np.abs(drho[0]).max()
    in_plane = min(np.abs(drho[1]).max(), np.abs(drho[2]).max())
    out_of_plane = np.abs(drho[3]).max()
    print(f"\ncharge probe: n={charge:.3e} m_x={np.abs(drho[1]).max():.3e} "
          f"m_y={np.abs(drho[2]).max():.3e} m_z={out_of_plane:.3e}")
    assert in_plane > 0.1 * charge
    assert out_of_plane < 1e-4 * charge


@pytest.mark.slow
def test_a_zeeman_probe_along_the_hard_axis_moves_no_charge():
    """The other cross term, and its null.

    A field along ``z`` is perpendicular to every moment of a planar texture, so
    to first order it tilts them out of the plane and moves no charge at all.
    Measured: an induced ``m_z`` of **0.918** with an induced charge of
    **7.3e-6** beside it.

    The null is the useful half. The same probe applied *in* the plane does move
    charge -- 0.176 against an induced ``m_x`` of 0.938 -- so this is a
    statement about the texture rather than a channel that is wired to zero,
    which is the distinction a clean zero cannot make on its own.
    """
    _, _, calculation, result = _cycloid()
    solver = make_sternheimer(
        calculation, result, metals=True, noncollinear=True
    )
    hard = np.asarray(solver.chi0(_probe(calculation, (0.0, 0.0, 0.0, 1.0))))
    easy = np.asarray(solver.chi0(_probe(calculation, (0.0, 1.0, 0.0, 0.0))))
    print(f"\nB || z: m_z={np.abs(hard[3]).max():.3e} n={np.abs(hard[0]).max():.3e}")
    print(f"B || x: m_x={np.abs(easy[1]).max():.3e} n={np.abs(easy[0]).max():.3e}")
    # The guard fires: a probe in the plane moves charge, one out of it does not.
    assert np.abs(hard[0]).max() < 1e-4 * np.abs(hard[3]).max()
    assert np.abs(easy[0]).max() > 0.1 * np.abs(easy[1]).max()


@lru_cache(maxsize=2)
def _silicon(noncolin: bool):
    """``si-epsilon`` as a scalar run and as a spinor with no magnetization.

    The spinor run is the *same input file* with one line added rather than a
    second committed input, so the identity below cannot be weakened by the two
    sides drifting apart. ``Si.pz-vbc.UPF`` is not relativistic and nothing
    seeds a moment, so ``nspin_mag`` stays 1 and the two runs are the same
    physics on a doubled space: four bands of two electrons against eight of
    one.
    """
    from defumat.scf import Calculation

    parsed = read_pw_input(CASES / "si-epsilon.in")
    if noncolin:
        parsed.namelists["system"]["noncolin"] = True
    system = build_system(parsed)
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    assert calculation.noncolin == noncolin
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-12,
                     max_iterations=80)
    assert result.converged
    return system, pseudos, calculation, result


@pytest.mark.slow
def test_a_spinor_with_no_magnetization_is_the_scalar_run():
    """The identity that catches a factor of two in the spin sum.

    Every ``KPoints`` constructor applies the unpolarized ``degspin``
    unconditionally and a spinor band holds **one** electron, so the bookkeeping
    reaches the k-point weights, the occupied-band count
    (:func:`~defumat.response.sternheimer.occupied_counts`'s ``degeneracy = 1 if
    noncolin``) and the density. A factor of two anywhere along that chain is
    100 per cent here and nothing else in this file would see it: the finite
    difference above shares the convention with the solve.

    Measured: the two total energies **1.8e-15 Ry** apart and ``chi_0``
    **1.3e-9** apart, which is the CG threshold rather than round-off -- the two
    solves converge their own ``dpsi`` to 1e-11 independently.
    """
    responses = {}
    for noncolin in (False, True):
        _, _, calculation, result = _silicon(noncolin)
        assert calculation.nspin_mag == 1
        solver = make_sternheimer(calculation, result, noncollinear=noncolin)
        responses[noncolin] = (
            np.asarray(solver.chi0(_probe(calculation, (1.0,)))),
            float(result.total_energy),
        )
    scalar, spinor = responses[False][0], responses[True][0]
    assert abs(responses[False][1] - responses[True][1]) < 1e-12
    relative = np.abs(scalar - spinor).max() / np.abs(scalar).max()
    print(f"\nspinor vs scalar chi_0: {relative:.3e}")
    assert relative < 1e-8


def _covariant_probe(calculation):
    """``dv_a = d(vltot)/dr_a`` -- three probes that really are a polar vector.

    A set of three perturbations labelled by a direction can only be
    symmetrised if it *transforms* as a vector under the group, and one
    arbitrary plane wave per direction does not: symmetrising such a set
    compares the response to two different perturbations and the comparison
    below would fail for both routes, which reads as a null rather than as a
    verdict. The gradient of an **invariant** scalar is covariant by
    construction, and ``vltot`` is invariant because the crystal built it.

    Only the charge channel is perturbed. The spin index of the *response*
    comes from the ground state's own texture, and that is the index the axial
    rule acts on.
    """
    from defumat.basis.fft import r_to_g
    from defumat.basis.gradients import gradient

    gvectors = calculation.basis.dense
    grad = gradient(
        r_to_g(jnp.asarray(calculation.vltot), gvectors.fft_index),
        gvectors, calculation.system.cell,
    )
    zero = jnp.zeros_like(grad[0])
    return jnp.stack([
        jnp.stack([grad[a]] + [zero] * (calculation.nspin_mag - 1))
        for a in range(3)
    ])


@pytest.mark.slow
def test_a_spinor_response_is_symmetrised_as_an_axial_vector():
    """The one piece of new physics here, against the whole zone.

    ``h4-cycloid-90`` and ``h4-cycloid-90-nosym`` are the same cell with and
    without symmetry -- 3 k-points under 4 operations against the closed 4-point
    grid -- and their total energies agree. So a response computed on the wedge
    and symmetrised must equal the one computed on the whole grid, and *how* it
    is symmetrised is the whole question: the perturbation direction is a polar
    index and the three magnetization channels of the induced density are an
    **axial** one, carrying ``det(R) (-1)^t_rev`` beside the same rotation.

    Measured, as a fraction of the largest response on the free run:

    * the axial rule: **2.6e-5**, which is the two ground states' own difference
      (the symmetrised run is planar to 1.7e-21 where the free one reaches 8e-6);
    * carrying the magnetization channels as three **scalars**, which is what
      this method did before: **0.665**;
    * no symmetrisation at all: **9.0e-4**.

    The middle number is the finding. A wrong rotation here is not a worse
    average -- it is 66 per cent, and **worse than doing nothing**, because the
    group's mirrors have ``det(R) = -1`` and rotating the magnetization without
    that sign cancels the very components it should be averaging.
    """
    from defumat.basis.fft import g_to_r, r_to_g
    from defumat.system.symmetry import (
        cartesian_rotations, symmetrize_vector_density,
    )

    def directions(name):
        _, _, calculation, result = _converged(
            name, conv_thr=1e-12, max_iterations=200
        )
        solver = make_sternheimer(
            calculation, result, metals=True, noncollinear=True
        )
        dv = _covariant_probe(calculation)
        return calculation, np.stack(
            [np.asarray(solver.chi0(dv[a])) for a in range(3)]
        )

    reduced_calc, reduced = directions("h4-cycloid-90")
    _, free = directions("h4-cycloid-90-nosym")
    assert reduced_calc.use_symmetry

    def as_scalars(calculation, fields):
        """The old route, kept so the guard can be shown to fire."""
        gvectors = calculation.basis.dense
        permutations, phases = calculation._symmetry_maps
        rotations = jnp.asarray(
            cartesian_rotations(calculation.system.cell, calculation.symmetries)
        )

        def channel(three):
            in_g = jnp.stack([r_to_g(f, gvectors.fft_index) for f in three])
            out_g = symmetrize_vector_density(
                in_g, permutations, phases, rotations
            )
            return jnp.stack([
                jnp.real(g_to_r(c, gvectors.fft_index, gvectors.grid))
                for c in out_g
            ])

        moved = jnp.moveaxis(jnp.asarray(fields), 1, 0)
        return np.asarray(
            jnp.moveaxis(jnp.stack([channel(c) for c in moved]), 0, 1)
        )

    scale = np.abs(free).max()
    axial = np.asarray(
        reduced_calc.symmetrize_directional(jnp.asarray(reduced))
    )
    scalarised = as_scalars(reduced_calc, reduced)
    errors = {
        "axial": np.abs(axial - free).max() / scale,
        "scalar": np.abs(scalarised - free).max() / scale,
        "none": np.abs(reduced - free).max() / scale,
    }
    print("\nwedge against the whole zone: " + "  ".join(
        f"{k}={v:.3e}" for k, v in errors.items()))
    # **Per channel, because the explanation has to be discriminated from the
    # alternative.** "The residual is the two ground states' own difference" is
    # an explanation that fits the number; the charge channel is the test of it.
    # That channel takes the *same* polar rotation under both routes and never
    # sees the axial sign, so if its error matches the magnetization channels'
    # the residual is common-mode and the explanation holds -- and if it is
    # orders smaller, something in the axial rule is slightly wrong.
    labels = ["n", "m_x", "m_y", "m_z"]
    print("  axial, per channel: " + "  ".join(
        f"{labels[c]}={np.abs(axial[:, c] - free[:, c]).max() / scale:.3e}"
        for c in range(free.shape[1])))
    assert errors["axial"] < 1e-4
    # The guard fires: the wrong rotation is not a worse average, and it is
    # worse than leaving the wedge sum alone.
    assert errors["scalar"] > 0.1
    assert errors["scalar"] > 100.0 * errors["none"]


def test_what_a_spinor_response_still_refuses():
    """The refusals, by name, with the missing term in the message.

    Cheap: no SCF, no solve -- the guard is a function of the calculation alone.
    """
    from defumat.response.sternheimer import require_a_sternheimer_regime
    from defumat.scf import Calculation

    system = build_system(read_pw_input(CASES / "h-chain-90deg.in"))
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)

    # Every caller keeps its refusal until its own assembly is validated.
    with pytest.raises(NotImplementedError, match="noncollinear"):
        require_a_sternheimer_regime(calculation, metals=True)
    require_a_sternheimer_regime(calculation, metals=True, noncollinear=True)

    # Ultrasoft and PAW stay refused, and the reason is a real missing term.
    ultrasoft = build_system(read_pw_input(CASES / "pt-soc-nosym.in"))
    calculation = Calculation(
        ultrasoft,
        tuple(read_upf(PSEUDO / s.pseudo_file)
              for s in ultrasoft.structure.species),
    )
    assert calculation.is_ultrasoft
    with pytest.raises(NotImplementedError, match="set_int3_nc"):
        require_a_sternheimer_regime(
            calculation, metals=True, noncollinear=True
        )
