"""P37: the bootstrap kernel of TDDFT, and the ``chi_0`` it stands on.

The phase is a transcription of Elk's ``tddftlr.f90`` and its two helpers, so
the checks that matter are the ones that share nothing with the transcription:

* **the identity that certifies everything at once.** The macroscopic
  dielectric constant of an insulator can be reached two ways here, and they
  have no machinery in common -- an Adler-Wiser sum over states followed by a
  Dyson equation in G space (:mod:`defumat.tddft`), and a projected conjugate
  gradient solve of the Sternheimer equation with the position operator on its
  right-hand side (:mod:`defumat.response.efield`). Agreement pins the pair
  matrix elements, the occupation weights, the Coulomb symmetrisation, the
  optical head, the wings and the matrix inversion in one number.

  **It is an identity only when the two kernels match**, which is the trap:
  ``efield``'s screening is ``dv_of_drho``, Hartree *plus* ``f_xc``, so it is
  ALDA and not RPA. Both pairings are checked -- ALDA against the default
  ``efield``, and RPA against ``efield`` with its exchange-correlation term
  switched off -- because getting that backwards leaves a percent-level residue
  that looks exactly like band truncation.
* **the body against the Sternheimer operator, column by column.** Sharper than
  the scalar above and blind to the head: ``chi_0`` applied to ``cos(G'.r)`` is
  a column of the same matrix the sum over states builds, and the Sternheimer
  route is band-*complete*, so the two agree only as ``nbnd`` grows. That is
  what makes it the measurement of this phase's one unrefusable error.
* **the kernels' shapes**, which are physics: ALDA's head and wings are exactly
  zero, and that -- not its size -- is why it cannot bind an exciton.

What is *not* checked here is a spectrum against another code's. The reference
would be Elk, which is all-electron LAPW where this is a pseudopotential plane
wave, so a peak height is not a comparable number; the notebook makes that
comparison qualitatively and this file stays with the identities.
"""

from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.basis.fft import g_to_r, r_to_g
from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.scf import Calculation, run_scf
from defumat.system import build_system
from defumat.units import RY_TO_EV
from defumat.tddft import (
    alda_matrix,
    independent_response,
    solve_dyson,
)
from defumat.workflows.nscf import fixed_density_states

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: The one committed cell this phase can run on: silicon on an **unshifted**
#: 4x4x4 grid with ``nosym``, so the k-set is the whole grid. A reduced wedge is
#: refused (``chi_0(G, G')`` would need a rotation in two G indices at once) and
#: an unshifted grid is what makes ``nosym`` sound in the first place -- the
#: same pairing ``si-epsilon-unshifted-nosym.in`` was committed for.
CASE = "si-epsilon-unshifted-nosym"

#: Enough empty states that the static value is converged to a few parts in
#: ten thousand, which is what the identity below is asserted at. It is not a
#: general recommendation: see ``static_residual``.
NBND = 60

#: The response cutoff, Ry. At 8 the local-field effect is converged to 2e-3 on
#: this cell; at 2 it is half missing.
ECUT_RESPONSE = 8.0

#: How far the two routes may sit apart. The measured differences are 1.3e-2 on
#: a dielectric constant of 22, and what is left at 60 bands is the sum over
#: states' truncation against a solve that has no truncation at all.
IDENTITY_TOLERANCE = 3.0e-2


@lru_cache(maxsize=None)
def _silicon(nbnd: int = NBND):
    """The converged ground state, and a fixed-density run with empty states."""
    system = build_system(read_pw_input(CASES / f"{CASE}.in"))
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    scf = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-12,
                  max_iterations=80)
    calculation, system, eigenvalues, wavefunctions = fixed_density_states(
        system, pseudos, scf.density, nbnd=nbnd, conv_thr=1e-10
    )
    return system, pseudos, scf, calculation, eigenvalues, wavefunctions


@lru_cache(maxsize=None)
def _chi(nbnd: int = NBND, ecut: float = ECUT_RESPONSE):
    _, _, scf, calculation, eigenvalues, wavefunctions = _silicon(nbnd)
    potential = calculation.potential(jnp.asarray(scf.density))
    # Index 0 is the static point and carries **no** broadening, so that it is
    # comparable with a Sternheimer solve, which has none either.
    return independent_response(
        calculation, wavefunctions, eigenvalues, potential.v_scf,
        np.array([0.0]), ecut_response=ecut, broadening=0.0,
    )


@lru_cache(maxsize=None)
def _sternheimer(screening: str) -> float:
    """``epsilon_infinity`` from the projected CG solve, isotropic average."""
    from defumat.response.efield import dielectric_tensor

    _, _, scf, calculation, eigenvalues, wavefunctions = _silicon()
    nocc = int(round(calculation.nelec / 2))
    response = dielectric_tensor(
        calculation, wavefunctions[:, :, :nocc], eigenvalues[:, :, :nocc],
        jnp.asarray(scf.density), screening=screening, born_charges=False,
    )
    return float(np.diag(np.asarray(response.epsilon)).mean())


@pytest.mark.parametrize("kernel,screening", [("rpa", "hartree"), ("alda", "full")])
def test_the_dyson_equation_reproduces_the_sternheimer_dielectric_constant(
    kernel, screening
):
    """The phase's certifying identity, in both of its kernel-matched pairings.

    Two routes to ``eps_M(0)`` that share the ground state and nothing else. The
    sum over states builds ``chi_0`` as a matrix from occupied-empty pairs and
    inverts a Dyson equation; the Sternheimer route never forms a matrix, never
    sees an empty state, and reaches the same number by solving
    ``(H - eps S)|dpsi> = -P_c r|psi>`` with the induced field mixed to
    self-consistency.

    **The pairing is the content of the test.** ``efield``'s kernel is one
    ``jvp`` of ``v_of_rho``, so it screens with Hartree *and* ``f_xc``: it is
    the ALDA answer. Comparing it against an RPA sum over states is not an
    identity at all, and the residue -- 1.3 on a constant of 22 here, six
    percent -- is the exchange-correlation kernel being measured by accident.
    """
    chi = _chi()
    context = {}
    if kernel == "alda":
        _, _, scf, calculation, _, _ = _silicon()
        context["alda_matrix"] = alda_matrix(
            calculation, jnp.asarray(scf.density), chi.sphere
        )
    solution = solve_dyson(chi, kernel, context)
    here = float(np.real(np.diag(np.asarray(solution.epsilon)[0])).mean())
    assert abs(here - _sternheimer(screening)) < IDENTITY_TOLERANCE


def test_the_two_kernel_pairings_are_not_interchangeable():
    """Mismatching them is worth six percent, which is why the test above pairs them.

    Stated as an assertion rather than left in a comment: if ``f_xc`` ever
    stopped reaching ``efield``'s screening kernel, the RPA and ALDA numbers
    would coincide and the identity above would still pass while checking half
    of what it claims to.
    """
    separation = abs(_sternheimer("full") - _sternheimer("hartree"))
    assert separation > 10.0 * IDENTITY_TOLERANCE


@pytest.mark.parametrize("nbnd,tolerance", [(16, 0.25), (30, 0.05), (60, 0.005)])
def test_the_body_converges_onto_the_sternheimer_operator_with_bands(nbnd, tolerance):
    """A column of ``chi_0`` against the same column from the projected CG solve.

    Blind to the head and to the Dyson inversion, so it isolates what the
    identity above can only certify jointly: the pair densities, the weights and
    the Coulomb symmetrisation.

    The probe is ``cos(G'.r)``, a *real* potential -- which is what the
    Sternheimer solver takes -- so it carries both ``+G'`` and ``-G'`` and the
    prediction is the average of two columns. The reference is band-complete
    where this sum is truncated, so the tolerance is a function of ``nbnd`` and
    the sequence, not any one entry, is the statement.
    """
    _, _, scf, calculation, eigenvalues, wavefunctions = _silicon(nbnd)
    chi = _chi(nbnd)
    sphere = chi.sphere
    coulomb = np.asarray(sphere.sqrt_coulomb)
    body = np.asarray(chi.x)[0][3:, 3:] / np.outer(coulomb, coulomb)
    reflection = np.asarray(sphere.reflection)

    solver = _solver(nbnd)
    column = 0
    coefficients = np.zeros(sphere.nbody, dtype=complex)
    coefficients[column] = 0.5
    coefficients[reflection[column]] += 0.5
    probe = np.real(np.asarray(g_to_r(
        jnp.asarray(coefficients), sphere.fft_index, calculation.basis.smooth.grid
    )))
    reference = np.asarray(r_to_g(
        solver.chi0(jnp.asarray(probe)[None])[0], sphere.fft_index
    ))
    predicted = 0.5 * (body[:, column] + body[:, reflection[column]])
    relative = np.abs(reference - predicted).max() / np.abs(reference).max()
    assert relative < tolerance


@lru_cache(maxsize=None)
def _solver(nbnd: int):
    from defumat.response.sternheimer import SternheimerSolver

    _, _, scf, calculation, eigenvalues, wavefunctions = _silicon(nbnd)
    potential = calculation.potential(jnp.asarray(scf.density))
    nocc = int(round(calculation.nelec / 2))
    weights, _ = calculation.occupations(eigenvalues)
    return SternheimerSolver(
        calculation, calculation.hamiltonian(potential.v_scf, None),
        wavefunctions[:, :, :nocc], eigenvalues[:, :, :nocc],
        jnp.asarray(weights)[:, :, :nocc], nocc, 1e-14, v_scf=potential.v_scf,
    )


def test_local_fields_are_what_the_macroscopic_inversion_is_for():
    """``eps_M`` is the inverse of the head, not the head of the inverse.

    The two differ by the whole local-field effect and by nothing else, and
    ``tddftlr.f90`` writes both from the same array thirty lines apart. Taking
    the wrong one gives a spectrum that is smooth, positive, has the right peaks
    and is nine percent too large -- so this asserts both that the code takes
    the head's inverse and that the two are genuinely different here.
    """
    chi = _chi()
    solution = solve_dyson(chi, "rpa", {})
    with_fields = float(np.real(np.diag(np.asarray(solution.epsilon)[0])).mean())
    without = float(np.real(
        np.diag(np.asarray(solution.epsilon_no_local_fields)[0])
    ).mean())
    # In RPA the head of the *full* matrix inverse is exactly the no-local-field
    # value, which is the identity that makes the mistake so easy to make.
    whole = np.linalg.inv(np.asarray(solution.epsilon_inverse)[0])[:3, :3]
    assert abs(float(np.real(np.diag(whole)).mean()) - without) < 1e-8
    assert with_fields < without
    assert (without - with_fields) / without > 0.05


def test_the_alda_kernel_has_no_head_and_no_wings():
    """Which is why no adiabatic local kernel binds an exciton, whatever its size.

    ``f_xc`` is finite as ``q -> 0`` and ``v`` diverges, so the symmetrised
    kernel ``f_xc / v`` vanishes in the optical limit exactly. The bootstrap's
    does not, and that -- not its magnitude -- is the whole difference between
    the two. ``genvfxc.f90`` writes the zeros in by hand; here they come out of
    the construction, so this checks the construction.
    """
    _, _, scf, calculation, _, _ = _silicon()
    chi = _chi()
    context = {"alda_matrix": alda_matrix(
        calculation, jnp.asarray(scf.density), chi.sphere
    )}
    alda = np.asarray(solve_dyson(chi, "alda", context).fxc)[0]
    assert np.abs(alda[:3, :]).max() == 0.0
    assert np.abs(alda[:, :3]).max() == 0.0
    assert np.abs(alda[3:, 3:]).max() > 0.0

    bootstrap = np.asarray(solve_dyson(chi, "bootstrap", {}).fxc)[0]
    assert np.abs(bootstrap[:3, :3]).max() > 0.0


def test_the_bootstrap_reaches_a_fixed_point_and_says_how_far_it_got():
    """``fxctype = 210``: a self-consistency, so convergence is a claim to check.

    The kernel is built from ``eps^-1`` which is built from the kernel, and the
    exit test is ``tddftlr.f90``'s -- the head of ``F X`` at ``omega = 0``
    ceasing to move. Non-convergence is an error there and here, because a
    spectrum from a kernel still in motion is the spectrum of nothing.
    """
    solution = solve_dyson(_chi(), "bootstrap", {})
    assert solution.converged
    assert 1 < solution.iterations < 50
    # Geometric, and monotone once past the first pass -- the first difference
    # is against Elk's crude seed ``eps^-1 = 1 + X`` and is not informative.
    tail = np.asarray(solution.history[1:])
    assert np.all(np.diff(tail) < 0.0)
    assert solution.alpha > 0.0


def test_a_kernel_moves_spectral_weight_downhill():
    """The claim the whole phase exists to make, at its smallest testable size.

    An excitonic kernel is attractive, so it moves oscillator strength *down* in
    energy. On silicon that is a redistribution rather than a bound state --
    there is no peak below the gap to find, which is why the wide-gap case is
    the notebook's -- so what is asserted is the redistribution itself, two ways
    that do not share a definition: the first moment of the spectrum, which
    needs no reference point, and the weight below a **common** cut.

    The cut has to be common. Taking each kernel's own absorption maximum moves
    the window with the spectrum and reverses the answer -- measured: 0.594 for
    the bootstrap against 0.606 for RPA on its own peak, and 0.643 against 0.606
    on a shared one. The quantity being compared has to be the same quantity.

    **ALDA moves weight too**, and asserting otherwise would be wrong: what it
    cannot do is *bind*, because its kernel has no ``1/q^2`` head
    (:func:`test_the_alda_kernel_has_no_head_and_no_wings`). Shifting a
    continuum and creating a bound state are different claims and only the
    second is the bootstrap's alone.
    """
    _, _, scf, calculation, eigenvalues, wavefunctions = _silicon()
    potential = calculation.potential(jnp.asarray(scf.density))
    grid = np.arange(0.0, 0.6, 0.005)
    chi = independent_response(
        calculation, wavefunctions, eigenvalues, potential.v_scf,
        np.concatenate([[0.0], grid]), ecut_response=ECUT_RESPONSE,
        broadening=0.012,
    )
    context = {"alda_matrix": alda_matrix(
        calculation, jnp.asarray(scf.density), chi.sphere
    )}

    centroid, fraction, cut = {}, {}, None
    for kernel in ("rpa", "alda", "bootstrap"):
        epsilon = np.asarray(solve_dyson(chi, kernel, context).epsilon)[1:]
        absorption = np.imag(np.trace(epsilon, axis1=1, axis2=2)) / 3.0
        if cut is None:  # RPA's peak, shared by all three
            cut = int(np.argmax(absorption))
        centroid[kernel] = float((grid * absorption).sum() / absorption.sum())
        fraction[kernel] = float(absorption[:cut].sum() / absorption.sum())

    assert centroid["bootstrap"] < centroid["rpa"]
    assert centroid["alda"] < centroid["rpa"]
    assert fraction["bootstrap"] > fraction["rpa"]
    assert fraction["alda"] > fraction["rpa"]


# --- LiF: the bound exciton, against Elk's own example -----------------------

#: Where Elk puts LiF's exciton, from ``examples/TDDFT-optics/LiF-bootstrap``
#: run unmodified (``reference.out.elk-lif-bootstrap``). The comparison is an
#: all-electron LAPW spectrum against a pseudopotential plane-wave one, so it is
#: a *soft* one and what is asserted is structure, not a peak height.
ELK_EXCITON_EV = 13.674
ELK_RPA_PEAK_EV = 24.490

#: How far the exciton may sit from Elk's. Measured: 0.14 eV on the 4x4x4 grid
#: below and 0.37 eV on the 8x8x8 one Elk uses -- and the k-offset is **not**
#: where that comes from. Repeating the 8x8x8 run on Elk's own shifted grid
#: (``vkloff = 0.25 0.5 0.625``) moves the peak height from 16.78 to 17.72,
#: against Elk's 18.53, and leaves the position where it was. What is left is
#: the pseudopotential: no Li semicore, and an LDA gap 0.5 eV smaller than the
#: all-electron one, hence a larger scissors shift.
EXCITON_TOLERANCE = 0.8


@pytest.mark.slow
def test_lif_has_a_bound_exciton_where_elk_puts_one():
    """The kernel's reason to exist, on the material it was demonstrated on.

    LiF's exciton is *bound*: a sharp peak below the gap carrying most of the
    oscillator strength, which RPA does not produce at all -- unlike silicon,
    where the same kernel only redistributes a continuum. So this asserts three
    things RPA fails, and each is a different claim:

    * a peak **below** the RPA peak, by several eV;
    * carrying **several times** its oscillator strength;
    * within :data:`EXCITON_TOLERANCE` of where Elk puts it.

    Run on a 4x4x4 grid rather than Elk's 8x8x8, which keeps this a test rather
    than an errand -- and the choice is what decides which claims are asserted.
    **The exciton survives the coarse grid and the RPA continuum does not**: the
    bound peak is at 13.81 eV here against 14.05 at 8x8x8 and Elk's 13.67, while
    the RPA maximum is at 15.65 eV here against 24.37 at 8x8x8 and Elk's 24.49.
    A bound state is a single transition and is sampled long before a continuum
    is. So the continuum's position is *recorded* rather than asserted, and what
    is checked is the exciton, which is the claim anyway.

    **Head-only**, matching Elk's ``gmaxrf = 0.0``: with no body there is no
    local-field effect and no plane-wave matrix element to form, so ``chi_0`` is
    built from the velocity operator alone.
    """
    from defumat.system.kpoints import KPoints
    from defumat.workflows import run_absorption

    system = build_system(read_pw_input(CASES / "lif-tddft.in"))
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    scf = run_scf(system, pseudos, conv_thr=1e-10, max_iterations=100, nbnd=8)

    eigenvalues = np.asarray(scf.eigenvalues)
    gap = float(eigenvalues[:, 4:].min() - eigenvalues[:, :4].max()) * RY_TO_EV
    # The scissors shift is defined by where it has to land -- the experimental
    # 14.2 eV gap -- so it is taken from *this* calculation's LDA gap and not
    # from Elk's 8.97 eV. That is what makes the two spectra comparable.
    scissor = (14.2 - gap) / RY_TO_EV

    grid = KPoints.automatic((4, 4, 4), (0, 0, 0), system.cell,
                             precision=system.kpoints.precision)
    omega = np.linspace(0.0, 3.0, 400)
    spectra = {
        kernel: run_absorption(
            system, pseudos, scf.density, omega, kernel=kernel, kpoints=grid,
            nbnd=20, ecut_response=0.0, broadening=0.02, scissor=scissor,
            static_residual=False,
        )
        for kernel in ("rpa", "bootstrap")
    }

    peak = {k: s.frequencies_ev[int(np.argmax(s.absorption))]
            for k, s in spectra.items()}
    height = {k: float(s.absorption.max()) for k, s in spectra.items()}

    assert abs(peak["bootstrap"] - ELK_EXCITON_EV) < EXCITON_TOLERANCE
    assert peak["bootstrap"] < peak["rpa"] - 1.0
    assert height["bootstrap"] > 3.0 * height["rpa"]
    # And the exciton is genuinely *bound*: below the gap the spectrum starts
    # from, which is what distinguishes it from silicon's redistributed
    # continuum. 14.2 eV is where the scissors shift put that gap.
    assert peak["bootstrap"] < 14.2
