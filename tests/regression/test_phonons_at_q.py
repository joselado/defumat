"""P71: the dynamical matrix at ``q != 0``, and the four things it is checked against.

The zone centre is a special case in every part of a phonon calculation, and
almost all of the ways it is special make a wrong finite-``q`` answer look
right. ``dpsi`` lives on the same sphere as ``psi``; the response density is
real; the ``G = 0`` Hartree term is dropped; the acoustic sum rule holds; the
matrix is real and symmetric. Each of those stops being true away from
``Gamma``, so the checks here are chosen to be ones ``Gamma`` cannot pass by
accident:

* **``q = 0`` through the two-sphere path.** The regression, and the reason it
  is first: everything new -- the second sphere, the shifted local potential,
  the cross-sphere nonlocal term, the transcribed ``d2ionq``, the contraction
  that replaces ``drhodv`` -- has to reproduce P25's matrix, which is validated
  against ``ph.x`` independently. What is left over is the deliberate Ewald
  swap and is measured rather than tolerated.
* **``ph.x`` at ``L`` and at ``X``**, from ``reference.out.ph-si-epsilon-unshifted-dispersion``
  -- one ``ldisp`` run on the 2x2x2 q-grid, which is commensurate with the
  4x4x4 k-grid both codes sample.
* **``D(q + G) = D(q)``.** The dynamical matrix is periodic in the reciprocal
  lattice, and the two sides of that identity are built on *different* plane-wave
  spheres from *different* diagonalisations -- so it is the check that the sphere
  at ``k + q`` and the phases that go with it are consistent, which is P16's
  zone-edge trap in a second place.
* **``D(-q) = conj(D(q))`` and ``D(q)`` hermitian.** Time reversal and the
  reality of the force constants, neither of which anything here imposes.

The acoustic sum rule is *not* among them: it holds at ``Gamma`` and nowhere
else, which is exactly why a finite-``q`` matrix has one fewer free diagnostic
than P25's does.
"""

from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.response.phonon import dynamical_matrix
from defumat.response.phononq import (
    dynamical_matrix_at_q,
    ewald_dynamical_matrix,
    frozen_force_constants,
    require_a_two_sphere_regime,
)
from defumat.response.sternheimer import make_sternheimer
from defumat.scf import Calculation, run_scf
from defumat.system import build_system

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: What the **vendored** ``ph.x`` prints for the 2x2x2 q-grid of
#: ``si-epsilon-unshifted.in``, committed as
#: ``reference.out.ph-si-epsilon-unshifted-dispersion``. Three wavevectors from
#: one run: the zone centre, ``L`` and ``X``, in cartesian units of 2 pi/alat.
#: The two zone-boundary sets are the quantity this phase exists for -- nothing
#: before it could compute a frequency anywhere but the first row.
QE_FREQUENCIES = {
    "Gamma": ((0.0, 0.0, 0.0),
              (3.273643, 3.273643, 3.273643,
               519.198154, 519.198154, 519.198154)),
    "L": ((0.5, -0.5, 0.5),
          (101.842827, 101.842827, 382.240553,
           405.120790, 488.019400, 488.019400)),
    "X": ((0.0, -1.0, 0.0),
          (132.805268, 132.805268, 405.358862,
           405.358862, 455.138711, 455.138711)),
}


@pytest.fixture(autouse=True)
def _bounded_compilation_cache():
    """``CLAUDE.md``'s bound for a file that runs several cells and several q.

    Each ``q`` builds a second plane-wave sphere and therefore a second set of
    shapes, so this file compiles the whole response stack once per wavevector
    and XLA would keep every one of them for the life of the process.
    """
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _ground_state(case: str):
    """The converged run. ``maxsize=2`` for the reason ``CLAUDE.md`` gives."""
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation,
                     conv_thr=1e-12, max_iterations=100)
    return calculation, result


def _phonons(case: str, q, **options):
    calculation, result = _ground_state(case)
    return dynamical_matrix_at_q(
        calculation, result.wavefunctions, result.eigenvalues, result.density,
        result.becsum, q=q, **options,
    )


# ---------------------------------------------------------------------------
# Against ph.x.
# ---------------------------------------------------------------------------

#: How far a frequency may sit from ``ph.x``'s, in cm^-1. Measured: **0.054**
#: at ``L``, the worst of the twelve zone-boundary modes. The floor is the same
#: one P25 quotes at ``Gamma`` (0.05 on this cell) and has the same cause -- QE
#: interpolates every radial form factor from a ``dq = 0.01`` table where this
#: code integrates it directly -- so there is no reason for a zone-boundary
#: mode to hold it more tightly than a zone-centre one.
FREQUENCY_TOLERANCE = 0.2

#: The three **acoustic** modes at ``Gamma`` are exempt, and that is a
#: statement about the reference rather than about this code. Translating the
#: crystal costs no energy, so what those three report is the finite basis's own
#: residue -- and the two runs make it differently, because ``ph.x`` is on the
#: eight-point wedge with the response symmetrised and this is on the whole
#: 64-point grid. 4.32 against 3.27, both of them the same 1e-4 of the force
#: constants. Away from ``Gamma`` there is no acoustic triplet and nothing is
#: exempt, which is why the zone-boundary rows are the sharper comparison.
ACOUSTIC_CEILING = 8.0


@pytest.mark.parametrize("point", ["Gamma", "L", "X"])
def test_the_dispersion_matches_quantum_espresso(point):
    """Silicon at ``Gamma``, ``L`` and ``X`` against the vendored ``ph.x``.

    The zone-boundary rows are what P71 exists for. Nothing in the response
    stack before it could produce a frequency at a wavevector other than zero,
    because a displacement pattern ``u e^{iqR}`` is not a displacement of the
    unit cell and there is no energy in this cell to differentiate twice. What
    makes it reachable is that the *first-order* quantities are still lattice
    periodic once ``e^{iqr}`` is factored out, so the whole calculation stays
    in the unit cell with one extra plane-wave sphere.

    ``ph.x`` runs on the eight-point wedge with the response symmetrised over
    the small group of ``q``; this runs the whole 64-point grid with nothing
    symmetrised at all. So the comparison is also the only check the two routes
    get against each other, in the way P25's wedge-against-grid test is at
    ``Gamma``.
    """
    q, reference = QE_FREQUENCIES[point]
    phonons = _phonons("si-epsilon-unshifted-nosym", q, q_cartesian=True)
    assert phonons.converged
    frequencies = np.asarray(phonons.frequencies)
    if point == "Gamma":
        assert np.abs(frequencies[:3]).max() < ACOUSTIC_CEILING
        frequencies, reference = frequencies[3:], np.asarray(reference)[3:]
    assert np.allclose(frequencies, np.asarray(reference),
                       atol=FREQUENCY_TOLERANCE), (
        f"{point}: {frequencies} against ph.x's {reference}"
    )


# ---------------------------------------------------------------------------
# Against the Gamma machinery.
# ---------------------------------------------------------------------------

#: What ``q = 0`` through the two-sphere path may differ from P25's matrix by,
#: in Ry/bohr^2 on force constants of 0.287. Measured: **1.94e-6**, and it is
#: not noise -- it is the whole of the deliberate Ewald swap. P25's frozen
#: Hessian contains the ground state's own Ewald sum, whose real-space cutoff is
#: ``4/sqrt(alpha)``: ``erfc(4) ~ 2e-8`` bounds the error in the *energy* and
#: not in a second derivative, which carries ``1/r^5``. ``d2ionq.f90`` uses
#: ``5/sqrt(alpha)`` for exactly that reason and this follows it. Extending the
#: ground-state sum to the same cutoff brings the two into agreement at
#: **1.07e-9**, which is what says the residue is that and nothing else.
GAMMA_AGREEMENT = 5e-6


def test_q_zero_reproduces_the_gamma_machinery():
    """Everything new, run at the wavevector where the old code also works.

    This is the regression the whole phase rests on, because the four things
    that change away from ``Gamma`` -- the second sphere, the shifted local
    potential, the ``G = 0`` Hartree term, the complex response density -- all
    have a limit here that is already validated against ``ph.x`` through P25.
    A sign or an index order that is wrong at finite ``q`` is very often still
    wrong at zero, where there is something to compare it with.
    """
    calculation, result = _ground_state("si-epsilon-unshifted-nosym")
    gamma = dynamical_matrix(
        calculation, result.wavefunctions, result.eigenvalues, result.density,
        result.becsum,
    )
    here = _phonons("si-epsilon-unshifted-nosym", (0.0, 0.0, 0.0))
    difference = np.abs(
        np.asarray(here.matrix).reshape(6, 6)
        - np.asarray(gamma.matrix).reshape(6, 6)
    ).max()
    assert difference < GAMMA_AGREEMENT, difference
    # The matrix is complex by construction and real by physics at q = 0.
    assert np.abs(np.imag(np.asarray(here.matrix))).max() < 1e-7


def test_the_ewald_term_reproduces_the_ground_state_hessian_at_q_zero():
    """``d2ionq(0)`` against ``jax.hessian`` of the Ewald energy already here.

    The transcription's own check, and it is worth having separately from the
    matrix: ``d2ionq`` is the one piece of the frozen Hessian that is *not*
    reused from ``Gamma``, so if it is wrong the error arrives inside a total
    that has three other terms in it.

    The comparison is made at the ground-state sum's own ``alpha`` and cutoff,
    which is what makes it an equality rather than a tolerance -- the two
    truncations otherwise differ by the 1.95e-6 that
    :data:`GAMMA_AGREEMENT` is about.
    """
    import jax.numpy as jnp

    from defumat.scf.ewald import EwaldSum
    from defumat.system.cell import lattice_translations, pair_separation_bound

    calculation, _ = _ground_state("si-epsilon-unshifted-nosym")
    cell = calculation.system.cell
    structure = calculation.system.structure
    positions = jnp.asarray(structure.positions)
    alpha = calculation.ewald_sum.alpha

    # The ground state's sum, with its real-space cutoff extended from
    # 4/sqrt(alpha) to d2ionq's 5/sqrt(alpha). Nothing else about it changes.
    rmax = 5.0 / np.sqrt(alpha)
    at = np.asarray(cell.at)
    radius = rmax + pair_separation_bound(at, np.asarray(structure.positions))
    extended = EwaldSum(
        charges=calculation.ewald_sum.charges,
        alpha=alpha,
        rmax=float(rmax),
        translations=jnp.asarray(lattice_translations(at, radius)),
        gamma_factor=calculation.ewald_sum.gamma_factor,
    )
    hessian = np.asarray(jax.hessian(
        lambda pos: extended.energy(cell, pos, calculation.basis.dense)
    )(positions)).reshape(6, 6)

    ours = np.asarray(ewald_dynamical_matrix(
        calculation, jnp.zeros(3), alpha=alpha
    ))
    assert np.abs(ours - hessian).max() < 5e-9
    # The ionic matrix obeys the acoustic sum rule *identically* at q = 0,
    # because the self term is constructed as the sum of the cross terms.
    assert np.abs(ours.reshape(2, 3, 2, 3).sum(axis=2)).max() < 1e-12
    assert np.abs(np.imag(ours)).max() < 1e-12


# ---------------------------------------------------------------------------
# Identities the zone centre cannot check.
# ---------------------------------------------------------------------------

def test_the_frozen_electronic_hessian_carries_no_q():
    """The fact the whole assembly rests on, measured instead of read.

    A displacement pattern ``u_s e^{iq.R}`` moves every cell, so almost
    everything in the second derivative depends on ``q``. The exception is the
    part taken at a **frozen** electronic state: it is diagonal in the atom
    index, so the phases ``e^{iq.R}`` of the two displacements it couples are
    attached to the same atom and cancel. That is why one ground state serves
    every wavevector and only the ion-ion sum has to be redone.

    It was established by reading ``dynmat_us.f90:105-124`` -- ``init_us_2`` is
    called at ``ikk`` and ``compute_nldyn`` uses only ``ikk`` and ``wk(ikk)``,
    never the ``k + q`` index. This is the same statement as a number: the
    frozen half's dependence on ``q`` is *entirely* its Ewald term, so removing
    that term must leave something constant.

    Measured: the frozen half differs by 1.515768e-6 between ``q`` and
    ``q + G``, and :func:`ewald_dynamical_matrix` differs by 1.515768e-6 -- the
    same number, which is the Ewald reciprocal sum's own truncation and is what
    sets the floor of the periodicity check below. The electronic remainder is
    constant to 1e-12, which is where this asserts.

    Cheap by construction: the frozen half needs no linear solve, so this runs
    in a ground state and two Ewald sums.
    """
    calculation, result = _ground_state("si-epsilon-unshifted-nosym")
    cell = calculation.system.cell
    solver = make_sternheimer(calculation, result, threshold=1.0e-14)
    positions = jnp.asarray(calculation.system.structure.positions)

    def frozen_minus_ewald(q_crystal):
        q_cart = np.asarray(
            cell.k_to_cartesian(np.asarray(q_crystal, dtype=float))
        ) * cell.tpiba
        whole = np.asarray(frozen_force_constants(
            calculation, solver, positions, result.density, q_cart
        ))
        return whole - np.asarray(ewald_dynamical_matrix(calculation, q_cart))

    reference = frozen_minus_ewald([0.0, 0.0, 0.0])
    for q in ([0.25, 0.25, 0.0], [0.5, -0.5, 0.5], [1.25, 0.25, -1.0]):
        difference = np.abs(frozen_minus_ewald(q) - reference).max()
        assert difference < 1e-12, (q, difference)


def test_the_matrix_is_periodic_in_the_reciprocal_lattice():
    """``D(q + G) = D(q)``, and the two sides share almost nothing.

    A phonon wavevector is defined modulo a reciprocal lattice vector, so this
    is exact physics -- and on this side of it the two calculations are
    genuinely different objects. ``k + q`` and ``k + q + G`` select **different
    plane-wave spheres**, are diagonalised separately, give first-order
    wavefunctions with different coefficients in different orders, and the
    shifted local potential is tabulated at ``|G' + q|`` against
    ``|G' + q + G|``. What has to cancel for them to agree is exactly the
    bookkeeping P16 records at the zone edge: a coefficient is labelled by its
    Miller index, and the wrap is a *shift* of that index.

    Checked on the matrix rather than the frequencies, since a frequency is
    blind to a phase convention that a matrix element is not.

    **Measured: 8.22e-7**, against a largest element of 0.3255 Ry/bohr^2, so
    2.5e-6 relative. The tolerance is deliberately loose against that, because
    the residue is not the electrons'. The *frozen* half alone differs by
    1.515768e-6 between the two wavevectors, and
    :func:`~defumat.response.phononq.ewald_dynamical_matrix` alone differs by
    1.515768e-6 -- the same number to every digit printed. So what bounds this
    check is the Ewald reciprocal sum's truncation, and the electronic response
    is clean well below it; the time-reversal identity, which has no such
    floor, lands at 3.18e-9.
    """
    here = _phonons("si-epsilon-unshifted-nosym", (0.25, 0.25, 0.0))
    shifted = _phonons("si-epsilon-unshifted-nosym", (1.25, 0.25, -1.0))
    difference = np.abs(
        np.asarray(here.matrix) - np.asarray(shifted.matrix)
    ).max()
    assert difference < 1e-5, difference


def test_time_reversal_conjugates_the_matrix():
    """``D(-q) = conj(D(q))``, and ``D(q)`` is hermitian.

    The force constants of a crystal are real, so their Fourier transform obeys
    both. Nothing here imposes either: the ``-q`` run builds its own second
    sphere and its own first-order states, and the hermitisation in the
    assembly is applied *after* :attr:`~defumat.response.phonon.Phonons.asymmetry`
    has recorded what it removed.

    Measured: **3.18e-9** on the matrix and 8.79e-10 on the asymmetry, both of
    them the linear solves' own residue at ``threshold = 1e-14``. Unlike the
    periodicity check above this one has no Ewald truncation floor -- the two
    runs truncate the ion-ion sum identically, since ``|-q| = |q|`` -- so it is
    the sharper of the two identities, and the tolerances are set to match
    rather than left at a round number the residue clears by four orders.
    """
    plus = _phonons("si-epsilon-unshifted-nosym", (0.25, 0.25, 0.0))
    minus = _phonons("si-epsilon-unshifted-nosym", (-0.25, -0.25, 0.0))
    assert plus.asymmetry < 1e-8, plus.asymmetry
    difference = np.abs(
        np.asarray(minus.matrix) - np.conj(np.asarray(plus.matrix))
    ).max()
    assert difference < 1e-7, difference


# ---------------------------------------------------------------------------
# The refusals.
# ---------------------------------------------------------------------------

def test_a_reduced_k_set_is_refused_by_name():
    """The small group of ``q`` is not written, so the wedge is not usable.

    A response on a reduced k-set has to be averaged over the operations that
    leave the *perturbation* invariant, and for a phonon at ``q`` those are the
    ones with ``S q = q + G`` rather than the crystal's whole group. Averaging
    over the wrong group projects the perturbation away, which is P24's trap
    and P25's, and here the right group does not exist yet.
    """
    calculation, _ = _ground_state("si-epsilon-unshifted")
    with pytest.raises(NotImplementedError, match="small group of q"):
        require_a_two_sphere_regime(calculation, np.array([0.25, 0.25, 0.0]))
