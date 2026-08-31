"""The shift current on a crystal: the sum rule, the tensor, and the refusals.

``sigma^abc(0; w, -w)`` is the bulk photovoltaic effect, and there is no
reference binary to compare it against -- ``pw.x`` computes no photocurrent of
any kind, and the one implementation in the vendored tree is Wannier90's
``berry_task = 'sc'``, which needs a wannierisation this project does not do.
So the validation is internal, and it is laid out as the three things that can
independently be wrong.

**The generalised derivative.** ``r^{c;a}_nm`` from the Aversa-Sipe sum rule
against a **parallel-transport finite difference** of the dipole itself, which
shares nothing with it: no second derivative of ``H``, no intermediate band
sum, only the phases of neighbouring k-points fixed to each other. The
comparison is decisive in one particular form and vacuous in every other, and
the reason is the whole point of this file: *the sum rule is an identity over a
complete basis*. Held on a frozen plane-wave sphere, the basis is finite -- 158
plane waves for AlAs at this cutoff -- so running the band set out to 158 makes
the intermediate sum complete and the identity exact. It agrees to **1.8e-4**
there, against 4.3e-2 at 120 bands and 6.0e-2 at 20. The sweep is the test: a
residue that falls off a cliff at completeness is a correct assembly with a
truncated sum, and one that plateaus is a bug.

**Two things about that stencil had to be got right and neither is obvious.**
The phase correction is ``r_nm -> ph_n conj(ph_m) r_nm`` and the transposed
version -- which is what one writes first -- leaves an O(1) phase error that
the difference divides by ``2 delta``, so the disagreement *grows* as the step
shrinks. And the sphere must be **frozen across the stencil**: rebuilt per
k-point it holds 158 plane waves at ``k`` and ``k - delta`` and **157** at
``k + delta`` on a generic AlAs point, and that fixed variational offset is
divided by ``2 delta`` in exactly the same way. It is P48's finding -- a
stencil must not straddle a change of basis -- one derivative further up, and
in the arm rather than the centre.

**The tensor.** AlAs is zincblende, ``-43m``: ``sigma^abc`` is a polar rank-3
tensor symmetric in its two field labels, so only the piezoelectric-like
components with all three labels different survive. It comes out that way on a
``nosym`` grid that imposes nothing. Silicon, which has an inversion centre, is
the control -- and a tensor that is zero for the wrong reason is what a control
is for.

**The spectrum.** It must vanish below the absorption edge, because a shift
current is carried by photoexcited carriers and there are none. That is the one
check here that no amount of index bookkeeping can accidentally pass.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pypresso.io.pwin import read_pw_input
from pypresso.pseudo.upf import read_upf
from pypresso.response.photocurrent import (
    dipole_matrix,
    generalized_derivative,
    require_a_shift_current_regime,
)
from pypresso.scf.driver import Calculation, run_scf
from pypresso.system.builder import build_system
from pypresso.system.kpoints import KPoints
from pypresso.units import RY_TO_EV
from pypresso.workflows.photocurrent import run_shift_current

pytestmark = pytest.mark.regression

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: A point of AlAs no symmetry maps to itself and where no two of the low bands
#: are close: ``[0.1875, 0.3125, 0]`` has bands 2 and 3 only 1.0e-2 Ry apart,
#: which individual-band parallel transport cannot follow.
GENERIC_K = np.array([0.11, 0.23, 0.31])


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """``CLAUDE.md``'s bound on a file that runs several distinct cells.

    Every cell here compiles the whole velocity stack afresh -- twice, since the
    second derivative is a ``jvp`` of a ``jvp`` -- and XLA keeps every
    executable for the life of the process.
    """
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def converged(name: str):
    """An SCF run of a committed input. ``maxsize=2``, never ``None``."""
    system = build_system(read_pw_input(CASES / name))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    return system, pseudos, run_scf(system, pseudos, conv_thr=1.0e-10)


def full_mesh(system, n: int) -> KPoints:
    """The whole unshifted ``n x n x n`` grid -- closed under the point group."""
    axis = np.arange(n) / n
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    return KPoints.from_crystal(
        points, np.full(len(points), 1.0 / len(points)), system.cell,
        precision=system.kpoints.precision,
    )


# --- the generalised derivative against a parallel-transport difference -------


def _frozen_sphere_pieces(system, pseudos, density, k_crystal, kcart, nbnd):
    """``(energies, states, v, w)`` at ``kcart`` on the sphere built for ``k_crystal``.

    Everything is held on **one** sphere: ``at_kcart`` moves ``k`` and leaves
    the plane-wave set alone, which is what the velocity operator differentiates
    and therefore what a finite difference of it has to use.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from exact_reference import exact_eigenpairs_all

    kp = KPoints.from_crystal(
        np.asarray(k_crystal, float).reshape(1, 3), np.ones(1), system.cell,
        precision=system.kpoints.precision,
    )
    calculation = Calculation(system, pseudos).at_kpoints(kp)
    v_scf = calculation.potential(density).v_scf
    moved = calculation.at_kcart(jnp.asarray(kcart))
    energies, states = exact_eigenpairs_all(moved.hamiltonian(v_scf, None, None)[0], nbnd)

    from pypresso.response.velocity import VelocityOperator

    operator = VelocityOperator(calculation, v_scf, kcart=jnp.asarray(kcart))
    psi = jnp.asarray(states)[None]
    bg = np.asarray(system.cell.bg)

    def block(applied):
        return np.asarray(jnp.einsum("skmg,skng->skmn", psi.conj(), applied)[0])

    velocity = np.stack([block(operator.apply(psi, bg[d])) for d in range(3)])
    second = np.stack([
        np.stack([block(operator.apply_second(psi, bg[a], bg[c])) for c in range(3)])
        for a in range(3)
    ])
    return np.asarray(energies), np.asarray(states), velocity, second, calculation


def _transported_dipole(reference_states, states, energies, velocity, component):
    """``r^c_nm`` with each band's phase fixed to the reference k-point's.

    ``r_nm -> ph_n conj(ph_m) r_nm``, and the transpose of that is the version
    that leaves an O(1) phase in place -- see this module's docstring.
    """
    overlap = np.conj(reference_states[0]) @ states[0].T
    phase = np.diagonal(overlap)
    phase = phase / np.abs(phase)
    r = np.asarray(dipole_matrix(energies, velocity))[component, 0]
    return r * (phase[:, None] * np.conj(phase)[None, :])


@pytest.mark.slow
def test_the_sum_rule_is_an_identity_over_a_complete_basis():
    """``r^{c;a}`` against a parallel-transport difference, swept to completeness.

    The sweep is the assertion. Truncated at 20, 80 or 120 of AlAs's 158 plane
    waves the sum rule is a few per cent from the difference; at 158, where the
    intermediate sum runs over the whole space it is an identity over, it is
    1.8e-4 -- two to three orders down in one step. Nothing but a correct
    assembly does that, and nothing that is merely close does it either: a
    wrong index would not become right at completeness.

    What is left at 158 is the difference's own ``O(delta^2)`` error.
    """
    system, pseudos, result = converged("alas-raman.in")
    delta = 1.0e-3
    step = np.array([delta, 0.0, 0.0])

    def cartesian(point):
        return np.asarray(KPoints.from_crystal(
            np.asarray(point).reshape(1, 3), np.ones(1), system.cell,
            precision=system.kpoints.precision,
        ).cartesian(system.cell))

    probe = Calculation(system, pseudos).at_kpoints(
        KPoints.from_crystal(GENERIC_K.reshape(1, 3), np.ones(1), system.cell,
                             precision=system.kpoints.precision)
    )
    complete = int(probe.basis.npwx)
    assert complete < 400  # the dense solve below is a fixture, not a solver

    pieces = [
        _frozen_sphere_pieces(system, pseudos, result.density, GENERIC_K,
                              cartesian(GENERIC_K + s), complete)
        for s in (np.zeros(3), step, -step)
    ]
    energies, states, velocity, second, _ = pieces[0]
    a, c = 0, 1
    difference = (
        _transported_dipole(states, pieces[1][1], pieces[1][0], pieces[1][2], c)
        - _transported_dipole(states, pieces[2][1], pieces[2][0], pieces[2][2], c)
    ) / (2 * delta)

    window = difference[:8, :8]
    large = np.abs(window) > 1.0e-2 * np.abs(window).max()
    assert large.sum() > 10

    def residue(bands: int) -> float:
        analytic = np.asarray(generalized_derivative(
            energies[:, :bands], velocity[:, :, :bands, :bands],
            second[:, :, :, :bands, :bands],
        ))[a, c, 0][:8, :8]
        return float(np.max(np.abs(analytic - window)[large] / np.abs(window)[large]))

    truncated, exact = residue(120), residue(complete)
    assert exact < 3.0e-3, f"the identity does not hold at completeness: {exact}"
    assert truncated > 30 * exact, (
        "the residue did not fall at completeness, so what is being measured is "
        f"not truncation: {truncated} at 120 bands against {exact} at {complete}"
    )


# --- the tensor ---------------------------------------------------------------


@lru_cache(maxsize=2)
def spectrum(name: str, n: int, nbnd: int, broadening: float = 0.01):
    """``run_shift_current`` on the whole ``n^3`` grid of a committed input."""
    system, pseudos, result = converged(name)
    return run_shift_current(
        system, pseudos, result.density, kpoints=full_mesh(system, n),
        nbnd=nbnd, window=0.9, nw=180, broadening=broadening,
    )


@pytest.mark.slow
def test_alas_comes_out_exactly_zincblende():
    """Only ``sigma^abc`` with ``a, b, c`` all different survives, on a nosym grid.

    ``-43m`` allows the piezoelectric-like components and nothing else, and the
    six permutations of ``xyz`` must be equal. Nothing here imposes any of it:
    the k-set is the whole unshifted grid, no point-group average is applied,
    and the three cartesian directions are treated identically by construction.

    It is a **weak** check on its own and this file says so twice: the spike
    that :data:`~pypresso.response.photocurrent.DEGENERACY_TOL` documents left
    it passing to five figures while moving the spectrum by two orders of
    magnitude. It is here to catch a transposed cartesian index, which is all
    it is good for.
    """
    sigma = np.asarray(spectrum("alas-raman.in", 6, 14).sigma)
    allowed = {(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)}
    peaks = {
        (a, b, c): np.max(np.abs(sigma[:, a, b, c]))
        for a in range(3) for b in range(3) for c in range(3)
    }
    survives = max(peaks[i] for i in allowed)
    forbidden = max(v for i, v in peaks.items() if i not in allowed)
    assert survives > 1.0e-5, "the case is vacuous: nothing survives"
    # Measured: 4.0e-4 on the 6x6x6 grid and 1.0e-5 on the 4x4x4. It is the
    # residue of the zone sum itself -- the grid is closed under the point
    # group, so nothing forces the forbidden components to cancel except the
    # arithmetic doing so k-point by k-point.
    assert forbidden < 1.0e-3 * survives
    equal = [peaks[i] for i in allowed]
    assert np.ptp(equal) < 1.0e-3 * survives


@pytest.mark.slow
def test_the_shift_current_vanishes_below_the_absorption_edge():
    """No absorption, no photocurrent -- and the edge is the band gap.

    The one statement here that no index bookkeeping can pass by accident: it
    ties the tensor to the *band structure*, through the resonance delta, and a
    sign or a transposition leaves it untouched while a wrong occupation factor
    or a wrong resonance condition destroys it.

    Measured on AlAs: ``sigma^xyz`` is 6.9e-25 A/V^2 at 2 eV against a peak of
    1.1e-4 around 4.4 eV. What is left below the edge is the Gaussian's own
    tail, which is why the bound is read well inside the gap.
    """
    result = spectrum("alas-raman.in", 6, 14)
    energies = result.frequencies_ev
    sigma = np.abs(np.asarray(result.sigma)[:, 0, 1, 2])
    peak = sigma.max()
    assert peak > 1.0e-5
    assert sigma[energies < 2.5].max() < 1.0e-2 * peak
    assert sigma[energies < 1.0].max() < 1.0e-6 * peak


@pytest.mark.slow
def test_silicon_has_no_shift_current_at_all():
    """The control: an inversion centre forbids every component of a rank-3 polar tensor.

    A number that is small has to be shown to be small *for the right reason*,
    which is what a case with the opposite symmetry and the same machinery is
    for: AlAs and silicon differ by one species and are otherwise the same
    calculation.
    """
    silicon = np.asarray(spectrum("si2-nosym.in", 6, 14).sigma)
    alas = np.asarray(spectrum("alas-raman.in", 6, 14).sigma)
    assert np.max(np.abs(silicon)) < 1.0e-3 * np.max(np.abs(alas))


@pytest.mark.slow
def test_a_spinor_run_gives_the_same_current_as_an_unpolarized_one():
    """The **absolute scale's** only anchor, and the reason it is here.

    Every other check in this file is blind to an overall constant: the
    complete-basis identity tests the generalised derivative, and the ``-43m``
    form, the silicon zero and the below-gap vanishing would all survive a
    ``sigma`` wrong by a factor of two. That is P50's trap in this module's
    coordinates -- "the wrong one is exactly zincblende, exactly symmetric,
    vanishes on silicon and is twice too large".

    A cell with no magnetization run as ``nspin = 1`` and as a two-component
    spinor is the check that catches a factor of two in the **spin sum**
    specifically -- P52's own construction, where the same factor is a factor
    of four in ``N(q)`` and invisible in its shape. The two paths reach the
    weights differently: ``for_spin`` gives the unpolarized k-set weights
    summing to 2 and the spinor's summing to 1, and a spinor band holds one
    electron where an unpolarized band holds two. Measured: **4.6e-9**, with the
    peak ratio 1.000000.

    It is also the only test behind the README row's claim that the shift
    current runs on a spinor calculation at all.
    """
    system, pseudos, result = converged("alas-raman.in")
    spinor = system.with_spin(4)
    from pypresso.scf.driver import run_scf as _run_scf

    spinor_result = _run_scf(spinor, pseudos, conv_thr=1.0e-10)

    unpolarized = run_shift_current(
        system, pseudos, result.density, kpoints=full_mesh(system, 4),
        nbnd=14, window=0.9, nw=90, broadening=0.02,
    )
    doubled = run_shift_current(
        spinor, pseudos, spinor_result.density, kpoints=full_mesh(spinor, 4),
        nbnd=28, window=0.9, nw=90, broadening=0.02,
    )
    one = np.asarray(unpolarized.sigma)
    two = np.asarray(doubled.sigma)
    scale = max(np.abs(one).max(), np.abs(two).max())
    assert scale > 1.0e-5, "the case is vacuous"
    assert np.max(np.abs(one - two)) < 1.0e-7 * scale


# --- the refusals -------------------------------------------------------------


@pytest.mark.slow
def test_a_symmetry_reduced_wedge_is_refused_by_name():
    """``sigma^abc`` is a polar rank-3 tensor and a wedge sum is not the cell's."""
    system = build_system(read_pw_input(CASES / "alas-raman-wedge.in"))
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    with pytest.raises(NotImplementedError, match="whole k-grid"):
        require_a_shift_current_regime(Calculation(system, pseudos))


@pytest.mark.slow
def test_dft_plus_u_does_not_fall_through_to_a_ragged_error():
    """A DFT+U run is refused by name, not by whatever breaks first.

    Worth a test because the failure it prevents is *ragged* rather than wrong:
    without a guard a DFT+U caller reaches
    :class:`~pypresso.response.velocity.VelocityOperator`'s request for the
    converged ``ns``, which ``run_shift_current`` has no parameter to pass it,
    so the amber box's promise -- that a run which starts is a run whose physics
    is there -- would be kept by accident and reported by the wrong error.

    **What this actually exercises is narrower than the guard**, and saying so
    is the point: every DFT+U input committed here uses ``Ni.pz-nd-rrkjus``, so
    the *ultrasoft* refusal fires first on all of them and the Hubbard branch is
    unreachable without a norm-conserving Hubbard cell, which this suite does
    not have. What is checked is that such a run stops with a refusal naming a
    missing term rather than with a missing keyword argument; the Hubbard
    branch's own message is asserted by
    ``tests/unit/test_photocurrent_machinery.py`` on the function directly.
    """
    system = build_system(read_pw_input(CASES / "ni-ldau-nospin.in"))
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    with pytest.raises(NotImplementedError, match="not implemented"):
        require_a_shift_current_regime(Calculation(system, pseudos))
