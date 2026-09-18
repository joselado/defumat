"""The transverse spin susceptibility and the magnon, against identities.

``PLAN.md`` P63. There is no reference output to compare against here --
``pw.x`` reaches magnons only through ``TDDFPT``'s Liouville-Lanczos solver,
which never forms a Dyson equation in G space, and an all-electron LAPW
spectrum from Elk is not a comparable number. So the validation closes inside
the package, and it closes twice on statements that have no free parameter:

* **Goldstone.** A global spin rotation costs no energy, so ``1 - X_0 F`` is
  singular at ``q = 0``, ``omega = 0`` with ``m`` its null vector. Since
  ``F m = B_xc`` identically that is ``X_0 B_xc = m``, an equation between two
  arrays computed by completely different routes -- one a sum over states, the
  other the converged density. It is exact only with a complete band set and a
  complete G-set, so what is asserted here is that it **converges in both**.
* **Periodicity.** ``X_0(q + G)`` must equal ``X_0(q)`` after relabelling the
  matrix's own G index, which is the entire content of the umklapp shift the
  ``k + q`` fold needs. It is exact rather than convergent, and it is the check
  that says the finite-``q`` machinery is right before anything is read off a
  dispersion.

Two further checks are here and are cheaper than either: the response of a
centrosymmetric crystal must be the same at ``q`` and ``-q`` after reflecting
its G indices -- which exercises the umklapp on a second, independent pattern --
and the Kohn-Sham response can have no spectral weight below the smallest
spin-flip energy the bands allow.

**What is deliberately not here is a comparison against a spiral ``E(q)``.**
``PLAN.md`` P63 records why: on a half-filled hydrogen lattice the spiral's own
self-consistent field leaves the magnetic branch -- at ``q = 1/2`` it converges
to the *nonmagnetic* solution -- so the energy difference is not a frozen-magnon
energy and comparing it against a susceptibility would be comparing two
different states.
"""

from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat import Calculator
from defumat.tddft.spinchi0 import transverse_response
from defumat.tddft.spinkernel import (
    goldstone_residual,
    transverse_kernel_matrix,
)
from defumat.workflows.nscf import fixed_density_states

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """Every cell here compiles the whole stack afresh and XLA keeps all of it.

    ``CLAUDE.md``'s rule for a file that sweeps more than about three distinct
    cells. The results stay cached below; only the executables are dropped.
    """
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _converged(name: str):
    """The ground state, **refused if it is not one**.

    Everything below takes ``scf.density`` to a functional entry point, which
    is a bare array and carries no convergence flag -- so without this call an
    unconverged cell arrives at a Goldstone assertion and fails as though the
    susceptibility were wrong. It has happened: ``h-fcc-magnon.in`` seeded at
    saturation stalled four orders short and three tests here reported 0.3958
    against a tolerance of 0.02 (``OPEN.md`` Part V).
    """
    calculator = Calculator.from_file(CASES / name, pseudo_dir=PSEUDO)
    scf = calculator.get_scf()
    scf.require_converged(f"the magnon tests on {name}")
    return calculator, scf


@lru_cache(maxsize=2)
def _states(name: str, nbnd: int):
    calculator, scf = _converged(name)
    return fixed_density_states(
        calculator.system, calculator.pseudos, scf.density, nbnd=nbnd
    )


def _response(name, nbnd, ecut, q, frequencies=(0.0,)):
    calculation, _, eigenvalues, wavefunctions = _states(name, nbnd)
    return calculation, transverse_response(
        calculation, wavefunctions, eigenvalues, q,
        np.asarray(frequencies, dtype=float),
        ecut_response=ecut, broadening=0.0,
    )


# --- Goldstone ----------------------------------------------------------------


def test_the_goldstone_identity_converges_in_the_band_count():
    """``X_0 B_xc = m`` improves as the sum over states is completed.

    The identity's completeness sum is over the *minority* manifold, so this is
    the axis that matters on a light element, where ``B_xc`` is smooth enough
    for a small sphere to hold it.
    """
    _, scf = _converged("h-fcc-magnon.in")
    residuals = []
    for nbnd in (12, 30):
        calculation, chi = _response("h-fcc-magnon.in", nbnd, 12.0, (0.0, 0.0, 0.0))
        residuals.append(
            goldstone_residual(chi, calculation, jnp.asarray(scf.density))
        )
    assert residuals[1] < residuals[0]
    assert residuals[1] < 0.02


def test_the_goldstone_identity_converges_in_the_response_sphere():
    """...and on the *other* axis, which is the one that binds on a 3d metal.

    The identity's left side is ``sum_G' X_0(G, G') B_G'``, so a sphere that
    cannot represent ``B_xc`` truncates it whatever the band count. On this
    cell the effect is visible and mild; on fcc nickel the residual is pinned
    at 10 per cent from ``nbnd = 30`` to ``nbnd = 100`` and only moves when the
    cutoff does.
    """
    _, scf = _converged("h-fcc-magnon.in")
    residuals = []
    for ecut in (2.0, 12.0):
        calculation, chi = _response("h-fcc-magnon.in", 30, ecut, (0.0, 0.0, 0.0))
        residuals.append(
            goldstone_residual(chi, calculation, jnp.asarray(scf.density))
        )
    assert residuals[1] < residuals[0]
    assert residuals[1] < 0.02


def test_the_leading_eigenvalue_at_zero_wavevector_is_one():
    """Goldstone again, read as the quantity the Dyson equation actually uses.

    ``lambda_max(X_0 F) = 1`` at ``q = 0``, ``omega = 0`` puts the uniform
    precession at exactly zero energy. It is a different contraction of the
    same identity and it is the number a magnon energy is only as good as.
    """
    calculator, scf = _converged("h-fcc-magnon.in")
    calculation, chi = _response("h-fcc-magnon.in", 30, 12.0, (0.0, 0.0, 0.0))
    kernel = np.asarray(
        transverse_kernel_matrix(calculation, jnp.asarray(scf.density), chi.sphere)
    )
    eigenvalue = np.linalg.eigvals(np.asarray(chi.x[0]) @ kernel).real.max()
    assert eigenvalue == pytest.approx(1.0, abs=0.05)


# --- periodicity, which is the umklapp fold -----------------------------------


def test_the_response_is_periodic_in_the_reciprocal_lattice():
    """``X_0(q + G)(G_a, G_b) = X_0(q)(G_a + G, G_b + G)``, exactly.

    Nothing about this is approximate: it is the same set of transitions read
    with the matrix's G labels shifted, and it is the whole content of the
    umklapp that brings ``k + q`` back into the grid. Without that shift a
    dispersion is smooth, positive and wrong wherever ``k + q`` leaves the
    first zone.
    """
    base = np.array([0.0, 0.0, 0.25])
    shift = np.array([0.0, 0.0, 1.0])
    _, here = _response("h-fcc-magnon.in", 12, 12.0, base)
    _, there = _response("h-fcc-magnon.in", 12, 12.0, base + shift)

    inner = np.asarray(here.sphere.miller)
    outer = np.asarray(there.sphere.miller)
    position = {tuple(vector): n for n, vector in enumerate(inner)}
    pairs = [
        (n, position[tuple(vector + shift.astype(int))])
        for n, vector in enumerate(outer)
        if tuple(vector + shift.astype(int)) in position
    ]
    assert len(pairs) > 20

    rows = np.array([pair[0] for pair in pairs])
    columns = np.array([pair[1] for pair in pairs])
    moved = np.asarray(there.x)[:, rows][:, :, rows]
    original = np.asarray(here.x)[:, columns][:, :, columns]
    assert np.abs(moved - original).max() < 1.0e-14 * np.abs(original).max() + 1e-18


# --- the kernel is what makes the mode ----------------------------------------


def test_without_the_kernel_there_is_no_collective_mode():
    """The Kohn-Sham response has weight only in the Stoner continuum.

    Independent spin flips cost at least the exchange splitting, so
    ``-Im X_0/pi`` is zero below it to the broadening; the collective mode at
    lower energy exists only once ``F`` is turned on. That is the difference
    between a magnon and a particle-hole pair, and it is one keyword apart.
    """
    calculator, scf = _converged("h-fcc-magnon.in")
    frequencies = np.linspace(0.0, 0.02, 9)
    bare = calculator.get_spin_susceptibility(
        (0.0, 0.0, 0.0), frequencies, nbnd=30, ecut_response=12.0,
        broadening=1.0e-3, interacting=False,
    )
    dressed = calculator.get_spin_susceptibility(
        (0.0, 0.0, 0.0), frequencies, nbnd=30, ecut_response=12.0,
        broadening=1.0e-3,
    )
    assert np.abs(bare.spectral_function).max() < 1.0e-3
    assert np.abs(dressed.chi).max() > 20 * np.abs(bare.chi).max()


# --- symmetry of the response ---------------------------------------------------


def test_the_response_is_the_same_at_q_and_minus_q():
    """``X_0(-q)(-G, -G') = X_0(q)(G, G')`` on a centrosymmetric crystal.

    A collinear magnet's Hamiltonian is real in each spin channel, so
    ``psi_{n,-k} = conj psi_{nk}``, and the pair ``(n k up, m k+q dn)`` maps
    onto ``(n -k up, m -k-q dn)`` with its matrix element reflected through the
    origin. The two wavevectors fold onto the grid with **different** umklapp
    vectors, so this exercises the shift on a second and independent pattern --
    where ``q -> q + G`` moves every k-point's umklapp by the same amount.

    **The tolerance is loose and the reason is not this identity.** The states
    at ``k`` and at ``-k`` are produced by two independent runs of the
    eigensolver, so what they agree to is its convergence and the arbitrary
    mixing it leaves inside a partly occupied degenerate multiplet -- 3e-6
    here, against 1e-17 for the ``q -> q + G`` relation, which reads the *same*
    states twice.
    """
    q = np.array([0.0, 0.25, 0.25])
    _, plus = _response("h-fcc-magnon.in", 12, 8.0, q)
    _, minus = _response("h-fcc-magnon.in", 12, 8.0, -q)

    miller = np.asarray(plus.sphere.miller)
    position = {tuple(vector): n for n, vector in enumerate(miller)}
    reflected = np.array([position[tuple(-vector)] for vector in miller])

    here = np.asarray(plus.x)
    there = np.asarray(minus.x)[:, reflected][:, :, reflected]
    assert np.abs(here - there).max() < 1e-4 * np.abs(here).max()


def test_the_stoner_continuum_starts_where_the_bands_say():
    """The Kohn-Sham response has no weight below the smallest spin-flip cost.

    ``X_0`` is a sum of poles at ``eps_dn(k+q) - eps_up(k)`` over pairs whose
    occupations differ, so the lowest of those energies is where its spectral
    weight can begin -- a statement relating the assembled matrix to the
    eigenvalues that went into it, with nothing fitted between them.

    What is read is the ``G = G' = 0`` element, which is the spectral function.
    The matrix as a whole is **Hermitian rather than real** below the onset --
    its off-diagonal entries carry the phase of the pair densities -- so
    checking every element's imaginary part would fail on a fact about the
    basis rather than about the physics.
    """
    calculation, _, eigenvalues, wavefunctions = _states("h-fcc-magnon.in", 12)
    eigenvalues = np.asarray(eigenvalues)
    weights, _ = calculation.occupations(jnp.asarray(eigenvalues))
    weights = np.asarray(weights)

    # The pairs that carry weight: majority occupied, minority not.
    occupied = weights[0] > _OCCUPIED
    empty = weights[1] < _OCCUPIED
    gaps = (eigenvalues[1][:, None, :] - eigenvalues[0][:, :, None])
    allowed = occupied[:, :, None] & empty[:, None, :]
    onset = float(gaps[allowed].min())
    assert onset > 0.0

    frequencies = np.linspace(0.0, 0.9 * onset, 12)
    _, chi = _response("h-fcc-magnon.in", 12, 8.0, (0.0, 0.0, 0.5),
                       frequencies=frequencies)
    # At eta = 0 a sum of poles above ``onset`` has no spectral weight below it.
    spectral = -np.imag(np.asarray(chi.x)[:, 0, 0]) / np.pi
    assert np.abs(spectral).max() < 1e-12


#: An occupation this far from zero counts as occupied, matching the tolerance
#: the sum itself drops pairs at.
_OCCUPIED = 1.0e-8


# --- nickel -------------------------------------------------------------------


def test_nickel_carries_a_magnon_below_its_stoner_continuum():
    """A real ferromagnet, and the one norm-conserving 3d dataset here.

    What is asserted is what this k-grid supports: the Goldstone residual is
    small enough for the enhancement to be near one, and the mode is found by
    the eigenvalue crossing rather than as a peak of a broadened spectrum. The
    energy itself is not a converged spin-wave stiffness and is not compared
    with one -- see the input file.
    """
    calculator, scf = _converged("ni-fcc-magnon.in")
    calculation, chi = _response("ni-fcc-magnon.in", 30, 60.0, (0.0, 0.0, 0.0))
    residual = goldstone_residual(chi, calculation, jnp.asarray(scf.density))
    assert residual < 0.05

    kernel = np.asarray(
        transverse_kernel_matrix(calculation, jnp.asarray(scf.density), chi.sphere)
    )
    eigenvalue = np.linalg.eigvals(np.asarray(chi.x[0]) @ kernel).real.max()
    assert eigenvalue == pytest.approx(1.0, abs=0.02)


# --- the k-set a caller builds itself -------------------------------------------


def test_a_caller_built_k_set_carries_one_electron_per_band():
    """``for_spin`` at the boundary, which both magnon entry points skipped.

    Every ``KPoints`` constructor applies the unpolarized ``degspin``
    unconditionally, since a set can be built long before it is known which
    regime will use it, and a transverse susceptibility is by construction an
    ``nspin = 2`` run where a band holds **one** electron. So a mesh from
    ``KPoints.automatic`` arrived with weights summing to 2 where this needs 1,
    and every k-integral in ``chi_0`` was wrong -- with nothing looking like an
    error, because the electron count is still met and the Fermi level simply
    lands elsewhere.

    Measured on this cell at ``nbnd = 8``: the Goldstone residual read
    **0.8486** and the enhancement **0.1908**, against **0.0274** and
    **1.0587** once the weights are right. Note it is *not* the clean factor of
    two the mechanism suggests: h-fcc is a smeared metal, so the doubled
    weights move the Fermi solve as well as the prefactor, and the enhancement
    is out by 5.5 rather than by 2. The identity that says which is honest is
    the enhancement itself -- a Goldstone mode at ``q = 0`` is the statement
    that ``1 - X_0 F`` is singular there, so it must be 1.

    The assertion is the exact one rather than a tolerance, because the raw
    ``4 4 4`` set **is** this input's own set: the two routes must give the
    same array bit for bit, and they do.
    """
    from defumat.system.kpoints import KPoints
    from defumat.workflows.magnons import run_spin_susceptibility

    calculator, scf = _converged("h-fcc-magnon.in")
    options = dict(nbnd=8, conv_thr=1.0e-9)
    args = (calculator.system, calculator.pseudos, jnp.asarray(scf.density),
            np.zeros(3), np.array([0.0]))

    own = run_spin_susceptibility(*args, **options)
    raw = KPoints.automatic((4, 4, 4), (0, 0, 0), calculator.system.cell,
                            rotations=(), time_reversal=False)
    assert float(np.asarray(raw.weights).sum()) == pytest.approx(2.0)
    given = run_spin_susceptibility(*args, kpoints=raw, **options)

    np.testing.assert_array_equal(np.asarray(given.chi0), np.asarray(own.chi0))
    assert given.enhancement == pytest.approx(own.enhancement, rel=1e-12)
    assert given.goldstone == pytest.approx(own.goldstone, rel=1e-12)
    assert given.goldstone < 0.05
