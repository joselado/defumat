"""P52: the Fermi-surface nesting function, Elk's task 105.

The fifth entry taken from ``ELK-FEATURES.md``, and the one whose validation
closes entirely inside the package -- which is what that file says to look for.
There is no ``pw.x`` reference at all (``nesting`` occurs nowhere in ``PW/src``
or ``PP/src``), and Elk's own number is an all-electron Fermi surface, so it is
a sanity check on the *shape* rather than a target: what it can be held to is
recorded in ``PLAN.md`` P52 and in ``PERFORMANCE.md``.

Five statements, each failing differently:

* the **unfold** -- a symmetry-reduced wedge unfolded onto the complete grid
  must give the same ``N(q)`` as diagonalising all of it. This is the one that
  catches a group mismatch, which is otherwise silent;
* the **weights are a density of states** -- ``(1/N_k) sum_k g(k)`` against
  :func:`~pypresso.workflows.dos.compute_dos` on the wedge, which reaches the
  same number through k-point weights instead of an unfold;
* the **two routes** -- the FFT correlation against ``nesting.f90``'s double
  loop, on a real band structure rather than on random numbers;
* the **hydrogen chain nests at the spiral's wavevector**. Half filling puts
  ``2 k_F`` at exactly ``q = 0.5``, and P21's ``relax_spiral_q`` goes downhill
  in the magnetic energy to 0.500014 from an unrelated starting point. Neither
  can make that check alone, and they share nothing: one is a Fermi-surface
  geometry on the paramagnet, the other a gradient of the total energy of a
  magnet;
* **aluminium does not nest**, which is the contrast that makes the chain's
  peak mean something. A nearly-free-electron Fermi sphere has ``N(q) ~ 1/q``
  and no structure at all;
* and **a cell with no magnetization gives the same answer in all three spin
  regimes**, which is P45's check on the same kind of object. ``g(k)`` carries
  the spin degeneracy and ``N`` is *quadratic* in it, so a factor of two
  between the unpolarized and the per-channel convention is a factor of four
  here -- and it would be invisible in the shape of ``N(q)``, which is what
  every other test in this file looks at.
"""

from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pypresso.io.pwin import parse_pw_input, read_pw_input
from pypresso.pseudo import read_upf
from pypresso.scf import run_scf
from pypresso.system import build_system
from pypresso.system.kpoints import grid_equivalence
from pypresso.workflows.dos import compute_dos
from pypresso.workflows.nesting import run_nesting
from pypresso.workflows.nscf import denser_grid, grid_symmetry, run_nscf

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


@pytest.fixture(autouse=True)
def _bounded_compilation():
    """``CLAUDE.md``'s memory rule: drop XLA's executables between cases.

    Aluminium and the hydrogen chain share no shape, and each grid size here is
    a fresh compilation of the whole NSCF stack.
    """
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _converged(case: str):
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    result = run_scf(system, pseudos, conv_thr=1e-10, max_iterations=200)
    return system, pseudos, result


# -- the unfold ----------------------------------------------------------------


def test_a_wedge_and_the_complete_grid_give_the_same_nesting():
    """72 diagonalisations against 1728, and they must agree exactly.

    ``eps_n(Rk) = eps_n(k)``, so this has no physical tolerance -- what is left
    is the eigensolver's own scatter between two runs that solve the same
    Hamiltonian at points reached in a different order. It is the certifying
    test for :func:`~pypresso.workflows.nscf.grid_symmetry`: the group the
    wedge was *reduced* with and the group it is *unfolded* with are two
    different questions on a ``nosym``, ``noinv`` or magnetic run, and a
    mismatch maps every point to the wrong representative while producing a
    smooth, positive, entirely plausible ``N(q)``.
    """
    system, pseudos, result = _converged("al-metal")
    grid = (8, 8, 8)
    wedge = run_nesting(system, pseudos, result.density, grid=grid)
    whole = run_nesting(system, pseudos, result.density, grid=grid, reduce=False)

    assert wedge.fermi_dos == pytest.approx(whole.fermi_dos, rel=1e-10)
    scale = whole.nesting.max()
    assert np.max(np.abs(wedge.nesting - whole.nesting)) < 1e-10 * scale


# -- the weights are a density of states ---------------------------------------


def test_the_fermi_surface_weights_integrate_to_the_density_of_states():
    """``(1/N_k) sum_k g(k)`` against ``compute_dos`` at ``E_F``.

    Two routes to one number that share nothing but the eigenvalues: the DOS
    sums over the *wedge* with its multiplicities and this sums over the
    complete grid with uniform weight. It is what pins the spin degeneracy --
    ``g`` carries ``degspin`` so that ``N`` is in (states/Ry/cell)^2, and a
    factor of two there is a factor of four in the nesting function.
    """
    system, pseudos, result = _converged("al-metal")
    grid = (8, 8, 8)
    nesting = run_nesting(system, pseudos, result.density, grid=grid)

    kpoints = denser_grid(system, grid)
    nscf = run_nscf(system, pseudos, result.density, kpoints, conv_thr=1e-8)
    dos = compute_dos(
        nscf.eigenvalues, kpoints.weights,
        jnp.asarray([nesting.fermi_energy]), "gaussian", degauss=nesting.degauss,
    )
    assert nesting.fermi_dos == pytest.approx(
        float(np.asarray(dos.dos).ravel()[0]), rel=1e-9
    )


# -- the two routes ------------------------------------------------------------


def test_the_transform_and_the_double_loop_agree_on_a_real_band_structure():
    """The check on the index fold, run on aluminium rather than on noise.

    A random ``g`` exercises the arithmetic; a real one exercises it on a
    weight distribution that is concentrated on a surface, which is where a
    wrap-around error would show.
    """
    system, pseudos, result = _converged("al-metal")
    grid = (8, 8, 8)
    fast = run_nesting(system, pseudos, result.density, grid=grid)
    slow = run_nesting(system, pseudos, result.density, grid=grid, method="direct")
    assert np.max(np.abs(fast.nesting - slow.nesting)) < 1e-12 * fast.nesting.max()


# -- the hydrogen chain, which is where the physics is -------------------------


def test_the_hydrogen_chain_nests_at_the_spirals_wavevector():
    """``2 k_F = pi/c``, so ``q = 0.5`` -- and P21 relaxes the spiral to 0.500014.

    One electron per cell in a spin-degenerate band is half filling, so the
    Fermi surface of this chain is the two points ``+-pi/2c`` and the only
    wavevector connecting them is ``q = 0.5`` in crystal coordinates. That is
    analytic and needs no reference.

    What makes it worth running is that **the same number is reached by a route
    that shares nothing with it**: ``relax_spiral_q`` writes the total energy of
    a *magnet* as a function of the spiral wavevector, takes ``jax.grad`` of it
    at frozen wavefunctions, and walks downhill from ``q = 0.30`` to 0.500014
    (P21 records 0.50003 at the input's own cutoff; the difference is the
    basis-set jump in ``E(q)`` that phase measures, not a disagreement).
    Nesting predicts the pitch from the paramagnet's Fermi surface geometry;
    the relaxation finds it in the magnetic energy. Agreement is the check
    neither can make alone, and it is the pairing ``ELK-FEATURES.md`` singles
    out for this entry.

    The peak is also **99.8 per cent of the Cauchy-Schwarz bound** ``N(0)``,
    which is what perfect one-dimensional nesting means: the whole Fermi
    surface maps onto itself under one translation. Aluminium's, below, reaches
    a few per cent of its own bound.
    """
    system, pseudos, result = _converged("h-chain-nesting")
    nesting = run_nesting(system, pseudos, result.density, grid=(1, 1, 60))

    where, value = nesting.peak()
    assert np.allclose(where, [0.0, 0.0, 0.5]), where
    assert value / nesting.nesting[0] > 0.99

    # The peak is a *peak*: an order of magnitude above the flat part between
    # the two Fermi points, which is what says it is nesting and not the
    # ``1/q`` tail every metal has.
    _, along = nesting.along(2)
    assert value > 10.0 * along[len(along) // 4]


def test_aluminium_does_not_nest():
    """A free-electron sphere has ``N(q) ~ 1/q`` and no nesting peak.

    The control that gives the chain's 0.998 its meaning. Aluminium's largest
    ``N(q)`` away from the origin is a small fraction of ``N(0)``, and it sits
    at the *smallest* wavevector on the grid -- which is the ``1/q`` tail, not
    a feature of the surface.
    """
    system, pseudos, result = _converged("al-metal")
    nesting = run_nesting(system, pseudos, result.density, grid=(12, 12, 12))

    where, value = nesting.peak()
    assert value / nesting.nesting[0] < 0.6
    # The peak is one grid step from the origin: the tail, not a nesting vector.
    assert np.isclose(np.linalg.norm(where - np.rint(where)), 1.0 / 12.0)


# -- the three spin regimes, on a cell that has no magnetization ---------------


def test_a_cell_with_no_magnetization_nests_the_same_in_every_spin_regime():
    """``nspin`` 1, 2 and 4 on the same chain, to 2.4e-13 and 1.8e-8.

    The check that catches a factor of two, and it has to be run because
    nothing refuses a polarized nesting function: ``g(k)`` carries ``degspin``
    so that ``(1/N_k) sum_k g`` is a density of states, and that factor is 2
    for an unpolarized band and 1 for one LSDA channel or one spinor. Get it
    wrong and ``N`` is four times out with an identical shape -- the peak still
    lands at ``q = 0.5``, the sum rule still closes against its own wrong
    ``D(E_F)``, and every other test here still passes.

    It exercises three more things at once. The k-weights, which
    ``for_spin`` halves for a polarized run and which reach the delta through
    the unfold rather than through a weighted sum; the **two-channel** ``g``,
    since one Fermi level is shared between the channels and the surface is
    their union; and the ``t_rev`` branch of the unfold map.

    The spinor agreement is 1.8e-8 rather than 1e-13 because its Hamiltonian
    acts on a doubled space and its eigenvalues converge to their own
    threshold; the physics is identical, the arithmetic is not the same
    arithmetic.
    """
    system, pseudos, unpolarized = _converged("h-chain-nesting")
    reference = run_nesting(system, pseudos, unpolarized.density, grid=(1, 1, 40))

    text = (CASES / "h-chain-nesting.in").read_text()
    # **The band count is held at the reference's** for the collinear case and
    # doubled for the spinor, because that is what each regime needs -- and
    # because a *different* band count is a different SCF trajectory, which on
    # a one-dimensional metal moves the Fermi level by 2e-5 Ry at
    # ``conv_thr = 1e-10`` and swamps the factor this test is looking for.
    for label, bands, extra, tolerance in (
        ("collinear", 6, "nspin = 2, starting_magnetization(1) = 0.0", 1e-11),
        ("spinor", 12, "noncolin = .true., starting_magnetization(1) = 0.0", 1e-6),
    ):
        spun = build_system(parse_pw_input(
            text.replace("nbnd = 6", f"nbnd = {bands}\n    {extra}")
        ))
        result = run_scf(spun, pseudos, conv_thr=1e-10, max_iterations=200)
        nesting = run_nesting(spun, pseudos, result.density, grid=(1, 1, 40))

        scale = reference.nesting.max()
        assert np.max(np.abs(nesting.nesting - reference.nesting)) < tolerance * scale, label
        assert nesting.fermi_dos == pytest.approx(reference.fermi_dos, rel=tolerance)
        jax.clear_caches()
