"""EP1 and EP2: the metallic phonon at ``q != 0`` and its coupling to the Fermi surface.

Norm-conserving LDA aluminium, Marzari-Vanderbilt at ``degauss = 0.05`` Ry on the
8x8x8 unshifted grid without symmetry (``al-elph-nosym.in``), against serial
``ph.x`` 7.5 with ``electron_phonon = 'simple'`` on the same input
(``reference.out.ph-al-elph-nosym`` at q = (1/4, 0, 0) and ``-q2`` at
q = (3/4, 1/4, 1/4), both in 2 pi/alat). Not against ``test-suite/ph_interpol_metal``'s
own benchmark: that one was written by QE 6.5 and its frequencies at
q = (1/4, 0, 0) are 2.5 cm^-1 above 7.5's from an identical ground state
(``al-elph-nosym.ph.in`` has the numbers).

The checks are ordered by how much machinery each one touches:

* the **frequencies**, which need the metallic two-sphere projector, the ``wk``
  contraction and the per-mode mixing (``PLAN.md`` EP1);
* ``E_F``, ``N(E_F)`` and the **double delta** at each broadening, which are
  functions of the eigenvalues at ``k`` and at ``k + q`` alone;
* ``gamma`` and ``lambda`` at each broadening, which add the matrix elements;
* the **folding identity**: one atom at q = (0, 0, 1/2) crystal on a 4x4x4 grid
  is the two-atom cell of ``al2-metal.in`` at ``Gamma``, which ``ph.x`` computed
  by a route that shares no machinery with this one.

Every run here states its own convergence: the ground state at ``conv_thr =
1e-12`` (the input's) and the response at ``tr2 = 1e-14``, ``ph.x``'s
``tr2_ph`` in the reference input.
"""

from functools import lru_cache
from pathlib import Path
import re

import jax
import numpy as np
import pytest

from defumat import Calculator

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: In cm^-1. Measured 2026-10-05: 3.8e-4 at q = (1/4, 0, 0) (73.9960, 73.9961,
#: 132.4755 against 73.9956, 73.9957, 132.4752) and 3.7e-3 at (3/4, 1/4, 1/4);
#: with a metal's solve cut at each k-point's own band count and one loop per
#: mode, 6.0e-4 (73.9961, 73.9963, 132.4750) and 3.9e-3 (179.8494, 223.9916,
#: 292.9979 against 179.8509, 223.9919, 293.0018), on D22.
FREQUENCY_TOLERANCE = 6.0e-3
#: ``ph.x`` prints ``gamma`` with two decimals in GHz and ``lambda`` with four,
#: so 5e-3 and 5e-5 of each bound is its rounding. Measured over both q and all
#: ten broadenings: 7.0e-3 GHz (on 18.48 at q2) and 5.0e-5.
GAMMA_TOLERANCE_GHZ = 1.0e-2
LAMBDA_TOLERANCE = 6.0e-5
#: Printed as f10.6.
EIGENVALUE_ONLY_TOLERANCE = 2.0e-6

RY_TO_EV = 13.605693122994

REFERENCES = {
    "q1": ((0.25, 0.0, 0.0), "reference.out.ph-al-elph-nosym"),
    "q2": ((0.75, 0.25, 0.25), "reference.out.ph-al-elph-nosym-q2"),
}


@pytest.fixture(autouse=True)
def _bounded_compilation_cache():
    """Each ``q`` is a second sphere and its own shapes (``CLAUDE.md``)."""
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _calculator(case: str) -> Calculator:
    calc = Calculator.from_file(CASES / f"{case}.in", pseudo_dir=PSEUDO)
    calc.get_scf()
    return calc


@lru_cache(maxsize=2)
def _coupling(point: str):
    q, _ = REFERENCES[point]
    return _calculator("al-elph-nosym").get_electron_phonon(
        q, q_cartesian=True, tr2=1.0e-14)


def _read_reference(name: str) -> dict:
    """``ph.x``'s frequencies and its ``elphsum_simple`` block, per broadening."""
    text = (CASES / name).read_text()
    frequencies = [float(x) for x in re.findall(
        r"freq \(\s*\d+\) =\s*[-\d.]+ \[THz\] =\s*([-\d.]+) \[cm-1\]", text)]
    blocks = re.findall(
        r"Gaussian Broadening:\s*([\d.]+) Ry.*?\n\s*DOS =\s*([\d.]+) states/spin/Ry/Unit Cell"
        r" at Ef=\s*([-\d.]+) eV\n\s*double delta at Ef =\s*([\d.]+)\n"
        r"((?:\s*lambda\(\s*\d+\)=\s*[-\d.]+\s+gamma=\s*[-\d.]+ GHz\n)+)", text)
    sigmas, dos, ef, phase, lambdas, gammas = [], [], [], [], [], []
    for sigma, d, e, p, lines in blocks:
        sigmas.append(float(sigma))
        dos.append(float(d))
        ef.append(float(e))
        phase.append(float(p))
        pairs = re.findall(r"lambda\(\s*\d+\)=\s*([-\d.]+)\s+gamma=\s*([-\d.]+)", lines)
        lambdas.append([float(a) for a, _ in pairs])
        gammas.append([float(b) for _, b in pairs])
    return {"frequencies": np.array(frequencies[:3]), "sigmas": np.array(sigmas),
            "dos": np.array(dos), "ef_ev": np.array(ef), "phase": np.array(phase),
            "lambda": np.array(lambdas), "gamma_ghz": np.array(gammas)}


@pytest.mark.parametrize("point", ["q1", "q2"])
def test_the_metallic_phonon_at_q_matches_ph_x(point):
    """The three frequencies at ``q``: the transverse pair and the longitudinal
    mode at (1/4, 0, 0), three distinct modes at (3/4, 1/4, 1/4)."""
    coupling = _coupling(point)
    reference = _read_reference(REFERENCES[point][1])
    assert coupling.phonons.converged
    assert np.allclose(coupling.frequencies, reference["frequencies"],
                       atol=FREQUENCY_TOLERANCE), (
        coupling.frequencies, reference["frequencies"])


@pytest.mark.parametrize("point", ["q1", "q2"])
def test_the_fermi_surface_sums_match_ph_x(point):
    """``E_F``, ``N(E_F)`` and the double delta at each of the ten broadenings.

    Eigenvalues only: the Fermi level recomputed at each width with
    Methfessel-Paxton, the density of states halved to per spin, and
    ``sum_k w_k sum_mn delta(e_nk) delta(e_(m k+q))`` with Gaussian deltas.
    """
    coupling = _coupling(point)
    reference = _read_reference(REFERENCES[point][1])
    assert np.allclose(coupling.sigmas, reference["sigmas"])
    assert np.allclose(coupling.fermi_energies * RY_TO_EV, reference["ef_ev"],
                       atol=EIGENVALUE_ONLY_TOLERANCE)
    assert np.allclose(coupling.dos, reference["dos"],
                       atol=EIGENVALUE_ONLY_TOLERANCE)
    assert np.allclose(coupling.phase_space, reference["phase"],
                       atol=10 * EIGENVALUE_ONLY_TOLERANCE)


@pytest.mark.parametrize("point", ["q1", "q2"])
def test_the_linewidths_and_lambda_match_ph_x(point):
    """``gamma_(q nu)`` in GHz and ``lambda_(q nu)`` at every broadening.

    The matrix elements enter here and nowhere above: the bare perturbation and
    the converged ``dV_scf`` between ``psi_k`` and ``psi_(k+q)``, contracted with
    the mode displacements.
    """
    coupling = _coupling(point)
    reference = _read_reference(REFERENCES[point][1])
    assert np.allclose(coupling.gamma_ghz, reference["gamma_ghz"],
                       atol=GAMMA_TOLERANCE_GHZ), (
        coupling.gamma_ghz, reference["gamma_ghz"])
    assert np.allclose(coupling.lambdas, reference["lambda"],
                       atol=LAMBDA_TOLERANCE), (coupling.lambdas, reference["lambda"])


#: In cm^-1 at the off-grid point, against ``ph.x``'s 41.2796 on the middle mode.
#: Measured 2026-10-05: 41.2870 on D22 and 41.2972 on the workstation, the same
#: code with the scheduled CG thresholds landing on different round-off, and
#: 41.2941 with the CG held at 1e-10; so ``ph.x``'s own scheduled residue is
#: 1.5e-2 here and ours scatters by 1e-2 around it, on a soft mode at 41 cm^-1.
OFFGRID_FREQUENCY_TOLERANCE = 3.0e-2


def test_a_wavevector_off_the_grid_matches_ph_x():
    """q = (0.1, 0.05, 0) on ``al-metal-nosym.in``'s 4x4x4 grid, against ``ph.x``.

    ``k + q`` lands on no grid point, so the second sphere pads to another width
    than the first and every perturbation applied to ``psi_k`` changes the length
    of a band on its way to ``k + q``. Every ``q`` checked before was on the grid,
    where the two widths happen to agree.
    """
    coupling = _calculator("al-metal-nosym").get_electron_phonon(
        (0.1, 0.05, 0.0), q_cartesian=True, tr2=1.0e-14)
    reference = _read_reference("reference.out.ph-al-metal-nosym-offgrid")
    assert coupling.phonons.converged
    assert np.allclose(coupling.frequencies, reference["frequencies"],
                       atol=OFFGRID_FREQUENCY_TOLERANCE), (
        coupling.frequencies, reference["frequencies"])
    assert np.allclose(coupling.dos, reference["dos"], atol=EIGENVALUE_ONLY_TOLERANCE)
    assert np.allclose(coupling.gamma_ghz, reference["gamma_ghz"],
                       atol=GAMMA_TOLERANCE_GHZ), (
        coupling.gamma_ghz, reference["gamma_ghz"])
    # ``lambda`` goes as ``gamma / omega^2``, so the frequency's relative residue
    # above enters it twice: measured 1.8e-3 relative (8.2e-4 on 0.45) on the
    # workstation, where ``gamma`` itself agrees to 6.8e-3 GHz.
    assert np.allclose(coupling.lambdas, reference["lambda"], rtol=3.0e-3,
                       atol=LAMBDA_TOLERANCE), (coupling.lambdas, reference["lambda"])


#: ``reference.out.ph-al2-metal``'s three optical modes at ``Gamma``: the
#: two-atom cell's zone-boundary modes folded to the zone centre.
FOLDED = (146.710511, 146.714378, 311.035401)
#: In cm^-1. Measured 2026-10-05: 0.014 with the atom at the origin, 0.015 at
#: (1/3, 2/3, 0) crystal, which is the same frequencies after a rigid shift.
FOLDING_TOLERANCE = 0.03


def test_one_atom_at_half_b3_is_two_atoms_at_gamma():
    """``D(q)`` of one atom at ``q = b_3 / 2`` against ``ph.x`` on the doubled cell.

    ``al2-metal.in`` is this cell doubled along ``a_3``, and its 4x4x2 grid
    unfolds to this 4x4x4. A mode of the supercell at ``Gamma`` with the two atoms
    in antiphase is the primitive cell's mode at ``b_3 / 2``, so the two routes
    compute the same three frequencies: one on two spheres at ``k`` and
    ``k + q``, the other on one sphere at ``Gamma`` with an ``ef_shift``.
    """
    calc = _calculator("al-metal-nosym")
    phonons = calc.get_phonons_at_q((0.0, 0.0, 0.5), tr2=1.0e-14)
    assert phonons.converged
    assert np.allclose(np.sort(phonons.frequencies), FOLDED,
                       atol=FOLDING_TOLERANCE), phonons.frequencies


def test_a_metal_at_q_zero_is_refused_by_name():
    """At ``q = 0`` the Fermi level moves and this route has no ``ef_shift``."""
    calc = _calculator("al-metal-nosym")
    with pytest.raises(NotImplementedError, match="ef_shift"):
        calc.get_phonons_at_q((0.0, 0.0, 0.0))
    with pytest.raises(NotImplementedError, match="ef_shift"):
        calc.get_phonons_at_q((1.0, 0.0, 0.0))


#: Streamed against whole on one converged state, both loops 13 iterations
#: with identical histories to the printed digits. Measured 2026-10-05 at
#: q = (1/4, 0, 0) on this cell: frequencies 1.7e-8 cm^-1, gamma 1.4e-10 GHz,
#: lambda 1.2e-10, el_ph_sum 6.6e-13, the double delta identical, D(q) 9.3e-12;
#: on the 4x4x4 cell at q = b_3/2, 1.2e-12 cm^-1 and 6.4e-14 GHz.
STREAMED_FREQUENCY_TOLERANCE = 1.0e-6
STREAMED_GAMMA_TOLERANCE_GHZ = 1.0e-8
STREAMED_LAMBDA_TOLERANCE = 1.0e-8


def test_the_k_chunked_coupling_is_the_whole_k_coupling():
    """The route an accelerator takes in ``memory_mode = 'memory'``, on the CPU.

    The ground state handed in as a host store makes ``dynamical_matrix_at_q``
    walk the k axis a chunk at a time (one k-point a chunk here), with the
    metal's smeared two-sphere projector inside each chunk's solve and the
    matrix elements walked the same way. Only the invariants are compared: a
    per-band ``g_mn`` is free in the rotation inside a degenerate multiplet at
    ``k`` or ``k + q`` (rule D4), and the two routes diagonalise ``k + q`` with
    different compiled programs.
    """
    from defumat.response.elph import electron_phonon_at_q

    q, name = REFERENCES["q1"]
    calc = _calculator("al-elph-nosym")
    result = calc.get_scf()
    streamed = electron_phonon_at_q(
        calc.calculation, np.asarray(result.wavefunctions), result.eigenvalues,
        result.density, result.becsum, q=q, q_cartesian=True, tr2=1.0e-14)
    whole = _coupling("q1")
    reference = _read_reference(name)
    assert streamed.phonons.converged
    assert np.allclose(streamed.frequencies, whole.frequencies,
                       rtol=0, atol=STREAMED_FREQUENCY_TOLERANCE)
    assert np.allclose(streamed.gamma_ghz, whole.gamma_ghz,
                       rtol=0, atol=STREAMED_GAMMA_TOLERANCE_GHZ)
    assert np.allclose(streamed.lambdas, whole.lambdas,
                       rtol=0, atol=STREAMED_LAMBDA_TOLERANCE)
    assert np.allclose(streamed.el_ph_sum, whole.el_ph_sum, rtol=0, atol=1.0e-10)
    assert np.allclose(streamed.phase_space, whole.phase_space, rtol=0, atol=1.0e-10)
    assert np.allclose(streamed.frequencies, reference["frequencies"],
                       atol=FREQUENCY_TOLERANCE)
    assert np.allclose(streamed.gamma_ghz, reference["gamma_ghz"],
                       atol=GAMMA_TOLERANCE_GHZ)
    assert np.allclose(streamed.lambdas, reference["lambda"], atol=LAMBDA_TOLERANCE)
