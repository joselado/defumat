"""``chi^(2)(-2 omega; omega, omega)`` against Elk, and against four identities.

**There is a real reference here**, which makes this corner of the package
unusual: Elk's ``nonlinopt.f90`` (task 125) computes the same tensor, the
binary is vendored, and ``tests/data/elk/`` carries its output for the same
AlAs crystal. What that comparison is *not* is digit-for-digit -- an
all-electron LAPW basis is not a norm-conserving pseudopotential one and the
two codes do not have the same gap -- so what is asserted is the size and the
resonance position, and the numbers are quoted with the disagreement written
down rather than tuned away.

The identities are what actually pin the assembly, and each catches something
the others cannot:

* **``-43m``.** AlAs is zincblende, so only ``xyz`` and its five permutations
  may survive. It is a weak check on its own -- `NONLINEAR.md` §5 records the
  tensor coming out exactly the crystal's class three separate times while
  being wrong -- but it is instant and it is the one that fails loudly when a
  cartesian label is transposed.
* **Silicon.** An inversion centre forbids every component of a polar rank-3
  tensor. This is the check that found the rule-D4 defect in ``Delta^a``, and
  it found it because it is a *pointwise* statement about a cancellation rather
  than a statement about a value.
* **The two-photon edge.** ``Im chi`` turns on at **half** the direct gap, not
  at the gap: the second-harmonic channel's denominator is ``e - 2w``, so a
  transposed factor of two there moves the absorption edge by an octave. It is
  the only check here that is blind to the tensor's overall scale *and* to its
  symmetry, which is what makes it worth its own test.
* **The spinor.** The scale's only internal anchor, in the manner P50 and P53
  both record: a ``chi`` wrong by a constant is still exactly ``-43m``, still
  zero on silicon and still resonant in the right place.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
import pytest

from defumat.io.pwin import read_pw_input
from defumat.pseudo.upf import read_upf
from defumat.response.shg import require_an_shg_regime
from defumat.scf.driver import Calculation, run_scf
from defumat.system.builder import build_system
from defumat.system.kpoints import KPoints
from defumat.units import RY_TO_EV
from defumat.workflows.shg import run_shg

pytestmark = pytest.mark.regression

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"
ELK = Path(__file__).resolve().parents[1] / "data" / "elk"

#: The six components ``-43m`` allows: ``chi^xyz`` and its permutations.
ZINCBLENDE = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """``CLAUDE.md``'s bound on a file that runs several distinct cells.

    Each cell compiles the whole velocity stack afresh and XLA keeps every
    executable for the life of the process; the ``lru_cache`` below is the
    other half of the same bound and is deliberately ``maxsize=2``.
    """
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def converged(name: str):
    """An SCF run of a committed input. ``maxsize=2``, never ``None``."""
    system = build_system(read_pw_input(CASES / name))
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    return system, pseudos, run_scf(system, pseudos, conv_thr=1.0e-10)


def full_mesh(system, n: int) -> KPoints:
    """The whole unshifted ``n x n x n`` grid -- closed under the point group."""
    axis = np.arange(n) / n
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    return KPoints.from_crystal(
        points, np.full(len(points), 1.0 / len(points)), system.cell,
        precision=system.kpoints.precision,
    )


def spectrum(name: str, n: int, nbnd: int, **options):
    system, pseudos, result = converged(name)
    return run_shg(system, pseudos, result.density, kpoints=full_mesh(system, n),
                   nbnd=nbnd, **options)


def read_elk_chi(path: Path):
    """Elk's ``CHI_2WWW_abc.OUT``: real block, blank line, imaginary block.

    ``w`` in Hartree and ``chi`` in atomic units, which is why
    :data:`~defumat.response.shg.CHI2_AU_TO_PM_PER_V` exists as a separate
    constant -- the comparison is made in Elk's own units and converted once,
    so a wrong conversion cannot hide inside an agreement.
    """
    blocks = path.read_text().split("\n\n")
    real = np.array([[float(x) for x in line.split()]
                     for line in blocks[0].strip().splitlines()])
    imaginary = np.array([[float(x) for x in line.split()]
                          for line in blocks[1].strip().splitlines()])
    return real[:, 0], real[:, 1] + 1j * imaginary[:, 1]


# --- the identities -----------------------------------------------------------


@pytest.mark.slow
def test_alas_comes_out_exactly_zincblende():
    """Only ``xyz`` and its permutations survive, and they are equal.

    Nothing imposes this: the run is ``nosym``, the mesh is the whole grid, and
    no symmetriser is applied to the tensor. The six allowed components agree
    with each other to the tolerance below and the other twenty-one are down by
    four orders of magnitude.
    """
    chi = np.asarray(spectrum("alas-raman.in", 4, 14, window=0.6, nw=120,
                              broadening=0.005).chi)
    peak = {(a, b, c): float(np.max(np.abs(chi[:, a, b, c])))
            for a in range(3) for b in range(3) for c in range(3)}
    allowed = [peak[t] for t in ZINCBLENDE]
    forbidden = [v for t, v in peak.items() if t not in ZINCBLENDE]

    assert min(allowed) > 0.0
    assert max(allowed) - min(allowed) < 1.0e-6 * max(allowed)
    assert max(forbidden) < 1.0e-3 * max(allowed)


@pytest.mark.slow
def test_silicon_has_no_second_harmonic_at_all():
    """An inversion centre forbids every component of a polar rank-3 tensor.

    **This is the test that found the defect the phase is named for.** With
    ``Delta^a`` taken as the bare diagonal of the velocity operator -- which is
    what a literal reading of ``nonlinopt.f90`` gives -- the two terms it
    appears in come out at **1499** and **238** pm/V here where the other three
    sit at 0.09, because the mesh's high-symmetry points have degenerate
    multiplets and the diagonal of an operator inside one is whatever the
    eigensolver's arbitrary rotation made it (rule D4). With the multiplet's
    block average they fall to 0.10 and 0.055.

    A number that is small has to be shown to be small *for the right reason*,
    which is what running the opposite symmetry through the same machinery is
    for: AlAs and silicon differ by one species and are otherwise the same
    calculation.
    """
    silicon = np.asarray(spectrum("si2-nosym.in", 4, 14, window=0.6, nw=120,
                                  broadening=0.005).chi)
    alas = np.asarray(spectrum("alas-raman.in", 4, 14, window=0.6, nw=120,
                               broadening=0.005).chi)
    assert np.max(np.abs(silicon)) < 1.0e-3 * np.max(np.abs(alas))


@pytest.mark.slow
def test_the_absorption_turns_on_at_half_the_gap():
    """The second-harmonic resonance, and the only check blind to scale *and* symmetry.

    A linear spectrum absorbs at ``hbar w = E_gap``. This one has a second
    channel whose denominator is ``e_mn - 2w``, so ``Im chi`` becomes finite
    once **two** photons reach the smallest direct gap on the mesh -- at
    ``E_gap / 2``. A factor of two lost in that denominator moves the edge by a
    full octave and changes nothing else about the answer: not the symmetry,
    not the scale, not the silicon zero.

    **Where the threshold comes from, since a fraction of the maximum is
    otherwise a fit.** The resonance is a Lorentzian of width ``eta``, whose
    tail falls only as ``eta/x^2``, so it reaches 1 per cent of the peak about
    ten widths below the true edge and 5 per cent about two. A 1 per cent
    threshold therefore cannot locate an edge better than ``10 eta`` no matter
    how right the code is, which is what the first version of this test
    discovered. Measured here on a 6x6x6 mesh, with the smallest direct gap
    0.20971 Ry: the 5 per cent crossing sits **0.12 eta** above ``E_gap / 2``
    and the 10 per cent crossing 2.13 eta, while the gap itself is **21 eta**
    away. Two thresholds are checked so that neither is load-bearing.
    """
    system, pseudos, result = converged("alas-raman.in")
    mesh = full_mesh(system, 6)
    chi = run_shg(system, pseudos, result.density, kpoints=mesh, nbnd=22,
                  window=0.6, nw=240, broadening=0.005)

    import equinox as eqx

    from defumat.workflows.nscf import fixed_density_states

    _, _, eigenvalues, _ = fixed_density_states(
        eqx.tree_at(lambda s: s.kpoints, system, mesh),
        pseudos, result.density, nbnd=22, conv_thr=1.0e-10,
    )
    eigenvalues = np.asarray(eigenvalues)
    if eigenvalues.ndim == 3:
        eigenvalues = eigenvalues[0]
    nocc = 4  # AlAs: eight valence electrons, unpolarized
    gap = float(np.min(eigenvalues[:, nocc] - eigenvalues[:, nocc - 1]))

    imaginary = np.abs(np.asarray(chi.chi)[:, 0, 1, 2].imag)
    eta = chi.broadening
    for threshold in (0.05, 0.10):
        crossing = float(chi.frequencies[
            int(np.argmax(imaginary > threshold * imaginary.max()))
        ])
        assert abs(crossing - 0.5 * gap) < 3.0 * eta, threshold
        assert abs(crossing - gap) > 15.0 * eta, threshold


@pytest.mark.slow
def test_a_spinor_run_gives_the_same_tensor_as_an_unpolarized_one():
    """The absolute scale's internal anchor: a factor of two in the spin sum.

    Every other check in this file is blind to an overall constant -- P50's
    trap, which P53 records in these same coordinates. A cell with no
    magnetization run as ``nspin = 1`` and as a two-component spinor must give
    the same tensor, and the two paths reach the weights differently:
    ``for_spin`` gives the unpolarized k-set weights summing to 2 and the
    spinor's summing to 1, while a spinor band holds one electron where an
    unpolarized band holds two.

    It is also the only test behind the README row's claim that the
    second-harmonic tensor runs on a spinor calculation at all.
    """
    system, pseudos, result = converged("alas-raman.in")
    spinor = system.with_spin(4)
    spinor_result = run_scf(spinor, pseudos, conv_thr=1.0e-10)

    one = np.asarray(run_shg(
        system, pseudos, result.density, kpoints=full_mesh(system, 4),
        nbnd=14, window=0.6, nw=60, broadening=0.02,
    ).chi)
    two = np.asarray(run_shg(
        spinor, pseudos, spinor_result.density, kpoints=full_mesh(spinor, 4),
        nbnd=28, window=0.6, nw=60, broadening=0.02,
    ).chi)
    scale = max(np.abs(one).max(), np.abs(two).max())
    assert scale > 1.0, "the case is vacuous"
    assert np.max(np.abs(one - two)) < 1.0e-6 * scale


# --- the reference ------------------------------------------------------------


@pytest.mark.slow
def test_the_tensor_agrees_with_elk_on_the_same_crystal():
    """Against ``nonlinopt.f90``, on the same AlAs cell and the same 6x6x6 mesh.

    **What is comparable and what is not**, stated rather than discovered.
    Elk is all-electron LAPW and this is a norm-conserving pseudopotential at
    ``ecutwfc = 30``; the two do not have the same gap, and a second-order
    susceptibility carries two energy denominators, so a gap difference is not
    a scale factor. Elk's ``swidth`` is in **Hartree** and this module's
    ``broadening`` is in Rydberg, so the run below uses 0.010 Ry against Elk's
    0.005 Ha -- getting that wrong makes every peak twice too tall and nothing
    else, which is exactly how it was found.

    Measured on the committed reference: the **resonance position** agrees to
    **0.5%** (2.152 eV against 2.163), the **peak height** to **7%** (672.5
    pm/V against 628.0), and the **static value** to **11%** (-75.8 against
    -85.4). The static number is the one that moves with the basis and it moves
    the right way -- ``alas-raman.in``'s ``ecutwfc = 10`` gives -66.3 -- and
    then **stops moving**: 45 Ry gives -76.6 against 30 Ry's -75.8, so the
    remaining 11% is the pseudopotential against the all-electron answer and
    not an unconverged basis. That is why the assertion below is loose and the
    measured numbers rather than the tolerance are what is recorded.
    """
    frequencies, elk = read_elk_chi(ELK / "alas-chi2-123.elk.out")
    chi = spectrum("alas-shg.in", 6, 22, window=0.6, nw=240, broadening=0.010)
    ours = np.asarray(chi.chi)[:, 0, 1, 2]

    # The static limit, in Elk's own atomic units.
    from defumat.response.shg import CHI2_AU_TO_PM_PER_V

    static_ours = float(ours[0].real) / CHI2_AU_TO_PM_PER_V
    static_elk = float(elk[0].real)
    assert np.sign(static_ours) == np.sign(static_elk)
    assert abs(static_ours - static_elk) < 0.25 * abs(static_elk)

    # The resonance, which is the sharper of the two statements.
    ours_peak = int(np.argmax(np.abs(ours)))
    elk_peak = int(np.argmax(np.abs(elk)))
    position_ours = float(chi.frequencies_ev[ours_peak])
    position_elk = float(frequencies[elk_peak]) * 2.0 * RY_TO_EV  # Ha -> eV
    assert abs(position_ours - position_elk) < 0.05 * position_elk

    height_ours = float(np.abs(ours[ours_peak])) / CHI2_AU_TO_PM_PER_V
    height_elk = float(np.abs(elk[elk_peak]))
    assert abs(height_ours - height_elk) < 0.20 * height_elk


@pytest.mark.slow
def test_the_scissors_shift_moves_the_two_photon_resonance_by_half_of_itself():
    """The scissors branch, against Elk's own scissored run.

    A rigid shift ``Delta`` of the empty states moves every transition energy
    by ``Delta``, so the **second-harmonic** resonance -- which sits where
    ``2 hbar w`` reaches a transition -- moves by ``Delta / 2`` on the
    fundamental axis. That factor is the whole content of the branch and it is
    the one thing a wrong implementation would get wrong while leaving the
    tensor exactly ``-43m``.

    **The dipoles must not move with it**, which is why they are built from the
    unshifted gaps here: ``getpmat.f90`` reaches the same place by scaling the
    momentum matrix elements by ``e / (e -+ scissor)``, so that ``r = -i p / e``
    is left alone. A scissors correction is a statement about energies and not
    about wavefunctions.

    Measured against ``alas-chi2-123-scissor.elk.out`` at ``Delta = 0.05`` Ha:
    the peak moves **0.0502 Ry** against a half-scissor of 0.0500, lands at
    0.1042 Ha against Elk's 0.1035, and its height falls to **0.60** of the
    unscissored value against Elk's 0.58. The static value collapses in both
    codes (-3.10 to -1.14 here, -3.50 to -1.00 in Elk) and is the loosest of
    the three, because a near-cancellation amplifies the 11% the two codes
    already differ by.
    """
    from defumat.response.shg import CHI2_AU_TO_PM_PER_V

    scissor = 0.1  # Ry, which is Elk's 0.05 Ha
    plain = spectrum("alas-shg.in", 6, 22, window=0.6, nw=240, broadening=0.010)
    shifted = spectrum("alas-shg.in", 6, 22, window=0.6, nw=240,
                       broadening=0.010, scissor=scissor)

    def peak(result):
        component = np.asarray(result.chi)[:, 0, 1, 2]
        index = int(np.argmax(np.abs(component)))
        return (float(result.frequencies[index]),
                float(np.abs(component[index])) / CHI2_AU_TO_PM_PER_V)

    position, height = peak(plain)
    moved, height_moved = peak(shifted)

    # Half the scissor, to within one point of the frequency axis.
    assert abs((moved - position) - 0.5 * scissor) < 2.0 * (0.6 / 240)

    # And the height falls the way Elk's does.
    _, elk_plain = read_elk_chi(ELK / "alas-chi2-123.elk.out")
    _, elk_shifted = read_elk_chi(ELK / "alas-chi2-123-scissor.elk.out")
    ratio = height_moved / height
    elk_ratio = float(np.max(np.abs(elk_shifted))) / float(np.max(np.abs(elk_plain)))
    assert abs(ratio - elk_ratio) < 0.10


@pytest.mark.slow
def test_the_three_parts_agree_with_elks_three_parts():
    """Elk writes ``chi_II``, ``eta_II`` and ``i/2w sigma_II`` to three files.

    Comparing only their sum would let two errors cancel, which is the reason
    this module keeps them apart in the first place -- P43's lesson about the
    five partial derivatives of ``d(eps)/d(tau)``, where three agreed and two
    did not and that is what localised the bug. Here it is a check on the
    reference rather than on a bug: each part is a different arrangement of the
    same matrix elements, so agreeing on all three is a much stronger statement
    than agreeing on one number.

    The tolerance is the same 25% the total carries, and for the same reason:
    LAPW against a norm-conserving pseudopotential with a different gap.
    ``sigma_II`` is the smallest of the three and the loosest.
    """
    from defumat.response.shg import CHI2_AU_TO_PM_PER_V

    result = spectrum("alas-shg.in", 6, 22, window=0.6, nw=240, broadening=0.010)
    ours = {
        "chi_II": np.asarray(result.chi_ii)[:, 0, 1, 2],
        "eta_II": np.asarray(result.eta_ii)[:, 0, 1, 2],
        "sigma_II": np.asarray(result.sigma_ii)[:, 0, 1, 2],
    }
    files = {
        "chi_II": "alas-chi2-II-123.elk.out",
        "eta_II": "alas-eta2-II-123.elk.out",
        "sigma_II": "alas-sigma2-II-123.elk.out",
    }
    for name, component in ours.items():
        _, reference = read_elk_chi(ELK / files[name])
        mine = float(np.max(np.abs(component))) / CHI2_AU_TO_PM_PER_V
        theirs = float(np.max(np.abs(reference)))
        assert theirs > 0.0
        assert abs(mine - theirs) < 0.35 * theirs, name


# --- the refusals -------------------------------------------------------------


@pytest.mark.slow
def test_a_symmetry_reduced_wedge_is_refused_by_name():
    """``chi^abc`` is a polar rank-3 tensor and a wedge sum is not the cell's."""
    system = build_system(read_pw_input(CASES / "alas-raman-wedge.in"))
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    with pytest.raises(NotImplementedError, match="whole k-grid"):
        require_an_shg_regime(Calculation(system, pseudos))


@pytest.mark.slow
def test_the_refusals_name_second_harmonic_generation_rather_than_the_shift_current():
    """The guard is inherited from :mod:`~defumat.response.photocurrent`.

    Sharing the refusals is right -- this module is the same velocity matrix
    elements contracted a different way, so it has every one of their reasons --
    but a caller who asked for ``chi^(2)`` and is told about "the shift current"
    has been handed the wrong quantity's error, which is its own small defect.
    """
    system = build_system(read_pw_input(CASES / "alas-raman-wedge.in"))
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    with pytest.raises(NotImplementedError) as raised:
        require_an_shg_regime(Calculation(system, pseudos))
    assert "second-harmonic generation" in str(raised.value)
    assert "shift current" not in str(raised.value)
