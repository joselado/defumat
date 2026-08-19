"""P0 check: the QE reference parser reads real benchmark outputs correctly.

Expected values are transcribed by hand from the committed benchmark files --
hence ``committed_benchmark`` rather than ``benchmark``: these tests are about
what is in a particular file, not about what the current QE would print.
This is the one place in the test suite where that is appropriate -- everywhere
else compares computed results against whatever the parser returns, so if the
parser is wrong every later phase is silently wrong too.

Coverage spans the four output shapes the first milestone needs: an SCF run with
stress (`pw_scf/scf.in`), a `bands` run (`scf-1.in`), an `nscf` run (`scf-2.in`),
a metal with smearing and a Fermi level (`pw_metal/metal.in`), and a
spin-polarised run whose eigenvalues come in two channels (`pw_lsda/lsda.in`).
"""

import numpy as np
import pytest

from pypresso.io import read_qe_output

pytestmark = pytest.mark.unit


def test_scf_silicon(committed_benchmark):
    """The canonical first target: Si diamond, LDA, 2 k-points, with stress."""
    ref = read_qe_output(committed_benchmark("pw_scf", "scf.in"))

    assert ref.calculation == "scf"
    assert (ref.ibrav, ref.nat, ref.ntyp, ref.nbnd) == (2, 2, 1, 4)
    assert ref.alat == pytest.approx(10.2)
    assert ref.volume == pytest.approx(265.3020)
    assert ref.nelec == pytest.approx(8.0)
    assert (ref.ecutwfc, ref.ecutrho) == (12.0, 48.0)
    assert ref.xc.startswith("SLA  PZ")

    # fcc primitive vectors in units of alat, as QE orders them for ibrav=2
    assert ref.at == pytest.approx(
        np.array([[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.0]])
    )
    assert ref.bg == pytest.approx(
        np.array([[-1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, -1.0]])
    )
    # at . bg^T = identity is a property of the printed units, and a good check
    # that the two blocks were not swapped.
    assert ref.at @ ref.bg.T == pytest.approx(np.eye(3), abs=1e-6)

    assert ref.ngm_dense == 1459
    assert ref.fft_dense == (15, 15, 15)
    assert ref.npw.tolist() == [180, 186]

    assert ref.kpoints == pytest.approx(np.array([[0.25, 0.25, 0.25], [0.25, 0.25, 0.75]]))
    assert ref.weights == pytest.approx(np.array([0.5, 1.5]))
    assert ref.weights.sum() == pytest.approx(2.0)  # QE normalises to 2 without spin

    assert ref.total_energy == pytest.approx(-15.79449593)
    assert ref.energy_terms == {
        "one-electron": pytest.approx(4.83378641),
        "hartree": pytest.approx(1.08429090),
        "xc": pytest.approx(-4.81281466),
        "ewald": pytest.approx(-16.89975858),
    }
    # The decomposition must reproduce the total; this catches a term being
    # dropped by a regex as well as a sign error.
    assert sum(ref.energy_terms.values()) == pytest.approx(ref.total_energy, abs=1e-8)

    assert ref.eigenvalues.shape == (1, 2, 4)
    assert ref.eigenvalues[0, 0] == pytest.approx([-4.8701, 2.3792, 5.5371, 5.5371])
    assert ref.eigenvalues[0, 1] == pytest.approx([-2.9165, -0.0653, 2.6795, 4.0355])
    assert ref.homo == pytest.approx(5.5371)
    assert ref.fermi_energy is None  # insulator: QE reports a HOMO instead
    assert ref.n_iterations == 5

    assert ref.pressure == pytest.approx(-30.30)
    assert np.diag(ref.stress) == pytest.approx([-0.00020597] * 3)
    assert ref.stress == pytest.approx(ref.stress.T, abs=1e-12)
    assert ref.forces is None  # not requested by this input


def test_bands_run(committed_benchmark):
    """`scf-1.in` restarts from scf.in's density: bands only, no energy."""
    ref = read_qe_output(committed_benchmark("pw_scf", "scf-1.in"))

    assert ref.calculation == "bands"
    assert ref.nbnd == 8
    assert ref.eigenvalues.shape == (1, 21, 8)
    assert len(ref.kpoints) == 21
    assert ref.total_energy is None
    assert ref.energy_terms == {}
    # Band energies must be ordered within each k-point.
    assert np.all(np.diff(ref.eigenvalues, axis=-1) >= -1e-8)


def test_nscf_run(committed_benchmark):
    """`scf-2.in`: same density, a new k-grid, occupations resolved."""
    ref = read_qe_output(committed_benchmark("pw_scf", "scf-2.in"))

    assert ref.calculation == "nscf"
    assert ref.eigenvalues.shape == (1, 10, 8)
    assert ref.homo == pytest.approx(6.0279)
    assert ref.total_energy is None


def test_metal_has_fermi_level_and_smearing(committed_benchmark):
    ref = read_qe_output(committed_benchmark("pw_metal", "metal.in"))

    assert ref.fermi_energy == pytest.approx(8.3513)
    assert ref.homo is None
    assert "smearing" in ref.energy_terms
    assert ref.energy_terms["smearing"] < 0.0  # -TS is negative by construction
    assert sum(ref.energy_terms.values()) == pytest.approx(ref.total_energy, abs=1e-8)


def test_spin_polarised_gives_two_channels(committed_benchmark):
    ref = read_qe_output(committed_benchmark("pw_lsda", "lsda.in"))

    assert ref.nspin == 2
    assert ref.eigenvalues.shape[0] == 2
    assert ref.eigenvalues.shape[1] == ref.nk
    # Exchange splitting: the two channels must not be the same numbers.
    assert not np.allclose(ref.eigenvalues[0], ref.eigenvalues[1])


def test_missing_quantities_are_none_not_errors(tmp_path):
    """A truncated output parses to a mostly-empty reference without raising."""
    stub = tmp_path / "truncated.out"
    stub.write_text("     Program PWSCF v.7.5 starts\n     bravais-lattice index     =            2\n")

    ref = read_qe_output(stub)
    assert ref.ibrav == 2
    assert ref.total_energy is None
    assert ref.eigenvalues is None
    assert ref.energy_terms == {}
