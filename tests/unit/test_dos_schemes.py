"""P8 unit checks: the scheme registry, the energy grid, and the ``.dos`` writer.

Nothing here runs an SCF; the eigenvalues are made up. What is checked is that
each scheme integrates to the right number of states, that the energy grid is
``dos.x``'s, and that the file written is byte-for-byte the one ``dos.x`` writes.
"""

import numpy as np
import pytest

from defumat.io.output import format_dos, fortran_exponential, write_dos
from defumat.scf.tetrahedra import build_tetrahedra
from defumat.system.kpoints import DEGSPIN, monkhorst_pack
from defumat.units import RY_TO_EV
from defumat.workflows.dos import (
    DOS_SCHEMES,
    DEFAULT_DELTA_E,
    compute_dos,
    energy_grid,
    get_dos_scheme,
)

pytestmark = pytest.mark.unit


def _free_electron(n: int):
    points, weights = monkhorst_pack((n, n, n))
    return np.sum(points**2, axis=1)[:, None], weights * DEGSPIN


def _tetrahedra(kind: str, n: int):
    return build_tetrahedra(
        kind, (n, n, n), (0, 0, 0), np.eye(3, dtype=int)[None], np.eye(3), time_reversal=False
    )


# --------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------


def test_registry_covers_every_scheme_a_pw_input_can_ask_for():
    for name in ("gaussian", "mp", "methfessel-paxton", "fermi-dirac", "cold", "smearing",
                 "tetrahedra", "tetrahedra-lin", "tetrahedra-opt",
                 "tetrahedra_lin", "tetrahedra_opt"):
        assert callable(get_dos_scheme(name))
    assert get_dos_scheme("GAUSSIAN") is DOS_SCHEMES["gaussian"]
    with pytest.raises(ValueError, match="unknown density-of-states scheme"):
        get_dos_scheme("lorentzian")


def test_smearing_scheme_refuses_a_zero_width():
    eigenvalues, weights = _free_electron(4)
    with pytest.raises(ValueError, match="positive degauss"):
        compute_dos(eigenvalues, weights, np.linspace(0.0, 1.0, 5), "gaussian", degauss=0.0)


def test_tetrahedron_scheme_refuses_without_tetrahedra():
    eigenvalues, weights = _free_electron(4)
    with pytest.raises(ValueError, match="needs the tetrahedra"):
        compute_dos(eigenvalues, weights, np.linspace(0.0, 1.0, 5), "tetrahedra")


# --------------------------------------------------------------------------
# The energy grid
# --------------------------------------------------------------------------


def test_energy_grid_follows_dos_x():
    """``dos.f90``: bottom of the lowest band to top of the highest, plus 3*degauss."""
    eigenvalues = np.array([[0.0, 2.0], [0.5, 1.5], [-1.0, 3.0]])
    grid = energy_grid(eigenvalues, delta_e=0.1)
    assert grid[0] == pytest.approx(-1.0)
    assert grid[1] - grid[0] == pytest.approx(0.1)
    # nint((3.0 - (-1.0))/0.1 + 0.500001) = 41 points, i.e. both ends included.
    assert len(grid) == 41
    assert grid[-1] == pytest.approx(3.0)

    padded = energy_grid(eigenvalues, delta_e=0.1, degauss=0.2)
    assert padded[0] == pytest.approx(-1.6)

    explicit = energy_grid(eigenvalues, emin=0.0, emax=1.0, delta_e=0.25)
    assert explicit == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])

    with pytest.raises(ValueError):
        energy_grid(eigenvalues, delta_e=0.0)


def test_default_step_is_ten_millielectronvolts():
    assert DEFAULT_DELTA_E * RY_TO_EV == pytest.approx(0.01)


# --------------------------------------------------------------------------
# Sum rules
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", ["gaussian", "mp", "fermi-dirac", "cold"])
def test_smearing_dos_integrates_to_every_state(scheme):
    """``N(E)`` above the band reaches ``2 * nbnd`` states -- all of them."""
    eigenvalues, weights = _free_electron(6)
    energies = np.linspace(-1.0, 2.0, 601)
    dos = compute_dos(eigenvalues, weights, energies, scheme, degauss=0.02)
    assert dos.integrated[-1] == pytest.approx(DEGSPIN, abs=1e-9)
    assert dos.integrated[0] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("scheme", ["gaussian", "fermi-dirac", "tetrahedra", "tetrahedra-opt"])
def test_dos_is_the_derivative_of_the_integrated_dos(scheme):
    """The sum rule holds by construction, so a trapezoid check is tight."""
    eigenvalues, weights = _free_electron(8)
    tetrahedra = _tetrahedra("bloechl" if scheme == "tetrahedra" else "optimized", 8)
    energies = np.linspace(-0.2, 1.0, 1201)
    dos = compute_dos(
        eigenvalues, weights, energies, scheme, degauss=0.02, tetrahedra=tetrahedra
    )
    step = energies[1] - energies[0]
    cumulative = np.concatenate([[0.0], np.cumsum((dos.dos[1:] + dos.dos[:-1]) / 2.0) * step])
    assert cumulative == pytest.approx(dos.integrated - dos.integrated[0], abs=2e-3)


def test_tetrahedra_and_smearing_agree_away_from_band_edges():
    """Two integrations of the same bands, on a window where the DOS is smooth.

    A smearing width has to be chosen between two errors and there is no limit
    in which both vanish at fixed k-grid: too narrow and the sum over discrete
    k-points shows through as spikes (0.32 relative at ``degauss = 0.005``
    here), too wide and the broadening is real (0.03 at 0.04). In between --
    wide compared to the k-point spacing, narrow compared to the band structure
    -- the two schemes agree to about a percent, and that agreement is the check
    that they are integrating the same thing.
    """
    eigenvalues, weights = _free_electron(20)
    tetrahedra = _tetrahedra("bloechl", 20)
    energies = np.linspace(0.05, 0.2, 61)
    sharp = compute_dos(eigenvalues, weights, energies, "tetrahedra", tetrahedra=tetrahedra)
    smeared = compute_dos(eigenvalues, weights, energies, "gaussian", degauss=0.02)
    assert smeared.dos == pytest.approx(sharp.dos, rel=0.03)


# --------------------------------------------------------------------------
# The .dos file
# --------------------------------------------------------------------------


def test_fortran_exponential_normalises_the_mantissa_like_fortran():
    """Fortran's E12.4 writes ``0.1234E+01``; Python's default writes ``1.2340E+00``."""
    assert fortran_exponential(0.0) == "  0.0000E+00"
    assert fortran_exponential(1.0) == "  0.1000E+01"
    assert fortran_exponential(-1.0) == " -0.1000E+01"
    assert fortran_exponential(1234.5) == "  0.1235E+04"
    assert fortran_exponential(9.99999e-3) == "  0.1000E-01"  # carried, not 0.10000E-02
    assert all(len(fortran_exponential(v)) == 12 for v in (1e-99, -1e-99, 3.14159))


def test_dos_file_matches_dos_x_layout(tmp_path):
    eigenvalues, weights = _free_electron(4)
    energies = np.linspace(0.0, 0.5, 6)
    dos = compute_dos(
        eigenvalues, weights, energies, "gaussian", degauss=0.05, fermi_energy=0.2
    )
    text = format_dos(dos)
    lines = text.splitlines()

    assert lines[0] == "#  E (eV)   dos(E)     Int dos(E) EFermi =    2.721 eV"
    assert len(lines) == 1 + len(energies)
    # (f8.3, 2e12.4): eight columns of energy in eV, then two twelve-wide fields.
    assert all(len(line) == 32 for line in lines[1:])
    assert float(lines[1][:8]) == pytest.approx(0.0)

    path = write_dos(tmp_path / "si.dos", dos)
    assert path.read_text() == text
