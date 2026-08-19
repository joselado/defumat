"""P8 check: densities of state on real systems, SCF -> NSCF -> integrate.

No ``dos.x`` reference is committed anywhere in QE's test suite, so what is
asserted here is what a DOS has to satisfy rather than a pointwise comparison:

* the integrated DOS returns the electron count at the Fermi level,
* the Fermi level recovered *from* the DOS is the one the NSCF run found,
* silicon's gap contains no states at all under the tetrahedron scheme,
* aluminium's DOS is the square root a nearly-free-electron metal must give, and
* tetrahedra and a smearing count the same states.

The band energies themselves are already compared against QE band by band in
``test_scf.py`` and ``test_tetrahedra.py``, so the eigenvalues going into this
are known to be right; what is new here is the integration on top of them.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.scf import run_scf
from pypresso.system import build_system
from pypresso.units import RY_TO_EV
from pypresso.workflows import run_dos
from tests.tolerances import DOS_STATES_EV, FERMI_EV

pytestmark = [pytest.mark.regression, pytest.mark.slow]

TETRAHEDRON_SCHEMES = ["tetrahedra", "tetrahedra-lin", "tetrahedra-opt"]


@lru_cache(maxsize=None)
def _converged(input_path: Path, pseudo_dir: Path):
    system = build_system(read_pw_input(input_path))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    return system, pseudos, run_scf(system, pseudos, conv_thr=1e-10, max_iterations=80)


@lru_cache(maxsize=None)
def _dos(input_path: Path, pseudo_dir: Path, scheme: str, n: int, nbnd: int, degauss: float):
    """Cached DOS run. An empty ``scheme`` means "whatever the input asked for"."""
    system, pseudos, scf = _converged(input_path, pseudo_dir)
    dos, nscf = run_dos(
        system,
        pseudos,
        scf.density,
        grid=(n, n, n),
        nbnd=nbnd,
        scheme=scheme or None,
        degauss=degauss or None,
    )
    return scf, dos, nscf


# --------------------------------------------------------------------------
# Silicon: a gap, and it must be empty
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", TETRAHEDRON_SCHEMES)
def test_silicon_integrated_dos_returns_the_electron_count(qe_testsuite, pseudo_dir, scheme):
    """``N(E)`` at the top of the valence band is exactly the eight valence electrons."""
    path = qe_testsuite / "pw_scf" / "scf.in"
    _, dos, nscf = _dos(path, pseudo_dir, scheme, 8, 8, 0.0)
    valence_top = float(nscf.eigenvalues[:, 3].max())

    assert dos.states_below(valence_top) == pytest.approx(8.0, abs=1e-3)
    # No states below the lowest band, all of them above the highest. The
    # optimised method reaches both only to 1e-3, and that is arithmetic rather
    # than error: its corner energies are a stencil with *negative* weights, so
    # a few of them fall outside the range of the eigenvalues themselves, which
    # is the range dos.x sizes the energy grid from.
    exact = 1e-6 if scheme != "tetrahedra-opt" else 1e-3
    assert dos.integrated[0] == pytest.approx(0.0, abs=exact)
    assert dos.integrated[-1] == pytest.approx(2.0 * 8, abs=exact)


@pytest.mark.parametrize("scheme", ["tetrahedra", "tetrahedra-lin"])
def test_silicon_gap_is_empty(qe_testsuite, pseudo_dir, scheme):
    """A tetrahedron DOS has no width, so an insulator's gap really is empty.

    ``tetrahedra-opt`` is excluded on purpose and it is not a defect: Kawamura's
    20-point stencil mixes in energies from *outside* the tetrahedron, so a band
    edge leaks a little weight across it. That is the price of the accuracy it
    buys on a smooth band, and QE's ``opt_tetra_dos_t`` behaves the same way.
    """
    path = qe_testsuite / "pw_scf" / "scf.in"
    _, dos, nscf = _dos(path, pseudo_dir, scheme, 8, 8, 0.0)
    valence_top = float(nscf.eigenvalues[:, 3].max())
    conduction_bottom = float(nscf.eigenvalues[:, 4].min())
    assert conduction_bottom > valence_top

    margin = 0.05 * (conduction_bottom - valence_top)
    inside = (dos.energies > valence_top + margin) & (dos.energies < conduction_bottom - margin)
    assert inside.sum() > 3
    assert np.abs(dos.dos[inside]).max() / RY_TO_EV == pytest.approx(0.0, abs=DOS_STATES_EV)


def test_silicon_smearing_dos_fills_the_gap_and_tetrahedra_do_not(qe_testsuite, pseudo_dir):
    """The qualitative difference between the two schemes, on the system that shows it."""
    path = qe_testsuite / "pw_scf" / "scf.in"
    _, sharp, nscf = _dos(path, pseudo_dir, "tetrahedra", 8, 8, 0.0)
    _, broad, _ = _dos(path, pseudo_dir, "gaussian", 8, 8, 0.02)

    middle = 0.5 * (float(nscf.eigenvalues[:, 3].max()) + float(nscf.eigenvalues[:, 4].min()))
    assert sharp.at(middle) / RY_TO_EV == pytest.approx(0.0, abs=DOS_STATES_EV)
    assert broad.at(middle) / RY_TO_EV > 10.0 * DOS_STATES_EV


# --------------------------------------------------------------------------
# Aluminium: a free-electron-like metal
# --------------------------------------------------------------------------


def test_aluminium_fermi_level_comes_back_out_of_the_dos(qe_testsuite, pseudo_dir):
    """``N(E_F) = nelec`` is the definition of ``E_F``; the DOS has to reproduce it.

    The Fermi level is found by bisection on the tetrahedron count during the
    NSCF run, and the DOS is built afterwards on an independent energy grid, so
    this closes the loop between the two -- the same ``N(E)`` seen through both
    paths. The scheme is left to default, which makes it the calculation's own
    (``tetrahedra-opt`` for this input); a *different* scheme would legitimately
    put ``N(E_F)`` a few thousandths of an electron away, since ``dos.x``
    likewise takes ``ef`` from the ``pw.x`` run and only the DOS from ``bz_sum``.
    """
    path = qe_testsuite / "pw_metal" / "metal-tetrahedra.in"
    _, dos, _ = _dos(path, pseudo_dir, "", 12, 6, 0.0)

    assert dos.scheme == "tetrahedra-opt"
    assert dos.fermi_energy is not None
    assert dos.states_below(dos.fermi_energy) == pytest.approx(3.0, abs=1e-4)
    # Inverting N(E) on the grid must give the Fermi level back.
    recovered = float(np.interp(3.0, dos.integrated, dos.energies))
    assert recovered * RY_TO_EV == pytest.approx(dos.fermi_energy * RY_TO_EV, abs=FERMI_EV)


@pytest.mark.parametrize("scheme", TETRAHEDRON_SCHEMES)
def test_aluminium_schemes_agree_on_where_the_fermi_level_is(qe_testsuite, pseudo_dir, scheme):
    """All three variants must put ``N(E) = 3`` within 20 meV of each other."""
    path = qe_testsuite / "pw_metal" / "metal-tetrahedra.in"
    _, dos, _ = _dos(path, pseudo_dir, scheme, 12, 6, 0.0)
    # ``dos.fermi_energy`` is the NSCF grid's, found with the *calculation's*
    # scheme; each variant's own N(E) has to invert to within 20 meV of it.
    recovered = float(np.interp(3.0, dos.integrated, dos.energies))
    assert recovered * RY_TO_EV == pytest.approx(dos.fermi_energy * RY_TO_EV, abs=0.02)


def test_aluminium_dos_is_free_electron_like(qe_testsuite, pseudo_dir):
    """``D(E) ~ sqrt(E - E_0)`` at the bottom of a nearly-free-electron band.

    Aluminium is the textbook case, and it is the reason this cell is the one
    QE's tetrahedron benchmarks use: the shape is known independently of QE.
    The window stops well below the Fermi level -- higher up the second band
    comes in and the zone boundary puts van Hove kinks in, neither of which a
    single square root describes.
    """
    path = qe_testsuite / "pw_metal" / "metal-tetrahedra.in"
    _, dos, nscf = _dos(path, pseudo_dir, "tetrahedra", 12, 6, 0.0)

    bottom = float(nscf.eigenvalues.min())
    window = (dos.energies > bottom + 0.03) & (dos.energies < bottom + 0.30)
    energies, values = dos.energies[window], dos.dos[window]
    # A square root is a straight line in D^2 against E.
    slope, intercept = np.polyfit(energies, values**2, 1)
    predicted = np.sqrt(np.maximum(slope * energies + intercept, 0.0))
    assert slope > 0.0
    assert np.abs(predicted - values).max() < 0.05 * values.max()


@pytest.mark.parametrize("scheme", TETRAHEDRON_SCHEMES)
def test_aluminium_tetrahedra_and_smearing_count_the_same_states(
    qe_testsuite, pseudo_dir, scheme
):
    """The two schemes agree on ``N(E)`` throughout the occupied band.

    ``N`` rather than ``D``: a symmetric broadening leaves the integrated count
    alone to first order but visibly rounds off van Hove structure, of which
    aluminium at a 12^3 grid has plenty, so a pointwise comparison of ``D``
    measures the smearing width rather than the agreement. That the two schemes
    give the same *DOS* where the DOS is smooth is checked in
    ``tests/unit/test_dos.py`` on a free-electron band, which has no such
    structure to argue about.
    """
    path = qe_testsuite / "pw_metal" / "metal-tetrahedra.in"
    _, sharp, nscf = _dos(path, pseudo_dir, scheme, 12, 6, 0.0)
    _, broad, _ = _dos(path, pseudo_dir, "gaussian", 12, 6, 0.02)

    bottom = float(nscf.eigenvalues.min())
    window = (sharp.energies > bottom + 0.15) & (sharp.energies < sharp.fermi_energy - 0.05)
    assert window.sum() > 10
    # The smearing grid is padded by 3*degauss, so the two grids differ;
    # compare on the tetrahedron one.
    counted = np.interp(sharp.energies[window], broad.energies, broad.integrated)
    assert counted == pytest.approx(sharp.integrated[window], abs=0.08)
