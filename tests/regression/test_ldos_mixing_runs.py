"""``mixing_mode = 'ldos'`` on the cells that motivated it, run to convergence.

The measurement is ``VACUUM-MIXING-NEXT.md``: a metal film beside vacuum is where
plain Anderson, Kerker and ``local-TF`` slow down as the vacuum grows, and where the
LDOS preconditioner does not. Each test states its count as measured and asserts a
bound a few iterations above it, since an iteration count is a property of one code
state; what must not move is the energy each run lands on.
"""

from pathlib import Path

import jax
import pytest

from defumat import Calculator
from tests.tolerances import TOTAL_ENERGY_RY

pytestmark = [pytest.mark.slow, pytest.mark.regression]

REPO = Path(__file__).resolve().parents[2]
PSEUDO = REPO / "tests" / "data" / "pseudo"


@pytest.fixture(autouse=True)
def _clear_caches():
    yield
    jax.clear_caches()


def _calc(path):
    return Calculator.from_file(path, pseudo_dir=PSEUDO, announce=False)


@pytest.mark.parametrize("name", ["al-slab.in", "al-slab-v64.in"])
def test_the_aluminium_slab_takes_the_same_count_at_16_and_64_bohr(name):
    """Measured 10 and 10 (flat fit, D22, 2026-10-04), where plain Anderson takes 25
    and 34 and Kerker 15 and 36."""
    path = REPO / "benchmarks" / name
    ldos = _calc(path).get_scf(mixing_mode="ldos")
    plain = _calc(path).get_scf()
    assert ldos.converged and ldos.iterations <= 14
    assert ldos.iterations < plain.iterations
    assert ldos.total_energy == pytest.approx(plain.total_energy, abs=TOTAL_ENERGY_RY)


def test_the_cobalt_film_converges_to_pw_x_at_its_own_beta():
    """Three Co(0001) layers, ultrasoft, spin-polarised, at ``mixing_beta = 0.7``:
    plain Anderson with this code's default fit does not converge in 150, and
    ``pw.x`` 7.5 takes 24 with ``local-TF`` to -223.13884423 Ry. Measured 20 (flat
    fit, D22, 2026-10-04), at a moment of 5.26 Bohr magnetons."""
    result = _calc(REPO / "tests" / "data" / "qe" / "co-slab-forcetheorem-sr.in").get_scf(
        mixing_mode="ldos")
    assert result.converged and result.iterations <= 24
    assert result.total_energy == pytest.approx(-223.13884423, abs=1e-7)
    assert result.magnetization == pytest.approx(5.26, abs=5e-3)
