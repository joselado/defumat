"""The sums over states of a collinear ``nspin = 2`` run: two channels that add.

The optical conductivity and the second-harmonic tensor of a spin-polarized
insulator are each channel's band structure's own, summed; the identity that
checks the channel loop is an unpolarized crystal run both ways, zincblende
AlAs with ``tot_magnetization = 0``, whose two channels are each half of the
``nspin = 1`` answer. ``PLAN.md`` P139 has the measurements.
"""

import re
from pathlib import Path

import jax
import numpy as np
import pytest

from defumat import Calculator

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"


@pytest.fixture(autouse=True)
def _bounded_compilation():
    yield
    jax.clear_caches()


def _alas(tmp_path, pseudo_dir, extra=""):
    text = (CASES / "alas-shg.in").read_text()
    text = re.sub(r"ecutwfc\s*=\s*[0-9.dD+-]+", "ecutwfc = 10.0", text)
    text = re.sub(r"K_POINTS.*\n\s*[0-9 ]+\n", "K_POINTS (automatic)\n 2 2 2 0 0 0\n", text)
    if extra:
        text = re.sub(r"&system", "&system\n    " + extra, text, count=1)
    path = tmp_path / f"alas{len(extra)}.in"
    path.write_text(text)
    calculator = Calculator.from_file(path, pseudo_dir=pseudo_dir, announce=False)
    calculator.get_scf(conv_thr=1e-12)
    return calculator


def test_two_unpolarized_channels_are_the_scalar_run(pseudo_dir, tmp_path):
    """``get_shg`` and ``get_optical_conductivity`` at ``nspin = 2``, no moment, against ``nspin = 1``.

    At nine bands, a clean cut (the gap to the tenth is 4.9e-3 Ry at every
    k-point; a cut inside a multiplet makes either sum depend on the rotation
    the eigensolver returned, which is a statement about the cut and not about
    the channels). Measured: chi^(2) 1.4e-9 of its largest component apart.
    """
    one = _alas(tmp_path, pseudo_dir)
    two = _alas(tmp_path, pseudo_dir, "nspin = 2, tot_magnetization = 0,")
    options = dict(nbnd=9, nw=4, window=0.2, broadening=0.01)
    a, b = np.asarray(one.get_shg(**options).chi), np.asarray(two.get_shg(**options).chi)
    assert np.abs(a - b).max() < 1e-7 * np.abs(a).max()
    options = dict(nbnd=9, nw=4, window=0.4, broadening=0.01)
    a = np.asarray(one.get_optical_conductivity(**options).sigma)
    b = np.asarray(two.get_optical_conductivity(**options).sigma)
    assert np.abs(a - b).max() < 1e-7 * np.abs(a).max()
