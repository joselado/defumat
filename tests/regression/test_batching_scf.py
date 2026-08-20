"""A whole SCF run must not notice how many k-points are held at once.

``pypresso.batching`` chunks the k axis so that a calculation costs what QE's
``k_loop`` costs in memory rather than ``nk`` times it. The chunk size is a
memory setting and nothing else, so every number a run produces has to be the
same at both ends of the dial -- and "the same" here means to round-off, since
the only thing that changes is the order the per-k contributions are summed in.

Two cases, because they exercise different chunked paths:

* **norm-conserving silicon** -- the eigensolver, ``sum_band`` and the
  Rayleigh-Ritz seeding.
* **ultrasoft silicon** -- all of the above plus ``sum_bec``, which accumulates
  the projector occupations per k-point and feeds ``D_ij``, the augmentation
  charge and the overlap operator. A chunking error there would move the
  density rather than merely the arithmetic that reads it.

The band axis is the same dial on a different axis -- ``vloc_psi_k``'s
``DO ibnd`` against one transform of the whole block -- and gets the same
end-to-end statement at the bottom of this file. It has to be made in a
subprocess: ``PYPRESSO_BAND_BATCH`` is read when a function is *traced*, so
changing it inside a live process would leave the already-compiled kernels on
the old setting and compare a run against itself.

The unit tests in ``tests/unit/test_batching.py`` cover the primitives; this is
the end-to-end statement.
"""

from pathlib import Path

import numpy as np
import pytest

from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.scf import run_scf
from pypresso.system import build_system
from tests.conftest import GENERATED

pytestmark = [pytest.mark.regression, pytest.mark.slow]

#: Round-off between two summation orders over a handful of k-points, in Ry.
#: Not a physics tolerance: anything a chunking bug does is orders larger.
SUMMATION_ROUNDOFF_RY = 1.0e-10

#: The chunkings compared against one ``vmap`` over the whole axis: QE's loop,
#: a chunk that does not divide the k-point count, and one that exceeds it.
CHUNKS = [1, 2, 32]


def _silicon(qe_testsuite, pseudo_dir):
    system = build_system(read_pw_input(qe_testsuite / "pw_scf" / "scf.in"))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    return system, pseudos


def _ultrasoft(pseudo_dir):
    system = build_system(read_pw_input(GENERATED / "si2-us.in"))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    return system, pseudos


def _compare(system, pseudos, batch):
    reference = run_scf(system, pseudos, conv_thr=1.0e-10, max_iterations=80, k_batch=None)
    chunked = run_scf(system, pseudos, conv_thr=1.0e-10, max_iterations=80, k_batch=batch)

    assert chunked.iterations == reference.iterations
    assert chunked.total_energy == pytest.approx(
        reference.total_energy, abs=SUMMATION_ROUNDOFF_RY
    )
    for term, value in reference.energy_terms.items():
        assert chunked.energy_terms[term] == pytest.approx(value, abs=SUMMATION_ROUNDOFF_RY)
    assert np.abs(
        np.asarray(chunked.eigenvalues) - np.asarray(reference.eigenvalues)
    ).max() < SUMMATION_ROUNDOFF_RY
    assert np.abs(
        np.asarray(chunked.density) - np.asarray(reference.density)
    ).max() < SUMMATION_ROUNDOFF_RY


@pytest.mark.parametrize("batch", CHUNKS)
def test_norm_conserving_silicon_is_unchanged(batch, qe_testsuite, pseudo_dir):
    system, pseudos = _silicon(qe_testsuite, pseudo_dir)
    _compare(system, pseudos, batch)


@pytest.mark.parametrize("batch", CHUNKS)
def test_ultrasoft_silicon_is_unchanged(batch, pseudo_dir):
    """The case that also chunks ``sum_bec``."""
    if not (GENERATED / "si2-us.in").is_file():  # pragma: no cover - data present
        pytest.skip("ultrasoft reference input not generated")
    system, pseudos = _ultrasoft(pseudo_dir)
    _compare(system, pseudos, batch)


# --- the band axis, end to end ------------------------------------------------

_BAND_BATCH_RUN = """
import json, sys
from pathlib import Path
from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.scf import run_scf
from pypresso.system import build_system

system = build_system(read_pw_input(Path(sys.argv[1])))
pseudos = tuple(read_upf(Path(sys.argv[2]) / s.pseudo_file) for s in system.structure.species)
result = run_scf(system, pseudos, conv_thr=1.0e-10, max_iterations=80)
print(json.dumps({
    "energy": float(result.total_energy),
    "iterations": int(result.iterations),
    "eigenvalues": [float(e) for e in result.eigenvalues.reshape(-1)],
}))
"""


def _run_with_band_batch(setting, input_path, pseudo_dir):
    """One SCF in a fresh process at ``PYPRESSO_BAND_BATCH=setting``."""
    import json
    import os
    import subprocess
    import sys

    environment = dict(os.environ, PYPRESSO_BAND_BATCH=setting)
    finished = subprocess.run(
        [sys.executable, "-c", _BAND_BATCH_RUN, str(input_path), str(pseudo_dir)],
        capture_output=True, text=True, env=environment,
        cwd=str(Path(__file__).resolve().parent.parent.parent),
    )
    assert finished.returncode == 0, finished.stderr[-2000:]
    return json.loads(finished.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("setting", ["1", "2", "all"])
def test_the_band_chunk_does_not_move_the_answer(setting, qe_testsuite, pseudo_dir):
    """Walking the bands is a working-set decision, not a physical one.

    ``h_psi``'s local term and ``sum_band`` both walk the band axis, and the
    first of those is where more than half the SCF's time goes -- so this is the
    statement that the 2.5x it is worth on a large box was bought with
    scheduling and nothing else.
    """
    input_path = qe_testsuite / "pw_scf" / "scf.in"
    reference = _run_with_band_batch("all", input_path, pseudo_dir)
    chunked = _run_with_band_batch(setting, input_path, pseudo_dir)

    assert chunked["iterations"] == reference["iterations"]
    assert chunked["energy"] == pytest.approx(reference["energy"],
                                              abs=SUMMATION_ROUNDOFF_RY)
    np.testing.assert_allclose(chunked["eigenvalues"], reference["eigenvalues"],
                               rtol=0, atol=1.0e-9)
