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
