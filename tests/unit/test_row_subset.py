"""A calculation restricted to a subset of its k-points, nothing rebuilt.

``Calculation.at_rows`` slices every array with a k index and shares every
other one, so that a consumer which needs a *Calculation* for one chunk of
k-points -- a velocity operator's ``jvp``, a Sternheimer solve -- gets exactly
what the whole set would give at those points (``GPU-MEMORY-NEXT.md`` items 2
and 6). The standard is bit-identity where the arithmetic is the same
expression on the same numbers, and round-off where a ``jvp`` over a shorter
axis regroups a sum.
"""

from __future__ import annotations

import re
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.calculator import Calculator
from defumat.response.velocity import VelocityOperator

pytestmark = pytest.mark.unit

PSEUDO = "tests/data/pseudo"
ROWS = np.array([2, 5, 7])


def _calculation(path):
    text = open(path).read()
    text = re.sub(r"K_POINTS.*", "K_POINTS automatic\n 3 3 1 0 0 0", text,
                  flags=re.S)
    text = text.replace("&system", "&system\n    nosym = .true.,", 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Calculator.from_text(text, PSEUDO, announce=False).calculation


def test_a_row_subset_is_the_whole_set_at_those_k_points():
    """Ultrasoft silicon on nine k-points, three of them taken."""
    whole = _calculation("tests/data/qe/si2-us.in")
    part = whole.at_rows(ROWS)
    assert part.system.kpoints.nk == len(ROWS)
    assert part.basis.planewaves.npwx == whole.basis.planewaves.npwx
    np.testing.assert_array_equal(np.asarray(part.system.kpoints.weights),
                                  np.asarray(whole.system.kpoints.weights)[ROWS])

    potential = whole.potential(whole.starting_density())
    hamiltonians = whole.hamiltonian(potential.v_scf)
    energies, psi = whole.diagonalize(hamiltonians, 6, None, 1e-11)
    part_energies, _ = part.diagonalize(part.hamiltonian(potential.v_scf), 6,
                                        None, 1e-11)
    np.testing.assert_array_equal(np.asarray(part_energies),
                                  np.asarray(energies)[:, ROWS])

    part_hamiltonian = part.hamiltonian(potential.v_scf)[0]
    for position, ik in enumerate(ROWS):
        np.testing.assert_array_equal(
            np.asarray(part_hamiltonian.apply(psi[0, ik], position)),
            np.asarray(hamiltonians[0].apply(psi[0, ik], ik)))

    weights = jnp.ones((1, len(ROWS), 6))
    chunk = psi[:, ROWS]
    for a, b in zip(whole.becsum(chunk, weights, rows=ROWS, symmetrize=False),
                    part.becsum(chunk, weights, symmetrize=False)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    # The velocity operator is a jvp of at_kcart over the k axis: the same
    # operator at each point, over a shorter axis.
    direction = np.array([1.0, 0.3, -0.2])
    dh, ds = VelocityOperator(whole, potential.v_scf).both(psi, direction)
    part_dh, part_ds = VelocityOperator(part, potential.v_scf).both(chunk, direction)
    np.testing.assert_allclose(np.asarray(part_dh), np.asarray(dh)[:, ROWS],
                               atol=1e-14)
    np.testing.assert_allclose(np.asarray(part_ds), np.asarray(ds)[:, ROWS],
                               atol=1e-14)


def test_a_spiral_is_refused_a_row_subset():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spiral = Calculator.from_file("tests/data/qe/h-chain-spiral.in",
                                      pseudo_dir=PSEUDO,
                                      announce=False).calculation
    with pytest.raises(NotImplementedError, match="at_rows on a spin spiral"):
        spiral.at_rows([0])
