"""The parts of the piezoelectric tensor that need no self-consistent field.

The Voigt convention and the polar-crystal guard, both of which are decisions
rather than computations -- and both of which are silent when wrong. A factor
of two in the first makes every published ``e_14`` disagree by two with no
symmetry saying so; the second is the difference between the improper mixed
derivative this phase computes and the proper piezoelectric response a
measurement sees.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pypresso.io.pwin import read_pw_input
from pypresso.response.piezo import (
    VOIGT,
    polar_direction,
    require_a_nonpolar_crystal,
    to_voigt,
)
from pypresso.system import build_system

pytestmark = [pytest.mark.unit]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"


def _crystal(case: str):
    """A stand-in for a calculation: the guard reads only the crystal.

    :func:`~pypresso.response.piezo.polar_direction` searches the symmetries of
    the structure rather than reading the run's -- deliberately, since a
    response is usually run with ``nosym`` and that list would call every
    crystal polar -- so nothing here needs pseudopotentials or a converged
    state.
    """
    return SimpleNamespace(system=build_system(read_pw_input(CASES / f"{case}.in")))


def test_voigt_carries_no_factor_of_two():
    """``e_iJ = e_(i)jk``: the engineering two is on the strain, not on this.

    ``P_i = sum_jk e_(i)jk eps_jk`` runs over all nine pairs, and Voigt's
    ``eps_4 = 2 eps_23`` absorbs the doubling of the two equal shear terms --
    so the coefficient is untouched. It is
    :class:`~pypresso.response.elastic.ElasticConstants`' convention one rank
    down, and the check is that a contraction gives the same answer in both.
    """
    rng = np.random.default_rng(0)
    e = rng.normal(size=(3, 3, 3))
    e = 0.5 * (e + e.transpose(0, 2, 1))
    strain = rng.normal(size=(3, 3))
    strain = 0.5 * (strain + strain.T)

    full = np.einsum("kij,ij->k", e, strain)
    voigt_strain = np.array([
        strain[i, j] if i == j else 2 * strain[i, j] for i, j in VOIGT
    ])
    assert np.allclose(to_voigt(e) @ voigt_strain, full)


def test_a_cubic_crystal_admits_no_spontaneous_polarization():
    """``-43m`` and ``m-3m`` both average their rotations to zero.

    A polarization has to be invariant under every operation of the point
    group, so the group average is the projector onto the directions one may
    point along. Zincblende AlAs and diamond silicon both give zero, which is
    what makes the improper-to-proper correction vanish for them.
    """
    for case in ("alas-raman", "si-electrostriction"):
        assert np.abs(polar_direction(_crystal(case))).max() < 1e-10
        require_a_nonpolar_crystal(_crystal(case))
