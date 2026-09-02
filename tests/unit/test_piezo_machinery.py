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

from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.response.piezo import (
    VOIGT,
    polar_direction,
    require_a_nonpolar_crystal,
    require_a_piezoelectric_tensor,
    to_voigt,
)
from defumat.scf import Calculation
from defumat.system import build_system

pytestmark = [pytest.mark.unit]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


def _crystal(case: str):
    """A stand-in for a calculation: the guard reads only the crystal.

    :func:`~defumat.response.piezo.polar_direction` searches the symmetries of
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
    :class:`~defumat.response.elastic.ElasticConstants`' convention one rank
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


def _calculation(case: str):
    """A real ``Calculation``, which is what the guard chain reads."""
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(
        read_upf(PSEUDO / sp.pseudo_file) for sp in system.structure.species
    )
    return Calculation(system, pseudos)


@pytest.mark.parametrize("case, message", [
    ("si2-us", "ultrasoft"),
    ("al-metal", "metal"),
    ("o-atom-fixed-lsda", "nspin = 2"),
    ("germanene-soc", "noncollinear"),
])
def test_the_regimes_this_was_never_run_in_are_refused(case, message):
    """Every one of these would return a number, and none of them is measured.

    The guard chain is deliberately made of the *bare* forms: the linear
    response solver runs for a metal and for two spin channels, and this
    assembly on top of it has been run with neither, so the flags that would
    say otherwise are not passed. An ultrasoft dataset is the interesting one --
    nothing in the assembly is norm-conserving, and what is missing is a
    non-centrosymmetric ultrasoft crystal to measure it on, since a
    centrosymmetric one agrees with zero whatever is wrong.
    """
    with pytest.raises(NotImplementedError, match=message):
        require_a_piezoelectric_tensor(_calculation(case))
