"""The PAW half of the piezoelectric wedge completion, which nothing exercised.

``test_piezoelectric_wedge.py`` says the wedge sum has been completed on an
**ultrasoft** dataset. The completion has a second half that only a PAW dataset
reaches: ``becsum`` enters a PAW energy directly, through the one-centre terms,
rather than only through the augmentation charge on the dense grid, so the
factor that has to be made whole before it is contracted is made whole in two
places -- :func:`~defumat.response.born._full_zone_field_response` on the grid
and :func:`~defumat.response.born._full_zone_becsum_response` in the spheres.
The second ran for nobody: the piezoelectric tensor refuses a PAW dataset at the
door, every committed PAW crystal is centrosymmetric, and a centrosymmetric
crystal's tensor is zero whatever the assembly does. Live code with no test is
what the refusal was standing in front of, and this is the test.

**The falsifier is the point of the file.** A wedge completion is identically
zero on a run with no symmetry, so a test that only ever runs *with* it cannot
tell a working completion from a deleted one -- ``CLAUDE.md``'s "a check whose
null result cannot be told from a pass". So the same wedge runs twice, with
``full_zone`` on and off, and what is asserted is the **separation** between
them rather than a single tolerance.
"""

from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.response.efield import _bare_plus_induced, dielectric_tensor
from defumat.response.electrostriction import refined_states
from defumat.response.piezo import clamped_ion_piezoelectric, to_voigt
from defumat.scf import Calculation, run_scf
from defumat.system import build_system
from defumat.units import E_BOHR2_TO_C_M2

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: The closed grid, eight k-points with no symmetry, and its wedge, three.
CLOSED = "alas-piezo-tiny-paw"
WEDGE = "alas-piezo-tiny-paw-wedge"


@pytest.fixture(autouse=True)
def _bounded_compilation():
    """Drop XLA's executables between cases -- ``CLAUDE.md``'s memory rule."""
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _field(case: str):
    """One converged ground state and one field response, per cell."""
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-10,
                     max_iterations=100)
    eigenvalues, psi = refined_states(calculation, result)
    density = jnp.asarray(result.density)
    field = dielectric_tensor(
        calculation, psi, eigenvalues, density, result.becsum,
        born_charges=False, keep_internals=True,
    )
    assert field.converged
    return calculation, result, eigenvalues, psi, density, field


def _taped(case: str, full_zone: bool = True) -> float:
    """``e_14`` in C/m^2 from the taped route, the only one PAW may use.

    ``piezoelectric_zstar_eu_style`` refuses PAW by name and for a reason that
    is not about this term, so the pair here is one route on two k-sets rather
    than two routes on one.
    """
    calculation, result, eigenvalues, psi, density, field = _field(case)
    internals = field.internals
    onecentre = internals["onecentre"]
    perturbations = [
        _bare_plus_induced(
            internals["solver"], internals["bare"][axis], internals["dvscf"][axis],
            None if onecentre is None else onecentre[axis], True,
        )
        for axis in range(3)
    ]
    e = clamped_ion_piezoelectric(
        calculation, psi, eigenvalues, jnp.asarray(internals["weights"]),
        density, result.becsum, internals["dpsi"], internals["solver"].nocc,
        solver=internals["solver"], field_perturbations=perturbations,
        commutators=internals["commutators"], full_zone=full_zone,
    )
    return float(to_voigt(e)[0, 3] * E_BOHR2_TO_C_M2)


#: ``e_14`` in C/m^2 on ``alas-piezo-tiny-paw.in``, the closed grid: eight
#: k-points, no symmetry, and therefore no completion to apply.
#:
#: **Quoted rather than recomputed**, for ``test_piezoelectric_wedge.py``'s
#: reason -- a second ground state and a second tape in this process is another
#: 14 GB -- and it is quoted safely because the completion is *identically zero*
#: on a run with no symmetry: measured on that cell, ``full_zone`` on and off
#: give 1.465022183 both, digit for digit, which is the statement that the
#: switch below changes nothing it should not.
CLOSED_GRID_E14 = 1.465022183


def test_the_paw_wedge_completes_and_the_falsifier_says_it_is_doing_it():
    """The PAW half of the completion, and a pair that can fail.

    On an augmented dataset the density moves with the strain at frozen states,
    so this mixed second derivative carries a term **quadratic** in a per-k
    tangent, and a wedge sum of a product is not the product of the full-zone
    objects. One factor has to be made whole before it is contracted rather than
    afterwards, which for PAW happens in two places: on the dense grid, where the
    augmentation charge lives, and in the one-centre terms, where ``becsum``
    reaches the energy directly and no grid integral can follow it. The second is
    :func:`~defumat.response.born._full_zone_becsum_response` and this is the
    first test it has ever had.

    Measured on this cell, three k-points against the closed grid's eight:
    **1.465023765** with the completion and **1.465718152** without, against
    1.465022183 on the closed grid. So the completion takes the wedge from
    6.96e-04 away to **1.58e-06** away, a factor of 440, and the pair is what
    makes that a measurement rather than a tolerance that would pass either way.
    """
    with_completion = _taped(WEDGE, full_zone=True)
    without = _taped(WEDGE, full_zone=False)

    completed = abs(with_completion - CLOSED_GRID_E14)
    raw = abs(without - CLOSED_GRID_E14)
    assert completed < 1.0e-5, (
        f"the completed wedge is {completed:.3e} C/m^2 from the closed grid"
    )
    # The falsifier. Without it this file would pass with the completion deleted,
    # since the term is identically zero on the ``nosym`` cell the reference
    # number comes from.
    assert raw / completed > 100.0, (
        f"the completion moved the wedge by a factor of {raw / completed:.0f} "
        "where 440 was measured: either it stopped running or the cell stopped "
        "being reduced"
    )
    assert abs(without - with_completion) > 1.0e-4

    # Three points and not eight: without the reduction both numbers above are
    # the closed grid and the test compares the cell with itself.
    calculation, *_ = _field(WEDGE)
    assert calculation.system.kpoints.nk == 3


def test_the_paw_ground_state_is_the_ultrasoft_one_at_the_same_geometry():
    """A cheap cutoff is allowed to be unphysical and not to be a different cell.

    ``ecutwfc = 10`` with ``ecutrho = 44`` is far below what
    ``Al/As.pbe-n-kjpaw_psl.1.0.0`` were made for, which is deliberate and is
    why no value here is quoted as physics. What must still hold is that the
    cell is the same problem as the ultrasoft pair beside it: the dielectric
    constant reads **42.051** here against the ultrasoft cell's 42.160, and the
    wedge and the closed grid share a ground state to 4e-09 Ry, so the identity
    above is being read on a converged state rather than on an artefact of the
    cutoff.
    """
    _, result, _, _, _, field = _field(WEDGE)
    assert result.converged
    assert float(np.trace(field.epsilon)) / 3 == pytest.approx(42.051, rel=1e-3)
