"""The parts of the transverse spin susceptibility that need no magnon.

``PLAN.md`` P63. Everything here is checkable without converging a magnet, and
each piece is one that fails **silently** if it is wrong -- a magnon dispersion
is a smooth positive curve whichever way the umklapp is folded or the kernel's
index order is written.

Four kinds of thing live here:

* the **kernel's limit at a node**. ``f_xc^{+-} = B_xc/m`` is ``0/0`` where the
  magnetization changes sign, which an antiferromagnet does by symmetry rather
  than by accident, and its limit is the *longitudinal* kernel. Elk's
  ``tfm2213`` sets it to zero there instead. That the limit taken by
  differentiating the functional agrees with the ratio as ``zeta -> 0`` is a
  statement about the functional alone;
* the **kernel matrix's index order**. It is ``ftilde(G' - G)`` and not
  ``ftilde(G - G')``, because the perturbation and the response are both
  expanded in ``e^{-i(q+G).r}``. Since the field is real the two differ by a
  conjugation, which is invisible on any centrosymmetric cell;
* the **``k + q`` fold**, which is where the finite-``q`` machinery lives;
* the **refusals**, which are the promise that a run which starts is a run
  whose physics is there.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.scf.driver import Calculation
from defumat.system import build_system
from defumat.tddft.spinchi0 import (
    commensurate_shift,
    require_a_transverse_regime,
    spin_response_sphere,
)
from defumat.tddft.spinkernel import node_limit, transverse_kernel_matrix
from defumat.workflows.magnons import _peak
from defumat.xc.functional import get_functional

pytestmark = [pytest.mark.unit]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


def _calculation(name: str) -> Calculation:
    system = build_system(read_pw_input(CASES / name))
    pseudos = tuple(
        read_upf(PSEUDO / species.pseudo_file)
        for species in system.structure.species
    )
    return Calculation(system, pseudos)


# --- the kernel's limit at a node --------------------------------------------


@pytest.mark.parametrize("name", ["pz", "pw"])
def test_the_node_limit_is_what_the_ratio_tends_to(name):
    """``B_xc/m -> dB_xc/dm`` as ``m -> 0``, and not to zero.

    Elk's ``tfm2213`` zeroes the transverse kernel below ``|m| = 1e-14``. The
    limit is the *longitudinal* entry -- the two kernels coincide at a node --
    and an antiferromagnet's nodal planes are exact, so the difference is not
    confined to a set of measure zero once a grid is involved.
    """
    functional = get_functional(name)
    charge = jnp.asarray([0.01, 0.1, 1.0])
    limit = np.asarray(node_limit(functional, charge))

    previous = None
    for zeta in (1.0e-2, 1.0e-3, 1.0e-4):
        magnetization = zeta * charge
        potential = functional.spin_potential(
            jnp.stack([(charge + magnetization) / 2, (charge - magnetization) / 2])
        )
        ratio = np.asarray((potential[0] - potential[1]) / (2 * magnetization))
        error = np.abs(ratio - limit).max()
        if previous is not None:
            assert error < previous
        previous = error
    assert previous < 1.0e-6
    # ...and it is nowhere near zero, which is the value Elk would use.
    assert np.abs(limit).min() > 0.1


# --- the response sphere ------------------------------------------------------


def test_the_sphere_contains_the_origin_and_starts_there():
    """``G = 0`` is an ordinary entry here, unlike the charge channel's head.

    There is no Coulomb interaction in the transverse channel, so nothing
    diverges at the origin and the magnon is read off exactly that element.
    """
    calculation = _calculation("h-chain-afm.in")
    sphere = spin_response_sphere(calculation, 4.0)
    miller = np.asarray(sphere.miller)
    assert sphere.nm == len(miller)
    assert np.array_equal(miller[0], [0, 0, 0])
    # A sphere is closed under inversion whatever the crystal's symmetry.
    entries = {tuple(row) for row in miller}
    assert all(tuple(-row) in entries for row in miller)


def test_a_negative_cutoff_is_refused():
    calculation = _calculation("h-chain-afm.in")
    with pytest.raises(ValueError, match="cannot be negative"):
        spin_response_sphere(calculation, -1.0)


# --- the kernel matrix --------------------------------------------------------


def test_the_kernel_matrix_is_the_transposed_difference():
    """``F(G, G') = ftilde(G' - G)``, and it is Hermitian because ``F`` is real.

    The two orderings differ by a conjugation -- ``ftilde(-K) = conj(ftilde(K))``
    for a real field -- so a centrosymmetric cell cannot tell them apart. What
    can is the definition itself, checked here against a transform taken by
    hand.
    """
    calculation = _calculation("h-chain-afm.in")
    density = jnp.asarray(calculation.starting_density())
    sphere = spin_response_sphere(calculation, 2.0)
    matrix = np.asarray(transverse_kernel_matrix(calculation, density, sphere))

    assert np.allclose(matrix, matrix.conj().T, atol=1e-12)

    from defumat.tddft.spinkernel import transverse_kernel_field

    field = np.asarray(transverse_kernel_field(calculation, density))
    grid = np.asarray(calculation.basis.dense.grid)
    coefficients = np.fft.fftn(field) / float(np.prod(grid))
    miller = np.asarray(sphere.miller)
    for a in range(min(sphere.nm, 6)):
        for b in range(min(sphere.nm, 6)):
            index = tuple((miller[b] - miller[a]) % grid)
            assert matrix[a, b] == pytest.approx(coefficients[index], abs=1e-14)


# --- the k + q fold -----------------------------------------------------------


def test_the_fold_at_zero_is_the_identity():
    calculation = _calculation("h-chain-afm.in")
    index, umklapp = commensurate_shift(calculation, (0.0, 0.0, 0.0))
    assert np.array_equal(index, np.arange(len(index)))
    assert np.array_equal(umklapp, np.zeros_like(umklapp))


def test_a_reciprocal_lattice_vector_folds_every_point_back():
    """``q = G`` maps the grid onto itself with a uniform umklapp.

    This is the whole content of the shift the matrix element gathers at, and
    it is the piece that is silent: without it a dispersion is smooth and wrong
    wherever ``k + q`` leaves the first zone, which is most of the grid.
    """
    calculation = _calculation("h-chain-afm.in")
    index, umklapp = commensurate_shift(calculation, (0.0, 0.0, 1.0))
    assert np.array_equal(index, np.arange(len(index)))
    assert np.array_equal(umklapp, np.tile([0, 0, 1], (len(index), 1)))


def test_a_wavevector_off_the_grid_is_refused():
    """``q`` has to be a difference of two k-points, which is Elk's ``vecql``
    commensurability reached by construction rather than by a check."""
    calculation = _calculation("h-chain-afm.in")
    with pytest.raises(NotImplementedError, match="does not map the k-set"):
        commensurate_shift(calculation, (0.0, 0.0, 0.137))


def test_every_difference_of_two_k_points_is_admissible():
    calculation = _calculation("h-chain-afm.in")
    points = np.asarray(calculation.system.kpoints.crystal(calculation.system.cell))
    for a in points:
        for b in points:
            index, umklapp = commensurate_shift(calculation, a - b)
            assert len(index) == len(points)


# --- the refusals -------------------------------------------------------------


def test_an_unpolarized_run_is_refused():
    calculation = _calculation("si2-nosym.in")
    with pytest.raises(NotImplementedError, match="needs nspin = 2"):
        require_a_transverse_regime(calculation)


def test_a_noncollinear_run_is_refused():
    calculation = _calculation("h-chain-spiral.in")
    with pytest.raises(NotImplementedError, match="noncollinear"):
        require_a_transverse_regime(calculation)


def test_a_polarized_collinear_run_is_accepted():
    require_a_transverse_regime(_calculation("h-chain-afm.in"))


# --- the peak finder ----------------------------------------------------------


def test_a_maximum_at_the_edge_is_not_a_magnon():
    """A peak at the end of the frequency grid means the grid misses the pole.

    Reporting the edge instead is how a dispersion acquires a flat branch that
    is nothing but the window it was computed in.
    """
    frequencies = np.linspace(0.0, 1.0, 11)
    assert _peak(frequencies, np.exp(-frequencies)) is None
    assert _peak(frequencies, np.exp(frequencies)) is None


def test_the_peak_is_refined_between_samples():
    frequencies = np.linspace(0.0, 1.0, 11)
    weight = np.exp(-((frequencies - 0.53) ** 2) / 0.01)
    assert _peak(frequencies, weight) == pytest.approx(0.53, abs=0.01)
