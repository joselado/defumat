"""``wfcinit``'s count for a fully-relativistic dataset, and why it matters.

The noncollinear atomic starting set is ``sum (2j + 1)`` over the dataset's
radial functions when it carries ``j`` channels (``atomic_wfc_so``), and
``sum 2 (2l + 1)`` when it does not (``atomic_wfc_nc``, one scalar orbital used
in each spin component). Those are different numbers -- 12 against 22 for
``Pt.rel-pz-n-rrkjus`` -- and the difference is **not** one of conditioning.

``wfcinit`` tops the atomic set up to ``nbnd`` with *random* vectors, and only
if it is short. Building 22 where QE builds 12 therefore does not merely start
somewhere else: at ``nbnd = 18`` it removes the six random vectors entirely,
and with them the only part of the starting span with generic angular
character. A state that no atomic orbital of the dataset can represent is then
unreachable -- Davidson's residual correction cannot leave a subspace the
Hamiltonian preserves -- and the run converges to the wrong occupied manifold
and reports success.

That is not hypothetical: ``Pt.rel-pz-n-rrkjus`` has ``6P`` channels with a
negative occupation, which ``n_atom_wfc`` skips, so platinum's band 11 at ``X``
is exactly orthogonal to every atomic orbital either code builds. The
regression that measures the energy is in ``tests/regression/test_spinorbit.py``.
"""

from pathlib import Path

import numpy as np
import pytest

from defumat import Calculator
from defumat.pseudo.atomic import (
    atomic_wavefunctions,
    count_spinor_wavefunctions,
)

pytestmark = pytest.mark.unit

CASE = Path("tests/data/qe/pt-soc-nosym.in")


@pytest.fixture(scope="module")
def _calculation(pytestconfig):
    path = pytestconfig.rootpath / CASE
    if not path.is_file():
        pytest.skip(f"{CASE.name} not present")
    return Calculator.from_file(
        path, pseudo_dir=pytestconfig.rootpath / "tests/data/pseudo",
        announce=False,
    ).calculation


def test_the_relativistic_count_is_not_the_doubled_scalar_one(_calculation):
    """``sum (2j + 1)`` against ``sum 2 (2l + 1)``: 12 against 22.

    ``pw.x`` prints "Starting wfcs are 12 randomized atomic wfcs + 6 random
    wfcs" for this input, which is the number this asserts.
    """
    calculation = _calculation
    scalar = atomic_wavefunctions(
        calculation.pseudos, calculation.system.structure, calculation.system.cell,
        calculation.basis.smooth, calculation.basis.planewaves,
        calculation.basis_kpoints,
    )
    doubled = 2 * scalar.shape[1]
    relativistic = count_spinor_wavefunctions(
        calculation.pseudos, calculation.system.structure, lspinorb=True
    )
    assert doubled == 22
    assert relativistic == 12


def test_a_relativistic_start_leaves_room_for_the_random_vectors(_calculation):
    """The consequence: at ``nbnd = 18`` the start is 12 atomic plus 6 random.

    Checked through the span rather than through the count, because it is the
    span the eigensolver sees. The atomic orbitals of this dataset are ``s`` and
    ``d`` only, so a ``p``-like state has *zero* overlap with every one of them;
    what the six random vectors buy is that such a state is reachable at all.
    """
    calculation = _calculation
    assert calculation._starts_from_spin_angle_functions()

    density = calculation.starting_density()
    potential = calculation.potential(density)
    hamiltonian = calculation.hamiltonian(
        potential.v_scf if hasattr(potential, "v_scf") else potential.total
    )
    hamiltonian = hamiltonian[0] if isinstance(hamiltonian, tuple) else hamiltonian

    nbnd = 18
    span = np.asarray(calculation.starting_wavefunctions([hamiltonian], nbnd))
    assert span.shape[-2] == nbnd
    # A Rayleigh-Ritz over 12 atomic vectors alone could not return 18
    # independent ones; that it does is the random top-up.
    vectors = span[0, 0]
    singular = np.linalg.svd(vectors, compute_uv=False)
    assert singular.min() > 1.0e-6 * singular.max()


def test_a_scalar_dataset_still_doubles(pytestconfig):
    """The other branch, so the fix is not applied where QE does not apply it.

    ``atomic_wfc_nc`` is what a dataset without ``j`` channels gets, and its
    count *is* the doubling. A spiral keeps it too -- it refuses spin-orbit
    coupling, so its dataset is never relativistic.
    """
    path = pytestconfig.rootpath / "tests/data/qe/h10-chain-noncolin.in"
    if not path.is_file():
        pytest.skip(f"{path.name} not present")
    calculation = Calculator.from_file(
        path, pseudo_dir=pytestconfig.rootpath / "tests/data/pseudo", announce=False
    ).calculation
    assert not calculation._starts_from_spin_angle_functions()
