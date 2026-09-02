"""The spin-orbit coefficients, checked against what they must be by algebra.

``fcoef`` is the one genuinely new object a spin-orbit calculation needs, and it
is built from three small transcribed functions (``rot_ylm``, ``spinor``,
``sph_ind``) whose individual outputs are hard to eyeball. It has an exact
characterisation, though, and the tests below use it: for each radial projector,
``fcoef`` is the **orthogonal projector onto that ``(l, j)`` shell**, written in
the basis of real spherical harmonics times spin. So it must be Hermitian and
idempotent, its trace must be ``2j+1``, and the two ``j`` shells of one ``l``
must add up to the identity. Nothing short of the whole construction being right
satisfies all four.
"""

from pathlib import Path

import numpy as np
import pytest

from defumat.pseudo import read_upf
from defumat.pseudo.spinorbit import (
    LMAXX,
    SpinOrbitCoupling,
    becsum_transform,
    pauli_blocks,
    rot_ylm,
    sph_ind,
    spinor,
)

pytestmark = pytest.mark.unit

PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: The fully-relativistic datasets committed here, one per pseudopotential kind.
RELATIVISTIC = [
    "Pt.rel-pz-n-rrkjus.UPF",
    "Pt.rel-pbe-n-rrkjus.UPF",
    "Pt.rel-pbe-n-kjpaw_psl.0.1.UPF",
    "Bi.rel-pbe-dn-rrkjus_psl.1.0.0.UPF",
]


def _coupling(name: str) -> SpinOrbitCoupling:
    return SpinOrbitCoupling(read_upf(PSEUDO / name))


def _shell_projector(coupling: SpinOrbitCoupling, nb: int) -> np.ndarray:
    """``fcoef`` restricted to one radial projector, as a ``(2 nh_l, 2 nh_l)`` matrix."""
    from defumat.pseudo.spinorbit import _channel_table

    indv = _channel_table(coupling.pseudo)[0]
    rows = np.flatnonzero(indv == nb)
    block = coupling.fcoef[np.ix_(rows, rows)]  # (n, n, 2, 2)
    n = len(rows)
    return np.transpose(block, (0, 2, 1, 3)).reshape(2 * n, 2 * n)


def test_rot_ylm_is_unitary():
    u = rot_ylm()
    assert np.allclose(u @ u.conj().T, np.eye(2 * LMAXX + 1))


def test_rot_ylm_reproduces_the_standard_combinations():
    """``Y_{l,+1} = (Y_cos + i Y_sin)/sqrt2`` and ``Y_{l,-1}`` its Condon-Shortley pair."""
    u = rot_ylm()
    root = 1.0 / np.sqrt(2.0)
    assert u[LMAXX, 0] == 1.0  # m = 0 is already real
    assert np.allclose(u[LMAXX + 1, 1:3], [root, 1j * root])
    assert np.allclose(u[LMAXX - 1, 1:3], [-root, 1j * root])


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_spinor_coefficients_are_normalised(l):
    """The two components of a spin-angle function square to one.

    For every ``m_j`` that the shell actually contains -- which is why the sum
    is taken over ``m`` from ``-l-1`` to ``l`` and the states outside the shell
    come out zero rather than being excluded by hand.
    """
    for j in ([l + 0.5] if l == 0 else [l - 0.5, l + 0.5]):
        total = 0.0
        for m in range(-l - 1, l + 1):
            norm = spinor(l, j, m, 0) ** 2 + spinor(l, j, m, 1) ** 2
            assert norm in (pytest.approx(0.0), pytest.approx(1.0))
            total += norm
        assert total == pytest.approx(2 * j + 1)


@pytest.mark.parametrize("l", [1, 2, 3])
def test_sph_ind_stays_in_the_shell(l):
    for j in (l - 0.5, l + 0.5):
        for m in range(-l - 1, l + 1):
            for spin in (0, 1):
                assert -l <= sph_ind(l, j, m, spin) <= l


@pytest.mark.parametrize("name", RELATIVISTIC)
def test_fcoef_is_the_shell_projector(name):
    """Hermitian, idempotent, and of trace ``2j+1`` -- one block per projector."""
    coupling = _coupling(name)
    for nb, projector in enumerate(coupling.pseudo.projectors):
        p = _shell_projector(coupling, nb)
        assert np.allclose(p, p.conj().T), f"{name}: beta {nb} is not Hermitian"
        assert np.allclose(p @ p, p), f"{name}: beta {nb} is not idempotent"
        assert np.trace(p).real == pytest.approx(2 * projector.j + 1)


@pytest.mark.parametrize("name", RELATIVISTIC)
def test_the_two_j_shells_are_complete(name):
    """``P_{l,l-1/2} + P_{l,l+1/2} = 1``: the pair spans the whole shell.

    This is what says the spin-orbit decomposition loses nothing -- averaging
    the two back together is exactly the scalar-relativistic potential.
    """
    coupling = _coupling(name)
    by_shell = {}
    for nb, projector in enumerate(coupling.pseudo.projectors):
        by_shell.setdefault((projector.l, projector.j), _shell_projector(coupling, nb))

    for l in sorted({p.l for p in coupling.pseudo.projectors}):
        js = sorted(j for (ll, j) in by_shell if ll == l)
        total = sum(by_shell[(l, j)] for j in js)
        assert np.allclose(total, np.eye(total.shape[0])), f"{name}: l = {l} incomplete"


@pytest.mark.parametrize("name", RELATIVISTIC)
def test_dvan_so_is_hermitian(name):
    """``D`` is an observable's matrix, so it is Hermitian in ``(channel, spin)``.

    Not in the channel index alone: the spin-orbit term's off-diagonal spin
    blocks are complex and are each other's conjugate transpose, so testing the
    blocks separately would pass on a version with the two swapped.
    """
    coupling = _coupling(name)
    nh = coupling.nh
    d = np.transpose(coupling.dvan_so, (0, 2, 1, 3)).reshape(2 * nh, 2 * nh)
    assert np.allclose(d, d.conj().T)


def test_a_scalar_relativistic_species_is_spin_diagonal():
    """No ``j`` in the file means no spin-orbit coupling, and ``fcoef`` says so."""
    coupling = _coupling("Si.pz-vbc.UPF")
    assert not coupling.has_so
    identity = np.eye(coupling.nh)
    assert np.allclose(coupling.fcoef[:, :, 0, 0], identity)
    assert np.allclose(coupling.fcoef[:, :, 1, 1], identity)
    assert np.allclose(coupling.fcoef[:, :, 0, 1], 0.0)
    assert np.allclose(coupling.dvan_so[:, :, 0, 1], 0.0)
    assert np.allclose(coupling.dvan_so[:, :, 0, 0], coupling.dvan_so[:, :, 1, 1])


def test_fcoef_used_for_dvan_is_not_the_zeroed_one():
    """``init_us_1`` zeroes ``fcoef`` across radial channels *after* using it.

    ``dvan_so`` therefore has entries that ``fcoef`` does not, wherever a species
    has two radial projectors in the same ``(l, j)`` shell -- and every other
    consumer relies on the zeroed array to kill exactly those terms. Using one
    array for both gives a correct ``dvan_so`` and a silently wrong ``qq_so``,
    ``deeq_nc`` and ``becsum``, which is the failure this test exists to catch.
    """
    coupling = _coupling("Pt.rel-pz-n-rrkjus.UPF")
    from defumat.pseudo.spinorbit import _channel_table

    indv = _channel_table(coupling.pseudo)[0]
    cross = indv[:, None] != indv[None, :]
    assert np.allclose(coupling.fcoef[cross], 0.0)
    # ...and there is something there to have been zeroed: Pt has two 5d(3/2)
    # projectors, so dvan_so couples them.
    assert np.abs(coupling.dvan_so[cross]).max() > 1e-6


@pytest.mark.parametrize("nspin_mag", [1, 4])
def test_becsum_transform_matches_the_host_implementation(nspin_mag):
    """The traced contraction and the setup-time one are the same algebra.

    ``add_becsum_so`` is written twice -- once in NumPy for readability next to
    ``fcoef``, once in JAX because ``becsum`` is rebuilt inside the SCF -- and
    the index order of its two ``fcoef`` factors differs between them in the
    Fortran. This is the check that they have not drifted apart.
    """
    import jax.numpy as jnp

    coupling = _coupling("Pt.rel-pz-n-rrkjus.UPF")
    nh = coupling.nh
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(2 * nh, 2 * nh)) + 1j * rng.normal(size=(2 * nh, 2 * nh))
    hermitian = raw @ raw.conj().T  # a legitimate spin-density matrix
    becsum_nc = hermitian.reshape(nh, 2, nh, 2)

    reference = coupling.becsum_so(becsum_nc, nspin_mag)
    traced = np.asarray(
        becsum_transform(jnp.asarray(coupling.fcoef), jnp.asarray(becsum_nc)[None], nspin_mag)
    )[:, 0]
    assert np.allclose(reference, traced)


def test_pauli_blocks():
    """``(n, m) -> n I + m . sigma``, and one component means no magnetization."""
    n, mx, my, mz = 2.0, 0.3, -0.4, 0.5
    blocks = pauli_blocks(np.array([n, mx, my, mz]))
    assert blocks[0, 0] == pytest.approx(n + mz)
    assert blocks[1, 1] == pytest.approx(n - mz)
    assert blocks[0, 1] == pytest.approx(mx - 1j * my)
    assert blocks[1, 0] == pytest.approx(mx + 1j * my)
    assert np.allclose(blocks, blocks.conj().T)

    scalar = pauli_blocks(np.array([n]))
    assert np.allclose(scalar, n * np.eye(2))


@pytest.mark.parametrize("name", RELATIVISTIC)
def test_qq_so_is_hermitian_and_reduces_correctly(name):
    """``qq_so`` from a spin-independent ``qq`` is Hermitian and trace-preserving.

    ``F`` is idempotent in the combined ``(channel, spin)`` index, so
    ``Tr(F q F) = Tr(F q)`` -- an identity the sandwich must satisfy for *any*
    ``q``, and one that fails if either ``fcoef`` factor is transposed or the
    two spin sums are paired the wrong way round.
    """
    coupling = _coupling(name)
    nh = coupling.nh
    rng = np.random.default_rng(1)
    q = rng.normal(size=(nh, nh))
    q = 0.5 * (q + q.T)

    blocks = coupling.qq_so(q)
    matrix = np.transpose(blocks, (0, 2, 1, 3)).reshape(2 * nh, 2 * nh)
    assert np.allclose(matrix, matrix.conj().T)

    expected = sum(np.trace(coupling.fcoef[:, :, s, s] @ q) for s in (0, 1))
    assert np.trace(matrix).real == pytest.approx(expected.real)


def test_upf_reads_the_total_angular_momentum():
    pseudo = read_upf(PSEUDO / "Pt.rel-pz-n-rrkjus.UPF")
    assert pseudo.has_so
    assert [(p.l, p.j) for p in pseudo.projectors] == [
        (2, 1.5), (2, 1.5), (2, 2.5), (2, 2.5), (1, 0.5), (1, 1.5)
    ]
    assert all(orbital.j is not None for orbital in pseudo.orbitals)

    scalar = read_upf(PSEUDO / "Si.pz-vbc.UPF")
    assert not scalar.has_so
    assert all(p.j is None for p in scalar.projectors)


def test_relativistic_paw_keeps_the_small_component_separate():
    """``PP_AEWFC_REL`` must not be read as part of ``PP_AEWFC``.

    A prefix match on the tag returns both series interleaved -- twice as many
    partial waves as there are projectors, each attached to the wrong channel --
    and nothing downstream notices except the one-centre energy, which comes out
    tens of Ry wrong.
    """
    pseudo = read_upf(PSEUDO / "Pt.rel-pbe-n-kjpaw_psl.0.1.UPF")
    assert pseudo.paw.ae_wfc.shape == (pseudo.nbeta, pseudo.mesh)
    assert pseudo.paw.ps_wfc.shape == (pseudo.nbeta, pseudo.mesh)
    assert pseudo.paw.ae_wfc_rel.shape == (pseudo.nbeta, pseudo.mesh)
    assert not np.allclose(pseudo.paw.ae_wfc, pseudo.paw.ae_wfc_rel)

    scalar = read_upf(PSEUDO / "Si.pz-n-kjpaw_psl.0.1.UPF")
    assert scalar.paw.ae_wfc_rel is None


# --- the spinor operator ------------------------------------------------------


@pytest.fixture(scope="module")
def spinor_hamiltonian():
    """A spin-orbit Hamiltonian on QE's platinum cell, at its starting density."""
    from defumat.io.pwin import read_pw_input
    from defumat.scf.driver import Calculation
    from defumat.system import build_system

    benchmark = Path(__file__).resolve().parents[2] / "benchmarks" / "pt-so-1k.in"
    system = build_system(read_pw_input(benchmark))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    potential = calculation.potential(calculation.starting_density())
    return calculation, calculation.hamiltonian(potential.v_scf)[0]


def test_the_spinor_state_space_is_twice_as_large(spinor_hamiltonian):
    calculation, hamiltonian = spinor_hamiltonian
    assert hamiltonian.npol == 2
    assert hamiltonian.ndim == 2 * hamiltonian.npwx
    assert hamiltonian.state_mask.shape == (hamiltonian.nk, hamiltonian.ndim)
    assert hamiltonian.deeq.shape[:2] == (2, 2)
    # Nonmagnetic, so the density and the potential stay scalar.
    assert hamiltonian.nspin_mag == 1
    assert calculation.nspin_mag == 1


def test_the_two_spinor_matrix_builds_agree(spinor_hamiltonian):
    """The formula and the operator must give the same matrix, to round-off.

    The same check :mod:`tests.unit.test_operator` makes of the collinear
    Hamiltonian, and it is worth more here: the spin-block structure gives four
    chances to transpose an index, and three of the four blocks are complex.
    """
    _, hamiltonian = spinor_hamiltonian
    direct = np.asarray(hamiltonian.matrix(0))
    applied = np.asarray(hamiltonian.matrix_by_application(0))
    scale = np.abs(applied).max()
    assert np.abs(direct - applied).max() < 1e-10 * scale


def test_the_spinor_hamiltonian_is_hermitian(spinor_hamiltonian):
    """...and genuinely spin-mixing, which is what makes it a spin-orbit one."""
    _, hamiltonian = spinor_hamiltonian
    matrix = np.asarray(hamiltonian.matrix_by_application(0))
    assert np.abs(matrix - matrix.conj().T).max() < 1e-10 * np.abs(matrix).max()

    npwx = hamiltonian.npwx
    upper_lower = matrix[:npwx, npwx:]
    assert np.abs(upper_lower).max() > 1e-6, "the spin blocks do not couple at all"


def test_the_spinor_overlap_is_positive_definite(spinor_hamiltonian):
    """``S`` has to stay positive definite or the Cholesky in ``cdiaghg`` fails.

    Platinum is ultrasoft *and* fully relativistic, so ``S`` is built from
    ``qq_so`` -- the augmentation integrals put through the spin transform --
    and a sign error there is not a small perturbation of the answer but a
    generalised eigenproblem with no solution.
    """
    _, hamiltonian = spinor_hamiltonian
    overlap = np.asarray(hamiltonian.overlap_matrix(0))
    assert np.abs(overlap - overlap.conj().T).max() < 1e-10
    assert np.linalg.eigvalsh(overlap).min() > 0.0


# -- using a relativistic dataset where there is no spin-orbit coupling --------


def _bismuth_system(**overrides):
    """A two-atom cell with the committed fully-relativistic bismuth dataset."""
    import dataclasses

    import jax.numpy as jnp

    from defumat.system.builder import System
    from defumat.system.cell import Cell, latgen
    from defumat.system.kpoints import KPoints
    from defumat.system.structure import Species, Structure

    cell = Cell.from_vectors(latgen(1, [9.0, 0, 0, 0, 0, 0]), alat=9.0)
    structure = Structure(
        positions=jnp.asarray([[0.0, 0.0, 0.0]]),
        types=(0,),
        species=(Species(name="Bi", mass=209.0,
                         pseudo_file="Bi.rel-pbe-dn-rrkjus_psl.1.0.0.UPF"),),
    )
    system = System(
        cell=cell, structure=structure,
        kpoints=KPoints(coords=np.zeros((1, 3)), weights=np.ones(1)),
        ecutwfc=12.0, ecutrho=48.0, nosym=True,
    )
    return dataclasses.replace(system, **overrides) if overrides else system


@pytest.mark.parametrize(
    "spin", [{}, {"nspin": 4}], ids=["nspin=1", "noncolin without lspinorb"]
)
def test_a_relativistic_dataset_without_spin_orbit_is_refused(spin):
    """``average_pp`` is not implemented, so the combination must not run.

    **The ``nspin = 1`` half of this is the one that matters** and it is the one
    that used to slip through: the refusal was written as ``noncolin and not
    lspinorb``, where ``setup.f90`` calls ``average_pp`` in the *else* branch of
    ``IF (lspinorb)`` and so reaches every run without spin-orbit coupling.
    Picking a ``rel-`` dataset off a pseudopotential table and running an
    ordinary scalar calculation with it is the common way to meet this, and it
    used to converge and report a total energy wrong by **20 Ry** -- measured on
    rhombohedral BN against ``pw.x``, with the Ewald and dispersion terms, the
    only two that touch no projector, still agreeing to 4e-9.
    """
    from defumat.scf import Calculation

    pseudo = read_upf(PSEUDO / "Bi.rel-pbe-dn-rrkjus_psl.1.0.0.UPF")
    assert pseudo.has_so, "this test needs a fully-relativistic dataset"
    with pytest.raises(NotImplementedError, match="fully-relativistic"):
        Calculation(_bismuth_system(**spin), (pseudo,))
