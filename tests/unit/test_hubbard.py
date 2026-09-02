"""P20 unit checks: the pieces of DFT+U, each against something independent.

The expensive comparison against Quantum ESPRESSO lives in
``tests/regression/test_ldau.py``. What is here needs no SCF: the manifold
resolution, the potential as ``jax.grad`` against QE's hand-derived expression,
the starting occupation matrix, the group average, and the refusals.
"""

import copy

import numpy as np
import pytest

import jax.numpy as jnp

from defumat.hubbard.energy import (
    coefficients_from_setup,
    hubbard_energy,
    hubbard_potential,
    ns_ddot,
    qe_hubbard_potential,
)
from defumat.hubbard.manifold import (
    HubbardInput,
    build_hubbard_setup,
    manifold_label,
    parse_manifold,
    reference_occupation,
)
from defumat.hubbard.occupations import (
    adjust_ns,
    initial_ns,
    ns_shape,
    spin_averaged_ns,
    uniform_ns,
)
from defumat.hubbard.operator import block_potential
from defumat.hubbard.projectors import lowdin_transform
from defumat.io.pwin import parse_pw_input
from defumat.pseudo import read_upf
from defumat.system import build_system
from defumat.units import RY_TO_EV

pytestmark = pytest.mark.unit


FEO_INPUT = """
 &control
    calculation = 'scf'
 /
 &system
    ibrav = 0, celldm(1) = 8.19, nat = 4, ntyp = 3,
    ecutwfc = 30.0, ecutrho = 240.0, nspin = 2,
    starting_magnetization(2) = 0.5, starting_magnetization(3) = -0.5,
    occupations = 'smearing', smearing = 'gauss', degauss = 0.01,
    hubbard_occ(2,1) = 6.0d0
    hubbard_occ(3,1) = 6.0d0
 /
CELL_PARAMETERS {alat}
0.50 0.50 1.00
0.50 1.00 0.50
1.00 0.50 0.50
ATOMIC_SPECIES
 O    1.  O.pz-rrkjus.UPF
 Fe1  1.  Fe.pz-nd-rrkjus.UPF
 Fe2  1.  Fe.pz-nd-rrkjus.UPF
ATOMIC_POSITIONS {crystal}
 O   0.25 0.25 0.25
 O   0.75 0.75 0.75
 Fe1 0.0  0.0  0.0
 Fe2 0.5  0.5  0.5
K_POINTS {automatic}
2 2 2 0 0 0
HUBBARD {atomic}
U Fe1-3d 4.3
U Fe2-3d 4.3
"""


@pytest.fixture(scope="module")
def feo(pseudo_dir):
    system = build_system(parse_pw_input(FEO_INPUT))
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    return system, pseudos, build_hubbard_setup(
        system.hubbard, system.structure, pseudos
    )


def test_manifold_spelling():
    assert parse_manifold("Fe1-3d") == ("Fe1", 3, 2)
    assert parse_manifold("Co-4f") == ("Co", 4, 3)
    assert manifold_label(3, 2) == "3D"
    with pytest.raises(ValueError):
        parse_manifold("Fe1-3x")
    with pytest.raises(ValueError):
        parse_manifold("3d")


def test_card_values_are_electronvolts(feo):
    """The HUBBARD card writes eV; everything internal is Ry (rule R6)."""
    _, _, setup = feo
    assert setup.parameter("u") == pytest.approx(4.3 / RY_TO_EV)


def test_offsets_follow_the_atomic_orbital_order(feo):
    """``offsetU`` and ``oatwfc``: the two indexings QE keeps apart.

    Oxygen contributes ``2s + 2p`` = 4 orbitals and iron ``4s + 3d`` = 6, so the
    3d manifold of the first iron starts at 4 + 4 + 1 = 9 in the full list and
    at 0 in the Hubbard-only one.
    """
    _, _, setup = feo
    assert setup.atoms == (2, 3)
    assert setup.ldims == (5, 5)
    assert setup.offsets == (0, 5)
    assert setup.atomwfc_offsets == (9, 15)
    assert setup.nwfcU == 10


def test_reference_occupation_from_the_file(pseudo_dir):
    """``determine_hubbard_occ`` reads the manifold's occupation out of the UPF."""
    iron = read_upf(pseudo_dir / "Fe.pz-nd-rrkjus.UPF")
    assert reference_occupation(iron, 3, 2) == pytest.approx(7.0)
    with pytest.raises(ValueError, match="no 4F orbital"):
        reference_occupation(iron, 4, 3)


def test_hubbard_occ_overrides_the_file(feo):
    """``hubbard_occ(2,1) = 6.0`` wins over the file's 7.0 -- the benchmark's own choice."""
    _, _, setup = feo
    assert [setup.species[t].occupation for t in setup.types] == [6.0, 6.0]


def test_initial_ns_follows_hunds_rule(feo):
    """``init_ns``: majority channel filled first, and QE prints the traces.

    The reference output of ``pw_lda+U/lda+U.in`` reports
    ``Tr[ns] (up, down, total) = 5.00000 1.00000 6.00000`` at the first
    iteration, for six electrons in a five-fold manifold.
    """
    system, _, setup = feo
    ns = initial_ns(setup, 2, system.starting_magnetization)
    traces = np.einsum("snmm->sn", np.asarray(ns))
    assert traces[:, 0] == pytest.approx([5.0, 1.0])
    assert traces[:, 1] == pytest.approx([1.0, 5.0])


@pytest.mark.parametrize("nspin", [1, 2])
@pytest.mark.parametrize(
    "u, j0, alpha, beta",
    [(4.3, 0.0, 0.0, 0.0), (5.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.1, 0.0),
     (3.0, 0.5, 0.2, 0.3), (0.0, 0.7, 0.0, 0.4)],
)
def test_potential_is_the_gradient_of_the_energy(feo, nspin, u, j0, alpha, beta):
    """``jax.grad`` of the energy against ``v_hubbard`` transcribed from QE.

    The two share nothing but the occupation matrix: one differentiates a
    written-down functional, the other is the Fortran's loop nest.
    """
    _, _, shared = feo
    # A *copy*: the fixture is module-scoped and the parameters below would
    # otherwise leak into every test after this one.
    setup = copy.deepcopy(shared)
    for item in setup.species:
        if item is not None:
            item.u, item.j0, item.alpha, item.beta = u, j0, alpha, beta
    coefficients = coefficients_from_setup(setup)

    rng = np.random.default_rng(20)
    block = rng.normal(size=(nspin, setup.nslot, setup.ldmx, setup.ldmx))
    ns = jnp.asarray(0.5 * (block + block.transpose(0, 1, 3, 2)))

    computed = np.asarray(hubbard_potential(ns, coefficients))
    assert computed == pytest.approx(qe_hubbard_potential(ns, setup), abs=1e-12)


def test_unpolarized_energy_is_doubled_and_the_potential_is_not(feo):
    """The factor of two QE applies to ``eth`` and never to ``v_hub``."""
    _, _, setup = feo
    coefficients = coefficients_from_setup(setup)
    rng = np.random.default_rng(3)
    block = rng.normal(size=(1, setup.nslot, setup.ldmx, setup.ldmx))
    ns = jnp.asarray(0.5 * (block + block.transpose(0, 1, 3, 2)))

    doubled = float(hubbard_energy(ns, coefficients))
    # The same occupations in two identical channels: two channels' worth of
    # energy, which is what the doubling stands for.
    pair = jnp.concatenate([ns, ns])
    assert doubled == pytest.approx(float(hubbard_energy(pair, coefficients)))
    # ...while the potential is per channel and identical in the two.
    one = np.asarray(hubbard_potential(ns, coefficients))
    two = np.asarray(hubbard_potential(pair, coefficients))
    assert one[0] == pytest.approx(two[0])


def test_energy_of_an_idempotent_occupation_is_zero(feo):
    """Dudarev's functional penalises fractional occupation and nothing else.

    ``E = U/2 sum Tr[n - n n]`` vanishes when every eigenvalue of ``n`` is 0 or
    1, which is the whole physical content of the correction: an integer-filled
    manifold costs nothing and a half-filled one costs the most.
    """
    _, _, setup = feo
    coefficients = coefficients_from_setup(setup)
    ns = np.zeros((2, setup.nslot, setup.ldmx, setup.ldmx))
    ns[0, :, np.arange(5), np.arange(5)] = 1.0
    assert float(hubbard_energy(jnp.asarray(ns), coefficients)) == pytest.approx(0.0)

    # Half filling is the maximum: (U/2)(Tr n - Tr n^2) per channel per atom,
    # which for n = I/2 in a five-fold manifold is (U/2)(2.5 - 1.25).
    ns[:] = 0.0
    ns[:, :, np.arange(5), np.arange(5)] = 0.5
    peak = float(hubbard_energy(jnp.asarray(ns), coefficients))
    assert peak == pytest.approx(2 * 0.5 * 1.25 * setup.parameter("u").sum(), rel=1e-12)


def test_ns_adj_replaces_eigenvalues_and_keeps_eigenvectors(feo):
    """``ns_adj``: the requested level is imposed, the orbital it belongs to is not."""
    system, _, shared = feo
    setup = copy.deepcopy(shared)
    setup.starting_ns = {(1, 1, 4): 1.0, (2, 0, 4): 1.0}
    ns = initial_ns(setup, 2, system.starting_magnetization)
    adjusted = np.asarray(adjust_ns(ns, setup))

    # slot 0 is species 1 (Fe1): its down channel's fifth eigenvalue becomes 1.
    values = np.linalg.eigvalsh(adjusted[1, 0, :5, :5])
    assert values[4] == pytest.approx(1.0)
    # The untouched channel is unchanged.
    assert adjusted[0, 0, :5, :5] == pytest.approx(np.asarray(ns)[0, 0, :5, :5])


def test_block_potential_is_block_diagonal(feo):
    """The per-atom blocks land on the diagonal of one ``(nwfcU, nwfcU)`` matrix."""
    _, _, setup = feo
    spin, slot, row, column = setup.block_indices()
    block_row, block_column = setup.padded_indices()
    indices = (jnp.asarray(spin), jnp.asarray(slot),
               jnp.asarray(row), jnp.asarray(column),
               jnp.asarray(block_row), jnp.asarray(block_column), setup.nwfcU)

    rng = np.random.default_rng(7)
    v = jnp.asarray(rng.normal(size=(2, setup.nslot, setup.ldmx, setup.ldmx)))
    matrix = np.asarray(block_potential(v, indices))

    assert matrix.shape == (2, 10, 10)
    assert matrix[:, :5, :5] == pytest.approx(np.asarray(v)[:, 0, :5, :5])
    assert matrix[:, 5:, 5:] == pytest.approx(np.asarray(v)[:, 1, :5, :5])
    assert np.abs(matrix[:, :5, 5:]).max() == 0.0


def test_operator_matrix_and_application_agree():
    """``vhpsi`` as a contraction and as an explicit matrix are one operator.

    The dense reference fixture (``tests/exact_reference.py``) solves
    :meth:`Hamiltonian.matrix` while the SCF solves :meth:`Hamiltonian.apply`.
    If the Hubbard term reached only one of them the two would diagonalise
    different Hamiltonians -- and every existing consistency test would still
    pass, because none of them switches a U on.
    """
    from defumat.hubbard.operator import HubbardTerm

    rng = np.random.default_rng(5)
    wfcU = jnp.asarray(
        rng.normal(size=(1, 30, 4)) + 1j * rng.normal(size=(1, 30, 4))
    )
    block = rng.normal(size=(4, 4))
    term = HubbardTerm(wfcU=wfcU, vns=jnp.asarray(0.5 * (block + block.T)))

    psi = jnp.asarray(rng.normal(size=(6, 30)) + 1j * rng.normal(size=(6, 30)))
    applied = np.asarray(term.apply(psi, 0))
    through_matrix = np.asarray(psi @ term.matrix(0).T)
    assert applied == pytest.approx(through_matrix, abs=1e-12)
    # ...and the diagonal the preconditioner uses is that matrix's diagonal.
    assert np.asarray(term.diagonal(0)) == pytest.approx(
        np.diag(np.asarray(term.matrix(0))).real, abs=1e-12
    )


def test_lowdin_transform_orthonormalises():
    """``O^{-1/2}`` applied to a set of vectors makes their overlap the identity."""
    rng = np.random.default_rng(11)
    basis = rng.normal(size=(40, 6)) + 1j * rng.normal(size=(40, 6))
    overlap = jnp.asarray(basis.conj().T @ basis)
    transform = lowdin_transform(overlap)
    # The same contraction ``ortho_swfc`` performs, transposition included.
    rotated = np.asarray(jnp.asarray(basis) @ transform.T)
    assert rotated.conj().T @ rotated == pytest.approx(np.eye(6), abs=1e-10)


def test_ns_ddot_is_the_mixing_metric(feo):
    """``ns_ddot``: ``U/2 sum |dns|^2``, doubled for one spin channel."""
    _, _, setup = feo
    coefficients = coefficients_from_setup(setup)
    residual = np.zeros((2, setup.nslot, setup.ldmx, setup.ldmx))
    residual[0, 0, 0, 0] = 0.1
    expected = 0.5 * setup.parameter("u")[0] * 0.01
    assert float(ns_ddot(jnp.asarray(residual), coefficients)) == pytest.approx(expected)
    assert float(
        ns_ddot(jnp.asarray(residual[:1]), coefficients)
    ) == pytest.approx(2.0 * expected)


def test_refused_variants(pseudo_dir):
    """The parameters that select a functional this does not implement."""
    def system_with(card):
        return build_system(parse_pw_input(FEO_INPUT.replace(
            "HUBBARD {atomic}\nU Fe1-3d 4.3\nU Fe2-3d 4.3", card
        )))

    # ``J`` is no longer among them: it selects the full (Liechtenstein)
    # functional, which is implemented (P62a) and tested in
    # ``tests/unit/test_hubbard_full.py``. What it still refuses is being
    # combined with the simplified functional's own parameters.
    with pytest.raises(NotImplementedError, match="Hund J0"):
        system_with("HUBBARD {atomic}\nU Fe1-3d 4.3\nJ Fe1-3d 0.5\nJ0 Fe1-3d 0.2")
    with pytest.raises(NotImplementedError, match="intersite"):
        system_with("HUBBARD {atomic}\nU Fe1-3d 4.3\nV Fe1-3d Fe2-3d 3 4 0.5")
    with pytest.raises(NotImplementedError, match="orbital-resolved"):
        system_with("HUBBARD {atomic}\nU Fe1-3d 4.3 3 4 5")

    system = system_with("HUBBARD {pseudo}\nU Fe1-3d 4.3")
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    with pytest.raises(NotImplementedError, match="Hubbard_projectors"):
        build_hubbard_setup(system.hubbard, system.structure, pseudos)


def test_manifold_absent_from_the_dataset_is_an_error(pseudo_dir):
    system = build_system(parse_pw_input(
        FEO_INPUT.replace("U Fe1-3d 4.3", "U Fe1-4f 4.3")
    ))
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    with pytest.raises(ValueError, match="4F"):
        build_hubbard_setup(system.hubbard, system.structure, pseudos)


def test_all_zero_parameters_mean_no_correction(pseudo_dir):
    """``init_hubbard``: a species is a Hubbard species only if something is nonzero."""
    system = build_system(parse_pw_input(FEO_INPUT))
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    empty = HubbardInput(projectors="atomic", parameters=(
        ("Fe1", 3, 2, 0.0, 0.0, 0.0, 0.0), ("Fe2", 3, 2, 0.0, 0.0, 0.0, 0.0),
    ))
    assert build_hubbard_setup(empty, system.structure, pseudos) is None
    assert build_hubbard_setup(None, system.structure, pseudos) is None


def test_atomic_orbitals_are_renormalised(pseudo_dir):
    """``upf_check_atwfc_norm``: <chi|S|chi> = 1 after reading, not before.

    QE prints ``wavefunction(s) 4S renormalized`` for both iron and nickel, so
    the file's own 4s is *not* normalised in the generalised metric. The DFT+U
    ortho-atomic projectors are where that matters -- see
    :func:`defumat.pseudo.upf._renormalize_orbitals`.
    """
    from defumat.pseudo.radial import simpson_weights

    pseudo = read_upf(pseudo_dir / "Ni.pz-nd-rrkjus.UPF")
    weights = np.asarray(simpson_weights(pseudo.rab))
    q = np.asarray(pseudo.augmentation.q)
    for orbital in pseudo.orbitals:
        chi = np.asarray(orbital.chi)
        norm = float(np.sum(chi * chi * weights))
        overlaps = np.zeros(len(pseudo.projectors))
        for i, projector in enumerate(pseudo.projectors):
            if projector.l != orbital.l:
                continue
            cut = projector.cutoff_index
            overlaps[i] = float(np.sum(projector.beta[:cut] * chi[:cut] * weights[:cut]))
        norm += float(overlaps @ q @ overlaps)
        assert norm == pytest.approx(1.0, abs=1e-6)


# --- a starting occupation matrix that is not Hund's rule -------------------
#
# ``init_ns`` reads Hund's rule off ``starting_magnetization``, so on a magnetic
# species it is strongly spin-polarised however small that number is -- which
# means the *unpolarised* start, the one a run aimed at the non-magnetic
# solution needs, cannot be reached by turning a knob down. These are the
# builders for it; ``tests/regression/test_scf_solvers.py`` is where one is used
# to reach a saddle.


@pytest.mark.unit
def test_hunds_rule_start_is_polarised_whatever_the_magnetization(feo):
    _, _, feo = feo
    """The premise. If this ever stops being true the builders below are
    pointless, so it is asserted rather than assumed."""
    strong = np.asarray(initial_ns(feo, 2, [1.0, 1.0]))
    weak = np.asarray(initial_ns(feo, 2, [0.01, 0.01]))
    assert np.allclose(strong, weak)
    assert not np.allclose(strong[0], strong[1])


@pytest.mark.unit
def test_uniform_ns_is_unpolarised_and_keeps_the_electron_count(feo):
    _, _, feo = feo
    ns = np.asarray(uniform_ns(feo, 2))
    assert ns.shape == ns_shape(feo, 2)
    assert np.allclose(ns[0], ns[1])
    hund = np.asarray(initial_ns(feo, 2, [1.0, 1.0]))
    # Same electrons in the manifold, differently arranged.
    assert np.trace(ns.sum(axis=0)[0]) == pytest.approx(np.trace(hund.sum(axis=0)[0]))
    # Diagonal, and every orbital of the manifold equal.
    diagonal = np.diagonal(ns[0, 0])
    filled = diagonal[diagonal > 0]
    assert np.allclose(filled, filled[0])
    assert np.allclose(ns[0, 0] - np.diag(diagonal), 0.0)


@pytest.mark.unit
def test_uniform_ns_takes_an_occupation_override(feo):
    _, _, feo = feo
    ns = np.asarray(uniform_ns(feo, 2, occupation=2.0))
    assert np.trace(ns.sum(axis=0)[0]) == pytest.approx(2.0)


@pytest.mark.unit
def test_uniform_ns_matches_init_ns_where_init_ns_is_unpolarised(feo):
    """``ns`` is per spin channel even when there is one of them, so the
    divisor is 2 and not ``nspin``. ``init_ns`` with no magnetization is the
    same unpolarised matrix, for both ``nspin``, and is the reference here --
    dividing by ``nspin`` instead would double the ``nspin = 1`` start and
    nothing else in the suite would notice."""
    _, _, feo = feo
    for nspin in (1, 2):
        assert np.allclose(
            np.asarray(uniform_ns(feo, nspin)),
            np.asarray(initial_ns(feo, nspin, [0.0, 0.0])),
        )


@pytest.mark.unit
def test_spin_averaged_ns_removes_spin_and_keeps_orbital_structure(feo):
    _, _, feo = feo
    rng = np.random.default_rng(0)
    ns = np.asarray(initial_ns(feo, 2, [1.0, -1.0]))
    ns = ns + 0.01 * rng.normal(size=ns.shape)
    averaged = np.asarray(spin_averaged_ns(ns))
    assert np.allclose(averaged[0], averaged[1])
    assert np.allclose(averaged.sum(axis=0), ns.sum(axis=0))


@pytest.mark.unit
def test_spin_averaged_ns_is_the_identity_without_spin(feo):
    _, _, feo = feo
    ns = initial_ns(feo, 1, [0.0, 0.0])
    assert np.allclose(np.asarray(spin_averaged_ns(ns)), np.asarray(ns))
