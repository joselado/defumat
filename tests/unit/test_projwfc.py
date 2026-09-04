"""P8 (projwfc) unit checks: the channel table, the weighted schemes, the grid.

Nothing here runs an SCF. What is checked is what the projected density of
states *is* independently of any reference:

* a projection weight of one gives back the plain density of states, to
  round-off, in **both** scheme families -- which is the whole reason the
  weights go through the same registry rather than into a second implementation;
* the channel table is the pseudopotential's own orbitals, in ``ylmr2``'s ``m``
  order, with the labels ``partialdos`` builds its file names from;
* Bloechl's corrected weights are refused for a projection rather than being
  differentiated into a wrong answer;
* ``partialdos``'s energy grid is ``dos.f90``'s plus exactly one point.
"""

import numpy as np
import pytest

from defumat.projwfc.projections import PROJECTION_KINDS, atomic_projections
from defumat.projwfc.channels import (
    L_LABELS,
    M_LABELS,
    channel_table,
    projection_channels,
)
from defumat.pseudo import read_upf
from defumat.scf.tetrahedra import build_tetrahedra, tetrahedron_projected_dos
from defumat.system.kpoints import DEGSPIN, monkhorst_pack
from defumat.workflows.dos import compute_dos, energy_grid, get_dos_scheme
from defumat.workflows.pdos import lowdin_charges, partial_energy_grid

pytestmark = pytest.mark.unit


def _free_electron(n: int, nbnd: int = 1):
    points, weights = monkhorst_pack((n, n, n))
    bands = np.sum(points**2, axis=1)[:, None] + 0.3 * np.arange(nbnd)[None, :]
    return bands, weights * DEGSPIN


def _tetrahedra(kind: str, n: int):
    return build_tetrahedra(
        kind, (n, n, n), (0, 0, 0), np.eye(3, dtype=int)[None], np.eye(3),
        time_reversal=False,
    )


# --------------------------------------------------------------------------
# The identity the shared registry buys
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", ["gaussian", "mp", "cold", "fermi-dirac"])
def test_a_unit_projection_reproduces_the_smearing_dos(scheme):
    """``sum_p D_p = D`` when the projections sum to one, to round-off."""
    eigenvalues, weights = _free_electron(4, nbnd=3)
    energies = np.linspace(-0.2, 1.2, 61)
    projections = np.zeros(eigenvalues.shape + (4,))
    # Four channels sharing each band's weight unequally, summing to one.
    projections[..., :] = np.array([0.1, 0.2, 0.3, 0.4])

    plain = get_dos_scheme(scheme)(eigenvalues, weights, energies, degauss=0.05)
    projected = get_dos_scheme(scheme)(
        eigenvalues, weights, energies, degauss=0.05, projections=projections
    )
    for reference, weighted in zip(plain, projected):
        assert np.asarray(weighted).shape == (energies.size, 4)
        assert np.abs(np.asarray(weighted).sum(axis=1) - np.asarray(reference)).max() < 1e-12


@pytest.mark.parametrize("kind", ["linear", "optimized"])
def test_a_unit_projection_reproduces_the_tetrahedron_dos(kind):
    """The same identity for the tetrahedra, where it is less obvious.

    The projected version resolves the occupation weight *per corner* before it
    differentiates, where :func:`tetrahedron_dos` differentiates one occupied
    fraction. That the two agree with a unit weight is the statement that the
    corner weights sum to that fraction -- ``opt_tetra_weights_only`` against
    ``opt_tetra_dos_t``, which QE writes as two unrelated routines.
    """
    eigenvalues, weights = _free_electron(4, nbnd=2)
    tetrahedra = _tetrahedra(kind, 4)
    energies = np.linspace(0.0, 1.0, 41)
    projections = np.ones(eigenvalues.shape + (1,))

    dos, integrated = tetrahedron_projected_dos(
        tetrahedra, eigenvalues, weights, projections, energies
    )
    reference = compute_dos(
        eigenvalues, weights, energies,
        "tetrahedra-lin" if kind == "linear" else "tetrahedra-opt",
        tetrahedra=tetrahedra,
    )
    assert np.abs(np.asarray(dos)[:, 0] - reference.dos).max() < 1e-10
    assert np.abs(np.asarray(integrated)[:, 0] - reference.integrated).max() < 1e-10


def test_bloechls_weights_are_refused_for_a_projection():
    """``do_projwfc`` substitutes the linear method; nothing here guesses."""
    eigenvalues, weights = _free_electron(4)
    with pytest.raises(ValueError, match="Bloechl"):
        tetrahedron_projected_dos(
            _tetrahedra("bloechl", 4), eigenvalues, weights,
            np.ones(eigenvalues.shape + (1,)), np.linspace(0.0, 1.0, 5),
        )


def test_the_projected_grid_is_one_point_longer_than_the_dos_grid():
    """``partialdos`` loops ``0..ne``; ``dos.f90`` writes ``1..ndos``."""
    eigenvalues = np.array([[0.0, 0.5], [0.1, 0.7]])
    plain = energy_grid(eigenvalues, delta_e=0.01)
    projected = partial_energy_grid(eigenvalues, delta_e=0.01)
    assert projected.size == plain.size + 1
    assert np.abs(projected[: plain.size] - plain).max() < 1e-14
    assert projected[-1] == pytest.approx(plain[-1] + 0.01)


# --------------------------------------------------------------------------
# The channel table
# --------------------------------------------------------------------------


def test_silicons_channels_are_one_s_and_three_p_per_atom(pseudo_dir):
    from defumat.system.structure import Species, Structure

    pseudo = read_upf(pseudo_dir / "Si.pz-vbc.UPF")
    structure = Structure(
        positions=np.zeros((2, 3)),
        types=(0, 0),
        species=(Species("Si", 28.086, "Si.pz-vbc.UPF"),),
    )
    channels = projection_channels((pseudo,), structure)

    assert len(channels) == 8
    assert [c.m_label for c in channels[:4]] == ["s", "pz", "px", "py"]
    assert [c.atom for c in channels] == [0] * 4 + [1] * 4
    # ``wfc #`` is the orbital's index in the file, which is what a filpdos file
    # name carries: ``pdos_atm#1(Si)_wfc#2(p)``.
    assert [c.wfc for c in channels[:4]] == [1, 2, 2, 2]
    assert "state #   1" in channel_table(channels)


def test_the_m_labels_are_ylmr2s_order():
    """``print_lowdin``'s ``lm_label_global_frame``, letter for letter."""
    assert L_LABELS == ("s", "p", "d", "f")
    assert M_LABELS[1] == ("z", "x", "y")
    assert M_LABELS[2] == ("z2", "xz", "yz", "x2-y2", "xy")
    assert len(M_LABELS[3]) == 7


# --------------------------------------------------------------------------
# Löwdin charges
# --------------------------------------------------------------------------


def test_lowdin_charges_use_the_weights_that_carry_the_k_point_weight(pseudo_dir):
    """``print_proj`` multiplies by ``wg``, which is ``w_k f_kb`` and not ``f``."""
    from defumat.system.structure import Species, Structure

    pseudo = read_upf(pseudo_dir / "Si.pz-vbc.UPF")
    structure = Structure(
        positions=np.zeros((1, 3)),
        types=(0,),
        species=(Species("Si", 28.086, "Si.pz-vbc.UPF"),),
    )
    channels = projection_channels((pseudo,), structure)  # s, pz, px, py

    # Two k-points of unequal weight, one band, all of it on the s channel.
    projections = np.zeros((1, 2, 4, 1))
    projections[:, :, 0, :] = 1.0
    occupations = np.array([[[0.25], [0.75]]])

    charges = lowdin_charges(projections, occupations, channels, nat=1, nelec=1.0)
    assert charges.charges.shape == (1, 2)
    assert charges.charges[0, 0] == pytest.approx(1.0)  # s
    assert charges.charges[0, 1] == pytest.approx(0.0)  # p
    assert charges.spilling == pytest.approx(0.0)
    assert charges.total[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# What is refused
# --------------------------------------------------------------------------


def _stub(nsym: int = 1, use_symmetry: bool = False, **system):
    """The smallest object :func:`atomic_projections` inspects before it works."""
    from types import SimpleNamespace

    return SimpleNamespace(
        system=SimpleNamespace(**system),
        use_symmetry=use_symmetry,
        symmetries=SimpleNamespace(nsym=nsym),
    )


def test_a_symmetrised_spinor_projection_is_refused_by_name():
    """What is refused is ``sym_proj_so``, not the spinor projection itself.

    The projector set is implemented (``atomic_wfc_so``); what is missing is the
    SU(2) representation of each point-group operation that its group average
    needs, so the refusal names that and offers ``nosym`` as the way out.
    """
    with pytest.raises(NotImplementedError, match="sym_proj_so"):
        atomic_projections(
            _stub(noncolin=True, lspinorb=True, nsym=48, use_symmetry=True), None
        )


def test_a_noncollinear_projection_without_spin_orbit_is_refused_by_name():
    """``partialdos_nc``'s ``nspin0 = 2`` layout is not implemented.

    The orbitals for that branch exist (``atomic_wfc_nc``, an up and a down copy
    of each harmonic) and the labels carry their ``s_z``; what is missing is that
    such a run's columns are routed into an up or a down density of states by
    ``ind <= 2l+1``, where ``compute_pdos`` would bin them as one. Refused rather
    than shipped as a plausible decomposition, and there is no generated
    reference for it either.

    A *relativistic* dataset never reaches this: ``Calculation`` already refuses
    ``has_so`` without ``lspinorb`` where QE calls ``average_pp``.
    """
    with pytest.raises(NotImplementedError, match="without spin-orbit"):
        atomic_projections(_stub(noncolin=True, lspinorb=False), None)


def test_a_spinor_projection_is_not_refused_without_symmetry():
    """The guard must not fire on the path its own message recommends.

    It gets past the refusal and fails later for want of a real calculation --
    an ``AttributeError`` on the stub -- which is what says the refusal was not
    what stopped it.
    """
    with pytest.raises(Exception) as caught:
        atomic_projections(_stub(noncolin=True, lspinorb=True), None)
    assert not isinstance(caught.value, NotImplementedError)


def test_an_unknown_projector_set_is_refused():
    assert PROJECTION_KINDS[0] == "ortho-atomic"  # projwfc.x's only one
    with pytest.raises(ValueError, match="unknown projector set"):
        atomic_projections(_stub(noncolin=False), None, kind="wf")
