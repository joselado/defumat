"""A stated texture has three consumers, and it used to reach one.

``STARTING_MOMENTS`` says which way each atom's moment points. Three things are
built from that before the first Hamiltonian exists: the charge density, the
Hubbard occupation matrix, and a PAW or ultrasoft dataset's atomic ``becsum``.
Only the charge saw it; the other two started every site of a species pointing
the same way, so iteration 1 contradicted itself on exactly the systems the card
exists for -- transition-metal magnets, which need a U and ship as PAW.

The tests are about *directions*, because the magnitudes are three different
things (a cell integral, a shell occupation, a one-centre occupation) and only
the axis has to be common.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from defumat.pseudo.upf import read_upf
from defumat.scf.driver import Calculation
from defumat.system.builder import system_from_file

QE = Path(__file__).resolve().parents[1] / "data" / "qe"
pytestmark = pytest.mark.unit


def _calculation(name, pseudo_dir):
    system = system_from_file(QE / name)
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return system, Calculation(system, pseudos)


def _direction(vector):
    vector = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(vector))
    return vector / length if length > 0 else vector


@pytest.mark.slow
def test_the_hubbard_shell_points_where_the_card_says(pseudo_dir):
    """``initial_ns_noncollinear`` took its axis per species.

    Its own docstring states the consequence: "nothing in the SCF turns a
    moment, so a spinor DFT+U run started with a moment along z on a species
    whose angle1 points elsewhere converges with the shell polarised along the
    wrong axis and reports success." A texture on one species is that
    situation for every atom but one, and noncollinear DFT+U is forced to
    ``nosym``, so this matrix is the only steering there is.
    """
    system, calculation = _calculation("n2-ldau-texture.in", pseudo_dir)
    setup = calculation.hubbard
    assert setup.nslot == 2, "one slot per correlated atom, or nothing is tested"
    assert setup.types == (0, 0), "both slots must be the SAME species"

    moments = np.asarray(system.local_moments, dtype=float)
    assert not np.allclose(
        _direction(moments[0]), _direction(moments[1])
    ), "the card must ask for two different directions"

    ns = np.asarray(calculation.starting_ns())
    for slot in range(setup.nslot):
        # ``ns[2 s1 + s2]`` is ``rho[s2, s1]`` and not ``rho[s1, s2]``: the
        # accumulation in :func:`~defumat.hubbard.occupations.occupation_matrix`
        # is ``conj(proj[s1]) proj[s2]``, whose first label is the **ket**. So
        # the entry read here as the up-down element is ``rho[1, 0]``, and
        # ``m_y`` comes out with ``+2 Im`` rather than ``-2 Im``. This decoder
        # used to use the seed's own (opposite) convention, which made the test
        # a round trip through one mistake and blind to it.
        block = np.array([[ns[0, slot, 0, 0], ns[2, slot, 0, 0]],
                          [ns[1, slot, 0, 0], ns[3, slot, 0, 0]]])
        shell = np.array([
            2.0 * block[0, 1].real,
            -2.0 * block[0, 1].imag,
            (block[0, 0] - block[1, 1]).real,
        ])
        wanted = _direction(moments[setup.atoms[slot]])
        assert _direction(shell) == pytest.approx(wanted, abs=1e-9), (
            f"slot {slot} (atom {setup.atoms[slot]}) points the wrong way"
        )


@pytest.mark.slow
def test_the_one_centre_occupations_point_where_the_card_says(pseudo_dir):
    """``_becsum_split`` was indexed by species and broadcast over the atoms.

    For PAW this is not merely a poorer guess: the one-centre terms are a
    *function* of ``becsum``, so a per-species split puts a different
    Hamiltonian on the two sites at iteration 1 than the charge density asks
    for.
    """
    system, calculation = _calculation("o2-paw-texture.in", pseudo_dir)
    assert calculation.pseudos[0].paw is not None, "this test is about PAW"
    moments = np.asarray(system.local_moments, dtype=float)

    becsum = np.asarray(calculation.starting_becsum()[0])
    assert becsum.shape[:2] == (4, 2)
    for atom in range(2):
        one_centre = np.array(
            [float(np.trace(becsum[k, atom]).real) for k in (1, 2, 3)]
        )
        assert _direction(one_centre) == pytest.approx(
            _direction(moments[atom]), abs=1e-9
        )
        assert float(np.trace(becsum[0, atom]).real) == pytest.approx(6.0, abs=1e-6)


@pytest.mark.slow
def test_the_charge_and_the_one_centre_guess_agree_per_atom(pseudo_dir):
    """The property ``spin_weights``' docstring asks for, checked per *atom*.

    "The two starting guesses have to agree about how polarized the atom is or
    the first iteration contradicts itself." That was written one species at a
    time; with a texture the statement only means anything per atom.
    """
    system, calculation = _calculation("o2-paw-texture.in", pseudo_dir)
    charge, moment = calculation.site_moments(calculation.starting_density())
    moment = np.asarray(moment)
    becsum = np.asarray(calculation.starting_becsum()[0])

    for atom in range(2):
        one_centre = np.array(
            [float(np.trace(becsum[k, atom]).real) for k in (1, 2, 3)]
        )
        assert _direction(moment[atom]) == pytest.approx(
            _direction(one_centre), abs=1e-4
        )
        # The card is in Bohr magnetons and both guesses honour it. The sphere
        # integral is a little short because the atomic charge has a tail
        # outside the integration radius, which is a property of the region and
        # not a disagreement.
        assert float(np.linalg.norm(moment[atom])) == pytest.approx(1.5, abs=1e-3)
        assert float(np.linalg.norm(one_centre)) == pytest.approx(1.5, abs=1e-9)


@pytest.mark.slow
def test_without_a_card_nothing_changes(pseudo_dir):
    """The per-species path is untouched, which is most existing inputs.

    ``bn-ldau-noncol.in`` is the committed spinor DFT+U case, validated against
    ``pw.x`` at 1.2e-7 Ry, and it has no card: its occupation matrix must still
    come from ``starting_magnetization``/``angle1``/``angle2``.
    """
    system, calculation = _calculation("bn-ldau-noncol.in", pseudo_dir)
    assert not system.starting_moments
    ns = np.asarray(calculation.starting_ns())
    block = np.array([[ns[0, 0, 0, 0], ns[1, 0, 0, 0]],
                      [ns[2, 0, 0, 0], ns[3, 0, 0, 0]]])
    shell = np.array([
        2.0 * block[0, 1].real,
        -2.0 * block[0, 1].imag,
        (block[0, 0] - block[1, 1]).real,
    ])
    # angle1 = angle2 = 0 by default, so the shell is along +z.
    assert _direction(shell) == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)


@pytest.mark.slow
def test_a_collinear_card_reaches_the_one_centre_occupations_too(pseudo_dir):
    """The regime the collinear symmetry filter opened up (P77).

    A ``STARTING_MOMENTS`` card is accepted for ``nspin = 2`` as long as x and
    y vanish, and a one-species antiferromagnet written that way now converges
    instead of being averaged to zero. That makes this branch reachable: left
    per species, a PAW or ultrasoft dataset would get a *ferromagnetic*
    ``becsum`` beside an antiferromagnetic charge -- the same iteration-1
    contradiction as the noncollinear case, one regime over, and newly so.
    """
    system, calculation = _calculation("o2-paw-afm.in", pseudo_dir)
    assert system.nspin == 2 and calculation.nspin_mag == 2
    assert calculation.pseudos[0].paw is not None
    wanted = np.asarray(system.local_moments, dtype=float)[:, 2]
    assert wanted[0] == pytest.approx(-wanted[1]), "the card must be staggered"

    becsum = np.asarray(calculation.starting_becsum()[0])
    assert becsum.shape[:2] == (2, 2)
    moments = np.array([
        float(np.trace(becsum[0, a]).real) - float(np.trace(becsum[1, a]).real)
        for a in range(2)
    ])
    assert moments[0] == pytest.approx(-moments[1], abs=1e-12), (
        "the one-centre occupations are ferromagnetic beside a staggered charge"
    )
    assert moments == pytest.approx(wanted, abs=1e-9)

    # And they agree with the charge, which is the property that matters.
    _, site = calculation.site_moments(calculation.starting_density())
    assert np.sign(np.asarray(site)[:, 0]) == pytest.approx(np.sign(moments))


@pytest.mark.slow
def test_a_collinear_card_reaches_the_hubbard_shell_too(pseudo_dir):
    """``initial_ns`` took its sign from the per-species number.

    Only the sign matters in a collinear run -- it says which channel is the
    majority one -- but it is per atom that it has to be right.
    """
    from defumat.hubbard.occupations import initial_ns

    system, calculation = _calculation("n2-ldau-texture.in", pseudo_dir)
    setup = calculation.hubbard
    staggered = np.array([+1.0, -1.0])[np.asarray(setup.atoms, dtype=int)]

    ns = np.asarray(initial_ns(setup, 2, np.array([0.5]), per_atom=staggered))
    per_slot = np.array([
        float(np.trace(ns[0, slot]) - np.trace(ns[1, slot]))
        for slot in range(setup.nslot)
    ])
    assert per_slot[0] > 0 and per_slot[1] < 0, "both shells took the same channel"

    # Without the card the per-species number decides, as it always did.
    plain = np.asarray(initial_ns(setup, 2, np.array([0.5])))
    flat = np.array([
        float(np.trace(plain[0, slot]) - np.trace(plain[1, slot]))
        for slot in range(setup.nslot)
    ])
    assert flat[0] == pytest.approx(flat[1])


def test_the_seeded_occupation_matrix_is_qes_own_init_ns_nc():
    """The packed pair is not the density matrix's index order, and it was.

    ``new_ns.f90`` accumulates ``conj(proj(m1, is1)) proj(m2, is2)``, whose
    first spin label is the **ket**, so ``nr(is1, is2)`` is
    ``<phi_{is2}| rho |phi_{is1}> = rho[is2, is1]``, and it packs that as
    ``i = 2 (is1 - 1) + is2``, which is this code's ``2 s1 + s2``.
    ``initial_ns_noncollinear`` wrote ``rho[s1, s2]`` into that slot instead.
    Transposing a Hermitian block conjugates it, so the charge, ``m_x`` and
    ``m_z`` are untouched and ``m_y`` is seeded with the **opposite sign**: a
    starting texture that is a mirror image of the one asked for, in one
    component only, on a run that is forced to ``nosym`` and where this matrix
    is the only steering there is.

    The reference is QE's ``init_ns_nc`` closed form rather than a re-derivation
    here, because a re-derivation shares the convention that is in question --
    which is what the sibling test above was doing when it decoded the seed with
    the seed's own index order and passed either way.
    """
    from defumat.hubbard.occupations import initial_ns_noncollinear

    import types as _types

    theta, phi = np.deg2rad(55.0), np.deg2rad(40.0)
    setup = _types.SimpleNamespace(
        nslot=1, types=(0,), atoms=(0,), ldmx=1,
        species=(_types.SimpleNamespace(ldim=1, occupation=1.4),),
    )
    ns = np.asarray(initial_ns_noncollinear(
        setup, starting_magnetization=[0.6],
        angle1=[np.rad2deg(theta)], angle2=[np.rad2deg(phi)],
    ))

    # QE: ns(2) = (m/2) sin(theta) e^{+i phi} at the packed pair (is1, is2) =
    # (1, 2), which is index 1 here, and ns(3) its conjugate at index 2.
    up_down = ns[1, 0, 0, 0]
    assert up_down.imag > 0.0, (
        "the packed (up, down) entry must carry e^{+i phi}, which is QE's "
        "ns(2); a negative imaginary part is rho[0, 1] written into rho[1, 0]'s "
        "slot, and it is m_y with the wrong sign"
    )
    assert np.angle(up_down) == pytest.approx(phi, abs=1e-9)
    assert ns[2, 0, 0, 0] == pytest.approx(np.conj(up_down), abs=1e-12)
