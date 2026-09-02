"""P62a unit checks: the full (Liechtenstein) functional, ``lda_plus_u_kind = 1``.

The expensive comparison against Quantum ESPRESSO lives in
``tests/regression/test_ldau.py``. What is here needs no SCF and each check
closes inside the package or against an analytic statement:

* the Coulomb matrix reduces to ``U`` for an ``s`` shell and to
  ``U delta delta`` whenever ``J = 0``, which is the whole angular assembly
  tested against one number;
* it is invariant under a **rotation**, which is what "rotationally invariant"
  names and what a wrong Gaunt normalisation breaks;
* ``J -> F^k -> J`` is a round trip, and ``F^4/F^2`` comes out at the physical
  0.625 rather than the 1.8 that ``hubbard_matrix`` alone would give;
* at ``J = 0`` the whole functional collapses onto P20's Dudarev energy and
  potential, in both spin regimes -- the reduction that catches a transposed
  index pair in ``vee``;
* the potential from ``jax.grad`` reproduces ``v_hubbard_full`` transcribed
  literally, which is the check that the ``E_dc/2`` halving for ``nspin = 1``
  is right.
"""

from dataclasses import replace

import numpy as np
import pytest
import scipy.linalg as sla

import jax.numpy as jnp

from defumat.hubbard.energy import (
    coefficients_from_setup,
    elk_amf_potential,
    hubbard_energy,
    hubbard_potential,
    qe_hubbard_full_potential,
)
from defumat.hubbard.interaction import (
    coulomb_matrix,
    default_racah,
    exchange_from_slater,
    slater_integrals,
)
from defumat.hubbard.manifold import HubbardSetup, HubbardSpecies, build_hubbard_setup
from defumat.io.pwin import parse_pw_input
from defumat.pseudo import read_upf
from defumat.pseudo.harmonics import real_spherical_harmonics
from defumat.system import build_system
from defumat.units import RY_TO_EV

pytestmark = pytest.mark.unit


def _setup(kind, l=2, u=4.3 / RY_TO_EV, j=0.0):
    """One slot, one species, nothing resolved against a structure."""
    species = HubbardSpecies(n=3, l=l, u=u, occupation=6.0)
    if kind == 1:
        species = replace(
            species, j=j, racah=default_racah(l, j, (0.0,) if l == 2 else (0.0, 0.0))
        )
    width = 2 * l + 1
    return HubbardSetup(
        species=(species,), atoms=(0,), ldims=(width,), ldmx=width,
        offsets=(0,), atomwfc_offsets=(0,), nwfcU=width, types=(0,), kind=kind,
    )


def _random_ns(nspin, width=5, seed=7):
    rng = np.random.default_rng(seed)
    block = rng.normal(size=(nspin, 1, width, width)) * 0.2
    return jnp.asarray(block + np.swapaxes(block, -1, -2) + np.eye(width) * 0.6)


# --------------------------------------------------------------------------
# The Coulomb matrix


def test_an_s_shell_interaction_is_exactly_u():
    """One orbital, one Slater integral: ``vee`` is the number ``U`` itself."""
    vee = coulomb_matrix(0, slater_integrals(0, 4.3, 0.0))
    assert vee.shape == (1, 1, 1, 1)
    assert float(vee[0, 0, 0, 0]) == pytest.approx(4.3, abs=1e-13)


@pytest.mark.parametrize("l", [1, 2, 3])
def test_without_exchange_the_matrix_is_the_hartree_one(l):
    """``F^k = 0`` for ``k > 0`` leaves ``vee(m1,m2,m3,m4) = U d_{m1m3} d_{m2m4}``.

    This is the angular assembly tested against a closed form: every Gaunt
    coefficient of the ``k = 0`` shell has to come out at ``1/sqrt(4 pi)`` and
    the ``4 pi/(2k+1)`` prefactor has to cancel it exactly.
    """
    width = 2 * l + 1
    f = slater_integrals(l, 4.3, 0.0, default_racah(l, 0.0, ()))
    expected = 4.3 * np.einsum("ac,bd->abcd", np.eye(width), np.eye(width))
    assert coulomb_matrix(l, f) == pytest.approx(expected, abs=1e-13)


@pytest.mark.parametrize("l", [1, 2, 3])
def test_the_interaction_matrix_is_rotationally_invariant(l):
    """A rotation of the crystal axes leaves ``vee`` alone.

    "Rotationally invariant" in Liechtenstein's sense is invariance under
    ``SO(3)``, not under an arbitrary unitary mixing of the manifold -- so the
    test rotates all four indices with the real-harmonic representation matrix
    of a genuine rotation, obtained from the harmonics themselves rather than
    from a table.
    """
    width = 2 * l + 1
    rng = np.random.default_rng(1)
    generator = rng.normal(size=(3, 3))
    rotation = sla.expm(generator - generator.T)

    directions = rng.normal(size=(400, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    block = slice(l * l, l * l + width)
    harmonics = real_spherical_harmonics(directions, l)[:, block]
    rotated = real_spherical_harmonics(directions @ rotation.T, l)[:, block]
    d, *_ = np.linalg.lstsq(harmonics, rotated, rcond=None)
    assert d @ d.T == pytest.approx(np.eye(width), abs=1e-12)

    j = 1.75 / RY_TO_EV
    vee = coulomb_matrix(l, slater_integrals(l, 4.3 / RY_TO_EV, j, default_racah(l, j, ())))
    turned = np.einsum("ma,nb,oc,pd,mnop->abcd", d, d, d, d, vee)
    assert turned == pytest.approx(vee, abs=1e-12)


@pytest.mark.parametrize("l", [1, 2, 3])
def test_the_exchange_comes_back_out_of_the_slater_integrals(l):
    """``J -> F^k -> J``: the coefficients of both directions, in one statement.

    Elk's ``genfdu.f90`` writes the second relation and QE's
    ``plus_u_full.f90`` the first; they are inverses, which is a check on both
    that needs neither code to be run.
    """
    j = 1.3
    f = slater_integrals(l, 2.0, j, default_racah(l, j, ()))
    assert exchange_from_slater(l, f) == pytest.approx(j, rel=1e-12)


def test_the_slater_ratio_is_the_atomic_one_not_the_bare_formula():
    """``F^4/F^2 = 0.625`` for a ``d`` shell, which is ``init_hubbard``'s doing.

    ``hubbard_matrix`` alone, with Racah's ``B`` left at zero, would give 1.8;
    ``ldaU.f90:421`` substitutes ``B = 0.114774 J`` first. QE prints the
    substituted ``B(Fe1-3d) = 0.1148`` in its own output for the case this
    number was read off, so it is a claim about the reference and not only
    about a formula.
    """
    j = 1.0
    (b,) = default_racah(2, j, (0.0,))
    assert b == pytest.approx(0.114774114774 * j, rel=1e-12)
    f = slater_integrals(2, 4.3, j, (b,))
    assert f[4] / f[2] == pytest.approx(0.625, rel=1e-11)
    # and a B that is given is kept, however small -- QE tests == 0, not a
    # tolerance, so a deliberate 1e-12 survives.
    assert default_racah(2, j, (1e-12,)) == (1e-12,)


def test_an_f_shell_takes_e2_and_e3_and_a_d_shell_takes_b():
    assert len(default_racah(3, 1.0, ())) == 2
    assert default_racah(1, 1.0, ()) == ()
    with pytest.raises(NotImplementedError, match="l = 4"):
        slater_integrals(4, 1.0, 0.0)


# --------------------------------------------------------------------------
# The functional


@pytest.mark.parametrize("nspin", [1, 2])
def test_without_exchange_the_full_functional_is_dudarev(nspin):
    """The reduction that closes inside the package.

    With ``J = 0`` only ``F^0`` survives, ``vee`` is ``U delta delta``, and the
    four-index energy collapses onto ``(U/2) sum_s Tr[n^s (1 - n^s)]`` -- P20's
    functional with ``alpha = beta = J0 = 0``, term for term. Not bit for bit:
    the contraction sums in a different order from the two traces, so what is
    asserted is round-off.
    """
    ns = _random_ns(nspin)
    full = coefficients_from_setup(_setup(1, j=0.0))
    simple = coefficients_from_setup(_setup(0))
    assert float(hubbard_energy(ns, full)) == pytest.approx(
        float(hubbard_energy(ns, simple)), abs=1e-13
    )
    assert np.asarray(hubbard_potential(ns, full)) == pytest.approx(
        np.asarray(hubbard_potential(ns, simple)), abs=1e-13
    )


@pytest.mark.parametrize("nspin", [1, 2])
@pytest.mark.parametrize("l", [1, 2, 3])
def test_the_potential_is_the_gradient_of_the_energy(nspin, l):
    """``jax.grad`` against ``v_hubbard_full`` transcribed literally.

    The two share nothing but ``vee`` and the input, and the ``nspin = 1`` case
    is where it bites: the reported energy is ``2 E_u - E_dc`` and the potential
    is the derivative of ``E_u - E_dc/2``, so differentiating the reported
    energy would give a double-counting potential twice too large.
    """
    j = 1.75 / RY_TO_EV
    setup = _setup(1, l=l, j=j)
    ns = _random_ns(nspin, width=2 * l + 1)
    coefficients = coefficients_from_setup(setup)
    v_qe, e_qe = qe_hubbard_full_potential(np.asarray(ns), setup)
    assert float(hubbard_energy(ns, coefficients)) == pytest.approx(e_qe, abs=1e-12)
    assert np.asarray(hubbard_potential(ns, coefficients)) == pytest.approx(
        v_qe, abs=1e-12
    )


def test_the_energy_is_invariant_under_a_rotation_of_the_manifold():
    """Rotating ``ns`` and leaving ``vee`` alone must not change the energy.

    ``vee`` is invariant under the same rotation (tested above), so this is that
    statement one level up and it is what the functional is named for. The
    simplified functional passes it too -- it is built from traces -- so what
    this pins down is that the *four-index* contraction did not lose it.
    """
    l, width = 2, 5
    rng = np.random.default_rng(3)
    generator = rng.normal(size=(3, 3))
    rotation = sla.expm(generator - generator.T)
    directions = rng.normal(size=(400, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    block = slice(l * l, l * l + width)
    d, *_ = np.linalg.lstsq(
        real_spherical_harmonics(directions, l)[:, block],
        real_spherical_harmonics(directions @ rotation.T, l)[:, block],
        rcond=None,
    )

    j = 1.75 / RY_TO_EV
    coefficients = coefficients_from_setup(_setup(1, j=j))
    ns = np.asarray(_random_ns(2))
    # ``ns`` measured in the rotated basis is ``D^T ns D``.
    turned = jnp.asarray(np.einsum("ma,pb,szmp->szab", d, d, ns))
    assert float(hubbard_energy(turned, coefficients)) == pytest.approx(
        float(hubbard_energy(jnp.asarray(ns), coefficients)), abs=1e-12
    )


# --------------------------------------------------------------------------
# Around mean field (P62c)


def _amf_setup(l=2, j=1.0 / RY_TO_EV):
    return replace(_setup(1, l=l, j=j), double_counting="amf")


@pytest.mark.parametrize("nspin", [1, 2])
@pytest.mark.parametrize("l", [1, 2, 3])
def test_around_mean_field_vanishes_for_a_uniform_shell(nspin, l):
    """The identity FLL does not satisfy, and the whole point of AMF.

    Every orbital equally occupied means ``n = nbar``, so the shifted matrix is
    zero and both the energy and the potential are **exactly** zero -- whatever
    the filling. FLL applies a correction there, which is right for a localised
    magnetic insulator and wrong for a metal.
    """
    width = 2 * l + 1
    coefficients = coefficients_from_setup(_amf_setup(l=l))
    for filling in (0.0, 0.3, 1.0):
        ns = jnp.asarray(np.tile(np.eye(width) * filling, (nspin, 1, 1, 1)))
        assert float(hubbard_energy(ns, coefficients)) == pytest.approx(0.0, abs=1e-14)
        assert float(jnp.abs(hubbard_potential(ns, coefficients)).max()) == pytest.approx(
            0.0, abs=1e-14
        )


@pytest.mark.parametrize("nspin", [1, 2])
@pytest.mark.parametrize("l", [1, 2, 3])
def test_around_mean_field_matches_elks_hand_derived_potential(nspin, l):
    """``jax.grad`` against ``vmatmtdu.f90``'s ``dftu = 2`` branch.

    The interesting half is that they agree *at all*: this potential
    differentiates through the mean-field shift and Elk's does not. The extra
    term is ``-delta_{ab} Tr(V)/(2l+1)``, and ``Tr(V)`` vanishes because
    ``sum_a vee(a,c,a,d)`` and ``sum_a vee(a,c,d,a)`` are rotationally
    invariant rank-two tensors -- hence proportional to ``delta_{cd}`` -- which
    they contract with a shifted matrix that is traceless by construction.
    """
    setup = _amf_setup(l=l)
    ns = _random_ns(nspin, width=2 * l + 1)
    coefficients = coefficients_from_setup(setup)
    v_elk, e_elk = elk_amf_potential(np.asarray(ns), setup)
    assert float(hubbard_energy(ns, coefficients)) == pytest.approx(e_elk, abs=1e-13)
    assert np.asarray(hubbard_potential(ns, coefficients)) == pytest.approx(
        v_elk, abs=1e-13
    )


def test_around_mean_field_and_the_fully_localised_limit_differ():
    """A guard against the shift silently doing nothing.

    Both identities above would pass if ``amf`` were quietly running FLL on a
    uniform shell and the shift were zero everywhere, so the last thing to
    assert is that on a *non*-uniform matrix the two are different numbers.
    """
    ns = _random_ns(2)
    amf = float(hubbard_energy(ns, coefficients_from_setup(_amf_setup())))
    fll = float(hubbard_energy(ns, coefficients_from_setup(_setup(1, j=1.0 / RY_TO_EV))))
    assert abs(amf - fll) > 0.1


def test_around_mean_field_needs_the_full_functional(pseudo_dir):
    """The simplified functional *is* the fully-localised limit: no ``vee``, no shift."""
    with pytest.raises(NotImplementedError, match="amf"):
        _built("U Fe1-3d 4.3", pseudo_dir, double_counting="amf")
    with pytest.raises(ValueError, match="hubbard_double_counting"):
        _built("U Fe1-3d 4.3\nJ Fe1-3d 1.0", pseudo_dir, double_counting="petukhov")


# --------------------------------------------------------------------------
# The card


FEO = """
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
%s
"""


def _built(card, pseudo_dir, double_counting=None):
    text = FEO % card
    if double_counting is not None:
        text = text.replace(
            " &system", f" &system\n    hubbard_double_counting = '{double_counting}'"
        )
    system = build_system(parse_pw_input(text))
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    return build_hubbard_setup(system.hubbard, system.structure, pseudos)


def test_the_card_selects_the_functional(pseudo_dir):
    """``read_cards.f90``: a ``J`` means ``kind = 1``, ``U`` alone means ``kind = 0``."""
    plain = _built("U Fe1-3d 4.3\nU Fe2-3d 4.3", pseudo_dir)
    assert plain.kind == 0
    full = _built(
        "U Fe1-3d 4.3\nU Fe2-3d 4.3\nJ Fe1-3d 1.0\nJ Fe2-3d 1.0", pseudo_dir
    )
    assert full.kind == 1
    iron = full.species[full.types[0]]
    assert iron.j == pytest.approx(1.0 / RY_TO_EV, rel=1e-12)
    assert iron.racah[0] == pytest.approx(0.114774114774 / RY_TO_EV, rel=1e-9)


def test_the_full_functional_refuses_the_simplified_parameters(pseudo_dir):
    """QE refuses each of these, in these places; so does this."""
    with pytest.raises(NotImplementedError, match="Hubbard_alpha"):
        _built("U Fe1-3d 4.3\nJ Fe1-3d 1.0\nALPHA Fe1-3d 0.1", pseudo_dir)
    with pytest.raises(NotImplementedError, match="Hund J0"):
        _built("U Fe1-3d 4.3\nJ Fe1-3d 1.0\nJ0 Fe1-3d 0.5", pseudo_dir)


def test_the_racah_parameters_belong_to_their_own_shells(pseudo_dir):
    with pytest.raises(ValueError, match="f shell"):
        _built("U Fe1-3d 4.3\nJ Fe1-3d 1.0\nE2 Fe1-3d 0.1", pseudo_dir)


def test_the_intersite_v_is_still_refused(pseudo_dir):
    with pytest.raises(NotImplementedError, match="intersite"):
        _built("U Fe1-3d 4.3\nV Fe1-3d Fe2-3d 1 2 0.5", pseudo_dir)
