"""P62d unit checks: Slater integrals computed from the radial functions.

Elk's ``inpdftu = 4`` and ``5`` -- ``F^k`` from the manifold's own orbital with
a Yukawa kernel, and the screening length solved for so that one chosen ``U``
fixes the whole interaction matrix. ``pw.x`` computes no Slater integral at all,
so every check here closes inside the package or against an analytic limit:

* **two independent discretisations** of the same integral -- a cumulative
  quadrature and :mod:`defumat.paw.hartree`'s Numerov solve of the radial
  Poisson equation -- which is what found a systematic 1-4 per cent error (the
  diagonal mesh point counted twice in both inner integrals);
* the **``lam -> 0`` limit** of the Yukawa kernel is the bare one, which is the
  statement that fixes its normalisation;
* ``U -> lam -> F^0`` is an exact **round trip**;
* the ratio ``F^4/F^2`` of a real 3d orbital comes out at the **0.625** that QE
  and Elk both hardcode, which is the one number here with an outside answer.
"""

import numpy as np
import pytest

from defumat.hubbard.interaction import (
    exchange_from_slater,
    racah_to_slater,
    slater_integrals,
)
from defumat.hubbard.yukawa import (
    LAMBDA_MAX,
    manifold_radial,
    screening_length,
    slater_from_poisson,
    slater_from_radial,
    slater_set,
)
from defumat.pseudo import read_upf
from defumat.units import RY_TO_EV

pytestmark = pytest.mark.unit

#: A PAW dataset, because only PAW carries the all-electron partial wave; a
#: 3d transition metal, because that is where the published Slater integrals are.
PAW_NICKEL = "Ni.rel-pbe-spn-kjpaw_psl.1.0.0.UPF"
#: Norm-conserving silicon: smooth, decaying to zero well inside the mesh, so
#: the two discretisations are compared where neither has a truncation to argue
#: about.
NORM_CONSERVING = "Si.pz-vbc.UPF"


def _nickel(pseudo_dir, cutoff=None):
    return manifold_radial(read_upf(pseudo_dir / PAW_NICKEL), 3, 2, cutoff)


def test_the_all_electron_partial_wave_is_cut_at_the_augmentation_radius(pseudo_dir):
    """Not cutting it is not a small error -- it is 1e17 in the norm.

    A PAW partial wave solves the atomic problem inside the sphere and is
    whatever the generator left outside it, which diverges. This is why the
    cutoff is read from the dataset rather than defaulted to the whole mesh.
    """
    inside = _nickel(pseudo_dir)
    assert inside.kind == "all-electron"
    assert inside.cutoff == pytest.approx(1.809, abs=0.01)
    assert inside.norm == pytest.approx(0.906, abs=0.01)

    whole = _nickel(pseudo_dir, cutoff=1.0e9)
    assert whole.norm > 1.0e10


def test_the_two_discretisations_agree(pseudo_dir):
    """A cumulative quadrature against the radial Poisson solve.

    They share the mesh and nothing else -- one accumulates
    ``r_<^k/r_>^(k+1)`` directly, the other integrates the Numerov scheme QE
    uses in ``upflib/radial_grids.f90``. Run on a smooth, untruncated
    norm-conserving orbital so that neither end has a boundary to argue about.
    """
    manifold = manifold_radial(read_upf(pseudo_dir / NORM_CONSERVING), 3, 1, 1.0e9)
    assert manifold.norm == pytest.approx(1.0, abs=1e-6)
    for k in (0, 2):
        quadrature = slater_from_radial(manifold, k)
        poisson = slater_from_poisson(manifold, k)
        assert quadrature == pytest.approx(poisson, rel=1e-3)


def test_the_two_discretisations_agree_on_the_all_electron_orbital(pseudo_dir):
    manifold = _nickel(pseudo_dir)
    for k in (0, 2, 4):
        assert slater_from_radial(manifold, k) == pytest.approx(
            slater_from_poisson(manifold, k), rel=2e-3
        )


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0, 2.0, 5.0])
def test_the_screened_integral_is_the_double_sum_it_claims_to_be(pseudo_dir, lam):
    """The check the other three could not make, at **finite** ``lambda``.

    :func:`slater_from_radial` factors the kernel's exponential into two
    recursions so that the scaled Bessel pair never has to be unscaled. That is
    the only part of it a ``lam -> 0`` limit, a self-consistent round trip and a
    Poisson solve at ``lam = 0`` all leave untested -- and it was wrong: the
    backward recursion indexed the gap *behind* each mesh point where it needed
    the one *ahead*, which on a logarithmic grid is a screening length off by
    one per cent and an ``F^k`` off by 0.1 to 0.3, growing with ``lambda``.

    The reference is the ``O(N^2)`` double sum written directly, with the
    *unscaled* ``spherical_in``/``spherical_kn`` -- which is why it is confined
    to the mesh points the orbital actually occupies, where nothing overflows.
    """
    from scipy.special import spherical_in, spherical_kn

    from defumat.units import E2

    manifold = _nickel(pseudo_dir)
    weighted = manifold.chi**2 * manifold.weights
    inside = weighted != 0.0
    f, r = weighted[inside], manifold.r[inside]
    lower, upper = np.minimum.outer(r, r), np.maximum.outer(r, r)

    for k in (0, 2, 4):
        if lam == 0.0:
            kernel = lower**k / upper ** (k + 1)
        else:
            kernel = (
                (2 * k + 1) * lam * spherical_in(k, lam * lower)
                * (2.0 / np.pi) * spherical_kn(k, lam * upper)
            )
        assert slater_from_radial(manifold, k, lam) == pytest.approx(
            E2 * float(f @ kernel @ f), rel=1e-12
        )


def test_the_yukawa_kernel_becomes_the_bare_one(pseudo_dir):
    """The limit that fixes the kernel's normalisation.

    ``i_k(x) -> x^k/(2k+1)!!`` and ``ktilde_k(x) -> (2k-1)!!/x^(k+1)``, so
    ``(2k+1) lam i_k(lam r_<) ktilde_k(lam r_>) -> r_<^k/r_>^(k+1)``. The
    approach is linear in ``lam``, which is what is asserted -- a wrong constant
    would show as a fixed offset instead.
    """
    manifold = _nickel(pseudo_dir)
    bare = [slater_from_radial(manifold, k, 0.0) for k in (0, 2, 4)]
    previous = None
    for lam in (1.0e-2, 1.0e-3, 1.0e-4):
        screened = [slater_from_radial(manifold, k, lam) for k in (0, 2, 4)]
        error = max(abs(a - b) / abs(b) for a, b in zip(screened, bare))
        assert error < 20.0 * lam
        if previous is not None:
            # Ten times smaller lambda, ten times smaller error.
            assert error == pytest.approx(previous / 10.0, rel=0.2)
        previous = error


def test_the_screened_integrals_stay_finite_at_the_largest_screening(pseudo_dir):
    """``lam = 50`` is inside ``findlambda``'s bracket and overflows a naive form.

    ``i_k(lam r)`` reaches ``e^2950`` on this mesh where ``ktilde_k`` reaches
    ``e^-2950``; multiplying the two after evaluating them separately gives
    ``inf * 0``. The scaled pair is what keeps the root find from crashing on
    its own bracket.
    """
    manifold = _nickel(pseudo_dir)
    value = slater_from_radial(manifold, 0, LAMBDA_MAX)
    assert np.isfinite(value)
    assert 0.0 < value < 0.01


def test_screening_lowers_the_monopole_and_leaves_the_higher_multipoles(pseudo_dir):
    """The physics of a screened interaction, as a monotone statement.

    ``F^0`` is the long-ranged part and is what screening removes; ``F^2`` and
    ``F^4`` are short-ranged and survive. So ``F^4/F^2`` *rises* toward one and
    ``J`` falls slowly -- which is why a screened ``U`` of 5 eV does not imply a
    ``J`` of 5/8 eV.
    """
    manifold = _nickel(pseudo_dir)
    previous = None
    for lam in (0.0, 0.5, 1.0, 2.0, 3.0):
        f = slater_set(manifold, 2, lam)
        if previous is not None:
            assert f[0] < previous[0]
            assert f[4] / f[2] > previous[4] / previous[2]
        previous = f
    # F^0 collapses by a factor of seven over that range; F^4 by less than a
    # third.
    bare = slater_set(manifold, 2, 0.0)
    assert previous[0] / bare[0] < 0.2
    assert previous[4] / bare[4] > 0.7


def test_the_screening_length_round_trips(pseudo_dir):
    """``U -> lam -> F^0(lam)`` returns the ``U`` that was asked for."""
    manifold = _nickel(pseudo_dir)
    for u_ev in (3.0, 5.0, 8.0):
        u = u_ev / RY_TO_EV
        lam = screening_length(manifold, u)
        assert slater_from_radial(manifold, 0, lam) == pytest.approx(u, rel=1e-9)


def test_only_a_paw_dataset_reaches_this_route(pseudo_dir):
    """The refusal the guide promises, made where the setup is built.

    Falling back to the pseudo orbital is the silent-wrong this project refuses
    by name: an ultrasoft ``chi`` carries 41 per cent of the norm and gives an
    ``F^0`` of 2.1 eV where the answer is above twenty, and the SCF converges
    and reports success either way.
    """
    from defumat.hubbard.manifold import HubbardInput, build_hubbard_setup
    from defumat.system import build_system
    from defumat.io.pwin import parse_pw_input
    from tests.unit.test_hubbard_full import FEO

    text = (FEO % "U Fe1-3d 4.3").replace(
        " &system", " &system\n    hubbard_slater = 'yukawa'"
    )
    system = build_system(parse_pw_input(text))
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    with pytest.raises(NotImplementedError, match="all-electron partial wave"):
        build_hubbard_setup(system.hubbard, system.structure, pseudos)


def test_an_unbound_manifold_is_refused(pseudo_dir):
    """Silicon's 3p has a norm of 0.49 inside 2.5 bohr: half an orbital.

    The norm is measured and reported for exactly this, and below the floor it
    is an error rather than a caveat -- an integral over part of an orbital is
    not that orbital's Slater integral.
    """
    from defumat.hubbard.manifold import NORM_FLOOR

    manifold = manifold_radial(
        read_upf(pseudo_dir / "Si.pz-n-kjpaw_psl.0.1.UPF"), 3, 1, cutoff=2.5
    )
    assert manifold.kind == "all-electron"
    assert manifold.norm < NORM_FLOOR


def test_a_u_above_the_bare_integral_is_refused(pseudo_dir):
    """Screening only lowers ``F^0``; asking for more is an error, not a clamp."""
    manifold = _nickel(pseudo_dir)
    bare = slater_from_radial(manifold, 0)
    with pytest.raises(ValueError, match="unscreened"):
        screening_length(manifold, 2.0 * bare)
    with pytest.raises(ValueError, match="positive U"):
        screening_length(manifold, -1.0)


def test_the_computed_slater_ratio_is_the_atomic_one(pseudo_dir):
    """The one number here with an answer from outside the package.

    QE and Elk both *impose* ``F^4/F^2 = 0.625`` on a ``d`` shell. Computed from
    nickel's own all-electron 3d partial wave it comes out at 0.626, and
    ``J = (F^2 + F^4)/14`` at 1.28 eV against a published 3d Hund exchange of
    about 1 eV. Neither is fitted and neither enters the calculation as an
    input.
    """
    f = slater_set(_nickel(pseudo_dir), 2, 0.0)
    assert f[4] / f[2] == pytest.approx(0.625, abs=0.01)
    assert exchange_from_slater(2, f) * RY_TO_EV == pytest.approx(1.28, abs=0.1)


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_the_racah_conversion_is_a_change_of_coordinates(l):
    """Racah in, Slater out, and QE's own parameterisation reproduces it.

    Elk's ``inpdftu = 3`` spans exactly the space ``(U, J, B)`` spans, so
    reading ``U`` and ``J`` back off the converted integrals and feeding them to
    :func:`slater_integrals` must return the same array. That closes the two
    conventions against each other without needing either code to be run.
    """
    e = np.array([0.30, 0.05, 0.010, 0.002])[: l + 1]
    f = racah_to_slater(l, e)
    u = f[0]
    j = exchange_from_slater(l, f)
    if l == 2:
        racah = ((f[2] - 5.0 * j) / 31.5,)
    elif l == 3:
        # ``F^2`` and ``F^4`` fix ``E2`` and ``E3`` given ``J``; solve the two
        # rows of ``slater_integrals`` for them rather than inverting by hand.
        a = np.array([[32175.0 / 42.0, 2475.0 / 42.0], [-141570.0 / 77.0, 4356.0 / 77.0]])
        racah = tuple(np.linalg.solve(a, [f[2] - 225.0 / 54.0 * j, f[4] - 11.0 * j]))
    else:
        racah = ()
    assert slater_integrals(l, u, j, racah) == pytest.approx(f, abs=1e-10)
