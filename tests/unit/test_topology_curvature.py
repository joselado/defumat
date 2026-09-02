"""Berry curvature: the sign convention, gauge invariance, and quantisation.

Three things have to be pinned and only one of them is caught by a
gauge-invariance test.

**The sign.** Conjugating every overlap matrix leaves the construction exactly as
gauge invariant as it was and negates every answer, so gauge invariance cannot
detect a sign error. It is pinned twice here: against ``elkpy``'s synthetic
link-phase placement, and against the *analytic* curvature of the spin-1/2
coherent state, ``Omega = -sin(theta)/2``, which is a closed-form result and not
a convention of either code.

**Gauge invariance.** A random phase per state, and a random unitary mixing of
the whole occupied manifold, must both leave every number unchanged. The second
is the one that matters: it is the freedom an eigensolver has inside a
degenerate multiplet, which crystals have everywhere, and only a
determinant-based construction survives it.

**Quantisation.** The Chern number of the Haldane model must be an *exact*
integer on a coarse mesh, not a converging approximation to one.
"""

import numpy as np
import pytest

from defumat.topology import ArrayStates, ModelSource, chern_number, plane_mesh
from defumat.topology.berry import berry_curvature, plaquette_flux
from defumat.topology.links import berry_phase, link_phase
from tests.models import haldane, random_gauge

pytestmark = pytest.mark.unit


def _coherent(theta, phi):
    """``|n(theta, phi)>``, the spin-1/2 coherent state."""
    return np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])


def test_sign_is_pinned_by_the_analytic_coherent_state_curvature():
    """``Omega_{theta phi} = -sin(theta)/2`` (Provost and Vallee, 1980).

    Berry's original example, and the only absolute statement about the sign
    available without trusting another implementation. ``A_phi = -sin^2(theta/2)``
    gives ``Omega = d_theta A_phi = -sin(theta)/2``, integrating to ``-2 pi``
    over the sphere -- the spin-1/2 monopole.
    """
    theta, phi, step = np.pi / 3, 0.4, 1.0e-3
    corner = {
        (a, b): _coherent(theta + a * step, phi + b * step)
        for a in (0, 1)
        for b in (0, 1)
    }

    def link(x, y):
        return complex(link_phase(np.array([[np.vdot(corner[x], corner[y])]])))

    loop = (
        link((0, 0), (1, 0))
        * link((1, 0), (1, 1))
        / (link((0, 1), (1, 1)) * link((0, 0), (0, 1)))
    )
    curvature = float(berry_phase(np.asarray(loop))) / step**2
    assert curvature == pytest.approx(-0.5 * np.sin(theta), abs=2.0e-3)


def test_sign_is_pinned_by_where_the_phase_sits_in_the_plaquette():
    """A phase on a numerator link gives ``-theta``, on a denominator ``+theta``.

    ``elkpy``'s pin, reproduced here so that a flux computed by either code
    means the same thing. The plaquette is
    ``U_1(i,j) U_2(i+1,j) / [U_1(i,j+1) U_2(i,j)]``.
    """
    theta = 0.7
    ones = np.ones((2, 2), dtype=complex)

    numerator = ones.copy()
    numerator[0, 0] = np.exp(1j * theta)
    assert plaquette_flux(numerator, ones)[0, 0] == pytest.approx(-theta, abs=1e-12)

    denominator = ones.copy()
    denominator[0, 1] = np.exp(1j * theta)
    assert plaquette_flux(denominator, ones)[0, 0] == pytest.approx(theta, abs=1e-12)


def test_identical_links_give_no_flux():
    ones = np.ones((3, 3), dtype=complex)
    assert np.allclose(plaquette_flux(ones, ones), 0.0, atol=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("unitary", [False, True])
def test_curvature_is_invariant_under_a_random_gauge(seed, unitary):
    """A phase per state, or a unitary mixing of the manifold, changes nothing.

    The second case is the real test. An eigensolver returns an arbitrary basis
    of every degenerate multiplet, and crystals are degenerate everywhere by
    symmetry, so an invariant that noticed the choice would not be an invariant.
    """
    mesh = plane_mesh((6, 6))

    # Two occupied bands are needed for a unitary mixing to be more than a
    # phase, so the mixing test uses the whole two-band spectrum.
    nocc = 2 if unitary else 1
    states = ModelSource(hamiltonian=haldane(), nocc=nocc).states(mesh.flat())
    gauge = random_gauge(seed, (mesh.nk, nocc), unitary=unitary)
    if unitary:
        rotated = np.einsum("kmn,kna->kma", gauge, np.asarray(states.coefficients))
    else:
        rotated = np.asarray(states.coefficients) * gauge

    plain = berry_curvature(states, mesh)
    turned = berry_curvature(ArrayStates(coefficients=rotated), mesh)
    assert np.allclose(plain.curvature, turned.curvature, atol=1e-10)
    assert plain.chern_number == pytest.approx(turned.chern_number, abs=1e-10)


@pytest.mark.parametrize("shape", [(6, 6), (11, 9), (24, 24)])
def test_haldane_chern_number_is_an_exact_integer(shape):
    """Not "close to 1" on a fine mesh -- exactly 1 on a coarse one.

    This is the property that makes the lattice construction the only one an
    invariant may use, and it is why the tolerance here is ``1e-12`` on a 6x6
    mesh rather than ``1e-2`` on a 200x200 one.
    """
    source = ModelSource(hamiltonian=haldane(t2=0.2, mass=0.0), nocc=1)
    result = chern_number(source, shape=shape)
    assert result.chern_number == pytest.approx(-1.0, abs=1e-12)
    assert result.max_flux < np.pi


def test_a_trivial_haldane_insulator_has_no_chern_number():
    source = ModelSource(hamiltonian=haldane(t2=0.2, mass=1.5), nocc=1)
    assert chern_number(source, shape=(8, 8)).chern_number == pytest.approx(0.0, abs=1e-12)


@pytest.mark.slow
def test_kubo_and_lattice_curvature_agree_on_the_integral():
    """The velocity-operator route converges to the quantised one.

    This is the measurement behind the choice of default: both are right, and
    only one of them is an *integer*. On this gapped model the Kubo sum
    converges spectrally -- 8.6e-3 off at 6x6, 1.7e-5 at 12x12 -- so it is an
    excellent approximation and never an invariant. The link construction is
    exact at 6x6 and stays exact.
    """
    source = ModelSource(hamiltonian=haldane(t2=0.2), nocc=1)
    coarse = chern_number(source, shape=(6, 6), method="kubo", nocc=1)
    finer = chern_number(source, shape=(12, 12), method="kubo", nocc=1)
    assert abs(coarse.chern_number + 1.0) > 1e-3   # not quantised
    assert abs(finer.chern_number + 1.0) < 1e-4    # but convergent
    assert chern_number(source, shape=(6, 6)).chern_number == pytest.approx(
        -1.0, abs=1e-12
    )


def test_kubo_refuses_a_state_set_without_a_hamiltonian():
    mesh = plane_mesh((4, 4))
    states = ArrayStates(coefficients=np.zeros((16, 1, 2), dtype=complex))
    with pytest.raises(NotImplementedError, match="differentiable"):
        berry_curvature(states, mesh, method="kubo")


def test_an_unknown_method_is_refused_by_name():
    mesh = plane_mesh((2, 2))
    states = ArrayStates(coefficients=np.zeros((4, 1, 2), dtype=complex))
    with pytest.raises(ValueError, match="unknown Berry curvature method"):
        berry_curvature(states, mesh, method="green")
