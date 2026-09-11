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

import warnings

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


# -- What the Kubo route does where the bands touch -------------------------
#
# ``OPEN.md`` B3. The Kubo curvature divides by a band gap, and the point a
# Chern number is *about* -- a Dirac cone, a Weyl node, the closing that a
# topological transition runs through -- is exactly where that gap is zero.
# The old guard tested ``|gap| > 1e-12`` in absolute energy, which is wrong in
# **both** directions at once: an exactly gapless model lands under it (8e-16
# of rounding) and is silently zeroed, while a gap of 2e-9 sails over it and
# returns 1.7e19. Either way nothing in the output said so.


def _almost_gapless_graphene():
    """Graphene with a sublattice mass of 1e-9: a 2e-9 gap at K and K'.

    ``t2 = 0`` removes the Haldane term, so the two Dirac points survive and a
    3x3 mesh lands on both of them exactly -- (1/3, 2/3) and (2/3, 1/3).
    """
    return ModelSource(hamiltonian=haldane(t2=0.0, mass=1.0e-9), nocc=1)


def test_a_band_touching_is_dropped_and_said_out_loud():
    """Counted, warned about, and recorded on the result.

    Measured against the old absolute guard on the same mesh: it returned
    ``Omega = 1.7e19`` at each Dirac point -- a finite number, larger than
    every other point by nineteen orders, and with nothing in the output to
    say the sum it went into is not a Chern number. Set the mass to zero
    instead and the same guard does the opposite: the 8e-16 residue falls
    *under* 1e-12, the point is zeroed, and again nothing is said. Silence
    that cannot be told from a pass is the trap ``CLAUDE.md`` names, and the
    old guard produced both of its forms depending on the last bit of a
    rounding.
    """
    mesh = plane_mesh((3, 3))
    states = _almost_gapless_graphene().states(mesh.flat())

    with pytest.warns(RuntimeWarning, match="singular at 2 of 9 mesh points"):
        result = berry_curvature(states, mesh, method="kubo", nocc=1)

    assert result.singular_points == 2
    # And the two points are *dropped*, not merely flagged: nothing enormous
    # survives into the map.
    assert np.all(np.isfinite(result.curvature))
    assert np.abs(result.curvature).max() < 1.0e3


def test_a_gapped_model_reports_no_singular_points():
    """The complement, so the count above is a discriminator and not a constant.

    A check whose answer is the same on the case it is meant to catch and on
    the case it is meant to pass says nothing at all.
    """
    mesh = plane_mesh((3, 3))
    states = ModelSource(hamiltonian=haldane(t2=0.2, mass=0.0), nocc=1).states(
        mesh.flat()
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = berry_curvature(states, mesh, method="kubo", nocc=1)
    assert result.singular_points == 0


def test_the_degeneracy_guard_uses_one_threshold_in_both_places():
    """``jnp.where`` gates a value *and* its tangent, and they must agree.

    The inner ``where`` that keeps ``1/gap^2`` finite used to test ``gap == 0``
    while the outer one tested ``|gap| > 1e-12``, so a gap between the two was
    evaluated in a branch that was then thrown away. **That part of the sweep's
    prediction was a null** -- the discarded value is a large finite number
    rather than an infinity for any gap an eigensolver can produce, and
    ``jnp.where`` multiplies its tangent by zero, so no NaN ever appeared
    (checked at gaps of 2e-9, 2e-13, 2e-14 and 8e-16). The two thresholds are
    now one because a guard that means two things is one library change away
    from meaning something wrong, and this asserts the property the rewrite
    bought rather than the bug it did not have.
    """
    import jax
    import jax.numpy as jnp

    from defumat.topology.berry import _kubo_point, _plane_directions

    mesh = plane_mesh((3, 3))
    axes = _plane_directions(mesh)
    hamiltonian = haldane(t2=0.0, mass=1.0e-9)
    # Band width 6, so this is the tolerance ``kubo_curvature`` would pick.
    tol = 1.0e-8 * 6.0

    at_k = jnp.asarray([1.0 / 3.0, 2.0 / 3.0, 0.0])
    value, singular = _kubo_point(hamiltonian, at_k, 1, axes, tol)
    assert int(singular) == 1
    assert float(value) == 0.0
    gradient = jax.grad(lambda k: _kubo_point(hamiltonian, k, 1, axes, tol)[0])
    assert np.all(np.isfinite(np.asarray(gradient(at_k))))
