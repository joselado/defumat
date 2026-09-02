"""The mixer a self-consistent response advances with.

``defumat.response.mixing`` exists because all three response loops advanced
with one line of linear mixing where QE uses a Broyden history
(``LR_Modules/mix_pot.f90``), and linear mixing of a map whose Jacobian has an
eigenvalue below -1 does not converge slowly -- it diverges. What is checked
here is the wrapper: that the pieces are packed into **one** history rather than
mixed independently, that the shapes survive, and that a linear map is what the
plain mixer reduces to.

The physics validation is elsewhere and is the point: the P24, P25 and P26
regression suites give the same answers with this in place, in about half the
iterations, and rhombohedral BN converges in 18 iterations where linear mixing
descended for 61 and then diverged.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.response.mixing import DEFAULT_RESPONSE_MIXING, ResponseMixer

pytestmark = pytest.mark.unit


def test_the_linear_mixer_is_the_line_it_replaced():
    """``dvscf + beta (induced - dvscf)``, which is what the loops used to do."""
    mixer = ResponseMixer("linear", beta=0.3)
    current = jnp.asarray(np.full((2, 3), 2.0))
    proposed = jnp.asarray(np.full((2, 3), 5.0))
    assert np.allclose(np.asarray(mixer.mix(current, proposed)), 2.0 + 0.3 * 3.0)


def test_the_shapes_survive_the_round_trip():
    mixer = ResponseMixer(beta=0.7)
    shapes = [(3, 3, 1, 4, 4, 4), (2, 5)]
    current = [jnp.zeros(s) for s in shapes]
    proposed = [jnp.ones(s) for s in shapes]
    out = mixer.mix(current, proposed)
    assert [tuple(np.asarray(a).shape) for a in out] == shapes


def test_two_pieces_are_one_anderson_problem_not_two():
    """The electric-field loop mixes ``dV_scf`` and the one-centre term together.

    Anderson's step is a least-squares problem over the history of *one* vector.
    Splitting a coupled state into two vectors and giving each its own history
    solves a different problem -- so the test is that mixing the pair together
    is **not** the same as mixing each alone, on a history where the two are
    correlated. If this ever passes trivially, the packing has been lost.
    """
    together = ResponseMixer(beta=0.7)
    apart = (ResponseMixer(beta=0.7), ResponseMixer(beta=0.7))

    generator = np.random.default_rng(0)
    a = [jnp.asarray(generator.standard_normal((4,))) for _ in range(2)]
    for _ in range(3):
        b = [jnp.asarray(0.5 * np.asarray(x) + generator.standard_normal((4,)))
             for x in a]
        joint = together.mix(a, b)
        split = [m.mix(x, y) for m, x, y in zip(apart, a, b)]
        a = joint
    assert not np.allclose(np.asarray(joint[0]), np.asarray(split[0]))


def test_a_linear_fixed_point_converges_where_linear_mixing_diverges():
    """The failure this module exists for, on a two-line model of it.

    ``F(x) = J x + c`` with an eigenvalue of ``J`` below -1 is exactly the
    induced-potential map of a cell whose smallest ``G`` makes ``4 pi e^2/G^2``
    large. Linear mixing at 0.7 walks away from the fixed point; Anderson finds
    it. The eigenvalue used, -2.3, is the one measured on rhombohedral BN.
    """
    jacobian = np.diag([-2.3, 0.4])
    constant = np.array([1.0, 1.0])
    exact = np.linalg.solve(np.eye(2) - jacobian, constant)

    def run(mode):
        mixer = ResponseMixer(mode, beta=0.7)
        x = jnp.zeros(2)
        for _ in range(60):
            x = mixer.mix(x, jnp.asarray(jacobian @ np.asarray(x) + constant))
        return np.asarray(x)

    assert not np.isfinite(run("linear")).all() or \
        np.abs(run("linear") - exact).max() > 1.0
    assert np.abs(run(DEFAULT_RESPONSE_MIXING) - exact).max() < 1e-8
