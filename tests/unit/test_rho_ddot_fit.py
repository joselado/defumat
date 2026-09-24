"""The Anderson fit in ``rho_ddot``'s inner product (``PLAN.md`` P113).

``pw.x`` fits its Broyden coefficients in the same ``rho_ddot`` it stops on
(``mix_rho.f90:403-425``). Here that inner product is written as a transform
``F`` whose Euclidean dots are ``rho_ddot``, so the identity that checks it is
``F(r) . F(r) == accuracy(r)``, against the functions the convergence test
already uses and which share no code with the transform beyond the FFT.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.scf.mixing import AndersonMixer
from defumat.scf.potential import (
    ns_ddot_vector,
    rho_ddot_vector,
    scf_accuracy,
    tau_accuracy,
    tau_ddot_vector,
)


def _accuracy_pieces():
    """A dense G set and a cell, built without an SCF."""
    from defumat.basis.builder import build_basis
    from defumat.io.pwin import read_pw_input
    from defumat.system import build_system

    case = Path(__file__).resolve().parents[1] / "data" / "qe" / "si2-tb09.in"
    system = build_system(read_pw_input(case))
    return build_basis(system).dense, system.cell


@pytest.fixture(scope="module")
def pieces():
    return _accuracy_pieces()


@pytest.mark.parametrize("nspin", [1, 2, 4])
def test_the_fit_vectors_self_dot_is_dr2(pieces, nspin):
    """``F(r) . F(r)`` is :func:`scf_accuracy` of the same residual, in every regime.

    One channel is the Hartree half alone; two are ``(up, down)`` and rotate into
    charge and magnetization; four are already ``(n, m)``.
    """
    gvectors, cell = pieces
    rng = np.random.default_rng(100 + nspin)
    residual = jnp.asarray(rng.normal(size=(nspin,) + gvectors.grid))
    f = rho_ddot_vector(residual, gvectors, cell)
    assert float(f @ f) == pytest.approx(float(scf_accuracy(residual, gvectors, cell)),
                                         rel=1e-12)


@pytest.mark.parametrize("nspin", [1, 2, 4])
def test_the_dot_of_two_residuals_is_the_polarised_form(pieces, nspin):
    """``F(a) . F(b) = (Q(a + b) - Q(a - b)) / 4``, with ``Q`` the accuracy.

    That is what makes the Gram matrix ``rho_ddot(r_i, r_j)`` and not merely a
    matrix whose diagonal is right, and it is symmetric by construction.
    """
    gvectors, cell = pieces
    rng = np.random.default_rng(200 + nspin)
    a = jnp.asarray(rng.normal(size=(nspin,) + gvectors.grid))
    b = jnp.asarray(rng.normal(size=(nspin,) + gvectors.grid))

    def q(x):
        return float(scf_accuracy(x, gvectors, cell))

    polarised = 0.25 * (q(a + b) - q(a - b))
    dot = float(rho_ddot_vector(a, gvectors, cell) @ rho_ddot_vector(b, gvectors, cell))
    assert dot == pytest.approx(polarised, rel=1e-10, abs=1e-12 * q(a))


def test_a_uniform_charge_residual_has_no_weight(pieces):
    """The charge half drops ``G = 0``, as neutrality and ``rho_ddot`` both do."""
    gvectors, cell = pieces
    uniform = jnp.ones((1,) + gvectors.grid)
    f = rho_ddot_vector(uniform, gvectors, cell)
    assert float(f @ f) == pytest.approx(0.0, abs=1e-20)


@pytest.mark.parametrize("nspin", [1, 2])
def test_the_tau_vector_is_tauk_ddot(pieces, nspin):
    """Same spin form as :func:`tau_accuracy`, including the ``(total, magnetization)`` rotation."""
    gvectors, cell = pieces
    rng = np.random.default_rng(300 + nspin)
    tau = jnp.asarray(rng.normal(size=(nspin,) + gvectors.grid))
    f = tau_ddot_vector(tau, gvectors, cell)
    assert float(f @ f) == pytest.approx(float(tau_accuracy(tau, gvectors, cell)), rel=1e-12)


@pytest.mark.parametrize("nspin, complex_ns", [(1, False), (2, False), (4, True)])
def test_the_ns_vector_is_ns_ddot(nspin, complex_ns):
    """``sqrt(U/2)`` per site, doubled at one channel, a norm for a spinor's complex ``ns``."""
    from defumat.hubbard.energy import ns_ddot

    rng = np.random.default_rng(400 + nspin)
    shape = (nspin, 3, 5, 5)
    residual = rng.normal(size=shape)
    if complex_ns:
        residual = residual + 1j * rng.normal(size=shape)
    u_metric = jnp.asarray([0.5 * 0.29, 0.0, 0.5 * 0.44])
    residual = jnp.asarray(residual)
    f = ns_ddot_vector(residual, u_metric)
    expected = float(ns_ddot(residual, {"u_metric": u_metric}))
    assert float(f @ f) == pytest.approx(expected, rel=1e-12)


def _history(steps, size, seed):
    rng = np.random.default_rng(seed)
    rho = rng.normal(size=size)
    outs = []
    for k in range(steps):
        outs.append((rho, rho + 0.5 ** k * rng.normal(size=size)))
        rho = rho + 0.1 * rng.normal(size=size)
    return outs


def test_fit_vectors_equal_to_the_residuals_reproduce_the_flat_fit():
    """The wiring alone: handing the residual itself as ``fit`` changes no number."""
    flat, fitted = AndersonMixer(beta=0.4, history=4), AndersonMixer(beta=0.4, history=4)
    for rho_in, rho_out in _history(7, 50, seed=5):
        a = flat.mix(rho_in, rho_out)
        b = fitted.mix(rho_in, rho_out, fit=rho_out - rho_in)
        np.testing.assert_array_equal(a, b)
    assert len(fitted._fits) == len(fitted._residuals) == 4
    assert fitted._fit_mask == "metric"


def test_a_history_without_fit_vectors_restarts_when_one_arrives():
    """A flat history cannot be fitted in another inner product, so it is dropped."""
    mixer = AndersonMixer(beta=0.4, history=4)
    steps = _history(4, 30, seed=6)
    for rho_in, rho_out in steps[:3]:
        mixer.mix(rho_in, rho_out)
    assert len(mixer._residuals) == 3 and not mixer._fits
    rho_in, rho_out = steps[3]
    mixer.mix(rho_in, rho_out, fit=2.0 * (rho_out - rho_in))
    assert len(mixer._residuals) == len(mixer._fits) == 1


def test_a_different_metric_gives_different_coefficients():
    """The fit really is in the vectors handed over: weight one entry heavily and the step moves."""
    plain, weighted = AndersonMixer(beta=0.4, history=4), AndersonMixer(beta=0.4, history=4)
    scale = np.ones(40)
    scale[:5] = 30.0
    last_plain = last_weighted = None
    for rho_in, rho_out in _history(5, 40, seed=7):
        last_plain = plain.mix(rho_in, rho_out, fit=rho_out - rho_in)
        last_weighted = weighted.mix(rho_in, rho_out, fit=scale * (rho_out - rho_in))
    assert not np.allclose(last_plain, last_weighted)


def test_the_checkpoint_carries_the_fit_vectors_and_not_the_metric(tmp_path):
    """The installed callable is derived, the stored vectors are state."""
    from defumat.scf.checkpoint import load_mixer, save_mixer, unhandled_mixer_fields

    mixer = AndersonMixer(beta=0.4, history=4)
    mixer.metric = lambda drho, dns, dtau: np.asarray(drho).ravel()
    steps = _history(3, 20, seed=8)
    for rho_in, rho_out in steps:
        mixer.mix(rho_in, rho_out, fit=rho_out - rho_in)
    assert unhandled_mixer_fields(mixer) == set()
    path = save_mixer(mixer, tmp_path / "mixer.npz")

    restored = load_mixer(AndersonMixer(beta=0.4, history=4), path)
    assert restored._fit_mask == "metric"
    assert len(restored._fits) == 3
    for a, b in zip(restored._fits, mixer._fits):
        np.testing.assert_array_equal(a, b)
    assert restored.metric is None


def test_mix_hands_the_metric_vector_to_the_mixer():
    """``_mix`` evaluates an installed metric on the structured residual and passes it on."""
    from defumat.scf import driver

    seen = {}

    class Recording(AndersonMixer):
        def mix(self, rho_in, rho_out, exclude=None, fit=None):
            seen["fit"] = fit
            return np.asarray(rho_in)

    mixer = Recording()
    mixer.metric = lambda drho, dns, dtau: np.asarray(3.0 * drho).ravel()
    rho = jnp.zeros((1, 2, 2, 2))
    rho_out = jnp.ones((1, 2, 2, 2))
    driver._mix(mixer, rho, rho_out, (None,), (None,))
    np.testing.assert_array_equal(seen["fit"], np.full(8, 3.0))
