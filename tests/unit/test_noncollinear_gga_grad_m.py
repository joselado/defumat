"""The noncollinear gradient correction built from ``grad m`` (``noncollinear_gga = 'grad_m'``).

Four properties, each against something that shares no machinery with the
branch: the collinear functional for a collinear field with nodes (pointwise and
on the grid, potential included), a global spin rotation, a finite difference
of the energy for a field that is not collinear, and finite derivatives where
``m`` is exactly zero. The SCF half is ``tests/regression/test_noncollinear_gga.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat.basis.fft import g_to_r
from defumat.basis.gvectors import generate_gvectors
from defumat.scf.potential import refuse_noncollinear_gga, v_of_rho
from defumat.system.cell import Cell
from defumat.xc.functional import get_functional

pytestmark = pytest.mark.unit

PBE = get_functional("PBE")


@pytest.fixture(scope="module")
def silicon():
    cell = Cell.from_ibrav(2, [10.2, 0, 0, 0, 0, 0])
    return cell, generate_gvectors(cell, 24.0)


def _band_limited(gvectors, seed, count=12, scale=1.0):
    """A real band-limited field on the first ``count`` G vectors past G = 0."""
    rng = np.random.default_rng(seed)
    coefficients = scale * (rng.normal(size=count) + 1j * rng.normal(size=count))
    field_g = jnp.zeros(gvectors.ngm, dtype=complex).at[1: count + 1].set(coefficients)
    return np.real(np.asarray(g_to_r(field_g, gvectors.fft_index, gvectors.grid)))


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _collinear_pair(n, mz):
    return jnp.stack([0.5 * (n + mz), 0.5 * (n - mz)])


def test_the_pointwise_exchange_is_the_collinear_one_for_a_collinear_field():
    """``c = +-1`` reduces the four terms to the spin-scaled collinear pair."""
    rng = np.random.default_rng(0)
    size = 4000
    n = np.exp(rng.uniform(-12.0, 0.0, size))
    mz = rng.uniform(-1.0, 1.0, size) * n
    mz[:40] = 0.0
    mz[40:80] = 1e-14 * n[40:80]
    grad_n = rng.normal(size=(3, size)) * n * 2.0
    grad_mz = rng.normal(size=(3, size)) * n * 2.0
    grad_mz[:, 80:120] = 0.0
    reference = np.asarray(PBE._spin_exchange_energy(
        _collinear_pair(n, mz), jnp.stack([0.5 * (grad_n + grad_mz), 0.5 * (grad_n - grad_mz)]),
    ))
    b = np.sum(grad_n * grad_mz, axis=0)
    for axis in ([0.0, 0.0, 1.0], _unit([0.3, -0.5, 0.8])):
        e = _unit(axis)
        m = e[:, None] * mz
        a = e[:, None] * b
        value = np.asarray(PBE.noncollinear_gradient_exchange_energy(
            jnp.asarray(n), jnp.abs(jnp.asarray(mz)),
            jnp.asarray(np.sum(grad_n**2, axis=0) + np.sum(grad_mz**2, axis=0)),
            jnp.abs(jnp.asarray(b)), jnp.asarray(np.sum(a * m, axis=0)),
        ))
        assert np.max(np.abs(value - reference)) <= 1e-13 * np.max(np.abs(reference))


def _potentials(rho, cell, gvectors, choice="grad_m"):
    return v_of_rho(jnp.asarray(rho), gvectors, cell, functional=PBE,
                    noncollinear_gga=choice)


def test_a_collinear_field_with_nodes_gives_the_collinear_energy_and_potential(silicon):
    """On the grid, divergence included: energy, charge potential, the longitudinal
    field and a transverse field that vanishes, along a tilted axis."""
    cell, gvectors = silicon
    n = 0.05 + _band_limited(gvectors, 1, scale=1e-3)
    mz = _band_limited(gvectors, 2, scale=1e-3)
    assert (mz > 0).any() and (mz < 0).any() and (np.abs(mz) < n).all()
    collinear = _potentials(_collinear_pair(n, mz), cell, gvectors)
    e = _unit([0.3, -0.5, 0.8])
    rho4 = np.concatenate([n[None], e[:, None, None, None] * mz[None]])
    vector = _potentials(rho4, cell, gvectors)

    assert float(vector.etxc) == pytest.approx(float(collinear.etxc), rel=1e-13)
    v_up, v_dw = np.asarray(collinear.v_scf)
    v = np.asarray(vector.v_scf)
    assert np.max(np.abs(v[0] - 0.5 * (v_up + v_dw))) < 1e-12
    field = 0.5 * (v_up - v_dw)
    assert np.max(np.abs(v[1:] - e[:, None, None, None] * field[None])) < 1e-12
    # and the 'kubler' branch, unsigned, has the same energy for this field
    kubler = _potentials(rho4, cell, gvectors, "kubler")
    assert float(kubler.etxc) == pytest.approx(float(collinear.etxc), rel=1e-9)


def _noncollinear(gvectors, seed):
    n = 0.05 + _band_limited(gvectors, seed, scale=1e-3)
    m = np.stack([_band_limited(gvectors, seed + k + 1, scale=6e-4) for k in range(3)])
    return np.concatenate([n[None], m])


def test_the_energy_is_invariant_under_a_global_spin_rotation(silicon):
    cell, gvectors = silicon
    rho = _noncollinear(gvectors, 10)
    rotation = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))[0]
    turned = rho.copy()
    turned[1:] = np.einsum("ij,j...->i...", rotation, rho[1:])
    one = _potentials(rho, cell, gvectors)
    two = _potentials(turned, cell, gvectors)
    assert float(two.etxc) == pytest.approx(float(one.etxc), rel=1e-13)
    v1, v2 = np.asarray(one.v_scf), np.asarray(two.v_scf)
    assert np.max(np.abs(v2[0] - v1[0])) < 1e-12
    assert np.max(np.abs(v2[1:] - np.einsum("ij,j...->i...", rotation, v1[1:]))) < 1e-12


def test_the_potential_is_the_derivative_of_the_energy_by_finite_difference(silicon):
    """The non-tautological check on a potential that is ``jax.grad`` of its energy:
    a central difference along two directions, on a field with nodes of every
    component and with ``a`` not parallel to ``m``."""
    cell, gvectors = silicon
    rho = _noncollinear(gvectors, 20)
    v = np.asarray(_potentials(rho, cell, gvectors).v_scf)
    npts = rho[0].size
    for seed in (31, 32):
        direction = np.stack([_band_limited(gvectors, seed + k, scale=1.0) for k in range(4)])
        h = 1e-6 * np.max(np.abs(rho[1:])) / np.max(np.abs(direction))

        def energy(x):
            p = _potentials(x, cell, gvectors)
            return float(p.etxc) + float(p.ehart)

        difference = (energy(rho + h * direction) - energy(rho - h * direction)) / (2 * h)
        predicted = cell.volume / npts * np.sum(v * direction)
        assert difference == pytest.approx(predicted, rel=1e-7)


def test_exact_zeros_of_m_give_finite_derivatives_to_second_order(silicon):
    """``s = 0`` and ``A = 0`` are exactly where the divided differences switch to
    derivatives; a NaN there would reach every force and every response."""
    cell, gvectors = silicon
    rho = _noncollinear(gvectors, 40)
    rho[1:, :4] = 0.0
    rho[1:, :, :3] = 0.0

    def potential(x):
        return _potentials(x, cell, gvectors).v_scf

    v = np.asarray(potential(jnp.asarray(rho)))
    tangent = np.zeros_like(rho)
    tangent[2] = 1.0
    _, dv = jax.jvp(potential, (jnp.asarray(rho),), (jnp.asarray(tangent),))
    assert np.isfinite(v).all() and np.isfinite(np.asarray(dv)).all()


def test_the_choice_is_refused_where_it_would_mix_two_functionals():
    lda = get_functional("PZ")
    assert refuse_noncollinear_gga("kubler", 4, lda, True, True) == "kubler"
    assert refuse_noncollinear_gga("grad_m", 4, PBE, False, False) == "grad_m"
    with pytest.raises(ValueError, match="choices"):
        refuse_noncollinear_gga("signed", 4, PBE, False, False)
    with pytest.raises(ValueError, match="noncolin"):
        refuse_noncollinear_gga("grad_m", 2, PBE, False, False)
    with pytest.raises(ValueError, match="none it applies to"):
        refuse_noncollinear_gga("grad_m", 4, lda, False, False)
    with pytest.raises(NotImplementedError, match="PAW"):
        refuse_noncollinear_gga("grad_m", 4, PBE, True, False)
    with pytest.raises(NotImplementedError, match="nosource"):
        refuse_noncollinear_gga("grad_m", 4, PBE, False, True)
