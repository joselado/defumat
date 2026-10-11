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
from defumat.scf.potential import (
    refuse_noncollinear_gga, resolve_noncollinear_gga, v_of_rho,
)
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
    # The 'kubler' branch without an axis has the collinear energy in the
    # continuum (the functional is even under the swap of up and down) but not
    # on the grid: grad |m| is the FFT gradient of a field with kinks, which is
    # not band-limited. Measured 7.17e-5 Ry on this field (2026-10-11).
    kubler = _potentials(rho4, cell, gvectors, "kubler")
    assert abs(float(kubler.etxc) - float(collinear.etxc)) > 1e-5


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
    # The second direction moves the magnetization alone, so the charge's
    # Hartree term cannot carry the comparison. The step is 1e-3 of the
    # magnetization: the error falls as h^2 to 1.1e-8 there and round-off in a
    # 10 Ry total takes over below 1e-4 (measured 2026-10-11).
    for seed, charge in ((31, 1.0), (35, 0.0)):
        direction = np.stack([_band_limited(gvectors, seed + k, scale=1.0) for k in range(4)])
        direction[0] *= charge
        h = 1e-3 * np.max(np.abs(rho[1:])) / np.max(np.abs(direction))

        def energy(x):
            p = _potentials(x, cell, gvectors)
            return float(p.etxc) + float(p.ehart)

        difference = (energy(rho + h * direction) - energy(rho - h * direction)) / (2 * h)
        predicted = cell.volume / npts * np.sum(v * direction)
        assert difference == pytest.approx(predicted, rel=1e-6)


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
    with pytest.raises(ValueError, match="has none"):
        refuse_noncollinear_gga("grad_m", 4, lda, False, False)
    assert refuse_noncollinear_gga("grad_m", 4, PBE, True, False) == "grad_m"
    assert refuse_noncollinear_gga("grad_m", 4, get_functional("TB09"), False, False) == "grad_m"
    with pytest.raises(NotImplementedError, match="nosource"):
        refuse_noncollinear_gga("grad_m", 4, PBE, False, True)


def test_grad_m_is_the_default_wherever_it_applies():
    """Nothing asked for: ``'grad_m'`` on a noncollinear GGA, ``'kubler'`` where
    the branch is refused, and an explicit choice is kept."""
    lda = get_functional("PZ")
    assert resolve_noncollinear_gga(None, 4, PBE, False, False) == "grad_m"
    assert resolve_noncollinear_gga(None, 2, PBE, False, False) == "kubler"
    assert resolve_noncollinear_gga(None, 4, lda, False, False) == "kubler"
    assert resolve_noncollinear_gga("kubler", 4, PBE, False, False) == "kubler"


# --- the meta-GGA's parity form, on the grid --------------------------------


def _collinear_meta_fields(gvectors):
    n = 0.05 + _band_limited(gvectors, 51, scale=1e-3)
    mz = _band_limited(gvectors, 52, scale=1e-3)
    tau0 = 0.03 + _band_limited(gvectors, 53, scale=1e-4)
    tauz = _band_limited(gvectors, 54, scale=1e-4)
    return n, mz, tau0, tauz


def test_the_meta_gga_parity_form_is_the_collinear_meta_gga_for_a_collinear_field(silicon):
    """Tran-Blaha on a collinear field with nodes, along a tilted axis: the
    ``nspin = 2`` branch's channel potentials, as ``v_0`` and ``B``."""
    cell, gvectors = silicon
    tb09 = get_functional("TB09")
    n, mz, tau0, tauz = _collinear_meta_fields(gvectors)
    collinear = v_of_rho(_collinear_pair(n, mz), gvectors, cell, functional=tb09,
                         tau=_collinear_pair(tau0, tauz))
    e = _unit([0.3, -0.5, 0.8])
    rho4 = np.concatenate([n[None], e[:, None, None, None] * mz[None]])
    tau4 = np.concatenate([tau0[None], e[:, None, None, None] * tauz[None]])
    vector = v_of_rho(jnp.asarray(rho4), gvectors, cell, functional=tb09,
                      tau=jnp.asarray(tau4), noncollinear_gga="grad_m")
    v_up, v_dw = np.asarray(collinear.v_scf)
    v = np.asarray(vector.v_scf)
    assert float(vector.meta_c) == pytest.approx(float(collinear.meta_c), rel=1e-13)
    assert np.max(np.abs(v[0] - 0.5 * (v_up + v_dw))) < 1e-11
    assert np.max(np.abs(v[1:] - e[:, None, None, None] * (0.5 * (v_up - v_dw))[None])) < 1e-11


def test_the_meta_gga_parity_form_turns_with_the_spins(silicon):
    cell, gvectors = silicon
    tb09 = get_functional("TB09")
    rho = _noncollinear(gvectors, 60)
    tau = np.concatenate([(0.03 + _band_limited(gvectors, 64, scale=1e-4))[None],
                          np.stack([_band_limited(gvectors, 65 + k, scale=5e-5)
                                    for k in range(3)])])
    rotation = np.linalg.qr(np.random.default_rng(5).normal(size=(3, 3)))[0]

    def turn(field):
        out = field.copy()
        out[1:] = np.einsum("ij,j...->i...", rotation, field[1:])
        return out

    one = np.asarray(v_of_rho(jnp.asarray(rho), gvectors, cell, functional=tb09,
                              tau=jnp.asarray(tau), noncollinear_gga="grad_m").v_scf)
    two = np.asarray(v_of_rho(jnp.asarray(turn(rho)), gvectors, cell, functional=tb09,
                              tau=jnp.asarray(turn(tau)), noncollinear_gga="grad_m").v_scf)
    assert np.max(np.abs(two[0] - one[0])) < 1e-11
    assert np.max(np.abs(two[1:] - np.einsum("ij,j...->i...", rotation, one[1:]))) < 1e-11


# --- the PAW sphere ------------------------------------------------------------


@pytest.fixture(scope="module")
def oxygen_paw():
    """``O.pbe-kjpaw.UPF``'s sphere under PBE and under TB09, and a ``becsum`` shape."""
    from pathlib import Path

    from defumat.paw.onecenter import _build_species
    from defumat.pseudo import read_upf
    from defumat.pseudo.projectors import projector_channels

    pseudo = read_upf(Path(__file__).resolve().parents[1] / "data" / "pseudo"
                      / "O.pbe-kjpaw.UPF")
    nh = len(projector_channels(pseudo))
    rng = np.random.default_rng(11)
    a = rng.normal(size=(nh, nh)) * 0.15
    charge = a @ a.T + np.diag(rng.uniform(0.2, 1.0, nh))
    b = rng.normal(size=(nh, nh)) * 0.1
    moment = 0.5 * (b + b.T)  # mixed signs, so the moment has nodes on the sphere
    return (_build_species(pseudo, PBE, 0.0), _build_species(pseudo, get_functional("TB09"), 0.0),
            charge, moment, rng)


def _sphere(paw, becsum, choice="grad_m"):
    from defumat.paw.onecenter import onecenter_species

    energy, ddd = onecenter_species(paw, jnp.asarray(becsum), axis=None,
                                    noncollinear_gga=choice)
    return float(energy), np.asarray(ddd)


def test_the_paw_sphere_is_the_collinear_sphere_for_a_collinear_moment(oxygen_paw):
    paw, _, charge, moment, _ = oxygen_paw
    e = _unit([0.3, -0.5, 0.8])
    energy2, ddd2 = _sphere(paw, np.stack([0.5 * (charge + moment), 0.5 * (charge - moment)]))
    energy4, ddd4 = _sphere(paw, np.concatenate([charge[None], np.multiply.outer(e, moment)]))
    assert energy4 == pytest.approx(energy2, rel=1e-12)
    assert np.max(np.abs(ddd4[0] - 0.5 * (ddd2[0] + ddd2[1]))) < 1e-10
    assert np.max(np.abs(ddd4[1:] - np.multiply.outer(e, 0.5 * (ddd2[0] - ddd2[1])))) < 1e-10


def test_the_paw_sphere_potential_is_the_derivative_of_its_energy(oxygen_paw):
    """``ddd`` against a central difference in ``becsum``, on a noncollinear moment.

    The bound is the sphere's own: ``v1 - div h`` assembled on the quadrature is
    not the exact derivative of the quadrature's energy, and the existing
    two-channel PBE sphere misses its finite difference by **1.6e-4** relative
    on this ``becsum`` shape (1.5e-5 unpolarized, 4.7e-4 under ``'kubler'``
    noncollinear; D22, 2026-10-12), and this branch reads 4.3e-4. A spiral
    adds nothing on the sphere
    (:func:`defumat.paw.gradient._noncollinear_gradient_grad_m`)."""
    paw, _, charge, moment, _ = oxygen_paw
    rng = np.random.default_rng(23)
    nh = charge.shape[0]
    transverse = np.stack([moment, 0.5 * np.roll(moment, 1, axis=0) + 0.5 * np.roll(moment, 1, axis=0).T,
                           0.3 * charge])
    becsum = np.concatenate([charge[None], transverse])
    _, ddd = _sphere(paw, becsum)
    d = rng.normal(size=(4, nh, nh))
    d = 0.5 * (d + np.swapaxes(d, 1, 2))
    h = 1e-5
    difference = (_sphere(paw, becsum + h * d)[0]
                  - _sphere(paw, becsum - h * d)[0]) / (2 * h)
    assert difference == pytest.approx(np.sum(ddd * d), rel=1e-3)


def test_the_one_centre_meta_gga_is_the_collinear_one_for_a_collinear_moment(oxygen_paw):
    from defumat.paw.onecenter import _meta_exchange_onecenter, _multipoles

    _, paw, charge, moment, _ = oxygen_paw
    e = _unit([0.3, -0.5, 0.8])
    pair = np.stack([0.5 * (charge + moment), 0.5 * (charge - moment)])
    four = np.concatenate([charge[None], np.multiply.outer(e, moment)])

    def potential(becsum, choice):
        becsum = jnp.asarray(becsum)
        rho_lm = _multipoles(paw, becsum, paw.radial_ae)
        tau_lm = paw.kinetic_ae.contract(becsum, paw.coefficients)
        return np.asarray(_meta_exchange_onecenter(rho_lm, tau_lm, paw, 1.1, choice))

    v2 = potential(pair, "kubler")
    v4 = potential(four, "grad_m")
    scale = np.max(np.abs(v2))
    assert np.max(np.abs(v4[0] - 0.5 * (v2[0] + v2[1]))) < 1e-11 * scale
    assert np.max(np.abs(v4[1:] - np.multiply.outer(e, 0.5 * (v2[0] - v2[1])))) < 1e-11 * scale
