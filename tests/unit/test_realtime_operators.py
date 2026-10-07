"""The real-time chunk's operators for spinors and augmented datasets, without a propagation.

What the equation of motion and the current of an ultrasoft, PAW or spinor run
are built from, each checked against code that shares nothing with the chunk
but the projectors' definition:

* ``X``, the augmentation's share of the position operator, against
  :func:`~defumat.response.efield.ultrasoft_position` (``adddvepsi_us.f90``),
  and ``X - X^dag`` against ``i dS/dk`` by a finite difference of the overlap;
* Woodbury's ``S^-1`` against ``S``;
* the current operator between generalised eigenstates against
  :meth:`~defumat.response.velocity.VelocityOperator.generalised_matrix_elements`
  (``PLAN.md`` P99), with and without spin-orbit coupling;
* the spinor chunk's kinetic and nonlocal form against the spinor Hamiltonian's
  own application.

``PLAN.md`` P140 and P141 have the measurements.
"""

import re
from functools import lru_cache
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla

from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.realtime.propagate import _prepare
from defumat.scf import Calculation
from defumat.system import build_system
from defumat.system.kpoints import KPoints, for_spin

pytestmark = pytest.mark.unit

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
#: Gamma and one point of no symmetry.
KSET = (np.array([[0.0, 0.0, 0.0], [0.25, 0.1, -0.05]]), np.array([0.5, 0.5]))


@pytest.fixture(autouse=True)
def _bounded_compilation():
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _chunk(pseudo_dir, case, ecut, ecutrho):
    """``(calculation, v_scf, chunk)`` on :data:`KSET` at the superposition of atomic charges."""
    text = (CASES / f"{case}.in").read_text()
    text = re.sub(r"ecutwfc\s*=\s*[0-9.dD+-]+", f"ecutwfc = {ecut}", text)
    text = re.sub(r"ecutrho\s*=\s*[0-9.dD+-]+", f"ecutrho = {ecutrho}", text)
    path = Path("/tmp") / f"defumat-operators-{case}-{ecut}.in"
    path.write_text(text)
    system = build_system(read_pw_input(path))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    system = eqx.tree_at(lambda s: s.kpoints, system,
                         for_spin(KPoints(coords=KSET[0], weights=KSET[1]), system.nspin))
    calculation = Calculation(system, pseudos)
    v_scf = calculation.potential(calculation.starting_density()).v_scf
    ndim = 2 * calculation.basis.planewaves.mask.shape[1] if calculation.noncolin else \
        calculation.basis.planewaves.mask.shape[1]
    dummy = np.zeros((len(KSET[1]), 1, ndim), dtype=complex)
    setup = _prepare(calculation, dummy, np.ones((len(KSET[1]), 1)), v_scf, 0.0, None,
                     "taylor4", None, None, bounds=False)
    return calculation, v_scf, setup.first


def _random(chunk, n, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.asarray(chunk.hamiltonian(jnp.zeros(3)).state_mask)
    x = rng.normal(size=mask.shape[:1] + (n,) + mask.shape[1:]) \
        + 1j * rng.normal(size=mask.shape[:1] + (n,) + mask.shape[1:])
    return jnp.asarray(np.where(mask[:, None, :], x, 0.0))


def _dense(f, chunk):
    """``f`` as a matrix per k-point, ``(nk, ndim, ndim)``, by applying it to the identity."""
    mask = np.asarray(chunk.hamiltonian(jnp.zeros(3)).state_mask)
    ndim = mask.shape[1]
    identity = jnp.broadcast_to(jnp.eye(ndim, dtype=jnp.complex128), (mask.shape[0], ndim, ndim))
    columns = np.asarray(f(identity))  # row b holds f e_b
    return np.swapaxes(columns, 1, 2)


@pytest.mark.parametrize("case", ["alas-epsilon-us", "alas-epsilon-us-soc"])
def test_the_position_tail_is_adddvepsi_us_and_its_adjoint_is_the_overlap_velocity(
        pseudo_dir, case):
    """``X x`` against ``ultrasoft_position`` and ``X - X^dag = i dS/dk`` by finite difference.

    The first shares only the projectors' definition with the chunk (the radial
    transform there, the Chebyshev table here); the second needs nothing but
    ``S`` at two points.
    """
    from defumat.response.efield import ultrasoft_position
    from defumat.response.velocity import VelocityOperator

    calculation, v_scf, chunk = _chunk(pseudo_dir, case, 10.0, 40.0)
    direction = np.array([0.3, -0.5, 0.8])
    zero = jnp.zeros(3)
    x = _random(chunk, 3)
    mine = np.asarray(chunk.position(zero, x, jnp.asarray(direction)))
    velocity = VelocityOperator(calculation, v_scf)
    from defumat.response.efield import _augmentation_dipole

    along = jnp.einsum("a,a...->...", jnp.asarray(direction, dtype=jnp.complex128),
                       _augmentation_dipole(calculation))
    hams = calculation.hamiltonian(v_scf)
    added = np.asarray(ultrasoft_position(
        calculation, hams, x[None], jnp.zeros_like(x)[None], along,
        velocity.projectors(jnp.asarray(direction))))[0]
    scale = np.abs(added).max()
    assert np.abs(mine - added).max() < 1e-10 * scale

    tail = _dense(lambda e: chunk.position(zero, e, jnp.asarray(direction)), chunk)
    step = 1e-4
    s_plus = _dense(lambda e: chunk.overlap(step * jnp.asarray(direction), e), chunk)
    s_minus = _dense(lambda e: chunk.overlap(-step * jnp.asarray(direction), e), chunk)
    ds = (s_plus - s_minus) / (2 * step)
    residual = tail - np.conj(np.swapaxes(tail, 1, 2)) - 1j * ds
    assert np.abs(ds).max() > 1e-3, "the overlap moves"
    assert np.abs(residual).max() < 1e-7 * np.abs(ds).max()


@pytest.mark.parametrize("case", ["alas-epsilon-us", "alas-epsilon-us-soc"])
def test_the_inverse_overlap_inverts_the_overlap(pseudo_dir, case):
    _, _, chunk = _chunk(pseudo_dir, case, 10.0, 40.0)
    kappa = jnp.asarray([0.02, -0.01, 0.03])
    x = _random(chunk, 4, seed=1)
    back = chunk.inverse_overlap(kappa, chunk.overlap(kappa, x))
    assert np.abs(np.asarray(back - x)).max() < 1e-12 * np.abs(np.asarray(x)).max()


@pytest.mark.parametrize("case", ["alas-epsilon-us", "alas-epsilon-us-soc"])
def test_the_current_between_eigenstates_is_the_generalised_velocity(pseudo_dir, case):
    """``<n|dH|m> + i(<Hn|S^-1 X m> - <S^-1 X n|Hm>)`` is P99's ``<n|dH - e_m dS|m> + (e_m - e_n) K``.

    On the lowest generalised eigenstates of the dense ``H(k)``, ``S(k)`` at
    each k-point; without the second term the two differ by the size of the
    augmentation's share.
    """
    from defumat.response.velocity import VelocityOperator

    calculation, v_scf, chunk = _chunk(pseudo_dir, case, 10.0, 40.0)
    zero = jnp.zeros(3)
    ham = chunk.hamiltonian(zero)
    mask = np.asarray(ham.state_mask)
    nb = 10
    psi = np.zeros(mask.shape[:1] + (nb,) + mask.shape[1:], dtype=complex)
    energies = np.zeros(mask.shape[:1] + (nb,))
    hmat = _dense(lambda e: chunk.applied(zero, e), chunk)
    smat = _dense(lambda e: chunk.overlap(zero, e), chunk)
    for ik in range(mask.shape[0]):
        keep = np.flatnonzero(mask[ik])
        h = 0.5 * (hmat[ik] + hmat[ik].conj().T)[np.ix_(keep, keep)]
        s = 0.5 * (smat[ik] + smat[ik].conj().T)[np.ix_(keep, keep)]
        e, v = sla.eigh(h, s)
        psi[ik][:, keep] = v[:, :nb].T
        energies[ik] = e[:nb]
    reference = np.asarray(VelocityOperator(calculation, v_scf).generalised_matrix_elements(
        jnp.asarray(psi)[None], jnp.asarray(energies)[None]))[:, 0]  # (3, nk, nb, nb)

    psi = jnp.asarray(psi)
    weights = jnp.ones(energies.shape)

    def element(kappa, a, b):
        return chunk.band_form(kappa, a, b, jnp.ones(a.shape[:2]))

    mine = np.zeros_like(reference)
    for ik in range(mask.shape[0]):
        for n in range(nb):
            for m in range(nb):
                a = jnp.zeros_like(psi).at[ik, 0].set(psi[ik, n])[:, :1]
                b = jnp.zeros_like(psi).at[ik, 0].set(psi[ik, m])[:, :1]
                slope = jax.jacfwd(lambda k: element(k, a, b))(zero)
                ha, hb = chunk.applied(zero, a), chunk.applied(zero, b)
                for axis in range(3):
                    unit = jnp.zeros(3).at[axis].set(1.0)
                    ya = chunk.inverse_overlap(zero, chunk.position(zero, a, unit))
                    yb = chunk.inverse_overlap(zero, chunk.position(zero, b, unit))
                    tail = 1j * (jnp.vdot(ha, yb) - jnp.vdot(ya, hb))
                    mine[axis, ik, n, m] = slope[axis] + tail
    del weights
    scale = np.abs(reference).max()
    assert np.abs(mine - reference).max() < 1e-9 * scale


def test_the_spinor_chunk_is_the_spinor_hamiltonian(pseudo_dir):
    """``(T + V_NL)(k) x`` plus the local term is ``SpinorHamiltonian.apply``, and the form is its matrix element."""
    _, _, chunk = _chunk(pseudo_dir, "alas-epsilon-us-soc", 10.0, 40.0)
    assert chunk.spinor and chunk.npol == 2
    zero = jnp.zeros(3)
    x = _random(chunk, 3, seed=2)
    y = _random(chunk, 3, seed=3)
    ham = chunk.hamiltonian(zero)
    full = np.asarray(chunk.applied(zero, x))
    # the local part alone, from the template with no kinetic energy and no projector
    nonlocal_free = np.asarray(chunk.kinetic_nonlocal(zero, x))
    local = full - nonlocal_free
    # the local part is kappa-independent: the same at another kappa
    kappa = jnp.asarray([0.05, 0.0, -0.02])
    again = np.asarray(chunk.applied(kappa, x)) - np.asarray(chunk.kinetic_nonlocal(kappa, x))
    assert np.abs(again - local).max() < 1e-11 * np.abs(full).max()
    w = jnp.ones(x.shape[:2])
    form = complex(chunk.band_form(zero, y, x, w))
    direct = complex(jnp.vdot(y, chunk.kinetic_nonlocal(zero, x)))
    assert abs(form - direct) < 1e-11 * abs(direct)
    del ham
