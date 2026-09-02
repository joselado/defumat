"""P37's cheap pieces: the response sphere, the kernel registry and the refusals.

Everything here runs off a bare :class:`~defumat.scf.driver.Calculation` --
no SCF, no states -- which is the point: a refusal is a statement about the
calculation and must be reachable before anything expensive has been paid for.
The identities that need a converged ground state are in
``tests/regression/test_tddft.py``.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.scf.driver import Calculation
from defumat.system import build_system
from defumat.tddft import (
    get_kernel,
    kernel_names,
    require_a_sum_over_states_regime,
    response_sphere,
)
from defumat.units import E2, FPI

pytestmark = pytest.mark.unit

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


def _calculation(case: str) -> Calculation:
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    return Calculation(system, pseudos)


# --- the response sphere -----------------------------------------------------

def test_the_response_sphere_excludes_the_origin_and_is_closed_under_inversion():
    """Both properties are load-bearing and neither is obvious from the code.

    ``G = 0`` is not a body entry because in the optical limit it is the *head*,
    a 3x3 block of directions rather than one number; leaving it in would give a
    matrix with two entries meaning the same thing and a divergent Coulomb
    factor on one of them.

    Closure under ``G -> -G`` is what makes the antiresonant half of the pair
    sum free: ``<u_j|e^{-iG.r}|u_i>`` is the reflection of
    ``<u_i|e^{-iG.r}|u_j>``, so the reversed pair costs a gather rather than a
    transform. A sphere sorted by ``|G|^2`` always has it, and asserting it
    keeps a future change to the selection honest.
    """
    calculation = _calculation("si-epsilon-unshifted-nosym")
    sphere = response_sphere(calculation, 8.0)

    assert sphere.nbody > 0
    assert sphere.nm == sphere.nbody + 3

    miller = np.asarray(sphere.miller)
    assert not np.any(np.all(miller == 0, axis=1))  # no G = 0 in the body

    reflection = np.asarray(sphere.reflection)
    assert np.array_equal(reflection[reflection], np.arange(sphere.nbody))
    assert not np.any(reflection == np.arange(sphere.nbody))


def test_the_coulomb_factor_is_the_rydberg_one():
    """``sqrt(8 pi / |G|^2)``, not ``sqrt(4 pi / |G|^2)``.

    ``e^2 = 2`` in Rydberg atomic units, which is exactly the factor
    :func:`~defumat.scf.potential.hartree` carries and the classic place to
    lose a two. The symmetrised ``chi_0`` has it on both sides, so a wrong
    constant here is a factor of two on every dielectric function.
    """
    calculation = _calculation("si-epsilon-unshifted-nosym")
    sphere = response_sphere(calculation, 4.0)
    g2 = np.asarray(calculation.basis.smooth.kinetic(calculation.system.cell))
    lookup = {tuple(m): n
              for n, m in enumerate(np.asarray(calculation.basis.smooth.miller))}
    order = np.array([lookup[tuple(m)] for m in np.asarray(sphere.miller)])
    assert np.allclose(
        np.asarray(sphere.sqrt_coulomb), np.sqrt(E2 * FPI / g2[order])
    )


def test_a_cutoff_with_no_body_is_the_head_only_kernel():
    """``ecut = 0`` is a named approximation, not a degenerate case.

    It leaves the 3x3 head alone -- the head-only kernel of the long-range
    correction literature, and what Elk's own ``LiF-bootstrap`` example asks for
    with ``gmaxrf = 0.0``. So it is supported rather than refused, and what it
    means is stated: no body means no local-field effect, so ``eps_M`` is
    ``1 - X_head``. A *negative* cutoff is still an error, because it is not a
    request for anything.
    """
    calculation = _calculation("si-epsilon-unshifted-nosym")
    head_only = response_sphere(calculation, 0.0)
    assert head_only.nbody == 0
    assert head_only.nm == 3
    with pytest.raises(ValueError, match="cannot be negative"):
        response_sphere(calculation, -1.0)


# --- the kernel registry -----------------------------------------------------

def test_every_kernel_is_registered_under_its_name():
    assert set(kernel_names()) == {"rpa", "alda", "lrc", "bootstrap", "bootstrap-1"}
    assert get_kernel("bootstrap").self_consistent
    assert not get_kernel("bootstrap-1").self_consistent
    # Elk's 211 rounds its loop **twice**: it increments after the first Dyson
    # solve and repeats while ``it <= 1``, so the kernel it ends on was built
    # from the first pass's answer rather than from the seed.
    assert get_kernel("bootstrap-1").iterations == 2
    assert not get_kernel("rpa").self_consistent
    with pytest.raises(ValueError, match="unknown exchange-correlation kernel"):
        get_kernel("nanoquanta")


def test_the_lrc_kernel_is_a_constant_on_the_diagonal_and_needs_its_parameter():
    """``F = -alpha / 4 pi``, head included -- and ``alpha`` has no default.

    The parameter is material-dependent and empirical, which is the entire
    reason the bootstrap kernel was proposed. Supplying a default would hide
    that behind a number that is right for nothing.
    """
    from defumat.tddft.chi0 import ChiZero, ResponseSphere

    sphere = ResponseSphere(
        fft_index=jnp.arange(2), sqrt_coulomb=jnp.ones(2),
        reflection=jnp.asarray([1, 0]),
        miller=jnp.asarray([[1, 0, 0], [-1, 0, 0]]), ecut=1.0,
    )
    chi = ChiZero(x=jnp.zeros((2, 5, 5), dtype=complex), frequencies=jnp.zeros(2),
                  sphere=sphere, npairs=1, nocc=1, nbnd=2)

    with pytest.raises(ValueError, match="needs its parameter"):
        get_kernel("lrc").build(chi, None, {})

    kernel = np.asarray(get_kernel("lrc").build(chi, None, {"alpha": 0.2}))
    assert kernel.shape == (2, 5, 5)
    assert np.allclose(np.diagonal(kernel, axis1=1, axis2=2), -0.2 / FPI)
    assert np.allclose(kernel - np.eye(5) * kernel[0, 0, 0], 0.0)


# --- the refusals ------------------------------------------------------------

def test_a_reduced_k_set_is_refused_before_anything_is_computed():
    """``chi_0(G, G')`` on a wedge would need a rotation in two G indices.

    P36's rank-N symmetriser is Cartesian and does not do it, so the whole grid
    is required. ``si-epsilon.in`` is the shifted wedge the Sternheimer response
    runs on and is exactly what must be refused here.
    """
    with pytest.raises(NotImplementedError, match="full k-grid"):
        require_a_sum_over_states_regime(_calculation("si-epsilon"))


def test_an_ultrasoft_dataset_is_refused_by_name():
    """The plane-wave matrix element gains ``Q_ij(G)`` and nothing here adds it."""
    with pytest.raises(NotImplementedError, match="ultrasoft or PAW"):
        require_a_sum_over_states_regime(_calculation("si-epsilon-us"))


def test_the_whole_grid_norm_conserving_insulator_is_accepted():
    """The complement of the refusals: the one regime that is supported."""
    require_a_sum_over_states_regime(_calculation("si-epsilon-unshifted-nosym"))
