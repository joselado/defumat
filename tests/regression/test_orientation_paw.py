"""The orientation torque on a PAW dataset (P122).

A PAW Hamiltonian carries its potential twice, on the grid from the density and
on the spheres as the one-centre coefficients ``ddd_paw`` from ``becsum``, and a
turn of the texture turns both. These tests hold the two routes to each other
and to the physics on fully-relativistic PAW nickel, at a cutoff below the
dataset's own because every statement here is an identity, which holds at any.
"""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat.calculator import Calculator
from tests.conftest import GENERATED

pytestmark = [pytest.mark.regression, pytest.mark.slow]


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    yield
    jax.clear_caches()


def _nickel_text(angle: float = 0.0) -> str:
    text = (GENERATED / "ni-tetragonal-relaxed-mae-paw.in").read_text()
    text = text.replace("ecutwfc = 75.0, ecutrho = 480.0,",
                        "ecutwfc = 40.0, ecutrho = 320.0,")
    return text.replace("angle1(1) = 0.0", f"angle1(1) = {angle}")


@lru_cache(maxsize=2)
def _source():
    """The one-file route's first leg: this dataset at ``soc_scale = 0``."""
    from defumat.scf.driver import run_scf

    spinor = Calculator.from_text(_nickel_text(), pseudo_dir=GENERATED.parent / "pseudo",
                                  announce=False)
    return spinor, run_scf(spinor.system.with_soc_scale(0.0), spinor.pseudos,
                           conv_thr=1.0e-10)


def test_the_paw_torque_is_the_grid_s_plus_the_one_centre_field_s():
    """``jax.grad`` through everything against the two closed-form parts.

    On the same states, Route A's torque must be the grid integral
    ``integral of m_out x B`` plus the one-centre term
    (``scf/driver.py:_onecenter_torque``, the derivative of ``ddd_paw . becsum_out``
    in a turn of the input), which is exactly what Route C adds each iteration.
    Measured at 1.2e-11. **The one-centre part is not a correction**: it came out
    larger than the whole and of the opposite sign, (-1.9, 4.5, 0.0) against the
    grid's (1.1, -3.1, 0.2) x 1e-6, which is why the torque refused PAW until the
    one-centre coefficients were rebuilt inside the differentiated energy.
    """
    from defumat.forces.torque import orientation_torque, rotate_texture
    from defumat.scf.driver import _onecenter_torque
    from defumat.scf.spin_torque import exchange_torque
    from defumat.workflows.anisotropy import (
        _reference_axis,
        _reference_texture,
        _reference_texture_of,
        _with_rotation,
        rotation_from_euler,
    )
    from defumat.workflows.nscf import fixed_density_states

    spinor, source = _source()
    rotation = rotation_from_euler(0.4, 0.9, -0.3)
    own = _reference_axis(spinor.system)
    texture = _reference_texture(source.density, own)
    becsum = tuple(None if b is None else _reference_texture_of(b, source.density, own)
                   for b in source.becsum)
    turned = tuple(None if b is None else rotate_texture(b, rotation) for b in becsum)
    calculation, _, eigenvalues, states = fixed_density_states(
        _with_rotation(spinor.system, rotation), spinor.pseudos,
        rotate_texture(texture, rotation), conv_thr=1.0e-10, becsum=turned)
    weights, _ = calculation.occupations(jnp.asarray(eigenvalues))

    torque = orientation_torque(calculation, states, weights, texture, rotation,
                                becsum=becsum)
    becsum_out = calculation.becsum(states, weights)
    output = calculation.density(states, weights, becsum_out)
    potential = calculation.potential(rotate_texture(texture, rotation), 1.0, None)
    grid = np.asarray(exchange_torque(output, potential.v_scf,
                                      calculation.system.cell).total)
    one_centre = _onecenter_torque(calculation, turned, becsum_out)

    np.testing.assert_allclose(grid + one_centre, torque, rtol=1.0e-9,
                               atol=1.0e-9 * np.linalg.norm(torque))
    assert np.linalg.norm(one_centre) > np.linalg.norm(torque)


def test_without_the_coupling_the_paw_torque_vanishes():
    """``soc_scale = 0`` on the one-shot as well: 1.3e-11 against 1.6e-6."""
    from defumat.workflows.anisotropy import rotation_from_euler, run_orientation_torque

    spinor, source = _source()
    rotation = rotation_from_euler(0.4, 0.9, -0.3)
    live = run_orientation_torque(spinor.system, spinor.pseudos, source.density,
                                  rotation=rotation, becsum=source.becsum)
    off = run_orientation_torque(spinor.system, spinor.pseudos, source.density,
                                 rotation=rotation, becsum=source.becsum,
                                 soc_scale=0.0)
    assert np.linalg.norm(off.torque) < 1.0e-4 * np.linalg.norm(live.torque)


def test_with_the_coupling_becsum_is_not_a_vector_under_a_spin_rotation():
    """Why Route C turns PAW's ``becsum`` through the states.

    ``becsum`` computed from spinors turned by ``U(w)`` against ``becsum`` turned
    by ``R(w)`` as a vector in its three magnetization components. Without the
    coupling they are the same (2.8e-16). With it they are not (4.6e-2 relative):
    the fully-relativistic ``becsum`` keeps only the blocks diagonal in ``j``, and
    a spin rotation mixes ``j = l + 1/2`` with ``l - 1/2``. The grid density,
    which carries the augmentation built from ``becsum``, differs by 1.1e-5.
    """
    from defumat.forces.torque import rotate_texture
    from defumat.scf.orientation import rotate_spinors, rotation_matrix
    from defumat.workflows.nscf import fixed_density_states

    spinor, source = _source()
    omega = np.array([0.1, 0.4, -0.2])
    rotation = rotation_matrix(omega)
    errors = {}
    for scale in (0.0, 1.0):
        calculation, _, eigenvalues, states = fixed_density_states(
            spinor.system.with_soc_scale(scale), spinor.pseudos, source.density,
            conv_thr=1.0e-9, becsum=source.becsum)
        weights, _ = calculation.occupations(jnp.asarray(eigenvalues))
        plain = calculation.becsum(states, weights)
        turned = calculation.becsum(
            jnp.asarray(rotate_spinors(np.asarray(states), omega)), weights)
        errors[scale] = max(
            np.linalg.norm(np.asarray(t) - np.asarray(rotate_texture(p, rotation)))
            / np.linalg.norm(np.asarray(p))
            for p, t in zip(plain, turned) if p is not None)
    assert errors[0.0] < 1.0e-12
    assert errors[1.0] > 1.0e-3
