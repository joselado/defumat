"""The six columns of ``C_ijkl`` are traced once, and the tensor does not move.

``OPEN.md`` Part III H7. Each column of the elastic tensor is one ``jvp`` of the
stress at the same primal point, and the loop called
:func:`~defumat.eager.compiled_jvp` once per column, which traces the whole
forward-over-reverse derivative at every call to find the program it kept. The
loop now traces it once (:func:`~defumat.eager.compiled_function`) and runs the
kept program six times. What is checked is the count of traces and that the
tensor is the old loop's to the bit, which is what makes this a free change
rather than a number-moving one.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat import Calculator
from defumat import eager
from defumat.eager import compiled_jvp
from defumat.forces.energy import FrozenState, energy_at
from defumat.response.elastic import VOIGT, elastic_constants
from defumat.response.strain import strain_tangent

pytestmark = [pytest.mark.regression, pytest.mark.slow]

DATA = Path(__file__).resolve().parents[1] / "data" / "qe"


def _six_compiled_jvps(calculation, result, response):
    """The loop as it stood before: one ``compiled_jvp`` per column."""
    eigenvalues = jnp.asarray(result.eigenvalues)
    if eigenvalues.ndim == 2:
        eigenvalues = eigenvalues[None]
    weights, _ = calculation.occupations(eigenvalues)
    psi = jnp.asarray(result.wavefunctions)
    weights = jnp.asarray(weights)
    nocc = response.dpsi[0, 0].shape[2]
    zero = jnp.zeros((3, 3))

    def energy(strain, states):
        return energy_at(calculation.at_strain(strain),
                         FrozenState(wavefunctions=states, weights=weights,
                                     eigenvalues=eigenvalues))

    gradient = jax.grad(energy, argnums=0)
    tensor = np.zeros((3, 3, 3, 3))
    for (k, l) in VOIGT:
        states = jnp.zeros_like(psi).at[:, :, :nocc].set(jnp.asarray(response.dpsi[k, l]))
        _, column = compiled_jvp(gradient, (zero, psi), (strain_tangent(k, l), states))
        column = np.asarray(column) / calculation.system.cell.volume
        column = 0.5 * (column + column.T)
        tensor[:, :, k, l] = tensor[:, :, l, k] = column
    return tensor


def test_the_six_columns_are_traced_once_and_the_tensor_is_unchanged(pseudo_dir, monkeypatch):
    calculator = Calculator.from_file(DATA / "si-electrostriction.in",
                                      pseudo_dir=pseudo_dir, announce=False)
    result = calculator.get_scf()
    response = calculator.get_strain_response()
    calculation = calculator.calculation

    traces = []
    original = eager._trace

    def counting(fn, args):
        traces.append(fn)
        return original(fn, args)

    monkeypatch.setattr(eager, "_trace", counting)
    tensor = elastic_constants(calculation, result.wavefunctions, result.eigenvalues,
                               result.density, response).tensor
    assert len(traces) == 1, f"{len(traces)} traces for six columns"

    monkeypatch.setattr(eager, "_trace", original)
    np.testing.assert_array_equal(tensor, _six_compiled_jvps(calculation, result, response))
