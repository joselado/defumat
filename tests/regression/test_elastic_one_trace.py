"""The six columns of ``C_ijkl`` are traced once, and the tensor does not move.

``OPEN.md`` Part III H7. Each column of the elastic tensor is one ``jvp`` of the
stress at the same primal point, and the loop called
:func:`~defumat.eager.compiled_jvp` once per column, which traces the whole
forward-over-reverse derivative at every call to find the program it kept. The
loop now traces it once (:func:`~defumat.eager.compiled_function`) and runs the
kept program six times. What is checked is the count of traces and that the
tensor is the old loop's to the bit, which is what makes this a free change
rather than a number-moving one.

Since 2026-10-10 the six columns share their primal as well, one ``vmap`` of
the ``jvp`` over the six tangents, whenever the compiled program's temporaries
are within ``SHARED_PRIMAL_BYTES`` (1 GiB); above it the one-trace loop runs.
The tests below take each side of the gate and check that the gate fires.
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


@pytest.fixture(scope="module")
def silicon(pseudo_dir):
    """The converged cell, its strain response, and the old loop's tensor."""
    calculator = Calculator.from_file(DATA / "si-electrostriction.in",
                                      pseudo_dir=pseudo_dir, announce=False)
    result = calculator.get_scf()
    response = calculator.get_strain_response()
    calculation = calculator.calculation
    return (calculation, result, response,
            _six_compiled_jvps(calculation, result, response))


def _tensor(silicon, **options):
    calculation, result, response, _ = silicon
    return elastic_constants(calculation, result.wavefunctions, result.eigenvalues,
                             result.density, response, **options).tensor


def test_the_six_columns_are_traced_once_and_the_tensor_is_unchanged(silicon, monkeypatch):
    """Below the shared-primal gate (0 bytes): one trace, the old tensor to the bit."""
    traces = []
    original = eager._trace

    def counting(fn, args):
        traces.append(fn)
        return original(fn, args)

    monkeypatch.setattr(eager, "_trace", counting)
    tensor = _tensor(silicon, shared_primal_bytes=0)
    assert len(traces) == 1, f"{len(traces)} traces for six columns"

    monkeypatch.setattr(eager, "_trace", original)
    np.testing.assert_array_equal(tensor, silicon[3])


def test_the_shared_primal_is_taken_under_the_gate_and_agrees(silicon, monkeypatch):
    """At the default 1 GiB the six columns are one ``vmap``, to round-off of the loop.

    ``si-electrostriction`` compiles to about 59 MB of temporaries, so the
    shared form is taken; the spy says it was, and that the size it was
    decided on is the one the compiler reported. The tensor agrees with the
    six separate columns to round-off and not to the bit, since the batched
    products sum in a different order (1.6e-11 on 26.2 on rBN).
    """
    from defumat.response import elastic

    seen = []
    original = elastic.sized_function

    def spy(*args):
        out = original(*args)
        seen.append(None if out is None else out[1])
        return out

    monkeypatch.setattr(elastic, "sized_function", spy)
    tensor = _tensor(silicon)
    assert len(seen) == 1 and seen[0] is not None, seen
    assert 0 < seen[0] <= elastic.SHARED_PRIMAL_BYTES, seen
    scale = np.max(np.abs(silicon[3]))
    assert np.max(np.abs(tensor - silicon[3])) <= 1e-10 * scale


def test_a_program_over_the_gate_falls_back_to_the_loop(silicon, monkeypatch):
    """The gate fires: a budget one byte under the compiled size takes the loop, to the bit."""
    from defumat.response import elastic

    seen = []
    original = elastic.sized_function

    def spy(*args):
        out = original(*args)
        seen.append(out[1])
        return out

    monkeypatch.setattr(elastic, "sized_function", spy)
    _tensor(silicon)                       # read the size the compiler reports
    tensor = _tensor(silicon, shared_primal_bytes=seen[0] - 1)
    assert len(seen) == 2 and seen[1] == seen[0]
    np.testing.assert_array_equal(tensor, silicon[3])
