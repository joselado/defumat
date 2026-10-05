"""``<beta|psi>`` is the same number whichever operand carries the conjugate.

``OPEN.md`` Part III M4. ``calbec`` conjugated ``vkb``, which in the compiled
Davidson is an ``(npwx, nkb)`` copy made inside the loop body at every rung of
the band ladder, where conjugating the band block and then the small result
gives the same sum. The change is only admissible if the two are equal to the
last bit, since every Davidson step reads them; that is what this checks, on
real projectors, under ``jit`` as the solver runs them, at block widths on both
sides of ``nkb`` so that both of the branches the shapes choose between run,
and for the three contractions the operators make: a scalar state, a spinor
whose components share a sphere, and a spiral's, whose components do not.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat import Calculator
from defumat.hamiltonian.operator import conjugated_contraction
from tests.backend import assert_same

pytestmark = pytest.mark.unit

ULTRASOFT = Path(__file__).resolve().parents[2] / "benchmarks" / "si2-us-1k.in"


@pytest.fixture(scope="module")
def vkb(pseudo_dir):
    return Calculator.from_file(ULTRASOFT, pseudo_dir=pseudo_dir).calculation.projectors.at_k(0)


@pytest.mark.parametrize("kind", ["scalar", "spinor", "spiral"])
def test_both_sides_of_the_conjugate_give_the_same_bits(vkb, kind):
    npwx, nkb = vkb.shape
    rng = np.random.default_rng(7)
    if kind == "spiral":
        # the two components' projectors are different arrays, stacked
        projectors = jnp.stack([vkb, vkb * jnp.exp(1j * jnp.arange(npwx))[:, None]])
    else:
        projectors = vkb
    subscripts = {"scalar": "gk,...g->...k", "spinor": "gk,...ag->...ak",
                  "spiral": "agk,...ag->...ak"}[kind]

    old = jax.jit(lambda p, v: jnp.einsum(subscripts, p.conj(), v))
    new = jax.jit(lambda p, v: jnp.einsum(subscripts, p, v.conj()).conj())
    chosen = jax.jit(lambda p, v: conjugated_contraction(p, v, subscripts))
    for nvec in (4, nkb // 2, nkb, 2 * nkb):
        shape = (nvec, npwx) if kind == "scalar" else (nvec, 2, npwx)
        states = jnp.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape),
                             dtype=vkb.dtype)
        reference = np.asarray(old(projectors, states))
        # Bit for bit is the CPU's promise; a card's contractions round in
        # their own order.
        atol = 1e-13 * float(np.max(np.abs(reference)))
        assert_same(new(projectors, states), reference, atol, err_msg=str(nvec))
        assert_same(chosen(projectors, states), reference, atol, err_msg=str(nvec))
