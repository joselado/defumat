"""On an ultrasoft dataset Davidson projects each block it applies ``H`` to once.

``OPEN.md`` Part XXIII item 18. ``h_psi`` computes ``becp = <beta|psi>`` once
(``h_psi.f90:231``) and ``s_psi`` reads it (``s_psi.f90:15``). The solver here
called ``apply`` and then ``s_projections`` on the same block, and each took its
own ``calbec``: the first on the masked block inside ``_nonlocal``, the second on
the block as passed. The two products have different operands, so XLA did not
merge them, while their values were equal because the solver masks every block
before it applies ``H``. ``apply_projected`` returns ``(H psi, becp, q becp)``
from one ``calbec``.

Two things are checked. The values: one call returns exactly, to the bit, what
the two calls returned on a masked block, which is what makes the change free of
any effect on a number. And the count: tracing one solve with ``narrow = False``
(so the expansion block is traced once rather than once per rung of the band
ladder) takes ``calbec`` three times, the starting block, the expansion block and
the refresh, where the two-call form took it five times. The compiled program of
``si8-us-1k.in`` shows the same thing, one fewer matrix product in every rung of
the expansion branch (five to four) and in the starting block.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat.calculator import Calculator
from defumat.hamiltonian.operator import Hamiltonian
from defumat.solvers.davidson import davidson_eigensolver

pytestmark = pytest.mark.unit

CELL = Path(__file__).resolve().parents[2] / "benchmarks" / "si2-us-1k.in"


@pytest.fixture(scope="module")
def ultrasoft(pseudo_dir):
    """An ultrasoft Hamiltonian at the starting potential, and its band count."""
    calculation = Calculator.from_file(str(CELL), pseudo_dir=str(pseudo_dir),
                                       announce=False).calculation
    potential = calculation.potential(calculation.starting_density())
    becsum = calculation.starting_becsum()
    _, ddd_paw = calculation.onecenter(becsum) if becsum else (None, None)
    hamiltonian = calculation.hamiltonian(potential.v_scf, ddd_paw)[0]
    assert hamiltonian.has_overlap, "the cell must carry an augmentation charge"
    return hamiltonian, calculation.band_count()


def test_one_call_returns_what_the_two_returned_bit_for_bit(ultrasoft):
    hamiltonian, nbnd = ultrasoft
    rng = np.random.default_rng(0)
    shape = (nbnd, hamiltonian.ndim)
    block = jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape),
                        dtype=hamiltonian.dtype)
    # Masked, as every block the solver hands over is.
    block = jnp.where(hamiltonian.state_mask[0], block, 0.0)

    h, becp, becq = hamiltonian.apply_projected(block, 0)
    becp_alone, becq_alone = hamiltonian.s_projections(block, 0)

    assert np.array_equal(np.asarray(h), np.asarray(hamiltonian.apply(block, 0)))
    assert np.array_equal(np.asarray(becp), np.asarray(becp_alone))
    assert np.array_equal(np.asarray(becq), np.asarray(becq_alone))


def test_a_solve_projects_each_block_once(ultrasoft, monkeypatch):
    hamiltonian, nbnd = ultrasoft
    traced = []
    real = Hamiltonian._becp

    def counting(self, vectors, vkb):
        traced.append(vectors.shape)
        return real(self, vectors, vkb)

    monkeypatch.setattr(Hamiltonian, "_becp", counting)
    jax.make_jaxpr(
        lambda h: davidson_eigensolver(h, 0, nbnd, narrow=False)
    )(hamiltonian)

    # The starting block, the expansion block inside the loop, and the refresh's
    # projections of the vectors it keeps. Five with ``apply`` and
    # ``s_projections`` called separately.
    assert len(traced) == 3, (
        f"calbec traced {len(traced)} times ({traced}); one per block is 3, "
        "and 5 is H and S projecting the starting and the expansion block twice"
    )
