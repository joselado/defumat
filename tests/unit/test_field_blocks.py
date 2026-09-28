"""The third-derivative drivers stack the field response's blocks once.

``GPU-MEMORY-NEXT.md`` item 12. ``raman_tensors`` and ``electrostriction``
read three lists off the field response -- the bare perturbations, their
solutions and the pre-tail solutions -- and each is a set of wavefunction-sized
blocks. They used to be stacked while the lists stayed alive in the field's
internals for the whole assembly, and the norm-conserving case stacked one
block twice, since there the pre-tail solutions *are* the bare perturbations.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.response.efield import DielectricTensor
from defumat.response.electrostriction import field_blocks

pytestmark = pytest.mark.unit


def _field(aliased: bool):
    rng = np.random.default_rng(0)
    bare = [jnp.asarray(rng.normal(size=(1, 2, 3, 5))) for _ in range(3)]
    dpsi = [jnp.asarray(rng.normal(size=(1, 2, 3, 5))) for _ in range(3)]
    commutators = list(bare) if aliased else [b + 1.0 for b in bare]
    internals = {"solver": "solver", "v_scf": "v_scf", "bare": bare,
                 "dpsi": dpsi, "commutators": commutators}
    field = DielectricTensor(epsilon=np.eye(3), born_charges=None,
                             internals=internals, induced_density=None,
                             history=[])
    return field, bare, dpsi, commutators


@pytest.mark.parametrize("aliased", [True, False], ids=["nc", "augmented"])
def test_each_block_is_stacked_once_and_the_internals_are_released(aliased):
    field, bare, dpsi, commutators = _field(aliased)
    out, solver, v_scf, b, u, stored = field_blocks(field)
    assert (solver, v_scf) == ("solver", "v_scf")
    assert out.internals is None and field.internals is not None
    np.testing.assert_array_equal(b, jnp.stack(bare))
    np.testing.assert_array_equal(u, jnp.stack(dpsi))
    np.testing.assert_array_equal(stored, jnp.stack(commutators))
    # One block where the two lists hold the same objects, two where not.
    assert (stored is b) is aliased
