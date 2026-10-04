"""An SCF iteration reads its scalars from the device in one fetch, not one by one.

``OPEN.md`` Part XXIII item 21. The loop used to read about fourteen device
values an iteration, each a separate ``float()`` or ``np.asarray`` that waits
for its own producer: the Davidson step counts, the three halves of ``dr2``,
``eband``, ``deband`` and ``|drho|``, the Hartree and exchange-correlation
energies, and ``c`` of a meta-GGA even for a functional that has none. They now
come back in one ``jax.device_get`` per attempt, with every expression that
produces them unchanged, so no number moves (that half is the SCF itself, run
against the old code bit for bit on four cells; it is not repeated here).

What is counted is a **blocking read**: an ``ArrayImpl._value`` on an array
that has no host copy yet, which is the path ``float()`` takes; an
``np.asarray`` or ``np.array`` of such an array, which on a CPU goes through the
buffer protocol and never reaches ``_value``; and one ``jax.device_get``, whatever
it holds. Iterations are told apart by ``next_ethr``, which the loop calls once
at the top of every one.

What is left on the norm-conserving, unpolarized, fixed-occupation path is
five: the eigensolver's finiteness guard, the HOMO the occupations read, the one
fetch, and the mixer's two densities. Before the change it was fifteen.
"""

import jax
import numpy as np
import pytest
from jax._src import array as jax_array

import defumat.scf.driver as driver
from defumat.calculator import Calculator

pytestmark = pytest.mark.unit

SILICON = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.20, nat = 2, ntyp = 1, ecutwfc = 12.0
/
&electrons
  conv_thr = 1.0d-9
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""


def test_a_steady_iteration_blocks_on_the_host_five_times(pseudo_dir, monkeypatch):
    calculator = Calculator.from_text(SILICON, pseudo_dir, announce=False)
    calculator.get_scf(max_iterations=2)  # compiled here, so only the run below is counted

    count = {"reads": 0, "inside": 0}
    marks = []

    value = jax_array.ArrayImpl._value

    def counted_value(self):
        if self._npy_value is None and count["inside"] == 0:
            count["reads"] += 1
        return value.fget(self)

    def entered(original, counts_as_one):
        def wrapped(x, *args, **kwargs):
            if count["inside"] == 0 and counts_as_one(x):
                count["reads"] += 1
            count["inside"] += 1
            try:
                return original(x, *args, **kwargs)
            finally:
                count["inside"] -= 1
        return wrapped

    def unread(x):
        return isinstance(x, jax.Array) and x._npy_value is None

    def any_unread(tree):
        return any(unread(leaf) for leaf in jax.tree_util.tree_leaves(tree))

    next_ethr = driver.next_ethr

    def marked(*args, **kwargs):
        marks.append(count["reads"])
        return next_ethr(*args, **kwargs)

    monkeypatch.setattr(jax_array.ArrayImpl, "_value", property(counted_value))
    monkeypatch.setattr(jax, "device_get", entered(jax.device_get, any_unread))
    monkeypatch.setattr(np, "asarray", entered(np.asarray, unread))
    monkeypatch.setattr(np, "array", entered(np.array, unread))
    monkeypatch.setattr(driver, "next_ethr", marked)

    result = calculator.get_scf(max_iterations=12)
    assert result.converged and result.iterations >= 5, result.iterations
    # One count per iteration but the last, which has no mark after it and is
    # the converged branch's. The first builds its starting states, so the
    # steady state is every one after it.
    per_iteration = np.diff(marks)
    steady = per_iteration[1:]
    assert steady.size >= 3
    assert steady.max() <= 5, per_iteration.tolist()
