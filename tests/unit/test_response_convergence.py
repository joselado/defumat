"""The response loops' convergence test and CG schedule, as ``ph.x`` has them.

``ddv_scf`` is ``mix_pot``'s ``dr2`` divided by ``npert``; ``pass_threshold`` is
``dfpt_kernels``' schedule; and a schedule that changes the CG threshold every
pass must not compile a new program every pass, which a Python float closed
over would (``defumat.eager`` prints it into the key).
"""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from defumat.response.mixing import ddv_scf
from defumat.response.sternheimer import FIRST_PASS_THRESHOLD, pass_threshold


def _mix_pot_dr2(vectors):
    """``mix_pot.f90:77-83`` literally: the whole vector in reals."""
    reals = np.concatenate([np.stack([v.real, v.imag], -1).ravel() for v in vectors])
    ndimtot = reals.size
    return (np.linalg.norm(reals) / ndimtot) ** 2


def test_the_joint_test_is_mix_pots_dr2_over_npert():
    rng = np.random.default_rng(0)
    changes = [rng.standard_normal((2, 5, 6)) + 1j * rng.standard_normal((2, 5, 6))
               for _ in range(3)]
    expected = _mix_pot_dr2(changes) / 3
    assert ddv_scf(changes, joint=True) == pytest.approx(expected, rel=1e-13)
    # The device route gives the same number.
    assert ddv_scf([jnp.asarray(c) for c in changes], joint=True) == pytest.approx(
        expected, rel=1e-13)


def test_the_one_centre_block_is_in_the_sum_and_the_count():
    rng = np.random.default_rng(1)
    grid = [rng.standard_normal(40) for _ in range(3)]
    onecentre = [rng.standard_normal(7) for _ in range(3)]
    expected = _mix_pot_dr2(
        [np.concatenate([g, o]) for g, o in zip(grid, onecentre)]) / 3
    assert ddv_scf(grid, onecentre, joint=True) == pytest.approx(expected, rel=1e-13)


def test_the_per_mode_test_is_the_worst_single_perturbation():
    rng = np.random.default_rng(2)
    changes = [scale * rng.standard_normal((4, 4)) for scale in (1.0, 3.0, 0.5)]
    expected = max(_mix_pot_dr2([c]) for c in changes)
    assert ddv_scf(changes, joint=False) == pytest.approx(expected, rel=1e-13)
    # ... which is never looser than the joint test over the same modes.
    assert ddv_scf(changes, joint=False) >= ddv_scf(changes, joint=True)


class _Solver:
    schedule = True


def test_the_schedule_is_dfpt_kernels():
    solver = _Solver()
    assert pass_threshold(solver, []) == FIRST_PASS_THRESHOLD == 1.0e-2
    assert pass_threshold(solver, [1.0e-6]) == pytest.approx(1.0e-4)
    assert pass_threshold(solver, [3.0e-2, 1.0e-10]) == pytest.approx(1.0e-6)
    # Capped at the first pass's value, however large the last change was.
    assert pass_threshold(solver, [1.0]) == FIRST_PASS_THRESHOLD


def test_an_unscheduled_solver_keeps_its_own_threshold():
    class Fixed:
        pass

    assert pass_threshold(Fixed(), [1.0e-6]) is None
    solver = _Solver()
    solver.schedule = False
    assert pass_threshold(solver, [1.0e-6]) is None


@pytest.mark.slow
def test_a_new_threshold_and_start_reuse_the_compiled_solve(tmp_path):
    """Two passes of a schedule: the second compiles nothing.

    The threshold reaches the loop as an array and the start as an array, both
    arguments of the program :mod:`defumat.eager` keeps, so a second solve with
    another threshold and the first one's solution as its start is the same
    program. Counted on the ``eager`` cache: a Python float in the closure would
    add one entry per distinct value.
    """
    from pathlib import Path

    from defumat import Calculator
    from defumat import eager
    from defumat.response.sternheimer import make_sternheimer

    root = Path(__file__).resolve().parents[2]
    calculator = Calculator.from_file(root / "benchmarks" / "si-1k.in",
                                      pseudo_dir=root / "tests" / "data" / "pseudo",
                                      announce=False)
    result = calculator.get_scf()
    solver = make_sternheimer(calculator.calculation, result)
    bare = jnp.asarray(solver.psi) * 0.1

    def perturbation(psi, ik, spin, b=bare):
        return b[spin][ik]

    first = solver.solve(perturbation, start=jnp.zeros_like(bare), threshold=1.0e-2)
    before = len(eager._PROGRAMS)
    second = solver.solve(perturbation, start=first.dpsi, threshold=1.0e-6)
    assert len(eager._PROGRAMS) == before
    # The warm start and the tighter threshold took effect.
    assert second.residual < 1.0e-5
