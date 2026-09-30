"""k-point pools (``defumat.parallel``): the split, the store, the collectives, the SCF.

The multi-process tests start real processes over a localhost coordinator, one
per pool, each with a hard timeout: a collective that one rank never reaches
hangs the others, and a test must fail rather than hang the gate.

The SCF invariance test is the ``k_batch`` rule extended to pools: the number
of pools must not be visible in any result beyond round-off, since the pools
change only the order the k contributions are added in. It runs at
``conv_thr = 1e-12`` and asserts 1e-10 Ry, and the starting states are drawn
with one fixed key per k-point (``Calculation.starting_wavefunctions``), so the
split cannot change where the SCF starts. Its guard, removing the all-reduce,
must move the energy by far more than that, which is what shows the invariance
test can see a pool that sums only its own share.

Every test that starts processes is ``slow``: each pays a process start, the
distributed handshake and a cold compile, 15 to 30 s, and none is a number
against a reference code, which is the gate's rule (the ``test-runs`` skill).
The split, the store and the refusals stay in the gate.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from defumat.parallel import PoolStore, Pools, SerialCommunicator

REPO = Path(__file__).resolve().parents[2]

SILICON_8K = """\
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.2, nat = 2, ntyp = 1, ecutwfc = 12.0,
  nosym = .true., noinv = .true.
/
&electrons
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""

# The regime the pools are for: a noncollinear magnet (so no time reversal),
# ultrasoft, smeared, without symmetry -- benchmarks/fe-mag-1k.in on a 2x2x2 grid.
IRON_NONCOLLINEAR_8K = (REPO / "benchmarks" / "fe-mag-1k.in").read_text()
IRON_NONCOLLINEAR_8K = IRON_NONCOLLINEAR_8K[: IRON_NONCOLLINEAR_8K.upper().index("K_POINTS")] \
    + "K_POINTS automatic\n 2 2 2 0 0 0\n"
IRON_NONCOLLINEAR_8K = IRON_NONCOLLINEAR_8K.replace(
    "&system", "&system\n  nosym = .true., noinv = .true.,", 1)


class _FakeCommunicator:
    def __init__(self, size, rank):
        self.size, self.rank = size, rank


def test_counts_and_rows_are_contiguous_blocks():
    """``divide_et_impera``'s split: contiguous, the remainder to the first pools."""
    pools = [Pools(_FakeCommunicator(3, rank)) for rank in range(3)]
    assert pools[0].counts(10) == [4, 3, 3]
    rows = [p.rows(10) for p in pools]
    assert [list(r) for r in rows] == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert np.array_equal(np.concatenate(rows), np.arange(10))


def test_serial_pools_are_the_identity():
    pools = Pools(SerialCommunicator())
    local = np.arange(12.0).reshape(1, 3, 4)
    assert pools.size == 1 and pools.rank == 0
    assert np.array_equal(pools.gather_k(local, 3), local)
    tree = (np.ones(3), (), None)
    assert pools.allreduce_sum(tree) is tree
    assert pools.broadcast_flag(True) is True


def test_a_pool_store_cannot_be_read_as_the_whole_set():
    """The whole point of the type: a whole-set consumer must fail, not misread."""
    store = PoolStore(np.zeros((1, 2, 3, 4), complex), [5, 6], nk=9)
    with pytest.raises(TypeError, match="one k-point pool's share"):
        np.asarray(store)
    with pytest.raises(ValueError):
        PoolStore(np.zeros((1, 2, 3, 4)), [5], nk=9)


def test_refusals_name_what_pools_do_not_cover():
    from types import SimpleNamespace

    from defumat.scf.driver import _refuse_under_pools

    pools = Pools(_FakeCommunicator(2, 0))
    calculation = SimpleNamespace(system=SimpleNamespace(kpoints=SimpleNamespace(nk=8)),
                                  spiral=False)
    quiet = dict(starting_from=None, starting_wavefunctions=None, checkpointing=False,
                 residual_solver=False, rotate_moments=False, field=None)
    _refuse_under_pools(pools, calculation, **quiet)
    for name, value, words in [("checkpointing", True, "checkpoint_dir"),
                               ("rotate_moments", True, "rotate_moments"),
                               ("field", object(), "magnetic field")]:
        with pytest.raises(NotImplementedError, match=words):
            _refuse_under_pools(pools, calculation, **{**quiet, name: value})
    few = SimpleNamespace(system=SimpleNamespace(kpoints=SimpleNamespace(nk=1)), spiral=False)
    with pytest.raises(NotImplementedError, match="2 pools for 1 k-points"):
        _refuse_under_pools(pools, few, **quiet)


# ----------------------------------------------------------------- processes


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]


def _run_pools(script: str, size: int, *args, timeout: float = 300.0) -> list[dict]:
    """Run ``script`` as ``size`` pools; each rank's last stdout line is JSON."""
    port = _free_port()
    base = {k: v for k, v in os.environ.items()
            if not k.startswith(("DEFUMAT_POOL", "DEFUMAT_COORDINATOR"))}
    base.update(JAX_PLATFORMS="cpu", DEFUMAT_THREADS="2",
                PYTHONPATH=str(REPO) + os.pathsep + base.get("PYTHONPATH", ""))
    processes = []
    for rank in range(size):
        env = dict(base)
        if size > 1:
            env.update(DEFUMAT_POOLS=str(size), DEFUMAT_POOL_RANK=str(rank),
                       DEFUMAT_COORDINATOR=f"localhost:{port}",
                       DEFUMAT_POOL_TIMEOUT="120")
        processes.append(subprocess.Popen(
            [sys.executable, "-c", script, *map(str, args)], env=env, cwd=REPO,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    results = []
    try:
        for process in processes:
            out, err = process.communicate(timeout=timeout)
            assert process.returncode == 0, err[-2000:]
            results.append(json.loads(out.strip().splitlines()[-1]))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
    return results


COMMUNICATOR_SCRIPT = textwrap.dedent("""
    import hashlib, json
    import numpy as np
    import defumat
    from defumat.parallel import current_pools
    p = current_pools()
    r = p.rank
    real = np.full((2, 3), r + 1.0)
    cplx = np.full((4,), (r + 1.0) + 1j * (r + 1.0))
    reduced = p.allreduce_sum((real, (), None, (cplx,)))
    local = np.full((1, 3 if r == 0 else 2, 5), float(r))
    gathered = p.gather_k(local, 5)
    told = p.broadcast_scalar(10.0 + r)
    flag = p.broadcast_flag(r == 0)
    digest = hashlib.sha256(np.asarray(reduced[0]).tobytes()
                            + np.asarray(reduced[3][0]).tobytes()).hexdigest()
    print(json.dumps({"rank": r, "real": np.asarray(reduced[0]).tolist(),
                      "cplx": [complex(z).real for z in np.asarray(reduced[3][0])],
                      "cplx_im": [complex(z).imag for z in np.asarray(reduced[3][0])],
                      "empty": reduced[1] == (), "none": reduced[2] is None,
                      "gathered": gathered[0, :, 0].tolist(), "told": told,
                      "flag": flag, "digest": digest}))
""")


@pytest.mark.slow
def test_the_facade_refuses_what_is_not_pool_aware(monkeypatch):
    """Every ``get_*`` outside ``POOL_AWARE`` refuses under pools, by name."""
    import inspect

    import defumat.parallel as parallel
    from defumat.calculator import POOL_AWARE, Calculator

    guarded = {name for name, method in vars(Calculator).items()
               if name.startswith("get_") and inspect.isfunction(method)
               and hasattr(method, "__wrapped__")}
    getters = {name for name, method in vars(Calculator).items()
               if name.startswith("get_") and inspect.isfunction(method)}
    assert guarded == getters - POOL_AWARE
    assert POOL_AWARE <= getters
    monkeypatch.setattr(parallel, "_CURRENT", Pools(_FakeCommunicator(2, 0)))
    with pytest.raises(NotImplementedError, match="get_bands is not available"):
        Calculator.get_bands(object())


@pytest.mark.slow
def test_collectives_over_two_processes():
    results = _run_pools(COMMUNICATOR_SCRIPT, 2)
    for result in results:
        assert np.allclose(result["real"], 3.0)
        assert np.allclose(result["cplx"], 3.0) and np.allclose(result["cplx_im"], 3.0)
        assert result["empty"] and result["none"]
        # rank 0 owns rows 0-2 (value 0), rank 1 rows 3-4 (value 1)
        assert result["gathered"] == [0.0, 0.0, 0.0, 1.0, 1.0]
        assert result["told"] == 10.0 and result["flag"] is True
    # Every rank holds the same bits of the reduced sum.
    assert results[0]["digest"] == results[1]["digest"]


SCF_SCRIPT = textwrap.dedent("""
    import json, sys, tempfile, warnings
    warnings.simplefilter("ignore")
    from pathlib import Path
    import numpy as np
    import defumat
    from defumat.parallel import Pools, current_pools
    from defumat.io.pwin import read_pw_input
    from defumat.pseudo import read_upf
    from defumat.scf.driver import run_scf
    from defumat.system import build_system
    text, drop_reduce = sys.argv[1], sys.argv[2] == "1"
    iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    if drop_reduce:
        Pools.allreduce_sum = lambda self, tree: tree
    with tempfile.NamedTemporaryFile("w", suffix=".in", delete=False) as f:
        f.write(text)
    system = build_system(read_pw_input(Path(f.name)))
    pseudos = tuple(read_upf(Path("tests/data/pseudo") / s.pseudo_file)
                    for s in system.structure.species)
    result = run_scf(system, pseudos, conv_thr=1e-12 if iterations == 100 else 1e-16,
                     max_iterations=iterations)
    forces = stress = None
    if len(sys.argv) > 4 and sys.argv[4] == "forces":
        from defumat.forces import compute_forces
        from defumat.scf.driver import Calculation
        from defumat.stress import compute_stress
        calculation = Calculation(system, pseudos)
        result = run_scf(system, pseudos, conv_thr=1e-12, calculation=calculation)
        forces = np.asarray(compute_forces(calculation, result).forces).tolist()
        stress = np.asarray(compute_stress(calculation, result).tensor).tolist()
    print(json.dumps({"rank": current_pools().rank, "energy": float(result.total_energy),
                      "forces": forces, "stress": stress,
                      "iterations": result.iterations,
                      "eigenvalues": np.asarray(result.eigenvalues).ravel().tolist(),
                      "store": type(result.wavefunctions).__name__}))
""")


@pytest.fixture(scope="module")
def serial_silicon():
    return _run_pools(SCF_SCRIPT, 1, SILICON_8K, 0)[0]


@pytest.mark.slow
def test_two_pools_reproduce_one(serial_silicon):
    pooled = _run_pools(SCF_SCRIPT, 2, SILICON_8K, 0)
    for result in pooled:
        assert abs(result["energy"] - serial_silicon["energy"]) < 1e-10
        assert result["iterations"] == serial_silicon["iterations"]
        assert np.allclose(result["eigenvalues"], serial_silicon["eigenvalues"], atol=1e-10)
        assert result["store"] == "PoolStore"
    assert pooled[0]["energy"] == pooled[1]["energy"]


@pytest.mark.slow
def test_three_pools_split_unevenly_and_reproduce_one(serial_silicon):
    pooled = _run_pools(SCF_SCRIPT, 3, SILICON_8K, 0)
    for result in pooled:
        assert abs(result["energy"] - serial_silicon["energy"]) < 1e-10


@pytest.mark.slow
def test_the_invariance_test_sees_a_missing_reduction(serial_silicon):
    """The guard: pools that each sum only their own share must fail the test above."""
    broken = _run_pools(SCF_SCRIPT, 2, SILICON_8K, 1)
    assert abs(broken[0]["energy"] - serial_silicon["energy"]) > 1e-3


@pytest.mark.slow
def test_pools_reproduce_one_on_a_noncollinear_magnet():
    """Iron with a noncollinear moment, ultrasoft, smeared, 8 k-points, no symmetry.

    Compared after eight iterations rather than at convergence: the pools do the
    same arithmetic up to the order of the k sums, so a pool defect shows at any
    iteration, and this SCF converges slowly. Measured 1.2e-13 Ry at 1, 2 and 3
    pools. The four-atom cobalt helix with spin-orbit coupling
    (``tests/data/qe/co-helix4-soc.in``) is *not* usable here: its early SCF is
    unstable enough that one pool with a different k-sum order parts by 8.7e-5 Ry
    at iteration 6, so only its first three iterations (3.1e-12 across pools)
    test anything, and it is too slow to converge in a test.
    """
    serial = _run_pools(SCF_SCRIPT, 1, IRON_NONCOLLINEAR_8K, 0, 8)[0]
    pooled = _run_pools(SCF_SCRIPT, 2, IRON_NONCOLLINEAR_8K, 0, 8)
    for result in pooled:
        assert abs(result["energy"] - serial["energy"]) < 1e-10
        assert result["iterations"] == serial["iterations"] == 8


SILICON_DISPLACED_8K = SILICON_8K.replace(" Si 0.25 0.25 0.25", " Si 0.27 0.25 0.24")


@pytest.mark.slow
def test_pooled_forces_and_stress_reproduce_one_pool():
    """One pool takes the single-pass gradient; two walk their rows and reduce.

    Two routes that share nothing but the energy expression, which agreed to
    every printed digit when measured: forces to 1e-12 Ry/bohr, stress to 1e-10.
    """
    serial = _run_pools(SCF_SCRIPT, 1, SILICON_DISPLACED_8K, 0, 100, "forces")[0]
    pooled = _run_pools(SCF_SCRIPT, 2, SILICON_DISPLACED_8K, 0, 100, "forces")
    assert np.abs(np.asarray(serial["forces"])).max() > 1e-2, "a force worth testing"
    for result in pooled:
        assert np.allclose(result["forces"], serial["forces"], atol=1e-10)
        assert np.allclose(result["stress"], serial["stress"], atol=1e-9)


RELAX_SCRIPT = textwrap.dedent("""
    import json, sys, tempfile, warnings
    warnings.simplefilter("ignore")
    from pathlib import Path
    import numpy as np
    import defumat
    from defumat import Calculator
    from defumat.parallel import current_pools
    with tempfile.NamedTemporaryFile("w", suffix=".in", delete=False) as f:
        f.write(sys.argv[1])
    calculator = Calculator.from_file(f.name, pseudo_dir="tests/data/pseudo")
    result = calculator.get_relax(conv_thr=1e-10)
    print(json.dumps({"rank": current_pools().rank,
                      "energies": [s.total_energy for s in result.steps],
                      "positions": np.asarray(result.steps[-1].positions).tolist(),
                      "converged": bool(result.converged)}))
""")


@pytest.mark.slow
def test_a_pooled_relaxation_takes_the_same_steps():
    """Rank 0's energy and forces drive every pool's optimizer.

    Measured: four ionic steps both ways, the energies equal to 1e-10 Ry and
    the final positions to 1e-8 bohr.
    """
    serial = _run_pools(RELAX_SCRIPT, 1, SILICON_DISPLACED_8K, timeout=900)[0]
    pooled = _run_pools(RELAX_SCRIPT, 2, SILICON_DISPLACED_8K, timeout=900)
    for result in pooled:
        assert result["converged"] and len(result["energies"]) == len(serial["energies"])
        assert np.allclose(result["energies"], serial["energies"], atol=1e-9)
        assert np.allclose(result["positions"], serial["positions"], atol=1e-7)
