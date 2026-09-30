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
import tempfile
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
    quiet = dict(checkpointing=False, residual_solver=False, rotate_moments=False,
                 field=None)
    _refuse_under_pools(pools, calculation, **quiet)
    _refuse_under_pools(pools, calculation, **{**quiet, "checkpointing": True})
    _refuse_under_pools(pools, calculation, **{**quiet, "field": object()})
    _refuse_under_pools(pools, calculation, **{**quiet, "rotate_moments": True})
    for name, value, words in [("residual_solver", True, "residual scf_solver")]:
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


def _start_pools(script: str, size: int, *args, timeout: float = 300.0,
                 env_extra: dict | None = None) -> list[tuple[int, str, str]]:
    """Run ``script`` as ``size`` pools; ``(returncode, stdout, stderr)`` per rank."""
    port = _free_port()
    base = {k: v for k, v in os.environ.items()
            if not k.startswith(("DEFUMAT_POOL", "DEFUMAT_COORDINATOR"))}
    base.update(JAX_PLATFORMS="cpu", DEFUMAT_THREADS="2",
                PYTHONPATH=str(REPO) + os.pathsep + base.get("PYTHONPATH", ""))
    base.update(env_extra or {})
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
    outcomes = []
    try:
        for process in processes:
            out, err = process.communicate(timeout=timeout)
            outcomes.append((process.returncode, out, err))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
    return outcomes


def _run_pools(script: str, size: int, *args, timeout: float = 300.0,
               env_extra: dict | None = None) -> list[dict]:
    """Run ``script`` as ``size`` pools; each rank's last stdout line is JSON."""
    results = []
    for code, out, err in _start_pools(script, size, *args, timeout=timeout,
                                       env_extra=env_extra):
        assert code == 0, err[-2000:]
        results.append(json.loads(out.strip().splitlines()[-1]))
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


@pytest.mark.slow
def test_gloo_can_be_routed_through_a_named_interface():
    """``DEFUMAT_POOL_INTERFACE`` reaches gloo; a name the machine lacks fails at start.

    On Triton the hostname resolves to ``eth0`` and gloo ran at about 100 MB/s
    between nodes; the interface is how ``ib0`` is chosen instead. A wrong name
    must stop the run at start-up rather than fall back silently.
    """
    for result in _run_pools(COMMUNICATOR_SCRIPT, 2,
                             env_extra={"DEFUMAT_POOL_INTERFACE": "lo"}):
        assert np.allclose(result["real"], 3.0)
    outcomes = _start_pools(COMMUNICATOR_SCRIPT, 2, timeout=120,
                            env_extra={"DEFUMAT_POOL_INTERFACE": "nosuchif0"})
    for code, _, err in outcomes:
        assert code != 0 and "nosuchif0" in err


DEAD_POOL_SCRIPT = textwrap.dedent("""
    import os, time
    import numpy as np
    import defumat
    from defumat.parallel import current_pools
    p = current_pools()
    for step in range(20):
        if p.rank == p.size - 1 and step == 3:
            os._exit(3)
        p.allreduce_sum(np.ones(8))
    print('{"survived": true}')
""")


@pytest.mark.slow
def test_a_dead_pool_takes_the_others_down_within_the_heartbeat():
    """One pool leaving must end the run, not hang it.

    Measured: the survivors block in their all-reduce until the coordination
    service misses the dead rank's heartbeat and aborts them, 23 s after the
    death at a 20 s heartbeat. A hang would show here as the harness's timeout.
    """
    import time

    start = time.perf_counter()
    outcomes = _start_pools(DEAD_POOL_SCRIPT, 3, timeout=150,
                            env_extra={"DEFUMAT_POOL_HEARTBEAT": "15"})
    elapsed = time.perf_counter() - start
    assert outcomes[-1][0] == 3
    for code, out, _ in outcomes[:-1]:
        assert code != 0 and "survived" not in out
    assert elapsed < 120


SPIRAL_SCRIPT = SCF_SCRIPT.replace(
    "text, drop_reduce = sys.argv[1], sys.argv[2] == \"1\"",
    "text, drop_reduce = Path(sys.argv[1]).read_text(), sys.argv[2] == \"1\"")


@pytest.mark.slow
@pytest.mark.parametrize("name, iterations", [("h-chain-spiral.in", 6),
                                              ("o-chain-spiral-us.in", 5)])
def test_pools_reproduce_one_on_a_spin_spiral(name, iterations):
    """A spiral's k row carries two basis rows, and the pools need nothing more.

    The up component lives at ``k + q/2`` and the down at ``k - q/2``, each on
    its own sphere, and the streamed passes hand the kernels global k rows that
    the spiral's Hamiltonian, ``becsum`` and density map to both. Measured at a
    fixed iteration count: 5e-15 Ry at 1, 2 and 3 pools on the hydrogen chain,
    7e-14 at 1 and 2 on the ultrasoft oxygen chain, whose augmentation charge is
    the table displaced by ``q``. The dropped reduction must move the energy.
    """
    path = str(REPO / "tests" / "data" / "qe" / name)
    serial = _run_pools(SPIRAL_SCRIPT, 1, path, 0, iterations, timeout=900)[0]
    pooled = _run_pools(SPIRAL_SCRIPT, 2, path, 0, iterations, timeout=900)
    for result in pooled:
        assert abs(result["energy"] - serial["energy"]) < 1e-10
        assert result["store"] == "PoolStore"
    if name.startswith("h-chain"):
        broken = _run_pools(SPIRAL_SCRIPT, 2, path, 1, iterations, timeout=900)
        assert abs(broken[0]["energy"] - serial["energy"]) > 1e-4


CONTINUE_SCRIPT = textwrap.dedent("""
    import json, sys, tempfile, warnings
    warnings.simplefilter("ignore")
    from pathlib import Path
    import defumat
    from defumat.parallel import Pools, SerialCommunicator, current_pools
    from defumat.io.pwin import read_pw_input
    from defumat.pseudo import read_upf
    from defumat.scf.driver import Calculation, run_scf
    from defumat.system import build_system
    with tempfile.NamedTemporaryFile("w", suffix=".in", delete=False) as f:
        f.write(sys.argv[1])
    system = build_system(read_pw_input(Path(f.name)))
    pseudos = tuple(read_upf(Path("tests/data/pseudo") / s.pseudo_file)
                    for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    first = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-12)
    again = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-12,
                    starting_from=first)
    serial = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-12,
                     pools=Pools(SerialCommunicator()))
    from_serial = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-12,
                          starting_from=serial)
    print(json.dumps({"rank": current_pools().rank,
                      "first": float(first.total_energy), "first_it": first.iterations,
                      "again": float(again.total_energy), "again_it": again.iterations,
                      "serial": float(serial.total_energy),
                      "from_serial": float(from_serial.total_energy),
                      "from_serial_it": from_serial.iterations,
                      "seed_store": type(first.wavefunctions).__name__,
                      "serial_store": type(serial.wavefunctions).__name__}))
""")


@pytest.mark.slow
def test_a_pooled_run_continues_from_a_pooled_or_a_whole_set_seed():
    """Both seeds a pool can meet: its own rows, and a whole set sliced to them.

    A continuation from a converged state must land on the same energy in one
    or two iterations; restarting from the atomic orbitals instead (what a span
    dropped for the wrong k count does, behind a warning) takes the full count.
    """
    for result in _run_pools(CONTINUE_SCRIPT, 2, SILICON_8K, timeout=900):
        assert result["seed_store"] == "PoolStore" and result["serial_store"] != "PoolStore"
        assert abs(result["again"] - result["first"]) < 1e-10
        assert abs(result["from_serial"] - result["serial"]) < 1e-10
        assert abs(result["serial"] - result["first"]) < 1e-10
        assert result["again_it"] <= 2 and result["from_serial_it"] <= 2
        assert result["first_it"] > 4


def test_a_span_from_another_pool_layout_is_refused_by_name():
    """A pool's span holding other rows must not be read by position."""
    from defumat.io.pwin import read_pw_input
    from defumat.pseudo import read_upf
    from defumat.scf.driver import Calculation
    from defumat.system import build_system

    with tempfile.NamedTemporaryFile("w", suffix=".in", delete=False) as handle:
        handle.write(SILICON_8K)
    system = build_system(read_pw_input(Path(handle.name)))
    pseudos = tuple(read_upf(REPO / "tests" / "data" / "pseudo" / s.pseudo_file)
                    for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    hamiltonians = calculation.hamiltonian(
        calculation.potential(calculation.starting_density()).v_scf)
    npwx = calculation.basis.npwx
    span = PoolStore(np.ones((1, 2, 4, npwx), complex), [0, 1], nk=8)
    taken = calculation.starting_wavefunctions(hamiltonians, 4, span=span, rows=[1])
    assert np.shape(taken)[1] == 1
    with pytest.raises(NotImplementedError, match="different k-point pool layout"):
        calculation.starting_wavefunctions(hamiltonians, 4, span=span, rows=[2, 3])


CHECKPOINT_SCRIPT = textwrap.dedent("""
    import json, sys, tempfile, warnings
    warnings.simplefilter("ignore")
    from pathlib import Path
    import defumat
    from defumat.parallel import current_pools
    from defumat.io.pwin import read_pw_input
    from defumat.pseudo import read_upf
    from defumat.scf.driver import run_scf
    from defumat.system import build_system
    text, directory, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    with tempfile.NamedTemporaryFile("w", suffix=".in", delete=False) as f:
        f.write(text)
    system = build_system(read_pw_input(Path(f.name)))
    pseudos = tuple(read_upf(Path("tests/data/pseudo") / s.pseudo_file)
                    for s in system.structure.species)
    if mode == "resume-drop":
        # What a span dropped for the wrong k count does: the density and the
        # mixer come back and the states restart from the atomic orbitals.
        import defumat.scf.continuation as continuation
        continuation.promote_wavefunctions = lambda result, calculation: None
    options = {"whole": {}, "stop": {"max_iterations": 4, "checkpoint_dir": directory,
                                     "checkpoint_every": 2},
               "resume": {"checkpoint_dir": directory},
               "resume-drop": {"checkpoint_dir": directory}}[mode]
    result = run_scf(system, pseudos, conv_thr=1e-12, **options)
    print(json.dumps({"rank": current_pools().rank, "energy": float(result.total_energy),
                      "iterations": result.iterations, "converged": bool(result.converged),
                      "history": [(h["iteration"], h["total_energy"])
                                  for h in result.history],
                      "files": sorted(p.name for p in Path(directory).glob("*"))}))
""")


@pytest.mark.slow
def test_a_pooled_checkpoint_resumes_at_any_pool_count(tmp_path):
    """Written by two pools, resumed by two, by three and by one process.

    The resumed run must converge in the same total number of iterations as the
    uninterrupted one and to the same energy, which is the serial restart test's
    bar (``test_scf_restart.py``); a store that came back with the wrong rows or
    from the atomic orbitals would cost iterations. Only one iteration's rows
    files may be left beside the state file, and only after it is written.
    """
    import shutil

    whole = _run_pools(CHECKPOINT_SCRIPT, 2, SILICON_8K, tmp_path / "unused", "whole",
                       timeout=900)[0]
    stopped_dir = tmp_path / "stopped"
    stopped = _run_pools(CHECKPOINT_SCRIPT, 2, SILICON_8K, stopped_dir, "stop",
                         timeout=900)[0]
    assert not stopped["converged"] and stopped["iterations"] == 4
    rows_files = [name for name in stopped["files"] if ".rows" in name]
    assert len(rows_files) == 2 and all(".it000004." in name for name in rows_files)
    assert "scf_iteration.npz" in stopped["files"] and "scf_mixer.npz" in stopped["files"]
    energies = dict(whole["history"])
    for size in (2, 3, 1):
        directory = tmp_path / f"resume{size}"
        shutil.copytree(stopped_dir, directory)
        for result in _run_pools(CHECKPOINT_SCRIPT, size, SILICON_8K, directory,
                                 "resume", timeout=900):
            assert result["converged"]
            assert result["iterations"] == whole["iterations"], size
            assert abs(result["energy"] - whole["energy"]) < 1e-10, size
            # It resumed rather than restarted: ``iterations`` is absolute, so a
            # fresh start would pass the two lines above. The first iteration
            # back is iteration 5, and it is the uninterrupted run's iteration 5
            # to round-off, which it can only be if the states came back too.
            first, energy = result["history"][0]
            assert first == 5 and len(result["history"]) == whole["iterations"] - 4
            assert abs(energy - energies[5]) < 1e-11, (size, energy - energies[5])
    # The witness fires: states dropped behind the restored density leave
    # iteration 5 somewhere else.
    directory = tmp_path / "dropped"
    shutil.copytree(stopped_dir, directory)
    dropped = _run_pools(CHECKPOINT_SCRIPT, 2, SILICON_8K, directory, "resume-drop",
                         timeout=900)[0]
    assert abs(dict(dropped["history"])[5] - energies[5]) > 1e-9
    # A rows file missing is refused by name, not a store with a hole.
    directory = tmp_path / "holed"
    shutil.copytree(stopped_dir, directory)
    next(directory.glob("*.rows*.npz")).unlink()
    for code, _, err in _start_pools(CHECKPOINT_SCRIPT, 2, SILICON_8K, directory,
                                     "resume", timeout=900):
        assert code != 0 and "could not be read by every k-point pool" in err
    assert any("no rows file of that iteration holds" in err
               for _, _, err in _start_pools(CHECKPOINT_SCRIPT, 2, SILICON_8K, directory,
                                             "resume", timeout=900))


RELAX_CHECKPOINT_SCRIPT = textwrap.dedent("""
    import json, sys, tempfile, warnings
    warnings.simplefilter("ignore")
    from pathlib import Path
    import numpy as np
    import defumat
    from defumat.io.pwin import read_pw_input
    from defumat.parallel import current_pools
    from defumat.pseudo import read_upf
    from defumat.system import build_system
    from defumat.workflows.relax import run_relax
    text, directory, nstep = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with tempfile.NamedTemporaryFile("w", suffix=".in", delete=False) as f:
        f.write(text)
    system = build_system(read_pw_input(Path(f.name)))
    pseudos = tuple(read_upf(Path("tests/data/pseudo") / s.pseudo_file)
                    for s in system.structure.species)
    result = run_relax(system, pseudos, nstep=nstep, conv_thr=1e-10,
                       checkpoint_dir=None if directory == "-" else directory)
    print(json.dumps({"rank": current_pools().rank, "steps": len(result.steps),
                      "converged": bool(result.converged),
                      "energy": float(result.scf.total_energy),
                      "positions": np.asarray(result.system.structure.positions).tolist(),
                      "files": sorted(p.name for p in Path(directory).glob("*"))
                               if directory != "-" else []}))
""")


@pytest.mark.slow
def test_a_pooled_relaxation_checkpoint_resumes_where_it_stopped(tmp_path):
    """Rank 0 writes the step, the optimizer and a state without the states.

    The serial test's bar: stopping at step 2 and resuming costs the same total
    number of ionic steps as not stopping, which only holds if the optimizer's
    history crossed the file, and the resumed geometry and energy are the
    uninterrupted run's.
    """
    whole = _run_pools(RELAX_CHECKPOINT_SCRIPT, 2, SILICON_DISPLACED_8K, "-", 20,
                       timeout=1500)[0]
    stopped = _run_pools(RELAX_CHECKPOINT_SCRIPT, 2, SILICON_DISPLACED_8K,
                         tmp_path, 2, timeout=1500)[0]
    assert not stopped["converged"] and stopped["steps"] == 2
    assert set(stopped["files"]) == {"scf_state.npz", "optimizer.npz", "relax_step.json"}
    resumed = _run_pools(RELAX_CHECKPOINT_SCRIPT, 2, SILICON_DISPLACED_8K,
                         tmp_path, 20, timeout=1500)
    for result in resumed:
        assert whole["converged"] and result["converged"]
        assert stopped["steps"] + result["steps"] == whole["steps"]
        assert abs(result["energy"] - whole["energy"]) < 1e-9
        assert np.allclose(result["positions"], whole["positions"], atol=1e-5)


SIGTERM_SCRIPT = textwrap.dedent("""
    import json, os, signal, sys, tempfile, threading, warnings
    warnings.simplefilter("ignore")
    from pathlib import Path
    import defumat
    from defumat.parallel import current_pools
    from defumat.io.pwin import read_pw_input
    from defumat.pseudo import read_upf
    from defumat.scf.driver import Calculation, run_scf
    from defumat.system import build_system
    text, directory, victim = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with tempfile.NamedTemporaryFile("w", suffix=".in", delete=False) as f:
        f.write(text)
    system = build_system(read_pw_input(Path(f.name)))
    pseudos = tuple(read_upf(Path("tests/data/pseudo") / s.pseudo_file)
                    for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    run_scf(system, pseudos, calculation=calculation, max_iterations=2, verbose=False)
    pools = current_pools()
    if pools.rank == victim:
        threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-30,
                     max_iterations=400, checkpoint_dir=directory,
                     checkpoint_every=1000, verbose=False)
    print(json.dumps({"rank": pools.rank, "iterations": result.iterations,
                      "converged": bool(result.converged),
                      "files": sorted(p.name for p in Path(directory).glob("*"))}))
""")


@pytest.mark.slow
@pytest.mark.parametrize("victim", [0, 1])
def test_a_sigterm_stops_every_pool_at_the_same_iteration(tmp_path, victim):
    """JAX's preemption service catches the signal; the pools agree where to stop.

    Whichever pool is signalled, every pool leaves the loop at the same
    iteration, long before its iteration budget, exits cleanly and leaves the
    checkpoint of that iteration, which is what a Slurm ``--signal=TERM@`` lead
    buys a job at its wall clock.

    **``not converged`` is the witness, and it was checked to fire.** Unsignalled,
    this SCF converges at iteration 63 even at ``conv_thr = 1e-30``, because its
    residual reaches exactly zero; with the poll disabled the signalled pools did
    the same, 63 iterations and converged, so a pool that ignored the notice
    fails here.
    """
    results = _run_pools(SIGTERM_SCRIPT, 2, SILICON_8K, tmp_path, victim, timeout=600)
    assert results[0]["iterations"] == results[1]["iterations"] < 400
    for result in results:
        assert not result["converged"]
        stamp = f".it{result['iterations']:06d}."
        assert "scf_iteration.npz" in result["files"]
        assert sum(stamp in name for name in result["files"]) == 2


IRON_FSM_8K = (REPO / "tests" / "data" / "qe" / "fe-fsm.in").read_text()
IRON_FSM_8K = IRON_FSM_8K[: IRON_FSM_8K.upper().index("K_POINTS")] \
    + "K_POINTS automatic\n 2 2 2 0 0 0\n"

FIELD_SCRIPT = SCF_SCRIPT.replace(
    'print(json.dumps({"rank": current_pools().rank, "energy": float(result.total_energy),',
    'print(json.dumps({"rank": current_pools().rank, "energy": float(result.total_energy),\n'
    '                  "field": np.asarray(result.magnetic_field.uniform).tolist(),\n'
    '                  "moment": float(result.magnetization),')


@pytest.mark.slow
def test_pools_reproduce_one_under_a_fixed_spin_moment_constraint():
    """Iron driven to 2 Bohr magnetons by Elk's feedback, at 8 k-points.

    ``fsm_update = 'elk'`` steps the field after every iteration from the output
    density, which each pool finishes itself; rank 0's field is everyone's after
    the step. After eight iterations the energy, the moment and the driven field
    must be one pool's, and identical on both pools.
    """
    serial = _run_pools(FIELD_SCRIPT, 1, IRON_FSM_8K, 0, 8, timeout=900)[0]
    pooled = _run_pools(FIELD_SCRIPT, 2, IRON_FSM_8K, 0, 8, timeout=900)
    assert np.abs(serial["field"]).max() > 1e-4, "a field that was driven"
    for result in pooled:
        assert abs(result["energy"] - serial["energy"]) < 1e-10
        assert abs(result["moment"] - serial["moment"]) < 1e-9
        assert np.allclose(result["field"], serial["field"], atol=1e-10)
    assert pooled[0]["field"] == pooled[1]["field"]


ROTATE_SCRIPT = textwrap.dedent("""
    import json, sys, tempfile, warnings
    warnings.simplefilter("ignore")
    from pathlib import Path
    import numpy as np
    import defumat
    from defumat.parallel import current_pools
    from defumat.io.pwin import read_pw_input
    from defumat.pseudo import read_upf
    from defumat.scf.driver import run_scf
    from defumat.system import build_system
    text, iterations, turn = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"
    start = float(sys.argv[4])
    with tempfile.NamedTemporaryFile("w", suffix=".in", delete=False) as f:
        f.write(text)
    system = build_system(read_pw_input(Path(f.name)))
    pseudos = tuple(read_upf(Path("tests/data/pseudo") / s.pseudo_file)
                    for s in system.structure.species)
    result = run_scf(system, pseudos, conv_thr=1e-16, max_iterations=iterations,
                     rotate_moments=turn, rotation_start=start, verbose=False)
    print(json.dumps({"rank": current_pools().rank,
                      "energy": float(result.total_energy),
                      "moment": list(map(float, result.magnetization_vector)),
                      "torque": list(map(float, result.orientation_torque))}))
""")

COBALT_OBLIQUE = (REPO / "tests" / "data" / "qe" / "co-tetragonal-relaxed-mae.in").read_text(
    ).replace("angle1(1) = 0.0, angle2(1) = 0.0,", "angle1(1) = 45.0, angle2(1) = 0.0,")

# The PAW anisotropy cell at 45 degrees, at a cutoff and a mesh a test can
# afford. It is far from converged in a few iterations, so the stepper is told
# to start at once (``rotation_start = 1``): the turn of the PAW states, and the
# ``becsum`` and density rebuilt from them, are what this case is for.
NICKEL_PAW_OBLIQUE = (REPO / "tests" / "data" / "qe" / "ni-tetragonal-relaxed-mae-paw.in"
                      ).read_text().replace(
    "angle1(1) = 0.0, angle2(1) = 0.0,", "angle1(1) = 45.0, angle2(1) = 0.0,").replace(
    "ecutwfc = 75.0, ecutrho = 480.0,", "ecutwfc = 45.0, ecutrho = 360.0,").replace(
    "3 3 2 0 0 0", "2 2 2 0 0 0")


@pytest.mark.slow
@pytest.mark.parametrize("text, iterations, start", [(COBALT_OBLIQUE, 10, 1e-5),
                                                     (NICKEL_PAW_OBLIQUE, 5, 1.0)],
                         ids=["ultrasoft-cobalt", "paw-nickel"])
def test_pools_reproduce_one_when_the_moments_are_turned(text, iterations, start):
    """``rotate_moments``: the torque is rank 0's, the turns are each pool's rows.

    The stepper keeps a quasi-Newton history of the torques it is handed and
    turns the states and, on PAW, rebuilds ``becsum`` and the density from the
    turned states, all of which are per k-point work reduced across the pools.
    After a fixed number of iterations with the moments turned, one pool and two
    must agree; a run that does not turn must land somewhere else, which is what
    shows the stepper acted.
    """
    serial = _run_pools(ROTATE_SCRIPT, 1, text, iterations, 1, start, timeout=1500)[0]
    still = _run_pools(ROTATE_SCRIPT, 1, text, iterations, 0, start, timeout=1500)[0]
    assert abs(serial["energy"] - still["energy"]) > 1e-8
    pooled = _run_pools(ROTATE_SCRIPT, 2, text, iterations, 1, start, timeout=1500)
    for result in pooled:
        assert abs(result["energy"] - serial["energy"]) < 1e-10
        assert np.allclose(result["moment"], serial["moment"], atol=1e-9)
        assert np.allclose(result["torque"], serial["torque"], atol=1e-10)
