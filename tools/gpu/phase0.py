"""GPU.md Phase 0 -- first contact: does it run, and does it give the same number?

One SCF, one case, one setting of each batching dial, with the five checks
``GPU.md`` names run in the order it names them and written to JSON. There is
nothing GPU-specific in here beyond the reporting: the *same* script is the CPU
baseline, run on a CPU node, which is the whole point -- §2.3's metric is GPU
pypresso against CPU pypresso on the same input and the same code, so the two
sides must not be two scripts.

    python3 tools/gpu/phase0.py si-1k --k-batch 1 --band-batch 1
    python3 tools/gpu/phase0.py al10-metal --k-batch all --band-batch all --json out.json

The case is a name resolved against ``benchmarks/`` and then ``tests/data/qe/``,
or a path to a ``pw.x`` input.

**Both dials are set explicitly and there is no default here**, deliberately.
``pypresso/batching.py`` reads ``PYPRESSO_K_BATCH`` and ``PYPRESSO_BAND_BATCH``
*at import time* and defaults both to one, which is what a cache wants and what
a GPU does not (``GPU.md`` §1) -- so this script sets both into the environment
before pypresso is imported and refuses to run without being told.

What each check answers, and what its failure means:

1. ``fp64`` -- x64 is asserted on a device array rather than trusted, because
   its failure mode is a plausible number rather than a crash; and the fp64/fp32
   rate is measured on the two kernels this code is actually made of, a matmul
   and a batched 3D FFT. That ratio is the datum ``GPU.md`` Phase 3's rank waits
   on.
2. ``energy`` -- the total energy against the committed reference where one
   exists, and against the paired CPU run through the JSON.
3. ``compile`` -- reported as its own line, never amortised: the cold run
   compiles, the warm ones do not, and the difference is the number.
4. ``memory`` -- from JAX's own allocator accounting (``memory_stats``), not
   from ``nvidia-smi``, which reports the preallocated pool and would give a
   *wrong number rather than a failed check*.
5. ``determinism`` -- the same SCF run twice in one process and compared bit for
   bit. ``basis/fft.py`` scatters plane waves into the box with an accumulating
   scatter over deliberately duplicated indices, and XLA may lower that through
   atomics whose summation order is not reproducible. Whether it does is
   unverified; this is the test.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = REPO_ROOT / "benchmarks"
QE_CASES = REPO_ROOT / "tests" / "data" / "qe"
PSEUDO_DIR = REPO_ROOT / "tests" / "data" / "pseudo"


# ------------------------------------------------------------------ the dials

def _dial(setting: str) -> str:
    """Normalise a dial argument into what ``batching.py`` parses."""
    text = str(setting).strip().lower()
    if text in ("all", "none", "0", "off"):
        return "all"
    value = int(text)
    if value < 1:
        raise ValueError(f"a batch size is a positive integer or 'all', got {setting!r}")
    return str(value)


def _apply_dials(k_batch: str, band_batch: str) -> dict:
    """Into the environment, *before* pypresso is imported."""
    os.environ["PYPRESSO_K_BATCH"] = k_batch
    os.environ["PYPRESSO_BAND_BATCH"] = band_batch
    return {"k_batch": k_batch, "band_batch": band_batch}


# ------------------------------------------------------------- the case files

def resolve_case(name: str) -> Path:
    """A path, or a name looked up in ``benchmarks/`` then ``tests/data/qe/``."""
    direct = Path(name)
    if direct.is_file():
        return direct
    stem = direct.stem if direct.suffix == ".in" else name
    for directory in (BENCHMARKS, QE_CASES):
        candidate = directory / f"{stem}.in"
        if candidate.is_file():
            return candidate
    raise SystemExit(f"no such case: {name!r} (looked in {BENCHMARKS} and {QE_CASES})")


def reference_energy(case: Path) -> float | None:
    """The committed QE total energy for this case, if there is one."""
    path = QE_CASES / f"reference.out.{case.stem}"
    if not path.is_file():
        return None
    from pypresso.io import read_qe_output
    return read_qe_output(path).total_energy


def conv_thr(case: Path, default: float = 1.0e-6) -> float:
    """The input's own ``conv_thr``, so both sides stop in the same place."""
    import re
    match = re.search(r"conv_thr\s*=\s*([\d.eEdD+-]+)", case.read_text())
    return float(match.group(1).lower().replace("d", "e")) if match else default


# ---------------------------------------------------------------- provenance

def _run(command: list[str]) -> str | None:
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def provenance() -> dict:
    """Everything §4a says must sit beside a GPU number so it stays true."""
    import jax
    import jaxlib
    import numpy

    devices = jax.devices()
    head = devices[0]
    record = {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": getattr(jaxlib, "__version__", None),
        "numpy": numpy.__version__,
        "platform": head.platform,
        "device_kind": head.device_kind,
        "device_count": len(devices),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "git_commit": _run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]),
        "git_dirty": bool(_run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"])),
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "pypresso_cache_dir": os.environ.get("PYPRESSO_CACHE_DIR"),
    }
    try:
        import jax_cuda12_plugin  # noqa: F401
        record["cuda_plugin"] = getattr(jax_cuda12_plugin, "__version__", "present")
    except ImportError:
        record["cuda_plugin"] = None
    driver = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                   "--format=csv,noheader"])
    record["nvidia_smi"] = driver
    return record


# --------------------------------------------------------- check 1: precision

def check_precision(grid: tuple[int, int, int] | None) -> dict:
    """x64 asserted on a *device* array, then the fp64/fp32 rate measured.

    ``jax_enable_x64`` is set in ``pypresso/__init__.py`` before anything else
    touches JAX, which is structurally right -- but a platform change is exactly
    when a configuration assumption stops holding, and this one fails by running
    the whole calculation in single precision and returning a plausible number.
    So it is asserted here on an array that has been to the device, not inferred
    from the call having been made.

    The rate is measured on a matmul and on a batched 3D FFT because those are
    what the Davidson subspace algebra and ``h_psi`` are made of; a generic
    FLOPs probe would measure a kernel this code never runs.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    probe = jnp.zeros(4, dtype=jnp.float64) + 1.0
    complex_probe = probe.astype(jnp.complex128)
    try:
        x64_flag = bool(jax.config.read("jax_enable_x64"))
    except (AttributeError, KeyError, ValueError):   # older jax spells it as an attribute
        x64_flag = bool(getattr(jax.config, "jax_enable_x64", False))
    record = {
        "x64_enabled": x64_flag,
        "float64_survives": str(probe.dtype) == "float64",
        "complex128_survives": str(complex_probe.dtype) == "complex128",
        "device_of_probe": str(list(probe.devices())[0]),
        "default_matmul_precision": os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
    }
    if not (record["float64_survives"] and record["complex128_survives"]):
        record["verdict"] = "FAIL: x64 did not survive to the device"
        return record

    def timed(fn, *arrays, repeats=5):
        jax.block_until_ready(fn(*arrays))          # compile
        start = time.perf_counter()
        for _ in range(repeats):
            out = fn(*arrays)
        jax.block_until_ready(out)
        return (time.perf_counter() - start) / repeats

    rng = np.random.default_rng(0)
    rates = {}

    n = 1024
    for label, dtype in (("complex128", np.complex128), ("complex64", np.complex64)):
        a = jnp.asarray(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)), dtype)
        rates[f"matmul_{n}_{label}_s"] = timed(jax.jit(lambda x: x @ x), a)

    if grid is not None:
        n1, n2, n3 = grid
        for label, dtype in (("complex128", np.complex128), ("complex64", np.complex64)):
            box = jnp.asarray(rng.standard_normal((32, n1, n2, n3)), dtype)
            rates[f"fft3d_{n1}x{n2}x{n3}_x32_{label}_s"] = timed(
                jax.jit(lambda x: jnp.fft.fftn(x, axes=(1, 2, 3))), box)

    record["timings"] = rates
    for kernel in ("matmul", "fft3d"):
        slow = next((v for k, v in rates.items() if k.startswith(kernel) and "128" in k), None)
        fast = next((v for k, v in rates.items() if k.startswith(kernel) and "64_s" in k
                     and "128" not in k), None)
        if slow and fast:
            record[f"{kernel}_fp64_slowdown"] = slow / fast
    record["verdict"] = "ok"
    return record


# ------------------------------------------------------------ check 4: memory

def memory_stats() -> dict:
    """JAX's own live-bytes accounting, and the reason it is not ``nvidia-smi``.

    XLA preallocates the bulk of the device on first use, so the pool's size is
    not the working set and reading it would report a number that is wrong
    rather than a check that failed. ``memory_stats`` is the allocator's view of
    what is actually live. On CPU it returns ``None``, and the host peak RSS is
    reported instead and labelled as such.
    """
    import jax
    device = jax.devices()[0]
    stats = None
    try:
        stats = device.memory_stats()
    except (AttributeError, NotImplementedError):
        stats = None
    if stats:
        return {
            "source": "jax.Device.memory_stats",
            "bytes_in_use": stats.get("bytes_in_use"),
            "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
            "bytes_limit": stats.get("bytes_limit"),
            "peak_gb": (stats.get("peak_bytes_in_use") or 0) / 2**30,
        }
    import resource
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform == "darwin" else 1024   # macOS reports bytes, Linux KiB
    return {"source": "host RSS (no device allocator)", "peak_gb": maxrss * scale / 2**30}


# ------------------------------------------------------------------- the run

def run_case(case: Path, repeats: int, threshold: float) -> dict:
    """Set up once, then run the SCF ``repeats`` times in the same process.

    The first call compiles and the rest do not, which is what makes check 3 a
    subtraction rather than an estimate, and running the *identical* SCF more
    than once in one process is what makes check 5 possible at all.
    """
    import jax
    from pypresso.io.pwin import read_pw_input
    from pypresso.pseudo import read_upf
    from pypresso.scf.driver import Calculation, run_scf
    from pypresso.system import build_system

    system = build_system(read_pw_input(case))
    pseudos = tuple(read_upf(PSEUDO_DIR / s.pseudo_file) for s in system.structure.species)

    start = time.perf_counter()
    calculation = Calculation(system, pseudos)
    jax.block_until_ready(calculation.vltot)
    setup = time.perf_counter() - start
    smooth_grid, dense_grid = _grid_of(calculation)

    runs = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        result = run_scf(system, pseudos, calculation=calculation, conv_thr=threshold)
        jax.block_until_ready(result.density)
        runs.append({
            "wall_s": time.perf_counter() - start,
            "total_energy": float(result.total_energy),
            "iterations": int(result.iterations),
            "converged": bool(result.converged),
            "accuracy": float(result.accuracy) if result.accuracy is not None else None,
            "eigenvalue_checksum": _bits(result.eigenvalues),
            "density_checksum": _bits(result.density),
        })
        last = result

    cold = runs[0]["wall_s"]
    warm = min(run["wall_s"] for run in runs[1:]) if len(runs) > 1 else None
    return {
        "case": case.name,
        "nat": int(system.structure.nat),
        "nk": int(system.kpoints.nk),
        "ecutwfc": float(system.ecutwfc),
        "nspin": int(system.nspin),
        "conv_thr": threshold,
        "fft_grid": smooth_grid,
        "dense_grid": dense_grid,
        "npwx": int(calculation.basis.npwx),
        "setup_s": setup,
        "scf_cold_s": cold,
        "scf_warm_s": warm,
        "compile_s": (cold - warm) if warm is not None else None,
        "per_iteration_s": (warm / runs[0]["iterations"]) if warm else None,
        "runs": runs,
        "total_energy": runs[0]["total_energy"],
        "converged": runs[0]["converged"],
        "iterations": runs[0]["iterations"],
        "_result": last,
    }


def _bits(array) -> str:
    """A bit-exact fingerprint of an array: SHA-256 of its raw bytes.

    **Not** Python's ``hash``, which is salted per process: a fingerprint that
    changes with ``PYTHONHASHSEED`` compares fine inside one run and is
    meaningless the moment it is written to JSON. GPU.md check 5 is literally
    "run the identical *job* twice and diff bit for bit", which is two
    submissions -- and two submissions is what catches the failure one process
    cannot, a compilation that is not itself reproducible.
    """
    import hashlib

    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _grid_of(calculation) -> tuple[tuple | None, tuple | None]:
    """``(smooth, dense)`` FFT dimensions -- the wavefunctions' box and the
    augmentation charge's, which are the same grid without ``ecutrho``."""
    basis = getattr(calculation, "basis", None)
    if basis is None:
        return None, None
    smooth = tuple(int(n) for n in basis.smooth.grid)
    dense = tuple(int(n) for n in basis.dense.grid)
    return smooth, dense


def check_determinism(runs: list[dict]) -> dict:
    """Two identical runs, compared bit for bit -- not "close".

    A single run compared once cannot see the way a GPU breaks reproducibility,
    and the plan's hardest rule is that no phase may change a validated number.
    If these differ, the determinism policy is settled *before* any GPU number
    is called validated.
    """
    if len(runs) < 2:
        return {"verdict": "not run: needs --repeats 2 or more"}
    energies = [run["total_energy"] for run in runs]
    checksums = [run["eigenvalue_checksum"] for run in runs]
    identical = len(set(map(repr, energies))) == 1 and len(set(checksums)) == 1
    spread = max(energies) - min(energies)
    return {
        "verdict": "bit-identical" if identical else "DIFFERS run to run",
        "energy_spread_ry": spread,
        "energies": energies,
        "eigenvalues_identical": len(set(checksums)) == 1,
    }


# ------------------------------------------------------------------------ cli

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", help="a name in benchmarks/ or tests/data/qe/, or a path")
    parser.add_argument("--k-batch", required=True,
                        help="k-points in flight: an integer, or 'all' for one vmap")
    parser.add_argument("--band-batch", required=True,
                        help="bands transformed at once: an integer, or 'all'")
    parser.add_argument("--repeats", type=int, default=2,
                        help="SCF runs in one process; 2 is what check 5 needs")
    parser.add_argument("--conv-thr", type=float, default=None,
                        help="override the input's own conv_thr")
    parser.add_argument("--json", type=Path, default=None, help="where to write the record")
    parser.add_argument("--skip-precision-probe", action="store_true",
                        help="skip check 1's fp64/fp32 timing (the assertion still runs)")
    arguments = parser.parse_args(argv)

    dials = _apply_dials(_dial(arguments.k_batch), _dial(arguments.band_batch))
    case = resolve_case(arguments.case)

    import pypresso  # noqa: F401  -- sets jax_enable_x64 before any array exists

    record = {"dials": dials, "provenance": provenance()}
    print(f"# pypresso GPU.md phase 0 -- {case.name}")
    print(f"# {record['provenance']['device_kind']} x{record['provenance']['device_count']}"
          f" ({record['provenance']['platform']}), jax {record['provenance']['jax']}")
    print(f"# dials: k_batch={dials['k_batch']} band_batch={dials['band_batch']}")

    threshold = arguments.conv_thr if arguments.conv_thr is not None else conv_thr(case)
    timing = run_case(case, arguments.repeats, threshold)
    result = timing.pop("_result")
    record["run"] = timing

    record["precision"] = check_precision(
        None if arguments.skip_precision_probe else timing["fft_grid"])
    if record["precision"].get("verdict", "").startswith("FAIL"):
        print("FAIL: x64 did not reach the device -- every number below is single precision")

    record["memory"] = memory_stats()
    record["determinism"] = check_determinism(timing["runs"])

    expected = reference_energy(case)
    record["reference"] = {
        "qe_total_energy": expected,
        "difference_ry": (timing["total_energy"] - expected) if expected is not None else None,
    }

    _report(record)
    if arguments.json:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(record, indent=2, default=str))
        print(f"\nwritten: {arguments.json}")
    del result
    return 0 if timing["converged"] else 1


def _report(record: dict) -> None:
    run, precision = record["run"], record["precision"]
    print(f"\n{run['case']}: {run['nat']} atoms, {run['nk']} k-points, "
          f"ecutwfc {run['ecutwfc']} Ry, grid {run['fft_grid']}")
    print(f"  converged            {run['converged']} in {run['iterations']} iterations")
    print(f"  total energy         {run['total_energy']:.10f} Ry")
    reference = record["reference"]
    if reference["qe_total_energy"] is not None:
        print(f"  against QE           {reference['qe_total_energy']:.10f} Ry"
              f"  ({reference['difference_ry']:+.2e})")
    print(f"  setup                {run['setup_s']:.2f} s")
    print(f"  scf cold / warm      {run['scf_cold_s']:.2f} / "
          f"{run['scf_warm_s'] if run['scf_warm_s'] else float('nan'):.2f} s")
    if run["compile_s"] is not None:
        print(f"  compile              {run['compile_s']:.2f} s  (reported, not amortised)")
    if run["per_iteration_s"]:
        print(f"  per iteration        {run['per_iteration_s'] * 1e3:.1f} ms")
    print(f"  x64 on device        {precision['float64_survives']} / "
          f"{precision['complex128_survives']} (complex)")
    for kernel in ("matmul", "fft3d"):
        slowdown = precision.get(f"{kernel}_fp64_slowdown")
        if slowdown:
            print(f"  fp64/fp32 {kernel:<10} {slowdown:.2f}x slower in double")
    memory = record["memory"]
    print(f"  peak memory          {memory['peak_gb']:.2f} GB  [{memory['source']}]")
    print(f"  determinism          {record['determinism']['verdict']}")


if __name__ == "__main__":
    raise SystemExit(main())
