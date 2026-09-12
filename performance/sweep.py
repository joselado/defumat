#!/usr/bin/env python3
"""The systematic benchmark: Quantum ESPRESSO, defumat on a CPU core, defumat on a GPU.

One harness, three legs, two sets, and the same code path whether it runs on a
laptop or as one task of a Slurm array. Nothing here measures anything itself --
the QE leg is ``tools/compare_qe.py``'s ``run_qe`` and both defumat legs are
``tools/gpu/phase0.py``'s ``run_case``, which is device-agnostic on purpose.
This file is the loop, the schema and the report.

    python3 performance/sweep.py --set fast                 # the laptop run
    python3 performance/sweep.py --set complete --gpu off   # no accelerator here
    python3 performance/sweep.py --measure si8-1k --leg cpu --json out.json
    python3 performance/sweep.py --report results/complete

**The three legs are two different claims and the report never mixes them.**

* ``qe`` and ``cpu`` are the project's own metric: single-core defumat against
  single-core Quantum ESPRESSO, same machine, same input, same ``conv_thr``.
  Both are pinned to one CPU by affinity mask, because XLA sizes its thread
  pool from that mask and ignores ``OMP_NUM_THREADS`` -- without the pinning the
  comparison flatters this code by the core count.
* ``gpu`` is a different metric and ``GPU.md`` §2.3 fixes it: defumat on one GPU
  against *defumat on CPU*, same code, same input, **per SCF iteration**, with
  compile time reported as its own column rather than amortised away. A GPU
  number put against single-core QE would be meaningless in the same way the
  unpinned CPU comparison is, only more so. Table B is therefore against the
  ``cpu`` leg and says so.

**Cold, warm, and the compile cache.** ``setup`` and the first SCF include XLA
compiling every kernel; the warm SCF re-runs the loop with them compiled, and is
the honest measure of the arithmetic. That split is only true with the
persistent kernel cache **off**, which is this script's default (``--cache``) --
with it on, a "cold" run is reading someone else's compilation, and on a cluster
that someone else is a concurrent array task. QE has no equivalent split; there
is nothing to compile.

**One process per (case, leg).** It bounds each unit two ways: a hard timeout,
and a fresh process, so one case's compilation cache and memory high-water mark
cannot colour the next. It is also what makes the Slurm array possible at all --
an array task is exactly one of these units, run by the same ``--measure`` mode
the local loop drives.

``pw.x`` is expected at ``quantum_espresso/.../bin/pw.x``; override with
``$PW_X``, which is what the cluster does (there the module's 7.2 is what is
available, and the version lands in the record beside the number).
"""

from __future__ import annotations

import os
import sys

LEGS = ("qe", "cpu", "gpu")


def _flag(argv: list[str], name: str) -> str | None:
    """One flag's value, read from argv before argparse exists.

    The affinity mask and the single-core environment have to be in place
    *before* JAX is imported, which is before there is a parsed namespace to
    consult -- the same reason ``tools/compare_qe.py`` re-executes itself.
    """
    for index, token in enumerate(argv):
        if token == name and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith(f"{name}="):
            return token.split("=", 1)[1]
    return None


_MEASURING = any(token == "--measure" or token.startswith("--measure=")
                for token in sys.argv)
_LEG = _flag(sys.argv, "--leg") or "cpu"

#: One core, and the affinity mask below is the only mechanism that delivers it
#: for XLA. The variables still matter for QE, for BLAS, and for anything the
#: subprocess inherits.
_SINGLE_CORE = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "DEFUMAT_THREADS": "1",
}

if _MEASURING and os.environ.get("DEFUMAT_BENCH_PINNED") != "1":
    os.environ.update(_SINGLE_CORE)
    os.environ["DEFUMAT_BENCH_PINNED"] = "1"
    # tools.compare_qe re-executes on import unless this is already set; it has
    # just been done here, and doing it twice would lose this process's argv.
    os.environ["DEFUMAT_PINNED"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])

# The GPU leg is *not* pinned to one core. Its host side dispatches rather than
# computes, and starving the dispatch thread would measure the host instead of
# the device; the thread counts above still hold it to one BLAS thread. Which
# was done lands in the record as ``pinned``.
_PINNED_TO = None
if _MEASURING and _LEG != "gpu" and hasattr(os, "sched_setaffinity"):
    _PINNED_TO = sorted(os.sched_getaffinity(0))[0]
    os.sched_setaffinity(0, {_PINNED_TO})

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from performance import sets  # noqa: E402


# --------------------------------------------------------------- one unit of work


def measure(case_name: str, leg: str, repeats: int, cache: str,
            k_batch: str, band_batch: str, max_iterations: int) -> dict:
    """Run one case through one leg and return the record. A whole process's work.

    ``repeats`` is how many times the SCF runs *in this process*: the first
    compiles and the rest do not, so the warm time is a measurement rather than
    an estimate. QE ignores it beyond taking the fastest of that many runs.
    """
    case = sets.resolve(case_name)
    record = {
        "case": case_name,
        "leg": leg,
        "input": str(case.relative_to(ROOT)) if case.is_relative_to(ROOT) else str(case),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pinned_to_cpu": _PINNED_TO,
        "repeats": repeats,
    }
    started = time.perf_counter()
    if leg == "qe":
        record.update(_measure_qe(case, repeats))
    else:
        record.update(_measure_defumat(case, repeats, cache, k_batch, band_batch,
                                       max_iterations))
    record["wall_s"] = time.perf_counter() - started
    record["peak_rss_gb"] = _peak_rss_gb()
    record["ok"] = True
    return record


def _measure_qe(case: Path, repeats: int) -> dict:
    """``pw.x`` on one core, timed by its own report rather than by a stopwatch.

    QE prints the wall time of every routine it instruments, so ``init_run`` and
    ``electrons`` are QE's numbers. The process wall time is kept beside them
    because that is what is comparable to a stopwatch around defumat.
    """
    from tools import compare_qe

    pw_x = Path(os.environ.get("PW_X", compare_qe.DEFAULT_PW_X))
    threshold = compare_qe._conv_thr(case)
    qe = compare_qe.run_qe(case, pw_x, repeats)
    iterations = max(qe["iterations"], 1)
    return {
        "code": "pw.x",
        "conv_thr": threshold,
        "setup_s": qe["init"],
        "scf_s": qe["scf"],
        "scf_cold_s": None,
        "scf_warm_s": None,
        "compile_s": None,
        "per_iteration_s": qe["scf"] / iterations,
        "process_wall_s": qe["total"],
        "iterations": qe["iterations"],
        "total_energy": qe["energy"],
        "converged": True,
        "pw_x": str(pw_x),
        "env": _environment(defumat=False),
    }


def _measure_defumat(case: Path, repeats: int, cache: str, k_batch: str,
                     band_batch: str, max_iterations: int) -> dict:
    """defumat on whatever device JAX finds, with both batching dials explicit.

    ``defumat/batching.py`` reads its dials at import time, so they go into the
    environment here -- before ``run_case`` performs the import. The CPU leg
    runs at QE's end of both dials (one k-point, one band batch, which is what
    ``c_bands.f90`` does) and the GPU leg at the batched end, because that is
    what each machine wants and comparing them at a single setting would measure
    the dial rather than the device.
    """
    os.environ["DEFUMAT_CACHE_DIR"] = cache
    os.environ["DEFUMAT_K_BATCH"] = k_batch
    os.environ["DEFUMAT_BAND_BATCH"] = band_batch

    sys.path.insert(0, str(ROOT / "tools" / "gpu"))
    from tools.gpu import phase0

    threshold = phase0.conv_thr(case)
    run = phase0.run_case(case, repeats, threshold, max_iterations=max_iterations)
    run.pop("_result", None)
    memory = phase0.memory_stats()

    warm = run["scf_warm_s"]
    return {
        "code": "defumat",
        "conv_thr": threshold,
        "nat": run["nat"],
        "nk": run["nk"],
        "ecutwfc": run["ecutwfc"],
        "nspin": run["nspin"],
        "npwx": run["npwx"],
        "fft_grid": run["fft_grid"],
        "dense_grid": run["dense_grid"],
        "setup_s": run["setup_s"],
        # The number the tables use: the warm SCF where there is one, and the
        # cold one where a single repeat was asked for. Which it is, is not left
        # to be inferred -- `scf_is_warm` says so.
        "scf_s": warm if warm is not None else run["scf_cold_s"],
        "scf_is_warm": warm is not None,
        "scf_cold_s": run["scf_cold_s"],
        "scf_warm_s": warm,
        "compile_s": run["compile_s"],
        "per_iteration_s": run["per_iteration_s"],
        "iterations": run["iterations"],
        "total_energy": run["total_energy"],
        "converged": run["converged"],
        "k_batch": k_batch,
        "band_batch": band_batch,
        "cache": cache,
        "device_memory": memory,
        "env": _environment(defumat=True),
    }


def _environment(defumat: bool) -> dict:
    """What a reader needs in order to know what the number describes."""
    if defumat:
        from tools.gpu import phase0
        return phase0.provenance()
    from performance import run_performance
    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "qe": run_performance.qe_version(),
        "git_commit": run_performance.git_commit(),
        "cpu": _cpu_model(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
    }


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:  # pragma: no cover - not Linux
        pass
    return platform.processor()


def _peak_rss_gb() -> float:
    """This process's high-water mark (``VmHWM``), in GB."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM"):
                return round(int(line.split()[1]) / 1048576.0, 3)
    except OSError:  # pragma: no cover - not Linux
        pass
    return float("nan")


# ------------------------------------------------------------------- the local loop


def has_gpu() -> bool:
    """Is there an accelerator? Asked in a subprocess, so the driver never imports JAX.

    The driver spawns one pinned process per unit and must not have a JAX
    runtime of its own in the way -- importing it here would put a thread pool
    and a device allocator inside the process that is supposed to be doing
    nothing but waiting.
    """
    probe = subprocess.run(
        [sys.executable, "-c", "import jax; print(jax.default_backend())"],
        capture_output=True, text=True, timeout=300, cwd=ROOT)
    return probe.returncode == 0 and probe.stdout.strip() not in ("", "cpu")


def unit_path(out_dir: Path, case: str, leg: str) -> Path:
    return out_dir / f"{case}.{leg}.json"


def run_unit(case_spec, leg: str, out_dir: Path, arguments) -> str:
    """One (case, leg) in its own process, with a hard timeout. Returns a verdict."""
    case = case_spec.name
    path = unit_path(out_dir, case, leg)
    if arguments.resume and path.is_file():
        try:
            if json.loads(path.read_text()).get("ok"):
                return "cached"
        except (OSError, ValueError):
            pass  # a truncated record is not a result; measure it again

    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--measure", case, "--leg", leg, "--json", str(path),
        "--repeats", str(arguments.repeats), "--cache", arguments.cache,
        # A cell with a known-higher cap carries it; see Case.max_iterations,
        # where reporting a converged run as non-converged is the failure being
        # avoided rather than a lost iteration.
        "--max-iterations", str(case_spec.max_iterations or arguments.max_iterations),
    ]
    if leg != "qe":
        k_batch, band_batch = _dials(leg, arguments)
        command += ["--k-batch", k_batch, "--band-batch", band_batch]

    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=arguments.max_seconds, cwd=ROOT)
    except subprocess.TimeoutExpired:
        _write_failure(path, case, leg, f"over budget ({arguments.max_seconds:.0f}s)")
        return "over budget"
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        _write_failure(path, case, leg, tail[-1] if tail else "failed with no output")
        if arguments.verbose:
            print((completed.stderr or completed.stdout)[-2000:], file=sys.stderr)
        return "failed"
    return "ok"


def _dials(leg: str, arguments) -> tuple[str, str]:
    """The batching dials for a leg, defaulting to what that machine wants.

    CPU defaults to QE's end -- one k-point and one band block at a time, which
    is ``c_bands.f90``'s loop and what a cache wants. GPU defaults to the
    batched end, where the k axis is the parallelism there is.
    """
    if leg == "gpu":
        return (arguments.k_batch or "all", arguments.band_batch or "all")
    return (arguments.k_batch or "1", arguments.band_batch or "1")


def available_ram_gb() -> float | None:
    """This machine's total RAM, for the guard below. ``None`` if it cannot be read."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return int(line.split()[1]) / 1048576.0
    except (OSError, ValueError):  # pragma: no cover - not Linux
        pass
    return None


def too_large_for(case, ram_gb: float | None, headroom: float = 0.8) -> str | None:
    """Why this case should not be started here, or ``None`` if it is fine.

    An out-of-memory kill on this project's workstation does not cost a case --
    it costs whatever else the terminal was holding, which has happened at least
    three times (``CLAUDE.md``, "An out-of-memory kill must cost one file"). The
    per-unit subprocess means the *driver* survives the child being killed, but
    the kernel chooses the victim and need not choose the child. So a case whose
    recorded peak does not fit is refused by name before it starts, rather than
    discovered.

    ``headroom`` is what the recorded figure is worth: it is a device peak
    standing in for a host peak, so a case that fits only marginally is treated
    as one that does not.
    """
    if ram_gb is None or case.peak_gb is None:
        return None
    if case.peak_gb > headroom * ram_gb:
        return (f"needs about {case.peak_gb:.0f} GB and this machine has "
                f"{ram_gb:.0f} GB -- skipped rather than risking an OOM kill "
                f"(--force-large to override, or run it on the cluster)")
    return None


def _write_failure(path: Path, case: str, leg: str, why: str) -> None:
    """A failure is a record too: an absent file cannot be told from an unrun one."""
    path.write_text(json.dumps(
        {"case": case, "leg": leg, "ok": False, "error": why,
         "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))


def drive(arguments) -> int:
    """The laptop launcher: every case, every enabled leg, one process each."""
    known = {case.name: case for case in sets.COMPLETE}
    chosen = ([known.get(name) or sets.Case(name, "") for name in arguments.cases]
              if arguments.cases else sets.cases(arguments.set))
    out_dir = arguments.out_dir or (HERE / "results" / arguments.set)
    out_dir.mkdir(parents=True, exist_ok=True)

    legs = list(arguments.legs) if arguments.legs else ["qe", "cpu", "gpu"]
    if arguments.gpu == "off":
        legs = [leg for leg in legs if leg != "gpu"]
    elif arguments.gpu == "auto" and "gpu" in legs and not has_gpu():
        print("no accelerator visible to JAX -- the GPU leg is skipped "
              "(--gpu on to force it, --gpu off to silence this)")
        legs = [leg for leg in legs if leg != "gpu"]

    print(f"set {arguments.set}: {len(chosen)} cases x {len(legs)} legs "
          f"({', '.join(legs)}) -> {out_dir}")
    print(f"  budget {arguments.max_seconds:.0f}s per unit, kernel cache {arguments.cache}\n")

    ram_gb = available_ram_gb()
    for case in chosen:
        refusal = None if arguments.force_large else too_large_for(case, ram_gb)
        if refusal:
            print(f"  {case.name:22s} --  skipped: {refusal}", flush=True)
            for leg in legs:
                _write_failure(unit_path(out_dir, case.name, leg), case.name, leg, refusal)
            continue
        for leg in legs:
            print(f"  {case.name:22s} {leg:3s} ", end="", flush=True)
            started = time.perf_counter()
            verdict = run_unit(case, leg, out_dir, arguments)
            print(f"{verdict:12s} {time.perf_counter() - started:6.1f}s", flush=True)

    print()
    return report(out_dir, arguments)


# ----------------------------------------------------------------------- the report


def load(out_dir: Path) -> dict[str, dict[str, dict]]:
    """Every record under ``out_dir``, as ``{case: {leg: record}}``."""
    by_case: dict[str, dict[str, dict]] = {}
    for path in sorted(out_dir.glob("*.json")):
        if path.name == "merged.json":
            continue
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        by_case.setdefault(record["case"], {})[record["leg"]] = record
    return by_case


def _ratio(ours: float | None, theirs: float | None) -> str:
    if not ours or not theirs or ours != ours or theirs != theirs:
        return "     -"
    return f"{ours / theirs:6.1f}"


def _seconds(value: float | None) -> str:
    return "       -" if value is None or value != value else f"{value:8.3f}"


def _ordered(by_case: dict, set_name: str) -> list[str]:
    """Report order is the set's order, with anything extra after it."""
    order = [case.name for case in sets.SETS.get(set_name, [])]
    known = [name for name in order if name in by_case]
    return known + sorted(name for name in by_case if name not in order)


def report(out_dir: Path, arguments) -> int:
    by_case = load(out_dir)
    if not by_case:
        print(f"no records in {out_dir}")
        return 1
    names = _ordered(by_case, arguments.set)

    lines = []
    lines += _header(by_case)
    lines += _table_a(by_case, names)
    lines += _table_b(by_case, names)
    lines += _failures(by_case, names)

    text = "\n".join(lines)
    print(text)
    (out_dir / "report.txt").write_text(text + "\n")
    merged = {"set": arguments.set, "generated": time.strftime("%Y-%m-%d %H:%M"),
              "cases": by_case}
    (out_dir / "merged.json").write_text(json.dumps(merged, indent=2, default=str))
    print(f"\nwritten: {out_dir/'report.txt'}, {out_dir/'merged.json'}")
    return 0


def _header(by_case: dict) -> list[str]:
    """What ran, where, and on what -- the numbers mean nothing without it."""
    any_qe = next((r for c in by_case.values() for r in [c.get("qe")] if r and r.get("ok")), None)
    any_cpu = next((r for c in by_case.values() for r in [c.get("cpu")] if r and r.get("ok")), None)
    any_gpu = next((r for c in by_case.values() for r in [c.get("gpu")] if r and r.get("ok")), None)

    lines = ["", f"benchmark sweep, {time.strftime('%Y-%m-%d %H:%M')}", ""]
    if any_qe:
        env = any_qe["env"]
        lines.append(f"  QE            {env.get('qe', '?')} on 1 core, {env.get('cpu', '?')}")
    if any_cpu:
        env = any_cpu["env"]
        lines.append(f"  defumat CPU   {env.get('device_kind', '?')} x1 core, "
                     f"jax {env.get('jax', '?')}, commit {str(env.get('git_commit'))[:8]}"
                     f"{' (dirty)' if env.get('git_dirty') else ''}")
    if any_gpu:
        env = any_gpu["env"]
        lines.append(f"  defumat GPU   {env.get('device_kind', '?')} x{env.get('device_count', '?')}, "
                     f"{env.get('nvidia_smi') or 'no nvidia-smi'}")
    lines.append("")
    return lines


def _table_a(by_case: dict, names: list[str]) -> list[str]:
    """The project's metric: both codes on one core, warm SCF and per iteration."""
    lines = [
        "  A. defumat against Quantum ESPRESSO, both pinned to one CPU core.",
        "     SCF is QE's own `electrons` against defumat's warm loop; the per-iteration",
        "     columns divide out the two codes taking different numbers of iterations.",
        "",
        f"     {'case':22s} {'nat':>4s} {'nk':>3s} {'npwx':>6s} | {'QE scf':>8s} "
        f"{'defumat':>8s} {'x':>6s} | {'QE/it':>8s} {'def/it':>8s} {'x':>6s} | "
        f"{'its':>7s} | {'dE (Ry)':>9s}",
        f"     {'-' * 115}",
    ]
    ratios = []
    for name in names:
        qe, cpu = by_case[name].get("qe"), by_case[name].get("cpu")
        if not (cpu and cpu.get("ok")):
            continue
        qe = qe if qe and qe.get("ok") else None
        delta = (f"{abs(qe['total_energy'] - cpu['total_energy']):9.1e}"
                 if qe and qe.get("total_energy") is not None else "        -")
        ratio = _ratio(cpu["per_iteration_s"], qe["per_iteration_s"]) if qe else "     -"
        if qe and qe["per_iteration_s"]:
            ratios.append(cpu["per_iteration_s"] / qe["per_iteration_s"])
        lines.append(
            f"     {name:22s} {cpu['nat']:4d} {cpu['nk']:3d} {cpu['npwx']:6d} | "
            f"{_seconds(qe['scf_s'] if qe else None)} {_seconds(cpu['scf_s'])} "
            f"{_ratio(cpu['scf_s'], qe['scf_s'] if qe else None)} | "
            f"{_seconds(qe['per_iteration_s'] if qe else None)} "
            f"{_seconds(cpu['per_iteration_s'])} {ratio} | "
            f"{(qe['iterations'] if qe else 0):3d}/{cpu['iterations']:<3d} | {delta}")
    if ratios:
        ratios.sort()
        middle = ratios[len(ratios) // 2]
        lines += ["", f"     median per-iteration ratio over {len(ratios)} cases: "
                      f"{middle:.1f}x  (range {ratios[0]:.1f}-{ratios[-1]:.1f}x)"]
    return lines + [""]


def _table_b(by_case: dict, names: list[str]) -> list[str]:
    """GPU.md 2.3's metric: one GPU against this code on CPU, per SCF iteration."""
    rows = [n for n in names if (by_case[n].get("gpu") or {}).get("ok")]
    if not rows:
        return []
    lines = [
        "  B. defumat on one GPU against defumat on one CPU core -- the same code and",
        "     the same input, per SCF iteration. Compile time is its own column and is",
        "     never amortised into the speedup (GPU.md 2.3). The CPU leg runs one",
        "     k-point at a time and the GPU leg batches the k axis, which is what each",
        "     machine wants; that dial setting is in the record.",
        "",
        f"     {'case':22s} {'nat':>4s} {'nk':>3s} | {'CPU/it':>8s} {'GPU/it':>8s} "
        f"{'speedup':>8s} | {'compile':>8s} {'GPU peak':>9s} | {'dE (Ry)':>9s}",
        f"     {'-' * 96}",
    ]
    speedups = []
    for name in rows:
        gpu, cpu = by_case[name]["gpu"], by_case[name].get("cpu")
        cpu = cpu if cpu and cpu.get("ok") else None
        speed = _ratio(cpu["per_iteration_s"] if cpu else None, gpu["per_iteration_s"])
        if cpu and cpu["per_iteration_s"] and gpu["per_iteration_s"]:
            speedups.append(cpu["per_iteration_s"] / gpu["per_iteration_s"])
        peak = (gpu.get("device_memory") or {}).get("peak_gb")
        delta = (f"{abs(cpu['total_energy'] - gpu['total_energy']):9.1e}"
                 if cpu and cpu.get("total_energy") is not None else "        -")
        lines.append(
            f"     {name:22s} {gpu['nat']:4d} {gpu['nk']:3d} | "
            f"{_seconds(cpu['per_iteration_s'] if cpu else None)} "
            f"{_seconds(gpu['per_iteration_s'])} {speed} | "
            f"{_seconds(gpu['compile_s'])} "
            f"{(f'{peak:8.2f}G' if peak else '        -')} | {delta}")
    if speedups:
        speedups.sort()
        lines += ["", f"     median GPU speedup over {len(speedups)} cases: "
                      f"{speedups[len(speedups) // 2]:.1f}x  "
                      f"(range {speedups[0]:.1f}-{speedups[-1]:.1f}x)"]
    return lines + [""]


def _failures(by_case: dict, names: list[str]) -> list[str]:
    """Named, not silently absent -- a missing row is otherwise an unrun one."""
    bad = [(name, leg, record.get("error", "?"))
           for name in names for leg, record in sorted(by_case[name].items())
           if not record.get("ok")]
    if not bad:
        return []
    return ["  Not in the tables:", ""] + \
        [f"     {name:22s} {leg:3s}  {why}" for name, leg, why in bad] + [""]


# ------------------------------------------------------------------------- the CLI


#: A per-unit budget that suits the set. `fast` is meant to finish while someone
#: waits; `complete` carries 64-atom cells that single-core QE takes a while over.
DEFAULT_BUDGET = {"fast": 300.0, "complete": 3600.0}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", default="fast", choices=sorted(sets.SETS),
                        help="which benchmark set to run (default: fast)")
    parser.add_argument("--cases", nargs="+", metavar="NAME",
                        help="these cases instead of the set's")
    parser.add_argument("--legs", nargs="+", choices=LEGS, metavar="LEG",
                        help=f"which legs to run (default: all of {', '.join(LEGS)})")
    parser.add_argument("--gpu", default="auto", choices=("auto", "on", "off"),
                        help="auto skips the GPU leg when JAX sees no accelerator")
    parser.add_argument("--out-dir", type=Path,
                        help="where the per-unit JSON goes (default: performance/results/<set>)")
    parser.add_argument("--repeats", type=int, default=2,
                        help="SCF runs per process: the first compiles, the rest are warm")
    parser.add_argument("--max-seconds", type=float,
                        help="per-unit budget; a unit over it is recorded as over budget")
    parser.add_argument("--max-iterations", type=int, default=200,
                        help="SCF iteration cap. 200, not run_scf's 100, because two "
                             "cells in `complete` converge at 104 and 151 and a cap "
                             "below that reports them as not converged")
    parser.add_argument("--force-large", action="store_true",
                        help="run a case whose recorded peak does not fit this machine")
    parser.add_argument("--cache", default="off",
                        help="DEFUMAT_CACHE_DIR for the measured processes. 'off' is the "
                             "default and is what makes the cold/warm split true")
    parser.add_argument("--k-batch", help="override the k dial for the defumat legs")
    parser.add_argument("--band-batch", help="override the band dial")
    parser.add_argument("--resume", action="store_true",
                        help="skip units that already have a successful record")
    parser.add_argument("--report", type=Path, metavar="DIR",
                        help="re-typeset an existing results directory and measure nothing")
    parser.add_argument("--verbose", action="store_true", help="show a failed unit's output")

    # The worker mode. One (case, leg) in this process, which is what an array
    # task runs; the local loop spawns exactly the same command.
    parser.add_argument("--measure", metavar="CASE", help=argparse.SUPPRESS)
    parser.add_argument("--leg", choices=LEGS, default="cpu", help=argparse.SUPPRESS)
    parser.add_argument("--json", type=Path, help=argparse.SUPPRESS)

    arguments = parser.parse_args(argv)
    if not hasattr(arguments, "force_large"):  # pragma: no cover - defensive
        arguments.force_large = False
    # `--report performance/results/complete` names the set in the path, and
    # taking it from there beats defaulting to `fast` and writing that into the
    # merged record of a complete run.
    if arguments.report and "--set" not in (argv or sys.argv):
        if arguments.report.name in sets.SETS:
            arguments.set = arguments.report.name
    if arguments.max_seconds is None:
        arguments.max_seconds = DEFAULT_BUDGET.get(arguments.set, 900.0)

    if arguments.report:
        return report(arguments.report, arguments)

    if arguments.measure:
        k_batch, band_batch = _dials(arguments.leg, arguments)
        record = measure(arguments.measure, arguments.leg, arguments.repeats,
                         arguments.cache, k_batch, band_batch,
                         arguments.max_iterations)
        text = json.dumps(record, indent=2, default=str)
        if arguments.json:
            arguments.json.parent.mkdir(parents=True, exist_ok=True)
            arguments.json.write_text(text)
            print(f"{record['case']} {record['leg']}: "
                  f"scf {record.get('scf_s')} s in {record.get('iterations')} iterations "
                  f"-> {arguments.json}")
        else:
            print(text)
        return 0

    return drive(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
