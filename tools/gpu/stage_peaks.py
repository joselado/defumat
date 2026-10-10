"""Where a run's device peak is set: the size estimate against the allocator, stage by stage.

``OPEN.md`` Part VII item 1 and ``MEMORY-AUDIT.md`` A17: the stage-peak numbers in
the record came from a ``DEFUMAT_STAGE_PEAKS`` patch that was never committed. This
is that instrument, in the repository. It prints :meth:`Calculator.estimate`'s
report, builds the ``Calculation``, inventories the live device arrays by shape,
then runs ``--iterations`` SCF iterations with every stage of the loop wrapped:
each wrapper blocks on its result and reads the device's whole ``memory_stats()``
before and after, so the stage after which the peak rises is the stage that set
it. ``peak_bytes_in_use`` never falls, so a rise is attributed to the stage it
appears after, and anything that ran inside that stage is billed to it.

**The brackets nest** (``plans/open-d-vii1-review.md``). ``diagonalize`` holds the
``psi0[spin]`` slice, the solver, its retry and the ``jnp.stack`` restack as one
stage, each of the order of the gap on the 45-atom slab, so two inner brackets
sit inside it: ``eigensolver`` (the instance's solver, one call per spin
channel) and ``_every_k`` / ``_every_k_donating`` (the compiled unit, one call
per pass, a second one being the robust retry). Every entry carries its
``depth``, its ``parent`` and the open/close sequence numbers ``seq_open`` and
``seq_close``, so whether an inner rise is contained in its outer one is read
off the record rather than assumed: the part of an outer stage's rise that no
inner entry holds is what ran between them (the slice, the restack).

``peak_bytes_in_use`` is monotone, so after the first tight solve it says
nothing new; every entry also carries ``in_use_before`` and ``in_use`` (and the
whole stats dict before and after), and the per-iteration ``in_use`` deltas are
the reading that still moves. The ``diagonalize`` wrapper logs ``ethr`` (min and
max over the per-band array), the Davidson steps and the unsettled bands, and
the SCF runs at ``verbose=True`` so its own ``ethr`` / ``avg # of iterations``
lines reach the log as well. Warnings are recorded with the stage they fired
in; run with ``PYTHONWARNINGS=always`` so a robust retry is never deduplicated.

    python3 tools/gpu/stage_peaks.py INPUT --pseudo-dir DIR --iterations 3 \
        [--k-batch 1] [--band-batch 16] [--david 2] [--memory-mode memory] \
        [--wfc-store device] [--projectors default] --out stages.json

``--memory-mode`` defaults to ``memory`` and is always passed: memory mode
switches code paths whatever the other dials say, so it is pinned rather than
left to the platform default, and the mode the ``Calculation`` resolved to is
recorded beside the one asked for. On a CPU backend the allocator statistics
are absent, every peak reads 0, and the record holds the estimate, the
inventory, the call order and the solver's counters.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import subprocess
import time
import warnings
from collections import defaultdict
from pathlib import Path

GIB = 1024.0 ** 3


def _stats() -> dict:
    """The device's whole ``memory_stats()`` as ints, ``{}`` on a backend without it."""
    import jax

    stats = jax.devices()[0].memory_stats() or {}
    return {key: int(value) for key, value in stats.items()
            if isinstance(value, (int, float))}


def _use_peak(stats: dict):
    return int(stats.get("bytes_in_use", 0)), int(stats.get("peak_bytes_in_use", 0))


def _rss():
    """This process's resident host memory now, in bytes (``VmRSS``)."""
    try:
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _inventory(top: int = 30):
    """The live device arrays grouped by ``(shape, dtype)``, largest first."""
    import jax

    groups = defaultdict(lambda: [0, 0])
    for array in jax.live_arrays():
        try:
            key = f"{tuple(array.shape)} {array.dtype}"
            groups[key][0] += 1
            groups[key][1] += int(array.nbytes)
        except Exception:  # a deleted or donated buffer
            continue
    rows = sorted(groups.items(), key=lambda kv: -kv[1][1])
    total = sum(v[1] for _, v in rows)
    return {"total_bytes": total,
            "rows": [{"shape": k, "count": v[0], "bytes": v[1]} for k, v in rows[:top]]}


def _commit():
    """The commit of the checkout this script runs from, and whether it is dirty."""
    here = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain",
                                "--untracked-files=no"],
                               capture_output=True, text=True, timeout=20).stdout.strip()
        return {"sha": sha or None, "dirty": bool(dirty)}
    except Exception:
        return {"sha": None, "dirty": None}


def _summary(array):
    """``(min, max)`` of an array-like as floats, ``None`` for ``None``."""
    import numpy as np

    if array is None:
        return None
    values = np.asarray(array)
    if values.size == 0:
        return None
    return [float(values.min()), float(values.max())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("--pseudo-dir", required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--k-batch", default=None)
    parser.add_argument("--band-batch", default=None)
    parser.add_argument("--david", type=int, default=None)
    parser.add_argument("--memory-mode", default="memory", choices=("memory", "speed"))
    parser.add_argument("--wfc-store", default=None)
    parser.add_argument("--projectors", default=None)
    parser.add_argument("--conv-thr", type=float, default=1.0e-14)
    # **The production run's starting texture.** The 45-atom slab's runs on
    # record start from a helix that a two-species input cannot express, built
    # by ``helix_starting_density(calculator, seed_input, pseudo_dir)`` in that
    # project's own driver (``NiBr2_defumat/run_scf.py``, not this
    # repository). The ``ethr`` schedule and the Davidson step counts follow
    # the density, so a run from the input's own guess would measure a
    # different trajectory; these two name the function's file and its seed
    # input, and the density it returns is passed as ``starting_density``.
    parser.add_argument("--seed-script", type=Path, default=None)
    parser.add_argument("--helix-seed", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args()

    import jax
    import numpy as np

    import defumat  # noqa: F401  (x64 before any array)
    import defumat.solvers as solvers
    import defumat.solvers.davidson as davidson
    from defumat import Calculator
    from defumat.scf import driver

    options = {"memory_mode": arguments.memory_mode}
    for name in ("k_batch", "band_batch"):
        value = getattr(arguments, name)
        if value is not None:
            options[name] = None if value == "all" else int(value)
    for name in ("wfc_store", "projectors"):
        value = getattr(arguments, name)
        if value is not None:
            options[name] = value
    if arguments.david is not None:
        options["david"] = arguments.david

    record = {
        "input": str(arguments.input), "pseudo_dir": arguments.pseudo_dir,
        "options": options, "iterations_asked": arguments.iterations,
        "conv_thr": arguments.conv_thr,
        "device": str(jax.devices()[0]), "commit": _commit(),
        # Every dial that is not on the command line, so a run whose numbers
        # moved can be checked for an environment it inherited.
        "environment": {key: value for key, value in sorted(os.environ.items())
                        if key.startswith(("DEFUMAT_", "XLA_", "JAX_", "TF_GPU",
                                           "PYTHONWARNINGS", "OMP_", "MKL_",
                                           "SLURM_JOB_ID", "SLURM_JOB_PARTITION",
                                           "SLURMD_NODENAME"))},
        "stages": [], "warnings": [], "solver": [],
    }
    record["before"] = {"stats": _stats()}

    def dump():
        if arguments.out is not None:
            arguments.out.write_text(json.dumps(record, indent=1, default=str))

    calculator = Calculator.from_file(arguments.input, pseudo_dir=arguments.pseudo_dir,
                                      announce=False, **options)
    estimate = calculator.estimate()
    print(estimate.report(), flush=True)
    record["estimate"] = {
        "peak_bytes": int(estimate.peak_bytes),
        "floor_bytes": int(estimate.total_bytes),
        "eigensolver_buffer": int(estimate.eigensolver_buffer),
        "start_buffer": int(estimate.start_buffer),
        "setup_transient": int(estimate.setup_transient),
        "arrays": {k: int(v) for k, v in estimate.arrays.items()},
        "host_arrays": {k: int(v) for k, v in
                        (getattr(estimate, "host_arrays", None) or {}).items()},
        "k_batch": estimate.k_batch, "band_batch": estimate.band_batch,
        "wfc_store": estimate.wfc_store, "projectors": estimate.projectors,
        "davidson_basis": estimate.davidson_basis,
        "nbnd": estimate.nbnd, "npwx": estimate.npwx, "nk": estimate.nk,
        "ngm": estimate.ngm, "smooth_grid": list(estimate.smooth_grid),
    }

    # **The solver bracket is installed before the ``Calculation`` exists**, in
    # the registry it reads at construction (``Calculation.eigensolver`` is an
    # instance attribute, which a wrap of the class misses), so the instance
    # holds the wrapped solver from birth and so does any Calculation rebuilt
    # later in the run. It is checked on the instance below.
    stack = []  # labels of the brackets currently open
    sequence = [0]
    counter = defaultdict(int)
    last = {"stage": "setup", "max_rss": 0}
    # The number of the ``diagonalize`` call, which is not the SCF iteration:
    # a first iteration whose ``ethr`` was too large diagonalises twice, as
    # ``electrons.f90`` does. The SCF's own iteration is in the verbose lines.
    iteration = [0]

    def bracket(original, label, on_result=None):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            parent = stack[-1] if stack else None
            sequence[0] += 1
            seq_open = sequence[0]
            before = _stats()
            stack.append(label)
            try:
                result = original(*args, **kwargs)
                try:
                    jax.block_until_ready(result)
                except Exception:
                    pass
            finally:
                stack.pop()
            after = _stats()
            sequence[0] += 1
            counter[label] += 1
            last["stage"] = f"{label}#{counter[label]}"
            before_use, before_peak = _use_peak(before)
            after_use, after_peak = _use_peak(after)
            rss = _rss()
            entry = {"stage": label, "call": counter[label], "depth": len(stack),
                     "parent": parent, "seq_open": seq_open, "seq_close": sequence[0],
                     "solve": iteration[0],
                     "in_use_before": before_use, "in_use": after_use,
                     "in_use_rise": after_use - before_use,
                     "peak_before": before_peak, "peak": after_peak,
                     "peak_rise": after_peak - before_peak, "host_rss": rss,
                     "stats_before": before, "stats": after}
            if on_result is not None:
                try:
                    entry.update(on_result(args, kwargs, result))
                except Exception as error:  # the instrument must not stop the run
                    entry["on_result_error"] = repr(error)
            record["stages"].append(entry)
            indent = "  " * len(stack)
            if (after_peak > before_peak or abs(after_use - before_use) > GIB // 4
                    or label in ("diagonalize", "eigensolver", "_every_k",
                                 "_every_k_donating", "stream_diagonalize")):
                print(f"[mem] {indent}{label:<22s}#{counter[label]:<3d} in_use "
                      f"{after_use / GIB:8.3f} ({(after_use - before_use) / GIB:+.3f})  "
                      f"peak {after_peak / GIB:8.3f} GiB "
                      f"(+{(after_peak - before_peak) / GIB:.3f})  host "
                      f"{rss / GIB:6.2f} GiB", flush=True)
            dump()
            return result

        # A jitted function carries methods the module calls on its global
        # (``_clear_every_k`` calls ``_every_k.clear_cache()``) and the memory
        # tool lowers it; ``functools.wraps`` copies neither.
        for name in ("clear_cache", "lower", "trace", "eval_shape"):
            if hasattr(original, name) and not hasattr(wrapped, name):
                setattr(wrapped, name, getattr(original, name))
        return wrapped

    def solver_counters(steps, unsettled, ethr, label):
        line = {"label": label, "solve": iteration[0],
                "ethr": _summary(ethr),
                "steps_sum": None if steps is None else int(np.asarray(steps).sum()),
                "steps_max": None if steps is None else int(np.asarray(steps).max()),
                "steps_mean": None if steps is None else float(np.asarray(steps).mean()),
                "unsettled_sum": None if unsettled is None
                else int(np.asarray(unsettled).sum()),
                "unsettled_max": None if unsettled is None
                else int(np.asarray(unsettled).max())}
        record["solver"].append(line)
        ethr_text = ("n/a" if line["ethr"] is None
                     else f"{line['ethr'][0]:.2e}..{line['ethr'][1]:.2e}")
        print(f"[dav] {label} solve {iteration[0]}: ethr {ethr_text}  steps "
              f"mean {line['steps_mean']} max {line['steps_max']}  unsettled "
              f"sum {line['unsettled_sum']} max {line['unsettled_max']}", flush=True)
        return {"ethr": line["ethr"], "steps_max": line["steps_max"],
                "steps_mean": line["steps_mean"],
                "unsettled_sum": line["unsettled_sum"]}

    def after_diagonalize(args, kwargs, result):
        # A class method: ``args[0]`` is the Calculation, then hamiltonians,
        # nbnd, psi0, ethr.
        ethr = kwargs.get("ethr", args[4] if len(args) > 4 else None)
        steps = unsettled = None
        if isinstance(result, tuple) and len(result) == 4:
            steps, unsettled = result[2], result[3]
        return solver_counters(steps, unsettled, ethr, "diagonalize")

    def after_stream(args, kwargs, result):
        ethr = kwargs.get("ethr", args[4] if len(args) > 4 else None)
        return solver_counters(result[1], result[2], ethr, "stream_diagonalize")

    def after_every_k(args, kwargs, result):
        return {"robust": bool(kwargs.get("robust", args[8] if len(args) > 8 else None)),
                "return_finite": bool(kwargs.get("return_finite", False)),
                "chunk": None if kwargs.get("indices") is None
                else int(np.asarray(kwargs["indices"]).shape[0])}

    def before_diagonalize(original):
        @functools.wraps(original)
        def counted(*args, **kwargs):
            iteration[0] += 1
            return original(*args, **kwargs)
        return counted

    real_solvers = dict(solvers.EIGENSOLVERS)
    for name, solver in real_solvers.items():
        solvers.EIGENSOLVERS[name] = bracket(solver, "eigensolver")
    davidson._every_k = bracket(davidson._every_k, "_every_k", after_every_k)
    davidson._every_k_donating = bracket(davidson._every_k_donating,
                                         "_every_k_donating", after_every_k)

    clock = time.time()
    calculation = calculator.calculation
    if getattr(calculation.eigensolver, "__wrapped__", None) is None:
        # Built before the registry was patched (a cached Calculation): wrap
        # the instance itself.
        calculation.eigensolver = bracket(calculation.eigensolver, "eigensolver")
    jax.block_until_ready(jax.live_arrays())
    stats = _stats()
    in_use, peak = _use_peak(stats)
    record["setup"] = {"in_use": in_use, "peak": peak, "stats": stats,
                       "seconds": time.time() - clock, "inventory": _inventory()}
    record["resolved"] = {
        "memory_mode": getattr(calculation, "memory_mode", None),
        "projectors": getattr(calculation, "projector_storage", None),
        "k_batch": getattr(calculation, "k_batch", None),
        "band_batch": getattr(calculation, "band_batch", None),
        "david": getattr(calculation, "david", None),
        "eigensolver": getattr(getattr(calculation.eigensolver, "__wrapped__", None),
                               "__name__", None),
    }
    print(f"[mem] resolved {record['resolved']}", flush=True)
    print(f"[mem] setup  in_use {in_use / GIB:8.3f}  peak {peak / GIB:8.3f} GiB",
          flush=True)
    for row in record["setup"]["inventory"]["rows"][:12]:
        print(f"        {row['bytes'] / GIB:8.3f} GiB  x{row['count']:<3d} {row['shape']}")
    dump()

    # **Host memory between the brackets.** A stage that blows up host RAM can
    # take the process down before its wrapper returns, which leaves no entry;
    # this thread writes the largest ``VmRSS`` seen and the last stage that
    # finished to ``<out>.host`` every half second, so a kill still names where.
    if arguments.out is not None:
        import threading

        host_file = arguments.out.with_suffix(".host")

        def sample():
            while True:
                rss = _rss()
                if rss > last["max_rss"]:
                    last["max_rss"] = rss
                    last["at"] = last["stage"]
                host_file.write_text(json.dumps({
                    "rss": rss, "max_rss": last["max_rss"],
                    "max_after": last.get("at"), "last_stage": last["stage"],
                    "open": list(stack), "time": time.time()}))
                time.sleep(0.5)

        threading.Thread(target=sample, daemon=True).start()

    # A warning is recorded with the brackets open when it fired; the robust
    # retry's only other trace is its warning.
    show = warnings.showwarning

    def showwarning(message, category, filename, lineno, file=None, line=None):
        record["warnings"].append({"category": category.__name__,
                                   "message": str(message)[:400],
                                   "open": list(stack), "after": last["stage"],
                                   "solve": iteration[0]})
        dump()
        return show(message, category, filename, lineno, file, line)

    warnings.showwarning = showwarning

    def wrap(owner, name, label, on_result=None):
        original = getattr(owner, name, None)
        if original is None:
            return
        setattr(owner, name, bracket(original, label, on_result))

    cls = type(calculation)
    for name in ("starting_wavefunctions", "becsum", "density",
                 "density_and_ldos", "potential", "onecenter", "hamiltonian",
                 "symmetrize", "accuracy", "site_moments", "occupations"):
        wrap(cls, name, name)
    wrap(cls, "diagonalize", "diagonalize", after_diagonalize)
    cls.diagonalize = before_diagonalize(cls.diagonalize)
    for name in ("_mix", "_onecenter_torque", "_paw_deband",
                 "_noncollinear_magnetization"):
        wrap(driver, name, name)
    wrap(driver, "stream_diagonalize", "stream_diagonalize", after_stream)
    # ``stream_diagonalize`` advances the iteration counter as ``diagonalize``
    # does, on the streamed store.
    driver.stream_diagonalize = before_diagonalize(driver.stream_diagonalize)

    scf_options = {}
    if arguments.helix_seed is not None:
        if arguments.seed_script is None:
            raise SystemExit("--helix-seed needs --seed-script, the file that "
                             "defines helix_starting_density")
        import importlib.util

        spec = importlib.util.spec_from_file_location("seed_script",
                                                      arguments.seed_script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        before = _stats()
        scf_options["starting_density"] = module.helix_starting_density(
            calculator, str(arguments.helix_seed), arguments.pseudo_dir)
        jax.block_until_ready(scf_options["starting_density"])
        after = _stats()
        record["helix_seed"] = {"script": str(arguments.seed_script),
                                "seed": str(arguments.helix_seed),
                                "stats_before": before, "stats": after}
        print(f"[mem] helix seed  in_use {_use_peak(after)[0] / GIB:8.3f}  peak "
              f"{_use_peak(after)[1] / GIB:8.3f} GiB", flush=True)
        dump()

    clock = time.time()
    result = calculator.get_scf(max_iterations=arguments.iterations,
                                conv_thr=arguments.conv_thr, verbose=True,
                                **scf_options)
    jax.block_until_ready(result.density)
    stats = _stats()
    in_use, peak = _use_peak(stats)
    record["scf"] = {"in_use": in_use, "peak": peak, "stats": stats,
                     "seconds": time.time() - clock,
                     "iterations": int(result.iterations),
                     "accuracy": float(result.accuracy),
                     "inventory": _inventory()}
    record["ratio_measured_over_estimate"] = (
        peak / record["estimate"]["peak_bytes"] if peak else None)
    print(f"[mem] end    in_use {in_use / GIB:8.3f}  peak {peak / GIB:8.3f} GiB  "
          f"estimate {record['estimate']['peak_bytes'] / GIB:8.3f} GiB", flush=True)
    dump()
    print("[mem] STAGE_PEAKS_DONE", flush=True)


if __name__ == "__main__":
    main()
