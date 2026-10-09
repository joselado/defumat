"""Where a run's device peak is set: the size estimate against the allocator, stage by stage.

``OPEN.md`` Part VII item 1 and ``MEMORY-AUDIT.md`` A17: the stage-peak numbers in
the record came from a ``DEFUMAT_STAGE_PEAKS`` patch that was never committed. This
is that instrument, in the repository. It prints :meth:`Calculator.estimate`'s
report, builds the ``Calculation``, inventories the live device arrays by shape,
then runs ``--iterations`` SCF iterations with every stage of the loop wrapped:
each wrapper blocks on its result and reads ``bytes_in_use`` and
``peak_bytes_in_use``, so the stage after which the peak rises is the stage that
set it. ``peak_bytes_in_use`` never falls, so a rise is attributed to the stage it
appears after, and anything that ran inside that stage is billed to it.

    python3 tools/gpu/stage_peaks.py INPUT --pseudo-dir DIR --iterations 3 \
        [--k-batch 1] [--band-batch 16] [--david 2] [--memory-mode memory] \
        [--wfc-store device] [--projectors default] --out stages.json

On a CPU backend the allocator statistics are absent and only the estimate and
the live-array inventory are written.
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from collections import defaultdict
from pathlib import Path

GIB = 1024.0 ** 3


def _stats():
    import jax

    stats = jax.devices()[0].memory_stats() or {}
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--pseudo-dir", required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--k-batch", default=None)
    parser.add_argument("--band-batch", default=None)
    parser.add_argument("--david", type=int, default=None)
    parser.add_argument("--memory-mode", default=None)
    parser.add_argument("--wfc-store", default=None)
    parser.add_argument("--projectors", default=None)
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args()

    import jax
    import numpy as np

    import defumat  # noqa: F401  (x64 before any array)
    from defumat import Calculator
    from defumat.scf import driver

    options = {}
    for name in ("k_batch", "band_batch"):
        value = getattr(arguments, name)
        if value is not None:
            options[name] = None if value == "all" else int(value)
    for name in ("memory_mode", "wfc_store", "projectors"):
        value = getattr(arguments, name)
        if value is not None:
            options[name] = value
    if arguments.david is not None:
        options["david"] = arguments.david

    record = {"input": str(arguments.input), "options": options,
              "device": str(jax.devices()[0]), "stages": []}
    in_use, peak = _stats()
    record["before"] = {"in_use": in_use, "peak": peak}

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
        "k_batch": estimate.k_batch, "band_batch": estimate.band_batch,
        "wfc_store": estimate.wfc_store, "projectors": estimate.projectors,
        "davidson_basis": estimate.davidson_basis,
        "nbnd": estimate.nbnd, "npwx": estimate.npwx, "nk": estimate.nk,
        "ngm": estimate.ngm, "smooth_grid": list(estimate.smooth_grid),
    }

    clock = time.time()
    calculation = calculator.calculation
    jax.block_until_ready(jax.live_arrays())
    in_use, peak = _stats()
    record["setup"] = {"in_use": in_use, "peak": peak, "seconds": time.time() - clock,
                       "inventory": _inventory()}
    print(f"[mem] setup  in_use {in_use / GIB:8.3f}  peak {peak / GIB:8.3f} GiB",
          flush=True)
    for row in record["setup"]["inventory"]["rows"][:12]:
        print(f"        {row['bytes'] / GIB:8.3f} GiB  x{row['count']:<3d} {row['shape']}")

    counter = defaultdict(int)
    # **Host memory between the brackets.** A stage that blows up host RAM can
    # take the process down before its wrapper returns, which leaves no entry;
    # this thread writes the largest ``VmRSS`` seen and the last stage that
    # finished to ``<out>.host`` every half second, so a kill still names where.
    last = {"stage": "setup", "max_rss": 0}
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
                    "time": time.time()}))
                time.sleep(0.5)

        threading.Thread(target=sample, daemon=True).start()

    def wrap(owner, name, label):
        original = getattr(owner, name, None)
        if original is None:
            return

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            before_use, before_peak = _stats()
            result = original(*args, **kwargs)
            try:
                jax.block_until_ready(result)
            except Exception:
                pass
            after_use, after_peak = _stats()
            counter[label] += 1
            last["stage"] = f"{label}#{counter[label]}"
            rss = _rss()
            entry = {"stage": label, "call": counter[label], "in_use_before": before_use,
                     "in_use": after_use, "peak": after_peak,
                     "peak_rise": after_peak - before_peak, "host_rss": rss}
            record["stages"].append(entry)
            if after_peak > before_peak or after_use > before_use + GIB // 4:
                print(f"[mem] {label:<22s}#{counter[label]:<3d} in_use "
                      f"{after_use / GIB:8.3f}  peak {after_peak / GIB:8.3f} GiB "
                      f"(+{(after_peak - before_peak) / GIB:.3f})  host "
                      f"{rss / GIB:6.2f} GiB", flush=True)
            if arguments.out is not None:
                arguments.out.write_text(json.dumps(record, indent=1, default=str))
            return result

        setattr(owner, name, wrapped)

    cls = type(calculation)
    for name in ("starting_wavefunctions", "diagonalize", "becsum", "density",
                 "density_and_ldos", "potential", "onecenter", "hamiltonian",
                 "symmetrize", "accuracy", "site_moments", "occupations"):
        wrap(cls, name, name)
    for name in ("_mix", "_onecenter_torque", "_paw_deband", "_magnetization"):
        wrap(driver, name, name)

    clock = time.time()
    result = calculator.get_scf(max_iterations=arguments.iterations, conv_thr=1.0e-14)
    jax.block_until_ready(result.density)
    in_use, peak = _stats()
    record["scf"] = {"in_use": in_use, "peak": peak, "seconds": time.time() - clock,
                     "iterations": int(result.iterations),
                     "accuracy": float(result.accuracy),
                     "inventory": _inventory()}
    record["ratio_measured_over_estimate"] = (
        peak / record["estimate"]["peak_bytes"] if peak else None)
    print(f"[mem] end    in_use {in_use / GIB:8.3f}  peak {peak / GIB:8.3f} GiB  "
          f"estimate {record['estimate']['peak_bytes'] / GIB:8.3f} GiB", flush=True)
    if arguments.out is not None:
        arguments.out.write_text(json.dumps(record, indent=1, default=str))
    del np


if __name__ == "__main__":
    main()
