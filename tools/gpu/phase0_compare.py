"""Pair the GPU and CPU phase-0 records and read GPU.md's checks off them.

    python3 tools/gpu/phase0_compare.py out/gpu-*.json --against out/cpu-*.json

Check 2 -- "the total energy against the CPU run, to the committed tolerance" --
is a comparison between two *runs*, so it cannot live inside either of them.
This is where it happens, and it is also where §2.3's metric is finally stated:
the ratio is GPU defumat against CPU defumat, per SCF iteration, with compile
time as its own column rather than folded into it.

**The tolerance is the case's own ``conv_thr``**, used as a *flag* rather than
as a derived bound. ``conv_thr`` is a threshold on ``dr2`` -- the Hartree energy
of the density residual -- not on the total energy, and the two are not the same
quantity, so passing this test is not a proof that the energies agree to any
particular accuracy. What it is good for is the thing it was built for: every
case that agrees does so by 1e-13 or better, so anything that lands near the
threshold is anomalous by the standard of its own peers and is worth looking at.
It has already earned that once (`PERFORMANCE.md`, Phase 1's 2.6e-09 on
``si16-1k-ecut30`` at ``band_batch=all``). A round number would not have flagged
it and a derived bound would have been a fiction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(paths: list[Path]) -> dict[str, dict]:
    """Keyed by (case, k_batch, band_batch), which is what makes two runs a pair."""
    records = {}
    for path in paths:
        record = json.loads(path.read_text())
        key = (record["run"]["case"], record["dials"]["k_batch"], record["dials"]["band_batch"])
        records["|".join(key)] = record
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("gpu", nargs="+", type=Path, help="GPU-side JSON records")
    parser.add_argument("--against", nargs="+", type=Path, required=True,
                        help="CPU-side JSON records")
    arguments = parser.parse_args(argv)

    gpu, cpu = load(arguments.gpu), load(arguments.against)
    first = next(iter(gpu.values()), None)
    if first:
        provenance = first["provenance"]
        print(f"GPU: {provenance['device_kind']}  jax {provenance['jax']}  "
              f"plugin {provenance['cuda_plugin']}")
        print(f"     {provenance['nvidia_smi']}")
        print(f"     commit {str(provenance['git_commit'])[:8]}"
              f"{' DIRTY' if provenance['git_dirty'] else ''}")
    baseline = next(iter(cpu.values()), None)
    if baseline:
        print(f"CPU: {baseline['provenance']['device_kind']}  "
              f"{baseline['provenance']['cpus_per_task']} cores allocated")

    same_platform = bool(first and baseline
                         and first["provenance"]["platform"] == baseline["provenance"]["platform"])
    if same_platform:
        print("\nboth sides are the same platform: this is check 5's *cross-job* form -- "
              "the identical job twice, which catches what one process cannot")

    header = (f"\n{'case / dials':<34}{'dE (Ry)':>12}{'GPU ms/it':>11}{'CPU ms/it':>11}"
              f"{'ratio':>8}{'compile':>9}{'peak GB':>9}  {'in-run':<15}across")
    print(header)
    print("-" * len(header))

    worst = 0.0
    failures = []
    for key, record in sorted(gpu.items()):
        run = record["run"]
        label = f"{run['case']} k={record['dials']['k_batch']} b={record['dials']['band_batch']}"
        other = cpu.get(key)
        if other is None:
            print(f"{label:<34}{'no CPU pair':>12}")
            continue
        difference = run["total_energy"] - other["run"]["total_energy"]
        worst = max(worst, abs(difference))
        gpu_iter = run["per_iteration_s"] or float("nan")
        cpu_iter = other["run"]["per_iteration_s"] or float("nan")
        verdict = record["determinism"]["verdict"]
        # Bit-for-bit across the two records. Two *different* platforms are not
        # expected to agree here -- cuFFT and pocketfft sum in different orders --
        # so it is reported for both and asserted only when the platforms match,
        # which is check 5's cross-job form.
        across = ("identical" if record["run"]["runs"][0]["density_checksum"]
                  == other["run"]["runs"][0]["density_checksum"] else "differs")
        print(f"{label:<34}{difference:>12.2e}{gpu_iter * 1e3:>11.0f}{cpu_iter * 1e3:>11.0f}"
              f"{cpu_iter / gpu_iter:>7.2f}x{record['run']['compile_s'] or 0:>8.1f}s"
              f"{record['memory']['peak_gb']:>9.2f}  {verdict:<15}{across}")

        if abs(difference) > run["conv_thr"]:
            failures.append(f"{label}: {difference:.2e} Ry exceeds conv_thr {run['conv_thr']:.1e}")
        if verdict != "bit-identical":
            failures.append(f"{label}: not reproducible run to run")
        if same_platform and across != "identical":
            failures.append(f"{label}: the same job on the same platform gave different bits")
        if not record["precision"]["float64_survives"]:
            failures.append(f"{label}: x64 did not reach the device")

    print(f"\nworst energy difference: {worst:.2e} Ry")
    # A *range* over the records, not a dict comprehension over them: the FFT's
    # cost in double depends on the grid, and collapsing six measurements into
    # whichever one happened to sort last reports the flattering end of a spread
    # as if it were the number. (It did, once: 1.01x quoted from a 16^3 box where
    # the al10 grid measures 1.44x.)
    slowdowns: dict[str, list] = {}
    for record in gpu.values():
        for key, value in record["precision"].items():
            if key.endswith("_fp64_slowdown"):
                slowdowns.setdefault(key, []).append(value)
    for kernel, values in slowdowns.items():
        low, high = min(values), max(values)
        spread = f"{low:.2f}x" if high - low < 0.05 else f"{low:.2f}-{high:.2f}x"
        print(f"{kernel.replace('_fp64_slowdown', ''):>24} fp64 is {spread} fp32 "
              f"over {len(values)} records -- GPU.md check 1, and what Phase 3's rank "
              f"waits on")

    if failures:
        print("\nFAILED:")
        for line in failures:
            print(f"  {line}")
        return 1
    print("\nall checks pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
