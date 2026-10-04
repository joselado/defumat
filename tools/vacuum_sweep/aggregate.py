"""Tables of the vacuum sweep: python3 aggregate.py <out-dir>."""
import json
import sys
from collections import defaultdict
from pathlib import Path

runs = {}
for path in Path(sys.argv[1]).glob("*.json"):
    if path.name.startswith("cost-"):
        continue
    cell, mode, fit = path.stem.split("_")
    runs[cell, mode, fit] = json.loads(path.read_text())

families = defaultdict(set)
for cell, _, _ in runs:
    families[cell.split("-")[0]].add(cell)

modes = ["anderson", "tf", "local-tf", "ldos", "ldos2"]


def entry(record):
    if record is None:
        return "--"
    if not record.get("converged"):
        return "n.c." if "error" not in record else "err"
    return str(record["iterations"])


for family in sorted(families):
    cells = sorted(families[family], key=lambda c: int(c.split("-")[1][1:]))
    print(f"\n{family}: iterations, flat fit / rho_ddot fit")
    print("| cell | " + " | ".join(modes) + " |")
    for cell in cells:
        row = [f"{entry(runs.get((cell, m, 'flat')))} / {entry(runs.get((cell, m, 'ddot')))}"
               for m in modes]
        print(f"| {cell} | " + " | ".join(row) + " |")
    # Every converged arm of one cell should land on one state.
    for cell in cells:
        energies = [r["energy"] for (c, _, _), r in runs.items()
                    if c == cell and r.get("converged")]
        moments = [r["magnetization"] for (c, _, _), r in runs.items()
                   if c == cell and r.get("converged") and "magnetization" in r]
        line = f"  {cell}: energy spread {max(energies) - min(energies):.1e} Ry over {len(energies)} runs"
        if moments:
            line += f", moment {min(moments):.5f}..{max(moments):.5f}"
        print(line)

print("\nLDOS arm: matvecs per call, preconditioner and LDOS seconds per call")
for (cell, mode, fit), r in sorted(runs.items()):
    if mode.startswith("ldos") and r.get("precond_calls"):
        print(f"  {cell} {mode} {fit}: {r['matvecs'] / r['precond_calls']:.1f} matvecs, "
              f"{r['precond_seconds'] / r['precond_calls']:.3f} s, "
              f"{r['ldos_seconds'] / max(r['ldos_calls'], 1):.3f} s LDOS, "
              f"of {r['seconds'] / r['iterations']:.2f} s an iteration")

print("\nfailures")
for key, r in sorted(runs.items()):
    if not r.get("converged"):
        print(" ", key, r.get("iterations"), r.get("error", "")[:200],
              "last energies", [round(e, 4) for e in r.get("energies", [])[-3:]])
