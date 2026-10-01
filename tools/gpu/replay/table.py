"""Print the JSON lines of ``tools/parallel/time_scf.py`` as one row each, with Davidson steps.

    python3 tools/gpu/replay/table.py scan.jsonl [more.jsonl]
"""
import json, sys
for path in sys.argv[1:]:
    for l in open(path):
        d = json.loads(l)
        if d.get("failed"):
            print("FAILED", d); continue
        print("%-18s %-6s %-7s proj=%-7s lay=%-6s bb=%-4s median %8.2f  it %s  steps %s  E %.10f" % (
            d["input"][:18], d["label"], d["memory_mode"], d.get("projectors"), d.get("fft_layout"),
            d.get("resolved_band_batch"), d["ms_per_iter_median"], d["iterations"][0],
            d.get("davidson_steps"), d["energy_ry"]))
