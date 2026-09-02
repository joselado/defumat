"""Every GPU-against-CPU measurement in one figure: the dials, the stages, the size.

This is §2.3's metric -- defumat on a GPU against defumat on a CPU, same input
and same code -- kept apart from the QE comparison, which measures code and
hardware together. Numbers from ``gpu-vs-cpu.json``; nothing is re-measured.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
D = json.loads((HERE / "gpu-vs-cpu.json").read_text())
PAIRS, STAGES, QE = D["pairs"], D["stages"], D["against_qe_serial"]
WIN, LOSE = "#55A868", "#C44E52"

fig = plt.figure(figsize=(13.2, 8.4))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.26, height_ratios=[1.0, 1.0])

# ---------- 1. the dial inversion, which is Phase 0's whole finding ----------
ax = fig.add_subplot(gs[0, 0])
dial = [p for p in PAIRS if p["case"] == "al10-metal"]
names = ["k=1, b=1\n(the defaults)", "k=all, b=1", "k=all, b=all"]
order = [next(p for p in dial if (p["k_batch"], p["band_batch"]) == k)
         for k in (("1", "1"), ("all", "1"), ("all", "all"))]
x = np.arange(3); w = 0.38
ax.bar(x - w/2, [p["cpu_ms"] for p in order], w, label="CPU (4 cores)", color="#4C72B0")
ax.bar(x + w/2, [p["gpu_ms"] for p in order], w, label="GPU (V100)", color="#DD8452")
for xi, p in zip(x, order):
    ax.text(xi + w/2, p["gpu_ms"] * 1.06, f"{p['ratio']:.2f}x", ha="center", fontsize=9,
            color=WIN if p["ratio"] > 1 else LOSE, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.5)
ax.set_ylabel("ms per SCF iteration"); ax.set_ylim(0, 2500)
ax.set_title("The batching dials invert on a GPU\n"
             "al10-metal, 10 atoms, 10 k-points", fontsize=10.5)
ax.legend(fontsize=8.5); ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6); ax.set_axisbelow(True)
ax.annotate("the CPU-tuned default\ncosts 4.5x on a GPU", xy=(0.19, 801), xytext=(0.55, 1750),
            fontsize=8, color=LOSE, arrowprops=dict(arrowstyle="->", color=LOSE, lw=0.9))

# ---------- 2. per stage ----------
ax = fig.add_subplot(gs[0, 1])
labels = [r"$H\psi$", "Davidson", r"$v[\rho]$"]
keys = ["h_psi", "davidson", "v_of_rho"]
sel = [s for s in STAGES if s["dials"] == "b=all"]
x = np.arange(3); w = 0.36
for i, s in enumerate(sel):
    r = [s["cpu"][k] / s["gpu"][k] for k in keys]
    b = ax.bar(x + (i - 0.5) * w, r, w, label=f"{s['nat']} atoms",
               color=["#8DA0CB", "#66C2A5"][i], edgecolor="black", lw=0.4)
    for rect, value in zip(b, r):
        ax.text(rect.get_x() + rect.get_width()/2, value * 1.05, f"{value:.1f}x",
                ha="center", fontsize=8)
ax.axhline(1.0, color="black", lw=1.0)
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("CPU time / GPU time"); ax.set_ylim(0.4, 40)
ax.set_title("Where the gain is, stage by stage\n"
             "and how it grows with the cell (ecut 30, b=all)", fontsize=10.5)
ax.legend(fontsize=8.5); ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6); ax.set_axisbelow(True)
ax.text(2.0, 0.52, r"$v[\rho]$ is the one the GPU can lose", fontsize=7.5,
        color=LOSE, ha="center")

# ---------- 3. every matched pair, by cell size ----------
ax = fig.add_subplot(gs[1, 0])
batched = [p for p in PAIRS if p["band_batch"] == "all"]
looped = [p for p in PAIRS if p["band_batch"] == "1"]
for group, marker, label, colour in ((batched, "o", "both dials batched", WIN),
                                     (looped, "v", "QE's loop (the defaults)", LOSE)):
    ax.scatter([p["nat"] for p in group], [p["ratio"] for p in group],
               marker=marker, s=70, color=colour, edgecolor="black", lw=0.5,
               label=label, zorder=3)
ax.axhline(1.0, color="black", lw=1.0)
ax.set_yscale("log"); ax.set_xscale("log")
ax.set_xlabel("atoms in the cell"); ax.set_ylabel("CPU time / GPU time, per iteration")
ax.set_title("Every matched pair: same code, same input\n"
             "defumat GPU against defumat on 4 CPU cores", fontsize=10.5)
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(ls=":", lw=0.6, alpha=0.6); ax.set_axisbelow(True)
ax.text(2.2, 0.62, "GPU slower below this line", fontsize=7.5, color="grey")
for p in batched:
    if p["nat"] in (10, 32):
        ax.annotate(p["case"], (p["nat"], p["ratio"]), textcoords="offset points",
                    xytext=(-12, 9), fontsize=7.5)

# ---------- 4. against serial QE ----------
ax = fig.add_subplot(gs[1, 1])
cases = [q["case"] for q in QE]
x = np.arange(len(QE)); w = 0.38
ax.bar(x - w/2, [q["qe_ms"]/1000 for q in QE], w, label="QE 7.2, 1 core", color="#4C72B0")
ax.bar(x + w/2, [q["gpu_ms"]/1000 for q in QE], w, label="defumat, 1 H200", color="#DD8452")
for xi, q in zip(x, QE):
    ax.text(xi + w/2, q["gpu_ms"]/1000 * 1.5, f"{q['qe_ms']/q['gpu_ms']:.1f}x",
            ha="center", fontsize=9, color=WIN, fontweight="bold")
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels([f"{q['case']}\n{q['nat']} atoms" for q in QE], fontsize=8.5)
ax.set_ylabel("seconds per SCF iteration (log)")
ax.set_title("A different metric: against serial QE\n"
             "one core is the softest baseline — see PERFORMANCE.md", fontsize=10.5)
ax.legend(fontsize=8.5); ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6); ax.set_axisbelow(True)

fig.suptitle("defumat on a GPU — every CPU comparison measured, 2026-08-25/26",
             fontsize=13, y=0.975)
fig.savefig(HERE / "gpu-vs-cpu-fig.pdf", bbox_inches="tight")
fig.savefig(HERE / "gpu-vs-cpu-fig.png", dpi=160, bbox_inches="tight")
print("wrote gpu-vs-cpu-fig.pdf / .png")
