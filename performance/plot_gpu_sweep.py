"""The GPU sweep as two figures: what it costs, and whether it is right.

Regenerate with ``python3 performance/plot_gpu_sweep.py``; the numbers come from
``gpu-sweep.json`` and nothing is re-measured here.

**The colour is the physics, and that is the whole point of the left panel.**
The speedup is not one number -- it runs from 2x to 115x -- and it is not random
either: it tracks how much work there is per k-point, which is what a GPU needs
to fill. Sorting by ratio and colouring by regime makes that visible in a way
the table cannot.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
DATA = json.loads((HERE / "gpu-sweep.json").read_text())

# Colour-blind-safe, and ordered so the regimes read as a progression.
COLOURS = {"insulator": "#4C72B0", "metal": "#DD8452",
           "magnetic": "#55A868", "spin-orbit": "#C44E52"}
NAMES = {"insulator": "non-magnetic", "metal": "metal (smearing)",
         "magnetic": "magnetic / DFT+U", "spin-orbit": "spin-orbit"}


def figure():
    cases = DATA["cases"]
    labels = [f"{c['case']}  ({c['nat']} at.)" for c in cases]
    ratio = np.array([c["speedup_per_iteration"] for c in cases])
    colours = [COLOURS[c["group"]] for c in cases]

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12.4, 5.4), gridspec_kw={"width_ratios": [1.55, 1.0]})

    # ---- left: the speedup, sorted, log scale because it spans two decades ----
    y = np.arange(len(cases))
    left.barh(y, ratio, color=colours, edgecolor="black", linewidth=0.4)
    left.set_yticks(y)
    left.set_yticklabels(labels, fontsize=9)
    left.set_xscale("log")
    left.set_xlim(1.0, 260.0)
    left.axvline(1.0, color="black", lw=1.0)
    left.set_xlabel("defumat on one H200  /  Quantum ESPRESSO on one CPU core\n"
                    "(per SCF iteration, log scale)", fontsize=9)
    for yi, value in zip(y, ratio):
        left.text(value * 1.10, yi, f"{value:.1f}x", va="center", fontsize=8.5)
    left.set_title("Speedup per SCF iteration", fontsize=11, pad=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOURS[k]) for k in NAMES]
    left.legend(handles, list(NAMES.values()), fontsize=8.5, loc="lower right",
                frameon=True, title="physics", title_fontsize=8.5)
    left.grid(axis="x", ls=":", lw=0.6, alpha=0.6)
    left.set_axisbelow(True)

    # ---- right: the agreement, which is the claim the timings rest on ----
    delta = np.array([abs(c["delta_energy"]) for c in cases])
    per_atom = delta / np.array([c["nat"] for c in cases])
    right.scatter([c["nat"] for c in cases], per_atom,
                  c=colours, s=64, edgecolor="black", linewidth=0.5, zorder=3)
    right.set_yscale("log")
    right.set_xlabel("atoms in the cell", fontsize=9)
    right.set_ylabel(r"$|E_{\rm defumat} - E_{\rm QE}|$  per atom  (Ry)", fontsize=9)
    right.set_title("Agreement with Quantum ESPRESSO", fontsize=11, pad=8)
    right.axhline(1e-9, color="grey", ls="--", lw=0.9)
    right.text(41, 1.25e-9, "1e-9 Ry/atom", fontsize=8, color="grey", ha="right")
    right.grid(ls=":", lw=0.6, alpha=0.6)
    right.set_axisbelow(True)
    # The two bismuth points are a *known* defumat discrepancy, not a GPU one --
    # labelling them is the difference between a plot and a misleading plot.
    for c, value in zip(cases, per_atom):
        if c["group"] == "spin-orbit":
            right.annotate(c["case"], (c["nat"], value), textcoords="offset points",
                           xytext=(6, -4), fontsize=8, color="#C44E52")
    right.annotate("bismuth: a pre-existing defumat/QE\n"
                   "difference (PLAN.md), reproduced\n"
                   "by the GPU rather than caused by it",
                   xy=(20, per_atom[-1]), xytext=(11, 2.0e-7),
                   fontsize=7.6, color="#C44E52",
                   arrowprops=dict(arrowstyle="->", color="#C44E52", lw=0.8))

    fig.suptitle("defumat on a GPU against Quantum ESPRESSO on one CPU core — "
                 f"{len(cases)} cases, conv_thr = {DATA['meta']['conv_thr']}",
                 fontsize=12.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = HERE / "gpu-sweep-fig.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(HERE / "gpu-sweep-fig.png", dpi=170, bbox_inches="tight")
    print(f"wrote {out} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    figure()
