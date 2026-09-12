"""The figure: the Fermi surface, and the tunnelling weight on the same zone.

Two Brillouin-zone maps side by side, both from the same run and the same
states. The left is the plain Fermi surface, where the zone-corner pockets are
most of what there is. The right is the vertical tunnelling weight, where they
are almost gone -- because a state at large in-plane momentum decays into the
vacuum as ``exp(-sqrt(kappa_0^2 + k_par^2) z)`` and 3.5 A of it is an order of
magnitude. Beside them, the height sweep the decay constants are fitted to.

Reads what ``tunnelling.py`` wrote; computes nothing.

Usage:  python3 calculations/nbse2-fermi-surface/plot.py [grid]
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
BOHR = 0.529177210903


def zone_image(kcartesian, values, ax, title, cmap):
    """One column on the zone, with the periodic images that close the hexagon.

    A Monkhorst-Pack grid covers ``[0, 1)`` in crystal coordinates, which is a
    *parallelogram* and not the hexagonal zone: drawing it raw puts the corner
    pockets in three of the six corners and leaves the other three empty. The
    nine in-plane translations put every point in its shortest image and the
    hexagon closes.
    """
    from defumat.transport.momentum import HEXAGONAL_CORNERS

    k = np.asarray(kcartesian)[:, :2] / BOHR          # 1/A
    shifts = []
    b = _reciprocal_from(k)
    for n1 in (-1, 0, 1):
        for n2 in (-1, 0, 1):
            shifts.append(k + n1 * b[0] + n2 * b[1])
    tiled = np.concatenate(shifts)
    repeated = np.tile(np.asarray(values), 9)
    inside = np.linalg.norm(tiled, axis=1) < 1.35 * np.abs(b).max()

    art = ax.scatter(tiled[inside, 0], tiled[inside, 1], c=repeated[inside],
                     s=14, cmap=cmap, linewidths=0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r"$k_x$  (1/$\AA$)")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    return art


def _reciprocal_from(k):
    """The two in-plane reciprocal vectors, recovered from the grid itself.

    The npz carries cartesian k-points and not the cell, and re-reading the
    input to get three numbers a plot needs would make this script depend on
    the pseudopotentials being present. A Monkhorst-Pack grid's own spacing is
    enough: the two shortest non-parallel differences from the origin, times
    the number of divisions.
    """
    n = int(round(np.sqrt(k.shape[0])))
    return np.array([k[n] * n, k[1] * n])


def main(n: int = 24) -> None:
    record = json.loads((HERE / f"tunnelling-{n}x{n}.json").read_text())
    data = np.load(HERE / f"tunnelling-{n}x{n}.npz")
    at = 2                                   # the 3.5 A row
    kcart = data["kcartesian"]

    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    zone_image(kcart, data["bare"][at], axes[0],
               "the Fermi surface\n" +
               fr"$K$ pockets carry {record['shares_at_3.5A']['bare']:.0%}",
               "magma")
    zone_image(kcart, np.log10(np.maximum(data["weight"][at], 1e-300)), axes[1],
               "what tunnels through, $\\log_{10}$\n" +
               fr"$K$ pockets carry {record['shares_at_3.5A']['weight']:.1%}",
               "viridis")

    distances = data["distances"]
    for name, label, style in (("tersoff_at_gamma", r"$\Gamma$", "o-"),
                               ("tersoff_at_k", r"$K$", "s-")):
        axes[2].semilogy(distances, record[name], style, label=label)
    kappa = record["kappa_per_angstrom"]
    axes[2].set_xlabel(r"tip height above the outer Se  ($\AA$)")
    axes[2].set_ylabel("tunnelling weight")
    axes[2].set_title(
        fr"$\kappa_\Gamma$ = {kappa['gamma']:.3f},  "
        fr"$\kappa_K$ = {kappa['k']:.3f} 1/$\AA$" "\n"
        fr"$\kappa_K^2-\kappa_\Gamma^2$ within "
        f"{record['identity']['relative_residual']:.1%} of $|K|^2$",
        fontsize=10)
    axes[2].legend(frameon=False)
    axes[0].set_ylabel(r"$k_y$  (1/$\AA$)")

    figure.suptitle(
        f"1H-NbSe$_2$, {n}x{n} k-mesh, Gaussian $\\eta$ = "
        f"{record['eta_ry'] / 2:.3f} Ha, planes 3.5 $\\AA$ out", fontsize=11)
    figure.tight_layout()
    out = HERE / f"nbse2-tunnelling-{n}x{n}.png"
    figure.savefig(out, dpi=150)
    print(f"-> {out}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 24)
