"""OPEN.md Part XXI item 1: the ultrasoft cobalt spiral against its unfolded supercell.

    xxi1.py --ecutrho 160 [--dft pz] [--inplane 3] --out tag
"""
import argparse, json, time, warnings
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

p = argparse.ArgumentParser()
p.add_argument("--ecutrho", type=float, default=160.0)
p.add_argument("--ecutwfc", type=float, default=20.0)
p.add_argument("--dft", default=None)
p.add_argument("--inplane", type=int, default=3)
p.add_argument("--out", required=True)
p.add_argument("--pair", default="co", choices=("co", "o"))
a = p.parse_args()

from defumat import Calculator
from defumat.scf.driver import Calculation
from defumat.workflows.spiral import unfold_spiral_density
from defumat.workflows.nscf import fixed_density_states

ROOT = Path("/l/ladovj1/w-a")
PSEUDO = ROOT / "tests/data/pseudo"


def edit(text):
    text = text.replace("ecutwfc = 20.0, ecutrho = 160.0,",
                        f"ecutwfc = {a.ecutwfc}, ecutrho = {a.ecutrho},")
    if a.dft:
        text = text.replace("occupations = 'smearing'", f"input_dft = '{a.dft}', occupations = 'smearing'")
    text = text.replace("3 3 4 0 0 0", f"{a.inplane} {a.inplane} 4 0 0 0")
    text = text.replace("3 3 1 0 0 1", f"{a.inplane} {a.inplane} 1 0 0 1")
    return text


rec = {"args": vars(a)}
names = {"co": ("co-helix4-spiral.in", "co-helix4-nosoc.in"),
         "o": ("o-chain-spiral-us.in", "o-chain-90deg-us.in")}[a.pair]
spiral = Calculator.from_text(edit((ROOT / "tests/data/qe" / names[0]).read_text()),
                              pseudo_dir=PSEUDO, announce=False)
sup = Calculator.from_text(edit((ROOT / "tests/data/qe" / names[1]).read_text()),
                           pseudo_dir=PSEUDO, announce=False)
t = time.time()
scf = spiral.get_scf(conv_thr=1e-11, max_iterations=150)
last = scf.history[-1]
rec["spiral_scf"] = dict(iterations=int(scf.iterations), converged=bool(scf.converged),
                         accuracy=float(scf.accuracy), seconds=time.time() - t,
                         energy=float(scf.total_energy),
                         last={k: float(v) for k, v in last.items()
                               if np.ndim(v) == 0 and isinstance(v, (int, float, np.floating, np.integer))})
print("spiral", json.dumps(rec["spiral_scf"]), flush=True)


def _ef(levels):
    if isinstance(levels, dict):
        for v in levels.values():
            try:
                return float(np.asarray(v).ravel()[0])
            except Exception:
                pass
        return 0.0
    return float(np.asarray(levels).ravel()[0])


def rel(out, inp):
    out, inp = np.asarray(out), np.asarray(inp)
    return dict(charge=float(np.abs(out[0] - inp[0]).max() / np.abs(inp[0]).max()),
                magnetization=float(np.abs(out[1:] - inp[1:]).max() / np.abs(inp[1:]).max()))


# The spiral against itself: a one-shot at its own converged density.
calc_s, _, eig_s, st_s = fixed_density_states(spiral.system, spiral.pseudos, scf.density,
                                              conv_thr=1e-11)
w_s, ef_s = calc_s.occupations(jnp.asarray(eig_s))
rec["spiral_self"] = rel(calc_s.density(st_s, w_s), scf.density)
rec["spiral_fermi"] = str(ef_s)
print("spiral self", rec["spiral_self"], flush=True)
del st_s

calc0 = Calculation(sup.system, sup.pseudos)
shape = tuple(calc0.basis.dense.grid)
rec["grids"] = dict(spiral=list(np.asarray(scf.density).shape[1:]), supercell=list(shape),
                    doublegrid=bool(calc0.basis.doublegrid),
                    smooth_spiral=None, smooth_supercell=list(calc0.basis.smooth.grid))
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    density = unfold_spiral_density(scf.density, spiral.system.spiral_q, (1, 1, 4), shape)
rec["unfold_warnings"] = [str(w.message)[:200] for w in caught]
t = time.time()
calc, _, eig, st = fixed_density_states(sup.system, sup.pseudos, jnp.asarray(density),
                                        conv_thr=1e-11, calculation=calc0)
w, ef = calc.occupations(jnp.asarray(eig))
out = np.asarray(calc.density(st, w))
rec["supercell_vs_unfolded"] = rel(out, density)
rec["supercell_fermi"] = str(ef)
rec["supercell_seconds"] = time.time() - t
print("supercell", rec["supercell_vs_unfolded"], rec["spiral_fermi"], rec["supercell_fermi"], flush=True)

# Eigenvalues: each supercell (kx, ky) against the union of the four spiral k-points there.
ks = np.asarray(spiral.system.kpoints.coords); kc = np.asarray(sup.system.kpoints.coords)
es = np.asarray(eig_s).reshape(len(ks), -1); ec = np.asarray(eig).reshape(len(kc), -1)
worst, worst_occ = 0.0, 0.0
efc = _ef(ef)
for i, k in enumerate(kc):
    same = np.where(np.abs(ks[:, :2] - k[:2]).max(axis=1) < 1e-6)[0]
    union = np.sort(es[same].ravel())
    n = min(len(union), ec.shape[1]) - 4  # leave the top few, which are not converged alike
    d = np.abs(union[:n] - ec[i, :n])
    worst = max(worst, float(d.max()))
    occ = ec[i, :n] < efc + 0.1
    if occ.any():
        worst_occ = max(worst_occ, float(d[occ].max()))
rec["eigenvalues_worst"] = worst
rec["eigenvalues_worst_below_ef_plus_0.1"] = worst_occ
print("eigenvalues", worst, worst_occ, flush=True)

# Where in G the output-minus-input difference lives: the shell a displaced sphere misses.
cell = sup.system.cell
dense = calc.basis.dense
miller = np.asarray(dense.miller)
G = np.asarray(dense.cartesian(cell))
gmax = np.sqrt(np.max(np.sum(G ** 2, axis=1)))
q_s = np.asarray(sup.system.cell.bg)[2] * 1.0  # one supercell b3 = q of the spiral (|q| = b3_unit/4)
diff = out - density


def coeff(field):
    f = np.fft.fftn(field) / field.size
    return f[tuple(np.mod(miller[:, i], shape[i]) for i in range(3))]


def shell_split(c):
    p2 = np.abs(c) ** 2
    tot = p2.sum()
    res = {}
    for name, sgn in (("plus", 1), ("minus", -1)):
        outside = np.sqrt(np.sum((G + sgn * q_s) ** 2, axis=1)) > gmax
        res[name] = float(p2[outside].sum() / tot)
    edge = np.sqrt(np.sum(G ** 2, axis=1)) > gmax - np.linalg.norm(q_s)
    res["edge_band"] = float(p2[edge].sum() / tot)
    res["edge_band_count_fraction"] = float(edge.mean())
    return res


rec["shell"] = dict(
    gmax=float(gmax), q=float(np.linalg.norm(q_s)),
    transverse=shell_split(coeff(diff[1] + 1j * diff[2])),
    charge=shell_split(coeff(diff[0])), mz=shell_split(coeff(diff[3])))
print("shell", json.dumps(rec["shell"]), flush=True)
Path(f"/l/ladovj1/review/open-a/xxi1-{a.out}.json").write_text(json.dumps(rec, indent=1))
print("XXI1-DONE", flush=True)
