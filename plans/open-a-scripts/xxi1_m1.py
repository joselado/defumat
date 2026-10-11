"""OPEN.md Part XXI item 1, M1 of plans/open-a-xxi1-cobalt-pbe.md, on the q = 0 cell.

Is the potential the derivative of the energy on the unsigned branch? At the
density an unsigned PBE run stalls at, ``jax.grad`` of ``ehart + etxc`` against
``v_scf * omega / N``, for the unsigned (axis None) and the signed (axis x)
branch at the same density, and the points where ``m . x < 0`` (the nodes the
unsigned |m| has a kink on).

    xxi1_m1.py --root W --iterations 60 [--dft pz] --out tag.json
"""
import argparse, json
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--iterations", type=int, default=60)
p.add_argument("--dft", default=None)
p.add_argument("--out", required=True)
a = p.parse_args()

from defumat import Calculator
from defumat.scf.driver import Calculation, run_scf

ROOT = Path(a.root)
text = (ROOT / "tests/data/qe/co-helix4-spiral.in").read_text()
text = text.replace("spiral_q(1) = 0.0, spiral_q(2) = 0.0, spiral_q(3) = -0.25,", "")
if a.dft:
    text = text.replace("occupations = 'smearing'", f"input_dft = '{a.dft}', occupations = 'smearing'")
front = Calculator.from_text(text, pseudo_dir=ROOT / "tests/data/pseudo", announce=False)
system, pseudos = front.system, front.pseudos
calc = Calculation(system, pseudos)
calc.quantization_axis = None
res = run_scf(system, pseudos, calculation=calc, conv_thr=1e-14,
              max_iterations=a.iterations)
rho = jnp.asarray(res.density)
omega = float(system.cell.volume)
npts = int(np.prod(rho.shape[1:]))
rec = {"args": vars(a), "accuracy": float(res.accuracy), "iterations": int(res.iterations)}

m = np.asarray(rho[1:])
mabs = np.sqrt((m ** 2).sum(0))
rec["points_m_x_negative"] = int((m[0] < 0).sum())
rec["points_small_m"] = int((mabs < 0.05 * mabs.max()).sum())
rec["points"] = npts
rec["min_m_x"], rec["max_m_x"] = float(m[0].min()), float(m[0].max())

for name, axis in (("unsigned", None), ("signed_x", (1.0, 0.0, 0.0))):
    calc.quantization_axis = axis
    def energy(r):
        pot = calc.potential(r)
        return pot.ehart + pot.etxc
    g = jax.grad(energy)(rho)
    v = calc.potential(rho).v_scf * omega / npts
    diff = np.asarray(g - v)
    scale = np.abs(np.asarray(v)).max(axis=(1, 2, 3))
    per = (np.abs(diff).max(axis=(1, 2, 3)) / scale).tolist()
    worst = np.unravel_index(np.abs(diff[1:]).argmax(), diff[1:].shape)
    rec[name] = dict(
        energy=float(energy(rho)),
        rel_mismatch_per_channel=per,
        worst_point_abs_m_over_max=float(mabs[worst[1:]] / mabs.max()),
        worst_point_m_x=float(m[0][worst[1:]]),
        # the share of the mismatch on the points where m . x < 0
        mismatch_share_on_m_x_negative=float(
            (diff[1:] ** 2).sum(0)[m[0] < 0].sum() / max((diff[1:] ** 2).sum(), 1e-300)),
        # the transverse potential's share there, against the point share
        v_m_share_on_m_x_negative=float(
            (np.asarray(v[1:]) ** 2).sum(0)[m[0] < 0].sum() / (np.asarray(v[1:]) ** 2).sum()),
    )
    print(name, rec[name], flush=True)
print(json.dumps({k: v for k, v in rec.items() if k not in ("unsigned", "signed_x")}), flush=True)
Path(a.out).write_text(json.dumps(rec, indent=1))
