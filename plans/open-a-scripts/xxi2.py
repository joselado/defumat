"""OPEN.md Part XXI item 2: Route C against the plain SCF on fully-relativistic PAW nickel, 40/320 Ry.

    xxi2.py plain|routec --iterations 150 [--beta 0.3]
"""
import argparse, json, time
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("arm", choices=("plain", "routec"))
p.add_argument("--iterations", type=int, default=150)
p.add_argument("--beta", type=float, default=0.3)
p.add_argument("--angle", type=float, default=45.0)
p.add_argument("--phi", type=float, default=0.0)
p.add_argument("--cutoffs", default="40.0,320.0")
p.add_argument("--beta-mag", type=float, default=None)
p.add_argument("--mode", default="anderson")
p.add_argument("--degauss", default=None)
p.add_argument("--kmesh", default=None)
p.add_argument("--tag", default="")
p.add_argument("--no-torque", action="store_true")
p.add_argument("--wfc-store", default="default")
p.add_argument("--split", type=int, default=1)
p.add_argument("--clear-each", action="store_true")
a = p.parse_args()

from defumat import Calculator
from defumat.scf.driver import run_scf
import threading, resource
import defumat.scf.driver as _driver


def _watch():
    while True:
        with open("/proc/self/status") as f:
            rss = [l for l in f if l.startswith(("VmRSS", "VmHWM"))]
        print("MEM", time.strftime("%H:%M:%S"), " ".join(x.split()[1] for x in rss), flush=True)
        time.sleep(20)


threading.Thread(target=_watch, daemon=True).start()
if a.no_torque:
    # The plain arm only: the one-centre torque feeds nothing without rotate_moments
    # (OPEN.md Part XXII item 1), and its tape is what the 6 GB cap kills.
    _driver._onecenter_torque = lambda *args, **kw: np.zeros(3)
if a.clear_each:
    # Route C's one-centre torque is an eager grad and jvp over fresh closures
    # every iteration, and the process grew by about 0.1 GB an iteration until
    # the 6 GB cap killed it at iteration 22; dropping the compiled code after
    # each call bounds it at the price of reloading kernels from the disk cache.
    import jax as _jax
    _original = _driver._onecenter_torque

    def _cleared(*args, **kw):
        value = _original(*args, **kw)
        _jax.clear_caches()
        return value

    _driver._onecenter_torque = _cleared

ROOT = Path("/l/ladovj1/w-a")
text = (ROOT / "tests/data/qe/ni-tetragonal-relaxed-mae-paw.in").read_text()
w_, r_ = a.cutoffs.split(",")
text = text.replace("ecutwfc = 75.0, ecutrho = 480.0,", f"ecutwfc = {w_}, ecutrho = {r_},")
text = text.replace("angle1(1) = 0.0, angle2(1) = 0.0", f"angle1(1) = {a.angle}, angle2(1) = {a.phi}")
assert f"angle2(1) = {a.phi}" in text
if a.degauss:
    text = text.replace("degauss = 0.02", f"degauss = {a.degauss}")
    assert f"degauss = {a.degauss}" in text
if a.kmesh:
    text = text.replace("3 3 2 0 0 0", a.kmesh.replace(",", " ") + " 0 0 0")
    assert a.kmesh.replace(",", " ") + " 0 0 0" in text
calc = Calculator.from_text(text, pseudo_dir=ROOT / "tests/data/pseudo", announce=False)
out = Path(f"/l/ladovj1/review/open-a/xxi2-{a.arm}{a.tag}.json")


def scalar(v):
    try:
        return float(v)
    except Exception:
        return None


t = time.time()
r = run_scf(calc.system, calc.pseudos, conv_thr=1e-10, max_iterations=a.iterations,
            mixing_beta=a.beta, mixing_beta_mag=a.beta_mag, mixing_mode=a.mode, residual_split=bool(a.split), verbose=True, wfc_store=a.wfc_store,
            rotate_moments=(a.arm == "routec"))
rows = []
for h in r.history:
    m = np.asarray(h.get("magnetization_vector", [np.nan] * 3), dtype=float)
    row = {k: scalar(h[k]) for k in ("accuracy", "becsum_accuracy", "becsum_magnetic_accuracy",
                                     "total_energy", "absolute_magnetization") if k in h}
    row["theta_deg"] = float(np.degrees(np.arccos(abs(m[2]) / np.linalg.norm(m))))
    row["phi_deg"] = float(np.degrees(np.arctan2(m[1], m[0])))
    row["m"] = m.tolist()
    if "orientation_torque" in h and h["orientation_torque"] is not None:
        row["torque"] = np.asarray(h["orientation_torque"], dtype=float).tolist()
    if "residual_split" in h:
        row["split"] = {k: (v if isinstance(v, (int, float, list)) else scalar(v))
                        for k, v in h["residual_split"].items()}
    rows.append(row)
rec = dict(arm=a.arm, beta=a.beta, converged=bool(r.converged), iterations=int(r.iterations),
           seconds=time.time() - t, history=rows, keys=sorted(r.history[-1].keys()))
out.write_text(json.dumps(rec, indent=1, default=str))
print("XXI2-DONE", a.arm, r.converged, r.iterations, flush=True)
