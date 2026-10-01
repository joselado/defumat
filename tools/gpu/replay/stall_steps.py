"""Per-step eigenvalue changes of the call that takes the most steps (or ``CALL=n``), at consecutive step caps.

    CALL=8 python3 tools/gpu/replay/stall_steps.py <input> <store|rebuild>

Caps ``max_iterations`` at 1, 2, 3, ... (``CAPS=2,3,4,30,31`` chooses them) and prints, for each pair of consecutive caps, how many bands
changed by more than ``ethr`` and by how much. Each cap is a recompilation. Also checks that ``H|psi>``
is bit-reproducible. The first reading of the card's stall came from this table.
"""
import os, sys, time
os.environ.setdefault("DEFUMAT_CACHE_DIR", os.path.expanduser("~/.cache/defumat/jax"))
os.environ.setdefault("DEFUMAT_MEMORY_MODE", "speed")
import jax, jax.numpy as jnp, numpy as np
from pathlib import Path
from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.scf.driver import Calculation, run_scf
from defumat.system import build_system

cell, proj = sys.argv[1], sys.argv[2]
os.environ["DEFUMAT_PROJECTORS"] = proj
system = build_system(read_pw_input(Path(cell)))
pseudos = tuple(read_upf(Path("tests/data/pseudo") / s.pseudo_file) for s in system.structure.species)
calc = Calculation(system, pseudos)

captured = []
orig = calc.diagonalize
def spy(hams, nbnd, psi0=None, ethr=None, return_steps=False):
    out = orig(hams, nbnd, psi0, ethr, return_steps=True)
    captured.append((hams, nbnd, psi0, ethr, out))
    return out if return_steps else out[:2]
calc.diagonalize = spy
r = run_scf(system, pseudos, calculation=calc, conv_thr=1e-10)
calc.diagonalize = orig
steps = [int(np.asarray(c[4][2]).max()) for c in captured]
print("captured", len(captured), "calls; steps", steps, flush=True)
worst = int(os.environ["CALL"]) if os.environ.get("CALL") else int(np.argmax(steps))
hams, nbnd, psi0, ethr, out = captured[worst]
print("replaying call", worst, "ethr", np.unique(np.asarray(ethr)), "nbnd", nbnd, flush=True)

def solve(m):
    kw = {} if calc.david is None else {"david": calc.david}
    return calc.eigensolver(hams[0], nbnd, None if psi0 is None else psi0[0], ethr[0] if jnp.ndim(ethr) == 3 else ethr,
                            k_batch=calc.k_batch, return_steps=True, max_iterations=m, **kw)
again = calc.diagonalize(hams, nbnd, psi0, ethr, return_steps=True)
print("replay steps", int(np.asarray(again[2]).max()), "(original %d)" % steps[worst], flush=True)

# H|psi> reproducibility on this device: same input twice, bitwise
psi_probe = jnp.asarray(np.asarray(again[1])[0, 0])
h = hams[0]
a1 = np.asarray(jax.jit(lambda p: h.apply(p, 0))(psi_probe)); a2 = np.asarray(jax.jit(lambda p: h.apply(p, 0))(psi_probe))
a3 = np.asarray(h.apply(psi_probe, 0))
print("H psi bitwise reproducible (same jit twice):", bool((a1 == a2).all()),
      " max|jit - eager|:", float(np.abs(a1 - a3).max()), " |H psi| max:", float(np.abs(a1).max()), flush=True)


caps = ([int(x) for x in os.environ["CAPS"].split(",")] if os.environ.get("CAPS") else list(range(1, 16)))
prev = None
print("per-step eigenvalue change (Ry) of the stalled call: the bands above ethr", flush=True)
thr = float(np.unique(np.asarray(ethr))[0])
for m in caps:
    e, psi, st, nc = solve(m)
    e = np.asarray(e)[0]
    if prev is not None and m == prev[0] + 1:
        d = np.abs(e - prev[1])
        over = np.where(d >= thr)[0]
        top = np.argsort(-d)[:3]
        print("steps %2d -> %2d  unsettled %2d  max|de| %.2e (band %2d)  next %.2e (band %2d)  bands over ethr %s" % (
            prev[0], m, len(over), d[top[0]], top[0], d[top[1]], top[1], over.tolist()[:8]), flush=True)
    prev = (m, e)
