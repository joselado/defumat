"""Davidson steps of one SCF call, over round-off-sized perturbations of its starting states.

    python3 tools/gpu/replay/stall_stats.py <input> <store|rebuild> [seeds] [call] [ethr,ethr,...]

Runs the SCF, captures the Davidson call ``call`` (default -2, the last iteration with the tight
threshold), then perturbs its starting states by 1e-13 relative noise ``seeds`` times at each
``ethr`` and prints the distribution of step counts. ``DAVID=n`` sets ``diago_david_ndim``;
``NOISE_ETA=x`` injects relative noise into the projected pair. Run it once per platform: a CPU
took exactly 3 steps for every seed where a card took 3 to 100 (``PERFORMANCE.md``, "The endgame on a
card is a stall"). One process per configuration, and nothing else on the card meanwhile.
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


ETA = float(os.environ.get("NOISE_ETA", "0"))
if ETA > 0:
    from defumat.solvers import davidson as _dav, subspace as _sub
    _orig = _sub.generalised_eigh
    def _noisy(h, s, robust=None):
        # relative noise of size ETA on every entry of the projected pair, a deterministic
        # function of the inputs (as round-off is), Hermitian so the problem stays Hermitian
        bits = jax.lax.bitcast_convert_type(jnp.sum(h.real) + 3.0 * jnp.sum(s.real), jnp.uint64)
        key = jax.random.fold_in(jax.random.PRNGKey(7), (bits % jnp.uint64(2**31 - 1)).astype(jnp.uint32))
        k1, k2 = jax.random.split(key)
        def sym(k):
            a = jax.random.normal(k, h.shape, dtype=jnp.float64) + 1j * jax.random.normal(jax.random.fold_in(k, 1), h.shape, dtype=jnp.float64)
            return 0.5 * (a + a.conj().T)
        hn = h + ETA * sym(k1) * jnp.abs(h)
        sn = s + ETA * sym(k2) * jnp.abs(s)
        return _orig(0.5 * (hn + hn.conj().T), 0.5 * (sn + sn.conj().T), robust=robust)
    _dav.generalised_eigh = _noisy

cell, proj = sys.argv[1], sys.argv[2]
nseed = int(sys.argv[3]) if len(sys.argv) > 3 else 20
call_index = int(sys.argv[4]) if len(sys.argv) > 4 else -2          # the last iteration with the tight threshold
os.environ["DEFUMAT_PROJECTORS"] = proj
system = build_system(read_pw_input(Path(cell)))
pseudos = tuple(read_upf(Path("tests/data/pseudo") / s.pseudo_file) for s in system.structure.species)
calc = Calculation(system, pseudos, david=(int(os.environ["DAVID"]) if os.environ.get("DAVID") else None))
captured = []
orig = calc.diagonalize
def spy(hams, nbnd, psi0=None, ethr=None, return_steps=False):
    out = orig(hams, nbnd, psi0, ethr, return_steps=True)
    captured.append((hams, nbnd, psi0, ethr, out))
    return out if return_steps else out[:2]
calc.diagonalize = spy
r = run_scf(system, pseudos, calculation=calc, conv_thr=1e-10)
calc.diagonalize = orig
hams, nbnd, psi0, ethr, out = captured[call_index]
print(jax.default_backend(), proj, "call", call_index % len(captured), "of", len(captured),
      "scf steps", [int(np.asarray(c[4][2]).max()) for c in captured], flush=True)
base = np.asarray(psi0)
rng = np.random.default_rng(1)
noise = 1e-13
for e in [float(x) for x in (sys.argv[5].split(",") if len(sys.argv) > 5 else "1e-13,2.1e-13,5e-13,1e-12,3e-12,1e-11,1e-10".split(","))]:
    steps = []
    for seed in range(nseed):
        pert = base * (1.0 + noise * (rng.standard_normal(base.shape) + 1j * rng.standard_normal(base.shape)))
        res = calc.diagonalize(hams, nbnd, jnp.asarray(pert), ethr=e, return_steps=True)
        steps.append(int(np.asarray(res[2]).max()))
    s = np.array(steps)
    print("ethr %.1e  steps: min %3d  median %5.1f  max %3d  fraction over 20: %.2f   %s" % (
        e, s.min(), np.median(s), s.max(), (s > 20).mean(), sorted(steps)), flush=True)
