"""One Davidson call exported from one machine and replayed on another, or with a piece moved to the host.

    python3 tools/gpu/replay/call_xplat.py <input> <store|rebuild> export <tag>   # writes <tag>_{psi0,V,ethr}.npy
    python3 tools/gpu/replay/call_xplat.py <input> <store|rebuild> replay <tag>   # prints the steps it takes here
    HOSTEIGH=all|chol|eigh  ...  replay <tag>    # the subspace solve, or only its Cholesky or only its eigh, on the host

This is how the card's stall was located: the same call took 3 steps on a CPU and 73 on the card, the
card took 6 on the CPU's inputs and the CPU 3 on the card's, 3 with only the ``eigh`` on the host and 63
with only the Cholesky there. Sets the ``ethr`` floor to 1e-13 explicitly. Norm-conserving cells only.
"""
import os, sys, warnings, dataclasses
warnings.filterwarnings("ignore")
os.environ.setdefault("DEFUMAT_CACHE_DIR", os.path.expanduser("~/.cache/defumat/jax"))
os.environ.setdefault("DEFUMAT_MEMORY_MODE", "speed")
os.environ["DEFUMAT_ETHR_MIN"] = "1e-13"
import numpy as np, jax, jax.numpy as jnp
from pathlib import Path
from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.scf.driver import Calculation, run_scf
from defumat.system import build_system
cell, proj, mode, tag = sys.argv[1:5]
if os.environ.get("HOSTEIGH") in ("chol", "eigh"):
    from defumat.solvers import subspace as _sub
    from jax.scipy.linalg import solve_triangular as _st
    which = os.environ["HOSTEIGH"]
    def _route(h, s):
        m = h.shape[0]
        if which == "chol":
            factor = jax.pure_callback(lambda a: np.linalg.cholesky(np.asarray(a)).astype(np.complex128),
                                       jax.ShapeDtypeStruct((m, m), jnp.complex128), s, vmap_method="sequential")
        else:
            factor = jnp.linalg.cholesky(s)
        reduced = _st(factor, h, lower=True)
        reduced = _st(factor, reduced.conj().T, lower=True).conj().T
        reduced = 0.5 * (reduced + reduced.conj().T)
        if which == "eigh":
            def cb(a):
                w, v = np.linalg.eigh(np.asarray(a))
                return w.astype(np.float64), v.astype(np.complex128)
            values, vectors = jax.pure_callback(cb, (jax.ShapeDtypeStruct((m,), jnp.float64),
                                                     jax.ShapeDtypeStruct((m, m), jnp.complex128)), reduced,
                                                vmap_method="sequential")
        else:
            values, vectors = jnp.linalg.eigh(reduced)
        return values, _st(factor.conj().T, vectors, lower=False)
    _sub._cholesky_route = _route
    print("subspace solve: only the", which, "on the host", flush=True)
elif os.environ.get("HOSTEIGH"):
    import scipy.linalg as sl
    from defumat.solvers import davidson as _dav
    def _host(h, s, robust=None):
        m = h.shape[0]
        def cb(hh, ss):
            w, v = sl.eigh(np.asarray(hh), np.asarray(ss), driver="gvd")
            return w.astype(np.float64), v.astype(np.complex128)
        return jax.pure_callback(cb, (jax.ShapeDtypeStruct((m,), jnp.float64),
                                      jax.ShapeDtypeStruct((m, m), jnp.complex128)), h, s,
                                 vmap_method="sequential")
    _dav.generalised_eigh = _host
    print("subspace solve: host LAPACK (scipy gvd)", flush=True)
os.environ["DEFUMAT_PROJECTORS"] = proj
system = build_system(read_pw_input(Path(cell)))
pseudos = tuple(read_upf(Path("tests/data/pseudo") / s.pseudo_file) for s in system.structure.species)
calc = Calculation(system, pseudos)
backend = jax.default_backend()
if mode == "export":
    captured = []
    orig = calc.diagonalize
    def spy(hams, nbnd, psi0=None, ethr=None, return_steps=False):
        out = orig(hams, nbnd, psi0, ethr, return_steps=True)
        captured.append((hams, nbnd, psi0, ethr, int(np.asarray(out[2]).max())))
        return out if return_steps else out[:2]
    calc.diagonalize = spy
    run_scf(system, pseudos, calculation=calc, conv_thr=1e-10)
    calc.diagonalize = orig
    k = 8
    hams, nbnd, psi0, ethr, steps = captured[k]
    np.save(tag + "_psi0.npy", np.asarray(psi0)); np.save(tag + "_V.npy", np.asarray(hams[0].potential))
    np.save(tag + "_ethr.npy", np.asarray(ethr))
    print(backend, "exported call", k, "which took", steps, "steps here", flush=True)
else:
    psi0 = jnp.asarray(np.load(tag + "_psi0.npy")); V = jnp.asarray(np.load(tag + "_V.npy")); ethr = jnp.asarray(np.load(tag + "_ethr.npy"))
    h = calc.hamiltonian(calc.potential(calc.starting_density()).v_scf)[0]
    h = dataclasses.replace(h, potential=V, potential_wave=jnp.moveaxis(V, -1, -3))
    nbnd = psi0.shape[-2]
    r = calc.diagonalize((h,), nbnd, psi0, ethr, return_steps=True)
    print(backend, "replayed the inputs of", tag, "->", int(np.asarray(r[2]).max()), "steps, unsettled", int(np.asarray(r[3]).max()), flush=True)
