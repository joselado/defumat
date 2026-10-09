"""OPEN.md Part XX item 2, third pass: the test's component ladder against a reference at 1e-11 and at 1e-14."""
import json
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
from tests.regression.test_ultracell import _noncollinear, _modulation, _fourier
from defumat.scf.driver import Calculation, run_scf
from defumat.ultracell import run_ultracell, with_external_potential

pseudo_dir = Path("/l/ladovj1/w-a/tests/data/pseudo")
tmp = Path("/l/ladovj1/review/open-a/xx2-tmp"); tmp.mkdir(parents=True, exist_ok=True)
shape, kgrid = (2, 1, 1), (2, 2, 2)
folded = tuple(n * m for n, m in zip(shape, kgrid))
calculator = _noncollinear(tmp, pseudo_dir, (1, 1, 1), folded, tag="lad_unit")
scf = calculator.get_scf(conv_thr=1e-11, nbnd=16)
supercell = _noncollinear(tmp, pseudo_dir, shape, kgrid, tag="lad_sup")
rc = Calculation(supercell.system, supercell.pseudos)
grid_sup = tuple(rc.basis.dense.grid)
axes = [np.arange(m) / m * n for m, n in zip(grid_sup, shape)]
coords = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
refs = {}
for thr in (1e-11, 1e-14):
    r = run_scf(supercell.system, supercell.pseudos,
                calculation=with_external_potential(rc, jnp.asarray(_modulation(shape)(coords))),
                conv_thr=thr, nbnd=32, max_iterations=300)
    refs[thr] = np.array(r.density)
    print("reference", thr, r.converged, r.iterations, flush=True)

out = {str(t): [] for t in refs}
miller = None
for nbnd in (8, 16, 32):
    res = run_ultracell(calculator.system, calculator.pseudos, scf, shape, kgrid,
                        nbnd=nbnd, conv_thr=1e-10, states_conv_thr=1e-8, mixing_beta=0.3,
                        max_iterations=200, external=_modulation(shape))
    grid_u = tuple(res.ultracell.grid)
    if miller is None:
        ax = [np.fft.fftfreq(min(u, s)) * min(u, s) for u, s in zip(grid_u, grid_sup)]
        miller = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1).reshape(-1, 3).astype(int)
    rho_u = np.asarray(res.density)
    for thr, rho_sup in refs.items():
        row = [float(np.abs(_fourier(rho_u[c], grid_u, miller) - _fourier(rho_sup[c], grid_sup, miller)).max()
                     / np.abs(_fourier(rho_sup[c], grid_sup, miller)).max()) for c in range(4)]
        out[str(thr)].append(row)
        print(nbnd, thr, row, "ratio", max(row[1:]) / min(row[1:]), flush=True)
    jax.clear_caches()
Path("/l/ladovj1/review/open-a/xx2c.json").write_text(json.dumps(out, indent=1))
print("XX2C-DONE", flush=True)
