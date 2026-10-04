"""What one preconditioner call costs on a large slab grid: python3 prec_cost.py <input> <out.json>.

The input supplies the cell and the cutoff, hence the dense grid; nothing is
diagonalised. The density and the LDOS are synthetic slab profiles (a metal of
thickness `SLAB` bohr centred in the cell, vacuum elsewhere) and the residual is
a smooth random field, so the inner solves see the inhomogeneity they are for.
Each preconditioner is called once to compile and then timed over five calls;
the median is reported. One FFT pair is timed as the unit.
"""
import json
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import defumat.scf.mixing as mixing
from defumat import Calculator

SLAB, RHO0, LDOS0 = 20.0, 0.03, 0.05

path, out = sys.argv[1:3]
calc = Calculator.from_file(path, announce=False)
calculation = calc.calculation
dense, cell = calculation.basis.dense, calculation.system.cell
grid = tuple(int(n) for n in dense.grid)
c = float(np.linalg.norm(np.asarray(cell.at)[2]))
z = (np.arange(grid[2]) + 0.5) / grid[2] * c
profile = 0.5 * (np.tanh((z - (c - SLAB) / 2) / 0.7) - np.tanh((z - (c + SLAB) / 2) / 0.7))
profile = np.broadcast_to(profile, grid)
density = (RHO0 * profile + 1.0e-8).reshape(1, *grid)
ldos = (LDOS0 * profile).reshape(-1)
rng = np.random.default_rng(0)
noise = rng.standard_normal(grid)
g2 = np.asarray(dense.kinetic(cell))
box = np.fft.fftn(noise).reshape(-1)
smooth = np.zeros_like(box)
index = np.asarray(dense.fft_index)
smooth[index] = box[index] * np.exp(-g2 / 0.5)
smooth[index[g2 < 1e-12]] = 0.0
residual = np.real(np.fft.ifftn(smooth.reshape(grid))).reshape(-1)
residual *= 1e-3 / np.sqrt(np.mean(residual ** 2))

shape = (1,) + grid


def median_time(call, repeats=5):
    jax.block_until_ready(call())
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(call())
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), samples


nelec = float(calculation.nelec)
kerker = mixing.kerker_preconditioner(dense, cell, shape, beta=0.7, nelec=nelec)
local_tf = mixing.local_tf_preconditioner(dense, cell, shape, beta=0.7)
fft = jax.jit(lambda x: jnp.real(jnp.fft.ifftn(jnp.fft.fftn(x.reshape(grid)))))
record = {"input": path, "grid": list(grid), "points": int(np.prod(grid))}
record["fft_pair"] = median_time(lambda: fft(jnp.asarray(residual)))[0]
record["kerker"] = median_time(lambda: kerker(residual))[0]
record["local_tf"] = median_time(lambda: local_tf(residual, density))[0]
for tol in (1e-4, 1e-3, 1e-2):
    mixing.LDOS_TOL = tol
    mixing.LDOS_COUNT_MATVECS = True
    counting = mixing.ldos_preconditioner(dense, cell, shape, beta=0.7)
    counting.update_ldos(ldos)
    counting(residual)
    mixing.LDOS_COUNT_MATVECS = False
    timed = mixing.ldos_preconditioner(dense, cell, shape, beta=0.7)
    timed.update_ldos(ldos)
    record[f"ldos_tol{tol:g}"] = median_time(lambda: timed(residual))[0]
    record[f"ldos_tol{tol:g}_matvecs"] = counting.matvecs
print(json.dumps(record, indent=1))
json.dump(record, open(out, "w"))
