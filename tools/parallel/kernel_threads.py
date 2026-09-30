"""Bare FFTs and one complex matrix product, timed in this process's mask.

    taskset -c 0,2,4,6 env JAX_PLATFORMS=cpu python3 tools/parallel/kernel_threads.py

The control for :mod:`time_scf`: if a whole SCF iteration does not speed up
with the thread count, this says whether the kernels it is made of do. A
single transform of one band's box, eight of them batched, and a
``(20000 x 400)^H (20000 x 400)`` product, each the median of seven compiled
calls. The third box is the 45-atom NiBr2 slab's smooth grid.
"""

from __future__ import annotations

import json
import os
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

BOXES = [(48, 48, 48), (96, 96, 96), (200, 240, 54)]


def median_ms(fn, *args, n=7):
    jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(n):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - start)
    return round(1e3 * statistics.median(samples), 2)


def main() -> None:
    rng = np.random.default_rng(0)
    out = {"cpus": len(os.sched_getaffinity(0))}
    for box in BOXES:
        one = jnp.asarray(rng.standard_normal(box) + 1j * rng.standard_normal(box))
        out[f"fft{box}"] = median_ms(jax.jit(jnp.fft.fftn), one)
        eight = jnp.asarray(rng.standard_normal((8,) + box)
                            + 1j * rng.standard_normal((8,) + box))
        out[f"fft8x{box}"] = median_ms(
            jax.jit(lambda a: jnp.fft.fftn(a, axes=(1, 2, 3))), eight)
    block = jnp.asarray(rng.standard_normal((20000, 400))
                        + 1j * rng.standard_normal((20000, 400)))
    out["zgemm 400x20000x400"] = median_ms(jax.jit(lambda m: m.conj().T @ m), block)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
