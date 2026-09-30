"""``h_psi`` over every band as ``G`` independent one-band loops in one executable.

    taskset -c 0,2,4,6 python3 tools/parallel/band_groups.py benchmarks/si64-1k-ecut30.in 1 4

Phase 4's cheap experiment (``PARALLEL-NEXT.local.md``): a band group without
any communication. ``map_bands`` walks the bands one at a time in a single
``lax.map``; here the bands are cut into ``G`` contiguous groups, each walked by
its own ``lax.map`` inside the same jitted function, so the groups are
independent operations that XLA's CPU runtime is free to run at once on its
thread pool. Batching the bands instead (``band_batch = T``) was measured to
lose at every width, which is why the groups stay separate loops rather than
one batched one. Prints one JSON line per ``G``: the warm time of one
application to every band, the median of ``--repeats``, and the largest
difference from ``G = 1``, which must be zero, since every band goes through
the same code.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import defumat.hamiltonian.operator as operator
from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.scf.driver import Calculation
from defumat.system import build_system

REPO = Path(__file__).resolve().parents[2]
_ORIGINAL = operator.map_bands


def grouped(groups: int):
    """``map_bands`` with the bands in ``groups`` independent one-band loops."""
    def map_bands(fn, states, *, batch="default"):
        if groups == 1 or states.ndim == 1:
            return _ORIGINAL(fn, states, batch=batch)
        shape = states.shape
        flat = states.reshape((-1,) + shape[-1:])
        edges = np.linspace(0, flat.shape[0], groups + 1).astype(int)
        parts = [lax.map(fn, flat[a:b][:, None, :])[:, 0, :]
                 for a, b in zip(edges[:-1], edges[1:]) if b > a]
        return jnp.concatenate(parts, axis=0).reshape(shape)
    return map_bands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("groups", type=int, nargs="+")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    system = build_system(read_pw_input(args.input))
    pseudos = tuple(read_upf(REPO / "tests" / "data" / "pseudo" / s.pseudo_file)
                    for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    nbnd = system.nbnd or max(int(round(calculation.nelec / 2)), 1)
    rho = calculation.starting_density()
    potential = calculation.potential(rho)
    rng = np.random.default_rng(0)
    npwx = calculation.basis.npwx * system.npol
    psi = jnp.asarray(rng.standard_normal((nbnd, npwx))
                      + 1j * rng.standard_normal((nbnd, npwx)))
    reference = None
    for groups in args.groups:
        operator.map_bands = grouped(groups)
        hamiltonian = calculation.hamiltonian(potential.v_scf)[0]
        apply = jax.jit(lambda p, h=hamiltonian: h.apply(p, 0))
        out = jax.block_until_ready(apply(psi))
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            jax.block_until_ready(apply(psi))
            samples.append(time.perf_counter() - start)
        if reference is None:
            reference = np.asarray(out)
        print(json.dumps({
            "input": args.input.name, "groups": groups, "nbnd": nbnd, "npwx": npwx,
            "cpus": sorted(os.sched_getaffinity(0)),
            "ms_median": round(1e3 * statistics.median(samples), 2),
            "ms_all": [round(1e3 * s, 2) for s in samples],
            "max_diff": float(np.max(np.abs(np.asarray(out) - reference))),
        }), flush=True)
    operator.map_bands = _ORIGINAL


if __name__ == "__main__":
    main()
