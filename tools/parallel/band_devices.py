"""``h_psi`` over every band with the bands split across ``D`` CPU devices of one process.

    taskset -c 0,2,4,6 python3 tools/parallel/band_devices.py benchmarks/si64-1k-ecut30.in 4

Phase 4's second experiment. A band group across *processes* would pay gloo
(100 to 600 MB/s measured on Triton) for every exchange of a block of states;
inside one process the same split costs a memory copy. ``D`` host devices
(``--xla_force_host_platform_device_count``) each take ``nbnd / D`` bands
through the one-band-at-a-time ``h_psi`` under ``shard_map``, which needs no
communication at all, since ``h_psi`` is independent per band; what it tests is
whether XLA's CPU runtime runs the devices' shares at once. Prints the warm
time on ``D`` devices and on one device with the same mask, and the largest
difference between the two results, which must be zero.

Set ``DEFUMAT_THREADS=off`` and pin the mask with ``taskset`` before Python
starts; the device count has to be set before JAX is imported, which this
script does from its argument.
"""

from __future__ import annotations

import os
import sys

DEVICES = int(sys.argv[2]) if len(sys.argv) > 2 else 1
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                           + f" --xla_force_host_platform_device_count={DEVICES}")

import json  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec  # noqa: E402

from defumat.io.pwin import read_pw_input  # noqa: E402
from defumat.pseudo import read_upf  # noqa: E402
from defumat.scf.driver import Calculation  # noqa: E402
from defumat.system import build_system  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def timed(fn, arg, repeats):
    jax.block_until_ready(fn(arg))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn(arg))
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), samples


def main() -> None:
    warnings.simplefilter("ignore")
    path = Path(sys.argv[1])
    repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    system = build_system(read_pw_input(path))
    pseudos = tuple(read_upf(REPO / "tests" / "data" / "pseudo" / s.pseudo_file)
                    for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    nbnd = system.nbnd or max(int(round(calculation.nelec / 2)), 1)
    nbnd -= nbnd % DEVICES
    potential = calculation.potential(calculation.starting_density())
    hamiltonian = calculation.hamiltonian(potential.v_scf)[0]
    rng = np.random.default_rng(0)
    npwx = calculation.basis.npwx * system.npol
    psi = jnp.asarray(rng.standard_normal((nbnd, npwx))
                      + 1j * rng.standard_normal((nbnd, npwx)))

    single = jax.jit(lambda p: hamiltonian.apply(p, 0))
    one, one_samples = timed(single, psi, repeats)
    reference = np.asarray(single(psi))

    devices = jax.devices()[:DEVICES]
    mesh = Mesh(np.array(devices), ("bands",))
    split = NamedSharding(mesh, PartitionSpec("bands"))
    sharded = jax.jit(jax.shard_map(lambda p: hamiltonian.apply(p, 0), mesh=mesh,
                                    in_specs=PartitionSpec("bands"),
                                    out_specs=PartitionSpec("bands")))
    placed = jax.device_put(psi, split)
    many, many_samples = timed(sharded, placed, repeats)
    out = np.asarray(sharded(placed))
    print(json.dumps({
        "input": path.name, "devices": DEVICES, "nbnd": nbnd, "npwx": npwx,
        "cpus": sorted(os.sched_getaffinity(0)),
        "one_device_ms": round(1e3 * one, 2), "devices_ms": round(1e3 * many, 2),
        "speedup": round(one / many, 3),
        "one_device_all": [round(1e3 * s, 1) for s in one_samples],
        "devices_all": [round(1e3 * s, 1) for s in many_samples],
        "max_diff": float(np.max(np.abs(out - reference))),
    }), flush=True)


if __name__ == "__main__":
    main()
