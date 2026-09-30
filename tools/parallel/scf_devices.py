"""A warm SCF with ``h_psi`` split over ``D`` CPU devices of one process, by bands.

    taskset -c 0,2,4,6 python3 tools/parallel/scf_devices.py benchmarks/si64-1k-ecut30.in 4 --max-iterations 4

Phase 4's third experiment, after ``band_devices.py`` found ``h_psi`` over all
bands 6.8x faster on six devices of one process than on one core, where six
threads gave 2.0x. What that is worth is what a whole solve gains, since the
Davidson's dense algebra stays where it is: here every ``Hamiltonian.apply``
(scalar and spinor) is wrapped in a ``shard_map`` over a ``bands`` mesh axis,
the block padded to a multiple of ``D`` and trimmed after, and nothing else
changes. ``D = 1`` runs the unwrapped code on the same mask, the threaded
reference. Prints one JSON line: ms per iteration (median of ``--repeats``
warm runs of ``--max-iterations`` iterations), and the energy, which must agree
with ``D = 1`` to round-off.

The device count is set from the argument before JAX is imported. Pin the mask
with ``taskset`` and set ``DEFUMAT_THREADS=off``.
"""

from __future__ import annotations

import os
import sys

DEVICES = int(sys.argv[2]) if len(sys.argv) > 2 else 1
if DEVICES > 1:
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                               + f" --xla_force_host_platform_device_count={DEVICES}")

import argparse  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, PartitionSpec  # noqa: E402

import defumat.hamiltonian.noncollinear as noncollinear  # noqa: E402
import defumat.hamiltonian.operator as operator  # noqa: E402
from defumat.io.pwin import read_pw_input  # noqa: E402
from defumat.pseudo import read_upf  # noqa: E402
from defumat.scf.driver import Calculation, run_scf  # noqa: E402
from defumat.system import build_system  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def split_over_bands(apply, mesh, devices):
    """``apply(self, psi, ik)`` with the rows of ``psi`` dealt out to the devices."""
    def wrapped(self, psi, ik):
        shape = psi.shape
        flat = psi.reshape((-1, shape[-1]))
        m = flat.shape[0]
        padded = -(-m // devices) * devices
        if padded > m:
            flat = jnp.concatenate([flat, jnp.zeros((padded - m, shape[-1]), flat.dtype)])
        out = jax.shard_map(
            lambda h, p, k: apply(h, p, k), mesh=mesh,
            in_specs=(PartitionSpec(), PartitionSpec("bands"), PartitionSpec()),
            out_specs=PartitionSpec("bands"), check_vma=False,
        )(self, flat, jnp.asarray(ik))
        return out[:m].reshape(shape)
    return wrapped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("devices", type=int)
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    if DEVICES > 1:
        mesh = Mesh(np.array(jax.devices()[:DEVICES]), ("bands",))
        operator.Hamiltonian.apply = split_over_bands(
            operator.Hamiltonian.apply, mesh, DEVICES)
        spinor = [c for c in vars(noncollinear).values()
                  if isinstance(c, type) and "apply" in vars(c)]
        for cls in spinor:
            cls.apply = split_over_bands(cls.apply, mesh, DEVICES)
    system = build_system(read_pw_input(args.input))
    pseudos = tuple(read_upf(REPO / "tests" / "data" / "pseudo" / s.pseudo_file)
                    for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    options = {"conv_thr": 1e-16, "max_iterations": args.max_iterations, "verbose": False}
    start = time.perf_counter()
    result = run_scf(system, pseudos, calculation=calculation, **options)
    cold = time.perf_counter() - start
    samples = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        result = run_scf(system, pseudos, calculation=calculation, **options)
        samples.append((time.perf_counter() - start) / result.iterations)
    print(json.dumps({
        "input": args.input.name, "devices": DEVICES,
        "cpus": sorted(os.sched_getaffinity(0)), "cold_s": round(cold, 1),
        "ms_per_iter_median": round(1e3 * statistics.median(samples), 1),
        "ms_per_iter_all": [round(1e3 * s, 1) for s in samples],
        "energy_ry": float(result.total_energy),
    }), flush=True)


if __name__ == "__main__":
    main()
