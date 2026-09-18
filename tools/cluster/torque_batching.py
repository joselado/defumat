"""What the magnetic torque's k dial is worth, in the answer and in the peak.

``AUDIT-2026-09-18.md`` hole.1: the torque's band energy walked the k axis with
a Python loop *inside the function ``jax.grad`` differentiates*, so the backward
pass held one real-space block per k-point **simultaneously** -- the array
``SpinorHamiltonian._local_block``'s own docstring sizes at
``nbnd x 2 x N_smooth``, 33 GB for one k-point of the P74 cell -- and no dial
reached it. ``k_batch`` stopped at the NSCF that produced the states, and
``DEFUMAT_BAND_BATCH`` reaches ``map_bands`` inside the operator, where a scan
stacks its residuals under ``jax.grad`` just the same. A run given ``k_batch=1``
so that it would fit then asked for the whole axis anyway.

The entry sized that structurally, from the shapes the tape must hold, and said
so: no torque memory figure exists anywhere in the record. This is the figure.

**One route per process**, because a peak is a high-water mark: two routes in
one process report the larger of the two twice. Both sides run with the kernel
cache off, so both are compilation misses and the comparison is not a
comparison of cache states -- which `CLAUDE.md` records as costing 6.3 GB on one
spinor PAW test, in the direction that flatters whichever side was the miss.

    python3 tools/cluster/torque_batching.py tetragonal whole
    python3 tools/cluster/torque_batching.py tetragonal chunked
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

CASES = {
    # One atom, 18 k-points: small enough that the two routes' agreement is the
    # statement, and the peak difference is expected to be small.
    "tetragonal": {
        "scalar": "tests/data/qe/co-tetragonal-anisotropy-sr.in",
        "spinor": "tests/data/qe/co-tetragonal-anisotropy-soc.in",
    },
    # Three atoms of cobalt with vacuum, ecutwfc 25 and ecutrho 200 on a
    # shifted 4x4x1 grid: the smallest committed cell whose real-space block is
    # a slab's rather than a bulk cell's, which is where the entry's argument
    # lives.
    "slab": {
        "scalar": "tests/data/qe/co-slab-forcetheorem-sr.in",
        "spinor": "tests/data/qe/co-slab-forcetheorem-par.in",
    },
}


def peak_gib() -> float:
    """The process's high-water resident set, in GiB (``ru_maxrss`` is KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("route", choices=("whole", "chunked"))
    parser.add_argument("--chunk", type=int, default=1,
                        help="k-points per chunk on the chunked route")
    parser.add_argument("--conv-thr", type=float, default=1.0e-10)
    parser.add_argument("--out", default=None)
    parser.add_argument("--pseudo-dir", default=str(ROOT / "tests" / "data" / "pseudo"))
    arguments = parser.parse_args()

    import jax.numpy as jnp

    from defumat.calculator import Calculator
    from defumat.forces.torque import band_energy_at_angle, torque_at_angle
    from defumat.scf.continuation import nc_magnetization_from_lsda
    from defumat.workflows.anisotropy import _with_quantization_axis
    from defumat.workflows.nscf import fixed_density_states

    case = CASES[arguments.case]
    k_batch = None if arguments.route == "whole" else arguments.chunk
    print(f"=== {arguments.case}, route {arguments.route} "
          f"(k_batch = {k_batch})", flush=True)

    scalar = Calculator.from_file(ROOT / case["scalar"],
                                  pseudo_dir=arguments.pseudo_dir, announce=False)
    spinor = Calculator.from_file(ROOT / case["spinor"],
                                  pseudo_dir=arguments.pseudo_dir, announce=False)

    start = time.time()
    scf = scalar.get_scf(conv_thr=arguments.conv_thr)
    print(f"    scalar SCF {scf.total_energy:.10f} Ry in {scf.iterations} "
          f"iterations, {time.time() - start:.1f} s, peak {peak_gib():.2f} GiB",
          flush=True)

    angle = np.pi / 4.0
    plane = ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
    direction = (np.cos(angle) * np.asarray(plane[0])
                 + np.sin(angle) * np.asarray(plane[1]))
    system = _with_quantization_axis(spinor.system, tuple(direction))
    rotated = nc_magnetization_from_lsda(scf.density, tuple(direction))

    start = time.time()
    calculation, system, eigenvalues, states = fixed_density_states(
        system, spinor.pseudos, rotated, conv_thr=arguments.conv_thr,
    )
    weights, _ = calculation.occupations(jnp.asarray(eigenvalues))
    nscf_seconds = time.time() - start
    print(f"    spinor NSCF over {int(np.asarray(states).shape[1])} k-points, "
          f"{nscf_seconds:.1f} s, peak {peak_gib():.2f} GiB", flush=True)

    # The energy first, so that the gradient below is the only thing between
    # this peak and the next one.
    check = float(band_energy_at_angle(calculation, states, weights,
                                       scf.density, plane, angle))
    before = peak_gib()

    start = time.time()
    torque = torque_at_angle(calculation, states, weights, scf.density, plane,
                             angle, k_batch=k_batch)
    seconds = time.time() - start
    after = peak_gib()

    print(f"    band energy {check:.10f} Ry", flush=True)
    print(f"    torque {torque:.12e} Ry/rad in {seconds:.1f} s", flush=True)
    print(f"    peak before the gradient {before:.2f} GiB, after {after:.2f} GiB",
          flush=True)

    if arguments.out:
        Path(arguments.out).write_text(json.dumps({
            "case": arguments.case,
            "route": arguments.route,
            "k_batch": k_batch,
            "nk": int(np.asarray(states).shape[1]),
            "band_energy": check,
            "torque": float(torque),
            "gradient_seconds": seconds,
            "nscf_seconds": nscf_seconds,
            "peak_before_gib": before,
            "peak_after_gib": after,
        }, indent=1))
        print(f"    wrote {arguments.out}", flush=True)


if __name__ == "__main__":
    main()
