"""Is the iodine atom's moment along ``z`` a minimum or a saddle?

P104 measured the transverse spin susceptibility of ``i-atom-soc.in`` at
**-658.9 mu_B/Ry**, negative and twenty-four thousand times the longitudinal one,
converged to a residual of 7.3e-9. A negative transverse susceptibility says the
moment direction is a stationary point that is **not** a minimum, and a
fixed-point iteration is stability-blind, so an SCF sits on such a point happily.
The finite-difference partner for that number could not be taken: under a
transverse field the SCF ran 200 iterations to 8.4e-8 Ry without converging,
which is what a run driven away from an unstable direction does.

That matters beyond the susceptibility, because ``i-atom-soc.in`` is the cell
P83's entire refusal rests on -- the one cell here that is at once an insulator,
textured and norm-conserving -- and nothing in P83 knew its moment direction
might be unstable.

**The measurement, and it needs no new machinery.** Seed the same cell with the
moment tilted off ``z`` by a few degrees, converge it with ``nosym`` so that
nothing forces the direction, and read where the moment ends up:

* it **returns to** ``z`` -- the direction is a minimum after all, the negative
  susceptibility is something else, and P83's comparison is about a stable state;
* it **rotates away** -- ``z`` is a saddle, both codes have been comparing
  responses about it, and that is a candidate for the 5.3 per cent that nobody
  had written down;
* it **does not converge** -- which is itself the answer a flat or unstable
  direction gives, and is what the transverse field leg already did.

**The seed angle is swept rather than guessed**, because one angle cannot tell a
slow return from a slow departure: a run started at 2 degrees that ends at 3 has
moved away, and a run started at 20 that ends at 3 has moved back, and only
comparing them says which.

Usage:
    python3 tools/cluster/p104_tilt.py --angles 2,10,30 --out tilt.json
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.scf import Calculation, run_scf
from defumat.system import build_system

CASES = Path("tests/data/qe")
PSEUDO = Path("tests/data/pseudo")


def polar(moment) -> tuple:
    """``(|m|, theta, phi)`` in mu_B and degrees, ``theta`` from ``z``."""
    moment = np.asarray(moment, dtype=float)
    length = float(np.linalg.norm(moment))
    if length == 0.0:
        return 0.0, None, None
    theta = float(np.degrees(np.arccos(np.clip(moment[2] / length, -1.0, 1.0))))
    phi = float(np.degrees(np.arctan2(moment[1], moment[0])))
    return length, theta, phi


def run_at(angle, conv_thr, max_iterations) -> dict:
    """One SCF seeded with the moment ``angle`` degrees off ``z``."""
    # `angle1` is QE's polar angle from `z` in degrees, per species, and is what
    # `build_system` turns into the starting moment direction. It is changed by
    # rewriting the input's own text rather than through the parser, which has no
    # setter and keeps per-species values in an indexed store; the substitution
    # is on the committed file so the cell stays one file and every other setting
    # is provably the same between the runs.
    text = (CASES / "i-atom-soc.in").read_text()
    assert "angle1(1) = 0.0" in text, "the committed cell no longer states angle1(1) = 0.0"
    tilted = text.replace("angle1(1) = 0.0", f"angle1(1) = {float(angle)}")
    scratch = Path(tempfile.mkdtemp()) / "i-atom-soc-tilted.in"
    scratch.write_text(tilted)
    system = build_system(read_pw_input(scratch))
    pseudos = tuple(
        read_upf(PSEUDO / species.pseudo_file)
        for species in system.structure.species
    )
    started = time.time()
    result = run_scf(
        system, pseudos, calculation=Calculation(system, pseudos),
        conv_thr=conv_thr, max_iterations=max_iterations, verbose=True,
    )
    seconds = time.time() - started

    moment = (None if result.magnetization_vector is None
              else [float(x) for x in result.magnetization_vector])
    length, theta, phi = polar(moment) if moment else (None, None, None)
    return {
        "seed_angle_degrees": float(angle),
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "accuracy": None if result.accuracy is None else float(result.accuracy),
        "total_energy": float(result.total_energy),
        "moment": moment,
        "moment_length": length,
        "theta_degrees": theta,
        "phi_degrees": phi,
        "seconds": seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--angles", default="0,2,10,30")
    parser.add_argument("--conv-thr", type=float, default=1e-11)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    angles = [float(value) for value in arguments.angles.split(",")]
    record = {"case": "i-atom-soc", "runs": []}
    for angle in angles:
        entry = run_at(angle, arguments.conv_thr, arguments.max_iterations)
        record["runs"].append(entry)
        print(f"seed {angle:5.1f} deg -> converged {entry['converged']} in "
              f"{entry['iterations']}, theta = {entry['theta_degrees']}, "
              f"|m| = {entry['moment_length']}, E = {entry['total_energy']:.9f} Ry")

    Path(arguments.out).write_text(json.dumps(record, indent=2))
    print("\nseed vs final theta, which is the whole reading:")
    for entry in record["runs"]:
        print(f"  {entry['seed_angle_degrees']:6.1f}  ->  "
              f"{entry['theta_degrees']}   "
              f"({'converged' if entry['converged'] else 'NOT converged'})")


if __name__ == "__main__":
    main()
