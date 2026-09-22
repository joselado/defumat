"""Which of the four directions a noncollinear SCF is slow in -- F2's Option 0.

``MAGNETISM-NEXT.md`` item F has stayed open because the question is put one
level too low: what decides how fast a damped fixed-point iteration converges is
not the mixer but the spectrum of the map the mixer damps, and in a noncollinear
cell there are four directions in that spectrum with four different physical
origins. The item's own rule is that **none of the options below Option 0 can be
chosen without this dump**, because no cell here has ever been told apart on
which of the four it is slow in, and four different operators are on offer.

Three things, and the first two are the ones the item names:

* **the deconfounder.** ``fe-noncolin-pbe-stress.in`` takes 43 iterations where
  ``pw.x`` takes 19. Its nonmagnetic twin has never been run, so the 43 is
  attributable to nothing -- the identical 2:1 reading of ``fe-mag-1k`` fell
  apart the moment its own twin was run at 21, leaving the magnetic part of the
  excess bounded by four iterations rather than thirteen.
* **the split**, per iteration: charge, the length of ``m``, the rigid rotation
  on the three generators, the rest of the transverse channel, and the magnetic
  part of ``becsum``, which is the blind spot -- it is mixed at the plain
  ``beta`` and is in no convergence measure at all, so on a PAW magnet a stall
  can live entirely inside it with nothing in the log to say so.
* **the trip test**, because the rotation bin has to be shown to fire. A rotated
  *converged* state fed back as input must give a residual that is zero to the
  level the run converged at, since ``F(R rho) = R F(rho)`` for a
  spin-rotation-invariant functional. That is the test that a flat direction is
  flat, and it is a statement about the physics rather than about the bin.

What this does **not** do is choose between F2's options. It is the measurement
they are chosen on.

Usage:
    python3 tools/cluster/f2_option0.py --stage split --out split.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from defumat.calculator import electrons_defaults
from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.scf import Calculation, run_scf
from defumat.scf.residual_split import residual_bins
from defumat.system import build_system

MAGNETIC = Path("tests/data/qe/fe-noncolin-pbe-stress.in")
NONMAGNETIC = Path("benchmarks/fe-noncolin-nonmagnetic.in")
PSEUDO = Path("tests/data/pseudo")


def load(path: Path):
    """The cell **and its own `&electrons` namelist**, which is the whole point.

    ``run_scf`` called directly does not read an input file's ``&electrons``;
    only ``Calculator.from_file`` does, through ``electrons_defaults``. Calling
    it directly therefore silently substitutes the code's own ``mixing_beta``
    for the input's, and on this cell that is the difference between the number
    the item is about and a different number: ``fe-noncolin-pbe-stress.in`` asks
    for 0.2 and takes 43 iterations, where the default takes 24. A deconfounder
    run at the wrong beta deconfounds nothing, so the namelist is adopted here
    exactly as the facade adopts it.
    """
    pwin = read_pw_input(path)
    system = build_system(pwin)
    pseudos = tuple(
        read_upf(PSEUDO / species.pseudo_file)
        for species in system.structure.species
    )
    return system, pseudos, Calculation(system, pseudos), electrons_defaults(pwin)


def run(path: Path, conv_thr, max_iterations, split: bool, **extra):
    system, pseudos, calculation, options = load(path)
    # **The namelist wins over the command line**, and the command line only
    # supplies what the namelist left out. That is the right way round here
    # because the whole quantity being measured is an iteration count at the
    # input's own settings, and a flag that silently replaced `mixing_beta`
    # would be the defect this function exists to prevent, one layer up.
    options = {**extra, **options}
    options.setdefault("conv_thr", conv_thr)
    options.setdefault("max_iterations", max_iterations)
    started = time.time()
    result = run_scf(
        system, pseudos, calculation=calculation,
        residual_split=split, verbose=True, **options,
    )
    return system, pseudos, calculation, result, time.time() - started


def summary(result, seconds):
    return {
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "accuracy": None if result.accuracy is None else float(result.accuracy),
        "total_energy": float(result.total_energy),
        "seconds": seconds,
    }


def rotate_about_z(density, angle):
    """Turn every moment in the cell together -- the Goldstone mode itself."""
    density = np.asarray(density).copy()
    cosine, sine = np.cos(angle), np.sin(angle)
    x, y = density[1].copy(), density[2].copy()
    density[1] = cosine * x - sine * y
    density[2] = sine * x + cosine * y
    return density


def stage_deconfounder(arguments) -> dict:
    """The magnetic cell and its nonmagnetic twin, iteration count against count."""
    record = {}
    for name, path in (("magnetic", MAGNETIC), ("nonmagnetic", NONMAGNETIC)):
        _, _, _, result, seconds = run(
            path, arguments.conv_thr, arguments.max_iterations, split=False,
        )
        record[name] = summary(result, seconds)
        print(f"{name:12s} {record[name]['iterations']} iterations, "
              f"{seconds:.1f} s, converged {record[name]['converged']}")
    magnetic = record["magnetic"]["iterations"]
    nonmagnetic = record["nonmagnetic"]["iterations"]
    record["excess_attributable_to_magnetism"] = magnetic - nonmagnetic
    # `pw.x`'s own count on the magnetic cell, from P86's table, so that the
    # comparison in the record is against the number the item quotes rather
    # than against a remembered one.
    record["pw_x_iterations_on_the_magnetic_cell"] = 19
    print(f"excess over the twin  {magnetic - nonmagnetic} iterations "
          f"(pw.x takes 19 on the magnetic cell)")
    return record


def stage_split(arguments) -> dict:
    """One noncollinear run's residual, split five ways per iteration."""
    _, _, _, result, seconds = run(
        MAGNETIC, arguments.conv_thr, arguments.max_iterations, split=True,
    )
    history = []
    for entry in result.history:
        bins = entry.get("residual_split")
        if bins is None:
            continue
        history.append({
            "iteration": entry.get("iteration"),
            "accuracy": entry.get("accuracy"),
            "charge_accuracy": entry.get("charge_accuracy"),
            "magnetic_accuracy": entry.get("magnetic_accuracy"),
            **bins,
        })
    print(f"{'it':>3} {'charge':>11} {'longitud':>11} {'rotation':>11} "
          f"{'transv':>11} {'becsum':>11}")
    for row in history:
        becsum = "-" if row["becsum"] is None else f"{row['becsum']:11.3e}"
        print(f"{row['iteration']:>3} {row['charge']:11.3e} "
              f"{row['longitudinal']:11.3e} {row['rotation']:11.3e} "
              f"{row['transverse']:11.3e} {becsum}")
    return {"summary": summary(result, seconds), "history": history}


def stage_trip(arguments) -> dict:
    """A rotated converged state, fed back: the residual must be zero.

    ``F(R rho) = R F(rho)`` for a spin-rotation-invariant functional, so the
    residual of a rotated converged state is the rotated residual of a converged
    one, which is zero. A number here that is *not* at the convergence level is
    the statement that something in the run -- the symmetriser, the quantization
    axis, a guard with a preferred direction -- is not invariant, and that is
    worth more than the bin it was built to test.
    """
    system, pseudos, calculation, result, seconds = run(
        MAGNETIC, arguments.conv_thr, arguments.max_iterations, split=False,
    )
    record = {"converged": summary(result, seconds), "angles": {}}

    for angle in (0.05, 0.25, 1.0):
        rotated = rotate_about_z(result.density, angle)
        # One iteration from the rotated state: what comes back is F(R rho), and
        # the residual is what the bin should see none of.
        stepped = run_scf(
            system, pseudos, calculation=calculation,
            conv_thr=arguments.conv_thr, max_iterations=1,
            starting_density=rotated, residual_split=True,
        )
        bins = next(
            (entry["residual_split"] for entry in stepped.history
             if "residual_split" in entry), None
        )
        record["angles"][f"{angle}"] = {
            "bins": bins,
            "accuracy": None if stepped.accuracy is None else float(stepped.accuracy),
        }
        if bins is not None:
            print(f"angle {angle:5.2f}  charge {bins['charge']:.3e}  "
                  f"longitudinal {bins['longitudinal']:.3e}  "
                  f"rotation {bins['rotation']:.3e}  "
                  f"transverse {bins['transverse']:.3e}")
    return record


STAGES = {
    "deconfounder": stage_deconfounder,
    "split": stage_split,
    "trip": stage_trip,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--conv-thr", type=float, default=1e-10)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    record = STAGES[arguments.stage](arguments)
    record["stage"] = arguments.stage
    Path(arguments.out).write_text(json.dumps(record, indent=2, default=str))
    print(f"written to {arguments.out}")


if __name__ == "__main__":
    main()
