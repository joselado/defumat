"""The Q5 comparison: defumat against `pw.x` on a textured augmented spinor.

Reads the JSON one side wrote and the output file the other did, and prints the
three quantities the pair is for -- the total energy, both site moments with the
angle between them, and the forces. Nothing is computed here that either side did
not already report, which is deliberate: this is a table, and a comparison script
that recomputes a quantity is a third implementation of it.

**Read the two cells together or not at all.** The antiferromagnet is the
calibration: a disagreement of the same size in both is about the augmented
spinor machinery generally, and only one that grows in the canted cell is about
the terms a texture switches on. Reading the canted number alone answers neither
question.

Usage:
    python3 tools/cluster/q5_compare.py --out-dir <where the job wrote> --job 20390511
"""
import argparse
import json
from pathlib import Path

import numpy as np

from defumat.io.qeref import read_qe_output

CASES = ("fe2-afm-soc", "fe2-canted-soc")


def angle_between(moments) -> float | None:
    moments = np.asarray(moments)
    if moments.shape[0] != 2:
        return None
    lengths = np.linalg.norm(moments, axis=-1)
    if lengths.min() <= 0.0:
        return None
    cosine = float(np.dot(moments[0], moments[1]) / (lengths[0] * lengths[1]))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def row(label, ours, theirs, unit=""):
    if ours is None or theirs is None:
        return f"  {label:<26} {'-':>18} {'-':>18} {'':>12}"
    difference = ours - theirs
    return (f"  {label:<26} {ours:>18.8f} {theirs:>18.8f} "
            f"{difference:>+12.2e} {unit}")


def compare(case: str, out_dir: Path, job: str) -> dict:
    ours = json.loads((out_dir / f"q5-{job}-{case}.json").read_text())
    theirs = read_qe_output(out_dir / f"qe-{job}-{case}.out")

    print(f"\n=== {case} " + "=" * (58 - len(case)))
    print(f"  defumat converged {ours['converged']} in {ours['iterations']} "
          f"iterations at {ours['accuracy']:.2e}")
    print(f"  {'':<26} {'defumat':>18} {'pw.x':>18} {'difference':>12}")
    print(row("total energy", ours["total_energy"], theirs.total_energy, "Ry"))

    our_moments = np.asarray(ours["site_moments"])
    their_moments = (None if theirs.local_moments is None
                     else np.asarray(theirs.local_moments))
    for index in range(len(our_moments)):
        ours_length = float(np.linalg.norm(our_moments[index]))
        theirs_length = (None if their_moments is None
                         else float(np.linalg.norm(their_moments[index])))
        print(row(f"|m| site {index + 1}", ours_length, theirs_length, "mu_B"))
    print(row("angle between moments",
              ours["site_moment_angle_degrees"],
              None if their_moments is None else angle_between(their_moments),
              "deg"))

    our_forces = np.asarray(ours["forces"])
    if theirs.forces is not None:
        worst = float(np.abs(our_forces - theirs.forces).max())
        for index in range(our_forces.shape[0]):
            for component, name in enumerate("xyz"):
                print(row(f"force {index + 1}{name}",
                          float(our_forces[index, component]),
                          float(theirs.forces[index, component]), "Ry/bohr"))
        print(f"  {'worst force component':<26} {worst:>18.2e} Ry/bohr")
    else:
        worst = None
        print("  pw.x printed no forces")

    return {
        "case": case,
        "energy_difference": (None if theirs.total_energy is None
                              else ours["total_energy"] - theirs.total_energy),
        "worst_force_difference": worst,
        "our_angle": ours["site_moment_angle_degrees"],
        "their_angle": (None if their_moments is None
                        else angle_between(their_moments)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--job", required=True)
    arguments = parser.parse_args()

    summary = []
    for case in CASES:
        try:
            summary.append(compare(case, Path(arguments.out_dir), arguments.job))
        except FileNotFoundError as error:
            print(f"\n=== {case}: missing, {error}")

    print("\n=== the reading " + "=" * 50)
    for entry in summary:
        energy = entry["energy_difference"]
        force = entry["worst_force_difference"]
        print(f"  {entry['case']:<18} dE = "
              f"{'-' if energy is None else f'{energy:+.2e}'} Ry, "
              f"worst force = {'-' if force is None else f'{force:.2e}'} Ry/bohr, "
              f"angle {entry['our_angle']} against {entry['their_angle']}")
    if len(summary) == 2 and all(e["energy_difference"] is not None for e in summary):
        calibration, textured = summary
        print(f"\n  The canted cell's energy disagreement is "
              f"{abs(textured['energy_difference']) / max(abs(calibration['energy_difference']), 1e-300):.1f}"
              f" times the antiferromagnet's, which is the number the pair exists to give: "
              f"about one means the augmented spinor machinery generally, much more than "
              f"one means the terms a texture switches on.")


if __name__ == "__main__":
    main()
