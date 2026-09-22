"""The external number an augmented spinor with several non-parallel moments
has never had -- ``MAGNETISM-NEXT.md`` Q5, the largest validation gap in the
noncollinear stack.

Everything that only turns on when a fully relativistic ultrasoft or PAW dataset
carries moments that are not all parallel has been unmeasured against anything:
``add_becsum_so`` on a textured ``becsum``, ``qq_so`` inside an overlap whose two
projectors sit on atoms with different local spin frames, and the ``fcoef``
recombination that dresses both. One atom cannot reach them, because one atom is
parallel to itself, and every committed noncollinear reference in this tree is
either one atom or norm-conserving.

**The pair is the measurement, not either cell alone.** ``fe2-afm-soc.in`` holds
the two moments antiparallel, which is a state one global spin axis still
describes, and ``fe2-canted-soc.in`` turns the second by 90 degrees, which is the
one that populates the off-diagonal spin blocks. A disagreement with ``pw.x`` of
the same size in both is about the augmented spinor machinery generally; one that
appears only in the canted cell is about the terms a texture switches on. Reading
the canted number alone would answer neither question.

**Three quantities, because they fail differently.** The total energy is the
integral and is the least sensitive; the two site moments are what a wrong
``add_becsum_so`` moves while leaving the energy alone; and the force is a
derivative, which is where an error in the augmentation charge shows even when
the energy is stationary. The second atom is displaced by 0.02 alat in both cells
precisely so that the force is a real number rather than the symmetry residue two
atoms related by inversion would leave.

This script runs the defumat side and writes JSON. The ``pw.x`` side is the same
input through the Fortran binary, run by the same array, and the comparison is
made afterwards from the two files.

Usage:
    python3 tools/cluster/q5_measure.py fe2-canted-soc --out result.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from defumat import Calculator

CASES = Path("tests/data/qe")
PSEUDO = Path("tests/data/pseudo")


def measure(name: str, conv_thr: float, max_iterations: int, k_batch):
    """Converge one cell and return its energy, site moments and forces."""
    calculator = Calculator.from_file(
        CASES / f"{name}.in", pseudo_dir=PSEUDO,
    )
    # `verbose` so that a run which is limit-cycling says so while it is
    # running rather than at the end: the first attempt at this cell spent
    # its whole wall clock silent, and what it was doing -- chattering at
    # 5.5e-6 Ry on too small a smearing -- was legible from iteration ten.
    options = {"conv_thr": conv_thr, "max_iterations": max_iterations,
               "verbose": True}
    if k_batch is not None:
        options["k_batch"] = k_batch

    started = time.time()
    scf = calculator.get_scf(**options)
    scf_seconds = time.time() - started

    started = time.time()
    # `get_forces` returns a `Forces`, whose `.forces` is the `(nat, 3)` array
    # in Ry/bohr cartesian -- the same frame and units `pw.x` prints, which is
    # why `test_spinor_forces.py` compares the two with no conversion at all.
    force_result = calculator.get_forces()
    forces = np.asarray(force_result.forces)
    force_seconds = time.time() - started

    moments = np.asarray(scf.site_moments)
    lengths = np.linalg.norm(moments, axis=-1)
    # The angle between the two site moments is the quantity the canted cell is
    # about, and it is what says whether the texture survived the SCF at all: a
    # run that collapsed to the collinear state reports 0 or 180 here and a
    # converged energy beside it, which is the "a check whose null result cannot
    # be told from a pass" trap if only the energy is read.
    if len(moments) == 2 and lengths.min() > 0.0:
        cosine = float(
            np.dot(moments[0], moments[1]) / (lengths[0] * lengths[1])
        )
        angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    else:
        angle = None

    return {
        "case": name,
        "converged": bool(scf.converged),
        "iterations": int(scf.iterations),
        "accuracy": None if scf.accuracy is None else float(scf.accuracy),
        "conv_thr": conv_thr,
        "total_energy": float(scf.total_energy),
        "energy_terms": {k: float(v) for k, v in scf.energy_terms.items()},
        "fermi_energy": None if scf.fermi_energy is None else float(scf.fermi_energy),
        "site_charges": np.asarray(scf.site_charges).tolist(),
        "site_moments": moments.tolist(),
        "site_moment_lengths": lengths.tolist(),
        "site_moment_angle_degrees": angle,
        "magnetization_vector": None if scf.magnetization_vector is None
                                else [float(x) for x in scf.magnetization_vector],
        "forces": forces.tolist(),
        "max_force": float(np.abs(forces).max()),
        "force_method": str(getattr(force_result, "method", "")),
        # QE's `sumfor`: the sum over atoms before it was subtracted off. It is
        # not a force, it is a convergence diagnostic, and a large value is what
        # says a comparison against `pw.x` is measuring the run rather than the
        # code.
        "total_before_correction":
            np.asarray(force_result.total_before_correction).tolist(),
        "scf_seconds": scf_seconds,
        "force_seconds": force_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", help="a stem under tests/data/qe, without .in")
    parser.add_argument("--out", required=True, help="where to write the JSON")
    parser.add_argument("--conv-thr", type=float, default=1e-11)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--k-batch", type=int, default=None)
    arguments = parser.parse_args()

    record = measure(
        arguments.case, arguments.conv_thr,
        arguments.max_iterations, arguments.k_batch,
    )
    Path(arguments.out).write_text(json.dumps(record, indent=2))

    print(f"case                 {record['case']}")
    print(f"converged            {record['converged']} in {record['iterations']}"
          f" at {record['accuracy']}")
    print(f"total energy         {record['total_energy']:.9f} Ry")
    for index, (length, moment) in enumerate(
        zip(record["site_moment_lengths"], record["site_moments"])
    ):
        vector = ", ".join(f"{x:+.6f}" for x in moment)
        print(f"site {index + 1} moment       {length:.6f} mu_B  ({vector})")
    print(f"angle between them   {record['site_moment_angle_degrees']}")
    for index, force in enumerate(record["forces"]):
        vector = ", ".join(f"{x:+.8f}" for x in force)
        print(f"force on atom {index + 1}      {vector} Ry/bohr")
    print(f"seconds              scf {record['scf_seconds']:.1f}, "
          f"force {record['force_seconds']:.1f}")


if __name__ == "__main__":
    main()
