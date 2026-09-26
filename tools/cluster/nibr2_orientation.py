"""Which plane the NiBr2 helix lies in, on a three-cell supercell (P122).

``ORIENTATION-NEXT.md`` step 5. The NiBr2 monolayer of the ultracell project
(``calculations/NiBr2_ultracell``, its ``paw_n3_k3.scf.in`` unit cell: LDA, the
fully-relativistic PAW datasets, ``ecutwfc = 45``) stacked three times along
``a1`` with the Ni moments 120 degrees apart, the commensurate approximant that
project's three-cell runs use.

**Route A** (``routea-tilted``, ``routea-steep``): the supercell converged at
``soc_scale = 0`` on the same fully-relativistic PAW files (the one-file route,
so its ``becsum`` fits the one-shot's projectors), then ``relax_orientation``
turns the whole texture from a start with the helix's plane off the orientation
it was seeded in, one diagonalisation with the coupling per step, and reads the
curvature at the end. **Route C** (``routec-tilted``): the supercell converged
self-consistently with the coupling, the moments turned by the torque after
every mix; on PAW nickel its steps along a nearly flat direction kept the run
from converging, so its trajectory, recorded every iteration, is the
measurement rather than its end.

    python3 tools/cluster/nibr2_orientation.py routea-tilted OUTDIR --pseudo-dir DIR
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

#: The unit cell, ``paw_n3_k3.scf.in``'s own numbers (bohr, crystal).
A1 = (3.571582118400, 6.186161692473, 0.0)
A2 = (0.0, 12.372323384946, 0.0)
A3 = (0.0, 0.0, 46.255853000000)
BROMINE = ((0.0, 0.333333, 0.054571), (0.0, 0.666667, 0.945429))
CELLS = 3

#: ZYZ Euler angles of the start, radians, per task: the plane's normal tilted
#: off its seeded direction by 0.5 rad or by 80 degrees, turned off every mirror.
STARTS = {"routea-tilted": (0.3, 0.5, 0.2),
          "routea-steep": (0.3, np.radians(80.0), 0.2),
          "routec-tilted": (0.3, 0.5, 0.2)}


def supercell_text(pseudo_dir: str) -> str:
    """Three cells along ``a1``, one species per Ni so each carries its angle."""
    a1 = [CELLS * x for x in A1]
    lines = [
        "&CONTROL", "  calculation = 'scf'", "/",
        "&SYSTEM", "  ibrav = 0", f"  nat = {3 * CELLS}", f"  ntyp = {CELLS + 1}",
        "  ecutwfc = 45.0", "  ecutrho = 360.0",
        "  occupations = 'smearing'", "  smearing = 'gaussian'", "  degauss = 0.005",
        f"  nbnd = {40 * CELLS}",
        "  noncolin = .true.", "  lspinorb = .true.", "  nosym = .true.", "  noinv = .true.",
        "  input_dft = 'LDA'",
    ]
    for i in range(CELLS):
        lines += [f"  starting_magnetization({i + 1}) = 0.20",
                  f"  angle1({i + 1}) = 90.0",
                  f"  angle2({i + 1}) = {360.0 * i / CELLS:.6f}"]
    lines += ["/", "&ELECTRONS", "  conv_thr = 1.0e-8", "  mixing_beta = 0.3",
              "  mixing_ndim = 8", "/", "ATOMIC_SPECIES"]
    for i in range(CELLS):
        lines.append(f"  Ni{i + 1} 58.6934 Ni.rel-pbe-n-kjpaw_psl.0.1.UPF")
    lines += ["  Br 79.9040 Br.rel-pbe-n-kjpaw_psl.1.0.0.UPF", "CELL_PARAMETERS bohr"]
    for vector in (a1, A2, A3):
        lines.append("  " + "  ".join(f"{x:.12f}" for x in vector))
    lines.append("ATOMIC_POSITIONS crystal")
    for i in range(CELLS):
        lines.append(f"  Ni{i + 1} {i / CELLS:.9f} 0.000000000 0.000000000")
    for i in range(CELLS):
        for x, y, z in BROMINE:
            lines.append(f"  Br {(x + i) / CELLS:.9f} {y:.9f} {z:.9f}")
    lines += ["K_POINTS automatic", "  1 3 1 0 0 0", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=sorted(STARTS))
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--pseudo-dir", required=True)
    parser.add_argument("--max-iterations", type=int, default=400)
    # The one-shots' threshold. 1e-10 over 72 electrons asks Davidson for
    # ethr = 1.4e-13, which it did not reach at the steep start's third
    # orientation (up to 103 of 120 bands unsettled after 100 steps, the energy
    # noisy and BFGS stalled); 1e-8 is ethr = 1.4e-11, far below what a torque of
    # 1e-5 Ry/rad needs.
    parser.add_argument("--one-shot-conv-thr", type=float, default=1.0e-10)
    arguments = parser.parse_args()
    arguments.outdir.mkdir(parents=True, exist_ok=True)

    from defumat import Calculator
    from defumat.scf.driver import run_scf
    from defumat.workflows.anisotropy import (
        _with_rotation,
        relax_orientation,
        rotation_from_euler,
    )

    text = supercell_text(arguments.pseudo_dir)
    (arguments.outdir / "supercell.in").write_text(text)
    calculator = Calculator.from_text(text, pseudo_dir=arguments.pseudo_dir,
                                      announce=False)
    start = rotation_from_euler(*STARTS[arguments.task])
    seeded_normal = np.array([0.0, 0.0, 1.0])     # the moments are seeded in xy
    record = {"task": arguments.task,
              "start_euler": list(map(float, STARTS[arguments.task])),
              "start_normal": (start @ seeded_normal).tolist()}
    clock = time.time()

    def normal_of(site_moments):
        nickel = np.asarray(site_moments)[:CELLS]
        _, vectors = np.linalg.eigh(nickel.T @ nickel)
        return vectors[:, 0]

    if arguments.task.startswith("routea"):
        source_dir = arguments.outdir / "source-soc0"
        source = run_scf(calculator.system.with_soc_scale(0.0), calculator.pseudos,
                         conv_thr=1.0e-9, max_iterations=arguments.max_iterations,
                         checkpoint_dir=str(source_dir), checkpoint_every=1,
                         verbose=True)
        record["source"] = {"converged": bool(source.converged),
                            "iterations": int(source.iterations),
                            "accuracy": float(source.accuracy),
                            "site_moments": np.asarray(source.site_moments).tolist()}
        print(json.dumps(record["source"]), flush=True)
        relaxed = relax_orientation(calculator.system, calculator.pseudos,
                                    source.density, rotation=start,
                                    becsum=source.becsum, curvature=True,
                                    conv_thr=arguments.one_shot_conv_thr,
                                    verbose=True)
        normals = [(step.rotation @ seeded_normal).tolist() for step in relaxed.steps]
        record["relax"] = {
            "converged": relaxed.converged, "steps": len(relaxed.steps),
            "normals": normals,
            "normal_tilt_from_z_deg": [float(np.degrees(np.arccos(min(1.0, abs(n[2])))))
                                       for n in normals],
            "free_energies": relaxed.free_energies.tolist(),
            "torques": [np.asarray(step.torque).tolist() for step in relaxed.steps],
            "curvature_eigenvalues": None if relaxed.curvature is None
            else relaxed.curvature_eigenvalues.tolist(),
            "final_rotation": relaxed.rotation.tolist(),
        }
        print("normal's tilt from z per step:",
              [round(t, 3) for t in record["relax"]["normal_tilt_from_z_deg"]], flush=True)
    else:
        system = _with_rotation(calculator.system, start)
        result = run_scf(
            system, calculator.pseudos, rotate_moments=True, conv_thr=1.0e-8,
            torque_conv_thr=1.0e-7, max_iterations=arguments.max_iterations,
            checkpoint_dir=str(arguments.outdir), checkpoint_every=1, verbose=True,
        )
        normals = [normal_of(entry["site_moments"]) for entry in result.history]
        record["scf"] = {
            "iterations": int(result.iterations), "converged": bool(result.converged),
            "accuracy": float(result.accuracy), "total_energy": float(result.total_energy),
            "normal_tilt_from_z_deg": [float(np.degrees(np.arccos(min(1.0, abs(n[2])))))
                                       for n in normals],
            "normals": [n.tolist() for n in normals],
            "torques": [list(e.get("orientation_torque", [np.nan] * 3))
                        for e in result.history],
            "accuracies": [float(e["accuracy"]) for e in result.history],
        }
        print("normal's tilt from z every tenth iteration:",
              [round(t, 3) for t in record["scf"]["normal_tilt_from_z_deg"][::10]],
              flush=True)
    record["seconds"] = time.time() - clock
    record["peak_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2
    (arguments.outdir / "orientation.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
