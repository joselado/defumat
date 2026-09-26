"""The orientation torque and relaxation on the four-cell cobalt helix (P122).

``ORIENTATION-NEXT.md`` steps 2 and 3 on the noncollinear cell: the helix
converged without spin-orbit coupling in its one-atom cell by the generalized
Bloch theorem (``tests/data/qe/co-helix4-spiral.in``), unfolded onto the four-cell
supercell (``unfold_spiral_density``), then turned rigidly and diagonalised once
with the coupling (``co-helix4-soc.in``). The supercell's own SCF limit-cycles
near 1e-7 Ry, the 90-degree helix of cobalt being an unstable stationary point,
which is why the source comes from the spiral.

``checks`` measures the torque at an oblique orientation and the numbers that
say it is right: each component against a central difference of the free energy
from two separate one-shots, the closed form ``integral of m_out x B``, and the
same torque with the coupling off. ``relax`` runs the relaxation from the same
start with the curvature at the end. One task per process, so each peak is its
own.

    python3 tools/cluster/orientation_helix.py checks --out checks.json
    python3 tools/cluster/orientation_helix.py relax --out relax.json
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPIRAL = ROOT / "tests/data/qe/co-helix4-spiral.in"
SPINOR = ROOT / "tests/data/qe/co-helix4-soc.in"

#: ZYZ Euler angles of the start, radians: the helix plane's normal tilted 0.5
#: rad from ``c`` and turned off the lattice's mirror planes, so that no
#: component of the torque is a symmetry's zero.
START = (0.3, 0.5, 0.2)
#: Central-difference step for the free energy, radians.
STEP = 2.0e-3


def peak_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("checks", "relax"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--pseudo-dir", default=str(ROOT / "tests/data/pseudo"))
    arguments = parser.parse_args()

    import jax.numpy as jnp

    from defumat import Calculator
    from defumat.forces.torque import orientation_torque, rotate_texture
    from defumat.scf.driver import Calculation
    from defumat.workflows.spiral import unfold_spiral_density
    from defumat.scf.spin_torque import exchange_torque
    from defumat.workflows.anisotropy import (
        _exp_rotation,
        _reference_axis,
        _reference_texture,
        _with_rotation,
        relax_orientation,
        rotation_from_euler,
        run_orientation_torque,
    )
    from defumat.workflows.nscf import fixed_density_states

    record: dict = {"task": arguments.task, "start_euler": START}
    clock = time.time()
    spiral = Calculator.from_file(SPIRAL, pseudo_dir=arguments.pseudo_dir,
                                  announce=False)
    spinor = Calculator.from_file(SPINOR, pseudo_dir=arguments.pseudo_dir,
                                  announce=False)
    scf = spiral.get_scf(verbose=True)
    record["scf"] = {
        "seconds": time.time() - clock, "iterations": int(scf.iterations),
        "converged": bool(scf.converged), "accuracy": float(scf.accuracy),
        "total_energy_per_cell": float(scf.total_energy),
    }
    print(json.dumps(record["scf"]), flush=True)
    shape = Calculation(spinor.system, spinor.pseudos).basis.dense.grid
    density = unfold_spiral_density(scf.density, spiral.system.spiral_q,
                                    (1, 1, 4), shape)

    rotation = rotation_from_euler(*START)
    system, pseudos = spinor.system, spinor.pseudos

    if arguments.task == "checks":
        clock = time.time()
        texture = _reference_texture(density, _reference_axis(system))
        calculation, _, eigenvalues, states = fixed_density_states(
            _with_rotation(system, rotation), pseudos,
            rotate_texture(texture, rotation), conv_thr=1.0e-10)
        weights, levels = calculation.occupations(jnp.asarray(eigenvalues))
        torque = orientation_torque(calculation, states, weights, texture, rotation)
        output = calculation.density(states, weights)
        potential = calculation.potential(rotate_texture(texture, rotation), 1.0, None)
        closed = np.asarray(exchange_torque(output, potential.v_scf,
                                            calculation.system.cell).total)
        record["torque"] = torque.tolist()
        record["closed_form"] = closed.tolist()
        record["closed_form_relative"] = float(
            np.linalg.norm(closed - torque) / np.linalg.norm(torque))
        record["one_shot_seconds"] = time.time() - clock
        print("torque", torque, "closed", closed, flush=True)

        central = []
        for axis in np.eye(3):
            energies = []
            for sign in (+1.0, -1.0):
                turned = _exp_rotation(sign * STEP * axis) @ rotation
                result = run_orientation_torque(system, pseudos, density,
                                                rotation=turned)
                energies.append(result.free_energy)
            central.append(-(energies[0] - energies[1]) / (2.0 * STEP))
            print("central", axis, central[-1], flush=True)
        record["central_difference"] = central
        record["central_relative"] = float(
            np.linalg.norm(np.asarray(central) - torque) / np.linalg.norm(torque))

        off = run_orientation_torque(system, pseudos, density,
                                     rotation=rotation, soc_scale=0.0)
        record["torque_soc_scale_0"] = np.asarray(off.torque).tolist()
        print("soc_scale 0", off.torque, flush=True)
    else:
        clock = time.time()
        relaxed = relax_orientation(system, pseudos, density,
                                    rotation=rotation, curvature=True,
                                    verbose=True)
        record["relax"] = {
            "seconds": time.time() - clock,
            "converged": relaxed.converged,
            "optimizer_failed": relaxed.optimizer_failed,
            "rotation": relaxed.rotation.tolist(),
            "euler": list(relaxed.euler_angles),
            "direction": list(relaxed.direction),
            "normal": (relaxed.rotation @ np.array([0.0, 0.0, 1.0])).tolist(),
            "free_energies": relaxed.free_energies.tolist(),
            "torques": [np.asarray(s.torque).tolist() for s in relaxed.steps],
            "curvature": None if relaxed.curvature is None
            else relaxed.curvature.tolist(),
            "curvature_eigenvalues": None if relaxed.curvature is None
            else relaxed.curvature_eigenvalues.tolist(),
        }
        print(json.dumps(record["relax"]), flush=True)

    record["peak_gib"] = peak_gib()
    Path(arguments.out).write_text(json.dumps(record, indent=2))
    print("peak GiB", record["peak_gib"], flush=True)


if __name__ == "__main__":
    main()
