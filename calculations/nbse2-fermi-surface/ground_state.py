"""The 1H-NbSe2 ground state, and the two numbers Elk's own run can be read against.

One SCF per invocation, then the Fermi-level density of states on the *same*
mesh Elk computes ``fermidos`` on and with the *same* delta -- the Fermi-Dirac
one at ``swidth``, not a Gaussian, which is a visible difference on a 12x12 mesh.

Elk reports ``FERMIDOS.OUT`` in states/Hartree/cell **with the spin degeneracy
already in it** (``occupy.f90:93``, ``fermidos = fermidos*occmax*t0`` and
``occmax = 2`` for a non-magnetic run), so the number to compare against is half
of it in states/Ry/cell.

Usage:  python3 calculations/nbse2-fermi-surface/ground_state.py nbse2.in
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from defumat import Calculator
from defumat.response.nesting import fermi_surface_weights

HERE = Path(__file__).parent
PSEUDO = HERE.parents[1] / "tests" / "data" / "pseudo"
#: Where the converged wavefunctions go. Gitignored: 43 k-points of 24 bands on
#: a 9804-plane-wave sphere is 162 MB, which is a scratch file and not a record.
STATE = HERE / "state"

#: Elk's own numbers for the 12x12x1 run, for the table this prints.
ELK = {
    "fermi_energy_ha": -0.0546337,
    "fermi_energy_ha_24": -0.0549902,
    "fermidos_states_per_ha": 64.393,
    "iterations": 17,
}


def main(name: str) -> None:
    calculator = Calculator.from_file(HERE / name, pseudo_dir=PSEUDO)
    system = calculator.system

    start = time.time()
    scf = calculator.get_scf()
    elapsed = time.time() - start

    # Before anything else: the wavefunctions are what the Fermi surface and the
    # tunnelling weights are built from, and this SCF is half an hour. A
    # checkpoint here is the difference between a post-processing bug costing a
    # minute and costing the run.
    checkpoint = STATE / f"{Path(name).stem}.state"
    STATE.mkdir(exist_ok=True)
    scf.save(checkpoint)

    # D(E_F) on the SCF's own mesh, with the run's own delta: this is exactly
    # what Elk's ``occupy.f90`` accumulates, and ``fermi_surface_weights``
    # carries the spin degeneracy the same way ``occmax`` does.
    weights = fermi_surface_weights(
        scf.eigenvalues_by_spin,
        scf.fermi_energy,
        system.degauss,
        smearing=system.smearing,
        degeneracy=2.0 if system.nspin == 1 else 1.0,
    )
    kweights = np.asarray(system.kpoints.weights)
    # ``KPoints`` weights already sum to the spin degeneracy, so divide it back
    # out: ``fermi_surface_weights`` has applied it once.
    fermi_dos = float(np.dot(kweights, weights) / kweights.sum())

    record = {
        "input": name,
        "grid": list(system.kpoints.grid),
        "nk_irreducible": int(system.kpoints.nk),
        "symmetry_operations": len(system.symmetry_group().rotations),
        "ecutwfc": float(system.ecutwfc),
        "nbnd": int(system.nbnd),
        "converged": bool(scf.converged),
        "iterations": int(scf.iterations),
        "total_energy_ry": float(scf.total_energy),
        "fermi_energy_ry": float(scf.fermi_energy),
        "fermi_dos_states_per_ry": fermi_dos,
        "accuracy_ry": float(scf.accuracy),
        "seconds": elapsed,
        "energy_terms": {k: float(v) for k, v in scf.energy_terms.items()},
        "eigenvalues_ry": np.asarray(scf.eigenvalues).tolist(),
        "kpoints_crystal": np.asarray(
            system.kpoints.crystal(system.cell)).tolist(),
        "kpoint_weights": kweights.tolist(),
    }
    out = HERE / f"{Path(name).stem}.json"
    out.write_text(json.dumps(record, indent=1))

    print(f"{name}: {system.kpoints.grid} grid, {system.kpoints.nk} irreducible "
          f"k-points, {record['symmetry_operations']} symmetry operations")
    print(f"  converged {scf.converged} in {scf.iterations} iterations "
          f"(Elk: {ELK['iterations']}), {elapsed:.0f} s")
    print(f"  E     = {scf.total_energy:.8f} Ry   "
          f"(no Elk counterpart: -8681.03 Ha is all-electron)")
    print(f"  E_F   = {scf.fermi_energy:.8f} Ry")
    print(f"  D(E_F)= {fermi_dos:.3f} states/Ry/cell   "
          f"(Elk: {ELK['fermidos_states_per_ha'] / 2:.3f})")
    print(f"  -> {out}, state in {checkpoint}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "nbse2.in")
