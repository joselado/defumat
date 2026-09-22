"""The spiral against Elk with both codes reporting the same quantity.

P105 located the factor of five between the two codes' ``E(q) - E(0)``: Elk
computes its entropy term only for Fermi-Dirac smearing (``energy.f90:242``,
``IF (stype == 3)``), so with the Gaussian its total is an **internal** energy
where this code's and QE's are **free** energies, and at ``degauss = 0.1 Ry`` the
entropy moves by 112 meV between ``q = 0`` and ``q = 1/4`` -- five times the
difference being compared. Subtracting the smearing term from this side brought
the two to 2.4 and 0.5 per cent.

**This is the version that needs no subtraction.** With ``fermi-dirac`` here and
``stype = 3`` there, both totals carry ``-TS`` and the two can be differenced
directly. That is worth having as a fixture rather than as a correction, because
a correction is a thing to remember and a fixture is not.

**No number carries across from the Gaussian pair.** A different smearing
function is a different free-energy surface, so the moment, the energy and
``E(q) - E(0)`` all move. What this buys is that the two codes are on the same
surface as each other, which the Gaussian pair was not.

The smearing term is reported beside every total, so that the next reader can see
what it is worth on this surface without rerunning anything -- which is the one
number the old fixture would have needed and did not print.

Usage:
    python3 tools/cluster/spiral_fd.py --q 0.25 --out fd-q025.json
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from defumat.calculator import electrons_defaults
from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.scf import Calculation, run_scf
from defumat.system import build_system

CASE = Path("tests/data/qe/h-chain-spiral-elk-fd.in")
PSEUDO = Path("tests/data/pseudo")


def run(q3: float, conv_thr, max_iterations) -> dict:
    text = CASE.read_text()
    marker = "spiral_q(3) = 0.25"
    assert marker in text, "the committed cell no longer states spiral_q(3) = 0.25"
    scratch = Path(tempfile.mkdtemp()) / "fd.in"
    scratch.write_text(text.replace(marker, f"spiral_q(3) = {q3}"))

    pwin = read_pw_input(scratch)
    system = build_system(pwin)
    pseudos = tuple(
        read_upf(PSEUDO / species.pseudo_file)
        for species in system.structure.species
    )
    options = dict(electrons_defaults(pwin))
    options.setdefault("conv_thr", conv_thr)
    options.setdefault("max_iterations", max_iterations)

    started = time.time()
    result = run_scf(
        system, pseudos, calculation=Calculation(system, pseudos),
        verbose=True, **options,
    )
    seconds = time.time() - started
    if not result.converged:
        raise SystemExit(f"q = {q3} did not converge: {result.accuracy}")

    smearing = float(result.energy_terms.get("smearing", 0.0))
    return {
        "q3": float(q3),
        "converged": True,
        "iterations": int(result.iterations),
        "accuracy": float(result.accuracy),
        # The free energy, which is what both codes report under this smearing.
        "total_energy": float(result.total_energy),
        "smearing_term": smearing,
        # And the internal energy beside it, so that the size of the term this
        # fixture exists to neutralise is on the record rather than inferred.
        "internal_energy": float(result.total_energy) - smearing,
        "absolute_magnetization": (
            None if result.absolute_magnetization is None
            else float(result.absolute_magnetization)
        ),
        "magnetization_vector": (
            None if result.magnetization_vector is None
            else [float(x) for x in result.magnetization_vector]
        ),
        "seconds": seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q", type=float, required=True)
    parser.add_argument("--conv-thr", type=float, default=1e-11)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    record = run(arguments.q, arguments.conv_thr, arguments.max_iterations)
    Path(arguments.out).write_text(json.dumps(record, indent=2))
    print(f"q_3               {record['q3']}")
    print(f"converged         {record['converged']} in {record['iterations']}")
    print(f"free energy       {record['total_energy']:.9f} Ry   <- compare with Elk")
    print(f"smearing (-TS)    {record['smearing_term']:+.9f} Ry")
    print(f"internal energy   {record['internal_energy']:.9f} Ry")
    print(f"|m| (int |m| dr)  {record['absolute_magnetization']}")


if __name__ == "__main__":
    main()
