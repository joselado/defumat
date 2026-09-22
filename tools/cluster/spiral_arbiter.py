"""The collinear arbiter for the spin spiral's factor of five.

defumat and Elk agree on the spiral's moments to 4 per cent at every wavevector
and disagree about ``E(q) - E(0)`` by about five, and every named candidate for
that is now dead: the k-grid is worth nothing, this code's basis a tenth, the
held field was measured under a conversion 274 times too large, and Elk's own
basis moves the answer by about one per cent between ``rgkmax`` 7 and 8.

**What was never separated is the machinery from the physics.** Both codes reach
``E(q)`` through their own implementation of the generalized Bloch theorem, so a
disagreement can be either, and no amount of convergence on either side tells
them apart.

This does. ``q = b3/2`` of the unit cell **is** the collinear antiferromagnet of
the cell doubled along ``z`` -- a statement about the two states, not about
either code -- so

    E(AFM) - E(FM) in the doubled cell = 2 [ E(q = 1/2) - E(q = 0) ],

and both sides of that can be computed with ordinary collinear machinery that
has no spiral in it anywhere. This code's own spiral already satisfies the
identity to **1e-10 Ry** (``tests/regression/test_spin_spirals.py``, whose
reference has ``nspin = 2``), so this run is not defumat checking itself: it is
the number Elk's *collinear* path can be asked for, on a cell where Elk's spiral
is not involved.

**The two outcomes, written down before the run.**

* Elk's collinear difference lands near **-113 meV**, which is twice this code's
  converged ``E(1/2) - E(0)``. Then the two codes agree about the physics of this
  chain and Elk's *spiral* is the outlier -- which its own manual leaves room for,
  calling ``spinsprl`` an "Experimental feature".
* It lands near **-520 meV**, which is twice Elk's own spiral number. Then both
  spiral implementations are exonerated and the codes genuinely disagree about
  this chain's magnetism, which points at the pseudopotential or the
  functional instead and is a different investigation.

Anything in between is a third answer and is worth having for that reason.

Usage:
    python3 tools/cluster/spiral_arbiter.py --arrangement afm --out afm.json
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
from defumat.system import build_system

CASES = Path("tests/data/qe")
PSEUDO = Path("tests/data/pseudo")

#: Rydberg to millielectronvolt.
RY_MEV = 13605.693122990


def run(arrangement: str, conv_thr, max_iterations) -> dict:
    pwin = read_pw_input(CASES / f"h-chain-doubled-{arrangement}.in")
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
        raise SystemExit(
            f"the {arrangement} leg did not converge: {result.accuracy}"
        )

    charges, moments = None, None
    if result.site_charges is not None:
        charges = np.asarray(result.site_charges).tolist()
        moments = np.asarray(result.site_moments).tolist()

    return {
        "arrangement": arrangement,
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "accuracy": float(result.accuracy),
        "total_energy": float(result.total_energy),
        # The Ewald sum is identical between the two arrangements -- same cell,
        # same positions -- so unlike the spiral identity this difference does
        # not have to have it removed, and it is reported so that can be checked
        # rather than assumed.
        "ewald": float(result.energy_terms["ewald"]),
        "magnetization": (None if result.magnetization is None
                          else float(result.magnetization)),
        "absolute_magnetization": (None if result.absolute_magnetization is None
                                   else float(result.absolute_magnetization)),
        "site_charges": charges,
        "site_moments": moments,
        "seconds": seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrangement", required=True, choices=("fm", "afm"))
    parser.add_argument("--conv-thr", type=float, default=1e-11)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    record = run(arguments.arrangement, arguments.conv_thr,
                 arguments.max_iterations)
    Path(arguments.out).write_text(json.dumps(record, indent=2))

    print(f"arrangement      {record['arrangement']}")
    print(f"converged        {record['converged']} in {record['iterations']}"
          f" at {record['accuracy']:.2e}")
    print(f"total energy     {record['total_energy']:.9f} Ry")
    print(f"ewald            {record['ewald']:.9f} Ry")
    print(f"|m| absolute     {record['absolute_magnetization']}")
    if record["site_moments"] is not None:
        print(f"site moments     {record['site_moments']}")
    print(f"seconds          {record['seconds']:.1f}")
    print("\nE(AFM) - E(FM) needs both legs; it is twice E(q=1/2) - E(q=0).")


if __name__ == "__main__":
    main()
