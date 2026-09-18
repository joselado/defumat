"""``e_14`` two ways, so that the piezoelectric tensor's dataset refusal can go
with a number rather than with an argument.

The refusal (``response/piezo.py``) says an ultrasoft or PAW dataset is missing
one term in the strain leg, and names ``response/strain.py`` as refusing the
same datasets for the same reason. That second half is stale -- P41 wired the
``Q_ij(r)`` strain term and ``test_electrostriction.py`` pins it at 4.6e-4 for
ultrasoft and 4.7e-4 for PAW -- but "the stated blocker is gone" is not a
measurement, and the composite has never been measured on an augmented dataset.
This script measures it.

**The two routes share no machinery.** One is this package's: a Sternheimer
field response, then one ``jvp`` of the stress along it, which is
``clamped_ion_piezoelectric``. The other is Elk's (``src/piezoelt.f90``): one
*converged ground state per strain* and a finite difference of the Berry-phase
polarization, which touches no response solver at all. The Voigt convention here
puts the factor of two on the strain (``eps_4 = 2 eps_23``), so with a strain
tensor carrying ``E[1,2] = E[2,1] = s`` the finite difference is
``dP_x / d(2 s)`` and that is ``e_14`` directly.

**The norm-conserving cell is the calibration and is not optional.** Run alone,
a disagreement on the ultrasoft cell says nothing about which side is wrong. The
same pair on a norm-conserving zincblende cell, where the response route is the
validated one, says what the finite-difference harness itself is worth -- the
strain step, the polarization mesh, the branch of the quantum -- and the
ultrasoft comparison is then read against that floor.

Run one case per invocation (the cells cost very different amounts):

    python3 tools/cluster/piezo_measure.py nc   --out results-nc.json
    python3 tools/cluster/piezo_measure.py us   --out results-us.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

#: The two cells, both zincblende (``-43m``), where ``e_14`` is the single
#: independent component. The norm-conserving one is the calibration.
CASES = {
    "nc": {
        "input": "tests/data/qe/alas-raman.in",
        "what": "AlAs, norm-conserving LDA (Al.pz-vbc, As.pz-bhs), ecutwfc 10",
        # Denser than the ultrasoft case's because this cell is cheap and it is
        # the one whose job is to say what the harness itself is worth.
        "nppstr": 11,
        "transverse": (6, 6),
    },
    "us": {
        "input": "tests/data/qe/alas-piezo.in",
        "what": "AlAs, ultrasoft PBE (Al/As.pbe-n-rrkjus_psl.1.0.0), ecutwfc 25",
        "nppstr": 7,
        "transverse": (4, 4),
    },
}


def shear(magnitude: float) -> np.ndarray:
    """``E[1,2] = E[2,1] = s``: the strain whose Voigt form is ``eps_4 = 2 s``."""
    strain = np.zeros((3, 3))
    strain[1, 2] = strain[2, 1] = magnitude
    return strain


def polarization_vector(calculator, strain, nppstr, transverse, conv_thr):
    """``P`` in e/bohr^2, cartesian, at clamped ions in the deformed cell.

    Clamped ions is what the tensor is: the atoms keep their **crystal**
    coordinates, which is what ``with_cell`` does when it is given no positions,
    so they follow the cell affinely and nothing relaxes.
    """
    import jax

    at = np.asarray(calculator.system.cell.at)
    strained = calculator.with_cell(at @ (np.eye(3) + strain).T)
    scf = strained.get_scf(conv_thr=conv_thr)
    cell = strained.system.cell
    # **``Cell.at`` is in bohr**, not in units of ``alat``: ``Cell.volume`` is
    # documented as bohr^3 of exactly this array and ``from_vectors`` takes
    # bohr. Multiplying by ``alat`` here, which is what a QE ``at`` would need,
    # put a factor of 10.575 on the first run of this script and made the finite
    # difference read -6.998 C/m^2 against the response route's -0.764. The
    # norm-conserving cell exists to catch that rather than to be believed.
    vectors = np.asarray(cell.at)  # bohr; row i is a_i
    phases, total = [], np.zeros(3)
    for gdir in range(3):
        polarization = strained.get_polarization(
            gdir=gdir, nppstr=nppstr, transverse=transverse
        )
        phases.append(float(polarization.total_phase))
        total += float(polarization.total_phase) * vectors[gdir]
    out = {
        "polarization": total / float(cell.volume),
        "phases": phases,
        "quantum": float(polarization.quantum),
        "energy": float(scf.total_energy),
        "volume": float(cell.volume),
        "vectors": vectors,
    }
    # **Every strained geometry is a new set of shapes and XLA keeps every
    # executable for the life of the process.** That is the accumulation
    # `CLAUDE.md` names for a test file that sweeps many cells, met inside one
    # script: the first run of this one died in the compiler, "LLVM compilation
    # error: Cannot allocate memory", on a two-atom cell with 120 GB. The
    # results stay; only the compiled code is dropped.
    del strained, scf
    jax.clear_caches()
    return out


def wrapped(difference: float) -> float:
    """A phase difference folded onto ``[-1/2, 1/2)``, since ``P`` is modulo one.

    The two strains are small and the branch should not move, but reading the
    difference through the wrap costs nothing and turns a branch jump from a
    wrong number into a visible one: the raw phases are reported beside it.
    """
    return float(np.angle(np.exp(2j * np.pi * difference)) / (2.0 * np.pi))


def finite_difference(calculator, magnitude, nppstr, transverse, conv_thr):
    """``e_14`` by Elk's route: one ground state per strain, differenced."""
    from defumat.units import BOHR_RADIUS_SI, ELECTRON_SI

    factor = ELECTRON_SI / BOHR_RADIUS_SI**2  # e/bohr^2 -> C/m^2
    plus = polarization_vector(calculator, shear(+magnitude), nppstr,
                               transverse, conv_thr)
    minus = polarization_vector(calculator, shear(-magnitude), nppstr,
                                transverse, conv_thr)

    # Through the phases rather than through the assembled vectors, so that the
    # wrap above applies before anything is multiplied by a lattice vector.
    derivative = np.zeros(3)
    for gdir in range(3):
        delta = wrapped(plus["phases"][gdir] - minus["phases"][gdir])
        derivative += delta * plus["vectors"][gdir]
    derivative = derivative / plus["volume"] / (2.0 * 2.0 * magnitude) * factor

    return {
        "e14": float(derivative[0]),
        "dP_deps4": [float(one) for one in derivative],
        "shear": magnitude,
        "phases_plus": plus["phases"],
        "phases_minus": minus["phases"],
        "quantum": plus["quantum"],
        "energies": [plus["energy"], minus["energy"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("--out", default=None, help="write the numbers here as JSON")
    parser.add_argument("--shears", type=float, nargs="+", default=[0.0025, 0.005],
                        help="E[1,2] = E[2,1]; the Voigt shear is twice each")
    parser.add_argument("--nppstr", type=int, default=None)
    # A *pair*: ``string_mesh`` takes how many strings run along each of the
    # two crystal directions that are not ``gdir``.
    parser.add_argument("--transverse", type=int, nargs=2, default=None)
    parser.add_argument("--conv-thr", type=float, default=1.0e-10)
    parser.add_argument("--pseudo-dir", default=str(ROOT / "tests" / "data" / "pseudo"))
    arguments = parser.parse_args()

    import defumat.response.piezo as piezo
    from defumat.calculator import Calculator

    case = CASES[arguments.case]
    nppstr = arguments.nppstr or case["nppstr"]
    transverse = tuple(arguments.transverse or case["transverse"])
    print(f"=== {arguments.case}: {case['what']}", flush=True)
    print(f"    {case['input']}, strings of {nppstr} over a "
          f"{transverse[0]}x{transverse[1]} transverse mesh", flush=True)

    # The dataset refusal is the thing being measured; every other guard stays.
    # It is patched here rather than removed in the package, so that the
    # repository still refuses until there is a number to lift it with.
    measured = arguments.case == "us"
    if measured:
        piezo.require_a_measured_dataset = lambda calculation: None
        print("    the dataset refusal is lifted for this run only", flush=True)

    calculator = Calculator.from_file(ROOT / case["input"],
                                      pseudo_dir=arguments.pseudo_dir,
                                      announce=False)
    results = {"case": arguments.case, "input": case["input"],
               "what": case["what"], "nppstr": nppstr,
               "transverse": list(transverse)}

    start = time.time()
    scf = calculator.get_scf(conv_thr=arguments.conv_thr)
    results["scf"] = {"energy": float(scf.total_energy),
                      "iterations": int(scf.iterations),
                      "seconds": time.time() - start}
    print(f"    SCF {scf.total_energy:.10f} Ry in {scf.iterations} iterations, "
          f"{results['scf']['seconds']:.1f} s", flush=True)

    start = time.time()
    tensor = calculator.get_piezoelectric_tensor()
    results["response"] = {
        "e14": float(tensor.e14),
        "voigt": np.asarray(tensor.voigt).tolist(),
        "converged": bool(tensor.converged),
        "seconds": time.time() - start,
    }
    print(f"    response route: e_14 = {tensor.e14: .6f} C/m^2 "
          f"(converged {tensor.converged}, {results['response']['seconds']:.1f} s)",
          flush=True)

    results["finite_difference"] = []
    for magnitude in arguments.shears:
        start = time.time()
        one = finite_difference(calculator, magnitude, nppstr,
                                transverse, arguments.conv_thr)
        one["seconds"] = time.time() - start
        results["finite_difference"].append(one)
        gap = one["e14"] - float(tensor.e14)
        print(f"    Berry phase at eps_4 = {2 * magnitude:.4f}: "
              f"e_14 = {one['e14']: .6f} C/m^2, "
              f"difference {gap: .2e} ({abs(gap / tensor.e14) * 100:.2f} per cent), "
              f"{one['seconds']:.1f} s", flush=True)
        print(f"      phases +/-: {np.array2string(np.asarray(one['phases_plus']), precision=6)}"
              f" {np.array2string(np.asarray(one['phases_minus']), precision=6)}"
              f", quantum {one['quantum']:.4f}", flush=True)

    if arguments.out:
        Path(arguments.out).write_text(json.dumps(results, indent=1))
        print(f"    wrote {arguments.out}", flush=True)


if __name__ == "__main__":
    main()
