"""What the dielectric response adds to a streamed SCF's device peak, per k-mesh.

``GPU-MEMORY-NEXT.md`` item 2. A streamed SCF (memory mode, ``k_batch`` smaller
than the mesh) keeps its wavefunctions in host memory and one chunk on the card;
the response stack then has to hold its own first-order states, and this
measures how much of the card that costs, mesh by mesh, so the before and after
of streaming the response are the same numbers taken the same way.

One property per process, because ``peak_bytes_in_use`` has no reset: ``scf``
runs the ground state alone, ``epsilon`` adds
``get_dielectric_tensor(born_charges=False)`` and ``born`` the default call with
the Born charges. Each point runs **twice**, in two fresh processes, and the
second is reported: the first warms the kernel cache, and a cache miss costs
more device memory on a card (``CLAUDE.md``, "A memory figure must say what the
cache held"). ``phonon`` adds ``get_phonons(atoms=...)``, the displacements of
the atoms ``--phonon-atoms`` names (the first by default), which is the
dynamical matrix's ``3 nat`` perturbations cut to three; a subset needs a
``nosym`` input. ``phonon_q`` adds ``get_phonons_at_q(q=...)`` at the crystal
wavevector ``--q`` (``(1/2, 0, 0)`` by default), every atom, which needs a
``nosym`` norm-conserving input with no core charge. ``piezo`` adds
``get_piezoelectric_tensor()``, the default differentiated route, with the
ultrasoft mesh refusal and the k-mesh warning lifted so that a coarse mesh can be
measured (``allow_a_coarse_mesh``, ``kmesh_warning``); it needs a nonpolar crystal
and an input the strain derivative admits (``alas-piezo.in``). ``strain``
adds ``get_strain_response()``, the six strains' self-consistent response, and
``elastic`` adds ``get_elastic_constants()`` on top of it, which admits a
norm-conserving dataset without symmetry only.

    python3 tools/gpu/response_memory.py benchmarks/si8-ecut20-nosym-k3.in \\
        --grids 3 4 --stages scf epsilon born --k-batch 1 --json out.json

The input's ``K_POINTS`` card is replaced by an unshifted ``n n n`` grid, which
is what a ``nosym`` directional response admits
(``efield.require_a_symmetrisable_response``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

STAGES = ("scf", "epsilon", "born", "phonon", "phonon_q", "piezo", "strain",
          "elastic")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input")
    parser.add_argument("--pseudo-dir", default="tests/data/pseudo")
    parser.add_argument("--grids", type=int, nargs="+", default=[3])
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    parser.add_argument("--k-batch", default="1",
                        help="a chunk size, 'fit' (sized to the card) or 'default'")
    parser.add_argument("--memory-mode", default="memory")
    parser.add_argument("--wfc-store", default="default",
                        help="'stream' forces the host store on a CPU, where "
                             "memory mode keeps it on the device")
    parser.add_argument("--repeats", type=int, default=2,
                        help="fresh processes per point; the last is reported")
    parser.add_argument("--phonon-atoms", type=int, nargs="+", default=[0],
                        help="the atoms whose displacements the phonon stage solves")
    parser.add_argument("--q", type=float, nargs=3, default=[0.5, 0.0, 0.0],
                        help="the phonon_q stage's wavevector, crystal coordinates")
    parser.add_argument("--json", default=None)
    parser.add_argument("--point", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.point is not None:
        grid, stage = json.loads(args.point)
        print("__POINT__" + json.dumps(_measure(args, grid, stage)), flush=True)
        return 0

    rows = []
    for grid in args.grids:
        for stage in args.stages:
            for repeat in range(args.repeats):
                out = subprocess.run(
                    [sys.executable, __file__, args.input,
                     "--point", json.dumps([grid, stage]),
                     "--pseudo-dir", args.pseudo_dir,
                     "--k-batch", str(args.k_batch),
                     "--memory-mode", args.memory_mode,
                     "--wfc-store", args.wfc_store,
                     "--phonon-atoms", *map(str, args.phonon_atoms),
                     "--q", *map(str, args.q)],
                    capture_output=True, text=True,
                )
                line = [l for l in out.stdout.splitlines() if l.startswith("__POINT__")]
                if not line:
                    print(out.stdout[-3000:], out.stderr[-3000:], file=sys.stderr)
                    raise SystemExit(f"point grid={grid} stage={stage} failed")
            rows.append(json.loads(line[0][len("__POINT__"):]))
            print(json.dumps(rows[-1]), flush=True)
            if args.json:
                with open(args.json, "w") as handle:
                    json.dump(rows, handle, indent=1)
    return 0


def _k_batch(value: str):
    return value if value in ("fit", "default") else int(value)


def _measure(args, grid: int, stage: str) -> dict:
    import warnings

    import jax
    import numpy as np

    from defumat import Calculator
    from defumat.scf.streaming import is_host_store

    text = open(args.input).read()
    text = re.sub(r"K_POINTS.*", f"K_POINTS automatic\n {grid} {grid} {grid} 0 0 0\n",
                  text, flags=re.S)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_text(
            text, args.pseudo_dir, announce=False, memory_mode=args.memory_mode,
            k_batch=_k_batch(args.k_batch), wfc_store=args.wfc_store)
        start = time.perf_counter()
        result = calculator.get_scf()
        scf_seconds = time.perf_counter() - start
        row = {
            "grid": grid, "stage": stage,
            "nk": int(calculator.calculation.system.kpoints.nk),
            "npwx": int(calculator.calculation.basis.planewaves.npwx),
            "nbnd": int(np.asarray(result.eigenvalues).shape[-1]),
            "k_batch": calculator.calculation.k_batch,
            "streamed": bool(is_host_store(result.wavefunctions)),
            "scf_s": round(scf_seconds, 2),
            "energy": float(result.total_energy),
        }
        if stage in ("strain", "elastic"):
            start = time.perf_counter()
            response = calculator.get_strain_response()
            row["response_s"] = round(time.perf_counter() - start, 2)
            row["drho_00_max"] = round(float(np.abs(np.asarray(response.drho[0, 0])).max()), 10)
            row["cg_average"] = round(float(response.average_iterations), 3)
            if stage == "elastic":
                start = time.perf_counter()
                constants = calculator.get_elastic_constants()
                row["elastic_s"] = round(time.perf_counter() - start, 2)
                row["c11_GPa"] = round(float(constants.voigt[0, 0]), 6)
                row["c44_GPa"] = round(float(constants.voigt[3, 3]), 6)
        elif stage == "piezo":
            start = time.perf_counter()
            tensor = calculator.get_piezoelectric_tensor(
                allow_a_coarse_mesh=True, kmesh_warning=False)
            row["response_s"] = round(time.perf_counter() - start, 2)
            row["e14"] = round(float(tensor.e14), 10)
            row["epsilon"] = round(float(tensor.dielectric.isotropic), 9)
        elif stage == "phonon_q":
            start = time.perf_counter()
            phonons = calculator.get_phonons_at_q(q=tuple(args.q))
            row["response_s"] = round(time.perf_counter() - start, 2)
            row["frequencies"] = [round(float(f), 6) for f in phonons.frequencies]
            row["d00_xx"] = round(float(abs(phonons.matrix[0, 0, 0, 0])), 10)
            row["average_iterations"] = round(float(phonons.average_iterations), 3)
        elif stage == "phonon":
            start = time.perf_counter()
            phonons = calculator.get_phonons(atoms=tuple(args.phonon_atoms))
            row["response_s"] = round(time.perf_counter() - start, 2)
            row["frequencies"] = [round(float(f), 6) for f in phonons.frequencies]
            row["d00_xx"] = round(float(phonons.matrix[0, 0, 0, 0]), 10)
            row["average_iterations"] = round(float(phonons.average_iterations), 3)
        elif stage != "scf":
            start = time.perf_counter()
            tensor = calculator.get_dielectric_tensor(born_charges=(stage == "born"))
            row["response_s"] = round(time.perf_counter() - start, 2)
            row["epsilon"] = round(float(tensor.isotropic), 9)
            row["cg_average"] = round(float(tensor.average_iterations), 3)
            if tensor.born_charges is not None:
                row["zstar_0_xx"] = round(float(tensor.born_charges[0, 0, 0]), 9)
    stats = jax.devices()[0].memory_stats() or {}
    row["platform"] = jax.devices()[0].platform
    if "peak_bytes_in_use" in stats:
        row["peak_MB"] = round(stats["peak_bytes_in_use"] / 2**20, 1)
    return row


if __name__ == "__main__":
    sys.exit(main())
