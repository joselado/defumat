"""GPU.md Phase 5 -- the response path: does the property *fit*, and does it match?

Everything `phase0.py` measures is the ground state. This is the other half, and
`GPU.md` §5 says why it is a phase of its own rather than a line in Phase 1:
``CLAUDE.md``'s "Why JAX" lists autodiff response as reason **one** and the GPU
as reason two, so a GPU roadmap that stops at the SCF has scoped out P24 through
P28, P35 and P36 -- the dielectric tensor, the Born charges, the phonons, the
elastic constants, the Raman tensors.

And the first question there is **feasibility, not speed**:

> the reverse-mode tape through the radial transforms reaches **11 GB on
> eight-atom ultrasoft silicon**, the largest single allocation anywhere in this
> code, and it is the reverse pass alone. On a card with a fraction of this
> workstation's memory that decides whether the calculation exists, not how fast
> it is.

So this script's headline is a **working set**, per property, beside the
parameters it should scale with. It runs one property per process on purpose:
peak RSS is a high-water mark, so two properties in one process report the
larger of the two twice.

    python3 tools/gpu/phase5.py si2-nc-stress --property stress
    python3 tools/gpu/phase5.py si-epsilon --property dielectric --json out.json
    python3 tools/gpu/phase5.py alas-raman --property all --json-dir records/

It is the *same* script on both platforms, for `phase0.py`'s reason -- §2.3's
metric is GPU pypresso against CPU pypresso on the same input and the same code,
so the two sides must not be two scripts. On a CPU it answers `GPU.md` §4 item 3
("Phase 5's tape measurement, per property -- the number that says whether the
response path fits on a card at all, and it needs no card to obtain"); on a GPU
it is the phase itself.

**The dials are not required here and that is the difference from `phase0.py`.**
Since 2026-08-26 `batching.py` defaults them per platform, and the response path
inherits that for free because everything that walks the k axis goes through
``map_k``/``sum_k``. Passing them is still allowed, and a *measurement* should,
so that the record says which setting produced it.

**What the mode column means, because it is the whole of the memory question.**

* ``reverse`` -- one ``jax.grad`` pass over the setup. Everything the forward
  pass computed is live at once; this is the 11 GB case.
* ``forward`` -- a ``jvp``, which carries a tangent alongside the primal and
  tapes nothing.
* ``forward-over-reverse`` -- a ``jvp`` *of* a gradient, which is what P25's
  dynamical matrix and P35's Raman tensors are. It tapes the inner reverse pass,
  so it is **not** free the way a plain ``jvp`` is, and how much it costs is
  exactly what this measures rather than assumes.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The case lookup, the provenance block, the dial handling and the device
# allocator reading are phase0's; a second copy of them would be a second thing
# to keep true.
from phase0 import (  # noqa: E402
    PSEUDO_DIR,
    _bits,
    _dial,
    _grid_of,
    conv_thr,
    memory_stats,
    provenance,
    resolve_case,
)


# ------------------------------------------------------------------- memory

def _host_peak_gb() -> float:
    """Peak RSS of this process, in GB -- the high-water mark, not the current.

    This is the measurement `PERFORMANCE.md`'s "What a stress costs" section is
    made of ("Peak RSS, same runs"), so the numbers here are comparable with the
    11 GB already recorded there rather than being a new scale.
    """
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform == "darwin" else 1024   # macOS bytes, Linux KiB
    return maxrss * scale / 2**30


def _snapshot() -> dict:
    """Host and device peaks together.

    Both are high-water marks and neither can be reset, which is why the stage
    deltas below are *lower bounds* on a stage that peaked under an earlier one
    -- and why one property per process is the discipline rather than a
    convenience.
    """
    device = memory_stats()
    return {
        "host_peak_gb": _host_peak_gb(),
        "device_peak_gb": device.get("peak_gb") if "Device" in device["source"] else None,
        "device_source": device["source"],
    }


# ---------------------------------------------------------------- properties

@dataclass(frozen=True)
class Property:
    """One response property: how to compute it, and what its derivative is."""

    mode: str
    what: str
    run: Callable[["Context"], tuple[dict, object]]
    refuses: str = ""


@dataclass
class Context:
    """The converged state every property below starts from."""

    system: object
    pseudos: tuple
    calculation: object
    result: object
    threshold: float
    verbose: bool = False


def _forces(context: Context):
    from pypresso.forces import compute_forces

    forces = compute_forces(context.calculation, context.result)
    return {"max_force_ry_bohr": float(forces.max_force),
            "checksum": _bits(forces.forces)}, forces


def _stress(context: Context):
    from pypresso.stress import compute_stress

    stress = compute_stress(context.calculation, context.result)
    tensor = stress.tensor
    return {"pressure_kbar": float(stress.pressure_kbar),
            "checksum": _bits(tensor)}, stress


def _dielectric(context: Context, born: bool):
    from pypresso.response import dielectric_tensor

    result = context.result
    response = dielectric_tensor(
        context.calculation, result.wavefunctions, result.eigenvalues,
        result.density, result.becsum,
        born_charges=born, threshold=1.0e-12, verbose=context.verbose,
    )
    summary = {"epsilon_xx": float(response.epsilon[0, 0]),
               "checksum": _bits(response.epsilon)}
    if born and response.born_charges is not None:
        summary["born_xx"] = float(response.born_charges[0][0, 0])
    return summary, response


def _phonon(context: Context):
    import jax.numpy as jnp
    from pypresso.response import dynamical_matrix
    from pypresso.response.electrostriction import refined_states

    result = context.result
    eigenvalues, psi = refined_states(context.calculation, result)
    phonons = dynamical_matrix(
        context.calculation, psi, eigenvalues, jnp.asarray(result.density),
        verbose=context.verbose,
    )
    frequencies = phonons.frequencies
    return {"max_frequency_cm1": float(max(frequencies)),
            "checksum": _bits(phonons.matrix)}, phonons


def _raman(context: Context):
    from pypresso.response import raman_tensors

    raman = raman_tensors(context.calculation, context.result,
                          born_charges=True, verbose=context.verbose)
    return {"epsilon_xx": float(raman.epsilon[0, 0]),
            "checksum": _bits(raman.raman)}, raman


#: Ordered cheapest first, which is also least to most taped.
PROPERTIES: dict[str, Property] = {
    "force": Property(
        mode="reverse",
        what="dE/d(tau) at frozen wavefunctions (P15)",
        run=_forces,
    ),
    "stress": Property(
        mode="reverse",
        what="dE/d(strain) at frozen wavefunctions (P11)",
        run=_stress,
        refuses="the analytic route has terms and no total",
    ),
    "dielectric": Property(
        mode="forward (Sternheimer)",
        what="epsilon_infinity from the field response (P24)",
        run=lambda context: _dielectric(context, born=False),
        refuses="metals need P24c; noncollinear, DFT+U and spirals are refused",
    ),
    "born": Property(
        mode="forward, jvp of the force (P24b)",
        what="Z* = dF/dE as one mixed second derivative",
        run=lambda context: _dielectric(context, born=True),
        refuses="PAW, by name, at 1.3e-3",
    ),
    "phonon": Property(
        mode="forward-over-reverse",
        what="the Gamma dynamical matrix, one jvp of the force (P25)",
        run=_phonon,
        refuses="norm-conserving only",
    ),
    "raman": Property(
        mode="forward-over-reverse",
        what="d(eps)/d(tau), a third derivative (P35)",
        run=_raman,
        refuses="norm-conserving, nspin=1, insulators, unshifted grid",
    ),
}


# -------------------------------------------------------------------- the run

def _parameters(system, calculation, pseudos, result) -> dict:
    """The terms `GPU.md` §2.4 says a working set must be quoted against.

    A GB figure on one cell is not the number that decides the phase; a GB
    figure beside `ngm`, `npwx`, `nk` and `nbnd` is, because it extrapolates.
    """
    smooth_grid, dense_grid = _grid_of(calculation)
    basis = calculation.basis
    eigenvalues = result.eigenvalues
    return {
        "nat": int(system.structure.nat),
        "nk": int(system.kpoints.nk),
        "nbnd": int(eigenvalues.shape[-1]),
        "nspin": int(system.nspin),
        "npol": int(system.npol),
        "ecutwfc": float(system.ecutwfc),
        "npwx": int(basis.npwx),
        "ngm": int(basis.dense.ngm),
        "ngms": int(basis.ngms),
        "doublegrid": bool(basis.doublegrid),
        "fft_grid": smooth_grid,
        "dense_grid": dense_grid,
        "max_kkbeta": max((int(p.kkbeta) for p in pseudos), default=0),
        "ultrasoft": any(bool(getattr(p, "is_ultrasoft", False)) for p in pseudos),
        "paw": any(bool(getattr(p, "is_paw", False)) for p in pseudos),
    }


def run_property(case: Path, name: str, threshold: float, repeats: int,
                 max_iterations: int, verbose: bool) -> dict:
    """Converge the cell, then evaluate one property, measuring both stages."""
    import jax
    from pypresso.io.pwin import read_pw_input
    from pypresso.pseudo import read_upf
    from pypresso.scf.driver import Calculation, run_scf
    from pypresso.system import build_system

    spec = PROPERTIES[name]
    record: dict = {"property": name, "mode": spec.mode, "what": spec.what,
                    "refuses": spec.refuses, "case": case.name}

    system = build_system(read_pw_input(case))
    pseudos = tuple(read_upf(PSEUDO_DIR / s.pseudo_file) for s in system.structure.species)

    start = time.perf_counter()
    calculation = Calculation(system, pseudos)
    jax.block_until_ready(calculation.vltot)
    record["setup_s"] = time.perf_counter() - start
    record["memory_after_setup"] = _snapshot()

    start = time.perf_counter()
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=threshold,
                     max_iterations=max_iterations)
    jax.block_until_ready(result.density)
    record["scf_s"] = time.perf_counter() - start
    record["scf"] = {
        "total_energy": float(result.total_energy),
        "iterations": int(result.iterations),
        "converged": bool(result.converged),
    }
    # One SCF, so this carries whatever compilation the persistent cache did not
    # already hold: it is an upper bound on the per-iteration cost and the
    # ``vs_scf_iteration`` ratio below is correspondingly a lower bound. The
    # phase's headline is the working set, not this ratio -- `PERFORMANCE.md`'s
    # "What a stress costs" is where a warm-against-warm timing lives.
    record["scf_per_iteration_s"] = record["scf_s"] / max(1, int(result.iterations))
    baseline = _snapshot()
    record["memory_after_scf"] = baseline
    record["parameters"] = _parameters(system, calculation, pseudos, result)

    context = Context(system, pseudos, calculation, result, threshold, verbose)

    runs = []
    summary = None
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        try:
            summary, value = spec.run(context)
        # A refusal is part of what this phase documents, so it is reported
        # rather than raised. Both spellings are caught because both are used:
        # `NotImplementedError` for "this dataset/regime is not implemented"
        # and `ValueError` for the guards a response puts on its own inputs
        # (an unconverged first-order solution, a k-grid that cannot carry a
        # symmetrised response). The type is recorded so the two are not
        # confused when the record is read back.
        except (NotImplementedError, ValueError) as refusal:
            record["refused"] = str(refusal)
            record["refused_type"] = type(refusal).__name__
            record["verdict"] = "REFUSED"
            return record
        jax.block_until_ready(jax.tree_util.tree_leaves(value))
        runs.append({"wall_s": time.perf_counter() - start, **summary})
        del value

    peak = _snapshot()
    cold = runs[0]["wall_s"]
    warm = min(run["wall_s"] for run in runs[1:]) if len(runs) > 1 else None
    record.update({
        "runs": runs,
        "summary": summary,
        "cold_s": cold,
        "warm_s": warm,
        "compile_s": (cold - warm) if warm is not None else None,
        "vs_scf_iteration": (warm or cold) / record["scf_per_iteration_s"],
        "memory_after_property": peak,
        "tape": {
            "host_gb_over_scf": peak["host_peak_gb"] - baseline["host_peak_gb"],
            "host_peak_gb": peak["host_peak_gb"],
            "device_gb_over_scf": (
                None if peak["device_peak_gb"] is None
                else peak["device_peak_gb"] - (baseline["device_peak_gb"] or 0.0)
            ),
            "device_peak_gb": peak["device_peak_gb"],
            "note": "peaks are high-water marks: a stage that peaked below an "
                    "earlier one reports a delta of ~0, which is a bound and "
                    "not a measurement of that stage in isolation",
        },
        "determinism": _determinism(runs),
        "verdict": "ok",
    })
    return record


def _determinism(runs: list[dict]) -> dict:
    """The same property twice in one process, compared bit for bit.

    `GPU.md` Phase 0 check 5's question, asked of the response path: the scatter
    into the FFT box is the same one, and a response evaluates it many more
    times than an SCF iteration does.
    """
    checksums = {run.get("checksum") for run in runs}
    if len(runs) < 2:
        return {"verdict": "not run (one evaluation)", "checksums": list(checksums)}
    return {"verdict": "bit-identical" if len(checksums) == 1 else "DIFFERS",
            "checksums": list(checksums)}


# ---------------------------------------------------------------- reporting

def _report(record: dict) -> None:
    if record.get("verdict") == "REFUSED":
        print(f"\n{record['property']}: REFUSED [{record['refused_type']}] "
              f"-- {record['refused']}")
        return
    parameters = record["parameters"]
    print(f"\n{record['case']}: {record['property']} [{record['mode']}]")
    print(f"  {record['what']}")
    print(f"  cell                 {parameters['nat']} atoms, {parameters['nk']} k, "
          f"nbnd {parameters['nbnd']}, npwx {parameters['npwx']}, "
          f"ngm {parameters['ngm']}")
    scf = record["scf"]
    print(f"  scf                  {scf['total_energy']:.10f} Ry in "
          f"{scf['iterations']} iterations, {record['scf_s']:.1f} s "
          f"({record['scf_per_iteration_s'] * 1e3:.0f} ms/it)")
    warm = record["warm_s"]
    print(f"  property cold/warm   {record['cold_s']:.2f} / "
          f"{'-' if warm is None else format(warm, '.2f')} s"
          f"   = {record['vs_scf_iteration']:.1f} SCF iterations (a lower bound: "
          f"the SCF's own run carries its compilation)")
    if record["compile_s"] is not None:
        print(f"  compile              {record['compile_s']:.2f} s (reported, not amortised)")
    tape = record["tape"]
    print(f"  peak host RSS        {tape['host_peak_gb']:.2f} GB "
          f"({tape['host_gb_over_scf']:+.2f} over the SCF)")
    if tape["device_peak_gb"] is not None:
        print(f"  peak device          {tape['device_peak_gb']:.2f} GB "
              f"({tape['device_gb_over_scf']:+.2f} over the SCF)")
    for key, value in (record["summary"] or {}).items():
        if key != "checksum":
            print(f"  {key:<20} {value}")
    print(f"  determinism          {record['determinism']['verdict']}")


# --------------------------------------------------------------------- main

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", help="a name in benchmarks/ or tests/data/qe/, or a path")
    parser.add_argument("--property", required=True,
                        choices=[*PROPERTIES, "all"],
                        help="which response property; 'all' runs each in its own process")
    parser.add_argument("--k-batch", default=None,
                        help="k-points in flight. Optional since the default follows the "
                             "platform, but a measurement should say which it used")
    parser.add_argument("--band-batch", default=None, help="bands transformed at once")
    parser.add_argument("--repeats", type=int, default=1,
                        help="evaluations in one process. 1 by default because these are "
                             "the expensive half; 2 separates compile and checks "
                             "determinism, at the price of a second solve")
    parser.add_argument("--conv-thr", type=float, default=None,
                        help="override the input's own conv_thr. A response wants a tight "
                             "one: the tests use 1e-12")
    parser.add_argument("--max-iterations", type=int, default=200,
                        help="SCF iteration cap -- 200, not run_scf's 100, for the reason "
                             "the ten-site regression tests use it")
    parser.add_argument("--verbose", action="store_true",
                        help="let the response solvers report their own iterations")
    parser.add_argument("--json", type=Path, default=None, help="where to write the record")
    parser.add_argument("--json-dir", type=Path, default=None,
                        help="with --property all: a directory for one record per property")
    arguments = parser.parse_args(argv)

    if arguments.property == "all":
        return _run_each(arguments)

    # Only the dials actually given are put into the environment, so that
    # naming one does not silently pin the other to QE's loop -- ``k=all, b=1``
    # is the worst setting measured anywhere here, and it is exactly the one a
    # forgiving default would produce.
    dials = {}
    for name, value in (("PYPRESSO_K_BATCH", arguments.k_batch),
                        ("PYPRESSO_BAND_BATCH", arguments.band_batch)):
        if value is not None:
            os.environ[name] = _dial(value)
            dials[name.lower().removeprefix("pypresso_")] = os.environ[name]

    case = resolve_case(arguments.case)

    import pypresso  # noqa: F401  -- sets jax_enable_x64 before any array exists
    from pypresso import batching

    resolved = {"k_batch": str(batching.DEFAULT_K_BATCH),
                "band_batch": str(batching.DEFAULT_BAND_BATCH),
                "given": dials or None,
                "source": ("explicit" if len(dials) == 2 else
                           "mixed" if dials else "platform default")}
    record = {"dials": resolved, "provenance": provenance()}
    print(f"# pypresso GPU.md phase 5 -- {case.name}, {arguments.property}")
    print(f"# {record['provenance']['device_kind']} "
          f"({record['provenance']['platform']}), jax {record['provenance']['jax']}")
    print(f"# dials: {record['dials']}")

    threshold = arguments.conv_thr if arguments.conv_thr is not None else conv_thr(case)
    record.update(run_property(case, arguments.property, threshold, arguments.repeats,
                               arguments.max_iterations, arguments.verbose))
    _report(record)

    if arguments.json:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(record, indent=2, default=str))
        print(f"\nwritten: {arguments.json}")
    return 0 if record.get("verdict") == "ok" else 1


def _run_each(arguments) -> int:
    """One subprocess per property, because peak RSS is a high-water mark.

    Two properties in one process report the larger of the two twice, which is
    the one way this measurement fails silently.
    """
    status = 0
    for name in PROPERTIES:
        command = [sys.executable, str(Path(__file__).resolve()), arguments.case,
                   "--property", name, "--repeats", str(arguments.repeats),
                   "--max-iterations", str(arguments.max_iterations)]
        if arguments.k_batch is not None:
            command += ["--k-batch", arguments.k_batch]
        if arguments.band_batch is not None:
            command += ["--band-batch", arguments.band_batch]
        if arguments.conv_thr is not None:
            command += ["--conv-thr", repr(arguments.conv_thr)]
        if arguments.json_dir is not None:
            arguments.json_dir.mkdir(parents=True, exist_ok=True)
            command += ["--json", str(arguments.json_dir / f"{arguments.case}.{name}.json")]
        print(f"\n=== {name} " + "=" * 56, flush=True)
        done = subprocess.run(command, env=os.environ.copy())
        status = status or done.returncode
    return status


if __name__ == "__main__":
    raise SystemExit(main())
