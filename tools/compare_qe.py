"""Run one input through Quantum ESPRESSO and through pypresso, single core.

This is the project's primary performance measurement. Everything else --
component breakdowns, flame graphs, guesses about where time goes -- is
secondary to the one number this prints: how much slower pypresso is than the
Fortran code it reimplements, on the same machine, on the same input, with both
restricted to one core.

Single core on both sides is what makes the comparison mean anything. QE is
built serial here (``configure --disable-parallel --disable-openmp``) and run
with ``OMP_NUM_THREADS=1``; pypresso is run with XLA's intra-op thread pool
pinned to one thread, since otherwise JAX quietly uses every core and the
comparison flatters it by the core count.

Usage::

    python3 tools/compare_qe.py benchmarks/si-1k.in
    python3 tools/compare_qe.py benchmarks/si-1k.in --repeats 3

``pw.x`` is expected at ``quantum_espresso/qe-7.5-ReleasePack/qe-7.5/bin/pw.x``
(override with ``$PW_X``); build it once with::

    cd quantum_espresso/qe-7.5-ReleasePack/qe-7.5
    ./configure --disable-parallel --disable-openmp && make -j pw

The three timings reported for pypresso are separate on purpose:

* **setup** is everything done once per calculation -- basis, projectors, local
  potential, Ewald, symmetry -- and on a first run it is mostly XLA compiling.
* **SCF (cold)** is the loop as a user first meets it, compilation included.
* **SCF (warm)** re-runs the loop with every kernel already compiled. It is the
  honest measure of the *arithmetic*, and the one that predicts what a long run
  or a large system will cost, since compilation is paid once whatever the size.

QE's own report has no equivalent split: there is nothing to compile.
"""

from __future__ import annotations

import os
import re
import sys

# XLA reads its flags when the backend is created, so they must be in the
# environment before JAX is imported anywhere. Re-exec once with them set rather
# than trusting the caller to have exported them.
_SINGLE_CORE = {
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NPROC": "1",
}

if os.environ.get("PYPRESSO_PINNED") != "1":
    os.environ.update(_SINGLE_CORE)
    os.environ["PYPRESSO_PINNED"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])

import argparse  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QE_ROOT = ROOT / "quantum_espresso" / "qe-7.5-ReleasePack" / "qe-7.5"
DEFAULT_PW_X = QE_ROOT / "bin" / "pw.x"
PSEUDO_DIR = ROOT / "tests" / "data" / "pseudo"

_TIMING = re.compile(r"^\s*(\S+)\s*:\s*([\d.]+)s CPU\s*([\d.]+)s WALL")
_CONV_THR = re.compile(r"conv_thr\s*=\s*([\d.eEdD+-]+)")
_ENERGY = re.compile(r"^!\s+total energy\s+=\s+(-?[\d.]+)\s+Ry")


def _conv_thr(input_path: Path, default: float = 1.0e-6) -> float:
    """The input's ``conv_thr``, so both codes stop at the same accuracy."""
    match = _CONV_THR.search(input_path.read_text())
    return float(match.group(1).lower().replace("d", "e")) if match else default


# --------------------------------------------------------------------------- QE


def run_qe(input_path: Path, pw_x: Path, repeats: int) -> dict:
    """Run ``pw.x`` on the input and read its own timing report.

    QE prints the wall time of every routine it instruments, so there is nothing
    to instrument here: the numbers below are QE's, not a stopwatch around it.
    ``total`` is the process wall time all the same, because that is what is
    comparable to a stopwatch around pypresso.
    """
    if not pw_x.exists():
        raise SystemExit(
            f"pw.x not found at {pw_x}.\n"
            "Build it once with:\n"
            f"  cd {QE_ROOT}\n"
            "  ./configure --disable-parallel --disable-openmp && make -j pw"
        )

    best = None
    for _ in range(repeats):
        with tempfile.TemporaryDirectory() as scratch:
            environment = dict(os.environ)
            environment.update(_SINGLE_CORE)
            environment["ESPRESSO_PSEUDO"] = str(PSEUDO_DIR)
            environment["ESPRESSO_TMPDIR"] = scratch

            start = time.perf_counter()
            completed = subprocess.run(
                [str(pw_x)],
                stdin=input_path.open("rb"),
                capture_output=True,
                text=True,
                cwd=scratch,
                env=environment,
            )
            wall = time.perf_counter() - start

        if completed.returncode != 0:
            raise SystemExit(f"pw.x failed:\n{completed.stdout[-3000:]}\n{completed.stderr[-2000:]}")

        record = _parse_qe(completed.stdout)
        record["total"] = wall
        if best is None or record["total"] < best["total"]:
            best = record
    return best


def _parse_qe(output: str) -> dict:
    routines, energy, iterations = {}, None, 0
    for line in output.splitlines():
        match = _TIMING.match(line)
        if match:
            routines[match.group(1)] = float(match.group(3))
        match = _ENERGY.match(line)
        if match:
            energy = float(match.group(1))
        if "iteration #" in line:
            iterations += 1
    return {
        "routines": routines,
        "energy": energy,
        "iterations": iterations,
        "init": routines.get("init_run", float("nan")),
        "scf": routines.get("electrons", float("nan")),
    }


# --------------------------------------------------------------------- pypresso


def run_pypresso(input_path: Path, repeats: int, conv_thr: float) -> dict:
    """Set up and run the same input, timing setup, cold SCF and warm SCF.

    ``conv_thr`` is read from the input's ``&electrons`` namelist so that both
    codes are asked to converge to the same place; pypresso does not yet take it
    from the input file itself.
    """
    from pypresso.io.pwin import read_pw_input
    from pypresso.pseudo import read_upf
    from pypresso.scf.driver import Calculation, run_scf
    from pypresso.system import build_system

    system = build_system(read_pw_input(input_path))
    pseudos = tuple(read_upf(PSEUDO_DIR / s.pseudo_file) for s in system.structure.species)

    start = time.perf_counter()
    calculation = Calculation(system, pseudos)
    jax.block_until_ready(calculation.vltot)
    setup = time.perf_counter() - start

    start = time.perf_counter()
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=conv_thr)
    cold = time.perf_counter() - start

    warm = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        result = run_scf(system, pseudos, calculation=calculation, conv_thr=conv_thr)
        warm = min(warm, time.perf_counter() - start)

    return {
        "setup": setup,
        "scf_cold": cold,
        "scf_warm": warm,
        "iterations": result.iterations,
        "energy": result.total_energy,
        "basis": (calculation.basis.dense.ngm, calculation.basis.npwx,
                  calculation.basis.dense.grid),
        "nk": system.kpoints.nk,
        "nat": system.structure.nat,
        "ecutwfc": system.ecutwfc,
    }


# ----------------------------------------------------------------------- report


def _ratio(ours: float, theirs: float) -> str:
    if not theirs or theirs != theirs:  # zero or NaN
        return "     -"
    return f"{ours / theirs:6.1f}x"


def report(name: str, qe: dict, ours: dict) -> None:
    ngm, npwx, grid = ours["basis"]
    print()
    print(f"{name}: {ours['nat']} atoms, {ours['nk']} k-point(s), ecutwfc {ours['ecutwfc']} Ry")
    print(f"  {ngm} G-vectors, {npwx} plane waves, FFT grid {grid}")
    print(f"  single core: XLA_FLAGS={_SINGLE_CORE['XLA_FLAGS']!r}, OMP_NUM_THREADS=1")
    print()
    print(f"  {'':22s} {'QE 7.5':>10s} {'pypresso':>10s} {'ratio':>8s}")
    print(f"  {'-' * 52}")
    print(f"  {'setup / init_run':22s} {qe['init']:9.3f}s {ours['setup']:9.3f}s "
          f"{_ratio(ours['setup'], qe['init'])}")
    print(f"  {'SCF, cold':22s} {qe['scf']:9.3f}s {ours['scf_cold']:9.3f}s "
          f"{_ratio(ours['scf_cold'], qe['scf'])}")
    print(f"  {'SCF, warm':22s} {qe['scf']:9.3f}s {ours['scf_warm']:9.3f}s "
          f"{_ratio(ours['scf_warm'], qe['scf'])}")

    per_qe = qe["scf"] / max(qe["iterations"], 1)
    per_ours = ours["scf_warm"] / max(ours["iterations"], 1)
    print(f"  {'per SCF iteration':22s} {per_qe:9.3f}s {per_ours:9.3f}s "
          f"{_ratio(per_ours, per_qe)}")
    print(f"  {'  (iterations)':22s} {qe['iterations']:9d}  {ours['iterations']:9d}")
    print(f"  {'total process wall':22s} {qe['total']:9.3f}s {'-':>10s}")
    print()
    print(f"  total energy  QE {qe['energy']:.8f} Ry   pypresso {ours['energy']:.8f} Ry"
          f"   delta {abs(qe['energy'] - ours['energy']):.2e} Ry")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="a pw.x input file")
    parser.add_argument("--repeats", type=int, default=1,
                        help="repeat each timed run and keep the fastest")
    parser.add_argument("--pw-x", type=Path, default=Path(os.environ.get("PW_X", DEFAULT_PW_X)))
    arguments = parser.parse_args()

    if shutil.which("gfortran") is None and not arguments.pw_x.exists():
        raise SystemExit("neither pw.x nor a Fortran compiler is available")

    qe = run_qe(arguments.input, arguments.pw_x, arguments.repeats)
    ours = run_pypresso(arguments.input, arguments.repeats, _conv_thr(arguments.input))
    report(arguments.input.name, qe, ours)


if __name__ == "__main__":
    main()
