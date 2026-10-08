"""chi^(3)(-3w; w, w, w) of silicon by the real-time route: one platform's time, and its check.

The card-against-CPU measurement of ``GPU.md`` §2.3 for the third harmonic:
the same input, the same commit and the same number of steps on both sides,
the CPU side pinned to the core count its job states, compilation its own
line. Two-atom silicon (``tests/data/qe/si2-symmetric.in``, ``Si.pz-vbc``) at
``ecutwfc`` 12 Ry by default, the ground state on its own 4x4x4 mesh, then
``Calculator.get_third_harmonic`` at 1.55 eV along ``[100]`` only (``chi_xxxx``)
on the field's wedge of ``GRID^3``.

The step is pinned (``--steps``, 450 a period by default, ``dt = 0.245``
Hartree a.u., 0.87 of the propagator's bound centred on the occupied bands,
which is ``dt <= 0.282`` at 12 Ry: 360 a period is past it and refused), so
both platforms take the same steps by construction; the one count left that
depends on the data is the fixed-density Davidson on the wedge, which is
timed apart from the propagation.

**Never a first call.** A one-period warm-up on the same grid compiles every
program the timed run uses, since the time grid is padded to whole blocks and
the field reaches the kept block as an argument; the timed call then runs
under ``jax_log_compiles`` with a counter on the ``jax`` logger, which prints
each program it compiled and its seconds. At ``9efd6bc`` the count is not zero:
``RadialTable.radial`` (``realtime/radial.py``), a ``fori_loop`` over a
closure, is called outside any ``jit`` while each k-chunk is set up, so two
``jit(scan)`` programs are traced and loaded again per chunk and per call, 5.5
ms each and about 28 mappings each on D22 (18 points on a CPU: 36 programs in
a 101 s call). That is the eager-closure trap at a size that does not move a
time; it is counted rather than hidden.

**The check** (``--hierarchy``): the frequency-domain hierarchy
(``get_nonlinear_spectrum``) at the same frequency, broadening, grid and
direction, called twice and timed on the second. It shares the Hamiltonian
with the propagation and nothing else, and the two agree to the propagation's
start transient: 8e-3 in ``chi(3w)`` at ``eta_t = 6`` and 5e-5 at 12 on silicon
(``PLAN.md`` P135).

    python3 tools/realtime/thg_timing.py OUT.json GRID ETA_T [--eta 0.2] [--ecut 12]
        [--steps 450] [--hierarchy] [--input benchmarks/si8-1k.in] [--grid3 2 2 1]

``--input`` replaces the two-atom cell by another norm-conserving input (the
supercells of ``benchmarks/``), whose own ``K_POINTS`` carry the ground state;
``--grid3`` gives the propagation's mesh axis by axis where ``GRID`` gives a
cube, for the 1x1xN stacks. Each case appends one record to ``OUT.json``,
written once the propagation is timed and again after the check, so a check
that refuses does not lose the timing.
"""
import argparse
import json
import logging
import math
import os
import re
import resource
import subprocess
import tempfile
import time
from pathlib import Path

import jax
import numpy as np

from defumat import Calculator
from defumat.workflows import realtime as workflow

FREQUENCY = 1.55

parser = argparse.ArgumentParser()
parser.add_argument("out", type=Path)
parser.add_argument("grid", type=int)
parser.add_argument("eta_t", type=float)
parser.add_argument("--eta", type=float, default=0.2, help="broadening in eV")
parser.add_argument("--ecut", type=float, default=12.0)
parser.add_argument("--steps", type=int, default=450, help="steps a period")
parser.add_argument("--hierarchy", action="store_true")
parser.add_argument("--input", type=Path, default=None,
                    help="another input in place of tests/data/qe/si2-symmetric.in")
parser.add_argument("--grid3", type=int, nargs=3, default=None,
                    help="the propagation's mesh, axis by axis, in place of GRID^3")
args = parser.parse_args()

repo = Path(__file__).resolve().parents[2]
device = jax.devices()[0]
try:
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
except OSError:
    commit = ""
print(f"{device.platform} {device.device_kind}, jax {jax.__version__}, commit {commit}, "
      f"affinity {len(os.sched_getaffinity(0))} CPUs", flush=True)

source = args.input if args.input is not None else repo / "tests/data/qe/si2-symmetric.in"
mesh = tuple(args.grid3) if args.grid3 is not None else (args.grid,) * 3
text = Path(source).read_text()
text = re.sub(r"ecutwfc\s*=\s*[0-9.dD+-]+", f"ecutwfc = {args.ecut}", text)
path = Path(tempfile.gettempdir()) / f"si2-thg-{args.ecut:g}-{os.getpid()}.in"
path.write_text(text)
calculator = Calculator.from_file(path, pseudo_dir=repo / "tests/data/pseudo", announce=False)
start = time.perf_counter()
scf = calculator.get_scf(conv_thr=1e-10)
scf_s = time.perf_counter() - start
print(f"scf {scf_s:.1f} s (first call), E = {scf.total_energy:.10f} Ry", flush=True)

# the fixed-density solve on the wedge, timed apart; it also hands back the
# calculation, whose k-chunk is the one the propagation walks
marks = {}
original = workflow._occupied_states


def timed(*a, **kw):
    t0 = time.perf_counter()
    out = original(*a, **kw)
    marks["fixed_density_s"] = time.perf_counter() - t0
    calc, states = out[0], out[1]
    marks["k_batch"] = calc.k_batch
    marks["states_shape"] = [int(n) for n in states.shape]
    return out


workflow._occupied_states = timed
options = dict(broadening=args.eta, both_directions=False, steps_per_period=args.steps,
               grid=mesh)

# the warm-up: one period on the same grid, padded to the same block
start = time.perf_counter()
calculator.get_third_harmonic(FREQUENCY, eta_t=1e-3, **options)
warmup_s = time.perf_counter() - start
print(f"warm-up {warmup_s:.1f} s (compilation and one block), "
      f"{len(Path('/proc/self/maps').read_text().splitlines())} mappings", flush=True)


class Counter(logging.Handler):
    """The compilations ``jax_log_compiles`` reports, each with its own seconds."""

    def __init__(self):
        super().__init__()
        self.count = 0
        self.finished = []

    def emit(self, record):
        message = record.getMessage()
        if "Finished XLA compilation" in message or "Compiling" in message:
            self.count += 1
        if "Finished XLA compilation" in message:
            self.finished.append(message[:200])


def counted(call):
    """``call()`` under the compile counter: its result, its seconds, its compilations."""
    counter = Counter()
    logger = logging.getLogger("jax")
    previous = logger.level
    logger.addHandler(counter)
    logger.setLevel(logging.DEBUG)
    jax.config.update("jax_log_compiles", True)
    try:
        t0 = time.perf_counter()
        value = call()
        seconds = time.perf_counter() - t0
    finally:
        jax.config.update("jax_log_compiles", False)
        logger.removeHandler(counter)
        logger.setLevel(previous)
    return value, seconds, (counter.count, counter.finished)


result, total_s, (compiles, compiled) = counted(
    lambda: calculator.get_third_harmonic(FREQUENCY, eta_t=args.eta_t, **options))
orders = result.orders["100"]
points = marks["states_shape"][1]   # (channel, k, band, plane wave)
nsteps = len(orders.times) - 1
propagation_s = total_s - marks["fixed_density_s"]
third, first = complex(result.chi_xxxx_3w), complex(result.chi_xxxx_w)
z = orders.shape.omega + 1j * orders.shape.eta
chi1 = 4.0 * math.pi * 2.0 * complex(orders.component(1, 1, axis=0)) / z**2
record = {
    "platform": device.platform, "device": device.device_kind, "jax": jax.__version__,
    "commit": commit, "host": os.uname().nodename, "cpu_model": next(
        (line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
         if line.startswith("model name")), ""),
    "affinity_cpus": len(os.sched_getaffinity(0)),
    "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
    "input": Path(source).name, "nat": int(re.search(r"nat\s*=\s*(\d+)", text).group(1)), "mesh": list(mesh),
    "ecut": args.ecut, "grid": args.grid, "eta_eV": args.eta, "eta_t": args.eta_t,
    "steps_per_period": args.steps, "frequency_eV": FREQUENCY,
    "points": points, "states_shape": marks["states_shape"], "k_batch": marks["k_batch"],
    "nsteps": nsteps, "dt": float(orders.dt), "norm_drift": float(orders.norm_drift),
    "scf_first_call_s": scf_s, "warmup_s": warmup_s, "total_s": total_s,
    "fixed_density_s": marks["fixed_density_s"], "propagation_s": propagation_s,
    "ms_per_point_step": 1e3 * propagation_s / (points * nsteps),
    "compiles_in_timed_call": compiles, "compiled_in_timed_call": compiled,
    "chi3_3w": [third.real, third.imag], "abs_chi3_3w": abs(third),
    "chi3_w": [first.real, first.imag], "abs_chi3_w": abs(first),
    "chi1_SI": [chi1.real, chi1.imag],
}
print(f"third harmonic {total_s:.1f} s: fixed density {marks['fixed_density_s']:.1f} s, "
      f"propagation {propagation_s:.1f} s over {points} points and {nsteps} steps "
      f"({record['ms_per_point_step']:.3f} ms a point and step), {compiles} compiles; "
      f"chi(3w) = {third:.6e}", flush=True)
for line in compiled:
    print("  compiled in the timed call:", line, flush=True)


def save():
    """Append (or, on the second call, replace) this case's record in ``OUT.json``."""
    if device.platform != "cpu":
        stats = device.memory_stats() or {}
        record["device_peak_bytes"] = stats.get("peak_bytes_in_use")
    record["host_peak_rss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # a Triton node allows 65,530 mappings (CLAUDE.local.md), and every executable
    # a process loads stays mapped
    record["process_mappings"] = len(Path("/proc/self/maps").read_text().splitlines())
    records = json.loads(args.out.read_text()) if args.out.exists() else []
    if records and records[-1].get("_id") == record["_id"]:
        records[-1] = record
    else:
        records.append(record)
    args.out.write_text(json.dumps(records, indent=1))


record["_id"] = f"{os.getpid()}-{time.time()}"
save()

if args.hierarchy:
    def spectrum():
        return calculator.get_nonlinear_spectrum(
            [FREQUENCY], broadening=args.eta, direction=(1.0, 0.0, 0.0), order=3,
            grid=mesh)

    t0 = time.perf_counter()
    spectrum()
    first_call = time.perf_counter() - t0
    hierarchy, hierarchy_s, (hierarchy_compiles, hierarchy_compiled) = counted(spectrum)
    h3w, hw = (complex(np.asarray(c)[0]) for c in hierarchy.chi3(axis=0))
    record.update({
        "hierarchy_first_call_s": first_call, "hierarchy_s": hierarchy_s,
        "hierarchy_compiles": hierarchy_compiles, "hierarchy_compiled": hierarchy_compiled,
        "hierarchy_iterations": int(np.max(hierarchy.iterations)),
        "hierarchy_residual": float(np.max(hierarchy.residual)),
        "hierarchy_chi3_3w": [h3w.real, h3w.imag], "hierarchy_chi3_w": [hw.real, hw.imag],
        "rel_diff_3w": abs(third - h3w) / abs(h3w), "rel_diff_w": abs(first - hw) / abs(hw),
    })
    print(f"hierarchy {hierarchy_s:.1f} s (first call {first_call:.1f} s, "
          f"{hierarchy_compiles} compiles); chi(3w) = {h3w:.6e}, the propagation "
          f"{record['rel_diff_3w']:.2e} from it, chi(w) {record['rel_diff_w']:.2e}", flush=True)

save()
print(json.dumps(record), flush=True)
path.unlink(missing_ok=True)
print("CASE DONE", flush=True)
