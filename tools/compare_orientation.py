"""The orientation torque and the in-loop rotation against ``pw.x``, one core each.

``PLAN.md`` P122. Two pairs, both on the one-atom tetragonal cobalt cell:

* **Route A's one-shot** against ``pw.x``'s ``lforcet`` NSCF: the collinear
  density converged without the coupling, turned 45 degrees off ``c`` and
  diagonalised once with it. That diagonalisation is the like-for-like unit; the
  three-component torque on top of it has no counterpart in ``pw.x`` and is timed
  on its own.
* **Route C** against ``pw.x``'s SCF with the coupling from the same seed, 45
  degrees off ``c``: the same work, one self-consistent run, with the iteration
  count and the angle each run ends at beside the time, since a plain run that
  stops without reaching the easy axis is the point of the comparison.

The pinning is ``tools/compare_qe.py``'s: the CPU affinity mask is set before JAX
is imported and inherited by the ``pw.x`` subprocess, because XLA sizes its thread
pool from that mask and ignores ``OMP_NUM_THREADS``. Every defumat timing is the
**second** call of the same thing, so no compilation is in it (``CLAUDE.md``,
never time a first call). ``pw.x``'s times are its own ``PWSCF ... WALL`` line.

    python3 tools/compare_orientation.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[0]})

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "tests/data/qe"
PSEUDO = ROOT / "tests/data/pseudo"
PW_X = Path(os.environ.get(
    "PW_X", ROOT / "quantum_espresso/qe-7.5-ReleasePack/qe-7.5/bin/pw.x"))


def pw_x(text: str, workdir: Path) -> tuple[float, str]:
    """Run ``pw.x`` on ``text`` in ``workdir``; its own wall time and its output."""
    environment = dict(os.environ, ESPRESSO_PSEUDO=str(PSEUDO),
                       ESPRESSO_TMPDIR=str(workdir))
    completed = subprocess.run([str(PW_X)], input=text, capture_output=True,
                               text=True, cwd=workdir, env=environment, check=True)
    wall = re.findall(r"PWSCF\s*:.*?([\d.]+)s WALL", completed.stdout)
    if not wall:
        minutes = re.findall(r"PWSCF\s*:.*?(\d+)m\s*([\d.]+)s WALL", completed.stdout)
        seconds = 60.0 * float(minutes[-1][0]) + float(minutes[-1][1])
    else:
        seconds = float(wall[-1])
    return seconds, completed.stdout


def pw_angle(output: str) -> float:
    """Degrees from ``c`` of ``pw.x``'s last printed total magnetization."""
    import numpy as np

    vectors = re.findall(
        r"total magnetization\s*=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", output)
    m = np.array([float(v) for v in vectors[-1]])
    return float(np.degrees(np.arccos(abs(m[2]) / np.linalg.norm(m))))


def twice(function):
    """Call once to compile, time the second call."""
    function()
    start = time.perf_counter()
    result = function()
    return time.perf_counter() - start, result


def main() -> None:
    import jax.numpy as jnp
    import numpy as np

    from defumat import Calculator
    from defumat.forces.torque import orientation_torque, rotate_texture
    from defumat.workflows.anisotropy import (
        _reference_axis,
        _reference_texture,
        _with_rotation,
        rotation_from_euler,
    )
    from defumat.workflows.nscf import fixed_density_states

    print("cores", len(os.sched_getaffinity(0)))
    scalar_text = (CASES / "co-tetragonal-anisotropy-sr.in").read_text()
    spinor_text = (CASES / "co-tetragonal-anisotropy-soc.in").read_text().replace(
        "starting_magnetization(1) = 0.6,",
        "starting_magnetization(1) = 0.6, angle1(1) = 45.0, angle2(1) = 0.0,")
    assert "angle1(1) = 45.0" in spinor_text

    # Route A: pw.x's SCF without the coupling, then its lforcet one-shot.
    with tempfile.TemporaryDirectory() as directory:
        workdir = Path(directory)
        scf_seconds, _ = pw_x(scalar_text, workdir)
        nscf_seconds, _ = pw_x(spinor_text, workdir)
    print(f"pw.x   scalar SCF {scf_seconds:.2f} s, lforcet one-shot {nscf_seconds:.2f} s")

    scalar = Calculator.from_text(scalar_text, PSEUDO, announce=False)
    spinor = Calculator.from_text(
        (CASES / "co-tetragonal-anisotropy-soc.in").read_text(), PSEUDO, announce=False)
    density = scalar.get_scf().density
    rotation = rotation_from_euler(0.0, np.pi / 4, 0.0)
    texture = _reference_texture(density, _reference_axis(spinor.system))
    turned = _with_rotation(spinor.system, rotation)

    def one_shot():
        calculation, _, eigenvalues, states = fixed_density_states(
            turned, spinor.pseudos, rotate_texture(texture, rotation), conv_thr=1.0e-10)
        weights, _ = calculation.occupations(jnp.asarray(eigenvalues))
        return calculation, states, weights

    shot_seconds, (calculation, states, weights) = twice(one_shot)
    grad_seconds, torque = twice(lambda: orientation_torque(
        calculation, states, weights, texture, rotation))
    print(f"defumat one-shot {shot_seconds:.2f} s, three-component torque "
          f"{grad_seconds:.2f} s, torque {torque}")

    # Route C: the same seed, 45 degrees off c, self-consistent with the coupling.
    oblique = (CASES / "co-tetragonal-relaxed-mae.in").read_text().replace(
        "angle1(1) = 0.0", "angle1(1) = 45.0").replace(
        "   mixing_beta = 0.3\n", "   mixing_beta = 0.3\n   electron_maxstep = 100\n")
    assert "angle1(1) = 45.0" in oblique and "electron_maxstep" in oblique
    with tempfile.TemporaryDirectory() as directory:
        pw_seconds, output = pw_x(oblique, Path(directory))
    iterations = re.findall(r"iteration #\s*(\d+)", output)
    converged = "convergence has been achieved" in output
    print(f"pw.x   SCF {pw_seconds:.2f} s, {iterations[-1]} iterations, converged "
          f"{converged}, ends {pw_angle(output):.2f} deg from c")

    def angle(result):
        m = np.asarray(result.magnetization_vector)
        return float(np.degrees(np.arccos(abs(m[2]) / np.linalg.norm(m))))

    for label, options in (("plain", {}), ("rotate_moments", {"rotate_moments": True})):
        seconds, result = twice(lambda: Calculator.from_text(
            oblique, PSEUDO, announce=False).get_scf(**options))
        print(f"defumat {label:15s} {seconds:.2f} s, {result.iterations} iterations, "
              f"converged {result.converged}, ends {angle(result):.4f} deg from c")


if __name__ == "__main__":
    sys.exit(main())
