"""One spin-spiral SCF per invocation: ``E(q)`` and ``dE/dq`` for NiI2.

Two quantities per point, and the second is the interesting one. ``E(q)`` is the
SCF total energy; ``dE/dq`` is P21's ``jax.grad`` of that same energy at *frozen*
wavefunctions and a *frozen* plane-wave sphere
(:func:`defumat.forces.spiral.compute_spiral_gradient`), which is the
Hellmann-Feynman argument in the spiral coordinate: at the variational minimum
the wavefunctions' own response contributes nothing, so the derivative of the
energy is the derivative of the two terms that carry ``q`` at all --
``|k +- q/2 + G|^2`` and ``vkb(k +- q/2)``.

That makes the curve reachable **two** ways, and they differ in one specific
thing:

* directly, as the SCF energies, on a plane-wave sphere rebuilt at every point;
  and
* by integrating ``dE/dq`` along the path, at a *frozen* sphere.

So the integrated curve is the smooth one and the direct one steps by the Pulay
error of a finite basis wherever a plane wave crosses the cutoff. What the
gradient route does **not** buy is a k-mesh gain: ``dE/dq`` at the frozen
converged state is the exact derivative of the *same* fixed-mesh ``E(q)``, so it
carries the same Brillouin-zone error the energies do. That was measured on the
hydrogen chain (``SpiralScan.integrated``) after this script was first written,
and it is why the gap between the two curves is read here as a *cutoff*
diagnostic rather than as a k-sampling one.

Their difference is therefore a *measurement* rather than a formality, and it is
the only check either curve has: ``pw.x`` has no spin spiral, so there is no
reference output to compare against.

**One caveat that belongs beside the gradient.** The sphere is frozen while
differentiating, which is exact between the wavevectors where a plane wave
crosses the cutoff and misses the jump at those -- the Pulay error of a finite
basis. So the integrated curve is smooth by construction and the direct one is
not, and where they part company at a fixed cutoff, that is the size of the
error rather than a bug in either.

The scan is an ``sbatch --array`` rather than a loop in one process, and that
costs nothing here: :func:`defumat.workflows.spiral.run_spiral_scan` starts
every point from the same superposition-of-atoms density -- it does *not* warm
start one ``q`` from the last -- so the points are genuinely independent and the
only thing a loop would save is the setup and the compile, which are seconds
against an SCF that is minutes.

``--q-index`` picks the point out of the path, so ``SLURM_ARRAY_TASK_ID`` is the
whole of the parallelism. Each task writes one JSON file, which is the storage
rule's "one pair per case, not per iteration".

**The JSON is written twice and the first write is the point of it.** The SCF is
hours and the gradient is minutes, so the record lands as soon as the SCF is
converged and is *updated* with the gradient afterwards: a gradient that fails
then costs the gradient rather than the energy. The first run of this
calculation lost twenty converged SCFs exactly that way -- every task ran to
completion and died in ``compute_spiral_gradient``, whose backward pass carried
``vkb(k +- q/2)`` and the states for all 81 k-points at once and asked for 133
GiB on a 141 GB card. ``--grad-k-batch`` is the dial that bounds it; a failure
is recorded in ``gradient_error`` rather than raised.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np


#: The hexagonal reciprocal metric, in crystal coordinates. ``b1`` and ``b2``
#: have equal length and meet at 60 degrees for ``ibrav = 4``, so the cartesian
#: length of a crystal-coordinate step is ``sqrt(x G x)`` with this ``G`` --
#: which is what lets the circuit be sampled at a *uniform spacing in reciprocal
#: space* rather than a uniform spacing in fractional coordinates, where the
#: three segments would get densities in the ratio 2 : 1 : 1.73.
HEX_METRIC = np.array([[1.0, 0.5, 0.0],
                       [0.5, 1.0, 0.0],
                       [0.0, 0.0, 0.0]])

#: The high-symmetry points of the hexagonal Brillouin zone, in the same lattice
#: coordinates ``spiral_q`` (Elk's ``vqlss``) is written in.
CORNERS = {"G": (0.0, 0.0, 0.0),
           "K": (1.0 / 3.0, 1.0 / 3.0, 0.0),
           "M": (0.5, 0.0, 0.0)}


def _hex_length(delta) -> float:
    return float(np.sqrt(delta @ HEX_METRIC @ delta))


def q_path(name: str, npoints: int) -> np.ndarray:
    """A circuit of spiral wavevectors in *lattice* coordinates (``vqlss``).

    ``npoints`` is the total number of points on the whole circuit, and they are
    distributed between the segments by cartesian length, so the sampling is
    uniform in reciprocal space. The corners are always hit exactly -- they are
    the points a symmetry statement can be made about, and ``M`` in particular is
    the commensurate antiferromagnet, the one wavevector on the circuit that a
    supercell can also compute.
    """
    letters = list(name)
    if not all(c in CORNERS for c in letters) or len(letters) < 2:
        raise ValueError(f"unknown q path {name!r}: use letters from {sorted(CORNERS)}")

    nodes = [np.array(CORNERS[c], dtype=float) for c in letters]
    lengths = [_hex_length(b - a) for a, b in zip(nodes, nodes[1:])]
    total = sum(lengths)

    # Hand out the npoints - 1 intervals in proportion to length, giving every
    # segment at least one so a short leg cannot vanish, then fix the rounding
    # on the longest.
    counts = [max(1, int(round((npoints - 1) * L / total))) for L in lengths]
    counts[int(np.argmax(lengths))] += (npoints - 1) - sum(counts)

    points = [nodes[0]]
    for a, b, n in zip(nodes, nodes[1:], counts):
        for i in range(1, n + 1):
            points.append(a + (b - a) * (i / n))
    return np.array(points)


def path_abscissa(qs: np.ndarray) -> np.ndarray:
    """Cumulative cartesian distance along the circuit, in units of ``|b1|``.

    This is the variable ``dE/dq`` is integrated against: the circuit is not a
    straight line, so there is no single lattice component to integrate, and the
    chain rule wants the projection of the gradient on the local tangent.
    """
    steps = np.diff(qs, axis=0)
    return np.concatenate([[0.0], np.cumsum([_hex_length(d) for d in steps])])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--pseudo-dir", type=Path, required=True)
    ap.add_argument("--path", default="GKMG",
                    help="a circuit through the hexagonal high-symmetry points, "
                         "as letters: GKMG is Gamma -> K -> M -> Gamma")
    ap.add_argument("--npoints", type=int, default=20)
    ap.add_argument("--q-index", type=int, required=True)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--k-batch", type=int, default=None,
                    help="k-points in flight; unset means the platform default "
                         "(the whole axis on a GPU), which at 81 k and 64 "
                         "spinor bands is not a setting to inherit blindly")
    # No --band-batch: it is not one of Calculator.SHARED_OPTIONS and a
    # Calculator refuses it by name, so a flag here would only fail a job late.
    # The band dial follows the platform, and GPU.md's measurement is that the
    # two axes move together or not at all -- k batched with bands looped is
    # worse than either end.
    ap.add_argument("--conv-thr", type=float, default=1.0e-8)
    ap.add_argument("--max-iterations", type=int, default=200)
    ap.add_argument("--grad-k-batch", type=int, default=None,
                    help="k-points differentiated at once when taking dE/dq; "
                         "unset follows --k-batch. The backward pass carries "
                         "vkb(k +- q/2) and the states for every k-point in "
                         "flight, which is where the first run of this "
                         "calculation ran out of HBM")
    ap.add_argument("--no-gradient", action="store_true",
                    help="skip dE/dq -- for a calibration run whose SCF is "
                         "deliberately stopped short, where the gradient at an "
                         "unconverged state means nothing")
    args = ap.parse_args()

    import jax

    from defumat import Calculator

    # GPU.md Phase 0, check 1: x64 is asserted on the device rather than
    # trusted, because its failure mode is a plausible number and a platform
    # change is exactly when a config assumption stops holding.
    probe = jax.numpy.zeros(1)
    assert probe.dtype == np.float64, f"x64 is off: default dtype is {probe.dtype}"

    qs = q_path(args.path, args.npoints)
    q = qs[args.q_index]

    options = dict(conv_thr=args.conv_thr, max_iterations=args.max_iterations)
    if args.k_batch is not None:
        options["k_batch"] = args.k_batch

    text = args.input.read_text()
    calc = Calculator.from_text(text, args.pseudo_dir, announce=False, **options)
    calc = calc.with_spiral_q(q) if hasattr(calc, "with_spiral_q") else calc

    system = calc.system
    if not np.allclose(system.spiral_q, q):
        # No ``with_spiral_q`` on the facade: rewrite the input's own card, which
        # is the documented way in and keeps the setup a single construction.
        text = _substitute_q(text, q)
        calc = Calculator.from_text(text, args.pseudo_dir, announce=False, **options)
        system = calc.system
    assert np.allclose(system.spiral_q, q), (system.spiral_q, q)

    basis = calc.calculation.basis
    meta = {
        "host": platform.node(),
        "devices": [str(d) for d in jax.devices()],
        "q_index": args.q_index,
        "q_path": args.path,
        "npoints": args.npoints,
        "q": [float(v) for v in q],
        # Where this point sits along the circuit, in units of |b1|. Written
        # here rather than recomputed downstream so that a results directory is
        # self-describing even if the path argument is forgotten.
        "abscissa": float(path_abscissa(qs)[args.q_index]),
        "nk": int(system.kpoints.nk),
        "npwx": int(basis.planewaves.npwx),
        "fft_grid": list(basis.smooth.grid),
        "nelec": float(calc.calculation.nelec),
        "k_batch": args.k_batch,
        "conv_thr": args.conv_thr,
    }
    print(json.dumps(meta, indent=2), flush=True)

    start = time.time()
    result = calc.get_scf()
    elapsed = time.time() - start

    record = dict(meta)
    record.update(
        total_energy=float(result.total_energy),
        converged=bool(result.converged),
        iterations=int(result.iterations),
        seconds=elapsed,
        magnetization_vector=[float(v) for v in (result.magnetization_vector or (0, 0, 0))],
        energy_terms={k: float(v) for k, v in result.energy_terms.items()},
        fermi_energy=float(result.fermi_energy) if result.fermi_energy is not None else None,
        seconds_per_iteration=elapsed / max(int(result.iterations), 1),
        peak_device_bytes=_peak_device_bytes(jax),
        gradient=None,
        gradient_cartesian=None,
        gradient_energy=None,
        gradient_seconds=None,
        gradient_error=None,
        frozen_energy_residual=None,
    )

    # The energy is on disk before the gradient is attempted. It cost hours and
    # the gradient costs minutes, so the ordering decides which of the two a
    # failure takes with it -- and on the first run of this calculation it took
    # both.
    _write(args.json, record)

    # ``dE/dq`` at the converged state. The functional entry point rather than a
    # facade method: ``Calculator`` carries ``get_spiral_scan`` and
    # ``get_spiral_relaxation`` but no ``get_spiral_gradient``, so this is the
    # documented way in.
    if not args.no_gradient:
        from defumat.forces.spiral import compute_spiral_gradient

        t_grad = time.time()
        try:
            grad = compute_spiral_gradient(
                calc.calculation, result,
                # Unset follows --k-batch; both unset follows the platform,
                # which on a GPU is the whole axis at once and is exactly the
                # thing that ran out of HBM. The sbatch always sets --k-batch.
                k_batch=(args.grad_k_batch if args.grad_k_batch is not None
                         else args.k_batch if args.k_batch is not None
                         else "default"),
            )
        except Exception as error:  # noqa: BLE001 -- the energy must survive it
            record["gradient_error"] = f"{type(error).__name__}: {error}"
            record["gradient_seconds"] = time.time() - t_grad
            print("dE/dq failed:", record["gradient_error"], flush=True)
        else:
            record.update(
                # dE/dq in lattice coordinates (the units spiral_q is written
                # in) and in units of 2 pi / alat, plus the energy the gradient
                # was taken at -- recomputed from the frozen state, so
                # |frozen - scf| is the identity check on the functional being
                # differentiated.
                gradient=[float(v) for v in grad.gradient],
                gradient_cartesian=[float(v) for v in grad.gradient_cartesian],
                gradient_energy=float(grad.total_energy),
                gradient_seconds=time.time() - t_grad,
                frozen_energy_residual=(float(grad.total_energy)
                                        - float(result.total_energy)),
            )
        record["peak_device_bytes"] = _peak_device_bytes(jax)
        _write(args.json, record)

    print(json.dumps({k: record[k] for k in
                      ("q", "total_energy", "converged", "iterations",
                       "seconds", "seconds_per_iteration", "peak_device_bytes",
                       "magnetization_vector", "gradient", "gradient_seconds",
                       "gradient_error", "frozen_energy_residual")},
                     indent=2), flush=True)
    if not result.converged:
        raise SystemExit("the SCF did not converge: the record is on disk and "
                         "its energy is not a point of E(q)")


def _write(path: Path, record: dict) -> None:
    """One JSON file, written whole each time it is updated."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))


def _peak_device_bytes(jax):
    """Peak HBM from ``memory_stats``, not from the allocator's pool size.

    XLA preallocates a large fraction of the card on first use, so reading the
    pool reports the card and not the working set -- a wrong number rather than
    a failed check, which is the worse of the two outcomes (GPU.md Phase 0,
    check 4). ``peak_bytes_in_use`` is the working set.
    """
    try:
        device = jax.devices()[0]
        stats = device.memory_stats() or {}
    except Exception:
        return None
    return stats.get("peak_bytes_in_use")


def _substitute_q(text: str, q) -> str:
    lines = []
    for line in text.splitlines():
        if "spiral_q(1)" in line and not line.lstrip().startswith("!"):
            lines.append("    spiral_q(1) = %.12f, spiral_q(2) = %.12f, "
                         "spiral_q(3) = %.12f" % tuple(q))
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
