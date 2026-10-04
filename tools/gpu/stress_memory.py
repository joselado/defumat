"""What the stress's reverse pass costs, before a ground state has been paid for.

The stress companion to :mod:`tools.gpu.force_memory`, built the same way:
``compute_stress``'s own compiled gradient (``stress/autodiff.py``,
``_energy_gradient``) is lowered against a
:class:`~defumat.forces.energy.FrozenState` of :class:`jax.ShapeDtypeStruct`
leaves and compiled, and ``memory_analysis()`` is read off the executable. No
wavefunction is allocated and no SCF is run, so a cell whose stress does not fit
on the card can still be sized on it -- which is the case it was written for:
``bn-ldau-noncol.in`` converges its SCF at about 0.9 GB on the GTX 1060 and
then dies on a single 5.57 GiB request from this executable.

    python3 tools/gpu/stress_memory.py tests/data/qe/bn-ldau-noncol.in \
        --pseudo-dir tests/data/pseudo

Each point runs in its own process, like the force probe, because a process that
has compiled one executable is not a clean place to size the next. The figure is
the compiler's, so it is a claim about the executable and not about the
allocator: whether the card fragments beside it is a separate question, answered
by running the stress.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--pseudo-dir", default=None)
    parser.add_argument("--nbnd", type=int, default=None)
    parser.add_argument("--kpoints", type=int, nargs=3, default=None,
                        help="replace the input's Monkhorst-Pack grid")
    parser.add_argument("--point", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    os.environ.setdefault("DEFUMAT_CACHE_DIR", "off")

    if args.point:
        print("__POINT__" + json.dumps(_measure(args)), flush=True)
        return 0

    import subprocess

    out = subprocess.run(
        [sys.executable, __file__, *sys.argv[1:], "--point"],
        capture_output=True, text=True,
    )
    line = [l for l in out.stdout.splitlines() if l.startswith("__POINT__")]
    if not line:
        print(out.stdout[-3000:], out.stderr[-3000:], file=sys.stderr)
        raise SystemExit("the stress executable did not compile")
    print(json.dumps(json.loads(line[0][len("__POINT__"):])), flush=True)
    return 0


def _measure(args):
    import jax

    from defumat import Calculator
    from defumat.forces.energy import hoisted
    from defumat.stress.autodiff import _energy_gradient, _zero

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from force_memory import _frozen_state_shapes

    kwargs = {} if args.pseudo_dir is None else {"pseudo_dir": args.pseudo_dir}
    options = {} if args.nbnd is None else {"nbnd": args.nbnd}
    calculator = Calculator.from_file(args.input, **kwargs, **options)
    calculation = calculator.calculation
    if args.kpoints is not None:
        # ``at_kpoints`` keeps everything the k-set does not decide, which is
        # what a scan in nk needs: the tape's k-dependence and nothing else.
        from defumat.system.kpoints import KPoints
        calculation = calculation.at_kpoints(
            KPoints.automatic(tuple(args.kpoints), (0, 0, 0),
                              calculation.system.cell, time_reversal=False)
        )

    state = _frozen_state_shapes(calculation, args.nbnd)
    gradient, geometry = _energy_gradient(calculation)
    compiled = gradient.lower(
        _zero(), state, hoisted(calculation), geometry
    ).compile()
    analysis = compiled.memory_analysis()
    return {
        "backend": jax.default_backend(),
        "nk": int(state.wavefunctions.shape[1]),
        "nbnd": int(state.wavefunctions.shape[2]),
        "ndim": int(state.wavefunctions.shape[3]),
        "npwx": int(calculation.basis.planewaves.npwx),
        "ngm": int(calculation.basis.dense.ngm),
        "dense_grid": list(calculation.basis.dense.grid),
        "temp_bytes": int(analysis.temp_size_in_bytes),
        "argument_bytes": int(analysis.argument_size_in_bytes),
        "generated_code_bytes": int(
            getattr(analysis, "generated_code_size_in_bytes", 0)),
        "temp_GiB": round(analysis.temp_size_in_bytes / 2**30, 3),
    }


if __name__ == "__main__":
    sys.exit(main())
