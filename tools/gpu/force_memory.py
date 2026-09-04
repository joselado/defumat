"""What the force's reverse pass costs, before a ground state has been paid for.

The companion to :mod:`tools.gpu.davidson_memory`, and it exists because of the
order the two run in: a ``calibrate`` stage converges an SCF and *then* takes
``compute_forces``, so a force that does not fit is discovered after the
expensive part rather than before it. ``defumat/sizing.py`` says in its own
docstring that its floor does not cover "the autodiff tape of a derivative that
has not been asked for; a reverse-mode force carries intermediates this cannot
see" -- this is how to see them.

``jax.jit(jax.grad(frozen_energy)).lower(...).compile().memory_analysis()``
runs the compiler and nothing else, and the ``FrozenState`` it is lowered
against is built from :class:`jax.ShapeDtypeStruct` leaves rather than from a
converged run, so **no wavefunction is ever allocated**. A configuration whose
SCF cannot finish on the card can still have its force sized on it.

    python3 tools/gpu/force_memory.py <pw input> --pseudo-dir ./pseudo \
        --nbnd 1020 900

The one thing to know about the answer: unlike the eigensolver's, this buffer
does not have a knob. ``diago_david_ndim`` and the band batch do not appear in
it -- the tape is set by ``nbnd``, ``npwx`` and the FFT grid alone -- so if it
does not fit, what moves is the cutoff or the band count.

Deliberately a separate file from ``davidson_memory.py`` rather than a stage
inside it: that one is being run against production inputs right now, and a
refactor of a diagnostic while it is in flight is how a diagnostic stops being
trusted. The subprocess harness is duplicated for the same reason.
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
    parser.add_argument("--nbnd", type=int, nargs="+", default=[None])
    parser.add_argument("--k-batch", type=int, default=1)
    parser.add_argument("--band-batch", type=int, nargs="+", default=[None])
    parser.add_argument("--json", default=None)
    parser.add_argument("--point", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    os.environ.setdefault("DEFUMAT_CACHE_DIR", "off")

    if args.point is not None:
        nbnd, band_batch = json.loads(args.point)
        print("__POINT__" + json.dumps(_measure(args, nbnd, band_batch)), flush=True)
        return 0

    import subprocess

    rows = []
    for band_batch in args.band_batch:
        for nbnd in args.nbnd:
            out = subprocess.run(
                [sys.executable, __file__, args.input,
                 "--point", json.dumps([nbnd, band_batch]),
                 "--k-batch", str(args.k_batch)]
                + (["--pseudo-dir", args.pseudo_dir] if args.pseudo_dir else []),
                capture_output=True, text=True,
            )
            line = [l for l in out.stdout.splitlines() if l.startswith("__POINT__")]
            if not line:
                print(out.stdout[-3000:], out.stderr[-3000:], file=sys.stderr)
                raise SystemExit(f"point nbnd={nbnd} band_batch={band_batch} failed")
            rows.append(json.loads(line[0][len("__POINT__"):]))
            print(json.dumps(rows[-1]), flush=True)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(rows, handle, indent=1)
    return 0


def _measure(args, nbnd, band_batch):
    if band_batch is not None:
        os.environ["DEFUMAT_BAND_BATCH"] = str(band_batch)
    else:
        os.environ.pop("DEFUMAT_BAND_BATCH", None)

    import jax
    from defumat import Calculator
    from defumat.forces.energy import FrozenState, frozen_energy

    options = {"k_batch": args.k_batch}
    if nbnd is not None:
        options["nbnd"] = nbnd
    kwargs = {} if args.pseudo_dir is None else {"pseudo_dir": args.pseudo_dir}
    calculator = Calculator.from_file(args.input, **kwargs, **options)
    calculation = calculator.calculation

    state = _frozen_state_shapes(calculation, nbnd)
    positions = calculation.system.structure.positions
    gradient = jax.jit(jax.grad(
        lambda tau, st: frozen_energy(calculation, tau, st, spinors=True)
    ))
    analysis = gradient.lower(positions, state).compile().memory_analysis()
    return {
        "nbnd": int(state.wavefunctions.shape[2]),
        "band_batch": band_batch,
        "nspin": int(state.wavefunctions.shape[0]),
        "nk": int(state.wavefunctions.shape[1]),
        "ndim": int(state.wavefunctions.shape[3]),
        "npwx": int(calculation.basis.planewaves.npwx),
        "dense_grid": list(calculation.basis.dense.grid),
        "temp_bytes": int(analysis.temp_size_in_bytes),
        "argument_bytes": int(analysis.argument_size_in_bytes),
        "output_bytes": int(analysis.output_size_in_bytes),
        "temp_GiB": round(analysis.temp_size_in_bytes / 2**30, 2),
    }


def _frozen_state_shapes(calculation, nbnd) -> FrozenState:
    """A :class:`~defumat.forces.energy.FrozenState` of shapes and no data.

    ``density`` and ``potential`` stay ``None``: they are what the *analytic*
    force consumes, and the autodiff route rebuilds the density as a function
    of the positions, which is the whole point of it (``forces/energy.py``).
    ``entropy`` is a real scalar rather than a shape because the field carries
    a ``jnp.asarray`` converter; it costs eight bytes.
    """
    import jax

    from defumat.forces.energy import FrozenState
    from defumat.scf.driver import default_nbnd

    system = calculation.system
    precision = system.cell.precision
    if nbnd is None:
        nbnd = system.nbnd
    if nbnd is None:
        nelup = neldw = None
        if system.tot_magnetization is not None and system.nspin == 2:
            nelup = 0.5 * (calculation.nelec + system.tot_magnetization)
            neldw = 0.5 * (calculation.nelec - system.tot_magnetization)
        nbnd = default_nbnd(
            calculation.nelec, system.occupations, nelup=nelup, neldw=neldw,
            noncolin=(system.nspin == 4),
        )
    nbnd = int(nbnd)

    # The wavefunctions' leading axis is the *channel* count, which is 2 only
    # for a collinear run: a spinor's two components live inside ``ndim``
    # (``CLAUDE.md``: nspin, npol and nspin_mag are three different numbers).
    channels = 2 if system.nspin == 2 else 1
    nk = len(calculation.system.kpoints.coords)
    ndim = int(calculation.basis.planewaves.npwx) * system.npol

    return FrozenState(
        wavefunctions=jax.ShapeDtypeStruct((channels, nk, nbnd, ndim),
                                          precision.complex),
        weights=jax.ShapeDtypeStruct((channels, nk, nbnd), precision.real),
        eigenvalues=jax.ShapeDtypeStruct((channels, nk, nbnd), precision.real),
        entropy=0.0,
    )


if __name__ == "__main__":
    sys.exit(main())
