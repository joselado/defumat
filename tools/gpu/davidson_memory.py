"""What the Davidson executable's XLA temp buffer costs, before it is allocated.

``defumat/sizing.py`` counts the arrays whose size the *basis* fixes and says so
-- it is a floor, and it does not cover "XLA's own scratch, the temporaries of a
fused kernel". On a 157-atom slab that omission is the whole feasibility
question: the sized floor is 95.7 GB and the single allocation the card is asked
for is **109.95 GiB**, one contiguous temp buffer for the compiled
``davidson_eigensolver_all``.

This measures that buffer directly. ``.lower(...).compile().memory_analysis()``
runs the compiler and nothing else, so a configuration that cannot possibly run
can still be *sized* on the card it would run on -- no OOM risk, one short job
for a whole sweep.

    python3 tools/gpu/davidson_memory.py <pw input> --pseudo-dir ./pseudo \
        --david 2 3 4 --band-batch 16 32 64

A fit over CPU shapes (si16-1k-ecut30, npwx 5900, sweeping ``nbnd``, ``david``
and ``DEFUMAT_BAND_BATCH``) gives, to 1.6 per cent,

    temp ~ 2.19 nvecx npwx zc + 4.86 nbnd npwx zc + 1.85 band_batch N_fft zc

with ``nvecx = david nbnd`` -- the subspace ``psi``/``hpsi`` pair, about five
more ``(nbnd, npwx)`` temporaries inside ``solve``, and one FFT box per band in
flight. Extrapolated to the slab's shapes it predicts 99.9 GiB against the
109.95 GiB the GPU asked for, so the *form* transfers and the coefficients are
about 10 per cent low on a different backend. Run this to get the number rather
than the extrapolation.
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
    parser.add_argument("--david", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--band-batch", type=int, nargs="+", default=[None])
    parser.add_argument("--nbnd", type=int, nargs="+", default=[None])
    parser.add_argument("--k-batch", type=int, default=1)
    parser.add_argument("--json", default=None)
    parser.add_argument("--point", default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    os.environ.setdefault("DEFUMAT_CACHE_DIR", "off")

    if args.point is not None:
        david, nbnd, band_batch = json.loads(args.point)
        print("__POINT__" + json.dumps(_measure(args, david, nbnd, band_batch)),
              flush=True)
        return 0

    # One subprocess per point. JAX caches a lowering for the life of a
    # process, and ``DEFUMAT_BAND_BATCH`` is a closure value rather than part
    # of the cache key, so a second point in the same process silently reports
    # the first one's buffer.
    import subprocess

    rows = []
    for band_batch in args.band_batch:
        for nbnd in args.nbnd:
            for david in args.david:
                point = json.dumps([david, nbnd, band_batch])
                out = subprocess.run(
                    [sys.executable, __file__, args.input, "--point", point]
                    + (["--pseudo-dir", args.pseudo_dir] if args.pseudo_dir else [])
                    + ["--k-batch", str(args.k_batch)],
                    capture_output=True, text=True,
                )
                line = [l for l in out.stdout.splitlines()
                        if l.startswith("__POINT__")]
                if not line:
                    print(out.stdout[-2000:], out.stderr[-2000:], file=sys.stderr)
                    raise SystemExit(f"point {point} failed")
                rows.append(json.loads(line[0][len("__POINT__"):]))
                print(json.dumps(rows[-1]), flush=True)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(rows, handle, indent=1)
    return 0


def _measure(args, david, nbnd, band_batch):
    """One (david, nbnd, band_batch) point, in a subprocess-free way.

    ``DEFUMAT_BAND_BATCH`` is read when the batching module resolves it, which
    is per call rather than at import, so it can be set here between points.
    """
    if band_batch is not None:
        os.environ["DEFUMAT_BAND_BATCH"] = str(band_batch)
    else:
        os.environ.pop("DEFUMAT_BAND_BATCH", None)

    import defumat.solvers as solvers
    import defumat.solvers.davidson as davidson
    from defumat import Calculator

    real = davidson.davidson_eigensolver_all
    captured = {}

    def spy(hamiltonian, bands, psi0=None, ethr=None, **kw):
        # Compile only. Nothing is executed and nothing the size of the
        # subspace is ever allocated, which is what makes this safe to run for
        # a configuration that would not fit.
        #
        # ``davidson_eigensolver_all`` is a host-side wrapper -- its retry is a
        # Python branch, deliberately (see its docstring) -- so what carries the
        # buffer is the jitted unit underneath it, and that is what is lowered.
        unit = getattr(davidson, "_every_k", None)
        if unit is None:  # a build where the whole thing is one jit
            lowered = real.lower(hamiltonian, bands, psi0, ethr, **kw)
        else:
            # ``return_finite=True`` because that is what production compiles:
            # ``robust_retry`` defaults on, and the flag is what moved the
            # finiteness guard's reduction *inside* this executable. Sizing it
            # without the flag would size a unit no run uses -- and the guard
            # sitting outside this one, invisible to every line of the size
            # report, is what let a 21.40 GiB allocation go unaccounted for on
            # a 45-atom NiBr2 slab (``OPEN.md`` Part VII).
            lowered = unit.lower(
                hamiltonian, bands, psi0, ethr, None,
                kw.get("david", davidson.DAVID_NDIM),
                kw.get("max_iterations", davidson.MAX_ITERATIONS),
                kw.get("k_batch", "default"), robust=False, return_finite=True,
            )
        analysis = lowered.compile().memory_analysis()
        captured.update(
            nbnd=int(bands),
            david=kw.get("david"),
            band_batch=band_batch,
            nk=int(hamiltonian.nk),
            temp_bytes=int(analysis.temp_size_in_bytes),
            argument_bytes=int(analysis.argument_size_in_bytes),
            output_bytes=int(analysis.output_size_in_bytes),
            # **Not a peak, despite the name -- it EXCLUDES scratch.**
            # Measured rather than assumed, because the name invites the
            # opposite reading and the first draft of this comment took it:
            # on ``si8-1k`` this executable reports temp 8,923,008 B against
            # ``peak_memory_in_bytes`` of 998,805, which is argument 809,618
            # plus output 189,081 to within 106 bytes. So the field is
            # ``argument + output - alias``, i.e. the half of the requirement
            # that donation can remove, and the executable's real requirement
            # is this **plus** ``temp``. It is recorded because that half was
            # never reported at all, not because it supersedes ``temp``.
            peak_bytes=int(analysis.peak_memory_in_bytes),
            # Zero unless an argument was donated, and equal to the donated
            # buffer when one was. It is here as a *check on this package*
            # rather than on the compiler: ``donate_argnums`` appears nowhere
            # in ``defumat`` (grep, 2026-09-14) although ``CLAUDE.md``'s JAX
            # rules ask for it on the large wavefunction and density buffers,
            # so this field is expected to be 0 and a nonzero reading means
            # that changed. On the same controlled compile, donating the one
            # wavefunction-shaped input took the peak from 48 MB to 32 -- one
            # whole input buffer, which at slab shapes is 12.10 GB.
            alias_bytes=int(analysis.alias_size_in_bytes),
        )
        raise _Done

    solvers.EIGENSOLVERS["davidson"] = spy
    solvers.EIGENSOLVERS["david"] = spy

    options = {"david": david, "k_batch": args.k_batch}
    if nbnd is not None:
        options["nbnd"] = nbnd
    kwargs = {} if args.pseudo_dir is None else {"pseudo_dir": args.pseudo_dir}
    calculator = Calculator.from_file(args.input, **kwargs, **options)
    calculation = calculator.calculation
    captured["npwx"] = int(calculation.basis.planewaves.npwx)
    captured["dense_grid"] = list(calculation.basis.dense.grid)
    try:
        calculator.get_scf(max_iterations=1)
    except _Done:
        pass
    finally:
        solvers.EIGENSOLVERS["davidson"] = real
        solvers.EIGENSOLVERS["david"] = real
    captured["temp_GiB"] = round(captured["temp_bytes"] / 2**30, 2)
    # Reported beside it rather than instead of it, because every figure on
    # record for this cell is a ``temp`` and the two have to be comparable.
    # The sum of the two is an **upper bound** and is named as one: scratch and
    # the io buffers are not proved disjoint, and the controlled compile says
    # they are not -- arg 32 + temp 16 + out 16 = 64 MB where the field read
    # 48. A single figure called "what it needs" would be one name over three
    # unknowns, which is the error this cell's record already carries once.
    captured["peak_GiB"] = round(captured["peak_bytes"] / 2**30, 2)
    captured["temp_plus_io_upper_GiB"] = round(
        (captured["temp_bytes"] + captured["peak_bytes"]) / 2**30, 2)
    return captured


class _Done(Exception):
    """Raised to stop the SCF the moment the executable has been sized."""


if __name__ == "__main__":
    sys.exit(main())
