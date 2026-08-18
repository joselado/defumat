"""Command-line entry point.

At P0 the only thing there is to drive is the QE reference parser -- ``inspect``
prints what pypresso reads out of a pw.x output, which is the fastest way to see
what a later phase will be compared against. The ``scf``/``bands``/``dos``
subcommands land with their phases.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from pypresso import __version__
from pypresso.io import read_qe_output


def _inspect(path: str) -> int:
    ref = read_qe_output(path)
    np.set_printoptions(precision=6, suppress=True)

    print(f"{ref.path}")
    print(f"  calculation      {ref.calculation}")
    print(f"  cell             ibrav={ref.ibrav} alat={ref.alat} volume={ref.volume}")
    print(f"  system           nat={ref.nat} ntyp={ref.ntyp} nelec={ref.nelec} nbnd={ref.nbnd}")
    print(f"  cutoffs (Ry)     wfc={ref.ecutwfc} rho={ref.ecutrho}   xc={ref.xc}")
    print(f"  dense grid       {ref.ngm_dense} G-vectors, FFT {ref.fft_dense}")
    if ref.kpoints is not None:
        print(f"  k-points         {len(ref.kpoints)} (weights sum {ref.weights.sum():.4f})")
    if ref.npw is not None:
        print(f"  plane waves      min {ref.npw.min()}  max {ref.npw.max()}")
    if ref.total_energy is not None:
        print(f"  total energy     {ref.total_energy:.8f} Ry  ({ref.n_iterations} iterations)")
        for name, value in ref.energy_terms.items():
            print(f"      {name:<16} {value:>16.8f} Ry")
    if ref.eigenvalues is not None:
        print(f"  eigenvalues      shape {ref.eigenvalues.shape} (nspin, nk, nbnd), eV")
    if ref.fermi_energy is not None:
        print(f"  Fermi energy     {ref.fermi_energy:.4f} eV")
    if ref.homo is not None:
        print(f"  HOMO             {ref.homo:.4f} eV")
    if ref.forces is not None:
        print(f"  forces           max |F| {np.abs(ref.forces).max():.6f} Ry/bohr")
    if ref.stress is not None:
        print(f"  pressure         {ref.pressure:.2f} kbar")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pypresso", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"pypresso {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="summarise a Quantum ESPRESSO pw.x output file")
    inspect.add_argument("path", help="path to a pw.x output (e.g. a test-suite benchmark)")

    args = parser.parse_args(argv)
    if args.command == "inspect":
        return _inspect(args.path)
    parser.error(f"unhandled command {args.command}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
