"""Command-line entry point.

``inspect`` prints what pypresso reads out of a pw.x output, which is the
fastest way to see what a comparison is being made against. ``dos`` runs the
sequence a density of states actually is -- SCF, then NSCF on a denser grid,
then the Brillouin-zone integration -- and writes ``dos.x``'s ``.dos`` file.
The ``scf``/``bands`` subcommands land with their phases.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from pypresso import __version__
from pypresso.io import read_qe_output, write_dos
from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.scf import run_scf
from pypresso.system import build_system
from pypresso.units import RY_TO_EV
from pypresso.workflows import run_dos


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


def _dos(args) -> int:
    """SCF, then NSCF on a denser grid, then integrate -- ``pw.x`` plus ``dos.x``."""
    input_path = Path(args.input)
    pseudo_dir = Path(args.pseudo_dir) if args.pseudo_dir else input_path.parent
    system = build_system(read_pw_input(input_path))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)

    scf = run_scf(system, pseudos, conv_thr=args.conv_thr, verbose=args.verbose)
    if not scf.converged:
        print(f"warning: the SCF stopped at accuracy {scf.accuracy:.2e}", file=sys.stderr)
    print(f"  total energy     {scf.total_energy:.8f} Ry ({scf.iterations} iterations)")

    grid = tuple(args.kgrid) if args.kgrid else None
    # Emin/Emax/DeltaE are given in eV, as dos.x takes them, and converted here
    # -- the input/output boundary is the only place that speaks eV.
    dos, nscf = run_dos(
        system,
        pseudos,
        scf.density,
        grid=grid,
        nbnd=args.nbnd,
        scheme=args.scheme,
        degauss=args.degauss,
        emin=None if args.emin is None else args.emin / RY_TO_EV,
        emax=None if args.emax is None else args.emax / RY_TO_EV,
        delta_e=args.delta_e / RY_TO_EV,
        conv_thr=args.conv_thr,
    )
    print(f"  k-points         {nscf.kpoints.nk} irreducible"
          + (f" from a {grid[0]}x{grid[1]}x{grid[2]} grid" if grid else ""))
    print(f"  scheme           {dos.scheme}")
    if dos.fermi_energy is not None:
        print(f"  Fermi energy     {dos.fermi_energy * RY_TO_EV:.4f} eV")
        print(f"  states below E_F {dos.states_below(dos.fermi_energy):.6f}")
        print(f"  D(E_F)           {dos.at(dos.fermi_energy) / RY_TO_EV:.4f} states/eV")

    output = Path(args.output) if args.output else input_path.with_suffix(".dos")
    write_dos(output, dos)
    print(f"  wrote            {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pypresso", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"pypresso {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="summarise a Quantum ESPRESSO pw.x output file")
    inspect.add_argument("path", help="path to a pw.x output (e.g. a test-suite benchmark)")

    dos = sub.add_parser("dos", help="SCF, then a density of states on a denser k-grid")
    dos.add_argument("input", help="path to a pw.x input file")
    dos.add_argument("--pseudo-dir", help="where the UPF files are (default: beside the input)")
    dos.add_argument("--kgrid", nargs=3, type=int, metavar=("N1", "N2", "N3"),
                     help="denser Monkhorst-Pack grid for the NSCF step")
    dos.add_argument("--nbnd", type=int, help="bands to compute (default: the input's)")
    dos.add_argument("--scheme", help="integration scheme (default: the input's occupations)")
    dos.add_argument("--degauss", type=float, help="smearing width in Ry, for a smearing scheme")
    dos.add_argument("--emin", type=float, help="lowest energy in eV (default: the lowest band)")
    dos.add_argument("--emax", type=float, help="highest energy in eV (default: the highest band)")
    dos.add_argument("--delta-e", type=float, default=0.01, help="energy step in eV (default 0.01)")
    dos.add_argument("--conv-thr", type=float, default=1e-8, help="SCF convergence threshold in Ry")
    dos.add_argument("-o", "--output", help="where to write the .dos file")
    dos.add_argument("-v", "--verbose", action="store_true", help="print each SCF iteration")

    args = parser.parse_args(argv)
    if args.command == "inspect":
        return _inspect(args.path)
    if args.command == "dos":
        return _dos(args)
    parser.error(f"unhandled command {args.command}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
