"""Time a pypresso calculation, component by component.

Run from the repository root:

    python3 tools/benchmark.py quantum_espresso/.../pw_scf/scf.in

This is the *diagnosis* tool: it says where pypresso's own time goes. The
measurement that matters is ``tools/compare_qe.py``, which runs the same input
through Quantum ESPRESSO and through pypresso with both pinned to one core; come
here when that comparison has produced a ratio in need of an explanation.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import numpy as np

from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.scf.driver import Calculation, run_scf
from pypresso.scf.potential import v_of_rho
from pypresso.solvers.dense import dense_eigensolver
from pypresso.system import build_system


def _block(value):
    """Force JAX's asynchronous dispatch to finish before stopping the clock."""
    return jax.block_until_ready(value)


class Timer:
    def __init__(self):
        self.records = {}

    def __call__(self, label, function, repeats=1):
        _block(function())  # warm up: the first call compiles
        start = time.perf_counter()
        for _ in range(repeats):
            result = _block(function())
        elapsed = (time.perf_counter() - start) / repeats
        self.records[label] = elapsed
        return result, elapsed


def benchmark(input_path: Path, pseudo_dir: Path):
    system = build_system(read_pw_input(input_path))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)

    timer = Timer()
    print(f"{input_path.name}: {system.structure.nat} atoms, {system.kpoints.nk} k-points, "
          f"ecutwfc {system.ecutwfc} Ry")

    start = time.perf_counter()
    calculation = Calculation(system, pseudos)
    setup = time.perf_counter() - start
    basis = calculation.basis
    print(f"  basis: {basis.dense.ngm} G-vectors, FFT {basis.dense.grid}, "
          f"npwx {basis.npwx}, {calculation.symmetries.nsym} symmetries")

    rho = calculation.starting_density()
    potential, _ = timer("v_of_rho", lambda: v_of_rho(rho, basis.dense, system.cell), repeats=3)
    hamiltonian = calculation.hamiltonian(potential.v_scf)[0]

    nbnd = system.nbnd or max(int(round(calculation.nelec / 2)), 1)
    psi = hamiltonian.apply(
        np.zeros((nbnd, basis.npwx), dtype=complex) + 1e-3, 0
    )
    timer("h_psi (one k, all bands)", lambda: hamiltonian.apply(psi, 0), repeats=10)
    timer("diagonalise (one k, dense)", lambda: dense_eigensolver(hamiltonian, 0, nbnd))
    timer("symmetrize density", lambda: calculation.symmetrize(rho), repeats=3)

    start = time.perf_counter()
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-8)
    cold = time.perf_counter() - start

    # Again, with every kernel compiled. The difference between the two is XLA,
    # not physics, and reporting only the first would put a fixed startup cost
    # into a number meant to describe the loop.
    start = time.perf_counter()
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-8)
    warm = time.perf_counter() - start

    print(f"  setup (basis, projectors, vloc, ewald, symmetry): {setup:8.3f} s")
    for label, elapsed in timer.records.items():
        print(f"  {label:<34} {elapsed:8.3f} s")
    print(f"  full SCF ({result.iterations} iterations), cold: {cold:8.3f} s")
    print(f"  full SCF ({result.iterations} iterations), warm: {warm:8.3f} s"
          f"   -> {warm / result.iterations:.3f} s/iteration")
    print(f"  total energy {result.total_energy:.8f} Ry")
    return {"setup": setup, "scf": warm, "scf_cold": cold,
            "iterations": result.iterations, **timer.records}


if __name__ == "__main__":
    path = Path(sys.argv[1])
    pseudo = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("tests/data/pseudo")
    benchmark(path, pseudo)
