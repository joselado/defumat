"""What share of a warm SCF iteration is ``Hamiltonian.apply``, from a count of its rows.

    taskset -c 0 python3 tools/parallel/hpsi_share.py benchmarks/si64-1k-ecut30.in 4

Wraps ``Hamiltonian.apply`` before anything is compiled so that every call, inside the compiled solve too,
adds the rows it was given to a counter (a ``jax.debug.callback``, whose cost is
a few microseconds against milliseconds a row). The warm iteration time and the
cost of one row, taken from a bare ``apply`` of the whole block, then say how
much of the iteration the application is; the rest is the dense algebra, the
density and the potential. This is the Amdahl figure for splitting ``h_psi``
alone (``scf_devices.py``) and the size of what a distributed Davidson would
have to take over.

Measured on D22's performance cores (2026-10-01), share of an iteration spent in
``apply``: si64 0.42 at one core and 0.52 at six, the ultrasoft spinor
``si16-spinor-us-6k.in`` 0.35 and 0.40. Set ``DEFUMAT_THREADS=off`` and pin with ``taskset``.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import defumat.hamiltonian.noncollinear as noncollinear
import defumat.hamiltonian.operator as operator
from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.scf.driver import Calculation, run_scf
from defumat.system import build_system

REPO = Path(__file__).resolve().parents[2]
ROWS = [0]


def counted(apply):
    def wrapped(self, psi, ik):
        jax.debug.callback(lambda n: ROWS.__setitem__(0, ROWS[0] + int(n)),
                           jnp.asarray(psi.size // psi.shape[-1]))
        return apply(self, psi, ik)
    return wrapped


def main() -> None:
    warnings.simplefilter("ignore")
    path, iterations = Path(sys.argv[1]), int(sys.argv[2])
    operator.Hamiltonian.apply = counted(operator.Hamiltonian.apply)
    for cls in [c for c in vars(noncollinear).values()
                if isinstance(c, type) and "apply" in vars(c)]:
        cls.apply = counted(cls.apply)
    system = build_system(read_pw_input(path))
    pseudos = tuple(read_upf(REPO / "tests" / "data" / "pseudo" / s.pseudo_file)
                    for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    # iterations = 0 runs to the input's own convergence threshold instead
    options = ({"conv_thr": 1e-16, "max_iterations": iterations, "verbose": False}
               if iterations else {"verbose": False})
    run_scf(system, pseudos, calculation=calculation, **options)   # compile
    ROWS[0] = 0
    samples = []
    for _ in range(2):
        start = time.perf_counter()
        result = run_scf(system, pseudos, calculation=calculation, **options)
        samples.append((time.perf_counter() - start) / result.iterations)
    rows = ROWS[0] / (2 * result.iterations)
    # the cost of one row: a bare apply of nbnd random rows at the first k-point
    nbnd = int(result.eigenvalues.shape[-1])
    potential = calculation.potential(calculation.starting_density())
    hamiltonian = calculation.hamiltonian(potential.v_scf)[0]
    rng = np.random.default_rng(0)
    psi = jnp.asarray(rng.standard_normal((nbnd, calculation.basis.npwx * system.npol))
                      + 1j * rng.standard_normal((nbnd, calculation.basis.npwx * system.npol)))
    bare = jax.jit(lambda p: hamiltonian.apply(p, 0))
    jax.block_until_ready(bare(psi))
    times = []
    for _ in range(5):
        start = time.perf_counter()
        jax.block_until_ready(bare(psi))
        times.append(time.perf_counter() - start)
    per_row = statistics.median(times) / nbnd
    print(json.dumps({"input": path.name, "ms_per_iter": round(1e3 * statistics.median(samples), 1),
                      "iterations": result.iterations,
                      "nbnd": nbnd, "ms_per_row": round(1e3 * per_row, 3),
                      "apply_rows_per_iter": rows,
                      "apply_share": round(rows * per_row / statistics.median(samples), 3)}), flush=True)


if __name__ == "__main__":
    main()
