# Silicon, part 3: making it fast, and how fast is fast enough

Parts 1 and 2 got the physics right: silicon's total energy within 1.1e-8 Ry of Quantum
ESPRESSO, its bands within 0.0002 eV. This notebook is about the other question a
reimplementation has to answer — **how much does being right in Python cost?**

The only honest way to ask that is to run the same input through both codes on the same
machine with both restricted to one core, which is what `tools/compare_qe.py` does. What
follows is the anatomy of the answer: where the time went, what moved it, and what the
remaining gap is made of.

The headline: on a production-cutoff silicon SCF, **one pypresso iteration costs about
three and a half times a Quantum ESPRESSO iteration**, and the two total energies agree
to 2.6e-9 Ry.

**What this covers:** the block Davidson eigensolver (completing P4), the dense
Hamiltonian's matrix elements, and the P10 optimisation work. It uses the same two-atom
silicon cell as the earlier notebooks.


```python
# XLA must be pinned before JAX is imported, so this comes first: every timing
# below is single core, which is the only way to compare against a serial pw.x.
import os

os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
os.environ["OMP_NUM_THREADS"] = "1"

import time
from pathlib import Path

import jax
import numpy as np

from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.scf.driver import Calculation, run_scf
from pypresso.scf.potential import v_of_rho
from pypresso.solvers import davidson_eigensolver_all, dense_eigensolver_all
from pypresso.system import build_system

BENCH = Path("../benchmarks")
PSEUDO = Path("../tests/data/pseudo")


def load(name):
    """A benchmark input and its pseudopotentials."""
    system = build_system(read_pw_input(BENCH / name))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    return system, pseudos


system, pseudos = load("si-1k.in")
calculation = Calculation(system, pseudos)
print(f"{system.structure.nat} atoms, {system.kpoints.nk} k-point, "
      f"ecutwfc {system.ecutwfc} Ry -> {calculation.basis.npwx} plane waves")

```

    2 atoms, 1 k-point, ecutwfc 12.0 Ry -> 180 plane waves


## 1. Why one k-point

The benchmark input is `benchmarks/si-1k.in`: the test suite's silicon with its two-point
k-set cut to one point. That is deliberate. Both codes parallelise over k-points, so a
multi-k comparison mostly measures how well each one batches — an interesting question,
but a different one. With a single k-point what is left is the cost of one Hamiltonian
solve, one density, and one potential, which is the thing to get right first.

## 2. Where the time was going

Profiling one converged SCF iteration answers this directly. The units below are
milliseconds, single core.


```python
def timed(label, function, repeats=20):
    jax.block_until_ready(function())          # warm up: the first call compiles
    start = time.perf_counter()
    for _ in range(repeats):
        result = jax.block_until_ready(function())
    return label, (time.perf_counter() - start) / repeats * 1e3, result


# The SCF loop calls a jitted v_of_rho; calling the bare function instead would
# time XLA dispatching its twenty operations one at a time, not the arithmetic.
potential_of_rho = jax.jit(v_of_rho)

rho = calculation.starting_density()
potential = potential_of_rho(rho, calculation.basis.dense, system.cell)
hamiltonian = calculation.hamiltonian(potential.v_scf)
nbnd = 4

eigenvalues, psi = davidson_eigensolver_all(hamiltonian, nbnd)
weights, _ = calculation.occupations(eigenvalues)

stages = [
    timed("v_of_rho", lambda: potential_of_rho(rho, calculation.basis.dense, system.cell)),
    timed("diagonalise (Davidson)", lambda: calculation.diagonalize(hamiltonian, nbnd, psi)),
    timed("occupations", lambda: calculation.occupations(eigenvalues)),
    timed("density + symmetrise", lambda: calculation.density(psi, weights)),
]
for label, milliseconds, _ in stages:
    print(f"  {label:26s} {milliseconds:7.2f} ms")

```

      v_of_rho                      1.85 ms
      diagonalise (Davidson)        3.24 ms
      occupations                   0.85 ms
      density + symmetrise          1.47 ms


The eigensolver dominates, and before any of this work it dominated far more: building the
Hamiltonian as an explicit matrix and calling `eigh` on it took **40 of the 52 ms** an
iteration then cost. Two separate things were wrong with that, and they have different
fixes.

## 3. The matrix elements: 180 FFTs replaced by one

The dense solver built `H` by applying the operator to every basis vector in turn. That is
unambiguously correct — it uses no matrix-element formula at all, which is exactly what
makes it a good reference — but it costs one FFT per plane wave.

It is not how the matrix has to be built. The local potential is diagonal in real space,
so in the plane-wave basis it is

$$\langle \mathbf{k}+\mathbf{G}_i | V | \mathbf{k}+\mathbf{G}_j\rangle = V(\mathbf{G}_i - \mathbf{G}_j)$$

— a *gather* from one Fourier transform of the potential. The kinetic term is the
diagonal, and the nonlocal term is $V_{\rm NL} = \beta D \beta^\dagger$, already a matrix.

The gather is exact only if the density grid can represent every difference
$\mathbf{G}_i - \mathbf{G}_j$. Those reach $2\sqrt{E_{\rm cut}^{\rm wfc}}$, so the
condition is `ecutrho >= 4 * ecutwfc` — which is precisely why QE's default dual is 4.
Both builds are kept, and the test suite asserts they agree.


```python
label, direct, _ = timed("formula", lambda: hamiltonian.matrix(0))
label, applied, _ = timed("by application", lambda: hamiltonian.matrix_by_application(0), 5)

difference = np.abs(np.asarray(hamiltonian.matrix(0))
                    - np.asarray(hamiltonian.matrix_by_application(0))).max()
print(f"  from matrix elements   {direct:7.2f} ms")
print(f"  by applying H          {applied:7.2f} ms   ({applied / direct:.0f}x slower)")
print(f"  largest disagreement   {difference:.2e} Ry")
```

      from matrix elements      5.70 ms
      by applying H            53.66 ms   (9x slower)
      largest disagreement   3.55e-15 Ry


## 4. The eigensolver: Davidson

Replacing the *build* leaves the *solve*: `eigh` on an `npw x npw` matrix, which is
`O(npw^3)` in time and `O(npw^2)` in memory. At 180 plane waves that is a few
milliseconds. At the 1131 plane waves of a production cutoff it is over a second per
iteration, and at the tens of thousands a real calculation needs it is impossible.

Nobody diagonalises the full matrix. The occupied states are a handful, and an iterative
solver finds only those, never forming `H`. QE's default is block Davidson
(`KS_Solvers/Davidson/cegterg.f90`), which pypresso now transcribes: given an estimate
`(e, psi)`, the residual `(H - e) psi` is what the estimate is missing, so add it to the
subspace, diagonalise there, and repeat.


```python
exact, _ = dense_eigensolver_all(hamiltonian, nbnd)
iterative, _ = davidson_eigensolver_all(hamiltonian, nbnd, None, max_iterations=60)

print("  band     dense (Ry)     Davidson (Ry)     difference")
for band, (a, b) in enumerate(zip(np.asarray(exact)[0], np.asarray(iterative)[0])):
    print(f"  {band:4d}   {a:13.9f}   {b:13.9f}   {abs(a - b):.1e}")
```

      band     dense (Ry)     Davidson (Ry)     difference
         0    -0.390299160    -0.390299160   3.3e-16
         1     0.140370280     0.140370280   1.7e-15
         2     0.366613820     0.366613820   2.3e-15
         3     0.366613820     0.366613820   9.5e-14


The same answer, and the subspace it came from was 16 x 16 rather than 180 x 180.

Two details of the transcription are worth pulling out, because both were found by a
number being wrong rather than by reading the Fortran.

**Converged roots must stop being expanded.** A converged band has a residual of order
1e-14. Normalising that to unit length — which the algorithm does to every correction
vector — turns pure round-off into a basis vector, the overlap matrix goes singular, and
the *other* bands stop converging. `cegterg` never hits this because it compacts the
unconverged roots to the front and works only on those.

**The subspace must grow by the number of unconverged roots, not by the block size.** With
one stubborn band left, advancing by the full block fills the subspace three times faster
than it fills with anything useful, and the periodic collapse throws the search direction
away before it has converged. Both symptoms looked identical from outside: silicon's
highest band sitting a few meV above the reference, and nothing else wrong.

Under JAX both fixes have to keep shapes static, so the compaction is a stable `argsort`
on the convergence flags and the subspace is masked rather than resized.

## 5. What it is worth

The gain from an iterative solver is invisible on a toy system and decisive on a real one.
Here is the same SCF at two cutoffs, with each solver.


```python
def scf_time(system, pseudos, solver, repeats=3):
    calculation = Calculation(system, pseudos, diagonalization=solver)
    run_scf(system, pseudos, calculation=calculation, conv_thr=1e-10)   # compile
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-10)
        best = min(best, time.perf_counter() - start)
    return best / result.iterations * 1e3, result.total_energy, calculation.basis.npwx


rows = []
for name in ("si-1k.in", "si-1k-ecut40.in"):
    this_system, these_pseudos = load(name)
    for solver in ("dense", "davidson"):
        milliseconds, energy, npwx = scf_time(this_system, these_pseudos, solver)
        rows.append((name, npwx, solver, milliseconds, energy))
        print(f"  {name:17s} npw {npwx:5d}  {solver:9s} {milliseconds:8.1f} ms/iteration"
              f"   E = {energy:.9f} Ry")
```

      si-1k.in          npw   180  dense         23.9 ms/iteration   E = -15.254449448 Ry


      si-1k.in          npw   180  davidson      13.1 ms/iteration   E = -15.254449448 Ry


      si-1k-ecut40.in   npw  1131  dense       1240.8 ms/iteration   E = -15.304610213 Ry


      si-1k-ecut40.in   npw  1131  davidson      74.0 ms/iteration   E = -15.304610213 Ry


Identical energies, and at 1131 plane waves Davidson is more than ten times faster. The
ratio grows as `npw` does, because the two are not the same algorithm: one is `O(npw^3)`,
the other is `nbnd` applications of `H` per step.

## 6. Asking for only as much accuracy as the density deserves

There is one more factor, and it is not in the code that does the arithmetic — it is in
how accurately the arithmetic is asked to be done. The eigenvalues can never be more
meaningful than the density they were computed from, so converging them to twelve digits
against a starting density that is wrong in the second is pure waste.

QE schedules its diagonalisation threshold accordingly (`electrons.f90`): `ethr` begins at
1e-2 and follows `0.1 dr2 / nelec` down as the density converges, where `dr2` is the
estimated self-consistency error — the Hartree energy of the density residual, which
weights long-wavelength errors by `1/G^2` because those are the ones that cost energy.
pypresso now does the same, and `conv_thr` is compared against the same quantity, so it
means what it means in a `pw.x` input.



```python
system40, pseudos40 = load("si-1k-ecut40.in")
result = run_scf(system40, pseudos40, conv_thr=1e-10)

print("  iter      total energy      accuracy       ethr")
for row in result.history:
    print(f"  {row['iteration']:4d}   {row['total_energy']:15.9f}   "
          f"{row['accuracy']:.2e}   {row['ethr']:.2e}")

```

      iter      total energy      accuracy       ethr
         1     -15.334298206   1.44e-01   1.00e-02
         2     -15.303542878   3.60e-03   1.79e-03
         3     -15.304584095   7.94e-05   4.50e-05
         4     -15.304609263   1.21e-06   9.93e-07
         5     -15.304610316   2.78e-07   1.51e-08
         6     -15.304610211   2.32e-10   3.47e-09
         7     -15.304610213   1.77e-11   2.90e-12


The thresholds track QE's own sequence for the same input (1e-2, 1.6e-3, 7.0e-5, 1.5e-6,
3.3e-8, 3.4e-10, 1.7e-11, 2.0e-12) closely enough that the two are visibly running the same
schedule. The effect: **75 Davidson steps become 33**, against QE's 25 — and silicon's
total energy still matches QE to 1.09e-8 Ry, term by term, exactly as before. The energy is
variational in the density, so stopping the eigenvalues early costs it nothing.


## 7. The other half: compilation, not arithmetic

Every JAX operation dispatched outside a `jit` is compiled by XLA separately, and on these
array sizes that compilation costs tens of milliseconds while the arithmetic costs
microseconds. Setup — the basis, the projectors, the local potential, Ewald — was 81
separate compilations and about 10 seconds, of which the actual work was under 0.2 s.

The fix is not cleverer arithmetic but fewer compiled units: wrap whole functions in `jit`
so one graph compiles instead of thirty, do host-side constant arithmetic in NumPy where
nothing differentiates through it, and replace Python loops over k-points, atoms and
symmetry operations with batched operations. Rebuilding the same objects a second time
shows what is compilation and what is work.


```python
import subprocess
import sys
import textwrap

# In a fresh process, because by this point in the notebook everything has been
# compiled already and the question is what a first run costs.
probe = textwrap.dedent("""
    import os, sys, time
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    os.environ["OMP_NUM_THREADS"] = "1"
    sys.path.insert(0, "..")
    import jax
    from pathlib import Path
    from pypresso.io.pwin import read_pw_input
    from pypresso.pseudo import read_upf
    from pypresso.scf.driver import Calculation
    from pypresso.system import build_system

    system = build_system(read_pw_input(Path("../benchmarks/si-1k.in")))
    pseudos = tuple(read_upf(Path("../tests/data/pseudo") / s.pseudo_file)
                    for s in system.structure.species)

    for attempt in ("first", "second"):
        start = time.perf_counter()
        calculation = Calculation(system, pseudos)
        jax.block_until_ready(calculation.vltot)
        print(f"  {attempt:6s} Calculation  {time.perf_counter() - start:6.3f} s")
""")
print(subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True).stdout)

```

      first  Calculation   4.084 s
      second Calculation   0.049 s
    


## 8. Against Quantum ESPRESSO, single core

`tools/compare_qe.py` runs both codes on the same input, QE built serial and pypresso with
XLA pinned to one thread, and reads QE's own timing report. On this machine:

| | `si-1k.in` (180 PWs) | `si-1k-ecut40.in` (1131 PWs) |
|---|---|---|
| QE, per SCF iteration | 0.003 s | 0.013 s |
| pypresso, per SCF iteration | 0.009 s | 0.044 s |
| ratio | **3.6x** | **3.5x** |
| total energy agreement | 7e-7 Ry | 3e-9 Ry |

Three and a half times a mature Fortran code, in Python, on one core, with the same
numbers. The remaining gap is not one thing: `h_psi` is applied to every band where
`cegterg` applies it only to the unconverged ones, each FFT is about 1.5x FFTW
single-threaded, and per-call dispatch overhead still shows on the small case.

`PERFORMANCE.md` carries the full breakdown and the backlog.
