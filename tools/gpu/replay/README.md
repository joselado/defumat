# Replaying a Davidson call: finding why two machines take different numbers of steps

Written 2026-10-01 for the card's endgame stall (`PERFORMANCE.md`, "The endgame on a card is a
stall"; `GPU-SPEED-NEXT.md`). Run everything from the repository root, one process per
configuration, with `DEFUMAT_THREADS=off OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` and the host pinned
(`taskset -c 0-3`) the way `tools/parallel/gpu_scan.sh` does, and nothing else on the card.

* `../../parallel/time_scf.py <input> <repeats> <label> [--max-iterations n] [--conv-thr x]`: one warm
  SCF timing as one JSON line, with `davidson_steps` (steps per SCF iteration) beside the time. Read
  every card time against it. `table.py` prints a file of those lines, one row each.
* `stall_stats.py`: the distribution of steps of one captured call over 1e-13 perturbations of its
  starting states. The first thing to run on a new platform; a CPU gave exactly 3 every time.
* `call_xplat.py`: export a call's inputs on one machine and replay them on another, or with the
  subspace solve (`HOSTEIGH=all`), only its Cholesky (`chol`) or only its `eigh` (`eigh`) on the host.
  This is what located the cause.
* `stall_steps.py`: per-step eigenvalue changes of a call at consecutive step caps (`CALL`, `CAPS`).
* `eigh_noise.py`: the subspace solve's error against SciPy by condition number, on random pairs.
* `kern_cats.py`: groups an `nsys stats -r cuda_gpu_kern_sum -f csv` table into FFT, matrix products,
  dense solve and elementwise.

To reproduce the behaviour that was fixed, set the old factor before anything is traced:

```python
import defumat.solvers.davidson as d
d.PARK_FACTOR = 1000.0      # the committed value is 4.0
```

The tools take norm-conserving cells (`benchmarks/si16-1k-ecut30.in`, `si64-1k-ecut30.in`);
`call_xplat.py` rebuilds the Hamiltonian from a saved potential and does not carry a `D_ij`.
