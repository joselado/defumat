# Fable review of `open-d-vii1-sizing.md` (2026-10-10)

The user approved the Triton jobs on 2026-10-10 "after the fable review". The review changes
their shape, so the jobs go back to the user before submission. Nothing was run by the review.

**Verdict: do not submit the two 2-hour jobs as the plan writes them.** `tools/gpu/stage_peaks.py`
survives 14 iterations at `conv_thr = 1e-14` and its dials reach the run, but as committed it cannot
answer the plan's question:

- it brackets `diagonalize` at the method boundary, which holds the `psi0[spin]` slice, the
  solver, its retry and the `jnp.stack` restack as one stage (`scf/driver.py:5994-6011`), each of
  the order of the gap on the slab (one wavefunction set is 12.10 GB); wrap
  `calculation.eigensolver` on the instance (`driver.py:2158`, an attribute the class loop misses)
  and `davidson._every_k` (`davidson.py:915`), so attribution is a containment;
- it runs the SCF at `verbose = False` (`stage_peaks.py:203`), so `ethr` and the Davidson step
  counts never reach the log; pass `verbose=True`;
- it records two of the ten allocator keys (`stage_peaks.py:37`); record the whole
  `memory_stats()` dict (`largest_free_block_bytes`, `largest_alloc_size`, `num_allocs`,
  `bytes_reserved`, `peak_bytes_reserved` classify a death);
- `memory_mode == "memory"` switches code paths whatever the dials say (`driver.py:2364, 3622,
  3666, 3814, 4385`): pin `--memory-mode memory` on every arm;
- the robustness retry compiles a second `_every_k` (`davidson.py:1190-1250`) whose only trace is
  a warning: run with `PYTHONWARNINGS=always`, and log `ethr`, steps and unsettled bands from the
  `diagonalize` wrapper;
- `peak_bytes_in_use` is monotone: read the per-iteration `in_use` deltas as well as the peak;
- add `estimate.host_arrays` to the record (the script predates `3346088`); wrap
  `_noncollinear_magnetization` (`driver.py:1368`), not `_magnetization`.

**A17's 11 GB card offset has a cheaper test.** A16 records the same solve asking 33.44 GiB on the
H100 and 23.18 GiB on the A100 at the same dials; the difference, 10.26 GiB = 11.02 GB, is A17's
offset to two digits. `tools/gpu/davidson_memory.py INPUT --pseudo-dir DIR --david 2 --band-batch
16 --nbnd 403 --k-batch 1 --json` compiles without executing, so a minutes-long pair on an A100
(`gpu-debug`) and an H100 names the offset or rules the executable out. Predictions before it
runs: H100 minus A100 about 10.3 GiB if A17 is the executable, and both about 3.8 GiB below
September's 23.18 and 33.44 (the second coefficient fell from 6.20 to 4.20, `sizing.py:94-98`).

**Then one long H100 job, two arms** (`rebuild` first, `store` second, the same-node pair A17
says has never been run), 14 or more iterations each, about 4 h after `--test-only`, with
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.97`, preallocation at its default, every `DEFUMAT_*` dial
unset so the script's flags are the only dials (`--k-batch 1 --band-batch 16 --david 2
--wfc-store device --memory-mode memory --projectors rebuild|store`). An A100 arm of two
iterations only if the compile pair does not name the offset. A 12 GB card cannot answer it.

**Stream D's changes explain none of the 28 GB**: the slab is `nosym` (no maps); the `(nat,
npad)` structure factor was in both run and model; the Anderson history and `site_moments`'
array are host memory; the torque tape postdates the run. Compare today's stage table with
today's `Calculator.estimate()`; A13's 65.47 GB (A100) and 76.51 GB (H100) at `rebuild` are the
only historical anchors, read with the 4 GB solver change between. The plan names no input path,
pseudopotential directory or commit; the job has to.
