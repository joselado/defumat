# Plan: the preconditioner's block-diagonal `D`, second attempt (OPEN.md Part III M5)

Written 2026-10-09 on branch `open-e`, for a fable review before any code. Nothing here is
implemented.

## History, read off the commits

- `680e01a` (2026-10-03) contracted `h_diag = sum_ij conj(vkb_gi) D_ij vkb_gj` atom block by
  atom block as one broadcast over `(plane waves, atoms, channel, channel)` and one sum. XLA
  fused it on `si8-us-1k` and `si8-paw-1k` (temporaries 8.4 to 6.5 MB), energies within
  6e-14 Ry, step counts unchanged on five cells.
- `dacdd8b` (2026-10-04) reverted it: on `bi10-soc`
  (`test_ten_site.py::test_spin_orbit_coupling_at_ten_sites`, ten Bi atoms, 150 spinor bands,
  216x45x81 grid) the broadcast was **not** fused and the test peaked at 12.66 and 12.72 GB on
  D22 where master reads 3.59 and 3.60. The Hubbard diagonal built on it (`6f366df`) went with
  it. A batched einsum had been measured slower than the broadcast on a CPU, a dot not being
  fused there.

Today (`aa08c82`) the four sites are dense again: `operator.py` `diagonal` and
`overlap_diagonal` (`einsum("gi,ij,gj->g", ...)`), and the spinor twins in
`noncollinear.py` (`:478`, `:492`).

## What the next form has to satisfy

Its largest intermediate must be bounded **by construction**, not by a fusion decision XLA may
or may not take on a given cell: at most the dense form's `(npwx, nkb)` and never
`(npwx, nat, nh, nh)`. The arithmetic gain is `nat`-fold (`npwx nkb nh` products against
`npwx nkb^2`), which is 16x at si16 and 10x on bi10-soc; the dense intermediate is the claim
that scales (6 MB at si16, 703 MB at the 157-atom slab).

## Design

Per species `s` with `nat_s` atoms and `nh_s` channels, contiguous in `vkb` (the layout
`Projectors.atom_of_channel` records):

```
V_s  = vkb[:, cols_s].reshape(npwx, nat_s, nh_s)          # a view, no copy
D_s  = dij_blocks[s]                                      # (nat_s, nh_s, nh_s), gathered once
W_s  = einsum("gai,aij->gaj", V_s, D_s)                   # (npwx, nat_s, nh_s): a batched dot
h_s  = sum_{a,j} real(conj(V_s) * W_s)                    # elementwise and a reduction
```

`W_s` is exactly `(npwx, nkb_s)`, the dense form's own intermediate size, so the peak cannot
exceed today's whatever XLA fuses; the reduction is elementwise and is free to fuse or not.
The species loop is a static Python loop (a handful of species). The `dij` blocks are
gathered from the dense matrix with static indices at Hamiltonian build time, so nothing
traced changes shape. Spinor sites: the same with the `(2 nh)` channel axis that
`noncollinear.py` already uses.

## Checks

1. `bi10-soc` first, on D22, inside `systemd-run --user --scope -p MemoryMax=10G`, the
   single test `test_spin_orbit_coupling_at_ten_sites`, compiled-kernel cache off, both arms
   back to back: the new peak must be at or below master's 3.59 GB. The cap is the guard that
   a repeat of the 12.7 GB failure dies at 137 instead of taking the machine.
2. `memory_analysis().temp_size_in_bytes` of the compiled `_every_k` on `si8-us-1k`,
   `si8-paw-1k`, `si16-1k-ecut30` and `pt-so-1k`, both arms.
3. The converged total to 1e-12 Ry and **the per-k Davidson step arrays identical** on those
   cells and on `bi10-soc`; the entry calls this number-moving, since the preconditioner
   changes in the last bits and a step count is what a last-bit change in a preconditioner
   can move.
4. Time through `tools/benchmark.py` on `si16-1k-ecut30` and `si8-us-1k`, one core.

Accept only if (1) holds and no step count moves; record a null otherwise, as `dacdd8b` did.
