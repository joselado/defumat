# The noncollinear ultracell's slow cases on the Kramers-closed basis (`OPEN.md` Part VI item 3)

*Written 2026-10-09 on branch `open-a`, for a fable review before any code. Nothing below is
implemented. The scripts are in `plans/open-a-scripts/` (run on D22 through `run.sh`, which sets
the environment and the 6 GB cap); their output is in `/l/ladovj1/review/open-a/` on D22.*

## What was measured

Four cells of hydrogen (`_noncollinear` of `tests/regression/test_ultracell.py`, simple cubic
`a = 5.5`, `kgrid = (1, 2, 2)`, `conv_thr = 1e-10`, `states_conv_thr = 1e-8`, reference
converged at `nbnd = 8`), on D22-0161 (`review/open-a/vi3.py`, `vi3b.py`). The old basis is
`kramers_pairs=False` at `nbnd = 16`; the closed basis, the default since `08db351` (P121), is
`nbnd = 8` with the partners, the same 16 states per folded k-point.

**The seeded helix, the entry's target pair**, `mixing_beta = 0.3`, seed turning about `z`:

| basis | reference along | iterations | energy, Ry |
|---|---|---|---|
| old | `z` | **14** (reproduced) | -0.9333093849 |
| old | `(1,1,1)/sqrt(3)` | **not converged in 400** (the entry's 290 is not reproduced) | -0.9333034848 |
| closed | `z` | **10** | -0.9337454218 |
| closed | `(1,1,1)/sqrt(3)` | **10** | -0.9337454218 |

On the closed basis the two frames give the same cell moments to **4.1e-9** after fitting one
global rotation, and the run with the reference along `(1,1,1)/sqrt(3)` comes back in the
*seed's* frame, with no rotation to traverse at all: the closed span is invariant under every
global spin rotation, so no frame is preferred and the seed's is a fixed point. On the old basis
the same alignment leaves 0.036 on moments of 0.33. **The flat-manifold traversal this entry was
opened for does not exist on the default basis.**

**The weak turning field, the entry's other case** (`magnetic_field = 0.002 (cos, sin, 0)` Ry
turning 90 degrees per cell about `z`, reference along `(1,1,1)/sqrt(3)`, no seed, which is how
the 2026-09-15 note reads the original's undocumented field):

| basis | `mixing_beta` | iterations | where it ends |
|---|---|---|---|
| old | 0.7 | 146, converged | net moment 0.031, cell moments of length 0.05 to 0.10 |
| old | 0.3 | not in 400 | net 0.070 |
| closed | 0.7 | **not in 400, not in 1500** | `dr2` between 1e-8 and 1e-7 with excursions to 3e-5; cell moments of length 0.61, 0.17, 0.08, 0.57, net 0.33 |
| closed | 0.3 | not in 400 | net 0.19 |
| closed, **seeded** with the helix | 0.7 | **9** | a flat helix of length 0.347, net below 1e-5 |
| closed, field 0.01 Ry, no seed | 0.7 | 55 | a flat helix of length 0.489, net 3.6e-5 |

In the 1500-iteration run the charge and the magnetic halves of `dr2` stay within a factor of
three of each other throughout, so it is not the shape of a magnetic soft direction being
traversed either; it is a limit cycle among textures of uneven moment length.

## What that says

The rigid rotation has no restoring force and the closed basis lets every frame be a fixed
point, so a seed is kept where it is put, which is better than traversing; and what is slow
under a weak field is **leaving the tiled ferromagnet**, which the seed does in one step (9
iterations against no convergence in 1500). The real four-atom supercell has the same bill:
under the 0.01 Ry ladder, from a ferromagnetic seed, it did not converge in 400 iterations
(this entry, 2026-09-15). So the projection the entry proposed, the rigid rotation taken out of
the magnetic residual, has no target in the ultracell on the default basis, and on the weak
field it would act on the wrong thing, since what moves there is the moments' length, not
their common direction. `MAGNETISM-NEXT.md` F2 keeps the generator form of that projection for
the ordinary SCF, where none of this was measured.

## What the change would be

Only the advice, which is wrong on the default basis in the one place a user reads it:

1. `defumat/ultracell/driver.py`, the `rigid` half of the non-convergence warning (now at lines
   1627-1650): it says the run is traversing a flat manifold, to raise `max_iterations` or the
   field, and **not** to lower `mixing_beta`, with 263 against 300 from the old basis. On the
   default basis neither `beta` converges this cell in 400 and a seed converges it in 9, so the
   warning should name `seed_magnetization` with the texture the field asks for as the first
   remedy, keep the beta advice only for `kramers_pairs=False`, and quote the numbers above.
2. The same paragraph in `docs/features.tex` (around line 7316, the 263) and `PERFORMANCE.md`
   (around line 6481), each carrying the old basis's numbers as the advice.
3. One test: a `kramers_pairs=True` ultracell that fails to converge under a weak turning field
   warns with `seed_magnetization` in the message, and the same with `kramers_pairs=False`
   still carries the beta advice; and a slow regression pinning the seeded weak-field run at
   10 iterations or fewer, so that the advice stays true.

## How to know it worked

The warning's text is what changes; the numbers do not, so the check is the two message
tests firing on the two bases and the seeded run's iteration count, 9 on D22 at `aa08c82`.

## Not decided here

Whether `run_ultracell` should seed by itself when a `magnetic_field` is given and no seed is,
taking the field's own direction as the seed. That changes what a default run computes and is
the user's.
