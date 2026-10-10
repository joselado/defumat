# Making the Davidson solver faster and more stable: what the literature offers

This file collects what arXiv has on speeding up and stabilising a block Davidson eigensolver
in a plane-wave code, mapped onto `defumat/solvers/davidson.py` and onto the Davidson entries
that are still open (`OPEN.md`, `GPU-SPEED-NEXT.md`, the step count against `pw.x` in
`PERFORMANCE.md`). It is a survey taken on 2026-10-10 by reading the code, the project's notes
and the papers. **Nothing in it was run**, so every effect it expects on defumat is a forecast
until a `davidson_steps` count says otherwise.

The strategies are ranked by how directly they address a problem this code is known to have.
For each one the file gives what it changes in the algorithm, where in the code it lands, what
the papers measured, what it costs, and the arXiv references behind it.

## The collapse condition is not `cegterg`'s

The most useful finding is not in any paper, and it was checked in the source on 2026-10-10.
defumat collapses its subspace when

```python
full = nbase + nbnd > nvecx          # defumat/solvers/davidson.py:750
```

while QE refreshes when

```fortran
nbase+notcnv > nvecx                  ! cegterg.f90:581 and :1128, regterg.f90:497
```

meaning that defumat makes room for a whole block of `nbnd` corrections where `pw.x` makes room
only for the `notcnv` bands that are still unconverged. At `diago_david_ndim = 2` we have
`nvecx = 2 nbnd`. After one expansion `nbase = nbnd + notcnv`, so `nbase + nbnd > nvecx` holds
for any `notcnv > 0`, and the solver collapses every second step regardless of how many bands
are left. It is then a restarted block steepest descent. `pw.x` keeps roughly
`(nvecx - nbnd)/notcnv` blocks of history, which is ten blocks when 100 of 1020 bands are
still unconverged. At `david = 4` defumat keeps at most three expansions, where QE keeps
`3 nbnd/notcnv`.

The code says `nbnd` because of the static write at `davidson.py:807`. It puts a full
`(nbnd, length)` block at `(nbase, 0)` with `dynamic_update_slice`, which clamps its start
index rather than failing, so past that bound it would silently overwrite live rows.
`PERFORMANCE.md:3542` and `OPEN.md:1506` both state the `nbnd` form as `cegterg`'s; that is
not what `cegterg` does.

This bears on three open items:

- the slab's 12x against `pw.x` ("The Davidson step count against `pw.x` on a 157-atom slab" in
  `PERFORMANCE.md`, run at `david = 2`);
- `OPEN.md` Part VII item 2, where 70 or more inner steps were seen at `ethr` 2.3e-6;
- the hard restart at 20, 8 and 4 steps tried in `GPU-SPEED-NEXT.md`, which did nothing; it
  restarts more often, and the defect here is restarting too often.

## Strategies, ranked

**1. Collapse on `nbase + notcnv > nvecx`, as `cegterg` does.**
- What it changes: the size of the history kept between collapses, which then follows the number
  of unconverged bands rather than the whole block.
- Fix (a), exact: pad `psi`, `hpsi`, `becp` and `becq` by `nbnd` rows and use QE's condition as
  written. On the slab this costs one more `(nbnd, npwx)` block for each of `psi` and `hpsi`,
  5.8 GiB apiece by the module docstring's own figure.
- Fix (b), no extra memory: pick the band rung covering `notcnv` before the `cond`, from the
  ladder `live_block` already uses (`davidson.py:839`), and test `nbase + rung > nvecx`. The
  write at `:807`, the `active` update and `_extend_projection`'s block width then move inside
  that `lax.switch`. This reproduces QE's condition to within one rung wherever `narrow` is
  true, and falls back to today's behaviour on an unnamed `vmap` chunk.
- Expected effect: none on cells that converge in two or three steps without reaching a
  collapse; the most on cold or loose calls at `david = 2` with few bands left unconverged. No
  paper gives a number for this case. KSSOLV 2.0 (2206.01993) describes the same trade between
  the subspace limit and the number of restarts.
- The test: `davidson_steps` from `tools/parallel/time_scf.py` at `david = 2` on a cell with
  empty bands, against the nine-row `pw.x` trace in `PERFORMANCE.md`.

**2. A locally optimal restart (GD+k) at the collapse.**
- What it changes: `refresh()` (`davidson.py:754`) keeps only the `nbnd` Ritz vectors and writes
  `diag(e)` against the identity (`:785`). GD+k also keeps, for the unconverged roots, the
  previous step's Ritz vectors or their difference, which is LOBPCG's search direction `P`. That
  restores the conjugate-gradient memory a restart throws away. The collapse onto the Ritz
  vectors is already a thick restart, so adding a thick restart is not a separate strategy.
- What it means for the open items: candidate 1, the hard restart, points the wrong way. The
  right restart keeps more history, not less.
- Evidence: the PRIMME authors call "+k" restarting crucial (1607.01404); Nottoli, Levitt and
  co-workers find Davidson best for most problems and LOBPCG competitive when memory is tight
  (2305.06668); DFTK's Davidson beats its LOBPCG by 5 to 11 per cent of SCF time (2608.24529).
  No plane-wave figure for Davidson with and without "+k" was found.
- Cost: carry the previous `(nvecx, nbnd)` coefficients, which are small, and compute one cross
  block of `hc` and `sc`. It needs room for `notcnv` more rows after the collapse, so at
  `david = 2` it needs fix (a)'s padding. Static.

**3. Buffer bands on a fixed-occupation insulator.**
- What it changes: solve `nbnd + g` bands and give the `g` extra rows a threshold that never
  binds, through the per-band `ethr` vector that already exists. Band `nbnd` then converges at a
  rate set by its gap to band `nbnd + g + 1`, not by the edge of the block.
- What it targets: the stubborn top band that `davidson.py`'s own docstring names. It applies on
  exactly the cells where `btype` does nothing because `nbnd = nelec/2` (`si8-1k`, `si16-1k`).
- Evidence: Vecharynski, Yang and Pask add 1 to 5 per cent extra columns and call the drop in
  iterations substantial, with no single number (1407.7506, §5.3).
- Cost: `h_psi` grows by `g` bands, and the run departs from `pw.x`, so a step count against
  `pw.x` is no longer like for like.

**4. A guard against stagnation.**
- (a) Floor the test at round-off. The test at `davidson.py:638` asks the change in an
  eigenvalue to fall below `ethr`, but an eigenvalue cannot be resolved below roughly `eps` times
  the norm of the reduced matrix. That norm is already computed, as the Gershgorin bound in
  `subspace._park_above`, so `max(ethr, c eps bound)` stops a call at round-off rather than at
  `MAX_ITERATIONS` (`:185`). This is the backward-stable residual criterion of Duersch and
  co-workers (1704.07458, eq. 9), which removed the failures of the original codes in their
  Table 2.
- (b) Recompute `H|evc>` now and then. The refresh stores the rotated `hevc` (`:779`). PRIMME
  traced stagnation at `sqrt(restarts) ||A|| eps` to that rotation and cured it by recomputing
  `A V` (1607.01404).
- What it targets: `GPU-SPEED-NEXT.md` item 7 (a `conv_thr` tighter than 1e-10, handled today by
  the global `DEFUMAT_ETHR_MIN`) and the static float32 floor. The stall on the card is solved,
  so this is a guard against a recurrence rather than the fix for a measured defect.
- Cost: (a) is a few scalars a step; (b) is one `h_psi` of `nbnd` bands per reset.

**5. Orthonormalise each correction block with shifted CholeskyQR2.**
- What it changes: every solve now reduces the basis overlap by Cholesky (`_cholesky_route`),
  with canonical orthogonalisation as a retry outside the batch (`davidson_eigensolver_all`) and
  the `orthonormality_defect` check behind it. Orthonormalising each new block in the `S` metric
  keeps the overlap at the identity, so the subspace problem becomes a standard `eigh`.
- Evidence: shifted CholeskyQR3 works in an oblique inner product up to a condition number of
  `1/u` (1809.11085); DFTK shifts by `100 ||X|| eps` and retries, at most four factorisations,
  and has not seen it fail (2305.06668); Duersch's `cond(R)^-3` test says when the plain factor
  is safe (1704.07458).
- What it targets: the retry on an indefinite overlap, the `NaN` at 64 atoms, and the 2.85x that
  a batched `cond` costs, since a factorisation that cannot fail needs no `cond`.
- Cost: roughly two more `(nvecx x nbnd x npwx)` products a step. DFTK measures
  orthogonalisation at 4.9 of 158.6 s on a CPU and 3.2 of 21.0 s on a GPU (2608.24529). It buys
  stability, not speed.

**6. Mixed precision inside the solve, not across the SCF.**
- What the literature does: the opposite of defumat's `'mixed'`. Gubler and co-workers keep the
  full space, `H V` and the residuals in float64 and put only the small sketched subspace in
  float32 (2608.24529). R-ChFSI applies the float32 or TF32 products to residuals, reaches 1e-8,
  and makes the whole solver 2.1x faster on a GPU (2503.22652). ELPA and BEAST start in single
  precision and switch to double, for 1.2x to 1.4x (1806.01036).
- Why `'mixed'` failed here: it switches the whole band side, and the perturbation of the
  mixer's history loses the superlinear Broyden step.
- A proposal, neither published nor measured: apply `H` to the correction block in float32 and
  keep the retained block and the projected matrices in float64. The error then enters weighted
  by a coefficient of the order of the residual, so it shrinks as the call converges, provided
  4(b) resets `H|evc>` in double.
- Where it could pay: only on a float32-class card. On the H200, float32 bands gave 0.99x to
  1.20x.

**7. Fewer Rayleigh-Ritz steps for large cells on a card: PPCG or Chebyshev filtering.**
- What it targets: `GPU-SPEED-NEXT.md` item 7, where the subspace `eigh` costs 1.1 s at
  `m = 2048` and 7.4 s at `m = 4096` a step on the A2000, and the slab's 1020 bands. It buys
  nothing at sixteen atoms.
- PPCG does a Rayleigh-Ritz every 5 to 10 iterations and is about 2x faster than Davidson
  inside QE on Li318 and Graphene512 (1407.7506). QE ships it
  (`KS_Solvers/PPCG_legacy/ppcg_k.f90`), so it is a transcription.
- Chebyshev filtering is a fixed-degree `fori_loop`, static by construction (cond-mat/0703239).
  In plane waves it is competitive on some systems and slower on others, for lack of a
  preconditioner (1406.4350). Abinit finds Rayleigh-Ritz dominating its GPU time (2604.11139);
  no single LOBPCG-against-filter timing was found in that paper.
- Cost: a new solver each, weeks of work.

**Lower priority, with no open problem behind them.**
- The `ethr` schedule is already QE's (`driver.next_ethr`), and small cells match `pw.x` step
  counts. KSSOLV 2.0's tolerance tied to the density residual is the same idea (2206.01993).
- The preconditioner: no plane-wave paper was found that beats a kinetic diagonal with numbers.
  `OPEN.md` M5 is a cost item, and 1308.2445 concerns a nonorthogonal basis.
- RMM-DIIS and ParO are in QE's tree but address nothing on defumat's lists.

**Differentiability.** No file that calls the solver contains `jax.grad`, `jax.jvp` or
`jax.vjp`, and the only gradient handling inside the solver is the `stop_gradient` in
`_park_above`. What binds the solver is therefore `jit` compatibility (`lax` loops, static
shapes), and every strategy above keeps it.

## References

Each was opened in full text unless marked otherwise.

- 2608.24529, Gubler, Park, Bussy, Laute, Marzari, Timrov, Grigori, "Randomized Block Davidson
  Eigensolvers for Plane-Wave DFT": DFTK's Davidson against its LOBPCG, the cost of
  orthogonalisation, the float32 subspace, soft locking.
- 1704.07458, Duersch, Shao, Yang, Gu, "A robust and efficient implementation of LOBPCG": the
  backward-stable criterion, the `cond(R)` test, SVQB, the failures in Table 2.
- 2305.06668, Nottoli, Giannì, Levitt, Lipparini, "A robust, open-source implementation of the
  locally optimal block preconditioned conjugate gradient": the shifted Cholesky recipe DFTK
  uses, locking, Davidson against LOBPCG.
- 1809.11085, Fukaya, Kannan, Nakatsukasa, Yamamoto, Yanagisawa, "Shifted CholeskyQR for
  computing the QR factorization of ill-conditioned matrices": shifted CholeskyQR3 up to
  `O(1/u)`, oblique inner products.
- 1607.01404, Wu, Romero, Stathopoulos, "PRIMME_SVDS": stagnation from the rotated `AV` at
  restarts and its cure; "+k" restarting.
- 1407.7506, Vecharynski, Yang, Pask, "A projected preconditioned conjugate gradient algorithm
  for computing many extreme eigenpairs of a Hermitian matrix": infrequent Rayleigh-Ritz, buffer
  vectors, about 2x inside QE.
- cond-mat/0703239, Zhou, Saad, Tiago, Chelikowsky, "Parallel self-consistent-field
  calculations via Chebyshev-filtered subspace acceleration": Chebyshev filtering inside the SCF.
- 1406.4350, Levitt, Torrent, "Parallel eigensolvers in plane-wave density functional theory":
  Chebyshev filtering in plane waves, the cutoff as the upper bound, slower without a
  preconditioner.
- 2604.11139, Lygatsika, Sarraute, Baguet, Kestener, Torrent, "GPU acceleration of plane-wave
  DFT in Abinit": Rayleigh-Ritz dominates on a GPU.
- 2503.22652, Kodali, Ramakrishnan, Motamarri, "Residual-based Chebyshev filtered subspace
  iteration": float32 products applied to residuals, 2.1x.
- 1806.01036, Alvermann, Basermann, Bungartz, Carbogno, Ernst and co-workers, ELPA-AEO and
  ESSEX-II: single precision first and then double, 1.2x to 1.4x.
- 2206.01993, Jiao, Zhang, Wu, Wan and co-workers, "KSSOLV 2.0": adaptive per-eigenpair
  tolerance, the trade between the subspace limit and restarts.
- 2409.05572, Liu, Ma, Shao, "Shrink-and-expand technique for symmetric block eigensolvers":
  20 to 30 per cent; not used above.
- 1702.04045, Pan, Dai, de Gironcoli, Gong, Rignanese, Zhou, ParO in plane waves.
- 1308.2445, Cai, Bai, Pask, Sukumar, hybrid preconditioning: cited only to rule it out.
- 2104.10502, Giannozzi and co-workers, "Quantum ESPRESSO toward the exascale": opened, nothing
  quantitative on eigensolvers.
- 2412.01695, Gong, Dal Corso: abstract only, not used.

## What could not be found or verified

- Not on arXiv, so not opened: Stathopoulos's GD+k papers (SISC 2007, William & Mary report
  WM-CS-2005-03), Zhou's block Chebyshev-Davidson (JCP 2010), Hetmaniuk and Lehoucq,
  Teter-Payne-Allan, Davidson-Liu, Stathopoulos and Wu's SVQB, and Kresse and Furthmüller's
  RMM-DIIS.
- No arXiv listing found for Feng and co-workers' 2024 PWDFT eigensolver comparison, for any
  paper on QE's own RMM-DIIS or `cegterg`, or for DFTK's adaptive diagonalisation tolerance,
  which is therefore unverified.
- No published figure for a plane-wave Davidson with and without "+k", or comparing QE's
  `g_psi` with Teter-Payne-Allan.
