# P72 checkpoint (Elk STATE.OUT -> defumat SCF seed) — 2026-09-09

Untracked scratch note. Written mid-task so a kill costs the step in flight, not the account
of what was done.

## Files created / modified

| path | state |
|---|---|
| `defumat/io/elk.py` | **working.** `ElkState`, `read_elk_state`, `ElkGeometry`, `read_elk_geometry`. Smoke-tested. |
| `defumat/io/elk_density.py` | **working.** `evaluate_at`, `muffin_tin_charges`, `characteristic_function`, `interstitial_charge`, `sharp_interstitial_charge`, `interstitial_coefficients`, `density_on`, `SeedReport`, `real_spherical_harmonics`, `poly4`, `spline_weights`. All validated below. |
| `tests/data/elk/h_sc/` | **complete.** Frozen copy of elkpy's fixture: `elk.in H.in GEOMETRY.OUT INFO.OUT STATE.OUT RHO3D.OUT regenerate.sh README.md`, 704 KB, uncompressed. Plus `scf.in` written here (the matching defumat input). |
| `tests/data/elk/h_sc/scf.in` | **working.** sc H, a=3.0 bohr, `H.pz-vbc.UPF`, `ecutwfc=40`, gaussian smearing `degauss=0.02`, 4x4x4, `conv_thr=1e-10`. `pseudo_dir` must be passed to `Calculator.from_file`, not written in the file. |
| `defumat/calculator.py` | **working.** One method added, `Calculator.get_elk_seed(directory, renormalise=True, report=None)` at line ~512, immediately before `_ground_state`. A pure one-line delegation to `ElkState.read(...).density_on(...)`; it is under the 30-line budget `tests/unit/test_calculator.py::test_every_get_method_is_a_delegation` enforces. **Nothing else in `calculator.py` was touched.** |

**Every file above is in a working, self-consistent state. Nothing is half-written.**

**Not yet started:** tests (unit + regression), `PLAN.md` P72, `README.md` row,
`docs/features.tex` entry, notebook 42, `PERFORMANCE.md`, `defumat/io/__init__.py`
exports. **No commit has been made** and `tools/test-fast.sh` has not been run.

## Numbers already verified against `tests/data/elk/h_sc`

- **Origin equality (the one-line reader check):** `rhomt[0,0,0] * y00 = 0.2859303011723094`
  against `RHO3D.OUT`'s first value `0.2859303012`. Exact.
- **Pointwise against `RHO3D.OUT`, all 4096 points:** max abs err **4.52e-11**,
  max rel err **3.55e-10**. That is the file's own `G18.10` print floor, not an
  interpolation residual — `poly4` is replicated. 1743 of 4096 points inside the sphere.
- **`chgmt`:** 0.61257620 against `INFO.OUT`'s 0.6125761996. Exact to the printed digits,
  using Elk's own `wsplint`/`splint` spline weights (transcribed).
- **`chgir`, Elk's own way** (closed-form `cfunir` from `gencfun.f90`'s header, then
  `(Omega/ngtot) sum rhoir*cfunir`): **0.38742380044346** against printed `0.3874238004`.
  Exact. So the closed-form characteristic function is right.
- **`chgir`, sharp sphere test on the same 12^3 grid:** **0.38465939570920**.
  Sharp-versus-truncated difference = **2.7644e-3**, i.e. **0.714%** of `chgir`.
  (Two effects mixed: Gibbs ringing of the truncated step, and a 12^3 grid staircasing
  a sphere. An order of magnitude, not a calibrated Gibbs measurement.)
- **Header:** version (11,0,2), `ngridg=(12,12,12)`, `ngvec=751`, `nrmt=197`,
  `lmmaxo=49`, `rmt=1.4`, `efermi=0.0739705280 Ha` -> 0.1479410560 Ry.
- **`nri = 129` recovered** from the zeroed high-`lm` tail (`r(nri) = 0.014062` bohr,
  1% of `rmt`), even though `STATE.OUT` never writes it.
- **Seed on defumat's grid** (15x15x15 dense, `ecutrho = 160` Ry):
  integrates to **1.000298** electrons before renormalisation against `nelec = 1`.

### The identity — the phase's most important number. **It holds.**

`tests/data/elk/h_sc/scf.in`, `conv_thr = 1e-10`, `ecutwfc = 40`:

```
atomic superposition : E = -1.080181442650 Ry   4 iterations   acc 2.9e-12
Elk seed             : E = -1.080181442650 Ry   4 iterations   acc 1.0e-11
dE = 6.35e-14 Ry ;  max |drho| between the two converged densities = 8.66e-08
```

So the transfer is a **seed** and not a perturbation: the two runs land on the same
state to well inside the SCF tolerance.

### What the seed buys, in iterations: **nothing. A null result, as expected.**

| `ecutwfc` | `ecutrho` | dense grid | atomic | Elk seed | dE (Ry) |
|---|---|---|---|---|---|
| 12 | 48 | 8^3 | 5 | 5 | 4.4e-16 |
| 20 | 80 | 9^3 | 4 | **5** | 3.7e-12 |
| 40 | 160 | 15^3 | 4 | 4 | 6.4e-14 |

At `ecutwfc = 20` the Elk seed costs **one extra iteration**. This is the effect the task
predicted: for one hydrogen atom defumat's superposition of atomic charges is already
almost the answer, and the all-electron cusp inside `r_c` is a place where the Elk density
is *further* from the pseudo one than the atomic guess is. Report it as measured; the
payoff case is magnetic and is stage 4.

### Aliasing: how much norm sits past defumat's dense sphere

Fraction of `sum |rho~(G)|^2` (over Elk's own `ngvec = 751` set) outside `ecutrho`:

| `ecutwfc` | `ecutrho` | G kept of 751 | outside fraction | integral before renormalisation |
|---|---|---|---|---|
| 12 | 48 Ry | 147 | **6.72e-08** | 1.002946 |
| 20 | 80 Ry | 341 | **2.50e-13** | 1.001697 |
| 40 | 160 Ry | 751 | **0.0 (nothing truncated)** | 1.000298 |

**`ecutrho = 160 Ry` is larger than Elk's `gmaxvr = 12 bohr^-1` (= 144 Ry)**, so at the
production cutoff the dense sphere contains the whole Elk G-set and there is no truncation
at all. The number only becomes nonzero below `ecutrho = 144 Ry`, and even at 48 Ry it is
7e-8 -- because what is being truncated is the *interstitial* field, which is smooth. **The
nuclear cusp is not in this number**: it lives inside the muffin tin, which is transferred
pointwise and never passes through a Fourier truncation at all. That is worth saying
explicitly in `PLAN.md`, because it is the opposite of what the phase brief anticipated.

### Timings (single core, one H in a 3 bohr cube)

- read `STATE.OUT` + `GEOMETRY.OUT`: **0.021 s**
- reconstruct onto the 15^3 dense grid: **0.128-0.16 s**
- pointwise `evaluate_at` at 4096 `RHO3D.OUT` points: order 0.2 s
- SCF from the atomic superposition: 2.5 s wall, but that includes the XLA compile;
  the seeded run right after it took 0.2 s on warm caches. **These two are not a fair
  pair and must not be quoted as one** -- re-time in separate processes for
  `PERFORMANCE.md`.
- no reference-code pair is owed: Elk has no counterpart to "read Elk's state into a
  plane-wave code", and that is what `PERFORMANCE.md` should say rather than leaving the
  absence unexplained.

## Coordinator assertions — in place or not

| assertion | state |
|---|---|
| record length `== (lmmaxo*nrmtmax*natmtot + ngtot)*8`, as an **equality** | **in place**, raises with the two-arrays-one-record explanation |
| origin equality as the first thing asserted | **verified in scratch, not yet a test** |
| `natmtot = sum(natoms)`, not `natoms(1)` | **in place**, with the "untested by both fixtures, one species each" comment |
| `[:, :nrmt(is), ias]` slicing, padding is garbage | **in place**, same comment |
| `rhonorm` correction (`chgmt+chgir = 1` exactly; the 7.4e-4 is pre-shift) | **understood, not chased.** `chgmt` asserted against the post-shift 0.6125761996 |
| `rhoir` on the fine grid, `ngtot = prod(ngridg)` | **in place** |
| `nelec` expressed as the *run's own* count, never Elk's total | **in place** — `density_on` uses `calculation.nelec` and refuses a >50% gap by name (the core case) |
| `ias` ordering explicit rather than incidental | **partly** — `geometry.species_of()` is species-outer/atom-inner; needs a unit test on a synthetic two-species geometry, since h_sc cannot catch it |
| `nspin_mag != 1` refused in `density_on` | **in place** |
| a *fixed* Elk density (no defumat SCF on top) refused by name | **in docstrings only** — there is no code path that would do it, so there is nothing to guard; say so in `PLAN.md` rather than inventing a flag to reject |
| spline weights rather than Simpson | **in place** (`spline_weights`, transcribing `wsplint`/`splint`) |

## Exact next step, for someone picking this up cold

The physics and the code are **done and measured**. What is left is the paperwork the
project requires, in this order:

1. **Export the reader.** Add `ElkState`, `read_elk_state`, `read_elk_geometry` to
   `defumat/io/__init__.py`'s imports and `__all__`. One edit, nothing subtle.
2. **Tests.** Two files, and the fixture is already committed at `tests/data/elk/h_sc/`,
   so nothing needs an Elk binary:
   - `tests/unit/test_elk_reader.py` (fast): the origin equality
     `rhomt[0,0,0]*y00 == RHO3D.OUT[0]` exactly; the header values (version, `ngridg`,
     `ngvec`, `nrmt`, `lmmaxo`, `rmt = 1.4`); `efermi` in **Ry** (0.14794105596912),
     i.e. the Hartree->Ry factor is on the potentials and not on the density; `nri == 129`;
     `real_spherical_harmonics` against closed forms for `l <= 2`; `spline_weights`
     integrating a polynomial exactly; **a synthetic two-species `ElkGeometry` asserting
     `species_of()` is species-outer/atom-inner**, which no fixture can catch; and the
     four refusals (`spinpol`, `dftu`/`ftmtype`, missing `GEOMETRY.OUT`, `nspin_mag != 1`)
     raising by name. Every test must `pytest.skip` if `tests/data/elk/h_sc` is absent.
   - `tests/regression/test_elk_seed.py` (mark `slow`): the 4096-point comparison against
     `RHO3D.OUT` at 1e-9; `chgmt` against 0.6125761996; `chgir` against 0.3874238004; and
     the identity (seeded vs atomic total energies agreeing to < 1e-10 Ry). Run it with
     `DEFUMAT_TEST_MEM_MAX=4G tools/run_regression.sh tests/regression/test_elk_seed.py`.
3. **`PLAN.md` P72** — every number in the section above, the traps in "Learned" below,
   and a "What is outstanding" naming: **stage 4** (spin: an Elk magnetization is a second
   field; spirals are spin-polarized Elk runs and fall under the same refusal); **stage 5**
   (heavier elements -- `rhomt` includes the core and `STATE.OUT` alone cannot separate it,
   so the route is a *difference* transfer against a one-iteration Elk state; the
   coordinator has decided the test case is **diamond**, two C at the `tests/data/qe/
   diamond.in` geometry, PBE, with both a converged and a first-iteration `STATE.OUT`);
   **DFT+U**; and **a fixed Elk density**, refused because Elk's all-electron density is
   not defumat's pseudo valence density and no SCF-free splice of the two is meaningful.
4. **`README.md`** — one row. `QE` blank (`pw.x` reads its own `charge-density.dat`, never
   Elk's; grep the vendored tree to confirm before setting it). `Elk` blank (Elk reads its
   own `STATE.OUT`, which is not the same claim as "Elk can seed a plane-wave code"; check
   `docs/elk_manual.txt` §5.127 before setting it). **Re-read `README.md` immediately
   before editing** -- another subagent is appending to it.
5. **`docs/features.tex`** — what it computes, `Calculator.get_elk_seed` and
   `defumat.io.elk.ElkState` checked by `grep`, a snippet that has actually been run
   (the two-line seed-then-SCF above is the one), and an amber box carrying the four
   refusals. Build twice with `xelatex docs/features.tex`.
6. **Notebook `42_...`** — physics only. The figure is Elk's all-electron density against
   defumat's converged pseudo density along a line through the cell, which shows exactly
   where the two must agree (the interstitial) and where they must not (inside `r_c`).
   Use `Calculator.get_elk_seed` rather than internals. Add `"42_..."` to `REWRITTEN` in
   `tests/unit/test_notebook_conventions.py` in the same commit. Re-export with
   `tools/export_notebooks.sh notebooks/42_*.ipynb` (it takes a single notebook).
   No em dashes, no phase numbers, no Fortran file names.
7. **`PERFORMANCE.md`** — the timings above, re-taken in separate processes so the compile
   is not counted on one side only, plus the sentence saying no reference pair is owed.
8. **Then** `tools/test-fast.sh`, and commit. **Delete this checkpoint file in that
   commit.**

Two saved scratch arrays, if useful: `rho_elk.npy` (the seed on the 15^3 grid) and
`rho_conv.npy` (defumat's converged density) under the session scratchpad. They are
cheap to regenerate; do not depend on them.

## Learned, not yet written into a tracked file

- `zfftifc(3, ngridg, -1, z)` is **`np.fft.fftn(z)/ngtot`** (FFTW sign -1 = forward, then
  Elk scales by 1/N). The `rfpts` sum is then `sum_G c(G) e^{+iG.r}`.
- The Miller range is Elk's `intgv = [n/2-n+1, n/2]` (for n=12: -5..6), which is **not**
  `np.fft.fftfreq`'s -6..5. Selecting "first `ngvec` sorted by |G|" is safe because
  `gengvec` cuts at the first vector past `gmaxvr`, so no degenerate shell is ever split
  (measured: kept |G|max = 11.848, next = 12.031, `gmaxvr = 12`).
- Fortran `nint` is **half away from zero**; `np.rint` is half to even. The radial index
  `nint(t1 log(r/rmin))+1` must use `floor(x+0.5)`, or a different `poly4` window is picked
  on exactly the boundary points and the residual stops being round-off.
- `rfpts` truncates the harmonic sum to `lmmaxi` when the interpolation window *starts*
  inside the inner region (`ir0 <= nri`). Neither `nrmti` nor `lmmaxi` is in `STATE.OUT`,
  so this code always sums `lmmaxo`. The two agree identically except on a three-point
  window straddling `nri` (here `r ~ 0.014` bohr, 1% of `rmt`), because the file's
  `lm > lmmaxi` entries are **exactly zero** for `ir <= nri`.
- `scipy.special.sph_harm_y(l, m, theta, phi)` matches Condon-Shortley exactly; the
  negative-`m` branch of `genrlmv` is `sqrt2 * Im Y_{l,-|m|}` and must be built from
  `Y_{l,-m} = (-1)^m conj(Y_{lm})` rather than from a guessed sign on `|m|`.
- `GEOMETRY.OUT`'s `scale` / `scale1..3` multiply `avec` and are applied on the way in.
- `Calculator.from_file` ignores a relative `pseudo_dir` written in the input file; pass it
  as a keyword. That is why `tests/data/elk/h_sc/scf.in` has no `pseudo_dir` line.
- `SCFResult` has **`total_energy`**, not `energy.total`.
- **`rhonorm` inverts the fixture README's charge story** (the coordinator corrected this
  mid-task). Elk's `rhonorm.f90`, called from `rhomag.f90:24` with `trhonorm` on by
  default, adds a uniform constant to `rhoir` and to the `l = 0` channel of every `rhomt`
  so the total comes out right, then updates `chgmt` and sets `chgir = chgtot - chgmttot`.
  So `chgmt + chgir = 1` **exactly, by construction**, and `chgmt = 0.6125761996` is
  post-shift and does describe the array in the file. The printed
  `total calculated charge = 1.000739542` and its 7.4e-4 error are **pre**-shift. There is
  no 1e-3 discrepancy to hunt for.
- The origin equality is exact **only at the origin** -- `rfpts` clamps `r` up to `rsp(1)`
  and its window starts at `ir0 = 1` there. Everywhere else it is a 4-point Lagrange
  interpolation, which is why `poly4` is replicated rather than substituted.
- **The aliasing worry was misdirected.** The nuclear cusp is inside the muffin tin, which
  is transferred pointwise; only the smooth interstitial field passes through a Fourier
  truncation, so the truncated norm is 7e-8 at worst and zero at the production cutoff.
- For a target that is a **regular grid**, the interstitial Fourier sum is an inverse FFT
  rather than an `npoints x ngvec` contraction. `density_on` does that and then overwrites
  the muffin-tin points; `evaluate_at` keeps the general chunked contraction for arbitrary
  points.
- **Stage 5 cautions, from the coordinator, not yet written anywhere tracked:** Elk
  re-solves the core states in the current potential every cycle (`rhocore.f90`), so the
  core does **not** cancel exactly in a difference -- measure the residue on carbon before
  relying on the scheme. And Elk mixes the potential in the *middle* of its own iteration
  (`gndstate.f90`'s `mixerifc`), so two states written on opposite sides of that call
  differ by one mixing step; a disagreement at roughly `epspot` is that, not a
  transcription bug. It has looked like a bug four times in the elkpy project.
- elkpy's `docs/design.md` section 34 is the written-down format reference (header record
  order, muffin-tin packing, real-SH convention, `rhonorm`, binary-layout traps, each with
  file and line), plus `tests/test_state_fixture.py` there with 9 binary-free tests. Read
  it before touching the parser again.
