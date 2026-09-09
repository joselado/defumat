# Checkpoint: spin-polarized tip for the vertical transport map (extends P66)

Written 2026-09-09, mid-task, uncommitted. Untracked scratch file; delete after the commit.

## Files touched

All of these are in a **working state** (imports clean, tests run). Nothing is half-written.

| path | what changed |
|---|---|
| `defumat/transport/green.py` | new `spin_transmission(...)` (in `__all__`); module docstring gained the magnetic-tip paragraph; `VerticalTransport` gained `tip_spin` / `tip_polarization` fields |
| `defumat/transport/substrate.py` | `spin_projector(direction, polarization, what="substrate")` -- only the error-message label is new |
| `defumat/workflows/transport.py` | `run_vertical_transport(..., tip_spin=None, tip_polarization=1.0)`; new `_tip_acceptance`; `_collinear_acceptance(..., what=)`; new `_label`; `_assemble` takes the two new arguments and branches on `tip_projector`; a second "identically zero" warning for two opposed fully polarized leads (`open_path`) |
| `defumat/calculator.py` | `get_vertical_transport` docstring only (the options pass through `_call_options` untouched -- verified, it filters by the callee's signature) |
| `tests/unit/test_transport_machinery.py` | +9 tests, 40 pass |
| `tests/regression/test_transport.py` | +5 tests, `_magnet(kind)` `lru_cache(maxsize=2)` helper, `MAGNETS` |
| `tests/data/qe/h-sheet-noncolin.in` | **new, untracked**: the committed magnetic spinor sheet |
| `PLAN.md` | P66 rewritten in place: the tip section, the amended refusal list, "not claimed" left with the s-wave item intact |
| `README.md` | line 207, the existing vertical-transport row edited (no new row) |
| `docs/features.tex` | the vertical-transport subsection: equation, snippet, validation paragraph, amber box |
| `PERFORMANCE.md` | one new row appended at the end |
| `notebooks/41_vertical_transport.ipynb` / `.md` | +1 markdown, +2 code cells. **Executed and exported, done**: 159 s against a 600 s ceiling, 74 code lines against a budget of 80, longest cell 20 against 25, no forbidden vocabulary. New figure `41_vertical_transport_files/41_vertical_transport_12_0.png` (untracked); `..._7_0.png` is the old figure regenerated |

## What passes

* `tests/unit/test_transport_machinery.py` -- **40 passed**, 2.2 s.
* `tests/regression/test_transport.py` -- full file, memory-capped: **23 passed, 1 failed**, 338 s.
  The failure is `test_the_three_spin_regimes_agree_where_there_is_no_magnetization`
  and it is **pre-existing**: reproduced at `HEAD` in a clean worktree with
  byte-identical numbers, `1.7007537e-05` against a `1e-05` tolerance. Not caused
  by this work and not touched by it. It also proves the `tip_spin=None` path is
  bit-for-bit unchanged, since the arrays match to the last printed digit.
* The eight magnet/refusal tests re-run after the committed input landed: **8 passed**, 56 s.
* `tools/test-fast.sh` has **not** been run yet.

## The numbers

* **Tersoff-Hamann limit, magnetic tip**: `exit_region="volume"` with
  `tip_spin=(1,1,1)`, `tip_polarization=0.85` against
  `run_stm(spin=(1,1,1), polarization=0.85)` -- **5.8e-13** relative.
* The y-mirrored tip (what a transposed index order returns) differs by a factor
  of **2** on that cell, so the check has something to catch.
* Tip partition `T(+n) + T(-n) = T_unpol`: **3.1e-16**, with the substrate
  polarized along a third direction at the same time.
* `tip_polarization=0` is exactly half the unpolarized map: **1.5e-16**.
* Real-space quadrature of `int dr' Tr[P_t G Gamma_s G^dag]` against the
  contraction, both leads' projectors off-diagonal: **1e-12** (unit test, no SCF).
* TMR: both leads at `P=0.9`, `T_par/T_anti = 0.189`, a factor **5.3**.
* Timing, one core, affinity set before JAX: spinor H sheet, 16 k, 16 bands,
  30x30 tip plane. 1 energy **1.63 -> 1.65 s** (1 per cent). 41 energies
  **2.39 -> 3.19 s** (1.33x). Per-energy contraction **0.019 -> 0.038 s**, 2.03x.

## The physics, implemented versus planned

**Implemented, and the spinor tip trace is done.** The Landauer trace over the
tip's spin is no longer taken: with the exit integral folded into `S_k`,

    T(r) = sum_k w_k Tr_spin[P_t M_k(r)],   M_k[s,s'] = a_s^T S_k a_{s'}^*

`spin_transmission` builds the full 2x2 `M` in tip-spin space (coherent and
incoherent branches both) and contracts it with `P_t = (1 + P n.sigma)/2`.
Collinear is a per-channel `(1 +- P)/2` weight that multiplies the substrate's.
`nspin = 1` and a transverse direction on a collinear run are refused by name.
Both polarizers work together, which is the new capability.

**Not implemented and deliberately not claimed**: a tip with spatial structure
beyond an s-wave. A magnetic tip is still a point contact; giving `Gamma_t` a
spin structure says nothing about its spatial one. P66's "not claimed" list keeps
that item.

**Not added**: a `+n` / `-n` contrast in `notes`. It would double the cost of
every call to report what two calls already give.

## Exactly which sections of the shared documents were changed

Another agent is working in this tree at the same time (`ELK-P72-CHECKPOINT.md`,
`defumat/io/elk.py`, `defumat/io/elk_density.py`, `tests/data/elk/h_sc/` are
theirs, not mine) and may have touched the same four files. These are the only
places this work edited, so the two sets reconcile without reading the whole diff.

**`PLAN.md`** -- only inside `### P66 - Vertical tunnelling transport through a
two-dimensional material. DONE.`, in three places and nowhere else:
1. the opening file list, where the two test counts went `(25)` -> `(40)` and
   `(17)` -> `(24)`;
2. a **new block of about sixty lines inserted immediately before the paragraph
   beginning `**Refused by name.**`**, starting `**A magnetic tip, and it is the
   coherent half of the same expression.**` and ending with the sentence about
   no `+n`/`-n` contrast being added to `notes`;
3. two sentences rewritten in place: inside `**Refused by name.**` the clause
   `A spin-selective substrate on a run with no magnetization, and a transverse
   direction on a collinear one (P65's reasons, unchanged).` now also names the
   tip; and inside `**Not claimed**` the item `a tip with structure beyond an
   s-wave` gained a clause saying a magnetic tip is still a point contact, so the
   item **stays outstanding**.
No other phase section was opened.

**`README.md`** -- **line 207 only**, the single existing row
`**Vertical tunnelling transport through a 2D material**`. **No row was added**:
per the project rule the table names quantities, and a magnetic tip is a variant
of one already there. Two edits inside that row: a sentence about either
electrode being magnetic replacing the trailing `with a spin-polarized substrate`
clause, and `tip_spin`, `tip_polarization` appended to the entry-point column.
The `QE` and `Elk` tick columns are **unchanged and still correct** -- `(✓)`
with footnote 15 and blank; neither code computes the quantity, so a magnetic tip
changes nothing there.

**`docs/features.tex`** -- only inside
`\subsection{Vertical tunnelling transport through a two-dimensional material}`,
in four places:
1. a new paragraph `\textbf{Either electrode can be magnetic.}` with the
   `Tr[P_t M]` equation, inserted directly after the sentence ending
   `blind to a rotation inside a degenerate multiplet.`;
2. the end of the existing `pycode` block, where a magnetic-tip / spin-valve
   snippet was appended after the `up`/`down` substrate lines;
3. the `\textbf{Validated by identities}` paragraph, extended after the sentence
   `one across the moment takes exactly half.` with the tip's four figures;
4. the `refuses` environment, where the sentence
   `A spin-selective substrate on a run with no magnetization, and a transverse
   direction on a collinear one.` was replaced by one naming both leads.
**The document has not been built.**

**`PERFORMANCE.md`** -- **one row appended at the very end of the file**, dated
`2026-09-09`, beginning `**A spin-polarized tip for the vertical transport map
(P66)**`. No existing line was touched.

**`notebooks/README.md`** -- **not edited at all.** See the outstanding list.

## Exact next step, for someone picking this up cold

The notebook is finished and exported. What remains, in order:

1. **`notebooks/README.md`.** Notebook `41` is **absent from the timing table**
   under `## Running them` (the table runs `00`-`40`); this is pre-existing, but
   the number is now known and should go in: **159 s**, which places it between
   `39` at 151 s and `08` at 171 s. Its one-line description at **line 165**
   still describes only the graphene/bilayer half and should gain the magnetic
   tip; the index entry at line 63 may want a second row for the spin valve.
2. **Verify the `docs/features.tex` snippet by running it.** The script is
   already written at
   `<scratchpad>/snippet.py` -- it is the snippet verbatim with the
   illustrative file names replaced by `tests/data/qe/h-sheet-noncolin.in`.
   Expect `tmr` near **5.3**, `tip_spin = (1.0, 0.0, 0.0)`,
   `tip_polarization = 0.9`. If the scratchpad is gone, the script is six lines
   and the `.tex` block is the specification. **The `.tex` snippet currently
   quotes `# 5.29 on this cell` from a 3x3 tip plane; the notebook at 20x20 gives
   5.31, so check which sampling the 24x24 snippet actually produces and correct
   that comment if it differs.**
3. **`xelatex docs/features.tex` twice** (for the table of contents). Not yet run,
   so the new LaTeX is **unverified** -- the additions use `\code{}`,
   `\textbf{}`, `equation*` and the `refuses` environment, all already used in
   that file, but nothing has compiled them.
4. **`tools/test-fast.sh`.** Not run. Expect it to pass: the only fast tests that
   reach the changed code are `tests/unit/test_transport_machinery.py` (40 pass)
   and `tests/unit/test_notebook_conventions.py` (checked by hand against every
   rule it enforces -- budgets, vocabulary, facade imports, `.md` fence count --
   but **not actually executed**).
5. **Decide what to do about the pre-existing regression failure** described
   above. It is red at `HEAD` and red here with identical numbers. Options: leave
   it and say so, or re-measure and record the drifted tolerance. It should
   **not** be quietly relaxed as part of this change.
6. **Commit**, message ending exactly with:

       Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
       Claude-Session: https://claude.ai/code/session_01Pto6e8KbcTKyuxjs1pWWCn

   Remember `tests/data/qe/h-sheet-noncolin.in` and
   `notebooks/41_vertical_transport_files/41_vertical_transport_12_0.png` are
   **untracked** and must be `git add`ed.
7. **Delete this file.**

## Learned, and not yet anywhere else

* **The index order of `M` is the P54/P66 trap one level up.** `M` is Hermitian,
  so its transpose is its conjugate, and `Tr[P_t M^T] = Tr[P_t^* M]` is exactly
  the answer for a tip at `(n_x, -n_y, n_z)`. Real, non-negative, positive
  semi-definite, blind to a degenerate rotation, and exact in the Tersoff-Hamann
  limit whenever the sample's moment has no `y` component. **The validation cell
  therefore has to have `m_y != 0`**, which is why `h-sheet-noncolin.in` puts the
  moment at `angle1 = 90, angle2 = 40` rather than along an axis. The existing
  spinor tests all use `angle1 = 90` alone and would have passed the wrong code.
* `defumat/scf/density.py` uses `cross = conj(up) down`, `m_x = 2 Re`, `m_y = 2 Im`,
  which is `psi^dag sigma psi`. That is what makes the Tersoff-Hamann comparison
  against `project_spin` exact; had it been the other sign the y-component would
  have flipped and the failure would have looked like this contraction's.
* **A notebook trap I hit and corrected.** The transmission is linear in the tip's
  moment, so `T(theta) = c0 + cx cos + cy sin` -- *not* `mean + swing cos(theta)`.
  The naive form assumes the junction's easy axis is the substrate's, and the
  sheet's own moment (at 40 degrees) tilts it: the first version of the figure was
  **31 per cent** off and looked fine. The correct phase from a three-coefficient
  fit is `-139.5` degrees, and the fit is exact to 7e-16. The notebook now draws
  the curve from three measured directions and shows the other ten landing on it.
* The vacuum amplitude at `E_F` on this sheet is almost entirely the channel
  **antiparallel** to the atomic moment, which is why the *antiparallel* electrode
  configuration passes 5.3x more current than the parallel one. Surprising but
  consistent across every measurement here.
* `_call_options` in `calculator.py` filters by the callee's signature and passes
  caller options straight through, so a new named parameter needs no `SHARED_OPTIONS`
  entry.
