# `sic_zb` — 3C-SiC, one carbon and one silicon

*Copied from the `elkpy` checkout at commit `66c1a0c`, where it was generated.
`regenerate.sh`'s paths are that repository's, not this one's, and
`STATE_INIT.OUT` was left behind: it is the free-atom initialisation state, which
belongs to the core-differencing work that is not implemented here. Nothing in
this directory is read by anything but* `tests/unit/test_elk_reader.py` *and*
`tests/regression/test_elk_seed.py`, *which use it for the two binary-layout
traps a one-species run cannot fail.*

The **two-species** fixture. Its whole reason for existing is two traps in
`STATE.OUT` that no one-species run can fail, so a reader gets them wrong
silently and nothing complains:

- **`natmtot` is the sum of `natoms` over all species**, not `natoms(1)`. Here
  `nspecies = 2` and `natoms = 1 1`, so a reader that writes `natoms(1)` reads
  half the muffin-tin block and then desyncs on `rhoir`.
- **`rfmt` is dimensioned to `nrmtmax`, and the rows past `nrmt(is)` are
  uninitialised buffer, not zero.** Elk's species files set `nrmt` by
  periodic-table row — 300 for row 2, 400 for row 3, less the `lradstp`
  rounding at `init0.f90:363` — so here `nrmt` is 297 (C) and 397 (Si) and
  `nrmtmax` is 397. Carbon's rows 298–397 hold leftovers of order 1e-3.

That second one is why this is SiC and **not** BN: boron and nitrogen are both
row 2, so `nrmt` would be 297 for both and equal to `nrmtmax`, and the trap
would stay invisible. `checkmt`/`autormt` changes `rmt`; it never changes
`nrmt`.

Carbon is species 1, i.e. the **short** mesh is the **first** one, so the
padding trap bites atom 1 — the atom a reader is most likely to check. Databases
conventionally list Si at the origin; the two are the same crystal (4a and 4c
have the same site symmetry `-43m`, and F-43m contains the map between them).

| | |
|---|---|
| Structure | zincblende, $a = 8.23845$ Bohr = 4.3596 Å, C at $(0,0,0)$, Si at $(\tfrac14,\tfrac14,\tfrac14)$ |
| Checked before running | Si-C = 3.56735 Bohr = 1.88776 Å, four neighbours, against a literature 1.888 Å |
| Functional | **PBE** (`xctype 20`), matching `c_diamond` |
| Spin | non-spin-polarised |
| k-mesh | $4\times4\times4$, **10** irreducible points (24 crystal symmetries, not diamond's 48) |
| `tshift` | `.false.` — a no-op here, zincblende has no inversion centre at all |
| Tasks | 0 and 33 in one run; **9006** in a second |
| Cost | 0.7 s for `STATE_INIT.OUT`, 6 s for the rest, one core |

Regenerate with `./regenerate.sh` (needs `./build_elk.sh` at the repo root
first, **with the elkpy patch series** — task 9006 is `patches/0024`).

## Numbers this run produces

Converged (`INFO.OUT` carries `Convergence targets achieved`).

| | C (species 1, `ias` 1) | Si (species 2, `ias` 2) |
|---|---|---|
| $r_{\rm MT}$ | 1.582809072 | 1.934544422 |
| `nrmt` / `nrmti` | **297** / 201 | **397** / 273 |
| core states / charge | 1s, 2 e | 1s 2s 2p₁ᐟ₂ 2p₃ᐟ₂, 10 e |
| muffin-tin charge | 4.832618957 | 11.73695755 |
| core leakage | 2.478794992e-5 | 5.234845918e-3 |
| $\rho$ at the nucleus | 129.9024835 | **2094.5177730** |

| | |
|---|---|
| `nrmtmax` / `nrcmtmax` | 397 / 100 |
| `lmmaxo` / `lmmaxi` | 49 / 4 |
| `ngridg` / `ngtot` | 24 24 24 / 13824 |
| `ngvec` | 4015 |
| $E_F$ | 0.33148292208740 Ha |
| core / valence charge | 12.0 / 8.0 |
| interstitial charge | 3.430423495 |
| total calculated / error | 19.99058944 / 9.411e-3 (**pre-`rhonorm`**) |
| total energy | −328.14885465278 Ha |
| indirect gap | 0.048495 Ha = 1.32 eV (PBE; experiment 2.36 eV) |

Both radii are rescaled at startup: `checkmt` finds the species defaults
(C 1.8, Si 2.2, sum 4.0) larger than the 3.567 Bohr bond and scales **both** by
the same factor. And `rgkmax` is then converted with the **atom-weighted
average** radius (`isgkmax = -1`, `init0.f90`), giving
$|G+k|_{\max} = 7/1.759 = 3.980$ — against 4.881 in `c_diamond`. In a
one-species fixture "average radius" and "the radius" are the same number, so
this rule is only ever tested here.

## Why the two columns settle the `ias` ordering

Diamond's two carbons are related by inversion, so their $l=0$ channels are
*identical* and only the odd-$l$ ones can see an `ias` swap. Here the two
columns share no symmetry relation at all: 2095 e/Bohr³ at the silicon nucleus
against 130 at the carbon, and 11.74 electrons in one sphere against 4.83.
Nothing subtle is needed.

`STATE_INIT.OUT` is the same task-9006 initial state as in `c_diamond` — see
that README for what it is and, more importantly, what it is not.

The layout of `STATE.OUT` itself, and the conventions inside it, are
`docs/design.md` §34.
