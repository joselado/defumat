# `h_sc` — simple cubic hydrogen, one atom

A deliberately plain, seconds-long Elk run whose purpose is to be a **ground
truth for reading `STATE.OUT`**. Hydrogen has no core, so the all-electron
and valence densities are the same function and nothing about the frozen
core confounds a reader test.

| | |
|---|---|
| Structure | simple cubic, $a = 3.0$ Bohr, one H at the origin |
| Spin | non-spin-polarised (`ndmag = 0`, no magnetic records in `STATE.OUT`) |
| k-mesh | $4\times4\times4$ |
| Everything else | Elk defaults — `gmaxvr`, `rgkmax`, `swidth`, `autormt` as it falls |
| `tshift` | **`.false.`**, pinned explicitly (see below) |
| Tasks | 0 (ground state) then 33 (3D density) in one run |
| `plot3d` | the unit cell, `np3d = 16 16 16` |

Regenerate with `./regenerate.sh` (needs `./build_elk.sh` at the repo root
first). It deletes everything Elk writes that is not one of the six
committed files.

## Numbers this run produces

Converged (`INFO.OUT` carries `Convergence targets achieved`).

| | |
|---|---|
| $r_{\rm MT}$ | 1.400000000 Bohr (`autormt`), $= r_{\rm sp}(n_{r{\rm MT}})$ in `STATE.OUT` |
| `nrmt` / `nrmti` | 197 / 129 |
| $E_F$ | 0.7397052798e-1 Ha |
| core / valence charge | 0.0 / 1.0 |
| muffin-tin charge | 0.6125761996 |
| interstitial charge | 0.3874238004 |
| total calculated | 1.000739542 (against an exact 1) |

That last row needs reading with `rhonorm` in mind. `src/rhonorm.f90` (called
from `rhomag.f90:24`, on by default) adds a uniform constant to `rhoir` and to
the $l=0$ channel of `rhomt` so the total charge is right, updates `chgmt` and
sets `chgir = chgtot - chgmttot` — but does *not* update `chgcalc`. So the
7.4e-4 "error" is what rhonorm **corrected**, not a floor: the density actually
in `STATE.OUT` integrates to the total, and 0.6125761996 + 0.3874238004 is
exactly 1.

## Why `tshift = .false.` is in `elk.in`

With one atom at the origin the inversion centre is already there, so it is
a no-op here. It is pinned anyway because Elk's default (`.true.`) moves the
origin onto the inversion centre, which leaves `STATE.OUT` and
`GEOMETRY.OUT` in a different frame from `elk.in`'s own `atoms` block. A
fixture whose frame is stated cannot teach a reader the wrong lesson.

## What to check against what

- **`RHO3D.OUT` pointwise** is the primary check. Elk builds it with the
  same muffin-tin plus interstitial reconstruction a reader has to write
  (`src/rfpts.f90`), so it is an independent ground truth at every point,
  and it carries the Cartesian coordinate beside each value.
- **`chgmt` per atom** (`INFO.OUT`, via `parsers.info.parse_charges`) is a
  clean integrated check: a pure radial integral inside the sphere, with the
  characteristic function nowhere in it, and post-`rhonorm`, so it describes
  the `rhomt` the file holds. Reintegrating the $l=0$ channel recovers it —
  `tests/test_state_fixture.py` does exactly that, to 9e-6, which is the
  difference between Simpson and Elk's spline weights.
- **`chgir` is not a clean check.** Before `rhonorm` it weights `rhoir` with
  the *smooth* characteristic function, a Fourier-truncated step; after it, it
  is whatever closes the total. A sharp in-or-out boundary sum on this fixture
  gives 0.38466 against the printed 0.38742 — 0.7% apart, and nearly four times
  the printed "error".

The layout of `STATE.OUT` itself, and the conventions inside it, are
`docs/design.md` §34.
