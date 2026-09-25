# Measurements left by the review of 2026-09-25

The review of the five single-session commits (`AUDIT-2026-09-25.md`) and the two phases
that followed it (`PLAN.md` P117 and P118) closed everything that was an edit. What is
left is below, and every item on it is a run: each says what it settles, the input and the
call, the number that decides it, and what it costs. They are ordered by what a wrong
answer would cost, and the first four need nothing but the workstation.

**All of it was run on 2026-09-25, on the workstation, one run at a time** (`PLAN.md`
P119, which has every number). What is left is one thing that is not a run, the
mechanism of the half-cutoff nickel floor; the term `frozen_expectation` left out, which
is not zero, was added in P120. Each section below keeps its plan and says
in one line what the run gave.

## The nickel PAW floor at `soc_scale = 0`

**Result**: real at 40/320 Ry (7.20e-6 meV at `1e-14`, the same as at `1e-12`), and gone
at the dataset's own 75/480 (3.9e-9 meV), so it belongs to the truncated cutoff.


The question is whether the 5e-10 Ry left between x and z on a fully-relativistic PAW
nickel leg is the reduced functional or an energy not yet converged. What we have is
`E(x) - E(z)` of -6.8e-6 meV at `conv_thr = 1e-10` and +7.3e-6 meV at `1e-12`, the same
size with the sign flipped, and the same leg along x twice bit-identical, so it is not
run-to-run scatter; the totals themselves moved by 1.7e-9 Ry between the two thresholds,
which leaves room for the second reading. Ultrasoft cobalt reaches 1.9e-10 meV on the same
route. Run it once more at `1e-14`: a spread that stays at 5e-10 Ry is the functional,
one that falls with the threshold is convergence. About 15 minutes on three cores.

```python
from pathlib import Path
from defumat import Calculator
from defumat.workflows.anisotropy import run_relaxed_anisotropy

src = Path("tests/data/qe/ni-tetragonal-relaxed-mae-paw.in").read_text()
src = (src.replace("ecutwfc = 75.0, ecutrho = 480.0", "ecutwfc = 40.0, ecutrho = 320.0")
          .replace("3 3 2 0 0 0", "2 2 2 0 0 0"))
Path("ni.in").write_text(src)
calc = Calculator.from_file("ni.in", pseudo_dir="tests/data/pseudo", announce=False)
r = run_relaxed_anisotropy(calc.system, calc.pseudos,
                           directions=((1, 0, 0), (0, 0, 1)),
                           soc_scale=0.0, require_spin_orbit=False,
                           conv_thr=1.0e-14, max_iterations=400)
print(r.total_energies, r.difference(0, 1) * 13.605693122994e3, "meV")
```

If it is the functional, the candidates in `OPEN.md` Part XIX item 2 are the smearing's
Fermi level, the PAW `becsum` symmetrisation on a `nosym` run, and the sphere's angular
quadrature under the noncollinear gradient; `onecenter_species` at the *converged*
`becsum`, rotated as a whole, is the first split.

## The GGA response kernel at negative density points

**Result**: the ratio is 3.2e-5. The full epsilon at the current tree is 17.732233153
against `ph.x`'s 17.732384482, 1.5e-4 below it; at `f3984b7`, before P116, it was
17.732643332, 2.6e-4 above. P116 moved it toward `ph.x` and across it. The kernel-only
A/B, which would attribute the remaining 1.5e-4, is not run.


P116 moved the ground state toward `pw.x` by keeping negative vacuum points in the
gradient correction, and because the kernel here is one `jvp` of `v_of_rho` it moved the
kernel too: at an active negative point (`rho + rho_core < -1e-6`, `sigma > 1e-10`) it is
now the second derivative of the signed energy, where `ph.x` sets all four kernel arrays
to zero (`setup_dgc.f90:153-159`). This code's kernel is the exact derivative of its own
potential, so this is a departure from `ph.x`'s convention rather than an inconsistency,
and its size and sign on epsilon are unknown. On `bismuthene-epsilon-us-soc.in`, which has
a committed `ph.x` in-plane epsilon of 17.732384482:

1. converge the ground state (11 iterations) and count the active negative points;
2. take the in-plane response density `drho_x` of the first Sternheimer iteration and form
   `<drho_x|(K_new - K_old) drho_x> / <drho_x|K drho_x>`, with the old signed gate patched
   into `xc/functional.py:_sanitise` for the kernel call only;
3. only if that ratio is above about 1e-6, run the full epsilon both ways against 17.732384482.

The full epsilon did not finish in 70 minutes here, which is why step 2 exists.

## `frozen_expectation`'s missing term

**Result**: not zero. +1.26e-2 meV in every direction on the cubic smoke cell, and a
first-order anisotropy of 1.79e-3 meV on tetragonal cobalt; `OPEN.md` Part XIX item 3.
Added in P120.


`frozen_expectation` evaluates `delta dvan_so - eps delta qq_so` and leaves out `newd_so`'s
sandwich against its spin trace, `F B F - T(B)`, which on an ultrasoft dataset is part of
the first-order operator. Its recorded +/-0.000001 meV was taken on the ultrasoft
`Co.rel-pbe-nd-rrkjus`, and its direction check sits on a simple cubic cell where symmetry
forces the spread to zero. Add `sum_k w <psi|beta> (D1 - D0) <beta|psi>`, with `D1` and `D0`
from `scf/driver.py:_newd_noncollinear` at 1 and 0 on the frozen potential, for x, y and z,
on `test_anisotropy.py`'s `_SMOKE_SOC` cell and on `co-tetragonal-relaxed-mae.in`, where
symmetry does not force the spread. The quenched orbital moment says the term vanishes at
first order too; a number above 1e-3 meV would say otherwise. Minutes.

## `dr2` against `pw.x`'s printed accuracy

**Result**: 7.9e-3 relative on `scf-kcrys` and 5.6e-4 on `lsda`, so no convention error;
the parser and gate test are not written.


Nothing pins `scf_accuracy` itself against `pw.x`: the tests of `rho_ddot`'s transform
share `total_charge`, `kinetic`, `E2`, `FPI` and the spin branching with it, so a
convention error in any of those passes on both sides. The first iteration starts from
the same superposition of atomic charges in both codes, so compare this code's first
accuracy with `pw.x`'s first "estimated scf accuracy" on `pw_scf/scf-kcrys` (0.06340640 Ry,
`tests/data/qe/reference.out.pw_scf-scf-kcrys`) and `pw_lsda/lsda` (0.91975683 Ry,
`reference.out.pw_lsda-lsda`); the second is the one that tests the magnetization half.
`io/qeref.py` does not parse that line yet, and a parser plus a gate test is the natural
home. Seconds per cell.

## Smaller checks, each a sentence in the record away from a number

**Results**: `pw.x` reaches the seeded nickel state from this code's eigenvalues
(-171.00255270 Ry); the high-G share is 2e-4 to 3e-3; the graphene bilayer has no negative
point; the PBE and LDA bismuthene terms differ from `pw.x` alike, to about 1e-6 Ry; and Part
III is bit-identical on any core count and on the tabulated route.


- **The seeded nickel DFT+U state.** `test_noncollinear_hubbard_resume.py` pins a seeded
  collinear source at -171.0025527 Ry that no reference code has produced; `pw.x` was run
  unseeded only (-171.0002509). One serial `pw.x` run of `ni-kind1-force.in` with the ten
  `starting_ns_eigenvalue(m, spin, 1)` lines the test's `_seeded` writes, `conv_thr =
  1e-8`: agreement to 1e-5 Ry and on the d traces 4.973 / 4.179 closes it.
- **The high-G share of the metric fit.** P113 compared the metric fit with `pw.x`'s on the
  nickel cell, and the fit here runs over the whole dense set where `pw.x` fits only
  `G < ngms`. On `ni-kind1-force.in` with `RHO_DDOT_FIT = True`, take the share of
  `scf_accuracy` and of one off-diagonal `F(r_k) . F(r_{k-1})` above `|G|^2 = 4 ecutwfc` at
  iterations 2, 5 and 10; below about 1e-3 the comparison stands.
- **Whether `test_dispersion.py`'s PBE stress can see P116.** Count the active negative
  points on the converged `graphene-bilayer-d2` density. Zero means its `pw.x` stress is a
  null for the `v2` sign, which `OPEN.md` Part XIX item 2 now says it is presumed to be.
- **The first-order effect of the `v2` sign without a new `pw.x` run.** Converge
  `bismuthene-soc-small.in` and `bismuthene-soc-small-lda.in` (8 iterations each) and put
  their one-electron and Hartree terms beside `pw.x`'s (`io/qeref.py` parses both). A PBE
  mismatch at the LDA control's level bounds the flip at first order; `bi10-soc`'s 5.1e-5 and
  5.9e-5 Ry say it might not be.
- **Part III's bit-identity on more than one core.** P112's three edits were measured
  bit-identical pinned to one core. On `benchmarks/si8-us-1k.in` (nine chunks), compare
  `Q_ij(G)` and `get_stress()` with `np.array_equal` between the current tree and a
  `93d882f` worktree, once pinned and once not, and repeat P112's `M1` script on
  `o2-paw-texture.in` with `DEFUMAT_AUG_MAX_BYTES=0` to force the tabulated route, which
  was never A/B'd.

## On Triton

**Results**, both run on the workstation instead: the full-cutoff nickel leg converges and
its identity holds to 3.9e-9 meV at `1e-14`; the PBE stress agrees with `pw.x` as well as
the LDA control does, 4e-8 to 6e-8 Ry/bohr^3.


- **The nickel PAW leg at its own cutoff.** `ni-tetragonal-relaxed-mae-paw.in` at 75/480 Ry
  diverged at `soc_scale = 0` before P115 (200 iterations to an accuracy of 5.2e+02 Ry).
  At 40/320 it now converges in about 10 minutes, before and after P117. Run the full
  cutoff on a 2x2x2 mesh at `conv_thr = 1e-10`, x and z, on `batch-milan`: a converged pair
  whose spread is at the floor above opens the PAW anisotropy, and a divergence says the
  cell rather than the reduction.
- **The `v2` stress against `pw.x`.** `pw.x` with `tstress = .true.` on
  `bismuthene-soc-small.in`, and `get_stress()` here at the same settings, in Ry/bohr^3 at
  1e-8. The stress sees the sign at first order, and the transcribed `stres_gradcorr` and
  the `jax.grad` stress share it, so only `pw.x` can.

## Not a measurement, and larger

The seeded ultracell's lean toward its reference moment (`OPEN.md` Part XX) wants a frozen
basis closed under time reversal. The discriminating run is written there: the three-cell
`z` reference at `nbnd = 40` with the closed basis, and the old basis at 80 as the
equal-size control.
