# NiI₂ monolayer: the spin-spiral energy surface `E(q)`, and `dE/dq` beside it

The Elk input for a NiI₂ monolayer (`spinsprl`, `vqlss`, a 9×9×1 grid), run here
instead, over the circuit **M → K → Γ → M** of the hexagonal Brillouin zone.
Two quantities per wavevector:

* `E(q)`, the SCF total energy; and
* `dE/dq`, which is P21's `jax.grad` of that same energy at frozen
  wavefunctions and a frozen plane-wave sphere.

So the curve comes out **twice** — directly, and by integrating the gradient
along the circuit. What separates them is the **basis**, not the k-mesh: the
energies are computed on a plane-wave sphere rebuilt at every point and step by
the Pulay error wherever a plane wave crosses the cutoff, where the gradient is
taken at a frozen sphere and does not. `dE/dq` at the frozen converged state is
the exact derivative of the *same* fixed-mesh `E(q)`, so it carries the same
Brillouin-zone error the energies do and integrating it buys nothing there —
that was measured on a hydrogen chain after this calculation was set up, and it
is why the gap between the two curves is read here as a **cutoff** diagnostic.
`pw.x` has no spin spiral: there is no reference output to check either against,
and the two curves plus the circuit closing on itself at M are the checks there
are.

## Files

| file | what it is |
|---|---|
| `nii2-spiral.in` | the calculation, in `pw.x` format, with the Elk mapping written out term by term in its header |
| `run_spiral.py` | one `q` per invocation: SCF, then `dE/dq`, then one JSON |
| `nii2-spiral.sbatch` | the 20-task array, one task per point of the circuit |
| `nii2-calibrate.sbatch` | ten iterations at `k_batch` ∈ {4, 8, 16} — timing and peak HBM only, **not run** |
| `analyse.py` | the table, the two curves, and the plot |
| `smoke.in`, `smoke2.in`, `smoke3.in` | reduced-cost copies used to shake the path out locally |

## The Elk input, term by term

| Elk | here | note |
|---|---|---|
| `avec` | `ibrav = 4`, `celldm(1) = 7.360483`, `celldm(3) = 6.2843502254` | Elk's matrix *is* QE's hexagonal convention; `a2 = (-a/2, a√3/2, 0)` agrees to the printed digits |
| `atoms` / `atposl` | `ATOMIC_POSITIONS (crystal)` | the trailing three zeros per Elk position line are `bfcmt`, and are zero |
| `spinsprl`, `vqlss` | `noncolin = .true.`, `spiral_q` | same lattice-coordinate convention |
| `ngridk 9 9 1` | `K_POINTS automatic 9 9 1 0 0 0`, `nosym = .true.` | a spiral needs the full 81-point grid: the spin space group is not implemented |
| `mixtype 3` | the default Broyden mixer | |
| `epspot 1e-7` | `conv_thr = 1e-8` | |
| `rgkmax`, `gmaxvr`, `lmaxapw`, `lmaxo` | — | LAPW basis knobs with no plane-wave counterpart; the cutoff comes from the pseudopotentials (`ecutwfc = 80 Ry`) |
| `bfieldc 0.05 0 0` | `starting_magnetization(Ni) = 0.15`, `angle1 = 90` | **deliberately not reproduced** — see below |
| `sppath` | `Ni.pbe-nc-sg15.UPF`, `I.pbe-nc-sg15.UPF` | Elk species files have no counterpart; see below |

### Why `bfieldc` is dropped rather than translated

In Elk that field is a transverse *seed*: it breaks spin symmetry so the spiral
can form, and `reducebf` is the usual way to take it back out. Three reasons not
to carry it:

1. an external field's energy is deliberately **outside** the reported total
   here (`defumat/scf/fields.py`, P18 — Elk excludes its own by the same
   convention), so the field would contaminate exactly the `E(q)` *differences*
   this calculation is for;
2. `dE/dq` is **refused** with a field, and correctly: the converged state is
   then stationary for a different functional than the one being
   differentiated, so the missing term would be silent. Half the deliverable
   would be unavailable;
3. `starting_magnetization` with `angle1 = 90` does the seeding job cleanly, and
   it is what P19's own validation case uses.

### Why these pseudopotentials

A chain of three refusals decides this, and none of it is a preference:

* a spiral **refuses ultrasoft and PAW** (the augmentation charge *between* the
  two spinor components is not threaded through), so the dataset must be
  norm-conserving — which rules out every Ni and I file in QE's own library, all
  of which are US or PAW;
* a **fully-relativistic** dataset without `lspinorb` is refused too (QE's
  `average_pp` is not implemented, and consuming `j`-resolved projectors as
  though they were `l`-resolved is worth ~20 Ry), which rules out the
  `*-nc-dojo.UPF` pair already in `tests/data/pseudo/`;
* and `lspinorb` **breaks the generalized Bloch theorem** outright, so it is not
  the way out.

That leaves scalar-relativistic norm-conserving: **SG15 ONCV PBE-1.2**. Ni
carries 18 valence electrons (3s 3p semicore) and I 17, so 52 per cell, and
`nbnd = 64` at `npol = 2`.

**`starting_magnetization` is a fraction of the *valence* charge**, and with an
18-electron Ni the usual `0.4` seeds 7.2 μ_B on an ion whose moment is about 2.
`0.15 × 18 = 2.7` is the seed used, checked by integrating the starting density:
52.0 electrons in channel 0 and 2.7 in channel 1 (`m_x`, since `angle1 = 90`),
with `m_y = m_z = 0`.

**SG15 files carry no `PP_PSWFC`**, so the Davidson start is random rather than
pseudo-atomic. That is handled (`Calculation.starting_wavefunctions` falls back)
but it costs iterations, which is why `mixing_beta = 0.3` and
`electron_maxstep = 200`.

## The circuit

Twenty points, spaced uniformly in **reciprocal space** rather than in
fractional coordinates — the three legs are 1 : 2 : 1.73 in cartesian length, so
equal fractional spacing would sample them at three different densities. The
corners land exactly on M (0), K (4) and Γ (12), and **point 19 repeats point
0**: the energy must come back to `E(M)` and the integral to zero, and running M
in two independent jobs is also a cross-job reproducibility check.

`M = (1/2, 0, 0)` is the commensurate antiferromagnet — the one wavevector on the
circuit a supercell can also compute, and therefore the point where an
independent check is available if one is wanted later.

## Reading the two curves against each other

`dE/dq` is taken at a **frozen plane-wave sphere**, so it does not see the jump
the energy takes wherever a plane wave crosses the cutoff. The integrated curve
is therefore smooth by construction and the direct one is not, and the gap
between them at fixed `ecutwfc` is the size of that Pulay error rather than a
fault in either.

Two traps, both met here and both of the kind that survives a spot check:

* `dE/dq` is a derivative with respect to **crystal** coordinates, so projecting
  it on a leg means contracting it with `(b - a)/L` — a covector with a vector,
  no metric. "Normalising the tangent in the hexagonal metric" first is wrong by
  `t·t`, which is `2/3` on K→Γ and `1` on Γ→M: wrong on two legs of the circuit
  and right on the third.
* **the tangent belongs to the interval, not to the point.** `dE/ds` is
  genuinely discontinuous at a corner — one gradient vector, projected on the
  arriving leg's tangent and on the leaving leg's, giving two different numbers —
  so the quadrature must use each interval's own tangent on *both* its
  endpoints, and a corner gradient is consumed twice. Assigning one tangent per
  point instead is wrong on every interval after a corner. **A synthetic test
  cannot see this if the fixture builds its gradients through the same
  assignment**, because the error then cancels exactly; the fixture that does
  see it is a genuine scalar function of `q` — the frozen-magnon form
  `E(q) = m² [J(0) − J(q)]` — differentiated analytically, which knows nothing
  about the path.

The integral is a **cubic spline through `dE/ds`, built per leg** rather than a
trapezoid, because on twenty points the trapezoid's own second-order error would
be the whole content of the comparison: on the analytic fixture it is 4.26 meV
against a 104 meV span, falling 4.26 → 1.09 → 0.27 → 0.068 as the sampling
doubles, a clean `h²`. The spline gives 0.85 meV at the same twenty points.
`analyse.py` reports both and prints the gap between them as an upper bound on
the integration error — loose by about 5x, since it is dominated by the
trapezoid.

## Running it

```bash
sbatch nii2-spiral.sbatch                       # 20 tasks, 4 at a time, one H200 each
python3 analyse.py <results-dir> --plot eq.png  # table + both curves
```

Results land as one JSON per point under
`$WRKDIR/calculations/nii2-spiral/results/`. Each record carries the energy, the
gradient, the moment, `converged`, `seconds_per_iteration` and peak device
memory, so the first tasks to finish say what the rest actually cost.

## What the moment does, measured

The one thing that decides whether the sweep is worth its GPU time, and it is
settled: at `q = 0`, `ecutwfc = 30`, a 2x2x1 grid, the SCF **converges in 23
iterations to `|m| = 1.89` Bohr magnetons** along the seeded direction, with
`m_y` and `m_z` at 1e-4. That is the right size for Ni(2+) `d8` (`S = 1`, so
about 2, reduced by hybridisation with the iodine).

**The first six iterations are not evidence of anything**, and this is worth
recording because it looks exactly like a collapse: the energy swings by 60 Ry
and the moment goes 2.00 -> 0.78 -> 2.48 -> 1.20 -> 0.51 -> 1.42 before settling.
That is the random start (SG15 carries no `PP_PSWFC`) working itself out, not the
magnetism dying. An earlier run stopped at fifteen iterations reported
`|m| = 0.02` and read as a failed seed; it was a run stopped in the middle of the
thrash.

Cost on this workstation for that reduced case: 248 s for 23 iterations, 4
k-points, `npwx = 6014`, a 25 x 25 x 162 grid.

## What has *not* been established

* **No calibration run.** `--k-batch 8` and `--time=1-00:00:00` are sized to be
  generous, not measured; `GPU.md` has nothing above ten atoms on a single
  k-point, and this is 81 k-points of two-component spinors.
* **No supercell check** at `q = M` yet.
* The moment above is measured at `ecutwfc = 30` on a 2x2x1 grid, not at the
  production 80 and 9x9x1.
* **No cutoff convergence check.** `E(q)` differences are mRy-scale and
  `ecutwfc = 80` is chosen from what SG15 datasets usually want, not from a
  measurement on this cell.
