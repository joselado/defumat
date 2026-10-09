# Route C on PAW nickel: the stall is the plain SCF's (`OPEN.md` Part XXI item 2)

*Written 2026-10-09 on branch `open-a`, for a fable review before any code. Nothing below is
implemented. The scripts are in `plans/open-a-scripts/` (run on D22 through `run.sh`, which sets
the environment and the 6 GB cap); their output is in `/l/ladovj1/review/open-a/` on D22.*

## What was measured

The entry's cell, `ni-tetragonal-relaxed-mae-paw.in` at 40/320 Ry, fully-relativistic PAW
(`Ni.rel-pbe-spn-kjpaw_psl.1.0.0.UPF`), `3 3 2`, Marzari-Vanderbilt `degauss = 0.02`,
`mixing_beta = 0.3`, `conv_thr = 1e-10`, `run_scf(residual_split=True)`, on D22-0161
(`review/open-a/xxi2.py`, four performance cores, a 6 GB cap). The plain arms replace
`_onecenter_torque` by zero, which changes nothing they compute, since without
`rotate_moments` that torque feeds nothing (`OPEN.md` Part XXII item 1), and is what lets them
fit under the cap. One iteration takes about a second.

**Route C could not be reproduced under 6 GB**, and the reason is the one-centre torque's
memory, not the physics: killed at iteration 4 with the whole store, at iteration 22 with
`wfc_store = 'stream'` (high-water 5.19 GB at iteration 8, 6.44 GB at 22, so growing about
0.1 GB an iteration), and at iteration 6 with `jax.clear_caches()` after every torque. The
torque is an eager `jax.grad` and, streamed, an eager `jax.jvp` per k-chunk over closures built
afresh at every iteration (`scf/driver.py:_onecenter_torque`, `_streamed_onecenter_torque`),
which is `OPEN.md` Part XXII item 1's code and the eager-closure trap.

**The plain SCF does not converge either, at any orientation, and that is the finding.**

| start (`angle1`, `angle2`) | cutoffs | iterations | `dr2` over the last 60 (min, median, max) | total energy's spread there, Ry | `becsum` magnetic residual, median |
|---|---|---|---|---|---|
| 45, 0 (the entry's) | 40/320 | 200, not converged | 5.7e-9, 1.6e-7, 4.8e-6 | 9.0e-4 | 1.1e-4 |
| 90, 0 (in the plane, on a mirror) | 40/320 | 200, not | 6.5e-7, 3.8e-6, 2.5e-5 | 1.9e-3 | 4.9e-5 |
| 90, 22.5 | 40/320 | 200, not | 5.0e-8, 8.4e-6, 1.1e-4 | 3.5e-3 | 9.4e-5 |
| 0, 0 (along `c`, on the four-fold axis) | 40/320 | 120, not | 2.6e-7, 1.3e-5, 3.0e-4 | 2.4e-2 | 7.4e-4 |
| 0, 0 | **75/480**, the dataset's own | 150, not | `dr2` 7e-9 to 1e-6 at the end | about 1e-3 | |

The 45-degree run carries a **constant rotation residual**, `c_y = 2.2e-4` to 2.5e-4 rad per
iteration about `y` from iteration 29 to 200 (the residual split's generator bin), while the moment's
polar angle random-walks between 44.78 and 45.99 degrees: that is the anisotropy's slow
rotation, which only Route C can traverse in fewer than about ten thousand iterations. **But
the two arms on a symmetry axis**, where the torque vanishes and the rotation bin reads 2e-6
to 2e-5, stall just as much, with the residual in the longitudinal (2.0e-3) and pointwise
transverse (2.5e-3) magnetization bins and the charge bin three orders below. So the stall
P122 saw after Route C reached the plane is not Route C's and not the in-plane angle: it is the
magnetic channel of the plain SCF on this cell, and Route C inherits it.

**What it responds to**, every arm along `c` at 40/320 and 150 iterations unless stated:

| change | outcome |
|---|---|
| `mixing_beta = 0.1` | not converged, `dr2` 3.5e-7 to 2.5e-6 at the end |
| `mixing_beta_mag = 0.1` (the grid magnetization only; `becsum` keeps `beta`) | not converged, 1.2e-5 at the end |
| `mixing_mode = 'local-TF'` | not converged, 1.5e-9 to 1.9e-6 at the end |
| `mixing_mode = 'adaptive'` | diverged, `dr2` 4e4 to 6e4 Ry from iteration 20; stopped at 72 |
| `K_POINTS 6 6 4` | worse: `dr2` 9e-3 at 150 with excursions to 6.4, the moment leaving `c` |
| **`degauss = 0.05`** | **converged in 78** |

So the stall gets worse with a finer mesh and goes away with a wider smearing, which is what a
Stoner-enhanced longitudinal response does near nickel's Fermi-level peak in the density of
states (`MAGNETISM-NEXT.md` F2, the second direction: `chi_0/(1 - I chi_0)`, with nothing
conditioning it), and is not what a coarse-mesh sampling artefact does.

**No `pw.x` reference could be taken on D22.** The serial `pw.x` 7.5 copied there
(`review/timing/pw.x`) diverges on this input from its first iteration at both cutoffs
(estimated accuracy 3.9e4 Ry and 7.3e4 Ry, absolute magnetization 300 and 408 mu_B, negative
rho about one electron), which looks like the adaptive arm above and not like a converging
code, so whether `pw.x` converges this cell is not known; that build has run the cobalt film
and the response pairs as committed, so the failure is this dataset or this regime in it, and
it is not chased here.

## What the plan is

1. **Prerequisite, not this entry's**: the one-centre torque's memory (Part XXII item 1, and
   the eager-closure trap in `_onecenter_torque` and `_streamed_onecenter_torque`), without
   which Route C on this one-atom cell does not fit 6 GB. The check is a Route C run of 200
   iterations under the cap with a flat high-water mark.
2. **The longitudinal magnetic channel of the plain SCF**, measured before anything about
   Route C: the Stoner term of P130 (`mixing_mode = 'ldos-stoner'`, branch `ldos-spin`, not
   merged) on this cell along `c` at `degauss = 0.02`, against the 150-iteration stall above
   and against the `degauss = 0.05` run's 78 as the bound of what conditioning the Stoner
   direction can buy. Also `becsum`'s magnetization taking `mixing_beta_mag` (today it keeps
   `beta`, `scf/mixing.py`'s docstring calling that "a second departure with no measurement
   behind it"); this cell is the measurement.
3. **Only then Route C**: from 45 degrees, with whichever of (2) converges the plain SCF, the
   question being whether Route C then reaches `conv_thr` and lands in the basal plane, and
   whether `rotation_flat_curvature` is needed at all. Read the code first on one point: the
   freeze removes a held generator from the step and the convergence test
   (`orientation.py:OrientationStepper.propose`, `driver.py:8155-8160`) but not from the
   residual the mixer mixes and `dr2` sums, so a held direction's residual stays in `dr2`.

## How to know it worked

Step 2: the plain SCF along `c` at 40/320 and `degauss = 0.02` reaching `1e-10`, and the 45
degree start too, with the energy's spread over the last iterations below 1e-8 Ry. Step 3:
Route C from 45 degrees reaching `conv_thr` with the torque below `torque_conv_thr` and the
moment in the basal plane, which every run so far has found.

## Not decided here

Whether Route C's reproduction is allowed a cap above 6 GB on D22 before step 1 lands (the
one-centre torque is about 0.1 GB an iteration above the plain run's 2.0 GB), which is the
user's.
