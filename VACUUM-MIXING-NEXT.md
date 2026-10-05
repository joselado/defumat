# What to do about converging cells with vacuum, sized

**Status, 2026-10-04 evening.** The user chose P-A and P-B; both are done, as `PLAN.md` P129,
on branch `ldos-precond`. P-C (`rho_ddot`'s fit) and P-D (`local-TF`'s cost, with this
document's probe added to `OPEN.md` Part VIII item 4) are open. **The numbers below are the
prototype's** and stay as the measurement that motivated the work; the production mode differs
in four ways the plan's review asked for, and its own counts are in P129: the LDOS comes out of
the density's pass under a `vmap` rather than a second pass, it is built with a Gaussian delta
and clamped to `D >= 0`, the inner solve is conjugate gradients in the Hartree norm at 1e-3, and
tetrahedra are refused. It takes the aluminium slab in 10 to 11 iterations at every vacuum and
the cobalt film in 20 to 22, and the film in **76.1 s against `pw.x`'s 40.9 s** and this code's
own `local-TF` at 106.0 s (`PERFORMANCE.md`, "Cells with vacuum"). **The magnetization**
(P130, 2026-10-05): a spin-resolved LDOS term and the paper's Stoner term were built on branch
`ldos-spin` and measured, and not merged: the magnets measured are not slow in the magnetization
under `'ldos'`, the Stoner term helps near iron's transition (2x2x2 27 -> 21) and fails on the
cobalt film (inner solve unconverged, 20 -> 36, ten times the time), and can converge onto the
nonmagnetic saddle.

A proposal, written 2026-10-04 on master at `df8f4a8`, from a literature survey on arXiv and
a measurement on `D22-0161`. The question was whether a cell with a lot of vacuum (a slab, a
2D material) needs a different density mixer, and if so which one. **The short answer is
that a monolayer does not and a metal film does**, that the two preconditioners the code has
for slabs (`'TF'` and `'local-TF'`) make every monolayer measured *slower* in proportion to
the cell length, and that the local-density-of-states preconditioner of Herbst and Levitt,
prototyped here on branch `ldos-precond` (not merged), is the one scheme that is flat in the
vacuum on every cell measured, metal film, 2D metal, semimetal and insulator alike: 10 to
12 iterations on a five-layer aluminium slab from 16 to 64 bohr of vacuum, and 21 to 23 on
a three-layer cobalt film from 21 to 59, where `pw.x` takes 14 and 24 on the one vacuum of
each it has been run at and plain Anderson under this code's default fit does not converge
the film at all.

**How to read the numbers.** Every count is an SCF iteration (one diagonalisation) to the
input's own `conv_thr`, at the input's own `mixing_beta`, on the CPU of `D22-0161`, one
thread per run (`taskset` to one core, `OMP_NUM_THREADS=1`, `DEFUMAT_THREADS=off`), from
the scripts in `tools/vacuum_sweep/` on branch `ldos-precond` (`make_inputs.py`,
`run_one.py`, `aggregate.py`, `prec_cost.py`, the `sweep*.sh` drivers); they ran in
`/l/ladovj1/review/vacuum/` on D22, where the per-run JSON is in `out/`. Wall times inside
the sweep are **not** comparable between arms: the LDOS arm of the four sweep cells ran on
the efficiency cores (12-15) and the rest on the performance cores, six at a time; section 3
has the one wall-clock comparison taken alone. "flat / ddot" is the Anderson fit's inner
product: the default Euclidean one over the packed real-space vector, and `rho_ddot`'s
Hartree metric (`driver.RHO_DDOT_FIT = True`, `PLAN.md` P113). Every converged arm of a
cell lands on one state: the energy spread over arms is at most 6.8e-8 Ry on the aluminium
slab (`conv_thr = 1e-8`), 2.9e-7 on NbSe2 (`1e-8`), 1.0e-8 on graphene and 7.8e-9 on hBN
(`1e-10`). On the cobalt film (`1e-10`) it is 1.5e-9 and 1.2e-9 Ry at c10 and c14 and
3.2e-8 at c6, where every arm stopped at an accuracy between 1.2e-11 and 9.6e-11 and the
moment agrees to 6.4e-6 mu_B (5.260360 to 5.260367), so one state reached from different
sides rather than two.

## 1. The measurement

**The cells.** Each is held fixed in bohr and only the cell length `c` along the normal
changes, so that a sweep changes the vacuum and nothing else (positions written in bohr;
crystal coordinates would stretch the layer).

- `al-v*`: `benchmarks/al-slab.in`, five Al(100) layers, 15 bohr of metal, LDA, Gaussian
  `degauss = 0.05`, 2x2x1, `beta = 0.7`. Vacuum 16, 32, 48, 64 bohr (`c` = 31, 47, 63, 79).
- `nbse2-v*`: `tests/data/qe/nbse2-monolayer.in`, a 2D metal, PBE SG15, Fermi-Dirac
  `degauss = 0.002`, 9x9x1, `beta = 0.7`. Se to Se 6.35 bohr; vacuum 16, 28, 48, 64.
- `graphene-v*`: `tests/data/qe/graphene-monolayer.in`, a semimetal, Gaussian
  `degauss = 0.02`, 12x12x1, `beta = 0.5`. `c` = 20, 40, 60.
- `hbn-v*`: an hBN monolayer built for this (`B.pbe-hgh`, `N.pbe-hgh`, `a = 4.7419` bohr,
  `ecutwfc = 40`, which is low for HGH nitrogen and the same at every `c`), fixed
  occupations, 6x6x1, `beta = 0.7`. `c` = 20, 40, 60.
- `co-c*`: `tests/data/qe/co-slab-forcetheorem-sr.in`, three Co(0001) layers, ultrasoft
  PBE, `nspin = 2`, Marzari-Vanderbilt `degauss = 0.005`, 12x12x1 shifted, `beta = 0.7`,
  `conv_thr = 1e-10`, `electron_maxstep = 150`. Positions are in angstrom in that input, so
  `celldm(3)` alone moves the vacuum: 6, 10 (the input's), 14, i.e. 20.7, 39.6 and 58.5
  bohr of vacuum. This is the film on which plain Anderson diverged to +335 Ry in P59.

**Iterations, flat fit / `rho_ddot` fit.** `ldos` is the prototype (section 3), at an
inner GMRES tolerance of 1e-4.

| cell | `anderson` | `TF` | `local-TF` | `ldos` |
|---|---|---|---|---|
| al-v16 | 25 / 14 | 15 / 14 | 13 / 14 | **10 / 11** |
| al-v32 | 27 / 15 | 21 / 20 | 14 / 14 | **11 / 11** |
| al-v48 | 33 / 18 | 27 / 26 | 17 / 16 | **10 / 12** |
| al-v64 | 34 / 15 | 36 / 36 | 20 / 18 | **10 / 11** |
| nbse2-v16 | 12 / 9 | 14 / 12 | 11 / 9 | 11 / 11 |
| nbse2-v28 | **9 / 9** | 24 / 18 | 13 / 10 | 11 / 11 |
| nbse2-v48 | **9 / 9** | 33 / 27 | 16 / 13 | 11 / 11 |
| nbse2-v64 | **9 / 9** | 43 / 30 | 18 / 14 | 11 / 11 |
| graphene-v20 | **8 / 9** | 16 / 15 | 12 / 11 | **8 / 8** |
| graphene-v40 | **8 / 8** | 31 / 25 | 16 / 14 | **8 / 9** |
| graphene-v60 | **8 / 8** | 37 / 37 | 22 / 18 | **8 / 9** |
| hbn-v20 | **11 / 10** | 19 / 16 | 15 / 15 | **11 / 10** |
| hbn-v40 | **11 / 12** | 34 / 27 | 18 / 14 | **11 / 12** |
| hbn-v60 | **11 / 12** | 39 / 34 | 24 / 19 | **11 / 12** |
| co-c6 | n.c. / 32 | 37 / 32 | 24 / 20 | **21 / 21** |
| co-c10 | n.c. / 31 | 51 / 44 | 30 / 24 | **21 / 21** |
| co-c14 | n.c. / 36 | 74 / 55 | 35 / 26 | **23 / 22** |

"n.c." is not converged at 150: the flat-fit Anderson total oscillates by hundreds of Ry to
the end on all three cobalt cells (its last three energies on co-c10: +751.2, -301.5,
-343.9 Ry). The c10 row reproduces the record: `local-TF` 30 flat and 24 with `rho_ddot`,
as P113 has it.

`pw.x` on the two cells it has been run on: `al-slab` (v16) 16 plain, 14 `TF`, 14
`local-TF` (`scf/mixing.py`, `DEFAULT_MIXING_SPACE`'s record); the cobalt film (c10) 24 with
`local-TF` (P59).

## 2. What the measurement says

**A monolayer has no vacuum problem under plain Anderson.** NbSe2, a 2D metal, takes 9
iterations at every vacuum from 16 to 64 bohr; graphene 8; hBN 11 or 12. The reading of
that, which the numbers are consistent with and do not prove: the long-wavelength modes
along the normal that a large `c` introduces are modes of the *vacuum*, whose dielectric
eigenvalue is 1, and a layer six bohr thick has nowhere to move charge across, so nothing
along `z` is amplified. The in-plane modes are fixed by the 1x1 cell and do not change with
`c`.

**`TF` and `local-TF` make every monolayer slower, linearly in the cell length.** Kerker
multiplies the vacuum mode at `G_z = 2 pi/c` by `G^2/(G^2 + q_TF^2) ~ (2 pi/c)^2/q_TF^2`,
which is a mode whose true eigenvalue is about 1, so the preconditioned Jacobian acquires an
eigenvalue falling as `c^-2`, a condition number growing as `c^2`, and an Anderson
(Krylov-like) count growing as `c`. Measured, `TF`'s count over the cell length is 0.43 to
0.48 on the aluminium slab (15/31, 21/47, 27/63, 36/79), and the slope is 0.44 iterations a
bohr there, 0.60 on NbSe2, 0.53 on graphene, 0.50 on hBN. `local-TF` grows at about a third
of that slope (0.15 to 0.25 a bohr) because its screening falls in the vacuum, but it still
screens the *dense* region as a metal whatever the layer is, which is wrong for hBN (15 to
24 against plain's 11) and for graphene. The single property that separates the right
preconditioner from these two is whether it asks if there are **states at the Fermi level**
rather than how much **charge** there is.

**A metal film does have a vacuum problem, and two things fix it.** On the five-layer
aluminium slab the flat fit grows 25, 27, 33, 34. **`rho_ddot`'s fit alone takes it to 14
to 18 with no trend**: the Hartree metric weights the residual by `1/G^2`, so the fit
attends most to the longest wavelengths, which is where the charge transfer between the two
surfaces lives and what the flat fit under-weights.
The LDOS preconditioner takes it to **10 to 12 under either fit**, flat, below `pw.x`'s 14 on
the one cell both have been run on.

**On the cobalt film the same holds, more sharply.** The flat fit does not converge at all;
`rho_ddot`'s fit converges it in 31 to 36; `TF` grows 37, 51, 74 with the vacuum and
`local-TF` 24, 30, 35. The LDOS preconditioner takes **21, 21, 23** under either fit,
below `pw.x`'s 24 with `local-TF` on the input's own cell (c10). The first iterations show
the difference: after the first step its total energy is never more than 1.5 to 3.1 Ry from
the converged one, where every other arm's strays by 29 to 107 Ry (`local-TF` 29 to 43, `TF`
39 to 57, `rho_ddot` Anderson 75 to 107). The reading is that the charge transfer between the
film's two surfaces, which the others damp after the fact, is not excited in the first
place.

**What the LDOS preconditioner costs in iterations where it is not needed.** On hBN the
LDOS is zero and the step is `beta R`, so the counts are plain's and the energies agree with
plain's to 3e-14 Ry. On graphene it is plain's count. On NbSe2 it is 11 against plain's 9
at every vacuum: two iterations, flat, the price of screening a metal that, being one layer
thick, did not need it.

## 3. The prototype

Branch `ldos-precond`, the prototype at `21b3a98` and the checks below at `a636201` (two
files, `scf/mixing.py` and `scf/driver.py`; local, not pushed, not for merge as it stands,
no tests). Herbst and Levitt, arXiv:2009.01665, J. Phys.: Condens. Matter 33, 085503 (2021):

    chi0~ dV = -D dV + D <D, dV> / <D, 1>,   eps~ = 1 - chi0~ v_H,   step = beta eps~^-1 R

with `D(r) = sum_nk w_k delta(e_F - e_nk) |psi_nk(r)|^2`, the local density of states at the
Fermi level, built with the run's own smearing as the delta, and `v_H = e2 4 pi/|G|^2`. The
second term is what keeps the electron count. **A uniform `D` gives `eps~ = 1 + 8 pi D/G^2`,
which is Kerker with `q_TF^2 = 8 pi D`; a `D` that vanishes in the vacuum leaves the vacuum
alone; a `D` that vanishes everywhere leaves the plain step.** It is QE's
`approx_screening2` with `rho(r)` replaced by the states at `e_F`, which is the whole
difference measured above.

- `ldos_preconditioner` (`scf/mixing.py`): `eps~` inverted by `jax.scipy.sparse.linalg.gmres`
  inside one `jax.jit`, Kerker at the cell-averaged `D` as its preconditioner and first
  guess, `restart = 20`, at most 4 restarts, no host synchronisation inside the solve. Charge
  only: the magnetization, `becsum`, `ns` and `tau` take `beta`, as in Kerker.
- `_fermi_ldos` (`scf/driver.py`): the density routine called a second time with
  `w_k wgauss'((e_F - e)/degauss)/degauss` as the weights, so it carries the augmentation
  charge and the symmetrisation. Zero for fixed occupations. **A separate pass**: it costs
  as much as the density.
- `mixing_mode = 'ldos'`; refused under `mixing_space = 'g'` and with the streamed store.

**Two checks that share no code with the solve** (`identity_check.py`, on `al-v16` after its
SCF). The LDOS integrates to **37.29653813** states/Ry against `dN/de_F` = **37.29653799**
from the eigenvalues and `wgauss` alone (a central difference at `h = 1e-5` Ry), so the spin
factor, the weights and the density routine's volume factors are right. And the
prototype's output solves `eps~ x = R` with `eps~` written again in numpy, to **2.5e-4**
relative with the SCF's own LDOS (GMRES at 1e-4) and to **1.0e-15** with a uniform one; a
uniform `D` reproduces `kerker_preconditioner(screening = 8 pi D)` to **4.6e-16**. On this
slab the cell-averaged LDOS gives `8 pi <D>` = 1.075 bohr^-2 against the Thomas-Fermi
`q_TF^2` = 1.017 that QE's `approx_screening` derives from `r_s`, which is the same number
reached by a different route.

**Inner solve, counted in the sweep** (operator applications per call at tolerance 1e-4,
averaged over the run): hBN 2.0 (the zero-LDOS short cut still runs a step), graphene 7,
aluminium 12 to 26 growing with the vacuum, NbSe2 18 to 25, cobalt 38. The LDOS pass cost
0.02 to 0.05 s an iteration on aluminium, 1.4 s on NbSe2-v64 and 1.3 to 1.5 s on co-c14,
where the whole iteration of the 1e-2 arm (below) took 6.0 s, so **the separate pass is
about 30 per cent of a cobalt iteration on its own**.

**One call, timed alone** (`prec_cost.py`: one performance core of D22, `taskset -c 0`, one
thread, a compiling call and then the median of five; a synthetic 20-bohr metal slab for
the density and the LDOS, a smooth random residual), seconds, with the operator
applications at each GMRES tolerance in parentheses:

| grid | points | FFT pair | `TF` | `local-TF` | LDOS 1e-4 | LDOS 1e-3 | LDOS 1e-2 |
|---|---|---|---|---|---|---|---|
| al-v64, 12x12x180 | 25,920 | 0.0004 | 0.0004 | 0.073 | 0.036 (37) | 0.023 (23) | 0.017 (17) |
| co-c14, 24x24x300 | 172,800 | 0.0023 | 0.0026 | 0.255 | 0.543 (37) | 0.336 (23) | 0.247 (17) |
| 60x14x54 bohr box, 243x60x225 | 3,280,500 | 0.090 | 0.121 | 11.15 | 14.64 (38) | 8.85 (23) | 6.10 (16) |

On the large grid an application costs 0.385 s, **4.3 FFT pairs**: the operator's own pair,
a second pair for the Kerker preconditioner inside GMRES (the prototype keeps its vectors in
real space, so a diagonal-in-G operator costs a transform), and the orthogonalisation
against up to 20 Krylov vectors of 3.3 million points. `local-TF` on the same grid is 11.1 s,
about 120 FFT pairs, which is **not** the 730 s an iteration of `OPEN.md` Part VIII item 4
on a grid of 4.1 million: the probe does not reproduce that number, and the gap is not in the
algorithm's transform count.

**Wall clock, on the cobalt film at the input's own vacuum (co-c10)**, each arm alone on one
performance core of D22 with the compile cache warm from the sweep, run twice and the second
read (`clean_time.sh`; the two runs agree to 0.6 s):

| arm | iterations | whole run (s) | per iteration (s) |
|---|---|---|---|
| `TF`, flat | 51 | 152.2 | 2.98 |
| `anderson`, `rho_ddot` | 31 | 105.0 | 3.39 |
| `local-TF`, flat (the input as it runs today) | 30 | 106.0 | 3.53 |
| `local-TF`, `rho_ddot` | 24 | 89.1 | 3.71 |
| `ldos`, flat (prototype) | **21** | **89.8** | 4.28 |

Of the prototype's extra 1.30 s an iteration over `TF`'s 2.98, 1.19 s is accounted for by
the two parts measured inside it, 0.34 s of inner solve (35 applications a call) and **0.85 s
of the separate LDOS pass**, which is consistent with equal Davidson work per iteration
between the arms but does not show it. The 0.85 s is a `block_until_ready` on the LDOS taken
right after the density was dispatched, so it may carry the tail of the density's own work.
As it stands the prototype is 15 per cent
faster than the input's own `local-TF` and level with `local-TF` under `rho_ddot`; fused
(P-A item 1) it would be about 72 s, 32 and 19 per cent faster than the two, by arithmetic
on these rows and not measured. On this cell the win is in the iteration count, and the
per-iteration cost is what decides how much of it reaches the clock.

**The inner tolerance cannot be loosened.** The same arm at DFTK's default, 1e-2 (17 and 16
applications against 37 and 38), iterations flat / `rho_ddot`:

| cell | at 1e-4 | at 1e-2 |
|---|---|---|
| al-v16, v32, v48, v64 | 10/11, 11/11, 10/12, 10/11 | 10/11, 10/12, 11/12, 12/12 |
| co-c6, c10, c14 | 21/21, 21/21, 23/22 | 25/24, 28/25, 31/28 |
| nbse2-v28, v64 | 11/11, 11/11 | 11/12, **18**/13 |
| graphene-v60 | 8/9 | 8/9 |

Loose, the cobalt film grows with the vacuum again (25 to 31) and NbSe2-v64 loses seven
iterations. Anderson assumes the step is one linear map of the residual at every iteration;
an inner solve that stops early is a different map each time, and the measurement says that
matters at 1e-2 and not at 1e-4.

## 4. What the literature has, and why the rest is not proposed

Read in full by two agents on 2026-10-04 (PDFs and notes in the session's scratchpad, not
kept). Counts are the papers' own.

- **LDOS preconditioner**, Herbst and Levitt (above). DFTK `LdosMixing`, ABINIT
  `iprcel = 200` (whose documentation suggests it "as a default" for smeared metals).
  Al(100) with vacuum as wide as the metal: N = 10 none 19, Kerker 22, LDOS 9; N = 20 none
  47, Kerker not converged in 50, LDOS 9 (condition numbers 170.8, 323.9, 3.5). Stated limit:
  on a semiconductor it is no preconditioning, and Al+GaAs needs the LDOS+dielectric hybrid
  (Kerker 26, LDOS 26, hybrid 13), which needs `eps_r` and `k_TF` as parameters.
- **Its magnetic extension**, Barat, Levitt and Torrent, arXiv:2606.26693 (preprint):
  `(1 - chi0_LDOS K_H - chi0_diag K_XC)^-1`, a Stoner-like diagonal for the soft magnetic
  mode, ABINIT `iprcel = 202`. Bulk Fe, Co, Ni only, no vacuum; the authors call the range
  where it helps "small".
- **Density-dependent Thomas-Fermi**: Raczkowski, Canning and Wang, PRB 64, 121101 (2001), a
  TFW minimisation (QE's `local-TF` is a linear variant, already here); Freysoldt, Mishra,
  Ashton and Neugebauer, PRB 102, 045403 (2020), the Lin-Yang elliptic operator with
  `b(r) = q_TF^2(rho~)/4 pi`, `b = 0` below 1e-3 e/bohr^3, `rho~` broadened by 1 bohr, which
  "entirely removes the vacuum dependence" on a 10-layer Al(111) slab (counts in a figure
  only). Both read the charge, so both screen an insulating layer as a metal, which is what
  `local-TF` does to hBN above. Not proposed: the LDOS scheme dominates it on every cell
  here at the same structure of cost.
- **Elliptic preconditioner**, Lin and Yang, arXiv:1206.2225, SIAM J. Sci. Comput. 35, S277
  (2013): `a(r)`, `b(r)` built by hand per system. Not proposed: not parameter-free.
- **Extrapolar**, Anglade and Gonze, PRB 78, 045126 (2008): the RPA dielectric matrix on a
  low cutoff from Adler-Wiser plus a closure. Flat on Sr(100) and Si(100) slabs, but it needs
  empty bands and an `O(N^4)` build. Not proposed.
- **Low-rank dielectric**, Das and Gavini, arXiv:2211.07894, PRB 107, 125133 (2023): a
  rank-5-to-14 Jacobian from density-matrix perturbation theory inside Chebyshev filtering.
  Pt slabs of 10, 20, 40 layers: Kerker 61, 126, 248; LRDM 23, 25, 28. It is the published
  relative of this code's Newton-Krylov (P22), which measured as a loss here because every
  Krylov vector is a diagonalisation. Not proposed now; it is the scheme to look at if a
  metal-semiconductor interface defeats the LDOS one.
- **Adaptive damping**, Herbst and Levitt, arXiv:2109.14018, J. Comput. Phys. 459, 111127
  (2022): a line search on the step length. On Al40 with vacuum, Kerker converges at no fixed
  damping and not with adaptive damping either. It fixes the step, not a wrong
  preconditioner. Not proposed for this.
- **Auxiliary functionals**, Hasnip and Probert, arXiv:1503.01420: the only vacuum-alone sweep
  in the literature (graphene 3 to 13 A, MgO, an Au4 cluster), where CASTEP's Pulay, Broyden
  and Kerker grow linearly and theirs is flat. Here plain Anderson is already flat on
  graphene (8, 8, 8). Research code; not proposed.
- **A preconditioner from 2D screening** (in-plane `1/|q|` rather than `1/q^2`): no paper
  found. It would matter for a large in-plane supercell of a 2D metal, not for the vacuum. If
  `assume_isolated = '2D'` is ever implemented (it is refused today, `system/builder.py`),
  the LDOS operator inherits the truncated kernel by building `v_H` with it, which gives the
  2D form with no new parameter; QE does not do this (its `rho_ddot` and `approx_screening`
  ignore the cutoff).
- What the codes recommend: VASP lowers `AMIN` (Kerker's floor) for slabs with a dipole
  correction; CASTEP recommends ensemble DFT over density mixing for a metal slab with a
  dipole correction; ABINIT recommends the extrapolar technique for "a highly polarisable part
  and some vacuum" and `iprcel = 200` for metals.

## 5. What is proposed, in order

**P-A. `mixing_mode = 'ldos'` as a supported mixer.** What the prototype lacks:

1. The LDOS accumulated **inside the density pass**: `sum_band` already forms `|psi_nk(r)|^2`
   for every band, and the LDOS is the same sum with a second weight vector, so fused it is
   one more multiply-add on the grid per band with nonzero weight (and the projections
   `<beta|psi>` that `becsum` needs are the same ones), against a whole second pass today
   (about 30 per cent of a cobalt iteration, 10 per cent of an NbSe2 one). The streamed store
   (`scf/streaming.py:stream_densities`) and the pools' all-reduce carry it as one more sum
   over k.
2. A `SphereLayout` version for `mixing_space = 'g'`: the operator multiplies in real space,
   so it needs a transform pair per application on the sphere, as `local_tf_preconditioner_g`
   does.
3. The inner solve kept at 1e-4 (section 3) and made cheaper per application. **The operator
   is symmetric positive definite in the right variables**: `chi0~` is self-adjoint and
   negative semidefinite (`<f, chi0~ f> = -<D f^2> + <D f>^2/<D> <= 0` by Cauchy-Schwarz with
   weight `D`), so `v^1/2 eps~ v^-1/2 = 1 - v^1/2 chi0~ v^1/2 >= 1`, and preconditioned
   conjugate gradients replaces GMRES. With the vectors held in G, `v^1/2` and the Kerker
   preconditioner are free multiplications and one application is **one FFT pair** with no
   orthogonalisation, against the prototype's 4.3. Arithmetic on the table above, not
   measured, and assuming conjugate gradients needs the 38 applications GMRES did: about
   3.4 s rather than 14.6 on the 3.3-million-point grid.
4. The width of the delta. Tetrahedra have none, and a fine smearing on a coarse mesh gives a
   noisy LDOS (NbSe2 here: `degauss = 0.002` on 9x9); DFTK widens the smearing for the LDOS
   alone. One decision covers both: build the LDOS with `max(degauss, a stated minimum)`, and
   with that Gaussian for tetrahedra, or refuse them.
5. Tests: a uniform `D` reproduces `kerker_preconditioner` with `q_TF^2 = 8 pi D` to
   round-off (an identity that shares no code with the GMRES); a zero `D` reproduces the plain
   Anderson trajectory bit for bit; a step conserves the electron count; `al-v64` converges
   in at most 13 (slow set).
6. `docs/features.tex`: the entry, and an amber box for what it does not do: the
   magnetization; a semiconductor (no states at `e_F`, so the plain step); a
   metal-semiconductor interface; and localized orbitals at `e_F`, which Herbst and Levitt name
   as the case where it is "as ineffective as homogeneous schemes" and which is a d-electron
   magnet's.

Size: about 250 lines and a day. It is not proposed as the default yet; one more cell
(a semiconductor slab, and a metal on an insulator) should come first.

**P-B. Say in the guide that `TF` and `local-TF` are for metal films and not for
monolayers**, with the numbers above, and warn at setup when either is asked for on a run
with fixed occupations (an insulator, where the measured cost is 15 to 24 iterations against
11). Small.

**P-C. Reopen P113's decision on `rho_ddot`'s fit, with the vacuum numbers.** Under plain
Anderson it halves the aluminium slab (34 to 15 at 64 bohr), and on the cobalt film it is
the difference between converging (31 to 36) and not converging in 150 at any vacuum; under
`local-TF` it takes the film from 30 to 24 at c10, which is `pw.x`'s count. Against it:
P113's DFT+U nickel cell, 79 against 100 and unconverged, and a history twice the size.
With P-A the films no longer need it (21 to 23 and 10 to 12 under either fit), so it matters
for a run that does not ask for `'ldos'`. The user's decision.

**P-D. `OPEN.md` Part VIII item 4, `local-TF`'s 730 s an iteration on the NiBr2 slab.** If
P-A replaces `local-TF` as the slab recommendation, this matters only for inputs that ask for
`local-TF` by name. The probe in section 3 does not reproduce the 730 s (11.1 s for one call
on 3.3 million points, one core), so the first step there is still the profile that item
asks for, on the NiBr2 cell itself.

## 6. Not measured

- A large in-plane supercell of a 2D metal, where the in-plane modes do slosh.
- A metal on a semiconductor (Herbst and Levitt: LDOS no better than Kerker there).
- A magnetic insulator such as the NiBr2 slab, whose soft mode is the magnetization: the LDOS
  scheme screens only the charge and is not expected to change it.
- Thicker metal films than five layers of aluminium and three of cobalt.
- Any GPU number.
