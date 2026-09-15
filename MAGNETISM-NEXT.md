# Magnetism: what is left, and what each piece needs

## What this file is

The forward-looking half of `NONCOLLINEAR.md`. That file is an **audit** — a snapshot of
what was true on 2026-09-12 at commit `314d676`, kept whole because the reasoning behind
each finding is why its fix has the shape it does. This one is the **queue**: what is still
open after P77–P79, ordered by what a wrong or missing answer costs, with the first concrete
step of each written down so that picking one up does not start with re-deriving why it
matters.

Where the other files fit (`CLAUDE.md` has the full table):

| question | file |
|---|---|
| what a phase found, and what it was measured at | `PLAN.md` §3 |
| the audit that produced this list, with the full reasoning per item | `NONCOLLINEAR.md` |
| what is known to be broken, and what to do about it | `OPEN.md` |
| what a feature refuses, and how a user reaches it | `docs/features.tex` |
| **what to do next about magnetism, and what it costs** | this file |

**Every item below is a phase unless it says otherwise.** That is the honest sizing and it
is the reason none of them was opened in the session that closed the rest: each needs new
physics, a new external reference, or a measurement that takes a machine rather than an
afternoon. Section numbers in brackets are `NONCOLLINEAR.md`'s own item numbers, kept so
the two files can be read against each other.

**File and line references go stale.** Function names are given wherever possible;
re-`grep` anything before trusting it, which is the same warning `NONCOLLINEAR.md` carries
about its own.

---

## 1. What was closed, so nobody reopens it

P77 through P79 closed all of Tier 1 and most of Tier 2/3. In one line each, with the
number, because the whole rule of this project is that a claim is a number:

| was | is | number |
|---|---|---|
| a one-species collinear antiferromagnet averaged to zero | `sgam_at_collin`'s filter, which a collinear run never had | 6.6 meV and the whole magnetic state; now matches `nosym` to 1e-8 Ry |
| nothing in a run said whether the texture survived | `SCFResult.site_moments`/`site_charges`, per iteration | 0.25–0.72% of an iteration; ~5 GB → ~31 MB on a 157-atom slab |
| three bare `\|m\|` in differentiated paths | `safe_modulus`, plus a regex sweep over six modules | nan on 243 components → finite, energy unchanged |
| `starting_magnetization >= 1` meant two different things in the two codes | QE's `input.f90:1448` rule, transcribed | factor of 6 on oxygen; `= 2` seeded −3 electrons |
| a stated texture never reached DFT+U's `ns` or PAW's `becsum` | per-atom axis in both, both regimes | one-centre moments 90° wrong → agree with the charge to 7.5e-6 |
| a seed field between 1e-12 and 1e-5 Ry was invisible to the filter | one scale-free rule for both thresholds | group stayed at 8 instead of 2; now cut down to 1e-12 |
| a run could converge with `reducebf`'s field still on, silently | a warning above `FADED_FIELD`, and Elk's `[0.5, 1]` range enforced | at 0.99: six iterations, 95% of the field still on, total 0.63 meV out |
| `'atomic texture'` crashed on its own first potential build | `fields.ATOM_RESOLVED`, one set instead of a hand-written tuple | an `AttributeError` for any input with no field card |
| `'total direction'` returned NaN whenever the moment lay along z | `atan2` with both arguments masked | QE's `fact1` to 1e-12 off the axis, zero on it |
| `mixing_ndim` was parsed and ignored | wired through `run_scf`, the facade and all three relaxation drivers | and **measured**: it does not help — see item 6 below |
| `dr2` folded charge and magnetization into one scalar | `scf_accuracy_split`, both halves in `history` and on the console | free: 12.4 ms against 12.2 on a 64³ grid, where computing them apart costs 24.6 |
| a texture could only be stated in an input file | `System.with_moments`, `Calculator.with_moments` | the `replace` workaround leaves 9 k-points reduced with 16 operations while the group recomputes to 4 |
| nothing held a texture that was not the ground state | it does — `'atomic'`, not `'atomic texture'` | 120° held to **0.55° per site** in 38 iterations, against a collapse to 180° in ten |
| every DFT+U continuation into `nspin = 4` was refused | promoted into the two diagonal spin blocks | 4 iterations against 78 from scratch |
| a spinor `ns` lost its imaginary part through `starting_ns` | the complex side of the precision policy | every `starting_ns=` and every noncollinear DFT+U resume, silently collinear since P62b |
| nothing said whether the group a run uses belongs to the density it starts from | `Calculation.symmetry_residual`, checked before iteration 1 and warned above 1e-3 | 6.7e-16 from a card against **1.0** for the same texture under a group too large; the charge reads 5.9e-16 in *both* (P80) |
| no **vector** texture had ever been carried through an SCF and inspected | one has: a 90-degree cycloid on four hydrogens, symmetrised and free | 0.4543 mu_B per site either way, angles 90.00 degrees, the pair 8.0e-9 Ry apart (P80) |
| a relaxation of a magnet said nothing per site | `site_charges`/`site_moments` on all three drivers' step objects | mechanical; and the *final* geometry is the one step that cannot show a collapse (P80) |
| noncollinear DFT+U under symmetry was refused for a matrix nobody had built | `spin_rotations`, QE's `d_spin_ldau`, plus the transpose an antiunitary operation needs | wedge against closed grid on fcc nickel: **5.9e-11 Ry**; the missing transpose is worth **5.1** against 8.9e-16 on the axial law (P82) |
| the noncollinear GGA had no measured derivative anywhere | bcc iron, ultrasoft, PBE, compared through its **stress** | 6.7e-9 Ry and 1.6e-7 Ry/bohr^3, the level the collinear ultrasoft cases reach; **signed branch only** (P80) |
| an `fsm` run that missed its target said so nowhere | `SCFResult.constraint_residual`, signed, plus a warning of its own | `constraint_energy` is 0 for a feedback field; a run with `accuracy = 3.9e-11` under a 1e-10 threshold and its moment 0.174 out read as an ordinary non-convergence (P80) |

Three of those were **not** in the audit and were found while fixing it: the spinor `ns`
cast, nickel's `conv_thr` (`OPEN.md` Y1), and `'atomic texture'`'s `1/|m|`. The first is
the one to remember — it was live inside a documented, validated feature, and only an
end-to-end test caught it. An array-algebra test passed either way.

### Closed on 2026-09-13 (P84 and after)

| was | is | number |
|---|---|---|
| three "defects" on three different days, two of them the same line of one input | one **saturated seed**: `starting_magnetization = 1.0` on hydrogen polarises the atom completely, so the first potential is built at `\|zeta\| = 1` | 126 iterations to m = 0.027 against **7** to the m = 0.53125 ferromagnet; `test_magnons.py` 8 passed (P84) |
| P63's spiral scan "no longer reproduces", cause unidentified, three candidates nominated | it reproduces to the digit at the right seed, and now **confirms** the magnon prediction | minimum at `q = (0,0,1/4)`, -150.1 meV, where the susceptibility says the ferromagnet first goes unstable — and P63's own `0, -150, -59` come back (P84) |
| a caller that wrote `scf.density` lost the fact that it converged | `SCFResult.require_converged`, the one implementation, with `Calculator._ground_state` wrapping it | three tests reported a Goldstone residual of 0.3958 that was an unconverged ground state, not a defect in the susceptibility (P84) |
| a committed input asked for a scheme the code warned does not converge, and said it did | `h2-texture-120.in` is `'atomic'` at `lambda = 10`, and neither test that cites it rewrites away from a stale literal any more | 38 iterations, 121.13 degrees, 0.576 per site (P84) |
| the spin spiral had no external number of any kind | it has two, and they disagree: Elk against defumat at `q = 0` and `q = 1/4` | moments agree to **4 per cent** at both; `E(1/4) - E(0)` is **-136.294 meV against -20.712**, a factor of 6.6, and Elk's held field explains 6 per cent of it (P86) |
| the anisotropy could be computed only at frozen density, so PAW and DFT+U were out | `run_relaxed_anisotropy`: one self-consistent run per direction, differencing **total** energies, which hands nothing over and so has no handoff to refuse for | 0.447 meV on tetragonal cobalt against the theorem's **free** energy 0.552 and its band sum 1.235 -- the relaxed route independently says the free energy is the right object (P87) |
| a held texture's residual angle was a property of penalties with no alternative | Elk's per-atom feedback field, both variants, transcribed and unit-tested -- and **measured not to win**, with the two updates failing for two different reasons | fixed gain: a *growing* ring over 2000 iterations at half Elk's gain. Secant: stable at `acc = 6.5e-6` and converged to the **wrong state** -- lengths right to 8 per cent, angles 145 deg out, because its `chi` is diagonal. Penalty: 0.576 deg in 38 (P85) |
| the relaxed route's precision was an argument, not a number | it is a number: an identity control with the coupling switched off, scanned in `conv_thr` | **0.011 meV** and it plateaus -- 1e-13 equals 1e-12 to 2 per cent while the density residual falls another order (P87) |

---

## 2. The queue, in order

### A. No linear response for a spinor, so a magnet with spin-orbit coupling has no phonons and no spectra [11]

**Partly closed. P81 did the solve; P83 did the dielectric tensor and the Born charges
for a spinor carrying no net moment. What is below is the state before those, kept
because its reasoning is why the rest has the shape it does. Still open: the textured
case (item A2, immediately after this one), the phonons and everything above them
(`symmetrize_displacement`'s axial landmine), and the ultrasoft spinor
(`set_int3_nc`).**

**Phase.** The largest item here by consequence: everything above the ground state is closed
for a noncollinear run — phonons, Born charges, the dielectric constant, LO-TO splitting,
Raman, the strain response, the elastic constants, electrostriction, the piezoelectric
tensor, and the cheap route to the magnetoelectric tensor, which currently does six SCF runs
and a central difference and says in its own docstring that it would rather not.

`response/sternheimer.py:1122` refuses the whole stack for `noncolin`, naming
`incdrhoscf_nc` and `set_int3_nc` as "a second implementation rather than a spin axis on
this one". Taken at face value that overstates it, and this project has already paid once
for inheriting a refusal without checking which machine it belongs to (P35's was about the
Sternheimer stack and never applied to a sum over states; taking it as read left P54's whole
quantity marked impossible).

**What already exists.** `scf/density.py:407`'s `spinor_sum_band` is the induced density's
spinor form, needing only the cross term per Pauli component. `int3` is by this project's
own rule one `jvp` of `newd`, whose noncollinear form is
`Calculation._noncollinear_coefficients`. The XC kernel is a `jvp` of `scf/potential.py`'s
`nspin = 4` branch. And `sternheimer.py:981` already reads
`degeneracy = 1 if calculation.noncolin else 2`, which is a striking thing to find behind a
blanket refusal. What is *not* reachable: `density_at` (`sternheimer.py:617`) hardcodes the
collinear `sum_band`.

**The landmine, and it is on nobody's list.** `Calculation.symmetrize_directional`
(`driver.py:2823`) applies a plain cartesian rotation to the perturbation-direction axis and
treats every `nspin_mag` channel as a scalar. At `nspin_mag = 4` three of those channels are
the magnetization and need `det(R)` and the time-reversal sign, which
`symmetry.py:498`'s `symmetrize_magnetization` has and this method does not call. Its own
docstring states the trap — an induced charge density is polar where a magnetization is
axial, "and applying the wrong one is a different symmetry rather than a worse average".

**First step, and it needs no new assembly.** `chi_0` under a potential probe against a
central difference of the density, on `tests/data/qe/h-chain-90deg.in` — four noncollinear
atoms, norm-conserving, so `set_int3_nc` does not arise and the test is `spinor_sum_band`
plus the existing CG. That is the check P45 used to close `nspin = 2`. It exercises the
kernel at a node, because a 90-degree texture has grid points where `m` passes through
zero.

### A2. A textured spinor's dielectric tensor is 5.3 per cent from `ph.x` and nobody knows whose fault it is [new, P83]

**Phase.** P83 opened the dielectric tensor and the Born charges for a spinor and validated
them for `nspin_mag = 1` -- the identity against the scalar run at 5.0e-14, the wedge against
the closed grid at 7.4e-13, `ph.x` at 4.3e-5. The **textured** case (`nspin_mag = 4`) runs,
passes two internal checks that are not weak, and disagrees with `ph.x`. It is refused by
name (`require_a_measured_spinor_response`), which is where it stays until this is located.

**The numbers, on `i-atom-soc.in`** -- an iodine atom, `lspinorb`, fixed occupations, a
0.164 eV gap, moment 1.00 mu_B, `nosym`, and the two codes agreeing on the ground state to
the printed digit (-25.80117002 Ry):

| | across the moment | along the moment |
|---|---|---|
| defumat | 1.356572109, 1.356572109 | **1.574482417** |
| `ph.x` | 1.357034400, 1.357092056 | **1.494593593** |

4.6e-4 across, which is `ph.x`'s own floor on this cell (its two transverse entries differ
from each other by 5.8e-5), and **5.3 per cent** along. The same solve in RPA gives
1.37741894, so the disagreement is 40 per cent of the whole `f_xc` contribution.

**What passes, and it is why this is interesting rather than obvious.** The tensor is
uniaxial along the moment with nothing imposing it (`nosym`, so the symmetriser returns its
argument and `symmatrix` is skipped), and turning the moment to `x` moves the distinct axis
and returns **the same two numbers to nine digits**. The assembly is not wildly wrong.

**Three explanations are already dead. Do not test them again.** `dmxc_nc` differs from a
`jvp` of `v_of_rho` in exactly three places, all thresholds, and each fires at **zero** of
this cell's 157464 grid points: the clamped `zeta` derivative (`max |zeta| = 0.3245` against
a clamp at `1 - 2e-6`), the `|zeta| > 1` / `n <= 1e-30` zeroing (`min n = 1.3e-9`), and the
`|m| <= 1e-10` rule. Both kernels are in their smooth interior. This is **not** P70's
convention trap one regime up, which was the obvious guess.

**The RPA control does not exist for a magnet and that is physics.** Dropping `f_xc` leaves
the magnetization with no restoring kernel at all, since Hartree is blind to it: `ph.x`
diverges outright (`|ddv_scf|^2` at 1e14 by iteration 44). defumat's `screening = "hartree"`
converges instead, which is a second unexplained difference and may be the cheaper thread to
pull.

**First step, and it is cheap.** P81's own check on this cell: `chi_0` under a *potential*
probe against a central difference of the density. It has no kernel in it at all, so it
separates the solve from the screening -- and **no Sternheimer solve here has ever run on an
`lspinorb` dataset**, since P81's three cells were all `H.pz-vbc` or `Si.pz-vbc`. The script
is written (`iodine_chi0.py` in P83's scratch) and is a few minutes. If it passes, the
kernel is the suspect and `dmxc_nc`'s own `dz = 1e-6` finite difference is as much a
candidate as this code's exact derivative; the tie-breaker is an independent sum-over-states
route on the same cell, which shares only the ground state.

### B. Elk's per-atom feedback field, so a held texture is exact rather than nearly [10, remaining half]

**P85 built it and measured it, and the measurement is negative on this cell. Read
`PLAN.md` P85 before picking this up.** `'atomic fsm'` and `'atomic fsm direction'` are
Elk's `fsmtype = 2` and `-2`, transcribed with `r3vo`, unit-tested against both routines,
wired through the driver and refused at input where they cannot work. What they do **not**
do is beat the penalty: on the 120-degree hydrogen pair the site residual rings over 2000
iterations with a *growing* envelope, at half Elk's default gain, where the vector penalty
holds 0.576 degrees per site in 38 iterations.

**The cause is the cell and it was predictable from this file.** Left alone that pair
carries |m| = 0.000235 mu_B, so its `m(B)` is nearly a step -- which is exactly what item
E(c) below records for the *other* hydrogen cell, where `fsm` also could not hold a target.
No fixed-gain controller is stable against a nearly vertical response, and the cell was
chosen for a penalty, which does not care.

**The two updates fail differently and each names its own fix.** The fixed-gain one is
*unstable* against a steep `m(B)`, which wants a robust magnet -- the same two-atom canted
**iron** cell Q5 asks for, so building it serves both items at once. The secant one is
*stable and blind*: it converges at `acc = 6.5e-6` with the moment **lengths** right to 8
per cent (0.239 against 0.26) and the **angles** wrong by 145 degrees per site, plateauing
at a 0.4 mu_B residual for 900 iterations. That is its diagonal `chi = dm/dB` -- three
scalars per atom -- and what sets a texture's angles is the exchange *between* atoms, which
is exactly the off-diagonal block a diagonal model discards. A per-atom 3x3 block is the
smallest honest replacement.

**The original entry follows.**

**Phase.** P79 measured that a *penalty* holds a 120° state to 0.55° per site at the largest
`lambda` the SCF tolerates. A penalty leaves a residual force at convergence by construction
(`scf/fields.py` says so in the `feedback` docstring), and 0.55° **is** that residual. Elk
fixes a moment per muffin tin instead — `mommtfix(:, ia, is)` with `fsmtype = 2` or `3`,
updating one field per atom, and a negative `fsmtype` fixing the *direction* alone by
projecting the field perpendicular to the target
(`~/apps/elk-9.6.8/src/bfieldfsm.f90:32-73`, with a `t1 >= 1000` per-atom skip sentinel) —
which converges to a genuine stationary point of the unconstrained functional under that
field.

**What exists here.** Nothing per atom: `constraint_targets` returns a single 3-vector for
both `'total'` and `'fsm'`, and `MagneticField.feedback` drives one uniform field from
`total_moment`. `grep mommtfix\|bfsmcmt` over `defumat` returns nothing. The machinery to
put it in is the `'atomic'` scheme's — the spheres, the per-atom moments, the per-atom
targets are all there and validated; what is missing is the update rule.

**Why it is second and not first.** It is an improvement on a route that now works, where
item A is a whole class of quantity that does not exist. Take it when 0.55° is not good
enough for something specific.

**First step.** Transcribe `bfieldfsm.f90:50-73` into the `'atomic'` machinery, and rerun
P79's own table (`tests/regression/test_holding_a_texture.py`) with the feedback field as a
sixth row. The claim to beat is 0.55° in 38 iterations.

### C. One missing matrix gated noncollinear DFT+U with symmetry and two more. ✅ CLOSED by P82 [13]

**Done, with one part deliberately left and sized.** `PLAN.md` §3 P82 has the full record;
in one line each:

- **DFT+U with `noncolin` under symmetry** runs. `system/symmetry.py`'s `spin_rotations` is
  QE's `d_spin_ldau`, built through the quaternion rather than through `find_u`'s case
  analysis, and pinned by three properties instead of by transcription.
- **`t_rev` is no longer refused for a spinor run**, only for a collinear one. It was not
  free: time reversal is antiunitary, so the unitary matrix is half the operation and the
  block must be **transposed** as well — `U rho U^dagger` misses the axial law by **5.1**
  where `U rho^T U^dagger` reproduces it to **8.9e-16**, which is why `new_ns_nc` reads
  `nr(m4, m3, is4, is3, nb)`.
- **`<L>`/`<S>` on a reduced k-set** run: `symmetrize_atom_cartesian_tensor(axial=True)`.
- **The number:** `ni-ldau-noncol.in`, fcc nickel with `U = 4` eV on 3d and the moment along
  z (`nsym = 16`, **eight** of them `t_rev = 1`), wedge against the closed 4x4x4 grid:
  **5.9e-11 Ry**, moments 0.526381 against 0.526388 mu_B. `ns` agrees to 1.6e-6 and the
  residual is the **`nosym`** run's — it keeps a spurious transverse moment of 5e-6 mu_B
  that the symmetrised run annihilates exactly, which is P80's cycloid finding on a
  production magnet.

**What is left, and it is smaller than a phase.** The **symmetrised spinor PDOS** is still
refused, and the old refusal was wrong about why: it named one matrix where there are two.
Without spin-orbit coupling `sym_proj_nc`'s operator is `D^l x S` and **both factors now
exist** (`harmonic_rotations`, `spin_rotations`) — what is missing there is plumbing, since
`ProjectionSymmetry` carries *real* coefficients over `2 lmax + 1` columns and needs complex
ones over `2 (2 lmax + 1)`, plus `sym_proj_nc`'s `ind` relabelling for a time-reversed
operation. With `lspinorb` it is `sym_proj_so`'s `D^j` (`d_matrix_so`), a genuinely
different matrix. **Take the non-SOC half first**: it is an afternoon on top of P82 and it
covers the commoner regime.

### D. Magnons refuse a noncollinear ground state, so the states whose excitations are interesting have none [12]

**Phase.** `tddft/spinchi0.py:229` refuses `system.noncolin` with a correctly named reason —
the 4×4 spin-density response no longer block-diagonalises, so the transverse channel is not
a matrix in `(G, G')` on its own, and Elk's `genspchi0` carries all sixteen blocks. It also
requires norm-conserving. This is a genuine refusal, not a stale one. A 120-degree Néel
state on a triangular lattice, a helix, a cycloid, a canted antiferromagnet: their magnons
are the interesting physics, and the code can compute the ground state and not the
excitations of it.

**The bounded first step is not the sixteen blocks.** For a **spin spiral** the generalized
Bloch theorem already reduces the problem: the transverse channel at wavevector `q` is the
collinear machinery in the rotating frame, which is Elk's `spinsprl` route. So `chi^{+-}`
for a spiral ground state at its own `q`, validated against the collinear antiferromagnet of
the doubled cell at `q = (0, 0, 1/2)` — the same identity P19 used for the spiral energy. It
is refused today for its own reason (two spheres), and it is the cheaper half.

### E. Spin spirals have no external number of any kind [15]

**P86 built the whole apparatus for (a) and did not finish taking the number. Read that
first** (`PLAN.md` P86): the matched input pair is committed
(`tests/data/elk/h_chain_spiral/`, `tests/data/qe/h-chain-spiral-elk.in`), the defumat side
of the scan is measured (`E(q) - E(0)` of 0, -20.71, -50.21 meV at `q_3 = 0, 1/4, 1/2` with
`|m|` of 0.515, 0.615, 0.611), and **three separate ways the Elk side quietly stops being
magnetic are identified and fixed**, each of which converged and reported success: a
`reducebf` that fades the seed field before the moment establishes, Elk's default Broyden
falling off the magnetic branch at loop 10 with its history full, and a binary that would
not start because `libopenblas` is gone from this machine. **It has been run, and the item is no longer "no external number" -- it is a measured
disagreement.** Elk at `q = 0` and `q = 1/4`, both converged:

* the **moments agree to 4 per cent** (0.5376 against 0.5149 at `q = 0`, 0.6380 against
  0.6151 at `q = 1/4`), which is the level this project already records for an
  all-electron-against-pseudopotential moment (bcc iron, 2.0613 against 2.2145);
* **`E(1/4) - E(0)` is -136.294 meV in Elk against -20.712 meV here, a factor of 6.6.**

So the two codes converge to recognisably the same magnetic state and disagree about what
turning it costs. **The obvious explanation is already dead**: Elk holds a small field and
defumat holds none, and repeating the defumat scan under Elk's own field moves the answer
by 6 per cent (-20.712 to -21.986), while five times that field reaches only -30.022.

**Locating the 6.6 is now item E(a)**, and `PLAN.md` P86 lists three candidates, none
tested and cheapest first: basis convergence on both sides (`rgkmax = 7` against
`ecutwfc = 25`, and a spiral needs two `G+k` sets); the `1 1 4` k-grid, which is very coarse
and samples *shifted* spheres at `q != 0` so its error need not be the same at the two
wavevectors; and the field convention, which the moment response hints at -- defumat's
moment reacts far more strongly to "the same" field (0.633 against Elk's 0.538), so the two
codes may not be applying the same field at all.

Two things to carry into that run. It is ~40 s per SCF loop at six threads on netlib and
tens of loops per wavevector, so budget half an hour a point and do not take the
`PERFORMANCE.md` pair beside anything (that one needs `PIN=1`, one core, idle). And Elk
carries a small held field where defumat carries none: the Zeeman energy is outside the
reported total in both codes and the field is the same at every `q`, so it largely cancels
in `E(q) - E(0)` -- the check nobody has done is the same defumat scan under the same field.

**The original entry follows.**

**Phase for (a), and (b) is done.** Five internal identities on a one-atom hydrogen chain
(4e-15, 7e-13, 3e-12 Ry, tests asserting 1e-9), plus P21's four for `dE/dq`. There is no
`pw.x` counterpart and **no Elk number was ever taken**, although Elk implements the same
ansatz (`gengkqvec.f90`, `vqlss`) and is built here at `~/apps/elk-9.6.8`.
`PERFORMANCE.md` has no Elk column for it either, which is `CLAUDE.md`'s standing rule left
open.

*(b) is closed:* the `G/2` grid invariance is now a checked precondition of a spiral run,
warning by name — it holds to 2e-9 Ry on a 1×1×4 grid and fails by **2e-3 Ry on 1×1×3**.

**(a) first step.** Run the same hydrogen chain at three wavevectors as an Elk spin-spiral
ground state and compare total energies after subtracting each code's own reference. That
converts the feature from a set of identities to an external comparison, is the only check
that catches an error the five identities share, and supplies the missing
`PERFORMANCE.md` pair in the same run.

**(c) is closed, and the answer is yes. A spiral accepts a constraint and `fsm` holds one;
what P80 measured as "this cell cannot answer it" was the cell's saturated seed. P84.**

On `h-fcc-spiral-scan.in`'s corrected `starting_magnetization = 0.9`, the same `fsm` run at
`q = 1/2` reaches its target to **-4.5e-4** against `FSM_TOLERANCE = 1e-3`, converged, in 5
iterations. Nothing in `fields.py` changed between the two verdicts. A run seeded at full
saturation has nowhere to go but the other saturated branch, which is what made `m(B)` read
as a step. **Everything below is still true of a saturated seed and is kept for that** — the
cell is genuinely a marginal magnet and a target away from the bare moment still overshoots
to the other side (target 0.10 lands at **-0.268**) — but it is no longer the verdict on
whether a spiral can be held. The iron pairing below is still the better fixture and is
still worth doing; it is now an improvement rather than the only way to get an answer.


Settled: a spiral SCF does not refuse `constrained_magnetization` or a field, and the quantity
the constraint acts on **is** the rotated-frame magnetization, which is the right object for a
helix and was documented nowhere near `constrained_magnetization`.

Not settled, and the two obvious suspects are both ruled out. It is **not a sign error** — at
`q = 0` with a target above the bare moment the field grows *positive* (+0.057 Ry) and the
moment rises — and it is **not the rotated frame**, because it fails at `q = 0` where that
frame is the laboratory frame. What it is: fcc hydrogen at `a = 6.5` bohr is a marginal magnet
(P63: the ferromagnet is metastable, 58 meV *above* the nonmagnetic solution), so `m(B)` is
nearly a **step** — 0.057 Ry of field takes the moment from 0.027 to 0.719 — with very little
in between to hold. Every constrained run lands at that saturated value and the inner SCF then
stops converging under the field that put it there (2.5e-4, 5.5e-3 against 1e-10); the secant
steps only on converged pairs, so the field freezes. *How* the field grew that large was not
traced — those runs predate `constraint_residual` and the trajectory was never printed.

**The cell was unsuitable before the run, and two numbers say so.** The bare `q = 1/2` moment
is **0.0435** and the target was **0.0273** — an initial error of 0.016, sixteen times
`FSM_TOLERANCE`, and *below* the bare value. The constraint was asking for a moment smaller
than the cell's own, on a cell whose only other stable point is the saturated one.

**First step is a different cell, not a different `lambda`, and the cell is already
committed**: `tests/data/qe/fe-noncolin-pbe-stress.in` (P80) is bcc iron at 1.95 mu_B, a
robust magnet whose `m(B)` is not a step, and it converges here in 43 iterations. Give
`max_iterations` room well above the bare SCF's own count — the budget is **shared** between
the inner SCF and the outer field loop, and the hydrogen cell's bare `q = 1/2` run alone takes
149 of 200. A spiral on an iron cell is the pairing E(c) actually wants.

**What P80 added so the next attempt is readable.** `MagneticField.cell_residual` and
`SCFResult.constraint_residual`: `m - m_target`, signed and per component, at the end and per
iteration in `history`. `fsm` had **no** number at all before — `constraint_energy` is 0 by
construction for a feedback field and `site_residuals` only covers the atom-resolved schemes,
so `satisfied` computed the error, tested it and discarded it. A dedicated non-convergence
warning now fires when the **density** converged and only the constraint did not (measured:
`accuracy = 3.9e-11` against `conv_thr = 1e-10`, moment 0.174 mu_B out), because the generic
advice — more iterations, smaller `mixing_beta` — is backwards there.

Two corrections to what was written down. `h-fcc-spiral-scan.in`'s header quoted
`|m| = 0.0001` at `q = 1/2`; the run gives **0.0435**, converged, at a different fixed point.
And **P63's whole spiral scan is stale**: it records `E(q) - E(0) = -59` meV at `q_3 = 1/2` and
the re-run gives **-0.41 meV**, because the `q = 0` end is no longer on the metastable
ferromagnetic branch (0.0273 mu_B on an atom seeded at 1.0). The conclusion stands and the
numbers do not; which of P77–P79's changes moved that minimum is unidentified.

### F. The mixer's metric — and its headroom is now bounded [17]

**Phase, and smaller than it looked.** `benchmarks/fe-mag-1k.in` takes **25** SCF iterations
where `pw.x` takes **12** at the same `conv_thr = 1e-8`, with the two energies agreeing to
6.7e-9 Ry. Three structural differences, all on the extrapolation rather than the step:
`AndersonMixer.mix` (`scf/mixing.py:117`) builds its Gram matrix as a flat Euclidean form on
the packed real-space vector where QE uses `rho_ddot`; QE extrapolates only over the smooth
sphere and linearly mixes above it; and `_mix` packs `becsum` unconditionally where QE
allocates `rho%bec` only `IF (okpaw)`.

**Most of the 43 on the second cell is the mixing parameter, measured 2026-09-14.**
`fe-noncolin-pbe-stress.in` at `mixing_beta = 0.7` instead of the input's 0.2 takes **24**
rather than 43, which is 1.26 times `pw.x`'s 19 rather than 2.26 times it. The 43 stands as a
measurement and the 2:1 reading of it does not: 0.2 is in that input because QE wanted it
there, not because it is the right value for this code. What is left to explain on that cell
is 24 against 19, and the same beta sweep has not been run on `fe-mag-1k`.

**P78 took the first measurement against this and it points away from it.** The deconfounder
`benchmarks/fe-unstable-nonmagnetic.in` — same cell, same dataset, same `mixing_beta = 0.3`,
`nspin = 1`, where a magnetization weighting cannot act at all — takes **21**. So most of
the excess over `pw.x` is not magnetic, and whatever the metric is worth it is bounded by
the four iterations between 25 and 21 rather than the thirteen between 25 and 12. Raising
`mixing_ndim`, the other obvious lever, makes both cells *worse*: 27/25/33/39 at 4/8/12/20
on the magnetic one and 26/21/30/26 on the nonmagnetic.

**A second cell showed the same 2:1 ratio, and most of it turned out to be the knob** (see
the correction above: at `mixing_beta = 0.7` the same cell takes **24**, which is 1.26 times
`pw.x` rather than 2.26). As originally measured,
`fe-noncolin-pbe-stress.in` (P80) takes **43** iterations where `pw.x` takes **19** at the
same `mixing_beta = 0.2` and the same `conv_thr = 1e-10`, with the two energies agreeing to
6.7e-9 Ry — noncollinear, ultrasoft, PBE, where `fe-mag-1k` is collinear. **That "two cells
at 2:1" no longer stands**: raising `mixing_beta` on this one takes it to 24/19, so only one
cell is at 2:1 and the second was a knob. The nonmagnetic twin of this cell still has not been
run, and the same beta sweep has not been run on `fe-mag-1k`, which is now the cheaper of the
two things to do: if 25/12 moves the same way, this item is smaller again.

**First step, if it is still worth one.** Dump one run's residual history and recompute the
Anderson coefficients under both quadratic forms — `scf_accuracy` gives QE's for free — and
report the angle between the two coefficient vectors. If it is small, the metric is not the
mechanism and this item can be closed as measured rather than fixed.

**The same packed vector has a third block nothing watches**, which is `OPEN.md` Y2 (opened
2026-09-14 from the NiBr2 helix run): `becsum` is mixed at the plain `beta` and is in no
convergence measure at all, since both halves of `accuracy` are of the smooth density and
what reaches them is only what `addusdens` already put on the grid. On a PAW magnet the
moment lives in the d-shell `becsum`, so a stall with a flat magnetic half and a large
energy swing is exactly the shape that half would make, and there is no number in the log
that says whether it is the thing still moving. The entry asks for a reported `becsum`
residual that is fed to nothing, and it is the cheapest of the three items here.

**A second route, and this is the place for it: minimise the energy instead of iterating
the density.** Raised by the user, 2026-09-13. Direct minimisation descends `E[psi]` under
orthonormality rather than looking for a fixed point of the density map, and it is
unusually cheap to *write* here because the energy is already written down and
differentiated — which is the whole argument for JAX. Three things decide whether it is
worth trying, and none of them has been measured:

* **It fixes a different failure from the one this item is about.** Direct minimisation
  cures charge sloshing — an unstable fixed-point map over a well-behaved functional. The
  25-against-12 here is not that: `fe-mag-1k` converges, it converges slowly, and its
  nonmagnetic twin takes 21, so most of the excess is not magnetic. Plain descent converges
  on the **condition number** where a secant-type method converges on its square root, so
  on a flat direction it is *worse* than what is here. Anderson already is the quasi-Newton.
* **Both benchmark cells are smeared metals**, so the object to minimise is the free energy
  with the occupations as variables too — Marzari-Vanderbilt ensemble minimisation. That is
  a different algorithm, not a different optimiser, and it is the bulk of the work.
* **It cannot sit on a saddle**, and P84 is the reminder of why that matters here: the
  states these magnetic cells are *for* are routinely metastable or stationary-but-not-minimal
  (`h-fcc-magnon.in`'s ferromagnet is 58 meV above the nonmagnetic solution of the same
  cell). A fixed-point iteration is happy on any stationary point; a minimiser is entitled
  to slide off one, and would do so silently.

**So it is a candidate for the *gapped* cells and for a future direct-minimisation solver
behind `scf_solver`, not a fix for this item.** The honest first measurement is the cheap
one: a gapped insulator where both routes must agree, timed, before anything is written for
a metal.

**Read F2 before picking this item up.** It puts the same question one level lower, at the
conditioning of the map rather than at the mixer that damps it, and the metric turns out to
be one of several candidate answers rather than the candidate. F2's Option 0 also carries
this item's own first step, the angle between the Anderson coefficients under the two
quadratic forms, because the dump it needs is the same dump.

### F2. Converging a noncollinear SCF is four problems, and the code preconditions one of them [new, opened 2026-09-14]

**Item F asks whether the mixer's metric is the reason a magnet takes twice the iterations
`pw.x` takes, and it has stayed open because the question is put one level too low.** What
decides how fast a damped fixed-point iteration converges is not the mixer, it is the
spectrum of the map the mixer is damping: a density error `e` comes back as `(1 - beta) e +
beta J e` with `J = chi_0 K`, so what matters is where the eigenvalues of `J` sit, and they
go wrong at **both** ends for different reasons.

**At one end `|J| >> 1`, and at the other `J -> 1`, and no single knob serves both.** Charge
sloshing is the first: the Hartree kernel makes one eigenvalue about `-q_TF^2/q^2` at long
wavelength, so the iteration diverges there unless `beta` is pulled down for the *whole*
vector, which is why the fix is a preconditioner that compresses that end rather than a
smaller step. The three magnetic directions below are the second: `J` approaches one from
below, `(1 - beta) + beta J` approaches one whatever `beta` is, and the residual `(J - 1) e`
vanishes along the direction, so a mixer that sees only the residual cannot see the error at
all. That is one sentence for two things this project has already measured separately: why
Kerker is structurally irrelevant to the magnetic end, and why `OPEN.md` Part VI item 3 found
the mixer stuck rather than unstable and found a *large* `mixing_beta` to be the way across.
Kerker, `mixing_beta`, Anderson's history depth and the Gram matrix's metric are four answers
aimed at different parts of that spectrum, and choosing between them without first saying
which end a given cell is slow at is what has made every previous attempt here a sweep over
knobs.

**In a noncollinear cell there are four such directions, they have four different physical
origins, and only the first of them has anything acting on it.** This is the reframing the
rest of the item rests on, and each line is a claim about the physics that can be checked
independently of any code:

- **Long-wavelength charge.** The Hartree kernel goes as `4 pi e2 / q^2`, so a charge error
  of wavelength `L` is amplified by roughly `q_TF^2 / q^2`, which is charge sloshing.
  Kerker (`approx_screening`) and `local-TF` (`approx_screening2`) divide it out, both are
  here, and `PERFORMANCE.md` has what each is worth on the aluminium slab.
- **The longitudinal magnetization, meaning the *length* of `m(r)`.** Here the kernel is the
  exchange-correlation one, which is local rather than `1/q^2`, so there is no long
  wavelength divergence at all and Kerker has nothing to say. What there is instead is the
  Stoner enhancement: the interacting susceptibility is `chi_0 / (1 - I chi_0)`, so a cell
  sitting near `I N(E_F) = 1` amplifies a *uniform* change in the moment by a factor that
  diverges at the transition, at every wavelength equally. This is the direction every
  itinerant magnet in the benchmark set is slow in, and there is **no preconditioner** on it:
  `beta * head[c]` in both routines is a step length, not an approximate inverse Jacobian.
  Anderson's secant fit does span this direction, which is the calibration item F already
  gives, so the claim is the narrow one, that nothing *conditions* it before the fit sees it.
- **The rigid rotation of every moment together, at `Q = 0`, without spin-orbit coupling.**
  A Goldstone mode of the broken spin-rotation symmetry: the restoring force is exactly
  zero, the fixed point is a two-parameter family rather than a point, and the residual has
  no component along it. `OPEN.md` Part VI item 3 measures this on a four-cell hydrogen
  ultracell and is the reference; the one-line summary is that the mixer is not unstable
  there, it is stuck, and the manifold has to be **traversed**, which is why a *large*
  `mixing_beta` is the right reflex and a small one is the wrong one.
- **The transverse channel at finite `q`, meaning a slow twist of the direction of `m(r)`.**
  This is the one nothing in this project has written down, and it is the one the hard cells
  live in. Rotating the moments by an angle that varies with wavevector `q` costs a
  spin-wave energy `D q^2` per moment, so the restoring force vanishes as `q -> 0` and the
  amplification goes as `1 / (D q^2 + K)`, with `K` the anisotropy gap that spin-orbit
  coupling opens. Three consequences, and they are what make this worth separating from the
  rotation above: a **large** cell is worse than a small one, because `q_min` goes as
  `1/L` and the softest available twist gets softer; spin-orbit coupling helps but only by
  `K`, which is milli-electronvolt scale, so the direction becomes stiff compared with
  exactly zero and not compared with the charge; and where `D` is **negative**, which is
  every cell whose ferromagnet is unstable to a spiral, the iteration is not slow in that
  direction but genuinely divergent, and damping it is the wrong thing to do because the run
  is supposed to move there.

**The fifth entry is not a direction, it is a blind spot, and it is the one that applies to
the run this came from.** `becsum` is mixed at the plain `beta` and appears in no
convergence measure at all, since both halves of `accuracy` are of the smooth density and
what reaches them is only what `addusdens` already put on the grid (`OPEN.md` Y2). On a PAW
magnet the moment lives in the d-shell `becsum`, so a stall with a flat magnetic half and an
energy still swinging is exactly the shape that half would make, and there is no number in
the log that separates it from convergence.

**What the code does about the four, in one line each**, because the gap is the argument for
everything below: the charge is preconditioned two ways; the longitudinal magnetization
takes a plain scalar, and it is the *same* scalar the charge takes, which is `pw.x`'s own
rule (one `alphamix` for every component of `mix_type`) and is what VASP and Elk both give
the magnetic channel its own control of; the rigid rotation has a warning naming the
mechanism and nothing
acting on it; and the transverse channel has neither. The magnetization is deliberately not
Kerker-screened, and that decision is right as far as it goes (stage 3a's reason: Kerker
would damp the long wavelengths a magnetic run has to move in), but "not Kerker" was allowed
to stand in for "nothing", which is a different statement.

**The internal tie worth keeping, and it is narrower than it first looks.** The quantity that
sets the conditioning of the transverse channel is the transverse spin susceptibility, and
this code computes it for a **collinear ferromagnet**: P63's magnon response is `chi_perp(q)`
itself, and item D of this file is the reminder that it refuses a noncollinear ground state,
which is the regime this item is about. The stiffness `D` is the small-`q` curvature of
`E(q)`, and what P86 has is a single pair of wavevectors rather than a curvature, at
`q = 1/4` of the zone where the quadratic form is not expected to hold, and the two codes
disagree about even that: defumat gives `E(1/4) - E(0) = -20.712` meV against Elk's
`-136.294`, a factor of 6.581 that P86 records as unresolved. So the tie is a direction to
pull rather than a number to use: a proper small-`q` scan on a cell with a stable
ferromagnet would give `D`, and `D` is what says in advance which cells are slow in the
transverse channel.

#### Option 0, and every other option's decisive number depends on it

**Two runs and one dump, and it is hours rather than a phase.** Nothing below can be chosen
on argument, because the four directions call for four different operators and no cell here
has ever been told apart on which one it is slow in.

- **The nonmagnetic twin of `tests/data/qe/fe-noncolin-pbe-stress.in`.** That cell takes 43
  iterations where `pw.x` takes 19, and item F already records that the 25-against-12 on
  `fe-mag-1k` turned out to be mostly *not* magnetic once `fe-unstable-nonmagnetic.in` was
  run at 21. The same deconfounder for the noncollinear cell does not exist, so the 43 is
  attributable to nothing yet. `benchmarks/` already carries two such twins
  (`fe-unstable-nonmagnetic.in`, `ni-u-nonmagnetic.in`), so this is a committed input and a
  run, not a design.
- **One noncollinear run's residual history, split five ways per iteration.** Charge,
  `dm` parallel to `m_in(r)` pointwise, `dm` perpendicular to it, the rigid-rotation part on
  its own, and the magnetic part of `becsum`. The first four say which of the four directions
  is still moving; the fifth is the blind spot above, and on a PAW magnet it is the only one
  that can be the answer with nothing in the log to show it. The whole decomposition is
  pointwise against the input magnetization and costs one pass over the grid.

  **The rotation bin is the one that has to be defined carefully, and the obvious definition
  is a null that cannot be told from a pass.** Taking it as the `Q = 0` transverse component,
  which is how `OPEN.md` Part VI item 3 writes it, is right for a ferromagnet and is
  identically zero for every compensated texture: an antiferromagnet, a spiral and the NiBr2
  helix all have `m_{Q = 0} = 0`, and a rigid rotation leaves it zero, so the bin reads zero
  whether or not the mode is live. The definition that works for any texture is the projection
  of the residual on the three **generators** of the rotation,

      c_a = ∫ dm(r) · (e_a × m_in(r)) dr / ∫ |e_a × m_in(r)|^2 dr,   a = x, y, z,

  which reduces to the transverse `Q = 0` component for a ferromagnet and costs the same one
  pass. **That is also a new fact about the projection option below**, which this item
  otherwise defers to `OPEN.md`: the mean-moment form written there is ferromagnet-only, and
  a helix needs the generator form.
- **The trip test for the rotation bin, because the obvious one does not fire either.**
  Rotating the input of a spin-rotation-invariant functional rotates its output with it,
  `F(R rho) = R F(rho)`, so the residual of a rotated state is the rotated residual and is
  *not* concentrated in the rotation bin. Two tests that do fire: the decomposition of
  `R rho - rho` at a small angle, which must lie entirely in the generator bin and nowhere
  else, and a rotated **converged** state fed back as input, whose residual must be zero to
  the level the run converged at. A collinear run of the same cell must give exactly zero in
  that bin, which is the other half of the check.
- **Item F's own first step, in the same dump.** Recompute the Anderson coefficients under
  the Euclidean form and under `rho_ddot`, and report the angle between the two coefficient
  vectors. A small angle closes F as measured rather than fixed, and it is free once the
  history is on disk.

#### The options

Each is written with what it buys, what it departs from, and the cell and number that would
decide it, because a departure from `pw.x` needs a number rather than an argument.

- **A separate `mixing_beta` for the magnetization.** VASP's `AMIX_MAG`/`BMIX_MAG`, which
  exists because one scalar for charge and moment is known not to serve both. Cheapest thing
  on the list: one input variable, one extra argument to the two preconditioners, no new
  physics. It addresses the longitudinal direction by brute force and the transverse one by
  accident. Decided by iterations on `fe-noncolin-pbe-stress` at fixed charge `beta`, with
  the ceiling set by Option 0: if the dump says the residual is charge-dominated, this buys
  nothing and should not land.
- **Elk's `mixadapt`. ✅ DONE, and it is the largest single number this item has.** Per
  component of the mixed vector, `beta_j` grows by `beta_0` while the residual keeps its sign
  and is halved toward `beta_0` when it flips, so the step lengthens on its own along a
  direction that is not turning around, which is what a flat manifold looks like from inside.
  A stall detector with no threshold in it. On `fe-noncolin-pbe-stress.in` it takes **16**
  iterations against the best `anderson` on that cell, which is 24 at `mixing_beta = 0.7`
  (and against 43 at the input's own 0.2, which is the number a user meets but is not the
  like-for-like one, since 0.2 is in that input because QE wanted it there). `pw.x` takes 19.
  Same energy within 3e-10 Ry, same moment length within 3.3e-5 mu_B. **The iteration ratio overstates it**: the
  Davidson work falls by 1.60x rather than 2.69x, because `ethr` is scheduled from `dr2` and
  a faster-falling residual buys a tighter eigenproblem. `PERFORMANCE.md` has the full table
  and the two honest headlines. Three things stated rather than discovered: Elk mixes the
  **potential** and this mixes the density; the parameter an input file's `mixing_beta`
  reaches is Elk's `beta0`, an increment and a floor rather than a step length, so the same
  number means two things depending on the mode; and it does not compose with Kerker, being
  pointwise in real space, so it is an alternative to the preconditioned mixer rather than a
  layer on it and the pair is refused. Still worth fetching before anyone writes a hybrid:
  Elk's `mixtype = 4`, the parameter-free "robust adaptive mixer" described as converging
  almost anything, which is **not** in the vendored 11.0.2.

  **And it failed the case this item said would decide it**, which is worth more than the
  iron number. On the four-cell hydrogen ultracell of `OPEN.md` Part VI item 3, the flat
  manifold itself, it converges in 165 to 172 iterations against `anderson`'s 265 and to a
  **different state**: the per-cell moment is 0.186 mu_B against 0.978, a factor of five in
  the *length*, so it is not the rotation the manifold is made of. Two values of `beta0` a
  factor of four apart agree to 6e-5, and it is the **mixer** rather than the step length:
  `anderson` run at 0.1, 0.3 and 0.7, a factor of seven, sits between 0.98 and 1.00 at all
  three. Both `adaptive` runs report converged three orders below `conv_thr`.

  **The 0.62 that was held against those numbers is a different cell, and that half is
  withdrawn** (2026-09-15). It is the unit cell's moment on the `(4, 2, 2)` k-grid every
  stage 3a and 3b test folds to; this comparison ran at `(4, 1, 1)`, where the same cell is
  **saturated** at 1.0000 and the ultracell starts there, so `anderson`'s 0.98 to 1.00 is the
  reference state rather than a state it found. `PERFORMANCE.md` has the grid table. On the
  unit cell itself both mixers give the same moment and the same total energy to ten digits at
  every grid and `beta` tried, so the mixer is cleared of the general charge. **And the
  comparison cannot be repeated**: neither record wrote down the field's functional form, and
  taking the description literally -- 0.01 Ry rotating 90 degrees per cell, `kerker = False` --
  all five runs converge to one state, `|m|` 0.461 to 0.462 turning -41 degrees per cell, with
  no factor of five anywhere. **So the flat manifold is still unmeasured** -- this item's decisive run has
  not been done, it has been attempted and invalidated -- and the projection option below is
  not displaced by the mixer. What the attempt did establish is a property to carry into
  every other option here: **a step that grows can change which solution is found, and
  `converged` does not say otherwise.**
- **Projecting the rigid rotation out of the magnetic residual.** Fully specified already in
  `OPEN.md` Part VI item 3, including the `lspinorb` gate (the mode is gapped there and the
  projection would be actively wrong) and the decisive run (0.002 Ry at `mixing_beta = 0.7`,
  263 iterations as it stands). Not respecified here. Its scope is worth repeating because it
  excludes the cell that prompted this: a run with spin-orbit coupling does not have this
  direction, so this is not the fix for the NiBr2 helix.
- **A per-iteration controller hook, with the hybrid policy as its first client.** The
  deliverable is the hook, not the policy: a callback that receives the `history` entry,
  which already carries the charge and magnetic halves of `accuracy`, the site moments and
  the Davidson step counts, and may change `mixer.beta`, clear the Anderson history, or swap
  the mixer. A deterministic policy is then one function against that interface, for instance
  raising `beta` after `n` iterations in which the magnetic half has not fallen and the
  `Q = 0` transverse component has kept its sign, and lowering it and resetting the history
  on a blow-up. Two constraints that are not optional: every decision the controller takes is
  written into `history`, so a run stays reproducible and a claim about what helped is
  checkable after the fact; and the controller's own state goes into the checkpoint beside
  the mixer history, or a resume silently restarts the policy.
- **Herbst and Levitt's adaptive damping, as the principled version of the same hook.** A
  backtracking line search on the damping, with a quadratic model of the energy along the
  search direction whose coefficients are built from `rho(V_n)` and `rho(V_n + alpha dV_n)`,
  quantities the next iteration needs in any case, so an accepted step costs no extra
  diagonalization and only a rejected one does. The step is accepted when either the energy
  or the preconditioned residual falls, which is what keeps it from reverting the useful
  steps Anderson takes late in a run, and their hard test cases are Heusler compounds and
  transition metals, which is the class this item is about. Two departures to weigh before
  promising it: the analysis and the algorithm are for **potential** mixing where this code
  and `pw.x` mix the density, and the functional whose decrease is guaranteed is the
  grand-canonical free energy, so the entropy term is part of it on every smeared metal here.
  Neither is fatal and both mean this is a phase rather than an afternoon.
- **The agent in the loop, and where it actually belongs.** Inside the SCF an agent is slow,
  non-reproducible and untestable by this project's own standard, and the moment its decisions
  are written down well enough to be tested it *is* the deterministic controller above. Between
  runs is a different matter and is available today with no code: `_MIXER_DERIVED` is
  `{"beta", "history", "condition_limit", "precondition"}`, so the checkpoint deliberately
  does not store the mixer's settings and `get_mixer` rebuilds them, meaning a
  `mixing_from` resume can change `mixing_beta`, `mixing_mode` and `mixing_ndim` while
  keeping the Anderson history the run had earned. The first deliverable on this option is
  therefore a protocol rather than a feature: run `n` iterations with `checkpoint_dir`, read
  `history`, resume with different settings, and keep the trace. It is also the honest way to
  *discover* the policy the hook should implement, since a session driving that loop by hand on
  a hard cell is an experiment whose log is exactly the training data a rule would be written
  from.
- **The Stoner preconditioner for the magnetic channel, generalized to a spinor.** Barat,
  Levitt and Torrent (arXiv 2606.26693, June 2026) build a hybrid `P = I - chi_0^LDOS K_H -
  chi_0^diag K_xc`, where the charge keeps an LDOS-based long-range preconditioner and the
  magnetic channel gets a local susceptibility made of an eigenvalue-variation term
  `sum_i f'(e_i - e_F) |rho_ii><rho_ii|` and a Fermi-level term, with the expensive orbital
  variation dropped. That is the longitudinal direction treated properly, and their result is
  the elimination of convergence plateaus near a magnetic transition. Two facts to carry: the
  paper is **collinear only**, so the four-component generalization is ours to write and is the
  bulk of the work; and because it drops the orbital variation it cannot see a rotation, which
  is entirely orbital variation, so it does not replace the projection or the controller. Size
  it as a phase, with `n_active x nr` orbital densities in the smearing window on the grid, and
  measure it on iron near its transition where the paper has plateaus to compare against.

#### What binds any of it

The five deliverables of a finished phase apply unchanged, and three of them are worth naming
here because this is a convergence feature and convergence features are where they go stale.
The number is **iterations to a fixed `conv_thr` on a named cell against `pw.x` on the same
input**, never a wall clock and never a ratio against a previous version of this code. Every
switch proposed above, the `lspinorb` gate included, is tested by feeding it a case that must
trip it rather than by reading a clean result as a pass. And any of these that changes what a
converged run *is*, rather than how it got there, has to show the same total energy and the
same site moments as the unmodified route on at least one cell, to the level the two agree at
now.

### G. The noncollinear derivative memory wall. ✅ The suspicion in this item was right, and it is fixed [22]

**Closed as diagnosed, 2026-09-13.** This item's own caveat -- "P73 replaced the *stored*
`Q_ij(G)`, while `augmentation.py` says the reverse-mode cost lives in `_qrad_kernel`'s
`(ngm, kkbeta)` intermediate, which the table route **evaluates** rather than avoids -- so
P73 may not have helped the backward pass at all" -- was correct in its conclusion and
wrong about the mechanism. The backward-pass cost was not `_qrad_kernel`'s intermediate: it
was the `lax.scan` **stacking its residuals**, so the dense table came back on the tape at
`>= 2 GiB` by construction. Both scan bodies are now rematted, and
`bismuthene-soc-small`'s force tape goes **2.32 GiB -> 0.99 GiB** by the compiler's own
`memory_analysis()` (`PERFORMANCE.md`, `MEMORY-AUDIT.md` A1).

**What is still owed here** is the item's original ask: an end-to-end spinor force on
`bismuthene-soc` itself, which is the cell P46 recorded as not running at all. The tape is
no longer the blocker it was, but nothing has been run on that cell since, and the P46
figure stands until something is. Run it under
`systemd-run --user --scope -p MemoryMax=...`, on an idle machine, and commit first.

### G2. The old text of item G, kept for its reasoning [22]

**Session for the measurement, phase for a derivative term in `SizeEstimate`.** `PLAN.md`
P46 records a bismuthene spinor force taking free memory from 24 GB to 0.65 GB and being
killed three times, because the augmentation table `Q_ij(G)` — `nh² × ngm` per atom, with
`nh` in the twenties for a fully-relativistic dataset — is live through the backward pass.
That predates P73's radial table and P74's band batching, which took the 45-atom NiBr2
noncollinear PAW cell from 78.51 GB to 32.30 GB peak. The two data points that stand are
three orders apart: doubled fcc platinum (204 bohr³, ultrasoft, `lspinorb`) runs a spinor
force in 33 s, and bismuthene (1770 bohr³) was killed at a commit before both phases.
`Calculator.estimate()` answers only for the SCF and says so, which is the "state, measure,
make selectable" rule being followed rather than a defect; what is missing is any
post-P73/P74 number.

**First step, and mind the caveat.** `tools/gpu/force_memory.py` on `bismuthene-soc-small`
twice, once at `DEFUMAT_AUG_MAX_BYTES=off` and once at a value small enough to force the
table, comparing the reported buffer — that settles both the staleness and the possible
non-monotonic cliff at the 2 GB default in one pass. The caveat: P73 replaced the *stored*
`Q_ij(G)`, while `augmentation.py` says the reverse-mode cost lives in `_qrad_kernel`'s
`(ngm, kkbeta)` intermediate, which the table route **evaluates** rather than avoids — so
P73 may not have helped the backward pass at all.

**This item has killed sessions.** Run it under
`systemd-run --user --scope -p MemoryMax=...`, on an otherwise idle machine, and commit
first.

---

### H. The magnetic quantities that are done "for one regime", collected [new, from `PLAN.md` §3's index]

**This file never listed these and it should have.** Each is a phase marked DONE whose
heading carries a qualifier, and a qualifier in a heading is an open item that nothing
tracks. They are smaller than items A-D and they are the ones a user meets first, because
each is a `get_*` that works on the cell in the tutorial and refuses the cell they brought.

| phase | done for | open for | what is missing |
|---|---|---|---|
| **P57** magnetoelectric tensor | the column **parallel to the field**, spin-only, clamped-ion | the other two columns, the lattice-mediated part, and any external calibration | it is uncalibrated against another code, which is the part to fix first: Elk's `magnetoelt.f90` is the counterpart and is built here |
| **P58** magnetocrystalline anisotropy | the **frozen-density force theorem**, norm-conserving and ultrasoft; and, as of **P87**, the **relaxed** route, which reaches PAW and DFT+U | a `pw.x` pair for the relaxed number, and `average_pp` | ✅ mostly closed by P87: 0.447 meV relaxed against the theorem's **free** energy 0.552 on the same cobalt cell, with a measured 0.011 meV floor. What is left is an external check and `average_pp`, which belongs to the *frozen* route |
| **P63** magnons | **collinear**, norm-conserving | noncollinear (item D), and ultrasoft/PAW | the 4x4 spin response does not block-diagonalise off a collinear axis; ultrasoft needs the augmentation charge inside the transverse channel |
| **P64** orbital magnetization | **norm-conserving** | ultrasoft and PAW | the neighbour overlap the covariant derivative is built from needs the augmentation term `bp_c_phase.f90`'s `q_ij(b)` supplies, which the Berry-phase polarization already has and this does not reuse |

**The pattern is worth naming rather than fixing four times.** Three of the four are the
*same* missing term -- an ultrasoft or PAW augmentation charge inside an object built from
wavefunctions at two different k-points or two different perturbations. P47's Kubo Berry
curvature (`e_n dS/dk`, "written and unvalidated") is a fourth instance. A single validated
augmented-overlap primitive would close parts of all of them, and `bp_c_phase.f90` is the
reference for it in every case.

## 3. Questions that need something run, not something written

These are `NONCOLLINEAR.md` §6's open questions, with the ones P77–P79 answered removed.
None is a phase; each is one or a few runs, and several are an afternoon.

**Q1 — has a *vector* texture ever survived an SCF? Yes; closed by P80.** The pair is
`h4-cycloid-90.in` and `h4-cycloid-90-nosym.in`: 0.4543 mu_B per site either way, four
moments 90.00 degrees apart, the two total energies 8.0e-9 Ry apart, and the symmetrised
run planar to **1.7e-21** where the free one reaches only 8e-6 -- the four surviving
operations forbid the out-of-plane component, so symmetry buys exactness here rather than
costing physics.

**What did not survive is this question's own discriminator, and that is the finding to
carry forward.** "Two nonzero singular values means the texture held" is **scale-free**. The
same cell at 3 bohr spacing instead of 5 gives `[3.235e-4, 3.231e-4, 1.4e-23]`, neighbour
angles 90.06/89.89/89.99/90.05, and the two runs agreeing to 1.1e-8 Ry -- a perfect cycloid
by that criterion, and *nothing*: the moments had fallen from 0.1465 at iteration 1 by a
factor of 450, because a hydrogen chain is not magnetic at 3 bohr. Assert the **pair**:
`sigma_2/sigma_1` for the shape, and `sigma_1` against its own value at iteration 1 for
whether there is anything left to have a shape. 5 bohr reads (0.99998, 1.012) and 3 bohr
(0.99988, 2.2e-3).

**Q2 — does one survive on a production magnet?** [O2] P75's own outstanding item: the
45-atom NiBr2 cycloid has not converged to a textured state at any k-mesh. The run that
supplies the figure is the open question, not the feature.

**Q3 — how much does the symmetriser remove? All of it; closed by P80.** It removes the
magnetization *entirely* -- residual **1.0**, every site moment from 0.460 mu_B to 1.7e-18 --
because averaging four directions 90 degrees apart over a group that permutes the four sites
gives zero rather than something smaller. From a `STARTING_MOMENTS` card the same figure is
6.7e-16, at both group sizes. `Calculation.symmetry_residual` reports the charge and the
magnetization **apart**, and that is not cosmetic: in the failing case the charge residual is
5.9e-16, so a residual computed on the density as one object reads as a clean pass.

**One correction to this question as it was posed.** It asked for the *first output* density.
That is the wrong array: an output density is a wedge sum and is not invariant by
construction -- putting the rest of the zone back is what `sym_rho` is for -- so the same
number there is large exactly when symmetry is working and cannot be told from the failure.
The seed is where there is no ambiguity, and it is also *earlier*: the warning fires before
iteration 1 rather than at iteration 6.

**Q4 — is the noncollinear GGA branch correct at all? Yes for the *signed* branch; closed by
P80. The unsigned one is still open and the obstacle is `pw.x`.** bcc iron with its moment in
the plane, ultrasoft, PBE (`fe-noncolin-pbe-stress.in`): the energy agrees to **6.7e-9 Ry**
and the **stress** — the only derivative a one-atom bcc cell has — to **1.6e-7 Ry/bohr^3**,
0.019 kbar out of 152.71, which is the same level the collinear ultrasoft cases reach on the
same quantity (2.7e-7, 2.4e-7).

**There are two branches and this question did not know it.** `compute_ux` takes a fixed
quantization axis whenever the starting moments are all parallel, and `compute_rho` then uses
`(n ± sign(m·ux)|m|)/2` — *signed*, removing the cusp `|m|` has at a node. One atom is
trivially parallel to itself, so iron runs the signed branch; both codes agree on the axis.
Plain `|m|` is what a genuinely **canted** cell takes and it is the one P77a's guard is about.

**Two attempts at it failed on the `pw.x` side, not here.** The obvious cell — this
question's own suggestion, `h4-noncolin-force.in` with `input_dft = 'PBE'` — limit-cycles at
an accuracy of **5e-6 Ry**: 100 iterations at `mixing_beta = 0.3` and 300 at 0.1. A two-atom
version reaches 5e-8 and no further. Both converge to 1e-11 under **LDA** in 62 iterations, so
the magnetic state is frustrated under PBE on a hydrogen chain. **First step:** a cell whose
canted PBE state converges in `pw.x` — iron or nickel sublattices at 90 degrees rather than
hydrogen, or the same hydrogen cell held by `constrained_magnetization = 'atomic'`, which both
codes can state per species (and whose penalty is then outside both totals, which is the
hazard to check before trusting agreement).

**Try a PBE dataset before reaching for a constraint.** Both failed attempts ran
`H.pz-vbc.UPF` — an *LDA* pseudopotential — under `input_dft = 'PBE'`, which was deliberate
(the same inconsistency on both sides isolates the functional) and is also a candidate cause
of the limit-cycle in its own right: a dataset generated for one functional has the wrong core
under another, and the gradient correction is what feels that most. `H.pbe-hgh.UPF` is
committed. Rule that out before concluding the cell is frustrated.

**Q5 — ultrasoft or PAW with several non-parallel moments has no external number in any
regime.** [O13] Take `fe-kind1-noncol.in` (`Fe.rel-pbe-spn-rrkjus`), build a two-atom
antiferromagnetic version (`angle2 = 0` and `180`) and a canted one at 90 degrees, and
generate a `pw.x` reference for each; compare total energy, both site moments and the
forces. Until then, everything that only turns on in that combination — `add_becsum_so` on a
textured `becsum`, `qq_so` in a textured overlap, PAW's one-centre local spin frame with
several different frames in one cell — has no external number. **This is the largest
validation gap in the noncollinear stack.**

**Q6 — what is a reduced k-set worth on a spinor run?** [O14] Every refusal in item C routes
the user onto the whole grid and `PERFORMANCE.md` has no entry pairing a symmetrised spinor
run against its `nosym` twin. The measurement exists in embryo: `test_spinor_forces.py`
already builds both the 8-point wedge and the 32-point closed grid of `pt2-soc-force.in` and
runs both to `conv_thr = 1e-12`. Time that pair and record its peak RSS. It is also what
says whether item C is worth a phase.

**Q7 — what do the noncollinear slow tests say today?** [O11] `tools/run_regression.sh`
restricted to the nine noncollinear files with `DEFUMAT_TEST_MEM_MAX=20G`, keeping the
durable per-file summary line as the dated record. P46 measured `pt2-soc-paw-force` at
12,204 MB alone and the pair at 16,961 MB, over the 12G default, so the cap matters.

**Q8 — is `conv_thr = 1e-12` the right default for a magnetic run?** [new, `OPEN.md` Y1]
Three separate measurements the same day found a magnetic quantity converged two to four
orders more loosely than the energy beside it: nickel's `|<L>|` spread 2.3e-7 at
`conv_thr = 1e-10` where the energies agreed to 6e-12; a promoted DFT+U state and its source
agreeing to 6e-9 Ry and 6e-5 in `Tr ns`; and a from-scratch spinor run landing 6.5e-6 Ry
above the promoted one in a neighbouring canted minimum. The mechanism is item [18]'s
weighting, and `scf.history` now carries the two halves. What is open is whether the `ethr`
schedule should be driven by the **magnetization half** rather than the sum on a magnetic
cell. `pw.x` uses the summed `rho_ddot`, so this would be a deliberate departure rather than
a correction, and it needs a number before it is anything.

---

## 4. Deliverables owed

**The notebook, for `STARTING_MOMENTS` and everything built on it.** `CLAUDE.md` requires
one per feature and this is deliverable 4 of five for the feature the whole of P77–P79 rests
on. **It is blocked by the checkout, not by the work**: `07_spin_polarization` and
`11_noncollinear_magnetism_and_fields` were both checked on 2026-09-12 and both build their
`Calculator` from `../quantum_espresso/qe-7.5-ReleasePack/qe-7.5/test-suite/...`, which is
gitignored and absent here, so neither can be re-executed and
`tools/export_notebooks.sh` cannot refresh them. Either write a new notebook sourcing only
`tests/data/qe` and `tests/data/pseudo`, which does run — the cells this work added
(`h2-texture-120.in`, `h4-chain-ferro.in`, `h2-mirror-afm.in`) are all there and all cheap —
or restore the vendored tree first. **Do not add a cell to one of those two and commit it
unexecuted.**

**Site moments are not reported by every driver. Closed by P80 for the relaxations; the
response stack is still open.** `RelaxStep`, `VCRelaxStep` and `SpiralRelaxStep` carry
`site_charges` and `site_moments` at each of their own steps, and the console line carries
the SCF's own `|m|_site = min..max` suffix. The response stack does not, and has no obvious
step object to hang them on.

**`colin_mag = 2` / `t_rev` is not implemented for a collinear run.** P77 added the
collinear magnetic filter in its `colin_mag == 1` form, which discards time-reversed
operations. Keeping them with `t_rev = 1` needs the channel swap `new_ns` performs, and no
committed benchmark exercises it — the magnetic sublattices in all of them are different
species.

**A relaxation does not recheck the magnetic group, and does not need to** — settled by
reading in P79: the moments are a fixed tuple in `ATOMIC_POSITIONS` order so a moving atom
carries its own row, and `checkallsym` verifies the permutation the magnetic condition is
stated in. The one hole is that `_maps_structure` tests a *setwise* map rather than pinning
the permutation, so two atoms of the same species exchanging roles would slip through. A
continuous relaxation cannot do that; nothing else here can either. Recorded so that the
absence of a check is not read as an oversight.

---

## 5. The traps to carry into any of it

Short list, all of them paid for in this stack rather than inherited from `CLAUDE.md`'s
general one.

- **A guard's clean zero is not a pass.** Test that the guard *fires* — feed it a case that
  must trip it. `'atomic direction'` returns exactly 0.0 for a cycloid **and** for the
  collinear state it should exclude, because QE's `i_cons = 2` constrains `m_z/|m|` alone.
- **An array-algebra test passes whether or not the driver routes anything through it.**
  `promote_ns` could have been right in every element while every resume was silently
  collinear, and was. The end-to-end path is the test.
- **A diagnostic must not change the run it is diagnosing.** Splitting `accuracy` and adding
  the halves back in Python differs by one ulp, and `accuracy` drives the `ethr` schedule.
- **A converged `dr2` bounds a moment far more weakly than an energy** — two to four orders,
  measured three times. Set a magnetic tolerance from a measurement, never from the energy's.
- **Constraining a direction is not the gentle option.** A direction-only penalty carries
  `1/|m|`, so a shrinking moment is amplified: positive feedback, and no `lambda` converged.
  The vector penalty's gradient is bounded and it is the one that works.
- **Inherit a refusal only after checking which machine it belongs to.** Twice now: P35's
  Sternheimer refusal left P54 marked impossible, and P20's `ns_nc` refusal closed the DFT+U
  continuation for fourteen phases after P62b removed the blocker.
