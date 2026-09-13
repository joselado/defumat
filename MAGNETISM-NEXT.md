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

---

## 2. The queue, in order

### A. No linear response for a spinor, so a magnet with spin-orbit coupling has no phonons and no spectra [11]

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

### B. Elk's per-atom feedback field, so a held texture is exact rather than nearly [10, remaining half]

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

**(c) is run. A spiral accepts a constraint; whether `fsm` can *hold* one is still open,
because this cell cannot answer it — `fsm` fails at `q = 0` too. P80.**

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

**P78 took the first measurement against this and it points away from it.** The deconfounder
`benchmarks/fe-unstable-nonmagnetic.in` — same cell, same dataset, same `mixing_beta = 0.3`,
`nspin = 1`, where a magnetization weighting cannot act at all — takes **21**. So most of
the excess over `pw.x` is not magnetic, and whatever the metric is worth it is bounded by
the four iterations between 25 and 21 rather than the thirteen between 25 and 12. Raising
`mixing_ndim`, the other obvious lever, makes both cells *worse*: 27/25/33/39 at 4/8/12/20
on the magnetic one and 26/21/30/26 on the nonmagnetic.

**A second cell shows the same 2:1 ratio, and it has no deconfounder yet.**
`fe-noncolin-pbe-stress.in` (P80) takes **43** iterations where `pw.x` takes **19** at the
same `mixing_beta = 0.2` and the same `conv_thr = 1e-10`, with the two energies agreeing to
6.7e-9 Ry — noncollinear, ultrasoft, PBE, where `fe-mag-1k` is collinear. Two cells at 2:1 is
worth more than one, and the nonmagnetic twin of *this* cell has not been run: without it the
43/19 is no more attributable to anything magnetic than the 25/12 was.

**First step, if it is still worth one.** Dump one run's residual history and recompute the
Anderson coefficients under both quadratic forms — `scf_accuracy` gives QE's for free — and
report the angle between the two coefficient vectors. If it is small, the metric is not the
mechanism and this item can be closed as measured rather than fixed.

### G. The noncollinear derivative memory wall, unmeasured since two memory phases moved it [22]

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
