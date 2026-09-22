# Magnetism: what is open, ranked by impact (2026-09-22)

## What this file is

A snapshot of the open magnetism items on 2026-09-22, in one list ordered by what a wrong
or missing answer costs, with the weight on the runs this project actually does, which are
noncollinear magnets and spirals on ultrasoft and PAW datasets. `MAGNETISM-NEXT.md` is the
sized queue with the reasoning behind each item and its first step; this file is the
ranking alone, one entry per item, so that a session picking something up starts from the
order rather than re-deriving it. Where the two disagree on a number, this file is the
later one, and the last section says where `MAGNETISM-NEXT.md` has drifted.

Each entry gives the physics first, then where it stands, then what it needs. Phase numbers
refer to `PLAN.md` §3, letters and Q-numbers to `MAGNETISM-NEXT.md`.

## What the top four now say (worked 2026-09-22, same day)

Items 1 to 4 were taken in this order and all four have measured answers. The
records are `PLAN.md` P102 to P105; this is the one-line version.

1. **Q5, the textured augmented spinor.** Half closed (**P103**). On the new
   two-atom iron antiferromagnet, against `pw.x`: total energy to **6e-9 Ry**,
   both site charges and the first site moment to every printed digit, the second
   site moment to 1.3e-5 mu_B, the force to 3.3e-7 Ry/bohr. The canted leg is the
   half the item is really about and needs a **constrained** angle rather than
   more iterations, since nothing protects 90 degrees on that cell.
2. **The spiral against Elk.** **Closed (P105), and the answer is a convention.**
   Elk's entropy term is non-zero only for Fermi-Dirac smearing and the fixture
   uses Gaussian, so Elk reports an internal energy where this code and QE report
   a free energy. At `degauss = 0.1 Ry` the entropy moves by 112 meV between
   `q = 0` and `q = 1/4`, five times the difference it was compared against;
   removing it gives **-132.986 and -263.813 meV against Elk's -136.294 and
   -265.206**. All four previously named candidates were measured and are dead,
   including a held field that had been converted **274 times too large**.
3. **The textured spinor dielectric tensor.** One suspect removed (**P104**). The
   longitudinal spin susceptibility from the screened response and from a central
   difference of two converged SCF runs agree to **0.238 per cent**, so the kernel
   and its self-consistency are right along the direction `ph.x` disagrees in. The
   two suspects left are the electric-field source term and `ph.x`. A transverse
   number of -658.9 mu_B/Ry was first read as an instability and **that reading
   was refuted the same day**: the moment is locally stable and the sign belongs
   to a nearly singular solve.
4. **Converging a noncollinear SCF.** Option 0 done (**P102**). The magnetic cell
   takes 43 iterations and its nonmagnetic twin **15**, so 28 of the 43 are
   magnetism where `fe-mag-1k`'s twin left 4 of 25, and the twin is below `pw.x`'s
   19. The slow direction is the **longitudinal** one, the rigid rotation is not
   excited on that cell, and **`becsum` grows fourfold mid-run** while every grid
   bin falls, with nothing in the log able to see it. Its units differ from the
   density's, so its rate is comparable and its magnitude is not.

Still running when this was written: the canted iron leg on both codes, the
collinear arbiter (which tests Elk's collinear path against Elk's own spiral, not
P105), and nothing else.

## The ranking

1. **An ultrasoft or PAW cell with several non-parallel moments has no external number in
   any regime** (Q5). Every production texture here runs on that combination, so
   `add_becsum_so` on a textured `becsum`, `qq_so` in a textured overlap and PAW's several
   local spin frames in one cell have never been checked against anything, and a defect
   there is a converged plausible number for the wrong physics. It needs the two-atom
   canted iron cell (`Fe.rel-pbe-spn-rrkjus`, `angle2 = 0` and `180`, and a 90 degree
   version) with a `pw.x` reference, comparing the total energy, both site moments and the
   forces.

2. **The spiral energy landscape disagrees with Elk by a factor of five** (P86, item E).
   The moments agree to about 4 per cent at three wavevectors on the matched `1 1 4` grid,
   and `E(q) - E(0)` is out by 5.42 at `q = 1/4` and 4.67 at `q = 1/2` after converging
   the k-grid (`nk = 8`) and the basis (`ecutwfc = 80`) on this side, so it is a
   difference in shape rather than a constant and cannot be a units error or a g-factor.
   `E(q)` is what the NiI2 surface on Triton produces, so this is the item that bears on
   the science directly. Untested: Elk's own `rgkmax` convergence, at about 40 minutes a
   point pinned, and the field convention, Elk's `bfieldc` entering as `(g_e/4c) sigma.B`
   in Hartree against a potential shift in Ry here. The converged defumat moment crosses
   Elk's rather than approaching it (0.573 against 0.538 at `ecutwfc = 80`).

3. **The dielectric tensor of a textured spinor is 5.3 per cent from `ph.x` along the
   moment and the fault is unlocated** (P83, item A2). On `i-atom-soc.in` the component
   along the moment is 1.574482417 against 1.494593593, which is 40 per cent of the whole
   `f_xc` contribution, while the two components across it agree to 4.6e-4 and the tensor
   is uniaxial with nothing imposing it. The solve was cleared on 2026-09-16 by a
   potential-probe finite difference (`scratchpad/iodine_chi0.py`) and the three threshold
   explanations are dead, so what is left is the kernel, the electric-field source term
   for a spinor, and `ph.x` itself. It is refused by name
   (`require_a_measured_spinor_response`), and a wrong kernel would propagate into every
   response built above it on a magnet, which is why it ranks above the class it blocks.
   The tie-breaker is a sum-over-states route on the same cell; the cheaper discriminator
   is the screened density response against a re-converged SCF under a static `+- h dv`.

4. **Whether a textured noncollinear run converges at all** (item F2, Q2, `OPEN.md` Part V
   item 3). Without spin-orbit coupling the rigid rotation of every moment together costs
   nothing and the residual has no component along it, so the mixer extrapolates along a
   flat direction; the 45-atom NiBr2 cycloid has not converged to a textured state at any
   k-mesh, and the decisive three-way run for the projection option, at `nbnd = 32` on
   the `(4, 2, 2)` ultracell grid, has not been done. That grid has three converged
   solutions, so the run must report which state each option lands in rather than an
   iteration count. Elk's adaptive mixer is in and is the largest single number the item
   has (`HANDOFF-adaptive-mixer.local.md`).

5. **Phonons of a spinor, and everything above them** (item A). P81's spinor Sternheimer
   solve has one assembly opted in, the electric-field one (`response/efield.py`,
   `noncollinear=True`), so a `noncolin` run still refuses the dynamical matrix at
   `Gamma`, LO-TO splitting, the strain response, Raman, the elastic constants,
   electrostriction and the piezoelectric tensor. Each needs its assembly validated on
   top of the solve. The augmented half the backlog still lists (`set_int3_nc`) is done
   (P98).

6. **Magnons of a noncollinear ground state** (item D), refused in `tddft/spinchi0.py`
   because the 4x4 spin-density response no longer block-diagonalises off a collinear
   axis, so the transverse channel is not a matrix in `(G, G')` on its own. A helix or a
   120 degree state therefore has a ground state and no excitations of it. The cheaper
   first half is `chi^{+-}` of a spiral at its own `q` in the rotating frame, checked
   against the collinear antiferromagnet of the doubled cell at `q = (0, 0, 1/2)`.

7. **The second derivatives of a collinear magnet** (`PLAN.md` outstanding index). At
   `nspin = 2` the dynamical matrix, the strain response and the two third derivatives are
   missing their assemblies rather than their solve, and a GGA kernel (`dgcxc_spin`) is
   refused with them. The `Gamma` dynamical matrix runs with the guards bypassed,
   reproduces the O-O stretch to 1.4e-6 relative, and is refused because the block a rigid
   translation reaches is wrong.

8. **The unsigned GGA branch of a genuinely canted cell has no external number** (Q4).
   Every canted PBE magnet takes plain `|m|` where a cell with parallel moments takes the
   signed branch validated on bcc iron to 6.7e-9 Ry and 1.6e-7 Ry/bohr^3. Both `pw.x`
   attempts limit-cycled on a hydrogen chain, at 5e-6 Ry, and both ran an LDA dataset
   under `input_dft = 'PBE'`; `H.pbe-hgh.UPF` is committed and should be tried before a
   constraint.

9. **Bismuthene's ground state sits 3.5e-5 Ry from `pw.x`** (`OPEN.md` Part XII item 1)
   where relativistic AlAs sits at 2e-9, with the same grids and both converged. The
   candidates (the radial interpolation floor on ten beta functions with a core
   correction, the `dn` semicore channels, the vacuum) all fit, so none is offered; the
   discriminator is the eigenvalues and a second heavy relativistic ultrasoft cell without
   vacuum. Its dielectric A/B, `bismuthene-epsilon-us-soc.in`, is committed and was
   stopped after 70 minutes.

10. **Any Sternheimer response of a run converged under a magnetic field or a constrained
    moment** (`OPEN.md` A2, `require_a_sternheimer_regime`), refused because the stack
    rebuilds its potential from the field the input asked for where `reducebf` and `fsm`
    make that the wrong field. For a hand-set field the missing part is plumbing, since
    the induced `2 lambda dm` falls out of the existing `jvp`; for `fsm` it is the induced
    field itself, which is a feedback update rather than a derivative.

11. **The end-to-end spinor force on bismuthene itself** (item G). The derivative tape
    went from 2.32 to 0.99 GiB by the compiler's `memory_analysis()`, and nothing has run
    on `bismuthene-soc` since P46's kill, so the slab capability is still a forecast. Run
    it under a memory cap, on an idle machine, after a commit.

12. **Elk's per-atom feedback field** (P85, item B), transcribed and unit-tested, and
    negative on the only cell measured: the fixed gain rings with a growing envelope over
    2000 iterations at half Elk's gain, the secant converges at `acc = 6.5e-6` to the wrong
    state (lengths right to 8 per cent, angles 145 degrees out) because its `chi` is
    diagonal, and the penalty holds 0.576 degrees per site in 38 iterations. It needs a
    robust magnet, the same iron cell as item 1, and a per-atom 3x3 susceptibility.

13. **PAW magnons**, refused for the one-centre part of the kernel `B_xc/m`; the
    augmentation inside the transverse matrix element is done (P93, Goldstone residual
    0.071 with it against 0.984 without on fcc nickel).

14. **The magnetoelectric tensor** (P57) has only the column parallel to the field,
    clamped-ion and spin-only, and is uncalibrated against anything; Elk's
    `magnetoelt.f90` is the counterpart and is built here.

15. **The orbital magnetization of an ultrasoft or PAW dataset** (P64), which needs dual
    states in the `S` metric with `H` contracted against them, and has no reference since
    `setup.f90:130` refuses `lorbm` for ultrasoft too.

16. **The force on an atom of a spin spiral**, whose two components live on different
    plane-wave spheres so the nonlocal term needs both sets of projectors, and `dE/dq` on
    a *tabulated* augmentation table, which reads `|q|` on the host and cannot take a
    tracer (P96).

17. **The small component's magnetization on a relativistic PAW sphere** (P101):
    implemented to QE's `add_small_mag` and `compute_g`, live, and sized at 8.9e-8 Ry in
    the total and 5.4e-7 Ry in `ddd` on `i-atom-soc-paw.in`, but unconfirmed because the
    baseline scatter between the codes is two to sixty times the term, and with none of
    the five deliverables. The anisotropy is the quantity that could confirm it, the
    correction being a projector on `r-hat` that does not cancel between two directions.

18. **The symmetrised projected DOS of an `lspinorb` run**, which needs `sym_proj_so`'s
    `D^j` (`d_matrix_so`); the non-SOC half is done (P91, wedge against closed grid to
    1.1e-5 electrons per Loewdin column).

19. **Four small measurements**: whether the `ethr` schedule should follow the
    magnetization half of `dr2`, since moments converge two to four orders more loosely
    than energies (Q8, `OPEN.md` Y1); the cost of a reduced k-set against its `nosym` twin
    on a spinor run, which `PERFORMANCE.md` has no entry for (Q6); the angle between the
    Anderson coefficients under the Euclidean and the `rho_ddot` metrics, which decides
    item F (the same dump as F2's Option 0); and a reported `becsum` residual, which
    nothing watches (`OPEN.md` Y2).

20. **A spin-polarized dielectric identity that read 2.709e-07 against 1e-08 on
    2026-09-20 and passed on 2026-09-21** (`OPEN.md` Part XIII item 4). The number is 2e-8
    relative on an `eps` of 13 through two SCF runs and two response solves, so it sits
    near the tolerance; the pair needs running a few times and its spread read before the
    pass discharges the failure.

21. **Deliverables owed**: site moments on the response stack, which has no step object
    to hang them on; `colin_mag = 2` (`t_rev`) for a collinear run, which needs
    `new_ns`'s channel swap and has no committed benchmark; the `pw.x` force for P77a's
    bare-`|m|` guard; the Elk timing pairs for the spiral (pinned, one core, idle) and for
    the adaptive mixer; and the NiI2 surface's own checks in
    `HANDOFF-nii2-spiral.local.md` (cutoff at 80 against 100 Ry, a supercell point at
    `q = M`, the moment at production settings).

## Where `MAGNETISM-NEXT.md` has drifted since its last commit (2026-09-17)

- Item A still lists `set_int3_nc` as open; P98 closed it, 9.528810788 against `ph.x`'s
  9.528846009 on fully relativistic ultrasoft AlAs.
- Item A's landmine about `symmetrize_directional` treating the magnetization channels as
  scalars is gone: `driver.py` now rotates the `nspin_mag = 4` block with
  `symmetrize_spin_vector_density` and `magnetization_signs`.
- Item E's headline factor of 6.6 is 5.42 and 4.67 after the k and basis sweeps in P86.
- Section 4 says the texture notebook is owed; `notebooks/43_magnetic_textures.ipynb`
  exists. Notebooks 07 and 11 still build from the gitignored QE tree and cannot be
  re-executed, which is the part of that item that stands.
- `PLAN.md`'s outstanding index still lists a relaxed magnetocrystalline anisotropy and
  PAW for it; P87 closed both at 0.447 meV against the theorem's 0.552 on tetragonal
  cobalt, with a measured 0.011 meV floor. `average_pp` is what is left of that entry.

None of these has been edited in the files named; this list records the drift so that the
next session corrects the record rather than re-deriving it.
