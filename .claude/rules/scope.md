---
paths:
  - "defumat/**"
  - "tests/**"
  - "notebooks/**"
  - "docs/features.tex"
---

# Scope: what is implemented, and the rule that binds each piece

*Moved out of `CLAUDE.md` verbatim on 2026-10-08; it loads only when files matching the paths above are read. The text is binding exactly as it was.*

**What is in scope and implemented**, one line each. The binding rule is stated; the
validation numbers and the traps are in the named phase, and the user-facing refusals are
in `docs/features.tex`'s amber boxes.

- **Ultrasoft and PAW** (P12): the two-grid split, the augmentation charge, the overlap
  operator, self-consistent `D_ij`, PAW's one-centre terms.
- **Gradient-corrected functionals** (P13): PBE, revPBE, PBEsol, on the plane-wave grid
  and on the PAW spheres. The functional comes from the pseudopotentials' headers unless
  `input_dft` overrides it, and **an unimplemented one is refused rather than silently
  replaced by LDA**.
- **Collinear spin** (P9): `nspin = 2` gives the density, potential, `becsum`, `D_ij`,
  eigenvalues and wavefunctions a leading channel axis. The occupation scheme decides how
  many Fermi levels there are — one shared, or one per channel when `tot_magnetization`
  constrains it. `occupations = 'fixed'` implements only the second, which is `pw.x`'s own
  rule (`input.f90:784-800`) and is enforced at input here in QE's order. A fixed
  occupation **cutting a degenerate multiplet** is diagnosed by name for the residual
  solver: which member the eigensolver returns is arbitrary, so `F` is not a function of
  the density.
- **Spin-orbit coupling** (P14): `noncolin` makes a wavefunction a two-component spinor of
  length `2 npwx`, so there is *one* Hamiltonian on a space twice as large rather than two
  Hamiltonians; `lspinorb` puts a fully-relativistic dataset's `j`-resolved projectors into
  it. Forces, stress and relaxation for this regime are P46, and they take `dvan_so` (the
  **bare** `D`, for the same reason the collinear branch takes `dion`) and `qq_so`.
- **Noncollinear magnetism, magnetic fields and spin spirals** (P17-P19;
  `defumat/scf/fields.py`, `scf/locals.py`). `sym_rho`
  rotates the magnetization as an **axial** vector. Fields: the *energy* is written down
  and the potential is `jax.grad` of it, with QE's five `add_bfield.f90` expressions as a
  test; **the field's energy is not in the reported total**, by QE's and Elk's shared
  convention, and is carried separately. Spirals (`spiral_q`, Elk's `vqlss`) put the up
  component at `k + q/2` and the down at `k - q/2`, each on its own sphere — the whole of
  the generalized Bloch theorem. An **ultrasoft or PAW** dataset runs too (P95, P96): the
  transverse block of the density pairs projectors at two different k-points, so its
  augmentation charge is the resident table displaced to `Q_ij(G - q)`, and that table is
  a function of `q` exactly as `|k ± q/2 + G|^2` and `vkb` are, so `dE/dq` rebuilds it
  rather than differentiating at a frozen one. Refused for a spiral: spin-orbit coupling
  switched on, permanently, symmetry (until the spin space group is written, so `nosym`
  and the full grid), and — for `dE/dq` alone — a *tabulated* augmentation table, which
  reads `|q|` on the host and cannot take a tracer.
- **Spin-orbit coupling on a spiral, to first order** (P123, `workflows/spiral_soc.py`).
  The spiral runs on a fully-relativistic dataset at `soc_scale = 0`, the one form of the
  pair the reader admits, and the operator is `dvan_so(1) - dvan_so(0)`. It is the *same*
  matrix in every cell while the spins turn, so only its spin-diagonal part in the
  spiral's frame has an expectation value, and **the spiral Hamiltonian's own nonlocal
  contraction must not be reused for it**: that contraction turns `D` with the spins,
  which is right for the exchange field's part of `D` and wrong for the coupling. The
  energy is `n . V(q)`, linear in the axis and odd in `q`. Norm-conserving only.
- **Relaxing the spiral wavevector** (P21): `q` is a coordinate like an atomic position, so
  `dE/dq` is `jax.grad` at frozen wavefunctions and a frozen sphere (`forces/spiral.py`,
  `workflows/spiral.relax_spiral_q`), with the same BFGS
  handed the **reciprocal** cell as its metric. On a norm-conserving dataset only
  `|k ± q/2 + G|^2` and `vkb(k ± q/2)` carry `q`; on an augmented one (P96) three more terms
  do, and they are the displaced table `Q_ij(G - q)`, the orthonormality constraint, which
  is `<psi|S|psi> - 1` and therefore moves with `q` where `<psi|psi> - 1` does not, and
  PAW's one-centre energy. Two traps: the compiled gradient closes over its sphere and must
  be dropped on every `at_spiral_q`, and BFGS's initial inverse Hessian is out by two orders on a
  milli-Rydberg magnetic surface (`BFGSSettings.hessian_scale`). A magnetic field is
  refused — its energy is outside the reported total, so the state is stationary for a
  different functional.
- **Forces, stress and both relaxations** (P15, P11, P29, P46). The default is `jax.grad`
  of the total energy at *frozen* wavefunctions, with the orthonormality constraint carried
  explicitly so ultrasoft's Pulay term falls out of the same gradient; QE's hand-derived
  expressions are transcribed **beside** them as the cross-check, never instead. A
  vc-relax is **two runs** — the relaxation in a frozen basis, then one more SCF from
  scratch — and the gap between their energies is the Pulay error, reported
  (`VCRelaxResult.pulay_error`) rather than left to be noticed.
- **Berry curvature, Chern numbers, Z2, and the Berry-phase polarization** (P16, P47, P56).
  Everything is built from one primitive, `<u_mk|S|u_nk'>`, because a determinant of
  overlaps is blind to the mixing a degenerate eigensolver leaves (rule D4) *and* is an
  exact integer on any mesh. Z2 has two independent routes and running both where both
  apply is the check; where they disagree the parity one is the answer.
- **DFT+U** (P20, P62): `lda_plus_u_kind = 0` and Elk's other flavours, from the `HUBBARD`
  card, whose parameters are **in eV** and are converted at the input boundary. The energy
  is written down and `v_ns` is `jax.grad` of it; `v_hubbard` is transcribed as a test, and
  `force_hub` is *not* transcribed at all — it is `jax.grad` through moving projectors.
- **Continuing one run from another** (P23, `defumat/scf/continuation.py` and
  `System.with_spin`): `run_scf(starting_from=result)` across a
  change of spin regime. A promotion is *decompose, decide what `m` should be, recompose*;
  the magnetization is **seeded** when the source has none, because nothing in the SCF
  breaks spin symmetry on its own. `with_spin` rebuilds the k-points rather than
  relabelling them.
- **Linear response** (P24 and its letters, P45): the velocity operator from one `jvp` of
  `H(k)`, the Sternheimer solve in place of a sum over states, and the dielectric constant,
  Born charges, phonons at `Gamma`, and the strain response on top. **The perturbations are
  gradients of code that already exists** rather than expressions derived a second time —
  `dv_of_drho` is one `jvp` of `v_of_rho`, `dvqpsi_us` one `jvp` through `at_positions`,
  `int3` one `jvp` of `newd`, `PAW_dpotential` one `jvp` of `onecenter`. Still refused:
  noncollinear magnetism, DFT+U, spin spirals, a potential-only meta-GGA, **a ground
  state converged under a magnetic field or a constrained moment** (the stack rebuilds
  its potential from the *input* field, which `reducebf` and the fixed-spin-moment
  scheme both make wrong), and — for
  `nspin = 2` — a **GGA** kernel (P70 covered the LDA; `dgcxc_spin` has its own thresholds
  and gates in a different routine) and the *assemblies* above the solve, which is the
  dynamical matrix, the strain response and the two third derivatives.
- **Third derivatives** (P26, P35, P36, P43): the Raman tensor, electrostriction, the
  elastic and elasto-optic constants, from one `jvp` of the second-order energy at frozen
  first-order wavefunctions — the 2n+1 theorem, which is P15's and P25's envelope argument
  one order up. `symtensor3`/`symmatrix3` are implemented at **any** rank.
- **Meta-GGA, potential-only** (P30-P32): `tb09` and `bj06` are potentials with no energy
  functional, so they invert the rule above — nothing is differentiated, the expression
  *is* `v_x`. The consequences are enforced rather than documented: `run_scf` warns that
  its total is not the value of anything it minimised, and every consumer of
  `forces/energy.py:energy_at` refuses. `tau` comes from the states and is **mixed with the
  density and converged on**, as `pw.x` does — `kin_g` is a field of `mix_type` carried
  under `IF (xclib_dft_is('meta'))` (`scf_mod.f90:320-326`, `:443-447`, `:368-375`) and
  `rho_ddot:828` adds `tauk_ddot`. It was neither until 2026-09-20, on a justification
  read off `mix_rho.f90`, the one file that does not mention `kin_r` because it works
  through those helpers; the repair takes `si2-tb09.in` from 10 iterations at 8.41e-10 to
  **8 at 1.53e-11**. **One factor is deliberately not QE's**: `kin_g` is stored
  `(up, down)` where `of_g` is `(total, magnetization)`, and `tauk_ddot`'s halving at
  `nspin = 2` gives an unpolarized two-channel run a quarter of what the identical
  one-channel run gets, so the term is written here on `(total, magnetization)` and the two
  regimes agree — at the price that an iteration count against `pw.x` for an **LSDA** meta
  run is not like-for-like. Energy-carrying meta-GGAs (TPSS, SCAN,
  M06L) are **not** in — their potential has a `dE/dtau` piece acting on the wavefunction.
- **Van der Waals** (P27): Grimme's **D2** only, as a pair sum over the nuclei outside
  `v_of_rho`. The other four are **refused by name**, where QE's `set_vdw_corr` warns and
  silently runs with no correction at all — D3's `C6` has a coordination derivative of its
  own, and TS/MBD/XDM are functionals of the self-consistent density.
- **Optical spectra with excitons** (P37): the one place a **sum over states** earns its
  keep, because a spectrum needs `chi_0` as a matrix over `G` at every frequency where the
  Sternheimer stack gives a static operator. The Dyson equation is solved with a kernel
  from a registry; the bootstrap one is parameter-free.
- **The magnetism tail** (P48, P51-P66): site-resolved `<L>`/`<S>`/`<J>`, effective masses,
  the piezoelectric tensor, optical conductivity and the Kerr angle, Fermi-surface nesting,
  the shift current, second-harmonic generation, LO-TO splitting, magnetocrystalline
  anisotropy and the magnetic torque, structure factors, magnons, the orbital
  magnetization, STM images and vertical tunnelling transport.
- **Running a calculation too large for one job** (P67): `defumat/sizing.py` budgets a
  run's peak before it starts, the SCF checkpoints, and a dynamical matrix can be built
  one atom's column at a time across jobs.
- **The `j`-resolved projected density of states** (P69): `projwfc.x` for a spinor band,
  resolved by `j` and `m_j` instead of by `m` — the regime every heavy-element run here
  advertises and had no orbital decomposition for.
- **The screened response and Born charges of a magnetic insulator** (P70): `nspin = 2`
  above `chi_0`. The LSDA kernel is **defined** to be zero at `|zeta| >= 1` (QE's
  `dmxc_lsda` does it on both branches), so what was refused as an analysis was a
  convention — and masking the *argument the derivative is taken at* is what works, where
  clipping the density leaves the primal singular and the tangent `0 * inf`.
- **Electron-phonon coupling at one `q`** (P132, `response/elph.py`): `ph.x`'s
  `electron_phonon = 'simple'`, the linewidth and `lambda` of each mode at a list of
  broadenings, on the metallic phonon at `q != 0` (which needed the two-sphere projector's
  weights at `k` and `k + q` apart, `wk` in the contraction, and one mixer per mode). `g`
  is the bare vector the right-hand side was built from plus the converged `dV_scf`, so
  nothing new is differentiated. Refused: a metal at `q = 0` (`ef_shift`), ultrasoft/PAW,
  spin, and the zone sum (`alpha^2 F`, `T_c`).
- **Real-time propagation, high harmonics and the third harmonic** (P134 to P138,
  `defumat/realtime/`, `workflows/realtime.py`): the occupied states under
  `H(k + kappa(t))`, `kappa = A/c`, on the sphere built for `k`, at the ground state's
  potential. Four rules bind it. **The current is `jax.grad` in `kappa` of the kinetic and
  nonlocal band energy**, which already holds the diamagnetic term and `[r, V_NL]`; adding
  Elk's `-(1/c) A N` counts the first twice. **The projectors at `k + kappa` come from the
  Chebyshev table of `g_l(q^2)`** (`realtime/radial.py`), the real-time route's only: the
  transform cost five Hamiltonian applications a rebuild, and the ground state keeps it.
  **A k-chunk's arrays are arguments of the kept block, never closed over**: a program
  `compiled_function` keeps replays the constants it was traced with, so a closure over
  one chunk returns that chunk's current for the next (wrong by 1.2e-4, silently), and a
  static field that differs per chunk retraces every block. **The perturbative orders are
  nested `jvp` in the field amplitude** under `exp(eta t) cos(w t)`, checked against the
  dense hierarchy of `realtime/dense.py`. Times are Hartree a.u. at the boundary, the step
  is `H_Ry dt/2`. **The potential updated in time** (P136, `realtime/selfconsistent.py`,
  `potential = 'hartree' | 'hxc'`) is written `v_scf + U[rho(t)] - U[rho0]`, never
  `U[rho(t)]`, so the start is stationary on any mesh, and the wedge's density is completed
  with the field's little group and its translations, never the crystal's group, which
  averages the response away. **The spectrum of the orders** (P137, `realtime/hierarchy.py`,
  `get_nonlinear_spectrum`) is the steady state solved at each frequency: the computed bands
  exactly through a projector, the rest by BiCGStab, and a solve that reaches its budget is
  refused rather than used, since `1/(2 eta)` amplifies it; at first order the induced potential
  can be in it (P138, `potential = 'hxc'`, the outer fixed point a GMRES, never the static
  response's mixing, which diverges at a resonance). Refused: ultrasoft/PAW
  (`P_kappa`), `nspin = 2`, spinors, DFT+U.
- **Converging a cell with vacuum** (P129, `mixing_mode = 'ldos'`): Herbst and Levitt's
  preconditioner, screening built from the LDOS at `e_F` out of the density's own pass. Two
  rules bind it. The LDOS is a **Gaussian** whatever the run smears with, because the inner
  conjugate gradients need `D >= 0` and cold and Methfessel-Paxton derivatives go negative.
  And the inner tolerance is a statement about a **norm**: 1e-3 in the Hartree norm is measured,
  and the prototype's 1e-2 in a Kerker-weighted one lost seven iterations. `TF` and `local-TF`
  are for metal films and slow every monolayer measured, linearly in the cell length.
- **Starting from an all-electron ground state** (P72): Elk's converged density read off
  its own `STATE.OUT` and put on this run's grid, `Calculator.get_elk_seed`. The unit read
  is a run *directory*, because `STATE.OUT` carries neither the cell nor the positions and
  `tshift` moves the origin. `rfpts` is transcribed **literally** so the check can be
  Elk's own `RHO3D.OUT`, pointwise. The result is a **null**: the seed buys no iterations,
  because the cusp it carries lives inside the pseudisation radius where the pseudo
  density is built not to look like it. A spin-polarized state, a spiral, DFT+U, an element
  with a **core** and a *fixed* Elk density are refused by name.

**One calculation on many cores** (P124, `defumat/parallel.py`): `DEFUMAT_POOLS` runs
one calculation as that many processes by k-point, `pw.x -nk`, joined through
`jax.distributed` at import. A pool walks its own rows through the **streamed** store, the
raw sums over k are all-reduced before they are finished, the occupations run on the whole
set from gathered eigenvalues, and **every decision is broadcast from rank 0**, since
pools with unequal thread counts can differ in the last bit. A pool's store is a
`PoolStore`, not an `ndarray`, so a whole-set consumer fails instead of misreading it. The
SCF, forces, stress and relaxations are pool-aware; every other `Calculator.get_*`
refuses by name. On a CPU the local term and the density go through the grid a chunk of
`z` planes at a time (`sticks_local`, `sticks_density`, `DEFUMAT_PLANE_CHUNK`), which is
what keeps a core from slowing down when others share its memory.

**Gamma-only storage** (P68) is a **memory** feature rather than a speed one:
`K_POINTS gamma` stores one plane wave of each `(G, -G)` pair, which halves `npwx` and
every array a band lives in — 96 GB against 189 on a 157-atom slab. **Only the
wavefunction sphere halves and the dense G set stays whole**, which is a deliberate
departure from `pw.x`: the memory is entirely in the plane-wave-sized arrays, and halving
the dense set would put a conjugate fill inside every consumer of a real field. The
consequence to know is that `ngm` here is **not** the `ngm` `pw.x` prints —
`ngm_full = 2 ngm_QE - 1`, asserted in `tests/regression/test_basis.py`. Three things
carry the trick and nothing else does: `g_to_r_gamma` rebuilds the field from both halves;
every plane-wave sum becomes `2 Re(sum) - (the G = 0 term)`, since `G = 0` is its own
conjugate partner (`gamma_inner`, `calbec_gamma`); and `Im c(0)` must stay zero
(`force_real_g0`, where `regterg.f90:174` and `:375` impose it). **Substituted rather than
refused**, with a warning, for ultrasoft/PAW, for a run that uses symmetry, and for a
spinor or spiral run — the same physics at twice the storage, and it says so.
