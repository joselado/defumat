# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A ground-up reimplementation of Quantum ESPRESSO in Python + JAX ("pypresso"). The
Fortran QE 7.5 release is vendored here **as reference material only** — it is read to
understand algorithms and to validate numerical results, never modified or compiled into
the deliverable.

**Status: the first milestone — SCF, band structure, DOS — is met**, with ultrasoft/PAW,
LDA/GGA and collinear spin, and **forces and structural relaxation** on top of it.
P0–P9, P12–P21, P23 and P25 are done bar Wyckoff input in P6; P10 has had one pass. A silicon SCF reproduces QE's total energy to **~1e-9 Ry** term by term, its
band structure to **0.0002 eV**, and metals with every smearing to ~2.5e-8 Ry.
**Ultrasoft and PAW pseudopotentials are supported** and match QE to **≤3e-9 Ry** on 2-
and 8-atom silicon (P12). **PBE, revPBE and PBEsol** work on all three pseudopotential
kinds, matching QE to **≤6e-9 Ry** and 5e-5 eV in the bands (P13). **The density of states**
(P8) has both the smearing and the tetrahedron families, the latter also as an occupation
scheme inside the SCF, matching QE's three aluminium benchmarks to 2.5e-8 Ry. **Collinear
spin** (P9) matches eight LSDA benchmarks — nickel's total energy to **1.2e-9 Ry** and its
magnetic moment to the two decimals QE prints (0.7280 against 0.73).
**Spin-orbit coupling** (P14) is in as well: `noncolin`/`lspinorb` give two-component
spinor wavefunctions and the `j`-resolved projectors of a fully-relativistic dataset, on
norm-conserving, ultrasoft and PAW pseudopotentials, matching QE's three platinum
benchmarks to **≤1.3e-8 Ry**.
**Forces and relaxation** (P15) are in: the force is `jax.grad` of the total energy at
frozen wavefunctions — Hellmann-Feynman, Pulay and the augmentation charge's own
derivative all falling out of one gradient — with QE's six hand-derived terms implemented
beside it as a cross-check, and a BFGS relaxation on top. They match QE to **≤2e-5 Ry/bohr**
on five references and reproduce its relaxed geometries to **1e-6 bohr**.
**Berry curvature, Chern numbers and Z2 invariants** (P16) are in too: the Chern number is
an exact integer on a 6x6 mesh, and the Z2 has both the Wilson-loop and the Fu-Kane parity
route, agreeing on every model case with a known answer.
**Noncollinear magnetism, magnetic fields and spin spirals** (P17-P19) are in: a magnetic
`nspin_mag = 4` run reproduces QE's bcc-iron benchmarks to **2.8e-9 Ry** with LDA and with
PBE, the moment and the magnetic symmetry group matching what QE prints; external and
per-atom fields and all three of QE's constrained-moment schemes match their (regenerated)
benchmarks to **≤2e-7 Ry**; and **spin spirals** by the generalized Bloch theorem
reproduce the collinear antiferromagnet of a doubled cell and a 90-degree noncollinear
supercell to **1e-12 Ry**, which is the validation they have — `pw.x` has no spin spiral.
**DFT+U** (P20) is in: the simplified rotationally-invariant functional with `U`, `J0`,
`alpha` and `beta`, on `atomic`, `ortho-atomic` and `norm-atomic` projectors, matching QE
to **≤6.7e-9 Ry** on seven cases (antiferromagnetic FeO and fcc nickel) with the Hubbard
term itself to 4.6e-7 Ry, and its forces — QE's `force_hub`, obtained by differentiating
through the projectors rather than transcribed — to 4.8e-6 Ry/bohr.
**The spin spiral's wavevector is relaxed** (P21) the way the atoms are: `dE/dq` is
`jax.grad` of the energy at frozen wavefunctions, and a BFGS on the reciprocal metric takes
a hydrogen chain from `q = 0.30` to its antiferromagnetic ground state at `0.50003` in six
SCF runs — validated by identities and finite differences, since `pw.x` has no spiral.
**One run continues another across a change of spin regime** (P23):
`run_scf(starting_from=result)` promotes a converged state into the target's variables —
a non-magnetic density into a collinear run, a collinear one into a noncollinear run, and
spin-orbit coupling switched on — reaching the *same* self-consistent solution as a fresh
run (≤4e-8 Ry on six cases) in 1 iteration instead of 25 where the magnetization only has
to be rotated. **The magnetization is seeded when the source has none**, because nothing in
the SCF breaks spin symmetry on its own and an unseeded promotion converges straight back to
the unpolarized solution.
**Linear response by autodiff** (P24) is in: the velocity operator from one `jvp` of `H(k)`
(rule D2 cashed in), the Sternheimer equation in place of a sum over states, and silicon's
**dielectric constant** against `ph.x` on norm-conserving, ultrasoft *and* PAW datasets
— agreeing to **≤1.2e-4** — with the
screening kernel, the field's commutator and the bare phonon term all gradients of code that
was already there.
**The Born effective charges are ultrasoft now too** (P24b), because `Z* = dF/dE` is a mixed
second derivative and is computed as one: a single `jvp` of the force along the field's
response, per field direction, which turns four of the five stages `zstar_eu_us.f90` adds
into terms of the same derivative. Against `ph.x`: **-0.0757150** norm-conserving silicon
(every printed digit), **-0.0794417** ultrasoft silicon (8.3e-6) and **+0.0415594** ultrasoft
carbon, whose sign is the *opposite* one. PAW is refused by name at 1.3e-3, with the missing
term identified (`int3_paw` against `becsumort`) rather than fitted.
**Metals are in the response** (P24c): `orthogonalize`'s smearing branch, `setup_alpha_pv`'s
metal value, `localdos` and `ef_shift`, with `chi_0` on fcc aluminium matching a finite
difference of the density to **2.5e-7** and the Fermi-level correction restoring charge
neutrality to 1e-15.
**The dynamical matrix of a metal is in** (P28), and it cost one weight rather than a
routine: a metal's `dpsi` already carries its occupation, so contracting it against an
energy functional weighted by `wg` counts that occupation twice, and QE's own layout says
so — `dynmat_us.f90` reads `wg` for the frozen Hessian and `drhodvnl.f90` reads `2 wk` for
the electronic term. Splitting P25's single `jvp` along those lines puts two-atom
aluminium's modes at **146.7093**, **146.7132** and **311.0335** cm⁻¹ against `ph.x`'s
146.710511 / 146.714378 and 311.035401 — **0.0019 cm⁻¹**, an order tighter than silicon's
0.05, the folded pair's real 0.0039 splitting reproduced rather than flattened — where
the unsplit assembly gave 197.96 and 309.26 and put the acoustic modes at 155.7 against
1.9. The acoustic sum rule is the diagnostic that said so and now holds to 1.06e-5
Ry/bohr². **The `df_n` term the refusal predicted is not needed**: it is already inside
`dpsi`, being the `(f_i - f_j)/(eps_i - eps_j)` structure of the smeared projector, which
vanishes identically for an insulator.
**A supercell is a regime of its own** (P28a), which running P28 on the four-atom
conventional cell of fcc aluminium established by finding two bugs no other committed
cell could see. Its atoms sit at exact fractions, so the structure factor vanishes
*exactly* (92 of 3287 G-vectors, against a 4e-16 floor on primitive cells) and the point
group's atom permutations acquire **3-cycles** where every other cell here has only
involutions. The first made `abs(rho)**2` in the reciprocal Ewald sum a `0/0` derivative —
the `abs` trap in a **fourth** place, and the first one *forced by symmetry* rather than
by an accidental node — and the second made `symdvscf` average over the atom each
operation moves instead of the one it moves onto. Both leave the energy and the forces
right and damage only the second derivative. **The two identities P25 rests on are blind
to the first**, because the acoustic sum rule and the rigid-translation test are both sums
over *atoms* and the error was a transfer between them; the first check that is not an
atom-sum is the per-mode response density against a finite difference. Four-atom aluminium
now matches `ph.x` to **0.020-0.034 cm⁻¹** on the first metal phonon computed on a
symmetry-reduced wedge.
`PLAN.md` §3 tracks the phases and records the transcription traps each one uncovered —
read it before writing code. P4 is complete: a block Davidson eigensolver behind a name
registry, seeded from the pseudo-atomic orbitals as QE seeds it, and the *only* solver the
package offers — forming `H` costs `O(npw^2)` memory, so a dense solve is a test fixture
(`tests/exact_reference.py`), never a `diagonalization` a run can select. P6 is complete too: automatic k-grids are reduced to the
irreducible wedge. P10's first pass puts pypresso within **2–4x of serial Quantum ESPRESSO
per SCF iteration** on the same machine, ultrasoft and PAW included — see
`PERFORMANCE.md`. **The projected density of states** (`projwfc.x`) is in as well, completing P8:
`<phi|S|psi>` on Löwdin-orthogonalised pseudo-atomic orbitals, resolved by atom, `l` and
`m`, feeding the *same* DOS registry as a per-band weight, with Löwdin charges and the
spilling parameter — matching a `projwfc.x` built for the purpose on seven cases, to the
resolution of everything it prints (6.9e-4 on a projection, 4.7e-5 on a charge).
**The stress tensor** (P11) is in too: `sigma = -(1/Omega) dE/d(epsilon)` from one `jax.grad`
of the energy at frozen wavefunctions, matching QE to **≤2.7e-7 Ry/bohr³** on thirteen cases
from norm-conserving LDA up through ultrasoft, PAW, PBE, `nspin = 2` and DFT+U.
**Phonons at `Gamma`** (P25) are in: the force constants are `jax.grad` of the total
energy differentiated *once more*, along a tangent that carries the positions, the states
and the density together — so QE's `dynmat0`/`d2ionq` (the frozen second derivative) and
`drhodv` (the electronic response) are two halves of one `jvp` of the gradient that
already gives the force. Silicon's optical mode is **510.102 cm⁻¹** against the vendored
`ph.x`'s 510.152, checked three further ways that share nothing with the assembly: a rigid
translation reproduces `-drho/dx` to 6.5e-5, finite-differenced forces reproduce whole
columns of the matrix to 2.1e-5 Ry/bohr², and the reduced wedge agrees with the whole
closed grid to 2.7e-14. **Norm-conserving only, and refused by name otherwise**, because
with `S` moving the orthonormality multipliers contribute a term of their own.
**Electrostriction** (P26) is in, and it is the first **third** derivative of the energy
here: `d(chi)/d(strain)` — the elasto-optic tensor, and through the thermodynamic identity
of Tanner, Bousquet and Janolin the four electrostriction tensors `m`, `q`, `M` and `Q` —
from **one `jvp` of the second-order energy at frozen first-order wavefunctions**, which is
the 2n+1 theorem and is the envelope argument P15 and P25 already make, one order up. The
strain perturbation it stands on (`dpsi/dx`, `drho/dx`) is Abinit's metric-tensor
formulation obtained for nothing, because `at_strain` was already written in reduced
coordinates; the **elastic constants** come with it, as one more `jvp` of the stress, and
reproduce a five-point second difference of the energy to five significant figures
(converged silicon: `C_11` = 198.5 GPa against a measured 165.7, `C_12` = 68.9 against 63.9).
The three independent components of `d(eps)/dx` match a central difference of `epsilon`
over re-converged strained cells to **2e-4**, the difference's own floor, and the whole
rank-4 tensor is cubic to 3e-14 with nothing imposing it. Norm-conserving, `nspin = 1`, insulators
and **clamped-ion**, on an **unshifted** k-grid — a symmetry-reduced wedge is refused by
name, because the object being differentiated carries a field label and a strain label at
once and the rank-3 average that would complete the sum is not written.

**Grimme's D2 van der Waals correction** (P27) is in: `vdw_corr = 'grimme-d2'`, written
as the Ewald sum's twin — a pair sum over the nuclei whose neighbour list is fixed once, so
the force and the stress are `jax.grad` of it in the two coordinates and QE's `force_london`
and `stres_london` are transcribed beside them as the check (they agree to 1e-14). Bilayer
graphene matches `pw.x` to **3.1e-9 Ry** in the total energy and 3.7e-7 Ry/bohr in the force,
and **binds at 6.10 bohr (3.23 Å) where PBE alone has no minimum at all**. The correction
never enters `v_of_rho`, and the test for that is an *equality*: the same cell with and
without it gives a bit-for-bit identical density, and `d(chi)/d(strain)` is unchanged to
0.0 while the elastic constants move by exactly the pair sum's own second derivative.
**P26's third derivative runs on the bilayer itself** on a k-grid that misses `K` — graphene
is a semimetal and the Sternheimer response here is the insulator one — reproducing a
five-point second difference of the energy to 5.8e-5 and a central difference of `epsilon`
to 2.2e-4. Getting there found a trap that is P26's rather than P27's: at QE's
`alpha_mix = 0.7` the strain response of a **slab** diverges, and a diverged first-order
solution was being consumed in silence, giving a `C_ijkl` that was not even symmetric under
`C_ijkl = C_klij`. It is refused now (`require_converged_responses`).

**Outstanding:** Wyckoff input, `vc-relax`, the dynamical matrix of an
ultrasoft dataset, PAW Born charges, phonons at `q != 0` (the perturbed states live
at `k + q`, so it needs the two-sphere machinery P19 built for the spin spirals, plus
`q2r`/`matdyn` for a dispersion), and the rest of P10 (k-axis sharding and GPU).

## Layout

- `quantum_espresso/qe-7.5-ReleasePack/qe-7.5/` — QE 7.5 Fortran sources. **Read-only.**
- `quantum_espresso/Doc-QE-7.5/Doc-7.5/` — input-file documentation (`INPUT_PW.txt` is the
  authoritative spec for the `pw.x` input namelists/cards) and theory PDFs.
- `pypresso/` — the Python package. `tests/` alongside it; `tests/data/pseudo/` holds the
  committed UPF files (QE's test-suite downloads rather than ships them).
- Git repository, with `quantum_espresso/` gitignored — 285 MB of vendored reference does
  not belong in history. Tests that need it skip cleanly when it is absent.
- The two configured working directories are one directory: one path is a symlink to the
  other, so the same file can arrive under either prefix. Do not treat them as separate
  copies.

## Scope

First milestone, in this order: **SCF → band structure → DOS**, for `pw.x` with
norm-conserving pseudopotentials, LDA/PBE and k-point grids — **now met**, and extended
since with ultrasoft/PAW, the PBE family and collinear spin. This is a large project that
will keep growing, so structure matters more than speed of delivery — see `PLAN.md` for
the architecture, the phase breakdown, and the validation strategy. Read it before writing
code.

**Gamma-only is a gap, not a feature.** `K_POINTS gamma` selects the half-sphere storage
of the gamma-point trick, and that storage is generated but not consumed anywhere:
`h_psi` would need `vloc_psi_gamma`'s packing, the eigensolver `regterg`'s real overlaps,
and `addusdens`/`newd` their `fact = 2`. Such a run is silently substituted by an explicit
k = 0 with the full sphere, which is the same physics at twice the storage, and it says so.

**Ultrasoft and PAW are in scope and implemented** (P12): the two-grid split, the
augmentation charge, the overlap operator, self-consistent `D_ij`, and PAW's one-centre
terms. **Gradient-corrected functionals are too** (P13) — PBE, revPBE and PBEsol, on the
plane-wave grid and on the PAW spheres — so the PBE datasets that most published
ultrasoft/PAW work uses run here. The functional comes from the pseudopotentials' headers
unless `input_dft` overrides it, and an unimplemented one is refused rather than silently
replaced by LDA.

**Collinear spin is in scope and implemented** (P9): `nspin = 2` gives the density, the
potential, `becsum`, `D_ij`, the eigenvalues and the wavefunctions a leading channel axis,
and one SCF iteration diagonalises a different Hamiltonian per channel. Whichever
occupation scheme is in use decides how many Fermi levels there are — one shared between
the channels, or one each when `tot_magnetization` constrains the magnetisation — and both
the smearing and the tetrahedron families implement both.

**Spin-orbit coupling is in scope and implemented** (P14): `noncolin = .true.` makes a
wavefunction a two-component spinor of length `2 npwx`, so there is *one* Hamiltonian on a
space twice as large rather than two Hamiltonians, and `lspinorb = .true.` puts the
`j`-resolved projectors of a fully-relativistic dataset into it. **Keep QE's three spin
numbers apart**, because collapsing them is the mistake that makes a spin-orbit run
allocate a magnetization it does not have: `nspin` says which regime (1, 2 or 4), `npol`
how many components a *wavefunction* has, and `nspin_mag` how many a *density* has —
which is **one** for a nonmagnetic spin-orbit run, exactly as for an unpolarized one. That
is why such a run costs about what a doubled unpolarized one costs: the density, the
potential, the exchange-correlation functional and the symmetrisation are untouched, and
all the new physics is in the spinors and in `D_ij` becoming a complex 2x2 matrix in spin
space. Non-collinear *magnetism* (`nspin_mag = 4`) is complete and validated as of P17:
`sym_rho` rotates the magnetization as an axial vector, the magnetic symmetry group keeps
the operations that need time reversal, and `gradcorr` runs in the local spin frame. The one
piece still refused is `PAW_gcxc_potential` with a magnetization — PAW plus a GGA plus
`nspin_mag = 4` — because the radial local-frame rotation (`compute_rho_spin_lm`) is a
second implementation rather than a call into the plane-wave one.

**Magnetic fields and constrained moments are in scope and implemented** (P18):
`pypresso/scf/fields.py` and `scf/locals.py`. A uniform field over the cell (QE's
`B_field`, Elk's `bfieldc`), a field inside one atom's sphere (Elk's `bfcmt`, through a
`LOCAL_MAGNETIC_FIELDS` card that `pw.x` has no counterpart for), Elk's `reducebf`, and all
four of QE's `constrained_magnetization` schemes. **The energy is written down and the
potential is `jax.grad` of it** — QE's five hand-derived expressions in `add_bfield.f90`
are then a *test*, not a second implementation. **The field's energy is not in the total
energy**: `add_bfield` is called from inside `v_of_rho`, so `deband` removes it again and
`etcon` is printed and never added; Elk excludes its external field's energy by the same
convention, and both numbers are carried separately.

**Spin spirals are in scope and implemented** (P19): `spiral_q` (Elk's `vqlss`, in lattice
coordinates) makes the up component of the spinor live at `k + q/2` and the down at
`k - q/2`, each on its own plane-wave sphere, which is the generalized Bloch theorem and the
whole of the implementation. In the rotated frame the density and the potential are lattice
periodic, so the SCF, the functional and the mixer are untouched. **Three things are
refused**: spin-orbit coupling permanently (it breaks the theorem, and Elk refuses it too);
symmetry, until the spin space group is written, so a spiral needs `nosym` and the full
k-grid; and ultrasoft/PAW, until the augmentation charge *between the two components* —
`q_ij(q)`, which `topology/augmentation.py` already builds, not `qq` — is threaded through.

**Relaxing the spiral wavevector is in scope and implemented** (P21): `q` is a coordinate
of the calculation the way the atomic positions are, so it gets the same treatment —
`forces/spiral.py` writes the total energy as a function of `q` at *frozen* wavefunctions
and `dE/dq` is `jax.grad` of it, and `workflows/spiral.relax_spiral_q` walks it downhill
with the same transcribed BFGS, handed the **reciprocal** cell so that its metric is
`b_i . b_j`. The frozen quantity is the *periodic part* of the spinor, which is what the
stored coefficients are, so freezing them lets the spiral turn — exactly the variational
parameter the SCF minimised over. **Only two terms of the energy depend on `q`**:
`|k +- q/2 + G|^2` and `vkb(k +- q/2)`. At frozen coefficients the rotated-frame density is
lattice periodic on an FFT box that does not move, so the Hartree, exchange-correlation,
local and Ewald terms are `q`-independent and the gradient never differentiates through an
FFT; they are written down anyway, because the identity against the SCF total energy is the
only check on them. **The plane-wave sphere is frozen while differentiating and rebuilt to
move** — sphere membership is piecewise constant in `q`, so that is exact between the
wavevectors where a plane wave crosses the cutoff, and the jump at those is the Pulay error
of a finite basis (measured against a sphere-rebuilding finite difference: 8.3e-4 Ry per
unit `q` at `ecutwfc = 25`, 5.8e-4 at 40 and 8.3e-6 at 60 — erratic rather than smoothly
convergent, because it counts the crossings inside one window rather than truncating a
series). Two
traps: the compiled gradient closes over its sphere and is dropped on every `at_spiral_q`;
and BFGS's initial inverse Hessian, which for atoms is right because a chemical bond is
1 Ry/bohr^2, is out by two orders of magnitude on a milli-Rydberg magnetic surface, so
`BFGSSettings.hessian_scale` sets the first step to the trust radius. A magnetic field is
refused — its energy is outside the reported total (P18), so the state is stationary for a
different functional than the one being differentiated.

**Berry curvature, Chern numbers and Z2 invariants are in scope and implemented** (P16):
`pypresso/topology/` and `workflows/topology.py`. Everything is built from one primitive,
`<u_mk|S|u_nk'>` — the overlap of the occupied manifolds at neighbouring k-points — because
a determinant of overlaps is blind to the unitary mixing a degenerate eigensolver leaves
(D4) *and* because the Fukui-Hatsugai-Suzuki lattice sum is an exact integer on any mesh
where a Riemann sum of a pointwise curvature is not. The velocity-operator route from
`jacfwd` of `H(k)` (D2) is registered as `kubo` for a smooth `Omega(k)` and is available
only where `H(k)` is a differentiable function; the plane-wave version needs `d(vkb)/dk`
and belongs to P11. Z2 has two independent methods — Wannier-charge-centre flow, which
needs only time-reversal symmetry, and the Fu-Kane parity products, which need an inversion
centre and cost eight k-points — and running both wherever they both apply is the check.
**The parity route has no mesh and the Wilson route does**, so where they disagree the
parity one is the answer; `WannierFlow.gap_step` is the Wilson result's own diagnostic
(how far the largest-gap reference line moves in one pumping step) and it is the number to
read before believing the integer.
**Two things bite in a plane-wave code and both are silent:** neighbouring k-points do not
share a G-sphere, so coefficients are aligned by Miller index; and the wrap at the zone
edge is a *shift* of that index (`u_{k+b}(G) = u_k(G+b)`), without which the Chern number
is smooth and non-integer. Ultrasoft `S` between two k-points is `q_ij(b)`, not `qq`, and a
relativistic dataset needs it through `transform_qq_so`.

**Forces and ionic relaxation are in scope and implemented** (P15): the force comes from
differentiating the total energy with respect to the atomic positions at *frozen*
wavefunctions, occupations and eigenvalues, with the orthonormality constraint carried
explicitly so that ultrasoft's Pulay term is part of the same gradient. QE's `force_lc`,
`force_cc`, `force_ew`, `force_us`, `addusforce` and `force_corr` are transcribed as well,
behind the same registry, because the two implementations share no machinery and checking
one against the other is what found the augmentation force's sign and the gradient
correction missing from `force_cc`. `calculation = 'relax'` runs QE's BFGS with its trust
radius and Wolfe line search. **Variable-cell relaxation is not in**: the cell gradient is
the stress, which is P11's, and a moving cell would also invalidate the rule that the FFT
grid and the symmetry group are fixed once for the whole run.

**DFT+U is in scope and implemented** (P20): `pypresso/hubbard/`. QE's
`lda_plus_u_kind = 0` — Dudarev's simplified rotationally-invariant functional with the
`J0`/`beta` extension — read from the `HUBBARD` card, whose parameters are **in eV** and
are converted to Ry at the input boundary. **The energy is written down and `v_ns` is
`jax.grad` of it**; QE's `v_hubbard` is transcribed as a *test*. The correction enters the
Hamiltonian as another separable term (`vhpsi.f90`), `ns` joins the mixed state beside
`becsum`, and the force is `jax.grad` through projectors that move with the atoms — which
is the whole of `force_hub` without transcribing it. **Three traps, all silent**:
`Hubbard_projectors = 'atomic'` still applies `S`, so `wfcU = S phi`; the atomic orbitals
are renormalised at read time in the *generalised* metric (`upf_check_atwfc_norm`), which
`ortho-atomic` projectors feel through the `4s` and `atomic` ones do not; and the
`nspin = 1` factor of two is on the **energy**, never on the potential. Refused by name:
the full (Liechtenstein) formulation, the intersite `V`, background channels, the
orbital-resolved variant, the `wf`/`pseudo` projector sets, and noncollinear `ns_nc`.

**Continuing one run from another is in scope and implemented** (P23):
`pypresso/scf/continuation.py` and `System.with_spin`. The three spin regimes are three ways
of writing the same pair `(n(r), m(r))`, so a promotion is *decompose, decide what `m` should
be, recompose* — and a demotion is the same function read the other way. The whole mixed
state crosses together (`rho`, `becsum`, `ns`) plus the wavefunctions, which cross as a
**span for the first Rayleigh-Ritz** rather than as wavefunctions, so they need not be
orthonormal in the target's overlap operator or number `nbnd`. **The magnetization is seeded
from the target's `starting_magnetization` when the source has none**: nothing in the SCF
breaks spin symmetry on its own, so a promotion that carried only the charge would converge
back to the unpolarized solution and report success. `with_spin` **rebuilds the k-points**
rather than relabelling them — the `degspin` factor and the magnetic symmetry group both
change with `nspin`. Refused rather than approximated: a target whose species point their
moments along different axes (a collinear source has one scalar field;
`magnetization="seed"` is the way out), a Hubbard `U` crossing into `nspin = 4`, splitting a
spinor back into two collinear channels, and a `becsum` from a different pseudopotential,
which is dropped with a warning instead of being reshaped.

**Linear response is in scope and implemented** (P24): `pypresso/response/`. Three layers,
and the point of all three is that the *perturbations* are gradients of code that already
exists rather than expressions derived a second time. The **velocity operator** is one
`jax.jvp` of `H(k)` at a frozen sphere, because `dH/dk_a = i[H, r_a]` in the periodic gauge
— which is what rule D2 asked P2-P4 to preserve, and why no radial form factor is ever a
table lookup. The **Sternheimer equation** `(H - eps_n S + alpha Q)|dpsi_n> = -P_c^+ dV|psi_n>`
replaces the sum over states with a projected CG solve per occupied band: no empty states,
and no division by `eps_n - eps_m`, which rule D4 forbids. And the **dielectric constant**
and **Born effective charges** follow, matching the **vendored** `ph.x` to 4.3e-5 on
`epsilon_infinity` (13.806646 against 13.806689) and to every digit it prints on `Z*`
(-0.075715 against -0.07571). The comparison is against a *regenerated* reference, not
`ph_base`'s committed one, which dates from release 6.0 and has drifted to 13.806375 —
the same staleness `tests/conftest.py` already documents for `pw.x`.

**Ultrasoft and PAW are in scope here too, and almost none of what they add is
transcribed.** `incdrhoscf` + `addusdbec` + `lr_addusddens` is one `jvp` of the density
builder with respect to the *states*; `newdq`'s `int3` is one `jvp` of `newd` with respect
to the potential; `PAW_dpotential` is one `jvp` of `onecenter` with respect to `becsum`.
The dielectric constant matches the vendored `ph.x` to **≤1.2e-4** on norm-conserving,
ultrasoft and PAW silicon and on ultrasoft carbon. Three things did have to be written and
each is a trap: `|psi|^2` must be `Re(conj(psi) psi)` and not `abs(psi)**2`, whose
derivative is `0/0` at a node; the projector derivative in `adddvepsi_us` is the one about
the atom's own centre, since `gen_us_dj`/`gen_us_dy` leave the structure factor alone and
the `tau` term is worth 2%; and `dbecsum` on a wedge is a **polar vector**
(`PAW_dusymmetrize`), worth 1.6e-2 on PAW. **Born effective charges stay norm-conserving**
— `zstar_eu_us.f90` is five further stages — and are refused by name for the other two,
because without them the expression is wrong in sign as well as size.

**The trap is that a response is direction-dependent and must be symmetrised as a polar
vector**, and the escape from that does not work where it looks like it should: running the
*whole* k-grid instead of a wedge is only sound if the grid is closed under the point group,
and a **shifted** Monkhorst-Pack grid is not — 2304 of the 3072 rotation images of a shifted
4x4x4 grid on fcc silicon land off it, giving a 2% asymmetric density and a dielectric tensor
with off-diagonal entries cubic symmetry forbids. That combination is refused by name; an
unshifted grid is closed exactly and is the independent check on the symmetrisation.
**Refused** rather than approximated: ultrasoft and PAW (`dbecsum`, the augmentation charge's
own response, `int3`), metals (`orthogonalize`'s smearing branch and `ef_shift`),
noncollinear magnetism, DFT+U (`adddvhubscf`) and spin spirals.

**Van der Waals corrections are in scope, and one of the five is implemented** (P27):
`pypresso/vdw/`, behind a name registry that `vdw_corr` selects from. Grimme's **D2** is a
pair potential over the nuclei and nothing else, so it is `pypresso/scf/ewald.py` again with
a different radial function — the energy is written down and the force and the stress are
`jax.grad` of it. The other four are **refused by name**, where `set_vdw_corr` warns and
silently runs with no correction at all: **D3** because its `C6` depends on each atom's
coordination number and so has a derivative of its own, and **Tkatchenko-Scheffler**, **MBD**
and **XDM** because their coefficients are functionals of the self-consistent density, which
puts them inside `v_of_rho` where D2 is outside it.

Out of scope until the above works: EXX, phonons
(`PHonon/`), Car-Parrinello (`CPV/`), and everything in `EPW/`, `TDDFPT/`, `HP/`, `GWW/`.
The code should nonetheless be shaped so these are additions, not rewrites.

## Why JAX (this drives the design)

Two reasons, both of which constrain how code is written:

1. **Autodifferentiation.** Response and higher-order properties — polarization, dielectric
   response, second harmonic generation, forces, stress — should come from differentiating
   the code rather than from separately hand-derived expressions. This is the main reason
   for JAX, not a bonus. Consequences are in `PLAN.md` §6 and they are binding: the compute
   path must be differentiable end to end, including the XC functional and the k-dependence
   of the Hamiltonian.
2. **GPU.** The same JAX code must run on GPU unchanged (development is CPU-only here — no
   GPU on this machine).

Performance matters. It does not have to be optimal in the first version, but no design
choice should make good performance unreachable without a rewrite.

## The README's feature table

`README.md` carries a table of every implemented feature, with the input variable or entry
point that reaches it and a column saying whether Quantum ESPRESSO has it at all — `pw.x`,
or `new` for the ones with no counterpart there (the spin spirals, the spiral relaxation,
the topological invariants, Elk's fields). It is the only place that distinction is written
down for a reader who is not going to read `PLAN.md`, and it is what tells someone
evaluating the code whether it is a reimplementation or an extension.

**The rows are physics, not knobs.** A row is a thing someone would want to compute — total
energies, bands, densities of states, forces, relaxation, magnetism, DFT+U, the invariants —
not an implementation setting underneath one. Smearing is the example: it belongs inside the
row about metals, not in a row of its own. A new feature adds *one* row, named by the
quantity it produces; its variants, schemes and internal terms go in `PLAN.md`.

**Every new feature adds a row, and every removed or renamed one edits its row.** Same
standing requirement as the notebooks: a phase is not finished until its row exists. Two
things make the table go stale silently and both have already happened once, so check them:

- **The provenance column is a claim about the Fortran**, not a guess. Before writing
  `pw.x`, find the routine; before writing `new`, grep the vendored tree for it. "QE
  probably has this" is how a wrong row gets in.
- **The entry-point column is a claim about *this* code.** Check the variable is actually
  read (`grep` it in `io/pwin.py` and `system/builder.py`) and the function actually
  exported. A row naming `tprnfor` when nothing parses it, or `UPF v1` when the reader
  refuses anything but v2, is worse than no row — both were in the first draft of the table.

## Tutorial notebooks

`notebooks/` holds worked examples on concrete systems — the readable counterpart to the
test suite. **Every new feature adds a notebook or extends an existing one; a phase is not
finished until its notebook exists.** Demonstrate on the two-atom silicon cell from
`test-suite/pw_scf/scf.in` wherever possible, compare against the committed QE benchmark
whenever the reference contains the quantity, and commit the notebook executed so it reads
without being run.

**A notebook is five minutes long, and it has a figure.** Header saying what this computes
and the headline number against QE; the shortest code that runs it; **one plot that shows
the physics** — a band structure wherever the feature shows in bands; one comparison table;
at most one "how it works" cell for the single best idea; a footer pointing at the `PLAN.md`
phase entry and the test file. About eight code cells. The derivations, the trap catalogue
and the per-case validation tables belong in `PLAN.md` and in the tests, not here, and an
expensive sweep is measured once offline and *quoted* rather than run. Each notebook also has a `.md` export committed beside it — raw `.ipynb` is unreadable in a
plain editor or a diff — regenerated together with the notebook by `tools/export_notebooks.sh`.
`notebooks/README.md` holds the index and the full conventions.

## Performance

**The measurement is single-core pypresso against single-core Quantum ESPRESSO on the
same machine and the same input.** That comparison is the starting point of any
performance discussion, not a summary of one:

```bash
python3 tools/compare_qe.py benchmarks/si-1k.in --repeats 5
```

It needs `pw.x` built serially once (`./configure --disable-parallel --disable-openmp &&
make -j pw` inside the vendored tree; the binary is gitignored along with the rest of it).
The tool pins both codes to one core — JAX otherwise uses every core and the comparison
flatters it by the core count — and reads QE's own timing report, so the numbers on the
QE side are QE's, not a stopwatch around it.

The benchmark inputs live in `benchmarks/`, and are **single k-point** on purpose: both
codes parallelise over k, so a multi-k comparison measures batching rather than the cost
of the physics. `si-1k.in` is the test suite's silicon at `ecutwfc = 12`; `si-1k-ecut40.in`
is the same cell at a production cutoff, where scaling starts to show.

`performance/run_performance.py` runs that comparison over a whole set of inputs and
typesets it — a PDF with the per-case ratios, what compilation costs, and each case's peak
memory, regenerable by hand at any time (`performance/README.md`). Each case is a
subprocess with a 2-minute budget, so a slow one is reported rather than waited on.

`PERFORMANCE.md` is the running log: the comparison, where the time goes, what each change
was worth, and the backlog. **Add a measurement to it whenever a feature lands or a hot
spot moves** — including the QE ratio, not only an internal timing. `tools/benchmark.py
<input>` gives the component breakdown when a ratio needs explaining.

## Non-negotiable conventions

- Pure Python. JAX for anything numerical that runs inside the SCF/diagonalization loop;
  Numba only for host-side setup loops (G-vector enumeration, symmetry search, radial
  tables), never inside a jitted path.
- JAX code must stay GPU-ready and differentiable: static shapes, no host syncs in the
  inner loop, no Python branching on traced values, no in-place tricks that break `grad`.
  Pad plane-wave arrays to `npwx` with a mask instead of using per-k shapes.
- **Object-oriented is encouraged, mutable global state is not.** This is deliberately not a
  literal transcription of the Fortran: use classes with bound methods where they make the
  code read better (`ham.apply(psi, k)`, `density.symmetrize()`, `pseudo.projectors(k)`).
  The constraint is that any class crossing a `jit`/`grad` boundary is frozen and
  pytree-registered — methods are fine, mutation and module-level globals are not. (QE's
  shared-module globals are exactly what not to copy.) **`equinox.Module` is the base
  class** for all such state objects; static config uses `eqx.field(static=True)`.
- **Never hardcode a dtype.** Single precision has to stay viable for GPU, so real and
  complex dtypes come from the policy object in `config.py`, never from literals like
  `jnp.complex128` or `1.0j`. x64 is still enabled and all QE validation is float64;
  float32 is a performance mode, never one a correctness claim is made in.
- Pluggable pieces — XC functionals, mixers, eigensolvers, smearing, DOS schemes — go
  behind a name registry, so adding one is a new file plus a registration, not an edit to a
  growing branch in the driver.
- Parallelism in JAX is not OpenMP: XLA already threads each op on CPU, and explicit
  parallelism comes from `vmap` over the k-point axis plus `jax.sharding` over that same
  axis (CPU cores as devices now, GPUs later). Keep k the leading independent axis of every
  wavefunction-shaped array so this stays available. Numba `prange` is the right tool for
  the host-side setup loops only.
- Rydberg atomic units internally (Ry, bohr), matching QE; convert only in `io/`.
- **`nspin`, `npol` and `nspin_mag` are three different numbers.** `nspin` says which
  regime is in force, `npol` is the number of spinor components of a *wavefunction*, and
  `nspin_mag` the number of components of a *density*. They coincide for 1 and 2 and come
  apart at 4, where `npol = 2` and `nspin_mag` is 4 only if the run actually carries a
  magnetization. All three are static; `System` exposes them as properties so no call site
  recomputes the rule.
- **The spin channel is the leading axis, and it is squeezed on the way out.** Densities,
  potentials and `becsum` are `(nspin, ...)` internally with no special case for one
  channel; the result objects (`SCFResult`, `NSCFResult`, `DensityOfStates`,
  `BandStructure`) drop that axis when `nspin = 1` and expose a `*_by_spin` property that
  always has it. `k` stays the leading *independent* axis inside each channel, which is
  what the batching and the eventual sharding rest on. `nspin` is static
  (`eqx.field(static=True)`) because it is an array rank, not a value.

## Where each subsystem lives in the reference source

Paths relative to `quantum_espresso/qe-7.5-ReleasePack/qe-7.5/`.

| Subsystem | Reference | Notes for the port |
|---|---|---|
| Top-level driver | `PW/src/run_pwscf.f90` → `init_run.f90` → `electrons.f90` | `electrons_scf` is the SCF loop; ignore the EXX/RISM/OSCDFT branches. Its `ethr` schedule and `dr2` convergence test are transcribed — `conv_thr` means the same thing here as in a `pw.x` input |
| SCF iteration body | `c_bands.f90`, `sum_band.f90`, `v_of_rho.f90`, `mix_rho.f90` | diagonalize → build density → build potential → Broyden mix |
| Hamiltonian application | `h_psi.f90`, `vloc_psi_*.f90`, `add_vuspsi.f90`, `g2_kin.f90`, `s_psi.f90` | the hot path; the natural unit of `jit`/`vmap`; `k` must stay a traced argument, see `PLAN.md` §6 |
| Iterative diagonalization | `KS_Solvers/Davidson/`, `KS_Solvers/CG/`, `KS_Solvers/PPCG_legacy/`, `KS_Solvers/RMM/` | Davidson is QE's default and is ported (`solvers/davidson.py`); note `c_bands.f90` re-enters `cegterg` up to 5 times, so QE's real budget is 100 steps |
| FFT / G-vector grids | `FFTXlib/`, `PW/src/data_structure.f90`, `Modules/recvec*.f90` | replace with `jax.numpy.fft`; the sphere-to-box G-vector mapping still has to be reproduced |
| Pseudopotentials | `upflib/` (`read_upf_new.f90`, `pseudo_types.f90`, `init_us_2.f90`, `sph_bes.f90`, `ylmr2.f90`) | UPF v2 XML parsing + radial→G-space transforms. **`msh` is one or two points *past* 10 bohr** — QE's loop takes the first index beyond the cutoff, not the last inside; getting that wrong is worth 1e-6 Ry on a `psl` dataset and nothing at all on `Si.pz-vbc` |
| Ultrasoft augmentation | `upflib/qvan2.f90`, `uspp.f90` (`aainit`), `qrad_mod.f90`, `PW/src/addusdens.f90`, `newd_acc.f90`, `s_psi.f90` | `Q_ij(G)`, `becsum`, the overlap operator, and `D_ij` rebuilt each iteration from the potential |
| PAW one-centre terms | `PW/src/paw_onecenter.f90`, `paw_init.f90`, `paw_symmetry.f90`, `upflib/radial_grids.f90` (`hartree`) | radial Poisson (a Numerov tridiagonal solve — transcribe it, do not substitute the closed form), a Gauss-Legendre×φ spherical quadrature for XC, and `becsum` symmetrisation, which is **not optional** on a reduced k-set. A GGA adds `PAW_gcxc_potential`: the quadrature grows (`xlm`), the vector field is expanded two multipoles past the density, and its θ component is divided by `sin θ` before projection |
| XC functionals | `XClib/`, `PW/src/gradcorr.f90` | must be reimplemented in pure JAX — a `libxc` binding is neither differentiable nor GPU-capable (see `PLAN.md` §6). Only the **energy** is written down; `v_xc`, and a GGA's `v1`/`v2`, come from `jax.grad`. QE composes a functional from four independently chosen slots and UPF headers name all four, so `xc/functional.py` does the same |
| Spin-orbit coupling | `upflib/init_us_1.f90` (`fcoef`, `dvan_so`), `upflib/spinor.f90`, `upflib/sph_ind.f90`, `upflib/upf_spinorb.f90` (`transform_qq_so`), `PW/src/newd_acc.f90` (`newd_so`), `PW/src/compute_becsum.f90` (`add_becsum_so`), `PW/src/vloc_psi_acc.f90` (`vloc_psi_nc`), `PW/src/add_vuspsi_acc.f90`, `PW/src/usnldiag.f90` | `init_us_1` builds `fcoef` for every matching `(l, j)` pair, uses it for `dvan_so`, and **then** zeroes the cross-radial entries — everything downstream consumes the *zeroed* array and has no check of its own, so one array used for both is a correct `dvan_so` and a silently wrong `qq_so`/`deeq_nc`/`becsum` |
| Structure / symmetry / k-points | `PW/src/symm_base.f90`, `symme.f90`, `kpoint_grid.f90`, `setup.f90`, `Modules/cell_base.f90` | `ibrav` lattice conventions live in `Modules/latgen.f90`. `kpoint_grid` is called with the *lattice* point group and fixed up afterwards; reducing directly with the crystal's symmetries reaches the same orbits. Two rules in `symm_base.f90` change the **FFT grid**: dimensions must be a multiple of the fractional translations' denominators (`fft_fact`), and a cell that is a supercell has fractional translations disabled altogether |
| Starting wavefunctions | `PW/src/wfcinit.f90`, `Modules/atomic_wfc_mod.f90`, `upflib/atwfc_mod.f90` | the projectors' expression with `chi` for `beta` — but the phase is `i^l`, not `(-i)^l` |
| Van der Waals dispersion | `Modules/mm_dispersion.f90` (`energy_london`, `force_london`, `stres_london`), `Modules/set_vdw_corr.f90`, `Modules/rgen.f90`, `upflib/atomic_number.f90` | the energy is written down (`pypresso/vdw/grimme.py`) and the force and stress are `jax.grad` of it; QE's two expressions are transcribed as the cross-check. `rgen`'s **fold** of the pair separation into the cell is kept, and it is what lets one neighbour list serve every geometry |
| Ewald / local potential | `PW/src/ewald.f90`, `setlocal.f90` | the ion-ion sum and `V_loc(G)`; the Ewald neighbour list is fixed for the *cell*, not the geometry, so it survives a relaxation |
| Forces | `PW/src/forces.f90`, `force_lc.f90`, `force_cc.f90`, `force_ew.f90`, `force_us.f90`, `addusforce.f90`, `force_corr.f90`, `symme.f90` (`symvector`) | the default is `jax.grad` of the energy at frozen wavefunctions (`forces/energy.py`); the Fortran expressions are transcribed as a cross-check. `gradcorr` is called from **inside** `v_xc`, so `force_cc` needs it |
| Structural relaxation | `Modules/bfgs_module.f90`, `PW/src/move_ions.f90`, `run_pwscf.f90`, `update_pot.f90`, `checkallsym.f90` | BFGS in crystal coordinates with the cell metric; the setup (FFT grid, symmetry, k-points) is done **once** and only checked afterwards |
| Stress | `PW/src/stress.f90`, `stres_knl.f90`, `stres_har.f90`, `stres_loc.f90`, `stres_cc.f90`, `stres_gradcorr.f90`, `stres_ewa.f90`, `symme.f90` (`symmatrix`) | the default is `jax.grad` of the energy with respect to a strain at frozen wavefunctions (`stress/energy.py`, through `Calculation.at_strain`); the Fortran expressions are transcribed as a cross-check. `stres_us`/`addusstress` are **not** transcribed, so the analytic route offers terms and no total. `ylmr2`'s `atan2` parameterisation is singular on the `z` axis and only a *cell* derivative reaches it |
| Magnetic symmetry | `PW/src/symm_base.f90` (`sgam_at_mag`), `symme.f90` (`sym_rho`'s `nspin = 4` branch), `PW/src/irrek.f90` | the magnetization is an **axial** vector, so its rotation carries `det(R)` and a further sign for an operation that is a symmetry only with time reversal; `irreducible_BZ` completes an explicit k-list from the lattice's wedge to the crystal's, and runs for every SCF |
| Fields and constraints | `PW/src/add_bfield.f90`, `make_pointlists.f90`, `get_locals.f90`, `report_mag.f90`, `PW/src/input.f90` (`i_cons`) | the penalty's *energy* is written here and its potential comes from `jax.grad`; QE's five expressions are transcribed as the cross-check. Elk's counterparts: manual §5.2/§5.12/§5.104, `src/bfieldfsm.f90` |
| Spin spirals | no QE counterpart — Elk's `src/gengkqvec.f90`, `init0.f90`, `findsymlat.f90`, manual §5.146 | up at `k + q/2`, down at `k - q/2`, each with its own `G+k` set; one basis call on the concatenated list gives both a common `npwx` |
| Spiral relaxation | no QE counterpart — `Modules/bfgs_module.f90` reused with the *reciprocal* cell as its lattice | `dE/dq` is `jax.grad` of the energy at frozen wavefunctions and a frozen sphere (`forces/spiral.py`); only the kinetic and nonlocal terms carry `q` |
| Berry phase / topology | `PW/src/bp_c_phase.f90` (the ultrasoft `q_ij(b)` and the k-string overlaps), `Modules/bfgs`-free | the invariants themselves have no QE counterpart to transcribe — `pypresso/topology/` follows Fukui-Hatsugai-Suzuki, Yu-Qi-Bernevig-Fang-Dai and Fu-Kane, with `bp_c_phase.f90` as the reference for how the augmentation charge enters an overlap between two different k-points |
| Velocity / position operator | `PW/src/commutator_Hx_psi.f90`, `PP/src/` Berry-phase code | QE hand-codes `[H,r]` term by term; here it is one `jvp` of `H(k)` at a frozen sphere (`response/velocity.py`), since `dH/dk_a = i[H, r_a]` in the periodic gauge. The overlap carries a velocity too, so a band velocity is `<psi|dH/dk - eps dS/dk|psi>` |
| Linear response / DFPT | `LR_Modules/cgsolve_all.f90`, `ch_psi_all.f90`, `orthogonalize.f90`, `h_prec.f90`, `setup_alpha_pv.f90`, `incdrhoscf.f90`, `symdvscf.f90`; `PHonon/PH/solve_e.f90`, `dvpsi_e.f90`, `dvqpsi_us.f90`, `dielec.f90`, `zstar_eu.f90` | the linear solve, the projector and the assembly are transcribed; the *perturbations* are not. `dv_of_drho` is one `jvp` of `v_of_rho` (which already drops the `G = 0` Hartree term), the E-field's commutator is the velocity operator, and `dvqpsi_us` is one `jvp` through `at_positions`. **A response on a reduced k-set is a polar vector field and must be symmetrised as one** |
| Input parsing | `Modules/read_input.f90`, `PW/src/input.f90`, `Modules/input_parameters.f90` | defaults for every input variable are declared in `input_parameters.f90` |
| DFT+U | `PW/src/ldaU.f90`, `hubbard.f90`, `new_ns.f90`, `init_ns.f90`, `ns_adj.f90`, `orthoUwfc.f90`, `offset_atom_wfc.f90`, `vhpsi.f90`, `v_of_rho.f90` (`v_hubbard`), `scf_mod.f90` (`ns_ddot`), `force_hub.f90` | the projectors are `S phi` even for `Hubbard_projectors = 'atomic'`; `ortho-atomic` orthogonalises over **all** `natomwfc`, not the Hubbard manifold alone, so `Modules/read_pseudo.f90`'s `upf_check_atwfc_norm` renormalisation of `chi` reaches the answer through the `4s`. `force_hub.f90` is *not* transcribed: it is `jax.grad` through `Calculation.at_positions` |
| Occupations / smearing | `PW/src/gweights.f90`, `Modules/wgauss.f90`, `Modules/w0gauss.f90`, `PW/src/set_occupations.f90` | |
| NSCF / band structure | `PW/src/non_scf.f90`, `PP/src/bands.f90`, `PP/src/plotband.f90` | fixed density, diagonalize once per k on an explicit path |
| DOS | `PW/src/tetra.f90`, `PP/src/dos.f90` | `tetra.f90` has both the linear and the Bloechl-corrected tetrahedron method |
| Projected DOS | `PP/src/projwfc.f90` (`projwave`, `sym_proj_k`, `print_lowdin`), `PP/src/projections_mod.f90` (`fill_nlmchi`), `PP/src/partialdos.f90`, `PW/src/tetra.f90` (`opt_tetra_partialdos`) | the projectors are `orthoUwfc`'s, so `hubbard/projectors.py` builds them for both; the weighted integration goes through the *same* DOS registry, and `do_projwfc` silently runs the **linear** tetrahedron method whatever the SCF used |

Fortran conventions that carry over: arrays are column-major and 1-indexed, so index order
must be reversed when transcribing loops; internal units are Rydberg atomic units (energy
in Ry, length in bohr) throughout `PW/`.

## Mirror QE in the performance-critical path

**Where performance matters, reproduce QE's implementation rather than inventing
one.** Not just its formulas — its data layout, its loop structure, and the order it
does things in. Thirty years of plane-wave practice is encoded in choices that look
arbitrary until they are measured, and the measurement usually agrees with the Fortran.

This is a standing rule because guessing has now been wrong more than once, always in
the same direction — an idiomatic-JAX version that looked equivalent and was slower:

- **The FFT layout.** QE transforms the wavefunction `z` axis only over the *sticks*
  the sphere occupies, then does a 2D `xy` pass — and its arrays are Fortran-ordered,
  so the `xy` plane is contiguous. Transcribing the decomposition into a C-ordered box
  puts the 2D pass on the two strided axes, where it costs more on its own than a fused
  3D transform of the whole box; done in QE's layout it is a win. Same algorithm,
  opposite result, and the difference is entirely the layout (`basis/sticks.py`).
- **The Davidson loop.** `cegterg` extends its projected matrices a block at a time and
  tests convergence *after* expanding. Recomputing the projections each step costs a
  factor of `nvecx/nbnd`; testing before expanding wastes one `h_psi` per call. Both
  were invisible on a two-atom cell and obvious on eight.
- **The diagonalisation threshold.** `electrons.f90` schedules `ethr` against the error
  in the density. A fixed tight threshold does three times the eigensolver work.

The corollary for measurement: **a two-atom cell will not show you any of this.**
Benchmark on `benchmarks/si8-1k*.in` or `si16-1k*.in`, where the cost is the physics
rather than fixed overheads, and check that a change helps *there* before believing it.

Two things this rule does not mean. It does not license transcribing QE's Fortran
control flow into Python — the JAX rules above still bind, and `cegterg`'s dynamic
reshaping becomes masks and static shapes. And it does not override differentiability:
where QE's fast path is a table lookup, the differentiable equivalent wins (`PLAN.md`
D1/D2), and that trade is recorded rather than silently taken.

## Memory is part of the design

**A design is not finished until its peak working set is known.** A plane-wave code is
memory-bound as often as it is compute-bound, and what decides whether a calculation runs
at all is usually a working set rather than a flop count. Before landing anything that
allocates per k-point, per band, or per G-vector, say what the peak costs in terms of
`nk`, `nbnd`, `npwx`, `npol` and the FFT grid, and put that number against the RAM of a
real machine — the same reflex the performance rule above asks for with time.

**Where QE spends effort to save memory, copy it unless something better is on offer.**
Thirty years of running problems larger than the machine is encoded in these choices, and
none of them is incidental to the algorithm:

- **One k-point at a time.** `c_bands.f90`'s `k_loop` diagonalises a single k-point and
  `sum_band.f90` accumulates the density inside the same loop, so QE's working set is one
  k-point's whatever `nks` is; the other k-points' `evc` sits in a buffer that is RAM or
  disk according to `io_level`/`disk_io`, and the parallelism over k comes from MPI pools.
  Batching the whole k axis with `vmap` is this code's deliberate deviation — it is what a
  GPU wants — so it is a **dial** (`pypresso/batching.py`), defaulting to QE's end of it,
  not a fixed choice. Rule R6 (k leading) is what keeps both available.
- **The sphere, not the box.** Wavefunctions live on the G-vectors inside the cutoff and
  are expanded into the FFT box only for the transform, and only over the sticks the
  sphere occupies (`basis/sticks.py`).
- **Two grids.** The smooth grid carries the wavefunctions and the dense one only the
  augmentation charge that needs it, which is most of the point of `ecutrho`.

A deviation is allowed and is sometimes right — this code trades memory for batching the
way QE trades it for MPI ranks, and `becsum` is carried as a full symmetric matrix where
QE packs the upper triangle. The rule is that such a trade is **stated, measured, and made
selectable when it is large**, never arrived at by accident: name it in the module
docstring, and put the number in `PERFORMANCE.md` beside the timing.

## Reading beyond the source

The vendored Fortran is the primary reference and transcription from it is the method.
Where an algorithm's *reasoning* is not in the source — why a preconditioner has the form
it does, what a method's convergence properties are, what the alternatives are — **arXiv
is a legitimate thing to consult during implementation.** Cite what was used in the module
docstring, the same way the Fortran file it came from is cited.

## Validation against reference QE

`quantum_espresso/qe-7.5-ReleasePack/qe-7.5/test-suite/` holds ~100 test cases with
committed reference outputs — use these as the ground truth rather than re-running QE.
For the SCF core, `test-suite/pw_scf/` is the relevant set: `scf-*.in` are the inputs and
`benchmark.out.git.inp=scf-*.in` the expected outputs (total energy, eigenvalues, forces,
stress are all parseable from those files). `test-suite/pw_atom/`, `pw_lsda/`, `pw_metal/`,
`pw_relax/` extend coverage. Test pseudopotentials are in `pseudo/` (e.g. `C.UPF`,
`Si_r.upf`, `N-PBE.upf`).

The test-suite's pseudopotential files are **not** shipped — inputs name files like
`Si.pz-vbc.UPF` that `test-suite/check_pseudo.sh` downloads from
`pseudopotentials.quantum-espresso.org`. Fetch them once into `tests/data/pseudo/` and
commit them. The canonical first target is `test-suite/pw_scf/scf.in` (Si diamond, LDA,
`ecutwfc=12`, 2 k-points, 15³ FFT grid).

The testing method is running the *same input* through real QE and through pypresso and
comparing numbers. Building the Fortran QE is only needed when a comparison is not already
covered by a committed benchmark (likely for `bands`/`dos` runs); when that happens, store
the generated reference output alongside the test so it never has to be regenerated.
Tolerances per quantity are listed in `PLAN.md`.

## Environment

Dependencies live in the **base anaconda env** — there is no project virtualenv, so
`python3` is already the right interpreter. JAX 0.11.0, NumPy 2.4.6, SciPy 1.18, Numba
0.65, equinox 0.13.8 (verified working with this JAX under x64). Development is CPU-only
here; the JAX paths must run unchanged on GPU, so correctness is established in float64 on
CPU and performance work is a later, separate phase.

Compiled kernels are cached in `~/.cache/pypresso/jax` so that only the first run of a
process pays for them; `PYPRESSO_CACHE_DIR` moves it and `PYPRESSO_CACHE_DIR=off` disables
it.

```
python3 -m pytest                      # whole suite
python3 -m pytest -m unit              # fast checks only (markers: unit, regression, slow)
python3 -m pytest tests/unit/test_qeref.py::test_scf_silicon   # a single test
python3 -m pypresso.cli inspect <qe-output>   # summarise what the parser reads
tools/export_notebooks.sh                     # re-execute notebooks + refresh .md exports
```

## JAX rules

- **`jax.config.update("jax_enable_x64", True)` must be set before any array is created.**
  JAX defaults to float32; SCF will not converge and no comparison against QE benchmarks
  will be meaningful in single precision. Set it once in the package `__init__`, before any
  other import that touches JAX. Enabling x64 only *permits* 64-bit — the actual dtype of
  every array still comes from `config.dtypes` (see conventions above).
- The SCF loop's convergence test is data-dependent, so keep the loop in Python; `jit` the
  iteration body (`h_psi` → diagonalize → density → potential → mix). Inside the
  eigensolver, use `lax.while_loop`/`fori_loop` with a fixed subspace size so the solver
  stays on device.
- **How many k-points are in flight is `pypresso/batching.py`'s dial, and its default is
  QE's loop** — one k-point at a time, as `c_bands.f90` and `sum_band.f90` do it. `k_batch`
  reaches every entry point (`run_scf`, `run_bands`, `run_nscf`, `run_dos`, `Calculation`,
  `PYPRESSO_K_BATCH`), `None` asks for one `vmap` over the whole axis, and the chunked form
  is a `lax.map`/`lax.scan` so it stays compiled once and differentiable. Anything new that
  walks the k axis goes through `map_k`/`sum_k` rather than calling `vmap` itself; the
  chunk size must never be visible in a result beyond round-off.
- Use `donate_argnums` for the large wavefunction and density buffers.

`PLAN.md` §1 and §5 hold the full reasoning and the rest of the GPU notes.
