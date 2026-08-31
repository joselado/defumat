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
**Forces and the stress are spinor now too** (P46): the frozen-energy functional P15
differentiates grew a two-component branch — the nonlocal quadratic form with `dvan_so` and
the orthonormality constraint with `qq_so`, both complex 2x2 matrices in spin space, on the
`2 npwx`-long coefficient vector a spinor actually is — so `noncolin = .true.`, with or
without `lspinorb`, has forces, a stress and a relaxation on norm-conserving, ultrasoft and
PAW datasets. The plumbing was the larger half of it and this file's own rule is why:
`nspin`, `npol` and `nspin_mag` are three different numbers, so a spinor state is
`(1, nk, nbnd, 2 npwx)` and the *kinetic* term has to read `state_kinetic` as well. Against
`pw.x`: a four-atom noncollinear hydrogen chain to **8.9e-7 Ry/bohr**, doubled fcc platinum
with spin-orbit coupling to **7.5e-6** (ultrasoft) and **7.3e-7** (PAW), and the stress on
six cases to **≤1.2e-6 Ry/bohr³** — three of which needed no new reference, QE's own
`pw_spinorbit` inputs already carrying `tstress`. The anchor underneath all of that is a
finite difference of the frozen energy, which agrees to **6.2e-9 Ry/bohr** and involves no
Fortran. Refused by name and each for its own missing term: the **analytic** transcriptions
(`force_us`/`stres_knl` have no spinor form), anything through the **Sternheimer** solver,
the **elastic constants** — which reach the functional directly, which is why the spinor path
is opt-in rather than merely allowed — and the force on an atom of a **spin spiral**, whose
two components sit on different spheres.
**Berry curvature, Chern numbers and Z2 invariants** (P16) are in too: the Chern number is
an exact integer on a 6x6 mesh, and the Z2 has both the Wilson-loop and the Fu-Kane parity
route, agreeing on every model case with a known answer. **The smooth `Omega(k)` map is in
as of P47**, by the Kubo route on P24's velocity operator rather than on a dense `H(k)` —
agreeing with the Fukui-Hatsugai-Suzuki flux to 1.45e-4 plaquette by plaquette on zincblende
AlAs, vanishing pointwise on silicon to 3.5e-5, and reporting the truncation of its sum over
empty states rather than tuning it away.
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
**The Sternheimer solve is spin-polarized now** (P45): the widest guard in the package — one
refusal that blocked every response quantity for every `nspin = 2` system — was an
occupied-band count, and it is one number *per channel* now (`occupied_counts`), which is
what QE gets for free by doubling `nks` in LSDA. `chi_0` matches a central difference of the
density to **1.8e-6** for an antiferromagnetic hydrogen chain (a smeared metal) and to
**1.1e-6** for triplet O2 (the sliced branch, ultrasoft, seven bands up and five down),
under a probe potential that is **different in the two channels** — `chi_0` is block-diagonal
in spin, so a probe equal in both would not tell the blocks apart. The dielectric constant of
a cell with no magnetization comes out identical run as `nspin = 1` and as `nspin = 2`
(**6.2e-14** on 13.806646105), which is the check that catches a factor of two in the spin
sum. **Two things fail silently and both are refused by name now.** A filling that cuts a
**degenerate multiplet** — the oxygen atom at `tot_magnetization = 2`, whose minority channel
splits the 2p shell — lets the CG converge in 42 iterations and returns a `chi_0` that is
**100 per cent** away from a finite difference, because the difference re-selects which
member falls below the cut and the solve keeps the arbitrary one the eigensolver handed it.
And the **screening kernel of a magnetic system with vacuum is not finite** (measured on one
cell): `dv_of_drho` for `nspin = 2` is the second derivative of the LSDA energy in the two
channel densities, which diverges where a channel density reaches zero — 1504 of triplet O2's
91125 grid points have `|m| >= |n|` and `dv_of_drho` has exactly 1504 NaN. That is the `abs`
trap of P28a one derivative further out, it is in `pypresso/xc` rather than in the response,
and pulling the clip inside does *not* fix it. **Also refused by name**: `tot_magnetization`
with a smearing (two Fermi levels, and `Smearing.ef` has no spin axis), Born charges, the
dynamical matrix and the strain response for `nspin = 2` (their assembly, not their count),
and the two third derivatives.
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
**Ten atoms per cell is where the whole feature set was run against `pw.x` at once**
(P28b), and it found three more bugs of the same family — things only a supercell can
see. The **lattice point group was searched over a fixed `range(-3, 4)` window**, which
cannot hold the entries of five that five stacked primitive cells put in a rotation
matrix: 2 operations found where QE finds 6, and a total energy 3.2e-6 Ry out with both
codes converged to 1e-10 and both reporting success. **A fractional translation was
accepted whatever its denominator**, where `sgam_at` takes only `1/n` with `n` in
{2,3,4,6}: five-layer graphite kept a real mirror plane QE drops, `fft_fact` then wanted a
20x20x**135** grid where `pw.x` chooses 128, and the totals differed by 1.7e-4 Ry with
neither code wrong. And **`dielectric_tensor` symmetrised a `nosym` run**, which is
invisible wherever the k-grid is closed under the point group and worth 0.97 in the
off-diagonal entries where it is not. A fourth divergence is nobody's bug and is worth
knowing: on a k-grid with **unequal divisions** the two codes build genuinely different
irreducible sets — QE completes the lattice wedge with `irreducible_BZ`, whose star
members leave such a grid — and `pw.x`'s own `nosym` run says which one is the grid's.
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
closed grid to 2.7e-14. **Ultrasoft and PAW are in as of P39**: with `S` moving, the source term becomes
`(dH/du - eps dS/du)|psi>`, the first-order state acquires an occupied block the solve
does not produce, the mixed state changes at *frozen* states (`drho.f90`), and the
orthonormality multipliers move as a **matrix** — a diagonal one is not invariant under
the occupied-manifold rotation the state tangent is free in, and the sum rule says so.
Ultrasoft silicon is **513.2947** cm⁻¹ against `ph.x`'s 513.275287 and PAW **513.3776**
against 513.404419 — 0.019 and 0.027, tighter than the norm-conserving 0.05. Getting
there found two bugs that are **not** ultrasoft terms: `addcore` was missing for every
dataset (no committed phonon case had a core charge, so a norm-conserving pseudopotential
with one was wrong too), and `addusforce` was missing from the differentiated gradient,
so what was being differentiated was not the force. And one the sum rule could not see —
the density's cross derivative `d^2 rho/du dpsi`, `addusdynmat`, which needs both tangents
in one `jvp` where P28's weight split had put them in two. An ultrasoft or PAW **metal**
is refused for exactly that reason.
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
and **clamped-ion**, on an **unshifted** k-grid — which is closed under the point group where a
shifted one is not. A symmetry-reduced wedge of it works as of P36; the *elastic constants*
still need the whole grid, and for a different reason (their functional builds its own density
and symmetrises it as a scalar, inside the chain rule).

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

**Variable-cell relaxation** (P29) is in: the cell relaxes with the atoms in one BFGS over
`3 nat + 9` coordinates at an applied pressure, matching `pw.x` on all four of its
`pw_vc-relax` BFGS cases — arsenic at 500 kbar compressing 10% and going simple cubic
(0.2722 → 0.2500) agrees on the relaxed volume to **7e-4 bohr³** and on the final energy to
**2.4e-6 Ry**, in the same number of ionic steps. Getting there found two bugs, both of the
P28a family — the energy right and something else not. **`at_strain` rebuilt its k-points
from `system.kpoints`**, whose cartesian coordinates describe a k-set only together with the
cell they were built for; every earlier caller deformed a cell whose k-points had just been
built for it, and a cell that has actually *moved* separates them, so a stress on a stepped
cell was differentiated **0.031 away in crystal coordinates** from where the SCF had run —
64 kbar, and 2% of the relaxed volume. A finite difference of the frozen-basis energy is
what settled it against the plausible story that autodiff was seeing a Pulay term QE misses.
And **the lattice symmetry tolerance was dimensional**: an absolute 1e-6 applied to bohr and
bohr² where `symm_base.f90`'s `eps1` applies to `at` in units of `alat`, so the same crystal
loses operations as its lattice constant grows — eight of twelve dropped on QE's own
`vc-relax4.in`.

**The Tran-Blaha potential is in** (P30), and it is the one functional here whose
*potential* is written down and whose energy does not exist: `input_dft = 'tb09'` gives the
modified Becke-Johnson meta-GGA, so silicon's gap goes from LDA's **0.49 eV to 1.13 eV**
against an experimental 1.17 and the published all-electron mBJ's 1.17, and diamond's from
3.89 to 4.43. There is nothing to transcribe — `pw.x` reaches TB09 only through libxc, and
then **passes a zero Laplacian** (`xc_wrapper_mgga.f90` calls the argument "not used in QE")
and **never sets `c`**, so what it runs under that name is Becke-Johnson without a Laplacian.
Both ingredients are here: the Laplacian is `-G^2 rho(G)`, one transform, and `c` is Tran and
Blaha's cell average, with `mbj_c` to impose it instead. Validated against two *analytic*
limits rather than another code's floating point — the hydrogen atom, where Becke-Roussel is
the exact Slater potential of the 1s orbital to 1e-6 and `E_x` is exactly -5/16 Ha, and the
uniform gas, where Becke-Johnson reproduces `v_x^LDA` to 6e-4 (which is the model's own error
at `gamma = 0.8`, and measuring it showed that 0.8 *is* the uniform-gas fit, to four digits).
The **total energy is not variational**, so forces, stress, phonons and response are refused
by name. **PAW works** (P32) and is what makes the difference: its one-centre `tau` comes from
the partial waves, and it recovers `c = 1.107` against the all-electron 1.12 where a
norm-conserving silicon measures 1.000 — the pseudised core is exactly what the average of
`|grad rho|/rho` misses. **Noncollinear magnetism and spin-orbit coupling work too** (P31),
with `tau` carried as the 2x2 matrix in spin space it is and resolved onto the density's local
axis. `pw.x` refuses both combinations outright (`setup.f90`: 'Meta-GGA not implemented with
USPP/PAW', 'Non-collinear Meta-GGA not implemented'). Plain **ultrasoft** stays refused: it
has no partial waves to reconstruct `tau` from inside the sphere, where PAW does. One thing a
UPF cannot supply is a **core kinetic energy density**, so the one-centre term sees the
valence density alone on both sides — including the all-electron core in `rho` with no `tau`
to match it inverts the functional's `c` dependence, measurably (`PLAN.md` P32).

**The Raman tensor is in** (P35), and it is P26's third derivative with the atoms as its
geometry variable rather than the cell: `d(eps)/d(tau)` is one `jvp` of the *same*
variational second-order energy, so the phase is an assembly of tangents that already
existed — the displacement response P25 solves for the dynamical matrix, the field response
P24 solves for `epsilon`, and P26's own extra Sternheimer solve for the position operator.
**The reference for it is broken and establishing that came first**: the vendored `ph.x`
7.5 does not reproduce QE's committed `PHonon/examples/example05` (v6.0, 2016) — -1.8681
against -0.78497 on the Raman tensor — and fails its *own* internal check, printing a
finite-difference dielectric constant of -0.288 beside its analytic 8.8143 where v6.0 has
8.8116 beside 8.8147. So the validation is a **finite difference of `epsilon` over
re-converged displaced cells**: **-3.118279** against **-3.118310**, 1.0e-5. What `pw.x`
cannot do at all is a **GGA** — `phq_setup.f90` stops on "third order derivatives not
implemented with GGA" because its third derivative of `E_xc` is a hardcoded Perdew-Zunger
parameterisation (`d2mxc.f90`), where here it is one more `jvp` of a kernel that already
exists. **`chi^(2)` and the electro-optic tensor are refused by name**, with the missing
term identified rather than fitted: the field enters only through the source term, so the
`<u_i|r_k|u_j>` piece of the 2n+1 expression (QE's `dvpsi_e2`/`solve_e2`) has nothing to
build it from — and it is **42% of the answer**, measured on its displacement counterpart.
**No symmetry check catches its absence**, which is the finding worth carrying: without it
the tensor still vanishes identically in a centrosymmetric crystal, still comes out exactly
zincblende, and is still symmetric under every permutation of its three labels to 2.5e-13.

**Raman and infrared spectra are in** (P36), and so is **the rank-3 symmetriser** that
made them cheap. `symme.f90`'s `symmatrix3`/`symtensor3` are written here **at any rank**,
which lifts the closed-grid refusal P26 introduced and P35 inherited: AlAs's eight-point
wedge reproduces its sixty-four-point closed grid to **3.3e-9** and silicon's rank-4
elasto-optic tensor to 7.9e-14, at half the cost. **The AlAs number was 8.7e-14 when the
phase landed and is not any more, for a reason that is not P36's**: a third derivative
multiplies the difference between two converged densities by the norm of a first-order
wavefunction (order 10^3), so what the wedge and the closed grid agree to is what their
SCFs agree to — and the mixer normalisation in `a351005` (Gram block cond 1.1e11 → 2.7e4,
and a NaN fixed with it) stopped the two k-sets landing on the same fixed point bit for
bit. It is convergence-limited and measured as such: 3.3e-9 at `conv_thr = 1e-12`, 6.5e-10
at 1e-14. Both routes are still right — each gives 3.119 against the -3.1183 of a finite
difference over re-converged displaced cells. The silicon 7.9e-14 is from a grid-sharing
pair that no test exercises and has **not** been re-measured since. The average alone is not enough,
and that is the phase's finding: it completes a wedge sum only where the tensor is a
*linear* k-sum of a covariant per-k quantity, and the screening term of `F` is **quadratic**
in one — so the *value* of the density response inside the functional must be the full-zone
object while its *derivative* stays the raw wedge sum. Getting that wrong is worth 2.5%, is
worse than doing nothing, and **no symmetry check sees it** — the sum rule is what caught
it, one more time. On top of that the spectra themselves: P35's per-atom tensors contracted
with P25's modes and P24b's `Z*` into per-mode activities, matching the vendored `dynmat.x`
on every digit it prints, with silicon's `T_2g` at **519.2 cm⁻¹** against an experimental
520 — Raman-active and infrared-silent, which is a symmetry statement rather than a fit.
**`dynmat.x` is the one reference above second order that still works**, because `RamanIR`
is post-processing and never touches the branch that regressed. **A degenerate multiplet is
comparable only as a sum**: the two eigensolvers land in different bases inside silicon's
acoustic triplet and print depolarisation ratios of 0.3544/0.7163/0.4065 against
0.5873/0.2446/0.7264, on modes whose activity both codes give as 0.0000.

**Optical spectra with excitons are in** (P37), and they are the first thing here built on
a **sum over states**: an absorption spectrum needs `chi_0` as a matrix over reciprocal
lattice vectors at every frequency, where the Sternheimer stack produces it as a static
operator. `run_absorption` solves the Dyson equation with a kernel from a registry —
`rpa`, `alda`, `lrc`, `bootstrap` (Elk's `fxctype = 210`) and `bootstrap-1` — the bootstrap
being a fixed point of the Dyson equation and its own definition, parameter-free and
convergent in **9 iterations** on silicon. **The reference is Elk and it is validated by an
identity instead**, because an all-electron LAPW spectrum is not a comparable number: the
same `eps_M(0)` reached by this sum over states plus a Dyson inversion and by the projected
CG solve of `dielectric_tensor` — which shares no machinery with it and never sees an empty
state — agree to **1.3e-2 on a constant of 22**, and that residue is the band truncation,
which is reported (`static_residual`) rather than tuned away. Three traps, all of them
producing a smooth, positive, plausible spectrum: **`eps_M` is the inverse of the 3x3 head
of `eps^-1`, not the head of the inverse** (Elk writes both from one array thirty lines
apart; the wrong one is exactly the no-local-field result, 9% high); the **identity holds
only when the two kernels match**, so `dielectric_tensor` gained a `screening = 'hartree'`
switch, since its own kernel is `dv_of_drho` and therefore ALDA; and **the diagnostic is
broken by a scissors shift** the same way, which turns a `+0.013` residual into `-3.46`.
**The head is the one line of Elk that must not be transcribed** — it reads momentum matrix
elements, right in an all-electron code and wrong with a nonlocal pseudopotential — so
`dH/dk` from P24's `jvp` takes their place, and it is this phase's only load-bearing
autodiff. Everything else is a transcription and says so.

**The whole of it is reachable from one object** (P38): `Calculator.from_file("scf.in")`
loads the input and the pseudopotentials it names, `get_scf()` caches the ground state,
and every quantity above is a method consuming that cache — with nothing mutating, the
refusals passing through untouched, and the functional API unchanged beneath it. It found
one real gap on the way in: `run_dos` never forwarded `becsum` or `ns`, so a **PAW or
DFT+U density of states on a denser grid was unreachable**, stopping on `run_nscf`'s own
refusal rather than being wrong.

**The strain response is ultrasoft and PAW too** (P41): `Q_ij(r)` is a function of the
*cell*, so a strain deforms the table where a displacement translates it, and `at_strain`
already rebuilds it — the four terms P39 wrote transfer across, and `drho/d(eps)` matches a
central difference of re-converged strained runs to **4.6e-4** (US) and **4.7e-4** (PAW)
against a norm-conserving 1.9e-4. **And the functional the strain derivatives stand on is
ultrasoft and PAW too** (P43): P26's second-order energy reproduces `dielec.f90`'s dielectric
constant to **3.4e-10** (US) and **6.9e-11** (PAW), which took `becsum` inside its density,
`ddd_paw` in its Hamiltonian, the `S` metric in its projector and multiplier — including
the distinction that a *state* takes `1 - Σ|psi><psi|S` and a *right-hand side* takes
`1 - Σ S|psi><psi|` — and PAW's one-centre screening term.

**The Raman tensor is ultrasoft and PAW now as well** (P43), at **1.2e-4** against a
finite difference of `epsilon` over re-converged displaced cells where the norm-conserving
control is 6.8e-4 — and it took **two tangents that are only right together**, which is why
one of them had already been measured and read as an exclusion. The state tangent is
`P_c dpsi + ort`, P39's occupied block; and **`b` is not the solution of its own linear
equation**, because `dvpsi_e` solves for `P_c r|psi>` and `adddvepsi_us` then applies `S`
and adds the augmentation dipole, so `db` is the tangent of a *composition* and the frozen
solution the residual is written about is `commutators`, not `b`. Either alone is worse
than neither (3.0e-2 → 8.0e-2 for the block, 2.1 apart for the tail); both together land
on the finite difference. What found them is that **`d(eps)/d(tau)` is a sum of five
partial derivatives and each can be measured against its own finite difference** — three
agreed to 7e-4 and two did not, which localised the bug instead of guessing at it. A third
thing had to change before either was reachable: `VelocityOperator.projectors` read the
atoms with `np.asarray`, so `d(beta)/dk` about the atom's own centre was not
differentiable in the geometry at all and the term vanished silently.

**The same third derivative in the *strain* coordinate is measured and still refused**
(P44) — the elastic constants, electrostriction and the elasto-optic tensor. Two of P43's
ingredients transfer and are wired in behind the guard (`dpsi + ort` and
`stored = commutators`), taking `d(eps)/d(strain)` from **4.6e-2 to 1.3e-2** on ultrasoft
and 5.5e-2 to 1.3e-2 on PAW against a central difference of `epsilon` over re-converged
strained cells, where the norm-conserving control on the same script is 2.3e-4 and does
not move. Thirty times better and fifty times the control, so the refusal stays. **The
decomposition says the whole residue is the `b` partial** — `ort` takes the `psi`
partial's error from +5.94 to +0.032 and the other three agree to 1.4e-3 — and it is
**−1.72 of 112, the same number on ultrasoft and on PAW**, which is what makes it
structural rather than a dataset's physics. **One candidate is excluded by measurement,
and that is the finding**: `_position_response`'s commutator *source* holds `eps_n` as a
frozen scalar where the operator beside it has carried the multiplier matrix since P26,
and removing that asymmetry — at no change of value — takes the strain to **1.7e-4** and
takes the *Raman* tensor from 1.2e-4 to **1.14e-3**, in every operator/source pairing
tried. One of the two coordinates has a compensating term; adopting this one on the
strain column alone would be a fit that regresses a validated result.

**Two things Elk has and `pw.x` does not are in** (P48), chosen from a survey of Elk's
task list against QE 7.5 that `ELK-FEATURES.md` keeps — the four not taken are recorded
there with the validation route each would need, because that is what decides whether a
phase is worth starting. **The effective mass tensor** is `(1/2) d^2 eps_n/dk_a dk_b`, and
the honest construction is **the first derivative by `jvp` and the second by one central
difference of it**: differentiating the Hellmann-Feynman expression again at frozen states
drops the whole `k.p` sum, the Sternheimer solver cannot supply an individual band's
`|dpsi/dk>` (its `P_c` removes the occupied manifold, and the band whose mass is wanted is
usually empty), and rule D4 forbids the eigensolver. That still beats Elk's difference of
*eigenvalues* — six stencil points against twenty-seven, and no fit. Against the vendored
all-electron Elk binary on silicon at `Gamma`: the non-degenerate `Gamma_1` curvature to
**0.02%** and `Gamma_2'` to 0.36% (`m* = 0.170` `m_e`), with the two routes here agreeing
to 1.2e-5 and the tensors isotropic to 2.9e-8 with nothing imposing cubic symmetry.
**The finding is that a stencil must not contain its own centre**: the plane-wave sphere is
rebuilt at every `k`, and a high-symmetry point is exactly where a shell sits on the cutoff
— `Gamma` holds **725** plane waves where every displaced point holds 733, so the centre
eigenvalue is variationally high by a fixed 1.2e-6 Ry and the second difference inherits
`-delta/h^2`, an error that *grows* as the stencil shrinks (measured growing fourfold per
halving). It is the **cutoff and not the pseudopotential** — norm-conserving LDA at the
same `ecutwfc = 30` has the identical 725/733 split and the same cell at 12 has none —
and **Elk has it too**: its own `Gamma_1` drifts 0.8583, 0.8595, 0.8603, 0.8642, 0.8697 as
`deltaem` shrinks, rising to a minimum-error point at its default and then diverging, so
only one of the two codes converges. **Site-resolved `<L>`, `<S>` and `<J>`** are the
second: the projection a projected DOS is made of, contracted with `L` (written in the
*real* harmonic basis by conjugating with `rot_ylm`) and with `sigma` instead of squared.
`pw.x` has `lorbm` — the **cell's** orbital magnetization — and nothing per atom. Validated
by identities rather than by Elk's number, since a muffin-tin expectation value and a
projector one differ by definition: `<L>` is **quenched to 1.7e-16** without spin-orbit
coupling, nickel's is **0.0364767** hbar with it (`|L|/|S| = 0.11665` against an experimental
0.1, `L` parallel to `S`), and driving the moment along `z`, `x` and `y` gives the same
`|<L>|` to 7.3e-11 with nothing imposing that a magnitude is a scalar. Refused by
name: a degenerate multiplet's *per-band* mass (the invariant multiplet sum is reported
instead), a symmetry-reduced k-set for the angular momenta (they are axial vectors; the
whole unshifted grid is the escape), a fully-relativistic **ultrasoft or PAW** dataset
there (`qq_so`'s off-diagonal spin blocks), and a spin spiral for both.

**The piezoelectric tensor is in** (P50), the third thing taken from
`ELK-FEATURES.md` and the first that fails that file's own cheapness filter:
`e_(k)ij = dP_k/d(eps_ij) = d(sigma_ij)/dE_k` is a mixed second derivative, and it is
P24b's construction with one coordinate changed — a Born charge is one `jvp` of the
*force* along the field's response, and this is one `jvp` of the **stress** along the same
response, three of them on top of a dielectric constant that was going to be solved anyway.
The strain leg is *cheaper* than the displacement leg it copies, because
`<psi|S|psi>` is a sum over a sphere of integers and carries no cell, so the multiplier
response `dLambda` — three of P24b's four added terms — has nothing to contribute.
**There is no reference**: `pw.x` computes no piezoelectric tensor (the word occurs once in
the vendored tree, in a citation in a comment in `bp_c_phase.f90`) and Elk's `piezoelt.f90`
finite-differences a Berry-phase polarization over one full ground state per strain. So the
validation is internal and it is four statements: silicon's whole tensor vanishes
(**2.4e-5** C/m² against AlAs's 0.764 from the same code), AlAs comes out exactly `-43m`
with only `e_14 = e_25 = e_36` surviving to **1.7e-14** on a `nosym` run that imposes
nothing, the eight-point wedge reproduces the sixty-four-point closed grid to **4.5e-9**,
and the same mixed derivative contracted the other two ways — `zstar_eu.f90`'s expression
with a strain label, which needs no strain response at all, and the strain response against
the field's bare perturbation — agrees to **6.2e-15** and **1.3e-7**. What anchors the sign
and the field's normalisation is that **the same assembly in the position coordinate is the
Born charge**, which is `ph.x`'s number. **The trap is a factor of two and it is Rydberg's
`e^2`**: `dielec.f90` contracts the *same* field response with a 4 because a susceptibility
is Coulomb-normalised, and a bare mixed derivative takes `zstar_eu.f90`'s 2 — and the wrong
one is exactly zincblende, exactly symmetric, vanishes on silicon and is twice too large.
**Refused by name**: a **polar** crystal, because what the derivative gives is the
*improper* tensor and the proper one differs by `delta_ki P_j - delta_ij P_k`, which needs
`P` itself (those terms vanish identically whenever the two labels they pair differ, so
`e_14` never carries the ambiguity, and they vanish for every component of a class with no
invariant vector — which is what is checked, from the *structure* rather than from a
`nosym` run's symmetry list). **Clamped-ion**, which is also what Elk's task computes; the
internal-strain term that makes it comparable with experiment nearly cancels it for
zincblende, and its one missing ingredient is a two-coordinate frozen functional `E(eps, u)`.
And **ultrasoft or PAW**, which is a gap rather than a missing term: nothing in the assembly
is norm-conserving and the *displacement* leg of it is validated on all three kinds, but
every ultrasoft and PAW case committed here is **centrosymmetric**, whose tensor vanishes
identically — so running one agrees with zero whatever is wrong, and P44 is the reason a
plausible argument about the strain coordinate is not enough on its own. Lifting it is one
non-centrosymmetric ultrasoft dataset.

**Outstanding:** Wyckoff input, the dynamical matrix of an
ultrasoft or PAW *metal*, the strain coordinate's third derivatives on ultrasoft and PAW
(P44 localised what is missing), the *second derivatives* of a spin-polarized system (P45 put
the solve in; the dynamical matrix's and the strain response's assembly are not there) and a
spin-polarized `Z*`, the elastic constants and electrostriction of a **spinor** run (P46 left
that refusal standing: they reach the energy functional directly and their first-order
wavefunctions come from a Sternheimer solve with no spinor form), the force on an atom of a
**spin spiral** (the two components live on different plane-wave spheres, so the nonlocal
term needs the projectors of both — `dE/dq` is what a spiral has instead), the Kubo curvature
of an **ultrasoft or PAW** dataset (P47: the `e_n dS/dk` term is written and unvalidated),
PAW Born charges, `chi^(2)` and the electro-optic tensor (the second-order
response `solve_e2` is, which P35 refuses for), the **non-analytic LO-TO term**
(`rigid.f90`'s `nonanal`, whose two ingredients — `Z*` and `eps` — are both here),
phonons at `q != 0` (the perturbed states live
at `k + q`, so it needs the two-sphere machinery P19 built for the spin spirals, plus
`q2r`/`matdyn` for a dispersion), the **relaxed-ion** piezoelectric constant (P50: `Z*`, the
`Gamma` force constants and the strain response are all here; what is missing is the
internal-strain tensor `d^2E/du d(eps)`, whose two legs are *both* coordinates of the energy
and therefore need a two-coordinate frozen functional), and the rest of P10 (k-axis sharding
and GPU).

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
the smearing and the tetrahedron families implement both. **`occupations = 'fixed'`
implements only the second**, and that is `pw.x`'s rule rather than a gap:
`input.f90:784-800` refuses fixed-occupation LSDA without a `tot_magnetization` and
requires an integer one, so each channel fills `NINT(nelup)` and `NINT(neldw)` bands
(`iweights_only` with `degspin = 1`) and there is no level to search for. Both refusals
are made at input here too, in QE's order. An oxygen atom at `tot_magnetization = 2`
matches `pw.x` to **4.9e-9 Ry**. What the *residual solver* cannot take is a fixed
occupation cutting a **degenerate multiplet** — a Hund's-rule atom, which is most of what
this combination is for: which member the eigensolver returns is arbitrary, so `F` is not
a function of the density and there is nothing for a Newton step to converge on. It is
diagnosed by name; the mixer is unaffected, and a *gapped* fixed LSDA cell agrees between
the two solvers to 5.3e-12 Ry.

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
the operations that need time reversal, and `gradcorr` runs in the local spin frame. `PAW_gcxc_potential` with a magnetization — PAW plus a GGA plus `nspin_mag = 4` — is in
as of P33: the radial local-frame rotation calls into the plane-wave one rather than
restating it, and the rotated channels' multipoles are recomputed by quadrature, because
the rotation runs through `|m|` and is not linear in the stored components.
Forces, the stress and relaxation are implemented for this regime as of P46. What they take
from it is two matrices — `dvan_so` for the nonlocal energy and `qq_so` for the
orthonormality constraint — and one layout: a spinor is one coefficient vector of length
`2 npwx`, so the frozen state's array is `(1, nk, nbnd, 2 npwx)` and the kinetic term reads
`state_kinetic`. `dvan_so` is the **bare** `D`, for the same reason the collinear branch
takes `dion` and not `deeq`: `newd_nc` sandwiches the self-consistent integrals between
`fcoef` and *adds* them, so the split survives one spin index up and taking `deeq_nc` instead
double-counts — which only the energy identity catches, never a finite difference. The
**analytic** transcriptions of `force_us` and `stres_knl` stay refused, and so does
everything above the Sternheimer solver.

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
`jacfwd` of `H(k)` (D2) is registered as `kubo` for a smooth `Omega(k)`, and **as of P47 it
runs on a real crystal too**: what it used to refuse for — `d(vkb)/dk`, and a plane-wave
sphere that changes with `k` — is what P24's `VelocityOperator` already is, one `jvp` of
`H(k)` at a frozen sphere. It is written as band matrix elements between the states an NSCF
produced, never as a dense `H(k)`, and validated against the FHS flux the two ways that
check different things: a plaquette shrunk around one k-point converges onto the pointwise
value to **3.2e-3**, and the whole 24x24 mesh agrees plaquette by plaquette to **1.45e-4**,
improving 65x over a sixfold refinement. Silicon's curvature vanishes **pointwise** to
3.5e-5, which is the check that AlAs's does not. The sum over empty states is truncated and
the truncation is **reported** (`BerryCurvature.truncation`). **Ultrasoft and PAW are
refused by name**: the `e_n dS/dk` term is identically zero for a norm-conserving dataset,
so nothing validated here can see whether its convention is right. Z2 has two independent methods — Wannier-charge-centre flow, which
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
radius and Wolfe line search.

**Variable-cell relaxation is in too** (P29): `calculation = 'vc-relax'`. The cell is nine
more coordinates of the same BFGS, its gradient is `cell_force`'s
`dH/dh = Omega (P I - sigma) h^-T`, and what is minimised is the **enthalpy** — so the
stationary point is `sigma = P I` and a relaxed crystal carries the applied pressure rather
than having no stress. **A vc-relax is two runs and that is what makes it obey the
fixed-setup rule**: `scale_h.f90` re-expresses the *same* G-vectors against the new
reciprocal cell and changes nothing else (`Calculation.at_cell`), so the relaxation is one
run with one setup; then `reset_gvectors` throws it away and runs **one more SCF from
scratch** at the relaxed geometry, which is a second run with a second setup. The gap
between their energies is the Pulay error of the frozen basis and is reported
(`VCRelaxResult.pulay_error`) rather than left to be noticed — on QE's own five-layer
graphite with the whole cell free it is **0.45 Ry**, and the relaxation goes downhill in the
frozen basis and uphill in reality. `treinit_gvecs` rebuilds everything per step and makes
it zero.

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
**Refused** rather than approximated — and the list is shorter than it was, because
ultrasoft and PAW (`dbecsum`, the augmentation charge's own response, `int3`) came in with
P24's own ultrasoft paragraph, metals with P24c and collinear spin with P45: what is left is
noncollinear magnetism (`incdrhoscf_nc`/`set_int3_nc` are a second implementation rather than
a spin axis on this one), DFT+U (`adddvhubscf`), spin spirals, and a **potential-only
meta-GGA**, which P45 made a named refusal — it used to surface as `v_of_rho` asking for
`tau`, which reads like a missing keyword argument and is not.

**Meta-GGA is in scope for the potential-only branch of it** (P30, P31, P32):
`pypresso/xc/mgga.py`. Tran-Blaha (`tb09`) and Becke-Johnson (`bj06`) are potentials, not
energy functionals, so they invert the rule above — nothing is differentiated, the expression
*is* `v_x`, and there is no `E_x` for the total energy to contain. The consequences are
enforced rather than documented: `run_scf` warns that its total is not the value of any
functional it minimised, and every consumer of `forces/energy.py:energy_at` refuses. The SCF
carries a second field, `tau`, which comes from the states rather than the density (three
extra transforms per band, `sum_band.f90`'s meta branch) and is **not mixed**, exactly as
`mix_rho.f90` leaves `kin_r` alone. **Energy-carrying meta-GGAs — TPSS, SCAN, M06L — are not
in**: their potential has a `dE/dtau` piece that acts on the wavefunction through
`h_psi_meta.f90`, and none of that is written; a potential-only functional needs no such term,
which is why this branch and not that one.

**Van der Waals corrections are in scope, and one of the five is implemented** (P27):
`pypresso/vdw/`, behind a name registry that `vdw_corr` selects from. Grimme's **D2** is a
pair potential over the nuclei and nothing else, so it is `pypresso/scf/ewald.py` again with
a different radial function — the energy is written down and the force and the stress are
`jax.grad` of it. The other four are **refused by name**, where `set_vdw_corr` warns and
silently runs with no correction at all: **D3** because its `C6` depends on each atom's
coordination number and so has a derivative of its own, and **Tkatchenko-Scheffler**, **MBD**
and **XDM** because their coefficients are functionals of the self-consistent density, which
puts them inside `v_of_rho` where D2 is outside it.

**Optical spectra and excitons are in scope and implemented** (P37): `pypresso/tddft/`.
This is the one place a **sum over states** earns its keep — an absorption spectrum needs
the frequency axis and needs `chi_0` as a *matrix* over reciprocal lattice vectors, where
everything in `pypresso/response/` produces it as an operator from a Sternheimer solve. The
Dyson equation is then solved with an exchange-correlation kernel from a name registry, and
the one that matters is Sharma, Dewhurst, Sanna and Gross's **bootstrap** (PRL 107, 186401
(2011); Elk's `fxctype = 210`), which is parameter-free, self-consistent with the Dyson
equation it feeds, and divergent as `1/q^2` — which is what binds an electron-hole pair
where ALDA's head and wings vanish identically. **This deliberately enters territory the
line below used to exclude**, and from Elk's side rather than QE's: `TDDFPT/` is a
Liouville-Lanczos solver with RPA and ALDA, has no bootstrap kernel and never forms a Dyson
equation in G space, so there is nothing there to transcribe.

Out of scope until the above works: EXX, real-time propagation and the Liouville-Lanczos
route to a spectrum (`TDDFPT/`), Car-Parrinello (`CPV/`), and everything in `EPW/`, `HP/`,
`GWW/`. The code should nonetheless be shaped so these are additions, not rewrites.

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

## The front door is `Calculator`

`pypresso/calculator.py` (P38). A `Calculator` is a `System` together with its
pseudopotentials, and every workflow, force, stress, response and invariant is a method
on it — `Calculator.from_file("scf.in")`, then `get_scf()`, `get_bands()`,
`get_dielectric_tensor()`. It is what the README, the user guide and new notebooks use,
and `from pypresso import Calculator` is the one import a script needs.

**It is a facade and nothing else.** No physics lives there: every method is a one-line
delegation to the functional entry point, which is unchanged and still the way anything
managing its own state is driven. A `get_*` that grew a computation of its own would be
a second implementation of something already validated against QE, and there is a test
asserting that none has.

Three things about it bind anything added to it:

- **State cannot move onto `System`.** `System` is an `eqx.Module` crossing `jit`/`grad`,
  so a `pseudos` field would change the pytree every compiled path sees and a cached
  result cannot live on a frozen module at all — and `System` does not *have* the
  pseudopotentials, only the file names. `System.calculator()` is a constructor, not a
  place to hang calculations.
- **Nothing mutates.** `with_positions`/`with_cell`/`with_spin` return a *new* calculator
  with an empty cache; the converged state crosses as a `starting_from` seed
  (`starting_state`), never as an answer. A cached result under a moved atom is the
  `test_geometry_invalidation` defect one layer up.
- **The refusals pass through untouched**, and the implicit SCF announces itself. The
  cache is one slot keyed by the options that filled it, because it holds the
  wavefunctions.

A new feature adds a `get_*` method in the same pass that adds its entry point. Shared
options go in `SHARED_OPTIONS` and are forwarded **by named parameter only** — a
`**kwargs` in a signature is not permission to pass everything, since several response
entry points forward theirs to solvers that would raise on `nbnd`.

## The README's feature table

`README.md` carries a table of every implemented feature, with the input variable or entry
point that reaches it and **two tick columns — `QE` and `Elk` — saying whether either
established code computes that quantity at all**. A tick means it is there, `(✓)` with a
numbered note means it is there only partly, and **blank in both columns is the mark of a
quantity neither code has** (the spiral relaxation, the topological invariants, the strain
response, the elastic and electrostriction constants). It is the only place that
distinction is written down for a reader who is not going to read `PLAN.md`, and it is what
tells someone evaluating the code whether it is a reimplementation or an extension.

**The table names quantities, not routines.** Which Fortran file a feature corresponds to,
what is transcribed and what is differentiated, belongs in this file and in `PLAN.md`; the
README says what the physics is and whether the other codes have it.

**The rows are physics, not knobs.** A row is a thing someone would want to compute — total
energies, bands, densities of states, forces, relaxation, magnetism, DFT+U, the invariants —
not an implementation setting underneath one. Smearing is the example: it belongs inside the
row about metals, not in a row of its own. A new feature adds *one* row, named by the
quantity it produces; its variants, schemes and internal terms go in `PLAN.md`.

**Every new feature adds a row, and every removed or renamed one edits its row.** Same
standing requirement as the notebooks: a phase is not finished until its row exists. Two
things make the table go stale silently and both have already happened once, so check them:

- **Each tick is a claim about someone else's source**, not a guess. Before ticking `QE`,
  find the routine in the vendored tree; before ticking `Elk`, find it in the task list of
  `docs/elk_manual.txt` (§5.127) or in `vendor/elk/src/` in the user's `elkpy` checkout;
  before leaving both blank, grep both. "QE probably has this" is how a wrong row gets in,
  and Elk is the easier of the two to get wrong — its `z2*.f90` files are complex-matrix
  helpers and have nothing to do with the Z2 invariant.
- **The entry-point column is a claim about *this* code.** Check the variable is actually
  read (`grep` it in `io/pwin.py` and `system/builder.py`) and the function actually
  exported. A row naming `tprnfor` when nothing parses it, or `UPF v1` when the reader
  refuses anything but v2, is worse than no row — both were in the first draft of the table.

## The user guide

`docs/features.tex` is the user-facing reference — every capability, the
equation behind it, a snippet that runs it, what it was validated against, and
what it refuses. `docs/features.pdf` is built from it with
`xelatex docs/features.tex` (twice, for the table of contents); there is no
markdown copy and none should be added, because two copies drift.

**Every new feature adds an entry, and a phase is not finished until it does.**
Same standing requirement as the README table and the notebooks, and it goes
stale the same way. A feature the code has and the guide does not is a feature
nobody can find.

An entry is four things, and the last two are the ones that get skipped:

- **what it computes**, as an equation where there is one — this is a physics
  document, not an API listing;
- **the entry point**, checked by `grep` rather than remembered;
- **a snippet that has been run.** Checking that a name exists is not enough:
  an audit that only checked `dir()` passed six broken snippets, because
  `run_dos` and `run_pdos` return `(result, states)` tuples, `ProjectedDOS`
  has no `spilling` (it is on the nested `charges`), `run_berry_curvature`
  already *is* the Chern number, and `SpiralRelaxResult` has no `.energy`.
  Execute it;
- **what it refuses**, in the amber box. The refusals are the promise that a run
  which starts is a run whose physics is there, and that promise is only usable
  if its edges are written down.

**The audit that catches drift** is a set difference, not a read-through: list
the workflow, response, force and stress entry points the package exports and
check each appears in the `.tex`. Run it when a phase lands. It found ten
missing at once — including *elastic constants and electrostriction*, a whole
implemented feature with no mention at all.

**Do not re-document standard `pw.x` variables.** `ecutwfc`, `ibrav`, `nbnd`
and the rest mean what they mean in QE and the guide says so once. Document the
knobs that are this code's own — `mbj_c`, `spiral_q`,
`LOCAL_MAGNETIC_FIELDS` — and the ones that gate a feature below.

## Tutorial notebooks

**That rewrite is finished** (`PLAN.md` P49), and what enforces it now is
`tests/unit/test_notebook_conventions.py`: **27 of the 30 are in its `REWRITTEN`
set** and held to the whole skeleton, `JVP_DEBT` is empty, and the three that are
not — `01`, `03` and `17` — are the under-the-hood tier by design, whose internals
are their subject. `REWRITTEN` **only ever grows**, so a new notebook joins it in
the commit that adds it. The shape it enforces is in `notebooks/README.md`, and
P49's six-point checklist is worth reading before touching a notebook: every item
on it is something that sweep got wrong at least once.

`notebooks/` holds worked examples on concrete systems — the readable counterpart to the
test suite. **Every new feature adds a notebook or extends an existing one; a phase is not
finished until its notebook exists.** Demonstrate on the two-atom silicon cell from
`test-suite/pw_scf/scf.in` wherever possible, compare against the committed QE benchmark
whenever the reference contains the quantity, and commit the notebook executed so it reads
without being run.

**A notebook is about the physics, and nothing else.** What the quantity is, the equation
that defines it, what the number means, how it compares with experiment or with Quantum
ESPRESSO. **The implementation is not the subject and must not appear**: no `PLAN.md` phase
numbers, no QE Fortran file names, no transcribed-versus-differentiated tables, no `jvp`,
tangents, frozen spheres, padding or compilation, no catalogue of traps, and no account of
how something was developed or debugged. That material is exactly what `PLAN.md` and the
tests are for, and a reader who wants the physics should not have to wade through it. Two
things survive from that side because they are claims about capability rather than about
code: one sentence saying a derivative is taken of the energy itself rather than derived by
hand, and one sentence where a reference is unusual and the reader would otherwise not trust
the comparison. **No em dashes** anywhere in a notebook.

**That rule binds the code cells, not only the prose, and this is the sentence that was
missing** (P49). Bounding only the prose is why the notebooks drifted into validation
reports: the project's validation instinct moved into the code, where the rule did not
reach, and 2800 code lines across 29 notebooks is what came of it. An identity check
looped over four pseudopotentials, a derivative checked against a closed form on a random
matrix, a hand-built linear solve with a probe potential: each is the test suite's job
being done in public. They belong in `tests/`, and the notebook's footer names the file
they went to. **Where a `get_*` method exists, the notebook uses it** rather than building
the same quantity from internals and remarking afterwards that the method also exists.
`notebooks/README.md` carries the cell-by-cell shape and the budgets, and indexes the set
**by the property a reader wants to compute** rather than by the order the code gained it.

**A notebook is five minutes long, and it has a figure.** Header saying what this computes
and the headline number against QE; the shortest code that runs it; **one plot that shows
the physics**, a band structure wherever the feature shows in bands; one comparison table;
at most one "how it works" cell for the single best idea, and it is a *physical* idea. About
eight code cells. The derivations, the trap catalogue and the per-case validation tables
belong in `PLAN.md` and in the tests, not here, and an expensive sweep is measured once
offline and *quoted* rather than run. Each notebook also has a `.md` export committed beside it — raw `.ipynb` is unreadable in a
plain editor or a diff — regenerated together with the notebook by `tools/export_notebooks.sh`.
`notebooks/README.md` holds the index and the full conventions.

**Ten minutes is the hard ceiling on executing one**, and it is a ceiling rather than a
target — the five minutes above is still what to aim for. A notebook is re-executed every
time the code under it changes, by `tools/export_notebooks.sh` and by anyone checking that
it still reads true, so its runtime is paid over and over by people who are not doing
physics at the time. **Time it before committing it**, the same way a peak working set is
sized before it is landed:

```bash
time jupyter nbconvert --to notebook --execute --inplace notebooks/<n>.ipynb
```

If it does not fit, the cell to cut is the *sweep*, not the physics: measure the expensive
series once offline and quote its numbers in prose, which is the same rule as the
per-case validation tables and for the same reason. A figure that needs ten SCF runs to
draw is a figure whose points belong in a test.

**`tools/export_notebooks.sh` times each notebook as it re-executes it and exits
non-zero over the ceiling**, so the set stays measured without anyone remembering
to measure it. Nothing is over: the slowest is `27` at 178 s, the median is under
30, and the full table is in `notebooks/README.md`. The one notebook that ever was
over is worth reading as a case study. `08_spin_orbit_coupling` took about **25 minutes**; it takes **174 s** (P49).
Two things were true at once and only one of them was the expensive part. The bismuthene
section was blamed, and the figure it draws — a Dirac point gapped by nothing but the
coupling — is the best physics in the notebook and was never the problem: as notebook 08 runs it,
at the default band count, that spinor SCF is **48 s**. The 281 s recorded here is the
same cell under `notebooks/10`, which asks for thirty spinor bands to build a curvature
from, and generalising it to "a spinor SCF on a slab" is what made the section look
unfixable. What actually cost the
time was a five-run finite-difference sweep of the spinor force and a two-run
noncollinear-equals-collinear identity, **both of them already in the test suite**
(`test_the_force_is_a_finite_difference_of_the_frozen_energy`,
`test_spinors_reproduce_the_collinear_answer`). The lesson is the one P49 is built on: a
notebook that is too slow is usually a notebook doing the tests' job, and the cell to cut
is the sweep rather than the physics. Every notebook's time is in `notebooks/README.md`.

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
| Meta-GGA (potential-only) | no QE counterpart to transcribe — `XClib/dft_setting_routines.f90` maps `tb09` to libxc 208; `PW/src/sum_band.f90` (the `kin_r` branch and its `sym_rho`), `PW/src/v_of_rho.f90` (`v_xc_meta`), `PW/src/potinit.f90` (the Thomas-Fermi `tau` guess), `PW/src/setup.f90` (what it refuses) | the functional itself follows libxc's own definition (`maple/mgga_vxc/mgga_x_tb09.mpl`, `maple/mgga_exc/mgga_x_br89.mpl`, `src/mgga_x_br89.c`), because QE has no native implementation. **QE passes a zero Laplacian and never sets `c`**, so its `tb09` is BJ06; both are here separately. `tau` is symmetrised — `sum_band` does it too, and skipping it is worth 0.47 eV in the eigenvalues |
| Van der Waals dispersion | `Modules/mm_dispersion.f90` (`energy_london`, `force_london`, `stres_london`), `Modules/set_vdw_corr.f90`, `Modules/rgen.f90`, `upflib/atomic_number.f90` | the energy is written down (`pypresso/vdw/grimme.py`) and the force and stress are `jax.grad` of it; QE's two expressions are transcribed as the cross-check. `rgen`'s **fold** of the pair separation into the cell is kept, and it is what lets one neighbour list serve every geometry |
| Ewald / local potential | `PW/src/ewald.f90`, `setlocal.f90` | the ion-ion sum and `V_loc(G)`; the Ewald neighbour list is fixed for the *cell*, not the geometry, so it survives a relaxation |
| Forces | `PW/src/forces.f90`, `force_lc.f90`, `force_cc.f90`, `force_ew.f90`, `force_us.f90`, `addusforce.f90`, `force_corr.f90`, `symme.f90` (`symvector`) | the default is `jax.grad` of the energy at frozen wavefunctions (`forces/energy.py`); the Fortran expressions are transcribed as a cross-check. `gradcorr` is called from **inside** `v_xc`, so `force_cc` needs it |
| Structural relaxation | `Modules/bfgs_module.f90`, `PW/src/move_ions.f90`, `run_pwscf.f90`, `update_pot.f90`, `checkallsym.f90` | BFGS in crystal coordinates with the cell metric; the setup (FFT grid, symmetry, k-points) is done **once** and only checked afterwards |
| Stress | `PW/src/stress.f90`, `stres_knl.f90`, `stres_har.f90`, `stres_loc.f90`, `stres_cc.f90`, `stres_gradcorr.f90`, `stres_ewa.f90`, `symme.f90` (`symmatrix`) | the default is `jax.grad` of the energy with respect to a strain at frozen wavefunctions (`stress/energy.py`, through `Calculation.at_strain`); the Fortran expressions are transcribed as a cross-check. `stres_us`/`addusstress` are **not** transcribed, so the analytic route offers terms and no total. `ylmr2`'s `atan2` parameterisation is singular on the `z` axis and only a *cell* derivative reaches it |
| Magnetic symmetry | `PW/src/symm_base.f90` (`sgam_at_mag`), `symme.f90` (`sym_rho`'s `nspin = 4` branch), `PW/src/irrek.f90` | the magnetization is an **axial** vector, so its rotation carries `det(R)` and a further sign for an operation that is a symmetry only with time reversal; `irreducible_BZ` completes an explicit k-list from the lattice's wedge to the crystal's, and runs for every SCF |
| Fields and constraints | `PW/src/add_bfield.f90`, `make_pointlists.f90`, `get_locals.f90`, `report_mag.f90`, `PW/src/input.f90` (`i_cons`) | the penalty's *energy* is written here and its potential comes from `jax.grad`; QE's five expressions are transcribed as the cross-check. Elk's counterparts: manual §5.2/§5.12/§5.104, `src/bfieldfsm.f90` |
| Spin spirals | no QE counterpart — Elk's `src/gengkqvec.f90`, `init0.f90`, `findsymlat.f90`, manual §5.146 | up at `k + q/2`, down at `k - q/2`, each with its own `G+k` set; one basis call on the concatenated list gives both a common `npwx` |
| Spiral relaxation | no QE counterpart — `Modules/bfgs_module.f90` reused with the *reciprocal* cell as its lattice | `dE/dq` is `jax.grad` of the energy at frozen wavefunctions and a frozen sphere (`forces/spiral.py`); only the kinetic and nonlocal terms carry `q` |
| Piezoelectric tensor | no QE counterpart — Elk's `src/piezoelt.f90` and `genstrain.f90`, manual task 380 | Elk runs one ground state per strain and finite-differences the Berry-phase polarization; here it is one `jvp` of the stress along the field's response (`pypresso/response/piezo.py`), so nothing is transcribed but the *check* — `zstar_eu.f90`'s contraction with a strain label where it has a displacement. `genstrain` symmetrises each candidate strain over the crystal's group, so on a cubic crystal the only strain it keeps is the isotropic one |
| Berry phase / topology | `PW/src/bp_c_phase.f90` (the ultrasoft `q_ij(b)` and the k-string overlaps), `Modules/bfgs`-free | the invariants themselves have no QE counterpart to transcribe — `pypresso/topology/` follows Fukui-Hatsugai-Suzuki, Yu-Qi-Bernevig-Fang-Dai and Fu-Kane, with `bp_c_phase.f90` as the reference for how the augmentation charge enters an overlap between two different k-points |
| Velocity / position operator | `PW/src/commutator_Hx_psi.f90`, `PP/src/` Berry-phase code | QE hand-codes `[H,r]` term by term; here it is one `jvp` of `H(k)` at a frozen sphere (`response/velocity.py`), since `dH/dk_a = i[H, r_a]` in the periodic gauge. The overlap carries a velocity too, so a band velocity is `<psi|dH/dk - eps dS/dk|psi>` |
| Linear response / DFPT | `LR_Modules/cgsolve_all.f90`, `ch_psi_all.f90`, `orthogonalize.f90`, `h_prec.f90`, `setup_alpha_pv.f90`, `incdrhoscf.f90`, `symdvscf.f90`; `PHonon/PH/solve_e.f90`, `dvpsi_e.f90`, `dvqpsi_us.f90`, `dielec.f90`, `zstar_eu.f90` | the linear solve, the projector and the assembly are transcribed; the *perturbations* are not. `dv_of_drho` is one `jvp` of `v_of_rho` (which already drops the `G = 0` Hartree term), the E-field's commutator is the velocity operator, and `dvqpsi_us` is one `jvp` through `at_positions`. **A response on a reduced k-set is a polar vector field and must be symmetrised as one** |
| TDDFT: `chi_0`, the Dyson equation, the bootstrap kernel | no QE counterpart — Elk's `src/tddftlr.f90` (the driver and the fixed point), `genvchi0.f90` (Adler-Wiser, the `t3hw` head/wing layout), `genvfxc.f90` (the kernels), `init3.f90` (`ngrf`, and `wrf(1) = 0 + i swidth`), `getpmat.f90` (the scissors renormalisation), manual `fxctype`/`gmaxrf`/`swidth` | the head is the one line **not** to transcribe: Elk reads momentum matrix elements, which is right in LAPW and wrong with a nonlocal pseudopotential, so `response/velocity.py`'s `dH/dk` takes their place. `eps_M` is the inverse of the **3x3 head** of `eps^-1`, not the head of the inverse — Elk writes both, thirty lines apart, and the wrong one is 9% too large and otherwise perfect |
| Non-linear response (Raman) | `PHonon/PH/raman.f90`, `raman_mat.f90`, `el_opt.f90`, `dhdrhopsi.f90`, `dvpsi_e2.f90`, `solve_e2.f90`, `d2mxc.f90`, `write_ramtns.f90`, `symme.f90` (`symtensor3`, `symmatrix3`) | none of it is transcribed: `d(eps)/d(tau)` is one `jvp` of the second-order energy P26 already differentiates, and `d2mxc`'s third derivative of `E_xc` is a `jvp` of the kernel rather than a parameterisation, so a GGA works where `phq_setup.f90` stops. **The vendored 7.5 build's `lraman`/`elop` branch does not reproduce QE's own v6.0 example and fails its own internal check** -- use it as evidence, not as a reference. `dynmat_sub.f90`'s `RamanIR` (reached by `dynmat.x`) is the exception and *is* a reference: it is post-processing, reads `dchi_dtau` off a file, and shares nothing with that branch. `symtensor3`/`symmatrix3` are implemented (P36), at any rank |
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
  GPU wants — so it is a **dial** (`pypresso/batching.py`), defaulting to QE's end of it
  on a CPU and to the batch on an accelerator, not a fixed choice. Rule R6 (k leading) is what keeps both available.
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

**A test file that sweeps many cells is a memory liability, and splitting the file is not
the fix.** Every such file has been killed on this machine eventually — `test_ten_site.py`
in P28b, `test_spinor_forces.py` in P46, and the second one did not inherit the first one's
cure. The mechanism is accumulation, not any one peak: cells that share no shape each
compile the whole SCF (and, for a derivative, the gradient) stack afresh and **XLA keeps
every executable for the life of the process**, while an unbounded `lru_cache` of converged
states holds their wavefunctions beside it. Both grow monotonically through the file.

Two bounds, and they belong on any file that runs more than about three distinct cells:

- **`jax.clear_caches()` in an autouse fixture**, after the `yield`. The results stay
  cached; only the compiled code is dropped, which trades recompilation for a peak the
  machine can afford.
- **`lru_cache(maxsize=2)` on the converged-state helper**, never `maxsize=None` — 2 is
  what a comparison between two cells needs and is the largest that is not a leak.

**Splitting the file only pays off under one of the three runners, which is why it is the
weaker lever.** `tools/run_regression.sh` invokes pytest once per file, so a file boundary
there *is* a process boundary and splitting does cap the peak. `tools/test-fast.sh` and a
plain `pytest -m slow` run everything in **one** process, where splitting changes nothing at
all. The two bounds above work under all three, so they are the rule and splitting is a
judgement call about what belongs together.

What neither bound touches is the peak *inside* one test, which is a real cost to be sized
in advance rather than discovered: the backward pass of an ultrasoft or PAW derivative
carries the augmentation table `Q_ij(G)` — `nh^2 x ngm` per atom, and `nh` is in the
twenties for a fully-relativistic dataset. On a **slab** that is tens of GB and is why a
bismuthene spinor force does not run here at all (P46), while the same physics on a small
bulk cell runs in 33 seconds.

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
tools/test-fast.sh                     # THE GATE: everything not marked slow, ~4.5 min
python3 -m pytest -m slow              # the other 588, over two hours
tools/run_regression.sh                # the same slow set, resumably, one file at a time
python3 -m pytest tests/unit/test_qeref.py::test_scf_silicon   # a single test
python3 -m pypresso.cli inspect <qe-output>   # summarise what the parser reads
tools/export_notebooks.sh                     # re-execute notebooks + refresh .md exports
```

**The suite is two groups and `slow` is the line.** `tools/test-fast.sh` is
`pytest -m "not slow"`: **1634 tests in 4.5 minutes**, and it is what runs before
a push. The slow set is 588 tests and **over two hours** — it runs when it is
asked for, not on every change. The split cuts across `unit` and `regression`
both, because it is about cost and not about kind: a cheap regression case
against a two-atom reference is in the gate, and an expensive unit test is not.

**The slow set is not optional, it is just not per-push.** Run it before a
release, after touching anything in the SCF, the eigensolver or the response
stack, and whenever a number in this file changes. It is two hours precisely
because it is the part that catches what the gate cannot, and the one time it
was run end to end it found **three phases' claims had drifted** — P29's stale
refusal list and its broken BFGS metric, P36's 8.7e-14 wedge agreement, and two
notebooks whose committed outputs no longer matched their code (`PLAN.md` P38).
`tools/run_regression.sh` exists for running it in pieces: one pytest invocation
per file, a durable summary line each, and a file already in the summary is
skipped, so an interrupted run resumes instead of restarting.

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
- **How many k-points are in flight is `pypresso/batching.py`'s dial, and its default
  follows the platform** — QE's loop on a CPU, one k-point at a time as `c_bands.f90` and
  `sum_band.f90` do it, and the whole axis at once on an accelerator, where the cache
  argument behind that loop does not exist and inheriting it gives up 4.5x. The band dial
  moves with it, never separately: `k=all, b=1` is measured to be worse than either end.
  `k_batch`
  reaches every entry point (`run_scf`, `run_bands`, `run_nscf`, `run_dos`, `Calculation`,
  `PYPRESSO_K_BATCH`), `None` asks for one `vmap` over the whole axis, and the chunked form
  is a `lax.map`/`lax.scan` so it stays compiled once and differentiable. Anything new that
  walks the k axis goes through `map_k`/`sum_k` rather than calling `vmap` itself; the
  chunk size must never be visible in a result beyond round-off.
- Use `donate_argnums` for the large wavefunction and density buffers.

`PLAN.md` §1 and §5 hold the full reasoning and the rest of the GPU notes.
