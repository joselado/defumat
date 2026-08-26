# pypresso

A plane-wave density-functional theory code written in Python, which reads
Quantum ESPRESSO's input files and reproduces its numbers.

It is a reimplementation of `pw.x`, not a wrapper: there is no Fortran
underneath. You give it the same input file you would give Quantum ESPRESSO and
it runs the calculation itself.

```
total energy   QE  -63.36038036 Ry
          pypresso  -63.36038036 Ry
```

That is an eight-atom silicon cell, agreeing to 3.5e-9 Ry. The agreement is
checked automatically, term by term, against Quantum ESPRESSO's own reference
outputs for around a hundred cases.

## What it can do today

The right-hand column says where each feature comes from. **"`pw.x`"** means
Quantum ESPRESSO has it and this reproduces its numbers against a reference
output; **"new"** means it has no counterpart in `pw.x` at all, and is validated
some other way — against Elk, against a supercell calculation of the same
physics, or against an identity the answer has to satisfy. In between are the
rows where the *quantity* is Quantum ESPRESSO's and the *route to it* is not.

The middle column is the input-file variable that controls the feature, where
there is one — it means what it means in a `pw.x` input — and the Python entry
point where there is not.

| Feature | How to ask for it | In Quantum ESPRESSO? |
|---|---|---|
| **Total energies**, self-consistently, broken down term by term — insulators and metals alike | `calculation = 'scf'` | `pw.x` |
| **Band structures** along a path through the Brillouin zone | `run_bands` | `pw.x` + `bands.x` |
| **Densities of states**, by smearing or by tetrahedra | `run_dos`, `pypresso dos` | `pw.x` + `dos.x` |
| **Projected densities of states** — resolved by atom, by `l` and by `m`, with Löwdin charges and the spilling parameter | `run_pdos`, `pypresso pdos` | `pw.x` + `projwfc.x` |
| **Forces on the atoms** | `compute_forces` | `pw.x` — but by differentiating the energy, with QE's hand-derived terms kept beside them as a cross-check |
| **Structural relaxation** — BFGS with QE's trust radius and line search | `calculation = 'relax'`, `pypresso relax` | `pw.x` |
| **Variable-cell relaxation** — the cell and the atoms relaxed together, at an applied pressure | `calculation = 'vc-relax'`, `run_vc_relax` | `pw.x` |
| **Stress tensor and pressure**, in Ry/bohr³ and kbar | `tstress = .true.`, `compute_stress`, `pypresso stress` | `pw.x` — but from one strain derivative of the energy, with QE's own term-by-term expressions transcribed beside it as a cross-check |
| **Magnetism**, collinear, with one Fermi level or two | `nspin = 2`, `tot_magnetization` | `pw.x` |
| **Magnetism as a vector** — noncollinear, with the magnetic symmetry group | `noncolin` | `pw.x` |
| **Spin-orbit coupling**, two-component spinors and `j`-resolved projectors | `lspinorb` | `pw.x` |
| **Magnetic fields and constrained moments** — all four of QE's schemes | `B_field`, `constrained_magnetization` | `pw.x` |
| **Magnetic fields inside one atom's sphere**, and a field that fades away | `LOCAL_MAGNETIC_FIELDS` card, `reducebf`, `constrained_magnetization = 'fsm'` | new — Elk's `bfcmt`, `reducebf` and `bfieldfsm.f90`; `pw.x` has no counterpart |
| **DFT+U** — Dudarev's functional with `U`, `J0`, `alpha`, `beta` | `HUBBARD` card, `run_scf(starting_ns=...)` | `pw.x` — plus a custom starting occupation matrix, where `pw.x` offers only `starting_ns_eigenvalue` |
| **Spin spirals** at any wavevector, without a supercell | `spiral_q`, `pypresso spiral` | new — Elk has it, `pw.x` does not |
| **Relaxing the spiral wavevector** down `dE/dq` to the ground-state pitch | `relax_spiral_q` | new |
| **Berry curvature and Chern numbers**, the latter exact integers on any mesh | `run_berry_curvature` | new — QE has the Berry *phase* (`bp_c_phase`), not the curvature |
| **Z2 invariants** in 2D and 3D, by Wannier charge centres *and* by parities | `run_z2`, `run_z2_3d` | new |
| **Continuing one run from another across a change of spin regime** — a converged non-magnetic density as the starting point of a magnetic run, a collinear one of a noncollinear run, spin-orbit coupling switched on | `run_scf(starting_from=...)`, `System.with_spin` | partly `pw.x` — `startingpot = 'file'` reads a density whose `nspin` differs but *zero-fills* the missing spin components, so 1 → 2 starts unpolarized and converges back; `nc_magnetization_from_lsda` rotates a collinear moment onto `angle1(1)` only inside the force-theorem path |
| **Reaching self-consistency** — Anderson/Broyden mixing, Kerker preconditioning, or solving the residual with its own Jacobian | `mixing_mode` in `&electrons`, `run_scf(scf_solver=...)` | `pw.x` has the mixing and `mixing_mode = 'TF'`; the residual solver is new, and it reaches SCF solutions no mixer does -- a genuine saddle in the DFT+U case, and a metastable root outside its own basin in the magnetic-metal one |
| **Band velocities** `d(eps)/dk`, with the nonlocal pseudopotential's own contribution — norm-conserving, ultrasoft and PAW | `band_velocities`, `VelocityOperator` | partly — `fermi_velocity.x` finite-differences eigenvalues across the k-grid and reports only the magnitude, for a Fermi-surface plot; the *operator*, which QE hand-codes as `[H, r]` in `commutator_Hx_psi.f90` and uses only inside `ph.x`, is here one `jvp` of `H(k)` |
| **Dielectric constant** `epsilon_infinity` — insulators, norm-conserving, ultrasoft and PAW — **and Born effective charges** (norm-conserving and ultrasoft; PAW refused by name) | `dielectric_tensor` | `ph.x` with `epsil = .true.` — but the perturbation, the screening kernel and the bare displacement term all come from differentiating here, where DFPT derives each by hand. (`epsilon.x` is a different quantity: a sum over states at the RPA level with local-field effects neglected.) |
| **Phonons at `Gamma`** — the force constants and their frequencies, insulators **and metals**, norm-conserving | `dynamical_matrix` | `ph.x` — but `dynmat0`/`d2ionq` (the frozen second derivative) and `drhodv` (the electronic response) are here two halves of one `jvp` of the gradient that already gives the force, so neither is derived a second time. A metal splits that `jvp` in two rather than adding a routine: the frozen Hessian keeps `wg` and the electronic half takes `wk`, because its `dpsi` already carries the occupation. `q != 0` and the dispersion are not in |
| **Elastic constants** `C_ijkl` and the compliance and bulk modulus that follow, clamped-ion, insulators, norm-conserving | `elastic_constants` | **new** — nothing in the vendored tree computes them (`grep` finds "elastic" only in NEB's elastic band). They are the stress differentiated once more along a strain response, so the assembly is `phonon`'s with the cell in place of the atoms. The internal-strain relaxation that turns clamped-ion into relaxed-ion is not in, so `C_44` is the one to read with care |
| **Electrostriction coefficients** `m`, `q`, `M` and `Q` — the quadratic electromechanical coupling — clamped-ion, insulators, norm-conserving | `electrostriction` | **new** — `pw.x` and `ph.x` have no counterpart. `d(chi)/d(strain)` is a *third* derivative of the energy, obtained from one `jvp` of the second-order energy at frozen first-order wavefunctions (the 2n+1 theorem), where the published route is a sweep of re-converged calculations |
| **Raman tensors** `d(eps)/d(tau)` — the derivative of the dielectric tensor with respect to an atomic coordinate, insulators, norm-conserving | `raman_tensors` | partly `ph.x` (`lraman = .true.`), and both halves of that are the point. It is one `jvp` of the *same* second-order energy P26 differentiates along a strain, so the phase is an assembly of tangents that already existed — and `phq_setup.f90` stops on `'third order derivatives not implemented with GGA'` because QE's third derivative of `E_xc` is a hand-coded Perdew-Zunger parameterisation (`d2mxc.f90`), where here it is one more `jvp` of the kernel and any functional works. The vendored `ph.x` 7.5 does not reproduce its own committed v6.0 example and fails its own internal check, so the reference is a finite difference of `eps` over re-converged displaced cells (1.0e-5). A **symmetry-reduced k-set** works, through the rank-3 average of the row below: the eight-point wedge of AlAs reproduces the sixty-four-point closed grid to 8.7e-14. **`chi^(2)` and the electro-optic tensor are refused**: the field enters only through the source term, so the `<u_i|r_k|u_j>` term of the 2n+1 expression — 42% of the answer, measured on its displacement counterpart — has nothing to build it from |
| **Raman and infrared spectra** — the per-mode activities, depolarisation ratios and electronic polarizability at `Gamma`, insulators, norm-conserving | `vibrational_spectrum` | `dynmat.x` (`LR_Modules/dynmat_sub.f90`'s `RamanIR`), and it is the one reference above second order that still works — it is pure post-processing and shares nothing with the `lraman` branch that has regressed, so `pypresso.io.dynmat` writes the file `ph.x` would have written and the vendored binary is run on our tensors: every digit either code prints. Silicon's `T_2g` comes out at 519.2 cm⁻¹ against an experimental 520, Raman-active and infrared-silent. The **non-analytic** LO-TO term (`nonanal`) is not in, so an optical triplet is unsplit. A degenerate multiplet is comparable only as a sum — the eigensolver's basis inside one is arbitrary and the per-mode depolarisation ratio is not invariant under it |
| **Van der Waals dispersion** — Grimme's D2 pair correction, in the energy, the forces, the stress and the elastic constants | `vdw_corr = 'grimme-d2'`, `london_s6`, `london_rcut`, `london_c6`, `london_rvdw` | `pw.x` (`Modules/mm_dispersion.f90`) — but the force and the stress are `grad` of the one pair sum in the two coordinates, with QE's `force_london` and `stres_london` transcribed beside them as the cross-check. D3, Tkatchenko-Scheffler, MBD and XDM are refused by name where `set_vdw_corr` warns and silently runs without one |
| **Band gaps from the Tran-Blaha potential** (mBJ) — the modified Becke-Johnson meta-GGA, on norm-conserving **and PAW** datasets, unpolarized, collinear, and noncollinear **with spin-orbit coupling** | `input_dft = 'tb09'` (or `'bj06'`), `mbj_c` | partly `pw.x` — and the gap between "partly" and "yes" is the point. QE reaches TB09 only by linking libxc, and then **passes a zero Laplacian** (`XClib/xc_wrapper_mgga.f90` calls its Laplacian argument "not used in QE") and **never sets `c`**, so it hands libxc the default `c = 1`. Both are ingredients of the functional: `input_dft = 'tb09'` in `pw.x` is Becke-Johnson without a Laplacian. Here the Laplacian is `-G²ρ(G)`, one transform, and `c` is Tran and Blaha's cell average — with `mbj_c` to impose it, which `pw.x` has no variable for. **The potential is not the derivative of an energy**, so the reported total is not variational and forces, stress, phonons and response are refused. Two combinations `pw.x` refuses outright work here: PAW (its one-centre `tau` comes from the partial waves, and it is what recovers the all-electron `c` = 1.11 that a pseudised core misses) and noncollinear magnetism with spin-orbit coupling. Plain ultrasoft is refused — no partial waves to reconstruct `tau` from |
| **Pseudopotentials**: norm-conserving, ultrasoft and PAW (UPF v2) | `ATOMIC_SPECIES` | `pw.x` |
| **Functionals**: LDA and GGA — Perdew-Zunger, Perdew-Wang, PBE, revPBE, PBEsol | `input_dft`, or the UPF header | `pw.x` |

The variants under each row — which smearing or tetrahedron method fixes the
occupations, which projectors DFT+U uses, which of the four constraint
schemes, which fixed-spin-moment update (`fsm_update`) — are chosen with the
same input variables as in `pw.x` where it has them, and `PLAN.md` lists them
phase by phase.

Not yet: phonons away from `Gamma` — the ones *at* `Gamma` are
in the table above, for insulators **and metals** on norm-conserving datasets; an ultrasoft
dataset's is refused by name, with the measurement behind the refusal in `PLAN.md`.
`K_POINTS gamma` runs, but at an explicit k = 0 with the
full G sphere — the half-sphere storage the gamma-point trick exists for is
generated and not consumed, so the answer is the same and the cost is twice the
plane waves, and the run says so. A functional or a combination that is not
implemented is refused with an error naming what *is*, rather than quietly
replaced by something that is.

If your calculation needs any of those, use Quantum ESPRESSO — this is not a
replacement for it, and on anything large it will be slower (about two to four
times, running on one core).

**Full feature reference:** [`docs/features.md`](docs/features.md), or
[`docs/features.pdf`](docs/features.pdf) if you want the equations typeset —
every capability, the equations behind it, a snippet that runs it, what it was
validated against, and what it refuses. The table above is the summary; that is
the detail. Rebuild the PDF with `tools/build_features_pdf.sh`.

## Installing

```bash
git clone https://github.com/joselado/pypresso
cd pypresso
pip install -e .
```

Python 3.10 or newer. The dependencies are JAX, NumPy, SciPy, Numba and equinox,
and `pip` will fetch them.

## A first calculation

Silicon, from the input file in `benchmarks/`:

```python
from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.scf import run_scf
from pypresso.system import build_system

system = build_system(read_pw_input("benchmarks/si-1k.in"))
pseudos = tuple(read_upf("tests/data/pseudo/" + s.pseudo_file)
                for s in system.structure.species)

result = run_scf(system, pseudos)

print(f"converged in {result.iterations} iterations")
print(f"total energy   {result.total_energy:.8f} Ry")
for name, value in result.energy_terms.items():
    print(f"  {name:<13} {value:>15.8f} Ry")
```

```
converged in 5 iterations
total energy   -15.25444941 Ry
  one-electron       5.26833739 Ry
  hartree            1.26305386 Ry
  xc                -4.88608209 Ry
  ewald            -16.89975858 Ry
```

`benchmarks/si-1k.in` is an ordinary `pw.x` input file. So is anything else you
point `read_pw_input` at — the `&control`, `&system` and `&electrons` namelists,
`ATOMIC_SPECIES`, `ATOMIC_POSITIONS` and `K_POINTS` cards all mean what they mean
in Quantum ESPRESSO, and `conv_thr` is compared against the same quantity.

## A band structure

Carrying on from the density that SCF converged:

```python
from pypresso.system.kpoints import KPoints
from pypresso.workflows import run_bands

path = KPoints.band_path(
    [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],   # L - Gamma - X
    [20, 20, 1], system.cell, crystal=False,
)
bands = run_bands(system, pseudos, result.density, kpoints=path, nbnd=8)

print(f"indirect gap   {bands.gap(8):.3f} eV")
```

`bands.eigenvalues_ev` is `(k-points, bands)` in eV and `bands.path_length` is
the x-axis for a plot. (The gap comes out small because LDA underestimates
gaps — that is the functional, not the code; Quantum ESPRESSO gives the same
answer, and so does PBE.)

## Examples

The `notebooks/` directory is the place to start. Each one is a worked
calculation on silicon with its output already in it, so they can be read
without being run, and each has a plain-text `.md` version beside it.

| | |
|---|---|
| [`01_silicon_setup`](notebooks/01_silicon_setup.ipynb) | Reading an input file, the crystal, k-points, and the plane-wave basis |
| [`02_silicon_scf_and_bands`](notebooks/02_silicon_scf_and_bands.ipynb) | What is inside a pseudopotential, the SCF, the bonding charge, and the band structure |
| [`03_eigensolver_and_performance`](notebooks/03_eigensolver_and_performance.ipynb) | How the calculation is made fast, and how it compares to Quantum ESPRESSO |
| [`04_ultrasoft_and_paw`](notebooks/04_ultrasoft_and_paw.ipynb) | Ultrasoft and PAW pseudopotentials: two grids, the augmentation charge, and PAW's one-centre terms |
| [`05_gradient_corrections`](notebooks/05_gradient_corrections.ipynb) | PBE and its relatives: what a gradient correction adds to the potential, on the grid and inside a PAW sphere |
| [`06_density_of_states`](notebooks/06_density_of_states.ipynb) | Smearing against tetrahedra, silicon's gap as the thing that separates them, and nickel's spin-resolved DOS |
| [`07_spin_polarization`](notebooks/07_spin_polarization.ipynb) | LSDA: exchange splitting, nickel's magnetic moment, and constraining the magnetization |
| [`08_spin_orbit_coupling`](notebooks/08_spin_orbit_coupling.ipynb) | Spinors, `j`-resolved projectors, platinum's 5d splitting, and a quantum spin Hall insulator |
| [`09_forces_and_relaxation`](notebooks/09_forces_and_relaxation.ipynb) | Forces as one gradient of the energy, checked against Quantum ESPRESSO term by term, and a structure relaxing back onto its lattice site |
| [`10_topological_invariants`](notebooks/10_topological_invariants.ipynb) | Berry curvature from one overlap rather than a derivative, Chern numbers that are exact integers, and Z2 by two independent routes |
| [`11_noncollinear_magnetism_and_fields`](notebooks/11_noncollinear_magnetism_and_fields.ipynb) | Magnetism as a vector, bcc iron against Quantum ESPRESSO, and magnetic fields and constrained moments |
| [`12_spin_spirals`](notebooks/12_spin_spirals.ipynb) | Two plane-wave spheres instead of one, three identities that validate them against calculations that are not spirals, and an `E(q)` magnon dispersion |
| [`13_dft_plus_u`](notebooks/13_dft_plus_u.ipynb) | The Hubbard correction as a penalty on fractional occupation, nickel four ways, antiferromagnetic FeO, and `force_hub` falling out of the gradient |
| [`14_spiral_relaxation`](notebooks/14_spiral_relaxation.ipynb) | `dE/dq`: which terms of the energy a spiral's wavevector touches, and a BFGS walking a hydrogen chain to its ground-state pitch |
| [`15_stress`](notebooks/15_stress.ipynb) | The stress as the strain derivative of the energy, silicon's equation of state, and the pressure against `-dE/dV` |
| [`16_projected_density_of_states`](notebooks/16_projected_density_of_states.ipynb) | `<phi|S|psi>` on Löwdin-orthogonalised orbitals, silicon's `s` and `p` densities of state against `projwfc.x`, and the same weights as fat bands |
| [`17_reaching_self_consistency`](notebooks/17_reaching_self_consistency.ipynb) | Kerker preconditioning, the SCF as a root-find, and the magnetic solutions no mixer reaches |
| [`19_linear_response`](notebooks/19_linear_response.ipynb) | The velocity operator from one `jvp`, the Sternheimer equation instead of a sum over states, silicon's dielectric constant against `ph.x`, and the Born charges as the mixed second derivative `dF/dE` |
| [`18_continuing_a_calculation`](notebooks/18_continuing_a_calculation.ipynb) | Starting one run from another's converged state across a change of spin regime: iron's moment rotated onto `x` in one iteration, and spin-orbit coupling switched on |
| [`20_phonons`](notebooks/20_phonons.ipynb) | The force constants as one more derivative of the gradient that already gives the force, and silicon's optical mode at Gamma against `ph.x` |
| [`21_electrostriction`](notebooks/21_electrostriction.ipynb) | Differentiating a *response*: `d(eps)/d(strain)` as a third derivative of the energy, the elastic constants that come with it, and the elasto-optic tensor |
| [`22_van_der_waals`](notebooks/22_van_der_waals.ipynb) | Grimme's D2 dispersion: a pair sum over the nuclei that never touches the density, and bilayer graphene binding where PBE alone has no minimum |
| [`23_variable_cell_relaxation`](notebooks/23_variable_cell_relaxation.ipynb) | The cell as nine more coordinates in the same BFGS, arsenic squeezed to simple cubic at 500 kbar against `pw.x`, and why a relaxed crystal carries the applied pressure rather than no stress |
| [`24_tran_blaha_band_gaps`](notebooks/24_tran_blaha_band_gaps.ipynb) | A functional whose *potential* is written down and whose energy does not exist: silicon's gap from 0.49 eV to 1.13 against an experimental 1.17, and what `pw.x` actually runs under the name `tb09` |
| [`25_raman_tensors`](notebooks/25_raman_tensors.ipynb) | `d(eps)/d(tau)` as one `jvp` of the same second-order energy notebook 21 differentiates along a strain, validated by a finite difference because the vendored `ph.x`'s third-derivative branch no longer reproduces its own example |
| [`26_raman_and_infrared_spectra`](notebooks/26_raman_and_infrared_spectra.ipynb) | Those tensors projected on the phonon modes: silicon's 519.2 cm⁻¹ line and its infrared silence, against the vendored `dynmat.x` run on our own tensors |

`benchmarks/` holds ready-to-run input files, from a two-atom silicon cell up to
a sixteen-atom one.

## Is it right?

That is the question the project is organised around. Quantum ESPRESSO ships a
test suite with reference outputs, and `pytest` compares against them:

```bash
pip install -e ".[dev]"
python3 -m pytest
```

Silicon's total energy agrees to about 1e-9 Ry term by term, its band structure
to 0.0002 eV, and metals with every smearing to about 2.5e-8 Ry. The ultrasoft
and PAW cases agree to 3e-9 Ry or better on 2- and 8-atom cells, and the PBE
ones to 6e-9 Ry with the band structure within 0.05 meV. Forces agree to 2e-5
Ry/bohr or better on five cases spanning all three kinds of pseudopotential and
a magnetic molecule — term by term, not only in total — and a relaxation ends on
the same geometry to 1e-6 bohr and the same energy to 3e-10 Ry.

Most regression tests need Quantum ESPRESSO's `test-suite` directory, which is not
shipped here; they skip cleanly without it. The ultrasoft, PAW and PBE ones do
not: no benchmark Quantum ESPRESSO ships covers those pseudopotentials and
functionals, so their reference outputs were generated once with `pw.x` and are
committed under `tests/data/qe/` (regenerate with `tools/generate_reference.py`).

## Comparing against Quantum ESPRESSO yourself

If you have `pw.x` built, this runs the same input through both and puts the
numbers side by side:

```bash
python3 tools/compare_qe.py benchmarks/si8-1k.in
```

## License

GPL v3 or later — see [LICENSE](LICENSE). Quantum ESPRESSO is itself GPL, and
this code was written by reading it.

The pseudopotential files under `tests/data/pseudo/` come from the Quantum
ESPRESSO pseudopotential library and carry their own terms.
