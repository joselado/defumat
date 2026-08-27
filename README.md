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

It also computes things `pw.x` cannot: spin spirals without a supercell,
topological invariants, elastic and electrostriction constants, optical spectra
with excitonic effects. The table below says which is which.

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
| **Forces on the atoms** | `compute_forces` | `pw.x` |
| **Structural relaxation** — BFGS with QE's trust radius and line search | `calculation = 'relax'`, `pypresso relax` | `pw.x` |
| **Variable-cell relaxation** — the cell and the atoms relaxed together, at an applied pressure | `calculation = 'vc-relax'`, `run_vc_relax` | `pw.x` |
| **Stress tensor and pressure**, in Ry/bohr³ and kbar | `tstress = .true.`, `compute_stress`, `pypresso stress` | `pw.x` |
| **Magnetism**, collinear, with one Fermi level or two | `nspin = 2`, `tot_magnetization` | `pw.x` |
| **Magnetism as a vector** — noncollinear, with the magnetic symmetry group | `noncolin` | `pw.x` |
| **Spin-orbit coupling**, two-component spinors and `j`-resolved projectors | `lspinorb` | `pw.x` |
| **Magnetic fields and constrained moments** — all four of QE's schemes | `B_field`, `constrained_magnetization` | `pw.x` |
| **Magnetic fields inside one atom's sphere**, and a field that fades away | `LOCAL_MAGNETIC_FIELDS` card, `reducebf`, `constrained_magnetization = 'fsm'` | **new** — Elk has it, `pw.x` does not |
| **DFT+U** — Dudarev's functional with `U`, `J0`, `alpha`, `beta` | `HUBBARD` card, `run_scf(starting_ns=...)` | `pw.x`. Dudarev's simplified functional; the full Liechtenstein form, the intersite `V` and noncollinear runs are refused by name |
| **Spin spirals** at any wavevector, without a supercell | `spiral_q`, `pypresso spiral` | **new** — Elk has it, `pw.x` does not. Needs `nosym`; ultrasoft, PAW and spin-orbit coupling are refused |
| **Relaxing the spiral wavevector** down `dE/dq` to the ground-state pitch | `relax_spiral_q` | **new** |
| **Berry curvature and Chern numbers**, the latter exact integers on any mesh | `run_berry_curvature` | **new** — QE has the Berry *phase*, not the curvature |
| **Z2 invariants** in 2D and 3D, by Wannier charge centres *and* by parities | `run_z2`, `run_z2_3d` | **new** |
| **Continuing one run from another across a change of spin regime** — a converged non-magnetic density as the starting point of a magnetic run, a collinear one of a noncollinear run, spin-orbit coupling switched on | `run_scf(starting_from=...)`, `System.with_spin` | partly `pw.x` — `startingpot = 'file'` reads a density across a change of `nspin`, but zero-fills the missing components, so a magnetic run started that way converges back to the unpolarized answer |
| **Reaching self-consistency** — Anderson/Broyden mixing, Kerker preconditioning, or solving the residual with its own Jacobian | `run_scf(mixing_mode=...)`, `run_scf(scf_solver=...)` | `pw.x` has the mixing; the residual solver is new, and reaches solutions no mixer does |
| **Band velocities** `d(eps)/dk`, with the nonlocal pseudopotential's own contribution — norm-conserving, ultrasoft and PAW | `band_velocities`, `VelocityOperator` | partly — `fermi_velocity.x` finite-differences eigenvalues and reports only the magnitude |
| **Dielectric constant** `epsilon_infinity` — insulators, norm-conserving, ultrasoft and PAW — **and Born effective charges** (norm-conserving and ultrasoft; PAW refused by name) | `dielectric_tensor` | `ph.x` with `epsil = .true.` |
| **Phonons at `Gamma`** — the force constants and their frequencies, insulators **and metals**, norm-conserving | `dynamical_matrix` | `ph.x`. Away from `Gamma`, and on ultrasoft datasets, they are refused |
| **Elastic constants** `C_ijkl` and the compliance and bulk modulus that follow, clamped-ion, insulators, norm-conserving | `elastic_constants` | **new** — nothing in `pw.x` or `ph.x` computes them |
| **Electrostriction coefficients** `m`, `q`, `M` and `Q` — the quadratic electromechanical coupling — clamped-ion, insulators, norm-conserving | `electrostriction` | **new** — no counterpart in `pw.x` or `ph.x` |
| **Raman tensors** `d(eps)/d(tau)` — the derivative of the dielectric tensor with respect to an atomic coordinate, insulators, norm-conserving | `raman_tensors` | partly `ph.x` (`lraman = .true.`), which refuses a gradient-corrected functional where this does not. `chi^(2)` and the electro-optic tensor are refused |
| **Raman and infrared spectra** — the per-mode activities, depolarisation ratios and electronic polarizability at `Gamma`, insulators, norm-conserving | `vibrational_spectrum` | `dynmat.x`. The non-analytic LO-TO splitting is not included, so an optical triplet comes out unsplit |
| **Optical absorption spectra with excitons** — `Im eps_M(omega)` from TDDFT, local-field effects included, on a **bootstrap** exchange-correlation kernel | `run_absorption`, `kernel = 'bootstrap'` (also `rpa`, `alda`, `lrc`, `bootstrap-1`), `ecut_response`, `scissor`, `broadening` | **new** — Elk has it (`fxctype = 210`); `pw.x` has nothing, and `TDDFPT/` is a different method with no bootstrap kernel. Needs the whole k-grid rather than a wedge |
| **Van der Waals dispersion** — Grimme's D2 pair correction, in the energy, the forces, the stress and the elastic constants | `vdw_corr = 'grimme-d2'`, `london_s6`, `london_rcut`, `london_c6`, `london_rvdw` | `pw.x`. D3, Tkatchenko-Scheffler, MBD and XDM are refused by name |
| **Band gaps from the Tran-Blaha potential** (mBJ) — the modified Becke-Johnson meta-GGA, on norm-conserving **and PAW** datasets, unpolarized, collinear, and noncollinear **with spin-orbit coupling** | `input_dft = 'tb09'` (or `'bj06'`), `mbj_c` | partly `pw.x` — which reaches it only through libxc, and then passes a zero Laplacian and never sets the functional's coefficient, so what it runs under that name is a different functional. **The total energy is not variational**, so forces, stress and response are refused |
| **Pseudopotentials**: norm-conserving, ultrasoft and PAW (UPF v2) | `ATOMIC_SPECIES` | `pw.x` |
| **Functionals**: LDA and GGA — Perdew-Zunger, Perdew-Wang, PBE, revPBE, PBEsol | `input_dft`, or the UPF header | `pw.x` |

The variants under each row — which smearing or tetrahedron method fixes the
occupations, which projectors DFT+U uses, which constraint scheme — are chosen
with the same input variables as in `pw.x` where it has them.

**Not yet:** phonons away from `Gamma`, exact exchange, real-time propagation.
`K_POINTS gamma` runs, but at an explicit k = 0 with the full G sphere — the
same answer at twice the cost, and the run says so.

**Anything not implemented is refused with an error naming what is**, rather
than quietly replaced by something else. That applies to combinations as well as
features, so a run that starts is one whose physics is all there.

If your calculation needs any of those, use Quantum ESPRESSO — this is not a
replacement for it, and on anything large it will be slower (about two to four
times, running on one core).

**Full feature reference:** [`docs/features.pdf`](docs/features.pdf) — every
capability, the equations behind it, a snippet that runs it, what it was
validated against, and what it refuses. The table above is the summary; that is
the detail. Source is `docs/features.tex`; rebuild with
`xelatex docs/features.tex` (twice, for the table of contents).

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
calculation with its output already in it, so they can be read without being
run, and each has a plain-text `.md` version beside it. Silicon is the default
subject; a second system appears where it shows something silicon cannot — a
metal for smearing, iron for magnetism, arsenic under pressure, bilayer graphene
for dispersion, LiF for a bound exciton.

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
| [`12_spin_spirals`](notebooks/12_spin_spirals.ipynb) | Spin spirals of any pitch without a supercell, and an `E(q)` magnon dispersion |
| [`13_dft_plus_u`](notebooks/13_dft_plus_u.ipynb) | The Hubbard correction: nickel four ways, and antiferromagnetic FeO |
| [`14_spiral_relaxation`](notebooks/14_spiral_relaxation.ipynb) | `dE/dq`: which terms of the energy a spiral's wavevector touches, and a BFGS walking a hydrogen chain to its ground-state pitch |
| [`15_stress`](notebooks/15_stress.ipynb) | The stress as the strain derivative of the energy, silicon's equation of state, and the pressure against `-dE/dV` |
| [`16_projected_density_of_states`](notebooks/16_projected_density_of_states.ipynb) | Silicon's `s` and `p` densities of state against `projwfc.x`, and the same weights as fat bands |
| [`17_reaching_self_consistency`](notebooks/17_reaching_self_consistency.ipynb) | Making a hard SCF converge: preconditioning, and the magnetic solutions no mixer reaches |
| [`18_continuing_a_calculation`](notebooks/18_continuing_a_calculation.ipynb) | Starting one run from another's converged state, across a change of spin regime |
| [`19_linear_response`](notebooks/19_linear_response.ipynb) | The dielectric constant and the Born effective charges of silicon, against `ph.x`, on all three kinds of pseudopotential |
| [`20_phonons`](notebooks/20_phonons.ipynb) | Phonon frequencies at `Gamma`: silicon's optical mode against `ph.x`, and a metal |
| [`21_electrostriction`](notebooks/21_electrostriction.ipynb) | Elastic constants, electrostriction and the elasto-optic tensor of silicon |
| [`22_van_der_waals`](notebooks/22_van_der_waals.ipynb) | Grimme's D2 dispersion, and bilayer graphene binding where PBE alone has no minimum |
| [`23_variable_cell_relaxation`](notebooks/23_variable_cell_relaxation.ipynb) | Relaxing the cell and the atoms together: arsenic squeezed to simple cubic at 500 kbar, against `pw.x` |
| [`24_tran_blaha_band_gaps`](notebooks/24_tran_blaha_band_gaps.ipynb) | Band gaps from the modified Becke-Johnson potential: silicon from LDA's 0.49 eV to 1.13, against an experimental 1.17 |
| [`25_raman_tensors`](notebooks/25_raman_tensors.ipynb) | Raman tensors — how the dielectric tensor changes when an atom moves |
| [`26_raman_and_infrared_spectra`](notebooks/26_raman_and_infrared_spectra.ipynb) | Raman and infrared activities per mode: silicon's 519.2 cm⁻¹ line, and why it is infrared-silent |
| [`27_excitons_and_tddft`](notebooks/27_excitons_and_tddft.ipynb) | Optical absorption from TDDFT with a bootstrap kernel, and the excitonic peak RPA does not have |

`benchmarks/` holds ready-to-run input files, from a two-atom silicon cell up to
a sixteen-atom one.

## Is it right?

That is the question the project is organised around. Quantum ESPRESSO ships a
test suite with reference outputs, and `pytest` compares against them:

```bash
pip install -e ".[dev]"
python3 -m pytest
```

Where Quantum ESPRESSO computes the same quantity, the answer is compared with
its number. Some of the headline agreements:

| | agrees to |
|---|---|
| total energies — silicon, term by term | 1e-9 Ry |
| band structures | 0.0002 eV |
| metals, every smearing and the tetrahedron methods | 2.5e-8 Ry |
| ultrasoft and PAW | 3e-9 Ry |
| PBE, revPBE and PBEsol | 6e-9 Ry, 0.05 meV in the bands |
| collinear spin — nickel's energy, and its moment | 1.2e-9 Ry; 0.7280 against 0.73 |
| spin-orbit coupling, and noncollinear magnetism | 1.3e-8 and 2.8e-9 Ry |
| DFT+U | 6.7e-9 Ry |
| forces, term by term | 2e-5 Ry/bohr |
| relaxation — the same geometry, and the same energy | 1e-6 bohr, 3e-10 Ry |
| stress | 2.7e-7 Ry/bohr³ |
| the dielectric constant, and Born effective charges | 1.2e-4; every digit `ph.x` prints |
| phonons at `Gamma` — silicon, and a metal | 0.05 and 0.0019 cm⁻¹ |
| Raman and infrared activities | every digit `dynmat.x` prints |

**The features marked "new" have no such reference**, since nothing can be
compared against a code that does not compute it. Each is pinned instead by a
statement the answer has to satisfy independently of how it was computed — a
Chern number that has to come out an exact integer, a spin spiral that has to
reproduce the supercell calculation of the same magnetic order (it does, to
1e-12 Ry), a derivative that has to match a finite difference of the thing it is
the derivative of. Where another code does compute it, that is used instead:
LiF's excitonic peak comes out at 14.05 eV against the 13.67 eV of Elk, whose
example it is. Where a second, independent route to the same number exists,
both are computed and compared. The per-feature detail is in
[`docs/features.pdf`](docs/features.pdf), which says for every capability what
it was validated against.

Most regression tests need Quantum ESPRESSO's `test-suite` directory, which is
not shipped here; they skip cleanly without it. The ultrasoft, PAW and PBE ones
do not: no benchmark Quantum ESPRESSO ships covers those pseudopotentials and
functionals, so their reference outputs were generated once with `pw.x` and are
committed under `tests/data/qe/`.

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
