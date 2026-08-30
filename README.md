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
with excitonic effects. The table below ticks off, quantity by quantity, what
Quantum ESPRESSO and the all-electron code Elk compute as well.

## What it can do today

Each row is a physical quantity you can compute. The two right-hand columns say
whether the established plane-wave and all-electron codes compute it as well:
**QE** is Quantum ESPRESSO (`pw.x` and its post-processing tools) and **Elk** is
the all-electron LAPW code. A tick means the quantity is there; **(✓)** means it
is there only partly, and the note under the table says how; **blank in both
columns is a quantity neither code computes**, and which is therefore pinned by
an identity or by an independent second route rather than by a reference output.

The middle column is the input-file variable that asks for it, where there is
one — it means what it means in a `pw.x` input — and the Python entry point
where there is not. Every one of them is also a method on a `Calculator`
(`calc.get_bands()`, `calc.get_dielectric_tensor()`), which is the short way to
drive any of this and is what the examples below use.

| Feature | How to ask for it | QE | Elk |
|---|---|:-:|:-:|
| **Total energies**, self-consistently, broken down term by term — insulators and metals alike | `calculation = 'scf'` | ✓ | ✓ |
| **Band structures** along a path through the Brillouin zone | `run_bands` | ✓ | ✓ |
| **Densities of states**, by smearing or by tetrahedra | `run_dos`, `pypresso dos` | ✓ | ✓ |
| **Projected densities of states** — resolved by atom, by `l` and by `m`, with Löwdin charges and the spilling parameter | `run_pdos`, `pypresso pdos` | ✓ | ✓ |
| **Forces on the atoms** — unpolarized, collinear spin and noncollinear/spin-orbit, on norm-conserving, ultrasoft and PAW. For a spinor the hand-derived cross-check has no counterpart and `method='analytic'` is refused | `compute_forces` | ✓ | ✓ |
| **Structural relaxation** — the atoms moved downhill to their equilibrium positions | `calculation = 'relax'`, `pypresso relax` | ✓ | ✓ |
| **Variable-cell relaxation** — the cell and the atoms relaxed together, at an applied pressure | `calculation = 'vc-relax'`, `run_vc_relax` | ✓ | ✓ |
| **Stress tensor and pressure**, in Ry/bohr³ and kbar — the same three spin regimes as the force | `tstress = .true.`, `compute_stress`, `pypresso stress` | ✓ | ✓ |
| **Magnetism**, collinear, with one Fermi level or two | `nspin = 2`, `tot_magnetization` | ✓ | ✓ |
| **Magnetism as a vector** — noncollinear, with the magnetic symmetry group | `noncolin` | ✓ | ✓ |
| **Spin-orbit coupling**, two-component spinors and `j`-resolved projectors | `lspinorb` | ✓ | ✓ |
| **Magnetic fields and constrained moments** — a uniform field, and four ways of holding a moment where you put it | `B_field`, `constrained_magnetization` | ✓ | ✓ |
| **Magnetic fields inside one atom's sphere**, and a field that fades away as the run converges | `LOCAL_MAGNETIC_FIELDS` card, `reducebf`, `constrained_magnetization = 'fsm'` | | ✓ |
| **DFT+U** — Dudarev's functional with `U`, `J0`, `alpha`, `beta`. The full Liechtenstein form, the intersite `V` and noncollinear `ns` are refused by name | `HUBBARD` card, `run_scf(starting_ns=...)` | ✓ | ✓ |
| **Spin spirals** at any wavevector, without a supercell. Needs `nosym`; ultrasoft, PAW and spin-orbit coupling are refused | `spiral_q`, `pypresso spiral` | | ✓ |
| **Relaxing the spiral wavevector** down `dE/dq` to the ground-state pitch | `relax_spiral_q` | | |
| **Berry curvature and Chern numbers** — exact integers on any mesh, and a smooth `Omega(k)` map with the truncation of its band sum reported | `run_berry_curvature`, `method="kubo"` for the map | | |
| **Z2 invariants** in 2D and 3D, by Wannier charge centres *and* by parities | `run_z2`, `run_z2_3d` | | |
| **Continuing one run from another across a change of spin regime** — a converged non-magnetic density as the starting point of a magnetic run, a collinear one of a noncollinear run, spin-orbit coupling switched on | `run_scf(starting_from=...)`, `System.with_spin` | (✓)¹ | |
| **Reaching self-consistency** — Anderson/Broyden mixing, Kerker preconditioning, or solving the residual with its own Jacobian, which reaches magnetic solutions no mixer does | `run_scf(mixing_mode=...)`, `run_scf(scf_solver=...)` | (✓)² | (✓)² |
| **Band velocities** `d(eps)/dk`, with the nonlocal pseudopotential's own contribution — norm-conserving, ultrasoft and PAW | `band_velocities`, `VelocityOperator` | (✓)³ | |
| **Effective mass tensor** `m*_ij` at any k-point, with the principal masses and the density-of-states mass. Bands inside a degenerate multiplet are reported as the multiplet's invariant sum | `effective_mass`, `Calculator.get_effective_mass` | | ✓ |
| **Orbital, spin and total angular momentum on each atom** — `<L>`, `<S>`, `<J>`, which is where the orbital moment of a spin-orbit magnet actually sits. Needs the whole k-grid; a relativistic ultrasoft or PAW dataset is refused | `angular_momenta`, `Calculator.get_angular_momenta` | (✓)⁴ | ✓ |
| **Dielectric constant** `epsilon_infinity` and **Born effective charges** — insulators, norm-conserving, ultrasoft and PAW (PAW `Z*` refused). The response solver underneath runs for collinear spin too | `dielectric_tensor` | ✓ | ✓ |
| **Phonons at `Gamma`** — the force constants and their frequencies, insulators and metals, on norm-conserving, ultrasoft and PAW datasets. Away from `Gamma`, and an ultrasoft or PAW metal, are refused | `dynamical_matrix` | ✓ | ✓ |
| **The strain response** `dpsi/d(eps)`, `drho/d(eps)` and the deformation potentials, on norm-conserving, ultrasoft and PAW datasets | `strain_response` | | |
| **Elastic constants** `C_ijkl` and the compliance and bulk modulus that follow — clamped-ion, insulators, norm-conserving | `elastic_constants` | | |
| **Electrostriction coefficients** `m`, `q`, `M` and `Q` — the quadratic electromechanical coupling, clamped-ion, insulators, norm-conserving | `electrostriction` | | |
| **Raman tensors** `d(eps)/d(tau)` — how the dielectric tensor changes when an atom moves. Insulators, norm-conserving/ultrasoft/PAW; `chi^(2)` and the electro-optic tensor are refused | `raman_tensors` | (✓)⁵ | |
| **Raman and infrared spectra** — the per-mode activities, depolarisation ratios and electronic polarizability at `Gamma`. The non-analytic LO-TO splitting is not included, so an optical triplet comes out unsplit | `vibrational_spectrum` | ✓ | |
| **Optical absorption spectra with excitons** — `Im eps_M(omega)` from TDDFT, local-field effects included, on a bootstrap exchange-correlation kernel. Needs the whole k-grid rather than a wedge | `run_absorption`, `kernel = 'bootstrap'` (also `rpa`, `alda`, `lrc`, `bootstrap-1`), `ecut_response`, `scissor`, `broadening` | | ✓ |
| **Van der Waals dispersion** — Grimme's D2 pair correction, in the energy, the forces, the stress and the elastic constants. D3, Tkatchenko-Scheffler, MBD and XDM are refused by name | `vdw_corr = 'grimme-d2'`, `london_s6`, `london_rcut`, `london_c6`, `london_rvdw` | ✓ | |
| **Band gaps from the Tran-Blaha potential** (mBJ) — the modified Becke-Johnson meta-GGA, on norm-conserving and PAW datasets, unpolarized, collinear, and noncollinear with spin-orbit coupling. The total energy is not variational, so forces, stress and response are refused | `input_dft = 'tb09'` (or `'bj06'`), `mbj_c` | (✓)⁶ | ✓ |
| **Pseudopotentials**: norm-conserving, ultrasoft and PAW (UPF v2) | `ATOMIC_SPECIES` | ✓ | |
| **Functionals**: LDA and GGA — Perdew-Zunger, Perdew-Wang, PBE, revPBE, PBEsol | `input_dft`, or the UPF header | ✓ | ✓ |

Where the tick is qualified:

- ¹ `startingpot = 'file'` reads a density across a change of `nspin`, but
  zero-fills the missing components, so a magnetic run started that way
  converges back to the unpolarized answer.
- ² both codes have the mixing; the residual solver, which is what reaches the
  extra solutions, is in neither.
- ³ `fermi_velocity.x` finite-differences eigenvalues and reports only the
  magnitude.
- ⁴ `lorbm` gives the **cell's** orbital magnetization and nothing per atom;
  Elk has the site decomposition.
- ⁵ `ph.x` refuses a gradient-corrected functional here, where this does not.
- ⁶ Quantum ESPRESSO reaches it only through libxc, and then passes a zero
  Laplacian and never sets the functional's coefficient, so what it runs under
  that name is a different functional.

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
from pypresso import Calculator

calc = Calculator.from_file("benchmarks/si-1k.in", pseudo_dir="tests/data/pseudo")
result = calc.get_scf()

print(f"converged in {result.iterations} iterations")
print(f"total energy   {result.total_energy:.8f} Ry")
for name, value in result.energy_terms.items():
    print(f"  {name:<13} {value:>15.8f} Ry")
```

```
converged in 5 iterations
total energy   -15.25444866 Ry
  one-electron       5.26858903 Ry
  hartree            1.26263517 Ry
  xc                -4.88591428 Ry
  ewald            -16.89975858 Ry
```

`benchmarks/si-1k.in` is an ordinary `pw.x` input file. So is anything else you
point `Calculator.from_file` at — the `&control`, `&system` and `&electrons`
namelists, `ATOMIC_SPECIES`, `ATOMIC_POSITIONS` and `K_POINTS` cards all mean
what they mean in Quantum ESPRESSO, and `conv_thr` is compared against the same
quantity. The pseudopotentials are read from the names the `ATOMIC_SPECIES` card
gives; `pseudo_dir` defaults to the input file's own directory.

Every other calculation is a method on the same object, and each runs the SCF
first if none is cached:

```python
calc.get_forces()             # and get_stress(), get_relax(), get_dos()
calc.get_dielectric_tensor()  # and get_phonons(), get_raman_tensors()
calc.get_chern()              # and get_z2(), get_berry_curvature()
```

The functional entry points named in the table above — `run_scf(system,
pseudos, ...)` and the rest — are unchanged and are still there for a script
that manages its own state.

## A band structure

Carrying on from the density that SCF converged:

```python
from pypresso.system.kpoints import KPoints

path = KPoints.band_path(
    [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],   # L - Gamma - X
    [20, 20, 1], calc.system.cell, crystal=False,
)
bands = calc.get_bands(kpoints=path, nbnd=8)

print(f"indirect gap   {bands.gap(8):.3f} eV")
bands.plot()
```

`bands.eigenvalues_ev` is `(k-points, bands)` in eV and `bands.path_length` is
the x-axis for a plot; `bands.plot()` draws one and returns the axes, with the
zero at the Fermi level the SCF found. `DensityOfStates`, `ProjectedDOS` and
`OpticalSpectrum` have the same method. (The gap comes out small because LDA underestimates
gaps — that is the functional, not the code; Quantum ESPRESSO gives the same
answer, and so does PBE.)

## Examples

The `notebooks/` directory is the place to start. Each one is a worked
calculation with its output already in it, so they can be read without being
run, and each has a plain-text `.md` version beside it. Silicon is the default
subject; a second system appears where it shows something silicon cannot — a
metal for smearing, iron for magnetism, arsenic under pressure, bilayer graphene
for dispersion, LiF for a bound exciton.

[`notebooks/README.md`](notebooks/README.md) indexes them by the property you want
to compute, which is the way to arrive at them. In file order:

| | |
|---|---|
| [`00_the_calculator`](notebooks/00_the_calculator.ipynb) | The front door: one object built from an input file, with a method per quantity |
| [`01_silicon_setup`](notebooks/01_silicon_setup.ipynb) | Reading an input file, the crystal, k-points, and the plane-wave basis |
| [`02_silicon_scf_and_bands`](notebooks/02_silicon_scf_and_bands.ipynb) | The SCF, the energy term by term against Quantum ESPRESSO, the band structure, and the bonding charge |
| [`03_eigensolver_and_performance`](notebooks/03_eigensolver_and_performance.ipynb) | How the calculation is made fast, and how it compares to Quantum ESPRESSO |
| [`04_ultrasoft_and_paw`](notebooks/04_ultrasoft_and_paw.ipynb) | Ultrasoft and PAW pseudopotentials: two grids, the augmentation charge, and PAW's one-centre terms |
| [`05_gradient_corrections`](notebooks/05_gradient_corrections.ipynb) | PBE and its relatives: what a gradient correction adds to the potential, on the grid and inside a PAW sphere |
| [`06_density_of_states`](notebooks/06_density_of_states.ipynb) | Smearing against tetrahedra, silicon's gap as the thing that separates them, and nickel's spin-resolved DOS |
| [`07_spin_polarization`](notebooks/07_spin_polarization.ipynb) | LSDA: exchange splitting, nickel's magnetic moment, and constraining the magnetization |
| [`08_spin_orbit_coupling`](notebooks/08_spin_orbit_coupling.ipynb) | Spinors, `j`-resolved projectors, platinum's 5d splitting, and a quantum spin Hall insulator |
| [`09_forces_and_relaxation`](notebooks/09_forces_and_relaxation.ipynb) | Forces as one gradient of the energy, against Quantum ESPRESSO, and a structure relaxing back onto its lattice site |
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
| [`29_effective_mass_and_angular_momenta`](notebooks/29_effective_mass_and_angular_momenta.ipynb) | Effective masses as one difference of an analytic velocity, against the all-electron Elk binary, and where a spin-orbit magnet's orbital moment sits |

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
| phonons at `Gamma` — ultrasoft and PAW silicon | 0.019 and 0.027 cm⁻¹ |
| Raman and infrared activities | every digit `dynmat.x` prints |

**The rows with no tick in either column have no such reference**, since
nothing can be compared against a code that does not compute it. Each is pinned instead by a
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
