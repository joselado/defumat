# Tutorial notebooks

**Start with [`00_the_calculator`](00_the_calculator.ipynb).** It is the front door: one
object built from a `pw.x` input file, with a method per quantity. Everything else here is
one of those methods, on a concrete crystal, with the number it produces put beside
Quantum ESPRESSO's.

**Then [`25_your_own_crystal`](25_your_own_crystal.ipynb)**, which is the one notebook that
does not open on a file QE shipped: a material you looked up, a pseudopotential you fetched,
and the two convergence tests that have to be run before any number means anything.

Each notebook is meant to be read in about **five minutes**: what the quantity is, the code
that computes it, one figure, and the comparison against QE. They are the readable
counterpart to the test suite — the tests assert that a number matches QE, the notebooks
show what the number *is*.

They are about the physics. The derivations, the implementation notes and the per-case
validation tables live in `PLAN.md` and in the tests, and are deliberately kept out of here.

## What do you want to compute?

The index is by the property, not by the order the code gained it. The call named is a
method on the `Calculator` of notebook 00, unless the entry names an input variable
instead, which means the physics is selected in the input file rather than at the call.

### The ground state

| To compute | Call | Notebook |
|---|---|---|
| A crystal of your own, start to finish | `Calculator.from_file()` | [25](25_your_own_crystal.ipynb) |
| Total energy and charge density | `get_scf()` | [02](02_silicon_scf_and_bands.ipynb) |
| Band structure | `get_bands()` | [02](02_silicon_scf_and_bands.ipynb) |
| Density of states | `get_dos()` | [06](06_density_of_states.ipynb) |
| Which atom and which orbital a band belongs to | `get_pdos()` | [16](16_projected_density_of_states.ipynb) |
| Eigenvalues on a denser grid at fixed density | `get_nscf()` | [06](06_density_of_states.ipynb) |

### Structure: forces, geometry, the cell

| To compute | Call | Notebook |
|---|---|---|
| Forces on the atoms | `get_forces()` | [09](09_forces_and_relaxation.ipynb) |
| A relaxed geometry | `get_relax()` | [09](09_forces_and_relaxation.ipynb) |
| Stress and pressure | `get_stress()` | [15](15_stress.ipynb) |
| A relaxed cell at an applied pressure | `get_relax(variable_cell=True)` | [23](23_variable_cell_relaxation.ipynb) |
| A geometry bound by dispersion | `vdw_corr = 'grimme-d2'` | [22](22_van_der_waals.ipynb) |

### Magnetism

| To compute | Call | Notebook |
|---|---|---|
| A collinear magnetic moment | `nspin = 2` | [07](07_spin_polarization.ipynb) |
| Spin-orbit split bands | `noncolin`, `lspinorb` | [08](08_spin_orbit_coupling.ipynb) |
| Magnetism as a vector, fields, constrained moments | `nspin = 4`, `B_field` | [11](11_noncollinear_magnetism_and_fields.ipynb) |
| A magnon `E(q)` without a supercell | `get_spiral_scan()` | [12](12_spin_spirals.ipynb) |
| A magnet's ground-state pitch | `get_spiral_relaxation()` | [14](14_spiral_relaxation.ipynb) |
| Site-resolved `<L>`, `<S>` and `<J>` | `get_angular_momenta()` | [29](29_effective_mass_and_angular_momenta.ipynb) |
| One run continued from another across a change of spin regime | `with_spin()` | [18](18_continuing_a_calculation.ipynb) |

### Response, vibrations and spectra

| To compute | Call | Notebook |
|---|---|---|
| The dielectric constant | `get_dielectric_tensor()` | [19](19_linear_response.ipynb) |
| Born effective charges | `get_born_charges()` | [19](19_linear_response.ipynb) |
| Band velocities | `get_band_velocities()` | [19](19_linear_response.ipynb) |
| Phonon frequencies at `Gamma` | `get_phonons()` | [20](20_phonons.ipynb) |
| Raman tensors | `get_raman_tensors()` | [26](26_raman_and_infrared_spectra.ipynb) |
| Raman and infrared activities per mode | `get_vibrational_spectrum()` | [26](26_raman_and_infrared_spectra.ipynb) |
| An optical absorption spectrum, with excitons | `get_absorption()` | [27](27_excitons_and_tddft.ipynb) |
| Elastic constants | `get_elastic_constants()` | [21](21_electrostriction.ipynb) |
| Electrostriction and the elasto-optic tensor | `get_electrostriction()`, `get_strain_response()` | [21](21_electrostriction.ipynb) |
| The piezoelectric tensor | `get_piezoelectric_tensor()` | [28](28_piezoelectricity.ipynb) |
| Optical conductivity, the Kerr angle, the anomalous Hall effect | `get_optical_conductivity()` | [30](30_magneto_optics.ipynb) |
| An effective mass | `get_effective_mass()` | [29](29_effective_mass_and_angular_momenta.ipynb) |

### Topology

| To compute | Call | Notebook |
|---|---|---|
| A Berry curvature map | `get_berry_curvature()` | [10](10_topological_invariants.ipynb) |
| A Chern number | `get_chern()` | [10](10_topological_invariants.ipynb) |
| A Z2 invariant | `get_z2()`, `get_z2_3d()` | [10](10_topological_invariants.ipynb) |

### Choosing the physics of the run

| To use | Selected by | Notebook |
|---|---|---|
| Ultrasoft and PAW pseudopotentials | the dataset named in the input | [04](04_ultrasoft_and_paw.ipynb) |
| PBE, revPBE, PBEsol | `input_dft`, or the dataset's header | [05](05_gradient_corrections.ipynb) |
| A Hubbard `U` | the `HUBBARD` card | [13](13_dft_plus_u.ipynb) |
| A band gap that LDA gets wrong | `input_dft = 'tb09'` | [24](24_tran_blaha_band_gaps.ipynb) |

### Under the hood

These three are about the machinery rather than about a property, and their internals are
the subject rather than an intrusion. Read them when a calculation misbehaves, not when you
want a number.

| | |
|---|---|
| [`01_silicon_setup`](01_silicon_setup.ipynb) | Input file to cell to k-points to the plane-wave basis, and what a cutoff can represent |
| [`03_eigensolver_and_performance`](03_eigensolver_and_performance.ipynb) | What an iterative eigensolver saves over a dense one, and the single-core comparison with QE |
| [`17_reaching_self_consistency`](17_reaching_self_consistency.ipynb) | Charge sloshing and Kerker screening, and the unstable magnetic solutions a mixer cannot reach |

## In file order

| Notebook | Covers |
|---|---|
| [`00_the_calculator.ipynb`](00_the_calculator.ipynb) | One object with a method per calculation, and the caching and immutability rules behind it |
| [`01_silicon_setup.ipynb`](01_silicon_setup.ipynb) | Input file to cell to k-points to the plane-wave basis, and what a cutoff can represent |
| [`02_silicon_scf_and_bands.ipynb`](02_silicon_scf_and_bands.ipynb) | The SCF, the energy term by term (1e-9 Ry against QE), silicon's band structure and its covalent bond |
| [`03_eigensolver_and_performance.ipynb`](03_eigensolver_and_performance.ipynb) | What an iterative eigensolver saves over a dense one, and the single-core comparison with QE |
| [`04_ultrasoft_and_paw.ipynb`](04_ultrasoft_and_paw.ipynb) | Softer pseudopotentials: the augmentation charge, the overlap operator, and the charge identity it has to satisfy |
| [`05_gradient_corrections.ipynb`](05_gradient_corrections.ipynb) | PBE, revPBE and PBEsol, what each is fitted for, and the bands they give |
| [`06_density_of_states.ipynb`](06_density_of_states.ipynb) | Smearing and tetrahedra, silicon's gap as what separates them, and free-electron aluminium |
| [`07_spin_polarization.ipynb`](07_spin_polarization.ipynb) | LSDA: nickel's moment as an output, the exchange splitting behind it, and the spin-resolved DOS |
| [`08_spin_orbit_coupling.ipynb`](08_spin_orbit_coupling.ipynb) | Spinors and `j`-resolved projectors, platinum against QE, and bismuthene's gap opening from 0.14 to 0.63 eV |
| [`09_forces_and_relaxation.ipynb`](09_forces_and_relaxation.ipynb) | Hellmann-Feynman and Pulay in one derivative, against Quantum ESPRESSO, then relaxation |
| [`10_topological_invariants.ipynb`](10_topological_invariants.ipynb) | Chern numbers as exact integers, Wannier-centre flow, Fu-Kane parities, and the curvature as a map |
| [`11_noncollinear_magnetism_and_fields.ipynb`](11_noncollinear_magnetism_and_fields.ipynb) | Magnetism as a vector field, bcc iron against QE, constrained moments, and the direction the energy cannot depend on |
| [`12_spin_spirals.ipynb`](12_spin_spirals.ipynb) | The generalized Bloch theorem, the limits that validate it, and a frozen-magnon `E(q)` curve |
| [`13_dft_plus_u.ipynb`](13_dft_plus_u.ipynb) | The occupation penalty that opens FeO's gap, against QE, and the natural occupations it drives to 0 and 1 |
| [`14_spiral_relaxation.ipynb`](14_spiral_relaxation.ipynb) | `dE/dq`, and a relaxation that finds a magnet's ground-state pitch in six SCF runs |
| [`15_stress.ipynb`](15_stress.ipynb) | The stress tensor, silicon's equation of state, and the Pulay stress a low cutoff carries |
| [`16_projected_density_of_states.ipynb`](16_projected_density_of_states.ipynb) | Which atom and which orbital a band belongs to, as a projected DOS, Löwdin charges and fat bands |
| [`17_reaching_self_consistency.ipynb`](17_reaching_self_consistency.ipynb) | Charge sloshing and Kerker screening, and the unstable magnetic solutions a mixer cannot reach |
| [`18_continuing_a_calculation.ipynb`](18_continuing_a_calculation.ipynb) | Starting one run from another across a change of spin regime: iron's moment rotated in one iteration |
| [`19_linear_response.ipynb`](19_linear_response.ipynb) | Silicon's dielectric constant and Born charges against `ph.x`, norm-conserving and ultrasoft, and the charge that does the screening |
| [`20_phonons.ipynb`](20_phonons.ipynb) | Phonons at Gamma: silicon's optical mode against `ph.x`, the charge that rearranges, and a metal |
| [`21_electrostriction.ipynb`](21_electrostriction.ipynb) | How a strain changes the dielectric constant: electrostriction, the elasto-optic tensor and elastic constants |
| [`22_van_der_waals.ipynb`](22_van_der_waals.ipynb) | Grimme's D2, and bilayer graphene binding at 3.23 A where PBE alone has no minimum at all |
| [`23_variable_cell_relaxation.ipynb`](23_variable_cell_relaxation.ipynb) | Relaxing the cell at an applied pressure: arsenic at 500 kbar going simple cubic |
| [`24_tran_blaha_band_gaps.ipynb`](24_tran_blaha_band_gaps.ipynb) | A functional that is a potential and not an energy: silicon's gap from 0.49 to 1.13 eV |
| [`25_your_own_crystal.ipynb`](25_your_own_crystal.ipynb) | Diamond from a lattice constant and a fetched pseudopotential: the cutoff and k-grid tests, then bands and a density of states |
| [`26_raman_and_infrared_spectra.ipynb`](26_raman_and_infrared_spectra.ipynb) | Modes, activities and depolarisation ratios: silicon's 519 cm-1 line, Raman-active and infrared-silent, and the tensor underneath them |
| [`27_excitons_and_tddft.ipynb`](27_excitons_and_tddft.ipynb) | Absorption spectra and the bootstrap kernel, and why no adiabatic local kernel binds an exciton |
| [`28_piezoelectricity.ipynb`](28_piezoelectricity.ipynb) | The voltage a squeezed crystal produces: AlAs's one independent component, and the inversion centre that gives silicon none |
| [`29_effective_mass_and_angular_momenta.ipynb`](29_effective_mass_and_angular_momenta.ipynb) | Band curvature as an effective mass, and site-resolved `<L>`, `<S>` and `<J>` against Elk |
| [`30_magneto_optics.ipynb`](30_magneto_optics.ipynb) | Light reflected off a magnet comes back rotated: nickel's Kerr angle, and the two ingredients it needs |

## Conventions

- **The subject is the physics.** What the quantity is, the equation that defines it, what
  the number means, and how it compares with experiment or with Quantum ESPRESSO. **The
  implementation is not the subject and does not belong here**: no `PLAN.md` phase numbers,
  no QE Fortran file names, no tables of what is transcribed against what is differentiated,
  no `jvp`, tangents, frozen spheres, padding or compilation, no catalogue of traps, and no
  account of how something was developed or debugged. That material is what `PLAN.md` and
  the tests are for. Two sentences survive from that side because they are claims about
  capability rather than about code: one saying a derivative is taken of the energy itself
  rather than derived by hand, and one where a reference is unusual and the reader would
  otherwise not trust the comparison.
- **This binds the code as much as the prose**, which is the rule that was missing and let
  the notebooks drift into validation reports. An identity check across four
  pseudopotentials, a derivative checked against a closed form on a random matrix, a
  hand-built linear solve with a probe potential: these are the test suite's job, and a
  notebook that carries one is doing it in public. They go in `tests/`, and the notebook's
  footer names the file they went to.
- **No em dashes.** Anywhere, in prose or in printed output.
- **Every new feature adds a notebook, or extends one.** A phase is not finished until its
  notebook exists. This is a standing requirement, not a per-phase decision. It adds a row
  to the task index above as well, keyed by the property, not by the feature's name.
- **Drive it with a `Calculator`.** `Calculator.from_file(input, pseudo_dir=...)` and its
  `get_*` methods (notebook 00) are how every notebook here opens, and a new one should
  do the same rather than reach for the `read_pw_input` / `read_upf` / `build_system`
  trio. The functional entry points are unchanged and still correct; a notebook that
  drops back to one says why in place, and briefly. **Where a `get_*` method exists, use
  it**: building the same quantity by hand and then remarking that the method also exists
  is backwards.
- **State the observable.** Every notebook heads with the equation of the quantity it
  computes, in display maths, which is the thing a reader wants before any code. Where the
  quantity is a derivative, say *of what, holding what fixed*: that distinction is physics
  and it stays.
- **Silicon first.** New capabilities are demonstrated on the two-atom fcc silicon cell from
  `test-suite/pw_scf/scf.in` wherever they can be. A second system appears only when it
  shows something silicon cannot (a metal for smearing, a magnetic system for spin).
- **Compare against Quantum ESPRESSO.** If the reference output contains the quantity, the
  notebook puts the two side by side. Numbers without a reference are labelled as such.
- **Expensive sweeps are quoted, not run.** A convergence study or a gigabyte-scale case is
  measured once, offline, and its numbers quoted in the text with a note saying so, so that
  the notebook stays fast to run as well as fast to read.
- **Committed with their outputs**, so they read on GitHub without being run, and **each has
  a `.md` export beside it**: raw `.ipynb` is JSON and unreadable in a plain editor or a
  diff. The `.ipynb` stays the source of truth: edit it, then re-export.

## The shape of a notebook

`tests/unit/test_notebook_conventions.py` enforces what follows, on the
notebook's JSON and without executing it. It is a ratchet: every notebook is
checked for the export being current and for implementation vocabulary, and the
ones listed in its `REWRITTEN` set are held to the whole skeleton. A notebook
joins that set in the commit that rewrites it.


Nine cells, 60 to 70 lines of code, and no code cell over 25 lines. The second cell is the
one the whole notebook is for.

1. **The observable.** Title is the property. Its defining equation in display maths. The
   headline number and the comparison against QE, Elk or experiment, **as a markdown table
   of quoted numbers** rather than as computed output.
2. **The run, and at most ten lines of it**: the imports, `Calculator.from_file(...)`, the
   one `get_X()` call, the number printed plainly. No local `load()` helper, no
   `read_pw_input`, no internals. Settings that matter belong in the input file.
3. **What the number means.** Sign, magnitude, what experiment says.
4. **The figure.** `result.plot(ax=...)` where the result object has one; hand-drawn only
   where it shows physics the result object does not hold.
5. **One live comparison against QE.** One case, one table, ten lines.
6. **Optionally, the single best physical idea**, in one cell, and it is *physical*.
7. **What the feature refuses.** The refusals are the promise that a run which starts is a
   run whose physics is there.
8. **A footer naming the tests**, including the file any identity checks live in.

## Running them

```bash
pip install -e ".[notebooks]"    # from the repository root: jupyter, matplotlib
jupyter lab notebooks/
```

**Every notebook in the set is timed and every one is far inside the ten-minute
ceiling.** `tools/export_notebooks.sh` measures them as it re-executes them and
fails over that ceiling, so the table below is a by-product of keeping the outputs
true rather than something anyone has to remember to do. Wall clock on one
workstation core, slowest last:

| | s | | s | | s | | s |
|---|---|---|---|---|---|---|---|
| `01` | 5 | `00` | 22 | `15` | 31 | `13` | 69 |
| `09` | 6 | `07` | 22 | `24` | 31 | `11` | 81 |
| `02` | 8 | `06` | 23 | `28` | 33 | `14` | 89 |
| `16` | 9 | `12` | 27 | `23` | 35 | `26` | 107 |
| `03` | 10 | `25` | 28 | `19` | 47 | `30` | 131 |
| `05` | 10 | `17` | 29 | `10` | 50 | `08` | 171 |
| `04` | 12 | `18` | 29 | `20` | 57 | `27` | 178 |
| `22` | 12 | `21` | 30 | `29` | 59 |  |  |

Three of those used to be much slower, and each for the same reason. `19` lost two
hand-built linear solves and a second self-consistent run that were demonstrating
machinery rather than physics; `18` ran fourteen self-consistent calculations
where two pairs make its point; and **`08` was the one notebook ever measured over
the ceiling** -- about 25 minutes -- which it no longer is: what went was a
five-run finite-difference sweep and a two-run identity check, both of them
already in the test suite.

Six notebooks quote a measurement rather than running it, and say so where they
do: `10`'s Wannier-charge-centre sweep on bismuthene (7.8 GB in one kernel, so it
was run in its own process), `14`'s cutoff sweep of the basis-set jumps in `E(q)`,
`18`'s silicon and platinum pairs, `21`'s five re-converged strained cells, `22`'s
eleven-point binding curve, whose dispersion half is recomputed live because it is
a pair sum over four nuclei, and `29`'s Elk comparison and moment rotation. `03`'s
comparison against `pw.x` is quoted for a different reason: it needs a `pw.x`
binary, which is not in the repository.

Some need the vendored Quantum ESPRESSO tree at `../quantum_espresso/` for their input files
and reference outputs; that tree is not in the repository (it is 285 MB) and the paths at the
top of each notebook say what they expect. `04`, `05`, `09`, `10`, `12` and `14` run without
it: their inputs and references are committed under `tests/data/qe/`, because no benchmark
QE ships covers those cases. `07`, `08`, `11` and `13` are mixtures: the inputs come from the
vendored tree and the references are regenerated and committed, since QE's own benchmarks for
those cases stop at `conv_thr = 1e-6` and their printed terms are only good to about 1e-4 Ry.

Six notebooks quote a measurement rather than running it, and say so where they do: `10`'s
Wannier-charge-centre sweep on bismuthene (7.8 GB in one kernel, so it was run in its own
process), `14`'s cutoff sweep of the basis-set jumps in `E(q)`, `18`'s silicon and platinum
pairs, `22`'s eleven-point binding curve, whose dispersion half is recomputed live because it
is a pair sum over four nuclei, `29`'s Elk comparison and moment rotation, and `26`'s
finite difference over re-converged displaced cells. `08` runs bismuthene at
the test size (20 Ry, 6x6x1); the converged pair (35 Ry, 12x12x1) is committed beside it with
its own QE reference and is one variable away, at about forty minutes and a 9.4 GB peak.

`19` and `20` run without it too: their inputs and the regenerated `ph.x` outputs they
are compared against are both committed under `tests/data/qe/`, because `ph_base`'s own
benchmark dates from release 6.0 and has drifted. `20` takes about a minute.

`16` runs without the vendored tree as well: its input and the `projwfc.x` reference it is
compared against are both committed under `tests/data/qe/`, because QE's test suite has no
`projwfc` case at all. It takes about two minutes.

After changing code the notebooks depend on, re-execute them and refresh the exports:

```bash
tools/export_notebooks.sh          # all of them
```

or for one notebook:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_silicon_setup.ipynb
jupyter nbconvert --to markdown --output-dir notebooks notebooks/01_silicon_setup.ipynb
```
