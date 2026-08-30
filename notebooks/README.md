# Tutorial notebooks

One notebook per capability, in the order the code gained them. Each is meant to be read in
about **five minutes**: what the quantity is, the code that computes it, one figure, and the
comparison against Quantum ESPRESSO. They are the readable counterpart to the test suite:
the tests assert that a number matches QE, the notebooks show what the number *is*.

They are about the physics. The derivations, the implementation notes and the per-case
validation tables live in `PLAN.md` and in the tests, and are deliberately kept out of here.

| Notebook | Covers |
|---|---|
| [`01_silicon_setup.ipynb`](01_silicon_setup.ipynb) | Input file to cell to k-points to the plane-wave basis, and what a cutoff can represent |
| [`02_silicon_scf_and_bands.ipynb`](02_silicon_scf_and_bands.ipynb) | The SCF, the energy term by term (1e-9 Ry against QE), silicon's band structure and its covalent bond |
| [`03_eigensolver_and_performance.ipynb`](03_eigensolver_and_performance.ipynb) | What an iterative eigensolver saves over a dense one, and the single-core comparison with QE |
| [`04_ultrasoft_and_paw.ipynb`](04_ultrasoft_and_paw.ipynb) | Softer pseudopotentials: the augmentation charge, the overlap operator, and the charge identity it has to satisfy |
| [`05_gradient_corrections.ipynb`](05_gradient_corrections.ipynb) | PBE, revPBE and PBEsol, what each is fitted for, and the bands they give |
| [`06_density_of_states.ipynb`](06_density_of_states.ipynb) | Smearing and tetrahedra, silicon's gap as what separates them, and free-electron aluminium |
| [`07_spin_polarization.ipynb`](07_spin_polarization.ipynb) | LSDA: nickel's moment as an output, the exchange splitting behind it, and the spin-resolved DOS |
| [`08_spin_orbit_coupling.ipynb`](08_spin_orbit_coupling.ipynb) | Spinors and `j`-resolved projectors, platinum against QE, spinor forces, and bismuthene's spin-orbit gap |
| [`09_forces_and_relaxation.ipynb`](09_forces_and_relaxation.ipynb) | Hellmann-Feynman and Pulay in one derivative, against finite differences, then relaxation |
| [`10_topological_invariants.ipynb`](10_topological_invariants.ipynb) | Chern numbers as exact integers, Wannier-centre flow, Fu-Kane parities, and the curvature as a map |
| [`11_noncollinear_magnetism_and_fields.ipynb`](11_noncollinear_magnetism_and_fields.ipynb) | Magnetism as a vector field, bcc iron against QE, constrained moments, and a fixed-spin-moment search |
| [`12_spin_spirals.ipynb`](12_spin_spirals.ipynb) | The generalized Bloch theorem, the limits that validate it, and a frozen-magnon `E(q)` curve |
| [`13_dft_plus_u.ipynb`](13_dft_plus_u.ipynb) | The occupation penalty that opens FeO's gap, the projectors it is defined by, and nickel and FeO against QE |
| [`14_spiral_relaxation.ipynb`](14_spiral_relaxation.ipynb) | `dE/dq`, and a relaxation that finds a magnet's ground-state pitch in six SCF runs |
| [`15_stress.ipynb`](15_stress.ipynb) | The stress tensor, silicon's equation of state, and the Pulay stress a low cutoff carries |
| [`16_projected_density_of_states.ipynb`](16_projected_density_of_states.ipynb) | Which atom and which orbital a band belongs to, as a projected DOS, Löwdin charges and fat bands |
| [`17_reaching_self_consistency.ipynb`](17_reaching_self_consistency.ipynb) | Charge sloshing and Kerker screening, and the unstable magnetic solutions a mixer cannot reach |
| [`18_continuing_a_calculation.ipynb`](18_continuing_a_calculation.ipynb) | Starting one run from another across a change of spin regime: iron's moment rotated in one iteration |
| [`19_linear_response.ipynb`](19_linear_response.ipynb) | The velocity operator, the Sternheimer equation, silicon's dielectric constant against `ph.x`, and Born charges |
| [`20_phonons.ipynb`](20_phonons.ipynb) | Phonons at Gamma: silicon's optical mode against `ph.x`, the charge that rearranges, and a metal |
| [`21_electrostriction.ipynb`](21_electrostriction.ipynb) | How a strain changes the dielectric constant: electrostriction, the elasto-optic tensor and elastic constants |
| [`22_van_der_waals.ipynb`](22_van_der_waals.ipynb) | Grimme's D2, and bilayer graphene binding at 3.23 A where PBE alone has no minimum at all |
| [`23_variable_cell_relaxation.ipynb`](23_variable_cell_relaxation.ipynb) | Relaxing the cell at an applied pressure: arsenic at 500 kbar going simple cubic |
| [`24_tran_blaha_band_gaps.ipynb`](24_tran_blaha_band_gaps.ipynb) | A functional that is a potential and not an energy: silicon's gap from 0.49 to 1.13 eV |
| [`25_raman_tensors.ipynb`](25_raman_tensors.ipynb) | The Raman tensor as the polarizability's derivative in an atomic position, against a finite difference |
| [`26_raman_and_infrared_spectra.ipynb`](26_raman_and_infrared_spectra.ipynb) | Modes, activities and depolarisation ratios: silicon's 519 cm-1 line, Raman-active and infrared-silent |
| [`27_excitons_and_tddft.ipynb`](27_excitons_and_tddft.ipynb) | Absorption spectra and the bootstrap kernel, and why no adiabatic local kernel binds an exciton |
| [`28_the_calculator.ipynb`](28_the_calculator.ipynb) | One object with a method per calculation, and the caching and immutability rules behind it |
| [`29_effective_mass_and_angular_momenta.ipynb`](29_effective_mass_and_angular_momenta.ipynb) | Band curvature as an effective mass, and site-resolved `<L>`, `<S>` and `<J>` against Elk |

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
- **No em dashes.** Anywhere, in prose or in printed output.
- **Every new feature adds a notebook, or extends one.** A phase is not finished until its
  notebook exists. This is a standing requirement, not a per-phase decision.
- **Drive it with a `Calculator`.** `Calculator.from_file(input, pseudo_dir=...)` and its
  `get_*` methods (notebook 28) are how every notebook here opens, and a new one should
  do the same rather than reach for the `read_pw_input` / `read_upf` / `build_system`
  trio. The functional entry points are unchanged and still correct; a notebook that
  drops back to one says why in place, and briefly.
- **State the observable.** Every notebook heads with the equation of the quantity it
  computes, in display maths, which is the thing a reader wants before any code. Where the
  quantity is a derivative, say *of what, holding what fixed*: that distinction is physics
  and it stays.
- **Five minutes.** Header saying what this computes and the headline number; the shortest
  code that runs it; **one figure that shows the physics**, a band structure wherever the
  feature shows in bands; one table against Quantum ESPRESSO; at most one "how it works"
  cell for the single best *physical* idea; a short footer naming the tests. About eight
  code cells, not twenty.
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

## Running them

```bash
pip install -e ".[notebooks]"    # from the repository root: jupyter, matplotlib
jupyter lab notebooks/
```

Most run in under a minute on one core. The exceptions are `08` (about five minutes, the
bismuthene pair), `11` and `12` (a few minutes each), `13` (its four SCF runs) and `18`
(about three minutes: it runs fourteen SCFs, which is the point, since every number in it
is a *pair* of runs, one from the atoms and one continued). `29` takes **63 s**, most of it the
one spinor SCF on nickel; its Elk comparison and its moment-rotation check are quoted from
offline runs rather than executed.

`22` needs neither: its inputs and its `pw.x` references are committed under `tests/data/qe/`, and its binding curve is quoted from an offline sweep with only the dispersion half recomputed.

Some need the vendored Quantum ESPRESSO tree at `../quantum_espresso/` for their input files
and reference outputs; that tree is not in the repository (it is 285 MB) and the paths at the
top of each notebook say what they expect. `04`, `05`, `09`, `10`, `12` and `14` run without
it: their inputs and references are committed under `tests/data/qe/`, because no benchmark
QE ships covers those cases. `07`, `08`, `11` and `13` are mixtures: the inputs come from the
vendored tree and the references are regenerated and committed, since QE's own benchmarks for
those cases stop at `conv_thr = 1e-6` and their printed terms are only good to about 1e-4 Ry.

Two notebooks quote a measurement rather than running it, and say so where they do: `10`'s
Wannier-charge-centre sweep on bismuthene (7.8 GB in one kernel, so it was run in its own
process) and `14`'s cutoff sweep of the basis-set jumps in `E(q)`. `08` runs bismuthene at
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
