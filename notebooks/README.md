# Tutorial notebooks

One notebook per capability, in the order the code gained them. Each is meant to be read in
about **five minutes**: what the calculation is, the code that runs it, one figure, and the
comparison against Quantum ESPRESSO. They are the readable counterpart to the test suite —
the tests assert that a number matches QE, the notebooks show what the number *is*.

The detail behind each one — the derivations, the transcription traps, the per-case
validation tables — lives in `PLAN.md`'s phase entry and in the tests, and every notebook
ends with a pointer to both.

| Notebook | Covers | Phases |
|---|---|---|
| [`01_silicon_setup.ipynb`](01_silicon_setup.ipynb) | Input file → cell → k-points → G-vectors → the plane-wave basis, against QE's printed header | P0–P2 |
| [`02_silicon_scf_and_bands.ipynb`](02_silicon_scf_and_bands.ipynb) | The SCF loop, the energy term by term (1e-9 Ry), and silicon's band structure (0.0002 eV) | P3–P7 |
| [`03_eigensolver_and_performance.ipynb`](03_eigensolver_and_performance.ipynb) | Davidson against a dense solve, the `ethr` schedule, and the single-core comparison with QE | P4, P10 |
| [`04_ultrasoft_and_paw.ipynb`](04_ultrasoft_and_paw.ipynb) | Two grids, the augmentation charge, the exact charge identity, and PAW's one-centre terms (≤3e-9 Ry) | P12 |
| [`05_gradient_corrections.ipynb`](05_gradient_corrections.ipynb) | PBE, revPBE and PBEsol, both potentials from `jax.grad`, and the bands they give (0.05 meV) | P13 |
| [`06_density_of_states.ipynb`](06_density_of_states.ipynb) | Smearing and tetrahedra, silicon's gap as what separates them, aluminium against QE | P8 |
| [`07_spin_polarization.ipynb`](07_spin_polarization.ipynb) | LSDA: nickel's moment, its exchange-split bands, and the same thing as a spin-resolved DOS | P9 |
| [`08_spin_orbit_coupling.ipynb`](08_spin_orbit_coupling.ipynb) | Spinors and `j`-resolved projectors, platinum against QE, and bismuthene's spin-orbit gap | P14 |
| [`09_forces_and_relaxation.ipynb`](09_forces_and_relaxation.ipynb) | The force as one gradient, against QE's six terms and against finite differences, then BFGS | P15 |
| [`10_topological_invariants.ipynb`](10_topological_invariants.ipynb) | Chern numbers as exact integers, Wannier-centre flow, Fu-Kane parities, and when they disagree | P16 |
| [`11_noncollinear_magnetism_and_fields.ipynb`](11_noncollinear_magnetism_and_fields.ipynb) | Magnetism as a vector, bcc iron against QE, constrained moments, and Elk's fading field | P17, P18 |
| [`12_spin_spirals.ipynb`](12_spin_spirals.ipynb) | The generalized Bloch theorem, the three identities that validate it, and an `E(q)` magnon curve | P19 |
| [`13_dft_plus_u.ipynb`](13_dft_plus_u.ipynb) | The occupation penalty, `S`-weighted projectors, nickel and FeO against QE (≤6.7e-9 Ry) | P20 |
| [`14_spiral_relaxation.ipynb`](14_spiral_relaxation.ipynb) | `dE/dq` by `jax.grad`, checked by finite differences, and a BFGS that finds the ground-state pitch | P21 |
| [`15_stress.ipynb`](15_stress.ipynb) | The stress as one strain derivative of the energy, silicon's equation of state with `-dE/dV` on top, and five references against `pw.x` (≤2.7e-7 Ry/bohr³) | P11 |
| [`16_projected_density_of_states.ipynb`](16_projected_density_of_states.ipynb) | `<phi|S|psi>` on Löwdin-orthogonalised orbitals, silicon's `s` and `p` densities of state against `projwfc.x`, and the same weights as fat bands | P8 |

## Conventions

- **Every new feature adds a notebook, or extends one.** A phase is not finished until its
  notebook exists. This is a standing requirement, not a per-phase decision.
- **Five minutes.** Header saying what this computes and the headline number; the shortest
  code that runs it; **one figure that shows the physics** — a band structure wherever the
  feature shows in bands; one table against Quantum ESPRESSO; at most one "how it works"
  cell for the single best idea; a footer pointing at the `PLAN.md` phase and the test file.
  About eight code cells, not twenty.
- **Silicon first.** New capabilities are demonstrated on the two-atom fcc silicon cell from
  `test-suite/pw_scf/scf.in` wherever they can be. A second system appears only when it
  shows something silicon cannot (a metal for smearing, a magnetic system for spin).
- **Compare against Quantum ESPRESSO.** If the reference output contains the quantity, the
  notebook puts the two side by side. Numbers without a reference are labelled as such.
- **Expensive sweeps are quoted, not run.** A convergence study or a gigabyte-scale case is
  measured once, offline, and its numbers quoted in the text with a note saying so — the
  notebook stays fast to run as well as fast to read.
- **Committed with their outputs**, so they read on GitHub without being run, and **each has
  a `.md` export beside it**: raw `.ipynb` is JSON and unreadable in a plain editor or a
  diff. The `.ipynb` stays the source of truth — edit it, then re-export.

## Running them

```bash
pip install -e ".[notebooks]"    # from the repository root: jupyter, matplotlib
jupyter lab notebooks/
```

Most run in under a minute on one core. The exceptions are `08` (about five minutes, the
bismuthene pair), `11` and `12` (a few minutes each) and `13` (its four SCF runs).

Some need the vendored Quantum ESPRESSO tree at `../quantum_espresso/` for their input files
and reference outputs; that tree is not in the repository (it is 285 MB) and the paths at the
top of each notebook say what they expect. `04`, `05`, `09`, `10`, `12` and `14` run without
it — their inputs and references are committed under `tests/data/qe/`, because no benchmark
QE ships covers those cases. `07`, `08`, `11` and `13` are mixtures: the inputs come from the
vendored tree and the references are regenerated and committed, since QE's own benchmarks for
those cases stop at `conv_thr = 1e-6` and their printed terms are only good to about 1e-4 Ry.

Two notebooks quote a measurement rather than running it, and say so where they do: `10`'s
Wannier-charge-centre sweep on bismuthene (7.8 GB in one kernel, so it was run in its own
process) and `14`'s cutoff sweep of the basis-set jumps in `E(q)`. `08` runs bismuthene at
the test size (20 Ry, 6x6x1); the converged pair (35 Ry, 12x12x1) is committed beside it with
its own QE reference and is one variable away, at about forty minutes and a 9.4 GB peak.

`16` runs without the vendored tree as well — its input and the `projwfc.x` reference
it is compared against are both committed under `tests/data/qe/`, because QE's test
suite has no `projwfc` case at all — and takes about two minutes.

After changing code the notebooks depend on, re-execute them and refresh the exports:

```bash
tools/export_notebooks.sh          # all of them
```

or for one notebook:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_silicon_setup.ipynb
jupyter nbconvert --to markdown --output-dir notebooks notebooks/01_silicon_setup.ipynb
```
