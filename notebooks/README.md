# Tutorial notebooks

Worked examples on concrete systems, one per capability, in the order the code gained
them. They are the readable counterpart to the test suite: the tests assert that a number
matches Quantum ESPRESSO, the notebooks show what the number *is* and why it comes out
that way.

| Notebook | Covers | Phases |
|---|---|---|
| [`01_silicon_setup.ipynb`](01_silicon_setup.ipynb) | Reading a `pw.x` input, the fcc cell, k-points, G-vectors and FFT grids, the per-k plane-wave basis, Fourier transforms, and the structure factor — all checked against the committed QE benchmark | P0–P2 |
| [`02_silicon_scf_and_bands.ipynb`](02_silicon_scf_and_bands.ipynb) | What is in a pseudopotential and how it reaches G space, why symmetry cannot be skipped, the SCF loop, the energy term by term against QE (under 1e-9 Ry), the bonding charge, and the band structure (0.0002 eV) | P3–P7 |
| [`03_eigensolver_and_performance.ipynb`](03_eigensolver_and_performance.ipynb) | Why the dense solver had to go: the Hamiltonian's matrix elements, the block Davidson solver and the two traps in transcribing it, what compilation costs when the arrays are small, and the single-core comparison against QE (within 2-4x per SCF iteration) | P4, P10 |
| [`04_ultrasoft_and_paw.ipynb`](04_ultrasoft_and_paw.ipynb) | Ultrasoft and PAW: the two FFT grids, the augmentation charge and the exact charge identity it guarantees, the generalised eigenproblem, `D_ij` becoming self-consistent, PAW's radial one-centre terms, and why `becsum` has to be symmetrised by hand — all against QE (≤3e-9 Ry) with the timing | P12 |

## Conventions

- **Every new feature adds a notebook, or extends one.** A phase is not finished until
  its notebook exists. This is a standing requirement, not a per-phase decision.
- **Silicon first.** New capabilities are demonstrated on the two-atom fcc silicon cell
  from `test-suite/pw_scf/scf.in` wherever they can be, so the notebooks build on a
  system the reader already knows. A second system appears only when it shows something
  silicon cannot (a metal for smearing, a magnetic system for spin).
- **Compare against Quantum ESPRESSO.** If the reference output contains the quantity
  being computed, the notebook ends with a table putting the two side by side. Numbers
  without a reference are labelled as such.
- **Committed with their outputs**, so they read on GitHub without being run.
- **Each notebook has a `.md` export beside it**, regenerated whenever the notebook is.
  Raw `.ipynb` is JSON and unreadable in a plain editor or a diff; the markdown version is
  what to read (and review) when working on the project from anywhere that is not a
  notebook viewer. The `.ipynb` stays the source of truth — edit that, then re-export.

## Running them

```bash
pip install -e ".[notebooks]"    # from the repository root: jupyter, matplotlib
jupyter lab notebooks/
```

They need the vendored Quantum ESPRESSO tree at `../quantum_espresso/` for the input
files and reference outputs. That tree is not in the repository (it is 285 MB); the paths
at the top of each notebook say what it expects. `04` is the exception: its inputs and its
QE references are committed under `tests/data/qe/`, because no benchmark QE ships covers
the ultrasoft and PAW pseudopotentials it uses, so it runs without the vendored tree.

After changing code the notebooks depend on, re-execute them and refresh the exports:

```bash
tools/export_notebooks.sh          # all of them
```

or for one notebook:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_silicon_setup.ipynb
jupyter nbconvert --to markdown --output-dir notebooks notebooks/01_silicon_setup.ipynb
```
