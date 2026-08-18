# Tutorial notebooks

Worked examples on concrete systems, one per capability, in the order the code gained
them. They are the readable counterpart to the test suite: the tests assert that a number
matches Quantum ESPRESSO, the notebooks show what the number *is* and why it comes out
that way.

| Notebook | Covers | Phases |
|---|---|---|
| [`01_silicon_setup.ipynb`](01_silicon_setup.ipynb) | Reading a `pw.x` input, the fcc cell, k-points, G-vectors and FFT grids, the per-k plane-wave basis, Fourier transforms, and the structure factor — all checked against the committed QE benchmark | P0–P2 |

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

## Running them

```bash
pip install -e ".[notebooks]"    # from the repository root: jupyter, matplotlib
jupyter lab notebooks/
```

They need the vendored Quantum ESPRESSO tree at `../quantum_espresso/` for the input
files and reference outputs. That tree is not in the repository (it is 285 MB); the paths
at the top of each notebook say what it expects.

To re-execute one after a code change:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_silicon_setup.ipynb
```
