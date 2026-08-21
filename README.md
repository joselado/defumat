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

* **Self-consistent field calculations** — total energies, broken down the same
  way `pw.x` prints them.
* **Band structures** along a k-point path.
* **Metals and insulators**, with every smearing scheme Quantum ESPRESSO offers
  (Gaussian, Methfessel-Paxton, Marzari-Vanderbilt, Fermi-Dirac).
* **Norm-conserving, ultrasoft and PAW pseudopotentials** in the UPF format.
* **LDA and GGA functionals** — Perdew-Zunger and Perdew-Wang, PBE, revPBE and
  PBEsol — taken from the pseudopotential's own header, or from `input_dft`.
* **Densities of states**, by smearing or by any of the three tetrahedron
  methods, which also work as an occupation scheme inside the SCF.
* **Collinear spin polarisation**, with one Fermi level or two, and
  **spin-orbit coupling** with two-component spinor wavefunctions.
* **DFT+U** — the simplified rotationally-invariant Hubbard correction, with `U`,
  `J0`, `alpha` and `beta`, on atomic or Löwdin-orthogonalised atomic
  projectors, read from the same `HUBBARD` card `pw.x` reads.
* **Forces**, by differentiating the total energy rather than by evaluating
  hand-derived expressions — and **structural relaxation** on top of them, with
  the same BFGS, trust radius and line search Quantum ESPRESSO uses.
* Crystal symmetry, automatic k-point grids reduced to the irreducible wedge,
  and gamma-only calculations.

Not yet: the stress (and so variable-cell relaxation), the projected density of
states, and phonons. A functional or a combination that is not
implemented is refused with an error naming what *is*, rather than quietly
replaced by something that is.

If your calculation needs any of those, use Quantum ESPRESSO — this is not a
replacement for it, and on anything large it will be slower (about two to four
times, running on one core).

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
