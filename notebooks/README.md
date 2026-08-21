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
| [`05_gradient_corrections.ipynb`](05_gradient_corrections.ipynb) | Gradient-corrected functionals: how QE composes a functional out of four slots, PBE's enhancement factor, both potential terms from `jax.grad` against QE's hand-derived algebra, the divergence term on the grid and on a PAW sphere, PBE/revPBE/PBEsol against QE on all three kinds of pseudopotential (≤6e-9 Ry), and the band structure (0.05 meV) | P13 |
| [`06_density_of_states.ipynb`](06_density_of_states.ipynb) | The density of states: why it is an NSCF run on a denser grid, the smeared delta as `jax.jvp` of the occupation function, `D(E)` as `jax.grad` of `N(E)`, silicon's gap as the thing that separates the two schemes, all three tetrahedron variants against QE's aluminium benchmarks (2.5e-8 Ry, and Fermi levels exact to QE's four decimals), and the NaN that appears only in the gradient; then nickel's spin-resolved DOS, where the Fermi level is found from both channels at once and the moment comes back out of the integrated curves | P8, P9 |
| [`07_spin_polarization.ipynb`](07_spin_polarization.ipynb) | LSDA: which parts of the energy split between the spin channels and which do not, exchange by the spin-scaling relation and correlation by interpolation, an oxygen atom with its occupations fixed by hand, nickel's magnetic moment and the exchange splitting of its d bands, the non-monotonic occupation that makes a spin-polarized metal's Fermi level a trap, and constraining the magnetization with two Fermi levels | P9 |
| [`08_spin_orbit_coupling.ipynb`](08_spin_orbit_coupling.ipynb) | Spin-orbit coupling: why a spinor needs three spin numbers where a collinear code needs one, the `j`-resolved projectors a fully-relativistic pseudopotential keeps and a scalar one throws away, `fcoef` verified as a shell projector rather than against a reference, the identity that gates the whole spinor path (switch the coupling off and the collinear answer must come back term by term), platinum's 5d splitting and Kramers degeneracy against QE, and **bismuthene** -- a quantum spin Hall insulator whose half-electronvolt gap is made of nothing but the spin-orbit coupling | P14 |
| [`09_forces_and_relaxation.ipynb`](09_forces_and_relaxation.ipynb) | Forces and structural relaxation: why the force is a *partial* derivative and what makes it one, the energy identity that has to hold before anything is differentiated, the force against QE and against finite differences, QE's six hand-derived terms transcribed beside the gradient — and the two errors that comparison found — the SCF-correction term as the exact difference between the two methods, BFGS with its trust radius putting displaced silicon back on its site (QE's geometry to 1e-6 bohr), and a CO molecule with an atom frozen by `if_pos` | P15 |
| [`10_topological_invariants.ipynb`](10_topological_invariants.ipynb) | Berry curvature, Chern numbers and Z2: why every invariant is built from one overlap and not from a derivative of the eigenproblem, the wrap at the zone edge measured (0.99 against 0.0096), a Chern number that is an *exact* integer on a 6x6 mesh against a Kubo sum that is 1e-3 off on a 24x24 one, silicon's curvature vanishing pointwise, the Wannier centres switching partners across the Kane-Mele transition, three independent routes agreeing on the doubled Qi-Wu-Zhang model, all four `(nu0; nu1nu2nu3)` phases of the lattice Dirac model, silicon's eight parity products, and **bismuthene** by both routes with an ultrasoft spin-orbit `S` | P16 |
| [`11_noncollinear_magnetism_and_fields.ipynb`](11_noncollinear_magnetism_and_fields.ipynb) | Magnetism as a *vector*: why the symmetry group shrinks and half of what is left needs time reversal, the rotation-invariance identity that gates the whole regime (and the 129 Ry bug it found), bcc iron against QE with LDA and with PBE's local spin frame, what "the moment on this atom" means without muffin tins, and fields and constrained moments — five of QE's hand-derived potentials reproduced by one `jax.grad`, three constrained benchmarks, Elk's fixed-spin-moment field beside them, and `reducebf` finding a magnetic state a nonmagnetic start could never reach | P17, P18 |
| [`12_spin_spirals.ipynb`](12_spin_spirals.ipynb) | Spin spirals by the generalized Bloch theorem: two plane-wave spheres instead of one, the three identities that validate them (`q = 0` against an ordinary noncollinear run, `q = b3/2` against the *collinear* antiferromagnet, `q = b3/4` against a four-cell noncollinear supercell — all to 1e-10 Ry), the shifted supercell k-grid that is the trap of the phase, an `E(q)` magnon dispersion with the exchange constants fitted out of it, and why the rotated-frame moment is a gauge and the energy is not | P19 |
| [`13_dft_plus_u.ipynb`](13_dft_plus_u.ipynb) | DFT+U: the correction as a penalty on fractional occupation, the two offsets a Hubbard manifold needs, why `Hubbard_projectors = 'atomic'` still applies `S`, the atomic-orbital renormalisation that only `ortho-atomic` feels (and the 7e-4 Ry it is worth), the potential as `jax.grad` against QE's `v_hubbard`, the `nspin = 1` factor of two that is on the energy and not the potential, nickel four ways and antiferromagnetic FeO against QE (≤6.7e-9 Ry), a second self-consistent solution reached with `starting_ns_eigenvalue`, and `force_hub` falling out of the gradient with nothing written down for it (QE's forces to 4.8e-6 Ry/bohr) | P20 |
| [`14_spiral_relaxation.ipynb`](14_spiral_relaxation.ipynb) | Relaxing a spin spiral: `q` treated as a coordinate the way an atomic position is — which terms of the energy depend on it (two out of seven, and why the density is not one of them), the identity that the differentiated functional *is* the total energy (2e-16 Ry), the gradient against a finite difference of a re-converged SCF, the two wavevectors symmetry pins to zero exactly, what the basis-set jumps in `E(q)` cost and how they fall with the cutoff, and a BFGS on the reciprocal metric taking a hydrogen chain from `q3 = 0.30` to its antiferromagnet at `0.50003` in six SCF runs | P21 |

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
at the top of each notebook say what it expects. `04` and `05` are the exceptions: their
inputs and their QE references are committed under `tests/data/qe/`, because no benchmark
QE ships covers the ultrasoft, PAW and PBE datasets they use, so they run without the
vendored tree. `07` is a mixture: its inputs come from the vendored tree but every
reference it compares against is committed, since QE's own benchmarks for those cases stop
at `conv_thr = 1e-6` and their printed energy terms are only good to about 1e-4 Ry. `09`
runs without the vendored tree too, except for one cell -- QE's own CO relaxation, whose
*input* lives in the test suite -- which says so and skips itself when the tree is absent. `08`
is a mixture for the same reason, and its bismuthene half -- input *and* reference -- is
committed, since QE ships no benchmark for it.

`10` runs without the vendored tree as well -- its two systems are `si2-us.in` and
`bismuthene-soc-small.in`, both committed. It takes about **nine minutes** on one core and
peaks at **6.0 GB**, nearly all of it the bismuthene half. It does *not* run the
Wannier-charge-centre sweep on bismuthene and quotes that measurement instead: an SCF and a
topology run each build their own gigabyte-scale `Calculation`, and doing both plus the
sweep in one kernel peaked at 7.8 GB. The notebook says so where it matters.

`08` runs bismuthene at the test-sized cutoff (20 Ry, 6x6x1) rather than the converged
one (35 Ry, 12x12x1). Both pairs are committed with their own QE references, and the
small pair is what the regression tests check and what `PLAN.md`'s P14 entry quotes.
Switching the notebook to the converged pair is one variable, `SIZE`, at the cost of about
forty minutes in total and a peak of 9.4 GB.

`11` needs the vendored tree for its inputs -- QE's `pw_noncolin` cases, which have no
committed copy here -- but every reference it compares against is regenerated and committed
under `tests/data/qe/`, for the reason its own text gives: the shipped 2017 output of
`noncolin-constrain_atomic.in` does not belong to the input beside it. `12` runs without the
vendored tree entirely; both of its systems (a hydrogen atom and a hydrogen chain, with the
supercells it is validated against) are committed. `11` takes about **eight minutes** on one
core and `12` about **five**. `14` runs without the vendored tree too — it is `12`'s chain
and nothing else — and takes about **twenty**, most of it the cutoff sweep that measures the
basis-set jumps and the `E(q)` scan the relaxation is drawn against.

`13` is a mixture in the other direction from `07`: its nickel half is committed under
`tests/data/qe/` (input *and* reference), and its FeO half reads QE's own `pw_lda+U/`
inputs from the vendored tree while comparing against regenerated references that are
committed. It takes about **eight minutes**, most of it the four FeO runs.

After changing code the notebooks depend on, re-execute them and refresh the exports:

```bash
tools/export_notebooks.sh          # all of them
```

or for one notebook:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_silicon_setup.ipynb
jupyter nbconvert --to markdown --output-dir notebooks notebooks/01_silicon_setup.ipynb
```
