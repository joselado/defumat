# defumat

Plane-wave density-functional theory in Python, driven from an ordinary Quantum
ESPRESSO `pw.x` input file. It computes the ground state, the band structure and
the density of states; forces, stress and relaxed geometries; the dielectric,
vibrational and optical response; magnetism from a collinear moment to a spin
spiral and a magnon; and a set of quantities that neither
[Quantum ESPRESSO](https://www.quantum-espresso.org) nor
[Elk](https://elk.sourceforge.io) computes, among them the Chern and
$\mathbb Z_2$ invariants, the shift current, the Heisenberg exchange constants
read off a spin-spiral scan, and the elastic constants. Every derivative
quantity, from a force to a Raman tensor, is a derivative of the total energy
itself rather than a formula derived by hand, and every number that Quantum
ESPRESSO also computes has been compared against it on the same input.

## Installing

```bash
git clone https://github.com/joselado/defumat
cd defumat
pip install -e .
```

Python 3.10 or newer. The dependencies are JAX, NumPy, SciPy, Numba and equinox,
and `pip` will fetch them.

## A first calculation

Silicon, from the input file in `benchmarks/`:

```python
from defumat import Calculator

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

`benchmarks/si-1k.in` is an ordinary `pw.x` input file, and so is anything else
you point `Calculator.from_file` at: the `&control`, `&system` and `&electrons`
namelists and the `ATOMIC_SPECIES`, `ATOMIC_POSITIONS` and `K_POINTS` cards mean
what they mean in Quantum ESPRESSO, and `conv_thr` is compared against the same
quantity. The pseudopotentials are read from the names on the `ATOMIC_SPECIES`
card, and `pseudo_dir` defaults to the input file's own directory.

Every other quantity is a method on the same object, and each runs the SCF first
if none is cached:

```python
calc.get_forces()             # and get_stress(), get_relax(), get_dos()
calc.get_dielectric_tensor()  # and get_phonons(), get_raman_tensors()
calc.get_chern()              # and get_z2(), get_berry_curvature()
```

## A band structure

Carrying on from the density the SCF converged:

```python
from defumat.system.kpoints import KPoints

path = KPoints.band_path(
    [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],   # L - Gamma - X
    [20, 20, 1], calc.system.cell, crystal=False,
)
bands = calc.get_bands(kpoints=path, nbnd=8)

print(f"indirect gap   {bands.gap(8):.3f} eV")
bands.plot()
```

`bands.eigenvalues_ev` is `(k-points, bands)` in eV, `bands.path_length` is the
x-axis for a plot, and `bands.plot()` draws one with the zero at the Fermi level
the SCF found. The gap comes out small because LDA underestimates gaps, which is
the functional and not the code: Quantum ESPRESSO gives the same answer.

## What you can compute

One functional is written down, the Kohn-Sham total energy of the wavefunctions,
the atomic positions and the strain,

$$
E[\{\psi\},\boldsymbol\tau,\varepsilon] = T_s + E_{\mathrm H} + E_{xc}
  + E_{\mathrm{loc}} + E_{\mathrm{nl}} + E_{\mathrm{Ewald}},
$$

and most of what follows is its minimum or one of its derivatives, the rest
being properties of the states that minimise it, a tunnelling image or a
structure factor. Each group opens with the equation of its headline quantity
and lists the others with the one call that computes each, or the input variable
that selects it, and the notebook that works it through on a real crystal. The
full guide, with a snippet per quantity and what each refuses, is
[`docs/features.pdf`](docs/features.pdf), and
[`notebooks/README.md`](notebooks/README.md) indexes the notebooks by the
property you want.

### The ground state

The Kohn-Sham equations, solved self-consistently in a plane-wave basis,

$$
H[n]\,\psi_{n\mathbf k} = \epsilon_{n\mathbf k}\,S\,\psi_{n\mathbf k},
\qquad
n(\mathbf r) = \sum_{n\mathbf k} f_{n\mathbf k}\,\lvert\psi_{n\mathbf k}(\mathbf r)\rvert^2
  + n_{\mathrm{aug}}(\mathbf r),
$$

where $S$ is the identity for a norm-conserving pseudopotential and the overlap
operator for an ultrasoft or PAW one, whose augmentation charge $n_{\mathrm{aug}}$
puts back the density the soft wavefunctions leave out. The total energy comes
back broken down term by term, for insulators and for metals, with the
occupations fixed by a smearing or by the tetrahedron method exactly as the
`occupations` variable asks. `calc.get_scf()`, notebooks
[02](notebooks/02_silicon_scf_and_bands.ipynb) and
[25](notebooks/25_your_own_crystal.ipynb).

- **Band structure** along a path through the Brillouin zone, $\epsilon_{n\mathbf k}$
  at the converged density with the Fermi level as its zero. `calc.get_bands()`,
  notebook [02](notebooks/02_silicon_scf_and_bands.ipynb). The eigenvalues on a
  denser grid at fixed density, which a density of states is built on, are
  `calc.get_nscf()`.
- **Density of states**, $g(E) = \sum_{n\mathbf k} w_{\mathbf k}\,\delta(E - \epsilon_{n\mathbf k})$,
  by a smearing or by tetrahedra. `calc.get_dos()`, notebook
  [06](notebooks/06_density_of_states.ipynb).
- **Projected density of states**, the same sum weighted by
  $\lvert\langle\phi^{I}_{lm}\vert\psi_{n\mathbf k}\rangle\rvert^2$, so resolved by
  atom, by $l$ and $m$, by spin channel where the run is magnetic, or by $j$ and
  $m_j$ for a spin-orbit run, with the Löwdin charges beside it. `calc.get_pdos()`,
  notebook [16](notebooks/16_projected_density_of_states.ipynb).
- **Band velocities**, $\mathbf v_{n\mathbf k} = \partial\epsilon_{n\mathbf k}/\partial\mathbf k$,
  including the nonlocal pseudopotential's own term. `calc.get_band_velocities()`,
  notebook [19](notebooks/19_linear_response.ipynb).
- **Effective mass tensor**, $(m^{\ast})^{-1}_{ij} = \partial^2\epsilon_{n\mathbf k}/\partial k_i\,\partial k_j$
  at a chosen k-point, with the principal masses and the density-of-states mass.
  `calc.get_effective_mass(kpoint)`, notebook
  [29](notebooks/29_effective_mass_and_angular_momenta.ipynb).
- **Starting from an all-electron ground state**: Elk's converged density, read
  from its run directory and put on this grid as the starting density, which
  shows where a pseudopotential density is allowed to differ from the real one.
  `calc.get_elk_seed(directory)`, notebook
  [42](notebooks/42_all_electron_start.ipynb).

### Choosing the physics of the run

The functional, the pseudopotentials and a Hubbard correction are chosen in the
input file, as in `pw.x`. The correction that changes the physics most is the
onsite $U$, in Dudarev's form

$$
E_U = \frac{U}{2}\sum_{I,\sigma}\mathrm{Tr}\bigl[\,n^{I\sigma}\,(1 - n^{I\sigma})\,\bigr],
$$

a penalty on fractional occupation of the correlated shell, which drives its
occupations to 0 and 1 and opens the gap of an oxide that LSDA leaves metallic.
Liechtenstein's full rotationally invariant form with $J$ is selected on the same
card. `HUBBARD` card, notebook [13](notebooks/13_dft_plus_u.ipynb).

- **Pseudopotentials**: norm-conserving, ultrasoft and PAW datasets in UPF v2,
  read from the names on the card. `ATOMIC_SPECIES`, notebook
  [04](notebooks/04_ultrasoft_and_paw.ipynb).
- **Functionals**: LDA (Perdew-Zunger, Perdew-Wang) and GGA (PBE, revPBE, PBEsol),
  on the grid and inside a PAW sphere. The functional is taken from the datasets'
  headers unless the input overrides it, and one that is not implemented is
  refused rather than replaced. `input_dft`, notebook
  [05](notebooks/05_gradient_corrections.ipynb).
- **Band gaps from the Tran-Blaha potential**, the modified Becke-Johnson
  meta-GGA, a potential with no energy functional behind it, which takes
  silicon's gap from LDA's 0.49 eV to 1.13 against an experimental 1.17; forces,
  stress and response are refused because the total is not variational.
  `input_dft = 'tb09'`, notebook [24](notebooks/24_tran_blaha_band_gaps.ipynb).
- **Tensor moments of the correlated shell**: the occupation matrix in an
  orthonormal basis of multipoles, in which the charge, the spin moment and
  $\mathbf L\cdot\mathbf S$ are single components, one of which can be held fixed
  to select an orbital ordering. `TENSOR_MOMENTS` card.
- **Around-mean-field double counting**, the alternative to the fully localised
  limit: the shell's mean occupation is subtracted before the interaction, so a
  uniformly filled shell is corrected by exactly nothing.
  `hubbard_double_counting = 'amf'`.
- **Slater integrals from the orbital**: $F^0$, $F^2$, $F^4$ and $J$ computed from
  the shell's own radial function with a screened Coulomb kernel, so one chosen
  $U$ fixes them all in place of an atomic table. `hubbard_slater = 'yukawa'`,
  notebook [13](notebooks/13_dft_plus_u.ipynb).
- **Van der Waals dispersion**, Grimme's D2 pair sum
  $-s_6\sum_{I<J} C_6^{IJ}\,f_{\mathrm{damp}}(R_{IJ})\,R_{IJ}^{-6}$, in the energy,
  the forces, the stress and the elastic constants; it is what binds bilayer
  graphene where PBE alone has no minimum. `vdw_corr = 'grimme-d2'`, notebook
  [22](notebooks/22_van_der_waals.ipynb).

### Structure and mechanics

The force on an atom is the derivative of the total energy with respect to its
position, taken at the converged wavefunctions,

$$
\mathbf F_I = -\frac{\partial E}{\partial \boldsymbol\tau_I},
$$

which at self-consistency is exact, since the energy is stationary in the
wavefunctions, and which for an ultrasoft or PAW dataset includes the term from
a basis that moves with the atom. In Ry/bohr, on norm-conserving, ultrasoft and
PAW datasets, unpolarized, collinear and spin-orbit alike. `calc.get_forces()`,
whose `.forces` is `(nat, 3)`, notebook
[09](notebooks/09_forces_and_relaxation.ipynb).

- **Structural relaxation**: the atoms moved downhill by BFGS until the forces
  vanish, and **variable-cell relaxation**, the cell and the atoms together at
  an applied pressure. `calc.get_relax()` and `calc.get_relax(variable_cell=True)`,
  notebooks [09](notebooks/09_forces_and_relaxation.ipynb) and
  [23](notebooks/23_variable_cell_relaxation.ipynb).
- **Stress tensor and pressure**, $\sigma_{ij} = -\Omega^{-1}\,\partial E/\partial\varepsilon_{ij}$,
  the strain derivative of the energy at fixed wavefunctions, in Ry/bohr³ and
  kbar. `calc.get_stress()`, notebook [15](notebooks/15_stress.ipynb).
- **The strain response**, $\partial\psi/\partial\varepsilon$ and
  $\partial n/\partial\varepsilon$, and the deformation potentials
  $\partial\epsilon_{n\mathbf k}/\partial\varepsilon_{ij}$ that follow.
  `calc.get_strain_response()`, notebook [21](notebooks/21_electrostriction.ipynb).
- **Elastic constants**, $C_{ijkl} = \partial\sigma_{ij}/\partial\varepsilon_{kl}$,
  with the compliances and the bulk modulus; clamped-ion, for insulators on
  norm-conserving datasets. `calc.get_elastic_constants()`, notebook
  [21](notebooks/21_electrostriction.ipynb).
- **Electrostriction and the elasto-optic tensor**: the quadratic coupling of a
  field to a strain, the coefficients $m$, $q$, $M$ and $Q$, and
  $\partial\chi_{ij}/\partial\varepsilon_{kl}$, how a strain changes the
  dielectric response, which is what makes a squeezed crystal birefringent.
  `calc.get_electrostriction()`, whose `.photoelastic` is the elasto-optic
  tensor, notebook [21](notebooks/21_electrostriction.ipynb).
- **Piezoelectric tensor**, $e_{k,ij} = \partial P_k/\partial\varepsilon_{ij}$, the
  polarization a strain induces, which is also the stress a field induces;
  clamped-ion, for insulators without a spontaneous polarization.
  `calc.get_piezoelectric_tensor()`, and `calc.get_piezoelectric_kmesh_ladder()`
  for its convergence with the k-mesh, which is far slower than the energy's;
  notebook [28](notebooks/28_piezoelectricity.ipynb).

### Magnetism

A moment can be collinear (`nspin = 2`), a vector field with its own magnetic
symmetry group (`noncolin`), or coupled to the orbital motion by spin-orbit
coupling (`lspinorb`), and a spin spiral of any pitch runs in the unit cell
without a supercell. The quantity that turns a set of such runs into a spin
model is the exchange, read off the energy of a spiral against its wavevector,

$$
E(\mathbf q) - E(0) = m^2 \sum_{\mathbf R} J(\mathbf R)\,\bigl[1 - \cos(\mathbf q\cdot\mathbf R)\bigr],
$$

fitted over neighbour shells, with the fit residual saying how well a Heisenberg
model describes the surface. `calc.get_spiral_scan(wavevectors)`, notebook
[12](notebooks/12_spin_spirals.ipynb).

- **Collinear magnetism**, with one Fermi level or two, the second when
  `tot_magnetization` constrains the moment; a compensated magnet whose
  sublattices are related by a rotation, an altermagnet, can be stated on one
  species. `nspin = 2`, notebook [07](notebooks/07_spin_polarization.ipynb).
- **Magnetism as a vector**,
  $\mathbf m(\mathbf r) = \sum_{n\mathbf k} f_{n\mathbf k}\,\psi^\dagger_{n\mathbf k}\,\boldsymbol\sigma\,\psi_{n\mathbf k}$,
  with the magnetic symmetry group. `noncolin`, notebook
  [11](notebooks/11_noncollinear_magnetism_and_fields.ipynb).
- **Spin-orbit coupling**, two-component spinors and $j$-resolved projectors from
  a fully relativistic dataset. `lspinorb`, notebook
  [08](notebooks/08_spin_orbit_coupling.ipynb).
- **The moment on each atom**, the charge and the magnetization integrated in a
  sphere around every atom, at convergence and at every iteration and ionic
  step; it is what separates a compensated magnet from the nonmagnetic state it
  can collapse into. `calc.get_scf().site_moments`, notebook
  [43](notebooks/43_magnetic_textures.ipynb).
- **Magnetic fields and constrained moments**: a uniform Zeeman field
  (`B_field`), or a moment held at a size or a direction by a penalty.
  `constrained_magnetization`, notebook
  [11](notebooks/11_noncollinear_magnetism_and_fields.ipynb).
- **Magnetic fields inside one atom's sphere**, and a field that fades away as
  the run converges. `LOCAL_MAGNETIC_FIELDS` card.
- **Spin spirals** at any wavevector $\mathbf q$, by the generalized Bloch
  theorem: the up component at $\mathbf k + \mathbf q/2$ and the down at
  $\mathbf k - \mathbf q/2$, on norm-conserving, ultrasoft and PAW datasets. Needs
  `nosym`, and spin-orbit coupling is refused. `spiral_q`, notebook
  [12](notebooks/12_spin_spirals.ipynb).
- **Relaxing the spiral wavevector**, $\mathrm dE/\mathrm d\mathbf q$ walked down
  to the ground-state pitch by BFGS. `calc.get_spiral_relaxation()`, notebook
  [14](notebooks/14_spiral_relaxation.ipynb).
- **Orbital, spin and total angular momentum on each atom**, $\langle L\rangle$,
  $\langle S\rangle$ and $\langle J\rangle$, which is where the orbital moment of a
  spin-orbit magnet sits. `calc.get_angular_momenta()`, notebook
  [29](notebooks/29_effective_mass_and_angular_momenta.ipynb).
- **Orbital magnetization of the cell**, $\mathbf M_{\mathrm{orb}}$ by the modern
  theory, the circulating half of a magnet's moment that no integral over the
  cell can give; needs spin-orbit coupling, broken time reversal and a gap.
  `calc.get_orbital_magnetization()`, notebook
  [39](notebooks/39_orbital_magnetization.ipynb).
- **Magnetocrystalline anisotropy** by the force theorem,
  $E_{\mathrm{MAE}} = \sum_{\mathrm{occ}}\epsilon(\hat{\mathbf n}_1) - \sum_{\mathrm{occ}}\epsilon(\hat{\mathbf n}_2)$,
  the band-energy sums of one diagonalisation per direction with spin-orbit
  coupling on, over a density converged without it. `calc.get_anisotropy(spinor)`,
  where `spinor` is the same crystal as a spin-orbit calculator;
  `calc.get_force_theorem(spinor)` is one direction's leg, and
  `calc.get_first_order_soc(spinor)` the spin-orbit term's expectation value at
  coupling-free states, which is the first-order estimate the theorem is often
  mistaken for. Notebook [36](notebooks/36_magnetic_anisotropy.ipynb).
- **Relaxed magnetocrystalline anisotropy**, the same energy from total energies,
  one self-consistent noncollinear run per direction with the density free to
  respond; it allows a Hubbard $U$ and reports how far each moment drifted from
  where it was started. `calc.get_relaxed_anisotropy()`, notebook
  [36](notebooks/36_magnetic_anisotropy.ipynb).
- **Magnetic torque**, $-\mathrm dF/\mathrm d\theta$, the anisotropy from one
  angle rather than a difference of two; for $E(\theta) = K_1\sin^2\theta$ the
  torque at 45 degrees is $-K_1$. `calc.get_torque(spinor)`, notebook
  [36](notebooks/36_magnetic_anisotropy.ipynb). With every spin turned by one
  rotation it is a vector, the torque for the three generators at once,
  `calc.get_orientation_torque(spinor, rotation=R)`.
- **Source-free exchange-correlation field, and the torque it exerts**: the
  longitudinal part of $\mathbf B_{xc}$ projected out so that
  $\nabla\cdot\mathbf B_{xc} = 0$, the one thing that lets a local functional
  turn a moment at all, with $\int \mathbf m\times\mathbf B_{xc}$ as the measure
  of how far a texture is from stationary. The field is selected by `nosource`;
  `calc.get_exchange_torque()` returns the torque.
- **Magnons**: the transverse spin susceptibility $\chi^{+-}(\mathbf q,\omega)$,
  whose pole below the Stoner continuum of independent spin flips is the spin
  wave, and the dispersion $\omega(\mathbf q)$ read off it, with the Goldstone
  theorem's $\omega(0) = 0$ as the calculation's own error bar.
  `calc.get_magnon_dispersion(qpoints, frequencies)`, and
  `calc.get_spin_susceptibility(q, frequencies)` for one wavevector; notebook
  [38](notebooks/38_magnons.ipynb).

**Getting a hard calculation to converge.** Anderson or Broyden mixing with
Kerker or local Thomas-Fermi preconditioning, Elk's adaptive scheme for an SCF
that crawls rather than oscillates, or a residual solver with its own Jacobian
that reaches magnetic solutions no mixer does, are all options of
`calc.get_scf()` (`mixing_mode`, `scf_solver`), and a long run checkpoints
itself (`checkpoint_dir`, `max_seconds`) so that a resubmitted job continues
rather than starting over. A converged run seeds another across a change of spin
regime, a nonmagnetic density starting a magnetic run or a collinear one a
noncollinear run, with `calc.with_spin()`. A magnetic texture is stated one atom
at a time with `calc.with_moments()` or the `STARTING_MOMENTS` card, which is
what a helix, a cycloid or a Néel state needs and which decides the magnetic
symmetry group; a texture that is not the ground state is held while the rest of
the density relaxes, by a per-atom penalty (`constrained_magnetization = 'atomic'`)
or by Elk's per-site feedback field (`'atomic fsm'`, `'atomic fsm direction'`),
which converges to a genuine stationary point where a penalty leaves a residual.
Notebooks [17](notebooks/17_reaching_self_consistency.ipynb),
[18](notebooks/18_continuing_a_calculation.ipynb) and
[43](notebooks/43_magnetic_textures.ipynb).

### Vibrations and dielectric response

The polarization a static electric field induces with the ions held fixed, and
the force the same field exerts on each ion,

$$
\epsilon^{\infty}_{ij} = \delta_{ij} + 4\pi\,\frac{\partial P_i}{\partial \mathcal E_j},
\qquad
Z^{\ast}_{I,ij} = \frac{\partial F_{Ij}}{\partial \mathcal E_i},
$$

the second in units of the electron charge. Both are second derivatives of the
same energy under the same field, so one calculation returns both;
$\epsilon^\infty$ is what infrared reflectivity measures above the phonon
frequencies, and $Z^{\ast}$ is what splits the longitudinal from the transverse
optical mode of a polar crystal. For insulators, on norm-conserving, ultrasoft
and PAW datasets, unpolarized, collinear, and for a spin-orbit insulator with no
net moment. `calc.get_dielectric_tensor()`, whose `.epsilon` is `(3, 3)` and
`.born_charges` is `(nat, 3, 3)`, and `calc.get_born_charges()` for the charges
alone from the same solve; notebook [19](notebooks/19_linear_response.ipynb).

- **Phonons at $\Gamma$**: the force constants
  $C_{I\alpha,J\beta} = \partial^2 E/\partial\tau_{I\alpha}\,\partial\tau_{J\beta}$
  and the frequencies of the zone-centre modes, the eigenvalues of
  $C_{I\alpha,J\beta}/\sqrt{M_I M_J}$, in cm⁻¹ with an unstable mode reported as
  a negative number; insulators and metals, and an ultrasoft or PAW metal is
  refused. `calc.get_phonons()`, whose `.frequencies` is `(3 nat,)`, notebook
  [20](notebooks/20_phonons.ipynb).
- **Phonons at $\mathbf q \neq 0$**: the dynamical matrix at one wavevector, from
  the perturbed states on their own $\mathbf k + \mathbf q$ plane-wave sphere;
  norm-conserving insulators on the full grid. `calc.get_phonons_at_q(q)`,
  notebook [20](notebooks/20_phonons.ipynb).
- **Raman tensors**, $\partial\epsilon_{ij}/\partial\tau_{I\alpha}$, how the
  dielectric tensor changes when an atom moves. `calc.get_raman_tensors()`,
  notebook [26](notebooks/26_raman_and_infrared_spectra.ipynb).
- **Raman and infrared spectra**: the activity and depolarisation ratio of each
  mode, the Raman tensors and the Born charges contracted with the eigenvectors,
  which is what a spectrum plots. `calc.get_vibrational_spectrum()`, notebook
  [26](notebooks/26_raman_and_infrared_spectra.ipynb).
- **LO-TO splitting and the static dielectric constant**,
  $\epsilon^0_{ij} = \epsilon^\infty_{ij} + (4\pi e^2/\Omega)\sum_\nu p^\nu_i p^\nu_j/\omega_\nu^2$:
  the macroscopic field a polar mode builds raises the longitudinal branch and
  screens a static field, and the two are tied together by Lyddane-Sachs-Teller.
  `calc.get_vibrational_spectrum(loto_direction=...)`, notebook
  [26](notebooks/26_raman_and_infrared_spectra.ipynb).

### Optical and nonlinear response

The absorption spectrum of an insulator, with the exciton that binds below the
gap, from time-dependent density-functional theory: the response of the
interacting electrons is the Dyson equation on the independent-particle
response,

$$
\chi(\omega) = \chi_0(\omega) + \chi_0(\omega)\,\bigl[v + f_{xc}(\omega)\bigr]\,\chi(\omega),
$$

with local-field effects included, and $\mathrm{Im}\,\epsilon_M(\omega)$, the
macroscopic dielectric function, the inverse of the head of $\epsilon^{-1}$, is
the spectrum. The bootstrap kernel is parameter-free and binds the exciton that no
adiabatic local kernel can. `calc.get_absorption(frequencies)`, notebook
[27](notebooks/27_excitons_and_tddft.ipynb).

- **Optical conductivity**, $\sigma_{ab}(\omega)$, interband plus a Drude term,
  whose antisymmetric part needs magnetism and spin-orbit coupling together and
  gives the magneto-optical **Kerr angle** and the **anomalous Hall
  conductivity**. `calc.get_optical_conductivity()`, notebook
  [30](notebooks/30_magneto_optics.ipynb).
- **Shift current**, $\sigma^{abc}(0;\omega,-\omega)$, the bulk photovoltaic
  effect: the direct current a crystal with no inversion centre carries under
  uniform illumination, with no junction and no built-in field.
  `calc.get_shift_current()`, notebook [32](notebooks/32_shift_current.ipynb).
- **Second-harmonic generation**, $\chi^{(2)}_{abc}(-2\omega;\omega,\omega)$, how
  much of the light shone on a crystal comes back at twice the frequency, a
  polar rank-3 tensor that vanishes in any centrosymmetric crystal.
  `calc.get_shg()`, notebook [33](notebooks/33_second_harmonic_generation.ipynb).

### Topology and polarization

The Berry curvature of the occupied bands and its integral over the zone,

$$
\Omega_z(\mathbf k) = \nabla_{\mathbf k}\times \mathbf A(\mathbf k),
\qquad
\mathbf A(\mathbf k) = i\sum_{n\,\mathrm{occ}} \langle u_{n\mathbf k}\rvert\nabla_{\mathbf k} u_{n\mathbf k}\rangle,
\qquad
C = \frac{1}{2\pi}\int_{\mathrm{BZ}} \Omega_z(\mathbf k)\, d^2k,
$$

an integer that counts the chiral edge states of a two-dimensional insulator
and is its quantized Hall conductance, $\sigma_{xy} = C\,e^2/h$. It comes out an
exact integer on any k-mesh, so the mesh sets the resolution of the map and not
the answer. On norm-conserving, ultrasoft and PAW datasets. `calc.get_chern()`,
and `calc.get_berry_curvature()` for the map, whose `.chern_number` is the same
integer; notebook [10](notebooks/10_topological_invariants.ipynb).

- **$\mathbb Z_2$ invariants** in 2D and 3D, by the flow of the Wannier charge
  centres and by the Fu-Kane parities, two independent routes whose agreement is
  the check. `calc.get_z2()` and `calc.get_z2_3d()`, notebook
  [10](notebooks/10_topological_invariants.ipynb).
- **Berry-phase polarization**: King-Smith and Vanderbilt's phase along one
  reciprocal lattice vector,
  $\phi = -\,\mathrm{Im}\ln\prod_{j}\det\langle u_{n\mathbf k_j}\vert u_{m\mathbf k_{j+1}}\rangle$,
  carried with the quantum it is defined modulo, and a Born charge can be read
  off a displacement. `calc.get_polarization()`, notebook
  [34](notebooks/34_electric_polarization.ipynb).
- **Magnetoelectric tensor**, $\alpha_{ij} = \partial P_i/\partial B_j$, the
  polarization a magnetic field induces, clamped-ion; needs spin-orbit coupling,
  a gap and a crystal without an inversion centre.
  `calc.get_magnetoelectric_tensor()`, notebook
  [35](notebooks/35_magnetoelectric_effect.ipynb).

### Fermi surface, diffraction and tunnelling

What a scanning-tunnelling microscope sees, by Tersoff and Hamann, is the local
density of states at the tip integrated over the bias window,

$$
I(\mathbf r, V) \propto \int_{E_F}^{E_F + eV} \rho(\mathbf r, E)\,\mathrm dE,
\qquad
\rho(\mathbf r, E) = \sum_{n\mathbf k} w_{\mathbf k}\,\lvert\psi_{n\mathbf k}(\mathbf r)\rvert^2\,\delta(E - \epsilon_{n\mathbf k}),
$$

at constant height or at constant current, and a magnetic tip reads
$[\rho + P\,\hat{\mathbf n}\cdot\mathbf m]/2$ instead, so on a noncollinear
crystal the image depends on which way the tip points. `calc.get_stm()`,
notebook [40](notebooks/40_stm_images.ipynb).

- **Tunnelling spectra**, $\mathrm dI/\mathrm dV(\mathbf r, V)$, the same sum
  sectioned the other way, one place over many biases rather than one picture at
  one bias, which is what resolves a gap, a band edge or a state inside a gap,
  with $I(V)$ beside it. `calc.get_sts()`, notebook
  [45](notebooks/45_imaging_a_modulation.ipynb).
- **Vertical tunnelling transport through a 2D material**,
  $T(\mathbf r;E) = \int_{\mathrm{plane}} \lvert G(\mathbf r,\mathbf r';E)\rvert^2\,\mathrm d^2r'$:
  an electron enters at a point above the sheet and leaves into the plane
  below, so the current is set by the nonlocal Green's function between the two
  rather than by the density of states at the tip, and either electrode can be
  magnetic, which makes the map a tunnelling magnetoresistance image.
  `calc.get_vertical_transport()`, notebook
  [41](notebooks/41_vertical_transport.ipynb).
- **Which k-points the tunnelling current comes out of**: the same junction with
  a planar tip, so the map collapses to one weight per $\mathbf k$; a state at
  large $\lvert\mathbf k_\parallel\rvert$ decays as
  $e^{-\sqrt{\kappa_0^2 + k_\parallel^2}\,z}$, so a zone-corner pocket can carry
  most of the Fermi surface and little of the current.
  `calc.get_momentum_transport()`.
- **Fermi-surface nesting function**,
  $N(\mathbf q) = \sum_{\mathbf k}\delta(\epsilon_{\mathbf k} - E_F)\,\delta(\epsilon_{\mathbf k+\mathbf q} - E_F)$,
  how much of the Fermi surface maps onto itself when translated by $\mathbf q$,
  which is where a phonon softens, a charge-density wave opens a gap or a spin
  spiral finds its pitch. `calc.get_nesting()`, notebook
  [31](notebooks/31_fermi_surface_nesting.ipynb).
- **X-ray and magnetic structure factors**,
  $F(\mathbf H) = \int_\Omega n(\mathbf r)\,e^{i\mathbf H\cdot\mathbf r}\,\mathrm d^3r$
  and the same of $\mathbf m$, the Fourier coefficients a diffraction experiment
  measures rather than a density; valence-only, so a forbidden reflection like
  silicon's (222) is bonding charge and nothing else.
  `calc.get_structure_factors()`, notebook
  [37](notebooks/37_structure_factors.ipynb).

### Long-range modulations

A density or potential varying over tens or hundreds of unit cells, solved in
the unit cell's own states at the k-points that fold onto the long cell,

$$
\Psi(\mathbf r) = \sum_{\mathbf k \in \mathcal K_N}\ \sum_{n=1}^{n_{\mathrm{bnd}}} c_{n\mathbf k}\,\psi_{n\mathbf k}(\mathbf r),
$$

where $\mathcal K_N$ are the $N$ k-points of the unit cell that fold onto the
$N$-cell supercell's $\Gamma$, so the cost is set by the number of bands rather
than of plane waves and the self-consistency runs on the envelope alone. It is a
variational truncation of the exact supercell, converging to it as `nbnd` grows.
An applied field drives a modulation and returns the $\mathbf Q$-resolved
susceptibility; a seed hands the loop a texture, a staggered wave or a helix, as
its initial condition, and the loop keeps it if it is a solution, since nothing
finds a wave on its own from a uniform state. `calc.get_ultracell(supercell)`,
notebooks [44](notebooks/44_ultra_long_range.ipynb) and
[46](notebooks/46_a_spin_wave_that_stays.ipynb).

- **The energy of a long-range modulation**, the Kohn-Sham free energy per unit
  cell, which is what says whether a modulated state is worth its cost against
  the uniform one, and the one quantity of the method that converges with a
  sign, as a monotone upper bound on the supercell's energy.
  `calc.get_ultracell(supercell).total_energy`, notebook
  [44](notebooks/44_ultra_long_range.ipynb).
- **What a modulation looks like to a tip**: the Tersoff-Hamann image, the
  tunnelling spectrum and the vertical transmission of an ultracell, which is
  how a charge or spin density wave is actually seen. An unpolarized tip sees a
  spin density wave's square, at twice the wavevector, and a magnetic tip the
  wave itself. `calc.get_ultracell_stm()`, `calc.get_ultracell_sts()` and
  `calc.get_ultracell_transport()`, notebook
  [45](notebooks/45_imaging_a_modulation.ipynb).

## Which of these Quantum ESPRESSO and Elk also compute

Each row is a quantity from the catalogue above, the call or input variable
that asks for it, and whether the two established codes compute it as well:
**QE** is Quantum ESPRESSO (`pw.x` and its post-processing tools) and **Elk** is
the all-electron LAPW code. A tick means the quantity is there; **(✓)** means it
is there only partly, and the numbered note under the table says how; **blank
in both columns is a quantity neither code computes**, which is what tells you
whether a row is a reimplementation or an extension. The evidence under each
note, the routine or task in the other code's source, is in
[`docs/reference-codes.md`](docs/reference-codes.md).

| Quantity | How to ask for it | QE | Elk |
|---|---|:-:|:-:|
| **Total energy**, self-consistent and term by term | `calc.get_scf()` | ✓ | ✓ |
| **Band structure** | `calc.get_bands()` | ✓ | ✓ |
| **Density of states** | `calc.get_dos()` | ✓ | ✓ |
| **Projected density of states**, by atom, $l$, $m$ and $j$ | `calc.get_pdos()` | ✓ | (✓)¹ |
| **Band velocities** | `calc.get_band_velocities()` | (✓)² | |
| **Effective mass tensor** | `calc.get_effective_mass(kpoint)` | | ✓ |
| **Starting from an all-electron ground state**³ | `calc.get_elk_seed(directory)` | | |
| **Pseudopotentials**: norm-conserving, ultrasoft and PAW | `ATOMIC_SPECIES` | ✓ | |
| **Functionals**: LDA, PBE, revPBE and PBEsol | `input_dft` | ✓ | ✓ |
| **Band gaps from the Tran-Blaha potential** | `input_dft = 'tb09'` | (✓)⁴ | ✓ |
| **DFT+U**, Dudarev's and Liechtenstein's functionals | `HUBBARD` card | ✓ | ✓ |
| **Tensor moments of the correlated shell** | `TENSOR_MOMENTS` card | | ✓ |
| **Around-mean-field double counting** | `hubbard_double_counting = 'amf'` | | ✓ |
| **Slater integrals from the orbital** | `hubbard_slater = 'yukawa'` | | ✓ |
| **Van der Waals dispersion**, Grimme's D2 | `vdw_corr = 'grimme-d2'` | ✓ | |
| **Forces on the atoms** | `calc.get_forces()` | ✓ | ✓ |
| **Structural relaxation** | `calc.get_relax()` | ✓ | ✓ |
| **Variable-cell relaxation** at an applied pressure | `calc.get_relax(variable_cell=True)` | ✓ | ✓ |
| **Stress tensor and pressure** | `calc.get_stress()` | ✓ | ✓ |
| **The strain response** and the deformation potentials | `calc.get_strain_response()` | | |
| **Elastic constants** | `calc.get_elastic_constants()` | | |
| **Electrostriction coefficients** | `calc.get_electrostriction()` | | |
| **Elasto-optic tensor** | `calc.get_electrostriction().photoelastic` | | |
| **Piezoelectric tensor** | `calc.get_piezoelectric_tensor()` | | ✓ |
| **Collinear magnetism**, with one Fermi level or two | `nspin = 2` | ✓ | ✓ |
| **Magnetism as a vector**, with the magnetic symmetry group | `noncolin` | ✓ | ✓ |
| **Spin-orbit coupling** | `lspinorb` | ✓ | ✓ |
| **The moment on each atom** | `calc.get_scf().site_moments` | ✓ | ✓ |
| **Magnetic fields and constrained moments** | `constrained_magnetization` | ✓ | ✓ |
| **Magnetic fields inside one atom's sphere**, and a fading field | `LOCAL_MAGNETIC_FIELDS` card | | ✓ |
| **Spin spirals** at any wavevector | `spiral_q` | | ✓ |
| **Relaxing the spiral wavevector** | `calc.get_spiral_relaxation()` | | |
| **$E(\mathbf q)$ and the Heisenberg exchange constants** | `calc.get_spiral_scan(wavevectors)` | | |
| **Orbital, spin and total angular momentum on each atom** | `calc.get_angular_momenta()` | (✓)⁵ | ✓ |
| **Orbital magnetization of the cell** | `calc.get_orbital_magnetization()` | ✓ | ⁶ |
| **Magnetocrystalline anisotropy**, by the force theorem | `calc.get_anisotropy(spinor)` | ✓ | (✓)⁷ |
| **Relaxed magnetocrystalline anisotropy** | `calc.get_relaxed_anisotropy()` | (✓)⁸ | ✓ |
| **Magnetic torque** | `calc.get_torque(spinor)`, `calc.get_orientation_torque(spinor)` | | |
| **Source-free exchange-correlation field**, and its torque | `calc.get_exchange_torque()` | | ✓ |
| **Magnons** and the transverse spin susceptibility | `calc.get_magnon_dispersion(qpoints, frequencies)` | (✓)⁹ | ✓ |
| **Dielectric constant** and **Born effective charges** | `calc.get_dielectric_tensor()` | ✓ | ✓ |
| **Phonons at $\Gamma$** | `calc.get_phonons()` | ✓ | ✓ |
| **Phonons at $\mathbf q \neq 0$** | `calc.get_phonons_at_q(q)` | ✓ | ✓ |
| **Raman tensors** | `calc.get_raman_tensors()` | (✓)¹⁰ | |
| **Raman and infrared spectra** | `calc.get_vibrational_spectrum()` | ✓ | |
| **LO-TO splitting and the static dielectric constant** | `calc.get_vibrational_spectrum(loto_direction=...)` | ✓ | (✓)¹¹ |
| **Optical absorption spectra with excitons** | `calc.get_absorption(frequencies)` | | ✓ |
| **Optical conductivity**, the Kerr angle and the anomalous Hall conductivity | `calc.get_optical_conductivity()` | (✓)¹² | ✓ |
| **Shift current** | `calc.get_shift_current()` | ¹³ | |
| **Second-harmonic generation** | `calc.get_shg()` | (✓)¹⁴ | ✓ |
| **Berry curvature and Chern numbers** | `calc.get_chern()` | | |
| **$\mathbb{Z}_2$ invariants** in 2D and 3D | `calc.get_z2()` | | |
| **Berry-phase polarization** | `calc.get_polarization()` | ✓ | ✓ |
| **Magnetoelectric tensor** | `calc.get_magnetoelectric_tensor()` | | ✓ |
| **Scanning-tunnelling microscopy images** | `calc.get_stm()` | (✓)¹⁵ | (✓)¹⁵ |
| **Tunnelling spectra** $\mathrm{d}I/\mathrm{d}V$ | `calc.get_sts()` | | |
| **Vertical tunnelling transport through a 2D material** | `calc.get_vertical_transport()` | (✓)¹⁶ | |
| **Which k-points the tunnelling current comes out of**¹⁷ | `calc.get_momentum_transport()` | | |
| **Fermi-surface nesting function** | `calc.get_nesting()` | | ✓ |
| **X-ray and magnetic structure factors** | `calc.get_structure_factors()` | | ✓ |
| **Ultra long-range modulations**, the ultracell | `calc.get_ultracell(supercell)` | | ✓ |
| **The energy of a long-range modulation** | `calc.get_ultracell(supercell).total_energy` | | |
| **What a modulation looks like to a tip** | `calc.get_ultracell_stm()` | | |

Where a tick is qualified, in one sentence each; the routines behind them are in
[`docs/reference-codes.md`](docs/reference-codes.md):

- ¹ Elk's partial density of states is resolved over $(l, m)$ and over spin but
  not over $j$; `projwfc.x` has the $j$ resolution and is what this is checked
  against.
- ² `fermi_velocity.x` finite-differences eigenvalues and reports only the
  magnitude.
- ³ Neither code reads the other's ground state; what is here is the crossing
  from a muffin-tin density into a plane-wave one, which neither has a reason to
  implement.
- ⁴ Quantum ESPRESSO reaches `tb09` only through libxc, with a zero Laplacian and
  the functional's coefficient never set, so what it runs under that name is a
  different functional.
- ⁵ `lorbm` gives the cell's orbital magnetization and nothing per atom.
- ⁶ Elk has no orbital magnetization by the modern theory; its moments are
  integrals of the magnetization over the muffin tins and the interstitial.
- ⁷ Elk's `mae` re-converges a ground state per direction, which is the relaxed
  row's method and not the force theorem.
- ⁸ `pw.x` can converge a spin-orbit run per direction and print its total
  energy, but has no routine that sets the directions up, holds the k-set fixed
  and reports how far a moment drifted.
- ⁹ turboMagnon propagates a response vector and never forms $\chi_0$, so there
  is no Dyson equation and no pole; Elk's tasks 330 and 331 do exactly this.
- ¹⁰ `ph.x` refuses a gradient-corrected functional here, where this does not.
- ¹¹ Elk adds the same non-analytic term and computes Born charges, but reads its
  static dielectric tensor in rather than summing the modes into it.
- ¹² `epsilon.x` forms the dielectric tensor but no conductivity and no Kerr
  angle, refuses ultrasoft datasets, and builds its dipole from momentum matrix
  elements, which is not $[H, \mathbf r]$ for a nonlocal pseudopotential.
- ¹³ Wannier90's `berry_task = 'sc'` computes a shift current, but it is a
  separate code bundled beside Quantum ESPRESSO and needs a wannierisation first;
  nothing in `PW`, `PP` or `PHonon` computes a photocurrent, and Elk has none.
- ¹⁴ `el_opt.f90` computes the static electro-optic tensor and not
  $\chi^{(2)}(-2\omega;\omega,\omega)$; Elk's task 125 is the reference.
- ¹⁵ Both codes compute the charge image and neither the spin-polarized one;
  constant current is Quantum ESPRESSO's alone and is quantised to the grid
  spacing there.
- ¹⁶ `PWCOND` is a Landauer transmission between two semi-infinite crystalline
  leads, one conductance per energy, with no point contact and so no map.
- ¹⁷ `pw.x` has nothing of the kind, `PWCOND` is the different geometry of the
  note above, and Elk's Fermi-surface tasks write bands for a plotting program
  rather than weighting them by anything.

The variants under each row, which smearing or tetrahedron method fixes the
occupations, which projectors DFT+U uses, which constraint scheme holds a
moment, are chosen with the same input variables as in `pw.x` where it has them.

## Is it right?

That is the question the project is organised around. Where Quantum ESPRESSO
computes the same quantity, the same input is run through both codes and the
numbers are compared; its test suite ships reference outputs, and `pytest`
compares against them:

```bash
pip install -e ".[dev]"
python3 -m pytest
```

Most of those tests need Quantum ESPRESSO's `test-suite` directory, which is
not shipped here, and skip cleanly without it; the ultrasoft, PAW and PBE
references were generated once with `pw.x` and are committed under
`tests/data/qe/`.

| | agrees to |
|---|---|
| total energy, silicon, term by term | 1e-9 Ry |
| band structures | 0.0002 eV |
| forces, term by term | 2e-5 Ry/bohr |
| stress | 2.7e-7 Ry/bohr³ |
| the dielectric constant, and the Born effective charges | 1.2e-4; every digit `ph.x` prints |
| phonons at $\Gamma$, silicon and a metal | 0.05 and 0.0019 cm⁻¹ |

The per-feature figures, for ultrasoft and PAW, the functionals, spin and
spin-orbit coupling, DFT+U, relaxation, the Raman and infrared spectra and the
rest, are in the "Accuracy summary" of [`docs/features.pdf`](docs/features.pdf).

A quantity neither code computes has no such reference, so it is pinned instead
by a statement the answer has to satisfy independently of how it was computed: a
Chern number that has to come out an exact integer, a spin spiral that has to
reproduce the supercell calculation of the same magnetic order (it does, to
1e-12 Ry), a derivative that has to match a finite difference of the thing it is
the derivative of. Where another code does compute it, that is used instead:
LiF's excitonic peak comes out at 14.05 eV against the 13.67 eV of Elk, whose
example it is. Where a second, independent route to the same number exists,
both are computed and compared.

## What it refuses, and what it does not do

**Anything not implemented is refused with an error naming what is**, rather
than quietly replaced by something else. That applies to combinations as well as
to features, so a run that starts is one whose physics is all there, and the
refusals of each quantity are the refusal notes of the guide.

**Not yet:** a phonon dispersion (one wavevector works; the star of $\mathbf q$
and the Fourier interpolation do not), exact exchange, real-time propagation.

**Substituted with a warning rather than refused:** `K_POINTS gamma` stores one
plane wave of each $(\mathbf G, -\mathbf G)$ pair, which halves every array a
band lives in and is what lets a large molecule or slab fit in memory, and it
agrees with an explicit k = 0 on the whole sphere to round-off. For an ultrasoft
or PAW dataset, a run that uses symmetry, and a spinor or spiral run, the whole
sphere is used instead and the run says so, which is the same physics at twice
the storage.

If your calculation needs any of those, use Quantum ESPRESSO. This is not a
replacement for it, and on one core it is slower, by about 2.2 times per SCF
iteration as the median over ten cases of different physics and between 1.3 and
3.5 times across them, measured against a `pw.x` linked to an optimised BLAS
(`PERFORMANCE.md`).

## Where to read more

- [`docs/features.pdf`](docs/features.pdf), the user guide: every capability, the
  equation behind it, a snippet that runs it, what it was validated against, and
  what it refuses. The source is `docs/features.tex`; rebuild it with
  `xelatex docs/features.tex`, twice for the table of contents.
- [`notebooks/README.md`](notebooks/README.md), the worked examples indexed by
  the property you want to compute, each executed and committed with its output
  so that it reads without being run. Start with `00_the_calculator` and
  `25_your_own_crystal`.
- `benchmarks/`, ready-to-run input files from a two-atom silicon cell up to a
  sixteen-atom one, and [`performance/README.md`](performance/README.md) for
  running the same inputs through `pw.x` and through this code side by side.

## License

GPL v3 or later, see [LICENSE](LICENSE). Quantum ESPRESSO is itself GPL, and
this code was written by reading it.

The pseudopotential files under `tests/data/pseudo/` come from the Quantum
ESPRESSO pseudopotential library and carry their own terms.
