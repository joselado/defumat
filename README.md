# defumat

A plane-wave density-functional theory code written in Python and JAX, in which
the derivatives are taken of the energy itself rather than derived by hand.

It is inspired by two established codes and benchmarked against both. **Quantum
ESPRESSO** is where the plane-wave machinery comes from, closely enough that
defumat reads its input files: you give it the same file you would give `pw.x`
and it runs the calculation itself, with no Fortran underneath. **Elk**, the
all-electron LAPW code, is where a second family of quantities comes from —
spin spirals without a supercell, magnetic fields inside a single atom's sphere,
effective masses, the Fermi-surface nesting function, second-harmonic
generation, X-ray and magnetic structure factors — none of which `pw.x`
computes at all.

Because the whole compute path is differentiable, a quantity that is usually a
second implementation is here a derivative of the first. The forces, the stress,
the phonons, the dielectric response and the third derivatives above them are
obtained by differentiating the total energy, so each agrees with the energy it
came from by construction rather than by transcription.

```
total energy   QE  -63.36038036 Ry
          defumat  -63.36038036 Ry
```

That is an eight-atom silicon cell, agreeing to 3.5e-9 Ry. Both codes are
references and not only influences: the agreement is checked automatically, term
by term, against Quantum ESPRESSO's own reference outputs for around a hundred
cases, and against Elk for the quantities Elk computes and `pw.x` does not.

On top of the two there are things neither of them computes — relaxing a spin
spiral's wavevector down $\mathrm{d}E/\mathrm{d}\mathbf{q}$, Chern numbers and
$\mathbb{Z}_2$ invariants, the strain response and the deformation potentials,
elastic and electrostriction constants, the magnetic torque. The table below
ticks off, quantity by quantity, what Quantum ESPRESSO and Elk compute as well;
a row blank in both columns is one neither has, and is pinned by an identity or
an independent second route rather than by a reference output.

## Capabilities at a glance

Everything the code computes, grouped by what kind of calculation it is. The
table below says, quantity by quantity, which input variable or entry point asks
for it, what it refuses, and whether Quantum ESPRESSO and Elk compute it too.

**Ground state**

- **Total energies**, self-consistently and broken down term by term, for
  insulators and for metals
- **Pseudopotentials**: norm-conserving, ultrasoft and PAW
- **Functionals**: LDA and GGA — Perdew-Zunger, Perdew-Wang, PBE, revPBE, PBEsol
- **Band gaps from the Tran-Blaha potential**, the modified Becke-Johnson
  meta-GGA
- **DFT+U**, Dudarev's simplified and Liechtenstein's full rotationally-invariant functionals
- **Van der Waals dispersion** — Grimme's D2 pair correction
- **Reaching self-consistency** — mixing, preconditioning, and a residual solver
  that reaches magnetic solutions no mixer does
- **Continuing one run from another across a change of spin regime**

**Electronic structure**

- **Band structures** along a path through the Brillouin zone
- **Densities of states**, by smearing or by tetrahedra
- **Projected densities of states**, resolved by atom, by $l$ and by $m$
- **Band velocities** $\partial\epsilon_n/\partial\mathbf{k}$, with the nonlocal
  pseudopotential's own term
- **Effective mass tensor** $m^{\ast}_{ij}$ at any k-point

**Structure and mechanics**

- **Forces on the atoms**
- **Structural relaxation**
- **Variable-cell relaxation**, the cell and the atoms together at an applied
  pressure
- **Stress tensor and pressure**
- **The strain response** and the deformation potentials
- **Elastic constants** $C_{ijkl}$, with the compliances and the bulk modulus
- **Electrostriction coefficients** $m$, $q$, $M$ and $Q$, and the
  **elasto-optic tensor** $\partial\chi_{ij}/\partial\varepsilon_{kl}$ they are
  obtained from
- **Piezoelectric tensor** $e_{k,ij}$

**Collinear magnetism**

- **Magnetism**, collinear, with one Fermi level or two
- **Magnetic fields and constrained moments** — a uniform field and four ways of
  holding a moment where you put it

**Noncollinear magnetism and spin-orbit coupling**

- **Magnetism as a vector**, with the magnetic symmetry group
- **Spin-orbit coupling** — two-component spinors and $j$-resolved projectors
- **Magnetic fields inside one atom's sphere**, and a field that fades away as
  the run converges
- **Spin spirals** at any wavevector, without a supercell
- **Relaxing the spiral wavevector** down $\mathrm{d}E/\mathrm{d}\mathbf{q}$ to
  the ground-state pitch
- **$E(\mathbf{q})$ and the Heisenberg exchange constants** $J(\mathbf{R})$ — a
  spiral scan fitted over neighbour shells, which is how a spiral scan becomes
  a spin model
- **Orbital, spin and total angular momentum on each atom** —
  $\langle L\rangle$, $\langle S\rangle$, $\langle J\rangle$
- **The cell's orbital magnetization** $\mathbf{M}_\mathrm{orb}$, by the modern
  theory — the circulating half of a magnet's moment, which no integral over
  the unit cell can give
- **Magnetocrystalline anisotropy**, by the force theorem
- **Magnetic torque** $-\mathrm{d}F/\mathrm{d}\theta$, the anisotropy from one
  angle rather than a difference of two
- **Magnons** — the spin-wave dispersion $\omega(\mathbf{q})$, from the pole of
  the transverse spin susceptibility $\chi^{+-}(\mathbf{q},\omega)$

**Vibrations and dielectric response**

- **Phonons at $\Gamma$** — the force constants and their frequencies
- **Dielectric constant** $\epsilon^\infty$ and **Born effective charges**
- **Raman tensors** $\partial\epsilon_{ij}/\partial\tau$
- **Raman and infrared spectra** — the per-mode activities and depolarisation
  ratios
- **LO-TO splitting and the static dielectric constant**

**Topology and polarization**

- **Berry curvature and Chern numbers**, and a smooth $\Omega(\mathbf{k})$ map
- **$\mathbb{Z}_2$ invariants** in 2D and 3D, by Wannier charge centres and by
  parities
- **Berry-phase polarization**
- **Magnetoelectric tensor** $\alpha_{ij} = \partial P_i/\partial B_j$

**Optical and nonlinear response**

- **Optical absorption spectra with excitons**, from TDDFT with local-field
  effects
- **Optical conductivity tensor** $\sigma_{ab}(\omega)$, the magneto-optical
  **Kerr angle** and the **anomalous Hall conductivity**
- **Shift current** $\sigma^{abc}(0;\omega,-\omega)$ — the bulk photovoltaic
  effect
- **Second-harmonic generation** $\chi^{(2)}(-2\omega;\omega,\omega)$

**Fermi surface and diffraction**

- **Fermi-surface nesting function** $N(\mathbf{q})$
- **X-ray and magnetic structure factors** $F(\mathbf{H})$

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
| **Densities of states**, by smearing or by tetrahedra | `run_dos`, `defumat dos` | ✓ | ✓ |
| **Projected densities of states** — resolved by atom, by $l$ and by $m$, with Löwdin charges and the spilling parameter | `run_pdos`, `defumat pdos` | ✓ | ✓ |
| **Forces on the atoms** — unpolarized, collinear spin and noncollinear/spin-orbit, on norm-conserving, ultrasoft and PAW. For a spinor the hand-derived cross-check has no counterpart and `method='analytic'` is refused | `compute_forces` | ✓ | ✓ |
| **Structural relaxation** — the atoms moved downhill to their equilibrium positions | `calculation = 'relax'`, `defumat relax` | ✓ | ✓ |
| **Variable-cell relaxation** — the cell and the atoms relaxed together, at an applied pressure | `calculation = 'vc-relax'`, `run_vc_relax` | ✓ | ✓ |
| **Stress tensor and pressure**, in Ry/bohr³ and kbar — the same three spin regimes as the force | `tstress = .true.`, `compute_stress`, `defumat stress` | ✓ | ✓ |
| **Magnetism**, collinear, with one Fermi level or two | `nspin = 2`, `tot_magnetization` | ✓ | ✓ |
| **Magnetism as a vector** — noncollinear, with the magnetic symmetry group | `noncolin` | ✓ | ✓ |
| **Spin-orbit coupling**, two-component spinors and $j$-resolved projectors | `lspinorb` | ✓ | ✓ |
| **Magnetic fields and constrained moments** — a uniform field, and four ways of holding a moment where you put it | `B_field`, `constrained_magnetization` | ✓ | ✓ |
| **Magnetic fields inside one atom's sphere**, and a field that fades away as the run converges | `LOCAL_MAGNETIC_FIELDS` card, `reducebf`, `constrained_magnetization = 'fsm'` | | ✓ |
| **DFT+U** — Dudarev's simplified functional with $U$, $J_0$, $\alpha$, $\beta$ and Liechtenstein's full one with $J$, $B$, $E_2$, $E_3$, selected by the card. Collinear or on a two-component **spinor** with spin-orbit coupling, where the occupation matrix is a 2x2 matrix in spin space. The intersite $V$ is refused by name | `HUBBARD` card, `noncolin`, `run_scf(starting_ns=...)` | ✓ | ✓ |
| **Tensor moments of the correlated shell** — the occupation matrix in an orthonormal basis of multipoles, where the charge, the spin moment and $\mathbf{L}\cdot\mathbf{S}$ are single components; and **holding one of them fixed**, which selects an orbital ordering the field would not find on its own | `TENSOR_MOMENTS` card, `tensor_moment_penalty` | | ✓ |
| **Around-mean-field double counting** — the alternative to the fully-localised limit: the shell's mean occupation is subtracted before the interaction, so a uniformly filled shell is corrected by exactly nothing | `hubbard_double_counting = 'amf'` | | ✓ |
| **Slater integrals from the orbital** — the interaction computed from the manifold's own all-electron radial function with a screened Coulomb kernel, so one chosen $U$ fixes $F^0$, $F^2$, $F^4$ and $J$ instead of an atomic table doing it | `hubbard_slater = 'yukawa'`, `LAMBDA` on the `HUBBARD` card | | ✓ |
| **Spin spirals** at any wavevector, without a supercell. Needs `nosym`; ultrasoft, PAW and spin-orbit coupling are refused | `spiral_q`, `defumat spiral` | | ✓ |
| **Relaxing the spiral wavevector** down $\mathrm{d}E/\mathrm{d}\mathbf{q}$ to the ground-state pitch | `relax_spiral_q` | | |
| **$E(\mathbf{q})$ and the Heisenberg exchange constants** — a spiral scan's energy against its wavevector, fitted over neighbour shells to $E(\mathbf{q}) - E(0) = m^2 \sum_{\mathbf{R}} J(\mathbf{R})\,[1 - \cos(\mathbf{q}\cdot\mathbf{R})]$, which is how a spiral scan becomes a spin model; the fit residual says how well a Heisenberg model describes the surface. $E(\mathbf{q})$ can be accumulated from $\mathrm{d}E/\mathrm{d}\mathbf{q}$ instead of read off the energies, which removes the steps a rebuilt plane-wave basis puts in the curve | `run_spiral_scan`, `heisenberg_exchange`, `Calculator.get_spiral_scan` | | |
| **Berry curvature and Chern numbers** — exact integers on any mesh, and a smooth $\Omega(\mathbf{k})$ map with the truncation of its band sum reported | `run_berry_curvature`, `method="kubo"` for the map | | |
| **$\mathbb{Z}_2$ invariants** in 2D and 3D, by Wannier charge centres *and* by parities | `run_z2`, `run_z2_3d` | | |
| **Berry-phase polarization** — King-Smith and Vanderbilt's phase along one reciprocal lattice vector, with the quantum it is defined modulo carried beside it. Norm-conserving, ultrasoft, PAW and spinor; metals, `nspin = 2` and spin spirals are refused | `lberry`/`gdir`/`nppstr`, `run_polarization`, `Calculator.get_polarization` | ✓ | ✓ |
| **Magnetoelectric tensor** $\alpha_{ij} = \partial P_i/\partial B_j$ — the polarization a magnetic field induces. Spin (Zeeman) response, clamped-ion, by a finite difference of the Berry phase over six ground states. Needs spin-orbit coupling, a gap, and a crystal without an inversion centre; the column parallel to the applied field, since a field transverse to the seeded magnetization converges slowly | `magnetoelectric_tensor`, `Calculator.get_magnetoelectric_tensor` | | ✓ |
| **Continuing one run from another across a change of spin regime** — a converged non-magnetic density as the starting point of a magnetic run, a collinear one of a noncollinear run, spin-orbit coupling switched on | `run_scf(starting_from=...)`, `System.with_spin` | (✓)¹ | |
| **Reaching self-consistency** — Anderson/Broyden mixing, Kerker or local Thomas-Fermi preconditioning (the latter screening by the *local* density, which is what a slab needs), or solving the residual with its own Jacobian, which reaches magnetic solutions no mixer does | `run_scf(mixing_mode=...)`, `run_scf(scf_solver=...)` | (✓)² | (✓)² |
| **Band velocities** $\partial\epsilon_n/\partial\mathbf{k}$, with the nonlocal pseudopotential's own contribution — norm-conserving, ultrasoft and PAW | `band_velocities`, `VelocityOperator` | (✓)³ | |
| **Effective mass tensor** $m^{\ast}_{ij}$ at any k-point, with the principal masses and the density-of-states mass. Bands inside a degenerate multiplet are reported as the multiplet's invariant sum | `effective_mass`, `Calculator.get_effective_mass` | | ✓ |
| **Orbital, spin and total angular momentum on each atom** — $\langle L\rangle$, $\langle S\rangle$, $\langle J\rangle$, which is where the orbital moment of a spin-orbit magnet actually sits. Needs the whole k-grid; a relativistic ultrasoft or PAW dataset is refused | `angular_momenta`, `Calculator.get_angular_momenta` | (✓)⁴ | ✓ |
| **Orbital magnetization of the cell** $\mathbf{M}_\mathrm{orb}$ — the modern theory's k-space expression, local plus itinerant circulation, which is the half of a magnet's moment no integral over the cell can give. Needs spin-orbit coupling, broken time reversal and a gapped manifold; norm-conserving, on the whole uniform grid. Agrees with `pw.x`'s `lorbm` on both Kubo terms to **2e-6** $\mu_B$/cell | `lorbm`, `run_orbital_magnetization`, `Calculator.get_orbital_magnetization` | ✓ | ¹³ |
| **Dielectric constant** $\epsilon^\infty$ and **Born effective charges** — insulators, norm-conserving, ultrasoft and PAW (PAW $Z^{\ast}$ refused). The response solver underneath runs for collinear spin too | `dielectric_tensor` | ✓ | ✓ |
| **Phonons at $\Gamma$** — the force constants and their frequencies, insulators and metals, on norm-conserving, ultrasoft and PAW datasets. Away from $\Gamma$, and an ultrasoft or PAW metal, are refused | `dynamical_matrix` | ✓ | ✓ |
| **The strain response** $\partial\psi/\partial\varepsilon$, $\partial\rho/\partial\varepsilon$ and the deformation potentials, on norm-conserving, ultrasoft and PAW datasets | `strain_response` | | |
| **Elastic constants** $C_{ijkl}$ and the compliance and bulk modulus that follow — clamped-ion, insulators, norm-conserving | `elastic_constants` | | |
| **Electrostriction coefficients** $m$, $q$, $M$ and $Q$ — the quadratic electromechanical coupling, clamped-ion, insulators, norm-conserving | `electrostriction` | | |
| **Elasto-optic tensor** $\partial\chi_{ij}/\partial\varepsilon_{kl}$ — how a strain changes the dielectric response, which is what makes a squeezed crystal birefringent, and the third derivative the electrostriction coefficients are obtained from. Clamped-ion, insulators, norm-conserving | `electrostriction(...).photoelastic`, `Calculator.get_electrostriction` | | |
| **Piezoelectric tensor** $e_{k,ij}$ — the polarization a strain induces, which is the stress a field induces. Clamped-ion, insulators, norm-conserving, and non-polar crystals only: a class that admits a spontaneous polarization is refused, since the proper response then needs $P$ itself | `piezoelectric_tensor`, `Calculator.get_piezoelectric_tensor` | | ✓ |
| **Raman tensors** $\partial\epsilon_{ij}/\partial\tau$ — how the dielectric tensor changes when an atom moves. Insulators, norm-conserving/ultrasoft/PAW; $\chi^{(2)}$ and the electro-optic tensor are refused | `raman_tensors` | (✓)⁵ | |
| **Raman and infrared spectra** — the per-mode activities, depolarisation ratios and electronic polarizability at $\Gamma$ | `vibrational_spectrum` | ✓ | |
| **LO-TO splitting and the static dielectric constant** — the macroscopic field a polar mode builds, which raises the longitudinal mode and screens a static field. $\epsilon^0_{ij} = \epsilon^\infty_{ij} + (4\pi e^2/\Omega)\sum_\nu p^\nu_i p^\nu_j/\omega_\nu^2$, and the two together satisfy the Lyddane-Sachs-Teller relation. Insulators; needs the Born charges, and a physical splitting needs them charge-neutral | `vibrational_spectrum(loto_direction=..., neutralize=True)`, `nonanal`, `polar_mode_permittivity` | ✓ | (✓)¹⁰ |
| **Optical conductivity tensor** $\sigma_{ab}(\omega)$, the magneto-optical **Kerr angle** and the **anomalous Hall conductivity** — interband plus a Drude term, from $\partial H/\partial\mathbf{k}$ rather than momentum matrix elements. Insulators and metals, norm-conserving; needs the whole k-grid rather than a wedge, since the antisymmetric part is an axial vector | `run_conductivity`, `Calculator.get_optical_conductivity` | (✓)⁷ | ✓ |
| **Fermi-surface nesting function** $N(\mathbf{q})$ — how much of the Fermi surface maps onto itself when translated by $\mathbf{q}$, which is where a phonon softens, a charge-density wave opens a gap or a spin spiral finds its pitch. Metals with a smearing; a symmetry-reduced wedge is unfolded rather than refused | `run_nesting`, `Calculator.get_nesting` | | ✓ |
| **X-ray and magnetic structure factors** $F(\mathbf{H})$ — the Fourier coefficients of the charge and of the magnetization on each reflection, which is what a diffraction experiment measures rather than a density. Norm-conserving, ultrasoft and PAW; valence-only, so a forbidden reflection (where the core cancels and only the bonding charge is left) is the one an all-electron code agrees with; an energy window rebuilds the density from a chosen range of states | `run_structure_factors`, `Calculator.get_structure_factors`, `hmax`, `window`, `core` | | ✓ |
| **Scanning-tunnelling microscopy images** — Tersoff-Hamann: the current an s-wave tip draws is the sample's local density of states at the tip, so the image is the density rebuilt from the states the bias selects — a delta at the Fermi level at zero bias, the window $[E_F, E_F+V]$ with one, and a negative $V$ images the filled states. Constant height or constant current; norm-conserving, ultrasoft and PAW, since the augmentation charge follows the tunnelling weights. **A magnetic tip is included** and is the part neither code has: $[\rho + P\,\hat{\mathbf{n}}\!\cdot\!\mathbf{m}]/2$, which on a noncollinear crystal makes the image depend on which way the tip points. Agrees with `pp.x`'s `plot_num = 5` to **6.7e-10** on the whole grid | `run_stm`, `Calculator.get_stm`, `height`, `plane`, `bias`, `spin`, `polarization`, `mode = 'constant-current'` | (✓)¹⁴ | (✓)¹⁴ |
| **Shift current** $\sigma^{abc}(0;\omega,-\omega)$ — the bulk photovoltaic effect: the direct current a crystal with no inversion centre carries under illumination, with no junction and no built-in field. Insulators, norm-conserving, `nspin = 1` or spinor; needs the whole k-grid rather than a wedge, and the band count is the convergence parameter because the generalised derivative's intermediate sum runs over the same bands | `run_shift_current`, `Calculator.get_shift_current` | ⁸ | |
| **Second-harmonic generation** $\chi^{(2)}(-2\omega;\omega,\omega)$ — how much of the light shone on a crystal comes back out at twice the frequency. A polar rank-3 tensor, zero in any centrosymmetric crystal. Insulators, norm-conserving, `nspin = 1` or spinor; needs the whole k-grid rather than a wedge, and the band count is the convergence parameter because the sum over the intermediate state is an identity only over a complete basis | `run_shg`, `Calculator.get_shg`, `scissor` | (✓)⁹ | ✓ |
| **Magnons and the transverse spin susceptibility** $\chi^{+-}(\mathbf{q},\omega)$ — the collective precession of a magnet's own magnetization, and the Stoner continuum of independent spin flips it separates from. Its pole is the spin wave, and the Goldstone theorem pins it to zero energy at $\mathbf{q}=0$, which is the calculation's own error bar. Collinear magnets, insulating or metallic, norm-conserving; $\mathbf{q}$ must be a difference of two k-points of the run's own grid, and the whole grid is needed rather than a wedge. Nickel's magnon agrees with Elk's committed reference to **0.8%** on the same cell and grid | `run_spin_susceptibility`, `run_magnon_dispersion`, `Calculator.get_magnon_dispersion`, `ecut_response`, `goldstone_correction` | (✓)¹² | ✓ |
| **Optical absorption spectra with excitons** — $\mathrm{Im}\,\epsilon_M(\omega)$ from TDDFT, local-field effects included, on a bootstrap exchange-correlation kernel. Needs the whole k-grid rather than a wedge | `run_absorption`, `kernel = 'bootstrap'` (also `rpa`, `alda`, `lrc`, `bootstrap-1`), `ecut_response`, `scissor`, `broadening` | | ✓ |
| **Magnetic torque** $-\mathrm{d}F/\mathrm{d}\theta$ — the anisotropy from **one** angle instead of a difference of two, which removes seven digits of cancellation and is robust to the smearing width where the difference is not. $E(\theta) = K_1\sin^2\theta$ makes the torque at 45 degrees equal to $-K_1$. Same refusals as the anisotropy row below | `run_torque`, `Calculator.get_torque` | | |
| **Magnetocrystalline anisotropy** — the energy it costs to point a magnet's moment one way rather than another, by the force theorem: converge without spin-orbit coupling, rotate the converged density onto $\hat{\mathbf{n}}$, diagonalise once with the coupling on. One diagonalisation per direction, no reconvergence. Ultrasoft and norm-conserving; PAW is refused, the handoff carrying no `becsum`. Includes the per-orbital decomposition and a knob that switches the coupling off inside one relativistic dataset | `run_anisotropy`, `run_force_theorem`, `Calculator.get_anisotropy`, `lforcet`, `soc_scale`, `frozen_expectation` | ✓ | (✓)¹¹ |
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
- ⁷ `epsilon.x`'s `offdiag_calc` forms the dielectric tensor, but computes no
  conductivity and no Kerr angle, refuses ultrasoft datasets outright, and
  builds its dipole from momentum matrix elements — which is not
  $[H,\mathbf{r}]$ when the pseudopotential is nonlocal.
- ⁸ Blank rather than ticked, and the distinction is worth stating because
  the QE tarball does contain an implementation: `external/wannier90`'s
  `berry_task = 'sc'` computes a shift current, but Wannier90 is a separate code
  bundled beside Quantum ESPRESSO rather than part of it, it needs a
  wannierisation first, and nothing in `PW/src`, `PP/src` or `PHonon` computes a
  photocurrent of any kind. Elk has none either — its `nonlinopt.f90` is
  second-harmonic generation, which is a different response.

- ⁹ `PHonon`'s `el_opt.f90` computes the **electro-optic** tensor, which is
  the *static* second-order response and not
  $\chi^{(2)}(-2\omega;\omega,\omega)$; nothing in the tree computes a
  frequency-dependent second-harmonic tensor, and the
  `lraman`/`elop` branch that reaches even the static one is the branch P35
  established does not reproduce QE's own committed example. Elk's
  `nonlinopt.f90` (task 125) is the real reference and is what this was
  validated against.

- ¹² `TDDFPT`'s turboMagnon (`lr_magnons_main.f90`) is a Liouville-Lanczos
  solver: it propagates a response vector and never forms $\chi_0$ as a matrix
  over reciprocal lattice vectors, so there is no Dyson equation and no
  eigenvalue whose crossing of one is the mode. Nothing in `PW/src` or `PP/src`
  computes a spin susceptibility at all. Elk's tasks 330/331
  (`tddftsplr.f90`) do exactly this, for the general $4\times4$ spin-density
  response of which the transverse block computed here is the collinear
  corner.

- ¹⁰ Elk adds the same non-analytic term (`dynqnat.f90`, under `tphnat`) and
  computes Born effective charges (task 208), but its static dielectric tensor
  is **read in** rather than assembled from the modes: nothing there sums the
  oscillator strengths into $\epsilon^0$, which is the half `dynmat.x`'s `lperm`
  does.

- ¹³ Elk has no orbital magnetization by the modern theory. Its moments are
  integrals of the magnetization over the muffin tins and the interstitial, and
  its orbital information is the per-atom `writelsj` decomposition of the row
  above; the phrase does not occur anywhere in its manual.

- ¹⁴ both codes compute the **charge** image and neither computes the spin-polarized one: QE's `PP/src/stm.f90` (`plot_num = 5`) sums $|\psi|^2$ over a
  bias window with no spin channel and no `addusdens`, so it is norm-conserving
  and charge-only; Elk's task 162 (`wfplot.f90`) is the zero-bias delta only, has
  no bias window, and plots `rhomt`/`rhoir` — the charge — whatever the run's
  magnetism. Constant current is QE's alone (`pp.x`'s `ISOSTM` card,
  `chdens_module.f90`) and Elk has none; QE's returns the **FFT plane index** at
  which the density first exceeds the set-point, along the third axis only, so its
  corrugation is quantised to the grid spacing where this one is interpolated
  between scan planes and takes the plane's own normal.

- ¹¹ Elk's `mae.f90` (tasks 28/29) computes a magnetic anisotropy energy, but by
  a **different method**: it re-converges a full ground state for each direction
  of the moment, rotating the lattice rather than the moment. It is not the
  force theorem, and the two answers differ by the self-consistency the force
  theorem does without. What transfers from it is `socscf`, its direction sets
  (`gentpmae`), and the binary as an independent check.

The variants under each row — which smearing or tetrahedron method fixes the
occupations, which projectors DFT+U uses, which constraint scheme — are chosen
with the same input variables as in `pw.x` where it has them.

**Not yet:** phonons away from $\Gamma$, exact exchange, real-time propagation.
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
from defumat.system.kpoints import KPoints

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
| [`08_spin_orbit_coupling`](notebooks/08_spin_orbit_coupling.ipynb) | Spinors and $j$-resolved projectors, platinum against Quantum ESPRESSO, and a bismuthene gap made of nothing but the coupling |
| [`09_forces_and_relaxation`](notebooks/09_forces_and_relaxation.ipynb) | Forces as one gradient of the energy, against Quantum ESPRESSO, and a structure relaxing back onto its lattice site |
| [`10_topological_invariants`](notebooks/10_topological_invariants.ipynb) | Berry curvature from one overlap rather than a derivative, Chern numbers that are exact integers, and $\mathbb{Z}_2$ by two independent routes |
| [`11_noncollinear_magnetism_and_fields`](notebooks/11_noncollinear_magnetism_and_fields.ipynb) | Magnetism as a vector, bcc iron against Quantum ESPRESSO, constrained moments, and the direction the energy cannot depend on |
| [`12_spin_spirals`](notebooks/12_spin_spirals.ipynb) | Spin spirals of any pitch without a supercell, and an $E(\mathbf{q})$ magnon dispersion |
| [`13_dft_plus_u`](notebooks/13_dft_plus_u.ipynb) | The Hubbard correction on antiferromagnetic FeO, and the occupations it drives to 0 and 1 |
| [`14_spiral_relaxation`](notebooks/14_spiral_relaxation.ipynb) | $\mathrm{d}E/\mathrm{d}\mathbf{q}$: which terms of the energy a spiral's wavevector touches, and a BFGS walking a hydrogen chain to its ground-state pitch |
| [`15_stress`](notebooks/15_stress.ipynb) | The stress as the strain derivative of the energy, silicon's equation of state, and the pressure against $-\mathrm{d}E/\mathrm{d}V$ |
| [`16_projected_density_of_states`](notebooks/16_projected_density_of_states.ipynb) | Silicon's $s$ and $p$ densities of state against `projwfc.x`, and the same weights as fat bands |
| [`17_reaching_self_consistency`](notebooks/17_reaching_self_consistency.ipynb) | Making a hard SCF converge: preconditioning, and the magnetic solutions no mixer reaches |
| [`18_continuing_a_calculation`](notebooks/18_continuing_a_calculation.ipynb) | Starting one run from another's converged state, across a change of spin regime |
| [`19_linear_response`](notebooks/19_linear_response.ipynb) | The dielectric constant and the Born effective charges of silicon, against `ph.x`, on all three kinds of pseudopotential |
| [`20_phonons`](notebooks/20_phonons.ipynb) | Phonon frequencies at $\Gamma$: silicon's optical mode against `ph.x`, and a metal |
| [`21_electrostriction`](notebooks/21_electrostriction.ipynb) | Elastic constants, electrostriction and the elasto-optic tensor of silicon |
| [`22_van_der_waals`](notebooks/22_van_der_waals.ipynb) | Grimme's D2 dispersion, and bilayer graphene binding where PBE alone has no minimum |
| [`23_variable_cell_relaxation`](notebooks/23_variable_cell_relaxation.ipynb) | Relaxing the cell and the atoms together: arsenic squeezed to simple cubic at 500 kbar, against `pw.x` |
| [`24_tran_blaha_band_gaps`](notebooks/24_tran_blaha_band_gaps.ipynb) | Band gaps from the modified Becke-Johnson potential: silicon from LDA's 0.49 eV to 1.13, against an experimental 1.17 |
| [`25_your_own_crystal`](notebooks/25_your_own_crystal.ipynb) | Running a material of your own: diamond from a lattice constant, a fetched pseudopotential, and the two convergence tests |
| [`26_raman_and_infrared_spectra`](notebooks/26_raman_and_infrared_spectra.ipynb) | Raman and infrared activities per mode: silicon's 519.2 cm⁻¹ line, and why it is infrared-silent |
| [`27_excitons_and_tddft`](notebooks/27_excitons_and_tddft.ipynb) | Optical absorption from TDDFT with a bootstrap kernel, and the excitonic peak RPA does not have |
| [`28_piezoelectricity`](notebooks/28_piezoelectricity.ipynb) | The voltage a squeezed crystal produces: AlAs's one independent component, and why silicon has none |
| [`29_effective_mass_and_angular_momenta`](notebooks/29_effective_mass_and_angular_momenta.ipynb) | Effective masses as one difference of an analytic velocity, against the all-electron Elk binary, and where a spin-orbit magnet's orbital moment sits |
| [`30_magneto_optics`](notebooks/30_magneto_optics.ipynb) | The Kerr effect: linearly polarized light coming back rotated off a magnet, and the off-diagonal conductivity that produces it |
| [`31_fermi_surface_nesting`](notebooks/31_fermi_surface_nesting.ipynb) | Where a metal will go unstable: the wavevector that slides one piece of the Fermi surface onto another |
| [`32_shift_current`](notebooks/32_shift_current.ipynb) | The bulk photovoltaic effect: a crystal with no inversion centre carrying a current under uniform light, with no junction |
| [`33_second_harmonic_generation`](notebooks/33_second_harmonic_generation.ipynb) | Frequency doubling in AlAs: the tensor zincblende allows, checked against the all-electron code Elk |
| [`34_electric_polarization`](notebooks/34_electric_polarization.ipynb) | Polarization as a Berry phase, the Born charge read off a displacement, and the sum rule that catches an undersampled zone |
| [`35_magnetoelectric_effect`](notebooks/35_magnetoelectric_effect.ipynb) | A magnetic field producing an electric polarization in AlAs, and the null that shows it is spin-orbit coupling and nothing else |
| [`36_magnetic_anisotropy`](notebooks/36_magnetic_anisotropy.ipynb) | Which way a magnet wants to point: a milli-electronvolt read off a stretched cobalt cell as a derivative rather than a difference |
| [`37_structure_factors`](notebooks/37_structure_factors.ipynb) | What a diffraction experiment sees: silicon's forbidden (222) reflection, which is bonding charge and nothing else |

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
| phonons at $\Gamma$ — silicon, and a metal | 0.05 and 0.0019 cm⁻¹ |
| phonons at $\Gamma$ — ultrasoft and PAW silicon | 0.019 and 0.027 cm⁻¹ |
| Raman and infrared activities | every digit `dynmat.x` prints |
| LO-TO splitting, and the static dielectric constant | every digit `dynmat.x` prints; Lyddane-Sachs-Teller to 5e-11 |

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
