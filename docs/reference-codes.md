# What each tick in the README's table rests on

The feature table in `README.md` says, quantity by quantity, whether Quantum
ESPRESSO and Elk compute it too, and each tick is a claim about someone else's
source. This file is the evidence: for every note the table qualifies a tick
with, the routine, task or file in the other code that was located, and what it
does and does not do. The numbers match the notes under the table. The audience
is whoever has to defend or update a tick, so the routine names, line numbers
and phase references are kept here where the README carries one sentence.

The vendored Quantum ESPRESSO 7.5 tree is under
`quantum_espresso/qe-7.5-ReleasePack/qe-7.5/`; Elk's task list is §5.127 of
`docs/elk_manual.txt` and `ELK-FEATURES.md` surveys it against QE 7.5.

## The notes

**1. Projected density of states, Elk (✓).** Elk's partial density of states
(task 10) is resolved over $(l, m)$ and over spin: `dosmsum` and `dosssum` sum
those away, and `lmirep` transforms the $Y_{lm}$ basis into irreducible
representations (manual §5.25, §5.26, §5.59). None of that is a $j$ resolution,
since there is no decomposition onto the spin-angle functions
$\lvert l\,j\,m_j\rangle$, which is what a spin-orbit run's orbital character
means. `projwfc.x` has it (`atomic_wfc_nc_proj`, `partialdos_nc`) and is what the
$j$-resolved projection here is validated against.

**2. Band velocities, QE (✓).** `fermi_velocity.x` finite-differences eigenvalues
and reports only the magnitude.

**3. Starting from an all-electron ground state, blank in both.** Neither code
reads the other's ground state. `pw.x` restarts from its own `charge-density.dat`
(`potinit.f90`'s `read_rhog`, reached by `startingpot = 'file'`) and has no reader
for a foreign format; Elk restarts from its own `STATE.OUT` and no task in its
list reads or writes another code's density. Elk's own `STATE.OUT` reader is not
the same claim: what is ticked here is crossing from an all-electron muffin-tin
representation into a plane-wave pseudopotential one, which is a transfer
neither code has a reason to implement.

**4. Band gaps from the Tran-Blaha potential, QE (✓).** Quantum ESPRESSO reaches
it only through libxc (`XClib/dft_setting_routines.f90` maps `tb09` to libxc
208), and then passes a zero Laplacian and never sets the functional's
coefficient, so what it runs under that name is a different functional.

**5. Orbital, spin and total angular momentum on each atom, QE (✓).** `lorbm`
gives the cell's orbital magnetization and nothing per atom; Elk has the site
decomposition (`writelsj`).

**6. Orbital magnetization of the cell, Elk blank.** Elk has no orbital
magnetization by the modern theory. Its moments are integrals of the
magnetization over the muffin tins and the interstitial, and its orbital
information is the per-atom `writelsj` decomposition of the row above; the
phrase does not occur anywhere in its manual.

**7. Magnetocrystalline anisotropy by the force theorem, Elk (✓).** Elk's
`mae.f90` (tasks 28/29) computes a magnetic anisotropy energy, but by a
different method: it re-converges a full ground state for each direction of the
moment, rotating the lattice rather than the moment. It is not the force theorem,
and the two answers differ by the self-consistency the force theorem does
without. What transfers from it is `socscf`, its direction sets (`gentpmae`), and
the binary as an independent check. That method is now here too, as the relaxed
row (note 8), so this note records why the two rows are separate rather than a
gap: the difference between them is measured, 0.447 against 0.552 meV on
tetragonal cobalt, and is the quantity the force theorem approximates.

**8. Relaxed magnetocrystalline anisotropy, QE (✓).** `pw.x` converges a
noncollinear spin-orbit run at a stated moment direction and prints its total
energy, so the quantity is reachable by running it once per direction and
subtracting by hand. There is no routine: nothing in QE sets up the directions,
holds the k-set fixed across them, or reports how far a moment drifted from
where it was put. Elk's `mae.f90` (tasks 28/29) is the full tick and is the same
method, down to rotating the lattice rather than the moment, which is a neater
way of avoiding the quantization-axis trap than rebuilding `angle1`/`angle2`,
and is the obvious thing to try if that rebuild ever becomes expensive. Done
that way against `pw.x` 7.5 on tetragonal cobalt: the two total energies agree
to the eight decimals `pw.x` prints and the anisotropy is 0.447302 meV here
against 0.4474 (`PLAN.md` P87).

**9. Magnons, QE (✓).** `TDDFPT`'s turboMagnon (`lr_magnons_main.f90`) is a
Liouville-Lanczos solver: it propagates a response vector and never forms
$\chi_0$ as a matrix over reciprocal lattice vectors, so there is no Dyson
equation and no eigenvalue whose crossing of one is the mode. Nothing in
`PW/src` or `PP/src` computes a spin susceptibility at all. Elk's tasks 330/331
(`tddftsplr.f90`) do exactly this, for the general $4\times4$ spin-density
response of which the transverse block computed here is the collinear corner.

**10. Raman tensors, QE (✓).** `ph.x` refuses a gradient-corrected functional
here (`phq_setup.f90`), where this does not.

**11. LO-TO splitting and the static dielectric constant, Elk (✓).** Elk adds
the same non-analytic term (`dynqnat.f90`, under `tphnat`) and computes Born
effective charges (task 208), but its static dielectric tensor is read in rather
than assembled from the modes: nothing there sums the oscillator strengths into
$\epsilon^0$, which is the half `dynmat.x`'s `lperm` does.

**12. Optical conductivity, the Kerr angle and the anomalous Hall conductivity,
QE (✓).** `epsilon.x`'s `offdiag_calc` forms the dielectric tensor, but computes
no conductivity and no Kerr angle, refuses ultrasoft datasets outright, and
builds its dipole from momentum matrix elements, which is not $[H, \mathbf r]$
when the pseudopotential is nonlocal.

**13. Shift current, QE blank.** Blank rather than ticked, and the distinction is
worth stating because the QE tarball does contain an implementation:
`external/wannier90`'s `berry_task = 'sc'` computes a shift current, but
Wannier90 is a separate code bundled beside Quantum ESPRESSO rather than part of
it, it needs a wannierisation first, and nothing in `PW/src`, `PP/src` or
`PHonon` computes a photocurrent of any kind. Elk has none either: its
`nonlinopt.f90` is second-harmonic generation, which is a different response.

**14. Second-harmonic generation, QE (✓).** `PHonon`'s `el_opt.f90` computes the
electro-optic tensor, which is the static second-order response and not
$\chi^{(2)}(-2\omega;\omega,\omega)$; nothing in the tree computes a
frequency-dependent second-harmonic tensor, and the `lraman`/`elop` branch that
reaches even the static one is the branch `PLAN.md` P35 established does not
reproduce QE's own committed example. Elk's `nonlinopt.f90` (task 125) is the
real reference and is what this was validated against.

**15. Scanning-tunnelling microscopy images, (✓) in both.** Both codes compute
the charge image and neither computes the spin-polarized one. QE's
`PP/src/stm.f90` (`plot_num = 5`) sums $\lvert\psi\rvert^2$ over a bias window
with no spin channel and no `addusdens`, so it is norm-conserving and
charge-only; Elk's task 162 (`wfplot.f90`) is the zero-bias delta only, has no
bias window, and plots `rhomt`/`rhoir`, the charge, whatever the run's
magnetism. Constant current is QE's alone (`pp.x`'s `ISOSTM` card,
`chdens_module.f90`) and Elk has none; QE's returns the FFT plane index at which
the density first exceeds the set-point, along the third axis only, so its
corrugation is quantised to the grid spacing where this one is interpolated
between scan planes and takes the plane's own normal.

**16. Vertical tunnelling transport through a 2D material, QE (✓).** QE computes
a Landauer transmission and it is a different geometry: `PWCOND` (`pwcond.x`,
Choi and Ihm's complex-band-structure method, `PWCOND/src/transmit.f90`) solves
the scattering problem between two semi-infinite crystalline leads with the
current along one axis, and returns one conductance per energy for that
junction. It has no point contact and therefore no map: nothing in it is a
function of where a tip is, which is the whole output here. Elk has neither: no
task in its list computes a conductance, and `ELK-FEATURES.md` records none.

**17. Which k-points the tunnelling current comes out of, blank in both.** That
is a claim about two sources rather than a gap in the search. `pw.x` has nothing
of the kind; `PWCOND/` is a Landauer transmission of a different geometry, two
semi-infinite crystalline leads, one conductance per energy, no tip and so no
momentum resolution. Elk's task list has no vertical junction at all, and its
Fermi-surface tasks (100/101, `fermisurf.f90`) write the bands for a plotting
program rather than weighting them by anything. The one other implementation
known is elkpy's Elk patch (task 9007), which is not stock Elk; it is what the
NbSe2 numbers here are checked against, and the two agree on the contraction
independently.

## The convergence workflows, which have no row

Six rows about getting a calculation to converge left the table when the README
was reorganised, because they are workflows rather than quantities; the README
describes them in one paragraph under "Magnetism". Their former ticks, and the
evidence under them, are kept here so that the claims are not lost.

| Former row | How to ask for it | QE | Elk |
|---|---|:-:|:-:|
| Restarting an SCF from the middle | `checkpoint_dir`, `checkpoint_every`, `max_seconds` on `run_scf` and `Calculator` | ✓ | ✓ |
| A starting magnetic texture, one direction per atom | `STARTING_MOMENTS` card, `Calculator.with_moments` | | |
| Converging a magnetic structure that is not the ground state | `constrained_magnetization = 'atomic'` with a `STARTING_MOMENTS` card | (✓) a | (✓) a |
| Holding a texture with a field instead of a penalty | `constrained_magnetization = 'atomic fsm'`, `'atomic fsm direction'` | | ✓ b |
| Continuing one run from another across a change of spin regime | `run_scf(starting_from=...)`, `System.with_spin` | (✓) c | |
| Reaching self-consistency: mixers, preconditioners, the residual solver | `run_scf(mixing_mode=...)`, `run_scf(scf_solver=...)` | (✓) d | (✓) d |

**a. Converging a magnetic structure that is not the ground state.** Both codes
hold a moment per atom and neither holds a texture the way this row means it.
QE's `constrained_magnetization = 'atomic'` (`i_cons = 1`, `add_bfield.f90`)
takes its target from `starting_magnetization` and `angle1`/`angle2`, which are
per species, so a 120-degree Néel state on one species has one target for all
three sites and cannot be stated. Elk's `fsmtype = 2`/`3` does fix
`mommtfix(:, ia, is)` per atom (`bfieldfsm.f90:32-73`) and is a feedback field
rather than a penalty, so it converges to a genuine stationary point where a
penalty leaves a residual; that is the better mechanism, and it is implemented
here as `'atomic fsm'` (entry b).

**b. Holding a texture with a field instead of a penalty.** Measured against Elk
on the same cell, and it wins on a robust magnet. Two iron moments at 90 degrees
without spin-orbit coupling (`tests/data/qe/fe2-canted-nosoc.in`):
`'atomic fsm'` converges in 46 iterations with the pair at 90.002 degrees and
both lengths on target, where the vector penalty holds 89.2 degrees in 76 and
Elk's own `fsmtype = -2` holds the direction in 55 loops. It needs Elk's three
choices: the moment read off the output density, Elk's history-free mixer
(`mixing_mode = 'adaptive'`, `mixing_beta = 0.05`), which is warned about when
absent, and Elk's gain in this code's units (0.02 Ry per $\mu_B$, the default).
The earlier verdict that it does not converge was measured on a hydrogen pair
that is barely magnetic unconstrained and with the first two choices wrong.
`PLAN.md` P108.

**c. Continuing one run from another.** `startingpot = 'file'` reads a density
across a change of `nspin`, but zero-fills the missing components, so a magnetic
run started that way converges back to the unpolarized answer.

**d. Reaching self-consistency.** Both codes have mixing and preconditioning, and
Elk additionally has the adaptive scheme (`mixtype = 1`, `src/mixadapt.f90`)
where `pw.x` has no adaptive mode at all; the residual solver, which is what
reaches the extra solutions, is in neither.
