# Van der Waals dispersion: Grimme's D2

A semilocal functional cannot see a van der Waals bond. Its correlation energy is a
functional of the density *where the orbitals are*, and the London force comes from the
correlated fluctuations of two charge distributions that do not overlap at all. Bilayer
graphene is the canonical demonstration: **PBE alone gives it no minimum in the interlayer
separation**, and the two layers drift apart.

Grimme's D2 ([J. Comp. Chem. **27**, 1787 (2006)](https://doi.org/10.1002/jcc.20495)) puts
the attraction back as a pair potential over the nuclei,

$$E_{\rm disp} = -\frac{s_6}{2} \sum_{i \neq j} \sum_{\mathbf L}
   \frac{C_6^{ij}}{|\mathbf r_{ij} + \mathbf L|^6}\;
   f_{\rm damp}\!\big(|\mathbf r_{ij} + \mathbf L|\big),
\qquad
f_{\rm damp}(R) = \frac{1}{1 + e^{-d\,(R/R_{\rm vdW} - 1)}}$$

with $C_6^{ij} = \sqrt{C_6^i C_6^j}$ from a table indexed by atomic number. The damping
function switches the $R^{-6}$ tail off where the two densities begin to overlap and the
functional is already describing the interaction.

It is the Ewald sum with a different radial function: a pair sum over the nuclei and their
periodic images, so the force and the stress are derivatives of it in the two coordinates.

Against `pw.x` on the same input: the total energy to **3.1e-9 Ry**, the dispersion term to
**4.9e-9 Ry**, the force to **3.7e-7 Ry/bohr** and the stress to **4.1e-8 Ry/bohr³**. And
the relaxation binds the bilayer at **6.10 bohr (3.23 Å)**, against a measured 3.35 Å.


```python
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pypresso import Calculator
from pypresso.io import read_qe_output
from pypresso.vdw.analytic import dispersion_force, dispersion_stress

CASES = Path("..") / "tests" / "data" / "qe"
PSEUDO = Path("..") / "tests" / "data" / "pseudo"

bilayer = Calculator.from_file(CASES / "graphene-bilayer-d2.in", pseudo_dir=PSEUDO,
                               announce=False, conv_thr=1e-10)
system, pseudos = bilayer.system, bilayer.pseudos
calculation = bilayer.calculation
reference = read_qe_output(CASES / "reference.out.graphene-bilayer-d2")

print(f"vdw_corr = {system.vdw_corr!r}, london_s6 = {system.london_s6}, "
      f"london_rcut = {system.london_rcut:g} bohr")
print(calculation.dispersion_sum.report([s.name for s in system.structure.species]))
print(f"\n  E_disp   {calculation.dispersion:.8f} Ry"
      f"      pw.x  {reference.energy_terms['dispersion']:.8f} Ry")
```

    vdw_corr = 'grimme-d2', london_s6 = 0.75, london_rcut = 200 bohr
    Parameters for Dispersion (Grimme-D2) Correction:
      atom      VdW radius       C_6
            C          2.744         60.710
    
      E_disp   -0.02305929 Ry      pw.x  -0.02305929 Ry


## 1. It is a function of the nuclei, and of nothing else

D2 never sees the electron density, which has a consequence sharper than any tolerance: the
density, the eigenvalues and the potential of a run with the correction are **bit for bit**
what the same run without it produces, and the two total energies differ by exactly the
dispersion energy. That is what distinguishes a correction of this kind from one that
enters the self-consistent potential, and it is what makes the electronic and the
dispersion energies separately meaningful below.


```python
# The same cell with `vdw_corr` absent -- a different input, so a different calculator.
plain = Calculator.from_file(CASES / "graphene-bilayer.in", pseudo_dir=PSEUDO,
                             announce=False, conv_thr=1e-10).get_scf()
corrected = bilayer.get_scf()

print("max |rho(D2) - rho(none)|        ",
      np.abs(np.asarray(plain.density) - np.asarray(corrected.density)).max())
print("max |eps(D2) - eps(none)|        ",
      np.abs(np.asarray(plain.eigenvalues) - np.asarray(corrected.eigenvalues)).max())
print(f"\nE(D2) - E(none)  {corrected.total_energy - plain.total_energy:+.10f} Ry")
print(f"E_disp           {corrected.energy_terms['dispersion']:+.10f} Ry")

qe = read_qe_output(CASES / "reference.out.graphene-bilayer")
print(f"\npw.x, same pair  {reference.total_energy - qe.total_energy:+.10f} Ry")

print(f"\n{'term':<16}{'pypresso':>16}{'pw.x':>16}{'difference':>14}")
for name, value in sorted(corrected.energy_terms.items()):
    other = reference.energy_terms.get(name)
    print(f"{name:<16}{value:>16.8f}{other:>16.8f}{value - other:>14.1e}")
print(f"{'total':<16}{corrected.total_energy:>16.8f}{reference.total_energy:>16.8f}"
      f"{corrected.total_energy - reference.total_energy:>14.1e}")
```

    max |rho(D2) - rho(none)|         0.0
    max |eps(D2) - eps(none)|         0.0
    
    E(D2) - E(none)  -0.0230592851 Ry
    E_disp           -0.0230592851 Ry
    
    pw.x, same pair  -0.0230592800 Ry
    
    term                    pypresso            pw.x    difference
    dispersion           -0.02305929     -0.02305929       4.9e-09
    ewald                11.66546446     11.66546446       7.9e-10
    hartree              53.78895022     53.78896923      -1.9e-05
    one-electron        -96.54073327    -96.54075437       2.1e-05
    smearing             -0.00021298     -0.00021298       1.4e-09
    xc                  -13.99480871    -13.99480662      -2.1e-06
    total               -45.10439956    -45.10439956      -3.1e-09


## 2. The figure: what the correction is for

Because the SCF never sees the correction, one sweep of self-consistent runs gives both
curves at once. The PBE energies were measured offline; the dispersion energy at each
separation is recomputed here, because it costs nothing.

PBE alone is repulsive all the way out. Adding $E_{\rm disp}$, which is *monotonic* and has
no minimum of its own, produces one: the binding is the balance between an attractive tail
and the functional's own wall.

The sweep stops at 10 bohr and the curves are referred to that point rather than to
infinity, because the cell is 20 bohr tall: at a separation of 10 bohr each layer is
equidistant from its neighbour and from its own periodic image, and past that the two swap
roles. The binding energy below is therefore a lower bound on the well depth.


```python
# Measured offline: thirteen SCF runs on the 12x12x1 grid at ecutwfc = 40.
SEPARATION = np.array([5.4, 5.7, 6.0, 6.1, 6.3, 6.6, 7.0, 7.5, 8.0, 9.0, 10.0])
E_PBE = np.array([-45.06839999, -45.0745402, -45.07779712, -45.07857165, -45.0793757, -45.08052449, -45.08134049, -45.08141908, -45.08163641, -45.08156401, -45.08157174])   # Ry, the total energy with no correction

height = float(np.asarray(system.cell.at)[2, 2])
crystal = np.asarray(system.structure.positions) @ np.linalg.inv(np.asarray(system.cell.at))

def dispersion_at(separation):
    "The pair sum at this interlayer separation -- no SCF, milliseconds."
    moved = crystal.copy()
    moved[:2, 2] = 0.5 - 0.5 * separation / height
    moved[2:, 2] = 0.5 + 0.5 * separation / height
    positions = jnp.asarray(moved @ np.asarray(system.cell.at))
    return float(calculation.dispersion_sum.energy(positions))

E_DISP = np.array([dispersion_at(d) for d in SEPARATION])
total = E_PBE + E_DISP
minimum = SEPARATION[total.argmin()]

fig, ax = plt.subplots(figsize=(6.4, 4.2))
zero = lambda y: (y - y[-1]) * 13605.7   # meV, referred to the widest separation
ax.plot(SEPARATION, zero(E_PBE), "o-", color="0.55", label="PBE alone")
ax.plot(SEPARATION, zero(E_DISP), "s-", color="tab:orange", label="$E_{\\rm disp}$ (D2)")
ax.plot(SEPARATION, zero(total), "o-", color="tab:blue", lw=2, label="PBE + D2")
ax.axvline(6.104, color="tab:blue", ls=":", lw=1)
ax.axvline(3.35 * 1.8897259886, color="k", ls="--", lw=1)
ax.annotate(" BFGS 6.10", xy=(6.104, 0.62), xycoords=("data", "axes fraction"),
            color="tab:blue", fontsize=9, va="top")
ax.annotate(" experiment\n 3.35 A", xy=(3.35 * 1.8897259886, 0.50),
            xycoords=("data", "axes fraction"), fontsize=9, va="top")
ax.axhline(0.0, color="0.8", lw=0.8)
ax.set_xlabel("interlayer separation (bohr)")
ax.set_ylabel("energy relative to 10 bohr (meV / cell)")
ax.set_title("Bilayer graphene: PBE has no minimum, PBE+D2 does")
ax.legend(frameon=False)
fig.tight_layout()

print(f"binding energy   {zero(total).min() / 4:.1f} meV per atom (a lower bound: see above)")
print(f"PBE alone        {zero(E_PBE).min() / 4:+.1f} meV per atom at its lowest point -- "
      "no well")
print(f"minimum at       {minimum:.2f} bohr on this sweep "
      f"({minimum / 1.8897259886:.2f} A); the BFGS relaxation gives 6.10")
```

    binding energy   -17.8 meV per atom (a lower bound: see above)
    PBE alone        -0.2 meV per atom at its lowest point -- no well
    minimum at       6.10 bohr on this sweep (3.23 A); the BFGS relaxation gives 6.10



    
![png](22_van_der_waals_files/22_van_der_waals_5_1.png)
    


## 3. The force and the stress

Both are derivatives of the same pair sum, in the positions and in the strain that deforms
the lattice translations. `pw.x` prints each of them as a block of its own, so the
comparison below is against QE's own dispersion force and dispersion stress rather than
against a total.


```python
positions = system.structure.positions
volume = float(system.cell.volume)
sum_ = calculation.dispersion_sum

autodiff_force = -jax.grad(sum_.energy)(positions)
transcribed_force = dispersion_force(sum_, positions)

def strained(strain):
    deformation = jnp.eye(3) + strain
    return sum_.at_cell(deformation).energy(positions @ deformation.T)

autodiff_stress = -jax.grad(strained)(jnp.zeros((3, 3))) / volume
transcribed_stress = dispersion_stress(sum_, positions, volume)

print("force, z component (Ry/bohr)")
print(f"{'atom':>6}{'jax.grad':>14}{'force_london':>15}{'pw.x':>14}")
for atom in range(4):
    print(f"{atom + 1:>6}{autodiff_force[atom, 2]:>14.8f}"
          f"{transcribed_force[atom, 2]:>15.8f}"
          f"{reference.force_terms['dispersion'][atom, 2]:>14.8f}")
print(f"\nmax |jax.grad - force_london|   "
      f"{np.abs(np.asarray(autodiff_force - transcribed_force)).max():.1e}")
print(f"max |force_london - pw.x|       "
      f"{np.abs(np.asarray(transcribed_force) - reference.force_terms['dispersion']).max():.1e}")
print(f"max |jax.grad - stres_london|   "
      f"{np.abs(np.asarray(autodiff_stress - transcribed_stress)).max():.1e}")
print(f"max |stres_london - pw.x|       "
      f"{np.abs(np.asarray(transcribed_stress) - reference.stress_terms['dispersion']).max():.1e}")
```

    force, z component (Ry/bohr)
      atom      jax.grad   force_london          pw.x
         1    0.00173177     0.00173177    0.00173177
         2    0.00172333     0.00172333    0.00172333
         3   -0.00173177    -0.00173177   -0.00173177
         4   -0.00172333    -0.00172333   -0.00172333
    
    max |jax.grad - force_london|   3.6e-17
    max |force_london - pw.x|       3.9e-09
    max |jax.grad - stres_london|   2.7e-19
    max |stres_london - pw.x|       1.2e-08


## 4. What it does to the elastic constants, and what it cannot touch

Because the correction reaches the stress, it reaches everything built on the stress,
including the **elastic constants**. Because it does not reach the Hamiltonian, it reaches
nothing built on the electronic response: $d\chi/dx$ is unchanged to **0.0**.

So the effect of a dispersion correction on an electrostriction tensor is predictable in
advance. The stress coefficients $m$ and $q$ do not move at all, and the strain ones $M$ and
$Q$ move only through the elastic compliance that converts them. Measured on the silicon
case of notebook 21 with D2 switched on, which is unphysical and exactly the point:

| | no D2 | with D2 | difference |
|---|---|---|---|
| $C_{11}$ (GPa) | 209.377 | 206.553 | -2.824 |
| $C_{12}$ (GPa) | 67.976 | 67.134 | -0.842 |
| $C_{44}$ (GPa) | 133.962 | 134.628 | +0.666 |
| $d\chi_{11}/dx_{11}$ | 197.015 | 197.015 | **0.0** |

The difference column is reproduced to 2.2e-18 by the second derivative of the pair sum
alone. The cell below is that calculation, run on the bilayer.

The whole third-derivative path runs on the bilayer too. Clamped ion: $C_{11} = 859.0$,
$C_{12} = 26.5$, $C_{33} = 56.6$ GPa, reproducing a five-point second difference of the
energy to $5.8\times10^{-5}$, and $d\varepsilon/dx$ reproducing a central difference over
re-converged strained cells to $2.2\times10^{-4}$. Those are properties of the supercell,
half of which is vacuum, so what they are quoted for is the agreement.

**Every entry is negative, and that is not a mistake.** The D2 pair potential has no minimum
of its own: in its attractive tail $E \sim -C_6/R^6$ has $d^2E/dR^2 < 0$ everywhere. What
binds the bilayer is D2's slope against PBE's, not its curvature, so its contribution to the
elastic constants is a softening. The positive $C_{33}$ of the relaxed bilayer comes from
PBE's repulsive wall; D2 moves where that wall is met and takes a couple of GPa off the
stiffness once it is.


```python
VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
from pypresso.response.elastic import RY_TO_GPA   # Ry/bohr^3 -> GPa

gradient = jax.grad(strained)
contribution = np.zeros((6, 6))
for column, (k, l) in enumerate(VOIGT):
    tangent = jnp.zeros((3, 3)).at[k, l].add(0.5).at[l, k].add(0.5)
    _, response = jax.jvp(gradient, (jnp.zeros((3, 3)),), (tangent,))
    response = 0.5 * (np.asarray(response) + np.asarray(response).T) / volume
    contribution[:, column] = [response[i, j] * RY_TO_GPA for (i, j) in VOIGT]

names = ["11", "22", "33", "23", "13", "12"]
print("what D2 adds to C_IJ of bilayer graphene, GPa")
print("      " + "".join(f"{n:>9}" for n in names))
for row, name in enumerate(names):
    print(f"{name:>4}  " + "".join(f"{contribution[row, c]:9.3f}" for c in range(6)))
```

    what D2 adds to C_IJ of bilayer graphene, GPa
                 11       22       33       23       13       12
      11     -6.024   -2.116   -1.123   -0.004   -0.000   -0.000
      22     -2.116   -6.024   -1.123    0.004   -0.000   -0.000
      33     -1.123   -1.123   -5.486   -0.000    0.000    0.000
      23     -0.004    0.004   -0.000   -0.763   -0.000   -0.000
      13     -0.000   -0.000    0.000   -0.000   -0.763   -0.004
      12     -0.000   -0.000   -0.000   -0.000   -0.004   -1.954


## What is not here

Four of the five corrections are refused rather than silently substituted, and the reason
splits them cleanly in two. **D3**'s $C_6$ coefficients depend on each atom's coordination
number, so they are functions of the geometry with derivatives of their own rather than
table entries. **Tkatchenko-Scheffler**, **MBD** and **XDM** rescale their coefficients by
quantities built from the self-consistent density, Hirshfeld volumes or the exchange hole,
which puts them inside the self-consistent potential where D2 sits outside it, and the
identity in section 1 is precisely what would stop holding.

---
The tests behind this notebook: `tests/unit/test_dispersion.py`,
`tests/regression/test_dispersion.py`.
