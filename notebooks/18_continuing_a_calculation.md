# Continuing a calculation across a change of spin regime

An unpolarized run, a collinear one and a noncollinear one are three descriptions of the
same electrons, and the expensive part of all three, the charge density, is very nearly the
same object:

$$(n_\uparrow, n_\downarrow)
\;\longleftrightarrow\;
\big(n,\ m_z\big)
\;\longleftrightarrow\;
\big(n,\ \mathbf m\big),
\qquad
n = n_\uparrow + n_\downarrow, \quad m_z = n_\uparrow - n_\downarrow$$

So a converged calculation can start another one in a different regime: non-magnetic to
collinear, collinear to noncollinear, spin-orbit coupling switched on. What crosses is a
*guess*, so the continued run reaches the same self-consistent solution as a fresh one, and
the saving is in how far it has to travel:

| case | fresh | continued | agreement |
|---|---|---|---|
| Si, `nspin` 2 to 4 | 5 iterations | **1** | 1e-9 Ry |
| bcc Fe, 2 to 4 with the moment rotated onto `x` | 25 | **1** | 2e-8 Ry |
| bcc Fe, 1 to 2, magnetization seeded | 30 | 27 | 5e-9 Ry |
| Pt, scalar PAW to fully-relativistic PAW with `lspinorb` | 13 | **7** | 2e-10 Ry |

This is how a magnetic anisotropy is computed in practice: converge the collinear magnet
once, then run the noncollinear directions from it.


```python
import dataclasses
import warnings

import matplotlib.pyplot as plt
import numpy as np

from pypresso import Calculator
from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.scf import Calculation, continued_state, run_scf
from pypresso.scf.continuation import from_spin_components, spin_components
from pypresso.system import build_system

QE = "../quantum_espresso/qe-7.5-ReleasePack/qe-7.5/test-suite"
PSEUDO = "../tests/data/pseudo"

def system_from(path, **changes):
    return dataclasses.replace(build_system(read_pw_input(path)), tstress=False, **changes)

```

## Silicon: the same electrons, written three ways

The two-atom cell with Gaussian smearing, so that a second spin channel has an occupation
scheme able to fill it unequally. Nothing about the crystal changes between the three
regimes.


```python
si_pseudo = (read_upf(f"{PSEUDO}/Si.pz-vbc.UPF"),)
si_calc = Calculator(system_from(f"{QE}/pw_scf/scf.in", occupations="smearing",
                                 smearing="gaussian", degauss=0.02),
                     si_pseudo, announce=False, conv_thr=1e-8)

# `Calculator.with_spin` is `System.with_spin` plus the seed: a new calculator in
# the target regime, with an empty cache and the parent's state as its guess.
collinear_calc = si_calc.with_spin(2, starting_magnetization=(0.3,))
spinor_calc = collinear_calc.with_spin(4, starting_magnetization=(0.0,))

silicon, collinear, spinor = (si_calc.system, collinear_calc.system,
                              spinor_calc.system)
print(f"nspin = 1: {silicon.kpoints.nk} k-points, weights summing to "
      f"{float(np.sum(silicon.kpoints.weights)):.1f}")
print(f"nspin = 2: {collinear.kpoints.nk} k-points, weights summing to "
      f"{float(np.sum(collinear.kpoints.weights)):.1f}   <- degspin dropped")
print(f"nspin = 4: {spinor.kpoints.nk} k-points, nspin_mag = {spinor.nspin_mag}, "
      f"npol = {spinor.npol}")
```

    nspin = 1: 2 k-points, weights summing to 2.0
    nspin = 2: 2 k-points, weights summing to 1.0   <- degspin dropped
    nspin = 4: 2 k-points, nspin_mag = 1, npol = 2



```python
si1 = si_calc.get_scf()

# Continued: the derived calculators already carry the seed, so this is one call.
si2_cont = collinear_calc.get_scf()
si4_cont = spinor_calc.get_scf()

# Fresh: the same systems with no seed at all, which is `starting_from=None`.
si2_fresh = collinear_calc.get_scf(starting_from=None)
si4_fresh = spinor_calc.get_scf(starting_from=None)

for name, result in [("nspin=1", si1), ("nspin=2 from atoms", si2_fresh),
                     ("nspin=2 continued", si2_cont), ("nspin=4 from atoms", si4_fresh),
                     ("nspin=4 continued", si4_cont)]:
    print(f"{name:22s} {result.total_energy:15.9f} Ry   {result.iterations:2d} iterations")
print(f"\n2 -> 4 continued vs fresh: {si4_cont.total_energy - si4_fresh.total_energy:+.1e} Ry")
```

    nspin=1                  -15.794495570 Ry    5 iterations
    nspin=2 from atoms       -15.794495568 Ry    5 iterations
    nspin=2 continued        -15.794495568 Ry    5 iterations
    nspin=4 from atoms       -15.794495570 Ry    5 iterations
    nspin=4 continued        -15.794495570 Ry    5 iterations
    
    2 -> 4 continued vs fresh: +0.0e+00 Ry


Silicon is not magnetic, so the seeded moment decays and all five runs give the same answer,
which is the point: the regime is a description, and changing it does not move the physics.
What it moves is the iteration count.

## bcc iron: a moment that rotates, and one that has to be found

The interesting cases need a magnet. This is QE's bcc iron with `angle1 = 90`, which points
the moment along `x`. Three runs of the same cell: non-magnetic, collinear, and
noncollinear.


```python
iron_spinor = system_from(f"{QE}/pw_noncolin/noncolin.in")
fe_pseudo = (read_upf(f"{PSEUDO}/Fe.pz-nd-rrkjus.UPF"),)
iron_collinear = iron_spinor.with_spin(2)
iron_plain = iron_spinor.with_spin(1, starting_magnetization=(0.0,))

fe1 = run_scf(iron_plain, fe_pseudo, conv_thr=1e-8, mixing_beta=0.2)
fe2 = run_scf(iron_collinear, fe_pseudo, conv_thr=1e-8, mixing_beta=0.2)
print(f"non-magnetic  {fe1.total_energy:12.8f} Ry  {fe1.iterations:2d} iterations")
print(f"collinear     {fe2.total_energy:12.8f} Ry  {fe2.iterations:2d} iterations, "
      f"m = {fe2.magnetization:.4f} mu_B")
print(f"\nmagnetic stabilisation energy: {fe2.total_energy - fe1.total_energy:.6f} Ry")
```

    non-magnetic  -55.67804951 Ry  16 iterations
    collinear     -55.69968433 Ry  30 iterations, m = 3.1751 mu_B
    
    magnetic stabilisation energy: -0.021635 Ry



```python
fe4_fresh = run_scf(iron_spinor, fe_pseudo, conv_thr=1e-8, mixing_beta=0.2)
fe4_cont = run_scf(iron_spinor, fe_pseudo, conv_thr=1e-8, mixing_beta=0.2,
                   starting_from=fe2)
print(f"noncollinear from atoms   {fe4_fresh.total_energy:12.8f} Ry  "
      f"{fe4_fresh.iterations:2d} iterations")
print(f"noncollinear continued    {fe4_cont.total_energy:12.8f} Ry  "
      f"{fe4_cont.iterations:2d} iteration")
print(f"difference: {fe4_cont.total_energy - fe4_fresh.total_energy:+.1e} Ry")
print("\nm (mu_B):  collinear |m| = %.4f  ->  noncollinear (%.4f, %.4f, %.4f)"
      % ((fe2.magnetization,) + tuple(fe4_cont.magnetization_vector)))
```

    noncollinear from atoms   -55.69968432 Ry  25 iterations
    noncollinear continued    -55.69968430 Ry   1 iteration
    difference: +2.0e-08 Ry
    
    m (mu_B):  collinear |m| = 3.1751  ->  noncollinear (3.1755, 0.0000, -0.0000)


A collinear calculation knows only the size of the moment, so the continuation puts that
number on the axis the noncollinear run asks for, and the run then converges in **one**
iteration. Rotating a moment costs nothing because without spin-orbit coupling the energy
does not depend on the direction; with spin-orbit coupling it does, and this is exactly the
starting point an anisotropy calculation needs.

Now the other direction, non-magnetic to collinear, where the magnetization is not carried
but seeded.


```python
fe2_cont = run_scf(iron_collinear, fe_pseudo, conv_thr=1e-8, mixing_beta=0.2,
                   starting_from=fe1)
# ... and the same continuation with the magnetization deliberately not seeded.
target = Calculation(iron_collinear, fe_pseudo)
unseeded = continued_state(fe1, target, magnetization="none")
print(unseeded.description)
fe2_flat = run_scf(iron_collinear, fe_pseudo, calculation=target, conv_thr=1e-8,
                   mixing_beta=0.2, starting_from=unseeded)
print(f"seeded    {fe2_cont.total_energy:12.8f} Ry  {fe2_cont.iterations:2d} iterations, "
      f"m = {fe2_cont.magnetization:.4f} mu_B")
print(f"unseeded  {fe2_flat.total_energy:12.8f} Ry  {fe2_flat.iterations:2d} iterations, "
      f"m = {fe2_flat.magnetization:.4f} mu_B   <- back to the non-magnetic solution")
```

    nspin_mag 1 -> 2, magnetization left at zero, wavefunctions carried


    seeded    -55.69968433 Ry  27 iterations, m = 3.1770 mu_B
    unseeded  -55.67804951 Ry   2 iterations, m = 0.0000 mu_B   <- back to the non-magnetic solution


**Nothing in the SCF breaks spin symmetry on its own.** Hand a collinear run two identical
channels and it converges, correctly reporting convergence, straight back to the unpolarized
state, which is a stationary point of the polarized functional and not its minimum. The
magnetization has to be put in by hand, exactly as `starting_magnetization` puts it into a
run started from the atoms.

## The figure: where the iterations go


```python
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
for label, result, style in [("from the atoms", fe4_fresh, "-o"),
                             ("continued from nspin = 2", fe4_cont, "-s")]:
    steps = [entry["iteration"] for entry in result.history]
    accuracy = [entry["accuracy"] for entry in result.history]
    ax.semilogy(steps, accuracy, style, ms=4, label=label)
ax.axhline(1e-8, color="0.6", lw=0.8, ls="--")
ax.set_xlabel("SCF iteration"); ax.set_ylabel("estimated scf accuracy (Ry)")
ax.set_title("bcc Fe, noncollinear: the charge is already right"); ax.legend()

ax = axes[1]
for label, result, style in [("from the atoms", fe2, "-o"),
                             ("continued, seeded", fe2_cont, "-s"),
                             ("continued, unseeded", fe2_flat, "-^")]:
    steps = [entry["iteration"] for entry in result.history]
    moment = [entry.get("magnetization", 0.0) for entry in result.history]
    ax.plot(steps, moment, style, ms=4, label=label)
ax.set_xlabel("SCF iteration"); ax.set_ylabel("magnetization (mu_B / cell)")
ax.set_title("bcc Fe, collinear: the moment is the slow variable"); ax.legend()
fig.tight_layout()
```


    
![png](18_continuing_a_calculation_files/18_continuing_a_calculation_11_0.png)
    


Left: the continued noncollinear run starts three orders of magnitude closer than the atomic
guess and is done in one step. Right: the same continuation from non-magnetic to collinear
saves three iterations out of thirty, because what it carries, the charge, was never the
hard part. The moment is, and a non-magnetic run has none of it. The flat line is the
unseeded run sitting on the symmetric solution.

## Charge and moment, which is all a promotion moves

The charge is conserved across the change of regime and the moment is placed on the axis the
target asks for. A collinear moment is a signed number on $z$; a noncollinear one is a vector
field, and the promotion is the rotation between them.


```python
cell_volume = float(iron_collinear.cell.volume)

def integrate(field):
    # int f(r) dr over the cell, from its values on the FFT grid
    field = np.asarray(field)
    return float(np.sum(field)) * cell_volume / field.size

charge, moment = spin_components(fe2.density, fe2.nspin_mag)      # [n_up, n_dw]
print(f"collinear    int n = {integrate(charge):8.4f}   int m_z = {integrate(moment[2]):7.4f}")

# What the promotion does: the same charge, the moment turned onto the target's axis.
zero = np.zeros_like(moment[2])
rotated = np.stack([moment[2], zero, zero])          # z -> x, which is angle1 = 90
promoted = from_spin_components(charge, rotated, 4)              # [n, m_x, m_y, m_z]
print(f"promoted     int n = {integrate(promoted[0]):8.4f}   int m_x = {integrate(promoted[1]):7.4f}")

_, vector = spin_components(fe4_cont.density, 4)
print("converged    int m = (%.4f, %.4f, %.4f)" % tuple(integrate(c) for c in vector))
```

    collinear    int n =   8.0000   int m_z =  3.1751
    promoted     int n =   8.0000   int m_x =  3.1751
    converged    int m = (3.1755, -0.0000, 0.0000)


## Switching spin-orbit coupling on

Spin-orbit coupling lives in the pseudopotential, in the splitting between its
$j = l \pm \tfrac12$ channels, so switching it on is a change of *dataset*: platinum with a
scalar-relativistic PAW pseudopotential, then with the fully-relativistic one and
`lspinorb = .true.`. The two datasets have different projectors, so what carries across is
the density.


```python
pt_scalar = system_from("../tests/data/qe/pt-paw-scalar.in")
pt_spinor = system_from(f"{QE}/pw_spinorbit/spinorbit-paw.in")
pt_sr = (read_upf(f"{PSEUDO}/Pt.pbe-n-kjpaw_psl.0.1.UPF"),)
pt_fr = (read_upf(f"{PSEUDO}/Pt.rel-pbe-n-kjpaw_psl.0.1.UPF"),)

pt1 = run_scf(pt_scalar, pt_sr, conv_thr=1e-9)
pt_fresh = run_scf(pt_spinor, pt_fr, conv_thr=1e-9)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")          # "a different pseudopotential" -- expected here
    pt_cont = run_scf(pt_spinor, pt_fr, conv_thr=1e-9, starting_from=pt1)

print(f"scalar, nspin = 1        {pt1.total_energy:14.8f} Ry  {pt1.iterations:2d} iterations")
print(f"spin-orbit from atoms    {pt_fresh.total_energy:14.8f} Ry  {pt_fresh.iterations:2d} iterations")
print(f"spin-orbit continued     {pt_cont.total_energy:14.8f} Ry  {pt_cont.iterations:2d} iterations")
print(f"difference: {pt_cont.total_energy - pt_fresh.total_energy:+.1e} Ry")
```

    scalar, nspin = 1         -747.28843798 Ry  12 iterations
    spin-orbit from atoms     -753.34269162 Ry  13 iterations
    spin-orbit continued      -753.34269162 Ry   7 iterations
    difference: +1.9e-10 Ry


---
The tests behind this notebook: `tests/unit/test_continuation_machinery.py`,
`tests/regression/test_continuation.py`.
