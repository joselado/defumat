# Forces and structural relaxation

The force on an atom is minus the derivative of the total energy with respect to its
position, taken at **frozen** wavefunctions, occupations and eigenvalues:

$$\mathbf F_I = -\left.\frac{\partial E_{\rm tot}
   [\{\psi\}, \{f\}, \boldsymbol\tau]}
   {\partial \boldsymbol\tau_I}\right|_{\psi,\, f,\, \varepsilon\ \rm fixed}$$

Freezing the states loses nothing because the energy is stationary in them at the
converged solution, which is the Hellmann-Feynman theorem; what remains beyond the bare
electrostatics is the Pulay contribution from a basis that moves with the atoms, and, for
an ultrasoft dataset, the augmentation charge's own displacement.

The derivative is taken of the energy itself rather than from hand-derived expressions,
and the six terms Quantum ESPRESSO writes out separately are what it expands into. Against
`pw.x`: **2e-5 Ry/bohr or better** on five references, and a BFGS relaxation that
reproduces QE's geometry to **1e-6 bohr**.


```python
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pypresso import Calculator
from pypresso.forces import compute_forces, frozen_energy, state_from_result
from pypresso.io import read_qe_output
from pypresso.scf import run_scf          # the finite difference below needs it

CASES, PSEUDO = Path("../tests/data/qe"), Path("../tests/data/pseudo")


def load(case):
    return Calculator.from_file(CASES / f"{case}.in", pseudo_dir=PSEUDO,
                                announce=False, conv_thr=1e-10)


def reference(case):
    return read_qe_output(CASES / f"reference.out.{case}")


# What is differentiated below must be the SCF's own total energy at the converged state,
# since otherwise its derivative is the force of some other calculation.
print("%-18s %18s %18s %12s"
      % ("case", "SCF total (Ry)", "frozen functional", "difference"))
for case in ("si2-nc-force", "si2-us-force", "si2-paw-force", "si2-us-pbe-force"):
    calc = load(case)
    scf = calc.get_scf()
    energy = float(frozen_energy(calc.calculation,
                                 calc.system.structure.positions,
                                 state_from_result(scf)))
    print("%-18s %18.10f %18.10f %12.1e"
          % (case, scf.total_energy, energy, energy - scf.total_energy))
```

    case                   SCF total (Ry)  frozen functional   difference


    si2-nc-force           -15.7874037087     -15.7874037087      1.8e-15


    si2-us-force           -22.7454400458     -22.7454400458      3.6e-15


    si2-paw-force          -89.2668867248     -89.2668867248      1.0e-12


    si2-us-pbe-force       -22.8143892008     -22.8143892008      0.0e+00


## The force on a displaced silicon cell

Two atoms pushed off their sites, so that there is a force to compute at all. Against QE,
and against a finite difference of the SCF energy itself, which is the check that trusts
neither implementation. Symmetry is switched off so that nothing is projected out.


```python
import dataclasses

calc = load("si2-nc-force")
forces = calc.get_forces()                       # autodiff, the default
qe = reference("si2-nc-force")

print("%4s %38s %38s" % ("atom", "pypresso (Ry/bohr)", "Quantum ESPRESSO"))
for atom, (ours, theirs) in enumerate(zip(forces.forces, qe.forces)):
    print("%4d  %s   %s" % (atom, " ".join("% .8f" % v for v in ours),
                            " ".join("% .8f" % v for v in theirs)))
print("largest difference %.2e Ry/bohr" % np.abs(forces.forces - qe.forces).max())

nosym = Calculator(dataclasses.replace(calc.system, nosym=True), calc.pseudos,
                   announce=False, conv_thr=1e-12)
autodiff = nosym.get_forces().forces

# The finite difference is of the *same* fixed setup, so it steps the positions
# through `at_positions` rather than building a new calculator each time: that is
# what keeps the basis frozen and the difference free of Pulay error.
plain, pseudos = nosym.calculation, nosym.pseudos


def energy_at(positions):
    moved = plain.at_positions(jnp.asarray(positions))
    return run_scf(moved.system, pseudos, calculation=moved, conv_thr=1e-12).total_energy


h, origin = 2.0e-3, np.asarray(nosym.system.structure.positions)
print("\n%12s %20s %16s %13s"
      % ("coordinate", "finite difference", "autodiff", "difference"))
for atom, direction in ((0, 0), (0, 1), (1, 2)):
    plus, minus = origin.copy(), origin.copy()
    plus[atom, direction] += h
    minus[atom, direction] -= h
    fd = -(energy_at(plus) - energy_at(minus)) / (2 * h)
    print("  atom %d %s   %20.8f %16.8f %13.1e"
          % (atom, "xyz"[direction], fd, autodiff[atom, direction],
             fd - autodiff[atom, direction]))
```

    atom                     pypresso (Ry/bohr)                       Quantum ESPRESSO
       0   0.06039736 -0.00000000  0.00000000    0.06039673  0.00000000  0.00000000
       1  -0.06039736  0.00000000 -0.00000000   -0.06039673  0.00000000  0.00000000
    largest difference 6.26e-07 Ry/bohr


    
      coordinate    finite difference         autodiff    difference


      atom 0 x             0.06076082       0.06076358      -2.8e-06


      atom 0 y            -0.00470094      -0.00470190       9.5e-07


      atom 1 z             0.01005466       0.01005364       1.0e-06


## The same force, term by term

The force splits into the pieces a textbook derives separately: the local potential's
electrostatic pull, the nonlinear core correction, the Ewald sum between the nuclei, the
nonlocal projectors' Pulay term and, for ultrasoft, the augmentation charge. Computing it
both ways is worth doing because the two routes share nothing.

They differ by exactly one term. The extra one is a correction for the density not being
quite converged, and it vanishes as `conv_thr` tightens: at 1e-10 Ry it is already down at
1e-7 Ry/bohr.


```python
us = load("si2-us-force")
system_us, calc_us = us.system, us.calculation
qe_us = reference("si2-us-force")

analytic = us.get_forces(method="analytic")
autodiff_us = us.get_forces(method="autodiff")

from pypresso.system.symmetry import atom_mapping, symmetrize_vector

mapping = atom_mapping(system_us.cell, system_us.structure, calc_us.symmetries)
terms = dict(analytic.terms)
terms["nonlocal"] = terms["nonlocal"] + terms.pop("augmentation")     # QE folds it in

print("%16s %22s %20s %12s"
      % ("term", "pypresso (x, atom 1)", "Quantum ESPRESSO", "difference"))
for name, qe_name in (("ewald", "ionic"), ("local", "local"), ("core", "core"),
                      ("nonlocal", "nonlocal"), ("scf_correction", "scf_correction")):
    ours = np.asarray(symmetrize_vector(np.asarray(terms[name]), system_us.cell,
                                        calc_us.symmetries, mapping))
    theirs = qe_us.force_terms[qe_name]
    print("%16s %22.8f %20.8f %12.1e"
          % (name, ours[0, 0], theirs[0, 0], np.abs(ours - theirs).max()))
print("\ntotal, analytic vs QE   %.1e Ry/bohr"
      % np.abs(analytic.forces - qe_us.forces).max())
print("total, autodiff vs QE   %.1e Ry/bohr"
      % np.abs(autodiff_us.forces - qe_us.forces).max())
```

                term   pypresso (x, atom 1)     Quantum ESPRESSO   difference
               ewald             0.10209222           0.10209221      1.1e-08
               local            -0.10042023          -0.10042054      3.1e-07
                core            -0.00600309          -0.00600313      4.5e-08
            nonlocal             0.06307452           0.06307509      5.7e-07
      scf_correction             0.00000007          -0.00000028      4.0e-07
    
    total, analytic vs QE   2.0e-07 Ry/bohr
    total, autodiff vs QE   1.3e-07 Ry/bohr


## Relaxation

`calculation = 'relax'` walks downhill with BFGS, a trust radius and a Wolfe line search,
in crystal coordinates with the cell metric so that the step is measured in the geometry
the crystal actually has. The symmetry group is fixed at the start and checked afterwards,
so a relaxation moves the atoms within their symmetry and cannot silently lower it.


```python
relax = load("si2-nc-relax")
relaxed = relax.get_relax()
qe_relax = reference("si2-nc-relax")
alat = float(relax.system.cell.alat)

print("%5s %20s %19s %19s" % ("step", "total energy (Ry)", "max |F| (Ry/bohr)",
                              "separation (alat)"))
for step in relaxed.steps:
    separation = (step.positions[1] - step.positions[0]) / alat
    print("%5d %20.8f %19.6f   (%.4f, %.4f, %.4f)"
          % (step.index, step.total_energy, step.max_force, *separation))
print("\nfinal energy   pypresso %.10f   QE %.10f   difference %.1e Ry"
      % (relaxed.total_energy, qe_relax.final_energy,
         relaxed.total_energy - qe_relax.final_energy))
print("final geometry differs from QE by %.1e bohr"
      % np.abs(relaxed.positions - qe_relax.final_positions).max())

fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 3.4))
steps = [s.index for s in relaxed.steps]
left.plot(steps, [s.total_energy for s in relaxed.steps], "o-")
left.set_xlabel("ionic step"); left.set_ylabel("total energy [Ry]")
left.set_title("the energy going downhill"); left.grid(alpha=0.3)
right.semilogy(steps, [max(s.max_force, 1e-12) for s in relaxed.steps], "o-")
right.axhline(1e-3, ls="--", c="k", lw=1, label="forc_conv_thr")
right.set_xlabel("ionic step"); right.set_ylabel("max |F| [Ry/bohr]")
right.set_title("and the force going to zero"); right.legend(); right.grid(alpha=0.3)
fig.tight_layout()
```

     step    total energy (Ry)   max |F| (Ry/bohr)   separation (alat)
        1         -15.78740371            0.060397   (0.2700, 0.2500, 0.2500)
        2         -15.79256155            0.024847   (0.2582, 0.2500, 0.2500)
        3         -15.79359588            0.000366   (0.2499, 0.2500, 0.2500)
        4         -15.79359610            0.000001   (0.2500, 0.2500, 0.2500)
    
    final energy   pypresso -15.7935961045   QE -15.7935961042   difference -2.8e-10 Ry
    final geometry differs from QE by 3.4e-07 bohr



    
![png](09_forces_and_relaxation_files/09_forces_and_relaxation_7_1.png)
    


QE's own CO relaxation runs here too, with the oxygen frozen by `if_pos`, which is how a
constrained geometry, a surface adsorbate or a reaction coordinate is set up.

---
The tests behind this notebook: `tests/regression/test_forces.py`,
`tests/regression/test_relax.py`, `tests/unit/test_force_machinery.py`.
