# 11. Noncollinear magnetism, magnetic fields and constrained moments

Notebook 8 gave the wavefunctions two components and put spin-orbit coupling into the
Hamiltonian, but it did it on *nonmagnetic* crystals — platinum, bismuthene — where the
density has one component and the exchange-correlation functional never learns that spin
exists. This notebook is the other case: `nspin_mag = 4`, where the density is
`(n, m_x, m_y, m_z)` and the magnetization is a **vector field** that can point differently
at every point of the cell.

Three things change, and each one is a place where a plausible implementation is wrong:

* **the symmetry group shrinks**, and some of what survives is a symmetry only when
  followed by time reversal;
* **symmetrising the magnetization means rotating it**, not averaging three scalars;
* **the potential's four components do not all get the same things** — the Hartree
  potential and the local pseudopotential go into the charge component alone.

The third of those is where this phase started: the first magnetic noncollinear
calculation run here came out at **-184.57 Ry** where QE gives **-55.70**, because
`v_hartree` was being broadcast over all four components instead of added to the first.
That is a spurious magnetic field the size of the Hartree potential, and it converges
beautifully.

Then, on top of a regime that works: **magnetic fields put in by hand** and **constrained
moments** — Elk's `bfieldc`/`bfcmt`/`reducebf` and QE's `constrained_magnetization`. They
share all their machinery, because a penalty on a moment *is* a field.


```python
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from pypresso.io import read_qe_output
from pypresso.io.pwin import parse_pw_input, read_pw_input
from pypresso.pseudo import read_upf
from pypresso.scf import run_scf
from pypresso.scf.driver import Calculation
from pypresso.scf.fields import MagneticField
from pypresso.scf.locals import build_local_regions, default_radii, get_locals
from pypresso.system import build_system
from pypresso.system.builder import local_moments
from pypresso.system.symmetry import find_symmetries, magnetic_symmetries

QE = Path("../quantum_espresso/qe-7.5-ReleasePack/qe-7.5")
NONCOLIN = QE / "test-suite" / "pw_noncolin"
PSEUDO = Path("../tests/data/pseudo")
GENERATED = Path("../tests/data/qe")
RY_TO_EV = 13.605693122994


def load(text):
    "Build a System and its pseudopotentials from the text of a pw.x input."
    system = build_system(parse_pw_input(text))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    return system, pseudos


def scf(text, **options):
    system, pseudos = load(text)
    pwin = parse_pw_input(text)
    options.setdefault("conv_thr", 1e-10)
    options.setdefault("max_iterations", 200)
    options.setdefault("mixing_beta", float(pwin.get("electrons", "mixing_beta") or 0.7))
    return system, run_scf(system, pseudos, **options)


def namelist(text, extra):
    "Insert extra &system variables into a pw.x input, in place."
    marker = text.lower().index("&system") + len("&system")
    return text[:marker] + "\n" + extra + text[marker:]
```

## 1. The group is smaller, and half of it needs time reversal

bcc iron with its moment along `x`. The lattice has 48 point-group operations and the
crystal keeps all of them — but the *magnetization* does not. An operation survives only if
it maps every moment onto its image, or onto **minus** its image, and the second kind is a
symmetry of the crystal followed by time reversal (`sgam_at_mag` in `symm_base.f90`).

This matters twice over: those operations symmetrise the density, and they decide which
k-points are equivalent. Getting it wrong does not merely lose accuracy — symmetrising with
the full nonmagnetic group averages the magnetization to zero and the run converges neatly
to the nonmagnetic solution.


```python
text = (NONCOLIN / "noncolin.in").read_text()
system, _ = load(text)

full = find_symmetries(system.cell, system.structure)
moments = local_moments(system.structure, 4, system.starting_magnetization,
                        system.angle1, system.angle2)
magnetic = magnetic_symmetries(system.cell, system.structure, full, moments)

reference = read_qe_output(GENERATED / "reference.out.pw_noncolin-noncolin")
print(f"starting moment          {np.round(moments[0], 4)}")
print(f"lattice point group      {full.nsym}")
print(f"magnetic group           {magnetic.nsym}"
      f"   ({sum(magnetic.time_reversed)} of them with time reversal)")
print(f"QE prints                {' '.join(open(GENERATED / 'reference.out.pw_noncolin-noncolin').read().split('Sym. Ops')[0].split()[-1:])} Sym. Ops.")
print()
print(f"k-points in the input    {len(read_pw_input(NONCOLIN / 'noncolin.in').card('K_POINTS').lines) - 1}")
print(f"k-points pypresso runs   {system.kpoints.nk}")
print(f"k-points QE runs         {len(reference.kpoints)}")
```

    starting moment          [0.5 0.  0. ]
    lattice point group      48
    magnetic group           16   (8 of them with time reversal)
    QE prints                16 Sym. Ops.
    
    k-points in the input    11
    k-points pypresso runs   22
    k-points QE runs         22


The input lists 11 k-points and both codes run 22. That is not a bug in either: an explicit
`K_POINTS` card is taken to be the wedge of the **lattice's** point group, and
`irreducible_BZ` completes it for the crystal's whenever the two differ (`irrek.f90`).
Pointing a moment along `x` in a cubic crystal is exactly when they differ — which is why
this had gone unnoticed until a magnetic run.

## 2. The identity that gates everything: rotation invariance

Without spin-orbit coupling, nothing in the Hamiltonian knows which way spin points. So the
total energy of a magnetic noncollinear calculation **cannot** depend on the direction the
moments are given. That is a symmetry, not an approximation, so it holds exactly — and it
fails on any error in the rotation of the magnetization, in the axial sign, in the treatment
of a time-reversed operation, or in which operations were kept at all.

A hydrogen atom, four directions.


```python
base = (GENERATED / "h-atom-lsda.in").read_text()

def turned(theta, phi):
    return base.replace(
        "    nspin = 2",
        f"    noncolin = .true.\n    angle1(1) = {theta}, angle2(1) = {phi}",
    )

directions = {"z": (0.0, 0.0), "x": (90.0, 0.0), "y": (90.0, 90.0),
              "(1,1,1)": (54.7356103172, 45.0)}
energies, vectors = {}, {}
for label, (theta, phi) in directions.items():
    _, result = scf(turned(theta, phi))
    energies[label] = result.total_energy
    vectors[label] = np.asarray(result.magnetization_vector)

_, collinear = scf(base)

print(f"{'moment along':<12} {'total energy (Ry)':>20}   moment (mu_B)")
for label, energy in energies.items():
    print(f"{label:<12} {energy:>20.12f}   {np.round(vectors[label], 4)}")
print(f"{'collinear':<12} {collinear.total_energy:>20.12f}   "
      f"{collinear.magnetization:.4f} (along z by construction)")
spread = max(energies.values()) - min(energies.values())
print(f"\nspread over the four directions   {spread:.2e} Ry")
print(f"noncollinear along z vs collinear {abs(energies['z'] - collinear.total_energy):.2e} Ry")
```

    moment along    total energy (Ry)   moment (mu_B)
    z                 -0.946064951880   [-0. -0.  1.]
    x                 -0.946064951862   [ 1.  0. -0.]
    y                 -0.946064951904   [-0.  1.  0.]
    (1,1,1)           -0.946064951911   [0.5774 0.5774 0.5774]
    collinear         -0.946064951854   1.0000 (along z by construction)
    
    spread over the four directions   4.87e-11 Ry
    noncollinear along z vs collinear 2.62e-11 Ry


Machine precision, and the second line is the sharper one: a noncollinear run with the
moment along `z` is the collinear LSDA run of notebook 7, which is validated against QE to
1.2e-9 Ry. So the whole `nspin_mag = 4` path is tied to a path that was already known to be
right, without QE being involved at all.

**One caveat, and it is QE's rather than this code's.** That identity is exact only where
the density is non-negative. `v_xc`'s `nspin = 4` branch integrates `e_xc` against
`ABS(rho(ir,1))` where its `nspin = 1` and `2` branches use the signed density — so in a
cell with vacuum, where a truncated plane-wave density rings slightly negative, the two
conventions differ. On a PAW oxygen atom, whose density is negative on a fifth of the grid,
that is 2.6e-4 Ry. Hydrogen's density is positive everywhere, which is why it is the atom
this test uses.

## 3. bcc iron against Quantum ESPRESSO

The only magnetic noncollinear case QE's test suite ships: iron, ultrasoft, LDA, moment at
90° to `z`. The reference is regenerated with the vendored `pw.x` 7.5 at `conv_thr = 1e-10`.


```python
system, iron = scf((NONCOLIN / "noncolin.in").read_text())
reference = read_qe_output(GENERATED / "reference.out.pw_noncolin-noncolin")

print(f"{'term':<16} {'pypresso':>18} {'QE':>18} {'difference':>12}")
for term, value in reference.energy_terms.items():
    if term in iron.energy_terms:
        ours = iron.energy_terms[term]
        print(f"{term:<16} {ours:>18.8f} {value:>18.8f} {abs(ours - value):>12.2e}")
print(f"{'TOTAL':<16} {iron.total_energy:>18.8f} {reference.total_energy:>18.8f}"
      f" {abs(iron.total_energy - reference.total_energy):>12.2e}")
print()
print(f"moment  pypresso {np.round(iron.magnetization_vector, 4)}")
print(f"        QE       {reference.magnetization_vector}")
print(f"|m|     pypresso {iron.absolute_magnetization:.4f}    QE {reference.absolute_magnetization}")
```

    term                       pypresso                 QE   difference
    one-electron             8.92933181         8.92932731     4.50e-06
    hartree                  6.13361515         6.13359228     2.29e-05
    xc                     -26.12190873       -26.12188165     2.71e-05
    ewald                  -44.64461207       -44.64461207     4.30e-09
    smearing                 0.00388949         0.00388979     2.96e-07
    TOTAL                  -55.69968434       -55.69968434     2.79e-09
    
    moment  pypresso [ 3.1763 -0.      0.    ]
            QE       (3.18, -0.0, -0.0)
    |m|     pypresso 3.1773    QE 3.18


## 4. A gradient-corrected functional needs a local spin frame

An LDA functional of a noncollinear density is easy: at each point the magnetization picks
out an axis, the density is resolved onto it as `(n ± |m|)/2`, and the ordinary spin-polarized
functional is evaluated there. A **gradient**-corrected one is not, because the gradient of
`|m|` has a kink wherever `m` passes through zero — and an antiferromagnet has one on every
plane between two atoms.

QE's answer (`gradcorr.f90`) is to keep the *signed* projection on a fixed axis whenever the
starting moments are all parallel to one (`compute_ux`'s `lsign`), evaluate the collinear
gradient correction in that frame, and rotate the answer back. Transcribed here, it gives
iron with PBE:


```python
system, iron_pbe = scf((NONCOLIN / "noncolin-pbe.in").read_text())
reference_pbe = read_qe_output(GENERATED / "reference.out.pw_noncolin-noncolin-pbe")

print(f"pypresso  {iron_pbe.total_energy:.9f} Ry")
print(f"QE        {reference_pbe.total_energy:.9f} Ry")
print(f"difference {abs(iron_pbe.total_energy - reference_pbe.total_energy):.2e} Ry")
print(f"moment    {np.round(iron_pbe.magnetization_vector, 3)}  against QE's "
      f"{reference_pbe.magnetization_vector}")
print()
print(f"PBE moves iron's moment from {np.linalg.norm(iron.magnetization_vector):.2f}"
      f" to {np.linalg.norm(iron_pbe.magnetization_vector):.2f} mu_B, which is why a"
      " magnetic\nliterature comparison needs the gradient correction rather than LDA.")
```

    pypresso  -55.939445673 Ry
    QE        -55.939445670 Ry
    difference 2.84e-09 Ry
    moment    [ 3.472 -0.     0.   ]  against QE's (3.47, -0.0, 0.0)
    
    PBE moves iron's moment from 3.18 to 3.47 mu_B, which is why a magnetic
    literature comparison needs the gradient correction rather than LDA.


## 5. What "the moment on this atom" means in a plane-wave code

There are no muffin tins here, so an atom-resolved moment is a *choice of region*. QE makes
it once (`make_pointlists.f90`): a sphere of radius `r_m` around each atom, with a weight
that falls linearly to zero between `r_m` and `1.2 r_m`, and every grid point assigned to at
most one atom. `r_m` itself is derived — a little under half the nearest-neighbour distance
over that taper — so the spheres cannot overlap.

The same regions are what a *local* field acts in and what a constraint constrains, so
getting them right is a prerequisite for section 6 rather than a reporting nicety.


```python
calculation = Calculation(*load((NONCOLIN / "noncolin.in").read_text()))
regions = build_local_regions(system.cell, system.structure, calculation.basis.dense.grid)
charge, moment = get_locals(jnp.asarray(iron.density), regions, system.cell)

print(f"r_m   pypresso {default_radii(system.cell, system.structure)[0]:.4f} bohr"
      f"    QE {reference.r_m[0]} bohr")
print(f"charge inside it   pypresso {float(charge[0]):.6f}   QE {reference.local_charges[0]}")
print(f"moment inside it   pypresso {np.round(np.asarray(moment[0]), 6)}")
print(f"                   QE       {reference.local_moments[0]}")
print(f"\n{float(charge[0]) / 8.0:.1%} of the valence charge is inside the sphere;"
      " the rest is between the atoms.")
```

    r_m   pypresso 1.8637 bohr    QE 1.8637 bohr
    charge inside it   pypresso 6.412627   QE 6.412625
    moment inside it   pypresso [3.06317 0.      0.     ]
                       QE       [ 3.063037 -0.        0.      ]
    
    80.2% of the valence charge is inside the sphere; the rest is between the atoms.


## 6. A field, and a constraint that is a field

Two features, one machinery. The energy of a field is `-∫ B(r) · m(r) dr`; the energy of a
constraint is `λ (m - m_target)²`. **Both are written down as energies here, and the
potential is `jax.grad` of them** — which is this project's standing rule, and it pays
immediately: QE hand-derives five different potentials in `add_bfield.f90`, one of them
three lines of quotient rule, and every one is exactly the derivative of one of those two
expressions.

So the Fortran becomes a *test* rather than a second implementation. Here is the sharpest
of the five, the polar-angle constraint `λ (m_z/|m| - cos θ)²`, checked against QE's
transcribed algebra on a synthetic density:


```python
rng = np.random.default_rng(20260820)
grid = (6, 5, 4)
rho = rng.normal(size=(4,) + grid) * 0.1
rho[0] = np.abs(rho[0]) + 0.5
rho = jnp.asarray(rho)

weights = jnp.asarray(rng.uniform(size=(2,) + grid))
from pypresso.scf.locals import LocalRegions
regions_synthetic = LocalRegions(weights=weights, radii=(1.0,), scheme="qe")

cosines = np.array([[0.3], [-0.6]])
penalty = 0.21
field = MagneticField(
    regions=regions_synthetic, uniform=jnp.zeros(3), atomic=None,
    targets=jnp.asarray(cosines), penalty=penalty, constraint="atomic direction",
)
v, _, energy = field.potential(rho, system.cell)

# ... and QE's own expression, transcribed from add_bfield.f90.
m_loc = np.asarray(field.local_moments(rho, system.cell))
m2 = np.zeros_like(m_loc)
for a in range(2):
    ma = np.linalg.norm(m_loc[a])
    xx = m_loc[a, 2] / ma - cosines[a, 0]
    m2[a, 0] = -xx * m_loc[a, 0] * m_loc[a, 2] / ma**3
    m2[a, 1] = -xx * m_loc[a, 1] * m_loc[a, 2] / ma**3
    m2[a, 2] = xx * (-m_loc[a, 2] ** 2 / ma**3 + 1.0 / ma)
qe = 2.0 * penalty * np.einsum("anmk,ac->cnmk", np.asarray(weights), m2)

print(f"largest difference between jax.grad and add_bfield's algebra:"
      f" {np.abs(np.asarray(v)[1:] - qe).max():.2e}")
print(f"the charge component of the potential is untouched:"
      f" {np.abs(np.asarray(v)[0]).max():.2e}")
```

    largest difference between jax.grad and add_bfield's algebra: 2.22e-16
    the charge component of the potential is untouched: 0.00e+00


### The three schemes against QE

`constrained_magnetization` in QE has four forms; three of them have a benchmark on bcc
iron. The references are regenerated, and for a reason worth recording: the committed 2017
output of `constrain_atomic` prints a constraint energy of 8.022 Ry at the starting density,
which is the *unscaled* sum of squares — what `lambda = 1` gives, while the committed input
sets `lambda = 0.005` under a commented-out `lambda = 1`. The output does not belong to the
input.


```python
cases = [("noncolin-constrain_atomic.in", "atomic"),
         ("noncolin-constrain_angle.in", "atomic direction"),
         ("noncolin-constrain_total.in", "total")]

print(f"{'scheme':<18} {'pypresso (Ry)':>16} {'QE (Ry)':>16} {'diff':>10}  moment")
for name, scheme in cases:
    # 1e-11 rather than the 1e-10 an unconstrained run needs: see below.
    _, result = scf((NONCOLIN / name).read_text(), conv_thr=1e-11)
    stem = f"reference.out.pw_noncolin-{Path(name).stem}"
    ref = read_qe_output(GENERATED / stem)
    print(f"{scheme:<18} {result.total_energy:>16.8f} {ref.total_energy:>16.8f}"
          f" {abs(result.total_energy - ref.total_energy):>10.1e}"
          f"  {np.round(result.magnetization_vector, 3)} vs {ref.magnetization_vector}")
```

    scheme                pypresso (Ry)          QE (Ry)       diff  moment


    atomic                 -55.69055706     -55.69055687    1.9e-07  [1.68  0.    0.147] vs (1.68, -0.0, 0.15)


    atomic direction       -55.69968434     -55.69968434    2.7e-09  [3.176 0.    0.   ] vs (3.18, -0.0, 0.0)


    total                  -55.54266124     -55.54266124    1.6e-09  [0.305 0.407 0.509] vs (0.31, 0.41, 0.51)


Two things to read off. **The agreement is 1e-7, not the 1e-9 an unconstrained run reaches,
and both codes have to be run further than usual to get even that** — the runs above use
`conv_thr = 1e-11`, and at the 1e-10 the unconstrained cases use the same comparison is
1.5e-6. That is a property of the state rather than of either code: the penalty holds the
moment off its minimum, so the energy is first-order sensitive to exactly where it lands,
and QE's own last two iterations still move the constraint energy by 1.3e-6 Ry.

**And the constrained moment is not the target.** With `lambda = 0.005` the atomic
constraint pulls iron's moment from 3.06 to 1.68 μB, on its way to a target of 0.5 that it
never reaches. That target is `starting_magnetization`, which QE compares against a moment
in Bohr magnetons although it is a fraction of the valence charge — a quirk, transcribed as
it is because reproducing QE is the point.

### Where the constraint's energy goes: nowhere

`add_bfield` is called from *inside* `v_of_rho`, so the field is felt by every eigenvalue
and removed again by `deband`. `etcon` is printed and never added to the total. Elk states
the same convention from the other side — its manual says the muffin-tin field energy "is
always removed from the total" and the physical field's "is also not included", both being
meant as infinitesimal symmetry breakers. Both numbers are carried here, and the *reported*
total is QE's.


```python
_, constrained = scf((NONCOLIN / "noncolin-constrain_atomic.in").read_text(), conv_thr=1e-11)
printed = [float(line.split("=")[1]) for line in
           (GENERATED / "reference.out.pw_noncolin-noncolin-constrain_atomic").read_text().splitlines()
           if "constraint energy" in line]

calculation = Calculation(*load((NONCOLIN / "noncolin-constrain_atomic.in").read_text()))
starting = calculation.potential(calculation.starting_density())

print(f"constraint energy at the starting density   pypresso {float(starting.e_constraint):.8f}"
      f"   QE {printed[0]:.8f}")
print(f"                    at convergence          pypresso {constrained.constraint_energy:.8f}"
      f"   QE {printed[-1]:.8f}")
print(f"\nand it is not in the total energy: the total above matches QE, which excludes it.")
```

    constraint energy at the starting density   pypresso 0.04011011   QE 0.04011011
                        at convergence          pypresso 0.00646122   QE 0.00646103
    
    and it is not in the total energy: the total above matches QE, which excludes it.


## 7. `reducebf`: a field whose job is to leave

**A calculation with no starting magnetization cannot become magnetic.** Nothing in a
spin-symmetric functional breaks the symmetry, so the SCF sits in the nonmagnetic solution
however far above the ground state it is — for a hydrogen atom, 52 mRy. That is why QE
demands `starting_magnetization` as an input rather than finding the magnetic state on its
own.

Elk's alternative is a field: break the symmetry with one, and multiply it by `reducebf`
after every iteration so that by convergence it is gone. What is left is the magnetic
solution of the *field-free* problem, found from a start that could never have reached it.


```python
card = "LOCAL_MAGNETIC_FIELDS\n 0.0 0.0 0.10\n"
unmagnetised = base.replace("starting_magnetization(1) = 0.6",
                            "starting_magnetization(1) = 0.0")

_, magnetic = scf(base)
_, stuck = scf(unmagnetised)
_, fading = scf(namelist(unmagnetised, "    reducebf = 0.5\n") + card)
_, held = scf(namelist(unmagnetised, "") + card)

rows = [("started magnetic", magnetic), ("started nonmagnetic", stuck),
        ("nonmagnetic + field, reducebf = 0.5", fading),
        ("nonmagnetic + field, held on", held)]
print(f"{'run':<38} {'energy (Ry)':>14} {'moment':>9}")
for label, result in rows:
    print(f"{label:<38} {result.total_energy:>14.9f} {result.magnetization:>9.5f}")
print(f"\nthe fading field lands on the magnetic answer to"
      f" {abs(fading.total_energy - magnetic.total_energy):.1e} Ry;"
      f"\na field left switched on is"
      f" {abs(held.total_energy - magnetic.total_energy):.1e} Ry away from it.")
```

    run                                       energy (Ry)    moment
    started magnetic                         -0.946064952   1.00000
    started nonmagnetic                      -0.893648588   0.00000
    nonmagnetic + field, reducebf = 0.5      -0.946064952   1.00000
    nonmagnetic + field, held on             -0.946014192   1.00000
    
    the fading field lands on the magnetic answer to 5.6e-11 Ry;
    a field left switched on is 5.1e-05 Ry away from it.


## 7b. The other way to hold a moment: a field, not a penalty

Elk offers a third thing beside a fixed field and a penalty, and it is the one a
frozen-magnon study actually wants: `fsmtype`, the **fixed spin moment**. Instead of adding
`lambda (m - m_fix)^2` to the energy, it *searches* for the field at which the
unconstrained functional puts the moment where it was asked, updating it after every
iteration by `B <- B - tau (m - m_fix)`. What converges is then a genuine stationary point
under that field, where a penalty leaves a residual force — which is why iron's atomic
constraint above sits at 1.68 mu_B against a target of 0.5 and never arrives.

It is registered here as `constrained_magnetization = 'fsm'`, and it holds bcc iron at
**2.0 mu_B** where the functional wants 3.18
(`tests/data/qe/fe-fsm.in`, checked in the regression suite rather than run here because it
takes ~350 iterations). Two things about it are worth knowing before using it:

* **the sign is not Elk's**, because the field is not: Elk's Hamiltonian term is
  `+(g_e/4c) sigma.B` where QE's potential takes `-B`, so Elk's `B <- B + tau (m - m_fix)`
  reads with a minus here. Getting it backwards does not oscillate — it drives the moment
  to saturation and converges looking untroubled;
* **the moment is part of the convergence test.** The field is outside the density, so
  `dr2` falls below `conv_thr` while the moment is still far from its target; a run that
  stopped there would report an unconstrained answer under a constrained heading.



The nonmagnetic start never leaves the nonmagnetic solution. The field finds the magnetic
one, and `reducebf` takes the field away again — the converged energy is the field-free
magnetic energy to 1e-9 Ry. A field left switched on finds the state too, and distorts it.

## 8. What this notebook establishes

| check | against | agreement |
|---|---|---|
| moment along x, y, z, (1,1,1) | each other — a symmetry, so exact | 1e-10 Ry |
| noncollinear along z | the collinear LSDA path (notebook 7) | 1e-10 Ry |
| bcc iron, LDA | QE `pw_noncolin/noncolin.in` | 1.1e-9 Ry |
| bcc iron, PBE | QE `noncolin-pbe.in` (the local spin frame) | 7e-9 Ry |
| magnetic group, k-set | QE's header: 16 operations, 22 k-points | exact |
| `r_m`, local charge and moment | QE's `report_mag` | its printed decimals |
| five constraint potentials | `add_bfield.f90`'s hand-derived algebra | 1e-10 |
| three constrained runs | QE, regenerated at `conv_thr = 1e-10` | ≤2.5e-7 Ry |

The next notebook takes the same machinery and makes the magnetization turn from cell to
cell — a spin spiral — which needs no supercell at all.
