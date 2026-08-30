# Effective masses, and where the orbital moment sits

Two quantities Elk computes and `pw.x` does not, both costing one NSCF at a single
k-point. Phase P48.

**The effective mass tensor**

$$\left(\frac{1}{m^*}\right)_{ab} = \frac{1}{2}\,
  \frac{\partial^2 \varepsilon_n(\mathbf k)}{\partial k_a \partial k_b}$$

in units of $1/m_e$. The half is Rydberg atomic units, not a fitted normalisation:
$\hbar^2/2m_e$ is exactly 1 Ry bohr², so a free electron has $\varepsilon = |\mathbf k|^2$
and the tensor is the identity.

**The site angular momenta**

$$\langle L_i\rangle = \sum_s \mathrm{Tr}_m \left(L_i\,\rho_{ss}\right),
  \qquad
  \langle S_i\rangle = \tfrac12 \sum_m \mathrm{Tr}_s\left(\sigma_i\,\rho_{mm}\right),
  \qquad J = L + S$$

from the site density matrix $\rho^a_{(ms),(m's')} = \sum_{n\mathbf k} w_{n\mathbf k}\,
c^a_{ms}\,\overline{c^a_{m's'}}$ built out of the projection
$c = \langle\phi|S|\psi\rangle$ a projected DOS already uses (notebook 16).

**The headline numbers.** Silicon's $\Gamma_{2'}$ conduction mass is **0.1886 $m_e$**
against a literature 0.19, and against the vendored all-electron **Elk** binary the
non-degenerate $\Gamma_1$ curvature agrees to **0.02 %**. Nickel's orbital moment is
**0.0365 $\hbar$** with spin-orbit coupling on and **zero to 10⁻¹⁶** with it off.



```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pypresso import Calculator
from pypresso.system.kpoints import KPoints

PSEUDO = Path("../tests/data/pseudo")

SILICON = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.2, nat = 2, ntyp = 1, ecutwfc = 12.0, nosym = .true.
/
&electrons
  conv_thr = 1e-10
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS crystal
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 4 4 4 0 0 0
"""

silicon = Calculator.from_text(SILICON, pseudo_dir=PSEUDO)
# Converge tightly on purpose: a second derivative divides the
# eigensolver's own scatter by the stencil width.
silicon.get_scf(conv_thr=1e-10)

mass = silicon.get_effective_mass((0.0, 0.0, 0.0), nbnd=10)
print(mass.eigenvalues[:8].round(5), "Ry")

```

    [-0.41622  0.4663   0.4663   0.4663   0.65429  0.65429  0.65429  0.7282 ] Ry


## What comes back, and what is refused

`nosym = .true.` is there for the *second* half of this notebook — $\langle L\rangle$ is
a vector and a wedge sum is a wedge sum — not for the mass, which is taken at one k-point
and never integrates over the zone.

At $\Gamma$ silicon has two non-degenerate bands and two threefold multiplets. **A band
inside a multiplet has no tensor of its own**: the eigensolver's arbitrary rotation within
the manifold rotates it, which is rule D4. Those come back as `nan`, and what is reported
in their place is the multiplet's *summed* inverse mass — the trace, which is invariant.



```python
for m in mass.multiplets[0]:
    kind = "degenerate, sum only" if m.degenerate else "band"
    print(f"bands {str(m.bands):12s} eps = {m.eigenvalue: .5f} Ry  "
          f"(1/m*)_xx = {m.inverse_mass_sum[0, 0]: 9.5f}   {kind}")

print()
print("per-band tensor of the threefold valence top:", mass.inverse_mass[1, 0])
print("Gamma_2' conduction mass:",
      f"{mass.density_of_states_mass(band=7):.4f} m_e  (literature 0.19)")

```

    bands (0,)         eps = -0.41622 Ry  (1/m*)_xx =   0.86518   band
    bands (1, 2, 3)    eps =  0.46630 Ry  (1/m*)_xx = -13.43314   degenerate, sum only
    bands (4, 5, 6)    eps =  0.65429 Ry  (1/m*)_xx =   7.89143   degenerate, sum only
    bands (7,)         eps =  0.72820 Ry  (1/m*)_xx =   5.30190   band
    bands (8,)         eps =  1.03483 Ry  (1/m*)_xx =   3.90877   band
    bands (9,)         eps =  1.05576 Ry  (1/m*)_xx =  -8.35596   band
    
    per-band tensor of the threefold valence top: [nan nan nan]
    Gamma_2' conduction mass: 0.1886 m_e  (literature 0.19)


## The figure: the mass *is* the band curvature

Nothing above touched a band structure. Overlaying the parabola
$\varepsilon(\Gamma) + |\mathbf k|^2/m^*$ on the bands themselves is the check that the
number means what it says — and it is a check the code cannot fake, because the mass came
from a seven-point stencil and the bands from an independent NSCF along a line.



```python
line = np.linspace(-0.16, 0.16, 41)                      # 1/bohr along [100]
tpiba = silicon.calculation.system.cell.tpiba
path = KPoints(coords=np.stack([line / tpiba, 0 * line, 0 * line], axis=1),
               weights=np.full(line.size, 1.0 / line.size))
bands = silicon.get_bands(kpoints=path, nbnd=10)
eps = np.asarray(bands.eigenvalues)                       # (nk, nbnd)

fig, ax = plt.subplots(figsize=(6.2, 4.4))
for band, colour in ((0, "#1f77b4"), (7, "#d62728")):
    curvature = mass.inverse_mass[band, 0, 0]
    ax.plot(line, eps[:, band], "o", ms=3.5, color=colour,
            label=f"band {band}, NSCF")
    ax.plot(line, eps[eps.shape[0] // 2, band] + curvature * line**2, "-",
            color=colour, lw=1.4,
            label=f"$\\varepsilon_\\Gamma + k^2/m^*$, $m^*$ = {1/curvature:.3f} $m_e$")
for band in (1, 2, 3, 4, 5, 6):
    ax.plot(line, eps[:, band], "-", color="0.8", lw=0.9, zorder=0)
ax.set_xlabel(r"$k_x$  (1/bohr),  from $\Gamma$ along [100]")
ax.set_ylabel("energy (Ry)")
ax.set_title("Silicon at $\\Gamma$: the effective mass against the bands")
ax.legend(fontsize=8, loc="center right")
fig.tight_layout()

```


    
![png](29_effective_mass_and_angular_momenta_files/29_effective_mass_and_angular_momenta_5_0.png)
    


The grey curves are the two degenerate multiplets, and they are why the refusal
exists: along this line they *split*, and which branch carries which band index is a
property of the direction rather than of the band.

## Two routes, and why the second one is kept

The first derivative here is **exact**: $\partial\varepsilon_n/\partial k$ is the
generalised Hellmann–Feynman expression built from one `jvp` of $H(\mathbf k)$
(notebook 19). Only the second is a difference. Elk instead differences *eigenvalues*, and
that route is implemented here as `method="eigenvalue"` — it never forms $\partial H/
\partial k$ at all, so agreement between the two is evidence about the velocity operator
rather than about the assembly around it.



```python
other = silicon.get_effective_mass((0.0, 0.0, 0.0), nbnd=10, method="eigenvalue")

print(f"{'band':>6} {'velocity route':>16} {'eigenvalue route':>18} {'differ by':>12}")
for band in (0, 7):
    a = mass.inverse_mass[band, 0, 0]
    b = other.inverse_mass[band, 0, 0]
    print(f"{band:6d} {a:16.9f} {b:18.9f} {abs(a - b):12.2e}")

tensor = mass.inverse_mass[7]
print("\nisotropic to", f"{np.abs(tensor - np.diag(np.diag(tensor))).max():.1e}",
      "with nothing imposing cubic symmetry")
print("O(h^2) truncation the Richardson step removed:",
      f"{mass.truncation[7]:.2e}")

```

      band   velocity route   eigenvalue route    differ by
         0      0.865176513        0.865176508     5.25e-09
         7      5.301896702        5.302611926     7.15e-04
    
    isotropic to 4.9e-09 with nothing imposing cubic symmetry
    O(h^2) truncation the Richardson step removed: 7.40e-02


### Against Elk

Run offline on the same cell at $a = 10.26$ bohr with PBE and a PAW dataset here, against
the vendored Elk binary (all-electron LAPW) at its default stencil. Elk's printed matrix of
eigenvalue derivatives is in Hartree atomic units, which is `inverse_mass` exactly.

| at $\Gamma$ | Elk (LAPW) | velocity | eigenvalue | agrees |
|---|---|---|---|---|
| $\Gamma_{1v}$, band 1 | 0.8603044 | 0.8600902 | 0.8600781 | **0.02 %** |
| $\Gamma_{25'v}$, bands 2–4 (sum) | −13.4317801 | −13.6015326 | −13.6040409 | 1.26 % |
| $\Gamma_{15c}$, bands 5–7 (sum) | 7.7424153 | 7.5886921 | 7.5892084 | 1.99 % |
| $\Gamma_{2'c}$, band 8 | 5.8460134 | 5.8671956 | 5.8682740 | 0.36 % |

**Only one of the two codes converges**, which is what the comparison found. Shrinking
Elk's `deltaem` at $\Gamma$ gives 0.8583, 0.8595, **0.8603**, 0.8642, 0.8697 — rising to a
minimum-error point at its *default* and then diverging. `effmass.f90` puts the centre
k-point in its stencil, and a high-symmetry k-point holds fewer basis functions than any
displaced one: here $\Gamma$ has **725** plane waves against **733**, so the centre
eigenvalue is variationally high by a fixed $\sim$1.2 µRy and the second difference
inherits $\delta/h^2$ — an error that *grows* as the stencil shrinks. The stencil in this
package is centre-free for that reason, and the velocity route never touches the centre at
all. Timings, one core each: Elk 1.08 s, here 4.3 s.

## Where the orbital moment sits

The same projection a projected DOS is made of, contracted with $L$ and $\sigma$ instead of
squared. **$\langle L\rangle$ is quenched to zero without spin-orbit coupling** — nothing
in a scalar-relativistic Hamiltonian locks the orbital moment to the lattice — so silicon
is the null test, and it is an identity rather than a tolerance.



```python
momenta = silicon.get_angular_momenta()
print(momenta.table())
print("\nlargest component of <L> anywhere:",
      f"{np.abs(momenta.orbital).max():.1e}  (machine zero)")

```

    Angular momenta on the ortho-atomic projectors (units of hbar)
    
      Si1  L = (-0.00000, 0.00000, 0.00000)  S = ( 0.00000, 0.00000, 0.00000)  J = (-0.00000, 0.00000, 0.00000)
      Si2  L = ( 0.00000,-0.00000, 0.00000)  S = ( 0.00000, 0.00000, 0.00000)  J = ( 0.00000,-0.00000, 0.00000)
    
      total L = (-0.00000,-0.00000, 0.00000)
      total S = ( 0.00000, 0.00000, 0.00000)
    
    largest component of <L> anywhere: 2.6e-16  (machine zero)


### Nickel, with the coupling switched on

A fully-relativistic *norm-conserving* dataset, so the overlap is the identity and the
projection is exact. The moment is driven along $z$.



```python
NICKEL = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 6.65, nat = 1, ntyp = 1, ecutwfc = 60.0,
  occupations = 'smearing', smearing = 'mv', degauss = 0.02, nosym = .true.,
  noncolin = .true., lspinorb = .true., starting_magnetization(1) = 0.3,
  angle1(1) = 0.0
/
&electrons
  conv_thr = 1e-8, mixing_beta = 0.3
/
ATOMIC_SPECIES
 Ni 58.69 Ni.rel-pbe-nc-dojo.UPF
ATOMIC_POSITIONS crystal
 Ni 0.00 0.00 0.00
K_POINTS automatic
 4 4 4 0 0 0
"""

nickel = Calculator.from_text(NICKEL, pseudo_dir=PSEUDO)
nickel.get_scf(conv_thr=1e-10)
ni = nickel.get_angular_momenta()
atom = ni.atoms[0]
print(atom)
print(f"\n|L|/|S| = {np.linalg.norm(atom.l) / np.linalg.norm(atom.s):.4f}"
      f"   (experiment: m_L/m_S about 0.1)")
print(f"L parallel to S: cos = "
      f"{atom.l @ atom.s / (np.linalg.norm(atom.l) * np.linalg.norm(atom.s)):.8f}"
      f"   (Hund's third rule, more than half filled)")

```

    Ni1  L = (-0.00000,-0.00000, 0.03648)  S = (-0.00000, 0.00000, 0.31270)  J = (-0.00000, 0.00000, 0.34918)
    
    |L|/|S| = 0.1167   (experiment: m_L/m_S about 0.1)
    L parallel to S: cos = 1.00000000   (Hund's third rule, more than half filled)


$\langle L_z\rangle = 0.0365\,\hbar$ against a measured orbital moment of about
0.05 $\mu_B$ — the underestimate GGA is known for, and the reason orbital-polarization
corrections exist. The *ratio* is the better number, and it lands on the experimental one.

**Two checks are quoted rather than run here**, both measured offline. Driving the moment
along $x$ and $y$ instead of $z$ gives $\langle L\rangle$ rotating with it and
$|\langle L\rangle| = 0.0364767$ in **all three**, with a spread of 7.3e-11 over the three cubic axes, with
$\cos(L,S) = 1.00000000$ — nothing in the code imposes that a magnitude is a scalar. And
$\langle S_z\rangle = 0.312699$ against the SCF's own magnetization of 0.61701/2, the gap
being what the projector set does not capture (its charge is 17.90 of 18).

`pw.x` has `lorbm`, which gives the **cell's** orbital magnetization and nothing per atom;
the site decomposition is Elk's tasks 15/16 and is what this is.

---

**Refusals.** A degenerate multiplet has no per-band mass (the sum is reported instead);
a symmetry-reduced k-set has no $\langle L\rangle$ (run the whole unshifted grid); a
fully-relativistic **ultrasoft or PAW** dataset is refused for the angular momenta,
because the spinor overlap's off-diagonal spin blocks are `qq_so`; and a spin spiral is
refused for both.

**Where the detail is.** `PLAN.md` P48a and P48b for the derivations, the stencil trap and
the full Elk comparison; `ELK-FEATURES.md` for the four Elk capabilities surveyed and *not*
taken, with the validation route each would need; `tests/unit/test_effective_mass.py` and
`tests/unit/test_angular_momenta.py` for the assertions.

