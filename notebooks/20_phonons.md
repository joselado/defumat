# Phonons at $\Gamma$: the second derivative of the energy

A phonon frequency is an eigenvalue of the force constant matrix, the second derivative of
the total energy with respect to the atomic positions, divided by the masses:

$$ C_{i j} = \frac{\partial^2 E}{\partial u_i\,\partial u_j},
\qquad
\det\!\left[\frac{C_{ij}}{\sqrt{M_i M_j}} - \omega^2\right] = 0 $$

Two things contribute. The atoms move in a frozen electronic cloud, and the electrons then
follow them, which is a linear response problem and the reason a phonon costs more than a
force:

$$ \frac{\partial^2 E}{\partial u_i\,\partial u_j}
   = \underbrace{\partial_i \partial_j L}_{\text{atoms move}}
   + \underbrace{(\partial_\psi \partial_j L)\cdot \frac{d\psi}{du_i}}_{\text{electrons follow}} $$

Both come from differentiating the force of notebook 09 one more time, along a direction
that carries the positions, the states and the density together, so nothing is derived by
hand and no second-order wavefunction is needed.

On the silicon QE runs with `ph.x`:

| | pypresso | `ph.x` |
|---|---|---|
| optical mode ($\Gamma_{25'}$, triply degenerate) | **510.102** cm⁻¹ | 510.152 |
| acoustic modes (zero by translation invariance) | 4.09 | 2.05 |

The optical mode agrees to $9.7\times10^{-5}$ relative. The acoustic residue is a
diagnostic rather than an answer, and section 4 is about what it measures.


```python
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pypresso import Calculator
from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.response import dynamical_matrix
from pypresso.scf import Calculation, run_scf
from pypresso.system import build_system

CASES = Path("../tests/data/qe")
PSEUDO = Path("../tests/data/pseudo")

silicon = Calculator.from_file(CASES / "si-epsilon.in", pseudo_dir=PSEUDO,
                              announce=False, conv_thr=1e-12)
system, pseudos = silicon.system, silicon.pseudos
calculation = silicon.calculation
scf = silicon.get_scf()

print(f"total energy   {scf.total_energy:.9f} Ry")
print(f"pw.x           -15.84452726 Ry")
```

    total energy   -15.844527263 Ry
    pw.x           -15.84452726 Ry


## 1. The dynamical matrix

One call. Inside it there is a bare perturbation per mode, six here, a self-consistent
response solved for all of them together, and then the second derivative itself.


```python
phonons = silicon.get_phonons()

print(f"converged in {len(phonons.history)} iterations, "
      f"{phonons.average_iterations:.0f} CG steps per band per solve")
print()
for n, f in enumerate(phonons.frequencies):
    print(f"  freq ({n + 1}) = {f:12.6f} cm-1")
print()
print("  ph.x:      2.045258 x3,   510.151844 x3")
```

    converged in 10 iterations, 28 CG steps per band per solve
    
      freq (1) =     4.087839 cm-1
      freq (2) =     4.087839 cm-1
      freq (3) =     4.087839 cm-1
      freq (4) =   510.102374 cm-1
      freq (5) =   510.102374 cm-1
      freq (6) =   510.102374 cm-1
    
      ph.x:      2.045258 x3,   510.151844 x3


The force constants, in Ry/bohr². Two atoms, so a $6\times6$ matrix, and because the two
silicons are related by symmetry there are only two independent numbers in it:


```python
on_site = phonons.matrix[0, :, 0, :]
between = phonons.matrix[0, :, 1, :]
print("D[Si_1, Si_1] =\n", np.round(on_site, 9))
print("D[Si_1, Si_2] =\n", np.round(between, 9))
print()
print(f"isotropic on-site block to      {np.abs(on_site - np.eye(3) * on_site[0, 0]).max():.1e}")
print(f"asymmetry max|D - D^T|          {phonons.asymmetry:.1e}")
print(f"acoustic sum rule D_00 + D_01   {on_site[0, 0] + between[0, 0]:.3e}")
```

    D[Si_1, Si_1] =
     [[ 0.276582  0.        0.      ]
     [ 0.        0.276582 -0.      ]
     [ 0.       -0.        0.276582]]
    D[Si_1, Si_2] =
     [[-0.27654648 -0.          0.        ]
     [-0.         -0.27654648  0.        ]
     [ 0.         -0.         -0.27654648]]
    
    isotropic on-site block to      5.6e-17
    asymmetry max|D - D^T|          1.4e-16
    acoustic sum rule D_00 + D_01   3.552e-05


## 2. What a phonon actually is

The optical mode is the two sublattices moving against each other, and what makes it cost
energy is the charge that has to rearrange when they do. The induced density $d\rho/du$ is
that charge.

**Left**, the charge induced by displacing one silicon along $x$, summed down one crystal
axis so that the whole cell is in view. It is antisymmetric about the displacement, as it
must be, since charge leaves one side of the atom and arrives on the other, and the second
sublattice, which has not moved, responds too. That rearrangement is the screening, and
paying for it is most of what makes the optical mode cost 510 cm⁻¹ rather than nothing.

**Right**, the check that costs nothing and tests everything. Displace *every* atom by the
same vector and the crystal has merely been translated, so the response must be the
ground-state density's own gradient, $\sum_a d\rho_{a,x} = -\partial\rho/\partial x$. The
right-hand side comes from differentiating $\rho$ in reciprocal space and shares no
machinery with the left. They agree to $6.5\times10^{-5}$, and only because the response is
*screened*: with the bare perturbation alone the two curves differ by 52%.


```python
from pypresso.basis.fft import g_to_r, r_to_g

gvectors = calculation.basis.dense
rho_g = r_to_g(np.asarray(scf.density)[0], gvectors.fft_index)
cartesian = gvectors.cartesian(calculation.system.cell)
exact = -np.real(g_to_r(1j * cartesian[:, 0] * rho_g, gvectors.fft_index, gvectors.grid))
translated = phonons.induced_density[:, 0, 0].sum(axis=0)

fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.0))

one_atom = phonons.induced_density[0, 0, 0].sum(axis=2)
limit = np.abs(one_atom).max()
image = left.imshow(one_atom.T, origin="lower", cmap="RdBu_r",
                    vmin=-limit, vmax=limit, interpolation="bilinear")
left.set(title=r"$d\rho/du$: one Si displaced along $x$",
         xlabel="crystal axis 1", ylabel="crystal axis 2")
fig.colorbar(image, ax=left, label="induced charge")

line = np.s_[:, 5, 5]
grid = np.arange(exact.shape[0])
right.plot(grid, exact[line], lw=3, alpha=0.45, label=r"$-\partial\rho/\partial x$")
right.plot(grid, translated[line], "--", lw=1.6, color="crimson",
           label=r"$\sum_a d\rho_{a,x}$  (screened)")
right.set(title="a rigid translation is a translation", xlabel="grid point along axis 1",
          ylabel="density response")
right.legend(frameon=False)
fig.tight_layout()

error = np.abs(translated - exact).max() / np.abs(exact).max()
print(f"max relative difference over the whole grid: {error:.1e}")
```

    max relative difference over the whole grid: 6.5e-05



    
![png](20_phonons_files/20_phonons_7_1.png)
    


## 3. How it is checked

`ph.x` is one of four references, and the other three come from inside the calculation
itself:

| check | what it reaches | result |
|---|---|---|
| a rigid translation reproduces $-\partial\rho/\partial x$ | the solve, the kernel, the symmetrisation | 6.5e-5 relative |
| finite-differenced forces (displace, re-converge, difference) | the **response** half of the derivative | 2.1e-5 Ry/bohr² of 0.2865 |
| the wedge against the whole closed grid | the two symmetrisations | 2.7e-14 |
| `ph.x` frequencies | all of it | 0.05 cm⁻¹ |

The second is the one that matters most. A finite difference of forces at *frozen* density
would give back only the first of the two terms; only a re-converged SCF has let the
electrons follow, so only a difference of converged forces sees the response. It is quoted
here rather than run, at two extra SCF runs per column.

## 4. The acoustic modes, which are a diagnostic and not an answer

Translating a whole crystal costs no energy, so three frequencies are zero *exactly*. What
comes out instead, 4.09 cm⁻¹ here and 2.05 from `ph.x`, is the plane-wave basis's own
error: the basis does not follow the atoms, so the energy depends slightly on where they
sit relative to it. Both numbers are $10^{-4}$ of the force constants, and neither is
physics.

Neither code imposes the sum rule by default, because the residue is the cheapest
diagnostic there is and hiding it would hide a real error just as effectively.
`acoustic_sum_rule=True` imposes it when a spectrum is what is wanted.


```python
imposed = silicon.get_phonons(acoustic_sum_rule=True)
print("without the sum rule:", np.array2string(phonons.frequencies, precision=3))
print("with it            :", np.array2string(imposed.frequencies, precision=3))
```

    without the sum rule: [  4.088   4.088   4.088 510.102 510.102 510.102]
    with it            : [-9.041e-06 -6.024e-06 -3.547e-06  5.101e+02  5.101e+02  5.101e+02]


## 5. A metal

Everything above is an insulator, where the occupied manifold is sharply separated from the
empty one. In a metal it is not: displacing an atom moves states across the Fermi level, so
the occupations themselves respond, and the response is weighted by how partially filled
each state is.

The cell below is two-atom aluminium, fcc with its cell doubled, so three of the six modes
are acoustic and three are folded in from the zone boundary.


```python
aluminium = Calculator.from_file(CASES / "al2-metal.in", pseudo_dir=PSEUDO,
                                 announce=False, conv_thr=1e-12)
metal = aluminium.get_phonons()

QE = np.array([1.108857, 1.827469, 1.924700, 146.710511, 146.714378, 311.035401])
print("        here        ph.x")
for here, there in zip(metal.frequencies, QE):
    print(f"  {here:10.4f}  {there:10.4f}")

residue = np.abs(metal.matrix.sum(axis=2)).max()
print(f"\nacoustic sum rule  {residue:.2e} Ry/bohr^2, against on-site force")
print(f"constants of       {np.abs(metal.matrix[0, :, 0, :]).max():.4f}")
```

            here        ph.x
          0.6997      1.1089
          1.6067      1.8275
          1.7247      1.9247
        146.7093    146.7105
        146.7132    146.7144
        311.0335    311.0354
    
    acoustic sum rule  1.06e-05 Ry/bohr^2, against on-site force
    constants of       0.0476


The three real modes land within **0.0019 cm⁻¹** of `ph.x`, tighter than silicon's 0.05,
and the folded pair splits by 0.0039 where `ph.x` splits it by 0.0039. That near-degeneracy
is the sharper statement: this cell is run with `nosym`, so nothing symmetrises the
assembled matrix and the splitting has to come out of the calculation rather than be
imposed on it.

The acoustic sum rule is the number to watch in a metal. A response that counts the
occupations wrongly still converges, still returns a symmetric matrix, and puts the optical
modes at 198 and 309 cm⁻¹ with the missing weight parked in the acoustic ones at 155.7.
Nothing but the sum rule and the reference says so.

The occupations' own first-order change looks like a separate term and is not: the
$(f_i - f_j)/(\varepsilon_i - \varepsilon_j)$ structure of the smeared projector already is
that term, and it vanishes identically for an insulator, where every occupied $f$ is 1.

## 6. Ultrasoft and PAW

With an ultrasoft dataset the overlap operator moves with the atoms, since the augmentation
charge and the projectors travel with their own nucleus. That changes the constraint the
states are subject to, and so changes what a second derivative of the energy means:

| | pypresso | `ph.x` |
|---|---|---|
| norm-conserving | **510.102** cm⁻¹ | 510.152 |
| **ultrasoft** | **513.295** | 513.275 |
| **PAW** | **513.378** | 513.404 |

Everything the two datasets add vanishes identically when the overlap is the identity, which
is what keeps the norm-conserving number unchanged to round-off.

What is still refused is one combination rather than a dataset: an ultrasoft or PAW
**metal**, where the moving overlap and the responding occupations meet.

---
The tests behind this notebook: `tests/regression/test_phonons.py`.

**What is left.** This is $\Gamma$ only. A phonon at $\mathbf q \neq 0$ needs the perturbed
states at $\mathbf k + \mathbf q$ as well as at $\mathbf k$, and a dispersion needs the
Fourier interpolation on top of that. Without it there is one point of one.
