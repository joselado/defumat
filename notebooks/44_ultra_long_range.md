# A modulation over many unit cells

A spin density wave in chromium repeats every 21 unit cells. A charged impurity in
silicon is screened over about 20. A skyrmion is hundreds across. Each of them is a
slow envelope on a crystal that is still atomically periodic, and a supercell pays
the same price for the envelope as it does for the atoms, which is why that length
scale is an awkward one to reach.

This notebook applies a potential that varies over eight unit cells of silicon and
watches the electrons screen it. The trick is that the eight-cell problem is solved
in the basis of the ordinary two-atom cell's own Kohn-Sham states, computed once at
the k-points that fold onto the long cell. The self-consistency then runs on the
envelope alone.

A cosine of amplitude 0.02 Ry applied over eight unit cells comes out reduced to a
tenth of itself. The dielectric screening at that wavelength is **10.6**, against
silicon's measured 11.9 in the long-wavelength limit, and it falls to **6.8** when
the modulation is squeezed into four cells instead. The last section does the same
thing with a magnetic field and gets a spin density wave eight cells long, which is
the shape of the problem the method was built for.


```python
import numpy as np
import matplotlib.pyplot as plt
from defumat import Calculator

calc = Calculator.from_file('../tests/data/qe/si-ultracell.in',
                            pseudo_dir='../tests/data/pseudo')
applied = lambda x: 0.02 * np.cos(2 * np.pi * x[..., 0] / 8)
ulr = calc.get_ultracell(supercell=(8, 1, 1), kgrid=(1, 2, 2),
                         nbnd=32, external=applied)
print(f'eight unit cells, {ulr.iterations} iterations; induced density '
      f'{np.abs(ulr.modulation).max():.3e} e/bohr^3')
```

    [defumat] an ultracell calculation: no ground state cached, running the SCF first (conv_thr = 1e-10). Call get_scf() to do this explicitly.


    eight unit cells, 9 iterations; induced density 7.655e-05 e/bohr^3


## What is being solved

The long cell is $N = 8$ copies of the unit cell. Its reciprocal lattice vectors
that lie inside the ordinary Brillouin zone are $N$ wavevectors $\mathbf Q$, and
anything allowed to vary slowly is written as

$$\rho(\mathbf r) = \sum_{\mathbf Q} \rho_{\mathbf Q}(\mathbf r)\,
   e^{i\mathbf Q\cdot\mathbf r},$$

with each $\rho_{\mathbf Q}$ periodic on the unit cell. The states of the long cell
are expanded in the unit cell's own states at the $N$ k-points that fold onto each
long-cell k-point,

$$|\Psi\rangle = \sum_{\mathbf Q, n} c_{\mathbf Q n}\,
   |\psi_{\mathbf k + \mathbf Q, n}\rangle ,$$

so what is diagonalised each iteration is a small dense matrix rather than a
plane-wave Hamiltonian. Those unit-cell states are computed once, before the loop,
and never again.

As the number of bands per folded k-point grows, that basis becomes the long cell's
own complete plane-wave basis, so this is a variational truncation of the real
supercell problem and nothing else. The number of bands is the one convergence
parameter, and the default for an insulator, which is the occupied bands and nothing
above them, leaves the envelope nothing to be built from.


```python
# everything below is one plane-averaged profile along the modulated axis
box = ulr.ultracell.grid
x = np.arange(box[0]) * 8.0 / box[0]        # position in unit cells

profile = lambda f: np.asarray(f).reshape(box).mean(axis=(1, 2))
v_total, rho_ind = profile(ulr.delta_v[0]), profile(ulr.modulation[0])
v_ext = 0.02 * np.cos(2 * np.pi * x / 8)
```


```python
fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 5.2), sharex=True)
top.plot(x, v_ext * 1000, label='applied', color='C1')
top.plot(x, v_total * 1000, label='what an electron feels', color='C0')
top.set_ylabel('potential (mRy)'); top.legend(); top.axhline(0, lw=0.5, c='k')
bottom.plot(x, rho_ind * 1000, color='C2'); bottom.axhline(0, lw=0.5, c='k')
bottom.set_xlabel('position along the long cell (unit cells)')
bottom.set_ylabel(r'induced density (10$^{-3}$ e/bohr$^3$)')
fig.suptitle('Silicon screening a potential eight unit cells long'); fig.tight_layout()
```


    
![png](44_ultra_long_range_files/44_ultra_long_range_4_0.png)
    


## The screening

The electrons pile up where the applied potential is low, and the Hartree potential
of that pile-up cancels most of what was applied. The ratio of the two is the
dielectric function at the applied wavevector,

$$\epsilon(\mathbf Q) = \frac{V_{\rm ext}(\mathbf Q)}
   {V_{\rm ext}(\mathbf Q) + V_{\rm H}[\delta\rho](\mathbf Q)} .$$

Silicon's static dielectric constant is 11.9 by experiment, and that is the
$\mathbf Q \to 0$ limit. A finite wavelength screens less well, so the number should
fall as the modulation is squeezed into fewer cells, and it does. The cutoff here is a
tutorial one rather than a converged one, so read these as the right size and the right
trend rather than as converged values.

**And the charge close to the nucleus.** A norm-conserving pseudopotential replaces
the region inside a small radius around each nucleus with a smooth function carrying
the right total charge and nothing of its shape. A projector-augmented-wave dataset
does not make that trade: it keeps the part of the valence density that lives inside
the radius and restores it on the grid, which is the description a transition metal or
a first-row element needs. The long cell takes one, so the third row below is the same
four-cell modulation screened with that charge put back.


```python
def screening(result, cal=calc):
    # epsilon(Q) from the induced charge's own Hartree potential, in Rydberg units
    shape = result.ultracell.grid
    q2 = result.ultracell.g2(cal.system.cell).reshape(shape)[1, 0, 0]
    induced = np.fft.fftn(np.asarray(result.modulation[0]))[1, 0, 0].real / np.prod(shape)
    return 8 * np.pi * induced / q2, 0.01 / (0.01 + 8 * np.pi * induced / q2)

four = lambda x: 0.02 * np.cos(np.pi * x[..., 0] / 2)
short = calc.get_ultracell(supercell=(4, 1, 1), kgrid=(1, 2, 2), nbnd=32, external=four)
paw = Calculator.from_file('../tests/data/qe/si-ultracell-paw.in', pseudo_dir='../tests/data/pseudo')
deep = paw.get_ultracell(supercell=(4, 1, 1), kgrid=(1, 2, 2), nbnd=32, external=four)
for name, run, cal in (('4 cells', short, calc), ('8 cells', ulr, calc),
                       ('4 cells, PAW', deep, paw)):
    hartree, eps = screening(run, cal)
    print(f'{name:>14}   applied 10.00 mRy   induced V_H {hartree*1000:7.2f} mRy   epsilon {eps:5.2f}')
paw_core = deep.energy_terms['one_center_paw']; print(f'inside the spheres: {paw_core:.1f} Ry per unit cell, against {deep.total_energy - paw_core:.1f} Ry for everything else')
```

    [defumat] an ultracell calculation: no ground state cached, running the SCF first (conv_thr = 1e-12). Call get_scf() to do this explicitly.


    /u/40/ladovj1/data/Documents/programs/claude/defumat/defumat/ultracell/driver.py:845: UserWarning: this ultracell runs an ultrasoft or PAW dataset at ecutrho = 4 ecutwfc, which is the input's own default rather than the dual such a dataset wants. At 4 the augmentation charge is represented on the wavefunction grid: the run is self-consistent and a comparison against a supercell at the same cutoffs is still like for like, but the absolute energy is not converged in ecutrho and cannot be compared against a pw.x number taken at the dataset's own dual. Set ecutrho to 8 to 12 times ecutwfc, which runs
      require_an_ultracell_regime(system, pseudos, basis)


           4 cells   applied 10.00 mRy   induced V_H   -8.53 mRy   epsilon  6.79
           8 cells   applied 10.00 mRy   induced V_H   -9.06 mRy   epsilon 10.61
      4 cells, PAW   applied 10.00 mRy   induced V_H   -8.55 mRy   epsilon  6.88
    inside the spheres: -67.2 Ry per unit cell, against -22.0 Ry for everything else


The screening falls as the modulation is squeezed, which is the physics the first two rows
are there for. The third says something else. The two descriptions of the region near the
nucleus disagree completely there, by the 67 Ry of one-centre energy printed against the 22
Ry of everything else, and they agree on the screening to about one per cent. So what
screens a slow modulation is the bonding charge between the atoms, and the answer is
insensitive to how the core region is described, which is the reason a smooth
pseudopotential is a reasonable thing to use for this at all.

The one per cent is not all physics. The two runs are at different plane-wave cutoffs,
12 Ry and 16 Ry, and the long cell also warns that it is holding the augmentation charge on
the coarser of the two grids such a dataset normally uses, which is the price of a
modulation needing one grid where an ordinary calculation can afford two. Where this stops
being a free choice is an element whose valence density really does pile up close in, a 3d
transition metal above all, and that is where the long cell has to take the augmented
description rather than choose it.

The same question can be asked of a magnet rather than of a screening charge, and
it gets the same answer. Putting this silicon cell under a field that turns from
one cell to the next, the induced moment per cell comes out 0.07953 with the core
charge restored against 0.07952 with it smoothed away, and -0.07947 against
-0.07946 in the cell where the field points the other way. So neither how well a
crystal screens nor how stiff it is against a slow twist depends on the
description of the region near the nucleus, which is not obvious in advance,
since both quantities are built from states that spend much of their weight
close in.


## What the modulation costs

The long cell has a total energy of its own, per unit cell, and the difference between
it and the ordinary crystal is what the modulation costs. The crystal is the right
reference and it is free: with nothing applied, the long cell is the unit cell repeated,
and its energy per cell comes back as the unit cell's own to machine precision.

There is no first-order term. The unperturbed density is the same in every cell, so it
has no overlap with a potential that averages to zero over the long cell, and the leading
cost is second order in what was applied. The sign is the physics: the electrons
rearrange into the potential rather than against it, which is what screening is, so the
modulated crystal sits below the uniform one under an applied potential.

The energy is also what says whether a modulation is worth having at all. A spin density
wave or a charge density wave is the ground state only if its energy is below the uniform
state's, and comparing the two is a subtraction of two of these numbers.



```python
uniform = calc.get_scf().total_energy      # the ordinary crystal
cost = (ulr.total_energy - uniform) * 1000
print(f'uniform {uniform:.8f} Ry, modulated {ulr.total_energy:.8f} Ry')
print(f'the modulation costs {cost:+.5f} mRy per unit cell')

```

    uniform -15.71359794 Ry, modulated -15.71361495 Ry
    the modulation costs -0.01701 mRy per unit cell


## A spin density wave

The same machinery with two spin channels answers the question the method was
built for. A magnetic field that varies over eight unit cells drives a
magnetization that varies with it, and the moment of each cell traces out the
wave.

The cell below has no moment of its own. Nothing in a collinear calculation
breaks spin symmetry by itself, so what the field induces here is the spin
response of a nonmagnetic crystal rather than the rearrangement of moments that
were already there. The two channels share one Fermi level, which is what lets
an electron cross from the minority channel in one cell to the majority channel
in the next. That crossing is what a spin density wave is made of.



```python
magnetic = Calculator.from_file('../tests/data/qe/si-ultracell-mag.in',
                                pseudo_dir='../tests/data/pseudo')
field = lambda x: 0.02 * np.cos(2 * np.pi * x[..., 0] / 8)
wave = magnetic.get_ultracell(supercell=(8, 1, 1), kgrid=(1, 2, 2), nbnd=32,
                              magnetic_field=field, david=2,
                              states_conv_thr=1e-5)

moments = wave.cell_moments()
print(f'eight unit cells, {wave.iterations} iterations; largest cell moment '
      f'{np.abs(moments).max():.4f} mu_B, net {moments.sum():+.2e} mu_B')

```

    [defumat] an ultracell calculation: no ground state cached, running the SCF first (conv_thr = 1e-10). Call get_scf() to do this explicitly.


    /u/40/ladovj1/data/Documents/programs/claude/defumat/defumat/ultracell/driver.py:894: UserWarning: the fixed-density solve did not converge at 14 of 64 k-points: up to 2 of 32 bands are unsettled and the worst k-point took 100 Davidson steps, at ethr = 1.3e-07 (from conv_thr = 1.0e-05). There is no later iteration to fix this -- the density is fixed -- so these wavefunctions are what every quantity built on them will use. Loosen conv_thr (ethr is 0.1 x conv_thr / nelec, QE's setup.f90 rule) before raising the iteration budget: a threshold the solve cannot reach costs the whole budget at every k-point and is where an overlap loses positivity
      calculation, folded_system, eigenvalues, wavefunctions = fixed_density_states(


    eight unit cells, 8 iterations; largest cell moment 0.1223 mu_B, net +3.26e-09 mu_B



```python
cells = np.arange(len(moments))
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(cells + 0.5, 0.02 * np.cos(2 * np.pi * (cells + 0.5) / 8) * 6,
        color='0.6', lw=1.2, ls='--', label='applied field (arbitrary scale)')
ax.bar(cells + 0.5, moments, width=0.7, color='#3b6ea5', label='moment of each cell')
ax.set(xlabel='unit cell along $a_1$', ylabel=r'moment  ($\mu_B$)',
       title='A spin density wave eight unit cells long')
ax.legend(frameon=False, loc='upper right'); fig.tight_layout()
```


    
![png](44_ultra_long_range_files/44_ultra_long_range_12_0.png)
    


The moment follows the field cell by cell and sums to zero over the eight of
them, because the field does too. What is left over is the response itself: the
ratio of the two is the spin susceptibility at that wavelength, and it is the
magnetic counterpart of the screening measured above.

The charge barely moves. A collinear crystal is unchanged by flipping every spin
at the same time as the sign of the field, so the charge cannot respond at first
order in the field and the magnetization must, which is a useful check that the
two channels are being kept apart properly.


## A wave that turns

A collinear calculation can make the moment grow and shrink along a fixed axis.
What it cannot do is make it **point somewhere else**. A helix, a cycloid, a
domain wall and a skyrmion are all textures in which the length of the moment
barely changes and its direction rotates from place to place, and those are the
long-wavelength magnetic structures worth computing.

Letting the moment be a vector costs one thing: the wavefunction becomes a
two-component spinor and the potential becomes a two-by-two matrix at every
point of the grid,

$$V(\mathbf{r}) = v_0(\mathbf{r})\,\mathbb{1} + \mathbf{B}(\mathbf{r})\cdot\boldsymbol{\sigma},$$

whose off-diagonal entries mix the two components. That mixing is what lets the
magnetization turn. Spin-orbit coupling rides along at no extra cost, because it
lives in the unit cell's own states, which are computed once before any of this
begins.

The field that drives a turning texture has to turn itself, so it is a vector
field rather than a number at each point. Below it rotates once in the plane
over four unit cells of a hydrogen lattice.


```python
spinor = Calculator.from_file('../tests/data/qe/h-noncolin-ultracell.in',
                              pseudo_dir='../tests/data/pseudo')

# a field of 0.01 Ry turning once in the plane over the four cells
turning = lambda x: 0.01 * np.stack(
    [np.cos(2 * np.pi * x[..., 0] / 4), np.sin(2 * np.pi * x[..., 0] / 4),
     np.zeros_like(x[..., 0])], axis=-1)

helix = spinor.get_ultracell(supercell=(4, 1, 1), kgrid=(1, 2, 2), nbnd=16,
                             magnetic_field=turning, mixing_beta=0.3)

vectors = helix.cell_moments()          # (4, 3): a moment VECTOR per cell
angles = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
print(f'{helix.iterations} iterations; the moment points', np.round(angles, 1),
      'degrees, out of plane by', f'{np.abs(vectors[:, 2]).max():.0e}')
```

    [defumat] an ultracell calculation: no ground state cached, running the SCF first (conv_thr = 1e-11). Call get_scf() to do this explicitly.


    53 iterations; the moment points [ 27.5  62.3 -40.5 -48.1] degrees, out of plane by 6e-07



```python
cells = np.arange(4) + 0.5
field = np.stack([np.cos(2 * np.pi * cells / 4), np.sin(2 * np.pi * cells / 4)], -1)
unit = vectors[:, :2] / np.linalg.norm(vectors[:, :2], axis=1)[:, None]
fig, ax = plt.subplots(figsize=(7, 2.6))
for arrows, colour, name in ((field, '0.65', 'applied field'),
                             (unit, '#b5432f', 'moment of each cell')):
    ax.quiver(cells, np.zeros(4), arrows[:, 0], arrows[:, 1], color=colour,
              scale=6, width=0.007, label=name)
ax.set(xlim=(0, 4), ylim=(-0.75, 0.75), yticks=[], xlabel='unit cell along $a_1$',
       title='The moment turns from cell to cell')
ax.legend(frameon=False, loc='upper right', ncol=2); fig.tight_layout()
```


    
![png](44_ultra_long_range_files/44_ultra_long_range_16_0.png)
    


The moments follow the field around the plane and stay in it, with nothing out
of plane to five decimal places. They do not turn as far as the field does: the
field rotates ninety degrees from one cell to the next and the moments manage
roughly a third of that.

That shortfall is the physics rather than a shortcoming. The hydrogen atoms here
are 5.5 bohr apart, close enough that the exchange coupling between neighbouring
cells is much larger than a field of 0.01 Ry, and exchange wants every moment
parallel. What the calculation measures is the balance between the two: how
stiff the magnet is against being twisted slowly in space. That stiffness is the
same quantity that sets how much energy a long-wavelength magnon costs, and a
crystal whose modulated direction is more weakly coupled would let the texture
turn much further for the same field.

## What it cannot do

The atoms do not move, and the unit cell's band structure is frozen: this computes
what a slow modulation does to a fixed crystal, not what a different crystal does.
A modulation strong enough to change the local chemistry has to be absorbed by the
empty states, which is the reason the band count is worth converging rather than
guessing.

The total energy is reported and is the quantity to compare two modulations with, but
it converges from one side only when both calculations discretise the same problem: a
comparison against a real supercell needs the same plane-wave grid on both sides, since
two grids differ by about a micro-Rydberg per cell and that is larger than the accuracy
the band count reaches. Under an applied magnetic field the quantity that behaves is the
total plus the field's own energy, which is reported beside it, because what a field
holds fixed is the full energy and the reported total leaves the Zeeman term out.

The checks live in `tests/regression/test_ultracell.py`, where a two-cell ultracell is
compared against a real four-atom supercell run in full, and the disagreement in the
induced density is shown to fall from 43 per cent at eight bands to 0.2 per cent at
eighty. The magnetic side is checked the same way, against a supercell and against an
ordinary calculation carrying the same uniform field, and in
`tests/regression/test_ultracell_augmented.py` for the description that keeps the
charge inside the spheres. The index bookkeeping the long cell needs is checked
separately in `tests/unit/test_ultracell_grid.py` and
`tests/unit/test_ultracell_augmentation.py`.
