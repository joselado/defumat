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
the modulation is squeezed into four cells instead.


```python
import numpy as np
import matplotlib.pyplot as plt
from defumat import Calculator

calc = Calculator.from_file('../tests/data/qe/si-ultracell.in',
                            pseudo_dir='../tests/data/pseudo')
applied = lambda x: 0.02 * np.cos(2 * np.pi * x[..., 0] / 8)
ulr = calc.get_ultracell(supercell=(8, 1, 1), kgrid=(1, 2, 2),
                         nbnd=32, external=applied)
print(f'eight unit cells, {ulr.iterations} iterations')
print(f'induced density   {np.abs(ulr.modulation).max():.3e} e/bohr^3')
```

    [defumat] an ultracell calculation: no ground state cached, running the SCF first (conv_thr = 1e-10). Call get_scf() to do this explicitly.


    eight unit cells, 9 iterations
    induced density   7.655e-05 e/bohr^3


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
v_total = profile(ulr.delta_v[0])
rho_ind = profile(ulr.modulation[0])
v_ext = 0.02 * np.cos(2 * np.pi * x / 8)
```


```python
fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 5.2), sharex=True)
top.plot(x, v_ext * 1000, label='applied', color='C1')
top.plot(x, v_total * 1000, label='what an electron feels', color='C0')
top.set_ylabel('potential (mRy)'); top.legend(); top.axhline(0, lw=0.5, c='k')
bottom.plot(x, rho_ind * 1000, color='C2')
bottom.axhline(0, lw=0.5, c='k')
bottom.set_xlabel('position along the long cell (unit cells)')
bottom.set_ylabel(r'induced density (10$^{-3}$ e/bohr$^3$)')
fig.suptitle('Silicon screening a potential eight unit cells long')
fig.tight_layout()
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


```python
def screening(result):
    # epsilon(Q) from the induced charge's own Hartree potential
    shape = result.ultracell.grid
    induced = np.fft.fftn(np.asarray(result.modulation[0])) / np.prod(shape)
    q2 = result.ultracell.g2(calc.system.cell).reshape(shape)[1, 0, 0]
    hartree = 8 * np.pi * induced[1, 0, 0].real / q2     # Rydberg atomic units
    return np.sqrt(q2), 0.01, hartree, 0.01 / (0.01 + hartree)

short = calc.get_ultracell(supercell=(4, 1, 1), kgrid=(1, 2, 2), nbnd=32,
                           external=lambda x: 0.02 * np.cos(2 * np.pi * x[..., 0] / 4))
print(f"{'cells':>6}{'|Q| (1/bohr)':>15}{'applied':>12}{'induced V_H':>14}{'epsilon':>10}")
for cells, run in ((4, short), (8, ulr)):
    q, applied, hartree, eps = screening(run)
    print(f'{cells:>6}{q:>15.4f}{applied*1000:>10.2f} mRy{hartree*1000:>11.2f} mRy{eps:>10.2f}')
```

     cells   |Q| (1/bohr)     applied   induced V_H   epsilon
         4         0.2667     10.00 mRy      -8.53 mRy      6.79
         8         0.1334     10.00 mRy      -9.06 mRy     10.61


## What it cannot do

The atoms do not move, and the unit cell's band structure is frozen: this computes
what a slow modulation does to a fixed crystal, not what a different crystal does.
A modulation strong enough to change the local chemistry has to be absorbed by the
empty states, which is the reason the band count is worth converging rather than
guessing.

There is also no total energy here. The occupied eigenvalues are summed and reported,
and that is all either this code or Elk offers for a calculation of this kind, so the
energy gain of a modulated state over a uniform one is not yet a quantity to read off.

The checks live in `tests/regression/test_ultracell.py`, where a two-cell ultracell is
compared against a real four-atom supercell run in full, and the disagreement in the
induced density is shown to fall from 43 per cent at eight bands to 0.2 per cent at
eighty. The index bookkeeping the long cell needs is checked separately in
`tests/unit/test_ultracell_grid.py`.
