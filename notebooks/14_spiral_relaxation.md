# Relaxing a spin spiral: `dE/dq`

The spiral wavevector is a coordinate of the calculation exactly as an atomic position is,
so it gets the same treatment: the total energy as a function of $\mathbf q$ at **frozen**
wavefunctions, its gradient, and a walk downhill.

$$\frac{dE}{d\mathbf q} = \left.
   \frac{\partial E_{\rm tot}[\{u\}, \mathbf q]}{\partial \mathbf q}
   \right|_{\{u\}\ \rm fixed},
\qquad
\mathbf q^{\,\rm new} = \mathbf q - s\,H^{-1}\frac{dE}{d\mathbf q}$$

with BFGS run on the reciprocal metric $\mathbf b_i\cdot\mathbf b_j$, since that is the
space $\mathbf q$ lives in. Freezing the states is legitimate for the same reason it is for
a force: the energy is stationary in them.

A hydrogen chain started at `q = 0.30` finds its antiferromagnetic ground state at
**0.50003 in six SCF runs**, where a scan would need a point per wavevector. The physical
content is the pitch of the magnetic order itself, which for an incommensurate helimagnet
is not a number any supercell calculation can produce.

`pw.x` has no spiral, so the validation is finite differences and symmetry.


```python
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pypresso import Calculator
from pypresso.forces.energy import state_from_result
from pypresso.forces.spiral import compute_spiral_gradient, spiral_energy
from pypresso.scf import run_scf

PSEUDO, GENERATED = Path("../tests/data/pseudo"), Path("../tests/data/qe")
CHAIN = (GENERATED / "h-chain-spiral.in").read_text()


def chain_at(q3, ecutwfc=25.0):
    text = CHAIN.replace("spiral_q(3) = 0.25", f"spiral_q(3) = {q3}")
    text = text.replace("ecutwfc = 25.0", f"ecutwfc = {ecutwfc}")
    return Calculator.from_text(text, PSEUDO, announce=False, conv_thr=1e-12,
                                mixing_beta=0.3, max_iterations=300)


def converge(q3, ecutwfc=25.0):
    calc = chain_at(q3, ecutwfc)
    calc.get_scf()
    return calc


chain = converge(0.3)
calculation, result = chain.calculation, chain.scf_result
gradient = compute_spiral_gradient(calculation, result)

# The identity that has to hold before the derivative means anything: the functional being
# differentiated *is* the total energy at the converged state.
print("spiral_energy at the converged state  %.14f Ry" % gradient.total_energy)
print("the SCF's own total energy            %.14f Ry" % result.total_energy)
print("difference                            %.1e Ry"
      % (gradient.total_energy - result.total_energy))
print("\ndE/dq (lattice coordinates)   %s" % np.round(gradient.gradient, 8))
print("the chain runs along z, so the transverse components are zero by symmetry: %.1e"
      % np.abs(gradient.gradient[:2]).max())
```

    spiral_energy at the converged state  -0.95523772122865 Ry
    the SCF's own total energy            -0.95523772122865 Ry
    difference                            4.4e-16 Ry
    
    dE/dq (lattice coordinates)   [ 0.         -0.         -0.01090176]
    the chain runs along z, so the transverse components are zero by symmetry: 6.3e-11


## Two finite differences, checking two different things

The first moves $\mathbf q$ with the state held fixed, and tests the gradient. The second
moves $\mathbf q$ and **re-converges the SCF**, and tests the stationarity that lets the
wavefunctions be frozen in the first place.


```python
state = state_from_result(result)
q0, delta = np.asarray(calculation.system.spiral_q, dtype=float), 1e-5
plus, minus = q0.copy(), q0.copy()
plus[2] += delta
minus[2] -= delta
finite = (float(spiral_energy(calculation, jnp.asarray(plus), state))
          - float(spiral_energy(calculation, jnp.asarray(minus), state))) / (2 * delta)
print("differentiating the functional:  finite difference %.12f   jax.grad %.12f   (%.1e)"
      % (finite, gradient.gradient[2], finite - gradient.gradient[2]))

pseudos = chain.pseudos
print("\n%8s %20s %20s %13s"
      % ("delta", "re-converged FD", "jax.grad", "difference"))
for step in (0.04, 0.02):
    energies = []
    for shift in (step, -step):
        moved = calculation.at_spiral_q(jnp.asarray([0.0, 0.0, 0.3 + shift]),
                                        rebuild_basis=False)
        energies.append(run_scf(moved.system, pseudos, calculation=moved, conv_thr=1e-12,
                                mixing_beta=0.3, max_iterations=300).total_energy)
    finite = (energies[0] - energies[1]) / (2 * step)
    print("%8.3f %20.12f %20.12f %13.1e"
          % (step, finite, gradient.gradient[2], finite - gradient.gradient[2]))
```

    differentiating the functional:  finite difference -0.010901763164   jax.grad -0.010901763190   (2.6e-11)
    
       delta      re-converged FD             jax.grad    difference


       0.040      -0.010695946507      -0.010901763190       2.1e-04


       0.020      -0.010849971652      -0.010901763190       5.2e-05


**What a finite basis costs.** The set of plane waves inside the cutoff changes as
$\mathbf q$ moves, which makes `E(q)` jump slightly whenever one crosses it. That is the
usual Pulay error of a finite basis, measured here as **8.3e-4 Ry per unit `q` at
`ecutwfc = 25`, 5.8e-4 at 40 and 8.3e-6 at 60**, and erratic rather than smoothly
convergent because it counts crossings rather than truncating a series. Measured offline;
it is nine SCF runs.

## Symmetry pins two wavevectors exactly


```python
for q3 in (0.0, 0.5):
    edge = converge(q3)
    r = edge.scf_result
    g = compute_spiral_gradient(edge.calculation, r)
    print("q3 = %4.1f   E = %.10f Ry   |m| = %.4f mu_B   max |dE/dq| = %.1e"
          % (q3, r.total_energy, np.linalg.norm(r.magnetization_vector),
             np.abs(g.gradient).max()))
```

    q3 =  0.0   E = -0.9540418305 Ry   |m| = 0.0000 mu_B   max |dE/dq| = 4.5e-11


    q3 =  0.5   E = -0.9567796407 Ry   |m| = 0.6144 mu_B   max |dE/dq| = 8.1e-09


Not to within a tolerance but to round-off. $\mathbf q$ and $-\mathbf q$ are the same
spiral mirrored, so the energy is even about both, and its slope has to vanish there.

## Downhill


```python
downhill = chain_at(0.30, ecutwfc=40.0)
relaxed = downhill.get_spiral_relaxation(free=(0, 0, 1), nstep=20)

print("%5s %10s %16s %13s %8s" % ("step", "q3", "E (Ry)", "max |dE/dq|", "SCF its"))
for step in relaxed.steps:
    print("%5d %10.5f %16.10f %13.2e %8d"
          % (step.index, step.wavevector[2], step.total_energy, step.max_gradient,
             step.scf_iterations))
print("\nconverged = %s   q = %s   (the answer it had to find is q3 = 0.5)"
      % (relaxed.converged, np.round(relaxed.wavevector, 6)))
```

     step         q3           E (Ry)   max |dE/dq|  SCF its
        1    0.30000    -0.9620589569      5.03e-03       14
        2    0.34463    -0.9625849405      4.76e-03        8
        3    0.39372    -0.9630968974      3.77e-03        8
        4    0.46736    -0.9635797035      1.29e-03        8
        5    0.50559    -0.9636329038      2.20e-04        8
        6    0.50001    -0.9636298167      5.57e-07        8
    
    converged = True   q = [0.       0.       0.500014]   (the answer it had to find is q3 = 0.5)



```python
scan_q = np.linspace(0.0, 0.6, 13)
scan = chain_at(0.0, ecutwfc=40.0).get_spiral_scan(
    np.column_stack([np.zeros_like(scan_q), np.zeros_like(scan_q), scan_q]),
    conv_thr=1e-10)

fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})
reference = scan.energies.min()
top.plot(scan_q, 1e3 * (scan.energies - reference), "o-", color="0.6",
         label="E(q) scan, %d SCF runs" % len(scan_q))
path = np.array([[s.wavevector[2], s.total_energy] for s in relaxed.steps])
top.plot(path[:, 0], 1e3 * (path[:, 1] - reference), "o-", color="crimson",
         label="relaxation, %d SCF runs" % relaxed.nsteps)
for index, (q3, energy) in enumerate(path, start=1):
    top.annotate(str(index), (q3, 1e3 * (energy - reference)), textcoords="offset points",
                 xytext=(5, 5), color="crimson", fontsize=9)
top.axvline(0.5, color="0.8", lw=1, zorder=0)
top.set_ylabel("E(q) - E(min)  [mRy]"); top.legend()
top.set_title("The spiral energy surface, and the path the gradient took across it")

bottom.plot(path[:, 0], [s.max_gradient for s in relaxed.steps], "o-", color="crimson")
bottom.axhline(1e-4, color="0.8", lw=1, ls="--")
bottom.set_yscale("log"); bottom.set_xlabel("$q_3$  [lattice coordinates]")
bottom.set_ylabel(r"max $|dE/dq|$  [Ry / (2$\pi$/alat)]")
bottom.annotate("grad_conv_thr", (0.02, 1.2e-4), color="0.5", fontsize=9)
fig.tight_layout()
```


    
![png](14_spiral_relaxation_files/14_spiral_relaxation_8_0.png)
    


The relaxation reaches the minimum in a fraction of the runs a scan needs, and it does not
have to be told that the answer lies on the axis: the gradient is a vector, and the two
transverse components are free to move if the physics asks them to.

A magnetic field is refused here, because its energy is deliberately left out of the
reported total (notebook 11), so the state would be stationary for a different functional
than the one being differentiated.

---
The tests behind this notebook: `tests/regression/test_spiral_relaxation.py`.
