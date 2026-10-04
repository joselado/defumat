"""Two checks of the prototype that share no code with its solve."""
import numpy as np
import jax.numpy as jnp

import defumat.scf.driver as driver
import defumat.scf.mixing as mixing
from defumat import Calculator
from defumat.scf.occupations import wgauss, smearing_order

calc = Calculator.from_file("local/al-v16.in", announce=False)
result = calc.get_scf()
calculation = calc.calculation
system = calculation.system
dense, cell = calculation.basis.dense, calculation.system.cell
grid = tuple(int(n) for n in dense.grid)
points = int(np.prod(grid))
volume = float(cell.volume)
dv = volume / points

# 1. The LDOS integrates to dN/de_F, computed from the eigenvalues alone.
eig = np.asarray(result.eigenvalues).reshape(1, *np.shape(result.eigenvalues)[-2:])
ef = float(result.fermi_energy)
levels = {"smearing": 0.0, "fermi_energy": ef}
ldos = np.asarray(driver._fermi_ldos(calculation, eig, result.wavefunctions, levels)).reshape(-1)
w = np.asarray(system.kpoints.weights)[None, :, None]
ng, sig = smearing_order(system.smearing), float(system.degauss)
count = lambda e: float(np.sum(w * np.asarray(wgauss(jnp.asarray((e - eig) / sig), ng))))
h = 1e-5
dos = (count(ef + h) - count(ef - h)) / (2 * h)
print(f"integral of LDOS {np.sum(ldos) * dv:.10f}  dN/de_F {dos:.10f}  states/Ry")

# 2. The output solves eps~ x = R, with eps~ written here in numpy.
g2 = np.asarray(dense.kinetic(cell))
index = np.asarray(dense.fft_index)
nonzero = g2 > 1e-12
coulomb = np.where(nonzero, 8 * np.pi / np.where(nonzero, g2, 1.0), 0.0)


def through_g(x, factor):
    box = np.fft.fftn(x.reshape(grid)).reshape(-1)
    out = np.zeros_like(box)
    out[index] = box[index] * factor
    return np.real(np.fft.ifftn(out.reshape(grid))).reshape(-1)


def eps(x, d):
    vx = through_g(x, coulomb)
    return x + d * vx - d * (np.sum(d * vx) * dv) / (np.sum(d) * dv)


rng = np.random.default_rng(1)
residual = through_g(rng.standard_normal(points), np.where(nonzero, np.exp(-g2), 0.0))
shape = (1,) + grid
beta = 0.7
for label, d in (("the SCF's LDOS", ldos), ("a uniform LDOS", np.full(points, ldos.mean()))):
    prec = mixing.ldos_preconditioner(dense, cell, shape, beta=beta)
    prec.update_ldos(d)
    x = np.asarray(prec(residual)) / beta
    rel = np.linalg.norm(eps(x, d) - residual) / np.linalg.norm(residual)
    print(f"{label}: |eps~ x - R| / |R| = {rel:.2e}")
d0 = ldos.mean()
prec = mixing.ldos_preconditioner(dense, cell, shape, beta=beta)
prec.update_ldos(np.full(points, d0))
kerker = mixing.kerker_preconditioner(dense, cell, shape, beta=beta, screening=8 * np.pi * d0)
a, b = np.asarray(prec(residual)), np.asarray(kerker(residual))
print(f"uniform D = {d0:.4e}: |ldos - kerker(8 pi D)| / |kerker| = "
      f"{np.linalg.norm(a - b) / np.linalg.norm(b):.2e}, q_TF^2 = {8 * np.pi * d0:.4f} "
      f"against the cell's Thomas-Fermi {mixing.thomas_fermi_screening(volume, float(calculation.nelec)):.4f}")
