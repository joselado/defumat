"""The linear response of the propagation against the Kubo sum with every band.

The first order of the current after a kick, ``J^(1)(t)`` from
:func:`defumat.realtime.orders.propagate_orders` under ``kappa = lam theta(t) e``,
transformed at ``z = w + i eta`` (:func:`defumat.realtime.spectra.conductivity_from_kick`),
against ``optical_conductivity``'s own resolvent sum
(``response/conductivity.py:_resolvent_sum``) fed every eigenstate of the dense
``H(k)`` on the frozen sphere, plus the one term the two do not share,

    sigma_RT(z) = sigma_Kubo(z) + i D / (Omega z),   D = sum_nk w d^2 eps_nk / dk^2,

the mesh sum of the band curvature, which the Kubo sum replaces by its f-sum
value and the propagation carries exactly. ``D`` is taken from the dense
eigenvalues of ``H(k + x e)`` by a central difference in ``x``, which shares
nothing with either side. Two-atom silicon at a small cutoff, at Gamma and at
one general point.

    JAX_PLATFORMS=cpu python3 tools/realtime/linear_vs_kubo.py [ecut] [dt] [eta]
"""
import json
import math
import sys
import time
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import defumat  # noqa: F401
from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.realtime.dense import dense_hamiltonians
from defumat.realtime.orders import propagate_orders
from defumat.realtime.pulse import Kick
from defumat.realtime.spectra import conductivity_from_kick
from defumat.response.conductivity import _resolvent_sum
from defumat.scf import Calculation
from defumat.system import build_system
from defumat.system.kpoints import KPoints

repo = Path(__file__).resolve().parents[2]
ecut = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
dt = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
eta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02          # Hartree
direction = np.array([1.0, 0.0, 0.0])

text = (repo / "tests/data/qe/si2-nosym.in").read_text()
import re
text = re.sub(r"ecutwfc\s*=\s*[0-9.dD+-]+", f"ecutwfc = {ecut}", text)
path = Path("/tmp") / f"si-linear-{ecut}.in"
path.write_text(text)
system = build_system(read_pw_input(path))
pseudos = tuple(read_upf(repo / "tests/data/pseudo" / s.pseudo_file) for s in system.structure.species)
kp = KPoints(coords=np.array([[0.0, 0.0, 0.0], [0.25, 0.1, -0.05]]), weights=np.array([0.5, 0.5]))
system = eqx.tree_at(lambda s: s.kpoints, system, kp)
calculation = Calculation(system, pseudos)
v_scf = calculation.potential(calculation.starting_density()).v_scf
terms = calculation.local_terms(v_scf)
nocc = int(round(calculation.nelec / 2))
mask = np.asarray(calculation.basis.planewaves.mask)
npwx = mask.shape[1]
weights = np.asarray(calculation.system.kpoints.weights)   # sums to 2
volume = float(calculation.system.cell.volume)
frequencies = np.linspace(0.02, 0.6, 30)                    # Hartree

states = np.zeros((len(weights), nocc, npwx), dtype=complex)
sigma_kubo = np.zeros(len(frequencies), dtype=complex)
curvature = 0.0
h_step = 1e-3
for ik in range(len(weights)):
    h = dense_hamiltonians(calculation, terms, ik, direction, 2)
    energies, vectors = np.linalg.eigh(h[0])
    keep = np.flatnonzero(mask[ik])
    states[ik][:, keep] = vectors[:, :nocc].T
    # <m|h1|n> over every band of the sphere; the resolvent sum wants (3, nb, nb)
    element = np.zeros((3,) + h[1].shape, dtype=complex)
    element[0] = vectors.conj().T @ h[1] @ vectors
    wg = np.zeros(len(energies))
    wg[:nocc] = weights[ik]
    filling = np.zeros(len(energies))
    filling[:nocc] = 1.0
    zomega = jnp.asarray(2.0 * (frequencies + 1j * eta))       # Ry
    value, _ = _resolvent_sum(jnp.asarray(element), jnp.asarray(energies), jnp.asarray(wg),
                              jnp.asarray(filling), zomega, 1e-8)
    sigma_kubo += np.asarray(value)[:, 0, 0] / volume
    # the band curvature by a difference of the dense eigenvalues on the frozen sphere
    sums = []
    for x in (-h_step, 0.0, h_step):
        hx = dense_hamiltonians(calculation, terms, ik, direction, 0,
                                kcart=np.asarray(calculation.system.kpoints.cartesian(
                                    calculation.system.cell)) + x * direction[None, :])[0]
        sums.append(np.sum(np.linalg.eigvalsh(hx)[:nocc]))
    curvature += weights[ik] * (sums[0] - 2 * sums[1] + sums[2]) / h_step**2

z = frequencies + 1j * eta
# D/z is blind to the energy unit (Ry/Ry), so D in Ry bohr^2 over z in Ry
sigma_reference = sigma_kubo + 1j * curvature / (volume * 2.0 * z)

T = 22.0 / eta
w = np.repeat(weights[:, None], nocc, axis=1)
t0 = time.time()
result = propagate_orders(calculation, jnp.asarray(states), w, v_scf,
                          Kick(strength=1.0, direction=tuple(direction)),
                          dt=dt, order=1, start=0.0, duration=T, k_batch=None,
                          block_steps=1000)
elapsed = time.time() - t0
kick = conductivity_from_kick(result.times, result.currents[1], 1.0, direction,
                              frequencies, eta)
sigma_rt = kick.sigma[:, 0]
scale = np.abs(sigma_reference).max()
print(json.dumps({
    "ecut": ecut, "dt": dt, "eta": eta, "npwx": npwx, "nsteps": len(result.times) - 1,
    "seconds": elapsed, "curvature_Ry_bohr2": curvature,
    "max_abs_diff": float(np.abs(sigma_rt - sigma_reference).max()),
    "max_abs_diff_without_D": float(np.abs(sigma_rt - sigma_kubo).max()),
    "scale": float(scale),
    "relative": float(np.abs(sigma_rt - sigma_reference).max() / scale),
    "sample": [[float(frequencies[i]), [sigma_rt[i].real, sigma_rt[i].imag],
                [sigma_reference[i].real, sigma_reference[i].imag]] for i in (0, 10, 20, 29)],
}), flush=True)
