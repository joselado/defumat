"""The perturbative orders by nested jvp through the propagation, against the dense hierarchy.

``HARMONICS-NEXT.md``'s ladder item 3, the check of ``tools/realtime/toy_orders.py``
carried to the code: on zincblende AlAs at a cutoff chosen small for it, the
orders J_(n,m) of the current under ``kappa = lam exp(eta t) cos(w t) e`` from
:func:`defumat.realtime.orders.propagate_orders`, against
:func:`defumat.realtime.dense.dense_orders` with every band of the sphere. The
two share ``H(k)`` and nothing else: the propagation builds its projectors from
the table of ``g_l(q^2)`` and steps in time, the hierarchy builds dense matrices
through ``at_kcart`` (the radial transform and its rows at ``k + G = 0``) and
solves at complex frequencies. Both start from the dense ground states, so the
states are exact eigenstates of the same matrix.

    JAX_PLATFORMS=cpu python3 tools/realtime/orders_vs_dense.py [ecut] [steps_per_period] [eta_T]
"""
import json
import math
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import defumat  # noqa: F401
from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.realtime.dense import dense_ground_states, dense_hamiltonians, dense_orders
from defumat.realtime.orders import propagate_orders
from defumat.realtime.pulse import Adiabatic
from defumat.scf import Calculation
from defumat.system import build_system
from defumat.system.kpoints import KPoints
import equinox as eqx

repo = Path(__file__).resolve().parents[2]
ecut = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
nper = int(sys.argv[2]) if len(sys.argv) > 2 else 400
eta_t = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
omega, eta = 0.05, 0.01                     # Hartree
direction = np.array([1.0, 0.0, 0.0])

text = (repo / "tests/data/qe/alas-shg.in").read_text().replace("ecutwfc = 30.0", f"ecutwfc = {ecut}")
path = Path("/tmp") / f"alas-orders-{ecut}.in"
path.write_text(text)
system = build_system(read_pw_input(path))
pseudos = tuple(read_upf(repo / "tests/data/pseudo" / s.pseudo_file) for s in system.structure.species)
kp = KPoints(coords=np.array([[0.0, 0.0, 0.0], [0.25, 0.1, -0.05]]), weights=np.array([0.5, 0.5]))
system = eqx.tree_at(lambda s: s.kpoints, system, kp)
calculation = Calculation(system, pseudos)
scf_density = calculation.starting_density()
v_scf = calculation.potential(scf_density).v_scf
terms = calculation.local_terms(v_scf)
nocc = int(round(calculation.nelec / 2))
mask = np.asarray(calculation.basis.planewaves.mask)
npwx = mask.shape[1]
weights = np.asarray(calculation.system.kpoints.weights)  # sums to 2
kcart = np.asarray(calculation.system.kpoints.cartesian(calculation.system.cell))

t0 = time.time()
states = np.zeros((len(kcart), nocc, npwx), dtype=complex)
dense = {}
for ik in range(len(kcart)):
    h = dense_hamiltonians(calculation, terms, ik, direction, 4)
    e, u = dense_ground_states(h[0], nocc)
    keep = np.flatnonzero(mask[ik])
    states[ik][:, keep] = u
    part = dense_orders(h, e, u, np.full(nocc, weights[ik]), 2 * omega, 2 * eta, nmax=3)
    for key, value in part.items():
        dense[key] = dense.get(key, 0.0) + value
t_dense = time.time() - t0
volume = float(calculation.system.cell.volume)
dense = {key: -value / (2.0 * volume) for key, value in dense.items()}

period = 2 * math.pi / omega
T = math.ceil(eta_t / eta / period) * period
shape = Adiabatic(amplitude=1.0, omega=omega, eta=eta, eta_t=T * eta, polarization=tuple(direction))
w = np.repeat(weights[:, None], nocc, axis=1)
t0 = time.time()
result = propagate_orders(calculation, jnp.asarray(states), w, v_scf, shape, dt=period / nper,
                          order=3, start=-T, duration=T, k_batch=None)
t_rt = time.time() - t0
rows = []
for n, m in ((1, 1), (2, 2), (2, 0), (3, 3), (3, 1)):
    rt = complex(result.component(n, m, axis=0))
    fd = complex(dense[(n, m)])
    rows.append({"n": n, "m": m, "realtime": [rt.real, rt.imag], "dense": [fd.real, fd.imag],
                 "relative": abs(rt - fd) / abs(fd)})
print(json.dumps({"ecut": ecut, "steps_per_period": nper, "eta_T": T * eta, "npwx": npwx,
                  "nsteps": len(result.times) - 1, "dense_s": t_dense, "realtime_s": t_rt,
                  "rows": rows}), flush=True)
