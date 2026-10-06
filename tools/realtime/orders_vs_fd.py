"""The third order of the current by nested jvp, against a finite difference in the amplitude.

``HARMONICS-NEXT.md``'s ladder item 5. On zincblende AlAs at a cutoff chosen
small for it and on the two k-points of ``tests/regression/test_realtime.py``,
``J^(3)(t)`` from :func:`defumat.realtime.orders.propagate_orders` against the
four-point stencil of full runs of :func:`defumat.realtime.propagate.propagate`
under ``kappa = lam exp(eta t) cos(w t) e`` at ``lam = +-h, +-2h``,

    J^(3) ~ ([J(2h) - J(-2h)] - 2 [J(h) - J(-h)]) / (12 h^3) = J^(3) + 5 h^2 J^(5) + ...,
    J^(1) ~ (8 [J(h) - J(-h)] - [J(2h) - J(-2h)]) / (12 h)   = J^(1) - 4 h^4 J^(5) + ...,

for several ``h``, so the ``h^2`` leg of the truncation and the floor the
rounding sets as ``1/h^2`` are both visible. The two routes share the
propagator, the time grid and the start transient, so the run can be short and
the comparison is of the differentiation alone. The states are the dense ground
states, exact eigenstates of the same ``H(k)``.

    JAX_PLATFORMS=cpu python3 tools/realtime/orders_vs_fd.py [ecut] [steps_per_period] [eta_T] [h,h,...]
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
from defumat.realtime.dense import dense_ground_states, dense_hamiltonians
from defumat.realtime.orders import fourier_component, propagate_orders
from defumat.realtime.propagate import _prepare, propagate
from defumat.realtime.pulse import Adiabatic
from defumat.scf import Calculation
from defumat.system import build_system
from defumat.system.kpoints import KPoints

repo = Path(__file__).resolve().parents[2]
ecut = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
nper = int(sys.argv[2]) if len(sys.argv) > 2 else 400
eta_t = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
hs = [float(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0.005, 0.01, 0.02, 0.04]
omega, eta = 0.05, 0.01                     # Hartree
direction = np.array([1.0, 0.0, 0.0])

text = (repo / "tests/data/qe/alas-shg.in").read_text().replace("ecutwfc = 30.0", f"ecutwfc = {ecut}")
path = Path("/tmp") / f"alas-orders-fd-{ecut}.in"
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
weights = np.asarray(calculation.system.kpoints.weights)
states = np.zeros((len(weights), nocc, mask.shape[1]), dtype=complex)
for ik in range(len(weights)):
    h0 = dense_hamiltonians(calculation, terms, ik, direction, 0)[0]
    _, u = dense_ground_states(h0, nocc)
    states[ik][:, np.flatnonzero(mask[ik])] = u
states = jnp.asarray(states)
w = np.repeat(weights[:, None], nocc, axis=1)

period = 2 * math.pi / omega
T = math.ceil(eta_t / eta / period) * period
dt = period / nper


def shape(amplitude):
    return Adiabatic(amplitude=amplitude, omega=omega, eta=eta, eta_t=T * eta,
                     polarization=tuple(direction))


# the propagator is the same map in both routes only if the centre of the step
# is: it is clamped by the top of the spectrum, which moves with kappa_max
centres = {k: _prepare(calculation, states, w, v_scf, k, dt, "taylor4", None, None).centre
           for k in (0.0, 2 * max(hs))}

t0 = time.time()
orders = propagate_orders(calculation, states, w, v_scf, shape(1.0), dt=dt, order=3,
                          start=-T, duration=T, k_batch=None)
t_jvp = time.time() - t0
times = orders.times
runs, t_runs = {}, 0.0
for lam in sorted({s * f * h for h in hs for f in (1, 2) for s in (1, -1)}):
    t0 = time.time()
    runs[lam] = propagate(calculation, states, w, v_scf, shape(lam), dt=dt, start=-T,
                          duration=T, k_batch=None).current
    t_runs += time.time() - t0

j1, j3 = orders.currents[1], orders.currents[3]
rows = []
for h in hs:
    odd1 = runs[h] - runs[-h]
    odd2 = runs[2 * h] - runs[-2 * h]
    fd3 = (odd2 - 2.0 * odd1) / (12.0 * h**3)
    fd1 = (8.0 * odd1 - odd2) / (12.0 * h)
    row = {"h": h,
           "J3_series": float(np.abs(fd3 - j3).max() / np.abs(j3).max()),
           "J1_series": float(np.abs(fd1 - j1).max() / np.abs(j1).max())}
    for m in (3, 1):
        ref = complex(fourier_component(times, j3, 3, m, omega, eta)[0])
        val = complex(fourier_component(times, fd3, 3, m, omega, eta)[0])
        row[f"J3{m}"] = abs(val - ref) / abs(ref)
    rows.append(row)
print(json.dumps({"ecut": ecut, "steps_per_period": nper, "eta_T": T * eta,
                  "nsteps": len(times) - 1, "npwx": int(mask.shape[1]),
                  "centres_Ry": centres, "jvp_s": t_jvp, "runs_s": t_runs,
                  "J33": [complex(orders.component(3, 3, axis=0)).real,
                          complex(orders.component(3, 3, axis=0)).imag],
                  "J31": [complex(orders.component(3, 1, axis=0)).real,
                          complex(orders.component(3, 1, axis=0)).imag],
                  "J11_abs": abs(complex(orders.component(1, 1, axis=0))),
                  "rows": rows}, indent=1), flush=True)
print("FD DONE", flush=True)
