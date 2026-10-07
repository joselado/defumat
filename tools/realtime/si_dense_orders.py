"""chi^(3)_xxxx of silicon at 1.55 eV from the dense frequency-domain hierarchy, mesh by mesh.

The steady state the real-time route converges to, without the propagation:
on each k-point of the ``[100]`` little group's wedge, ``H(k + x e)`` and its
first four derivatives in ``x`` as dense matrices on the frozen sphere, every
band of the sphere, and the hierarchy of :func:`defumat.realtime.dense.dense_orders`
solved at ``m w + i n eta``. It is the same per-k quantity the propagation
gives, so it carries the same velocity-gauge mesh artefact (the mesh sums of
total k-derivatives, :mod:`tools.realtime.si_band_curvature`), and it is the
convergence series in the mesh that the propagation cannot afford; where both
run, they are the same number up to the propagation's start transient and its
step. The current along ``x`` is the same at every member of a star, so the
wedge sum is the mesh sum.

Measured on D22 at 12 Ry, ``|chi_xxxx(3w)|`` in 1e-18 m^2/V^2 on 4^3, 6^3, 8^3,
10^3, 12^3, 16^3, 20^3, 24^3, 28^3: 1.69, 1.14, 1.08, 1.10, 1.02, 0.70, 0.66,
0.67, 0.65 at ``eta = 0.2`` eV and 2.03, 1.44, 1.57, 1.64, 1.77, 0.97, 0.95,
1.13, 0.90 at 0.1 eV; ``|chi_xxxx(w)|`` 8.28, 3.72, 1.82, 0.96, 0.62, 0.81,
0.94, 0.95, 0.93 at 0.2 eV and 9.76, 4.34, 1.94, 1.19, 0.70, 1.09, 1.36, 1.38,
1.35 at 0.1 eV; ``chi^(1)`` 59.0, 26.9, 18.2, 15.5, 14.7, 14.35, 14.31, 14.31,
14.30 at 0.2 eV. So at 0.2 eV both components are converged in the mesh to
two per cent from 20^3 to 28^3, and at 0.1 eV the first-harmonic one is and
the third harmonic oscillates by ten per cent about 1.0. The delays
``arg(chi)/(m w)`` at 20^3 to 28^3 are 0.51 to 0.52 fs for ``chi(w)`` and 0.63
to 0.64 fs for ``chi(3w)`` (modulo its 0.89 fs period) at 0.2 eV. At 16 Ry
against 12 Ry, on 12^3: ``chi^(1)`` +4.3, ``|chi(w)|`` +8 and ``|chi(3w)|``
+1 to +2 per cent. The propagation on 2^3 and 4^3 at 0.2 eV gives these
numbers to 0.85 and 0.7 per cent in ``chi(3w)``, its start transient at
``eta T = 6.5``. arXiv:1810.06500, adiabatic LDA: 1.3 and 2.2, and 15.2.

    JAX_PLATFORMS=cpu python3 tools/realtime/si_dense_orders.py OUT.json ECUT GRID[,GRID...] ETA[,ETA...]
"""
import json
import math
import re
import sys
import tempfile
import time
from pathlib import Path

import equinox as eqx
import numpy as np

from defumat import Calculator
from defumat.realtime.dense import dense_ground_states, dense_hamiltonians, dense_orders
from defumat.realtime.pulse import EV_TO_HA, Adiabatic
from defumat.scf import Calculation
from defumat.workflows.realtime import CHI3_AU_TO_SI, _kset

repo = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1])
ecut = float(sys.argv[2])
grids = [int(g) for g in sys.argv[3].split(",")]
etas = [float(e) for e in sys.argv[4].split(",")]
limit = int(sys.argv[5]) if len(sys.argv) > 5 else None   # k-points, for a timing
omega = 1.55 * EV_TO_HA
direction = np.array([1.0, 0.0, 0.0])

text = (repo / "tests/data/qe/si2-symmetric.in").read_text()
text = re.sub(r"ecutwfc\s*=\s*[0-9.dD+-]+", f"ecutwfc = {ecut}", text)
path = Path(tempfile.gettempdir()) / f"si2-dense-orders-{ecut:g}.in"
path.write_text(text)
calculator = Calculator.from_file(path, pseudo_dir=repo / "tests/data/pseudo", announce=False)
scf = calculator.get_scf(conv_thr=1e-10)
shape = Adiabatic(amplitude=1.0, omega=omega, eta=0.1 * EV_TO_HA, polarization=(1, 0, 0))
records = json.loads(out.read_text()) if out.exists() else []
for grid in grids:
    t0 = time.time()
    kset, _ = _kset(calculator.system, shape, None, (grid,) * 3, True)
    system = eqx.tree_at(lambda s: s.kpoints, calculator.system, kset)
    calculation = Calculation(system, calculator.pseudos)
    terms = calculation.local_terms(calculation.potential(scf.density).v_scf)
    nocc = int(round(calculation.nelec / 2))
    weights = np.asarray(kset.weights)
    volume = float(system.cell.volume)
    sums = {eta: {} for eta in etas}
    nk = len(weights) if limit is None else min(limit, len(weights))
    for ik in range(nk):
        h = dense_hamiltonians(calculation, terms, ik, direction, 4)
        e, u = dense_ground_states(h[0], nocc)
        for eta in etas:
            part = dense_orders(h, e, u, np.full(nocc, weights[ik]), 2 * omega,
                                2 * eta * EV_TO_HA, nmax=3)
            for key, value in part.items():
                sums[eta][key] = sums[eta].get(key, 0.0) + value
        if ik == 0:
            print(f"first k-point {time.time() - t0:.1f} s, npw "
                  f"{int(np.asarray(calculation.basis.planewaves.mask[0]).sum())}", flush=True)
    for eta in etas:
        current = {key: -value / (2.0 * volume) for key, value in sums[eta].items()}
        z = omega + 1j * eta * EV_TO_HA
        chi1 = 4.0 * math.pi * 2.0 * current[(1, 1)] / z**2
        third = -8.0 * current[(3, 3)] / (3.0 * z**4) * CHI3_AU_TO_SI
        first = 8.0 * current[(3, 1)] / (3.0 * (2 * z - np.conj(z)) * z**2 * np.conj(z)) \
            * CHI3_AU_TO_SI
        record = {"grid": grid, "nk": nk, "eta_eV": eta, "ecut": ecut,
                  "seconds": time.time() - t0,
                  "J11": [current[(1, 1)].real, current[(1, 1)].imag],
                  "J33": [current[(3, 3)].real, current[(3, 3)].imag],
                  "J31": [current[(3, 1)].real, current[(3, 1)].imag],
                  "chi1_SI": [chi1.real, chi1.imag],
                  "chi3_3w": [third.real, third.imag], "abs_chi3_3w": abs(third),
                  "chi3_w": [first.real, first.imag], "abs_chi3_w": abs(first)}
        records.append(record)
        print(json.dumps(record), flush=True)
    out.write_text(json.dumps(records, indent=1))
print("DENSE DONE", flush=True)
