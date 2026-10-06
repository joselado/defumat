"""The mesh sum of the band curvature on silicon's little-group wedges: the velocity gauge's linear artefact.

The real-time route carries the diamagnetic current exactly and the
paramagnetic one through every band of the sphere, so on a finite mesh its
linear response is the interband (Kubo) sum plus ``i D/(Omega z)`` in the
conductivity (``tests/regression/test_realtime.py``), with

    D = sum_k w_k d^2/dk_x^2 sum_(n occ) eps_nk     (Hartree bohr^2, on the frozen sphere),

the mesh sum of a total derivative, which vanishes on the whole zone and not
on a mesh. In the susceptibility it is ``chi_D = -D/(Omega z^2)``, ``4 pi``
times that in SI, so it grows as ``1/w^2`` below the gap. This script takes it
on each wedge by dense diagonalisation at ``k +- delta x`` on the sphere of
``k``, so that ``chi1 - chi_D`` of :mod:`tools.realtime.si_third_harmonic`'s runs
is the mesh's interband susceptibility. The ``xx`` curvature is the same at
every member of a star of the ``[100]`` little group, so the wedge sum is the
mesh sum.

    JAX_PLATFORMS=cpu python3 tools/realtime/si_band_curvature.py OUT.json ECUT GRID[,GRID...]
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
from defumat.realtime.dense import dense_hamiltonians
from defumat.realtime.pulse import EV_TO_HA, Adiabatic
from defumat.scf import Calculation
from defumat.workflows.realtime import _kset

repo = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1])
ecut = float(sys.argv[2])
grids = [int(g) for g in sys.argv[3].split(",")]
step = 3e-4
direction = np.array([1.0, 0.0, 0.0])

text = (repo / "tests/data/qe/si2-symmetric.in").read_text()
text = re.sub(r"ecutwfc\s*=\s*[0-9.dD+-]+", f"ecutwfc = {ecut}", text)
path = Path(tempfile.gettempdir()) / f"si2-curvature-{ecut:g}.in"
path.write_text(text)
calculator = Calculator.from_file(path, pseudo_dir=repo / "tests/data/pseudo", announce=False)
scf = calculator.get_scf(conv_thr=1e-10)
omega = 1.55 * EV_TO_HA
shape = Adiabatic(amplitude=1.0, omega=omega, eta=0.1 * EV_TO_HA, polarization=(1, 0, 0))
records = []
for grid in grids:
    t0 = time.time()
    kset, _ = _kset(calculator.system, shape, None, (grid,) * 3, True)
    system = eqx.tree_at(lambda s: s.kpoints, calculator.system, kset)
    calculation = Calculation(system, calculator.pseudos)
    terms = calculation.local_terms(calculation.potential(scf.density).v_scf)
    nocc = int(round(calculation.nelec / 2))
    kcart = np.asarray(kset.cartesian(system.cell))
    weights = np.asarray(kset.weights)
    total = 0.0
    for ik in range(len(weights)):
        sums = [np.sum(np.linalg.eigvalsh(dense_hamiltonians(
            calculation, terms, ik, direction, 0, kcart=kcart + x * direction[None, :])[0])[:nocc])
            for x in (-step, 0.0, step)]
        total += weights[ik] * (sums[0] - 2.0 * sums[1] + sums[2]) / step**2
    curvature = 0.5 * total                      # Ry to Hartree
    volume = float(system.cell.volume)
    record = {"grid": grid, "nk": int(len(weights)), "weights_sum": float(weights.sum()),
              "D_Ha_bohr2": float(curvature), "seconds": time.time() - t0}
    for eta in (0.1, 0.2):
        z = omega + 1j * eta * EV_TO_HA
        chi = -4.0 * math.pi * curvature / (volume * z**2)
        record[f"chi_D_SI_eta{eta}"] = [chi.real, chi.imag]
    records.append(record)
    out.write_text(json.dumps(records, indent=1))
    print(json.dumps(record), flush=True)
print("CURVATURE DONE", flush=True)
