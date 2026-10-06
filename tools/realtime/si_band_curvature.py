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

Its sibling at third order is the band followed adiabatically,
``J = -(1/Omega) sum w d eps/dk (k + kappa)``, whose cubic term
``-(1/Omega) (S4/6) kappa^3`` with ``S4 = sum w d^4 eps_occ/dk_x^4`` gives
``chi(3w) = S4/(18 Omega z^4)`` and ``chi(w) = -S4/(6 Omega (2z - zbar) z^2 zbar)``
through ``chi3_from_orders``'s formulas. Unlike ``D`` it is only the leading
piece of the third-order artefact in ``1/w``: the interband parts carry total
derivatives of their own, and nothing here takes them.

Measured on D22 at 12 Ry (``STEP4 = 0.005``, within 2 per cent of 0.01):
``D`` is -9.71, -2.14, -0.539, -0.128, -0.0116 Ha bohr^2 on 2^3 to 10^3, a
factor of four per step in the mesh, so ``chi_D`` at 1.55 eV and 0.2 eV is
135, 29.8, 7.5, 1.8, 0.16 against a ``chi^(1)`` of about 15; it then changes sign
and sits at +0.0215 (12^3) and +0.034 (16^3), a floor that is the frozen
sphere's at this cutoff and not the mesh's, since at 16 Ry it is +0.0035 at
12^3 and +0.017 at 16^3: half the 12 Ry floor, and worth ``chi_D = -0.47`` at
12 Ry, three per cent of ``chi^(1)``. ``S4`` falls more slowly, 3846, 1344, 607, 279, 122, 50, 7.6 Ha bohr^4 on
2^3 to 16^3, as the ``k^4`` in its weights would have it, and barely moves
with the cutoff (51.6 at 16 Ry, 12^3); its ``chi(3w)`` is 3.5e-18 on 2^3,
the size of the whole answer, and 7e-21 on 16^3.

    JAX_PLATFORMS=cpu python3 tools/realtime/si_band_curvature.py OUT.json ECUT GRID[,GRID...] [STEP4]
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
from defumat.workflows.realtime import CHI3_AU_TO_SI, _kset

repo = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1])
ecut = float(sys.argv[2])
grids = [int(g) for g in sys.argv[3].split(",")]
step = 3e-4
#: the fourth derivative's step, where the stencil's h^2 and the rounding's
#: 1/h^4 are both below a per cent of it
step4 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02
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
    total, total4 = 0.0, 0.0
    for ik in range(len(weights)):
        sums = {x: np.sum(np.linalg.eigvalsh(dense_hamiltonians(
            calculation, terms, ik, direction, 0, kcart=kcart + x * direction[None, :])[0])[:nocc])
            for x in (-step, 0.0, step, -step4, -2 * step4, step4, 2 * step4)}
        total += weights[ik] * (sums[-step] - 2.0 * sums[0.0] + sums[step]) / step**2
        total4 += weights[ik] * (sums[2 * step4] - 4.0 * sums[step4] + 6.0 * sums[0.0]
                                 - 4.0 * sums[-step4] + sums[-2 * step4]) / step4**4
    curvature = 0.5 * total                      # Ry to Hartree
    quartic = 0.5 * total4
    volume = float(system.cell.volume)
    record = {"grid": grid, "nk": int(len(weights)), "weights_sum": float(weights.sum()),
              "D_Ha_bohr2": float(curvature), "S4_Ha_bohr4": float(quartic), "step4": step4,
              "seconds": time.time() - t0}
    for eta in (0.1, 0.2):
        z = omega + 1j * eta * EV_TO_HA
        chi = -4.0 * math.pi * curvature / (volume * z**2)
        record[f"chi_D_SI_eta{eta}"] = [chi.real, chi.imag]
        # the adiabatic intraband current's cubic term, -(1/Omega)(S4/6) kappa^3,
        # through chi3_from_orders' two formulas
        third = quartic / (18.0 * volume * z**4) * CHI3_AU_TO_SI
        first = (-quartic / (6.0 * volume * (2 * z - np.conj(z)) * z**2 * np.conj(z))
                 * CHI3_AU_TO_SI)
        record[f"chi3_3w_S4_SI_eta{eta}"] = [third.real, third.imag]
        record[f"chi3_w_S4_SI_eta{eta}"] = [first.real, first.imag]
    records.append(record)
    out.write_text(json.dumps(records, indent=1))
    print(json.dumps(record), flush=True)
print("CURVATURE DONE", flush=True)
