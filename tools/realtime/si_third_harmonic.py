"""chi^(3)_xxxx of bulk silicon at 1.55 eV by the real-time route, at several meshes and broadenings.

``HARMONICS-NEXT.md``'s ladder item 4, the check of scale against
arXiv:1810.06500 (Uemoto, Kuwabara, Sato and Yabana), whose Table V gives, in
adiabatic LDA, ``|chi^(3)_1111(w)| = 2.2e-18`` and ``|chi^(3)_1111(3w)| = 1.3e-18``
m^2/V^2 and ``chi^(1) = 15.2``. It is a check of scale and not of digits: that
calculation updates the Hartree and exchange-correlation potentials, runs a
20 fs pulse with no explicit broadening and fits a real amplitude with a delay,
on the eight-atom cubic cell at ``a = 10.26`` bohr with a 16^3 mesh of it, and
``3w`` is above the LDA gap, so the value depends on the broadening.

Here: QE's two-atom cell on ``Si.pz-vbc`` at ``ecutwfc`` (12 Ry by default),
the ground state on its own 4x4x4 mesh, whose density is the frozen potential
of every run below; ``Calculator.get_third_harmonic`` along ``[100]`` only
(``chi_xxxx``), on the unshifted mesh reduced by the field's little group. For
each run, ``chi^(1)_xxxx = 4 pi * 2 J_(1,1) / z^2`` in SI from the same orders,
the field normalisation checked once at first order beside the paper's 15.2.

    JAX_PLATFORMS=cpu python3 tools/realtime/si_third_harmonic.py OUT.json ECUT STEPS_PER_PERIOD ETA_T GRID:ETA[,GRID:ETA...]

``GRID:ETA`` is ``8:0.1`` for an 8x8x8 mesh at ``eta = 0.1`` eV. The result of
every run is appended to ``OUT.json`` as it finishes.
"""
import json
import math
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from defumat import Calculator
from defumat.realtime.pulse import EV_TO_HA
from defumat.workflows.realtime import CHI3_AU_TO_SI

repo = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1])
ecut = float(sys.argv[2])
steps_per_period = int(sys.argv[3])
eta_t = float(sys.argv[4])
cases = [(int(g), float(e)) for g, e in (item.split(":") for item in sys.argv[5].split(","))]
frequency = 1.55

text = (repo / "tests/data/qe/si2-symmetric.in").read_text()
text = re.sub(r"ecutwfc\s*=\s*[0-9.dD+-]+", f"ecutwfc = {ecut}", text)
path = Path(tempfile.gettempdir()) / f"si2-third-harmonic-{ecut:g}.in"
path.write_text(text)
calculator = Calculator.from_file(path, pseudo_dir=repo / "tests/data/pseudo", announce=False)
t0 = time.time()
scf = calculator.get_scf(conv_thr=1e-10)
print(f"scf {time.time() - t0:.1f} s, E = {scf.total_energy:.10f} Ry", flush=True)

records = json.loads(out.read_text()) if out.exists() else []
for grid, eta in cases:
    t0 = time.time()
    result = calculator.get_third_harmonic(
        frequency, broadening=eta, both_directions=False, eta_t=eta_t,
        steps_per_period=steps_per_period, grid=(grid, grid, grid))
    seconds = time.time() - t0
    orders = result.orders["100"]
    z = orders.shape.omega + 1j * orders.shape.eta
    chi1 = 4.0 * math.pi * 2.0 * complex(orders.component(1, 1, axis=0)) / z**2
    third, first = result.chi_xxxx_3w, result.chi_xxxx_w
    record = {
        "ecut": ecut, "grid": grid, "eta_eV": eta, "eta_t": eta_t,
        "steps_per_period": steps_per_period, "dt": orders.dt,
        "nsteps": len(orders.times) - 1, "eta_T": float(-orders.times[0] * orders.shape.eta),
        "seconds": seconds,
        "chi1_SI": [chi1.real, chi1.imag],
        "chi3_3w": [third.real, third.imag], "abs_chi3_3w": abs(third),
        "chi3_w": [first.real, first.imag], "abs_chi3_w": abs(first),
        # the delay a real amplitude fitted to this phase would carry, arg/(m w)
        "delay_fs_3w": float(np.angle(third) / (3 * orders.shape.omega) * 2.4188843e-2),
        "delay_fs_w": float(np.angle(first) / orders.shape.omega * 2.4188843e-2),
        "J33": [complex(orders.component(3, 3, axis=0)).real,
                complex(orders.component(3, 3, axis=0)).imag],
        "J31": [complex(orders.component(3, 1, axis=0)).real,
                complex(orders.component(3, 1, axis=0)).imag],
        "transverse_J3_max": float(np.abs(orders.currents[3][:, 1:]).max()),
        "chi3_au_to_si": CHI3_AU_TO_SI,
    }
    records.append(record)
    out.write_text(json.dumps(records, indent=1))
    print(json.dumps(record), flush=True)
print("RUNS DONE", flush=True)
