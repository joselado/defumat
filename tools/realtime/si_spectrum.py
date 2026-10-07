"""chi^(3)_xxxx of silicon over a frequency axis, from the frequency-domain hierarchy.

``Calculator.get_nonlinear_spectrum`` on the field's ``[100]`` wedge of a
``GRID^3`` mesh at ``ECUT``: the same per-k steady state that
``tools/realtime/si_dense_orders.py`` solves densely at 1.55 eV, on the whole
plane-wave sphere and at every frequency, with the counts of the iterative
solve. One JSON record per frequency, and the run's wall clock.

    JAX_PLATFORMS=cpu python3 tools/realtime/si_spectrum.py OUT.json ECUT GRID ETA W0 W1 DW
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

repo = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1])
ecut, grid, eta = float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
frequencies = np.arange(float(sys.argv[5]), float(sys.argv[6]) + 1e-9, float(sys.argv[7]))
text = (repo / "tests/data/qe/si2-symmetric.in").read_text()
text = re.sub(r"ecutwfc\s*=\s*[0-9.dD+-]+", f"ecutwfc = {ecut}", text)
path = Path(tempfile.gettempdir()) / f"si2-spectrum-{ecut:g}.in"
path.write_text(text)
calculator = Calculator.from_file(path, pseudo_dir=repo / "tests/data/pseudo", announce=False)
calculator.get_scf(conv_thr=1e-10)
t0 = time.time()
spec = calculator.get_nonlinear_spectrum(frequencies, broadening=eta, grid=(grid,) * 3, order=3)
seconds = time.time() - t0
third, first = spec.chi3(axis=0)
z = (frequencies + 1j * eta) * EV_TO_HA
chi1 = 4.0 * math.pi * 2.0 * spec.component(1, 1, axis=0) / z**2
records = [{"grid": grid, "ecut": ecut, "eta_eV": eta, "w_eV": float(w),
            "chi1": [complex(c).real, complex(c).imag],
            "chi3_3w": [complex(a).real, complex(a).imag], "abs_chi3_3w": float(abs(a)),
            "chi3_w": [complex(b).real, complex(b).imag], "abs_chi3_w": float(abs(b)),
            "iterations": int(n)}
           for w, c, a, b, n in zip(frequencies, chi1, third, first, spec.iterations)]
out.write_text(json.dumps({"seconds": seconds, "computed_bands": spec.computed_bands,
                           "records": records}, indent=1))
for r in records:
    print(json.dumps(r), flush=True)
print(json.dumps({"seconds": seconds, "nw": len(frequencies)}), flush=True)
