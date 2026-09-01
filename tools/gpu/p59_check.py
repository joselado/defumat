"""Does the P58/P59 work run on a GPU, and does it give the same number?

GPU.md Phase 0's two questions, asked of the two things added since it was
written: the force-theorem magnetic anisotropy (P58) and QE's ``local-TF``
mixer (P59). Same script on both platforms, which is the harness's own rule --
run it on a CPU first, then in ``p59-gpu.sbatch``, and diff the JSON.

**Three of the four checks need no reference at all**, which is what makes this
worth running on a card rather than a table of numbers to trust:

* a **preconditioner may not move the fixed point**, so ``local-TF`` and the
  plain mixer must converge to the same total energy -- on any backend;
* with the spin-orbit coupling switched off (``soc_scale = 0``) the Hamiltonian
  is invariant under a global spin rotation, so the anisotropy is **exactly
  zero** -- on any backend;
* and the anisotropy at full coupling is a number, compared against the CPU run
  of this same script.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import jax

from pypresso import Calculator
from pypresso.units import RY_TO_EV

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "tests", "data")
QE, PSEUDO = os.path.join(DATA, "qe"), os.path.join(DATA, "pseudo")

out = {"platform": jax.default_backend(),
       "devices": [str(d) for d in jax.devices()],
       "jax": jax.__version__}
print(json.dumps({k: out[k] for k in ("platform", "devices", "jax")}), flush=True)

# --- 1. x64, which every number below depends on -------------------------
out["x64"] = str(jax.numpy.zeros(1).dtype) if not jax.config.jax_enable_x64 \
    else str(jax.numpy.zeros(1, dtype=float).dtype)
assert "64" in out["x64"], f"x64 is off: {out['x64']}"

# --- 2. local-TF must not move the fixed point (P59) ---------------------
t0 = time.time()
silicon = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "benchmarks", "si-1k.in")
plain = Calculator.from_file(silicon, pseudo_dir=PSEUDO, announce=False).get_scf(
    conv_thr=1e-10)
local = Calculator.from_file(silicon, pseudo_dir=PSEUDO, announce=False).get_scf(
    conv_thr=1e-10, mixing_mode="local-TF")
out["si_plain_ry"] = float(plain.total_energy)
out["si_local_tf_ry"] = float(local.total_energy)
out["si_local_tf_delta_ry"] = abs(float(local.total_energy - plain.total_energy))
out["si_local_tf_converged"] = bool(local.converged)
out["si_iterations"] = [int(plain.iterations), int(local.iterations)]
out["t_local_tf_s"] = round(time.time() - t0, 1)
print(json.dumps({k: out[k] for k in out if k.startswith(("si_", "t_local"))}),
      flush=True)

# --- 3 and 4. the anisotropy (P58), coupling off and on ------------------
t0 = time.time()
scalar = Calculator.from_file(os.path.join(QE, "co-hcp-anisotropy-sr.in"),
                              pseudo_dir=PSEUDO, announce=False)
spinor = Calculator.from_file(os.path.join(QE, "co-hcp-anisotropy-soc.in"),
                              pseudo_dir=PSEUDO, announce=False)
scf = scalar.get_scf(conv_thr=1e-10)
out["co_scf_ry"] = float(scf.total_energy)
out["co_scf_converged"] = bool(scf.converged)

free = scalar.get_anisotropy(spinor, directions="xz", soc_scale=0.0)
out["mae_soc_off_mev"] = float(free.anisotropy_mev)
full = scalar.get_anisotropy(spinor, directions="xz")
out["mae_mev"] = float(full.anisotropy_mev)
out["mae_band_energies_ev"] = [float(e * RY_TO_EV) for e in full.band_energies]
out["t_anisotropy_s"] = round(time.time() - t0, 1)

if out["platform"] == "gpu":
    try:
        stats = jax.devices()[0].memory_stats()
        out["peak_device_gb"] = round(stats["peak_bytes_in_use"] / 2**30, 3)
    except Exception as error:                       # not every backend has it
        out["peak_device_gb"] = f"unavailable: {error}"

print(json.dumps(out, indent=1), flush=True)
name = os.environ.get("P59_OUT", f"p59-{out['platform']}.json")
with open(name, "w") as handle:
    json.dump(out, handle, indent=1)
print(f"wrote {name}", flush=True)
