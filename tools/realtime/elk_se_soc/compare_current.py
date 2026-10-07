"""defumat's current against Elk's JTOT_TD.OUT / Omega, along the field (z) and across it.

    python3 compare_current.py defumat.npz JTOT_TD.OUT [AFIELDT.OUT]

Elk's listed J(t_i) is the paramagnetic current of the states at t_(i+1) plus the
diamagnetic term at A(t_i) (tools/realtime/elk_compare.py's docstring), so it is
compared with defumat both at t_i and at t_(i+1). Elk's coupling is p - A/c and
defumat's p + A/c, so for the same A the first-order currents have the same sign.
"""
import json
import sys

import numpy as np

C_ELK = 137.035999084
d = np.load(sys.argv[1])
t, j, vol = d["times"], d["current"], float(d["volume"])
elk = np.loadtxt(sys.argv[2])
te, je = elk[:, 0], elk[:, 1:4] / vol
n = min(len(t), len(te))
assert np.allclose(t[:n], te[:n], atol=1e-8), (t[:3], te[:3])
dt = t[1] - t[0]
out = {"volume": vol, "steps_compared": n - 1, "dt": dt}
jz, ez = j[:n, 2], je[:n, 2]
scale = np.abs(jz).max()
out["max_abs_jz_defumat"] = float(scale)
out["max_abs_jz_elk"] = float(np.abs(ez).max())
out["t_at_max_defumat"] = float(t[np.argmax(np.abs(jz))])
for shift in (0, 1):
    a = jz[shift:n]
    b = ez[:n - shift]
    out[f"shift{shift}"] = {
        "rel_max_diff": float(np.abs(a - b).max() / scale),
        "rel_l2_diff": float(np.linalg.norm(a - b) / np.linalg.norm(a)),
        "ratio_of_maxima": float(np.abs(b).max() / np.abs(a).max()),
        "lstsq_scale_elk_over_defumat": float(np.dot(a, b) / np.dot(a, a)),
    }
out["max_abs_jxy_defumat"] = float(np.abs(j[:n, :2]).max())
out["max_abs_jxy_elk"] = float(np.abs(je[:n, :2]).max())
# a few sample times
idx = sorted(set([n // 4, n // 2, int(np.argmax(np.abs(jz))), 3 * n // 4, n - 2]))
out["samples"] = [{"t": float(t[i]), "jz_defumat": float(jz[i]), "jz_elk_same_label": float(ez[i]),
                   "jz_elk_label_minus_one": float(ez[i - 1])} for i in idx]
if len(sys.argv) > 3:
    af = np.loadtxt(sys.argv[3], skiprows=1)
    kap = d["kappa"][:n, 2]
    out["afield_max_rel_diff"] = float(np.abs(af[:n, 4] / C_ELK - kap).max() / np.abs(kap).max())
print(json.dumps(out, indent=1))
