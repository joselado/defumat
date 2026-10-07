"""The gaps of an Elk EIGVAL.OUT, the same measures se_probe.py prints for defumat.

    python3 elk_gap.py EIGVAL.OUT
"""
import json
import sys

import numpy as np

HARTREE_EV = 27.211386245988
lines = open(sys.argv[1]).read().splitlines()
nk = int(lines[0].split()[0])
nst = int(lines[1].split()[0])
ks, ev, occ = [], [], []
i = 2
while len(ks) < nk:
    if ": k-point" in lines[i]:
        ks.append([float(x) for x in lines[i].split()[1:4]])
        rows = [lines[i + 2 + j].split() for j in range(nst)]
        ev.append([float(r[1]) for r in rows])
        occ.append([float(r[2]) for r in rows])
        i += 2 + nst
    i += 1
k, e, o = np.array(ks), np.array(ev) * HARTREE_EV, np.array(occ)
nocc = int(round(o[0].sum()))
gamma = int(np.argmin(np.linalg.norm(k, axis=1)))
iv, ic = int(np.argmax(e[:, nocc - 1])), int(np.argmin(e[:, nocc]))
direct = e[:, nocc] - e[:, nocc - 1]
print(json.dumps({
    "nks": nk, "nstsv": nst, "nocc": nocc,
    "gap_mesh_eV": float(e[ic, nocc] - e[iv, nocc - 1]),
    "vbm_k": k[iv].tolist(), "cbm_k": k[ic].tolist(),
    "direct_min_eV": float(direct.min()), "direct_min_k": k[int(np.argmin(direct))].tolist(),
    "gamma_direct_eV": float(direct[gamma]),
    "gamma_bands_eV_rel_vbm": (e[gamma, nocc - 6:nocc + 6] - e[iv, nocc - 1]).round(4).tolist(),
    "kpoints": k.tolist(), "direct_eV": direct.round(4).tolist(),
}))
