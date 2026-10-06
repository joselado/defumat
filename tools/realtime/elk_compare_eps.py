"""defumat's current on Elk's Si-dielectric input against Elk's, and both dielectric functions.

Reads the current ``tools/realtime/time_realtime.py`` saved and an Elk run
directory of the same input at scissor 0 (``JTOT_TD.OUT``, ``AFIELDT.OUT``,
``INFO.OUT``, ``EPSILON_TDRT_11.OUT`` and the independent-particle
``EPSILON_11.OUT`` of task 121), and prints:

* the current in time, ``J_x(t)`` here against ``JTOT_TD/Omega`` from Elk, on
  Elk's own labels (Elk writes the current of step ``i + 1`` at ``t_i``, so the
  comparison is also made one step shifted);
* ``eps_xx`` from both currents through the same transcription of Elk's task 481
  (``tools/realtime/elk_tdrt.py``), against Elk's printed ``EPSILON_TDRT_11`` and
  ``EPSILON_11``, at the peak positions and the static limit.

What is not like-for-like, by construction: Elk's propagation updates the
Hartree and xc potentials at every step (``tddft.f90`` calls ``rhomag`` and
``potkst``), so its ``EPSILON_TDRT`` has local fields and the ALDA where this
one is the independent-particle response, which pairs with Elk's task 121; Elk
propagates in 25 ground-state bands of an LAPW basis, this on the full sphere of
norm-conserving plane waves at 16 Ry.

    python3 tools/realtime/elk_compare_eps.py <defumat.npz> <elk run directory>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from elk_tdrt import epsilon_481, read_epsilon, read_jtot  # noqa: E402

from defumat.units import HARTREE_TO_EV  # noqa: E402


def peaks(w, values, low=0.1, high=0.25):
    """Positions of the two largest maxima of ``values`` between ``low`` and ``high`` Hartree."""
    inside = (w > low) & (w < high)
    ww, vv = w[inside], values[inside]
    local = [i for i in range(1, len(vv) - 1) if vv[i] > vv[i - 1] and vv[i] >= vv[i + 1]]
    local.sort(key=lambda i: -vv[i])
    return sorted(float(ww[i]) * HARTREE_TO_EV for i in local[:2])


def main(npz_path, elk_dir):
    elk_dir = Path(elk_dir)
    ours = np.load(npz_path)
    times, current, volume = ours["times"], ours["current"], float(ours["volume"])
    elk_times, jtot = read_jtot(elk_dir / "JTOT_TD.OUT")
    a0 = np.loadtxt(elk_dir / "AFIELDT.OUT", skiprows=1)[0, 2:5]
    elk_j = jtot / volume
    n = min(len(elk_times), len(times))
    same = np.abs(current[:n, 0] - elk_j[:n, 0]).max()
    shifted = np.abs(current[1:n, 0] - elk_j[:n - 1, 0]).max()
    scale = np.abs(elk_j[:n, 0]).max()
    # eps through the same post-processing; Elk's ntimes = tstime/dtimes + 1 rows
    full_elk = np.vstack([jtot, jtot[-1:]])
    grid = np.arange(len(full_elk)) * float(elk_times[1] - elk_times[0])
    w, eps_elk_rebuilt = epsilon_481(grid, full_elk, a0, volume)
    w, eps_ours = epsilon_481(times[:len(full_elk)], current[:len(full_elk)] * volume, a0, volume)
    w_tdrt, eps_tdrt = read_epsilon(elk_dir / "EPSILON_TDRT_11.OUT")
    w_ip, eps_ip = read_epsilon(elk_dir / "EPSILON_11.OUT")
    out = {
        "current_scale": float(scale),
        "max_dJ_same_labels": float(same), "max_dJ_one_step_shifted": float(shifted),
        "eps_rebuilt_vs_elk_tdrt": float(np.abs(eps_elk_rebuilt[:, 0, 0] - eps_tdrt).max()),
        "static_ours": complex(eps_ours[1, 0, 0]).real,
        "static_elk_tdrt": complex(eps_tdrt[1]).real, "static_elk_ip": complex(eps_ip[1]).real,
        "peaks_im_ours_eV": peaks(w, eps_ours[:, 0, 0].imag),
        "peaks_im_elk_tdrt_eV": peaks(w_tdrt, eps_tdrt.imag),
        "peaks_im_elk_ip_eV": peaks(w_ip, eps_ip.imag),
        "max_im_ours": float(eps_ours[:, 0, 0].imag.max()),
        "max_im_elk_tdrt": float(eps_tdrt.imag.max()), "max_im_elk_ip": float(eps_ip.imag.max()),
    }
    print(json.dumps(out), flush=True)
    np.savez(Path(npz_path).with_suffix(".eps.npz"), w=w, ours=eps_ours[:, 0, 0],
             elk_tdrt=eps_tdrt, elk_ip=eps_ip, times=times, current=current[:, 0],
             elk_times=elk_times, elk_current=elk_j[:, 0])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
