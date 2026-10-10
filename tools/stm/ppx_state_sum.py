"""Separate the states from the weights in the al-metal-nosym STM gap.

pp.x's plot_num = 7 dumps |psi_nk|^2 / omega for every one of the 64 x 6
states (``local_dos`` with ``iflag = 0``, ``wg = 1``); this script

1. rebuilds ``stm.f90``'s image from those dumps and pw.x's own eigenvalues,
   with its window rule transcribed literally, and checks it against pp.x's
   own STM dump (``al.stm.b20``) -- if this does not reproduce it, the
   transcription is wrong and nothing below means anything;
2. compares every state's density with defumat's, multiplet by multiplet
   (rule D4: a single member of a degenerate multiplet is not comparable);
3. builds the image both ways, each with each code's window rule.

``RUN`` is the directory ``ppx_states.sh`` filled (pw.x's ``out/``, the
``psi_K###_B###`` dumps and ``al.stm.b20``); ``REPO`` is the checkout to import.
``OPEN.md`` Part XXIV item 1 is what it found (2026-10-10, D22): the literal
rule reproduces pp.x's image to 1.2e-12 and this code's image is the sum
without the stale ``first_band`` to 3.2e-10.

usage: RUN=<dir> REPO=<checkout> python3 tools/stm/ppx_state_sum.py
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

REPO = os.environ.get("REPO", os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
RUN = os.environ["RUN"]
HA = 2.0  # Ry per Ha


def read(path):
    lines = open(path).read().splitlines()
    n1x, n2x, _, _, _, n3, nat, ntyp = (int(x) for x in lines[1].split())
    values = np.array(" ".join(lines[4 + ntyp + nat:]).split(), dtype=float)
    return np.transpose(values[:n1x * n2x * n3].reshape((n3, n2x, n1x)), (2, 1, 0))


def w0gauss_mv(x):
    arg = np.minimum(200.0, (x - 1.0 / np.sqrt(2.0)) ** 2)
    return np.exp(-arg) * (2.0 - np.sqrt(2.0) * x) / np.sqrt(np.pi)


# -- pw.x's eigenvalues, weights and Fermi level, from its own XML ----------
root = ET.parse(f"{RUN}/out/al.save/data-file-schema.xml").getroot()
ks = root.findall(".//ks_energies")
eig_qe = np.array([[float(v) for v in k.find("eigenvalues").text.split()] for k in ks]) * HA
wk_qe = np.array([float(k.find("k_point").attrib["weight"]) for k in ks])
xk_qe = np.array([[float(v) for v in k.find("k_point").text.split()] for k in ks])
ef = float(root.find(".//fermi_energy").text) * HA
degauss = 0.05
nk, nb = eig_qe.shape
print(f"pw.x: nk {nk} nbnd {nb} sum wk {wk_qe.sum():.6f} E_F {ef:.10f} Ry", flush=True)

P = np.array([[read(f"{RUN}/psi_K{k + 1:03d}_B{b + 1:03d}") for b in range(nb)]
              for k in range(nk)])
print("per-state dumps", P.shape, "norms (mean, min, max of int |psi|^2 / omega * omega / N)",
      flush=True)
ppx = read(f"{RUN}/al.stm.b20")


def stm_f90_mask(eig, order, stale=True):
    """Which states stm.f90 sums, its first_band/last_band rule transcribed."""
    down1, up1 = ef - 3 * degauss, ef + 3 * degauss
    mask = np.zeros_like(eig)
    first = last = None
    n_stale = 0
    for k in order:
        set_first = False
        for b in range(eig.shape[1]):
            if eig[k, b] < down1:
                first, set_first = b + 1, True
            if eig[k, b] < up1:
                last = b
        if not stale and not set_first:
            first = 0
        if not set_first:
            n_stale += 1
        f = 0 if first is None else first
        mask[k, f:last + 1] = 1.0
    return mask, n_stale


def window_mask(eig):
    return ((eig >= ef - 3 * degauss) & (eig <= ef + 3 * degauss)).astype(float)


wg = wk_qe[:, None] * w0gauss_mv((ef - eig_qe) / degauss)
mask_stale, n_stale = stm_f90_mask(eig_qe, range(nk), stale=True)
mask_fresh, _ = stm_f90_mask(eig_qe, range(nk), stale=False)
mask_win = window_mask(eig_qe)
print(f"stale k-points {n_stale}; states stale vs fresh differ "
      f"{int(np.abs(mask_stale - mask_fresh).sum())}; fresh vs window "
      f"{int(np.abs(mask_fresh - mask_win).sum())}", flush=True)
for name, mask in (("stale", mask_stale), ("fresh", mask_fresh), ("window", mask_win)):
    image = np.einsum("kb,kbxyz->xyz", wg * mask, P)
    print(f"pp.x states, {name} mask: max |image - pp.x STM| = "
          f"{np.abs(image - ppx).max():.3e} (peak {ppx.max():.5e})", flush=True)
for k in np.nonzero(np.abs(mask_stale - mask_fresh).sum(1))[0]:
    b = np.nonzero(mask_stale[k] != mask_fresh[k])[0]
    print(f"  stale drops k {k + 1} {xk_qe[k]} band(s) {b + 1} e-E_F {eig_qe[k, b] - ef} Ry "
          f"weight {wg[k, b]}", flush=True)

# -- defumat's states --------------------------------------------------------
from defumat import Calculator  # noqa: E402

calc = Calculator.from_file(REPO + "/tests/data/qe/al-metal-nosym.in",
                            pseudo_dir=REPO + "/tests/data/pseudo")
scf = calc.get_scf()
cal = calc.calculation
eig = np.asarray(scf.eigenvalues_by_spin)[0]
coords = np.asarray(cal.system.kpoints.coords)
order = [int(np.argmin(np.linalg.norm(coords - k, axis=1))) for k in xk_qe]
worst = max(float(np.linalg.norm(coords[o] - k)) for o, k in zip(order, xk_qe))
print(f"defumat: E_F {float(scf.fermi_energy):.10f} Ry (pw.x {ef:.10f}); k match "
      f"worst {worst:.1e}, identity order {order == list(range(nk))}", flush=True)
eig = eig[order]
print(f"eigenvalues: max |defumat - pw.x| = {np.abs(eig - eig_qe).max():.3e} Ry", flush=True)
wfc = scf.wavefunctions
wk = np.asarray(cal.system.kpoints.weights)
print(f"defumat sum wk {wk.sum():.6f}", flush=True)

D = np.zeros_like(P)
for i, k in enumerate(order):
    for b in range(nb):
        w = np.zeros((1, nk, nb))
        w[0, k, b] = 1.0
        D[i, b] = np.asarray(cal.density(wfc, w))[0]

# Compare multiplet by multiplet: states of equal energy (1e-6 Ry) summed.
worst = []
for k in range(nk):
    b = 0
    while b < nb:
        e = b + 1
        while e < nb and abs(eig_qe[k, e] - eig_qe[k, b]) < 1e-6:
            e += 1
        diff = np.abs(D[k, b:e].sum(0) - P[k, b:e].sum(0)).max()
        cut = e == nb  # the multiplet may continue past nbnd
        worst.append((diff, k + 1, b + 1, e, cut, eig_qe[k, b] - ef))
        b = e
worst.sort(reverse=True)
print("largest per-multiplet density differences (diff, k, first band, last band, "
      "touches nbnd, e - E_F):", flush=True)
for row in worst[:12]:
    print("  ", row, flush=True)

wgd = wk[order][:, None] * w0gauss_mv((ef - eig) / degauss)
for name, mask in (("defumat window", window_mask(eig)), ("stm.f90 stale", mask_stale)):
    image = np.einsum("kb,kbxyz->xyz", wgd * mask, D)
    print(f"defumat states, {name} mask: max |image - pp.x STM| = "
          f"{np.abs(image - ppx).max():.3e}", flush=True)
image = calc.get_stm(height=0.0, width=0.05, smearing="marzari-vanderbilt", band_cutoff=3.0)
ours = np.asarray(image.density) * image.width
print(f"get_stm vs pp.x STM {np.abs(ours - ppx).max():.3e}; vs pp.x states with the "
      f"window mask {np.abs(ours - np.einsum('kb,kbxyz->xyz', wg * mask_win, P)).max():.3e}",
      flush=True)
print("STS-STATES-DONE")
