"""Shared setup for the selenium spin-orbit real-time pair against Elk (D22 only).

The input is tests/data/qe/se-trigonal-soc.in with the cutoff and the mesh set,
nosym/noinv removed so that the field's little group (C3 for a field along c)
reduces the mesh as Elk's does, and optionally nbnd for the gap probe.
"""
import os
import re
from pathlib import Path

REPO = Path(os.environ.get("DEFUMAT_REPO", "/l/ladovj1/defumat-hspin"))
PSEUDO = REPO / "tests" / "data" / "pseudo"
HERE = Path(__file__).resolve().parent


def make_input(ecut, grid, nbnd=None, conv_thr=None, tag=""):
    text = (REPO / "tests/data/qe/se-trigonal-soc.in").read_text()
    text = re.sub(r"ecutwfc\s*=\s*[0-9.dD+-]+", f"ecutwfc = {ecut}", text)
    text = re.sub(r"^\s*nosym\s*=.*$", "", text, flags=re.M)
    if nbnd is not None:
        text = text.replace("&system", f"&system\n    nbnd = {nbnd},", 1)
    if conv_thr is not None:
        text = re.sub(r"conv_thr\s*=\s*[0-9.dDeE+-]+", f"conv_thr = {conv_thr}", text)
    g = " ".join(str(int(n)) for n in grid)
    text = re.sub(r"K_POINTS.*\n\s*[0-9 ]+\n", f"K_POINTS (automatic)\n {g} 0 0 0\n", text)
    path = HERE / f"se-soc-e{ecut:g}-k{''.join(str(n) for n in grid)}{tag}.in"
    path.write_text(text)
    return path


def pulse(intensity=1.0e10, photon_ev=1.55, fwhm_fs=1.0, peak_fs=None, direction=(0, 0, 1)):
    """Elk's Gaussian, along c: A0 exp(-(t-t0)^2/2s^2) sin(w(t-t0)), kappa0 = E0/w."""
    from defumat.realtime.pulse import Gaussian

    return Gaussian.from_intensity(intensity, photon_ev, fwhm_fs, peak_fs=peak_fs,
                                   polarization=tuple(float(x) for x in direction))
