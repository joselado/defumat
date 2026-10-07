"""Write an elk.in for trigonal selenium with spin-orbit coupling (no defumat import).

    python3 write_elk.py OUT TASKS N [--rgkmax 7] [--nempty 4] [--lorbcnd]
                         [--tstime T --dtimes DT --intensity I --photon EV --fwhm FS --peak FS]

The structure is tests/data/qe/se-trigonal-soc.in's (ibrav 4, a = 8.2509 bohr,
c/a = 1.1345, x = 0.2254), PBE (xctype 20), tshift off for the time evolution.
The pulse is Elk's Gaussian along c with A0 = c E0/w, E0 = sqrt(I/3.509e16 W/cm^2),
which is defumat's Gaussian.from_intensity with amplitude kappa0 = A0/c.
"""
import argparse
import math

HARTREE_SI = 4.3597447222071e-18
AU_SEC = 2.4188843265857e-17
BOHR_CM = 5.29177210903e-9
C_AU = 137.035999084
HARTREE_EV = 27.211386245988
INTENSITY_AU = (C_AU / (8.0 * math.pi)) * HARTREE_SI / (AU_SEC * BOHR_CM ** 2)
FS = 1.0e-15 / AU_SEC

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("tasks")
ap.add_argument("n", type=int)
ap.add_argument("--rgkmax", type=float, default=7.0)
ap.add_argument("--nempty", type=float, default=4.0)
ap.add_argument("--lorbcnd", action="store_true")
ap.add_argument("--lmaxapw", type=int, default=None)
ap.add_argument("--gmaxvr", type=float, default=None)
ap.add_argument("--tstime", type=float, default=None)
ap.add_argument("--dtimes", type=float, default=None)
ap.add_argument("--intensity", type=float, default=1.0e10)
ap.add_argument("--photon", type=float, default=1.55)
ap.add_argument("--fwhm", type=float, default=1.0)
ap.add_argument("--peak", type=float, default=None)
ap.add_argument("--phase", type=float, default=0.0)
ap.add_argument("--sppath", default="./")
ap.add_argument("--extra", default="")
a = ap.parse_args()

lines = ["tasks"] + [f"  {t}" for t in a.tasks.split(",")] + [""]
lines += ["spinorb", " .true.", "", "xctype", "  20", "", "tshift", " .false.", "",
          "rgkmax", f"  {a.rgkmax}", "", "nempty", f"  {a.nempty}", ""]
if a.lorbcnd:
    lines += ["lorbcnd", " .true.", ""]
if a.lmaxapw is not None:
    lines += ["lmaxapw", f"  {a.lmaxapw}", ""]
if a.gmaxvr is not None:
    lines += ["gmaxvr", f"  {a.gmaxvr}", ""]
lines += ["ngridk", f"  {a.n} {a.n} {a.n}", ""]
if a.tstime is not None:
    omega = a.photon / HARTREE_EV
    e0 = math.sqrt(a.intensity / INTENSITY_AU)
    a0 = C_AU * e0 / omega
    fwhm = a.fwhm * FS
    peak = 3.0 * fwhm if a.peak is None else a.peak * FS
    lines += ["tstime", f"  {a.tstime!r}", "", "dtimes", f"  {a.dtimes!r}", ""]
    lines += ["pulse", "  1", f"  0.0 0.0 {a0!r}   {omega!r}   {a.phase!r}   0.0   {peak!r}   {fwhm!r}", ""]
if a.extra:
    lines += a.extra.split("\\n") + [""]
lines += ["avec", "  1.0  0.0  0.0", " -0.5  0.86602540378443864676  0.0", "  0.0  0.0  1.1345", "",
          "scale", "  8.2509", "", "sppath", f"  '{a.sppath}'", "",
          "atoms", "  1                                 : nspecies",
          "  'Se.in'                           : spfname",
          "  3                                 : natoms; atposl below",
          "  0.2254   0.0000   0.333333333   0.0 0.0 0.0",
          "  0.0000   0.2254   0.666666667   0.0 0.0 0.0",
          " -0.2254  -0.2254   0.000000000   0.0 0.0 0.0", ""]
open(a.out, "w").write("\n".join(lines))
