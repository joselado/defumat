"""Elk's task 481 on any current: ``eps(w)`` from ``J(t)`` after a step in ``A``.

A literal transcription of Elk 11.0.2's post-processing, so that a current
computed here and the current Elk writes to ``JTOT_TD.OUT`` can be turned into
a dielectric function by the same arithmetic:

* ``dielectric_tdrt.f90`` (``task = 481``): ``E(w) = -(1/c) A(t=0)`` at every
  frequency, ``J(t) = jtot(t) / Omega``, the constant part removed when
  ``jtconst0`` is set, ``J(w)`` filtered by a Lorentzian of width ``swidth``,
  ``eps = 1 + 4 pi i (J/E) / (w + i swidth)``, and ``eps`` filtered again with
  ``2 swidth``;
* ``wsplint.f90`` and ``splint.f90``: the integration weights on the time grid,
  a cubic fitted piecewise to four neighbouring points, nine points at a time;
* ``zftft.f90``: ``f(w) = int f(t) exp(i w t) dt`` with those weights;
* ``zlrzncnv.f90``: the Lorentzian convolution, a rectangle rule over the
  frequency grid itself, which covers ``w >= 0`` only, so the lowest
  frequencies of every filtered array lose roughly half of their window;
* ``readjtot.f90``: the current of the last time step is copied from the one
  before it.

Units are Elk's: Hartree atomic units, ``jtot`` the current summed over the
cell (``Omega`` times the density). The functions take arrays and read nothing
but what they are handed; :func:`read_jtot` and :func:`read_epsilon` read Elk's
files.

    python3 tools/realtime/elk_tdrt.py <elk run directory>   # rebuilds EPSILON_TDRT_11 from JTOT_TD
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

#: Elk's speed of light, ``solsc`` in ``modmain.f90``.
ELK_SOLSC = 137.035999084


def splint(x, f) -> float:
    """``splint.f90``: the integral of ``f`` over ``x``, piecewise cubics through four points."""
    x = np.asarray(x, dtype=float)
    f = np.asarray(f, dtype=float)
    n = len(x)
    if n <= 4:
        coefficients = np.polyfit(x - x[0], f, n - 1)
        return float(np.polyval(np.polyint(coefficients), x[-1] - x[0]))
    x0 = x[0]
    x1, x2, x3 = x[1] - x0, x[2] - x0, x[3] - x0
    t4, t5, t6 = x1 - x2, x1 - x3, x2 - x3
    y0 = f[0]
    y1, y2, y3 = f[1] - y0, f[2] - y0, f[3] - y0
    t1 = x1 * x2 * y3
    t2 = x2 * x3 * y1
    t3 = x1 * x3
    t0 = 0.5 / (t3 * t4 * t5 * t6)
    t3 = t3 * y2
    t7 = t1 * t4 + t2 * t6 - t3 * t5
    t4, t5, t6 = x1**2, x2**2, x3**2
    y1 = t3 * t6 - t1 * t5
    y3 = t2 * t5 - t3 * t4
    y2 = t1 * t4 - t2 * t6
    t1 = x1 * y1 + x2 * y2 + x3 * y3
    t2 = y1 + y2 + y3
    total = x2 * (y0 + t0 * (t1 + x2 * (0.5 * t7 * x2 - 0.6666666666666666667 * t2)))
    for i in range(2, n - 3):  # Fortran i = 3 .. n-3
        x0 = x[i]
        x1, x2, x3 = x[i - 1] - x0, x[i + 1] - x0, x[i + 2] - x0
        t4, t5, t6 = x1 - x2, x1 - x3, x2 - x3
        t3 = x1 * x3
        y0 = f[i]
        y1, y2, y3 = f[i - 1] - y0, f[i + 1] - y0, f[i + 2] - y0
        t1 = x1 * x2 * y3
        t2 = x2 * x3 * y1
        t0 = 0.5 / (t3 * t4 * t5 * t6)
        t3 = t3 * y2
        t7 = t1 * t4 + t2 * t6 - t3 * t5
        t4, t5, t6 = x1**2, x2**2, x3**2
        y1 = t3 * t6 - t1 * t5
        y2 = t1 * t4 - t2 * t6
        y3 = t2 * t5 - t3 * t4
        t1 = x1 * y1 + x2 * y2 + x3 * y3
        t2 = y1 + y2 + y3
        total += x2 * (y0 + t0 * (t1 + x2 * (0.5 * t7 * x2 - 0.6666666666666666667 * t2)))
    x0 = x[n - 3]
    x1, x2, x3 = x[n - 4] - x0, x[n - 2] - x0, x[n - 1] - x0
    t4, t5, t6 = x1 - x2, x1 - x3, x2 - x3
    y0 = f[n - 3]
    y1, y2, y3 = f[n - 4] - y0, f[n - 2] - y0, f[n - 1] - y0
    t1 = x1 * x2
    t2 = x2 * x3 * y1
    t3 = x1 * x3 * y2
    t0 = 0.5 / (t1 * t4 * t5 * t6)
    t1 = t1 * y3
    t7 = t1 * t4 + t2 * t6 - t3 * t5
    t4, t5, t6 = x1**2, x2**2, x3**2
    y1 = t3 * t6 - t1 * t5
    y2 = t1 * t4 - t2 * t6
    y3 = t2 * t5 - t3 * t4
    t1 = x1 * y1 + x2 * y2 + x3 * y3
    t2 = y1 + y2 + y3
    total += x3 * (y0 + t0 * (t1 + x3 * (0.5 * t7 * x3 - 0.6666666666666666667 * t2)))
    return float(total)


def wsplint(x) -> np.ndarray:
    """``wsplint.f90``: weights ``w`` with ``sum w f = splint(x, f)``, nine points at a time."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    w = np.zeros(n)
    if n <= 9:
        for i in range(n):
            f = np.zeros(n)
            f[i] = 1.0
            w[i] = splint(x, f)
        return w
    for i in range(4):
        f = np.zeros(9)
        f[i] = 1.0
        w[i] = splint(x[:9], f)
    f = np.zeros(9)
    f[4] = 1.0
    for i in range(4, n - 4):
        w[i] = splint(x[i - 4:i + 5], f)
    for i in range(4):
        f = np.zeros(9)
        f[i + 5] = 1.0
        w[n - 4 + i] = splint(x[n - 9:], f)
    return w


def zftft(w, weights, times, values) -> np.ndarray:
    """``zftft.f90``: ``sum_t weights f(t) exp(i w t)`` on the frequencies ``w``."""
    phase = np.outer(np.asarray(w, dtype=float), np.asarray(times, dtype=float))
    return (np.cos(phase) + 1j * np.sin(phase)) @ (np.asarray(weights) * np.asarray(values))


def zlrzncnv(width: float, w, values) -> np.ndarray:
    """``zlrzncnv.f90``: Lorentzian convolution by a rectangle rule on ``w`` itself."""
    w = np.asarray(w, dtype=float)
    values = np.asarray(values)
    dw = np.diff(w)  # (n-1,)
    kernel = dw[None, :] / ((w[None, :-1] - w[:, None]) ** 2 + width**2)  # (n, n-1)
    return (width / math.pi) * (kernel @ values[:-1])


def frequency_grid(nwplot: int = 400, wplot=(0.0, 0.5)) -> np.ndarray:
    """``dielectric_tdrt.f90``'s grid: ``w1 + (w2 - w1)/nwplot (iw - 1)``, ``iw = 1 .. nwplot``."""
    w1 = max(float(wplot[0]), 0.0)
    w2 = max(float(wplot[1]), w1)
    return w1 + (w2 - w1) / nwplot * np.arange(nwplot)


def epsilon_481(times, jtot, a0, volume: float, *, swidth: float = 0.001,
                nwplot: int = 400, wplot=(0.0, 0.5), jtconst0: bool = True,
                solsc: float = ELK_SOLSC):
    """``(w, eps)`` as Elk's ``dielectric_tdrt`` computes them for ``task = 481``.

    Args:
        times: ``(ntimes,)``, Elk's grid ``(its - 1) dtimes``, Hartree atomic units.
        jtot: ``(ntimes, 3)``, the current summed over the cell; a current
            density times ``volume``. Only the first ``ntimes - 1`` rows are
            used and the last is a copy of the one before, as ``readjtot`` does.
        a0: ``(3,)``, the vector potential at ``t = 0`` in Elk's Hartree units
            (``kappa0 * c``).
        volume: the cell volume in bohr^3.

    Returns:
        ``w`` ``(nwplot,)`` and ``eps`` ``(nwplot, 3, 3)``; a column ``j`` whose
        field component vanishes is left at the identity, as Elk leaves it.
    """
    times = np.asarray(times, dtype=float)
    jt = np.array(jtot, dtype=float, copy=True) / float(volume)
    jt[-1] = jt[-2]
    w = frequency_grid(nwplot, wplot)
    weights = wsplint(times)
    ew = -np.asarray(a0, dtype=float) / solsc  # (3,)
    if jtconst0:
        jt = jt - (weights @ jt) / times[-1]
    jw = np.stack([zlrzncnv(swidth, w, zftft(w, weights, times, jt[:, i]))
                   for i in range(3)], axis=-1)  # (nw, 3)
    eps = np.zeros((len(w), 3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            z2 = ew[j]
            ratio = jw[:, i] / z2 if abs(z2) > 1e-8 else np.zeros(len(w))
            z1 = 4.0 * math.pi * 1j * ratio / (w + 1j * swidth)
            if i == j:
                z1 = z1 + 1.0
            eps[:, i, j] = zlrzncnv(2.0 * swidth, w, z1)
    return w, eps


def read_jtot(path) -> tuple[np.ndarray, np.ndarray]:
    """``(times, jtot)`` from an Elk ``JTOT_TD.OUT``: ``ntimes - 1`` rows of ``t, J_x, J_y, J_z``."""
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1:4]


def read_epsilon(path) -> tuple[np.ndarray, np.ndarray]:
    """``(w, eps)`` from an Elk ``EPSILON*_ij.OUT``: a block of ``Re`` and one of ``Im``."""
    blocks = [b for b in Path(path).read_text().split("\n \n") if b.strip()]
    if len(blocks) != 2:
        blocks = [b for b in Path(path).read_text().split("\n\n") if b.strip()]
    real = np.loadtxt(blocks[0].splitlines())
    imag = np.loadtxt(blocks[1].splitlines())
    return real[:, 0], real[:, 1] + 1j * imag[:, 1]


def _main(directory: Path) -> None:
    """Rebuild ``EPSILON_TDRT_11.OUT`` from the same run's ``JTOT_TD.OUT`` and compare."""
    import re

    text = (directory / "elk.in").read_text()
    times_elk, jtot = read_jtot(directory / "JTOT_TD.OUT")
    dt = float(times_elk[1] - times_elk[0])
    tstime = float(re.search(r"^tstime\s*\n\s*([0-9.eEdD+-]+)", text, re.M).group(1))
    ntimes = int(round(tstime / dt)) + 1
    times = dt * np.arange(ntimes)
    full = np.vstack([jtot, jtot[-1:]])
    # the vector potential at t = 0 as Elk tabulated it (task 450)
    a0 = np.loadtxt(directory / "AFIELDT.OUT", skiprows=1)[0, 2:5]
    info = (directory / "INFO.OUT").read_text()
    volume = float(re.search(r"Unit cell volume\s*:\s*([0-9.eE+-]+)", info).group(1))
    swidth = 0.001
    match = re.search(r"^swidth\s*\n\s*([0-9.eEdD+-]+)", text, re.M)
    if match:
        swidth = float(match.group(1))
    w, eps = epsilon_481(times, full, a0, volume, swidth=swidth,
                         jtconst0=bool(re.search(r"^jtconst0\s*\n\s*\.true\.", text, re.M)))
    w_elk, eps_elk = read_epsilon(directory / "EPSILON_TDRT_11.OUT")
    scale = np.max(np.abs(eps_elk))
    print(f"volume {volume:.6f}  swidth {swidth}  a0 {a0}")
    print(f"max |eps_here - eps_Elk| = {np.max(np.abs(eps[:, 0, 0] - eps_elk)):.3e} "
          f"on a scale of {scale:.3e} (Elk prints 10 significant digits)")


if __name__ == "__main__":
    _main(Path(sys.argv[1]))
