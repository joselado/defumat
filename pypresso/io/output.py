"""Writing results out in the formats QE's post-processing tools produce.

So far: the ``.dos`` file ``PP/src/dos.f90`` writes. Everything here converts
out of Rydberg atomic units, which is this layer's privilege and nowhere else's.

The formatting is Fortran's, down to the exponent style, so that a file written
here can be diffed against one written by ``dos.x`` rather than merely plotted
next to it. Fortran's ``E`` edit descriptor normalises the mantissa into
``[0.1, 1)`` -- ``0.1234E+01`` where Python writes ``1.2340E+00`` -- and nothing
but an explicit reimplementation gets that.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pypresso.units import RY_TO_EV

__all__ = ["write_dos", "format_dos", "fortran_exponential"]


def fortran_exponential(value: float, width: int = 12, decimals: int = 4) -> str:
    """Fortran's ``Ew.d``: a mantissa in ``[0.1, 1)`` and a two-digit exponent."""
    if not np.isfinite(value):  # pragma: no cover - defensive
        return "NaN".rjust(width)
    if value == 0.0:
        mantissa, exponent = 0.0, 0
    else:
        exponent = int(np.floor(np.log10(abs(value)))) + 1
        mantissa = abs(value) / 10.0**exponent
        # Rounding the mantissa can carry it back up to 1.0; renormalise if so.
        if round(mantissa, decimals) >= 1.0:
            mantissa /= 10.0
            exponent += 1
    digits = f"{mantissa:.{decimals}f}"[2:]
    sign = "-" if value < 0.0 else ""
    exponent_sign = "+" if exponent >= 0 else "-"
    return f"{sign}0.{digits}E{exponent_sign}{abs(exponent):02d}".rjust(width)


def format_dos(dos) -> str:
    """A :class:`~pypresso.workflows.dos.DensityOfStates` as a ``.dos`` file.

    ``dos.f90``'s layout: a comment header carrying the Fermi level, then one
    line per energy with ``(f8.3, 2e12.4)`` -- energy in eV, ``D(E)`` in
    states/eV, and the integrated count.
    """
    fermi = "" if dos.fermi_energy is None else f" EFermi = {dos.fermi_energy * RY_TO_EV:8.3f} eV"
    lines = [f"#  E (eV)   dos(E)     Int dos(E){fermi}"]
    for energy, value, integrated in zip(dos.energies_ev, dos.dos_ev, dos.integrated):
        lines.append(
            f"{energy:8.3f}"
            + fortran_exponential(float(value))
            + fortran_exponential(float(integrated))
        )
    return "\n".join(lines) + "\n"


def write_dos(path: str | Path, dos) -> Path:
    """Write a ``.dos`` file and return where it went."""
    path = Path(path)
    path.write_text(format_dos(dos))
    return path
