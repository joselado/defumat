"""Writing results out in the formats QE's post-processing tools produce.

So far: the ``.dos`` file ``PP/src/dos.f90`` writes and the ``filpdos``
files ``PP/src/partialdos.f90`` writes. Everything here converts
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

__all__ = [
    "write_dos",
    "format_dos",
    "fortran_exponential",
    "write_pdos",
    "format_pdos_shell",
    "format_pdos_total",
    "pdos_file_name",
]


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

    With two spin channels the format changes in exactly the way ``dos.f90``
    changes it: the header becomes ``dosup(E)  dosdw(E)  Int dos(E)``, each line
    is ``(f8.3, 3e12.4)``, and the **one** integrated column is the sum over both
    channels. There is one integrated column and two differential ones because
    the sum rule is a statement about the total number of electrons; which
    channel they are in is what the two ``dos`` columns say. A run with the
    magnetization constrained prints both Fermi levels, ``2f8.3``, as
    ``EFermi = <up> <down> eV``.
    """
    if dos.fermi_energy_up is not None:
        fermi = (
            f" EFermi = {dos.fermi_energy_up * RY_TO_EV:8.3f}"
            f"{dos.fermi_energy_down * RY_TO_EV:8.3f} eV"
        )
    elif dos.fermi_energy is not None:
        fermi = f" EFermi = {dos.fermi_energy * RY_TO_EV:8.3f} eV"
    else:
        fermi = ""

    if dos.nspin == 1:
        lines = [f"#  E (eV)   dos(E)     Int dos(E){fermi}"]
        columns = [dos.dos_ev]
    else:
        lines = [f"#  E (eV)   dosup(E)     dosdw(E)   Int dos(E){fermi}"]
        columns = list(dos.dos_ev)

    integrated = dos.total_integrated
    for index, energy in enumerate(dos.energies_ev):
        line = f"{energy:8.3f}"
        for column in columns:
            line += fortran_exponential(float(column[index]))
        lines.append(line + fortran_exponential(float(integrated[index])))
    return "\n".join(lines) + "\n"


def write_dos(path: str | Path, dos) -> Path:
    """Write a ``.dos`` file and return where it went."""
    path = Path(path)
    path.write_text(format_dos(dos))
    return path


# --------------------------------------------------------------------------
# projwfc.x's filpdos files
# --------------------------------------------------------------------------


def pdos_file_name(prefix: str, channel) -> str:
    """``partialdos``'s file name for one shell: ``<filpdos>.pdos_atm#1(Si)_wfc#2(p)``.

    Built from the same fields the Fortran builds it from -- the atom's index,
    its ``ATOMIC_SPECIES`` label, the orbital's index *in the pseudopotential
    file*, and the shell's letter -- so the files can be diffed against
    ``projwfc.x``'s by name as well as by content.
    """
    from pypresso.projwfc.channels import L_LABELS

    return (
        f"{prefix}.pdos_atm#{channel.atom + 1}({channel.species})"
        f"_wfc#{channel.wfc}({L_LABELS[channel.l]})"
    )


def format_pdos_shell(pdos, shell) -> str:
    """One shell's ``filpdos`` file: ``ldos`` and then one ``pdos`` per ``m``.

    ``partialdos``'s layout: ``(f8.3, Ne11.3)`` with the energy in eV and every
    density in states/eV, the shell's own sum first and the ``m`` resolution
    after it, spin channels interleaved within each column group.
    """
    channels = list(shell)
    atom, wfc = channels[0].atom, channels[0].wfc
    nspin = pdos.nspin

    header = "#" + (" E (eV)   ldos(E)  " if nspin == 1
                    else " E (eV)  ldosup(E)  ldosdw(E)")
    header += "".join(
        " pdos(E)   " if nspin == 1 else " pdosup(E)  pdosdw(E) "
        for _ in channels
    )

    ldos = np.atleast_2d(pdos.select(atom=atom, wfc=wfc)) / RY_TO_EV
    columns = [ldos[spin] for spin in range(nspin)]
    for channel in channels:
        weight = np.atleast_2d(pdos.pdos_by_spin[:, channel.index]) / RY_TO_EV
        columns += [weight[spin] for spin in range(nspin)]

    lines = [header]
    for index, energy in enumerate(pdos.energies_ev):
        lines.append(
            f"{energy:8.3f}"
            + "".join(fortran_exponential(float(c[index]), 11, 3) for c in columns)
        )
    return "\n".join(lines) + "\n"


def format_pdos_total(pdos) -> str:
    """``<filpdos>.pdos_tot``: the plain DOS and the sum of every channel."""
    nspin = pdos.nspin
    header = "#" + (
        " E (eV)  dos(E)    pdos(E)" if nspin == 1
        else " E (eV)  dosup(E)   dosdw(E)  pdosup(E)  pdosdw(E)"
    )
    dos = np.atleast_2d(pdos.total.dos) / RY_TO_EV
    summed = np.atleast_2d(pdos.summed) / RY_TO_EV
    columns = [dos[s] for s in range(nspin)] + [summed[s] for s in range(nspin)]

    lines = [header]
    for index, energy in enumerate(pdos.energies_ev):
        lines.append(
            f"{energy:8.3f}"
            + "".join(fortran_exponential(float(c[index]), 11, 3) for c in columns)
        )
    return "\n".join(lines) + "\n"


def write_pdos(prefix: str | Path, pdos) -> list[Path]:
    """Write ``projwfc.x``'s whole set of ``filpdos`` files; return the paths.

    One file per ``(atom, shell)`` plus the ``.pdos_tot`` summary, named exactly
    as ``partialdos`` names them.
    """
    prefix = Path(prefix)
    written = []
    for shell in pdos.shells():
        path = prefix.with_name(pdos_file_name(prefix.name, shell[0]))
        path.write_text(format_pdos_shell(pdos, shell))
        written.append(path)
    total = prefix.with_name(f"{prefix.name}.pdos_tot")
    total.write_text(format_pdos_total(pdos))
    return written + [total]
