"""Parser for ``pw.x`` output files, used as the reference for every check.

The project's testing method is to run the same input through Quantum ESPRESSO
and through pypresso and compare numbers, so this parser is the backbone of the
test suite: it turns a QE output -- in practice one of the committed
``test-suite/*/benchmark.out.git.inp=*.in`` files -- into a
:class:`QEReference` of plain NumPy arrays.

Deliberately NumPy and not JAX: reference values are host-side constants that a
test compares against, never something a gradient flows through.

Units follow QE's printout exactly, and each field says which it is: energies in
Ry, eigenvalues and Fermi level in **eV** (that is how pw.x prints them),
positions and forces in atomic units, stress in Ry/bohr^3.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["QEReference", "read_qe_output"]

_FLOAT = r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?"


def _floats(text: str) -> list[float]:
    """Every float in ``text``, tolerating Fortran ``D`` exponents."""
    return [float(m.replace("D", "E").replace("d", "e")) for m in re.findall(_FLOAT, text)]


def _scalar(text: str, pattern: str) -> float | None:
    """First float captured by ``pattern``, or None if the pattern is absent."""
    m = re.search(pattern, text)
    return None if m is None else _floats(m.group(1))[0]


@dataclass(frozen=True)
class QEReference:
    """Everything a pypresso run can be checked against, from one QE output."""

    path: Path
    calculation: str | None = None  # 'scf' | 'bands' | 'nscf', inferred from banners

    # --- cell and system ---
    ibrav: int | None = None
    alat: float | None = None  # bohr
    volume: float | None = None  # bohr^3
    nat: int | None = None
    ntyp: int | None = None
    nelec: float | None = None
    nbnd: int | None = None
    ecutwfc: float | None = None  # Ry
    ecutrho: float | None = None  # Ry
    xc: str | None = None
    at: np.ndarray | None = None  # (3,3) rows a_i, units of alat
    bg: np.ndarray | None = None  # (3,3) rows b_i, units of 2*pi/alat

    # --- basis ---
    ngm_dense: int | None = None
    fft_dense: tuple[int, int, int] | None = None
    ngm_smooth: int | None = None
    fft_smooth: tuple[int, int, int] | None = None
    npw: np.ndarray | None = None  # (nk,) plane waves per k-point

    # --- k-points (cartesian, units 2*pi/alat) ---
    kpoints: np.ndarray | None = None  # (nk,3)
    weights: np.ndarray | None = None  # (nk,)

    # --- results ---
    total_energy: float | None = None  # Ry
    energy_terms: dict[str, float] = field(default_factory=dict)  # Ry
    eigenvalues: np.ndarray | None = None  # (nspin, nk, nbnd) in eV
    occupations: np.ndarray | None = None  # (nspin, nk, nbnd), if printed
    fermi_energy: float | None = None  # eV
    homo: float | None = None  # eV, insulators
    lumo: float | None = None  # eV
    forces: np.ndarray | None = None  # (nat,3) Ry/bohr
    stress: np.ndarray | None = None  # (3,3) Ry/bohr^3
    pressure: float | None = None  # kbar
    n_iterations: int | None = None

    @property
    def nk(self) -> int:
        if self.kpoints is not None:
            return len(self.kpoints)
        if self.eigenvalues is not None:
            return self.eigenvalues.shape[1]
        raise ValueError(f"{self.path}: no k-point information in this output")

    @property
    def nspin(self) -> int:
        return 1 if self.eigenvalues is None else self.eigenvalues.shape[0]

    @classmethod
    def from_file(cls, path: str | Path) -> "QEReference":
        return read_qe_output(path)


def read_qe_output(path: str | Path) -> QEReference:
    """Parse a ``pw.x`` output file into a :class:`QEReference`.

    Missing quantities stay ``None`` rather than raising: a ``bands`` run has no
    total energy, an insulator has no Fermi level, most runs have no stress.
    Only a genuinely malformed block is an error.
    """
    path = Path(path)
    text = path.read_text(errors="replace")

    kpoints, weights = _parse_kpoints(text)
    eigenvalues, occupations, npw = _parse_bands(text)
    ngm_dense, fft_dense = _parse_grid(text, "Dense")
    ngm_smooth, fft_smooth = _parse_grid(text, "Smooth")
    stress, pressure = _parse_stress(text)

    xc = re.search(r"Exchange-correlation\s*=\s*(.+?)\s*\(", text)

    return QEReference(
        path=path,
        calculation=_parse_calculation(text),
        ibrav=_as_int(_scalar(text, r"bravais-lattice index\s*=\s*(" + _FLOAT + ")")),
        alat=_scalar(text, r"lattice parameter \(alat\)\s*=\s*(" + _FLOAT + ")"),
        volume=_scalar(text, r"unit-cell volume\s*=\s*(" + _FLOAT + ")"),
        nat=_as_int(_scalar(text, r"number of atoms/cell\s*=\s*(" + _FLOAT + ")")),
        ntyp=_as_int(_scalar(text, r"number of atomic types\s*=\s*(" + _FLOAT + ")")),
        nelec=_scalar(text, r"number of electrons\s*=\s*(" + _FLOAT + ")"),
        nbnd=_as_int(_scalar(text, r"number of Kohn-Sham states\s*=\s*(" + _FLOAT + ")")),
        ecutwfc=_scalar(text, r"kinetic-energy cutoff\s*=\s*(" + _FLOAT + ")"),
        ecutrho=_scalar(text, r"charge density cutoff\s*=\s*(" + _FLOAT + ")"),
        xc=xc.group(1).strip() if xc else None,
        at=_parse_axes(text, "crystal axes", "a"),
        bg=_parse_axes(text, "reciprocal axes", "b"),
        ngm_dense=ngm_dense,
        fft_dense=fft_dense,
        ngm_smooth=ngm_smooth,
        fft_smooth=fft_smooth,
        npw=npw,
        kpoints=kpoints,
        weights=weights,
        total_energy=_scalar(text, r"!\s+total energy\s*=\s*(" + _FLOAT + ")"),
        energy_terms=_parse_energy_terms(text),
        eigenvalues=eigenvalues,
        occupations=occupations,
        fermi_energy=_scalar(text, r"the Fermi energy is\s*(" + _FLOAT + ")"),
        homo=_parse_homo(text),
        lumo=_parse_lumo(text),
        forces=_parse_forces(text),
        stress=stress,
        pressure=pressure,
        n_iterations=_as_int(
            _scalar(text, r"convergence has been achieved in\s*(" + _FLOAT + r")\s*iterations")
        ),
    )


def _as_int(value: float | None) -> int | None:
    return None if value is None else int(round(value))


def _parse_calculation(text: str) -> str | None:
    if "Self-consistent Calculation" in text:
        return "scf"
    if "Band Structure Calculation" in text:
        # pw.x prints the same banner for 'bands' and 'nscf'; the difference is
        # that 'nscf' recomputes occupations and reports a Fermi energy or HOMO.
        return "nscf" if "End of band structure calculation" in text and (
            "the Fermi energy is" in text or "highest occupied" in text
        ) else "bands"
    return None


def _parse_axes(text: str, header: str, label: str) -> np.ndarray | None:
    """Parse the ``a(1) = ( x y z )`` block following ``header``."""
    block = re.search(header + r".*?\n((?:\s*" + label + r"\(\d\)\s*=.*\n)+)", text)
    if block is None:
        return None
    rows = [_floats(line.split("=", 1)[1])[:3] for line in block.group(1).strip().splitlines()]
    return np.array(rows, dtype=float)


def _parse_grid(text: str, which: str) -> tuple[int | None, tuple[int, int, int] | None]:
    """``Dense  grid:  1459 G-vectors   FFT dimensions: (  15,  15,  15)``."""
    m = re.search(
        which + r"\s+grid:\s*(\d+)\s*G-vectors\s*FFT dimensions:\s*\(\s*(\d+),\s*(\d+),\s*(\d+)\)",
        text,
    )
    if m is None:
        return None, None
    return int(m.group(1)), (int(m.group(2)), int(m.group(3)), int(m.group(4)))


def _parse_kpoints(text: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Parse the header k-point list.

    Absent when QE decides there are too many to print ("Number of k-points >=
    100: set verbosity='high' to print them"), which is not an error here.
    """
    rows = re.findall(
        r"k\(\s*\d+\)\s*=\s*\(\s*(" + _FLOAT + r")\s+(" + _FLOAT + r")\s+(" + _FLOAT
        + r")\s*\),\s*wk\s*=\s*(" + _FLOAT + r")",
        text,
    )
    if not rows:
        return None, None
    data = np.array([[float(v) for v in row] for row in rows], dtype=float)
    # QE prints the list twice when it also reports crystal coordinates; the
    # first block is the cartesian one in units of 2*pi/alat, which is what the
    # rest of the code works in.
    nk = _as_int(_scalar(text, r"number of k points\s*=?\s*(" + _FLOAT + ")"))
    if nk is not None and len(data) > nk:
        data = data[:nk]
    return data[:, :3], data[:, 3]


_KBLOCK = re.compile(
    r"^\s*k\s*=\s*(" + _FLOAT + r")\s*(" + _FLOAT + r")\s*(" + _FLOAT
    + r")\s*\(\s*(\d+)\s*PWs\)\s*bands \(ev\):\s*$"
)


def _parse_bands(
    text: str,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Parse the eigenvalue blocks printed after the final SCF/band step.

    Returns ``(eigenvalues, occupations, npw)`` with eigenvalues shaped
    ``(nspin, nk, nbnd)`` in eV. Only the last printed set is kept -- with
    ``verbosity='high'`` QE prints one set per SCF iteration and only the
    converged one is a meaningful reference.
    """
    starts = [m.end() for m in re.finditer(r"End of (?:self-consistent|band structure) calculation", text)]
    tail = text[starts[-1] :] if starts else text

    spin_up = tail.find("------ SPIN UP")
    spin_dw = tail.find("------ SPIN DOWN")
    if spin_up != -1 and spin_dw != -1:
        segments = [tail[spin_up:spin_dw], tail[spin_dw:]]
    else:
        segments = [tail]

    eig_spin, occ_spin, npw_list = [], [], []
    for seg in segments:
        eigs, occs, npws = _parse_band_segment(seg)
        if not eigs:
            continue
        eig_spin.append(eigs)
        occ_spin.append(occs)
        npw_list = npws  # identical across spin channels

    if not eig_spin:
        return None, None, None

    nbnd = min(len(e) for spin in eig_spin for e in spin)
    eigenvalues = np.array([[e[:nbnd] for e in spin] for spin in eig_spin], dtype=float)
    occupations = (
        np.array([[o[:nbnd] for o in spin] for spin in occ_spin], dtype=float)
        if all(all(len(o) >= nbnd for o in spin) for spin in occ_spin)
        else None
    )
    return eigenvalues, occupations, np.array(npw_list, dtype=int)


def _parse_band_segment(segment: str) -> tuple[list, list, list]:
    """Walk one spin channel line by line, collecting per-k numeric blocks."""
    eigs: list[list[float]] = []
    occs: list[list[float]] = []
    npws: list[int] = []

    lines = segment.splitlines()
    i = 0
    while i < len(lines):
        m = _KBLOCK.match(lines[i])
        if m is None:
            i += 1
            continue
        npws.append(int(m.group(4)))
        values, i = _collect_numeric_block(lines, i + 1)
        eigs.append(values)
        # An "occupation numbers" block may follow the eigenvalues.
        j = i
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines) and "occupation numbers" in lines[j]:
            occ, i = _collect_numeric_block(lines, j + 1)
            occs.append(occ)
    return eigs, occs, npws


def _collect_numeric_block(lines: list[str], start: int) -> tuple[list[float], int]:
    """Consume lines of bare floats from ``start``, skipping leading blanks."""
    values: list[float] = []
    i = start
    while i < len(lines) and not lines[i].strip():
        i += 1
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped or re.search(r"[A-Za-z]", stripped.replace("E", "").replace("e", "")):
            break
        values.extend(_floats(stripped))
        i += 1
    return values, i


_ENERGY_TERMS = {
    "one-electron": r"one-electron contribution\s*=\s*(" + _FLOAT + ")",
    "hartree": r"hartree contribution\s*=\s*(" + _FLOAT + ")",
    "xc": r"xc contribution\s*=\s*(" + _FLOAT + ")",
    "ewald": r"ewald contribution\s*=\s*(" + _FLOAT + ")",
    "smearing": r"smearing contrib\. \(-TS\)\s*=\s*(" + _FLOAT + ")",
    "hubbard": r"Hubbard energy\s*=\s*(" + _FLOAT + ")",
    "one_center_paw": r"one-center paw contrib\.\s*=\s*(" + _FLOAT + ")",
    "dispersion": r"Dispersion Correction\s*=\s*(" + _FLOAT + ")",
    "internal": r"internal energy E=F\+TS\s*=\s*(" + _FLOAT + ")",
}


def _parse_energy_terms(text: str) -> dict[str, float]:
    """The decomposition QE prints under "the sum of the following terms"."""
    terms = {}
    for name, pattern in _ENERGY_TERMS.items():
        value = _scalar(text, pattern)
        if value is not None:
            terms[name] = value
    return terms


def _parse_homo(text: str) -> float | None:
    value = _scalar(text, r"highest occupied level \(ev\):\s*(" + _FLOAT + ")")
    if value is not None:
        return value
    m = re.search(
        r"highest occupied, lowest unoccupied level \(ev\):\s*(" + _FLOAT + r")\s+(" + _FLOAT + ")",
        text,
    )
    return float(m.group(1)) if m else None


def _parse_lumo(text: str) -> float | None:
    m = re.search(
        r"highest occupied, lowest unoccupied level \(ev\):\s*(" + _FLOAT + r")\s+(" + _FLOAT + ")",
        text,
    )
    return float(m.group(2)) if m else None


def _parse_forces(text: str) -> np.ndarray | None:
    """``atom    1 type  1   force =  fx  fy  fz`` in Ry/bohr."""
    block = re.search(r"Forces acting on atoms.*?\n((?:.*force\s*=.*\n)+)", text)
    if block is None:
        return None
    rows = [_floats(line.split("force", 1)[1])[:3] for line in block.group(1).strip().splitlines()]
    return np.array(rows, dtype=float)


def _parse_stress(text: str) -> tuple[np.ndarray | None, float | None]:
    """The 3x3 stress in Ry/bohr^3 and the pressure in kbar.

    QE prints each row as three Ry/bohr^3 values followed by the same row in
    kbar; only the first three are kept.
    """
    m = re.search(
        r"total\s+stress\s*\(Ry/bohr\*\*3\).*?P=\s*(" + _FLOAT + r")\s*\n((?:.*\n){3})", text
    )
    if m is None:
        return None, None
    rows = [_floats(line)[:3] for line in m.group(2).strip().splitlines()]
    return np.array(rows, dtype=float), float(m.group(1))
