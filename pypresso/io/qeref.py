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

from pypresso.units import ANGSTROM_TO_BOHR, RY_TO_KBAR

__all__ = [
    "QEReference",
    "ProjwfcReference",
    "read_qe_output",
    "read_projwfc_output",
    "read_pdos_file",
]

_FLOAT = r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?"


def _floats(text: str) -> list[float]:
    """Every float in ``text``, tolerating Fortran ``D`` exponents."""
    return [float(m.replace("D", "E").replace("d", "e")) for m in re.findall(_FLOAT, text)]


def _scalar(text: str, pattern: str) -> float | None:
    """First float captured by ``pattern``, or None if the pattern is absent."""
    m = re.search(pattern, text)
    return None if m is None else _floats(m.group(1))[0]


def _last_scalar(text: str, pattern: str) -> float | None:
    """The **last** float captured by ``pattern``.

    A relaxation prints its quantities once per ionic step, so ``_scalar`` reads
    the *starting* geometry's -- which is right for the SCF runs everything else
    here compares and wrong for the one number a relaxation exists to produce.
    """
    matches = re.findall(pattern, text)
    return None if not matches else _floats(matches[-1])[0]


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
    #: ``E = F + TS`` for a smeared run: the total with the entropy added back.
    #: Printed alongside the terms but not one of them.
    internal_energy: float | None = None  # Ry
    eigenvalues: np.ndarray | None = None  # (nspin, nk, nbnd) in eV
    occupations: np.ndarray | None = None  # (nspin, nk, nbnd), if printed
    fermi_energy: float | None = None  # eV
    #: The two independent Fermi levels of a run with ``tot_magnetization``
    #: constrained, in eV, or ``None``. pw.x prints them on one line as "the
    #: spin up/dw Fermi energies are" and prints no single Fermi energy at all.
    fermi_energy_up: float | None = None
    fermi_energy_down: float | None = None
    #: ``int (rho_up - rho_dw)`` and ``int |rho_up - rho_dw|`` in Bohr magnetons
    #: per cell, as printed at the end of an LSDA run (2 decimals).
    magnetization: float | None = None
    absolute_magnetization: float | None = None
    #: The three components a *noncollinear* run prints on the same line, in
    #: Bohr magnetons per cell. ``magnetization`` then holds the first of them,
    #: which is what the single-float pattern picks up.
    magnetization_vector: tuple | None = None
    #: ``report_mag``'s per-atom integrals: the charge and the moment inside
    #: each atom's sphere of radius ``r_m``, and that radius in bohr. One row
    #: per atom, three columns for a noncollinear run and one for a collinear
    #: one. Empty when the run printed no report.
    local_charges: np.ndarray | None = None
    local_moments: np.ndarray | None = None
    r_m: tuple | None = None
    homo: float | None = None  # eV, insulators
    lumo: float | None = None  # eV
    forces: np.ndarray | None = None  # (nat,3) Ry/bohr
    #: The force broken into its contributions, which ``pw.x`` prints only with
    #: ``verbosity = 'high'``: ``nonlocal``, ``ionic``, ``local``, ``core``,
    #: ``hubbard``, ``scf_correction``, each ``(nat, 3)`` in Ry/bohr. Empty when
    #: the run did not ask for them. Note that QE's *nonlocal* term already has
    #: the augmentation charge's contribution in it (``addusforce`` is called
    #: from inside ``force_us``) and is symmetrised on its own.
    force_terms: dict = field(default_factory=dict)
    #: The relaxed geometry of a ``calculation = 'relax'`` run, in **bohr**,
    #: cartesian -- what pw.x prints between "Begin final coordinates" and "End
    #: final coordinates", converted out of whatever units the card used.
    final_positions: np.ndarray | None = None
    #: The relaxed **cell** of a ``calculation = 'vc-relax'`` run, in bohr, with
    #: the lattice vectors as rows -- what pw.x prints as ``CELL_PARAMETERS``
    #: inside the same block. ``None`` for a fixed-cell run, which prints none.
    final_cell: np.ndarray | None = None
    #: ``new unit-cell volume``, in bohr^3. Read rather than derived from
    #: :attr:`final_cell`, so that the two are an independent check on each
    #: other -- pw.x prints both and a transposed cell reproduces the volume.
    final_volume: float | None = None
    #: The energy at that geometry ("Final energy"), in Ry. It is not the same
    #: number as :attr:`total_energy`, which is the *first* ionic step's.
    final_energy: float | None = None
    #: The **last** ``!  total energy`` in the file, which for a relaxation is
    #: the answer: the final SCF's for a ``vc-relax`` that runs one, and the
    #: last ionic step's for a ``treinit_gvecs`` run, which does not (every step
    #: already had its own grids). :attr:`total_energy` is the *first*, i.e. the
    #: starting geometry's, and comparing a relaxed result against that is a
    #: mistake with no shape error to announce it.
    final_total_energy: float | None = None
    #: ``Final enthalpy``, in Ry -- what a ``vc-relax`` minimises. ``None`` for
    #: a fixed-cell run, which prints ``Final energy`` instead.
    final_enthalpy: float | None = None
    #: False when the output contains ``convergence NOT achieved``. Worth
    #: having because ``pw.x`` prints its total energy with a leading ``!``
    #: **only** when the SCF converged, so :attr:`final_total_energy` -- "the
    #: last one" -- is really "the last *converged* one", and on a run whose
    #: final SCF failed that is a different quantity from the one its name
    #: suggests: the last ionic step's, in the relaxation's own basis. That is
    #: exactly what a ``vc-relax`` that closes a gap under pressure produces,
    #: and comparing against it silently compares two different calculations.
    scf_converged: bool = True
    #: ``(scf cycles, bfgs steps)`` from "bfgs converged in N scf cycles and M
    #: bfgs steps", or ``None`` for a run that is not a relaxation.
    bfgs_steps: tuple[int, int] | None = None
    stress: np.ndarray | None = None  # (3,3) Ry/bohr^3
    pressure: float | None = None  # kbar
    #: The per-contribution stress ``verbosity = 'high'`` prints, keyed by the
    #: same names :mod:`pypresso.stress.analytic` uses, **in Ry/bohr^3** -- QE
    #: prints this table in kbar and nowhere else, so the conversion happens
    #: here, which is the only layer allowed to do it. Empty for a run that did
    #: not ask for the breakdown.
    stress_terms: dict = field(default_factory=dict)
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
    local_charges, local_moments = _parse_local_moments(text)
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
        internal_energy=_scalar(text, _INTERNAL_ENERGY),
        eigenvalues=eigenvalues,
        occupations=occupations,
        fermi_energy=_scalar(text, r"the Fermi energy is\s*(" + _FLOAT + ")"),
        fermi_energy_up=_spin_fermi(text, 0),
        fermi_energy_down=_spin_fermi(text, 1),
        magnetization=_last(text, r"total magnetization\s*=\s*(" + _FLOAT + ")"),
        absolute_magnetization=_last(
            text, r"absolute magnetization\s*=\s*(" + _FLOAT + ")"
        ),
        magnetization_vector=_magnetization_vector(text),
        local_charges=local_charges,
        local_moments=local_moments,
        r_m=_parse_r_m(text),
        homo=_parse_homo(text),
        lumo=_parse_lumo(text),
        forces=_parse_forces(text),
        force_terms=_parse_force_terms(text),
        final_positions=_parse_final_positions(text),
        final_cell=_parse_final_cell(text),
        final_volume=_scalar(
            text, r"new unit-cell volume\s*=\s*(" + _FLOAT + r")\s*a\.u\.\^3"
        ),
        final_energy=_scalar(text, r"Final energy\s*=\s*(" + _FLOAT + ")"),
        final_total_energy=_last_scalar(
            text, r"!\s+total energy\s*=\s*(" + _FLOAT + ")"
        ),
        final_enthalpy=_scalar(text, r"Final enthalpy\s*=\s*(" + _FLOAT + ")"),
        scf_converged="convergence NOT achieved" not in text,
        bfgs_steps=_parse_bfgs_steps(text),
        stress=stress,
        pressure=pressure,
        stress_terms=_parse_stress_terms(text),
        n_iterations=_as_int(
            _scalar(text, r"convergence has been achieved in\s*(" + _FLOAT + r")\s*iterations")
        ),
    )


def _magnetization_vector(text: str) -> tuple | None:
    """The last "total magnetization = x y z" line of a noncollinear run."""
    matches = re.findall(
        r"total magnetization\s*=\s*(" + _FLOAT + r")\s+(" + _FLOAT + r")\s+("
        + _FLOAT + r")\s+Bohr",
        text,
    )
    return tuple(float(v) for v in matches[-1]) if matches else None


def _parse_r_m(text: str) -> tuple | None:
    """``new r_m : 0.3572 (alat units) 1.8637 (a.u.) for type 1``, in bohr."""
    matches = re.findall(
        r"new r_m :\s*" + _FLOAT + r"\s*\(alat units\)\s*(" + _FLOAT + r")\s*\(a\.u\.\)",
        text,
    )
    return tuple(float(v) for v in matches) if matches else None


def _parse_local_moments(text: str):
    """``report_mag``'s per-atom block, the *last* one printed.

    Every atom appears once per report, and a run reports at the start and at
    convergence, so the converged values are the last ``nat`` blocks. The
    collinear form prints one number and the noncollinear three.
    """
    blocks = re.findall(
        r"atom number\s*(\d+)\s*relative position :.*?\n"
        # 7.5 appends "(integrated on a sphere of radius ...)" to the charge line
        # and 6.1 does not, so the rest of the line is skipped rather than matched.
        r"\s*charge :\s*(" + _FLOAT + r")[^\n]*\n"
        r"\s*magnetization :\s*((?:\s*" + _FLOAT + r"){1,3})",
        text,
    )
    if not blocks:
        return None, None
    per_atom: dict[int, tuple] = {}
    for index, charge, moment in blocks:
        per_atom[int(index)] = (float(charge), [float(v) for v in moment.split()])
    atoms = sorted(per_atom)
    charges = np.array([per_atom[a][0] for a in atoms])
    moments = np.array([per_atom[a][1] for a in atoms])
    return charges, moments


def _as_int(value: float | None) -> int | None:
    return None if value is None else int(round(value))


def _last(text: str, pattern: str) -> float | None:
    """The **last** match of a pattern printed once per SCF iteration.

    The magnetization is reported at every iteration and again at the end; what
    a comparison wants is the converged one, not the first guess.
    """
    matches = re.findall(pattern, text)
    return float(matches[-1]) if matches else None


def _spin_fermi(text: str, index: int) -> float | None:
    """One of the two Fermi levels a constrained-magnetization run prints."""
    match = re.search(
        r"the spin up/dw Fermi energies are\s*(" + _FLOAT + r")\s+(" + _FLOAT + ")",
        text,
    )
    return float(match.group(index + 1)) if match else None


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
    # PAW's one-centre correction, printed only for a PAW run.
    "one_center_paw": r"one-center paw contrib\.\s*=\s*(" + _FLOAT + ")",
    "dispersion": r"Dispersion Correction\s*=\s*(" + _FLOAT + ")",
}

#: ``internal energy E=F+TS`` is **not** one of the terms, although pw.x prints
#: it in the same block and in the same format. It is the total with the
#: smearing entropy added back, so counting it as a term makes the decomposition
#: fail to sum to the total and makes a term-by-term comparison compare a
#: quantity against itself. It gets its own field.
_INTERNAL_ENERGY = r"internal energy E=F\+TS\s*=\s*(" + _FLOAT + ")"


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
    """``atom    1 type  1   force =  fx  fy  fz`` in Ry/bohr.

    The *first* such block, which is the one belonging to the run's own
    geometry; a relaxation prints one per ionic step and
    :func:`_parse_relaxation` collects those.
    """
    return _forces_after(text, 0)


def _forces_after(text: str, position: int) -> np.ndarray | None:
    """The first force block at or after ``position`` in ``text``."""
    block = re.search(
        r"Forces acting on atoms[^\n]*\n\s*\n((?:\s*atom.*force\s*=.*\n)+)",
        text[position:],
    )
    if block is None:
        return None
    rows = [
        _floats(line.split("force", 1)[1])[:3]
        for line in block.group(1).strip().splitlines()
    ]
    return np.array(rows, dtype=float)


#: The headers ``forces.f90`` prints each contribution under, and the names they
#: are stored under here.
_FORCE_TERMS = {
    "The non-local contrib.  to forces": "nonlocal",
    "The ionic contribution  to forces": "ionic",
    "The local contribution  to forces": "local",
    "The core correction contribution to forces": "core",
    "The Hubbard contrib.    to forces": "hubbard",
    "The SCF correction term to forces": "scf_correction",
    # ``force_london``'s own line. Unlike the others it has no "The" and no
    # column padding -- ``forces.f90`` prints it from a separate statement.
    "Dispersion contribution to forces:": "dispersion",
}


def _parse_force_terms(text: str) -> dict:
    """The per-contribution forces ``verbosity = 'high'`` prints, keyed by name."""
    terms = {}
    for header, name in _FORCE_TERMS.items():
        block = re.search(
            re.escape(header) + r"\n((?:\s*atom.*force\s*=.*\n)+)", text
        )
        if block is None:
            continue
        terms[name] = np.array(
            [
                _floats(line.split("force", 1)[1])[:3]
                for line in block.group(1).strip().splitlines()
            ],
            dtype=float,
        )
    return terms


def _last_final_block(text: str) -> str | None:
    """The **last** "Begin final coordinates" block, which is the answer.

    An ``nspin = 2`` run that converges to zero magnetization prints one, then
    restarts with a nonzero moment to check (``ions_status = 2``,
    ``reset_magn``) and prints another; a ``vc-relax`` prints one before its
    final SCF and QE's own is the later of them. Taking the first -- which a
    non-greedy match over the whole file does -- reads a geometry the run
    subsequently moved away from.
    """
    blocks = re.findall(
        r"Begin final coordinates(.*?)End final coordinates", text, re.S
    )
    return blocks[-1] if blocks else None


def _parse_final_cell(text: str) -> np.ndarray | None:
    """``CELL_PARAMETERS (alat= ...)`` from the final-coordinates block, in bohr.

    The ``alat`` in the header is the run's own and is what the three rows are
    in units of; it is *not* recomputed from the relaxed cell, which is why a
    ``vc-relax``'s printed ``CELL_PARAMETERS`` can have rows of length far from
    one.
    """
    block = _last_final_block(text)
    if block is None:
        return None
    match = re.search(
        r"CELL_PARAMETERS\s*\(\s*alat\s*=\s*(" + _FLOAT + r")\s*\)\s*\n"
        r"((?:\s*" + _FLOAT + r"\s+" + _FLOAT + r"\s+" + _FLOAT + r"\s*\n){3})",
        block,
    )
    if match is None:
        return None
    alat = float(match.group(1))
    rows = [_floats(line)[:3] for line in match.group(2).strip().splitlines()]
    return np.array(rows, dtype=float) * alat


def _parse_final_positions(text: str) -> np.ndarray | None:
    """The relaxed geometry, converted to cartesian bohr.

    ``pw.x`` echoes the ``ATOMIC_POSITIONS`` card it would write, in the units
    the *input* used, so the conversion has to be done here -- and ``crystal``
    needs the cell, which is why the lattice is read first.
    """
    block = _last_final_block(text)
    if block is None:
        return None
    match = re.search(
        r"ATOMIC_POSITIONS\s*\(?(\w+)\)?\s*\n(.*)", block, re.S
    )
    if match is None:
        return None
    units = match.group(1).lower()
    block = match
    positions = np.array(
        [
            _floats(line)[:3]
            for line in block.group(2).strip().splitlines()
            if line.strip()
        ],
        dtype=float,
    )
    if units == "bohr":
        return positions
    if units == "angstrom":
        return positions * ANGSTROM_TO_BOHR
    alat = _scalar(text, r"lattice parameter \(alat\)\s*=\s*(" + _FLOAT + ")")
    if units == "alat":
        return positions * alat
    if units == "crystal":
        # **The final cell, not the starting one.** For a fixed-cell relaxation
        # they are the same array and this distinction is invisible; for a
        # ``vc-relax`` the starting cell is the wrong one by however much the
        # relaxation moved it, and the error is a plausible geometry rather
        # than a shape mismatch. QE prints ``CELL_PARAMETERS`` in the same
        # block precisely because the coordinates are relative to it.
        final = _parse_final_cell(text)
        at = final / alat if final is not None else _parse_axes(text, "crystal axes", "a")
        return positions @ (at * alat)
    raise ValueError(f"unknown ATOMIC_POSITIONS units in a final geometry: {units!r}")


def _parse_bfgs_steps(text: str) -> tuple[int, int] | None:
    """``bfgs converged in N scf cycles and M bfgs steps``."""
    match = re.search(
        r"bfgs converged in\s*(\d+)\s*scf cycles and\s*(\d+)\s*bfgs steps", text
    )
    return None if match is None else (int(match.group(1)), int(match.group(2)))


#: The headers ``stress.f90``'s ``iverbosity > 0`` block prints, mapped onto the
#: names :mod:`pypresso.stress.analytic` gives the same contributions. Only the
#: ones with a counterpart here are listed; the dispersion, XDM, vdW and RISM
#: rows are printed as zeros by every run this project compares against.
_STRESS_TERMS = {
    "kinetic stress (kbar)": "kinetic",
    "local   stress (kbar)": "local",
    "nonloc. stress (kbar)": "nonlocal",
    "hartree stress (kbar)": "hartree",
    "exc-cor stress (kbar)": "xc",
    "corecor stress (kbar)": "core",
    "ewald   stress (kbar)": "ewald",
    "hubbard stress (kbar)": "hubbard",
    # ``stres_london``'s row, which ``stress.f90`` labels by the correction
    # family rather than by the routine: D2 and D3 share it.
    "DFT-D   stress (kbar)": "dispersion",
}


def _parse_stress_terms(text: str) -> dict:
    """The per-contribution stress ``verbosity = 'high'`` prints, in Ry/bohr^3.

    Each block is a label and three numbers, then two unlabelled rows of three.
    QE prints the table in kbar only -- unlike the total, which it prints in both
    -- so this is the one parser here that converts, and it converts *into* the
    internal unit rather than out of it.

    **``exc-cor`` already contains the gradient correction.** ``stres_gradcorr``
    is called on ``sigmaxc`` before the table is printed, so a GGA run's
    ``exc-cor`` row is the diagonal plus the non-diagonal ``v2 grad grad``
    together and there is no separate row for it. Comparing it against a
    transcription's diagonal alone is a mistake that looks like a small error on
    a weakly-inhomogeneous system.
    """
    terms = {}
    for header, name in _STRESS_TERMS.items():
        block = re.search(re.escape(header) + r"((?:.*\n){3})", text)
        if block is None:
            continue
        rows = [_floats(line)[-3:] for line in block.group(1).splitlines()]
        if len(rows) != 3 or any(len(row) != 3 for row in rows):
            continue
        terms[name] = np.array(rows, dtype=float) / RY_TO_KBAR
    return terms


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


# --------------------------------------------------------------------------
# projwfc.x
# --------------------------------------------------------------------------
#
# ``projwfc.x`` is a separate program with its own stdout and its own output
# files, so it gets its own reader rather than fields on :class:`QEReference`.
# Two things are read: the standard output, which carries the state table, the
# per-band projections and the Löwdin charges, and the ``filpdos`` files, which
# carry the projected density of states itself, in states/eV against eV.


@dataclass(frozen=True)
class ProjwfcReference:
    """What one ``projwfc.x`` run printed.

    ``projections`` is ``(nk, nbnd, nstate)`` and is **sparse by construction**:
    ``print_proj`` writes only the projections above 0.001 and rounds them to
    three decimals, so an entry that is zero here may be anything below that.
    That is a limit on what can be compared, not a defect -- the array is
    exactly what QE let out, and 1e-3 is the tolerance it deserves.

    ``charges`` is ``{atom (1-based): {"total": q, "s": q, "pz": q, ...}}``,
    with ``"up"``, ``"down"`` and ``"polarization"`` sub-dictionaries of the
    same shape when the run was spin-polarized.
    """

    path: Path
    states: tuple[dict, ...] = ()
    kpoints: np.ndarray | None = None  # (nk, 3), 2pi/alat, spin blocks stacked
    eigenvalues: np.ndarray | None = None  # (nk, nbnd), eV
    projections: np.ndarray | None = None  # (nk, nbnd, nstate)
    psi2: np.ndarray | None = None  # (nk, nbnd): QE's printed |psi|^2
    charges: dict = field(default_factory=dict)
    spilling: float | None = None
    degauss: float | None = None  # Ry
    ngauss: int | None = None
    natomwfc: int | None = None
    nbnd: int | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "ProjwfcReference":
        return read_projwfc_output(path)


def read_projwfc_output(path: str | Path) -> ProjwfcReference:
    """Parse ``projwfc.x``'s standard output."""
    path = Path(path)
    text = path.read_text(errors="replace")

    states = tuple(
        {
            "index": int(index),
            "atom": int(atom),
            "species": species.strip(),
            "wfc": int(wfc),
            "l": int(l),
            "m": int(m),
        }
        for index, atom, species, wfc, l, m in re.findall(
            r"state #\s*(\d+):\s*atom\s*(\d+)\s*\((\S+)\s*\),\s*wfc\s*(\d+)\s*"
            r"\(l=(\d+)\s*m=\s*(-?\d+)\)",
            text,
        )
    )
    broadening = re.search(r"ngauss,degauss=\s*(-?\d+)\s*(" + _FLOAT + ")", text)
    kpoints, eigenvalues, projections, psi2 = _parse_projections(text, len(states))

    return ProjwfcReference(
        path=path,
        states=states,
        kpoints=kpoints,
        eigenvalues=eigenvalues,
        projections=projections,
        psi2=psi2,
        charges=_parse_lowdin(text),
        spilling=_scalar(text, r"Spilling Parameter:\s*(" + _FLOAT + ")"),
        degauss=None if broadening is None else float(broadening.group(2)),
        ngauss=None if broadening is None else int(broadening.group(1)),
        natomwfc=_as_int(_scalar(text, r"natomwfc\s*=\s*(" + _FLOAT + ")")),
        nbnd=_as_int(_scalar(text, r"nbnd\s*=\s*(" + _FLOAT + ")")),
    )


def _parse_projections(text: str, nstate: int):
    """``print_proj``'s per-k, per-band block."""
    blocks = re.split(r"\n k =", text)[1:]
    if not blocks or nstate == 0:
        return None, None, None, None

    kpoints, energies, weights, norms = [], [], [], []
    for block in blocks:
        block = block.split("Lowdin Charges")[0]
        head, *bands = re.split(r"====\s*e\(", block)
        kpoints.append(_floats(head.splitlines()[0])[:3])
        energies.append([])
        weights.append([])
        norms.append([])
        for band in bands:
            energies[-1].append(_floats(band.split("=")[1])[0])
            row = np.zeros(nstate)
            for value, index in re.findall(r"(" + _FLOAT + r")\*\[#\s*(\d+)\]", band):
                row[int(index) - 1] = float(value)
            weights[-1].append(row)
            norm = re.search(r"\|psi\|\^2\s*=\s*(" + _FLOAT + ")", band)
            norms[-1].append(float(norm.group(1)) if norm else np.nan)

    return np.array(kpoints), np.array(energies), np.array(weights), np.array(norms)


#: A ``label = value`` pair inside a Löwdin charge line: ``s =  1.1008`` or
#: ``pz=  0.9555``. Anchored on the spectroscopic letter so that the ``down`` of
#: "spin down" and the ``polarization`` of the last line cannot be mistaken for
#: an ``l`` -- both start with one, which is why the header is stripped first.
_LOWDIN_PAIR = re.compile(r"\b([spdf][a-z0-9+\-]*)\s*=\s*(" + _FLOAT + ")")


def _parse_lowdin(text: str) -> dict:
    """``print_lowdin``'s block, as ``{atom: {label: charge}}``."""
    block = text.split("Lowdin Charges:")
    if len(block) < 2:
        return {}
    charges: dict = {}
    atom = None
    for line in block[1].splitlines():
        if "Spilling Parameter" in line:
            break
        header = re.match(
            r"\s*Atom #\s*(\d+):\s*total charge\s*=\s*(" + _FLOAT + ")", line
        )
        if header:
            atom = int(header.group(1))
            bucket = charges.setdefault(atom, {})
            bucket["total"] = float(header.group(2))
            bucket.update(_lowdin_pairs(line))
            continue
        spin = re.match(
            r"\s*(spin up|spin down|polarization)\s*=\s*(" + _FLOAT + ")", line
        )
        if spin and atom is not None:
            key = {"spin up": "up", "spin down": "down"}.get(
                spin.group(1), "polarization"
            )
            bucket = charges[atom].setdefault(key, {})
            bucket["total"] = float(spin.group(2))
            bucket.update(_lowdin_pairs(line))
    return charges


def _lowdin_pairs(line: str) -> dict:
    """The ``l`` and ``lm`` charges on one line, past its ``... = value`` header."""
    _, _, rest = line.partition("=")
    return {
        label: float(value.replace("D", "E"))
        for label, value in _LOWDIN_PAIR.findall(rest)
    }


def read_pdos_file(path: str | Path) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """One ``filpdos`` file: ``(energies_eV, columns, names)``.

    ``columns`` is ``(ncolumn, nE)`` in states/eV -- ``partialdos`` writes the
    energy first and then ``ldos`` (and ``pdos``) columns, one per spin channel
    per ``m``. The names come from the header line, which is the only place the
    file says what its columns are.
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    names = tuple(lines[0].lstrip("#").split())
    values = np.array([_floats(line) for line in lines[1:] if line.strip()])
    return values[:, 0], values[:, 1:].T, names[2:]
