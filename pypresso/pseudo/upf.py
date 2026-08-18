"""Reader for UPF (Unified Pseudopotential Format) version 2 files.

Only what a plane-wave SCF needs is extracted: the radial mesh, the local
potential, the nonlocal projectors with their ``D_ij`` coefficients, the atomic
charge used to start the SCF, the pseudo-atomic orbitals, and the core charge
when the pseudopotential has a nonlinear core correction.

Everything is kept in the file's units, which are Rydberg atomic units
throughout -- ``PP_LOCAL`` and ``PP_DIJ`` in Ry, radii in bohr -- so no
conversion happens here.

Arrays are NumPy: a pseudopotential file is setup data, read once, and the
radial tables are constants that later phases transform into G space (rule R2).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pypresso.pseudo.radial import mesh_cutoff_index

__all__ = ["Pseudopotential", "Projector", "AtomicOrbital", "read_upf"]


@dataclass(frozen=True, eq=False)
class Projector:
    """One nonlocal projector ``beta_l(r)``, as tabulated (already times r)."""

    l: int
    beta: np.ndarray  # (mesh,) -- r * beta_l(r), QE's convention
    cutoff_index: int  # kkbeta: beyond this the projector is zero
    label: str = ""


@dataclass(frozen=True, eq=False)
class AtomicOrbital:
    """A pseudo-atomic orbital ``chi(r)``, used to seed the wavefunctions."""

    l: int
    occupation: float
    chi: np.ndarray  # (mesh,) -- r * chi(r)
    label: str = ""


#: ``eq=False`` gives these identity-based equality and hashing. They hold NumPy
#: arrays, which are unhashable, and they are carried as *static* fields of JAX
#: modules -- so they must be hashable, and identity is the right notion anyway:
#: a pseudopotential is read once and never rebuilt.
@dataclass(frozen=True, eq=False)
class Pseudopotential:
    """A norm-conserving or ultrasoft pseudopotential read from a UPF file."""

    element: str
    z_valence: float
    pseudo_type: str  # 'NC', 'SL', 'US', 'PAW'
    functional: str
    r: np.ndarray  # (mesh,) radial grid, bohr
    rab: np.ndarray  # (mesh,) dr/di, for Simpson integration
    vloc: np.ndarray  # (mesh,) local potential, Ry
    projectors: tuple[Projector, ...] = ()
    dij: np.ndarray | None = None  # (nbeta, nbeta), Ry
    rho_atom: np.ndarray | None = None  # (mesh,) 4 pi r^2 rho(r)
    rho_core: np.ndarray | None = None  # (mesh,) core charge for NLCC
    orbitals: tuple[AtomicOrbital, ...] = ()
    path: Path | None = None
    header: dict = field(default_factory=dict)

    @property
    def mesh(self) -> int:
        return len(self.r)

    @property
    def nbeta(self) -> int:
        return len(self.projectors)

    @property
    def lmax(self) -> int:
        """Highest projector angular momentum (-1 when there are none)."""
        return max((p.l for p in self.projectors), default=-1)

    @property
    def is_ultrasoft(self) -> bool:
        return self.pseudo_type.upper() in ("US", "PAW")

    @property
    def has_nlcc(self) -> bool:
        return self.rho_core is not None

    @property
    def msh(self) -> int:
        """Mesh length QE integrates over: truncated at 10 bohr, odd."""
        return mesh_cutoff_index(self.r)

    @property
    def nh(self) -> int:
        """Number of (projector, m) pairs -- the size of the ``D`` matrix block."""
        return sum(2 * p.l + 1 for p in self.projectors)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Pseudopotential({self.element}, {self.pseudo_type}, Z={self.z_valence:g}, "
            f"{self.nbeta} projectors (l={[p.l for p in self.projectors]}), "
            f"mesh {self.mesh} (msh {self.msh}), {self.functional.strip()})"
        )


def read_upf(path: str | Path) -> Pseudopotential:
    """Parse a UPF v2 file."""
    path = Path(path)
    root = ET.parse(path).getroot()

    header = dict(_require(root, "PP_HEADER").attrib)
    version = root.attrib.get("version", "")
    if version and not version.startswith("2"):
        raise NotImplementedError(f"{path}: UPF version {version!r}; only v2 is supported")

    mesh_node = _require(root, "PP_MESH")
    r = _numbers(_require(mesh_node, "PP_R"))
    rab = _numbers(_require(mesh_node, "PP_RAB"))
    if len(r) != len(rab):
        raise ValueError(f"{path}: PP_R has {len(r)} points but PP_RAB has {len(rab)}")

    vloc = _numbers(_require(root, "PP_LOCAL"))

    projectors, dij = _read_nonlocal(root, path)
    orbitals = _read_orbitals(root)

    rho_atom = _optional_numbers(root, "PP_RHOATOM")
    rho_core = _optional_numbers(root, "PP_NLCC")

    for name, array in (("PP_LOCAL", vloc), ("PP_RHOATOM", rho_atom)):
        if array is not None and len(array) != len(r):
            raise ValueError(f"{path}: {name} has {len(array)} points, mesh has {len(r)}")

    return Pseudopotential(
        element=header.get("element", "").strip(),
        z_valence=float(header.get("z_valence", "nan")),
        pseudo_type=header.get("pseudo_type", "NC").strip(),
        functional=header.get("functional", "").strip(),
        r=r,
        rab=rab,
        vloc=vloc,
        projectors=projectors,
        dij=dij,
        rho_atom=rho_atom,
        rho_core=rho_core,
        orbitals=orbitals,
        path=path,
        header=header,
    )


def _require(node: ET.Element, tag: str) -> ET.Element:
    found = node.find(tag)
    if found is None:
        raise ValueError(f"UPF file is missing the required section <{tag}>")
    return found


def _numbers(node: ET.Element) -> np.ndarray:
    text = (node.text or "").replace("D", "E").replace("d", "e")
    return np.fromstring(text, sep=" ")


def _optional_numbers(root: ET.Element, tag: str) -> np.ndarray | None:
    node = root.find(tag)
    return None if node is None else _numbers(node)


def _read_nonlocal(root: ET.Element, path: Path):
    section = root.find("PP_NONLOCAL")
    if section is None:
        return (), None

    projectors = []
    for node in sorted(
        (child for child in section if child.tag.startswith("PP_BETA")),
        key=lambda child: int(child.attrib.get("index", "0")),
    ):
        beta = _numbers(node)
        # kkbeta: the projector vanishes beyond this index. Some files leave it
        # at 0, meaning "use the whole mesh".
        cutoff = int(node.attrib.get("cutoff_radius_index", "0") or 0)
        projectors.append(
            Projector(
                l=int(node.attrib["angular_momentum"]),
                beta=beta,
                cutoff_index=min(cutoff, len(beta)) if cutoff > 0 else len(beta),
                label=node.attrib.get("label", ""),
            )
        )

    dij_node = section.find("PP_DIJ")
    dij = None
    if dij_node is not None and projectors:
        values = _numbers(dij_node)
        n = len(projectors)
        if values.size != n * n:
            raise ValueError(f"{path}: PP_DIJ has {values.size} entries, expected {n * n}")
        dij = values.reshape(n, n)

    return tuple(projectors), dij


def _read_orbitals(root: ET.Element) -> tuple[AtomicOrbital, ...]:
    section = root.find("PP_PSWFC")
    if section is None:
        return ()

    orbitals = []
    for node in sorted(
        (child for child in section if child.tag.startswith("PP_CHI")),
        key=lambda child: int(child.attrib.get("index", "0")),
    ):
        orbitals.append(
            AtomicOrbital(
                l=int(node.attrib["l"]),
                occupation=float(node.attrib.get("occupation", "0")),
                chi=_numbers(node),
                label=node.attrib.get("label", ""),
            )
        )
    return tuple(orbitals)
