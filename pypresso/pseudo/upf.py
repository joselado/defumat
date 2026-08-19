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

__all__ = ["Pseudopotential", "Projector", "AtomicOrbital", "Augmentation", "PawData",
           "read_upf"]


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


@dataclass(frozen=True, eq=False)
class Augmentation:
    """The augmentation charges ``Q_ij(r)`` of an ultrasoft or PAW potential.

    An ultrasoft pseudopotential drops the norm-conservation constraint, so
    ``<psi|psi>`` is no longer the electron count and the missing charge has to
    be put back explicitly. What restores it is

        Q_ij(r) = phi_i^AE*(r) phi_j^AE(r) - phi_i^PS*(r) phi_j^PS(r)

    -- the difference between the all-electron and the pseudo partial waves --
    added to the density wherever the projectors ``beta_i`` overlap, and added to
    the overlap operator as ``q_ij = int Q_ij(r) dr``. That charge is sharply
    peaked, which is the whole reason for the second, denser FFT grid.

    ``qfuncl[nb, mb, L]`` is ``r^2 Q^L_{nb mb}(r)`` for the angular momentum
    ``L`` component, as ``PP_QIJL`` tabulates it -- the ``q_with_l`` form. The
    older format instead stores one ``Q_ij(r)`` per pair plus ``qfcoef``
    polynomial coefficients to re-pseudize it inside ``rinner``; that branch is
    not implemented, so such a file parses (``q_with_l`` is false, ``qfuncl`` is
    ``None``) and is refused only when a calculation actually asks for the
    augmentation charge. Reading it has to keep working: several of the
    committed test files are in the old format, and everything *else* about them
    is read correctly.
    """

    q: np.ndarray  # (nbeta, nbeta) the integrated q_ij, from PP_Q
    qfuncl: np.ndarray | None  # (nbeta, nbeta, nqlc, mesh), r^2 Q^L_ij(r)
    nqlc: int  # number of L components stored, 2*lmax+1
    q_with_l: bool = True

    @property
    def nbeta(self) -> int:
        return self.q.shape[0]


@dataclass(frozen=True, eq=False)
class PawData:
    """The one-centre data of a PAW dataset (``PP_PAW`` and ``PP_FULL_WFC``).

    PAW is ultrasoft plus a pair of atom-centred corrections: everything the
    plane-wave grid gets wrong inside the augmentation sphere is recomputed
    twice on the atom's radial mesh -- once with the all-electron partial waves
    and once with the pseudo ones -- and the difference is added back. So this
    carries the two sets of partial waves, the all-electron local potential and
    core charge they are to be evaluated against, and the reference occupations
    that fix the zero of the correction.
    """

    ae_wfc: np.ndarray  # (nbeta, mesh) r * phi^AE_i(r)
    ps_wfc: np.ndarray  # (nbeta, mesh) r * phi^PS_i(r)
    ae_vloc: np.ndarray  # (mesh,) the all-electron local potential, Ry
    ae_rho_core: np.ndarray | None  # (mesh,) the true core charge
    occupations: np.ndarray  # (nbeta,) reference occupation of each channel
    core_energy: float  # Ry, the frozen core's own energy
    augmentation_shape: str  # 'PSQ', 'GAUSS', ... -- how Q was pseudized
    cutoff_index: int  # iraug: the augmentation sphere's mesh index


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
    #: Logarithmic mesh step (``PP_MESH``'s ``dx``). Only the PAW radial Poisson
    #: solver needs it -- it is a Numerov scheme on the log mesh, so the step is
    #: part of the discretisation rather than a property of the tabulated data.
    dx: float = 0.0
    augmentation: Augmentation | None = None
    paw: PawData | None = None
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
        """Whether the potential carries an augmentation charge (QE's ``tvanp``).

        ``pseudo_type`` is a free-text header field and the generators disagree:
        ``atomic`` writes ``USPP`` where the format documentation says ``US``,
        and PAW files say ``PAW``. Matching only ``US`` reads a perfectly good
        ultrasoft file as norm-conserving -- which is not an error anywhere, just
        a calculation with the augmentation charge missing and an energy wrong in
        the first decimal.
        """
        return self.pseudo_type.upper() in ("US", "USPP", "PAW")

    @property
    def is_paw(self) -> bool:
        return self.pseudo_type.upper() == "PAW"

    @property
    def has_nlcc(self) -> bool:
        return self.rho_core is not None

    @property
    def msh(self) -> int:
        """Mesh length QE integrates over: truncated at 10 bohr, odd."""
        return mesh_cutoff_index(self.r)

    @property
    def kkbeta(self) -> int:
        """The mesh index the projectors and ``Q`` functions are integrated to.

        QE's ``upf%kkbeta``: the largest of the projectors' own cutoffs, widened
        for PAW to cover the augmentation sphere. The ``Q`` functions are
        integrated over exactly this range and not over the 10-bohr mesh the
        local potential uses -- a different range there is a ~1e-4 error in the
        augmentation charge, small enough to survive a loose check and large
        enough to ruin the total energy.
        """
        cutoff = max((p.cutoff_index for p in self.projectors), default=self.mesh)
        if self.paw is not None:
            cutoff = max(cutoff, self.paw.cutoff_index)
        return min(cutoff, self.mesh)

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
    dx = float(mesh_node.attrib.get("dx", "0") or 0.0)
    r = _numbers(_require(mesh_node, "PP_R"))
    rab = _numbers(_require(mesh_node, "PP_RAB"))
    if len(r) != len(rab):
        raise ValueError(f"{path}: PP_R has {len(r)} points but PP_RAB has {len(rab)}")

    vloc = _numbers(_require(root, "PP_LOCAL"))

    projectors, dij, augmentation = _read_nonlocal(root, path)
    orbitals = _read_orbitals(root)
    paw = _read_paw(root, len(projectors))

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
        dx=dx,
        vloc=vloc,
        projectors=projectors,
        dij=dij,
        rho_atom=rho_atom,
        rho_core=rho_core,
        orbitals=orbitals,
        augmentation=augmentation,
        paw=paw,
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
        return (), None, None

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

    return tuple(projectors), dij, _read_augmentation(section, len(projectors), path)


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


def _read_augmentation(section: ET.Element, nbeta: int, path: Path) -> Augmentation | None:
    """Parse ``PP_AUGMENTATION``: the integrated ``q_ij`` and the ``Q^L_ij(r)``."""
    node = section.find("PP_AUGMENTATION")
    if node is None or nbeta == 0:
        return None

    q_with_l = node.attrib.get("q_with_l", "F").strip().upper() in ("T", "TRUE", ".TRUE.")

    q_node = _require(node, "PP_Q")
    q = _numbers(q_node)
    if q.size != nbeta * nbeta:
        raise ValueError(f"{path}: PP_Q has {q.size} entries, expected {nbeta * nbeta}")
    q = q.reshape(nbeta, nbeta)

    nqlc = int(node.attrib.get("nqlc", "0") or 0)
    cutoff_index = int(node.attrib.get("cutoff_r_index", "0") or 0)
    if not q_with_l:
        # The qfcoef/rinner re-pseudization is not implemented; q_ij is still
        # read, since it is what the overlap operator needs and it is tabulated
        # directly rather than derived from Q(r).
        return Augmentation(q=q, qfuncl=None, nqlc=nqlc, q_with_l=False)

    mesh = None
    entries = []
    for child in node:
        if not child.tag.startswith("PP_QIJL"):
            continue
        values = _numbers(child)
        mesh = values.size if mesh is None else mesh
        entries.append((
            int(child.attrib["first_index"]) - 1,
            int(child.attrib["second_index"]) - 1,
            int(child.attrib["angular_momentum"]),
            values,
        ))
    if not entries:
        raise ValueError(f"{path}: PP_AUGMENTATION has q_with_l='T' but no PP_QIJL sections")

    nqlc = max(nqlc, max(l for _, _, l, _ in entries) + 1)
    qfuncl = np.zeros((nbeta, nbeta, nqlc, mesh))
    for nb, mb, l, values in entries:
        # Only the upper triangle is stored; Q is symmetric in its two indices.
        qfuncl[nb, mb, l] = values
        qfuncl[mb, nb, l] = values
    if cutoff_index:
        # ``read_upf_new`` zeroes Q beyond the augmentation radius for a PAW
        # dataset. The files here already are zero there, but a dataset that is
        # not would otherwise carry its tail into the one-centre integrals,
        # which run over the whole mesh.
        qfuncl[:, :, :, cutoff_index:] = 0.0

    return Augmentation(q=q, qfuncl=qfuncl, nqlc=nqlc, q_with_l=True)


def _read_paw(root: ET.Element, nbeta: int) -> PawData | None:
    """Parse ``PP_PAW`` and ``PP_FULL_WFC`` -- the one-centre data."""
    section = root.find("PP_PAW")
    full = root.find("PP_FULL_WFC")
    if section is None or full is None:
        return None

    def _wavefunctions(prefix: str) -> np.ndarray:
        nodes = sorted(
            (child for child in full if child.tag.startswith(prefix)),
            key=lambda child: int(child.attrib.get("index", "0")),
        )
        return np.stack([_numbers(node) for node in nodes])

    occupations = _optional_numbers(section, "PP_OCCUPATIONS")
    augmentation = root.find("PP_NONLOCAL/PP_AUGMENTATION")
    attrib = {} if augmentation is None else augmentation.attrib

    return PawData(
        ae_wfc=_wavefunctions("PP_AEWFC"),
        ps_wfc=_wavefunctions("PP_PSWFC"),
        ae_vloc=_numbers(_require(section, "PP_AE_VLOC")),
        ae_rho_core=_optional_numbers(section, "PP_AE_NLCC"),
        occupations=np.zeros(nbeta) if occupations is None else occupations,
        core_energy=float(section.attrib.get("core_energy", "0") or 0.0),
        augmentation_shape=attrib.get("shape", "").strip(),
        cutoff_index=int(attrib.get("cutoff_r_index", "0") or 0),
    )
