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

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from defumat.pseudo.radial import mesh_cutoff_index, simpson_weights

#: QE's ``upf_const`` thresholds, used by :func:`_renormalize_orbitals` exactly
#: as ``upf_check_atwfc_norm`` uses them.
_EPS6 = 1.0e-6
_EPS8 = 1.0e-8

__all__ = ["Pseudopotential", "Projector", "AtomicOrbital", "Augmentation", "PawData",
           "read_upf"]


@dataclass(frozen=True, eq=False)
class Projector:
    """One nonlocal projector ``beta_l(r)``, as tabulated (already times r)."""

    l: int
    beta: np.ndarray  # (mesh,) -- r * beta_l(r), QE's convention
    cutoff_index: int  # kkbeta: beyond this the projector is zero
    label: str = ""
    #: ``jjj``, the total angular momentum ``j = l +- 1/2`` this projector was
    #: generated for, from ``PP_RELBETA``. ``None`` on a scalar-relativistic
    #: file. It is the *only* thing that distinguishes the two projectors of a
    #: fully-relativistic set that share an ``l``, and it is what the spin-orbit
    #: coupling is built from -- see :mod:`defumat.pseudo.spinorbit`.
    j: float | None = None


@dataclass(frozen=True, eq=False)
class AtomicOrbital:
    """A pseudo-atomic orbital ``chi(r)``, used to seed the wavefunctions."""

    l: int
    occupation: float
    chi: np.ndarray  # (mesh,) -- r * chi(r)
    label: str = ""
    #: ``jchi`` from ``PP_RELWFC``; ``None`` on a scalar-relativistic file.
    j: float | None = None


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
    ``L`` component. ``PP_QIJL`` tabulates exactly that (the ``q_with_l`` form);
    the older Vanderbilt form stores one ``Q_ij(r)`` per pair, valid for every
    ``L`` the pair couples to outside ``rinner(L)``, plus the ``qfcoef``
    polynomial that replaces it inside. Both are expanded to the same array when
    the file is read, which is what ``set_upf_q`` does in QE and for the same
    reason: only one place should know that there are two forms. ``q_with_l``
    records which one the file used.
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
    #: ``PP_AEWFC_REL``: the *small* component of the Dirac partial waves, which
    #: only a fully-relativistic dataset carries. QE uses it in one place --
    #: ``PAW_potential``'s ``with_small_so`` branch, which is entered only when
    #: ``nspin_mag == 4``, i.e. for a noncollinear run with a magnetization. It
    #: is read here so that the file is parsed completely and so that the tag
    #: cannot be confused with ``PP_AEWFC`` again; nothing consumes it yet, and
    #: :mod:`defumat.scf.potential` refuses that combination anyway.
    ae_wfc_rel: np.ndarray | None = None


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
    def has_so(self) -> bool:
        """Whether the file carries the ``j`` of each projector (``has_so``).

        True exactly when the pseudopotential was generated fully
        relativistically *and* kept the two ``j`` channels apart, which is what
        a spin-orbit calculation needs. A "scalar-relativistic" file solved the
        same Dirac equation and then averaged the pair away, so it is not a
        weaker version of this -- the information is gone.
        """
        return any(projector.j is not None for projector in self.projectors)

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


#: An ``&`` that is not the start of an XML entity. ``ld1.x`` writes the
#: generator's own namelist input into ``PP_INPUTFILE`` verbatim, and older
#: releases did not escape it -- ``qe-7.5/pseudo/Fe.pz-n-nc.UPF`` carries a bare
#: ``&input`` on line 27 and is not well-formed XML because of it. QE reads such
#: a file: ``upflib/xmltools.f90`` is a hand-written scanner, not an XML parser,
#: and never looks at an entity. See :func:`_upf_document`.
_STRAY_AMPERSAND = re.compile(rb"&(?!(?:amp|lt|gt|quot|apos|#[0-9]+|#x[0-9a-fA-F]+);)")

#: What the first non-blank line of each format this reader does *not* read looks
#: like, so the refusal can name the format instead of leaking a parse error.
_FOREIGN_FORMATS = (
    ("<PP_INFO", "UPF version 1, whose elements are bare tags with no <UPF> root"),
    ("<PP_HEADER", "UPF version 1, whose elements are bare tags with no <UPF> root"),
)


def _upf_document(path: Path) -> ET.Element:
    """The file's root element, with the two things that are not XML dealt with.

    **A stray ``&`` is repaired rather than refused.** It only ever appears
    inside ``PP_INPUTFILE``, which nothing here reads, and the substitution
    cannot turn a well-formed document into a malformed one -- it rewrites
    exactly those ``&`` that no XML parser would accept.

    **A file that is not UPF v2 is refused by name.** ``ET.parse`` on a v1 file
    says ``junk after document element: line 14, column 0``, which names neither
    the file's format nor this reader's, and the same goes for the Vanderbilt
    ``.van`` and ``.RRKJ3`` tables QE still reads. The version attribute is
    checked below for the files that *do* have a root element; this catches the
    ones that have none.
    """
    text = path.read_bytes()
    if b"<UPF" not in text[:4096]:
        head = next(
            (line.strip() for line in text[:4096].splitlines() if line.strip()), b""
        ).decode("latin-1")[:60]
        for marker, description in _FOREIGN_FORMATS:
            if head.startswith(marker):
                raise NotImplementedError(
                    f"{path}: this looks like {description}. Only UPF v2 is read "
                    "here; convert it with upflib/upfconv.x"
                )
        raise NotImplementedError(
            f"{path}: not a UPF v2 file -- it has no <UPF> root element and "
            f"starts {head!r}. QE also reads the Vanderbilt (.van), RRKJ3 and "
            "UPF v1 tables; this reader does not. Convert it with "
            "upflib/upfconv.x"
        )
    return ET.fromstring(_STRAY_AMPERSAND.sub(b"&amp;", text))


def read_upf(path: str | Path) -> Pseudopotential:
    """Parse a UPF v2 file."""
    path = Path(path)
    root = _upf_document(path)

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

    projectors, dij, augmentation = _read_nonlocal(root, path, r)
    orbitals = _read_orbitals(root)
    projectors, orbitals = _read_spin_orbit(root, path, header, projectors, orbitals)
    paw = _read_paw(root, len(projectors))
    orbitals = _renormalize_orbitals(orbitals, projectors, augmentation, rab)

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


def _renormalize_orbitals(orbitals, projectors, augmentation, rab):
    """``upf_check_atwfc_norm``: rescale the atomic orbitals to unit norm.

    ``Modules/read_pseudo.f90`` calls this on every file as it is read, so
    everything downstream of QE's reader sees *renormalised* ``chi``, and the
    generator's own normalisation is discarded. The norm is taken **in the
    generalised metric**,

        <chi|S|chi> = int (r chi)^2 dr
                      + sum_ij q_ij <beta_i|chi> <beta_j|chi>

    with the second term present only for an ultrasoft or PAW dataset -- which
    is why an orbital can be normalised in the file and not here, and vice
    versa. QE prints the labels it rescaled (``wavefunction(s) 4S
    renormalized``), and both ``Fe.pz-nd-rrkjus`` and ``Ni.pz-nd-rrkjus``
    appear in that list.

    **This is not cosmetic and it is silent where it is not.** The starting
    wavefunctions do not care -- they are rotated afterwards. The DFT+U
    projectors do: with ``ortho-atomic`` projectors the ``4s`` orbital enters
    the overlap matrix whose inverse square root orthogonalises the ``3d``
    manifold, so an unrescaled ``4s`` moves the occupation matrix of the ``3d``
    shell. On fcc nickel that is 4e-3 in ``Tr[ns]`` and 7e-4 Ry in the total
    energy -- small enough to look like a convergence difference and large
    enough to be wrong.

    An orbital whose norm underflows is not rescaled but *dropped*: QE sets its
    occupation to a small negative number, which is the flag every consumer
    already reads as "not an atomic wavefunction".

    The one thing it changes outside DFT+U is the *starting wavefunctions*, which
    are built from the same ``chi``. That is a different seed for the
    eigensolver and nothing more -- except where an SCF trajectory is itself
    sensitive to where it starts, which the fixed-spin-moment feedback of P18 is:
    ``fe-fsm.in`` reaches the same moment and the same field, and takes 746
    iterations instead of ~350 to stop ringing on the way.
    """
    if not orbitals:
        return orbitals
    weights = np.asarray(simpson_weights(rab))
    qqq = None if augmentation is None else np.asarray(augmentation.q)

    renormalized = []
    for orbital in orbitals:
        chi = np.asarray(orbital.chi)
        norm = float(np.sum(chi * chi * weights))
        if norm < _EPS8:
            renormalized.append(
                replace(orbital, occupation=-_EPS8)
            )
            continue
        if orbital.occupation < 0.0:
            renormalized.append(orbital)
            continue
        if qqq is not None and projectors:
            overlaps = np.zeros(len(projectors))
            for i, projector in enumerate(projectors):
                if projector.l != orbital.l:
                    continue
                if (
                    projector.j is not None and orbital.j is not None
                    and abs(projector.j - orbital.j) >= _EPS6
                ):
                    continue
                cut = projector.cutoff_index
                overlaps[i] = float(
                    np.sum(projector.beta[:cut] * chi[:cut] * weights[:cut])
                )
            norm += float(overlaps @ qqq @ overlaps)
        norm = np.sqrt(norm)
        if abs(norm - 1.0) > _EPS6:
            renormalized.append(replace(orbital, chi=chi / norm))
        else:
            renormalized.append(orbital)
    return tuple(renormalized)


def _read_spin_orbit(root, path: Path, header: dict, projectors, orbitals):
    """Attach ``j`` from ``PP_SPIN_ORB`` to the projectors and the orbitals.

    A fully-relativistic pseudopotential is generated by solving the Dirac
    equation, so each ``l > 0`` channel comes in two: ``j = l - 1/2`` and
    ``j = l + 1/2``. The radial functions are already in ``PP_BETA`` -- there is
    nothing extra to read there -- and the *only* new information in this
    section is which ``j`` each of them belongs to. That single number is what
    the whole spin-orbit term is built from
    (:mod:`defumat.pseudo.spinorbit`): without it the two projectors of a pair
    are indistinguishable and the potential is the ``j``-averaged, scalar-
    relativistic one.

    ``has_so`` in the header and the presence of the section are checked against
    each other, because a file claiming one and carrying the other would
    otherwise run as though it were scalar-relativistic and be wrong by the
    entire spin-orbit splitting -- tenths of an eV on a heavy element.
    """
    claimed = str(header.get("has_so", "")).strip().lower() in ("t", "true", ".true.")
    section = root.find("PP_SPIN_ORB")
    if section is None:
        if claimed:
            raise ValueError(
                f"{path}: PP_HEADER says has_so but there is no PP_SPIN_ORB section"
            )
        return projectors, orbitals
    if not claimed:
        raise ValueError(
            f"{path}: there is a PP_SPIN_ORB section but PP_HEADER does not say has_so"
        )

    def indexed(prefix: str, count: int, attribute: str):
        found = {}
        for child in section:
            if not child.tag.startswith(prefix):
                continue
            index = int(child.attrib.get("index", "0"))
            found[index] = float(child.attrib[attribute])
        missing = [i for i in range(1, count + 1) if i not in found]
        if missing:
            raise ValueError(
                f"{path}: PP_SPIN_ORB is missing {prefix} entries {missing}"
            )
        return [found[i] for i in range(1, count + 1)]

    jjj = indexed("PP_RELBETA", len(projectors), "jjj")
    for projector, j in zip(projectors, jjj):
        if abs(abs(j - projector.l) - 0.5) > 1.0e-6:
            raise ValueError(
                f"{path}: PP_RELBETA gives j = {j} for a projector with l = "
                f"{projector.l}; j must be l +- 1/2"
            )
    projectors = tuple(
        replace(projector, j=j) for projector, j in zip(projectors, jjj)
    )

    if orbitals:
        jchi = indexed("PP_RELWFC", len(orbitals), "jchi")
        orbitals = tuple(
            replace(orbital, j=j) for orbital, j in zip(orbitals, jchi)
        )
    return projectors, orbitals


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


def _read_nonlocal(root: ET.Element, path: Path, r: np.ndarray):
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

    return tuple(projectors), dij, _read_augmentation(section, projectors, r, path)


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


def _read_augmentation(
    section: ET.Element, projectors, r: np.ndarray, path: Path
) -> Augmentation | None:
    """Parse ``PP_AUGMENTATION``: the integrated ``q_ij`` and the ``Q^L_ij(r)``.

    Two storage forms, and both end up as the same ``qfuncl`` array, which is
    what ``upflib/upf_to_internal.f90``'s ``set_upf_q`` exists to arrange:

    * ``q_with_l = T`` tabulates one ``PP_QIJL`` per ``(i, j, L)`` and there is
      nothing to do;
    * ``q_with_l = F`` -- the Vanderbilt form every ``rrkjus`` file in the test
      set uses -- tabulates a single ``PP_QIJ`` per pair, which stands for
      *every* ``L`` the pair couples to, optionally re-pseudized inside
      ``rinner(L)`` by the ``nqf`` polynomial coefficients of ``PP_QFCOEF``.

    Expanding the second form here rather than in the consumer is QE's own
    choice, and for the same reason: it is the difference between one place
    knowing about ``qfcoef`` and every place that touches ``Q`` knowing.
    """
    node = section.find("PP_AUGMENTATION")
    nbeta = len(projectors)
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

    if q_with_l:
        qfuncl, nqlc = _qijl_sections(node, nbeta, nqlc, path)
    else:
        qfuncl, nqlc = _expand_qij(node, projectors, r, nqlc, path)

    if cutoff_index:
        # ``read_upf_new`` zeroes Q beyond the augmentation radius for a PAW
        # dataset. The files here already are zero there, but a dataset that is
        # not would otherwise carry its tail into the one-centre integrals,
        # which run over the whole mesh.
        qfuncl[:, :, :, cutoff_index:] = 0.0

    return Augmentation(q=q, qfuncl=qfuncl, nqlc=nqlc, q_with_l=q_with_l)


def _qijl_sections(node: ET.Element, nbeta: int, nqlc: int, path: Path):
    """The ``q_with_l = T`` form: one tabulated ``Q^L_ij(r)`` per section."""
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
    return qfuncl, nqlc


def _expand_qij(node: ET.Element, projectors, r: np.ndarray, nqlc: int, path: Path):
    """The ``q_with_l = F`` form, expanded onto the L-dependent grid.

    Transcribed from ``set_upf_q``: one ``Q_ij(r)`` per pair is copied to every
    ``L`` with ``|l_i - l_j| <= L <= l_i + l_j`` and ``L + l_i + l_j`` even --
    the same triangle-and-parity rule the transform to G space applies -- and
    then, where ``nqf > 0`` and ``rinner(L) > 0``, the region inside
    ``rinner(L)`` is replaced by ``setqfnew``'s polynomial,

        r^2 Q^L(r) = r^(L+2) sum_i qfcoef(i, L) r^(2(i-1)).

    The tabulated ``PP_QIJ`` is the *same function for every L* outside
    ``rinner``; only the inner part is L-dependent, which is exactly why the
    coefficients exist.
    """
    ls = [projector.l for projector in projectors]
    nbeta = len(projectors)
    nqf = int(node.attrib.get("nqf", "0") or 0)

    entries = []
    mesh = None
    for child in node:
        if not child.tag.startswith("PP_QIJ.") and child.tag != "PP_QIJ":
            continue
        values = _numbers(child)
        mesh = values.size if mesh is None else mesh
        entries.append((
            int(child.attrib["first_index"]) - 1,
            int(child.attrib["second_index"]) - 1,
            values,
        ))
    if not entries:
        raise ValueError(f"{path}: PP_AUGMENTATION has q_with_l='F' but no PP_QIJ sections")

    nqlc = max(nqlc, 2 * max(ls) + 1)
    qfuncl = np.zeros((nbeta, nbeta, nqlc, mesh))

    rinner = np.zeros(nqlc)
    qfcoef = None
    if nqf > 0:
        rinner_node = node.find("PP_RINNER")
        qfcoef_node = node.find("PP_QFCOEF")
        if rinner_node is None or qfcoef_node is None:
            raise ValueError(
                f"{path}: PP_AUGMENTATION declares nqf={nqf} but has no "
                "PP_QFCOEF/PP_RINNER to go with it"
            )
        values = _numbers(rinner_node)
        rinner[: values.size] = values
        # Fortran order: qfcoef(nqf, nqlc, nbeta, nbeta), first index fastest.
        qfcoef = _numbers(qfcoef_node).reshape(
            (nqf, nqlc, nbeta, nbeta), order="F"
        )

    for nb, mb, values in entries:
        l1, l2 = ls[nb], ls[mb]
        for l in range(abs(l1 - l2), l1 + l2 + 1, 2):
            if l >= nqlc:
                continue
            column = values.copy()
            if qfcoef is not None and rinner[l] > 0.0:
                inside = r[:mesh] < rinner[l]
                # ``setqfnew`` with n = 2: the tabulated quantity is r^2 Q(r).
                powers = np.ones(mesh)
                polynomial = np.full(mesh, qfcoef[0, l, nb, mb])
                for i in range(1, nqf):
                    powers = powers * r[:mesh] ** 2
                    polynomial = polynomial + qfcoef[i, l, nb, mb] * powers
                column = np.where(inside, polynomial * r[:mesh] ** (l + 2), column)
            qfuncl[nb, mb, l] = column
            qfuncl[mb, nb, l] = column

    return qfuncl, nqlc


def _read_paw(root: ET.Element, nbeta: int) -> PawData | None:
    """Parse ``PP_PAW`` and ``PP_FULL_WFC`` -- the one-centre data."""
    section = root.find("PP_PAW")
    full = root.find("PP_FULL_WFC")
    if section is None or full is None:
        return None

    def _wavefunctions(name: str) -> np.ndarray | None:
        """The ``<name>.i`` series, in index order.

        The tag is matched **exactly** up to its ``.i`` suffix rather than by
        prefix. A fully-relativistic PAW file carries ``PP_AEWFC_REL`` beside
        ``PP_AEWFC`` -- the small component of the Dirac solution -- and a prefix
        match silently returns both series interleaved, which is twice as many
        partial waves as there are projectors and every one of them attached to
        the wrong channel. That is not a parse error anywhere downstream: the
        one-centre energy simply comes out tens of Ry wrong.
        """
        nodes = sorted(
            (child for child in full if child.tag.rsplit(".", 1)[0] == name),
            key=lambda child: int(child.attrib.get("index", "0")),
        )
        if not nodes:
            return None
        return np.stack([_numbers(node) for node in nodes])

    occupations = _optional_numbers(section, "PP_OCCUPATIONS")
    augmentation = root.find("PP_NONLOCAL/PP_AUGMENTATION")
    attrib = {} if augmentation is None else augmentation.attrib

    return PawData(
        ae_wfc=_wavefunctions("PP_AEWFC"),
        ps_wfc=_wavefunctions("PP_PSWFC"),
        ae_wfc_rel=_wavefunctions("PP_AEWFC_REL"),
        ae_vloc=_numbers(_require(section, "PP_AE_VLOC")),
        ae_rho_core=_optional_numbers(section, "PP_AE_NLCC"),
        occupations=np.zeros(nbeta) if occupations is None else occupations,
        core_energy=float(section.attrib.get("core_energy", "0") or 0.0),
        augmentation_shape=attrib.get("shape", "").strip(),
        cutoff_index=int(attrib.get("cutoff_r_index", "0") or 0),
    )
