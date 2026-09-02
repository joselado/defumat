"""Which orbitals carry a Hubbard U, and with what parameters.

The whole of DFT+U rests on a choice of *localised subspace*: a set of atomic
orbitals per atom, on which an occupation matrix ``ns`` is measured and a
correction is applied. This module is the host-side setup that fixes that
choice -- it does what ``PW/src/ldaU.f90`` (``init_hubbard``),
``PW/src/hubbard.f90`` (``determine_hubbard_occ``) and
``PW/src/offset_atom_wfc.f90`` do between them, and nothing numerical.

Three things it has to get right, all of which QE spells out and none of which
is guessable:

* **The manifold is named, not derived.** The ``HUBBARD`` card says ``Fe1-3d``,
  and the orbital that matches is the one whose UPF ``label`` is ``3D`` -- not
  "the d orbital", because a dataset can carry two of them, and not "the last
  one", because the order in the file is the generator's. A requested manifold
  the file does not have is an error, not a fallback.
* **The reference occupation comes from the file** (``determine_hubbard_occ``):
  the ``occupation`` attribute of that same orbital, summed over both ``j``
  shells if the dataset is fully relativistic. ``hubbard_occ`` in the input
  overrides it, which is what the ``lda+U.in`` benchmark does -- it asks for 6
  where ``Fe.pz-nd-rrkjus.UPF`` says 7.
* **Two different offsets.** ``atomwfc_offset`` is where the manifold starts in
  the *full* list of atomic orbitals (QE's ``oatwfc``), which is the list the
  orthogonalisation runs over; ``offset`` is where it starts in the list of
  Hubbard orbitals only (QE's ``offsetU``), which is what ``wfcU`` and the
  occupation matrix are indexed by. Both count only orbitals with non-negative
  occupation, exactly as :func:`defumat.pseudo.atomic.atomic_channels` does,
  because that is the list the atomic orbitals are actually built in.

**U, J0, alpha and beta arrive in eV** (the ``HUBBARD`` card's unit) and are
stored here in Ry: the conversion happens once, at the input boundary, and
nothing downstream sees an eV.

What is *not* here, and is refused rather than approximated:
``lda_plus_u_kind = 1`` (Liechtenstein's full formulation with J, B, E2, E3),
``lda_plus_u_kind = 2`` (the intersite V), the background channels
(``Hubbard_U2``), the orbital-resolved variant, and the ``wf`` and ``pseudo``
projector types.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from defumat.pseudo.upf import Pseudopotential

__all__ = [
    "HubbardInput",
    "HubbardSpecies",
    "HubbardSetup",
    "PROJECTOR_TYPES",
    "build_hubbard_setup",
    "manifold_label",
    "parse_manifold",
]

#: The projector choices this implementation supports. ``wf`` (Wannier
#: functions from ``pmw.x``) and ``pseudo`` (the beta projectors with the
#: all-electron overlaps ``q_ae``) are QE options that are refused here.
PROJECTOR_TYPES = ("atomic", "ortho-atomic", "norm-atomic")

_SPDF = "SPDF"


def manifold_label(n: int, l: int) -> str:
    """``(3, 2) -> "3D"``: the label a UPF orbital carries, in QE's spelling.

    ``upf_utils``'s ``l_to_spdf`` returns capitals and ``capital()`` is applied
    to the file's own label before comparison, so both sides of the match are
    upper case.
    """
    if not 0 <= l < len(_SPDF):
        raise ValueError(f"angular momentum {l} has no spectroscopic letter")
    return f"{n}{_SPDF[l]}"


def normalize_label(label: str) -> str:
    """A UPF orbital label in the form comparisons are made in (QE's ``capital``)."""
    return label.strip().upper()


def parse_manifold(text: str) -> tuple[str, int, int]:
    """``"Fe1-3d" -> ("Fe1", 3, 2)``: species name, principal number, ``l``.

    The species name is the ``ATOMIC_SPECIES`` label, which is not the element
    -- the antiferromagnetic benchmarks give the two iron sublattices different
    names precisely so that they can carry different parameters.
    """
    name, _, manifold = text.rpartition("-")
    if not name or len(manifold) < 2:
        raise ValueError(
            f"malformed Hubbard manifold {text!r}: expected 'label-nl', e.g. 'Fe1-3d'"
        )
    letter = manifold[-1].upper()
    if letter not in _SPDF:
        raise ValueError(f"{text!r}: {manifold[-1]!r} is not one of s, p, d, f")
    try:
        n = int(manifold[:-1])
    except ValueError as error:
        raise ValueError(
            f"{text!r}: {manifold[:-1]!r} is not a principal quantum number"
        ) from error
    return name, n, _SPDF.index(letter)


@dataclass(frozen=True)
class HubbardInput:
    """The ``HUBBARD`` card and its namelist companions, as read from the input.

    Frozen and built out of tuples so that it can sit on :class:`System` as a
    static field: it selects code paths and array shapes, so a change to it must
    retrace, and equinox needs it hashable to know that.

    ``parameters`` entries are ``(species name, n, l, U, J0, alpha, beta)``
    **in Ry** -- the card writes eV and :mod:`defumat.io.pwin` converts, so
    this is already internal units (rule R6).
    """

    projectors: str
    parameters: tuple[tuple[str, int, int, float, float, float, float], ...]
    #: ``hubbard_occ(ityp, 1)``: the reference occupation, overriding the
    #: pseudopotential's, as ``(species name, occupation)``.
    occupations: tuple[tuple[str, float], ...] = ()
    #: ``starting_ns_eigenvalue(m, ispin, ityp)`` as
    #: ``(species index, spin, m, value)``, all zero-based.
    starting_ns: tuple[tuple[int, int, int, float], ...] = ()


@dataclass
class HubbardSpecies:
    """The Hubbard parameters of one atomic species, in Ry."""

    n: int
    l: int
    u: float = 0.0
    j0: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    #: Reference occupation of the manifold: the input's ``hubbard_occ`` if it
    #: gave one, the pseudopotential's otherwise. Only the starting ``ns`` uses
    #: it.
    occupation: float = -1.0

    @property
    def ldim(self) -> int:
        return 2 * self.l + 1

    @property
    def effective_u(self) -> float:
        """``U - J0``, QE's ``effU``: what the quadratic term actually uses."""
        return self.u - self.j0


@dataclass
class HubbardSetup:
    """Everything fixed about a DFT+U run: manifolds, parameters, offsets.

    ``atoms`` lists the atoms that carry a correction, in ascending order, and
    every array indexed by a "slot" below is indexed by position in that list
    rather than by atom number -- ``ns`` is ``(nspin, len(atoms), ldmx, ldmx)``,
    with the manifolds of different ``l`` padded up to ``ldmx``.
    """

    #: One entry per species; ``None`` for a species with no U.
    species: tuple[HubbardSpecies | None, ...]
    #: Atom indices carrying a correction.
    atoms: tuple[int, ...]
    #: ``ldim`` of each slot, and the largest of them.
    ldims: tuple[int, ...]
    ldmx: int
    #: Start of each slot's block in ``wfcU`` (QE's ``offsetU``).
    offsets: tuple[int, ...]
    #: Start of each slot's block in the full atomic-orbital list (``oatwfc``).
    atomwfc_offsets: tuple[int, ...]
    #: Total number of Hubbard projectors, QE's ``nwfcU``.
    nwfcU: int
    #: The species index of each slot.
    types: tuple[int, ...]
    projectors: str = "atomic"
    #: ``starting_ns_eigenvalue(m, ispin, ityp)`` from the input, as
    #: ``{(species, spin, m): value}`` with zero-based ``spin`` and ``m``.
    starting_ns: dict = field(default_factory=dict)

    @property
    def nslot(self) -> int:
        return len(self.atoms)

    def slot_mask(self) -> np.ndarray:
        """``(nslot, ldmx)``: which rows of a padded block are real orbitals."""
        mask = np.zeros((self.nslot, self.ldmx), dtype=bool)
        for slot, ldim in enumerate(self.ldims):
            mask[slot, :ldim] = True
        return mask

    def parameter(self, name: str) -> np.ndarray:
        """``(nslot,)`` of one parameter, gathered from each slot's species."""
        return np.array(
            [getattr(self.species[t], name) for t in self.types], dtype=float
        )

    def block_indices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Where each ``ns`` entry lands in the ``(nwfcU, nwfcU)`` operator.

        Returns ``(slot, row, column)`` index arrays over the real entries of
        every block, flattened -- the scatter that turns a padded per-atom
        occupation matrix into the block-diagonal matrix the Hamiltonian's
        separable term contracts with.
        """
        slots, rows, columns = [], [], []
        for slot, (ldim, offset) in enumerate(zip(self.ldims, self.offsets)):
            i, j = np.meshgrid(np.arange(ldim), np.arange(ldim), indexing="ij")
            slots.append(np.full(i.size, slot))
            rows.append((offset + i).ravel())
            columns.append((offset + j).ravel())
        if not slots:
            empty = np.zeros(0, dtype=int)
            return empty, empty, empty
        return (
            np.concatenate(slots), np.concatenate(rows), np.concatenate(columns)
        )

    def padded_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """The ``(row, column)`` positions inside each padded block, matching
        :meth:`block_indices` entry for entry."""
        rows, columns = [], []
        for ldim in self.ldims:
            i, j = np.meshgrid(np.arange(ldim), np.arange(ldim), indexing="ij")
            rows.append(i.ravel())
            columns.append(j.ravel())
        if not rows:
            empty = np.zeros(0, dtype=int)
            return empty, empty
        return np.concatenate(rows), np.concatenate(columns)


def _atomic_orbitals(pseudo: Pseudopotential):
    """The orbitals in the order the atomic wavefunctions are built in.

    :func:`defumat.pseudo.atomic.atomic_channels` skips negative occupations,
    as ``offset_atom_wfc`` does; the width of each kept orbital is ``2l+1``.
    """
    return [orbital for orbital in pseudo.orbitals if orbital.occupation >= 0.0]


def reference_occupation(pseudo: Pseudopotential, n: int, l: int) -> float:
    """``determine_hubbard_occ``: the file's occupation of the named manifold.

    Summed over every orbital carrying the label, because a fully-relativistic
    dataset stores the two ``j`` shells of one manifold as two orbitals and the
    Hubbard occupation is the whole shell's.
    """
    label = manifold_label(n, l)
    total, found = 0.0, False
    for orbital in pseudo.orbitals:
        if normalize_label(orbital.label) == label:
            total += float(orbital.occupation)
            found = True
    if not found:
        available = ", ".join(normalize_label(o.label) for o in pseudo.orbitals)
        raise ValueError(
            f"{pseudo.element}: the pseudopotential has no {label} orbital "
            f"(it has: {available or 'none'}); the Hubbard manifold must be one "
            "the dataset carries"
        )
    return total


def build_hubbard_setup(
    hubbard: HubbardInput | None,
    structure,
    pseudos: tuple[Pseudopotential, ...],
) -> HubbardSetup | None:
    """Resolve a :class:`HubbardInput` against the structure and the datasets.

    ``None`` comes back when there is no correction to apply -- no card, or a
    card whose parameters are all zero -- which is what makes every call site's
    "is there a U?" test a single ``is None``.
    """
    if hubbard is None:
        return None
    projectors = hubbard.projectors
    parameters = {
        name: {"n": n, "l": l, "u": u, "j0": j0, "alpha": alpha, "beta": beta}
        for name, n, l, u, j0, alpha, beta in hubbard.parameters
    }
    hubbard_occ = dict(hubbard.occupations)
    starting_ns = {
        (kind, spin, m): value for kind, spin, m, value in hubbard.starting_ns
    }
    if projectors not in PROJECTOR_TYPES:
        raise NotImplementedError(
            f"Hubbard_projectors = {projectors!r} is not implemented; "
            f"available: {', '.join(PROJECTOR_TYPES)}. QE's 'wf' reads Wannier "
            "functions from pmw.x and 'pseudo' uses the beta projectors with "
            "the all-electron overlaps q_ae, and neither has an implementation here"
        )
    names = [species.name for species in structure.species]
    unknown = set(parameters) - set(names)
    if unknown:
        raise ValueError(
            f"HUBBARD card names {sorted(unknown)}, which are not in "
            f"ATOMIC_SPECIES ({names})"
        )

    resolved: list[HubbardSpecies | None] = []
    for t, name in enumerate(names):
        entry = parameters.get(name)
        if entry is None:
            resolved.append(None)
            continue
        item = HubbardSpecies(
            n=int(entry["n"]),
            l=int(entry["l"]),
            u=float(entry.get("u", 0.0)),
            j0=float(entry.get("j0", 0.0)),
            alpha=float(entry.get("alpha", 0.0)),
            beta=float(entry.get("beta", 0.0)),
        )
        # ``init_hubbard``: a species is a Hubbard species if *any* of the
        # parameters is nonzero, not only U -- an alpha-only run is how a
        # linear-response U is measured.
        if not any((item.u, item.j0, item.alpha, item.beta)):
            resolved.append(None)
            continue
        given = hubbard_occ.get(name)
        item.occupation = (
            float(given) if given is not None
            else reference_occupation(pseudos[t], item.n, item.l)
        )
        if item.occupation <= 0.0:
            raise ValueError(
                f"{name}: the {manifold_label(item.n, item.l)} manifold has "
                "zero occupation, which QE refuses as a Hubbard manifold"
            )
        resolved.append(item)

    if all(item is None for item in resolved):
        return None

    atoms, ldims, offsets, atomwfc_offsets, types = [], [], [], [], []
    counter = wfc_counter = 0
    for atom, t in enumerate(structure.types):
        item = resolved[t]
        orbitals = _atomic_orbitals(pseudos[t])
        if item is not None:
            label = manifold_label(item.n, item.l)
            position, width = None, 0
            for orbital in orbitals:
                if normalize_label(orbital.label) == label:
                    if position is None:
                        position = width
                    break
                width += 2 * orbital.l + 1
            if position is None:
                available = ", ".join(normalize_label(o.label) for o in orbitals)
                raise ValueError(
                    f"atom {atom} ({structure.species[t].name}): the "
                    f"pseudopotential has no {label} orbital with a "
                    f"non-negative occupation (it has: {available or 'none'})"
                )
            atoms.append(atom)
            types.append(t)
            ldims.append(item.ldim)
            offsets.append(wfc_counter)
            atomwfc_offsets.append(counter + position)
            wfc_counter += item.ldim
        counter += sum(2 * orbital.l + 1 for orbital in orbitals)

    setup = HubbardSetup(
        species=tuple(resolved),
        atoms=tuple(atoms),
        ldims=tuple(ldims),
        ldmx=max(ldims),
        offsets=tuple(offsets),
        atomwfc_offsets=tuple(atomwfc_offsets),
        nwfcU=wfc_counter,
        types=tuple(types),
        projectors=projectors,
        starting_ns=starting_ns,
    )
    return setup
