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

**The kind is inferred from the card, not given.** ``read_cards.f90:3240``
selects ``lda_plus_u_kind = 1`` from the presence of any nonzero ``J``, ``B``,
``E2`` or ``E3`` and ``kind = 2`` from a ``V``, and warns that the namelist's
own ``lda_plus_u_kind`` is obsolete. The same inference is made here, and
:attr:`HubbardSetup.kind` carries the answer. ``init_hubbard``'s substitutions
for the ``kind = 1`` parameters left at zero are applied here too, because they
are what puts ``F^4/F^2`` at its physical 0.625 -- see
:mod:`defumat.hubbard.interaction`.

What is *not* here, and is refused rather than approximated:
``lda_plus_u_kind = 2`` (the intersite V), the background channels
(``Hubbard_U2``), the orbital-resolved variant, and the ``wf`` and ``pseudo``
projector types. ``alpha``, ``beta`` and ``J0`` are ``kind = 0`` parameters and
are refused under ``kind = 1``, in QE's own places: ``init_hubbard`` stops on
``Hubbard_alpha`` and ``card_hubbard`` stops on ``J`` together with ``J0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from defumat.hubbard.ftm import build_constraints
from defumat.pseudo.upf import Pseudopotential

__all__ = [
    "DOUBLE_COUNTING",
    "NORM_FLOOR",
    "SLATER_SOURCES",
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

#: Where the Slater integrals come from: Elk's ``inpdftu``, collapsed to the
#: two that are not a change of coordinates. ``"parameters"`` is ``inpdftu = 1``
#: (``U`` and ``J``, with QE's ratios); ``"yukawa"`` is ``inpdftu = 4`` and
#: ``5`` -- computed from the radial function with a Yukawa kernel, the
#: screening length either given or solved for. ``inpdftu = 2`` and ``3``, the
#: Slater and Racah *inputs*, are a linear change of coordinates on the same
#: three numbers ``(U, J, B)`` already span:
#: :func:`defumat.hubbard.interaction.racah_to_slater` performs it and is
#: tested, and no card syntax is given for it because QE's card already spends
#: the names ``E2`` and ``E3`` on its own f-shell parameters.
SLATER_SOURCES = ("parameters", "yukawa")

#: How much of the manifold has to be inside the cutoff for its Slater integrals
#: to mean anything. A bound 3d or 5d shell measures 0.90 to 0.96 inside its own
#: augmentation radius; silicon's 3p measures 0.49 at 2.5 bohr, and an integral
#: over half an orbital is not that orbital's.
NORM_FLOOR = 0.7

#: The two double countings, Elk's ``dftu = 1`` and ``dftu = 2``. The
#: interpolation between them was Elk's ``dftu = 3`` and is **not** here: 11.0.2
#: documents it in the manual and has removed it from the source ("Cleaned up
#: and removed options, September 2021", ``genvmatmt.f90``), ``readinput.f90``
#: answering ``readadu`` with "no longer used" and ``writeinfo.f90`` stopping on
#: any value but 1 or 2. There is no reference implementation to check it
#: against, which is the reason rather than the effort.
DOUBLE_COUNTING = ("fll", "amf")

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
    #: The Liechtenstein parameters as ``(species name, J, B, E2, E3)`` in Ry,
    #: for each species whose card line carries one. **A non-empty tuple is
    #: what selects ``lda_plus_u_kind = 1``**, which is QE's own rule; the
    #: entries QE does not accept for a given ``l`` are refused at the card.
    full: tuple[tuple[str, float, float, float, float], ...] = ()
    #: ``hubbard_double_counting``: ``"fll"`` (QE's only one, and Elk's
    #: ``dftu = 1``) or ``"amf"`` (Elk's ``dftu = 2``). This code's own input
    #: variable -- ``pw.x`` has no counterpart.
    double_counting: str = "fll"
    #: ``hubbard_slater``: ``"parameters"`` (the default -- ``U`` and ``J`` fix
    #: the Slater integrals by fixed ratios) or ``"yukawa"``, which computes
    #: them from the manifold's own radial function. Elk's ``inpdftu``; ``pw.x``
    #: has no counterpart.
    slater: str = "parameters"
    #: ``(species name, lambda)`` in inverse bohr -- Elk's ``inpdftu = 4``, a
    #: given screening length. A species with ``hubbard_slater = 'yukawa'`` and
    #: no entry here takes ``inpdftu = 5`` instead: its ``U`` is what is given
    #: and ``lambda`` is solved for.
    lambdas: tuple[tuple[str, float], ...] = ()
    #: ``hubbard_radial_cutoff`` in bohr, overriding the dataset's own
    #: augmentation radius. Only the ``yukawa`` route reads it.
    radial_cutoff: float | None = None
    #: The ``TENSOR_MOMENTS`` card: ``(species, n, l, k, p, r, t, value)``, Elk's
    #: ``tm3fix``. This code's own card -- ``pw.x`` has neither the constraint
    #: nor the quantity.
    tensor_moments: tuple = ()
    #: ``tensor_moment_penalty``: the strength ``lambda`` of the quadratic
    #: penalty, in Ry per unit squared.
    tensor_moment_penalty: float = 1.0


@dataclass
class HubbardSpecies:
    """The Hubbard parameters of one atomic species, in Ry."""

    n: int
    l: int
    u: float = 0.0
    j0: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    #: ``Hubbard_J(1)``, the Liechtenstein exchange. Zero under ``kind = 0``,
    #: where ``j0`` is the Hund coupling instead -- the two are different
    #: parameters of different functionals and QE refuses them together.
    j: float = 0.0
    #: ``Hubbard_J(2:3)``: ``(B,)`` for a ``d`` shell, ``(E2, E3)`` for an
    #: ``f`` shell, ``()`` otherwise, **after** ``init_hubbard``'s substitution
    #: for a value left at zero.
    racah: tuple = ()
    #: ``F[0:7]`` in Ry when they were *computed* from the radial function
    #: rather than parameterised (:mod:`defumat.hubbard.yukawa`); ``None``
    #: otherwise. When it is set it overrides ``u``, ``j`` and ``racah``, which
    #: are then reported back from it rather than being inputs.
    computed_slater: tuple | None = None
    #: How the Slater integrals were obtained, for reporting: ``None`` for the
    #: parameterised route, otherwise ``(radial function kind, cutoff in bohr,
    #: norm inside it, screening length)``.
    provenance: tuple | None = None
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

    @property
    def slater(self):
        """``F[0:7]`` in Ry -- the Liechtenstein functional's radial integrals.

        Computed from the manifold's radial function when
        ``hubbard_slater = 'yukawa'`` asked for that, parameterised from ``U``
        and ``J`` otherwise.
        """
        import numpy as _np

        from defumat.hubbard.interaction import slater_integrals

        if self.computed_slater is not None:
            return _np.asarray(self.computed_slater, dtype=float)
        return slater_integrals(self.l, self.u, self.j, self.racah)


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
    #: ``lda_plus_u_kind``: 0 for the simplified functional, 1 for the full
    #: (Liechtenstein) one. Inferred from the card, never given.
    kind: int = 0
    #: Whether the states are two-component spinors. It doubles the number of
    #: *projector columns* per manifold -- ``2 (2l+1)``, spin slowest, which is
    #: ``atomic_wfc_nc``'s own order -- and makes the occupation matrix a 2x2
    #: matrix in spin space. It does **not** change ``ldims``, which stays the
    #: number of ``m`` values.
    noncolin: bool = False
    #: ``"fll"`` or ``"amf"`` -- Elk's ``dftu = 1`` and ``dftu = 2``. The full
    #: functional only; the simplified one *is* the fully-localised limit.
    double_counting: str = "fll"
    #: The resolved ``TENSOR_MOMENTS`` constraint, or ``None``.
    constraints: object = None
    #: ``starting_ns_eigenvalue(m, ispin, ityp)`` from the input, as
    #: ``{(species, spin, m): value}`` with zero-based ``spin`` and ``m``.
    starting_ns: dict = field(default_factory=dict)

    @property
    def nslot(self) -> int:
        return len(self.atoms)

    @property
    def npol(self) -> int:
        """Spinor components of a *projector*: 2 for a spinor run, 1 otherwise."""
        return 2 if self.noncolin else 1

    def column_map(self) -> np.ndarray:
        """``(nslot, npol, ldmx)``: which ``wfcU`` column each orbital is.

        ``offsetU(na) + m + ldim (is - 1)`` of ``new_ns_nc``, with the padding
        of a manifold shorter than ``ldmx`` pointing at the slot's first column
        -- :meth:`slot_mask` is what zeroes it, exactly as it does for the
        scalar map this generalises.
        """
        columns = np.zeros((self.nslot, self.npol, self.ldmx), dtype=int)
        for slot, (offset, ldim) in enumerate(zip(self.offsets, self.ldims)):
            for spin in range(self.npol):
                columns[slot, spin, :ldim] = np.arange(ldim) + offset + spin * ldim
        return columns

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

    def block_indices(self) -> tuple[np.ndarray, ...]:
        """Where each ``ns`` entry lands in the ``(nwfcU, nwfcU)`` operator.

        Returns ``(spin, slot, row, column)`` index arrays over the real entries
        of every block, flattened -- the scatter that turns a padded per-atom
        occupation matrix into the block-diagonal matrix the Hamiltonian's
        separable term contracts with. ``spin`` is the *leading axis of* ``ns``:
        the channel index for a collinear run, and the packed pair
        ``2 s1 + s2`` for a spinor, whose four blocks land in the four quadrants
        of one ``2 ldim`` block rather than in four separate matrices. That is
        the whole of what makes a spinor Hubbard term one operator instead of
        two.
        """
        spins, slots, rows, columns = [], [], [], []
        npol = self.npol
        for slot, (ldim, offset) in enumerate(zip(self.ldims, self.offsets)):
            i, j = np.meshgrid(np.arange(ldim), np.arange(ldim), indexing="ij")
            for s1 in range(npol):
                for s2 in range(npol):
                    spins.append(np.full(i.size, s1 * npol + s2))
                    slots.append(np.full(i.size, slot))
                    rows.append((offset + s1 * ldim + i).ravel())
                    columns.append((offset + s2 * ldim + j).ravel())
        if not slots:
            empty = np.zeros(0, dtype=int)
            return empty, empty, empty, empty
        return (
            np.concatenate(spins), np.concatenate(slots),
            np.concatenate(rows), np.concatenate(columns),
        )

    def padded_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """The ``(row, column)`` positions inside each padded block, matching
        :meth:`block_indices` entry for entry."""
        rows, columns = [], []
        for ldim in self.ldims:
            i, j = np.meshgrid(np.arange(ldim), np.arange(ldim), indexing="ij")
            for _ in range(self.npol**2):
                rows.append(i.ravel())
                columns.append(j.ravel())
        if not rows:
            empty = np.zeros(0, dtype=int)
            return empty, empty
        return np.concatenate(rows), np.concatenate(columns)


def _atomic_orbitals(pseudo: Pseudopotential, noncolin: bool = False):
    """The orbitals in the order the atomic wavefunctions are built in.

    :func:`defumat.pseudo.atomic.atomic_channels` skips negative occupations,
    as ``offset_atom_wfc`` does; the width of each kept orbital is ``2l+1``.

    **For a spinor run on a fully-relativistic dataset the two ``j`` channels of
    a shell are one channel, not two.** ``atomic_wfc_so_mag`` returns
    immediately for ``j = l - 1/2`` and builds the ``j``-averaged radial
    function under the other one, so the orbital list this offset arithmetic
    runs over is *shorter* than the scalar list. Doubling the scalar offsets
    instead happens to be right when the manifold is the first shell with
    ``l > 0`` and is wrong after that -- a ``3d`` manifold behind a ``3p`` one
    lands 6 columns past where it belongs, silently.
    """
    kept = [orbital for orbital in pseudo.orbitals if orbital.occupation >= 0.0]
    if not noncolin:
        return kept
    return [
        orbital for orbital in kept
        if orbital.l == 0 or getattr(orbital, "j", None) is None
        or abs(orbital.j - orbital.l + 0.5) >= 1.0e-4
    ]


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
    noncolin: bool = False,
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
    liechtenstein = {name: rest for name, *rest in getattr(hubbard, "full", ())}
    lambdas = dict(getattr(hubbard, "lambdas", ()))
    slater_source = getattr(hubbard, "slater", "parameters")
    if slater_source not in SLATER_SOURCES:
        raise ValueError(
            f"hubbard_slater = {slater_source!r}; expected one of "
            f"{', '.join(SLATER_SOURCES)}"
        )
    # Computed Slater integrals are the whole interaction matrix, so they select
    # the full functional the way an explicit J does.
    kind = 1 if (liechtenstein or slater_source == "yukawa") else 0
    double_counting = getattr(hubbard, "double_counting", "fll")
    if double_counting not in DOUBLE_COUNTING:
        raise ValueError(
            f"hubbard_double_counting = {double_counting!r}; expected one of "
            f"{', '.join(DOUBLE_COUNTING)}"
        )
    if double_counting == "amf" and kind == 0:
        # The simplified functional *is* the fully-localised limit written out
        # -- there is no ``vee`` in it to feed a shifted matrix to. Elk reaches
        # AMF only through the full one, and so does this.
        raise NotImplementedError(
            "hubbard_double_counting = 'amf' needs the full (Liechtenstein) "
            "functional, which the HUBBARD card selects with a J: the "
            "simplified functional has no interaction matrix to apply the "
            "mean-field shift to, being the fully-localised limit itself"
        )
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
        if kind == 1 and slater_source == "yukawa":
            item = _yukawa_species(
                name, item, pseudos[t], lambdas.get(name),
                getattr(hubbard, "radial_cutoff", None),
            )
        elif kind == 1:
            item = _liechtenstein_species(name, item, liechtenstein.get(name))
        # ``init_hubbard``: a species is a Hubbard species if *any* of the
        # parameters is nonzero, not only U -- an alpha-only run is how a
        # linear-response U is measured.
        if not any((item.u, item.j0, item.alpha, item.beta, item.j)):
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
    width_scale = 2 if noncolin else 1
    for atom, t in enumerate(structure.types):
        item = resolved[t]
        orbitals = _atomic_orbitals(pseudos[t], noncolin)
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
            # ``atomic_wfc_nc`` emits two columns per orbital -- the same
            # spatial function with a pure up and a pure down spin -- keeping
            # each channel's ``2l+1`` contiguous and putting the two spins one
            # after the other. So every offset into the atomic-orbital list
            # doubles, and each Hubbard slot occupies ``2 (2l+1)`` columns.
            atomwfc_offsets.append(width_scale * (counter + position))
            wfc_counter += width_scale * item.ldim
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
        kind=kind,
        noncolin=bool(noncolin),
        double_counting=double_counting,
        starting_ns=starting_ns,
    )
    setup.constraints = build_constraints(
        setup,
        getattr(hubbard, "tensor_moments", ()),
        getattr(hubbard, "tensor_moment_penalty", 1.0),
        names,
    )
    return setup


def _yukawa_species(name, item, pseudo, lam, cutoff):
    """Elk's ``inpdftu = 4`` and ``5``: the Slater integrals from the orbital.

    With a ``lambda`` given this is ``inpdftu = 4``; without one it is
    ``inpdftu = 5``, where the card's ``U`` is what is fixed and ``lambda`` is
    solved for. Either way the whole ``F^k`` set comes out, so ``J`` and the
    Racah parameters are *results* here and are written back onto the species
    so that a run reports what it actually used.
    """
    from dataclasses import replace

    from defumat.hubbard.interaction import exchange_from_slater
    from defumat.hubbard.yukawa import (
        manifold_radial,
        screening_length,
        slater_set,
    )

    if item.j0 or item.alpha or item.beta:
        raise NotImplementedError(
            f"{name}: J0, ALPHA and BETA belong to the simplified functional; "
            "hubbard_slater = 'yukawa' computes the full interaction matrix"
        )
    radial = manifold_radial(pseudo, item.n, item.l, cutoff)
    # **Only PAW carries the object this integral is over**, and falling back to
    # the pseudo orbital is the silent-wrong this refuses by name: a
    # norm-conserving ``chi`` is normalised and gives an ``F^0`` far too small
    # (silicon's 3p, 8.8 eV), and an *ultrasoft* one is not normalised at all --
    # iron's 3d has ``int (r chi)^2 = 0.41``, the rest of the norm living in the
    # augmentation charge, and its ``F^0`` comes out at 2.1 eV where the answer
    # is above twenty. Both converge an SCF and report success.
    if radial.kind != "all-electron":
        raise NotImplementedError(
            f"{name}: hubbard_slater = 'yukawa' computes the Slater integrals "
            "from the manifold's all-electron partial wave, which only a PAW "
            f"dataset carries; this one has only the pseudo-orbital chi "
            f"(norm {radial.norm:.4f}), whose Coulomb integrals are smaller "
            "than the answer by a factor of two or more"
        )
    if radial.norm < NORM_FLOOR:
        raise ValueError(
            f"{name}: the {manifold_label(item.n, item.l)} partial wave has "
            f"norm {radial.norm:.4f} inside {radial.cutoff:.2f} bohr, below the "
            f"floor of {NORM_FLOOR} -- the manifold is not bound inside the "
            "augmentation radius, so its Slater integrals are an integral over "
            "part of an orbital. Set hubbard_radial_cutoff if a larger radius "
            "is meant"
        )
    if lam is None:
        if item.u <= 0.0:
            raise ValueError(
                f"{name}: hubbard_slater = 'yukawa' with no LAMBDA solves for "
                "the screening length that reproduces a given U, so U must be "
                "positive"
            )
        lam = screening_length(radial, item.u)
    f = slater_set(radial, item.l, lam)
    return replace(
        item,
        u=float(f[0]),
        j=float(exchange_from_slater(item.l, f)),
        racah=(),
        computed_slater=tuple(float(x) for x in f),
        provenance=(radial.kind, radial.cutoff, radial.norm, float(lam)),
    )


def _liechtenstein_species(name, item, given):
    """``init_hubbard``'s ``lda_plus_u_kind = 1`` branch, for one species.

    Three things happen there and each of them changes the answer:
    ``Hubbard_alpha`` is refused outright; a species given only a ``J`` has its
    ``U`` set to ``1e-14`` rather than being dropped, so that ``is_hubbard``
    stays true and the manifold is still corrected; and a Racah parameter left
    at zero is replaced by its multiple of ``J``, which is what puts
    ``F^4/F^2`` at 0.625 instead of 1.8 (:mod:`defumat.hubbard.interaction`).
    """
    from dataclasses import replace

    from defumat.hubbard.interaction import default_racah

    if item.alpha != 0.0:
        raise NotImplementedError(
            f"{name}: ALPHA together with J selects the full DFT+U functional "
            "with a linear term, which QE refuses too ('full DFT+U does not "
            "support Hubbard_alpha calculation', init_hubbard)"
        )
    if item.j0 != 0.0 or item.beta != 0.0:
        raise NotImplementedError(
            f"{name}: J0 and BETA are parameters of the simplified functional "
            "(lda_plus_u_kind = 0) and J of the full one; QE refuses them "
            "together ('Hund J is not compatible with Hund J0', card_hubbard)"
        )
    j, b, e2, e3 = (given if given is not None else (0.0, 0.0, 0.0, 0.0))
    if item.l < 2 and (b or e2 or e3):
        raise ValueError(
            f"{name}: a {manifold_label(item.n, item.l)} manifold takes only U "
            "and J; B, E2 and E3 belong to d and f shells"
        )
    if item.l == 2 and (e2 or e3):
        raise ValueError(
            f"{name}: E2 and E3 belong to an f shell; a d shell's third Slater "
            "parameter is B"
        )
    if item.l == 3 and b:
        raise ValueError(
            f"{name}: B belongs to a d shell; an f shell's are E2 and E3"
        )
    given_racah = {2: (b,), 3: (e2, e3)}.get(item.l, ())
    # ``IF (Hubbard_U(nt) == 0.0_dp) Hubbard_U(nt) = 1.d-14``
    u = item.u if item.u != 0.0 else 1.0e-14
    return replace(item, u=u, j=j, racah=default_racah(item.l, j, given_racah))
