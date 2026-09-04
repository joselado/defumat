"""What the ``natomwfc`` projection columns *are*: atom, shell, ``l`` and ``m``.

``PP/src/projections_mod.f90``'s ``fill_nlmchi``. The projection itself is a
matrix of numbers; everything a reader wants from a projected density of states
-- "the ``d`` weight on iron 2", the file name ``pdos_atm#1(Si)_wfc#2(p)``, the
``pz``/``px``/``py`` columns of a Löwdin charge -- comes from this label table
instead, and it has to agree channel for channel with the order
:func:`defumat.pseudo.atomic.atomic_wavefunctions` builds its columns in.

Two things are inherited from that function rather than decided here, and both
are QE's:

* **an orbital with negative occupation is skipped**, so a dataset's
  ``natomwfc`` is not its number of ``PP_CHI`` entries;
* the ``m`` ordering inside a shell is ``ylmr2``'s, which for ``l = 1`` is
  ``(z, x, y)`` and not ``(x, y, z)``. ``print_lowdin``'s labels say so and this
  module repeats them, because a table headed ``px`` that holds ``pz`` is wrong
  in a way no total ever shows.

The ``wfc #`` a file name carries is the orbital's index in the pseudopotential
file counting from one **including the skipped ones** -- ``fill_nlmchi`` uses
its loop variable, not a counter of what it kept.
"""

from __future__ import annotations

from dataclasses import dataclass

from defumat.pseudo.spinorbit import spinor
from defumat.pseudo.upf import Pseudopotential
from defumat.system.structure import Structure

__all__ = [
    "AtomicChannel",
    "projection_channels",
    "L_LABELS",
    "M_LABELS",
    "channel_table",
]

#: ``print_lowdin``'s ``l_label``.
L_LABELS = ("s", "p", "d", "f")

#: ``print_lowdin``'s ``lm_label_global_frame``: the name of each ``m`` within a
#: shell, in ``ylmr2``'s order. The ``l = 0`` row is empty in the Fortran (a
#: single ``s`` needs no suffix) and is spelled out here.
M_LABELS = (
    ("",),
    ("z", "x", "y"),
    ("z2", "xz", "yz", "x2-y2", "xy"),
    ("z3", "xz2", "yz2", "zx2-zy2", "xyz", "x3-3xy2", "3yx2-y3"),
)


@dataclass(frozen=True)
class AtomicChannel:
    """One column of the projection: ``nlmchi(nwfc)``, spelled out.

    ``atom`` and ``m`` are 0-based here, where the Fortran counts from one;
    :attr:`wfc` is left 1-based because it is what appears in a file name.
    """

    index: int  # position among the natomwfc columns
    atom: int  # 0-based atom index
    species: str  # the ATOMIC_SPECIES label, which is not always the element
    wfc: int  # QE's nlmchi%n: the orbital's 1-based index in the UPF file
    l: int
    m: int  # 0-based within the shell, in ylmr2's order
    label: str = ""  # the UPF's own els, e.g. "3S"
    #: ``nlmchi%jj``: the total angular momentum, on a spinor column of a
    #: fully-relativistic dataset. ``None`` on every scalar column.
    j: float | None = None
    #: ``compute_mj``: the projection of ``j``. ``None`` unless ``j`` is set.
    mj: float | None = None
    #: ``+-1/2``: which spin the column carries, on a noncollinear run
    #: *without* spin-orbit coupling, where the two are still good labels.
    s_z: float | None = None

    @property
    def l_label(self) -> str:
        return L_LABELS[self.l]

    @property
    def spinor(self) -> bool:
        """Whether this column is one component of a two-component orbital."""
        return self.j is not None or self.s_z is not None

    @property
    def m_label(self) -> str:
        """``"pz"``, ``"dxy"``, ``"s"`` -- what ``print_lowdin`` prints.

        A ``j``-resolved column has no such name: it is a combination of every
        ``m`` of the shell, so it is labelled ``p_j1.5 m_j=-0.5`` instead. A
        noncollinear column without spin-orbit coupling keeps the harmonic's
        name and gains an arrow.
        """
        if self.j is not None:
            return f"{L_LABELS[self.l]}_j{self.j:.1f} m_j={self.mj:+.1f}"
        name = f"{L_LABELS[self.l]}{M_LABELS[self.l][self.m]}"
        if self.s_z is not None:
            return name + ("(up)" if self.s_z > 0 else "(dn)")
        return name

    @property
    def shell(self) -> str:
        """``"Si 1 3S"``-ish: the atom and shell this column belongs to.

        A ``j``-resolved column carries ``j`` in its shell name, because
        ``partialdos_nc`` writes one file per ``(atom, wfc, l, j)`` -- the two
        ``j`` of a shell are different files, named ``..._wfc#n(p_j0.5)``.
        """
        name = f"{self.species}{self.atom + 1} {self.label or self.l_label}"
        return f"{name} j={self.j:.1f}" if self.j is not None else name

    def __str__(self) -> str:
        return f"#{self.index + 1} {self.species}{self.atom + 1} {self.m_label}"


def _compute_mj(j: float, l: int, m: int) -> float:
    """``compute_mj``: the ``m_j`` of the ``m``-th spin-angle function."""
    if abs(j - l - 0.5) < 1.0e-4:
        return m + 0.5
    if abs(j - l + 0.5) < 1.0e-4:
        return m - 0.5
    raise ValueError(f"j = {j} is not compatible with l = {l}")


def projection_channels(
    pseudos: tuple[Pseudopotential, ...],
    structure: Structure,
    noncolin: bool = False,
    lspinorb: bool = False,
) -> tuple[AtomicChannel, ...]:
    """The label of every atomic-orbital column, in the order they are built.

    ``fill_nlmchi``. The three branches follow the Fortran's, and the order has
    to agree column for column with
    :func:`defumat.pseudo.atomic.spinor_orbital_blocks`, which builds the
    orbitals themselves -- a label table that disagrees with the basis is a
    projected density of states with the right total and the wrong decomposition.
    """
    if noncolin:
        return _spinor_projection_channels(pseudos, structure, lspinorb)
    channels: list[AtomicChannel] = []
    for atom, species in enumerate(structure.types):
        pseudo = pseudos[species]
        counters = [1, 2, 3, 4]  # fill_nlmchi's nn: the guessed principal number
        for orbital_index, orbital in enumerate(pseudo.orbitals, start=1):
            if orbital.occupation < 0.0:
                continue
            label = (orbital.label or "").strip()
            if not label or label.upper() == "XN":
                # The file did not name the shell; QE invents one from a per-l
                # counter, so a dataset with two s channels gets 1S and 2S.
                label = f"{counters[orbital.l]}{L_LABELS[orbital.l].upper()}"
                counters[orbital.l] += 1
            for m in range(2 * orbital.l + 1):
                channels.append(
                    AtomicChannel(
                        index=len(channels),
                        atom=atom,
                        species=structure.species[species].name,
                        wfc=orbital_index,
                        l=orbital.l,
                        m=m,
                        label=label,
                    )
                )
    return tuple(channels)


def channel_table(channels: tuple[AtomicChannel, ...]) -> str:
    """``print_proj``'s "Atomic states used for projection" block."""
    lines = ["Atomic states used for projection:"]
    lines += [
        f"     state #{channel.index + 1:4d}: atom {channel.atom + 1:3d} "
        f"({channel.species:<3}), wfc {channel.wfc:2d} "
        f"(l={channel.l} m={channel.m + 1:2d})"
        for channel in channels
    ]
    return "\n".join(lines)


def _spinor_projection_channels(
    pseudos: tuple[Pseudopotential, ...], structure: Structure, lspinorb: bool
) -> tuple[AtomicChannel, ...]:
    """``fill_nlmchi``'s ``noncolin`` branches.

    **The ``wfc`` number is not the same counter in the two spin-orbit
    branches**, and it reaches the reader through a file name. For a
    fully-relativistic dataset it is the orbital's index in the UPF file, so the
    two ``j`` of one shell share it; for a scalar dataset under ``lspinorb`` it
    counts the *synthesised* ``j`` shells, so they do not. That is ``n`` against
    ``n2`` in the Fortran and it is easy to miss.
    """
    channels: list[AtomicChannel] = []

    def emit(**kwargs) -> None:
        channels.append(AtomicChannel(index=len(channels), **kwargs))

    for atom, species in enumerate(structure.types):
        pseudo = pseudos[species]
        # The same refusal :func:`~defumat.pseudo.atomic.spinor_orbital_blocks`
        # makes, and it has to be made here too rather than left to the orbital
        # builder: the two are reached by *different* branches of
        # ``build_atomic_projectors``, so without it a relativistic dataset at
        # ``lspinorb = .false.`` gets ``2 (2l+1)`` labels per PP_CHI entry from
        # here and the j-averaged ``atomic_wfc_so_mag`` columns from there --
        # 22 labels against 12 columns on platinum.
        if (any(getattr(o, "j", None) is not None
                for o in pseudo.orbitals if o.occupation >= 0.0)
                and not lspinorb):
            raise NotImplementedError(
                "a fully-relativistic dataset with lspinorb = .false. is not "
                "implemented for the projection: QE dispatches the orbitals on "
                "has_so and their labels on lspinorb, so it builds j-resolved "
                "columns and calls them up/down ones. Run with lspinorb = "
                ".true., which is what a relativistic dataset is for"
            )
        counters = [1, 2, 3, 4]
        name = structure.species[species].name
        synthesised = 0  # fill_nlmchi's n2
        for orbital_index, orbital in enumerate(pseudo.orbitals, start=1):
            if orbital.occupation < 0.0:
                continue
            label = (orbital.label or "").strip()
            if not label or label.upper() == "XN":
                label = f"{counters[orbital.l]}{L_LABELS[orbital.l].upper()}"
                counters[orbital.l] += 1
            l = orbital.l
            common = dict(atom=atom, species=name, l=l, label=label)

            if not lspinorb:
                # ``atomic_wfc_nc``: every m up, then every m down.
                for spin, s_z in enumerate((0.5, -0.5)):
                    for m in range(2 * l + 1):
                        emit(wfc=orbital_index, m=m, s_z=s_z, **common)
                continue

            j_of_orbital = getattr(orbital, "j", None)
            if j_of_orbital is not None:
                shells = [(orbital_index, j_of_orbital)]
            else:
                shells = []
                for n1 in (l, l + 1):
                    j = n1 - 0.5
                    if j > 0.0:
                        synthesised += 1
                        shells.append((synthesised, j))

            for wfc, j in shells:
                ind = 0
                for m in range(-l - 1, l + 1):
                    if (abs(spinor(l, j, m, 0)) <= 1.0e-8
                            and abs(spinor(l, j, m, 1)) <= 1.0e-8):
                        continue
                    emit(wfc=wfc, m=ind, j=j, mj=_compute_mj(j, l, m), **common)
                    ind += 1
    return tuple(channels)
