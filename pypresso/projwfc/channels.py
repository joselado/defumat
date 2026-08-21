"""What the ``natomwfc`` projection columns *are*: atom, shell, ``l`` and ``m``.

``PP/src/projections_mod.f90``'s ``fill_nlmchi``. The projection itself is a
matrix of numbers; everything a reader wants from a projected density of states
-- "the ``d`` weight on iron 2", the file name ``pdos_atm#1(Si)_wfc#2(p)``, the
``pz``/``px``/``py`` columns of a Löwdin charge -- comes from this label table
instead, and it has to agree channel for channel with the order
:func:`pypresso.pseudo.atomic.atomic_wavefunctions` builds its columns in.

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

from pypresso.pseudo.upf import Pseudopotential
from pypresso.system.structure import Structure

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

    @property
    def l_label(self) -> str:
        return L_LABELS[self.l]

    @property
    def m_label(self) -> str:
        """``"pz"``, ``"dxy"``, ``"s"`` -- what ``print_lowdin`` prints."""
        return f"{L_LABELS[self.l]}{M_LABELS[self.l][self.m]}"

    @property
    def shell(self) -> str:
        """``"Si 1 3S"``-ish: the atom and shell this column belongs to."""
        return f"{self.species}{self.atom + 1} {self.label or self.l_label}"

    def __str__(self) -> str:
        return f"#{self.index + 1} {self.species}{self.atom + 1} {self.m_label}"


def projection_channels(
    pseudos: tuple[Pseudopotential, ...], structure: Structure
) -> tuple[AtomicChannel, ...]:
    """The label of every atomic-orbital column, in the order they are built.

    ``fill_nlmchi``, minus the ``lspinorb``/``noncolin`` branches: a spinor
    projection doubles the columns and resolves them by ``j`` and ``m_j``, and
    :mod:`pypresso.projwfc.projections` refuses that regime by name rather than
    silently labelling twice as many columns as it has.
    """
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
