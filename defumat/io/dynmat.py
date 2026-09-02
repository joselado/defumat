"""Writing QE's dynamical-matrix file, so that ``dynmat.x`` can read ours.

``PLAN.md`` P36. This exists for one reason: it makes a **reference** available
where P35 established there is none. The vendored ``ph.x``'s third-derivative
branch does not reproduce QE's own committed example and fails its own internal
check, so nothing it prints can validate a Raman tensor -- but ``dynmat.x`` is
not that code. ``LR_Modules/dynmat_sub.f90``'s ``RamanIR`` reads ``dchi_dtau``,
``zstar`` and ``eps0`` off a file and contracts them with the phonon
eigendisplacements; it solves nothing and shares nothing with the branch that
regressed. Writing the file ``ph.x`` would have written lets that arithmetic be
run on *our* tensors and compared against
:func:`~defumat.response.spectra.mode_activities`, which is a transcription
check of exactly the kind this project runs everywhere else.

**The format is taken from the reader, not from the writer.**
``dynmat_sub.f90``'s ``readmat2`` is what has to accept the file, and it is a
sequence of positional list-directed reads with four ``read`` statements of
plain text in the middle -- so what matters is the *number of records* between
the blocks, not the column layout.
``PHonon/examples/example05/reference/alas.dynG`` is the layout this
reproduces, line for line.

Three things are easy to get wrong and all three are silent -- the reader would
take the file and give different numbers:

* **the masses are in Rydberg mass units**, not amu (``readmat2`` divides by
  ``amu_ry`` on the way in);
* **``at`` goes in units of ``alat``** when ``ibrav = 0``, which is what this
  always writes so that a cell of any shape round-trips;
* **the Raman block is indexed ``(i, j, displacement, atom)``** where this
  code's :attr:`~defumat.response.nonlinear.RamanTensors.raman` is
  ``[atom, displacement, i, j]``, and it is in ``A^2`` -- ``Omega/(4 pi)`` times
  the derivative of ``eps`` -- rather than in inverse bohr.

Only writing is implemented. Reading one back is not needed by anything here and
is not written.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from defumat.units import AMU_TO_RY, BOHR_TO_ANGSTROM, FPI

__all__ = ["write_dynamical_matrix"]


def _matrix_lines(rows: np.ndarray, width: str = "24.12f") -> list[str]:
    return ["".join(f"{value:{width}}" for value in row) for row in rows]


def write_dynamical_matrix(
    path,
    cell,
    structure,
    force_constants: np.ndarray,
    epsilon: np.ndarray | None = None,
    born: np.ndarray | None = None,
    raman: np.ndarray | None = None,
    title: str = "written by defumat",
) -> Path:
    """Write a ``Gamma``-point ``fildyn`` that ``dynmat.x`` accepts.

    Args:
        cell: the :class:`~defumat.system.cell.Cell`.
        structure: the :class:`~defumat.system.structure.Structure`.
        force_constants: ``(nat, 3, nat, 3)`` in Ry/bohr^2 --
            :attr:`~defumat.response.phonon.Phonons.matrix`. Written as the
            real part of a complex matrix, which is what ``q = 0`` makes it.
        epsilon: ``(3, 3)``, the electronic dielectric tensor. Written together
            with ``born``; ``readmat2`` reads the two as one block and falls
            back to ``eps = 1``, ``Z* = 0`` when the header is missing, so
            passing one without the other is refused.
        born: ``(nat, 3, 3)`` indexed ``[atom, field, displacement]``, in ``e``.
        raman: ``(nat, 3, 3, 3)`` indexed ``[atom, displacement, i, j]``, in
            **inverse bohr** -- the raw
            :attr:`~defumat.response.nonlinear.RamanTensors.raman`. Converted
            to the ``A^2`` the file carries here, in one place.

    Returns the path written.
    """
    if (epsilon is None) != (born is None):
        raise ValueError(
            "epsilon and the Born charges are one block in this format and "
            "readmat2 reads them together: pass both or neither"
        )
    if raman is not None and epsilon is None:
        raise ValueError(
            "the Raman block sits inside the dielectric block and readmat2 "
            "only looks for it after reading epsilon and Z*"
        )

    path = Path(path)
    positions = np.asarray(structure.positions, dtype=float) / cell.alat
    at = np.asarray(cell.at, dtype=float) / cell.alat
    nat, ntyp = structure.nat, structure.ntyp
    lines = ["Dynamical matrix file", title]

    # ``ibrav = 0`` always: the file then carries the lattice vectors
    # explicitly, so a cell this code did not build from an ``ibrav`` -- a
    # relaxed one, for instance -- round-trips as readily as a cubic one.
    # ``celldm(1)`` is still ``alat``, and ``latgen`` uses it to scale ``at``.
    lines.append(f"  {ntyp:3d} {nat:4d} {0:2d} " + "".join(
        f"{value:11.7f}" for value in [cell.alat, 0.0, 0.0, 0.0, 0.0, 0.0]
    ))
    lines.append("Basis vectors")
    lines += _matrix_lines(at)

    for index, species in enumerate(structure.species, start=1):
        # **Rydberg mass units**, which is what ``readmat2`` divides out again.
        lines.append(
            f"{index:12d}  '{species.name:3s}'   {species.mass * AMU_TO_RY:20.12f}"
        )
    for atom in range(nat):
        lines.append(
            f"{atom + 1:5d}{structure.types[atom] + 1:5d}"
            + "".join(f"{value:18.10f}" for value in positions[atom])
        )

    lines += ["", "     Dynamical  Matrix in cartesian axes", "",
              "     q = (    0.000000000   0.000000000   0.000000000 ) ", ""]
    matrix = np.asarray(force_constants, dtype=float)
    for a in range(nat):
        for b in range(nat):
            lines.append(f"{a + 1:5d}{b + 1:5d}")
            for row in matrix[a, :, b, :]:
                # Real and imaginary parts interleaved, three pairs to a line.
                lines.append("".join(f"{value:12.8f}{0.0:12.8f}" for value in row))

    if epsilon is not None:
        lines += ["", "     Dielectric Tensor:", ""]
        lines += _matrix_lines(np.asarray(epsilon, dtype=float))
        lines += ["", "     Effective Charges E-U: Z_{alpha}{s,beta}", ""]
        charges = np.asarray(born, dtype=float)
        for atom in range(nat):
            lines.append(f"     atom # {atom + 1:4d}")
            lines += _matrix_lines(charges[atom])

    if raman is not None:
        # ``write_ramtns.f90``: ``Omega/(4 pi)`` times ``d(eps)/d(tau)``, with
        # bohr^2 turned into A^2. The same conversion
        # :attr:`~defumat.response.nonlinear.RamanTensors.raman_angstrom2`
        # applies, done here so that the caller hands over the raw tensor.
        scale = float(cell.volume) / FPI * BOHR_TO_ANGSTROM**2
        tensors = np.asarray(raman, dtype=float) * scale
        lines += ["", "     Raman tensor (A^2)", ""]
        for atom in range(nat):
            for cart in range(3):
                lines.append(f"     atom # {atom + 1:4d}    pol.{cart + 1:3d}")
                lines += _matrix_lines(tensors[atom, cart], width="24.12E")

    path.write_text("\n".join(lines) + "\n")
    return path
