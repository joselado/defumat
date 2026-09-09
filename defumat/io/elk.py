"""Reading a converged Elk ground state, so a defumat SCF can start from it.

Elk is an all-electron LAPW code and defumat is a pseudopotential plane-wave
one, so the two converged densities are **different objects**: Elk's carries the
core electrons and the nuclear cusp, defumat's is the smooth pseudo valence
density normalised to ``nelec``. Nothing here changes that. What is transferred
is a **seed** -- a starting guess that is closer to the answer than a
superposition of isolated atomic charges -- and defumat's own SCF still runs on
top of it. Using an Elk density *fixed*, with no SCF above it, is refused by
name in :meth:`ElkState.density_on`.

``STATE.OUT`` is unformatted sequential Fortran binary written by Elk's
``writestate.f90``, and it is **not self-contained**: it holds no lattice
vectors, no atomic positions and no species names. The unit that can be read is
therefore an Elk *run directory* -- ``GEOMETRY.OUT`` for the geometry and
``STATE.OUT`` for the fields.

The layout, and the conventions that are easy to get wrong, are all from Elk
11.0.2 (``vendor/elk/src`` in the user's elkpy checkout; ``modmain.f90:1276``
carries the version):

* **One Fortran record holds two arrays.** ``write(100) rhomt, rhoir`` is a
  *single* record with both concatenated, and likewise for every magnetic pair.
  A reader that expects one array per record desyncs at the very first density
  record, and the symptom looks like a byte-order problem rather than a framing
  one.
* ``rhomt`` stores :math:`f_{lm}(r)`, **not** :math:`r^2 f_{lm}(r)`. The
  :math:`r^2` lives in the integration weight (``genrmesh.f90:66-68`` builds
  ``wr2mt`` as spline weights and *then* multiplies by :math:`r^2`).
* The muffin-tin block is written **unpacked**, ``(lmmaxo, nrmtmax, natmtot)``,
  with the ``lm > lmmaxi`` entries zeroed for ``ir <= nri``
  (``writestate.f90:58-60`` calls ``rfmtpack(.false., ...)``). Neither ``nrmti``
  nor ``lmmaxi`` is written, so the inner region needs no handling on the way
  in -- but see :mod:`defumat.io.elk_density` for where it does matter.
* **Padding past ``nrmt(is)`` is uninitialised buffer, not zeros.** Every slice
  here is ``[:, :nrmt(is), ias]``. Invisible with one species; silently wrong
  with two.
* ``rmt(is) = rsp(nrmt(is), is)`` exactly (``genrmesh.f90:56-59``), and that is
  the post-``autormt`` radius, which the species file's own ``rmt`` is not.
* ``ias`` runs species-outer, atom-inner (``init0.f90:78-91``).
* ``rhoir`` is raw: not multiplied by ``cfunir``, not by ``omega``, in
  e/bohr^3.
* **Elk is Hartree and defumat is Rydberg**, and only the *potentials* and
  fields carry the factor of two -- ``vclmt``, ``vxcmt``, ``vsmt``, ``efermi``,
  ``dlefe``. Densities are e/bohr^3 in both and take no factor at all. The
  conversion happens here, at the ``io`` boundary, as the project rule requires.
* ``tshift`` defaults to ``.true.``, which moves the origin onto the inversion
  centre. ``STATE.OUT`` and ``GEOMETRY.OUT`` are in the shifted frame and
  ``elk.in``'s ``atoms`` block is not, which is why the positions are read from
  ``GEOMETRY.OUT``.

**This is a gfortran build's file**: 4-byte record markers and a 4-byte logical.
An Intel-built ``STATE.OUT`` is not generally byte-compatible and nothing here
has been tested against one.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

__all__ = ["ElkState", "read_elk_state", "read_elk_geometry", "ElkGeometry"]

#: Hartree -> Rydberg. Potentials, ``efermi`` and every magnetic field carry it;
#: densities and magnetizations do not.
HARTREE_TO_RY = 2.0


@dataclasses.dataclass(frozen=True)
class ElkGeometry:
    """What ``GEOMETRY.OUT`` carries: the cell and where the atoms are.

    ``avec`` is stored the way defumat stores a cell -- **row** ``i`` is the
    ``i``-th lattice vector in bohr -- which is the transpose of Elk's own
    ``avec(:,i)`` column layout. ``positions`` are in lattice (fractional)
    coordinates, ``natoms`` is per species and ``species`` are the species file
    names, in the order that fixes the ``ias`` index.
    """

    avec: np.ndarray            # (3, 3), row i is lattice vector i, bohr
    species: tuple[str, ...]
    natoms: tuple[int, ...]
    positions: np.ndarray       # (natmtot, 3), lattice coordinates
    magnetic_fields: np.ndarray  # (natmtot, 3), Elk's per-atom bfcmt

    @property
    def natmtot(self) -> int:
        return int(sum(self.natoms))

    @property
    def omega(self) -> float:
        """The cell volume in bohr^3."""
        return abs(float(np.linalg.det(self.avec)))

    def species_of(self) -> np.ndarray:
        """``idxis``: the species index of each ``ias``, species-outer."""
        return np.concatenate([np.full(n, i) for i, n in enumerate(self.natoms)])


def read_elk_geometry(path) -> ElkGeometry:
    """Parse ``GEOMETRY.OUT``.

    The format is Elk's own ``writegeom.f90``: labelled blocks ``scale``,
    ``scale1..3``, ``avec``, ``atoms``. The scale factors multiply the lattice
    vectors -- ``scale`` all three and ``scale1..3`` one each -- and Elk applies
    them in ``readinput.f90`` before anything else sees ``avec``, so they are
    applied here too.
    """
    path = Path(path)
    tokens = _blocks(path.read_text())

    scale = float(tokens.get("scale", ["1.0"])[0])
    scales = np.array(
        [float(tokens.get(f"scale{i}", ["1.0"])[0]) for i in (1, 2, 3)]
    )
    if "avec" not in tokens:
        raise ValueError(f"{path} has no 'avec' block; is it a GEOMETRY.OUT?")
    avec = np.array([float(x) for x in tokens["avec"][:9]]).reshape(3, 3)
    avec = avec * (scale * scales)[:, None]

    atoms = tokens.get("atoms")
    if atoms is None:
        raise ValueError(f"{path} has no 'atoms' block; is it a GEOMETRY.OUT?")
    cursor = 0
    nspecies = int(float(atoms[cursor])); cursor += 1
    species, natoms, positions, fields = [], [], [], []
    for _ in range(nspecies):
        species.append(atoms[cursor].strip().strip("'").strip('"')); cursor += 1
        n = int(float(atoms[cursor])); cursor += 1
        natoms.append(n)
        for _ in range(n):
            row = [float(x) for x in atoms[cursor:cursor + 6]]; cursor += 6
            positions.append(row[:3])
            fields.append(row[3:6])

    return ElkGeometry(
        avec=avec,
        species=tuple(species),
        natoms=tuple(natoms),
        positions=np.array(positions, dtype=float).reshape(-1, 3),
        magnetic_fields=np.array(fields, dtype=float).reshape(-1, 3),
    )


def _blocks(text: str) -> dict[str, list[str]]:
    """Elk's block format: a bare keyword line, then its values until the next.

    Everything after a ``:`` on a line is Elk's own annotation
    (``: nspecies``, ``: natoms; atpos, bfcmt below``) and is dropped, as is a
    ``!`` comment.
    """
    out: dict[str, list[str]] = {}
    key = None
    for line in text.splitlines():
        line = line.split("!")[0].split(":")[0].strip()
        if not line:
            continue
        first = line.split()[0]
        if len(line.split()) == 1 and not _numeric(first) and not first.startswith("'"):
            key = first
            out.setdefault(key, [])
            continue
        if key is not None:
            out[key].extend(line.split())
    return out


def _numeric(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


@dataclasses.dataclass(frozen=True)
class ElkState:
    """A converged Elk ground state, read from an Elk run directory.

    The density is in e/bohr^3 and the potentials are in **Rydberg**, converted
    on the way in. Everything else is Elk's own, unchanged.

    Reconstructing the density at a point, integrating it, and turning it into a
    seed for a defumat SCF all live in :mod:`defumat.io.elk_density` and are
    reached through the methods here.
    """

    version: tuple[int, int, int]
    geometry: ElkGeometry
    lmmaxo: int
    nrmtmax: int
    nrcmtmax: int
    nrmt: tuple[int, ...]
    rsp: tuple[np.ndarray, ...]
    nrcmt: tuple[int, ...]
    rcmt: tuple[np.ndarray, ...]
    ngridg: tuple[int, int, int]
    ngvec: int
    ndmag: int
    nspinor: int
    fsmtype: int
    ftmtype: int
    dftu: int
    lmmaxdm: int
    xcgrad: int
    efermi: float          # Ry
    dlefe: float           # Ry
    rhomt: np.ndarray      # (lmmaxo, nrmtmax, natmtot), e/bohr^3
    rhoir: np.ndarray      # ngridg, e/bohr^3
    vclmt: np.ndarray      # Ry
    vclir: np.ndarray
    vxcmt: np.ndarray
    vxcir: np.ndarray
    vsmt: np.ndarray
    vsir: np.ndarray
    directory: Path

    # -- what the file says about itself --------------------------------------

    @property
    def natmtot(self) -> int:
        return self.geometry.natmtot

    @property
    def nspecies(self) -> int:
        return len(self.geometry.natoms)

    @property
    def rmt(self) -> np.ndarray:
        """``rmt(is) = rsp(nrmt(is), is)``, the post-``autormt`` radius."""
        return np.array([self.rsp[i][self.nrmt[i] - 1] for i in range(self.nspecies)])

    @property
    def omega(self) -> float:
        return self.geometry.omega

    @property
    def lmaxo(self) -> int:
        return int(round(np.sqrt(self.lmmaxo))) - 1

    @property
    def nri(self) -> np.ndarray:
        """``nrmti`` per species, recovered from the zeroed inner block.

        ``rfmtpack(.false., ...)`` zeroes the ``lm > lmmaxi`` entries for
        ``ir <= nri`` on the way out, and neither ``nrmti`` nor ``lmmaxi`` is
        written to the file. The zeros are exact, so ``nri`` is the number of
        leading radial points whose high-``lm`` tail vanishes -- which is what
        this returns. Where the whole sphere is spherically symmetric there is
        nothing to find and it returns ``nrmt``, which is harmless: the terms it
        would then include are zero anyway.
        """
        out = []
        species = self.geometry.species_of()
        top = self.lmaxo ** 2   # the first ``lm`` of the highest ``l`` shell
        for i, nr in enumerate(self.nrmt):
            atoms = np.nonzero(species == i)[0]
            # The highest ``l`` shell is certainly above ``lmaxi``, whatever
            # ``lmaxi`` was. Channels that vanish on the whole mesh carry no
            # information about the boundary and are skipped.
            counts = []
            for lm in range(top, self.lmmaxo):
                column = np.abs(self.rhomt[lm, :nr, atoms]).max(axis=0)
                nonzero = np.nonzero(column)[0]
                if nonzero.size:
                    counts.append(int(nonzero[0]))
            out.append(min(counts) if counts else nr)
        return np.array(out)

    # -- what it can do -------------------------------------------------------

    def evaluate_at(self, points, coordinates: str = "lattice") -> np.ndarray:
        """The density at arbitrary points, exactly as Elk's ``rfpts`` does it.

        See :func:`defumat.io.elk_density.evaluate_at`.
        """
        from defumat.io.elk_density import evaluate_at

        return evaluate_at(self, points, coordinates=coordinates)

    def muffin_tin_charges(self) -> np.ndarray:
        """``chgmt`` per atom: :math:`4\\pi Y_{00}\\int f_{00}(r) r^2 dr`."""
        from defumat.io.elk_density import muffin_tin_charges

        return muffin_tin_charges(self)

    def characteristic_function(self) -> np.ndarray:
        """Elk's ``cfunir``: the Fourier-truncated step, 0 inside, 1 outside."""
        from defumat.io.elk_density import characteristic_function

        return characteristic_function(self)

    def interstitial_charge(self) -> float:
        """``chgir``, Elk's own way: ``rhoir`` against the *truncated* step."""
        from defumat.io.elk_density import interstitial_charge

        return interstitial_charge(self)

    def density_on(self, calculation, renormalise: bool = True, report=None):
        """The density reconstructed on a defumat calculation's dense grid.

        See :func:`defumat.io.elk_density.density_on`.
        """
        from defumat.io.elk_density import density_on

        return density_on(self, calculation, renormalise=renormalise, report=report)

    # -- reading --------------------------------------------------------------

    @classmethod
    def read(cls, directory) -> "ElkState":
        """Read ``STATE.OUT`` and ``GEOMETRY.OUT`` from an Elk run directory."""
        return read_elk_state(directory)


def read_elk_state(directory) -> ElkState:
    """Read an Elk run directory: ``STATE.OUT`` beside ``GEOMETRY.OUT``.

    **What this refuses, and why.** Each of these is a whole further piece of
    physics rather than a format detail, and a seed built from a state whose
    extra content is silently dropped is a wrong answer that starts:

    * a **spin-polarized** state (``spinpol``), because a magnetization is a
      second field with its own transfer rule and its own defumat side. That
      covers spin spirals too, which are spin-polarized runs in Elk;
    * a **DFT+U** state (``dftu``) and a **fixed tensor moment** one
      (``ftmtype``), whose occupation matrices sit in trailing records that
      nothing here consumes;
    * anything written by Elk **older than 2.0.0**, which ``readstate.f90``
      refuses itself.
    """
    from scipy.io import FortranFile

    directory = Path(directory)
    state_path = directory / "STATE.OUT"
    geometry_path = directory / "GEOMETRY.OUT"
    if not state_path.exists():
        raise FileNotFoundError(
            f"no STATE.OUT in {directory}. An Elk state is a run *directory*: "
            "STATE.OUT holds no lattice vectors, no positions and no species "
            "names, so GEOMETRY.OUT has to be beside it."
        )
    if not geometry_path.exists():
        raise FileNotFoundError(
            f"no GEOMETRY.OUT in {directory}. STATE.OUT is not self-contained "
            "-- it carries neither the cell nor the atomic positions -- and "
            "elk.in is not a substitute, because Elk's default tshift moves "
            "the origin onto the inversion centre and only GEOMETRY.OUT is "
            "written in that shifted frame."
        )

    geometry = read_elk_geometry(geometry_path)

    with FortranFile(state_path, "r") as f:
        version = tuple(int(x) for x in f.read_ints(np.int32))
        if version[0] < 2:
            raise NotImplementedError(
                f"STATE.OUT from Elk {version[0]}.{version[1]}.{version[2]}: "
                "readstate.f90 refuses anything earlier than 2.0.0 and so does "
                "this reader"
            )
        spinpol = bool(f.read_record(np.int32)[0])
        nspecies = int(f.read_ints(np.int32)[0])
        if nspecies != len(geometry.natoms):
            raise ValueError(
                f"STATE.OUT has {nspecies} species and GEOMETRY.OUT has "
                f"{len(geometry.natoms)}; these are not the same run"
            )
        lmmaxo = int(f.read_ints(np.int32)[0])
        nrmtmax = int(f.read_ints(np.int32)[0])
        nrcmtmax = int(f.read_ints(np.int32)[0])

        nrmt, rsp, nrcmt, rcmt = [], [], [], []
        for i in range(nspecies):
            natoms_i = int(f.read_ints(np.int32)[0])
            if natoms_i != geometry.natoms[i]:
                raise ValueError(
                    f"species {i + 1}: STATE.OUT has {natoms_i} atoms and "
                    f"GEOMETRY.OUT has {geometry.natoms[i]}"
                )
            nrmt.append(int(f.read_ints(np.int32)[0]))
            rsp.append(f.read_reals(np.float64))
            nrcmt.append(int(f.read_ints(np.int32)[0]))
            rcmt.append(f.read_reals(np.float64))

        ngridg = tuple(int(x) for x in f.read_ints(np.int32))
        ngvec = int(f.read_ints(np.int32)[0])
        ndmag = int(f.read_ints(np.int32)[0])
        nspinor = int(f.read_ints(np.int32)[0])
        fsmtype = int(f.read_ints(np.int32)[0])
        # ``ftmtype`` only exists from version 3 (readstate.f90:107-111).
        ftmtype = int(f.read_ints(np.int32)[0]) if version[0] > 2 else 0
        dftu = int(f.read_ints(np.int32)[0])
        lmmaxdm = int(f.read_ints(np.int32)[0])
        xcgrad = (
            int(f.read_ints(np.int32)[0])
            if (version[0] > 5 or (version[0] == 5 and version[1] > 0))
            else 0
        )
        # ``efermi`` from 9.6 and ``dlefe`` from 10.7. Below that Elk falls back
        # on a separate EFERMI.OUT, which is not read here -- the seed does not
        # use either number, and a NaN says so rather than a plausible zero.
        if version[0] > 9 or (version[0] == 9 and version[1] > 5):
            efermi = float(f.read_reals(np.float64)[0])
        else:
            efermi = float("nan")
        if version[0] > 10 or (version[0] == 10 and version[1] > 6):
            dlefe = float(f.read_reals(np.float64)[0])
        else:
            dlefe = float("nan")

        if spinpol:
            raise NotImplementedError(
                "this STATE.OUT is spin-polarized (spinpol = .true.), and only "
                "an unpolarized Elk state can be transferred. A magnetization "
                "is a second field with a transfer rule of its own, and a "
                "seed that carries the charge and silently drops the moment "
                "would start a magnetic run from a non-magnetic guess. Spin "
                "spirals are spin-polarized Elk runs and are refused here too."
            )
        if dftu != 0 or ftmtype != 0:
            raise NotImplementedError(
                f"this STATE.OUT carries DFT+U (dftu = {dftu}) or a fixed "
                f"tensor moment (ftmtype = {ftmtype}). The occupation and "
                "potential matrices are trailing records that nothing here "
                "reads, and a Hubbard ground state seeded by its density alone "
                "is not the state Elk converged."
            )

        # ``natmtot`` is the sum of ``natoms(is)`` over **all** species, not
        # ``natoms(1)``: ``writestate.f90:57-62`` dimensions the muffin-tin
        # block ``(lmmaxo, nrmtmax, natmtot)``, so a reader that confuses the
        # two reads the wrong number of bytes the moment a second species
        # exists. **Untested by every fixture here -- they have one species
        # each** -- which is why the record length is asserted as an equality
        # below rather than left to be noticed.
        natmtot = geometry.natmtot
        ngtot = int(np.prod(ngridg))
        nmt = lmmaxo * nrmtmax * natmtot

        def _pair():
            """One record holding ``(muffin-tin block, interstitial block)``."""
            record = f.read_reals(np.float64)
            if record.size != nmt + ngtot:
                raise ValueError(
                    f"record of {record.size} reals where {nmt + ngtot} were "
                    "expected. Elk writes the muffin-tin and interstitial "
                    "arrays as ONE record; a reader that takes them as two "
                    "desyncs here."
                )
            mt = record[:nmt].reshape((lmmaxo, nrmtmax, natmtot), order="F")
            return mt, record[nmt:].reshape(ngridg, order="F")

        # The muffin-tin block is dimensioned to ``nrmtmax``, and for any
        # species with ``nrmt(is) < nrmtmax`` the rows past ``nrmt(is)`` are
        # **uninitialised buffer, not zeros**. Every consumer slices
        # ``[:, :nrmt(is), ias]``; nothing here integrates the padding.
        # **Untested by every fixture here -- one species each, so
        # ``nrmt(1) == nrmtmax`` identically and the bug cannot show.**
        rhomt, rhoir = _pair()
        vclmt, vclir = _pair()
        vxcmt, vxcir = _pair()
        vsmt, vsir = _pair()

    return ElkState(
        version=version,
        geometry=geometry,
        lmmaxo=lmmaxo,
        nrmtmax=nrmtmax,
        nrcmtmax=nrcmtmax,
        nrmt=tuple(nrmt),
        rsp=tuple(rsp),
        nrcmt=tuple(nrcmt),
        rcmt=tuple(rcmt),
        ngridg=ngridg,
        ngvec=ngvec,
        ndmag=ndmag,
        nspinor=nspinor,
        fsmtype=fsmtype,
        ftmtype=ftmtype,
        dftu=dftu,
        lmmaxdm=lmmaxdm,
        xcgrad=xcgrad,
        efermi=efermi * HARTREE_TO_RY,
        dlefe=dlefe * HARTREE_TO_RY,
        rhomt=rhomt,
        rhoir=rhoir,
        vclmt=vclmt * HARTREE_TO_RY,
        vclir=vclir * HARTREE_TO_RY,
        vxcmt=vxcmt * HARTREE_TO_RY,
        vxcir=vxcir * HARTREE_TO_RY,
        vsmt=vsmt * HARTREE_TO_RY,
        vsir=vsir * HARTREE_TO_RY,
        directory=directory,
    )
