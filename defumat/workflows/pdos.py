"""``projwfc.x``: project the states on atomic orbitals and resolve the DOS.

``PP/src/projwfc.f90`` and ``PP/src/partialdos.f90``. The sequence ``pw.x`` +
``projwfc.x`` is, in this order:

1. an SCF (or an NSCF on a denser grid) leaves a set of wavefunctions;
2. every band at every k-point is projected onto the Löwdin-orthogonalised
   pseudo-atomic orbitals of the crystal (:mod:`defumat.projwfc.projections`);
3. those projections weight the same Brillouin-zone integration a plain density
   of states is, and the same weights integrated against the occupations are
   the Löwdin charges.

Step 3 is here and it is bookkeeping plus one extra weight:

    D_p(E) = sum_k w_k sum_b delta(E - e_kb) |<phi_p|S|psi_kb>|^2

is the *same* integration :mod:`defumat.workflows.dos` performs, with a
per-band weight in front of the delta -- so it goes through the same registry
rather than growing a second implementation. Every scheme takes an optional
``projections`` argument, both families implement it, and ``sum_p D_p = D``
holds to round-off whenever the projections sum to one.

**Nothing is diagonalised in step 2.** ``projwfc.x`` reads the wavefunctions a
``pw.x`` run wrote and projects those, and this follows it: given an
:class:`~defumat.scf.driver.SCFResult` the projection is of the states the SCF
converged, on the k-points it converged them at. That is also the only route
open to a PAW dataset, because a fixed-density re-diagonalisation needs a
``becsum`` that is not carried across (see
:func:`defumat.workflows.nscf.fixed_density_states`) -- and it costs nothing,
since re-solving the same Hamiltonian at the same k-points returns the same
states. A **denser grid** (``grid=``) is the usual thing to want for a density
of states, and that route is an NSCF run: the density is frozen, the bands are
re-solved on the finer grid, and *those* states are projected.

The **spilling parameter** ``1 - sum n / nelec`` (Sanchez-Portal et al., Solid
State Commun. **95**, 685 (1995)) is how much of the occupied subspace the
atomic basis fails to span, and it is the number that says how much of the sum
rule a pseudo-atomic basis can be expected to satisfy at all. Silicon's is
0.008, so the projected density of states adds up to the total to about a
percent and no better -- by construction, not by error.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from defumat.projwfc.channels import L_LABELS, M_LABELS, AtomicChannel, projection_channels
from defumat.projwfc.projections import atomic_projections
from defumat.pseudo.upf import Pseudopotential
from defumat.scf.driver import Calculation
from defumat.scf.tetrahedra import build_tetrahedra, tetrahedron_kind
from defumat.system.builder import System
from defumat.system.kpoints import KPoints
from defumat.units import RY_TO_EV
from defumat.workflows.dos import (
    DEFAULT_DELTA_E,
    DensityOfStates,
    compute_dos,
    default_scheme,
    energy_grid,
    get_dos_scheme,
    is_tetrahedron_scheme,
)
from defumat.workflows.nscf import NSCFResult, denser_grid, fixed_density_states

__all__ = [
    "ProjectedDOS",
    "LowdinCharges",
    "compute_pdos",
    "lowdin_charges",
    "partial_energy_grid",
    "project_states",
    "run_pdos",
]


def partial_energy_grid(
    eigenvalues: np.ndarray,
    emin: float | None = None,
    emax: float | None = None,
    delta_e: float = DEFAULT_DELTA_E,
    degauss: float = 0.0,
) -> np.ndarray:
    """``partialdos``'s energy grid, in Ry.

    ``dos.f90`` and ``partialdos.f90`` compute the same ``ne`` and then loop
    over it differently: ``dos.f90`` writes ``ndos`` points ``1..ndos`` and
    ``partialdos`` writes ``ne + 1`` points ``0..ne``. So a projected density of
    states carries exactly one energy point more than the total one on the same
    settings, and a comparison against ``filpdos`` that assumes otherwise is off
    by a row at one end.
    """
    grid = energy_grid(eigenvalues, emin, emax, delta_e, degauss)
    return np.append(grid, grid[-1] + delta_e)


# --------------------------------------------------------------------------
# Löwdin charges
# --------------------------------------------------------------------------


@dataclass
class LowdinCharges:
    """``print_lowdin``'s table: charge per atom, per ``l``, and per ``m``.

    ``charges`` is ``(nat, lmax+1)`` and ``charges_lm`` ``(nat, lmax+1,
    2 lmax+1)``, with a leading spin axis when ``nspin = 2`` -- squeezed away
    otherwise, the convention every result object here follows. Entries for an
    ``(atom, l)`` the crystal does not have are zero, as QE's arrays are.
    """

    charges: np.ndarray
    charges_lm: np.ndarray
    spilling: float
    nelec: float
    nspin: int = 1

    @property
    def charges_by_spin(self) -> np.ndarray:
        return self.charges if self.nspin == 2 else self.charges[None]

    @property
    def charges_lm_by_spin(self) -> np.ndarray:
        return self.charges_lm if self.nspin == 2 else self.charges_lm[None]

    @property
    def total(self) -> np.ndarray:
        """``(nat,)``: the total charge on each atom, both channels summed."""
        return self.charges_by_spin.sum(axis=(0, 2))

    @property
    def polarization(self) -> np.ndarray:
        """``(nat,)``: up minus down per atom. Zero for an unpolarized run."""
        by_spin = self.charges_by_spin.sum(axis=2)
        return by_spin[0] - by_spin[1] if self.nspin == 2 else np.zeros_like(by_spin[0])

    def format(self, species: tuple[str, ...] = ()) -> str:
        """``print_lowdin``'s block, close enough to diff against by eye."""
        lines = ["Lowdin Charges:", ""]
        by_spin = self.charges_by_spin
        lm_by_spin = self.charges_lm_by_spin
        for atom in range(by_spin.shape[1]):
            name = f" ({species[atom]})" if atom < len(species) else ""
            lines.append(
                f"     Atom #{atom + 1:4d}{name}: total charge = "
                f"{by_spin[:, atom].sum():8.4f}"
            )
            for spin in range(by_spin.shape[0]):
                prefix = (
                    "       " if self.nspin == 1
                    else ("       spin up    " if spin == 0 else "       spin down  ")
                )
                for l in range(by_spin.shape[2]):
                    if abs(by_spin[spin, atom, l]) < 1.0e-8:
                        continue
                    detail = ", ".join(
                        f"{L_LABELS[l]}{M_LABELS[l][m]}="
                        f"{lm_by_spin[spin, atom, l, m]:8.4f}"
                        for m in range(2 * l + 1)
                    ) if l else ""
                    lines.append(
                        f"{prefix}{L_LABELS[l]} = {by_spin[spin, atom, l]:8.4f}"
                        + (f", {detail}" if detail else "")
                    )
        lines.append(f"     Spilling Parameter: {self.spilling:8.4f}")
        return "\n".join(lines)


def lowdin_charges(
    projections: np.ndarray,
    occupations: np.ndarray,
    channels: tuple[AtomicChannel, ...],
    nat: int,
    nelec: float,
) -> LowdinCharges:
    """``print_proj``'s charge estimate.

    ``projections`` is ``(nspin, nk, nproj, nbnd)`` and ``occupations`` the
    matching ``(nspin, nk, nbnd)`` weights -- occupations *times* k-point
    weights, which is what QE's ``wg`` is and what an SCF or NSCF result
    carries. Using ``f_kb`` without ``w_k`` gives a number wrong by the size of
    the irreducible wedge that still looks like a charge.
    """
    projections = np.asarray(projections)
    occupations = np.asarray(occupations)
    if projections.ndim == 3:
        projections = projections[None]
    if occupations.ndim == 2:
        occupations = occupations[None]

    nspin = projections.shape[0]
    lmax = max((channel.l for channel in channels), default=0)
    charges = np.zeros((nspin, nat, lmax + 1))
    charges_lm = np.zeros((nspin, nat, lmax + 1, 2 * lmax + 1))

    # (nspin, nproj): the occupied weight in each column, summed over k and band.
    weight = np.einsum(
        "skpb,skb->sp", projections, occupations[..., : projections.shape[-1]]
    )
    for channel in channels:
        charges[:, channel.atom, channel.l] += weight[:, channel.index]
        charges_lm[:, channel.atom, channel.l, channel.m] += weight[:, channel.index]

    return LowdinCharges(
        charges=charges if nspin == 2 else charges[0],
        charges_lm=charges_lm if nspin == 2 else charges_lm[0],
        spilling=1.0 - float(charges.sum()) / nelec,
        nelec=float(nelec),
        nspin=nspin,
    )


# --------------------------------------------------------------------------
# The result
# --------------------------------------------------------------------------


@dataclass
class ProjectedDOS:
    """``D_p(E)`` per atomic orbital, with the total beside it.

    ``pdos`` is ``(nproj, nE)`` for an unpolarized run and ``(2, nproj, nE)``
    for LSDA -- the spin axis squeezed away when there is one channel, as
    everywhere else, with :attr:`pdos_by_spin` always carrying it. Units are
    states/Ry per cell, with the spin degeneracy already in, so summing over a
    complete set of channels would give back :attr:`total`'s ``dos``; the atomic
    basis is not complete, and the difference is the spilling.
    """

    energies: np.ndarray  # (nE,), Ry
    pdos: np.ndarray  # (nproj, nE) or (2, nproj, nE), states/Ry
    integrated: np.ndarray  # same shape, states
    channels: tuple[AtomicChannel, ...]
    #: The *unprojected* density of states on the same grid and the same scheme
    #: -- ``pdos_tot``'s ``dos(E)`` column, which is what the sum rule compares
    #: against.
    total: DensityOfStates
    scheme: str
    nspin: int = 1
    fermi_energy: float | None = None  # Ry
    charges: LowdinCharges | None = None
    projectors: str = "ortho-atomic"

    @property
    def energies_ev(self) -> np.ndarray:
        return self.energies * RY_TO_EV

    @property
    def pdos_by_spin(self) -> np.ndarray:
        """``(nspin, nproj, nE)`` whatever ``nspin`` is."""
        return self.pdos if self.nspin == 2 else self.pdos[None]

    @property
    def integrated_by_spin(self) -> np.ndarray:
        return self.integrated if self.nspin == 2 else self.integrated[None]

    @property
    def summed(self) -> np.ndarray:
        """``pdostot``: every channel added up, the shape of a plain DOS."""
        return (
            self.pdos_by_spin.sum(axis=1) if self.nspin == 2 else self.pdos.sum(axis=0)
        )

    def select(
        self,
        atom: int | None = None,
        l: int | str | None = None,
        species: str | None = None,
        m: int | None = None,
        wfc: int | None = None,
    ) -> np.ndarray:
        """The channels matching every constraint given, summed.

        ``(nE,)``, or ``(2, nE)`` for a polarized run. ``l`` accepts either the
        number or its letter, so ``select(species="Si", l="p")`` reads the way
        the question is asked.
        """
        if isinstance(l, str):
            l = L_LABELS.index(l.lower())
        picked = [
            channel.index
            for channel in self.channels
            if (atom is None or channel.atom == atom)
            and (l is None or channel.l == l)
            and (species is None or channel.species == species)
            and (m is None or channel.m == m)
            and (wfc is None or channel.wfc == wfc)
        ]
        chosen = self.pdos_by_spin[:, picked].sum(axis=1)
        return chosen if self.nspin == 2 else chosen[0]

    def shells(self) -> tuple[tuple[AtomicChannel, ...], ...]:
        """The channels grouped by ``(atom, wfc)`` -- one group per output file.

        ``partialdos`` writes one file per shell, with an ``ldos`` column (the
        shell's own sum) and one ``pdos`` column per ``m``.
        """
        groups: dict[tuple[int, int], list[AtomicChannel]] = {}
        for channel in self.channels:
            groups.setdefault((channel.atom, channel.wfc), []).append(channel)
        return tuple(tuple(group) for group in groups.values())

    def plot(self, ax=None, ev: bool = True, zero: bool = True, by: str = "shell",
             total: bool = True, **kwargs):
        """Draw the projected density of states, and return the axes.

        ``by`` groups the curves: ``"shell"`` gives one per ``(atom, l)`` --
        what ``projwfc.x`` writes one file per -- ``"species"`` sums the
        equivalent atoms together, and ``"l"`` keeps only the angular momentum.
        ``total`` overlays the unprojected DOS, which is the sum rule made
        visible: where the coloured curves fall short of it, that weight is the
        spilling.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        reference = self.fermi_energy if (zero and self.fermi_energy is not None) else 0.0
        energies = (self.energies - reference) * (RY_TO_EV if ev else 1.0)
        scale = 1.0 / RY_TO_EV if ev else 1.0

        curves: dict[str, np.ndarray] = {}
        summed = self.pdos_by_spin.sum(axis=0)
        for group in self.shells():
            channel = group[0]
            # ``shell`` is already "Si1 3S" -- the atom label is in it, so
            # only the coarser groupings have to build a key of their own.
            if by == "species":
                key = f"{channel.species} {channel.label or channel.l_label}"
            elif by == "l":
                key = channel.l_label
            else:
                key = channel.shell
            weight = sum(summed[c.index] for c in group)
            curves[key] = curves.get(key, 0.0) + weight

        if total:
            ax.plot(energies, self.total.total_dos * scale, color="0.4", lw=1.0,
                    label="total")
        for label, curve in curves.items():
            ax.plot(energies, curve * scale, label=label, **kwargs)
        if reference:
            ax.axvline(0.0, color="0.6", lw=0.8, ls="--")
        ax.set_ylim(bottom=0.0)
        unit = "eV" if ev else "Ry"
        ax.set_xlabel(f"E - E$_F$ ({unit})" if reference else f"Energy ({unit})")
        ax.set_ylabel("PDOS (states/eV)" if ev else "PDOS (states/Ry)")
        ax.legend()
        return ax

    def at(self, energy: float, **selection) -> float:
        """The selected channels' ``D(E)`` at one energy, in states/Ry."""
        values = self.select(**selection)
        if self.nspin == 2:
            values = values.sum(axis=0)
        return float(np.interp(energy, self.energies, values))


# --------------------------------------------------------------------------
# The integration
# --------------------------------------------------------------------------


def compute_pdos(
    eigenvalues,
    weights,
    projections,
    energies,
    scheme: str,
    channels: tuple[AtomicChannel, ...],
    degauss: float | None = None,
    tetrahedra=None,
    fermi_energy: float | None = None,
    chunk: int | None = None,
    occupations=None,
    nelec: float | None = None,
    nat: int | None = None,
    projectors: str = "ortho-atomic",
) -> ProjectedDOS:
    """Integrate eigenvalues and projections into a projected density of states.

    ``eigenvalues`` is ``(nk, nbnd)`` or ``(nspin, nk, nbnd)`` and
    ``projections`` the matching ``(nk, nproj, nbnd)`` or
    ``(nspin, nk, nproj, nbnd)`` -- the layout
    :func:`defumat.projwfc.projections.atomic_projections` returns, which is
    QE's ``proj(nwfc, ibnd, ik)`` with the axes in this project's order.

    The channels are looped over here and nowhere else, exactly as
    :func:`defumat.workflows.dos.compute_dos` does it.
    """
    energies = jnp.asarray(energies)
    eigenvalues = jnp.asarray(eigenvalues)
    weights = jnp.asarray(weights)
    projections = jnp.asarray(projections)
    scheme_function = get_dos_scheme(scheme)

    bands = eigenvalues if eigenvalues.ndim == 3 else eigenvalues[None]
    projected = projections if projections.ndim == 4 else projections[None]
    options = {} if chunk is None else {"chunk": chunk}

    results = [
        scheme_function(
            channel_bands,
            weights,
            energies,
            degauss=degauss,
            tetrahedra=tetrahedra,
            # ``(nk, nbnd, nproj)``: the schemes take a projection as a weight
            # on a band, so the channel axis is trailing there.
            projections=jnp.transpose(channel_projections, (0, 2, 1)),
            **options,
        )
        for channel_bands, channel_projections in zip(bands, projected)
    ]
    nspin = len(results)
    pdos = np.stack([np.asarray(d).T for d, _ in results])
    integrated = np.stack([np.asarray(n).T for _, n in results])

    total = compute_dos(
        eigenvalues,
        weights,
        energies,
        scheme,
        degauss=degauss,
        tetrahedra=tetrahedra,
        fermi_energy=fermi_energy,
        **options,
    )

    charges = None
    if occupations is not None and nelec:
        charges = lowdin_charges(
            np.asarray(projections),
            np.asarray(occupations),
            channels,
            nat if nat is not None else 1 + max(c.atom for c in channels),
            nelec,
        )

    return ProjectedDOS(
        energies=np.asarray(energies),
        pdos=pdos if nspin == 2 else pdos[0],
        integrated=integrated if nspin == 2 else integrated[0],
        channels=tuple(channels),
        total=total,
        scheme=scheme.lower(),
        nspin=nspin,
        fermi_energy=fermi_energy,
        charges=charges,
        projectors=projectors,
    )


# --------------------------------------------------------------------------
# The workflow
# --------------------------------------------------------------------------


def _projection_tetrahedra(scheme: str, kpoints: KPoints, symmetries, cell):
    """The tetrahedra a *projected* density of states integrates over.

    ``do_projwfc`` promotes ``tetra_type = 0`` to 1 before it builds anything --
    it runs the **linear** method for a PDOS whatever the SCF occupied its bands
    with, because Bloechl's curvature correction is not the derivative of an
    occupation and there is nothing to weight per corner with. That substitution
    has to happen here rather than in the integration, because the two families
    do not cut a microcell into the same tetrahedra at all (`PLAN.md` P8, trap
    2): Bloechl fixes one body diagonal, and the linear and optimised methods
    pick the shortest of the four.
    """
    if kpoints.grid is None:
        raise ValueError(
            "the tetrahedron method needs an automatic k-point grid "
            "(K_POINTS automatic); an explicit list carries no tetrahedra"
        )
    kind = tetrahedron_kind(scheme)
    kind = "linear" if kind == "bloechl" else kind
    return build_tetrahedra(
        kind,
        kpoints.grid,
        kpoints.shift or (0, 0, 0),
        symmetries.rotation_array(),
        np.asarray(cell.bg_2pi_alat),
        precision=kpoints.precision,
    )


def project_states(
    calculation: Calculation,
    wavefunctions,
    eigenvalues,
    occupations,
    scheme: str | None = None,
    degauss: float | None = None,
    emin: float | None = None,
    emax: float | None = None,
    delta_e: float = DEFAULT_DELTA_E,
    fermi_energy: float | None = None,
    projectors: str = "ortho-atomic",
    symmetrize: bool = True,
    chunk: int | None = None,
) -> ProjectedDOS:
    """Project a set of states and integrate them, with nothing re-solved.

    ``wavefunctions`` is ``(nspin, nk, nbnd, npwx)`` on ``calculation``'s own
    k-points, ``eigenvalues`` and ``occupations`` the matching
    ``(nspin, nk, nbnd)`` -- ``occupations`` being ``wg``, occupations times
    k-point weights.
    """
    system = calculation.system
    channels = projection_channels(calculation.pseudos, system.structure)
    projections = atomic_projections(
        calculation, wavefunctions, kind=projectors, symmetrize=symmetrize
    )

    scheme, degauss = default_scheme(system, scheme, degauss, delta_e)
    tetrahedra = None
    if is_tetrahedron_scheme(scheme):
        tetrahedra = _projection_tetrahedra(
            scheme, system.kpoints, calculation.symmetries, system.cell
        )
        degauss = 0.0

    energies = partial_energy_grid(
        np.asarray(eigenvalues), emin, emax, delta_e, degauss or 0.0
    )
    return compute_pdos(
        eigenvalues,
        system.kpoints.weights,
        projections,
        energies,
        scheme,
        channels,
        degauss=degauss,
        tetrahedra=tetrahedra,
        fermi_energy=fermi_energy,
        chunk=chunk,
        occupations=occupations,
        nelec=calculation.nelec,
        nat=system.structure.nat,
        projectors=projectors,
    )


def run_pdos(
    system: System,
    pseudos: tuple[Pseudopotential, ...],
    result,
    grid: tuple[int, int, int] | None = None,
    shift: tuple[int, int, int] | None = None,
    kpoints: KPoints | None = None,
    nbnd: int | None = None,
    scheme: str | None = None,
    degauss: float | None = None,
    emin: float | None = None,
    emax: float | None = None,
    delta_e: float = DEFAULT_DELTA_E,
    projectors: str = "ortho-atomic",
    symmetrize: bool = True,
    conv_thr: float = 1.0e-6,
    chunk: int | None = None,
    k_batch: int | None | str = "default",
) -> tuple[ProjectedDOS, NSCFResult]:
    """A converged SCF in, ``(ProjectedDOS, NSCFResult)`` out.

    ``result`` is the :class:`~defumat.scf.driver.SCFResult` of that run -- the
    density alone is not enough, because what is projected is the
    *wavefunctions*.

    Without ``grid`` or ``kpoints`` the SCF's own states are projected on the
    SCF's own k-points, which is what ``projwfc.x`` does when it is pointed at a
    ``pw.x`` ``outdir``. With either, the bands are re-solved there first.
    """
    if kpoints is None and grid is not None:
        kpoints = denser_grid(system, grid, shift)

    if kpoints is None:
        calculation = Calculation(system, pseudos, k_batch=k_batch)
        eigenvalues = jnp.asarray(result.eigenvalues_by_spin)
        wavefunctions = result.wavefunctions
        occupations = np.asarray(
            result.occupations if result.nspin == 2 else result.occupations[None]
        )
        levels = {
            "fermi_energy": result.fermi_energy,
            "homo": result.homo,
            "lumo": result.lumo,
            "fermi_energy_up": result.fermi_energy_up,
            "fermi_energy_down": result.fermi_energy_down,
        }
    else:
        # **The whole of the mixed state, not one member of it.** This passed
        # nine positional arguments into a signature that takes more, so ``tau``
        # and ``becsum`` were dropped and a PAW or meta-GGA projected DOS on a
        # denser grid stopped on ``fixed_density_states``' own refusal -- with
        # no argument through which to satisfy it. That is ``PLAN.md`` P38's
        # defect (``run_dos`` never forwarded ``becsum`` or ``ns``) surviving in
        # the sibling workflow P38 did not touch. Every one of these is sitting
        # on ``result`` and always was.
        calculation, system, eigenvalues, wavefunctions = fixed_density_states(
            system, pseudos, result.density, kpoints, nbnd, conv_thr, k_batch,
            ns=getattr(result, "ns", None),
            tau=getattr(result, "tau", None),
            becsum=tuple(getattr(result, "becsum", ()) or ()),
            field=getattr(result, "magnetic_field", None),
            field_scale=getattr(result, "field_scale", None),
        )
        eigenvalues = jnp.asarray(eigenvalues)
        wg, levels = calculation.occupations(eigenvalues)
        occupations = np.asarray(wg)

    fermi_energy = levels.get("fermi_energy")
    if fermi_energy is None:
        fermi_energy = levels.get("homo")

    pdos = project_states(
        calculation,
        wavefunctions,
        eigenvalues,
        occupations,
        scheme=scheme,
        degauss=degauss,
        emin=emin,
        emax=emax,
        delta_e=delta_e,
        fermi_energy=fermi_energy,
        projectors=projectors,
        symmetrize=symmetrize,
        chunk=chunk,
    )
    nspin = calculation.nspin
    states = NSCFResult(
        kpoints=calculation.system.kpoints,
        eigenvalues=np.asarray(eigenvalues if nspin == 2 else eigenvalues[0]),
        occupations=occupations if nspin == 2 else occupations[0],
        fermi_energy=levels.get("fermi_energy"),
        homo=levels.get("homo"),
        lumo=levels.get("lumo"),
        nspin=nspin,
        fermi_energy_up=levels.get("fermi_energy_up"),
        fermi_energy_down=levels.get("fermi_energy_down"),
    )
    return pdos, states
