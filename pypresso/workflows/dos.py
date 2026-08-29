"""The density of states, behind a name registry of integration schemes.

``PP/src/dos.f90``, which is a thin driver over two ways of turning a set of
eigenvalues into ``D(E)``:

* **smearing** (``PP/src/dosg.f90``) -- replace each level by a normalised
  delta of width ``degauss`` and sum over the k-points. Cheap, works on any
  k-set including one with no grid behind it, and broadens sharp structure by
  construction.
* **tetrahedra** (:mod:`pypresso.scf.tetrahedra`) -- interpolate the bands
  linearly inside the tetrahedra of the k-grid and integrate exactly. No width,
  so a gap really is empty, but it needs the uniform grid the points came from.

Both are written the same way here: only the **integrated** density of states
``N(E)`` is coded, and ``D(E)`` is ``jax.grad`` of it. That is not a shortcut --
it is what makes ``int D dE = N`` an identity rather than a test, and for the
smearing case it also guarantees the delta is exactly the derivative of the
occupation function the SCF used (see ``w0gauss`` in ``scf/occupations.py``).

A DOS is an NSCF calculation: the density converges on a coarse grid, but
resolving structure to a few tens of meV takes an order of magnitude more
k-points, so the sequence is SCF -> NSCF on a denser grid -> integrate.

**Spin.** Both schemes stay per channel -- neither knows about spin, and neither
needs to. What a polarized run changes is that there are two of them, plotted
against each other: ``dos.x`` writes ``dosup(E)`` and ``dosdw(E)`` as separate
columns and a single ``Int dos(E)`` summing both, and a magnetic material's
exchange splitting is exactly the offset between the two curves. The one thing
that is *not* per channel is the Fermi level, which the tetrahedron and smearing
occupations both find from the two channels together (see
:func:`pypresso.scf.tetrahedra.tetrahedron_occupations_spin`).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from pypresso.pseudo.upf import Pseudopotential
from pypresso.scf.occupations import SMEARING_ORDER, w0gauss, wgauss
from pypresso.scf.tetrahedra import (
    ENERGY_CHUNK,
    PROJECTED_ENERGY_CHUNK,
    TETRAHEDRON_KINDS,
    Tetrahedra,
    tetrahedra_for,
    tetrahedron_dos,
    tetrahedron_projected_dos,
)
from pypresso.system.builder import System
from pypresso.system.kpoints import KPoints
from pypresso.units import RY_TO_EV
from pypresso.workflows.nscf import denser_grid, run_nscf

__all__ = [
    "DensityOfStates",
    "default_scheme",
    "is_tetrahedron_scheme",
    "DOS_SCHEMES",
    "get_dos_scheme",
    "energy_grid",
    "compute_dos",
    "run_dos",
]

#: ``dos.x``'s default energy step, 0.01 eV, in Ry.
DEFAULT_DELTA_E = 0.01 / RY_TO_EV


@dataclass
class DensityOfStates:
    """``D(E)`` and ``N(E)`` on an energy grid, in Rydberg atomic units.

    ``dos`` is states per Ry per cell and ``integrated`` is states per cell, both
    already carrying the spin degeneracy -- so ``integrated`` reaches the number
    of valence electrons at the Fermi level, which is the sum rule worth
    checking. ``dos.x`` writes eV instead; that conversion belongs to
    :mod:`pypresso.io.output`.

    With two spin channels ``dos`` and ``integrated`` are ``(2, nE)``; with one
    the axis is squeezed away, the same convention
    :class:`~pypresso.scf.driver.SCFResult` uses for the density. The summed
    quantities are :attr:`total_dos` and :attr:`total_integrated`, and it is the
    *total* that satisfies the sum rule -- which is why ``dos.x`` prints two
    ``dos`` columns and only one ``Int dos``.
    """

    energies: np.ndarray  # (nE,), Ry
    dos: np.ndarray  # (nE,) or (2, nE), states/Ry
    integrated: np.ndarray  # same shape, states
    scheme: str
    fermi_energy: float | None = None  # Ry
    nspin: int = 1
    #: Only when ``tot_magnetization`` constrained the channels separately.
    fermi_energy_up: float | None = None
    fermi_energy_down: float | None = None

    @property
    def energies_ev(self) -> np.ndarray:
        return self.energies * RY_TO_EV

    @property
    def dos_by_spin(self) -> np.ndarray:
        """``(nspin, nE)`` whatever ``nspin`` is."""
        return self.dos if self.nspin == 2 else self.dos[None]

    @property
    def total_dos(self) -> np.ndarray:
        """``D(E)`` summed over the channels, ``(nE,)``."""
        return self.dos if self.nspin == 1 else np.sum(self.dos, axis=0)

    @property
    def total_integrated(self) -> np.ndarray:
        """``N(E)`` summed over the channels, ``(nE,)``."""
        return (
            self.integrated if self.nspin == 1 else np.sum(self.integrated, axis=0)
        )

    @property
    def dos_ev(self) -> np.ndarray:
        """States per eV, which is what a DOS is conventionally plotted in."""
        return self.dos / RY_TO_EV

    def plot(self, ax=None, ev: bool = True, zero: bool = True, **kwargs):
        """Draw the density of states, and return the axes.

        A spin-polarized DOS is drawn the way it is read: the two channels
        mirrored about zero, which is the only presentation in which the
        exchange splitting is a shape rather than two overlapping curves.
        matplotlib is imported inside the method, so it stays out of the
        dependencies of a calculation.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        reference = self.fermi_energy if (zero and self.fermi_energy is not None) else 0.0
        energies = (self.energies - reference) * (RY_TO_EV if ev else 1.0)
        scale = 1.0 / RY_TO_EV if ev else 1.0
        if self.nspin == 2:
            up, down = self.dos_by_spin
            ax.plot(energies, up * scale, label="up", **kwargs)
            ax.plot(energies, -down * scale, label="down", **kwargs)
            ax.axhline(0.0, color="0.6", lw=0.8)
            ax.legend()
        else:
            ax.plot(energies, self.total_dos * scale, **kwargs)
            ax.set_ylim(bottom=0.0)
        if reference:
            ax.axvline(0.0, color="0.6", lw=0.8, ls="--")
        unit = "eV" if ev else "Ry"
        ax.set_xlabel(f"E - E$_F$ ({unit})" if reference else f"Energy ({unit})")
        ax.set_ylabel("DOS (states/eV)" if ev else "DOS (states/Ry)")
        return ax

    def states_below(self, energy: float) -> float:
        """``N(E)`` at an arbitrary energy, interpolated from the grid.

        The total over both channels: that is the quantity the sum rule is about.
        """
        return float(np.interp(energy, self.energies, self.total_integrated))

    def at(self, energy: float) -> float:
        """``D(E)`` at an arbitrary energy, in states/Ry, summed over channels."""
        return float(np.interp(energy, self.energies, self.total_dos))


# --------------------------------------------------------------------------
# The schemes
# --------------------------------------------------------------------------


def _smearing_scheme(ngauss: int):
    """``dos_g``: a normalised delta per level, summed with the k-point weights.

    ``N(E) = sum_k w_k sum_n wgauss((E - e_nk)/degauss)`` is written down and
    ``D(E)`` is its derivative, which is ``w0gauss`` over ``degauss`` -- exactly
    ``dos_g``'s expression, but reached by differentiating rather than by
    transcribing a second formula. The intermediate is ``(nE, nk, nbnd)``, five
    orders of magnitude smaller than the tetrahedron one, so it is not chunked.
    """

    def scheme(eigenvalues, weights, energies, *, degauss=None, projections=None, **_):
        if not degauss:
            raise ValueError("a smearing density of states needs a positive degauss")
        x = (energies[:, None, None] - eigenvalues[None, :, :]) / degauss
        if projections is None:
            dos = jnp.einsum("k,ekb->e", weights, w0gauss(x, ngauss)) / degauss
            integrated = jnp.einsum("k,ekb->e", weights, wgauss(x, ngauss))
            return dos, integrated
        # ``partialdos``: the same delta, weighted by how much of each band sits
        # in each channel. QE truncates the Gaussian at 5 ``degauss`` to keep the
        # loop short; the whole grid is evaluated here, which is the same number
        # to the accuracy the tail is worth and one branch fewer.
        dos = jnp.einsum(
            "k,ekb,kbp->ep", weights, w0gauss(x, ngauss), projections
        ) / degauss
        integrated = jnp.einsum(
            "k,ekb,kbp->ep", weights, wgauss(x, ngauss), projections
        )
        return dos, integrated

    scheme.__name__ = f"smearing_dos_ngauss_{ngauss}"
    return scheme


def _tetrahedron_scheme(
    eigenvalues,
    weights,
    energies,
    *,
    tetrahedra=None,
    chunk=ENERGY_CHUNK,
    projections=None,
    **_,
):
    """``tetra_dos_t`` / ``opt_tetra_dos_t``, via :mod:`pypresso.scf.tetrahedra`."""
    if tetrahedra is None:
        raise ValueError(
            "a tetrahedron density of states needs the tetrahedra of the k-grid; "
            "pass tetrahedra=... or use run_dos, which builds them"
        )
    if projections is None:
        return tetrahedron_dos(tetrahedra, eigenvalues, weights, energies, chunk=chunk)
    return tetrahedron_projected_dos(
        tetrahedra, eigenvalues, weights, projections, energies,
        chunk=min(chunk, PROJECTED_ENERGY_CHUNK),
    )


#: Scheme name -> implementation. Every entry has the signature
#: ``scheme(eigenvalues, weights, energies, **options) -> (dos, integrated)``
#: with ``eigenvalues`` ``(nk, nbnd)`` and everything in Rydberg atomic units.
#: Adding a scheme is a registration, not a branch in the workflow (rule R4).
#:
#: Every entry also takes ``projections``, ``(nk, nbnd, nproj)``: a weight per
#: band per k-point, which turns the same integration into a *projected* density
#: of states with a trailing ``nproj`` axis on both returned arrays
#: (:mod:`pypresso.projwfc`). It is one implementation per family and not two,
#: which is what keeps ``sum_p pdos_p == dos`` exact for a complete set of
#: channels rather than approximately true.
DOS_SCHEMES = {name: _smearing_scheme(ngauss) for name, ngauss in SMEARING_ORDER.items()}
DOS_SCHEMES.update({name: _tetrahedron_scheme for name in TETRAHEDRON_KINDS})
#: ``dos.x`` spells the smearing family this way in its ``bz_sum`` variable.
DOS_SCHEMES["smearing"] = DOS_SCHEMES["gaussian"]


def get_dos_scheme(name: str):
    """Look up an integration scheme by the name an input file would use."""
    try:
        return DOS_SCHEMES[name.lower()]
    except KeyError as error:
        raise ValueError(
            f"unknown density-of-states scheme {name!r}; expected one of {sorted(DOS_SCHEMES)}"
        ) from error


def is_tetrahedron_scheme(name: str) -> bool:
    return name.lower() in TETRAHEDRON_KINDS


# --------------------------------------------------------------------------
# The energy grid
# --------------------------------------------------------------------------


def energy_grid(
    eigenvalues: np.ndarray,
    emin: float | None = None,
    emax: float | None = None,
    delta_e: float = DEFAULT_DELTA_E,
    degauss: float = 0.0,
) -> np.ndarray:
    """``dos.x``'s energy grid, in Ry.

    Its defaults are the bottom of the *lowest* band and the top of the
    *highest* one -- ``MINVAL(et(1,:))`` and ``MAXVAL(et(nbnd,:))``, not the
    extremes of the whole array, which differ whenever the bands cross -- padded
    by ``3*degauss`` when there is a smearing, since a delta of that width puts
    weight outside the band. The point count is
    ``nint((Emax - Emin)/DeltaE + 0.500001)``, whose odd constant is Fortran
    rounding half away from zero.
    """
    eigenvalues = np.asarray(eigenvalues)
    # ``...`` rather than a leading colon so that a ``(nspin, nk, nbnd)`` array
    # is spanned across *both* channels, which is what ``dos.f90`` does: its
    # ``et`` is one array of ``2 nkstot`` k-points and ``MINVAL(et(1,:))`` runs
    # over all of them.
    if emin is None:
        emin = float(eigenvalues[..., 0].min()) - (3.0 * degauss if degauss > 0.0 else 0.0)
    if emax is None:
        emax = float(eigenvalues[..., -1].max()) + (3.0 * degauss if degauss > 0.0 else 0.0)
    if delta_e <= 0.0:
        raise ValueError(f"the energy step must be positive, got {delta_e}")
    ndos = int(np.floor((emax - emin) / delta_e + 0.500001 + 0.5))
    return emin + np.arange(max(ndos, 1)) * delta_e


# --------------------------------------------------------------------------
# The calculation
# --------------------------------------------------------------------------


def compute_dos(
    eigenvalues,
    weights,
    energies,
    scheme: str,
    degauss: float | None = None,
    tetrahedra: Tetrahedra | None = None,
    fermi_energy: float | None = None,
    chunk: int = ENERGY_CHUNK,
    fermi_energy_up: float | None = None,
    fermi_energy_down: float | None = None,
) -> DensityOfStates:
    """Integrate a set of eigenvalues into a density of states.

    ``eigenvalues`` is ``(nk, nbnd)`` or ``(nspin, nk, nbnd)``; ``weights`` is
    ``(nk,)``, shared by the channels, and its sum carries the spin degeneracy
    exactly as everywhere else -- 2 unpolarized, 1 per polarized channel, which
    is what makes the two curves add up to the total without any factor written
    down here.

    The schemes themselves stay per channel. This function is the only place
    that loops over them, which is the whole of what spin costs the DOS.
    """
    energies = jnp.asarray(energies)
    eigenvalues = jnp.asarray(eigenvalues)
    weights = jnp.asarray(weights)
    scheme_function = get_dos_scheme(scheme)

    channels = eigenvalues if eigenvalues.ndim == 3 else eigenvalues[None]
    results = [
        scheme_function(
            channel,
            weights,
            energies,
            degauss=degauss,
            tetrahedra=tetrahedra,
            chunk=chunk,
        )
        for channel in channels
    ]
    nspin = len(results)
    dos = np.stack([np.asarray(d) for d, _ in results])
    integrated = np.stack([np.asarray(n) for _, n in results])

    return DensityOfStates(
        energies=np.asarray(energies),
        dos=dos if nspin == 2 else dos[0],
        integrated=integrated if nspin == 2 else integrated[0],
        scheme=scheme.lower(),
        fermi_energy=fermi_energy,
        nspin=nspin,
        fermi_energy_up=fermi_energy_up,
        fermi_energy_down=fermi_energy_down,
    )



def default_scheme(
    system: System,
    scheme: str | None,
    degauss: float | None,
    delta_e: float = DEFAULT_DELTA_E,
) -> tuple[str, float | None]:
    """Which integration scheme a run's own settings ask for, and how wide.

    ``do_projwfc`` and ``dos.f90`` agree on this ladder and it is written once
    here: the tetrahedron variant the SCF used if it used one, otherwise its
    smearing at its own ``degauss``, otherwise -- a run with fixed occupations
    carries no broadening at all -- a Gaussian one energy step wide.
    """
    if scheme is not None:
        return scheme, degauss
    if is_tetrahedron_scheme(system.occupations):
        return system.occupations, degauss
    if system.occupations == "smearing" and system.degauss > 0.0:
        return system.smearing, degauss or system.degauss
    return "gaussian", degauss or delta_e


def run_dos(
    system: System,
    pseudos: tuple[Pseudopotential, ...],
    density: jnp.ndarray,
    grid: tuple[int, int, int] | None = None,
    shift: tuple[int, int, int] | None = None,
    kpoints: KPoints | None = None,
    nbnd: int | None = None,
    scheme: str | None = None,
    degauss: float | None = None,
    emin: float | None = None,
    emax: float | None = None,
    delta_e: float = DEFAULT_DELTA_E,
    conv_thr: float = 1.0e-6,
    chunk: int = ENERGY_CHUNK,
    k_batch: int | None | str = "default",
    tau: jnp.ndarray | None = None,
    ns: jnp.ndarray | None = None,
    becsum: tuple = (),
    field=None,
    field_scale: float | None = None,
):
    """SCF density in, ``(DensityOfStates, NSCFResult)`` out.

    ``grid`` asks for a denser Monkhorst-Pack grid than the one the density was
    converged on, reduced with the same crystal symmetries; without it the
    calculation's own k-points are reused, which is rarely enough for a DOS.

    ``tau``, ``ns`` and ``becsum`` are the rest of the converged state, and they
    are carried for one reason: a *denser* grid is an NSCF run, and none of the
    three can be rebuilt from the density it is handed. ``tau`` is a property of
    the occupied states over the whole zone, ``ns`` builds the Hubbard term and
    ``becsum`` builds ``ddd_paw``. ``run_nscf`` refuses without them rather than
    computing something else, so a PAW or DFT+U density of states on a denser
    grid needs all of ``SCFResult``, not its density
    (:class:`~pypresso.calculator.Calculator` passes them for the caller).

    ``scheme`` defaults to whatever the calculation itself used to occupy its
    bands -- the tetrahedron variant if it ran with one, otherwise its smearing
    -- so the states counted here are the states the SCF counted. Failing both,
    ``dos.x``'s own fallback: a Gaussian of width ``DeltaE``.
    """
    if kpoints is None and grid is not None:
        kpoints = denser_grid(system, grid, shift)

    nscf = run_nscf(system, pseudos, density, kpoints, nbnd, conv_thr, k_batch,
                    ns=ns, tau=tau, becsum=becsum, field=field,
                    field_scale=field_scale)

    scheme, degauss = default_scheme(system, scheme, degauss, delta_e)

    tetrahedra = None
    if is_tetrahedron_scheme(scheme):
        # **The same group the k-set was reduced with**, which is the run's and
        # not the crystal's. ``build_tetrahedra`` maps every corner of the full
        # grid onto a point of the *list* it is given, using these rotations to
        # find it -- so a group larger than the one that built the list sends a
        # corner to a symmetry-equivalent point whose eigenvalues, in a run that
        # set ``nosym`` precisely because its states are not equivalent, are the
        # wrong ones. The pair has to be decided once: see
        # :func:`~pypresso.workflows.nscf.denser_grid`.
        symmetries = system.symmetry_group(nosym=system.nosym)
        tetrahedra = tetrahedra_for(scheme, nscf.kpoints, symmetries, system.cell)
        degauss = 0.0
    elif degauss is None:
        degauss = system.degauss or delta_e

    energies = energy_grid(nscf.eigenvalues, emin, emax, delta_e, degauss or 0.0)
    dos = compute_dos(
        nscf.eigenvalues,
        nscf.kpoints.weights,
        energies,
        scheme,
        degauss=degauss,
        tetrahedra=tetrahedra,
        fermi_energy=nscf.fermi_energy if nscf.fermi_energy is not None else nscf.homo,
        chunk=chunk,
        fermi_energy_up=nscf.fermi_energy_up,
        fermi_energy_down=nscf.fermi_energy_down,
    )
    return dos, nscf
