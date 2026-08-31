"""The Fermi-surface nesting function.

    N(q) = (1/N_k) sum_k  g(k) g(k + q),
    g(k)  = degspin sum_{s,n} delta(eps_snk - E_F)

Elk task 105 (``nesting.f90``); ``pw.x`` and its post-processing tools have no
counterpart, which is a ``grep`` over the vendored tree rather than a
recollection -- ``nesting`` occurs nowhere in ``PW/src`` or ``PP/src``. EPW has
it, and EPW is out of scope.

**What it is for.** ``g(k)`` is the density of states *at one k-point*, so it
is a picture of the Fermi surface on the grid, and ``N(q)`` counts how much of
that surface maps onto itself when translated by ``q``. Where it is large a
small perturbation of wavevector ``q`` connects many occupied states to many
empty ones at no energy cost, which is what makes a phonon soften, a
charge-density wave open a gap, or a spin spiral of that pitch beat the
ferromagnet. It is the ``omega -> 0``, ``Im`` part of the Lindhard function
stripped of its matrix elements: the *geometric* half of the instability, and
the half that is cheap.

**It is a convolution, and that is the one place this is not a transcription.**
``nesting.f90`` writes an ``O(N_q N_k)`` double loop, with ``k + q`` folded back
onto the grid by ``mod(ivk + ivq, ngridk)``. That fold is what makes the sum a
*cyclic cross-correlation* of ``g`` with itself over the k-grid, so

    N = ifftn(|fftn(g)|^2) / N_k

gives the whole ``q`` dependence in one transform. The double loop is
implemented beside it (``method = "direct"``) and is not decoration: the two
share no arithmetic and agree to round-off, which is the check that the index
fold and the transform's conventions are both right.

**Two identities come with the normalisation, and both are used as tests.**
The weights here are this package's DOS convention -- ``g`` carries the spin
degeneracy, so ``(1/N_k) sum_k g(k)`` is ``D(E_F)`` in states/Ry/cell -- and
therefore

* the mean of ``N`` over the whole ``q`` grid is exactly ``D(E_F)^2``, which
  ties the unfolded eigenvalues, the delta function and the transform together
  in one number that :func:`pypresso.workflows.dos.compute_dos` computes by a
  completely different route;
* ``N(0) >= N(q)`` for every ``q``, by Cauchy-Schwarz. **The nesting peak is
  therefore always a peak away from the origin**, and
  :meth:`NestingFunction.peak` excludes ``q = 0`` for that reason rather than
  as a plotting convenience.

**Units, and how to reach Elk's number.** ``N`` is in states^2 Ry^-2 per cell;
:attr:`NestingFunction.ratio` divides it by ``D(E_F)^2`` and is dimensionless.
``nesting.f90`` reports ``occmax * Omega_BZ * (1/N_k) sum_k g~ g~`` with
``g~`` carrying no degeneracy and energies in Hartree, so its number is
``(Omega_BZ / occmax) * 4 * N`` -- :attr:`NestingFunction.elk_units` applies
exactly that, and it is what a comparison against ``NEST3D.OUT`` goes through.

**The delta is the smearing registry's**, so ``smearing`` takes the same names
the SCF does. The default is a Gaussian even when the run itself used
Methfessel-Paxton or cold smearing, and that is deliberate: those two go
negative on the wings, so ``g(k)`` can be negative at a k-point whose bands sit
just off the Fermi level, and ``N(q)`` -- a product of two such -- then has no
sign at all. Ask for them by name if that is wanted; nothing here forbids it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pypresso.scf.occupations import smearing_order, w0gauss

__all__ = [
    "NestingFunction",
    "NESTING_METHODS",
    "fermi_surface_weights",
    "nesting_from_eigenvalues",
    "require_a_fermi_surface",
]

#: ``fft`` is the cyclic cross-correlation in one transform; ``direct`` is
#: ``nesting.f90``'s own double loop, kept as the independent route.
NESTING_METHODS = ("fft", "direct")


@dataclass(frozen=True)
class NestingFunction:
    """``N(q)`` on the q-grid, with the Fermi-surface weights it was built from.

    ``qpoints`` is in **crystal** coordinates on the *unshifted* grid of the
    same divisions as the k-grid -- which is where ``k + q`` lands whatever the
    k-grid's own shift is, the shift cancelling in the difference. The ordering
    is ``monkhorst_pack``'s, last index fastest, so :meth:`as_grid` reshapes
    without a permutation.
    """

    grid: tuple[int, int, int]
    qpoints: np.ndarray  # (nq, 3), crystal
    nesting: np.ndarray  # (nq,), states^2 / Ry^2 / cell
    weights: np.ndarray  # (nk,) g(k), states/Ry at one k-point
    fermi_dos: float  # D(E_F), states/Ry/cell
    fermi_energy: float  # Ry
    degauss: float  # Ry
    smearing: str
    method: str
    cell_volume: float  # bohr^3, only so that ``elk_units`` can be formed
    nspin: int = 1
    #: ``(1/N_q) sum_q N(q) - D(E_F)^2``, which is zero by construction and is
    #: carried so that the identity can be read rather than re-derived.
    sum_rule: float = 0.0

    @property
    def nq(self) -> int:
        return self.qpoints.shape[0]

    @property
    def ratio(self) -> np.ndarray:
        """``N(q) / D(E_F)^2`` -- dimensionless, and 1 on average by the sum rule."""
        return self.nesting / self.fermi_dos**2

    @property
    def elk_units(self) -> np.ndarray:
        """``NEST3D.OUT``'s column: ``(Omega_BZ / occmax) * 4 * N``.

        The 4 is Rydberg against Hartree on a quantity quadratic in a density
        of states, and ``occmax`` is Elk's spin degeneracy -- 2 without
        magnetism, 1 with. See the module docstring.
        """
        occmax = 2.0 if self.nspin == 1 else 1.0
        omega_bz = (2.0 * np.pi) ** 3 / self.cell_volume
        return (omega_bz / occmax) * 4.0 * self.nesting

    def as_grid(self) -> np.ndarray:
        """``N(q)`` reshaped to ``(n1, n2, n3)``, for plotting a slice."""
        return self.nesting.reshape(self.grid)

    def peak(self) -> tuple[np.ndarray, float]:
        """The largest ``N(q)`` over ``q != 0``, and where it is.

        ``q = 0`` is excluded because it is the maximum on every crystal --
        ``sum_k g(k) g(k+q) <= sum_k g(k)^2`` by Cauchy-Schwarz -- so including
        it would report the same uninformative point for every material. What
        is wanted is the wavevector at which the surface nests *onto a
        different part of itself*.
        """
        at_origin = np.all(np.isclose(self.qpoints, 0.0), axis=1)
        candidates = np.where(~at_origin)[0]
        if candidates.size == 0:
            raise ValueError(
                "a nesting peak needs more than one q-point; the grid is 1x1x1"
            )
        best = candidates[np.argmax(self.nesting[candidates])]
        return self.qpoints[best], float(self.nesting[best])

    def along(self, direction, count: int | None = None):
        """``(x, N)`` along one crystal direction of the grid.

        ``direction`` is a lattice-vector index (0, 1 or 2); the abscissa is the
        crystal coordinate ``q_i`` in [0, 1). This is the line plot Elk's
        ``NEST3D.OUT`` is usually read along, and it needs no interpolation
        because the grid points lie on it exactly.
        """
        axis = int(direction)
        grid = self.as_grid()
        index = [0, 0, 0]
        n = self.grid[axis]
        values = np.empty(n)
        for i in range(n):
            index[axis] = i
            values[i] = grid[tuple(index)]
        return np.arange(n) / n, values


# -- what it refuses -----------------------------------------------------------


def require_a_fermi_surface(calculation) -> None:
    """Three refusals, each naming what is missing.

    The nesting function is a statement about *one* Fermi surface, so what it
    cannot take is a run that has no Fermi level or has two of them. The order
    is by how specific the message is, not by how likely the case: a
    ``tot_magnetization`` run is also a fixed-occupation one (``input.f90``
    requires the pair), and the two-level statement is the informative half.
    """
    system = calculation.system
    if getattr(calculation, "spiral", False):
        raise NotImplementedError(
            "the nesting function of a spin spiral is not implemented: the "
            "quantity predicts the wavevector at which a spiral will win, so "
            "it is a statement about the state the spiral instability grows "
            "out of -- compute it on the non-magnetic or collinear reference "
            "and compare its peak with relax_spiral_q's answer, which is the "
            "one validation the two have against each other"
        )
    if getattr(system, "tot_magnetization", None) is not None:
        raise NotImplementedError(
            "the nesting function with a constrained tot_magnetization is not "
            "implemented: that regime has one Fermi level per spin channel "
            "(input.f90's two-level branch), so g(k) is two different "
            "surfaces and N(q) would have to be resolved by channel pair. It "
            "is P45's refusal on the same object for the same reason"
        )
    occupations = str(getattr(system, "occupations", "") or "")
    if occupations.startswith("fixed") or occupations in ("", "from_input"):
        raise NotImplementedError(
            "the nesting function needs a Fermi level and occupations = "
            f"{occupations!r} has none: a fixed-occupation run fills a set "
            "number of bands per channel and never searches for a level, so "
            "delta(eps - E_F) has no argument. Run the metal with a smearing"
        )


# -- the assembly --------------------------------------------------------------


def fermi_surface_weights(
    eigenvalues,
    fermi_energy: float,
    degauss: float,
    smearing: str = "gaussian",
    degeneracy: float = 2.0,
) -> np.ndarray:
    """``g(k) = degspin sum_{s,n} delta(eps_snk - E_F)`` in states/Ry.

    Args:
        eigenvalues: ``(nspin, nk, nbnd)`` in Ry, on the **complete** grid.
        fermi_energy: in Ry.
        degauss: the delta's width in Ry.
        smearing: a name from
            :data:`~pypresso.scf.occupations.SMEARING_ORDER`.
        degeneracy: how many electrons a band holds -- 2 for ``nspin = 1``, 1
            for a spinor or for one channel of an LSDA pair. It is the same
            ``degspin`` the k-point weights carry, so that ``(1/N_k) sum_k g``
            is the density of states this package's ``compute_dos`` reports.

    The spin axis is *summed over*, not kept: ``nspin = 2`` has one Fermi level
    shared between its channels (the two-level case is refused above), so the
    surface it nests on is the union of the two.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if eigenvalues.ndim == 2:
        eigenvalues = eigenvalues[None]
    if degauss <= 0.0:
        raise ValueError(f"degauss must be positive, got {degauss}")
    ngauss = smearing_order(smearing)
    # ``w0gauss``'s argument convention is dosg.f90's, ``x = (E - e)/degauss``
    # with ``E`` in the role of the level: positive for a state below it. The
    # sign is invisible for a Gaussian and wrong for everything else.
    x = (fermi_energy - eigenvalues) / degauss
    delta = np.asarray(w0gauss(x, ngauss)) / degauss
    return float(degeneracy) * delta.sum(axis=(0, 2))


def nesting_from_eigenvalues(
    eigenvalues,
    grid: tuple[int, int, int],
    fermi_energy: float,
    degauss: float,
    *,
    smearing: str = "gaussian",
    degeneracy: float = 2.0,
    method: str = "fft",
    cell_volume: float = 1.0,
    nspin: int = 1,
) -> NestingFunction:
    """``N(q)`` from eigenvalues on the complete grid. No SCF anywhere.

    This is the whole physics of the phase and it takes nothing but numbers, so
    a free-electron band can be fed straight in -- which is how the analytic
    limit ``N(q) = Omega / (4 pi^2 q)`` below ``2 k_F`` is checked without a
    pseudopotential.

    Args:
        eigenvalues: ``(nspin, nk, nbnd)`` or ``(nk, nbnd)`` in Ry, on the
            **complete** ``n1 n2 n3`` grid in ``monkhorst_pack`` order (last
            index fastest). :func:`~pypresso.workflows.nesting.run_nesting`
            unfolds a symmetry-reduced wedge onto it.
        grid: the divisions ``(n1, n2, n3)``.
        fermi_energy: in Ry.
        degauss: the delta's width in Ry.
        method: ``"fft"`` or ``"direct"``; see :data:`NESTING_METHODS`.
    """
    if method not in NESTING_METHODS:
        raise ValueError(
            f"unknown nesting method {method!r}; expected one of {NESTING_METHODS}"
        )
    divisions = tuple(int(n) for n in grid)
    nk = int(np.prod(divisions))
    g = fermi_surface_weights(
        eigenvalues, fermi_energy, degauss, smearing, degeneracy
    )
    if g.shape[0] != nk:
        raise ValueError(
            f"the nesting function needs eigenvalues on the complete "
            f"{divisions[0]}x{divisions[1]}x{divisions[2]} grid ({nk} points), "
            f"got {g.shape[0]}. A symmetry-reduced wedge has to be unfolded "
            f"first (pypresso.workflows.nesting.run_nesting does it)"
        )

    box = g.reshape(divisions)
    if method == "fft":
        transformed = np.fft.fftn(box)
        correlation = np.real(np.fft.ifftn(np.abs(transformed) ** 2))
    else:
        # ``nesting.f90``'s own loop, vectorised over ``k`` but not over ``q``:
        # ``np.roll(box, -q)`` is ``g(k + q)`` with exactly the fold
        # ``mod(ivk + ivq, ngridk)`` the Fortran writes.
        correlation = np.empty(divisions)
        for i in range(divisions[0]):
            for j in range(divisions[1]):
                for k in range(divisions[2]):
                    shifted = np.roll(box, shift=(-i, -j, -k), axis=(0, 1, 2))
                    correlation[i, j, k] = float(np.sum(box * shifted))
    nesting = (correlation / nk).reshape(nk)

    i, j, k = np.meshgrid(*(np.arange(n) for n in divisions), indexing="ij")
    qpoints = np.stack(
        [i.ravel() / divisions[0], j.ravel() / divisions[1], k.ravel() / divisions[2]],
        axis=1,
    )

    fermi_dos = float(g.sum() / nk)
    return NestingFunction(
        grid=divisions,
        qpoints=qpoints,
        nesting=nesting,
        weights=g,
        fermi_dos=fermi_dos,
        fermi_energy=float(fermi_energy),
        degauss=float(degauss),
        smearing=smearing,
        method=method,
        cell_volume=float(cell_volume),
        nspin=int(nspin),
        sum_rule=float(nesting.mean() - fermi_dos**2),
    )
