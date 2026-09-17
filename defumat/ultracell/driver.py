"""The ultracell self-consistent loop.

``PLAN.md`` P88. The shape of the calculation, and the one line that says why
it is cheap:

1. converge the **unit cell** once, as usual;
2. diagonalise the unit cell once more on the folded k-set ``k0 + Q``, at that
   fixed density -- an ordinary NSCF run, and **this is the expensive step**;
3. then loop: build the ``(N nbnd, N nbnd)`` matrix at each ``k0``, diagonalise
   it, accumulate the ultracell density, rebuild the potential, mix.

Step 2 happens **once, before the loop**, and step 3 never touches a plane-wave
Hamiltonian again. Elk's ``gndstulr`` has the same structure (``genevfsv``
before ``do iscl``) and it is the whole reason a modulation over tens or
hundreds of cells is affordable: the fast, chemical part of the wavefunction is
solved once and frozen, and the self-consistency runs on the envelope alone.

**What is variational here and what is not.** The ultracell basis
``{psi_{k0+Q,n} : all Q, n <= nbnd}`` spans, as ``nbnd`` grows to the full
plane-wave count, exactly the ultracell's own plane-wave basis at ``k0``. So
this is a truncation of the exact ``N``-cell supercell problem to ``nbnd`` bands
per folded k-point and nothing else. **The convergence of an eigenvalue in
``nbnd`` carries no sign**: Rayleigh-Ritz bounds the eigenvalues of a *fixed*
Hamiltonian, and this one moves with its own truncated density, so an ultracell
eigenvalue is not an upper bound on the supercell's, and neither is a density.

**The total energy is the one quantity here that does carry a sign**, and it is
``total_energy`` on the result (``ultracell/energy.py``; Elk computes none --
``energyulr.f90`` is the eigenvalue sum alone, and ``pw.x`` has no ultracell).
The reported number is the Kohn-Sham free energy of the state the iteration
produced, and the bases are **nested** in ``nbnd``, so it falls monotonically
towards the supercell's own from above: +8.15e-05, +3.52e-06, +3.62e-07 and
+1.43e-07 Ry at ``nbnd = 12, 24, 48, 64`` on two-cell silicon under an applied
modulation.

**The density is mixed, not the potential.** Elk mixes its ``Q``-resolved
potential and reports an RMS change in it; here the mixed quantity is the
density on the ultracell box and the convergence test is the Hartree energy of
the density residual -- QE's ``dr2``, generalised to ``|G+Q|``. That is a
deliberate departure: it means ``conv_thr`` means the same thing in an ultracell
run as in every other run in this package, and it lets the Kerker
preconditioner, which is defined on a density, damp the long-wavelength
sloshing that a large cell is prone to and that forces Elk's example down to
``beta0 = 0.001``.

**Two spin channels, and what they share.** Without spin-orbit coupling the
ultracell matrix is **block diagonal in spin** -- a collinear potential is
diagonal in spin and the basis is built per channel -- so ``nspin = 2`` is two
``(N nbnd)`` problems rather than one of twice the size, each with its own
``dV[s]``. The two channels share exactly one thing, and it is the interesting
one: a **single Fermi level** over all ``2 N nbnd`` states, which is what lets
an electron cross from the minority channel in one cell to the majority channel
in the next. That crossing *is* a spin density wave, and two separate Fermi
levels forbid it -- which is why the constrained branch here is the ``Q = 0``
moment constraint and nothing finer.

**A spinor run is one block, and the magnetization is then a vector field.**
At ``nspin = 4`` a basis function has two components, the potential acts on it
as the 2x2 matrix ``v_0 I + B . sigma``, and nothing factorises: the
off-diagonal Pauli terms couple the components at every point of the grid. What
that buys is the modulation a collinear run cannot write down -- one whose
*direction* turns from cell to cell, which is a helix or a cycloid rather than
an amplitude wave -- and it is the regime the method's own headline case, a
noncollinear spin density wave, lives in. Spin-orbit coupling comes along for
free, because it is entirely inside the frozen states and adds no term here.
The density has four components, the states hold **one** electron each rather
than two, and there is one Fermi level over all ``N nbnd`` of them.

**Nothing breaks spin symmetry on its own**, so an unpolarized unit cell put in
an ultracell stays unpolarized however many cells it has, and a *tiled* magnetic
one stays tiled -- the tiled state is an exact fixed point of this loop. A
modulation therefore comes from one of two places, and they are different
quantities. ``magnetic_field`` **drives** one, and what comes back is the
``Q``-resolved spin susceptibility, a response rather than an ordered state.
``seed_magnetization`` **hands one over** as the initial condition and the loop
keeps it, which is the ordered state itself (:mod:`defumat.ultracell.seed`).
Elk's route is the second with a fade in it -- a random field (``rndbfcu``)
taken away geometrically (``reducebf``) -- because Elk's self-consistency runs
on the potential, where a seed is a term in the functional and has to leave;
here the mixed quantity is the density, so a seed is an initial condition,
nothing is faded and the convergence test is untouched. A *random* seed is an
ordinary argument here, so letting a spontaneous wave find its own period is
available and is **not measured**; what is measured is a seed whose period was
handed over.

**What the peak costs.** Two objects grow with ``N`` and nothing else does.

* the ultracell density and potential, ``N x (dense FFT grid) x nspin_mag``
  float64 apiece, with a handful of copies alive at once (the mixer's history is
  ``mixing_ndim`` more). Two-atom silicon at ``N = 20`` is 540 kB; a 21-cell
  chromium cell on a 27^3 grid with four magnetization components is 13 MB; a
  10^4-cell ultracell is 6.4 GB and is the wall a one-dimensional modulation
  hits first.
* the ultracell Hamiltonian, ``(N nbnd)^2`` complex per ``k0``, which is 6.4 MB
  at ``nbnd = 30, N = 21`` and does not become the constraint before the
  ``(N nbnd)^3`` dense solve has already stopped the run.

What is *not* on that list is the thing a naive implementation would allocate:
one ultracell box per basis function. ``state_batch`` is one by default for that
reason -- the matrix build and the density both walk their state axis through
``lax.map`` rather than ``vmap``, so a single box is in flight at a time.

The k0 loop is a **Python** loop rather than a batched axis, which is QE's
``k_loop`` and the same trade: one ``k0``'s wavefunctions and one ultracell box
are the working set, whatever the sampling. Batching it is what an accelerator
would want and is not written.

What is refused, and why each: see :func:`require_an_ultracell_regime`.
"""

from __future__ import annotations

import dataclasses
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from defumat.basis.builder import build_basis
from defumat.scf.mixing import MIXERS, get_mixer
from defumat.xc.functional import resolve_functional
from defumat.scf.occupations import fixed_occupations, smeared_occupations
from defumat.system.kpoints import for_spin
from defumat.ultracell.energy import (
    total_of,
    ultracell_energy,
    ultracell_entropy,
)
from defumat.ultracell.augmentation import (
    as_onecentre_becsum,
    augmentation_matrix,
    becsum_per_copy,
    build_ultracell_augmentation,
    coefficients_per_difference,
    from_onecentre_coefficients,
    onecentre_deeq,
    ultracell_augmentation_charge,
    spinor_augmentation_matrix,
    spinor_ultracell_becsum,
    spinor_ultracell_deeq,
    ultracell_becsum,
    ultracell_deeq,
    ultracell_projections,
)
from defumat.ultracell.grid import Ultracell, folded_kpoints
from defumat.ultracell.hamiltonian import multiplet_cut, ultracell_matrix
from defumat.ultracell.mixing import box_kerker
from defumat.ultracell.states import (
    UltracellStates,
    ultracell_band_density,
)
from defumat.scf.continuation import spin_components
from defumat.ultracell.seed import (
    reference_axis,
    seeded_becsum,
    seeded_density,
)
from defumat.ultracell.potential import (
    delta_potential,
    hartree_kernel,
    require_an_ultracell_functional,
    ultracell_potential,
)
from defumat.units import E2, FPI, RY_TO_EV

#: Below this gap (Ry) the ``nbnd`` truncation is cutting a degenerate
#: multiplet rather than falling in a gap, and the basis is then whatever the
#: eigensolver happened to return. Set from the measured cases on silicon:
#: ``nbnd = 8`` cuts at 2.6e-14 Ry and does not reliably converge, ``nbnd = 24``
#: cuts at 2.6e-6 Ry and converges in nine iterations. A cut this degenerate is
#: *common* high in the empty manifold (``nbnd = 32``, ``48`` and ``80`` all
#: have one) and harmless there, so the warning it raises is gated on the cut
#: lying among the low empty bands as well -- see :func:`run_ultracell`.
DEGENERATE_CUT = 1.0e-8

__all__ = ["UltracellResult", "run_ultracell", "require_an_ultracell_regime",
           "DEGENERATE_CUT"]


@dataclasses.dataclass
class UltracellResult:
    """What an ultracell run produces."""

    #: ``(nspin, *box)`` density on the ultracell grid, normalised **per unit
    #: cell**: it integrates to ``N`` times the electron count over the
    #: ultracell, and equals the tiled unit-cell density when nothing is
    #: modulated.
    density: jnp.ndarray
    #: ``(nspin, *box)`` the difference potential the matrix was built from.
    delta_v: jnp.ndarray
    #: ``(nk0, N nbnd)`` ultracell eigenvalues in Ry.
    eigenvalues: np.ndarray
    #: ``(nk0, N nbnd)`` weighted occupations, QE's ``wg``.
    occupations: np.ndarray
    fermi_energy: float
    #: The occupied eigenvalue sum per unit cell -- Elk's ``evalsum``, which is
    #: the only energy Elk's own ultracell reports (``energyulr.f90``). It is
    #: **not** the total energy: it double-counts the Hartree and
    #: exchange-correlation terms, exactly as an ordinary ``eband`` does. See
    #: :attr:`total_energy`.
    band_energy: float
    #: The **total energy per unit cell**, in Ry, which neither ``pw.x`` nor
    #: Elk computes for an ultracell. It is the Kohn-Sham free energy of the
    #: state the last iteration produced, so it is the quantity that says
    #: whether a modulation is worth its own cost -- the energy *gain* of a spin
    #: density wave over the uniform state is a difference of two of these.
    #:
    #: **It is the one quantity here that converges with a sign.** The bases are
    #: nested in ``nbnd``, so it falls monotonically towards the ``N``-cell
    #: supercell's own energy from above, where an eigenvalue or a density does
    #: not: +8.15e-05, +3.52e-06, +3.62e-07 and +1.43e-07 Ry at
    #: ``nbnd = 12, 24, 48, 64`` on two-cell silicon under an applied
    #: modulation. Comparing it against a supercell needs the **same** FFT box
    #: on both sides, since two discretisations of the same functional differ by
    #: about 1e-6 Ry per cell here and that is larger than the top of the ladder.
    #:
    #: **Under an applied ``magnetic_field`` the bounded quantity is
    #: ``total_energy + field_energy``**, not this on its own: what is minimised
    #: is the full energy and the Zeeman term is deliberately outside what is
    #: reported, so the total alone is a bound on nothing and was measured going
    #: the wrong way -- 7.4e-07 Ry above an ordinary SCF at ``nbnd = 12`` and
    #: 9.7e-07 and 1.6e-06 *below* it at 24 and 40, where the sum is +4.15e-06,
    #: +2.47e-06 and +5.15e-07, above at every rung and falling.
    total_energy: float
    #: The terms it is made of, in ``SCFResult.energy_terms``'s shape:
    #: ``one-electron``, ``hartree``, ``xc``, ``ewald``, and ``dispersion`` and
    #: ``smearing`` where they apply. They sum to :attr:`total_energy`.
    energy_terms: dict
    converged: bool
    iterations: int
    #: The Hartree energy of the last density residual, in Ry: QE's ``dr2`` on
    #: the ultracell box, **charge and magnetization together** -- the number
    #: compared against ``conv_thr``.
    accuracy: float
    #: The tightest gap the ``nbnd`` truncation opened over the folded k-set,
    #: in Ry. Small means the basis cut a degenerate multiplet and is arbitrary
    #: there -- which matters where the cut is among the low empty bands the
    #: response is carried by, and was measured not to matter thirty bands up.
    multiplet_gap: float
    ultracell: Ultracell
    #: The converged unit-cell state this was built on.
    reference: object = None
    seconds: float = 0.0
    #: How complex the augmented density came back, relative to its own largest
    #: value, or ``0`` for a norm-conserving run that has no augmentation charge.
    #:
    #: **It is a convergence number in ``ecutrho`` and not an error.** The
    #: ultracell's reciprocal cutoff set is the unit cell's dense sphere at every
    #: ``Q`` (Elk's choice, :meth:`Ultracell.reciprocal_mask`), and that set is
    #: **not closed under negation** once ``Q`` is non-zero: the negative of
    #: ``G + Q`` is ``-G - Q``, which is the sphere displaced rather than the
    #: sphere: on the silicon cell this was measured on, **235 of 4554** kept
    #: entries have no negative partner at ``N = 2`` and 470 of 6831 at
    #: ``N = 3``, where at ``N = 1`` every one of the 2277 does. A real
    #: function represented on such a set is not exactly real, and the imaginary
    #: part that is dropped is the augmentation charge's weight at the edge of
    #: the sphere.
    #:
    #: Measured on ultrasoft silicon: **exactly zero at ``N = 1``**, where the
    #: only ``Q`` is zero and the set is symmetric; 1.71e-4 at ``N = 2`` with
    #: ``ecutrho = 64`` Ry, falling to 7.21e-5 at 96 and 2.96e-5 at 144, which is
    #: ``ecutrho^-2.2``; and 1.32e-4 at ``N = 3``, the same order, because what
    #: it measures is the displacement of the sphere rather than ``N``. So it is
    #: read the way a cutoff convergence is read, and the Hartree term of a
    #: *norm-conserving* ultracell carries the same truncation without reporting
    #: it.
    augmentation_residual: float = 0.0
    #: The charge half of :attr:`accuracy` -- the residual weighted by
    #: ``1/|G+Q|^2``.
    charge_accuracy: float = 0.0
    #: The magnetization half, weighted by a constant and **including** its
    #: ``G + Q = 0`` component. Reported apart because the sum hides which one
    #: is still moving, and on a magnetic cell the two are weighted an order
    #: apart: ``OPEN.md`` Y1 measured a charge-only ``dr2`` of 9e-11 leaving the
    #: total energy 1.15e-8 Ry out. For a spin density wave the magnetization
    #: *is* the answer, so this is the half that decides whether a run is
    #: finished.
    magnetic_accuracy: float = 0.0
    #: ``(iterations, 3)``: ``(dr2, charge, magnetization)`` per iteration.
    history: tuple = ()
    #: The total energy per iteration, in Ry, beside :attr:`history` rather than
    #: inside it -- a fourth column would have widened a tuple this file unpacks
    #: as a triple. Every entry is the Kohn-Sham energy of a state that exists,
    #: because the Hartree and exchange-correlation terms are taken at the
    #: iteration's **output** density rather than at the mixture the next one
    #: starts from, which is what makes QE's ``descf`` correction unnecessary.
    #: So the sequence is an upper bound approaching from above and its decrease
    #: is readable as convergence.
    energy_history: tuple = ()
    #: ``-int B . m`` per unit cell for an applied ``magnetic_field``, in Ry,
    #: and ``None`` without one. Carried beside the total rather than inside
    #: it, which is QE's and Elk's shared convention and this package's
    #: (``SCFResult.field_energy``). Add it back to :attr:`total_energy` to
    #: recover the quantity the calculation actually minimises, which is the
    #: one that converges with a sign.
    field_energy: float | None = None
    #: The **unit cell's** volume in bohr^3, carried so that an integral over
    #: one cell of the ultracell can be taken without the caller holding the
    #: system as well as the result.
    cell_volume: float = 1.0
    #: The frozen states and the envelope amplitudes, as
    #: :class:`~defumat.ultracell.states.UltracellStates` -- what an ultracell
    #: *wavefunction* is made of, and what anything looking at ``Psi`` rather
    #: than at ``|Psi|^2`` needs. Kept unless ``keep_states=False``; both arrays
    #: are alive for the whole loop in any case, so keeping them raises no peak.
    states: object = None

    @property
    def eigenvalues_ev(self) -> np.ndarray:
        return self.eigenvalues * RY_TO_EV

    @property
    def modulation(self) -> jnp.ndarray:
        """``rho - tile(rho_0)``: what the ultracell has that the unit cell has not."""
        tiled = self.ultracell.tile(jnp.asarray(self.reference.density))
        return self.density - tiled

    @property
    def magnetization(self) -> jnp.ndarray:
        """The magnetization on the box, per unit cell.

        ``rho_up - rho_dw`` and shaped ``(*box)`` for a collinear run;
        ``(3, *box)`` -- the cartesian components ``(m_x, m_y, m_z)`` -- for a
        noncollinear one, where the *direction* is as much of the answer as the
        length. Zero, with the collinear shape, when there is no magnetization
        at all.

        The envelope a spin density wave lives in, and the reason the phase has
        a stage 3 at all: it is what a modulated field induces, what
        ``seed_magnetization`` hands the loop and the loop keeps, and what
        ``magnetic_accuracy`` bounds.
        """
        components = self.density.shape[0]
        if components == 1:
            return jnp.zeros_like(self.density[0])
        if components == 4:
            return self.density[1:]
        return self.density[0] - self.density[1]

    def cell_moments(self) -> np.ndarray:
        """The moment of each unit cell of the ultracell, in mu_B/2.

        ``(N,)`` for a collinear run and ``(N, 3)`` for a noncollinear one: the
        integral of the magnetization over one cell ``R``, which is the quantity
        a spin density wave is *read off*. For a collinear wave its sign
        alternates with the envelope and its profile is the wave; for a helical
        one the length is constant and the whole answer is in how the direction
        turns from one cell to the next. **None** of it needs a sphere -- the
        cells tile the ultracell exactly, so the partition is the geometry
        rather than a choice of radius, which is the one thing an ultracell has
        over an atom-resolved moment.
        """
        shape = self.ultracell.shape
        cell_grid = self.ultracell.cell_grid
        m = np.asarray(self.magnetization)
        vector = m.ndim == 4
        # Split each axis into (cell index, point inside the cell) and sum the
        # inside. The box index is ``J_i = n_i G_i + q_i`` in reciprocal space,
        # but in *real* space a cell boundary falls every ``Nd_i`` points, so
        # the reshape is (n_i, Nd_i) in that order and not the other.
        lead = (3,) if vector else ()
        m = m.reshape(lead + (shape[0], cell_grid[0], shape[1], cell_grid[1],
                              shape[2], cell_grid[2]))
        axes = (1, 3, 5) if not vector else (2, 4, 6)
        per_cell = m.sum(axis=axes)
        element = float(self.cell_volume) / float(np.prod(cell_grid))
        per_cell = per_cell * element
        if not vector:
            return per_cell.reshape(-1)
        # Components last, so that a row is one cell's moment vector -- the
        # same ``(nat, 3)`` layout ``SCFResult.site_moments`` uses, and what
        # every angle and singular-value check downstream expects.
        return per_cell.reshape(3, -1).T

    def __repr__(self) -> str:
        n = "x".join(str(m) for m in self.ultracell.shape)
        state = "converged" if self.converged else "NOT converged"
        levels = np.atleast_1d(np.asarray(self.fermi_energy, dtype=float))
        fermi = ", ".join(f"{e * RY_TO_EV:.4f}" for e in levels)
        return (
            f"UltracellResult({n} cells, {self.iterations} iterations, {state}, "
            f"E = {self.total_energy:.8f} Ry, dr2 = {self.accuracy:.3e} Ry, "
            f"E_F = {fermi} eV)"
        )


def require_an_ultracell_regime(system, pseudos, basis) -> None:
    """Refuse every regime this has not been measured in, by name.

    Called **before** the frozen states are computed, not after: that
    diagonalisation is the expensive step of the whole method, and a run
    that is going to be refused should not pay for it first. Everything
    checked here is available from the system, its pseudopotentials and its
    basis, so none of it needs a ``Calculation`` to have been built.

    **All three spin regimes are here**, and spin-orbit coupling with them --
    it lives entirely in the frozen unit-cell states and adds no term to
    anything below. An **ultrasoft or PAW** dataset runs too, in all three
    (``PLAN.md`` P88 stages 5 and 6): the displaced tables of
    :mod:`defumat.ultracell.augmentation` carry the augmentation charge, PAW's
    one-centre terms are evaluated per atom copy, and for a spinor the
    integrals are recombined into spin blocks and sandwiched between ``fcoef``
    one Q-difference at a time, which is ``newd_so`` with a wavevector on it.
    The practical wall such a dataset meets first is the double grid below.

    What is *not* here and is refused where it is reached: a
    ground state converged under a field (:func:`run_ultracell`, because the
    frozen eigenvalues then carry a field ``dV`` does not), and an applied
    field on a noncollinear cell that carries no magnetization
    (:func:`_as_field`, because its density and its potential have one
    component and a field has nothing to be added to). Both depend on the call
    rather than on the system, which is why neither is in this function.
    """
    require_an_ultracell_functional(
        resolve_functional([p.functional for p in pseudos], system.input_dft),
        bool(getattr(system, "nosource", False)),
    )
    if getattr(system, "hubbard", None) is not None:
        raise NotImplementedError(
            "the ultracell refuses DFT+U: the occupation matrix is per atom and "
            "an ultracell holds N copies of every atom, each with its own"
        )
    if getattr(system, "spiral_q", None) is not None:
        raise NotImplementedError(
            "the ultracell refuses a spin spiral: it is a second modulation of "
            "the same states. Elk's gndstulr does not check spinsprl and so "
            "accepts the combination untested, which is a reason to refuse it "
            "rather than a precedent"
        )
    if not bool(getattr(system, "nosym", False)):
        raise NotImplementedError(
            "the ultracell needs symmetry switched off (nosym = .true., and "
            "noinv = .true. with it). A modulation breaks the crystal's point "
            "group and the ultracell's own group is not written, so nothing "
            "here symmetrises anything -- but the unit-cell density this "
            "expands around would have been integrated over a reduced wedge, "
            "and then the tiled density is not the fixed point the folded "
            "k-set reproduces. Elk does the same thing by force, setting "
            "reducek = 0 for the whole run"
        )
    if basis.planewaves.gamma_only:
        raise NotImplementedError(
            "the ultracell refuses gamma-only storage: the basis is built at "
            "k0 + Q for N different Q and none of them is Gamma"
        )
    if basis.doublegrid:
        # **This is the refusal an ultrasoft or PAW run meets first** whenever
        # its input sets the ``ecutrho`` such a dataset actually wants. The
        # ultracell box is built on the *dense* grid and everything already
        # lives on it, so what is missing is not an interpolation between two
        # boxes but the truncation that makes the two halves agree: ``h_psi``
        # multiplies a wavefunction by the potential **interpolated down to the
        # smooth sphere**, so a ``dV`` carrying its dense components into the
        # matrix element would give the ultracell a term the frozen eigenvalue
        # and the reference supercell do not have. Both halves are one line and
        # neither is measured, which is why this is refused rather than taken.
        raise NotImplementedError(
            "the ultracell needs the smooth and dense grids to coincide, which "
            "they do at ecutrho = 4 ecutwfc. With a double grid the smooth half "
            "of the matrix element would carry dV's dense components where "
            "h_psi truncates them to the smooth sphere, so the ultracell and "
            "the supercell it is validated against would not be discretising "
            "the same functional. Lower ecutrho to 4 ecutwfc, and read the "
            "warning that comes with doing so"
        )
    if any(p.is_ultrasoft or p.is_paw for p in pseudos):
        # **The silent half of the refusal above, and it is the one that will
        # actually be met.** ``ecutrho`` defaults to ``4 ecutwfc`` here as it
        # does in ``pw.x`` (``system/builder.py``), so an ultrasoft or PAW input
        # that simply leaves it out is *accepted* by the double-grid check and
        # runs with the augmentation charge represented on the wavefunction
        # grid -- which is the one thing such a dataset is built not to do. The
        # run is internally consistent and a comparison against a supercell at
        # the same pair is still like-for-like; what is wrong is the absolute
        # number, and nothing downstream would say so.
        warnings.warn(
            "this ultracell runs an ultrasoft or PAW dataset at ecutrho = "
            "4 ecutwfc, because that is the only value it accepts -- a double "
            "grid is refused. Such a dataset is normally run at 8 to 12 times "
            "ecutwfc, and at 4 the augmentation charge is represented on the "
            "wavefunction grid: the run is self-consistent and a comparison "
            "against a supercell at the same cutoffs is still like for like, "
            "but the absolute energy is not converged in ecutrho and cannot be "
            "compared against a pw.x number taken at the dataset's own dual",
            stacklevel=2,
        )


def _box_indices(ultracell: Ultracell, calculation, nk0: int) -> np.ndarray:
    """``(nk0, N, npwx)`` flat ultracell box index of every plane wave.

    Per ``k0`` as well as per ``Q``, because the plane-wave sphere is rebuilt at
    every k-point and two folded points do not hold the same Miller indices.
    """
    planewaves = calculation.basis.planewaves
    miller = np.asarray(calculation.basis.dense.miller)[np.asarray(planewaves.indices)]
    cells = ultracell.cells
    miller = miller.reshape(nk0, cells, -1, 3)

    index = np.empty(miller.shape[:3], dtype=np.int64)
    for iq in range(cells):
        index[:, iq, :] = ultracell.box_index(miller[:, iq, :, :], iq)
    # A padded plane wave points at G = 0 and carries a zero coefficient, so
    # the accumulating scatter downstream is safe wherever it lands: it sends
    # the pad to its own Q-row's G = 0 and adds nothing there.
    return index


#: How complex an atom copy's ``becsum`` is allowed to come back, relative to its
#: own largest entry. It is a **diagnostic rather than a tolerance**: the
#: transform over the Q-difference index involves no truncation at all, so it is
#: real to round-off and anything at this level means the array is not what it
#: should be.
#:
#: **What it catches, stated narrowly, because the wide version was measured and
#: was wrong.** A pairing ``sum_q conj(A[q]) A[f(q)]`` transforms over the
#: difference index into ``conj(a(R)) a(R)`` whenever ``f`` is a **translation**
#: of the Q index, which is real once the channel pair is symmetrised. So this
#: catches a corrupted array, an array built from the wrong species, and a
#: *reflected* index -- ``q -> d - q`` gives ``conj(a(-R)) a(R)`` and was
#: measured at 9.0e-7, three decades above the threshold -- and it is **blind to
#: a shift**, the flipped displacement included, which at ``N = 4`` moves the
#: total energy by 2.0e-7 Ry and the iteration count from 10 to 12 while leaving
#: this at 4.7e-17 (``PLAN.md`` P88, the stage 5 amendment, which has the table).
#: **A shifted index is caught by the supercell ladder and by nothing here.**
#:
#: **It does not apply to the augmented density**, which is a different quantity
#: for a reason worth knowing -- see
#: :attr:`UltracellResult.augmentation_residual`.
_BECSUM_REAL_TOLERANCE = 1.0e-9

#: How complex the augmented **density** may come back before it stops being the
#: truncation it normally is. Seven decades above what the truncation was
#: measured at, which is the gap that makes it readable; it is **not** two
#: decades below what a wrong pairing gives, as this used to say, because the
#: pairing errors measured at ``N = 4`` move it to between 8.3e-5 and 1.7e-3
#: where a correct run reads 8.4e-5 -- see the note on the tolerance above.
_AUGMENTATION_REAL_TOLERANCE = 1.0e-2


def _tiled_becsum(becsum, cells: int) -> tuple:
    """The unit cell's ``becsum`` on every copy -- the loop's starting state.

    It is the fixed point of the tiled null by construction, which is what makes
    that check a statement about the machinery rather than about the seed.
    """
    out = []
    for values in becsum:
        if values is None:
            out.append(None)
            continue
        values = jnp.asarray(values)
        out.append(jnp.broadcast_to(values[None], (int(cells),) + values.shape))
    return tuple(out)


def _ultracell_becsum(becp, vectors, occupations, augmentation, blocks, nk0,
                      nspin_mag: int, fcoef=None) -> tuple:
    """``becsum`` by Q-difference, accumulated over the blocks and over ``k0``.

    The spin axis is the block index, because a collinear ultracell solves one
    matrix per channel and a channel's projector occupations are built from that
    channel's states alone -- the same statement the density makes one line
    above, and the reason neither needs a spin argument of its own.

    ``fcoef``, one ``(nh, nh, 2, 2)`` array per species, switches to the spinor
    route, where there is one block and the spin axis is instead the four Pauli
    traces of a spin-density matrix (``add_becsum_so``). What comes back has the
    same layout either way, so nothing downstream branches.
    """
    totals = None
    for block in range(int(blocks)):
        for ik in range(int(nk0)):
            if fcoef is not None:
                values = spinor_ultracell_becsum(
                    becp[block, ik], vectors[block][ik],
                    jnp.asarray(occupations[block][ik]), augmentation,
                    fcoef, int(nspin_mag),
                )
                if totals is None:
                    totals = [
                        None if v is None else jnp.zeros_like(v) for v in values
                    ]
                for t, v in enumerate(values):
                    if v is not None:
                        totals[t] = totals[t] + v
                continue
            values = ultracell_becsum(
                becp[block, ik], vectors[block][ik],
                jnp.asarray(occupations[block][ik]), augmentation,
            )
            if totals is None:
                totals = [
                    None if v is None else jnp.zeros(
                        (v.shape[0], int(nspin_mag)) + v.shape[1:], dtype=v.dtype)
                    for v in values
                ]
            for t, v in enumerate(values):
                if v is not None:
                    totals[t] = totals[t].at[:, block].add(v)
    return tuple(totals)


def _becsum_copies(becsum_q, augmentation) -> tuple:
    """Every atom copy's own occupations, and how complex they came back.

    The second number is a check that costs nothing: a projector occupation
    matrix is Hermitian on each copy, so its transform over the difference index
    is real, and the residual is reported relative to the largest entry rather
    than absolutely. **What it is a check *of* is narrower than it looks** --
    see :data:`_BECSUM_REAL_TOLERANCE`, where four deliberate index errors are
    measured to leave it at round-off.
    """
    copies, residual = [], 0.0
    for values in becsum_q:
        if values is None:
            copies.append(None)
            continue
        real, worst = becsum_per_copy(values, augmentation.shape)
        scale = float(jnp.max(jnp.abs(real)))
        residual = max(residual, worst / max(scale, 1.0e-30))
        copies.append(real)
    return tuple(copies), residual


def _flatten_becsum(becsum) -> np.ndarray:
    """The per-copy occupations as one real vector, for the mixer."""
    parts = [np.asarray(v).ravel() for v in becsum if v is not None]
    return np.concatenate(parts) if parts else np.zeros(0)


def _unflatten_becsum(vector, template) -> tuple:
    """:func:`_flatten_becsum` inverted, against the shapes it was built from."""
    out, offset = [], 0
    for values in template:
        if values is None:
            out.append(None)
            continue
        size = int(np.prod(values.shape))
        out.append(jnp.asarray(vector[offset:offset + size]).reshape(values.shape))
        offset += size
    return tuple(out)


def _paw_deband(ddd, becsum_copies, cells: int) -> float:
    """``-(1/N) sum_R sum_s sum_ij ddd^R_ij becsum^R_ij``, per unit cell.

    ``delta_e``'s PAW half (``scf/driver.py``'s ``_paw_deband``) with the copies
    summed and divided by ``N``, and it pairs the coefficients built from the
    **input** ``becsum`` with the **output** one, exactly as the unit cell does
    -- the asymmetry is QE's, ``delta_e`` running before ``PAW_potential`` is
    called again, and it vanishes at self-consistency.
    """
    total = 0.0
    for coefficients, values in zip(ddd, becsum_copies):
        if coefficients is None or values is None:
            continue
        total += float(jnp.sum(jnp.real(coefficients) * jnp.asarray(values)))
    return -total / int(cells)


def _accuracy(residual, ultracell, cell, g2_inverse, keep):
    """QE's ``dr2`` on the ultracell box, and its two halves.

    ``scf_accuracy_split`` transcribed onto the ``G + Q`` set: the charge half
    is the Hartree energy of the residual, weighted by ``1/|G+Q|^2`` with
    ``G = Q = 0`` dropped, and the magnetization half carries a **constant**
    weight ``e2 4 pi / (2 pi)^2`` and **keeps** its ``G + Q = 0`` component --
    ``rho_ddot`` in ``PW/src/scf_mod.f90``, whose asymmetry is physical: an
    error in the total charge is forbidden by neutrality, where a uniform shift
    of the magnetization is a real error the loop has to see.

    **Reporting the two apart is not bookkeeping here, it is the phase's own
    convergence criterion.** ``OPEN.md`` Y1 measured what a summed ``dr2``
    hides: at ``dr2 = 9e-11`` on a magnetic cell the total energy was still
    1.15e-8 Ry out, because the charge half is weighted by ``1/|G+Q|^2`` and
    the magnetic half is not. In an ultracell the gap is *wider* than in a unit
    cell rather than the same -- the smallest ``|G+Q|`` is ``N`` times smaller,
    so the charge half's weight at the envelope's own wavevector is ``N^2``
    larger, and a spin density wave is exactly a state whose whole answer lives
    in the half that is not amplified. Returns ``(total, charge, magnetic)``.
    """
    points = ultracell.points
    volume = ultracell.volume(cell)
    residual_g = jnp.fft.fftn(residual, axes=(-3, -2, -1)) / points
    residual_g = jnp.where(keep, residual_g, 0.0)

    # **Which array is the charge depends on the representation and not on the
    # number of components.** A collinear density is stored ``(up, down)`` and
    # its charge is the sum; a noncollinear one is stored ``(n, m_x, m_y, m_z)``
    # and its charge is already the first component. Summing the four would add
    # the magnetization into the charge, which is dimensionally fine, silently
    # wrong, and invisible on any cell whose moment is small.
    components = residual.shape[0]
    charge_g = residual_g[0] if components == 4 else jnp.sum(residual_g, axis=0)
    charge = float(
        0.5 * volume * E2 * FPI
        * jnp.sum(jnp.real(jnp.conj(charge_g) * charge_g) * g2_inverse)
    )
    if components == 1:
        return charge, charge, 0.0

    # ``rho_ddot``'s ``of_g(:, 2:nspin)`` slice: **all three** magnetization
    # components for a noncollinear density, the single difference for a
    # collinear one, each at the constant weight and each keeping its
    # ``G + Q = 0`` term.
    moment_g = (
        residual_g[1:] if components == 4 else (residual_g[0] - residual_g[1])[None]
    )
    weight = E2 * FPI / (2.0 * np.pi) ** 2
    magnetic = float(
        0.5 * volume * weight * jnp.sum(jnp.real(jnp.conj(moment_g) * moment_g))
    )
    return charge + magnetic, charge, magnetic


def run_ultracell(
    system,
    pseudos,
    ground_state,
    supercell,
    kgrid=(1, 1, 1),
    nbnd: int | None = None,
    external=None,
    magnetic_field=None,
    seed_magnetization=None,
    conv_thr: float = 1.0e-8,
    states_conv_thr: float | None = None,
    david: int | None = None,
    max_iterations: int = 200,
    mixing_mode: str = "anderson",
    mixing_beta: float | None = None,
    mixing_ndim: int | None = None,
    kerker: bool = True,
    state_batch: int | None = 1,
    keep_states: bool = True,
    verbose: bool = False,
) -> UltracellResult:
    """Converge a modulation over ``supercell`` unit cells.

    Args:
        system: the **unit cell**. Its atoms are where the ultracell's atoms
            are: nothing here moves them.
        pseudos: its pseudopotentials.
        ground_state: the converged unit-cell
            :class:`~defumat.scf.driver.SCFResult` the frozen basis is built
            on. Everything below is an expansion around *this* state.
        supercell: ``(n1, n2, n3)``, how many unit cells the ultracell is. The
            ultracell lattice vectors are derived from these -- a non-integer
            ultracell is refused, where Elk's ``avecu`` is an independent input
            with no consistency check against ``ngridq`` at all.
        kgrid: the mesh over the **ultracell's** Brillouin zone, which is ``N``
            times smaller than the unit cell's. The folded set the frozen states
            are computed on is ``supercell * kgrid``.
        states_conv_thr: how tightly the frozen states are diagonalised. They
            *are* the basis, so this bounds everything, and the default is the
            ordinary fixed-density one (1e-6, from which QE's rule makes
            ``ethr = 0.1 conv_thr / nelec``). Tighten it for a null test and
            **not** blindly for a large ``nbnd``: the Davidson budget is per
            k-point, and asking for 1e-12 on bands high in the empty manifold
            spends the whole of it and warns rather than converging.
        david: ``diago_david_ndim`` for the one diagonalisation this does. The
            default subspace is ``4 nbnd`` and is **not** capped against the
            number of plane waves, so a large ``nbnd`` -- which is exactly what
            an accurate ultracell basis is -- needs this lowered: ``4 nbnd``
            must stay below ``npw``.
        nbnd: bands per folded k-point. Pass it: this is the one knob the
            method's accuracy depends on, and the insulating default gives no
            empty bands at all, which leaves the envelope nothing to be built
            from.
        external: an applied potential in Ry, either ``(*box)`` on the ultracell
            grid or a callable taking ``(..., 3)`` **unit-cell** crystal
            coordinates -- which run over ``[0, n_i)`` across the ultracell, so
            a modulation of one ultracell period is ``cos(2 pi x_i / n_i)``. This is what
            a screening calculation applies and what the supercell comparison
            reproduces through
            :func:`~defumat.ultracell.potential.with_external_potential`.
        magnetic_field: an applied field ``B(r)`` in Ry, which is what drives a
            modulated magnetization -- nothing in an SCF breaks spin symmetry on
            its own -- and what it induces is the ``Q``-resolved spin
            susceptibility. **A scalar for ``nspin = 2`` and a vector for
            ``nspin = 4``**: either an array, ``(*box)`` or ``(3, *box)``, or a
            callable of unit-cell crystal coordinates returning one number or
            three per point. The noncollinear form is the one that can *turn* --
            a field whose direction rotates from cell to cell drives a helical
            spin density wave, which a collinear run cannot express at all.
        seed_magnetization: the texture the loop **starts** in, as a factor
            ``s(r)`` multiplying the converged unit cell's own magnetization --
            ``starting_magnetization``'s role, one level up, and ``|s| <= 1``
            for the same reason. A scalar for ``nspin = 2`` and a vector for
            ``nspin = 4``, in the shapes ``magnetic_field`` takes, so a helix of
            pitch ``n`` cells is
            ``lambda x: stack([cos(2 pi x[..., 0] / n), sin(...), zeros], -1)``.
            **This is the other way to get a modulated magnetization and the two
            are different quantities**: a field drives one and measures the
            ``Q``-resolved susceptibility, where a seed hands the loop an
            initial condition and lets it keep it, which is the ordered state
            itself. Nothing is faded -- the mixed quantity here is the density,
            so a seed is not a term in the functional and the convergence test
            is untouched, where Elk seeds its *potential* and has to take the
            seed away again with ``reducebf``. See
            :mod:`defumat.ultracell.seed`.
        keep_states: keep the frozen states and the envelope amplitudes on the
            result, as :class:`~defumat.ultracell.states.UltracellStates`. They
            are what an ultracell *wavefunction* is made of, so an STM image
            (:func:`~defumat.workflows.ultracell.run_ultracell_stm`) or a
            tunnelling transmission
            (:func:`~defumat.workflows.ultracell.run_ultracell_transport`)
            needs them and cannot rebuild them -- the diagonalisation that made
            them is the expensive step of the whole method. Keeping them raises
            no peak, because both arrays are alive for the whole loop in any
            case; ``False`` drops them once it is over.

    **Two thresholds set the floor and neither of them is this one.** The frozen
    states are eigenstates of the density the *unit-cell* SCF stopped at, and
    they are diagonalised to ``states_conv_thr``, so with no modulation at all
    the first iteration's residual is the larger of those two and not round-off.
    Converge the unit cell tighter than the ultracell, and read a null against
    the pair rather than against machine precision.
    """
    started = time.time()
    if int(max_iterations) < 1:
        raise ValueError('max_iterations must be at least 1')
    # **Refused before anything is built, not at the first mix.** ``kerker``
    # defaults to ``True``, so a caller who changed only ``mixing_mode`` would
    # otherwise reach the guard inside ``AdaptiveMixer.mix`` after a ground
    # state and a supercell had already been paid for. ``accepts_precondition``
    # is a class attribute, so this asks the registry rather than an instance.
    if kerker and not MIXERS[str(mixing_mode).lower()].accepts_precondition:
        raise ValueError(
            f"mixing_mode = {mixing_mode!r} does not take a preconditioner and "
            f"kerker is on. Its step is pointwise in real space with a factor "
            f"per point, where box_kerker is one scalar in G-space; pass "
            f"kerker=False to use it, or choose a mixer that screens. On an "
            f"ultracell the two answer different directions -- Kerker the "
            f"long-wavelength charge a longer box makes worse, this the rigid "
            f"spin rotation that has no restoring force at all"
        )
    supercell = tuple(int(n) for n in supercell)
    reference = ground_state
    pseudos = tuple(pseudos)
    cell = system.cell

    basis = build_basis(system)
    require_an_ultracell_regime(system, pseudos, basis)
    # **The frozen states are only a basis if they are eigenstates of a
    # converged density**, and nothing below this line would notice if they
    # were not: ``fixed_density_states`` takes whatever density it is handed.
    # The ``Calculator`` route guards this and the functional one did not,
    # which is ``OPEN.md`` Part V item 2's "a refusal that one caller has and
    # its sibling does not" -- so it is asked for here, where both doors pass.
    if not bool(getattr(reference, "converged", True)):
        raise ValueError(
            "an ultracell is expanded in the unit cell's own Kohn-Sham states, "
            "so it needs a converged ground state; this one stopped at an "
            f"accuracy of {float(getattr(reference, 'accuracy', float('nan'))):g} "
            f"Ry after {int(getattr(reference, 'iterations', 0))} iterations. "
            "Every eigenvalue and every coefficient the envelope is built from "
            "comes from that density, and there is no later iteration to fix "
            "it -- the density of the *unit cell* is what is held fixed. "
            "Converge it first, with a looser conv_thr, more max_iterations or "
            "a different mixing"
        )
    # **A unit cell converged under a field is not a basis this can expand
    # around**, and the reason is the same one ``fixed_density_states`` gives
    # one layer down. The frozen eigenvalues carry whatever field the SCF ended
    # with -- which ``reducebf`` and the fixed-spin-moment scheme both make
    # different from the input -- while ``dV`` subtracts a potential rebuilt
    # here from the density alone. The two would then disagree by a Zeeman
    # term, and an ultracell would report a modulation that is a rigid shift.
    if getattr(reference, "magnetic_field", None) is not None:
        raise NotImplementedError(
            "the ultracell refuses a ground state converged under a magnetic "
            "field or a constrained moment: the frozen eigenvalues carry the "
            "field the SCF *ended* with, which reducebf and the "
            "fixed-spin-moment scheme both change from the input, and the dV "
            "this subtracts is rebuilt from the density alone -- so the two "
            "differ by a Zeeman term that would read as a modulation. Converge "
            "the unit cell without a field and apply the modulation here, "
            "through magnetic_field=, which is the quantity a Q-resolved "
            "susceptibility is anyway"
        )
    ultracell = Ultracell.build(supercell, basis.dense.grid, precision=system.kpoints.precision)
    k0, folded = folded_kpoints(ultracell, kgrid, cell, precision=system.kpoints.precision)

    # A caller-built k-set is a ``for_spin`` boundary: every constructor applies
    # the unpolarized degeneracy factor unconditionally, and a k-set that
    # reaches a polarized run without this counts every electron twice.
    k0 = for_spin(k0, int(system.nspin))
    folded = for_spin(folded, int(system.nspin))

    from defumat.workflows.nscf import fixed_density_states

    calculation, folded_system, eigenvalues, wavefunctions = fixed_density_states(
        system, pseudos, jnp.asarray(reference.density), kpoints=folded, nbnd=nbnd,
        # **The frozen states of a PAW run need the converged ``becsum`` and not
        # only the converged density.** The one-centre potential is a functional
        # of ``becsum`` rather than of ``rho``, so a fixed-density solve handed
        # nothing rebuilds ``ddd_paw`` from the *atomic* occupations and returns
        # eigenvalues of a Hamiltonian this loop never subtracts -- the whole
        # expansion is around a state that is not the ground state, and the
        # tiled null would fail by the one-centre energy rather than by
        # round-off.
        becsum=tuple(getattr(reference, "becsum", ()) or ()),
        **({} if states_conv_thr is None else {"conv_thr": states_conv_thr}),
        david=david,
    )

    cells = ultracell.cells
    nk0 = k0.nk
    # **Three numbers, not one** (``CLAUDE.md``): ``nspin`` says which regime is
    # in force, ``npol`` is how many spinor components a *wavefunction* has, and
    # ``nspin_mag`` how many components a *density* has. ``blocks`` is a fourth
    # and it belongs to this method: how many independent matrices there are per
    # ``k0``. Two for a collinear run, because a collinear potential is diagonal
    # in spin; **one** for a spinor run, on a space twice as large.
    nspin = int(system.nspin)
    npol = int(system.npol)
    nspin_mag = int(system.nspin_mag)
    blocks = 2 if nspin == 2 else 1
    nbnd = int(eigenvalues.shape[-1])
    npwx = int(wavefunctions.shape[-1]) // npol

    eigenvalues = np.asarray(eigenvalues).reshape(blocks, nk0, cells, nbnd)
    coefficients = jnp.asarray(wavefunctions).reshape(
        blocks, nk0, cells, nbnd, npol * npwx
    )
    del wavefunctions
    # **The padding is zeroed here rather than trusted.** Every plane-wave array
    # is padded to a common ``npwx`` and a padded entry points at ``G = 0``, so
    # a non-zero one would be scattered onto that ``Q``-row's own ``G = 0``
    # component of the ultracell box -- a charge added at the longest wavelength
    # the cell has, which is the one place this method cannot afford noise. The
    # eigensolver does leave them zero; this makes that a property of the input
    # rather than an assumption about the solver.
    #
    # A spinor state is ``[c_up, c_down]`` of length ``2 npwx``, so the mask is
    # the doubled one -- ``SpinorHamiltonian._as_state`` of it. Masking with the
    # single copy would zero the *first half of the down component* and leave
    # the padding of the up one alone, which is neither of the two things the
    # mask is for.
    mask = np.asarray(calculation.basis.planewaves.mask)
    padding = mask.reshape(nk0, cells, npwx)
    if npol == 2:
        mask = np.concatenate([mask, mask], axis=-1)
    coefficients = coefficients * jnp.asarray(
        mask.reshape(1, nk0, cells, 1, npol * npwx)
    )

    # **A degenerate cut is only fatal where the cut is, and that was
    # measured rather than assumed.** ``nbnd = 8`` on silicon -- twice the four
    # occupied bands -- cut a multiplet with a 2.6e-14 Ry gap and did not
    # converge at all; ``nbnd = 32``, ``48`` and ``80`` each cut one just as
    # exactly (6e-15, 5e-12, 4.7e-11 Ry) and each converged in ten iterations to
    # a monotonically improving answer. The difference is *where*: the response
    # is carried by the low empty bands, so an arbitrary rotation among them
    # changes what the basis spans, and one thirty bands up changes nothing that
    # carries weight. The warning is therefore gated on the cut lying inside
    # that group -- no more empty bands than occupied ones; the gap itself
    # is on the result either way
    # (:attr:`UltracellResult.multiplet_gap`) and the non-convergence warning
    # names it, so a high cut is reported without being cried over.
    gap = multiplet_cut(jnp.asarray(eigenvalues))
    # A spinor band holds **one** electron where a scalar band holds two, so
    # the count of bands the electrons need is ``nelec`` rather than half of it
    # -- the same factor ``for_spin`` takes out of the k-point weights, arriving
    # here instead as a band count.
    occupied = int(np.ceil(float(calculation.nelec) / (2.0 / npol)))
    if nspin == 2:
        # A polarized channel can hold more bands than half the electrons --
        # fully polarized, the majority channel holds all of them -- so the
        # "no more empty bands than occupied ones" gate has to be read per
        # channel. ``nelup``/``neldw`` are only meaningful when the moment is
        # constrained; unconstrained they fall back to ``nelec/2`` each, which
        # is what the maximum reduces to.
        occupied = max(
            occupied,
            int(np.ceil(float(calculation.nelup))),
            int(np.ceil(float(calculation.neldw))),
        )
    if gap < DEGENERATE_CUT and nbnd <= 2 * occupied:
        warnings.warn(
            f"the nbnd = {nbnd} truncation cuts a degenerate multiplet "
            f"somewhere on the folded k-set (smallest gap {gap:.2e} Ry), and "
            f"the cut is among the low empty bands -- {occupied} of these "
            f"{nbnd} are needed to hold the electrons. Which member of the "
            f"multiplet the eigensolver returned is arbitrary, so the "
            f"ultracell basis is arbitrary exactly where the response lives "
            f"and the fixed point is not a function of the density: on this "
            f"cell that is what stopped an nbnd = 8 run converging at all, "
            f"where nbnd = 24 took nine iterations. Raise nbnd until the cut "
            f"falls in a gap",
            stacklevel=2,
        )

    box_index = jnp.asarray(_box_indices(ultracell, calculation, nk0))
    grid = ultracell.grid

    # **The augmented layer, and it is built once.** The ``N`` displaced tables
    # and the projections of the frozen states are both functions of the basis
    # alone, so neither moves as the modulation converges -- the same sentence
    # that makes the frozen states a basis at all. What the loop rebuilds is the
    # integral of ``dV`` against those tables, which is ``newd`` ``N`` times.
    augmentation = build_ultracell_augmentation(calculation, ultracell, cell)
    becp = None
    if augmentation is not None:
        becp = ultracell_projections(
            coefficients, calculation.projectors, npol=npol
        )
    # The unit cell's own one-centre coefficients, which every frozen eigenvalue
    # already carries: what the matrix takes is the ultracell's *minus* these,
    # for exactly the reason ``delta_potential`` subtracts the unit cell's
    # ``v_scf``. Put the whole one-centre operator in instead and every
    # eigenvalue is doubled in it.
    reference_ddd = None
    if calculation.is_paw:
        _, reference_ddd = calculation.paw.energy_and_coefficients(
            tuple(reference.becsum), axis=calculation.quantization_axis,
        )

    cell_mask = np.zeros(basis.dense.grid, dtype=bool)
    cell_mask.reshape(-1)[np.asarray(basis.dense.fft_index)] = True
    keep = jnp.asarray(ultracell.reciprocal_mask(cell_mask))
    g2_inverse = hartree_kernel(ultracell, cell, np.asarray(keep))

    rho_core_tiled = None
    if calculation.rho_core is not None:
        rho_core_tiled = ultracell.tile(jnp.asarray(calculation.rho_core))
    reference_potential = calculation.potential(jnp.asarray(reference.density)).v_scf

    external_field = _as_field(
        external, magnetic_field, ultracell, nspin, nspin_mag
    )
    # The field's half of that, kept apart because its energy leaves the total
    # again where the scalar potential's stays in it -- see
    # :func:`~defumat.ultracell.energy.ultracell_energy`.
    magnetic_potential = _magnetic_potential(magnetic_field, ultracell, nspin_mag)

    weights0 = np.asarray(k0.weights)
    # **A pair sum over the nuclei, and the nuclei do not move**, so this enters
    # the ultracell's energy the way the Ewald term does: the unit cell's value,
    # unchanged, per cell. Nothing refuses a van der Waals correction here, so a
    # D2 dataset reaches this loop and its total would otherwise be short by a
    # real energy.
    dispersion = float(getattr(calculation, "dispersion", 0.0) or 0.0)
    nelec = float(calculation.nelec)
    smearing = system.occupations not in (None, "fixed")

    density = ultracell.tile(jnp.asarray(reference.density))
    assert density.shape[0] == nspin_mag, (
        f"the reference density has {density.shape[0]} components where "
        f"nspin_mag is {nspin_mag}"
    )
    # **The tiled state is an exact fixed point, so a modulation has to be put
    # in by hand or driven.** This is the first of the two doors: the loop
    # starts in the texture and keeps it, because the seed is an initial
    # condition rather than a term in the functional -- the whole of
    # :mod:`defumat.ultracell.seed`, including why nothing is faded.
    seed_field = _seed_field(seed_magnetization, ultracell, nspin, nspin_mag)
    seed_axis = None
    if seed_field is not None:
        _warn_if_a_constraint_fights_the_seed(system, seed_field, nspin_mag)
        moment = spin_components(density, nspin_mag)[1]
        if nspin_mag == 4:
            # Read off the density being seeded rather than off the system's
            # ``angle1``/``angle2``: what the frozen states carry is the
            # direction the SCF *ended* at, and those two differ by whatever
            # the unit cell's own relaxation did.
            seed_axis = reference_axis(moment, float(cell.volume))
        density = seeded_density(density, seed_field, nspin_mag,
                                 float(cell.volume), axis=seed_axis)
    # ``get_mixer`` drops a ``None`` keyword, so an unset history leaves the
    # mixer its own default and ``mixing_mode = "linear"``, which has no
    # history at all, is not a TypeError.
    mixer = get_mixer(mixing_mode, beta=mixing_beta, history=mixing_ndim)
    if kerker:
        # A cell ``N`` times longer carries a smallest ``|G+Q|`` that is ``N``
        # times smaller, so the Hartree kernel there is ``N^2`` larger and the
        # sloshing an ordinary SCF merely tolerates becomes the whole problem.
        # This is why Elk's own example runs at ``beta0 = 0.001`` -- which is
        # this mixer's parameter, in Elk's own adaptive scheme.
        mixer.precondition = box_kerker(
            ultracell, cell, nelec, (nspin_mag,) + grid, beta=mixer.beta,
        )
    # **PAW's second mixed variable.** The one-centre potential is a functional
    # of ``becsum`` and not of ``rho``, so it has to be part of the state the
    # loop carries rather than something rebuilt from the density -- which is
    # exactly why ``mix_rho.f90`` mixes ``rho%bec`` under ``okpaw`` and nothing
    # else does. An ultrasoft dataset needs no such thing: its ``D`` is an
    # integral of the potential, and the potential is a functional of the
    # density the box already carries.
    becsum_state = ()
    if augmentation is not None:
        becsum_state = _tiled_becsum(reference.becsum, cells)
        if seed_field is not None:
            # **Most of a transition metal's moment is inside the projector
            # spheres**, so a seed that reached the grid alone would seed almost
            # nothing on a PAW magnet and would leave the first iteration's
            # one-centre potential disagreeing with its density. Same field,
            # read at each atom's own grid point.
            becsum_state = seeded_becsum(
                becsum_state, seed_field, ultracell,
                np.asarray(system.structure.positions_crystal(cell)),
                augmentation.tables[0].species_atoms, nspin_mag,
                axis=seed_axis,
            )

    result = None
    history = []
    energies = []

    for iteration in range(1, max_iterations + 1):
        potential = ultracell_potential(
            density, ultracell, cell, rho_core_tiled, calculation.functional,
            g2_inverse, keep,
        )
        delta_v = delta_potential(
            potential.v_scf, reference_potential, ultracell, external_field
        )

        # **How many matrices there are per ``k0`` is the regime.** For a
        # collinear run nothing in ``dV`` couples the two channels -- a
        # collinear potential is diagonal in spin -- so the ``(2 N nbnd)``
        # problem is two ``(N nbnd)`` ones, each with its own ``dV[s]``, and the
        # only thing the channels share is the Fermi level the occupations are
        # found at. For a **spinor** run there is one matrix and it is not
        # block diagonal in anything: the potential's off-diagonal Pauli terms
        # mix the components at every point of the grid, which is what lets the
        # magnetization turn from one cell of the ultracell to the next.
        # **The augmented half of the same matrix element.** ``newd`` once per
        # Q-difference against the displaced tables, plus -- for PAW -- the
        # one-centre coefficients of every atom copy, transformed to the same
        # difference index and with the unit cell's subtracted at ``Q_d = 0``.
        # The two are added before either reaches the matrix, exactly as
        # ``add_paw_to_deeq`` adds them in the unit cell.
        deeq, ddd_state = None, None
        if augmentation is not None:
            deeq = ultracell_deeq(delta_v, augmentation, grid)
            if calculation.is_paw:
                _, coefficients_in = calculation.paw.energy_and_coefficients(
                    as_onecentre_becsum(becsum_state),
                    axis=calculation.quantization_axis,
                )
                ddd_state = from_onecentre_coefficients(coefficients_in, cells)
                deeq = deeq + onecentre_deeq(
                    ddd_state, reference_ddd, augmentation, nspin_mag
                )
            if npol == 2:
                # ``newd_so``: the components are recombined into spin blocks
                # and sandwiched between ``fcoef`` **after** PAW's one-centre
                # coefficients have been added to them, which is the order
                # ``_noncollinear_coefficients`` uses in the unit cell and the
                # reason both additions above happen first. Adding them to the
                # sandwiched matrix instead puts a one-centre term into the
                # wrong spin structure, which converges perfectly well to the
                # wrong answer.
                deeq = spinor_ultracell_deeq(
                    deeq, calculation.fcoef_matrix, float(system.soc_scale),
                )

        levels = np.empty((blocks, nk0, cells * nbnd))
        vectors = [[] for _ in range(blocks)]
        for spin in range(blocks):
            channel = delta_v[spin] if blocks == 2 else delta_v
            for ik in range(nk0):
                extra = None
                if deeq is not None:
                    extra = (
                        spinor_augmentation_matrix(
                            becp[spin, ik], deeq, augmentation.difference,
                        )
                        if npol == 2 else
                        augmentation_matrix(
                            becp[spin, ik], deeq[spin], augmentation.difference,
                        )
                    )
                matrix = ultracell_matrix(
                    coefficients[spin, ik], jnp.asarray(eigenvalues[spin, ik]),
                    box_index[ik], channel, grid, batch=state_batch, npol=npol,
                    augmentation=extra,
                )
                values, states = jnp.linalg.eigh(matrix)
                levels[spin, ik] = np.asarray(values)
                vectors[spin].append(states)

        occupations, fermi = _occupy(
            levels, weights0, nelec, cells, system, smearing, calculation,
        )

        # **The density is the occupations and nothing else**, which is worth
        # having in one function rather than two: an STM image is this same
        # accumulation with a smeared delta at the tip energy in place of
        # ``occupations`` (``PLAN.md`` P89), so a second copy of the loop here
        # would be a second place for the normalisation to drift.
        new = ultracell_band_density(
            coefficients, vectors, occupations, box_index, grid,
            float(cell.volume), npol=npol, nspin_mag=nspin_mag,
            batch=state_batch, dtype=density.dtype,
        )

        # **The charge inside the projector spheres, which no state carries.**
        # ``becsum`` here is resolved by Q-difference rather than by atom copy,
        # because that is the form the displaced tables consume and because the
        # copies' own occupations are its transform rather than the other way
        # round. PAW needs them per copy as well, and that is where the loop
        # takes them apart again.
        becsum_out, becsum_copies, augmentation_residual = (), None, 0.0
        if augmentation is not None:
            becsum_out = _ultracell_becsum(
                becp, vectors, occupations, augmentation, blocks, nk0, nspin_mag,
                fcoef=(
                    tuple(so.fcoef for so in calculation.spin_orbit)
                    if npol == 2 else None
                ),
            )
            charge, augmentation_residual = ultracell_augmentation_charge(
                becsum_out, augmentation, grid, nspin_mag
            )
            new = new + charge.astype(new.dtype)
            # **The pairing across ``d`` is checked where it is exact and
            # reported where it is not**, and the two halves are different
            # quantities. ``becsum`` per copy involves no truncation -- it is a
            # sum over the whole Q-set -- so it is real to round-off and
            # anything above the tolerance is an index order. The augmented
            # *density* is truncated to the tiled dense sphere, which is
            # **not closed under negation** at a non-zero ``Q``, so a real
            # function represented on it is not exactly real and what is dropped
            # is a statement about ``ecutrho``. See
            # :attr:`UltracellResult.augmentation_residual`.
            if calculation.is_paw:
                becsum_copies, worst = _becsum_copies(becsum_out, augmentation)
                if worst > _BECSUM_REAL_TOLERANCE and iteration == 1:
                    warnings.warn(
                        f"an atom copy's becsum came back complex, by "
                        f"{worst:.2e} relative, where the sum over the whole "
                        f"Q-set makes it real with no truncation anywhere in "
                        f"it. That is the array itself being wrong rather than "
                        f"a convergence problem -- a wrong Q-pairing does not "
                        f"show here and is caught by a supercell comparison -- "
                        f"and PAW's one-centre terms built from it are wrong "
                        f"rather than noisy",
                        stacklevel=2,
                    )
            if augmentation_residual > _AUGMENTATION_REAL_TOLERANCE and iteration == 1:
                warnings.warn(
                    f"the ultracell's augmented density came back complex by "
                    f"{augmentation_residual:.2e} relative, which is far above "
                    f"the sphere truncation this quantity normally carries "
                    f"(1.7e-4 on silicon at ecutrho = 4 ecutwfc, falling as "
                    f"ecutrho^-2.2 and exactly zero at N = 1). At this size "
                    f"something other than the cutoff is wrong with the "
                    f"augmentation charge",
                    stacklevel=2,
                )

        accuracy, charge_dr2, magnetic_dr2 = _accuracy(
            new - density, ultracell, cell, g2_inverse, keep
        )

        # **The energy of the state this iteration produced**, which is not the
        # density the next one will start from. ``eband`` carries the potential
        # the matrix was built from, so ``deband`` pairs ``new`` with
        # ``potential`` -- the input one -- and the Hartree and
        # exchange-correlation energies are taken at ``new`` instead, which is
        # one more ``v_of_rho`` and is what makes QE's ``descf`` unnecessary:
        # every iteration's number is then the Kohn-Sham energy of a state that
        # exists, rather than a mixture's corrected to first order.
        band = float(np.sum(occupations * levels))
        potential_out = ultracell_potential(
            new, ultracell, cell, rho_core_tiled, calculation.functional,
            g2_inverse, keep,
        )
        entropy = 0.0
        if smearing:
            entropy = ultracell_entropy(levels, weights0, fermi, ultracell, system)
        # **PAW's two terms, and they are the unit cell's with the copies summed
        # and divided by ``N``.** The one-centre energy is taken at the output
        # ``becsum`` where the Hartree and exchange-correlation energies are
        # taken at the output density, and ``deband`` subtracts the one-centre
        # potential at the *input* ``becsum`` contracted against the output one
        # -- the pairing ``delta_e`` uses, for the reason it uses it.
        onecentre, paw_deband = 0.0, 0.0
        if calculation.is_paw and becsum_copies is not None:
            epaw, _ = calculation.paw.energy_and_coefficients(
                as_onecentre_becsum(becsum_copies),
                axis=calculation.quantization_axis,
            )
            onecentre = float(epaw) / cells
            paw_deband = _paw_deband(ddd_state, becsum_copies, cells)
        terms = ultracell_energy(
            new, potential, potential_out, band, ultracell, cell,
            float(calculation.ewald), magnetic_potential, entropy,
            dispersion=dispersion, onecentre=onecentre, paw_deband=paw_deband,
        )
        energy = total_of(terms)

        history.append((accuracy, charge_dr2, magnetic_dr2))
        # A **second** list rather than a fourth column, because ``history`` is
        # unpacked as a triple in this file and its shape is documented in the
        # user guide: widening it would break the non-convergence branch below
        # rather than extend it.
        energies.append(energy)
        if verbose:
            extra = "" if nspin_mag == 1 else (
                f"   (charge {charge_dr2:.3e}, magnetic {magnetic_dr2:.3e})"
            )
            print(f"  iteration {iteration:3d}   dr2 = {accuracy:.6e} Ry   "
                  f"E = {energy:.8f} Ry{extra}")

        converged = accuracy < conv_thr
        if not converged:
            # **One packed vector, because a mixer with a history cannot be
            # asked to mix two states independently** -- two Anderson instances
            # would each extrapolate along their own residual and the pair they
            # produced would not be a step of anything. ``_mix`` packs ``rho``,
            # ``becsum`` and ``ns`` the same way in the unit cell, and the
            # preconditioner reads the density's shape to know where the tail
            # starts: Kerker is an approximate inverse Jacobian and has nothing
            # to say about a quantity that lives on the atoms rather than on the
            # grid, which gets the plain ``beta`` instead.
            head_in, head_out = np.asarray(density), np.asarray(new)
            if calculation.is_paw and becsum_copies is not None:
                vector = mixer.mix(
                    np.concatenate([head_in.ravel(), _flatten_becsum(becsum_state)]),
                    np.concatenate([head_out.ravel(), _flatten_becsum(becsum_copies)]),
                )
                size = head_in.size
                density = jnp.asarray(
                    vector[:size].reshape(head_in.shape), dtype=density.dtype
                )
                becsum_state = _unflatten_becsum(vector[size:], becsum_copies)
            else:
                density = jnp.asarray(
                    mixer.mix(head_in, head_out), dtype=density.dtype
                )
        else:
            density = new
            if calculation.is_paw and becsum_copies is not None:
                becsum_state = becsum_copies

        if converged:
            result = _result(
                density, delta_v, levels, occupations, fermi, True, iteration,
                (accuracy, charge_dr2, magnetic_dr2), gap, ultracell, reference,
                started, history, float(cell.volume), blocks, band, terms,
                energies, augmentation_residual,
            )
            break

    if result is None:
        total, charge_dr2, magnetic_dr2 = history[-1]
        # **Which half is still moving is the first question and it used to be
        # unanswerable.** On a magnetic cell the two are weighted an order
        # apart (``OPEN.md`` Y1), so a run that has converged its charge and is
        # still hunting for its moment looks exactly like one that has
        # converged neither -- and the fixes are different: the second wants a
        # tighter seed or more iterations, the first wants Kerker.
        halves = "" if nspin_mag == 1 else (
            f" (charge {charge_dr2:.3e}, magnetization {magnetic_dr2:.3e}; "
            f"the magnetization half carries no 1/|G+Q|^2 weight, so on a "
            f"magnetic cell it is the one that decides)"
        )
        # **A noncollinear cell has a soft direction a collinear one does not,
        # and the advice it needs is the opposite of the reflex.** Turning every
        # moment rigidly costs no energy without spin-orbit coupling or
        # anisotropy, so the ``Q = 0`` transverse magnetization has no restoring
        # force and the fixed point sits in a nearly flat manifold that the
        # mixer has to *traverse*. A smaller ``mixing_beta`` traverses it more
        # slowly, so lowering it makes this worse rather than better -- measured
        # on a four-cell hydrogen ultracell under a weak (0.002 Ry) turning
        # field, where ``beta = 0.7`` converges in 263 iterations and ``0.3``
        # does not converge in 300 (``OPEN.md`` Part VI item 3). At a field five
        # times stronger every value converges and the best is 0.5. So the
        # message asks for iterations, or a stronger field, and says explicitly
        # not to reach for a smaller beta.
        rigid = "" if nspin_mag != 4 else (
            f" This is a noncollinear run, where turning every moment together "
            f"costs no energy without spin-orbit coupling -- so the Q = 0 "
            f"transverse magnetization has no restoring force, the solution "
            f"sits in a nearly flat manifold, and the mixer has to traverse it. "
            f"That is slow when the field pinning the direction is weak: raise "
            f"max_iterations (currently {max_iterations}), or pin the direction "
            f"harder with a stronger magnetic_field. **Do not lower "
            f"mixing_beta** (currently {mixer.beta:g}) -- it was measured to "
            f"make this worse, 0.3 failing in 300 iterations where 0.7 "
            f"converged in 263 on the same cell."
        )
        warnings.warn(
            f"the ultracell loop did not converge: dr2 = {total:.3e} Ry"
            f"{halves} after {max_iterations} iterations, against conv_thr = "
            f"{conv_thr:g}. An ultracell sloshes at long wavelength -- the "
            f"smallest |G+Q| it carries is N times smaller than the unit "
            f"cell's -- so check that Kerker is on before raising the "
            f"iteration budget. A truncation that cuts a degenerate multiplet "
            f"(multiplet_gap = {gap:.2e} Ry here) is the other way this stalls, "
            f"and the fix for that one is a larger nbnd rather than more "
            f"iterations.{rigid}",
            stacklevel=2,
        )
        result = _result(
            density, delta_v, levels, occupations, fermi, False, max_iterations,
            history[-1], gap, ultracell, reference, started, history,
            float(cell.volume), blocks, band, terms, energies,
            augmentation_residual,
        )
    if keep_states:
        # **The last iteration's amplitudes, which are the ones the density
        # came from.** Both branches above leave ``vectors`` and ``levels`` at
        # the state ``result`` was packed from, converged or not, so an image
        # built from these is an image of the density that was reported.
        result.states = UltracellStates(
            coefficients=coefficients,
            vectors=jnp.stack([jnp.stack(block, axis=0) for block in vectors],
                              axis=0),
            eigenvalues=np.asarray(levels),
            box_index=np.asarray(box_index), padding=padding,
            k0_crystal=np.asarray(k0.crystal(cell)), weights=weights0,
            ultracell=ultracell, cell=cell, npol=npol, nspin=nspin,
        )
    return result


def _result(density, delta_v, levels, occupations, fermi, converged, iterations,
            accuracy, gap, ultracell, reference, started, history, cell_volume,
            blocks, band, terms, energies,
            augmentation_residual=0.0) -> UltracellResult:
    """Pack the loop's state, squeezing the spin axis the way every result does.

    One matrix block drops the leading channel axis from the eigenvalues and the
    occupations (``CLAUDE.md``'s rule for every result object), so a stage 1
    caller sees exactly the ``(nk0, N nbnd)`` arrays it saw before spin existed.
    The axis that is squeezed is the **matrix** one and not the density's, which
    is why the argument is ``blocks``: a noncollinear run has four density
    components and *one* block, and its eigenvalues are one list per ``k0`` in
    which every state is a spinor. The **density** keeps its axis either way,
    because it is fed straight back into ``tile`` and the potential, both of
    which want it.
    """
    total, charge_dr2, magnetic_dr2 = accuracy
    if blocks == 1:
        levels, occupations = levels[0], occupations[0]
    return UltracellResult(
        density=density, delta_v=delta_v, eigenvalues=levels,
        occupations=occupations,
        fermi_energy=float(fermi) if np.ndim(fermi) == 0 else tuple(
            float(f) for f in np.atleast_1d(np.asarray(fermi))
        ),
        band_energy=band,
        total_energy=total_of(terms),
        energy_terms={k: v for k, v in terms.items() if not k.startswith("_")},
        converged=converged, iterations=iterations,
        accuracy=total, charge_accuracy=charge_dr2,
        magnetic_accuracy=magnetic_dr2, multiplet_gap=gap,
        augmentation_residual=float(augmentation_residual),
        ultracell=ultracell, reference=reference,
        seconds=time.time() - started, history=tuple(history),
        energy_history=tuple(energies),
        # Keyed on there *being* a field, not on its energy being non-zero:
        # ``SCFResult`` does the same (``None if field is None``), and a field
        # whose Zeeman energy happens to vanish -- one perpendicular to every
        # moment, which a turning field passes through -- is not the same thing
        # as no field at all.
        field_energy=(None if terms.get("_field_energy") is None
                      else float(terms["_field_energy"])),
        cell_volume=cell_volume,
    )


def _occupy(levels, weights, nelec, cells, system, smearing, calculation):
    """One Fermi level over all ``N nbnd`` ultracell states at every ``k0``.

    **The occupation is done for the ultracell and rescaled afterwards**, which
    is one rule where doing it per unit cell would need two. The search runs
    with the k0 weights as they are and ``N * nelec`` electrons -- the number
    the ultracell actually holds -- and the returned weights are divided by
    ``N`` at the end to bring the density back to one cell.

    Doing it the other way round does not work, and the reason is that the two
    occupation schemes read their electron count differently:
    ``fixed_occupations`` uses it to *count bands to fill*, where
    ``smeared_occupations`` uses it as the *weighted sum* the Fermi level must
    reproduce. Pre-dividing the weights by ``N`` would then need ``N nelec`` in
    one branch and ``nelec`` in the other, and getting that backwards is a
    factor of ``N`` in the density that converges perfectly well. Elk's
    ``occupyulr`` takes the other convention (``wkpt/nkpa`` against the unit
    cell's ``chgval``) and can afford to, having only the smeared branch.
    """
    eigenvalues = jnp.asarray(levels)
    weights = jnp.asarray(weights)
    total = nelec * cells
    # **One Fermi level over both channels and all ``N nbnd`` states**, which
    # is the unconstrained collinear rule and the only one that lets the
    # magnetization move: a spin density wave is electrons crossing between the
    # channels from one cell to the next, and two separate Fermi levels forbid
    # exactly that. The constrained case is the other branch below, and it is
    # ``occupations = 'fixed'`` (QE's rule: LSDA with fixed occupations *needs*
    # ``tot_magnetization``) or an explicit ``tot_magnetization`` under
    # smearing -- which is the ``Q = 0`` moment constraint, applied to the
    # uniform component of the envelope and to nothing else.
    counts = None
    if int(system.nspin) == 2 and bool(getattr(calculation, "two_fermi_energies", False)):
        counts = (float(calculation.nelup) * cells, float(calculation.neldw) * cells)
    if smearing:
        # The smeared branch needs no degeneracy argument: it searches for the
        # level whose *weighted sum* reproduces the electron count, and the
        # factor of two is already out of the weights (``for_spin``).
        wg, ef = smeared_occupations(
            eigenvalues, weights, total,
            float(system.degauss), system.smearing or "gaussian", counts=counts,
        )
        return np.asarray(wg) / cells, ef
    # **The fixed branch does need it, and that asymmetry is the trap.** It
    # counts *bands to fill* rather than matching a weighted sum, so it has to
    # be told how many electrons one band holds -- two for a scalar band, and
    # **one** for a spinor, where there is no second spin state to put an
    # electron in. This is the ``for_spin`` factor arriving a second time, in
    # the band count instead of the k-point weights, and
    # ``driver.py:_occupations`` applies exactly this rule
    # (``degeneracy = 1 if noncolin else 2``).
    #
    # It went missing here and nothing saw it, because **every null this phase
    # ran used smearing**: the two branches read their electron count
    # differently, so the argument that is wrong in one is absent from the
    # other. What found it was the first spin-orbit cell, an iodine atom with
    # seven valence electrons and ``occupations = 'fixed'``, where it raises
    # "7.0 electrons cannot fill 2-fold bands" -- loudly, which is the one
    # merciful thing about it.
    degeneracy = 1 if int(system.nspin) == 4 else 2
    wg, homo, lumo = fixed_occupations(
        eigenvalues, weights, total, degeneracy, counts=counts
    )
    return np.asarray(wg) / cells, homo


def _on_the_box(field, ultracell: Ultracell, what: str):
    """One applied field as a real ``(*box)`` array, callable or not."""
    if callable(field):
        grid = ultracell.grid
        axes = [np.arange(m) / n for m, n in zip(grid, ultracell.cell_grid)]
        coordinates = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        field = field(coordinates)
    field = jnp.asarray(field)
    if field.shape != tuple(ultracell.grid):
        raise ValueError(
            f"the applied {what} is on {tuple(field.shape)} and the "
            f"ultracell box is {tuple(ultracell.grid)}"
        )
    return jnp.real(field)


def _as_field(external, magnetic_field, ultracell: Ultracell, nspin: int,
              nspin_mag: int):
    """The applied fields as one ``(nspin_mag, *box)`` addition to ``dV``.

    A scalar potential is felt in full by both channels of a collinear run and
    by the charge component alone of a noncollinear one -- ``set_vrs``'s rule,
    and :func:`~defumat.scf.potential.as_potential_components` is where it is
    written down.

    A **magnetic field** is where the two regimes differ, and the sign is the
    same one twice. Collinear: ``v_up -= B`` and ``v_dw += B``, which is
    ``add_bfield.f90:237-238``. Noncollinear: ``v(:, 2:4) -= B``, which is the
    same routine's line 244 -- and in the ``(v_0, B_x, B_y, B_z)`` layout the
    two say exactly the same thing, since ``v_up = v_0 + v_z``. Both are the
    sign the Zeeman energy ``-B . m`` gives when differentiated. Everything is
    in Ry and ``B`` is the field times the Bohr magneton, which is what QE's
    ``bfield`` already is.

    **A noncollinear field is a vector and a collinear one is a number**, so
    the argument changes shape with the regime: ``(*box)`` for ``nspin = 2``,
    and ``(3, *box)`` for ``nspin = 4`` -- or a callable returning ``(..., 3)``
    from crystal coordinates, which is how a *rotating* field is written and is
    the thing a collinear run cannot express at all. A field that turns from
    cell to cell is what drives a helical spin density wave, and it is the
    reason the noncollinear regime is worth having here.

    **Why the field is the interesting one for stage 3 and the potential was
    for stage 1.** Nothing in an SCF breaks spin symmetry on its own, so an
    unpolarized unit cell put in an ultracell stays unpolarized however many
    cells it has -- the modulated magnetization has to be *driven*, either by a
    modulated field (this) or by a seed the loop is allowed to keep
    (``seed_magnetization``, :mod:`defumat.ultracell.seed`). The two are
    different quantities rather than two routes to one: what a field induces is
    the ``Q``-resolved spin susceptibility, a response, and what a seed keeps is
    an ordered state.
    """
    if magnetic_field is not None and nspin_mag == 1:
        # **Two different cells land here and the advice is different.** An
        # ``nspin = 1`` run has one density and nothing for a field to split. A
        # spin-orbit run on a *nonmagnetic* cell has spinor wavefunctions and
        # still ``nspin_mag = 1``: its density, its potential and the frozen
        # reference potential all have one component, and inventing three more
        # here would leave ``dV`` and the eigenvalues it is subtracted from in
        # different representations.
        if nspin == 4:
            raise ValueError(
                "an applied magnetic field needs a noncollinear run that "
                "carries a magnetization: this one has nspin_mag = 1 (a "
                "spin-orbit calculation with no magnetization), so its "
                "density and its frozen potential are a single component and "
                "there is nothing for a field to be added to. Give the unit "
                "cell a small starting_magnetization so that domag is true -- "
                "it may converge back to nearly zero, which is fine, and the "
                "Q-resolved susceptibility this measures is then the Pauli one"
            )
        raise ValueError(
            "an applied magnetic field needs two spin channels to split: this "
            "run is nspin = 1, where there is one density and the field has "
            "nothing to act on. Set nspin = 2 in the unit cell"
        )
    if external is None and magnetic_field is None:
        return None

    grid = tuple(ultracell.grid)
    field = jnp.zeros((nspin_mag,) + grid, dtype=ultracell.precision.real)
    if external is not None:
        scalar = _on_the_box(external, ultracell, "potential")
        field = field + (
            jnp.zeros_like(field).at[0].add(scalar) if nspin_mag == 4
            else scalar[None]
        )
    magnetic = _magnetic_potential(magnetic_field, ultracell, nspin_mag)
    if magnetic is not None:
        field = field + magnetic
    return field


def _seed_field(seed, ultracell: Ultracell, nspin: int, nspin_mag: int):
    """The magnetization seed as one array on the box, in the field's shapes.

    ``(*box)`` for a collinear run and ``(3, *box)`` for a noncollinear one,
    through the *same* two evaluators an applied field goes through, so the
    coordinate convention cannot come apart between a run driven by a field and
    one started from a seed. What the array means is
    :mod:`defumat.ultracell.seed`'s subject and not this function's.

    The refusal is the field's, for the field's two reasons: at ``nspin_mag = 1``
    there is one density component and no magnetization for a seed to scale --
    which is an ``nspin = 1`` run, or a spin-orbit run on a cell that carries no
    moment.
    """
    if seed is None:
        return None
    if nspin_mag == 1:
        if nspin == 4:
            raise ValueError(
                "a magnetization seed scales the unit cell's own moment, and "
                "this run has nspin_mag = 1 (a spin-orbit calculation with no "
                "magnetization): there is no moment to scale and no "
                "magnetization component of the density to put one in. Give "
                "the unit cell a small starting_magnetization so that domag is "
                "true, converge it, and seed the modulation on top of that"
            )
        raise ValueError(
            "a magnetization seed needs two spin channels: this run is "
            "nspin = 1, where there is one density and no magnetization at "
            "all. Set nspin = 2 in the unit cell, or noncolin for a texture "
            "whose direction turns"
        )
    if nspin_mag == 4:
        return _on_the_vector_box(seed, ultracell)
    return _on_the_box(seed, ultracell, "magnetization seed")


def _warn_if_a_constraint_fights_the_seed(system, seed, nspin_mag: int) -> None:
    """A constrained moment holds ``Q = 0`` and the seed generally moves it.

    ``tot_magnetization`` gives each channel its own Fermi level, which pins
    ``N_up - N_dw`` at the target every iteration -- so what a seed can deliver
    under it is the modulation at ``Q != 0`` and not its own net, and an
    amplitude wave's net is zero where the constraint's is not. The two are
    consistent and the loop resolves them by keeping the constraint, which is
    worth saying once rather than leaving to be noticed in the moments.
    """
    if getattr(system, "tot_magnetization", None) is None:
        return
    values = np.asarray(seed, dtype=float)
    mean = float(np.mean(values)) if nspin_mag == 2 else float(
        np.linalg.norm(np.mean(values.reshape(3, -1), axis=1))
    )
    if abs(mean - 1.0) < 1.0e-6:
        return
    warnings.warn(
        f"this run constrains the moment (tot_magnetization = "
        f"{float(system.tot_magnetization):g}), so each channel has its own "
        f"Fermi level and N_up - N_dw is pinned at that target at every "
        f"iteration. The seed's own net is {mean:.3f} of the reference's, so "
        f"what it can deliver here is the modulation at Q != 0 and not the net "
        f"moment -- the constraint wins that one, by construction. Drop "
        f"tot_magnetization to let the seed's net stand",
        stacklevel=3,
    )


def _magnetic_potential(magnetic_field, ultracell: Ultracell, nspin_mag: int):
    """The applied field's own contribution to ``dV``, on its own.

    Factored out of :func:`_as_field` because the **energy** needs it apart
    from the scalar potential's, and the two go opposite ways: a scalar
    ``external`` potential is ``vltot`` and its energy stays in the total,
    while a field's Zeeman energy is carried beside it, by this package's
    convention and QE's and Elk's. The way to keep it out is to pair it into
    ``deband``, which needs exactly this array (``ultracell/energy.py``).
    """
    if magnetic_field is None:
        return None
    grid = tuple(ultracell.grid)
    field = jnp.zeros((nspin_mag,) + grid, dtype=ultracell.precision.real)
    if nspin_mag == 4:
        b = _on_the_vector_box(magnetic_field, ultracell)
        return field.at[1:].add(-b)
    b = _on_the_box(magnetic_field, ultracell, "magnetic field")
    return field.at[0].add(-b).at[1].add(b)


def _on_the_vector_box(field, ultracell: Ultracell):
    """A noncollinear applied field as a real ``(3, *box)`` array.

    The callable form takes **unit-cell crystal coordinates** shaped
    ``(..., 3)`` -- running over ``[0, n_i)`` across the ultracell, so one
    ultracell period along axis ``i`` is ``2 pi x_i / n_i`` -- and returns the
    three cartesian components of ``B`` at each point, shaped ``(..., 3)``.
    That is the natural way to write a field that turns:
    ``lambda x: B * stack([cos(2 pi x[..., 2] / n), sin(...), zeros], -1)``.
    An array is ``(3, *box)``, components first, like every other field here.
    """
    grid = tuple(ultracell.grid)
    if callable(field):
        axes = [np.arange(m) / n for m, n in zip(grid, ultracell.cell_grid)]
        coordinates = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        values = jnp.asarray(field(coordinates))
        if values.shape != grid + (3,):
            raise ValueError(
                f"a noncollinear magnetic field callable must return the three "
                f"cartesian components at every point, shaped {grid + (3,)}; "
                f"this one returned {tuple(values.shape)}"
            )
        return jnp.real(jnp.moveaxis(values, -1, 0))
    values = jnp.asarray(field)
    if values.shape != (3,) + grid:
        raise ValueError(
            f"a noncollinear applied magnetic field is on {(3,) + grid} -- the "
            f"three cartesian components over the ultracell box -- and this one "
            f"is {tuple(values.shape)}. A collinear run takes a scalar B(r); a "
            f"noncollinear one takes a vector, which is the whole difference"
        )
    return jnp.real(values)
