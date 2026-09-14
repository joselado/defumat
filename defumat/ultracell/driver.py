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
per folded k-point and nothing else. **The convergence in ``nbnd`` carries no
sign**: Rayleigh-Ritz bounds the eigenvalues of a *fixed* Hamiltonian, and this
one moves with its own truncated density, so an ultracell eigenvalue is not an
upper bound on the supercell's. Only a total energy would be variational, and
there is none here (stage 4; Elk has none either -- ``energyulr.f90`` is the
eigenvalue sum alone).

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

**Nothing breaks spin symmetry on its own**, so an unpolarized unit cell put in
an ultracell stays unpolarized however many cells it has. A modulated
magnetization has to be driven, and ``magnetic_field`` is what drives it: what
comes back is then the ``Q``-resolved response rather than an initial
condition. Elk's other route -- a random seed field (``rndbfcu``) faded away by
``reducebf``, which lets a *spontaneous* wave find its own period -- is not
written here.

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
from defumat.scf.mixing import get_mixer
from defumat.xc.functional import resolve_functional
from defumat.scf.occupations import fixed_occupations, smeared_occupations
from defumat.system.kpoints import for_spin
from defumat.ultracell.density import ultracell_density
from defumat.ultracell.grid import Ultracell, folded_kpoints
from defumat.ultracell.hamiltonian import multiplet_cut, ultracell_matrix
from defumat.ultracell.mixing import box_kerker
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
    #: the only energy an ultracell run reports in either code.
    band_energy: float
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
    #: The **unit cell's** volume in bohr^3, carried so that an integral over
    #: one cell of the ultracell can be taken without the caller holding the
    #: system as well as the result.
    cell_volume: float = 1.0

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
        """``rho_up - rho_dw`` on the box, per unit cell. Zero for ``nspin = 1``.

        The envelope a spin density wave lives in, and the reason the phase has
        a stage 3 at all: it is what a modulated field induces, what Elk's
        ``rndbfcu`` seed is looking for, and what ``magnetic_accuracy`` bounds.
        """
        if self.density.shape[0] == 1:
            return jnp.zeros_like(self.density[0])
        return self.density[0] - self.density[1]

    def cell_moments(self) -> np.ndarray:
        """``(N,)`` the moment of each unit cell of the ultracell, in mu_B/2.

        The integral of ``rho_up - rho_dw`` over one cell ``R``, which is the
        quantity a spin density wave is *read off*: its sign alternates with the
        envelope and its profile is the wave. ``None`` of it needs a sphere --
        the cells tile the ultracell exactly, so the partition is the geometry
        rather than a choice of radius, which is the one thing an ultracell has
        over an atom-resolved moment.
        """
        shape = self.ultracell.shape
        cell_grid = self.ultracell.cell_grid
        m = np.asarray(self.magnetization)
        # Split each axis into (cell index, point inside the cell) and sum the
        # inside. The box index is ``J_i = n_i G_i + q_i`` in reciprocal space,
        # but in *real* space a cell boundary falls every ``Nd_i`` points, so
        # the reshape is (n_i, Nd_i) in that order and not the other.
        m = m.reshape(shape[0], cell_grid[0], shape[1], cell_grid[1],
                      shape[2], cell_grid[2])
        per_cell = m.sum(axis=(1, 3, 5))
        element = float(self.cell_volume) / float(np.prod(cell_grid))
        return (per_cell * element).reshape(-1)

    def __repr__(self) -> str:
        n = "x".join(str(m) for m in self.ultracell.shape)
        state = "converged" if self.converged else "NOT converged"
        levels = np.atleast_1d(np.asarray(self.fermi_energy, dtype=float))
        fermi = ", ".join(f"{e * RY_TO_EV:.4f}" for e in levels)
        return (
            f"UltracellResult({n} cells, {self.iterations} iterations, {state}, "
            f"dr2 = {self.accuracy:.3e} Ry, E_F = {fermi} eV)"
        )


def require_an_ultracell_regime(system, pseudos, basis) -> None:
    """Refuse every regime stage 1 has not been measured in, by name.

    Called **before** the frozen states are computed, not after: that
    diagonalisation is the expensive step of the whole method, and a run
    that is going to be refused should not pay for it first. Everything
    checked here is available from the system, its pseudopotentials and its
    basis, so none of it needs a ``Calculation`` to have been built.
    """
    require_an_ultracell_functional(
        resolve_functional([p.functional for p in pseudos], system.input_dft)
    )
    if any(p.is_ultrasoft or p.is_paw for p in pseudos):
        raise NotImplementedError(
            "the ultracell refuses ultrasoft and PAW datasets (PLAN.md P88, "
            "stage 1 is norm-conserving). The augmentation charge is a function "
            "of the density through D_ij, so the frozen unit-cell states stop "
            "being a fixed basis the moment the modulation moves, and S enters "
            "every ultracell overlap"
        )
    if int(system.nspin) == 4:
        raise NotImplementedError(
            "the ultracell refuses noncollinear magnetism and spin-orbit "
            "coupling (PLAN.md P88; nspin = 2 is stage 3a and is in, nspin = 4 "
            "is not). The reason is the shape of the matrix rather than the "
            "physics: a collinear ultracell matrix is block diagonal in spin, "
            "so it is two of the same build with dV[0] and dV[1], where a "
            "spinor state is one vector of 2 npwx components and the potential "
            "acts on it as V_0 + sigma . B -- a different matrix element, with "
            "the two components transformed separately and recombined. Use "
            "nspin = 2 for a spin density wave along an axis"
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
        raise NotImplementedError(
            "the ultracell needs the smooth and dense grids to coincide, which "
            "they do for a norm-conserving dataset at ecutrho = 4 ecutwfc. With "
            "a double grid the wavefunctions and the density live on different "
            "ultracell boxes and the interpolation between them is not written"
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

    charge_g = jnp.sum(residual_g, axis=0)
    charge = float(
        0.5 * volume * E2 * FPI
        * jnp.sum(jnp.real(jnp.conj(charge_g) * charge_g) * g2_inverse)
    )
    if residual.shape[0] == 1:
        return charge, charge, 0.0

    moment_g = residual_g[0] - residual_g[1]
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
    conv_thr: float = 1.0e-8,
    states_conv_thr: float | None = None,
    david: int | None = None,
    max_iterations: int = 200,
    mixing_mode: str = "anderson",
    mixing_beta: float = 0.7,
    mixing_ndim: int | None = None,
    kerker: bool = True,
    state_batch: int | None = 1,
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
        magnetic_field: a **collinear** applied field ``B(r)`` in Ry, in the
            same two forms, which enters as ``v_up -= B`` and ``v_dw += B``
            (``add_bfield.f90``'s ``i_cons = 4``). Needs ``nspin = 2``: it is
            what drives a modulated magnetization, since nothing in a collinear
            SCF breaks spin symmetry on its own, and what it induces is the
            ``Q``-resolved spin susceptibility.

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
        **({} if states_conv_thr is None else {"conv_thr": states_conv_thr}),
        david=david,
    )

    cells = ultracell.cells
    nk0 = k0.nk
    nspin = int(system.nspin)
    nbnd = int(eigenvalues.shape[-1])
    npwx = int(wavefunctions.shape[-1])

    eigenvalues = np.asarray(eigenvalues).reshape(nspin, nk0, cells, nbnd)
    coefficients = jnp.asarray(wavefunctions).reshape(nspin, nk0, cells, nbnd, npwx)
    del wavefunctions
    # **The padding is zeroed here rather than trusted.** Every plane-wave array
    # is padded to a common ``npwx`` and a padded entry points at ``G = 0``, so
    # a non-zero one would be scattered onto that ``Q``-row's own ``G = 0``
    # component of the ultracell box -- a charge added at the longest wavelength
    # the cell has, which is the one place this method cannot afford noise. The
    # eigensolver does leave them zero; this makes that a property of the input
    # rather than an assumption about the solver.
    coefficients = coefficients * jnp.asarray(
        np.asarray(calculation.basis.planewaves.mask).reshape(1, nk0, cells, 1, npwx)
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
    occupied = int(np.ceil(float(calculation.nelec) / 2.0))
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

    cell_mask = np.zeros(basis.dense.grid, dtype=bool)
    cell_mask.reshape(-1)[np.asarray(basis.dense.fft_index)] = True
    keep = jnp.asarray(ultracell.reciprocal_mask(cell_mask))
    g2_inverse = hartree_kernel(ultracell, cell, np.asarray(keep))

    rho_core_tiled = None
    if calculation.rho_core is not None:
        rho_core_tiled = ultracell.tile(jnp.asarray(calculation.rho_core))
    reference_potential = calculation.potential(jnp.asarray(reference.density)).v_scf

    external_field = _as_field(external, magnetic_field, ultracell, nspin)

    weights0 = np.asarray(k0.weights)
    nelec = float(calculation.nelec)
    smearing = system.occupations not in (None, "fixed")

    density = ultracell.tile(jnp.asarray(reference.density))
    # ``get_mixer`` drops a ``None`` keyword, so an unset history leaves the
    # mixer its own default and ``mixing_mode = "linear"``, which has no
    # history at all, is not a TypeError.
    mixer = get_mixer(mixing_mode, beta=mixing_beta, history=mixing_ndim)
    if kerker:
        # A cell ``N`` times longer carries a smallest ``|G+Q|`` that is ``N``
        # times smaller, so the Hartree kernel there is ``N^2`` larger and the
        # sloshing an ordinary SCF merely tolerates becomes the whole problem.
        # This is why Elk's own example runs at ``beta0 = 0.001``.
        mixer.precondition = box_kerker(
            ultracell, cell, nelec, (nspin,) + grid, beta=mixing_beta,
        )
    result = None
    history = []

    for iteration in range(1, max_iterations + 1):
        potential = ultracell_potential(
            density, ultracell, cell, rho_core_tiled, calculation.functional,
            g2_inverse, keep,
        )
        delta_v = delta_potential(
            potential.v_scf, reference_potential, ultracell, external_field
        )

        # **The matrix is block diagonal in spin and is built one block at a
        # time.** Without spin-orbit coupling nothing in ``dV`` couples the two
        # channels -- a collinear potential is diagonal in spin -- so the
        # ``(2 N nbnd)`` problem is two ``(N nbnd)`` ones, each with its own
        # ``dV[s]``, and the only thing the channels share is the Fermi level
        # the occupations are found at. That is why ``ultracell_matrix`` needs
        # no spin argument: it is called twice.
        levels = np.empty((nspin, nk0, cells * nbnd))
        vectors = [[] for _ in range(nspin)]
        for spin in range(nspin):
            for ik in range(nk0):
                matrix = ultracell_matrix(
                    coefficients[spin, ik], jnp.asarray(eigenvalues[spin, ik]),
                    box_index[ik], delta_v[spin], grid, batch=state_batch,
                )
                values, states = jnp.linalg.eigh(matrix)
                levels[spin, ik] = np.asarray(values)
                vectors[spin].append(states)

        occupations, fermi = _occupy(
            levels, weights0, nelec, cells, system, smearing, calculation,
        )

        new = jnp.zeros((nspin,) + grid, dtype=density.dtype)
        for spin in range(nspin):
            for ik in range(nk0):
                new = new.at[spin].add(
                    ultracell_density(
                        coefficients[spin, ik], vectors[spin][ik],
                        jnp.asarray(occupations[spin, ik]),
                        box_index[ik], grid, float(cell.volume), batch=state_batch,
                    )
                )

        accuracy, charge_dr2, magnetic_dr2 = _accuracy(
            new - density, ultracell, cell, g2_inverse, keep
        )
        history.append((accuracy, charge_dr2, magnetic_dr2))
        if verbose:
            extra = "" if nspin == 1 else (
                f"   (charge {charge_dr2:.3e}, magnetic {magnetic_dr2:.3e})"
            )
            print(f"  iteration {iteration:3d}   dr2 = {accuracy:.6e} Ry{extra}")

        converged = accuracy < conv_thr
        if not converged:
            density = jnp.asarray(
                mixer.mix(np.asarray(density), np.asarray(new)), dtype=density.dtype
            )
        else:
            density = new

        if converged:
            result = _result(
                density, delta_v, levels, occupations, fermi, True, iteration,
                (accuracy, charge_dr2, magnetic_dr2), gap, ultracell, reference,
                started, history, float(cell.volume), nspin,
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
        halves = "" if nspin == 1 else (
            f" (charge {charge_dr2:.3e}, magnetization {magnetic_dr2:.3e}; "
            f"the magnetization half carries no 1/|G+Q|^2 weight, so on a "
            f"magnetic cell it is the one that decides)"
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
            f"iterations",
            stacklevel=2,
        )
        result = _result(
            density, delta_v, levels, occupations, fermi, False, max_iterations,
            history[-1], gap, ultracell, reference, started, history,
            float(cell.volume), nspin,
        )
    return result


def _result(density, delta_v, levels, occupations, fermi, converged, iterations,
            accuracy, gap, ultracell, reference, started, history, cell_volume,
            nspin) -> UltracellResult:
    """Pack the loop's state, squeezing the spin axis the way every result does.

    ``nspin = 1`` drops the leading channel axis from the eigenvalues and the
    occupations (``CLAUDE.md``'s rule for every result object), so a stage 1
    caller sees exactly the ``(nk0, N nbnd)`` arrays it saw before spin
    existed. The **density** keeps its axis, because it is fed straight back
    into ``tile`` and the potential, both of which want it.
    """
    total, charge_dr2, magnetic_dr2 = accuracy
    band = float(np.sum(occupations * levels))
    if nspin == 1:
        levels, occupations = levels[0], occupations[0]
    return UltracellResult(
        density=density, delta_v=delta_v, eigenvalues=levels,
        occupations=occupations,
        fermi_energy=float(fermi) if np.ndim(fermi) == 0 else tuple(
            float(f) for f in np.atleast_1d(np.asarray(fermi))
        ),
        band_energy=band, converged=converged, iterations=iterations,
        accuracy=total, charge_accuracy=charge_dr2,
        magnetic_accuracy=magnetic_dr2, multiplet_gap=gap,
        ultracell=ultracell, reference=reference,
        seconds=time.time() - started, history=tuple(history),
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
        wg, ef = smeared_occupations(
            eigenvalues, weights, total,
            float(system.degauss), system.smearing or "gaussian", counts=counts,
        )
        return np.asarray(wg) / cells, ef
    wg, homo, lumo = fixed_occupations(eigenvalues, weights, total, counts=counts)
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


def _as_field(external, magnetic_field, ultracell: Ultracell, nspin: int):
    """The applied fields as one ``(nspin, *box)`` addition to ``dV``.

    A scalar potential is felt in full by both channels; a **collinear magnetic
    field** splits them, ``v_up -= B`` and ``v_dw += B``, which is
    ``add_bfield.f90:237-238`` (``i_cons = 4``) and the same sign the Zeeman
    energy ``-B . m`` gives when differentiated with ``m = rho_up - rho_dw``.
    Both are in Ry, and ``B`` here is the field times the Bohr magneton, which
    is what QE's ``bfield`` already is.

    **Why the field is the interesting one for stage 3 and the potential was
    for stage 1.** Nothing in a collinear SCF breaks spin symmetry on its own,
    so an unpolarized unit cell put in an ultracell stays unpolarized however
    many cells it has -- the modulated magnetization has to be *driven*, either
    by a modulated field (this) or by a seed the loop is allowed to keep (which
    is Elk's ``rndbfcu`` with ``reducebf``, and is not written). A field is the
    honest half: what it induces is the ``Q``-resolved spin susceptibility, a
    quantity rather than an initial condition.
    """
    if magnetic_field is not None and nspin == 1:
        raise ValueError(
            "an applied magnetic field needs two spin channels to split: this "
            "run is nspin = 1, where there is one density and the field has "
            "nothing to act on. Set nspin = 2 in the unit cell"
        )
    if external is None and magnetic_field is None:
        return None

    grid = tuple(ultracell.grid)
    field = jnp.zeros((nspin,) + grid, dtype=ultracell.precision.real)
    if external is not None:
        field = field + _on_the_box(external, ultracell, "potential")[None]
    if magnetic_field is not None:
        b = _on_the_box(magnetic_field, ultracell, "magnetic field")
        field = field.at[0].add(-b).at[1].add(b)
    return field
