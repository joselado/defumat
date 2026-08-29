"""Phonons at ``Gamma``: the force constants as a second derivative of the energy.

``PLAN.md`` P25. This is the other half of what the Sternheimer solver was
built for (P24): the electric field's response gave ``epsilon_infinity`` and the
Born charges, and the *ionic* perturbation gives the dynamical matrix.

**The whole assembly is one derivative of code that already exists.** The force
is ``jax.grad`` of :func:`~pypresso.forces.energy.frozen_energy` with respect to
the positions at a frozen electronic state (P15); the force *constants* are that
same gradient differentiated once more, and the only new ingredient is which
direction to differentiate it in. Write the energy as a function of both the
coordinate and the state, ``L(u, psi)``, chosen so that it is stationary in
``psi`` at the solution -- which is what carrying the orthonormality constraint
with its multipliers buys, and what makes the force a *partial* derivative. Then

    dE/du_j            = d_j L                              (the force, P15)
    d^2E/du_i du_j     = d_i d_j L + (d_psi d_j L) . dpsi_i

exactly: no second-order wavefunction, no ``<dpsi|H - eps S|dpsi>`` term, and no
factor to get right. The second piece is a *directional* derivative of the same
gradient along ``dpsi_i``, so **one ``jvp`` of ``jax.grad(L)`` along the tangent
``(e_i, dpsi_i)`` returns a whole column of the matrix** -- the frozen Hessian
and the electronic response together, in one call, with nothing separating them
by hand.

What that replaces on the Fortran side is most of ``PHonon/PH``:

| QE | here |
|---|---|
| ``dynmat0`` + ``d2ionq`` + ``dvloc`` second derivatives | the ``u`` half of the ``jvp`` |
| ``drhodv`` | the ``psi`` half of the same ``jvp`` |
| ``dvqpsi_us`` (the bare perturbation) | one ``jvp`` through ``at_positions`` |
| ``dv_of_drho`` (the screening kernel) | one ``jvp`` of ``v_of_rho`` |
| ``dyndia`` | mass-weighting and an ``eigh`` |

``solve_linter``'s self-consistent loop, ``symdvscf``, ``symdynph_gq`` and the
diagonalisation are transcribed; the perturbations and the second derivative are
not.

**The trap this phase adds is the third appearance of ``abs``.**
:func:`~pypresso.forces.energy.frozen_energy` was written to be differentiated
with respect to the *positions*, and it contained ``jnp.abs(psi)**2`` in its
kinetic and constraint terms because on that path nothing ever differentiated it
with respect to the states. Here something does, and ``abs``'s derivative is
``Re(conj z t)/|z|`` -- ``0/0`` at a coefficient that vanishes, which by symmetry
happens on the nose. It is
:func:`pypresso.scf.density.band_density`'s trap and
:func:`pypresso.basis.gvectors.modulus`'s, in a third place, and the symptom is
a NaN in every force constant rather than a wrong number.

**The symmetrisation is P24's trap with one more index**, and it appears twice
for two different reasons. Inside the self-consistent loop, the response density
of a perturbation on a symmetry-reduced k-set is not what the whole grid would
give, and what has to be averaged is not three scalar fields but ``3 nat``
fields carrying *both* a direction and an atom label
(:meth:`~pypresso.scf.driver.Calculation.symmetrize_atom_displacement`,
``symdvscf``): an operation rotates the displacement and carries it to the atom
it maps onto. That fixes the screening each perturbation sees. It does **not**
fix the wedge sum in the assembled matrix, which is a rank-2 tensor with two
atom indices and needs
:func:`~pypresso.system.symmetry.symmetrize_atom_pair_tensor` (``symdynph_gq``)
-- the same argument that makes ``symvector`` non-cosmetic for a force, two
ranks up. Both are measured in ``PLAN.md`` P25.

**Insulators and metals, and the metal is one weight rather than a routine.**
The frozen energy weights its states by ``wg = wk f``, which is right for the
frozen Hessian and wrong for the electronic half, because a metal's ``dpsi``
already carries ``f`` from ``orthogonalize``'s smeared right-hand side. So the
one ``jvp`` above is two: the coordinate and the density at ``wg``, the states
at ``wk``. That is what ``dynmat_us.f90`` (``wg``) and ``drhodvnl.f90``
(``2 wk``) are, and it is the whole of ``PLAN.md`` P28 --
:func:`_state_weights`. Two-atom aluminium's modes come out at 146.711240 and
311.033545 cm^-1 against ``ph.x``'s 146.710511/146.714378 and 311.035401, and
the acoustic sum rule holds to 1.06e-5 Ry/bohr^2 where the unsplit assembly
violated it by half the spectrum.

**Norm-conserving only, and the reason is in the formula rather than in a
missing routine.** The identity above holds because ``L`` is stationary in
``psi`` at *fixed* multipliers, and the multipliers are attached to the
constraint ``<psi|S(u)|psi> - 1``. Differentiating a second time leaves a term
``-<psi|dS/du_j|psi> . deps_i`` which vanishes identically when ``S`` does not
move with the atoms -- that is, for a norm-conserving dataset, where ``S = 1``.
For an ultrasoft or PAW one it does not vanish, and there is a second gap beside
it: the augmentation charge ``Q_ij(r - tau)`` moves at frozen ``becsum``, which
``addusdynmat`` and ``drhodvus`` account for. Both are refused by name
(:func:`require_norm_conserving`), and the measurement behind the refusal is the
same shape as ``zstar_eu_us``'s: with the guard lifted, ultrasoft silicon's
optical mode comes out at **-504.3 cm^-1** against ``ph.x``'s **+513.3** --
imaginary where the crystal is stable -- from a run that converges and gives a
cubic, symmetric matrix.

**What this costs in memory**, since a design is not finished until its peak
working set is known: ``3 nat`` bare perturbations and ``3 nat`` first-order
wavefunctions are held at once, each ``(nspin, nk, nocc, npwx)`` complex. On the
two-atom silicon of ``si-epsilon.in`` that is 2 MB; on a 16-atom cell with 100
k-points and 3000 plane waves it is 7 GB, and the way down is QE's -- solve one
irreducible representation at a time, which cuts the count as well as the
storage. The bare perturbations are stored rather than recomputed because the
self-consistent loop re-uses them at every iteration, which is the same trade
:mod:`pypresso.response.efield` makes for its three.

**An ultrasoft or PAW run adds three more arrays of that same shape and two
smaller ones**, and they are all built once rather than per iteration: the
occupied block ``dpsi^ort`` (``3 nat`` of ``(nspin, nk, nocc, npwx)``, so the
state storage goes from two such sets to three), the mixed state's own change
``drhous`` (``3 nat`` dense grids, which is the *density* shape and an order
smaller), and ``becsumort`` plus ``dLambda``, which are ``nh^2`` and ``nbnd^2``
per mode and negligible beside either. Held together the ultrasoft peak is
about **1.5x** the norm-conserving one at the same cell -- 3 MB on
``si-epsilon-us.in``. PAW adds the one-centre response beside ``dvscf`` in the
mixer, ``3 nat`` times ``(nspin_mag, nkb, nkb)``, which is kilobytes.

References for the method rather than the code: Baroni, de Gironcoli, Dal Corso
and Giannozzi, *Rev. Mod. Phys.* **73**, 515 (2001), whose Eq. (14) is the
identity above.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from pypresso.forces.energy import FrozenState, frozen_energy
from pypresso.response.efield import require_a_symmetrisable_response
from pypresso.response.mixing import DEFAULT_RESPONSE_MIXING, ResponseMixer
from pypresso.response.sternheimer import (
    SternheimerSolver,
    paw_response,
    require_a_sternheimer_regime,
    smearing_of,
)
from pypresso.response.velocity import over_kpoints
from pypresso.batching import map_k
from pypresso.system.symmetry import atom_mapping, cartesian_rotations
from pypresso.units import AMU_TO_RY, RY_TO_CMM1, RY_TO_THZ

__all__ = ["Phonons", "DisplacementResponse", "dynamical_matrix",
           "require_norm_conserving", "self_consistent_response",
           "orthogonality_states", "non_variational_response",
           "multiplier_response", "overlap_derivatives",
           "symmetrize_becsum_modes"]

#: QE's ``alpha_mix(1)``: the weight the mixer gives the residual. It is no
#: longer the *whole* of the mixing -- :mod:`pypresso.response.mixing` builds an
#: Anderson history on top of it -- which is what makes 0.7 a safe default here
#: rather than a value each system has to be tuned to.
ALPHA_MIX = 0.7

#: Convergence on ``|ddv_scf|^2``. QE's default ``tr2_ph`` is 1e-12.
TR2 = 1.0e-14

MAX_ITERATIONS = 60


@dataclass
class Phonons:
    """The force constants at ``Gamma``, and what they diagonalise to."""

    #: ``(nat, 3, nat, 3)`` in Ry/bohr^2 -- ``d^2E/du_(a i) du_(b j)``,
    #: hermitised and symmetrised over the crystal's point group. This is the
    #: matrix QE writes to ``fildyn``, in its own units and layout.
    matrix: np.ndarray
    #: ``(3 nat,)`` in cm^-1, signed: an imaginary frequency is reported as a
    #: negative number, exactly as ``dyndia`` prints one.
    frequencies: np.ndarray
    #: ``(3 nat, 3 nat)`` displacement patterns, columns matching
    #: :attr:`frequencies`, in the mass-weighted basis ``eigh`` returns.
    eigenvectors: np.ndarray
    #: ``(nat, 3, nspin_mag, n1, n2, n3)``: the induced charge density per unit
    #: displacement, symmetrised, on the dense grid. Summed over the atoms it is
    #: the ground-state density's own gradient, which is what
    #: :func:`self_consistent_response` checks.
    induced_density: np.ndarray | None = None
    #: ``max|D - D^T|`` after the group average and **before** the
    #: hermitisation, in Ry/bohr^2. The force constants are symmetric exactly,
    #: so this is a measurement and not a construction: it is the free
    #: diagnostic of whether the linear solves converged and whether the
    #: response half of the derivative is right, and nothing else here shows
    #: either. Measured *before* the group average it would show the wedge
    #: residue instead, which is large and says nothing.
    asymmetry: float = 0.0
    #: ``|ddv_scf|^2`` at each iteration of the self-consistent loop.
    history: list = field(default_factory=list)
    #: Mean CG iterations per band per solve, QE's ``av.it.``.
    average_iterations: float = 0.0
    converged: bool = False

    @property
    def omega2(self) -> np.ndarray:
        """``(3 nat,)`` eigenvalues of the mass-weighted matrix, in Ry^2."""
        signs = np.sign(self.frequencies)
        return signs * (self.frequencies / RY_TO_CMM1) ** 2

    @property
    def frequencies_thz(self) -> np.ndarray:
        """:attr:`frequencies` in THz, the other unit ``dyndia`` prints."""
        return self.frequencies / RY_TO_CMM1 * RY_TO_THZ

    @property
    def acoustic_residue(self) -> float:
        """The largest of the three lowest frequencies, in cm^-1.

        Translating the whole crystal costs no energy, so three frequencies are
        zero exactly and what comes out instead measures the calculation --
        chiefly the finite plane-wave basis, whose energy depends on where the
        atoms sit relative to the FFT grid. ``ph.x`` prints 2.045258 cm^-1 for
        the silicon of ``si-epsilon.in`` and does **not** impose the sum rule,
        so this number is a target rather than a nuisance: reproducing it means
        the same basis-set error is being made, which is a sharper statement
        than reproducing the optical mode.
        """
        return float(np.max(np.abs(self.frequencies[:3])))


@dataclass
class DisplacementResponse:
    """``solve_linter``'s output: the ``3 nat`` first-order states and densities.

    A carrier, so that a caller which has already solved these does not solve
    them again -- which is the expensive half of both a dynamical matrix and a
    Raman tensor, and they are the *same* half.
    :class:`~pypresso.response.strain.StrainResponse` is the same idea for the
    strain perturbation.
    """

    #: ``(nat, 3, nspin, nk, nocc, ndim)`` -- ``dpsi`` per mode.
    dpsi: object
    #: ``(nat, 3, nspin_mag, n1, n2, n3)`` -- the induced density per mode.
    drho: object
    #: ``|ddv_scf|^2`` per iteration.
    history: list
    converged: bool = True
    #: The bare perturbations, kept only because an ultrasoft or PAW assembly
    #: needs to rebuild ``dLambda`` from the converged perturbation and cannot
    #: do that from ``dpsi`` alone. ``None`` on the norm-conserving path, where
    #: the term it feeds is zero.
    bare: object = None
    #: ``{"dvscf", "onecentre", "dbecsum"}`` at convergence, for the same
    #: reason.
    extras: dict | None = None


def dynamical_matrix(
    calculation,
    wavefunctions,
    eigenvalues,
    density,
    becsum=(),
    alpha_mix: float = ALPHA_MIX,
    tr2: float = TR2,
    max_iterations: int = MAX_ITERATIONS,
    threshold: float = 1.0e-12,
    acoustic_sum_rule: bool = False,
    response: "DisplacementResponse | None" = None,
    verbose: bool = False,
) -> Phonons:
    """The ``Gamma``-point force constants and phonon frequencies.

    Args:
        calculation: the :class:`~pypresso.scf.driver.Calculation` the states
            belong to, with its own symmetry-reduced k-set. A ``nosym`` run is
            accepted only on an **unshifted** grid, for the reason
            :func:`~pypresso.response.efield.require_a_symmetrisable_response`
            gives.
        wavefunctions: ``(nspin, nk, nbnd, ndim)`` from the converged run.
        eigenvalues: ``(nspin, nk, nbnd)`` or the squeezed ``(nk, nbnd)``.
        density: the converged density the fixed potential is built from.
        becsum: unused for a norm-conserving dataset, which is the only kind
            this accepts; present so the signature matches
            :func:`~pypresso.response.efield.dielectric_tensor`.
        acoustic_sum_rule: whether to impose ``sum_b D_(a i)(b j) = 0`` before
            diagonalising. **Off by default, because ``ph.x`` does not impose
            it** and the residue is the diagnostic described in
            :attr:`Phonons.acoustic_residue`.
        response: a :class:`DisplacementResponse` solved earlier, if there is
            one. It is steps 1 and 2 below and the whole cost of this function;
            :func:`~pypresso.response.nonlinear.raman_tensors` solves the same
            object and hands it back through its ``keep_internals``, which is
            what lets a spectrum cost one displacement response rather than two.
    """
    eigenvalues = jnp.asarray(eigenvalues)
    if eigenvalues.ndim == 2:
        eigenvalues = eigenvalues[None]
    require_a_symmetrisable_response(calculation)
    # ``metals = True``: a Fermi surface does not stop a dynamical matrix
    # existing, and the solve handles one (``PLAN.md`` P24c). What it adds is
    # ``ef_shift``, inside the loop below.
    require_a_sternheimer_regime(calculation, metals=True)
    _require_a_moving_overlap_regime(calculation)
    _require_one_spin_channel(calculation)

    structure = calculation.system.structure
    positions = jnp.asarray(structure.positions)

    weights, _ = calculation.occupations(eigenvalues)
    weights = jnp.asarray(weights)
    nocc = int(round(calculation.nelec / 2))
    potential = calculation.potential(density)
    _, ddd_paw = calculation.onecenter(becsum)
    hamiltonians = calculation.hamiltonian(potential.v_scf, ddd_paw)
    solver = SternheimerSolver(
        calculation, hamiltonians, wavefunctions, eigenvalues, weights,
        nocc, threshold, v_scf=potential.v_scf, becsum=becsum,
        smearing=smearing_of(calculation, _fermi_level(calculation, eigenvalues)),
        kpoint_weights=calculation.system.kpoints.weights,
    )

    # 0. What ``S`` moving with the atoms adds, and all of it is zero for a
    #    norm-conserving dataset: the occupied block of the first-order state,
    #    and the mixed state's own change at frozen states (``drho.f90``).
    derivatives = overlap_derivatives(calculation, solver, positions)
    ort = orthogonality_states(calculation, solver, positions)
    (rho_moved, bec_moved), (rho_ort, bec_ort) = non_variational_response(
        calculation, positions, jnp.asarray(wavefunctions), weights,
        jnp.asarray(density), becsum, ort,
    )
    drhous_stacked = moved_stacked = becsumort = bec_moved_sym = None
    if rho_moved is not None:
        moved_stacked = _stack_modes(_symmetrize_modes(calculation, rho_moved))
        drhous_stacked = moved_stacked + _stack_modes(
            _symmetrize_modes(calculation, rho_ort)
        )
        bec_moved_sym = symmetrize_becsum_modes(calculation, bec_moved)
        becsumort = _add_becsum(
            bec_moved_sym, symmetrize_becsum_modes(calculation, bec_ort)
        )

    if response is None:
        # 1. The bare perturbation ``(dH/du - eps dS/du)|psi>``, once per mode
        #    and stored: the loop below drives on it at every iteration.
        bare = _bare_displacements(calculation, solver, potential.v_scf, positions)

        # 2. ``solve_linter``'s loop, one perturbation per (atom, direction).
        dpsi, drho, history, average_iterations, converged, extras = (
            self_consistent_response(
                calculation, solver, bare, density, positions=positions,
                alpha_mix=alpha_mix, tr2=tr2, max_iterations=max_iterations,
                becsumort=becsumort, drhous=drhous_stacked, verbose=verbose,
            )
        )
    else:
        dpsi, drho = response.dpsi, response.drho
        history, converged = response.history, response.converged
        average_iterations = float("nan")
        bare, extras = response.bare, response.extras

    # 2b. The three tangents ``S``'s motion adds to the assembly. ``dLambda``
    #     is a matrix element of the *same* perturbation the last solve was
    #     driven by, rebuilt at the converged ``dV_scf`` --
    #     :func:`~pypresso.response.born._multiplier_response`'s argument, with
    #     a displacement in place of the field.
    multipliers = dbecsum = None
    if calculation.is_ultrasoft and bare is not None:
        multipliers = multiplier_response(
            calculation, solver, bare, extras, weights,
            jnp.asarray(wavefunctions).shape[2], derivatives,
        )
        # The assembly rebuilds the raw mixed-state response itself, so what
        # it is handed is the *symmetrised* total; the difference between the
        # two is the wedge sum's own correction and is zero on a closed grid.
        dbecsum = extras.get("dbecsum")

    # 3. The second derivative, two jvp of the force's own gradient per mode:
    #    the frozen Hessian at ``wg`` and the electronic response at ``wk``.
    matrix = _force_constants(
        calculation, positions, jnp.asarray(wavefunctions), weights,
        _state_weights(solver, weights), eigenvalues, jnp.asarray(density),
        dpsi, drho, solver.nocc, becsum=becsum, dbecsum=dbecsum,
        multipliers=multipliers, ort=ort,
    )
    # ``symdynph_gq`` first and the hermitisation second, which is the order
    # that makes the second one a *measurement*. A column of the raw matrix is a
    # sum over the irreducible wedge, and such a sum is not symmetric in
    # ``(a i) <-> (b j)`` until the group has put back what the reduction left
    # out -- on the ten-point wedge of ``si-epsilon.in`` the raw asymmetry is
    # 5.1e-2 against force constants of 0.28, and essentially all of it is that.
    # After the average it is 2e-16 on every case here, shifted wedge included,
    # so the hermitisation has nothing left to do and what it would have removed
    # is a report on the linear solves. Whether the average *must* leave a
    # symmetric matrix is not claimed -- it is measured.
    matrix = calculation.symmetrize_atom_pair_tensor(matrix)
    asymmetry = float(np.abs(matrix - matrix.transpose(2, 3, 0, 1)).max())
    matrix = 0.5 * (matrix + matrix.transpose(2, 3, 0, 1))
    if acoustic_sum_rule:
        matrix = _impose_acoustic_sum_rule(matrix)

    frequencies, vectors = _diagonalize(matrix, structure.masses)
    return Phonons(
        matrix=np.asarray(matrix),
        frequencies=frequencies,
        eigenvectors=vectors,
        induced_density=np.asarray(drho),
        asymmetry=asymmetry,
        history=history,
        average_iterations=average_iterations,
        converged=converged,
    )


class _Levels:
    """The one field :func:`~pypresso.response.sternheimer.smearing_of` reads.

    :func:`dynamical_matrix` is handed a density and eigenvalues rather than an
    ``SCFResult``, so the Fermi level is recomputed from the occupations the
    calculation itself would assign -- the same call the SCF made, on the same
    eigenvalues, so it is the same number.
    """

    def __init__(self, fermi_energy):
        self.fermi_energy = fermi_energy


def _fermi_level(calculation, eigenvalues):
    if calculation.system.occupations == "fixed":
        return _Levels(None)
    _, levels = calculation.occupations(eigenvalues)
    return _Levels(levels["fermi_energy"])


def self_consistent_response(
    calculation,
    solver,
    bare,
    density,
    positions=None,
    alpha_mix: float = ALPHA_MIX,
    tr2: float = TR2,
    max_iterations: int = MAX_ITERATIONS,
    mixing_mode: str = DEFAULT_RESPONSE_MIXING,
    becsumort=None,
    drhous=None,
    verbose: bool = False,
):
    """``solve_linter``'s loop for the ``3 nat`` displacement patterns.

    Split out from :func:`dynamical_matrix` because the first-order
    wavefunctions have a check the assembled matrix does not offer: **displace
    every atom by the same vector and the response must be the ground-state
    density's own gradient**, ``sum_a drho_(a i) = -d(rho)/dr_i``, because a
    rigid translation of the crystal *is* a translation of its self-consistent
    solution. The identity holds only for the fully screened response -- the
    bare one is 52% off on silicon -- so it tests the loop, the kernel and the
    symmetrisation together against a quantity obtained by differentiating the
    converged density on the grid, which shares no machinery with any of them.
    It is the phonon's counterpart of the ``chi_0`` finite-difference check
    :mod:`pypresso.response.sternheimer` rests on.

    **The modes are independent.** Each perturbation screens only itself, so
    nothing in the physics couples them; the only thing that mixes them is the
    symmetrisation, which is why they are iterated together rather than one
    after another.

    **A metal goes through this loop unchanged**, and that was true one phase
    before the matrix above could consume the result: P24c put ``ef_shift`` and
    ``orthogonalize``'s smearing branch here and the assembly still counted the
    occupation twice, so the loop was kept metallic while
    ``dynamical_matrix`` refused one at the door. That is what let the refusal
    be *measured* against ``ph.x`` rather than asserted, and lifting it (P28)
    changed nothing in this function: the fix was the weight the **matrix**
    contracts ``dpsi`` with, not anything the loop produces
    (:func:`_state_weights`).

    Returns ``(dpsi, drho, history, average_iterations, converged)``, with
    ``dpsi`` an object array of shape ``(nat, 3)``.
    """
    nat = calculation.system.structure.nat
    grid_shape = jnp.asarray(density).shape
    core = _core_charge_response(calculation, density, positions)
    dvscf = jnp.zeros((nat, 3) + grid_shape)
    history, total_iterations, solves = [], 0, 0
    dpsi = np.empty((nat, 3), dtype=object)
    symmetrised = jnp.zeros_like(dvscf)
    converged = False
    # PAW's one-centre coefficients respond too, and they are *not* a function
    # of the density -- they come from ``becsum``. Carried and mixed beside
    # ``dvscf`` exactly as :mod:`pypresso.response.efield` carries its three,
    # which is ``dfpt_kernels``' ``int3_paw`` beside ``dvscfin``.
    onecentre = None if solver.ddd_paw is None else jnp.zeros(
        (nat, 3) + solver.ddd_paw.shape
    )

    mixer = ResponseMixer(mixing_mode, beta=alpha_mix)
    for iteration in range(max_iterations):
        response, becsum_response = [], []
        for atom in range(nat):
            for cart in range(3):
                perturbation = _bare_plus_induced(
                    solver, bare[atom, cart], dvscf[atom, cart], iteration > 0,
                    None if onecentre is None else onecentre[atom, cart],
                )
                solution = solver.solve(perturbation)
                dpsi[atom, cart] = solution.dpsi
                total_iterations += solution.iterations
                solves += 1
                response.append(solver.response_density(solution.dpsi))
                if onecentre is not None:
                    becsum_response.append(solver.response_becsum(solution.dpsi))

        # ``ef_shift``: a displacement at ``q = 0`` moves charge in and out of
        # the cell, so a metal's Fermi level moves with it and the response
        # density has to be corrected by ``def ldos`` before it screens
        # anything. Applied to the raw per-mode densities and *before* the
        # symmetrisation, which is where QE applies it too -- ``ldos`` is a
        # scalar under the group, so symmetrising ``def ldos`` as part of the
        # displacement-labelled vector field is ``sym_def`` by another route.
        shifts = None
        if solver.smearing is not None:
            corrected = [solver.fermi_level_shift(r) for r in response]
            response = [r for r, _ in corrected]
            shifts = [float(d) for _, d in corrected]

        # ``symdvscf``: the 3 nat responses are symmetrised *together*, after
        # the loop over modes and before the kernel, because an operation mixes
        # them -- rotating the direction and permuting the atom.
        stacked = jnp.stack(response).reshape((nat, 3) + grid_shape)
        if drhous is not None:
            # The first-order density the potential responds to is the *whole*
            # of it, and for an ultrasoft or PAW dataset the variational part is
            # not the whole: the augmentation charge moves with its atom and the
            # occupied block of ``dpsi`` is not zero, so ``drho.f90``'s
            # "change at fixed wavefunctions" screens beside it.
            stacked = stacked + drhous
        symmetrised = calculation.symmetrize_atom_displacement(stacked)

        # ``dv_of_drho``: one jvp of the potential this code already writes.
        induced = jnp.stack([
            jax.jvp(
                lambda r: calculation.potential(r).v_scf,
                (jnp.asarray(density),),
                (symmetrised[atom, cart],),
            )[1]
            for atom in range(nat) for cart in range(3)
        ]).reshape(dvscf.shape)

        induced_onecentre = None
        if onecentre is not None:
            # ``dfpt_kernels``: for a *phonon* the one-centre potential
            # responds to ``2 dbecsum + becsumort`` and not to the variational
            # part alone -- the orthogonality correction changes the one-centre
            # occupations, and PAW's energy is a second, independent functional
            # of them. (The grid density is the other way round: ``drhop``
            # there is the variational response only, and ``drhous`` is carried
            # separately into the assembly.) The factor of two in QE's line is
            # a storage convention rather than a term: ``addusdbec``
            # accumulates one of the two cross terms and
            # :meth:`SternheimerSolver.response_becsum` -- a ``jvp``, so
            # ``2 Re`` -- accumulates both.
            per_mode = np.empty((nat, 3), dtype=object)
            index = 0
            for atom in range(nat):
                for cart in range(3):
                    per_mode[atom, cart] = becsum_response[index]
                    index += 1
            if becsumort is not None:
                per_mode = _add_becsum(per_mode, becsumort)
            per_mode = symmetrize_becsum_modes(calculation, per_mode)
            induced_onecentre = jnp.stack([
                paw_response(calculation, per_mode[atom, cart], solver.becsum)
                for atom in range(nat) for cart in range(3)
            ]).reshape(onecentre.shape)

        if core is not None:
            # ``addcore``/``drhoc``: the core charge travels with its atom, so
            # ``v_xc`` changes even at a frozen valence density. It is
            # independent of the iteration -- it is not a response to anything
            # -- so it is built once and added to the induced potential here.
            induced = induced + core

        change = float(jnp.sum((induced - dvscf) ** 2))
        history.append(change)
        if verbose:
            print(f"  iter {iteration + 1}: |ddv_scf|^2 = {change:.3e}")
        if onecentre is None:
            dvscf = mixer.mix(dvscf, induced)
        else:
            # **One Anderson problem over both.** The one-centre potential and
            # ``dV_scf`` are coupled through the same ``dbecsum``, which is why
            # ``mix_pot`` concatenates them rather than mixing them apart.
            dvscf, onecentre = mixer.mix(
                [dvscf, onecentre], [induced, induced_onecentre]
            )
        if change < tr2:
            converged = True
            break

    # ``ef_shift_wfc``: the level's motion belongs to the first-order *states*
    # as well, and it is applied once at the end, as QE applies it -- the loop
    # itself is driven by the density, which already carries the correction.
    # It matters because the second derivative consumes ``dpsi`` as a tangent
    # rather than through the density it builds.
    if solver.smearing is not None and shifts is not None:
        for index, (atom, cart) in enumerate(
            (a, c) for a in range(nat) for c in range(3)
        ):
            dpsi[atom, cart] = solver.fermi_level_shift_states(
                dpsi[atom, cart], shifts[index]
            )

    extras = {
        "dvscf": dvscf,
        "onecentre": onecentre,
        "dbecsum": None if onecentre is None else per_mode,
    }
    return (dpsi, symmetrised, history,
            total_iterations / max(solves, 1), converged, extras)


def _core_charge_response(calculation, density, positions):
    """``dv_xc`` from the core charge travelling with its atom -- ``addcore``.

    ``(nat, 3, nspin_mag, n1, n2, n3)``, or ``None`` when no species has a
    nonlinear core correction.

    **This is not an ultrasoft term and it was missing before the ultrasoft
    ones.** :func:`_bare_displacements` builds ``dV_bare|psi>`` at a *frozen*
    ``v_scf``, which is right for the local potential and the projectors and
    wrong for the exchange-correlation potential: ``rho_core(r - tau)`` moves
    with the atom, so ``v_xc[rho + rho_core]`` changes even at a frozen valence
    density. QE keeps it as ``drhoc`` and hands it to ``dv_of_drho`` beside the
    response density (``solve_linter.f90``'s ``addcore``); here it is one
    ``jvp`` of the potential through
    :meth:`~pypresso.scf.driver.Calculation.at_positions` at a *fixed* density,
    which is the same object and needs no second expression.

    It is independent of the self-consistent iteration -- nothing responds to
    it -- so it is built once and added to the induced potential every step.

    **Every committed phonon case before this had no core charge**, which is
    why it went unnoticed: ``Si.pz-vbc`` and ``Al.pz-vbc`` are
    ``core_correction="false"`` and the ultrasoft and PAW silicon datasets that
    first exercised it are ``"T"``. Its size, on ``si-epsilon-us``: leaving it
    out puts the response density **45%** away from a finite difference of
    re-converged densities, and the optical mode at 785 cm^-1 against
    ``ph.x``'s 513. It is the same omission ``force_cc`` exists for one
    derivative down, and :func:`~pypresso.forces.energy.frozen_energy` already
    carries it there -- so the assembly was right and only the response was not.
    """
    if calculation.rho_core is None:
        return None
    if positions is None:
        raise ValueError(
            "the displacement response of a dataset with a nonlinear core "
            "correction needs the positions: the core charge travels with its "
            "atom, so v_xc changes at a frozen valence density (addcore). "
            "Pass positions=... to self_consistent_response"
        )
    positions = jnp.asarray(positions)
    nat = positions.shape[0]
    rho = jnp.asarray(density)

    def potential_at(pos):
        return calculation.at_positions(pos).potential(rho).v_scf

    fields = []
    for atom in range(nat):
        for cart in range(3):
            tangent = jnp.zeros_like(positions).at[atom, cart].set(1.0)
            fields.append(jax.jvp(potential_at, (positions,), (tangent,))[1])
    return jnp.stack(fields).reshape((nat, 3) + rho.shape)


def _bare_displacements(calculation, solver, v_scf, positions) -> np.ndarray:
    """``dV_bare/du |psi>`` for every atom and cartesian direction.

    ``dvqpsi_us.f90`` builds this term by term from ``dvloc/dtau``, the
    projectors' derivatives and (for a core correction) ``drhoc``. Here it is
    one ``jvp`` through
    :meth:`~pypresso.scf.driver.Calculation.at_positions` at frozen ``v_scf`` --
    the same method the force differentiates, and the same call
    :func:`~pypresso.response.efield._born_charges` already makes for ``Z*``,
    which is why this is the one piece of the phase that was written before it.

    "Bare" is the whole point: the self-consistent part of the perturbation is
    the induced ``dV_scf`` that the loop above adds, so the potential handed to
    ``at_positions`` here is held *fixed* while the atoms move under it.

    Returns an object array of shape ``(nat, 3)`` whose entries are
    ``(nspin, nk, nocc, npwx)`` -- see the module docstring on what that costs.
    """
    batch = calculation.k_batch
    psi = solver.psi
    nat = positions.shape[0]

    eigenvalues = solver.eigenvalues

    def h_psi(moved_positions):
        moved = calculation.at_positions(moved_positions)
        hamiltonians = moved.hamiltonian(v_scf, solver.ddd_paw)
        applied = []
        for spin, hamiltonian in enumerate(hamiltonians):
            values = over_kpoints(hamiltonian, psi[spin], batch)
            if hamiltonian.has_overlap:
                # ``compute_deff``: the right-hand side of the Sternheimer
                # equation for a *displacement* is ``(dH/du - eps dS/du)|psi>``,
                # which ``dvqpsi_us_only`` builds from
                # ``deff = deeq - et qq`` rather than from ``deeq``. The second
                # half is identically zero when ``S`` is the identity, so this
                # branch is the ultrasoft and PAW one and nothing else here
                # changes. Leaving it out solves a different equation whose
                # solution is still orthogonal to the occupied manifold and
                # still converges.
                overlap = over_kpoints(hamiltonian, psi[spin], batch, overlap=True)
                values = values - eigenvalues[spin][..., None] * overlap
            applied.append(values)
        return jnp.stack(applied)

    bare = np.empty((nat, 3), dtype=object)
    for atom in range(nat):
        for cart in range(3):
            tangent = jnp.zeros_like(positions).at[atom, cart].set(1.0)
            bare[atom, cart] = jax.jvp(h_psi, (positions,), (tangent,))[1]
    return bare


def orthogonality_states(calculation, solver, positions) -> np.ndarray | None:
    """``dpsi^ort_n = -1/2 sum_m psi_m <psi_m|dS/du|psi_n>``, one per mode.

    ``compute_drhous.f90``'s ingredient, and the piece of the first-order
    wavefunction the Sternheimer solve does **not** produce. ``solve``'s answer
    is orthogonal to the occupied manifold in the ``S`` metric -- that is what
    ``orthogonalize``'s projector imposes -- but the physical first-order state
    is not, because the constraint it has to satisfy is
    ``<psi + dpsi|S(u + du)|psi + dpsi> = 1`` and ``S`` itself has moved. The
    occupied-occupied block is therefore fixed rather than free, and this is it.

    ``None`` for a norm-conserving dataset, where ``S`` does not move and the
    block is zero. Returns an object array of shape ``(nat, 3)`` whose entries
    have ``dpsi``'s own shape.
    """
    derivatives = overlap_derivatives(calculation, solver, positions)
    if derivatives is None:
        return None
    psi = solver.psi
    out = np.empty(derivatives.shape, dtype=object)
    for index in np.ndindex(derivatives.shape):
        out[index] = -0.5 * jnp.einsum("skmg,skmn->skng", psi, derivatives[index])
    return out


def overlap_derivatives(calculation, solver, positions) -> np.ndarray | None:
    """``S'_mn = <psi_m|dS/du|psi_n>`` over the occupied block, per mode.

    ``None`` for a norm-conserving dataset. The object two other things are
    built from -- :func:`orthogonality_states` and the gauge correction in
    :func:`multiplier_response` -- so it is computed once and shared.
    """
    if not calculation.is_ultrasoft:
        return None
    psi = solver.psi
    nat = positions.shape[0]
    batch = calculation.k_batch

    def overlap_matrix(pos):
        """``<psi_m|S(u)|psi_n>`` -- only the augmentation half moves."""
        moved = calculation.at_positions(pos)
        vkb = moved.projectors.vkb
        qq = moved.projectors.qq.astype(psi.dtype)
        blocks = []
        for spin in range(psi.shape[0]):
            def one_k(ik, spin=spin):
                becp = jnp.einsum("gc,ng->nc", vkb[ik].conj(), psi[spin][ik])
                return jnp.einsum("mi,ij,nj->mn", becp.conj(), qq, becp)

            blocks.append(map_k(one_k, jnp.arange(psi.shape[1]), batch=batch))
        return jnp.stack(blocks)

    out = np.empty((nat, 3), dtype=object)
    for atom in range(nat):
        for cart in range(3):
            tangent = jnp.zeros_like(positions).at[atom, cart].set(1.0)
            out[atom, cart] = jax.jvp(overlap_matrix, (positions,), (tangent,))[1]
    return out


def non_variational_response(calculation, positions, psi, weights, density,
                             becsum, ort):
    """``drhous`` and ``becsumort``: the mixed state's change at frozen ``dpsi``.

    ``PHonon/PH/drho.f90``, whose own summary is the definition -- "the change
    of the charge density due to the displacement, at fixed wavefunctions; the
    orthogonality part is included in the computed change". Two things move it
    and neither exists for a norm-conserving dataset:

    * the augmentation charge ``Q_ij(r - tau)`` and the projectors ``beta(r -
      tau)`` travel with their atom, so ``rho`` and ``becsum`` depend on the
      positions explicitly (``addusddens``'s ``alpha bb`` term and
      ``alphasum``);
    * the occupied block of the first-order state is not zero
      (:func:`orthogonality_states`), and it changes both.

    Both fall out of ``jvp`` of the raw mixed-state builders --
    :func:`~pypresso.response.born._raw_mixed_state`'s, reused rather than
    restated -- and **the two halves are returned apart**, because they are
    consumed in different places. Their sum is what screens
    (:func:`self_consistent_response`) and what the assembly's density tangent
    has to carry; but the ``moved`` half is *also* generated by the assembly's
    own position tangent, since :func:`_force_constants` hands the mixed state
    to :func:`~pypresso.forces.energy.frozen_energy` as a function of where the
    atoms are rather than as an array -- which it must, because the gradient
    being differentiated is otherwise not the force at all. It would then be
    counted twice.

    **That is the term ``addusforce`` is**, one derivative down: at a frozen
    density the ultrasoft force is missing the augmentation charge's own
    motion, and P25 could freeze it because a norm-conserving ``rho`` has no
    explicit position dependence. Measured on ultrasoft silicon, freezing it
    puts a whole column of the force constants at **0.26** of a
    finite-differenced one, with the acoustic sum rule holding throughout --
    which is P28a's lesson again, that an atom-sum is blind to a transfer
    between atoms.

    Returns ``((drho_moved, becsum_moved), (drho_ort, becsum_ort))`` as object
    arrays of shape ``(nat, 3)``, or ``(None, None)`` for a norm-conserving
    dataset.
    """
    if ort is None:
        return (None, None), (None, None)
    from pypresso.response.born import _raw_mixed_state

    density_of, becsum_of = _raw_mixed_state(
        calculation, positions, psi, weights, density, becsum
    )

    def mixed(pos, states):
        moved = calculation.at_positions(pos)
        parts = becsum_of(moved, states, weights)
        return density_of(moved, states, weights, parts), parts

    nat = positions.shape[0]
    zero_p, zero_s = jnp.zeros_like(positions), jnp.zeros_like(psi)
    halves = {}
    for name in ("moved", "ort"):
        halves[name] = (np.empty((nat, 3), dtype=object),
                        np.empty((nat, 3), dtype=object))
    for atom in range(nat):
        for cart in range(3):
            tangent = jnp.zeros_like(positions).at[atom, cart].set(1.0)
            for name, pair in (("moved", (tangent, zero_s)),
                               ("ort", (zero_p, ort[atom, cart]))):
                _, (rho, parts) = jax.jvp(mixed, (positions, psi), pair)
                halves[name][0][atom, cart] = rho
                halves[name][1][atom, cart] = parts
    return halves["moved"], halves["ort"]


def symmetrize_becsum_modes(calculation, per_mode):
    """``PAW_dusymmetrize`` on the ``3 nat`` displacement patterns.

    The counterpart of :meth:`~pypresso.scf.driver.Calculation.
    symmetrize_atom_displacement` one level down, and the reason it is not
    optional is P24a's: a ``becsum`` response on a reduced k-set is a **polar**
    object carrying a direction, so a wedge sum of it is not the crystal's until
    the group has put back what the reduction left out. Worth 1.6e-2 on the
    dielectric constant of PAW silicon.

    ``per_mode`` is an object array of shape ``(nat, 3)`` whose entries are
    per-species ``becsum`` tuples; the result has the same layout. A no-op when
    there is no PAW species or no symmetry to average over.
    """
    if per_mode is None or calculation._becsum_symmetry is None:
        return per_mode
    nat, ncart = per_mode.shape
    sample = per_mode[0, 0]
    if not sample:
        return per_mode
    stacked = tuple(
        None if sample[species] is None else jnp.stack([
            jnp.stack([per_mode[atom, cart][species] for cart in range(ncart)])
            for atom in range(nat)
        ])
        for species in range(len(sample))
    )
    rotations = cartesian_rotations(calculation.system.cell, calculation.symmetries)
    mapping = atom_mapping(
        calculation.system.cell, calculation.system.structure, calculation.symmetries
    )
    symmetrised = calculation._becsum_symmetry.apply_atom_displacement(
        stacked, rotations, mapping
    )
    out = np.empty((nat, ncart), dtype=object)
    for atom in range(nat):
        for cart in range(ncart):
            out[atom, cart] = tuple(
                None if values is None else values[atom, cart]
                for values in symmetrised
            )
    return out


def _symmetrize_modes(calculation, per_mode):
    """``symdvscf`` on an object array of ``3 nat`` grid responses."""
    nat, ncart = per_mode.shape
    stacked = jnp.stack([
        jnp.stack([per_mode[atom, cart] for cart in range(ncart)])
        for atom in range(nat)
    ])
    symmetrised = calculation.symmetrize_atom_displacement(stacked)
    out = np.empty((nat, ncart), dtype=object)
    for atom in range(nat):
        for cart in range(ncart):
            out[atom, cart] = symmetrised[atom, cart]
    return out


def _ground_state_multipliers(weights, eigenvalues, dtype):
    """``Lambda_mn = delta_mn w_n eps_n``: the multipliers at the solution.

    :func:`~pypresso.response.born._ground_state_multipliers`, and the same
    argument for passing it explicitly: it is what puts the matrix on the
    tangent's argument list, so that ``dLambda`` can be a tangent of it.
    """
    identity = jnp.eye(weights.shape[-1], dtype=dtype)
    return (weights * eigenvalues)[..., :, None].astype(dtype) * identity


def multiplier_response(calculation, solver, bare, extras, weights, nbnd,
                        derivatives):
    """``dLambda_mn``: the multipliers' own tangent, in the gauge ``ort`` fixes.

    **The multipliers are variables of the functional and they move.** P15
    writes the energy with the orthonormality constraint carried explicitly and
    its multipliers among the *frozen* variables, which is what makes the force
    a partial derivative. Differentiating that gradient a second time needs
    their tangent beside the states' and the density's:

        d^2E/du_i du_j = d_i d_j L + (d_psi d_j L).dpsi_i + (d_Lambda d_j L).dLambda_i

    and ``d_Lambda d_j L = -<psi|dS/du_j|psi>``, which vanishes identically when
    ``S`` is the identity. That is the whole of why P25 was norm-conserving: not
    a missing routine, a missing tangent.

    **It is a matrix and not a diagonal, and the reason is a gauge.** Write the
    constraint with a diagonal multiplier ``w_n eps_n`` and the functional stops
    being invariant under a unitary mixing of the occupied states -- the
    eigenvalues weight the bands differently -- while the *state* tangent is
    only defined up to exactly such a mixing: the constraint fixes
    ``c + c^dagger = -S'`` and leaves the antihermitian part of ``c`` free.
    :func:`orthogonality_states` picks the hermitian representative; the
    physical branch picks another; and a diagonal multiplier can tell them
    apart. **The acoustic sum rule is what says so**: with the diagonal form it
    stops at 1.7e-2 Ry/bohr^2 on ultrasoft silicon whatever else is switched on,
    where the matrix form takes it to the basis-set floor. Carrying the full
    ``Lambda_mn`` restores the invariance, because the pair ``(dpsi, dLambda)``
    then transforms together and the gauge cancels between them.

    In that gauge, differentiating ``Lambda_mn = w_n <psi_m|H|psi_n>`` along the
    branch gives

        dLambda_mn = w_n [ <psi_m|dH|psi_n> - 1/2 (eps_m + eps_n) S'_mn ]

    -- the ``1/2 (eps_m + eps_n)`` and not ``eps_n``, which is
    :func:`~pypresso.response.born._multiplier_response`'s expression and is
    right *there* because a field leaves ``dpsi`` orthogonal to the occupied
    manifold, so the two terms carrying it drop out. A displacement does not,
    and using the field's expression here is wrong by ``(eps_n - eps_m)/2`` on
    every off-diagonal entry. Since the perturbation handed in already carries
    ``-eps_n dS`` (``compute_deff``), what has to be added back is
    ``+1/2 (eps_n - eps_m) S'_mn``.
    """
    from pypresso.response.born import _multiplier_response

    nat = bare.shape[0]
    dvscf, onecentre = extras["dvscf"], extras["onecentre"]
    nocc = solver.nocc
    eps = solver.eigenvalues[..., :nocc]
    # ``(nspin, nk, nocc, nocc)``: (eps_n - eps_m) with m the row.
    gaps = eps[..., None, :] - eps[..., :, None]
    out = np.empty((nat, 3), dtype=object)
    for atom in range(nat):
        for cart in range(3):
            perturbation = _bare_plus_induced(
                solver, bare[atom, cart], dvscf[atom, cart], True,
                None if onecentre is None else onecentre[atom, cart],
            )
            matrix = _multiplier_response(
                solver, perturbation, weights, nbnd, nocc
            )
            correction = 0.5 * gaps * derivatives[atom, cart] * (
                weights[:, :, None, :nocc]
            )
            out[atom, cart] = matrix.at[:, :, :nocc, :nocc].add(correction)
    return out


def _stack_modes(per_mode):
    """``(nat, 3)`` object array of grid fields -> one stacked array."""
    nat, ncart = per_mode.shape
    return jnp.stack([
        jnp.stack([per_mode[atom, cart] for cart in range(ncart)])
        for atom in range(nat)
    ])


def _subtract_becsum(left, right):
    """``left - right`` on two ``(nat, 3)`` object arrays of ``becsum`` tuples."""
    out = np.empty(left.shape, dtype=object)
    for index in np.ndindex(left.shape):
        out[index] = tuple(
            None if a is None else a - b
            for a, b in zip(left[index], right[index])
        )
    return out


def _add_becsum(left, right):
    """Add two ``(nat, 3)`` object arrays of per-species ``becsum`` tuples."""
    out = np.empty(left.shape, dtype=object)
    for index in np.ndindex(left.shape):
        out[index] = tuple(
            None if a is None else a + b
            for a, b in zip(left[index], right[index])
        )
    return out


def _bare_plus_induced(solver, bare_mode, dv, include_induced: bool,
                       dddd_paw=None):
    """``dV_bare|psi> + dV_scf|psi>`` as the solver's callback wants it.

    The sign convention is :meth:`SternheimerSolver.chi0`'s and not
    :func:`~pypresso.response.efield._solve_stored`'s: what is handed in is an
    honest ``dH|psi>``, and ``project`` carries ``orthogonalize``'s sign
    already. The extra minus in the electric-field path belongs to
    ``dvpsi_e``'s commutator, which is a different right-hand side.
    """
    if not include_induced:
        return lambda psi, ik, spin: bare_mode[spin][ik]

    induced = solver.perturbation(dv, dddd_paw)

    def perturbation(psi, ik, spin):
        return bare_mode[spin][ik] + induced(psi, ik, spin)

    return perturbation


def _state_weights(solver, weights):
    """The weight the *state* tangent is contracted with -- ``wg`` or ``wk``.

    An insulator's ``dpsi`` is a plain wavefunction response and the functional's
    own ``wg`` is right for it. A metal's is not: ``orthogonalize``'s smearing
    branch scales the right-hand side by ``wg1 = f``, so the occupation is
    already inside the tangent and contracting against ``wg = wk f`` would apply
    it twice. ``incdrhoscf`` is called with ``wk`` for the same reason, which is
    what :attr:`SternheimerSolver.density_weights` holds -- and reusing that
    array rather than rebuilding the broadcast is deliberate: it is the one the
    ``response_density == def ldos`` identity was measured against, so a
    normalisation convention cannot differ between the density and the matrix.

    It is ``nocc``-sliced for an insulator and full-length for a metal (where
    ``keep = nbnd``), which is why the insulator branch returns ``weights``
    whole instead: :func:`_force_constants` contracts a tangent padded to
    ``nbnd``, and the padding is zero exactly where the two would disagree.
    """
    if solver.smearing is None:
        return weights
    return jnp.broadcast_to(solver.density_weights[:, :, :1], weights.shape)


def _force_constants(
    calculation, positions, psi, weights, state_weights, eigenvalues, density,
    dpsi, drho, nocc, becsum=(), dbecsum=None, multipliers=None, ort=None,
) -> np.ndarray:
    """``d^2E/du_i du_j``: two ``jvp`` of the force's gradient per mode.

    The frozen-state energy is a function of three things that move -- where the
    atoms are, what the states are and what the density is -- and the total
    second derivative is its derivative along the tangent that carries all
    three:

        D[:, i] = jvp( grad_u L )(u, psi, rho ; e_i, dpsi_i, drho_i)

    The ``e_i`` half is the frozen Hessian (``dynmat0``, ``d2ionq``, the local
    potential's and the projectors' second derivatives); the ``dpsi_i`` and
    ``drho_i`` halves are the electronic response (``drhodv``). Nothing
    separates them in the mathematics: they are components of one tangent
    vector, and for an insulator one ``jvp`` is what this was.

    **The state tangent is contracted with a different weight from the rest,
    and that is the whole of what a metal adds.** ``L`` weights the states by
    ``wg = wk f``, which is right for the frozen Hessian -- ``dynmat_us.f90``
    reads ``wg(ibnd, ikk)`` -- and wrong for the electronic half, because a
    metal's ``dpsi`` already carries its occupation: ``orthogonalize``'s
    smearing branch scales the right-hand side by ``wg1 = f`` and
    ``incdrhoscf`` then accumulates with ``wk``. Contracting such a tangent
    against a ``wg``-weighted functional counts ``f`` twice. QE never does,
    because its two halves are two routines: ``drhodvnl.f90`` contracts with
    ``2 wk(ikk)`` while ``dynmat_us.f90`` uses ``wg``. So the ``jvp`` is split
    in two and the state tangent is taken against ``L[wk]``:

        D[:, i] = jvp_(u, rho)( grad_u L[wg] )(e_i, drho_i)
                + jvp_psi(     grad_u L[wk] )(dpsi_i)

    which is de Gironcoli's Eq. (B19) structure -- the frozen Hessian at ``wg``,
    the electronic response with the metal's own weights.

    **The occupations' own first-order change needs no term of its own**, which
    is the part of this that is not obvious. ``df_n`` and the Fermi level's
    motion are already inside ``dpsi``: the ``(f_i - f_j)/(eps_i - eps_j)``
    structure of ``orthogonalize``'s ``wwg`` is what puts the valence-valence
    block there -- it vanishes identically for an insulator, where every
    occupied ``f`` is 1 -- and ``ef_shift_wfc`` puts the rest. The check that
    it is complete is one this code already makes: ``wk 2 Re[psi* dpsi]`` equals
    the corrected response density to 1e-10, and for a local perturbation
    contracting the tangent *is* ``int drho dV_bare``, which is ``drhodvloc``.

    **The split is unconditional and an insulator is the proof of it.** There
    ``wk = wg`` on every occupied band and ``dpsi`` is zero on the rest, so the
    two ``jvp`` sum to exactly the one they replace -- silicon's optical mode is
    unchanged to round-off, which is the regression that guards this refactor.
    The density tangent goes with the frozen Hessian rather than with the
    states because the terms it reaches -- ``int vltot(tau) rho``,
    ``E_xc[rho + rho_core(tau)]`` -- carry no state weight at all, so it makes
    no difference which half it is put in.

    **The density is a separate argument rather than a function of the states,
    and that is not a convenience.** ``L`` builds its density with the SCF's own
    *scalar* symmetrisation, which is how a wedge sum is completed and is right
    for the ground state. A response must not go through it: displacing one atom
    breaks the crystal's symmetry, and averaging the result over the full group
    of the undisplaced crystal projects the perturbation away. Left to the chain
    rule the state tangent goes straight through that average, and the symptom
    is a matrix that looks perfectly cubic and violates the acoustic sum rule by
    0.72 Ry/bohr^2 -- 580 cm^-1 where the answer is 2. So ``drho`` comes from
    :func:`self_consistent_response`, already symmetrised the way ``symdvscf``
    symmetrises a response, and is handed in.

    **Which bands the tangent carries differs between the two regimes**, and
    the padding is exact in both for different reasons. For an insulator
    ``dpsi`` holds the ``nocc`` bands the Sternheimer equation solves for and
    the rest is padded with zero: the empty bands have ``wg = 0`` in every term
    of ``L``, so no derivative of it could see them anyway. For a metal
    ``solver.nocc`` is ``nbnd`` -- ``orthogonalize``'s smearing branch keeps
    every band, since "occupied" is not a count there -- and nothing is padded.
    The bands above the smearing carry ``wg1 = 0`` inside ``dpsi`` instead,
    which is what makes them harmless against a ``wk`` that is *not* zero there.
    """
    nat = positions.shape[0]
    ultrasoft = calculation.is_ultrasoft

    ground = (
        _ground_state_multipliers(weights, eigenvalues, psi.dtype)
        if ultrasoft else None
    )
    if ultrasoft:
        from pypresso.response.born import _raw_mixed_state

        raw_density, raw_becsum = _raw_mixed_state(
            calculation, positions, psi, weights, density, becsum
        )

        def raw_mixed(pos, states):
            """The unsymmetrised mixed state, as a function of both."""
            moved = calculation.at_positions(pos)
            parts = raw_becsum(moved, states, weights)
            return raw_density(moved, states, weights, parts), parts

    def energy(pos, states, rho, parts, lambdas, w):
        """``L(u, psi, Lambda)`` with the mixed state where it belongs.

        For an ultrasoft or PAW dataset the mixed state stays a **function of
        both the positions and the states**, and ``rho``/``parts`` carry only
        the correction from the wedge sum to the symmetrised response. Freezing
        it as an array -- right for a norm-conserving run, and P25's choice --
        costs two different things: the gradient loses ``addusforce``, so what
        is differentiated is not the force; and the second derivative loses the
        cross term ``d^2 rho / du dpsi . dpsi``, the augmentation charge's own
        position dependence applied to the state response, which is what
        ``addusdynmat`` and ``drhodvus`` are.
        """
        if not ultrasoft:
            return frozen_energy(
                calculation, pos,
                FrozenState(wavefunctions=states, weights=w,
                            eigenvalues=eigenvalues),
                density=rho, becsum=parts, multipliers=lambdas,
            )

        def becsum_builder(moved, built_states, occupations):
            return tuple(
                None if raw is None else raw + extra
                for raw, extra in zip(
                    raw_becsum(moved, built_states, occupations), parts
                )
            )

        def density_builder(moved, built_states, occupations, built):
            return raw_density(moved, built_states, occupations, built) + rho

        return frozen_energy(
            calculation, pos,
            FrozenState(wavefunctions=states, weights=w, eigenvalues=eigenvalues),
            density=density_builder, becsum=becsum_builder, multipliers=lambdas,
        )

    gradient = jax.grad(energy, argnums=0)

    def frozen(pos, rho, parts, lambdas):
        """The coordinate, the mixed state and the multipliers, at ``wg``."""
        return gradient(pos, psi, rho, parts, lambdas, weights)

    def electronic(states):
        """The states, at ``wk`` -- the weight ``drhodvnl`` contracts with."""
        return gradient(positions, states, zero_density, zero_becsum, ground,
                        state_weights)

    def energy_gradient(pos, states, rho, parts, lambdas):
        """Everything in one call, which an ultrasoft cross term needs."""
        return gradient(pos, states, rho, parts, lambdas, weights)

    # For a norm-conserving dataset there is no ``becsum`` and no multiplier
    # term with any position dependence, so the extra primals are dropped
    # rather than carried at zero -- which keeps that path the two-argument
    # ``jvp`` it was, bit for bit.
    zero_becsum = tuple(None if b is None else jnp.zeros_like(b) for b in becsum)
    zero_density = jnp.zeros_like(jnp.asarray(density))
    if not ultrasoft:
        zero_density = jnp.asarray(density)

        def frozen(pos, rho):  # noqa: F811 -- the norm-conserving signature
            return gradient(pos, psi, rho, becsum, None, weights)

    matrix = np.zeros((nat, 3, nat, 3))
    for atom in range(nat):
        for cart in range(3):
            tangent = jnp.zeros_like(positions).at[atom, cart].set(1.0)
            states = jnp.zeros_like(psi).at[:, :, :nocc].set(dpsi[atom, cart])
            if ort is not None:
                # The occupied block, which the Sternheimer solve does not
                # produce and the identity this module rests on needs: ``dpsi``
                # in ``d^2E = ... + (d_psi d_j L).dpsi_i`` is the *whole*
                # first-order state, and for an ultrasoft or PAW dataset its
                # occupied part is fixed by the constraint rather than free.
                states = states + ort[atom, cart]
            if not ultrasoft:
                _, hessian = jax.jvp(
                    frozen, (positions, density), (tangent, drho[atom, cart])
                )
                _, response = jax.jvp(electronic, (psi,), (states,))
                matrix[atom, cart] = np.asarray(hessian + response)
                continue

            # **One ``jvp`` and not two, and that is what an ultrasoft dataset
            # forces.** The metal's weight split (``PLAN.md`` P28) puts the
            # coordinate in one call and the states in another, which is exact
            # only while the two do not meet -- and here they do: ``rho``
            # depends on the positions *and* on the states, so the cross term
            # ``d^2 rho / du dpsi`` exists and a split jvp cannot see it. It is
            # ``addusdynmat``, it is 0.49 Ry/bohr^2 on two-atom ultrasoft
            # silicon against force constants of 0.37, and the acoustic sum
            # rule is what says so. An ultrasoft *metal* is refused for exactly
            # this reason (:func:`_require_a_moving_overlap_regime`).
            raw = jax.jvp(raw_mixed, (positions, psi), (tangent, states))[1]
            correction = drho[atom, cart] - raw[0]
            parts = (
                tuple(None if b is None else jnp.zeros_like(b) for b in becsum)
                if dbecsum is None else
                tuple(
                    None if a is None else a - b
                    for a, b in zip(dbecsum[atom, cart], raw[1])
                )
            )
            _, whole = jax.jvp(
                energy_gradient,
                (positions, psi, zero_density, zero_becsum, ground),
                (tangent, states, correction, parts,
                 jnp.zeros_like(ground) if multipliers is None
                 else multipliers[atom, cart]),
            )
            matrix[atom, cart] = np.asarray(whole)
    return matrix



def _impose_acoustic_sum_rule(matrix: np.ndarray) -> np.ndarray:
    """``sum_b D_(a i)(b j) = 0``: translating the crystal costs nothing.

    QE's ``asr = 'simple'`` in ``q2r``/``matdyn``, and **not** what ``ph.x``
    does by default, which is why this is off unless asked for. The correction
    goes on the diagonal block of each atom, as QE puts it.
    """
    corrected = np.array(matrix, dtype=float)
    for atom in range(matrix.shape[0]):
        corrected[atom, :, atom, :] -= matrix[atom].sum(axis=1)
    return corrected


def _diagonalize(matrix: np.ndarray, masses: np.ndarray):
    """``dyndia``: mass-weight, diagonalise, and report in cm^-1.

    ``amass`` is in amu and the force constants are in Ry/bohr^2, so the
    conversion is :data:`~pypresso.units.AMU_TO_RY` -- QE's ``amu_ry``, applied
    in exactly the same place. A negative eigenvalue is an imaginary frequency
    and is reported as a negative number rather than as a complex one, which is
    ``dyndia``'s convention and the reason the acoustic modes of a PAW silicon
    run print as ``-6.1``.
    """
    nat = matrix.shape[0]
    flat = np.asarray(matrix, dtype=float).reshape(3 * nat, 3 * nat)
    scale = 1.0 / np.sqrt(np.repeat(masses, 3) * AMU_TO_RY)
    weighted = flat * scale[:, None] * scale[None, :]
    omega2, vectors = np.linalg.eigh(0.5 * (weighted + weighted.T))
    frequencies = np.sign(omega2) * np.sqrt(np.abs(omega2)) * RY_TO_CMM1
    return frequencies, vectors


def _require_one_spin_channel(calculation) -> None:
    """``nspin = 2`` is refused because ``nocc`` is a single number here.

    Everything in this module and in :mod:`pypresso.response.efield` counts the
    occupied bands as ``nelec / 2`` and slices *both* spin channels to that
    depth. For an unpolarized insulator that is right. For a magnetic one it is
    not: the two channels have different occupancies, and a shared count silently
    solves for the wrong bands in at least one of them -- with no shape error and
    no failed convergence to show for it.

    Refused here rather than approximated, and refused again -- for the same
    reason and with its own message -- in
    :func:`~pypresso.response.sternheimer.require_a_sternheimer_regime`, which
    every other entry point goes through. This one stays because the message
    names the dynamical matrix. Making ``nocc`` per-channel is one change in
    :class:`~pypresso.response.sternheimer.SternheimerSolver` and would lift
    both, and it needs a magnetic insulator to validate against -- which is why
    it is named here and left.
    """
    if calculation.nspin == 2:
        raise NotImplementedError(
            "the dynamical matrix for a spin-polarized calculation is not "
            "implemented: the occupied-band count here is one number for both "
            "channels (nelec/2), and a magnetic insulator's channels are "
            "occupied to different depths, so the response would be solved for "
            "the wrong bands in one of them without any sign of it"
        )


def _require_a_moving_overlap_regime(calculation) -> None:
    """What is still refused once ``S`` is allowed to move with the atoms.

    ``PLAN.md`` P39. The norm-conserving restriction P25 introduced is gone --
    the two terms it named are implemented
    (:func:`orthogonality_states` for the multipliers' own motion,
    :func:`non_variational_response` for ``Q_ij(r - tau)`` at frozen
    ``becsum``) -- and what is left is one combination rather than one dataset.

    **An ultrasoft or PAW metal is refused.** The weight split
    :func:`_state_weights` makes is between a state tangent at ``wk`` and
    everything else at ``wg``, and it was derived for a response whose whole
    ``becsum`` dependence goes through ``dpsi``. With ``S`` moving there are
    three further tangents -- ``dpsi^ort``, ``becsumort`` and ``dLambda`` --
    and which weight each belongs with is a question the insulating case cannot
    answer, because there the two weights are equal. Refused rather than
    guessed: the symptom would be an acoustic sum rule that is violated by a
    plausible amount, and P28 is the record of how long that took to find once.
    """
    if calculation.is_ultrasoft and calculation.system.occupations != "fixed":
        raise NotImplementedError(
            "the dynamical matrix of a *metal* with an ultrasoft or PAW "
            "pseudopotential is not implemented: the wg/wk weight split of "
            "PLAN.md P28 was derived for a response whose becsum dependence is "
            "entirely inside dpsi, and an ultrasoft one has three further "
            "tangents (dpsi^ort, becsumort, dLambda) whose weight an insulator "
            "cannot distinguish. Insulators are implemented on all three "
            "pseudopotential kinds; a norm-conserving metal is too"
        )


def require_norm_conserving(calculation) -> None:
    """Norm-conserving only, for the responses in the **strain** coordinate.

    **This no longer guards the dynamical matrix.** P39 lifted that: with ``S``
    moving there are four further tangents and all four are written, and what
    :func:`dynamical_matrix` checks now is
    :func:`_require_a_moving_overlap_regime`, which refuses one *combination*
    -- an ultrasoft or PAW metal -- rather than a dataset. What still calls
    this is the strain-coordinate stack built on top of it:
    :mod:`pypresso.response.strain`, :mod:`pypresso.response.electrostriction`
    and :mod:`pypresso.response.nonlinear`.

    **And it no longer guards the Raman tensor either** (``PLAN.md`` P43). The
    second-order energy ``F`` these phases differentiate is exact on all three
    pseudopotential kinds -- it reproduces ``dielec.f90``'s dielectric constant
    to **3.4e-10** (ultrasoft) and **6.9e-11** (PAW) against a norm-conserving
    8.4e-10 -- and the ``jvp`` of it in the *displacement* coordinate is right
    too, at **1.2e-4** on both against a finite difference where the
    norm-conserving control is 6.8e-4. That took two tangents which are only
    correct together: the state tangent is ``P_c dpsi + ort``, and ``db`` is
    the tangent of a *composition*, because ``adddvepsi_us`` applies ``S`` to
    the linear solve's answer and adds the augmentation dipole after it
    (:func:`~pypresso.response.efield.ultrasoft_position`).

    **What is left is the same third derivative in the *strain* coordinate** --
    the elastic constants, electrostriction and the elasto-optic tensor. They
    share :func:`~pypresso.response.electrostriction._position_response` with
    the Raman tensor and would inherit the tail that closed it, but the
    occupied block's analogue under a strain has not been measured, and P43's
    own lesson is that one of the pair alone is *worse* than neither -- the
    block by itself moved the Raman tensor from 3.0e-2 to 8.0e-2.

    The measurement behind the original refusal is kept, because it is what
    makes the case for refusing rather than warning: with the guard lifted and
    none of the terms written, the norm-conserving expression gave ultrasoft
    silicon an optical mode of **-504.3 cm^-1** against ``ph.x``'s **+513.3**
    -- imaginary where the crystal is stable -- from a run that converged and
    gave a cubic, symmetric matrix. Nothing but the numbers looked wrong.
    """
    if calculation.is_ultrasoft:
        raise NotImplementedError(
            "this third derivative in the *strain* coordinate with an "
            "ultrasoft or PAW pseudopotential is not implemented: the "
            "second-order energy under it is exact on all three dataset kinds "
            "and so is its jvp in the *displacement* coordinate, so "
            "raman_tensors works (PLAN.md P43, 1.2e-4 against a finite "
            "difference), but the two tangents that made it work -- the "
            "occupied block of the state and adddvepsi_us's tail on db -- have "
            "not been measured under a strain, and one of the pair alone is "
            "worse than neither. The strain *response*, the dynamical matrix "
            "and the Raman tensor are implemented on all three kinds "
            "(P39, P41, P43); it is the elastic constants, electrostriction "
            "and the elasto-optic tensor that are refused here"
        )
