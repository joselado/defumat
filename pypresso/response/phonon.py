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
    require_a_sternheimer_regime,
)
from pypresso.response.velocity import over_kpoints
from pypresso.system.symmetry import atom_mapping, symmetrize_atom_pair_tensor
from pypresso.units import AMU_TO_RY, RY_TO_CMM1, RY_TO_THZ

__all__ = ["Phonons", "dynamical_matrix", "require_norm_conserving",
           "self_consistent_response"]

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
    """
    eigenvalues = jnp.asarray(eigenvalues)
    if eigenvalues.ndim == 2:
        eigenvalues = eigenvalues[None]
    require_a_symmetrisable_response(calculation)
    require_a_sternheimer_regime(calculation)
    require_norm_conserving(calculation)
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
    )

    # 1. The bare perturbation ``dV_bare/du |psi>``, once per mode and stored:
    #    the self-consistent loop below drives on it at every iteration.
    bare = _bare_displacements(calculation, solver, potential.v_scf, positions)

    # 2. ``solve_linter``'s loop, one perturbation per (atom, direction).
    dpsi, drho, history, average_iterations, converged = self_consistent_response(
        calculation, solver, bare, density,
        alpha_mix=alpha_mix, tr2=tr2, max_iterations=max_iterations,
        verbose=verbose,
    )

    # 3. The second derivative, one jvp of the force's own gradient per mode.
    matrix = _force_constants(
        calculation, positions, jnp.asarray(wavefunctions), weights,
        eigenvalues, jnp.asarray(density), dpsi, drho, nocc,
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
    matrix = symmetrize_atom_pair_tensor(
        matrix, calculation.system.cell, calculation.symmetries,
        atom_mapping(calculation.system.cell, structure, calculation.symmetries),
    )
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


def self_consistent_response(
    calculation,
    solver,
    bare,
    density,
    alpha_mix: float = ALPHA_MIX,
    tr2: float = TR2,
    max_iterations: int = MAX_ITERATIONS,
    mixing_mode: str = DEFAULT_RESPONSE_MIXING,
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

    Returns ``(dpsi, drho, history, average_iterations, converged)``, with
    ``dpsi`` an object array of shape ``(nat, 3)``.
    """
    nat = calculation.system.structure.nat
    grid_shape = jnp.asarray(density).shape
    dvscf = jnp.zeros((nat, 3) + grid_shape)
    history, total_iterations, solves = [], 0, 0
    dpsi = np.empty((nat, 3), dtype=object)
    symmetrised = jnp.zeros_like(dvscf)
    converged = False

    mixer = ResponseMixer(mixing_mode, beta=alpha_mix)
    for iteration in range(max_iterations):
        response = []
        for atom in range(nat):
            for cart in range(3):
                perturbation = _bare_plus_induced(
                    solver, bare[atom, cart], dvscf[atom, cart], iteration > 0
                )
                solution = solver.solve(perturbation)
                dpsi[atom, cart] = solution.dpsi
                total_iterations += solution.iterations
                solves += 1
                response.append(solver.response_density(solution.dpsi))

        # ``symdvscf``: the 3 nat responses are symmetrised *together*, after
        # the loop over modes and before the kernel, because an operation mixes
        # them -- rotating the direction and permuting the atom.
        stacked = jnp.stack(response).reshape((nat, 3) + grid_shape)
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

        change = float(jnp.sum((induced - dvscf) ** 2))
        history.append(change)
        if verbose:
            print(f"  iter {iteration + 1}: |ddv_scf|^2 = {change:.3e}")
        dvscf = mixer.mix(dvscf, induced)
        if change < tr2:
            converged = True
            break

    return (dpsi, symmetrised, history,
            total_iterations / max(solves, 1), converged)


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

    def h_psi(moved_positions):
        moved = calculation.at_positions(moved_positions)
        hamiltonians = moved.hamiltonian(v_scf, solver.ddd_paw)
        return jnp.stack([
            over_kpoints(hamiltonian, psi[spin], batch)
            for spin, hamiltonian in enumerate(hamiltonians)
        ])

    bare = np.empty((nat, 3), dtype=object)
    for atom in range(nat):
        for cart in range(3):
            tangent = jnp.zeros_like(positions).at[atom, cart].set(1.0)
            bare[atom, cart] = jax.jvp(h_psi, (positions,), (tangent,))[1]
    return bare


def _bare_plus_induced(solver, bare_mode, dv, include_induced: bool):
    """``dV_bare|psi> + dV_scf|psi>`` as the solver's callback wants it.

    The sign convention is :meth:`SternheimerSolver.chi0`'s and not
    :func:`~pypresso.response.efield._solve_stored`'s: what is handed in is an
    honest ``dH|psi>``, and ``project`` carries ``orthogonalize``'s sign
    already. The extra minus in the electric-field path belongs to
    ``dvpsi_e``'s commutator, which is a different right-hand side.
    """
    if not include_induced:
        return lambda psi, ik, spin: bare_mode[spin][ik]

    induced = solver.perturbation(dv)

    def perturbation(psi, ik, spin):
        return bare_mode[spin][ik] + induced(psi, ik, spin)

    return perturbation


def _force_constants(
    calculation, positions, psi, weights, eigenvalues, density, dpsi, drho, nocc
) -> np.ndarray:
    """``d^2E/du_i du_j``: one ``jvp`` of the force's gradient per mode.

    The frozen-state energy is a function of three things that move -- where the
    atoms are, what the states are and what the density is -- and the total
    second derivative is its derivative along the tangent that carries all
    three:

        D[:, i] = jvp( grad_u L )(u, psi, rho ; e_i, dpsi_i, drho_i)

    The ``e_i`` half is the frozen Hessian (``dynmat0``, ``d2ionq``, the local
    potential's and the projectors' second derivatives); the ``dpsi_i`` and
    ``drho_i`` halves are the electronic response (``drhodv``). Nothing
    separates them here because nothing separates them in the mathematics: they
    are components of one tangent vector.

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

    ``dpsi`` carries only the occupied bands, which is all the Sternheimer
    equation solves for; the tangent is padded with zeros for the rest. That is
    exact rather than an approximation -- the empty bands have zero weight in
    every term of ``L``, so no derivative of ``L`` can see them.
    """
    nat = positions.shape[0]

    def energy(pos, states, rho):
        return frozen_energy(
            calculation, pos,
            FrozenState(
                wavefunctions=states, weights=weights, eigenvalues=eigenvalues
            ),
            density=rho,
        )

    gradient = jax.grad(energy, argnums=0)

    matrix = np.zeros((nat, 3, nat, 3))
    for atom in range(nat):
        for cart in range(3):
            tangent = jnp.zeros_like(positions).at[atom, cart].set(1.0)
            states = jnp.zeros_like(psi).at[:, :, :nocc].set(dpsi[atom, cart])
            _, column = jax.jvp(
                gradient, (positions, psi, density),
                (tangent, states, drho[atom, cart]),
            )
            matrix[atom, cart] = np.asarray(column)
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

    Refused here rather than approximated. The same arithmetic is in
    ``dielectric_tensor`` and is *not* refused there, which is a gap in P24 and
    not a decision: a magnetic insulator's dielectric constant would have the
    same problem. Making ``nocc`` per-channel is one change in
    :class:`~pypresso.response.sternheimer.SternheimerSolver` and would serve
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


def require_norm_conserving(calculation) -> None:
    """The dynamical matrix is norm-conserving only, and it is refused by name.

    Two things go missing at once when ``S`` moves with the atoms, and neither
    is a routine that could simply be added beside the others:

    * **The multipliers move.** The identity this module rests on holds because
      the frozen-state energy is stationary in ``psi`` at *fixed* Lagrange
      multipliers. Differentiating it a second time leaves a term
      ``-<psi|dS/du_j|psi> deps_i/du``, which is identically zero when ``S`` is
      the identity and is not otherwise.
    * **The augmentation charge moves at frozen ``becsum``.** ``Q_ij(r - tau)``
      is a function of the positions in its own right, which ``addusdynmat`` and
      ``drhodvus`` account for and the response density here does not.

    Refused rather than returned for the same reason
    :func:`~pypresso.response.efield._require_born_charges` refuses ``Z*`` on
    these datasets, and the measurement is the same shape: **wrong in sign as
    well as size**. With the guard lifted, the norm-conserving expression gives
    ultrasoft silicon an optical mode of **-504.3 cm^-1** against ``ph.x``'s
    **+513.3** -- an imaginary frequency where the crystal is stable -- and an
    acoustic residue of 618 where ``ph.x`` prints 6.1. PAW is the same to a
    Rydberg: -503.6 against +513.4. What makes it a refusal rather than a
    warning is that the run *converges*, the matrix comes out cubic and
    symmetric to 1e-16, and everything except the numbers looks correct.

    The **dielectric constant** from the same solver is right for both datasets
    to 5e-5 and is not affected by any of this: the two quantities share the
    Sternheimer solve and nothing else.
    """
    if calculation.is_ultrasoft:
        raise NotImplementedError(
            "the dynamical matrix with an ultrasoft or PAW pseudopotential is "
            "not implemented: the overlap operator moves with the atoms, so the "
            "orthonormality multipliers contribute a term of their own to the "
            "second derivative, and the augmentation charge Q_ij(r - tau) moves "
            "at frozen becsum (addusdynmat, drhodvus). Without them the "
            "norm-conserving expression gives ultrasoft silicon -504.3 cm^-1 "
            "against ph.x's +513.3 -- imaginary where the crystal is stable -- "
            "while converging cleanly and coming out cubic. The dielectric "
            "constant from the same solver is unaffected and is right for these "
            "datasets"
        )
