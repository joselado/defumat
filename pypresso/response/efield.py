"""The response to a uniform electric field: ``epsilon_infinity``.

An electric field is the one perturbation a periodic code cannot simply write
down. ``V = E.r`` is neither bounded nor lattice periodic, so ``r|psi>`` has no
meaning as it stands; what *is* well defined is its projection onto the
conduction manifold, ``P_c r|psi>``, and the standard way to reach it is through
a commutator (``PHonon/PH/dvpsi_e.f90``):

    (H - eps_v S) P_c r|psi_v> = P_c^+ [H - eps_v S, r] |psi_v>

-- one more solve of the same Sternheimer equation
(:mod:`pypresso.response.sternheimer`), with the commutator on the right.

**And the commutator is the velocity operator, so it is not derived here
either.** In the periodic gauge ``H(k) = e^{-ik.r} H e^{ik.r}``, so
``dH/dk_a = i [H, r_a]`` and

    [H - eps S, r_a] |psi> = -i (dH/dk_a - eps dS/dk_a) |psi>,

which is exactly what :mod:`pypresso.response.velocity` produces from one
``jvp``. ``commutator_Hx_psi.f90`` hand-codes the same expression term by term
-- the kinetic ``-2i(k+G)_a`` and ``gen_us_dj``/``gen_us_dy`` for the
projectors' angular and radial derivatives -- and none of that is transcribed.

**The self-consistent part is `solve_e.f90`'s loop, and its kernel is free.**
The induced charge screens the field, so the perturbation seen by a state is the
bare ``P_c r|psi>`` plus ``dV_scf drho`` applied to it, and the loop is

    dpsi    <- Sternheimer(P_c r|psi> + dH_induced |psi>)
    drho    <- d/dlambda rho[psi + lambda dpsi]
    dV_scf  <- K drho,   mixed with the previous one

``K = dV_scf/drho`` is ``dv_of_drho.f90`` -- the Hartree kernel with its ``G=0``
component dropped, plus ``f_xc`` -- and here it is **one ``jax.jvp`` of
``Calculation.potential``** and nothing else, because ``v_of_rho`` is already
written as a differentiable function of the density (rule D1) and already drops
that ``G = 0`` term (:func:`pypresso.scf.potential.hartree`). The exchange-
correlation kernel that QE tabulates in ``setup_dmuxc`` is the second derivative
of the energy this code writes down once.

**The Born charges come from the same two solutions**, and they are a *mixed
second derivative* rather than a formula: ``Z* = dF/dE`` is one ``jvp`` of the
force along the field's response, which is :mod:`pypresso.response.born` and is
P25's machinery with a different tangent. What that buys is **ultrasoft**
datasets, where ``zstar_eu_us.f90`` adds five stages on top of ``zstar_eu.f90``
and the norm-conserving expression is wrong in sign as well as size (+0.1625
against ``ph.x``'s -0.07945). Four of the five are terms of the same
derivative once the mixed state stays a function of the positions and the
constraint's multipliers get a matrix tangent; the fifth,
``add_for_charges``, is transcribed because it contains the position operator's
occupied-occupied block, which is finite only in the combination it appears in.
Against the vendored ``ph.x``: **-0.075715** norm-conserving (every digit of its
-0.07571) and **-0.079442** ultrasoft (8e-6 from its -0.07945). PAW is refused by
name -- :func:`~pypresso.response.born.require_born_charges` carries the
measurement. ``zstar_eu.f90`` itself is transcribed beside it as the cross-check
(:func:`born_charges_zstar_eu`), and the two agree to 1.3e-14 where both apply.

The tensor itself is ``dielec.f90``:

    eps_ij = delta_ij - (16 pi / Omega) sum_kn w_kn Re <P_c r_i psi_n | dpsi_j>

**The response has to be symmetrised, and it is a vector when it is.** A field
along one direction breaks the crystal's point group, so on a symmetry-reduced
k-set the three response densities are not what the whole grid would give.
``symdvscf.f90`` puts the difference back, and what it averages is not three
scalar densities but a **polar vector field** --
:meth:`~pypresso.scf.driver.Calculation.symmetrize_directional`, which is
``sym_rho``'s machinery with the components rotated into each other and without
the axial sign a magnetization carries. The assembled tensor is symmetrised
again at the end (``dielec.f90`` ends with ``symmatrix``), because a
wedge sum of a rank-2 tensor is exact only for its scalar part.

**The escape that does not work, measured.** The obvious alternative is to run
the *whole* k-grid, where a reduction has nothing to put back -- and it is only
sound if that grid is closed under the point group. A **shifted** Monkhorst-Pack
grid is not: on fcc silicon **2304 of the 3072 rotation images of a shifted
4x4x4 grid land off it**, so even the unreduced grid gives a density that is 2%
asymmetric and a ground-state energy 3.1e-5 Ry above the reduced run's. Its
dielectric tensor comes out with a diagonal of 13.848 against 13.806 and
**off-diagonal entries of 3.77 that cubic symmetry forbids**. An *unshifted*
grid is closed exactly, and running one with the symmetrisation switched off is
how this module is checked against itself.

**Ultrasoft and PAW work, and the field is where they cost something extra.**
Everywhere else in this phase they came free, because the density and ``newd``
were already differentiable functions of the state and the potential
(:mod:`pypresso.response.sternheimer`). The *position* operator is different: an
ultrasoft state's charge is not all in ``|psi|^2``, and the part inside the
augmentation spheres has a dipole of its own and moves with ``k``. That is
``adddvepsi_us.f90``, transcribed in :func:`_ultrasoft_position`, and the
augmentation dipole it needs is
:func:`~pypresso.pseudo.augmentation.augmentation_dipole`. Two traps come with
it and both are measured there and in ``PLAN.md`` P24a: the projector derivative
is the one about the atom's *own centre* (worth 2%), and ``dbecsum`` on a wedge
is a polar vector like everything else here (worth 1.6e-2 on PAW).

Against the vendored ``ph.x``: **13.806646** against 13.806689 (norm-conserving
Si), **14.325321** against 14.325270 (ultrasoft Si), **14.320211** against
14.320177 (PAW Si) and **5.756059** against 5.756182 (ultrasoft C).

What is still refused is what :func:`~pypresso.response.sternheimer.
require_a_sternheimer_regime` refuses -- metals, noncollinear magnetism, DFT+U,
spirals -- and a metal has no ``epsilon_infinity`` in any case, which is why
``pw.x`` refuses ``epsil`` for one too.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from pypresso.basis.fft import g_to_r, r_to_g
from pypresso.batching import map_k
from pypresso.pseudo.augmentation import augmentation_dipole
from pypresso.response.born import born_effective_charges, require_born_charges
from pypresso.response.mixing import DEFAULT_RESPONSE_MIXING, ResponseMixer
from pypresso.response.sternheimer import (
    SternheimerSolver,
    paw_response,
    require_a_sternheimer_regime,
)
from pypresso.response.velocity import VelocityOperator, over_kpoints
from pypresso.scf.potential import hartree, total_charge
from pypresso.system.symmetry import cartesian_rotations, symmetrize_matrix
from pypresso.units import FPI

__all__ = ["DielectricTensor", "dielectric_tensor",
           "require_a_symmetrisable_response"]

#: QE's ``alpha_mix(1)``: the weight the mixer gives the residual. It is no
#: longer the *whole* of the mixing -- :mod:`pypresso.response.mixing` builds an
#: Anderson history on top of it -- which is what makes 0.7 a safe default here
#: rather than a value each system has to be tuned to.
ALPHA_MIX = 0.7

#: Convergence on ``|ddv_scf|^2``, the quantity ``dfpt_kernels`` prints and
#: tests. QE's default ``tr2_ph`` is 1e-12 and ``si.phG.in`` asks for 1e-14.
TR2 = 1.0e-14

MAX_ITERATIONS = 40


@dataclass
class DielectricTensor:
    """``epsilon_infinity``, and what the self-consistent response cost."""

    #: ``(3, 3)``, dimensionless, in cartesian axes.
    epsilon: np.ndarray
    #: ``(nat, 3, 3)`` Born effective charges ``Z*_ij = dF_j/dE_i``, in units of
    #: the electron charge, or ``None`` when they were not asked for.
    born_charges: np.ndarray | None = None
    #: ``(3, nspin_mag, n1, n2, n3)``: the induced charge density per unit field,
    #: symmetrised, on the dense grid. It is what screening *is* -- the charge
    #: that piles up against the field and makes ``epsilon`` 13.8 rather than 1 --
    #: and it is carried out because nothing else here shows it.
    induced_density: np.ndarray | None = None
    #: ``|ddv_scf|^2`` at each iteration -- the trajectory ``ph.x`` prints, and
    #: the only intermediate quantity there is a reference for.
    history: list = field(default_factory=list)
    #: Mean CG iterations per band per solve, QE's ``av.it.``.
    average_iterations: float = 0.0
    converged: bool = False
    #: ``{"solver", "bare", "dpsi", "dvscf", "v_scf"}`` when the caller asked
    #: for them, and ``None`` otherwise. The self-consistent response *is* the
    #: expensive part of this calculation, and a **third** derivative needs the
    #: same first-order wavefunctions rather than a second solve of them
    #: (:mod:`pypresso.response.electrostriction`). Kept behind a flag because
    #: they are large -- ``3 (nspin, nk, nocc, npwx)`` complex -- and no ordinary
    #: caller wants them.
    internals: dict | None = None

    @property
    def isotropic(self) -> float:
        """``Tr(eps)/3`` -- the number quoted for a cubic crystal."""
        return float(np.trace(self.epsilon) / 3.0)

    @property
    def anisotropy(self) -> float:
        """The largest departure from a scalar tensor.

        On a cubic crystal this is zero by symmetry, and it is a *measurement*
        rather than a construction: what the group average
        (``symmetrize_directional``, ``symmatrix``) imposes is invariance under
        the operations the crystal has, which is not the same as projecting onto
        the cubic form. A wrong rotation convention, a missing fractional
        translation, or an axial sign where a polar one belongs would all survive
        the average and show up here.
        """
        return float(np.abs(self.epsilon - np.eye(3) * self.isotropic).max())


def dielectric_tensor(
    calculation,
    wavefunctions,
    eigenvalues,
    density,
    becsum=(),
    alpha_mix: float = ALPHA_MIX,
    tr2: float = TR2,
    max_iterations: int = MAX_ITERATIONS,
    threshold: float = 1.0e-12,
    mixing_mode: str = DEFAULT_RESPONSE_MIXING,
    screening: str = "full",
    born_charges: bool = True,
    keep_internals: bool = False,
    verbose: bool = False,
) -> DielectricTensor:
    """``epsilon_infinity`` and the Born charges for a converged insulator.

    Args:
        calculation: the :class:`~pypresso.scf.driver.Calculation` the states
            belong to. The normal case is an ordinary symmetry-reduced wedge,
            which the response is symmetrised over; a ``nosym`` run is accepted
            only on an **unshifted** k-grid, since a shifted one does not carry
            the symmetry itself (see the module docstring).
        wavefunctions: ``(nspin, nk, nbnd, ndim)`` from the converged run.
        eigenvalues: ``(nspin, nk, nbnd)`` or the squeezed ``(nk, nbnd)``.
        density: the converged density, which the fixed potential is built from.
        becsum: the converged projector occupations (``SCFResult.becsum``).
            Required for an ultrasoft or PAW dataset and empty otherwise: PAW's
            one-centre coefficients are built from it and cannot be rebuilt from
            the density.
        screening: which kernel screens the field. ``"full"`` is
            ``dv_of_drho`` -- Hartree plus ``f_xc``, which is what QE's
            ``solve_e`` uses and what ``epsilon_infinity`` means. ``"hartree"``
            drops the exchange-correlation term and gives the **RPA** dielectric
            constant instead, which is not a physical improvement but is the
            only way to compare this solve with a sum-over-states response run
            in RPA (:mod:`pypresso.tddft`): the two routes are identities of
            each other only when their kernels match.
    """
    eigenvalues = jnp.asarray(eigenvalues)
    if eigenvalues.ndim == 2:
        eigenvalues = eigenvalues[None]
    require_a_symmetrisable_response(calculation)
    require_a_sternheimer_regime(calculation)
    if born_charges:
        # Checked first of all: the refusal is a statement about the dataset, so
        # it should not cost a whole self-consistent response -- nor a converged
        # ``becsum`` the caller only needs for the quantity being refused.
        require_born_charges(calculation)
    if calculation.is_paw and not becsum:
        # The same rule ``VelocityOperator`` enforces for ``ddd_paw``: PAW's
        # one-centre coefficients are built from ``becsum``, and a Hamiltonian
        # without them is a different operator whose response is plausible and
        # wrong by the whole PAW correction.
        raise ValueError(
            "a PAW dielectric response needs the converged projector "
            "occupations: pass becsum = scf_result.becsum. They are part of the "
            "mixed state, not a function of the density, and the one-centre "
            "potential is built from them"
        )
    # After the refusals: they are checked on the *calculation* and must not
    # need a state, so that a caller can ask "is this supported?" cheaply.
    wavefunctions = jnp.asarray(wavefunctions)
    if wavefunctions.ndim == 3:
        wavefunctions = wavefunctions[None]
    weights, _ = calculation.occupations(eigenvalues)
    nocc = int(round(calculation.nelec / 2))
    potential = calculation.potential(density)
    # PAW's one-centre coefficients are built from ``becsum``, which is why the
    # caller has to supply it: it is part of the mixed state and not a function
    # of the density.
    _, ddd_paw = calculation.onecenter(becsum)
    hamiltonians = calculation.hamiltonian(potential.v_scf, ddd_paw)
    solver = SternheimerSolver(
        calculation, hamiltonians, wavefunctions, eigenvalues, jnp.asarray(weights),
        nocc, threshold, v_scf=potential.v_scf, becsum=becsum,
    )

    # 1. The bare perturbation, once: ``P_c r_a |psi>`` for the three cartesian
    #    directions. The commutator is computed for the whole k axis in three
    #    ``jvp`` calls and *stored*; taking it inside the per-k callback would
    #    differentiate every k-point's projectors to use one of them.
    velocity = VelocityOperator(calculation, potential.v_scf, solver.ddd_paw)
    occupied = solver.psi
    occupied_eigenvalues = solver.eigenvalues
    dipole = _augmentation_dipole(calculation)
    bare, commutators, projector_velocities = [], [], []
    for axis, direction in enumerate(np.eye(3)):
        # ``[H - eps S, r_a] = -i (dH/dk_a - eps dS/dk_a)``, both tangents from
        # one ``jvp`` -- the projector rebuild they share is the whole cost.
        derivative, overlap = velocity.both(occupied, direction)
        commutator = -1j * (
            derivative - occupied_eigenvalues[..., None] * overlap
        )
        position = _solve_stored(solver, commutator)
        # ``iucom``: ``P_c r|psi>`` *before* ``S`` and before the augmentation
        # dipole. QE stores it separately because the Born charges need it
        # (``add_for_charges``), and so does
        # :func:`pypresso.response.born.constraint_position_term`.
        commutators.append(position)
        if dipole is not None:
            derivative = velocity.projectors(direction)
            projector_velocities.append(derivative)
            position = _ultrasoft_position(
                solver, velocity, position, direction, dipole[axis], derivative
            )
        bare.append(position)

    # 2. The self-consistent loop. Only the induced term changes between
    #    iterations; the bare one above is what the whole loop is driven by.
    grid_shape = jnp.asarray(density).shape
    dvscf = jnp.zeros((3,) + grid_shape)
    drho = jnp.zeros((3,) + grid_shape)
    # PAW's one-centre coefficients respond too, and they are *not* a function
    # of the density: they come from ``becsum``. They are therefore carried and
    # mixed beside ``dvscf`` rather than rebuilt from it, exactly as
    # ``dfpt_kernels`` carries ``int3_paw`` beside ``dvscfin``.
    onecentre = None if solver.ddd_paw is None else jnp.zeros(
        (3,) + solver.ddd_paw.shape
    )
    history, total_iterations, solves = [], 0, 0
    dpsi = [None, None, None]
    converged = False

    screen = _screening_kernel(calculation, density, screening)
    mixer = ResponseMixer(mixing_mode, beta=alpha_mix)
    for iteration in range(max_iterations):
        response, becsum_response = [], []
        for axis in range(3):
            perturbation = _bare_plus_induced(
                solver, bare[axis], dvscf[axis],
                None if onecentre is None else onecentre[axis],
                iteration > 0,
            )
            solution = solver.solve(perturbation)
            dpsi[axis] = solution.dpsi
            total_iterations += solution.iterations
            solves += 1
            response.append(solver.response_density(solution.dpsi))
            if onecentre is not None:
                becsum_response.append(solver.response_becsum(solution.dpsi))


        # ``psymdvscf(drhop)``: the three responses are symmetrised *together*,
        # after the loop over directions and before the kernel, because a
        # rotation mixes them.
        symmetrised = calculation.symmetrize_directional(jnp.stack(response))
        drho = symmetrised
        # ``PAW_dusymmetrize``: the same average one level down, on the three
        # ``becsum`` responses, which PAW's one-centre potential is built from.
        becsum_response = _symmetrize_becsum_response(calculation, becsum_response)
        induced, induced_onecentre = [], []
        for axis in range(3):
            # ``dv_of_drho``: the Hartree kernel without its G = 0 component,
            # plus f_xc -- one jvp of the potential this code already writes.
            induced.append(screen(symmetrised[axis]))
            if onecentre is not None:
                # ``PAW_dpotential``, from the same becsum response.
                induced_onecentre.append(
                    paw_response(calculation, becsum_response[axis], solver.becsum)
                )

        proposed = jnp.stack(induced)
        change = float(jnp.sum((proposed - dvscf) ** 2))
        history.append(change)
        if verbose:
            print(f"  iter {iteration + 1}: |ddv_scf|^2 = {change:.3e}")
        if onecentre is None:
            dvscf = mixer.mix(dvscf, proposed)
        else:
            # **One Anderson problem over both**, not two: the one-centre
            # potential and ``dV_scf`` are coupled through the same ``dbecsum``,
            # and ``mix_pot`` concatenates them for exactly this reason.
            dvscf, onecentre = mixer.mix(
                [dvscf, onecentre], [proposed, jnp.stack(induced_onecentre)]
            )
        if change < tr2:
            converged = True
            break

    epsilon = _assemble(calculation, solver, bare, dpsi)
    charges = None
    if born_charges:
        # ``dLambda`` is a matrix element of the *same* perturbation the last
        # solve was driven by, rebuilt here at the converged ``dV_scf``.
        perturbations = [
            _bare_plus_induced(
                solver, bare[axis], dvscf[axis],
                None if onecentre is None else onecentre[axis], True,
            )
            for axis in range(3)
        ]
        charges = born_effective_charges(
            calculation, solver, jnp.asarray(wavefunctions), eigenvalues,
            jnp.asarray(weights), jnp.asarray(density), becsum, dpsi,
            perturbations, commutators,
            projector_velocities=projector_velocities or None,
        )
    internals = None
    if keep_internals:
        internals = {
            "calculation": calculation, "solver": solver, "bare": bare, "dpsi": dpsi,
            "dvscf": dvscf, "v_scf": potential.v_scf,
            "onecentre": onecentre, "weights": weights, "nocc": nocc,
            "commutators": commutators,
        }
    return DielectricTensor(
        epsilon=epsilon,
        born_charges=charges,
        internals=internals,
        induced_density=np.asarray(drho),
        history=history,
        average_iterations=total_iterations / max(solves, 1),
        converged=converged,
    )


def _screening_kernel(calculation, density, screening: str):
    """``dV_scf/drho``: the kernel the induced density is screened by.

    ``"full"`` is ``dv_of_drho.f90`` -- one ``jvp`` of ``v_of_rho``, so Hartree
    *and* ``f_xc``, which is what ``solve_e`` screens with and therefore what
    ``epsilon_infinity`` means here.

    ``"hartree"`` keeps the Hartree term alone. That is the **RPA** kernel, and
    it exists for one reason: a sum-over-states response (:mod:`pypresso.tddft`)
    solved in RPA is an *identity* of this solve only when the two kernels
    match, and the physical one does not match RPA. Comparing an RPA spectrum
    against the ``"full"`` dielectric constant measures ``f_xc``, not the
    agreement of two routes -- so the referee needs this switch to be honest.
    The exchange-correlation term is dropped, not approximated: nothing else
    about the solve changes.
    """
    density = jnp.asarray(density)
    if screening == "full":
        def screen(drho):
            _, dv = jax.jvp(
                lambda r: calculation.potential(r).v_scf, (density,), (drho,)
            )
            return dv

        return screen
    if screening != "hartree":
        raise ValueError(
            f"unknown screening kernel {screening!r}: 'full' is Hartree plus "
            "f_xc (dv_of_drho, the physical one) and 'hartree' is RPA"
        )

    gvectors = calculation.basis.dense
    cell = calculation.system.cell

    def screen(drho):
        # ``hartree`` already drops the G = 0 component, which is the same
        # reason ``dv_of_drho`` may be a plain jvp of the potential: a response
        # carries no net charge and the divergent term is not there to remove.
        drho_g = jax.vmap(r_to_g, in_axes=(0, None))(drho, gvectors.fft_index)
        v_g, _ = hartree(total_charge(drho_g), gvectors, cell)
        v_r = jnp.real(g_to_r(v_g, gvectors.fft_index, gvectors.grid))
        # The Hartree potential is built from the total charge and is the same
        # in every channel, exactly as ``_potential_of_rho`` adds it.
        return jnp.broadcast_to(v_r, drho.shape)

    return screen


def _solve_stored(solver, rhs):
    """Solve with a right-hand side that is already an array, not a callback.

    ``dvpsi_e`` applies ``-P_c^+`` to the commutator and then flips the sign, so
    the right-hand side handed to the CG is ``+P_c^+ [H - eps S, r]|psi>``.
    :meth:`SternheimerSolver.project` is ``orthogonalize``, sign included, so the
    flip is the minus below.
    """
    batch = solver.calculation.k_batch
    blocks = []
    for spin in range(solver.nspin):
        def one_k(ik, spin=spin):
            projected = -solver.project(rhs[spin][ik], ik, spin)
            return solver.solve_at(projected, ik, spin)[0]

        blocks.append(map_k(one_k, jnp.arange(rhs.shape[1]), batch=batch))
    return jnp.stack(blocks)


def _augmentation_dipole(calculation):
    """``dpqq`` as the ``(3, nkb, nkb)`` block matrix a projection contracts against.

    ``None`` for a norm-conserving run, where the augmentation charge -- and so
    its dipole -- does not exist.
    """
    if not calculation.is_ultrasoft:
        return None
    per_species = [augmentation_dipole(pseudo) for pseudo in calculation.pseudos]
    blocks = []
    for values, atoms in zip(per_species, calculation.augmentation.species_atoms):
        nh = values.shape[-1]
        blocks.append(jnp.asarray(np.broadcast_to(
            values[None], (len(atoms), 3, nh, nh)
        )))
    # ``block_matrix`` puts one atom's channels on the diagonal; the three
    # cartesian components ride along as a leading axis of each block.
    return jnp.stack([
        calculation.augmentation.block_matrix(
            tuple(None if b.shape[0] == 0 else b[:, axis] for b in blocks)
        )
        for axis in range(3)
    ])


def _ultrasoft_position(solver, velocity, position, direction, dipole,
                        derivative=None):
    """``dvpsi_e``'s ultrasoft tail: ``S P_c r|psi>`` plus the augmentation dipole.

    ``P_c r|psi>`` is what the linear solve above returns, and for an ultrasoft
    dataset it is not what the polarization is built from. QE's comment says it
    plainly: *"In the US case we obtain P_c x |psi>, but we need P_c^+ x |psi>,
    therefore we apply S again, and then subtract the additional term;
    furthermore we add the term due to dipole of the augmentation charges."*
    ``adddvepsi_us.f90``, which is Eq. 10 of Dal Corso and Mauri:

        |chi_a> = S P_c r_a |psi> + sum_ij |beta_i> [ i q_ij <d(beta_j)/dk_a|psi>
                                                    + dpqq^a_ij <beta_j|psi> ]

    Both new terms exist because an ultrasoft state's charge is not all in
    ``|psi|^2``: part of it sits in the augmentation spheres, and that part has a
    dipole (``dpqq``) and moves with ``k`` (``d(beta)/dk``). On a norm-conserving
    dataset ``q_ij`` and ``dpqq`` are both zero and this whole function is the
    identity -- which is why it is applied only when ``qq`` exists.
    """
    calculation = solver.calculation
    batch = calculation.k_batch
    if derivative is None:
        derivative = velocity.projectors(direction)   # (nk, npwx, nkb)
    vkb = calculation.projectors.vkb
    dipole = dipole.astype(vkb.dtype)
    blocks = []
    for spin, hamiltonian in enumerate(solver.hamiltonians):
        states = solver.psi[spin]

        def one_k(ik, hamiltonian=hamiltonian, states=states, spin=spin):
            overlapped = hamiltonian.apply_s(position[spin][ik], ik)
            qq = hamiltonian.projectors.qq.astype(vkb.dtype)
            becp1 = jnp.einsum("gk,ng->nk", vkb[ik].conj(), states[ik])
            becp2 = jnp.einsum("gk,ng->nk", derivative[ik].conj(), states[ik])
            coefficients = 1j * (becp2 @ qq.T) + becp1 @ dipole.T
            return overlapped + jnp.einsum("gk,nk->ng", vkb[ik], coefficients)

        blocks.append(map_k(one_k, jnp.arange(states.shape[0]), batch=batch))
    return jnp.stack(blocks)


def _bare_plus_induced(solver, bare_axis, dv, dddd_paw, include_induced: bool):
    """``P_c r|psi> + dH_induced|psi>``, as ``sternheimer_kernel`` assembles it.

    The induced part is not simply ``dV_scf(r)|psi>`` once the dataset is
    ultrasoft: it carries ``int3`` and, for PAW, the one-centre response as well
    (:func:`~pypresso.response.sternheimer.local_perturbation`).
    """
    if not include_induced:
        return lambda psi, ik, spin: bare_axis[spin][ik]

    induced = solver.perturbation(dv, dddd_paw)

    def perturbation(psi, ik, spin):
        return bare_axis[spin][ik] + induced(psi, ik, spin)

    return perturbation


def _assemble(calculation, solver, bare, dpsi) -> np.ndarray:
    """``dielec.f90``: the tensor from the bare and self-consistent responses.

        eps_ij = delta_ij - 4 (4 pi w_k / Omega) sum_n Re <P_c r_i psi | dpsi_j>

    The weights are the ground state's own ``wg``, which carries the spin
    degeneracy exactly as QE's ``wk`` does, so no factor of two appears here
    that is not in ``sum_band``.
    """
    volume = calculation.system.cell.volume
    weights = solver.weights  # (nspin, nk, nocc)
    epsilon = np.eye(3)
    for i in range(3):
        for j in range(3):
            overlap = jnp.einsum(
                "skng,skng->skn", jnp.conj(bare[i]), dpsi[j]
            )
            total = jnp.sum(weights * jnp.real(overlap))
            epsilon[i, j] -= 4.0 * FPI * float(total) / volume
    # ``dielec.f90`` ends with ``symmatrix``: a Brillouin-zone sum over the
    # wedge is exact for a scalar and not for a rank-2 tensor, so the components
    # the crystal's symmetry forbids are a residue of the reduction. On cubic
    # silicon they are what this removes.
    #
    # **Under ``nosym`` there is nothing to put back and it must not be done**,
    # which is the guard :meth:`Calculation.symmetrize_atom_tensor` was written
    # to hold in one place and this call site was still missing. It is invisible
    # on every cell committed before P28b, because there the k-grid is closed
    # under the point group and the raw tensor is already symmetric -- a no-op
    # applied to a tensor that does not need it. On the ten-site supercell,
    # whose 4x4x1 grid the three-fold does *not* preserve, it is worth 0.97 in
    # the off-diagonal entries and 1.06 in the diagonal against ``ph.x``'s
    # ``nosym`` run, with the isotropic average -- the part symmetrisation
    # cannot move -- agreeing to 5e-6.
    identity = np.eye(3)
    if not calculation.use_symmetry:
        return epsilon
    return identity + symmetrize_matrix(
        epsilon - identity, calculation.system.cell, calculation.symmetries
    )


def born_charges_zstar_eu(calculation, solver, v_scf, dpsi) -> np.ndarray:
    """``zstar_eu.f90`` transcribed: the cross-check on the mixed derivative.

    **This is not how the Born charges are computed here.** They come from
    differentiating the force along the field's response
    (:mod:`pypresso.response.born`), and this is the Fortran expression put
    beside it, in the project's usual arrangement -- the two share the field
    response ``dpsi`` and nothing else, and they agree to **1.3e-14** on
    norm-conserving silicon.

    It is **norm-conserving only** and is not guarded, because nothing calls it
    but the test: on ultrasoft silicon it gives +0.1625 where ``ph.x`` gives
    -0.07945, wrong in sign as well as size, which is the five stages
    ``zstar_eu_us.f90`` adds and :mod:`pypresso.response.born` gets from one more
    tangent.

    The bare perturbation is ``dV_bare/du |psi>`` at frozen ``v_scf``, which is
    one ``jvp`` through :meth:`~pypresso.scf.driver.Calculation.at_positions`
    (see the module docstring). ``dvqpsi_us`` returns ``+dV/du`` -- the local
    part is ``v(G) (-i)(G.u) e^{-iG.tau}``, which is the derivative with respect
    to the *displacement* with no sign of its own -- so the transcription is
    literal.
    """

    structure = calculation.system.structure
    positions = jnp.asarray(structure.positions)
    natoms = positions.shape[0]
    batch = calculation.k_batch
    psi = solver.psi
    weights = solver.weights

    def h_psi(moved_positions):
        moved = calculation.at_positions(moved_positions)
        # PAW's one-centre coefficients are constant under a displacement --
        # ``becsum`` is frozen with the states -- but they have to be *in* the
        # operator being differentiated, or the bare term is the wrong operator's.
        hamiltonians = moved.hamiltonian(v_scf, solver.ddd_paw)
        return jnp.stack([
            over_kpoints(hamiltonian, psi[spin], batch)
            for spin, hamiltonian in enumerate(hamiltonians)
        ])

    charges = np.zeros((natoms, 3, 3))
    valence = np.array([
        calculation.pseudos[t].z_valence for t in structure.types
    ])
    for atom in range(natoms):
        for cart in range(3):
            tangent = jnp.zeros_like(positions).at[atom, cart].set(1.0)
            _, bare = jax.jvp(h_psi, (positions,), (tangent,))
            for direction in range(3):
                overlap = jnp.einsum(
                    "skng,skng->skn", jnp.conj(dpsi[direction]), bare
                )
                charges[atom, direction, cart] = -2.0 * float(
                    jnp.sum(weights * jnp.real(overlap))
                )
    charges += np.eye(3)[None] * valence[:, None, None]

    # ``symtensor``: the wedge sum is exact for a scalar and not for a rank-2
    # tensor carried between atoms by the group.
    return calculation.symmetrize_atom_tensor(charges)


def _symmetrize_becsum_response(calculation, per_axis):
    """``PAW_dusymmetrize`` on the three directions' ``dbecsum``.

    A no-op when there is nothing to symmetrise -- no PAW species, or a run
    with no symmetry, where the k-grid carries it instead.
    """
    if not per_axis or calculation._becsum_symmetry is None:
        return per_axis
    rotations = cartesian_rotations(calculation.system.cell, calculation.symmetries)
    stacked = tuple(
        None if per_axis[0][species] is None
        else jnp.stack([per_axis[axis][species] for axis in range(3)])
        for species in range(len(per_axis[0]))
    )
    symmetrised = calculation._becsum_symmetry.apply_directional(stacked, rotations)
    return [
        tuple(None if values is None else values[axis] for values in symmetrised)
        for axis in range(3)
    ]


def require_a_symmetrisable_response(calculation) -> None:
    """Either the group can restore the response, or the grid must be closed.

    With symmetry on there is nothing to check: the wedge sum is completed by
    :meth:`~pypresso.scf.driver.Calculation.symmetrize_directional`. With
    ``nosym`` there is no group to average over, so the k-grid has to carry the
    symmetry itself -- and a **shifted** Monkhorst-Pack grid does not (see the
    module docstring for the measurement). An unshifted one does.
    """
    if not calculation.system.nosym and calculation.symmetries.nsym > 1:
        return
    shift = getattr(calculation.system.kpoints, "shift", None)
    if shift is not None and any(shift):
        raise NotImplementedError(
            "a directional response with nosym on a *shifted* k-grid is refused: "
            "a shifted Monkhorst-Pack grid is not closed under the point group "
            "(2304 of 3072 rotation images leave a shifted 4x4x4 grid on fcc "
            "silicon), so the response it gives is asymmetric and no "
            "symmetrisation is available to repair it. Use an unshifted grid, "
            "which is closed exactly, or drop nosym and let symdvscf's average "
            "do the work"
        )
