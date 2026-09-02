"""Born effective charges as the mixed second derivative they are.

``Z*_(a)ij = dF_(a)j / dE_i = -d^2 E / du_(a)j dE_i``, and writing it that way
rather than as a formula is what makes an ultrasoft dataset cost one more
tangent instead of ``zstar_eu_us.f90``'s five further stages.

**The route.** P15 writes the total energy as a function of the positions at a
*frozen* electronic state, with the orthonormality constraint carried explicitly
and its multipliers among the frozen variables
(:func:`~defumat.forces.energy.frozen_energy`), and the force is ``jax.grad`` of
it. P25 differentiates that gradient once more along a tangent that carries the
positions and the states together, and gets the force constants. This module is
the same operation with the *electric field's* response as the tangent:

    L(u, psi, Lambda)      the stationary functional, P15's
    dE/du_j    = d_j L                                        (the force)
    d^2E/du_j dE_i = d_E d_j L + (d_psi d_j L).dpsi_i
                              + (d_Lambda d_j L).dLambda_i

so **one ``jvp`` of the force per field direction returns a whole column of
``Z*``** -- all ``3 nat`` of it, since the position tangent is zero and only the
electronic one is switched on. Three ``jvp`` calls and the tensor is complete.

**Why this is affordable here and a ``Gamma`` phonon on the same dataset is
not.** The term that P25's identity leaves over is
``-<psi|dS/du_j|psi> . dLambda_i``: it vanishes when ``S`` does not move with
the atoms, which is the norm-conserving case, and otherwise needs the
multipliers' own response. For a *phonon* both legs of the second derivative
move ``S``, and ``dLambda`` is then a response that has to be solved for. For a
Born charge only the ``u`` leg moves ``S``; the ``E`` leg's ``dLambda`` is a
matrix element of objects the field response has already built. That asymmetry
is the whole reason ultrasoft Born charges are in and ultrasoft phonons are
still refused (:func:`~defumat.response.phonon.require_norm_conserving`).

Four things had to be supplied to the tangent, and each is a term QE writes a
routine for:

======================================  =====================================
QE                                      here
======================================  =====================================
``zstar_eu``'s main term                the ``dpsi`` half of the ``jvp``
``iudrhous`` x ``dv_of_drho``           the ``dLambda`` half, screening part
``psidspsi``                            the ``dLambda`` half, bare part
``add_dkmds``                           ``jax.grad`` of :func:`frozen_polarization`
``add_for_charges``                     :func:`constraint_position_term`
======================================  =====================================

**The mixed state has to stay a *function* of where the atoms are.** The density
and ``becsum`` are handed to :func:`~defumat.forces.energy.energy_at` as
builders rather than as arrays, for two reasons that pull the same way. One,
:meth:`~defumat.scf.driver.Calculation.density` ends with the SCF's *scalar*
symmetrisation, and a response must not go through it -- left to the chain rule
the state tangent is averaged over the full group of the unperturbed crystal and
most of it is projected away (measured: **-3.96** instead of -0.0757 on
norm-conserving silicon, so it is not subtle). Two, and this is what P25's
constant-array override could not do, for an ultrasoft dataset the density
*itself* moves with the atoms -- the augmentation charge ``Q_ij(r - tau)`` is
part of it -- and freezing it as an array deletes exactly the dependence the
second derivative is made of. The builders here rebuild the raw, unsymmetrised
mixed state at the moved calculation and add a constant offset, so the *value*
is the converged symmetrised one and the *tangent* is the wedge's own. That is
``zstar_eu``'s convention, and the wedge sum it produces is completed by
``symtensor`` on the assembled tensor at the end.

**One term is transcribed rather than differentiated, and the reason is a
coordinate singularity** -- the same exception ``dpqq`` already is (P24a).
``dLambda_mn = w_n <psi_m|X|psi_n>`` needs the occupied-occupied block of the
position operator, and ``<psi_m|r|psi_n>`` is the Berry connection: not a matrix
element at all in a periodic cell. It is nevertheless well defined *in the
combination it enters*, contracted with ``<psi_n|dS/du|psi_m>``, because
``dS/du`` is localised on one atom. ``add_for_charges.f90`` is that combination
and :func:`constraint_position_term` is it transcribed. It is worth **0.55** on
ultrasoft silicon -- the difference between +0.47 and -0.079 -- so it is not
optional and it is not small.

Against the vendored ``ph.x``:

===================  ==============  ==============  =========
case                 here            ``ph.x``        difference
===================  ==============  ==============  =========
norm-conserving Si   -0.0757150      -0.07571        every digit
ultrasoft Si         -0.0794420      -0.07945        **8e-6**
===================  ==============  ==============  =========

and the norm-conserving number agrees with the transcribed ``zstar_eu.f90``
beside it (:func:`~defumat.response.efield.born_charges_zstar_eu`) to
**1.3e-14**, which is the regression gate on the whole assembly: every term this
module adds for an ultrasoft dataset has to switch itself off for a
norm-conserving one, and that is the test that says it does.

**PAW is refused by name** (:func:`require_born_charges`) and the measurement
behind the refusal is in the docstring there: everything above gets it to
-0.078293 against ``ph.x``'s -0.07961, 1.3e-3, and what is left is QE's fifth
stage -- ``int3_paw`` against ``becsumort``, the one-centre twin of
:func:`constraint_position_term`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from defumat.basis.interpolate import to_dense
from defumat.batching import map_k
from defumat.forces.energy import FrozenState, frozen_energy
from defumat.scf.density import becsum as becsum_of, sum_band

__all__ = ["born_effective_charges", "require_born_charges"]


def require_born_charges(calculation) -> None:
    """PAW is refused, and the gap is one named term rather than a whole method.

    Everything this module does works for a PAW dataset up to **1.3e-3**:
    -0.078293 against the vendored ``ph.x``'s -0.07961, where the ultrasoft case
    of the *same* assembly reaches 8e-6 and the norm-conserving one is exact.
    The 1.3e-3 is QE's fifth stage, ``zstar_eu_us.f90``'s last block:

        zstareu0 -= int3_paw(ih,jh,na,is,jpol) * becsumort(ijh,na,is,mode)

    -- the one-centre twin of :func:`constraint_position_term`, pairing the
    field's response of the one-centre coefficients (which
    :func:`~defumat.response.sternheimer.paw_response` already produces) with
    the displacement's orthogonality ``becsum``. It is the term that has no
    counterpart in the plane-wave part, because the constraint
    ``<psi|S|psi> = 1`` carries the whole of ``becsum``'s share of the energy for
    an ultrasoft dataset and not for a PAW one, whose one-centre energy is a
    second, independent function of ``becsum``.

    **Two candidates were measured and both were rejected**, which is what makes
    this a refusal with a shape rather than an open question (``PLAN.md`` P39a).
    QE's fifth stage assembled directly from the objects P39 built --
    ``paw_response`` along the field's ``dbecsum`` for ``int3_paw``,
    :func:`~defumat.response.phonon.non_variational_response` for
    ``becsumort`` -- comes to **0.004882**, where the gap to close is 0.001317:
    3.7 times too large in either sign, so what is missing here is not that
    term, and the reason is that the Lagrangian already carries it through the
    multiplier tangent. And symmetrising the ``becsum`` response before it
    enters the one-centre energy -- P36's "the value inside a nonlinear
    functional must be the full-zone object", where the raw and symmetrised
    field responses differ by **19 to 46 per cent** -- moves PAW the *wrong*
    way, from 1.3e-3 to **2.8e-3**, while leaving the norm-conserving case
    exact to every digit and the ultrasoft one at 1.0e-5. The raw chain-rule
    tangent is the better one and ``symtensor`` really does complete it.

    It is refused rather than returned because 1.3e-3 is sixteen times the last
    digit ``ph.x`` prints, and because the sign of that term could not be settled
    from the Fortran with confidence: ``compute_drhous`` builds its ``dbecsum``
    without the one-half that the orthogonality correction
    ``dpsi^ort = -1/2 sum_m psi_m <psi_m|dS/du|psi_n>`` carries, and
    ``addusdbec`` accumulates one of the two cross terms rather than both, so the
    factor is a product of two conventions rather than a derivation. Fitting it
    to the reference would make this number a measurement of ``ph.x`` and not of
    the code.
    """
    if calculation.is_paw:
        raise NotImplementedError(
            "Born effective charges with a PAW pseudopotential are not "
            "implemented: zstar_eu_us.f90's one-centre stage (int3_paw against "
            "becsumort) is missing, which leaves -0.078293 against ph.x's "
            "-0.07961 -- 1.3e-3, sixteen times the last digit it prints. "
            "Norm-conserving and ultrasoft datasets are implemented and match "
            "ph.x to 8e-6. Pass born_charges=False for the dielectric tensor "
            "alone, which *is* right for PAW"
        )


def born_effective_charges(
    calculation, solver, psi, eigenvalues, weights, density, becsum,
    dpsi, perturbations, commutators, projector_velocities=None,
) -> np.ndarray:
    """``(nat, 3, 3)``: ``Z*[a, i, j] = dF_(a)j / dE_i``, symmetrised.

    Args:
        calculation: the :class:`~defumat.scf.driver.Calculation`, at the
            positions the states belong to.
        solver: the :class:`~defumat.response.sternheimer.SternheimerSolver`
            the field response was solved with -- it carries the occupied block
            and the k-batching.
        psi: ``(nspin, nk, nbnd, ndim)``, all bands.
        eigenvalues: ``(nspin, nk, nbnd)`` -- the multipliers' diagonal.
        weights: ``(nspin, nk, nbnd)``, QE's ``wg``.
        density, becsum: the converged mixed state.
        dpsi: three ``(nspin, nk, nocc, ndim)`` field responses.
        perturbations: three callables ``(states, ik, spin) -> dV_E|states>``,
            the *unprojected* right-hand side the Sternheimer solve was driven
            by -- bare plus induced. Their occupied-occupied matrix elements are
            ``dLambda``.
        commutators: three ``P_c r|psi>``, **before** ``S`` and before
            ``adddvepsi_us`` -- QE's ``iucom``, which is what
            :func:`constraint_position_term` needs.
        projector_velocities: three ``d(vkb)/dk_a`` about each atom's own
            centre, which :mod:`defumat.response.efield` has already built for
            ``adddvepsi_us`` and which :func:`frozen_polarization` needs.
            Required for an ultrasoft dataset and ignored otherwise.
    """
    require_born_charges(calculation)
    structure = calculation.system.structure
    positions = jnp.asarray(structure.positions)
    natoms = positions.shape[0]
    nocc = solver.nocc

    density_of, becsum_of_ = _raw_mixed_state(
        calculation, positions, psi, weights, density, becsum
    )

    def energy(pos, states, multipliers):
        return frozen_energy(
            calculation, pos,
            FrozenState(
                wavefunctions=states, weights=weights, eigenvalues=eigenvalues
            ),
            density=density_of, becsum=becsum_of_, multipliers=multipliers,
        )

    gradient = jax.grad(energy, argnums=0)
    ground = _ground_state_multipliers(weights, eigenvalues, psi.dtype)

    # The explicit ``d_E d_u`` term: how the polarization moves when the atoms do
    # and the states do not. ``(3, nat, 3)``, field axis leading.
    operator = _position_operator(calculation, projector_velocities)
    frozen = np.asarray(jax.jacfwd(
        lambda pos: frozen_polarization(calculation, pos, psi, weights, operator)
    )(positions))

    charges = np.zeros((natoms, 3, 3))
    for axis in range(3):
        states = jnp.zeros_like(psi).at[:, :, :nocc].set(dpsi[axis])
        multipliers = _multiplier_response(
            solver, perturbations[axis], weights, psi.shape[2], nocc
        )
        _, column = jax.jvp(
            gradient, (positions, psi, ground),
            (jnp.zeros_like(positions), states, multipliers),
        )
        charges[:, axis, :] = (
            frozen[axis]
            - np.asarray(column)
            - np.real(constraint_position_term(
                calculation, positions, solver, weights, commutators[axis]
            ))
        )

    # ``symtensor``: a wedge sum is exact for a scalar and not for a rank-2
    # tensor the group carries between atoms.
    return calculation.symmetrize_atom_tensor(charges)


def _ground_state_multipliers(weights, eigenvalues, dtype):
    """``Lambda_mn = delta_mn w_n eps_n``: the multipliers at the solution.

    Stationarity of ``L`` gives ``Lambda_mn = w_n <psi_m|H|psi_n>``, which is
    this at a converged ground state. Passing it explicitly rather than letting
    :func:`~defumat.forces.energy.energy_at` use its diagonal form is what puts
    the matrix on the tangent's argument list.
    """
    identity = jnp.eye(weights.shape[-1], dtype=dtype)
    return (weights * eigenvalues)[..., :, None].astype(dtype) * identity


def _raw_mixed_state(calculation, positions, psi, weights, density, becsum):
    """Builders for ``(rho, becsum)`` that move with the atoms and are not symmetrised.

    Returned as a pair of callables in
    :func:`~defumat.forces.energy._mixed_state_part`'s convention. Each rebuilds
    the raw quantity at the *moved* calculation and adds a constant offset fixed
    so that the value at the undisplaced geometry is the converged, symmetrised
    one. The offset is a constant, so it does not touch the tangent; what it buys
    is that the functional is evaluated at the right density -- the Hartree and
    exchange-correlation kernels are nonlinear, and a wedge's raw density is not
    the crystal's.
    """
    def raw_becsum(moved, states, occupations):
        if not moved.is_ultrasoft:
            return ()
        return becsum_of(
            states, moved.projectors.vkb, occupations, moved.species_channels,
            moved.k_batch,
        )

    def raw_density(moved, states, occupations, parts):
        smooth, dense = moved.basis.smooth, moved.basis.dense
        bands = sum_band(
            states, moved.fft_index, smooth.grid, occupations,
            moved.system.cell, moved.k_batch,
        )
        return moved.augmented(to_dense(bands, smooth, dense), parts)

    here = calculation.at_positions(positions)
    becsum_offset = tuple(
        None if value is None else jnp.asarray(value) - raw
        for value, raw in zip(becsum, raw_becsum(here, psi, weights))
    )

    def becsum_builder(moved, states, occupations):
        return tuple(
            None if raw is None else raw + offset
            for raw, offset in zip(raw_becsum(moved, states, occupations),
                                   becsum_offset)
        )

    offset = jnp.asarray(density) - raw_density(
        here, psi, weights, becsum_builder(here, psi, weights)
    )

    def density_builder(moved, states, occupations, parts):
        return raw_density(moved, states, occupations, parts) + offset

    return density_builder, becsum_builder


def frozen_polarization(calculation, positions, psi, weights, operator):
    """``Omega P`` at frozen states, as a function of where the atoms are.

    The ions contribute ``sum_a Z_a tau_a`` and the electrons contribute their
    dipole with a minus sign. **At frozen coefficients the smooth charge does not
    move at all** -- the plane-wave basis is a set of Miller indices and the
    coefficients are held -- so the only electronic term that survives
    differentiation is the augmentation charge's, and its operator is exactly the
    one ``adddvepsi_us.f90`` adds to the position operator:

        A_a = sum_ij |beta_i> [ i q_ij <d(beta_j)/dk_a| + dpqq^a_ij <beta_j| ]

    ``jax.grad`` of ``-sum_kn w <psi|A_a|psi>`` is ``add_dkmds.f90``, three
    hundred lines of Fortran that are not transcribed.

    **The projectors move with the atoms through their structure factor and
    nothing else**, and both ``vkb`` and ``d(vkb)/dk`` about the atom's own
    centre carry the *same* factor: what the own-centre convention leaves out is
    precisely ``d/dk_a`` of ``e^{-i(k+G).tau}``, so what remains is a radial and
    angular function of ``k+G`` times that phase. The ``u`` dependence is
    therefore written down here as a phase rather than rebuilt, which keeps a
    ``jvp`` of the projectors out of a ``jacfwd`` over the positions.

    The naive alternative -- the moment of ``Q_ij(r - tau)``, that is
    ``tau_a q_ij + dpqq^a_ij`` in place of ``A_a`` -- is **wrong, by 0.38** on
    ultrasoft silicon. ``i q_ij <d(beta_j)/dk_a|`` is not ``tau_a q_ij <beta_j|``:
    the own-centre derivative excludes the structure factor's ``-i tau``, and the
    difference is the projector's internal ``k`` dependence, which is a real part
    of the position operator of a periodic crystal.
    """
    valence = jnp.asarray([
        float(calculation.pseudos[t].z_valence)
        for t in calculation.system.structure.types
    ])
    ionic = jnp.einsum("a,ac->c", valence, positions)
    if operator is None:
        return ionic
    kg, atom_of, vkb0, derivatives, qq, dipole = operator
    delta = positions - jnp.asarray(calculation.system.structure.positions)
    phase = jnp.take(
        jnp.exp(-1j * jnp.einsum("kgc,ac->kga", kg, delta)), atom_of, axis=-1
    )
    vkb = vkb0 * phase
    return ionic - jnp.stack([
        _augmentation_expectation(
            calculation, psi, weights, vkb, derivatives[axis] * phase, qq,
            dipole[axis],
        )
        for axis in range(3)
    ])


def _augmentation_expectation(calculation, psi, weights, vkb, dvkb, qq, dipole):
    """``sum_kn w_n <psi_n| A_a |psi_n>`` for one cartesian direction."""
    total = jnp.zeros(())
    for spin in range(psi.shape[0]):
        states = psi[spin]

        def one_k(ik, states=states):
            projected = jnp.einsum("gc,ng->nc", vkb[ik].conj(), states[ik])
            derived = jnp.einsum("gc,ng->nc", dvkb[ik].conj(), states[ik])
            return jnp.real(
                jnp.einsum("ni,ij,nj->n", projected.conj(), 1j * qq, derived)
                + jnp.einsum("ni,ij,nj->n", projected.conj(), dipole, projected)
            )

        values = map_k(one_k, jnp.arange(states.shape[0]), batch=calculation.k_batch)
        total = total + jnp.sum(weights[spin] * values)
    return total


def _position_operator(calculation, projector_velocities):
    """The pieces :func:`frozen_polarization` contracts, or ``None`` if there are none.

    ``d(vkb)/dk`` about each atom's own centre costs one ``jvp`` of the
    projectors per direction; :mod:`defumat.response.efield` has already paid
    for it building ``adddvepsi_us``' position operator, so it is handed in
    rather than recomputed.
    """
    from defumat.response.efield import _augmentation_dipole

    if not calculation.is_ultrasoft:
        return None
    if projector_velocities is None:
        raise ValueError(
            "an ultrasoft Born charge needs d(vkb)/dk about each atom's own "
            "centre (VelocityOperator.projectors), which is the same object "
            "adddvepsi_us uses and which the caller already has"
        )
    projectors = calculation.projectors
    vkb = projectors.vkb
    dipole = _augmentation_dipole(calculation)
    return (
        jnp.asarray(calculation.projector_core.kg),
        jnp.asarray(np.asarray(list(projectors.atom_of_channel))),
        vkb,
        [jnp.asarray(d) for d in projector_velocities],
        jnp.asarray(projectors.qq).astype(vkb.dtype),
        [dipole[axis].astype(vkb.dtype) for axis in range(3)],
    )


def _multiplier_response(solver, perturbation, weights, nbnd, nocc):
    """``dLambda_mn = w_n <psi_m|dV_E|psi_n>`` -- ``psidspsi`` plus its screening.

    Stationarity of ``L`` fixes ``Lambda_mn = w_n <psi_m|H|psi_n>``, diagonal at
    the ground state. To first order in the field, ``<psi_m|dH|psi_n>`` survives
    and the two terms carrying ``dpsi`` do not, because the field's response is
    orthogonal to the occupied manifold in the ``S`` metric. So ``dLambda`` is a
    matrix element of the *same* perturbation the Sternheimer solve was driven
    by -- nothing new is computed, only contracted differently.

    It is the whole of QE's first two ultrasoft stages at once: the induced part
    of ``dV_E`` gives ``int dvscf . drhous`` (the constraint's own density
    against the field's induced potential) and the bare part gives ``psidspsi``.
    Both vanish identically for a norm-conserving dataset, where the thing they
    multiply -- ``<psi_m|dS/du|psi_n>`` -- is zero.

    **The index order is not a convention.** ``Lambda_mn`` pairs with
    ``<psi_n|S|psi_m>``, so the weight belongs to the *column*. Transposing it
    costs 0.28 on ultrasoft silicon and nothing at all on a norm-conserving one,
    where the term is zero either way -- which is exactly the kind of error the
    norm-conserving regression gate cannot see.
    """
    occupied = solver.psi
    blocks = []
    for spin in range(solver.nspin):
        def one_k(ik, spin=spin):
            applied = perturbation(occupied[spin][ik], ik, spin)
            return jnp.einsum("mg,ng->mn", jnp.conj(occupied[spin][ik]), applied)

        blocks.append(map_k(
            one_k, jnp.arange(occupied.shape[1]), batch=solver.calculation.k_batch
        ))
    matrix = jnp.stack(blocks) * weights[:, :, None, :nocc]
    shape = (occupied.shape[0], occupied.shape[1], nbnd, nbnd)
    return jnp.zeros(shape, dtype=matrix.dtype).at[:, :, :nocc, :nocc].set(matrix)


def constraint_position_term(calculation, positions, solver, weights, commutator):
    """``sum_n w_n <psi_n| dS/du | P_c r psi_n>`` -- ``add_for_charges.f90``.

    ``(nat, 3)`` complex, one entry per displaced coordinate.

    **The one term here that is transcribed, and the reason is the same
    coordinate singularity ``dpqq`` has.** ``dLambda_mn = w_n <psi_m|X|psi_n>``
    wants the occupied-occupied block of the position operator, and
    ``<psi_m|r|psi_n>`` is the Berry connection -- gauge-dependent, and not a
    matrix element of any operator in a periodic cell.
    :func:`_multiplier_response` therefore reaches only the part of it that
    ``P_c^+ r|psi>`` carries, whose occupied block is the augmentation dipole and
    is small. What is left is finite only in the combination it appears in:
    contracted with ``<psi_n|dS/du|psi_m>``, which is localised on one atom.
    ``add_for_charges`` is that combination, and this is it.

    It is worth **0.55** on ultrasoft silicon -- the difference between +0.47 and
    -0.079 -- and it is identically zero for a norm-conserving dataset, where
    ``S = 1``.

    ``commutator`` is ``P_c r|psi>`` **before** ``S`` and before
    ``adddvepsi_us``: QE stores it separately in ``iucom`` for exactly this use,
    and :mod:`defumat.response.efield` keeps it for the same reason.
    """
    natoms = positions.shape[0]
    if not calculation.is_ultrasoft:
        return np.zeros((natoms, 3))
    occupied = solver.psi
    nocc = solver.nocc
    batch = calculation.k_batch

    def sandwich(pos):
        moved = calculation.at_positions(pos)
        vkb = moved.projectors.vkb
        qq = moved.projectors.qq.astype(vkb.dtype)
        total = jnp.zeros((), dtype=vkb.dtype)
        for spin in range(occupied.shape[0]):
            def one_k(ik, spin=spin):
                left = jnp.einsum("gc,ng->nc", vkb[ik].conj(), occupied[spin][ik])
                right = jnp.einsum("gc,ng->nc", vkb[ik].conj(), commutator[spin][ik])
                return jnp.einsum("ni,ij,nj->n", left.conj(), qq, right)

            values = map_k(one_k, jnp.arange(occupied.shape[1]), batch=batch)
            total = total + jnp.sum(weights[spin][:, :nocc] * values)
        return total

    out = np.zeros((natoms, 3), dtype=complex)
    for atom in range(natoms):
        for cart in range(3):
            tangent = jnp.zeros_like(positions).at[atom, cart].set(1.0)
            _, derivative = jax.jvp(sandwich, (positions,), (tangent,))
            out[atom, cart] = complex(derivative)
    return out
