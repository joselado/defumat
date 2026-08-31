"""The clamped-ion piezoelectric tensor as the mixed second derivative it is.

``PLAN.md`` P50. ``e_(k)ij = dP_k/d(eps_ij) = d(sigma_ij)/dE_k``: the
polarization a strain induces, which is the same number as the stress an
electric field induces, because both are

    e_(k)ij = -(1/Omega) d^2 E / d(eps_ij) dE_k

and a mixed second derivative does not care which leg is taken first. Baroni,
de Gironcoli, Dal Corso and Giannozzi's review (`Rev. Mod. Phys. 73, 515
<https://arxiv.org/abs/cond-mat/0012092>`_, §II.C.2) states the equivalence and
records that the stress-under-a-field route is the one de Gironcoli, Baroni and
Resta took for the III-V compounds (PRL **62**, 2853 (1989)).

**It is P24b's construction with one coordinate changed.** A Born effective
charge is ``Z* = dF/dE``, and :mod:`pypresso.response.born` computes it as one
``jvp`` of the *force* along the field's response. The force is ``jax.grad`` of
the frozen-state energy in the atomic positions; the **stress** is ``jax.grad``
of the same functional in a strain (:mod:`pypresso.stress.energy`). So the
piezoelectric tensor is one ``jvp`` of the stress along the same field
response, per field direction -- three of them and the tensor is complete, at
the cost of the dielectric constant that was going to be solved anyway.

**The strain leg has no orthonormality term, and that is what makes this
cheaper than the Born charge it copies.** ``<psi|S|psi>`` is a sum over the
plane-wave sphere of ``|c_G|^2`` and the sphere is a set of *integers*; the
ultrasoft part ``qq_ij = int Q_ij(r) d3r`` is an atom-centred integral over all
space with no cell in it either. So the constraint is strain-independent, its
strain derivative vanishes identically as a function of the states too, and the
multiplier response ``dLambda`` that :mod:`pypresso.response.born` needs (QE's
``psidspsi``, ``add_dkmds``, ``add_for_charges``) has nothing to contribute
here. What is left is one term::

    Omega e_(k)ij = -d/dE_k [ dE/d(eps_ij) ] = -(d_psi d_eps E) . dpsi_k

*Units.* The field response is the one :mod:`pypresso.response.born` divides
into the force to get a charge in units of ``e``, so the same tangent through a
gradient in the dimensionless strain gives ``e bohr``; divided by the cell
volume that is ``e/bohr^2``, and :data:`~pypresso.units.E_BOHR2_TO_C_M2` takes
it to C/m^2. Nothing about the field's normalisation has to be known here, and
that is deliberate: :func:`born_charges_from_stress_route` runs this module's
own assembly with :meth:`~pypresso.scf.driver.Calculation.at_positions` in
place of :meth:`~pypresso.scf.driver.Calculation.at_strain` and must reproduce
the Born charges, which are validated against ``ph.x`` to every digit it
prints. That is the test the scale and the sign rest on.

**Proper against improper, and why every target here is a crystal where they
coincide.** The tensor above is the *improper* one -- the bare mixed second
derivative. What a measurement sees is Vanderbilt's proper piezoelectric
response (`J. Phys. Chem. Solids 61, 147 (2000)
<https://doi.org/10.1016/S0022-3697(99)00273-5>`_; QE cites it in
``PW/src/bp_c_phase.f90`` and computes nothing from it), and the two differ by
terms built from the polarization itself,

    e^proper_(k)ij = e^improper_(k)ij + delta_ki P_j - delta_ij P_k,

which arise because a strain carries the charge distribution with the cell
(``d(Omega P_k)/d(eps_ij)|_frozen = delta_ki Omega P_j``) and changes the volume
that divides it. **Both corrections vanish identically whenever the two
Cartesian labels they pair are different**, so the shear components with all
three indices distinct -- ``e_14`` of a zincblende crystal, which is its only
independent component -- carry no ambiguity at all. And they vanish for *every*
component of a crystal whose spontaneous polarization is zero, which any
non-pyroelectric class is: the class ``-43m`` of AlAs has no invariant vector.
A **polar crystal is refused by name** rather than corrected, because the
correction needs ``P`` itself and a Berry-phase polarization is not implemented
in this package (:func:`require_a_nonpolar_crystal`).

**What is left out and is not an approximation.** This is the *clamped-ion*
constant: the atoms are carried along by the strain in crystal coordinates and
are not allowed to relax. The measured constant adds the internal-strain term
``sum_a Z*_a (C^-1)_a,b Lambda_b`` of the review's Eq. (111), and the review is
also where the warning about it belongs -- the two contributions "are often of
opposite sign and close in absolute value, so that a well converged calculation
is needed in order to extract a reliable value for their sum". Every ingredient
of that term is in this package (``Z*`` from P24b, the force constants from
P25, and ``Lambda = -d^2E/du d(eps)`` which is this module's ``jvp`` with the
strain response as its tangent instead of the field's), and it is the next step
rather than part of this one.

**Elk is the only established code with a piezoelectric tensor and it takes the
expensive route.** ``piezoelt.f90`` (task 380) runs a full ground state per
strain tensor, computes the Berry-phase polarization of each, and finite-
differences them with a ``2 pi`` branch fix-up between the two -- ``nstrain``
self-consistent calculations where this is one. ``pw.x`` has no piezoelectric
tensor at all: the only occurrence of the word in the vendored tree is a
citation of Vanderbilt's paper in a comment in ``PW/src/bp_c_phase.f90``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from pypresso.forces.energy import FrozenState, energy_at
from pypresso.response.born import _raw_mixed_state
from pypresso.response.efield import (
    dielectric_tensor,
    require_a_symmetrisable_response,
)
from pypresso.response.electrostriction import (
    refined_states,
    require_converged_responses,
)
from pypresso.response.sternheimer import require_a_sternheimer_regime
from pypresso.stress.energy import require_a_differentiable_cell
from pypresso.units import E_BOHR2_TO_C_M2

__all__ = [
    "PiezoelectricTensor",
    "piezoelectric_tensor",
    "clamped_ion_piezoelectric",
    "born_charges_from_stress_route",
    "piezoelectric_zstar_eu_style",
    "piezoelectric_from_strain_response",
    "require_a_piezoelectric_tensor",
    "require_a_nonpolar_crystal",
    "polar_direction",
    "to_voigt",
]

#: ``(i, j)`` of each Voigt index, in the standard order ``xx yy zz yz xz xy``.
VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


@dataclass(frozen=True)
class PiezoelectricTensor:
    """What :func:`piezoelectric_tensor` returns.

    Attributes:
        e: ``(3, 3, 3)`` in ``e/bohr^2``, indexed ``e[k, i, j]`` -- the
            polarization along ``k`` induced by the strain component ``ij``,
            symmetric in the last two indices.
        voigt: ``(3, 6)`` in **C/m^2**, the form a table quotes ``e_14`` in:
            the columns are ``xx yy zz yz xz xy`` and there are no factors of
            two, for :class:`~pypresso.response.elastic.ElasticConstants`'
            reason -- the engineering convention puts the two on the *strain*
            (``eps_4 = 2 eps_23``), so ``P_i = e_iJ eps_J`` needs none on the
            coefficient.
        dielectric: the :class:`~pypresso.response.efield.DielectricTensor` the
            field response was taken from, so that ``epsilon_infinity`` and (if
            it was asked for) the Born charges come back with it rather than
            costing a second solve.
        converged: whether that response converged.
    """

    e: np.ndarray
    voigt: np.ndarray
    dielectric: object = None
    converged: bool = True

    @property
    def e14(self) -> float:
        """The single independent component of a zincblende crystal, in C/m^2."""
        return float(self.voigt[0, 3])


def to_voigt(e: np.ndarray) -> np.ndarray:
    """``(3, 3, 3)`` -> ``(3, 6)``: ``e_iJ = e_(i)jk``, no factors of two.

    The engineering convention puts the two on the **strain**
    (``eps_4 = 2 eps_23``), so that ``P_i = sum_jk e_(i)jk eps_jk`` is
    ``sum_J e_iJ eps_J`` with the coefficient untouched -- and equivalently
    ``e_iJ = d(sigma_J)/dE_i``, where a stress in Voigt form carries no two
    either. It is :class:`~pypresso.response.elastic.ElasticConstants`'
    convention one rank down, and it is the one every table of ``e_14`` is in.
    """
    e = np.asarray(e)
    return np.stack([e[:, i, j] for i, j in VOIGT], axis=1)


# -- the refusals ------------------------------------------------------------


def polar_direction(calculation) -> np.ndarray:
    """The projector onto the directions a spontaneous polarization may point.

    ``(1/N) sum_R R`` over the crystal's point group: a vector survives it only
    if every operation leaves it alone, which is exactly the condition for the
    class to be polar. Zero for ``-43m`` (AlAs) and for any centrosymmetric
    class (silicon), rank one for ``6mm`` (wurtzite), rank three for ``P1``.

    **Taken from the crystal rather than from the run.** A response is often
    computed with ``nosym``, whose symmetry list is the identity alone and
    would call every crystal polar; what decides whether a polarization can
    exist is the crystal, so the operations are searched for here.
    """
    from pypresso.system.symmetry import cartesian_rotations, find_symmetries

    cell = calculation.system.cell
    symmetries = find_symmetries(cell, calculation.system.structure)
    return cartesian_rotations(cell, symmetries).mean(axis=0)


def require_a_nonpolar_crystal(calculation) -> None:
    """A polar class needs ``P`` itself, which this package does not have.

    The improper-to-proper correction is ``delta_ki P_j - delta_ij P_k`` and
    every term of it is built from the spontaneous polarization. A Berry-phase
    polarization is not implemented here -- :mod:`pypresso.topology` has the
    k-string overlaps it would be built from and no polarization on top of them
    -- so a crystal whose class *permits* a polarization is refused rather than
    reported with a term missing that no symmetry check would catch.
    """
    projector = polar_direction(calculation)
    if np.abs(projector).max() > 1.0e-6:
        directions = np.linalg.matrix_rank(projector, tol=1.0e-6)
        raise NotImplementedError(
            "the piezoelectric tensor of a polar crystal is not implemented: "
            f"this crystal's point group leaves {directions} direction(s) "
            "invariant, so it may carry a spontaneous polarization, and the "
            "proper piezoelectric response then differs from the mixed second "
            "derivative computed here by delta_ki P_j - delta_ij P_k "
            "(Vanderbilt, J. Phys. Chem. Solids 61, 147 (2000)). P is a "
            "Berry-phase quantity and pypresso.topology has the overlaps but "
            "no polarization built on them. A non-polar class -- zincblende, "
            "diamond, rocksalt -- has no such term"
        )


def require_a_piezoelectric_tensor(calculation) -> None:
    """Everything that makes the mixed derivative above not be the answer."""
    require_a_symmetrisable_response(calculation)
    require_a_sternheimer_regime(calculation)
    require_a_differentiable_cell(calculation)
    require_a_nonpolar_crystal(calculation)


# -- the assembly ------------------------------------------------------------


def _frozen_energy_of(calculation, psi, eigenvalues, weights, density, becsum):
    """``E(coordinate, states)`` with the mixed state a *function* of both.

    The density and ``becsum`` are handed over as builders rather than as
    arrays, and :mod:`pypresso.response.born` gives the two reasons in full:
    the SCF's *scalar* symmetrisation must not stand inside a chain rule whose
    tangent is a response, and for an ultrasoft dataset the density itself
    carries the coordinate. Under a strain there is a third reason of the same
    kind -- the density is stored on a grid that does not move and carries a
    factor ``1/Omega``, so it responds to a strain even at frozen states.
    """
    positions = jnp.asarray(calculation.system.structure.positions)
    density_of, becsum_of = _raw_mixed_state(
        calculation, positions, psi, weights, density, becsum
    )

    def energy(moved, states):
        return energy_at(
            moved,
            FrozenState(
                wavefunctions=states, weights=weights, eigenvalues=eigenvalues
            ),
            density=density_of, becsum=becsum_of,
        )

    return energy


def _field_column(gradient, coordinate, psi, dpsi, nocc):
    """One ``jvp`` of a coordinate gradient along one field response.

    The tangent is the field's first-order wavefunction in the occupied block
    and zero in the coordinate, so what comes back is
    ``d/dE_k [dE/d(coordinate)]`` and nothing else. JAX's ``jvp`` of a
    real-valued function of complex primals is the real-linear tangent map,
    which is where the ``+ c.c.`` of the hand-derived expression comes from.
    """
    states = jnp.zeros_like(psi).at[:, :, :nocc].set(dpsi)
    _, column = jax.jvp(
        gradient, (coordinate, psi), (jnp.zeros_like(coordinate), states)
    )
    return np.asarray(column)


def clamped_ion_piezoelectric(
    calculation, psi, eigenvalues, weights, density, becsum, dpsi, nocc,
) -> np.ndarray:
    """``(3, 3, 3)`` in ``e/bohr^2``: ``e[k, i, j]``, symmetrised.

    Args:
        calculation: the one the states belong to, at its own cell.
        psi: ``(nspin, nk, nbnd, ndim)``, all bands.
        eigenvalues, weights: ``(nspin, nk, nbnd)``.
        density, becsum: the converged mixed state.
        dpsi: three ``(nspin, nk, nocc, ndim)`` field responses.
        nocc: how many bands they cover -- the solver's own ``nocc``, which is
            one number across the spin channels.
    """
    energy = _frozen_energy_of(
        calculation, psi, eigenvalues, weights, density, becsum
    )
    gradient = jax.grad(
        lambda strain, states: energy(calculation.at_strain(strain), states),
        argnums=0,
    )
    zero = jnp.zeros((3, 3))
    volume = calculation.system.cell.volume

    tensor = np.stack([
        -_field_column(gradient, zero, psi, dpsi[axis], nocc) / volume
        for axis in range(3)
    ])
    # ``symmatrix3``: a wedge sum is exact for a scalar and not for a rank-3
    # tensor, and this one is *linear* in the response -- so unlike P36's
    # screening term there is no quadratic-in-a-wedge-sum trap here, and the
    # average of the assembled tensor is the whole of the completion.
    return calculation.symmetrize_cartesian_tensor(tensor)


def born_charges_from_stress_route(
    calculation, psi, eigenvalues, weights, density, becsum, dpsi, nocc,
) -> np.ndarray:
    """The same ``jvp``, with the atoms as the coordinate: ``(nat, 3, 3)``.

    **This exists to be compared, not to be used.** Run in the position
    coordinate, :func:`clamped_ion_piezoelectric`'s assembly is the *electronic*
    half of a Born effective charge -- ``Z*`` minus its bare ionic term and
    minus the constraint term an ultrasoft dataset adds -- so for a
    norm-conserving crystal ``Z_a delta_ij - this`` is
    :func:`~pypresso.response.born.born_effective_charges` exactly. Since that
    number is validated against ``ph.x`` to every digit it prints, the
    comparison fixes the sign, the field's normalisation and the volume factor
    of the piezoelectric tensor, none of which any symmetry check would catch.

    Returned unsymmetrised and indexed ``[a, k, j]``: field along ``k``, atom
    ``a`` displaced along ``j``.
    """
    energy = _frozen_energy_of(
        calculation, psi, eigenvalues, weights, density, becsum
    )
    positions = jnp.asarray(calculation.system.structure.positions)
    gradient = jax.grad(
        lambda pos, states: energy(calculation.at_positions(pos), states),
        argnums=0,
    )
    return np.stack([
        _field_column(gradient, positions, psi, dpsi[axis], nocc)
        for axis in range(3)
    ], axis=1)


def piezoelectric_zstar_eu_style(
    calculation, solver, density, dpsi,
) -> np.ndarray:
    """``zstar_eu.f90``'s contraction with the strain in place of the atom.

    **The transcribed expression put beside the differentiated one**, in this
    project's usual arrangement. QE writes a Born charge as a bare perturbation
    against the field's self-consistent response,

        Z*_(a)ij = Z_a delta_ij - 2 sum_n w_n Re <dpsi^(E_i)_n | dV/du_(a)j psi_n>

    (:func:`~pypresso.response.efield.born_charges_zstar_eu`), and a
    piezoelectric constant is the same object with ``d/d(eps_ab)`` where that
    has ``d/du_(a)j``::

        e_(k)ab = -(2/Omega) sum_n w_n Re <dpsi^(E_k)_n | dH/d(eps_ab) psi_n>

    with no ionic term, because the frozen polarization's own strain derivative
    is ``delta_ki Omega P_j`` and vanishes for the non-polar crystals this is
    allowed on. The bare strain perturbation is
    :func:`~pypresso.response.strain._bare_strains` -- the same ``jvp`` through
    :meth:`~pypresso.scf.driver.Calculation.at_strain` P26 drives its response
    with -- so **this route needs no strain response at all**: it is three
    ``jvp`` calls of ``H|psi>`` and a contraction, and it shares with
    :func:`clamped_ion_piezoelectric` only the field response both consume.

    **The factor is 2 and not 4, and the difference is Rydberg's ``e^2``.**
    :func:`~pypresso.response.efield._assemble` builds ``epsilon`` from the same
    field response with a 4 in front, because a *susceptibility* is a
    Coulomb-normalised quantity and carries ``e^2 = 2`` where a bare mixed
    second derivative does not -- and the piezoelectric constant is a mixed
    derivative, in units of ``e/bohr^2``, exactly as the Born charge it copies
    is in units of ``e``. Taking the 4 gives a tensor that is right in every
    symmetry and twice too large, which no symmetry check sees; what says so is
    that ``zstar_eu.f90``'s own constant, on the leg that is validated against
    ``ph.x``, is 2.
    """
    from pypresso.response.strain import _bare_strains

    volume = calculation.system.cell.volume
    weights = solver.weights
    bare = _bare_strains(calculation, solver, density)
    tensor = np.zeros((3, 3, 3))
    for k in range(3):
        for a in range(3):
            for b in range(a, 3):
                overlap = jnp.einsum(
                    "skng,skng->skn", jnp.conj(dpsi[k]), bare[a, b]
                )
                value = -2.0 * float(
                    jnp.sum(weights * jnp.real(overlap))
                ) / volume
                tensor[k, a, b] = tensor[k, b, a] = value
    return calculation.symmetrize_cartesian_tensor(tensor)


def piezoelectric_from_strain_response(calculation, solver, bare, strain) -> np.ndarray:
    """The same tensor with the two perturbations interchanged -- ``(3, 3, 3)``.

    A mixed second derivative can be contracted either way round, and this is
    the other one: the **strain's** self-consistent response against the
    field's bare perturbation::

        e_(k)ab = -(2/Omega) sum_n w_n Re <b_k | dpsi^(ab)_n>

    where ``b_k`` is ``P_c r_k|psi>``, the array
    :func:`~pypresso.response.efield._assemble` builds ``epsilon`` from, and
    ``dpsi^(ab)`` is P26's strain response. It costs six more Sternheimer
    solves, which :func:`piezoelectric_zstar_eu_style` does not, and it is here
    because it is the only route that puts the **strain** response on the
    screened side -- so agreement between the three is a statement about that
    response as well as about the assembly.

    The factor is 2 for :func:`piezoelectric_zstar_eu_style`'s reason.
    """
    volume = calculation.system.cell.volume
    weights = solver.weights
    tensor = np.zeros((3, 3, 3))
    for k in range(3):
        for a in range(3):
            for b in range(3):
                overlap = jnp.einsum(
                    "skng,skng->skn", jnp.conj(bare[k]), strain.dpsi[a, b]
                )
                tensor[k, a, b] = -2.0 * float(
                    jnp.sum(weights * jnp.real(overlap))
                ) / volume
    return calculation.symmetrize_cartesian_tensor(tensor)


# -- the driver --------------------------------------------------------------


def piezoelectric_tensor(
    calculation,
    result,
    verbose: bool = False,
    allow_unconverged: bool = False,
    **response_options,
) -> PiezoelectricTensor:
    """The clamped-ion piezoelectric tensor of a converged insulator.

    Args:
        calculation: the :class:`~pypresso.scf.driver.Calculation` the run used.
            An ordinary symmetry-reduced wedge is fine and is averaged as a
            rank-3 tensor; a ``nosym`` run is accepted only on an unshifted
            grid, which is :mod:`pypresso.response.efield`'s own rule.
        result: the converged :class:`~pypresso.scf.driver.SCFResult`. Its
            states are re-diagonalised first
            (:func:`~pypresso.response.electrostriction.refined_states`).
        allow_unconverged: return an answer even when the field response did
            not converge. Off by default.
        response_options: passed to the field response.
    """
    require_a_piezoelectric_tensor(calculation)

    eigenvalues, psi = refined_states(calculation, result)
    density = jnp.asarray(result.density)
    field = dielectric_tensor(
        calculation, psi, eigenvalues, density, result.becsum,
        born_charges=False, keep_internals=True, verbose=verbose,
        **response_options,
    )
    if not allow_unconverged:
        require_converged_responses(field, None)

    internals = field.internals
    e = clamped_ion_piezoelectric(
        calculation, psi, eigenvalues, jnp.asarray(internals["weights"]),
        density, result.becsum, internals["dpsi"], internals["solver"].nocc,
    )
    return PiezoelectricTensor(
        e=e,
        voigt=to_voigt(e) * E_BOHR2_TO_C_M2,
        dielectric=field,
        converged=bool(field.converged),
    )
