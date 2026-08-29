"""Raman tensors: ``d(eps)/d(tau)`` as a mixed third derivative.

``PLAN.md`` P35. The third derivative of the energy with respect to **two
electric fields and one atomic displacement** -- the derivative of the
dielectric tensor with respect to where an atom is, which is what a non-resonant
Raman intensity is computed from. It is built the way
:mod:`pypresso.response.electrostriction` built P26's: **one ``jvp`` of the
variational second-order energy at frozen first-order wavefunctions**, which is
the 2n+1 theorem. P26 differentiated ``F_ij`` along a *strain*; this
differentiates the same functional along an atomic position::

    d(eps_ij)/d(tau_c) = jvp(F_ij)( tau, psi, rho, b ; e_c, dpsi_c, drho_c, db_c )

**Every tangent already exists.** ``dpsi_c`` and ``drho_c`` are the displacement
response :mod:`pypresso.response.phonon` solves for the dynamical matrix, ``b``
and ``u`` are the field response :mod:`pypresso.response.efield` hands back
through ``keep_internals``, and ``db_c`` is the one further Sternheimer solve
P26 already wrote (:func:`~pypresso.response.electrostriction._position_response`,
used here with the atomic positions as its geometry variable rather than a
strain). What is new is the assembly.

**What ``pw.x`` refuses and this does not.** QE reaches the Raman tensor through
``ph.x`` with ``lraman = .true.``, and refuses by name in ``phq_readin.f90`` and
``phq_setup.f90``: PAW, ultrasoft, noncollinear magnetism, Hubbard ``U``,
``lsda``, metals, ``q /= 0`` -- and

    IF (xclib_dft_is('gradient').and.(lraman.or.elop)) call errore('phq_setup', &
       'third order derivatives not implemented with GGA', 1)

The reason for the last one is ``PHonon/PH/d2mxc.f90``: the third derivative of
the exchange-correlation energy, hand-coded as a Perdew-Zunger parameterisation
of the Ceperley-Alder functional and nothing else. Here that object is not
written down at all. ``dv_of_drho`` is one ``jvp`` of ``v_of_rho`` (P24), the
screening term of ``F`` contracts two density responses against it, and
differentiating ``F`` a third time differentiates *that* -- so
``delta^3 E_Hxc/delta n^3`` is whatever the loaded functional's is, LDA or GGA,
and no third derivative is transcribed.

**The reference for this phase is broken, and establishing that had to come
before validating anything against it.** The vendored ``ph.x`` 7.5 does not
reproduce its own committed example (``PHonon/examples/example05``, generated
with v6.0): on the example's own input it gives an electro-optic tensor of
**157.87** where the reference says **40.4578**, and a Raman tensor of
**-1.8681** against **-0.78497**. Its *own* internal consistency check fails
too. ``dhdrhopsi`` obtains the k-derivative of the wavefunctions by finite
differences and prints the dielectric constant they imply beside the analytic
one; where the v6.0 reference has 8.8116 against 8.8147, the vendored build
gives **-0.288** against 8.8143. Tightening ``eth_rps`` and ``eth_ns`` by four
orders moves it by 1e-2. On the unshifted grid used here it violates the
translational sum rule below by **43%**, where this module gives 2.8e-4.

So the validation is **a finite difference of the dielectric tensor over
re-converged displaced geometries** -- which shares nothing with the third
derivative but the linear response underneath both, and is the route P26 used
for the same reason. On AlAs: **-3.118310** against the analytic **-3.118279**,
1.0e-5 relative, which is the difference's own floor.

**``chi^(2)`` and the electro-optic tensor are refused by name, and the missing
term is identified rather than fitted.** The same functional differentiated
along a *third field* would give them, and every tangent for it exists -- but
the field enters this code **only through the source term** ``b = P_c r|psi>``
and through the density; ``H`` itself is built from ``rho`` and carries no field
at all. The 2n+1 expression has a term in which the perturbing operator sits
between two first-order wavefunctions (``<u_i|r_k|u_j>``, and QE builds it in
``dvpsi_e2``/``solve_e2`` by going to *second*-order response), and nothing here
produces it: the position operator is available only as ``P_c r|psi>``, through
a commutator solve that uses ``psi``'s own eigenvalue and does not apply to a
general first-order state.

**How large that term is, is a measurement rather than an estimate**, because
its displacement counterpart *is* computed here: zeroing the geometry tangent in
the Raman derivative -- which puts it in exactly the position the field
derivative is in -- changes ``d(eps_yz)/d(tau)`` from **-3.118279** to
**-1.809983**. The explicit ``dH/d(parameter)`` term is **42% of the answer**.
And **no symmetry check catches its absence**: without it the field tensor still
vanishes identically in a centrosymmetric crystal (1.2e-13 on silicon), still
comes out in the exact zincblende form, and is still symmetric under every
permutation of its three labels to 2.5e-13 -- because the missing term has all
of those properties itself. That is why
:func:`susceptibility_field_derivative` is kept and refused rather than deleted:
it is what those tests measure.

**Refusals, inherited rather than restated.** Norm-conserving, ``nspin = 1``,
insulators, ``Gamma`` and an **unshifted** k-grid: everything P25 and P26 refuse
is refused here, through the same functions. A *shifted* grid is refused because
it is not closed under the point group, so neither running it whole nor
symmetrising a wedge of it is sound (P24).

**A symmetry-reduced wedge is not refused any more** (``PLAN.md`` P36). A Raman
tensor carries two field labels and an atom, so its wedge sum is incomplete in
all three and is completed by ``symme.f90``'s ``symtensor3`` --
:func:`~pypresso.system.symmetry.symmetrize_atom_cartesian_tensor`, applied to
the assembled tensor at the end of :func:`raman_tensors`. On AlAs the eight-point
wedge reproduces the sixty-four-point closed grid to **8.7e-14** relative.

That average is exact only because every term of ``F`` is a *linear*
Brillouin-zone sum of a covariant per-k quantity -- and one is not. The screening
term is quadratic in a k-sum, and what makes it fit the same argument is
described where it is done
(:func:`~pypresso.response.electrostriction._second_order_energy_at`); getting it
wrong is worth 2.5% here and no symmetry check sees it.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field

import jax
import jax.numpy as jnp
import numpy as np

from pypresso.response.efield import (
    dielectric_tensor,
    require_a_symmetrisable_response,
)
from pypresso.response.electrostriction import (
    _position_response,
    _project_conduction,
    _second_order_energy_at,
    refined_states,
)
from pypresso.response.phonon import (
    DisplacementResponse,
    _add_becsum,
    _bare_displacements,
    _stack_modes,
    _symmetrize_modes,
    non_variational_response,
    orthogonality_states,
    _require_a_moving_overlap_regime,
    self_consistent_response,
    symmetrize_becsum_modes,
)
from pypresso.response.sternheimer import require_a_sternheimer_regime
from pypresso.units import BOHR_TO_ANGSTROM, FPI

__all__ = [
    "RamanTensors",
    "raman_tensors",
    "susceptibility_displacement_derivative",
    "susceptibility_field_derivative",
    "permutation_asymmetry",
    "translational_residue",
    "require_a_complete_third_derivative",
]


@dataclass
class RamanTensors:
    """``d(eps)/d(tau)`` for every atom, and what the responses under it cost."""

    #: ``(nat, 3, 3, 3)`` -- ``d(eps_ij)/d(tau_(atom, cart))`` in inverse bohr,
    #: indexed ``[atom, cart, i, j]``. QE's ``ramtns`` is the same array with
    #: its axes in the order ``(i, j, cart, atom)``.
    raman: np.ndarray
    #: ``(nat, 3, 3, 3)`` -- the same tensors in Angstrom squared, which is what
    #: ``write_ramtns.f90`` puts in the dynamical-matrix file: ``Omega/(4 pi)``
    #: times :attr:`raman`, with bohr squared converted.
    raman_angstrom2: np.ndarray
    #: ``(3, 3)`` -- the dielectric tensor the derivative was taken at.
    epsilon: np.ndarray
    #: ``max |sum_atoms d(eps)/d(tau)|``. A rigid translation of the crystal
    #: cannot change ``eps``, so the sum over atoms vanishes; this is the
    #: acoustic sum rule of P25 one derivative up and, like it, is **reported
    #: rather than imposed** (Veithen, Gonze and Ghosez, `arXiv:cond-mat/0409067
    #: <https://arxiv.org/abs/cond-mat/0409067>`_, Eq. 33). Measured 8.9e-4
    #: against tensors of 3.1 on AlAs; the vendored ``ph.x`` gives 1.11.
    translational_residue: float
    #: The field response the two ``eps`` labels came from, carried so that its
    #: convergence history -- and, when it was asked for, its Born effective
    #: charges -- are available.
    field: object | None = None
    #: The :class:`~pypresso.response.phonon.DisplacementResponse` this was
    #: built on, when ``keep_internals`` asked for it. It is the same object a
    #: dynamical matrix needs and the expensive half of both, so handing it back
    #: is what lets a vibrational spectrum cost one solve rather than two.
    #: ``None`` otherwise, because ``dpsi`` is ``3 nat`` wavefunction-sized
    #: arrays and holding them is a real working set.
    displacement: object | None = None
    #: ``|ddv_scf|^2`` per iteration of the displacement response.
    phonon_history: list = _field(default_factory=list)
    converged: bool = True

    @property
    def sum_rule_relative(self) -> float:
        """:attr:`translational_residue` over the tensors' own scale."""
        return float(self.translational_residue / np.abs(self.raman).max())


def _epsilon_at(moved, psi, rho, b, u, weights, reference=None):
    """``eps_ij = delta_ij - 16 pi F_ij / Omega`` on an already-moved calculation.

    :func:`~pypresso.response.electrostriction._epsilon_at` with the geometry
    already applied, so that the same expression can be differentiated along a
    displacement, along a field, or not at all.
    """
    energies = _second_order_energy_at(moved, psi, rho, b, u, weights, reference)
    return jnp.eye(3) - 4.0 * FPI * energies / moved.system.cell.volume


# -- the third derivative along an atomic displacement -----------------------


def susceptibility_displacement_derivative(
    calculation, solver, rho, b, u, positions, dpsi, drho,
    geometry_tangent: bool = True, verbose: bool = False, ort=None,
    stored=None,
) -> np.ndarray:
    """``d(eps_ij)/d(tau_(a, c))``: the Raman tensors, one ``jvp`` per mode.

    The geometry variable is the atomic positions where P26's is a strain, which
    is the whole of the difference between this function and
    :func:`~pypresso.response.electrostriction.susceptibility_strain_derivative`.

    Args:
        ort: the occupied block of the first-order state
            (:func:`~pypresso.response.phonon.orthogonality_states`). **Part of
            the state tangent**, not an option: with ``S`` moving, the
            orthonormality constraint fixes a piece of ``dpsi`` that the
            Sternheimer solve does not produce, and it is identically zero for
            a norm-conserving dataset.
        stored: the pre-tail electric-field solution,
            ``internals["commutators"]``, threaded to
            :func:`~pypresso.response.electrostriction._position_response` so
            that ``db`` carries ``adddvepsi_us`` as well as the linear equation.
        geometry_tangent: whether to carry the displacement's own tangent
            through :meth:`~pypresso.scf.driver.Calculation.at_positions`, which
            is the ``dH/d(parameter)`` term of the 2n+1 expression. **Always
            true in use**; ``False`` puts this derivative in exactly the
            position the field derivative below is in, and is how the size of
            what that one is missing was measured -- 42% of the answer (module
            docstring).

    Returns ``(nat, 3, 3, 3)`` indexed ``[atom, cart, i, j]``.
    """
    psi, weights = solver.psi, solver.weights
    rho = jnp.asarray(rho)
    positions = jnp.asarray(positions)
    nat = positions.shape[0]

    def epsilon(moved_positions, states, density, position):
        moved = calculation.at_positions(moved_positions)
        return _epsilon_at(
            moved, states, density, position, u, weights, calculation
        )

    out = np.zeros((nat, 3, 3, 3))
    for atom in range(nat):
        for cart in range(3):
            tangent = jnp.zeros_like(positions).at[atom, cart].set(1.0)
            # The same projection the field response gets in
            # :func:`raman_tensors`, and for the same reason: a few hundred CG
            # steps without reorthogonalisation leave a part in 1e-6 of the
            # occupied manifold in the solution, which nothing at first order
            # notices and a triple product does.
            mode_psi = _project_conduction(
                psi, jnp.asarray(dpsi[atom, cart])[None],
                solver.hamiltonians, calculation.k_batch,
            )[0]
            if ort is not None:
                # **The occupied block, and it only works with its partner.**
                # With ``S`` moving, the orthonormality constraint fixes a piece
                # of the first-order state that the Sternheimer solve does not
                # produce, so the state tangent is ``P_c dpsi + ort`` -- this is
                # P39's lesson, and P39's warning with it: the block *alone*
                # moves ``d(eps)/d(tau)`` from 3.0e-2 to 8.0e-2 against a finite
                # difference. Its partner in this coordinate is the tail
                # ``db`` acquires below, and the two were found by measuring
                # ``F``'s five partial derivatives against their own finite
                # differences one at a time: with both, the ultrasoft answer is
                # **1.2e-4** and PAW's 1.2e-4, against a norm-conserving control
                # of 6.8e-4 that does not move at all (``PLAN.md`` P43).
                mode_psi = mode_psi + jnp.asarray(ort[atom, cart])
            mode_rho = jnp.asarray(drho[atom, cart])
            db = _position_response(
                calculation, solver, rho, b, tangent, mode_psi, mode_rho,
                moved_at=calculation.at_positions, geometry=positions,
                stored=stored,
            )
            carried = tangent if geometry_tangent else jnp.zeros_like(tangent)
            _, column = jax.jvp(
                epsilon, (positions, psi, rho, b),
                (carried, mode_psi, mode_rho, db),
            )
            out[atom, cart] = np.asarray(column)
            if verbose:
                print(f"  d(eps)/d(tau_{atom}{cart}): "
                      f"max = {np.abs(column).max():.6f}")
    return out


# -- the third derivative along a second field, which is incomplete ----------


def susceptibility_field_derivative(
    calculation, solver, rho, b, u, drho, verbose: bool = False
) -> np.ndarray:
    """``d(eps_ij)/dE_k`` **without its explicit field term** -- see the module.

    Kept, and refused by :func:`require_a_complete_third_derivative` rather than
    exposed, because it is what measures the thing it is missing: the tensor it
    returns is symmetry-correct in every way that can be checked without a
    reference -- identically zero in a centrosymmetric crystal, exactly
    zincblende in AlAs, permutation-symmetric to 2.5e-13 -- and still wrong,
    because the omitted term has all of those properties too.

    What it computes is the derivative through the ground state alone: ``psi``,
    ``rho`` and ``b`` all move under the field and their tangents are carried.
    What it omits is the field's *own* appearance in the operator, which in this
    code is nowhere -- ``H`` is built from ``rho``, and the field lives only in
    the source term.
    """
    psi, weights = solver.psi, solver.weights
    rho = jnp.asarray(rho)

    def epsilon(states, density, position):
        return _epsilon_at(
            calculation, states, density, position, u, weights, calculation
        )

    out = np.zeros((3, 3, 3))
    for axis in range(3):
        db = _position_response(
            calculation, solver, rho, b,
            jnp.zeros(()), u[axis], drho[axis],
            # The geometry is frozen, so it is a scalar nobody reads and its
            # tangent is zero. Going through ``at_strain(0)`` instead would put
            # a whole cell rebuild inside the trace to differentiate nothing.
            moved_at=lambda _frozen: calculation, geometry=jnp.zeros(()),
        )
        _, column = jax.jvp(
            epsilon, (psi, rho, b), (u[axis], drho[axis], db)
        )
        out[:, :, axis] = np.asarray(column)
        if verbose:
            print(f"  d(eps)/dE_{axis}: max = {np.abs(column).max():.6f}")
    return out


def require_a_complete_third_derivative() -> None:
    """Refuse ``chi^(2)`` and the electro-optic tensor by name."""
    raise NotImplementedError(
        "chi^(2) and the electro-optic tensor are not implemented: the field "
        "enters this code only through the source term P_c r|psi> and through "
        "the density, so the term of the 2n+1 expression in which the "
        "perturbing operator sits between two first-order wavefunctions "
        "(<u_i|r_k|u_j>, QE's dvpsi_e2/solve_e2) has nothing to build it from. "
        "Its displacement counterpart is computed here and is 42% of the Raman "
        "tensor, and no symmetry check catches its absence -- see "
        "pypresso.response.nonlinear's module docstring. The Raman tensor, "
        "which needs no such term, is raman_tensors()"
    )


# -- what the tensors are checked by -----------------------------------------


def permutation_asymmetry(tensor: np.ndarray) -> float:
    """``max |T_ijk - T_(any permutation)|`` for a rank-3 cartesian tensor.

    The static third derivative of one scalar energy with respect to three
    components of the *same* field is symmetric under every permutation of the
    three labels -- Kleinman's condition. ``F_ij`` is symmetric in its own two
    labels by construction, so what this measures is the exchange of the
    differentiated label with either of them.

    **It is a weaker check than it looks**, and this phase is where that was
    established: the term :func:`susceptibility_field_derivative` is missing is
    itself fully symmetric, so the asymmetry stays at 2.5e-13 with 42% of the
    answer absent. It is kept because it would still catch an index or a
    conjugation error, and it is documented as what it is.
    """
    tensor = np.asarray(tensor)
    worst = 0.0
    for axes in ((1, 0, 2), (2, 1, 0), (0, 2, 1), (1, 2, 0), (2, 0, 1)):
        worst = max(worst, float(np.abs(tensor - tensor.transpose(axes)).max()))
    return worst


def translational_residue(raman: np.ndarray) -> float:
    """``max |sum_atoms d(eps)/d(tau)|`` -- the Raman tensors' sum rule.

    Translating every atom by the same vector translates the crystal, and a
    translated crystal has the same dielectric tensor, so the sum over atoms of
    the Raman tensors vanishes. It holds only for the fully screened response
    and it shares no machinery with the assembly it checks.
    """
    return float(np.abs(np.asarray(raman).sum(axis=0)).max())


# -- the driver --------------------------------------------------------------


def raman_tensors(
    calculation,
    result,
    born_charges: bool = False,
    keep_internals: bool = False,
    verbose: bool = False,
    allow_unconverged: bool = False,
    **response_options,
) -> RamanTensors:
    """``d(eps)/d(tau)`` for a converged insulator.

    Args:
        calculation: the :class:`~pypresso.scf.driver.Calculation` the run used.
            A symmetry-reduced k-set is **refused** (module docstring), as is a
            shifted grid run with ``nosym``.
        result: the converged :class:`~pypresso.scf.driver.SCFResult`. Its
            states are re-diagonalised first
            (:func:`~pypresso.response.electrostriction.refined_states`), which
            is not optional for a third derivative.
        born_charges: also assemble the Born effective charges from the field
            response, which costs nothing beyond what is already solved and is
            what an *infrared* activity needs beside the Raman one. They arrive
            on ``field.born_charges``.
        keep_internals: hand back the displacement response on
            :attr:`RamanTensors.displacement`, so that a dynamical matrix can be
            built without solving it a second time
            (:func:`~pypresso.response.phonon.dynamical_matrix` takes it).
        allow_unconverged: return an answer even when a first-order response did
            not converge. Off, and for the reason
            :func:`~pypresso.response.electrostriction.require_converged_responses`
            documents: a diverged first-order solution consumed by a *third*
            derivative is wrong at first order, not at second.
        response_options: passed to the two self-consistent responses.
    """
    require_a_symmetrisable_response(calculation)
    require_a_sternheimer_regime(calculation)
    # **Ultrasoft and PAW are in as of P43**, so what is checked here is the
    # same *combination* the dynamical matrix refuses -- a metal with a moving
    # overlap -- and not the dataset. The strain-coordinate third derivatives
    # (elastic constants, electrostriction, the elasto-optic tensor) still
    # carry ``require_norm_conserving``: they share ``_position_response`` with
    # this one, but neither the occupied block's analogue under a strain nor
    # the tail's behaviour there has been measured.
    _require_a_moving_overlap_regime(calculation)

    eigenvalues, psi = refined_states(calculation, result)
    density = jnp.asarray(result.density)

    field = dielectric_tensor(
        calculation, psi, eigenvalues, density, result.becsum,
        born_charges=born_charges, keep_internals=True, verbose=verbose,
        **response_options,
    )
    internals = field.internals
    solver = internals["solver"]
    # **Handed over unprojected.** ``F`` projects both itself, and with the
    # *right* projector for each: a state takes ``1 - sum |psi><psi| S`` and a
    # right-hand side takes ``1 - sum S|psi><psi|``. Pre-projecting here applied
    # the state form to both, which for an ultrasoft dataset is not idempotent
    # against the other and undoes it -- measured on the identity below,
    # 2.2e-3 against 3.4e-10.
    b = jnp.stack(internals["bare"])
    u = jnp.stack(internals["dpsi"])
    if not (field.converged or allow_unconverged):
        raise ValueError(
            "the electric-field response did not converge, and a third "
            "derivative of a diverged first-order solution is wrong at first "
            "order rather than at second: raise max_iterations, lower "
            "alpha_mix, or pass allow_unconverged=True for a diagnostic run"
        )

    positions = jnp.asarray(calculation.system.structure.positions)
    bare = _bare_displacements(calculation, solver, internals["v_scf"], positions)
    ort = orthogonality_states(calculation, solver, positions)
    (rho_moved, bec_moved), (rho_ort, bec_ort) = non_variational_response(
        calculation, positions, psi, solver.weights, density, result.becsum, ort,
    )
    drhous = becsumort = None
    if rho_moved is not None:
        drhous = _stack_modes(_symmetrize_modes(calculation, rho_moved)) + \
            _stack_modes(_symmetrize_modes(calculation, rho_ort))
        becsumort = _add_becsum(
            symmetrize_becsum_modes(calculation, bec_moved),
            symmetrize_becsum_modes(calculation, bec_ort),
        )
    dpsi, drho, history, _, phonon_converged, _ = self_consistent_response(
        calculation, solver, bare, density, positions=positions,
        becsumort=becsumort, drhous=drhous, verbose=verbose, **response_options,
    )
    if not (phonon_converged or allow_unconverged):
        raise ValueError(
            "the displacement response did not converge; see the electric "
            "field's message above for why that is fatal here"
        )

    tensors = susceptibility_displacement_derivative(
        calculation, solver, density, b, u, positions, dpsi, drho,
        verbose=verbose, ort=ort,
        stored=jnp.stack(internals["commutators"]),
    )
    # ``symtensor3``, and it is a no-op on the closed-grid runs this phase was
    # validated on -- :meth:`~pypresso.scf.driver.Calculation.symmetrize_atom_cartesian_tensor`
    # returns its argument when the run set ``nosym``. On a wedge it is what
    # completes the sum (module docstring).
    tensors = calculation.symmetrize_atom_cartesian_tensor(tensors)
    volume = float(calculation.system.cell.volume)
    return RamanTensors(
        raman=tensors,
        raman_angstrom2=tensors * volume / FPI * BOHR_TO_ANGSTROM**2,
        epsilon=np.asarray(field.epsilon),
        translational_residue=translational_residue(tensors),
        field=field,
        displacement=None if not keep_internals else DisplacementResponse(
            dpsi=dpsi, drho=drho, history=history, converged=phonon_converged,
        ),
        phonon_history=history,
        converged=bool(field.converged and phonon_converged),
    )
