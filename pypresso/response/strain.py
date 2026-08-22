"""The response to a homogeneous strain: ``dpsi/d(eps)``, ``drho/d(eps)``.

``PLAN.md`` P26. The third perturbation of the linear-response family, after the
electric field (P24) and the atomic displacement (P25), and the one that carries
a **rank-2** label rather than a direction or a direction-and-an-atom.

**Nothing here is transcribed, and the reason is the same one twice.** Abinit's
strain perturbation is a phase of its own -- the metric-tensor formulation of
Hamann, Wu and Vanderbilt (`PRB 71, 035117 <https://arxiv.org/abs/cond-mat/0409269>`_,
`PRB 72, 035105 <https://arxiv.org/abs/cond-mat/0501548>`_), whose whole
difficulty is that a strain moves the plane-wave basis, so the nonlocal
projectors' strain derivative has to be derived by hand in reduced coordinates.
:meth:`~pypresso.scf.driver.Calculation.at_strain` is already written in exactly
those coordinates -- what is stored is a set of **Miller indices** and the
sphere is frozen while differentiating (P11) -- so the bare perturbation is one
``jvp`` through it, the way ``dvqpsi_us`` became one ``jvp`` through
``at_positions``. The metric-tensor trick is not implemented here; it is what
the data structure already was.

**Two things separate this from the displacement perturbation, and both are
silent.**

*The frozen potential is not a frozen array of values.* For a displacement,
``dV_H/du`` at frozen ``rho`` is zero -- the Hartree kernel does not know where
the atoms are -- so ``_bare_displacements`` can hold ``v_scf`` fixed as an
array. Under a strain it *is* not: the Hartree kernel is ``4 pi / G^2`` and
``G`` moves with the cell, and a GGA's gradient moves with it too. The bare term
here therefore rebuilds the potential **from the frozen density, inside the
trace**::

    bare = jvp[ eps -> at_strain(eps).hamiltonian(at_strain(eps).potential(rho)) ]

so that ``dH/d(eps) = d_eps H|_rho + K . drho/d(eps)`` is the exact chain rule
with the induced half unchanged. Freezing the array instead drops
``dV_H/d(eps)|_rho``, which is first order in every material.

*The density responds even at frozen states.* ``rho`` is stored as values on an
FFT grid that does not move, and the self-consistent density carries a factor
``1/Omega``; a strain changes ``Omega``, so ``drho/d(eps)`` has a piece that is
there with ``dpsi = 0``. It is not a normalisation convention that could be
chosen away -- it is what keeps the electron count fixed as the cell deforms --
and it is obtained here from the same ``jvp``, by making the density a function
of **both** the strain and the states and differentiating along both at once.
The displacement perturbation has no counterpart (moving an atom leaves
``Omega`` alone), which is why this is the first place it appears.

**The eigenvalue response is a diagnostic and has no consumer.**
``deps_n/d(eps)`` is returned because it is the cheapest sharp check on the
perturbation -- its trace over the occupied bands at each k-point is invariant
and can be differenced against a re-converged run -- and *not* because anything
downstream needs it. :mod:`pypresso.response.electrostriction` writes every
operator that would need it with the multiplier **matrix**
``Lambda_mn = <psi_m|H|psi_n>`` instead, which is a strictly better object:
inside a degenerate multiplet -- which silicon's valence bands are at most
k-points -- the diagonal ``deps_n`` is basis-dependent where the matrix is not.

Two things to know before reading the numbers. They carry the ``G = 0``
ambiguity of every absolute deformation potential: the mean electrostatic
potential of a periodic solid is undefined, ``dv_of_drho`` drops the ``G = 0``
Hartree component, and a constant added to ``dH`` shifts every ``deps_n``
together -- so only *differences* between them mean anything. And a single
``deps_n`` is defined only up to the rotation a degenerate multiplet is free in,
so the trace is what a comparison can use.

**Refused rather than approximated**: ultrasoft and PAW (the augmentation charge
``Q_ij(r)`` is a function of the cell, so ``dbecsum`` acquires a strain term of
its own beside the one ``jvp`` gives), and everything
:func:`~pypresso.response.sternheimer.require_a_sternheimer_regime` refuses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from pypresso.basis.interpolate import to_dense
from pypresso.batching import map_k
from pypresso.response.efield import require_a_symmetrisable_response
from pypresso.response.phonon import require_norm_conserving
from pypresso.response.mixing import DEFAULT_RESPONSE_MIXING, ResponseMixer
from pypresso.response.sternheimer import (
    SternheimerSolver,
    require_a_sternheimer_regime,
)
from pypresso.response.velocity import over_kpoints
from pypresso.scf.density import sum_band

__all__ = ["StrainResponse", "strain_response", "strain_tangent",
           "density_of_strained_states"]

#: ``alpha_mix(1)``, as the other two perturbations use it.
#: QE's ``alpha_mix(1)``: the weight the mixer gives the residual. It is no
#: longer the *whole* of the mixing -- :mod:`pypresso.response.mixing` builds an
#: Anderson history on top of it -- which is what makes 0.7 a safe default here
#: rather than a value each system has to be tuned to.
ALPHA_MIX = 0.7

#: Convergence on ``|ddv_scf|^2``.
TR2 = 1.0e-14

MAX_ITERATIONS = 60


def strain_tangent(a: int, b: int) -> jnp.ndarray:
    """The symmetric strain direction ``T^(ab) = (E_ab + E_ba) / 2``.

    Nine of these span the six-dimensional space of symmetric strains, with
    ``T^(ab) = T^(ba)``, and a derivative along ``T^(ab)`` is exactly the
    symmetrised gradient ``(dF/d(eps_ab) + dF/d(eps_ba)) / 2`` -- which is
    ``dF/dx_ab`` for the **tensor** strain ``x``, the convention in which
    ``delta F = sum_ab (dF/dx_ab) x_ab`` runs over all nine pairs with no
    factors of two anywhere. Voigt's engineering shear, where the off-diagonal
    carries a two, is a reporting convention and is applied at the boundary
    (:mod:`pypresso.response.electrostriction`) rather than here.
    """
    e = jnp.zeros((3, 3))
    return 0.5 * (e.at[a, b].add(1.0).at[b, a].add(1.0))


@dataclass
class StrainResponse:
    """The first-order state, density and eigenvalues under the six strains."""

    #: Object array ``(3, 3)``; each entry ``(nspin, nk, nocc, npwx)`` complex.
    #: Symmetric in its two labels -- ``dpsi[a, b] is dpsi[b, a]``.
    dpsi: np.ndarray
    #: ``(3, 3, nspin_mag, n1, n2, n3)`` -- the total ``drho/dx_ab`` on the
    #: dense grid, symmetrised as a rank-2 tensor field, **including** the
    #: frozen-state part the changing volume contributes.
    drho: jnp.ndarray
    #: ``(3, 3, nspin, nk, nocc)`` -- ``<psi_n|dH|psi_n>`` with the induced
    #: potential in it. Defined only up to a common constant; see the module
    #: docstring.
    deigenvalues: np.ndarray
    #: ``(3, 3, nspin_mag, n1, n2, n3)`` -- the converged induced potential.
    dvscf: jnp.ndarray
    history: list = field(default_factory=list)
    average_iterations: float = 0.0
    converged: bool = False


def density_of_strained_states(calculation, states, weights, strain):
    """``rho`` from occupied states in a cell deformed by ``strain``.

    :meth:`~pypresso.response.sternheimer.SternheimerSolver.density_at` with the
    strain as a second argument, and **without the symmetrisation** for that
    method's reason: a response is symmetrised as a tensor, not as a scalar, so
    the caller does it once at the end.

    Differentiable in both arguments, which is the whole point: one ``jvp``
    along ``(tangent_strain, dpsi)`` returns the total density response, the
    volume's own contribution included.
    """
    moved = calculation.at_strain(strain)
    smooth, dense = moved.basis.smooth, moved.basis.dense
    rho = sum_band(
        states, moved.fft_index, smooth.grid, weights,
        moved.system.cell, moved.k_batch,
    )
    return moved.augmented(to_dense(rho, smooth, dense), ())


def strain_response(
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
    verbose: bool = False,
) -> StrainResponse:
    """``solve_linter``'s loop for the six independent homogeneous strains.

    Args:
        calculation: the :class:`~pypresso.scf.driver.Calculation` the states
            belong to, with its own k-set. A ``nosym`` run is accepted only on
            an **unshifted** grid, for
            :func:`~pypresso.response.efield.require_a_symmetrisable_response`'s
            reason.
        wavefunctions: ``(nspin, nk, nbnd, ndim)`` from the converged run.
        eigenvalues: ``(nspin, nk, nbnd)`` or the squeezed ``(nk, nbnd)``.
        density: the converged density the fixed potential is built from.
        becsum: accepted so the signature matches the other two perturbations;
            a nonempty one is refused with the dataset it comes from.
    """
    eigenvalues = jnp.asarray(eigenvalues)
    if eigenvalues.ndim == 2:
        eigenvalues = eigenvalues[None]
    require_a_symmetrisable_response(calculation)
    require_a_sternheimer_regime(calculation)
    require_norm_conserving(calculation)
    _require_one_spin_channel(calculation)

    weights, _ = calculation.occupations(eigenvalues)
    weights = jnp.asarray(weights)
    nocc = int(round(calculation.nelec / 2))
    potential = calculation.potential(density)
    hamiltonians = calculation.hamiltonian(potential.v_scf, None)
    solver = SternheimerSolver(
        calculation, hamiltonians, wavefunctions, eigenvalues, weights,
        nocc, threshold, v_scf=potential.v_scf,
    )
    density = jnp.asarray(density)

    # 1. The bare perturbation and the frozen-state half of ``drho``, both from
    #    ``at_strain`` and both stored: the loop below drives on them at every
    #    iteration and neither changes.
    bare = _bare_strains(calculation, solver, density)
    frozen_drho = _frozen_density_response(calculation, solver, weights)

    # 2. ``solve_linter``'s loop.
    dpsi, drho, dvscf, history, average_iterations, converged = (
        _self_consistent_response(
            calculation, solver, bare, frozen_drho, density,
            alpha_mix=alpha_mix, tr2=tr2, max_iterations=max_iterations,
            mixing_mode=mixing_mode, verbose=verbose,
        )
    )

    # 3. The eigenvalue response, from the converged perturbation.
    deigenvalues = _eigenvalue_response(solver, bare, dvscf)

    return StrainResponse(
        dpsi=dpsi,
        drho=drho,
        deigenvalues=deigenvalues,
        dvscf=dvscf,
        history=history,
        average_iterations=average_iterations,
        converged=converged,
    )


def _bare_strains(calculation, solver, density) -> np.ndarray:
    """``dH/d(eps)|_rho |psi>`` for the six independent strains.

    The potential is rebuilt from the frozen density **inside** the traced
    function -- see the module docstring on why holding an array of values fixed
    is the wrong "bare" here.
    """
    batch = calculation.k_batch
    psi = solver.psi
    zero = jnp.zeros((3, 3))

    def h_psi(strain):
        moved = calculation.at_strain(strain)
        hamiltonians = moved.hamiltonian(moved.potential(density).v_scf, None)
        return jnp.stack([
            over_kpoints(hamiltonian, psi[spin], batch)
            for spin, hamiltonian in enumerate(hamiltonians)
        ])

    bare = np.empty((3, 3), dtype=object)
    for a in range(3):
        for b in range(a, 3):
            value = jax.jvp(h_psi, (zero,), (strain_tangent(a, b),))[1]
            bare[a, b] = bare[b, a] = value
    return bare


def _frozen_density_response(calculation, solver, weights) -> jnp.ndarray:
    """``drho/d(eps)`` at ``dpsi = 0``: what the changing volume alone does.

    Zero for a displacement and not for a strain, which is the trap the module
    docstring names. Computed here rather than folded into the loop because it
    does not change between iterations.
    """
    zero = jnp.zeros((3, 3))
    psi = solver.psi
    return jnp.stack([
        jnp.stack([
            jax.jvp(
                lambda s: density_of_strained_states(
                    calculation, psi, solver.weights, s
                ),
                (zero,), (strain_tangent(a, b),),
            )[1]
            for b in range(3)
        ])
        for a in range(3)
    ])


def _self_consistent_response(
    calculation, solver, bare, frozen_drho, density,
    alpha_mix, tr2, max_iterations, verbose, mixing_mode=DEFAULT_RESPONSE_MIXING,
):
    """The loop, with the frozen-state density response added at every pass.

    Structurally :func:`~pypresso.response.phonon.self_consistent_response` with
    the rank-2 symmetriser and one extra term: ``drho`` is the states' response
    **plus** ``frozen_drho``, so the induced potential the next iteration sees
    carries the volume's contribution as well.
    """
    grid_shape = jnp.asarray(density).shape
    dvscf = jnp.zeros((3, 3) + grid_shape)
    history, total_iterations, solves = [], 0, 0
    dpsi = np.empty((3, 3), dtype=object)
    symmetrised = jnp.zeros_like(dvscf)
    converged = False
    mixer = ResponseMixer(mixing_mode, beta=alpha_mix)

    for iteration in range(max_iterations):
        response = np.empty((3, 3), dtype=object)
        for a in range(3):
            for b in range(a, 3):
                perturbation = _bare_plus_induced(
                    solver, bare[a, b], dvscf[a, b], iteration > 0
                )
                solution = solver.solve(perturbation)
                dpsi[a, b] = dpsi[b, a] = solution.dpsi
                total_iterations += solution.iterations
                solves += 1
                value = solver.response_density(solution.dpsi)
                response[a, b] = response[b, a] = value

        stacked = jnp.stack([
            jnp.stack([response[a, b] for b in range(3)]) for a in range(3)
        ]) + frozen_drho
        symmetrised = calculation.symmetrize_strain_response(stacked)

        induced = jnp.stack([
            jnp.stack([
                jax.jvp(
                    lambda r: calculation.potential(r).v_scf,
                    (jnp.asarray(density),), (symmetrised[a, b],),
                )[1]
                for b in range(3)
            ])
            for a in range(3)
        ])

        change = float(jnp.sum((induced - dvscf) ** 2))
        history.append(change)
        if verbose:
            print(f"  iter {iteration + 1}: |ddv_scf|^2 = {change:.3e}")
        dvscf = mixer.mix(dvscf, induced)
        if change < tr2:
            converged = True
            break

    return (dpsi, symmetrised, dvscf, history,
            total_iterations / max(solves, 1), converged)


def _bare_plus_induced(solver, bare_component, dv, include_induced: bool):
    """``dH_bare|psi> + dV_scf|psi>`` -- :mod:`pypresso.response.phonon`'s."""
    if not include_induced:
        return lambda psi, ik, spin: bare_component[spin][ik]

    induced = solver.perturbation(dv)

    def perturbation(psi, ik, spin):
        return bare_component[spin][ik] + induced(psi, ik, spin)

    return perturbation


def _eigenvalue_response(solver, bare, dvscf) -> np.ndarray:
    """``deps_n = <psi_n|dH_bare + dV_scf|psi_n>``, the first-order eigenvalues.

    QE computes this nowhere for a strain: ``ph.x`` has no strain perturbation.
    It is needed because the Sternheimer operator carries ``eps_n`` explicitly,
    so the operator itself has a strain derivative
    (:mod:`pypresso.response.electrostriction`, ``db/d(eps)``).

    Defined up to a common constant -- see the module docstring.
    """
    batch = solver.calculation.k_batch
    psi = solver.psi
    out = np.empty((3, 3) + psi.shape[:3])
    for a in range(3):
        for b in range(a, 3):
            induced = solver.perturbation(dvscf[a, b])
            blocks = []
            for spin in range(solver.nspin):
                def one_k(ik, spin=spin, a=a, b=b):
                    states = psi[spin][ik]
                    total = bare[a, b][spin][ik] + induced(states, ik, spin)
                    return jnp.real(
                        jnp.einsum("ng,ng->n", jnp.conj(states), total)
                    )

                blocks.append(
                    map_k(one_k, jnp.arange(psi.shape[1]), batch=batch)
                )
            value = np.asarray(jnp.stack(blocks))
            out[a, b] = out[b, a] = value
    return out


def _require_one_spin_channel(calculation) -> None:
    """``nspin = 1`` only, for :mod:`pypresso.response.phonon`'s reason."""
    if calculation.nspin != 1:
        raise NotImplementedError(
            f"nspin = {calculation.nspin}: the strain response here is the "
            "unpolarized one. Nothing in the construction is spin-specific, "
            "but nothing has been checked against a reference either, so it is "
            "refused rather than run"
        )
