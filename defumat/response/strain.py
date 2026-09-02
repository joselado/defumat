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
:meth:`~defumat.scf.driver.Calculation.at_strain` is already written in exactly
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
downstream needs it. :mod:`defumat.response.electrostriction` writes every
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
:func:`~defumat.response.sternheimer.require_a_sternheimer_regime` refuses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from defumat.basis.interpolate import to_dense
from defumat.batching import map_k
from defumat.response.efield import require_a_symmetrisable_response
from defumat.response.phonon import require_norm_conserving
from defumat.response.mixing import DEFAULT_RESPONSE_MIXING, ResponseMixer
from defumat.response.sternheimer import (
    paw_response,
    SternheimerSolver,
    occupied_counts,
    require_a_sternheimer_regime,
)
from defumat.response.velocity import over_kpoints
from defumat.scf.density import becsum as becsum_of
from defumat.system.symmetry import cartesian_rotations
from defumat.scf.density import sum_band

__all__ = ["StrainResponse", "strain_response", "strain_tangent",
           "density_of_strained_states"]

#: ``alpha_mix(1)``, as the other two perturbations use it.
#: QE's ``alpha_mix(1)``: the weight the mixer gives the residual. It is no
#: longer the *whole* of the mixing -- :mod:`defumat.response.mixing` builds an
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
    (:mod:`defumat.response.electrostriction`) rather than here.
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
    #: The occupied block of the first-order state, ``(3, 3)`` object array, or
    #: ``None`` for a norm-conserving dataset where ``S`` does not deform.
    ort: np.ndarray | None = None
    #: ``<psi_m|dS/d(eps)|psi_n>``, which the multipliers' own tangent is built
    #: from. ``None`` for the same reason.
    overlap_derivatives: np.ndarray | None = None
    #: The ``moved`` half of the frozen-state response -- the mixed state's
    #: change at frozen *states*, without the orthogonality block. A consumer
    #: that hands the mixed state to an energy as a function of the strain
    #: generates this half itself and has to subtract it; one that freezes the
    #: density as an array does not. ``PLAN.md`` P39 records what happens when
    #: the distinction is missed.
    moved_drho: jnp.ndarray | None = None
    #: The same split for ``becsum``: ``(total, moved)`` per strain.
    becsum: np.ndarray | None = None
    moved_becsum: np.ndarray | None = None


def density_of_strained_states(calculation, states, weights, strain):
    """``rho`` from occupied states in a cell deformed by ``strain``.

    :meth:`~defumat.response.sternheimer.SternheimerSolver.density_at` with the
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
    return moved.augmented(to_dense(rho, smooth, dense),
                           mixed_becsum(moved, states, weights))


def mixed_becsum(moved, states, weights) -> tuple:
    """``becsum`` at the strained cell, unsymmetrised -- ``()`` for a NC dataset.

    **The augmentation charge is a function of the cell**, which is what makes a
    strain different from a displacement one level deeper than it looks: a
    displacement moves ``Q_ij(r - tau)`` rigidly, and a strain deforms the
    reciprocal-space table it is tabulated on. Both fall out of the same ``jvp``
    through :meth:`~defumat.scf.driver.Calculation.at_strain`, which is why
    this is a builder and not an array.
    """
    if not moved.is_ultrasoft:
        return ()
    return becsum_of(
        states, moved.projectors.vkb, weights, moved.species_channels,
        moved.k_batch,
    )


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
        calculation: the :class:`~defumat.scf.driver.Calculation` the states
            belong to, with its own k-set. A ``nosym`` run is accepted only on
            an **unshifted** grid, for
            :func:`~defumat.response.efield.require_a_symmetrisable_response`'s
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
    # Before the generic guard, so that the message names the strain response.
    _require_one_spin_channel(calculation)
    require_a_sternheimer_regime(calculation)

    weights, _ = calculation.occupations(eigenvalues)
    weights = jnp.asarray(weights)
    nocc = occupied_counts(calculation)
    potential = calculation.potential(density)
    _, ddd_paw = calculation.onecenter(becsum)
    hamiltonians = calculation.hamiltonian(potential.v_scf, ddd_paw)
    solver = SternheimerSolver(
        calculation, hamiltonians, wavefunctions, eigenvalues, weights,
        nocc, threshold, v_scf=potential.v_scf, becsum=becsum,
    )
    density = jnp.asarray(density)

    # 0. What a deforming ``S`` adds, and all of it is zero for a
    #    norm-conserving dataset (``PLAN.md`` P41).
    derivatives = overlap_derivatives(calculation, solver)
    ort = orthogonality_states(calculation, solver, derivatives)

    # 1. The bare perturbation and the frozen-state half of ``drho``, both from
    #    ``at_strain`` and both stored: the loop below drives on them at every
    #    iteration and neither changes.
    bare = _bare_strains(calculation, solver, density)
    frozen_drho, moved_drho, frozen_becsum, moved_becsum = (
        _frozen_density_response(calculation, solver, weights, ort)
    )

    # 2. ``solve_linter``'s loop.
    dpsi, drho, dvscf, history, average_iterations, converged = (
        _self_consistent_response(
            calculation, solver, bare, frozen_drho, density,
            alpha_mix=alpha_mix, tr2=tr2, max_iterations=max_iterations,
            mixing_mode=mixing_mode, verbose=verbose,
            frozen_becsum=frozen_becsum,
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
        ort=ort,
        overlap_derivatives=derivatives,
        moved_drho=moved_drho,
        moved_becsum=moved_becsum,
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

    eigenvalues = solver.eigenvalues

    def h_psi(strain):
        moved = calculation.at_strain(strain)
        hamiltonians = moved.hamiltonian(
            moved.potential(density).v_scf, solver.ddd_paw
        )
        applied = []
        for spin, hamiltonian in enumerate(hamiltonians):
            values = over_kpoints(hamiltonian, psi[spin], batch)
            if hamiltonian.has_overlap:
                # ``compute_deff`` again (``PLAN.md`` P39): the source term of
                # the Sternheimer equation is ``(dH - eps dS)|psi>`` whenever
                # ``S`` moves with the perturbation, and a strain deforms the
                # augmentation charge exactly as a displacement translates it.
                overlap = over_kpoints(hamiltonian, psi[spin], batch, overlap=True)
                values = values - eigenvalues[spin][..., None] * overlap
            applied.append(values)
        return jnp.stack(applied)

    bare = np.empty((3, 3), dtype=object)
    for a in range(3):
        for b in range(a, 3):
            value = jax.jvp(h_psi, (zero,), (strain_tangent(a, b),))[1]
            bare[a, b] = bare[b, a] = value
    return bare


def overlap_derivatives(calculation, solver) -> np.ndarray | None:
    """``S'_mn = <psi_m|dS/d(eps_ab)|psi_n>`` for the six strains, or ``None``.

    ``PLAN.md`` P39's object in the strain coordinate. ``None`` for a
    norm-conserving dataset, where ``S`` is the identity and does not deform.
    """
    if not calculation.is_ultrasoft:
        return None
    psi = solver.psi
    batch = calculation.k_batch
    zero = jnp.zeros((3, 3))

    def overlap_matrix(strain):
        moved = calculation.at_strain(strain)
        vkb = moved.projectors.vkb
        qq = moved.projectors.qq.astype(psi.dtype)
        blocks = []
        for spin in range(psi.shape[0]):
            def one_k(ik, spin=spin):
                becp = jnp.einsum("gc,ng->nc", vkb[ik].conj(), psi[spin][ik])
                return jnp.einsum("mi,ij,nj->mn", becp.conj(), qq, becp)

            blocks.append(map_k(one_k, jnp.arange(psi.shape[1]), batch=batch))
        return jnp.stack(blocks)

    out = np.empty((3, 3), dtype=object)
    for a in range(3):
        for b in range(a, 3):
            out[a, b] = out[b, a] = jax.jvp(
                overlap_matrix, (zero,), (strain_tangent(a, b),)
            )[1]
    return out


def orthogonality_states(calculation, solver, derivatives) -> np.ndarray | None:
    """``dpsi^ort = -1/2 sum_m psi_m <psi_m|dS/d(eps)|psi_n>``, per strain.

    The occupied block the Sternheimer solve does not produce, because
    ``orthogonalize``'s projector makes its answer orthogonal to the occupied
    manifold while the physical first-order state satisfies
    ``<psi + dpsi|S(eps + deps)|psi + dpsi> = 1`` with ``S`` itself deformed.
    ``PLAN.md`` P39, and the identity that checks it is the same one: the
    first-order constraint residual has to vanish.
    """
    if derivatives is None:
        return None
    psi = solver.psi
    out = np.empty((3, 3), dtype=object)
    for a in range(3):
        for b in range(a, 3):
            out[a, b] = out[b, a] = -0.5 * jnp.einsum(
                "skmg,skmn->skng", psi, derivatives[a, b]
            )
    return out


def _frozen_density_response(calculation, solver, weights, ort=None):
    """``drho/d(eps)`` at frozen *variational* states -- ``drho.f90``'s twin.

    Zero for a displacement and not for a strain, which is the trap the module
    docstring names: the volume changes even when nothing else does. Computed
    here rather than folded into the loop because it does not change between
    iterations.

    **For an ultrasoft or PAW dataset it carries two more things**, and they are
    the same two ``PLAN.md`` P39 adds one coordinate over: the augmentation
    charge deforms with the cell (so ``becsum`` and ``Q_ij`` both move, which
    :func:`density_of_strained_states` now differentiates), and the occupied
    block of the first-order state is not zero. Returns
    ``(total, moved_half, becsum_total, becsum_moved)`` -- the ``moved`` halves
    apart because a consumer that hands the mixed state to the energy as a
    *function* of the strain generates them itself and would count them twice.
    """
    zero = jnp.zeros((3, 3))
    psi = solver.psi
    zero_states = jnp.zeros_like(psi)

    def mixed(strain, states):
        moved = calculation.at_strain(strain)
        parts = mixed_becsum(moved, states, solver.weights)
        smooth, dense = moved.basis.smooth, moved.basis.dense
        rho = sum_band(
            states, moved.fft_index, smooth.grid, solver.weights,
            moved.system.cell, moved.k_batch,
        )
        return moved.augmented(to_dense(rho, smooth, dense), parts), parts

    grids, moved_grids = [], []
    parts_total = np.empty((3, 3), dtype=object)
    parts_moved = np.empty((3, 3), dtype=object)
    for a in range(3):
        row, moved_row = [], []
        for b in range(3):
            tangent = strain_tangent(a, b)
            rho_m, bec_m = jax.jvp(mixed, (zero, psi), (tangent, zero_states))[1]
            if ort is None:
                rho_t, bec_t = rho_m, bec_m
            else:
                rho_o, bec_o = jax.jvp(
                    mixed, (zero, psi), (jnp.zeros((3, 3)), ort[a, b])
                )[1]
                rho_t = rho_m + rho_o
                bec_t = tuple(
                    None if x is None else x + y for x, y in zip(bec_m, bec_o)
                )
            row.append(rho_t)
            moved_row.append(rho_m)
            parts_total[a, b] = bec_t
            parts_moved[a, b] = bec_m
        grids.append(jnp.stack(row))
        moved_grids.append(jnp.stack(moved_row))
    return (jnp.stack(grids), jnp.stack(moved_grids), parts_total, parts_moved)


def _self_consistent_response(
    calculation, solver, bare, frozen_drho, density,
    alpha_mix, tr2, max_iterations, verbose, mixing_mode=DEFAULT_RESPONSE_MIXING,
    frozen_becsum=None,
):
    """The loop, with the frozen-state density response added at every pass.

    Structurally :func:`~defumat.response.phonon.self_consistent_response` with
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

    onecentre = None if solver.ddd_paw is None else jnp.zeros(
        (3, 3) + solver.ddd_paw.shape
    )

    for iteration in range(max_iterations):
        response = np.empty((3, 3), dtype=object)
        becsum_response = np.empty((3, 3), dtype=object)
        for a in range(3):
            for b in range(a, 3):
                perturbation = _bare_plus_induced(
                    solver, bare[a, b], dvscf[a, b], iteration > 0,
                    None if onecentre is None else onecentre[a, b],
                )
                solution = solver.solve(perturbation)
                dpsi[a, b] = dpsi[b, a] = solution.dpsi
                total_iterations += solution.iterations
                solves += 1
                value = solver.response_density(solution.dpsi)
                response[a, b] = response[b, a] = value
                if onecentre is not None:
                    parts = solver.response_becsum(solution.dpsi)
                    if frozen_becsum is not None:
                        parts = tuple(
                            None if x is None else x + y
                            for x, y in zip(parts, frozen_becsum[a, b])
                        )
                    becsum_response[a, b] = becsum_response[b, a] = parts

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

        induced_onecentre = None
        if onecentre is not None:
            # ``PAW_dpotential``, from the ``becsum`` response -- the
            # variational part plus the cell's own, exactly as the phonon loop
            # adds ``becsumort`` (``PLAN.md`` P39).
            symmetrised_becsum = _symmetrize_becsum_strain(
                calculation, becsum_response
            )
            induced_onecentre = jnp.stack([
                jnp.stack([
                    paw_response(calculation, symmetrised_becsum[a, b],
                                 solver.becsum)
                    for b in range(3)
                ])
                for a in range(3)
            ])

        change = float(jnp.sum((induced - dvscf) ** 2))
        history.append(change)
        if verbose:
            print(f"  iter {iteration + 1}: |ddv_scf|^2 = {change:.3e}")
        if onecentre is None:
            dvscf = mixer.mix(dvscf, induced)
        else:
            dvscf, onecentre = mixer.mix(
                [dvscf, onecentre], [induced, induced_onecentre]
            )
        if change < tr2:
            converged = True
            break

    return (dpsi, symmetrised, dvscf, history,
            total_iterations / max(solves, 1), converged)


def _symmetrize_becsum_strain(calculation, per_strain):
    """``PAW_dusymmetrize`` on the six strains -- :meth:`BecsumSymmetry.apply_strain`.

    A no-op when there is no PAW species or no symmetry to average over.
    """
    sample = per_strain[0, 0]
    if calculation._becsum_symmetry is None or not sample:
        return per_strain
    rotations = cartesian_rotations(calculation.system.cell, calculation.symmetries)
    stacked = tuple(
        None if sample[species] is None else jnp.stack([
            jnp.stack([per_strain[a, b][species] for b in range(3)])
            for a in range(3)
        ])
        for species in range(len(sample))
    )
    symmetrised = calculation._becsum_symmetry.apply_strain(stacked, rotations)
    out = np.empty((3, 3), dtype=object)
    for a in range(3):
        for b in range(3):
            out[a, b] = tuple(
                None if values is None else values[a, b] for values in symmetrised
            )
    return out


def _bare_plus_induced(solver, bare_component, dv, include_induced: bool,
                       dddd_paw=None):
    """``dH_bare|psi> + dV_scf|psi>`` -- :mod:`defumat.response.phonon`'s."""
    if not include_induced:
        return lambda psi, ik, spin: bare_component[spin][ik]

    induced = solver.perturbation(dv, dddd_paw)

    def perturbation(psi, ik, spin):
        return bare_component[spin][ik] + induced(psi, ik, spin)

    return perturbation


def _eigenvalue_response(solver, bare, dvscf) -> np.ndarray:
    """``deps_n = <psi_n|dH_bare + dV_scf|psi_n>``, the first-order eigenvalues.

    QE computes this nowhere for a strain: ``ph.x`` has no strain perturbation.
    It is needed because the Sternheimer operator carries ``eps_n`` explicitly,
    so the operator itself has a strain derivative
    (:mod:`defumat.response.electrostriction`, ``db/d(eps)``).

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
    """``nspin = 1`` only, for :func:`defumat.response.phonon._require_one_spin_channel`'s reason.

    The occupied-band count is no longer part of it: it is per channel now
    (:func:`~defumat.response.sternheimer.occupied_counts`) and the ``nocc``
    this module derives is that pair. What stays is the assembly -- the strain
    coordinate's own ``dpsi + ort`` block and its multiplier matrix, whose spin
    axis has never been run -- and there is no reference at all here to run it
    against, since ``ph.x`` has no strain perturbation.
    """
    if calculation.nspin != 1:
        raise NotImplementedError(
            f"nspin = {calculation.nspin}: the strain response here is the "
            "unpolarized one. The Sternheimer solve is spin-polarized (the "
            "occupied-band count is per channel now); the assembly above it is "
            "not, and ph.x has no strain perturbation to generate a reference "
            "from, so it is refused rather than run"
        )
