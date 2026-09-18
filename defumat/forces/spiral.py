"""``dE/dq``: the gradient of the energy with respect to a spin spiral's wavevector.

A spin spiral's ground state depends on ``q`` exactly as it depends on where the
atoms are (P19, :mod:`defumat.system.spiral`): ``q`` is a coordinate of the
calculation, ``E(q)`` is a surface over it, and the pitch a magnet actually
adopts is that surface's minimum. Finding it by scanning
(:func:`~defumat.workflows.spiral.run_spiral_scan`) costs one SCF per grid
point of a three-dimensional space; finding it by following a gradient costs one
SCF per *step*, and the gradient is what this module produces.

**It is the same construction as the force, term for term.** The energy is
written down as a function of ``q`` at *frozen* wavefunctions, occupations and
eigenvalues, and the gradient is ``jax.grad`` of it -- no expression is derived
for any contribution, and there is no Fortran counterpart to transcribe, because
``pw.x`` has no spin spiral at all. What makes it correct rather than merely
convenient is the same stationarity argument :mod:`defumat.forces.energy`
makes for the positions, and it needs one extra step here:

**The frozen quantity is the periodic part of the spinor, and that is the right
one.** The generalized Bloch theorem writes the state as

    Psi^q_k(r) = ( U_up(r) e^{i(k + q/2).r},  U_dn(r) e^{i(k - q/2).r} )

with ``U_up``, ``U_dn`` lattice periodic. The stored coefficients *are* those
periodic parts -- the plane-wave sphere carries the ``e^{i(k +- q/2).r}``
factor -- so holding the coefficients fixed while ``q`` moves holds ``U`` fixed
and lets the spiral turn, which is precisely the variational parameter the SCF
minimised over. The orthonormality constraint is where the two dataset kinds part, and this
paragraph described only the first until 2026-09-18. On a **norm-conserving**
dataset ``S`` is the identity, ``<U|U> = 1`` has no ``q`` in it, and unlike the
ultrasoft force there is no Pulay term to carry, so the total derivative is the
partial one at frozen state. On an **ultrasoft or PAW** one, which P96 made
work rather than refuse, the constraint is ``<U|S|U> = 1`` and therefore *does*
move with ``q``, so the Pulay term is back and is carried -- together with the
displaced table ``Q_ij(G - q)`` and PAW's one-centre energy, which is what
``compute_spiral_gradient`` below means by rebuilding the sphere at a traced
``q`` rather than differentiating at a frozen one.

**What actually depends on ``q``, and what visibly does not.** In the rotated
frame the density is lattice periodic and is built from the same coefficients on
the same FFT box, so at frozen state it *does not move with* ``q`` at all --
neither does the Hartree energy, the exchange-correlation energy, the local
pseudopotential term or the Ewald sum. The whole gradient comes from the two
places the shifted spheres appear: ``|k +- q/2 + G|^2`` and
``vkb(k +- q/2)``. The energy below is nonetheless written out in full, for two
reasons: evaluated at the converged geometry it must reproduce the SCF total
energy to round-off, which is the only check there is on the rest of it; and
the augmentation charge between the two components is now threaded through, so
on an augmented dataset the density *does* depend on ``q`` -- which is why that
combination is refused here rather than quietly differentiated at a frozen
table.

**The plane-wave sphere is held fixed while differentiating, and that loses
nothing.** Which plane waves satisfy ``|k +- q/2 + G|^2 <= ecutwfc`` is a
host-side decision and cannot be traced -- but it is also *piecewise constant*
in ``q``, so on each piece the frozen-sphere derivative is the exact derivative.
What it does not see is the jump at the isolated wavevectors where a plane wave
crosses the cutoff, and that jump is the Pulay-like error of a finite basis: the
crossing coefficient is by construction the one at the cutoff, so it is of the
size of the basis-set incompleteness and it shrinks with ``ecutwfc``. It is the
floor on how tightly ``dE/dq`` can be driven to zero, and
:func:`~defumat.workflows.spiral.relax_spiral_q` is what has to know that.

**What it costs, and the one dial that bounds it.** The backward pass carries
``vkb(k +- q/2)`` -- both shifted spheres, every projector channel -- and the
states beside it, for every k-point it has in flight. On a two-atom cell that is
nothing; on a three-atom monolayer with 81 k-points, 64 spinor bands and 26315
plane waves per component it is **133 GiB**, which is more than an H200 has and
is where a first attempt at NiI2's ``E(q)`` died after its SCF had converged.
``k_batch`` bounds it: the sum over k is regrouped into chunks, each one a
separate ``value_and_grad`` whose tape is discarded before the next begins, so
the peak falls with the chunk size and the answer does not change at all (the
two routes agree to 1e-16 on a spinor silicon,
``tests/regression/test_spiral_relaxation.py``). It is a **Python loop and not a
mapped one** on purpose: reverse mode through ``lax.map`` stacks every chunk's
residuals for the backward pass and would hold the same peak. The chunked route
is the default wherever the calculation carries a chunk size, which is QE's
end of :mod:`defumat.batching`'s dial on a CPU; ``k_batch = None`` asks for the
single pass the whole thing used to be.

**A magnetic field or a constrained moment is refused rather than corrected.**
The field's own energy is deliberately outside the reported total
(:mod:`defumat.scf.fields`), so the converged state minimises total *plus*
field while the number being differentiated is the total alone -- the
stationarity the whole method rests on does not hold, and the missing term is
invisible.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from defumat.basis.planewaves import PlaneWaveBasis
from defumat.batching import resolve_k_batch
from defumat.forces.energy import FrozenState, state_from_result
from defumat.pseudo.projectors import build_projector_core
from defumat.scf.potential import total_charge
from defumat.system.spiral import spiral_kcart

__all__ = ["SpiralGradient", "spiral_energy", "q_dependent_energy",
           "q_independent_energy", "compute_spiral_gradient"]


@dataclass
class SpiralGradient:
    """``dE/dq`` at one spiral wavevector, and the step direction it implies.

    Two coordinate systems, because the two are used for different things.
    :attr:`gradient` is with respect to ``q`` in **lattice** coordinates -- the
    units ``spiral_q`` is given in, so it says how the energy responds to
    turning the spiral by a fraction of a reciprocal lattice vector.
    :attr:`gradient_cartesian` is with respect to ``q`` in units of
    ``2 pi / alat``, which is the metric an optimizer has to measure a step in:
    a fractional coordinate means a different distance along each axis of a cell
    that is not cubic.
    """

    #: The wavevector it was evaluated at, in lattice coordinates.
    wavevector: np.ndarray
    #: ``(3,)`` ``dE/dq`` with ``q`` in lattice coordinates, Ry.
    gradient: np.ndarray
    #: ``(3,)`` ``dE/dq`` with ``q`` in units of ``2 pi / alat``, Ry.
    gradient_cartesian: np.ndarray
    #: The total energy the gradient was taken at, in Ry -- recomputed from the
    #: frozen state rather than copied from the SCF, so that comparing the two
    #: is the identity check on the functional being differentiated.
    total_energy: float

    @property
    def force(self) -> np.ndarray:
        """``-dE/dq`` in units of ``2 pi / alat``: the direction ``q`` moves in."""
        return -self.gradient_cartesian

    @property
    def max_gradient(self) -> float:
        """``max |dE/dq|`` in Ry per ``2 pi / alat`` -- what convergence tests."""
        return float(np.abs(self.gradient_cartesian).max())


def spiral_energy(calculation, q_crystal, state: FrozenState):
    """The total energy at spiral wavevector ``q_crystal``, state frozen.

    ``calculation`` supplies everything ``q`` does not decide; it is moved with
    :meth:`~defumat.scf.driver.Calculation.at_spiral_q` at a *frozen basis*, so
    the result is a differentiable function of ``q``. A scalar in Ry.
    """
    _require_a_differentiable_spiral(calculation)

    moved = calculation.at_spiral_q(q_crystal, rebuild_basis=False)
    psi, weights = state.wavefunctions, state.weights

    # The density, and everything built from it. None of it moves with ``q`` at
    # frozen coefficients -- the rotated frame is what makes it lattice periodic
    # -- so ``grad`` finds zero here and never differentiates through an FFT or
    # the exchange-correlation functional. It is written out because the
    # functional has to *be* the total energy before it is differentiated.
    becsum_ = moved.becsum(psi, weights)
    rho = moved.density(psi, weights, becsum_)
    potential = moved.potential(rho)
    # PAW's one-centre terms, which are a function of ``becsum`` and therefore
    # of ``q`` through the same projectors the augmentation charge goes through.
    # ``(0, None)`` when no species is PAW.
    epaw, _ = moved.onecenter(becsum_)
    volume = calculation.system.cell.volume
    local = volume / rho[0].size * jnp.sum(moved.vltot * total_charge(rho))

    kinetic = _kinetic_energy(psi, moved.state_kinetic, weights)
    nonlocal_, overlap = _projector_energies(
        psi, moved.projectors.vkb, moved.dvan_so, moved.qq_so, weights,
        state.eigenvalues, calculation.system.kpoints.nk,
    )
    # The orthonormality constraint, in two pieces. ``<psi|psi> - 1`` carries no
    # ``q`` -- the plane-wave coefficients are frozen and the sphere with them --
    # and is here for the same reason the density is, so that the identity
    # against the SCF total energy is a check on everything else. The
    # augmentation half above it is *not* inert: ``S`` pairs each spinor
    # component with the projectors of **its own** shifted sphere, so
    # ``<psi|S|psi>`` moves with ``q`` and its derivative is the spiral's
    # Pulay term. A norm-conserving dataset has ``S = 1`` and never reaches it.
    norm = jnp.sum(weights * state.eigenvalues * (_norms(psi) - 1.0))

    return (
        kinetic
        + nonlocal_
        + local
        + potential.ehart
        + potential.etxc
        + epaw
        - overlap
        + moved.ewald
        # Carried for the same reason the density is: a spiral does not move an
        # atom, so the dispersion energy has no ``q`` dependence and contributes
        # nothing to the gradient -- but the identity against the SCF total
        # energy is the only check on the terms that do.
        + moved.dispersion
        - norm
        + state.entropy
    )


def q_dependent_energy(calculation, q_crystal, psi, weights, eigenvalues, rows):
    """The terms that carry ``q``, over a *subset* of the k-points.

    ``rows`` indexes the ``2 nk`` axis both shifted spheres live on -- the up
    component's ``nk`` rows first -- so a chunk of ``m`` k-points is the ``m``
    up rows and the ``m`` down rows that go with them. The sub-basis is
    **selected** from the calculation's own spheres rather than rebuilt: which
    plane waves are in each sphere, in which order, is exactly what the frozen
    coefficients are written against, so a rebuilt sphere would be a different
    Miller ordering against the same numbers and silently wrong.

    Everything else in :func:`spiral_energy` -- the density, Hartree,
    exchange-correlation, the local term, Ewald, ``<psi|psi> - 1`` -- is
    ``q``-independent at frozen coefficients and is *not* here, because its
    gradient is zero and the chunking exists to keep the backward pass off it.
    :func:`q_independent_energy` evaluates it once, forward only, so the
    identity against the SCF total energy survives.

    **That split is what makes this route norm-conserving only.** On an
    augmented dataset the density itself carries ``q``, through the displaced
    table and through ``becsum``, and the Hartree energy is *quadratic* in the
    density -- so the sum of per-chunk gradients is no longer the gradient of
    the sum. :func:`compute_spiral_gradient` sends such a run down the single
    pass instead. The augmentation half of the constraint is written here
    nonetheless, and is identically zero on everything that reaches this
    function, because the two routes differing term by term is how a later
    lifting of that restriction would go wrong quietly.
    """
    cell = calculation.system.cell
    smooth = calculation.basis.smooth
    planewaves = calculation.basis.planewaves
    sub = PlaneWaveBasis(
        indices=planewaves.indices[rows],
        mask=planewaves.mask[rows],
        # ``npw`` is host-side bookkeeping -- how many of the padded entries at
        # each row are real -- and nothing on this path reads it; the sphere
        # itself is carried by ``indices`` and ``mask``, which is what makes
        # ``rows`` free to be a traced index and the chunks free to share one
        # compilation.
        npw=(),
        ecutwfc=planewaves.ecutwfc,
    )
    kcart = spiral_kcart(calculation.system.kpoints, q_crystal, cell)[rows]
    kinetic = sub.kinetic(smooth, calculation.basis_kpoints, cell, kcart)
    core = build_projector_core(
        calculation.pseudos, calculation.system.structure, cell, smooth, sub,
        calculation.basis_kpoints, kcart,
    )
    projectors = core.at_positions(
        calculation.system.structure.positions, qq=calculation.projectors.qq
    )

    m = kinetic.shape[0] // 2
    state_kinetic = jnp.concatenate([kinetic[:m], kinetic[m:]], axis=-1)
    nonlocal_, overlap = _projector_energies(
        psi, projectors.vkb, calculation.dvan_so, calculation.qq_so, weights,
        eigenvalues, m,
    )
    return _kinetic_energy(psi, state_kinetic, weights) + nonlocal_ - overlap


def q_independent_energy(calculation, state: FrozenState):
    """Everything :func:`q_dependent_energy` leaves out, at frozen state.

    It is a constant of ``q`` and therefore contributes nothing to ``dE/dq``;
    it is evaluated -- once, forward only, never under ``grad`` -- so that the
    chunked route still reports the *total* energy and
    ``|E_frozen - E_scf|`` remains the check that what was differentiated is
    the energy the SCF converged.
    """
    psi, weights = state.wavefunctions, state.weights
    becsum_ = calculation.becsum(psi, weights)
    rho = calculation.density(psi, weights, becsum_)
    potential = calculation.potential(rho)
    volume = calculation.system.cell.volume
    local = volume / rho[0].size * jnp.sum(calculation.vltot * total_charge(rho))
    norm = jnp.sum(weights * state.eigenvalues * (_norms(psi) - 1.0))
    return (
        local
        + potential.ehart
        + potential.etxc
        + calculation.ewald
        + calculation.dispersion
        - norm
        + state.entropy
    )


def _chunked_energy_and_gradient(calculation, q, state: FrozenState, k_batch: int):
    """``(E, dE/dq)`` accumulated over ``k_batch`` k-points at a time.

    A Python loop of per-chunk ``value_and_grad`` calls, **not** one
    ``value_and_grad`` around a ``lax.map``: reverse mode through a scan stacks
    every chunk's residuals for the backward pass, so the mapped form would
    hold the same peak the single pass does. The loop discards each chunk's
    tape before the next one starts, which is the whole of the saving.

    Every chunk is padded to exactly ``k_batch`` k-points with a repeat of its
    own first one at **zero weight**, so all chunks share one shape and
    therefore one compilation, and the padding contributes nothing to either
    the energy or the gradient (both are linear in ``weights``).
    """
    nk = calculation.system.kpoints.nk
    fn = calculation.__dict__.get("_spiral_gradient_chunk")
    if fn is None:
        fn = jax.jit(jax.value_and_grad(
            lambda q, psi, weights, eigenvalues, rows:
                q_dependent_energy(calculation, q, psi, weights, eigenvalues, rows)
        ))
        calculation._spiral_gradient_chunk = fn

    energy = 0.0
    gradient = jnp.zeros((3,), dtype=float)
    for start in range(0, nk, k_batch):
        ks = np.arange(start, min(start + k_batch, nk))
        pad = k_batch - len(ks)
        padded = np.concatenate([ks, np.full(pad, ks[0], dtype=int)])
        live = np.concatenate([np.ones(len(ks)), np.zeros(pad)])
        psi = state.wavefunctions[:, padded]
        weights = state.weights[:, padded] * live[None, :, None]
        eigenvalues = state.eigenvalues[:, padded]
        rows = jnp.asarray(np.concatenate([padded, nk + padded]))
        value, slope = fn(q, psi, weights, eigenvalues, rows)
        energy += value
        gradient = gradient + slope
    return energy + q_independent_energy(calculation, state), gradient


def compute_spiral_gradient(
    calculation, result_or_state, k_batch: int | None | str = "default"
) -> SpiralGradient:
    """``dE/dq`` for ``calculation`` in its converged state.

    Args:
        calculation: the :class:`~defumat.scf.driver.Calculation` the state
            belongs to -- its ``spiral_q`` is where the gradient is evaluated.
        result_or_state: an :class:`~defumat.scf.driver.SCFResult`, or the
            :class:`~defumat.forces.energy.FrozenState` taken from one.
        k_batch: how many k-points are differentiated at once. ``"default"``
            follows the calculation's own dial, ``None`` asks for the whole
            axis in one pass, and an integer bounds the working set: the
            backward pass carries ``vkb(k +- q/2)`` and the states for every
            k-point it has in flight, which on a cell with many k-points and
            two spinor components is tens of gigabytes and is the one place
            ``dE/dq`` costs more than the SCF it follows.
    """
    _require_a_differentiable_spiral(calculation)
    state = (
        result_or_state
        if isinstance(result_or_state, FrozenState)
        else state_from_result(result_or_state)
    )

    q = jnp.asarray(calculation.system.spiral_q, dtype=float)
    batch = (calculation.k_batch if isinstance(k_batch, str) and k_batch == "default"
             else resolve_k_batch(k_batch))
    if (calculation.is_ultrasoft and batch is not None
            and batch < calculation.system.kpoints.nk):
        # **The chunked route cannot express an augmented spiral**, and the
        # reason is the Hartree energy rather than the augmentation: on this
        # dataset the density carries ``q``, and a quadratic functional of a
        # sum is not the sum of the per-chunk quadratic functionals, so
        # accumulating :func:`q_dependent_energy`'s gradients would be wrong by
        # every cross term. The dial's default on a CPU is the chunked end, so
        # this overrides the dial rather than leaving the dispatch to pick.
        # It is said out loud because the cost is real: the single pass carries
        # every k-point's ``vkb(k +- q/2)`` and the displaced table's radial
        # intermediates on one tape.
        #
        # ``batch < nk`` rather than ``batch is not None``, because a chunk at
        # or above the whole axis is not chunking: the dispatch below sends it
        # to the same single pass, so warning about it would be announcing an
        # override that did not happen.
        warnings.warn(
            "dE/dq on an ultrasoft or PAW spiral is evaluated in a single pass "
            f"over all {calculation.system.kpoints.nk} k-points rather than in "
            f"chunks of {batch}: the density carries q on this dataset and the "
            "Hartree energy is quadratic in it, so a sum of per-chunk "
            "gradients is not the gradient. The peak working set scales with "
            "nk rather than with the chunk size",
            RuntimeWarning,
            stacklevel=2,
        )
        batch = None
    if batch is None or batch >= calculation.system.kpoints.nk:
        energy, gradient = _energy_and_gradient(calculation)(q, state)
    else:
        energy, gradient = _chunked_energy_and_gradient(calculation, q, state, batch)

    cell = calculation.system.cell
    # ``q_cart = q_cryst @ B``, so ``dE/dq_cryst = B dE/dq_cart`` and the
    # cartesian gradient comes back through the same matrix. Doing it this way
    # rather than differentiating a cartesian parameterisation keeps one
    # definition of what ``q`` is -- the lattice coordinates ``spiral_q`` is
    # written in, and the only ones the rest of the code names.
    bg = np.asarray(cell.bg_2pi_alat)
    gradient = np.asarray(gradient, dtype=float)
    cartesian = np.linalg.solve(bg, gradient)

    return SpiralGradient(
        wavevector=np.asarray(calculation.system.spiral_q, dtype=float),
        gradient=gradient,
        gradient_cartesian=cartesian,
        total_energy=float(energy),
    )


def _energy_and_gradient(calculation):
    """``(E, dE/dq)`` in one pass, compiled once per plane-wave sphere.

    ``value_and_grad`` rather than ``grad``: the energy is wanted anyway -- it
    is what checks the functional against the SCF total -- and the forward pass
    is shared.

    The compiled function is cached on the calculation, and unlike the force's
    cache it is **dropped whenever the sphere changes**
    (:meth:`~defumat.scf.driver.Calculation.at_spiral_q`). A moved ``q`` is a
    different set of plane waves and therefore a different ``npwx``, so the
    cache buys nothing across the steps of a relaxation and would be actively
    wrong if it were carried: the closure holds the sphere the gradient was
    compiled with. Each accepted step pays one compilation, which is small
    beside the SCF it goes with.
    """
    cached = calculation.__dict__.get("_spiral_gradient")
    if cached is None:
        cached = jax.jit(jax.value_and_grad(
            lambda q, state: spiral_energy(calculation, q, state)
        ))
        calculation._spiral_gradient = cached
    return cached


def _require_a_differentiable_spiral(calculation) -> None:
    """The two things that make ``dE/dq`` at frozen state not be the answer.

    An ultrasoft or PAW dataset **was** the third and is no longer (P96): the
    displaced table is rebuilt with a traced ``q`` inside
    :meth:`~defumat.scf.driver.Calculation.at_spiral_q`, which is where the
    remaining refusal lives -- a *tabulated* table reads ``|q|`` on the host
    and cannot take a tracer at all.
    """
    if not calculation.spiral:
        raise ValueError(
            "dE/dq needs a spin spiral: set spiral_q, which is the coordinate "
            "being differentiated with respect to"
        )
    if calculation.magnetic_field is not None:
        raise NotImplementedError(
            "dE/dq with a magnetic field or a constrained moment is not "
            "implemented: the field's energy is deliberately outside the "
            "reported total (see defumat.scf.fields), so the converged state "
            "is stationary for a different functional than the one being "
            "differentiated and the missing term would be silent"
        )


@jax.jit
def _kinetic_energy(psi, kinetic, weights):
    """``sum w f <psi| |k +- q/2 + G|^2 |psi>``.

    ``kinetic`` is already in the layout a *spinor* is stored in -- the up
    component's row and the down component's row laid end to end -- so the two
    components meet their own sphere without the sum having to know there are
    two of them.
    """
    return jnp.sum(weights * jnp.einsum("skbg,kg->skb", jnp.abs(psi) ** 2, kinetic))


@jax.jit
def _norms(psi):
    return jnp.sum(jnp.abs(psi) ** 2, axis=-1)


@partial(jax.jit, static_argnames=("nk",))
def _projector_energies(psi, vkb, dvan_so, qq_so, weights, eigenvalues, nk):
    """``<psi|V_NL|psi>`` and ``<psi|S - 1|psi>``, with both shifted spheres.

    ``add_vuspsi_nc``'s quadratic form and ``s_psi_nc``'s beside it: ``D`` is a
    2x2 matrix in spin space (``dvan_so``, which for a scalar-relativistic
    dataset is the ordinary ``dion`` on each diagonal block), and the two spinor
    components are projected onto *different* projectors -- ``vkb(k + q/2)`` and
    ``vkb(k - q/2)`` -- which is where the spiral enters both terms.

    ``dvan_so`` is the **bare** ``D``, for the reason
    :func:`defumat.forces.energy._spinor_projector_energies` gives: the
    self-consistent ``int V_eff Q_ij`` half of ``deeq`` is already inside the
    augmented density, so taking the screened one here would count it twice.

    The second return is the augmentation half of the orthonormality
    constraint, ``sum w f eps <psi|Q|psi>``, and it is zero for a
    norm-conserving dataset, where ``qq_so`` is ``None`` and ``S`` is the
    identity. ``qq_so`` is the **resident** ``Omega Q_ij(G = 0)`` rather than
    the displaced table: ``S`` is diagonal in the spinor index without
    spin-orbit coupling -- which a spiral refuses permanently -- so each
    component pairs projectors at one and the same k-point, and no displacement
    enters. What makes the term move with ``q`` is ``vkb`` alone.

    The indices below are ``s`` spin (one, for a spinor run), ``k`` k-point,
    ``n`` band, ``a``/``b`` spinor component, ``i``/``j`` projector channel.
    """
    if vkb.shape[-1] == 0:
        return jnp.zeros(()), jnp.zeros(())
    npwx = vkb.shape[1]
    components = psi.reshape(psi.shape[:-1] + (2, npwx))
    # The spiral's two blocks of rows -- the up component's ``nk`` first --
    # stacked into a spinor axis, which is what ``SpinorHamiltonian._project``
    # does one k-point at a time.
    pair = jnp.stack([vkb[:nk], vkb[nk:]], axis=1)  # (nk, 2, npwx, nkb)
    becp = jnp.einsum("kagi,sknag->sknai", pair.conj(), components)
    bands = jnp.real(jnp.einsum(
        "sknai,abij,sknbj->skn",
        becp.conj(), dvan_so.astype(becp.dtype), becp, optimize=True,
    ))
    nonlocal_ = jnp.sum(weights * bands)
    if qq_so is None:
        return nonlocal_, jnp.zeros(())
    overlap = jnp.real(jnp.einsum(
        "sknai,abij,sknbj->skn",
        becp.conj(), qq_so.astype(becp.dtype), becp, optimize=True,
    ))
    return nonlocal_, jnp.sum(weights * eigenvalues * overlap)
