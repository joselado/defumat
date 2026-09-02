"""Quantum ESPRESSO's BFGS, transcribed.

``Modules/bfgs_module.f90``: a quasi-Newton minimiser with a trust radius and a
Wolfe-condition line search, working in *crystal* coordinates with the cell
metric. Sbraccia's implementation, following Fletcher and Billeter et al.; the
references are in the Fortran header.

**Why crystal coordinates and a metric.** The step is ``-H grad`` and its length
has to mean something in bohr, so lengths are measured with ``g_ij = a_i . a_j``
rather than as a Euclidean norm of fractional displacements. QE goes further and
defines the size of a whole configuration change as the *largest single atom's*
displacement (``scnorm``), which is what makes ``trust_radius_max = 0.8`` mean
"no atom moves more than 0.8 bohr" whatever the cell holds.

**Why this is host-side NumPy.** Every branch is a decision about values -- was
the step accepted, is the curvature positive, has the trust radius hit its floor
-- over a few dozen iterations of a few hundred numbers. That is setup-world
code (rule R2), and putting it under ``jit`` would buy nothing and cost every
branch. The gradient it consumes is where JAX belongs
(:mod:`defumat.forces`).

**The cell is nine more coordinates** (P29). QE appends ten entries to the same
vectors -- the nine of ``h`` and the FCP charge -- and ``lmovecell`` is the whole
of the difference: the gradient of the extra nine is ``cell_force``'s
``dH/dh``, their metric block is ``0.04 omega g^-1`` where an atom's is ``g``,
and ``scnorm`` measures them with the same trust radius. So this optimizer takes
``nat + 3`` blocks of three rather than ``nat``, the last three being the rows of
``h``, and everything between is unchanged arithmetic.

**The metric is rebuilt every step and the Hessian is not.** QE allocates
``metric``, ``inv_metric`` and ``hinv_block`` inside ``bfgs()`` from the ``h``
it was passed, on every ionic step, while ``inv_hess`` is read back from file
untouched. That asymmetry is the point: the metric says what a length *is* in
the current cell and has to follow it, whereas the accumulated curvature is the
history being carried. Computing the metric once in the constructor is exact at
fixed cell and silently wrong at variable cell, which is why
:meth:`_rebuild_metric` runs at the top of every :meth:`step`.

**What is left out.** ``bfgs_ndim > 1`` (the GDIIS extrapolation) and the FCP
block. With no FCP its gradient is identically zero and its metric block is
diagonal, so ``n = 3 (nat + 3)`` here is the same arithmetic as QE's
``n = 3 nat + 10``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from defumat.relax.cell import cell_force
from defumat.units import RY_TO_KBAR

__all__ = ["BFGS", "BFGSSettings"]

#: ``Modules/read_namelists.f90``, ``ions_defaults``.
TRUST_RADIUS_MAX = 0.8  # bohr
TRUST_RADIUS_MIN = 1.0e-4  # bohr
TRUST_RADIUS_INI = 0.5  # bohr
W_1 = 0.01
W_2 = 0.50

_EPS8 = 1.0e-8
_EPS16 = 1.0e-16

#: The cell block's metric is ``0.04 * omega * g^-1`` where an atom's is ``g``
#: (``bfgs_module.f90``, the ``FORALL(k=nat:nat+2)`` lines). The factor has no
#: derivation in the Fortran and is not a unit conversion: it is what makes a
#: change of the cell of "the same size" as an atom moving by the trust radius
#: come out the same length under ``scnorm``, so that one ``trust_radius_max``
#: can govern both. Carried over verbatim, because a different value is a
#: different optimizer and would not reproduce QE's trajectory.
CELL_METRIC_SCALE = 0.04


@dataclass
class BFGSSettings:
    """The ``&IONS`` variables this optimizer reads, with QE's defaults."""

    trust_radius_max: float = TRUST_RADIUS_MAX
    trust_radius_min: float = TRUST_RADIUS_MIN
    trust_radius_ini: float = TRUST_RADIUS_INI
    w_1: float = W_1
    w_2: float = W_2
    #: What the inverse Hessian is reset to, as a multiple of the inverse
    #: metric. QE has no such variable because for atoms it does not need one:
    #: the inverse metric means "a curvature of 1 Ry/bohr^2", and Rydberg atomic
    #: units are chosen so that a chemical bond is of that order, which is why
    #: the very first ionic step is a sensible length with no Hessian at all.
    #: A coordinate that is *not* a position has no such coincidence to draw on
    #: -- a spin spiral's ``q`` is in reciprocal units and its energy scale is
    #: milli-Rydberg (:mod:`defumat.workflows.spiral`) -- so the caller has to
    #: say what one unit of curvature means there. ``1.0`` is QE's, exactly.
    hessian_scale: float = 1.0


@dataclass
class BFGS:
    """One relaxation's worth of BFGS state.

    Args:
        at: ``(3, 3)`` lattice vectors as **rows**, in bohr.
        energy_thr: ``etot_conv_thr``, in Ry.
        grad_thr: ``forc_conv_thr``, in Ry/bohr.

    Call :meth:`step` once per ionic step with the energy and the *cartesian
    force*; it returns the cartesian positions to try next and whether the
    relaxation has converged.
    """

    at: np.ndarray
    energy_thr: float = 1.0e-4
    grad_thr: float = 1.0e-3
    settings: BFGSSettings = field(default_factory=BFGSSettings)
    #: ``lmovecell``: whether the cell is nine more coordinates. When it is,
    #: :meth:`step` needs a stress as well as a force, and the relaxed cell is
    #: read back from :attr:`at`, which this object updates in place -- QE
    #: passes ``h`` in and out of ``bfgs()`` for the same reason.
    variable_cell: bool = False
    #: ``press``, in **Ry/bohr^3**. The quantity minimised is then the enthalpy
    #: ``E + P Omega`` and not the energy.
    pressure: float = 0.0
    #: ``epsp``/``press_conv_thr``, in **Ry/bohr^3**: how close the stress has
    #: to come to the target pressure. Only read when :attr:`variable_cell`.
    cell_thr: float = 0.5 / RY_TO_KBAR
    #: ``iforceh``, the ``(3, 3)`` mask ``cell_dofree`` builds
    #: (:mod:`defumat.relax.cell`). ``None`` is ``cell_dofree = 'all'``.
    cell_mask: np.ndarray | None = None

    def __post_init__(self):
        self.at = np.asarray(self.at, dtype=float)
        self.cell_mask = (
            np.ones((3, 3)) if self.cell_mask is None
            else np.asarray(self.cell_mask, dtype=float)
        )
        self.blocks = 0
        self._rebuild_metric(0)
        self.inverse_hessian = None
        #: ``cell_error``: ``max |P I - sigma|`` over the free components, in
        #: Ry/bohr^3. Infinite until the first variable-cell step.
        self.cell_error = np.inf

        self.positions_previous = None
        self.gradient_previous = None
        self.energy_previous = None
        self.direction = None
        self.direction_previous = None
        self.step_length = 0.0
        self.step_length_previous = 0.0
        self.trust_radius = 0.0
        self.trust_radius_previous = 0.0

        #: QE's ``scf_iter``: every call, accepted or not.
        self.iterations = 0
        #: QE's ``bfgs_iter``: only the calls that were not line-search retries.
        self.accepted_steps = 0
        #: QE's ``tr_min_hit``: 1 when the history was reset because the trust
        #: radius hit its floor, 2 when that happened twice running -- which is
        #: QE's signal that the line search is going nowhere and the run must
        #: stop rather than converge.
        self.trust_radius_floor_hits = 0

        self.energy_error = np.inf
        self.gradient_error = np.inf
        self.step_accepted = False
        self.failed = False

    def _rebuild_metric(self, nat: int) -> None:
        """The metric of the *current* cell, and the block layout it implies.

        ``bfgs_module.f90`` builds ``metric``, ``inv_metric`` and ``hinv_block``
        from the ``h`` it was handed, every call. An atom's block is
        ``g_ij = a_i . a_j``, because its coordinates are crystal ones; a cell
        row's is ``0.04 omega g^-1``, because its coordinates are lengths in
        the *reciprocal* index (:data:`CELL_METRIC_SCALE`).
        """
        self.h = self.at.T  # QE's ``h``: lattice vectors as *columns*
        self.metric = self.at @ self.at.T  # g_ij = a_i . a_j
        self.inverse_metric = np.linalg.inv(self.metric)
        self.omega = abs(float(np.linalg.det(self.at)))

        self.blocks = nat + 3 if self.variable_cell else nat
        metric = np.repeat(self.metric[None], self.blocks, axis=0)
        inverse = np.repeat(self.inverse_metric[None], self.blocks, axis=0)
        if self.variable_cell:
            scale = CELL_METRIC_SCALE * self.omega
            metric[nat:] = self.inverse_metric * scale
            inverse[nat:] = self.metric / scale
        self.metric_blocks = metric
        self.inverse_metric_blocks = inverse

    # ------------------------------------------------------------------ norms
    def to_crystal(self, cartesian: np.ndarray) -> np.ndarray:
        return np.asarray(cartesian) @ np.linalg.inv(self.at)

    def to_cartesian(self, crystal: np.ndarray) -> np.ndarray:
        return np.asarray(crystal) @ self.at

    def gradient_from_force(self, force: np.ndarray) -> np.ndarray:
        """``dE/d(crystal position) = -h^T F``, with ``h`` the cell in bohr."""
        return -np.asarray(force) @ self.at.T

    def _norm(self, vector: np.ndarray) -> float:
        """``scnorm``: the largest cartesian displacement of any single atom.

        Not the Euclidean norm of the configuration -- QE's choice, and the
        reason a trust radius is a distance in bohr rather than a number that
        grows with the number of atoms.
        """
        lengths = np.einsum("ai,aij,aj->a", vector, self.metric_blocks, vector)
        return float(np.sqrt(np.maximum(lengths, 0.0)).max())

    def _newton_step(self, gradient: np.ndarray) -> np.ndarray:
        """``-H grad``, with the frozen cell components masked back out.

        The mask has to be applied *after* every product with the inverse
        Hessian and not only once to the gradient: ``inv_hess`` is not block
        diagonal after the first update, so it mixes a free component back into
        a frozen one and the constraint would leak. QE re-masks at each of its
        four ``step(:) = -(inv_hess .times. grad)`` sites for this reason.
        """
        step = -(self.inverse_hessian @ gradient.reshape(-1)).reshape(gradient.shape)
        return self._mask(step)

    def _mask(self, vector: np.ndarray) -> np.ndarray:
        """``iforceh`` on the cell rows; the atoms are masked by ``if_pos``
        upstream, where QE applies it (to the force, not to the step)."""
        if not self.variable_cell:
            return vector
        vector = np.array(vector, dtype=float)
        vector[-3:] *= self.cell_mask
        return vector

    def _reset(self) -> None:
        """``reset_bfgs``: forget the history; the guess is the inverse metric."""
        size = self.inverse_hessian.shape[0]
        block = np.zeros((size, size))
        for index, inverse in enumerate(self.inverse_metric_blocks):
            start = 3 * index
            block[start : start + 3, start : start + 3] = (
                self.settings.hessian_scale * inverse
            )
        self.inverse_hessian = block

    # ------------------------------------------------------------- the update
    def _update_inverse_hessian(self, positions, gradient) -> None:
        """The BFGS update, with the damping QE calls the "curvature trap".

        ``s . y < 0`` cannot happen for a quadratic with a positive definite
        Hessian and happens routinely in a real relaxation, where bonds form and
        break. The damped update (Nocedal and Wright section 18.2) replaces
        ``y`` by a combination of ``y`` and ``B s`` that restores the curvature
        condition, instead of letting the update produce an indefinite matrix
        and a step that goes uphill.
        """
        s = (positions - self.positions_previous).reshape(-1)
        y = (gradient - self.gradient_previous).reshape(-1)
        sdoty = float(s @ y)

        if abs(sdoty) < _EPS16:
            self._reset()
            return

        bs = np.linalg.solve(self.inverse_hessian, s)  # B s, i.e. solve H x = s
        sbs = float(s @ bs)
        if sdoty < 0.20 * sbs:
            theta = 0.8 * sbs / (sbs - sdoty)
            y = theta * y + (1.0 - theta) * bs
            sdoty = float(s @ y)

        hy = self.inverse_hessian @ y
        yh = y @ self.inverse_hessian
        self.inverse_hessian = self.inverse_hessian + (
            (1.0 + float(y @ hy) / sdoty) * np.outer(s, s)
            - (np.outer(s, yh) + np.outer(hy, s))
        ) / sdoty

    # --------------------------------------------------------- Wolfe and trust
    def _energy_wolfe(self, energy: float) -> bool:
        """The first Wolfe condition: the energy fell by enough to accept."""
        slope = float(
            self.gradient_previous.reshape(-1) @ self.direction_previous.reshape(-1)
        )
        return (energy - self.energy_previous) < (
            self.settings.w_1 * slope * self.trust_radius_previous
        )

    def _gradient_wolfe(self, gradient: np.ndarray) -> bool:
        """The second: the slope along the search direction flattened enough."""
        direction = self.direction_previous.reshape(-1)
        slope = float(self.gradient_previous.reshape(-1) @ direction)
        return abs(float(gradient.reshape(-1) @ direction)) < -self.settings.w_2 * slope

    def _compute_trust_radius(self, energy, gradient, wolfe: bool) -> None:
        """``compute_trust_radius``: lengthen the leash when a step goes well."""
        settings = self.settings
        grew = self._energy_wolfe(energy) and (
            self.step_length_previous > self.trust_radius_previous + _EPS8
        )
        factor = (1.5 if grew else 1.1) * (2.0 if wolfe else 1.0)

        self.trust_radius = min(
            settings.trust_radius_max,
            factor * self.trust_radius_previous,
            self.step_length,
        )
        if self.trust_radius >= settings.trust_radius_min:
            self.trust_radius_floor_hits = 0
            return

        self.trust_radius_floor_hits = 2 if self.trust_radius_floor_hits == 1 else 1
        self._reset()
        direction = self._newton_step(gradient)
        self.step_length = self._norm(direction)
        self.direction = direction / self.step_length
        self.trust_radius = min(settings.trust_radius_min, self.step_length)

    # ------------------------------------------------------------------ driver
    def step(self, positions, energy: float, force, stress=None):
        """One ionic step.

        Args:
            positions: ``(nat, 3)`` cartesian, bohr.
            energy: the total energy there, in Ry. With :attr:`variable_cell`
                this is still the *energy*: the enthalpy ``E + P Omega`` is
                formed here, as ``move_ions.f90`` forms it just before calling
                ``bfgs``, so that a caller cannot pass one where the other is
                meant.
            force: ``(nat, 3)`` cartesian force in Ry/bohr, already masked by
                ``if_pos`` -- QE freezes a coordinate by zeroing its force and
                nothing else, so that is where the constraint has to be applied.
            stress: ``(3, 3)`` cartesian stress in Ry/bohr^3, required when
                :attr:`variable_cell` and ignored otherwise.

        Returns ``(next positions, converged)``, the positions cartesian **in
        the new cell**; with :attr:`variable_cell` the new cell is left on
        :attr:`at`, which is QE's ``h`` being passed in and out. When it has
        converged both come back unchanged.
        """
        positions = np.asarray(positions, dtype=float)
        self._rebuild_metric(positions.shape[0])
        crystal = self.to_crystal(positions)
        gradient = self.gradient_from_force(force)
        energy = float(energy)

        if self.variable_cell:
            if stress is None:
                raise ValueError(
                    "a variable-cell BFGS step needs the stress: the cell's "
                    "nine coordinates have no gradient without it"
                )
            energy += self.pressure * self.omega  # the enthalpy
            cell_gradient = self._mask(
                np.vstack([np.zeros_like(crystal), cell_force(
                    np.asarray(stress, dtype=float), self.h, self.omega,
                    self.pressure,
                )])
            )[-3:]
            crystal = np.vstack([crystal, self.h])
            gradient = np.vstack([gradient, cell_gradient])

        if self.inverse_hessian is None:
            self.inverse_hessian = np.zeros((crystal.size, crystal.size))
            self._reset()

        positions = crystal
        self.iterations += 1
        self.energy_error = self._energy_error(gradient, energy)
        self.gradient_error = float(np.abs(force).max())

        converged = (
            self.energy_error < self.energy_thr and self.gradient_error < self.grad_thr
        )
        if self.variable_cell:
            # ``cell_error``: the masked cell gradient carried back onto a
            # stress, which is ``max |P I - sigma|`` when nothing is frozen.
            self.cell_error = float(
                np.abs(gradient[-3:] @ self.h.T).max() / self.omega
            )
            converged = converged and self.cell_error < self.cell_thr
        if not converged and self.trust_radius_floor_hits > 1:
            # The line search has reset its history twice running and is not
            # going anywhere. QE stops here rather than reporting convergence.
            self.failed = True
            converged = True
        if converged:
            return self.to_cartesian(self._split(positions)), True

        if self.iterations > 1 and not self._energy_wolfe(energy):
            positions, energy, gradient = self._reject(positions, energy, gradient)
        elif not self._accept(positions, energy, gradient):
            # A Newton step of zero length: the gradient vanished in every
            # direction the Hessian can see. QE calls errore here ("NR
            # step-length unreasonably short"); there is nowhere to move, so
            # this is a stationary point whether or not the energy criterion
            # has caught up, and stopping is the honest answer.
            return self.to_cartesian(self._split(positions)), True

        # Saved *before* the positions move, as QE's write_bfgs_file is called.
        self.positions_previous = positions
        self.gradient_previous = gradient
        self.energy_previous = energy
        self.direction_previous = self.direction
        self.trust_radius_previous = self.trust_radius

        moved = self._split(positions + self.trust_radius * self.direction)
        # ``move_ions.f90`` sets ``at = h/alat`` *before* ``cryst_to_cart``, so
        # the crystal coordinates that come out of the optimizer are placed in
        # the **new** cell. Using the old one here is a silent error of the size
        # of the cell step, and it is invisible at zero pressure on a cell that
        # barely moves.
        return self.to_cartesian(moved), False

    def _split(self, vector: np.ndarray) -> np.ndarray:
        """Undo :meth:`step`'s packing, leaving the new cell on :attr:`at`."""
        if not self.variable_cell:
            return vector
        self.at = np.array(vector[-3:], dtype=float).T  # rows of h -> rows of at
        self._rebuild_metric(vector.shape[0] - 3)
        return vector[:-3]

    def _energy_error(self, gradient, energy: float) -> float:
        """``|E - E_prev|``, or on the first step the predicted reduction.

        There is no previous energy to difference against at the start, so QE
        uses what the first step could gain instead (Nocedal and Wright 6.2):
        the Hessian is still the metric there, so this costs nothing to
        evaluate.
        """
        if self.iterations > 1:
            return abs(self.energy_previous - energy)
        trial = -self._newton_step(gradient)  # inv_metric . grad, masked
        return abs(
            float(gradient.reshape(-1) @ trial.reshape(-1))
            + 0.5 * float(
                np.einsum("ai,aij,aj->", trial, self.metric_blocks, trial)
            )
        )

    def _reject(self, positions, energy, gradient):
        """The energy went up: interpolate a shorter step along the same line.

        ``E(s) = a s^2 + b s + c`` through ``E(0)``, ``dE(0)`` and ``E(s')``
        gives the minimum at ``-0.5 dE(0) s'^2 / (E(s') - E(0) - dE(0) s')``.
        The rejected point is discarded entirely -- positions, energy and
        gradient all revert to the last accepted step -- so the line search
        walks back along one direction rather than wandering.
        """
        self.step_accepted = False
        self.direction = self.direction_previous

        slope = float(
            self.gradient_previous.reshape(-1) @ self.direction.reshape(-1)
        ) * self.trust_radius_previous
        if slope > 0.0:
            raise RuntimeError(
                "bfgs: the search direction points uphill, which cannot happen "
                "unless the force and the energy disagree"
            )
        denominator = energy - self.energy_previous - slope
        self.trust_radius = -0.5 * slope * self.trust_radius_previous / denominator

        positions = self.positions_previous
        energy = self.energy_previous
        gradient = self.gradient_previous

        if self.trust_radius < self.settings.trust_radius_min:
            self.trust_radius_floor_hits = (
                2 if self.trust_radius_floor_hits == 1 else 1
            )
            self._reset()
            direction = self._newton_step(gradient)
            self.step_length = self._norm(direction)
            self.direction = direction / self.step_length
            self.trust_radius = min(self.settings.trust_radius_ini, self.step_length)
        else:
            self.trust_radius_floor_hits = 0
        return positions, energy, gradient

    def _accept(self, positions, energy, gradient) -> bool:
        """The step was good: update the Hessian and pick a new direction.

        Returns ``False`` when the Newton step has no length left, which means
        the gradient has vanished.
        """
        self.accepted_steps += 1
        wolfe = False
        if self.accepted_steps == 1:
            self.step_accepted = False
        else:
            self.step_accepted = True
            self.step_length_previous = self.step_length
            wolfe = self._energy_wolfe(energy) and self._gradient_wolfe(gradient)
            self._update_inverse_hessian(positions, gradient)

        direction = self._newton_step(gradient)
        if float(gradient.reshape(-1) @ direction.reshape(-1)) > 0.0:
            # An uphill Newton step means the accumulated Hessian is no longer
            # positive definite; QE throws it away rather than stepping.
            self._reset()
            direction = self._newton_step(gradient)

        self.step_length = self._norm(direction)
        if self.step_length < _EPS16:
            return False
        self.direction = direction / self.step_length

        if self.accepted_steps == 1:
            self.trust_radius = min(self.settings.trust_radius_ini, self.step_length)
            self.trust_radius_floor_hits = 0
        else:
            self._compute_trust_radius(energy, gradient, wolfe)
        return True
