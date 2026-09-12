"""External magnetic fields and constrained magnetic moments.

Two features that share all their machinery, and QE's ``add_bfield.f90`` treats
them as one for that reason:

* a **field** put in by hand -- over the whole cell (QE's ``B_field``, Elk's
  ``bfieldc``) or inside one atom's sphere (Elk's ``bfcmt``, which QE has no
  input for) -- whose energy is ``-int B(r) . m(r) dr``;
* a **constraint** on a moment, imposed by a penalty ``lambda (m - m_target)^2``
  whose derivative *is* a field. QE's ``constrained_magnetization`` has four
  forms: the moment on each atom, the direction of the moment on each atom, the
  total moment of the cell, and the direction of that total.

Both are used the same way -- to break a symmetry the SCF would otherwise keep,
or to hold a magnetic configuration that is not the ground state so that its
energy can be measured -- and both are how a spin spiral's energy surface gets
sampled at all.

**The energy is the primitive here and the potential comes from `jax.grad`.**
That is this project's rule (`PLAN.md` §6) and it pays immediately: QE writes
each constraint's potential out by hand -- five expressions, one of them three
lines of quotient rule -- and every one of them is exactly the derivative of the
penalty stated above, so writing the penalty once gives all five, and the
Fortran expressions become a *test* rather than a second implementation
(``tests/unit/test_magnetic_fields.py``).

**What the total energy includes, and it is not obvious.** ``add_bfield`` is
called from inside ``v_of_rho``, so the field is felt by every eigenvalue and
removed again by ``deband``; ``etcon`` is printed and never added to ``etot``.
Elk reaches the same convention from the other side -- its manual says the
muffin-tin field energy "is always removed from the total" and the physical
field's "is also not included", both being meant as infinitesimal symmetry
breakers -- and reports ``engybext`` separately for the case where the field is
finite. So the total energy this code reports excludes the field, exactly as
QE's does, and the field's own energy is carried beside it.

**The region a "local" moment is integrated over** is
:mod:`defumat.scf.locals` -- a sphere of radius ``r_m`` with a linear taper --
and which weight scheme was used is recorded there.

*Units.* Everything here is in QE's: the field is a Rydberg energy conjugate to
the magnetization as this code carries it, so it enters the potential's
magnetization components directly. Elk's ``bfieldc`` is a Hartree field coupling
as ``(g_e/4c) sigma . B``; converting is ``io/``'s business and not this
module's.
"""

from __future__ import annotations

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from defumat.scf.locals import LocalRegions
from defumat.system.cell import Cell

__all__ = [
    "MagneticField",
    "CONSTRAINTS",
    "ATOM_RESOLVED",
    "FSM_UPDATES",
    "DEFAULT_FSM_UPDATE",
    "constraint_targets",
    "magnetization_components",
]

#: The constraint schemes, by the name ``constrained_magnetization`` takes in a
#: pw.x input, with QE's ``i_cons`` beside each. ``'fsm'`` is Elk's alternative
#: (a field updated by feedback rather than a penalty) and is a defumat
#: extension; see :meth:`MagneticField.feedback`.
CONSTRAINTS = {
    "none": 0,
    "atomic": 1,
    "atomic direction": 2,
    "total": 3,
    "total direction": 6,
    "fsm": -1,
    # Not QE's, and it has no ``i_cons`` number because ``input.f90`` has no
    # such scheme. ``atomic direction`` is ``i_cons = 2`` and constrains
    # ``m_z/|m|`` -- the polar angle *alone* -- so it is exactly satisfied by
    # every texture lying in a plane containing ``z``, helix and collinear
    # alike, and cannot hold one. This constrains the full unit vector.
    "atomic texture": None,
}

#: The constraints whose penalty is a sum over *atoms*, so each one needs the
#: per-atom integration spheres to exist. **Read this set rather than spelling
#: the names again**: a caller that built its own tuple left ``atomic texture``
#: out of it, and the omission surfaced as an ``AttributeError`` on ``None``
#: inside the first potential build rather than as a refusal at input
#: (`NONCOLLINEAR.md` item 8).
ATOM_RESOLVED = frozenset({"atomic", "atomic direction", "atomic texture"})

#: Below this moment a direction constraint has nothing to act on, and QE stops
#: rather than dividing (``add_bfield``'s ``1.d-30`` / ``1.D-12``).
VANISHING_MOMENT = 1.0e-12

#: Below this **transverse** moment the polar angle has no gradient to speak of,
#: because the polar angle is genuinely not differentiable on the ``z`` axis --
#: which way it moves depends on which way you leave the axis. QE's
#: ``add_bfield.f90:185-192`` picks the convention: zero the transverse
#: derivative and add a tiny field along ``x`` so the moment can start turning.
#: The number is QE's ``1.D-14``.
VANISHING_TRANSVERSE = 1.0e-14

#: The size of that escape field, QE's literal ``fact1(1) = 1.D-14``. It exists
#: to break a tie rather than to change an answer: a moment seeded from
#: ``starting_magnetization`` with no angles lies **exactly** on the axis, so
#: without it a polar-angle constraint has nothing to push with and the run sits
#: at an unstable stationary point of its own penalty.
AXIS_ESCAPE = 1.0e-14

#: How close the fixed-spin-moment scheme has to get before a run counts as
#: converged, in Bohr magnetons. **This is a convergence criterion in its own
#: right and not a formality**: the field is not part of the density, so the
#: density residual can fall below ``conv_thr`` while the moment is still far
#: from its target and the field is still being driven. A run that stopped there
#: would report an unconstrained answer under a "constrained" heading.
FSM_TOLERANCE = 1.0e-3

#: How much applied field, in Ry, a ``reducebf`` run may still be carrying when
#: it stops before the run says so. ``reducebf`` decays the field *after* the
#: convergence test, so nothing requires it to be small before the loop breaks --
#: and the state that is then reported is the ground state of a functional that
#: includes a Zeeman term whose energy is, by QE's and Elk's shared convention,
#: **not in the reported total**.
#:
#: The number is measured rather than picked. On the hydrogen atom of
#: ``test_a_local_field_breaks_the_symmetry_it_should``, with a 0.1 Ry local
#: field and ``conv_thr = 1e-10``, against the properly seeded field-free run:
#:
#: ===========  ==========  ==============  =====================
#: ``reducebf``  iterations  residual ``|B|``  error in the total
#: ===========  ==========  ==============  =====================
#: 0.5                  15        6.1e-06 Ry             -5e-14 Ry
#: 0.9                  39        1.8e-03 Ry            +2.2e-08 Ry
#: 0.95                 44        1.1e-02 Ry            +7.8e-07 Ry
#: 0.99                  6        9.5e-02 Ry            +4.7e-05 Ry
#: ===========  ==========  ==============  =====================
#:
#: The error is **second order** in the residual, because the state is
#: stationary -- 8.6x the field is 60x the error, and 8.6^2 is 74. That is why
#: the threshold is on the field and not on ``field_energy``, which is first
#: order and is 6e-06 Ry in the row whose total is right to 5e-14. ``1e-4``
#: sits between the two cleanest rows and corresponds to an error around
#: 1e-10 Ry, below any threshold a run is held to.
#:
#: The last row is the case worth seeing: six iterations, 95% of the field still
#: applied, and a total 0.63 meV out.
FADED_FIELD = 1.0e-4

#: The fixed-spin-moment update rules, by name.
#:
#: ``"elk"``
#:     ``B <- B - tau (m - m_fix)`` after *every* SCF iteration, transcribed
#:     from ``bfieldfsm.f90``. Correct, and slow for a reason that is worth
#:     understanding: the field is nudged while the density is still moving, so
#:     the controller reads a moment that has not finished responding to the
#:     last nudge and it rings. On ``fe-fsm.in`` the apparent susceptibility it
#:     sees swings between +2591 and -1252 mu_B/Ry from one iteration to the
#:     next, and the ringing takes 1380 iterations to damp below 1e-3.
#: ``"secant"``
#:     Update only when the inner SCF has converged, and step by the measured
#:     susceptibility rather than by a fixed gain. At converged density ``m(B)``
#:     is smooth -- 2.499, 2.274, 2.036, 1.837 mu_B at B = 0, -0.005, -0.010,
#:     -0.020 Ry on that same case -- so a secant on it is a Newton step, and
#:     the whole run costs a handful of SCF solves instead.
#:
#: **The gain was never the problem.** Elk's ``tau`` of 0.02 against a measured
#: ``1/chi`` of 0.022 is already the right step size; what makes the difference
#: is *when* the step is taken.
FSM_UPDATES: tuple[str, ...] = ("secant", "elk")

#: The default, because it is the same answer for a tenth of the iterations.
DEFAULT_FSM_UPDATE = "secant"

#: A secant step is refused if the two measurements are this close in field --
#: below it the susceptibility is a ratio of two round-off differences.
FSM_MIN_STEP = 1.0e-6

#: ... and it is never longer than this many times the previous step, so a
#: near-singular chi cannot throw the field across the phase diagram.
FSM_TRUST = 4.0



def magnetization_components(rho_r: jnp.ndarray) -> jnp.ndarray:
    """The magnetization out of a density, ``(ncomponent, ...grid)``.

    One component for a collinear density -- ``rho_up - rho_down`` -- and three
    for a noncollinear one. QE's ``add_bfield`` writes the same thing as
    ``npol = nspin - 1`` and indexes ``rho(:, ipol+1)``, which works because both
    representations keep the magnetization after the charge.
    """
    if rho_r.shape[0] == 2:
        return (rho_r[0] - rho_r[1])[None]
    return rho_r[1:]


def constraint_targets(
    constraint: str,
    types,
    starting_magnetization,
    angle1,
    angle2,
    fixed_magnetization,
    ntyp: int,
    noncollinear: bool,
    per_atom=(),
) -> np.ndarray:
    """``mcons``: what each constraint compares the moment against (``input.f90``).

    The shapes differ by scheme, and so does the *meaning*, which is worth
    stating because one of them is surprising:

    * ``atomic`` -- ``(nat, 3)`` (or ``(nat, 1)``): the starting magnetization
      times the direction ``(angle1, angle2)`` points in. **QE compares this
      against the moment in Bohr magnetons**, although it is built from
      ``starting_magnetization``, which is a fraction of the valence charge. The
      benchmark shows the consequence plainly: iron's moment falls from 3.06 to
      1.6 mu_B under ``constrained_magnetization = 'atomic'`` with a target of
      0.5. Transcribed as it is, because reproducing QE is the point;
    * ``atomic direction`` -- ``(nat, 1)``: the cosine of the polar angle;
    * ``total`` -- ``(3,)``: ``fixed_magnetization``, in Bohr magnetons;
    * ``total direction`` -- ``(1,)``: the polar angle in *degrees*, which QE
      converts where it uses it.
    """
    types = np.asarray(types, dtype=int)
    magnitudes = np.zeros(ntyp)
    given = np.asarray(starting_magnetization, dtype=float)
    magnitudes[: len(given)] = given

    # ``STARTING_MOMENTS`` overrides the per-species construction for the two
    # atom-resolved schemes, and for the same reason it overrides ``m_loc``: a
    # texture has one direction per *atom* and QE's input cannot say one. The
    # cell-wide schemes are untouched -- ``fixed_magnetization`` is already a
    # single vector and has nothing per-atom about it.
    if constraint == "atomic texture":
        if not len(per_atom):
            raise ValueError(
                "constrained_magnetization = 'atomic texture' needs a "
                "STARTING_MOMENTS card: it constrains one direction per atom and "
                "starting_magnetization/angle1/angle2 are per species, so there "
                "is nothing per-atom for it to aim at"
            )
        targets = np.asarray(per_atom, dtype=float).reshape(-1, 3)
        modulus = np.linalg.norm(targets, axis=-1, keepdims=True)
        if np.any(modulus <= VANISHING_MOMENT):
            raise ValueError(
                "constrained_magnetization = 'atomic texture' with a zero row in "
                "STARTING_MOMENTS: a zero vector carries no direction. Give every "
                "atom a direction, or use 'atomic' to constrain magnitudes too"
            )
        return targets / modulus

    if len(per_atom) and constraint in ("atomic", "atomic direction"):
        targets = np.asarray(per_atom, dtype=float).reshape(-1, 3)
        if constraint == "atomic":
            return targets if noncollinear else targets[:, 2:3]
        modulus = np.linalg.norm(targets, axis=-1)
        safe = np.where(modulus > VANISHING_MOMENT, modulus, 1.0)
        cosine = np.where(modulus > VANISHING_MOMENT, targets[:, -1] / safe, 0.0)
        return cosine[:, None]

    if constraint == "atomic":
        if not noncollinear:
            return magnitudes[types][:, None]
        theta = np.zeros(ntyp)
        phi = np.zeros(ntyp)
        for target, values in ((theta, angle1), (phi, angle2)):
            values = np.asarray(values, dtype=float)
            target[: len(values)] = np.deg2rad(values)
        directions = np.stack(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
            axis=1,
        )
        return magnitudes[types, None] * directions[types]

    if constraint == "atomic direction":
        theta = np.zeros(ntyp)
        values = np.asarray(angle1, dtype=float)
        theta[: len(values)] = np.deg2rad(values)
        return np.cos(theta)[types][:, None]

    if constraint in ("total", "fsm"):
        fixed = np.asarray(fixed_magnetization, dtype=float)
        # A collinear run's magnetization is the z component by construction, so
        # that is the component of ``fixed_magnetization`` it is compared with --
        # ``fixed_magnetization(3)``, not the first entry. (QE never reaches this
        # branch collinearly: it refuses ``constrained_magnetization = 'total'``
        # for ``nspin = 2`` and offers ``tot_magnetization`` instead. Elk's
        # ``momfix`` is the three-vector for both regimes, and this follows Elk.)
        return fixed if noncollinear else fixed[2:3]

    if constraint == "total direction":
        return np.asarray(fixed_magnetization, dtype=float)[2:3]

    return np.zeros(0)


class MagneticField(eqx.Module):
    """The fields and constraints of one calculation.

    Everything is optional and the zero case is exactly free: with no field and
    no constraint the energy is identically zero and its gradient with it, so
    the object is simply absent from the calculation rather than adding a zero
    to every potential.
    """

    #: Per-atom integration regions; ``None`` when nothing is atom-resolved.
    regions: LocalRegions | None
    #: ``(ncomponent,)`` uniform field over the whole cell, Ry.
    uniform: jnp.ndarray
    #: ``(nat, ncomponent)`` field inside each atom's sphere, Ry, or ``None``.
    atomic: jnp.ndarray | None
    #: The constraint's target; see :func:`constraint_targets`.
    targets: jnp.ndarray | None
    #: ``lambda``, the penalty's stiffness. A convergence parameter, not a
    #: physical one -- QE's own advice is to converge with a small one and
    #: restart with a larger.
    penalty: float
    constraint: str = eqx.field(static=True, default="none")
    #: ``reducebf`` (Elk 5.104): after each SCF iteration the *external* fields
    #: are multiplied by this, so a field that breaks a symmetry at the start is
    #: effectively zero at the end. 1.0 leaves them alone. Constraints are not
    #: reduced -- a penalty is not a symmetry breaker.
    reducebf: float = eqx.field(static=True, default=1.0)
    #: Which fixed-spin-moment update to use; see :data:`FSM_UPDATES`.
    fsm_update: str = eqx.field(static=True, default=DEFAULT_FSM_UPDATE)
    #: The previous ``(field, moment)`` the secant update measured, or ``None``
    #: before it has one. Carried on the object because the field *is* the
    #: state that survives an SCF iteration -- there is nowhere else to put it
    #: that does not make the driver hold a controller of its own.
    previous_uniform: jnp.ndarray | None = None
    previous_moment: jnp.ndarray | None = None

    @property
    def has_field(self) -> bool:
        return bool(np.any(np.asarray(self.uniform) != 0.0)) or self.atomic is not None

    @property
    def active(self) -> bool:
        return self.has_field or self.constraint != "none"

    # --- the energy, which is the primitive ----------------------------------

    def sphere_moments(self, rho_r: jnp.ndarray, cell: Cell) -> jnp.ndarray:
        """``(nat, ncomponent)``: the moment inside each atom's sphere *now*.

        **Not** :attr:`System.local_moments`, which was also called
        ``local_moments`` and is a different object: that one is the per-atom
        moment an input file *asked for*, in Bohr magnetons, and it is what the
        magnetic symmetry group is decided from. This one is what the current
        density actually has in each sphere, and it exists only when a field or
        an atom-resolved constraint built the spheres. Two things a constraint
        compares against each other should not share a name.

        Refuses by name when the spheres were never built, rather than letting
        ``None.integrate`` surface as an ``AttributeError`` two frames down: the
        caller decides whether an atom-resolved scheme is in force, and getting
        that decision wrong is a *construction* bug in the caller
        (:data:`ATOM_RESOLVED`).
        """
        if self.regions is None:
            raise ValueError(
                f"constrained_magnetization = '{self.constraint}' is resolved "
                "by atom and needs the per-atom integration spheres, which "
                "this MagneticField was built without. Every atom-resolved "
                "scheme is listed in defumat.scf.fields.ATOM_RESOLVED -- read "
                "that set rather than naming the schemes again"
            )
        magnetization = magnetization_components(rho_r)
        scale = cell.volume / magnetization[0].size
        return scale * self.regions.integrate(magnetization)

    def total_moment(self, rho_r: jnp.ndarray, cell: Cell) -> jnp.ndarray:
        """``(ncomponent,)``: the moment of the whole cell."""
        magnetization = magnetization_components(rho_r)
        scale = cell.volume / magnetization[0].size
        return scale * jnp.sum(magnetization, axis=(1, 2, 3))

    def field_energy(self, rho_r: jnp.ndarray, cell: Cell, scale: float = 1.0):
        """``-int B . m``: the Zeeman energy of the fields put in by hand."""
        energy = -scale * jnp.dot(
            jnp.asarray(self.uniform), self.total_moment(rho_r, cell)
        )
        if self.atomic is not None:
            energy = energy - scale * jnp.sum(
                jnp.asarray(self.atomic) * self.sphere_moments(rho_r, cell)
            )
        return energy

    def constraint_energy(self, rho_r: jnp.ndarray, cell: Cell):
        """The penalty functional, ``etcon``, in Ry.

        The four forms are QE's ``i_cons`` 1, 2, 3 and 6, and each one is
        written here as the *energy* whose derivative ``add_bfield`` adds to the
        potential.
        """
        if self.constraint in ("none", "fsm"):
            return jnp.asarray(0.0)
        targets = jnp.asarray(self.targets)

        if self.constraint == "atomic":
            difference = self.sphere_moments(rho_r, cell) - targets
            return self.penalty * jnp.sum(difference**2)

        if self.constraint == "atomic texture":
            # ``sum_i (1 - m_i . n_i / |m_i|)``, zero when every moment points
            # where it was asked to and rising with the angle. The full unit
            # vector, where ``atomic direction`` takes the polar angle alone --
            # see :data:`CONSTRAINTS`.
            #
            # **Its gradient carries a 1/|m|**, which is what constraining a
            # direction and not a length means, and it is why this scheme is
            # hard to converge: as a site's moment shrinks its constraint
            # potential *grows*, which is positive feedback. Measured on a
            # two-hydrogen 120-degree cell: no lambda converged, and above
            # lambda = 2 one site's moment blew up to 2 mu_B while the other
            # went to zero. ``'atomic'`` constrains the vector, so its gradient
            # is ``2 lambda (m - m_target)`` and bounded -- see the warning in
            # :func:`constraint_targets`.
            moments = self.sphere_moments(rho_r, cell)
            cosine = _unit_cosine(moments, targets)
            return self.penalty * jnp.sum(1.0 - cosine)

        if self.constraint == "atomic direction":
            moments = self.sphere_moments(rho_r, cell)
            cosine = _polar_cosine(moments)
            return self.penalty * jnp.sum((cosine - targets[:, 0]) ** 2)

        if self.constraint == "total":
            return self.penalty * jnp.sum((self.total_moment(rho_r, cell) - targets) ** 2)

        if self.constraint == "total direction":
            moment = self.total_moment(rho_r, cell)
            angle, off_axis = _polar_angle(moment[None])
            error = angle[0] - jnp.deg2rad(targets[0])
            energy = self.penalty * error**2
            # QE's escape from the axis, as the energy term whose derivative is
            # its ``fact1(1) = 1.D-14``. ``stop_gradient`` on the prefactor is
            # what makes that true: the whole point is a *constant* field of
            # 1e-14 along x, not a second contribution to the penalty. It is
            # off by construction when the moment already points where it was
            # asked to, since ``error`` is then zero -- which covers QE's
            # ``IF (mcons(3,1) > 0)`` and one case QE's test does not: a moment
            # on the **-z** axis with a target of 0, where QE declines to kick
            # and the run stays stuck at pi.
            escape = AXIS_ESCAPE * moment[0] * jax.lax.stop_gradient(
                2.0 * self.penalty * error
            )
            return energy + jnp.where(off_axis[0], 0.0, escape)

        raise NotImplementedError(
            f"constrained_magnetization = {self.constraint!r} is not implemented; "
            f"available: {sorted(CONSTRAINTS)}"
        )

    def site_residuals(self, rho_r: jnp.ndarray, cell: Cell):
        """``(nat,)``: how far each atom is from its own target, or ``None``.

        ``constraint_energy`` is one scalar over every site, so under
        ``'atomic texture'`` a single flipped site out of fifteen reads as a
        small number indistinguishable from partial convergence everywhere.
        This is the same comparison, site by site.

        The unit follows the scheme, because the schemes constrain different
        things: **degrees** for the two direction schemes, which is the angle
        between the converged moment and the target, and **Bohr magnetons** for
        ``'atomic'``, which is the length of the vector difference. ``None``
        for the schemes with no per-site target at all.
        """
        if self.constraint not in ATOM_RESOLVED:
            return None
        targets = jnp.asarray(self.targets)
        moments = self.sphere_moments(rho_r, cell)
        if self.constraint == "atomic":
            return jnp.linalg.norm(moments - targets, axis=-1)
        if self.constraint == "atomic texture":
            cosine = _unit_cosine(moments, targets)
            return jnp.rad2deg(jnp.arccos(jnp.clip(cosine, -1.0, 1.0)))
        # ``atomic direction``: the target is the cosine of the polar angle, so
        # the residual is the difference of the two angles rather than of their
        # cosines -- a cosine difference is a very uneven measure of an angle
        # near the poles, and the poles are where a seeded run starts.
        angle, _ = _polar_angle(moments)
        target = jnp.arccos(jnp.clip(targets[:, 0], -1.0, 1.0))
        return jnp.rad2deg(jnp.abs(angle - target))

    def energy(self, rho_r: jnp.ndarray, cell: Cell, scale: float = 1.0):
        return self.field_energy(rho_r, cell, scale) + self.constraint_energy(rho_r, cell)

    # --- and the potential, which is its derivative --------------------------

    def potential(self, rho_r: jnp.ndarray, cell: Cell, scale: float = 1.0):
        """``(v, e_field, e_constraint)``: what ``add_bfield`` adds to ``v``.

        The gradient is with respect to the density *at the grid points*, so it
        carries the quadrature weight ``omega / N`` that the integrals put in;
        dividing it out is what makes ``v`` a potential rather than an energy
        per point, and is why the result pairs correctly with ``deband``'s
        ``-int rho v``.
        """
        weight = rho_r[0].size / cell.volume
        gradient = jax.grad(lambda rho: self.energy(rho, cell, scale))(rho_r)
        return (
            gradient * weight,
            self.field_energy(rho_r, cell, scale),
            self.constraint_energy(rho_r, cell),
        )

    def feedback(self, rho_r: jnp.ndarray, cell: Cell) -> "MagneticField":
        """Elk's fixed-spin-moment update: a field driven by the error, not a penalty.

        ``bfieldfsm.f90``: instead of adding ``lambda (m - m_fix)^2`` to the
        energy, the external field is nudged by ``tau (m - m_fix)`` after every
        iteration until the moment sits where it was asked to. The converged
        state is a genuine stationary point of the *unconstrained* functional
        under that field, where a penalty leaves a residual force -- which is
        why Elk reports the effective field it ended up with.

        Returns a new object with the uniform field updated; the calculation
        replaces its own with it once per SCF iteration.

        **The sign is not Elk's, because the field is not Elk's.** Elk writes
        ``B <- B + tau (M - M_fix)`` with a Hamiltonian term ``+(g_e/4c) sigma.B``,
        where a positive field *raises* the majority channel and so reduces the
        moment. Everything here is in QE's convention instead -- the field enters
        the potential as ``-B`` and its energy is ``-int B.m`` -- so the same
        feedback reads with a minus. Getting it backwards does not oscillate or
        diverge: it drives the field the wrong way until the moment saturates,
        and the run converges to the *unconstrained* answer looking untroubled.
        """
        if self.constraint != "fsm":
            return self
        moment = self.total_moment(rho_r, cell)
        error = moment - jnp.asarray(self.targets)
        if self.fsm_update == "elk":
            return eqx.tree_at(
                lambda field: field.uniform, self, self.uniform - self.penalty * error
            )
        return self._secant_step(moment, error)

    def _secant_step(self, moment: jnp.ndarray, error: jnp.ndarray) -> "MagneticField":
        """``B <- B - (m - m_fix) / chi`` with ``chi`` measured, not assumed.

        The susceptibility comes from the two most recent *converged* pairs, so
        this is a secant iteration on ``m(B)`` -- the same construction BFGS uses
        on a gradient, one dimension at a time because the components of a
        uniform field do not mix in the cases this scheme is for.

        Three guards, in the order they bite. Before there is a second point
        there is no secant, so the first step is Elk's, which is what ``tau`` is
        good for -- an order-of-magnitude guess at ``1/chi``. A susceptibility
        that is not positive and finite is a measurement of noise rather than of
        physics and is refused the same way. And a step is never longer than
        :data:`FSM_TRUST` times the last one, because ``chi`` near zero -- a
        saturated moment -- would otherwise ask for a field that leaves the
        physics the constraint was posed in.
        """
        step = -self.penalty * error
        if self.previous_uniform is not None:
            change = self.uniform - self.previous_uniform
            response = moment - self.previous_moment
            usable = jnp.abs(change) > FSM_MIN_STEP
            chi = jnp.where(usable, response / jnp.where(usable, change, 1.0), 0.0)
            secant = -error / jnp.where(chi > 0.0, chi, 1.0)
            trusted = FSM_TRUST * jnp.abs(change)
            secant = jnp.clip(secant, -trusted, trusted)
            step = jnp.where(usable & (chi > 0.0) & jnp.isfinite(secant), secant, step)
        updated = eqx.tree_at(lambda field: field.uniform, self, self.uniform + step)
        updated = eqx.tree_at(
            lambda field: field.previous_uniform, updated, self.uniform,
            is_leaf=lambda x: x is None,
        )
        return eqx.tree_at(
            lambda field: field.previous_moment, updated, moment,
            is_leaf=lambda x: x is None,
        )

    def cell_residual(self, rho_r: jnp.ndarray, cell: Cell):
        """``m - m_target`` as a 3-vector for a cell-wide constraint, or ``None``.

        The companion of :meth:`site_residuals` for the schemes that have **no**
        per-site target -- ``'total'``, ``'total direction'`` and ``'fsm'``.

        **It exists because ``fsm`` had no number at all.** A penalty's miss is
        visible in ``constraint_energy``, which is part of the energy and goes to
        zero as the constraint is met. ``fsm`` is a *feedback field* rather than a
        penalty, so its ``constraint_energy`` is **0 by construction** and
        nothing on the result said how far the run ended from what it was asked
        for: :meth:`satisfied` computed exactly this error, compared it against
        :data:`FSM_TOLERANCE`, returned a bool and threw the number away. A run
        that stops with ``converged = False`` then looks like an ordinary
        non-convergence even when its *density* is converged -- measured at
        ``accuracy = 3.9e-11`` against a ``conv_thr`` of 1e-10, with the moment
        0.174 mu_B from its target (``PLAN.md`` P80).

        In Bohr magnetons, signed and per component, because the sign is the
        whole of the diagnosis: a moment that has gone to the *other side* of its
        target is a different failure from one that has not got there yet, and a
        magnitude cannot tell them apart.

        ``None`` for ``'total direction'`` as well as for the atom-resolved
        schemes and for no constraint at all: that scheme's target is a polar
        **angle in degrees** (``constraint_targets``) rather than a moment, so a
        vector difference against it is not a residual of anything. Its own
        residual is an angle and belongs beside ``_polar_angle``; it is not
        written here rather than written wrongly.
        """
        if self.constraint not in ("total", "fsm"):
            return None
        return self.total_moment(rho_r, cell) - jnp.asarray(self.targets)

    def satisfied(self, rho_r: jnp.ndarray, cell: Cell) -> bool:
        """Whether a fixed-spin-moment run has actually fixed the moment.

        ``True`` for every other scheme -- a penalty is part of the energy, so
        its convergence is the density's -- and for ``fsm`` it is the extra
        condition the SCF loop has to pass before it stops. See
        :data:`FSM_TOLERANCE`.
        """
        if self.constraint != "fsm":
            return True
        error = self.total_moment(rho_r, cell) - jnp.asarray(self.targets)
        return bool(jnp.max(jnp.abs(error)) < FSM_TOLERANCE)

    def residual(self, scale: float) -> float:
        """The largest field component still applied, in Ry.

        ``scale`` is the driver's accumulated ``reducebf`` factor, which is a
        loop variable and is deliberately **not** on this object: ``reducebf``
        multiplies the loop's copy and leaves the field itself alone, so the
        applied field is only reconstructible from the pair.
        """
        largest = float(jnp.max(jnp.abs(self.uniform))) if self.uniform.size else 0.0
        if self.atomic is not None:
            largest = max(largest, float(jnp.max(jnp.abs(self.atomic))))
        return scale * largest

    def reduced(self) -> "MagneticField":
        """The fields multiplied by ``reducebf``, as Elk does after each loop."""
        if self.reducebf == 1.0:
            return self
        updated = eqx.tree_at(lambda f: f.uniform, self, self.uniform * self.reducebf)
        if self.atomic is not None:
            updated = eqx.tree_at(
                lambda f: f.atomic, updated, updated.atomic * self.reducebf
            )
        return updated


def _safe_modulus(moments: jnp.ndarray):
    """``(|m|, is_there_a_moment)`` with a finite derivative at ``m = 0``.

    ``sqrt`` has an **infinite** derivative at zero, so masking its *result*
    leaves ``0 * inf`` -- a NaN -- in the tangent however carefully the division
    afterwards is guarded. The mask has to go on the argument the derivative is
    taken at, which is the sum of squares, and that is the P70 lesson in its
    smallest form. A cell with vacuum, or a ligand with no induced moment,
    reaches this on every gradient.
    """
    square = jnp.sum(moments**2, axis=-1)
    present = square > VANISHING_MOMENT**2
    return jnp.sqrt(jnp.where(present, square, 1.0)), present


def _polar_cosine(moments: jnp.ndarray) -> jnp.ndarray:
    """``m_z / |m|`` per row, zero where there is no moment to take it of."""
    modulus, present = _safe_modulus(moments)
    return jnp.where(present, moments[..., -1] / modulus, 0.0)


def _polar_angle(moments: jnp.ndarray):
    """``(theta, is_it_off_the_axis)`` per row, with QE's guard on the axis.

    Written as ``atan2(|m_perp|, m_z)`` rather than as ``arccos(m_z / |m|)``,
    which is the same angle and a different derivative. ``arccos'`` diverges at
    ``+-1`` while ``d|m_perp|/dm_x`` vanishes there, so the chain is ``0 * inf``
    and every component of the potential comes back NaN -- and the axis is not
    a corner case but the state a run seeded from ``starting_magnetization``
    with no angles **starts** in.

    Both arguments of ``atan2`` are masked at the value the derivative is taken
    at, which is the P70 lesson: ``|m_perp|`` at its own square, so its tangent
    is zero rather than infinite on the axis, and ``m_z`` where there is no
    moment at all, since ``atan2``'s own JVP is ``0/0`` at the origin. What
    comes out is QE's ``fact1`` exactly -- ``m_x m_z / (m_perp |m|^2)`` and
    ``-m_perp / |m|^2`` off the axis, and zero on it
    (``add_bfield.f90:184-192``).
    """
    square_perp = jnp.sum(moments[..., :2] ** 2, axis=-1)
    off_axis = square_perp > VANISHING_TRANSVERSE**2
    perp = jnp.where(
        off_axis, jnp.sqrt(jnp.where(off_axis, square_perp, 1.0)), 0.0
    )
    _, present = _safe_modulus(moments)
    along_z = jnp.where(present, moments[..., 2], 1.0)
    return jnp.arctan2(perp, along_z), off_axis


def _unit_cosine(moments: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """``m . n / |m|`` per row for unit ``n``, and **1** where there is no moment.

    One rather than zero: the penalty is ``sum(1 - cos)``, so a site with nothing
    to point contributes nothing rather than a full unit of penalty it cannot act
    on. That is the same choice ``add_bfield`` makes when it stops instead of
    dividing, written as a value because this expression is differentiated.
    """
    modulus, present = _safe_modulus(moments)
    cosine = jnp.sum(moments * targets, axis=-1) / modulus
    return jnp.where(present, cosine, 1.0)
