"""The magnetoelectric tensor: the polarization a magnetic field induces.

``alpha_ij = dP_i / dB_j`` -- how much electric polarization appears along ``i``
when a magnetic field is applied along ``j``. It is the defining response of a
magnetoelectric crystal and, by the Maxwell relation, equally the magnetization
a *electric* field induces.

This is Elk's task 390 (``magnetoelt.f90``) and it is computed Elk's way, which
is the expensive way and here the only available one: a full ground state at
``B -+ delta/2`` for each Cartesian direction, the Berry-phase polarization of
each, and a central difference. The cheap route -- one ``jvp`` of the
magnetization along an electric field's response, which is P50's assembly with
the stress replaced by ``M`` -- needs a *noncollinear* Sternheimer solve, which
is the largest of the response stack's remaining refusals, and it needs an
electric field as a ground-state perturbation, which this package does not have
(``tefield`` and ``lelfield`` are both refused at input). The magnetic field is
the differentiable coordinate here and the electric one is not, which is the
reverse of Elk's situation and is what fixes the route.

**Two symmetries must both be broken or the answer is zero for a reason that
has nothing to do with the calculation.** Time reversal maps ``alpha`` to
``-alpha``, and so does inversion. Elk's own GaAs example is built around the
first: a non-magnetic semiconductor has no linear tensor at all, so a *large*
external field is applied to break time reversal by hand and the derivative is
taken against a small further change in it. The second is why the crystal must
not be centrosymmetric -- and why a centrosymmetric magnet is not a test case
but a zero, the trap ``ELK-FEATURES.md`` records for the piezoelectric tensor.

**Spin-orbit coupling is required and is the sharpest check.** Without it the
spin and the lattice decouple: a global spin rotation maps ``B`` to ``-B`` and
leaves every charge observable fixed, so ``P(+B) = P(-B)`` and the tensor
vanishes identically. Running the same cell with ``lspinorb = .false.`` is
therefore a null that no plausible bug survives, and it costs one more pass.

**Only the column parallel to the seeded magnetization is reachable on the
committed cell, and the reason is the seed rather than the field.** A field along
the direction ``starting_magnetization`` points converges in 12 iterations; one
with a *transverse* component does not reach ``conv_thr = 1e-10`` in 150, because
rotating the moment is the slow mode and the transverse magnetization is three
orders of magnitude smaller than the longitudinal one it has to be resolved
against. Measured on AlAs: ``B`` along ``z`` (the seed) converges in 12
iterations; ``B`` along ``x`` does not converge in 150, and neither does a field
tilted 11 degrees off ``z``. It is **not** the tilt -- a pure transverse field
fails on its own -- which is what the diagnostic was run to separate.

So ``directions`` defaults to all three and a caller wanting an off-axis column
should seed the magnetization along that column (``angle1``/``angle2`` in the
input) rather than raise ``max_iterations``, which does not help. The guard on an
unconverged ground state is what makes this visible instead of returning the
mixer's residue divided by ``delta``.

**What this computes is the spin (Zeeman) response, clamped-ion.** The field
enters through :mod:`defumat.scf.fields`, which is a Zeeman term on the
magnetization and carries no orbital vector potential, so the orbital
contribution to the magnetoelectric coupling is absent -- as it is from Elk's
task 390, which applies ``bfieldc`` the same way. The atoms are held, so this is
the clamped-ion tensor, the same restriction :mod:`defumat.response.piezo`
carries.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

__all__ = ["MagnetoelectricTensor", "magnetoelectric_tensor"]

#: Above this fraction of a polarization quantum, a step of the field has moved
#: the phase far enough that the two ends of the difference may be on different
#: branches -- and a difference across a branch is the quantum divided by the
#: step, which is large, smooth and completely wrong.
BRANCH_FRACTION = 0.25


@dataclass(frozen=True)
class MagnetoelectricTensor:
    """``alpha_ij = dP_i/dB_j``, and the pieces it was built from."""

    #: ``(3, 3)`` in cartesian coordinates: rows are the polarization direction
    #: ``i``, columns the field direction ``j``. Units are ``e/bohr^2`` per
    #: whatever unit :attr:`field` is in, which is the package's internal
    #: Rydberg field -- **not** Elk's ``bfieldc``, whose normalisation differs.
    alpha: np.ndarray
    #: ``(3, 3)``: the raw ``d(phase)/dB_j`` along each reciprocal direction,
    #: which is what ``magnetoelt.f90`` prints as its lattice-coordinate tensor.
    phase_derivative: np.ndarray
    #: The field the difference was taken about, and the step used.
    field: np.ndarray
    delta: float
    #: The polarization quantum, and the largest step any phase took as a
    #: fraction of it. A large value means the branch tracking is doing work and
    #: the answer should not be trusted -- see :data:`BRANCH_FRACTION`.
    quantum: float
    largest_step: float
    #: The cell volume in bohr^3, which the cartesian tensor is divided by.
    volume: float

    @property
    def is_symmetric(self) -> bool:
        """Whether ``alpha`` is symmetric, which it need not be in general."""
        return bool(np.allclose(self.alpha, self.alpha.T, atol=1e-8))

    @property
    def magnitude(self) -> float:
        """The largest absolute component, for a quick "is it zero" read."""
        return float(np.max(np.abs(self.alpha)))


def magnetoelectric_tensor(
    system,
    pseudos,
    delta: float = 0.05,
    nppstr: int = 8,
    transverse: tuple[int, int] = (4, 4),
    directions=(0, 1, 2),
    chain: bool = False,
    scf_conv_thr: float = 1.0e-10,
    polarization_conv_thr: float = 1.0e-11,
    max_iterations: int = 120,
    scf_options: dict | None = None,
    **polarization_options,
) -> MagnetoelectricTensor:
    """``alpha_ij = dP_i/dB_j`` by a central difference of the Berry phase.

    Args:
        delta: the change in the field, Elk's ``deltabf``. The whole step is
            ``delta``; the two ground states are at ``B -+ delta/2``, as
            ``magnetoelt.f90`` does it.
        directions: which cartesian components of the field to step. Each costs
            two self-consistent runs and three polarizations.
        chain: seed each run from the previous converged state, which is Elk's
            ``trdstate = .true.`` after the first ground state. **Off by
            default here, and that is a departure from Elk with a measured
            reason.** Seeding the ``-delta/2`` run from the ``+delta/2`` one
            breaks the very symmetry a null rests on -- without spin-orbit
            coupling the two differ by ``B -> -B``, which is exactly what the
            chain makes asymmetric -- and it biases the difference at the level
            a small response lives at. Measured on the spin-orbit null: the
            residue is 3.9e-8 chained and 3.4e-9 unchained, and *only* once the
            polarization threshold is tightened with it. The two are right
            together and neither alone helps: chained at a tight threshold is
            6.9e-8, unchained at a loose one 5.3e-8. It buys little here anyway
            (45 s either way on the committed cell), because each run converges
            in nine iterations from the atomic density.
        scf_conv_thr: how tightly each of the six ground states is converged.
        polarization_conv_thr: how tightly the fixed-density diagonalisation
            *inside* each polarization is converged. **The two are named apart
            because they are not interchangeable and the difference is not
            small.** The residue of the spin-orbit null is floored by this one,
            not by ``scf_conv_thr``, which is why a diagnostic that varied only
            the latter found nothing; and a facade that forwarded the input
            file's ``conv_thr`` -- an *SCF* number -- into the polarization slot
            while leaving the ground states on their defaults turned a null
            ratio of 1293 into 4.
        max_iterations: the ground states' iteration cap.
        scf_options: further arguments for
            :func:`~defumat.scf.driver.run_scf`; the three above are merged
            into it and an explicit key here wins.
        polarization_options: forwarded to
            :func:`~defumat.workflows.polarization.run_polarization`.

    **Choose ``delta`` against the noise floor rather than by making it small.**
    A magnetoelectric response is small, and the difference of two independently
    converged ground states has a floor set by ``conv_thr``; two re-converged
    *identical* runs bound it. Too small a step measures that floor and too
    large a one leaves the linear regime, which is why Elk's example carries a
    ``deltabf`` at all rather than a rule.
    """
    from defumat.scf.driver import run_scf
    from defumat.workflows.polarization import run_polarization

    scf_options = dict(scf_options or {})
    scf_options.setdefault("conv_thr", scf_conv_thr)
    scf_options.setdefault("max_iterations", max_iterations)
    polarization_options.setdefault("conv_thr", polarization_conv_thr)
    base = np.asarray(system.b_field, dtype=float)
    if system.nspin_mag != 4:
        raise ValueError(
            "a magnetoelectric tensor needs a run that carries a magnetization "
            f"vector (nspin_mag = 4), and this one has nspin_mag = "
            f"{system.nspin_mag}. A noncollinear run whose starting_magnetization "
            "is zero everywhere allocates no magnetic density, so the field has "
            "nothing to couple to and every polarization comes out identical"
        )

    phase_derivative = np.zeros((3, 3))
    quantum = 0.0
    volume = 0.0
    lattice = None
    largest_step = 0.0
    previous = None

    for axis in directions:
        phases = {}
        for sign in (+1.0, -1.0):
            field = base.copy()
            field[axis] += sign * 0.5 * delta
            # ``b_field`` is a *static* field of the module, so ``tree_at``
            # cannot reach it -- it walks leaves and a static field is not one.
            # ``dataclasses.replace`` is the idiom for a frozen module's static
            # configuration, and it shares every array with the original, so the
            # six systems here cost one cell between them.
            moved = dataclasses.replace(
                system, b_field=tuple(float(v) for v in field)
            )
            options = dict(scf_options)
            if chain and previous is not None:
                options["starting_from"] = previous
            result = run_scf(moved, pseudos, **options)
            if not result.converged:
                raise RuntimeError(
                    f"the ground state at B = {tuple(field)} did not converge; "
                    "a magnetoelectric difference of an unconverged state is "
                    "the mixer's residue divided by delta"
                )
            previous = result if chain else None
            phases[sign] = _phases(moved, pseudos, result, nppstr, transverse,
                                   run_polarization, polarization_options)
            quantum = phases[sign][1]
            volume = phases[sign][2]
            lattice = phases[sign][3]

        step = np.asarray(phases[+1.0][0]) - np.asarray(phases[-1.0][0])
        largest_step = max(largest_step, float(np.max(np.abs(step)) / quantum))
        phase_derivative[:, axis] = step / delta

    if largest_step > BRANCH_FRACTION:
        raise RuntimeError(
            f"a field step moved the Berry phase by {largest_step:.2f} of a "
            f"quantum, which is more than {BRANCH_FRACTION}: the two ends of "
            "the difference may be on different branches, and a difference "
            "taken across one is the quantum over delta -- large, smooth and "
            "meaningless. Reduce delta"
        )

    # Omega P_cart = sum_k phi_k a_k, so d(P_cart)_i/dB_j is the same sum of the
    # phase derivatives over the lattice vectors, divided by the volume.
    alpha = (lattice.T @ phase_derivative) / volume
    return MagnetoelectricTensor(
        alpha=alpha,
        phase_derivative=phase_derivative,
        field=base,
        delta=float(delta),
        quantum=float(quantum),
        largest_step=float(largest_step),
        volume=float(volume),
    )


def _phases(system, pseudos, result, nppstr, transverse, run_polarization,
            options) -> tuple:
    """The three directions' total phases at one converged state.

    The whole mixed state crosses -- ``becsum`` for an ultrasoft or PAW dataset,
    and the *converged* field rather than the input's, which is the distinction
    ``DFTSource`` refuses without.
    """
    phases = np.zeros(3)
    quantum = volume = 0.0
    lattice = None
    for gdir in range(3):
        polarization = run_polarization(
            system, pseudos, result.density, gdir=gdir, nppstr=nppstr,
            transverse=transverse, becsum=result.becsum,
            field=result.magnetic_field, field_scale=result.field_scale,
            **options,
        )
        phases[gdir] = polarization.total_phase
        quantum = polarization.quantum
        volume = polarization.volume
    lattice = np.asarray(system.cell.to_cartesian(np.eye(3)))
    return phases, quantum, volume, lattice
