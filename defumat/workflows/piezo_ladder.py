"""The piezoelectric tensor on a ladder of k-meshes, which is how it is checked.

``PLAN.md`` P50. The clamped-ion piezoelectric tensor has a validation problem
that none of its own checks can reach: **every internal statement it carries is
blind to the Brillouin-zone sum.** The three routes to it share one field
response, so they check the assembly and not the integral; the symmetry
statements hold on any mesh; the wedge against the closed grid is the same
sample twice; and the ``Z*`` anchor is the same assembly run in the position
coordinate, so it moves with the mesh exactly as the tensor does. Measured
against a Berry-phase finite difference, which shares no machinery at all,
``e_14`` on zincblende AlAs reads -0.7638 at ``4 4 4`` and -0.6699 at
``10 10 10`` against a Berry value of -0.6614 to -0.6620, so a committed-quality
input returns a number **thirteen per cent** out with nothing to say about it.

What is left, then, is to move the mesh and watch. That is what this does: one
ground state and one response per rung, the tensor at each, and the relative
change across each step. The last step goes on the tensor as
:attr:`~defumat.response.piezo.PiezoelectricTensor.kmesh_drift`, which is what
silences :func:`~defumat.response.piezo._warn_about_the_kmesh` once it is small
enough to stop being the dominant error.

**The step is not the error and the distance between them is measured.** On the
cell above the ladder moves 10 per cent, 2.1 per cent and 0.44 per cent across
its three steps while the last rung is still 1.2 per cent from the Berry value,
so the remaining distance runs about three times the last step; the approach is
a slow tail rather than a geometric one.
:data:`~defumat.response.piezo.KMESH_STEP` is a place to stop warning and not a
claim of convergence, and the honest reading of a quiet ladder is that the mesh
is no longer the largest thing wrong.

**The rungs are wedges by default, and that is a cost decision with a
measurement behind it.** An unshifted Monkhorst-Pack grid is closed under the
point group, so its irreducible wedge and the whole grid are two routes to one
integral: on ultrasoft AlAs the contracted route took the symmetrised 8-point
wedge and the whole 64-point grid to the same tensor, 1.6e-06 on a value of
0.82. Sixteen points where the grid has 216 is what makes a ``6 6 6`` rung a
workstation job. ``wedge = False`` runs the whole grid for a cell whose
reduction is in doubt, and a ``nosym`` system gets the whole grid whatever this
argument says, because it has no group to reduce with.

**Every rung is unshifted**, whatever the input asked for. A *shifted* grid is
not closed under the point group, so a response on its wedge is not a vector
field the symmetriser can complete (P24), and comparing rungs that sample the
zone differently would measure the shift rather than the mesh.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass

import numpy as np

from defumat.response.piezo import KMESH_STEP, piezoelectric_tensor
from defumat.scf import Calculation, run_scf
from defumat.system.kpoints import KPoints, for_spin

__all__ = ["PiezoelectricLadder", "piezoelectric_kmesh_ladder"]


@dataclass(frozen=True)
class PiezoelectricLadder:
    """What :func:`piezoelectric_kmesh_ladder` returns.

    Attributes:
        meshes: the Monkhorst-Pack divisions of each rung, coarsest first.
        nk: how many k-points each rung integrated, which is the cost and is
            the number that says whether the reduction happened.
        tensors: the :class:`~defumat.response.piezo.PiezoelectricTensor` of
            each rung, the last of which carries ``kmesh_drift``.
        e14: each rung's ``e_14`` in C/m^2, which is the single independent
            component of a zincblende crystal and the one every table quotes.
            It is a convenience and not the ladder: ``steps`` is taken on the
            whole tensor.
        steps: the relative change of the whole tensor across each step,
            ``max|e_n - e_(n-1)| / max|e_n|``, with one fewer entry than
            ``meshes``.
        drift: the last entry of ``steps``, and ``None`` for a one-rung ladder.
        seconds: the wall clock of each rung, ground state and response
            together.
        converged: whether every rung's field response converged.
    """

    meshes: tuple
    nk: tuple
    tensors: tuple
    e14: tuple
    steps: tuple
    drift: float | None
    seconds: tuple
    converged: bool

    @property
    def tensor(self):
        """The densest rung's tensor, which is the answer the ladder is for."""
        return self.tensors[-1]

    @property
    def settled(self) -> bool:
        """Whether the last step is below :data:`KMESH_STEP`.

        **A quiet ladder, not a converged one.** The step is smaller than the
        error and the module docstring carries the factor.
        """
        return self.drift is not None and self.drift < KMESH_STEP


def _rung(system, mesh: int, wedge: bool):
    """``system`` sampled on the unshifted ``mesh^3`` grid, or on its wedge.

    Built the way :meth:`defumat.system.builder.System._rebuild_kpoints` builds
    it, from the crystal's own group with time reversal unless ``noinv``, and
    normalised with :func:`~defumat.system.kpoints.for_spin` for the reason
    :meth:`defumat.calculator.Calculator.with_kpoints` gives at length: every
    ``KPoints`` constructor applies the unpolarized ``degspin`` unconditionally,
    and a set substituted without that step counts every electron twice and does
    not fail, it moves the Fermi level.
    """
    rotations = t_rev = None
    if wedge and not system.nosym:
        symmetries = system.symmetry_group()
        rotations = symmetries.rotation_array()
        t_rev = symmetries.t_rev_array()
    kpoints = KPoints.automatic(
        (int(mesh),) * 3, (0, 0, 0), system.cell,
        precision=system.kpoints.precision,
        rotations=rotations,
        time_reversal=not system.noinv and not system.domag,
        t_rev=t_rev,
    )
    return dataclasses.replace(system, kpoints=for_spin(kpoints, system.nspin))


def piezoelectric_kmesh_ladder(
    system,
    pseudos,
    *,
    meshes=(4, 6, 8),
    wedge: bool = True,
    conv_thr: float = 1.0e-10,
    max_iterations: int = 100,
    verbose: bool = False,
    scf_options: dict | None = None,
    **options,
) -> PiezoelectricLadder:
    """The tensor at each mesh, and how much the last step moved it.

    Args:
        system: the crystal. Its own k-set is not used: each rung builds an
            unshifted grid of its own, for the reason the module docstring
            gives about a shifted grid and the point group.
        pseudos: the pseudopotentials, as :func:`~defumat.scf.run_scf` takes
            them.
        meshes: the Monkhorst-Pack divisions, coarsest first. Each is one
            ground state and one field response, so the cost is the whole
            calculation over again per rung and the last rung dominates it.
        wedge: reduce each grid with the crystal's point group. The same
            sample and a fraction of the cost; see the module docstring for
            the measurement that licenses it.
        conv_thr: the ground state's, tight by default because what is being
            read is a difference between rungs.
        scf_options: anything else for :func:`~defumat.scf.run_scf`.
        options: passed to
            :func:`~defumat.response.piezo.piezoelectric_tensor`, so
            ``method = 'zstar_eu'`` reaches it -- and on an augmented dataset
            that is the route to use, since the taped one holds a
            forward-over-reverse tape of 1.6 GiB a k-point and a ladder is
            exactly where that bites.

    Returns:
        :class:`PiezoelectricLadder`.

    **The rungs are run coarsest first and nothing is warm-started between
    them.** A denser grid is a different integral over the zone rather than a
    refinement of the same one, and seeding it with the coarse density would
    make each rung's answer depend on the rung below it, which is the one thing
    a convergence test must not do.
    """
    meshes = tuple(int(mesh) for mesh in meshes)
    if not meshes:
        raise ValueError("a ladder needs at least one mesh")
    if len(set(meshes)) != len(meshes):
        raise ValueError(f"the meshes repeat: {meshes}")
    if list(meshes) != sorted(meshes):
        raise ValueError(
            f"the meshes run coarsest first, and these do not: {meshes}. The "
            "last rung is the answer and the last step is the drift, so the "
            "order is part of the result rather than a presentation choice"
        )

    tensors, counts, seconds = [], [], []
    calculation = None
    for mesh in meshes:
        started = time.time()
        sampled = _rung(system, mesh, wedge)
        calculation = Calculation(sampled, pseudos)
        result = run_scf(sampled, pseudos, calculation=calculation,
                         conv_thr=conv_thr, max_iterations=max_iterations,
                         **(scf_options or {}))
        tensor = piezoelectric_tensor(
            calculation, result, verbose=verbose, kmesh_warning=False, **options
        )
        tensors.append(tensor)
        counts.append(int(sampled.kpoints.nk))
        seconds.append(time.time() - started)
        if verbose:
            print(f"    {mesh} {mesh} {mesh}: {counts[-1]} k-points, "
                  f"e_14 = {tensor.e14:.6f} C/m^2, {seconds[-1]:.1f} s",
                  flush=True)

    steps = []
    for before, after in zip(tensors, tensors[1:]):
        scale = float(np.abs(np.asarray(after.e)).max())
        moved = float(np.abs(np.asarray(after.e) - np.asarray(before.e)).max())
        steps.append(moved / scale if scale > 0.0 else moved)
    drift = steps[-1] if steps else None

    # The densest rung is the answer, so it is the one that carries the drift
    # and the one whose warning has to be able to fire. Above ``KMESH_STEP`` the
    # entry point says so with the measured number in place of AlAs's curve.
    tensors[-1] = dataclasses.replace(tensors[-1], kmesh_drift=drift)
    if drift is not None and drift >= KMESH_STEP:
        from defumat.response.piezo import _warn_about_the_kmesh

        _warn_about_the_kmesh(calculation, drift)

    return PiezoelectricLadder(
        meshes=meshes,
        nk=tuple(counts),
        tensors=tuple(tensors),
        e14=tuple(float(tensor.e14) for tensor in tensors),
        steps=tuple(steps),
        drift=drift,
        seconds=tuple(seconds),
        converged=all(bool(tensor.converged) for tensor in tensors),
    )
