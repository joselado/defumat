"""The ``&control``, ``&ions`` and ``&cell`` variables an optimisation reads.

Until this existed a ``pw.x`` input's convergence thresholds were **parsed and
then ignored**: ``run_relax`` took ``etot_conv_thr``, ``forc_conv_thr`` and
``nstep`` as Python arguments with QE's defaults baked in, and nothing carried
the file's values to them. The two codes then stopped at different points on
the same curve and both reported success -- measured on ``si10-nc-relax.in``,
whose ``forc_conv_thr = 1e-4`` ``pw.x`` honoured over 26 BFGS steps while this
code stopped after 11 at a residual force of 9.5e-4, leaving the geometries
0.057 bohr apart (`PLAN.md` P28b). That is the failure mode this package treats
as worse than a crash, so the namelists are read here and threaded onto
:class:`~defumat.system.builder.System` like every other input variable.

**Units are the input's at this boundary and Rydberg atomic units past it.**
``press`` and ``press_conv_thr`` are in **kbar** in a ``pw.x`` file, as
``INPUT_PW.txt`` says, and are converted exactly once -- in
:func:`~defumat.workflows.vc_relax.run_vc_relax`, at the point of use, the way
``RY_TO_KBAR`` is applied at the printing boundary and nowhere else.

Defaults are ``Modules/read_namelists.f90``'s, including the one that is
conditional: ``calculation = 'vc-relax'`` sets ``ion_dynamics = 'bfgs'`` *and*
``cell_dynamics = 'bfgs'`` for ``pw.x``, where the bare ``&cell`` default is
``'none'``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["RelaxSettings"]


@dataclass(frozen=True)
class RelaxSettings:
    """What a relaxation reads from the input, in the input's own units."""

    #: ``&control``. Both thresholds must be met, and ``nstep`` caps the ionic
    #: steps (QE's default is 50 for a relaxation and 1 for an SCF).
    etot_conv_thr: float = 1.0e-4
    forc_conv_thr: float = 1.0e-3
    nstep: int = 50
    #: ``&ions``.
    ion_dynamics: str = "bfgs"
    #: ``upscale``: how much tighter than the input ``conv_thr`` the SCF may
    #: become as the forces get small (``move_ions.f90``).
    upscale: float = 100.0
    #: ``&cell``. ``press`` and ``press_conv_thr`` are in **kbar**.
    cell_dynamics: str = "bfgs"
    press: float = 0.0
    press_conv_thr: float = 0.5
    cell_dofree: str = "all"
    #: ``treinit_gvecs``: rebuild the grids on every accepted step instead of
    #: once for the run.
    treinit_gvectors: bool = False
