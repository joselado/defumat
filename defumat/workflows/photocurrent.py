"""``run_shift_current``: the bulk photovoltaic effect from a converged density.

One fixed-density run with empty states, then the second-order sum of
:mod:`defumat.response.photocurrent`. The same shape as
:func:`~defumat.workflows.conductivity.run_conductivity`, and the same two
questions decide whether the answer is any good.

**The k-set.** A shift current is a Brillouin-zone integral of a quantity with
a delta-function resonance in it, so it wants a grid far denser than the one
that converged the density -- and it wants the *whole* grid, because
``sigma^abc`` is a polar rank-3 tensor and a symmetry-reduced wedge does not
sum to the cell's until it is averaged over the point group. ``kpoints`` is what
lets the density be converged where it is cheap and the current evaluated where
it is right.

**The band count.** It is the convergence parameter of the whole phase and it
is worse here than for an absorption spectrum, because the *intermediate* sum
of the generalised derivative runs over the same bands and converges only when
they span the space. :attr:`~defumat.response.photocurrent.ShiftCurrent.
truncation` measures it per run; the module docstring measures what it is worth
on AlAs.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from defumat.response.photocurrent import (
    ShiftCurrent,
    require_a_shift_current_regime,
    shift_current,
)
from defumat.workflows.nscf import fixed_density_states

__all__ = ["run_shift_current"]


def run_shift_current(
    system,
    pseudos,
    density,
    *,
    kpoints=None,
    nbnd: int | None = None,
    frequencies=None,
    window: float = 1.0,
    nw: int = 200,
    broadening: float = 0.01,
    smearing: str = "gaussian",
    degeneracy_tol: float | None = None,
    conv_thr: float = 1.0e-10,
    k_batch="default",
) -> ShiftCurrent:
    """``sigma^abc(0; w, -w)`` in A/V^2 for a converged run.

    Args:
        system: the converged :class:`~defumat.system.builder.System`.
        pseudos: its pseudopotentials.
        density: the converged density.
        kpoints: a denser k-set to evaluate on. It must be the **whole**
            unshifted grid; a wedge is refused by name.
        nbnd: how many bands to diagonalise. Both sums use them.

    The remaining arguments are
    :func:`~defumat.response.photocurrent.shift_current`'s.
    """
    from defumat.scf.driver import Calculation

    if kpoints is not None:
        import equinox as eqx

        from defumat.system.kpoints import for_spin

        # The ``for_spin`` boundary P51 records: every ``KPoints`` constructor
        # applies the unpolarized factor of two unconditionally and a spinor
        # band holds one electron, so a caller-built mesh has to cross it here.
        # It is idempotent, so a correctly scaled set passes through untouched.
        system = eqx.tree_at(
            lambda s: s.kpoints, system, for_spin(kpoints, system.nspin)
        )
    # Checked before the fixed-density run, as ``run_conductivity`` checks its
    # own: a caller asking for something this cannot do should not first pay
    # for the empty states.
    require_a_shift_current_regime(Calculation(system, pseudos, k_batch=k_batch))

    if nbnd is None:
        raise ValueError(
            "run_shift_current needs an explicit nbnd: the intermediate sum of "
            "the generalised derivative runs over the same bands as the pair "
            "sum, so the band count is the convergence parameter and there is "
            "no default that is right for a material. Start from four times "
            "the occupied count and read ShiftCurrent.truncation"
        )
    nbnd = int(nbnd)

    # One band more than the sum uses, for ``band_cut_gap`` -- the same
    # measurement ``run_conductivity`` buys the same way, and for the same
    # reason: where the truncation falls matters more than how far out it is.
    calculation, system, eigenvalues, wavefunctions = fixed_density_states(
        system, pseudos, density, nbnd=nbnd + 1,
        conv_thr=conv_thr, k_batch=k_batch,
    )
    eigenvalues = jnp.asarray(eigenvalues)
    if eigenvalues.ndim == 2:
        eigenvalues = eigenvalues[None]
    band_cut_gap = float(
        np.min(np.asarray(eigenvalues)[..., nbnd]
               - np.asarray(eigenvalues)[..., nbnd - 1])
    )
    eigenvalues = eigenvalues[..., :nbnd]
    wavefunctions = jnp.asarray(wavefunctions)[..., :nbnd, :]
    potential = calculation.potential(jnp.asarray(density))

    return shift_current(
        calculation, wavefunctions, eigenvalues, potential.v_scf,
        frequencies=frequencies, window=window, nw=nw, broadening=broadening,
        smearing=smearing, degeneracy_tol=degeneracy_tol,
        band_cut_gap=band_cut_gap, k_batch=k_batch,
    )
