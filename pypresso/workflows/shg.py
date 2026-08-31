"""``run_shg``: second-harmonic generation from a converged density.

One fixed-density run with empty states, then the second-order sum of
:mod:`pypresso.response.shg`. The same shape as
:func:`~pypresso.workflows.photocurrent.run_shift_current`, and the same two
questions decide whether the answer is any good.

**The k-set.** ``chi^abc`` is a Brillouin-zone integral of a quantity with two
resonant denominators in it, so it wants a grid far denser than the one that
converged the density -- Elk's own GaAs example asks for 42x42x42 and says so
in a comment -- and it wants the *whole* grid, because ``chi^abc`` is a polar
rank-3 tensor and a symmetry-reduced wedge does not sum to the cell's until it
is averaged over the point group. ``kpoints`` is what lets the density be
converged where it is cheap and the susceptibility evaluated where it is right.

**The band count.** It is the convergence parameter of the whole quantity, and
worse here than for an absorption spectrum: the sum over the intermediate state
is the sum-rule expansion of the generalised derivative, which is an identity
only over a complete basis. :attr:`~pypresso.response.shg.SecondHarmonic.
truncation` measures it per run.

**And the gap.** A second-order susceptibility carries two energy denominators
rather than one, so an LDA or GGA gap error does not scale the answer, it moves
its resonances and changes its shape. ``scissor`` is the usual rigid shift;
Elk's example applies 1.243 eV to GaAs and cites a paper for it.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from pypresso.response.shg import (
    SecondHarmonic,
    require_an_shg_regime,
    second_harmonic,
)
from pypresso.workflows.nscf import fixed_density_states

__all__ = ["run_shg"]


def run_shg(
    system,
    pseudos,
    density,
    *,
    kpoints=None,
    nbnd: int | None = None,
    frequencies=None,
    window: float = 0.6,
    nw: int = 200,
    broadening: float = 0.003,
    scissor: float = 0.0,
    degeneracy_tol: float | None = None,
    conv_thr: float = 1.0e-10,
    k_batch="default",
) -> SecondHarmonic:
    """``chi^(2)(-2w; w, w)`` in pm/V for a converged run.

    Args:
        system: the converged :class:`~pypresso.system.builder.System`.
        pseudos: its pseudopotentials.
        density: the converged density.
        kpoints: a denser k-set to evaluate on. It must be the **whole**
            unshifted grid; a wedge is refused by name.
        nbnd: how many bands to diagonalise. Both sums use them.

    The remaining arguments are
    :func:`~pypresso.response.shg.second_harmonic`'s.
    """
    from pypresso.scf.driver import Calculation

    if kpoints is not None:
        import equinox as eqx

        from pypresso.system.kpoints import for_spin

        # The ``for_spin`` boundary P51 records: every ``KPoints`` constructor
        # applies the unpolarized factor of two unconditionally and a spinor
        # band holds one electron, so a caller-built mesh has to cross it here.
        system = eqx.tree_at(
            lambda s: s.kpoints, system, for_spin(kpoints, system.nspin)
        )
    # Checked before the fixed-density run, as ``run_shift_current`` checks its
    # own: a caller asking for something this cannot do should not first pay
    # for the empty states.
    require_an_shg_regime(Calculation(system, pseudos, k_batch=k_batch))

    if nbnd is None:
        raise ValueError(
            "run_shg needs an explicit nbnd: the sum over the intermediate "
            "state runs over the same bands as the pair sum, so the band "
            "count is the convergence parameter and there is no default that "
            "is right for a material. Start from four times the occupied "
            "count and read SecondHarmonic.truncation"
        )
    nbnd = int(nbnd)

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

    return second_harmonic(
        calculation, wavefunctions, eigenvalues, potential.v_scf,
        frequencies=frequencies, window=window, nw=nw, broadening=broadening,
        scissor=scissor, degeneracy_tol=degeneracy_tol,
        band_cut_gap=band_cut_gap, k_batch=k_batch,
    )
