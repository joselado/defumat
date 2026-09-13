"""The ultracell Kohn-Sham potential, and the subtraction that stops it doubling.

``PLAN.md`` P88. Three things happen here and only the last one is subtle.

**Hartree.** ``4 pi rho(G+Q) / |G+Q|^2`` over the ultracell box, which is
:meth:`~defumat.ultracell.grid.Ultracell.g2`. The ``G = 0, Q != 0`` elements are
**finite and kept**: they are the terms that screen a long-wavelength external
potential, which is the physics the whole method exists for, so dropping them
the way a zone-centre calculation drops ``G = 0`` would remove the answer. Only
``G = Q = 0`` goes, against the compensating background, exactly as
``hartree`` does in the unit cell and as ``gengclqu`` does in Elk
(``gclq(1) = 0``, and ``gengclgq`` then puts ``gclq(iq)`` into the ``G = 0``
element of every other ``Q``).

``q0cut`` (Elk's manual 5.100) is not implemented, and if it ever is, implement
the **code** and not the manual: the manual says the Green's function is zeroed
for every ``|G+Q| < q0cut``, while ``gengclqu.f90`` applies the cut-off only to
``gclq``, the ``G = 0`` element of each ``Q``. The two differ.

**Exchange-correlation.** Evaluated pointwise on the ultracell real-space grid.
For an LDA that is *exact* -- the functional is local, so the ultracell
functional is the unit-cell one evaluated at the ultracell density -- and it is
a strict improvement on Elk's ``potxcu``, which calls ``potxc`` once per cell
``R`` on that cell's own density. The difference is invisible for an LDA and is
not for a GGA: the gradient taken inside one cell misses the envelope's own
gradient, an error of order ``Q`` in ``grad rho``. A GGA is therefore refused
here by name rather than inherited silently.

**The subtraction, which is the first thing to get wrong.** The ultracell
Hamiltonian's diagonal is the *unit cell's* eigenvalue ``eps_{k+Q,n}``, and
that eigenvalue already contains the whole unit-cell potential -- the local
pseudopotential, the Hartree term and the exchange-correlation term at the
unmodulated density. So the potential that goes into the matrix is only the
*difference*,

    dV(r) = v_scf[rho_ULR](r) - tile(v_scf[rho_0](r)) + V_ext(r),

which is Elk's ``vblocalu``. ``vltot`` cancels identically between the two,
since the atoms do not move, so it never appears. Put the whole potential in
instead of the difference and every eigenvalue is doubled in its potential
part, which converges perfectly well to the wrong answer.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from defumat.scf.potential import as_potential_components, exchange_correlation
from defumat.system.cell import Cell
from defumat.ultracell.grid import Ultracell
from defumat.units import E2, FPI

__all__ = [
    "UltracellPotential",
    "ultracell_potential",
    "delta_potential",
    "with_external_potential",
    "require_an_ultracell_functional",
]


class UltracellPotential:
    """Hartree plus exchange-correlation on the ultracell box, and its pieces."""

    __slots__ = ("v_scf", "ehart", "etxc")

    def __init__(self, v_scf, ehart, etxc):
        self.v_scf = v_scf
        self.ehart = ehart
        self.etxc = etxc


def require_an_ultracell_functional(functional) -> None:
    """Refuse what the box's pointwise exchange-correlation cannot represent."""
    if functional.is_meta:
        raise NotImplementedError(
            f"the ultracell refuses the meta-GGA {functional.name}: a "
            "potential-only functional has no energy to be self-consistent in "
            "per cell, and tau is a property of the states rather than of the "
            "density the loop carries"
        )
    if functional.is_gradient:
        raise NotImplementedError(
            f"the ultracell refuses the gradient-corrected functional "
            f"{functional.name} (PLAN.md P88, stage 1 is LDA). The gradient of "
            "the ultracell density is not the gradient of any one cell's "
            "density -- it carries the envelope's own gradient, an error of "
            "order Q that Elk's potxcu takes silently by calling potxc once per "
            "cell. Use an LDA dataset, or input_dft = 'LDA'"
        )


def ultracell_potential(
    rho_r: jnp.ndarray,
    ultracell: Ultracell,
    cell: Cell,
    rho_core_tiled: jnp.ndarray | None,
    functional,
    g2_inverse: jnp.ndarray,
    keep: jnp.ndarray,
) -> UltracellPotential:
    """``v_of_rho`` on the ultracell box.

    Args:
        rho_r: ``(nspin, *box)`` real density, normalised per unit cell -- so
            its integral over the ultracell is ``N`` times the electron count.
        g2_inverse: ``1/|G+Q|^2`` on the box, zero at ``G = Q = 0``, already
            masked to the tiled dense sphere. Precomputed because it is fixed
            for the whole run.
        keep: the same mask as a boolean, for the density.

    The density is masked to the tiled dense sphere before the Hartree term and
    the exchange-correlation term is **not** masked, which is what the unit-cell
    path does and therefore what the ``Q = 0`` subtraction needs: masking one
    side and not the other would leave a residual ``dV`` at the null.
    """
    points = ultracell.points
    volume = ultracell.volume(cell)
    nspin = rho_r.shape[0]

    charge = rho_r[0] if nspin == 4 else jnp.sum(rho_r, axis=0)
    rho_g = jnp.fft.fftn(charge, axes=(-3, -2, -1)) / points
    rho_g = jnp.where(keep, rho_g, 0.0)

    v_g = E2 * FPI * rho_g * g2_inverse
    # ``Re(conj(rho) rho)`` rather than ``|rho|^2``, for the reason
    # ``scf.potential.hartree`` gives: a structure factor vanishing *exactly* is
    # what symmetry arranges, and ``abs`` has no derivative there.
    ehart = 0.5 * volume * E2 * FPI * jnp.sum(
        jnp.real(jnp.conj(rho_g) * rho_g) * g2_inverse
    )
    v_hartree = jnp.real(jnp.fft.ifftn(v_g, axes=(-3, -2, -1)) * points)

    v_xc, etxc = exchange_correlation(rho_r, cell, rho_core_tiled, functional)
    # ``exchange_correlation`` scales its energy by ``cell.volume / n`` with
    # ``n`` the number of points it was handed, which is the ultracell's. The
    # element of volume is the same on both grids, so what it applied is the
    # unit cell's share of the integral and the ultracell's is ``N`` times it.
    etxc = etxc * ultracell.cells

    return UltracellPotential(
        v_scf=as_potential_components(v_hartree, nspin) + v_xc,
        ehart=ehart,
        etxc=etxc,
    )


def delta_potential(potential_box, potential_cell, ultracell: Ultracell, external=None):
    """``dV`` for the Hamiltonian: the ultracell potential minus the unit cell's.

    ``potential_cell`` is ``(nspin, *cell grid)``, the converged unit-cell
    ``v_scf`` the frozen eigenvalues already carry; ``external`` is an optional
    ``(*box,)`` or ``(nspin, *box)`` real field added on top, which is what an
    applied modulation is.
    """
    delta = potential_box - ultracell.tile(potential_cell)
    if external is None:
        return delta
    external = jnp.asarray(external)
    if external.ndim == 3:
        external = jnp.broadcast_to(external, delta.shape)
    return delta + external


def with_external_potential(calculation, v_ext_r):
    """A copy of ``calculation`` whose local potential carries ``v_ext_r``.

    The hook the ultracell's *reference* calculation needs and nothing else
    does: validating an ultracell run means doing the same physics in a real
    ``N``-cell supercell, and the modulation has to reach the supercell's SCF
    somehow. An external scalar potential belongs in ``vltot`` -- it is felt by
    both spin channels in full and it is not a function of the density, which
    is exactly what ``vltot`` already is.

    ``v_ext_r`` is on that calculation's own dense grid, in Ry.

    **Shallow, and deliberately so.** The copy shares every array with the
    original; only ``vltot`` is replaced. Anything that rebuilds ``vltot`` --
    ``at_positions``, ``at_cell``, ``at_strain`` -- drops the external
    potential, which is correct for a moving geometry and is why this is a
    helper here rather than a field on :class:`~defumat.scf.driver.Calculation`.
    """
    import copy

    v_ext_r = jnp.asarray(v_ext_r)
    if v_ext_r.shape != tuple(calculation.basis.dense.grid):
        raise ValueError(
            f"the external potential is on {v_ext_r.shape} and this "
            f"calculation's dense grid is {tuple(calculation.basis.dense.grid)}"
        )
    moved = copy.copy(calculation)
    moved.vltot = calculation.vltot + jnp.real(v_ext_r)
    return moved


def hartree_kernel(ultracell: Ultracell, cell: Cell, keep: np.ndarray):
    """``1/|G+Q|^2`` on the box, zero at ``G = Q = 0`` and outside ``keep``."""
    g2 = ultracell.g2(cell)
    inverse = np.where(g2 > 1e-12, 1.0 / np.where(g2 > 1e-12, g2, 1.0), 0.0)
    return jnp.asarray(np.where(keep, inverse, 0.0), dtype=ultracell.precision.real)
