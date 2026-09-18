"""The exchange-correlation spin torque, ``tau = int m(r) x B_xc(r) d3r``.

Elk's task 160 (``src/torque.f90``, which is 37 lines because both fields
already exist there too). It measures how far a magnetic configuration is from
being stationary under the exchange-correlation field: a converged state has
nothing left to turn its moments, so the torque vanishes, and a constrained or
non-self-consistent one does not. It is the quantity the adiabatic picture of a
spin wave rests on, and the one an atomistic spin dynamics is driven by.

**Read this before using it as a check, because on its own it is not one.**
In the local spin-density approximation, and in every gradient-corrected
functional built on top of it, ``B_xc(r)`` is parallel to ``m(r)`` at every
point of the grid -- the functional is handed ``|m|`` and attaches its answer to
``m``-hat, which :func:`defumat.scf.potential._noncollinear_xc` states in its
own docstring. The cross product is therefore **identically zero at every
density**, converged or not, constrained or not. A test that asserts this
quantity vanishes at self-consistency passes on a broken code exactly as it
passes on a working one, which is ``CLAUDE.md``'s first trap in its purest form.

**So the torque is a diagnostic of a field that is not parallel to ``m``**, and
there are two of those here:

* a **source-free** exchange-correlation field (:mod:`defumat.scf.sourcefree`,
  Elk's ``nosource``), where the longitudinal part has been projected out and
  what is left is no longer along ``m``. This is the case Elk's own task is for.
* a **Hubbard** term, whose potential is a matrix in the shell rather than a
  local field, and which therefore does turn a moment. That contribution is
  *not* in what this function integrates -- it is not part of ``B_xc`` -- and
  saying so matters, because a DFT+U run whose torque reads zero has not been
  shown to be stationary.

The function therefore returns the field's own torque together with the
magnitude of the field's component along ``m``, so that a zero can be told from
a silence: ``parallel_fraction`` near one means the field is collinear with the
magnetization and the zero is the approximation speaking rather than the state.

**And the *total* is a second null of the same kind, which Elk's own task does
not say.** Without spin-orbit coupling the energy is invariant under a global
rotation of every spin together, so the net torque on the cell is zero by that
symmetry -- identically, at any density, converged or not, and whether or not
the field has been made source-free. So the number to read without spin-orbit
coupling is :attr:`ExchangeTorque.sites`, and the total is a check on the
assembly rather than on the state; with spin-orbit coupling the global rotation
stops being free and the total becomes informative in its own right.

**What that leaves, measured.** On fcc nickel with its moment along a
three-fold axis, one site's magnetization turned about that axis inside its own
sphere -- a configuration neither functional converged to:

====================  ====================  ========================
turned by             local functional      source-free field
====================  ====================  ========================
0 degrees             7e-22 Ry              2e-18 Ry
15 degrees            4e-20 Ry              1.64e-4 Ry
45 degrees            5e-20 Ry              4.47e-4 Ry
90 degrees            1e-21 Ry              6.32e-4 Ry
====================  ====================  ========================

The left column is the point: the local functional reports **zero at every
angle**, not only at the one it converged to.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from defumat.xc.functional import safe_modulus

__all__ = ["ExchangeTorque", "exchange_torque"]


class ExchangeTorque:
    """What ``torque.f90`` prints, plus the two numbers that make it readable.

    Attributes:
        total: ``(3,)`` in Ry -- ``int m x B_xc``, over the whole cell.
        sites: ``(nat, 3)`` in Ry, the same integrand over each atom's sphere
            (:mod:`defumat.scf.locals`), or ``None`` when no region set was
            given. Elk reports only the total; the per-atom split is what says
            *which* moment is being turned.
        parallel_fraction: ``int |m . B_xc| / int |m| |B_xc|``, dimensionless.
            One when the field is everywhere along the magnetization, which is
            what a plain local functional gives and what makes ``total`` zero by
            construction rather than by convergence.
    """

    __slots__ = ("total", "sites", "parallel_fraction")

    def __init__(self, total, sites, parallel_fraction):
        self.total = total
        self.sites = sites
        self.parallel_fraction = parallel_fraction

    def __repr__(self) -> str:
        magnitude = float(np.linalg.norm(np.asarray(self.total)))
        return (
            f"ExchangeTorque(|total| = {magnitude:.6e} Ry, "
            f"parallel_fraction = {float(self.parallel_fraction):.6f})"
        )


def exchange_torque(density, v_xc, cell, regions=None) -> ExchangeTorque:
    """``int m x B_xc`` for a noncollinear density and its potential.

    Args:
        density: ``(4, n1, n2, n3)`` -- ``(n, m_x, m_y, m_z)``.
        v_xc: ``(4, n1, n2, n3)`` -- ``(v_0, B_x, B_y, B_z)``, the
            exchange-correlation potential alone. The Hartree term is blind to
            the magnetization and contributes nothing, but passing the *total*
            potential would fold in a magnetic field if the run carries one,
            which is a different quantity.
        regions: an optional :class:`~defumat.scf.locals.LocalRegions` for the
            per-atom split.
    """
    density = jnp.asarray(density)
    v_xc = jnp.asarray(v_xc)
    if density.shape[0] != 4 or v_xc.shape[0] != 4:
        raise ValueError(
            "the exchange-correlation spin torque is a noncollinear quantity "
            f"and needs four components of each; got {density.shape[0]} of the "
            f"density and {v_xc.shape[0]} of the potential"
        )
    magnetization = jnp.real(density[1:])  # (3, ...)
    field = jnp.real(v_xc[1:])
    cross = jnp.cross(magnetization, field, axis=0)  # (3, ...)

    scale = cell.volume / density[0].size
    total = scale * jnp.sum(cross.reshape(3, -1), axis=1)

    sites = None
    if regions is not None:
        sites = scale * regions.integrate(cross)

    # ``safe_modulus`` and not ``jnp.linalg.norm``: ``|m|`` has a node wherever
    # the magnetization passes through zero, which an antiferromagnet has on
    # every plane between two atoms, and a bare square root is not
    # differentiable there.
    m_norm = safe_modulus(magnetization, axis=0)
    b_norm = safe_modulus(field, axis=0)
    aligned = jnp.sum(jnp.abs(jnp.sum(magnetization * field, axis=0)))
    product = jnp.sum(m_norm * b_norm)
    parallel_fraction = jnp.where(product > 0.0, aligned / jnp.where(
        product > 0.0, product, 1.0), 0.0)
    return ExchangeTorque(total, sites, parallel_fraction)


def torque_of_result(calculation, result, density=None) -> ExchangeTorque:
    """:func:`exchange_torque` for a run, against its own exchange-correlation
    field.

    ``density`` defaults to the run's converged one; passing another is the
    point of the quantity rather than a convenience, since a torque is only
    nonzero for a configuration that is *not* stationary and a converged
    texture on a symmetric cell has none by symmetry.

    **``B_xc`` is taken as the vector part of the self-consistent potential**,
    which is exactly what it is: ``v_of_rho`` puts the Hartree term in the
    charge component alone (``v_h``'s own ``IF (nspin == 4)``) and the local
    pseudopotential is added later by ``set_vrs``, so components 1 to 3 of what
    comes back are the exchange-correlation field and nothing else -- gradient
    correction and source-free projection included. Rebuilding it from the local
    part alone instead would leave a gradient-corrected run reporting the torque
    of a functional it was not run with.

    The *field* the run carries, if it has one, is deliberately left out: it is
    added by ``add_bfield`` after this point, it is an external field rather
    than the exchange-correlation one, and a torque taken against their sum is
    the torque of something else.
    """
    from defumat.scf.potential import v_of_rho

    if density is None:
        # ``SCFResult.density`` keeps its ``(nspin_mag, ...)`` axis at one
        # channel -- it is the *eigenvalue-shaped* quantities and the DOS that
        # are squeezed. Reinstating it here gave a five-dimensional array that
        # rode through ``v_of_rho`` and the transforms without complaint,
        # because both broadcast over leading axes.
        density = result.density
    density = jnp.asarray(density)
    if calculation.functional.is_meta:
        raise NotImplementedError(
            "the exchange-correlation spin torque of a potential-only "
            "meta-GGA is not implemented: its potential needs tau, which comes "
            "from the states rather than from the density, and a torque "
            "evaluated at a density alone would be of a different potential"
        )
    potential = v_of_rho(
        density,
        calculation.basis.dense,
        calculation.system.cell,
        calculation.rho_core,
        calculation.functional,
        calculation.rho_core_g,
        calculation.quantization_axis,
        None,
        calculation.source_free,
    )
    return exchange_torque(
        density, potential.v_scf, calculation.system.cell,
        calculation.local_regions(),
    )
