"""The source-free exchange-correlation magnetic field: Elk's ``nosource``.

**What the local spin-density approximation cannot do, and why that matters.**
At every point of the grid the functional is handed ``n`` and ``|m|``, evaluates
the collinear expression there, and attaches the splitting to the *direction* of
``m`` -- which is :func:`defumat.scf.potential._noncollinear_xc` and it says so
itself. So ``B_xc(r)`` is parallel to ``m(r)`` at every point, by construction,
and the torque the field exerts on the magnetization,

    tau = int m(r) x B_xc(r) d3r,

is identically zero at every density, converged or not. A functional of ``|m|``
alone cannot turn a moment, which is the known limitation of the approximation
rather than an artefact of this implementation, and it is the reason a
noncollinear texture here is held only by the *shape* of the density and never
by a local field.

**What this module does about it.** A magnetic field in nature has no sources:
``div B = 0``. The exchange-correlation field built pointwise from ``m`` does
not obey that, so Sharma, Dewhurst, Sanna and Gross project the longitudinal
part out of it (Elk's ``src/projsbf.f90``, reached by ``nosource``). In real
space Elk takes the divergence, solves a Poisson equation for it and adds the
gradient of the solution back; on a plane-wave grid the same operation is one
line in reciprocal space,

    B(G) -> B(G) - G (G . B(G)) / |G|^2,

with ``G = 0`` left alone, since a Poisson equation cannot move a constant and a
uniform field is divergence-free already. The projected field is **not** parallel
to ``m`` any more, so the torque above stops being zero and the functional gains
the one thing the local approximation threw away.

**It is a potential with no energy functional behind it**, which is the same
position :mod:`defumat.xc.mgga`'s ``tb09`` is in and it has the same
consequences rather than a fresh set: nothing is differentiated to get it, the
total energy a run reports is not the value of anything the run minimised, and
every consumer that differentiates the energy -- forces, stress, the whole
response stack -- refuses by name. ``run_scf`` warns, exactly as it does for a
potential-only meta-GGA.

**The projection removes the longitudinal part rather than rebuilding the
field.** Writing it as ``B - (the longitudinal part)`` and not as "transform to
the sphere, project, transform back" matters: the second form band-limits the
whole of ``B_xc`` to the dense G sphere, which changes the potential even when
there is nothing longitudinal to remove, and that change would be silent. This
way a field that is already divergence-free comes back untouched to round-off,
and the sphere's truncation reaches only the part being subtracted.

**Refused, and each for its own reason** (:func:`refuse_source_free`):

* a collinear run, ``nspin = 2``. The transverse projection of a field along
  ``z`` has ``x`` and ``y`` components, which a collinear density has nowhere to
  put; Elk refuses it the same way and for the same reason
  (``init0.f90:184-186`` requires ``ncmag``).
* **PAW**, and the reason is the projection rather than a missing term: it is
  a *nonlocal* operator, so applying it to the smooth field and leaving the two
  one-centre fields alone is not applying it to their sum. What a source-free
  PAW field would need is one Poisson equation solved across both
  representations, the sphere solution's multipoles matched to the smooth one
  outside -- the compensation-charge problem :mod:`defumat.paw.hartree` already
  solves for the density. Elk's own projection is one line because its muffin
  tins and its interstitial partition space, where PAW's three terms overlap.
* a **spin spiral**, because ``div B`` is a statement in the laboratory frame
  and the magnetization is stored in the rotating one.

**Elk's other half is not built.** ``sxcscf`` (``tssxc``) scales the spin part
of the exchange-correlation field by a constant, which is the same paper's
second knob and a different approximation; it is not here, and asking for it is
not the same as asking for this.
"""

from __future__ import annotations

import jax.numpy as jnp

from defumat.basis.fft import g_to_r, r_to_g
from defumat.basis.gvectors import GVectors
from defumat.system.cell import Cell

__all__ = ["project_source_free", "longitudinal_field", "refuse_source_free"]


def longitudinal_field(
    field_r: jnp.ndarray, gvectors: GVectors, cell: Cell
) -> jnp.ndarray:
    """``G (G . B(G)) / |G|^2`` back on the grid: what ``nosource`` removes.

    Args:
        field_r: ``(3, n1, n2, n3)`` real -- the vector part of the
            exchange-correlation potential.

    Returns the same shape. The ``G = 0`` component is excluded with a mask
    rather than by dividing and repairing afterwards: ``|G|^2`` is exactly zero
    there, so the division is ``0/0`` and its *tangent* is a nan that survives
    into every derivative taken through the potential -- the trap
    ``CLAUDE.md`` lists first and this is one more site of it.
    """
    g = gvectors.cartesian(cell)  # (ngm, 3), 1/bohr
    # ``|G|^2`` from the same array the numerator uses, and **not**
    # ``GVectors.g2``, which is QE's ``gg`` and is in units of ``tpiba^2``: the
    # two differ by ``(2 pi / alat)^2``, so mixing them scales the projection by
    # a number that depends on the lattice constant and removes a fraction of
    # the longitudinal part rather than all of it. Measured on a 10 bohr cube
    # before it was fixed: ``div B`` fell by a factor of 2.7 where it has to
    # fall to round-off, and nothing but a unit test on a pure gradient saw it.
    g2 = jnp.sum(g * g, axis=-1)  # (ngm,), exactly zero at G = 0
    field_g = r_to_g(field_r, gvectors.fft_index)  # (3, ngm)
    projection = jnp.sum(g.T * field_g, axis=0)  # (ngm,), G . B(G)
    safe = jnp.where(g2 > 0.0, g2, 1.0)
    weight = jnp.where(g2 > 0.0, projection / safe, 0.0)
    return jnp.real(g_to_r(g.T * weight[None, :], gvectors.fft_index, gvectors.grid))


def project_source_free(
    v_xc: jnp.ndarray, gvectors: GVectors, cell: Cell
) -> jnp.ndarray:
    """``projsbf``: the exchange-correlation potential with ``div B_xc = 0``.

    Args:
        v_xc: ``(4, n1, n2, n3)`` -- ``(v_0, B_x, B_y, B_z)``, the noncollinear
            exchange-correlation potential.

    The scalar component is untouched: only the field has a divergence to
    remove.
    """
    if v_xc.shape[0] != 4:
        raise ValueError(
            "a source-free exchange-correlation field needs the four-component "
            f"noncollinear potential and this one has {v_xc.shape[0]} "
            "components; refuse_source_free is where that is caught with a "
            "reason"
        )
    field = jnp.real(v_xc[1:])
    return v_xc.at[1:].add(-longitudinal_field(field, gvectors, cell))


def refuse_source_free(system, functional=None, pseudos=()) -> None:
    """The regimes ``nosource`` cannot be asked for, each by name.

    ``pseudos`` is optional so that a caller with only a system in hand keeps
    working; a PAW dataset is refused when it is given, and
    :class:`~defumat.scf.driver.Calculation` gives it. The refusal used to live
    in the driver and this docstring's own module header claimed it lived here,
    which is the reading a caller of this function got wrong.
    """
    if any(getattr(pseudo, "is_paw", False) for pseudo in pseudos or ()):
        raise NotImplementedError(
            "a source-free exchange-correlation field with a PAW dataset is "
            "not implemented, and what is missing is the projection rather "
            "than a term in it. The projection is P = 1 - grad (lap)^-1 div, "
            "which is nonlocal: (lap)^-1 couples every point of the cell, so "
            "P applied to a sum is not the sum of P applied to each part. A "
            "PAW field is a sum of three parts in two representations -- the "
            "smooth one on the plane-wave grid and, inside every sphere, the "
            "all-electron and pseudo one-centre fields -- and running the "
            "reciprocal-space line on the first of them alone is a projection "
            "of nothing. Doing it properly means solving one Poisson equation "
            "for div B across both representations, with the multipoles of "
            "the sphere solution matched to the smooth one outside, which is "
            "the compensation-charge problem PAW's Hartree term already "
            "solves for the density (defumat.paw.hartree). Elk gets it in one "
            "line because its muffin tins and its interstitial *partition* "
            "space, where PAW's three terms overlap. Use a norm-conserving or "
            "ultrasoft dataset"
        )
    if not bool(getattr(system, "noncolin", False)):
        raise NotImplementedError(
            "a source-free exchange-correlation field needs a noncollinear "
            "run: the projection is transverse, so it gives a field along z "
            "components along x and y that a collinear density has nowhere to "
            "carry. Elk requires the same thing (init0.f90 refuses nosource "
            "without ncmag). Run with noncolin = .true."
        )
    if getattr(system, "spiral_q", None) is not None:
        raise NotImplementedError(
            "a source-free exchange-correlation field on a spin spiral is not "
            "implemented: div B_xc is a statement in the laboratory frame and "
            "the magnetization of a spiral is stored in the rotating one, so "
            "the projection would be taken of a field that is not the field"
        )
    if functional is not None and getattr(functional, "is_meta", False):
        raise NotImplementedError(
            "a source-free exchange-correlation field with a potential-only "
            "meta-GGA is not implemented: both are potentials with no energy "
            "functional behind them and stacking two of them leaves nothing "
            "that says what the run is a stationary point of"
        )
