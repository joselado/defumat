"""``run_conductivity``: the optical conductivity from a converged density.

``PLAN.md`` P51. One fixed-density run with empty states, then the
Kubo-Greenwood sum of :mod:`defumat.response.conductivity`. It is the same
shape as :func:`~defumat.workflows.tddft.run_absorption` and for the same
reason -- a sum over states needs the empty ones, and how many is the
convergence parameter of the whole phase.

**The k-set is the thing to think about**, and it is a different question from
the one a ground state answers. An optical spectrum is an integral over the
Brillouin zone of a quantity with a sharp frequency dependence, so it wants a
grid far denser than the density needed; and the antisymmetric part is an axial
vector, so it wants the **whole** grid rather than a wedge. ``kpoints`` is the
argument that lets both be true at once: converge the density where it is
cheap, evaluate the conductivity where it is right.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from defumat.response.conductivity import (
    OpticalConductivity,
    optical_conductivity,
    require_a_conductivity_regime,
)
from defumat.workflows.nscf import fixed_density_states

__all__ = ["run_conductivity"]


def run_conductivity(
    system,
    pseudos,
    density,
    *,
    kpoints=None,
    nbnd: int | None = None,
    frequencies=None,
    window: float = 1.5,
    nw: int = 300,
    broadening: float = 0.01,
    relaxation: float | None = None,
    intraband: bool = True,
    scissor: float = 0.0,
    method: str = "frequency",
    becsum: tuple = (),
    ns=None,
    tau=None,
    field=None,
    field_scale=None,
    fermi_energy: float | None = None,
    conv_thr: float = 1.0e-10,
    k_batch="default",
) -> OpticalConductivity:
    """``sigma_ab(omega)`` for a converged run.

    Args:
        system: the converged :class:`~defumat.system.builder.System`.
        pseudos: its pseudopotentials.
        density: the converged density (``SCFResult.density``).
        kpoints: the k-set the conductivity is evaluated on, if it is not the
            one the density converged on. It must be the **whole** grid; see
            :func:`~defumat.response.conductivity.require_a_conductivity_regime`.
        nbnd: how many bands the fixed-density run resolves. Defaults to three
            times the occupied count. It is the truncation the f-sum rule
            measures and is not a knob to raise until a test passes.
        fermi_energy: in Ry. Defaults to the level the fixed-density run finds
            for its own band set, which is the right one whenever the k-set is
            the ground state's; pass the SCF's own when it is not.

    The remaining arguments are
    :func:`~defumat.response.conductivity.optical_conductivity`'s.
    """
    from defumat.scf.driver import Calculation

    # The refusals are checked **before** the fixed-density run, as
    # ``run_absorption`` checks its own: they are statements about the
    # calculation, and a caller asking for something this cannot do should not
    # first pay for three times the bands at every k-point.
    if kpoints is not None:
        import equinox as eqx

        from defumat.system.kpoints import for_spin

        # **``for_spin`` has to be applied here**, and this boundary is exactly
        # the one its docstring warns about: every ``KPoints`` constructor
        # applies the unpolarized factor of two unconditionally, and a spinor
        # band holds *one* electron rather than two. A caller who builds a
        # denser mesh with ``KPoints.automatic`` and hands it over gets weights
        # summing to 2 where the run needs 1, which does not look like an
        # error -- the Fermi level simply lands somewhere else. Measured on fcc
        # nickel with spin-orbit coupling, that put the plasma frequency at
        # 13.11 eV instead of 0.60 and flipped the sign of the anomalous Hall
        # conductivity, on the same density and the same 64 k-points. It is
        # idempotent, so a correctly scaled set passes through untouched.
        system = eqx.tree_at(
            lambda s: s.kpoints, system, for_spin(kpoints, system.nspin)
        )
    require_a_conductivity_regime(Calculation(system, pseudos, k_batch=k_batch))

    # **One band more than the sum uses**, and it is not an accident of
    # rounding. Where the truncation falls is the difference between an
    # antisymmetric residue of 4e-13 and one of 2e-6 on a crystal whose
    # antisymmetric conductivity is zero by time reversal -- cutting *inside* a
    # degenerate multiplet keeps some of its members and drops others, and the
    # cancellation they were making between them does not happen. The gap that
    # says which of the two happened is between the last band kept and the
    # first dropped, so it cannot be measured from the sum's own band set. One
    # extra band buys it, and it is one band out of dozens.
    nbnd = int(nbnd or _default_nbnd(system, pseudos))
    calculation, system, eigenvalues, wavefunctions = fixed_density_states(
        system, pseudos, density, nbnd=nbnd + 1,
        conv_thr=conv_thr, k_batch=k_batch, ns=ns, becsum=becsum, tau=tau,
        field=field, field_scale=field_scale,
    )
    eigenvalues = jnp.asarray(eigenvalues)
    if eigenvalues.ndim == 2:
        eigenvalues = eigenvalues[None]
    band_cut_gap = float(
        np.min(np.asarray(eigenvalues)[..., nbnd] -
               np.asarray(eigenvalues)[..., nbnd - 1])
    )
    eigenvalues = eigenvalues[..., :nbnd]
    wavefunctions = jnp.asarray(wavefunctions)[..., :nbnd, :]
    potential = calculation.potential(jnp.asarray(density))
    _, ddd_paw = calculation.onecenter(becsum)

    if fermi_energy is None:
        fermi_energy = _fermi_level(calculation, eigenvalues)

    return optical_conductivity(
        calculation, wavefunctions, eigenvalues, potential.v_scf,
        fermi_energy=fermi_energy, frequencies=frequencies, window=window,
        nw=nw, broadening=broadening, relaxation=relaxation,
        intraband=intraband, scissor=scissor, method=method,
        ddd_paw=ddd_paw, ns=ns, band_cut_gap=band_cut_gap, k_batch=k_batch,
    )


def _default_nbnd(system, pseudos) -> int:
    """Three times the occupied count, which is a starting point and not a choice.

    **A spinor band holds one electron and an unpolarized band holds two**, so
    the occupied count is ``nelec`` for ``noncolin`` and ``nelec/2`` otherwise
    -- the same rule ``Calculation.occupations`` calls ``degeneracy``. Getting
    it wrong on a spinor run asks for half the bands and truncates the sum
    silently.
    """
    from defumat.scf.driver import Calculation

    calculation = Calculation(system, pseudos)
    degeneracy = 1 if calculation.noncolin else 2
    return max(4, int(np.ceil(3.0 * calculation.nelec / degeneracy)))


def _fermi_level(calculation, eigenvalues) -> float:
    """The level the fixed-density band set implies, whatever the scheme.

    ``Calculation.occupations`` returns it beside the weights for a smeared
    run and returns the HOMO for a fixed one; the Drude term wants the first
    and does not run at all in the second case, so either is fine here.
    """
    eigenvalues = jnp.asarray(eigenvalues)
    if eigenvalues.ndim == 2:
        eigenvalues = eigenvalues[None]
    _, statistic = calculation.occupations(eigenvalues)
    for key in ("fermi_energy", "homo"):
        value = statistic.get(key) if isinstance(statistic, dict) else None
        if value is not None:
            return float(value)
    return 0.0
