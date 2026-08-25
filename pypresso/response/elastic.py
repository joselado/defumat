"""Elastic constants: the second derivative of the energy with respect to strain.

``PLAN.md`` P26. ``C_ijkl = (1/Omega) d^2E/dx_ij dx_kl``, and it is
:mod:`pypresso.response.phonon`'s construction with the *cell* in place of the
atoms -- the same sentence one coordinate over:

    the **stress** is ``jax.grad`` of the frozen-state energy with respect to a
    strain (P11); the **elastic constants** are that gradient differentiated
    once more, along a tangent that carries the strain, the states and the
    density together.

    C[:, :, k, l] = (1/Omega) jvp( grad_x E )( x, psi, rho ; T_kl, dpsi_kl, drho_kl )

so nothing is derived: the ``T_kl`` half is the frozen second derivative -- QE
has no counterpart at all, ``pw.x`` computing no elastic constants -- and the
``dpsi``/``drho`` half is the electronic relaxation, which comes from
:mod:`pypresso.response.strain`. The two are components of one tangent vector,
exactly as ``dynmat0`` and ``drhodv`` are for a phonon.

**These are the clamped-ion constants**, and the distinction is not small for
``C_44``. A strain of a crystal with more than one atom in the cell relaxes the
internal coordinates as well, and the correction is
``C^relaxed = C - Lambda^T Phi^-1 Lambda`` with ``Phi`` the ``Gamma``-point force
constants (P25 has them) and ``Lambda_ij,ak = d^2E/dx_ij du_ak`` the internal-strain
tensor -- one more mixed ``jvp`` of the same two families, and not written here.
For diamond silicon ``C_11`` and ``C_12`` are unaffected by symmetry (no
displacement is compatible with a tetragonal strain) and ``C_44`` is: the
relaxation is what takes the clamped-ion value down to the measured one, so
``C_44`` here is expected to sit *above* experiment's 79.6 GPa and is not a
disagreement. ``C_11`` and ``C_12`` are the ones to compare.

**The Pulay error is the stress's**, one order up: the plane-wave sphere is
frozen while differentiating (:meth:`~pypresso.scf.driver.Calculation.at_strain`),
so what is missing is the jump at the strains where a plane wave crosses the
cutoff. That makes the *comparison* to run a finite difference that freezes the
sphere too -- which is what
:meth:`~pypresso.scf.driver.Calculation.at_strain` does, so re-running the SCF on
a strained calculation gives a reference on the same footing. Measured that way,
``C_11`` on two-atom silicon at ``ecutwfc = 12`` is **209.38 GPa** from one
``jvp`` and **209.38** from a five-point second difference of the energy, against
a measured 166 -- and the cutoff dependence of the difference is flat, so the
residual is the functional and the pseudopotential rather than the basis.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from pypresso.forces.energy import FrozenState, energy_at
from pypresso.response.strain import StrainResponse, strain_tangent
from pypresso.units import RY_TO_KBAR

__all__ = ["ElasticConstants", "elastic_constants", "VOIGT"]

#: The six independent components of a symmetric strain, in Voigt order.
VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))

#: Ry/bohr^3 to GPa. ``RY_TO_KBAR`` is what the stress is printed in.
RY_TO_GPA = RY_TO_KBAR / 10.0


@dataclass
class ElasticConstants:
    """``C_ijkl`` and the compliance that inverts it.

    **Clamped-ion**, which for ``C_11`` and ``C_12`` in the diamond structure is
    the whole answer by symmetry and for ``C_44`` is not: the internal relaxation
    that is missing is what brings ``C_44`` down toward the measured value, so it
    is expected high here. ``M`` and ``Q`` in
    :class:`~pypresso.response.electrostriction.Electrostriction` are built from
    the compliance and inherit that; ``m``, ``q`` and the elasto-optic tensor do
    not.
    """

    #: ``(3, 3, 3, 3)`` in **Ry/bohr^3**, clamped-ion.
    tensor: np.ndarray
    #: ``(6, 6)`` in **GPa**. ``C_IJ = C_ijkl`` with no factors of two: Voigt's
    #: doubling lives in the engineering strain, so ``sigma_I = C_IJ e_J`` holds
    #: with ``e_4 = 2 x_23`` and ``C_44 = C_2323``.
    voigt: np.ndarray

    @property
    def compliance(self) -> np.ndarray:
        """``S = C^-1`` as ``(6, 6)`` in 1/GPa, the inverse in Voigt form."""
        return np.linalg.inv(self.voigt)

    @property
    def bulk_modulus(self) -> float:
        """``B = (C_11 + 2 C_12)/3`` for a cubic crystal, in GPa.

        Written as ``1/(S_11 + 2 S_12 + ...)`` so that it is the *hydrostatic*
        modulus of whatever symmetry the crystal has: ``B = 1 / sum_IJ S_IJ``
        over the ``3x3`` block, which reduces to the cubic expression.
        """
        return float(1.0 / self.compliance[:3, :3].sum())


def elastic_constants(
    calculation, wavefunctions, eigenvalues, density, response: StrainResponse,
    allow_unconverged: bool = False,
) -> ElasticConstants:
    """``C_ijkl`` from one ``jvp`` of the stress per independent strain.

    Args:
        calculation: the converged run's calculation.
        wavefunctions: ``(nspin, nk, nbnd, npwx)`` -- **all** bands, since the
            energy functional sums over all of them.
        eigenvalues: ``(nspin, nk, nbnd)`` or the squeezed shape.
        density: the converged density.
        response: the strain response, whose ``dpsi`` and ``drho`` are the
            tangent's other two components.

    **The density is rebuilt inside, and this is where a strain parts company
    with a displacement.** :func:`~pypresso.response.phonon._force_constants`
    hands the density in as an independent argument, because the functional
    symmetrises its own as a *scalar* -- right for a ground state, wrong for a
    response. Doing the same here is **wrong by a factor of three**, and the
    reason is that ``jax.grad`` of the functional at a *fixed density array* is
    not the stress. The stress is the total derivative, and

        dE/dx = d_x E|_rho + (dE/d rho) . (d rho/dx)|_psi

    where the second term vanishes for a displacement -- moving an atom does not
    change ``sum_n w |psi_n(r)|^2`` at frozen coefficients -- and does not vanish
    for a strain, because the density carries a ``1/Omega``
    (:mod:`pypresso.response.strain`). Letting the functional build its own
    density puts that term back through the chain rule, and then the tangent is
    ``(strain, dpsi)`` with no density component at all. Measured: 671 GPa the
    wrong way against 209 the right way, on silicon whose ``C_11`` is 166.

    **The price is that a symmetry-reduced k-set is refused**, since the
    functional's scalar symmetrisation would then be in the chain rule. That is
    the condition :mod:`pypresso.response.electrostriction` imposes anyway.
    """
    if not allow_unconverged and response is not None and not response.converged:
        # The same refusal :func:`pypresso.response.electrostriction.
        # require_converged_responses` makes, stated here as well because this
        # function is a public entry point of its own. What it caught: a
        # diverged strain response gives a ``C_ijkl`` that is not symmetric
        # under ``C_ijkl = C_klij``, which no amount of reading the numbers
        # reveals -- 49817 GPa against -243233 for the same index pair.
        last = response.history[-1] if response.history else float("nan")
        raise ValueError(
            f"the strain response did not converge (|ddv_scf|^2 = {last:.3e} "
            f"after {len(response.history)} iterations, against the requested "
            "tr2); the elastic constants built on it would not even be "
            "symmetric under C_ijkl = C_klij. If this run set "
            "mixing_mode='linear', try the default Anderson mixer "
            "(pypresso.response.mixing); otherwise raise max_iterations or "
            "lower alpha_mix"
        )
    psi = jnp.asarray(wavefunctions)
    eigenvalues = jnp.asarray(eigenvalues)
    if eigenvalues.ndim == 2:
        eigenvalues = eigenvalues[None]
    _require_a_closed_grid(calculation)
    weights, _ = calculation.occupations(eigenvalues)
    weights = jnp.asarray(weights)
    nocc = np.asarray(response.dpsi[0, 0]).shape[2]
    zero = jnp.zeros((3, 3))

    def energy(strain, states):
        return energy_at(
            calculation.at_strain(strain),
            FrozenState(
                wavefunctions=states, weights=weights, eigenvalues=eigenvalues
            ),
        )

    gradient = jax.grad(energy, argnums=0)
    volume = calculation.system.cell.volume

    tensor = np.zeros((3, 3, 3, 3))
    for (k, l) in VOIGT:
        # ``dpsi`` carries the occupied bands only, which is all the Sternheimer
        # equation solves for; the empty ones have zero weight in every term of
        # the functional, so padding with zeros is exact rather than an
        # approximation. P25's rule, verbatim.
        states = jnp.zeros_like(psi).at[:, :, :nocc].set(
            jnp.asarray(response.dpsi[k, l])
        )
        _, column = jax.jvp(
            gradient, (zero, psi), (strain_tangent(k, l), states)
        )
        column = np.asarray(column) / volume
        # The gradient's own two indices are the *full-matrix* ones; the tensor
        # convention wants them symmetrised as the ``(k, l)`` pair already is.
        column = 0.5 * (column + column.T)
        tensor[:, :, k, l] = tensor[:, :, l, k] = column

    voigt = np.array([
        [tensor[i][j][k][l] for (k, l) in VOIGT] for (i, j) in VOIGT
    ]) * RY_TO_GPA
    return ElasticConstants(tensor=tensor, voigt=voigt)


def _require_a_closed_grid(calculation) -> None:
    """A wedge is refused, for the reason :func:`elastic_constants` documents."""
    if getattr(calculation, "_symmetry_maps", None) is not None:
        raise NotImplementedError(
            "elastic constants on a symmetry-reduced k-set are not implemented: "
            "the energy functional has to build its own density here (so that "
            "its gradient is the stress and not a partial derivative at fixed "
            "rho), and it symmetrises that density as a *scalar*, which a "
            "response must not go through. Run the whole grid instead -- an "
            "unshifted Monkhorst-Pack grid with nosym is closed under the point "
            "group. This is the one refusal a rank-4 average does not lift: "
            "pypresso.response.electrostriction's own half runs on a wedge now, "
            "and takes elastic=False to leave this one out"
        )
