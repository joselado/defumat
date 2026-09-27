"""Spin-orbit coupling on a spin spiral, to first order: the Dzyaloshinskii-Moriya energy.

A spiral is exact in the unit cell only without spin-orbit coupling
(:mod:`defumat.system.spiral`): translating by ``R`` and turning every spin by
``q . R`` is then a symmetry, and the state at ``k`` has its up component at
``k + q/2`` and its down one at ``k - q/2``. The coupling ties the spin to the
lattice and breaks that symmetry, so the spiral with the coupling is not a
single-``q`` state. What survives is perturbation theory around the
coupling-free spiral, and its first order is the subject here.

**The zeroth order is the same fully-relativistic dataset at ``soc_scale =
0``**, which is the only form in which a first-order operator exists at all:
``soc_scale`` blends the coupling's parts of ``dvan_so`` linearly
(:class:`~defumat.pseudo.spinorbit.SpinOrbitCoupling`), so the operator is the
coupled nonlocal coefficients minus the reduced ones, ``dD = dvan_so(1) -
dvan_so(0)``, and the first-order energy is ``dF/d(soc_scale)`` at zero by
Hellmann-Feynman at frozen occupations. A scalar-relativistic partner file has a
different ``D`` and different projectors, so its difference from the relativistic
file is not a perturbation of anything. At ``soc_scale = 0`` ``dvan_so`` is its
own spin trace, the Hamiltonian commutes with every spin rotation, and the
generalized Bloch theorem holds exactly (``system/builder.py:_spiral_q`` admits
the pair there and nowhere else).

**Which part of the operator has a first-order energy.** Write ``dD = sum_a
dD^a sigma_a`` with ``a = x, y, z`` (its spin trace is zero, because
``soc_scale`` scales exactly the traceless half) and take the spiral's rotation
axis, the spin direction of its up component, along ``n``. In the frame where
``n`` is the quantization axis the operator splits into a part along
``sigma_n`` and a transverse part along ``sigma_+-``, and the two do different
things to a spiral state, because ``dD`` is the *same* matrix in every cell
while the spins turn from one cell to the next:

* the ``sigma_n`` part, ``(n . dD) sigma_n``, keeps each component on its own
  sphere (the up one at ``k + q/2``, the down one at ``k - q/2``), so it connects
  the state at ``k`` with the states at ``k``, and it alone has an expectation
  value;
* ``sigma_+`` turns a down component at ``k - q/2`` into an up one at the same
  wavevector, which is the up component of a state at ``k - q``, so the
  transverse part connects ``k`` with ``k -+ q``. Summed over the cells its
  expectation value is ``sum_R exp(-i q . R) = 0`` for any ``q`` outside the
  reciprocal lattice. It is what first-order perturbation theory mixes in, and
  it has no first-order energy.

So the first-order energy is

    E1(n) = sum_k w_k sum_n f_nk [ <u^up_nk| n.dD |u^up_nk>_(k+q/2)
                                   - <u^dn_nk| n.dD |u^dn_nk>_(k-q/2) ]
          = n . V(q),

each component projected on the projectors of its own sphere, and **it is
linear in the axis**: one vector ``V(q)`` gives it for every orientation of the
spiral plane, because at zeroth order turning the whole spiral rigidly in spin
space costs nothing and leaves ``V`` unchanged. That is why the run's own axis
(``z``, the one the spiral code turns the moment about) is no restriction.

**Two exact properties follow and are what the tests hold it to.** The spiral
at ``(q, n)`` is the same texture as the one at ``(-q, -n)``, so ``V(-q) =
-V(q)``: the whole first-order energy is odd in ``q``, it is the chirality of the
spiral, and there is no first-order part even in ``q`` (the anisotropy of a
spiral, like that of a ferromagnet, starts at second order). And inversion
through a site sends ``(q, n)`` to ``(-q, n)``, so ``V(q) = 0`` identically on a
centrosymmetric crystal, which is the null. For a small ``q``, ``V(q) = D q`` to
leading order, and ``D`` is the micromagnetic Dzyaloshinskii-Moriya tensor,
``E = D_ij n_i q_j``.

**Why the mask is the physics rather than a convenience.** The spiral
Hamiltonian's own nonlocal term (``hamiltonian/noncollinear.py``) projects the
down component on ``vkb(k - q/2)`` and returns it on ``vkb(k + q/2)`` through
``D^{up,dn}``, which is right for a ``D`` that turns with the spins, the
exchange field's part of an ultrasoft ``D`` for instance. ``dD`` does not turn,
so contracting its full 2x2 structure in that layout would compute the
transverse blocks as though the lattice turned with the spins: a nonzero,
plausible and wrong number. Only the spin-diagonal expectation is kept, and
the transverse contraction is returned beside it (:attr:`SpiralSpinOrbit.dropped`)
so a test can say what was removed. At ``q`` in the reciprocal lattice the two
spheres coincide and the transverse part *does* have an expectation value, which
is the ferromagnet's first-order term and :func:`~defumat.workflows.anisotropy.
frozen_expectation`'s business, so that ``q`` is refused.

**Norm-conserving only.** On an ultrasoft dataset ``soc_scale`` also changes
``qq_so`` and ``newd_so``'s sandwich ``F B F`` of the augmentation integrals,
and the exchange field inside ``B`` turns with the spiral while ``fcoef`` does
not, so which blocks of that sandwich keep a component on its own sphere is a
separate derivation, not written. PAW adds the one-centre small component on
top (``PLAN.md`` P117).

This is the step FLEUR takes for the Dzyaloshinskii-Moriya interaction on top of
its generalized-Bloch spirals (Heide, Bihlmayer and Blugel, Physica B 404, 2678
(2009); Kurz et al., PRB 69, 024415 (2004)); both references are from memory
and have not been checked against the papers. ``pw.x`` has no spin spiral and
Elk refuses spin-orbit coupling on one (``init0.f90`` sets ``spinorb =
.false.`` when ``spinsprl``), so neither has this quantity. ``PLAN.md`` P123.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from defumat.pseudo.upf import Pseudopotential
from defumat.system.builder import System

__all__ = ["SpiralSpinOrbit", "spiral_spin_orbit_energy"]

RY_TO_MEV = 13605.693122994

#: The Pauli matrices in the order the vector ``V`` is indexed by.
_PAULI = np.array([
    [[0.0, 1.0], [1.0, 0.0]],
    [[0.0, -1.0j], [1.0j, 0.0]],
    [[1.0, 0.0], [0.0, -1.0]],
])


@dataclass
class SpiralSpinOrbit:
    """``E1(n) = n . V(q)``: the first-order spin-orbit energy of a spiral."""

    #: The spiral wavevector, in lattice coordinates (``spiral_q``).
    q: tuple
    #: ``V(q)``, Ry per unit cell: the first-order energy of this spiral with
    #: its rotation axis turned from ``z`` to ``n`` is ``n . V``. Cartesian.
    vector: np.ndarray
    #: ``(nk, 3)``, the same sum resolved by k-point (the weights inside). A
    #: centrosymmetric crystal cancels it between ``k`` and ``-k``; a single
    #: point is what a supercell's folded k-point is compared with.
    by_k: np.ndarray
    #: The transverse contraction the mask removed, ``(2,)`` complex, Ry: the
    #: ``(up, dn)`` and ``(dn, up)`` blocks of ``dD`` across the two spheres,
    #: which is what ``dD`` would contribute in the spiral's layout if it
    #: turned with the spins. Zero is *not* expected; see the module docstring.
    dropped: np.ndarray
    #: ``dD``'s spin trace, which ``soc_scale`` leaves alone, so it is zero to
    #: round-off; carried so the claim is checked rather than assumed.
    trace: float
    #: The Fermi energy the occupations were taken at, Ry (``None`` if fixed).
    fermi_energy: float | None = None

    def energy(self, axis=(0.0, 0.0, 1.0)) -> float:
        """``E1`` in Ry per cell for the spiral turned so its axis lies along ``axis``.

        The axis is the spin direction of the up component. With the up
        component at ``k + q/2`` the laboratory moment turns by ``-q . r``
        about it (``unfold_spiral_density``), so reversing ``axis`` is the
        same texture as reversing ``q``, and ``energy(-n) = -energy(n)``.
        """
        axis = np.asarray(axis, dtype=float).reshape(3)
        return float(axis @ self.vector / np.linalg.norm(axis))

    def energy_mev(self, axis=(0.0, 0.0, 1.0)) -> float:
        return self.energy(axis) * RY_TO_MEV

    @property
    def vector_mev(self) -> np.ndarray:
        return self.vector * RY_TO_MEV

    @property
    def magnitude(self) -> float:
        """``|V|``, Ry: the largest first-order energy any axis gives."""
        return float(np.linalg.norm(self.vector))

    @property
    def easy_axis(self) -> np.ndarray:
        """The axis with the lowest first-order energy, ``-V/|V|``.

        For a spiral that is the handedness the coupling prefers at this
        ``q``: the moment turns by ``-q . r`` about this axis.
        """
        norm = np.linalg.norm(self.vector)
        if norm == 0.0:
            return np.zeros(3)
        return -self.vector / norm


def spiral_spin_orbit_energy(
    system: System,
    pseudos: tuple[Pseudopotential, ...],
    density: jnp.ndarray,
    nbnd: int | None = None,
    conv_thr: float = 1.0e-10,
    k_batch: int | None | str = "default",
) -> SpiralSpinOrbit:
    """``V(q)``: the spin-orbit coupling's first-order energy on a spin spiral.

    ``system`` is the spiral with ``lspinorb`` on a fully-relativistic
    norm-conserving dataset at ``soc_scale = 0``, and ``density`` its converged
    ``SCFResult.density``. The states are recomputed once at that fixed density
    at ``conv_thr`` (P120's reason: the SCF's last states carry the SCF's last
    eigensolver threshold), and ``dD = dvan_so(1) - dvan_so(0)`` is contracted
    with each spinor component on its own sphere. See the module docstring for
    why only the spin-diagonal expectation is an energy and what the result
    means.
    """
    from defumat.workflows.anisotropy import _first_order_operator
    from defumat.workflows.nscf import fixed_density_states

    _refuse(system, pseudos)
    calculation, system, eigenvalues, wavefunctions = fixed_density_states(
        system, pseudos, density, nbnd=nbnd, conv_thr=conv_thr, k_batch=k_batch,
    )
    wg, levels = calculation.occupations(jnp.asarray(eigenvalues))
    delta_d, delta_qq = _first_order_operator(calculation, None)
    if delta_qq is not None:  # pragma: no cover - refused above
        raise AssertionError("an augmented dataset reached the spiral's E1")
    vector, by_k, dropped, trace = spiral_expectation(
        calculation, wavefunctions, wg, delta_d
    )
    fermi = levels.get("fermi_energy") if isinstance(levels, dict) else None
    return SpiralSpinOrbit(
        q=tuple(float(x) for x in system.spiral_q),
        vector=vector,
        by_k=by_k,
        dropped=dropped,
        trace=trace,
        fermi_energy=None if fermi is None else float(fermi),
    )


def spiral_expectation(calculation, wavefunctions, weights, delta_d) -> tuple:
    """``(V, V by k, the dropped cross terms, |dD^0|)`` at given spiral states.

    ``wavefunctions`` is ``(1, nk, nbnd, 2 npwx)`` in the spiral's layout,
    ``weights`` the occupations times the k-weights, ``delta_d`` the
    ``(2, 2, nkb, nkb)`` first-order operator. Split out of
    :func:`spiral_spin_orbit_energy` so that the contraction can be held
    against the operator's blocks without a diagonalisation in between.
    """
    # A spiral's projectors have ``2 nk`` rows, the up component's ``k + q/2``
    # first and the down component's ``k - q/2`` after them.
    vkb = jnp.asarray(calculation.projectors.vkb)
    nk = vkb.shape[0] // 2
    npwx = vkb.shape[1]
    psi = jnp.asarray(wavefunctions)
    components = psi.reshape(psi.shape[:-1] + (2, npwx))
    # ``<beta_i(k + q/2)|u_up>`` and ``<beta_i(k - q/2)|u_dn>``: each component
    # on the projectors of its own sphere, which is what
    # ``SpinorHamiltonian._project`` does for a spiral.
    up = jnp.einsum("kgi,skng->skni", vkb[:nk].conj(), components[..., 0, :])
    down = jnp.einsum("kgi,skng->skni", vkb[nk:].conj(), components[..., 1, :])

    delta = np.asarray(delta_d)
    # ``dD^a = (1/2) sum_st (sigma_a)_ts dD^{st}``, Hermitian in ``(i, j)``.
    parts = jnp.asarray(0.5 * np.einsum("ats,stij->aij", _PAULI, delta))
    trace = 0.5 * (delta[0, 0] + delta[1, 1])
    weights = jnp.asarray(weights)

    def expectation(left, operator, right):
        return jnp.einsum("skni,aij,sknj->askn", left.conj(), operator, right)

    diagonal = jnp.real(expectation(up, parts, up) - expectation(down, parts, down))
    by_k = jnp.einsum("askn,skn->ka", diagonal, weights)

    # What the spiral layout would make of ``dD``'s transverse blocks: the
    # ``(up, dn)`` and ``(dn, up)`` contractions across the two spheres.
    # Carried beside the energy, never added to it.
    def across(left, block, right):
        return jnp.sum(weights * jnp.einsum(
            "skni,ij,sknj->skn", left.conj(), jnp.asarray(block), right))

    dropped = np.asarray([across(up, delta[0, 1], down),
                          across(down, delta[1, 0], up)])
    return (np.asarray(jnp.sum(by_k, axis=0)), np.asarray(by_k), dropped,
            float(np.max(np.abs(trace))))


def _refuse(system: System, pseudos) -> None:
    """What this first order cannot carry, decided from the input alone."""
    if not system.spiral:
        raise ValueError(
            "spiral_spin_orbit_energy needs a spin spiral (spiral_q): for a "
            "uniform magnet the first-order spin-orbit term is "
            "workflows.anisotropy.frozen_expectation"
        )
    if not system.lspinorb or not any(p.has_so for p in pseudos):
        raise ValueError(
            "spiral_spin_orbit_energy needs lspinorb = .true. and a "
            "fully-relativistic dataset run at soc_scale = 0: the first-order "
            "operator is that dataset's coupled nonlocal coefficients minus its "
            "reduced ones, and a scalar-relativistic file has no coupling to "
            "take the difference of"
        )
    if float(system.soc_scale) != 0.0:
        raise ValueError(
            f"spiral_spin_orbit_energy expands around soc_scale = 0 and this "
            f"system is at soc_scale = {system.soc_scale}: the spiral with the "
            "coupling on is not a calculation"
        )
    q = np.asarray(system.spiral_q, dtype=float)
    if np.max(np.abs(q - np.round(q))) < 1.0e-10:
        raise ValueError(
            f"spiral_q = {tuple(q)} is a reciprocal-lattice vector, where the "
            "two spheres coincide and the coupling's transverse blocks have an "
            "expectation value of their own: that is a uniform magnet, and its "
            "first-order term is workflows.anisotropy.frozen_expectation"
        )
    if any(p.is_ultrasoft for p in pseudos):
        raise NotImplementedError(
            "spiral_spin_orbit_energy on an ultrasoft or PAW dataset: soc_scale "
            "also changes qq_so and newd_so's sandwich F B F of the augmentation "
            "integrals, and the exchange field inside B turns with the spiral "
            "while fcoef does not, so which blocks of that sandwich keep a "
            "component on its own sphere is a derivation not yet written. PAW "
            "adds the one-centre small component on top (PLAN.md P117)"
        )
    if system.hubbard:
        raise NotImplementedError(
            "spiral_spin_orbit_energy with a Hubbard U: DFT+U on a spiral is "
            "refused by the SCF itself"
        )
