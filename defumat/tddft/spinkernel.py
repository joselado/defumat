"""The transverse exchange-correlation kernel, ``f_xc^{+-}(r) = B_xc(r)/m(r)``.

``PLAN.md`` P63, and Elk's ``genspfxcr.f90``/``genspfxcg.f90`` -- of which this
is the collinear, transverse corner. It is the one ingredient of a magnon
calculation that is not a sum over states, and it is written down rather than
differentiated, because **it is not a second derivative of anything**:

    a rigid rotation of the whole magnetization by an angle ``theta`` takes
    ``delta m_perp = theta m`` and ``delta B_xc,perp = theta B_xc``, so the
    transverse response of the exchange-correlation field to the magnetization
    is ``delta B_perp = (B_xc/m) delta m_perp``.

That argument is exact and uses nothing but rotational invariance of
``E_xc[n, |m|]``, so the kernel is ``B_xc/m`` whatever the functional --
including a gradient-corrected one, where the *longitudinal* kernel is not
local at all. What the argument does not give is the transverse kernel's action
on anything other than the rotation mode; taking it to be the same
multiplication operator is the adiabatic local approximation, and it is what
makes the Goldstone theorem hold **by construction** here rather than by
cancellation. That is worth knowing before reading the Goldstone check as a
test of the kernel: it tests the *matrix* -- the transform, the ``G - G'``
folding and the sphere truncation -- and not the physics of ``f_xc``.

**``B_xc`` is the spin splitting of the self-consistent potential.** Hartree and
the local pseudopotential are spin-independent, so

    B_xc = (v_scf^up - v_scf^dn) / 2

exactly, with no separate exchange-correlation call and no functional-specific
branch -- a gradient correction is inside ``v_scf`` already. The potential is
rebuilt here from the density with **no magnetic field**, which matters: an
external or constrained field is added inside the driver's ``v_of_rho`` and is
not part of ``B_xc``, and including it would put the *applied* field into the
rotation argument, where it does not belong (an applied field breaks the
rotational symmetry, which is exactly why a magnon in a field is gapped).

**The node is where this gets interesting, and Elk gets it wrong.**
``tfm2213`` sets the transverse entry to zero wherever ``|m| < 1e-14``. The
limit is not zero: as ``m -> 0`` at fixed ``n``,

    B_xc/m  ->  dB_xc/dm  =  (f_uu - 2 f_ud + f_dd) / 4

which is Elk's own *longitudinal* entry -- the two kernels coincide at a node.
An antiferromagnet has exact nodal planes by symmetry (translation-plus-spin-
flip), so this is not a corner case there, and it is the ``abs`` trap of
``PLAN.md`` P28a one more time. Here the limit is taken by differentiating the
functional (:func:`node_limit`), so the ratio and its limit come from one
functional and agree by construction where they meet.


**The identity's proof names a condition that is a second reason for two of the refusals.**
Step one is `F m = B_xc`, exact by construction. Step two is
`X_0(0,0) B_xc = m`, and it goes through because

    <n up| B_xc |m dn> = (1/2) <n up| (H_up - H_dn) |m dn>
                       = (1/2) (eps_n^up - eps_m^dn) <n up|m dn>,

the energy denominator cancelling exactly, after which two closure sums over the minority
manifold give `n_up - n_dn`. That first line needs `H_up - H_dn` to be **a local
multiplicative potential**, which is true for LDA and GGA on a norm-conserving dataset and
false the moment the Hamiltonian's spin dependence acquires a nonlocal part: ultrasoft and
PAW have spin-dependent `D_ij`, and a Hubbard `U` has a spin-dependent projector term. So
those refusals are not only about the missing `Q_ij(q+G)` in the matrix elements -- the
**kernel** would need a nonlocal piece too, and `B_xc/m` would no longer be the whole of it.
Spin-orbit coupling breaks the same step.

**And the vacuum is where it has to be clipped, which is measured rather than
feared.** ``B_xc/m`` grows without bound as the density falls -- exchange alone
gives ``-n^{-2/3}`` -- so a cell with vacuum has a kernel that is enormous where
there is nothing to respond. Below ``density_threshold`` the kernel is held at
zero. That is a convergence aid rather than an approximation, because the pair
densities that multiply it vanish there too, but it is a *stated* one and the
check is that the answer does not move when the threshold does.

**On a hydrogen chain it does move, and that is why a cell with vacuum is the
wrong cell for this quantity.** With 8 bohr between images ``|B_xc/m|`` reaches
**173** in the tail against 21 in the core, and the leading eigenvalue of
``X_0 F`` -- the number that decides where the magnon is -- runs
**1.129, 1.033, 0.924** as the threshold moves 1e-6, 1e-3, 1e-2. On a compact
fcc cell of the same element the same eigenvalue moves by two per cent over the
same range. This is ``PLAN.md`` P45's "the screening kernel of a magnetic system
with vacuum is not finite" arriving in the transverse channel, and it is a
reason to choose the cell rather than to tune the clip.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from defumat.scf.potential import v_of_rho

__all__ = [
    "transverse_kernel_field",
    "transverse_kernel_matrix",
    "exchange_field",
    "goldstone_residual",
    "node_limit",
]

#: Below this density (electrons/bohr^3) the kernel is held at zero. QE's own
#: small-density branches sit at 1e-10 (``VANISHING_CHARGE``); this is looser
#: because what is being guarded is a *ratio* rather than a functional value.
DENSITY_THRESHOLD = 1.0e-6

#: Below this spin polarization ``|m|/n`` the ratio is replaced by its limit.
#: The two branches agree to ``O(zeta^2)``, so the crossover is not sharp and
#: its position is not a tuned number -- moving it two orders either way is
#: worth nothing, which is the check.
POLARIZATION_THRESHOLD = 1.0e-4


def exchange_field(calculation, density) -> jnp.ndarray:
    """``B_xc(r)``, ``(n1, n2, n3)`` in Ry, from a converged collinear density.

    The spin splitting of the self-consistent potential, halved. No magnetic
    field is applied -- see the module docstring.
    """
    density = jnp.asarray(density)
    if density.shape[0] != 2:
        raise ValueError(
            "the transverse kernel needs a collinear spin-polarized density, "
            f"(2, ...); got leading axis {density.shape[0]}"
        )
    # ``v_of_rho`` directly rather than ``Calculation.potential``, which adds
    # the converged magnetic field: this has to be field-free by construction
    # and not by the caller remembering.
    potential = v_of_rho(
        density,
        calculation.basis.dense,
        calculation.system.cell,
        calculation.rho_core,
        calculation.functional,
        calculation.rho_core_g,
        calculation.quantization_axis,
    )
    return 0.5 * (potential.v_scf[0] - potential.v_scf[1])


def transverse_kernel_field(
    calculation,
    density,
    *,
    density_threshold: float = DENSITY_THRESHOLD,
    polarization_threshold: float = POLARIZATION_THRESHOLD,
) -> jnp.ndarray:
    """``f_xc^{+-}(r) = B_xc(r)/m(r)`` on the dense grid, ``(n1, n2, n3)``, Ry.

    Three branches, and only the first is the formula:

    * ``|m| > polarization_threshold n``: the ratio itself;
    * ``|m|`` below that but the density above ``density_threshold``: the
      ``m -> 0`` limit, :func:`node_limit`;
    * density below ``density_threshold``: zero.
    """
    density = jnp.asarray(density)
    charge = density[0] + density[1]
    magnetization = density[0] - density[1]
    field = exchange_field(calculation, density)

    dense = charge > density_threshold
    polarized = jnp.abs(magnetization) > polarization_threshold * jnp.abs(charge)

    # The double ``where``: the *denominator* is made safe before the division
    # so that neither branch ever evaluates a division by zero. Taking only the
    # outer ``where`` leaves a NaN that survives the selection.
    safe = jnp.where(polarized, magnetization, 1.0)
    ratio = field / safe

    limit = node_limit(calculation.functional, charge)
    return jnp.where(dense, jnp.where(polarized, ratio, limit), 0.0)


def node_limit(functional, charge) -> jnp.ndarray:
    """``dB_xc/dm`` at ``m = 0``, which is what ``B_xc/m`` tends to.

    One directional derivative of the functional's own spin potential along
    ``(+1/2, -1/2)`` in the two channel densities -- the direction that changes
    ``m`` at fixed ``n``. ``B_xc = (v_up - v_dn)/2``, so the limit is half the
    splitting's derivative, and that is exactly Elk's longitudinal entry
    ``(f_uu - 2 f_ud + f_dd)/4`` written as a derivative instead of as four
    second derivatives.

    **The local part only**, and deliberately: for a gradient-corrected
    functional the true limit carries gradient terms, and evaluating them means
    differentiating through ``gradcorr`` at exactly the points where ``|m|``
    has a node -- which is P45's diverging ``dv_of_drho``, the object that has
    NaNs on 1504 of triplet O2's grid points. The local limit is finite there
    and is right to the order the local part dominates. It is used on a set of
    measure zero plus a window, so what it must be is *finite and of the right
    size*, not exact.
    """
    charge = jnp.asarray(charge)
    half = 0.5 * charge

    def splitting(pair):
        potential = functional.spin_potential(pair)
        return 0.5 * (potential[0] - potential[1])

    channels = jnp.stack([half, half])
    tangent = jnp.stack([0.5 * jnp.ones_like(half), -0.5 * jnp.ones_like(half)])
    _, derivative = jax.jvp(splitting, (channels,), (tangent,))
    return derivative


def transverse_kernel_matrix(calculation, density, sphere, **kwargs) -> jnp.ndarray:
    """``F(G, G') = ftilde(G' - G)`` over the response sphere, ``(nm, nm)``.

    The index order is the transposed one and it is not a slip. A transverse
    perturbation of wavevector ``q`` and the ``rho_up,dn`` block it drives are
    both expanded in ``e^{-i(q+G).r}`` -- the up state sits at ``k`` and the
    down state at ``k + q``, so their product carries ``-q`` -- and multiplying
    such an expansion by a lattice-periodic ``F(r)`` gives
    ``(F g)_G = sum_G' ftilde(G' - G) g_G'``. Since ``F`` is real,
    ``ftilde(-K) = conj(ftilde(K))``, so getting this backwards conjugates the
    kernel: invisible on any centrosymmetric cell and wrong elsewhere.
    """
    field = transverse_kernel_field(calculation, density, **kwargs)
    grid = np.asarray(calculation.basis.dense.grid)
    coefficients = jnp.fft.fftn(field) / float(np.prod(grid))

    miller = np.asarray(sphere.miller)
    index = (miller[None, :, :] - miller[:, None, :]) % grid
    return coefficients[index[..., 0], index[..., 1], index[..., 2]]


def goldstone_residual(chi0, calculation, density, index: int = 0, **kwargs) -> float:
    """How far ``X_0 B_xc = m`` is from holding, relative to ``|m|``.

    **The one check on a magnon that needs no second code and has no free
    parameter.** A global spin rotation costs no energy, so the interacting
    transverse response must have a pole at ``omega = 0`` when ``q = 0``:
    ``1 - X_0 F`` is singular there, with ``m`` its null vector. Since
    ``F m = B_xc`` identically, that is the same statement as

        X_0(q = 0, omega = 0) B_xc = m.

    It is exact only with a complete band set *and* a complete G-set, so what
    comes back is a **double truncation** and it converges in both: on a
    ferromagnetic hydrogen chain, 8.0 -> 0.50 per cent over ``nbnd`` 12 -> 140
    at a fixed sphere, and 20 -> 2 per cent over ``ecut_response`` 2 -> 16 Ry at
    a fixed band count. Read it before believing a dispersion; a factor of two
    in the response's normalisation fails it by a factor of two.

    ``index`` selects the frequency, which must be ``omega = 0`` with **no
    broadening** -- an ``eta`` puts an ``O(eta)`` error into the identity that
    reads as an assembly bug.
    """
    field = np.asarray(exchange_field(calculation, jnp.asarray(density)))
    magnetization = np.asarray(density[0] - density[1])
    grid = np.asarray(calculation.basis.dense.grid)
    miller = np.asarray(chi0.sphere.miller) % grid

    def plus(f):
        # The coefficient in an ``e^{-i(q+G).r}`` expansion, which for a real
        # field is the conjugate of the ordinary transform -- the same
        # convention :func:`transverse_kernel_matrix` transposes for.
        coefficients = np.fft.fftn(f) / float(np.prod(grid))
        return np.conj(coefficients[miller[:, 0], miller[:, 1], miller[:, 2]])

    target = plus(magnetization)
    got = np.asarray(chi0.x[index]) @ plus(field)
    return float(np.linalg.norm(got - target) / np.linalg.norm(target))
