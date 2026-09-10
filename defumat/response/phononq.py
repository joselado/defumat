"""The dynamical matrix at ``q != 0``: the perturbed states live on a second sphere.

``PLAN.md`` P71. Everything the response stack has computed so far is a
zone-centre quantity -- the dielectric constant, the Born charges, the
``Gamma`` dynamical matrix -- and what separates those from a phonon
*dispersion* is one fact: a displacement pattern

    u_s(R) = u_s e^{i q . R}

is not a displacement of the unit cell. It is periodic only on a supercell,
and a supercell is what a frozen-phonon calculation pays for. Linear response
does not pay it, because the *first-order* quantities are still lattice
periodic once their ``e^{i q . r}`` is factored out: the perturbation is
``e^{i q . r}`` times a periodic function, so ``dV_q |psi_k>`` is a Bloch state
at ``k + q`` and the whole calculation stays in the unit cell with one extra
plane-wave sphere. That is the same trick the spin spirals use (``PLAN.md``
P19, Elk's ``gengkqvec``), one perturbation instead of one spinor component.

**What changes, term by term, and what does not.**

| | ``q = 0`` (P25) | ``q != 0`` (here) |
|---|---|---|
| where ``dpsi`` lives | the ``k`` sphere | the ``k + q`` sphere |
| ``H - eps S`` | both at ``k`` | ``H``, ``S`` at ``k+q``, ``eps`` at ``k`` |
| ``P_c^+`` | occupied at ``k`` | occupied at ``k+q`` |
| the bare local term | ``jvp`` of ``V_loc(G)`` | ``jvp`` of ``V_loc(G+q)`` |
| the Hartree kernel | ``8 pi / |G|^2``, ``G = 0`` dropped | ``8 pi / |G+q|^2``, **``G = 0`` kept** |
| ``drho`` | ``2 Re[psi* dpsi]`` | ``2 psi* dpsi``, complex |
| the frozen Hessian | ``dynmat_us + d2ionq(0)`` | ``dynmat_us`` unchanged ``+ d2ionq(q)`` |

The last row is the one worth reading twice, and it was checked in the Fortran
rather than assumed. ``dynmat_us.f90:105-124`` fills ``dynwrk(na_icart,
na_jcart)`` -- **the same atom twice** -- from ``g(:,ng)`` and ``tau(na)``, with
no ``xq`` anywhere in the routine. The second derivative of the external
potential at frozen density is diagonal in the atom index, so under a modulated
displacement its two phases cancel and it is *the same matrix at every* ``q``.
The whole ``q`` dependence of the part that does not involve the response is
therefore in the Ewald sum, ``d2ionq``.

**What is refused, and it is most things.** This lands the norm-conserving
insulator on a full k-grid: no ultrasoft or PAW (``S`` moves with the atoms and
the multiplier matrix has no two-sphere form), no metal, no spin polarization,
no spinor, no nonlinear core correction (``dynmatcc.f90:105`` calls
``set_drhoc(xq, drc)``, so unlike ``dynmat_us`` the core term *is* a function of
``q``), and no symmetry -- the small group of ``q`` and the star of ``q`` are
what a dispersion needs and are not written. Each is named at the door.

References: Baroni, de Gironcoli, Dal Corso and Giannozzi, *Rev. Mod. Phys.*
**73**, 515 (2001), §II.B for the two-sphere structure; ``PHonon/PH`` for the
transcribed halves (``dvqpsi_us``, ``compute_dvloc``, ``incdrhoscf``,
``d2ionq``, ``drhodv``, ``dynmat_us``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from defumat.basis.fft import g_to_r, r_to_g
from defumat.response.sternheimer import SternheimerSolver
from defumat.system.cell import Cell
from defumat.system.kpoints import KPoints

__all__ = [
    "dynamical_matrix_at_q", "kpoints_plus_q", "states_at_k_plus_q",
    "TwoSphereSolver", "bare_displacements_at_q", "self_consistent_response_at_q",
    "hartree_at_q", "induced_potential_at_q", "induced_perturbation_at_q",
    "ewald_dynamical_matrix", "frozen_force_constants", "response_force_constants",
    "require_a_two_sphere_regime",
]


def kpoints_plus_q(kpoints: KPoints, q_cart, cell: Cell) -> KPoints:
    """The list ``k + q``. ``q_cart`` is in **1/bohr**, and that matters.

    **Two cartesian conventions meet here and only this function sees both.**
    Everything else in this module works against
    :meth:`~defumat.basis.gvectors.GVectors.cartesian`, which is 1/bohr, so
    ``q_cart`` is in 1/bohr throughout -- while :class:`KPoints` stores
    ``coords`` in units of ``2 pi / alat``, which is what ``pw.x`` prints.
    Adding one to the other is dimensionally silent: the k-points move by
    ``tpiba`` times too much, every array is the right shape, every solve
    converges, and the answer is **exactly right at** ``q = 0`` because zero
    scales to zero. It was worth 1822 cm^-1 at ``X`` and nothing at ``Gamma``,
    which is why ``PLAN.md`` P71 has a regression at ``q = 0`` *and* a number at
    the zone boundary rather than only the first.

    The shifted points are kept **literal** -- not wrapped back into the first
    Brillouin zone -- for the reason :mod:`defumat.system.spiral` gives for the
    same choice: the wrapped point is the same physics through a different
    G-index map, and reconciling the two is P16's zone-edge Miller shift. A
    sphere built directly at ``k + q`` needs none of it.

    The weights are carried through unchanged. They are the weights of ``k``,
    which is what every sum over the zone here is over; the ``k + q`` list is a
    place to put a second sphere and a second diagonalisation, not a second
    integration grid.
    """
    shift = np.asarray(q_cart, dtype=float) / cell.tpiba
    coords = np.asarray(kpoints.coords) + shift[None, :]
    return KPoints.from_cartesian(
        coords, np.asarray(kpoints.weights), precision=kpoints.precision
    )


def states_at_k_plus_q(calculation, v_scf, q_cart, nbnd, ethr: float = 1e-13):
    """The ground state again, on the ``k + q`` sphere, at the converged potential.

    ``phq_init``'s job: ``evq``, the unperturbed states the projector ``P_c^+``
    and the preconditioner are built from. It is a non-self-consistent
    diagonalisation and nothing about it is new -- the density, the potential
    and every radial table are the ones the SCF ended with, and only the things
    carrying a ``k`` index are rebuilt.

    That is what :meth:`~defumat.scf.driver.Calculation.at_kpoints` is for, and
    using it rather than a fresh :class:`~defumat.scf.driver.Calculation` is a
    memory decision as much as a speed one: the dense G set, the FFT box, the
    local potential, the Ewald neighbour list and the augmentation tables are
    *shared* between the two calculations rather than built twice.

    Returns ``(calculation_kq, hamiltonians_kq, eigenvalues_kq, psi_kq)``.
    """
    moved = calculation.at_kpoints(kpoints_plus_q(
        calculation.system.kpoints, q_cart, calculation.system.cell
    ))
    hamiltonians = moved.hamiltonian(v_scf)
    eigenvalues, wavefunctions = moved.diagonalize(hamiltonians, nbnd, None, ethr)
    return moved, hamiltonians, np.asarray(eigenvalues), wavefunctions


class TwoSphereSolver(SternheimerSolver):
    """``(H_{k+q} - eps_k S_{k+q} + alpha Q) dpsi = -P_c^{k+q} dV_q |psi_k>``.

    The Sternheimer equation with its two sides on different spheres, and the
    asymmetry is the whole of it: the operator, the projector and the
    preconditioner belong to ``k + q``, while the eigenvalue subtracted is the
    one of the band being perturbed, at ``k``. QE writes exactly that split --
    ``ch_psi_all.f90`` subtracts ``e(ibnd)`` from ``et(:,ikk)`` while applying
    ``h_psi`` at ``ikq``, ``orthogonalize.f90`` takes ``evq``, and
    ``h_prec.f90``'s docstring says "evq wavefunction at k+q point" in as many
    words -- and getting either half from the wrong sphere gives an equation
    that still converges.

    A **subclass** rather than a pair of optional arguments on
    :class:`~defumat.response.sternheimer.SternheimerSolver`, because that class
    is what P24, P25, P28, P45 and P70 are validated through and threading a
    second state set into six of its methods would put every one of those
    numbers behind an untested branch. What this owes in exchange is that
    ``q = 0`` through *this* class must reproduce the parent exactly, which is
    the first thing its tests assert.
    """

    def __init__(self, base: SternheimerSolver, hamiltonians_kq, psi_kq,
                 eigenvalues_kq, calculation_kq):
        # Built from a converged parent rather than from scratch: everything
        # that is a property of the ground state at ``k`` -- the occupied
        # counts, ``alpha_pv``, the weights, the thresholds -- is already
        # decided there, and duplicating that logic is how the two would drift.
        self.__dict__.update(base.__dict__)
        self.calculation_kq = calculation_kq
        # ``self.hamiltonians`` **is** the second sphere's here. The solve is
        # the only thing that reads it -- the operator, the projector, the
        # preconditioner and the band mask -- and all four belong at ``k + q``,
        # so replacing it wholesale is what keeps ``solve_at``'s sixty lines of
        # conjugate gradient inherited rather than copied. The states at ``k``
        # are still ``self.psi``, which is what the right-hand side and the
        # response density are built from.
        self.hamiltonians = tuple(hamiltonians_kq)
        keep = self.psi.shape[2]
        self.psi_kq = jnp.asarray(psi_kq)[:, :, :keep]
        self.eigenvalues_kq = jnp.asarray(eigenvalues_kq)[:, :, :keep]
        # The projector runs over the occupied manifold **at k + q**, and the
        # right-hand side runs over the bands being solved for, **at k**. They
        # are the same count for an insulator with a gap everywhere -- which is
        # the only regime this class admits -- but they are not the same object,
        # and writing one mask for both is the kind of thing that is right until
        # a band crosses.
        bands = jnp.arange(self.eigenvalues_kq.shape[2])
        counts = jnp.asarray(self.occupied_counts)
        self.projector_mask_kq = jnp.broadcast_to(
            (bands[None, :] < counts[:, None])[:, None, :],
            self.eigenvalues_kq.shape,
        )

    # -- the three pieces that move to the second sphere -------------------

    def _operator(self, vectors, ik, spin):
        """``ch_psi_all`` with ``H`` and ``S`` at ``k+q`` and ``eps`` at ``k``."""
        hamiltonian = self.hamiltonians[spin]
        occupied = self.psi_kq[spin][ik]
        eps = self.eigenvalues[spin][ik][:, None]

        h = hamiltonian.apply(vectors, ik)
        s = hamiltonian.apply_s(vectors, ik)
        out = h - eps * s

        overlaps = jnp.einsum("mg,ng->mn", jnp.conj(occupied), s)
        overlaps = jnp.where(self.projector_mask_kq[spin][ik][:, None], overlaps, 0.0)
        lifted = jnp.einsum("mn,mg->ng", overlaps, occupied)
        return out + self.alpha_pv * hamiltonian.apply_s(lifted, ik)

    def project(self, rhs, ik, spin):
        """``-P_c^+ rhs`` with the occupied manifold taken at ``k + q``.

        Two masks rather than one: the **row** mask is over the bands being
        solved for and is the ground state's at ``k``; the **column** mask is
        over the manifold projected out and is the one at ``k + q``.
        """
        hamiltonian = self.hamiltonians[spin]
        occupied = self.psi_kq[spin][ik]
        s_occupied = hamiltonian.apply_s(occupied, ik)

        rows = self.projector_mask[spin][ik][:, None]
        columns = self.projector_mask_kq[spin][ik][:, None]
        rhs = jnp.where(rows, rhs, 0.0)
        overlaps = jnp.where(
            columns, jnp.einsum("mg,ng->mn", jnp.conj(occupied), rhs), 0.0
        )
        return -(rhs - jnp.einsum("mn,mg->ng", overlaps, s_occupied))

    def _preconditioner(self, ik, spin):
        """``h_prec`` on the second sphere: ``|k+q+G|^2`` and ``evq``."""
        hamiltonian = self.hamiltonians[spin]
        occupied = self.psi_kq[spin][ik]
        kinetic = hamiltonian.state_kinetic[ik]
        expectation = jnp.real(
            jnp.einsum("ng,g,ng->n", jnp.conj(occupied), kinetic, occupied)
        )
        eprec = 1.35 * expectation
        return 1.0 / jnp.maximum(1.0, kinetic[None, :] / eprec[:, None])

    # -- the density it produces -------------------------------------------

    def response_density_at_q(self, dpsi) -> jnp.ndarray:
        """``drho_q(r)``: ``incdrhoscf``, and here it is transcribed.

        At ``Gamma`` the response density is one ``jvp`` of the *code that
        builds a density* -- ``rho = sum wg Re[psi* psi]``, differentiated along
        ``psi -> psi + dpsi`` -- and every ultrasoft term comes with it (P24).
        At ``q != 0`` that route is not available and the reason is physical
        rather than technical: the perturbed crystal has no density in this
        cell at all. What is periodic is the ``+q`` Fourier component,

            drho_q(r) = (2 / Omega) sum_k w_k sum_v conj(psi_vk(r)) dpsi_vk(r)

        with ``psi`` on the ``k`` sphere and ``dpsi`` on the ``k+q`` one, and
        the two factors in it are **not** a modulus. The leading 2 is the
        ``-q`` half of a real perturbation, which contributes the same amount
        (``incdrhoscf.f90:76``, ``wgt = 2 weight / omega``); at ``Gamma`` the
        two halves are complex conjugates of each other and collapse to the
        ``2 Re[...]`` the ``jvp`` produces, which is what makes the two
        expressions agree there.

        On the **dense** grid, like every other density here, and with the
        ``(nspin_mag, ...)`` leading axis every consumer of one expects.
        """
        from defumat.basis.interpolate import to_dense
        from defumat.batching import sum_bands, sum_k

        calculation, kq = self.calculation, self.calculation_kq
        grid = calculation.basis.smooth.grid
        volume = calculation.system.cell.volume
        dpsi = jnp.asarray(dpsi)
        index_k, index_kq = calculation.fft_index, kq.fft_index

        def one_k(item):
            states, tangent, here, there, weight = item

            # **The bands are walked, not batched**, which is
            # ``incdrhoscf.f90``'s own ``DO ibnd`` and is the same working-set
            # argument the ground-state ``sum_band`` makes -- with *three*
            # band-sized real-space arrays here rather than one, since both
            # factors are transformed and then multiplied (``PLAN.md`` P74).
            def one_band(arrays):
                state, tangent_band, occupation = arrays
                psi_r = g_to_r(state, here, grid)
                dpsi_r = g_to_r(tangent_band, there, grid)
                return occupation.astype(states.dtype) * jnp.conj(psi_r) * dpsi_r

            return sum_bands(one_band, (states, tangent, weight))

        total = jnp.zeros(grid, dtype=self.psi.dtype)
        for spin in range(self.nspin):
            total = total + sum_k(
                one_k,
                (self.psi[spin], dpsi[spin], index_k, index_kq,
                 self.density_weights[spin]),
                batch=calculation.k_batch,
            )
        smooth, dense = calculation.basis.smooth, calculation.basis.dense
        return to_dense(2.0 * total / volume, smooth, dense)[None]


# ---------------------------------------------------------------------------
# The bare perturbation.
# ---------------------------------------------------------------------------

def bare_displacements_at_q(calculation, calculation_kq, solver, q_cart, positions,
                            atoms=None) -> np.ndarray:
    """``dV_bare_q/du |psi_k>`` for every atom and direction, on the ``k+q`` sphere.

    ``dvqpsi_us.f90``, and it stays what P25 made it: **one ``jvp`` through the
    positions of code that applies a potential**, rather than a second,
    hand-derived expression. What ``q`` changes is the code being
    differentiated, not the way the derivative is taken -- two substitutions and
    nothing else:

    * the local potential is built on the shifted argument
      (:func:`~defumat.pseudo.potentials.local_potential_at_q`), so
      differentiating its phase brings down ``-i(G+q)`` where at ``Gamma`` it
      brought down ``-iG``, and the product with ``psi`` is gathered out of the
      box onto the ``k+q`` sphere rather than back onto ``k``;
    * the nonlocal term is written across the two spheres,
      ``|beta_{k+q}> D <beta_k|psi_k>``. **Both** factors carry the positions,
      so one ``jvp`` produces both halves of ``dvqpsi_us_only`` --
      ``|dbeta_{k+q}> D <beta_k|psi>`` and ``|beta_{k+q}> D <dbeta_k|psi>`` --
      without either being written down.

    The *value* of that function is not a physical operator: it applies
    ``V_loc(G+q)`` as if it were a potential, and its nonlocal part pairs
    projectors at two different k. Only the tangent is used, and the tangent is
    right, which is the same trade
    :func:`~defumat.response.phonon._bare_displacements` already makes at
    ``Gamma``.

    Returns an object array of shape ``(nat, 3)`` -- or ``(len(atoms), 3)`` --
    whose entries are ``(nspin, nk, nocc, npwx_kq)``.
    """
    import equinox as eqx

    from defumat.basis.fft import gather_from_box
    from defumat.batching import map_bands, map_k
    from defumat.pseudo.potentials import local_potential_at_q

    smooth = calculation.basis.smooth
    grid = smooth.grid
    points = grid[0] * grid[1] * grid[2]
    cell = calculation.system.cell
    structure = calculation.system.structure
    pseudos = calculation.pseudos
    batch = calculation.k_batch
    psi = solver.psi
    positions = jnp.asarray(positions)

    index_k = calculation.fft_index
    index_kq = calculation_kq.fft_index
    mask_kq = calculation_kq.basis.planewaves.mask
    dij = tuple(h.coefficients for h in solver.hamiltonians)

    def applied(pos):
        moved = eqx.tree_at(lambda s: s.positions, structure, pos)
        potential = g_to_r(
            local_potential_at_q(pseudos, moved, cell, smooth, q_cart),
            smooth.fft_index, grid,
        )
        vkb_k = calculation.projector_core.at_positions(pos).vkb
        vkb_kq = calculation_kq.projector_core.at_positions(pos).vkb

        blocks = []
        for spin in range(psi.shape[0]):
            coefficients = dij[spin].astype(vkb_kq.dtype)

            def one_k(item, coefficients=coefficients):
                states, here, there, beta_k, beta_kq, keep = item

                # The local term is a band's box in and a band's box out, so
                # it takes the band dial; the two einsums below are
                # ``(n, npwx) x (npwx, nkb)`` and hold nothing grid-sized.
                def local_block(block):
                    field = g_to_r(block, here, grid)
                    box = jnp.fft.fftn(
                        field * potential, axes=(-3, -2, -1)) / points
                    return gather_from_box(box, there)

                local = map_bands(local_block, states)
                projected = jnp.einsum("gk,ng->nk", beta_k.conj(), states)
                nonlocal_ = jnp.einsum(
                    "gk,nk->ng", beta_kq, projected @ coefficients.T
                )
                return jnp.where(keep, local + nonlocal_, 0.0)

            blocks.append(map_k(
                one_k,
                (psi[spin], index_k, index_kq, vkb_k, vkb_kq, mask_kq),
                batch=batch,
            ))
        return jnp.stack(blocks)

    nat = positions.shape[0]
    chosen = tuple(range(nat)) if atoms is None else tuple(atoms)
    bare = np.empty((len(chosen), 3), dtype=object)
    for row, atom in enumerate(chosen):
        for cart in range(3):
            tangent = jnp.zeros_like(positions).at[atom, cart].set(1.0)
            bare[row, cart] = jax.jvp(applied, (positions,), (tangent,))[1]
    return bare


# ---------------------------------------------------------------------------
# The screening kernel, and the loop that applies it.
# ---------------------------------------------------------------------------

def hartree_at_q(calculation, drho, q_cart) -> jnp.ndarray:
    """``8 pi drho(G) / |G+q|^2`` on the dense grid, in real space.

    The one term of the screening kernel that knows about ``q``, and the ``G =
    0`` entry is where it shows. At ``Gamma`` that entry is dropped -- the
    average electrostatic potential of a periodic solid is not defined, and its
    divergence cancels against the Ewald and local-potential ones
    (``dv_of_drho.f90`` drops it, and so does
    :func:`defumat.scf.potential.hartree`). At ``q != 0`` there is nothing to
    drop: ``|G+q|`` is bounded away from zero, the ``G = 0`` coefficient is an
    ordinary finite number, and **leaving it out is a real error** rather than
    a convention -- it is the long-wavelength part of the screening, which is
    the largest part.

    The guard below is written as ``|G+q|^2 > 0`` rather than as a branch on
    ``q``, so ``q = 0`` reproduces :func:`~defumat.scf.potential.hartree`'s own
    exclusion exactly. That is what makes the substitution in
    :func:`induced_potential_at_q` a piece of arithmetic rather than an
    approximation.
    """
    from defumat.basis.fft import g_to_r as _g_to_r
    from defumat.units import E2, FPI

    dense = calculation.basis.dense
    cell = calculation.system.cell
    g = dense.cartesian(cell) + jnp.asarray(q_cart)[None, :]
    g2 = jnp.sum(g * g, axis=-1)
    finite = g2 > 1e-12
    inverse = jnp.where(finite, 1.0 / jnp.where(finite, g2, 1.0), 0.0)

    charge = jnp.sum(jnp.asarray(drho), axis=0)
    v = E2 * FPI * r_to_g(charge, dense.fft_index) * inverse
    return _g_to_r(v, dense.fft_index, dense.grid)[None]


def induced_potential_at_q(calculation, density, drho, q_cart) -> jnp.ndarray:
    """``dV_Hxc`` at wavevector ``q``: ``dv_of_drho.f90``.

    Built out of the ``Gamma`` kernel rather than beside it, and the two edits
    are the two ways a finite ``q`` reaches a *local* functional of the density:

    * the **exchange-correlation** part does not know about ``q`` at all. It is
      ``f_xc(r) drho(r)``, pointwise in real space, so the operator is the same
      one at every wavevector. What changes is only that ``drho`` is now
      complex, and a real linear kernel applied to a complex field is the same
      kernel on its two parts -- which is also why the ``jvp`` below is taken
      twice rather than handed a complex tangent it would refuse: the primal
      density is real, and JAX requires a tangent of the primal's dtype.
    * the **Hartree** part does, through ``1/|G+q|^2`` and through the ``G = 0``
      term it no longer drops. It is subtracted at ``q = 0`` and added back at
      ``q``, which is exact because :func:`hartree_at_q` at ``q = 0`` *is* the
      expression inside ``v_of_rho`` -- and the check on that claim is that
      ``q = 0`` through this function returns the P25 kernel to round-off.
    """
    density = jnp.asarray(density)
    drho = jnp.asarray(drho)

    def potential(rho):
        return calculation.potential(rho).v_scf

    real = jax.jvp(potential, (density,), (jnp.real(drho),))[1]
    imaginary = jax.jvp(potential, (density,), (jnp.imag(drho),))[1]
    kernel = real + 1j * imaginary
    return (
        kernel
        - hartree_at_q(calculation, drho, jnp.zeros(3))
        + hartree_at_q(calculation, drho, q_cart)
    )


def induced_perturbation_at_q(calculation, calculation_kq, dv):
    """``dV_scf(r) |psi_k>`` gathered onto the ``k+q`` sphere.

    The self-consistent half of the perturbation. ``dV_scf`` is the *periodic
    part* -- the ``e^{i q . r}`` is carried by which sphere the answer is
    gathered onto, not by the field -- so this is an ordinary local operator
    applied through the FFT box, with one index set going in and another coming
    out. That asymmetry is the only difference from
    :func:`~defumat.response.sternheimer.local_perturbation`.
    """
    from defumat.basis.fft import gather_from_box
    from defumat.basis.interpolate import to_smooth
    from defumat.batching import map_bands

    smooth, dense = calculation.basis.smooth, calculation.basis.dense
    grid = smooth.grid
    points = grid[0] * grid[1] * grid[2]
    field = jnp.stack([to_smooth(component, dense, smooth) for component in dv])
    index_k, index_kq = calculation.fft_index, calculation_kq.fft_index
    mask = calculation_kq.basis.planewaves.mask

    def apply(states, ik, spin):
        def block(chunk):
            box = jnp.fft.fftn(
                g_to_r(chunk, index_k[ik], grid) * field[spin],
                axes=(-3, -2, -1),
            ) / points
            return gather_from_box(box, index_kq[ik])

        return jnp.where(mask[ik], map_bands(block, states), 0.0)

    return apply


def self_consistent_response_at_q(
    calculation, calculation_kq, solver, bare, density, q_cart,
    alpha_mix: float = 0.7, tr2: float = 1e-14, max_iterations: int = 100,
    mixing_mode: str = "anderson", verbose: bool = False,
):
    """``solve_linter``'s loop at ``q != 0``.

    The same fixed point as :func:`~defumat.response.phonon.self_consistent_response`
    -- solve for ``dpsi``, build ``drho``, screen it, mix, repeat -- with three
    differences and no fourth:

    * ``dpsi`` and ``drho`` are **complex** and stay complex. At ``Gamma`` the
      response of a real perturbation is real and the code takes the real part;
      here the ``+q`` and ``-q`` halves of the perturbation are different
      operators and their responses are complex conjugates rather than equal.
      The mixer sees the two parts as one real vector of twice the length, which
      is what ``mix_pot.f90`` does with the same array.
    * the kernel is :func:`induced_potential_at_q`;
    * **nothing is symmetrised.** ``symdvscf`` averages a response over the
      small group of ``q`` -- the operations with ``S q = q + G`` -- which is
      not the crystal's group and is not written here. Until it is, this needs
      the whole k-grid, and :func:`dynamical_matrix_at_q` refuses a reduced one
      by name.

    Returns ``(dpsi, drho, history, average_iterations, converged)``.
    """
    from defumat.response.mixing import ResponseMixer

    nat = np.asarray(calculation.system.structure.positions).shape[0]
    grid_shape = jnp.asarray(density).shape
    dvscf = jnp.zeros((nat, 3) + grid_shape, dtype=solver.psi.dtype)
    dpsi = np.empty((nat, 3), dtype=object)
    history, total_iterations, solves = [], 0, 0
    converged = False

    mixer = ResponseMixer(mixing_mode, beta=alpha_mix)
    for iteration in range(max_iterations):
        response = []
        for atom in range(nat):
            for cart in range(3):
                if iteration == 0:
                    perturbation = (
                        lambda psi, ik, spin, b=bare[atom, cart]: b[spin][ik]
                    )
                else:
                    induced = induced_perturbation_at_q(
                        calculation, calculation_kq, dvscf[atom, cart]
                    )
                    perturbation = (
                        lambda psi, ik, spin, b=bare[atom, cart], f=induced:
                        b[spin][ik] + f(psi, ik, spin)
                    )
                solution = solver.solve(perturbation)
                dpsi[atom, cart] = solution.dpsi
                total_iterations += solution.iterations
                solves += 1
                response.append(solver.response_density_at_q(solution.dpsi))

        drho = jnp.stack(response).reshape((nat, 3) + grid_shape)
        induced = jnp.stack([
            induced_potential_at_q(calculation, density, drho[atom, cart], q_cart)
            for atom in range(nat) for cart in range(3)
        ]).reshape(dvscf.shape)

        change = float(jnp.max(jnp.abs(induced - dvscf)) ** 2)
        history.append(change)
        if verbose:
            print(f"  response iteration {iteration + 1}: |ddV|^2 = {change:.3e}")
        if change < tr2:
            dvscf = induced
            converged = True
            break

        real, imaginary = mixer.mix(
            [np.real(dvscf), np.imag(dvscf)],
            [np.real(induced), np.imag(induced)],
        )
        dvscf = real + 1j * imaginary

    return dpsi, drho, history, total_iterations / max(1, solves), converged


# ---------------------------------------------------------------------------
# The ionic second derivative: d2ionq.
# ---------------------------------------------------------------------------

def ewald_dynamical_matrix(calculation, q_cart, alpha: float | None = None):
    """``d2ionq``: the Ewald sum's second derivative under a modulated displacement.

    **The only term of the frozen Hessian that knows about ``q``.** The
    electronic half of ``dynmat0`` -- the second derivative of the external
    potential at frozen density -- is diagonal in the atom index
    (``dynmat_us.f90`` fills ``dynwrk(na_icart, na_jcart)``, the same atom
    twice), so its two modulation phases cancel and it is the same matrix at
    every wavevector. The ion-ion sum is not diagonal, and this is what
    replaces it.

    Both halves of Ewald's split have the same shape, and it is the shape a
    lattice sum of a *pair* function always has:

        D_(ai)(bj)(q) = C_(ai)(bj)(q)  -  delta_ab sum_c C_(ai)(cj)(0)

    -- the cross term evaluated at ``q``, minus the same object at ``q = 0``
    summed over the other atom. The second piece is the atom's own recoil: an
    atom displaced alone feels the force its neighbours no longer exert, and
    that term is unmodulated because both derivatives are taken at the same
    site. It is also why the ionic matrix obeys the acoustic sum rule at
    ``q = 0`` identically rather than to the accuracy of the sum.

    Transcribed from ``PHonon/PH/d2ionq.f90``, including its choices, because
    matching them is what makes agreement with ``ph.x`` exact rather than
    approximate: ``alpha`` stepped down from 2.9 until the reciprocal
    truncation error is below **1e-9** (the ground-state ``ewald.f90`` uses 1e-7
    -- the sum is independent of ``alpha``, but the two truncations are not),
    and a real-space cutoff of ``5/sqrt(alpha)`` rather than ``4/sqrt(alpha)``.

    Returns ``(3 nat, 3 nat)`` complex, in Ry/bohr^2, with the row index
    conjugated -- the convention ``D(q)^dagger = D(q)`` is stated in.
    """
    from jax.scipy.special import erfc

    from defumat.scf.ewald import ewald_alpha
    from defumat.system.cell import lattice_translations, pair_separation_bound
    from defumat.units import E2, FPI

    cell = calculation.system.cell
    structure = calculation.system.structure
    dense = calculation.basis.dense
    positions = jnp.asarray(structure.positions)
    charges = jnp.asarray(calculation.charges)
    nat = positions.shape[0]
    q = jnp.asarray(q_cart)

    if alpha is None:
        tpiba2 = cell.tpiba**2
        alpha = ewald_alpha(
            float(np.sum(np.asarray(charges))), dense.ecut / tpiba2, tpiba2,
            tolerance=1.0e-9,
        )

    # -- reciprocal space --------------------------------------------------
    g = dense.cartesian(cell)
    volume = cell.volume

    def reciprocal(shift):
        gq = g + jnp.asarray(shift)[None, :]
        g2 = jnp.sum(gq * gq, axis=-1)
        finite = g2 > 1.0e-8
        safe = jnp.where(finite, g2, 1.0)
        factor = jnp.where(
            finite, -E2 * FPI / volume * jnp.exp(-safe / alpha / 4.0) / safe, 0.0
        )
        phases = jnp.exp(1j * (gq @ positions.T))       # (ngm, nat)
        weighted = factor[:, None, None] * gq[:, :, None] * gq[:, None, :]
        return jnp.einsum("ga,gij,gb->abij", phases, weighted, jnp.conj(phases))

    pairs = charges[:, None] * charges[None, :]
    cross = pairs[:, :, None, None] * reciprocal(q)
    self_term = pairs[:, :, None, None] * reciprocal(jnp.zeros(3))

    # -- real space --------------------------------------------------------
    rmax = 5.0 / np.sqrt(alpha)
    at = np.asarray(cell.at)
    radius = rmax + pair_separation_bound(at, np.asarray(structure.positions))
    translations = jnp.asarray(lattice_translations(at, radius))

    separation = positions[:, None, :] - positions[None, :, :]      # (nat, nat, 3)
    vectors = translations[None, None, :, :] - separation[:, :, None, :]
    square = jnp.sum(vectors * vectors, axis=-1)
    keep = (square > 1.0e-16) & (square <= rmax**2)
    distance = jnp.sqrt(jnp.where(keep, square, 1.0))
    ar = np.sqrt(alpha) * distance
    root = 2.0 / np.sqrt(np.pi)
    second = jnp.where(
        keep,
        (3.0 * erfc(ar) + root * ar * (3.0 + 2.0 * ar**2) * jnp.exp(-ar**2))
        / distance**5,
        0.0,
    )
    first = jnp.where(
        keep, (-erfc(ar) - root * ar * jnp.exp(-ar**2)) / distance**3, 0.0
    )
    # ``qrg = q . (r + dtau)``, and ``r + dtau`` is the lattice vector itself.
    phase = jnp.exp(1j * (translations @ q))[None, None, :]
    outer = vectors[..., :, None] * vectors[..., None, :]
    identity = jnp.eye(3)

    def assemble(weight):
        return E2 * pairs[:, :, None, None] * (
            jnp.einsum("abt,abtij->abij", weight * second, outer)
            + jnp.einsum("abt,ij->abij", weight * first, identity)
        )

    cross = cross + assemble(phase)
    self_term = self_term + assemble(jnp.ones_like(phase))

    matrix = cross - jnp.eye(nat)[:, :, None, None] * jnp.sum(
        self_term, axis=1
    )[:, None, :, :]
    return -jnp.transpose(matrix, (0, 2, 1, 3)).reshape(3 * nat, 3 * nat)


# ---------------------------------------------------------------------------
# The assembly.
# ---------------------------------------------------------------------------

def _diagonalize_at_q(matrix: np.ndarray, masses: np.ndarray):
    """``dyndia`` for a complex hermitian ``D(q)``.

    :func:`~defumat.response.phonon._diagonalize` casts to ``float``, which is
    right at ``Gamma`` -- the force constants of a real crystal are real there
    -- and is not right anywhere else. Everything else is the same: mass-weight
    with ``amu_ry``, diagonalise, and report an imaginary frequency as a
    negative number.
    """
    from defumat.units import AMU_TO_RY, RY_TO_CMM1

    nat = masses.shape[0]
    flat = np.asarray(matrix, dtype=complex).reshape(3 * nat, 3 * nat)
    scale = 1.0 / np.sqrt(np.repeat(masses, 3) * AMU_TO_RY)
    weighted = flat * scale[:, None] * scale[None, :]
    omega2, vectors = np.linalg.eigh(0.5 * (weighted + weighted.conj().T))
    return np.sign(omega2) * np.sqrt(np.abs(omega2)) * RY_TO_CMM1, vectors


def response_force_constants(solver, dpsi, bare, nat) -> np.ndarray:
    """``drhodv``: the electronic half of ``D(q)``, in one contraction.

    QE splits this in two because its two halves live in different places --
    ``drhodvloc`` integrates ``conj(drho_i) dV_loc_j`` on the FFT grid, and
    ``drhodvnl`` rebuilds the nonlocal half in the projector basis from
    ``dbecq`` and ``dalpq``. Here they are one inner product, because the
    object QE has to rebuild is one this code already holds: ``bare[j]`` **is**
    ``dV_bare_j |psi>`` on the ``k+q`` sphere, local and nonlocal together, and
    contracting it against ``dpsi_i`` on that same sphere gives both halves at
    once.

        D^resp_(ai)(bj) = 2 sum_{k,n} w_kn <dpsi_(ai) | dV_bare_(bj) | psi_n k>

    **The leading 2 is not a derivation, it is a measurement**, and so is the
    index order. Both are the shape a Hermitian-looking answer can be wrong in
    (``PLAN.md`` P54, P66), so they were fixed at ``q = 0`` against the P25
    matrix on the 64-point unshifted silicon grid rather than argued for: the
    ratio of ``D_P25 - D_frozen`` to this contraction is **2.000000** on every
    element the contraction reaches, with the off-diagonal blocks agreeing at
    the 1e-5 level where both are numerically zero. It is the same 2 that
    ``incdrhoscf`` carries and for the same reason -- the ``-q`` half of a real
    perturbation contributes as much as the ``+q`` one.

    The conjugate sits on ``dpsi`` and the row index is the perturbation
    ``dpsi`` solves, which is ``drhodvloc``'s ``dot_product(drhos(ipert), dvloc(nu_j))``
    read through ``drho ~ conj(psi) dpsi``.

    **What this costs**, since a design is not finished until its working set is
    known: the loop is ``(3 nat)^2`` passes over an ``(nspin, nk, nocc, npwx)``
    block, which is the same flop count a single ``(3 nat, N)`` Gram matrix
    would be and worse constants. It allocates nothing -- ``dpsi`` and ``bare``
    are already held -- and stacking them into that matrix would hold a second
    copy of the largest arrays a phonon run has. On the cells this phase admits
    the Sternheimer solves dominate it by two orders; on a large cell the trade
    is worth revisiting, and the stack is where to start.
    """
    weights = solver.weights
    matrix = np.zeros((3 * nat, 3 * nat), dtype=complex)
    for atom in range(nat):
        for cart in range(3):
            row = 3 * atom + cart
            tangent = jnp.conj(dpsi[atom, cart])
            for other in range(nat):
                for direction in range(3):
                    matrix[row, 3 * other + direction] = complex(2.0 * jnp.sum(
                        weights * jnp.einsum(
                            "skng,skng->skn", tangent, bare[other, direction]
                        )
                    ))
    return matrix


def frozen_force_constants(calculation, solver, positions, density, q_cart):
    """``dynmat0`` at ``q``: the second derivative at a frozen electronic state.

    Three terms, and only the last of them knows about ``q``:

        dynmat0(q) = dynmat_us + dynmatcc + d2ionq(q)

    ``dynmat_us`` and ``dynmatcc`` are taken from the ``Gamma`` machinery
    unchanged -- :func:`~defumat.response.phonon._force_constants` with the
    state and density tangents set to zero is exactly ``dynmat0(0)``, which is
    those two plus ``d2ionq(0)``. The ionic term is then swapped: the **same**
    ``jax.hessian`` of the **same** Ewald energy that is inside that functional
    is subtracted, so the swap is exact to round-off and carries no
    disagreement of its own, and :func:`ewald_dynamical_matrix` is added in its
    place.

    That the other two survive the swap is the fact this whole phase rests on
    and it was read in the Fortran rather than assumed:
    ``dynmat_us.f90:105-124`` writes into ``dynwrk(na_icart, na_jcart)`` -- one
    atom index, used twice -- and ``init_us_2`` there is called at ``ikk``, so
    nothing in the routine is a function of ``xq``. A second derivative taken
    twice at the *same* site carries ``e^{iqR} e^{-iqR} = 1``.

    ``dynmatcc`` is the exception that is refused rather than handled:
    ``dynmatcc.f90:105`` calls ``set_drhoc(xq, drc)``, because the core charge
    of two *different* atoms overlaps inside a nonlinear ``E_xc`` and that
    second derivative is not diagonal. :func:`dynamical_matrix_at_q` refuses a
    dataset with a nonlinear core correction for that reason.
    """
    from defumat.response.phonon import _force_constants, _state_weights

    nat = np.asarray(positions).shape[0]
    empty = np.empty((nat, 3), dtype=object)
    for atom in range(nat):
        for cart in range(3):
            empty[atom, cart] = jnp.zeros_like(solver.psi)
    frozen = np.asarray(_force_constants(
        calculation, jnp.asarray(positions), solver.psi, solver.weights,
        _state_weights(solver, solver.weights), solver.eigenvalues,
        jnp.asarray(density), empty,
        jnp.zeros((nat, 3) + jnp.asarray(density).shape), solver.nocc,
    )).reshape(3 * nat, 3 * nat)

    cell = calculation.system.cell
    dense = calculation.basis.dense
    ionic = np.asarray(jax.hessian(
        lambda pos: calculation.ewald_sum.energy(cell, pos, dense)
    )(jnp.asarray(positions))).reshape(3 * nat, 3 * nat)

    return frozen - ionic + np.asarray(ewald_dynamical_matrix(calculation, q_cart))


def require_a_two_sphere_regime(calculation, q_crystal) -> None:
    """What a phonon at ``q != 0`` refuses, and why each one is named.

    Every entry here is a term that is *absent* rather than a feature that is
    unwritten, which is the distinction ``PLAN.md`` asks a refusal to make.
    """
    system = calculation.system
    if calculation.is_ultrasoft:
        raise NotImplementedError(
            "a phonon at q != 0 with an ultrasoft or PAW dataset is not "
            "implemented: S moves with the atoms, so the orthonormality "
            "multipliers carry a term <psi|dS/du|psi> that has no two-sphere "
            "form here -- the bra is at k and the ket at k+q, and qq_ij pairs "
            "projectors on one sphere. P39 wrote that term at Gamma and it is "
            "the piece that does not follow"
        )
    if system.nspin != 1 or system.noncolin:
        raise NotImplementedError(
            "a phonon at q != 0 needs one spin channel: the k+q ground state, "
            "the projector and the response density are each written for a "
            "single channel here, and a magnetic one is P45's refusal (the "
            "assembly rather than the solve) on a second sphere"
        )
    if calculation.rho_core_g is not None:
        raise NotImplementedError(
            "a phonon at q != 0 with a nonlinear core correction is not "
            "implemented: unlike the rest of the frozen Hessian the core term "
            "IS a function of q (dynmatcc.f90:105 calls set_drhoc(xq, drc)), "
            "because two atoms' core charges overlap inside a nonlinear E_xc "
            "and that second derivative is not diagonal in the atom. Use a "
            "dataset without PP_NLCC"
        )
    if calculation.use_symmetry:
        raise NotImplementedError(
            "a phonon at q != 0 needs the whole k-grid: the response has to be "
            "symmetrised over the small group of q -- the operations with "
            "S q = q + G -- which is not the crystal's group and is not "
            "written here. Run with nosym = .true. and noinv = .true. on an "
            "unshifted grid, which is the grid that is closed under the point "
            "group (see si-epsilon-unshifted-nosym.in)"
        )
    if system.spiral_q is not None:
        raise NotImplementedError(
            "a phonon at q != 0 on a spin spiral is not implemented: both "
            "already put the states on shifted spheres and the two shifts "
            "compose, so the perturbation would need four"
        )
    if calculation.functional.is_meta:
        raise NotImplementedError(
            "a phonon under a potential-only meta-GGA is not implemented, at "
            "any q: tb09 and bj06 are potentials with no energy functional, so "
            "there is no second derivative of an energy to take (P30)"
        )
    if calculation.is_hubbard:
        raise NotImplementedError(
            "a phonon at q != 0 with a Hubbard U is not implemented: the "
            "projectors move with the atoms and their q-modulated response is "
            "dnsq_bare/dnsq_scf, which is a second implementation rather than "
            "a shift of this one"
        )


def dynamical_matrix_at_q(
    calculation, wavefunctions, eigenvalues, density, becsum=(),
    q=(0.0, 0.0, 0.0), q_cartesian: bool = False, nbnd: int | None = None,
    threshold: float = 1.0e-14, alpha_mix: float = 0.7, tr2: float = 1.0e-14,
    max_iterations: int = 100, verbose: bool = False,
):
    """``D(q)``: the dynamical matrix at one wavevector.

    ``q`` is in **crystal** coordinates of the reciprocal lattice unless
    ``q_cartesian`` is set, in which case it is in units of ``2 pi / alat`` --
    the units ``ph.x`` prints ``xq`` in and the units
    :class:`~defumat.system.kpoints.KPoints` stores.

    The sequence is ``phonon.f90``'s: diagonalise the converged potential again
    on the ``k+q`` sphere, build the bare perturbations there, run
    ``solve_linter``'s fixed point, and assemble. What differs from ``Gamma``
    is only where things live, which is the subject of the module docstring.

    Returns a :class:`~defumat.response.phonon.Phonons` whose ``matrix`` is
    **complex**: ``D(q)`` is hermitian rather than symmetric, and only at
    ``q = 0`` (or at a wavevector where the crystal has inversion) does it
    collapse to a real matrix.

    **There is no** ``atoms=`` **here**, which is the first lever a large cell
    would reach for at ``Gamma`` (:func:`~defumat.response.phonon.dynamical_matrix`
    has it). It is left out rather than half-written: the response half of the
    assembly would index a subset's rows while the frozen half and
    :func:`ewald_dynamical_matrix` are whole-cell objects, and a partial
    dynamical matrix that silently mixed the two would be a plausible wrong
    answer rather than an error. Adding it means slicing all three together.
    """
    from defumat.response.phonon import Phonons
    from defumat.response.sternheimer import make_sternheimer

    cell = calculation.system.cell
    q_crystal = None if q_cartesian else np.asarray(q, dtype=float)
    require_a_two_sphere_regime(calculation, q_crystal)

    if q_cartesian:
        q_cart = np.asarray(q, dtype=float) * cell.tpiba
    else:
        q_cart = np.asarray(
            cell.k_to_cartesian(np.asarray(q, dtype=float))
        ) * cell.tpiba

    structure = calculation.system.structure
    positions = jnp.asarray(structure.positions)
    nat = structure.nat

    result = _GroundState(wavefunctions, eigenvalues, density, becsum)
    solver = make_sternheimer(calculation, result, threshold=threshold)
    potential = calculation.potential(density)

    kq, hamiltonians_kq, eigenvalues_kq, psi_kq = states_at_k_plus_q(
        calculation, potential.v_scf, q_cart,
        nbnd=nbnd or np.asarray(wavefunctions).shape[2],
    )
    two = TwoSphereSolver(solver, hamiltonians_kq, psi_kq, eigenvalues_kq, kq)

    bare = bare_displacements_at_q(calculation, kq, two, q_cart, positions)
    dpsi, drho, history, average, converged = self_consistent_response_at_q(
        calculation, kq, two, bare, density, q_cart,
        alpha_mix=alpha_mix, tr2=tr2, max_iterations=max_iterations,
        verbose=verbose,
    )

    matrix = (
        frozen_force_constants(calculation, solver, positions, density, q_cart)
        + response_force_constants(two, dpsi, bare, nat)
    )
    asymmetry = float(np.max(np.abs(matrix - matrix.conj().T)))
    matrix = 0.5 * (matrix + matrix.conj().T)

    frequencies, vectors = _diagonalize_at_q(matrix, np.asarray(structure.masses))
    return Phonons(
        matrix=matrix.reshape(nat, 3, nat, 3),
        frequencies=frequencies,
        eigenvectors=vectors,
        induced_density=np.asarray(drho),
        asymmetry=asymmetry,
        history=history,
        average_iterations=average,
        converged=converged,
    )


class _GroundState:
    """The four fields :func:`make_sternheimer` reads off an ``SCFResult``.

    A shim rather than a refactor: the solver's constructor takes a result
    object, and this entry point takes the four arrays directly, as
    :func:`~defumat.response.phonon.dynamical_matrix` does.
    """

    def __init__(self, wavefunctions, eigenvalues, density, becsum):
        self.wavefunctions = wavefunctions
        self.eigenvalues = eigenvalues
        self.density = density
        self.becsum = becsum
        self.occupations = None
        self.fermi_energy = None
