"""The velocity operator ``v = dH/dk``, by ``jvp`` at a frozen sphere.

For a local potential the velocity operator would be ``p``, and a plane-wave
code would need nothing: ``<k+G|p|k+G> = k+G``. A pseudopotential is not local,
and ``[V_NL, r] != 0`` -- QE hand-codes the correction in
``PW/src/commutator_Hx_psi.f90``. Rule D2 of ``PLAN.md`` says the whole of
``H(k)`` is built by differentiable JAX code from ``k`` precisely so that the
operator falls out of differentiating it, nonlocal contributions and all, with
nothing derived by hand. This module is that rule being cashed in.

**What carries ``k``, and what visibly does not.** Two terms only:
``|k+G|^2`` and ``vkb(k)`` -- and ``wfcU(k)`` when there is a Hubbard ``U``,
whose atomic orbitals live at ``k+G`` as the projectors do. The local potential
is a field on a real-space box that does not move with ``k``, so its tangent is
symbolically zero and the ``jvp`` never issues its FFTs.
:meth:`~pypresso.scf.driver.Calculation.at_kcart` is what rebuilds the two
traced arrays over shared everything-else.

**The frozen quantity is the coefficient vector, and it is the right one.** In
the periodic gauge ``H(k) = e^{-ik.r} H e^{ik.r}`` acts on the lattice-periodic
part ``u_nk``, whose expansion coefficients are exactly what is stored. Holding
them fixed while ``k`` moves is therefore differentiating ``H(k)`` at fixed
basis function, which is what the velocity operator *is*. What is held fixed
besides is which plane waves are in the sphere: that is a host-side decision
that cannot be traced, it is piecewise constant in ``k``, and on each piece the
frozen-sphere derivative is the exact one. The jump at the isolated ``k`` where
a plane wave crosses the cutoff is the Pulay error of a finite basis, the same
term :mod:`pypresso.forces.spiral` measures for ``dE/dq``.

**Ultrasoft and PAW are carried rather than refused, because ``S`` moves
too.**
``S(k) = 1 + sum |beta(k)> q <beta(k)|`` has the same ``k`` in it as ``H``, so
the band velocity is the *generalised* Hellmann-Feynman derivative

    d(eps_n)/dk_a = <psi_n| dH/dk_a - eps_n dS/dk_a |psi_n>,

which is ``commutator_Hx_psi``'s ultrasoft correction and comes from the same
``jvp`` -- :meth:`VelocityOperator.apply_s` is that second tangent. PAW adds
nothing to the differentiation and one thing to the *setup*: its one-centre
coefficients ``ddd_paw`` are built from ``becsum`` rather than from the density,
and since they multiply ``vkb(k)`` they belong to ``dH/dk`` as much as to ``H``.
They have to be handed in, and a PAW calculation without them raises rather than
returning the 2% error it would otherwise give.

**Nothing dense is ever formed.** ``dH/dk`` as a matrix is ``npw^2``, the same
reason a dense diagonalisation is a test fixture here and never a solver. One
``jvp`` per cartesian direction gives ``v_a|psi>`` for every band at every
k-point, and the matrix elements a Kubo sum needs are contractions of that
against the states already held.

**Memory.** One direction's tangent doubles the two traced arrays while it is
live: ``vkb`` is ``(nk, npwx, nkb)`` complex and ``|k+G|^2`` is ``(nk, npwx)``,
so the peak is one extra ``vkb``. The three directions are separate ``jvp``
calls for that reason -- a ``jacfwd`` over all three would hold three tangents
at once, and ``vkb`` is the largest ``k``-indexed array a calculation has after
the wavefunctions themselves.

**Degeneracies.** Nothing here differentiates an eigendecomposition (rule D4).
:meth:`VelocityOperator.band_velocities` is a diagonal expectation value, which
is wrong inside a degenerate manifold in the same way any diagonal element is --
the eigensolver's arbitrary mixing rotates it -- so what it reports there is one
particular basis of the manifold. :meth:`VelocityOperator.matrix_elements`
returns the whole block and is what a degeneracy-safe consumer contracts.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from pypresso.batching import map_k

__all__ = ["VelocityOperator", "BandVelocities", "band_velocities",
           "over_kpoints"]

#: The three cartesian directions, as tangents for ``k``.
_CARTESIAN = jnp.eye(3)


@dataclass
class BandVelocities:
    """``d(eps_n)/dk`` at every band and k-point, in Ry bohr.

    Rydberg atomic units have ``hbar = 1``, so the group velocity is this number
    outright; it is reported as the energy derivative rather than as a velocity
    because that is what is computed and what a finite difference of the band
    structure checks it against.
    """

    #: ``(nspin, nk, nbnd, 3)`` in Ry bohr. The spin axis is squeezed on the way
    #: out by :attr:`velocities`, following the convention of ``SCFResult``.
    velocities_by_spin: np.ndarray
    #: ``(nspin, nk, nbnd)`` in Ry -- the eigenvalues these belong to.
    eigenvalues_by_spin: np.ndarray
    nspin: int = 1

    @property
    def velocities(self) -> np.ndarray:
        """``(nk, nbnd, 3)``, or ``(2, nk, nbnd, 3)`` for a collinear run."""
        return (
            self.velocities_by_spin
            if self.nspin == 2
            else self.velocities_by_spin[0]
        )

    @property
    def speeds(self) -> np.ndarray:
        """``|d(eps)/dk|``, same shape as the eigenvalues."""
        return np.linalg.norm(self.velocities_by_spin, axis=-1)


class VelocityOperator:
    """``dH/dk`` and ``dS/dk`` at a fixed potential, applied to states.

    Built from the same three things :meth:`~pypresso.scf.driver.Calculation.
    hamiltonian` takes -- the self-consistent potential, PAW's one-centre
    coefficients and the Hubbard occupation matrix -- because the operator is
    the derivative of *that* Hamiltonian and of no other. The potential is a
    field on a box that does not move with ``k``, so freezing it is not an
    approximation: it is what ``dH/dk`` at fixed density means.
    """

    def __init__(self, calculation, v_scf, ddd_paw=None, ns=None, kcart=None):
        if calculation.spiral:
            raise NotImplementedError(
                "the velocity operator on a spin spiral is not implemented: the "
                "two spinor components sit on spheres centred at k + q/2 and "
                "k - q/2, so dH/dk moves both (see Calculation.at_spiral_q)"
            )
        if calculation.is_paw and ddd_paw is None:
            # The same rule the Hubbard branch below follows, and for the same
            # reason: a PAW Hamiltonian's nonlocal coefficients are
            # ``D^(0) + int V Q + ddd_paw``, the last of which comes from
            # ``becsum`` and cannot be rebuilt from the density. It multiplies
            # ``vkb(k)``, so it is part of ``dH/dk`` and not only of ``H`` --
            # leaving it out gives a velocity that is wrong by **2%** and looks
            # entirely ordinary (measured: 1.7e-2 Ry bohr against 8.7e-7 on
            # two-atom PAW silicon).
            raise ValueError(
                "dH/dk with a PAW pseudopotential needs the one-centre "
                "coefficients as well as the potential: pass ddd_paw = "
                "calculation.onecenter(scf_result.becsum)[1]. They are built "
                "from becsum, not from the density, and they multiply vkb(k), "
                "so they are part of the velocity operator"
            )
        if calculation.is_hubbard and ns is None:
            raise ValueError(
                "dH/dk with a Hubbard U needs the converged occupation matrix: "
                "pass ns = scf_result.ns. wfcU is built at k+G and therefore "
                "carries a velocity of its own, and leaving the term out gives "
                "a plausible operator that is missing the Hubbard shift"
            )
        self.calculation = calculation
        self.v_scf = v_scf
        self.ddd_paw = ddd_paw
        self.ns = None if ns is None else jnp.asarray(ns)
        # **Not always ``kpoints.cartesian(cell)``.** That expression reads
        # ``KPoints.coords``, which are cartesian in units of ``2 pi / alat``
        # and therefore do *not* move when the cell is strained -- so on a
        # calculation from :meth:`~pypresso.scf.driver.Calculation.at_strain` it
        # silently returns the undeformed k-points and this would be ``dH/dk``
        # at the wrong ``k``. Such a calculation records the ``kcart`` it was
        # built with and it is preferred here; ``kcart`` may also be given
        # outright, which is what a *traced* strain needs.
        if kcart is None:
            kcart = getattr(calculation, "_kcart", None)
        if kcart is None:
            kcart = calculation.system.kpoints.cartesian(calculation.system.cell)
        self.kcart = jnp.asarray(kcart)

    # -- the two operators ------------------------------------------------

    def apply(self, psi: jnp.ndarray, direction) -> jnp.ndarray:
        """``dH/dk_a |psi>`` for a cartesian ``direction``.

        ``psi`` is ``(nspin, nk, nbnd, ndim)`` -- the shape ``SCFResult.
        wavefunctions`` has -- and the result matches it. ``direction`` is a
        cartesian 3-vector in the reciprocal-space basis ``k`` is written in
        (1/bohr); it need not be normalised, and the result is linear in it.
        """
        return self._tangent(psi, direction, overlap=False)

    def apply_s(self, psi: jnp.ndarray, direction) -> jnp.ndarray:
        """``dS/dk_a |psi>``: zero for a norm-conserving dataset.

        ``S`` is built from the same ``vkb(k)`` the nonlocal potential is, so it
        carries a velocity whenever ``qq`` is not ``None``. Skipping it is what
        would make an ultrasoft band velocity wrong by the norm's own motion.
        """
        return self._tangent(psi, direction, overlap=True)

    def projectors(self, direction) -> jnp.ndarray:
        """The projectors' derivative **about their own atom**, ``(nk, npwx, nkb)``.

        ``gen_us_dj`` and ``gen_us_dy``, which ``commutator_Hx_psi`` combines by
        hand into ``dvkb1 + dvkb (k+G).vpol/|k+G|`` -- the angular derivative
        plus the radial one along the unit vector -- and here the tangent of
        ``vkb`` under the same ``jvp`` everything else in this module uses, so
        the split into angular and radial never has to be made.

        **With one term taken back out, and it is a trap.** ``vkb`` carries the
        structure factor ``e^{-i(k+G).tau}``, so the true ``d(vkb)/dk_a``
        contains ``-i tau_a vkb``; QE's two routines differentiate the radial and
        angular parts and leave the structure factor alone, so what they build is
        the derivative *about the atom's own centre*. Everywhere else that
        distinction is invisible, because a projector always appears as
        ``|beta_i> D_ij <beta_j|`` and the two ``tau`` terms cancel between the
        ket and the bra -- which is why the velocity operator itself never had to
        care. Here a *single* projector derivative is contracted with a state,
        the cancellation does not happen, and the ``tau`` term is the difference
        between the right answer and one that is 2% too large.

        It is wanted only by an ultrasoft electric field, whose position operator
        carries ``<d(beta)/dk|psi>`` beside the augmentation dipole, and the two
        are on the same convention: ``dpqq`` is also about the atom's centre.
        """
        tangent = jnp.broadcast_to(
            jnp.asarray(direction, dtype=self.kcart.dtype), self.kcart.shape
        )

        def build(kcart):
            moved = self.calculation.at_kcart(kcart)
            return moved.projectors.vkb

        _, derivative = jax.jvp(build, (self.kcart,), (jnp.asarray(tangent),))

        projectors = self.calculation.projectors
        if projectors.nkb == 0:
            return derivative
        # **``tau`` stays traced.** This was ``np.asarray(positions)``, which
        # reads the atoms off the host and so silently drops them from any
        # derivative taken *through* this expression -- and a third derivative
        # in the displacement coordinate takes exactly that one (``PLAN.md``
        # P43). The channel index is static and stays a Python list.
        positions = jnp.asarray(self.calculation.system.structure.positions)
        along = positions @ jnp.asarray(  # tau . e_a, bohr
            direction, dtype=positions.dtype
        )
        shift = along[np.asarray(list(projectors.atom_of_channel))]
        return derivative + 1j * shift * projectors.vkb

    def both(self, psi: jnp.ndarray, direction):
        """``(dH/dk_a |psi>, dS/dk_a |psi>)`` from **one** ``jvp``.

        The two tangents share the whole cost -- rebuilding ``|k+G|^2`` and
        ``vkb(k)`` as differentiable functions of ``k`` -- so a band velocity,
        which needs both, asks for them together rather than paying for the
        rebuild twice per direction.
        """
        return self._tangent(psi, direction, overlap=None)

    def _tangent(self, psi, direction, overlap):
        psi = jnp.asarray(psi)
        tangent = jnp.broadcast_to(
            jnp.asarray(direction, dtype=self.kcart.dtype), self.kcart.shape
        )
        _, out = jax.jvp(
            lambda kc: self._operator(psi, kc, overlap),
            (self.kcart,),
            (jnp.asarray(tangent),),
        )
        return out

    def _operator(self, psi, kcart, overlap):
        """``H|psi>``, ``S|psi>``, or both, at every k-point, as a function of ``kcart``.

        ``overlap = None`` returns the pair, which is what
        :meth:`both` differentiates in one pass.
        """
        moved = self.calculation.at_kcart(kcart)
        hubbard = (
            None if self.ns is None else moved.hubbard_terms(self.ns)[2]
        )
        hamiltonians = moved.hamiltonian(self.v_scf, self.ddd_paw, hubbard)
        batch = self.calculation.k_batch

        def applied(want_overlap):
            return jnp.stack([
                over_kpoints(ham, psi[spin], batch, want_overlap)
                for spin, ham in enumerate(hamiltonians)
            ])

        if overlap is None:
            return applied(False), applied(True)
        return applied(overlap)

    # -- what is built from them ------------------------------------------

    def matrix_elements(self, psi: jnp.ndarray) -> jnp.ndarray:
        """``<psi_m| dH/dk_a |psi_n>``, ``(3, nspin, nk, nbnd, nbnd)`` in Ry bohr.

        The whole block, not its diagonal, because this is what survives a
        degenerate manifold: a Kubo sum contracts off-diagonal elements and the
        eigensolver's arbitrary rotation inside a manifold cancels between the
        two factors (rule D4).
        """
        psi = jnp.asarray(psi)
        return jnp.stack([
            jnp.einsum("skmg,skng->skmn", psi.conj(), self.apply(psi, axis))
            for axis in _CARTESIAN
        ])

    def band_velocities(self, psi, eigenvalues) -> BandVelocities:
        """``d(eps_n)/dk`` by the generalised Hellmann-Feynman theorem.

        ``<psi_n| dH/dk - eps_n dS/dk |psi_n>``. The second term is identically
        zero for a norm-conserving dataset and is not skipped for one -- the
        ``jvp`` is what decides that, not a branch here.
        """
        psi = jnp.asarray(psi)
        eigenvalues = jnp.asarray(eigenvalues)
        if eigenvalues.ndim == 2:  # (nk, nbnd) -- the squeezed unpolarized shape
            eigenvalues = eigenvalues[None]

        columns = []
        for axis in _CARTESIAN:
            # One ``jvp`` for the pair: the projector rebuild is the whole cost
            # and both tangents come out of the same one.
            velocity, overlap = self.both(psi, axis)
            dh = jnp.einsum("skng,skng->skn", psi.conj(), velocity)
            ds = jnp.einsum("skng,skng->skn", psi.conj(), overlap)
            columns.append(jnp.real(dh) - eigenvalues * jnp.real(ds))
        return BandVelocities(
            velocities_by_spin=np.asarray(jnp.stack(columns, axis=-1)),
            eigenvalues_by_spin=np.asarray(eigenvalues),
            nspin=self.calculation.nspin,
        )


def over_kpoints(hamiltonian, states, batch, overlap: bool = False):
    """``H|psi>`` (or ``S|psi>``) at every k-point, through the batching dial.

    ``Hamiltonian.apply`` takes a k *index* rather than a slice, so the mapped
    quantity is the index and the states are gathered inside -- which is how
    :mod:`pypresso.solvers.davidson` walks the same axis.
    """
    apply = hamiltonian.apply_s if overlap else hamiltonian.apply
    indices = jnp.arange(states.shape[0])
    return map_k(lambda ik: apply(states[ik], ik), indices, batch=batch)


def band_velocities(calculation, result, kpoints=None, nbnd=None,
                    conv_thr: float = 1.0e-6, k_batch="default") -> BandVelocities:
    """``d(eps)/dk`` for a converged run, in one call.

    ``result`` is an :class:`~pypresso.scf.driver.SCFResult`; the potential and
    the states are taken from it. When ``kpoints`` is given the velocities are
    computed there instead -- a band path, typically -- which is an NSCF
    diagonalisation followed by the same operator, and ``nbnd`` then says how
    many bands that diagonalisation resolves. A path is usually drawn with more
    bands than the ground state carried, so it is a parameter rather than the
    ground state's count; on the SCF's own k-points there is nothing to
    re-diagonalise and it does not apply.

    ``conv_thr`` is that diagonalisation's threshold, and it is worth setting
    rather than leaving at the default: a band velocity inside a **degenerate
    multiplet** is not a property of a band at all, so where two bands touch --
    which along silicon's Lambda axis is most of the path -- a loosely converged
    eigensolver mixes them differently from a tight one, and the per-band
    velocities then differ by order one while the multiplet's *set* of them does
    not. Measured: 1e-6 against 1e-12 on that path moves an individual band's
    velocity by 0.5 Ry bohr.
    """
    from pypresso.workflows.nscf import fixed_density_states

    if kpoints is None and nbnd is not None:
        raise ValueError(
            "nbnd applies to the NSCF diagonalisation this does at new "
            "kpoints, and none was asked for: on the ground state's own "
            "k-points the velocities come off the states it converged. Pass "
            "kpoints=, or raise nbnd on the SCF itself"
        )

    if kpoints is not None:
        # The whole mixed state crosses, not just ``ns``. ``becsum`` for a PAW
        # dataset, ``tau`` for a meta-GGA and the converged ``field`` /
        # ``field_scale`` are properties of the *states* and cannot be rebuilt
        # from the density this hands over -- forwarding one of them and not the
        # rest is the defect P38 closed in ``run_dos`` and ``run_pdos``, in a
        # third place.
        calculation, _, eigenvalues, psi = fixed_density_states(
            result.system, calculation.pseudos, result.density,
            kpoints=kpoints, nbnd=nbnd, conv_thr=conv_thr, k_batch=k_batch,
            ns=result.ns, tau=getattr(result, "tau", None),
            becsum=result.becsum or (),
            field=result.magnetic_field, field_scale=result.field_scale,
        )
        eigenvalues = jnp.asarray(eigenvalues)
    else:
        eigenvalues = jnp.asarray(result.eigenvalues)
        psi = result.wavefunctions

    potential = calculation.potential(result.density)
    # PAW's one-centre coefficients come from ``becsum``, which the result
    # carries for exactly this reason (it is part of the mixed state, not a
    # function of the density).
    _, ddd_paw = calculation.onecenter(result.becsum, getattr(result, "meta_c", None))
    operator = VelocityOperator(calculation, potential.v_scf, ddd_paw, result.ns)
    return operator.band_velocities(psi, eigenvalues)
