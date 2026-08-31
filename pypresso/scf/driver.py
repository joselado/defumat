"""The self-consistent field loop.

Following ``PW/src/electrons.f90``: from a starting density, build the
potential, diagonalise the Hamiltonian, rebuild the density from the occupied
states, and mix -- until the density stops changing.

The loop itself stays in Python because its termination test depends on the
values being computed (rule: data-dependent control flow does not belong inside
``jit``). The work inside one iteration is what gets compiled.

The total energy is assembled exactly as QE prints it, term by term, because
comparing the decomposition rather than only the total is what localises an
error to one physical contribution:

    E = (eband + deband) + E_Hartree + E_xc + E_Ewald [+ smearing (-TS)]

``eband`` is the sum of occupied eigenvalues, which double-counts the Hartree
and exchange-correlation energies; ``deband = -int rho v_scf`` removes exactly
that double counting, so their sum is the kinetic plus external-potential
energy that QE labels "one-electron contribution".

**Spin.** With ``nspin = 2`` the density, the potential, ``becsum``, ``D_ij``,
the eigenvalues and the wavefunctions all grow a leading channel axis, and one
SCF iteration diagonalises a *different Hamiltonian per channel* -- same kinetic
term and same local pseudopotential, different self-consistent potential and
different ``D_ij``. QE arranges this by storing both channels in a single
k-point list of length ``2 nks`` with ``isk(ik)`` saying which; here the channel
is an explicit axis and the loop over it is a Python loop, since ``nspin`` is
static and two Hamiltonians are two compiled kernels either way. What must not
change is that ``k`` stays the leading *independent* axis inside each channel,
because that is the axis the batching and the eventual sharding use.

**Noncollinear.** With ``nspin = 4`` there is again *one* Hamiltonian, not two:
the spin channels are not separate problems but the two components of a single
spinor, and the loop below therefore runs once. What grows is the state itself
-- ``2 npwx`` numbers instead of ``npwx``
(:class:`~pypresso.hamiltonian.noncollinear.SpinorHamiltonian`) -- and, when the
calculation carries a magnetization, the density and the potential, which become
``(n, m_x, m_y, m_z)``. The spin-channel axis of every array is kept at length
one so that occupations, mixing and the result objects need no third case; it is
``npol`` and ``nspin_mag`` that carry the noncollinear shapes, not ``nspin``.
"""

from __future__ import annotations

import copy
import dataclasses
import warnings
from dataclasses import dataclass, field
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from pypresso.basis.builder import Basis, build_basis
from pypresso.basis.planewaves import build_plane_wave_basis
from pypresso.basis.interpolate import to_dense, to_smooth
from pypresso.basis.sticks import build_sticks
from pypresso.basis.fft import g_to_r, r_to_g
from pypresso.hamiltonian.noncollinear import SpinorHamiltonian
from pypresso.hamiltonian.operator import Hamiltonian
from pypresso.pseudo.atomic import atomic_wavefunctions
from pypresso.paw.onecenter import build_paw
from pypresso.hubbard.energy import (
    coefficients_from_setup,
    hubbard_energy,
    hubbard_potential,
    ns_ddot,
)
from pypresso.hubbard.manifold import build_hubbard_setup
from pypresso.hubbard.occupations import (
    adjust_ns,
    build_ns_symmetry,
    initial_ns,
    ns_shape,
    occupation_matrix,
)
from pypresso.hubbard.operator import HubbardTerm, block_potential
from pypresso.hubbard.projectors import build_hubbard_projectors
from pypresso.paw.symmetry import build_becsum_symmetry
from pypresso.pseudo.augmentation import build_augmentation
from pypresso.pseudo.potentials import (
    combine_species,
    species_atomic_charge,
    species_core_charge,
    species_local_potential,
    starting_charge,
)
from pypresso.pseudo.projectors import build_projector_core, projector_channels
from pypresso.pseudo.upf import Pseudopotential
from pypresso.pseudo.spinorbit import becsum_transform, build_spin_orbit
from pypresso.batching import map_k, resolve_k_batch
from pypresso.scf.continuation import ContinuedState, continued_state
from pypresso.scf.density import (
    becsum,
    kinetic_energy_density,
    spinor_becsum,
    spinor_kinetic_energy_density,
    spinor_sum_band,
    sum_band,
)
from pypresso.scf.ewald import build_ewald
from pypresso.scf.mixing import PRECONDITIONED, get_mixer, kerker_preconditioner
from pypresso.scf.residual import make_residual
from pypresso.scf.solvers import get_scf_solver
from pypresso.scf.occupations import (
    fixed_occupations,
    input_occupations,
    smeared_occupations,
    smearing_entropy,
    spin_electron_counts,
    tetrahedra_for,
    tetrahedron_occupations_spin,
)
from pypresso.scf.fields import MagneticField, constraint_targets
from pypresso.scf.locals import build_local_regions
from pypresso.scf.potential import (
    Potential,
    as_potential_components,
    fixed_quantization_axis,
    scf_accuracy,
    v_of_rho,
)
from pypresso.xc.mgga import thomas_fermi_tau
from pypresso.xc.functional import resolve_functional
from pypresso.solvers import get_eigensolver
from pypresso.solvers.davidson import ETHR_MIN, starting_vectors
from pypresso.solvers.subspace import rayleigh_ritz
from pypresso.system.builder import System
from pypresso.system.kpoints import KPoints
from pypresso.system.spiral import spiral_kcart, spiral_kpoints
from pypresso.system.symmetry import (
    apply_symmetry_maps,
    atom_mapping,
    cartesian_rotations,
    magnetization_signs,
    symmetry_maps,
    symmetrize_atom_displacement_density,
    symmetrize_tensor_density,
    symmetrize_magnetization,
    symmetrize_vector_density,
)
from pypresso.units import RY_TO_EV
from pypresso.vdw.registry import build_vdw_correction, vdw_options

__all__ = ["SCFResult", "Calculation", "run_scf", "default_nbnd"]


# The iteration body is compiled in three units rather than one, because the
# occupation weights that separate them are decided on the host: the Fermi level
# is a bisection whose bracket is data. Everything on either side of that
# decision is one compiled kernel each, so a whole SCF iteration costs three
# dispatches instead of the hundreds the eager version issued.

@jax.jit
def _field_potential(field, rho_r, cell, scale):
    """``add_bfield``: the potential of the external fields and the constraint."""
    return field.potential(rho_r, cell, scale)


#: ``quantization_axis`` is a fixed three-vector or ``None``, so it is static:
#: it comes from the *input* magnetization and cannot change during a run.
_potential_of_rho = jax.jit(v_of_rho, static_argnums=(6,))
_accuracy = jax.jit(scf_accuracy)


#: Where ``ethr`` starts, from ``PW/src/setup.f90``: the starting potential is a
#: superposition of atomic charges and is nowhere near self-consistent, so there
#: is nothing to be gained by diagonalising against it accurately.
ETHR_INIT = 1.0e-2


@partial(jax.jit, static_argnames=("grid",))
def _symmetrize(rho_r, fft_index, grid, maps):
    """Impose the crystal symmetry on a real-space density, in one kernel.

    Per channel: a collinear symmetry operation acts on positions and leaves
    the spin alone, so it symmetrises the two densities independently. (QE
    symmetrises in the ``(up, down)`` representation too -- ``sym_rho`` runs
    before ``rhoz_or_updw`` converts.)
    """
    permutations, phases = maps

    def channel(rho):
        rho_g = apply_symmetry_maps(r_to_g(rho, fft_index), permutations, phases)
        return jnp.real(g_to_r(rho_g, fft_index, grid))

    return jax.vmap(channel)(rho_r)


@partial(jax.jit, static_argnames=("grid",))
def _symmetrize_noncollinear(rho_r, fft_index, grid, maps, rotations):
    """``sym_rho``'s ``nspin = 4`` branch: a scalar and an axial vector field.

    The charge is symmetrised exactly as it is in any other regime; the three
    magnetization components are *not* three scalars -- a symmetry operation
    rotates them into each other, with a sign for the axial character and
    another for time reversal. Doing them independently is not a worse average
    but a different, wrong, symmetry: on bcc iron with the moment along ``x`` it
    keeps only the component the operations happen to leave alone.
    """
    permutations, phases = maps
    charge_g = apply_symmetry_maps(r_to_g(rho_r[0], fft_index), permutations, phases)
    magnetization_g = jnp.stack([r_to_g(rho_r[i], fft_index) for i in (1, 2, 3)])
    magnetization_g = symmetrize_magnetization(
        magnetization_g, permutations, phases, rotations
    )
    stacked = jnp.concatenate([charge_g[None], magnetization_g], axis=0)
    return jax.vmap(lambda rho_g: jnp.real(g_to_r(rho_g, fft_index, grid)))(stacked)


@partial(jax.jit, static_argnames=("grid", "k_batch"))
def _density_of_bands(psi, fft_index, grid, weights, cell, k_batch):
    """``sum_band`` on the smooth grid, in one kernel.

    The symmetrisation used to be fused in here. It cannot be any more: it acts
    on the dense grid, and with a double grid the density has to be lifted there
    first.
    """
    return sum_band(psi, fft_index, grid, weights, cell, k_batch)


@partial(jax.jit, static_argnames=("grid", "k_batch"))
def _kinetic_of_bands(psi, fft_index, grid, weights, cell, kplusg, k_batch):
    """``sum_band``'s meta-GGA branch on the smooth grid, in one kernel."""
    return kinetic_energy_density(psi, fft_index, grid, weights, cell, kplusg, k_batch)


@partial(jax.jit, static_argnames=("grid", "nspin_mag", "k_batch"))
def _spinor_kinetic_of_bands(psi, fft_index, grid, weights, cell, kplusg,
                             nspin_mag, k_batch):
    """The same for spinors: ``tau`` on the Pauli basis."""
    return spinor_kinetic_energy_density(
        psi, fft_index, grid, weights, cell, kplusg, nspin_mag, k_batch
    )


@jax.jit
def _paw_deband(ddd_paw, augmentation, becsum_):
    """``sum_s sum_a sum_ij ddd_paw^a_ij becsum^a_ij``, out of the block matrix.

    Reading the blocks back out of the assembled ``(nspin, nkb, nkb)`` matrix
    rather than keeping them separately is deliberate: it is the *same* array
    the Hamiltonian used, so the two cannot drift apart.
    """
    total = jnp.asarray(0.0)
    nspin = ddd_paw.shape[0]
    for values, atoms in zip(becsum_, augmentation.species_atoms):
        if values is None or not atoms:
            continue
        nh = values.shape[-1]
        for n, atom in enumerate(atoms):
            start = augmentation.channel_offsets[atom]
            block = jax.lax.dynamic_slice(
                ddd_paw, (0, start, start), (nspin, nh, nh)
            )
            total = total + jnp.sum(block * values[:, n])
    return total


def _mix(mixer, rho, rho_out, becsum_in, becsum_out, ns_in=None, ns_out=None):
    """One mixing step over the density and, for PAW and DFT+U, its companions.

    All of them are packed into a single vector so that the extrapolation
    coefficients Anderson computes from the density residual are applied to
    every part -- they are views of one fixed point, and mixing them with
    different histories makes the iteration inconsistent rather than merely
    slower. ``mix_rho.f90`` says the same thing about ``becsum``, and QE's
    ``mix_type`` carries ``ns`` in the same structure for the same reason: the
    Hubbard potential is built from ``ns``, so an unmixed ``ns`` would drive the
    Hamiltonian from the *output* of the previous step while the density came
    from the mixed one.
    """
    flat = [np.asarray(rho).ravel()]
    flat_out = [np.asarray(rho_out).ravel()]
    for old, new in zip(becsum_in, becsum_out):
        if old is None:
            continue
        flat.append(np.asarray(old).ravel())
        flat_out.append(np.asarray(new).ravel())
    if ns_in is not None:
        flat.append(np.asarray(ns_in).ravel())
        flat_out.append(np.asarray(ns_out).ravel())

    mixed = mixer.mix(np.concatenate(flat), np.concatenate(flat_out))

    offset = rho.size
    rho_mixed = jnp.asarray(mixed[:offset].reshape(rho.shape))
    becsum_mixed = []
    for old in becsum_in:
        if old is None:
            becsum_mixed.append(None)
            continue
        becsum_mixed.append(jnp.asarray(mixed[offset : offset + old.size].reshape(old.shape)))
        offset += old.size
    ns_mixed = None
    if ns_in is not None:
        ns_mixed = jnp.asarray(mixed[offset : offset + ns_in.size].reshape(ns_in.shape))
    return rho_mixed, tuple(becsum_mixed), ns_mixed


@jax.jit
def _paw_onecenter(paw, becsum_, meta_c=None, axis=None):
    """``PAW_potential``: the one-centre energy and its ``ddd``, in one kernel."""
    return paw.energy_and_coefficients(becsum_, meta_c, axis)


@jax.jit
def _newd(potential, fft_index, dij, augmentation):
    """``D^(0)_ij + int V_eff Q_ij``, as the block matrix the Hamiltonian takes.

    One block matrix per spin channel: ``newq_acc`` integrates ``v%of_r(:,is)``
    against the same ``Q_ij`` for each ``is``, so the augmentation charge is
    shared and only the potential differs.
    """
    blocks = [
        augmentation.block_matrix(augmentation.integrals(r_to_g(channel, fft_index)))
        for channel in potential
    ]
    return dij[None] + jnp.stack(blocks)


@partial(jax.jit, static_argnames=("grid",))
def _addusdens(rho_r, fft_index, grid, augmentation, becsum_):
    """The augmentation charge, added to the density on the dense grid.

    Per channel, from that channel's ``becsum``: ``addusdens`` is called on
    ``rho%of_g(:,:)`` while the density is still in the ``(up, down)``
    representation, so each channel gets its own augmentation charge.
    """
    charge = jnp.stack([
        augmentation.charge(
            tuple(None if b is None else b[spin] for b in becsum_)
        )
        for spin in range(rho_r.shape[0])
    ])
    return rho_r + jnp.real(g_to_r(charge, fft_index, grid))


@partial(jax.jit, static_argnames=("nbnd", "k_batch"))
def _rotate_all(hamiltonian, vectors, nbnd: int, k_batch):
    """Rayleigh-Ritz at ``k_batch`` k-points at a time.

    ``wfcinit`` does this inside its own ``DO ik`` loop, and the working set is
    the same one the eigensolver has: the atomic orbitals of every k-point, plus
    ``H`` applied to them.
    """
    return map_k(
        lambda pair: rayleigh_ritz(hamiltonian, pair[0], pair[1], nbnd),
        (jnp.arange(hamiltonian.nk), vectors),
        batch=resolve_k_batch(k_batch),
    )


@partial(jax.jit, static_argnames=("grid", "nspin_mag", "k_batch"))
def _spinor_density_of_bands(psi, fft_index, grid, weights, cell, nspin_mag, k_batch):
    """``sum_band`` for spinors, in one kernel."""
    return spinor_sum_band(psi, fft_index, grid, weights, cell, nspin_mag, k_batch)


@jax.jit
def _newd_noncollinear(deeq_components, dvan_so, fcoef):
    """``newd_so``/``newd_nc``: the scalar integrals as a 2x2 spin matrix.

    ``deeq_components`` is ``(nspin_mag, nkb, nkb)`` -- one integral of the
    augmentation charge against each component of the potential -- and the
    recombination is the same one the local potential undergoes:
    ``sum_a d_a sigma^a``, then sandwiched between the spin-orbit coefficients.

    ``fcoef`` is block diagonal over atoms, so the full ``nkb x nkb`` matrix
    products below never mix two atoms and the sandwich is the per-species one
    of ``newd_so_acc`` written once. For a scalar-relativistic species ``fcoef``
    is the identity on each diagonal spin block, and the sandwich collapses to
    the plain recombination of ``newd_nc_acc``.
    """
    nspin_mag = deeq_components.shape[0]
    charge = deeq_components[0]
    zero = jnp.zeros_like(charge)
    if nspin_mag == 1:
        blocks = jnp.stack([jnp.stack([charge, zero]), jnp.stack([zero, charge])])
    else:
        mx, my, mz = deeq_components[1], deeq_components[2], deeq_components[3]
        blocks = jnp.stack([
            jnp.stack([charge + mz, mx - 1j * my]),
            jnp.stack([mx + 1j * my, charge - mz]),
        ])
    blocks = blocks.astype(fcoef.dtype)
    return dvan_so + jnp.einsum("asij,stjk,tbkl->abil", fcoef, blocks, fcoef, optimize=True)


@jax.jit
def _noncollinear_magnetization(rho_r, volume):
    """``(m_x, m_y, m_z, |m|)`` integrated over the cell, in Bohr magnetons.

    ``report_mag`` prints the three components of the total moment and, as in
    the collinear case, the integral of the modulus. The two say different
    things: an antiferromagnet or a spiral integrates to zero and still has a
    large absolute moment, and with spin-orbit coupling the *direction* of the
    total moment is a result rather than an input, which is why all three
    components are worth printing.
    """
    magnetization = rho_r[1:]
    scale = volume / magnetization[0].size
    modulus = jnp.sqrt(jnp.sum(magnetization**2, axis=0))
    return jnp.concatenate([
        scale * jnp.sum(magnetization, axis=(1, 2, 3)),
        jnp.asarray([scale * jnp.sum(modulus)]),
    ])


@jax.jit
def _magnetization(rho_r, volume):
    """``(total, absolute)`` magnetization in Bohr magnetons per cell.

    ``report_mag``/``electrons.f90`` print both: the integral of ``rho_up -
    rho_dw`` and the integral of its absolute value. They differ whenever the
    magnetization changes sign inside the cell -- an antiferromagnet has zero
    total and a large absolute moment -- so printing only the first would hide
    exactly the states LSDA exists to describe.
    """
    magnetization = rho_r[0] - rho_r[1]
    scale = volume / magnetization.size
    return jnp.stack([
        scale * jnp.sum(magnetization),
        scale * jnp.sum(jnp.abs(magnetization)),
    ])


@jax.jit
def _iteration_scalars(eigenvalues, weights, rho_in, rho_out, v_scf, volume):
    """``(eband, deband, residual)`` as one array, so the loop syncs once.

    Each ``float()`` on a device array is a host round trip; computing the three
    together and transferring them in one go is the difference between one
    synchronisation per iteration and one per printed number.

    ``deband`` integrates the **output** density against the **input**
    potential, which is where ``delta_e()`` sits in ``electrons.f90``'s
    ordering -- see the note in :func:`run_scf`.
    """
    eband = jnp.sum(weights * eigenvalues)
    # ``delta_e``: one integral per channel of that channel's own density
    # against its own potential. QE writes the same number in the (total,
    # magnetization) representation as ``0.5[(v_up + v_dw) rho + (v_up - v_dw)
    # m]``, which is the same contraction rearranged.
    deband = -volume / rho_out[0].size * jnp.sum(rho_out * v_scf)
    residual = jnp.max(jnp.abs(rho_out - rho_in))
    return jnp.stack([eband, deband, residual])


def _without_gamma_storage(system: System) -> System:
    """Turn ``K_POINTS gamma`` into an explicit single k-point at the origin.

    ``ggen`` keeps only half the G sphere when the calculation is at k = 0,
    because a real wavefunction has ``c(-G) = conj(c(G))``. That storage is
    generated here (``basis/gvectors.py``) but it is not *consumed* anywhere:
    ``vloc_psi`` would need QE's ``vloc_psi_gamma`` packing, the eigensolver
    QE's real ``regterg`` overlaps, ``calbec``/``addusdens``/``newd`` their
    ``fact = 2`` doubling, and the symmetry maps a way to follow a rotation that
    leaves the stored half. None of that exists, so running a gamma-only basis
    through the SCF would be silently wrong (where it does not simply fail in
    ``symmetry_maps``).

    The substitution below is exact rather than approximate: the same cell at
    the same cutoffs has the same FFT dimensions and the same physics, and the
    full sphere is what every routine here already expects. What it costs is
    the factor of two in storage and transforms that the trick exists to save,
    which is a performance matter and not a correctness one.

    The k-point weights are carried over rather than rebuilt, because
    :meth:`KPoints.from_cartesian` renormalises and reapplies ``DEGSPIN`` --
    which would undo the halving an ``nspin = 2`` run has already applied.
    """
    kpoints = system.kpoints
    if not kpoints.gamma_only:
        return system
    warnings.warn(
        "K_POINTS gamma asks for the half-sphere storage of the gamma-point "
        "trick, which is not implemented; running at an explicit k = 0 with the "
        "full G sphere instead. The result is the same, the cost is twice the "
        "plane waves",
        stacklevel=3,
    )
    replacement = KPoints(
        coords=kpoints.coords,
        weights=kpoints.weights,
        gamma_only=False,
        precision=kpoints.precision,
    )
    return eqx.tree_at(lambda s: s.kpoints, system, replacement)


def _meta_c(potential):
    """The Tran-Blaha ``c`` a potential was built with, or ``None``.

    ``Potential.meta_c`` is ``0`` for every functional that is not a meta-GGA,
    and the PAW one-centre terms want ``None`` there rather than a coefficient
    that means nothing.
    """
    c = potential.meta_c
    return None if np.ndim(c) == 0 and float(c) == 0.0 else c


def _starting_tau(rho, calculation) -> jnp.ndarray:
    """``potinit.f90``'s Thomas-Fermi ``rho%kin_r``, per spin channel, Ry.

    ``(3/5)(3 pi^2)^(2/3) rho^(5/3)`` for one channel, and the spin-scaled form
    for two. The density and ``tau`` are in the *same* layout here -- the total
    when unpolarized, ``(up, down)`` when not -- so no conversion happens, which
    is worth stating because QE's do not: ``potinit.f90`` carries a comment
    saying "for LSDA rho is (tot,magn), rho_kin is (up,down)" and converts
    between them at this exact point. This package stores ``(up, down)`` for
    both (:func:`pypresso.scf.potential.with_core`), so the conversion would be
    a bug rather than a transcription.
    """
    rho = jnp.real(jnp.asarray(rho))
    # ``nspin_mag`` and not ``nspin``: the spin scaling belongs to how many
    # channels the *density* has, and a nonmagnetic spin-orbit run has one of
    # them while its ``nspin`` is 4. Keying off ``nspin`` applied the two-channel
    # form ``|2 rho|^(5/3) / 2`` to a total density and started such a run from a
    # guess too large by ``2^(2/3)``.
    return thomas_fermi_tau(rho, 1 if calculation.nspin_mag == 1 else 2)


def next_ethr(ethr: float, accuracy: float, nelec: float, iteration: int) -> float:
    """QE's diagonalisation-threshold schedule (``PW/src/electrons.f90``).

    The eigenvalues never need to be more accurate than the density they are
    computed from. ``ethr`` therefore tracks ``dr2``, the estimated error in the
    density, and only tightens -- 1e-2 while the density is still wrong in the
    second decimal, 1e-13 by the time it is converged.

    Three details are QE's and matter: the threshold is *reset* to 1e-2 at the
    second iteration rather than carried over from the first, it can only ever
    decrease (``MIN``), and it is floored at 1e-13 because an iterative
    diagonalisation asked for more than that becomes unstable rather than more
    accurate.
    """
    if iteration <= 1:
        return ethr
    if iteration == 2:
        ethr = ETHR_INIT
    return max(min(ethr, 0.1 * accuracy / max(1.0, nelec)), ETHR_MIN)


def default_nbnd(
    nelec: float, occupations: str, nelup=None, neldw=None, noncolin: bool = False
) -> int:
    """QE's default band count (``PW/src/setup.f90``).

    An insulator needs exactly the occupied bands; a smeared calculation needs
    20% more so there are empty states for the Fermi level to sit among.

    ``IF (noncolin) nbnd = 2 * nbnd``, applied last and to whichever count the
    lines above arrived at: a spinor band holds one electron rather than two, so
    a noncollinear run of the same crystal needs twice as many. QE says why in a
    comment -- "bands are NOT twofold degenerate" -- and the doubling is not
    optional even when spin-orbit coupling leaves them degenerate anyway.

    With two channels the count is ``MAX(nelec/degspin, nelup, neldw)`` and
    ``degspin`` is 1, so what decides it is the *fuller* channel: eight
    electrons split 5/3 need five bands, not four. Getting this wrong is
    immediately visible -- QE prints "number of Kohn-Sham states" -- but only if
    it is looked at, and a run with too few bands converges to a wrong energy
    rather than failing.
    """
    occupied = int(round(nelec / 2.0))
    if nelup is not None:
        occupied = max(occupied, int(round(nelup)), int(round(neldw)))
    degeneracy = 2 if noncolin else 1
    if occupations == "fixed":
        return degeneracy * max(occupied, 1)
    if nelup is not None:
        return max(
            int(round(1.2 * nelec / 2.0)),
            int(round(1.2 * nelup)),
            int(round(1.2 * neldw)),
            occupied + 4,
        )
    return degeneracy * max(int(round(1.2 * nelec / 2.0)), occupied + 4)


@dataclass
class SCFResult:
    """Everything an SCF produces, with the energy split QE's way."""

    converged: bool
    iterations: int
    total_energy: float
    energy_terms: dict
    #: ``(nk, nbnd)`` for an unpolarized run and ``(2, nk, nbnd)`` for LSDA, in
    #: Ry. The spin axis is *squeezed away* when there is only one channel, so
    #: that everything written against the unpolarized shape keeps working and
    #: a polarized result cannot be mistaken for one. Use
    #: :attr:`eigenvalues_by_spin` where the axis must always be there.
    eigenvalues: np.ndarray
    occupations: np.ndarray  # same shape as eigenvalues -- the weights wg
    wavefunctions: jnp.ndarray  # (nspin, nk, nbnd, npwx)
    density: jnp.ndarray  # (nspin, n1, n2, n3), electrons/bohr^3
    potential: jnp.ndarray  # (nspin, n1, n2, n3), Ry -- the total local potential
    #: ``V[rho_out] - V[rho_in]`` at the last iteration -- QE's ``vnew``, the
    #: self-consistency the run did not reach. Zero to the extent that it
    #: converged, and the only input to ``force_corr``.
    potential_change: jnp.ndarray | None = None
    fermi_energy: float | None = None
    homo: float | None = None
    lumo: float | None = None
    #: QE's estimated scf accuracy at the last iteration, in Ry.
    accuracy: float | None = None
    nspin: int = 1
    #: ``int (rho_up - rho_dw)`` and ``int |rho_up - rho_dw|``, in Bohr
    #: magnetons per cell -- the two numbers QE prints. ``None`` unpolarized.
    magnetization: float | None = None
    absolute_magnetization: float | None = None
    #: Only when ``tot_magnetization`` constrained the two channels separately.
    fermi_energy_up: float | None = None
    fermi_energy_down: float | None = None
    #: ``int m(r)`` as a cartesian vector, in Bohr magnetons per cell. Only a
    #: noncollinear run has one -- a collinear one has :attr:`magnetization`,
    #: which is the same quantity when the axis is fixed by construction.
    magnetization_vector: tuple | None = None
    #: ``-int B . m`` and the constraint penalty at the converged density, in
    #: Ry. **Neither is part of** :attr:`total_energy` -- QE prints ``etcon``
    #: and never adds it, and Elk excludes its external field's energy by the
    #: same convention. ``None`` when the run had no field.
    field_energy: float | None = None
    constraint_energy: float | None = None
    #: The field the run ended with, which is not the one it started with when
    #: ``reducebf`` or the fixed-spin-moment scheme was in use.
    magnetic_field: object | None = None
    #: The accumulated ``reducebf`` factor the field was applied with at the
    #: last iteration. **It is not on the field object**: ``reducebf`` multiplies
    #: a loop variable and leaves the field itself alone, so the converged
    #: potential is reproducible only from the pair. Elk's field exists to break
    #: a symmetry and then be scaled away -- after ~25 iterations at 0.9 it is
    #: 7% of its input value -- so a fixed-density run afterwards that rebuilt
    #: the field from the *input* would apply a field the SCF had almost
    #: switched off.
    field_scale: float = 1.0
    #: 1, 2 or 4: how many components :attr:`density` and :attr:`potential`
    #: have. It is 1 for a *nonmagnetic* spin-orbit run, where ``nspin`` is 4.
    nspin_mag: int = 1
    #: The converged Hubbard occupation matrix, ``(nspin, nslot, ldmx, ldmx)``
    #: with one slot per correlated atom, or ``None`` for a run without a U.
    #: **The spin axis is kept even for** ``nspin = 1``, unlike the density's,
    #: because ``ns`` is per channel by construction there (it is halved in
    #: ``new_ns``) and squeezing it would hide the factor of two that the energy
    #: carries. :attr:`hubbard_setup` says which atom each slot is.
    ns: jnp.ndarray | None = None
    hubbard_setup: object | None = None
    #: The converged projector occupations, one real ``(nspin_mag, nat_t, nh,
    #: nh)`` array per ultrasoft/PAW species and ``()`` for a norm-conserving
    #: run. It is here because it is **part of the mixed state**, not a function
    #: of the density (``mix_rho.f90`` mixes "rho in g-space ... and becsum (for
    #: paw)"): a run continued from this one has to be given all of ``(rho,
    #: becsum, ns)`` or it starts from two different states at once.
    becsum: tuple = ()
    #: The :class:`~pypresso.system.builder.System` this was computed for. What
    #: makes the result self-describing enough to continue from -- a
    #: continuation has to check that the k-points and the electron count of the
    #: two runs match before carrying anything over
    #: (:mod:`pypresso.scf.continuation`).
    system: object | None = None
    #: The kinetic energy density, ``(nspin, n1, n2, n3)`` in Ry and per spin
    #: *channel*, or ``None`` when the functional is not a meta-GGA. Carried on
    #: the result because it is part of the converged state: an NSCF run or a
    #: band-structure path under a meta-GGA rebuilds the potential from the
    #: density *and* this, and a fixed density alone does not determine it.
    tau: jnp.ndarray | None = None
    #: The Tran-Blaha coefficient the converged potential used, or ``None``.
    #: The headline number of a TB09 run after the gap itself -- it is a
    #: property of the material and the published values (Si ~1.1, wide-gap
    #: insulators ~1.3-1.6) are what a new implementation is checked against.
    meta_c: float | None = None
    #: The stress tensor (:class:`~pypresso.stress.Stress`) when the run was
    #: asked for one, and ``None`` otherwise -- QE's ``tstress``, which is the
    #: same switch: ``stress()`` is called from ``run_pwscf`` after the SCF, not
    #: inside it, and it costs a strain gradient whether or not anyone reads it.
    #: The forces are deliberately *not* here, and the asymmetry is QE's too: a
    #: relaxation calls ``forces()`` itself every ionic step, so they belong to
    #: the loop that moves the atoms (:mod:`pypresso.workflows.relax`) rather
    #: than to one SCF's result.
    stress: object | None = None
    #: What a residual solver cost, when one was used
    #: (:class:`~pypresso.scf.solvers.NewtonKrylovResult`). ``None`` for a
    #: mixing run. Its ``steps`` is the currency to compare against
    #: :attr:`iterations`: both count diagonalisations.
    solver: object | None = None
    history: list = field(default_factory=list)

    @property
    def hubbard_occupations(self) -> dict:
        """``{atom: (Tr n_up, Tr n_down, total)}`` -- what ``write_ns`` prints."""
        if self.ns is None:
            return {}
        traces = np.asarray(jnp.einsum("snmm->sn", self.ns))
        if traces.shape[0] == 1:
            traces = np.concatenate([traces, traces])
        return {
            atom: (float(traces[0, slot]), float(traces[1, slot]),
                   float(traces[0, slot] + traces[1, slot]))
            for slot, atom in enumerate(self.hubbard_setup.atoms)
        }

    @property
    def eigenvalues_ev(self) -> np.ndarray:
        return self.eigenvalues * RY_TO_EV

    @property
    def eigenvalues_by_spin(self) -> np.ndarray:
        """``(nspin, nk, nbnd)`` whatever ``nspin`` is."""
        return self.eigenvalues if self.nspin == 2 else self.eigenvalues[None]

    @property
    def total_density(self) -> jnp.ndarray:
        """``(n1, n2, n3)``: the charge density.

        The two channels summed when they are ``(up, down)``, and the first
        component alone when they are ``(n, m_x, m_y, m_z)``.
        """
        from pypresso.scf.potential import total_charge

        return total_charge(self.density)

    def __repr__(self) -> str:
        """What the object is, rather than every array it holds.

        The generated dataclass ``__repr__`` prints the wavefunctions, the
        density and the potential in full, which in a REPL is several screens
        of numbers in place of the four facts anyone wants. ``@dataclass`` does
        not overwrite a ``__repr__`` defined in the body, so this one stands.
        """
        state = (f"converged in {self.iterations} iterations"
                 if self.converged else
                 f"NOT converged after {self.iterations} iterations")
        parts = [state, f"E = {self.total_energy:.8f} Ry"]
        if self.accuracy is not None:
            parts.append(f"accuracy = {self.accuracy:.2g} Ry")
        if self.fermi_energy is not None:
            parts.append(f"E_F = {self.fermi_energy * RY_TO_EV:.4f} eV")
        elif self.homo is not None:
            parts.append(f"HOMO = {self.homo * RY_TO_EV:.4f} eV")
        if self.magnetization is not None:
            parts.append(f"M = {self.magnetization:.4f} mu_B")
        elif self.magnetization_vector is not None:
            parts.append("M = ({:.4f}, {:.4f}, {:.4f}) mu_B".format(
                *self.magnetization_vector))
        if self.nspin != 1:
            parts.append(f"nspin = {self.nspin}")
        return f"<SCFResult: {', '.join(parts)}>"


def _spin_block_diagonal(per_atom) -> np.ndarray:
    """Per-atom ``(nh, nh, 2, 2)`` blocks -> one ``(2, 2, nkb, nkb)`` matrix.

    The same block-diagonal assembly ``build_projectors`` does for ``D_ij``,
    with a spin pair in front. Laying the spin indices outermost is what lets
    the sandwich in :func:`_newd_noncollinear` be four ordinary ``nkb x nkb``
    matrix products: the block structure keeps the atoms from mixing, so nothing
    downstream has to loop over them.
    """
    sizes = [block.shape[0] for block in per_atom]
    nkb = sum(sizes)
    out = np.zeros((2, 2, nkb, nkb), dtype=complex)
    offset = 0
    for block, nh in zip(per_atom, sizes):
        out[:, :, offset : offset + nh, offset : offset + nh] = np.transpose(
            block, (2, 3, 0, 1)
        )
        offset += nh
    return out


def _is_traced(x) -> bool:
    """Is this array a JAX tracer rather than a value the host can read?"""
    return isinstance(x, jax.core.Tracer)


class Calculation:
    """Everything that stays fixed while the density changes.

    Separating this from the loop keeps the setup work (rule R2: host-side,
    dynamic shapes, done once) out of the iteration, and makes a band-structure
    run reuse exactly the same objects with a different k-point set.
    """

    def __init__(
        self,
        system: System,
        pseudos: tuple[Pseudopotential, ...],
        basis: Basis | None = None,
        diagonalization: str | None = None,
        k_batch: int | None | str = "default",
    ):
        system = _without_gamma_storage(system)
        self.system = system
        self.eigensolver = get_eigensolver(diagonalization)
        # How many k-points are in flight at once, everywhere this calculation
        # touches the k axis. One -- QE's ``k_loop`` -- unless asked otherwise;
        # ``None`` is a single ``vmap`` over all of them. See
        # :mod:`pypresso.batching`.
        self.k_batch = resolve_k_batch(k_batch)
        self.pseudos = tuple(pseudos)
        self.basis = basis if basis is not None else build_basis(system)
        if self.basis.dense.gamma_only:
            raise NotImplementedError(
                "the gamma-only half-sphere storage is not implemented in h_psi, "
                "sum_band or the eigensolver; pass a basis built with "
                "gamma_only=False"
            )

        # Which exchange-correlation functional this run uses is decided once,
        # here, from the pseudopotentials and the input -- not defaulted to
        # anywhere downstream. Every consumer takes it as an argument, so a PBE
        # dataset cannot end up running under LDA by omission.
        self.functional = resolve_functional(
            [pseudo.functional for pseudo in self.pseudos], system.input_dft
        ).with_meta_coefficient(getattr(system, "mbj_c", None))

        # QE's three spin numbers, kept apart (``set_spin_vars``): how many
        # regimes there are, how many components a *state* has, and how many a
        # *density* has. See :class:`pypresso.system.builder.System`.
        self.nspin = int(system.nspin)
        self.npol = int(system.npol)
        self.nspin_mag = int(system.nspin_mag)
        self.noncolin = bool(system.noncolin)
        self.lspinorb = bool(system.lspinorb)
        if self.nspin == 2 or self.nspin_mag == 4:
            # Refused here rather than where it would first divide by something:
            # a functional whose correlation has no polarized parameterisation
            # would otherwise run with the unpolarized one and converge to a
            # number that is wrong and looks right. A noncollinear magnetization
            # needs it for the same reason -- the functional is evaluated along
            # the local spin axis, which is the polarized one.
            self.functional.require_spin()
        if self.functional.is_meta:
            self._require_meta_supported(system)
        if self.lspinorb and not any(p.has_so for p in self.pseudos):
            raise ValueError(
                "lspinorb = .true. but no pseudopotential is fully relativistic; "
                "a spin-orbit run needs a rel- dataset with a PP_SPIN_ORB section"
            )
        if not self.lspinorb:
            for pseudo in self.pseudos:
                if pseudo.has_so:
                    # ``average_pp``: QE j-averages the projectors back to the
                    # scalar-relativistic ones. That is a different
                    # pseudopotential from the one in the file, so it is refused
                    # rather than done silently.
                    #
                    # **The condition is ``not lspinorb`` and not ``noncolin and
                    # not lspinorb``**, which is what it used to say and what
                    # ``setup.f90`` does not say: QE calls ``average_pp`` in the
                    # *else* branch of ``IF (lspinorb)``, so every run without
                    # spin-orbit coupling gets it -- an ordinary ``nspin = 1``
                    # one above all, which is the common way to pick a `rel-`
                    # dataset off a table by accident. Written the narrow way,
                    # such a run consumed the ``j``-resolved projectors as
                    # though they were ``l``-resolved, converged, and reported a
                    # total energy wrong by **20 Ry** on rhombohedral BN with
                    # ONCVPSP's B and N -- with the Ewald and dispersion terms,
                    # the two that touch no projector, still agreeing to 4e-9.
                    raise NotImplementedError(
                        f"{pseudo.element}: this is a fully-relativistic "
                        "pseudopotential (has_so, a PP_SPIN_ORB section) and the "
                        "run has no spin-orbit coupling, so its two j = l +- 1/2 "
                        "channels would be used as though they were one. QE "
                        "averages them here (average_pp); that is not "
                        "implemented, so it is refused rather than done wrong. "
                        "Use lspinorb = .true., or a scalar-relativistic dataset"
                    )

        self.nelec = sum(self.pseudos[t].z_valence for t in system.structure.types)
        # ``set_nelup_neldw``. Only used when the magnetization is constrained,
        # but computed always because ``default_nbnd`` needs both counts.
        self.nelup, self.neldw = spin_electron_counts(
            self.nelec, system.tot_magnetization
        )
        self.charges = np.array([self.pseudos[t].z_valence for t in system.structure.types])

        # Two grids, and which quantity lives on which is QE's split: the
        # wavefunctions and everything built from them are on the *smooth* grid,
        # the density and the potential on the *dense* one. They are the same
        # object unless the input asks for a dual above 4.
        dense, smooth = self.basis.dense, self.basis.smooth
        # A **spin spiral** (P19) puts the two spinor components on different
        # spheres, ``k + q/2`` and ``k - q/2``, so everything carrying a k index
        # is built for a list of ``2 nk`` points -- the up component's first.
        # One call rather than two is what gives both halves a common ``npwx``,
        # which is what rule R7's padding, the vmap over k and the stick layout
        # all rest on.
        self.spiral = bool(system.spiral)
        self.basis_kpoints = (
            spiral_kpoints(system.kpoints, system.spiral_q, system.cell)
            if self.spiral else system.kpoints
        )
        if self.spiral:
            if basis is not None:
                raise ValueError(
                    "a spin spiral builds its own plane-wave basis on the "
                    "shifted k-points; pass basis = None"
                )
            self.basis = Basis(
                dense=dense,
                smooth=smooth,
                planewaves=build_plane_wave_basis(
                    smooth, self.basis_kpoints, system.cell, system.ecutwfc
                ),
            )
        planewaves = self.basis.planewaves
        self.kinetic = planewaves.kinetic(smooth, self.basis_kpoints, system.cell)
        self.fft_index = planewaves.fft_index(smooth)
        # ``k + G`` itself, and only where a meta-GGA needs it: it is
        # ``(nk, npwx, 3)`` doubles, three times the size of ``kinetic``, and
        # every other consumer wants the modulus. ``None`` otherwise, so that
        # the allocation is not paid for by runs that cannot use it.
        self.kplusg = (
            planewaves.kplusg(smooth, self.basis_kpoints, system.cell)
            if self.functional.is_meta else None
        )

        # QE's FFT layout for the wavefunction transforms; see basis/sticks.py.
        self.sticks = build_sticks(self.fft_index, planewaves.mask, smooth.grid)

        # The projectors are built in two halves -- the species-dependent
        # columns once, the structure factor per geometry -- so that moving the
        # atoms costs one exponential per atom and so that ``grad`` with respect
        # to the positions never reaches the radial integrals. See
        # :class:`pypresso.pseudo.projectors.ProjectorCore` and
        # :meth:`at_positions`.
        self.projector_core = build_projector_core(
            self.pseudos, system.structure, system.cell, smooth, planewaves,
            self.basis_kpoints,
        )
        self.projectors = self.projector_core.at_positions(system.structure.positions)

        # The augmentation charge lives on the *dense* grid: it is sharply
        # peaked, and holding it is what the second grid is for. Everything
        # ultrasoft hangs off this object being non-None.
        self.augmentation = build_augmentation(
            self.pseudos, system.structure, system.cell, dense
        )
        self.species_channels = ()
        self.paw = None
        if self.augmentation is not None:
            self.projectors = eqx.tree_at(
                lambda p: p.qq,
                self.projectors,
                self._per_atom(self.augmentation.qq),
                is_leaf=lambda x: x is None,
            )
            self.species_channels = self._species_channels()

        if self.spiral and self.augmentation is not None:
            # The cross-spin block of ``becsum`` pairs projectors at two
            # *different* k-points, so the augmentation charge it needs is
            # ``q_ij(q)`` -- the arbitrary-wavevector form
            # :mod:`pypresso.topology.augmentation` already builds for P16 --
            # and PAW's transverse one-centre term additionally needs Elk's
            # per-atom phase ``e^{-i q.tau/2}`` (``zqss``, ``init0.f90``).
            # Using the plain ``qq`` instead leaves the overlap plausible and
            # the answer wrong, which is the failure this repository has met
            # twice already, so the combination is refused rather than
            # approximated.
            raise NotImplementedError(
                "a spin spiral with an ultrasoft or PAW dataset is not "
                "implemented: the augmentation charge between the two "
                "components is q_ij(q), not qq; use a norm-conserving "
                "pseudopotential"
            )

        # PAW adds the one-centre corrections on top of everything ultrasoft
        # does. They depend on ``becsum`` and on nothing else that changes, so
        # like ``newd`` they are rebuilt once per SCF iteration.
        self.paw = build_paw(self.pseudos, system.structure, self.functional)

        # The spin-orbit coefficients: ``fcoef`` per species and, assembled over
        # the atoms, the two block matrices the spinor Hamiltonian takes --
        # ``dvan_so`` (which *is* the nonlocal potential for a norm-conserving
        # run) and ``qq_so``. Built once: they depend on the pseudopotentials
        # and on nothing that changes during the SCF.
        self.spin_orbit = ()
        self.dvan_so = None
        self.fcoef_matrix = None
        self.qq_so = None
        if self.noncolin:
            self.spin_orbit = build_spin_orbit(self.pseudos)
            types = system.structure.types
            self.dvan_so = jnp.asarray(_spin_block_diagonal(
                [self.spin_orbit[t].dvan_so for t in types]
            ))
            self.fcoef_matrix = jnp.asarray(_spin_block_diagonal(
                [self.spin_orbit[t].fcoef for t in types]
            ))
            if self.augmentation is not None:
                # ``qq`` is empty for a norm-conserving species even inside an
                # ultrasoft calculation, while its projector block is not: the
                # atom has projectors and no augmentation charge. Padding it to
                # ``nh`` here is what ``block_matrix`` does implicitly by
                # writing nothing, and what keeps the assembled matrix square.
                def species_qq(t: int) -> np.ndarray:
                    nh = self.pseudos[t].nh
                    values = np.asarray(self.augmentation.qq[t])
                    return values if values.shape == (nh, nh) else np.zeros((nh, nh))

                self.qq_so = jnp.asarray(_spin_block_diagonal(
                    [self.spin_orbit[t].qq_so(species_qq(t)) for t in types]
                ))

        # Per-species radial transforms, which the structure factor multiplies:
        # cached so a moved geometry re-contracts them instead of re-integrating
        # them (``setlocal``/``set_rhoc`` are called afresh by QE each ionic
        # step, but their ``vloc``/``rhoc`` tables are not rebuilt either).
        self.vloc_species = species_local_potential(self.pseudos, system.cell, dense)
        self.rho_core_species = species_core_charge(self.pseudos, system.cell, dense)
        # The atomic charge, on the same footing: it is what the SCF starts from
        # and what ``force_corr`` pairs with the potential the last iteration
        # did not apply. QE tabulates it once (``init_tab_rhoat``) for the same
        # reason -- the radial integration does not depend on the geometry.
        self.rho_atomic_species = species_atomic_charge(self.pseudos, system.cell, dense)

        vloc_g = combine_species(self.vloc_species, system.structure, system.cell, dense)
        self.vltot = jnp.real(g_to_r(vloc_g, dense.fft_index, dense.grid))

        # The nonlinear core correction, on the dense grid (``set_rhoc``). It is
        # ``None`` when no species has a PP_NLCC section, and ``None`` is an
        # empty pytree, so the two cases compile separately with no runtime
        # branch in the potential.
        rho_core_g = (
            None if self.rho_core_species is None
            else combine_species(
                self.rho_core_species, system.structure, system.cell, dense
            )
        )
        self.rho_core = (
            None if rho_core_g is None
            else jnp.real(g_to_r(rho_core_g, dense.fft_index, dense.grid))
        )
        # Kept as well as its transform: a gradient-corrected functional needs
        # the core charge's gradient, which is taken in G space alongside the
        # valence density's (``gradcorr`` adds ``rhog_core`` to ``rhogaux``).
        self.rho_core_g = rho_core_g

        # ``alpha`` and the neighbour list are fixed here and the sum is then a
        # differentiable function of the positions -- which is what makes the
        # ionic part of the force ``grad`` of it rather than a second expression.
        # **The k-points in crystal coordinates, which is the invariant.**
        # ``KPoints.coords`` are cartesian in units of ``2 pi / alat``, so they
        # describe a k-set only together with the cell they were built for.
        # Every mover that deforms the cell at a frozen sphere -- ``at_strain``,
        # and ``at_cell`` on top of it -- leaves ``k`` fixed *here* and moves
        # the cartesian ones, so this is what those must rebuild from.
        # ``at_strain`` used to recompute it as
        # ``system.kpoints.crystal(system.cell)`` at the point of use, which is
        # the same number only while the two are consistent: after one cell
        # move they are not, and a *second* ``at_strain`` -- which is what a
        # stress on a moved cell is -- differentiated at k-points 0.031 (in
        # crystal units) away from the ones the SCF had just run at. The energy
        # was right and its derivative was not, which put 64 kbar into the
        # stress of a variable-cell step and moved the relaxed volume of QE's
        # ``vc-relax4`` by 2%. Host-side and concrete: it is a property of the
        # run, decided once.
        self._kcrystal = np.asarray(system.kpoints.crystal(system.cell))

        self.ewald_sum = build_ewald(system.cell, system.structure, dense, self.charges)
        self.ewald = float(
            self.ewald_sum.energy(system.cell, system.structure.positions, dense)
        )

        # The van der Waals correction, on exactly the same footing: a pair sum
        # over the nuclei whose neighbour list is fixed here and whose energy is
        # then a differentiable function of the positions, so its force and its
        # stress are ``grad`` of it and not two more expressions
        # (:mod:`pypresso.vdw`). ``None`` when ``vdw_corr`` is 'none', which is
        # an empty pytree, so a run without one compiles exactly as before.
        #
        # **It never reaches the potential.** ``electrons.f90`` adds ``elondon``
        # to ``etot`` after the SCF loop, so the density, the eigenvalues and
        # every response are bit for bit what they would be without it; what it
        # changes is the total energy, the force, the stress -- and therefore a
        # relaxation and the elastic constants.
        self.dispersion_sum = build_vdw_correction(
            system.vdw_corr, system.cell, system.structure, **vdw_options(system)
        )
        self.dispersion = (
            0.0 if self.dispersion_sum is None
            else float(self.dispersion_sum.energy(system.structure.positions))
        )

        # The density built from a symmetry-reduced k-point set is not itself
        # symmetric; QE restores the symmetry explicitly and so must we, or
        # degenerate levels split by tens of meV and the energy is wrong in the
        # third decimal. See pypresso.system.symmetry.
        # QE's default dual is 4, which is exactly the condition for the density
        # grid to resolve every G - G' between two wavefunction plane waves --
        # see Hamiltonian.matrix. An input with a smaller dual is legal, so the
        # question is asked rather than assumed. Gamma-only storage keeps half
        # the sphere, so a difference of two stored G's need not be a stored G at
        # all and the gather does not apply however large the dual is.
        # The gather reads the potential stored on the *smooth* grid, since that
        # is the grid H|psi> runs on, so the question is whether the smooth
        # cutoff reaches 4*ecutwfc -- which it does by construction whenever the
        # dual exceeds 4, and only then depends on ecutrho.
        self.resolves_differences = bool(
            smooth.ecut >= 4.0 * system.ecutwfc - 1e-8 and not smooth.gamma_only
        )

        # ``nosym`` is not a speed switch. An input can ask for a state that
        # does not have the crystal's symmetry -- one of an atom's three p
        # channels occupied and the others empty, which is what
        # ``pw_pawatom/paw-atom_spin*.in`` does -- and symmetrising the density
        # or ``becsum`` anyway averages the three and converges somewhere else
        # entirely. QE requires ``nosym = .true.`` for those inputs; honouring it
        # is a correctness matter.
        # The magnetic case has a *smaller* group and operations that are only
        # symmetries together with time reversal; ``System.symmetry_group`` is
        # the single place that rule lives, so the group used here is the one
        # the k-point set was reduced with.
        # ``compute_ux``: the fixed axis a gradient-corrected noncollinear run
        # takes the sign of the magnetization along. ``None`` -- QE's
        # ``lsign = .FALSE.`` -- whenever the starting moments are not all
        # parallel. A tuple rather than an array because it crosses a ``jit``
        # boundary as a static argument.
        axis = (
            fixed_quantization_axis(system.local_moments)
            if self.nspin_mag == 4 and not self.spiral else None
        )
        self.quantization_axis = None if axis is None else tuple(float(v) for v in axis)

        self.symmetries = system.symmetry_group()
        #: Whether this run actually symmetrises with :attr:`symmetries`.
        #:
        #: **The group is kept whole and the switch sits beside it**, because
        #: the group is a property of the crystal and is wanted in places an
        #: input's ``nosym`` has no say over -- ``basis/builder.py`` needs the
        #: fractional translations to size the FFT box whatever the input says.
        #: The consequence is that ``calculation.symmetries`` is **not** the
        #: group a run symmetrises with, and every consumer has to say so; the
        #: ``symmetrize_*`` methods on this class are where that is said, which
        #: is why a caller should reach for them rather than for the group.
        #: Three places in ``response/`` reached for the group instead and so
        #: symmetrised a ``nosym`` run (``PLAN.md`` P28a).
        self.use_symmetry = bool(not system.nosym and self.symmetries.nsym > 1)
        use_symmetry = self.use_symmetry
        if self.spiral and use_symmetry:
            # The reader refuses this pair too (``_spiral_q``), but a ``System``
            # can be built without going through it -- directly, or through the
            # ``dataclasses.replace`` that ``at_spiral_q`` and
            # ``workflows/spiral`` use -- and then nothing else here looks at the
            # combination. Symmetrising a spiral averages the rotated-frame
            # magnetization over operations of the *crystal* point group, which
            # are not symmetries of the spiral, and reduces the k-set to the same
            # wrong wedge; the run converges and reports an energy that is
            # silently wrong. The spin space group is what would make it right.
            raise NotImplementedError(
                "a spin spiral needs nosym = .true.: the symmetry group here is "
                "the crystal's, and an operation that is a symmetry of the "
                "lattice need not be one of the spiral -- the spin space group "
                "is not implemented, so the density symmetrisation and the "
                "k-point reduction would both be wrong"
            )
        # The axial-vector rotations, signs folded in, that the magnetization
        # needs and the charge does not.
        self._magnetization_rotations = (
            jnp.asarray(
                magnetization_signs(system.cell, self.symmetries)[:, None, None]
                * cartesian_rotations(system.cell, self.symmetries)
            )
            if use_symmetry and self.nspin_mag == 4 else None
        )
        # PAW's projector occupations need the symmetry imposed on them
        # explicitly -- see pypresso.paw.symmetry. Built after the symmetry
        # search, which is why it is not up with the rest of the PAW setup.
        self._becsum_symmetry = (
            build_becsum_symmetry(self.pseudos, system.structure, system.cell,
                                  self.symmetries, self.nspin_mag)
            if self.paw is not None and use_symmetry else None
        )
        self._symmetry_maps = symmetry_maps(dense, self.symmetries) if use_symmetry else None

        # DFT+U (P20). Built after the symmetry search, because the occupation
        # matrix needs the same group average ``becsum`` does, and after the
        # projectors, because the Hubbard projectors are ``S`` applied to the
        # atomic orbitals. ``None`` -- the normal case -- costs nothing: the term
        # is absent from the Hamiltonian rather than added as a zero.
        self.hubbard = build_hubbard_setup(
            system.hubbard, system.structure, self.pseudos
        )
        self._setup_hubbard(use_symmetry)

        # External fields and constrained moments (P18). ``None`` -- the normal
        # case -- costs nothing: the whole term is absent rather than added as a
        # zero to every potential.
        self.magnetic_field = self._build_magnetic_field()


    def _require_meta_supported(self, system) -> None:
        """What a potential-only meta-GGA cannot be combined with, refused by name.

        Each of these is refused because the *implementation* is missing, not
        because the physics is:

        * **Ultrasoft**, but not PAW. ``tau`` on the grid is the smooth
          states' and needs a one-centre correction inside the spheres; a PAW
          dataset carries the partial waves that supply it (``PLAN.md`` P32)
          and a plain ultrasoft one does not.
        Noncollinear magnetism and spin-orbit coupling **are** supported
        (``PLAN.md`` P31), which is where this implementation goes beyond
        ``pw.x``: ``setup.f90`` raises "Non-collinear Meta-GGA not implemented"
        and stops. Here ``tau`` is carried as the 2x2 matrix it is and resolved
        onto the density's local spin axis
        (:func:`~pypresso.scf.potential._noncollinear_meta_exchange`).
        * **Spin spirals.** The two spinor components live on different spheres,
          so ``grad psi`` carries two different ``k + q/2`` sets and the sum is
          not the kinetic energy density of anything periodic.
        """
        if any(p.is_ultrasoft and not p.is_paw for p in self.pseudos):
            # **Ultrasoft, but not PAW.** The distinction is the partial waves:
            # a PAW dataset carries the all-electron and pseudo pairs that let
            # ``tau`` be reconstructed inside the sphere (``PLAN.md`` P32), and
            # a plain ultrasoft one carries only the augmentation charge, which
            # is a correction to the *density* with no kinetic counterpart
            # anywhere -- not here and not in QE. Running it would pair a full
            # density with a smooth tau, and their ratio is what the whole
            # Becke-Roussel fit is built on.
            raise NotImplementedError(
                f"{self.functional.name} is a meta-GGA and needs the kinetic "
                "energy density tau inside the augmentation spheres; an "
                "ultrasoft dataset has no partial waves to reconstruct it from, "
                "where a PAW one does. Use a PAW or norm-conserving dataset -- "
                "pw.x refuses both (setup.f90: 'Meta-GGA not implemented with "
                "USPP/PAW')"
            )
        if getattr(system, "spiral_q", None) is not None:
            raise NotImplementedError(
                f"{self.functional.name} with a spin spiral is not implemented: "
                "the two spinor components live on different plane-wave spheres, "
                "so their gradients do not add to a lattice-periodic tau"
            )
        if getattr(system, "hubbard", None):
            # Nothing in the physics forbids it -- ``vhpsi`` is a separate term
            # and does not touch ``tau``. What forbids it is that the
            # combination is unvalidated, and one concrete thing breaks with it:
            # ``_solve_residual``'s convergence measure reads the Hubbard block
            # off the *end* of the packed state, and with tau packed after it
            # that slice is tau.
            raise NotImplementedError(
                f"{self.functional.name} with a Hubbard U is not implemented: "
                "the combination is unvalidated here, and the residual solver's "
                "convergence measure reads ns off the end of the packed state, "
                "which tau now occupies"
            )

    def _setup_hubbard(self, use_symmetry: bool) -> None:
        """Everything a DFT+U run needs beyond the manifold list.

        The projectors themselves, the static index arrays that move an
        occupation matrix between its padded per-atom form and the block matrix
        the Hamiltonian contracts with, the group average, and the parameter
        coefficients. All of it is fixed for the geometry, so it is built here
        and rebuilt only where the geometry or the k-list changes
        (:meth:`at_positions`, :meth:`at_kpoints`).
        """
        if self.hubbard is None:
            self.wfcU = None
            self.hubbard_coefficients = None
            self.hubbard_symmetry = None
            self._hubbard_columns = self._hubbard_mask = None
            self._hubbard_blocks = None
            return

        setup = self.hubbard
        if self.noncolin:
            # ``new_ns_nc`` measures a 2x2 spin matrix per pair of orbitals and
            # ``v_hubbard_nc`` builds a potential from it; neither is written
            # here, and running the collinear expressions on one spinor channel
            # would apply a correction with the wrong spin structure.
            raise NotImplementedError(
                "DFT+U with noncolin = .true. is not implemented: the "
                "occupation matrix is a 2x2 matrix in spin space (new_ns_nc), "
                "not one real matrix per channel"
            )
        if self.spiral:
            raise NotImplementedError(
                "DFT+U together with a spin spiral is not implemented: the two "
                "spinor components live on different plane-wave spheres, so a "
                "projector would have to be built on each"
            )
        if use_symmetry and np.any(np.asarray(self.symmetries.t_rev_array()) != 0):
            # ``new_ns`` flips the spin index of an operation that is a symmetry
            # only with time reversal. Nothing validates that branch here -- see
            # :func:`pypresso.hubbard.occupations.build_ns_symmetry` -- so it is
            # refused rather than silently skipped, which would symmetrise the
            # two channels into each other's frame.
            raise NotImplementedError(
                "DFT+U on a symmetry group carrying time-reversed operations "
                "(t_rev) is not implemented; run with nosym = .true."
            )

        self.hubbard_coefficients = coefficients_from_setup(setup)
        columns = np.zeros((setup.nslot, setup.ldmx), dtype=int)
        mask = setup.slot_mask()
        for slot, (offset, ldim) in enumerate(zip(setup.offsets, setup.ldims)):
            columns[slot, :ldim] = np.arange(offset, offset + ldim)
        self._hubbard_columns = jnp.asarray(columns)
        self._hubbard_mask = jnp.asarray(mask)
        slot_index, row, column = setup.block_indices()
        block_row, block_column = setup.padded_indices()
        self._hubbard_blocks = (
            jnp.asarray(slot_index), jnp.asarray(row), jnp.asarray(column),
            jnp.asarray(block_row), jnp.asarray(block_column), setup.nwfcU,
        )
        self.hubbard_symmetry = (
            build_ns_symmetry(
                setup, self.system.cell, self.system.structure, self.symmetries
            ) if use_symmetry else None
        )
        self.wfcU = self._build_hubbard_projectors()

    def _overlap(self, psi: jnp.ndarray, ik: int) -> jnp.ndarray:
        """``S|psi>`` without a Hamiltonian to hang it on (``s_psi``).

        The Hubbard projectors are built before any potential exists, so they
        cannot go through :meth:`Hamiltonian.apply_s`; this is the same operator
        written against the projectors alone.
        """
        if self.projectors.qq is None:
            return psi
        vkb = self.projectors.vkb[ik]
        becp = jnp.einsum("gk,...g->...k", vkb.conj(), psi)
        qq = self.projectors.qq.astype(vkb.dtype)
        return psi + jnp.einsum("gk,...k->...g", vkb, becp @ qq.T)

    def _build_hubbard_projectors(self, kcart=None) -> jnp.ndarray:
        """``wfcU`` at every k-point of this calculation.

        ``kcart`` is for :meth:`at_strain` and for nothing else: the atomic
        orbitals are built at ``k + G``, and under a strain the k-points move
        with the reciprocal cell while ``KPoints.coords`` -- being in units of a
        *static* ``alat`` -- do not.
        """
        return build_hubbard_projectors(
            self.hubbard, self.pseudos, self.system.structure, self.system.cell,
            self.basis.smooth, self.basis.planewaves, self.basis_kpoints,
            self._overlap, kcart,
        )

    @property
    def is_hubbard(self) -> bool:
        """Whether any species carries a Hubbard U."""
        return self.hubbard is not None

    def occupation_matrix(self, wavefunctions, weights) -> jnp.ndarray:
        """``new_ns``: the symmetrised occupation matrix of the current states."""
        ns = occupation_matrix(
            self.wfcU, wavefunctions, weights,
            self._hubbard_columns, self._hubbard_mask, self.k_batch,
        )
        if self.hubbard_symmetry is not None:
            ns = self.hubbard_symmetry.apply(ns)
        return ns

    def hubbard_terms(self, ns: jnp.ndarray) -> tuple:
        """``(eth, v_ns, one HubbardTerm per spin channel)`` for an occupation matrix.

        The energy is QE's ``eth``, added to the total energy; ``v_ns`` is its
        derivative, which ``deband`` pairs with the *output* occupation matrix
        exactly as it pairs ``v_scf`` with the output density.
        """
        energy = hubbard_energy(ns, self.hubbard_coefficients)
        v_ns = hubbard_potential(ns, self.hubbard_coefficients)
        blocks = block_potential(v_ns, self._hubbard_blocks)
        return energy, v_ns, tuple(
            HubbardTerm(wfcU=self.wfcU, vns=blocks[spin]) for spin in range(ns.shape[0])
        )

    def starting_ns(self) -> jnp.ndarray:
        """``init_ns``: the occupation matrix the first potential is built from.

        Diagonal, filled by Hund's rule from the reference occupation. **Not**
        adjusted by ``starting_ns_eigenvalue`` -- ``ns_adj`` runs at the end of
        the *first* iteration, on the matrix measured from the first
        diagonalisation, not on this one (``electrons.f90``, after ``sum_band``).
        Applying it here instead would steer the first Hamiltonian rather than
        the second and give a different self-consistent solution.
        """
        return initial_ns(self.hubbard, self.nspin, self.starting_magnetization)

    def adjust_ns(self, ns: jnp.ndarray) -> jnp.ndarray:
        """``ns_adj``: impose ``starting_ns_eigenvalue`` on a measured ``ns``."""
        return adjust_ns(ns, self.hubbard)

    def ns_accuracy(self, residual: jnp.ndarray) -> jnp.ndarray:
        """``ns_ddot``: the occupation matrix's share of ``dr2``."""
        return ns_ddot(residual, self.hubbard_coefficients)

    def _build_magnetic_field(self) -> MagneticField | None:
        """``add_bfield``'s inputs, resolved once (``input.f90``'s ``i_cons``)."""
        system = self.system
        constraint = system.constrained_magnetization
        uniform = np.asarray(system.b_field, dtype=float)
        atomic = (
            np.asarray(system.atomic_b_field, dtype=float)
            if system.atomic_b_field else None
        )
        if constraint == "none" and not uniform.any() and atomic is None:
            return None
        if self.nspin == 1:
            raise ValueError(
                "a magnetic field needs nspin = 2 or noncolin = .true.: there is "
                "no magnetization for it to act on"
            )
        if self.noncolin and self.nspin_mag != 4:
            # ``domag`` is decided by ``starting_magnetization`` and by nothing
            # else (``setup.f90:219``), so a noncollinear run with a field and no
            # starting moment has ``nspin_mag = 1``: a one-channel density and a
            # one-channel potential, with nowhere to put a field that is three
            # components wide. **``pw.x`` does not do this either, and the way it
            # fails is worse.** It allocates ``rho%of_r`` with ``nspin = 4``
            # whatever ``domag`` says (``scf_mod.f90:140``), so ``add_bfield``
            # has channels 2:4 to write into and writes the field there -- and
            # then ``vloc_psi_nc`` applies the magnetization channels only
            # ``IF (domag)`` (``vloc_psi_acc.f90:331``), so the field never
            # reaches a wavefunction. Such a run converges, reports success, and
            # is the field-free calculation.
            raise ValueError(
                "a magnetic field in a noncollinear run needs a magnetization to "
                "act on, and this run has none: every starting_magnetization is "
                "zero, so nspin_mag = 1 and the density has no magnetization "
                "channels (setup.f90's domag). Set starting_magnetization for at "
                "least one species -- any nonzero value will do, the field is "
                "what decides where the moment ends up. Note that pw.x accepts "
                "this input and silently ignores the field (vloc_psi_nc applies "
                "the magnetization channels only if domag), so its answer for it "
                "is the field-free one"
            )
        # A collinear run has one magnetization component and a noncollinear one
        # has three, and every array here follows that -- ``npol = nspin - 1`` in
        # ``add_bfield``.
        components = 3 if self.nspin_mag == 4 else 1
        if components == 1:
            uniform = uniform[2:3]
            if atomic is not None:
                atomic = atomic[:, 2:3]

        # The atom-resolved schemes need the integration spheres; the total ones
        # do not, and building them costs a pass over the dense grid per atom.
        needs_regions = atomic is not None or constraint in ("atomic", "atomic direction")
        regions = (
            build_local_regions(
                system.cell, system.structure, self.basis.dense.grid,
                radii=system.integration_radii or None,
                scheme=system.local_weights,
            )
            if needs_regions else None
        )
        targets = constraint_targets(
            constraint,
            system.structure.types,
            system.starting_magnetization,
            system.angle1,
            system.angle2,
            system.fixed_magnetization,
            system.structure.ntyp,
            self.nspin_mag == 4,
        )
        return MagneticField(
            regions=regions,
            uniform=jnp.asarray(uniform),
            atomic=None if atomic is None else jnp.asarray(atomic),
            targets=jnp.asarray(targets) if len(targets) else None,
            penalty=float(system.constraint_lambda),
            constraint=constraint,
            reducebf=float(system.reducebf),
            fsm_update=system.fsm_update,
        )

    def _moved_magnetic_field(self, system):
        """The field's integration spheres, rebuilt for a geometry that moved.

        ``make_pointlists`` assigns every dense-grid point to the atom whose
        sphere it falls in, so the spheres are a function of the positions and
        of the cell -- and ``at_positions``/``at_strain`` move both. Only the
        ``regions`` depend on the geometry; the uniform field, the per-atom
        fields and the constraint targets do not.

        **Frozen while differentiating, rebuilt when the atoms actually move.**
        ``build_local_regions`` is host-side NumPy and cannot run on a traced
        geometry -- and it does not need to: the assignment is piecewise
        constant in the positions, since a grid point changes owner discretely,
        so its derivative vanishes away from the crossings and freezing it is
        exact between them. That is the trade the spiral's plane-wave sphere
        makes (P21), for the same reason. What is *not* allowed is carrying a
        stale assignment across a concrete move, which is what a relaxation
        under ``constrained_magnetization = 'atomic'`` did: the penalty was
        integrated over spheres centred where the atoms started.
        """
        field = self.magnetic_field
        if field is None or field.regions is None:
            return field
        if _is_traced(system.structure.positions) or _is_traced(system.cell.at):
            return field
        return eqx.tree_at(
            lambda f: f.regions,
            field,
            build_local_regions(
                system.cell, system.structure, self.basis.dense.grid,
                radii=system.integration_radii or None,
                scheme=system.local_weights,
            ),
        )

    def _per_atom(self, per_species) -> jnp.ndarray:
        """A per-species ``(nh, nh)`` quantity, as the ``(nkb, nkb)`` block matrix."""
        blocks = tuple(
            jnp.broadcast_to(block, (len(atoms),) + block.shape)
            for block, atoms in zip(per_species, self.augmentation.species_atoms)
        )
        return self.augmentation.block_matrix(blocks)

    def _species_channels(self) -> tuple:
        """For each species, the projector columns of each of its atoms.

        ``None`` for a norm-conserving species: it has projectors, but no
        augmentation charge, so no ``becsum`` is needed from it.
        """
        offsets = self.augmentation.channel_offsets
        channels = []
        for t, atoms in enumerate(self.augmentation.species_atoms):
            nh = self.pseudos[t].nh
            if not self.pseudos[t].is_ultrasoft or nh == 0 or not atoms:
                channels.append(None)
                continue
            channels.append(
                jnp.asarray([[offsets[a] + i for i in range(nh)] for a in atoms])
            )
        return tuple(channels)

    def at_positions(self, positions: jnp.ndarray) -> "Calculation":
        """The same calculation with the atoms somewhere else.

        Everything that does not depend on where the atoms are is shared with
        ``self`` rather than rebuilt: the plane-wave basis and both FFT grids,
        the radial tables of every species, the symmetry group, the PAW and
        spin-orbit coefficients. What is rebuilt is exactly what the structure
        factor multiplies -- the local potential, the core charge, the
        projectors, the augmentation charge's phases -- plus the Ewald sum.

        The method is **traceable**: given traced ``positions`` it returns a
        calculation whose position-dependent arrays are traced, which is what
        makes the force ``grad`` of an energy evaluated through it
        (:mod:`pypresso.forces`). It is also what a relaxation step uses to move
        the atoms.

        **Two things are deliberately not recomputed**, both following
        ``setup.f90``, which runs once whatever the ion dynamics does:

        * the **FFT grid**. Its dimensions must be a multiple of the
          denominators of the crystal's fractional translations, so a geometry
          that breaks a symmetry would be given a *different grid* -- and the
          exchange-correlation energy is evaluated pointwise on it, so the
          energy would jump by ~1e-6 Ry in the middle of a relaxation for a
          reason that has nothing to do with the physics.
        * the **symmetry group**. QE finds it once and afterwards only *checks*
          it (``checkallsym``); re-searching it here would symmetrise a
          distorted structure with operations it no longer has, or -- worse --
          quietly stop symmetrising and change the answer between two steps of
          the same relaxation. :func:`pypresso.system.symmetry.check_symmetry`
          is the check to run instead.
        """
        moved = copy.copy(self)
        # The spiral gradient's compiled kernel closes over *this* calculation --
        # its local potential, its Ewald sum, its projector positions -- so it
        # cannot follow the atoms, and would otherwise be evaluated in silence at
        # the geometry it was built at. The analytic force's and the stress's
        # kernels close over the calculation the same way; they do not need a pop
        # here because they are keyed on the calculation they captured and the
        # copy below is a different object, which is the invalidation this one
        # gets by name.
        moved.__dict__.pop("_spiral_gradient", None)
        moved.__dict__.pop("_spiral_gradient_chunk", None)
        moved.system = eqx.tree_at(
            lambda sys: sys.structure.positions, self.system, positions
        )
        dense = self.basis.dense
        cell = self.system.cell
        structure = moved.system.structure

        qq = None if self.projectors.qq is None else self.projectors.qq
        moved.projectors = self.projector_core.at_positions(positions, qq=qq)
        if self.augmentation is not None:
            moved.augmentation = self.augmentation.at_positions(
                positions, dense.cartesian(cell)
            )

        vloc_g = combine_species(self.vloc_species, structure, cell, dense)
        moved.vltot = jnp.real(g_to_r(vloc_g, dense.fft_index, dense.grid))
        if self.rho_core_species is None:
            moved.rho_core_g = moved.rho_core = None
        else:
            moved.rho_core_g = combine_species(
                self.rho_core_species, structure, cell, dense
            )
            moved.rho_core = jnp.real(
                g_to_r(moved.rho_core_g, dense.fft_index, dense.grid)
            )

        moved.ewald = self.ewald_sum.energy(cell, positions, dense)
        # The dispersion sum's neighbour list is a property of the cell, so a
        # displacement only moves where it is evaluated.
        if self.dispersion_sum is not None:
            moved.dispersion = self.dispersion_sum.energy(positions)
        # The Hubbard projectors are atomic orbitals centred on the atoms, so
        # they move with them -- and the fact that they do is the whole of the
        # Hubbard force (``force_hub``), which falls out of differentiating the
        # energy through this call rather than from a transcribed expression.
        if self.hubbard is not None:
            moved.wfcU = moved._build_hubbard_projectors()
        moved.magnetic_field = moved._moved_magnetic_field(moved.system)
        return moved

    def at_cell(self, at: jnp.ndarray) -> "Calculation":
        """The same calculation in a different cell, host-side lists rebuilt.

        :meth:`at_strain`'s counterpart for a cell that has *moved* rather than
        been differentiated, and the pair is exactly :meth:`at_spiral_q`'s:
        **frozen while differentiating, rebuilt to move**. A stress is a
        derivative at one geometry, so freezing the Ewald and dispersion
        neighbour lists there is right -- no image is gained or lost, and the
        ``rmax``/``rcut`` boundary sits where the terms are 1e-8 and 1e-12 Ry.
        A variable-cell *step* is not a derivative: the cell can shrink by
        several per cent, and an image that was outside the enumeration radius
        at the starting cell is then inside ``rmax`` and simply missing. The
        error is an ``erfc`` tail rather than a shape mismatch, so it converges
        and reports success -- ``erfc(3.6) = 3.6e-7`` against a 1e-9 Ry
        comparison. ``rgen`` and ``ewald`` run afresh on every ionic step in
        QE for this reason, and so do they here.

        What stays frozen is what makes the run one run: the FFT grid, the
        G-sphere's Miller indices, the symmetry group and the k-points in
        crystal coordinates -- ``scale_h.f90`` exactly, which re-expresses the
        *same* G-vectors against the new reciprocal cell and changes nothing
        else. That is what lets a whole variable-cell relaxation be a single
        setup, with the basis rebuilt once at the end
        (:mod:`pypresso.workflows.vc_relax`).
        """
        at = jnp.asarray(at)
        current = self.system.cell.at
        # ``at_strain`` deforms by ``a_i -> D a_i``, i.e. ``at -> at @ D.T``.
        deformation = jnp.asarray(at).T @ jnp.linalg.inv(jnp.asarray(current)).T
        moved = self.at_strain(deformation - jnp.eye(3, dtype=deformation.dtype))

        cell, structure = moved.system.cell, moved.system.structure
        dense = moved.basis.dense
        moved.ewald_sum = build_ewald(cell, structure, dense, moved.charges)
        moved.ewald = float(
            moved.ewald_sum.energy(cell, structure.positions, dense)
        )
        if moved.dispersion_sum is not None:
            moved.dispersion_sum = build_vdw_correction(
                moved.system.vdw_corr, cell, structure,
                **vdw_options(moved.system),
            )
            moved.dispersion = float(
                moved.dispersion_sum.energy(structure.positions)
            )
        # Used only by the analytic ``force_corr`` (:mod:`pypresso.forces`), and
        # left stale by ``at_strain`` because nothing on the differentiated path
        # reads it. A relaxation does read it, whenever it is asked for the
        # analytic force as a cross-check.
        moved.rho_atomic_species = species_atomic_charge(
            moved.pseudos, cell, dense
        )
        # ``at_strain`` leaves ``system.kpoints`` holding the *starting* cell's
        # cartesian coordinates, which is harmless while it is only ever a
        # tangent (nothing reads them inside a derivative) and is a stale
        # k-point list the moment the cell has actually moved. Everything on the
        # compute path goes through ``_kcart``, but a caller that asks the
        # system what its k-points are deserves the truth -- and
        # ``with_cell`` at the end of a relaxation reads exactly that.
        moved.system = eqx.tree_at(
            lambda sys: sys.kpoints.coords,
            moved.system,
            moved.system.cell.precision.as_real(
                np.asarray(self._kcrystal) @ np.asarray(cell.bg) / float(cell.tpiba)
            ),
        )
        return moved

    def at_strain(self, strain: jnp.ndarray) -> "Calculation":
        """The same calculation in a cell deformed by ``h -> (1 + epsilon) h``.

        The third member of the family, after :meth:`at_positions` and
        :meth:`at_spiral_q`, and the one the stress is ``grad`` of (P11,
        :mod:`pypresso.stress`). ``strain`` is a ``(3, 3)`` array; the
        lattice vectors become ``a_i -> (1 + eps) a_i``, the atoms are carried
        along in **crystal** coordinates so ``tau_a -> (1 + eps) tau_a``, and
        every reciprocal-space quantity follows from ``G -> (1 + eps)^-T G``
        because the Miller indices are what is stored.

        Like :meth:`at_positions` it is **traceable**: given a traced ``strain``
        it returns a calculation whose cell-dependent arrays are traced, which is
        the whole of the stress tensor.

        **What is frozen, and why each is exact or nearly so.**

        * **Which plane waves are in the sphere.** Membership is
          ``|k + G|^2 <= ecutwfc`` evaluated on the host, so it cannot be traced;
          it is also piecewise constant in ``epsilon``, so on each piece the
          frozen-sphere derivative is the exact derivative. What it misses is the
          jump at the strains where a plane wave crosses the cutoff -- the
          **Pulay stress**, which is the finite basis's own error and is quoted
          against the cutoff in `PLAN.md`'s P11 section. This is
          :meth:`at_spiral_q`'s trade in the other coordinate, and unlike a
          spiral's it is what makes a stress at a low cutoff disagree with a
          re-converged finite difference.
        * **The FFT grid and the symmetry group**, for ``setup.f90``'s reason
          (:meth:`at_positions`, and the same in reverse: a strained cell would
          be given different FFT dimensions, and the exchange-correlation energy
          is evaluated pointwise on them).
        * **Ewald's ``alpha`` and its neighbour list.** The Ewald split is exact
          for any ``alpha``, so holding it fixed changes only how the total is
          divided between the two sums. The list of *images* is held fixed as a
          set of integer lattice translations and deformed with the cell, so no
          image is gained or lost; what a strain does change is which of them the
          ``rmax`` mask keeps, and that boundary is at ``erfc(4) ~ 2e-8``.
        * **``qq``**, the integral of the augmentation charge. It is
          ``int Q_ij(r) dr`` over all space and carries no cell at all.

        The atomic starting charge is deliberately *not* rebuilt: nothing in the
        energy being differentiated uses it (only the SCF's first guess and
        ``force_corr`` do).
        """
        if self.spiral:
            raise NotImplementedError(
                "the stress of a spin spiral is not implemented: the two "
                "components' spheres are centred at k +- q/2, so a strain moves "
                "q as well and the generalized Bloch theorem's own term would "
                "be missing"
            )

        strain = jnp.asarray(strain)
        deformation = jnp.eye(3, dtype=strain.dtype) + strain

        strained = copy.copy(self)
        # As in ``at_positions`` and ``at_spiral_q``: any compiled kernel that
        # closed over *this* cell cannot follow one that has been deformed.
        strained.__dict__.pop("_spiral_gradient", None)
        strained.__dict__.pop("_spiral_gradient_chunk", None)
        strained.__dict__.pop("_energy_gradient", None)
        strained.__dict__.pop("_analytic_terms", None)
        strained.__dict__.pop("_tetrahedra", None)

        # ``at[i]`` is a lattice vector as a *row*, so ``a_i -> D a_i`` is a
        # right-multiplication by ``D^T``. Getting this transpose wrong is
        # invisible for an isotropic strain and wrong for every shear.
        cell = eqx.tree_at(
            lambda c: c.at, self.system.cell, self.system.cell.at @ deformation.T
        )
        positions = self.system.structure.positions @ deformation.T
        system = eqx.tree_at(
            lambda sys: (sys.cell.at, sys.structure.positions),
            self.system,
            (cell.at, positions),
        )
        strained.system = system
        structure = system.structure
        dense, smooth = self.basis.dense, self.basis.smooth

        # **The k-points move with the reciprocal cell.** ``KPoints.coords`` are
        # cartesian in units of ``2 pi / alat`` and ``alat`` is static, so
        # ``kpoints.cartesian(strained_cell)`` returns the *unstrained* k-points
        # -- silently, and exactly zero at Gamma. What is fixed under a strain is
        # k in crystal coordinates, which is the same rule the G-vectors follow.
        kcart = jnp.asarray(self._kcrystal) @ cell.bg
        # Recorded because *nothing else can recover it*: ``KPoints.coords`` are
        # cartesian in units of ``2 pi / alat``, so
        # ``kpoints.cartesian(strained_cell)`` gives the unstrained k-points back
        # (the paragraph above). Anything that needs ``k`` in 1/bohr at this cell
        # -- the velocity operator, and so the position operator a dielectric
        # response is built from -- has to read this rather than recompute it.
        strained._kcart = kcart
        strained.kinetic = self.basis.planewaves.kinetic(
            smooth, self.basis_kpoints, cell, kcart
        )

        # The projectors: rebuilt whole, radial integrals included. Unlike a
        # change of position, ``|k+G|`` itself moves, so the form factors are
        # part of the derivative rather than a cached table it multiplies.
        strained.projector_core = build_projector_core(
            self.pseudos, structure, cell, smooth, self.basis.planewaves,
            self.basis_kpoints, kcart,
        )
        strained.projectors = strained.projector_core.at_positions(
            positions, qq=self.projectors.qq
        )

        if self.augmentation is not None:
            strained.augmentation = build_augmentation(
                self.pseudos, structure, cell, dense
            )

        # The local potential and the core charge: new radial transforms against
        # the moved ``|G|``, times structure factors that are themselves
        # invariant (``G . tau`` is integers against crystal coordinates), so
        # every bit of their strain dependence is in the radial half.
        vloc_species = species_local_potential(self.pseudos, cell, dense)
        vloc_g = combine_species(vloc_species, structure, cell, dense)
        strained.vloc_species = vloc_species
        strained.vltot = jnp.real(g_to_r(vloc_g, dense.fft_index, dense.grid))

        if self.rho_core_species is None:
            strained.rho_core_g = strained.rho_core = None
        else:
            strained.rho_core_species = species_core_charge(self.pseudos, cell, dense)
            strained.rho_core_g = combine_species(
                strained.rho_core_species, structure, cell, dense
            )
            strained.rho_core = jnp.real(
                g_to_r(strained.rho_core_g, dense.fft_index, dense.grid)
            )

        # The Ewald neighbour list is a set of lattice translations, so it
        # deforms with the cell exactly as the lattice vectors do.
        strained.ewald_sum = eqx.tree_at(
            lambda e: e.translations,
            self.ewald_sum,
            self.ewald_sum.translations @ deformation.T,
        )
        strained.ewald = strained.ewald_sum.energy(cell, positions, dense)
        # ... and the dispersion sum's list is deformed the same way, for the
        # same reason: it is a set of lattice translations.
        if self.dispersion_sum is not None:
            strained.dispersion_sum = self.dispersion_sum.at_cell(deformation)
            strained.dispersion = strained.dispersion_sum.energy(positions)

        if self.hubbard is not None:
            # The Hubbard projectors are atomic orbitals at ``k + G``, so they
            # need the *strained* k-points as well as the strained cell -- and
            # the fact that they move at all is the whole of ``stres_hub``,
            # which falls out of differentiating through this call rather than
            # from 2291 lines of transcribed Fortran.
            strained.wfcU = strained._build_hubbard_projectors(kcart)
        # ``build_local_regions`` takes its minimum-image distances in the cell's
        # own metric, so a deformed cell measures the spheres differently.
        strained.magnetic_field = strained._moved_magnetic_field(strained.system)
        return strained

    def at_kpoints(self, kpoints) -> "Calculation":
        """The same calculation on a different k-point list.

        The counterpart of :meth:`at_positions` on the other axis, and it exists
        for the same reason: **almost nothing depends on which k-points are
        asked for.** The G-vector sets of both grids, the FFT dimensions, the
        local potential, the core charge, the augmentation charge, the Ewald
        sum, the symmetry group, the radial tables, PAW's one-centre setup and
        the spin-orbit coefficients are all properties of the cell and the
        atoms. What a new k-list changes is only what carries a ``k`` index --
        the plane-wave spheres and their padding width, ``|k+G|^2``, the box
        indices, the stick layout, and the projectors ``vkb(k)``.

        This is what makes a Berry phase affordable. Every quantity in
        :mod:`pypresso.topology` is built from states on a k-mesh or a loop, and
        the natural way to write that is one call per row -- which, when each
        call built a whole :class:`Calculation`, meant rebuilding the dense
        G set and ``Q_ij(G)`` every time: **~1 GB and 70 s per call** on
        bismuthene, more than the states of the entire mesh cost to hold. The
        choice was then between streaming (cheap in memory, ruinous in time) and
        holding the whole mesh (the opposite). Sharing the setup removes the
        choice.

        The k-independent arrays are *shared*, not copied, so ``n`` calls hold
        one copy between them.
        """
        if self.spiral:
            # A spiral's basis is built on the *shifted* list, so replacing the
            # k-points here would silently rebuild it without the shift. The
            # counterpart that keeps the shift is :meth:`at_spiral_q`; a new
            # k-list for a spiral means a new calculation.
            raise NotImplementedError(
                "at_kpoints on a spin spiral is not implemented: the basis is "
                "built at k +- q/2, so a new k-list has to rebuild both halves "
                "(see at_spiral_q, which does the same for a new q)"
            )
        system = eqx.tree_at(lambda sys: sys.kpoints, self.system, kpoints)
        moved = copy.copy(self)
        # As in ``at_positions`` and ``at_spiral_q``: a compiled ``dE/dq`` holds
        # the k-list and the sphere it was built with, so it cannot cross this.
        moved.__dict__.pop("_spiral_gradient", None)
        moved.__dict__.pop("_spiral_gradient_chunk", None)
        # The tetrahedra are the k-grid's own object -- corner indices into the
        # irreducible list, built from this grid's equivalence -- so on a
        # different k-set they index the new eigenvalues with the old grid's
        # corners. A JAX gather clamps rather than raising, so the Fermi level
        # and the weights would come out wrong in silence.
        moved.__dict__.pop("_tetrahedra", None)
        # The force's gradient takes the positions as an argument, so it crosses
        # ``at_positions`` legitimately -- but it closes over this k-set and its
        # weights, which is what changes here.
        moved.__dict__.pop("_energy_gradient", None)
        moved.system = system
        # The list everything with a ``k`` index is built on. Without a spiral it
        # *is* the system's, and it has to move with it: leaving it stale gives
        # the pseudo-atomic orbitals the old k-points, which is a merely worse
        # starting guess (they are diagonalised afterwards) and a genuinely wrong
        # set of Hubbard projectors (they are not).
        moved.basis_kpoints = kpoints
        # A genuinely different k-set, so the crystal coordinates that
        # ``at_strain`` rebuilds from are a different list too. Carrying the old
        # one over -- which ``copy.copy`` does by itself -- would give a stress
        # or a velocity on a band path the SCF grid's k-points.
        moved._kcrystal = np.asarray(kpoints.crystal(self.system.cell))

        smooth, cell = self.basis.smooth, self.system.cell
        planewaves = build_plane_wave_basis(smooth, kpoints, cell, system.ecutwfc)
        moved.basis = Basis(
            dense=self.basis.dense, smooth=smooth, planewaves=planewaves
        )
        moved.kinetic = planewaves.kinetic(smooth, kpoints, cell)
        moved.fft_index = planewaves.fft_index(smooth)
        moved.sticks = build_sticks(moved.fft_index, planewaves.mask, smooth.grid)

        # The projectors are rebuilt whole: their radial half is tabulated
        # against ``|k+G|``, so unlike a change of position this is not a matter
        # of a new structure factor over a cached core.
        moved.projector_core = build_projector_core(
            self.pseudos, system.structure, cell, smooth, planewaves, kpoints
        )
        moved.projectors = moved.projector_core.at_positions(
            system.structure.positions, qq=self.projectors.qq
        )
        # ``wfcU`` carries a k index, so a new k-list needs new projectors --
        # QE rebuilds them in ``orthoUwfc`` at the start of every run for the
        # same reason. Without this a band-structure run on a converged density
        # would apply the Hubbard potential through the *old* k-points'
        # projectors, which is silently wrong rather than an error.
        if self.hubbard is not None:
            moved.wfcU = moved._build_hubbard_projectors()
        return moved

    def at_kcart(self, kcart) -> "Calculation":
        """The same calculation with the k-points moved, at a **frozen sphere**.

        :meth:`at_kpoints` keeps the k-points and rebuilds the basis; this keeps
        the basis and moves the k-points. It is the ``k`` counterpart of
        :meth:`at_spiral_q` with ``rebuild_basis = False``, and exists for the same
        reason: which plane waves satisfy ``|k+G|^2 <= ecutwfc`` is a host-side
        decision that cannot be traced, while the arithmetic on top of it can.
        Given a *traced* ``kcart`` -- ``(nk, 3)`` in 1/bohr -- this returns a
        calculation whose ``k``-dependent arrays are traced, which is what makes
        the velocity operator a ``jvp`` (:mod:`pypresso.response.velocity`).

        Only two things carry ``k``: ``|k+G|^2`` and ``vkb(k)``, plus ``wfcU``
        when there is a Hubbard ``U``, whose atomic orbitals live at ``k+G`` as
        the projectors do. The G sets, the FFT box, the stick layout, the mask,
        the local potential, the augmentation charge and the Ewald sum are all
        properties of the cell and the atoms, so they are shared rather than
        rebuilt.

        **The returned calculation has a** ``system.kpoints`` **that is not where
        its arrays are**, exactly as :meth:`at_spiral_q` leaves ``spiral_q`` behind:
        the k-points decide array lengths and so cannot hold a tracer. Every
        host-side consumer wants the original; the differentiated operator never
        reads it back. For an actual *move* of the k-list, use
        :meth:`at_kpoints`, which rebuilds the sphere.
        """
        if self.spiral:
            # A spiral has its spheres centred at ``k +- q/2``, so one cartesian
            # k-list is not what its arrays are built on. :meth:`at_spiral_q`
            # moves both halves together and is what a spiral differentiates.
            raise NotImplementedError(
                "at_kcart on a spin spiral is not implemented: the basis is "
                "built at k +- q/2, so moving k means moving both halves "
                "(see at_spiral_q)"
            )
        smooth, cell = self.basis.smooth, self.system.cell
        planewaves = self.basis.planewaves
        moved = copy.copy(self)
        # Both compiled gradients close over the sphere *and* the k-points they
        # were built with, so neither can follow this.
        moved.__dict__.pop("_spiral_gradient", None)
        moved.__dict__.pop("_spiral_gradient_chunk", None)
        moved.__dict__.pop("_velocity", None)
        moved.__dict__.pop("_tetrahedra", None)
        moved.__dict__.pop("_energy_gradient", None)

        moved.kinetic = planewaves.kinetic(smooth, self.basis_kpoints, cell, kcart)
        moved.projector_core = build_projector_core(
            self.pseudos, self.system.structure, cell, smooth, planewaves,
            self.basis_kpoints, kcart,
        )
        moved.projectors = moved.projector_core.at_positions(
            self.system.structure.positions, qq=self.projectors.qq
        )
        if self.hubbard is not None:
            moved.wfcU = moved._build_hubbard_projectors(kcart)
        return moved

    def at_spiral_q(self, q_crystal, rebuild_basis: bool = True) -> "Calculation":
        """The same calculation at a different spin-spiral wavevector.

        :meth:`at_kpoints` in the one direction a spiral moves: ``q`` changes
        where the two components' spheres are centred and nothing else. The
        cell, the atoms, both G sets, the local potential, the core charge, the
        augmentation charge, the Ewald sum and the radial tables are all
        independent of it, and an ``E(q)`` scan is a loop over this method for
        that reason (:mod:`pypresso.workflows.spiral`).

        ``rebuild_basis = False`` is the counterpart of :meth:`at_positions`:
        it keeps *this* calculation's plane-wave spheres -- which plane waves
        are in them, their padding, their mask, their box indices and their
        stick layout -- and rebuilds only what is arithmetic on top of them,
        ``|k+G|^2`` and ``vkb(k +- q/2)``. It is therefore **traceable**: given
        a traced ``q`` it returns a calculation whose ``q``-dependent arrays are
        traced, which is what makes ``dE/dq`` a ``grad``
        (:mod:`pypresso.forces.spiral`). Which plane waves are inside the cutoff
        is a host-side decision that cannot be traced, and it is also
        piecewise constant in ``q``, so freezing it loses no derivative except
        at the isolated wavevectors where a plane wave crosses the cutoff.

        It is the wrong thing to use for an actual *move*: after a step of any
        size the frozen sphere is no longer the one ``q`` asks for, and the
        energy it gives is the one a slightly wrong cutoff would give. A
        relaxation therefore evaluates the gradient with ``rebuild_basis =
        False`` and moves with ``rebuild_basis = True``.
        """
        if not self.spiral:
            raise ValueError(
                "at_spiral_q needs a calculation that is already a spiral: "
                "spiral_q decides the basis, which is built once"
            )
        smooth, cell = self.basis.smooth, self.system.cell
        moved = copy.copy(self)
        # The compiled ``dE/dq`` closes over the sphere it was built with, so it
        # cannot follow a calculation whose sphere is a different one. Dropping
        # it here rather than letting ``copy.copy`` carry it across is the whole
        # of that: a stale one would be silently evaluated at the old cutoff.
        moved.__dict__.pop("_spiral_gradient", None)
        moved.__dict__.pop("_spiral_gradient_chunk", None)

        if not rebuild_basis:
            # ``spiral_q`` stays a *static* field -- static fields cannot hold a
            # tracer, and this one decides array lengths -- so the system is
            # left at the wavevector the frozen basis belongs to and only the
            # arrays move. **The returned calculation's ``system.spiral_q`` is
            # therefore not where its arrays are**, which is exactly right for
            # every host-side consumer of it (array lengths, the ``spiral``
            # flag) and wrong for anything that reports it. The caller here is
            # the differentiated energy, which never reads it back.
            planewaves = self.basis.planewaves
            kcart = spiral_kcart(self.system.kpoints, q_crystal, cell)
            moved.kinetic = planewaves.kinetic(smooth, self.basis_kpoints, cell, kcart)
            moved.projector_core = build_projector_core(
                self.pseudos, self.system.structure, cell, smooth, planewaves,
                self.basis_kpoints, kcart,
            )
            moved.projectors = moved.projector_core.at_positions(
                self.system.structure.positions, qq=self.projectors.qq
            )
            return moved

        # ``spiral_q`` is a *static* field -- it decides array lengths -- so it
        # is replaced as a dataclass field rather than through ``tree_at``,
        # which only reaches pytree leaves.
        system = dataclasses.replace(
            self.system, spiral_q=tuple(float(v) for v in q_crystal)
        )
        moved.system = system
        moved.basis_kpoints = spiral_kpoints(system.kpoints, system.spiral_q, cell)
        planewaves = build_plane_wave_basis(
            smooth, moved.basis_kpoints, cell, system.ecutwfc
        )
        moved.basis = Basis(
            dense=self.basis.dense, smooth=smooth, planewaves=planewaves
        )
        moved.kinetic = planewaves.kinetic(smooth, moved.basis_kpoints, cell)
        moved.fft_index = planewaves.fft_index(smooth)
        moved.sticks = build_sticks(moved.fft_index, planewaves.mask, smooth.grid)
        moved.projector_core = build_projector_core(
            self.pseudos, system.structure, cell, smooth, planewaves,
            moved.basis_kpoints,
        )
        moved.projectors = moved.projector_core.at_positions(
            system.structure.positions, qq=self.projectors.qq
        )
        return moved

    @property
    def is_ultrasoft(self) -> bool:
        return self.augmentation is not None

    @property
    def is_paw(self) -> bool:
        return self.paw is not None

    def potential(self, rho_r: jnp.ndarray, field_scale: float = 1.0, field=None,
                  tau: jnp.ndarray | None = None):
        """``v_of_rho`` for this calculation: Hartree plus exchange-correlation.

        Everything the potential needs and the density does not carry -- the
        core charge on both grids, and which functional is in use -- comes from
        here rather than from a default, so that the same density cannot produce
        two different potentials depending on which call site built it.
        """
        potential = _potential_of_rho(
            rho_r,
            self.basis.dense,
            self.system.cell,
            self.rho_core,
            self.functional,
            self.rho_core_g,
            self.quantization_axis,
            tau,
        )
        field = self.magnetic_field if field is None else field
        if field is None:
            return potential
        # ``add_bfield``, and QE calls it from *inside* ``v_of_rho`` for a
        # reason that is not stylistic: the field is then part of ``v_scf``, so
        # every eigenvalue feels it and ``deband`` removes it again, which is
        # what keeps it out of the reported total energy.
        v_field, e_field, e_constraint = _field_potential(
            field, rho_r, self.system.cell, field_scale
        )
        return Potential(
            v_scf=potential.v_scf + v_field,
            ehart=potential.ehart,
            etxc=potential.etxc,
            e_field=e_field,
            e_constraint=e_constraint,
            meta_c=potential.meta_c,
        )

    def onecenter(self, becsum_, meta_c=None):
        """``(epaw, ddd_paw)`` for the current ``becsum``, as a block matrix.

        ``(0, None)`` when no species is PAW. ``ddd_paw`` is
        ``(nspin_mag, nkb, nkb)``: the one-centre potential differs between the
        components exactly as the grid potential does, and there are as many of
        them as the density has -- one for a nonmagnetic spin-orbit run, whose
        spin structure comes entirely from ``fcoef`` afterwards and not from
        the one-centre terms.
        """
        if self.paw is None:
            return jnp.asarray(0.0), None
        if self.functional.is_meta and meta_c is None:
            # ``c`` is an average over the *cell*, so no sphere can compute its
            # own and the plane-wave part has to hand it down. Refused rather
            # than defaulted: silently using 1 here while the grid used 1.03
            # would make the two halves of the same potential belong to two
            # different functionals, and the only symptom would be a slightly
            # wrong gap.
            raise ValueError(
                f"{self.functional.name} needs its cell-averaged c passed to "
                "onecenter(): the PAW spheres and the plane-wave grid must use "
                "the same one. Potential.meta_c is where it comes from"
            )
        energy, blocks = _paw_onecenter(
            self.paw, becsum_, meta_c, self.quantization_axis
        )
        return energy, jnp.stack([
            self.augmentation.block_matrix(
                tuple(None if b is None else b[spin] for b in blocks)
            )
            for spin in range(self.nspin_mag)
        ])

    def becsum(self, wavefunctions, weights) -> tuple:
        """``becsum`` for every ultrasoft species, or ``()`` when there are none."""
        if not self.is_ultrasoft:
            return ()
        if self.noncolin:
            values = self._noncollinear_becsum(wavefunctions, weights)
        else:
            values = becsum(
                wavefunctions, self.projectors.vkb, weights, self.species_channels,
                self.k_batch,
            )
        if self._becsum_symmetry is not None:
            values = self._becsum_symmetry.apply(values)
        return values

    def _noncollinear_becsum(self, wavefunctions, weights) -> tuple:
        """``sum_bec`` then ``add_becsum_so``, per species.

        The projector occupations are accumulated as a spin-density *matrix* and
        only then contracted with the spin-orbit coefficients into the
        ``nspin_mag`` real components everything downstream wants. Doing it in
        that order is not an implementation detail: the intermediate is the only
        place the two spin components of the spinor still appear separately, and
        it is what a fully-relativistic species needs in order to know which
        ``j`` shell its occupation belongs to.
        """
        spinors = spinor_becsum(
            wavefunctions[0], self.projectors.vkb, weights[0], self.species_channels,
            self.k_batch,
        )
        values = []
        for t, block in enumerate(spinors):
            if block is None:
                values.append(None)
                continue
            values.append(
                becsum_transform(
                    jnp.asarray(self.spin_orbit[t].fcoef).astype(block.dtype),
                    block,
                    self.nspin_mag,
                )
            )
        return tuple(values)

    def coefficients(self, potential: jnp.ndarray, ddd_paw=None) -> jnp.ndarray | None:
        """``newd``: the ``D_ij`` the Hamiltonian should use with this potential.

        ``PW/src/newd_acc.f90``. ``D_ij^a = D_ij^(0) + int V_eff(r) Q_ij^a(r) dr``
        with ``V_eff`` the **total** local potential -- ``vltot`` included, which
        ``newq_acc`` folds in through its ``skip_vltot = .false.`` argument. The
        integral is done on the dense grid in G space, where ``Q_ij(G)`` already
        is. ``None`` means "nothing to rebuild": the norm-conserving case, where
        the file's ``D_ij`` is the answer for the whole run.
        """
        if self.noncolin:
            return self._noncollinear_coefficients(potential, ddd_paw)
        if not self.is_ultrasoft:
            return None
        dense = self.basis.dense
        deeq = _newd(
            potential, dense.fft_index, self.projectors.dij, self.augmentation
        )
        # ``add_paw_to_deeq``: the one-centre coefficients enter the nonlocal
        # term in exactly the same place the ultrasoft integral does.
        return deeq if ddd_paw is None else deeq + ddd_paw

    def _noncollinear_coefficients(self, potential, ddd_paw=None):
        """``deeq_nc``: the ``(2, 2, nkb, nkb)`` coefficients of a spinor run.

        Three things happen in the order ``newd_us`` does them, and the order is
        the point:

        1. the augmentation charge is integrated against each component of the
           potential, giving one *scalar* matrix per component;
        2. ``add_paw_to_deeq`` adds PAW's one-centre coefficients to those
           scalars -- **before** the spin transform, not after, because they are
           an addition to the same integral;
        3. only then are the components recombined into spin blocks and
           sandwiched between the spin-orbit coefficients.

        Adding ``ddd_paw`` after step 3 instead would put the one-centre term
        into the wrong spin structure, which converges perfectly well to the
        wrong answer.

        A norm-conserving spin-orbit run has no step 1 or 2 at all -- there is
        no augmentation charge to integrate -- and its coefficients are
        ``dvan_so``, fixed for the whole run.
        """
        if not self.is_ultrasoft:
            return self.dvan_so
        dense = self.basis.dense
        components = jnp.stack([
            self.augmentation.block_matrix(
                self.augmentation.integrals(r_to_g(channel, dense.fft_index))
            )
            for channel in potential
        ])
        if ddd_paw is not None:
            components = components + ddd_paw
        return _newd_noncollinear(components, self.dvan_so, self.fcoef_matrix)

    def augmented(self, rho_r: jnp.ndarray, becsum_) -> jnp.ndarray:
        """``addusdens``: the augmentation charge added to a real-space density."""
        if not self.is_ultrasoft:
            return rho_r
        dense = self.basis.dense
        return _addusdens(rho_r, dense.fft_index, dense.grid, self.augmentation, becsum_)

    @property
    def state_fft_index(self) -> jnp.ndarray:
        """The FFT index a whole *spinor* at one k-point needs.

        ``(nk, npwx)`` normally, and ``(nk, 2, npwx)`` for a spin spiral, where
        the two components are on different spheres.
        """
        if not self.spiral:
            return self.fft_index
        nk = self.system.kpoints.nk
        return jnp.stack([self.fft_index[:nk], self.fft_index[nk:]], axis=1)

    @property
    def state_kinetic(self) -> jnp.ndarray:
        """``|k+G|^2`` in the layout a whole *spinor* is stored in.

        ``(nk, 2 npwx)``, the sibling of :attr:`state_fft_index` and the same
        rule: a spiral's two components read different rows of ``kinetic``, so
        the two blocks are laid end to end along the plane-wave axis rather than
        one row being used twice. It is ``SpinorHamiltonian._as_state`` written
        where something that is not the Hamiltonian can reach it -- the energy
        differentiated for ``dE/dq`` is the caller
        (:mod:`pypresso.forces.spiral`).
        """
        if not self.spiral:
            return jnp.concatenate([self.kinetic, self.kinetic], axis=-1)
        nk = self.system.kpoints.nk
        return jnp.concatenate([self.kinetic[:nk], self.kinetic[nk:]], axis=-1)

    def symmetrize(self, rho_r: jnp.ndarray) -> jnp.ndarray:
        """Impose the crystal symmetry on a real-space density."""
        if self._symmetry_maps is None:
            return rho_r
        gvectors = self.basis.dense
        if self.nspin_mag == 4:
            return _symmetrize_noncollinear(
                rho_r, gvectors.fft_index, gvectors.grid, self._symmetry_maps,
                self._magnetization_rotations,
            )
        return _symmetrize(rho_r, gvectors.fft_index, gvectors.grid, self._symmetry_maps)

    def symmetrize_directional(self, fields: jnp.ndarray) -> jnp.ndarray:
        """Impose the crystal symmetry on three densities that form a vector.

        ``LR_Modules/symdvscf.f90`` at ``q = 0``. A linear response to a
        perturbation along a *direction* is not three scalar densities: a
        symmetry operation rotates the directions into each other as well, so
        the average that a symmetry-reduced k-set needs is

            drho_a(r) <- (1/N) sum_S R_ab drho_b({S|f}^-1 r),

        with ``R`` the plain cartesian rotation. It is
        :func:`~pypresso.system.symmetry.symmetrize_magnetization`'s
        construction without the axial sign -- an induced charge density is a
        **polar** vector under the group where a magnetization is an axial one,
        and applying the wrong one is a different symmetry rather than a worse
        average.

        **Skipping it entirely is not an option a shifted grid offers**, which
        is the trap this method exists for. A response is direction-dependent,
        so the usual escape -- run the *whole* k-grid, where the reduction has
        nothing to put back -- only works if that grid is closed under the point
        group, and a **shifted** Monkhorst-Pack grid is not: on fcc silicon
        2304 of the 3072 rotation images of a shifted 4x4x4 grid land off it.
        An unshifted grid is closed exactly, and both routes are tested against
        each other (``tests/regression/test_response.py``).

        Args:
            fields: ``(3, nspin_mag, n1, n2, n3)`` real, on the dense grid.
        """
        if self._symmetry_maps is None:
            return fields
        gvectors = self.basis.dense
        permutations, phases = self._symmetry_maps
        rotations = jnp.asarray(cartesian_rotations(self.system.cell, self.symmetries))

        def channel(three):
            in_g = jnp.stack([r_to_g(f, gvectors.fft_index) for f in three])
            out_g = symmetrize_vector_density(in_g, permutations, phases, rotations)
            return jnp.stack([
                jnp.real(g_to_r(component, gvectors.fft_index, gvectors.grid))
                for component in out_g
            ])

        moved = jnp.moveaxis(jnp.asarray(fields), 1, 0)  # (nspin, 3, ...)
        return jnp.moveaxis(jnp.stack([channel(c) for c in moved]), 0, 1)

    def symmetrize_atom_tensor(self, tensors) -> np.ndarray:
        """``symtensor``: a rank-2 tensor per atom, carried between atoms.

        Born effective charges. Here rather than at the two call sites in
        :mod:`pypresso.response.efield` and :mod:`pypresso.response.born`
        because both of them reached for :attr:`symmetries` directly and so
        **symmetrised a ``nosym`` run** -- the group is kept whole beside
        :attr:`use_symmetry` and is not by itself the group a run uses. The
        stress and the forces got this right by writing the same two-clause
        guard a third and fourth time; writing it once is what this method is
        for.
        """
        from pypresso.system.symmetry import atom_mapping, symmetrize_atom_tensor

        tensors = np.asarray(tensors)
        if not self.use_symmetry:
            return tensors
        return np.asarray(symmetrize_atom_tensor(
            tensors, self.system.cell, self.symmetries,
            atom_mapping(self.system.cell, self.system.structure, self.symmetries),
        ))

    def symmetrize_atom_pair_tensor(self, tensors) -> np.ndarray:
        """``symdynph_gq`` at ``q = 0``: the force constants' two atom indices.

        The companion of :meth:`symmetrize_atom_displacement` and **not** its
        mirror image: this one carries two atom labels and no spatial argument,
        so it uses ``irt`` where the displacement density needs ``irt^-1``
        (``PLAN.md`` P28a, and the docstrings of both functions in
        :mod:`pypresso.system.symmetry`). Guarded on :attr:`use_symmetry` for
        the reason :meth:`symmetrize_atom_tensor` is.
        """
        from pypresso.system.symmetry import atom_mapping, symmetrize_atom_pair_tensor

        tensors = np.asarray(tensors)
        if not self.use_symmetry:
            return tensors
        return np.asarray(symmetrize_atom_pair_tensor(
            tensors, self.system.cell, self.symmetries,
            atom_mapping(self.system.cell, self.system.structure, self.symmetries),
        ))

    def symmetrize_cartesian_tensor(self, tensor) -> np.ndarray:
        """``symmatrix3`` and its rank-4 sibling: a tensor of any rank.

        The third derivatives are what this is for. P26's elasto-optic tensor
        carries two field labels and two strain labels and P35's ``chi^(2)``
        three field labels, so both are wedge sums with more than one free
        index -- which is the *same* statement that makes the stress need
        ``symmatrix`` and the forces ``symvector``, at a rank Fortran had to
        write out again and this does not.

        Guarded on :attr:`use_symmetry` for the reason
        :meth:`symmetrize_atom_tensor` is.
        """
        from pypresso.system.symmetry import symmetrize_cartesian_tensor

        tensor = np.asarray(tensor)
        if not self.use_symmetry:
            return tensor
        return np.asarray(symmetrize_cartesian_tensor(
            tensor, self.system.cell, self.symmetries
        ))

    def symmetrize_atom_cartesian_tensor(self, tensors) -> np.ndarray:
        """``symtensor3``: a rank-3 tensor per atom, carried between atoms.

        The Raman tensors. :meth:`symmetrize_atom_tensor` at the rank P35's
        object has, and it uses ``irt`` rather than its inverse for the reason
        that one does -- there is no spatial argument here, only labels.

        Args:
            tensors: ``(nat, 3, ..., 3)``, the atom axis leading.
        """
        from pypresso.system.symmetry import (
            atom_mapping, symmetrize_atom_cartesian_tensor,
        )

        tensors = np.asarray(tensors)
        if not self.use_symmetry:
            return tensors
        return np.asarray(symmetrize_atom_cartesian_tensor(
            tensors, self.system.cell, self.symmetries,
            atom_mapping(self.system.cell, self.system.structure, self.symmetries),
        ))

    def symmetrize_atom_displacement(self, fields: jnp.ndarray) -> jnp.ndarray:
        """:meth:`symmetrize_directional` for the ``3 nat`` displacement patterns.

        ``LR_Modules/symdvscf.f90`` at ``q = 0``, for the *phonon* perturbation
        rather than the electric field's. The difference is one index and it is
        not cosmetic: an operation rotates the displacement direction **and**
        carries it to the atom it maps onto, so the average runs over both
        (:func:`~pypresso.system.symmetry.symmetrize_atom_displacement_density`).
        On diamond silicon the operations that exchange the two sublattices are
        half the group.

        Args:
            fields: ``(nat, 3, nspin_mag, n1, n2, n3)`` real, on the dense grid.
        """
        if self._symmetry_maps is None:
            return fields
        gvectors = self.basis.dense
        permutations, phases = self._symmetry_maps
        rotations = jnp.asarray(cartesian_rotations(self.system.cell, self.symmetries))
        mapping = atom_mapping(
            self.system.cell, self.system.structure, self.symmetries
        )

        def channel(patterns):
            """``patterns``: (nat, 3, n1, n2, n3) for one spin channel."""
            nat, ncart = patterns.shape[:2]
            flat = patterns.reshape((nat * ncart,) + patterns.shape[2:])
            in_g = jnp.stack([r_to_g(f, gvectors.fft_index) for f in flat])
            out_g = symmetrize_atom_displacement_density(
                in_g.reshape((nat, ncart, -1)), permutations, phases,
                rotations, mapping,
            )
            back = jnp.stack([
                jnp.real(g_to_r(component, gvectors.fft_index, gvectors.grid))
                for component in out_g.reshape((nat * ncart, -1))
            ])
            return back.reshape(patterns.shape)

        moved = jnp.moveaxis(jnp.asarray(fields), 2, 0)  # (nspin, nat, 3, ...)
        return jnp.moveaxis(jnp.stack([channel(c) for c in moved]), 0, 2)

    def symmetrize_strain_response(self, fields: jnp.ndarray) -> jnp.ndarray:
        """:meth:`symmetrize_directional` for the nine **strain** perturbations.

        ``symdvscf`` again, for the perturbation P26 adds. A strain is a rank-2
        tensor, so an operation rotates *both* of its indices
        (:func:`~pypresso.system.symmetry.symmetrize_tensor_density`); the
        electric field's version rotates one and the phonon's rotates one and
        permutes the atoms.

        The escape hatch is the same one and so is the trap: a response computed
        on a symmetry-reduced k-set is not what the whole grid would give, and
        running the whole grid instead is only sound when that grid is closed
        under the point group -- true of an unshifted Monkhorst-Pack grid and
        false of a shifted one.

        Args:
            fields: ``(3, 3, nspin_mag, n1, n2, n3)`` real, on the dense grid.
        """
        if self._symmetry_maps is None:
            return fields
        gvectors = self.basis.dense
        permutations, phases = self._symmetry_maps
        rotations = jnp.asarray(cartesian_rotations(self.system.cell, self.symmetries))

        def channel(patterns):
            """``patterns``: (3, 3, n1, n2, n3) for one spin channel."""
            flat = patterns.reshape((9,) + patterns.shape[2:])
            in_g = jnp.stack([r_to_g(f, gvectors.fft_index) for f in flat])
            out_g = symmetrize_tensor_density(
                in_g.reshape((3, 3, -1)), permutations, phases, rotations
            )
            back = jnp.stack([
                jnp.real(g_to_r(component, gvectors.fft_index, gvectors.grid))
                for component in out_g.reshape((9, -1))
            ])
            return back.reshape(patterns.shape)

        moved = jnp.moveaxis(jnp.asarray(fields), 2, 0)  # (nspin, 3, 3, ...)
        return jnp.moveaxis(jnp.stack([channel(c) for c in moved]), 0, 2)

    def density(self, wavefunctions, weights, becsum_=None) -> jnp.ndarray:
        """The symmetrised output density from the occupied states.

        ``sum_band`` runs on the smooth grid, where the wavefunctions are; the
        result is lifted to the dense grid before it is symmetrised, because the
        dense grid is where the density is mixed, where the potential is built
        from it, and -- once there is an augmentation charge -- where the rest
        of it is added.
        """
        dense, smooth = self.basis.dense, self.basis.smooth
        if becsum_ is None:
            becsum_ = self.becsum(wavefunctions, weights)
        if self.noncolin:
            rho = _spinor_density_of_bands(
                wavefunctions[0], self.state_fft_index, smooth.grid, weights[0],
                self.system.cell, self.nspin_mag, self.k_batch,
            )
        else:
            rho = _density_of_bands(
                wavefunctions, self.fft_index, smooth.grid, weights, self.system.cell,
                self.k_batch,
            )
        return self.symmetrize(self.augmented(to_dense(rho, smooth, dense), becsum_))

    def kinetic_energy_density(self, wavefunctions, weights,
                               symmetrize: bool = True) -> jnp.ndarray:
        """``tau`` from the occupied states, on the **dense** grid, Ry.

        ``sum_band.f90``'s meta-GGA branch, lifted to the dense grid the same
        way the density is -- and for the same reason: the potential is built
        there. QE lifts it through G space too (``rho_r2g`` on the smooth grid,
        ``rho_g2r`` on the dense one, at the end of ``sum_band``).

        **Symmetrised, and it is not optional.** ``sum_band.f90`` calls
        ``sym_rho`` on ``rho%kin_g`` immediately after it calls it on
        ``rho%of_g``, with ``nspin`` and in the ``(up, down)`` representation --
        which is exactly this call. ``tau(r)`` is a scalar field of the crystal
        and has the crystal's symmetry; a sum over an irreducible wedge does
        not, and the gap between the two is not small. On QE's own silicon the
        unsymmetrised ``tau`` is **11% asymmetric** and running with it moves the
        eigenvalues by **0.47 eV** and the total energy by 1.3e-2 Ry
        (``tests/regression/test_mbj.py``). ``symmetrize = False`` exists so
        that number can be measured rather than asserted; nothing should use it.

        ``tau`` is a scalar under the point group in every regime this
        functional runs in, so the density's own ``sym_rho`` serves without a
        new routine -- unlike a *response*, which is a polar vector and needs
        :meth:`symmetrize_directional`.
        """
        if self.kplusg is None:
            raise ValueError(
                "this calculation's functional is not a meta-GGA, so k + G was "
                "never built; tau has no consumer here"
            )
        dense, smooth = self.basis.dense, self.basis.smooth
        if self.noncolin:
            # ``tau`` is a 2x2 matrix in spin space, carried on the Pauli basis
            # exactly as the density is -- a trace and an axial vector, and the
            # same ``nspin_mag`` decides whether the vector part exists at all.
            tau = _spinor_kinetic_of_bands(
                wavefunctions[0], self.state_fft_index, smooth.grid, weights[0],
                self.system.cell, self.kplusg, self.nspin_mag, self.k_batch,
            )
        else:
            tau = _kinetic_of_bands(
                wavefunctions, self.fft_index, smooth.grid, weights,
                self.system.cell, self.kplusg, self.k_batch,
            )
        tau = to_dense(tau, smooth, dense)
        return self.symmetrize(tau) if symmetrize else tau

    def hamiltonian(self, v_scf: jnp.ndarray, ddd_paw=None, hubbard=None) -> tuple:
        """One :class:`Hamiltonian` per spin channel.

        The kinetic term, the projectors and the local pseudopotential are
        shared; the self-consistent potential and ``D_ij`` are not. Returned as
        a tuple rather than one object with a spin axis because the eigensolver
        is written for a single Hamiltonian and a channel is simply another
        problem to solve -- which is also how QE sees it, its ``2 nks`` k-list
        differing only in which ``vrs(:, isk)`` each point reads.
        """
        # ``set_vrs`` adds the fixed local pseudopotential to the self-consistent
        # part on the dense grid, and ``interpolate`` hands the wavefunction
        # transforms a smooth-grid copy. ``newd`` reads the *dense* one, since
        # the augmentation charge it integrates against is only representable
        # there.
        # ``set_vrs``: the local pseudopotential is felt in full by both
        # channels of an (up, down) potential and only by the charge component
        # of an (n, m) one. That is *not* the rule an unpolarized density
        # follows -- see as_potential_components -- and using the density's rule
        # here halves the local pseudopotential in an LSDA run.
        total = v_scf + as_potential_components(self.vltot, self.nspin_mag)
        deeq = self.coefficients(total, ddd_paw)
        if self.noncolin:
            return (self._spinor_hamiltonian(total, deeq),)
        hamiltonians = []
        for spin in range(self.nspin):
            # ``vhpsi`` is a separate term, not a contribution to ``deeq``: it
            # has its own projectors and its own coefficients. QE *does* fold it
            # into ``deeq`` in ``add_vhub_to_deeq`` -- but only when
            # ``Hubbard_projectors = 'pseudo'``, where the Hubbard projectors
            # *are* the beta functions and the two terms therefore share a
            # separable form. That projector set is refused here.
            potential = to_smooth(total[spin], self.basis.dense, self.basis.smooth)
            hamiltonians.append(Hamiltonian(
                kinetic=self.kinetic,
                potential=potential,
                # the same potential with its xy plane contiguous, which is the
                # layout the stick transforms hold the field in
                potential_wave=jnp.moveaxis(potential, -1, -3),
                sticks=self.sticks,
                fft_index=self.fft_index,
                mask=self.basis.planewaves.mask,
                projectors=self.projectors,
                grid=self.basis.smooth.grid,
                resolves_differences=self.resolves_differences,
                deeq=None if deeq is None else deeq[spin],
                hubbard=None if hubbard is None else hubbard[spin],
            ))
        return tuple(hamiltonians)

    def _spinor_hamiltonian(self, total: jnp.ndarray, deeq) -> SpinorHamiltonian:
        """The single noncollinear Hamiltonian, at the given total potential."""
        potential = jnp.stack([
            to_smooth(component, self.basis.dense, self.basis.smooth)
            for component in total
        ])
        return SpinorHamiltonian(
            kinetic=self.kinetic,
            potential=potential,
            potential_wave=jnp.moveaxis(potential, -1, -3),
            spiral=self.spiral,
            sticks=self.sticks,
            fft_index=self.fft_index,
            mask=self.basis.planewaves.mask,
            projectors=self.projectors,
            deeq=deeq,
            grid=self.basis.smooth.grid,
            resolves_differences=self.resolves_differences,
            qq=self.qq_so,
        )

    def starting_density(self) -> jnp.ndarray:
        """The superposition of atomic charges, split by ``starting_magnetization``.

        ``atomic_rho_g``: the total is the sum of the species' tabulated atomic
        charges, and the magnetization is the same sum weighted by each species'
        ``starting_magnetization``. Nothing else in the calculation breaks the
        spin symmetry -- with every ``starting_magnetization`` at zero the two
        channels start identical, stay identical, and the run converges to the
        unpolarized solution, which for a symmetric crystal is always a
        stationary point. That is why QE demands the input variable rather than
        finding the magnetic state on its own.
        """
        dense = self.basis.dense
        if self.nspin_mag == 1:
            rho_g = starting_charge(
                self.pseudos, self.system.structure, self.system.cell, dense,
                self.nelec,
            )
            return jnp.real(g_to_r(rho_g, dense.fft_index, dense.grid))[None]

        if self.noncolin:
            return self._noncollinear_starting_density()

        rho_g, magnetization_g = starting_charge(
            self.pseudos, self.system.structure, self.system.cell, dense, self.nelec,
            magnetization=self.spin_weights[0] - self.spin_weights[1],
        )
        channels = jnp.stack([rho_g + magnetization_g, rho_g - magnetization_g]) / 2.0
        return jnp.real(g_to_r(channels, dense.fft_index, dense.grid))

    def _noncollinear_starting_density(self) -> jnp.ndarray:
        """``atomic_rho_g`` with ``nspin = 4``: charge, and a magnetization vector.

        The magnetization is the same superposition of atomic charges the
        collinear case builds, once per cartesian component, with each species'
        contribution weighted by its ``starting_magnetization`` *and* by the
        direction ``(angle1, angle2)`` points in. Three sums rather than one is
        the whole difference -- and it is why the angles are per species: two
        sublattices pointing different ways is exactly what a noncollinear
        calculation exists to describe, and a single common axis would make it a
        collinear one in disguise.
        """
        dense = self.basis.dense
        directions = self.magnetization_directions  # (ntyp, 3)
        magnitudes = self.starting_magnetization
        components = []
        rho_g = None
        for axis in range(3):
            rho_g, component = starting_charge(
                self.pseudos, self.system.structure, self.system.cell, dense,
                self.nelec, magnetization=magnitudes * directions[:, axis],
            )
            components.append(component)
        channels = jnp.stack([rho_g, *components])
        return jnp.real(g_to_r(channels, dense.fft_index, dense.grid))

    @property
    def starting_magnetization(self) -> np.ndarray:
        """``starting_magnetization`` per species, padded to ``ntyp``."""
        ntyp = self.system.structure.ntyp
        values = np.zeros(ntyp)
        given = np.asarray(self.system.starting_magnetization, dtype=float)
        values[: given.size] = given[:ntyp]
        return values

    @property
    def magnetization_directions(self) -> np.ndarray:
        """``(ntyp, 3)`` unit vectors from ``angle1``/``angle2``, in degrees.

        ``input.f90`` converts the input's degrees to radians and
        ``angle1``/``angle2`` are the polar and azimuthal angles of that
        species' moment; both default to zero, which points along ``z`` and is
        what makes an unspecified noncollinear run start out collinear.
        """
        ntyp = self.system.structure.ntyp
        theta = np.zeros(ntyp)
        phi = np.zeros(ntyp)
        for target, given in ((theta, self.system.angle1), (phi, self.system.angle2)):
            values = np.asarray(given, dtype=float)
            target[: values.size] = np.radians(values[:ntyp])
        return np.stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ], axis=1)

    @property
    def spin_weights(self) -> np.ndarray:
        """``0.5 (1 +- starting_magnetization)`` per species, or ``[1]``.

        The split ``PAW_atomic_becsum`` applies to the reference occupations,
        which is the same split :meth:`starting_density` applies to the atomic
        charge -- the two starting guesses have to agree about how polarized the
        atom is or the first iteration contradicts itself.
        """
        ntyp = self.system.structure.ntyp
        magnetization = np.zeros(ntyp)
        given = np.asarray(self.system.starting_magnetization, dtype=float)
        magnetization[: given.size] = given[:ntyp]
        if self.nspin == 1:
            return np.ones((1, ntyp))
        return 0.5 * np.stack([1.0 + magnetization, 1.0 - magnetization])

    def starting_becsum(self) -> tuple:
        """The projector occupations of isolated atoms (``PAW_atomic_becsum``).

        PAW needs a ``becsum`` before there are any wavefunctions, because the
        one-centre potential enters the very first Hamiltonian. QE takes it from
        the reference occupations the pseudopotential file records, spread evenly
        over the ``2l+1`` channels of each projector -- the atom's own ground
        state, which is the same information the starting density comes from.
        """
        if not self.is_ultrasoft:
            return ()
        becsum = []
        for t, atoms in enumerate(self.augmentation.species_atoms):
            channels = self.species_channels[t]
            if channels is None or not atoms:
                becsum.append(None)
                continue
            pseudo = self.pseudos[t]
            occupations = (
                pseudo.paw.occupations if pseudo.paw is not None
                else np.zeros(pseudo.nbeta)
            )
            diagonal = np.array([
                occupations[nb] / (2 * l + 1) for nb, l, _ in projector_channels(pseudo)
            ])
            per_spin = self._becsum_split(t) [:, None] * diagonal[None, :]
            becsum.append(
                jnp.broadcast_to(
                    jnp.stack([jnp.diag(jnp.asarray(row)) for row in per_spin])[:, None],
                    (self.nspin_mag, len(atoms), pseudo.nh, pseudo.nh),
                )
            )
        return tuple(becsum)

    def _becsum_split(self, t: int) -> np.ndarray:
        """How one species' reference occupations divide over ``nspin_mag``.

        ``PAW_atomic_becsum``. Collinear: the ``(1 +- m)/2`` pair. Noncollinear
        with a magnetization: the occupation itself, then ``m`` times the
        direction it points in -- which is the same information written in the
        other representation, and has to agree with what
        :meth:`_noncollinear_starting_density` does to the charge or the first
        iteration contradicts itself.
        """
        if not self.noncolin:
            return self.spin_weights[:, t]
        if self.nspin_mag == 1:
            return np.ones(1)
        moment = self.starting_magnetization[t] * self.magnetization_directions[t]
        return np.concatenate([[1.0], moment])

    def starting_wavefunctions(self, hamiltonians, nbnd: int, span=None) -> jnp.ndarray:
        """The first guess at the wavefunctions, from the atomic orbitals.

        QE's ``wfcinit``: build the pseudo-atomic orbitals of every atom, then
        diagonalise the Hamiltonian inside their span. What comes out is not the
        answer, but it is close enough that the first Davidson call costs two
        steps instead of eight -- the atoms already know roughly where their
        electrons are, and the pseudopotential file carries that knowledge.

        Falls back to random vectors for a pseudopotential with no ``PP_PSWFC``
        section, and tops up with them when a species has fewer orbitals than
        the calculation has bands.

        ``span`` replaces the atomic orbitals with any other set of vectors --
        ``(nk, nvec, npol npwx)`` shared by every channel, or one set per channel
        with a leading axis. **It is a span and not a set of wavefunctions**:
        what follows is the same Rayleigh-Ritz, so the vectors need not be
        orthonormal in the target's overlap operator, need not number ``nbnd``,
        and need not be sorted. That is what makes it safe to hand over the
        converged states of a *different* spin regime
        (:mod:`pypresso.scf.continuation`).
        """
        if span is None:
            atomic = atomic_wavefunctions(
                self.pseudos, self.system.structure, self.system.cell,
                self.basis.smooth, self.basis.planewaves, self.basis_kpoints,
            )
            if self.noncolin:
                atomic = self._as_spinors(atomic)
        else:
            atomic = jnp.asarray(span)
            expected = self.npol * self.basis.npwx
            # ``hamiltonians[0].nk`` and not ``basis_kpoints.nk``: a spiral's
            # basis list is the doubled one (``k +- q/2``) while its *states*
            # number one per physical k-point.
            nk = hamiltonians[0].nk
            if atomic.shape[-1] != expected or atomic.shape[-3] != nk:
                raise ValueError(
                    f"span has shape {tuple(atomic.shape)}; this calculation "
                    f"needs (..., {nk}, nvec, {expected})"
                )

        # One span per channel, or one shared by all of them -- which is what
        # the atomic orbitals are, since what splits the channels is the
        # Hamiltonian they are diagonalised inside and not the vectors.
        per_channel = atomic.ndim == 4
        missing = nbnd - atomic.shape[-2]
        if missing > 0:
            # Aluminium has four atomic orbitals and a smeared calculation asks
            # for six bands; the rest are random, exactly as QE tops up.
            ndim = self.npol * self.basis.npwx
            # One row per *state*, which for a spiral means the two components'
            # rows side by side rather than one row used twice.
            reference = hamiltonians[0]
            kinetic = (
                reference.state_kinetic if self.noncolin
                else jnp.tile(self.kinetic, (1, self.npol))
            )
            mask = (
                reference.state_mask if self.noncolin
                else jnp.tile(self.basis.planewaves.mask, (1, self.npol))
            )
            extra = map_k(
                lambda arrays: starting_vectors(
                    None, missing, ndim, arrays[0], arrays[1], atomic.dtype
                ),
                (kinetic, mask),
                batch=self.k_batch,
            )
            if per_channel:
                extra = jnp.broadcast_to(extra[None], atomic.shape[:1] + extra.shape)
            atomic = jnp.concatenate([atomic, extra], axis=-2)

        # One Rayleigh-Ritz per channel. A shared span -- the atomic orbitals,
        # or another regime's states -- seeds both channels with the same
        # vectors, and what splits them is the Hamiltonian they are diagonalised
        # inside, which is already spin-split at the first iteration because the
        # starting density is.
        return jnp.stack([
            _rotate_all(
                hamiltonian, atomic[channel] if per_channel else atomic,
                nbnd, self.k_batch,
            )[1]
            for channel, hamiltonian in enumerate(hamiltonians)
        ])

    def _as_spinors(self, atomic: jnp.ndarray) -> jnp.ndarray:
        """Scalar atomic orbitals -> twice as many spinors, ``(nk, 2 n, 2 npwx)``.

        Each orbital is used twice, once in each spin component, which spans the
        same space as QE's ``atomic_wfc_nc``. It is *not* the same set of
        vectors: with spin-orbit coupling QE builds the ``j``-resolved
        spin-angle functions (``atomic_wfc_so``), which are already close to the
        eigenstates. The difference is entirely one of convergence -- both spans
        are then diagonalised by ``rotate_wfc``, and what comes out of that is
        what the SCF starts from -- so this is a slower start, not a different
        calculation.
        """
        if self.spiral:
            # The two halves of the doubled list are the two components'
            # orbitals -- at ``k + q/2`` and at ``k - q/2`` -- so each one seeds
            # its own component and neither seeds the other.
            nk = self.system.kpoints.nk
            upper, lower = atomic[:nk], atomic[nk:]
            up = jnp.concatenate([upper, jnp.zeros_like(upper)], axis=-1)
            down = jnp.concatenate([jnp.zeros_like(lower), lower], axis=-1)
            return jnp.concatenate([up, down], axis=1)
        nk, count, npwx = atomic.shape
        zero = jnp.zeros_like(atomic)
        up = jnp.concatenate([atomic, zero], axis=-1)
        down = jnp.concatenate([zero, atomic], axis=-1)
        return jnp.concatenate([up, down], axis=1)

    def diagonalize(self, hamiltonians, nbnd: int, psi0=None, ethr=None):
        """Solve at every k-point of every channel.

        Returns ``(eigenvalues, wavefunctions)`` shaped ``(nspin, nk, nbnd)``
        and ``(nspin, nk, nbnd, npwx)``.

        ``psi0`` is the previous iteration's wavefunctions. Passing them is what
        makes an iterative solver cheap after the first SCF iteration: the
        density moves a little, so the eigenvectors do too, and Davidson starts
        one step away from the answer instead of from a random guess.

        ``ethr`` is how accurately to converge each eigenvalue. A direct solver
        ignores both.
        """
        solved = [
            self.eigensolver(
                hamiltonian, nbnd, None if psi0 is None else psi0[spin], ethr,
                k_batch=self.k_batch,
            )
            for spin, hamiltonian in enumerate(hamiltonians)
        ]
        return (
            jnp.stack([values for values, _ in solved]),
            jnp.stack([vectors for _, vectors in solved]),
        )

    @property
    def two_fermi_energies(self) -> bool:
        """Whether the magnetization is constrained (QE's ``two_fermi_energies``).

        Set by ``tot_magnetization`` being given at all, not by its value:
        ``input.f90`` turns the sentinel into the flag, and the flag decides
        whether the channels share a Fermi level or each gets its own.
        """
        return self.nspin == 2 and self.system.tot_magnetization is not None

    def occupations(self, eigenvalues):
        """Occupation weights, plus whichever level statistic applies.

        ``eigenvalues`` is ``(nspin, nk, nbnd)`` and the weights come back the
        same shape.
        """
        weights = self.system.kpoints.weights
        scheme = self.system.occupations

        # How many electrons one band holds. It is the factor the k-point
        # weights already carry, and it appears here only where a *count* of
        # bands is taken -- a spinor band holds one electron, not two.
        degeneracy = 1 if self.noncolin else 2

        if scheme == "fixed":
            counts = (self.nelup, self.neldw) if self.two_fermi_energies else None
            wg, homo, lumo = fixed_occupations(
                eigenvalues, weights, self.nelec, degeneracy, counts=counts
            )
            if counts is None:
                return wg, {"homo": float(homo),
                            "lumo": None if lumo is None else float(lumo)}
            # ``iweights`` returns one level per channel and QE prints the pair
            # as ``ef_up``/``ef_dw``; the scalar HOMO and LUMO it also prints are
            # the extremes over both channels, which is why they can coincide --
            # a partly-filled degenerate shell in one channel puts its HOMO and
            # the other channel's LUMO at the same energy (the oxygen atom of
            # ``o-atom-fixed-lsda`` does exactly that).
            homo, lumo = np.asarray(homo), np.asarray(lumo)
            finite = lumo[np.isfinite(lumo)]
            return wg, {
                "homo": float(np.max(homo)),
                "lumo": float(np.min(finite)) if finite.size else None,
                "fermi_energy_up": float(homo[0]),
                "fermi_energy_down": float(homo[1]),
            }

        if scheme == "from_input":
            if self.system.input_occupations is None:
                raise ValueError("occupations='from_input' needs an OCCUPATIONS card")
            return input_occupations(
                self.system.input_occupations, eigenvalues, weights, degeneracy
            ), {}

        if scheme.startswith("tetrahedra"):
            # The tetrahedra are a property of the k-grid and the symmetry, not
            # of the density, so they are built once and kept. There is no
            # ``-TS`` term: the method integrates the true step function, which
            # is why QE prints no "smearing contrib." for a tetrahedron run.
            if getattr(self, "_tetrahedra", None) is None:
                self._tetrahedra = tetrahedra_for(
                    scheme, self.system.kpoints, self.symmetries, self.system.cell
                )
            counts = (self.nelup, self.neldw) if self.two_fermi_energies else None
            wg, ef = tetrahedron_occupations_spin(
                self._tetrahedra, eigenvalues, weights, self.nelec, counts=counts
            )
            if counts is None:
                return wg, {"fermi_energy": float(ef)}
            return wg, {
                "fermi_energy_up": float(ef[0]),
                "fermi_energy_down": float(ef[1]),
                "fermi_energy": 0.5 * (float(ef[0]) + float(ef[1])),
            }

        counts = (self.nelup, self.neldw) if self.two_fermi_energies else None
        wg, ef = smeared_occupations(
            eigenvalues, weights, self.nelec, self.system.degauss,
            self.system.smearing, counts=counts,
        )
        entropy = float(
            smearing_entropy(
                eigenvalues, weights, ef, self.system.degauss, self.system.smearing
            )
        )
        levels = {"smearing": entropy}
        if self.two_fermi_energies:
            levels["fermi_energy_up"] = float(ef[0])
            levels["fermi_energy_down"] = float(ef[1])
            # QE prints the mean as "the" Fermi energy in this case, with a
            # comment saying it is only to keep the printed value from being NaN.
            levels["fermi_energy"] = 0.5 * (float(ef[0]) + float(ef[1]))
        else:
            levels["fermi_energy"] = float(ef)
        return wg, levels


def _result_for_stress(calculation, eigenvalues, wg, wavefunctions, rho, terms):
    """The frozen state the stress is evaluated at, straight from the loop.

    :func:`~pypresso.forces.energy.state_from_result` wants a finished
    :class:`SCFResult`, which does not exist yet at the point ``tstress`` is
    honoured -- the stress is what is being put *into* it. This builds the same
    :class:`~pypresso.forces.energy.FrozenState` out of the loop's own arrays.
    """
    from pypresso.forces.energy import FrozenState

    return FrozenState(
        wavefunctions=wavefunctions,
        weights=jnp.asarray(wg),
        eigenvalues=jnp.asarray(eigenvalues),
        entropy=float(terms.get("smearing", 0.0)),
        density=rho,
    )


def _solve_residual(
    calculation, system, nbnd, rho, becsum_, ns_, conv_thr,
    scf_solver, options, mixing_beta, verbose, tau_=None,
):
    """Find the fixed point with a residual solver, before the mixing loop runs.

    Returns the converged state in the loop's own variables, plus the solver's
    own record of what it cost.

    Three choices are made here rather than inside the solver, because they are
    the driver's to make:

    * **``ethr`` is fixed and tight for the whole solve.** QE's schedule
      (``next_ethr``) loosens the diagonalisation while the density is still
      wrong, which is most of why a mixing run is cheap -- but it makes ``F`` a
      different function at every iteration, and a root-finder handed a moving
      target does not converge. The price is real and is stated in
      ``PERFORMANCE.md``: every evaluation of ``F`` here costs what a *converged*
      mixing iteration costs, not what an early one does.
    * **The Krylov system is preconditioned with Kerker by default**
      (``kerker=False`` turns it off), at ``beta = 1``, because a preconditioner
      for GMRES is an approximate *inverse Jacobian* and not a step length.
      **On a problem with more than one solution this is not a speed knob: it
      decides which one is found.** An inexact Newton step is only the Newton
      direction to the extent the inner solve converges, and a badly conditioned
      Krylov system degrades it towards a damped-mixing step -- which flows to
      the *stable* fixed point. On bcc iron the preconditioned solve reaches the
      non-magnetic saddle and the unpreconditioned one reaches the ferromagnetic
      ground state, both reporting an accuracy below ``conv_thr`` (``PLAN.md``
      P22, ``test_scf_solvers.py``).
    * **A warm-up of ordinary mixing** is allowed and defaults to none. Newton
      converges from a good enough guess and wanders from a bad one, and the
      atomic superposition is a bad one for exactly the systems worth using this
      on -- but a warm-up is a *mixing* step, so it pulls towards the stable
      solution and should be left at zero when an unstable one is wanted.
    """
    solver = get_scf_solver(scf_solver)
    ethr = options.pop("ethr", max(1.0e-3 * conv_thr, ETHR_MIN))
    warmup = int(options.pop("warmup", 0))
    precondition = options.pop("precondition", None)
    wavefunctions = None

    residual = make_residual(calculation, nbnd, ethr)
    x0 = residual.pack(
        rho, becsum_, ns_,
        None if residual.tau_shape is None
        else (tau_ if tau_ is not None else _starting_tau(rho, calculation)),
    )
    if precondition is True or (precondition is None and options.get("kerker", True)):
        precondition = kerker_preconditioner(
            calculation.basis.dense, calculation.system.cell, residual.shapes[0],
            beta=1.0,
            nelec=calculation.nelec,
        )
    elif precondition is not True and not callable(precondition):
        precondition = None
    options.pop("kerker", None)

    shape = residual.shapes[0]
    size = int(np.prod(shape))

    def accuracy_of(r):
        """The mixing loop's own convergence measure, so ``conv_thr`` means one
        thing in this file.

        ``rho_ddot`` on the density part -- ``becsum``'s share is the ``paw``
        term, which the loop adds through ``addusdens`` rather than separately --
        plus ``ns_ddot`` when there is a Hubbard U, exactly as the loop adds it.
        A residual solver that converged on a *different* measure than the mixer
        could not be compared with it at all."""
        accuracy = _accuracy(
            jnp.asarray(r[:size]).reshape(shape), calculation.basis.dense, calculation.system.cell
        )
        if residual.ns_shape is not None:
            # Sliced by ``unpack`` and not off the end of the vector: ``ns`` is
            # the last block only while nothing follows it, and ``tau`` does
            # (:class:`ScfResidual`). ``tau`` itself is deliberately *not* in
            # this measure -- ``conv_thr`` has to mean the same thing here as it
            # means in the mixing loop, which converges on the density alone.
            accuracy = accuracy + calculation.ns_accuracy(residual.unpack(r)[2])
        return float(accuracy)

    if warmup:
        # Ordinary mixing, run through the *residual's* own step so that the
        # ultrasoft and PAW parts of the state come along without a second
        # unpacking. Newton converges from a good enough guess and wanders from
        # a bad one, and the superposition of atomic charges is a bad one for
        # exactly the systems this solver is worth using on.
        warmup_mixing = options.pop("warmup_mixing", "kerker")
        warm_mixer = get_mixer(warmup_mixing, beta=mixing_beta)
        if warmup_mixing.lower() in PRECONDITIONED:
            # Its own preconditioner at ``mixing_beta``, *not* the Krylov
            # system's, which is built at beta = 1 because a preconditioner for
            # GMRES is an approximate inverse Jacobian and not a step length.
            # Reusing that one here silently ran the warm-up at beta = 1, which
            # is a different and much more aggressive mixer than the one the
            # caller asked for -- and converged the slab on its own, hiding the
            # solver it was supposed to be warming up.
            warm_mixer.precondition = kerker_preconditioner(
                calculation.basis.dense, calculation.system.cell, residual.shapes[0],
                beta=mixing_beta, nelec=calculation.nelec,
            )
        for _ in range(warmup):
            fx, wavefunctions = residual.step(x0, wavefunctions)
            x0 = np.asarray(warm_mixer.mix(x0, np.asarray(fx, dtype=float)), dtype=float)
        options["steps_already_taken"] = warmup

    result = solver(
        residual, x0, wavefunctions, accuracy_of, conv_thr=conv_thr,
        precondition=precondition, verbose=verbose, **options,
    )
    rho_out, becsum_out, ns_out, tau_out = residual.unpack(result.x)
    return rho_out, becsum_out, ns_out, tau_out, result.psi, result


def run_scf(
    system: System,
    pseudos: tuple[Pseudopotential, ...],
    nbnd: int | None = None,
    conv_thr: float = 1.0e-6,
    max_iterations: int = 100,
    mixing_mode: str = "anderson",
    mixing_beta: float = 0.7,
    calculation: Calculation | None = None,
    diagonalization: str | None = None,
    verbose: bool = False,
    k_batch: int | None | str = "default",
    starting_density: jnp.ndarray | None = None,
    starting_becsum: tuple | None = None,
    starting_ns: jnp.ndarray | None = None,
    starting_wavefunctions: jnp.ndarray | None = None,
    starting_from: object | None = None,
    starting_tau: jnp.ndarray | None = None,
    mixing_fixed_ns: int = 0,
    tstress: bool | None = None,
    scf_solver: str = "mixing",
    scf_solver_options: dict | None = None,
) -> SCFResult:
    """Run the self-consistent field loop to convergence.

    ``conv_thr`` is compared against the estimated self-consistency error --
    QE's ``dr2``, the Hartree energy of the density residual, in Ry -- so it
    means the same thing here as in a ``pw.x`` input.

    ``k_batch`` is how many k-points are held in flight at once: 1 is QE's
    ``k_loop``, ``None`` is one ``vmap`` over the whole axis, and it trades
    memory against speed without touching the answer
    (:mod:`pypresso.batching`). It is ignored when ``calculation`` is given,
    which already carries its own.

    ``starting_density`` replaces the superposition of atomic charges the run
    would otherwise start from. It is what a relaxation hands the next geometry
    (``PW/src/update_pot.f90``): the density of the previous ionic step is a far
    better guess than the atomic one and costs several SCF iterations less.
    ``starting_becsum`` is its ultrasoft/PAW counterpart, and ``starting_ns``
    its DFT+U one. **The mixed state is all three together**, and giving one
    without the others starts the run from two different states at once.

    ``starting_wavefunctions`` is a *span* for the first Rayleigh-Ritz rather
    than a set of wavefunctions -- see
    :meth:`Calculation.starting_wavefunctions`. It replaces the pseudo-atomic
    orbitals, and it is ignored by a residual solver, which starts its own.

    ``starting_from`` is all four at once, taken from another run's
    :class:`SCFResult` **and promoted into this run's spin regime**: a converged
    unpolarized density becomes the starting point of a collinear run, a
    collinear one of a noncollinear run, and spin-orbit coupling is switched on
    and off without going back to the atoms
    (:mod:`pypresso.scf.continuation`). Pass a
    :class:`~pypresso.scf.continuation.ContinuedState` instead of the result
    itself to control how the magnetization crosses -- carried, or seeded from
    this run's ``starting_magnetization``, which is what decides whether a
    magnetic run started from a non-magnetic one can leave the symmetric
    solution at all.

    ``starting_ns`` also decides *which* self-consistent solution a DFT+U run
    finds, which ``starting_density`` alone does not: ``init_ns`` fills the
    occupation matrix diagonally by **Hund's rule**, so the default start is
    strongly spin-polarised however small ``starting_magnetization`` is. A run
    meant to begin near the unpolarised solution has to say so here.

    ``mixing_fixed_ns`` is QE's ``&electrons`` variable of the same name: for
    that many iterations the Hubbard occupation matrix is held at its starting
    value while the density relaxes around it. It is how a magnetic insulator is
    steered towards a particular one of several self-consistent occupations, and
    it does nothing at all in a run with no Hubbard U.

    ``tstress`` is QE's ``&control`` switch of the same name: compute the stress
    tensor once the density has converged and put it on
    :attr:`SCFResult.stress`. ``None`` -- the default -- takes it from the
    input, through :attr:`~pypresso.system.builder.System.tstress`, so that a
    file carrying ``tstress = .true.`` gets a tensor here exactly as it does
    from ``pw.x``. It is not on unconditionally because it costs a strain
    gradient (roughly two SCF iterations, see `PERFORMANCE.md`), which is why
    ``run_pwscf`` calls ``stress()`` after ``electrons()`` rather than inside it.

    ``scf_solver`` chooses *how* the fixed point is found (rule R4, registry in
    :mod:`pypresso.scf.solvers`). ``"mixing"`` is the loop below and the
    default. ``"newton-krylov"`` instead solves ``F(rho) - rho = 0`` with an
    inexact Newton method whose Jacobian action comes from differentiating one
    SCF step, and then hands its answer to this same loop as a starting density
    -- which converges in one iteration and is what builds the result, so no
    energy term has a second implementation. **Anderson mixing is already a
    quasi-Newton method on that residual**, so the difference is an exact
    Jacobian against a fitted one, and it pays only where the fit is bad; see
    :mod:`pypresso.scf.solvers` and ``PLAN.md`` P22.

    **A stress the input asked for and this code cannot produce is skipped with
    a warning, not raised.** The regimes P11 does not cover -- noncollinear,
    spin spirals, an external field -- are common in inputs that also carry
    ``tstress = .true.``, and an optional diagnostic must not be able to fail
    the SCF that produced it. Passing ``tstress=True`` explicitly raises
    instead, because then the caller wants the number.
    """
    calculation = calculation or Calculation(
        system, pseudos, diagonalization=diagonalization, k_batch=k_batch
    )
    nbnd = nbnd or system.nbnd or default_nbnd(
        calculation.nelec,
        system.occupations,
        *((calculation.nelup, calculation.neldw) if system.nspin == 2 else (None, None)),
        noncolin=system.noncolin,
    )

    if starting_from is not None:
        if any(x is not None for x in (starting_density, starting_becsum,
                                       starting_ns, starting_wavefunctions)):
            raise ValueError(
                "starting_from already carries the density, becsum, ns and the "
                "wavefunctions; giving one of them separately would start the "
                "run from two states at once"
            )
        state = (
            starting_from if isinstance(starting_from, ContinuedState)
            else continued_state(starting_from, calculation)
        )
        starting_density = state.density
        starting_becsum = state.becsum or None
        starting_ns = state.ns
        starting_wavefunctions = state.wavefunctions
        # ``tau`` crosses only when the two runs agree about what shape it is.
        # A promotion between spin regimes reshapes the density through
        # :mod:`pypresso.scf.continuation` and there is no counterpart for the
        # kinetic energy density, so rather than reshape it here the guess is
        # dropped and the Thomas-Fermi one is used -- which costs iterations and
        # cannot be wrong.
        source_tau = getattr(starting_from, "tau", None)
        if starting_tau is None and source_tau is not None:
            # The density's own shape, whole: ``tau`` has ``nspin_mag`` channels
            # and rebuilding the count from ``nspin`` made every promotion into a
            # spin-orbit run drop its converged ``tau`` without a word.
            expected = tuple(np.shape(calculation.starting_density()))
            if tuple(np.shape(source_tau)) == expected:
                starting_tau = source_tau
        if verbose:
            print(f"  continuing a previous run: {state.description}")

    mixer = get_mixer(mixing_mode, beta=mixing_beta)
    rho = (
        calculation.starting_density() if starting_density is None
        else jnp.asarray(starting_density)
    )
    # ``becsum`` is mixed alongside the density, not derived from it. For an
    # ultrasoft run it could be recomputed from the wavefunctions at any point,
    # but for PAW the one-centre potential is built from it *before* the
    # Hamiltonian exists, so it has to be part of the mixed state -- which is
    # why ``mix_rho.f90`` says it mixes "rho in g-space ... and becsum (for
    # paw)". The starting value is the isolated atoms', matching the starting
    # density.
    becsum_state = (
        calculation.starting_becsum() if starting_becsum is None else starting_becsum
    )
    # ``init_ns`` + ``ns_adj``. Like ``becsum`` it is part of the mixed state
    # rather than a function of the density: the Hubbard potential is built from
    # it before the Hamiltonian exists.
    ns_state = None
    if calculation.is_hubbard:
        if starting_ns is None:
            ns_state = calculation.starting_ns()
        else:
            # Through the precision policy, never a literal dtype (config.py).
            ns_state = system.kpoints.precision.as_real(starting_ns)
            expected = ns_shape(calculation.hubbard, calculation.nspin)
            if ns_state.shape != expected:
                raise ValueError(
                    f"starting_ns has shape {ns_state.shape}, expected {expected} "
                    f"= (nspin, nslot, ldmx, ldmx). There is one slot per "
                    f"*correlated* atom, not per atom, and manifolds of different "
                    f"l are zero-padded to the largest -- build it with "
                    f"pypresso.hubbard.uniform_ns / spin_averaged_ns, or start "
                    f"from Calculation.starting_ns()"
                )
    elif starting_ns is not None:
        raise ValueError(
            "starting_ns was given but this calculation has no Hubbard U; "
            "add a HUBBARD card rather than having the matrix silently ignored"
        )

    if mixing_mode.lower() in PRECONDITIONED:
        # Kerker's ``beta`` is an operator on the grid, so it cannot be built
        # inside ``get_mixer``, which knows only a number. It is installed here,
        # where the density's shape and the dense G-vectors are both in hand.
        mixer.precondition = kerker_preconditioner(
            calculation.basis.dense, calculation.system.cell, tuple(np.shape(rho)),
            beta=mixing_beta, nelec=calculation.nelec,
        )

    previous_energy, history = None, []
    potential_change = None
    converged = False
    wavefunctions = None
    ethr, accuracy = ETHR_INIT, None
    # External fields (P18). ``field`` is a loop variable rather than a property
    # of the calculation because two of its uses change it as the loop runs:
    # Elk's ``reducebf`` multiplies it down towards zero, and its fixed-spin-
    # moment scheme drives it from the moment's error. Both leave the
    # *calculation* untouched, so the same object can be reused afterwards.
    field = calculation.magnetic_field
    field_scale = 1.0

    # A residual solver runs *before* the loop and hands it a density that is
    # already self-consistent, so the loop's first iteration is what turns that
    # density into an ``SCFResult`` -- every energy term, the magnetization, the
    # stress. Nothing about the result has a second implementation, and the one
    # iteration it costs is counted in ``solver.steps`` like any other.
    if calculation.functional.is_meta:
        warnings.warn(
            f"{calculation.functional.name} is a potential and not the "
            "derivative of an energy: the total energy this run reports is the "
            "band term plus the electrostatics plus *correlation only*, and is "
            "not the value of any functional the SCF minimised. It is not "
            "comparable with a total energy from any other functional, and "
            "forces, stress and response are refused for it. The eigenvalues, "
            "the band gap and the density are what this functional is for",
            RuntimeWarning,
            stacklevel=2,
        )
    solver_result = None
    solved_tau = None
    if get_scf_solver(scf_solver) is not None:
        (
            rho, becsum_state, ns_state, solved_tau, wavefunctions, solver_result
        ) = _solve_residual(
            calculation, system, nbnd, rho, becsum_state, ns_state, conv_thr,
            scf_solver, dict(scf_solver_options or {}), mixing_beta, verbose,
            starting_tau,
        )
        # The density is converged, so the eigenvalues must be too: the loose
        # start of the ``ethr`` schedule would otherwise throw the hand-off away
        # on the very first diagonalisation.
        ethr = max(0.1 * solver_result.accuracy / max(1.0, calculation.nelec), ETHR_MIN)

    # ``potinit.f90``'s Thomas-Fermi guess. The first iteration has no states to
    # build ``tau`` from and the meta-GGA potential cannot be evaluated without
    # one; from the second iteration on it comes from the states. It is **not
    # mixed** -- ``mix_rho.f90`` does not touch ``kin_r`` -- so what the
    # potential sees at iteration ``i`` is the previous iteration's output
    # ``tau`` against the *mixed* density, which is QE's pairing and is the
    # reason a meta-GGA SCF converges differently from a GGA one.
    tau_state = None
    if calculation.functional.is_meta:
        # In order of how much is known about ``tau``: a residual solver has
        # already found the joint fixed point in ``(rho, tau)``, so its answer
        # wins; then a converged ``tau`` handed in by the caller (another run at
        # a nearby geometry, ``starting_from``); then ``potinit.f90``'s
        # Thomas-Fermi guess, which costs iterations and cannot be wrong.
        if solver_result is not None and solved_tau is not None:
            tau_state = solved_tau
        elif starting_tau is not None:
            tau_state = jnp.asarray(starting_tau)
        else:
            tau_state = _starting_tau(rho, calculation)

    for iteration in range(1, max_iterations + 1):
        ethr = next_ethr(ethr, accuracy, calculation.nelec, iteration)

        potential = calculation.potential(rho, field_scale, field, tau=tau_state)
        epaw, ddd_paw = calculation.onecenter(becsum_state, _meta_c(potential))
        hubbard_terms = v_ns = None
        eth = 0.0
        if calculation.is_hubbard:
            eth, v_ns, hubbard_terms = calculation.hubbard_terms(ns_state)
        hamiltonians = calculation.hamiltonian(potential.v_scf, ddd_paw, hubbard_terms)

        # QE's threshold for judging the *first* diagonalisation after the fact:
        # if the density turns out to be better than the eigenvalues, the loose
        # starting ethr was a false economy and the iteration is redone.
        floor = ethr * max(1.0, calculation.nelec)
        if wavefunctions is None:
            wavefunctions = calculation.starting_wavefunctions(
                hamiltonians, nbnd, span=starting_wavefunctions
            )

        for attempt in range(2):
            eigenvalues, wavefunctions = calculation.diagonalize(
                hamiltonians, nbnd, wavefunctions, ethr
            )
            wg, levels = calculation.occupations(eigenvalues)
            becsum_out = calculation.becsum(wavefunctions, wg)
            rho_out = calculation.density(wavefunctions, wg, becsum_out)
            if tau_state is not None:
                tau_out = calculation.kinetic_energy_density(wavefunctions, wg)
            # On the dense grid, which is the grid the residual lives on. QE
            # sums rho_ddot over the *smooth* set instead (and says so, in a
            # comment noting the change from ngm to ngms); the difference is the
            # residual's high-G tail, which can only make dr2 larger, so this is
            # the conservative direction. Using the smooth GVectors here would
            # be a silent error whenever they differ: their fft_index addresses
            # a smaller box than the array being gathered from.
            accuracy = float(_accuracy(
                rho_out - rho, calculation.basis.dense, calculation.system.cell
            ))
            if calculation.is_hubbard:
                ns_out = calculation.occupation_matrix(wavefunctions, wg)
                if iteration == 1 and starting_density is None and starting_ns is None:
                    # ``IF (first .AND. starting_pot == 'atomic') CALL ns_adj()``:
                    # skipped when the caller supplied ``starting_ns`` as well:
                    # ``ns_adj`` exists to steer a *fresh* run towards one of
                    # several solutions, and overriding an explicitly given
                    # matrix with ``starting_ns_eigenvalue`` would defeat the
                    # only mechanism that targets a solution reliably.
                    # the requested eigenvalues are imposed on the *measured*
                    # matrix and become both the output and the input of this
                    # step, so the residual of the ns block is zero here and the
                    # second Hamiltonian is the one that is steered.
                    ns_out = calculation.adjust_ns(ns_out)
                    ns_state = ns_out
                if iteration <= mixing_fixed_ns:
                    # ``RESET ns to initial values (iter <= mixing_fixed_ns)``:
                    # the density relaxes around a *frozen* occupation matrix.
                    ns_out = ns_state
                # ``rho_ddot`` gains ``ns_ddot`` when there is a Hubbard term,
                # so that ``dr2`` estimates the error in the whole functional
                # and the ``ethr`` schedule it drives is QE's.
                accuracy += float(calculation.ns_accuracy(ns_out - ns_state))

            if iteration > 1 or attempt > 0 or accuracy >= floor:
                break
            ethr = max(0.1 * accuracy / max(1.0, calculation.nelec), ETHR_MIN)
            if verbose:
                print(f"  iteration {iteration:3d}   ethr was too large; "
                      f"diagonalising again at {ethr:.2e}")

        # Which density each term is evaluated at is QE's convention, and it is
        # not uniform (``electrons.f90``, and the comment there justifying
        # ``descf``):
        #
        #   * ``eband``  -- the eigenvalues, hence the potential of the *input*
        #     density, the one the Hamiltonian was built from;
        #   * ``deband`` -- ``delta_e()``, which runs *before* ``v_of_rho`` is
        #     called again, so it pairs the **output** density with the
        #     **input** potential;
        #   * ``ehart``/``etxc`` -- ``v_of_rho`` on the density that will be
        #     used next, which at convergence is the unmixed **output** one.
        #
        # ``descf`` is QE's first-order correction for that mismatch, and it is
        # identically zero at convergence, which is the only iteration whose
        # terms are compared. Evaluating all of them at the input density
        # instead leaves each one ~1e-5 Ry away from QE's while the total --
        # being variational -- still agrees to 1e-9.
        eband, deband, residual = (
            float(x) for x in
            _iteration_scalars(
                eigenvalues, wg, rho, rho_out, potential.v_scf,
                # ``calculation.system``, never the ``system`` argument. They are
                # the same object for every ordinary call and *not* for one that
                # supplies its own ``calculation`` -- which is exactly what a run
                # on a deformed cell does (:meth:`Calculation.at_strain`). With
                # the caller's volume here, ``deband`` is scaled wrongly and the
                # reported total energy acquires a slope in the strain of
                # **3.9 Ry per unit strain** on two-atom silicon, against a true
                # ``dE/d(eps)`` of 0.09. The density, the potential and every
                # response are unaffected, which is why it survived: only the
                # number printed at the end is wrong.
                calculation.system.cell.volume
            )
        )

        converged = accuracy < conv_thr
        # The density is self-consistent *at this field*, which is the state the
        # secant update is allowed to measure: see ``MagneticField.feedback``.
        inner_converged = converged
        if converged and field is not None and not field.satisfied(
            rho_out, calculation.system.cell
        ):
            # A fixed-spin-moment run is not converged until the moment is where
            # it was asked to be: the constraining field is outside the density,
            # so ``dr2`` can fall below ``conv_thr`` while the field is still
            # being driven and the moment is still moving.
            converged = False
        if converged:
            # QE's ``vnew``: the potential the last step did *not* apply,
            # V[rho_out] - V[rho_in]. It is zero at exact self-consistency and it
            # is what ``force_corr`` pairs with the atomic charges to correct a
            # force for a run that stopped short (``PW/src/force_corr.f90``).
            v_in = potential.v_scf
            potential = calculation.potential(
                rho_out, field_scale, field,
                tau=tau_out if tau_state is not None else None,
            )
            potential_change = potential.v_scf - v_in
            # ... and the one-centre energy with it. ``ddd_paw`` is deliberately
            # *not* refreshed: ``deband`` below pairs it with the output becsum
            # exactly as ``delta_e`` does, which runs before QE recomputes it.
            epaw, _ = calculation.onecenter(becsum_out, _meta_c(potential))
            if calculation.is_hubbard:
                # ``eth`` is recomputed by ``v_of_rho`` on the density that will
                # be used next, which at convergence is the unmixed output one.
                # ``v_ns`` is not refreshed, for the same reason ``ddd_paw`` is
                # not: ``deband`` pairs it with the output occupations.
                eth = hubbard_energy(ns_out, calculation.hubbard_coefficients)

        # PAW's contribution to ``deband``: ``delta_e`` subtracts
        # ``sum ddd_paw * becsum`` for the same reason it subtracts
        # ``int rho v_scf`` -- the one-centre potential is already inside every
        # eigenvalue through ``deeq``, and ``eband`` would double-count it.
        if calculation.is_paw:
            deband -= float(_paw_deband(ddd_paw, calculation.augmentation, becsum_out))
        if calculation.is_hubbard:
            # ``delta_e``'s ``- SUM(rho%ns * v%ns)``, doubled when there is one
            # spin channel. Same pairing as every other term there: the
            # **output** occupation matrix against the **input** potential, since
            # ``delta_e`` runs before ``v_of_rho`` is called again. The Hubbard
            # potential is inside every eigenvalue through ``vhpsi``, so
            # ``eband`` double-counts it and this removes exactly that.
            overlap = float(jnp.sum(ns_out * v_ns))
            deband -= 2.0 * overlap if calculation.nspin == 1 else overlap

        terms = {
            "one-electron": eband + deband,
            "hartree": float(potential.ehart),
            "xc": float(potential.etxc),
            "ewald": float(calculation.ewald),
        }
        if calculation.dispersion_sum is not None:
            # QE's ``Dispersion Correction`` line. It is a constant of the
            # geometry -- nothing in the loop above depends on it -- so it is
            # added to the printed total and to nothing else, exactly as
            # ``electrons.f90`` adds ``elondon``.
            terms["dispersion"] = float(calculation.dispersion)
        if calculation.is_paw:
            terms["one_center_paw"] = float(epaw)
        if calculation.is_hubbard:
            terms["hubbard"] = float(eth)
        if "smearing" in levels:
            terms["smearing"] = levels["smearing"]
        total = sum(terms.values())

        change = None if previous_energy is None else abs(total - previous_energy)
        magnetization = moment = None
        if calculation.nspin == 2:
            magnetization = [
                float(x) for x in _magnetization(rho_out, calculation.system.cell.volume)
            ]
        elif calculation.nspin_mag == 4:
            values = [
                float(x) for x in _noncollinear_magnetization(
                    rho_out, calculation.system.cell.volume
                )
            ]
            moment, magnetization = tuple(values[:3]), [None, values[3]]

        entry = {"iteration": iteration, "total_energy": total,
                 "accuracy": accuracy, "ethr": ethr,
                 "residual": residual, "change": change}
        if tau_state is not None:
            # ``c`` is a cell average of the density, so it moves with the SCF
            # and settling is part of convergence: a run whose density has
            # stopped moving but whose ``c`` has not is not converged, and the
            # only way to see that is to have the number per iteration.
            entry["meta_c"] = float(potential.meta_c)
        if field is not None:
            # Reported, never added: ``etcon`` is printed by ``add_bfield`` and
            # never returned to ``electrons.f90``, and Elk keeps its external
            # field energy out of the total for the same reason. See
            # :mod:`pypresso.scf.fields`.
            entry["field_energy"] = float(potential.e_field)
            entry["constraint_energy"] = float(potential.e_constraint)
        if calculation.is_hubbard:
            entry["hubbard_energy"] = float(eth)
            # ``write_ns``'s headline number, per correlated atom: the trace of
            # the occupation matrix in each channel. It is what says whether the
            # run settled on the intended orbital occupation, and it moves long
            # after the total energy has stopped moving.
            entry["hubbard_traces"] = np.asarray(
                jnp.einsum("snmm->sn", ns_out)
            ).T.tolist()
        if magnetization is not None:
            entry["magnetization"] = magnetization[0]
            entry["absolute_magnetization"] = magnetization[1]
        if moment is not None:
            entry["magnetization_vector"] = moment
        history.append(entry)
        if verbose:
            if moment is not None:
                extra = (
                    f"   m = ({moment[0]:6.3f}, {moment[1]:6.3f}, {moment[2]:6.3f})"
                    f" [{magnetization[1]:6.3f}] mu_B"
                )
            elif magnetization is None:
                extra = ""
            else:
                extra = f"   m = {magnetization[0]:7.4f} ({magnetization[1]:7.4f}) mu_B"
            print(f"  iteration {iteration:3d}   E = {total:16.8f} Ry"
                  f"   accuracy = {accuracy:.2e}   ethr = {ethr:.2e}"
                  f"   |drho| = {residual:.2e}{extra}")

        if converged:
            rho = rho_out
            becsum_state = becsum_out
            if tau_state is not None:
                tau_state = tau_out
            if calculation.is_hubbard:
                ns_state = ns_out
            break

        previous_energy = total
        if tau_state is not None:
            # Replaced, not mixed. See the Thomas-Fermi comment above.
            tau_state = tau_out
        rho, becsum_state, ns_state = _mix(
            mixer, rho, rho_out, becsum_state, becsum_out,
            ns_state if calculation.is_hubbard else None,
            ns_out if calculation.is_hubbard else None,
        )
        if field is not None:
            # ``reducebf`` (Elk 5.104), and the fixed-spin-moment feedback, both
            # act between iterations -- after the density is mixed and before
            # the next potential is built.
            field_scale *= field.reducebf
            if field.fsm_update == "elk" or field.constraint != "fsm":
                field = field.feedback(rho, calculation.system.cell)
            elif inner_converged:
                # The secant scheme steps on *converged* pairs only. Between
                # steps the field is held and the SCF is an ordinary one, which
                # is the whole of why it costs a handful of solves rather than a
                # thousand interleaved iterations -- ``m(B)`` is smooth where
                # ``m`` at iteration ``i`` is not. The mixer keeps its history
                # across the step: the density it holds is a better start for
                # the next field than the atomic guess, and the field moves by
                # less each time.
                field = field.feedback(rho_out, calculation.system.cell)

    nspin = calculation.nspin
    stress = None
    asked_by_hand = tstress is not None
    if system.tstress if tstress is None else tstress:
        # Imported here rather than at the top of the module: ``pypresso.stress``
        # sits *above* ``scf`` in the layering (rule R3) because it
        # differentiates through a whole calculation, so a module-level import
        # would point the wrong way. The deferral is the price of letting one
        # switch in ``&control`` reach the result object the way ``pw.x`` does.
        from pypresso.stress import compute_stress

        state = _result_for_stress(
            calculation, eigenvalues, wg, wavefunctions, rho, terms
        )
        try:
            stress = compute_stress(calculation, state)
        except NotImplementedError as unsupported:
            # **An optional diagnostic must not be able to fail the run that
            # produced it.** ``tstress = .true.`` is a common thing to leave in
            # an input -- three of QE's own spin-orbit benchmarks carry it --
            # and a noncollinear SCF that converged perfectly must not end in an
            # exception because the *stress* for that regime is not written.
            #
            # This is QE's convention, not a softening of the house rule:
            # ``input.f90`` computes ``tstress_ = lmovecell .OR. (tstress .AND.
            # lscf)`` and then switches it off again for combinations it cannot
            # do (``tefield``, ``gate``, ``do_comp_mt``), and ``stress()``
            # itself opens with an ``infomsg`` and a bare ``RETURN`` for
            # electric fields with ultrasoft. Asked for by *hand* -- an explicit
            # ``tstress=True``, or a direct ``compute_stress`` -- it still
            # raises, because then somebody wants the number rather than having
            # left a flag in a file.
            if asked_by_hand:
                raise
            warnings.warn(
                f"tstress = .true. in the input, but {unsupported}. The SCF is "
                "unaffected and SCFResult.stress is None.",
                RuntimeWarning,
                stacklevel=2,
            )

    return SCFResult(
        converged=converged,
        iterations=iteration,
        total_energy=total,
        energy_terms=terms,
        # The spin axis is dropped when there is only one channel: an
        # unpolarized result then has exactly the shape it always had.
        eigenvalues=np.asarray(eigenvalues if nspin == 2 else eigenvalues[0]),
        occupations=np.asarray(wg if nspin == 2 else wg[0]),
        wavefunctions=wavefunctions,
        density=rho,
        # ``set_vrs`` again: the local pseudopotential belongs to the charge
        # component alone once the potential is ``(n, m)``, so it cannot be
        # broadcast onto the magnetic components.
        potential=as_potential_components(calculation.vltot, calculation.nspin_mag)
        + potential.v_scf,
        potential_change=potential_change,
        accuracy=accuracy,
        nspin=nspin,
        nspin_mag=calculation.nspin_mag,
        ns=ns_state,
        hubbard_setup=calculation.hubbard,
        becsum=tuple(becsum_state),
        system=calculation.system,
        stress=stress,
        magnetization=None if magnetization is None else magnetization[0],
        absolute_magnetization=None if magnetization is None else magnetization[1],
        magnetization_vector=moment,
        field_energy=None if field is None else float(potential.e_field),
        constraint_energy=None if field is None else float(potential.e_constraint),
        magnetic_field=field,
        field_scale=float(field_scale),
        fermi_energy=levels.get("fermi_energy"),
        fermi_energy_up=levels.get("fermi_energy_up"),
        fermi_energy_down=levels.get("fermi_energy_down"),
        homo=levels.get("homo"),
        lumo=levels.get("lumo"),
        history=history,
        solver=solver_result,
        tau=tau_state,
        meta_c=None if tau_state is None else float(potential.meta_c),
    )
