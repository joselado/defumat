"""The Kubo Berry curvature on a plane-wave calculation (P47).

``pypresso.topology.berry`` has had a ``kubo`` method since P16 and it was
reachable only from a tight-binding model, because a plane-wave ``H(k)`` is not
a dense matrix ``jacfwd`` can differentiate and ``eigh`` can diagonalise. P24's
:class:`~pypresso.response.velocity.VelocityOperator` is what removes the
obstruction -- one ``jax.jvp`` of ``H(k)`` at a frozen sphere -- and
:mod:`pypresso.topology.kubo` contracts it into band matrix elements, so
nothing of size ``npw^2`` is ever formed.

The file is laid out as the three independent things that can be wrong.

**The operator.** ``<psi_m|dH/dk_a|psi_n>`` against a *central finite
difference* of the same matrix element at a frozen sphere and frozen states.
This is a QE-free anchor and it isolates the ``jvp`` and the crystal-direction
tangent (the reciprocal lattice vector ``bg[d]``) from everything downstream.

**The assembly.** ``Omega_kubo`` against the Fukui-Hatsugai-Suzuki lattice flux,
which shares no machinery with it -- FHS is determinants of overlaps between
neighbouring k-points and never differentiates anything. Two forms of the
comparison, and both are needed: a *shrinking centred plaquette* around one
k-point, which converges onto the pointwise value; and the whole mesh
plaquette by plaquette, which is what a curvature map is actually read off and
whose agreement must improve as the mesh is refined.

**The symmetry.** On a centrosymmetric crystal ``Omega(k)`` must vanish
*pointwise*, not merely integrate to zero -- silicon. And inside a degenerate
multiplet only the sum over the members is defined, which is checked by
rotating the multiplet and watching the members move while the sum does not.

The crystal for the non-trivial half is **AlAs**, because it is zincblende and
therefore has time reversal but no inversion centre: ``Omega(k)`` is nonzero
pointwise while the Chern number is zero. Silicon, which has both, is the
control -- and a curvature that is zero for the wrong reason is the trap that
control exists for.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from pypresso.io.pwin import read_pw_input
from pypresso.pseudo.upf import read_upf
from pypresso.scf.driver import run_scf
from pypresso.system.builder import build_system
from pypresso.topology.berry import berry_curvature
from pypresso.topology.kubo import kubo_from_matrices, velocity_matrices
from pypresso.topology.links import berry_phase, link_phase
from pypresso.topology.mesh import plane_mesh
from pypresso.workflows.topology import DFTSource, run_berry_curvature

pytestmark = pytest.mark.regression

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: A generic point of the ``k_z = 0`` crystal plane of AlAs: no symmetry of the
#: crystal maps it to itself, so nothing forces the curvature there to anything.
GENERIC_K = np.array([0.1875, 0.3125, 0.0])
#: A point where the second and third valence bands are exactly degenerate
#: (measured: 9.4e-16 Ry apart), which is what the gauge test needs.
DEGENERATE_K = np.array([0.0, 0.375, 0.0])


@lru_cache(maxsize=4)
def converged(name: str, conv_thr: float = 1.0e-10):
    """An SCF run of a committed input, cached for the whole module."""
    system = build_system(read_pw_input(CASES / name))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    return system, pseudos, run_scf(system, pseudos, conv_thr=conv_thr)


def source(name: str, nocc: int, nbnd: int | None = None,
           conv_thr: float = 1.0e-12) -> DFTSource:
    system, pseudos, result = converged(name)
    return DFTSource(
        system=system, pseudos=pseudos, density=result.density, nocc=nocc,
        nbnd=nbnd, conv_thr=conv_thr,
    )


def blocks(states):
    """``(dh1, ds1, dh2, ds2)`` along the two crystal directions of the plane."""
    bg = np.asarray(states.bg)
    dh1, ds1 = velocity_matrices(states, bg[0])
    dh2, ds2 = velocity_matrices(states, bg[1])
    return dh1, ds1, dh2, ds2


def kubo_at(name, points, nocc=4, nbnd=30):
    """``Omega(k)`` and its per-band decomposition at an explicit k-list."""
    states = source(name, nocc, nbnd).states(
        np.asarray(points, dtype=float).reshape(-1, 3), keep_velocity=True
    )
    dh1, ds1, dh2, ds2 = blocks(states)
    energies = np.asarray(states.energies)
    total, by_band = kubo_from_matrices(dh1, ds1, dh2, ds2, energies, nocc)
    return np.asarray(total), np.asarray(by_band), states


# --- the operator -----------------------------------------------------------

def test_the_velocity_matrix_is_the_finite_difference_of_h():
    """``<psi_m|dH/dk_a|psi_n>`` against a central difference of ``H(k)``.

    The QE-free anchor, and the one that isolates the new code's *operator*
    from its assembly. Both sides hold the plane-wave sphere and the states
    fixed and move only ``k``, which is exactly what the ``jvp`` differentiates
    -- sphere membership is piecewise constant in ``k``, so on each piece the
    frozen-sphere derivative is the exact one and a finite difference of the
    same frozen object must reproduce it to the step's own order.

    It also pins the **tangent**: the mesh spans crystal directions, so the
    cartesian vector handed to the velocity operator is the reciprocal lattice
    vector ``bg[d]``, and a wrong one here would be a curvature in the wrong
    units that no downstream check distinguishes from a wrong assembly.

    Measured: 1.8e-7 at ``eps = 1e-3`` and 1.8e-9 at ``1e-4`` on a matrix whose
    largest element is 1.05 -- the exact factor of 100 a second-order difference
    owes a ten-fold smaller step.
    """
    import jax.numpy as jnp

    from pypresso.response.velocity import over_kpoints

    points = np.array([[0.13, 0.21, 0.07], [0.30, -0.10, 0.25]])
    states = source("alas-raman.in", 4, nbnd=10).states(points, keep_velocity=True)
    bg = np.asarray(states.bg)
    analytic = np.asarray(velocity_matrices(states, bg[0])[0])

    velocity = states.velocity
    psi = jnp.asarray(states.all_coefficients)
    kcart = np.asarray(velocity.kcart)

    def block(moved_k):
        moved = velocity.calculation.at_kcart(jnp.asarray(moved_k))
        hamiltonians = moved.hamiltonian(velocity.v_scf, velocity.ddd_paw, None)
        applied = over_kpoints(
            hamiltonians[0], psi, velocity.calculation.k_batch, False
        )
        return np.asarray(jnp.einsum("kmg,kng->kmn", psi.conj(), applied))

    eps = 1.0e-4
    difference = (block(kcart + eps * bg[0]) - block(kcart - eps * bg[0])) / (2 * eps)
    assert np.max(np.abs(analytic)) > 0.5  # the case would be vacuous otherwise
    assert np.max(np.abs(difference - analytic)) < 1.0e-7


def test_the_velocity_matrix_is_hermitian_and_the_overlap_does_not_move():
    """``dH/dk`` is Hermitian; ``dS/dk`` is **identically** zero here.

    Two statements the assembly quietly assumes. The first is a property of the
    operator that a wrong contraction (a missing conjugate, a transposed
    einsum) would break; measured at 3.8e-15.

    The second is why an ultrasoft or PAW dataset is refused rather than
    trusted. ``S`` is the identity for a norm-conserving pseudopotential and has
    no ``k`` in it at all, so the ``- e_n dS/dk_a`` term of the generalised
    velocity is *exactly* zero -- not small, zero -- and no norm-conserving
    validation anywhere in this file can see whether its convention is right.
    """
    states = source("alas-raman.in", 4, nbnd=10).states(
        GENERIC_K.reshape(1, 3), keep_velocity=True
    )
    dh1, ds1, _, _ = blocks(states)
    dh1 = np.asarray(dh1)
    assert np.max(np.abs(dh1 - dh1.conj().swapaxes(-1, -2))) < 1.0e-10
    assert np.max(np.abs(np.asarray(ds1))) == 0.0


# --- the assembly against Fukui-Hatsugai-Suzuki -----------------------------

def _fhs_plaquette(states_source, k0, h):
    """``Omega`` from one FHS plaquette of side ``h`` **centred** on ``k0``.

    Centred rather than anchored: a plaquette whose corner is ``k0`` samples the
    curvature half a step away and its error is first order in ``h``, which
    would hide a second-order convergence under a first-order offset.
    """
    corner = np.asarray(k0, dtype=float) - np.array([h / 2, h / 2, 0.0])
    points = np.array([
        corner,
        corner + [h, 0, 0],
        corner + [h, h, 0],
        corner + [0, h, 0],
    ])
    states = states_source.states(points)
    link = lambda a, b: complex(link_phase(states.overlap(a, b, None)))
    loop = link(0, 1) * link(1, 2) * link(2, 3) * link(3, 0)
    return float(berry_phase(np.array(loop))) / h ** 2


def test_a_shrinking_fhs_plaquette_converges_onto_the_kubo_value():
    """The decisive check, in its pointwise form.

    ``Omega_kubo(k)`` is a point sample; an FHS plaquette flux is an *integral*
    over the plaquette, so the two agree only in the limit of a small plaquette
    -- and they share no machinery, the first being one ``jvp`` of the
    Hamiltonian contracted against the states and the second a product of four
    determinants of overlaps between separate diagonalisations.

    Measured at ``k = (0.1875, 0.3125, 0)``, against ``Omega_kubo = 0.96471``
    at ``nbnd = 30``::

        h = 0.08   0.93502   3.1e-2
        h = 0.04   0.96056   4.3e-3
        h = 0.02   0.96784   3.2e-3
        h = 0.01   0.97956   1.5e-2

    -- second-order convergence down to ``h = 0.02`` and then a **noise floor**,
    because the flux itself is 1e-4 rad at that size and dividing an eigensolver's
    round-off by ``h^2`` amplifies it. So the agreement this establishes is
    3e-3 relative and the mesh cannot be refined past it; the number is reported
    rather than chased.
    """
    total, _, _ = kubo_at("alas-raman.in", GENERIC_K, nocc=4, nbnd=30)
    reference = float(total[0])
    assert abs(reference) > 0.5  # a vanishing curvature would make this vacuous

    states_source = source("alas-raman.in", 4)
    coarse = _fhs_plaquette(states_source, GENERIC_K, 0.08)
    fine = _fhs_plaquette(states_source, GENERIC_K, 0.04)

    assert abs(coarse - reference) / abs(reference) < 5.0e-2
    assert abs(fine - reference) / abs(reference) < 1.0e-2
    # and the refinement has to be an improvement, not a coincidence
    assert abs(fine - reference) < 0.3 * abs(coarse - reference)


@pytest.mark.slow
def test_the_whole_mesh_agrees_plaquette_by_plaquette_and_improves_with_it():
    """The same check in the form a curvature map is actually read in.

    One fine Kubo mesh (24x24, at the plaquette centres) is integrated over the
    plaquettes of a coarser FHS mesh by the midpoint rule, and the two arrays
    are compared entry by entry. Measured, ``nbnd = 30``::

        n     max|flux|    max|integral - flux|   relative
        4     3.97e-2      9.49e-3                0.239
        6     2.18e-2      3.59e-3                0.164
        8     1.39e-2      1.63e-3                0.118
        12    8.79e-3      6.56e-4                0.075
        24    2.29e-3      1.45e-4                0.063

    The absolute agreement improves by 65x over a six-fold refinement -- faster
    than ``h^2``, which is what both sides' errors are -- while the relative
    figure improves fourfold and then flattens, because the flux it is measured
    against is itself shrinking like ``h^2`` and the same round-off floor as the
    pointwise test is being approached from the other side.
    """
    fine_n, nbnd = 24, 30
    fine = plane_mesh((fine_n, fine_n), axis=2, offset=0.0,
                      origin=(0.5 / fine_n, 0.5 / fine_n, 0.0))
    states = source("alas-raman.in", 4, nbnd).states(fine.flat(), keep_velocity=True)
    dh1, ds1, dh2, ds2 = blocks(states)
    total, _ = kubo_from_matrices(
        dh1, ds1, dh2, ds2, np.asarray(states.energies), 4
    )
    omega = np.asarray(total).reshape(fine_n, fine_n)

    coarse_source = source("alas-raman.in", 4)
    errors = {}
    for n in (4, 12):
        sub = fine_n // n
        mesh = plane_mesh((n, n), axis=2, offset=0.0)
        flux = np.asarray(
            berry_curvature(coarse_source.states(mesh.flat()), mesh, method="fhs").flux
        )
        integrated = omega.reshape(n, sub, n, sub).mean(axis=(1, 3)) / (n * n)
        errors[n] = (
            float(np.max(np.abs(integrated - flux))),
            float(np.max(np.abs(flux))),
        )

    assert errors[4][0] / errors[4][1] < 0.30
    assert errors[12][0] / errors[12][1] < 0.10
    assert errors[12][0] / errors[12][1] < errors[4][0] / errors[4][1]
    assert errors[12][0] < 0.1 * errors[4][0]


# --- what symmetry forces ---------------------------------------------------

def test_silicon_kubo_curvature_vanishes_pointwise():
    """Time reversal *and* inversion make ``Omega(k) = 0`` at every k.

    The control for the AlAs numbers above, and it is the sharp statement: not a
    Chern number that integrates to zero but a curvature that is zero point by
    point. The scale it is zero *against* is the per-band curvature, which is
    5.7 on the same mesh -- so this is five orders, not a small number next to
    nothing.

    Measured: ``max |Omega| = 3.5e-5`` on a 4x4 mesh with ``nbnd = 12``.
    """
    system, pseudos, result = converged("si2-nc-pbe.in")
    curvature = run_berry_curvature(
        system, pseudos, result.density, shape=(4, 4), nocc=4, nbnd=12,
        method="kubo",
    )
    assert curvature.method == "kubo"
    assert np.max(np.abs(curvature.curvature)) < 1.0e-3
    assert abs(curvature.chern_number) < 1.0e-5
    # The per-band curvature is emphatically *not* zero, which is what makes the
    # line above a cancellation rather than an empty calculation.
    assert np.max(np.abs(curvature.curvature_by_band)) > 1.0


def test_a_degenerate_multiplet_is_gauge_invariant_only_as_a_sum():
    """Rotate a degenerate pair: the members move, their sum does not.

    At ``k = (0, 0.375, 0)`` the second and third valence bands of AlAs are
    degenerate to 9.4e-16 Ry, so any unitary mixing of the two is as valid an
    eigenbasis as the one the eigensolver returned. The Kubo weight
    ``1/(e_n - e_m)^2`` is then *constant* across the block, which makes the
    manifold total a trace over it and therefore invariant -- while the
    individual ``Omega_n`` are not, and are not properties of a band at all.

    This is P36's degenerate-multiplet finding one quantity over, and the
    reason :attr:`~pypresso.topology.berry.BerryCurvature.curvature_by_band`
    carries the warning it does. Measured: the pair goes from
    ``(+0.203989, -0.218475)`` to ``(+0.273508, -0.287993)`` -- moving by
    **0.0695** -- while the manifold total stays at ``-0.0144856`` to
    **1.1e-15** and the multiplet's own sum to 1.0e-15.
    """
    total, by_band, states = kubo_at(
        "alas-raman.in", DEGENERATE_K, nocc=4, nbnd=16
    )
    energies = np.asarray(states.energies)
    assert abs(energies[0, 2] - energies[0, 3]) < 1.0e-9

    dh1, ds1, dh2, ds2 = blocks(states)
    nband = energies.shape[1]
    rotation = np.eye(nband, dtype=complex)
    angle, phase = 0.7, 0.4
    rotation[2:4, 2:4] = np.array([
        [np.cos(angle), -np.sin(angle) * np.exp(1j * phase)],
        [np.sin(angle) * np.exp(-1j * phase), np.cos(angle)],
    ])

    def rotate(matrix):
        return rotation.conj().T @ np.asarray(matrix)[0] @ rotation

    rotated = [rotate(m)[None] for m in (dh1, ds1, dh2, ds2)]
    total_r, by_band_r = kubo_from_matrices(*rotated, energies, 4)

    moved = np.max(np.abs(np.asarray(by_band_r)[0, 2:4] - by_band[0, 2:4]))
    assert moved > 0.05, "the rotation did not actually move the multiplet"
    assert abs(float(total_r[0]) - float(total[0])) < 1.0e-10
    assert abs(
        float(np.sum(np.asarray(by_band_r)[0, 2:4]))
        - float(np.sum(by_band[0, 2:4]))
    ) < 1.0e-10


# --- what is reported, and what is refused ----------------------------------

def test_the_truncation_is_reported_and_the_sum_moves_with_nbnd():
    """The sum over ``m`` stops where the eigensolver stopped, and says so.

    The Sternheimer stack (P24) exists to avoid a sum over empty states; this
    one does not avoid it, so the truncation is reported the way P37 reports
    ``static_residual`` -- a number the caller reads, never a knob tuned until
    a test passes. Measured at ``k = (0.1875, 0.3125, 0)``, one
    diagonalisation at ``nbnd = 45`` truncated after the fact::

        nbnd = 12   0.978886
        nbnd = 20   0.965663
        nbnd = 30   0.964705
        nbnd = 45   0.961639

    -- 1.8% between 12 and 45, and 0.3% between 30 and 45. What that costs on
    the *plaquette* comparison is the whole reason it is worth a number: at
    ``nbnd = 8`` the same mesh sum is 1.7 relative against the FHS flux, at 12
    it is 0.20, and from 25 upwards it settles at 0.26 -- the residue there
    being FHS's own coarse-plaquette error, not the truncation.
    """
    states = source("alas-raman.in", 4, nbnd=45).states(
        GENERIC_K.reshape(1, 3), keep_velocity=True
    )
    dh1, ds1, dh2, ds2 = blocks(states)
    energies = np.asarray(states.energies)
    values = {}
    for nbnd in (12, 20, 30, 45):
        total, _ = kubo_from_matrices(
            dh1, ds1, dh2, ds2, energies, 4, nbnd=nbnd
        )
        values[nbnd] = float(total[0])

    assert abs(values[12] - values[45]) > 5.0e-3     # truncation is real
    assert abs(values[30] - values[45]) < 5.0e-3     # and it is converging
    assert abs(values[30] - values[45]) < abs(values[12] - values[45])

    mesh = plane_mesh((2, 2), axis=2, offset=0.0, origin=(0.25, 0.25, 0.0))
    reported = berry_curvature(
        source("alas-raman.in", 4, nbnd=20).states(mesh.flat(), keep_velocity=True),
        mesh, method="kubo",
    )
    assert reported.nbnd == 20 and reported.nocc == 4
    assert reported.truncation is not None and reported.truncation > 0.0
    assert reported.truncation_abs is not None


def test_an_ultrasoft_kubo_curvature_is_refused_by_name():
    """``dS/dk`` is zero for every case validated here, so US/PAW is refused.

    The refusal is the honest state of it rather than a gap: the term *is*
    written (``VelocityOperator.apply_s``, the second tangent of the same
    ``jvp``), and it is identically zero on a norm-conserving dataset, so
    nothing in this file measures whether its convention -- ``e_n`` in both
    factors, the outer band's energy and not the inner one's -- is right. An
    off-diagonal element with a moving ``S`` is exactly the thing that comes out
    plausible when it is wrong.
    """
    system, pseudos, result = converged("si2-us.in")
    with pytest.raises(NotImplementedError, match="dS/dk"):
        run_berry_curvature(
            system, pseudos, result.density, shape=(2, 2), nocc=4, nbnd=8,
            method="kubo",
        )


@pytest.mark.slow
def test_a_paw_kubo_curvature_is_refused_by_the_same_name():
    """And with ``becsum`` supplied, so the refusal reached is this one.

    Without it a PAW source stops earlier, on ``DFTSource``'s own missing-
    ``becsum`` refusal, and the test would pass while establishing nothing
    about the curvature.
    """
    system, pseudos, result = converged("si2-paw.in")
    with pytest.raises(NotImplementedError, match="dS/dk"):
        run_berry_curvature(
            system, pseudos, result.density, shape=(2, 2), nocc=4, nbnd=8,
            method="kubo", becsum=result.becsum,
        )


def test_a_state_set_without_the_velocity_operator_is_refused():
    """The occupied manifold alone is enough for ``fhs`` and not for ``kubo``.

    A state set built the ordinary way carries the occupied bands and no
    velocity operator, and a sum over *empty* states cannot be taken from it.
    Refused by name rather than by an attribute error three frames down.
    """
    mesh = plane_mesh((2, 2), axis=2, offset=0.0)
    states = source("alas-raman.in", 4).states(mesh.flat())
    with pytest.raises(ValueError, match="keep_velocity"):
        berry_curvature(states, mesh, method="kubo")


def test_selecting_a_subset_drops_the_velocity_operator():
    """Because it is built on the *whole* k-list and no longer lines up.

    ``select`` is what a streaming Wilson loop uses; a velocity operator sliced
    along with the coefficients would be differentiating at the wrong k-points,
    which is the failure that would be silent. It is dropped, and the next
    ``kubo`` call refuses.
    """
    mesh = plane_mesh((2, 2), axis=2, offset=0.0)
    states = source("alas-raman.in", 4, nbnd=8).states(
        mesh.flat(), keep_velocity=True
    )
    assert states.velocity is not None
    subset = states.select([0, 1])
    assert subset.velocity is None and subset.all_coefficients is None


def test_fhs_is_still_the_default_and_is_untouched():
    """The registry default has not moved, and neither has the ``fhs`` answer.

    ``kubo`` divides by ``(e_n - e_m)^2``, which rule D4 forbids, and its
    Brillouin-zone sum is a Riemann sum that converges to an integer without
    ever being one. It is for the *map*; the invariant stays with the lattice
    construction.
    """
    from pypresso.topology.registry import DEFAULT_CURVATURE_METHOD, curvature_methods

    assert DEFAULT_CURVATURE_METHOD == "fhs"
    assert set(curvature_methods()) >= {"fhs", "kubo"}

    system, pseudos, result = converged("si2-nc-pbe.in")
    fhs = run_berry_curvature(system, pseudos, result.density, shape=(4, 4), nocc=4)
    assert fhs.method == "fhs"
    assert fhs.chern_number == pytest.approx(0.0, abs=1e-9)
    assert fhs.max_flux < 1e-5
    assert fhs.curvature_by_band is None and fhs.truncation is None
