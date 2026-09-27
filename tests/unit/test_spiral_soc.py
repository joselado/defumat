"""Spin-orbit coupling on a spin spiral, to first order (P123), without an SCF.

What can be held without converging anything is held here: that the reader and
the calculation admit a spiral on a fully-relativistic dataset at ``soc_scale =
0`` and nowhere else, that the zeroth order they admit is spin-rotation
invariant (which is the whole of the generalized Bloch theorem's premise), and
that :func:`~defumat.workflows.spiral_soc.spiral_expectation` is the diagonal
spin blocks of the coupled operator minus the free one, projected on each
component's own sphere, for every orientation of the spiral's axis. The
converged numbers, the chirality and the supercell reference are in
``tests/regression/test_spiral_soc.py``.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from defumat.io.pwin import parse_pw_input
from defumat.pseudo import read_upf
from defumat.scf.driver import Calculation
from defumat.system import build_system
from defumat.workflows.anisotropy import _first_order_operator
from defumat.workflows.spiral_soc import (spiral_expectation,
                                          spiral_spin_orbit_energy)

pytestmark = pytest.mark.unit

#: One iodine atom in a small box, turned along ``z`` by a spiral: the cheapest
#: magnetic cell with a fully-relativistic norm-conserving dataset. Nothing here
#: is converged or needs to be.
INPUT = """
 &control
    calculation = 'scf'
 /
 &system
    ibrav = 1, celldm(1) = 8.0, nat = 1, ntyp = 1, ecutwfc = 15.0, nbnd = 8,
    occupations = 'smearing', smearing = 'gaussian', degauss = 0.05,
    noncolin = .true., lspinorb = .true., soc_scale = 0.0, nosym = .true.,
    starting_magnetization(1) = 0.5, angle1(1) = 90.0,
    spiral_q(1) = 0.0, spiral_q(2) = 0.0, spiral_q(3) = 0.25
 /
ATOMIC_SPECIES
 I 126.9 I.rel-pbe-nc-dojo.UPF
ATOMIC_POSITIONS crystal
 I 0.00 0.00 0.00
K_POINTS (automatic)
 1 1 2 0 0 0
"""


def _system(text=INPUT):
    return build_system(parse_pw_input(text))


def _pseudos(system, pseudo_dir: Path):
    return tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)


def test_the_reader_admits_the_spiral_at_soc_scale_0_and_refuses_it_at_1():
    assert _system().spiral and _system().lspinorb
    assert _system().soc_scale == 0.0
    with pytest.raises(ValueError, match="generalized Bloch theorem"):
        _system(INPUT.replace("soc_scale = 0.0,", ""))


def test_the_calculation_refuses_the_coupling_switched_on_after_the_reader(pseudo_dir):
    """``with_soc_scale`` is a field replacement and never meets the reader."""
    system = _system()
    with pytest.raises(NotImplementedError, match="generalized Bloch"):
        Calculation(system.with_soc_scale(1.0), _pseudos(system, pseudo_dir))


def test_the_zeroth_order_is_spin_rotation_invariant(pseudo_dir):
    """``dvan_so`` at ``soc_scale = 0`` is the identity in spin.

    That is what makes the spiral a calculation at all: a nonlocal term with a
    spin structure would not commute with the spin rotation the generalized
    Bloch theorem pairs with each translation.
    """
    system = _system()
    calculation = Calculation(system, _pseudos(system, pseudo_dir))
    d = np.asarray(calculation.dvan_so)
    assert np.max(np.abs(d[0, 1])) == 0.0 and np.max(np.abs(d[1, 0])) == 0.0
    np.testing.assert_array_equal(d[0, 0], d[1, 1])
    assert np.max(np.abs(d[0, 0])) > 0.1


def _states(calculation, seed=3):
    rng = np.random.default_rng(seed)
    nk = calculation.projectors.vkb.shape[0] // 2
    ndim = 2 * calculation.projectors.vkb.shape[1]
    psi = rng.normal(size=(1, nk, 5, ndim)) + 1j * rng.normal(size=(1, nk, 5, ndim))
    weights = rng.random((1, nk, 5))
    return psi, weights


def _su2_taking_z_to(axis):
    """``U`` with ``U sigma_z U^dagger = sigma . axis``, by the half-angle formula."""
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    theta = np.arccos(np.clip(axis[2], -1.0, 1.0))
    phi = np.arctan2(axis[1], axis[0])
    return np.array([
        [np.cos(theta / 2), -np.exp(-1j * phi) * np.sin(theta / 2)],
        [np.exp(1j * phi) * np.sin(theta / 2), np.cos(theta / 2)],
    ])


def test_the_first_order_vector_is_the_diagonal_blocks_on_each_components_sphere(pseudo_dir):
    """P120's template, one layout over: the operator, then its contraction.

    The operator is held entry by entry against two *non*-spiral calculations
    of the same cell, the coupled one and the free one, so it is the
    difference of Hamiltonians this code builds rather than a list of terms.
    The contraction is then rebuilt for four axes by the independent route --
    turn the operator's spin indices by the SU(2) matrix taking ``z`` to the
    axis, keep its two diagonal blocks, and project the up component on
    ``vkb(k + q/2)`` and the down on ``vkb(k - q/2)`` -- and it must be ``n . V``
    each time, which is the claim that the whole axis dependence is linear.
    """
    system = _system()
    pseudos = _pseudos(system, pseudo_dir)
    calculation = Calculation(system, pseudos)
    plain = dataclasses.replace(system, spiral_q=None)
    coupled = Calculation(plain.with_soc_scale(1.0), pseudos)
    free = Calculation(plain, pseudos)
    delta, qq = _first_order_operator(calculation, None)
    assert qq is None
    np.testing.assert_allclose(
        np.asarray(delta), np.asarray(coupled.dvan_so) - np.asarray(free.dvan_so),
        rtol=0.0, atol=1e-14,
    )

    psi, weights = _states(calculation)
    vector, by_k, dropped, trace = spiral_expectation(calculation, psi, weights, delta)
    assert trace < 1e-14
    np.testing.assert_allclose(np.sum(by_k, axis=0), vector, rtol=1e-13)

    vkb = np.asarray(calculation.projectors.vkb)
    nk, npwx = vkb.shape[0] // 2, vkb.shape[1]
    components = psi.reshape(psi.shape[:-1] + (2, npwx))
    up = np.einsum("kgi,skng->skni", vkb[:nk].conj(), components[..., 0, :])
    down = np.einsum("kgi,skng->skni", vkb[nk:].conj(), components[..., 1, :])
    d = np.asarray(delta)
    for axis in ((0, 0, 1), (1, 0, 0), (0, 1, 0), (0.3, -0.5, 0.8)):
        u = _su2_taking_z_to(axis)
        turned = np.einsum("sa,stij,tb->abij", u.conj(), d, u)
        direct = np.sum(weights * np.real(
            np.einsum("skni,ij,sknj->skn", up.conj(), turned[0, 0], up)
            + np.einsum("skni,ij,sknj->skn", down.conj(), turned[1, 1], down)
        ))
        axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
        assert axis @ vector == pytest.approx(direct, rel=1e-11, abs=1e-14)
    assert abs(vector[2]) > 1e-6


def test_an_intermediate_soc_scale_on_a_norm_conserving_dataset_is_the_blend(pseudo_dir):
    """``soc_scale = s`` is ``H(0) + s dD`` exactly, entry by entry.

    What admitting it rests on: on a norm-conserving dataset the scale reaches
    ``dvan_so`` alone, and linearly, so a calculation at ``s`` is the
    first-order operator's own Hamiltonian at coupling ``s``. The same scale on
    an ultrasoft dataset is refused where the calculation meets the dataset.
    """
    system = dataclasses.replace(_system(), spiral_q=None)
    pseudos = _pseudos(system, pseudo_dir)
    free = np.asarray(Calculation(system, pseudos).dvan_so)
    coupled = np.asarray(Calculation(system.with_soc_scale(1.0), pseudos).dvan_so)
    blended = np.asarray(Calculation(system.with_soc_scale(0.3), pseudos).dvan_so)
    np.testing.assert_allclose(blended, free + 0.3 * (coupled - free), rtol=0.0, atol=1e-14)
    assert np.max(np.abs(blended[0, 1])) > 1e-3

    augmented = _system(INPUT.replace("I.rel-pbe-nc-dojo.UPF", "I.rel-pbe-n-rrkjus_psl.1.0.0.UPF"))
    augmented = dataclasses.replace(augmented, spiral_q=None).with_soc_scale(0.3)
    with pytest.raises(ValueError, match="ultrasoft or PAW"):
        Calculation(augmented, _pseudos(augmented, pseudo_dir))
    with pytest.raises(ValueError, match="not a blend"):
        system.with_soc_scale(-0.1)


def test_the_transverse_blocks_are_what_the_mask_removes(pseudo_dir):
    """The cross terms are not small in this layout, which is why they are masked.

    Contracted across the two spheres, ``dD``'s off-diagonal spin blocks give a
    number of the same order as the kept ones -- the plausible wrong answer the
    spiral Hamiltonian's own nonlocal term would produce, since that term turns
    ``D`` with the spins and ``dD`` does not turn.
    """
    system = _system()
    calculation = Calculation(system, _pseudos(system, pseudo_dir))
    delta, _ = _first_order_operator(calculation, None)
    psi, weights = _states(calculation)
    vector, _, dropped, _ = spiral_expectation(calculation, psi, weights, delta)
    assert np.max(np.abs(dropped)) > 1e-2 * np.max(np.abs(vector))
    # The two cross blocks are each other's conjugates, the operator being
    # Hermitian, so what was dropped would have been a real number.
    assert dropped[1] == pytest.approx(np.conj(dropped[0]), rel=1e-12)


@pytest.mark.parametrize("change, error, match", [
    (("spiral_q(3) = 0.25", "spiral_q(3) = 1.0"), ValueError, "reciprocal-lattice"),
    (("lspinorb = .true., soc_scale = 0.0,", ""), ValueError, "lspinorb"),
])
def test_what_the_first_order_refuses(change, error, match, pseudo_dir):
    system = _system(INPUT.replace(*change))
    with pytest.raises(error, match=match):
        spiral_spin_orbit_energy(system, _pseudos(system, pseudo_dir), None)


def test_an_augmented_dataset_is_refused_by_name(pseudo_dir):
    text = INPUT.replace("I.rel-pbe-nc-dojo.UPF", "I.rel-pbe-n-rrkjus_psl.1.0.0.UPF")
    system = _system(text)
    with pytest.raises(NotImplementedError, match="newd_so"):
        spiral_spin_orbit_energy(system, _pseudos(system, pseudo_dir), None)


def test_a_uniform_magnet_is_pointed_at_frozen_expectation(pseudo_dir):
    text = INPUT.replace(
        "    spiral_q(1) = 0.0, spiral_q(2) = 0.0, spiral_q(3) = 0.25\n", "")
    system = _system(text)
    with pytest.raises(ValueError, match="frozen_expectation"):
        spiral_spin_orbit_energy(system, _pseudos(system, pseudo_dir), None)


def _texture_density(turning: bool):
    """A moment of 0.5 turning about ``z`` along the grid's third axis, or not."""
    n = 12
    phase = 2.0 * np.pi * np.arange(n) / n
    density = np.zeros((4, 3, 3, n))
    density[0] = 1.0
    if turning:
        density[1] = 0.5 * np.cos(phase)
        density[2] = -0.5 * np.sin(phase)
    else:
        # A collinear antiferromagnet along x: the sign flips only at its nodes.
        density[1] = 0.5 * np.cos(phase)
    return density


@pytest.mark.parametrize("turning, axis, gradient, fires", [
    (True, (1.0, 0.0, 0.0), True, True),
    (False, (1.0, 0.0, 0.0), True, False),
    (True, None, True, False),
    (True, (1.0, 0.0, 0.0), False, False),
])
def test_a_fixed_gradient_axis_under_a_turning_texture_is_warned(turning, axis, gradient, fires):
    """The warning P123's supercell reference needed, and that it stays quiet otherwise.

    A turning texture handed to a run whose starting moments were all parallel
    is the case that put a supercell's first-order energy 33 times away from
    the spiral's; an antiferromagnet on the same axis only flips sign at its
    nodes, which is what the signed projection is for.
    """
    import warnings
    from types import SimpleNamespace

    from defumat.workflows.nscf import _warn_if_the_sign_axis_cuts_the_texture

    calculation = SimpleNamespace(quantization_axis=axis,
                                  functional=SimpleNamespace(is_gradient=gradient))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_if_the_sign_axis_cuts_the_texture(calculation, _texture_density(turning))
    assert any("fixed axis" in str(w.message) for w in caught) == fires
