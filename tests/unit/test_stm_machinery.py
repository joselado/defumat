"""The pieces an STM image is assembled from, on objects rather than on cells.

``PLAN.md`` P65. The expensive end -- the comparison against ``pp.x``, the
antiferromagnet's two spin images, graphite's missing sublattice -- is in
``tests/regression/test_stm.py``. What is here is everything that can be checked
without an SCF: the point sampler, the plotting plane, the occupation weights,
the spin projection and the refusals.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from defumat.basis.fft import g_to_r, r_to_g
from defumat.basis.gvectors import generate_gvectors
from defumat.basis.sample import sample_field
from defumat.scf.occupations import w0gauss
from defumat.stm.image import (
    constant_current_height,
    project_spin,
    tunnelling_weights,
)
from defumat.stm.plane import plot_plane
from defumat.stm.spectrum import (
    STMSpectrum,
    spectrum_weights,
    state_densities,
)
from defumat.system.cell import Cell
from defumat.workflows.stm import (
    _plane,
    _refuse_a_reduced_k_set,
    _refuse_a_window_under_an_axis,
    _refuse_what_has_no_fermi_level,
)


@pytest.fixture(scope="module")
def cubic():
    return Cell.from_vectors(np.diag([8.0, 8.0, 10.0]))


@pytest.fixture(scope="module")
def sphere(cubic):
    return generate_gvectors(cubic, ecut=6.0)


# --------------------------------------------------------------------------
# the point sampler
# --------------------------------------------------------------------------


def test_the_sampler_is_exact_at_the_grid_points(cubic, sphere):
    """Evaluating the series where the FFT already knows the answer.

    The check that the Miller indices, the crystal coordinates and the sign of
    the phase all agree with :func:`defumat.basis.fft.g_to_r` -- an index
    convention is what a Fourier evaluation gets wrong, and it gets it wrong
    smoothly.
    """
    rng = np.random.default_rng(0)
    coefficients = rng.normal(size=sphere.ngm) + 1.0j * rng.normal(size=sphere.ngm)
    field = np.real(np.asarray(g_to_r(coefficients, sphere.fft_index, sphere.grid)))
    projected = np.real(np.asarray(
        g_to_r(r_to_g(field, sphere.fft_index), sphere.fft_index, sphere.grid)))

    axes = [np.arange(n) / n for n in sphere.grid]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape((-1, 3))
    sampled = sample_field(projected, sphere, points).reshape(sphere.grid)
    assert np.abs(sampled - projected).max() < 1.0e-12


def test_the_sampler_is_the_closed_form_of_a_single_plane_wave(cubic, sphere):
    """One coefficient in, a cosine out -- checked away from every grid point."""
    index = 7
    miller = np.asarray(sphere.miller)[index]
    coefficients = np.zeros(sphere.ngm, dtype=complex)
    coefficients[index] = 1.0
    # The real part of one plane wave is the pair ``(h, -h)`` at half weight
    # each, so what the sampler sees is a cosine and it must reproduce it away
    # from every grid point, where a mistaken index convention still agrees.
    field = np.real(np.asarray(g_to_r(coefficients, sphere.fft_index, sphere.grid)))
    rng = np.random.default_rng(1)
    points = rng.uniform(-1.5, 1.5, size=(23, 3))
    expected = np.cos(2.0 * np.pi * points @ miller)
    assert np.abs(sample_field(field, sphere, points) - expected).max() < 1.0e-12


def test_the_sampler_carries_leading_axes(cubic, sphere):
    rng = np.random.default_rng(2)
    fields = rng.normal(size=(4,) + tuple(sphere.grid))
    points = rng.uniform(0.0, 1.0, size=(5, 3))
    together = sample_field(fields, sphere, points)
    assert together.shape == (4, 5)
    for i, one in enumerate(fields):
        assert np.allclose(together[i], sample_field(one, sphere, points))


def test_the_chunking_does_not_change_the_answer(cubic, sphere):
    rng = np.random.default_rng(3)
    field = rng.normal(size=sphere.grid)
    points = rng.uniform(0.0, 1.0, size=(37, 3))
    whole = sample_field(field, sphere, points, chunk=1000)
    pieces = sample_field(field, sphere, points, chunk=4)
    assert np.abs(whole - pieces).max() < 1.0e-14


# --------------------------------------------------------------------------
# the plotting plane
# --------------------------------------------------------------------------


def test_the_plane_excludes_its_far_edge(cubic):
    """Elk's ``t = i/n``, which is what makes a whole-cell image tile."""
    plane = plot_plane(cubic, (0, 0, 0.5), (1, 0, 0.5), (0, 1, 0.5), shape=(4, 4))
    assert np.allclose(plane.points[0, 0], (0.0, 0.0, 0.5))
    assert np.allclose(plane.points[1, 0], (0.25, 0.0, 0.5))
    assert plane.points[..., 0].max() == pytest.approx(0.75)


def test_the_in_plane_coordinates_are_true_distances():
    """A hexagonal surface cell: 60 degrees between the edges, not 90."""
    a = 4.65
    cell = Cell.from_vectors([[a, 0.0, 0.0],
                              [-0.5 * a, np.sqrt(3) / 2 * a, 0.0],
                              [0.0, 0.0, 20.0]])
    plane = plot_plane(cell, (0, 0, 0.8), (1, 0, 0.8), (0, 1, 0.8), shape=(6, 6))
    assert plane.lengths == pytest.approx((a, a))
    assert np.degrees(plane.angle) == pytest.approx(120.0)
    # the second edge leans back by cos(120) = -1/2
    step = plane.coordinates[0, 1] - plane.coordinates[0, 0]
    assert step[0] == pytest.approx(-0.5 * a / 6)
    assert step[1] == pytest.approx(np.sqrt(3) / 2 * a / 6)


def test_the_normal_is_the_crystal_direction_perpendicular_to_the_plane():
    """``vpnl``: a cartesian normal written back in crystal coordinates.

    On a non-orthogonal cell that is *not* the crystal-coordinate cross product
    of the edges, which is the shape of the mistake.
    """
    a = 4.65
    at = np.array([[a, 0.0, 0.0],
                   [-0.5 * a, np.sqrt(3) / 2 * a, 0.0],
                   [0.0, 0.0, 20.0]])
    cell = Cell.from_vectors(at)
    plane = plot_plane(cell, (0, 0, 0.8), (1, 0, 0.8), (0, 1, 0.8))
    cartesian = plane.normal @ at
    assert np.allclose(cartesian, (0.0, 0.0, 1.0))


def test_a_degenerate_plane_is_refused(cubic):
    with pytest.raises(ValueError, match="zero length"):
        plot_plane(cubic, (0, 0, 0), (0, 0, 0), (0, 1, 0))
    with pytest.raises(ValueError, match="collinear"):
        plot_plane(cubic, (0, 0, 0), (1, 0, 0), (2, 0, 0))


def test_height_and_plane_say_the_same_thing_and_are_exclusive(cubic):
    shortcut = _plane(cubic, 0.75, 2, None, (5, 5))
    corners = _plane(cubic, None, 2,
                     ((0, 0, 0.75), (1, 0, 0.75), (0, 1, 0.75)), (5, 5))
    assert np.allclose(shortcut.points, corners.points)
    with pytest.raises(ValueError, match="two ways of saying the same thing"):
        _plane(cubic, 0.5, 2, ((0, 0, 0), (1, 0, 0), (0, 1, 0)), (5, 5))
    with pytest.raises(ValueError, match="needs a plane"):
        _plane(cubic, None, 2, None, (5, 5))


def test_the_height_shortcut_follows_its_axis(cubic):
    for axis in (0, 1, 2):
        plane = _plane(cubic, 0.4, axis, None, (3, 3))
        assert np.allclose(plane.points[..., axis], 0.4)


# --------------------------------------------------------------------------
# the occupation weights
# --------------------------------------------------------------------------


def _levels(seed=0, nspin=1, nk=6, nbnd=8):
    rng = np.random.default_rng(seed)
    eigenvalues = np.sort(rng.uniform(-0.5, 1.0, size=(nspin, nk, nbnd)), axis=-1)
    weights = rng.uniform(0.5, 2.0, size=nk)
    return eigenvalues, weights * (2.0 / weights.sum())


def test_the_delta_weights_sum_to_the_density_of_states():
    """``sum_kn w_k delta(E - e_kn) = D(E)``, which is the whole sum rule.

    Written against the same expression ``workflows/dos.py`` uses, because the
    point is that the STM density and the density of states integrate to the
    same number -- not that two copies of one formula agree.
    """
    eigenvalues, weights = _levels()
    energy, width = 0.3, 0.05
    for name, ngauss in (("gaussian", 0), ("methfessel-paxton", 1),
                         ("marzari-vanderbilt", -1)):
        w = tunnelling_weights(eigenvalues, weights, energy, width, name)
        x = (energy - eigenvalues) / width
        dos = float(np.einsum("k,skb->", weights, np.asarray(w0gauss(x, ngauss))) / width)
        assert w.sum() == pytest.approx(dos, rel=1e-14)


def test_the_window_weights_count_the_states_inside_it():
    """A window far wider than the smearing counts whole states, no more."""
    eigenvalues, weights = _levels(seed=4)
    energy, width = 0.0, 1.0e-6
    w = tunnelling_weights(eigenvalues, weights, energy, width, bias=0.5)
    inside = (eigenvalues > 0.0) & (eigenvalues < 0.5)
    assert w.sum() == pytest.approx(float((weights[None, :, None] * inside).sum()),
                                    abs=1e-9)


def test_a_negative_bias_images_the_filled_states():
    eigenvalues, weights = _levels(seed=5)
    w = tunnelling_weights(eigenvalues, weights, 0.0, 1.0e-6, bias=-0.5)
    below = (eigenvalues > -0.5) & (eigenvalues < 0.0)
    assert w.sum() == pytest.approx(float((weights[None, :, None] * below).sum()),
                                    abs=1e-9)


def test_the_band_cutoff_is_qes_three_widths():
    """``first_band``/``last_band``: states past ``3 degauss`` are dropped."""
    eigenvalues, weights = _levels(seed=6)
    energy, width = 0.3, 0.05
    full = tunnelling_weights(eigenvalues, weights, energy, width)
    cut = tunnelling_weights(eigenvalues, weights, energy, width, band_cutoff=3.0)
    far = np.abs(eigenvalues - energy) > 3.0 * width
    assert np.all(cut[far] == 0.0)
    assert np.allclose(cut[~far], full[~far])
    # it is an approximation, and on a smearing with wings not a small one
    assert cut.sum() < full.sum()


def test_a_zero_width_is_refused():
    eigenvalues, weights = _levels()
    with pytest.raises(ValueError, match="width must be positive"):
        tunnelling_weights(eigenvalues, weights, 0.0, 0.0)


# --------------------------------------------------------------------------
# the spin projection
# --------------------------------------------------------------------------


def test_the_two_spin_projections_add_back_to_the_charge():
    rng = np.random.default_rng(7)
    density = rng.uniform(0.1, 1.0, size=(2, 4, 4))
    up = project_spin(density, "up")
    down = project_spin(density, "down")
    assert np.allclose(up + down, density[0] + density[1])
    assert np.allclose(up, density[0])
    assert np.allclose(down, density[1])


def test_a_nonmagnetic_tip_sees_the_mean_of_the_two_channels():
    rng = np.random.default_rng(8)
    density = rng.uniform(0.1, 1.0, size=(2, 3, 3))
    flat = project_spin(density, "up", polarization=0.0)
    assert np.allclose(flat, 0.5 * (density[0] + density[1]))


def test_a_noncollinear_projection_follows_the_direction():
    """``(n + n.m)/2``, and a tip perpendicular to the moment sees nothing."""
    charge = np.full((2, 2), 4.0)
    moment = np.zeros((3, 2, 2))
    moment[0] = 1.0  # m along x
    density = np.concatenate([charge[None], moment])
    assert np.allclose(project_spin(density, (1, 0, 0)), 2.5)
    assert np.allclose(project_spin(density, (-1, 0, 0)), 1.5)
    assert np.allclose(project_spin(density, (0, 0, 1)), 2.0)
    assert np.allclose(project_spin(density, (0, 1, 0)), 2.0)
    # the direction is normalised, not taken as given
    assert np.allclose(project_spin(density, (5, 0, 0)), 2.5)


def test_a_transverse_direction_is_refused_for_a_collinear_run():
    """``m_x`` is absent rather than zero, and the two are different claims."""
    density = np.ones((2, 2, 2))
    with pytest.raises(NotImplementedError, match="only m_z"):
        project_spin(density, (1, 0, 0))
    with pytest.raises(NotImplementedError, match="only m_z"):
        project_spin(density, (0, 1, 1))


def test_a_run_with_no_magnetization_has_no_spin_image():
    with pytest.raises(NotImplementedError, match="nothing for the tip"):
        project_spin(np.ones((1, 2, 2)), "up")


def test_an_impossible_polarization_is_refused():
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        project_spin(np.ones((2, 2, 2)), "up", polarization=1.5)


# --------------------------------------------------------------------------
# the constant-current inversion
# --------------------------------------------------------------------------


def test_the_inversion_is_exact_on_an_exponential():
    """The log-linear interpolant is the right one, so it has no error here."""
    kappa = 1.7
    heights = np.linspace(0.0, 4.0, 9)
    values = np.exp(-kappa * heights)[:, None, None] * np.array([[1.0, 3.0]])
    target = 0.05
    found = constant_current_height(heights, values, target)
    expected = -np.log(target / np.array([[1.0, 3.0]])) / kappa
    assert np.abs(found - expected).max() < 1.0e-12


def test_the_outermost_crossing_is_the_one_taken():
    """A tip is withdrawn until the current falls, so the far crossing wins."""
    heights = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    values = np.array([10.0, 0.5, 10.0, 0.5, 0.1]).reshape((5, 1, 1))
    assert constant_current_height(heights, values, 1.0)[0, 0] > 2.0


def test_a_set_point_never_reached_comes_back_nan():
    heights = np.linspace(0.0, 2.0, 5)
    decaying = np.exp(-heights)[:, None, None]
    assert np.isnan(constant_current_height(heights, decaying, 100.0)).all()
    assert np.isnan(constant_current_height(heights, decaying, 1.0e-8)).all()


def test_a_scan_must_be_ordered():
    with pytest.raises(ValueError, match="strictly increasing"):
        constant_current_height([1.0, 0.0], np.ones((2, 1, 1)), 0.5)


def _flat_sample(cell):
    """A tunnelling density that is the same everywhere: ``G = 0`` alone.

    ``_constant_current`` takes the field as a callable rather than as a pair
    ``(coefficients, G-set)``, because an ultracell image samples its own
    ``G + Q`` set in its own coordinates; what the guards below are about is
    the scan rather than the field, so a constant one is enough to reach them.
    """
    from defumat.basis.sample import sample_coefficients

    sphere = generate_gvectors(cell, ecut=6.0)
    coefficients = np.zeros(sphere.ngm, dtype=complex)
    coefficients[0] = 1.0
    return lambda points: sample_coefficients(coefficients, sphere, points)


def test_a_scan_may_not_reach_past_the_cell():
    """The trap the guard is for, and it is silent without one.

    Withdrawing the tip further than the lattice period along the plane's own
    normal brings it up underneath the periodic image of the surface, where the
    tunnelling density rises again -- so the outermost crossing is the wrong
    one, or there is none and the pixel comes back ``nan``. That reads exactly
    like a set-point which is merely too low.
    """
    from defumat.stm.plane import plot_plane
    from defumat.workflows.stm import _constant_current

    cell = Cell.from_vectors(np.diag([12.0, 12.0, 10.0]))
    sample = _flat_sample(cell)
    plane = plot_plane(cell, (0, 0, 0.5), (1, 0, 0.5), (0, 1, 0.5), shape=(2, 2))
    with pytest.raises(ValueError, match="periodic image"):
        _constant_current(sample, plane, cell, 0.5, (0.0, 11.0), 8)


def test_a_scan_measures_its_reach_from_the_plane_not_from_an_axis():
    """A plane whose normal is not the third lattice vector still scans.

    ``_CHAIN_PLANE`` in the regression tests is normal to ``a1``, and taking
    the reach from the ``axis`` argument instead made it zero -- which refused
    every constant-current scan on any plane the shortcut did not build.
    """
    from defumat.stm.plane import plot_plane
    from defumat.workflows.stm import _constant_current

    cell = Cell.from_vectors(np.diag([12.0, 12.0, 10.0]))
    sample = _flat_sample(cell)
    plane = plot_plane(cell, (0.15, 0, 0), (0.15, 0.3, 0), (0.15, 0, 1),
                       shape=(2, 2))
    # the normal is along a1, whose period is 12 bohr -- a scan of 11 fits and
    # one of 13 does not
    with pytest.warns(UserWarning, match="never cross the set-point"):
        _constant_current(sample, plane, cell, 0.5, (0.0, 11.0), 8)
    with pytest.raises(ValueError, match="periodic image"):
        _constant_current(sample, plane, cell, 0.5, (0.0, 13.0), 8)


# --------------------------------------------------------------------------
# what an STM image refuses
# --------------------------------------------------------------------------


def test_a_spin_spiral_is_refused():
    system = SimpleNamespace(spiral_q=(0.0, 0.0, 0.5))
    with pytest.raises(NotImplementedError, match="spin spiral"):
        _refuse_what_has_no_fermi_level(system, SimpleNamespace())


def test_two_fermi_levels_are_refused():
    system = SimpleNamespace(spiral_q=None)
    result = SimpleNamespace(fermi_energy_up=0.1, magnetic_field=None)
    with pytest.raises(NotImplementedError, match="tot_magnetization"):
        _refuse_what_has_no_fermi_level(system, result)


def test_an_applied_field_is_refused():
    system = SimpleNamespace(spiral_q=None)
    result = SimpleNamespace(fermi_energy_up=None, magnetic_field=object())
    with pytest.raises(NotImplementedError, match="magnetic field"):
        _refuse_what_has_no_fermi_level(system, result)


def test_the_calculator_reaches_it():
    from defumat.calculator import Calculator

    assert hasattr(Calculator, "get_stm")


# --------------------------------------------------------------------------
# the spectrum: the same sum, sectioned the other way (P90)
# --------------------------------------------------------------------------


def test_a_spectrum_at_one_energy_is_the_image_s_own_weights():
    """``spectrum_weights`` is ``tunnelling_weights`` stacked and nothing else.

    The reason to check something this thin: the delta, the window, the
    k-weights and ``band_cutoff`` have one implementation, and a second one
    would drift. A spectrum at one energy has to be the image at that energy to
    the last bit, not to a tolerance.
    """
    rng = np.random.default_rng(7)
    eigenvalues = np.sort(rng.normal(size=(2, 3, 5)), axis=-1)
    kweights = np.array([0.5, 0.25, 0.25])
    for bias, cutoff in ((None, None), (0.4, None), (-0.4, 3.0)):
        stacked = spectrum_weights(eigenvalues, kweights, [0.1, 0.3],
                                   width=0.2, bias=bias, band_cutoff=cutoff)
        for at, energy in enumerate((0.1, 0.3)):
            one = tunnelling_weights(eigenvalues, kweights, energy=energy,
                                     width=0.2, bias=bias, band_cutoff=cutoff)
            assert np.array_equal(stacked[at], one)


def test_a_spinor_state_s_four_channels_are_the_density_s_own_bilinears():
    """``(n, m_x, m_y, m_z)`` in ``spinor_band_density``'s convention.

    **The transverse pair is what needs checking and a collinear state cannot
    check it**: ``m_y = 2 Im(conj(u) d)`` with the two factors exchanged is
    still real, still the right size and wrong in sign, and it is zero on
    anything polarized along ``z``. So the state here has a moment in the
    ``xy`` plane, where a swapped index reverses ``m_y`` and shows.
    """
    # one state, one point: a spinor at 60 degrees from z in the xz plane,
    # then given a phase so that m_y is not zero either.
    theta, phi = np.pi / 3.0, 0.7
    up = np.cos(theta / 2.0)
    down = np.sin(theta / 2.0) * np.exp(1.0j * phi)
    amplitudes = np.array([[[up], [down]]])           # (nbnd, npol, npoints)
    n, mx, my, mz = state_densities(amplitudes, nspin_mag=4)[:, 0, 0]

    assert n == pytest.approx(1.0)
    assert mx == pytest.approx(np.sin(theta) * np.cos(phi))
    assert my == pytest.approx(np.sin(theta) * np.sin(phi))
    assert mz == pytest.approx(np.cos(theta))
    # the positive control: the exchanged pair is what a transposed index gives
    swapped = 2.0 * np.imag(np.conj(down) * up)
    assert swapped == pytest.approx(-my)
    assert abs(my) > 0.5           # so the sign is not a null

    # and a spin-orbit run with no magnetization keeps only the charge
    charge = state_densities(amplitudes, nspin_mag=1)
    assert charge.shape == (1, 1, 1)
    assert charge[0, 0, 0] == pytest.approx(n)


def test_a_degenerate_multiplet_can_be_rotated_without_moving_the_spectrum():
    """Rule D4 does not reach this sum, and here is the reason made into a test.

    A weight that depends on the state only through its eigenvalue is constant
    inside a degenerate block, so the sum over the block is the trace of its
    projector and no unitary mixing of its members can change it. The control is
    the same rotation applied to states that are **not** degenerate, which does
    move the answer -- without it this test would pass on a spectrum that had
    quietly become independent of the states.
    """
    rng = np.random.default_rng(11)
    amplitudes = (rng.normal(size=(4, 1, 6))
                  + 1.0j * rng.normal(size=(4, 1, 6)))
    block = np.linalg.qr(rng.normal(size=(4, 4))
                         + 1.0j * rng.normal(size=(4, 4)))[0]
    mixed = np.einsum("jn,npq->jpq", block, amplitudes)

    degenerate = np.full((1, 1, 4), 0.3)
    weights = spectrum_weights(degenerate, np.array([1.0]), [0.3, 0.5],
                               width=0.1)[:, 0, 0]
    plain = np.einsum("en,cnp->cep", weights, state_densities(amplitudes))
    rotated = np.einsum("en,cnp->cep", weights, state_densities(mixed))
    assert np.abs(rotated - plain).max() < 1e-13

    split = np.array([[[0.3, 0.32, 0.34, 0.36]]])
    weights = spectrum_weights(split, np.array([1.0]), [0.3, 0.5],
                               width=0.1)[:, 0, 0]
    plain = np.einsum("en,cnp->cep", weights, state_densities(amplitudes))
    rotated = np.einsum("en,cnp->cep", weights, state_densities(mixed))
    assert np.abs(rotated - plain).max() > 1e-3


def test_the_current_is_the_integral_of_the_conductance():
    """``I(V)`` from ``dI/dV``, outwards from zero bias and with its sign."""
    axis = np.linspace(-0.4, 0.4, 81)
    flat = STMSpectrum(values=np.ones((81, 2)), energies=axis,
                       points=np.zeros((2, 3)), fermi_energy=0.0, width=0.05)
    assert np.abs(flat.current - axis[:, None]).max() < 1e-13
    assert flat.current[np.argmin(np.abs(axis))] == pytest.approx(0.0, abs=1e-14)
    # the filled side comes back negative, which is the experiment's sign
    assert flat.current[0].max() < 0.0 < flat.current[-1].min()


def test_an_axis_that_cannot_resolve_the_delta_is_refused_rather_than_integrated():
    """The guard fires, and it is fed the case that must trip it.

    A trapezoid over a delta it steps across does not return a slightly wrong
    number, it returns a number of the wrong order -- and the default width for
    a fixed-occupation run is ``1e-5`` Ry, which no sane axis resolves. So the
    refusal is checked by the axis that would have produced the wrong answer,
    and the neighbouring case that must still pass.
    """
    axis = np.linspace(-0.4, 0.4, 41)
    common = dict(energies=axis, points=np.zeros((2, 3)), fermi_energy=0.0)
    with pytest.raises(ValueError, match="steps over the levels"):
        STMSpectrum(values=np.ones((41, 2)), width=1.0e-5, **common).current
    assert STMSpectrum(values=np.ones((41, 2)), width=0.02, **common).current is not None

    with pytest.raises(ValueError, match="no zero of bias"):
        STMSpectrum(values=np.ones((41, 2)), energies=axis, width=0.02,
                    points=np.zeros((2, 3))).current
    with pytest.raises(ValueError, match="already a window count"):
        STMSpectrum(values=np.ones((41, 2)), width=0.02, bias=0.1,
                    **common).current
    with pytest.warns(UserWarning, match="does not include V = 0"):
        STMSpectrum(values=np.ones((41, 2)), energies=axis + 1.0, width=0.02,
                    points=np.zeros((2, 3)), fermi_energy=0.0).current


def test_the_two_knobs_an_energy_axis_turns_into_another_experiment():
    """``bias=`` and constant current, refused by name under an axis."""
    with pytest.raises(ValueError, match="asking for I\\(V\\) twice"):
        _refuse_a_window_under_an_axis(0.1, "constant-height")
    with pytest.raises(NotImplementedError, match="the tip moves"):
        _refuse_a_window_under_an_axis(None, "constant-current")
    assert _refuse_a_window_under_an_axis(None, "constant-height") is None


def test_a_wedge_is_refused_because_nothing_here_symmetrises_it():
    """The one thing a spectrum does not inherit from an image.

    ``run_stm`` sums through ``Calculation.density``, which symmetrises; the
    amplitude route adds ``|psi_k(r)|^2`` with nothing to symmetrise it, so a
    reduced k-set would give the sum over the wedge alone. Both branches are
    checked, since a guard that never fires and a guard that always fires look
    the same from one side.
    """
    reduced = SimpleNamespace(kpoints=SimpleNamespace(
        weights=np.array([0.25, 0.75])))
    with pytest.raises(NotImplementedError, match="symmetry-reduced"):
        _refuse_a_reduced_k_set(reduced)
    whole = SimpleNamespace(kpoints=SimpleNamespace(weights=np.full(4, 0.5)))
    assert _refuse_a_reduced_k_set(whole) is None
    one = SimpleNamespace(kpoints=SimpleNamespace(weights=np.array([2.0])))
    assert _refuse_a_reduced_k_set(one) is None
