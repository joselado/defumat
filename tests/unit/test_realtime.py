"""Real-time propagation at a frozen potential: the step, the field, the current.

Each test here is a property that holds without a reference: the field is the
derivative of the vector potential, the Taylor step is the exponential to its
order, a state with no field is stationary, the work the field does is the
energy it puts in, an unstable step is refused, and the program a block
compiles serves every k-chunk after the first. The comparisons against the
Kubo sum and the dense hierarchy are in ``tests/regression/test_realtime.py``.
"""

import logging
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.realtime.propagate import propagate
from defumat.realtime.propagators import TAYLOR4_BOUND, taylor4
from defumat.realtime.pulse import (
    FS_TO_AU, Adiabatic, Gaussian, Kick, Ramp, Sin2, field_amplitude, get_pulse)
from defumat.scf import Calculation
from defumat.system import build_system
from defumat.system.kpoints import KPoints

pytestmark = pytest.mark.unit

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"


@pytest.fixture(autouse=True)
def _bounded_compilation():
    yield
    jax.clear_caches()


def test_the_field_is_minus_the_derivative_of_kappa():
    """``E = -dkappa/dt`` by differentiation, against the closed form, for every shape."""
    t = np.linspace(-50.0, 400.0, 301)
    gauss = Gaussian(amplitude=0.1, omega=0.06, fwhm=80.0, peak=150.0, phase=30.0)
    sigma = 80.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    s = t - 150.0
    envelope = np.exp(-0.5 * (s / sigma) ** 2)
    arg = 0.06 * s + np.pi / 6.0
    closed = -0.1 * envelope * (-s / sigma**2 * np.sin(arg) + 0.06 * np.cos(arg))
    np.testing.assert_allclose(gauss.efield(t)[:, 0], closed, atol=1e-14)
    ramp = Ramp(amplitude=2.0, coefficients=(0.5, 0.01), direction=(0, 0, 1), start_time=10.0)
    later = t[t > 10.5]
    np.testing.assert_allclose(ramp.efield(later)[:, 2],
                               -2.0 * (0.5 + 2 * 0.01 * (later - 10.0)), rtol=1e-13)
    adiabatic = Adiabatic(amplitude=1.0, omega=0.3, eta=0.05, eta_t=10.0)
    assert adiabatic.start == pytest.approx(-200.0)
    assert np.all(Kick(0.01).efield(t) == 0.0)


def test_an_intensity_is_the_field_elk_prints():
    """``E0 = sqrt(I / 3.51e16 W/cm^2)``: 1e12 W/cm^2 is 5.338e-3 a.u., as Elk's ``ppd`` reads."""
    assert field_amplitude(3.5094455205905376e16) == pytest.approx(1.0, rel=1e-14)
    assert field_amplitude(1e12) == pytest.approx(5.33807e-3, rel=1e-5)
    pulse = Sin2.from_intensity(1e12, 1.55, 4)
    period = 2 * np.pi / pulse.omega
    assert pulse.natural_duration == pytest.approx(4 * period)
    # the peak field is E0 to the envelope's own slope: 0.966 E0 over four cycles
    t = np.linspace(0.0, pulse.natural_duration, 4001)
    assert np.abs(pulse.efield(t)).max() == pytest.approx(5.33807e-3, rel=5e-2)
    assert FS_TO_AU == pytest.approx(41.3413746, rel=1e-7)  # units.AU_SEC, 24.189 as
    assert isinstance(get_pulse("kick", strength=1e-3), Kick)


def test_the_taylor_step_is_the_exponential_to_fourth_order():
    """Against ``expm`` on a random Hermitian matrix: the error falls as ``dt^5``.

    And the bound is where it says: inside ``|dt rho| <= 2 sqrt 2`` the step
    never grows a norm, just outside it does.
    """
    rng = np.random.default_rng(1)
    a = rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12))
    h = 0.5 * (a + a.conj().T)
    h = h / np.abs(np.linalg.eigvalsh(h)).max()  # spectrum in [-1, 1]
    psi = rng.normal(size=12) + 1j * rng.normal(size=12)
    psi = psi / np.linalg.norm(psi)
    errors = []
    for dt in (0.1, 0.05):
        exact = scipy.linalg.expm(-1j * dt * h) @ psi
        step = np.asarray(taylor4(lambda v: jnp.asarray(h) @ v, jnp.asarray(psi), dt, 0.0))
        errors.append(np.linalg.norm(step - exact))
    assert errors[0] / errors[1] == pytest.approx(32.0, rel=0.05)
    inside = np.asarray(taylor4(lambda v: jnp.asarray(h) @ v, jnp.asarray(psi),
                                0.99 * TAYLOR4_BOUND, 0.0))
    assert np.linalg.norm(inside) <= 1.0 + 1e-12
    eigvec = np.linalg.eigh(h)[1][:, -1]
    outside = np.asarray(taylor4(lambda v: jnp.asarray(h) @ v, jnp.asarray(eigvec),
                                 1.05 * TAYLOR4_BOUND / np.abs(np.linalg.eigvalsh(h)).max(), 0.0))
    assert np.linalg.norm(outside) > 1.0


@lru_cache(maxsize=2)
def _silicon(pseudo_dir, ecut=6.0):
    """Two-atom silicon at a small cutoff on a 2x2x2 grid, with its dense ground states."""
    import re

    from defumat.realtime.dense import dense_hamiltonians

    text = re.sub(r"ecutwfc\s*=\s*[0-9.]+", f"ecutwfc = {ecut}",
                  (CASES / "si2-nosym.in").read_text())
    path = Path("/tmp") / f"defumat-realtime-si-{ecut}.in"
    path.write_text(text)
    import equinox as eqx

    system = build_system(read_pw_input(path))
    system = eqx.tree_at(lambda s: s.kpoints, system,
                         KPoints.automatic((2, 2, 2), (0, 0, 0), system.cell))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    v_scf = calculation.potential(calculation.starting_density()).v_scf
    terms = calculation.local_terms(v_scf)
    mask = np.asarray(calculation.basis.planewaves.mask)
    nk, npwx = mask.shape
    states = np.zeros((nk, 4, npwx), dtype=complex)
    for ik in range(nk):
        h = dense_hamiltonians(calculation, terms, ik, (1.0, 0.0, 0.0), 0)[0]
        vectors = np.linalg.eigh(h)[1]
        states[ik][:, np.flatnonzero(mask[ik])] = vectors[:, :4].T
    weights = np.repeat(np.asarray(calculation.system.kpoints.weights)[:, None], 4, axis=1)
    return calculation, jnp.asarray(states), weights, v_scf


def test_with_no_field_the_states_are_stationary(pseudo_dir):
    """``J`` and the energy constant to round-off, the norm to the step's ``y^6/144``.

    Measured on two-atom silicon at 6 Ry, 2x2x2, 200 steps of 0.1 from exact
    dense eigenstates: the energy moves by 2.0e-10 Ha, linearly in time, which
    is the norm the step loses at ``(dt (e - centre))^6/144`` a step on each
    band times its energy. The current is zero by symmetry on
    the whole grid, so its value says nothing; what is asserted of it is that
    it does not move.
    """
    calculation, states, weights, v_scf = _silicon(pseudo_dir)
    result = propagate(calculation, states, weights, v_scf, Kick(0.0), dt=0.1,
                       duration=20.0, block_steps=100)
    assert np.abs(result.energy - result.energy[0]).max() < 1e-9
    assert result.norm_drift < 1e-8
    assert np.abs(result.current - result.current[0]).max() < 1e-10
    assert result.excited < 1e-8


def test_the_work_done_is_the_energy_gained(pseudo_dir):
    """``Omega int J.E dt = Delta E`` for a pulse, to the trapezoid rule's ``dt^2``.

    Blind to the time unit and to any constant on ``J`` or ``E`` (a step
    under ``2H`` satisfies it); what it sees is a current that is not the
    ``kappa`` derivative of the Hamiltonian the step applies, and a step that
    is not unitary. Measured 2.4e-6 relative on the 12 Ry cell at ``dt = 0.1``.
    """
    calculation, states, weights, v_scf = _silicon(pseudo_dir)
    pulse = Sin2.from_intensity(1e12, 1.55, 2, (1.0, 0.0, 0.0))
    result = propagate(calculation, states, weights, v_scf, pulse, dt=0.1, block_steps=500)
    assert result.energy_gained > 1e-4, "the pulse did work"
    assert result.work == pytest.approx(result.energy_gained, rel=1e-4)


def test_a_step_past_the_stability_bound_is_refused(pseudo_dir):
    calculation, states, weights, v_scf = _silicon(pseudo_dir)
    with pytest.raises(ValueError, match="stability bound"):
        propagate(calculation, states, weights, v_scf, Kick(0.0), dt=5.0, duration=10.0)


def test_a_second_k_chunk_compiles_nothing(pseudo_dir):
    """The eager-closure trap, checked the way ``CLAUDE.md`` says: count compilations.

    Every chunk's arrays reach the kept programs as arguments, so after the first
    chunk of a pass has compiled them the others compile none, in the pass that
    bounds the spectrum and in the propagation. The counter is validated on the
    first chunk of the spectrum's pass, which must compile something.
    """
    from defumat.realtime import propagate as module

    calculation, states, weights, v_scf = _silicon(pseudo_dir)
    count = [0]

    class Counter(logging.Handler):
        def emit(self, record):
            if "Finished XLA compilation" in record.getMessage() or \
                    "Compiling" in record.getMessage():
                count[0] += 1

    handler = Counter()
    logger = logging.getLogger("jax")
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    built = []
    original = module._Chunk.build

    def build(*args, **kwargs):
        built.append(count[0])
        return original(*args, **kwargs)

    module._Chunk.build = build
    jax.config.update("jax_log_compiles", True)
    try:
        propagate(calculation, states, weights, v_scf,
                  Sin2.from_intensity(1e11, 1.55, 1), dt=0.2, block_steps=50,
                  k_batch=2)
        built.append(count[0])
    finally:
        jax.config.update("jax_log_compiles", False)
        module._Chunk.build = original
        logger.removeHandler(handler)
        logger.setLevel(previous)
    # Four chunks of two k-points. The builds are: the setup's first chunk, the
    # spectrum's pass over all four, then the propagation's chunks 1 to 3 (its
    # chunk 0 is the setup's), and the end of the run. Each pass compiles on its
    # first chunk and on no other.
    assert len(built) == 9, built
    assert built[2] > built[1], "the counter saw the spectrum's first chunk compile"
    assert built[2] == built[3] == built[4], built
    assert built[6] == built[7] == built[8], built


def test_what_is_refused_is_refused_by_name(pseudo_dir):
    """An ultrasoft dataset, a collinear run and a wedge passed as the k-set."""
    from defumat.realtime.propagate import require_a_realtime_regime
    from defumat.workflows.realtime import _kset
    from defumat.realtime.pulse import Sin2

    system = build_system(read_pw_input(CASES / "si2-us.in"))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    with pytest.raises(NotImplementedError, match="ultrasoft or PAW"):
        require_a_realtime_regime(Calculation(system, pseudos))
    silicon = build_system(read_pw_input(CASES / "si2-symmetric.in"))
    pulse = Sin2.from_intensity(1e11, 1.55, 1)
    with pytest.raises(NotImplementedError, match="symmetry-reduced"):
        _kset(silicon, pulse, silicon.kpoints, None, True)
    _, rotations = _kset(silicon, pulse, None, (4, 4, 4), True)
    assert len(rotations) == 8, "a [100] field keeps eight of silicon's 48"


def test_a_checkpoint_resumes_after_the_chunks_it_records(pseudo_dir, tmp_path):
    """A run that finds its own checkpoint returns the recorded current unchanged.

    The unit of restart is a finished k-chunk, which in the frozen mode is
    independent of every other: the file holds the current and the energy
    summed over the chunks done, and a run on the same time grid and k-set
    starts after them. A file from another grid is ignored.
    """
    calculation, states, weights, v_scf = _silicon(pseudo_dir)
    pulse = Sin2.from_intensity(1e11, 1.55, 1)
    path = tmp_path / "rt.npz"
    first = propagate(calculation, states, weights, v_scf, pulse, dt=0.2,
                      block_steps=200, k_batch=4, checkpoint=path)
    saved = np.load(path)
    assert int(saved["done"]) == 2
    again = propagate(calculation, states, weights, v_scf, pulse, dt=0.2,
                      block_steps=200, k_batch=4, checkpoint=path)
    np.testing.assert_array_equal(again.current, first.current)
    other = propagate(calculation, states, weights, v_scf, pulse, dt=0.25,
                      block_steps=200, k_batch=4, checkpoint=path)
    assert other.current.shape != first.current.shape


def test_a_checkpoint_of_another_run_is_not_resumed(pseudo_dir, tmp_path):
    """The digest holds the field, so a pulse of twice the amplitude at the same length is another run.

    Found in review: the signature was the grid and the k-count alone, and a
    run at twice the amplitude returned the first one's current bit for bit,
    43 per cent off.
    """
    calculation, states, weights, v_scf = _silicon(pseudo_dir)
    path = tmp_path / "rt.npz"
    weak = propagate(calculation, states, weights, v_scf, Sin2.from_intensity(1e11, 1.55, 1),
                     dt=0.2, block_steps=200, k_batch=4, checkpoint=path)
    strong = propagate(calculation, states, weights, v_scf, Sin2.from_intensity(4e11, 1.55, 1),
                       dt=0.2, block_steps=200, k_batch=4, checkpoint=path)
    clean = propagate(calculation, states, weights, v_scf, Sin2.from_intensity(4e11, 1.55, 1),
                      dt=0.2, block_steps=200, k_batch=4)
    np.testing.assert_array_equal(strong.current, clean.current)
    assert np.abs(strong.current - weak.current).max() > 0.5 * np.abs(weak.current).max()


def test_the_chunk_size_is_not_in_the_current(pseudo_dir):
    """``k_batch`` moves the current by round-off and nothing more.

    The centre of the step and the spectrum are taken over every k-point; built
    on the first chunk, the centre was 0.224, 0.117 and 0.096 Ry at ``k_batch``
    1, 4 and 8 and the current moved by 3.9e-7 of its size (found in review).
    ``'fit'`` reads the calculation's own chunk.
    """
    calculation, states, weights, v_scf = _silicon(pseudo_dir)
    pulse = Sin2.from_intensity(1e12, 1.55, 1)
    runs = [propagate(calculation, states, weights, v_scf, pulse, dt=0.2, block_steps=200,
                      k_batch=batch) for batch in (1, 4, "fit")]
    scale = np.abs(runs[0].current).max()
    for other in runs[1:]:
        assert np.abs(other.current - runs[0].current).max() < 1e-12 * scale


def test_a_projection_over_a_broken_period_is_refused():
    """``fourier_component`` needs a step that divides the period, or it leaks the other harmonics."""
    from defumat.realtime.orders import fourier_component

    omega = 0.05
    times = np.arange(0.0, 400.0, 0.3)
    with pytest.raises(ValueError, match="does not divide the period"):
        fourier_component(times, np.zeros((len(times), 3)), 3, 3, omega, 0.01)
    period = 2 * np.pi / omega
    times = -period * 10 + np.arange(4001) * period / 400
    signal = np.cos(3 * omega * times)[:, None] * np.ones(3)
    value = fourier_component(times, signal, 0, 3, omega, 0.0)
    np.testing.assert_allclose(value, 0.5, atol=1e-12)


def test_the_cutoff_is_the_end_of_an_unbroken_plateau():
    """An isolated peak past a gap does not move the cutoff; found in review, where it read 31 for 9."""
    from defumat.realtime.spectra import HarmonicSpectrum

    orders = np.linspace(0, 40, 4001)
    total = np.full_like(orders, 1e-12)
    for n, height in ((1, 1.0), (3, 0.5), (5, 0.3), (7, 0.2), (9, 0.1), (31, 2e-3)):
        total[np.abs(orders - n) < 0.05] = height
    spectrum = HarmonicSpectrum(frequencies=orders, orders=orders,
                                intensity=total[:, None] * np.ones(3), total=total,
                                omega=1.0, window="none")
    assert spectrum.cutoff(floor=1e-3) == 9
