"""The mixer's density on the smooth sphere in G, ``mixing_space = 'g'``.

``pw.x``'s ``mix_rho`` keeps ``of_g(1:ngms, nspin)`` (``scf_mod.f90:216``,
``:316``), fits on it (``ngm0 = ngms``, ``mix_rho.f90:132``) and mixes the shell
between ``ngms`` and ``ngm`` linearly without storing it
(``high_frequency_mixing``, ``scf_mod.f90:549-553``).
:class:`~defumat.scf.mixing.SphereLayout` is that layout. What is pinned here:
the shell is mixed at ``beta`` and never reaches the history; an entry is
``ngms`` reals a channel; the stored vector's flat dot is the real-space dot, so
at dual 4, where the shell is empty, the layout is the real-space mixer to
round-off and so are its Kerker and ``rho_ddot``; a converged dual-8 run lands on
the real-space energy; and a resume carries the G history, and drops a history
written in the other layout with a warning.
"""

from pathlib import Path
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.scf.mixing import (
    SphereLayout,
    get_mixer,
    kerker_preconditioner,
    kerker_preconditioner_g,
    local_tf_preconditioner,
    local_tf_preconditioner_g,
    resolve_mixing_space,
)
from defumat.scf.potential import scf_accuracy

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"


def _pieces(name):
    """The basis and cell of an input, built without an SCF."""
    from defumat.basis.builder import build_basis
    from defumat.io.pwin import read_pw_input
    from defumat.system import build_system

    system = build_system(read_pw_input(BENCHMARKS / name))
    return build_basis(system), system.cell


@pytest.fixture(scope="module")
def dual8():
    """Ultrasoft silicon at ``ecutrho = 8 ecutwfc``: a shell to mix."""
    return _pieces("si2-us-1k.in")


@pytest.fixture(scope="module")
def dual4():
    """Norm-conserving silicon at dual 4: ``ngms = ngm``, no shell at all."""
    return _pieces("si-1k.in")


def _layout(pieces, nspin):
    basis, cell = pieces
    return SphereLayout(basis.dense, basis.ngms, cell, (nspin,) + basis.dense.grid)


def _shell_coefficients(layout, rng):
    shape = (layout.nspin, layout.nshell)
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)) * 1e-2


def _field(layout, rng, smooth=True, shell=True):
    """A real field band-limited to the dense sphere, from random stored parts."""
    stored = rng.normal(size=layout.stored_shape) if smooth else np.zeros(
        layout.stored_shape)
    coefficients = _shell_coefficients(layout, rng) if shell else None
    return np.asarray(layout.field(stored, coefficients))


def _density(layout, rng):
    """A positive field on the smooth sphere: a mean of 0.05 and small ripples."""
    stored = 1e-3 * rng.normal(size=layout.stored_shape)
    stored[:, 0] = 0.05 * np.sqrt(layout.points)
    return np.asarray(layout.field(stored))


def _shell(layout, field):
    return np.asarray(layout.forward(field))[:, layout.nsmooth:]


def test_a_history_entry_is_the_smooth_sphere_and_not_the_box(dual8):
    """``ngms`` reals a channel, against ``n1 n2 n3`` for the box it replaced."""
    from defumat.scf.driver import _mix

    layout = _layout(dual8, 2)
    assert layout.ngms < layout.ngm < layout.points
    rng = np.random.default_rng(1)
    mixer = get_mixer("anderson", beta=0.4)
    mixer.layout, mixer.shape = layout, layout.stored_shape
    _mix(mixer, _field(layout, rng), _field(layout, rng), (), ())
    _mix(mixer, _field(layout, rng), _field(layout, rng), (), ())
    assert len(mixer._densities) == len(mixer._residuals) == 2
    for entry in mixer._densities + mixer._residuals:
        assert entry.size == 2 * layout.ngms
    assert layout.history_bytes(8) == 2 * 8 * 2 * layout.ngms * 8
    # The box an entry used to be, against the smooth sphere it is: 0.14 of it
    # on this cell (``ngms`` reals against ``n1 n2 n3``), and half of what
    # ``pw.x``'s complex ``of_g(ngms)`` holds.
    assert layout.ngms < 0.2 * layout.points


def test_the_shell_is_mixed_linearly_and_not_stored(dual8):
    """A change confined to the shell moves the shell by ``beta`` and nothing else.

    Two runs of three Anderson steps that differ only in the shell of every
    output density: the histories are equal, so the shell was neither stored
    nor fitted, and the mixed densities differ by exactly ``beta`` times the
    shell difference, which is ``high_frequency_mixing``.
    """
    from defumat.scf.driver import _mix

    layout = _layout(dual8, 1)
    beta = 0.3
    rng = np.random.default_rng(2)
    inputs = [_field(layout, rng) for _ in range(3)]
    outputs = [_field(layout, rng) for _ in range(3)]
    kicks = [_field(layout, rng, smooth=False) for _ in range(3)]

    runs = []
    for kicked in (False, True):
        mixer = get_mixer("anderson", beta=beta)
        mixer.layout, mixer.shape = layout, layout.stored_shape
        mixed = []
        for rho_in, rho_out, kick in zip(inputs, outputs, kicks):
            out = rho_out + kick if kicked else rho_out
            mixed.append(np.asarray(_mix(mixer, rho_in, out, (), ())[0]))
            shell_in, shell_out = _shell(layout, rho_in), _shell(layout, out)
            np.testing.assert_allclose(
                _shell(layout, mixed[-1]), shell_in + beta * (shell_out - shell_in),
                rtol=0, atol=1e-13)
        runs.append((mixer, mixed))

    (plain, plain_mixed), (kicked, kicked_mixed) = runs
    for one, two in zip(plain._residuals + plain._densities,
                        kicked._residuals + kicked._densities):
        np.testing.assert_allclose(one, two, rtol=0, atol=1e-12)
    for one, two, kick in zip(plain_mixed, kicked_mixed, kicks):
        np.testing.assert_allclose(two - one, beta * kick, rtol=0, atol=1e-12)


def test_the_stored_dot_is_the_real_space_dot(dual8):
    """Parseval with the pair counted twice: the flat fit's weights are unchanged."""
    layout = _layout(dual8, 2)
    rng = np.random.default_rng(3)
    f, g = _field(layout, rng, shell=False), _field(layout, rng, shell=False)
    assert np.max(np.abs(_shell(layout, f))) < 1e-15 * np.max(np.abs(f))
    stored = float(layout.stored_of(f).ravel() @ layout.stored_of(g).ravel())
    assert stored == pytest.approx(float(np.sum(f * g)), rel=1e-12)


@pytest.mark.parametrize("nspin", [1, 2])
def test_at_dual_four_the_layout_is_the_real_space_mixer(dual4, nspin):
    """No shell, so the G history is the box history in another basis.

    Fed the same pairs, four Anderson steps on either layout give the same
    densities to round-off, which is what the stored scaling is for.
    """
    from defumat.scf.driver import _mix

    layout = _layout(dual4, nspin)
    assert layout.ngms == layout.ngm
    rng = np.random.default_rng(4 + nspin)
    pairs = [(_field(layout, rng), _field(layout, rng)) for _ in range(4)]
    real, sphere = get_mixer("anderson", beta=0.5), get_mixer("anderson", beta=0.5)
    sphere.layout, sphere.shape = layout, layout.stored_shape
    for rho_in, rho_out in pairs:
        a = np.asarray(_mix(real, rho_in, rho_out, (), ())[0])
        b = np.asarray(_mix(sphere, rho_in, rho_out, (), ())[0])
        np.testing.assert_allclose(b, a, rtol=0, atol=1e-12 * np.max(np.abs(a)))


@pytest.mark.parametrize("nspin", [1, 2, 4])
def test_kerker_in_g_is_a_multiplication_and_the_real_space_kerker(dual4, nspin):
    """``approx_screening`` on the stored charge, no transform, the same operator."""
    basis, cell = dual4
    layout = _layout(dual4, nspin)
    rng = np.random.default_rng(10 + nspin)
    residual = _field(layout, rng) - _field(layout, rng)
    real = kerker_preconditioner(basis.dense, cell, layout.shape, beta=0.6, nelec=8.0)
    sphere = kerker_preconditioner_g(layout, cell, beta=0.6, nelec=8.0)
    expected = np.asarray(real(residual.ravel())).reshape(layout.shape)
    stored = sphere(layout.stored_of(residual).ravel())
    got = np.asarray(layout.field(stored))
    np.testing.assert_allclose(got, expected, rtol=0,
                               atol=1e-12 * np.max(np.abs(expected)))


def test_local_tf_in_g_is_the_real_space_solver_on_the_same_sphere(dual4):
    """At dual 4 the smooth sphere is the dense one, so the two solve one system."""
    basis, cell = dual4
    layout = _layout(dual4, 1)
    rng = np.random.default_rng(20)
    residual = 1e-3 * (_field(layout, rng) - _field(layout, rng))
    density = _density(layout, rng)
    assert np.all(density > 0.0)
    real = local_tf_preconditioner(basis.dense, cell, layout.shape, beta=0.7)
    sphere = local_tf_preconditioner_g(layout, basis.dense, cell, beta=0.7)
    expected = np.asarray(real(residual.ravel(), density.ravel())).reshape(layout.shape)
    stored = sphere(layout.stored_of(residual).ravel(),
                    layout.stored_of(density).ravel())
    got = np.asarray(layout.field(stored))
    np.testing.assert_allclose(got, expected, rtol=0,
                               atol=1e-9 * np.max(np.abs(expected)))


@pytest.mark.parametrize("nspin", [1, 2, 4])
def test_the_stored_rho_ddot_is_the_loops_accuracy_on_the_smooth_sphere(dual4, nspin):
    """``rho_ddot(r, r)`` over ``ngm0``, which at dual 4 is the loop's own ``dr2``."""
    basis, cell = dual4
    layout = _layout(dual4, nspin)
    rng = np.random.default_rng(30 + nspin)
    residual = _field(layout, rng) - _field(layout, rng)
    f = layout.rho_ddot_vector(layout.stored_of(residual))
    expected = float(scf_accuracy(jnp.asarray(residual), basis.dense, cell))
    assert float(f @ f) == pytest.approx(expected, rel=1e-11)


@pytest.mark.parametrize("nspin", [1, 2])
def test_at_dual_four_an_installed_augmentation_changes_no_bit(dual4, nspin):
    """No shell, so the rebuild is never reached and the mixed density is byte-equal.

    ``run_scf`` installs the augmentation only where there is a shell; this
    is the guard that a layout which has one installed anyway at dual 4 does
    not call it and moves nothing.
    """
    from defumat.scf.driver import _mix

    def augmentation(base, becsum):
        raise AssertionError("a dual-4 layout has no shell to rebuild")

    bare, hooked = _layout(dual4, nspin), _layout(dual4, nspin)
    hooked.augmentation = augmentation
    rng = np.random.default_rng(50 + nspin)
    # A ``becsum`` block rides along, so the rebuild would be reached if a
    # shell-less layout did not stop before it.
    pairs = [(_field(hooked, rng), _field(hooked, rng),
              (rng.normal(size=(nspin, 2, 4, 4)),), (rng.normal(size=(nspin, 2, 4, 4)),))
             for _ in range(3)]
    mixers = []
    for layout in (bare, hooked):
        mixer = get_mixer("anderson", beta=0.4)
        mixer.layout, mixer.shape = layout, layout.stored_shape
        mixers.append(mixer)
    for rho_in, rho_out, becsum_in, becsum_out in pairs:
        a = _mix(mixers[0], rho_in, rho_out, becsum_in, becsum_out)
        b = _mix(mixers[1], rho_in, rho_out, becsum_in, becsum_out)
        assert np.asarray(a[0]).tobytes() == np.asarray(b[0]).tobytes()
        assert np.asarray(a[1][0]).tobytes() == np.asarray(b[1][0]).tobytes()


def test_the_adaptive_mixer_keeps_real_space_and_says_so():
    assert resolve_mixing_space(None, get_mixer("adaptive")) == "r"
    with pytest.raises(ValueError, match="sign"):
        resolve_mixing_space("g", get_mixer("adaptive"))
    with pytest.raises(ValueError, match="not a place"):
        resolve_mixing_space("fourier", get_mixer("anderson"))
    assert resolve_mixing_space("reciprocal", get_mixer("anderson")) == "g"
    assert resolve_mixing_space("real", get_mixer("anderson")) == "r"


# --- whole runs, on the smallest dual-8 cell ---------------------------------


def _calculator(pseudo_dir):
    from defumat import Calculator

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Calculator.from_file(BENCHMARKS / "si2-us-1k.in", pseudo_dir=pseudo_dir,
                                    announce=False)


def test_the_shell_of_an_output_density_is_its_symmetrised_augmentation_charge(
        pseudo_dir, monkeypatch):
    """What the rebuild rests on: above ``ngms``, ``rho_out`` is ``sym_rho(addusdens(becsum_out))``.

    The smooth density reaches the dense grid through a zero-padded extension,
    so nothing but the augmentation charge can be in the shell. Read off the
    first two output densities of a real-space run, the shell agrees with the
    rebuild to round-off; and the symmetrisation is part of it, since an
    ultrasoft ``becsum`` summed over a k-wedge is not symmetrised, and on this
    cell ``Q_ij(G) becsum_out`` alone is far off the shell.
    """
    import defumat.scf.driver as driver

    calculator = _calculator(pseudo_dir)
    calculation = calculator.calculation
    captured = []
    original = driver._mix

    def capture(mixer, rho, rho_out, becsum_in, becsum_out, *args, **kwargs):
        captured.append((np.asarray(rho_out), becsum_out))
        return original(mixer, rho, rho_out, becsum_in, becsum_out, *args, **kwargs)

    monkeypatch.setattr(driver, "_mix", capture)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        driver.run_scf(calculator.system, calculator.pseudos, calculation=calculation,
                       max_iterations=2, mixing_space="r", verbose=False)
    assert len(captured) == 2
    basis = calculation.basis
    rebuild = driver._augmentation_of(calculation)
    for rho_out, becsum_out in captured:
        layout = SphereLayout(basis.dense, basis.ngms, calculation.system.cell,
                              rho_out.shape)
        zero = jnp.zeros(rho_out.shape, dtype=rho_out.dtype)
        shell = _shell(layout, rho_out)
        scale = np.max(np.abs(shell))
        assert scale > 0.0
        rebuilt = _shell(layout, np.asarray(rebuild(zero, becsum_out)))
        bare = _shell(layout, np.asarray(calculation.augmented(zero, becsum_out)))
        assert np.max(np.abs(rebuilt - shell)) < 1e-10 * scale
        assert np.max(np.abs(bare - shell)) > 1e-3 * scale


@pytest.mark.parametrize("preconditioned", [False, True])
def test_the_rebuilt_shell_is_the_anderson_step_on_every_entrys_shell(pseudo_dir,
                                                                      preconditioned):
    """The mixed density's shell is the history's combination of shells, nothing stored.

    Every pair is a random field on the smooth sphere plus the symmetrised
    augmentation charge of a random ``becsum``, which is what an output density
    is above ``ngms`` (the test above). The Anderson coefficients are read back
    off the ``becsum`` block, which the mixer steps at the plain ``beta`` with
    or without Kerker, by least squares; the shell of the mixed density must
    then be ``sum_i c_i ((1 - beta) s_in_i + beta s_out_i)`` over the shells of
    the fields actually handed in. A shell mixed linearly at ``beta`` is the
    newest pair's step alone and fails from the second call on.
    """
    from defumat.scf.driver import _augmentation_of, _mix

    calculation = _calculator(pseudo_dir).calculation
    basis, cell = calculation.basis, calculation.system.cell
    shape = (calculation.nspin_mag,) + tuple(basis.dense.grid)
    layout = SphereLayout(basis.dense, basis.ngms, cell, shape)
    assert layout.nshell > 0
    layout.augmentation = _augmentation_of(calculation)
    beta = 0.3
    mixer = get_mixer("anderson", beta=beta)
    mixer.layout, mixer.shape = layout, layout.stored_shape
    if preconditioned:
        mixer.precondition = kerker_preconditioner_g(layout, cell, beta=beta,
                                                     nelec=calculation.nelec)

    template = calculation.starting_becsum()
    rng = np.random.default_rng(60 + int(preconditioned))

    def random_becsum():
        drawn = []
        for block in template:
            if block is None:
                drawn.append(None)
                continue
            m = rng.normal(size=np.shape(block))
            drawn.append(0.5 * (m + np.swapaxes(m, -1, -2)))
        return tuple(drawn)

    def flat(becsum):
        return np.concatenate([np.ravel(b) for b in becsum if b is not None])

    def density(becsum):
        smooth = _field(layout, rng, shell=False)
        zero = jnp.zeros(shape, dtype=smooth.dtype)
        return smooth + np.asarray(layout.augmentation(zero, becsum))

    entries = []
    for call in range(4):
        becsum_in, becsum_out = random_becsum(), random_becsum()
        rho_in, rho_out = density(becsum_in), density(becsum_out)
        mixed, becsum_mixed, _, _ = _mix(mixer, rho_in, rho_out, becsum_in, becsum_out)
        entries.append(((1 - beta) * _shell(layout, rho_in) + beta * _shell(layout, rho_out),
                        (1 - beta) * flat(becsum_in) + beta * flat(becsum_out)))
        columns = np.stack([b for _, b in entries], axis=1)
        coefficients = np.linalg.lstsq(columns, flat(becsum_mixed), rcond=None)[0]
        np.testing.assert_allclose(columns @ coefficients, flat(becsum_mixed),
                                   rtol=0, atol=1e-12)
        assert coefficients.sum() == pytest.approx(1.0, abs=1e-10)
        if call:
            # A genuine combination, not the newest pair alone.
            assert np.max(np.abs(coefficients[:-1])) > 1e-3
        expected = sum(c * s for c, (s, _) in zip(coefficients, entries))
        coefficients_mixed = np.asarray(layout.forward(np.asarray(mixed)))
        got = coefficients_mixed[:, layout.nsmooth:]
        # Round-off is the whole field's (about 1e-18 here), and the shell is
        # 1e-7, so the bound sits five orders below what it discriminates.
        np.testing.assert_allclose(got, expected, rtol=0,
                                   atol=1e-12 * np.max(np.abs(coefficients_mixed)))
        assert np.max(np.abs(expected)) > 1e5 * 1e-12 * np.max(np.abs(coefficients_mixed))


def test_a_dual_eight_run_reaches_the_real_space_energy(pseudo_dir):
    """Same fixed point, different path: the energies agree to the run's ``conv_thr``.

    ``si2-us-1k.in`` is ultrasoft at ``ecutrho = 8 ecutwfc``, so the shell is
    the augmentation charge's and is mixed linearly where the old layout fitted
    it. On its eight-atom sibling ``benchmarks/si8-us-1k.in`` the two layouts
    took 9 iterations each and printed the same energy to every digit.
    """
    from defumat.scf.driver import run_scf

    calculator = _calculator(pseudo_dir)
    results = {}
    for space in ("r", "g"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results[space] = run_scf(calculator.system, calculator.pseudos,
                                     conv_thr=1e-10, mixing_space=space, verbose=False)
    assert results["r"].converged and results["g"].converged
    assert abs(results["g"].iterations - results["r"].iterations) <= 1
    assert results["g"].total_energy == pytest.approx(results["r"].total_energy,
                                                      abs=1e-9)


def test_a_magnetic_ultrasoft_run_takes_the_real_space_count(pseudo_dir):
    """Noncollinear ultrasoft iron at its own ``beta = 0.3``: the two layouts, one count.

    With the shell mixed linearly the G layout took 17 iterations here against
    the real-space 11, the tail being the shell's magnetization falling by
    ``(1 - beta)^2`` an iteration under a stopping test that reads the dense
    set. Rebuilt from the mixed ``becsum`` it takes 11, and the energies agree
    to 3e-11 Ry at the input's ``conv_thr = 1e-8``.
    """
    from defumat import Calculator

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_file(BENCHMARKS / "fe-mag-1k.in",
                                          pseudo_dir=pseudo_dir, announce=False)
        results = {space: calculator.get_scf(mixing_space=space, verbose=False)
                   for space in ("r", "g")}
    assert results["r"].converged and results["g"].converged
    assert results["g"].iterations <= results["r"].iterations + 1
    assert results["g"].total_energy == pytest.approx(results["r"].total_energy,
                                                      abs=1e-9)


def test_a_resume_carries_the_g_history_and_drops_a_real_space_one(pseudo_dir, tmp_path,
                                                                   dual8):
    """A G checkpoint resumes on the same count; a real-space one is dropped, said so.

    The count is the assertion for the first half, as in
    ``test_an_interrupted_scf_costs_the_same_as_an_uninterrupted_one``, and the
    settings are the ones where it can fail: with the history dropped at the
    resume (the tag check forced to mismatch) this resume took 17 iterations
    against the whole run's 19, where at ``conv_thr = 1e-10``, ``beta = 0.3``
    and a stop at 4 both took 10 and the check could not have seen it.
    """
    from defumat.scf.checkpoint import load_mixer
    from defumat.scf.driver import SCF_MIXER, run_scf

    calculator = _calculator(pseudo_dir)
    options = dict(conv_thr=1e-12, mixing_beta=0.2, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        whole = run_scf(calculator.system, calculator.pseudos, mixing_space="g",
                        **options)
        stopped = run_scf(calculator.system, calculator.pseudos, mixing_space="g",
                          max_iterations=5, checkpoint_dir=tmp_path / "g",
                          checkpoint_every=1, **options)
    assert not stopped.converged
    saved = load_mixer(get_mixer("anderson"), tmp_path / "g" / SCF_MIXER)
    assert saved._history_space == "g"
    layout = _layout(dual8, 1)
    # The density block, ``ngms`` reals, then ``becsum``; nowhere near the box.
    assert layout.ngms < saved._densities[-1].size < layout.points
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resumed = run_scf(calculator.system, calculator.pseudos, mixing_space="g",
                          checkpoint_dir=tmp_path / "g", checkpoint_every=1, **options)
    assert resumed.converged
    assert resumed.iterations == whole.iterations
    assert resumed.total_energy == pytest.approx(whole.total_energy, abs=1e-10)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_scf(calculator.system, calculator.pseudos, mixing_space="r",
                max_iterations=4, checkpoint_dir=tmp_path / "r", checkpoint_every=1,
                **options)
    with pytest.warns(RuntimeWarning, match="keeps the density in real space"):
        crossed = run_scf(calculator.system, calculator.pseudos, mixing_space="g",
                          checkpoint_dir=tmp_path / "r", checkpoint_every=1, **options)
    assert crossed.converged
    assert crossed.total_energy == pytest.approx(whole.total_energy, abs=1e-9)


def test_the_rho_ddot_fit_runs_on_the_stored_sphere(pseudo_dir, monkeypatch):
    """``RHO_DDOT_FIT`` under the layout: fit vectors over ``ngm0``, same fixed point.

    The flag is off by default (``PLAN.md`` P113), so nothing else runs this
    plumbing: ``_mix`` hands the stored residual to the metric, whose density
    part is :meth:`SphereLayout.rho_ddot_vector`. One channel here, so a fit
    vector is the charge alone, ``ngms`` reals.
    """
    import defumat.scf.driver as driver

    calculator = _calculator(pseudo_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = driver.run_scf(calculator.system, calculator.pseudos, conv_thr=1e-10,
                               mixing_space="g", verbose=False)
        monkeypatch.setattr(driver, "RHO_DDOT_FIT", True)
        fitted = driver.run_scf(calculator.system, calculator.pseudos, conv_thr=1e-10,
                                mixing_space="g", verbose=False)
    assert fitted.converged
    assert fitted.total_energy == pytest.approx(plain.total_energy, abs=1e-9)

    calculation = calculator.calculation
    layout = SphereLayout(calculation.basis.dense, calculation.basis.ngms,
                          calculation.system.cell, (1,) + calculation.basis.dense.grid)
    mixer = get_mixer("anderson", beta=0.4)
    mixer.layout, mixer.shape = layout, layout.stored_shape
    mixer.metric = driver._rho_ddot_metric(calculation, layout)
    rng = np.random.default_rng(40)
    rho_in, rho_out = _field(layout, rng), _field(layout, rng)
    driver._mix(mixer, rho_in, rho_out, (), ())
    (fit,) = mixer._fits
    assert fit.size == layout.ngms
    smooth = np.asarray(layout.field(layout.stored_of(rho_out - rho_in)))
    expected = float(scf_accuracy(jnp.asarray(smooth), calculation.basis.dense,
                                  calculation.system.cell))
    assert float(fit @ fit) == pytest.approx(expected, rel=1e-11)
