"""``mixing_mode = 'local-TF'``: QE's ``approx_screening2`` as one compiled loop.

What is pinned here: the solve is the Fortran's least-squares Krylov method and
nothing else, against the literal transcription it replaced and against the
closed form a uniform density admits; the singular-system guard fires rather
than handing the SCF a NaN; and above dual 4 the solve runs on the smooth sphere
and grid, as ``pw.x``'s does, with the two mixer layouts solving the same system
there and the shell above the smooth sphere passing through at ``beta``.
"""

import contextlib
import logging
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat.basis.gvectors import GVectors
from defumat.scf.mixing import (
    LOCAL_TF_EPS,
    LOCAL_TF_MMX,
    LOCAL_TF_REFRESHES,
    SphereLayout,
    _approx_screening2,
    _field_of,
    _screening_sphere,
    local_tf_preconditioner,
    local_tf_preconditioner_g,
)

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"


def _pieces(name):
    """The basis and cell of an input, built without an SCF."""
    from defumat.basis.builder import build_basis
    from defumat.io.pwin import read_pw_input
    from defumat.system import build_system

    system = build_system(read_pw_input(BENCHMARKS / name))
    return build_basis(system), system.cell


@pytest.fixture(scope="module")
def dual4():
    """Norm-conserving silicon at dual 4: the smooth set is the dense one."""
    return _pieces("si-1k.in")


@pytest.fixture(scope="module")
def dual8():
    """Ultrasoft silicon at ``ecutrho = 8 ecutwfc``: a smooth grid and a shell."""
    return _pieces("si2-us-1k.in")


def _band_limited(gvectors, rng, scale=1.0):
    """A real field on the grid whose coefficients lie on ``gvectors``' sphere."""
    grid = tuple(gvectors.grid)
    noise = rng.normal(size=grid)
    box = np.fft.fftn(noise).ravel()
    keep = np.zeros(box.size, dtype=bool)
    keep[np.asarray(gvectors.fft_index)] = True
    box[~keep] = 0.0
    return scale * np.real(np.fft.ifftn(box.reshape(grid))).ravel()


def _slab_density(gvectors, cell):
    """A metal sheet in vacuum, 0.1 at its centre and 1e-6 at the far edge.

    ``alpha`` then spans three orders of magnitude, which is what makes the
    Krylov solve run past ``mmx`` and restart.
    """
    grid = tuple(gvectors.grid)
    z = (np.arange(grid[2]) / grid[2])[None, None, :]
    profile = 1e-6 + 0.1 * np.exp(-(((z - 0.5) / 0.08) ** 2))
    field = np.broadcast_to(profile, grid).ravel()
    coefficients = np.fft.fftn(field.reshape(grid)).ravel()
    keep = np.zeros(coefficients.size, dtype=bool)
    keep[np.asarray(gvectors.fft_index)] = True
    coefficients[~keep] = 0.0
    return np.real(np.fft.ifftn(coefficients.reshape(grid))).ravel()


def _transcription(gvectors, cell, residual, density, mmx=LOCAL_TF_MMX,
                   refreshes=LOCAL_TF_REFRESHES):
    """``mix_rho.f90:628-1024`` line by line, on the whole sphere with complex FFTs.

    This is the Python loop the compiled solver replaced, kept as the reference
    it has to reproduce. Returns the screened field and the steps it took.
    """
    grid = tuple(gvectors.grid)
    points = int(np.prod(grid))
    index = np.asarray(gvectors.fft_index)
    g2 = np.asarray(gvectors.kinetic(cell))
    volume = float(cell.volume)
    fpi_e2 = 8.0 * np.pi
    nonzero = g2 > 1e-12
    weight = np.where(nonzero, 1.0 / np.where(nonzero, g2, 1.0), 0.0)

    def to_sphere(field):
        return np.fft.fftn(field.reshape(grid)).ravel()[index] / points

    def to_grid(c):
        box = np.zeros(points, dtype=complex)
        box[index] = c
        return np.real(np.fft.ifftn(box.reshape(grid))).ravel() * points

    rho = np.abs(to_grid(to_sphere(density)))
    dense = rho > LOCAL_TF_EPS
    radius = np.zeros_like(rho)
    radius[dense] = (3.0 / (4.0 * np.pi * rho[dense])) ** (1.0 / 3.0)
    agg0 = (12.0 / np.pi) ** (2.0 / 3.0) / (points / np.sum(1.0 / radius[dense]))
    alpha = 3.0 * (2.0 * np.pi / 3.0) ** (5.0 / 3.0) * radius

    def operator(v):
        return fpi_e2 * v + g2 * to_sphere(alpha * to_grid(v))

    def dot(a, b):
        return fpi_e2 * 0.5 * volume * np.sum(weight * np.real(np.conj(a) * b))

    drho = np.where(nonzero, to_sphere(residual), 0.0)
    dv = np.where(nonzero, g2 * to_sphere(alpha * to_grid(drho)), 0.0)
    directions, applied, aa, bb = [dv / (g2 + agg0)], [], np.zeros((0, 0)), []
    target, best, restarts, steps = 0.0, None, 0, 0
    while True:
        applied.append(operator(directions[-1]))
        steps += 1
        m = len(applied)
        aa = np.pad(aa, ((0, 1), (0, 1)))
        for i in range(m):
            aa[i, m - 1] = aa[m - 1, i] = dot(applied[i], applied[m - 1])
        bb.append(dot(applied[m - 1], dv))
        vec = np.linalg.solve(aa, np.asarray(bb))
        best = sum(c * v for c, v in zip(vec, directions))
        residue = dv - sum(c * w for c, w in zip(vec, applied))
        error = dot(residue, residue)
        if target == 0.0:
            target = max(1.0e-12, 1.0e-6 * error)
        if error < target:
            break
        if m >= mmx:
            if restarts >= refreshes:
                break
            restarts += 1
            directions, applied, aa, bb = [best], [], np.zeros((0, 0)), []
            continue
        directions.append(residue / (g2 + agg0))
    return to_grid(np.where(nonzero, best, 0.0)), steps


def _solve(gvectors, cell, residual, density, mmx=LOCAL_TF_MMX,
           refreshes=LOCAL_TF_REFRESHES):
    """The compiled solver on one charge: ``(field, steps)``."""
    rows, sphere = _screening_sphere(gvectors, cell)
    grid = tuple(int(n) for n in gvectors.grid)

    def half(field):
        box = np.fft.rfftn(np.asarray(field).reshape(grid)).ravel()
        return jnp.asarray(box[np.asarray(sphere.positions)] / np.prod(grid))

    screened, steps, _ = _approx_screening2(half(residual), half(density), sphere,
                                            float(cell.volume), grid=grid, mmx=mmx,
                                            refreshes=refreshes)
    return np.asarray(_field_of(screened, sphere, grid)), int(steps)


def test_a_uniform_density_gives_the_closed_form(dual4):
    """Constant ``alpha`` makes the operator diagonal: ``v = G^2 a drho / (8 pi + G^2 a)``.

    ``agg0`` is built so that ``(8 pi + G^2 alpha) / (G^2 + agg0)`` is within
    0.2 per cent of the constant ``alpha``, so the preconditioned system is
    nearly the identity and the solve stops in a step or two; what is checked
    is that it stops at the right answer, which pins the operator, the
    transforms' normalisation and the metric together.
    """
    basis, cell = dual4
    gvectors = basis.dense
    rng = np.random.default_rng(1)
    residual = _band_limited(gvectors, rng, 1e-3)
    density = np.full(residual.size, 0.05)
    got, steps = _solve(gvectors, cell, residual, density)

    grid = tuple(gvectors.grid)
    index = np.asarray(gvectors.fft_index)
    g2 = np.asarray(gvectors.kinetic(cell))
    alpha = 3.0 * (2.0 * np.pi / 3.0) ** (5.0 / 3.0) * (3.0 / (4.0 * np.pi * 0.05)) ** (1 / 3)
    box = np.fft.fftn(residual.reshape(grid)).ravel()
    factor = np.zeros(box.size)
    factor[index] = g2 * alpha / (8.0 * np.pi + g2 * alpha)
    expected = np.real(np.fft.ifftn((factor * box).reshape(grid))).ravel()
    assert steps <= 3
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6 * np.max(np.abs(expected)))


@pytest.mark.parametrize("mmx, refreshes", [(LOCAL_TF_MMX, LOCAL_TF_REFRESHES), (4, 4),
                                             (3, 1)], ids=["qe", "restarts", "gives-up"])
def test_the_compiled_loop_is_the_transcription(dual4, mmx, refreshes):
    """The Python loop it replaced, step for step, through a restart and the cap.

    On this cell QE's width converges in 11 steps without a restart, so the
    restart and the give-up branches are reached by narrowing the Krylov space:
    four directions restart three times and converge at step 16, and three with
    one restart stop at the cap of six steps, short of the tolerance.
    """
    basis, cell = dual4
    gvectors = basis.dense
    rng = np.random.default_rng(2)
    residual = _band_limited(gvectors, rng, 1e-3)
    density = _slab_density(gvectors, cell)
    expected, reference_steps = _transcription(gvectors, cell, residual, density,
                                               mmx, refreshes)
    got, steps = _solve(gvectors, cell, residual, density, mmx, refreshes)
    if mmx < LOCAL_TF_MMX:
        assert reference_steps > mmx, "the solve did not restart"
    if refreshes == 1:
        assert reference_steps == mmx * (refreshes + 1), "the solve did not hit its cap"
    assert steps == reference_steps
    np.testing.assert_allclose(got, expected, rtol=0,
                               atol=1e-10 * np.max(np.abs(expected)))


def test_a_singular_system_keeps_the_last_estimate(dual4):
    """A zero residual makes the first Gram matrix ``[[0]]``: the guard must fire.

    Without it the loop would carry ``0/0`` into every later step and hand the
    SCF a NaN step; with it the loop stops after one application and returns
    the first direction, which is zero.
    """
    basis, cell = dual4
    gvectors = basis.dense
    got, steps = _solve(gvectors, cell, np.zeros(int(np.prod(gvectors.grid))),
                        _slab_density(gvectors, cell))
    assert steps == 1
    assert np.all(np.isfinite(got)) and not np.any(got)


def test_above_dual_4_both_layouts_solve_on_the_smooth_grid(dual8):
    """The real-space and the G layout give one step, the shell at plain ``beta``.

    ``pw.x``'s ``approx_screening2`` acts on ``of_g(:ngms)`` on ``dffts`` and
    leaves the shell to ``high_frequency_mixing``. The real-space layout's
    residual has a shell and the G layout's has none, so the two are compared on
    the smooth sphere, and the real-space layout's shell is checked to come out
    as ``beta`` times what went in.
    """
    basis, cell = dual8
    assert basis.smooth.grid != basis.dense.grid
    layout = SphereLayout(basis.dense, basis.ngms, cell, (2,) + basis.dense.grid)
    rng = np.random.default_rng(3)
    residual = np.stack([_band_limited(basis.dense, rng, 1e-3)
                         for _ in range(2)]).reshape(layout.shape)
    smooth_on_dense = GVectors(miller=basis.dense.miller[:basis.ngms],
                               grid=basis.dense.grid, ecut=basis.dense.ecut,
                               gamma_only=basis.dense.gamma_only)
    density = np.stack([0.5 * _slab_density(smooth_on_dense, cell)] * 2)
    density = density.reshape(layout.shape)
    shape = layout.shape
    real = local_tf_preconditioner(basis.dense, cell, shape, beta=0.6, smooth=basis.smooth)
    sphere = local_tf_preconditioner_g(layout, basis.smooth, cell, beta=0.6)

    out = np.asarray(real(residual.ravel(), density.ravel())).reshape(shape)
    stored = sphere(layout.stored_of(residual).ravel(), layout.stored_of(density).ravel())
    smooth_part = layout.stored_of(out).ravel()
    np.testing.assert_allclose(smooth_part, stored, rtol=0,
                               atol=1e-9 * np.max(np.abs(stored)))

    shell_in = np.asarray(layout.forward(residual))[:, layout.nsmooth:]
    shell_out = np.asarray(layout.forward(out))[:, layout.nsmooth:]
    assert np.max(np.abs(shell_in)) > 0
    np.testing.assert_allclose(shell_out, 0.6 * shell_in, rtol=0,
                               atol=1e-12 * np.max(np.abs(shell_in)))


@contextlib.contextmanager
def _counting_compiles():
    """Every XLA compilation inside the block, read off the ``jax`` logger (``test_eager.py``)."""
    names = []

    class Grab(logging.Handler):
        def emit(self, record):
            found = re.search(r"Finished XLA compilation of (.+?) in", record.getMessage())
            if found:
                names.append(found.group(1))

    handler, logger = Grab(), logging.getLogger("jax")
    before = jax.config.jax_log_compiles
    jax.config.update("jax_log_compiles", True)
    logger.addHandler(handler)
    try:
        yield names
    finally:
        logger.removeHandler(handler)
        jax.config.update("jax_log_compiles", before)


def test_a_second_preconditioner_and_a_second_call_compile_nothing(dual8):
    """One program per grid and shape, shared by every instance and every call.

    The SCF builds a preconditioner once a run and a relaxation once a step, so
    a program keyed on the instance would compile again at every geometry. The
    counter is first shown to see the cold call, so that an empty second count
    is a measurement and not a silent logger.
    """
    basis, cell = dual8
    layout = SphereLayout(basis.dense, basis.ngms, cell, (2,) + basis.dense.grid)
    rng = np.random.default_rng(4)
    residual = np.stack([_band_limited(basis.dense, rng, 1e-3) for _ in range(2)])
    smooth_on_dense = GVectors(miller=basis.dense.miller[:basis.ngms],
                               grid=basis.dense.grid, ecut=basis.dense.ecut,
                               gamma_only=basis.dense.gamma_only)
    density = np.stack([0.5 * _slab_density(smooth_on_dense, cell)] * 2)
    stored_residual = layout.stored_of(residual.reshape(layout.shape)).ravel()
    stored_density = layout.stored_of(density.reshape(layout.shape)).ravel()

    jax.clear_caches()
    with _counting_compiles() as cold:
        local_tf_preconditioner(basis.dense, cell, layout.shape, beta=0.7,
                                smooth=basis.smooth)(residual.ravel(), density.ravel())
        local_tf_preconditioner_g(layout, basis.smooth, cell, beta=0.7)(
            stored_residual, stored_density)
    assert cold, "the counter saw no compilation on a cold call"

    with _counting_compiles() as warm:
        local_tf_preconditioner(basis.dense, cell, layout.shape, beta=0.4,
                                smooth=basis.smooth)(2.0 * residual.ravel(), density.ravel())
        local_tf_preconditioner_g(layout, basis.smooth, cell, beta=0.4)(
            2.0 * stored_residual, stored_density)
    assert not warm, f"a second instance compiled {warm}"
