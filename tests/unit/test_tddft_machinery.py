"""P37's cheap pieces: the response sphere, the kernel registry and the refusals.

Everything here runs off a bare :class:`~defumat.scf.driver.Calculation` --
no SCF, no states -- which is the point: a refusal is a statement about the
calculation and must be reachable before anything expensive has been paid for.
The identities that need a converged ground state are in
``tests/regression/test_tddft.py``.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.scf.driver import Calculation
from defumat.system import build_system
from defumat.tddft import (
    get_kernel,
    kernel_names,
    require_a_sum_over_states_regime,
    response_sphere,
)
from defumat.units import E2, FPI

pytestmark = pytest.mark.unit

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


def _calculation(case: str) -> Calculation:
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    return Calculation(system, pseudos)


# --- the response sphere -----------------------------------------------------

def test_the_response_sphere_excludes_the_origin_and_is_closed_under_inversion():
    """Both properties are load-bearing and neither is obvious from the code.

    ``G = 0`` is not a body entry because in the optical limit it is the *head*,
    a 3x3 block of directions rather than one number; leaving it in would give a
    matrix with two entries meaning the same thing and a divergent Coulomb
    factor on one of them.

    Closure under ``G -> -G`` is what makes the antiresonant half of the pair
    sum free: ``<u_j|e^{-iG.r}|u_i>`` is the reflection of
    ``<u_i|e^{-iG.r}|u_j>``, so the reversed pair costs a gather rather than a
    transform. A sphere sorted by ``|G|^2`` always has it, and asserting it
    keeps a future change to the selection honest.
    """
    calculation = _calculation("si-epsilon-unshifted-nosym")
    sphere = response_sphere(calculation, 8.0)

    assert sphere.nbody > 0
    assert sphere.nm == sphere.nbody + 3

    miller = np.asarray(sphere.miller)
    assert not np.any(np.all(miller == 0, axis=1))  # no G = 0 in the body

    reflection = np.asarray(sphere.reflection)
    assert np.array_equal(reflection[reflection], np.arange(sphere.nbody))
    assert not np.any(reflection == np.arange(sphere.nbody))


def test_the_coulomb_factor_is_the_rydberg_one():
    """``sqrt(8 pi / |G|^2)``, not ``sqrt(4 pi / |G|^2)``.

    ``e^2 = 2`` in Rydberg atomic units, which is exactly the factor
    :func:`~defumat.scf.potential.hartree` carries and the classic place to
    lose a two. The symmetrised ``chi_0`` has it on both sides, so a wrong
    constant here is a factor of two on every dielectric function.
    """
    calculation = _calculation("si-epsilon-unshifted-nosym")
    sphere = response_sphere(calculation, 4.0)
    g2 = np.asarray(calculation.basis.smooth.kinetic(calculation.system.cell))
    lookup = {tuple(m): n
              for n, m in enumerate(np.asarray(calculation.basis.smooth.miller))}
    order = np.array([lookup[tuple(m)] for m in np.asarray(sphere.miller)])
    assert np.allclose(
        np.asarray(sphere.sqrt_coulomb), np.sqrt(E2 * FPI / g2[order])
    )


def test_a_cutoff_with_no_body_is_the_head_only_kernel():
    """``ecut = 0`` is a named approximation, not a degenerate case.

    It leaves the 3x3 head alone -- the head-only kernel of the long-range
    correction literature, and what Elk's own ``LiF-bootstrap`` example asks for
    with ``gmaxrf = 0.0``. So it is supported rather than refused, and what it
    means is stated: no body means no local-field effect, so ``eps_M`` is
    ``1 - X_head``. A *negative* cutoff is still an error, because it is not a
    request for anything.
    """
    calculation = _calculation("si-epsilon-unshifted-nosym")
    head_only = response_sphere(calculation, 0.0)
    assert head_only.nbody == 0
    assert head_only.nm == 3
    with pytest.raises(ValueError, match="cannot be negative"):
        response_sphere(calculation, -1.0)


# --- the kernel registry -----------------------------------------------------

def test_every_kernel_is_registered_under_its_name():
    assert set(kernel_names()) == {"rpa", "alda", "lrc", "bootstrap", "bootstrap-1"}
    assert get_kernel("bootstrap").self_consistent
    assert not get_kernel("bootstrap-1").self_consistent
    # Elk's 211 rounds its loop **twice**: it increments after the first Dyson
    # solve and repeats while ``it <= 1``, so the kernel it ends on was built
    # from the first pass's answer rather than from the seed.
    assert get_kernel("bootstrap-1").iterations == 2
    assert not get_kernel("rpa").self_consistent
    with pytest.raises(ValueError, match="unknown exchange-correlation kernel"):
        get_kernel("nanoquanta")


def test_the_lrc_kernel_is_a_constant_on_the_diagonal_and_needs_its_parameter():
    """``F = -alpha / 4 pi``, head included -- and ``alpha`` has no default.

    The parameter is material-dependent and empirical, which is the entire
    reason the bootstrap kernel was proposed. Supplying a default would hide
    that behind a number that is right for nothing.
    """
    from defumat.tddft.chi0 import ChiZero, ResponseSphere

    sphere = ResponseSphere(
        fft_index=jnp.arange(2), sqrt_coulomb=jnp.ones(2),
        reflection=jnp.asarray([1, 0]),
        miller=jnp.asarray([[1, 0, 0], [-1, 0, 0]]), ecut=1.0,
    )
    chi = ChiZero(x=jnp.zeros((2, 5, 5), dtype=complex), frequencies=jnp.zeros(2),
                  sphere=sphere, npairs=1, nocc=1, nbnd=2)

    with pytest.raises(ValueError, match="needs its parameter"):
        get_kernel("lrc").build(chi, None, {})

    kernel = np.asarray(get_kernel("lrc").build(chi, None, {"alpha": 0.2}))
    assert kernel.shape == (2, 5, 5)
    assert np.allclose(np.diagonal(kernel, axis1=1, axis2=2), -0.2 / FPI)
    assert np.allclose(kernel - np.eye(5) * kernel[0, 0, 0], 0.0)


# --- the refusals ------------------------------------------------------------

def test_a_reduced_k_set_is_refused_before_anything_is_computed():
    """``chi_0(G, G')`` on a wedge would need a rotation in two G indices.

    P36's rank-N symmetriser is Cartesian and does not do it, so the whole grid
    is required. ``si-epsilon.in`` is the shifted wedge the Sternheimer response
    runs on and is exactly what must be refused here.
    """
    with pytest.raises(NotImplementedError, match="full k-grid"):
        require_a_sum_over_states_regime(_calculation("si-epsilon"))


def test_an_ultrasoft_dataset_is_refused_by_name():
    """The plane-wave matrix element gains ``Q_ij(G)`` and nothing here adds it."""
    with pytest.raises(NotImplementedError, match="ultrasoft or PAW"):
        require_a_sum_over_states_regime(_calculation("si-epsilon-us"))


def test_the_whole_grid_norm_conserving_insulator_is_accepted():
    """The complement of the refusals: the one regime that is supported."""
    require_a_sum_over_states_regime(_calculation("si-epsilon-unshifted-nosym"))


# --- the pair axis's working set ---------------------------------------------

def test_the_pair_densities_are_bounded_by_the_pair_dial():
    """One pair density is a whole FFT box, and there are ``npairs`` of them.

    The transforms are the *time* of this phase and the pair densities are its
    **memory**: ``<u_i|e^{-iG.r}|u_j>`` is formed on the smooth grid, so every
    occupied-empty pair costs one complex box in flight while ``nm`` numbers
    are kept from it. Forming them all at once is quadratic in the band count
    and reaches tens of gigabytes on a cell of production size, where the
    states they are built from are hundreds of megabytes.

    Two assertions, and they are the two halves of a dial: the compiler's own
    temporary budget falls with the chunk, and the vectors it returns do not
    move. Every pair goes through the same transform whatever the chunk, so
    this is ``map_bands``'s "identical rather than round-off" case and the
    bound is tight.

    ``memory_analysis()`` runs the compiler and allocates nothing, which is
    what makes this a unit test rather than a run.
    """
    import jax

    from defumat.tddft.chi0 import _pair_terms, _pairs, response_sphere

    calculation = _calculation("si-epsilon-unshifted-nosym")
    sphere = response_sphere(calculation, 8.0)
    precision = calculation.system.cell.precision
    grid = calculation.basis.smooth.grid
    fft_index = jnp.asarray(calculation.fft_index)[0]
    band_mask = jnp.asarray(calculation.basis.planewaves.mask)[0]
    npwx = int(band_mask.shape[0])

    nocc, nbnd = 4, 12
    rows, columns = _pairs(nocc, nbnd)
    rng = np.random.default_rng(0)
    psi = jnp.asarray(rng.normal(size=(nbnd, npwx))
                      + 1j * rng.normal(size=(nbnd, npwx)), precision.complex)
    element = jnp.asarray(rng.normal(size=(3, nbnd, nbnd))
                          + 1j * rng.normal(size=(3, nbnd, nbnd)),
                          precision.complex)
    eig = jnp.asarray(np.sort(rng.normal(size=nbnd)), precision.real)
    occupation = jnp.asarray(
        np.where(np.arange(nbnd) < nocc, 2.0, 0.0), precision.real)
    zomega = jnp.asarray([0.0 + 0.01j], precision.complex)

    def measure(pair_batch):
        def f(psi, element, eig, occupation):
            return _pair_terms(
                psi, fft_index, band_mask, eig, occupation, element,
                rows, columns, sphere, grid, 270.0, zomega, 0.0, precision,
                pair_batch,
            )

        compiled = jax.jit(f).lower(psi, element, eig, occupation).compile()
        temporary = compiled.memory_analysis().temp_size_in_bytes
        return compiled(psi, element, eig, occupation), int(temporary)

    (whole, _), big = measure(None)
    (chunked, _), small = measure(1)

    box = int(np.prod(grid)) * 16
    npairs = int(rows.size)
    assert npairs == nocc * (nbnd - nocc)
    # **There is a floor and it is not the pairs.** ``fields`` -- the ``nbnd``
    # states in real space -- is transformed whatever the chunk, and its
    # transform's own input and output are about ``2 nbnd`` boxes; that is
    # 3.13 MB of the 3.13 MB left at a chunk of one. What has to fall is the
    # part that scales with ``npairs``, which is the whole difference.
    assert big - small > 0.5 * npairs * box
    assert small < 4 * nbnd * box
    assert np.abs(np.asarray(whole) - np.asarray(chunked)).max() < 1.0e-14


# --- the frequency axis --------------------------------------------------------

def test_the_frequency_dial_bounds_the_assembly_and_does_not_move_it():
    """``chi_0``'s per-k assembly, whole axis against chunks of frequencies.

    ``GPU-MEMORY-NEXT.md`` item 12. The whole-axis ``einsum`` is contracted
    through ``v_pa conj(v_pb)`` whenever ``nw`` exceeds ``nm`` -- a ``(2 npairs,
    nm, nm)`` block -- and a chunk forms only its own frequencies'
    pair-weighted ``(2 npairs, nm)`` blocks. Two assertions, the two halves of
    a dial: the compiler's temporary falls, and the matrix does not move
    beyond the order of a sum. Seven frequencies in chunks of three, so the
    last chunk is short.
    """
    import jax

    from defumat.tddft.chi0 import _assemble

    rng = np.random.default_rng(0)
    nw, npairs2, nm = 7, 48, 20
    scalars = jnp.asarray(rng.normal(size=(nw, npairs2))
                          + 1j * rng.normal(size=(nw, npairs2)))
    vectors = jnp.asarray(rng.normal(size=(npairs2, nm))
                          + 1j * rng.normal(size=(npairs2, nm)))
    whole = np.asarray(_assemble(scalars, vectors, None))
    for batch in (3, 1):
        chunked = np.asarray(_assemble(scalars, vectors, batch))
        assert np.abs(chunked - whole).max() < 1e-12 * np.abs(whole).max()

    def temporary(batch, nw=64, npairs2=96, nm=40):
        s = jax.ShapeDtypeStruct((nw, npairs2), jnp.complex128)
        v = jax.ShapeDtypeStruct((npairs2, nm), jnp.complex128)
        compiled = jax.jit(lambda s, v: _assemble(s, v, batch)).lower(s, v).compile()
        return compiled.memory_analysis().temp_size_in_bytes

    block = 96 * 40 * 40 * 16   # the (2 npairs, nm, nm) contraction, 2.4 MB
    assert temporary(None) > block
    assert temporary(None) - temporary(4) > 0.5 * block


def _synthetic_chi(nw: int = 7, nm: int = 8):
    """A ``ChiZero`` whose static slice is negative definite, as ``X`` is."""
    from defumat.tddft.chi0 import ChiZero, ResponseSphere

    rng = np.random.default_rng(1)
    sphere = ResponseSphere(
        fft_index=jnp.arange(nm - 3), sqrt_coulomb=jnp.ones(nm - 3),
        reflection=jnp.arange(nm - 3), miller=jnp.zeros((nm - 3, 3), int),
        ecut=1.0,
    )
    a = rng.normal(size=(nm, nm)) + 1j * rng.normal(size=(nm, nm))
    static = -(a @ a.conj().T) / nm * 0.8
    omega = np.linspace(0.0, 0.5, nw)
    x = np.stack([static / (1.0 - (w + 0.05j) ** 2) for w in omega])
    return ChiZero(x=jnp.asarray(x), frequencies=jnp.asarray(omega),
                   sphere=sphere, npairs=1, nocc=1, nbnd=2)


def _whole_axis_dyson(chi, kernel, context, static_index):
    """The Dyson solve as it was: every frequency on every pass.

    Kept here as the reference the one-frequency fixed point is held against:
    ``tddftlr.f90``'s loop over the whole axis, ``F`` built at every frequency.
    """
    rule = get_kernel(kernel)
    context = {**context, "static_index": static_index}
    x = chi.x
    identity = jnp.eye(x.shape[-1], dtype=x.dtype)
    eps0 = identity[None] - x
    epsi, previous, iterations = None, None, 0
    passes = 500 if rule.self_consistent else (rule.iterations or 1)
    for iterations in range(1, passes + 1):
        fxc = rule.build(chi, epsi, context)
        fxc_x = fxc @ x
        epsi = x @ jnp.linalg.inv(eps0 - fxc_x) + identity[None]
        if not rule.self_consistent:
            continue
        current = complex(fxc_x[static_index, 0, 0])
        if previous is not None and abs(abs(previous) - abs(current)) <= 1e-8:
            break
        previous = current
    return epsi, fxc[static_index], iterations


@pytest.mark.parametrize("kernel, context", [
    ("rpa", {}), ("lrc", {"alpha": 0.2}), ("bootstrap-1", {}), ("bootstrap", {}),
])
def test_the_dyson_solve_iterates_one_frequency_and_screens_the_rest_in_chunks(
        kernel, context):
    """The fixed point on the static slice alone, then the axis in chunks.

    A static kernel is built from ``eps^-1`` at ``omega = 0`` and nothing else,
    and so is the convergence test, so iterating that one frequency reaches
    the same ``F`` in the same number of passes; the rest of the axis is then
    screened once, ``w_batch`` frequencies at a time. Held against the
    whole-axis loop it replaced, with seven frequencies in chunks of three and
    of one.
    """
    from defumat.tddft.dyson import solve_dyson

    chi = _synthetic_chi()
    epsi, fxc, iterations = _whole_axis_dyson(chi, kernel, context, 1)
    for batch in (None, 3, 1):
        solution = solve_dyson(chi, kernel, context, static_index=1,
                               w_batch=batch)
        assert solution.iterations == iterations
        assert solution.fxc.shape == (1,) + fxc.shape
        np.testing.assert_allclose(np.asarray(solution.fxc[0]), np.asarray(fxc),
                                   atol=1e-12)
        np.testing.assert_allclose(np.asarray(solution.epsilon_inverse),
                                   np.asarray(epsi), atol=1e-12)


def test_the_frequency_dial_chunks_on_a_card_and_not_on_a_cpu(monkeypatch):
    """The accelerator default is a budget, and it fires; the CPU's is the axis."""
    from defumat import batching

    monkeypatch.delenv("DEFUMAT_W_BATCH", raising=False)
    block = 2**20
    monkeypatch.setattr(batching, "_backend", lambda: "cpu")
    assert batching.resolve_w_batch(block_bytes=block, nw=1000) is None
    monkeypatch.setattr(batching, "_backend", lambda: "gpu")
    chunk = batching.resolve_w_batch(block_bytes=block, nw=1000)
    assert chunk == batching.W_BUDGET_BYTES // block < 1000
    assert batching.resolve_w_batch(block_bytes=block, nw=10) is None
    assert batching.resolve_w_batch(8, block_bytes=block, nw=1000) == 8
    assert batching.resolve_w_batch(None, block_bytes=block, nw=1000) is None
    monkeypatch.setenv("DEFUMAT_W_BATCH", "5")
    assert batching.resolve_w_batch(block_bytes=block, nw=1000) == 5
    monkeypatch.setenv("DEFUMAT_W_BATCH", "all")
    assert batching.resolve_w_batch(block_bytes=block, nw=1000) is None
