"""A closure called eagerly is compiled once per structure, and never wrongly reused.

:mod:`defumat.eager` keys a compiled program on the printed jaxpr of a closure
with its arrays hoisted out, so the property that matters is two-sided: calls
that differ only in their arrays share one program, and calls that differ in
anything the printed form does not show (a nested constant, a callback) do not.
Each refusal here is fed a case that must trip it, since a guard that cannot fire
reads exactly like one that passes.

The response-stack counts are measured on the one-k-point silicon cell; the
AlAs and phonon-at-q figures behind the change are in ``PERFORMANCE.md``, "The
response stack compiled its k loops again at every iteration".
"""

import contextlib
import logging
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from defumat import eager
from defumat.calculator import Calculator
from defumat.eager import compiled

pytestmark = pytest.mark.unit

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"


@contextlib.contextmanager
def counting_compiles():
    """Every XLA compilation inside the block, by name, read off the ``jax`` logger."""
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


@pytest.fixture(autouse=True)
def fresh_cache():
    eager.clear()
    yield
    eager.clear()


def _loop(arr):
    return compiled(lambda xs: lax.map(lambda x: x * arr + jnp.sin(x), xs),
                    jnp.arange(4.0))


def test_closures_over_different_arrays_share_one_program():
    a, b = jnp.arange(4.0), jnp.linspace(1.0, 2.0, 4)
    first = _loop(a)
    with counting_compiles() as names:
        second = _loop(b)
    assert names == []
    assert len(eager._PROGRAMS) == 1
    for arr, got in ((a, first), (b, second)):
        expected = lax.map(lambda x: x * arr + jnp.sin(x), jnp.arange(4.0))
        np.testing.assert_array_equal(got, expected)


def test_a_different_structure_is_a_different_program():
    _loop(jnp.arange(4.0))
    compiled(lambda xs: lax.map(lambda x: x + 1.0, xs), jnp.arange(4.0))
    compiled(lambda xs: lax.map(lambda x: x + 1.0 + 2**-50, xs), jnp.arange(4.0))
    assert len(eager._PROGRAMS) == 3


def test_a_nested_constant_is_part_of_the_key():
    """A ``jit`` closing over an array keeps it unprinted inside its own jaxpr."""
    def nested(arr):
        inner = jax.jit(lambda y: y + arr)
        return compiled(lambda xs: lax.map(inner, xs), jnp.arange(4.0))

    a, b = jnp.arange(4.0), jnp.ones(4)
    got_a = nested(a)
    got_b = nested(b)
    assert len(eager._PROGRAMS) == 2
    np.testing.assert_array_equal(got_a, jnp.arange(4.0)[:, None] + a)
    np.testing.assert_array_equal(got_b, jnp.arange(4.0)[:, None] + b)
    nested(a)
    assert len(eager._PROGRAMS) == 2


def test_forget_drops_what_the_next_pass_did_not_use_and_keeps_the_rest():
    """``tracking`` and ``forget``, as an orientation relaxation uses them.

    Each pass reaches one program shared by every pass and one keyed on a
    nested constant that changes between passes (the quantization axis was
    one, in the relaxation, until it became an argument). Dropping what the
    first pass used and the second did not
    must drop the first pass's own program and nothing else: the shared one is
    then reused without a compile, and the dropped one compiles again.
    """
    def one_pass(arr):
        inner = jax.jit(lambda y: y + arr)
        compiled(lambda xs: lax.map(inner, xs), jnp.arange(4.0))
        _loop(jnp.arange(4.0))

    with eager.tracking() as first:
        one_pass(jnp.arange(4.0))
    with eager.tracking() as second:
        one_pass(jnp.ones(4))
    assert len(first) == len(second) == 2 and len(first & second) == 1
    assert eager.forget(first - second) == 1
    assert len(eager._PROGRAMS) == 2
    with counting_compiles() as names:
        _loop(jnp.linspace(1.0, 2.0, 4))
    assert names == []
    with counting_compiles() as names:
        one_pass(jnp.arange(4.0))
    assert len(names) == 1
    assert eager._TRACKERS == []


def test_a_nested_constant_past_the_limit_is_not_kept(monkeypatch):
    monkeypatch.setattr(eager, "NESTED_LIMIT", 8)
    arr = jnp.arange(4.0)
    inner = jax.jit(lambda y: y + arr)
    got = compiled(lambda xs: lax.map(inner, xs), jnp.arange(4.0))
    assert len(eager._PROGRAMS) == 0
    np.testing.assert_array_equal(got, jnp.arange(4.0)[:, None] + arr)


def test_a_callback_is_not_kept():
    def host(x):
        return np.asarray(x) * 2.0

    def body(x):
        return jax.pure_callback(host, jax.ShapeDtypeStruct(x.shape, x.dtype), x)

    got = compiled(lambda xs: lax.map(body, xs), jnp.arange(4.0))
    assert len(eager._PROGRAMS) == 0
    np.testing.assert_array_equal(got, 2.0 * jnp.arange(4.0))


def test_under_a_transformation_it_is_the_plain_call():
    arr = jnp.arange(4.0)
    primal, tangent = jax.jvp(lambda s: _loop(arr * s), (1.0,), (1.0,))
    assert len(eager._PROGRAMS) == 0
    np.testing.assert_allclose(tangent, jnp.arange(4.0)[:, None] * arr)
    np.testing.assert_allclose(jax.jit(lambda s: _loop(arr * s))(2.0),
                               _loop(2.0 * arr))


@pytest.fixture(scope="module")
def silicon(pseudo_dir):
    calculator = Calculator.from_file(BENCHMARKS / "si-1k.in", pseudo_dir,
                                      announce=False)
    calculator.get_scf()
    return calculator


def test_a_second_band_velocity_call_compiles_nothing(silicon):
    first = silicon.get_band_velocities()
    with counting_compiles() as names:
        second = silicon.get_band_velocities()
    assert names == []
    np.testing.assert_array_equal(first.velocities_by_spin, second.velocities_by_spin)


@pytest.mark.slow
def test_a_second_dielectric_tensor_call_compiles_nothing(silicon):
    """Before :mod:`defumat.eager` a second call compiled 127 programs here."""
    first = silicon.get_dielectric_tensor()
    with counting_compiles() as names:
        second = silicon.get_dielectric_tensor()
    assert names == []
    np.testing.assert_allclose(second.epsilon, first.epsilon, rtol=0, atol=1e-12)


def _hubbard_arrays(seed, nspin=2, nk=3, nbnd=4, npwx=7, nwfcU=5):
    """Random states, projectors and weights shaped as ``occupation_matrix`` takes them."""
    rng = np.random.default_rng(seed)

    def complex_normal(*shape):
        return jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape))

    return (complex_normal(nk, npwx, nwfcU), complex_normal(nspin, nk, nbnd, npwx),
            jnp.asarray(rng.random((nspin, nk, nbnd))))


@pytest.mark.parametrize("nk", [1, 3])
def test_a_second_hubbard_occupation_matrix_compiles_nothing(nk):
    """``projections`` walked k with an eager ``lax.map`` over a closure built per call.

    ``run_scf`` calls :func:`~defumat.hubbard.occupations.occupation_matrix`
    once an iteration outside any ``jit``, once per spin channel inside it, so on
    a CPU (one k-point a step) with more than one k-point every iteration
    compiled the loop again: two programs an iteration at ``nspin = 2``. Two
    calls with different arrays of the same shapes, the second compiling
    nothing, and the value the per-k contraction written out.
    """
    from defumat.hubbard.occupations import occupation_matrix

    columns = jnp.asarray([[[0, 1, 2]], [[2, 3, 4]]])  # two slots of l = 1
    mask = jnp.ones((2, 3), dtype=bool)
    occupation_matrix(*_hubbard_arrays(0, nk=nk)[:3], columns, mask, 1)
    wfcU, psi, weights = _hubbard_arrays(1, nk=nk)
    with counting_compiles() as names:
        ns = occupation_matrix(wfcU, psi, weights, columns, mask, 1)
    assert names == []
    proj = jnp.einsum("kgi,skbg->skbi", jnp.conj(wfcU), psi)[..., columns[:, 0]]
    expected = jnp.einsum("skb,skbna,skbnc->snac", weights, jnp.conj(proj), proj).real
    np.testing.assert_allclose(ns, expected, rtol=1e-13, atol=1e-13)


class _TurningCalculation:
    """The three methods the torque's chunked derivative calls, on random arrays.

    The potential is linear in the density and the Hamiltonian multiplies a
    state by a number read off the magnetization, so the band energy turns with
    the moment and its derivative is not zero; a new instance holds new arrays
    of the same shapes, which is what ``run_torque`` builds at every call.
    """

    k_batch = 1

    def __init__(self, seed, nk):
        rng = np.random.default_rng(seed)
        self.scale = jnp.asarray(1.0 + rng.random(4))
        self.coupling = jnp.asarray(rng.normal(size=(nk, 3)))

    def potential(self, density, *_):
        return type("Potential", (), {"v_scf": self.scale[:, None] * density})

    def hamiltonian(self, v_scf, ddd_paw):
        field = jnp.sum(v_scf[1:], axis=-1)  # (3,)
        coupling = self.coupling

        class Hamiltonian:
            @staticmethod
            def apply(psi, ik):
                return (coupling[ik] @ field) * psi

        return (Hamiltonian,)


def test_a_second_chunked_torque_compiles_nothing():
    """``torque_at_angle`` jitted its chunk's ``value_and_grad`` afresh at every call.

    Two calls, each on a new calculation and a new density of the same shapes,
    k-points one at a time: the second compiles nothing, and both match the
    whole-axis gradient taken without chunks.
    """
    from defumat.forces.torque import torque_at_angle

    nk, nbnd, npw = 3, 2, 5
    rng = np.random.default_rng(7)
    plane = ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0))

    def case(seed):
        states = jnp.asarray(rng.normal(size=(1, nk, nbnd, npw))
                             + 1j * rng.normal(size=(1, nk, nbnd, npw)))
        weights = jnp.asarray(rng.random((1, nk, nbnd)))
        density = jnp.asarray(1.0 + rng.random((2, 6)))
        return _TurningCalculation(seed, nk), states, weights, density

    first = case(0)
    torque_at_angle(*first, plane, 0.3, k_batch=1)
    second = case(1)
    with counting_compiles() as names:
        chunked = torque_at_angle(*second, plane, 0.3, k_batch=1)
    assert names == []
    whole = torque_at_angle(*second, plane, 0.3, k_batch=None)
    assert chunked == pytest.approx(whole, rel=1e-12)
    assert abs(whole) > 1e-3


def test_a_kept_function_takes_another_structure_through_compiled():
    """``compiled_function`` traces once; a call of another shape is keyed afresh.

    And under a trace it is ``jax.jit(fn)``, keeping nothing, which is what the
    loops that use it did before.
    """
    arr = jnp.arange(4.0)
    run = eager.compiled_function(lambda xs: lax.map(lambda x: x * arr, xs), jnp.arange(4.0))
    assert len(eager._PROGRAMS) == 1
    run(jnp.arange(4.0))
    ones = jnp.ones(4)
    with counting_compiles() as names:
        got = run(ones)
    assert names == []
    np.testing.assert_array_equal(got, ones[:, None] * arr)
    np.testing.assert_array_equal(run(jnp.ones(3)), jnp.ones(3)[:, None] * arr)
    assert len(eager._PROGRAMS) == 2
    eager.clear()
    traced = jax.jit(lambda s: eager.compiled_function(lambda xs: xs * s, jnp.ones(2))(
        jnp.ones(2)))(2.0)
    assert len(eager._PROGRAMS) == 0
    np.testing.assert_array_equal(traced, 2.0 * jnp.ones(2))


def test_a_full_cache_says_so_once(monkeypatch):
    monkeypatch.setattr(eager, "CACHE_SIZE", 1)
    _loop(jnp.arange(4.0))
    with pytest.warns(RuntimeWarning, match="dropped the oldest"):
        compiled(lambda xs: lax.map(lambda x: x + 1.0, xs), jnp.arange(4.0))
    assert len(eager._PROGRAMS) == 1
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compiled(lambda xs: lax.map(lambda x: x - 1.0, xs), jnp.arange(4.0))


def test_host_arithmetic_on_constants_is_evaluated_while_tracing():
    """``augmentation_dipole`` reads ``np.asarray`` of a ``jnp`` result on constants."""
    table = np.arange(4.0)

    def fn(x):
        weights = np.asarray(jnp.cumsum(jnp.asarray(table)))
        return x * float(weights.sum())

    got = compiled(fn, jnp.asarray(2.0))
    assert len(eager._PROGRAMS) == 1
    np.testing.assert_allclose(got, 2.0 * 10.0)


def test_a_value_read_off_an_argument_falls_back_to_the_plain_call():
    """``map_axis`` with one entry calls its body on concrete values, which a trace cannot."""
    from defumat.batching import map_k

    def body(x):
        return x * 2.0 if float(x) > 0 else x

    got = compiled(lambda xs: map_k(body, xs, batch=1), jnp.ones(1))
    assert len(eager._PROGRAMS) == 0
    np.testing.assert_allclose(got, [2.0])


#: The smallest ultrasoft cell with more than one k-point, which is what puts
#: ``becsum``'s sum over k on a scan at ``k_batch = 1``.
SILICON_US = Path(__file__).resolve().parents[1] / "data" / "qe" / "si2-us.in"


@pytest.mark.parametrize("regime", ["", "nspin = 2, starting_magnetization(1) = 0.5, "
                                    "occupations = 'smearing', degauss = 0.02",
                                    "noncolin = .true."],
                         ids=["unpolarized", "lsda", "noncollinear"])
def test_a_second_becsum_call_compiles_nothing(pseudo_dir, regime):
    """The SCF calls ``becsum`` eagerly once an iteration, outside any ``jit``.

    Its sum over k is a scan over a closure built at the call, and before it
    went through :func:`compiled` the second call compiled that scan again,
    once per spin channel (``spinor_becsum`` once): two compilations a warm
    iteration on ``ni-ldau-ortho.in``. No SCF is needed to see it, since the
    contraction does not care whether the states are converged.
    """
    text = SILICON_US.read_text()
    if regime:
        text = text.replace("ecutrho=160.0", f"ecutrho=160.0, {regime}")
    calculation = Calculator.from_text(text, pseudo_dir, k_batch=1,
                                       announce=False).calculation
    nk = calculation.system.kpoints.nk
    channels = 2 if "nspin" in regime else 1
    width = calculation.basis.npwx * (2 if "noncolin" in regime else 1)
    rng = np.random.default_rng(0)
    shape = (channels, nk, 4, width)
    states = jnp.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    weights = jnp.full((channels, nk, 4), 0.5)

    first = calculation.becsum(states, weights)
    with counting_compiles() as names:
        second = calculation.becsum(states, weights)
    assert names == []
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_a_second_tetrahedron_occupation_call_compiles_nothing(pseudo_dir):
    """The Fermi level's bisection is a ``fori_loop`` the SCF reaches eagerly.

    ``Calculation.occupations`` is called once an SCF iteration outside any
    ``jit``, and with ``occupations = 'tetrahedra'`` the shared Fermi level was
    bisected by a loop over a closure built at the call, compiled again every
    time: six compilations over a warm five-iteration SCF on this cell.
    """
    calculation = Calculator.from_file(SILICON_US.parent / "al-tetrahedra.in",
                                       pseudo_dir, announce=False).calculation
    nk = calculation.system.kpoints.nk
    rng = np.random.default_rng(0)
    eigenvalues = jnp.asarray(np.sort(rng.uniform(-0.5, 1.0, (1, nk, 6)), axis=-1))

    first, first_levels = calculation.occupations(eigenvalues)
    with counting_compiles() as names:
        second, second_levels = calculation.occupations(eigenvalues)
    assert names == []
    np.testing.assert_array_equal(first, second)
    assert first_levels == second_levels
