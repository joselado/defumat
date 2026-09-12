"""The gamma-only primitives, against the full sphere they replace.

At ``k = 0`` a state can be chosen real and ``c(-G) = conj(c(G))``, so half the
sphere carries all of it. These are the transforms and the inner product that
*consume* that storage; everything above them in the gamma path is written in
terms of these three, so they are tested on their own before anything else uses
them.

**The reference is the full sphere**, which needs no external code: the same
cell at the same cutoff, transformed the ordinary way, is the exact answer.
"""

import numpy as np
import pytest

import jax.numpy as jnp

from defumat.basis.fft import (
    force_real_g0, g_to_r, g_to_r_gamma, gamma_inner, r_to_g, r_to_g_gamma,
)
from defumat.basis.gvectors import generate_gvectors
from defumat.system.cell import Cell

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def spheres():
    cell = Cell.from_ibrav(2, celldm=[10.2, 0, 0, 0, 0, 0])
    full = generate_gvectors(cell, 40.0, gamma_only=False)
    half = generate_gvectors(cell, 40.0, gamma_only=True)
    return cell, full, half


def _band_limited(full, rng):
    """A real field that the sphere represents exactly, so a round trip is one.

    A random grid field has frequencies outside the cutoff and both spheres
    truncate them identically -- which is a fine equality test and a useless
    round-trip one. Building the field *from* the sphere removes that.
    """
    coefficients = (rng.standard_normal(full.ngm) + 1j * rng.standard_normal(full.ngm))
    box = np.zeros(int(np.prod(full.grid)), dtype=complex)
    index = np.asarray(full.fft_index)
    minus = np.asarray(full.fft_index_minus)
    box[index] = coefficients
    # Impose reality: c(-G) = conj(c(G)), and c(0) real.
    box[minus] = np.conj(coefficients)
    box[index[0]] = coefficients[0].real
    field = np.fft.ifftn(box.reshape(full.grid)) * np.prod(full.grid)
    return jnp.asarray(field.real)


def test_the_half_sphere_holds_half_the_vectors(spheres):
    """``(ngm_full + 1)/2`` -- G = 0 is kept exactly once."""
    _, full, half = spheres
    assert half.ngm == (full.ngm + 1) // 2
    assert tuple(full.grid) == tuple(half.grid)


def test_g_zero_is_its_own_conjugate_partner(spheres):
    """``nlm[0] == nl[0]``, which is why every caller has to skip it.

    Adding both would write ``G = 0`` into the box twice, and the scatter
    accumulates rather than sets.
    """
    _, _, half = spheres
    assert int(half.fft_index[0]) == int(half.fft_index_minus[0])


def test_the_gamma_transform_is_the_full_sphere_transform(spheres):
    """The whole claim of the storage, in one equality.

    A real field's half-sphere coefficients rebuild exactly the field its full
    sphere does -- so anything written against the full transform is correct
    against this one.
    """
    _, full, half = spheres
    field = _band_limited(full, np.random.default_rng(0))

    whole = g_to_r(r_to_g(field, full.fft_index), full.fft_index, full.grid)
    stored = r_to_g_gamma(field, half.fft_index)
    rebuilt = g_to_r_gamma(stored, half.fft_index, half.fft_index_minus, half.grid)

    np.testing.assert_allclose(np.asarray(rebuilt), np.asarray(whole.real), atol=1e-13)
    # ... and it is a genuine round trip, the field being band-limited.
    np.testing.assert_allclose(np.asarray(rebuilt), np.asarray(field), atol=1e-13)


def test_the_inner_product_doubles_the_half_and_counts_g_zero_once(spheres):
    """``2 Re sum - Re(conj(a_0) b_0)`` against the full-sphere sum.

    The ``G = 0`` correction is the term that gets dropped in one call site out
    of ten, and only an energy comparison notices -- hence one helper and this
    test of it.
    """
    _, full, half = spheres
    rng = np.random.default_rng(1)
    a = _band_limited(full, rng)
    b = _band_limited(full, rng)

    reference = complex(jnp.sum(r_to_g(a, full.fft_index).conj()
                                * r_to_g(b, full.fft_index)))
    gamma = float(gamma_inner(r_to_g_gamma(a, half.fft_index),
                              r_to_g_gamma(b, half.fft_index), True))
    assert reference.imag == pytest.approx(0.0, abs=1e-14)
    assert gamma == pytest.approx(reference.real, abs=1e-14)


def test_dropping_the_g_zero_correction_is_visible(spheres):
    """The correction is not round-off, so a test that ignores it would pass.

    Pinned because the failure it guards is quiet: without it every overlap is
    wrong by one G-component, which shifts an eigenvalue rather than raising.
    """
    _, full, half = spheres
    rng = np.random.default_rng(2)
    a = _band_limited(full, rng)
    stored = r_to_g_gamma(a, half.fft_index)

    correct = float(gamma_inner(stored, stored, True))
    without = float(2.0 * jnp.sum(stored.conj() * stored).real)
    assert abs(without - correct) > 1e-6 * abs(correct)


def test_the_g_zero_coefficient_is_forced_real(spheres):
    """``regterg.f90:174``. An imaginary part there makes the field complex."""
    _, _, half = spheres
    coefficients = jnp.asarray(np.random.default_rng(3).standard_normal(half.ngm)
                               + 1j * np.random.default_rng(4).standard_normal(half.ngm))
    forced = force_real_g0(coefficients, True)
    assert complex(forced[0]).imag == 0.0
    # ... and nothing else is touched.
    np.testing.assert_array_equal(np.asarray(forced[1:]), np.asarray(coefficients[1:]))
    # A non-gamma set passes through untouched, including G = 0.
    np.testing.assert_array_equal(
        np.asarray(force_real_g0(coefficients, False)), np.asarray(coefficients)
    )


# -- who may consume the half sphere, and who must say they cannot ------------
#
# ``gamma_storage_is_consumable`` decides what the *SCF* can do with the
# storage. It says nothing about what happens to the states afterwards, and
# ``gamma_only`` appeared nowhere in ``hubbard/``, ``projwfc/``, ``tddft/`` or
# ``basis/sample.py`` -- four consumers summing a half sphere as if it were a
# whole one, each returning a plausible number about a factor of two out.


_GAMMA_HUBBARD = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.2, nat = 2, ntyp = 1, ecutwfc = 12.0
  nosym = .true.
/
&electrons
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS crystal
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS gamma
HUBBARD ortho-atomic
 U Si-3p 2.0
"""


def test_a_hubbard_run_cannot_consume_the_half_sphere():
    """``ns`` is built from a plain ``<wfcU|psi>`` inside the SCF.

    This is the one of the four that is wrong *before* the run finishes: the
    Hubbard potential and energy follow the occupations, so the SCF converges
    silently to a different ground state rather than reporting a bad number at
    the end. It is therefore a substitution -- run the same cell at an explicit
    k = 0 -- and not something left to a consumer to refuse.
    """
    from defumat.io.pwin import parse_pw_input
    from defumat.scf.driver import gamma_storage_is_consumable
    from defumat.system.builder import build_system

    system = build_system(parse_pw_input(_GAMMA_HUBBARD))
    assert system.kpoints.gamma_only and system.nosym
    assert system.hubbard is not None
    assert not gamma_storage_is_consumable(system, ())

    # ... and without the U the same cell keeps the storage, so the test above
    # is about the U and not about the cell.
    plain = build_system(parse_pw_input(
        _GAMMA_HUBBARD.split("HUBBARD")[0]
    ))
    assert gamma_storage_is_consumable(plain, ())


def test_the_post_scf_consumers_refuse_the_half_sphere_by_name():
    """The other three are after the run, so a refusal is enough.

    Each has a full-sphere sibling that is an *exact* substitution -- the same
    cell as an explicit single k-point at the origin -- so the refusal costs a
    factor of two in storage and nothing in physics.
    """
    from defumat.basis.gvectors import refuse_gamma_storage

    refuse_gamma_storage(False, "anything", "on the whole sphere")  # silent
    with pytest.raises(NotImplementedError, match="2 Re"):
        refuse_gamma_storage(True, "the projected density of states", "...")


@pytest.mark.parametrize("module,function", [
    ("defumat.projwfc.projections", "atomic_projections"),
    ("defumat.tddft.chi0", "require_a_sum_over_states_regime"),
    ("defumat.workflows.transport", "run_vertical_transport"),
    ("defumat.workflows.stm", "run_stm"),
])
def test_each_named_consumer_calls_the_guard(module, function):
    """A set difference, not a read-through: the four sites are named here so a
    fifth consumer added later has to be argued about rather than forgotten."""
    import importlib
    import inspect

    source = inspect.getsource(getattr(importlib.import_module(module), function))
    assert "refuse_gamma_storage" in source


# -- trap 1: abs and sqrt at a zero symmetry forces --------------------------
#
# ``CLAUDE.md``'s first recurring trap. The primal survives at an exact zero and
# only the *tangent* is NaN, so a cell that never lands on one -- every small
# bulk case in the suite -- reports nothing. The test therefore forces the zero
# rather than hoping to meet one.


def test_the_hartree_energy_is_the_same_number_with_and_without_abs(spheres):
    """**A measured null, recorded so it is not re-derived.**

    ``scf/potential.py`` carried ``jnp.abs(rho_g) ** 2`` where every other
    differentiated site in the package uses ``Re(conj(rho) rho)``, and the
    prediction was trap 1: ``abs`` has no derivative at zero, and a structure
    factor vanishing *exactly* is what symmetry arranges on a supercell. This
    is the one term every force, every stress and every phonon differentiates,
    so the prediction was worth checking rather than assuming.

    **It does not hold in JAX.** ``jnp.abs`` of a complex number has a finite
    derivative at exactly zero here -- measured as 0, in reverse mode, in
    forward mode and in the Hessian -- so ``abs(z)**2`` was never a NaN. The
    expression was changed anyway, because it is the package's convention and
    the two agree to 1.9e-16 in the energy and 2.8e-16 in its gradient -- the
    old form routes through a ``sqrt`` and squares it back, which is one
    rounding the new one does not do, and that is the whole of the
    difference. What must not happen is the change being remembered as a
    fix. This test is what says which it was.

    If a future JAX changes that rule the first assertion below fails, which is
    the warning the prediction was actually worth.
    """
    import jax

    from defumat.scf.potential import hartree

    cell, full, _ = spheres
    rng = np.random.default_rng(20260911)
    rho = (rng.standard_normal(full.ngm) + 1j * rng.standard_normal(full.ngm))
    rho[7] = 0.0 + 0.0j  # bit-exact, the way a cancelling phase sum leaves it
    parts = jnp.stack([jnp.real(jnp.asarray(rho)), jnp.imag(jnp.asarray(rho))])

    g2 = full.kinetic(cell)
    inverse = jnp.where(g2 > 1e-12, 1.0 / jnp.where(g2 > 1e-12, g2, 1.0), 0.0)

    def banned(p):
        return jnp.sum(jnp.abs(p[0] + 1j * p[1]) ** 2 * inverse)

    def kept(p):
        r = p[0] + 1j * p[1]
        return jnp.sum(jnp.real(jnp.conj(r) * r) * inverse)

    assert np.all(np.isfinite(np.asarray(jax.grad(banned)(parts)))), (
        "jnp.abs is no longer differentiable at an exact zero: the Hartree "
        "energy's old form would have been a NaN in every force and stress"
    )
    # Equal in value bit for bit, and in the gradient to the last bit -- the
    # old form routes through a sqrt and squares it back, which is one rounding
    # the new one does not do.
    assert float(kept(parts)) == pytest.approx(float(banned(parts)), rel=1e-15)
    assert np.asarray(jax.grad(kept)(parts)) == pytest.approx(
        np.asarray(jax.grad(banned)(parts)), rel=1e-14
    )

    # And the function itself, on the same input, is finite in value and slope.
    def energy(p):
        return hartree(p[0] + 1j * p[1], full, cell)[1]

    assert np.isfinite(float(energy(parts)))
    assert np.all(np.isfinite(np.asarray(jax.grad(energy)(parts))))


def test_the_local_spin_frame_has_a_finite_gradient_where_the_magnetization_vanishes():
    """``d|m|/dm = m/|m|`` is ``0/0`` at an exact zero, and the zero is forced.

    Two mechanisms produce a bit-exact one: ``sym_rho``'s axial-vector average
    at a grid point whose magnetic little group admits no invariant axial vector
    (``m + (-m)`` with +-1 rotation entries is exact), and a vacuum region where
    the density underflows. Every spinor force, every spinor stress and every
    ``jvp`` in the response stack differentiates ``v_of_rho``, and so this.

    Guarding the *division* after the ``sqrt`` -- which is what stood here -- is
    not enough: ``sqrt`` has an infinite derivative at zero, so the tangent is
    ``0 * inf`` however careful everything downstream is. The mask has to go on
    the sum of squares.
    """
    import jax

    from defumat.xc.functional import local_spin_frame, safe_modulus

    charge = jnp.asarray([1.0, 2.0, 0.5, 3.0])
    magnetization = jnp.asarray([
        [0.1, 0.0, -0.2, 0.3],
        [0.0, 0.0, 0.4, -0.1],
        [0.2, 0.0, 0.0, 0.5],
    ])  # column 1 is bit-exactly zero in all three components
    assert not np.any(np.asarray(magnetization)[:, 1])

    for target in (safe_modulus,
                   lambda m: local_spin_frame(charge, m)[1],
                   lambda m: local_spin_frame(charge, m)[0]):
        gradient = jax.grad(lambda m: jnp.sum(target(m)))(magnetization)
        assert np.all(np.isfinite(np.asarray(gradient))), target

    # The value is untouched: sqrt(0) is 0 either way.
    assert float(safe_modulus(magnetization)[1]) == 0.0
    assert np.asarray(safe_modulus(magnetization)) == pytest.approx(
        np.linalg.norm(np.asarray(magnetization), axis=0)
    )


def test_every_differentiated_magnetization_site_shares_the_one_guard():
    """The same three lines, and the same defect, were written five times.

    The first audit of this found two and its name said "both", which is why
    the other three were never looked at -- and ``paw/gradient.py`` claimed in
    a comment to use "the same guard the plane-wave branch uses", which was
    false for the GGA branch. So this is a **sweep with an allowlist** rather
    than a list of the sites somebody remembered: any new bare ``|m|`` in the
    package fails it, and exempting one costs an entry here with a reason.

    What the sweep looks for is the *per-point* modulus of a three-component
    field -- a sum of squares over ``axis=0`` under a ``sqrt`` -- because that
    is the defect: `|m|` evaluated at every grid point, where a bit-exact zero
    is reached rather than approached. A norm with no axis is a different
    thing and is not flagged: ``rotated_density`` normalises a *direction*
    whose norm is identically one by construction (``cos^2 + sin^2``), so its
    ``sqrt`` sits at 1 where it is perfectly smooth.

    The two exempt sites are exempt because **no tangent flows through them**:
    ``_noncollinear_magnetization`` and ``_absolute_magnetization`` are reports
    that end in a ``float()``.
    """
    import inspect
    import re

    from defumat.forces import torque
    from defumat.paw import gradient as paw_gradient
    from defumat.scf import continuation, driver, potential
    from defumat.xc import functional

    guarded = [
        paw_gradient._noncollinear_gradient,
        functional.local_spin_frame,
        potential._noncollinear_gradient_correction,
        potential._noncollinear_meta_exchange,
        torque.rotated_density,
    ]
    for target in guarded:
        source = inspect.getsource(target)
        assert "safe_modulus(" in source, f"{target.__name__} lost its guard"

    # The per-point modulus of a vector field, in both spellings it has been
    # written in here: ``jnp.sqrt(jnp.sum(x**2, axis=0))`` and
    # ``jnp.sum(x**2, axis=0) ** 0.5``.
    bare = re.compile(
        r"sqrt\(\s*jnp\.sum\([^)]*\*\*\s*2\s*,\s*axis\s*="
        r"|\*\*\s*2\s*,\s*axis\s*=\s*0\s*\)\s*\*\*\s*0\.5"
    )
    exempt = {
        driver._noncollinear_magnetization,   # a report; ends in float()
        continuation._absolute_magnetization,  # a report; ends in float()
    }
    offenders = []
    for module in (paw_gradient, functional, potential, torque, driver,
                   continuation):
        for name, value in vars(module).items():
            if not callable(value) or not hasattr(value, "__code__"):
                continue
            if getattr(value, "__module__", None) != module.__name__:
                continue
            try:
                source = inspect.getsource(value)
            except (OSError, TypeError):
                continue
            if bare.search(source) and value not in exempt:
                offenders.append(f"{module.__name__}.{name}")
    assert not offenders, (
        "a bare |m| is back in a differentiated path -- use safe_modulus, or "
        f"add it to the allowlist above with a reason: {sorted(offenders)}"
    )


def test_the_noncollinear_gga_potential_is_finite_at_a_vanishing_moment():
    """The site that every spinor force differentiates and no test did.

    ``_noncollinear_gradient_correction`` is what ``v_of_rho`` dispatches into
    at ``nspin_mag = 4`` with a gradient-corrected functional, and every spinor
    force, spinor stress and response ``jvp`` goes through it. The one
    regression case that looks as though it covers the branch sets
    ``starting_magnetization = 0``, so ``domag`` is false and the *unpolarized*
    branch is what it measures.
    """
    import jax

    from defumat.basis.gvectors import generate_gvectors
    from defumat.scf.potential import _noncollinear_gradient_correction
    from defumat.system.cell import Cell
    from defumat.xc.functional import get_functional

    cell = Cell.from_ibrav(1, [8.0, 0, 0, 0, 0, 0])
    gvectors = generate_gvectors(cell, 12.0)
    grid = tuple(gvectors.grid)
    functional = get_functional("pbe")

    rng = np.random.default_rng(4)
    charge = jnp.asarray(0.5 + 0.1 * rng.random(grid))
    magnetization = jnp.asarray(0.05 * rng.normal(size=(3,) + grid))
    # A whole plane of bit-exact zeros, which is what ``sym_rho``'s axial
    # average leaves behind and what a vacuum layer underflows to.
    magnetization = magnetization.at[:, 0, :, :].set(0.0)
    assert not np.any(np.asarray(magnetization)[:, 0])

    def energy(m):
        rho = jnp.concatenate([charge[None], m])
        _, value = _noncollinear_gradient_correction(
            rho, gvectors, cell, functional, None, None
        )
        return value

    gradient = np.asarray(jax.grad(energy)(magnetization))
    assert np.all(np.isfinite(gradient)), (
        "d E_xc / d m is not finite where |m| is bit-exactly zero"
    )
    # The tangent on the zero plane is a one-sided derivative, which is what a
    # potential at a vanishing moment *is* -- finite, and not a direction
    # picked out of a conical singularity.
    assert np.all(np.isfinite(gradient[:, 0, :, :]))
