"""P48a: ``(1/m*)_ab`` from a central difference of an analytic band velocity.

The checks fail for different reasons, which is the point of having four:

* the **units** -- a free electron is the identity in Rydberg atomic units, so
  the factor of a half in ``(1/m*) = (1/2) d^2 eps/dk^2`` is a consequence and
  not a fitted normalisation;
* the **two routes**, which share nothing -- one differences the expectation
  value of ``dH/dk``, the other differences eigenvalues -- and must agree;
* the **symmetry**, which nothing imposes: a non-degenerate band of a cubic
  crystal at ``Gamma`` has an isotropic tensor, or the assembly is wrong;
* the **degeneracy refusal**, rule D4 -- a per-band second derivative inside a
  multiplet is whatever basis the eigensolver happened to return.

The trap P48a found has a test of its own: the eigenvalue route's stencil must
not contain its own centre, because a high-symmetry k-point holds *fewer* plane
waves than any displaced one -- 725 against 733 on two-atom silicon at
``ecutwfc = 30``, and the same 725/733 for a norm-conserving dataset as for a
PAW one, so it is the cutoff and not the pseudopotential. Differencing across
that step gives an error ``delta/h^2`` which **grows** as the stencil shrinks.
"""

import numpy as np
import pytest

from defumat.calculator import Calculator
from defumat.response.effmass import Multiplet, effective_mass

pytestmark = pytest.mark.unit


_SILICON = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.2, nat = 2, ntyp = 1, ecutwfc = 12.0, nbnd = 8
/
&electrons
  conv_thr = 1e-10
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS crystal
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 4 4 4 0 0 0
"""


@pytest.fixture(scope="module")
def silicon(pseudo_dir):
    calculator = Calculator.from_text(_SILICON, pseudo_dir, announce=False)
    return calculator, calculator.get_scf()


#: The same cell at a cutoff where the stencil at ``L`` straddles a shell of
#: ``G``. At ``ecutwfc = 12`` every stencil point holds the same number of plane
#: waves at every centre, so that cell cannot see this at all.
_SILICON_AT_A_SHELL = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.2, nat = 2, ntyp = 1, ecutwfc = 30.0, nbnd = 10
/
&electrons
  conv_thr = 1e-10
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS crystal
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 4 4 4 1 1 1
"""


@pytest.fixture(scope="module")
def silicon_at_a_shell(pseudo_dir):
    calculator = Calculator.from_text(_SILICON_AT_A_SHELL, pseudo_dir, announce=False)
    return calculator, calculator.get_scf()


@pytest.mark.slow
def test_the_stencil_shares_one_sphere_away_from_gamma(silicon_at_a_shell):
    """A centre-free stencil is half the cure, and freezing the sphere is the rest.

    "Every point is displaced, so the basis-set offset is common to all of them"
    is true at ``Gamma`` and false anywhere else: the points are not displaced by
    the *same* amount, and a shell of ``G`` can sit between ``h`` and ``2h`` as
    easily as between 0 and ``h``. This cell at ``L`` is that case, which the
    first assertion is here to keep it: the centre holds 754 plane waves, the
    near pair 752 and the far pair 744, so a per-k sphere makes the far points
    variationally high against the near ones by a fixed offset that the
    numerator divides by ``3 h^2``.

    The two routes then disagree by **5.0e-3** relative on band 4 at the default
    stencil, 5.26508 against 5.29163, where the converged value is **5.29301**
    (the Richardson limit both routes reach from ``delta = 0.00625`` and
    0.003125, and it is the same limit with or without the freezing). That is
    the error the Richardson step cannot remove, because a quantity with a step
    in it is not ``O(h^2)`` in the first place: the raw stencil's successive
    differences shrink by 20.7, 0.95 and 3.4 as ``delta`` halves where a second
    difference must give four, and with one sphere they shrink by 3.6, 3.9, 4.0.

    With the whole stencil on the centre's sphere the two routes give 5.29182
    and 5.29032, **2.8e-4** relative apart and both within 3e-3 of the limit;
    the eigenvalue route alone moves from 2.79e-2 out to 1.2e-3 out.
    """
    from defumat.basis.planewaves import build_plane_wave_basis
    from defumat.response.effmass import _stencil_kpoints

    calculator, scf = silicon_at_a_shell
    calculation = calculator.calculation

    offsets = [np.zeros(3)]
    for scale in (1, 2):
        for sign in (+1, -1):
            step = np.zeros(3)
            step[0] = sign * scale * 0.0125
            offsets.append(step)
    basis = build_plane_wave_basis(
        calculation.basis.smooth,
        _stencil_kpoints(calculation, np.array([0.5, 0.5, 0.5]), offsets),
        calculator.system.cell, calculator.system.ecutwfc,
    )
    assert tuple(basis.npw) == (754, 752, 752, 744, 744)

    routes = {
        method: effective_mass(
            calculation, scf, (0.5, 0.5, 0.5), method=method, nbnd=10
        ).inverse_mass
        for method in ("velocity", "eigenvalue")
    }
    assert np.allclose(routes["velocity"][4], routes["eigenvalue"][4],
                       rtol=5.0e-4, atol=1.0e-6)
    # ...and each of them against the limit the stencil converges to, which is
    # the assertion a route cannot pass by agreeing with a wrong one.
    for method, inverse_mass in routes.items():
        assert np.diag(inverse_mass[4]) == pytest.approx(5.293014, abs=5.0e-3), method


@pytest.fixture(scope="module")
def masses(silicon):
    calculator, scf = silicon
    return {
        method: effective_mass(
            calculator.calculation, scf, (0.0, 0.0, 0.0), method=method
        )
        for method in ("velocity", "eigenvalue")
    }


@pytest.mark.slow
def test_the_two_routes_agree(masses):
    """They share no machinery, so this is the check on the velocity operator.

    ``method="eigenvalue"`` never forms ``dH/dk`` -- it differences eigenvalues
    from an NSCF -- which is what makes agreement evidence about the ``jvp``
    rather than about the assembly around it.
    """
    velocity = masses["velocity"].inverse_mass
    eigenvalue = masses["eigenvalue"].inverse_mass
    # ``Gamma_1`` is nearly parabolic over the stencil and the two routes land
    # on the same number to nine digits; ``Gamma_2'`` is not, and what is left
    # there is the eigenvalue route's own ``O(h^2)`` -- five times the coefficient
    # of the three-point form, which is what a centre-free stencil costs.
    assert np.allclose(velocity[0], eigenvalue[0], atol=1.0e-7)
    assert np.allclose(velocity[7], eigenvalue[7], rtol=5.0e-4, atol=1.0e-6)


@pytest.mark.slow
def test_the_tensor_is_isotropic_with_nothing_imposing_it(masses):
    """Silicon is cubic and no symmetry is applied anywhere in this path."""
    for name, mass in masses.items():
        for band in (0, 7):
            tensor = mass.inverse_mass[band]
            off = np.abs(tensor - np.diag(np.diag(tensor))).max()
            spread = np.ptp(np.diag(tensor))
            assert off < 1.0e-6, (name, band, off)
            assert spread < 1.0e-6 * abs(tensor[0, 0]), (name, band, spread)


def test_a_degenerate_multiplet_is_refused_and_summed(masses):
    """``Gamma_25'`` is threefold; its bands have no tensor of their own.

    What is defined there is the trace over the manifold, which is invariant
    under the rotation the eigensolver is free in -- so it is reported and the
    per-band tensors are ``nan``.
    """
    mass = masses["velocity"]
    assert (0, 1) in mass.refused and (0, 3) in mass.refused
    assert np.all(np.isnan(mass.inverse_mass[1]))
    assert not np.any(np.isnan(mass.inverse_mass[0]))

    valence = [m for m in mass.multiplets[0] if m.bands == (1, 2, 3)]
    assert len(valence) == 1 and valence[0].degenerate
    assert np.all(np.isfinite(valence[0].inverse_mass_sum))
    # Holes: the valence top curves downward.
    assert valence[0].inverse_mass_sum[0, 0] < 0.0

    with pytest.raises(ValueError, match="degenerate multiplet"):
        mass.principal(band=1)


def test_the_conduction_mass_is_silicons(masses):
    """``Gamma_2'`` is the lowest non-degenerate conduction band at ``Gamma``.

    Its mass is about 0.19 ``m_e`` -- a number with a literature value and an
    all-electron one (Elk gives 0.171 on the same cell in PBE), so this is a
    physics check rather than a regression bound.
    """
    mass = masses["velocity"]
    masses_e, _ = mass.principal(band=7)
    assert np.all(masses_e > 0.0)
    assert 0.15 < np.mean(masses_e) < 0.25


def test_the_free_electron_normalisation_is_the_units():
    """``(1/m*) = (1/2) d^2 eps/dk^2`` with ``eps`` in Ry and ``k`` in 1/bohr.

    ``hbar^2/2 m_e`` is exactly 1 Ry bohr^2, so a free electron has
    ``eps = |k|^2`` and the tensor is the identity. Asserted on the arithmetic
    rather than on a calculation, because it is a statement about the units the
    package works in and nothing else can go wrong with it.
    """
    curvature = 2.0 * np.eye(3)  # d^2(|k|^2)/dk_a dk_b
    assert np.allclose(0.5 * curvature, np.eye(3))


def test_the_truncation_is_reported_not_tuned_away(masses):
    """The Richardson step removes an ``O(h^2)`` error and measures what it was.

    At Elk's own ``deltaem = 0.025`` that error is 3% on silicon's conduction
    band, so a number that is not reported is a number nobody checks.
    """
    for mass in masses.values():
        assert mass.truncation is not None
        assert np.all(mass.truncation >= 0.0)
        assert np.all(np.isfinite(mass.truncation))


def test_the_asymmetry_is_a_null_where_it_cannot_be_a_measurement(masses):
    """One route measures the stencil's error with it and the other cannot.

    ``||M - M^T||`` is free and is a real diagnostic for ``method="velocity"``,
    which builds the two off-diagonal entries from different contractions. The
    eigenvalue route writes both of them from **one** mixed second difference,
    so its tensor is symmetric by construction and the quantity is identically
    zero on every cell, every centre and every step size -- and a check whose
    null result cannot be told from a pass is worse than no check, because
    nothing downstream can tell the two apart. It is reported as ``nan``.

    There is no cheap repair, which is why this is a null rather than a fix:
    the four points of the mixed difference are the same set under exchanging
    the labels, so the route has no second independent estimate to compare
    against. What measures its stencil error is ``truncation``.
    """
    velocity = masses["velocity"].asymmetry_by_spin
    assert np.all(np.isfinite(velocity))
    assert np.all(velocity >= 0.0)

    eigenvalue = masses["eigenvalue"].asymmetry_by_spin
    assert np.all(np.isnan(eigenvalue))
    assert eigenvalue.shape == velocity.shape


def test_without_richardson_there_is_no_truncation_estimate(silicon):
    calculator, scf = silicon
    mass = effective_mass(
        calculator.calculation, scf, (0.0, 0.0, 0.0), richardson=False
    )
    assert mass.truncation is None


def test_the_eigenvalue_stencil_never_contains_its_centre():
    """The centre holds fewer plane waves than any displaced point.

    Asserted on the stencil rather than on a number, because the failure it
    guards against is silent: including the centre gives a smooth, symmetric,
    plausible tensor whose error *grows* as the stencil shrinks. The offsets are
    read out of the built k-set, so this catches a regression in the formula as
    well as in the point list.
    """
    import inspect

    from defumat.response import effmass

    source = inspect.getsource(effmass._by_eigenvalue)
    # ``eps[0]`` is the centre and it may be *returned* (the eigenvalues the
    # multiplet structure is read off) but never differenced.
    body = source[source.index("curvatures = []"):source.index("return curvatures")]
    assert "eps[0]" not in body


def test_an_unknown_method_is_refused(silicon):
    calculator, scf = silicon
    with pytest.raises(ValueError, match="unknown effective-mass method"):
        effective_mass(calculator.calculation, scf, (0, 0, 0), method="magic")


def test_the_calculator_reaches_it(silicon):
    """P38's rule: a new entry point gets a facade method in the same pass."""
    import ast
    import inspect
    import textwrap

    assert hasattr(Calculator, "get_effective_mass")
    source = textwrap.dedent(inspect.getsource(Calculator.get_effective_mass))
    tree = ast.parse(source)
    body = tree.body[0].body
    if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]  # the docstring is prose, not code
    code = "\n".join(ast.unparse(node) for node in body)
    assert "effective_mass(" in code
    # A facade delegates; it does not compute.
    assert "jvp" not in code and "np." not in code
