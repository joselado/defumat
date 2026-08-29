"""Refusals that existed on one entry point and not on its sibling.

Every case here is the same shape, and it is the shape the capability audit of
2026-08-29 found over and over: a combination this package does not implement is
refused *somewhere* -- in the functional, in the stress, in an upstream driver,
in a docstring -- and a second path reaching the same physics had no check at
all and returned a plausible number. None of them raised; each answered.

* the elastic constants refuse ultrasoft and PAW through
  :func:`~pypresso.response.phonon.require_norm_conserving`, whose own docstring
  names them -- and :func:`~pypresso.response.elastic.elastic_constants`, which
  :meth:`pypresso.calculator.Calculator.get_elastic_constants` calls directly,
  never called it;
* a potential-only functional has no energy to differentiate, and
  ``method='analytic'`` did not go through the functional that says so;
* a magnetic field's energy is outside the reported total, so the stress refuses
  it and the *force* did not;
* ``chi^(2)`` is refused by name by a function that nothing called;
* and five input combinations ``pw.x`` stops on that this reader accepted --
  including an LSDA run with no ``starting_magnetization``, which converges to
  the unpolarized solution and reports success.

They are unit tests on purpose: every one is a guard, so none of them needs an
SCF to fire.
"""

import warnings

import pytest

from pypresso.io.pwin import parse_pw_input
from pypresso.system.builder import build_system

pytestmark = pytest.mark.unit


_SILICON = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.2, nat = 2, ntyp = 1, ecutwfc = 12.0
  {extra}
/
&electrons
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS crystal
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""


def _system(extra: str, calculation: str = "scf"):
    text = _SILICON.format(extra=extra)
    if calculation != "scf":
        text = text.replace("calculation = 'scf'", f"calculation = {calculation!r}")
    return build_system(parse_pw_input(text))


# --- the input reader: QE's own checks, which were not made ------------------


def test_lsda_without_a_starting_magnetization_is_refused():
    """``input.f90:1506``. Nothing in the SCF breaks the spin symmetry.

    With every ``starting_magnetization`` at zero the two channels start
    identical, stay identical and converge to the unpolarized solution --
    ``Calculation.starting_density`` says so in its own docstring, and no caller
    enforced it. The run reported success and printed a total magnetization of
    zero. :mod:`pypresso.scf.continuation` has the equivalent guard on the
    *promotion* path and has had it since P23.
    """
    with pytest.raises(ValueError, match="no starting_magnetization"):
        _system("nspin = 2, occupations = 'smearing', degauss = 0.02")


@pytest.mark.parametrize(
    "extra",
    [
        # Presence, not value: QE's test is ``sm_wasnt_set``, so an explicit
        # zero is a request for the unpolarized solution and is honoured. The
        # smearing is not incidental: ``input.f90`` checks fixed-occupation LSDA
        # *first* (line 784, against 1507 for this one), so a default-occupation
        # input never reaches this rule in either code.
        "nspin = 2, starting_magnetization(1) = 0.0, "
        "occupations = 'smearing', degauss = 0.02",
        # ``two_fermi_energies``: the magnetization is fixed by another route.
        "nspin = 2, tot_magnetization = 1.0",
        # ``tfixed_occ``.
        "nspin = 2, occupations = 'from_input', nbnd = 8",
    ],
)
def test_the_ways_qe_lets_an_lsda_run_off_that_check(extra):
    assert _system(extra).nspin == 2


def test_a_non_self_consistent_run_is_not_asked_for_a_magnetization():
    """``lscf`` is part of QE's condition: an NSCF run converges nothing."""
    assert _system("nspin = 2", calculation="nscf").nspin == 2


def test_an_unknown_occupations_keyword_is_refused():
    """``set_occupations.f90``'s ``CASE DEFAULT``.

    ``Calculation.occupations`` dispatches ``fixed``, ``from_input`` and
    ``tetrahedra*`` and then falls through to the smearing branch with no
    ``else``, so a typo selected Gaussian smearing and ran.
    """
    with pytest.raises(ValueError, match="occupations = 'tetrahedron'"):
        _system("occupations = 'tetrahedron'")


def test_smearing_without_a_width_is_refused():
    """``lgauss = ( degauss > 0.0_dp )``.

    ``degauss`` defaults to zero, and ``(ef - e)/degauss`` is then ``0/0`` at
    whichever band the Fermi bisection lands on: the weights come back with a
    NaN in them and it propagates into the density. The *smearing function's*
    name was already validated, so one half of the pair stopped and the other
    did not.
    """
    with pytest.raises(ValueError, match="no degauss"):
        _system("occupations = 'smearing', smearing = 'gaussian'")


def test_fixed_occupations_with_a_width_only_warns():
    """QE's ``errore(..., -1)``: a negative severity prints and continues."""
    with pytest.warns(UserWarning, match="ignored"):
        _system("occupations = 'fixed', degauss = 0.02")


@pytest.mark.parametrize("component", [1, 2])
def test_an_off_axis_b_field_is_refused_for_a_collinear_run(component):
    """``input.f90:1612``.

    ``Calculation._build_magnetic_field`` slices ``uniform[2:3]`` for a
    collinear run, so the x and y components were dropped in silence: the run
    converged, reported the z component's field energy, and said nothing about
    the other two.
    """
    with pytest.raises(ValueError, match="only B_field"):
        _system(
            "nspin = 2, starting_magnetization(1) = 0.1, "
            "occupations = 'smearing', degauss = 0.02, "
            f"b_field({component}) = 0.01"
        )


def test_the_z_component_alone_is_still_accepted():
    assert _system(
        "nspin = 2, starting_magnetization(1) = 0.1, b_field(3) = 0.01, "
        "occupations = 'smearing', degauss = 0.02"
    ).b_field[2] == 0.01


def test_a_direction_constraint_is_refused_for_a_collinear_run():
    """``add_bfield.f90``'s ``i_cons == 2`` check, made at input time here.

    The penalty is ``(m_z/|m| - cos theta)^2`` and ``_polar_cosine`` reads the
    *last* component of the moment -- which for a collinear one is the moment
    itself, so the cosine is +-1 whatever the density does. The constraint was
    a no-op with a vanishing gradient.
    """
    with pytest.raises(ValueError, match="atomic direction"):
        _system(
            "nspin = 2, starting_magnetization(1) = 0.1, "
            "occupations = 'smearing', degauss = 0.02, "
            "constrained_magnetization = 'atomic direction'"
        )


def test_a_total_moment_constraint_needs_a_vector_to_constrain():
    """``input.f90``: ``i_cons = 3`` requires ``nspin = 4``.

    ``fixed_magnetization`` is a vector; a collinear moment has one component,
    which NumPy broadcasts against the ``(3,)`` target instead of failing.
    """
    with pytest.raises(ValueError, match="'total' requires"):
        _system(
            "nspin = 2, starting_magnetization(1) = 0.1, "
            "occupations = 'smearing', degauss = 0.02, "
            "constrained_magnetization = 'total'"
        )


def test_an_atomic_constraint_needs_something_to_build_its_targets_from():
    """``input.f90``: the targets *are* ``starting_magnetization``."""
    with pytest.raises(ValueError, match="needs some starting_magnetization"):
        _system("nspin = 2, constrained_magnetization = 'atomic', "
                "occupations = 'smearing', degauss = 0.02")


# --- the response and force paths -------------------------------------------


def test_chi2_is_refused_on_the_public_path_and_not_only_by_name():
    """The refusal existed and the only call to it was in a test.

    :func:`~pypresso.response.nonlinear.susceptibility_field_derivative` is in
    ``__all__`` and its docstring said it was "kept and refused rather than
    exposed" -- it was exposed. The tensor it returns is missing the
    ``<u_i|r_k|u_j>`` term, is 42% wrong on its displacement counterpart, and no
    symmetry check sees the difference. ``allow_incomplete=True`` is how the
    measurements that establish that reach it.
    """
    import inspect

    from pypresso.response.nonlinear import susceptibility_field_derivative

    signature = inspect.signature(susceptibility_field_derivative)
    assert signature.parameters["allow_incomplete"].default is False
    with pytest.raises(NotImplementedError, match="dvpsi_e2"):
        susceptibility_field_derivative(
            None, None, None, None, None, None
        )


def test_the_elastic_constants_call_the_refusal_that_names_them():
    """``require_norm_conserving`` guards the strain coordinate's third
    derivative, its docstring says so, and this entry point did not call it.

    :mod:`pypresso.response.electrostriction` calls it before reaching
    :func:`~pypresso.response.elastic.elastic_constants`, so the hole was only
    on the direct path --
    :meth:`pypresso.calculator.Calculator.get_elastic_constants`, which is the
    one a user takes. P44 measured what comes back without it: 1.3e-2 against a
    finite difference on ultrasoft and PAW, where the norm-conserving control on
    the same script is 2.3e-4.
    """
    import inspect

    from pypresso.response import elastic

    body = inspect.getsource(elastic.elastic_constants)
    assert "require_norm_conserving(calculation)" in body


@pytest.mark.parametrize(
    "guard, name",
    [
        ("reject_potential_only", "the meta-GGA refusal"),
        ("reject_magnetic_field", "the magnetic-field refusal"),
    ],
)
def test_every_force_method_passes_the_same_guards(guard, name):
    """Before dispatch, so the analytic route is covered too.

    The analytic forces are a transcription of QE's six expressions and share no
    machinery with :func:`~pypresso.forces.energy.energy_at`, so a refusal
    written into the functional never reached ``method='analytic'``: a
    Tran-Blaha run came back with a force. The field refusal is the stress
    path's (:func:`pypresso.stress.energy.require_a_differentiable_cell`), which
    the force path did not have at all.
    """
    import inspect

    import pypresso.forces as forces

    assert f"{guard}(calculation)" in inspect.getsource(forces.compute_forces), name
