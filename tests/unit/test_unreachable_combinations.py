"""Section 2 of ``GAPS.md``: the physics works and the plumbing could not say so.

Where ``test_sibling_refusals`` collects combinations that should have stopped
and did not, and ``test_state_across_boundaries`` collects state that failed to
cross a call, these are combinations that were **unreachable**: an entry point
with no facade method, a channel count rebuilt from the wrong spin number, and a
field with nowhere to act.

* ``Calculator.get_band_velocities`` did not exist. ``band_velocities`` has had
  the facade's own calling convention since P24 and no ``get_*`` beside it, so a
  Fermi velocity or an effective mass needed a ``Calculation`` and an
  ``SCFResult`` threaded by hand -- and the same function forwarded only ``ns``
  into its NSCF branch, which is P38's mixed-state defect in a third place.
* ``tau``'s channel count was rebuilt as ``nspin`` where it is ``nspin_mag``.
  The two come apart at 4, so a *nonmagnetic* spin-orbit meta-GGA run asked for
  four channels of a quantity produced with one: the residual solver died on the
  reshape and a ``starting_from`` promotion dropped its converged ``tau`` in
  silence and fell back to Thomas-Fermi.
* a magnetic field in a noncollinear run with no starting moment had
  ``nspin_mag = 1`` and no magnetization channels to act on.
* ``occupations = 'fixed'`` with ``nspin = 2`` was refused for want of a
  benchmark. One is generated now (``o-atom-fixed-lsda``), and with it QE's own
  input rules, which are narrower than the refusal implied.
"""

import ast
import inspect
import textwrap
import warnings

import numpy as np
import pytest

from pypresso.io.pwin import parse_pw_input
from pypresso.scf.driver import Calculation
from pypresso.scf.occupations import fixed_occupations
from pypresso.scf.residual import make_residual
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


def _system(extra: str = ""):
    return build_system(parse_pw_input(_SILICON.format(extra=extra)))


def _calculation(pseudo_dir, extra: str = ""):
    from pypresso.pseudo import read_upf

    system = _system(extra)
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    return Calculation(system, pseudos)


# --- the entry point with no front door --------------------------------------


def test_the_calculator_reaches_the_band_velocities():
    """A ``get_*`` for every entry point, which is CLAUDE.md's own rule."""
    from pypresso.calculator import Calculator

    assert hasattr(Calculator, "get_band_velocities")
    parameters = inspect.signature(Calculator.get_band_velocities).parameters
    assert "kpoints" in parameters


def test_the_band_velocity_facade_is_a_delegation_and_not_a_computation():
    """The facade rule: one call out, no physics of its own."""
    from pypresso.calculator import Calculator

    # The docstring is prose; what must be a delegation is the *code*, so the
    # statements are read off the parse tree with the docstring dropped.
    tree = ast.parse(textwrap.dedent(inspect.getsource(Calculator.get_band_velocities)))
    body = tree.body[0].body
    if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    code = "\n".join(ast.unparse(node) for node in body)
    assert "band_velocities(" in code
    for forbidden in ("jnp.", "jax.", "np.", "for "):
        assert forbidden not in code, forbidden


def test_the_band_velocities_forward_the_whole_mixed_state():
    """``ns`` alone was crossing, so PAW raised and a meta-GGA had no ``tau``.

    ``fixed_density_states`` has taken ``becsum``, ``tau``, ``field`` and
    ``field_scale`` since the previous pass; forwarding one of them and not the
    rest is the same defect ``run_dos`` and ``run_pdos`` were fixed for.
    """
    from pypresso.response.velocity import band_velocities
    from pypresso.workflows.nscf import fixed_density_states

    accepted = set(inspect.signature(fixed_density_states).parameters)
    body = inspect.getsource(band_velocities)
    for name in ("ns", "becsum", "tau", "field", "field_scale"):
        assert name in accepted, name
        assert f"{name}=" in body, name


def test_paw_band_velocities_on_a_path_run(pseudo_dir):
    """The forwarding, measured rather than inspected.

    ``fixed_density_states`` refuses a PAW run without ``becsum`` by name -- the
    one-centre coefficients cannot be rebuilt from the density -- so this call
    used to stop on that refusal with no argument that could satisfy it.
    """
    from pypresso.calculator import Calculator
    from pypresso.system.kpoints import KPoints

    paw = _SILICON.format(extra="").replace(
        "Si.pz-vbc.UPF", "Si.pz-n-kjpaw_psl.0.1.UPF"
    ).replace("ecutwfc = 12.0", "ecutwfc = 20.0, ecutrho = 120.0")
    calculator = Calculator.from_text(paw, pseudo_dir, announce=False)

    path = KPoints(coords=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
                   weights=np.array([1.0, 1.0]))
    velocities = np.asarray(
        calculator.get_band_velocities(kpoints=path).velocities
    )
    assert velocities.shape == (2, 4, 3)
    # ``Gamma`` is a point of the cubic group, so every occupied band's velocity
    # vanishes there -- the cheapest statement that this is a velocity and not
    # an array of the right shape.
    assert np.max(np.abs(velocities[0])) < 1.0e-3


# --- tau's channel count is nspin_mag, never nspin ---------------------------


def test_the_residual_sizes_tau_like_the_density(pseudo_dir):
    """``nspin`` is 4 for a nonmagnetic spin-orbit run and ``tau`` has one channel.

    The reshape this used to die on. Read off the density rather than rebuilt
    from a spin number, which is correct in all three regimes at once.
    """
    smearing = ", occupations = 'smearing', degauss = 0.02"
    for extra, expected in (
        (", input_dft = 'tb09'", 1),
        (", input_dft = 'tb09', nspin = 2, starting_magnetization(1) = 0.1"
         + smearing, 2),
        (", input_dft = 'tb09', noncolin = .true.", 1),
        (", input_dft = 'tb09', noncolin = .true., "
         "starting_magnetization(1) = 0.1", 4),
    ):
        with warnings.catch_warnings():
            # The datasets are PZ and the functional is TB09; that mismatch has
            # its own warning and its own test.
            warnings.simplefilter("ignore")
            calculation = _calculation(pseudo_dir, extra)
        residual = make_residual(calculation, 8, 1.0e-6)
        density = np.shape(calculation.starting_density())
        assert residual.tau_shape == density
        assert residual.tau_shape[0] == expected == calculation.nspin_mag


def test_the_thomas_fermi_guess_scales_by_nspin_mag(pseudo_dir):
    """A one-channel density takes the one-channel form, whatever ``nspin`` says.

    ``|2 rho|^(5/3) / 2`` applied to a *total* density is too large by
    ``2^(2/3)``, which is what a nonmagnetic spin-orbit run was started from.
    """
    from pypresso.scf.driver import _starting_tau
    from pypresso.xc.mgga import thomas_fermi_tau

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculation = _calculation(pseudo_dir, ", input_dft = 'tb09', noncolin = .true.")
    assert calculation.nspin == 4 and calculation.nspin_mag == 1

    rho = calculation.starting_density()
    assert np.asarray(_starting_tau(rho, calculation)) == pytest.approx(
        np.asarray(thomas_fermi_tau(rho, 1))
    )


def test_a_continued_run_compares_tau_against_the_densitys_shape():
    """The silent half: the shapes were unequal, so ``tau`` was dropped.

    ``driver.py`` builds the expected shape to decide whether a converged ``tau``
    may cross a ``starting_from``. Built from ``nspin`` it never matched for a
    spin-orbit run, and the fallback -- Thomas-Fermi -- costs iterations and
    says nothing.
    """
    from pypresso.scf import driver

    source = inspect.getsource(driver.run_scf)
    assert "expected = tuple(np.shape(calculation.starting_density()))" in source
    assert "expected = (calculation.nspin,)" not in source


# --- a field needs a magnetization to act on ---------------------------------


def test_a_field_without_a_starting_moment_is_refused(pseudo_dir):
    """``domag`` comes from ``starting_magnetization`` and from nothing else.

    ``pw.x`` accepts this input and the field never reaches a wavefunction:
    ``add_bfield`` writes it into ``v(:, 2:4)`` and ``vloc_psi_nc`` applies those
    channels only ``IF (domag)``. So the run converges, reports success, and is
    the field-free calculation -- which is why this is refused here rather than
    reproduced.
    """
    with pytest.raises(ValueError, match="starting_magnetization"):
        _calculation(pseudo_dir, ", noncolin = .true., b_field(3) = 0.01")


def test_a_starting_moment_is_the_way_out(pseudo_dir):
    """The refusal names a one-variable fix, so the fix has to work."""
    calculation = _calculation(
        pseudo_dir,
        ", noncolin = .true., b_field(3) = 0.01, starting_magnetization(1) = 0.1",
    )
    assert calculation.nspin_mag == 4
    assert calculation.magnetic_field is not None


def test_the_collinear_field_is_untouched(pseudo_dir):
    """``nspin = 2`` needs no ``domag`` and must not acquire the new guard."""
    calculation = _calculation(
        pseudo_dir,
        ", nspin = 2, starting_magnetization(1) = 0.1, b_field(3) = 0.01"
        ", occupations = 'smearing', degauss = 0.02",
    )
    assert calculation.magnetic_field is not None


# --- fixed occupations with two channels -------------------------------------


def test_fixed_lsda_needs_tot_magnetization():
    """``input.f90:797``, transcribed: there is no shared-Fermi fixed branch.

    The old refusal said QE fills the channels "from tot_magnetization or from a
    shared Fermi level". It does not: without ``tot_magnetization`` it stops.
    """
    with pytest.raises(ValueError, match="tot_magnetization"):
        _system(", nspin = 2, starting_magnetization(1) = 0.1, occupations = 'fixed'")


def test_fixed_lsda_needs_an_integer_tot_magnetization():
    """A channel fills a whole number of bands, so the split must be an integer."""
    with pytest.raises(ValueError, match="integer tot_magnetization"):
        _system(
            ", nspin = 2, starting_magnetization(1) = 0.1, occupations = 'fixed'"
            ", tot_magnetization = 1.5"
        )


def test_fixed_lsda_with_an_integer_magnetization_builds():
    system = _system(
        ", nspin = 2, starting_magnetization(1) = 0.1, occupations = 'fixed'"
        ", tot_magnetization = 2"
    )
    assert system.tot_magnetization == 2


def test_the_fixed_rule_is_checked_before_the_starting_magnetization_one():
    """``input.f90`` checks them at lines 784 and 1507, in that order.

    Worth pinning because both fire on the same bare ``nspin = 2`` input and the
    two messages send a reader somewhere different. QE's order is the useful one:
    an LSDA input with the default occupations has no way to fill its channels
    whatever its starting moment says, so that is the thing to fix first.
    """
    with pytest.raises(ValueError, match="tot_magnetization"):
        _system(", nspin = 2")


def test_a_band_structure_of_a_magnet_is_not_asked_for_the_split():
    """``.AND. lscf``: an nscf run fills nothing, so the rule does not apply."""
    text = _SILICON.format(extra=", nspin = 2, starting_magnetization(1) = 0.1")
    for calculation in ("nscf", "bands"):
        system = build_system(
            parse_pw_input(text.replace("calculation = 'scf'",
                                        f"calculation = {calculation!r}"))
        )
        assert system.nspin == 2


def test_the_unpolarized_refusal_still_names_the_input_variable():
    """Reached where the builder cannot see it -- through a hand-built call."""
    eigenvalues = np.zeros((2, 1, 4))
    with pytest.raises(NotImplementedError, match="tot_magnetization"):
        fixed_occupations(eigenvalues, np.ones(1), 8.0)


def test_each_channel_fills_its_own_count():
    """``iweights_only`` with ``degspin = 1``, once per channel."""
    eigenvalues = np.tile(np.arange(6.0), (2, 1, 1))
    weights = np.full(1, 0.5)
    wg, homo, lumo = fixed_occupations(
        eigenvalues, weights, 6.0, 2, counts=(4.0, 2.0)
    )
    wg = np.asarray(wg)
    assert wg.shape == (2, 1, 6)
    assert list((wg > 0).sum(axis=-1)[0]) == [4]
    assert list((wg > 0).sum(axis=-1)[1]) == [2]
    assert np.allclose(wg[wg > 0], 0.5)
    # ``ef_up`` / ``ef_dw``: each channel's own highest occupied level.
    assert np.asarray(homo) == pytest.approx([3.0, 1.0])
    assert np.asarray(lumo) == pytest.approx([4.0, 2.0])


def test_a_half_integer_channel_count_rounds_the_way_nint_does():
    """QE's ``NINT``, not a floor: an odd charge with an even moment splits 3.5."""
    eigenvalues = np.tile(np.arange(6.0), (2, 1, 1))
    wg, _, _ = fixed_occupations(
        eigenvalues, np.ones(1), 5.0, 2, counts=(3.5, 1.5)
    )
    assert list((np.asarray(wg) > 0).sum(axis=-1)[0]) == [4]
    assert list((np.asarray(wg) > 0).sum(axis=-1)[1]) == [2]


def test_a_channel_needing_more_bands_than_were_computed_is_named():
    eigenvalues = np.tile(np.arange(3.0), (2, 1, 1))
    with pytest.raises(ValueError, match="raise nbnd"):
        fixed_occupations(eigenvalues, np.ones(1), 6.0, 2, counts=(4.0, 2.0))


def test_the_residual_solver_forwards_the_channel_counts(pseudo_dir):
    """The sibling path: ``residual.py`` mirrors the driver's dispatch."""
    from pypresso.scf import residual as residual_module

    source = inspect.getsource(residual_module._weights)
    fixed = source.split('if scheme == "fixed"')[1].split("counts =")[0]
    assert "return" not in fixed.split("\n")[0]
    assert "counts=counts" in source
