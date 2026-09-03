"""What a derived :class:`Calculation` must rebuild, and what it must refuse.

Three defects with one shape: a quantity that depends on the geometry or on the
regime was computed once and then carried, unchanged and unchecked, into a
calculation it no longer described. None of them raised; each returned a number.

* the magnetic field's integration spheres survived ``at_positions``, so a
  relaxation under ``constrained_magnetization = 'atomic'`` integrated the
  penalty over spheres centred where the atoms started;
* a spin spiral built without going through the input reader kept the crystal's
  symmetry group, which is not the spiral's, and symmetrised with it;
* ``calculation`` was parsed and read by nobody, so ``md`` and ``vc-relax`` ran
  as a plain SCF and reported success.
"""

import dataclasses
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat.io.pwin import parse_pw_input
from defumat.pseudo import read_upf
from defumat.scf.driver import Calculation
from defumat.system.builder import build_system

pytestmark = pytest.mark.unit


def _build(text, pseudo_dir):
    system = build_system(parse_pw_input(text))
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    return system, Calculation(system, pseudos)


_CONSTRAINED = """
&control
  calculation = 'scf'
/
&system
  ibrav = 1, celldm(1) = 9.0, nat = 2, ntyp = 1,
  ecutwfc = 20.0, ecutrho = 160.0,
  noncolin = .true., starting_magnetization(1) = 0.4,
  constrained_magnetization = 'atomic', lambda = 1.0,
  occupations = 'smearing', smearing = 'gaussian', degauss = 0.02
/
&electrons
/
ATOMIC_SPECIES
 Fe 55.845 Fe.pz-nd-rrkjus.UPF
ATOMIC_POSITIONS crystal
 Fe 0.00 0.00 0.00
 Fe 0.50 0.50 0.50
K_POINTS automatic
 2 2 2 0 0 0
"""


def test_local_regions_follow_the_atoms(pseudo_dir):
    """``make_pointlists`` again after a concrete move, not the starting spheres."""
    system, calculation = _build(_CONSTRAINED, pseudo_dir)
    field = calculation.magnetic_field
    assert field is not None and field.regions is not None

    before = np.asarray(field.regions.weights)
    shifted = np.asarray(system.structure.positions) + np.array([0.8, 0.0, 0.0])
    after = np.asarray(
        calculation.at_positions(jnp.asarray(shifted)).magnetic_field.regions.weights
    )
    assert np.abs(after - before).max() > 1e-6, "the spheres did not move with the atoms"


def test_local_regions_are_frozen_while_differentiating(pseudo_dir):
    """...and a *traced* move keeps them, since the assignment is host-side.

    The sphere a grid point belongs to is piecewise constant in the positions,
    so its derivative vanishes away from the crossings and freezing it is exact
    between them -- the trade the spiral's plane-wave sphere makes (P21). What
    matters here is that the force path still differentiates at all.
    """
    system, calculation = _build(_CONSTRAINED, pseudo_dir)

    def total(positions):
        return jnp.sum(calculation.at_positions(positions).vltot)

    gradient = jax.grad(total)(jnp.asarray(system.structure.positions))
    assert np.all(np.isfinite(np.asarray(gradient)))


#: Norm-conserving on purpose: an ultrasoft or PAW spiral is refused for its own
#: reason (``q_ij(q)``), which would mask the one under test here.
_NC_NONCOLLINEAR = """
&control
  calculation = 'scf'
/
&system
  ibrav = 1, celldm(1) = 6.0, nat = 1, ntyp = 1, ecutwfc = 15.0,
  noncolin = .true., starting_magnetization(1) = 0.5,
  occupations = 'smearing', smearing = 'gaussian', degauss = 0.02
/
&electrons
/
ATOMIC_SPECIES
 H 1.008 H.pz-vbc.UPF
ATOMIC_POSITIONS crystal
 H 0.00 0.00 0.00
K_POINTS automatic
 2 2 2 0 0 0
"""


def test_a_spiral_built_without_the_reader_still_refuses_symmetry(pseudo_dir):
    """The guard belongs where the calculation is built, not only in the reader.

    ``at_spiral_q`` and ``workflows.spiral`` install ``spiral_q`` through
    ``dataclasses.replace``, which never passes ``build_system`` -- and a
    ``System`` can simply be constructed. Symmetrising a spiral averages the
    rotated-frame magnetization over operations of the crystal point group that
    are not symmetries of the spiral, and reduces the k-set to the same wrong
    wedge, converging to an energy that is silently wrong.
    """
    system, _ = _build(_NC_NONCOLLINEAR, pseudo_dir)
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    spiral = dataclasses.replace(system, spiral_q=(0.0, 0.0, 0.25))
    assert not spiral.nosym, "the point of this case is that symmetry is on"
    with pytest.raises(NotImplementedError, match="nosym"):
        Calculation(spiral, pseudos)


@pytest.mark.parametrize("mode", ["md", "vc-md"])
def test_a_calculation_without_a_driver_is_refused(mode):
    """Not run as a plain SCF, which reported success for a run that never was."""
    text = _CONSTRAINED.replace("calculation = 'scf'", f"calculation = '{mode}'")
    with pytest.raises(NotImplementedError, match="calculation"):
        build_system(parse_pw_input(text))


def test_vc_relax_has_a_driver_now_and_reads_its_cell_namelist():
    """P29 gave ``vc-relax`` a driver, so it must stop being refused.

    The other half of the same statement, and the half that would rot
    silently: the ``&cell`` namelist has to reach
    :attr:`System.relax`. A ``vc-relax`` that parsed and then ignored its
    ``press`` would relax to the zero-pressure cell and report the pressure it
    was asked for -- which is P28b's ``forc_conv_thr`` gap one namelist over.
    """
    text = _CONSTRAINED.replace("calculation = 'scf'", "calculation = 'vc-relax'")
    text += "\n &cell\n   press = 250.0\n   cell_dofree = 'xyz'\n /\n"
    system = build_system(parse_pw_input(text))
    assert system.calculation == "vc-relax"
    assert system.relax.press == 250.0
    assert system.relax.cell_dofree == "xyz"
    # ``read_namelists.f90`` resets ``cell_dynamics`` to 'bfgs' for a pw.x
    # vc-relax, where the bare namelist default is 'none'.
    assert system.relax.cell_dynamics == "bfgs"


def test_an_unknown_calculation_is_rejected():
    """``pw.x`` stops on one; a typo must not be read as ``scf``."""
    text = _CONSTRAINED.replace("calculation = 'scf'", "calculation = 'scv'")
    with pytest.raises(ValueError, match="calculation"):
        build_system(parse_pw_input(text))


# --- the group a relaxation checks against ---------------------------------
#
# ``Calculation.symmetries`` is the full detected group whatever the input said,
# with ``use_symmetry`` beside it as the switch. ``workflows/relax.py`` used to
# check the group rather than the switch, which refuses a valid ionic step under
# ``nosym`` -- exactly what an adsorbate on a surface does at its first
# displacement, and the whole reason such a run sets ``nosym`` in the first
# place.


_RELAX_SILICON = """
&control
  calculation = 'relax'
/
&system
  ibrav = 2, celldm(1) = 10.20, nat = 2, ntyp = 1, ecutwfc = 12.0{extra}
/
&electrons
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""


def _displaced(pseudo_dir, extra):
    """A calculation and the same one with one atom moved off every symmetry."""
    import warnings

    import jax.numpy as jnp
    import numpy as np

    from defumat.calculator import Calculator

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = Calculator.from_text(
            _RELAX_SILICON.format(extra=extra), pseudo_dir, announce=False
        ).calculation
    moved = np.asarray(built.system.structure.positions).copy()
    moved[0, 0] += 0.13
    return built, built.at_positions(jnp.asarray(moved))


def test_a_symmetric_run_still_refuses_a_step_that_breaks_its_group(pseudo_dir):
    """The protective half of the gate, which must keep working.

    The FFT grid and the k-set were chosen for this group, so a step that
    leaves it is a bug and has to be caught.
    """
    from defumat.system.symmetry import check_symmetry

    built, moved = _displaced(pseudo_dir, "")
    assert built.use_symmetry
    assert not check_symmetry(
        moved.system.cell, moved.system.structure, moved.symmetries
    )


def test_nosym_does_not_check_a_group_it_never_applies(pseudo_dir):
    """The fix: under ``nosym`` the same step must be allowed.

    The group is still *detected* -- it is a property of the crystal, and
    ``basis/builder.py`` wants its fractional translations -- so the thing that
    distinguishes the two cases is ``use_symmetry`` and nothing else. Under
    ``nosym`` the FFT grid takes ``fft_fact = (1, 1, 1)`` and the k-set is not
    reduced, so nothing the check protects is at risk.
    """
    from defumat.system.symmetry import check_symmetry

    built, moved = _displaced(pseudo_dir, ", nosym = .true.")
    assert not built.use_symmetry
    # The group is detected all the same, and the displaced structure does not
    # have it -- so a check written against the group alone would raise here.
    assert built.symmetries.nsym > 1
    assert not check_symmetry(
        moved.system.cell, moved.system.structure, moved.symmetries
    )
    # ... and the gate as written does not.
    assert not (moved.use_symmetry and not check_symmetry(
        moved.system.cell, moved.system.structure, moved.symmetries
    ))
