"""What a k-set, a symmetry group and a converged field must carry across a call.

The second half of the capability audit of 2026-08-29. Where
``test_sibling_refusals`` collects combinations that should have *stopped* and
did not, these are combinations that ran and returned the wrong number, because
something the ground state knows did not reach the function that needed it:

* ``denser_grid`` reduced with the crystal's symmetry group whatever the run
  used, so a ``nosym`` or magnetic DOS was folded onto k-points its own SCF
  had kept apart;
* the projected DOS averaged over the same group, for the same reason;
* the topology workflows built their Hamiltonian without the Hubbard term, and
  then certified the manifold isolated using the eigenvalues that came out;
* ``with_kpoints`` skipped ``degspin``, counting every electron twice;
* ``SHARED_OPTIONS`` were documented as universal, dropped by nine methods and
  -- in the one method that did forward them -- capping a *Dyson* iteration
  with a number chosen for the SCF;
* and the field a run converged under was rebuilt from the input afterwards,
  which is a different field whenever ``reducebf`` or the fixed-spin-moment
  scheme was in use.

All unit tests: every one is a boundary, and none of them needs an SCF.
"""

import inspect

import numpy as np
import pytest

from defumat.io.pwin import parse_pw_input
from defumat.system.builder import build_system
from defumat.system.kpoints import KPoints, for_spin
from defumat.workflows.nscf import denser_grid

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


# --- the denser grid's symmetry ---------------------------------------------


def test_a_denser_grid_uses_the_runs_symmetry_and_not_the_crystals():
    """``nosym`` means ``nsym = 1``, here as in ``setup.f90``.

    A spin spiral is *required* to be ``nosym`` -- the spin space group is not
    written -- so reducing its DOS grid with the crystal's operations folds
    k-points the SCF deliberately kept apart. ``denser_grid`` called
    ``find_symmetries`` directly and never consulted the system.
    """
    assert denser_grid(_system(), (4, 4, 4)).nk == 8
    assert denser_grid(_system(", nosym = .true."), (4, 4, 4)).nk == 4 ** 3


def test_a_denser_grid_still_carries_the_spin_weight_convention():
    """The fix must not lose what the function already did right."""
    weights = denser_grid(_system(), (4, 4, 4)).weights
    assert float(np.asarray(weights).sum()) == pytest.approx(2.0)
    polarized = denser_grid(
        _system(", nspin = 2, starting_magnetization(1) = 0.1"
                ", occupations = 'smearing', degauss = 0.02"), (4, 4, 4)
    )
    assert float(np.asarray(polarized.weights).sum()) == pytest.approx(1.0)


# --- the spin weight convention, applied exactly once ------------------------


def test_for_spin_is_idempotent():
    """The division is not, so the flag is what makes the function safe.

    ``for_spin`` reaches a k-set from four directions -- ``build_system``,
    ``System._recelled_kpoints``, ``denser_grid`` and now
    ``Calculator.with_kpoints`` -- and none of them can see where the set came
    from. Applying the factor twice is the same silent error as not applying it
    once, in the other direction.
    """
    system = _system()
    raw = KPoints.automatic((2, 2, 2), (0, 0, 0), system.cell)
    once = for_spin(raw, 2)
    twice = for_spin(once, 2)

    assert float(np.asarray(raw.weights).sum()) == pytest.approx(2.0)
    assert float(np.asarray(once.weights).sum()) == pytest.approx(1.0)
    assert float(np.asarray(twice.weights).sum()) == pytest.approx(1.0)
    assert once.spin_normalized and twice.spin_normalized


def test_an_unpolarized_k_set_is_not_marked_normalized():
    """Or a set built for ``nspin = 1`` would skip the halving it needs later."""
    raw = KPoints.automatic((2, 2, 2), (0, 0, 0), _system().cell)
    assert not for_spin(raw, 1).spin_normalized
    assert float(np.asarray(for_spin(for_spin(raw, 1), 2).weights).sum()) == (
        pytest.approx(1.0)
    )


def test_with_kpoints_normalizes_a_raw_k_set_and_leaves_a_normalized_one(pseudo_dir):
    """The comparison ``with_kpoints``' own docstring recommends.

    "The unreduced grid beside its irreducible wedge … same energy, fewer
    k-points" -- and handing in a raw ``KPoints.automatic`` counted every
    electron twice, which does not fail: the Fermi level moves and the run
    integrates to the right electron count at the wrong energy.
    """
    from defumat.pseudo import read_upf

    system = _system(", nspin = 2, starting_magnetization(1) = 0.1"
                     ", occupations = 'smearing', degauss = 0.02")
    pseudos = tuple(
        read_upf(pseudo_dir / species.pseudo_file)
        for species in system.structure.species
    )
    calculator = system.calculator(pseudos=pseudos)

    raw = KPoints.automatic((4, 4, 4), (0, 0, 0), system.cell)
    assert float(
        np.asarray(calculator.with_kpoints(raw).system.kpoints.weights).sum()
    ) == pytest.approx(1.0)

    already = denser_grid(system, (4, 4, 4))
    assert float(
        np.asarray(calculator.with_kpoints(already).system.kpoints.weights).sum()
    ) == pytest.approx(1.0)


# --- the state a fixed-density run needs -------------------------------------


def test_the_topology_workflows_can_be_given_ns():
    """``hamiltonian`` was called with two positional arguments of three.

    The third is ``hubbard=None``, so a DFT+U crystal was diagonalised without
    its Hubbard term -- and ``DFTSource._check_gap`` then certified the manifold
    isolated using those eigenvalues, which is a confident integer off the wrong
    bands.
    """
    from defumat.workflows.topology import run_berry_curvature, run_z2, run_z2_3d

    for entry in (run_berry_curvature, run_z2, run_z2_3d):
        assert "ns" in inspect.signature(entry).parameters, entry.__name__


def test_a_topological_invariant_takes_the_converged_field_too():
    """The same defect as the fixed-density run below, one layer over.

    ``DFTSource`` rebuilds the potential from a frozen density, and it did that
    with no field argument at all -- so ``calculation.potential`` fell back to
    ``self.magnetic_field``, the field the *input* asked for, at full scale.
    Whenever ``reducebf`` or the fixed-spin-moment scheme had changed it, every
    eigenvalue was shifted by a field the SCF never converged under, and an
    invariant computed from those bands is still an integer.

    Found while wiring the polarization through the same source; the sibling
    refusal in ``fixed_density_states`` had been there since the 2026-08-29
    sweep and this one had not.
    """
    from defumat.workflows.polarization import run_polarization
    from defumat.workflows.topology import DFTSource, run_berry_curvature, run_z2, run_z2_3d

    for entry in (run_berry_curvature, run_z2, run_z2_3d, run_polarization):
        parameters = inspect.signature(entry).parameters
        for name in ("field", "field_scale"):
            assert name in parameters, f"{entry.__name__} is missing {name}"
    assert "field" in DFTSource.__dataclass_fields__
    assert "field_scale" in DFTSource.__dataclass_fields__
    # and the source must forward them rather than accept and drop them
    body = inspect.getsource(DFTSource.states)
    assert "self.field" in body and "self.field_scale" in body


def test_a_topological_invariant_actually_refuses_a_field_it_was_not_given():
    """The behavioural half: the guard fires rather than merely existing.

    A noncollinear silicon carrying ``b_field(3)`` reaches ``DFTSource.states``
    and is refused before anything is diagonalised. Without the guard the run
    would have built its potential from the *input* field at full scale and
    returned bands, which is the failure that has no symptom.
    """
    from pathlib import Path

    from defumat.pseudo.upf import read_upf
    from defumat.workflows.topology import DFTSource

    cases = Path(__file__).resolve().parents[1] / "data" / "qe"
    pseudo = Path(__file__).resolve().parents[1] / "data" / "pseudo"
    text = (cases / "si2-nosym.in").read_text().replace(
        "ecutwfc = 12.0, nosym = .true.",
        "ecutwfc = 12.0, nosym = .true., noncolin = .true.,\n"
        "    starting_magnetization(1) = 0.3, b_field(3) = 0.01,")
    system = build_system(parse_pw_input(text))
    pseudos = tuple(read_upf(pseudo / sp.pseudo_file)
                    for sp in system.structure.species)

    source = DFTSource(system=system, pseudos=pseudos,
                       density=np.zeros((4, 8, 8, 8)), nocc=4)
    with pytest.raises(ValueError, match="field the SCF ended with"):
        source.states(np.array([[0.0, 0.0, 0.0]]))


def test_a_fixed_density_run_takes_the_whole_mixed_state():
    """Including the field, which the input does not describe after reducebf."""
    from defumat.workflows.nscf import fixed_density_states, run_nscf
    from defumat.workflows.bands import run_bands
    from defumat.workflows.dos import run_dos

    for entry in (fixed_density_states, run_nscf, run_bands, run_dos):
        parameters = inspect.signature(entry).parameters
        for name in ("ns", "tau", "becsum", "field", "field_scale"):
            assert name in parameters, f"{entry.__name__} is missing {name}"


def test_the_projected_dos_forwards_becsum_and_tau():
    """P38's defect, surviving in the sibling workflow P38 did not touch.

    ``run_pdos`` passed nine positional arguments into a longer signature, so a
    PAW or meta-GGA projected DOS on a denser grid stopped on
    ``fixed_density_states``' own refusal -- with no argument to satisfy it.
    Everything it needs is on the ``result`` it already receives.
    """
    from defumat.workflows import pdos

    body = inspect.getsource(pdos.run_pdos)
    for name in ("becsum=", "tau=", "field=", "field_scale="):
        assert name in body, name


def test_the_converged_field_is_carried_on_the_result():
    """``reducebf`` scales a loop variable and leaves the field object alone.

    So the converged potential is reproducible only from the pair, and
    ``field_scale`` was not on the result at all.
    """
    from defumat.scf.driver import SCFResult

    assert "field_scale" in SCFResult.__dataclass_fields__
    assert "magnetic_field" in SCFResult.__dataclass_fields__


def test_the_state_arguments_map_parameter_names_to_result_attributes():
    """``field`` comes from ``magnetic_field``, so it cannot be a plain getattr."""
    from defumat.calculator import _STATE_ARGUMENTS

    assert _STATE_ARGUMENTS["field"] == "magnetic_field"
    assert _STATE_ARGUMENTS["field_scale"] == "field_scale"


# --- the shared options ------------------------------------------------------


def test_scf_only_options_are_not_forwarded_past_the_scf():
    """``max_iterations`` is three different loops in three different callees.

    The SCF's in ``run_scf``, the self-consistent response's in
    ``dielectric_tensor``, and the Dyson fixed point's in ``run_absorption`` --
    which *was* being fed the calculator's SCF value, silently.
    """
    from defumat.calculator import SCF_ONLY_OPTIONS, SHARED_OPTIONS

    assert SCF_ONLY_OPTIONS < SHARED_OPTIONS
    assert "max_iterations" in SCF_ONLY_OPTIONS
    assert "mixing_mode" in SCF_ONLY_OPTIONS
    # What means the same thing wherever it is named stays shared.
    for name in ("nbnd", "k_batch", "verbose", "diagonalization", "conv_thr"):
        assert name not in SCF_ONLY_OPTIONS, name


def test_the_response_methods_forward_the_options_they_name():
    """Nine methods dropped them entirely, against a docstring promising
    they are "applied to every method that names them"."""
    from defumat.calculator import Calculator

    methods = [
        "get_dielectric_tensor", "get_phonons", "get_raman_tensors",
        "get_vibrational_spectrum", "get_strain_response",
        "get_elastic_constants", "get_electrostriction",
    ]
    for name in methods:
        body = inspect.getsource(getattr(Calculator, name))
        assert "_defaults_for" in body, name
        assert "SCF_ONLY_OPTIONS" in body, name

    absorption = inspect.getsource(Calculator.get_absorption)
    assert "SCF_ONLY_OPTIONS" in absorption


# --- the switch that was read by nothing -------------------------------------


def test_qcutz_is_refused_rather_than_ignored():
    """`_REFUSED_SWITCHES`' own criterion, applied to a switch it had missed.

    QE's smoothed constant-cutoff kinetic functional (``g2_kin.f90``) was read
    by nothing here, so such an input ran with the plain ``G^2`` and converged
    somewhere else. Refused rather than implemented because every vendored input
    that sets it is a ``CPV/`` case -- there is no ``pw.x`` benchmark to check
    an implementation against.
    """
    for variable in ("qcutz = 1.0", "qcutz = 1.0, ecfixed = 8.0, q2sigma = 2.0"):
        with pytest.raises(NotImplementedError, match="g2_kin"):
            _system(", " + variable)


def test_a_zero_qcutz_is_not_asked_for():
    """QE's own sentinel is ``qcutz > 0``, so the default must pass through."""
    assert _system(", qcutz = 0.0").ecutwfc == 12.0
