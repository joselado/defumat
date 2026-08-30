"""The facade must be a shortcut, never a second implementation.

Every test here is one of three claims, and only the first is about brevity:

* what the bound method computes is what the functional entry point computes,
  bit for bit -- there is no physics in :mod:`pypresso.calculator`;
* the pieces of the mixed state that cannot be rebuilt from the density
  (``ns``, ``tau``, ``becsum``) are supplied automatically. The entry points
  already *refuse* without them rather than computing something else, so what
  this closes is a stopped run and a puzzle, not a wrong number;
* a derived calculator does not serve its parent's ground state. This is the
  defect ``test_geometry_invalidation`` documents one layer down: a quantity
  that depended on the geometry, carried unchanged into a calculation it no
  longer described, returning a number rather than raising.
"""

import inspect

import numpy as np
import pytest

from pypresso.calculator import (SCF_ONLY_OPTIONS, SHARED_OPTIONS,
                                 _ELECTRONS_OPTIONS, Calculator,
                                 electrons_defaults)
from pypresso.io.pwin import parse_pw_input
from pypresso.pseudo import read_upf
from pypresso.scf.driver import run_scf
from pypresso.system.builder import build_system
from pypresso.system.kpoints import KPoints

pytestmark = pytest.mark.unit


SILICON = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.20, nat = 2, ntyp = 1, ecutwfc = 12.0
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

#: Silicon with a PAW dataset, whose ``becsum`` is a property of the states and
#: is what a hand-threaded call has to remember.
SILICON_PAW = SILICON.replace(
    "Si 28.086 Si.pz-vbc.UPF", "Si 28.086 Si.pz-n-kjpaw_psl.0.1.UPF"
).replace("ecutwfc = 12.0", "ecutwfc = 20.0, ecutrho = 120.0")


@pytest.fixture(scope="module")
def silicon(pseudo_dir):
    return Calculator.from_text(SILICON, pseudo_dir, announce=False)


def test_from_text_loads_the_pseudopotentials_the_card_names(silicon):
    assert len(silicon.pseudos) == 1
    assert silicon.pseudos[0].element.strip() == "Si"


def test_from_file_defaults_pseudo_dir_to_the_inputs_own_directory(tmp_path,
                                                                   pseudo_dir):
    (tmp_path / "scf.in").write_text(SILICON)
    (tmp_path / "Si.pz-vbc.UPF").write_bytes(
        (pseudo_dir / "Si.pz-vbc.UPF").read_bytes()
    )
    calc = Calculator.from_file(tmp_path / "scf.in", announce=False)
    assert calc.pseudos[0].element.strip() == "Si"


def test_a_missing_file_names_the_species_that_asked_for_it(tmp_path):
    (tmp_path / "scf.in").write_text(SILICON)
    with pytest.raises(FileNotFoundError, match="Si.*Si.pz-vbc.UPF"):
        Calculator.from_file(tmp_path / "scf.in")


def test_an_unknown_constructor_option_is_refused_by_name(silicon, pseudo_dir):
    # The ergonomic risk of ``**defaults`` is that a typo becomes a silently
    # ignored setting; it is a TypeError instead.
    with pytest.raises(TypeError, match="conv_th"):
        Calculator.from_text(SILICON, pseudo_dir, conv_th=1.0e-8)


def test_shared_options_are_all_real_parameters_of_run_scf():
    # SHARED_OPTIONS is filtered against each entry point's signature, so a
    # name that no longer exists would simply stop being forwarded rather than
    # raising anywhere. This is the check that keeps it from going stale.
    parameters = set(inspect.signature(run_scf).parameters)
    assert SHARED_OPTIONS <= parameters


def test_the_bound_method_reproduces_the_functional_entry_point(silicon,
                                                                pseudo_dir):
    system = build_system(parse_pw_input(SILICON))
    pseudos = (read_upf(pseudo_dir / "Si.pz-vbc.UPF"),)
    direct = run_scf(system, pseudos)
    assert silicon.get_scf().total_energy == pytest.approx(direct.total_energy,
                                                           abs=1e-10)


def test_the_cache_is_one_slot_and_options_key_it(silicon):
    first = silicon.get_scf()
    assert silicon.get_scf() is first, "the same options must not rerun"
    tighter = silicon.get_scf(conv_thr=1.0e-10)
    assert tighter is not first, "different options must rerun"
    assert silicon.scf_result is tighter, "and must replace the slot"
    # Restore the module-scoped fixture's cheaper state for the tests below.
    silicon.get_scf()


def test_reading_the_cache_never_starts_a_run(pseudo_dir):
    calc = Calculator.from_text(SILICON, pseudo_dir, announce=False)
    assert calc.scf_result is None
    assert calc.converged is False
    assert calc._calculation is None, "nor may it build the basis"


def test_an_implicit_scf_announces_itself(pseudo_dir, capsys):
    calc = Calculator.from_text(SILICON, pseudo_dir)
    calc.get_forces()
    assert "running the SCF" in capsys.readouterr().err


def test_an_unconverged_ground_state_is_refused_rather_than_used(pseudo_dir):
    calc = Calculator.from_text(SILICON, pseudo_dir, announce=False)
    calc.get_scf(max_iterations=1, conv_thr=1.0e-14)
    assert not calc.converged
    with pytest.raises(ValueError, match="converged ground state"):
        calc.get_forces()


def test_bands_carry_the_scfs_own_zero(silicon):
    path = KPoints.band_path([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], [4, 1],
                             silicon.system.cell, crystal=True)
    bands = silicon.get_bands(kpoints=path)
    result = silicon.scf_result
    # Hand-threaded, this is the argument everyone omits and the band plot
    # then has no zero.
    assert bands.homo == pytest.approx(result.homo)


def test_stress_is_reused_when_the_scf_already_computed_it(pseudo_dir):
    calc = Calculator.from_text(
        SILICON.replace("calculation = 'scf'", "calculation = 'scf'\n  tprnfor = .true.\n  tstress = .true."),
        pseudo_dir, announce=False,
    )
    result = calc.get_scf()
    assert result.stress is not None
    assert calc.get_stress() is result.stress, "it must not differentiate twice"


@pytest.mark.slow
def test_the_paw_becsum_is_supplied_without_being_asked_for(pseudo_dir):
    """The threading the facade takes over, on the dataset that needs it.

    ``becsum`` is a property of the states and cannot be recovered from the
    density, and ``fixed_density_bands`` refuses without it rather than
    dropping ``ddd_paw`` -- the refusal holds, which is the point of having
    it. What the facade changes is that the caller never meets it: the bound
    method passes the whole converged state, so the run that raises by hand
    simply works.
    """
    from pypresso.workflows.bands import run_bands

    calc = Calculator.from_text(SILICON_PAW, pseudo_dir, announce=False)
    result = calc.get_scf()
    assert result.becsum, "a PAW run must produce one"

    path = KPoints.band_path([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], [3, 1],
                             calc.system.cell, crystal=True)
    through_facade = calc.get_bands(kpoints=path)

    with pytest.raises(NotImplementedError, match="becsum"):
        run_bands(calc.system, calc.pseudos, result.density, kpoints=path)

    by_hand = run_bands(calc.system, calc.pseudos, result.density,
                        kpoints=path, becsum=result.becsum)
    assert np.allclose(through_facade.eigenvalues, by_hand.eigenvalues)


@pytest.mark.slow
def test_a_paw_density_of_states_reaches_a_denser_grid(pseudo_dir):
    """``run_dos`` did not forward ``becsum``, so this raised.

    A denser grid is an NSCF run, which needs the whole converged state; the
    density of states was passing only the density and ``tau``, so every PAW or
    DFT+U run with ``grid=`` stopped on the refusal above. The facade is what
    made it visible -- it passes what each entry point names, and this one
    named too little.
    """
    calc = Calculator.from_text(SILICON_PAW, pseudo_dir, announce=False)
    dos = calc.get_dos(grid=(2, 2, 2))
    assert dos.dos.shape == dos.energies.shape
    assert np.all(dos.dos >= 0.0)


def test_a_derived_calculator_does_not_serve_its_parents_ground_state(silicon):
    parent = silicon.get_scf()
    moved = silicon.with_positions(
        np.asarray(silicon.system.structure.positions) + np.array([0.1, 0.0, 0.0])
    )
    assert moved.scf_result is None, "the cache must not cross a geometry change"
    assert moved.starting_state is parent, "but it is a starting guess"
    assert silicon.scf_result is parent, "and the parent is left alone"


def test_a_derived_calculator_shares_the_pseudos_and_the_options(pseudo_dir):
    calc = Calculator.from_text(SILICON, pseudo_dir, announce=False, nbnd=6)
    derived = calc.with_cell(np.asarray(calc.system.cell.at) * 1.01)
    assert derived.pseudos is calc.pseudos
    assert derived.defaults == calc.defaults
    assert derived.scf_result is None


def test_with_spin_promotes_into_the_targets_regime(silicon):
    silicon.get_scf()
    polarized = silicon.with_spin(2, starting_magnetization=(0.3,))
    assert polarized.system.nspin == 2
    assert polarized.scf_result is None
    assert polarized.starting_state is silicon.scf_result


def test_the_repr_says_what_it_is_without_printing_arrays(silicon):
    text = repr(silicon.get_scf())
    assert "SCFResult" in text and "Ry" in text
    assert len(text) < 200, "the generated dataclass repr is screens of numbers"
    assert "Si2" in repr(silicon)


def test_every_get_method_is_a_delegation():
    """No physics in the facade.

    A method here that grew a computation of its own would be a second
    implementation of something already validated against QE -- exactly what
    the package's cross-checks exist to prevent. Length is a crude proxy and a
    deliberately generous one.
    """
    for name, method in inspect.getmembers(Calculator, inspect.isfunction):
        if not name.startswith("get_"):
            continue
        body = inspect.getsource(method)
        code = [line for line in body.splitlines()
                if line.strip() and not line.strip().startswith("#")]
        assert len(code) < 30, f"{name} is doing too much to be a delegation"


def test_a_per_call_setup_option_rebuilds_the_calculation(pseudo_dir):
    """``diagonalization`` and ``k_batch`` were a silent no-op per call.

    ``run_scf`` ignores both when it is handed a ``Calculation``, and the
    calculator always hands it one -- so a per-call value did nothing while
    still counting as a cache miss, which is the same run again under a
    different name. They rebuild the setup instead.
    """
    calc = Calculator.from_text(SILICON, pseudo_dir, announce=False)
    calc.get_scf()
    built = calc.calculation
    assert calc.defaults.get("k_batch") is None

    calc.get_scf(k_batch=1)
    assert calc.calculation is not built, "the setup must be rebuilt"
    assert calc.defaults["k_batch"] == 1, "and the change is the calculator's"


def test_a_relaxed_calculator_does_not_claim_its_parents_scf_options(silicon):
    """Its state came out of the relaxation's loop, not out of ``get_scf``.

    Copying the parent's options onto it would let a later
    ``get_scf(conv_thr=...)`` cache-hit on a state converged to something else.
    """
    silicon.get_scf()
    derived = silicon._derived(silicon.system, scf=silicon.scf_result)
    assert derived.scf_result is silicon.scf_result
    assert derived._scf_options is None, "so any explicit get_scf reruns"


# ----------------------------------------------------------------------
# the &electrons namelist (P49)
# ----------------------------------------------------------------------

SILICON_ELECTRONS = SILICON.replace(
    "&electrons\n/",
    "&electrons\n"
    "  conv_thr = 1.0d-9\n"
    "  mixing_beta = 0.3\n"
    "  mixing_mode = 'plain'\n"
    "  electron_maxstep = 42\n"
    "/",
)


def test_the_electrons_namelist_becomes_this_calculators_defaults(pseudo_dir):
    # pw.x states how to converge a run in the input file. Nothing here read
    # that namelist, though ``&control`` has always been read -- an asymmetry,
    # not a decision, and it is what eleven notebooks wrote a ``load()`` helper
    # to work around.
    calc = Calculator.from_text(SILICON_ELECTRONS, pseudo_dir, announce=False)
    assert calc.defaults["conv_thr"] == pytest.approx(1.0e-9)
    assert calc.defaults["mixing_beta"] == pytest.approx(0.3)
    assert calc.defaults["mixing_mode"] == "plain"
    # renamed on the way in: ``max_iterations`` means three different loops
    # here, and the input file's number is unambiguously the SCF's.
    assert calc.defaults["max_iterations"] == 42


def test_an_explicit_option_still_wins_over_the_input_file(pseudo_dir):
    calc = Calculator.from_text(SILICON_ELECTRONS, pseudo_dir,
                                announce=False, conv_thr=1.0e-12)
    assert calc.defaults["conv_thr"] == pytest.approx(1.0e-12)
    assert calc.defaults["mixing_beta"] == pytest.approx(0.3)   # untouched


def test_an_absent_variable_is_absent_rather_than_defaulted(pseudo_dir):
    # Whatever ``run_scf`` already defaults to keeps deciding: inventing a
    # default here would hide that QE's are context-dependent.
    calc = Calculator.from_text(SILICON, pseudo_dir, announce=False)
    assert not set(_ELECTRONS_OPTIONS.values()) & set(calc.defaults)


def test_from_file_adopts_the_namelist_too(tmp_path, pseudo_dir):
    (tmp_path / "scf.in").write_text(SILICON_ELECTRONS)
    (tmp_path / "Si.pz-vbc.UPF").write_bytes(
        (pseudo_dir / "Si.pz-vbc.UPF").read_bytes()
    )
    calc = Calculator.from_file(tmp_path / "scf.in", announce=False)
    assert calc.defaults["conv_thr"] == pytest.approx(1.0e-9)


def test_every_adopted_electrons_option_is_one_the_calculator_accepts():
    # The mapping is a claim about *this* code: an entry naming an option no
    # method forwards would be silently dropped rather than refused.
    assert set(_ELECTRONS_OPTIONS.values()) <= SHARED_OPTIONS
    # and all five describe the SCF loop rather than a response loop, which is
    # what makes reading them off the SCF's own input file correct.
    assert set(_ELECTRONS_OPTIONS.values()) - {"conv_thr"} <= SCF_ONLY_OPTIONS


def test_diagonalization_is_read_by_pw_x_and_deliberately_not_adopted(pseudo_dir):
    # Valid pw.x input, and this package has one eigensolver. Adopting it would
    # turn a run that works into a ValueError; mapping it onto Davidson would be
    # the silent substitution the package refuses elsewhere. So it is neither.
    text = SILICON.replace("&electrons\n/",
                           "&electrons\n  diagonalization = 'cg'\n/")
    calc = Calculator.from_text(text, pseudo_dir, announce=False)
    assert "diagonalization" not in calc.defaults
    assert calc.calculation is not None          # the run is still reachable


def test_a_refused_mixing_mode_is_refused_by_name_rather_than_substituted(
        pseudo_dir):
    # local-TF is QE's approx_screening2 and is not implemented. Adopting the
    # namelist must surface that refusal, not quietly run the uniform one.
    text = SILICON.replace("&electrons\n/",
                           "&electrons\n  mixing_mode = 'local-TF'\n/")
    calc = Calculator.from_text(text, pseudo_dir, announce=False)
    assert calc.defaults["mixing_mode"] == "local-TF"
    with pytest.raises(NotImplementedError, match="approx_screening2"):
        calc.get_scf()


def test_the_adopted_namelist_reaches_the_run(pseudo_dir):
    # Not just stored: an SCF told to stop after one iteration stops after one.
    text = SILICON.replace("&electrons\n/",
                           "&electrons\n  electron_maxstep = 1\n/")
    calc = Calculator.from_text(text, pseudo_dir, announce=False)
    result = calc.get_scf()
    assert result.iterations == 1 and not result.converged
