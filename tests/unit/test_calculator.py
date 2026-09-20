"""The facade must be a shortcut, never a second implementation.

Every test here is one of three claims, and only the first is about brevity:

* what the bound method computes is what the functional entry point computes,
  bit for bit -- there is no physics in :mod:`defumat.calculator`;
* the pieces of the mixed state that cannot be rebuilt from the density
  (``ns``, ``tau``, ``becsum``) are supplied automatically. The entry points
  already *refuse* without them rather than computing something else, so what
  this closes is a stopped run and a puzzle, not a wrong number;
* a derived calculator does not serve its parent's ground state. This is the
  defect ``test_geometry_invalidation`` documents one layer down: a quantity
  that depended on the geometry, carried unchanged into a calculation it no
  longer described, returning a number rather than raising.
"""

import ast
import inspect
import textwrap

import numpy as np
import pytest

from defumat.calculator import (SCF_ONLY_OPTIONS, SHARED_OPTIONS,
                                 _ELECTRONS_OPTIONS, Calculator,
                                 electrons_defaults)
from defumat.io.pwin import parse_pw_input
from defumat.pseudo import read_upf
from defumat.scf.driver import run_scf
from defumat.system.builder import build_system
from defumat.system.kpoints import KPoints

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


def test_from_file_reads_the_inputs_own_pseudo_dir(tmp_path, pseudo_dir):
    """``&control``'s ``pseudo_dir`` is what ``pw.x`` resolves the card against.

    It was parsed and read by nothing, so an input whose pseudopotentials live
    only where it says -- which is how QE's own test-suite is arranged,
    ``pseudo_dir = '../../pseudo'`` -- raised on a file ``pw.x`` runs.
    """
    (tmp_path / "held").mkdir()
    (tmp_path / "held" / "Si.pz-vbc.UPF").write_bytes(
        (pseudo_dir / "Si.pz-vbc.UPF").read_bytes()
    )
    (tmp_path / "scf.in").write_text(
        SILICON.replace("&control", "&control\n  pseudo_dir = 'held'")
    )
    calc = Calculator.from_file(tmp_path / "scf.in", announce=False)
    assert calc.pseudos[0].element.strip() == "Si"


def test_a_pseudo_dir_that_does_not_hold_the_file_falls_back(tmp_path, pseudo_dir):
    """The resolution is per *file*, so this code's own default stays behind it.

    A relative ``pseudo_dir`` written on someone else's machine is the ordinary
    case in a committed input, and refusing to look beside the input file would
    break every one of them for no gain. A file missing from both places names
    both.
    """
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "Si.pz-vbc.UPF").write_bytes(
        (pseudo_dir / "Si.pz-vbc.UPF").read_bytes()
    )
    (tmp_path / "scf.in").write_text(
        SILICON.replace("&control", "&control\n  pseudo_dir = 'elsewhere'")
    )
    assert Calculator.from_file(
        tmp_path / "scf.in", announce=False
    ).pseudos[0].element.strip() == "Si"

    (tmp_path / "Si.pz-vbc.UPF").unlink()
    with pytest.raises(FileNotFoundError, match="elsewhere.*Si.pz-vbc.UPF"):
        Calculator.from_file(tmp_path / "scf.in", announce=False)


def test_relaxed_reruns_when_its_options_change(pseudo_dir):
    """The one cached quantity that had no options key at all.

    ``relaxed()`` then ``relaxed(forc_conv_thr=1e-5)`` returned the calculator
    built on the *loose* relaxation and never passed the threshold to
    ``run_relax``, with nothing on the result saying which relaxation it came
    from -- and a dynamical matrix computed next sits at a geometry carrying the
    first one's residual forces, where the acoustic sum rule is an atom sum and
    blind to exactly that.
    """
    from types import SimpleNamespace

    from defumat.calculator import _relax_entry_point

    calc = Calculator.from_text(SILICON, pseudo_dir, announce=False)
    calls = []

    def fake_get_relax(variable_cell=False, **options):
        calls.append(options)
        calc._relax = SimpleNamespace(converged=True, system=calc.system,
                                      scf=None, scf_in_relaxed_basis=True)
        calc._relax_variable_cell = variable_cell
        calc._relax_options = calc._defaults_for(
            _relax_entry_point(variable_cell), options
        )
        return calc._relax

    calc.get_relax = fake_get_relax
    calc.relaxed()
    calc.relaxed()
    assert len(calls) == 1, "the same options must not rerun"
    calc.relaxed(forc_conv_thr=1.0e-5)
    assert len(calls) == 2 and calls[-1]["forc_conv_thr"] == 1.0e-5
    calc.relaxed(variable_cell=True)
    assert len(calls) == 3, "and neither does the other kind of relaxation"


def test_the_strain_response_is_keyed_by_the_options_that_filled_it(pseudo_dir,
                                                                   monkeypatch):
    """``or options`` is not a key: it lets a later call reuse an earlier one's.

    A slot filled at ``tr2 = 1e-6`` came straight back out of
    ``get_elastic_constants``, which asks for the response with no options at
    all, and nothing on the result recorded which tolerance it was built from.
    """
    from types import SimpleNamespace

    import defumat.response.strain as strain_module

    calls = []

    def fake(calculation, psi, eigenvalues, density, becsum, **options):
        calls.append(options)
        return f"response {len(calls)}"

    monkeypatch.setattr(strain_module, "strain_response", fake)
    calc = Calculator.from_text(SILICON, pseudo_dir, announce=False)
    monkeypatch.setattr(
        calc, "_ground_state",
        lambda *args, **kwargs: SimpleNamespace(
            wavefunctions=None, eigenvalues=None, density=None, becsum=(),
        ),
    )

    first = calc.get_strain_response(tr2=1.0e-6)
    assert len(calls) == 1 and calls[0]["tr2"] == 1.0e-6
    plain = calc.get_strain_response()
    assert plain is not first, "a call with no options is not that call"
    assert len(calls) == 2 and "tr2" not in calls[1]
    assert calc.get_strain_response() is plain, "and its own repeat is cached"
    assert len(calls) == 2


def test_the_relaxed_anisotropy_gets_the_shared_scf_options(pseudo_dir,
                                                            monkeypatch):
    """A ``**kwargs`` signature is not permission to forward everything.

    It is also not a reason to forward *nothing*, which is what the strict
    filter did here: an input whose ``&electrons`` sets ``mixing_beta = 0.2``
    because the magnetic cell diverges at 0.7 ran its noncollinear SCFs at the
    library default. ``conv_thr`` is the one held back, because
    ``run_relaxed_direction`` tightens it to 1e-10 on purpose.
    """
    import defumat.workflows.anisotropy as anisotropy

    seen = {}

    def fake(system, pseudos, directions=None, **options):
        seen.update(options)
        return "anisotropy"

    monkeypatch.setattr(anisotropy, "run_relaxed_anisotropy", fake)
    calc = Calculator.from_text(SILICON, pseudo_dir, announce=False,
                                mixing_beta=0.2, k_batch=1, conv_thr=1.0e-6)
    assert calc.get_relaxed_anisotropy(directions="xz") == "anisotropy"
    assert seen["mixing_beta"] == 0.2
    assert seen["k_batch"] == 1
    assert "conv_thr" not in seen


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
    from defumat.workflows.bands import run_bands

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

    **The docstring is not counted.** It was, and the first method to fail this
    test failed it on documentation: ``get_vertical_transport``'s body is 11
    lines under a budget of 30, and its 22-line docstring took the total to 31.
    Explaining what a quantity is has nothing to do with computing it here, so
    counting the two together measured the wrong thing and pushed the wrong way.
    The budget on the body alone is 20, measured as the span from the first
    statement after the docstring to the last. That is tighter than the old rule
    was for every method in the class: the longest span is
    ``get_scf``'s 17, and ``get_vertical_transport``'s is 12.
    """
    for name, method in inspect.getmembers(Calculator, inspect.isfunction):
        if not name.startswith("get_"):
            continue
        tree = ast.parse(textwrap.dedent(inspect.getsource(method))).body[0]
        statements = tree.body
        if (statements and isinstance(statements[0], ast.Expr)
                and isinstance(statements[0].value, ast.Constant)
                and isinstance(statements[0].value.value, str)):
            statements = statements[1:]
        assert statements, f"{name} has no body at all"
        span = statements[-1].end_lineno - statements[0].lineno + 1
        assert span <= 20, (
            f"{name} is {span} lines of body, budget 20 -- too much to be a "
            f"delegation"
        )


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
    # ... and each describes the SCF loop or the Calculation underneath it,
    # rather than a *response* loop -- which is what makes reading them off the
    # SCF's own input file correct. The two kinds are both legitimate and are
    # not the same thing: `max_iterations` is a run's own iteration count and
    # would mean something different to a response solver, so it must not be
    # forwarded past the SCF; `david` is `diago_david_ndim`, which belongs to
    # the eigensolver and so to the `Calculation`, and is therefore never
    # forwarded as a run keyword at all -- giving it per call rebuilds the
    # setup instead.
    kinds = SCF_ONLY_OPTIONS | set(Calculator.SETUP_OPTIONS)
    assert set(_ELECTRONS_OPTIONS.values()) - {"conv_thr"} <= kinds


def test_diagonalization_is_read_by_pw_x_and_deliberately_not_adopted(pseudo_dir):
    # Valid pw.x input, and this package has one eigensolver. Adopting it would
    # turn a run that works into a ValueError; mapping it onto Davidson would be
    # the silent substitution the package refuses elsewhere. So it is neither.
    text = SILICON.replace("&electrons\n/",
                           "&electrons\n  diagonalization = 'cg'\n/")
    calc = Calculator.from_text(text, pseudo_dir, announce=False)
    assert "diagonalization" not in calc.defaults
    assert calc.calculation is not None          # the run is still reachable


def test_local_tf_is_adopted_from_the_namelist_and_runs(pseudo_dir):
    # local-TF is QE's ``approx_screening2`` and used to be refused here. It is
    # implemented now, so an input naming it must *run* it -- and reach the same
    # fixed point as any other mixer, since a preconditioner changes the path
    # and not the answer.
    text = SILICON.replace("&electrons\n/",
                           "&electrons\n  mixing_mode = 'local-TF'\n/")
    calc = Calculator.from_text(text, pseudo_dir, announce=False)
    assert calc.defaults["mixing_mode"] == "local-TF"
    local = calc.get_scf()
    assert local.converged
    plain = Calculator.from_text(SILICON, pseudo_dir, announce=False).get_scf()
    assert local.total_energy == pytest.approx(plain.total_energy, abs=1.0e-8)


def test_the_adopted_namelist_reaches_the_run(pseudo_dir):
    # Not just stored: an SCF told to stop after one iteration stops after one.
    text = SILICON.replace("&electrons\n/",
                           "&electrons\n  electron_maxstep = 1\n/")
    calc = Calculator.from_text(text, pseudo_dir, announce=False)
    result = calc.get_scf()
    assert result.iterations == 1 and not result.converged


# --- diago_david_ndim: the eigensolver's one memory dial --------------------
#
# ``psi`` and ``hpsi`` are ``(nvecx, ndim)`` each with ``nvecx = david * nbnd``,
# so on a large cell this is the biggest array in the run rather than a detail
# -- 46 GB against 23 GB per k-point in flight on a 157-atom slab at
# ``ecutwfc = 60``. QE exposes it and so does this, from the input file as
# ``diago_david_ndim``, per call, and at construction.


_DAVID = SILICON.replace("&electrons\n", "&electrons\n  diago_david_ndim = 2\n")


def test_diago_david_ndim_is_read_from_the_input(pseudo_dir):
    """``&electrons`` carries it, and it reaches the ``Calculation``.

    Adopted where ``diagonalization`` beside it is deliberately not, and the
    difference is that this one cannot fail: it is an integer that means the
    same thing to the one solver here, where a solver *name* this package does
    not have would turn a valid ``pw.x`` input into a ``ValueError``.
    """
    calc = Calculator.from_text(_DAVID, pseudo_dir, announce=False)
    assert calc.calculation.david == 2


def test_the_default_is_qes_code_default_and_not_its_documented_one(pseudo_dir):
    """``None``, which the solver reads as ``DAVID_NDIM``.

    QE's own source and documentation disagree: ``input_parameters.f90:926``
    sets 4 and ``input.f90:960`` assigns it straight through, while
    ``INPUT_PW.txt`` says "Default: 2". The number ``pw.x`` runs is **4**, so
    that is the number here.
    """
    from defumat.solvers.davidson import DAVID_NDIM

    calc = Calculator.from_text(SILICON, pseudo_dir, announce=False)
    assert calc.calculation.david is None
    assert DAVID_NDIM == 4


def test_david_is_a_setup_option_and_rebuilds_the_calculation(pseudo_dir):
    """Given per call it has to replace the ``Calculation``, not a run default.

    It decides how the eigensolver behaves, so it belongs with
    ``diagonalization`` and ``k_batch`` rather than with ``conv_thr``.
    """
    calc = Calculator.from_text(SILICON, pseudo_dir, announce=False)
    assert "david" in Calculator.SETUP_OPTIONS
    assert calc.calculation.david is None
    calc.get_scf(david=2)
    assert calc.calculation.david == 2


@pytest.mark.slow
def test_the_subspace_size_changes_the_cost_and_not_the_answer(pseudo_dir):
    """The whole point: a smaller workspace must reach the same solution.

    Measured on this cell: the two totals agree to **7.3e-11** Ry and the
    smaller subspace takes one more iteration, which is the trade QE's own
    documentation describes ("a larger value may yield a smaller number of
    iterations ... but uses more memory").
    """
    import numpy as np

    from defumat.sizing import estimate_size

    wide = Calculator.from_text(SILICON, pseudo_dir, announce=False).get_scf()
    narrow = Calculator.from_text(_DAVID, pseudo_dir, announce=False).get_scf()
    assert wide.converged and narrow.converged
    assert narrow.total_energy == pytest.approx(wide.total_energy, abs=1e-8)

    # And the workspace really is half the size, which is what it was for.
    calc = Calculator.from_text(SILICON, pseudo_dir, announce=False)
    four = estimate_size(calc.system, calc.pseudos, davidson_basis=4)
    two = estimate_size(calc.system, calc.pseudos, davidson_basis=2)
    assert (two.arrays["Davidson subspace psi+hpsi"]
            == four.arrays["Davidson subspace psi+hpsi"] // 2)


# --- what a relaxation does with the namelist it adopted ---------------------


def test_the_relaxation_drivers_name_every_option_the_scf_does():
    """``_defaults_for`` forwards by **named parameter only**, so an option a
    relaxation does not name is an option a relaxation never receives.

    All three drivers take ``**scf_options`` and pass it straight to
    ``run_scf``, which reads as though the options get through -- and they do
    when a caller writes them at the call site. What they do not survive is the
    facade: ``_defaults_for`` inspects the signature, and a ``**kwargs`` is
    deliberately not permission to pass everything. So ``electron_maxstep``,
    ``diago_full_acc`` and ``mixing_fixed_ns``, all three adopted from the
    input's own ``&electrons`` namelist, were dropped on the way in and every
    SCF inside a relaxation ran at the defaults.
    """
    from defumat.workflows.relax import run_relax
    from defumat.workflows.spiral import relax_spiral_q
    from defumat.workflows.vc_relax import run_vc_relax

    wanted = {
        name for name in _ELECTRONS_OPTIONS.values()
        if name in inspect.signature(run_scf).parameters
    }
    for driver in (run_relax, run_vc_relax, relax_spiral_q):
        named = set(inspect.signature(driver).parameters)
        assert wanted <= named, f"{driver.__name__} drops {sorted(wanted - named)}"


def test_an_option_left_alone_does_not_override_run_scfs_own_default():
    """The named parameters are ``None``-defaulted rather than repeating
    ``run_scf``'s numbers, so "not given" stays distinguishable from "given the
    default" and the two cannot drift apart."""
    from defumat.workflows.relax import SCF_LOOP_OPTIONS, _scf_loop_options

    assert _scf_loop_options({}, max_iterations=None, david=None) == {}
    assert _scf_loop_options({}, max_iterations=3) == {"max_iterations": 3}
    # An explicit call-site value still wins over the calculator's default.
    assert _scf_loop_options({"max_iterations": 7}, max_iterations=3) == {
        "max_iterations": 7
    }
    assert set(SCF_LOOP_OPTIONS) <= set(inspect.signature(run_scf).parameters)


def _spy(monkeypatch, module, name, seen):
    """Replace ``module.name`` with a recorder that keeps the real signature.

    The signature is the point. ``_defaults_for`` filters strictly by named
    parameter, so a stub written as ``(*args, **options)`` is handed nothing at
    all and every assertion about what it did *not* receive passes for the
    wrong reason -- a check whose null cannot be told from a pass, which is the
    trap this file is testing a guard against.
    """
    real = getattr(module, name)

    def fake(*args, **options):
        seen[name] = options
        return name

    fake.__signature__ = inspect.signature(real)
    monkeypatch.setattr(module, name, fake)
    return fake


def test_a_workflow_that_chose_its_own_conv_thr_keeps_it(pseudo_dir,
                                                         monkeypatch):
    """The input file's ``&electrons conv_thr`` must not reach a workflow that
    picked a tighter one for a reason it wrote down.

    Fifteen entry points declare a ``conv_thr`` between 1e-8 and 1e-12 -- an
    anisotropy is a difference of band-energy sums in the fifth decimal of an
    eV, an effective mass is a second difference of eigenvalues -- and
    ``pw.x``'s own 1e-6 is what an input file usually states. Both legs of a
    force theorem moved together, so the workflow's own ``drifts`` could not
    see it.
    """
    import defumat.workflows.anisotropy as anisotropy
    import defumat.workflows.topology as topology

    seen = {}
    for name in ("run_anisotropy", "run_torque", "run_force_theorem",
                 "frozen_expectation"):
        _spy(monkeypatch, anisotropy, name, seen)
    _spy(monkeypatch, topology, "run_z2", seen)

    calc = Calculator.from_text(SILICON, pseudo_dir, announce=False,
                                conv_thr=1.0e-6)
    spinor = Calculator.from_text(SILICON, pseudo_dir, announce=False)
    calc._scf = _converged_stub()

    calc.get_anisotropy(spinor)
    calc.get_torque(spinor)
    calc.get_force_theorem(spinor)
    calc.get_first_order_soc(spinor)
    calc.get_z2()
    assert set(seen) == {"run_anisotropy", "run_torque", "run_force_theorem",
                         "frozen_expectation", "run_z2"}, (
        f"a spy was never called: {sorted(seen)}"
    )
    for name, options in seen.items():
        assert "conv_thr" not in options, (
            f"{name} was handed the calculator's 1e-6"
        )


def test_the_same_conv_thr_still_reaches_a_workflow_with_no_opinion(pseudo_dir,
                                                                    monkeypatch):
    """The other half, and the reason the rule reads the signature.

    ``run_bands`` repeats ``run_scf``'s own 1e-6, which is the callee saying it
    has no opinion, so the calculator's number is exactly what it is for.
    Without this the rule would read as "``conv_thr`` is never forwarded",
    which is a different thing and a wrong one.
    """
    import defumat.workflows.bands as bands

    seen = {}
    _spy(monkeypatch, bands, "run_bands", seen)
    calc = Calculator.from_text(SILICON, pseudo_dir, announce=False,
                                conv_thr=1.0e-8)
    calc._scf = _converged_stub()
    calc.get_bands()
    assert seen["run_bands"]["conv_thr"] == 1.0e-8


def test_naming_it_at_the_call_site_still_reaches_the_workflow(pseudo_dir,
                                                               monkeypatch):
    """A default is not a refusal, which is the line ``withheld`` already drew."""
    import defumat.workflows.anisotropy as anisotropy

    seen = {}
    _spy(monkeypatch, anisotropy, "run_anisotropy", seen)
    calc = Calculator.from_text(SILICON, pseudo_dir, announce=False,
                                conv_thr=1.0e-6)
    spinor = Calculator.from_text(SILICON, pseudo_dir, announce=False)
    calc._scf = _converged_stub()
    calc.get_anisotropy(spinor, conv_thr=1.0e-4)
    assert seen["run_anisotropy"]["conv_thr"] == 1.0e-4


def test_the_collinear_legs_band_count_does_not_cross_to_the_spinor_one(
        pseudo_dir, monkeypatch):
    """A spinor band holds one electron where a collinear one holds two.

    ``self`` is the scalar-relativistic leg and the run happens on the *other*
    calculator's system, so ``nbnd`` -- a property of the system whose bands
    are being counted -- stays behind, and the spinor calculator's own crosses
    in its place rather than being lost with it.
    """
    import defumat.workflows.anisotropy as anisotropy

    seen = {}
    _spy(monkeypatch, anisotropy, "run_anisotropy", seen)
    calc = Calculator.from_text(SILICON, pseudo_dir, announce=False, nbnd=8)
    calc._scf = _converged_stub()

    plain = Calculator.from_text(SILICON, pseudo_dir, announce=False)
    calc.get_anisotropy(plain)
    assert seen["run_anisotropy"].get("nbnd") is None, (
        "the collinear leg's band count crossed into the spinor run"
    )

    stated = Calculator.from_text(SILICON, pseudo_dir, announce=False, nbnd=16)
    calc.get_anisotropy(stated)
    assert seen["run_anisotropy"]["nbnd"] == 16, (
        "the spinor leg's own band count was dropped with the other one"
    )

    calc.get_anisotropy(stated, nbnd=20)
    assert seen["run_anisotropy"]["nbnd"] == 20, "a call-site count must win"


def test_every_spinor_leg_callee_names_every_withheld_option():
    """The ``setdefault`` that carries the second leg's options across would
    otherwise hand an entry point a keyword it does not take."""
    from defumat.calculator import _SPINOR_LEG_OPTIONS
    from defumat.workflows.anisotropy import (frozen_expectation,
                                              run_anisotropy,
                                              run_force_theorem, run_torque)

    for func in (run_anisotropy, run_torque, frozen_expectation,
                 run_force_theorem):
        named = set(inspect.signature(func).parameters)
        assert _SPINOR_LEG_OPTIONS <= named, (
            f"{func.__name__} does not name {sorted(_SPINOR_LEG_OPTIONS - named)}"
        )


def test_the_rule_is_read_off_the_signature_and_not_off_a_list():
    """``_callee_chose`` is the whole guard, so its four cases are asserted
    directly: a stated value that differs is the callee's, one that repeats
    ``run_scf``'s is not, ``None`` is the callee having no opinion, and a
    parameter with no default at all is one the calculator must supply or the
    call fails."""
    from defumat.calculator import _callee_chose

    def entry(required, chosen=1.0e-10, silent=None, agreeing=1.0e-6):
        pass

    p = inspect.signature(entry).parameters
    assert _callee_chose({"conv_thr": p["chosen"]}, "conv_thr")
    assert not _callee_chose({"conv_thr": p["agreeing"]}, "conv_thr")
    assert not _callee_chose({"max_iterations": p["silent"]}, "max_iterations")
    assert not _callee_chose({"conv_thr": p["required"]}, "conv_thr")


def test_a_tighter_conv_thr_from_the_calculator_still_tightens(pseudo_dir,
                                                               monkeypatch):
    """The rule is not symmetric, and the asymmetry is the physics.

    A threshold is a bound on an error, so a smaller one is never the wrong
    thing to hand a workflow. Without this an input stating 1e-10 would have
    been withheld from ``run_ultracell``, whose own 1e-8 is a second driver's
    default for the *same* quantity rather than a tightening -- that module
    says so -- and the run would have come back looser than the file asked for.
    """
    import defumat.workflows.anisotropy as anisotropy

    seen = {}
    _spy(monkeypatch, anisotropy, "run_anisotropy", seen)
    spinor = Calculator.from_text(SILICON, pseudo_dir, announce=False)

    tighter = Calculator.from_text(SILICON, pseudo_dir, announce=False,
                                   conv_thr=1.0e-12)
    tighter._scf = _converged_stub()
    tighter.get_anisotropy(spinor)
    assert seen["run_anisotropy"]["conv_thr"] == 1.0e-12

    # ... and the callee's own value is not "chosen" enough to beat an equal one
    equal = Calculator.from_text(SILICON, pseudo_dir, announce=False,
                                 conv_thr=1.0e-10)
    equal._scf = _converged_stub()
    equal.get_anisotropy(spinor)
    assert "conv_thr" not in seen["run_anisotropy"]


def test_only_a_tolerance_is_ordered_and_max_iterations_is_not(pseudo_dir):
    """``max_iterations`` of 40 against 100 is not better or worse, so no value
    of it from the calculator reaches a callee that chose one."""
    from defumat.calculator import _yields_to_callee

    def entry(conv_thr=1.0e-10, max_iterations=40):
        pass

    p = inspect.signature(entry).parameters
    assert _yields_to_callee(p, "conv_thr", 1.0e-6)
    assert not _yields_to_callee(p, "conv_thr", 1.0e-12)
    assert _yields_to_callee(p, "conv_thr", 1.0e-10)
    for value in (10, 40, 500):
        assert _yields_to_callee(p, "max_iterations", value)


def test_the_rule_reaches_the_entry_points_it_was_written_for():
    """Which entry points it bites on, listed rather than left to be assumed.

    Every one of these declares a ``conv_thr`` its own docstring argues for,
    and every one of them was being handed the calculator's instead.
    """
    from defumat.calculator import _callee_chose
    from defumat.response.effmass import effective_mass
    from defumat.workflows.anisotropy import (frozen_expectation,
                                              run_anisotropy,
                                              run_force_theorem, run_torque)
    from defumat.workflows.nesting import run_nesting
    from defumat.workflows.spiral import relax_spiral_q
    from defumat.workflows.orbital_magnetization import (
        run_orbital_magnetization)
    from defumat.workflows.polarization import run_polarization
    from defumat.workflows.topology import (run_berry_curvature, run_z2,
                                            run_z2_3d)

    for func in (run_anisotropy, run_torque, frozen_expectation,
                 run_force_theorem, effective_mass, run_nesting,
                 relax_spiral_q, run_berry_curvature, run_z2, run_z2_3d,
                 run_polarization, run_orbital_magnetization):
        parameters = inspect.signature(func).parameters
        assert _callee_chose(parameters, "conv_thr"), (
            f"{func.__name__} no longer states a conv_thr of its own, so this "
            "guard no longer covers it"
        )
    # ... and run_scf itself is the reference, so it can never be withheld.
    assert not _callee_chose(inspect.signature(run_scf).parameters, "conv_thr")


def _converged_stub():
    """A minimal converged :class:`SCFResult` stand-in for the facade tests.

    The workflows above are replaced by :func:`_spy`, so nothing reads the
    arrays -- what is exercised is which options reach them.
    """
    import dataclasses

    from defumat.scf.driver import SCFResult

    fields = {f.name: None for f in dataclasses.fields(SCFResult)}
    fields.update(converged=True, density=np.zeros((2, 4, 4, 4)), nspin=2)
    return SCFResult(**fields)


def test_the_adopted_namelist_reaches_the_scf_inside_a_relaxation(pseudo_dir):
    """The running counterpart of the signature test above.

    ``electron_maxstep = 1`` caps the *electronic* loop, not the ionic one, so
    the relaxation's first step must report exactly one SCF iteration.
    """
    text = (SILICON.replace("calculation = 'scf'", "calculation = 'relax'")
            .replace("&electrons\n/", "&electrons\n  electron_maxstep = 1\n/"))
    if "&ions" not in text:
        text = text.replace("ATOMIC_SPECIES", "&ions\n/\nATOMIC_SPECIES")
    calc = Calculator.from_text(text, pseudo_dir, announce=False)
    relax = calc.get_relax(nstep=1)
    assert relax.steps[0].scf_iterations == 1
