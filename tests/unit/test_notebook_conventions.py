"""The notebook conventions, enforced on the JSON rather than on trust.

`CLAUDE.md` and `notebooks/README.md` have carried these rules for a long time
and the notebooks drifted anyway, because the rules were prose about prose:
they bound what a notebook *said* and never what its code cells *did*. So the
project's validation instinct moved into the code, where the rule did not reach,
and 29 notebooks accumulated 2800 lines of it -- identity checks looped over
four pseudopotentials, a derivative checked against a closed form on a random
matrix, hand-built linear solves. All of that is the test suite's job (P49).

This file is the rule with teeth. It parses the ``.ipynb`` JSON and executes
nothing, so it costs milliseconds and belongs in the fast gate.

**It is a ratchet, not a wall.** Only the notebooks in :data:`REWRITTEN` are held
to the full skeleton; the rest are checked for the things that are cheap to keep
true everywhere. A notebook joins ``REWRITTEN`` when it is rewritten, and the
set only ever grows -- which is what stops this file from being deleted the
first time it is inconvenient. The baseline when it was written: 4 of 29 passed
the full skeleton, and the other 25 failed on code volume (up to 149 lines
against a budget of 80), on cell length (up to 48 against 25), or on reaching
past the facade (up to 7 imports).
"""

import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

NOTEBOOKS = Path(__file__).resolve().parents[2] / "notebooks"

#: Held to the whole skeleton. **This set only grows.** Add a notebook here in
#: the same commit that rewrites it; never remove one to make a test pass.
REWRITTEN = {
    "00_the_calculator",
    "02_silicon_scf_and_bands",
    "04_ultrasoft_and_paw",
    "05_gradient_corrections",
    "06_density_of_states",
    "07_spin_polarization",
    "08_spin_orbit_coupling",
    "09_forces_and_relaxation",
    "10_topological_invariants",
    "11_noncollinear_magnetism_and_fields",
    "12_spin_spirals",
    "13_dft_plus_u",
    "14_spiral_relaxation",
    "15_stress",
    "16_projected_density_of_states",
    "18_continuing_a_calculation",
    "19_linear_response",
    "20_phonons",
    "21_electrostriction",
    "22_van_der_waals",
    "23_variable_cell_relaxation",
    "24_tran_blaha_band_gaps",
    "25_your_own_crystal",
    "26_raman_and_infrared_spectra",
    "27_excitons_and_tddft",
    "28_piezoelectricity",
    "29_effective_mass_and_angular_momenta",
}

#: Notebooks that still say ``jvp`` in a plot label or a code cell. Prose debt
#: from before the rule was enforced, and it goes when P49's phase 5 reaches
#: them. **This set only shrinks.**
JVP_DEBT = set()

#: What a notebook may import from ``pypresso``. Everything else is the
#: implementation, and a notebook that needs it says why on the same line.
FACADE = {"pypresso", "pypresso.units", "pypresso.system.kpoints", "pypresso.io"}

#: At most this many justified exceptions to :data:`FACADE`, per notebook. Two,
#: because the two figures that survived the rewrite of ``02`` and ``19`` both
#: need ``r_to_g`` and there is no facade route to ``rho(G)``.
DEEP_IMPORT_BUDGET = 2

#: Non-comment code lines, per notebook and per cell.
LINE_BUDGET, CELL_BUDGET = 80, 25

#: The first code cell: the one the whole notebook is for.
FIRST_CELL_BUDGET = 12

#: Vocabulary that belongs in ``PLAN.md`` and in the tests. ``P\d\d`` is the
#: phase reference; it has never fired on a space group here (``P6_3/mmc`` is
#: safe, the underscore being a word character) but ``P4/mmm`` would, and the
#: fix if it ever does is to name that notebook in an allowlist beside this,
#: not to weaken the pattern.
IMPLEMENTATION_VOCABULARY = {
    "jvp": r"\bjvp\b",
    "a Fortran file name": r"\.f90\b",
    "a PLAN.md reference": r"PLAN\.md",
    "a phase number": r"\bP\d{1,2}[a-c]?\b",
    "an em dash": "—",
}

#: An entry point a ``get_*`` method already wraps. Importing one of these into
#: a notebook is the ``19`` pattern: build the quantity by hand, then remark
#: that the one-line call exists.
HAS_A_FACADE_METHOD = {
    "run_scf": "get_scf",
    "run_bands": "get_bands",
    "run_dos": "get_dos",
    "run_nscf": "get_nscf",
    "run_pdos": "get_pdos",
    "run_relax": "get_relax",
    "compute_forces": "get_forces",
    "compute_stress": "get_stress",
    "dielectric_tensor": "get_dielectric_tensor",
    "raman_tensors": "get_raman_tensors",
    "vibrational_spectrum": "get_vibrational_spectrum",
    "run_absorption": "get_absorption",
    "elastic_constants": "get_elastic_constants",
    "electrostriction": "get_electrostriction",
    "piezoelectric_tensor": "get_piezoelectric_tensor",
    "effective_mass": "get_effective_mass",
    "run_berry_curvature": "get_berry_curvature",
    "run_z2": "get_z2",
    "run_z2_3d": "get_z2_3d",
    "VelocityOperator": "get_band_velocities",
}


def _notebooks():
    return sorted(p for p in NOTEBOOKS.glob("[0-9]*.ipynb"))


def _cells(path):
    cells = json.loads(path.read_text())["cells"]
    return [c for c in cells if c["cell_type"] == "code"]


def _lines(cell):
    return [l for l in cell["source"]
            if l.strip() and not l.strip().startswith("#")]


ALL = pytest.mark.parametrize(
    "path", _notebooks(), ids=lambda p: p.stem)
SKELETON = pytest.mark.parametrize(
    "path", [p for p in _notebooks() if p.stem in REWRITTEN], ids=lambda p: p.stem)


# ----------------------------------------------------------------------
# every notebook
# ----------------------------------------------------------------------

@ALL
def test_the_markdown_export_is_current(path):
    """``.ipynb`` is the source of truth and the ``.md`` is how it is read.

    A cheap staleness canary that costs no execution: one fenced ``python``
    block per code cell. It cannot see a changed *output*, but it catches the
    common failure, which is editing the notebook and forgetting the export.
    """
    export = path.with_suffix(".md")
    assert export.is_file(), f"no .md export beside {path.name}"
    fences = len(re.findall(r"^```python$", export.read_text(), re.M))
    assert fences == len(_cells(path)), (
        f"{export.name} has {fences} code blocks against the notebook's "
        f"{len(_cells(path))}: re-run tools/export_notebooks.sh"
    )


@ALL
def test_no_implementation_vocabulary(path):
    """A notebook is about the physics. This is that rule, mechanised."""
    text = "".join("".join(c["source"])
                   for c in json.loads(path.read_text())["cells"])
    for name, pattern in IMPLEMENTATION_VOCABULARY.items():
        if name == "jvp" and path.stem in JVP_DEBT:
            continue
        found = re.search(pattern, text)
        assert not found, (
            f"{path.name} says {name} ({found.group(0)!r}). That belongs in "
            f"PLAN.md or in the tests, not in a tutorial."
        )


@ALL
def test_the_debt_lists_only_name_notebooks_that_exist(path):
    """A ratchet that names a deleted notebook has stopped ratcheting."""
    stems = {p.stem for p in _notebooks()}
    assert REWRITTEN <= stems, f"REWRITTEN names a missing notebook: {REWRITTEN - stems}"
    assert JVP_DEBT <= stems, f"JVP_DEBT names a missing notebook: {JVP_DEBT - stems}"


# ----------------------------------------------------------------------
# the notebooks held to the skeleton
# ----------------------------------------------------------------------

@SKELETON
def test_the_first_code_cell_opens_a_calculator(path):
    """Cell 2 of the skeleton is what the whole notebook is for.

    Imports, ``Calculator.from_file``, the one ``get_X()`` call, the number
    printed. A reader who stops after it has still seen the answer.
    """
    first = _cells(path)[0]
    assert "Calculator.from_" in "".join(first["source"]), (
        f"{path.name}'s first code cell does not build a Calculator"
    )
    assert len(_lines(first)) <= FIRST_CELL_BUDGET, (
        f"{path.name}'s first code cell is {len(_lines(first))} lines, "
        f"budget {FIRST_CELL_BUDGET}"
    )


@SKELETON
def test_the_code_budget_is_kept(path):
    cells = _cells(path)
    total = sum(len(_lines(c)) for c in cells)
    longest = max(len(_lines(c)) for c in cells)
    assert total <= LINE_BUDGET, f"{path.name} has {total} code lines, budget {LINE_BUDGET}"
    assert longest <= CELL_BUDGET, (
        f"{path.name}'s longest cell is {longest} lines, budget {CELL_BUDGET}"
    )


@SKELETON
def test_imports_stay_at_the_facade(path):
    """Past ``Calculator`` is the implementation, and it is capped.

    An exception carries its reason on the same line, so that the next reader
    knows whether it is still true -- ``# no facade route to rho(G)`` is the
    one that exists today.
    """
    unjustified, justified = [], []
    for cell in _cells(path):
        for line in cell["source"]:
            match = re.match(r"\s*from (pypresso[\w.]*) import", line)
            if not match or match.group(1) in FACADE:
                continue
            (justified if "#" in line else unjustified).append(match.group(1))
    assert not unjustified, (
        f"{path.name} imports {unjustified} from past the facade with no "
        f"reason given on the line"
    )
    assert len(justified) <= DEEP_IMPORT_BUDGET, (
        f"{path.name} has {len(justified)} justified deep imports, "
        f"budget {DEEP_IMPORT_BUDGET}"
    )


@SKELETON
def test_no_input_plumbing_helpers(path):
    """``def load(...)`` was a workaround for the library and is now a smell.

    Eleven notebooks defined one, four of them only to re-parse the input file
    and hand ``&electrons`` straight back to a ``Calculator`` built from the
    same text. ``from_file`` adopts that namelist itself now, so a helper whose
    body is one constructor call has nothing left to do.
    """
    text = "".join("".join(c["source"]) for c in _cells(path))
    for helper in (r"def load\(", r"def scf\(", r"def namelist\("):
        assert not re.search(helper, text), (
            f"{path.name} defines {helper[4:-2]}(): put the settings in the "
            f"input file, which from_file now reads"
        )


@SKELETON
def test_a_facade_method_is_used_where_one_exists(path):
    """The ``19`` pattern: build it by hand, then mention the one-line call."""
    text = "".join("".join(c["source"]) for c in _cells(path))
    for entry_point, method in HAS_A_FACADE_METHOD.items():
        if re.search(rf"\b{entry_point}\b", text) and method not in text:
            pytest.fail(
                f"{path.name} uses {entry_point} where Calculator.{method}() "
                f"exists and is not called"
            )
