"""The performance report writes valid LaTeX for whatever the run produced.

``performance/run_performance.py`` is meant to be run by hand at any time, so
the failure to avoid is the one that only shows up at the end of a long
measurement: a report that will not typeset. These checks run the LaTeX
generation on synthetic results -- no measurement, no ``pdflatex`` -- so a
missing template field or an unescaped case name is caught in the fast suite.
"""

import importlib.util
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


def _module():
    spec = importlib.util.spec_from_file_location(
        "run_performance", ROOT / "performance" / "run_performance.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(case="si8-1k", **overrides):
    record = {
        "case": case, "description": "eight atoms", "nat": 8, "nk": 1,
        "ecutwfc": 12.0, "ngm": 4477, "npwx": 738, "grid": [30, 30, 30],
        "conv_thr": 1e-10, "qe_init": 0.5, "qe_scf": 0.216, "qe_total": 1.2,
        "qe_iterations": 8, "qe_energy": -62.9, "setup": 2.0, "scf_cold": 3.0,
        "scf_warm": 0.6, "iterations": 8, "energy": -62.9, "wall": 30.0,
        "peak_rss_gb": 1.25,
    }
    record.update(overrides)
    return record


ENVIRONMENT = {
    "date": "2026-08-20 10:00", "host": "l26-0350", "cpu": "Intel(R) Core(TM) Ultra 5",
    "cores": 14, "python": "3.14.6", "jax": "0.11.0", "pypresso": "0.0.1",
    "qe": "v.7.5", "commit": "abc1234",
}


def test_every_template_field_is_supplied(tmp_path):
    """The bug this catches: a field added to the template and not to the call.

    ``str.format`` raises ``KeyError`` for a missing one, which would otherwise
    surface only after every case had been measured.
    """
    module = _module()
    path = module.write_tex([_record()], ENVIRONMENT, tmp_path / "performance.tex")
    text = path.read_text()
    assert text.startswith("\\documentclass")
    assert text.rstrip().endswith("\\end{document}")
    # The measured numbers reach the page, not just the scaffolding.
    assert "si8-1k" in text
    assert "0.216" in text and "0.600" in text


def test_a_case_over_budget_is_reported_not_dropped(tmp_path):
    """A skipped case has to appear somewhere, or the report overstates coverage."""
    module = _module()
    results = [_record(), {"case": "si16-1k", "description": "", "over_budget": True}]
    text = module.write_tex(results, ENVIRONMENT, tmp_path / "performance.tex").read_text()
    assert "over budget" in text
    assert "si16-1k" in text


def test_case_names_are_escaped(tmp_path):
    """An underscore in a benchmark name must not end the run in a LaTeX error."""
    module = _module()
    text = module.write_tex([_record(case="si_8_test")], ENVIRONMENT,
                            tmp_path / "performance.tex").read_text()
    assert "si\\_8\\_test" in text


def test_the_bar_chart_axis_lists_the_cases_it_plots(tmp_path):
    """``symbolic y coords`` and the coordinates have to agree, or pgfplots stops."""
    module = _module()
    text = module.write_tex([_record("si-1k"), _record("si8-1k")], ENVIRONMENT,
                            tmp_path / "performance.tex").read_text()
    axis = text.split("symbolic y coords/.expanded={")[1].split("}")[0]
    coordinates = text.split("\\addplot coordinates {")[1].split("};")[0]
    plotted = set(re.findall(r"\(\s*[\d.]+\s*,\{([^}]+)\}\)", coordinates))
    assert plotted, coordinates
    assert plotted == set(axis.split(","))


def test_the_default_cases_exist():
    """Every case in the standard set is a benchmark that is actually committed."""
    module = _module()
    for name, _ in module.DEFAULT_CASES:
        assert (ROOT / "benchmarks" / f"{name}.in").is_file(), name
