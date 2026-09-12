"""Converging a magnetic structure that is **not** the ground state.

Stating a 120-degree Neel state, a cone or a canted configuration is one thing;
converging one is another, and nothing in the SCF turns a moment back to where
it was asked to be. Left alone, the two-sublattice 120-degree cell here relaxes
to the collinear antiferromagnet that is lower and reports success.

What holds it is a penalty on each atom's moment, and **which** penalty matters
more than the stiffness does:

* ``'atomic'`` constrains the moment as a *vector*, so its potential is
  ``2 lambda (m - m_target)`` and bounded. It converges, and the angle it holds
  tightens monotonically with ``lambda``.
* ``'atomic texture'`` constrains the *direction alone*, so its potential
  carries a ``1/|m|`` and grows as a site's moment shrinks. That is positive
  feedback, and on this cell no ``lambda`` converged in 400 iterations.

Measured here, two hydrogen atoms of one species at +-60 degrees from z
(``tests/data/qe/h2-texture-120.in``), ``conv_thr = 1e-8``, targets at the
converged sphere moment of 0.26 mu_B:

    scheme            lambda   converged   iterations   angle    error/site
    (none)                --      yes           10      180.00     30.00
    atomic              0.05      yes           10      170.95     25.47
    atomic               0.2      yes            7      153.52     16.76
    atomic               1.0      yes            9      130.30      5.15
    atomic               3.0      yes           17      123.65      1.83
    atomic              10.0      yes           38      121.13      0.55
    atomic              30.0       NO          200       11.91     64.41
    atomic texture       0.1       NO          200      118.39      4.30
    atomic texture      0.02       NO          400      140.03     10.02
    atomic texture      >= 2        NO          200    collapsed       --

So the largest ``lambda`` the SCF tolerates is 10, and there it holds the angle
to **0.55 degrees**. ``'atomic texture'`` gets closer on the angle and never
converges, which is the trade the docstring above explains.
"""

import numpy as np
import pytest

from defumat import Calculator

#: **Not blanket-slow.** The audit's own finding was that the fast gate contained
#: no magnetic *noncollinear* SCF at all -- every `pw.x` noncollinear comparison
#: is module-level slow and the only spinor SCF in the gate is nonmagnetic
#: platinum. The two cheap tests here are the pair worth having: a canted state
#: collapsing to collinear (7.8 s) and one held by a constraint (4.3 s). The
#: expensive two are marked individually.
pytestmark = [pytest.mark.regression]

INPUT = "tests/data/qe/h2-texture-120.in"
UNIT = np.array([[np.sin(a), 0.0, np.cos(a)] for a in np.deg2rad([60.0, -60.0])])


def _card(length):
    rows = UNIT * length
    return ("STARTING_MOMENTS\n"
            + "\n".join(f"  {r[0]:.10f}  {r[1]:.10f}  {r[2]:.10f}" for r in rows)
            + "\n")


def _text(scheme=None, lam=None, length=0.26):
    """The committed input with its constraint replaced and its card rescaled."""
    from pathlib import Path
    base = Path(INPUT).read_text()
    head = base[:base.rindex("STARTING_MOMENTS")]
    if scheme is None:
        head = head.replace(
            "    constrained_magnetization = 'atomic texture'\n", ""
        ).replace("    lambda = 0.5\n", "")
    else:
        head = head.replace("constrained_magnetization = 'atomic texture'",
                            f"constrained_magnetization = '{scheme}'")
        head = head.replace("lambda = 0.5", f"lambda = {lam}")
    return head + _card(length)


def _converge(text, pseudo_dir, **options):
    calculator = Calculator.from_text(text, pseudo_dir, announce=False)
    scf = calculator.get_scf(conv_thr=1e-8, max_iterations=100, **options)
    moments = np.asarray(scf.site_moments)
    lengths = np.linalg.norm(moments, axis=1)
    hats = moments / lengths[:, None]
    between = np.degrees(np.arccos(np.clip(hats[0] @ hats[1], -1.0, 1.0)))
    errors = np.degrees(np.arccos(np.clip(
        np.einsum("ac,ac->a", hats, UNIT), -1.0, 1.0)))
    return scf, between, errors


@pytest.fixture(scope="module")
def pseudo_dir():
    return "tests/data/pseudo"


def test_nothing_holds_a_canted_state_on_its_own(pseudo_dir):
    """The reason the constraint has to exist.

    Two moments seeded 120 degrees apart converge to **180** -- the collinear
    antiferromagnet -- in ten iterations, and the run reports success. This is
    the failure `docs/features.tex` records for a 15-degree cone on fcc
    hydrogen, on a cell small enough to assert.
    """
    scf, between, errors = _converge(_text(), pseudo_dir)
    assert scf.converged
    assert between == pytest.approx(180.0, abs=0.5), between
    assert errors.min() > 25.0, errors


@pytest.mark.parametrize("lam, angle, tolerance", [
    (1.0, 130.30, 1.0),
    pytest.param(10.0, 121.13, 1.0, marks=pytest.mark.slow),
])
def test_the_vector_penalty_holds_it_and_tightens_with_lambda(
        lam, angle, tolerance, pseudo_dir):
    """``'atomic'`` converges at every stiffness up to 10 and holds the angle.

    At the largest ``lambda`` the SCF tolerates the error is **0.55 degrees per
    site**, which is the number that says a texture which is not the ground
    state can be converged here at all.
    """
    scf, between, errors = _converge(_text("atomic", lam), pseudo_dir)
    assert scf.converged, scf.accuracy
    assert between == pytest.approx(angle, abs=tolerance), between
    assert between < 175.0, "the constraint did nothing"


@pytest.mark.slow
def test_the_direction_only_penalty_does_not_converge_on_this_cell(pseudo_dir):
    """The guard that must fire, rather than a clean number read as a pass.

    ``'atomic texture'``'s potential carries a ``1/|m|``, so a shrinking moment
    is amplified rather than damped. The warning at input says so and names the
    scheme that works; this asserts the warning is there and that the run it
    warns about really does fail to converge.
    """
    text = _text("atomic texture", 0.1, length=0.6)
    with pytest.warns(RuntimeWarning, match=r"1/\|m\|"):
        Calculator.from_text(text, pseudo_dir, announce=False)
    scf, _, _ = _converge(text, pseudo_dir)
    assert not scf.converged, (
        "'atomic texture' converged on this cell -- if that is now true the "
        "warning and the table in this module's docstring are both stale"
    )
