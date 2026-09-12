"""What an ``fsm`` run reports when it does not reach its target.

A penalty's miss is visible in ``constraint_energy``, which is part of the energy
and falls to zero as the constraint is met. ``constrained_magnetization = 'fsm'``
is not a penalty -- it drives a *feedback field* -- so its ``constraint_energy``
is **0 by construction**, and until this was added nothing on the result said how
far the run ended from what it was asked for:
:meth:`~defumat.scf.fields.MagneticField.satisfied` computed exactly that error,
compared it against :data:`~defumat.scf.fields.FSM_TOLERANCE`, returned a bool and
threw the number away.

The consequence was measured before it was fixed (``PLAN.md`` P80). An fcc
hydrogen spiral under ``fsm`` stopped after 200 iterations with
``accuracy = 3.9e-11`` -- *below* its own ``conv_thr`` of 1e-10 -- and a moment
0.174 Bohr magnetons from its target. So ``converged = False`` on the result, the
generic "unconverged density" warning, and no residual anywhere: a run whose
density was fine and whose constraint had failed read as an ordinary
non-convergence, and the generic advice (more iterations, smaller
``mixing_beta``) is backwards for it.

Two things are checked here and **both are guards being made to fire**, which is
this project's rule about guards: the residual appears and carries a *sign*, and
the dedicated warning fires exactly when the density converged and the constraint
did not -- not on an ordinary non-convergence, which must keep the generic
message.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

from defumat.io.pwin import parse_pw_input
from defumat.pseudo import read_upf
from defumat.scf import Calculation, run_scf
from defumat.scf.fields import FSM_TOLERANCE
from defumat.system.builder import build_system

pytestmark = pytest.mark.unit

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"

#: The spiral of ``h-fcc-spiral-scan.in`` at ``q_3 = 1/2`` under ``fsm``, with the
#: target at the same cell's own ``q = 0`` moment. This cell cannot be held --
#: ``m(B)`` is nearly a step, so the secant overshoots (``PLAN.md`` P80) -- which
#: is exactly why it is the right fixture for a *reporting* test.
TARGET = 0.027282


def _run(pseudo_dir, **options):
    text = (CASES / "h-fcc-spiral-scan.in").read_text()
    text = text.replace("spiral_q(3) = 0.0", "spiral_q(3) = 0.5").replace(
        "    nbnd = 8\n",
        "    nbnd = 8\n    constrained_magnetization = 'fsm'\n"
        f"    fixed_magnetization(1) = {TARGET:.6f}\n",
    )
    system = build_system(parse_pw_input(text))
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_scf(system, pseudos, calculation=calculation,
                         verbose=False, **options)
    messages = [str(w.message) for w in caught]
    return result, messages


def test_the_residual_is_reported_signed_and_per_iteration(pseudo_dir):
    """``m - m_target`` on the result and in ``history``.

    Signed and per component, because a moment that has crossed to the **other
    side** of its target is a different failure from one that has not arrived --
    the first wants a smaller step and the second more of them, and a magnitude
    cannot tell them apart. (The first write-up of P80's measurement reported
    ``-0.1468`` against a target of ``+0.0273`` as an "overshoot by a factor of
    five", which is precisely that mistake.)
    """
    result, _ = _run(pseudo_dir, conv_thr=0.2, max_iterations=6)

    residual = np.asarray(result.constraint_residual)
    assert residual.shape == (3,)
    # It is genuinely the signed difference and not a norm.
    moment = np.asarray(result.magnetization_vector)
    assert residual[0] == pytest.approx(moment[0] - TARGET, abs=1e-9)
    # Not an atom-resolved scheme, so the per-site companion stays absent.
    assert result.site_residuals is None
    # And the trajectory, which is what shows the field's own steps: the secant
    # updates only on self-consistent pairs, so a run that stopped on its
    # iteration budget looks identical to one that diverged unless this is here.
    assert all("constraint_residual" in entry for entry in result.history)
    assert np.asarray(result.history[-1]["constraint_residual"]).shape == (3,)


def test_a_converged_density_with_an_unmet_constraint_gets_its_own_warning(pseudo_dir):
    """The guard, fed the case that must trip it.

    ``conv_thr = 0.2`` is deliberately absurd: it lets the *density* pass in six
    iterations while the moment is still 0.1 Bohr magnetons from its target, which
    is the same situation the 200-iteration run at ``conv_thr = 1e-10`` reached
    and reaches it in seconds. What must be said is that the density is fine, and
    what must not be said is "lower mixing_beta".
    """
    result, messages = _run(pseudo_dir, conv_thr=0.2, max_iterations=6)

    assert not result.converged
    assert result.accuracy < 0.2, "the density must have passed for this to be the case"
    assert abs(np.asarray(result.constraint_residual)[0]) > FSM_TOLERANCE

    stopped = [m for m in messages if "SCF stopped" in m]
    assert len(stopped) == 1
    message = stopped[0]
    assert "its **density** did" in message
    assert "'fsm'" in message
    assert "constraint_residual" in message
    assert "shared between the inner SCF and the outer field loop" in message
    # The generic advice is wrong here and must not appear.
    assert "lower mixing_beta" not in message


def test_an_ordinary_non_convergence_still_gets_the_generic_warning(pseudo_dir):
    """The other half: the split must not swallow the case it does not own.

    A tight threshold the density does not reach is an ordinary non-convergence
    even with a constraint in force, and the advice there *is* the generic one.
    """
    result, messages = _run(pseudo_dir, conv_thr=1e-6, max_iterations=6)

    assert not result.converged
    assert result.accuracy > 1e-6
    stopped = [m for m in messages if "SCF stopped" in m]
    assert len(stopped) == 1
    assert "its **density** did" not in stopped[0]
    assert "lower mixing_beta" in stopped[0]
