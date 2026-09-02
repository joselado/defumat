"""What the spinor force and stress still refuse, checked in the fast gate.

P46 narrowed :func:`defumat.forces.energy.reject_spinors` rather than deleting
it, and four things stayed refused, each for its own missing term. Those four
assertions used to live in ``tests/regression/test_spinor_forces.py``, which is
marked ``slow`` in its entirety and takes a quarter of an hour -- so a refusal
lifted by accident would not have been caught until someone ran the slow set,
which by ``PLAN.md`` P38's account is where three phases' claims had already
drifted unnoticed.

They are here instead, and they are **not** slow, because a refusal fires on the
*calculation* and not on a converged state: three of the four need no SCF at all
and the other two need one iteration, which is enough to have a
:class:`~defumat.forces.energy.FrozenState` to hand in. That is the whole
reason this file is cheap -- the expensive part of the sibling file is the
physics, and a refusal has none.

``GAPS.md`` §4 is the argument for the file existing: almost every gap that
sweep found was *the same guard missing from a sibling entry point*, and the
general answer it asks for is to enumerate the paths that reach a refusal and
assert it fires on each. This is that, for the one phase that lifted a refusal
half-way.
"""

from functools import lru_cache
from pathlib import Path

import pytest

from defumat.forces import compute_forces
from defumat.forces.energy import energy_at, reject_spinor_spiral, state_from_result
from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.scf import Calculation, run_scf
from defumat.system import build_system

pytestmark = [pytest.mark.unit]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"


@lru_cache(maxsize=2)
def _calculation(case: str, pseudo_dir: Path):
    """The calculation alone -- which is all a refusal is a function of."""
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    return Calculation(system, pseudos)


@lru_cache(maxsize=1)
def _one_iteration(case: str, pseudo_dir: Path):
    """A state to hand in, and deliberately not a converged one.

    The two refusals below fire before anything reads a number out of it, so
    paying for convergence would buy nothing at all -- and this file's whole
    point is that it costs seconds.
    """
    calculation = _calculation(case, pseudo_dir)
    result = run_scf(calculation.system, calculation.pseudos, calculation=calculation,
                     max_iterations=1, conv_thr=1.0)
    return calculation, result


def test_the_analytic_expressions_still_refuse_a_spinor(pseudo_dir):
    """``force_us``/``stres_knl`` are a transcription with no spinor form.

    The functional does spinors; the six hand-derived expressions beside it do
    not, and a refusal that lived only inside the functional would not have
    reached them -- which is the shape of defect the whole refusal sweep is
    about.
    """
    from defumat.stress.analytic import analytic_terms

    calculation, result = _one_iteration("h4-noncolin-force", pseudo_dir)
    with pytest.raises(NotImplementedError, match="noncollinear|spinor"):
        compute_forces(calculation, result, method="analytic")
    with pytest.raises(NotImplementedError, match="noncollinear|spinor"):
        analytic_terms(calculation, state_from_result(result))


def test_energy_at_refuses_a_spinor_unless_asked(pseudo_dir):
    """The default is still a refusal, and that is what guards the consumers.

    :mod:`defumat.response.elastic` calls ``energy_at`` directly and never
    reaches the Sternheimer solver's own ``noncolin`` guard, so the opt-in is
    what keeps a third derivative from inheriting a spinor path its first-order
    wavefunctions do not have.
    """
    calculation, result = _one_iteration("h4-noncolin-force", pseudo_dir)
    with pytest.raises(NotImplementedError, match="noncollinear"):
        energy_at(calculation, state_from_result(result))


def test_the_sternheimer_refusal_still_stands(pseudo_dir):
    """Phonons and the dielectric response are a different missing term.

    ``incdrhoscf_nc``/``set_int3_nc`` are a second implementation rather than a
    spin axis on this one, and that guard is not P46's to lift.
    """
    from defumat.response.sternheimer import require_a_sternheimer_regime

    with pytest.raises(NotImplementedError, match="noncollinear"):
        require_a_sternheimer_regime(_calculation("h4-noncolin-force", pseudo_dir))


def test_a_spiral_force_is_refused_by_name(pseudo_dir):
    """A spiral is ``noncolin`` with two spheres, and P46 writes down one.

    Before the narrowing it was caught by ``reject_spinors`` along with
    everything else; afterwards it would have walked into the spinor projector
    contraction and died on an einsum shape instead of on a refusal.
    """
    calculation = _calculation("h-chain-spiral", pseudo_dir)
    assert calculation.spiral
    with pytest.raises(NotImplementedError, match="spin spiral"):
        reject_spinor_spiral(calculation)
