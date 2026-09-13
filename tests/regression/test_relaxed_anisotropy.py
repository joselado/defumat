"""The magnetocrystalline anisotropy from total energies, not band sums.

:func:`~defumat.workflows.anisotropy.run_anisotropy` freezes the density
converged without spin-orbit coupling and diagonalises once per direction, so
every term of the total energy except the band sum cancels between two
directions by construction. :func:`~defumat.workflows.anisotropy.
run_relaxed_anisotropy` does not cancel anything: it converges a whole
noncollinear run per direction and subtracts two total energies of order 100 Ry
in their eighth decimal.

The two answer slightly different questions. The relaxed one lets the density
respond to the spin-orbit field, which is variational and so lowers both
directions; the anisotropy is the difference of two such gains. **That is why
this is worth its two SCF runs**: nothing bounds the difference between the two
routes in general, and the regime the frozen one cannot reach at all -- PAW,
whose handoff would have to carry a ``becsum`` belonging to a run with a
different pseudopotential file -- is exactly where a production magnet lives.
"""

import warnings

import numpy as np
import pytest

from defumat import Calculator
from defumat.units import RY_TO_EV

pytestmark = [pytest.mark.regression]

PSEUDO = "tests/data/pseudo"
XZ = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))


def _calculator(name):
    return Calculator.from_file(f"tests/data/qe/{name}", pseudo_dir=PSEUDO,
                                announce=False)


@pytest.mark.slow
def test_without_spin_orbit_coupling_every_direction_has_the_same_energy():
    """The identity that pins the whole route, and it is not a weak one.

    Without spin-orbit coupling the Hamiltonian commutes with a global spin
    rotation, so the total energy cannot depend on where the moment points --
    **exactly**, not approximately. Any dependence is machinery: the k-set
    moving between directions, or the gradient-corrected functional's
    quantization axis left behind while the moment is turned, which is the bug
    this same identity found on the frozen route and was worth 36.8 meV there.
    """
    from defumat.workflows.anisotropy import run_relaxed_anisotropy

    calculator = _calculator("co-tetragonal-relaxed-mae.in")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_relaxed_anisotropy(
            calculator.system, calculator.pseudos, directions=XZ,
            soc_scale=0.0, require_spin_orbit=False,
            conv_thr=1.0e-10, max_iterations=200,
        )
    assert result.converged
    spread = abs(result.difference(0, 1)) * RY_TO_EV * 1000.0
    assert spread < 1.0e-3, (
        f"no spin-orbit coupling, so the two directions must have the same "
        f"total energy; they differ by {spread:.3e} meV"
    )


def test_paw_is_refused_by_the_frozen_route_and_allowed_by_the_relaxed_one():
    """The asymmetry is the reason this route exists, so it is asserted.

    The force theorem's handoff is a density and nothing else; a PAW
    Hamiltonian needs ``ddd_paw``, built from a ``becsum`` that belongs to the
    *other* leg's wavefunctions, with a different pseudopotential file and a
    different projector count. A relaxed run hands nothing over, so the refusal
    does not apply -- and this test fails if either half of that ever changes
    silently.
    """
    from defumat.workflows.anisotropy import _refuse_relaxed, _refuse_system

    calculator = _calculator("ni-tetragonal-relaxed-mae-paw.in")
    assert any(p.is_paw for p in calculator.pseudos), "the cell must be PAW"

    with pytest.raises(NotImplementedError, match="PAW"):
        _refuse_system(calculator.system, calculator.pseudos)

    # ... and the relaxed route accepts the identical system.
    _refuse_relaxed(calculator.system, calculator.pseudos)


def test_the_relaxed_route_refuses_what_would_make_its_total_meaningless():
    """Three refusals that are *not* inherited from the frozen route.

    Each is the same argument arriving at the same answer for its own reason,
    which `CLAUDE.md`'s "inherit a refusal only after checking which machine it
    belongs to" is about: a field's energy is outside the reported total, so two
    directions differ by a Zeeman term neither total accounts for; a spiral has
    no spin-orbit coupling to be anisotropic about; and a collinear run has no
    direction to point.
    """
    from defumat.workflows.anisotropy import _refuse_relaxed

    spiral = _calculator("h-fcc-spiral-scan.in")
    with pytest.raises(NotImplementedError, match="spiral"):
        _refuse_relaxed(spiral.system, spiral.pseudos)

    collinear = _calculator("co-tetragonal-anisotropy-sr.in")
    with pytest.raises(ValueError, match="noncolin"):
        _refuse_relaxed(collinear.system, collinear.pseudos)


def test_a_drift_is_reported_rather_than_absorbed():
    """Nothing holds the moment, so the result has to say where it ended.

    A relaxed direction that is not stationary by symmetry can converge with its
    moment somewhere else entirely, and the total energy then belongs to a state
    other than the one asked for. This checks the *reporting*, on synthetic
    results, because the physics of when a drift happens is a property of the
    crystal and not of this code.
    """
    from defumat.workflows.anisotropy import (RELAXED_DRIFT_TOL,
                                              RelaxedAnisotropy,
                                              RelaxedDirection)

    straight = RelaxedDirection(
        direction=(0.0, 0.0, 1.0), total_energy=-100.0, converged=True,
        iterations=12, accuracy=1e-11, moment=(0.0, 0.0, 1.6), drift=0.0,
        moment_length=1.6,
    )
    wandered = RelaxedDirection(
        direction=(1.0, 0.0, 0.0), total_energy=-100.001, converged=True,
        iterations=40, accuracy=1e-11, moment=(0.7, 0.0, 1.4),
        drift=63.4, moment_length=1.565,
    )
    result = RelaxedAnisotropy(
        directions=(straight.direction, wandered.direction),
        results=(straight, wandered),
    )
    assert result.drifts.tolist() == [0.0, 63.4]
    assert result.drifts[1] > RELAXED_DRIFT_TOL
    assert result.converged
    assert np.isclose(result.anisotropy, 0.001)
