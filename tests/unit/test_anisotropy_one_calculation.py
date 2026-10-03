"""A force-theorem scan and an orientation relaxation build one ``Calculation``.

``OPEN.md`` Part XXIII, item 14. Every direction of
:func:`~defumat.workflows.anisotropy.run_anisotropy` and every one-shot of
:func:`~defumat.workflows.anisotropy.relax_orientation` used to build the
one-shot leg's whole ``Calculation`` again, although only the magnetic texture
moves between them and turning it requires ``nosym``, so the k-set, the G sets,
the projectors and every table are the same. Now the first builds it and the
rest turn it with :meth:`~defumat.scf.driver.Calculation.with_texture`.

**How the count is taken without diagonalising.** ``Calculation.diagonalize``
is replaced by a stand-in that returns ordered levels and zero states of the
right shapes and records the calculation it was called on, and the torque by
zeros, so a scan runs end to end through every constructor the old code made
(three for ``"xyz"``, seven for a relaxation that stops at its first step and
then takes its six curvature one-shots) at the cost of the setup alone. What
the stand-in records is also checked against a fresh build of the turned
system: the quantization axis a gradient-corrected functional reads, which is
the one thing that moving rather than building could get wrong silently.

The cell is the cubic one-atom cobalt of ``tests/regression/test_anisotropy.py``,
noncollinear with the fully-relativistic dataset and ``nosym``.
"""

import functools

import jax.numpy as jnp
import numpy as np
import pytest

from defumat import Calculator
from defumat.scf.driver import Calculation
from defumat.workflows.anisotropy import (
    _with_quantization_axis,
    relax_orientation,
    run_anisotropy,
)

pytestmark = pytest.mark.unit


_COBALT = """
&control
   calculation = 'nscf'
/
&system
   ibrav = 1, celldm(1) = 5.0, nat = 1, ntyp = 1,
   noncolin = .true., lspinorb = .true., lforcet = .true., nosym = .true.,
   ecutwfc = 12.0, ecutrho = 96.0,
   occupations = 'smearing', smearing = 'mv', degauss = 0.02,
   starting_magnetization(1) = 0.5,
/
&electrons
/
ATOMIC_SPECIES
Co 58.933 Co.rel-pbe-nd-rrkjus.UPF
ATOMIC_POSITIONS crystal
Co 0.0 0.0 0.0
K_POINTS automatic
2 2 2 0 0 0
"""


@pytest.fixture(scope="module")
def cobalt(pseudo_dir):
    calculator = Calculator.from_text(_COBALT, pseudo_dir, announce=False)
    system, pseudos = calculator.system, calculator.pseudos
    # A collinear density along z, as the scalar-relativistic leg hands over.
    noncollinear = Calculation(system, pseudos).starting_density()
    density = jnp.stack([(noncollinear[0] + noncollinear[3]) / 2.0,
                         (noncollinear[0] - noncollinear[3]) / 2.0])
    return system, pseudos, density


def _stand_ins(monkeypatch):
    """Count the constructors, and record every calculation a solve is asked in."""
    built, solved = [], []
    original = Calculation.__init__

    @functools.wraps(original)
    def counting(self, *args, **kwargs):
        built.append(self)
        original(self, *args, **kwargs)

    def diagonalize(self, hamiltonians, nbnd, psi0=None, ethr=None,
                    return_steps=False):
        solved.append(self)
        nk = self.system.kpoints.nk
        width = self.npol * self.basis.planewaves.mask.shape[-1]
        levels = jnp.broadcast_to(0.1 * jnp.arange(nbnd, dtype=float), (1, nk, nbnd))
        states = jnp.zeros((1, nk, nbnd, width), dtype=complex)
        if not return_steps:
            return levels, states
        return levels, states, jnp.zeros((1, nk), int), jnp.zeros((1, nk), int)

    monkeypatch.setattr(Calculation, "__init__", counting)
    monkeypatch.setattr(Calculation, "diagonalize", diagonalize)
    return built, solved


def test_a_force_theorem_scan_builds_one_calculation(cobalt, monkeypatch):
    """Three directions, one constructor, and each solve at its own axis."""
    system, pseudos, density = cobalt
    axes = np.eye(3)
    turned = [_with_quantization_axis(system, tuple(axis)) for axis in axes]
    fresh = Calculation(turned[1], pseudos)
    built, solved = _stand_ins(monkeypatch)

    run_anisotropy(system, pseudos, density, directions="xyz")

    assert len(solved) == 3, "the scan needs more than one direction to count"
    assert len(built) == 1, f"{len(built)} Calculation objects built, not 1"
    # The first direction, x, is the build and y is turned from it: its axis
    # and its magnetic group are a fresh build's of y, not the build's of x.
    assert solved[1].quantization_axis != solved[0].quantization_axis
    assert solved[1].quantization_axis == fresh.quantization_axis
    assert solved[1].symmetries.nsym == fresh.symmetries.nsym
    for one, system_ in zip(solved, turned):
        assert (one.system.angle1, one.system.angle2) == (system_.angle1, system_.angle2)
    assert all(s.basis is solved[0].basis for s in solved)


def test_an_orientation_relaxation_builds_one_calculation(cobalt, monkeypatch):
    """A first step and six curvature one-shots, one constructor."""
    system, pseudos, density = cobalt
    built, solved = _stand_ins(monkeypatch)
    monkeypatch.setattr("defumat.forces.torque.orientation_torque",
                        lambda *args, **kwargs: np.zeros(3))
    monkeypatch.setattr("defumat.forces.torque.band_energy_at_rotation",
                        lambda *args, **kwargs: 0.0)
    # The relaxation drops the compiled code after every one-shot, which here
    # would only recompile the potential seven times.
    monkeypatch.setattr("jax.clear_caches", lambda: None)

    with pytest.warns(UserWarning, match="already has a torque below"):
        relax_orientation(system, pseudos, density, curvature=True)

    assert len(solved) == 7, "the relaxation needs more than one one-shot to count"
    assert len(built) == 1, f"{len(built)} Calculation objects built, not 1"
    assert all(s.basis is solved[0].basis for s in solved)
