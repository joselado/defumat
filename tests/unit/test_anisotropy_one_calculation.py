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

import contextlib
import functools
import logging
import re

import jax
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

    with pytest.warns(UserWarning, match="already has a torque below"):
        relax_orientation(system, pseudos, density, curvature=True)

    assert len(solved) == 7, "the relaxation needs more than one one-shot to count"
    assert len(built) == 1, f"{len(built)} Calculation objects built, not 1"
    assert all(s.basis is solved[0].basis for s in solved)


@contextlib.contextmanager
def _compiles():
    """The name of every XLA compilation inside the block, persistent-cache hits included."""
    names = []

    class Grab(logging.Handler):
        def emit(self, record):
            found = re.search(r"Finished XLA compilation of (.+?) in", record.getMessage())
            if found:
                names.append(found.group(1))

    handler, logger = Grab(), logging.getLogger("jax")
    before = jax.config.jax_log_compiles
    jax.config.update("jax_log_compiles", True)
    logger.addHandler(handler)
    try:
        yield names
    finally:
        logger.removeHandler(handler)
        jax.config.update("jax_log_compiles", before)


def test_a_turned_texture_compiles_the_potential_once(cobalt):
    """Two orientations of a gradient-corrected spinor run, one potential executable.

    The quantization axis turns with the texture (:func:`_with_quantization_axis`)
    and the potential reads it for the sign of ``m . u_x``. It used to be a
    static argument of the compiled potential, so every orientation of a scan
    or a relaxation compiled the potential again; it is an array argument now.
    """
    from defumat.scf.driver import _potential_of_rho

    system, pseudos, _ = cobalt
    first = Calculation(_with_quantization_axis(system, (1.0, 0.0, 0.0)), pseudos)
    second = first.with_texture(_with_quantization_axis(system, (0.0, 1.0, 0.0)))
    assert first.functional.is_gradient
    assert first.quantization_axis != second.quantization_axis
    # A texture whose x component changes sign along the first lattice vector
    # and whose y component changes sign along the second, so that the sign of
    # m . u_x has a different pattern of nodes along the two axes. (A uniform
    # flip would not do: the functional is even under a global sign.)
    density = first.starting_density()
    size_a, size_b = density.shape[1], density.shape[2]
    along_a = jnp.cos(2.0 * jnp.pi * jnp.arange(size_a) / size_a)[:, None, None]
    along_b = jnp.cos(2.0 * jnp.pi * jnp.arange(size_b) / size_b)[None, :, None]
    moment = density[1]
    density = density.at[1].set(moment * along_a).at[2].set(moment * along_b)

    _potential_of_rho.clear_cache()
    with _compiles() as names:
        one = first.potential(density).v_scf
        other = second.potential(density).v_scf
    assert names.count("jit(v_of_rho)") == 1, names
    # The axis is an argument the executable reads rather than one it lost.
    assert not np.array_equal(np.asarray(one), np.asarray(other))


_OXYGEN_PAW = """
&control
   calculation = 'scf'
/
&system
   ibrav = 1, celldm(1) = 6.0, nat = 1, ntyp = 1,
   noncolin = .true., nosym = .true.,
   ecutwfc = 10.0, ecutrho = 40.0,
   occupations = 'smearing', smearing = 'gaussian', degauss = 0.05,
   starting_magnetization(1) = 0.5,
/
&electrons
/
ATOMIC_SPECIES
O 16.0 O.pbe-kjpaw.UPF
ATOMIC_POSITIONS crystal
O 0.0 0.0 0.0
K_POINTS automatic
1 1 1 0 0 0
"""


@pytest.mark.slow
def test_a_turned_texture_traces_to_the_same_program(pseudo_dir):
    """What traces the potential and the one-centre terms holds no axis in its program.

    The torque's derivative is a program :mod:`defumat.eager` keeps by the
    printed jaxpr and the bytes of every constant nested in it, and it traces
    both potentials. The axis used to reach the grid potential as a static
    argument, a constant nested in that jaxpr, and the PAW one-centre terms as
    three Python floats, which print as literals: so the kept torque differed
    at every orientation, on PAW for the second reason as well as the first.
    Both are a hoisted array now, an argument of the kept program, and the two
    orientations trace to one key.
    """
    from defumat import eager

    calculator = Calculator.from_text(_OXYGEN_PAW, pseudo_dir, announce=False)
    system, pseudos = calculator.system, calculator.pseudos
    first = Calculation(_with_quantization_axis(system, (1.0, 0.0, 0.0)), pseudos)
    second = first.with_texture(_with_quantization_axis(system, (0.0, 1.0, 0.0)))
    assert first.is_paw and first.functional.is_gradient
    assert first.quantization_axis != second.quantization_axis
    density, becsum = first.starting_density(), first.starting_becsum()

    def key(calculation):
        def both(rho, values):
            return (calculation.potential(rho).v_scf,
                    calculation.onecenter(values)[1])

        closed = jax.make_jaxpr(both)(density, becsum)
        return str(closed.jaxpr), eager._nested_digest(closed.jaxpr)

    assert key(first) == key(second)


@contextlib.contextmanager
def _per_one_shot(monkeypatch):
    """Each one-shot's XLA compilations by name, and the kept programs before it.

    The names are read off the ``jax`` logger, which sees a persistent-cache
    hit as well as a fresh compile. The count of :mod:`defumat.eager`'s kept
    programs is read as each one-shot starts, so after whatever the one
    before it dropped.
    """
    from defumat import eager
    from defumat.workflows import anisotropy

    shots = {"compiled": [], "kept": []}
    original = anisotropy._orientation_torque

    with _compiles() as names:
        def shot(*args, **kwargs):
            shots["kept"].append(len(eager._PROGRAMS))
            start = len(names)
            result = original(*args, **kwargs)
            shots["compiled"].append(names[start:])
            return result

        monkeypatch.setattr(anisotropy, "_orientation_torque", shot)
        yield shots


@pytest.mark.slow
def test_an_orientation_relaxation_keeps_what_every_one_shot_shares(cobalt, monkeypatch):
    """No global clear, and nothing compiled again after the first one-shot.

    The relaxation used to call ``jax.clear_caches()`` after every one-shot,
    which on tetragonal cobalt meant 79 compilations a one-shot, the
    eigensolver's and the Hamiltonian's among them. Without it two programs
    were still new at every orientation: the potential, whose quantization
    axis was a static argument, and the torque's derivative, which traced
    that potential and held the axis as a nested constant, so kept they
    accumulated, 1755 mappings a one-shot. The axis is an array argument now.
    The torque and the band sum here are the real ones, evaluated on the
    stand-in's zero states, which give a zero torque and stop the relaxation
    at its first step: the seven one-shots, at six different axes, must
    compile at the first one-shot only, keep the same programs after every
    one-shot, and hold one potential executable.
    """
    from defumat import eager
    from defumat.scf.driver import _potential_of_rho

    system, pseudos, density = cobalt
    _stand_ins(monkeypatch)
    cleared = []
    monkeypatch.setattr("jax.clear_caches", lambda: cleared.append(True))
    eager.clear()
    _potential_of_rho.clear_cache()
    with _per_one_shot(monkeypatch) as shots:
        with pytest.warns(UserWarning, match="already has a torque below"):
            relax_orientation(system, pseudos, density, curvature=True)
    kept = shots["kept"] + [len(eager._PROGRAMS)]
    eager.clear()

    assert len(shots["compiled"]) == 7, "the relaxation needs more than one one-shot"
    assert cleared == [], "the relaxation dropped every compiled program"
    assert "jit(run)" in shots["compiled"][0], "the first one-shot kept no torque"
    for index, names in enumerate(shots["compiled"][1:], start=2):
        assert names == [], f"one-shot {index} compiled {names}"
    assert kept[1:] == [kept[1]] * (len(kept) - 1), f"kept programs per one-shot {kept}"
    assert _potential_of_rho._cache_size() == 1
