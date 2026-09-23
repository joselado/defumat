"""A sum-over-states workflow builds its ``Calculation`` once.

``OPEN.md`` Part III, H3. Every workflow here checks its refusals on a
:class:`~defumat.scf.driver.Calculation` before the fixed-density run, so that a
regime it cannot do is refused before the empty states are paid for, and each
of them used to build that calculation, discard it, and let
:func:`~defumat.workflows.nscf.fixed_density_states` build the identical object
again -- and four of them built a third one for ``_default_nbnd`` to read the
electron count off. Each discarded build is the whole constructor: both G sets,
both FFT grids, the symmetry search, the local potential and the projector
core, which is 1.53 s on P69's one-atom Pt PAW cell. The counts were three for
:func:`~defumat.workflows.conductivity.run_conductivity`,
:func:`~defumat.workflows.tddft.run_absorption` and the two magnon entry
points, and two for :func:`~defumat.workflows.shg.run_shg` and
:func:`~defumat.workflows.photocurrent.run_shift_current`.

**How the count is taken without an SCF.** ``Calculation.__init__`` is wrapped
with a counter, and ``Calculation.potential`` is replaced by a sentinel that
raises: it is the first thing ``fixed_density_states`` does with its
calculation that is not a property read, and nothing before it in any of these
workflows touches the density, so the run stops exactly where the old code had
finished building and the new one has built once. A refused regime would *not*
do: the refusal is checked on the first build, so the old code also stops at
one. The cell is therefore one every refusal lets through -- silicon on the
whole unshifted 2x2x2 grid (``nosym``, ``noinv``), insulating for the optical
entry points and with a spin axis and a smearing for the magnon ones.
"""

import functools
import types

import numpy as np
import pytest

from defumat.io.pwin import parse_pw_input
from defumat.pseudo import read_upf
from defumat.scf.driver import Calculation
from defumat.system.builder import build_system

pytestmark = pytest.mark.unit


_SILICON = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.2, nat = 2, ntyp = 1, ecutwfc = 12.0
  nosym = .true., noinv = .true.
  {extra}
/
&electrons
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS crystal
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""

_MAGNETIC = (
    "nspin = 2, starting_magnetization(1) = 0.1, "
    "occupations = 'smearing', degauss = 0.02"
)


class _ReachedTheSolve(Exception):
    """Raised where ``fixed_density_states`` first uses its calculation."""


def _silicon(extra, pseudo_dir):
    system = build_system(parse_pw_input(_SILICON.format(extra=extra)))
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    return system, pseudos


def _conductivity(system, pseudos):
    from defumat.workflows.conductivity import run_conductivity

    run_conductivity(system, pseudos, None)


def _absorption(system, pseudos):
    from defumat.workflows.tddft import run_absorption

    run_absorption(system, pseudos, None, np.array([0.1]))


def _shg(system, pseudos):
    from defumat.workflows.shg import run_shg

    run_shg(system, pseudos, None, nbnd=8)


def _shift_current(system, pseudos):
    from defumat.workflows.photocurrent import run_shift_current

    run_shift_current(system, pseudos, None, nbnd=8)


def _spin_susceptibility(system, pseudos):
    from defumat.workflows.magnons import run_spin_susceptibility

    run_spin_susceptibility(system, pseudos, None, np.zeros(3), np.array([0.01]))


def _magnon_dispersion(system, pseudos):
    from defumat.workflows.magnons import run_magnon_dispersion

    run_magnon_dispersion(
        system, pseudos, None, np.zeros((1, 3)), np.array([0.01])
    )


@pytest.mark.parametrize(
    "entry, extra",
    [
        (_conductivity, ""),
        (_absorption, ""),
        (_shg, ""),
        (_shift_current, ""),
        (_spin_susceptibility, _MAGNETIC),
        (_magnon_dispersion, _MAGNETIC),
    ],
    ids=["conductivity", "absorption", "shg", "shift-current",
         "spin-susceptibility", "magnon-dispersion"],
)
def test_the_refusals_read_the_calculation_the_run_diagonalises_in(
    entry, extra, pseudo_dir, monkeypatch
):
    """One constructor, and the object that reaches the solve is that one.

    The old code gives three builds here for the conductivity, the absorption
    spectrum and both magnon entry points, and two for the second-harmonic and
    shift-current tensors; the fix gives one. The second assertion says the
    one build is not merely the only one but the one ``fixed_density_states``
    goes on to use, which is what makes the refusal a statement about the run
    that follows it.
    """
    system, pseudos = _silicon(extra, pseudo_dir)

    built, reached = [], []
    original = Calculation.__init__

    @functools.wraps(original)
    def counting(self, *args, **kwargs):
        built.append(self)
        original(self, *args, **kwargs)

    def stop(self, *args, **kwargs):
        reached.append(self)
        raise _ReachedTheSolve

    monkeypatch.setattr(Calculation, "__init__", counting)
    monkeypatch.setattr(Calculation, "potential", stop)

    with pytest.raises(_ReachedTheSolve):
        entry(system, pseudos)
    assert len(built) == 1, f"{len(built)} Calculation objects built, not 1"
    assert len(reached) == 1 and reached[0] is built[0]


# -- the argument that carries it, and what it refuses --------------------------


def _a_built_calculation(k_batch=1, david=None):
    """The two attributes the guard reads, and nothing else.

    The guard is the first statement of ``fixed_density_states``, before the
    system, the pseudopotentials or the density are touched, so a whole
    constructor would be the slow way to check a message.
    """
    return types.SimpleNamespace(k_batch=k_batch, david=david)


def test_a_threaded_calculation_and_a_k_set_are_refused_together(pseudo_dir):
    """The calculation's spheres and projectors are already on its own k-set."""
    from defumat.workflows.nscf import fixed_density_states

    system, pseudos = _silicon("", pseudo_dir)
    with pytest.raises(ValueError, match="both a calculation and kpoints"):
        fixed_density_states(
            system, pseudos, None, kpoints=system.kpoints, k_batch=1,
            calculation=_a_built_calculation(),
        )


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"k_batch": 2}, "chunk size is fixed"),
        ({"k_batch": 1, "david": 2}, "Davidson subspace is fixed"),
    ],
    ids=["k_batch", "david"],
)
def test_a_threaded_calculation_built_otherwise_is_refused(
    arguments, message, pseudo_dir
):
    """A build argument that disagrees with the build is not settled silently.

    ``k_batch`` changes only the memory a run takes and ``david`` the Davidson
    subspace, so neither can make a result wrong by more than round-off and
    convergence; the refusal is there because the argument would otherwise be
    accepted and ignored, which is the shape every refusal in this package
    exists to prevent.
    """
    from defumat.workflows.nscf import fixed_density_states

    system, pseudos = _silicon("", pseudo_dir)
    with pytest.raises(ValueError, match=message):
        fixed_density_states(
            system, pseudos, None, calculation=_a_built_calculation(),
            **arguments,
        )
