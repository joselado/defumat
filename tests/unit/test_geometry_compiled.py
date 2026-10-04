"""A variable-cell step reuses the compiled force and stress.

`OPEN.md` Part XXIII item 7. The force and stress gradients used to close over
the calculation they were compiled for -- its cell, its radial tables, its
Ewald list -- so ``at_cell`` (every step of a variable-cell relaxation after the
first) compiled both again. They now take every array the geometry moves as an
argument and are keyed on what they still close over
(:class:`~defumat.forces.energy.GeometryKey`).

The step chosen here also *loses* Ewald images (177 at the starting cell, 141
at the expanded one), so it reaches ``at_cell``'s padding of the list to the
starting count: without it the list's length is a new shape and both gradients
compile again for that reason alone. What is checked beside the count of
compilations is the thing the count could hide: that the reused gradient is
the one *at the new geometry*, against a gradient compiled fresh there.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import re
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat.forces.autodiff import autodiff_forces
from defumat.forces.energy import frozen_energy, hoisted, state_from_result, with_hoisted
from defumat.io.pwin import parse_pw_input
from defumat.pseudo import read_upf
from defumat.scf.driver import Calculation, run_scf
from defumat.scf.ewald import build_ewald
from defumat.stress.autodiff import autodiff_stress
from defumat.stress.energy import strained_energy
from defumat.system.builder import build_system

pytestmark = pytest.mark.unit

#: Ultrasoft silicon with a core correction, one atom off its site, in a cell
#: compressed to 0.88 of the equilibrium lattice constant, where its Ewald list
#: holds 177 images.
CELL = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 8.976, nat = 2, ntyp = 1,
  ecutwfc = 12.0, ecutrho = 96.0
/
&electrons
  conv_thr = 1.0d-6
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-n-rrkjus_psl.0.1.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.27 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""

#: The step: the cell expanded by 5 per cent (141 images), and the displaced
#: atom moved further along x.
EXPANSION = 1.05
SHIFT = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])


@contextlib.contextmanager
def counting_compiles():
    """Every XLA compilation inside the block, by name, read off the ``jax`` logger.

    ``Finished XLA compilation`` is logged around ``compile_or_get_cached``, so
    a persistent-cache hit is counted as well as a fresh compile.
    """
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


def _step(base):
    """One variable-cell step from ``base``, built as ``run_vc_relax``'s ``_advance`` builds it."""
    at = np.asarray(base.system.cell.at) * EXPANSION
    positions = np.asarray(base.system.structure.positions) * EXPANSION + SHIFT
    return base.at_cell(jnp.asarray(at)).at_positions(jnp.asarray(positions))


@pytest.fixture(scope="module")
def stepped(pseudo_dir):
    """``(base, moved, state)``: the starting calculation, one cell step, and a frozen state."""
    system = build_system(parse_pw_input(CELL))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = Calculation(system, pseudos)
        state = state_from_result(run_scf(system, pseudos, calculation=base))
    return base, _step(base), state


def test_a_cell_step_compiles_neither_gradient_again(stepped):
    base, moved, state = stepped
    unpadded = build_ewald(moved.system.cell, moved.system.structure,
                           moved.basis.dense, moved.charges).translations.shape[0]
    assert unpadded < base.ewald_sum.translations.shape[0], (
        "the step must lose Ewald images, or the padding is not exercised")
    assert moved.ewald_sum.translations.shape == base.ewald_sum.translations.shape

    # As in a relaxation: the first step's gradients are compiled before the
    # next step's calculation is made, which inherits them.
    first_force = np.asarray(autodiff_forces(base, state))
    first_stress = np.asarray(autodiff_stress(base, state))
    moved = _step(base)
    with counting_compiles() as names:
        force = np.asarray(autodiff_forces(moved, state))
        stress = np.asarray(autodiff_stress(moved, state))
    assert names == [], f"the step compiled {names}"

    # A stale geometry would answer with the first step's numbers.
    assert np.abs(force - first_force).max() > 1e-3
    assert np.abs(stress - first_stress).max() > 1e-4

    # ... and the reused gradients are the ones compiled fresh at the step.
    positions = moved.system.structure.positions
    fresh_force = -jax.jit(jax.grad(
        lambda tau, frozen, big: frozen_energy(with_hoisted(moved, big), tau, frozen,
                                               spinors=True)
    ))(positions, state, hoisted(moved))
    fresh_stress = -jax.jit(jax.grad(
        lambda eps, frozen, big: strained_energy(with_hoisted(moved, big), eps, frozen,
                                                 spinors=True)
    ))(jnp.zeros((3, 3)), state, hoisted(moved)) / moved.system.cell.volume
    np.testing.assert_allclose(force, np.asarray(fresh_force), rtol=0, atol=1e-12)
    np.testing.assert_allclose(stress, np.asarray(fresh_stress), rtol=0, atol=1e-12)


def test_the_key_misses_when_what_the_gradient_closes_over_changes(stepped):
    """The safety is in the comparison: anything captured that differs is a miss."""
    from defumat.forces.energy import split_geometry

    base, moved, _ = stepped
    key, _ = split_geometry(base)
    assert key.matches(split_geometry(moved)[0])

    charged = copy.copy(moved)
    charged.charges = charged.charges * 2.0
    assert not key.matches(split_geometry(charged)[0])

    relabelled = copy.copy(moved)
    relabelled.nelec = moved.nelec + 1.0
    assert not key.matches(split_geometry(relabelled)[0])

    # A geometry field keeps its float leaves out of the key, and its structure in.
    lighter = copy.copy(moved)
    lighter.ewald_sum = build_ewald(moved.system.cell, moved.system.structure,
                                    moved.basis.dense, moved.charges)
    assert key.matches(split_geometry(lighter)[0])
