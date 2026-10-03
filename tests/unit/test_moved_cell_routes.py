"""A moved cell runs its SCF on the constructor's routes, not the strain derivative's.

`OPEN.md` Part XXIII item 15. ``at_cell`` is ``at_strain`` at a concrete
deformation, and ``at_strain`` chooses its augmentation charge for a tape: in
memory mode the exact scanned table, which evaluates the radial transforms of
``Q_ij`` inside every chunk of ``charge()`` and ``integrals()``. A variable-cell
step after the first then paid those transforms twice an SCF iteration at a
``|G|`` that does not change, and built its projector core whole where the
constructor builds it in k-chunks. ``at_cell`` now builds both as the
constructor does, while a strain derivative keeps the scanned table.

The radial transforms are counted as *executions* (a debug callback in front of
the kernel), not as Python calls: the kernel sits inside a compiled scan, which
is traced once and run at every call.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import defumat.pseudo.augmentation as augmentation
import defumat.scf.driver as driver
from defumat.io.pwin import parse_pw_input
from defumat.pseudo import read_upf
from defumat.system.builder import build_system

pytestmark = pytest.mark.unit

#: Ultrasoft silicon on the whole 2x2x2 grid, eight k-points.
CELL = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.2, nat = 2, ntyp = 1,
  ecutwfc = 12.0, ecutrho = 96.0, nosym = .true.
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-n-rrkjus_psl.0.1.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.26 0.24 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""


@pytest.fixture
def radial_evaluations(monkeypatch):
    """A counter of the radial kernel's executions, read after an effects barrier."""
    count = [0]
    original = augmentation._qrad_kernel

    def bump():
        count[0] += 1

    def counted(q, r, weights, functions, prefactor, l):
        jax.debug.callback(bump)
        return original(q, r, weights, functions, prefactor, l=l)

    monkeypatch.setattr(augmentation, "_qrad_kernel", counted)

    def read():
        jax.effects_barrier()
        return count[0]

    return read


#: A compression with a shear in it.
DEFORMATION = np.eye(3) + np.array([[-0.02, 0.01, 0.0], [0.01, -0.01, 0.0],
                                    [0.0, 0.0, 0.005]])


@pytest.fixture
def base(pseudo_dir, monkeypatch):
    """A memory-mode calculation whose projector core is built a k-point at a time.

    The core's chunk is made one k-point, so that a moved core is built in
    eight pieces and each has to be handed its own row of ``kcart``.
    """
    monkeypatch.setattr(driver, "CORE_CHUNK_BYTES", 1)
    system = build_system(parse_pw_input(CELL))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = driver.Calculation(system, pseudos, memory_mode="memory")
    assert base.memory_mode == "memory"
    return base


def _moved(base):
    return base.at_cell(jnp.asarray(np.asarray(base.system.cell.at) @ DEFORMATION.T))


@pytest.fixture
def moved(base):
    """``base`` moved to a compressed, sheared cell."""
    return _moved(base)


def _exercise(charge, ngm):
    """``charge()`` and ``integrals()`` once, on a symmetric ``becsum`` and a potential."""
    rng = np.random.default_rng(1)
    becsum = []
    for atoms, nh in zip(charge.species_atoms, charge.nh_species):
        b = rng.normal(size=(len(atoms), nh, nh))
        becsum.append(jnp.asarray(b + np.swapaxes(b, 1, 2)))
    potential = jnp.asarray(rng.normal(size=ngm) + 1j * rng.normal(size=ngm))
    return charge.charge(tuple(becsum)), charge.integrals(potential)


def test_a_moved_cell_evaluates_no_radial_transform_in_its_iterations(
        moved, radial_evaluations):
    ngm = moved.basis.dense.ngm
    start = radial_evaluations()
    rho, integrals = _exercise(moved.augmentation, ngm)
    assert radial_evaluations() == start, (
        "charge() and integrals() of a moved cell evaluated the radial transforms")

    # The guard can fire: the strain derivative's table evaluates them, and
    # agrees with the stored one to round-off.
    strained = moved.at_strain(jnp.zeros((3, 3)))
    start = radial_evaluations()
    rho_exact, integrals_exact = _exercise(strained.augmentation, ngm)
    assert radial_evaluations() > start, "memory mode's strain derivative must stay scanned"
    np.testing.assert_allclose(np.asarray(rho), np.asarray(rho_exact),
                               rtol=0, atol=1e-13 * float(np.abs(rho).max()))
    for got, want in zip(integrals, integrals_exact):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=0,
                                   atol=1e-12 * float(np.abs(want).max()))


def test_a_moved_core_is_chunked_and_at_the_moved_k_points(base, monkeypatch):
    """Built a k-point at a time, each at its own row of the moved cell's ``kcart``."""
    rows = []
    original = driver.build_projector_core

    def recorded(*args, **kwargs):
        rows.append(args[4].nk)  # the plane-wave basis this piece is built on
        return original(*args, **kwargs)

    monkeypatch.setattr(driver, "build_projector_core", recorded)
    moved = _moved(base)
    assert rows == [1] * base.system.kpoints.nk, f"the moved core was built as {rows}"

    whole = moved.at_strain(jnp.zeros((3, 3))).projector_core
    np.testing.assert_allclose(np.asarray(moved.projector_core.columns),
                               np.asarray(whole.columns), rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.asarray(moved.projector_core.kg),
                               np.asarray(whole.kg), rtol=0, atol=1e-13)
