"""The exact scanned augmentation table: the stored table's numbers, not its storage.

``GPU-MEMORY-NEXT.md`` item 15. Under a strain the stored route assembles the
whole ``(nh, nh, ngm)`` array on the gradient's tape; the scanned route rebuilds
it a chunk at a time with :class:`~defumat.pseudo.augmentation.ExactRadial` --
the radial integral itself, not the interpolated table -- so its charge, its
``D_ij`` integrals and its ``q_ij`` are the stored route's to round-off, and a
stress taken through it is the derivative of the energy the SCF minimised.
"""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.calculator import Calculator
from defumat.pseudo.augmentation import (
    AugmentationCharge, ExactRadial, TabulatedAugmentation, build_augmentation,
)

pytestmark = pytest.mark.unit

SILICON_US = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.20, nat = 2, ntyp = 1,
  ecutwfc = 20.0, ecutrho = 160.0
/
&electrons
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-n-rrkjus_psl.0.1.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""


def _calculation(pseudo_dir, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Calculator.from_text(SILICON_US, pseudo_dir, announce=False,
                                    **options).calculation


def test_the_scanned_table_is_the_stored_one_to_round_off(pseudo_dir, monkeypatch):
    # A small chunk, so the scan really walks several blocks and a padded tail.
    monkeypatch.setenv("DEFUMAT_AUG_CHUNK", "1000")
    calculation = _calculation(pseudo_dir)
    stored = calculation.augmentation
    system = calculation.system
    scanned = build_augmentation(calculation.pseudos, system.structure,
                                 system.cell, calculation.basis.dense,
                                 scanned=True)
    assert isinstance(stored, AugmentationCharge)
    assert isinstance(scanned, TabulatedAugmentation)
    assert isinstance(scanned.tables[0], ExactRadial)
    assert scanned.chunk == 1000 and scanned.mask.shape[0] > scanned.ngm

    rng = np.random.default_rng(0)
    becsum = []
    for nh, atoms in zip(stored.nh_species, stored.species_atoms):
        block = rng.normal(size=(len(atoms), nh, nh)) * 0.01
        becsum.append(jnp.asarray(block + np.swapaxes(block, 1, 2)))
    becsum = tuple(becsum)
    potential = jnp.asarray(rng.normal(size=scanned.ngm)
                            + 1j * rng.normal(size=scanned.ngm))

    charge = stored.charge(becsum)
    np.testing.assert_allclose(scanned.charge(becsum), charge,
                               rtol=0, atol=1e-14 * float(jnp.abs(charge).max()))
    for a, b in zip(stored.integrals(potential), scanned.integrals(potential)):
        np.testing.assert_allclose(b, a, rtol=0, atol=1e-13 * float(jnp.abs(a).max()))
    for a, b in zip(stored.qq, scanned.qq):
        np.testing.assert_allclose(b, a, rtol=0, atol=1e-14)


@pytest.mark.parametrize("mode, scanned", [("memory", True), ("speed", False)])
def test_the_memory_mode_decides_the_strained_route(pseudo_dir, mode, scanned):
    calculation = _calculation(pseudo_dir, memory_mode=mode)
    strained = calculation.at_strain(jnp.zeros((3, 3)))
    assert isinstance(strained.augmentation, TabulatedAugmentation) is scanned
