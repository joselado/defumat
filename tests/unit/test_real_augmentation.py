"""The stored augmentation table is real, and its phase belongs to the pair.

``GPU-MEMORY-NEXT.md`` item 19. ``Q_ij(G) = sum_LM (-i)^L ap(LM,i,j) Y_LM(G)
Q^L_ij(|G|)`` and ``ap`` vanishes unless ``l_i + l_j + L`` is even, so
``Q_ij(G) = (-i)^(l_i + l_j) R_ij(G)`` with ``R`` real. The stored table is ``R``
and the phase is folded in where it is read. The standard is the old complex
construction (:func:`~defumat.pseudo.augmentation._assemble_qgm`), rebuilt here
from the same pieces: the charge, the ``D_ij`` integrals, the complex ones a
displaced table needs, ``q_ij`` and the per-atom charge ``addusforce`` reads, on
an ultrasoft, a PAW and a fully-relativistic PAW dataset -- round-off, not
equality, because the wrong-parity part of ``ap`` is round-off rather than zero
and the real table drops it.
"""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.basis.builder import build_basis
from defumat.basis.gvectors import modulus
from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.pseudo.augmentation import (
    _assemble_qgm, _atom_phases, build_augmentation,
    radial_augmentation_transforms,
)
from defumat.pseudo.coupling import harmonic_products
from defumat.pseudo.harmonics import real_spherical_harmonics
from defumat.pseudo.projectors import projector_channels
from defumat.system.builder import build_system

pytestmark = pytest.mark.unit

CASES = ["si2-us", "si2-paw", "pt-soc-paw-nosym"]


def _setup(case, pseudo_dir):
    system = build_system(read_pw_input(f"tests/data/qe/{case}.in"))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file)
                    for s in system.structure.species)
    return system, pseudos, build_basis(system).dense


def _complex_tables(system, pseudos, dense, shift=None):
    """The old construction: one complex ``(nh, nh, ngm)`` ``Q_ij(G)`` per species."""
    lmax = max(p.lmax for p in pseudos)
    ap = harmonic_products(lmax)
    nl = 2 * lmax + 1
    gcart = dense.cartesian(system.cell)
    if shift is not None:
        gcart = gcart + jnp.asarray(shift, dtype=gcart.dtype)
    ylm = real_spherical_harmonics(gcart, 2 * lmax)
    tables = []
    for pseudo in pseudos:
        channels = projector_channels(pseudo)
        nl_species = min(nl, pseudo.augmentation.nqlc)
        radial = radial_augmentation_transforms(pseudo, modulus(gcart),
                                                system.cell.volume, nl_species)
        lm_of = np.array([lm for _, _, lm in channels])
        tables.append(_assemble_qgm(
            jnp.asarray(ap[:, lm_of[:, None], lm_of[None, :]]), ylm, radial,
            jnp.asarray(np.array([nb for nb, _, _ in channels])), nl_species))
    return tables, gcart


@pytest.mark.parametrize("case", CASES)
def test_the_real_table_is_the_complex_one(case, pseudo_dir):
    system, pseudos, dense = _setup(case, pseudo_dir)
    augmentation = build_augmentation(pseudos, system.structure, system.cell,
                                      dense)
    tables, gcart = _complex_tables(system, pseudos, dense)
    volume = system.cell.volume

    for t, (q, table) in enumerate(zip(tables, augmentation.qgm)):
        assert jnp.isrealobj(table), "the stored table must be real"
        rebuilt = augmentation.pair_phase[t][:, :, None] * table
        scale = float(jnp.max(jnp.abs(q)))
        # The wrong-parity terms the real table drops: round-off.
        assert float(jnp.max(jnp.abs(rebuilt - q))) < 1e-14 * scale
        np.testing.assert_allclose(np.asarray(augmentation.qq[t]),
                                   volume * np.real(np.asarray(q[:, :, 0])),
                                   rtol=0, atol=1e-14 * volume * scale)

    rng = np.random.default_rng(0)
    phases = _atom_phases(gcart, system.structure.positions)
    becsum = []
    reference = 0.0
    for t, (q, atoms) in enumerate(zip(tables, augmentation.species_atoms)):
        nh = q.shape[0]
        b = rng.normal(size=(len(atoms), nh, nh))
        b = jnp.asarray(0.5 * (b + b.transpose(0, 2, 1)))
        becsum.append(b)
        reference = reference + jnp.einsum(
            "aij,ijg,ag->g", b.astype(q.dtype), q, phases[jnp.asarray(atoms)])
        np.testing.assert_allclose(
            np.asarray(augmentation.species_channels(t, b)),
            np.asarray(jnp.einsum("aij,ijg->ag", b.astype(q.dtype), q)),
            rtol=0, atol=1e-13 * float(jnp.max(jnp.abs(q))))
    charge = np.asarray(augmentation.charge(tuple(becsum)))
    np.testing.assert_allclose(charge, np.asarray(reference), rtol=0,
                               atol=1e-13 * np.abs(charge).max())

    potential = jnp.asarray(rng.normal(size=dense.ngm)
                            + 1j * rng.normal(size=dense.ngm))
    for t, (q, atoms) in enumerate(zip(tables, augmentation.species_atoms)):
        shifted = potential[None, :] * jnp.conj(phases[jnp.asarray(atoms)])
        cross = volume * jnp.einsum("ijg,ag->aij", jnp.conj(q), shifted)
        real = np.asarray(jnp.real(cross))
        np.testing.assert_allclose(np.asarray(augmentation.integrals(potential)[t]),
                                   real, rtol=0, atol=1e-13 * np.abs(real).max())
        np.testing.assert_allclose(
            np.asarray(augmentation.cross_integrals(potential)[t]),
            np.asarray(cross), rtol=0, atol=1e-13 * np.abs(np.asarray(cross)).max())


def test_a_displaced_table_is_real_times_the_same_phase(pseudo_dir):
    """``Q_ij(G + b)``: ``G + b`` is a real vector, so the split holds there too.

    The spin spiral's transverse block reads this table with a complex
    ``becsum`` and a complex potential, and ``qq`` is ``Omega Q_ij(b)``, complex.
    """
    system, pseudos, dense = _setup("si2-us", pseudo_dir)
    shift = np.array([0.0, 0.0, -0.31])
    augmentation = build_augmentation(pseudos, system.structure, system.cell,
                                      dense, shift=shift)
    tables, gcart = _complex_tables(system, pseudos, dense, shift=shift)
    q = tables[0]
    scale = float(jnp.max(jnp.abs(q)))
    assert jnp.isrealobj(augmentation.qgm[0])
    np.testing.assert_allclose(
        np.asarray(augmentation.pair_phase[0][:, :, None] * augmentation.qgm[0]),
        np.asarray(q), rtol=0, atol=1e-14 * scale)
    np.testing.assert_allclose(np.asarray(augmentation.qq[0]),
                               system.cell.volume * np.asarray(q[:, :, 0]),
                               rtol=0, atol=1e-14 * system.cell.volume * scale)

    rng = np.random.default_rng(3)
    nat, nh = len(augmentation.species_atoms[0]), q.shape[0]
    becsum = jnp.asarray(rng.normal(size=(nat, nh, nh))
                         + 1j * rng.normal(size=(nat, nh, nh)))
    phases = _atom_phases(gcart, system.structure.positions)
    reference = jnp.einsum("aij,ijg,ag->g", becsum, q, phases)
    charge = np.asarray(augmentation.charge((becsum,)))
    np.testing.assert_allclose(charge, np.asarray(reference), rtol=0,
                               atol=1e-13 * np.abs(charge).max())


def test_the_stored_table_is_half_the_complex_one(pseudo_dir):
    """The resident bytes, and the sizing estimate that counts them."""
    system, pseudos, dense = _setup("pt-soc-paw-nosym", pseudo_dir)
    augmentation = build_augmentation(pseudos, system.structure, system.cell,
                                      dense)
    nh = augmentation.qgm[0].shape[0]
    complex_bytes = nh * nh * dense.ngm * system.cell.precision.complex.itemsize
    assert augmentation.qgm[0].nbytes * 2 == complex_bytes
