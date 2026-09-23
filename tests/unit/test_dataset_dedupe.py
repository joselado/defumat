"""Two species labels naming one UPF file share one setup rather than two copies.

``OPEN.md`` Part III, M2. ``angle1``/``angle2`` are per *species*, so the
standard way to write a noncollinear texture is one species per magnetic site,
all of them naming the same file, and site-resolved DFT+U is the same pattern.
P73 made the augmentation charge ``Q_ij(G)`` a property of the dataset rather
than of the label (``tests/regression/test_uspp.py``,
``test_two_species_naming_one_dataset_share_one_augmentation_charge``); this
file pins the same property for the two setups that still built one copy per
label: the projectors' phase-free columns and the PAW one-centre tensors.

**The assertion is sharing and not agreement**, as it is in P73's test: two
copies that agree numerically still cost twice the memory, which is the whole
defect. For PAW that is object identity. For the projectors it cannot be,
because :class:`~defumat.pseudo.projectors.ProjectorCore` holds one concatenated
``(nk, npwx, ncs)`` array with no per-species block to compare: sharing there
means the second label's channels select the first label's columns, and the
array is one dataset wide.

**Each positive test has a guard beside it that must fire**: two datasets on the
same radial mesh with the same channels, separated only by their content, must
*not* share. A key that returned a constant would pass every positive
assertion here, which is the "null result that reads as a pass" trap.

Nothing here runs an SCF: the setups are built and compared, and the file costs
one basis and a handful of projector builds on two-atom silicon.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from defumat.basis.builder import build_basis
from defumat.io.pwin import read_pw_input
from defumat.paw.onecenter import build_paw
from defumat.pseudo import read_upf
from defumat.pseudo.projectors import build_projector_core, projector_channels
from defumat.system import build_system
from defumat.system.structure import Structure
from defumat.xc.functional import resolve_functional

pytestmark = pytest.mark.unit

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"

#: The smallest committed PAW dataset (``mesh = 1141``, ``nh = 8``), and the
#: cell ``si2-paw.in`` names it in.
PAW = "Si.pz-n-kjpaw_psl.0.1.UPF"

#: The same element from the same generator on the same mesh (``mesh_size =
#: 1141``) with the same four projectors (``l = 0, 0, 1, 1``), generated with a
#: different functional -- so everything but the content agrees with :data:`PAW`.
OTHER = "Si.pbe-n-kjpaw_psl.0.1.UPF"


@pytest.fixture(scope="module")
def silicon():
    system = build_system(read_pw_input(CASES / "si2-paw.in"))
    return system, build_basis(system)


def _two_labels(structure):
    """The same two atoms written as two species labels, ``Si1`` and ``Si2``.

    Built on the one-label cell's own basis and k-points rather than parsed from
    an input with ``ntyp = 2``: two labels on identical atoms remove the
    operation that swaps them, so a parsed input would reduce to a different
    k-set and a byte comparison against the one-label cell would compare two
    different calculations.
    """
    base = structure.species[0]
    return Structure(
        positions=structure.positions,
        types=(0, 1),
        species=(dataclasses.replace(base, name="Si1"),
                 dataclasses.replace(base, name="Si2")),
        precision=structure.precision,
    )


def _core(pseudos, structure, system, basis):
    return build_projector_core(
        pseudos, structure, system.cell, basis.smooth, basis.planewaves,
        system.kpoints,
    )


def _nudged_beta(pseudo):
    """``pseudo`` with one entry of its first ``beta`` moved by one ulp.

    Same mesh, same ``r``, same ``rab``, same channels and the same ``Q``
    functions -- so ``augmentation._dataset_key`` cannot tell it from the
    original, and the projectors are exactly what differs.
    """
    projector = pseudo.projectors[0]
    beta = np.array(projector.beta, copy=True)
    at = int(np.argmax(np.abs(beta[: pseudo.kkbeta])))
    beta[at] = np.nextafter(beta[at], np.inf)
    return dataclasses.replace(
        pseudo,
        projectors=(dataclasses.replace(projector, beta=beta),)
        + tuple(pseudo.projectors[1:]),
    )


# --------------------------------------------------------------------------
# the projectors
# --------------------------------------------------------------------------

def test_two_labels_naming_one_file_share_one_block_of_projector_columns(
        silicon, pseudo_dir):
    system, basis = silicon
    # The same file read twice, as ``Calculator`` reads it: two distinct
    # objects, so nothing can be deduplicated by identity.
    first, second = read_upf(pseudo_dir / PAW), read_upf(pseudo_dir / PAW)
    assert first is not second
    nh = len(projector_channels(first))

    split = _core((first, second), _two_labels(system.structure), system, basis)

    # One dataset's columns, not one block per label ...
    assert split.columns.shape[-1] == nh
    # ... and the second label's atom selects exactly the first label's.
    column_of = np.asarray(split.column_of_channel)
    assert column_of.shape == (2 * nh,)
    assert np.array_equal(column_of[nh:], column_of[:nh])

    # Nothing moved: the same bytes as the same cell written with one label,
    # the columns, the coefficients and the projectors they build.
    single = _core((first,), system.structure, system, basis)
    assert split.columns.shape == single.columns.shape
    assert (np.asarray(split.columns).tobytes()
            == np.asarray(single.columns).tobytes())
    assert np.array_equal(column_of, np.asarray(single.column_of_channel))
    assert split.atom_of_channel == single.atom_of_channel
    assert np.asarray(split.dij).tobytes() == np.asarray(single.dij).tobytes()
    positions = system.structure.positions
    assert (np.asarray(split.at_positions(positions).vkb).tobytes()
            == np.asarray(single.at_positions(positions).vkb).tobytes())


def test_two_datasets_on_one_mesh_keep_their_own_projector_columns(
        silicon, pseudo_dir):
    """The guard fires: only the content separates these, and it must."""
    system, basis = silicon
    lda, pbe = read_upf(pseudo_dir / PAW), read_upf(pseudo_dir / OTHER)
    assert lda.mesh == pbe.mesh
    assert projector_channels(lda) == projector_channels(pbe)
    nh = len(projector_channels(lda))
    structure = _two_labels(system.structure)

    assert _core((lda, pbe), structure, system, basis).columns.shape[-1] == 2 * nh

    # And the sharpest form: one ulp in one ``beta``, which is the one input
    # the augmentation charge's fingerprint does not read.
    nudged = _nudged_beta(lda)
    core = _core((lda, nudged), structure, system, basis)
    assert core.columns.shape[-1] == 2 * nh
    column_of = np.asarray(core.column_of_channel)
    assert np.array_equal(column_of[nh:], column_of[:nh] + nh)


# --------------------------------------------------------------------------
# the PAW one-centre tensors
# --------------------------------------------------------------------------

def test_two_labels_naming_one_file_share_one_paw_species(silicon, pseudo_dir):
    system, _ = silicon
    first, second = read_upf(pseudo_dir / PAW), read_upf(pseudo_dir / PAW)
    assert first is not second
    functional = resolve_functional([first.functional])

    split = build_paw((first, second), _two_labels(system.structure), functional)
    assert split.species[0] is split.species[1]
    # Each label still owns its own atoms, which is what pairs the shared
    # tensors with the right ``becsum``.
    assert split.species_atoms == ((0,), (1,))

    # The shared object is the one a one-label cell builds, array for array.
    single = build_paw((first,), system.structure, functional).species[0]
    shared = split.species[0]
    for name in ("density_ae", "density_ps", "core_ae", "core_ps", "r",
                 "weights_full", "weights_core"):
        assert (np.asarray(getattr(shared, name)).tobytes()
                == np.asarray(getattr(single, name)).tobytes()), name
    assert (shared.nh, shared.nlm, shared.dx) == (single.nh, single.nlm, single.dx)


def test_two_datasets_on_one_mesh_keep_their_own_paw_species(silicon, pseudo_dir):
    """The guard fires, including past ``kkbeta`` where ``Q`` does not look.

    A PAW sphere integrates its energies over the whole mesh, so a difference
    in ``rab`` beyond ``kkbeta`` -- invisible to the augmentation charge, whose
    fingerprint stops there -- changes ``weights_full`` and must keep the two
    datasets apart.
    """
    system, _ = silicon
    structure = _two_labels(system.structure)
    lda, pbe = read_upf(pseudo_dir / PAW), read_upf(pseudo_dir / OTHER)
    functional = resolve_functional([lda.functional])

    paw = build_paw((lda, pbe), structure, functional)
    assert paw.species[0] is not paw.species[1]

    # A relative 1e-9 rather than one ulp: the Simpson weight is ``rab / 3``,
    # and two neighbouring doubles divided by three can round to one.
    rab = np.array(lda.rab, copy=True)
    assert lda.kkbeta < lda.mesh
    rab[-1] *= 1.0 + 1.0e-9
    tail = dataclasses.replace(lda, rab=rab)
    paw = build_paw((lda, tail), structure, functional)
    assert paw.species[0] is not paw.species[1]
    # ... and the nudge is one the tensors see, or the test above is about
    # nothing.
    assert not np.array_equal(np.asarray(paw.species[0].weights_full),
                              np.asarray(paw.species[1].weights_full))


# --------------------------------------------------------------------------
# the local potential, the atomic charge and the core charge
# --------------------------------------------------------------------------

def _nudged(pseudo, field):
    """``pseudo`` with one entry of the radial array ``field`` moved by one ulp."""
    values = np.array(getattr(pseudo, field), copy=True)
    at = int(np.argmax(np.abs(values[: pseudo.msh])))
    values[at] = np.nextafter(values[at], np.inf)
    return dataclasses.replace(pseudo, **{field: values})


def test_two_labels_naming_one_file_share_their_radial_tables(silicon, pseudo_dir):
    """The three per-species tables on the dense grid, one object per dataset.

    Each is a radial transform over every dense G-vector, so a label per site
    paid for it once per label: on ``si8-paw-1k.in`` written as eight labels
    they were 19.8 s of a 26.7 s constructor. A shared entry is the table a
    second transform would have produced, so asserting identity asserts both
    the saving and that nothing moved.
    """
    from defumat.pseudo.potentials import (
        species_atomic_charge, species_core_charge, species_local_potential,
    )

    system, basis = silicon
    first, second = read_upf(pseudo_dir / PAW), read_upf(pseudo_dir / PAW)
    assert first is not second
    pair = (first, second)
    for build in (species_local_potential, species_atomic_charge,
                  species_core_charge):
        tables = build(pair, system.cell, basis.dense)
        assert tables is not None and tables[0] is tables[1], build.__name__


@pytest.mark.parametrize("field, build_name", [
    ("vloc", "species_local_potential"),
    ("rho_atom", "species_atomic_charge"),
    ("rho_core", "species_core_charge"),
])
def test_one_ulp_in_the_radial_array_keeps_two_tables(silicon, pseudo_dir, field,
                                                      build_name):
    """The guard fires: one ulp in the array a table is built from separates it."""
    import defumat.pseudo.potentials as potentials

    system, basis = silicon
    pseudo = read_upf(pseudo_dir / PAW)
    assert getattr(pseudo, field) is not None
    tables = getattr(potentials, build_name)(
        (pseudo, _nudged(pseudo, field)), system.cell, basis.dense)
    assert tables[0] is not tables[1]
