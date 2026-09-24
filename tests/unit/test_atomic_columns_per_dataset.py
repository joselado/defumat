"""Two species labels naming one UPF file share one build of the atomic orbitals.

``OPEN.md`` Part XVII item 3. P110 made the projectors' phase-free columns a
property of the dataset rather than of the label
(``tests/unit/test_dataset_dedupe.py``). The pseudo-atomic orbitals are the
same expression with ``chi`` for ``beta``, and they were still built once per
label -- one radial transform over every ``k + G`` of every k-point per label --
for the three things that read them: the starting wavefunctions, DFT+U's
``wfcU`` and the projected density of states. One species per magnetic site is
the standard way to write a noncollinear texture or a site-resolved ``HUBBARD``
card, so a fifteen-site helix paid for fifteen identical transforms.

**The bytes and the count are asserted separately, because only one of them was
wrong.** The bytes are the promise to every consumer: two labels on one file
give the orbitals of the same cell written with one label, atom by atom and in
the same order. That held before as well, since two copies of one transform
agree, so it cannot be the test of the defect; the count is. On the per-label
build the transform ran twice for two labels, and it runs once.

**Beside them, the guard that must fire**: one ulp in one ``chi`` is a second
dataset and gets its own transform. A key that returned a constant would pass
both positive assertions, which is the null result that reads as a pass. The
same nudge is the case the projectors' own key cannot see, since it never reads
an orbital, and that is why the orbitals have a key of their own.

**And the bytes against the build this replaced**, which the two-label
comparison cannot give, since it holds the new code against itself. That build
was per *atom* and not only per label -- one column per atom channel, gathered
with ``arange`` -- so the sharing changes the columns on every cell, the
ordinary one-label cell included, and the promise to every existing consumer is
that the output does not move there. :func:`_per_atom_orbitals` is that build
transcribed on the same private helpers, and it is compared on ``si-1k.in`` as
read and on a two-dataset cell, which is the one that reads a second block of
columns.

Nothing here runs an SCF: two bases for two-atom silicon at one k-point and a
handful of orbital builds.
"""

import dataclasses
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import defumat.pseudo.atomic as atomic
from defumat.basis.builder import build_basis
from defumat.io.pwin import parse_pw_input, read_pw_input
from defumat.pseudo import read_upf
from defumat.pseudo.atomic import (
    atomic_channels,
    atomic_wavefunctions,
    spinor_atomic_wavefunctions,
)
from defumat.pseudo.formfactors import atomic_form_factors
from defumat.pseudo.projectors import (
    _angular_part,
    _apply_phases,
    _projector_dataset_key,
    _radial_table,
    _species_columns,
)
from defumat.system import build_system

pytestmark = pytest.mark.unit

#: QE's two-atom silicon cut to one k-point, with ``Si.pz-vbc``'s 3s and 3p:
#: four orbitals per atom, eight in the cell.
INPUT = Path(__file__).resolve().parents[2] / "benchmarks" / "si-1k.in"


def _replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f"{old!r} is not in {INPUT.name} exactly once"
    return text.replace(old, new)


def _two_label_text() -> str:
    """``si-1k.in`` with its two atoms written as ``Si1`` and ``Si2``.

    Both labels name ``Si.pz-vbc.UPF``, so the cell is one dataset under two
    labels, which is the shape a one-species-per-site input has.
    """
    text = INPUT.read_text()
    text = _replace_once(text, "ntyp=1", "ntyp=2")
    text = _replace_once(
        text, " Si  28.086  Si.pz-vbc.UPF",
        " Si1  28.086  Si.pz-vbc.UPF\n Si2  28.086  Si.pz-vbc.UPF")
    text = _replace_once(text, " Si 0.00 0.00 0.00", " Si1 0.00 0.00 0.00")
    return _replace_once(text, " Si 0.25 0.25 0.25", " Si2 0.25 0.25 0.25")


@pytest.fixture(scope="module")
def silicon(pseudo_dir):
    """The two-label cell, its basis, and the file read once per label.

    The one-label build below is made on **this** basis and these k-points,
    with the species list collapsed, rather than parsed from the untouched
    input: two labels on identical atoms remove the operation that swaps them,
    so the two inputs need not reduce to the same k-set, and a byte comparison
    across two k-sets compares two different calculations. What the untouched
    input is used for is to check that the edit renamed the atoms and moved
    nothing.
    """
    system = build_system(parse_pw_input(_two_label_text()))
    structure = system.structure
    assert [s.name for s in structure.species] == ["Si1", "Si2"]
    assert structure.types == (0, 1)

    original = build_system(read_pw_input(INPUT))
    assert (np.asarray(structure.positions).tobytes()
            == np.asarray(original.structure.positions).tobytes())
    assert (np.asarray(system.cell.at).tobytes()
            == np.asarray(original.cell.at).tobytes())

    # Read once per label, as ``Calculator`` reads them: two distinct objects,
    # so nothing can be shared by identity.
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in structure.species)
    assert pseudos[0] is not pseudos[1]
    return system, build_basis(system), pseudos


@pytest.fixture(scope="module")
def silicon_as_read(pseudo_dir):
    """``si-1k.in`` untouched: two atoms under one label, one dataset."""
    system = build_system(read_pw_input(INPUT))
    assert system.structure.types == (0, 0)
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    return system, build_basis(system), pseudos


def _one_label(structure):
    """The same atoms at the same positions with one label, ``Si1``, for both."""
    return dataclasses.replace(
        structure, types=(0, 0), species=(structure.species[0],))


def _orbitals(pseudos, structure, system, basis):
    return atomic_wavefunctions(
        pseudos, structure, system.cell, basis.smooth, basis.planewaves,
        system.kpoints,
    )


def _per_atom_orbitals(pseudos, structure, system, basis):
    """``atomic_wavefunctions`` as it stood before the sharing, transcribed.

    One radial transform per species label, one column per *atom* channel and
    ``arange`` for the gather, on the same private helpers the module uses, so
    that any difference in the bytes is the assembly and nothing else. The
    ``natomwfc = 0`` branch and the ``kcart`` override are left out, because
    neither is reached here.
    """
    cell, kpoints = system.cell, system.kpoints
    gvectors, planewaves = basis.smooth, basis.planewaves
    channels_by_species = [atomic_channels(p) for p in pseudos]
    lmax = max(l for channels in channels_by_species for _, l, _ in channels)
    kg, kg_norm, ylm = _angular_part(
        gvectors.cartesian(cell), planewaves.indices, kpoints.cartesian(cell),
        lmax)

    shape = kg_norm.shape
    flat = kg_norm.reshape(-1)
    form_factors = tuple(
        atomic_form_factors(p, flat, cell.volume) for p in pseudos)
    radial = _radial_table(form_factors, shape)
    offset = np.cumsum([0] + [f.shape[0] for f in form_factors])

    chi_of, lm_of, l_of, atom_of = [], [], [], []
    for atom, species in enumerate(structure.types):
        for nb, l, lm in channels_by_species[species]:
            chi_of.append(offset[species] + nb)
            lm_of.append(lm)
            l_of.append(l)
            atom_of.append(atom)

    columns = _species_columns(
        ylm, radial, jnp.asarray(chi_of), jnp.asarray(lm_of),
        jnp.asarray((1j) ** np.asarray(l_of)))
    wfc = _apply_phases(
        columns, kg, structure.positions, planewaves.mask,
        jnp.asarray(atom_of), jnp.arange(len(atom_of)))
    return jnp.transpose(wfc, (0, 2, 1)).astype(cell.precision.complex)


def _counted(monkeypatch) -> list:
    """Record every radial transform ``atomic_wavefunctions`` asks for.

    ``atomic`` imports ``atomic_form_factors`` by name, so it is that module's
    binding that is replaced, and the original still does the work.
    """
    calls = []
    real = atomic.atomic_form_factors

    def counted(pseudo, q, omega):
        calls.append(pseudo)
        return real(pseudo, q, omega)

    monkeypatch.setattr(atomic, "atomic_form_factors", counted)
    return calls


def _nudged_chi(pseudo):
    """``pseudo`` with one entry of its first orbital's ``chi`` moved by one ulp.

    Same mesh, same ``r`` and ``rab``, same projectors and the same channels,
    so the projectors' key cannot tell it from the original, and the orbital is
    exactly what differs.
    """
    orbital = pseudo.orbitals[0]
    assert orbital.occupation >= 0.0, "a dropped orbital is not read at all"
    chi = np.array(orbital.chi, copy=True)
    at = int(np.argmax(np.abs(chi[: pseudo.msh])))
    chi[at] = np.nextafter(chi[at], np.inf)
    return dataclasses.replace(
        pseudo,
        orbitals=(dataclasses.replace(orbital, chi=chi),)
        + tuple(pseudo.orbitals[1:]),
    )


def _doubled_orbitals(pseudo):
    """``pseudo`` with every orbital's ``chi`` multiplied by two.

    Not an atom any more, which does not matter for a byte comparison: it is a
    second dataset under the orbitals' key, and its transform is exactly twice the
    original's, since the radial integral is linear in ``chi`` and a factor of
    two commutes with rounding.
    """
    return dataclasses.replace(
        pseudo,
        orbitals=tuple(
            dataclasses.replace(orbital, chi=2.0 * np.asarray(orbital.chi))
            for orbital in pseudo.orbitals),
    )


def test_two_labels_give_the_one_label_orbitals_atom_by_atom(silicon):
    """The same bytes, per atom and in QE's ``natomwfc`` order.

    Every consumer reads this array -- the starting wavefunctions, DFT+U's
    projectors and the projected density of states through
    :func:`atomic_wavefunctions`, the ``j``-resolved projection through
    :func:`spinor_atomic_wavefunctions` -- so the spinor forms are checked too,
    on both of the branches a scalar dataset can take.
    """
    system, basis, pseudos = silicon
    one_label = _one_label(system.structure)
    width = len(atomic_channels(pseudos[0]))
    assert width == 4

    split = np.asarray(_orbitals(pseudos, system.structure, system, basis))
    single = np.asarray(_orbitals(pseudos[:1], one_label, system, basis))
    assert split.shape == single.shape == (
        system.kpoints.nk, 2 * width, basis.planewaves.npwx)
    for atom in range(2):
        block = slice(atom * width, (atom + 1) * width)
        assert split[:, block].tobytes() == single[:, block].tobytes(), (
            f"atom {atom}")

    for lspinorb in (False, True):
        split = np.asarray(spinor_atomic_wavefunctions(
            pseudos, system.structure, system.cell, basis.smooth,
            basis.planewaves, system.kpoints, lspinorb))
        single = np.asarray(spinor_atomic_wavefunctions(
            pseudos[:1], one_label, system.cell, basis.smooth,
            basis.planewaves, system.kpoints, lspinorb))
        assert split.shape == single.shape
        assert split.tobytes() == single.tobytes(), f"lspinorb = {lspinorb}"


def test_one_label_gives_the_bytes_of_the_per_atom_build(silicon_as_read):
    """The ordinary cell: two atoms under one label, as ``si-1k.in`` has them.

    This is the cell every existing consumer runs, and the sharing narrows its
    columns from one per atom channel to one per dataset channel, so it is the
    comparison that says the output did not move where nobody wrote two labels.
    """
    system, basis, pseudos = silicon_as_read
    shared = np.asarray(_orbitals(pseudos, system.structure, system, basis))
    per_atom = np.asarray(
        _per_atom_orbitals(pseudos, system.structure, system, basis))
    assert shared.shape == per_atom.shape == (
        system.kpoints.nk, 8, basis.planewaves.npwx)
    assert shared.tobytes() == per_atom.tobytes()


def test_two_datasets_give_the_bytes_of_the_per_atom_build(silicon):
    """Two labels on two datasets, so the second atom reads the second block.

    The second label's orbitals are doubled, which makes it a dataset of its
    own. That is the case where ``column_offset`` and the dataset slot are not
    zero, and a wrong one would hand the second atom the first atom's orbitals.
    Doubled and not nudged by one ulp, because this comparison needs the two
    blocks to *differ* in the output, and an ulp in one ``chi`` can be rounded
    away in the transform where a factor of two is exact and cannot.
    """
    system, basis, pseudos = silicon
    both = (pseudos[0], _doubled_orbitals(pseudos[0]))
    shared = np.asarray(_orbitals(both, system.structure, system, basis))
    per_atom = np.asarray(
        _per_atom_orbitals(both, system.structure, system, basis))
    assert shared.shape == per_atom.shape
    assert shared.tobytes() == per_atom.tobytes()
    # ... and the second block is not the first, or this is the one-dataset
    # case and a wrong slot would have passed.
    width = len(atomic_channels(pseudos[0]))
    single = np.asarray(_orbitals(pseudos[:1], _one_label(system.structure),
                                  system, basis))
    assert shared[:, width:].tobytes() != single[:, width:].tobytes()


def test_the_radial_transform_runs_once_per_dataset(silicon, monkeypatch):
    """Two labels naming one file: one transform, and it is the first label's."""
    system, basis, pseudos = silicon
    calls = _counted(monkeypatch)
    _orbitals(pseudos, system.structure, system, basis)
    assert len(calls) == 1
    assert calls[0] is pseudos[0]


def test_one_ulp_in_one_orbital_gets_its_own_transform(silicon, monkeypatch):
    """The guard fires: only the content of one ``chi`` separates these."""
    system, basis, pseudos = silicon
    nudged = _nudged_chi(pseudos[0])
    calls = _counted(monkeypatch)
    _orbitals((pseudos[0], nudged), system.structure, system, basis)
    assert len(calls) == 2
    assert calls[0] is pseudos[0] and calls[1] is nudged


def test_the_orbitals_key_reads_what_the_projectors_key_does_not(silicon):
    """Why the orbitals carry a key of their own rather than the projectors'.

    The nudged dataset has the original's projectors, so the projectors' key
    calls the two the same dataset, and sharing on it would hand the second
    label the first label's orbitals.
    """
    from defumat.pseudo.atomic import _atomic_dataset_key

    _, _, pseudos = silicon
    nudged = _nudged_chi(pseudos[0])
    assert _projector_dataset_key(nudged) == _projector_dataset_key(pseudos[0])
    assert _atomic_dataset_key(nudged) != _atomic_dataset_key(pseudos[0])
    # ... and two reads of one file are one dataset under it.
    assert _atomic_dataset_key(pseudos[1]) == _atomic_dataset_key(pseudos[0])
