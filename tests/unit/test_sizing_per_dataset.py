"""The size estimate counts a dataset once, however many species labels name it.

``OPEN.md`` Part XVII, item 2. One species label per magnetic site is how an
antiferromagnet's ``Fe1``/``Fe2`` and a noncollinear texture are written, and
P110 made the two setups that scale with that pattern hold one copy per
distinct *dataset*: the projectors' phase-free columns
(:func:`~defumat.pseudo.projectors._projector_dataset_key`) and the PAW
one-centre tensors (:func:`~defumat.paw.onecenter._paw_dataset_key`). The
estimate went on summing the first over labels, so a two-label cell reported
twice the columns the run holds, and it had no line at all for the second.

**The one-label and two-label cells are both ``nosym``, and that is
load-bearing.** Two labels on identical atoms remove the operations that swap
them, and with symmetry on that changes the FFT box's divisibility and the
k-set, which moves ``npwx`` and ``nk`` and with them every per-k line: the
comparison would be between two different calculations. ``si2-nosym.in`` takes
the crystal group out of both, and each comparison asserts the counts agree
before it compares bytes.

**Each positive test has a guard beside it that must fire**, as in
``tests/unit/test_dataset_dedupe.py``: a pair of datasets one ulp apart in
something a key reads must be counted twice, and only by the line whose setup
reads it. A key that returned a constant would pass every positive assertion
here, which is the "null result that reads as a pass" trap.

Nothing here runs an SCF. The estimates are host-side ``numpy``; two tests
build the arrays themselves, the projector columns and the PAW tensors, to hold
each line to what the setup allocates rather than to a formula written twice.
"""

import dataclasses
import warnings
from pathlib import Path

import numpy as np
import pytest

from defumat.basis.builder import build_basis
from defumat.calculator import Calculator
from defumat.paw.onecenter import build_paw
from defumat.pseudo.projectors import build_projector_core
from defumat.sizing import estimate_size
from defumat.xc.functional import resolve_functional
# One ulp in the first ``beta``: a different projector dataset, and the same
# PAW sphere, since ``_paw_dataset_key`` does not read ``beta``.
from tests.unit.test_dataset_dedupe import _nudged_beta

pytestmark = pytest.mark.unit

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"

NC = "Si.pz-vbc.UPF"
#: ``nh = 8``, ``l_max_rho = 2`` and ``mesh = 1141``, scalar-relativistic.
PAW = "Si.pz-n-kjpaw_psl.0.1.UPF"

COLUMNS = "projector core columns (nk,npwx,ncs)"
ONECENTRE = "PAW one-centre (nh,nh,nlm,mesh)"

#: Every ``(nh, nh, nlm, mesh)`` tensor a :class:`PawSpecies` can hold.
TENSORS = ("density_ae", "density_ps", "density_rel", "kinetic_ae", "kinetic_ps")


def _once(text, old, new):
    """``text`` with ``old`` replaced, asserting it occurs exactly once.

    The committed input's comment block quotes its own ``nosym = .true.``, so a
    replacement that is not checked can land in a comment and change nothing.
    """
    assert text.count(old) == 1, old
    return text.replace(old, new)


def _input(dataset, two_labels, system=""):
    """``si2-nosym.in`` on ``dataset``, with one label per atom if asked.

    ``system`` is appended to the ``&system`` line that carries ``ecutwfc``.
    """
    text = (CASES / "si2-nosym.in").read_text()
    species = f" Si 28.086 {dataset}\n"
    positions = " Si 0.00 0.00 0.00\n Si 0.25 0.25 0.25\n"
    if two_labels:
        text = _once(text, "ntyp = 1", "ntyp = 2")
        species = f" Si1 28.086 {dataset}\n Si2 28.086 {dataset}\n"
        positions = " Si1 0.00 0.00 0.00\n Si2 0.25 0.25 0.25\n"
    text = _once(text, f" Si 28.086 {NC}\n", species)
    text = _once(text, " Si 0.00 0.00 0.00\n Si 0.25 0.25 0.25\n", positions)
    cutoffs = "ecutwfc = 20.0, ecutrho = 120.0," if dataset == PAW else "ecutwfc = 12.0,"
    return _once(text, "ecutwfc = 12.0,", f"{cutoffs} {system}" if system else cutoffs)


def _calculator(text, pseudo_dir):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Calculator.from_text(text, pseudo_dir, announce=False)


def _estimate(calculator, pseudos=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimate_size(calculator.system, pseudos or calculator.pseudos)


def _like_for_like(first, second):
    """The two estimates size the same k-set on the same sphere."""
    assert (first.nk, first.npwx, first.npw, first.nkb) == (
        second.nk, second.npwx, second.npw, second.nkb
    )


def _nudged_partial_wave(pseudo):
    """``pseudo`` with one entry of an all-electron partial wave moved by one ulp.

    A different PAW sphere and the same projectors:
    ``_projector_dataset_key`` does not read ``ae_wfc``.
    """
    waves = np.array(pseudo.paw.ae_wfc, copy=True)
    at = np.unravel_index(int(np.argmax(np.abs(waves))), waves.shape)
    waves[at] = np.nextafter(waves[at], np.inf)
    return dataclasses.replace(
        pseudo, paw=dataclasses.replace(pseudo.paw, ae_wfc=waves)
    )


def _with_small_component(pseudo):
    """``pseudo`` given a Dirac small component, a copy of its large one.

    Not physics, a shape: ``_build_species`` builds ``density_rel`` from
    ``ae_wfc_rel`` exactly as it builds ``density_ae`` from ``ae_wfc``, so any
    array of that shape reaches the fully relativistic branch. The committed
    datasets that carry ``PP_AEWFC_REL`` (Pt, I, Ni) all have ten projectors
    and ``l_max_rho = 4``, so ``nh = 34`` and ``nlm = 25``, which is about
    0.9 GB of tensors for Pt and no unit test.
    """
    small = np.array(pseudo.paw.ae_wfc, copy=True)
    return dataclasses.replace(
        pseudo, paw=dataclasses.replace(pseudo.paw, ae_wfc_rel=small)
    )


@pytest.mark.parametrize("dataset", [NC, PAW], ids=["nc", "paw"])
def test_two_labels_on_one_file_report_one_block_of_projector_columns(
        dataset, pseudo_dir):
    """``ncs`` counts datasets: two labels on one file are one block."""
    one = _estimate(_calculator(_input(dataset, two_labels=False), pseudo_dir))
    calculator = _calculator(_input(dataset, two_labels=True), pseudo_dir)
    two = _estimate(calculator)
    _like_for_like(one, two)
    assert two.nsp == 2 and one.nsp == 1
    assert two.arrays[COLUMNS] == one.arrays[COLUMNS]

    # And against the array itself: what ``Calculation.__init__`` would keep
    # as ``projector_core`` on the two-label cell, so ``ncs`` is held to the
    # build and not only to the one-label estimate.
    system = calculator.system
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        basis = build_basis(system)
        core = build_projector_core(
            calculator.pseudos, system.structure, system.cell, basis.smooth,
            basis.planewaves, system.kpoints,
        )
    assert two.arrays[COLUMNS] == core.columns.nbytes


def test_two_labels_on_one_paw_file_report_one_set_of_onecentre_tensors(
        pseudo_dir):
    """The one-centre line exists, and it counts datasets rather than labels."""
    one = _estimate(_calculator(_input(PAW, two_labels=False), pseudo_dir))
    two = _estimate(_calculator(_input(PAW, two_labels=True), pseudo_dir))
    assert ONECENTRE in one.arrays
    assert one.arrays[ONECENTRE] > 0
    assert two.arrays[ONECENTRE] == one.arrays[ONECENTRE]
    # And a norm-conserving cell is charged nothing for it.
    nc = _estimate(_calculator(_input(NC, two_labels=True), pseudo_dir))
    assert ONECENTRE not in nc.arrays


@pytest.mark.parametrize(
    "system, meta, relativistic",
    [("", False, False), ("input_dft = 'tb09',", True, False), ("", False, True)],
    ids=["lda", "tb09", "relativistic"],
)
def test_the_onecentre_line_is_what_build_paw_allocates(
        system, meta, relativistic, pseudo_dir):
    """The line against the tensors ``Calculation.__init__`` keeps as ``self.paw``.

    ``build_paw`` is the call the constructor makes, on the same
    pseudopotentials and the functional the run resolves. The line is
    ``n_t`` tensors of one shape, ``n_t = 2 + [ae_wfc_rel] + 2 [meta-GGA]``,
    and each parameter reaches one term of it: a meta-GGA adds the two
    kinetic-energy-density tensors, a fully relativistic dataset the small
    component's ``density_rel``. Each is checked to reach its branch rather
    than assumed to.
    """
    calculator = _calculator(_input(PAW, two_labels=True, system=system), pseudo_dir)
    pseudos = calculator.pseudos
    if relativistic:
        # Two separate copies of equal content, as two labels read from one
        # file are: the key, not the identity, has to make them one dataset.
        pseudos = tuple(_with_small_component(p) for p in pseudos)
    estimate = _estimate(calculator, pseudos)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        functional = resolve_functional(
            [p.functional for p in pseudos], calculator.system.input_dft,
        )
        paw = build_paw(pseudos, calculator.system.structure, functional)
    assert bool(functional.is_meta) == meta

    # P110 shares one object between the two labels, so the run holds one set.
    held = {id(species): species for species in paw.species if species is not None}
    assert len(held) == 1
    (species,) = held.values()
    assert (species.kinetic_ae is not None) is meta
    assert (species.density_rel is not None) is relativistic
    built = sum(
        getattr(species, name).nbytes for name in TENSORS
        if getattr(species, name) is not None
    )
    assert estimate.arrays[ONECENTRE] == built
    tensors = 2 + int(relativistic) + 2 * int(meta)
    assert estimate.arrays[ONECENTRE] == tensors * species.density_ae.nbytes


def test_each_line_is_keyed_on_what_its_own_setup_reads(pseudo_dir):
    """The guard fires, and on the right line.

    One ulp in an all-electron partial wave is a second PAW sphere and the same
    projectors; one ulp in a ``beta`` is the reverse. Each must double exactly
    one of the two lines, so a key that ignored its content, or the two lines
    sharing one key, would show here.
    """
    calculator = _calculator(_input(PAW, two_labels=True), pseudo_dir)
    first, second = calculator.pseudos
    assert first is not second
    shared = _estimate(calculator)

    sphere = _estimate(calculator, (first, _nudged_partial_wave(first)))
    _like_for_like(shared, sphere)
    assert sphere.arrays[ONECENTRE] == 2 * shared.arrays[ONECENTRE]
    assert sphere.arrays[COLUMNS] == shared.arrays[COLUMNS]

    projector = _estimate(calculator, (first, _nudged_beta(first)))
    _like_for_like(shared, projector)
    assert projector.arrays[COLUMNS] == 2 * shared.arrays[COLUMNS]
    assert projector.arrays[ONECENTRE] == shared.arrays[ONECENTRE]
