"""X-ray and magnetic structure factors, against an all-electron code and
against the identities a diffraction pattern has to obey.

``PLAN.md`` P61. The reference is the vendored Elk binary's tasks 195/196, whose
output is committed under ``tests/data/elk/`` (see ``sfac.notes``), and the
comparison is **structured by reflection class rather than blanket**, because a
pseudopotential density is valence-only:

* an **allowed** reflection is core-dominated and is not comparable at all --
  silicon's ``F(000)`` is 8 here and 28 all-electron, and ``(111)`` is 1.75
  against 15.14;
* a **forbidden** reflection is comparable and is the interesting one: the
  spherical part of every atom cancels by symmetry, so the core contributes
  nothing and what is left is the aspherical bonding density, which is the part
  a pseudopotential keeps. Silicon's ``(222)`` agrees to 4%;
* a **magnetic** reflection is comparable where the magnetization is, which is
  outside the pseudisation radius: iron's normalised form factor agrees to 6% at
  the first reflection and drifts to tens of percent by ``|H| ~ 3`` 1/bohr, and
  raising ``ecutwfc`` from 30 to 45 moves it by less than 0.1%, so that drift is
  the pseudopotential rather than the basis.

The checks that need no other code are the sharper half: a spherical-atom
density gives ``F(222) = 0`` to machine precision, so the value is bonding
charge and nothing else; and an antiferromagnet scatters neutrons exactly where
it does not scatter X-rays.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
import pytest

from defumat.basis.builder import build_basis
from defumat.basis.fft import g_to_r
from defumat.diffraction.structure_factor import (
    conventional_transform,
    h_vectors,
    structure_factors_of_field,
)
from defumat.io.pwin import parse_pw_input
from defumat.pseudo.potentials import starting_charge
from defumat.pseudo.upf import read_upf
from defumat.scf.driver import run_scf
from defumat.system.builder import build_system
from defumat.workflows.sfac import run_structure_factors

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
ELK = Path(__file__).resolve().parents[1] / "data" / "elk"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """Three cells, three compilations of the SCF stack -- see ``CLAUDE.md``."""
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def converged(name: str):
    """A converged run of a committed input."""
    system = build_system(parse_pw_input((CASES / name).read_text()))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    return system, pseudos, run_scf(system, pseudos, conv_thr=1.0e-10)


@lru_cache(maxsize=2)
def silicon(hmax: float = 5.0):
    """QE's own silicon at a cutoff and a k-grid the reflections are converged
    on -- ``scf.in``'s 12 Ry and two k-points are neither."""
    system, pseudos, result = converged("si-sfac.in")
    return system, pseudos, result, run_structure_factors(
        system, pseudos, result, hmax=hmax)


def elk_table(name: str) -> dict[tuple[int, ...], float]:
    """``|F|`` per reflection from an Elk ``SFAC*.OUT``, keyed by the sorted
    absolute conventional index -- which is the label the two codes share.

    Their primitive cells do not: QE's ``ibrav = 2`` and Elk's ``avec`` are
    different settings of the same fcc lattice, so a *primitive* triple means
    different things in the two files and the phase of ``F`` is referred to
    different origins. ``|F|`` on the conventional index is what survives that.
    """
    values: dict[tuple[int, ...], float] = {}
    for line in (ELK / name).read_text().splitlines():
        fields = line.split()
        if len(fields) != 8:
            continue
        try:
            key = tuple(sorted(abs(int(v)) for v in fields[:3]))
            modulus = float(fields[7])
        except ValueError:  # the header, and any non-integer (h k l)
            continue
        values.setdefault(key, modulus)
    return values


def by_index(factors, vectors) -> dict[tuple[int, ...], np.ndarray]:
    """Our ``|F|`` keyed the same way, so the two tables can be joined."""
    grouped: dict[tuple[int, ...], list[float]] = {}
    for row in range(len(vectors)):
        key = tuple(sorted(abs(int(round(v))) for v in vectors.indices[row]))
        grouped.setdefault(key, []).append(abs(complex(factors[row])))
    return {key: np.array(value) for key, value in grouped.items()}


# --- the identities ---------------------------------------------------------


def test_the_origin_reflection_counts_the_valence_electrons():
    """``F(0) = int rho`` exactly, which fixes the volume and the ``1/N``.

    Nothing else in the assembly is checked by a number known in advance, and
    a wrong normalisation is invisible in every relative comparison below.
    """
    *_, factors = silicon()
    assert abs(complex(factors.charge[0]) - 8.0) < 1.0e-9


def test_the_space_group_forbidden_reflections_vanish():
    """Diamond's glide forbids ``h00`` unless ``h = 4n``, and it must be exact.

    This is the one place a *non-symmorphic* operation reaches the answer: the
    zero comes from the fractional translation's phase, so a density
    symmetrised without it, or a star collapsed with it, would leave a residue
    here rather than raising anything.
    """
    *_, factors = silicon()
    forbidden = [(0, 0, 2), (0, 2, 4), (0, 0, 6)]
    grouped = by_index(factors.charge, factors.vectors)
    for index in forbidden:
        key = tuple(sorted(index))
        assert grouped[key].max() < 1.0e-9, index
    assert grouped[(1, 1, 1)].min() > 1.0


def test_the_forbidden_222_is_bonding_charge_and_nothing_else():
    """Spherical atoms give ``F(222) = 0``; the crystal gives 0.347.

    The (222) reflection of diamond and silicon is allowed by the space group
    and forbidden for a superposition of spherical atoms, which is why it was
    measured historically: its intensity *is* the aspherical bonding density.
    Here both halves of that statement are computed with the same transform --
    the superposition of free-atom charges is this package's own SCF starting
    guess -- so the comparison isolates the physics from the machinery.
    """
    system, pseudos, _, factors = silicon()
    dense = build_basis(system).dense
    atoms = np.real(np.asarray(g_to_r(
        starting_charge(pseudos, system.structure, system.cell, dense),
        dense.fft_index, dense.grid)))
    spherical = structure_factors_of_field(atoms, system.cell, factors.vectors.miller)

    grouped = by_index(factors.charge, factors.vectors)
    atomic = by_index(spherical, factors.vectors)
    assert atomic[(0, 0, 0)][0] == pytest.approx(8.0, abs=1.0e-9)
    assert atomic[(2, 2, 2)].max() < 1.0e-10
    assert grouped[(2, 2, 2)].min() > 0.30
    # and the same number reached by the label the table prints
    assert abs(factors.of((2, -2, -2))) == pytest.approx(0.347406, abs=1.0e-5)


def test_the_definition_reproduces_the_transform_on_a_converged_density():
    """``method="direct"`` sums the definition over the grid, and shares no
    index arithmetic with the gather that ``"fft"`` does."""
    system, pseudos, result, factors = silicon(3.0)
    direct = run_structure_factors(system, pseudos, result, hmax=3.0, method="direct")
    assert np.abs(direct.charge - factors.charge).max() < 1.0e-12


def test_a_star_is_flat_under_the_operations_it_was_collapsed_with():
    """The reduction's own check, and the glide is the positive control.

    Every image of a representative under a **symmorphic** operation must carry
    the same complex ``F``, computed independently on the unreduced set. Under
    a *glide* the modulus is the same and the phase is not -- which is the
    whole reason :func:`symmorphic_rotations` exists, and asserting it here is
    what makes the first half a real check rather than a tautology.
    """
    from defumat.diffraction.structure_factor import symmorphic_rotations

    system, pseudos, result, reduced = silicon(3.0)
    whole = run_structure_factors(system, pseudos, result, hmax=3.0, reduce=False)
    lookup = {tuple(int(v) for v in m): complex(whole.charge[row])
              for row, m in enumerate(np.asarray(whole.vectors.miller))}
    symmetries = system.symmetry_group()
    symmorphic = symmorphic_rotations(symmetries)

    for row in range(len(reduced)):
        vector = np.asarray(reduced.vectors.miller[row])
        value = complex(reduced.charge[row])
        for rotation in symmorphic:
            image = tuple(int(v) for v in rotation @ vector)
            assert abs(lookup[image] - value) < 1.0e-9, (image, row)

    glides = [(np.asarray(r), np.asarray(f))
              for r, f in zip(symmetries.rotations, symmetries.translations)
              if np.any(np.abs(np.asarray(f)) > 1.0e-6)]
    assert glides, "silicon's diamond glide is what this half is about"
    forbidden = [row for row in range(len(reduced))
                 if sorted(abs(int(round(v))) for v in reduced.vectors.indices[row])
                 == [2, 2, 2]][0]
    vector = np.asarray(reduced.vectors.miller[forbidden])
    value = complex(reduced.charge[forbidden])
    moved = 0
    for rotation, translation in glides:
        image = tuple(int(v) for v in rotation @ vector)
        other = lookup[image]
        assert abs(abs(other) - abs(value)) < 1.0e-9
        if abs(other - value) > 1.0e-3:
            moved += 1
    assert moved, "a glide left every phase alone, which the (222) cannot do"


# --- against the all-electron code ------------------------------------------


def test_the_allowed_reflections_are_core_dominated_and_say_so():
    """Not agreement -- the *size* of the disagreement is the measurement.

    A pseudopotential density carries 8 of silicon's 28 electrons, and the 20
    it drops are the ones that scatter X-rays hardest. Recording that here is
    what keeps the quantity honestly named: this is the valence structure
    factor.
    """
    *_, factors = silicon()
    ours = by_index(factors.charge, factors.vectors)
    theirs = elk_table("si-sfac.elk.out")
    assert theirs[(0, 0, 0)] == pytest.approx(28.0, abs=1.0e-4)
    assert ours[(0, 0, 0)][0] == pytest.approx(8.0, abs=1.0e-9)
    for index in [(1, 1, 1), (0, 2, 2), (0, 0, 4)]:
        assert theirs[index] / ours[index].mean() > 5.0


def test_the_forbidden_reflection_agrees_with_the_all_electron_one():
    """Silicon's (222) against Elk on the same cell and the same k-grid.

    4% between an all-electron LAPW basis and a valence-only plane-wave one, on
    a reflection whose whole value is the bonding charge. What is *not*
    comparable is stated rather than discovered: different bases, different
    LDA parameterisations (Perdew-Zunger here, Perdew-Wang there), and a
    quantity that is a difference of two nearly equal atomic contributions.
    """
    *_, factors = silicon()
    ours = by_index(factors.charge, factors.vectors)[(2, 2, 2)]
    theirs = elk_table("si-sfac.elk.out")[(2, 2, 2)]
    assert theirs == pytest.approx(0.33416, abs=1.0e-4)
    assert ours.max() - ours.min() < 1.0e-9
    assert abs(ours.mean() - theirs) / theirs < 0.05


def test_the_forbidden_reflections_agree_on_being_zero():
    """Both codes give the glide's zeros to their own round-off."""
    *_, factors = silicon()
    theirs = elk_table("si-sfac.elk.out")
    ours = by_index(factors.charge, factors.vectors)
    for index in [(0, 0, 2), (0, 2, 4)]:
        assert theirs[index] < 1.0e-13
        assert ours[index].max() < 1.0e-9


# --- ultrasoft and PAW ------------------------------------------------------

#: The same silicon cell on the three dataset kinds. The cutoffs are each
#: dataset's own; what is compared is a density, and a density is a density.
DATASETS = {
    "norm-conserving": ("Si.pz-vbc.UPF", "ecutwfc = 30.0"),
    "ultrasoft": ("Si.pz-n-rrkjus_psl.0.1.UPF", "ecutwfc = 30.0, ecutrho = 240.0"),
    "PAW": ("Si.pz-n-kjpaw_psl.0.1.UPF", "ecutwfc = 30.0, ecutrho = 240.0"),
}


@lru_cache(maxsize=1)
def silicon_by_dataset():
    """``{kind: {sorted (hkl): |F|}}``, the three run once and kept as numbers.

    The converged states are dropped deliberately: three cells' wavefunctions
    held at once is the accumulation ``CLAUDE.md`` warns about, and every
    statement below is about the factors.
    """
    base = (CASES / "si-sfac.in").read_text()
    out = {}
    for kind, (upf, cutoffs) in DATASETS.items():
        text = base.replace("Si.pz-vbc.UPF", upf).replace("ecutwfc = 30.0", cutoffs)
        system = build_system(parse_pw_input(text))
        pseudos = tuple(read_upf(PSEUDO / s.pseudo_file)
                        for s in system.structure.species)
        result = run_scf(system, pseudos, conv_thr=1.0e-10)
        factors = run_structure_factors(system, pseudos, result, hmax=5.0)
        out[kind] = by_index(factors.charge, factors.vectors)
    return out


def test_ultrasoft_and_paw_put_the_augmentation_charge_in_the_transform():
    """``F(0) = 8`` exactly on all three kinds, and the zeros stay zero.

    For a soft dataset most of the density near the nucleus *is* the
    augmentation charge, so this is the check that it reached the transform at
    all: a run that dropped it would still converge, still have the crystal's
    symmetry, and be short of electrons at the origin.
    """
    for kind, factors in silicon_by_dataset().items():
        assert factors[(0, 0, 0)][0] == pytest.approx(8.0, abs=1.0e-8), kind
        for index in [(0, 0, 2), (0, 2, 4)]:
            assert factors[index].max() < 1.0e-8, (kind, index)


def test_the_augmentation_charge_halves_the_error_on_an_allowed_reflection():
    """Against Elk's **valence-only** run, which is the like-for-like reference.

    Elk's ``wsfac`` rebuilds its density from the states inside an energy
    window, so a window above the 2p core gives an all-electron *valence*
    structure factor -- the same quantity a pseudopotential computes, rather
    than the all-electron total which is 28 electrons against 8. Measured
    against it, ultrasoft and PAW are about twice as close as norm-conserving:
    the augmentation charge is restoring the valence density inside the sphere,
    and this is where that shows up as a number.
    """
    factors = silicon_by_dataset()
    reference = elk_table("si-sfac-valence.elk.out")
    assert reference[(0, 0, 0)] == pytest.approx(8.0, abs=1.0e-4)

    def error(kind, index):
        return abs(factors[kind][index].mean() - reference[index]) / reference[index]

    for index in [(1, 1, 1), (1, 1, 3), (0, 0, 4)]:
        hard = error("norm-conserving", index)
        for soft in ("ultrasoft", "PAW"):
            assert error(soft, index) < 0.6 * hard, (index, soft)
    assert error("ultrasoft", (1, 1, 1)) < 0.01


def test_the_forbidden_reflection_is_the_same_on_every_dataset_kind():
    """And that is a finding rather than a reassurance.

    The (222) comes out at 0.3474 on all three to 3e-4, while the augmentation
    charge is worth a factor of two on the allowed reflections above -- so
    whatever separates all of them from the all-electron 0.3342 is not
    something a softer dataset repairs. The site symmetry says why it could not
    be: a silicon atom in diamond sits at ``-43m``, whose lowest non-spherical
    invariant is ``l = 3``, and an ``s, p`` dataset's augmentation charge
    carries multipoles only to ``L = 2``. All three datasets here are ``s, p``.
    """
    factors = silicon_by_dataset()
    values = [factors[kind][(2, 2, 2)].mean() for kind in DATASETS]
    assert max(values) - min(values) < 1.0e-3
    assert all(abs(v - 0.3474) < 1.0e-3 for v in values)
    for kind, pseudo in DATASETS.items():
        upf = read_upf(PSEUDO / pseudo[0])
        assert upf.lmax == 1, kind


# --- magnetism --------------------------------------------------------------


@lru_cache(maxsize=1)
def iron():
    """Ferromagnetic bcc iron on Elk's own cell, ultrasoft and LSDA."""
    system, pseudos, result = converged("fe-bcc-sfac.in")
    return system, pseudos, result, run_structure_factors(
        system, pseudos, result, hmax=5.0)


def test_the_magnetic_origin_reflection_is_the_total_moment():
    """``F_mag(0) = int m(r)``, the number the SCF prints.

    The magnetic path's own normalisation, independent of the charge one, and
    on an **ultrasoft** dataset -- so it also says the augmentation charge
    reached the magnetization channel, which is where ``becsum`` per channel
    could have been dropped without changing any energy.
    """
    _, _, result, factors = iron()
    assert factors.magnetization is not None
    moment = complex(factors.magnetization[0, 0])
    assert abs(moment.imag) < 1.0e-9
    assert moment.real == pytest.approx(result.magnetization, abs=1.0e-8)


def test_the_magnetic_form_factor_tracks_the_all_electron_one_where_it_can():
    """Iron's normalised ``F_mag(H)/F_mag(0)`` against Elk, by ``|H|``.

    The moments differ (2.214 mu_B here, 2.061 there), so the comparison is of
    the *shape* of the magnetization. It holds to 6% at the first reflection
    and degrades outwards, because iron's moment lives in the 3d shell, inside
    the radius the pseudopotential smooths -- which is what a large ``|H|``
    looks at. Raising ``ecutwfc`` from 30 to 45 moves these by less than 0.1%,
    so the drift is the dataset and not the basis.
    """
    *_, factors = iron()
    ours = by_index(factors.magnetization[:, 0], factors.vectors)
    theirs = elk_table("fe-bcc-sfac-mag.elk.out")
    ratio = lambda table, index: table[index] / table[(0, 0, 0)]
    ours_zero = ours[(0, 0, 0)][0]
    assert abs(ours[(0, 1, 1)].mean() / ours_zero - ratio(theirs, (0, 1, 1))) < 0.05
    assert abs(ours[(0, 0, 2)].mean() / ours_zero - ratio(theirs, (0, 0, 2))) < 0.10
    # and the charge one, on the same run, is core-dominated as silicon's is
    charge = by_index(factors.charge, factors.vectors)
    assert charge[(0, 0, 0)][0] == pytest.approx(8.0, abs=1.0e-8)
    assert elk_table("fe-bcc-sfac-rho.elk.out")[(0, 0, 0)] == pytest.approx(26.0, abs=1e-4)


def test_a_noncollinear_moment_comes_back_as_a_vector():
    """``F_mag(0)`` is the moment *vector* of a ``nspin_mag = 4`` run.

    The same identity as the collinear one, on the branch where the
    magnetization has three components rather than one, and it is the only
    check that the components are carried in the crystal's own axes and in the
    right order. What it does **not** check, and what the module docstring
    states instead, is the star relation: a magnetization is an axial vector,
    so its members are related by ``det(R) R`` rather than being equal, and
    only the representative is reported.
    """
    system, pseudos, result = converged("h4-noncolin-force.in")
    factors = run_structure_factors(system, pseudos, result, hmax=1.5,
                                    transform=None, reduce=False)
    moment = np.asarray(result.magnetization_vector, dtype=float)
    computed = np.asarray(factors.magnetization[0], dtype=complex)
    assert factors.magnetization.shape[1] == 3
    assert np.abs(computed.imag).max() < 1.0e-9
    assert np.abs(computed.real - moment).max() < 1.0e-8


def test_the_antiferromagnet_scatters_neutrons_where_it_scatters_no_x_rays():
    """The magnetic superlattice reflection, in the cheapest cell that has one.

    An antiferromagnetic chain has a charge density of half the magnetic
    period, so the odd reflections along it are **purely magnetic** and the
    even ones purely nuclear. It is Elk's MnO example's point -- reflections
    with zero X-ray and non-zero magnetic intensity -- and it is exact here
    rather than approximate: the operation that halves the charge's period is a
    symmetry of the crystal only together with time reversal.
    """
    system, pseudos, result = converged("h-chain-afm.in")
    factors = run_structure_factors(system, pseudos, result, hmax=2.0,
                                    transform=None, reduce=False)
    along = {int(m[2]): row for row, m in enumerate(np.asarray(factors.vectors.miller))
             if m[0] == 0 and m[1] == 0}
    charge = {l: abs(complex(factors.charge[row])) for l, row in along.items()}
    magnetic = {l: abs(complex(factors.magnetization[row, 0]))
                for l, row in along.items()}

    assert charge[0] == pytest.approx(2.0, abs=1.0e-9)   # two electrons
    assert magnetic[0] < 1.0e-4                          # no net moment
    for odd in (1, -1, 3, -3):
        assert charge[odd] < 1.0e-3
        assert magnetic[odd] > 0.3
    for even in (2, -2):
        assert charge[even] > 0.8
        assert magnetic[even] < 1.0e-3


# --- the energy window ------------------------------------------------------


def test_an_open_window_rebuilds_the_density_the_scf_converged_on():
    """Elk's ``wsfac`` rebuilds the density from the states inside it, so an
    open window must reproduce the run's own density exactly.

    That is the check on the rebuild rather than on the window: the same
    ``sum_band``, the same augmentation and the same symmetrisation, driven
    from the stored wavefunctions instead of from the loop.
    """
    system, pseudos, result, factors = silicon(3.0)
    everything = run_structure_factors(system, pseudos, result, hmax=3.0,
                                       window=(-1.0e6, 1.0e6))
    assert np.abs(everything.charge - factors.charge).max() < 1.0e-8


def test_a_window_counts_exactly_the_states_inside_it():
    """``F(0)`` is the electron count of the window, exactly -- and that count
    is not an integer, which is the thing to know about ``wsfac``.

    A window selects *states*, not bands. Silicon's two lowest valence bands
    touch at ``X``, so no cut separates them: putting one midway between the
    highest point of the first and the lowest of the second leaves 5.965
    electrons rather than 6, and the missing 0.035 is the states that straddle
    it. The identity that holds regardless is that ``F(0)`` is the sum of the
    masked weights, which is what makes the rebuilt density the right one.
    """
    system, pseudos, result, factors = silicon(3.0)
    eigenvalues = np.asarray(result.eigenvalues)
    weights = np.asarray(result.occupations)
    cut = 0.5 * (eigenvalues[:, 0].max() + eigenvalues[:, 1].min())
    inside = float(weights[eigenvalues >= cut].sum())

    upper = run_structure_factors(system, pseudos, result, hmax=3.0,
                                  window=(cut, 1.0e6))
    assert complex(upper.charge[0]).real == pytest.approx(inside, abs=1.0e-8)
    assert 5.9 < inside < 6.0
    assert abs(upper.charge[1] - factors.charge[1]) > 0.1
