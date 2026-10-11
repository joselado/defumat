"""A spin spiral on an ultrasoft or PAW dataset, validated against supercells.

The companion of ``test_spin_spirals.py``, which does the same for a
norm-conserving dataset. Everything there about why the validation is an
identity rather than a reference number still holds -- neither ``pw.x`` nor Elk
computes a plane-wave spin spiral -- and this file adds the one thing an
augmented dataset brings with it.

**What is being tested.** A spiral's two spinor components live at ``k + q/2``
and ``k - q/2``, so the transverse block of the density pairs projectors at two
*different* k-points. The augmentation charge that block multiplies is then the
lattice sum ``sum_R e^{-i q.R} Q_ij(r - tau_a - R)`` rather than the periodic
one, and in the rotated frame that is the transform of
``Q_ij(G - q) e^{-i (G - q).tau_a}`` -- the resident table displaced by ``-q``.
The charge and ``m_z`` blocks pair projectors at the same k-point and keep the
resident table. Using the resident table for the transverse block as well
leaves a density that is normalised, plausible and wrong.

**The moment is deliberately in the plane** (``angle1 = 90``), so the whole of
it sits in the transverse channel the displacement governs. With the moment
along ``z`` the transverse block is zero and every test below would pass
without the displaced table existing, which is this repository's
"a check whose null result cannot be told from a pass".

**Why the tolerance is not the norm-conserving one, and why it is not what it
first looked like either.** The identity against the collinear antiferromagnet
leaves a residue that falls with ``ecutrho`` -- -3.61e-07, -5.86e-08,
+6.08e-09 Ry at 200, 300 and 400 -- and is insensitive to ``nbnd`` (10 to 24)
and to ``conv_thr`` (1e-10 to 1e-12), which move it in the fourth digit only.

That looks exactly like a G-sphere truncation, and ``PLAN.md`` P95 records the
argument that says it is one, together with the A/B that refutes it. Running
the *same doubled cell* twice with **no spiral in it at all** -- once as
``nspin = 2`` and once as ``noncolin`` with the two moments antiparallel in the
plane -- gives 7.03e-07, 1.32e-07 and 6.11e-13 at the same three cutoffs, which
is **twice** each residue above, the factor a doubled cell's energy is divided
by. The collinear-referenced identity is therefore measuring the gap between
this code's own collinear and noncollinear paths, and the spiral rides on it.

What is left when that gap vanishes is the spiral's own error: 6.08e-09 Ry at
``ecutrho = 400`` on the resident sphere, against the quarter turn's 1.65e-09
measured at ``ecutrho = 200`` on a supercell that shares the noncollinear
machinery and so never sees the gap at all.

**That 1.65e-09, and PAW's 3.26e-07 beside it, were the truncation shell after
all** (``OPEN.md`` Part XXI item 1, 2026-10-11). With the transverse pair on
the laboratory sphere ``|G - q| < G_rho`` the quarter turn reads **3.5e-12** for
ultrasoft and **2.1e-11** for PAW, and the control arm, the resident spheres,
reproduces 1.641e-09 and 3.255e-07. So PAW's two orders over ultrasoft there
were not its one-centre path; the collinear-referenced identities keep their
gap (3.52e-07 ultrasoft at ``q = b3/2``, against 3.61e-07 before).

:data:`CONSISTENCY_RY` covers the collinear-referenced identity, a floor set by
the code's collinear/noncollinear consistency rather than by the spiral; the
quarter turn, which shares the noncollinear machinery, is held to
:data:`QUARTER_TURN_RY`, which the resident spheres fail.

**What the tolerance has to be read against.** A tolerance is only a test if
something fails it, so each identity is asserted together with the size of the
error the displacement removes: dropping it and using the resident table for
the transverse block leaves the antiferromagnet identity out by **1.15e-03 Ry**
at the same cutoff, three thousand times the residue that is left with it in.
:func:`test_the_resident_table_is_not_enough` is that measurement.
"""

from functools import lru_cache
from pathlib import Path

import pytest

from defumat.io import read_qe_output
from defumat.io.pwin import parse_pw_input
from defumat.pseudo import read_upf
from defumat.scf import run_scf
from defumat.system import build_system
from tests.conftest import GENERATED

pytestmark = [pytest.mark.regression, pytest.mark.slow]

#: The consistency floor described in the module docstring, at the
#: ``ecutrho = 200`` these inputs carry, for the identity against the collinear
#: antiferromagnet: 3.52e-07 measured on ultrasoft (3.61e-07 on the resident
#: sphere), the gap between this code's collinear and noncollinear paths.
#: It was set when PAW's quarter turn read 3.26e-07, which the laboratory
#: sphere took to 2.1e-11 (:data:`QUARTER_TURN_RY`).
#:
#: PAW's collinear/noncollinear gap has been traced as far as it goes and it is
#: not a defect here: ``pw.x`` run on the same control cells has the same gap to
#: four digits (``OPEN.md`` Y4).
CONSISTENCY_RY = 2.0e-06

#: The quarter turn against its four-cell noncollinear supercell, both sides on
#: the same machinery. Measured on D22 (2026-10-11): 3.5e-12 Ry ultrasoft and
#: 2.1e-11 PAW on the laboratory sphere; 1.641e-09 and 3.255e-07 on the resident
#: spheres, which this fails, so it is a test of the sphere and not only of the
#: displacement.
QUARTER_TURN_RY = 1.0e-09

#: What the displaced table is worth. Anything above this and the transverse
#: augmentation charge is simply absent; the measured value is 1.15e-03.
WITHOUT_THE_DISPLACEMENT_RY = 1.0e-04


#: What each control cell may differ from its ``pw.x`` reference by. The
#: reference is QE's stdout, which prints the total energy to eight decimals,
#: so 1e-08 Ry is the number's own resolution and nothing tighter can be
#: asserted against it. Measured: 1.0e-09, 2.2e-09, 5.2e-09 and 8.3e-09 Ry on
#: the four cells.
REFERENCE_RY = 3.0e-08

#: What this code's collinear/noncollinear gap may differ from ``pw.x``'s by.
#: Looser than :data:`REFERENCE_RY` for a bookkeeping reason rather than a
#: physical one: the reference gap is a difference of two eight-decimal
#: numbers, so it carries 1e-08 before anything in either code moves.
#: Measured: 3.2e-09 for ultrasoft and 1.4e-08 for PAW, against gaps of
#: 7.03e-07 and 8.51e-05.
GAP_RY = 4.0e-08


def _pseudos(system, pseudo_dir: Path):
    return tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)


@lru_cache(maxsize=2)
def _run(text: str, pseudo_dir: Path):
    """SCF on an input given as text, so a test can vary ``spiral_q`` in place.

    ``maxsize = 2`` rather than ``None``, which is this repository's bound on a
    converged-state cache: two is what a comparison between two cells needs and
    is the largest that is not a leak (``CLAUDE.md``, "Memory is part of the
    design"). These cells hold spinor wavefunctions on a 60x60x96 box.
    """
    pwin = parse_pw_input(text)
    system = build_system(pwin)
    return system, run_scf(
        system,
        _pseudos(system, pseudo_dir),
        conv_thr=1e-11,
        mixing_beta=0.3,
        max_iterations=300,
    )


def _at_q(name: str, q3: float) -> str:
    text = (GENERATED / name).read_text()
    return text.replace("spiral_q(3) = 0.25", f"spiral_q(3) = {q3}")


def _electronic(result) -> float:
    """The total energy without the Ewald term, per unit cell of the input.

    Two cells of different size do not compute the same Ewald sum to better than
    QE's own truncation tolerance, and that difference would swamp the identity.
    ``test_spin_spirals.py`` has the full reasoning.
    """
    return result.total_energy - result.energy_terms["ewald"]


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """Keep XLA's executable cache from accumulating across these cells.

    Every cell here has its own shapes, so each compiles the whole SCF stack
    afresh and XLA keeps every executable for the life of the process. The
    results stay cached; only the compiled code is dropped.
    """
    yield
    import jax

    jax.clear_caches()


@pytest.mark.parametrize("dataset", ["us", "paw"])
def test_a_zero_spiral_is_an_ordinary_noncollinear_run(dataset, pseudo_dir):
    """``q = 0``: the displaced table *is* the resident one, term for term.

    The cheapest of the identities and the one that isolates the plumbing --
    the doubled k-list, the per-component projectors, the cross ``becsum``
    between two spheres, and the recombination of the two real transverse
    components into one complex field and back. At ``q = 0`` all of it has to
    collapse onto the ordinary noncollinear ultrasoft path that P17 validated,
    and it does so to round-off rather than to the consistency floor below,
    because both sides are then literally the same calculation: the displaced
    table at ``q = 0`` is the resident one, array for array.

    It says nothing about the *displacement*: at ``q = 0`` a wrong sign, a
    wrong magnitude and a missing structure factor are all invisible. The
    supercell identities below are what see those.
    """
    spiral = _at_q(f"o-chain-spiral-{dataset}.in", 0.0)
    plain = spiral.replace(
        "    spiral_q(1) = 0.0, spiral_q(2) = 0.0, spiral_q(3) = 0.0", ""
    )

    system, result = _run(spiral, pseudo_dir)
    plain_system, reference = _run(plain, pseudo_dir)

    assert system.spiral and not plain_system.spiral
    assert result.converged and reference.converged
    assert result.total_energy == pytest.approx(reference.total_energy, abs=1e-10)


@pytest.mark.parametrize("dataset", ["us", "paw"])
def test_a_quarter_turn_is_the_ninety_degree_supercell(dataset, pseudo_dir):
    """``q = b3/4`` against a four-cell noncollinear supercell.

    **The sharp one**, and the only test here that sees the *sign* of the
    displacement. At ``q = b3/2`` the wavevector is its own negative modulo
    ``b3``, so ``Q_ij(G - q)`` and ``Q_ij(G + q)`` differ by a relabelling of
    the G set and the sign is nearly invisible; at ``q = b3/4`` it is not.
    Both sides are noncollinear and augmented, so nothing differs between them
    except whether the transverse augmentation charge is carried by the
    displaced table or by four cells of supercell.

    Measured: **3.5e-12 Ry** for ultrasoft and **2.1e-11 Ry** for PAW, both
    at ``ecutrho = 200`` (2026-10-11, laboratory sphere); 1.65e-09 and 3.26e-07
    before it, which was the shell between the resident sphere and the
    supercell's.
    """
    system, spiral = _run(_at_q(f"o-chain-spiral-{dataset}.in", 0.25), pseudo_dir)
    supercell_system, supercell = _run(
        (GENERATED / f"o-chain-90deg-{dataset}.in").read_text(), pseudo_dir
    )

    assert spiral.converged and supercell.converged
    assert supercell_system.nspin_mag == 4 and not supercell_system.spiral
    assert _electronic(spiral) == pytest.approx(
        _electronic(supercell) / 4.0, abs=QUARTER_TURN_RY
    )


@pytest.mark.parametrize("dataset", ["us", "paw"])
def test_moving_the_atom_does_not_move_the_energy(dataset, pseudo_dir):
    """A one-atom cell is translation invariant, displaced table included.

    The structure factor of the transverse block is ``e^{-i (G - q).tau_a}``
    and not ``e^{-i G.tau_a}``, and every other test here puts the atom at the
    origin, where the two are both 1. This is the one that tells them apart:
    with a wrong phase in either the plane-wave augmentation or PAW's
    one-centre terms, the total energy acquires a dependence on where the atom
    sits.

    **It is also the measurement behind a refusal that was withdrawn.** The
    old refusal of this combination named a per-atom phase ``e^{-i q.tau/2}``
    (Elk's ``zqss``) that PAW's transverse one-centre term was said to need.
    It does not need one: ``becsum`` between the two components already carries
    ``e^{i q.tau}`` through its two structure factors, and the one-centre
    energy depends on ``|m|``, which a position-dependent spin rotation leaves
    alone pointwise. This test is what turns that argument into a number --
    **2.9e-12 Ry** for ultrasoft and **1.4e-11** for PAW; on the laboratory
    sphere (2026-10-11, D22) 3.8e-12 and 4.6e-11, the same as the resident
    spheres on the same machine.

    ``tau_z = 1/3`` rather than an arbitrary displacement, because 1/3 of the
    cell is an exact number of points on both FFT grids here (24 dense, 18
    smooth). At ``tau_z = 0.3``, which is not, the same comparison reads
    4.4e-06 Ry -- the egg-box error of a density sampled on a fixed grid, which
    has nothing to do with the spiral and would otherwise be mistaken for one.
    """
    at_origin = _at_q(f"o-chain-spiral-{dataset}.in", 0.5)
    moved = at_origin.replace(" O 0.0 0.0 0.0", " O 0.0 0.0 0.33333333333333")

    _, here = _run(at_origin, pseudo_dir)
    _, there = _run(moved, pseudo_dir)

    assert here.converged and there.converged
    assert there.total_energy == pytest.approx(here.total_energy, abs=1e-10)


def test_half_a_reciprocal_vector_is_the_antiferromagnet(pseudo_dir):
    """``q = b3/2`` against the collinear LSDA antiferromagnet of the doubled cell.

    Weaker than the quarter turn in two ways worth stating rather than
    discovering: ``q = b3/2`` is its own negative modulo ``b3``, so it is blind
    to the sign of the displacement; and the reference is **collinear**, so the
    two sides do not share the noncollinear machinery and the comparison
    carries that difference as well. It earns its place by tying the augmented
    spiral to the LSDA path rather than to another noncollinear run.

    Measured at 3.61e-07 Ry (3.52e-07 on the laboratory sphere, 2026-10-11),
    and the module docstring's ladder is this identity
    at three cutoffs -- together with the control that shows the ladder is the
    collinear/noncollinear gap rather than the spiral's.
    """
    system, spiral = _run(_at_q("o-chain-spiral-us.in", 0.5), pseudo_dir)
    doubled_system, doubled = _run(
        (GENERATED / "o-chain-afm-us.in").read_text(), pseudo_dir
    )

    assert spiral.converged and doubled.converged
    assert doubled_system.nspin == 2  # the reference is not a spiral at all
    assert _electronic(spiral) == pytest.approx(
        _electronic(doubled) / 2.0, abs=CONSISTENCY_RY
    )


@pytest.mark.parametrize("dataset", ["us", "paw"])
def test_the_two_paths_disagree_by_what_pw_x_disagrees_by(dataset, pseudo_dir):
    """The collinear/noncollinear gap is the reference's gap, not this code's.

    The same doubled cell in the same antiferromagnetic state, run once as
    ``nspin = 2`` and once as ``noncolin`` with both moments turned into the
    plane. There is no spin-orbit coupling, so the energy cannot know which
    axis the moments lie on and the two totals are one number computed by two
    of this code's SCF paths. They do not agree: at ``ecutrho = 200`` the
    ultrasoft pair sit 7.03e-07 Ry apart and the PAW pair 8.51e-05, and PAW's
    is the floor :data:`CONSISTENCY_RY` is set by.

    **``pw.x`` has both paths too, and it has the same gaps** -- 7.03e-07 and
    8.510e-05 on the same inputs -- so what this measures is a property of the
    method both codes implement rather than a defect here. That is the whole
    point of the test: an assertion that the two paths *agree* would fail, and
    an assertion about this code alone could not tell a shared convention from
    a transcription error. Each path is therefore pinned to its own ``pw.x``
    reference, and the two gaps are compared with each other.

    ``OPEN.md`` Y4 has the cutoff ladder, where PAW flattens at 2.65e-06 Ry
    while ultrasoft falls to round-off -- in both codes.
    """
    collinear_system, collinear = _run(
        (GENERATED / f"o-chain-afm-{dataset}.in").read_text(), pseudo_dir
    )
    noncollinear_system, noncollinear = _run(
        (GENERATED / f"o-chain-afm-nc-{dataset}.in").read_text(), pseudo_dir
    )
    assert collinear.converged and noncollinear.converged
    assert collinear_system.nspin == 2 and noncollinear_system.nspin == 4

    reference = {
        name: read_qe_output(GENERATED / f"reference.out.{name}").total_energy
        for name in (f"o-chain-afm-{dataset}", f"o-chain-afm-nc-{dataset}")
    }
    assert collinear.total_energy == pytest.approx(
        reference[f"o-chain-afm-{dataset}"], abs=REFERENCE_RY
    )
    assert noncollinear.total_energy == pytest.approx(
        reference[f"o-chain-afm-nc-{dataset}"], abs=REFERENCE_RY
    )

    gap = collinear.total_energy - noncollinear.total_energy
    reference_gap = (reference[f"o-chain-afm-{dataset}"]
                     - reference[f"o-chain-afm-nc-{dataset}"])
    assert gap == pytest.approx(reference_gap, abs=GAP_RY)


def _cobalt(ecutrho: float, q3: float = -0.25) -> str:
    """The cobalt helix's one-atom cell at another dense cutoff or wavevector."""
    text = (GENERATED / "co-helix4-spiral.in").read_text()
    return (text.replace("ecutrho = 160.0", f"ecutrho = {ecutrho}")
                .replace("spiral_q(3) = -0.25", f"spiral_q(3) = {q3}"))


def _resident_sets(monkeypatch):
    """The control arm: the transverse pair back on the resident spheres.

    Exactly the code before ``OPEN.md`` Part XXI item 1 -- the displaced table
    on ``|G| < G_rho``, the pair interpolated on ``|G| < G_s`` -- since
    :func:`~defumat.scf.driver._spiral_gvectors` is the one place the
    laboratory spheres come from.
    """
    import defumat.scf.driver as driver

    monkeypatch.setattr(
        driver, "_spiral_gvectors",
        lambda basis, cell, qcart, needed: ((basis.dense, basis.smooth) if needed
                                            else (None, None)),
    )


def _cobalt_density(text: str, pseudo_dir):
    """``(system, pseudos, density)``: a cobalt spiral's density after twenty iterations.

    Any density will do for the identities below, which hold for one diagonalisation
    at a fixed density whatever it is; twenty iterations make it a magnetic one
    near the ground state (accuracy about 1e-7 Ry), with the whole transverse
    moment in play. Not converged on purpose: the PBE cobalt helix limit-cycles
    near 1e-10 at 160 Ry on the laboratory sphere and near 1e-7 at 80
    (``OPEN.md`` Part XXI item 1), so a converged state is not what these cells
    offer and not what the identity needs.
    """
    import warnings

    system = build_system(parse_pw_input(text))
    pseudos = _pseudos(system, pseudo_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = run_scf(system, pseudos, conv_thr=1e-11, max_iterations=20)
    return system, pseudos, result.density


def _one_shot(system, pseudos, density, calculation=None, nbnd=None):
    """``(eigenvalues, output density)`` of one diagonalisation at ``density``."""
    import jax.numpy as jnp
    import numpy as np

    from defumat.workflows.nscf import fixed_density_states

    calculation, _, eigenvalues, states = fixed_density_states(
        system, pseudos, jnp.asarray(density), conv_thr=1e-11,
        calculation=calculation, nbnd=nbnd,
    )
    weights, _ = calculation.occupations(jnp.asarray(eigenvalues))
    return np.asarray(eigenvalues), np.asarray(calculation.density(states, weights))


def _supercell_identity(spiral_system, pseudos, density, ecutrho: float, pseudo_dir):
    """``(worst eigenvalue, output magnetization)``: the spiral against its unfolded supercell.

    Both sides diagonalised once at one density -- the spiral at ``density``,
    the four-cell ``co-helix4-nosoc.in`` (``3 3 1 0 0 1``, which samples
    exactly the spiral's ``3 3 4``) at ``unfold_spiral_density`` of it -- and
    compared on two quantities first order in the density: each supercell
    ``(kx, ky)`` band by band against the union of the four spiral k-points over
    it, the top four bands left out, in Ry; and the supercell's output
    magnetization against the unfolded output of the spiral, relative to its
    largest value. The generalized Bloch theorem makes both zero at any
    density, so what is left is the two discretisations' difference.
    """
    import numpy as np

    from defumat.scf.driver import Calculation
    from defumat.workflows.spiral import unfold_spiral_density

    spiral_e, spiral_out = _one_shot(spiral_system, pseudos, density)

    text = (GENERATED / "co-helix4-nosoc.in").read_text().replace(
        "ecutrho = 160.0", f"ecutrho = {ecutrho}")
    system = build_system(parse_pw_input(text))
    supercell_pseudos = _pseudos(system, pseudo_dir)
    calculation = Calculation(system, supercell_pseudos)
    shape = tuple(calculation.basis.dense.grid)
    q = spiral_system.spiral_q
    # Four times the spiral's bands, so the two sides hold the same states: at
    # its own default of 44 against 4 x 16 the supercell's top band carried a
    # weight of 7.7e-5 under the smearing, which moved its Fermi level by 9.4e-7
    # Ry and its output magnetization by 1.3e-5, and read as a discretisation
    # difference.
    super_e, super_out = _one_shot(
        system, supercell_pseudos, unfold_spiral_density(density, q, (1, 1, 4), shape),
        calculation=calculation, nbnd=4 * spiral_e.shape[-1],
    )
    expected = unfold_spiral_density(spiral_out, q, (1, 1, 4), shape)
    magnetization = float(np.abs(super_out[1:] - expected[1:]).max()
                          / np.abs(expected[1:]).max())

    spiral_k = np.asarray(spiral_system.kpoints.coords)
    spiral_e = spiral_e.reshape(len(spiral_k), -1)
    super_k = np.asarray(system.kpoints.coords)
    super_e = super_e.reshape(len(super_k), -1)
    worst = 0.0
    for row, k in zip(super_e, super_k):
        over = np.abs(spiral_k[:, :2] - k[:2]).max(axis=1) < 1e-6
        union = np.sort(spiral_e[over].ravel())
        n = min(len(union), len(row)) - 4
        worst = max(worst, float(np.abs(union[:n] - row[:n]).max()))
    return worst, magnetization


#: What the supercell may differ from the spiral by on the cobalt helix at
#: ``ecutrho = 160`` with the transverse pair on the laboratory sphere: the worst
#: eigenvalue in Ry, and the output magnetization relative to its largest value.
LABORATORY_SPHERE = (1.0e-10, 2.0e-8)

#: The same two with the pair back on the resident spheres, the code before
#: ``OPEN.md`` Part XXI item 1: the control arm has to exceed these, or the
#: bounds above are not a test.
RESIDENT_SPHERE = (1.0e-5, 1.0e-3)


def test_the_spiral_is_its_unfolded_supercell_on_the_laboratory_sphere(pseudo_dir,
                                                                       monkeypatch):
    """The cobalt helix and its four-cell supercell, before and after the laboratory sphere.

    ``co-helix4-spiral.in`` at its own 20/160 Ry (``q = -b3/4``, a separate
    smooth grid, so both the dense and the smooth cut are in play), against
    ``co-helix4-nosoc.in`` at the unfolded density. With the transverse pair on
    the resident spheres the two disagree by the shell ``|G| < G_rho < |G - q|``
    and its smooth counterpart, which held 99.5 per cent of the transverse
    difference; on the laboratory sphere they discretise one functional.

    Measured on D22 (2026-10-11), worst eigenvalue and output magnetization:
    **2.98e-14 Ry and 1.2e-09** on the laboratory sphere, **7.266e-05 Ry and
    4.20e-03** on the resident ones, which is the entry's 7.27e-05 reproduced.
    At 80 Ry, with no smooth grid, the same pair reads 9.8e-14 and 2.9e-09
    against 3.966e-03 and 2.25e-02.
    """
    after = _supercell_identity(*_cobalt_density(_cobalt(160.0), pseudo_dir), 160.0,
                                pseudo_dir)
    _resident_sets(monkeypatch)
    before = _supercell_identity(*_cobalt_density(_cobalt(160.0), pseudo_dir), 160.0,
                                 pseudo_dir)
    assert before[0] > RESIDENT_SPHERE[0] and before[1] > RESIDENT_SPHERE[1]
    assert after[0] < LABORATORY_SPHERE[0] and after[1] < LABORATORY_SPHERE[1]


#: What a calculation moved to a wavevector may differ from one built there, in
#: any eigenvalue (Ry) and in the output magnetization (relative), at one density.
SCAN_POINT = (1.0e-10, 1.0e-9)


def test_a_scan_moves_the_displaced_table_with_q(pseudo_dir, monkeypatch):
    """Every point of an ``E(q)`` scan carries *its own* laboratory sphere and table.

    A scan walks :meth:`~defumat.scf.driver.Calculation.at_spiral_q` with
    ``rebuild_basis = True``, which rebuilds the spheres, ``vkb``, the
    laboratory sphere of the transverse pair and the displaced table on it. A
    point that kept the first point's sphere or table would be **silent**: the
    energies smooth and ordered and their values wrong. So the calculation a
    scan would carry to ``q = -b3/4`` from ``q = 0`` is compared with one built
    at ``q = -b3/4`` from scratch, both diagonalised once at one density.

    **On quantities first order in the density, and on cobalt at 80 Ry**, since
    2026-10-11. The energy on the oxygen chain, which this compared before, is
    second order and could not see a stale *sphere*: the shell there moves the
    energy below 1e-10 Ry. The guard is the same comparison with the resident
    spheres in the fresh calculation, which is what a scan that kept the
    ``q = 0`` sphere would carry. Measured: **0 and 0**, the two calculations
    doing the same arithmetic, against **2.56e-03 Ry and 2.13e-02** for the guard.
    """
    import dataclasses

    import numpy as np

    from defumat.scf.driver import Calculation

    system, pseudos, density = _cobalt_density(_cobalt(80.0), pseudo_dir)
    start = Calculation(dataclasses.replace(system, spiral_q=(0.0, 0.0, 0.0)), pseudos)
    moved = start.at_spiral_q(system.spiral_q)
    assert moved.spiral_dense.ngm == Calculation(system, pseudos).spiral_dense.ngm

    def difference(calculation):
        e_moved, out_moved = _one_shot(system, pseudos, density, calculation=moved)
        e_fresh, out_fresh = _one_shot(system, pseudos, density, calculation=calculation)
        return (float(np.abs(e_moved - e_fresh).max()),
                float(np.abs(out_moved[1:] - out_fresh[1:]).max()
                      / np.abs(out_fresh[1:]).max()))

    same = difference(Calculation(system, pseudos))
    assert same[0] < SCAN_POINT[0] and same[1] < SCAN_POINT[1]

    _resident_sets(monkeypatch)
    stale = difference(Calculation(system, pseudos))
    assert stale[0] > 1000 * SCAN_POINT[0] and stale[1] > 1000 * SCAN_POINT[1]


def test_the_resident_table_is_not_enough(pseudo_dir, monkeypatch):
    """The guard fires: without the displacement the identity fails outright.

    A tolerance is only a test if something fails it, and every number above is
    small. This feeds the same calculation a transverse augmentation charge
    built from the **resident** ``Q_ij(G)`` -- which is what the code did before
    the displaced table existed, and what it would silently go back to if the
    ``shift`` were ever dropped on the way through -- and measures how far out
    that is: **1.15e-03 Ry**, against the 3.6e-07 the displacement leaves.
    """
    import defumat.scf.driver as driver
    from defumat.pseudo.augmentation import build_augmentation

    def without_the_shift(*args, **kwargs):
        kwargs.pop("shift", None)
        return build_augmentation(*args, **kwargs)

    monkeypatch.setattr(driver, "build_augmentation", without_the_shift)

    # Not through ``_run``: its cache is keyed on the text alone and would hand
    # back the correctly built state from another test in the same process.
    def run(text):
        pwin = parse_pw_input(text)
        system = build_system(pwin)
        return run_scf(
            system, _pseudos(system, pseudo_dir), conv_thr=1e-11,
            mixing_beta=0.3, max_iterations=300,
        )

    spiral = run(_at_q("o-chain-spiral-us.in", 0.5))
    doubled = run((GENERATED / "o-chain-afm-us.in").read_text())

    assert spiral.converged and doubled.converged
    error = abs(_electronic(spiral) - _electronic(doubled) / 2.0)
    assert error > WITHOUT_THE_DISPLACEMENT_RY
