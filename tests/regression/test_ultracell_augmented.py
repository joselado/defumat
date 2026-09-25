"""P88 stage 5: the ultracell with an ultrasoft or PAW dataset.

The claim the phase adds is that the augmentation charge changes **nothing
structural** about the method -- the frozen states stay a fixed basis, the
overlap operator stays the identity across ``Q``, and the only new object is
the resident table displaced by a Q-vector. So the checks here are the same
three the norm-conserving file makes, run on a dataset that carries charge
inside the projector spheres, and the reference is the same one: an ``N``-cell
supercell run through this package's own SCF, which shares the unit-cell
machinery and none of the ultracell assembly.

**The tiled null cannot see the sign and the supercell can**, which is why both
are here and why the second is the one that decides. With nothing applied every
``becsum`` at a non-zero Q-difference is identically zero, so the whole set of
displaced tables is multiplied by nothing and a null that passes says only that
the ``Q_d = 0`` table and the normalisation are right. Hermiticity does not
discriminate either -- the opposite index order in the pair ``(Q, Q')`` is
Hermitian too. What separates them is the *modulated* comparison, and it was
measured: with the sign flipped the induced density's error stalls at 6.7e-2
where it falls to 3.7e-3, and the total energy drops 2.6e-5 Ry **below** the
supercell's, which the variational argument forbids. One rung alone would not
have caught it, since at ``nbnd = 12`` the wrong sign sits nearer the supercell
than the right one.

**The transmission is here too, and it is a refusal that dissolved rather than a
term that was written.** ``run_ultracell_transport`` refused an augmented
dataset outright, on the reading that an exit-plane Gram matrix needs the
overlap operator ``S``. It does not: the plane sits in the vacuum, where a
pseudo-wavefunction is the true one, and the unit cell hands ``S`` to the
*whole-cell* diagnostic alone. So the checks the norm-conserving file makes of
that quantity are made here on a dataset that carries charge inside the
spheres, on a sheet in vacuum rather than on bulk silicon -- because with an
augmentation sphere of 2.6 bohr against silicon's 5.9 bohr interplanar spacing
there is no height in that cell a tip may legally sit at, which the guard says
by name.

This is a separate file from ``test_ultracell.py`` rather than a section of it,
and the reason is the runner: ``tools/run_regression.sh`` invokes pytest once
per file, so a file boundary there is a *process* boundary, and these two
datasets each bring a unit cell and a four-atom supercell that share no shape
with anything already in that file.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from defumat import Calculator
from defumat.basis.builder import build_basis
from defumat.scf.driver import Calculation, run_scf
from defumat.ultracell import run_ultracell, with_external_potential
from defumat.workflows.transport import run_vertical_transport
from defumat.workflows.ultracell import run_ultracell_transport

pytestmark = pytest.mark.regression


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """Bound the peak: four distinct cells here, none sharing a shape."""
    yield
    jax.clear_caches()


#: The applied modulation, in Ry -- ``test_ultracell.py``'s, so the two ladders
#: are comparable rung for rung.
AMPLITUDE = 0.05

#: The two LDA datasets committed for silicon. The functional has to be an LDA
#: because the ultracell refuses a gradient-corrected one by name, and both of
#: these are ``pz``.
DATASETS = {
    "ultrasoft": "Si.pz-n-rrkjus_psl.0.1.UPF",
    "paw": "Si.pz-n-kjpaw_psl.0.1.UPF",
}

#: ``ecutrho = 4 ecutwfc`` and not the dataset's own default of 8 to 12, which
#: is a choice here rather than a limit: both sides of every comparison run at
#: the same pair, so what is measured is the method rather than the
#: representation of the augmentation charge, and the smaller dense grid is
#: cheaper. The dual runs too and has its own section below.
ECUTWFC, ECUTRHO = 16.0, 64.0

SILICON = """&control
 calculation='scf'
/
&system
 ibrav=2, celldm(1)=10.20, nat=2, ntyp=1,
 ecutwfc={ecutwfc:.1f}, ecutrho={ecutrho:.1f},
 nosym=.true., noinv=.true.
/
&electrons
 conv_thr=1.0d-12
/
ATOMIC_SPECIES
 Si 28.086 {upf}
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 {k0} {k1} {k2} 0 0 0
"""


#: The committed inputs, for the one cell that is read rather than written.
CASES = Path(__file__).resolve().parents[1] / "data" / "qe"


@pytest.fixture
def pseudo_dir():
    return Path(__file__).resolve().parents[1] / "data" / "pseudo"


def _silicon(tmp_path, pseudo_dir, dataset, kgrid) -> Calculator:
    """The two-atom cell with an augmented dataset, unshifted and unreduced."""
    path = tmp_path / f"si-{dataset}.in"
    path.write_text(SILICON.format(
        upf=DATASETS[dataset], ecutwfc=ECUTWFC, ecutrho=ECUTRHO,
        k0=kgrid[0], k1=kgrid[1], k2=kgrid[2],
    ))
    return Calculator.from_file(path, pseudo_dir=pseudo_dir)


def _supercell(tmp_path, pseudo_dir, calculator, dataset, shape, kgrid) -> Calculator:
    """The same crystal as a real ``shape`` supercell, atoms and all.

    ``a^s_i = n_i a_i`` exactly, so the supercell's reciprocal lattice *is* the
    ultracell's and the two densities are compared Fourier component by Fourier
    component -- the two FFT grids are chosen independently and need not agree.
    """
    cell = calculator.system.cell
    vectors = np.asarray(cell.at_alat) * np.asarray(shape)[:, None]
    tau = np.asarray(calculator.system.structure.positions_alat(cell))
    shifts = np.stack(
        np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    positions = np.concatenate(
        [tau + shift @ np.asarray(cell.at_alat) for shift in shifts]
    )
    rows = "\n".join(f" {v[0]:.12f} {v[1]:.12f} {v[2]:.12f}" for v in vectors)
    atoms = "\n".join(f" Si {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}" for p in positions)
    path = tmp_path / f"si-{dataset}-supercell.in"
    path.write_text(f"""&control
 calculation='scf'
/
&system
 ibrav=0, celldm(1)={float(cell.alat):.10f}, nat={len(positions)}, ntyp=1,
 ecutwfc={ECUTWFC:.1f}, ecutrho={ECUTRHO:.1f}, nosym=.true., noinv=.true.
/
&electrons
 conv_thr=1.0d-12
/
CELL_PARAMETERS alat
{rows}
ATOMIC_SPECIES
 Si 28.086 {DATASETS[dataset]}
ATOMIC_POSITIONS alat
{atoms}
K_POINTS automatic
 {kgrid[0]} {kgrid[1]} {kgrid[2]} 0 0 0
""")
    return Calculator.from_file(path, pseudo_dir=pseudo_dir)


def _modulation(shape, axis=0, amplitude=AMPLITUDE):
    """One period of ``cos`` over the ultracell, in unit-cell coordinates."""
    return lambda x: amplitude * np.cos(2 * np.pi * x[..., axis] / shape[axis])


def _fourier(field, grid, miller):
    """A real field read at a given list of Miller indices.

    ``grid`` is the box ``field`` actually lives on and it is **not** the same
    box on both sides, so it is passed rather than inferred -- wrapping one
    field's indices with the other's shape is a permutation that reads like a
    comparison.
    """
    box = np.asarray(grid)
    J = np.asarray(miller) % box
    flat = J[:, 0] * (box[1] * box[2]) + J[:, 1] * box[2] + J[:, 2]
    spectrum = np.fft.fftn(np.asarray(field)) / np.asarray(field).size
    return spectrum.reshape(-1)[flat]


# -- the null ----------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("dataset", list(DATASETS))
def test_an_augmented_ultracell_is_the_tiled_unit_cell(dataset, tmp_path, pseudo_dir):
    """With nothing applied, the augmentation charge tiles like everything else.

    What this checks that the norm-conserving null cannot: the ``Q_d = 0``
    displaced table, the normalisation of ``becsum`` over the ``N`` copies, and
    -- on the PAW dataset -- the whole one-centre path, which carries
    **-67.18 Ry** of a -89.09 Ry total here and would be missing or doubled
    rather than slightly wrong if the per-copy bookkeeping were off.

    Measured: the total agrees with the unit cell's own SCF to 8.5e-12 Ry
    ultrasoft and 2.5e-11 Ry PAW. The individual terms move by about 2e-6 Ry,
    which is the variational cancellation rather than an error -- the frozen
    states are diagonalised to a finite threshold and the energy is stationary
    in them where a term is not.
    """
    shape, kgrid = (2, 1, 1), (1, 2, 2)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    calculator = _silicon(tmp_path, pseudo_dir, dataset, folded)
    scf = calculator.get_scf(conv_thr=1e-12, nbnd=8)
    assert scf.converged
    if dataset == "paw":
        assert "one_center_paw" in scf.energy_terms

    result = run_ultracell(
        calculator.system, calculator.pseudos, scf, shape, kgrid, nbnd=16,
        conv_thr=1e-12, states_conv_thr=1e-12, max_iterations=40,
    )
    assert result.converged

    tiled = np.asarray(result.ultracell.tile(np.asarray(scf.density)))
    assert np.max(np.abs(np.asarray(result.density) - tiled)) < 1.0e-6

    assert abs(result.total_energy - scf.total_energy) < 1.0e-9
    # **The augmented density is exactly real here and is not in general**, which
    # is worth asserting at both ends. At the null only the ``Q_d = 0`` block of
    # ``becsum`` is alive, and the cutoff set at ``Q = 0`` is the unit cell's own
    # sphere, which is closed under negation; switch a modulation on and the
    # other blocks read a sphere *displaced* by ``Q``, which is not. See the
    # ladder test below and ``UltracellResult.augmentation_residual``.
    assert result.augmentation_residual < 1.0e-12
    # the one-centre term is reported, and it is the unit cell's own
    if dataset == "paw":
        assert abs(float(result.energy_terms["one_center_paw"])
                   - float(scf.energy_terms["one_center_paw"])) < 1.0e-4


# -- the check that discriminates --------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("dataset", list(DATASETS))
def test_the_augmented_ultracell_converges_to_the_supercell(
        dataset, tmp_path, pseudo_dir):
    """The number for the whole phase, and the only check the sign cannot pass.

    A two-cell ultracell and a real four-atom supercell under the same applied
    potential. The induced density's error falls with ``nbnd`` and the total
    energy sits **above** the supercell's at every rung, both measured:

    ====== ============== ============== ============== ==============
    nbnd   error (US)     E - E_s (US)   error (PAW)    E - E_s (PAW)
    ====== ============== ============== ============== ==============
    12     1.18e-1        +1.07e-4 Ry    1.17e-1        +1.07e-4 Ry
    24     1.56e-2        +4.46e-6 Ry    1.71e-2        +5.27e-6 Ry
    48     3.72e-3        +4.82e-7 Ry    5.09e-3        +8.19e-7 Ry
    ====== ============== ============== ============== ==============

    The energy's *sign* is asserted as well as its size, and that is the half
    the density cannot give: the bases are nested in ``nbnd``, so the Kohn-Sham
    free energy minimised over them is a monotone non-increasing upper bound on
    the supercell's, and the wrong displaced table breaks the bound before it
    breaks anything a tolerance would catch.
    """
    shape, kgrid = (2, 1, 1), (1, 2, 2)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    calculator = _silicon(tmp_path, pseudo_dir, dataset, folded)
    scf = calculator.get_scf(conv_thr=1e-12, nbnd=8)
    assert scf.converged

    supercell = _supercell(tmp_path, pseudo_dir, calculator, dataset, shape, kgrid)
    basis = build_basis(supercell.system)
    grid = basis.dense.grid
    coordinates = np.stack(
        np.meshgrid(*[np.arange(m) / m for m in grid], indexing="ij"), axis=-1
    )
    calculation = with_external_potential(
        Calculation(supercell.system, supercell.pseudos),
        jnp.asarray(AMPLITUDE * np.cos(2 * np.pi * coordinates[..., 0])),
    )
    reference = run_scf(
        supercell.system, supercell.pseudos, calculation=calculation,
        conv_thr=1e-11, nbnd=12,
    )
    assert reference.converged

    miller = np.asarray(basis.dense.miller)
    exact = _fourier(np.asarray(reference.density)[0], grid, miller)
    per_cell = float(reference.total_energy) / int(np.prod(shape))

    errors, gaps = {}, {}
    for nbnd in (12, 24, 48):
        result = run_ultracell(
            calculator.system, calculator.pseudos, scf, shape, kgrid,
            nbnd=nbnd, external=_modulation(shape), conv_thr=1e-10,
            states_conv_thr=1e-10,
        )
        assert result.converged
        box = result.ultracell.grid
        ours = _fourier(np.asarray(result.density)[0], box, miller)
        flat = _fourier(
            np.asarray(result.ultracell.tile(jnp.asarray(scf.density)))[0],
            box, miller,
        )
        induced, induced_exact = ours - flat, exact - flat
        errors[nbnd] = float(
            np.abs(induced - induced_exact).max() / np.abs(induced_exact).max()
        )
        gaps[nbnd] = float(result.total_energy) - per_cell
        # Under a modulation the displaced blocks are alive and the cutoff set
        # they read is not closed under negation, so this is no longer zero --
        # 5.5e-5 here at ``ecutrho = 4 ecutwfc``, falling as ``ecutrho^-2.2``.
        # What it must not be is the 1e-2 an index order in ``(Q, Q')`` gives.
        assert 0.0 < result.augmentation_residual < 1.0e-3

    assert errors[48] < errors[24] < errors[12]
    assert errors[48] < 0.01
    # **Above at every rung and falling**, which is what the flipped sign broke.
    assert all(gap > 0.0 for gap in gaps.values()), gaps
    assert gaps[48] < gaps[24] < gaps[12]
    assert gaps[48] < 2.0e-6


# -- a spinor, where D_ij is the integrals sandwiched between fcoef ----------


SPINOR = """&control
 calculation='scf'
/
&system
 ibrav=2, celldm(1)=10.20, nat=2, ntyp=1,
 ecutwfc={ecutwfc:.1f}, ecutrho={ecutrho:.1f},
 nosym=.true., noinv=.true.,
 noncolin=.true., starting_magnetization(1)=0.2,
 angle1(1)={angle1:.1f}, angle2(1)=0.0,
 occupations='smearing', smearing='gaussian', degauss=0.02
/
&electrons
 conv_thr=1.0d-11
/
ATOMIC_SPECIES
 Si 28.086 {upf}
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 {k0} 2 2 0 0 0
"""


def _spinor(tmp_path, pseudo_dir, dataset, k0=2, angle1=0.0) -> Calculator:
    """The same cell as a spinor, with the moment along ``z`` or along ``x``."""
    path = tmp_path / f"si-{dataset}-spinor-{int(angle1)}-{k0}.in"
    path.write_text(SPINOR.format(
        upf=DATASETS[dataset], ecutwfc=ECUTWFC, ecutrho=ECUTRHO,
        angle1=angle1, k0=k0,
    ))
    return Calculator.from_file(path, pseudo_dir=pseudo_dir)


def _spinor_with_field(tmp_path, pseudo_dir, dataset, field) -> Calculator:
    """The same spinor cell under a uniform ``B_z``, for the reference SCF."""
    text = SPINOR.format(
        upf=DATASETS[dataset], ecutwfc=ECUTWFC, ecutrho=ECUTRHO,
        angle1=0.0, k0=2,
    ).replace(" degauss=0.02\n", f" degauss=0.02\n B_field(3) = {field}\n")
    path = tmp_path / f"si-{dataset}-spinor-field.in"
    path.write_text(text)
    return Calculator.from_file(path, pseudo_dir=pseudo_dir)


@pytest.mark.slow
@pytest.mark.parametrize("dataset", list(DATASETS))
def test_a_spinor_augmented_ultracell_is_the_tiled_unit_cell(
        dataset, tmp_path, pseudo_dir):
    """``npol = 2`` and an augmented dataset, with nothing applied.

    Measured: the total reproduces the unit cell's own SCF to 2.8e-12 Ry
    ultrasoft and 8.5e-13 PAW, and the density to 2e-7 pointwise.

    **What this cannot see is the same thing the collinear null could not**:
    with nothing applied every ``becsum`` at a non-zero Q-difference is zero, so
    the displaced tables are multiplied by nothing and the spin blocks are only
    exercised at ``Q_d = 0``. What it does establish is that the sandwich and
    the four-component ``becsum`` reproduce the unit cell exactly there, which
    the tests below build on rather than repeat.
    """
    calculator = _spinor(tmp_path, pseudo_dir, dataset)
    scf = calculator.get_scf(conv_thr=1e-11, nbnd=24)
    assert scf.converged

    result = run_ultracell(
        calculator.system, calculator.pseudos, scf, (2, 1, 1), (1, 2, 2),
        nbnd=32, conv_thr=1e-11, states_conv_thr=1e-11, max_iterations=60,
    )
    assert result.converged
    tiled = np.asarray(result.ultracell.tile(np.asarray(scf.density)))
    assert np.max(np.abs(np.asarray(result.density) - tiled)) < 1.0e-6
    assert abs(result.total_energy - scf.total_energy) < 1.0e-9


@pytest.mark.slow
def test_the_two_spin_regimes_agree_at_a_matched_band_count(tmp_path, pseudo_dir):
    """The spinor route against the collinear one, which shares none of it.

    A collinear ultracell solves one matrix per channel and builds each
    channel's ``becsum`` from that channel's states; a spinor solves one matrix
    on a space twice as large and builds four Pauli components through the
    ``fcoef`` sandwich. With the moment along ``z`` and a uniform field along
    ``z`` the two describe the same physics, so they have to give the same
    answer -- and the only thing between them is the band count, because **a
    spinor band holds one electron where a collinear band holds two**.

    Measured under ``B_z = 0.01`` Ry against ``m = 0.53861`` from each regime's
    own ``run_scf``: collinear ``nbnd = 12, 24, 40`` gives 1.91e-2, 7.38e-3 and
    2.25e-3 relative, and spinor ``nbnd = 24, 48, 80`` gives the same three to
    every digit printed (``m_z`` 0.52832511 against 0.52831902 at the first
    rung). Comparing the two at the *same* ``nbnd`` instead reads as a factor
    of two of missing convergence and is the trap this test exists to name.

    One rung is run here; the ladder is in ``PLAN.md`` P88 stage 6.
    """
    field, nbnd = 0.01, 12
    results = {}
    for regime in ("collinear", "spinor"):
        factor = 1 if regime == "collinear" else 2
        build = _magnetic if regime == "collinear" else _spinor
        plain = build(tmp_path, pseudo_dir, "ultrasoft")
        seed = plain.get_scf(conv_thr=1e-11, nbnd=12 * factor)
        reference = (
            _magnetic(tmp_path, pseudo_dir, "ultrasoft", field=field)
            if regime == "collinear" else
            _spinor_with_field(tmp_path, pseudo_dir, "ultrasoft", field)
        ).get_scf(conv_thr=1e-11, nbnd=12 * factor)
        moment = (float(reference.magnetization) if regime == "collinear"
                  else float(np.asarray(reference.magnetization_vector)[2]))
        result = run_ultracell(
            plain.system, plain.pseudos, seed, (1, 1, 1), (2, 2, 2),
            nbnd=nbnd * factor, conv_thr=1e-11, states_conv_thr=1e-8,
            max_iterations=120,
            magnetic_field=(
                (lambda x: np.full(x.shape[:-1], field)) if factor == 1 else
                (lambda x: np.stack([np.zeros(x.shape[:-1]),
                                     np.zeros(x.shape[:-1]),
                                     np.full(x.shape[:-1], field)], axis=-1))),
        )
        assert result.converged
        got = (float(result.cell_moments().sum()) if factor == 1 else
               float(np.asarray(result.cell_moments()).reshape(-1, 3).sum(0)[2]))
        results[regime] = (got, abs(got - moment) / abs(moment))

    (collinear, error), (spinor, spinor_error) = (
        results["collinear"], results["spinor"])
    assert error < 2.5e-2 and spinor_error < 2.5e-2
    # The two regimes, not the two band counts: this is the assertion. The
    # measured agreement is 1.2e-5 relative, so the bound has room to catch a
    # regression rather than room to hide one.
    assert abs(spinor - collinear) < 5.0e-5 * abs(collinear)


@pytest.mark.slow
def test_a_turning_field_puts_the_moment_where_the_field_points(
        tmp_path, pseudo_dir):
    """A field that turns from one cell to the next, on an augmented dataset.

    This is the run where the spin blocks carry content at a non-zero
    Q-difference: the null cannot, and a field along one fixed axis leaves the
    two off-diagonal blocks empty. The witness that it is not another null is
    ``augmentation_residual``, which reads 3.4e-5 here against 1e-16 with
    nothing applied -- the displaced tables have something to multiply.

    The assertion is the **sense**. Silicon is paramagnetic, so the induced
    moment follows the applied field cell by cell; the opposite sign in the
    off-diagonal block gives the mirror texture, which is degenerate in energy
    with this one (there is no spin-orbit coupling in this dataset), converges
    just as well, and is caught by nothing else here. Measured: +7.95e-2 in the
    cell where the field is positive and -7.95e-2 in the cell where it is not,
    on both datasets.

    **``N = 2`` cannot see the sign of the *displacement*** and is not asked to:
    the only non-zero ``Q`` is the zone boundary, where ``-Q`` and ``Q`` are the
    same point of the reciprocal lattice, so the difference table is symmetric
    and flipping it changes nothing at all. That sign is stage 5's and was
    measured at ``N = 4``. What this test covers is the sign the spin blocks
    carry, which is visible at any ``N``.
    """
    shape, kgrid, amplitude = (2, 1, 1), (2, 2, 2), 0.01
    calculator = _spinor(tmp_path, pseudo_dir, "paw", k0=4, angle1=90.0)
    scf = calculator.get_scf(conv_thr=1e-11, nbnd=24)
    assert scf.converged

    def turning(x):
        """``+y`` in the first cell and ``-y`` in the second."""
        sign = np.cos(2 * np.pi * x[..., 0] / shape[0] - np.pi / 2)
        zero = np.zeros_like(sign)
        return np.stack([zero, amplitude * sign, zero], axis=-1)

    result = run_ultracell(
        calculator.system, calculator.pseudos, scf, shape, kgrid, nbnd=24,
        conv_thr=1e-9, states_conv_thr=1e-9, mixing_beta=0.3,
        magnetic_field=turning, max_iterations=150,
    )
    assert result.converged
    moments = np.asarray(result.cell_moments())
    assert moments[0][1] > 0.0 and moments[1][1] < 0.0
    assert moments[0][1] == pytest.approx(-moments[1][1], rel=1e-3)
    # Not a null: the displaced tables are carrying content at Q_d != 0.
    assert result.augmentation_residual > 1.0e-6


# -- the dual, which is what such a dataset is actually run at ---------------


#: The dataset's own dual, ``8 ecutwfc``, where the two grids come apart. Every
#: other number in this file is at ``ECUTRHO`` and stays there: what the tests
#: below add is the *double grid*, not a second cutoff for everything.
ECUTRHO_DUAL = 8.0 * ECUTWFC


def _silicon_at_a_dual(tmp_path, pseudo_dir, dataset, kgrid) -> Calculator:
    """The same cell as :func:`_silicon`, at ``ecutrho = 8 ecutwfc``."""
    path = tmp_path / f"si-{dataset}-dual.in"
    path.write_text(SILICON.format(
        upf=DATASETS[dataset], ecutwfc=ECUTWFC, ecutrho=ECUTRHO_DUAL,
        k0=kgrid[0], k1=kgrid[1], k2=kgrid[2],
    ))
    return Calculator.from_file(path, pseudo_dir=pseudo_dir)


@pytest.mark.slow
def test_a_double_grid_runs_and_warns_only_below_it(tmp_path, pseudo_dir):
    """The refusal is gone, and the warning that replaced it fires the other way.

    An ultrasoft or PAW dataset asks for ``ecutrho`` of 8 to 12 times
    ``ecutwfc``, which is the pair a first attempt actually writes; what is
    worth saying now is the opposite case, a dataset run at the input default
    of ``4 ecutwfc``, where the augmentation charge sits on the wavefunction
    grid and the absolute energy is not the dataset's.
    """
    from defumat.ultracell.driver import require_an_ultracell_regime

    dual = _silicon_at_a_dual(tmp_path, pseudo_dir, "ultrasoft", (2, 2, 2))
    basis = build_basis(dual.system)
    assert basis.doublegrid
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        require_an_ultracell_regime(dual.system, dual.pseudos, basis)

    flat = _silicon(tmp_path, pseudo_dir, "ultrasoft", (2, 2, 2))
    single = build_basis(flat.system)
    assert not single.doublegrid
    with pytest.warns(UserWarning, match="ecutrho = 4 ecutwfc"):
        require_an_ultracell_regime(flat.system, flat.pseudos, single)


@pytest.mark.slow
@pytest.mark.parametrize("dataset", list(DATASETS))
def test_an_ultracell_at_a_dual_is_the_tiled_unit_cell(
        dataset, tmp_path, pseudo_dir):
    """The null at a double grid: the plumbing on the larger box.

    What it checks is the box, the displaced tables and ``becsum`` at a grid
    pair where the smooth set is a strict subset of the dense one. What it
    **cannot** check is the truncation the lift is about, because ``dV`` is
    identically zero here -- that is the test below, and the two are separate
    for exactly that reason.
    """
    shape, kgrid = (2, 1, 1), (1, 2, 2)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    calculator = _silicon_at_a_dual(tmp_path, pseudo_dir, dataset, folded)
    scf = calculator.get_scf(conv_thr=1e-12, nbnd=8)
    assert scf.converged

    result = run_ultracell(
        calculator.system, calculator.pseudos, scf, shape, kgrid,
        nbnd=16, conv_thr=1e-11,
    )
    assert result.converged
    assert float(result.total_energy) == pytest.approx(
        float(scf.total_energy), abs=1e-9
    )
    # Nothing is applied, so every displaced block is multiplied by zero.
    assert result.augmentation_residual < 1.0e-12


@pytest.mark.slow
@pytest.mark.parametrize("dataset", list(DATASETS))
def test_the_dense_half_of_dv_cannot_reach_the_matrix_element(
        dataset, tmp_path, pseudo_dir):
    """The argument the lifted refusal rests on, measured at ``N = 2``.

    The refusal said that ``h_psi`` multiplies a wavefunction by the potential
    truncated to the smooth sphere, so an ultracell ``dV`` living on the dense
    box would carry a term neither the frozen eigenvalues nor the reference
    supercell has. The premise is right and the conclusion does not follow: the
    gather reads ``dV`` only where two wavefunction spheres differ, at
    ``(G'' - G) + Q_d`` with both ``G`` inside the ``ecutwfc`` sphere, and that
    lies inside the ``4 ecutwfc`` sphere -- a sphere in the ultracell's own
    reciprocal space, which is the supercell's smooth sphere and which the dense
    box holds whole. The tiled *unit-cell* smooth sphere masked below is a
    slightly different set, which is why the two arms agree at 1e-12 rather than
    bit for bit.

    So truncating ``dV`` must change nothing, and the arm that truncates it is
    run here to say so with a number rather than with the argument alone. The
    displaced blocks are alive at ``N = 2`` under the modulation, which is what
    makes this more than the null: the difference carries ``Q_d`` as well.

    **The third arm is the one that says the probe is live.** A patch point
    that silently missed would make the first two agree for the wrong reason,
    which is this project's "a check whose null result cannot be told from a
    pass"; zeroing ``dV`` at the same point moves the total by 6.5e-4 Ry.
    """
    import defumat.ultracell.driver as driver
    from defumat.ultracell.grid import Ultracell

    shape, kgrid = (2, 1, 1), (1, 2, 2)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    calculator = _silicon_at_a_dual(tmp_path, pseudo_dir, dataset, folded)
    basis = build_basis(calculator.system)
    assert basis.doublegrid
    scf = calculator.get_scf(conv_thr=1e-12, nbnd=8)
    assert scf.converged

    ultracell = Ultracell.build(shape, basis.dense.grid)
    mask = np.zeros(basis.dense.grid, dtype=bool)
    mask.reshape(-1)[np.asarray(basis.dense.fft_index)[: basis.smooth.ngm]] = True
    keep_smooth = jnp.asarray(ultracell.reciprocal_mask(mask))

    def run(patch=None):
        # **The patch point is the matrix element and not ``delta_potential``**,
        # because the two halves of ``dV`` are used in different places: the
        # matrix element is the one the argument is about, while ``newd``'s
        # integral against the augmentation charge reads the dense ``dV`` and
        # is *meant* to. Truncating both instead moves the total by 1.2e-10 Ry
        # on this cell, which is a measurement of what the augmentation
        # integral takes from the dense half and not of what this tests.
        original = driver.ultracell_matrix
        if patch is not None:
            driver.ultracell_matrix = (
                lambda c, e, b, v, g, **kw: original(c, e, b, patch(v), g, **kw)
            )
        try:
            result = run_ultracell(
                calculator.system, calculator.pseudos, scf, shape, kgrid,
                nbnd=24, external=_modulation(shape), conv_thr=1e-11,
                states_conv_thr=1e-11,
            )
        finally:
            driver.ultracell_matrix = original
        assert result.converged
        return result

    def truncate(delta_v):
        coefficients = jnp.fft.fftn(delta_v, axes=(-3, -2, -1))
        return jnp.real(jnp.fft.ifftn(
            jnp.where(keep_smooth, coefficients, 0.0), axes=(-3, -2, -1)
        ))

    dense = run()
    smooth = run(truncate)
    zeroed = run(lambda delta_v: 0.0 * delta_v)

    # Not a null: the displaced blocks are carrying content under the
    # modulation, so what agrees below agrees with ``Q_d != 0`` alive.
    assert dense.augmentation_residual > 1.0e-6
    assert float(smooth.total_energy) == pytest.approx(
        float(dense.total_energy), abs=1.0e-10
    )
    assert abs(float(zeroed.total_energy) - float(dense.total_energy)) > 1.0e-5


# -- two spin channels, which the nulls above cannot see ----------------------


MAGNETIC = """&control
 calculation='scf'
/
&system
 ibrav=2, celldm(1)=10.20, nat=2, ntyp=1,
 ecutwfc={ecutwfc:.1f}, ecutrho={ecutrho:.1f},
 nosym=.true., noinv=.true.,
 nspin=2, starting_magnetization(1)=0.2,
 occupations='smearing', smearing='gaussian', degauss=0.02
/
&electrons
 conv_thr=1.0d-11
/
ATOMIC_SPECIES
 Si 28.086 {upf}
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""


def _magnetic(tmp_path, pseudo_dir, dataset, field=None) -> Calculator:
    """The same cell at ``nspin = 2``, optionally under a uniform ``B_field``."""
    text = MAGNETIC.format(upf=DATASETS[dataset], ecutwfc=ECUTWFC, ecutrho=ECUTRHO)
    if field is not None:
        text = text.replace(" degauss=0.02\n",
                            f" degauss=0.02\n B_field(3) = {field}\n")
    name = f"si-{dataset}-mag{'' if field is None else '-field'}.in"
    path = tmp_path / name
    path.write_text(text)
    return Calculator.from_file(path, pseudo_dir=pseudo_dir)


@pytest.mark.slow
@pytest.mark.parametrize("dataset", list(DATASETS))
def test_a_polarized_augmented_ultracell_is_the_tiled_unit_cell(
        dataset, tmp_path, pseudo_dir):
    """``nspin = 2`` and an augmented dataset, with nothing applied.

    The spin axis of ``becsum`` here is the *block* index, because a collinear
    ultracell solves one matrix per channel and each channel's projector
    occupations come from that channel's states alone. Measured: the total
    reproduces the unit cell's own SCF to 7.2e-13 Ry ultrasoft and 5.4e-13 PAW,
    in one iteration.

    **On silicon the two channels come out nearly equal**, so this is a check of
    the plumbing rather than of the spin physics -- what exercises the two
    channels against each other, with opposite signs in ``dV`` and therefore in
    each channel's ``D_ij``, is the applied field below.
    """
    calculator = _magnetic(tmp_path, pseudo_dir, dataset)
    scf = calculator.get_scf(conv_thr=1e-11, nbnd=12)
    assert scf.converged

    result = run_ultracell(
        calculator.system, calculator.pseudos, scf, (2, 1, 1), (1, 2, 2),
        nbnd=16, conv_thr=1e-11, states_conv_thr=1e-11, max_iterations=60,
    )
    assert result.converged
    tiled = np.asarray(result.ultracell.tile(np.asarray(scf.density)))
    assert np.max(np.abs(np.asarray(result.density) - tiled)) < 1.0e-6
    assert abs(result.total_energy - scf.total_energy) < 1.0e-9


@pytest.mark.slow
def test_a_uniform_field_on_an_augmented_ultracell_is_the_unit_cells_own(
        tmp_path, pseudo_dir):
    """The number for the augmented spin path, against a route that shares no code.

    ``N = 1`` under a uniform ``B`` is the same physics as an ordinary SCF with
    ``B_field(3)``, which goes through ``add_bfield.f90``'s expression inside a
    plane-wave SCF where this expands the field-free states of the same cell in
    a basis and never applies ``H`` again. With an augmented dataset the field
    also reaches ``D_ij``, through ``int dV Q`` with opposite signs in the two
    channels, so the comparison tests the per-channel augmented ``D`` and the
    sign of the field at once -- and it must converge in ``nbnd``, because the
    basis truncation is the only approximation between the two.

    Measured at ``B = 0.02`` Ry against ``m = 0.73089286``: 5.0e-3 relative at
    ``nbnd = 12``, 1.8e-3 at 24 and 4.9e-4 at 40, which is the norm-conserving
    ladder's own shape (9.6e-3, 3.2e-3, 7.7e-4). The PAW dataset gives
    5.2e-3, 1.9e-3 and 5.3e-4 against ``m = 0.73114462``; only the ultrasoft one
    is run here, because the two ladders agree to five per cent of each other
    and the second costs what the first does.

    **The k-sets of the two sides have to be the same grid**, which is not a
    detail: running the seed on ``(1, 2, 2)`` while the ultracell folds onto
    ``(2, 2, 2)`` gives a moment 27 per cent off that does *not* improve with
    ``nbnd``, and it reads exactly like a missing term.
    """
    dataset, field = "ultrasoft", 0.02
    calculator = _magnetic(tmp_path, pseudo_dir, dataset)
    scf = calculator.get_scf(conv_thr=1e-11, nbnd=12)
    reference = _magnetic(tmp_path, pseudo_dir, dataset, field=field).get_scf(
        conv_thr=1e-11, nbnd=12)
    assert reference.converged and abs(reference.magnetization) > 0.5

    errors = []
    for nbnd in (12, 24, 40):
        result = run_ultracell(
            calculator.system, calculator.pseudos, scf, (1, 1, 1), (2, 2, 2),
            nbnd=nbnd, magnetic_field=lambda x: np.full(x.shape[:-1], field),
            conv_thr=1e-11, states_conv_thr=1e-8, max_iterations=120,
        )
        assert result.converged
        moment = float(result.cell_moments().sum())
        errors.append(abs(moment - reference.magnetization)
                      / abs(reference.magnetization))

    assert errors == sorted(errors, reverse=True), errors
    assert errors[-1] < 1.5e-3


# -- the budget that falls back to QE's radial table -------------------------


@pytest.mark.slow
@pytest.mark.parametrize("regime", ["collinear", "spinor"])
def test_an_augmented_ultracell_runs_through_the_radial_table(
        regime, tmp_path, pseudo_dir, monkeypatch):
    """The displaced tables are ``N`` of them, so the budget is ``max_bytes / N``.

    An ultracell therefore falls to
    :class:`~defumat.pseudo.augmentation.TabulatedAugmentation` ``N`` times
    sooner than the unit cell does, and on a real slab it is the route that is
    taken rather than an option: the 45-atom NiBr2 cell this was reported from
    needs 1.7 GiB per table against a 683 MiB share of the default budget.
    PAW's one-centre assembly read the block sizes off ``Q_ij(G)``, which that
    table does not have, and died on an ``IndexError`` after a converged SCF.

    Measured here, with the modulation applied so that every Q-difference
    carries weight: through the table against off it, the total is 3.90e-10 Ry
    apart collinear and 3.94e-10 spinor, both in the same number of iterations
    either way. **That 3.9e-10 is the radial interpolation and not the
    ultracell**, which is asserted rather than said: the unit cell's own SCF
    moves by the same amount between the two routes, so the ``N`` displaced
    tables add nothing to it.

    **The induced density is first order in the residual where the total is
    second order, so its bound is set by the threshold.** Collinear it agrees to
    3.5e-10 of its own maximum. Spinor it read 2.8e-9 when this was written, and
    6.0e-7 at the same ``conv_thr = 1e-11`` after the Anderson fit changed
    (``1705a0a``), with the energies unmoved; at 1e-13 it is **6.65e-8**, ten
    times smaller for a hundredfold smaller residual, which is the square-root
    scaling of a stopping point and not of the interpolation. So the spinor
    branch runs at 1e-13 and holds the density at three times that measurement
    (``OPEN.md`` Part XIX, 2026-09-25).
    """
    modulation = _modulation((2, 1, 1))
    energies, iterations, induced, seeds = [], [], [], []
    for budget in ("off", "0"):
        monkeypatch.setenv("DEFUMAT_AUG_MAX_BYTES", budget)
        if regime == "collinear":
            calculator = _silicon(tmp_path, pseudo_dir, "paw", (2, 2, 1))
            scf = calculator.get_scf(conv_thr=1e-12)
            kgrid = (1, 2, 1)
            kwargs = dict(nbnd=12, conv_thr=1e-10, states_conv_thr=1e-10)
        else:
            calculator = _spinor(tmp_path, pseudo_dir, "paw")
            scf = calculator.get_scf(conv_thr=1e-13, nbnd=24)
            kgrid = (1, 2, 2)
            kwargs = dict(nbnd=32, conv_thr=1e-13, states_conv_thr=1e-13,
                          max_iterations=60)
        assert scf.converged
        seeds.append(float(scf.total_energy))
        result = run_ultracell(
            calculator.system, calculator.pseudos, scf, (2, 1, 1), kgrid,
            external=modulation, **kwargs,
        )
        assert result.converged
        energies.append(float(result.total_energy))
        iterations.append(len(result.history))
        tiled = np.asarray(result.ultracell.tile(np.asarray(scf.density)))
        induced.append(np.asarray(result.density) - tiled)

    assert iterations[0] == iterations[1]
    assert abs(energies[0] - energies[1]) < 1.0e-8
    scale = np.abs(induced[0]).max()
    assert scale > 1.0e-4  # the modulation is what makes the check discriminate
    density_bound = 1.0e-8 if regime == "collinear" else 2.0e-7
    assert np.abs(induced[0] - induced[1]).max() < density_bound * scale
    # The whole difference is the frozen states', carried in from the unit
    # cell's own SCF: the displaced tables add nothing to it.
    assert abs((energies[0] - energies[1]) - (seeds[0] - seeds[1])) < 1.0e-11


# -- the transmission, which was refused for the wrong reason -----------------

#: Where the two planes go, and how wide the map is. One k-division along the
#: stacking axis is the quantity's own requirement rather than an economy.
SHEET_PLANES = dict(exit_height=0.15, exit_axis=2, height=0.85, axis=2,
                    broadening=0.05)


def _sheet(tmp_path, pseudo_dir, dataset) -> Calculator:
    """The committed silicon sheet, whose dataset is swapped for the other one.

    A tunnelling geometry needs vacuum and bulk silicon has none: with an
    augmentation sphere of 2.6 bohr against a 5.9 bohr interplanar spacing and
    an atom every quarter of it, no height in that cell clears the spheres and
    the guard refuses every one of them. The cell is committed rather than
    written from a template here because ``docs/features.tex`` runs a snippet
    on it, and two copies of one cell drift.
    """
    text = (CASES / "ultracell-sheet-paw.in").read_text()
    if dataset != "paw":
        text = text.replace(DATASETS["paw"], DATASETS[dataset])
    path = tmp_path / f"sheet-{dataset}.in"
    path.write_text(text)
    return Calculator.from_file(path, pseudo_dir=pseudo_dir)


def _sheet_supercell(tmp_path, pseudo_dir, calculator, dataset, shape,
                     kgrid) -> Calculator:
    """The sheet as a real ``shape`` supercell, atoms and all.

    :func:`_supercell`'s twin for this cell: ``a^s_i = n_i a_i`` exactly, so a
    plane spanning one supercell is the plane spanning ``n_i`` unit cells point
    for point and the two maps are compared without interpolating either.
    """
    cell = calculator.system.cell
    vectors = np.asarray(cell.at_alat) * np.asarray(shape)[:, None]
    tau = np.asarray(calculator.system.structure.positions_alat(cell))
    shifts = np.stack(
        np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    positions = np.concatenate(
        [tau + shift @ np.asarray(cell.at_alat) for shift in shifts]
    )
    rows = "\n".join(f" {v[0]:.12f} {v[1]:.12f} {v[2]:.12f}" for v in vectors)
    atoms = "\n".join(f" Si {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}" for p in positions)
    path = tmp_path / f"sheet-{dataset}-supercell.in"
    path.write_text(f"""&control
 calculation='scf'
/
&system
 ibrav=0, celldm(1)={float(cell.alat):.10f}, nat={len(positions)}, ntyp=1,
 ecutwfc={float(calculator.system.ecutwfc):.1f},
 ecutrho={float(calculator.system.ecutrho):.1f},
 nosym=.true., noinv=.true.,
 occupations='smearing', smearing='gaussian', degauss=0.05
/
&electrons
 conv_thr=1.0d-11
/
CELL_PARAMETERS alat
{rows}
ATOMIC_SPECIES
 Si 28.086 {DATASETS[dataset]}
ATOMIC_POSITIONS alat
{atoms}
K_POINTS automatic
 {kgrid[0]} {kgrid[1]} {kgrid[2]} 0 0 0
""")
    return Calculator.from_file(path, pseudo_dir=pseudo_dir)


@pytest.mark.slow
@pytest.mark.parametrize("dataset", list(DATASETS))
def test_the_augmented_exit_plane_transmission_is_the_unit_cell_s_tiled(
        dataset, tmp_path, pseudo_dir):
    """The plane needs no ``S``, and this is the null that says the wiring runs.

    With nothing applied the ultracell states are the folded unit-cell states,
    and two of them at different ``Q`` carry different lateral momentum, so the
    exit integral cannot mix them: the map comes back as the unit cell's,
    tiled. Measured 6.2e-7 ultrasoft and 9.7e-7 PAW relative. The floor is 1e-5
    and it is the resolvent rather than either threshold -- the two sides
    diagonalise the same Hamiltonian in two different bases, so their levels
    disagree slightly and ``1/(E - e + i eta)`` carries that into the map, which
    is measured as a function of ``eta`` on the norm-conserving cell in
    ``test_ultracell_stm.py`` rather than again here.

    **What it cannot see is whether ``S`` was needed**, because a folded state
    is one basis function and its normalisation is the unit cell's. The rung
    that can is the supercell comparison below, where the state mixes ``Q`` and
    ``sum_Q |c_Q|^2 = 1`` becomes the claim that the frozen basis is
    ``S``-orthonormal across ``Q``.

    The two refusals are asserted here rather than in a section of their own,
    because both need a converged ultracell and this test has one in hand.
    """
    shape, kgrid = (2, 1, 1), (2, 4, 1)
    calculator = _sheet(tmp_path, pseudo_dir, dataset)
    scf = calculator.get_scf()
    assert scf.converged
    geometry = dict(energies=float(scf.fermi_energy), **SHEET_PLANES)

    plain = run_vertical_transport(calculator.system, calculator.pseudos, scf,
                                   shape=(6, 4), **geometry)
    result = run_ultracell(calculator.system, calculator.pseudos, scf, shape,
                           kgrid, nbnd=12, conv_thr=1e-10,
                           states_conv_thr=1e-12)
    assert result.converged
    ours = run_ultracell_transport(calculator.system, calculator.pseudos,
                                   result, shape=(12, 4), **geometry)

    tiled = np.concatenate([np.asarray(plain.values)] * shape[0], axis=0)
    assert tiled.max() > 0.0
    assert np.abs(np.asarray(ours.values) - tiled).max() / tiled.max() < 1.0e-5
    # What the unit cell's own augmented test asserts of its Gram matrices, on
    # the ultracell's: Hermitian and positive semi-definite, neither imposed.
    # The hermiticity comes back **exactly** zero here where the unit cell's
    # reads 3.5e-17, which is `b.conj() @ b.T` being Hermitian to the last bit
    # at these shapes rather than a diagnostic that never ran -- and what rules
    # the second reading out is the comparison above, which no overlap at all
    # would fail first.
    assert ours.notes["hermiticity"] < 1.0e-13
    assert ours.least_eigenvalue > -1.0e-12

    # **The whole-cell diagnostic is the one thing still refused**, because it
    # is the one Gram matrix that is <psi|S|psi>. Widening the exit region to
    # the cell puts both planes inside the spheres by construction, so this is
    # the refusal by name rather than the guard below.
    with pytest.raises(NotImplementedError, match="whole-cell Gram matrix"):
        run_ultracell_transport(calculator.system, calculator.pseudos, result,
                                shape=(4, 4), exit_region="volume", **geometry)

    # And the guard that replaces the old dataset refusal fires: a plane at the
    # atom is inside the augmentation sphere, where the smooth state is not the
    # true one. Tested rather than trusted -- it is a refusal that has to be
    # *reached* through two new call sites.
    for moved in (dict(exit_height=0.5), dict(height=0.5)):
        with pytest.raises(NotImplementedError, match="augmentation sphere"):
            run_ultracell_transport(
                calculator.system, calculator.pseudos, result, shape=(4, 4),
                **{**geometry, **moved})


@pytest.mark.slow
def test_the_augmented_transmission_converges_to_the_supercell(
        tmp_path, pseudo_dir):
    """The rung that discriminates: a state that genuinely mixes ``Q``.

    The image's ladder one quantity along, on a dataset that carries charge
    inside the spheres. What is open here and nowhere else is the
    ``S``-orthonormality of the frozen basis: an ultracell state is
    ``sum_Q c_Q psi_Q`` normalised as ``sum_Q |c_Q|^2 = 1``, which is the true
    normalisation only because the augmentation part of
    ``<psi_Q|S|psi_Q'>`` carries ``sum_R e^{i(Q'-Q).R}`` and vanishes off the
    diagonal. A transmission is a quadratic form in that state, so an overlap
    short by the augmentation charge would show here as a map that is off by a
    few per cent and does not converge to the supercell's.

    Measured against a real two-cell supercell's own transmission, PAW:
    1.09e-2 at ``nbnd = 12`` falling to 1.96e-3 at 24. The ultrasoft dataset is
    not run here, because the one-centre terms PAW adds are a superset of it and
    the rung above covers both.
    """
    shape, kgrid = (2, 1, 1), (2, 4, 1)
    calculator = _sheet(tmp_path, pseudo_dir, "paw")
    scf = calculator.get_scf()
    assert scf.converged
    geometry = dict(energies=float(scf.fermi_energy), shape=(12, 4),
                    **SHEET_PLANES)

    supercell = _sheet_supercell(tmp_path, pseudo_dir, calculator, "paw",
                                 shape, kgrid)
    grid = build_basis(supercell.system).dense.grid
    coordinates = np.stack(
        np.meshgrid(*[np.arange(m) / m for m in grid], indexing="ij"), axis=-1)
    calculation = with_external_potential(
        Calculation(supercell.system, supercell.pseudos),
        jnp.asarray(AMPLITUDE * np.cos(2 * np.pi * coordinates[..., 0])))
    exact = run_scf(supercell.system, supercell.pseudos,
                    calculation=calculation, conv_thr=1e-11, nbnd=16)
    assert exact.converged
    reference = np.asarray(run_vertical_transport(
        supercell.system, supercell.pseudos, exact, **geometry).values)
    assert reference.max() > 0.0

    errors = {}
    for nbnd in (12, 24):
        result = run_ultracell(
            calculator.system, calculator.pseudos, scf, shape, kgrid,
            nbnd=nbnd, external=_modulation(shape), conv_thr=1e-10,
            states_conv_thr=1e-11)
        assert result.converged
        ours = np.asarray(run_ultracell_transport(
            calculator.system, calculator.pseudos, result, **geometry).values)
        errors[nbnd] = float(np.abs(ours - reference).max() / reference.max())

    assert errors[24] < errors[12]
    assert errors[24] < 0.05
