"""P88: the ultracell against the supercell it is an approximation to.

The method claims one thing and this file checks that one thing: expanding the
ultracell's states in the unit cell's own Kohn-Sham states at the ``N`` folded
k-points is a **variational truncation of the exact supercell problem to
``nbnd`` bands per folded k-point, and nothing else**. So the reference is not
Elk and not a benchmark file -- it is an ``N``-cell supercell run through this
package's own SCF, under the same applied potential, which shares the unit-cell
machinery and none of the ultracell assembly.

Three nulls come first, and the third is the one that matters:

* ``N = 1`` must reproduce the unit-cell SCF. This is the weakest of the three
  and it cannot see a factor of ``N`` anywhere.
* ``N > 1`` with nothing applied must give the **tiled** unit-cell density, and
  ``N`` times the electrons. This is what catches a normalisation, and it did:
  the occupation count is the ultracell's, not the unit cell's.
* and the null must be **capable of failing**. A test that only ever asserts
  zero cannot tell a working calculation from a dead code path, so the same
  machinery is handed an applied potential and required to move -- by the right
  amount, at the right wavevector.
"""

from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from defumat import Calculator
from defumat.basis.builder import build_basis
from defumat.scf.driver import Calculation, run_scf
from defumat.ultracell import run_ultracell, with_external_potential
from defumat.units import RY_TO_EV

pytestmark = pytest.mark.regression


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """Bound the peak, because this file sweeps cells that share no shape.

    The two-atom cell on four different folded k-grids, two supercell shapes, a
    magnetic cell and a symmetric one each compile the whole SCF stack afresh,
    and **XLA keeps every executable for the life of the process** -- so the
    peak here grows monotonically through the file rather than being any one
    test's. ``CLAUDE.md`` asks for this on any file over about three distinct
    cells. Only the compiled code is dropped; the results are untouched, and
    the trade is recompilation for a peak the machine can afford.
    """
    yield
    jax.clear_caches()

#: The applied modulation, in Ry. Small enough that the frozen basis is a good
#: one and large enough that the induced density is far above the two SCFs'
#: own convergence floors.
AMPLITUDE = 0.05

SILICON = """&control
 calculation='scf'
/
&system
 ibrav=2, celldm(1)=10.20, nat=2, ntyp=1, ecutwfc={ecut},
 nosym=.true., noinv=.true.
/
&electrons
 conv_thr=1.0d-12
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 {k0} {k1} {k2} 0 0 0
"""


def _silicon(tmp_path, pseudo_dir, kgrid, ecut=12.0) -> Calculator:
    """The two-atom cell on an unshifted, unreduced ``kgrid``.

    Unreduced because a modulation breaks the crystal's point group and the
    ultracell's own group is not written, so the whole phase runs ``nosym``;
    unshifted because the folded set ``k0 + Q`` is a Monkhorst-Pack grid only
    if its origin is.
    """
    path = tmp_path / f"si-{ecut:g}.in"
    path.write_text(SILICON.format(k0=kgrid[0], k1=kgrid[1], k2=kgrid[2],
                                   ecut=f"{ecut:.1f}"))
    return Calculator.from_file(path, pseudo_dir=pseudo_dir)


def _supercell(tmp_path, pseudo_dir, calculator, shape, kgrid,
               ecut=12.0) -> Calculator:
    """The same crystal as a real ``shape`` supercell, atoms and all.

    The lattice vectors are ``a^s_i = n_i a_i`` exactly, so the supercell's
    reciprocal lattice *is* the ultracell's and the two densities can be
    compared Fourier component by Fourier component -- which is what the
    comparison below does, because the two codes' FFT grids are chosen
    independently and need not agree (30 against 32 here).
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
    atoms = "\n".join(f" Si {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}" for p in positions)
    rows = "\n".join(
        f" {v[0]:.12f} {v[1]:.12f} {v[2]:.12f}" for v in vectors
    )
    path = tmp_path / f"si_supercell-{ecut:g}.in"
    path.write_text(f"""&control
 calculation='scf'
/
&system
 ibrav=0, celldm(1)={float(cell.alat):.10f}, nat={len(positions)}, ntyp=1,
 ecutwfc={ecut:.1f}, nosym=.true., noinv=.true.
/
&electrons
 conv_thr=1.0d-12
/
CELL_PARAMETERS alat
{rows}
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
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

    The supercell and the ultracell share a reciprocal lattice, so the same
    ``miller`` list names the same plane waves in both and the two spectra can
    be compared component by component. Comparing here rather than in real
    space avoids interpolating between two independently chosen FFT grids,
    which would put an interpolation error into a number that is supposed to be
    measuring a basis truncation.

    **``grid`` is the box ``field`` actually lives on, and it is not the same
    box on both sides.** The supercell picks its own FFT dimensions from its own
    cutoff -- ``(32, 15, 15)`` where the two-cell ultracell box is
    ``(30, 15, 15)`` -- so wrapping one field's indices with the other's shape
    reads the right plane waves off the wrong strides. It looks like a
    comparison and it is a permutation: every error it produced came out at
    0.9998, for all three band counts at once.
    """
    box = np.asarray(grid)
    J = np.asarray(miller) % box
    flat = J[:, 0] * (box[1] * box[2]) + J[:, 1] * box[2] + J[:, 2]
    spectrum = np.fft.fftn(np.asarray(field)) / np.asarray(field).size
    return spectrum.reshape(-1)[flat]


# -- the three nulls ---------------------------------------------------------


@pytest.mark.slow
def test_the_unit_cell_is_its_own_ultracell(tmp_path, pseudo_dir):
    """``N = 1``: the same eigenvalues, the same density, and ``dV = 0``.

    ``dV`` is the sharp part. It is the difference between the ultracell
    potential and the unit-cell one the frozen eigenvalues already carry, so it
    must vanish to *machine* precision rather than to a convergence threshold:
    nothing about the SCF's accuracy enters a difference of two potentials built
    from the same density.
    """
    calculator = _silicon(tmp_path, pseudo_dir, (2, 2, 2))
    scf = calculator.get_scf(conv_thr=1e-12, nbnd=8)

    result = run_ultracell(
        calculator.system, calculator.pseudos, scf, (1, 1, 1), (2, 2, 2),
        nbnd=8, conv_thr=1e-10, states_conv_thr=1e-10,
    )

    assert result.converged and result.iterations == 1
    assert np.abs(np.asarray(result.delta_v)).max() < 1e-14

    # The eigenvalues and the density agree to the *eigensolver's* floor and no
    # further, and that is the honest bound rather than a loose one: ``dV`` above
    # is machine zero, so the ultracell matrix is exactly the unit cell's -- what
    # is left is two independent Davidson solves of the same Hamiltonian
    # disagreeing at 1.4e-7 Ry, which is 2e-6 eV and is not a property of this
    # method at all.
    ours = np.sort(result.eigenvalues.reshape(-1))
    theirs = np.sort(np.asarray(scf.eigenvalues).reshape(-1))
    assert np.abs(ours - theirs).max() < 1e-6
    density, reference = np.asarray(result.density), np.asarray(scf.density)
    assert np.abs(density - reference).max() / reference.max() < 1e-5

    # **The total energy is the sharp part of this null, not the loose one**,
    # and it is sharper than the eigenvalues above: the input potential cancels
    # identically out of ``eband + deband``, so what is left is the same
    # Kohn-Sham functional evaluated at two states that differ by the two
    # Davidson solves -- and the functional is stationary there, so the
    # difference is second order. Machine precision, not a threshold.
    #
    # It is also the assertion that pins the ``deband`` pairing: computing it
    # against the *output* potential instead leaves this at 2.9e-7 Ry, a factor
    # of 10^7, while every other number in this test is unchanged.
    assert result.total_energy == pytest.approx(scf.total_energy, abs=1e-11)
    assert result.field_energy is None
    assert set(result.energy_terms) == {"one-electron", "hartree", "xc", "ewald"}
    assert sum(result.energy_terms.values()) == pytest.approx(
        result.total_energy, abs=1e-12
    )
    assert len(result.energy_history) == result.iterations


@pytest.mark.slow
@pytest.mark.parametrize("shape", [(2, 1, 1), (2, 2, 1)])
def test_an_unmodulated_ultracell_is_the_tiled_unit_cell(shape, tmp_path, pseudo_dir):
    """``N > 1`` and nothing applied: the tiled density, and ``N`` times the charge.

    **This is the test that catches a factor of ``N``,** and it caught one: the
    occupation search has to be done for the ultracell's electron count and
    rescaled, because ``fixed_occupations`` reads its argument as a number of
    bands to fill while ``smeared_occupations`` reads it as a weighted sum. With
    the unit cell's count the density came out ``N`` times too small, converged
    in one iteration, and passed ``N = 1`` perfectly.
    """
    kgrid = tuple(max(1, 2 // n) for n in shape)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    calculator = _silicon(tmp_path, pseudo_dir, folded)
    scf = calculator.get_scf(conv_thr=1e-12, nbnd=12)

    result = run_ultracell(
        calculator.system, calculator.pseudos, scf, shape, kgrid,
        nbnd=12, conv_thr=1e-10, states_conv_thr=1e-10,
    )
    assert result.converged and result.iterations == 1

    tiled = np.asarray(result.ultracell.tile(jnp.asarray(scf.density)))
    density = np.asarray(result.density)
    assert np.abs(density - tiled).max() / tiled.max() < 1e-6
    assert np.abs(np.asarray(result.delta_v)).max() < 1e-14

    cells = result.ultracell.cells
    volume = result.ultracell.volume(calculator.system.cell)
    charge = float(density.sum()) * volume / density[0].size
    assert charge == pytest.approx(8.0 * cells, abs=1e-8)

    # **The energy is per unit cell, so a tiled ultracell has the unit cell's
    # own.** This is where a factor of ``N`` would show and where ``N = 1``
    # cannot see one: ``deband`` is an integral over the whole ultracell and
    # the Hartree and exchange-correlation terms arrive as ultracell totals, so
    # each carries a ``/N`` and each is a separate chance to drop it.
    assert result.total_energy == pytest.approx(scf.total_energy, abs=1e-11)


@pytest.mark.slow
@pytest.mark.parametrize("shape", [(2, 1, 1), (1, 2, 1)])
def test_the_null_can_fail(shape, tmp_path, pseudo_dir):
    """The guard fires: an applied potential moves the state, at its own ``Q``.

    A clean zero from a dead code path reads exactly like a clean zero from a
    working calculation, so the two nulls above are only evidence if the same
    machinery *can* produce a non-zero. It does, and the induced density's
    Fourier weight sits where the perturbation put it: at the applied ``Q`` and
    its harmonics along the modulated axis, and nowhere else.
    """
    kgrid = tuple(max(1, 2 // n) for n in shape)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    calculator = _silicon(tmp_path, pseudo_dir, folded)
    scf = calculator.get_scf(conv_thr=1e-12, nbnd=12)

    axis = int(np.argmax(shape))
    result = run_ultracell(
        calculator.system, calculator.pseudos, scf, shape, kgrid,
        nbnd=12, external=_modulation(shape, axis), conv_thr=1e-10,
        states_conv_thr=1e-10,
    )
    assert result.converged

    induced = np.asarray(result.modulation)[0]
    tiled = np.asarray(result.ultracell.tile(jnp.asarray(scf.density)))[0]
    assert np.abs(induced).max() > 1e-5 * tiled.max()

    # **The induced density's dominant Fourier weight is at the applied
    # wavevector.** That is the threshold-free form of "it moved for the right
    # reason", and it is what a wrong index map would break: putting a plane
    # wave in the wrong cell of the ultracell moves the response to a different
    # Q and changes nothing else.
    spectrum = np.fft.fftn(induced)
    q_along = np.arange(result.ultracell.grid[axis]) % shape[axis]
    rolled = np.moveaxis(spectrum, axis, 0)
    peak = np.unravel_index(np.abs(rolled).argmax(), rolled.shape)
    # The applied modulation is one ultracell period, so its Q index is 1 (or
    # its conjugate, n - 1), and that is where the peak has to be -- not merely
    # somewhere off Q = 0.
    assert q_along[peak[0]] in (1, shape[axis] - 1)

    # The Q = 0 part is not zero and should not be: a cos(Qx) perturbation at
    # this amplitude has a real second-order response at Q = 0 and 2Q, and it
    # is 1.1 per cent of the linear one here. The bound is on its *size*, not
    # on its existence -- and it scales as the square of the amplitude, which
    # is how one tells it from an error in the bookkeeping.
    unmodulated = rolled[q_along == 0]
    modulated = rolled[q_along != 0]
    assert np.abs(unmodulated).max() < 0.1 * np.abs(modulated).max()


# -- the number --------------------------------------------------------------


@pytest.mark.slow
def test_the_ultracell_converges_to_the_supercell(tmp_path, pseudo_dir):
    """The central claim: the error is the band truncation and nothing else.

    A two-cell ultracell and a real four-atom supercell, under the same applied
    potential, compared Fourier component by Fourier component. What is checked
    is not that the two agree at some ``nbnd`` -- they do not, and should not --
    but that **the disagreement falls as ``nbnd`` grows**, which is the whole
    content of "variational truncation". The ladder starts at 12 rather than at
    the occupied count: ``nbnd = 8`` is measured at 43 per cent and is also not
    reliably *convergent* here, which is its own reason to pass ``nbnd``.

    The convergence carries no *sign*: this Hamiltonian moves with its own
    truncated density, so an ultracell eigenvalue is not an upper bound on the
    supercell's, and asserting one would fail for the right reason.
    """
    shape, kgrid = (2, 1, 1), (1, 2, 2)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    calculator = _silicon(tmp_path, pseudo_dir, folded)
    scf = calculator.get_scf(conv_thr=1e-12, nbnd=8)

    supercell = _supercell(tmp_path, pseudo_dir, calculator, shape, kgrid)
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

    errors = {}
    # **The top rung used to carry ``david = 2`` and no longer does.** The
    # Davidson subspace is ``david * nbnd``, and it was not capped against the
    # size of the space (``OPEN.md`` Part VI item 1): at ``nbnd = 48`` the
    # default asked for 192 vectors where these k-points hold between 169 and
    # 192 plane waves, the overlaps went non-finite, and what surfaced three
    # layers up was "the ultracell did not converge". It is capped now, at
    # ``min_k npw``, so the default subspace is what this rung runs at -- and
    # the rung is the regression test for the cap: at ``david = 4`` on this cell
    # 25 of 32 k-points came back non-finite before and none does now.
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

    # It converges, and it converges monotonically over this range.
    assert errors[48] < errors[24] < errors[12]
    # and it gets somewhere: below a per cent of the induced modulation.
    assert errors[48] < 0.01




#: The cutoff at which the supercell's own dense FFT grid **is** the tiled unit
#: cell's: 13 Ry puts the unit cell on (18, 18, 18) and the two-cell supercell
#: on (36, 18, 18), where the default 12 gives 15 and 32 and the two do not
#: match. It has to be reached through ``ecutwfc`` rather than ``ecutrho``,
#: because the ultracell refuses a double grid by name.
#:
#: **Why this matters only for the energy.** Every other ladder in this file
#: compares a *density* Fourier component by Fourier component, where two boxes
#: cost a floor of about 1e-4 relative and the claim is about a trend. The
#: energy claim is about a **sign**, and two discretisations of one functional
#: differ by 1.08e-6 Ry per cell on this cell -- measured, by running the
#: supercell at four dense cutoffs -- which is larger than the top of the
#: ladder below. On the mismatched pair the third rung comes out 1.06e-7 Ry
#: *under* the supercell and the bound reads as violated.
MATCHED_ECUT = 13.0


@pytest.mark.slow
def test_the_total_energy_bounds_the_supercells_from_above(tmp_path, pseudo_dir):
    """The one quantity in this phase that converges with a sign.

    An ultracell eigenvalue is not an upper bound on the supercell's, and
    neither is a density: the Hamiltonian moves with its own truncated density,
    so Rayleigh-Ritz says nothing about either. The **energy** is different.
    What the run reports is the Kohn-Sham free energy of the state the loop
    converged to, the bases are *nested* in ``nbnd`` -- adding bands keeps every
    old one -- and the union over ``Q`` of the folded k-point spheres is the
    supercell's own plane-wave space at ``k0``. So the sequence is a monotone
    non-increasing upper bound on the supercell's own energy.

    Neither ``pw.x`` nor Elk computes an ultracell total energy, so the
    reference here is a real four-atom supercell through this package's own SCF
    -- which shares the unit-cell machinery and none of the ultracell assembly.

    **Two preconditions, and both are physics rather than tolerance.** The two
    sides must discretise the *same* functional, which is what ``MATCHED_ECUT``
    is for. And the frozen states must be Ritz vectors of the unit cell's
    Hamiltonian, which Davidson returns by construction (``evc`` is a rotation
    of the trial set), so ``diag(eps)`` is the exact projected ``H_cell`` at any
    ``states_conv_thr`` and a loose one only makes the span slightly worse.

    **What this cannot see, measured rather than assumed.** Pairing ``deband``
    with the *output* potential instead of the input one -- the defect worth a
    factor of 10^7 at ``N = 1`` -- leaves this test passing unchanged, because
    at ``conv_thr = 1e-11`` the two densities are equal to far better than the
    gaps below. The null at ``N = 1`` is what catches that one, where the loop
    converges in a single iteration and the two densities differ by the two
    Davidson solves. What *this* test catches is a term that is wrong by a
    factor rather than by an increment: dropping the per-cell rescaling of the
    Hartree energy breaks the monotonicity here and is invisible at ``N = 1``.
    The pair is the coverage, and neither half is the other.
    """
    shape, kgrid = (2, 1, 1), (1, 2, 2)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    calculator = _silicon(tmp_path, pseudo_dir, folded, ecut=MATCHED_ECUT)
    scf = calculator.get_scf(conv_thr=1e-12, nbnd=8)

    supercell = _supercell(tmp_path, pseudo_dir, calculator, shape, kgrid,
                           ecut=MATCHED_ECUT)
    basis = build_basis(supercell.system)
    grid = tuple(basis.dense.grid)
    unit_grid = tuple(build_basis(calculator.system).dense.grid)
    # **Asserted rather than assumed**, because the whole test rests on it and
    # a cutoff change elsewhere would break it silently, leaving a bound that
    # fails for a reason nothing in the message would name.
    assert grid == tuple(n * m for n, m in zip(shape, unit_grid)), (
        f"the supercell is on {grid} and the tiled unit cell on "
        f"{tuple(n * m for n, m in zip(shape, unit_grid))}: the two must "
        f"discretise the same functional for the bound to mean anything"
    )

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
    exact = reference.total_energy / int(np.prod(shape))

    energies = {}
    # The top rung ran at ``david = 2`` until the Davidson subspace was capped
    # at the size of the space it lives in; it is the default now, for the
    # reason the density ladder above gives.
    for nbnd in (12, 24, 48):
        result = run_ultracell(
            calculator.system, calculator.pseudos, scf, shape, kgrid,
            nbnd=nbnd, external=_modulation(shape), conv_thr=1e-11,
            states_conv_thr=1e-10,
        )
        assert result.converged
        energies[nbnd] = result.total_energy

    # Monotone, which is the nesting, ...
    assert energies[48] < energies[24] < energies[12]
    # ... and above, which is the variational principle. Measured at
    # +8.14e-05, +3.56e-06 and +4.06e-07 Ry; the bound is asserted with a
    # tolerance of zero because it is a sign rather than a size, and the
    # smallest gap is three orders above both SCFs' own convergence.
    for nbnd, energy in energies.items():
        assert energy > exact, (
            f"nbnd = {nbnd} gives {energy:.12f} Ry against the supercell's "
            f"{exact:.12f}: an upper bound cannot be below what it bounds"
        )
    # and it gets somewhere: within a micro-Rydberg of the supercell it
    # approximates, on a modulation whose own energy is a milli-Rydberg.
    assert energies[48] - exact < 1e-6


# -- the refusals ------------------------------------------------------------


# -- stage 3a: two spin channels ---------------------------------------------

#: A hydrogen simple-cubic lattice that is **partly** polarized, which is what
#: this needs and what neither end of the range gives. At ``a = 5.0`` the same
#: cell has a moment of 0.027 and takes 56 iterations -- it sits on the Stoner
#: threshold -- and at ``a = 6.0`` it is 0.9997, a saturated atom whose moment
#: cannot grow and whose ``|zeta| = 1`` is the clamp-tangent trap ``CLAUDE.md``
#: names. At 5.5 with a 0.8 seed it is 0.62 in six iterations.
#:
#: **That 0.62 is a k-grid as much as it is a lattice constant**, and leaving
#: the grid off this line cost a later session a whole comparison. The moment
#: is 0.6234 on the ``(4, 2, 2)`` grid every test below folds to, and on
#: nothing else: ``(4, 1, 1)`` gives **1.0000**, a saturated atom, ``(4, 4, 4)``
#: 0.7901 and ``(6, 6, 6)`` 0.8020, so the partly-polarized cell this fixture
#: exists to provide is the one this grid picks out. Quote the number with the
#: grid or the cell it names is a different cell.
HYDROGEN = """&control
 calculation='scf'
/
&system
 ibrav=1, celldm(1)=5.5, nat=1, ntyp=1, ecutwfc=15.0,
 nosym=.true., noinv=.true.,
 nspin=2, starting_magnetization(1)=0.8,
 occupations='smearing', smearing='gaussian', degauss=0.02
/
&electrons
 conv_thr=1.0d-11
/
ATOMIC_SPECIES
 H 1.008 H.pz-vbc.UPF
ATOMIC_POSITIONS crystal
 H 0.0 0.0 0.0
K_POINTS automatic
 {k0} {k1} {k2} 0 0 0
"""


def _hydrogen(tmp_path, pseudo_dir, kgrid) -> Calculator:
    path = tmp_path / "h.in"
    path.write_text(HYDROGEN.format(k0=kgrid[0], k1=kgrid[1], k2=kgrid[2]))
    return Calculator.from_file(path, pseudo_dir=pseudo_dir)


def _hydrogen_supercell(tmp_path, pseudo_dir, shape, kgrid) -> Calculator:
    """The same lattice as a real ``shape`` supercell of hydrogens."""
    rows = "\n".join(
        " %.10f %.10f %.10f" % tuple(v) for v in np.diag(shape).astype(float)
    )
    frac = np.stack(
        np.meshgrid(*[np.arange(n) / n for n in shape], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    atoms = "\n".join(" H %.10f %.10f %.10f" % tuple(x) for x in frac)
    path = tmp_path / "h_supercell.in"
    path.write_text(f"""&control
 calculation='scf'
/
&system
 ibrav=0, celldm(1)=5.5, nat={len(frac)}, ntyp=1, ecutwfc=15.0,
 nosym=.true., noinv=.true.,
 nspin=2, starting_magnetization(1)=0.8,
 occupations='smearing', smearing='gaussian', degauss=0.02
/
&electrons
 conv_thr=1.0d-11
/
CELL_PARAMETERS alat
{rows}
ATOMIC_SPECIES
 H 1.008 H.pz-vbc.UPF
ATOMIC_POSITIONS crystal
{atoms}
K_POINTS automatic
 {kgrid[0]} {kgrid[1]} {kgrid[2]} 0 0 0
""")
    return Calculator.from_file(path, pseudo_dir=pseudo_dir)


@pytest.mark.slow
def test_a_polarized_ultracell_is_the_tiled_unit_cell(tmp_path, pseudo_dir):
    """The ``nspin = 2`` null: both channels tile, and the moment tiles with them.

    The unpolarized null (above) cannot see the spin plumbing at all -- the two
    channels are equal there, so an occupation rule that fills the wrong one, a
    matrix built with ``dV[0]`` for both, or a density accumulated into channel
    zero twice would all pass it. This runs the same null on a cell that has a
    moment, where each of those is a different answer.
    """
    shape, kgrid = (2, 1, 1), (2, 2, 2)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    calculator = _hydrogen(tmp_path, pseudo_dir, folded)
    scf = calculator.get_scf(conv_thr=1e-11, nbnd=8)
    assert scf.converged and abs(scf.magnetization) > 0.3

    result = run_ultracell(
        calculator.system, calculator.pseudos, scf, shape, kgrid,
        nbnd=8, conv_thr=1e-9, states_conv_thr=1e-8,
    )
    assert result.converged and result.iterations == 1
    assert np.abs(np.asarray(result.delta_v)).max() < 1e-14

    tiled = np.asarray(result.ultracell.tile(jnp.asarray(scf.density)))
    density = np.asarray(result.density)
    assert np.abs(density - tiled).max() / tiled.max() < 1e-5

    # Every cell carries the unit cell's own moment, and that is the number the
    # unpolarized null has no counterpart for.
    moments = result.cell_moments()
    assert len(moments) == result.ultracell.cells
    assert moments == pytest.approx(
        np.full(result.ultracell.cells, scf.magnetization), rel=2e-4
    )

    # **The total charge and the split between the channels are bounded by two
    # different things, and only the second is the method's own error.** The
    # total is ``N`` times the unit cell's to round-off, because the occupation
    # search is done for the ultracell's electron count and nothing else can
    # move it. How those electrons divide between the channels is the moment,
    # and it comes back 1.3e-6 out of 0.62 -- which is the frozen states'
    # accuracy, not a normalisation: a wrong ``N`` anywhere here would be a
    # factor rather than a sixth digit.
    volume = result.ultracell.volume(calculator.system.cell)
    element = volume / density[0].size
    per_channel = density.sum(axis=(1, 2, 3)) * element
    cells = result.ultracell.cells
    reference = np.asarray(scf.density).sum(axis=(1, 2, 3)) * (
        float(calculator.system.cell.volume) / np.asarray(scf.density)[0].size
    )
    assert float(per_channel.sum()) == pytest.approx(
        float(reference.sum()) * cells, abs=1e-9
    )
    assert per_channel == pytest.approx(reference * cells, abs=5e-6)


@pytest.mark.slow
def test_a_constrained_moment_gives_each_channel_its_own_fermi_level(
        tmp_path, pseudo_dir):
    """``tot_magnetization`` is the ``Q = 0`` moment constraint, and it scales by ``N``.

    With the moment constrained the two channels stop sharing a Fermi level and
    each is filled against its own electron count -- QE's
    ``two_fermi_energies``. The count is per unit cell in the input and the
    occupation search here runs for the **ultracell**, so both halves have to be
    multiplied by ``N`` and then divided out again with the weights. That is
    stage 1's own trap, one level in: getting it backwards is a factor of ``N``
    in one channel that converges perfectly well.

    The null is the sharp form of it. With ``tot_magnetization = 0`` on a cell
    whose unconstrained moment is already zero, the constraint changes nothing
    and the answer must still be the tiled unit-cell density in one iteration --
    but it now arrives through the two-count branch, which the shared-Fermi
    tests never enter.
    """
    shape, kgrid = (2, 1, 1), (1, 2, 2)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    path = tmp_path / "si_fsm.in"
    path.write_text(
        MAGNETIC.replace("2 2 2 0 0 0", "{} {} {} 0 0 0".format(*folded))
        .replace(" degauss=0.02\n", " degauss=0.02, tot_magnetization=0.0\n")
    )
    calculator = Calculator.from_file(path, pseudo_dir=pseudo_dir)
    assert calculator.system.tot_magnetization is not None
    scf = calculator.get_scf(conv_thr=1e-12, nbnd=12)

    result = run_ultracell(
        calculator.system, calculator.pseudos, scf, shape, kgrid, nbnd=12,
        conv_thr=1e-10, states_conv_thr=1e-9,
    )
    assert result.converged and result.iterations == 1
    assert np.abs(np.asarray(result.delta_v)).max() < 1e-14

    # Two Fermi levels rather than one, which is what says the branch ran.
    assert len(np.atleast_1d(np.asarray(result.fermi_energy, dtype=float))) == 2

    tiled = np.asarray(result.ultracell.tile(jnp.asarray(scf.density)))
    density = np.asarray(result.density)
    assert np.abs(density - tiled).max() / tiled.max() < 1e-5

    # The constraint is met on the ultracell as a whole, which is the Q = 0
    # component of the envelope and the only component it constrains.
    assert float(result.cell_moments().sum()) == pytest.approx(0.0, abs=1e-6)

    volume = result.ultracell.volume(calculator.system.cell)
    charge = float(density.sum()) * volume / density[0].size
    assert charge == pytest.approx(8.0 * result.ultracell.cells, abs=1e-8)


@pytest.mark.slow
def test_a_modulated_field_makes_a_spin_density_wave(tmp_path, pseudo_dir):
    """The magnetic null **can fail**: an applied ``B(r)`` modulates the moment.

    Nothing in a collinear SCF breaks spin symmetry on its own, so the two nulls
    above would read exactly the same from a spin path that had been deleted.
    This drives one: a field ``B cos(2 pi x / n)`` on a cell whose own moment is
    zero, and what comes back is a spin density wave -- equal and opposite
    moments in the two cells, with its Fourier weight at the applied wavevector
    and nowhere else.

    The **charge** response is the discriminating half. A collinear system is
    invariant under flipping every spin together with the sign of ``B``, so the
    charge cannot respond at linear order in ``B`` and the magnetization must:
    a run in which both moved by the same order has coupled the channels
    somewhere they are not coupled.
    """
    shape, kgrid = (2, 1, 1), (1, 2, 2)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    path = tmp_path / "si_mag.in"
    path.write_text(MAGNETIC.replace("2 2 2 0 0 0", "{} {} {} 0 0 0".format(*folded)))
    calculator = Calculator.from_file(path, pseudo_dir=pseudo_dir)
    scf = calculator.get_scf(conv_thr=1e-12, nbnd=12)
    assert abs(scf.magnetization) < 1e-5

    amplitude = 0.05
    result = run_ultracell(
        calculator.system, calculator.pseudos, scf, shape, kgrid, nbnd=12,
        magnetic_field=lambda x: amplitude * np.cos(2 * np.pi * x[..., 0] / shape[0]),
        conv_thr=1e-10, states_conv_thr=1e-9,
    )
    assert result.converged

    moments = result.cell_moments()
    assert abs(moments[0]) > 1e-3
    # Equal and opposite: the field has zero mean over the ultracell, so the
    # induced moment does too, to the order the response is linear in.
    assert moments[0] == pytest.approx(-moments[1], rel=1e-3)

    magnetization = np.asarray(result.magnetization)
    spectrum = np.moveaxis(np.fft.fftn(magnetization), 0, 0)
    q_along = np.arange(result.ultracell.grid[0]) % shape[0]
    peak = np.unravel_index(np.abs(spectrum).argmax(), spectrum.shape)
    assert q_along[peak[0]] in (1, shape[0] - 1)
    assert (np.abs(spectrum[q_along == 0]).max()
            < 1e-3 * np.abs(spectrum[q_along != 0]).max())

    # The charge moved by far less than the moment did, which is the symmetry
    # statement above turned into a number.
    charge = np.abs(np.asarray(result.modulation).sum(axis=0)).max()
    assert charge < 0.7 * np.abs(magnetization).max()


@pytest.mark.slow
def test_a_uniform_field_is_the_unit_cell_under_the_same_field(tmp_path, pseudo_dir):
    """The number for the field: ``N = 1`` under a uniform ``B``, against ``pw.x``'s route.

    An ultracell with one cell and a *uniform* applied field is the same physics
    as an ordinary SCF with ``B_field(3)`` -- and the two share nothing: the
    reference goes through ``add_bfield.f90``'s expression inside a plane-wave
    SCF, where this expands the field-free states of the same cell in a basis
    and never applies ``H`` again. So the comparison tests the sign of the
    field, its magnitude, and the whole spin path at once, and it must converge
    in ``nbnd`` for the same reason the charge does: the basis truncation is
    the only approximation between them.

    Measured on this cell at ``B = 0.02`` Ry, against ``m = 0.71159883``:
    ``nbnd = 12`` gives 9.6e-3 relative, 24 gives 3.2e-3, 40 gives 7.7e-4.
    """
    path = tmp_path / "si_mag.in"
    path.write_text(MAGNETIC)
    calculator = Calculator.from_file(path, pseudo_dir=pseudo_dir)
    scf = calculator.get_scf(conv_thr=1e-12, nbnd=12)

    field = 0.02
    with_field = tmp_path / "si_field.in"
    with_field.write_text(MAGNETIC.replace(
        " degauss=0.02\n", f" degauss=0.02\n B_field(3) = {field}\n"))
    reference = Calculator.from_file(
        with_field, pseudo_dir=pseudo_dir).get_scf(conv_thr=1e-12, nbnd=12)
    assert reference.converged and abs(reference.magnetization) > 0.5

    errors, energies = [], []
    for nbnd in (12, 24, 40):
        result = run_ultracell(
            calculator.system, calculator.pseudos, scf, (1, 1, 1), (2, 2, 2),
            nbnd=nbnd, magnetic_field=lambda x: np.full(x.shape[:-1], field),
            conv_thr=1e-11, states_conv_thr=1e-8,
        )
        assert result.converged
        moment = float(result.cell_moments().sum())
        errors.append(abs(moment - reference.magnetization)
                      / abs(reference.magnetization))
        energies.append((result.total_energy, result.field_energy, moment))

    # **Monotone, and the sign is what a flipped field would break.** A moment
    # that came back at -0.71 would fail the first assertion; one at half the
    # size would fail every rung of the ladder and would not improve with nbnd.
    assert errors == sorted(errors, reverse=True), errors
    assert errors[-1] < 2e-3

    # **The field's energy is outside the total on both sides**, which is QE's
    # and Elk's shared convention and this package's.
    #
    # What is asserted first is the *internal* identity, because it is exact
    # where an agreement with the reference is only as good as the moment: a
    # uniform field's Zeeman energy is ``-B`` times the moment the same run
    # reports, whatever the basis truncation has done to that moment. Asserting
    # instead that the two runs' field energies agree is asserting the moment
    # ladder a second time, and at ``nbnd = 12`` it is out by a per cent.
    for total, field_energy, moment in energies:
        assert field_energy == pytest.approx(-field * moment, rel=1e-9)

    # ... and the convention itself, which is the thing this test is for: the
    # totals agree by four orders more than the quantity a wrong convention
    # would have left inside one of them. Read at the converged end of the
    # ladder, where the basis is no longer the limit.
    residual = abs(energies[-1][0] - reference.total_energy)
    assert residual < 1e-3 * abs(reference.field_energy), (
        f"{residual:.3e} Ry against a Zeeman energy of "
        f"{reference.field_energy:.3e}: the total is carrying the field"
    )

    # **And the variational bound lives on the sum, not on the total**, which
    # is the price of that convention and is worth asserting rather than
    # assuming. What the calculation minimises is the full energy, the Zeeman
    # term included; the reported total has that term removed, so it is a bound
    # on nothing -- measured on a hydrogen cell it goes 7.4e-07 Ry *above* an
    # ordinary SCF at ``nbnd = 12`` and 9.7e-07 *below* it at 24. Add
    # ``field_energy`` back and the bound returns, monotone.
    free = [total + field_energy for total, field_energy, _ in energies]
    exact = reference.total_energy + reference.field_energy
    assert free == sorted(free, reverse=True), free
    for nbnd, value in zip((12, 24, 40), free):
        assert value > exact, (
            f"nbnd = {nbnd}: {value:.12f} Ry against {exact:.12f}; the "
            f"minimised quantity is total_energy + field_energy"
        )


@pytest.mark.slow
def test_the_magnetic_ultracell_converges_to_the_supercell(tmp_path, pseudo_dir):
    """The number for the spin plumbing: a polarized ultracell against a supercell.

    The same comparison stage 1 makes, on a cell that has a moment, and with
    the **magnetization** compared beside the charge. The perturbation is a
    scalar potential rather than a field, so the magnetization's response is
    entirely indirect -- the local exchange splitting follows the local charge
    -- which is what makes it a test of the coupled two-channel loop rather
    than of a field's sign.

    Measured at ``AMPLITUDE = 0.05`` Ry on a two-cell hydrogen ultracell:
    charge 5.2e-4 / 2.3e-4 / 1.7e-4 and magnetization 6.2e-4 / 2.4e-4 / 1.4e-4
    at ``nbnd = 8 / 16 / 24``, relative to the largest Fourier component of
    each. The floor near 1.5e-4 is the two boxes, not the method: the supercell
    picks a 27-point FFT grid along the modulated axis where the ultracell's is
    30, and a density is not band-limited.
    """
    shape, kgrid = (2, 1, 1), (2, 2, 2)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    calculator = _hydrogen(tmp_path, pseudo_dir, folded)
    scf = calculator.get_scf(conv_thr=1e-11, nbnd=8)
    assert scf.converged

    supercell = _hydrogen_supercell(tmp_path, pseudo_dir, shape, kgrid)
    calculation = Calculation(supercell.system, supercell.pseudos)
    grid = tuple(calculation.basis.dense.grid)
    axes = [np.arange(m) / m for m in grid]
    coordinates = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    applied = AMPLITUDE * np.cos(2 * np.pi * coordinates[..., 0])
    reference = run_scf(
        supercell.system, supercell.pseudos,
        calculation=with_external_potential(calculation, jnp.asarray(applied)),
        conv_thr=1e-11, nbnd=16,
    )
    assert reference.converged

    miller = np.stack(np.meshgrid(
        np.arange(-4, 5), np.arange(-2, 3), np.arange(-2, 3), indexing="ij"
    ), axis=-1).reshape(-1, 3)
    theirs = np.asarray(reference.density)
    their_charge = _fourier(theirs.sum(axis=0), grid, miller)
    their_moment = _fourier(theirs[0] - theirs[1], grid, miller)

    charge_errors, moment_errors = [], []
    for nbnd in (8, 16, 24):
        result = run_ultracell(
            calculator.system, calculator.pseudos, scf, shape, kgrid, nbnd=nbnd,
            external=_modulation(shape, 0), conv_thr=1e-10,
            states_conv_thr=1e-8,
        )
        assert result.converged
        box = result.ultracell.grid
        ours = np.asarray(result.density)
        charge_errors.append(
            np.abs(_fourier(ours.sum(axis=0), box, miller) - their_charge).max()
            / np.abs(their_charge).max()
        )
        moment_errors.append(
            np.abs(_fourier(ours[0] - ours[1], box, miller) - their_moment).max()
            / np.abs(their_moment).max()
        )

    # **No sign is asserted** -- the Hamiltonian moves with its own truncated
    # density, so an ultracell eigenvalue is not an upper bound on the
    # supercell's. What is asserted is that both halves *improve* with the one
    # knob the method has, which is the whole content of "a variational
    # truncation to nbnd bands per folded k-point and nothing else".
    assert charge_errors == sorted(charge_errors, reverse=True), charge_errors
    assert moment_errors == sorted(moment_errors, reverse=True), moment_errors
    assert charge_errors[-1] < 5e-4 and moment_errors[-1] < 5e-4


#: The same lattice as a two-cell supercell with its two moments **opposite**,
#: which is what a seeded antiferromagnet is: two species pointing at the same
#: UPF, because ``starting_magnetization`` is per species and there is no other
#: way to make two atoms of one element start differently.
ANTIFERROMAGNET = """&control
 calculation='scf'
/
&system
 ibrav=0, celldm(1)=5.5, nat=2, ntyp=2, ecutwfc=15.0,
 nosym=.true., noinv=.true.,
 nspin=2, starting_magnetization(1)=0.8, starting_magnetization(2)={m2},
 occupations='smearing', smearing='gaussian', degauss=0.02
/
&electrons
 conv_thr=1.0d-11
 mixing_beta=0.3
/
CELL_PARAMETERS alat
 2.0 0.0 0.0
 0.0 1.0 0.0
 0.0 0.0 1.0
ATOMIC_SPECIES
 H1 1.008 H.pz-vbc.UPF
 H2 1.008 H.pz-vbc.UPF
ATOMIC_POSITIONS crystal
 H1 0.0 0.0 0.0
 H2 0.5 0.0 0.0
K_POINTS automatic
 {k0} {k1} {k2} 0 0 0
"""


def _hydrogen_pair(tmp_path, pseudo_dir, kgrid, m2) -> Calculator:
    """The two-cell supercell, ferromagnetic at ``m2 = 0.8`` and staggered at -0.8."""
    path = tmp_path / f"h_pair{m2:+.1f}.in"
    path.write_text(ANTIFERROMAGNET.format(
        m2=m2, k0=kgrid[0], k1=kgrid[1], k2=kgrid[2]))
    return Calculator.from_file(path, pseudo_dir=pseudo_dir)


@pytest.mark.slow
def test_a_collinear_seed_reaches_the_state_the_tiled_loop_cannot(
        tmp_path, pseudo_dir):
    """The seed's whole claim, as two numbers: it moves, and it moves to the
    right place.

    The tiled state is an **exact fixed point** of this loop -- nothing in a
    self-consistent iteration breaks spin symmetry on its own -- so a run with
    no seed and nothing applied converges in one iteration and stays
    ferromagnetic however many cells it has. Here the same cell is seeded with
    ``cos(pi x)``, one period over two cells, and what it converges to is the
    antiferromagnet: **3.085 mRy per cell below** the tiled ferromagnet, which
    is a state the method could not previously reach at all.

    The reference is the same two atoms as a real supercell, seeded the same way
    through ``starting_magnetization`` on two species, and what is compared is
    the ``(1, 0, 0)`` Fourier component of the magnetization -- the amplitude of
    the wave itself, and the one quantity here that needs no partition of space.
    Measured against that supercell's 0.36546537 mu_B per cell:

    | ``nbnd`` | ``|m_Q|`` error | ``E`` above the supercell, Ry |
    |---|---|---|
    | 16 | 1.51e-03 | +1.53e-04 |
    | 24 | 5.04e-04 | +6.80e-05 |
    | 40 | 1.74e-04 | +2.32e-05 |
    | 64 | 3.10e-05 | +5.57e-06 |

    so the seeded state converges to the supercell's own in the one knob the
    method has, and the energy falls towards it **from above** at every rung,
    which is the nested-basis property the total energy is the only quantity
    here to have. ``nbnd = 64`` is left out of the test and quoted: three rungs
    say the same thing and the fourth costs a minute.

    **Why a smooth seed rather than a per-cell sign.** ``cos(pi x)`` changes
    sign halfway through each cell, so the *cell* moments it leaves are small
    (0.0739 against the atoms' 0.365) while the *atoms* are cleanly opposite --
    and the supercell agreement is what says that is the physical state rather
    than a weak wave. It is why the number checked here is the Fourier amplitude
    and not
    :meth:`~defumat.ultracell.driver.UltracellResult.cell_moments`.
    """
    shape, kgrid = (2, 1, 1), (2, 2, 2)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    calculator = _hydrogen(tmp_path, pseudo_dir, folded)
    scf = calculator.get_scf(conv_thr=1e-11, nbnd=8)
    assert scf.converged

    miller = np.array([[1, 0, 0]])
    # ``_fourier`` is the *mean* of the field against a plane wave, so it is
    # already comparable across two different boxes; the unit cell's volume
    # turns it into Bohr magnetons per cell, which is what the numbers above
    # are quoted in.
    volume = float(calculator.system.cell.volume)
    staggered = _hydrogen_pair(tmp_path, pseudo_dir, kgrid, -0.8)
    reference = staggered.get_scf(conv_thr=1e-11, nbnd=16, max_iterations=200)
    assert reference.converged
    their_grid = tuple(np.asarray(reference.density).shape[1:])
    theirs = np.asarray(reference.density)
    their_moment = float(np.abs(_fourier(
        theirs[0] - theirs[1], their_grid, miller))[0]) * volume
    their_energy = float(reference.total_energy) / 2.0
    assert their_moment > 0.1, their_moment

    # The tiled arm: no seed, nothing applied, and it does not move.
    tiled = run_ultracell(
        calculator.system, calculator.pseudos, scf, shape, kgrid, nbnd=16,
        conv_thr=1e-10, states_conv_thr=1e-8,
    )
    assert tiled.converged and tiled.iterations == 1
    assert np.abs(np.asarray(tiled.cell_moments())
                  - float(scf.magnetization)).max() < 1e-5
    assert float(tiled.total_energy) > their_energy + 2.5e-3

    seed = lambda x: np.cos(np.pi * x[..., 0])
    moments, energies = [], []
    for nbnd in (16, 24, 40):
        result = run_ultracell(
            calculator.system, calculator.pseudos, scf, shape, kgrid,
            nbnd=nbnd, seed_magnetization=seed, conv_thr=1e-10,
            states_conv_thr=1e-8, max_iterations=200,
            david=None if nbnd < 40 else 2,
        )
        assert result.converged
        box = tuple(result.ultracell.grid)
        ours = np.asarray(result.density)
        moments.append(abs(float(np.abs(_fourier(
            ours[0] - ours[1], box, miller))[0]) * volume - their_moment)
            / their_moment)
        energies.append(float(result.total_energy) - their_energy)
        jax.clear_caches()

    assert moments == sorted(moments, reverse=True), moments
    assert energies == sorted(energies, reverse=True), energies
    assert all(e > 0.0 for e in energies), energies
    assert moments[-1] < 3e-4 and energies[-1] < 4e-5, (moments, energies)


class _Overridden:
    """A system with one attribute replaced, for firing one refusal at a time.

    The refusals are cheap to write and easy to leave **dead** -- the DFT+U one
    read ``system.hubbard_u`` for a while, an attribute ``System`` does not have,
    so it never fired and the ultracell would have run a Hubbard calculation
    without its ``U``. A refusal is only a promise if it has been shown to fire.
    """

    def __init__(self, system, **replaced):
        self._system, self._replaced = system, replaced

    def __getattr__(self, name):
        if name in self._replaced:
            return self._replaced[name]
        return getattr(self._system, name)


@pytest.mark.parametrize("replaced, message", [
    ({"spiral_q": (0.0, 0.0, 0.25)}, "spin spiral"),
    ({"nosym": False}, "nosym"),
])
def test_every_refusal_fires(replaced, message, tmp_path, pseudo_dir):
    """Each one is triggered on its own, on a cell that otherwise passes."""
    from defumat.ultracell.driver import require_an_ultracell_regime

    calculator = _silicon(tmp_path, pseudo_dir, (2, 2, 2))
    basis = build_basis(calculator.system)
    # the unmodified system is accepted, so a refusal below is the override
    require_an_ultracell_regime(calculator.system, calculator.pseudos, basis)

    with pytest.raises(NotImplementedError, match=message):
        require_an_ultracell_regime(
            _Overridden(calculator.system, **replaced), calculator.pseudos, basis
        )


def test_the_hubbard_refusal_fires(tmp_path, pseudo_dir):
    """Separately, because it reads a field whose name was wrong once."""
    from defumat.hubbard.manifold import HubbardInput
    from defumat.ultracell.driver import require_an_ultracell_regime

    calculator = _silicon(tmp_path, pseudo_dir, (2, 2, 2))
    basis = build_basis(calculator.system)
    hubbard = HubbardInput(projectors="atomic",
                           parameters=(("Si", 3, 1, 0.1, 0.0, 0.0, 0.0),))
    with pytest.raises(NotImplementedError, match="DFT\\+U"):
        require_an_ultracell_regime(
            _Overridden(calculator.system, hubbard=hubbard),
            calculator.pseudos, basis,
        )


def test_the_refusals_name_themselves():
    """Every regime stage 1 has not been measured in is refused by name."""
    from defumat.ultracell.potential import require_an_ultracell_functional
    from defumat.xc.functional import get_functional

    with pytest.raises(NotImplementedError, match="gradient-corrected"):
        require_an_ultracell_functional(get_functional("PBE"))

    # and the meta-GGA branch, which refuses for a different reason: tau is a
    # property of the states, not of the density the ultracell loop carries.
    with pytest.raises(NotImplementedError, match="meta-GGA"):
        require_an_ultracell_functional(get_functional("tb09"))


MAGNETIC = """&control
 calculation='scf'
/
&system
 ibrav=2, celldm(1)=10.20, nat=2, ntyp=1, ecutwfc=12.0,
 nosym=.true., noinv=.true.,
 nspin=2, starting_magnetization(1)=0.2,
 occupations='smearing', smearing='gaussian', degauss=0.02
/
&electrons
 conv_thr=1.0d-8
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""


def test_a_field_needs_two_channels_to_split(tmp_path, pseudo_dir):
    """An applied ``B`` with ``nspin = 1`` is refused rather than ignored.

    There is one density and no channel for the field to move an electron
    into, so the field would be silently dropped -- and a run that reports a
    perfectly converged unmodulated state is the worst way to be told.
    """
    from defumat.ultracell.driver import _as_field
    from defumat.ultracell.grid import Ultracell

    ultracell = Ultracell.build((2, 1, 1), (4, 4, 4))
    with pytest.raises(ValueError, match="two spin channels"):
        _as_field(None, lambda x: np.zeros(x.shape[:-1]), ultracell, 1, 1)


def test_a_field_needs_a_magnetization_to_act_on(tmp_path, pseudo_dir):
    """The same refusal for the *other* cell that reaches it: ``nspin_mag = 1``.

    A spin-orbit run on a nonmagnetic crystal has spinor wavefunctions and a
    **scalar** density, so its potential and the frozen reference potential it
    is subtracted from both have one component. Promoting them to four here so
    that a field has somewhere to go would leave ``dV`` in a representation the
    eigenvalues are not in. It is refused with different advice from the
    ``nspin = 1`` case, which is why the two are separate branches and separate
    tests: here the fix is to give the unit cell a seed so that ``domag`` is
    true, and the susceptibility that comes back is then the Pauli one.
    """
    from defumat.ultracell.driver import _as_field
    from defumat.ultracell.grid import Ultracell

    ultracell = Ultracell.build((2, 1, 1), (4, 4, 4))
    with pytest.raises(ValueError, match="carries a magnetization"):
        _as_field(None, lambda x: np.zeros(x.shape[:-1] + (3,)),
                  ultracell, 4, 1)


def test_the_collinear_field_carries_add_bfields_own_sign():
    """``v_up -= B``, ``v_dw += B`` -- ``add_bfield.f90:237-238``.

    Transcribed here rather than inferred, because a flipped sign is a
    calculation that converges to the state with the moment the other way up:
    every symmetry check passes, the energy is the same by time reversal, and
    only a comparison against a field applied through some *other* route sees
    it. That comparison is
    :func:`test_a_uniform_field_is_the_unit_cell_under_the_same_field`; this is
    the cheap half of it.
    """
    from defumat.ultracell.driver import _as_field
    from defumat.ultracell.grid import Ultracell

    ultracell = Ultracell.build((2, 1, 1), (4, 4, 4))
    field = np.asarray(_as_field(None, lambda x: np.full(x.shape[:-1], 0.3),
                                 ultracell, 2, 2))
    assert field[0] == pytest.approx(np.full(ultracell.grid, -0.3), abs=1e-14)
    assert field[1] == pytest.approx(np.full(ultracell.grid, +0.3), abs=1e-14)

    # and a scalar potential is felt in full by both, which is the other rule
    # in the same function and the one that would otherwise be half of it.
    both = np.asarray(_as_field(lambda x: np.full(x.shape[:-1], 0.1), None,
                                ultracell, 2, 2))
    assert both[0] == pytest.approx(np.full(ultracell.grid, 0.1), abs=1e-14)
    assert both[1] == pytest.approx(np.full(ultracell.grid, 0.1), abs=1e-14)


@pytest.mark.slow
def test_a_field_converged_ground_state_is_refused(tmp_path, pseudo_dir):
    """The frozen eigenvalues would carry a Zeeman term ``dV`` does not.

    ``dV`` is rebuilt here from the density alone, where the eigenvalues the
    matrix diagonal is made of were converged under whatever field the SCF
    *ended* with -- which ``reducebf`` and the fixed-spin-moment scheme both
    make different from the input. The difference is a rigid shift between the
    channels, and an ultracell would report it as a modulation.
    """
    calculator = _silicon(tmp_path, pseudo_dir, (2, 2, 2))
    scf = calculator.get_scf(conv_thr=1e-10, nbnd=8)
    under_field = _Overridden(scf, magnetic_field=object())
    with pytest.raises(NotImplementedError, match="converged under a magnetic"):
        run_ultracell(
            calculator.system, calculator.pseudos, under_field, (2, 1, 1),
            (1, 2, 2), nbnd=8, conv_thr=1e-8,
        )


@pytest.mark.slow
def test_a_symmetric_run_is_refused(tmp_path, pseudo_dir):
    """Symmetry on is refused, and the reason is not that nothing symmetrises.

    Nothing here does. What breaks is upstream: the unit-cell density being
    expanded around would have been integrated over a reduced wedge, and then
    the tiled density is not the fixed point the folded k-set reproduces, so
    the first iteration starts from a state that is not stationary and nothing
    says so.
    """
    path = tmp_path / "si_sym.in"
    path.write_text(SILICON.format(k0=2, k1=2, k2=2, ecut="12.0").replace(
        "ecutwfc=12.0,\n nosym=.true., noinv=.true.\n", "ecutwfc=12.0\n"))
    calculator = Calculator.from_file(path, pseudo_dir=pseudo_dir)
    scf = calculator.get_scf(conv_thr=1e-8, nbnd=8)
    with pytest.raises(NotImplementedError, match="nosym"):
        run_ultracell(
            calculator.system, calculator.pseudos, scf, (2, 1, 1), (1, 2, 2),
            nbnd=8, conv_thr=1e-8,
        )


@pytest.mark.slow
def test_an_unconverged_seed_is_refused(tmp_path, pseudo_dir):
    """The states are only a basis if the density they diagonalise is the answer.

    Nothing downstream of ``fixed_density_states`` asks whether the density it
    was handed is converged, and the unit cell's density is precisely the thing
    an ultracell holds *fixed* -- there is no later iteration in which a bad
    seed could work itself out. ``OPEN.md`` Part V item 2 is this same hole
    found in the magnon stack, where the ``Calculator`` door refuses and the
    functional door does not; here both do.
    """
    calculator = _silicon(tmp_path, pseudo_dir, (2, 2, 2))
    scf = calculator.get_scf(conv_thr=1e-10, nbnd=8)
    stalled = _Overridden(scf, converged=False, accuracy=3.1e-4, iterations=100)
    with pytest.raises(ValueError, match="converged ground state"):
        run_ultracell(
            calculator.system, calculator.pseudos, stalled, (2, 1, 1), (1, 2, 2),
            nbnd=8, conv_thr=1e-8,
        )


# -- stage 3b: a spinor ultracell, where the magnetization is a vector -------

#: The same hydrogen lattice as stage 3a, one regime up. The moment is seeded
#: **off every axis**, along ``(1,1,1)/sqrt(3)``, and that is the whole point of
#: the cell: all three magnetization components are then live, and the magnetic
#: part of the potential is largely *off-diagonal* in the spinor basis, which is
#: exactly the machinery a collinear run does not have. Seeded along ``z`` it
#: would be a collinear run wearing a spinor's clothes, and every one of the
#: tests below would pass with the transverse terms deleted.
NONCOLLINEAR = """&control
 calculation='scf'
/
&system
 ibrav=0, celldm(1)=5.5, nat={nat}, ntyp=1, ecutwfc=15.0,
 nosym=.true., noinv=.true.,
 noncolin=.true., starting_magnetization(1)=0.8,
 angle1(1)={angle1}, angle2(1)={angle2},
 occupations='smearing', smearing='gaussian', degauss=0.02
/
&electrons
 conv_thr=1.0d-11
 mixing_beta=0.3
/
CELL_PARAMETERS alat
{rows}
ATOMIC_SPECIES
 H 1.008 H.pz-vbc.UPF
ATOMIC_POSITIONS crystal
{atoms}
K_POINTS automatic
 {k0} {k1} {k2} 0 0 0
"""


def _noncollinear(tmp_path, pseudo_dir, shape, kgrid, angle1=54.735610,
                  angle2=45.0, tag="nc") -> Calculator:
    """The lattice as a ``shape`` supercell of hydrogens, moments all parallel."""
    rows = "\n".join(" %.10f %.10f %.10f" % tuple(v)
                     for v in np.diag(shape).astype(float))
    frac = np.stack(
        np.meshgrid(*[np.arange(n) / n for n in shape], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    atoms = "\n".join(" H %.10f %.10f %.10f" % tuple(x) for x in frac)
    path = tmp_path / f"h_{tag}.in"
    path.write_text(NONCOLLINEAR.format(
        nat=len(frac), rows=rows, atoms=atoms, angle1=angle1, angle2=angle2,
        k0=kgrid[0], k1=kgrid[1], k2=kgrid[2]))
    return Calculator.from_file(path, pseudo_dir=pseudo_dir)


def _total_moment(density, volume):
    """The cell-integrated magnetization vector of an ``(4, *box)`` density."""
    m = np.asarray(density)[1:]
    return m.reshape(3, -1).sum(axis=1) * float(volume) / m[0].size


@pytest.mark.slow
def test_a_noncollinear_ultracell_is_the_tiled_unit_cell(tmp_path, pseudo_dir):
    """The ``nspin = 4`` null, and it sees three things the others cannot.

    The unpolarized null cannot see the spin path at all and the collinear one
    cannot see the transverse components: a build that dropped ``m_x`` and
    ``m_y``, or contracted the spinor halves in the wrong order, passes both.
    Here the moment points along ``(1,1,1)/sqrt(3)``, so every component of the
    density is non-zero, the potential's magnetic part is mostly off-diagonal in
    spin, and the null still has to come back as the exactly tiled state.
    """
    shape, kgrid = (2, 1, 1), (2, 2, 2)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    calculator = _noncollinear(tmp_path, pseudo_dir, (1, 1, 1), folded)
    assert calculator.system.nspin == 4 and calculator.system.nspin_mag == 4
    scf = calculator.get_scf(conv_thr=1e-11, nbnd=16)
    assert scf.converged

    result = run_ultracell(
        calculator.system, calculator.pseudos, scf, shape, kgrid,
        nbnd=16, conv_thr=1e-9, states_conv_thr=1e-8, mixing_beta=0.3,
    )
    assert result.converged and result.iterations == 1
    assert np.abs(np.asarray(result.delta_v)).max() < 1e-14

    tiled = np.asarray(result.ultracell.tile(jnp.asarray(scf.density)))
    density = np.asarray(result.density)
    assert density.shape[0] == 4
    assert np.abs(density - tiled).max() / np.abs(tiled).max() < 1e-5

    # The electron count is the ultracell's, which is what caught stage 1's one
    # real bug, and it is asserted apart from the moment for the reason stage 3a
    # gives: a wrong ``N`` is a factor, not a last digit.
    element = float(calculator.system.cell.volume) / np.prod(
        result.ultracell.cell_grid)
    assert density[0].sum() * element == pytest.approx(
        float(calculator.system.nelec if hasattr(calculator.system, "nelec")
              else 1.0) * shape[0], rel=1e-8)

    # Every cell carries the same moment **vector**, direction included.
    moments = result.cell_moments()
    assert moments.shape == (shape[0], 3)
    assert np.abs(moments - moments[0]).max() < 1e-9
    direction = moments[0] / np.linalg.norm(moments[0])
    assert direction == pytest.approx(np.full(3, 1 / np.sqrt(3)), abs=2e-3)

    # **The energy, per unit cell, on the one regime where its contraction has
    # four components.** ``deband`` is one sum over ``(n, m_x, m_y, m_z)``
    # against the same four components of the potential, which is QE's
    # ``delta_e`` and has no spinor branch -- and the unit tests check that
    # contraction on arithmetic, where this checks it on a converged state.
    # Smeared, so ``demet`` is live and its own ``/N`` is exercised here rather
    # than on a synthetic array.
    assert result.total_energy == pytest.approx(scf.total_energy, abs=1e-10)
    assert result.field_energy is None


@pytest.mark.slow
def test_a_turning_field_turns_the_magnetization(tmp_path, pseudo_dir):
    """The magnetic null shown capable of failing, in the way only a spinor can.

    A collinear run can be driven to modulate the *length* of its moment and
    nothing else. What a spinor adds is a modulation of the **direction**, and
    the field that drives one has to turn from cell to cell -- which is a vector
    field, and a quantity a collinear ``B(r)`` cannot express at all.

    The discriminating assertion is the **sense** of the turn: the moments pick
    up a transverse component whose sign follows the applied field's, cell by
    cell. A flipped ``m_y`` would give the mirror texture, which is degenerate
    with this one in energy (there is no spin-orbit coupling here), converges
    just as well, and is caught by nothing else in this file.
    """
    shape, kgrid = (2, 1, 1), (2, 2, 2)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    # The moment along ``x``, so a field along ``+-y`` turns it in a plane and
    # the sign of the turn is unambiguous.
    calculator = _noncollinear(tmp_path, pseudo_dir, (1, 1, 1), folded,
                              angle1=90.0, angle2=0.0, tag="turn")
    scf = calculator.get_scf(conv_thr=1e-11, nbnd=16)
    assert scf.converged

    amplitude = 0.01

    def turning(x):
        """``+y`` in the first cell and ``-y`` in the second."""
        sign = np.cos(2 * np.pi * x[..., 0] / shape[0] - np.pi / 2)
        zero = np.zeros_like(sign)
        return amplitude * np.stack([zero, sign, zero], axis=-1)

    result = run_ultracell(
        calculator.system, calculator.pseudos, scf, shape, kgrid,
        nbnd=16, conv_thr=1e-9, states_conv_thr=1e-8, mixing_beta=0.3,
        magnetic_field=turning, max_iterations=120,
    )
    assert result.converged

    moments = result.cell_moments()
    # The field averages to ``+y`` over the first cell and ``-y`` over the
    # second, so the transverse components are equal and opposite...
    assert moments[0][1] > 0.0 and moments[1][1] < 0.0
    assert moments[0][1] == pytest.approx(-moments[1][1], rel=1e-3)
    # ...and large enough that this is a response rather than round-off.
    assert abs(moments[0][1]) > 1e-3 * abs(moments[0][0])

    # The **charge** must not respond at linear order, which is the half that
    # says the two channels are not coupled where they should not be: the
    # energy is invariant under flipping every spin together with the sign of
    # ``B``, so the charge is even in ``B`` and the magnetization odd.
    base = run_ultracell(
        calculator.system, calculator.pseudos, scf, shape, kgrid,
        nbnd=16, conv_thr=1e-9, states_conv_thr=1e-8, mixing_beta=0.3,
    )
    charge = np.asarray(result.density)[0]
    unperturbed = np.asarray(base.density)[0]
    moved = np.abs(charge - unperturbed).max() / unperturbed.max()
    transverse = abs(moments[0][1]) / abs(moments[0][0])
    assert moved < 0.2 * transverse, (moved, transverse)


@pytest.mark.slow
def test_a_uniform_vector_field_is_the_unit_cell_under_the_same_field(
        tmp_path, pseudo_dir):
    """The applied field's sign and magnitude, against a route with no ultracell.

    At ``N = 1`` a *uniform* applied field is the same physics as an ordinary
    SCF carrying ``B_field(1:3)``, which shares nothing with this code path: it
    goes through ``add_bfield.f90``'s expression inside a plane-wave SCF, where
    this expands the **field-free** states of the same cell in a basis and never
    applies ``H`` again. For a *vector* field that comparison reaches the
    transverse components, which no collinear identity can -- and a sign there
    is the mirror texture, degenerate in energy and invisible to every symmetry
    check.

    The reference run says something clean on its own and it is asserted first:
    **the moment aligns with B**. Without anisotropy or spin-orbit coupling,
    turning the moment costs nothing, so a field of any size turns it all the
    way and its length is set by the exchange alone. That is what fixes the sign
    of the coupling, independently of how this code spells it.

    The field is along the ground-state moment with a *small* transverse tilt,
    so the response stays linear: an oblique field rotates the moment by tens of
    degrees, which a truncated basis reproduces far more slowly and which would
    hide the sign inside a large deformation.

    **The frozen-state solve warns at the top rungs and that is expected.**
    ``states_conv_thr = 1e-8`` puts ``ethr`` at 1e-9, which the Davidson budget
    does not reach for a handful of bands high in the empty manifold -- the
    threshold ``run_ultracell``'s own docstring says not to tighten blindly for
    a large ``nbnd``. It is kept because the recorded ladder was measured there
    and because the monotone fall is itself the evidence that those bands carry
    no weight: a basis corrupted at the top would not converge to the reference.
    """
    kgrid = (2, 2, 2)
    calculator = _noncollinear(tmp_path, pseudo_dir, (1, 1, 1), kgrid,
                               angle1=90.0, angle2=0.0, tag="uniform_free")
    scf = calculator.get_scf(conv_thr=1e-11, nbnd=16)
    assert scf.converged

    field = (0.005, 0.0005, 0.0)
    path = tmp_path / "h_uniform_held.in"
    text = (tmp_path / "h_uniform_free.in").read_text().replace(
        "occupations='smearing'",
        f"B_field(1)={field[0]}, B_field(2)={field[1]}, B_field(3)={field[2]},\n"
        " occupations='smearing'")
    path.write_text(text)
    held = Calculator.from_file(path, pseudo_dir=pseudo_dir)
    reference = held.get_scf(conv_thr=1e-11, nbnd=16)
    assert reference.converged

    volume = float(calculator.system.cell.volume)
    m_ref = _total_moment(reference.density, volume)
    # The moment aligns with the field, which is the sign statement.
    assert m_ref @ np.asarray(field) > 0
    direction = m_ref / np.linalg.norm(m_ref)
    expected = np.asarray(field) / np.linalg.norm(field)
    assert direction == pytest.approx(expected, abs=2e-3)

    def uniform(x):
        return np.broadcast_to(np.asarray(field), x.shape[:-1] + (3,))

    errors, energies = [], []
    for nbnd in (16, 32, 64):
        result = run_ultracell(
            calculator.system, calculator.pseudos, scf, (1, 1, 1), kgrid,
            nbnd=nbnd, conv_thr=1e-11, states_conv_thr=1e-8,
            mixing_beta=0.3, max_iterations=150, magnetic_field=uniform,
        )
        assert result.converged
        moment = result.cell_moments()[0]
        errors.append(float(np.linalg.norm(moment - m_ref)
                            / np.linalg.norm(m_ref)))
        energies.append((result.total_energy, result.field_energy))
        jax.clear_caches()

    # **Monotone, and no sign is asserted on the approach.** The ultracell
    # Hamiltonian moves with its own truncated density, so an ultracell answer
    # is not bounded on either side of the reference -- the same warning stage 1
    # attaches to its charge ladder, and a test written as "from below" would
    # fail spuriously.
    assert errors == sorted(errors, reverse=True), errors
    assert errors[0] < 3e-2 and errors[-1] < 3e-3, errors

    # **The energy under a *vector* field, where the Zeeman contraction has
    # three components rather than one.** The convention is QE's and Elk's --
    # a field put in by hand is carried beside the total, not inside it -- and
    # what is bounded is therefore the **sum**, since the quantity a run under a
    # field minimises is the full energy. The collinear test measures the same
    # pair; this is the one where a sign living in a transverse component could
    # hide, because ``m_y`` is a twentieth of ``m_x`` here.
    free = [total + field_energy for total, field_energy in energies]
    exact = reference.total_energy + reference.field_energy
    assert free == sorted(free, reverse=True), free
    for nbnd, value in zip((16, 32, 64), free):
        assert value > exact, (
            f"nbnd = {nbnd}: {value:.12f} Ry against {exact:.12f}; what a run "
            f"under a field minimises is total_energy + field_energy"
        )


@pytest.mark.slow
def test_the_noncollinear_ultracell_converges_to_the_supercell(
        tmp_path, pseudo_dir):
    """The canonical check, one regime up: all four density components at once.

    Stages 1 and 3a both measured the method the same way and so does this --
    against a real ``N``-cell supercell run through this package's own SCF,
    under the same applied potential, which shares the unit-cell machinery and
    none of the ultracell assembly. What ``nspin = 4`` adds is that the thing
    being compared is a *vector field* rather than one or two scalars.

    **The perturbation is a scalar potential on purpose.** A spin-resolved
    external field has no slot in a supercell's ``vltot``, which is one scalar
    broadcast to every channel -- stage 3a's finding, unchanged. So the
    magnetization's response here is entirely **indirect**: the local exchange
    splitting follows the local charge, which makes this a test of the coupled
    spinor loop rather than of a field's sign. The sign is pinned separately, by
    the uniform-field identity above.

    **The two magnetizations are aligned before they are compared, and the
    angle is asserted to be negligible rather than assumed to be.** Without
    spin-orbit coupling the moment's direction is a Goldstone mode, so two SCFs
    started from the same seed could in principle converge to directions
    differing by a small rigid rotation -- and a component-by-component
    comparison would read that rotation as basis error, putting a floor under
    the ladder that has nothing to do with ``nbnd``. It is measured at 7.7e-7
    rad on this cell, so the alignment is a no-op; the check is here because the
    failure it guards against would look like a wrong number rather than a wrong
    answer.
    """
    shape, kgrid = (2, 1, 1), (2, 2, 2)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    calculator = _noncollinear(tmp_path, pseudo_dir, (1, 1, 1), folded,
                               tag="lad_unit")
    scf = calculator.get_scf(conv_thr=1e-11, nbnd=16)
    assert scf.converged

    supercell = _noncollinear(tmp_path, pseudo_dir, shape, kgrid, tag="lad_sup")
    reference_calculation = Calculation(supercell.system, supercell.pseudos)
    grid_sup = tuple(reference_calculation.basis.dense.grid)
    axes = [np.arange(m) / m * n for m, n in zip(grid_sup, shape)]
    coordinates = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    reference = run_scf(
        supercell.system, supercell.pseudos,
        calculation=with_external_potential(
            reference_calculation,
            jnp.asarray(_modulation(shape)(coordinates)),
        ),
        conv_thr=1e-11, nbnd=32, max_iterations=300,
    )
    assert reference.converged
    rho_sup = np.array(reference.density)

    volume = float(calculator.system.cell.volume)
    a = _total_moment(rho_sup, volume)
    b = _total_moment(scf.density, volume)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    angle = float(np.arccos(np.clip(a @ b, -1.0, 1.0)))
    assert angle < 1e-4, f"the two moments differ by a rigid rotation of {angle} rad"

    miller = None
    errors = []
    for nbnd in (8, 16, 32):
        result = run_ultracell(
            calculator.system, calculator.pseudos, scf, shape, kgrid,
            nbnd=nbnd, conv_thr=1e-10, states_conv_thr=1e-8,
            mixing_beta=0.3, max_iterations=200, external=_modulation(shape),
        )
        assert result.converged
        grid_u = tuple(result.ultracell.grid)
        if miller is None:
            axes = [np.fft.fftfreq(min(u, s)) * min(u, s)
                    for u, s in zip(grid_u, grid_sup)]
            miller = np.stack(np.meshgrid(*axes, indexing="ij"),
                              axis=-1).reshape(-1, 3).astype(int)
        rho_u = np.asarray(result.density)
        errors.append([
            float(np.abs(_fourier(rho_u[c], grid_u, miller)
                         - _fourier(rho_sup[c], grid_sup, miller)).max()
                  / np.abs(_fourier(rho_sup[c], grid_sup, miller)).max())
            for c in range(4)
        ])
        jax.clear_caches()

    # Every component falls, and the three magnetic ones track each other --
    # they must, since the moment lies along (1,1,1)/sqrt(3), and their agreeing
    # is a check on the whole component axis that costs nothing to read.
    for component in range(4):
        column = [row[component] for row in errors]
        assert column == sorted(column, reverse=True), (component, column)
    for row in errors:
        assert max(row[1:]) / min(row[1:]) < 1.1, row
    assert max(errors[0]) < 3e-3 and max(errors[-1]) < 5e-4, errors


#: A helix as a real supercell: ``n`` species pointing at one UPF, because
#: ``angle1``/``angle2`` are per species and a texture is per atom. The moments
#: lie in the plane perpendicular to the modulated axis is not the point --- what
#: matters is that they are perpendicular to the **reference**'s own direction,
#: which the test below sets along ``z``, for the reason it gives.
HELIX = """&control
 calculation='scf'
/
&system
 ibrav=0, celldm(1)=5.5, nat={nat}, ntyp={nat}, ecutwfc=15.0,
 nosym=.true., noinv=.true.,
 noncolin=.true.,
{magnetism}
 occupations='smearing', smearing='gaussian', degauss=0.02
/
&electrons
 conv_thr=1.0d-11
 mixing_beta=0.3
/
CELL_PARAMETERS alat
{rows}
ATOMIC_SPECIES
{species}
ATOMIC_POSITIONS crystal
{atoms}
K_POINTS automatic
 {k0} {k1} {k2} 0 0 0
"""


def _helix_supercell(tmp_path, pseudo_dir, n, kgrid) -> Calculator:
    """``n`` hydrogens along ``x``, each moment turned ``360/n`` from the last."""
    rows = "\n".join(" %.10f %.10f %.10f" % tuple(v)
                     for v in np.diag((n, 1, 1)).astype(float))
    names = [f"H{i + 1}" for i in range(n)]
    species = "\n".join(f" {name} 1.008 H.pz-vbc.UPF" for name in names)
    atoms = "\n".join(" %s %.10f 0.0 0.0" % (names[i], i / n) for i in range(n))
    magnetism = "\n".join(
        f" starting_magnetization({i + 1})=0.8, angle1({i + 1})=90.0, "
        f"angle2({i + 1})={360.0 * i / n},"
        for i in range(n))
    path = tmp_path / f"h_helix{n}.in"
    path.write_text(HELIX.format(nat=n, rows=rows, species=species, atoms=atoms,
                                 magnetism=magnetism, k0=kgrid[0], k1=kgrid[1],
                                 k2=kgrid[2]))
    return Calculator.from_file(path, pseudo_dir=pseudo_dir)


def _pitch(moments):
    """The angle each cell's moment turns about ``z``, in degrees.

    The projection onto the plane perpendicular to ``z`` is the right thing
    *here* and would not be in general: the reference's magnetization is along
    ``z`` and the rotation the protected sector is protected by is the rotation
    about ``z``, so this is the pitch about the seed's own axis. What the
    projection throws away is the moments' common out-of-plane component, and
    that is a quantity in its own right -- see :func:`_cone`.
    """
    angles = np.arctan2(np.asarray(moments)[:, 1], np.asarray(moments)[:, 0])
    return np.degrees(np.diff(np.unwrap(angles)))


def _cone(moments):
    """How far each cell's moment stands off the helix plane, in degrees.

    An exact helix is flat, so this is zero; a truncated one cants uniformly
    towards the direction its basis was built around, which is a ferromagnetic
    remnant at ``Q = 0`` that the wave's own Fourier component cannot see.
    """
    m = np.asarray(moments)
    return np.degrees(np.arcsin(
        np.clip(m[:, 2] / np.linalg.norm(m, axis=1), -1.0, 1.0)))


@pytest.mark.slow
def test_a_seeded_helix_keeps_the_pitch_it_was_given(tmp_path, pseudo_dir):
    """A spontaneous helix: which sector the loop stays in, and what the
    truncation charges for it.

    A helix of pitch ``N`` cells is invariant under a translation by one cell
    followed by a spin rotation of ``360/N`` about its own axis, and the
    self-consistent map commutes with both -- so that sector is closed and a run
    started in it stays in it. **In the truncated problem it is closed only if
    the basis is**, and the basis is the unit cell's own spinors: a rotation
    about the reference's own magnetization is a *phase* on each of them, since
    without spin-orbit coupling they are eigenstates of ``sigma . e_0``, and a
    rotation about any other axis is not. So there is one closed sector and its
    axis is ``e_0``: **write the seed about the direction the reference's moment
    already points along.**

    **What that buys is the iteration count and the frame, not the pitch.**
    Seeded about ``z`` with the reference along ``z``, four cells at
    ``nbnd = 16`` converge in **14** iterations; seeded about ``z`` with the
    reference along ``(1,1,1)/sqrt(3)`` the same run takes **290** and arrives
    at a helix turning about ``(1,1,1)/sqrt(3)`` instead. It is the same state:
    after one global rotation the two runs' cell moments agree to 1.23e-3 on
    moments of 0.272, 0.45 per cent, and their energies to 3.5e-9 Ry. A global
    spin rotation costs nothing without spin-orbit coupling, so the unprotected
    run is not wrong -- it is traversing a flat manifold to reach the frame its
    basis prefers, which is what 290 iterations buys and why the fix is the axis
    rather than ``mixing_beta``. That is this reference's outcome and not every
    reference's: with the reference in the helix plane the same basis converges
    to a distorted helix instead, and the fix for both, and for the cone below,
    is the Kramers-closed basis (:func:`test_kramers_pairs_remove_the_lean_in_both_frames`).

    **What the truncation costs is the canting**, and that is the half a pitch
    check cannot see. The converged moments stand off the helix plane by a
    *uniform* angle -- which the closed sector allows, a cone being as invariant
    under the pair as a flat helix -- and it is a ferromagnetic remnant at
    ``Q = 0``, along the axis, that the wave's own Fourier component is blind to.
    It falls with the one knob the method has, where the pitch has nothing left
    to converge:

    | ``nbnd`` | pitch error, deg | cone, deg | net moment | ``E`` above the supercell |
    |---|---|---|---|---|
    | 16 | 1.35e-04 | 10.64 | 0.0628 | +4.37e-04 |
    | 24 | 8.50e-05 | 6.30 | 0.0370 | +2.54e-04 |
    | 40 | 3.27e-05 | 3.00 | 0.0176 | +1.22e-04 |
    | 64 | 1.20e-05 | 0.91 | 0.0054 | +3.74e-05 |

    so reading the pitch alone at ``nbnd = 16`` would call the state exact while
    18 per cent of its moment stood off the helix. Across that range the cone
    falls by 11.65 and the energy error by 11.68, which is reported as a fact
    rather than explained. The reference is the same four atoms as a real
    supercell with the helix seeded per species, whose site moments are flat to
    1e-5 and whose energy is **0.739 mRy per cell below** the ferromagnet the
    ultracell expands around -- which is what makes reaching it worth a seed.

    ``nbnd = 8`` is quoted and not run: it holds the pitch to 4.2e-3 degrees,
    the closure not needing convergence, and does not reach ``conv_thr`` in 300
    iterations.
    """
    n, kgrid = 4, (1, 2, 2)
    shape = (n, 1, 1)
    folded = tuple(a * b for a, b in zip(shape, kgrid))
    # The reference points along ``z`` and the helix turns about ``z``: the one
    # arrangement in which the truncated basis is closed under the rotations
    # relating the cells.
    calculator = _noncollinear(tmp_path, pseudo_dir, (1, 1, 1), folded,
                               angle1=0.0, angle2=0.0, tag="helix_unit")
    scf = calculator.get_scf(conv_thr=1e-11, nbnd=8, max_iterations=300)
    assert scf.converged
    moment = np.asarray(scf.magnetization_vector)
    assert abs(moment[2]) > 0.5 and np.abs(moment[:2]).max() < 1e-4, moment

    supercell = _helix_supercell(tmp_path, pseudo_dir, n, kgrid)
    reference = supercell.get_scf(conv_thr=1e-11, nbnd=16, max_iterations=300)
    assert reference.converged
    sites = np.asarray(reference.site_moments)
    assert np.abs(_pitch(sites) - 360.0 / n).max() < 1e-2, sites
    # The state this is a reference for is a *flat* helix, which is the claim
    # the canting below is measured against.
    assert np.abs(_cone(sites)).max() < 1e-2, sites
    volume = float(calculator.system.cell.volume)
    miller = np.array([[1, 0, 0]])
    theirs = np.asarray(reference.density)[1:]
    their_grid = tuple(theirs.shape[1:])
    their_amplitude = float(np.linalg.norm([
        _fourier(theirs[c], their_grid, miller)[0] for c in range(3)
    ])) * volume
    their_energy = float(reference.total_energy) / n
    # The helix is the lower state, which is what makes reaching it worth a seed.
    assert their_energy < float(scf.total_energy) - 5e-4

    def helix(x):
        phase = 2 * np.pi * x[..., 0] / n
        return np.stack([np.cos(phase), np.sin(phase),
                         np.zeros_like(phase)], axis=-1)

    amplitudes, energies, cones = [], [], []
    for nbnd in (16, 24):
        result = run_ultracell(
            calculator.system, calculator.pseudos, scf, shape, kgrid,
            nbnd=nbnd, seed_magnetization=helix, conv_thr=1e-10,
            states_conv_thr=1e-8, mixing_beta=0.3, max_iterations=300,
            david=None if nbnd < 24 else 2,
        )
        assert result.converged
        moments = np.asarray(result.cell_moments())
        # Three separate statements about the texture. The pitch is exact
        # because the sector is closed; the lengths are equal and the canting
        # uniform because the same closure makes every cell the image of the
        # first -- and none of the three needs the other two.
        assert np.abs(_pitch(moments) - 360.0 / n).max() < 1e-3, moments
        lengths = np.linalg.norm(moments, axis=1)
        assert np.abs(lengths / lengths[0] - 1.0).max() < 1e-4, lengths
        cone = _cone(moments)
        assert np.abs(cone - cone[0]).max() < 1e-3, cone
        cones.append(abs(float(cone[0])))
        # The cone's axis stays on the reference's direction, which is what
        # says the run never left the closed sector: the remnant is along z
        # and has nothing in the plane the moments turn in.
        net = moments.mean(axis=0)
        assert np.abs(net[:2]).max() < 1e-4 * abs(net[2]) + 1e-6, net
        ours = np.asarray(result.magnetization)
        box = tuple(result.ultracell.grid)
        amplitudes.append(abs(float(np.linalg.norm([
            _fourier(ours[c], box, miller)[0] for c in range(3)
        ])) * volume - their_amplitude) / their_amplitude)
        energies.append(float(result.total_energy) - their_energy)
        jax.clear_caches()

    assert amplitudes == sorted(amplitudes, reverse=True), amplitudes
    assert energies == sorted(energies, reverse=True), energies
    assert cones == sorted(cones, reverse=True), cones
    assert all(e > 0.0 for e in energies), energies
    assert amplitudes[-1] < 1e-2 and energies[-1] < 3e-4, (amplitudes, energies)
    assert cones[-1] < 8.0, cones


def test_kramers_pairs_are_refused_on_a_collinear_cell(tmp_path, pseudo_dir):
    """A collinear channel's partner is the other channel, which is already there.

    Refused before the frozen solve, so no ground state is needed to reach it.
    """
    calculator = _hydrogen(tmp_path, pseudo_dir, (2, 1, 1))
    assert calculator.system.nspin == 2
    with pytest.raises(ValueError, match="kramers_pairs"):
        run_ultracell(calculator.system, calculator.pseudos, None, (2, 1, 1),
                      (1, 1, 1), nbnd=4, kramers_pairs=True)


@pytest.mark.slow
def test_kramers_pairs_keep_the_tiled_null(tmp_path, pseudo_dir):
    """The closed basis is expanded around the same state, so the null holds.

    The reference's own states are in the union, so the lowest Ritz values of
    the reference Hamiltonian there are its eigenvalues and the occupied
    subspace is the one the unit cell has: the tiled density has to come back
    in one iteration, with the unit cell's energy. On the moment along
    ``(1,1,1)/sqrt(3)``, where every component of the density is live.
    """
    shape, kgrid = (2, 1, 1), (2, 2, 2)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    calculator = _noncollinear(tmp_path, pseudo_dir, (1, 1, 1), folded)
    scf = calculator.get_scf(conv_thr=1e-11, nbnd=16)
    assert scf.converged

    result = run_ultracell(
        calculator.system, calculator.pseudos, scf, shape, kgrid,
        nbnd=8, conv_thr=1e-9, states_conv_thr=1e-8, mixing_beta=0.3,
        kramers_pairs=True,
    )
    assert result.converged and result.iterations == 1
    assert np.all(result.kramers_ranks == 16), result.kramers_ranks
    tiled = np.asarray(result.ultracell.tile(jnp.asarray(scf.density)))
    density = np.asarray(result.density)
    assert np.abs(density - tiled).max() / np.abs(tiled).max() < 1e-5
    assert result.total_energy == pytest.approx(scf.total_energy, abs=1e-10)


@pytest.mark.slow
def test_kramers_pairs_remove_the_lean_in_both_frames(tmp_path, pseudo_dir):
    """The four-cell helix on a basis closed under time reversal (``PLAN.md`` P121).

    The old basis prefers the reference's direction. With the reference along
    the helix axis it charges a cone, 10.64 degrees at ``nbnd = 16``
    (:func:`test_a_seeded_helix_keeps_the_pitch_it_was_given`); with the
    reference in the helix plane it converges in 23 iterations to a distorted
    helix, steps of 94, 74 and 84 degrees and a uniform moment of 0.069 along
    the reference, 4.99e-4 Ry per cell above the supercell. That second run is
    the control here, and it has to fail the assertions the closed basis
    passes, or they test nothing.

    With the Kramers partners at ``nbnd = 8``, the same 16 states per folded
    k-point, both frames converge in 10 iterations to the same state: 90
    degrees, no cone, no uniform moment, and 1.28e-6 Ry per cell above the
    supercell, where the old basis at 16 and at 32 bands is 4.37e-4 and
    1.56e-4. Without spin-orbit coupling the closed span is the orbitals times
    both spinors, which is invariant under every global spin rotation, so the
    two frames must agree, and they do to 1e-8 Ry.
    """
    n, kgrid = 4, (1, 2, 2)
    shape = (n, 1, 1)
    folded = tuple(a * b for a, b in zip(shape, kgrid))

    def helix(x):
        phase = 2 * np.pi * x[..., 0] / n
        return np.stack([np.cos(phase), np.sin(phase),
                         np.zeros_like(phase)], axis=-1)

    supercell = _helix_supercell(tmp_path, pseudo_dir, n, kgrid)
    exact = supercell.get_scf(conv_thr=1e-11, nbnd=16, max_iterations=300)
    assert exact.converged
    exact_energy = float(exact.total_energy) / n

    def run(angle1, nbnd, pairs):
        calculator = _noncollinear(tmp_path, pseudo_dir, (1, 1, 1), folded,
                                   angle1=angle1, angle2=0.0,
                                   tag=f"kramers_{angle1:g}")
        scf = calculator.get_scf(conv_thr=1e-11, nbnd=8, max_iterations=300)
        assert scf.converged
        result = run_ultracell(
            calculator.system, calculator.pseudos, scf, shape, kgrid,
            nbnd=nbnd, seed_magnetization=helix, conv_thr=1e-10,
            states_conv_thr=1e-8, mixing_beta=0.3, max_iterations=300,
            kramers_pairs=pairs,
        )
        assert result.converged
        jax.clear_caches()
        return result, np.asarray(result.cell_moments())

    def lean(moments):
        """The worst step error, the worst cone and the uniform moment."""
        return (float(np.abs(_pitch(moments) - 360.0 / n).max()),
                float(np.abs(_cone(moments)).max()),
                float(np.linalg.norm(moments.mean(axis=0))))

    energies = []
    for angle1 in (0.0, 90.0):
        result, moments = run(angle1, 8, True)
        step, cone, net = lean(moments)
        assert step < 1e-3 and cone < 1e-3 and net < 1e-5, (angle1, step, cone, net)
        assert result.iterations <= 12, result.iterations
        above = float(result.total_energy) - exact_energy
        assert 0.0 < above < 3e-6, above
        energies.append(float(result.total_energy))
    assert energies[0] == pytest.approx(energies[1], abs=1e-8)

    # The control: the old basis, twice the bands, the reference in the plane.
    result, moments = run(90.0, 16, False)
    step, cone, net = lean(moments)
    assert step > 1.0 and net > 1e-2, (step, cone, net)
    assert float(result.total_energy) - exact_energy > 1e-4


@pytest.mark.slow
def test_the_reversed_paw_reference_is_the_time_reversed_one(pseudo_dir):
    """The partners of a PAW reference come from the reversed ``becsum`` as well.

    ``kramers_pairs`` takes its partners from a second frozen solve at the
    reversed density and, on an augmented dataset, the reversed ``becsum``,
    whose one-centre terms are what most of a transition metal's moment is.
    The matrix is the exact reference Hamiltonian projected onto whatever span
    that gives, so a wrong reversal would not show as a wrong number: it would
    leave the span unclosed and the lean in place. What it must satisfy is
    ``Theta H[m] Theta^-1 = H[-m]``, so the reversed spectrum over an
    inversion-closed grid is the reference's. On the PAW oxygen texture at
    ``Gamma`` it is, to 1.1e-14 Ry, where reversing the grid and not
    ``becsum`` misses by 2.1e-2 -- that control is asserted too. (On
    spin-orbit PAW nickel under LDA, 40/320 Ry and a 2x2x2 mesh, the same
    check gives 7.0e-14 against a control of 9.3e-3, which is NiBr2's regime;
    it is not a test because its SCF takes three minutes.)
    """
    from defumat.ultracell.kramers import time_reversed
    from defumat.workflows.nscf import fixed_density_states

    calculator = Calculator.from_file(
        Path(__file__).resolve().parents[1] / "data" / "qe" / "o2-paw-texture.in",
        pseudo_dir=pseudo_dir, announce=False, conv_thr=1e-10)
    scf = calculator.get_scf()
    assert scf.converged
    becsum = tuple(scf.becsum)
    nbnd = int(np.asarray(scf.eigenvalues).shape[-1])
    options = dict(nbnd=nbnd, conv_thr=1e-10)
    calculation, _, reference, _ = fixed_density_states(
        calculator.system, calculator.pseudos, jnp.asarray(scf.density),
        becsum=becsum, **options)
    density, reversed_becsum = time_reversed(scf.density, becsum, 4)
    _, _, partners, _ = fixed_density_states(
        calculator.system, calculator.pseudos, density, becsum=reversed_becsum,
        calculation=calculation, **options)
    _, _, control, _ = fixed_density_states(
        calculator.system, calculator.pseudos, density, becsum=becsum,
        calculation=calculation, **options)
    keep = nbnd - 4

    def spectrum(values):
        return np.sort(np.asarray(values).reshape(-1, nbnd)[:, :keep].ravel())

    assert np.abs(spectrum(partners) - spectrum(reference)).max() < 1e-10
    assert np.abs(spectrum(control) - spectrum(reference)).max() > 1e-3


@pytest.mark.slow
def test_fixed_occupations_fill_spinor_bands_one_electron_at_a_time(pseudo_dir):
    """Spin-orbit coupling in an ultracell, and the bug that found itself here.

    **The claim this run exists to support is that spin-orbit coupling costs
    this method nothing.** It is a nonlocal term in the Hamiltonian the frozen
    unit-cell states were diagonalised with, so the ultracell only ever sees
    their eigenvalues and their coefficients; there is no spin-orbit term
    anywhere in ``defumat/ultracell/``. This is the run that says so rather than
    the argument.

    **What it caught on the way is the real reason it is here.** The two
    occupation schemes read their electron count differently -- ``smearing``
    searches for the level whose *weighted sum* reproduces it, while ``fixed``
    counts *bands to fill* -- so only the second has to be told how many
    electrons one band holds. A spinor band holds **one**, where a scalar band
    holds two, and that is the same factor ``for_spin`` takes out of the
    k-point weights, arriving a second time in a place the weights cannot
    reach. The ultracell's occupation was not passing it.

    Nothing in this file could have seen that: **every other noncollinear case
    here uses smearing**, where the argument is not merely right but absent. An
    iodine atom with seven valence electrons and ``occupations = 'fixed'`` is
    the first cell to take the other branch, and it stops with "7.0 electrons
    cannot fill 2-fold bands". Loudly, which is the one merciful thing about it.

    **``N = 1`` and not more, deliberately.** This cell samples the Brillouin
    zone at ``Gamma`` alone, which is right for an isolated atom and wrong for a
    tiling null: at ``N = 2`` the folded set is two k-points, and the tiled
    density of a ``Gamma``-only SCF is not the fixed point two points reproduce
    -- it comes back 2.6 per cent away, which is the k-sampling and not the
    method. The tiling null belongs on a crystal and is asserted on the hydrogen
    cells above; what this cell is for is the spin-orbit regime and the
    occupation count.
    """
    calculator = Calculator.from_file(
        Path(__file__).resolve().parents[1] / "data" / "qe" / "i-atom-soc-lda.in",
        pseudo_dir=pseudo_dir,
    )
    assert calculator.system.noncolin and calculator.system.lspinorb
    assert calculator.system.occupations == "fixed"
    scf = calculator.get_scf()
    assert scf.converged

    result = run_ultracell(
        calculator.system, calculator.pseudos, scf, (1, 1, 1), (1, 1, 1),
        nbnd=16, conv_thr=1e-9, states_conv_thr=1e-8,
        mixing_beta=0.3, max_iterations=60,
    )
    assert result.converged and result.iterations == 1
    assert np.abs(np.asarray(result.delta_v)).max() < 1e-13

    # The electron count is what the missing degeneracy would have broken, and
    # it is asserted on its own: seven spinor bands hold seven electrons.
    element = float(calculator.system.cell.volume) / np.prod(
        result.ultracell.cell_grid)
    charge = float(np.asarray(result.density)[0].sum() * element)
    nelec = float(Calculation(calculator.system, calculator.pseudos).nelec)
    assert nelec == pytest.approx(7.0)
    assert charge == pytest.approx(nelec, rel=1e-7)

    # ...and the moment survives the round trip, which the charge alone cannot
    # say: a spin-orbit ground state has a direction, and an ultracell that
    # filled the wrong bands would keep the charge and lose it.
    #
    # **The tolerance is the band truncation and nothing else.** ``dV`` here is
    # 4e-15, so the only difference between the two is that one expands the
    # state in 16 frozen bands and the other does not expand it at all: the
    # moment comes back 7.4e-6 away on 0.2273, which is 3.3e-5 relative and sits
    # exactly on the ``nbnd = 16`` rung of the ladders above. Tightening this
    # past the truncation would be asserting something the method does not claim.
    moment = result.cell_moments()[0]
    reference = _total_moment(scf.density, float(calculator.system.cell.volume))
    assert moment == pytest.approx(reference, abs=5e-5)
    assert abs(moment[2]) > 0.2 and np.abs(moment[:2]).max() < 1e-5
