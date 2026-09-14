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
 ibrav=2, celldm(1)=10.20, nat=2, ntyp=1, ecutwfc=12.0,
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


def _silicon(tmp_path, pseudo_dir, kgrid) -> Calculator:
    """The two-atom cell on an unshifted, unreduced ``kgrid``.

    Unreduced because a modulation breaks the crystal's point group and the
    ultracell's own group is not written, so the whole phase runs ``nosym``;
    unshifted because the folded set ``k0 + Q`` is a Monkhorst-Pack grid only
    if its origin is.
    """
    path = tmp_path / "si.in"
    path.write_text(SILICON.format(k0=kgrid[0], k1=kgrid[1], k2=kgrid[2]))
    return Calculator.from_file(path, pseudo_dir=pseudo_dir)


def _supercell(tmp_path, pseudo_dir, calculator, shape, kgrid) -> Calculator:
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
    path = tmp_path / "si_supercell.in"
    path.write_text(f"""&control
 calculation='scf'
/
&system
 ibrav=0, celldm(1)={float(cell.alat):.10f}, nat={len(positions)}, ntyp=1,
 ecutwfc=12.0, nosym=.true., noinv=.true.
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
    # **The top rung carries its own ``david`` and that is not a tuning knob.**
    # The Davidson subspace is ``david * nbnd`` and it is not capped against the
    # size of the space (``OPEN.md`` Part VI item 1): at ``nbnd = 48`` the
    # default asks for 192 vectors where these k-points hold between 169 and
    # 190 plane waves, seven of eight overlaps then go non-finite, and what
    # surfaces three layers up is "the ultracell did not converge". Two is what
    # the measurement in ``PLAN.md`` P88 was taken at. It changes how the frozen
    # states are *found*, not what they are once converged, so the ladder below
    # is the same claim either way.
    for nbnd, david in ((12, None), (24, None), (48, 2)):
        result = run_ultracell(
            calculator.system, calculator.pseudos, scf, shape, kgrid,
            nbnd=nbnd, external=_modulation(shape), conv_thr=1e-10,
            states_conv_thr=1e-10, david=david,
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


# -- the refusals ------------------------------------------------------------


# -- stage 3a: two spin channels ---------------------------------------------

#: A hydrogen simple-cubic lattice that is **partly** polarized, which is what
#: this needs and what neither end of the range gives. At ``a = 5.0`` the same
#: cell has a moment of 0.027 and takes 56 iterations -- it sits on the Stoner
#: threshold -- and at ``a = 6.0`` it is 0.9997, a saturated atom whose moment
#: cannot grow and whose ``|zeta| = 1`` is the clamp-tangent trap ``CLAUDE.md``
#: names. At 5.5 with a 0.8 seed it is 0.62 in six iterations.
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

    errors = []
    for nbnd in (12, 24, 40):
        result = run_ultracell(
            calculator.system, calculator.pseudos, scf, (1, 1, 1), (2, 2, 2),
            nbnd=nbnd, magnetic_field=lambda x: np.full(x.shape[:-1], field),
            conv_thr=1e-11, states_conv_thr=1e-8, david=2,
        )
        assert result.converged
        moment = float(result.cell_moments().sum())
        errors.append(abs(moment - reference.magnetization)
                      / abs(reference.magnetization))

    # **Monotone, and the sign is what a flipped field would break.** A moment
    # that came back at -0.71 would fail the first assertion; one at half the
    # size would fail every rung of the ladder and would not improve with nbnd.
    assert errors == sorted(errors, reverse=True), errors
    assert errors[-1] < 2e-3


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
            states_conv_thr=1e-8, david=2,
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
    ({"nspin": 4}, "nspin = 4"),
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
        _as_field(None, lambda x: np.zeros(x.shape[:-1]), ultracell, 1)


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
                                 ultracell, 2))
    assert field[0] == pytest.approx(np.full(ultracell.grid, -0.3), abs=1e-14)
    assert field[1] == pytest.approx(np.full(ultracell.grid, +0.3), abs=1e-14)

    # and a scalar potential is felt in full by both, which is the other rule
    # in the same function and the one that would otherwise be half of it.
    both = np.asarray(_as_field(lambda x: np.full(x.shape[:-1], 0.1), None,
                                ultracell, 2))
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
    path.write_text(SILICON.format(k0=2, k1=2, k2=2).replace(
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
