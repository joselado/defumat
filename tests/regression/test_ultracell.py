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

import jax.numpy as jnp

from defumat import Calculator
from defumat.basis.builder import build_basis
from defumat.scf.driver import Calculation, run_scf
from defumat.ultracell import run_ultracell, with_external_potential
from defumat.units import RY_TO_EV

pytestmark = pytest.mark.regression

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
    ({"nspin": 2}, "nspin = 2"),
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


def test_a_polarized_run_is_refused_before_it_is_paid_for(tmp_path, pseudo_dir):
    """``nspin = 2`` is stage 3, and it says so *before* the expensive step.

    The frozen-state diagonalisation over the folded k-set is the whole cost of
    the method, so a refusal that fires after it would charge a user the full
    price of a calculation they are not going to get.
    """
    path = tmp_path / "si_mag.in"
    path.write_text(MAGNETIC)
    calculator = Calculator.from_file(path, pseudo_dir=pseudo_dir)
    scf = calculator.get_scf(conv_thr=1e-6, nbnd=8)
    with pytest.raises(NotImplementedError, match="nspin = 2"):
        run_ultracell(
            calculator.system, calculator.pseudos, scf, (2, 1, 1), (1, 2, 2),
            nbnd=8, conv_thr=1e-8,
        )


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
