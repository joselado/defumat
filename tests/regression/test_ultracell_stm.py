"""P89: what a modulation looks like to a tip, against what a supercell says.

An ultracell (P88) converges a density over ``N`` unit cells in the unit cell's
own frozen states; P65 and P66 are the Tersoff-Hamann image and the vertical
tunnelling transmission a tip measures above a surface. This file checks the
join, and the reference is the same one P88 uses: an ``N``-cell supercell run
through this package's own SCF under the same applied potential, which shares
the unit-cell machinery and none of the ultracell assembly.

The checks, in the order they are worth reading:

* **the relabelling itself**, which is the one thing here that is new. An
  ultracell state is claimed to be a single coefficient vector on the
  ultracell's own sphere, so sampling it with
  :func:`~defumat.basis.sample.sample_wavefunctions` must reproduce the box
  transform :mod:`defumat.ultracell.density` builds the density from. It is run
  on a **modulated** state, because the sign of ``q`` and the value of ``k0`` in
  the ultracell's coordinates are invisible in any state built from one ``Q``;
* the image against the unit cell's, tiled -- on the **folded** k-set, which is
  the one an ultracell actually integrates over;
* the sum rule against ``compute_dos``, which closes inside the package and
  shares only the smeared delta;
* the image against a real four-atom supercell's image, which is P88's own
  claim read on the observable instead of on the density;
* the transmission with the exit region widened to the whole ultracell, which
  must be the image exactly and with no factor -- P66's own diagnostic, and here
  it is what says the two normalisations agree;
* a magnetic tip on a spin density wave, which sees the wave itself over the
  eight cells where an unpolarized tip sees it squared, at twice the
  wavevector.
"""

import tempfile
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat import Calculator
from defumat.basis.builder import build_basis
from defumat.basis.sample import sample_wavefunctions
from defumat.scf.driver import Calculation, run_scf
from defumat.ultracell import run_ultracell, with_external_potential
from defumat.workflows.stm import run_stm
from defumat.workflows.transport import run_vertical_transport
from defumat.workflows.ultracell import (
    run_ultracell_stm,
    run_ultracell_transport,
)

pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """Bound the peak: this file runs a unit cell, an ultracell and a supercell.

    Cells that share no shape each compile the whole SCF stack afresh and XLA
    keeps every executable for the life of the process, so the peak grows
    monotonically through the file rather than being any one test's.
    ``CLAUDE.md`` asks for this on any file over about three distinct cells.
    """
    yield
    jax.clear_caches()


#: The applied modulation, in Ry. P88's own, for the same reason: small enough
#: that the frozen basis is a good one and large enough that what it induces is
#: far above the two SCFs' convergence floors.
AMPLITUDE = 0.05

#: The tip plane, and a window covering the whole valence band. **Both window
#: edges are far from every state on purpose**: an edge that lands on a band is
#: a step of 1 against 0.56 for a state that moves by the eigenvalue difference
#: between two diagonalisations, which reads as a 4 per cent error in the image
#: and is a property of the test rather than of the code.
PLANE = dict(height=0.35, axis=2)
WINDOW = dict(bias=-1.5, width=0.01)

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

NONCOLLINEAR = """&control
 calculation='scf'
/
&system
 ibrav=2, celldm(1)=10.20, nat=2, ntyp=1, ecutwfc=12.0,
 nosym=.true., noinv=.true.,
 noncolin=.true., starting_magnetization(1)=0.2, angle1(1)=60.0,
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
 {k0} {k1} {k2} 0 0 0
"""

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
 {k0} {k1} {k2} 0 0 0
"""


#: One directory for every input this file writes, so that the cache below is
#: keyed on the physics rather than on which test asked first: ``tmp_path`` is
#: per-test, and with it in the key every cell would be converged again for
#: every test that wants it.
_WORK = Path(tempfile.mkdtemp(prefix="defumat-p89-"))


def _calculator(pseudo_dir, name, template, grid) -> Calculator:
    path = _WORK / name
    path.write_text(template.format(k0=grid[0], k1=grid[1], k2=grid[2]))
    return Calculator.from_file(path, pseudo_dir=pseudo_dir)


#: **``maxsize=2`` and never ``None``.** Two is what a comparison between two
#: cells needs and is the largest that is not a leak: what is held is the
#: wavefunctions.
@lru_cache(maxsize=2)
def _converged(pseudos, grid, nbnd=12, magnetic=False):
    """The unit cell on a folded grid, converged tight enough to be a basis."""
    name = f"si{'_mag' if magnetic else ''}_{grid[0]}{grid[1]}{grid[2]}.in"
    calculator = _calculator(Path(pseudos), name,
                             MAGNETIC if magnetic else SILICON, grid)
    scf = calculator.get_scf(conv_thr=1e-10 if magnetic else 1e-12, nbnd=nbnd)
    assert scf.converged
    return calculator, scf


def _supercell(pseudo_dir, calculator, shape, kgrid) -> Calculator:
    """The same crystal as a real ``shape`` supercell, atoms and all.

    ``a^s_i = n_i a_i`` exactly, so a plane spanning one supercell is the plane
    spanning ``n_i`` unit cells, point for point, and the two images can be
    compared without interpolating either.
    """
    cell = calculator.system.cell
    vectors = np.asarray(cell.at_alat) * np.asarray(shape)[:, None]
    tau = np.asarray(calculator.system.structure.positions_alat(cell))
    shifts = np.stack(
        np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    positions = np.concatenate(
        [tau + shift @ np.asarray(cell.at_alat) for shift in shifts])
    rows = "\n".join(f" {v[0]:.12f} {v[1]:.12f} {v[2]:.12f}" for v in vectors)
    atoms = "\n".join(f" Si {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}" for p in positions)
    path = _WORK / "si_supercell.in"
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


def _midgap(scf) -> float:
    return 0.5 * (float(scf.homo) + float(scf.lumo))


# -- the relabelling, which is the only new thing here -----------------------


def test_an_ultracell_state_is_one_plane_wave_vector(pseudo_dir):
    """``Psi_j`` sampled from its own sphere is ``Psi_j`` off the ultracell box.

    The claim the phase rests on: the union over ``Q`` of the ``N`` unit-cell
    spheres, relabelled by ``h = n G + q``, is the ultracell's own sphere at
    ``k0``, so an ultracell state is one coefficient vector on it. Two routes to
    the same function then have to agree -- the scatter-and-transform
    :mod:`defumat.ultracell.density` builds the density with, and the direct
    plane-wave sum :func:`~defumat.basis.sample.sample_wavefunctions` takes.

    **On a modulated state, not an unmodulated one.** A state built from a
    single ``Q`` differs from the right answer by ``e^{-2iQ.r}`` if the sign of
    ``q`` is wrong and by nothing at all in modulus, so every null in this file
    passes with the sign reversed; a state that mixes ``Q`` does not.
    """
    shape, kgrid = (2, 1, 1), (1, 2, 2)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    calculator, scf = _converged(str(pseudo_dir), folded)
    result = run_ultracell(
        calculator.system, calculator.pseudos, scf, shape, kgrid, nbnd=12,
        external=_modulation(shape), conv_thr=1e-10, states_conv_thr=1e-10)
    assert result.converged

    states = result.states
    ik0 = 1
    block = np.asarray(states.block(0, ik0))
    mask = states.mask(ik0)

    # orthonormal, which is what makes the Gram matrix of P66 meaningful
    gram = block.conj() @ block.T
    assert np.abs(gram - np.eye(states.nstate)).max() < 1e-12

    grid = states.ultracell.grid
    points = np.stack(
        np.meshgrid(*[np.arange(m) / m for m in grid], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    volume = states.ultracell.volume(states.cell)
    sampled = sample_wavefunctions(block[:4], states.miller(ik0),
                                   states.kcrystal[ik0], points, volume,
                                   mask=mask)

    flat_index = np.asarray(states.box_index)[ik0].reshape(-1)
    npoints = int(np.prod(grid))
    phase = np.exp(2.0j * np.pi * points @ states.kcrystal[ik0])
    for j in range(4):
        scattered = np.zeros(npoints, dtype=complex)
        np.add.at(scattered, flat_index, block[j].reshape(-1))
        box = (np.fft.ifftn(scattered.reshape(grid)) * npoints).reshape(-1)
        box = box / np.sqrt(volume) * phase
        assert np.abs(sampled[j] - box).max() < 1e-14 * np.abs(box).max() + 1e-15
        # and it is normalised over the **ultracell**, which is the convention
        # the transmission's sampling and the density's weights share
        assert float((np.abs(sampled[j]) ** 2).mean() * volume) == pytest.approx(
            1.0, rel=1e-10)


# -- the image ---------------------------------------------------------------


@pytest.mark.parametrize("shape,kgrid", [((1, 1, 1), (2, 2, 2)),
                                         ((2, 1, 1), (1, 2, 2))])
def test_an_unmodulated_image_is_the_tiled_unit_cell_image(
        shape, kgrid, pseudo_dir):
    """With nothing applied, the image is the unit cell's, ``N`` times over.

    **Against the unit cell on the folded grid**, which is the k-set an
    ultracell integrates over: an image is a sum over states at the tip energy
    and a different Brillouin-zone sampling is a different sum, so comparing
    against the ``k0`` grid instead would be comparing two right answers to
    different questions -- a factor of two on this cell, and nothing to do with
    the assembly.
    """
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    calculator, scf = _converged(str(pseudo_dir), folded)
    energy = _midgap(scf)
    plain = run_stm(calculator.system, calculator.pseudos, scf, shape=(12, 12),
                    energy=energy, **PLANE, **WINDOW)
    result = run_ultracell(
        calculator.system, calculator.pseudos, scf, shape, kgrid, nbnd=12,
        conv_thr=1e-10, states_conv_thr=1e-12)
    assert result.converged
    image = run_ultracell_stm(
        calculator.system, calculator.pseudos, result,
        shape=(12 * shape[0], 12), energy=energy, **PLANE, **WINDOW)

    tiled = np.concatenate([plain.values] * shape[0], axis=0)
    assert np.abs(image.values - tiled).max() / np.abs(tiled).max() < 1e-6

    # the same statement on the integral, which is the electron count in the
    # window and is per unit cell on both sides.
    #
    # **The floor is the ultracell's own ``conv_thr`` and not round-off.** The
    # two sides are built from different wavefunctions -- the SCF's own, and
    # the fixed-density solve's at the folded points -- and the ultracell's
    # ``dV`` is converged to 1e-10 rather than to zero, so the states it
    # diagonalises are not exactly the unit cell's. Measured at 1.0e-8 on this
    # cell, which is what the tolerance is set against.
    assert image.integral == pytest.approx(plain.integral, rel=1e-7)

    # and the same null in the other mode, which is a different path through
    # the same field: the scan walks the tip outwards and inverts for the
    # height at the set-point, so it samples at points the plane does not
    # contain and takes its reach from the *unit* cell rather than from the
    # ultracell.
    #
    # **Half the pixels come back nan and that is the physics, not a failure.**
    # This plane is a cut through bulk silicon rather than a surface above one,
    # so withdrawing the tip walks it towards the next atomic layer and the
    # density rises again; the set-point is only reached above the atoms, which
    # is what the warning says. What is asserted is therefore that both routes
    # refuse the *same* pixels and agree on the rest -- and that the rest is a
    # stated fraction, since a scan that came back all nan would satisfy any
    # comparison of what was left.
    current = 3.0 * float(np.median(np.asarray(plain.values)))
    scan = dict(mode="constant-current", current=current, heights=(0.0, 5.0),
                nheights=40)
    with pytest.warns(UserWarning, match="never cross the set-point"):
        one = run_stm(calculator.system, calculator.pseudos, scf,
                      shape=(12, 12), energy=energy,
                      **PLANE, **WINDOW, **scan)
    with pytest.warns(UserWarning, match="never cross the set-point"):
        many = run_ultracell_stm(
            calculator.system, calculator.pseudos, result,
            shape=(12 * shape[0], 12), energy=energy,
            **PLANE, **WINDOW, **scan)
    reference = np.concatenate([np.asarray(one.heights)] * shape[0], axis=0)
    ours = np.asarray(many.heights)
    crossed = np.isfinite(reference)
    assert crossed.mean() == pytest.approx(0.5, abs=0.05)   # 72 of 144 here
    assert np.array_equal(crossed, np.isfinite(ours))
    assert np.ptp(reference[crossed]) > 1.0     # 1.65 bohr of corrugation
    # measured 5.1e-7 bohr, on a corrugation of 1.65
    assert np.abs(ours[crossed] - reference[crossed]).max() < 1e-5


def test_the_image_integrates_to_the_density_of_states(pseudo_dir):
    """``int rho_STM d3r = D(E)`` per unit cell, against ``compute_dos``.

    The check that would catch the factor of ``N`` the two nulls above cannot
    see on their own: the weights are divided by ``N`` and the density by the
    **unit cell's** volume, and only their product is constrained here.
    """
    from defumat.workflows.dos import compute_dos

    shape, kgrid = (2, 1, 1), (1, 2, 2)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    calculator, scf = _converged(str(pseudo_dir), folded)
    result = run_ultracell(
        calculator.system, calculator.pseudos, scf, shape, kgrid, nbnd=12,
        external=_modulation(shape), conv_thr=1e-10, states_conv_thr=1e-10)
    assert result.converged

    energy = float(scf.homo) - 0.05
    image = run_ultracell_stm(calculator.system, calculator.pseudos, result,
                              shape=(8, 8), energy=energy, width=0.02,
                              **PLANE)
    states = result.states
    dos = compute_dos(np.asarray(states.eigenvalues),
                      np.asarray(states.weights), np.array([energy]),
                      "gaussian", degauss=0.02)
    per_cell = float(np.asarray(dos.dos).ravel()[0]) / states.cells
    assert per_cell > 0.0
    assert image.integral == pytest.approx(per_cell, rel=1e-10)



def test_the_image_converges_to_the_supercell(pseudo_dir):
    """The number for the phase: the image against a real supercell's image.

    P88's own claim, read on the observable instead of on the density. What is
    checked is not that the two agree at some ``nbnd`` -- they do not, and
    should not -- but that **the disagreement falls as ``nbnd`` grows**, which
    is the whole content of "a variational truncation to ``nbnd`` bands per
    folded k-point".

    Both sides are given the **same** explicit window rather than each its own
    band edge: the two HOMOs differ by the truncation, and an image anchored to
    each side's own edge compares two different measurements.
    """
    shape, kgrid = (2, 1, 1), (1, 2, 2)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    calculator, scf = _converged(str(pseudo_dir), folded)
    energy = _midgap(scf)
    geometry = dict(shape=(24, 12), energy=energy, **PLANE, **WINDOW)

    supercell = _supercell(pseudo_dir, calculator, shape, kgrid)
    grid = build_basis(supercell.system).dense.grid
    coordinates = np.stack(
        np.meshgrid(*[np.arange(m) / m for m in grid], indexing="ij"), axis=-1)
    calculation = with_external_potential(
        Calculation(supercell.system, supercell.pseudos),
        jnp.asarray(AMPLITUDE * np.cos(2 * np.pi * coordinates[..., 0])))
    exact = run_scf(supercell.system, supercell.pseudos,
                    calculation=calculation, conv_thr=1e-11, nbnd=12)
    assert exact.converged
    reference = run_stm(supercell.system, supercell.pseudos, exact, **geometry)

    flat = run_stm(calculator.system, calculator.pseudos, scf,
                   shape=(12, 12), energy=energy, **PLANE, **WINDOW)
    tiled = np.concatenate([flat.values] * shape[0], axis=0)
    induced_exact = np.asarray(reference.values) - tiled
    # the modulation is a corrugation of 0.6 per cent of the image, so an
    # assertion about it is an assertion about the modulation and not about the
    # crystal underneath it
    assert np.abs(induced_exact).max() / np.abs(reference.values).max() > 1e-3

    errors = {}
    for nbnd, david in ((12, None), (24, None), (48, 2)):
        result = run_ultracell(
            calculator.system, calculator.pseudos, scf, shape, kgrid,
            nbnd=nbnd, external=_modulation(shape), conv_thr=1e-10,
            states_conv_thr=1e-10, david=david)
        assert result.converged
        image = run_ultracell_stm(calculator.system, calculator.pseudos,
                                  result, **geometry)
        induced = np.asarray(image.values) - tiled
        errors[nbnd] = float(np.abs(induced - induced_exact).max()
                             / np.abs(induced_exact).max())

    assert errors[48] < errors[24] < errors[12]
    assert errors[48] < 0.05


# -- the transmission --------------------------------------------------------


def test_the_whole_cell_transmission_is_the_image(pseudo_dir):
    """``S_k -> delta_nn'`` by orthonormality, and the ultracell image comes back.

    P66's own diagnostic on an ultracell, and here it is the one check that
    holds the transmission's conventions against the image's: the image carries
    its ``N`` in the weights and the transmission carries it in the volume the
    states are normalised over, so if either is wrong the two differ by it.

    **On a modulated state**, for the reason the first test in this file gives:
    the transmission samples ``Psi`` with the ultracell Miller indices and
    ``k0`` directly where the image goes through the box transform, so an
    unmodulated ultracell would leave the two agreeing for the wrong reason.

    Bulk silicon is not a slab, so the run warns that the atoms do not lie
    between the two planes. That is the right warning and this is the wrong
    geometry for it: what is being checked is an identity between two
    contractions, which holds wherever the planes are put.
    """
    shape, kgrid = (2, 1, 1), (1, 2, 1)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    calculator, scf = _converged(str(pseudo_dir), folded)
    result = run_ultracell(
        calculator.system, calculator.pseudos, scf, shape, kgrid, nbnd=12,
        external=_modulation(shape), conv_thr=1e-10, states_conv_thr=1e-10)
    assert result.converged

    energy = float(scf.homo) - 0.05
    geometry = dict(shape=(8, 6), **PLANE)
    ours = run_ultracell_transport(
        calculator.system, calculator.pseudos, result, exit_height=0.05,
        exit_axis=2, energies=energy, broadening=0.02, exit_region="volume",
        **geometry)
    image = run_ultracell_stm(calculator.system, calculator.pseudos, result,
                              energy=energy, width=0.02, **geometry)
    reference = np.asarray(image.values)
    assert reference.max() > 0.0
    assert np.abs(np.asarray(ours.image) - reference).max() / reference.max() < 1e-11


def test_the_exit_plane_transmission_is_the_unit_cell_s_tiled(pseudo_dir):
    """The plane path, which the whole-cell identity above does not reach.

    ``exit_region="volume"`` goes through :func:`~defumat.transport.substrate.volume_overlap`,
    which is one matrix product and knows nothing about the cell; the substrate
    the feature is for is :func:`~defumat.transport.substrate.exit_overlap`,
    which groups plane waves by their **in-plane** Miller index and scales by
    the cell's own area over its volume. Handing it the ultracell's indices and
    the ultracell's cell is the whole of what P89 does to it, so it needs a
    check of its own.

    With nothing applied the ultracell states are the folded unit-cell states,
    and two of them at different ``Q`` carry different lateral momentum, so the
    exit integral cannot mix them: the map has to come back as the unit cell's,
    tiled. The area and the volume both double with the cell here, which is why
    their **ratio** surviving is the thing worth asserting -- an ultracell cell
    that had kept the unit cell's volume would pass a check on either one alone.
    """
    shape, kgrid = (2, 1, 1), (1, 2, 1)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    calculator, scf = _converged(str(pseudo_dir), folded)
    energy = float(scf.homo) - 0.05
    geometry = dict(exit_height=0.05, exit_axis=2, energies=energy,
                    broadening=0.02, **PLANE)

    plain = run_vertical_transport(calculator.system, calculator.pseudos, scf,
                                   shape=(8, 6), **geometry)
    result = run_ultracell(calculator.system, calculator.pseudos, scf, shape,
                           kgrid, nbnd=12, conv_thr=1e-10, states_conv_thr=1e-12)
    assert result.converged
    ours = run_ultracell_transport(calculator.system, calculator.pseudos,
                                   result, shape=(16, 6), **geometry)

    tiled = np.concatenate([np.asarray(plain.values)] * shape[0], axis=0)
    assert tiled.max() > 0.0
    # **The floor here is the broadening and not either threshold**, which is
    # what separates a transmission from the image above. The two sides
    # diagonalise the same Hamiltonian in two different bases -- the unit
    # cell's sphere at the folded points, and the frozen envelope basis on the
    # ultracell's own box -- and their levels come out a median 1.2e-8 Ry
    # apart, which is the ultracell's density differing from the tiled one by
    # 2.1e-8 of 0.107. A transmission divides that by the broadening, since
    # what it is built from is ``1/(E - e + i eta)``: measured 1.2e-6 at
    # ``eta = 0.02``, 6.2e-6 at 0.01 and 4.6e-7 at 0.04, while tightening
    # ``conv_thr`` or ``states_conv_thr`` by four orders each moves it in the
    # third digit. The image's own null on the same cell is 5.8e-8, because a
    # density is not divided by anything.
    assert np.abs(np.asarray(ours.values) - tiled).max() / tiled.max() < 1e-5


def test_the_transmission_converges_to_the_supercell(pseudo_dir):
    """And the modulated plane path, against a real supercell's transmission.

    The image's ladder one quantity along. It is a weaker statement than that
    one -- two rungs rather than three -- because what is being asked of it is
    different: the image has already shown that the truncation converges, and
    what is open here is whether the exit integral over an ultracell whose
    states genuinely mix ``Q`` is the supercell's.
    """
    shape, kgrid = (2, 1, 1), (1, 2, 1)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    calculator, scf = _converged(str(pseudo_dir), folded)
    energy = float(scf.homo) - 0.05
    geometry = dict(exit_height=0.05, exit_axis=2, energies=energy,
                    broadening=0.02, shape=(16, 6), **PLANE)

    supercell = _supercell(pseudo_dir, calculator, shape, kgrid)
    grid = build_basis(supercell.system).dense.grid
    coordinates = np.stack(
        np.meshgrid(*[np.arange(m) / m for m in grid], indexing="ij"), axis=-1)
    calculation = with_external_potential(
        Calculation(supercell.system, supercell.pseudos),
        jnp.asarray(AMPLITUDE * np.cos(2 * np.pi * coordinates[..., 0])))
    exact = run_scf(supercell.system, supercell.pseudos,
                    calculation=calculation, conv_thr=1e-11, nbnd=12)
    assert exact.converged
    reference = np.asarray(run_vertical_transport(
        supercell.system, supercell.pseudos, exact, **geometry).values)

    errors = {}
    for nbnd in (12, 24):
        result = run_ultracell(
            calculator.system, calculator.pseudos, scf, shape, kgrid,
            nbnd=nbnd, external=_modulation(shape), conv_thr=1e-10,
            states_conv_thr=1e-10)
        assert result.converged
        ours = np.asarray(run_ultracell_transport(
            calculator.system, calculator.pseudos, result, **geometry).values)
        errors[nbnd] = float(np.abs(ours - reference).max() / reference.max())

    assert errors[24] < errors[12]
    assert errors[24] < 0.05


# -- spin --------------------------------------------------------------------


def test_a_magnetic_tip_sees_a_spin_density_wave_a_plain_one_does_not(
        pseudo_dir):
    """The image a spin density wave is actually measured with.

    A modulated ``B(r)`` on a cell whose own moment is zero drives equal and
    opposite moments in the two cells (P88), and a **magnetic tip** is what sees
    it: the two images here are the same calculation with one argument changed.

    **What the charge image is, precisely**, because "it is flat" is the
    plausible wrong version. A collinear system is invariant under flipping every
    spin together with the sign of ``B``, so the charge cannot respond at *odd*
    order in ``B`` -- it responds at twice the wavevector, following ``|m|^2``,
    and on a longer ultracell that response is large rather than absent (37 per
    cent of its mean, cell to cell, against 82 for one spin channel, on the
    eight-cell cell of ``notebooks/45``). What is zero is its **odd** part, which
    is what the contrast below measures: the difference between the two halves of
    the ultracell, where the magnetization changes sign and the charge does not.
    A tunnelling density of states at one energy is a far more sensitive quantity
    than a density, and reading "the charge barely moves" off the density and
    expecting it of the image is the mistake this docstring exists to stop.
    """
    shape, kgrid = (2, 1, 1), (1, 2, 2)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    calculator, scf = _converged(str(pseudo_dir), folded, magnetic=True)
    assert abs(scf.magnetization) < 1e-5
    result = run_ultracell(
        calculator.system, calculator.pseudos, scf, shape, kgrid, nbnd=12,
        magnetic_field=_modulation(shape), conv_thr=1e-10,
        states_conv_thr=1e-9)
    assert result.converged

    geometry = dict(shape=(16, 6), **PLANE, **WINDOW)
    charge = run_ultracell_stm(calculator.system, calculator.pseudos, result,
                               **geometry)
    up = run_ultracell_stm(calculator.system, calculator.pseudos, result,
                           spin="up", **geometry)

    def contrast(values):
        """How much of the image is the modulation, cell against cell."""
        half = values.shape[0] // 2
        first, second = values[:half], values[half:]
        return float(np.abs(first - second).max() / np.abs(values).max())

    assert contrast(np.asarray(up.values)) > 10 * contrast(np.asarray(charge.values))


def test_a_spinor_state_keeps_its_two_components_in_the_right_halves(pseudo_dir):
    """The spinor layout of an ultracell state, which only the transmission reads.

    An ultracell state of a noncollinear run is one vector whose two halves are
    the up and down components, which is how every wavefunction in this package
    is stored and what the exit-plane Gram matrix and the tip sampler both
    expect. Orthonormality cannot see that: an interleaved vector is just as
    orthonormal. What can is the **density**, which is built by the other route
    entirely -- the scatter into the ultracell box, one component at a time --
    so squaring the halves and comparing is a check of the layout and of
    nothing else.
    """
    calculator = _calculator(Path(str(pseudo_dir)), "si_nc.in", NONCOLLINEAR,
                             (2, 2, 2))
    scf = calculator.get_scf(conv_thr=1e-10, nbnd=12)
    assert scf.converged
    shape, kgrid = (2, 1, 1), (1, 2, 2)
    result = run_ultracell(calculator.system, calculator.pseudos, scf, shape,
                           kgrid, nbnd=12, conv_thr=1e-9, states_conv_thr=1e-9)
    states = result.states
    assert int(states.npol) == 2

    ik0, nstate = 0, 4
    block = np.asarray(states.block(0, ik0))[:nstate]
    grid = states.ultracell.grid
    points = np.stack(
        np.meshgrid(*[np.arange(m) / m for m in grid], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    volume = states.ultracell.volume(states.cell)
    sampled = sample_wavefunctions(
        block.reshape((nstate, 2, states.cells * states.npwx)),
        states.miller(ik0), states.kcrystal[ik0], points, volume,
        mask=states.mask(ik0))
    charge = (np.abs(sampled) ** 2).sum(axis=1)

    # the same four states through the other route, with unit weights. Only
    # the *shape* is compared: every prefactor divides out of a field over its
    # own mean, and what is being checked is which coefficients went where.
    weights = np.zeros((1, states.nk0, states.nstate))
    weights[0, ik0, :nstate] = 1.0
    reference = np.asarray(states.density(weights, nspin_mag=4))[0].reshape(-1)
    ours = charge.sum(axis=0)
    assert np.abs(ours / ours.mean() - reference / reference.mean()).max() < 1e-10


# -- the refusals ------------------------------------------------------------


def test_an_ultracell_result_is_refused_by_name(pseudo_dir):
    """The three unit-cell entry points say what to call instead.

    Without this the failure is an ``AttributeError`` on ``eigenvalues_by_spin``
    three lines into ``run_stm``, which says nothing about why -- and the
    quantity does exist, one function away.
    """
    shape, kgrid = (1, 1, 1), (2, 2, 2)
    calculator, scf = _converged(str(pseudo_dir), (2, 2, 2))
    result = run_ultracell(calculator.system, calculator.pseudos, scf, shape,
                           kgrid, nbnd=12, conv_thr=1e-10)
    for entry, kwargs in (
        (run_stm, dict(height=0.35)),
        (run_vertical_transport, dict(exit_height=0.05, height=0.35)),
    ):
        with pytest.raises(NotImplementedError, match="UltracellResult"):
            entry(calculator.system, calculator.pseudos, result, **kwargs)

    # and the ultracell's own entry points refuse what they cannot sum
    with pytest.raises(NotImplementedError, match="deep along the stacking axis"):
        run_ultracell_transport(
            calculator.system, calculator.pseudos,
            run_ultracell(calculator.system, calculator.pseudos, scf,
                          (1, 1, 2), (1, 1, 1), nbnd=12, conv_thr=1e-10),
            exit_height=0.05, height=0.35, shape=(4, 4))
    with pytest.raises(NotImplementedError, match="divisions along the stacking"):
        run_ultracell_transport(calculator.system, calculator.pseudos, result,
                                exit_height=0.05, height=0.35, shape=(4, 4))

    # a run that dropped its states says how to get them back
    dropped = run_ultracell(calculator.system, calculator.pseudos, scf, shape,
                            kgrid, nbnd=12, conv_thr=1e-10, keep_states=False)
    assert dropped.states is None
    with pytest.raises(ValueError, match="keep_states"):
        run_ultracell_stm(calculator.system, calculator.pseudos, dropped,
                          height=0.35)
