"""STM images against ``pp.x``, against a sum rule, and against the physics.

``PLAN.md`` P65. Elk's task 162 (``wfplot.f90``) and QE's ``PP/src/stm.f90``.
The machinery -- the sampler, the plane, the weights, the spin projection -- is
checked on objects in ``tests/unit/test_stm_machinery.py``; what is here needs
a converged cell.

Four kinds of check, in the order they are worth reading:

* ``pp.x`` on the same cell and the same pseudopotential, whose ``plot_num = 5``
  at ``sample_bias = 0`` is exactly the zero-bias image times ``degauss``;
* the sum rule ``int rho_STM d3r = D(E_F)``, which shares nothing with the
  reference and closes inside the package;
* an antiferromagnet whose two spin projections are each other's mirror while
  its charge image is flat, and a noncollinear chain where turning the tip
  selects a different sublattice -- the spin-polarized image, which neither
  ``pp.x`` nor Elk's task 162 computes;
* graphite, whose STM image famously shows one of its two sublattices, and the
  exponential decay of the tunnelling density into the vacuum.
"""

from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
import pytest

from defumat import Calculator

pytestmark = pytest.mark.slow

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """Every cell here compiles its own SCF stack and XLA keeps them all.

    ``CLAUDE.md``'s rule for a file that sweeps cells: the results stay cached
    below, only the executables are dropped.
    """
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _converged(case: str):
    calculator = Calculator.from_file(CASES / f"{case}.in", pseudo_dir=PSEUDO)
    calculator.get_scf()
    return calculator


def _read_filplot(path: Path) -> np.ndarray:
    """``Modules/plot_io.f90``'s text dump of the dense grid, as ``(n1, n2, n3)``.

    The header is fixed-shape apart from the species and atom tables, whose
    lengths the second line gives, and the data is Fortran-ordered.
    """
    lines = path.read_text().splitlines()
    n1x, n2x, _, _, _, n3, nat, ntyp = (int(x) for x in lines[1].split())
    values = np.array(" ".join(lines[4 + ntyp + nat:]).split(), dtype=float)
    return np.transpose(values[:n1x * n2x * n3].reshape((n3, n2x, n1x)), (2, 1, 0))


# --------------------------------------------------------------------------
# against pp.x
# --------------------------------------------------------------------------


def test_the_tunnelling_density_matches_pp_x():
    """``plot_num = 5``, ``sample_bias = 0``, on QE's own fcc aluminium.

    Three things make this an exact comparison rather than an approximate one.
    ``stm.f90`` divides by the cell volume and *not* by ``degauss``, so its
    image is the zero-bias one times the width. It uses the run's own smearing,
    which here is Marzari-Vanderbilt, not the Gaussian this defaults to. And it
    truncates the band sum at three widths either side of the window
    (``first_band``/``last_band``), which is worth 0.4 per cent of the image on
    this cell -- so ``band_cutoff = 3`` is what makes the two the same sum, and
    leaving it off is what makes ours the complete one.
    """
    reference = _read_filplot(CASES / "reference.stm.al-metal")
    calculator = _converged("al-metal")
    image = calculator.get_stm(height=0.0, width=0.05,
                               smearing="marzari-vanderbilt", band_cutoff=3.0)
    ours = np.asarray(image.density) * image.width
    assert ours.shape == reference.shape
    assert np.abs(ours - reference).max() < 1.0e-8


def test_the_band_truncation_is_the_whole_of_the_difference():
    """Without QE's cutoff the two differ, and by the states it drops.

    The pair to the test above: it is the truncation that separates them and
    not anything in the assembly. **The complete sum is the smaller one**,
    which is the direction that says what QE's three widths are really doing
    here -- Marzari-Vanderbilt's delta is negative for ``x > sqrt(2)``, so the
    states past the cutoff carry *negative* weight and dropping them raises the
    image. It is the same objection ``PLAN.md`` P52 makes to using a
    non-positive delta for a quantity that has a sign, one order out: a
    tunnelling density built from cold smearing is not positive by
    construction, and that is why the default here is a Gaussian.
    """
    reference = _read_filplot(CASES / "reference.stm.al-metal")
    calculator = _converged("al-metal")
    whole = calculator.get_stm(height=0.0, width=0.05,
                               smearing="marzari-vanderbilt")
    ours = np.asarray(whole.density) * whole.width
    gap = np.abs(ours - reference).max() / reference.max()
    assert 1.0e-3 < gap < 1.0e-2
    assert ours.mean() < reference.mean()

    # and a Gaussian, which is positive everywhere, loses nothing to the cutoff
    positive = calculator.get_stm(height=0.0, width=0.05, smearing="gaussian")
    assert np.asarray(positive.density).min() > 0.0


def test_the_image_integrates_to_the_density_of_states():
    """``int rho_STM d3r = D(E_F)``: the sum rule, against ``compute_dos``.

    Nothing of the reference is involved. It is the check that would catch a
    factor of two in the spin degeneracy, a missing ``1/Omega``, or a delta
    that is not normalised.
    """
    from defumat.workflows.dos import compute_dos

    calculator = _converged("al-metal")
    scf = calculator.get_scf()
    image = calculator.get_stm(height=0.0, width=0.05,
                               smearing="marzari-vanderbilt")
    dos = compute_dos(scf.eigenvalues_by_spin,
                      calculator.system.kpoints.weights,
                      np.array([scf.fermi_energy]),
                      "marzari-vanderbilt", degauss=0.05)
    assert image.integral == pytest.approx(float(np.asarray(dos.dos).ravel()[0]),
                                           rel=1e-10)


def test_a_denser_grid_re_solves_the_bands_and_moves_the_fermi_level():
    """The knob every docstring here leads with, and it had no test.

    A delta at the Fermi level on the SCF's own handful of k-points is a sum
    over the few bands that happen to be near it, so ``grid=`` is the
    convergence parameter of the whole quantity. Two things have to be true and
    only the first is obvious: the bands are re-solved *there*, and the Fermi
    level is recomputed *there* -- it is the level of that k-set that makes the
    cell neutral, and the delta has to sit on it for the sum rule to hold. The
    check is the sum rule against the new k-set's own density of states, which
    fails if either half is stale.
    """
    from defumat.workflows.dos import compute_dos
    from defumat.workflows.nscf import denser_grid

    calculator = _converged("al-metal")
    scf = calculator.get_scf()
    image = calculator.get_stm(height=0.0, width=0.05,
                               smearing="marzari-vanderbilt", grid=(6, 6, 6))
    assert image.grid == (6, 6, 6)
    assert image.energy != scf.fermi_energy      # recomputed, not carried over
    assert abs(image.energy - scf.fermi_energy) < 0.05

    kpoints = denser_grid(calculator.system, (6, 6, 6))
    states = calculator.get_nscf(kpoints=kpoints)
    dos = compute_dos(states.eigenvalues, states.kpoints.weights,
                      np.array([image.energy]), "marzari-vanderbilt", degauss=0.05)
    assert image.integral == pytest.approx(float(np.asarray(dos.dos).ravel()[0]),
                                           rel=1e-8)


def test_the_plane_is_read_out_of_the_grid_exactly():
    """Sampling on the grid's own points returns the grid, on a real cell."""
    from defumat.basis.builder import build_basis
    from defumat.basis.sample import sample_field

    calculator = _converged("al-metal")
    image = calculator.get_stm(height=0.0, width=0.05)
    dense = build_basis(calculator.system).dense
    axes = [np.arange(n) / n for n in dense.grid]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape((-1, 3))
    back = sample_field(np.asarray(image.density), dense, points).reshape(dense.grid)
    assert np.abs(back - np.asarray(image.density)).max() < 1.0e-12


# --------------------------------------------------------------------------
# spin-polarized STM
# --------------------------------------------------------------------------


#: The plane both hydrogen chains are imaged on: it contains the chain axis and
#: stands off it by 0.15 of the cell's ``a``, which is 1.8 bohr.
_CHAIN_PLANE = ((0.15, 0.0, 0.0), (0.15, 0.30, 0.0), (0.15, 0.0, 1.0))


def test_an_antiferromagnet_is_flat_in_charge_and_alternates_in_spin():
    """The point of a spin-polarized tip, on the cheapest cell that has one.

    Both atoms of the antiferromagnetic hydrogen chain carry the same charge,
    so an ordinary STM image cannot tell them apart -- and it does not, to six
    figures. A tip magnetized along ``+z`` sees one of them and a tip along
    ``-z`` sees the other, by equal and opposite amounts, which is the
    antiferromagnetic symmetry reproducing itself rather than being imposed.
    """
    calculator = _converged("h-chain-afm")
    common = dict(plane=_CHAIN_PLANE, shape=(4, 32), width=0.10)
    charge = calculator.get_stm(**common)
    up = calculator.get_stm(spin="up", **common)
    down = calculator.get_stm(spin="down", **common)

    # the two channels are a partition of the charge, exactly
    assert np.abs(up.values + down.values - charge.values).max() < 1.0e-14

    first, second = 0, 16          # the atoms sit at z = 0 and z = 1/2
    assert charge.values[0, first] == pytest.approx(charge.values[0, second],
                                                    rel=1e-5)
    asymmetry = ((up.values - down.values) / charge.values)[0]
    assert asymmetry[first] > 0.02
    assert asymmetry[second] == pytest.approx(-asymmetry[first], rel=1e-3)


def test_a_noncollinear_tip_selects_the_sublattice_it_points_at():
    """Four moments, each 90 degrees from the last, and a tip that turns.

    The strongest statement here has no reference in it: a tip along ``x``
    sees atoms 1 and 3 with opposite sign and atoms 2 and 4 **not at all**,
    because their moments are perpendicular to it -- and a tip along ``z`` sees
    none of the four, every moment lying in the plane. The contrast has the
    same magnitude in every direction, which says the four moments are equal
    without anything having imposed it.
    """
    calculator = _converged("h-chain-90deg")
    common = dict(plane=_CHAIN_PLANE, shape=(4, 64), width=0.10)
    charge = calculator.get_stm(**common)
    atoms = [0, 16, 32, 48]        # z = 0, 1/4, 1/2, 3/4

    def contrast(direction):
        image = calculator.get_stm(spin=direction, **common)
        return ((2.0 * image.values - charge.values) / charge.values)[0, atoms]

    flat = charge.values[0, atoms]
    assert np.ptp(flat) / flat.mean() < 1.0e-4

    along_x = contrast((1, 0, 0))
    assert along_x[0] > 0.05
    assert along_x[2] == pytest.approx(-along_x[0], rel=1e-3)
    assert abs(along_x[1]) < 1.0e-4 and abs(along_x[3]) < 1.0e-4

    along_y = contrast((0, 1, 0))
    assert along_y[1] == pytest.approx(along_x[0], rel=1e-3)
    assert along_y[3] == pytest.approx(-along_x[0], rel=1e-3)
    assert abs(along_y[0]) < 1.0e-4 and abs(along_y[2]) < 1.0e-4

    assert np.abs(contrast((-1, 0, 0)) + along_x).max() < 1.0e-8
    assert np.abs(contrast((0, 0, 1))).max() < 1.0e-4


def test_a_spinor_image_obeys_the_same_sum_rule():
    """The check that catches a factor of two on the noncollinear path.

    A spinor band holds one electron where a scalar one holds two, and every
    ``KPoints`` constructor applies the unpolarized degeneracy unconditionally
    (``PLAN.md`` P51's ``for_spin`` trap). Neither the contrast test above nor
    ``up + down = charge`` can see that factor: both are ratios. This is the
    same ``int rho_STM = D(E_F)`` the aluminium test makes, on the branch where
    the degeneracy is different.
    """
    from defumat.workflows.dos import compute_dos

    calculator = _converged("h-chain-90deg")
    scf = calculator.get_scf()
    image = calculator.get_stm(plane=_CHAIN_PLANE, shape=(2, 2), width=0.10)
    dos = compute_dos(scf.eigenvalues_by_spin,
                      calculator.system.kpoints.weights,
                      np.array([scf.fermi_energy]), "gaussian", degauss=0.10)
    assert image.integral == pytest.approx(float(np.asarray(dos.dos).ravel()[0]),
                                           rel=1e-10)


@pytest.mark.parametrize("case", ["si2-us", "si2-paw"])
def test_a_soft_dataset_image_carries_its_augmentation_charge(case):
    """``becsum`` is rebuilt from the tunnelling weights, not left behind.

    A window wide enough to hold whole states integrates to the number of
    states in it, because ``<psi|S|psi> = 1`` -- and the ``S`` is where the
    augmentation charge is. Dropping it, or rebuilding ``becsum`` from the
    ground state's occupations instead of the tip's, moves this number by the
    augmentation's share of the norm, which for silicon is percent-sized.

    ``pp.x`` cannot be the reference here: ``stm.f90`` sums ``|psi|^2`` and
    never calls ``addusdens``, so it is norm-conserving only.
    """
    from defumat.stm.image import tunnelling_weights

    calculator = _converged(case)
    scf = calculator.get_scf()
    energy = scf.homo if scf.fermi_energy is None else scf.fermi_energy
    image = calculator.get_stm(height=0.0, shape=(4, 4), bias=-0.5,
                               energy=energy, width=1.0e-4)
    expected = tunnelling_weights(
        np.asarray(scf.eigenvalues_by_spin),
        np.asarray(calculator.system.kpoints.weights),
        energy=energy, width=1.0e-4, bias=-0.5,
    ).sum()
    assert image.integral == pytest.approx(float(expected), rel=2e-6)


# --------------------------------------------------------------------------
# the physics, on a surface
# --------------------------------------------------------------------------


def test_graphite_shows_one_of_its_two_sublattices():
    """The classic STM result, and it is a statement about stacking.

    In AB-stacked graphite one sublattice of the surface layer sits above an
    atom of the layer below and the other above a hollow, so the two are
    inequivalent and only one is bright at the Fermi level. Both are carbon and
    both carry the same charge; what differs is the density of states *at the
    Fermi level*, which is what an STM measures and a total-charge plot does
    not.
    """
    calculator = _converged("graphene-bilayer")
    image = calculator.get_stm(height=0.85, shape=(48, 48))
    values = image.values

    def at(fx, fy):
        return values[int(round(fx * 48)) % 48, int(round(fy * 48)) % 48]

    over_a, over_b = at(0.0, 0.0), at(2 / 3, 1 / 3)
    hollow = at(1 / 3, 2 / 3)
    assert over_b / over_a > 1.3          # the two sublattices are not alike
    assert over_a / hollow > 100.0        # and both are far above the hollow


def test_the_tunnelling_density_decays_exponentially_into_the_vacuum():
    """Tersoff-Hamann's own statement, and it involves none of the assembly.

    A state at energy ``E`` under a vacuum level ``V`` decays as
    ``exp(-sqrt(V - E + k_par^2) z)`` and its square twice as fast, so the
    tunnelling density falls off log-linearly at a rate of **at least**
    ``2 sqrt(V - E_F)``. The bound is one-sided because the states at graphene's
    Fermi level sit at ``K`` rather than at ``Gamma``, and their in-plane
    momentum steepens the decay -- 1.86 measured here against 0.98 for
    ``k_par = 0`` and 2.92 for ``|K|``.
    """
    calculator = _converged("graphene-bilayer")
    scf = calculator.get_scf()
    image = calculator.get_stm(height=0.85, shape=(8, 8))

    density = np.asarray(image.density)
    potential = np.asarray(scf.potential)
    if potential.ndim == 4:
        potential = potential[0]
    length = float(np.asarray(calculator.system.cell.at)[2, 2])
    z = np.arange(density.shape[-1]) / density.shape[-1] * length

    planar = density.mean(axis=(0, 1))
    vacuum_level = float(potential.mean(axis=(0, 1))[(z > 18.5) | (z < 1.5)].max())
    kappa = np.sqrt(vacuum_level - scf.fermi_energy)

    band = (z > 15.5) & (z < 18.5)
    logarithm = np.log(planar[band])
    fit = np.polyfit(z[band], logarithm, 1)
    assert -fit[0] > 2.0 * kappa                     # no slower than the bound
    # and it really is a line: two decay constants superpose here (the layer
    # below shows through, and the states at E_F carry a spread of in-plane
    # momenta), so what is claimed is a straight log and not a single exponent.
    residual = logarithm - np.polyval(fit, z[band])
    assert 1.0 - residual.var() / logarithm.var() > 0.999   # measured 0.9992 over five orders of decay
