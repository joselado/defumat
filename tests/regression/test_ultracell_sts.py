"""P90: the tunnelling spectrum of a modulation, against the image it sections.

``tests/regression/test_ultracell_stm.py`` checks the image -- the local density
of states at one tip energy, on a plane. This file checks the other section of
the same function, the curve at one place over many biases, which is what
scanning tunnelling spectroscopy measures and is the thing a charge or spin
density wave is actually read with.

The route is different and that is the whole point of the checks here. An image
rebuilds the ultracell density from every state and reads it on the plane; a
spectrum samples ``Psi_j`` at the tip points once and contracts the energy axis
against ``|Psi_j|^2``, never touching the box. So:

* **the spectrum at one energy is the image at that energy**, which is the
  phase's own number: two routes with nothing in common but the smeared delta,
  on a **modulated** state, because P89's trap applies unchanged -- a state built
  from a single ``Q`` hides the sign of ``q`` in its modulus;
* the same against the whole-cell transmission over the *whole axis*, which
  shares the sampler and pins the two ``energies`` conventions against each
  other;
* **tiled**, against the unit cell's own image at each energy on the folded
  k-set, which is what says the ``N`` is in the right place;
* **the sum rule made falsifiable**: the integral this reports is a sum of
  weights and is exact by orthonormality, so comparing it against
  ``compute_dos`` is one equation satisfied by construction. Sampling the
  spectrum at the box's own grid points and integrating it is not;
* the spinor channels against the density route's, which is where the ``pw.x``
  number behind the Pauli convention lives;
* ``current`` against the window it integrates, **with the control that says
  why they differ**: QE's window damps a state outside it by the delta's value
  and the integral gives the cumulative one, so the two agree only once every
  level is a few widths clear of both edges.
"""

import tempfile
from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
import pytest

from defumat import Calculator
from defumat.ultracell import run_ultracell
from defumat.workflows.stm import run_stm, run_sts
from defumat.workflows.ultracell import (
    run_ultracell_stm,
    run_ultracell_sts,
    run_ultracell_transport,
)

pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """Bound the peak: this file runs a unit cell and three ultracells.

    ``CLAUDE.md``'s rule for any file over about three distinct cells -- cells
    that share no shape each compile the whole stack afresh and XLA keeps every
    executable for the life of the process.
    """
    yield
    jax.clear_caches()


AMPLITUDE = 0.05
WIDTH = 0.02
PLANE = dict(height=0.35, axis=2)

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

#: **A magnet, and with its moment off every axis**, which silicon is not: a
#: nonmagnetic cell seeded noncollinear relaxes to a moment of parts in ten
#: thousand, and a transverse component that small cannot tell a reversed
#: ``m_y`` from round-off. This is the committed simple-cubic hydrogen of
#: ``tests/data/qe/h-noncolin-ultracell.in``, whose lattice constant is chosen
#: so the moment is 0.62 and responds to everything, seeded along
#: ``(1,1,1)/sqrt(3)`` so that ``m_x`` and ``m_y`` are both large and equal --
#: which makes a swapped index in ``m_y = 2 Im(conj(u) d)`` a sign reversal of a
#: quantity the size of the moment rather than of a null.
NONCOLLINEAR = """&control
 calculation='scf'
/
&system
 ibrav=1, celldm(1)=5.5, nat=1, ntyp=1, ecutwfc=15.0,
 nosym=.true., noinv=.true.,
 noncolin=.true., starting_magnetization(1)=0.8,
 angle1(1)=54.7356, angle2(1)=45.0,
 occupations='smearing', smearing='gaussian', degauss=0.02
/
&electrons
 conv_thr=1.0d-10
 mixing_beta=0.3
/
ATOMIC_SPECIES
 H 1.008 H.pz-vbc.UPF
ATOMIC_POSITIONS crystal
 H 0.0 0.0 0.0
K_POINTS automatic
 {k0} {k1} {k2} 0 0 0
"""

#: Two channels, which is the regime neither of the cells above exercises: an
#: unpolarized run has one and a spinor run has four, and the two-channel path is
#: the only one where a tip's projection is a *difference of channels* rather
#: than a contraction of a spinor. Silicon's moment collapses to nothing here and
#: that does not matter: what is being compared is two routes to the same two
#: channels, which is exact whether or not there is a moment between them.
COLLINEAR = SILICON.replace(
    " nosym=.true., noinv=.true.",
    " nosym=.true., noinv=.true., nspin=2, starting_magnetization(1)=0.1,\n"
    " occupations='smearing', smearing='gaussian', degauss=0.02"
).replace(" conv_thr=1.0d-12", " conv_thr=1.0d-9")

#: Symmetry left **on**, which every other cell here turns off: the one case
#: the wedge refusal is about.
SYMMETRIC = SILICON.replace(" nosym=.true., noinv=.true.\n", "")

_WORK = Path(tempfile.mkdtemp(prefix="defumat-sts-"))


def _calculator(pseudo_dir, name, template, grid) -> Calculator:
    path = _WORK / name
    path.write_text(template.format(k0=grid[0], k1=grid[1], k2=grid[2]))
    return Calculator.from_file(path, pseudo_dir=pseudo_dir)


@lru_cache(maxsize=2)
def _converged(pseudos, grid, nbnd=12, template="silicon"):
    """The unit cell on a folded grid, tight enough to be a basis.

    ``maxsize=2`` and never ``None``: what is held is the wavefunctions.
    """
    templates = {"silicon": SILICON, "noncollinear": NONCOLLINEAR,
                 "collinear": COLLINEAR, "symmetric": SYMMETRIC}
    calculator = _calculator(Path(pseudos), f"si_{template}_{''.join(map(str, grid))}.in",
                             templates[template], grid)
    loose = {"noncollinear": 1e-10, "collinear": 1e-9}
    scf = calculator.get_scf(conv_thr=loose.get(template, 1e-12), nbnd=nbnd)
    assert scf.converged
    return calculator, scf


def _modulation(shape, axis=0, amplitude=AMPLITUDE):
    return lambda x: amplitude * np.cos(2 * np.pi * x[..., axis] / shape[axis])


def _midgap(scf) -> float:
    return 0.5 * (float(scf.homo) + float(scf.lumo))


@lru_cache(maxsize=2)
def _modulated(pseudos, shape, kgrid, nbnd=12, template="silicon"):
    """A converged, modulated ultracell and the unit cell behind it."""
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    calculator, scf = _converged(pseudos, folded, nbnd, template)
    result = run_ultracell(
        calculator.system, calculator.pseudos, scf, shape, kgrid, nbnd=nbnd,
        external=_modulation(shape), conv_thr=1e-9, states_conv_thr=1e-9)
    assert result.converged
    return calculator, scf, result


# -- the phase's own number ---------------------------------------------------


def test_a_spectrum_at_one_energy_is_the_image_at_that_energy(pseudo_dir):
    """Null 1, and everything else here rests on it.

    Two routes to the same function with nothing shared but
    ``tunnelling_weights``: the image scatters the frozen coefficients into the
    ultracell box, transforms, squares and reads the result off the density's
    own ``G + Q`` sphere; the spectrum relabels the same coefficients as one
    vector on the ultracell's plane-wave sphere and evaluates ``Psi`` at the
    points directly. Agreement pins the ``N`` -- which sits in the weights on one
    side and in ``Omega_u`` on the other -- and the whole relabelling with it.

    **On a modulated state.** A state built from a single ``Q`` differs by
    ``e^{-2iQ.r}`` if the sign of ``q`` is wrong and by nothing at all in
    modulus, so an unmodulated cell passes this with the sign reversed.

    The floor is the *image's*, not the spectrum's: ``_box_coefficients`` reads
    a box that aliases its outermost shell from about ``N = 5``, and the
    spectrum never touches a box at all. At ``N = 2`` both are exact and the
    number is round-off, 2.5e-15 of the peak.
    """
    calculator, scf, result = _modulated(str(pseudo_dir), (2, 1, 1), (1, 2, 1))
    geometry = dict(shape=(8, 6), **PLANE)
    energies = float(scf.homo) - 0.05 + np.linspace(-0.05, 0.05, 5)

    spectrum = run_sts_of(calculator, result, energies, **geometry)
    for at, energy in enumerate(energies):
        image = run_ultracell_stm(calculator.system, calculator.pseudos, result,
                                  energy=float(energy), width=WIDTH, **geometry)
        reference = np.asarray(image.values)
        assert reference.max() > 0.0
        error = np.abs(np.asarray(spectrum.values)[at] - reference).max()
        assert error / reference.max() < 1e-12
        assert spectrum.integral[at] == pytest.approx(image.integral, rel=1e-12)


def run_sts_of(calculator, result, energies, **options):
    return run_ultracell_sts(calculator.system, calculator.pseudos, result,
                             energies=energies, width=WIDTH, **options)


def test_the_whole_cell_transmission_is_the_spectrum(pseudo_dir):
    """P89's Tersoff-Hamann identity, now over a whole axis.

    ``exit_region="volume"`` widens the exit region until the overlap is the
    identity, and the transmission is then the same sum this is. It shares the
    sampler and the smeared delta with the spectrum, so it is not an independent
    route -- what it pins is the two ``energies`` conventions against each other
    and the ``sqrt(delta)`` amplitude splitting, which null 1 cannot see.
    """
    calculator, scf, result = _modulated(str(pseudo_dir), (2, 1, 1), (1, 2, 1))
    geometry = dict(shape=(8, 6), **PLANE)
    energies = float(scf.homo) - 0.05 + np.linspace(-0.04, 0.04, 5)

    spectrum = run_sts_of(calculator, result, energies, **geometry)
    with pytest.warns(UserWarning, match="do not lie between"):
        transport = run_ultracell_transport(
            calculator.system, calculator.pseudos, result, exit_height=0.05,
            exit_axis=2, energies=energies, broadening=WIDTH,
            exit_region="volume", **geometry)
    reference = np.asarray(spectrum.values)
    assert reference.max() > 0.0
    assert np.abs(np.asarray(transport.values) - reference).max() \
        / reference.max() < 1e-12


# -- where the N sits ---------------------------------------------------------


@pytest.mark.parametrize("shape,kgrid", [((1, 1, 1), (2, 2, 1)),
                                         ((2, 1, 1), (1, 2, 1))])
def test_an_unmodulated_spectrum_is_the_tiled_unit_cell_s(shape, kgrid,
                                                          pseudo_dir):
    """With nothing applied, the ultracell's curve is the unit cell's, tiled.

    **Against the unit cell's image on the folded grid**, one call per energy,
    which is the route with nothing of this one in it -- comparing against
    ``run_sts`` instead would share the sampler and the contraction and would
    check the plumbing rather than the answer. The k-set has to be the folded
    one for P89's reason: a spectrum is a sum over states and a different
    Brillouin-zone sampling is a different sum.

    **The floor is the two diagonalisations seen through the delta's slope, and
    it is looser here than P89's 1e-8 by a factor the width sets.** The two
    sides are built from different wavefunctions -- the SCF's own, and the
    fixed-density solve's rediagonalised in the frozen envelope basis -- whose
    levels come out a median **1.24e-8 Ry** apart however tightly either side is
    converged. P89's image runs in a **window**, where a state's weight is 1 or 0
    and a shift of 1e-8 moves nothing; a **delta** of width ``w`` has slope
    ``1/w`` there, so the same shift is worth ``1.2e-8/w``. Measured on this
    cell at width 0.01, 0.02, 0.04 and 0.08: **7.7e-7, 2.9e-7, 1.3e-7 and
    3.9e-8**, which is that ratio and is why the tolerance is on the width
    rather than on a threshold.
    """
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    calculator, scf = _converged(str(pseudo_dir), folded)
    result = run_ultracell(calculator.system, calculator.pseudos, scf, shape,
                           kgrid, nbnd=12, conv_thr=1e-10, states_conv_thr=1e-12)
    assert result.converged

    energies = float(scf.homo) - 0.05 + np.linspace(-0.04, 0.04, 3)
    geometry = dict(shape=(12 * shape[0], 12), **PLANE)
    spectrum = run_sts_of(calculator, result, energies, **geometry)

    for at, energy in enumerate(energies):
        plain = run_stm(calculator.system, calculator.pseudos, scf,
                        shape=(12, 12), energy=float(energy), width=WIDTH,
                        **PLANE)
        tiled = np.concatenate([plain.values] * shape[0], axis=0)
        assert np.abs(np.asarray(spectrum.values)[at] - tiled).max() \
            / np.abs(tiled).max() < 1e-6
        # 2.9e-7 at this width, and it is the level difference above
        assert spectrum.integral[at] == pytest.approx(plain.integral, rel=1e-6)


def test_the_unit_cell_spectrum_is_the_unit_cell_image_energy_by_energy(
        pseudo_dir):
    """``run_sts`` against ``run_stm``, which is the same null one cell down.

    It is the check that the geometry-agnostic contraction really is agnostic:
    the same function, handed a plane-wave run's own sphere and cell instead of
    an ultracell's, has to reproduce the route that goes through
    ``Calculation.density``.
    """
    calculator, scf = _converged(str(pseudo_dir), (2, 2, 1))
    energies = float(scf.homo) - 0.05 + np.linspace(-0.04, 0.04, 4)
    spectrum = run_sts(calculator.system, calculator.pseudos, scf,
                       energies=energies, width=WIDTH, shape=(10, 10), **PLANE)
    for at, energy in enumerate(energies):
        image = run_stm(calculator.system, calculator.pseudos, scf,
                        shape=(10, 10), energy=float(energy), width=WIDTH,
                        **PLANE)
        reference = np.asarray(image.values)
        assert np.abs(np.asarray(spectrum.values)[at] - reference).max() \
            / reference.max() < 1e-12


def test_a_collinear_tip_picks_the_same_channel_the_image_does(pseudo_dir):
    """The two-channel regime, which neither of the other cells reaches.

    An unpolarized run carries one component and a spinor run four; two is the
    only case where a magnetic tip is a **difference of channels** rather than a
    contraction of a spinor, and it goes through
    :func:`~defumat.stm.image.project_spin`'s collinear branch. The check is the
    image at the same energies, so it is the same null as the unpolarized one
    with the projection in the path, and each channel is taken separately as
    well as projected -- the two together are what say the channel axis did not
    get transposed, since ``up`` and ``down`` are the same shape and only the
    reference tells them apart.
    """
    calculator, scf = _converged(str(pseudo_dir), (2, 2, 1), 12, "collinear")
    energies = float(scf.fermi_energy) + np.linspace(-0.04, 0.04, 3)
    for spin in ("up", "down"):
        spectrum = run_sts(calculator.system, calculator.pseudos, scf,
                           energies=energies, width=WIDTH, shape=(8, 8),
                           spin=spin, **PLANE)
        for at, energy in enumerate(energies):
            image = run_stm(calculator.system, calculator.pseudos, scf,
                            shape=(8, 8), energy=float(energy), width=WIDTH,
                            spin=spin, **PLANE)
            reference = np.asarray(image.values)
            assert reference.max() > 0.0
            assert np.abs(np.asarray(spectrum.values)[at] - reference).max() \
                / reference.max() < 1e-12
            # and the two raw channels beside the projection
            channels = np.asarray(spectrum.values_by_spin)[:, at]
            assert channels.shape[0] == 2
            assert np.abs(channels - np.asarray(image.values_by_spin)).max() \
                / reference.max() < 1e-12


def test_the_sum_rule_is_the_spectrum_integrated_not_the_weights_resummed(
        pseudo_dir):
    """``integral`` against the field it claims to be the integral of.

    The weak form of this check compares ``integral`` with ``compute_dos``, and
    both are a sum of the same ``w0gauss`` terms, so it is one equation
    satisfied by construction -- a null that cannot be told from a pass. The
    form here samples the spectrum at the **box's own grid points** and
    integrates it, which is a different quantity reached by a different path:
    it is exact only because ``sum_G |c|^2 = 1``, which is the assumption the
    whole amplitude route rests on and is what would break on an ultrasoft
    dataset.
    """
    calculator, scf, result = _modulated(str(pseudo_dir), (2, 1, 1), (1, 2, 1))
    states = result.states
    grid = states.ultracell.grid
    # the box points, in **unit-cell** coordinates over [0, n_i), which is the
    # convention run_ultracell_sts takes.
    points = np.stack(np.meshgrid(*[np.arange(m) / m for m in grid],
                                  indexing="ij"), axis=-1).reshape(-1, 3)
    points = points * np.asarray(states.ultracell.shape, dtype=float)

    energies = float(scf.homo) - 0.05 + np.linspace(-0.04, 0.04, 3)
    spectrum = run_sts_of(calculator, result, energies, tip=points)
    volume = float(result.cell_volume)
    integrated = np.asarray(spectrum.values).mean(axis=1) * volume
    assert integrated.min() > 0.0
    assert np.abs(integrated - spectrum.integral).max() \
        / spectrum.integral.max() < 1e-10


# -- the spinor channels ------------------------------------------------------


def test_a_spinor_spectrum_carries_the_density_s_own_four_channels(pseudo_dir):
    """``(n, m_x, m_y, m_z)`` against the route the ``pw.x`` number is behind.

    The transverse pair is the half that needs a state with a moment off the
    ``z`` axis: ``m_y = 2 Im(conj(u) d)`` with its two factors exchanged is
    still real, still the right size and wrong in sign, and it is identically
    zero on anything collinear. This cell is seeded at 60 degrees, so both
    transverse components are live, and the reference is the image's
    ``values_by_spin``, built by scattering into the box one component at a
    time.
    """
    shape, kgrid = (2, 1, 1), (1, 2, 2)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    calculator, scf = _converged(str(pseudo_dir), folded, 16, "noncollinear")
    result = run_ultracell(calculator.system, calculator.pseudos, scf, shape,
                           kgrid, nbnd=16, external=_modulation(shape),
                           conv_thr=1e-8, states_conv_thr=1e-8,
                           mixing_beta=0.3, max_iterations=120)
    assert result.converged
    assert int(result.states.npol) == 2

    geometry = dict(shape=(8, 6), height=0.5, axis=2)
    energy = float(scf.fermi_energy)
    spectrum = run_ultracell_sts(calculator.system, calculator.pseudos, result,
                                 energies=[energy], width=WIDTH, **geometry)
    image = run_ultracell_stm(calculator.system, calculator.pseudos, result,
                              energy=energy, width=WIDTH, **geometry)
    reference = np.asarray(image.values_by_spin)
    assert reference.shape[0] == 4
    ours = np.asarray(spectrum.values_by_spin)[:, 0]

    charge = np.abs(reference[0]).max()
    assert charge > 0.0
    for channel in range(4):
        scale = np.abs(reference[channel]).max()
        # **each channel against its own size**, so that a transverse component
        # is compared with itself rather than with the charge, where a reversed
        # sign would hide behind a small ratio
        assert scale / charge > 1e-2, f"channel {channel} is a null, not a test"
        assert np.abs(ours[channel] - reference[channel]).max() / scale < 1e-8
    # and the two transverse components are equal, which is the seeded
    # direction and is what makes a swap between them visible as well
    assert np.abs(reference[1]).max() == pytest.approx(
        np.abs(reference[2]).max(), rel=1e-3)


# -- the current, and why it is not the window to the last digit --------------


def test_the_current_is_the_window_once_the_edges_are_clear_of_the_levels(
        pseudo_dir):
    """``I(V)`` against ``bias=``, and the control that explains the gap.

    ``tunnelling_weights`` damps a state outside the window by the **delta's
    value undivided by the width**, which is ``stm.f90``'s expression
    transcribed rather than corrected; integrating the delta gives its
    cumulative one instead -- 0.564 against 0.5 for a Gaussian level exactly on
    an edge. So the two are different conventions at an edge and the same
    quantity away from one, and what is asserted is that **moving the edge away
    collapses the difference**: a single number here would read as agreement or
    as a bug and could not tell which.

    The width is the only free variable that moves an edge away in units of
    itself, because this cell's smeared gap is about one width wide and there is
    nowhere else to put an edge. Measured: 8.5e-2, 3.1e-3, 4.0e-9 as the nearest
    level goes 0.4, 1.6 and 4.0 widths out.
    """
    calculator, scf, result = _modulated(str(pseudo_dir), (2, 1, 1), (1, 2, 1))
    levels = np.asarray(result.states.eigenvalues).ravel()
    geometry = dict(shape=(6, 6), **PLANE)
    high = _midgap(scf)
    low = float(levels.min()) - 0.4

    errors = []
    for width in (0.02, 0.005, 0.002):
        window = run_ultracell_stm(
            calculator.system, calculator.pseudos, result, energy=high,
            bias=-(high - low), width=width, **geometry)
        axis = np.linspace(low, high, int(round((high - low) / (width / 3.0))) + 1)
        spectrum = run_ultracell_sts(
            calculator.system, calculator.pseudos, result, energies=axis,
            width=width, **geometry)
        reference = np.asarray(window.values)
        got = np.abs(np.asarray(spectrum.current)[0])
        errors.append(float(np.abs(got - reference).max()
                            / np.abs(reference).max()))
    assert errors[0] > 1e-2                      # an edge on a level: 8.5e-2
    assert errors[-1] < 1e-7                     # four widths clear: 4.0e-9
    assert errors[0] > errors[1] > errors[2]


# -- the refusals -------------------------------------------------------------


def test_a_wedge_is_refused_and_the_whole_grid_is_not(pseudo_dir):
    """The refusal a spectrum has and an image does not.

    ``run_stm`` sums through ``Calculation.density``, which symmetrises, so a
    reduced k-set gives it the whole zone's answer; nothing symmetrises the
    amplitude route. Both branches are exercised, because a guard that always
    fires and a guard that never does look the same from one side -- and the
    ``grid=`` escape has to actually work, since it is what the message offers.
    """
    calculator, scf = _converged(str(pseudo_dir), (4, 4, 4), 12, "symmetric")
    assert np.ptp(np.asarray(calculator.system.kpoints.weights)) > 1e-8
    energies = float(scf.homo) + np.array([-0.05, 0.0])
    with pytest.raises(NotImplementedError, match="symmetry-reduced"):
        run_sts(calculator.system, calculator.pseudos, scf, energies=energies,
                width=WIDTH, shape=(4, 4), **PLANE)

    spectrum = run_sts(calculator.system, calculator.pseudos, scf,
                       energies=energies, width=WIDTH, shape=(4, 4),
                       grid=(2, 2, 2), conv_thr=1e-8, **PLANE)
    assert np.asarray(spectrum.values).shape == (2, 4, 4)
    assert np.asarray(spectrum.values).min() > 0.0


def test_the_wedge_sum_the_guard_refuses_is_wrong_by_a_whole_picture(pseudo_dir):
    """**What the refusal is worth**, built by hand past the guard.

    A refusal with no number behind it is an untested claim, and this one has
    three numbers that only make sense together:

    * the wedge sum of ``|psi_k(r)|^2`` differs from the whole grid's by **98
      per cent** of the peak -- it is not a small error, it is a different
      picture;
    * its **integral is right to about 1 per cent**, because the integral is a
      sum of weights and the weights of a wedge are correct. So the sum rule
      this file checks elsewhere would have *passed* on the wedge, which is why
      the guard is a guard and not a tolerance;
    * the **image** on the very same reduced k-set agrees with the whole grid's
      spectrum to **0.3 per cent**, which is the symmetrisation inside
      ``Calculation.density`` doing its job -- so the refusal is about this route
      and not about wedges, exactly as it claims.
    """
    from defumat.scf.driver import Calculation
    from defumat.workflows.stm import _finish_spectrum, sample_spectrum
    from defumat.workflows.transport import _geometry, _tip_points

    calculator, scf = _converged(str(pseudo_dir), (4, 4, 4), 12, "symmetric")
    geometry = dict(height=0.35, axis=2, shape=(12, 12))
    energies = float(scf.homo) + np.linspace(-0.05, 0.05, 3)

    whole = run_sts(calculator.system, calculator.pseudos, scf,
                    energies=energies, width=WIDTH, grid=(4, 4, 4),
                    conv_thr=1e-8, **geometry)

    calculation = Calculation(calculator.system, calculator.pseudos)
    plane, points = _tip_points(calculator.system.cell, geometry["height"],
                                geometry["axis"], None, geometry["shape"], None)
    channels, dos = sample_spectrum(
        _geometry(calculation), scf.wavefunctions,
        np.asarray(scf.eigenvalues_by_spin), points, energies=energies,
        width=WIDTH, nspin_mag=int(calculator.system.nspin_mag))
    wedge = _finish_spectrum(channels, dos, energies, points, plane, None, 1.0,
                             width=WIDTH, smearing="gaussian", bias=None,
                             fermi=None)

    right = np.asarray(whole.values)
    assert np.abs(np.asarray(wedge.values) - right).max() / right.max() > 0.5
    # and the sum rule, which cannot see any of it
    assert wedge.integral[1] == pytest.approx(whole.integral[1], rel=0.05)

    image = run_stm(calculator.system, calculator.pseudos, scf,
                    energy=float(energies[1]), width=WIDTH, **geometry)
    reference = np.asarray(image.values)
    assert np.abs(reference - right[1]).max() / right[1].max() < 0.01


def test_an_ultracell_result_is_refused_by_name(pseudo_dir):
    """``run_sts`` says which function to call, as its three relatives do."""
    calculator, scf, result = _modulated(str(pseudo_dir), (2, 1, 1), (1, 2, 1))
    with pytest.raises(NotImplementedError, match="run_ultracell_sts"):
        run_sts(calculator.system, calculator.pseudos, result,
                energies=[float(scf.homo)], width=WIDTH, **PLANE)
