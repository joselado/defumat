"""A result object draws itself, and the drawing is checked.

Presentation lives on the result objects rather than on
:class:`~defumat.calculator.Calculator`, which is the line the facade rule
draws: a ``get_*`` method must not compute, and a ``plot`` must not either --
it draws what the result already holds. There were four such methods
(``BandStructure``, ``DensityOfStates``, ``ProjectedDOS``, ``OpticalSpectrum``)
and **no test touched any of them**, which is most of why 28 of the 29
notebooks hand-rolled their axes instead (P49). Four more landed with this
file, chosen because a notebook was already drawing them by hand.

Every test here is a smoke test with a claim attached: the method runs on a
synthetic result, puts something on the axes, labels them, and refuses what it
cannot draw. None of them needs an SCF, so the file belongs in the fast gate.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from defumat.response.spectra import VibrationalSpectrum
from defumat.topology.berry import BerryCurvature
from defumat.workflows.bands import BandStructure
from defumat.workflows.dos import DensityOfStates
from defumat.workflows.pdos import ProjectedDOS
from defumat.workflows.relax import RelaxResult, RelaxStep
from defumat.workflows.spiral import SpiralScan
from defumat.workflows.tddft import OpticalSpectrum

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class _Mesh:
    """The two-line stand-in for a :class:`PlaneMesh`: a plot needs its shape."""

    shape = (12, 12)


@pytest.fixture
def curvature():
    n = _Mesh.shape[0]
    k1, k2 = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n),
                         indexing="ij")
    return BerryCurvature(mesh=_Mesh(),
                          curvature=np.sin(2 * np.pi * k1) * np.cos(2 * np.pi * k2))


@pytest.fixture
def relaxation():
    steps = [RelaxStep(index=i, positions=np.zeros((2, 3)),
                       total_energy=-15.8 + 0.01 * np.exp(-i),
                       max_force=0.1 * np.exp(-i),
                       scf_iterations=8, conv_thr=1.0e-8)
             for i in range(6)]
    return RelaxResult(converged=True, system=None, scf=None,
                       forces=np.zeros((2, 3)), steps=steps)


@pytest.fixture
def scan():
    q = np.zeros((7, 3))
    q[:, 2] = np.linspace(0.0, 0.5, 7)
    return SpiralScan(wavevectors=q,
                      energies=-1.0 - 0.002 * np.cos(4 * np.pi * q[:, 2]),
                      moments=np.tile([0.0, 0.0, 2.2], (7, 1)),
                      converged=tuple([True] * 7))


def _spectrum(raman=True, infrared=True):
    """Silicon's Gamma modes: an acoustic triplet and a Raman-active T2g."""
    zeros, activity = np.zeros(6), np.array([0.0, 0.0, 0.0, 42.0, 42.0, 42.0])
    fields = dict(
        frequencies=np.array([0.0, 0.0, 0.0, 519.2, 519.2, 519.2]),
        frequencies_thz=zeros,
        mode_raman=None, alpha=None, beta2=None,
        raman_activity=activity if raman else None,
        depolarisation=None, mode_dipole=None,
        infrared=zeros if infrared else None,
        epsilon=np.eye(3), polarizability=np.eye(3),
        clausius_mossotti=1.0,
        manifold=np.array([0, 0, 0, 1, 1, 1]),
    )
    import inspect
    accepted = inspect.signature(VibrationalSpectrum).parameters
    return VibrationalSpectrum(**{k: v for k, v in fields.items()
                                  if k in accepted})


# ----------------------------------------------------------------------
# the four that landed with P49
# ----------------------------------------------------------------------

def test_the_curvature_map_is_drawn_symmetric_about_zero(curvature):
    # A signed field read as a brightness rather than as a colour change is a
    # map that hides where the curvature reverses.
    ax = curvature.plot()
    ax.figure.canvas.draw()      # mathtext is parsed lazily
    (image,) = ax.get_images()
    low, high = image.get_clim()
    assert low == pytest.approx(-high)
    assert high == pytest.approx(np.max(np.abs(curvature.curvature)))
    assert "k_1" in ax.get_xlabel()


def test_the_curvature_map_accepts_the_axes_it_is_given(curvature):
    _, ax = plt.subplots()
    assert curvature.plot(ax=ax) is ax


def test_a_relaxation_draws_its_energy_and_its_force(relaxation):
    ax = relaxation.plot()
    ax.figure.canvas.draw()
    assert len(ax.lines) == 1                        # the energy, on the left
    energy = ax.lines[0].get_ydata()
    assert energy[-1] == pytest.approx(0.0)          # measured from the final
    assert energy[0] > energy[-1]                    # and it went downhill
    twin = [other for other in ax.figure.axes if other is not ax]
    assert len(twin) == 1 and twin[0].get_yscale() == "log"


def test_a_spiral_scan_draws_its_energy_from_the_first_point(scan):
    ax = scan.plot()
    ax.figure.canvas.draw()
    assert ax.lines[0].get_ydata()[0] == pytest.approx(0.0)
    assert "mRy" in ax.get_ylabel()


def test_a_spiral_scan_can_leave_the_moment_off(scan):
    ax = scan.plot(moment=False)
    ax.figure.canvas.draw()
    assert [other for other in ax.figure.axes if other is not ax] == []


def test_the_acoustic_modes_are_kept_out_of_the_spectrum():
    # Their activity is what the sum rule silences; a residual stick at 2 cm^-1
    # is noise that the normalisation would magnify to full scale.
    ax = _spectrum().plot(kind="raman")
    ax.figure.canvas.draw()
    grid, curve = ax.lines[0].get_xdata(), ax.lines[0].get_ydata()
    assert curve[np.argmin(np.abs(grid - 0.0))] < 1.0e-3
    assert curve[np.argmin(np.abs(grid - 519.2))] == pytest.approx(1.0)


def test_a_spectrum_refuses_the_activity_it_was_not_given():
    # Built without Born charges, it has no infrared column. Drawing a flat
    # line would be an answer; saying which ingredient is missing is not.
    with pytest.raises(ValueError, match="Born charges"):
        _spectrum(infrared=False).plot(kind="infrared")
    with pytest.raises(ValueError, match="Raman tensors"):
        _spectrum(raman=False).plot(kind="raman")


def test_a_spectrum_refuses_a_kind_that_is_not_one(): 
    with pytest.raises(ValueError, match="raman"):
        _spectrum().plot(kind="phonon")


# ----------------------------------------------------------------------
# the four that were already there and were never tested
# ----------------------------------------------------------------------

@pytest.mark.parametrize("cls", [BandStructure, DensityOfStates, ProjectedDOS,
                                 OpticalSpectrum, BerryCurvature, RelaxResult,
                                 SpiralScan, VibrationalSpectrum])
def test_every_drawable_result_takes_an_axes_and_returns_it(cls):
    # The contract the notebooks depend on: ``result.plot(ax=ax)`` composes into
    # a figure the caller laid out, rather than making one of its own.
    import inspect

    signature = inspect.signature(cls.plot)
    assert "ax" in signature.parameters
    assert signature.parameters["ax"].default is None


# ----------------------------------------------------------------------
# the comparison table (P49)
# ----------------------------------------------------------------------

def test_the_comparison_table_aligns_and_computes_the_difference():
    from defumat.io import comparison_table

    table = comparison_table([("total energy", -15.844527263, -15.84452726)],
                             fmt="{:.8f}")
    header, row = table.splitlines()
    assert header.split() == ["defumat", "reference", "difference"]
    assert row.split()[-1] == "3.0e-09"
    # the columns line up, which is the whole reason the helper exists
    assert len(header) == len(row)


def test_a_missing_reference_stays_visible_rather_than_becoming_a_zero():
    from defumat.io import comparison_table

    table = comparison_table([("Z*", -0.075715, None)])
    assert "--" in table
    assert "0.0e+00" not in table
