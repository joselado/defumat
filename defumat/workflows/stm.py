"""``run_stm``: a scanning-tunnelling image of a converged surface.

Elk's task 162 and QE's ``PP/src/stm.f90``, which compute the same thing two
ways round the same idea -- Tersoff-Hamann, in which the tunnelling current at
the tip is the sample's local density of states at the tip's position and at the
energy the bias selects. Both codes get there by rebuilding the density with
different occupations, and so does this: the physics is in
:mod:`defumat.stm.image`, and what is here is the part specific to a plane-wave
run -- which states are summed, on which k-set, and how a field on an FFT grid
is read on a plane that does not pass through its points.

**Nothing new is summed over bands.** The tunnelling weights go straight into
:meth:`~defumat.scf.driver.Calculation.density`, the same masked-weight call
:func:`defumat.workflows.sfac.run_structure_factors` makes for Elk's ``wsfac``
window, so the symmetrisation and the augmentation charge follow the weights
instead of being left behind -- which is one thing this has over ``stm.f90``,
whose sum is over ``|psi|^2`` alone and is therefore norm-conserving only.

**The plane is read exactly, not interpolated.** The tunnelling density is a
finite sum of plane waves, so its value at a point between grid points is that
sum evaluated there (:mod:`defumat.basis.sample`) rather than a spline through
neighbours. That matters for the one thing an STM image is about: the
corrugation in the vacuum is a small modulation on a quantity falling by orders
of magnitude across one grid spacing, and a trilinear interpolant of it carries
the grid's own periodicity as a false corrugation.

**A denser k-grid is usually wanted and is an NSCF run**, exactly as it is for a
density of states: a delta at the Fermi level on the SCF's own handful of
k-points is a sum over the few bands that happen to be near it. ``grid=``
re-solves the bands at fixed density and **recomputes the Fermi level there**,
because it is the level of that k-set that makes the cell neutral and the one
the delta has to sit on for the sum rule to hold; ``energy=`` overrides it.
"""

from __future__ import annotations

import warnings

import numpy as np

from defumat.basis.builder import build_basis
from defumat.basis.fft import r_to_g
from defumat.basis.sample import sample_coefficients
from defumat.scf.driver import Calculation
from defumat.stm.image import (
    STMImage,
    constant_current_height,
    project_spin,
    tunnelling_weights,
)
from defumat.stm.plane import PlotPlane, plot_plane
from defumat.workflows.nscf import denser_grid, fixed_density_states

__all__ = ["run_stm"]

#: QE's ``stm.f90`` broadening for a run with no smearing of its own, in Ry.
INSULATOR_WIDTH = 1.0e-5


def run_stm(
    system,
    pseudos,
    result,
    *,
    height: float | None = None,
    axis: int = 2,
    plane: tuple | PlotPlane | None = None,
    shape: tuple[int, int] = (40, 40),
    mode: str = "constant-height",
    current: float | None = None,
    heights: tuple[float, float] | None = None,
    nheights: int = 60,
    spin=None,
    polarization: float = 1.0,
    bias: float | None = None,
    band_cutoff: float | None = None,
    energy: float | None = None,
    width: float | None = None,
    smearing: str = "gaussian",
    grid: tuple[int, int, int] | None = None,
    shift: tuple[int, int, int] | None = None,
    kpoints=None,
    nbnd: int | None = None,
    conv_thr: float = 1.0e-6,
    k_batch: int | None | str = "default",
) -> STMImage:
    """A Tersoff-Hamann STM image of a converged run.

    Args:
        system, pseudos: the run's :class:`~defumat.system.builder.System` and
            its pseudopotentials.
        result: the :class:`~defumat.scf.driver.SCFResult`. Its wavefunctions
            are needed, not only its density.
        height: crystal coordinate of the tip plane along ``axis`` -- the
            shortcut for a slab, spanning the whole surface cell. In
            constant-current mode it is where the scan starts.
        axis: which lattice vector ``height`` measures along, 0, 1 or 2.
            It describes the shortcut and nothing else: a scan takes its
            direction from the plane's own normal, so an explicit ``plane``
            need not be perpendicular to a lattice vector.
        plane: the general form, ``(origin, edge1, edge2)`` in crystal
            coordinates -- Elk's ``vclp2d`` -- or a
            :class:`~defumat.stm.plane.PlotPlane` already built. Mutually
            exclusive with ``height``.
        shape: the sampling, Elk's ``np2d``.
        mode: ``"constant-height"``, Elk's, or ``"constant-current"``.
        current: the set-point for constant-current, in the units of the
            tunnelling density -- 1/(bohr^3 Ry) at zero bias, electrons/bohr^3
            in a window.
        heights: ``(lo, hi)`` in **bohr** along the surface normal, measured
            from the plane, over which the tip is scanned. Defaults to one
            lattice period along that normal, which is the largest scan that
            cannot repeat itself. It is a bound and not a recommendation: the
            scan should stay inside the *vacuum*, and a slab whose vacuum is
            thinner than the period will show the next surface coming the
            other way well before the guard fires.
        nheights: planes in the scan.
        spin: image one spin channel instead of the charge -- a magnetic tip.
            A cartesian 3-vector (the tip's moment direction), or ``"up"`` /
            ``"down"`` for a collinear run. ``None`` is the ordinary charge
            image. See :func:`defumat.stm.image.project_spin`.
        polarization: the tip's spin polarization in ``[-1, 1]``. ``1`` is a
            fully polarized tip, which makes the image a genuine spin channel.
        band_cutoff: drop states further than this many widths outside the
            window, ``stm.f90``'s fixed 3. Off by default; it is what makes an
            image reproduce ``pp.x`` bit for bit and it is an approximation.
        bias: the sample bias in Ry. ``None`` is Elk's zero-bias delta;
            a positive value images empty states between ``E`` and ``E + V``
            and a negative one the filled states below ``E``.
        energy: the energy the tip is tuned to, in Ry. Defaults to the Fermi
            level of the k-set actually used.
        width: the smeared delta's width in Ry. Defaults to the run's
            ``degauss``, and to ``1e-5`` for a run with none -- ``stm.f90``'s
            own choice for an insulator.
        smearing: which smeared delta. Gaussian whatever the run used; see
            :mod:`defumat.stm.image`.
        grid, shift, kpoints, nbnd, conv_thr: re-solve the bands at fixed
            density on a denser k-set first.
        k_batch: the k-axis batching dial.

    Returns an :class:`~defumat.stm.image.STMImage`.
    """
    _refuse_what_has_no_fermi_level(system, result)

    if kpoints is None and grid is not None:
        kpoints = denser_grid(system, grid, shift)

    if kpoints is None:
        calculation = Calculation(system, pseudos, k_batch=k_batch)
        eigenvalues = np.asarray(result.eigenvalues_by_spin)
        wavefunctions = result.wavefunctions
        levels = {"fermi_energy": result.fermi_energy,
                  "homo": result.homo, "lumo": result.lumo}
    else:
        calculation, system, eigenvalues, wavefunctions = fixed_density_states(
            system, pseudos, result.density, kpoints, nbnd, conv_thr, k_batch,
            ns=getattr(result, "ns", None),
            tau=getattr(result, "tau", None),
            becsum=tuple(getattr(result, "becsum", ()) or ()),
            field=getattr(result, "magnetic_field", None),
            field_scale=getattr(result, "field_scale", None),
        )
        eigenvalues = np.asarray(eigenvalues)
        _, levels = calculation.occupations(eigenvalues)

    if wavefunctions is None:
        raise ValueError(
            "an STM image is built from the wavefunctions and this result "
            "carries none: run the SCF without discarding them, or pass a grid "
            "so the bands are re-solved"
        )

    energy = _tip_energy(energy, levels, bias)
    width = _tip_width(width, system)

    weights = tunnelling_weights(
        eigenvalues, np.asarray(calculation.system.kpoints.weights),
        energy=energy, width=width, smearing=smearing, bias=bias,
        band_cutoff=band_cutoff,
    )
    density = np.asarray(calculation.density(wavefunctions, weights))

    dense = build_basis(calculation.system).dense
    volume = float(calculation.system.cell.volume)
    # What the tip measures: the charge, or one spin channel of it. The
    # projection is linear, so doing it on the grid and doing it on the plane
    # are the same thing -- it is done here so that a constant-current scan has
    # a single scalar field to invert.
    if spin is None:
        field = density[0] + density[1] if density.shape[0] == 2 else density[0]
    else:
        field = project_spin(density, spin, polarization)
    integral = float(field.mean() * volume)

    geometry = _plane(calculation.system.cell, height, axis, plane, shape)
    coefficients = np.asarray(r_to_g(field, dense.fft_index))
    channels = (np.asarray(r_to_g(density, dense.fft_index))
                if density.shape[0] > 1 else None)

    if mode == "constant-height":
        values = sample_coefficients(coefficients, dense, geometry.flat())
        values = values.reshape(geometry.shape)
        by_spin = None if channels is None else sample_coefficients(
            channels, dense, geometry.flat()
        ).reshape((density.shape[0],) + geometry.shape)
        image = STMImage(values=values, plane=geometry, values_by_spin=by_spin)
    elif mode == "constant-current":
        if current is None:
            raise ValueError(
                "constant-current mode needs a set-point: pass current=, in "
                + ("1/(bohr^3 Ry)" if bias is None else "electrons/bohr^3")
            )
        image = _constant_current(
            coefficients, dense, geometry, calculation.system.cell,
            current, heights, nheights,
        )
    else:
        raise ValueError(
            f"unknown mode {mode!r}: use 'constant-height' (Elk's) or "
            "'constant-current'"
        )

    image.density = density[0] if density.shape[0] == 1 else density
    image.mode = mode
    image.energy = float(energy)
    image.bias = None if bias is None else float(bias)
    image.width = float(width)
    image.smearing = smearing
    image.spin = None if spin is None else (
        spin if isinstance(spin, str) else tuple(float(c) for c in np.ravel(spin)))
    image.polarization = float(polarization)
    image.integral = integral
    image.grid = None if grid is None else tuple(int(n) for n in grid)
    return image


# --------------------------------------------------------------------------
# the plane, and the scan built out of it
# --------------------------------------------------------------------------


def _plane(cell, height, axis, plane, shape):
    """Either the ``height`` shortcut or Elk's three corners."""
    if isinstance(plane, PlotPlane):
        return plane
    if plane is not None:
        if height is not None:
            raise ValueError(
                "height= and plane= are two ways of saying the same thing: "
                "pass one. height is the surface-cell plane at that crystal "
                "coordinate; plane is Elk's three corners"
            )
        origin, edge1, edge2 = plane
        return plot_plane(cell, origin, edge1, edge2, shape)
    if height is None:
        raise ValueError(
            "an STM image needs a plane: pass height= (the crystal coordinate "
            "of the tip plane above the slab) or plane= (Elk's three corners)"
        )
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2, got {axis}")
    origin = np.zeros(3)
    origin[axis] = float(height)
    first, second = [i for i in (0, 1, 2) if i != axis]
    edge1, edge2 = origin.copy(), origin.copy()
    edge1[first] = 1.0
    edge2[second] = 1.0
    return plot_plane(cell, origin, edge1, edge2, shape)


def _constant_current(coefficients, dense, geometry, cell, current,
                      heights, nheights):
    """Scan the plane outwards and invert for the height at the set-point."""
    at = np.asarray(cell.at, dtype=float)
    normal = np.asarray(geometry.normal, dtype=float)
    # The scan coordinate is bohr along the surface normal, so the corrugation
    # is a length rather than a fraction of a cell nobody thinks in.
    # How far the tip can be withdrawn before it meets the periodic image of
    # the surface: the shortest lattice period along the plane's own normal,
    # which is the plane's property and not the ``axis`` argument's -- an
    # explicit ``plane=`` need not be perpendicular to a lattice vector at all.
    projections = np.abs(at @ (normal @ at))
    moving = projections[projections > 1.0e-8]
    reach = float(moving.min()) if moving.size else float(projections.max())
    if heights is None:
        heights = (0.0, reach)
    lo, hi = float(heights[0]), float(heights[1])
    if not hi > lo:
        raise ValueError(f"the scan must have hi > lo, got {heights}")
    if int(nheights) < 2:
        raise ValueError(f"a scan needs at least two planes, got {nheights}")
    if hi - lo > reach + 1.0e-8:
        # **The tunnelling density is periodic and a long scan finds the next
        # slab.** Withdrawing the tip past the cell brings it up underneath the
        # image of the surface it started on, where the density rises again --
        # so the outermost crossing is the wrong one, or there is none at all
        # and the pixel comes back ``nan``. It is silent and it looks like a
        # set-point that is merely too low.
        raise ValueError(
            f"the scan spans {hi - lo:.3f} bohr and the cell is only "
            f"{reach:.3f} bohr along this normal: past that the tip is under "
            "the periodic image of the surface and the density rises again"
        )

    scan = np.linspace(lo, hi, int(nheights))
    points = np.concatenate([geometry.offset(z * normal).flat() for z in scan])
    values = sample_coefficients(coefficients, dense, points)
    values = values.reshape((scan.shape[0],) + geometry.shape)

    corrugation = constant_current_height(scan, values, current)
    missed = int(np.isnan(corrugation).sum())
    if missed:
        warnings.warn(
            f"{missed} of {corrugation.size} points never cross the set-point "
            f"{current:g} inside the scan and come back nan: the tunnelling "
            "density is below it everywhere there (raise the set-point or "
            "start closer in), or above it everywhere (lower it)",
            stacklevel=3,
        )
    return STMImage(
        values=np.full(geometry.shape, float(current)),
        plane=geometry,
        heights=corrugation,
        heights_bohr=corrugation,
    )


# --------------------------------------------------------------------------
# what the image needs from the run, and what it refuses
# --------------------------------------------------------------------------


def _tip_energy(energy, levels, bias):
    """The energy the delta or the window edge sits at, in Ry."""
    if energy is not None:
        return float(energy)
    fermi = levels.get("fermi_energy")
    if fermi is not None:
        return float(fermi)
    homo, lumo = levels.get("homo"), levels.get("lumo")
    if homo is None:
        raise ValueError(
            "this run has no Fermi level and no HOMO to put the tip at: pass "
            "energy= in Ry"
        )
    if lumo is None:
        raise ValueError(
            "this run fills every band it has, so there is no gap to place a "
            "tip energy in: add empty bands (nbnd), or pass energy= in Ry"
        )
    midgap = 0.5 * (float(homo) + float(lumo))
    if bias is None:
        warnings.warn(
            "a zero-bias image of a gapped run is identically zero: the delta "
            f"sits at midgap ({midgap:.4f} Ry) where there are no states. Pass "
            "bias= to image the states on one side of the gap, which is what "
            "an experiment does",
            stacklevel=3,
        )
    else:
        warnings.warn(
            "this run has fixed occupations, so there is no Fermi level: the "
            f"tip energy is the middle of the gap, {midgap:.4f} Ry, which is "
            "stm.f90's own rule",
            stacklevel=3,
        )
    return midgap


def _tip_width(width, system):
    if width is not None:
        if not width > 0.0:
            raise ValueError(f"the smearing width must be positive, got {width}")
        return float(width)
    degauss = float(getattr(system, "degauss", 0.0) or 0.0)
    return degauss if degauss > 0.0 else INSULATOR_WIDTH


def _refuse_what_has_no_fermi_level(system, result):
    """The combinations whose energy selection is not one number.

    Every one of them is a statement about what "the states at the tip energy"
    means, not about the plane or the sampling.
    """
    if getattr(system, "spiral_q", None) is not None:
        raise NotImplementedError(
            "an STM image of a spin spiral is refused: the two spinor "
            "components live on different plane-wave spheres, so |psi(r)|^2 is "
            "not the lattice-periodic object this sum builds"
        )
    if getattr(result, "fermi_energy_up", None) is not None:
        raise NotImplementedError(
            "an STM image with a constrained tot_magnetization is refused: the "
            "two channels have their own Fermi levels and a tip sees one "
            "energy. Pass energy= only if the two levels are known to coincide"
        )
    if getattr(result, "magnetic_field", None) is not None:
        raise NotImplementedError(
            "an STM image of a run with an applied magnetic field is refused: "
            "the field's energy is outside the reported total (PLAN.md P18), "
            "so the Fermi level this would put the tip at is not the "
            "field-free one the image would be read as"
        )
