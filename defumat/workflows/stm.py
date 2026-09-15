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
from defumat.basis.gvectors import refuse_gamma_storage
from defumat.basis.fft import r_to_g
from defumat.basis.sample import sample_coefficients, sample_wavefunctions
from defumat.scf.driver import Calculation, gamma_storage_is_consumable
from defumat.stm.image import (
    STMImage,
    constant_current_height,
    project_spin,
    tunnelling_weights,
)
from defumat.stm.spectrum import (
    STMSpectrum,
    accumulate_spectrum,
    spectrum_weights,
)
from defumat.stm.plane import PlotPlane, plot_plane
from defumat.workflows.nscf import denser_grid, fixed_density_states

__all__ = ["run_stm", "run_sts", "sample_spectrum",
           "refuse_an_ultracell_result"]

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
    refuse_an_ultracell_result(
        result, "an STM image", "defumat.workflows.ultracell.run_ultracell_stm")
    _refuse_what_has_no_fermi_level(system, result)
    # A real-space wavefunction from a half sphere loses the conjugate half and
    # gains a spurious imaginary part, and nothing downstream notices.
    #
    # **The test is whether the run *consumed* the storage, not whether the
    # input asked for it.** An ultrasoft or symmetric ``K_POINTS gamma`` run is
    # substituted to an explicit k = 0 before the SCF starts
    # (``_without_gamma_storage``), so its states are on the whole sphere and
    # there is nothing here to refuse; reading ``system.kpoints.gamma_only``
    # would stop a run whose wavefunctions are perfectly good.
    refuse_gamma_storage(
        gamma_storage_is_consumable(system, pseudos), "an STM image",
        "psi(r) is evaluated as a bare sum over the stored k + G list "
        "(basis/sample.py)",
    )

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
            lambda points: sample_coefficients(coefficients, dense, points),
            geometry, calculation.system.cell, current, heights, nheights,
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


def run_sts(
    system,
    pseudos,
    result,
    *,
    energies=None,
    height: float | None = None,
    axis: int = 2,
    plane: tuple | PlotPlane | None = None,
    shape: tuple[int, int] | None = None,
    tip=None,
    spin=None,
    polarization: float = 1.0,
    bias: float | None = None,
    band_cutoff: float | None = None,
    width: float | None = None,
    smearing: str = "gaussian",
    mode: str = "constant-height",
    grid=None,
    shift=None,
    kpoints=None,
    nbnd: int | None = None,
    conv_thr: float | None = None,
    k_batch: int | None | str = "default",
) -> STMSpectrum:
    """``dI/dV(r, V)``: a tunnelling spectrum at a point, a line or a plane.

    :func:`run_stm` gives the local density of states at one tip energy, which
    is one picture at one bias. This is the other section of the same function,
    the curve at one place over many biases, which is what scanning tunnelling
    spectroscopy measures and what resolves a gap, a band edge or a defect state
    where an image only shows where the charge is.

    Args:
        system, pseudos, result: as :func:`run_stm`.
        energies: ``(nE,)`` the tip energies in Ry, ``E_F + V``. There is no
            default: the axis *is* the measurement, and a range has no natural
            width.
        tip: ``(npoints, 3)`` crystal coordinates of the tip, the usual form of
            this quantity -- one point is a spectrum and a row of them is a line
            cut. Mutually exclusive with ``height``/``plane``.
        height, axis, plane, shape: a whole plane instead, giving a map at
            every energy. ``shape`` defaults to a coarse ``(24, 24)``, because
            a spectrum over a fine map is a large array rather than a slow one.
        spin, polarization: a magnetic tip, exactly :func:`run_stm`'s.
        bias: refused here. An energy axis is the derivative of the window, so
            the window is :attr:`~defumat.stm.spectrum.STMSpectrum.current`.
        band_cutoff, width, smearing: the smeared delta, as in :func:`run_stm`.
        mode: ``"constant-height"`` only; a constant-current spectrum moves the
            tip as the bias is swept and is a different experiment.
        grid, shift, kpoints, nbnd, conv_thr: re-solve the bands at fixed
            density on a **complete** k-set first, which a spectrum wants more
            than an image does -- every feature in a dI/dV curve is a band edge,
            and a handful of k-points puts edges where the k-set happens to have
            states. It is :func:`~defumat.workflows.transport.whole_grid` rather
            than :func:`~defumat.workflows.nscf.denser_grid`, for the reason the
            wedge is refused below.
        k_batch: the k-axis batching dial, for the band solve.

    Returns an :class:`~defumat.stm.spectrum.STMSpectrum`.
    """
    refuse_an_ultracell_result(
        result, "a tunnelling spectrum",
        "defumat.workflows.ultracell.run_ultracell_sts")
    _refuse_what_has_no_fermi_level(system, result)
    refuse_gamma_storage(
        gamma_storage_is_consumable(system, pseudos), "a tunnelling spectrum",
        "psi(r) is evaluated as a bare sum over the stored k + G list "
        "(basis/sample.py)",
    )
    _refuse_a_window_under_an_axis(bias, mode)

    from defumat.workflows.transport import (
        _geometry, _refuse_an_augmented_plane, _tip_points, whole_grid,
    )

    if kpoints is None and grid is not None:
        kpoints = whole_grid(system, grid, shift)
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
            "a tunnelling spectrum is built from the wavefunctions and this "
            "result carries none: run the SCF without discarding them, or pass "
            "a grid so the bands are re-solved"
        )

    _refuse_a_reduced_k_set(calculation.system)
    axis_energies, fermi = _spectrum_energies(energies, levels)
    width = _tip_width(width, system)
    used = calculation.system
    geometry, points = _tip_points(used.cell, height, axis, plane,
                                   shape or (24, 24), tip)
    _refuse_an_augmented_plane(
        used, pseudos, axis, np.unique(np.round(points[:, axis], 10)),
        "a tunnelling spectrum's tip")
    channels, dos = sample_spectrum(
        _geometry(calculation), wavefunctions, eigenvalues, points,
        energies=axis_energies, width=width, smearing=smearing, bias=bias,
        band_cutoff=band_cutoff, nspin_mag=int(used.nspin_mag),
    )
    return _finish_spectrum(
        channels, dos, axis_energies, points, geometry, spin, polarization,
        width=width, smearing=smearing, bias=bias, fermi=fermi,
        grid=grid, supercell=None,
    )


def _finish_spectrum(channels, dos, energies, points, geometry, spin,
                     polarization, *, width, smearing, bias, fermi, grid=None,
                     supercell=None):
    """The spin projection, the sum rule and the shape, for either route.

    **The integral is exact rather than sampled**, which is the one place the
    spectrum differs from the image: an image integrates its own field over the
    grid, and here there is no field, only its value at the tip points. What
    there is instead is orthonormality -- every state integrates to 1 over the
    cell it is normalised in -- so the integral of ``dI/dV`` over that cell is
    the sum of the weights, exactly. It is bookkeeping and not a measurement,
    and what measures it is the test against ``compute_dos``.

    **It is the charge's integral whatever the tip is**, and that is a real
    limitation rather than a convention: a polarized tip's own field integrates
    to ``[D(E) + P n.M(E)]/2``, and while the collinear ``M`` is a difference of
    two weight sums, the transverse one is a spin expectation value and no sum
    of weights gives it. One quantity that is exact in every regime is better
    than one that is exact in two of the three.
    """
    cells = 1 if supercell is None else int(np.prod(supercell))
    if spin is None:
        values = (channels[0] + channels[1] if channels.shape[0] == 2
                  else channels[0])
    else:
        values = project_spin(channels, spin, polarization)
    shaped = geometry.shape if geometry is not None else (points.shape[0],)
    by_spin = (None if channels.shape[0] == 1 else
               channels.reshape(channels.shape[:2] + shaped))
    return STMSpectrum(
        values=values.reshape((values.shape[0],) + shaped),
        energies=np.asarray(energies, dtype=float),
        points=points,
        plane=geometry,
        values_by_spin=by_spin,
        integral=np.asarray(dos, dtype=float) / cells,
        spin=None if spin is None else (
            spin if isinstance(spin, str)
            else tuple(float(c) for c in np.ravel(spin))),
        polarization=float(polarization),
        width=float(width),
        smearing=smearing,
        bias=None if bias is None else float(bias),
        fermi_energy=None if fermi is None else float(fermi),
        supercell=supercell,
        notes={"grid": None if grid is None else tuple(int(n) for n in grid),
               "cells": cells},
    )


# --------------------------------------------------------------------------
# the spectrum: the same sum, sectioned the other way
# --------------------------------------------------------------------------


def sample_spectrum(geometry, wavefunctions, eigenvalues, points, *,
                    energies, width, smearing="gaussian", bias=None,
                    band_cutoff=None, nspin_mag=1):
    """``(nspin_mag, nE, npoints)``: dI/dV at every point and every energy.

    The counterpart of :func:`defumat.workflows.transport._assemble`, and it
    takes the same bundle for the same reason -- an ordinary run and an
    ultracell (``PLAN.md`` P89) differ only in which sphere, cell and k-points
    the bands live on, and neither this contraction nor the sampler has to know
    which it was handed.

    **The states are sampled once and the energy axis is a matrix product.**
    ``psi_n(r)`` does not depend on the energy, so what depends on it is the
    per-state weight alone: sampling is ``nk nbnd npoints npw`` and is paid
    once, and each energy after that costs one ``(nE, nbnd) x (nbnd, npoints)``
    contraction. That is the whole reason a spectrum is not a loop over
    :func:`run_stm`, which would rebuild the density from every state at every
    energy.

    Args:
        geometry: the sphere, the cell and the k-points
            (:class:`~defumat.transport.green.TransportGeometry`).
        wavefunctions: indexable as ``[ispin, ik]``, giving
            ``(nbnd, npol npwx)`` -- an array, or an
            :class:`~defumat.ultracell.states.UltracellStates`.
        eigenvalues: ``(nspin, nk, nbnd)`` in Ry.
        points: ``(npoints, 3)`` crystal coordinates of the tip, in the basis
            of ``geometry.cell``.
        nspin_mag: 1 for a charge alone, 2 for the two collinear channels, 4
            for ``(n, m_x, m_y, m_z)``.

    **There is no k dial here and there is nothing for one to bound**, which is
    the difference from :func:`~defumat.workflows.transport._assemble`: that
    one holds ``(npol, nk, nbnd, npoints)`` amplitudes at once because the
    contraction needs a whole chunk of them together, and each k-point here is
    sampled and added straight into the total, so the peak is one k-point's
    ``(nbnd, npol, npoints)`` complex whatever is asked. On an ultracell the
    larger term is the state block itself, ``N^2 nbnd npwx npol`` complex
    (:class:`~defumat.ultracell.states.UltracellStates`), and it is also per
    ``k0``.

    The volume the states are normalised over is ``geometry.cell``'s, so an
    ultracell's weights go in **undivided**: the ``N`` sits in ``Omega_u``
    there, where the density route puts it in the occupations instead
    (:class:`~defumat.ultracell.states.UltracellStates`).

    Returns ``(channels, dos)`` -- the spectrum and ``D(E)`` of the states it
    was summed from, in states/Ry over the whole of ``geometry.cell``.
    """
    miller = np.asarray(geometry.miller)
    mask = np.asarray(geometry.mask)
    kcrystal = np.asarray(geometry.kcrystal)
    kweights = np.asarray(geometry.kweights, dtype=float)
    volume = geometry.volume
    npol = int(geometry.npol)
    npwx = geometry.npwx

    eigenvalues = np.asarray(eigenvalues, dtype=float)
    nspin, nk, nbnd = eigenvalues.shape
    weights = spectrum_weights(eigenvalues, kweights, energies, width=width,
                               smearing=smearing, bias=bias,
                               band_cutoff=band_cutoff)
    points = np.atleast_2d(np.asarray(points, dtype=float))
    total = np.zeros((int(nspin_mag), weights.shape[0], points.shape[0]))
    # ``D(E)`` of these states, which is what the spectrum integrates to over
    # the cell the states are normalised in -- exactly, by orthonormality, so
    # it is carried out of here rather than sampled back.
    density_of_states = weights.reshape(weights.shape[0], -1).sum(axis=1)

    # A collinear run is two independent channels and a spinor state carries
    # every component itself, which is the one structural difference between
    # the two and the same split ``ultracell_band_density`` makes.
    channels = None if npol == 2 else (None if int(nspin_mag) == 1 else 0)
    for ispin in range(nspin):
        channel = None if channels is None else ispin
        for ik in range(nk):
            block = np.asarray(wavefunctions[ispin, ik])
            sampled = sample_wavefunctions(
                block.reshape((nbnd, npol, npwx)), miller[ik],
                kcrystal[ik], points, volume, mask=mask[ik],
            )
            accumulate_spectrum(total, weights[:, ispin, ik], sampled,
                                channel=channel, nspin_mag=int(nspin_mag))
    return total, density_of_states


def _spectrum_energies(energies, levels):
    """The axis, and the zero the bias is measured from.

    **A spectrum needs that zero where an image does not**, which is the one
    thing here that is not shared with :func:`run_stm`: an image is taken at an
    energy and reports it, while ``dI/dV`` is plotted against a *bias* and
    ``I(V)`` is integrated outwards from ``V = 0``, so a spectrum with no zero
    has an axis with no origin and a current that starts nowhere. The zero is
    the Fermi level, and for a fixed-occupation run it is the middle of the gap
    -- the same fallback :func:`_tip_energy` and
    :func:`defumat.workflows.transport._energies` already take, said once here
    so that a run without a Fermi level gets a defined ``V = 0`` rather than
    0 Ry, which is a point in the middle of the valence band and is where an
    unset reference silently puts it.
    """
    if energies is None:
        raise ValueError(
            "a tunnelling spectrum needs an energy axis and there is no "
            "default width for one: pass energies= in Ry, usually the Fermi "
            "level plus a range of biases. An image at one energy is run_stm"
        )
    axis = np.atleast_1d(np.asarray(energies, dtype=float))
    if axis.ndim != 1 or axis.size == 0:
        raise ValueError(
            f"the energies are a one-dimensional axis, got shape {axis.shape}")

    fermi = levels.get("fermi_energy")
    if fermi is not None:
        return axis, float(fermi)
    homo, lumo = levels.get("homo"), levels.get("lumo")
    if homo is None or lumo is None:
        warnings.warn(
            "this run has no Fermi level and no gap to put one in, so the "
            "spectrum carries no zero of bias: STMSpectrum.bias_axis is then "
            "the energies themselves and STMSpectrum.current is refused. Pass "
            "energies= measured from wherever the tip is referenced",
            stacklevel=3,
        )
        return axis, None
    midgap = 0.5 * (float(homo) + float(lumo))
    warnings.warn(
        "this run has fixed occupations, so there is no Fermi level: the zero "
        f"of the bias axis is the middle of the gap, {midgap:.4f} Ry, which is "
        "stm.f90's own rule. Every dI/dV value is unaffected -- it is the bias "
        "axis and the current that are measured from it",
        stacklevel=3,
    )
    return axis, midgap


def _refuse_a_reduced_k_set(system):
    """A wedge sum of ``|psi_k(r)|^2`` is not the local density of states.

    **This is the one thing a spectrum does not inherit from an image**, and it
    is the wedge trap ``CLAUDE.md`` lists. :func:`run_stm` sums through
    :meth:`~defumat.scf.driver.Calculation.density`, which **symmetrises**, so a
    reduced k-set gives it the whole zone's answer; the amplitude route adds
    ``|psi_k(r)|^2`` point by point with nothing to symmetrise it, so on a wedge
    it returns the sum over that wedge alone -- a plausible, smooth, wrong
    density of states everywhere off a symmetry axis. Unfolding is not the
    escape it is for a scalar: unfolding a *wavefunction* means rotating it,
    which is :func:`~defumat.workflows.transport.whole_grid`'s own argument.

    The test is the k-weights, exactly as
    :func:`~defumat.workflows.transport._refuse_a_k_set_this_cannot_sum` tests
    them, and it is a sufficient condition rather than a necessary one: a
    reduced set whose orbits all happen to have the same size would pass it.
    What makes that acceptable is that it is the *reduction* that varies the
    weights on every real k-set, and a run meaning to do this passes
    ``nosym = .true.`` or ``grid=``, both of which are complete by construction.
    """
    weights = np.asarray(system.kpoints.weights, dtype=float)
    if weights.size > 1 and np.ptp(weights) > 1.0e-8 * np.abs(weights).max():
        raise NotImplementedError(
            "a tunnelling spectrum on a symmetry-reduced k-set is refused: "
            "psi(r) is sampled and squared point by point, so the sum is over "
            "the wedge alone and nothing symmetrises it -- unlike an image, "
            "which goes through the density and is symmetrised there. Pass "
            "grid= to re-solve on the whole grid, or run with nosym = .true."
        )


def _refuse_a_window_under_an_axis(bias, mode):
    """Two knobs an energy axis turns into a different experiment."""
    if bias is not None:
        raise ValueError(
            "bias= integrates the states in a window and an energy axis is "
            "the derivative of exactly that integral, so asking for both is "
            "asking for I(V) twice: run without bias= and read "
            "STMSpectrum.current, which is the same window count and is "
            "carried rather than recomputed"
        )
    if mode != "constant-height":
        raise NotImplementedError(
            f"a spectrum in {mode!r} mode is refused: a constant-current "
            "spectrum re-inverts the scan at every energy, which is a "
            "different experiment (the tip moves as the bias is swept) and a "
            "different cost. Take the spectrum at fixed height, which is what "
            "dI/dV spectroscopy is"
        )


# --------------------------------------------------------------------------
# the plane, and the scan built out of it
# --------------------------------------------------------------------------


def _plane(cell, height, axis, plane, shape, span=(1.0, 1.0, 1.0)):
    """Either the ``height`` shortcut or Elk's three corners.

    ``span`` is how far the shortcut's plane reaches along each lattice vector,
    in crystal coordinates. One cell everywhere is an ordinary surface image;
    an ultracell image spans ``n_i``, because the modulation is the thing being
    looked at (:func:`defumat.workflows.ultracell.run_ultracell_stm`).
    """
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
    edge1[first] = float(span[first])
    edge2[second] = float(span[second])
    return plot_plane(cell, origin, edge1, edge2, shape)


def _constant_current(sample, geometry, cell, current, heights, nheights):
    """Scan the plane outwards and invert for the height at the set-point.

    ``sample`` takes ``(np, 3)`` crystal coordinates of ``cell`` and returns the
    tunnelling density there. It is a callable rather than the pair
    ``(coefficients, G-set)`` because an ultracell image samples the same field
    on the ultracell's own ``G + Q`` set and in its own coordinates, and
    everything else about the scan -- how far the tip may be withdrawn, where
    the set-point is crossed -- is a statement about the cell it is scanning
    over.
    """
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
    values = sample(points)
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


def refuse_an_ultracell_result(result, quantity: str, instead: str):
    """An ultracell result is not a ground state and says so by name.

    It has no wavefunctions on the unit cell's spheres and no eigenvalues on its
    k-set -- its states live on the ultracell's own sphere at ``k0``
    (:mod:`defumat.ultracell.states`) -- so what would otherwise happen here is
    an ``AttributeError`` three lines in, which says nothing about why. The
    quantity itself exists and is one function away, which is what the message
    is for.
    """
    if getattr(result, "ultracell", None) is None:
        return
    raise NotImplementedError(
        f"{quantity} of an ultracell is {instead}, not this: an "
        "UltracellResult carries states on the ultracell's own plane-wave "
        "sphere at k0 rather than wavefunctions on the unit cell's, so nothing "
        "here can read it. Pass the unit cell's own SCF result to image the "
        "unmodulated crystal"
    )


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
