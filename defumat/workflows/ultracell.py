"""What a modulation looks like to a tip: an ultracell STM image and transmission.

``PLAN.md`` P89. P88 converges a density, a magnetization and a potential over
``N`` unit cells at once; P65 and P66 are the two things a tip measures above a
surface, a Tersoff-Hamann image and a vertical tunnelling transmission. This is
the join, and it exists because the observable a charge or spin density wave is
actually *seen* with is an STM image, which a run in the unit cell's own states
could not produce.

**Nothing new is computed.** An ultracell state is one plane-wave vector on the
ultracell's own sphere (:mod:`defumat.ultracell.states`), so the image is
:func:`~defumat.ultracell.states.ultracell_band_density` with a smeared delta at
the tip energy in place of the occupations -- which is what an STM image *is* in
Elk's ``wfplot.f90`` and in QE's ``stm.f90`` -- and the transmission is
:mod:`defumat.transport`'s own contraction handed the ultracell's Miller
indices, cell and ``k0`` instead of the unit cell's.

**The two coordinate conventions, which are the only wart here.** A plane is
given in **unit-cell** crystal coordinates running over ``[0, n_i)`` across the
ultracell -- the convention ``external=`` already uses, so ``height = 0.8``
means the same thing it means in a unit-cell run and a map spans the whole
modulation by default. The sampling then happens in the ultracell's own
coordinates, which is one division by ``n_i`` at the boundary and nothing a
caller sees.

**What the k-set is.** The image is built on the ``k0`` mesh the ultracell ran
on, and there is no ``grid=`` here: re-solving on a denser k-set is not
post-processing for an ultracell, it is another run of the whole loop. Pass a
larger ``kgrid`` to :func:`~defumat.ultracell.driver.run_ultracell` instead.
"""

from __future__ import annotations

import numpy as np

from defumat.basis.fft import r_to_g
from defumat.basis.sample import sample_miller
from defumat.stm.image import STMImage, project_spin, tunnelling_weights
from defumat.stm.spectrum import STMSpectrum
from defumat.stm.plane import PlotPlane
from defumat.transport.green import (
    TransportGeometry,
    VerticalTransport,
)
from defumat.workflows.stm import (
    _constant_current,
    _finish_spectrum,
    _plane,
    _refuse_a_window_under_an_axis,
    _spectrum_energies,
    _tip_energy,
    _tip_width,
    sample_spectrum,
)
from defumat.workflows.transport import (
    DEFAULT_BROADENING,
    _assemble,
    _energies,
    _label,
    _tip_points,
    _warn_if_the_slab_is_not_between,
)

__all__ = ["run_ultracell_stm", "run_ultracell_sts",
           "run_ultracell_transport"]


def run_ultracell_stm(
    system,
    pseudos,
    result,
    *,
    height: float | None = None,
    axis: int = 2,
    plane: tuple | PlotPlane | None = None,
    shape: tuple[int, int] = (60, 40),
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
    state_batch: int | None = 1,
) -> STMImage:
    """A Tersoff-Hamann image of a converged ultracell.

    The tunnelling density of states of a modulated crystal: what an experiment
    sees above a charge density wave, a spin density wave, a screened impurity
    or a domain wall, none of which a unit cell can carry.

    Args:
        system, pseudos: the **unit cell's**, the ones the ultracell ran on.
        result: the :class:`~defumat.ultracell.driver.UltracellResult`. Its
            ``states`` are needed, not only its density, so the run must have
            kept them (``keep_states``, the default).
        height: crystal coordinate of the tip plane along ``axis``, in the
            **unit cell's** units -- the same number a unit-cell image takes.
            The plane spans the whole ultracell laterally, which is where the
            modulation is.
        axis: which lattice vector ``height`` measures along, 0, 1 or 2.
        plane: the general form, ``(origin, edge1, edge2)`` in unit-cell
            crystal coordinates, which run over ``[0, n_i)`` across the
            ultracell. Mutually exclusive with ``height``.
        shape: the sampling. Wider than a unit-cell image's by default along
            the first edge, because an ultracell is longer than it is wide.
        mode: ``"constant-height"`` or ``"constant-current"``.
        current: the set-point for constant-current, in the units of the
            tunnelling density -- 1/(bohr^3 Ry) at zero bias, electrons/bohr^3
            in a window.
        heights, nheights: the scan along the surface normal, in bohr.
        spin: image one spin channel -- a magnetic tip. A cartesian 3-vector or
            ``"up"``/``"down"``; ``None`` is the charge image. This is the one
            an ultracell run is usually for: a spin density wave lives in the
            magnetization, and a collinear crystal is unchanged by flipping
            every spin together with the sign of the field, so the charge
            cannot respond at first order and a nonmagnetic tip sees the wave
            **squared** instead, at twice its wavevector. Measured on an
            eight-cell silicon cell carrying 0.12 Bohr magnetons: 82 per cent
            of its mean from cell to cell for a polarized tip at one period,
            37 per cent for a plain one at two, and 1.9e-4 in the charge
            *density* itself.
        polarization: the tip's spin polarization in ``[-1, 1]``.
        bias: the sample bias in Ry. ``None`` is the zero-bias delta.
        band_cutoff: drop states further than this many widths outside the
            window. Off by default, as in :func:`~defumat.workflows.stm.run_stm`.
        energy: the energy the tip is tuned to, in Ry. Defaults to the
            ultracell's Fermi level, or to the middle of its gap.
        width: the smeared delta's width in Ry.
        smearing: which smeared delta.
        state_batch: ultracell states in flight at once, the dial
            :func:`~defumat.ultracell.driver.run_ultracell` uses for the same
            reason -- each one holds a whole ultracell box.

    Returns an :class:`~defumat.stm.image.STMImage` whose ``integral`` is the
    tunnelling density of states **per unit cell**, so that it is the same
    number a unit-cell image reports and the two can be compared directly.
    """
    states = _states_of(result, "an STM image")
    _refuse_what_has_no_tip_energy(system, result)
    ultracell = states.ultracell
    cells = ultracell.cells

    eigenvalues = np.asarray(states.eigenvalues)
    levels = _levels(system, result, states)
    energy = _tip_energy(energy, levels, bias)
    width = _tip_width(width, system)

    # **The occupations of the loop, divided by ``N`` exactly as they were.**
    # ``_occupy`` hands the density ``w_k0 f_j / N`` so that what comes out is
    # per unit cell; a tunnelling weight is on the same scale for the same
    # reason, and the ``N`` here is the whole of the difference between an
    # image that integrates to ``D(E)`` per cell and one that integrates to
    # ``N D(E)``.
    weights = tunnelling_weights(
        eigenvalues, np.asarray(states.weights), energy=energy, width=width,
        smearing=smearing, bias=bias, band_cutoff=band_cutoff,
    ) / cells

    nspin_mag = int(np.asarray(result.density).shape[0])
    density = np.asarray(states.density(weights, nspin_mag=nspin_mag,
                                        batch=state_batch))

    volume = float(result.cell_volume)
    if spin is None:
        field = density[0] + density[1] if density.shape[0] == 2 else density[0]
    else:
        field = project_spin(density, spin, polarization)
    integral = float(field.mean() * volume)

    geometry = _plane(system.cell, height, axis, plane, shape,
                      span=ultracell.shape)
    coefficients, miller = _box_coefficients(field, states, system)
    channels = (_box_coefficients(density, states, system)[0]
                if density.shape[0] > 1 else None)

    scale = np.asarray(ultracell.shape, dtype=float)
    if mode == "constant-height":
        values = sample_miller(coefficients, miller, geometry.flat() / scale)
        values = values.reshape(geometry.shape)
        by_spin = None if channels is None else sample_miller(
            channels, miller, geometry.flat() / scale
        ).reshape((density.shape[0],) + geometry.shape)
        image = STMImage(values=values, plane=geometry, values_by_spin=by_spin)
    elif mode == "constant-current":
        if current is None:
            raise ValueError(
                "constant-current mode needs a set-point: pass current=, in "
                + ("1/(bohr^3 Ry)" if bias is None else "electrons/bohr^3")
            )
        image = _constant_current(
            lambda points: sample_miller(coefficients, miller, points / scale),
            # The *unit* cell, and that is not an oversight: how far the tip
            # may be withdrawn before it meets the periodic image of the
            # surface is set by the surface's own period, which an ultracell
            # modulated along the surface does not change.
            geometry, system.cell, current, heights, nheights,
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
    image.supercell = tuple(int(n) for n in ultracell.shape)
    return image


def run_ultracell_transport(
    system,
    pseudos,
    result,
    *,
    exit_height: float,
    exit_axis: int = 2,
    height: float | None = None,
    axis: int | None = None,
    plane: tuple | PlotPlane | None = None,
    shape: tuple[int, int] = (60, 40),
    tip=None,
    energies=None,
    bias: float | None = None,
    nenergies: int = 1,
    broadening: float = DEFAULT_BROADENING,
    method: str = "spectral",
    smearing: str = "gaussian",
    spin=None,
    polarization: float = 1.0,
    tip_spin=None,
    tip_polarization: float = 1.0,
    incoherent: bool = True,
    exit_region: str = "plane",
    k_batch: int | None = 1,
) -> VerticalTransport:
    """``T(r; E)`` through a modulated two-dimensional material.

    P66's junction with an ultracell in it: the electron enters at a point above
    the sheet and leaves into an infinite plane below, and what decides the
    current is the nonlocal Green's function between the two. Where the sheet
    carries a modulation -- a charge density wave, a domain wall, a moire
    period -- the map is a map *of the modulation*, and it departs from the
    image :func:`run_ultracell_stm` gives by the interference between states
    degenerate at the tip energy, which is reported beside it.

    **Everything is P66's**, including the contraction, the channel basis, the
    two spin polarizers and the diagnostics: what changes is the geometry the
    bands live on, which is the ultracell's sphere, its cell and the ultracell
    Brillouin zone's ``k0`` (:class:`~defumat.transport.green.TransportGeometry`).

    Args:
        system, pseudos: the **unit cell's**.
        result: the :class:`~defumat.ultracell.driver.UltracellResult`, with its
            states kept.
        exit_height: the substrate plane's crystal coordinate along
            ``exit_axis``. It is the same number in both cells, because an
            ultracell modulated along the stacking axis is refused below.
        exit_axis: the stacking axis, 2 for an ordinary slab.
        height, axis, plane, shape, tip: the tip plane, in **unit-cell** crystal
            coordinates running over ``[0, n_i)`` -- :func:`run_ultracell_stm`'s
            convention, so a map spans the whole modulation.
        energies, bias, nenergies, broadening, method, smearing: the energy
            selection, exactly :func:`~defumat.workflows.transport.run_vertical_transport`'s.
        spin, polarization: the substrate's spin acceptance; ``tip_spin`` and
            ``tip_polarization`` the tip's. Both leads may be magnetic, and with
            both the map depends on the angle between them.
        incoherent: report the interference-free map beside the coherent one.
        exit_region: ``"plane"``, the substrate, or ``"volume"``, which widens
            the exit region to the whole ultracell and must then reproduce
            :func:`run_ultracell_stm` exactly -- the Tersoff-Hamann limit, and
            the check that the two normalisations agree.
        k_batch: how many ``k0`` points' amplitudes are held at once. It bounds
            a **host** array, ``(npol, k_batch, N nbnd, npoints)`` with another
            of the same size beside it inside the contraction, and an ultracell
            has ``N`` times as many states per k-point as a unit cell does: a
            40x40 map over 630 ultracell states is 16 MB per ``k0``, twice
            over. One at a time by default, which is the whole subpackage's end
            of that trade (``state_batch``, and the Python ``k0`` loop).
    """
    states = _states_of(result, "a vertical transmission")
    if exit_axis not in (0, 1, 2):
        raise ValueError(f"exit_axis must be 0, 1 or 2, got {exit_axis}")
    if exit_region not in ("plane", "volume"):
        raise ValueError(
            f"unknown exit_region {exit_region!r}: use 'plane' (the substrate) "
            "or 'volume' (the Tersoff-Hamann diagnostic)"
        )
    if axis is None:
        axis = exit_axis
    _refuse_what_has_no_tip_energy(system, result)
    _refuse_a_stacked_ultracell(states, exit_axis)
    _refuse_a_k_set_this_cannot_sum(states, exit_axis)

    ultracell = states.ultracell
    scale = np.asarray(ultracell.shape, dtype=float)
    geometry, points = _tip_points(system.cell, height, axis, plane, shape, tip,
                                   span=ultracell.shape)
    _warn_if_the_slab_is_not_between(system, exit_axis, exit_height, points, axis)

    levels = _levels(system, result, states)
    grid_energies = _energies(energies, levels, bias, nenergies,
                              float(broadening))

    # **The exit plane's height needs no conversion and the tip points do.**
    # The ultracell is one cell deep along the stacking axis, so a crystal
    # coordinate along it is the same number in both; laterally it is ``n_i``
    # cells, so a point given over ``[0, n_i)`` is ``s / n_i`` in the
    # coordinates the ultracell's own Miller indices are written against.
    values, extras = _assemble(
        _ultracell_geometry(states), states, np.asarray(states.eigenvalues),
        points / scale,
        exit_height=float(exit_height), exit_axis=exit_axis,
        energies=grid_energies, broadening=float(broadening),
        spin=spin, polarization=float(polarization),
        tip_spin=tip_spin, tip_polarization=float(tip_polarization),
        incoherent=bool(incoherent), exit_region=exit_region,
        method=method, smearing=smearing, k_batch=k_batch,
    )

    if bias is not None:
        values = {key: np.trapezoid(array, grid_energies, axis=0)[None]
                  for key, array in values.items()}

    shaped = geometry.shape if geometry is not None else (points.shape[0],)
    coherent = values["coherent"].reshape((-1,) + shaped)
    incoherent_map = (None if "incoherent" not in values
                      else values["incoherent"].reshape((-1,) + shaped))
    if coherent.shape[0] == 1:
        coherent = coherent[0]
        if incoherent_map is not None:
            incoherent_map = incoherent_map[0]

    transport = VerticalTransport(
        values=coherent,
        plane=geometry,
        energies=np.atleast_1d(grid_energies if bias is None
                               else np.array([grid_energies[0]])),
        broadening=float(broadening),
        exit_height=float(exit_height),
        exit_axis=int(exit_axis),
        incoherent=incoherent_map,
        fermi_energy=levels.get("fermi_energy"),
        spin=_label(spin),
        polarization=float(polarization),
        tip_spin=_label(tip_spin),
        tip_polarization=float(tip_polarization),
        least_eigenvalue=extras["least_eigenvalue"],
        offdiagonal_weight=extras["offdiagonal_weight"],
        notes={**extras["notes"], "supercell": tuple(int(n) for n in ultracell.shape)},
    )
    return transport


def run_ultracell_sts(
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
) -> STMSpectrum:
    """``dI/dV(r, V)`` across a modulation: a tunnelling spectrum of an ultracell.

    :func:`run_ultracell_stm` gives one picture at one bias, and a modulated
    crystal is the case where that is least of what a tip measures: a charge
    density wave is a **gap that opens in antiphase with the charge maxima**, a
    spin density wave moves the two spin channels in opposite directions from
    cell to cell, and a domain wall carries a state inside the gap that lives
    nowhere else. All three are an energy axis at every position, and none of
    them shows in an image at a single energy.

    The sum is the image's, sectioned the other way: the ultracell states are
    sampled at the tip points **once** (:mod:`defumat.ultracell.states`, the
    relabelling P89 rests on) and the axis is then one matrix product per
    k-point, so ``nE`` energies cost barely more than one -- where a loop over
    :func:`run_ultracell_stm` would rebuild the whole ultracell density ``nE``
    times.

    Args:
        system, pseudos: the **unit cell's**, the ones the ultracell ran on.
        result: the :class:`~defumat.ultracell.driver.UltracellResult`, with its
            states kept.
        energies: ``(nE,)`` the tip energies in Ry. No default: the axis is the
            measurement.
        tip: ``(npoints, 3)`` tip positions in **unit-cell** crystal
            coordinates running over ``[0, n_i)``, which is
            :func:`run_ultracell_stm`'s convention. One point per cell along
            the modulation is the line cut this quantity is for.
        height, axis, plane, shape: a plane instead, giving a map at every
            energy. ``shape`` defaults to ``(24, 8)``, coarser than an image's:
            a spectrum over a fine map is ``nE`` times the array.
        spin, polarization: a magnetic tip, which is what a spin density wave
            needs for the reason :func:`run_ultracell_stm` gives.
        bias: refused under an axis; the window is
            :attr:`~defumat.stm.spectrum.STMSpectrum.current`.
        band_cutoff, width, smearing: the smeared delta.
        mode: ``"constant-height"`` only. There is no ``k_batch`` and nothing
            for one to bound: each ``k0`` is sampled and added straight into
            the total, so the peak is one ``k0``'s state block, ``N^2 nbnd
            npwx npol`` complex, which is the same one the loop itself pays.

    Returns an :class:`~defumat.stm.spectrum.STMSpectrum` whose ``integral`` is
    ``D(E)`` **per unit cell**, the same number a unit-cell spectrum reports.

    **What this does not inherit** is the transmission's two refusals: there is
    no exit plane here, so a modulation along the stacking axis and a ``k0``
    mesh with more than one division along it are both fine. A spectrum is
    available wherever an image is.
    """
    states = _states_of(result, "a tunnelling spectrum")
    _refuse_what_has_no_tip_energy(system, result)
    _refuse_a_window_under_an_axis(bias, mode)

    ultracell = states.ultracell
    scale = np.asarray(ultracell.shape, dtype=float)
    levels = _levels(system, result, states)
    axis_energies, fermi = _spectrum_energies(energies, levels)
    width = _tip_width(width, system)

    geometry, points = _tip_points(system.cell, height, axis, plane,
                                   shape or (24, 8), tip,
                                   span=ultracell.shape)
    nspin_mag = int(np.asarray(result.density).shape[0])
    channels, dos = sample_spectrum(
        _ultracell_geometry(states), states, np.asarray(states.eigenvalues),
        points / scale, energies=axis_energies, width=width, smearing=smearing,
        bias=bias, band_cutoff=band_cutoff, nspin_mag=nspin_mag,
    )
    return _finish_spectrum(
        channels, dos, axis_energies, points, geometry, spin, polarization,
        width=width, smearing=smearing, bias=bias, fermi=fermi,
        supercell=tuple(int(n) for n in ultracell.shape),
    )


def _ultracell_geometry(states) -> TransportGeometry:
    """Where an ultracell's bands live, as the transmission's own bundle.

    The overlap is ``None`` rather than a function, and that is exact: the
    ultracell refuses ultrasoft and PAW datasets, so ``S`` is the identity and
    ``sum_G c* c`` is orthonormality itself. The peak here is the stacked Miller
    indices, ``nk0 x N npwx x 3`` integers, which is small beside the one
    ``k0`` block :meth:`~defumat.ultracell.states.UltracellStates.block` builds
    inside the loop.
    """
    return TransportGeometry(
        miller=np.stack([states.miller(ik) for ik in range(states.nk0)]),
        mask=np.stack([states.mask(ik) for ik in range(states.nk0)]),
        kcrystal=states.kcrystal,
        kweights=np.asarray(states.weights, dtype=float),
        cell=states.ultracell_cell,
        npol=int(states.npol),
        apply_s=None,
    )


# --------------------------------------------------------------------------
# what an ultracell image needs from the run, and what it refuses
# --------------------------------------------------------------------------


def _states_of(result, quantity: str):
    """The states, or the refusal that says how to get them back."""
    states = getattr(result, "states", None)
    if states is None:
        raise ValueError(
            f"{quantity} of an ultracell is built from its states and this "
            "result carries none: run the ultracell again with "
            "keep_states=True (the default). They cannot be rebuilt from the "
            "result, because what makes them is the one diagonalisation on the "
            "folded k-set -- the expensive step of the whole method"
        )
    return states


def _refuse_what_has_no_tip_energy(system, result):
    """The one combination whose energy selection is not a single number.

    An applied ``magnetic_field`` is **not** refused here, where a unit-cell
    image refuses it: there the field is something the ground state was
    converged under and the reported total does not carry, so the Fermi level
    is not the one the image would be read against; here the field is the
    modulation's own driver, the ultracell eigenvalues are the eigenvalues
    under it, and the image is of that state -- which is the thing being asked
    for rather than a confusion about it.
    """
    if np.ndim(result.fermi_energy) != 0:
        raise NotImplementedError(
            "an STM image of an ultracell with a constrained tot_magnetization "
            "is refused: the two channels have their own Fermi levels and a tip "
            "sees one energy. Pass energy= only if the two are known to coincide"
        )


def _levels(system, result, states):
    """The Fermi level, or the two band edges, of the ultracell's own states.

    **A fixed-occupation run's ``fermi_energy`` is its HOMO**
    (:func:`~defumat.ultracell.driver._occupy` returns it there), and a tip
    needs the gap's other edge as well, so both are read off the occupations
    rather than taken from the one number. The test is exact where it matters:
    with fixed occupations a state is filled with the whole of its k-point's
    weight or with none of it, so half the row's maximum separates the two.
    """
    if system.occupations not in (None, "fixed"):
        return {"fermi_energy": float(result.fermi_energy)}
    eigenvalues = np.asarray(states.eigenvalues)
    occupations = np.asarray(result.occupations).reshape(eigenvalues.shape)
    filled = occupations > 0.5 * occupations.max(axis=-1, keepdims=True)
    homo = float(eigenvalues[filled].max()) if filled.any() else None
    empty = ~filled
    lumo = float(eigenvalues[empty].min()) if empty.any() else None
    return {"fermi_energy": None, "homo": homo, "lumo": lumo}


def _box_coefficients(field, states, system):
    """A field on the ultracell box, as coefficients on its ``G + Q`` sphere.

    The set is ``|G + Q|^2 <= ecutrho`` on the box
    (:meth:`~defumat.ultracell.grid.Ultracell.g2`), which is the *density's own
    sphere in the ultracell's reciprocal space* -- not the whole box, and not
    the ``keep`` set P88's Hartree kernel and convergence test live on.

    **Both departures are measured rather than argued.** Summing the whole box
    reproduces the grid values exactly and is wrong everywhere between them,
    for :mod:`defumat.basis.sample`'s reason, and a slab's box is 10^8 points in
    any case. Elk's ``keep`` -- every ``G`` inside the *unit cell's* dense
    sphere, at every ``Q`` -- is a differently shaped set that misses a thin
    outer shell of the ultracell's own: read back at the box's own grid points,
    the sphere reproduces a modulated silicon density to **3e-16** against a
    peak of 9.7e-2, where ``keep`` is out by 5e-9 at ``N = 2`` and 2e-8 at
    ``N = 3``.

    **The box itself aliases at large ``N`` and that is P88's, not this
    function's.** The density's support reaches ``|h| ~ 8n`` in ultracell
    Miller units and the box's centred interval ends at ``7.5 n`` on this cell,
    so the outermost shell wraps from about ``n = 5``: the same read-back is
    9.9e-12 there rather than 5e-16. It is the ordinary ``ecutrho = 4 ecutwfc``
    corner aliasing one level out, and it bounds what an ultracell image can
    claim, so it is stated instead of being discovered.
    """
    ultracell = states.ultracell
    index = np.flatnonzero(
        ultracell.g2(system.cell).reshape(-1) <= float(system.ecutrho) + 1.0e-8
    )
    return np.asarray(r_to_g(np.asarray(field), index)), ultracell.miller_at(index)


def _refuse_a_stacked_ultracell(states, exit_axis):
    """A modulation along the stacking axis is not a junction this can sum.

    The ultracell would then be ``n`` slabs stacked with the exit plane between
    two of them, so the electron leaves through the material rather than out of
    it -- and two ``Q`` differing only along that axis share every in-plane
    index, which the exit integral cannot tell apart and the unit-cell code
    would sum incoherently as different k-points. It is also what makes
    ``exit_height`` the same number in both cells.
    """
    n = int(states.ultracell.shape[int(exit_axis)])
    if n != 1:
        raise NotImplementedError(
            f"the ultracell is {n} cells deep along the stacking axis "
            f"{exit_axis} and a vertical junction needs one: the exit plane "
            "would sit between two of the stacked slabs, so the electron "
            "leaves through the material rather than out of it. Modulate the "
            "surface directions instead, which is where a charge density wave, "
            "a domain wall or a moire period lives"
        )


def _refuse_a_k_set_this_cannot_sum(states, exit_axis):
    """One division along the stacking axis, on the ``k0`` mesh this time.

    :func:`defumat.workflows.transport._refuse_a_k_set_this_cannot_sum`'s
    statement, re-expressed on the set the ultracell samples: lateral momentum
    is conserved exactly and the momentum along the normal is not, so states at
    the same ``k_parallel`` and different ``k_perp`` interfere with a phase that
    depends on where the exit plane sits. The wedge half of it does not arise --
    a ``k0`` mesh is unreduced and equally weighted by construction.
    """
    along = np.unique(np.round(np.asarray(states.k0_crystal)[:, int(exit_axis)], 8))
    if along.size > 1:
        raise NotImplementedError(
            f"the ultracell k-set has {along.size} divisions along the stacking "
            f"axis and this quantity needs one: pass a kgrid with 1 along axis "
            f"{exit_axis} to run_ultracell. A two-dimensional material is a "
            "slab with one k-point along its normal"
        )
