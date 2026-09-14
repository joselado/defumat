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

import warnings

import numpy as np

from defumat.basis.fft import r_to_g
from defumat.basis.sample import sample_miller
from defumat.stm.image import STMImage, project_spin, tunnelling_weights
from defumat.stm.plane import PlotPlane
from defumat.workflows.stm import (
    _constant_current,
    _plane,
    _tip_energy,
    _tip_width,
)

__all__ = ["run_ultracell_stm"]


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
            an ultracell run is usually for: a spin density wave is flat in the
            charge and modulated in the magnetization, so a nonmagnetic tip
            sees almost nothing of it.
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
