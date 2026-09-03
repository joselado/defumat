"""``run_vertical_transport``: tunnelling *through* a two-dimensional material.

An electron enters at the tip, at a point ``r`` above the material, and leaves
into the substrate, anywhere in a plane below it. What decides how much current
flows is the nonlocal Green's function between the two,
``G(r, r') ~ sum psi(r) psi*(r')``, so this is a scanning-tunnelling image of a
quantity that has been *through* the sample rather than of the sample's surface
density of states.

**The two limits are what it is for.** When one band carries the current -- a
single sheet of graphene at the Fermi level -- the exit-plane Gram matrix is
effectively one-dimensional and the map is proportional to the Tersoff-Hamann
image at the tip: the same picture P65 draws. When several bands are degenerate
at the same lateral momentum -- a bilayer, a stack, a moire -- they interfere on
the way through, and the map departs from any local density of states. That
departure is reported separately (:attr:`~defumat.transport.green.
VerticalTransport.interference`) rather than left implicit, because it is the
only part of the answer P65 cannot already give.

**What the substrate is.** An infinite, featureless plane, invariant under every
lateral lattice translation -- a metal the material sits on. That makes lateral
momentum conserved, so the sum over ``k`` is incoherent and the interference is
between bands at the same ``k``. :mod:`defumat.transport.substrate` derives it.
A *finite* contact patch is a different physical regime and is not this.

**Where the two planes go.** The exit plane is a crystal coordinate along the
stacking axis and so is the tip plane, and the material has to lie **between**
them -- a cell is periodic, so "above" and "below" are only meaningful relative
to where the atoms are. That is checked and warned about rather than assumed.
"""

from __future__ import annotations

import warnings

import numpy as np

from defumat.basis.builder import build_basis
from defumat.basis.sample import sample_wavefunctions
from defumat.scf.driver import Calculation
from defumat.stm.plane import PlotPlane
from defumat.transport.green import (
    VerticalTransport,
    amplitude_weights,
    channel_basis,
    transmission,
)
from defumat.transport.substrate import (
    exit_overlap,
    spin_projector,
    surface_area,
    volume_overlap,
)
from defumat.system.kpoints import KPoints
from defumat.system.kpoints import for_spin as kpoints_for_spin
from defumat.workflows.nscf import fixed_density_states
from defumat.workflows.stm import _plane, _refuse_what_has_no_fermi_level

__all__ = ["run_vertical_transport", "whole_grid"]

#: The leads' broadening when none is given, in Ry. Small enough to resolve a
#: band structure and large enough that a discrete k-mesh does not show as
#: spikes; it is the one number here with no first-principles value.
DEFAULT_BROADENING = 1.0e-3


def run_vertical_transport(
    system,
    pseudos,
    result,
    *,
    exit_height: float,
    exit_axis: int = 2,
    height: float | None = None,
    axis: int | None = None,
    plane: tuple | PlotPlane | None = None,
    shape: tuple[int, int] = (40, 40),
    tip=None,
    energies=None,
    bias: float | None = None,
    nenergies: int = 1,
    broadening: float = DEFAULT_BROADENING,
    method: str = "spectral",
    smearing: str = "gaussian",
    spin=None,
    polarization: float = 1.0,
    incoherent: bool = True,
    exit_region: str = "plane",
    grid: tuple[int, int, int] | None = None,
    shift: tuple[int, int, int] | None = None,
    kpoints=None,
    nbnd: int | None = None,
    conv_thr: float = 1.0e-6,
    k_batch: int | None | str = "default",
) -> VerticalTransport:
    """``T(r; E)``: the vertical transmission from a tip at ``r`` to a substrate.

    Args:
        system, pseudos, result: the converged run. Its wavefunctions are
            needed, not only its density.
        exit_height: the substrate plane's crystal coordinate along
            ``exit_axis``. It should sit in the vacuum below the material.
        exit_axis: the stacking axis, 2 for an ordinary slab.
        height: the tip plane's crystal coordinate, spanning the surface cell
            -- P65's shortcut, and the usual way to ask for an image.
        axis: which lattice vector ``height`` measures along; defaults to
            ``exit_axis``.
        plane: the general form for the *tip* plane, ``(origin, edge1, edge2)``
            in crystal coordinates, or a
            :class:`~defumat.stm.plane.PlotPlane`. The tip plane may be tilted;
            the exit plane may not, for the reason
            :mod:`defumat.transport.substrate` gives.
        shape: the tip sampling.
        tip: explicit tip points, ``(np, 3)`` in crystal coordinates, instead
            of a plane -- for a spectrum at one place rather than a map.
        energies: the energies in Ry, a scalar or a sequence. Defaults to the
            Fermi level of the k-set actually used.
        bias: a sample bias in Ry. The transmission is integrated over
            ``[E, E + V]`` on ``nenergies`` points, which is the finite-bias
            current; ``None`` is the zero-bias conductance at ``energies``.
        nenergies: points in the bias window.
        broadening: ``eta`` in Ry -- the width of the energy window the tip
            and the substrate let states through in, which is the leads' own
            coupling and not a numerical smearing.
        method: ``"spectral"``, the on-shell amplitude, which is what converges;
            or ``"resolvent"``, the exact Landauer denominator, which a
            truncated band sum cannot evaluate and which therefore warns. The
            module docstring of :mod:`defumat.transport.green` measures why.
        smearing: which delta the on-shell amplitude is the square root of. A
            Gaussian by default, and a delta that goes negative is refused --
            an amplitude has no square root there.
        spin: a spin-selective substrate. ``"up"``/``"down"`` for a collinear
            run, a cartesian direction for a spinor one, ``None`` for a
            substrate that takes both spins equally.
        polarization: the substrate's spin polarization, in ``[-1, 1]``.
        incoherent: also build the map with every band tunnelling
            independently, so that the interference can be read off.
        exit_region: ``"plane"``, or ``"volume"`` for the diagnostic in which
            the substrate is the whole cell -- which is Tersoff-Hamann exactly
            and is what :func:`defumat.workflows.stm.run_stm` computes.
        grid, shift, kpoints, nbnd, conv_thr: re-solve the bands at fixed
            density on a denser k-set first, as a density of states wants.
            The grid is built **whole**, not reduced to a wedge, because a
            wedge is refused here -- see :func:`whole_grid`.
        k_batch: the k-axis batching dial.

    Returns a :class:`~defumat.transport.green.VerticalTransport`.
    """
    _refuse_what_has_no_fermi_level(system, result)
    if method.strip().lower() == "resolvent":
        warnings.warn(
            "method='resolvent' is the exact Landauer denominator and a "
            "truncated band sum cannot evaluate it: its far-from-E states "
            "carry the barrier's evanescent decay entirely by cancellation, "
            "measured at a factor of 349 on a cell diagonalised completely. "
            "The result depends on nbnd at every band count and is not a "
            "converged transmission",
            stacklevel=2,
        )
    if exit_axis not in (0, 1, 2):
        raise ValueError(f"exit_axis must be 0, 1 or 2, got {exit_axis}")
    if axis is None:
        axis = exit_axis
    if exit_region not in ("plane", "volume"):
        raise ValueError(
            f"unknown exit_region {exit_region!r}: use 'plane' (the substrate) "
            "or 'volume' (the Tersoff-Hamann diagnostic)"
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
            "a transmission is built from the wavefunctions and this result "
            "carries none: run the SCF without discarding them, or pass a grid "
            "so the bands are re-solved"
        )

    used = calculation.system
    _refuse_a_k_set_this_cannot_sum(used, exit_axis)
    _refuse_an_augmented_plane(used, pseudos, exit_axis,
                              (float(exit_height),), "the exit plane")

    geometry, points = _tip_points(used.cell, height, axis, plane, shape, tip)
    _warn_if_the_slab_is_not_between(used, exit_axis, exit_height, points, axis)
    _refuse_an_augmented_plane(
        used, pseudos, exit_axis,
        tuple(float(s) for s in np.unique(np.round(points[:, exit_axis], 12))),
        "the tip plane")

    grid_energies = _energies(energies, levels, bias, nenergies)
    wavefunctions = np.asarray(wavefunctions)

    values, extras = _assemble(
        calculation, wavefunctions, eigenvalues, points,
        exit_height=float(exit_height), exit_axis=exit_axis,
        energies=grid_energies, broadening=float(broadening),
        spin=spin, polarization=float(polarization),
        incoherent=bool(incoherent), exit_region=exit_region,
        method=method, smearing=smearing,
    )

    if bias is not None:
        # The finite-bias current: the conductance integrated over the window.
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

    return VerticalTransport(
        values=coherent,
        plane=geometry,
        energies=np.atleast_1d(grid_energies if bias is None
                               else np.array([grid_energies[0]])),
        broadening=float(broadening),
        exit_height=float(exit_height),
        exit_axis=int(exit_axis),
        incoherent=incoherent_map,
        fermi_energy=levels.get("fermi_energy"),
        spin=None if spin is None else (
            spin if isinstance(spin, str)
            else tuple(float(c) for c in np.ravel(spin))),
        polarization=float(polarization),
        grid=None if grid is None else tuple(int(n) for n in grid),
        least_eigenvalue=extras["least_eigenvalue"],
        offdiagonal_weight=extras["offdiagonal_weight"],
        notes=extras["notes"],
    )


# --------------------------------------------------------------------------
# the k-set
# --------------------------------------------------------------------------


def whole_grid(system, grid, shift=None) -> KPoints:
    """A denser Monkhorst-Pack grid, **complete** rather than reduced.

    :func:`defumat.workflows.nscf.denser_grid` reduces with the run's own
    symmetry, which is right for a density of states and wrong here: a wedge is
    refused (:func:`_refuse_a_k_set_this_cannot_sum` says why), and unfolding it
    is not the escape it is for the nesting function, because unfolding a
    *wavefunction* means rotating it and not merely relabelling a scalar.

    So the whole grid is built instead -- ``KPoints.automatic`` with no
    rotations, which is what a ``nosym`` run gets -- and the spin degeneracy is
    applied through :func:`~defumat.system.kpoints.kpoints_for_spin` for the
    reason ``denser_grid`` documents at length: every constructor applies it
    unconditionally, and a spinor band holds one electron.
    """
    if shift is None:
        shift = system.kpoints.shift or (0, 0, 0)
    return kpoints_for_spin(
        KPoints.automatic(
            tuple(int(n) for n in grid),
            tuple(int(s) for s in shift),
            system.cell,
            precision=system.kpoints.precision,
            rotations=None,
        ),
        system.nspin,
    )


# --------------------------------------------------------------------------
# the assembly
# --------------------------------------------------------------------------




def _assemble(calculation, wavefunctions, eigenvalues, points, *,
              exit_height, exit_axis, energies, broadening, spin,
              polarization, incoherent, exit_region, method, smearing):
    """Sample the tip, build every ``S_k``, contract. One channel at a time.

    **A spinor's two components are two amplitude vectors, not one.** The tip
    here is not spin-selective -- the substrate is -- so what tunnels in is the
    whole spinor, and tracing the Landauer expression over the tip's spin gives

        T = sum_s  a_s^dagger S a_s

    with ``a_s`` the amplitude vector of one spinor component and ``S`` the
    exit-plane overlap, which carries the substrate's spin acceptance inside
    it. It is a *sum of two* quadratic forms and not one form on a doubled
    vector: the two components leave through the same substrate but enter
    through independent tip channels.
    """
    used = calculation.system
    basis = build_basis(used)
    miller = np.asarray(basis.planewaves.miller(basis.smooth))
    mask = np.asarray(basis.planewaves.mask)
    kcrystal = np.asarray(used.kpoints.crystal(used.cell))
    kweights = np.asarray(used.kpoints.weights, dtype=float)
    volume = float(used.cell.volume)
    npol = 2 if calculation.noncolin else 1
    npwx = basis.npwx

    projector, channel_scale = _substrate_acceptance(
        spin, polarization, npol, wavefunctions.shape[0])
    # ``S`` without a Hamiltonian to hang it on, which the volume diagnostic
    # needs and the plane does not: the augmentation charge is zero in the
    # vacuum where both planes of a tunnelling geometry sit.
    apply_s = (calculation._spinor_overlap if calculation.noncolin
               else calculation._overlap)

    nspin, nk, nbnd, _ = wavefunctions.shape
    total = np.zeros((energies.shape[0], points.shape[0]))
    total_incoherent = np.zeros_like(total) if incoherent else None
    top_band = np.zeros_like(total) if incoherent else None
    least, offdiagonal, hermiticity, channels = np.inf, [], 0.0, []

    for ispin in range(nspin):
        amplitudes = np.empty((npol, nk, nbnd, points.shape[0]), dtype=complex)
        overlaps = np.empty((nk, nbnd, nbnd), dtype=complex)
        for ik in range(nk):
            block = np.asarray(wavefunctions[ispin, ik])
            if exit_region == "volume":
                overlaps[ik] = volume_overlap(block, mask[ik], npol,
                                              overlap=lambda p, i=ik: apply_s(p, i))
            else:
                overlaps[ik] = exit_overlap(
                    block, miller[ik], exit_height, exit_axis, used.cell,
                    mask=mask[ik], npol=npol, projector=projector,
                )
            sampled = sample_wavefunctions(
                block.reshape((nbnd, npol, npwx)), miller[ik], kcrystal[ik],
                points, volume, mask=mask[ik],
            )
            amplitudes[:, ik] = np.moveaxis(sampled, 1, 0)

        hermiticity = max(hermiticity, float(
            np.abs(overlaps - np.conj(np.swapaxes(overlaps, 1, 2))).max()))
        hermitian = 0.5 * (overlaps + np.conj(np.swapaxes(overlaps, 1, 2)))
        spectrum = np.linalg.eigvalsh(hermitian)
        least = min(least, float(spectrum.min()))
        # How many independent ways there are through the substrate: the
        # participation ratio of S_k's spectrum, which is the number of open
        # transmission channels. In the vacuum it is close to **one** -- every
        # band's evanescent tail has nearly the same shape on the plane and
        # differs only by a coefficient -- and that is precisely why the
        # interference here is large rather than a correction: the plane sees
        # one amplitude, so what tunnels is |sum_n a_n c_n|^2 and not sum |a_n|^2.
        positive = np.clip(spectrum, 0.0, None)
        norms2 = (positive ** 2).sum(axis=1)
        channels.extend(
            np.where(norms2 > 0.0, positive.sum(axis=1) ** 2
                     / np.where(norms2 > 0.0, norms2, 1.0), 0.0))
        # In the channel basis, so that "how much sits off the diagonal" is a
        # property of the substrate and not of which basis the eigensolver
        # returned inside a multiplet.
        u = channel_basis(overlaps, eigenvalues[ispin])
        rotated = np.einsum("kni,knm,kmj->kij", u.conj(), overlaps, u,
                            optimize=True)
        norms = np.linalg.norm(rotated, axis=(1, 2))
        diagonals = np.linalg.norm(np.einsum("knn->kn", rotated), axis=1)
        offdiagonal.extend(
            np.sqrt(np.clip(norms ** 2 - diagonals ** 2, 0.0, None))
            / np.where(norms > 0.0, norms, 1.0))

        scale = 1.0 if channel_scale is None else channel_scale[ispin]
        if scale == 0.0:
            continue
        for ie, energy in enumerate(energies):
            weights = amplitude_weights(
                eigenvalues[ispin], energy, broadening, method, smearing)
            for component in range(npol):
                total[ie] += scale * transmission(
                    amplitudes[component], overlaps, kweights, weights,
                    coherent=True)
                if incoherent:
                    total_incoherent[ie] += scale * transmission(
                        amplitudes[component], overlaps, kweights, weights,
                        coherent=False, eigenvalues=eigenvalues[ispin])
                    top_band[ie] += scale * transmission(
                        amplitudes[component][:, -1:], overlaps[:, -1:, -1:],
                        kweights, weights[:, -1:], coherent=False)

    if not np.any(total > 0.0):
        # **A Gaussian delta returns exactly zero, not something small.** With
        # no state within a few ``broadening`` of the tip energy every
        # amplitude underflows and the map is identically 0.0 -- which reads
        # like a bug in the assembly and is a statement about the k-set. It is
        # P65's "a zero-bias image of a gapped run is identically zero" with a
        # second way in: a semimetal whose states sit at a symmetry point the
        # grid misses (a 4x4 mesh has no K, so graphene has nothing at E_F).
        warnings.warn(
            "the transmission is identically zero: no state lies within "
            f"{broadening:g} Ry of the tip energy on this k-set. Either the "
            "run is gapped there, or the grid misses the point the states sit "
            "at -- graphene's are at K, which a mesh whose divisions are not a "
            "multiple of three does not contain. Widen broadening=, move "
            "energies=, or use a grid that carries the states",
            stacklevel=4,
        )

    values = {"coherent": total}
    if incoherent:
        values["incoherent"] = total_incoherent
    notes = {
        "exit_region": exit_region,
        "npol": npol,
        # The open transmission channels, averaged over the k-set.
        "channels": float(np.mean(channels)) if channels else 0.0,
        # A Gram matrix is Hermitian; that it comes back Hermitian is a check
        # on the h3 sum and the grouping, and it is free.
        "hermiticity": hermiticity,
        # How much of the transmission the topmost band carries: the band-count
        # truncation, which the resolvent suppresses as 1/(E - e)^2 and which is
        # therefore mild here in a way a sum-over-states response never is.
        "band_edge_weight": (
            float("nan") if not incoherent or total_incoherent.sum() == 0.0
            else float(top_band.sum() / total_incoherent.sum())),
    }
    extras = {
        "least_eigenvalue": float(least),
        "offdiagonal_weight": float(np.mean(offdiagonal)) if offdiagonal else 0.0,
        "notes": notes,
    }
    return values, extras


def _substrate_acceptance(spin, polarization, npol, nspin):
    """The substrate's spin selection: a 2x2 matrix, or a weight per channel."""
    if spin is None:
        return None, None
    if npol == 2:
        return spin_projector(spin, polarization), None
    if nspin == 2:
        return None, _collinear_acceptance(spin, polarization)
    raise NotImplementedError(
        "a spin-selective substrate needs a magnetization to select from and "
        "this run has none: every direction would take half of everything, "
        "which is the charge map again"
    )


def _collinear_acceptance(spin, polarization):
    """``(1 +- P)/2`` per channel -- P65's ``[rho + P n.m]/2``, one level down.

    A collinear run's two channels are two calculations, so a polarized
    substrate is a weight on each rather than a matrix between them. The
    identity is the same one :func:`defumat.stm.image.project_spin` writes:
    ``[rho + P m]/2 = (1+P)/2 rho_up + (1-P)/2 rho_down``.
    """
    from defumat.stm.image import _collinear_axis

    axis = _collinear_axis(spin)
    p = float(polarization) * axis
    if not -1.0 <= float(polarization) <= 1.0:
        raise ValueError(
            f"the substrate polarization must be in [-1, 1], got {polarization}")
    return (0.5 * (1.0 + p), 0.5 * (1.0 - p))


# --------------------------------------------------------------------------
# where the tip goes, which energies, and what is refused
# --------------------------------------------------------------------------


def _tip_points(cell, height, axis, plane, shape, tip):
    """``(geometry, points)``: a plane to make a map on, or explicit points."""
    if tip is not None:
        if height is not None or plane is not None:
            raise ValueError(
                "tip= gives the tip positions explicitly and height=/plane= "
                "build them from a plane: pass one or the other"
            )
        points = np.atleast_2d(np.asarray(tip, dtype=float))
        if points.shape[-1] != 3:
            raise ValueError(
                f"tip points are (np, 3) crystal coordinates, got {points.shape}")
        return None, points
    geometry = _plane(cell, height, axis, plane, shape)
    return geometry, geometry.flat()


def _energies(energies, levels, bias, nenergies):
    """The energies in Ry: one, a list, or a bias window to integrate over."""
    if energies is None:
        fermi = levels.get("fermi_energy")
        if fermi is None:
            homo, lumo = levels.get("homo"), levels.get("lumo")
            if homo is None or lumo is None:
                raise ValueError(
                    "this run has no Fermi level to put the tip at: pass "
                    "energies= in Ry"
                )
            fermi = 0.5 * (float(homo) + float(lumo))
            warnings.warn(
                "this run has fixed occupations, so there is no Fermi level: "
                f"the energy is the middle of the gap, {fermi:.4f} Ry",
                stacklevel=3,
            )
        energies = float(fermi)
    grid = np.atleast_1d(np.asarray(energies, dtype=float))

    if bias is None:
        return grid
    if grid.size != 1:
        raise ValueError(
            "a bias window starts at one energy: pass a scalar energies= with "
            "bias=, or a list of energies without it"
        )
    if int(nenergies) < 2:
        raise ValueError(
            f"integrating a bias window needs at least two points, got "
            f"{nenergies}: without bias= a single energy is the zero-bias "
            "conductance"
        )
    low, high = sorted((float(grid[0]), float(grid[0]) + float(bias)))
    return np.linspace(low, high, int(nenergies))


def _refuse_a_k_set_this_cannot_sum(system, exit_axis):
    """The two k-set assumptions, both of them exact statements rather than taste."""
    crystal = np.asarray(system.kpoints.crystal(system.cell))
    along = np.unique(np.round(crystal[:, exit_axis], 8))
    if along.size > 1:
        raise NotImplementedError(
            f"the k-set has {along.size} divisions along the stacking axis and "
            "this quantity needs one: lateral momentum is conserved exactly, "
            "the momentum along the normal is not, so states at the same "
            "k_parallel and different k_perp interfere with a phase that "
            "depends on where the exit plane sits -- and the counting of "
            "lateral cells changes with them. A two-dimensional material is a "
            "slab with one k-point along its normal"
        )
    weights = np.asarray(system.kpoints.weights, dtype=float)
    if weights.size > 1 and np.ptp(weights) > 1.0e-8 * np.abs(weights).max():
        raise NotImplementedError(
            "a symmetry-reduced k-set is refused: the wedge sum returns the "
            "map symmetrised over the *whole* point group, and only the "
            "subgroup that leaves the exit plane where it is belongs to this "
            "geometry -- a mirror through the slab exchanges the tip side with "
            "the substrate side, which is not a symmetry of a tip above a "
            "substrate. Run the whole grid (nosym = .true.)"
        )


def _refuse_an_augmented_plane(system, pseudos, axis, heights, what):
    """A plane that cuts an augmentation sphere is not the pseudo-density's.

    In the vacuum a pseudo-wavefunction *is* the true one, which is why an
    ultrasoft or PAW dataset needs nothing extra here; inside a sphere the two
    differ and the overlap would want ``Q_ij``. Both planes of a tunnelling
    geometry are in vacuum by construction, so this is a guard on a mistake
    rather than a restriction on the physics.
    """
    radius = _augmentation_radius(pseudos)
    if radius <= 0.0:
        return
    spacing = float(system.cell.volume) / surface_area(system.cell, axis)
    positions = np.asarray(system.structure.positions_crystal(system.cell))
    for height in heights:
        offset = positions[:, axis] - float(height)
        distance = np.abs(offset - np.round(offset)) * spacing
        if distance.min() < radius:
            raise NotImplementedError(
                f"{what} at crystal coordinate {height:g} passes "
                f"{distance.min():.3f} bohr from an atom, inside the "
                f"{radius:.3f} bohr augmentation sphere of this dataset: there "
                "the pseudo-wavefunction is not the true one and the overlap "
                "would need the augmentation charge. Put both planes in the "
                "vacuum, which is where a tip and a substrate are"
            )


def _augmentation_radius(pseudos) -> float:
    """The largest radius any dataset's augmentation charge reaches."""
    radius = 0.0
    for pseudo in pseudos:
        if pseudo.augmentation is None and pseudo.paw is None:
            continue
        r = np.asarray(pseudo.r, dtype=float)
        indices = [p.cutoff_index for p in pseudo.projectors]
        if pseudo.paw is not None:
            indices.append(pseudo.paw.cutoff_index)
        for index in indices:
            if 0 < int(index) < r.size:
                radius = max(radius, float(r[int(index)]))
    return radius


def _warn_if_the_slab_is_not_between(system, exit_axis, exit_height, points, axis):
    """The cell is periodic, so "above" and "below" are relative to the atoms.

    A tip and a substrate on the *same* side of the slab measure tunnelling
    through the vacuum gap and around the periodic image, which is a real
    number and not the one anybody wants.
    """
    if axis != exit_axis:
        return
    positions = np.asarray(system.structure.positions_crystal(system.cell))
    slab = positions[:, exit_axis]
    tip = np.unique(np.round(points[:, exit_axis], 12))
    if tip.size != 1:
        return
    # Fold the slab into the window that starts at the exit plane: the material
    # lies between the two planes exactly when every atom is below the tip.
    folded = np.mod(slab - float(exit_height), 1.0)
    edge = np.mod(float(tip[0]) - float(exit_height), 1.0)
    if edge <= 0.0 or folded.max() >= edge or folded.min() <= 0.0:
        warnings.warn(
            f"the atoms do not lie between the exit plane ({exit_height:g}) "
            f"and the tip ({tip[0]:g}) along axis {exit_axis}: the electron "
            "then tunnels through the vacuum rather than through the material, "
            "which is a different calculation from the one this is for",
            stacklevel=3,
        )
