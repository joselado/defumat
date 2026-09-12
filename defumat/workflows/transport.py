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

**The conjugate question is :func:`run_momentum_transport`.** Replace the point
tip by a *plane* and the real-space map collapses; what is left is one weight
per k-point -- which pocket of the Fermi surface the current actually comes out
of. It is the same object: integrating the point-tip map over the tip plane
gives the sum over k of the other. :mod:`defumat.transport.momentum` derives it,
and it costs a small fraction of the map, because nothing is sampled in real
space.
"""

from __future__ import annotations

import warnings

import numpy as np

from defumat.basis.builder import build_basis
from defumat.batching import resolve_k_batch
from defumat.basis.gvectors import refuse_gamma_storage
from defumat.basis.sample import sample_wavefunctions
from defumat.scf.driver import Calculation, gamma_storage_is_consumable
from defumat.stm.plane import PlotPlane
from defumat.transport.green import (
    DEGENERACY_TOL,
    VerticalTransport,
    amplitude_weights,
    channel_basis,
    spin_transmission,
    transmission,
)
from defumat.transport.momentum import MomentumTransport, momentum_weights
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

__all__ = ["run_vertical_transport", "run_momentum_transport",
           "whole_grid"]

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
    tip_spin=None,
    tip_polarization: float = 1.0,
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
        spin: a spin-selective **substrate**. ``"up"``/``"down"`` for a
            collinear run, a cartesian direction for a spinor one, ``None`` for
            a substrate that takes both spins equally. **Note the asymmetry
            with** :func:`defumat.workflows.stm.run_stm`, where ``spin``
            describes the *tip*: here the substrate was the polarizer first and
            the name is kept for the runs already written against it. The tip's
            own moment is ``tip_spin``.
        polarization: the substrate's spin polarization, in ``[-1, 1]``.
        tip_spin: a spin-polarized **tip**, which is P65's ``spin`` and the
            magnetic counter-electrode of a tunnelling-magnetoresistance
            measurement. Same spelling of a direction as ``spin``, and the two
            may be used together: the answer then depends on the angle between
            the tip's moment and the substrate's.
        tip_polarization: the tip's spin polarization, in ``[-1, 1]``. ``0`` is
            a nonmagnetic tip and gives exactly **half** the unpolarized map,
            which is :func:`defumat.stm.image.project_spin`'s convention.
        incoherent: also build the map with every band tunnelling
            independently, so that the interference can be read off.
        exit_region: ``"plane"``, or ``"volume"`` for the diagnostic in which
            the substrate is the whole cell -- which is Tersoff-Hamann exactly
            and is what :func:`defumat.workflows.stm.run_stm` computes.
        grid, shift, kpoints, nbnd, conv_thr: re-solve the bands at fixed
            density on a denser k-set first, as a density of states wants.
            The grid is built **whole**, not reduced to a wedge, because a
            wedge is refused here -- see :func:`whole_grid`.
        k_batch: the k-axis batching dial. It reaches two places here: the
            band solve, as everywhere else, **and** the assembly's host array
            of tip amplitudes, which is ``(npol, nk, nbnd, npoints)`` complex
            and is the largest thing this workflow allocates. See
            :func:`_assemble` for why an accelerator's default does not bound
            the second one.

    Returns a :class:`~defumat.transport.green.VerticalTransport`.
    """
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
        gamma_storage_is_consumable(system, pseudos), "the vertical tunnelling transmission",
        "psi(r) is evaluated as a bare sum over the stored k + G list "
        "(basis/sample.py)",
    )
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
        tip_spin=tip_spin, tip_polarization=float(tip_polarization),
        incoherent=bool(incoherent), exit_region=exit_region,
        method=method, smearing=smearing,
        k_batch=resolve_k_batch(k_batch),
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
        spin=_label(spin),
        polarization=float(polarization),
        tip_spin=_label(tip_spin),
        tip_polarization=float(tip_polarization),
        grid=None if grid is None else tuple(int(n) for n in grid),
        least_eigenvalue=extras["least_eigenvalue"],
        offdiagonal_weight=extras["offdiagonal_weight"],
        notes=extras["notes"],
    )


def run_momentum_transport(
    system,
    pseudos,
    result,
    *,
    exit_height: float,
    height: float,
    exit_axis: int = 2,
    energies=None,
    bias: float | None = None,
    nenergies: int = 1,
    broadening: float = DEFAULT_BROADENING,
    method: str = "spectral",
    smearing: str = "gaussian",
    spin=None,
    polarization: float = 1.0,
    grid: tuple[int, int, int] | None = None,
    shift: tuple[int, int, int] | None = None,
    kpoints=None,
    nbnd: int | None = None,
    conv_thr: float = 1.0e-6,
    k_batch: int | None | str = "default",
) -> MomentumTransport:
    """``W(k; E)``: which k-points the tunnelling current comes out of.

    The tip is a **plane** here rather than a point, so there is no image and no
    real-space sampling: one Gram matrix per k-point at each of the two heights
    and one ``nbnd^2`` trace between them, which is
    :func:`~defumat.transport.momentum.momentum_weights`. That derivation, and
    the reason the answer is the plane integral of
    :func:`run_vertical_transport`'s map, are in
    :mod:`defumat.transport.momentum`.

    Args:
        system, pseudos, result: the converged run, wavefunctions and all.
        exit_height: the substrate plane's crystal coordinate along
            ``exit_axis``, in the vacuum below the material.
        height: the **tip** plane's crystal coordinate along the same axis, in
            the vacuum above it -- or a *sequence* of them, which is a height
            sweep and is what the vacuum decay constants are fitted to. Unlike
            :func:`run_vertical_transport` this is required and may not be
            tilted: a tilted plane is not spanned by two lattice vectors, and
            the exact Miller-index orthogonality
            :func:`~defumat.transport.substrate.exit_overlap` rests on is what
            makes both sides cheap.

            **A sweep costs one band solve, not one per height.** The bands do
            not know where the tip is; only the tip's Gram matrix does, and
            that is two gathers and an ``nbnd x n_hpar`` product. The whole
            six-height sweep is therefore a rounding error on top of the
            diagonalisations, which is the same observation the energy axis
            rests on one level down.
        exit_axis: the stacking axis, 2 for an ordinary slab.
        energies, bias, nenergies, broadening, method, smearing: exactly as in
            :func:`run_vertical_transport`, and they mean the same thing.
        spin, polarization: a spin-selective **substrate**, as there. There is
            no ``tip_spin`` here: a magnetic *plane* tip would contract the two
            spinor components through its own 2x2 projector, which the trace
            over a plane does not collapse the same way, so it is left to the
            map rather than approximated.
        grid, shift, kpoints, nbnd, conv_thr: re-solve at fixed density on a
            denser k-set. This is the convergence parameter of the whole
            quantity -- ``W(k)`` *is* a function of k, so the grid is its
            resolution -- and it is built **whole** rather than reduced, for
            the reason :func:`whole_grid` gives.
        k_batch: the band solve's batching dial. The assembly here allocates
            only ``(nk, nbnd, nbnd)``, so it needs no bound of its own.

    Returns a :class:`~defumat.transport.momentum.MomentumTransport`, carrying
    the transmission and its two limits -- the Tersoff-Hamann one and the plain
    Fermi surface -- because all three come from the same two arrays and the
    physics is in their ratio. Its columns are ``(nheights, nenergies, nk)``
    with each of the first two axes **squeezed away when a scalar was asked
    for**, which is the convention the rest of this package uses for the energy
    axis already.
    """
    _refuse_what_has_no_fermi_level(system, result)
    refuse_gamma_storage(
        gamma_storage_is_consumable(system, pseudos),
        "the momentum-resolved tunnelling weight",
        "the exit-plane Gram matrix is a sum over the stored k + G list and a "
        "half sphere is missing its conjugate partner (transport/substrate.py)",
    )
    if method.strip().lower() == "resolvent":
        warnings.warn(
            "method='resolvent' is the exact Landauer denominator and a "
            "truncated band sum cannot evaluate it: measured at a cancellation "
            "factor of 349 on a cell diagonalised completely. The result "
            "depends on nbnd at every band count",
            stacklevel=2,
        )
    if exit_axis not in (0, 1, 2):
        raise ValueError(f"exit_axis must be 0, 1 or 2, got {exit_axis}")

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
            "a tunnelling weight is built from the wavefunctions and this "
            "result carries none: run the SCF without discarding them, or pass "
            "a grid so the bands are re-solved"
        )

    sweep = np.atleast_1d(np.asarray(height, dtype=float))
    if sweep.ndim != 1 or sweep.size == 0:
        raise ValueError(
            f"height must be a number or a sequence of them, got {height!r}")
    scalar_height = np.ndim(height) == 0

    used = calculation.system
    _refuse_a_k_set_this_cannot_sum(used, exit_axis)
    _refuse_an_augmented_plane(used, pseudos, exit_axis,
                               (float(exit_height), *map(float, sweep)),
                               "the two planes")
    for one in sweep:
        _warn_if_the_planes_do_not_straddle(used, exit_axis, float(exit_height),
                                            float(one))

    grid_energies = _energies(energies, levels, bias, nenergies)
    wavefunctions = np.asarray(wavefunctions)

    columns, extras = _assemble_momentum(
        calculation, wavefunctions, eigenvalues,
        exit_height=float(exit_height), heights=sweep,
        exit_axis=exit_axis, energies=grid_energies,
        broadening=float(broadening), spin=spin,
        polarization=float(polarization), method=method, smearing=smearing,
    )

    if bias is not None:
        # The finite-bias current: the conductance integrated over the window.
        columns = {key: np.trapezoid(array, grid_energies, axis=1)[:, None]
                   for key, array in columns.items()}
    if columns["weight"].shape[1] == 1:
        columns = {key: array[:, 0] for key, array in columns.items()}
    if scalar_height:
        columns = {key: array[0] for key, array in columns.items()}

    crystal = np.asarray(used.kpoints.crystal(used.cell))
    return MomentumTransport(
        weight=columns["weight"],
        tersoff_hamann=columns["tersoff_hamann"],
        bare=columns["bare"],
        incoherent=columns["incoherent"],
        kpoints=crystal,
        kcartesian=np.asarray(used.kpoints.cartesian(used.cell)),
        kweights=np.asarray(used.kpoints.weights, dtype=float),
        energies=np.atleast_1d(grid_energies if bias is None
                               else np.array([grid_energies[0]])),
        broadening=float(broadening),
        smearing=str(smearing),
        height=float(sweep[0]) if scalar_height else tuple(map(float, sweep)),
        exit_height=float(exit_height),
        exit_axis=int(exit_axis),
        grid=None if grid is None else tuple(int(n) for n in grid),
        fermi_energy=levels.get("fermi_energy"),
        least_eigenvalue=extras["least_eigenvalue"],
        hermiticity=extras["hermiticity"],
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
              polarization, tip_spin, tip_polarization, incoherent,
              exit_region, method, smearing, k_batch=None):
    """Sample the tip, build every ``S_k``, contract. One channel at a time.

    **A spinor's two components are two amplitude vectors, not one**, and what
    joins them is the tip's own spin structure. With a nonmagnetic tip the
    Landauer expression is traced over the tip's spin and the two components
    add incoherently,

        T = sum_s  a_s^T S a_s^*

    with ``a_s`` the amplitude vector of one spinor component and ``S`` the
    exit-plane overlap, which carries the *substrate's* spin acceptance inside
    it. It is a sum of two quadratic forms and not one form on a doubled
    vector: the two components leave through the same substrate but enter
    through independent tip channels.

    A **magnetic** tip couples through ``P_t = (1 + P n.sigma)/2`` instead of
    through the identity, and then the two components are contracted rather
    than added: ``T = Tr[P_t M]`` with ``M[s,s'] = a_s^T S a_{s'}^*``
    (:func:`defumat.transport.green.spin_transmission`). ``P_t = 1`` is the sum
    above, so ``tip_spin=None`` takes the branch it always took, unchanged and
    to the last bit.

    **``k_batch`` bounds the amplitudes, and it is the one dial that reaches a
    host array.** Everywhere else in this package the dial decides how much is
    resident on the *device*; here the tip amplitudes are sampled into a numpy
    array of ``(npol, nk, nbnd, npoints)`` and the working set is the same on a
    CPU and on a GPU, so ``k_batch = None`` -- an accelerator's default,
    chosen because a device wants the whole axis -- does **not** bound it. A
    machine with a GPU and an image-sized ``npoints`` should pass a number.
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
    tip_projector, tip_scale = _tip_acceptance(
        tip_spin, tip_polarization, npol, wavefunctions.shape[0])
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

    # **The k axis is walked in chunks, and this is a host array rather than a
    # device one.** ``amplitudes`` is ``(npol, nk, nbnd, npoints)`` complex and
    # ``npoints`` is an image -- a 100x100 map over 100 k-points and 50 spinor
    # bands is 1.6 GB, and the ``sa`` intermediate inside
    # :func:`~defumat.transport.green.transmission` is another ``(nk, nbnd,
    # npoints)`` beside it. Every contraction downstream ends in
    # ``kweights @ term``, so the sum over k is exact term by term and a chunk
    # changes only the order the contributions are added in.
    chunk = nk if k_batch is None else min(int(k_batch), nk)
    open_path = 0.0
    for ispin in range(nspin):
        scale = 1.0 if channel_scale is None else channel_scale[ispin]
        # Two polarizers multiply: on a collinear run each is a weight on the
        # channel, and a channel the tip does not accept is one the substrate
        # never sees.
        if tip_scale is not None:
            scale *= tip_scale[ispin]
        open_path += scale
        # Which bands the band-count truncation actually cuts: the topmost
        # *multiplet*, in the same channel basis the denominator is taken in.
        # It is decided k-point by k-point, so it is built once for the whole
        # axis and sliced -- rebuilding it per chunk would be the same numbers
        # and a second place for the rule to drift out of step.
        top_multiplet = _top_multiplet_mask(eigenvalues[ispin])

        for start in range(0, nk, chunk):
            here = slice(start, min(start + chunk, nk))
            live = here.stop - here.start
            amplitudes = np.empty(
                (npol, live, nbnd, points.shape[0]), dtype=complex)
            overlaps = np.empty((live, nbnd, nbnd), dtype=complex)
            for ik in range(here.start, here.stop):
                at = ik - here.start
                block = np.asarray(wavefunctions[ispin, ik])
                if exit_region == "volume":
                    overlaps[at] = volume_overlap(
                        block, mask[ik], npol,
                        overlap=lambda p, i=ik: apply_s(p, i))
                else:
                    overlaps[at] = exit_overlap(
                        block, miller[ik], exit_height, exit_axis, used.cell,
                        mask=mask[ik], npol=npol, projector=projector,
                    )
                sampled = sample_wavefunctions(
                    block.reshape((nbnd, npol, npwx)), miller[ik],
                    kcrystal[ik], points, volume, mask=mask[ik],
                )
                amplitudes[:, at] = np.moveaxis(sampled, 1, 0)

            bands = eigenvalues[ispin][here]
            weight_of_k = kweights[here]
            hermiticity = max(hermiticity, float(
                np.abs(overlaps - np.conj(np.swapaxes(overlaps, 1, 2))).max()))
            hermitian = 0.5 * (overlaps + np.conj(np.swapaxes(overlaps, 1, 2)))
            spectrum = np.linalg.eigvalsh(hermitian)
            least = min(least, float(spectrum.min()))
            # How many independent ways there are through the substrate: the
            # participation ratio of S_k's spectrum, which is the number of
            # open transmission channels. In the vacuum it is close to **one**
            # -- every band's evanescent tail has nearly the same shape on the
            # plane and differs only by a coefficient -- and that is precisely
            # why the interference here is large rather than a correction: the
            # plane sees one amplitude, so what tunnels is |sum_n a_n c_n|^2
            # and not sum |a_n|^2.
            positive = np.clip(spectrum, 0.0, None)
            norms2 = (positive ** 2).sum(axis=1)
            channels.extend(
                np.where(norms2 > 0.0, positive.sum(axis=1) ** 2
                         / np.where(norms2 > 0.0, norms2, 1.0), 0.0))
            # In the channel basis, so that "how much sits off the diagonal" is
            # a property of the substrate and not of which basis the
            # eigensolver returned inside a multiplet.
            u = channel_basis(overlaps, bands)
            rotated = np.einsum("kni,knm,kmj->kij", u.conj(), overlaps, u,
                                optimize=True)
            norms = np.linalg.norm(rotated, axis=(1, 2))
            diagonals = np.linalg.norm(np.einsum("knn->kn", rotated), axis=1)
            offdiagonal.extend(
                np.sqrt(np.clip(norms ** 2 - diagonals ** 2, 0.0, None))
                / np.where(norms > 0.0, norms, 1.0))

            if scale == 0.0:
                continue
            cut = top_multiplet[here]
            for ie, energy in enumerate(energies):
                weights = amplitude_weights(
                    bands, energy, broadening, method, smearing)
                if tip_projector is not None:
                    total[ie] += scale * spin_transmission(
                        amplitudes, overlaps, weight_of_k, weights,
                        tip_projector, coherent=True)
                    if incoherent:
                        total_incoherent[ie] += scale * spin_transmission(
                            amplitudes, overlaps, weight_of_k, weights,
                            tip_projector, coherent=False, eigenvalues=bands)
                        top_band[ie] += scale * spin_transmission(
                            amplitudes, overlaps, weight_of_k,
                            weights * cut, tip_projector,
                            coherent=False, eigenvalues=bands)
                    continue
                for component in range(npol):
                    total[ie] += scale * transmission(
                        amplitudes[component], overlaps, weight_of_k, weights,
                        coherent=True)
                    if incoherent:
                        total_incoherent[ie] += scale * transmission(
                            amplitudes[component], overlaps, weight_of_k,
                            weights, coherent=False, eigenvalues=bands)
                        top_band[ie] += scale * transmission(
                            amplitudes[component], overlaps, weight_of_k,
                            weights * cut, coherent=False, eigenvalues=bands)

    if not np.any(total > 0.0) and open_path == 0.0:
        # Two spin filters in series with nothing in common pass nothing, and
        # that is the answer rather than a k-set that misses the states: a
        # fully polarized tip on one channel of a collinear run and a fully
        # polarized substrate on the other leave no path at all. Saying the
        # k-set is at fault here would be a wrong diagnosis of a right number.
        warnings.warn(
            "the transmission is identically zero because the tip's and the "
            "substrate's spin acceptances leave no channel open between them: "
            "two fully polarized leads on opposite channels of a collinear run "
            "pass nothing. That is the answer, not a numerical accident",
            stacklevel=4,
        )
    elif not np.any(total > 0.0):
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



def _assemble_momentum(calculation, wavefunctions, eigenvalues, *,
                       exit_height, heights, exit_axis, energies, broadening,
                       spin, polarization, method, smearing):
    """A Gram matrix per plane per k-point, then one trace. No real-space sampling.

    The whole cost of :func:`_assemble` is the ``(npol, nk, nbnd, npoints)``
    array of tip amplitudes; there is none here, because a plane tip integrates
    the map away analytically. What is allocated is ``(nk, nbnd, nbnd)`` per
    plane, and the energy loop is free on top of it -- the overlaps do not
    depend on ``E`` and only the per-state weight does, which is the same
    observation the map's docstring makes. **The height axis is free in the same
    sense and for a different reason**: the *bands* do not depend on where the
    tip is, so a sweep pays one extra Gram matrix per height against a whole
    band solve.

    **A spinor's two components are added, not contracted**, which is the
    nonmagnetic-tip trace of :func:`_assemble`; ``exit_overlap`` already sums
    them when ``projector`` is ``None``, and the substrate's own acceptance goes
    inside it exactly as it does there. The two heights use the *same*
    projector: it describes which spins leave, and the tip plane is where they
    come from.
    """
    used = calculation.system
    basis = build_basis(used)
    miller = np.asarray(basis.planewaves.miller(basis.smooth))
    mask = np.asarray(basis.planewaves.mask)
    kweights = np.asarray(used.kpoints.weights, dtype=float)
    npol = 2 if calculation.noncolin else 1

    projector, channel_scale = _substrate_acceptance(
        spin, polarization, npol, wavefunctions.shape[0])

    nspin, nk, nbnd, _ = wavefunctions.shape
    shape = (len(heights), energies.shape[0], nk)
    columns = {name: np.zeros(shape) for name in
               ("weight", "tersoff_hamann", "bare", "incoherent")}
    least, hermiticity = np.inf, 0.0

    def gram_at(block_of, plane_height):
        out = np.empty((nk, nbnd, nbnd), dtype=complex)
        for ik in range(nk):
            out[ik] = exit_overlap(
                block_of(ik), miller[ik], plane_height, exit_axis, used.cell,
                mask=mask[ik], npol=npol, projector=projector,
            )
        return out

    for ispin in range(nspin):
        scale = 1.0 if channel_scale is None else channel_scale[ispin]
        block_of = lambda ik, s=ispin: np.asarray(wavefunctions[s, ik])
        exit_gram = gram_at(block_of, exit_height)
        for ih, plane_height in enumerate(heights):
            tip_gram = gram_at(block_of, float(plane_height))
            for gram in (exit_gram, tip_gram):
                hermiticity = max(hermiticity, float(
                    np.abs(gram - np.conj(np.swapaxes(gram, 1, 2))).max()))
                hermitian = 0.5 * (gram + np.conj(np.swapaxes(gram, 1, 2)))
                least = min(least, float(np.linalg.eigvalsh(hermitian).min()))

            for ie, energy in enumerate(energies):
                weights = amplitude_weights(
                    eigenvalues[ispin], float(energy), broadening, method,
                    smearing)
                here = momentum_weights(exit_gram, tip_gram, kweights,
                                        np.real(weights),
                                        eigenvalues=eigenvalues[ispin])
                for name, column in here.items():
                    columns[name][ih, ie] += scale * column

    notes = {
        "bands": int(nbnd),
        "planes": (tuple(map(float, heights)), float(exit_height)),
        # What the plane sees of the topmost multiplet: the band-count
        # truncation measure, in the same spirit as the map's.
        "top_multiplet_share": _top_multiplet_share(
            columns["weight"], eigenvalues, energies, broadening),
    }
    return columns, {"least_eigenvalue": float(least),
                     "hermiticity": float(hermiticity), "notes": notes}


def _top_multiplet_share(weight, eigenvalues, energies, broadening):
    """How much of the answer the highest band carries -- a truncation measure.

    A band sum stops somewhere, and the honest question is not "how many bands"
    but "how much is the last one worth". The highest band's amplitude weight,
    summed over the k-set and divided by every band's, is that number: it is
    zero when the band set reaches past the window and grows as the window
    approaches the top of it.
    """
    eigenvalues = np.asarray(eigenvalues)
    from defumat.stm.image import smeared_delta

    share = 0.0
    for energy in np.atleast_1d(energies):
        delta = smeared_delta((float(energy) - eigenvalues) / broadening,
                              "gaussian")
        total = float(delta.sum())
        if total > 0.0:
            share = max(share, float(delta[..., -1].sum()) / total)
    return share


def _warn_if_the_planes_do_not_straddle(system, axis, exit_height, height):
    """Both planes in the vacuum, with the atoms between them.

    A cell is periodic, so "above" and "below" mean nothing until the atoms say
    where they are -- and a tip plane on the *same* side as the exit plane is a
    perfectly well-defined number that is not a transmission through anything.
    """
    positions = np.asarray(system.structure.positions_crystal(system.cell))
    along = np.sort(positions[:, axis] % 1.0)
    lo, hi = float(along[0]), float(along[-1])
    inside = [name for name, s in (("the exit plane", exit_height % 1.0),
                                   ("the tip plane", height % 1.0))
              if lo - 1.0e-9 <= s <= hi + 1.0e-9]
    if inside:
        warnings.warn(
            f"{' and '.join(inside)} sits inside the slab, whose atoms span "
            f"[{lo:.3f}, {hi:.3f}] along axis {axis}: the material has to lie "
            "between the two planes for this to be a transmission through it",
            stacklevel=3,
        )
    elif (exit_height % 1.0 < lo) == (height % 1.0 < lo):
        warnings.warn(
            f"both planes are on the same side of the slab (atoms span "
            f"[{lo:.3f}, {hi:.3f}] along axis {axis}): nothing tunnels through "
            "the material between a tip and a substrate that are both above it",
            stacklevel=3,
        )


def _top_multiplet_mask(eigenvalues, tol: float = DEGENERACY_TOL):
    """``(nk, nbnd)``: which bands share the topmost band's multiplet.

    **The band-count diagnostic is a diagonal, and rule D4 says a diagonal is
    not invariant under the rotation a degenerate eigensolver is free in.** The
    quantity is "how much of the transmission the topmost band carries", and it
    was built from the raw diagonal of the *single* topmost band -- so wherever
    that band sits in a multiplet it depended on which basis the solver
    happened to return, while the denominator it was divided by had already
    been rotated into the substrate's channels (:func:`channel_basis`). Two
    bases, one ratio, and the check that certifies the truncation was itself
    the thing rule D4 says cannot be trusted band by band.

    Taking the whole multiplet fixes both halves at once, and it is also the
    better diagnostic: a truncation at ``nbnd`` cuts the multiplet, not one
    member of it.

    The grouping rule is :func:`channel_basis`'s own -- compare to the *first*
    of the group, on sorted eigenvalues -- so the mask and the rotation always
    agree about where the block is.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    nk, nbnd = eigenvalues.shape
    mask = np.zeros((nk, nbnd), dtype=float)
    for ik in range(nk):
        order = np.argsort(eigenvalues[ik], kind="stable")
        start = 0
        while start < nbnd:
            stop = start + 1
            while (stop < nbnd
                   and eigenvalues[ik][order[stop]]
                   - eigenvalues[ik][order[start]] < tol):
                stop += 1
            block = order[start:stop]
            start = stop
        mask[ik, block] = 1.0  # the last block the scan built is the top one
    return mask


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


def _tip_acceptance(spin, polarization, npol, nspin):
    """The **tip's** spin selection: a 2x2 matrix, or a weight per channel.

    The mirror image of :func:`_substrate_acceptance` and physically the more
    familiar of the two -- this is the magnetic tip of spin-polarized STM,
    P65's ``spin``. It is spelled ``tip_spin`` here only because the substrate
    got the plain name first.

    On a **collinear** run it is exact rather than approximate: spin is
    conserved through the junction, the two channels are two independent
    calculations, and a polarized tip is the weight ``(1 +- P)/2`` on each --
    the same three lines the substrate's acceptance is. The two weights then
    simply multiply, which is a series of two spin filters.
    """
    if spin is None:
        return None, None
    if npol == 2:
        return spin_projector(spin, polarization, what="tip"), None
    if nspin == 2:
        return None, _collinear_acceptance(spin, polarization, what="tip")
    raise NotImplementedError(
        "a spin-polarized tip needs a magnetization to couple to and this run "
        "has none: every direction would take half of everything, which is the "
        "charge map again"
    )


def _collinear_acceptance(spin, polarization, what: str = "substrate"):
    """``(1 +- P)/2`` per channel -- P65's ``[rho + P n.m]/2``, one level down.

    A collinear run's two channels are two calculations, so a polarized lead is
    a weight on each rather than a matrix between them. The identity is the
    same one :func:`defumat.stm.image.project_spin` writes:
    ``[rho + P m]/2 = (1+P)/2 rho_up + (1-P)/2 rho_down``.
    """
    from defumat.stm.image import _collinear_axis

    axis = _collinear_axis(spin)
    p = float(polarization) * axis
    if not -1.0 <= float(polarization) <= 1.0:
        raise ValueError(
            f"the {what} polarization must be in [-1, 1], got {polarization}")
    return (0.5 * (1.0 + p), 0.5 * (1.0 - p))


def _label(spin):
    """A direction as something a result object can carry and print."""
    if spin is None:
        return None
    if isinstance(spin, str):
        return spin
    return tuple(float(c) for c in np.ravel(spin))


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
