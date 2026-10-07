"""Real-time propagation from a converged density: the current, the linear response,
the high harmonics and the third harmonic.

Each entry point here is one fixed-density diagonalisation on the k-set the
run propagates, then :mod:`defumat.realtime`. The potential is the ground
state's and is frozen, so what is computed is the independent-particle
response to the field, which is what the published silicon high-harmonic
calculation found sufficient for its conditions (arXiv:1609.09298: the
spectrum "does not change" between the full evolution and the static
ground-state potential, for one material, the LDA and one weak mid-infrared
pulse). The update of the potential in time is a later stage of
``HARMONICS-NEXT.md``.

**The k-set** is a whole unshifted Monkhorst-Pack grid, because the current is
a polar vector and a wedge of the crystal's group does not sum to it, and time
reversal does not hold under a field: the state at ``-k`` evolves under
``-kappa``, which is not the same field. What may reduce the grid is the
**little group of the field**, the operations that leave every polarisation
of the pulse invariant at every time, with time reversal off (Elk's
``tdinit.f90``): each k-point of its wedge then stands for its star, the
current is averaged over the group as a polar vector afterwards, and a field
along ``[100]`` in silicon keeps eight of the 48 operations. A shifted grid is
not closed under the group and is refused, by the rule P24 found for the
response stack.

**Units at this boundary.** Times are Hartree atomic units of time (24.189 as)
as in Elk, frequencies and broadenings are given in eV where an entry point
takes them and converted once, and every result is in Hartree atomic units
except the susceptibilities, which are in SI: ``chi^(3)`` in m^2/V^2, from
``4 pi / E_au^2 = 4.75e-23`` m^2/V^2 per atomic unit, the square of the factor
:data:`~defumat.response.shg.CHI2_AU_TO_PM_PER_V` is the first power of.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from defumat.realtime.orders import OrdersResult, propagate_orders
from defumat.realtime.propagate import RealTimeResult, propagate, require_a_realtime_regime
from defumat.realtime.pulse import EV_TO_HA, Adiabatic, Kick, Pulse, Sum
from defumat.realtime.spectra import (
    HarmonicSpectrum, KickResponse, conductivity_from_kick, harmonic_spectrum)
from defumat.units import C_AU

__all__ = ["run_realtime", "run_realtime_dielectric", "run_hhg", "run_harmonic_orders",
           "run_third_harmonic", "ThirdHarmonic", "run_nonlinear_spectrum",
           "NonlinearSpectrum", "field_little_group", "field_symmetries",
           "chi2_from_orders", "CHI3_AU_TO_SI"]

#: One atomic unit of ``chi^(3)`` in m^2/V^2, in the SI convention
#: ``P = eps_0 chi^(3) E E E``: ``(e/(a_0^2 eps_0 E_au)) / E_au^2`` with
#: ``e/(a_0^2 eps_0 E_au) = 4 pi`` exactly, ``E_au = 5.142e11`` V/m. The
#: same derivation gives 24.4377 pm/V for ``chi^(2)``, which is
#: :data:`~defumat.response.shg.CHI2_AU_TO_PM_PER_V`.
CHI3_AU_TO_SI = 4.0 * math.pi / (5.14220674763e11) ** 2


def _polarisations(pulse) -> list[np.ndarray]:
    if isinstance(pulse, Sum):
        return [v for p in pulse.pulses for v in _polarisations(p)]
    for name in ("polarization", "direction"):
        if hasattr(pulse, name):
            vector = np.asarray(getattr(pulse, name), dtype=float)
            return [vector / np.linalg.norm(vector)]
    raise ValueError(f"cannot read a polarisation off {type(pulse).__name__}")


def field_symmetries(system, pulse, tolerance: float = 1e-8):
    """The field's little group as :class:`~defumat.system.symmetry.Symmetries`.

    The crystal's operations (``System.symmetry_group``) whose cartesian
    rotation fixes every polarisation vector the pulse has, so that ``A(t)``
    is invariant at every time, with their fractional translations, which the
    density's symmetrisation needs (silicon's group along ``[100]`` has
    nonsymmorphic members). Time reversal is never in it, **and neither is an
    operation of a magnetic group that is a symmetry only when followed by
    time reversal** (``t_rev = 1``): the field breaks time reversal, so such an
    operation does not map the driven state onto itself, and the returned
    object cannot carry the flag for anything downstream to see. A ``nosym``
    run keeps the identity alone.
    """
    from defumat.system.symmetry import Symmetries, cartesian_rotations

    symmetries = system.symmetry_group()
    crystal = symmetries.rotation_array()
    cartesian = cartesian_rotations(system.cell, symmetries)
    vectors = _polarisations(pulse)
    t_rev = np.asarray(symmetries.t_rev_array())
    keep = [s for s in range(len(crystal))
            if t_rev[s] == 0
            and all(np.linalg.norm(cartesian[s] @ v - v) < tolerance for v in vectors)]
    if system.nosym:
        keep = [s for s in keep if np.allclose(crystal[s], np.eye(3))]
    return Symmetries(rotations=tuple(symmetries.rotations[s] for s in keep),
                      translations=tuple(symmetries.translations[s] for s in keep))


def field_little_group(system, pulse, tolerance: float = 1e-8):
    """``(crystal rotations, cartesian rotations)`` that leave the pulse's field invariant.

    :func:`field_symmetries` as two arrays, ``(nsym, 3, 3)`` each.
    """
    from defumat.system.symmetry import cartesian_rotations

    group = field_symmetries(system, pulse, tolerance)
    return group.rotation_array(), cartesian_rotations(system.cell, group)


def _kset(system, pulse, kpoints, grid, little_group: bool):
    """``(k-set, cartesian rotations, Symmetries)``: what a run propagates and its group.

    The rotations average the current and the group completes the density
    when the potential is updated; both are ``None`` on a whole k-set.
    """
    from defumat.system.kpoints import KPoints, for_spin, is_reduced

    cell = system.cell
    rotations = None
    shift = (0, 0, 0)
    if kpoints is not None:
        if is_reduced(kpoints):
            raise NotImplementedError(
                "real-time propagation on a symmetry-reduced k-set is not "
                "implemented by passing one: the current is a polar vector and "
                "time reversal does not hold under a field, so the wedge must be "
                "the field's own little group's. Pass grid= instead, which "
                "builds it, or the whole unshifted grid")
        return for_spin(kpoints, system.nspin), None, None
    if grid is None:
        grid = getattr(system.kpoints, "grid", None)
        shift = tuple(int(x) for x in (getattr(system.kpoints, "shift", None) or (0, 0, 0)))
        if grid is None:
            if is_reduced(system.kpoints):
                raise NotImplementedError(
                    "the run's own k-set is reduced and carries no grid to "
                    "rebuild the whole one from; pass grid= or kpoints=")
            return for_spin(system.kpoints, system.nspin), None, None
        if little_group and any(shift):
            raise NotImplementedError(
                "the little group of the field on a shifted Monkhorst-Pack grid is "
                "not implemented: a shifted grid is not closed under the point "
                "group, so its wedge does not sum the current correctly (the rule "
                "P24 found for the response stack). Pass little_group=False to "
                "propagate the whole shifted grid, whose k-points are independent "
                "at a frozen potential, or grid= for an unshifted one")
    grid = tuple(int(n) for n in grid)
    group = None
    if little_group:
        from defumat.system.symmetry import cartesian_rotations

        group = field_symmetries(system, pulse)
        rotations = cartesian_rotations(cell, group)
        kset = KPoints.automatic(grid, (0, 0, 0), cell, rotations=group.rotation_array(),
                                 time_reversal=False)
        if len(rotations) <= 1:
            rotations, group = None, None
    else:
        kset = KPoints.automatic(grid, shift, cell)
    return for_spin(kset, system.nspin), rotations, group


def _occupied_count(system, nelec: float) -> int:
    """The most bands a channel of ``system`` fills: one electron a spinor band, two a scalar one.

    A collinear ``nspin = 2`` channel holds one electron a band, and the
    majority channel holds ``(nelec + m)/2`` of them with ``m`` the moment, two
    when it is not fixed.
    """
    if system.noncolin:
        return int(math.ceil(nelec))
    if system.nspin == 2:
        moment = getattr(system, "tot_magnetization", None)
        moment = 2.0 if moment is None or float(moment) < 0.0 else float(moment)
        return int(math.ceil((nelec + moment) / 2.0))
    return int(math.ceil(nelec / 2.0))


def _occupied_states(system, pseudos, density, kset, *, nbnd, conv_thr, k_batch,
                     calculation, becsum=()):
    """``(calculation, states, weights, v_scf, ddd_paw)`` with fixed occupations.

    ``states`` is ``(nchannel, nk, nb, ndim)`` and ``weights``
    ``(nchannel, nk, nb)``, one channel for ``nspin = 1`` and for a spinor run,
    two for a collinear ``nspin = 2`` one, the shorter padded with zero
    weights. For an insulator with ``occupations = 'fixed'`` the occupied bands
    alone, and a cut through a degenerate multiplet refused by name; for a
    smeared run every band carrying weight above ``1e-10``, held at its
    ground-state occupation.
    """
    calc, states, weights, v_scf, _, _, ddd_paw = _solved_states(
        system, pseudos, density, kset, nbnd=nbnd, conv_thr=conv_thr, k_batch=k_batch,
        calculation=calculation, becsum=becsum)
    return calc, states, weights, v_scf, ddd_paw


def _solved_states(system, pseudos, density, kset, *, nbnd, conv_thr, k_batch,
                   calculation, becsum=()):
    """:func:`_occupied_states` and every band the solve resolved, with their energies.

    Returns ``(calculation, states, weights, v_scf, bands, energies, ddd_paw)``:
    the last three are ``(nchannel, nk, nbnd, ndim)``, ``(nchannel, nk, nbnd)``
    in Ry, for the frequency-domain hierarchy's projector on the computed bands,
    and PAW's one-centre ``D`` from the ground state's ``becsum`` (``None``
    without PAW), frozen with the potential.

    **How many bands a channel carries is read off its occupations**, since a
    collinear run with ``tot_magnetization`` fills its two channels to
    different counts and a spinor band holds one electron where a scalar band
    holds two (``nelec / 2`` was right for ``nspin = 1`` alone).
    """
    from defumat.workflows.nscf import fixed_density_states, threaded_calculation

    nelec_guess = sum(pseudos[t].z_valence for t in system.structure.types)
    fixed = str(getattr(system, "occupations", "fixed")).lower() == "fixed"
    if nbnd is None:
        # Four past the occupied: two ended inside a degenerate conduction pair
        # at three of AlAs's twenty wedge points, where the solve then reports
        # an unconverged root (found writing the harmonic notebook).
        nbnd = _occupied_count(system, nelec_guess) + (4 if fixed else 6)
    # The caller's own calculation, when handed one, is moved to this k-set
    # rather than a second one built beside it.
    calculation, moved, kpoints, k_batch = threaded_calculation(
        calculation, system, kset, k_batch)
    if calculation is None:
        moved = eqx.tree_at(lambda s: s.kpoints, system, kset)
        kpoints = None
    calc, _, eigenvalues, wavefunctions = fixed_density_states(
        moved, pseudos, density, kpoints=kpoints, nbnd=nbnd, conv_thr=conv_thr,
        k_batch=k_batch, calculation=calculation, becsum=becsum)
    require_a_realtime_regime(calc)
    eigenvalues = np.asarray(eigenvalues)
    if eigenvalues.ndim == 2:
        eigenvalues = eigenvalues[None]
    wavefunctions = np.asarray(wavefunctions)
    if wavefunctions.ndim == 3:
        wavefunctions = wavefunctions[None]
    wg, _ = calc.occupations(jnp.asarray(eigenvalues))
    wg = np.asarray(wg)
    channels, nb = eigenvalues.shape[0], eigenvalues.shape[-1]
    keeps = []
    for c in range(channels):
        carrying = np.flatnonzero(np.any(wg[c] > 1e-10, axis=0))
        keep = int(carrying[-1]) + 1 if len(carrying) else 0
        if keep >= nb:
            raise ValueError(
                f"every one of the {nb} bands of channel {c} carries weight; pass a "
                "larger nbnd so the propagated set ends where the occupations do")
        if fixed and keep > 0:
            gaps = eigenvalues[c, :, keep] - eigenvalues[c, :, keep - 1]
            if np.min(gaps) < 1e-5:
                raise ValueError(
                    "occupations = 'fixed' cuts a degenerate multiplet at some k-point "
                    f"(gap {float(np.min(gaps)):.2e} Ry between bands {keep} and "
                    f"{keep + 1}" + (f" of channel {c}" if channels > 1 else "") + "): "
                    "the weights then differ inside a multiplet and the current depends "
                    "on the rotation the eigensolver returned. Use a smearing, or a "
                    "k-set on which the occupied manifold is gapped")
        keeps.append(keep)
    keep = max(keeps)
    weights = np.zeros((channels,) + wg.shape[1:2] + (keep,))
    for c in range(channels):
        weights[c, :, :keeps[c]] = wg[c, :, :keeps[c]]
    states = wavefunctions[:, :, :keep]
    v_scf = calc.potential(jnp.asarray(density)).v_scf
    _, ddd_paw = calc.onecenter(becsum)
    return calc, states, weights, v_scf, wavefunctions, eigenvalues, ddd_paw


def run_realtime(system, pseudos, density, pulse: Pulse, *, dt: float = 0.1,
                 duration: float | None = None, start: float | None = None,
                 kpoints=None, grid=None, little_group: bool = True,
                 nbnd: int | None = None, conv_thr: float = 1.0e-10,
                 k_batch="default", block_steps: int = 400,
                 propagator: str = "taylor4", checkpoint=None,
                 potential: str = "frozen", corrector: int = 1,
                 calculation=None, becsum=()) -> RealTimeResult:
    """The current ``J(t)`` of a crystal driven by ``pulse``.

    Args:
        system, pseudos, density: the converged ground state.
        pulse: a :class:`~defumat.realtime.pulse.Pulse`.
        dt: the step in Hartree atomic units of time (Elk's ``dtimes``).
        duration, start: the run's span; the pulse's own when not given.
        kpoints: an explicit whole k-set; refused if it is a wedge.
        grid: an unshifted Monkhorst-Pack grid to build, reduced by the field's
            little group when ``little_group`` is set; the run's own grid when
            neither this nor ``kpoints`` is given.
        nbnd: how many bands the fixed-density solve resolves before the
            occupied ones are kept.
        conv_thr: that solve's threshold. The ground state's residual is a
            perturbation that oscillates at interband frequencies for the whole
            run, so a current asserted to some digits needs it tight.
        checkpoint: a path the driver saves its progress to after every k-chunk.
        potential: ``'frozen'`` at the ground state's, the independent-particle
            response; ``'hartree'`` with the Hartree potential updated in time
            (local fields); ``'hxc'`` with the Hartree and exchange-correlation
            potentials updated, the adiabatic functional
            (:mod:`defumat.realtime.selfconsistent`).
        corrector: with the potential updated, the corrections of the midpoint
            potential after its extrapolation; 0 is the extrapolation alone.
    """
    kset, rotations, group = _kset(system, pulse, kpoints, grid, little_group)
    calc, states, weights, v_scf, ddd_paw = _occupied_states(
        system, pseudos, density, kset, nbnd=nbnd, conv_thr=conv_thr,
        k_batch=k_batch, calculation=calculation, becsum=becsum)
    return propagate(calc, states, weights, v_scf, pulse, dt=dt, duration=duration,
                     start=start, propagator=propagator, k_batch=k_batch,
                     block_steps=block_steps, checkpoint=checkpoint,
                     symmetrise=rotations, potential=potential, corrector=corrector,
                     density_symmetry=group, ddd_paw=ddd_paw)


def run_realtime_dielectric(system, pseudos, density, *, direction=(1.0, 0.0, 0.0),
                            broadening: float = 0.1, frequencies=None,
                            window: float = 10.0, nw: int = 400,
                            strength: float | None = None, dt: float = 0.1,
                            duration: float | None = None, kpoints=None, grid=None,
                            little_group: bool = True, nbnd: int | None = None,
                            conv_thr: float = 1.0e-10, k_batch="default",
                            block_steps: int = 1000, subtract_static: bool = False,
                            potential: str = "frozen", corrector: int = 1,
                            calculation=None, becsum=()) -> KickResponse:
    """``sigma(w + i eta)`` and ``eps`` from a kick, Elk's task 481.

    ``broadening`` and ``frequencies`` (or ``window`` and ``nw``) are in eV.
    With ``strength = None`` the response is the **exact first order**, one
    ``jvp`` through the propagation in the kick's amplitude
    (:func:`~defumat.realtime.orders.propagate_orders`), so nothing nonlinear
    is in it and no strength has to be chosen small; a number runs a finite
    kick of that many 1/bohr, as Elk does. The run lasts ``22/eta`` unless
    ``duration`` says otherwise, which leaves ``exp(-22)`` of the window
    uncovered. ``potential`` is :func:`run_realtime`'s: with ``'hartree'`` or
    ``'hxc'`` the result is ``eps_M`` with local fields, the applied field
    being the whole macroscopic one.
    """
    eta = float(broadening) * EV_TO_HA
    if frequencies is None:
        frequencies = np.linspace(0.0, float(window), int(nw))
    frequencies = np.asarray(frequencies, dtype=float) * EV_TO_HA
    if duration is None:
        duration = 22.0 / eta
    unit = np.asarray(direction, dtype=float)
    unit = unit / np.linalg.norm(unit)
    kick = Kick(strength=1.0 if strength is None else float(strength),
                direction=tuple(unit))
    kset, rotations, group = _kset(system, kick, kpoints, grid, little_group)
    calc, states, weights, v_scf, ddd_paw = _occupied_states(
        system, pseudos, density, kset, nbnd=nbnd, conv_thr=conv_thr,
        k_batch=k_batch, calculation=calculation, becsum=becsum)
    update = dict(potential=potential, corrector=corrector, symmetrise=rotations,
                  density_symmetry=group)
    if potential == "frozen":
        update["ddd_paw"] = ddd_paw
    if strength is None:
        orders = propagate_orders(calc, states, weights, v_scf, kick, dt=dt, order=1,
                                  start=0.0, duration=duration, k_batch=k_batch,
                                  block_steps=block_steps, **update)
        times, current = orders.times, orders.currents[1]
    else:
        result = propagate(calc, states, weights, v_scf, kick, dt=dt, start=0.0,
                           duration=duration, k_batch=k_batch, block_steps=block_steps,
                           **update)
        times, current = result.times, result.current
    return conductivity_from_kick(times, current, kick.strength, unit, frequencies, eta,
                                  subtract_static=subtract_static)


def run_hhg(system, pseudos, density, pulse: Pulse, *, dt: float = 0.1,
            duration: float | None = None, window: str = "hann", highest: int = 40,
            samples: int = 4000, kpoints=None, grid=None, little_group: bool = True,
            nbnd: int | None = None, conv_thr: float = 1.0e-10, k_batch="default",
            block_steps: int = 400, checkpoint=None, potential: str = "frozen",
            corrector: int = 1, calculation=None, becsum=()) -> HarmonicSpectrum:
    """The high-harmonic spectrum ``|w J(w)|^2`` of a crystal driven by ``pulse``.

    The propagation of :func:`run_realtime`, then
    :func:`~defumat.realtime.spectra.harmonic_spectrum` against the pulse's own
    frequency. The spectrum of one cell without dephasing is noisy above the gap
    and converges slowly in k; the window and the mesh are the two knobs, and
    published work either adds a dephasing time or propagates the pulse through
    the sample (arXiv:1705.10707). The run itself is on the result as
    ``.realtime``.
    """
    omega = getattr(pulse, "omega", None)
    if omega is None:
        raise ValueError("run_hhg needs a pulse with a carrier frequency (omega)")
    result = run_realtime(system, pseudos, density, pulse, dt=dt, duration=duration,
                          kpoints=kpoints, grid=grid, little_group=little_group,
                          nbnd=nbnd, conv_thr=conv_thr, k_batch=k_batch,
                          block_steps=block_steps, checkpoint=checkpoint,
                          potential=potential, corrector=corrector,
                          calculation=calculation, becsum=becsum)
    spectrum = harmonic_spectrum(result.times, result.current, float(omega),
                                 window=window, highest=highest, samples=samples)
    spectrum.realtime = result
    return spectrum


def run_harmonic_orders(system, pseudos, density, *, frequency: float,
                        broadening: float = 0.1, direction=(1.0, 0.0, 0.0),
                        order: int = 3, eta_t: float = 20.0,
                        steps_per_period: int | None = None, kpoints=None, grid=None,
                        little_group: bool = True, nbnd: int | None = None,
                        conv_thr: float = 1.0e-10, k_batch="default",
                        block_steps: int = 400, potential: str = "frozen",
                        corrector: int = 1, calculation=None, becsum=()) -> OrdersResult:
    """``J^(n)(t)`` under the adiabatic field ``lam exp(eta t) cos(w t) e``, ``n <= order``.

    ``frequency`` and ``broadening`` in eV. The run starts ``eta_t/eta``
    before ``t = 0``, rounded up to whole periods, and the step divides the
    period ``steps_per_period`` times, so the Fourier projection over the last
    period is exact for the harmonics the grid resolves. Without it the step
    is the largest whole fraction of the period inside 0.9 of the
    propagator's stability bound, and never fewer than 200 a period. ``exp(-eta_t)`` times
    a resonance factor is the start transient left in every order, and it
    does not decay because the evolution is unitary: measured on zincblende
    AlAs against the dense hierarchy, 1.4e-5 to 3.0e-5 at ``eta_t = 20`` and
    400 steps a period, against 0.4 to 4.6 per cent at ``eta_t = 6``.
    """
    omega = float(frequency) * EV_TO_HA
    eta = float(broadening) * EV_TO_HA
    unit = np.asarray(direction, dtype=float)
    unit = unit / np.linalg.norm(unit)
    period = 2.0 * math.pi / omega
    length = math.ceil(float(eta_t) / eta / period) * period
    shape = Adiabatic(amplitude=1.0, omega=omega, eta=eta, eta_t=length * eta,
                      polarization=tuple(unit))
    kset, rotations, group = _kset(system, shape, kpoints, grid, little_group)
    calc, states, weights, v_scf, ddd_paw = _occupied_states(
        system, pseudos, density, kset, nbnd=nbnd, conv_thr=conv_thr,
        k_batch=k_batch, calculation=calculation, becsum=becsum)
    if steps_per_period is None:
        from defumat.realtime.propagate import largest_stable_step

        largest = largest_stable_step(calc, states, weights, v_scf, k_batch=k_batch,
                                      ddd_paw=ddd_paw)
        steps_per_period = max(200, int(math.ceil(period / (0.9 * largest))))
    extra = {"ddd_paw": ddd_paw} if potential == "frozen" else {}
    result = propagate_orders(calc, states, weights, v_scf, shape,
                              dt=period / int(steps_per_period), order=order,
                              start=-length, duration=length, k_batch=k_batch,
                              block_steps=block_steps, potential=potential,
                              corrector=corrector, symmetrise=rotations,
                              density_symmetry=group, **extra)
    return result


@dataclass
class ThirdHarmonic:
    """``chi^(3)`` of a cubic crystal at one frequency, from the real-time orders.

    Attributes:
        frequency: ``hbar w`` in eV.
        broadening: ``eta`` in eV, per photon.
        chi_xxxx_3w: ``chi^(3)_xxxx(-3w; w, w, w)`` in m^2/V^2, at ``3w + 3 i eta``.
        chi_xxyy_3w: ``chi^(3)_xxyy(-3w; w, w, w)``, from a field along
            ``[110]``, or ``nan`` when only ``[100]`` was run.
        chi_xxxx_w: ``chi^(3)_xxxx(-w; w, w, -w)`` in m^2/V^2, at ``w + 3 i eta``,
            the intensity-dependent refractive index's.
        orders: the :class:`~defumat.realtime.orders.OrdersResult` of each field
            direction, by name.
    """

    frequency: float
    broadening: float
    chi_xxxx_3w: complex
    chi_xxyy_3w: complex
    chi_xxxx_w: complex
    orders: dict = field(default_factory=dict)


def chi2_from_orders(orders: OrdersResult, axis=0):
    """``chi(-2w; w, w)`` in atomic units, contracted with the field, along ``axis``.

    With ``kappa = (lam/2)(exp(-i z t) + exp(i zbar t))``, ``z = w + i eta``,
    the field's amplitude is ``E(z) = i lam z/2``, the polarisation at the
    output frequency ``2z`` is ``P = i J_(2,2) / (2z)``, since ``J = dP/dt``,
    and ``P(2z) = chi E(z) E(z)`` with one ordering of the two equal input
    frequencies. So ``chi(2w) = -2 i J_(2,2) / z^3``, the response at
    ``2w + 2 i eta``, and what it is is ``sum_bc chi^abc e_b e_c`` for the unit
    polarisation ``e`` and the component ``a`` that ``axis`` selects: in a
    zincblende crystal ``chi_xyz`` itself in the ``x`` current of a field along
    ``[011]``, and two thirds of it along ``[111]``. Times
    :data:`~defumat.response.shg.CHI2_AU_TO_PM_PER_V` it is in pm/V.

    **It is minus the conjugate of** :func:`~defumat.response.shg.second_harmonic`,
    which evaluates at ``w - i eta`` and carries no charge, where this is the
    response of an electron of charge -1 (``chi^(2)`` is odd in the charge).
    Measured on AlAs, 1.9e-2 apart at 23 bands; with the projectors removed,
    this formula on the dense hierarchy's ``J_(2,2)`` and every band of the sum
    agree to 4.3e-5, and the 5 to 9 per cent left with them in is the curvature
    of the projectors, which the sum over states does not carry
    (``tests/regression/test_realtime_shg.py``).
    """
    shape = orders.shape
    z = shape.omega + 1j * shape.eta
    j22 = complex(orders.component(2, 2, axis=axis))
    return -2j * j22 / z**3


def chi3_from_orders(orders: OrdersResult, axis=0):
    """``(chi(-3w; w,w,w), chi(-w; w,w,-w))`` in atomic units, along ``axis``.

    With ``kappa = (lam/2)(exp(-i z t) + exp(i zbar t))``, ``z = w + i eta``,
    the field's amplitudes are ``E(z) = i lam z/2`` and ``E(-zbar) = -i lam zbar/2``,
    the polarisation is ``P = i J / Omega_s`` at the output frequency, and
    ``P(Omega_s) = D chi E E E`` with ``D`` the number of distinct
    permutations of the input frequencies (one for ``3z``, three for
    ``z + z - zbar``). So ``chi(3w) = -8 J_(3,3) / (3 z^4)`` and
    ``chi(w) = 8 J_(3,1) / (3 (2z - zbar) z^2 zbar)``.
    """
    shape = orders.shape
    z = shape.omega + 1j * shape.eta
    j33 = complex(orders.component(3, 3, axis=axis))
    j31 = complex(orders.component(3, 1, axis=axis))
    third = -8.0 * j33 / (3.0 * z**4)
    first = 8.0 * j31 / (3.0 * (2.0 * z - np.conj(z)) * z**2 * np.conj(z))
    return third, first


def run_third_harmonic(system, pseudos, density, *, frequency: float,
                       broadening: float = 0.1, both_directions: bool = True,
                       eta_t: float = 20.0, steps_per_period: int | None = None,
                       kpoints=None, grid=None, little_group: bool = True,
                       nbnd: int | None = None, conv_thr: float = 1.0e-10,
                       k_batch="default", block_steps: int = 400,
                       potential: str = "frozen", corrector: int = 1,
                       calculation=None, becsum=()) -> ThirdHarmonic:
    """``chi^(3)`` of a cubic crystal at ``frequency`` (eV) by the real-time route.

    A field along ``[100]`` gives ``chi_xxxx``; one along ``[110]`` gives
    ``(chi_xxxx + 3 chi_xxyy)/(2 sqrt 2)`` in the ``x`` component of the
    current, by the intrinsic permutation symmetry of the third harmonic, and
    with it ``chi_xxyy``. One frequency per pair of runs: a spectrum is the
    frequency-domain hierarchy's job (``HARMONICS-NEXT.md``).
    """
    options = dict(frequency=frequency, broadening=broadening, eta_t=eta_t,
                   steps_per_period=steps_per_period, kpoints=kpoints, grid=grid,
                   little_group=little_group, nbnd=nbnd, conv_thr=conv_thr,
                   k_batch=k_batch, block_steps=block_steps, potential=potential,
                   corrector=corrector, calculation=calculation, becsum=becsum)
    along = run_harmonic_orders(system, pseudos, density, direction=(1.0, 0.0, 0.0),
                                **options)
    third, first = chi3_from_orders(along, axis=0)
    runs = {"100": along}
    xxyy = complex("nan")
    if both_directions:
        diagonal = run_harmonic_orders(system, pseudos, density,
                                       direction=(1.0, 1.0, 0.0), **options)
        runs["110"] = diagonal
        combined, _ = chi3_from_orders(diagonal, axis=0)
        # the [110] run has unit amplitude along (1,1,0)/sqrt 2, so its x current
        # is (chi_xxxx + 3 chi_xxyy) / (2 sqrt 2) per unit amplitude cubed
        xxyy = (2.0 * math.sqrt(2.0) * combined - third) / 3.0
    return ThirdHarmonic(frequency=float(frequency), broadening=float(broadening),
                         chi_xxxx_3w=third * CHI3_AU_TO_SI,
                         chi_xxyy_3w=xxyy * CHI3_AU_TO_SI,
                         chi_xxxx_w=first * CHI3_AU_TO_SI, orders=runs)


@dataclass
class NonlinearSpectrum:
    """The perturbative orders of the current over a frequency axis, at a frozen potential.

    From the frequency-domain hierarchy (:mod:`defumat.realtime.hierarchy`), the
    same orders the real-time route projects out of one run per frequency.

    Attributes:
        frequencies: ``(nw,)``, the drive's ``hbar w`` in eV.
        broadening: ``eta`` in eV, per photon.
        direction: the field's unit vector.
        components: ``{(n, m): (nw, 3)}``, ``J_(n,m)`` in Hartree atomic units per
            unit of the amplitude to the ``n``, the response at ``m w + i n eta``
            (:meth:`~defumat.realtime.orders.OrdersResult.component`'s).
        iterations: ``(nw,)``, the largest BiCGStab iteration count of each
            frequency over the k-points, the bands and the orders.
        residual: ``(nw,)``, the largest final relative residual.
        computed_bands: the bands the hierarchy's projector holds exactly.
        volume: the cell volume in bohr^3.
    """

    frequencies: np.ndarray
    broadening: float
    direction: np.ndarray
    components: dict
    iterations: np.ndarray
    residual: np.ndarray
    computed_bands: int
    volume: float
    #: With the potential updated, the same run's currents at the ground
    #: state's potential, ``{(1, 1): (nw, 3)}``; the shift between the two is
    #: free of the frozen route's band-curvature term. ``None`` otherwise.
    frozen_components: dict | None = None
    #: With the potential updated, the outer GMRES products of each frequency.
    outer: np.ndarray | None = None
    potential: str = "frozen"

    def _z(self):
        return (np.asarray(self.frequencies) + 1j * float(self.broadening)) * EV_TO_HA

    def component(self, n: int, m: int, axis=None):
        """``J_(n,m)`` at every frequency: ``(nw, 3)``, or ``(nw,)`` along ``axis``."""
        value = np.asarray(self.components[(n, m)])
        if axis is None:
            return value
        if np.ndim(axis) == 0:
            return value[:, int(axis)]
        return value @ np.asarray(axis, dtype=float)

    def epsilon(self, axis=0, frozen: bool = False):
        """``eps(w + i eta)`` along ``axis`` from ``J_(1,1)``: ``1 + 8 pi J / z^2``.

        With ``E(z) = i z lam/2``, ``sigma = J_(1,1) / E`` and
        ``eps = 1 + 4 pi i sigma / z``. With the potential updated it is ``eps_M``
        with local fields; ``frozen = True`` gives the same run's frozen value.
        On a coarse mesh both carry the velocity gauge's band-curvature term,
        ``i D/(Omega z)`` in ``sigma``, and their difference does not.
        """
        z = self._z()
        source = self.frozen_components if frozen else self.components
        if source is None:
            raise ValueError("no frozen currents: this run is at the frozen potential already")
        value = np.asarray(source[(1, 1)])
        current = value[:, int(axis)] if np.ndim(axis) == 0 else value @ np.asarray(axis, dtype=float)
        return 1.0 + 8.0 * math.pi * current / z**2

    def chi2(self, axis=0):
        """``chi^(2)(-2w; w, w)`` in pm/V along ``axis``, :func:`chi2_from_orders`'s formula."""
        from defumat.response.shg import CHI2_AU_TO_PM_PER_V

        z = self._z()
        return -2j * self.component(2, 2, axis) / z**3 * CHI2_AU_TO_PM_PER_V

    def chi3(self, axis=0):
        """``(chi^(3)(-3w; w,w,w), chi^(3)(-w; w,w,-w))`` in m^2/V^2, :func:`chi3_from_orders`'s."""
        z = self._z()
        third = -8.0 * self.component(3, 3, axis) / (3.0 * z**4)
        first = 8.0 * self.component(3, 1, axis) / (3.0 * (2.0 * z - np.conj(z)) * z**2
                                                     * np.conj(z))
        return third * CHI3_AU_TO_SI, first * CHI3_AU_TO_SI


def run_nonlinear_spectrum(system, pseudos, density, *, frequencies, broadening: float = 0.1,
                           direction=(1.0, 0.0, 0.0), order: int = 3, kpoints=None,
                           grid=None, little_group: bool = True, nbnd: int | None = None,
                           conv_thr: float = 1.0e-10, k_batch="default",
                           tolerance: float = 1.0e-10, max_iterations: int = 500,
                           potential: str = "frozen", calculation=None,
                           becsum=()) -> NonlinearSpectrum:
    """The orders of the current, ``chi^(2)`` and ``chi^(3)`` over ``frequencies`` (eV).

    The frequency-domain hierarchy at the ground state's frozen potential, the
    spectrum the real-time route of :func:`run_harmonic_orders` gives one
    frequency at a time, with no time step and no start transient: each
    frequency is nine iterative solves per band and k-point. ``nbnd`` is the
    fixed-density solve's band count, ``3 nocc + 4`` by default, and every band
    below its top four is held exactly by the hierarchy's projector, which is
    what bounds the iteration count. The k-set and the little group of the field
    are :func:`run_realtime`'s. For a smeared run the carried bands keep their
    ground-state occupations, so there is no occupation response, as in the
    real-time route.
    """
    from defumat.realtime.hierarchy import hierarchy_linear_self_consistent, hierarchy_orders

    if potential != "frozen" and order != 1:
        raise NotImplementedError(
            "the frequency-domain hierarchy with the potential updated is implemented at "
            "first order only: the second and third orders need the induced potentials at 2w "
            "and 3w, each a fixed point at every frequency, and they would carry the velocity "
            "gauge's second-order density artefact, 7 per cent of rho on a 2x2x2 mesh. The "
            "real-time route has them at one frequency a run "
            "(get_harmonic_orders(potential='hxc'))")
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
    unit = np.asarray(direction, dtype=float)
    unit = unit / np.linalg.norm(unit)
    kick = Kick(strength=1.0, direction=tuple(unit))
    kset, rotations, group = _kset(system, kick, kpoints, grid, little_group)
    if nbnd is None:
        nelec_guess = sum(pseudos[t].z_valence for t in system.structure.types)
        nbnd = 3 * _occupied_count(system, nelec_guess) + 4
    calc, states, weights, v_scf, bands, energies, ddd_paw = _solved_states(
        system, pseudos, density, kset, nbnd=nbnd, conv_thr=conv_thr, k_batch=k_batch,
        calculation=calculation, becsum=becsum)
    computed = bands.shape[-2] - 4
    if computed <= weights.shape[-1]:
        raise ValueError(
            f"nbnd = {bands.shape[1]} leaves no conduction band below the top four for the "
            "hierarchy's projector; pass a larger nbnd")
    common = dict(omegas=frequencies * EV_TO_HA, eta=float(broadening) * EV_TO_HA,
                  direction=unit, tolerance=tolerance, max_iterations=max_iterations,
                  symmetrise=rotations, k_batch=k_batch)
    if potential == "frozen":
        out = hierarchy_orders(calc, bands[:, :, :computed], energies[:, :, :computed],
                               weights, v_scf, order=order, ddd_paw=ddd_paw, **common)
        return NonlinearSpectrum(frequencies=frequencies, broadening=float(broadening),
                                 direction=unit, components=out["components"],
                                 iterations=out["iterations"], residual=out["residual"],
                                 computed_bands=computed, volume=out["volume"])
    out = hierarchy_linear_self_consistent(calc, bands[:, :, :computed],
                                           energies[:, :, :computed],
                                           weights, v_scf, potential=potential,
                                           density_symmetry=group, **common)
    return NonlinearSpectrum(frequencies=frequencies, broadening=float(broadening),
                             direction=unit, components=out["components"],
                             iterations=out["iterations"], residual=out["outer_residual"],
                             computed_bands=computed, volume=out["volume"],
                             frozen_components=out["frozen"], outer=out["outer"],
                             potential=potential)
