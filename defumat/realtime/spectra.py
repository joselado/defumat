"""What a current in time becomes: a conductivity, a dielectric function, a harmonic spectrum.

**The linear response from a kick.** A step in ``kappa`` at ``t = 0`` is a field
``E(t) = -kappa_0 delta(t)``, whose transform is ``-kappa_0`` at every
frequency, so the conductivity is

    sigma_ab(z) = -(1/kappa_0) int_0^inf J_a(t) exp(i z t) dt,   z = w + i eta,

and ``eps(z) = 1 + 4 pi i sigma(z)/z`` in Gaussian atomic units, which is Elk's
task 481 (``dielectric_tdrt.f90`` with ``E(w)`` a constant). The window
``exp(-eta t)`` is the imaginary part of ``z``, so the transform is exactly the
response at the complex frequency ``w + i eta``, prefactor included, and a
reference that is analytic in ``z`` can be compared as an identity rather than
an agreement. The integral is Simpson's rule on the recorded grid: after the
kick the Hamiltonian is constant and the step is exact to its Taylor order, so
the quadrature is the whole of the error and its ``dt^4`` is what keeps it
below the propagation's.

**What is not in it and is in a sum over states.** The current here carries the
exact diamagnetic term, ``sum w <d^2 H/dk^2>``, where a Kubo sum
replaces it by its value from the f-sum rule. On a finite mesh the two differ
by the band curvature summed over the mesh,
``D = sum_nk w d^2 eps_nk / dk^2``, which is exponentially small in the mesh
density for an insulator and is not small on a coarse one: the real-time
conductivity is the Kubo one plus ``i D / (Omega z)``, a spurious Drude term
(measured in review at ``-0.71`` against ``Pi(0) = -0.94`` at Gamma on
two-atom silicon, so the size of the reference). ``subtract_static`` removes it
the way Elk's ``jtconst0`` does, by taking the constant part of ``J`` out,
which is right only when ``J`` has settled to it; on a converged mesh it is not
needed.

**The harmonic spectrum** is the power the current radiates,
``|FT[dJ/dt]|^2 = w^2 |J(w)|^2``, taken with a window that goes to zero at
both ends of the run (``hann`` by default), since a current cut off while it is
still oscillating puts a sinc on every harmonic that hides the weaker ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from defumat.units import HARTREE_TO_EV

__all__ = ["conductivity_from_kick", "KickResponse", "harmonic_spectrum",
           "HarmonicSpectrum", "WINDOWS"]


def _simpson(values, dt):
    """Composite Simpson's rule along axis 0, an odd number of samples or one trapezoid at the end."""
    values = np.asarray(values)
    n = values.shape[0]
    if n < 3:
        return np.trapezoid(values, dx=dt, axis=0)
    if n % 2 == 1:
        weights = np.ones(n)
        weights[1:-1:2] = 4.0
        weights[2:-1:2] = 2.0
        return dt / 3.0 * np.tensordot(weights, values, axes=(0, 0))
    head = _simpson(values[:-1], dt)
    return head + 0.5 * dt * (values[-2] + values[-1])


@dataclass
class KickResponse:
    """The linear response to a kick along one direction, Hartree atomic units.

    Attributes:
        frequencies: ``(nw,)`` in Hartree.
        eta: the broadening in Hartree, the imaginary part of every ``z``.
        sigma: ``(nw, 3)``, ``sigma_ab(w + i eta)`` for the kick's direction
            ``b`` and every ``a``, in ``e^2/(hbar a_0)``.
        epsilon: ``(nw, 3)``, ``delta_ab + 4 pi i sigma_ab / z``.
        direction: the kick's unit vector.
    """

    frequencies: np.ndarray
    eta: float
    sigma: np.ndarray
    epsilon: np.ndarray
    direction: np.ndarray

    @property
    def frequencies_ev(self) -> np.ndarray:
        return np.asarray(self.frequencies) * HARTREE_TO_EV


def conductivity_from_kick(times, current, strength: float, direction, frequencies,
                           eta: float, kick_time: float = 0.0,
                           subtract_static: bool = False) -> KickResponse:
    """``sigma(w + i eta)`` and ``eps`` from the current after a kick of ``strength``.

    ``times`` and ``current`` are a :class:`~defumat.realtime.propagate.RealTimeResult`'s
    (or an order of :class:`~defumat.realtime.orders.OrdersResult` with
    ``strength = 1``); the integral runs from ``kick_time`` to the end, and the
    run must be long enough for ``exp(-eta T)`` to be negligible, since the
    truncation is not corrected. ``subtract_static`` removes the time average of
    ``J`` after the kick first (Elk's ``jtconst0``), which takes the mesh's
    spurious Drude term out with it.
    """
    times = np.asarray(times, dtype=float)
    current = np.asarray(current)
    after = times >= kick_time - 1e-12
    t = times[after] - kick_time
    j = current[after]
    if subtract_static:
        j = j - _simpson(j, t[1] - t[0]) / (t[-1] - t[0])
    dt = float(t[1] - t[0])
    z = np.asarray(frequencies, dtype=float) + 1j * float(eta)
    phase = np.exp(1j * np.outer(t, z))  # (nt, nw)
    transform = _simpson(phase[:, :, None] * j[:, None, :], dt)  # (nw, 3)
    sigma = -transform / float(strength)
    unit = np.asarray(direction, dtype=float)
    unit = unit / np.linalg.norm(unit)
    epsilon = unit[None, :] + 4.0 * np.pi * 1j * sigma / z[:, None]
    return KickResponse(frequencies=np.asarray(frequencies, dtype=float), eta=float(eta),
                        sigma=sigma, epsilon=epsilon, direction=unit)


def _hann(n):
    return np.sin(np.pi * np.arange(n) / (n - 1)) ** 2


#: Windows for :func:`harmonic_spectrum`, each a function of the sample count.
WINDOWS = {
    "hann": _hann,
    "none": lambda n: np.ones(n),
    "blackman": lambda n: np.blackman(n),
}


@dataclass
class HarmonicSpectrum:
    """``w^2 |J(w)|^2``, the power a current radiates, against the drive's harmonics.

    Attributes:
        frequencies: ``(nw,)`` in Hartree.
        orders: ``frequencies / omega``, the harmonic order.
        intensity: ``(nw, 3)``, ``|w J_a(w)|^2`` for each cartesian component,
            in Hartree atomic units, with the window applied.
        total: ``(nw,)``, the sum over components.
        omega: the drive's frequency in Hartree.
        window: the window's name.
    """

    frequencies: np.ndarray
    orders: np.ndarray
    intensity: np.ndarray
    total: np.ndarray
    omega: float
    window: str

    @property
    def frequencies_ev(self) -> np.ndarray:
        return np.asarray(self.frequencies) * HARTREE_TO_EV

    def harmonic(self, order: int, width: float = 0.25) -> float:
        """The largest intensity within ``width`` of a harmonic order."""
        near = np.abs(self.orders - order) <= width
        return float(self.total[near].max()) if near.any() else float("nan")

    def harmonics(self, highest: int, width: float = 0.25) -> np.ndarray:
        """:meth:`harmonic` for orders 1 to ``highest``."""
        return np.asarray([self.harmonic(n, width) for n in range(1, highest + 1)])

    def cutoff(self, floor: float = 1e-3, highest: int | None = None,
               width: float = 0.25) -> int:
        """The highest odd harmonic whose peak is above ``floor`` times the fundamental's.

        A plateau of nearly equal odd harmonics followed by a fall is what
        marks the cutoff of a solid's spectrum; this rule reads the end of the
        plateau at a stated level and is not a fit.
        """
        if highest is None:
            highest = int(self.orders.max())
        peaks = self.harmonics(highest, width)
        reference = peaks[0]
        odd = [n for n in range(1, highest + 1, 2)
               if np.isfinite(peaks[n - 1]) and peaks[n - 1] >= floor * reference]
        return max(odd) if odd else 1


def harmonic_spectrum(times, current, omega: float, *, window: str = "hann",
                      highest: int = 40, samples: int = 4000) -> HarmonicSpectrum:
    """``|w J(w)|^2`` from ``J(t)`` up to ``highest`` harmonics of ``omega`` (Hartree).

    The transform is taken directly on a frequency grid of ``samples`` points
    rather than by an FFT, so the grid is the one the harmonics are read on and
    not the one the run length happens to give.
    """
    times = np.asarray(times, dtype=float)
    current = np.asarray(current)
    shape = WINDOWS[window](len(times))
    dt = float(times[1] - times[0])
    frequencies = np.linspace(0.0, highest * float(omega), samples)
    phase = np.exp(1j * np.outer(frequencies, times - times[0]))  # (nw, nt)
    transform = phase @ (current * shape[:, None]) * dt  # (nw, 3)
    intensity = np.abs(frequencies[:, None] * transform) ** 2
    return HarmonicSpectrum(frequencies=frequencies, orders=frequencies / float(omega),
                            intensity=intensity, total=intensity.sum(axis=-1),
                            omega=float(omega), window=window)
