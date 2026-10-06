"""The field a real-time run is driven by, carried as a wavevector ``kappa(t)``.

A uniform field in the velocity gauge is a vector potential ``A(t)`` with
``E = -(1/c) dA/dt``, and what the electrons see is a shift of the crystal
momentum, so the variable carried here is

    kappa(t) = A(t) / c          (1/bohr),       dkappa/dt = -E(t),

in Hartree atomic units, where ``c = 137.036``. The number is the same in
Hartree and in Rydberg units, since it is a wavevector, and it is the only
form in which the field enters the Hamiltonian: ``H(k + kappa(t))`` on the
sphere built for ``k`` (:mod:`defumat.realtime.propagate`).

**Times here are Hartree atomic units of time, 24.189 as**, Elk's ``dtimes``
and ``tstime``, and every frequency is in Hartree, so that a pulse written for
an Elk input reads the same here. The propagator works in ``hbar/Ry``, twice
that, and the conversion is made once, in the driver. The pulses are written in
``jax.numpy`` so that ``E(t)`` is the derivative of ``kappa(t)`` taken by
differentiation, never a second expression.

The shapes are Elk's three (``src/genafieldt.f90``: a Gaussian-enveloped sine
with a phase and a chirp, a polynomial ramp, a step), the ``sin^2`` envelope of
the high-harmonic literature, and the adiabatic switch-on
``exp(eta t) cos(w t)`` on ``[-T, 0]`` that turns the perturbative orders of the
current into response functions at complex frequencies
(``HARMONICS-NEXT.md``, "The physics"). Each is behind a name registry
(:func:`get_pulse`), and :class:`Sum` superposes any of them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from defumat.units import AU_SEC, C_AU, HARTREE_TO_EV, INTENSITY_AU_W_CM2

__all__ = [
    "Pulse", "Kick", "Gaussian", "Sin2", "Ramp", "Adiabatic", "Sum",
    "register_pulse", "get_pulse", "pulse_names", "field_amplitude",
    "FS_TO_AU", "EV_TO_HA",
]

#: One femtosecond in Hartree atomic units of time.
FS_TO_AU = 1.0e-15 / AU_SEC
#: One electronvolt in Hartree.
EV_TO_HA = 1.0 / HARTREE_TO_EV


def field_amplitude(intensity_w_cm2: float) -> float:
    """``E0`` in Hartree atomic units for a peak intensity in W/cm^2.

    ``I = c E0^2 / 8 pi``, the cycle-averaged intensity of a linearly polarised
    field of peak strength ``E0``, which is what Elk prints as the peak power
    density of a pulse, so ``E0 = sqrt(I / 3.51e16 W/cm^2)``.
    """
    return math.sqrt(float(intensity_w_cm2) / INTENSITY_AU_W_CM2)


def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(3)
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError("a polarisation vector must not be zero")
    return vector / norm


class Pulse:
    """A field, as ``kappa(t)`` in 1/bohr on Hartree atomic units of time.

    A subclass defines :meth:`kappa_at`, a ``jax.numpy`` function of a scalar
    time returning the ``(3,)`` wavevector. :meth:`kappa` evaluates it on a
    grid, and :meth:`efield` is ``-dkappa/dt`` by differentiation.
    """

    #: Where the run starts when the caller does not say, in Hartree a.u.
    start: float = 0.0

    def kappa_at(self, t):
        raise NotImplementedError

    @property
    def natural_duration(self) -> float | None:
        """The length of the pulse when it has one, in Hartree a.u."""
        return None

    def kappa(self, times) -> np.ndarray:
        """``(nt, 3)`` on the times given."""
        return np.asarray(jax.vmap(self.kappa_at)(jnp.asarray(times, dtype=float)))

    def efield(self, times) -> np.ndarray:
        """``E(t) = -dkappa/dt``, ``(nt, 3)`` in Hartree atomic units.

        A step in ``kappa`` is a delta in ``E`` that a grid cannot hold, so a
        :class:`Kick` reports zero here and carries its strength itself.
        """
        derivative = jax.vmap(jax.jacfwd(self.kappa_at))(
            jnp.asarray(times, dtype=float))
        return -np.asarray(derivative)

    def __add__(self, other: "Pulse") -> "Sum":
        return Sum((self, other))


@dataclass(frozen=True)
class Kick(Pulse):
    """A step in ``A`` at ``time``: ``kappa = strength * direction`` from then on.

    The field is ``E(t) = -strength delta(t - time)``, whose Fourier transform
    is the constant ``-strength`` at every frequency, so the response to it is
    the linear response at every frequency at once (Elk's task 481). It must be
    small enough to be linear: the response is ``J = sigma E`` only to first
    order in ``strength``.
    """

    strength: float
    direction: tuple = (1.0, 0.0, 0.0)
    time: float = 0.0

    def kappa_at(self, t):
        on = jnp.where(t >= self.time - 1.0e-14, 1.0, 0.0)
        return on * self.strength * jnp.asarray(_unit(self.direction))


@dataclass(frozen=True)
class Gaussian(Pulse):
    """Elk's laser pulse: ``A0 exp(-(t-t0)^2/2 s^2) sin(w (t-t0) + phi + rc t^2/2)``.

    ``amplitude`` is ``kappa0`` in 1/bohr along ``polarization`` (Elk's vector
    amplitude divided by ``c``), ``omega`` in Hartree, ``fwhm`` the full width at
    half maximum of the envelope, ``peak`` the time of its maximum, ``phase`` in
    degrees and ``chirp`` in Hartree a.u., all as ``genafieldt.f90`` takes them.
    :meth:`from_intensity` builds one from a peak intensity and a photon energy.
    """

    amplitude: float
    omega: float
    fwhm: float
    peak: float
    polarization: tuple = (1.0, 0.0, 0.0)
    phase: float = 0.0
    chirp: float = 0.0

    @classmethod
    def from_intensity(cls, intensity_w_cm2: float, photon_ev: float,
                       fwhm_fs: float, peak_fs: float | None = None,
                       polarization=(1.0, 0.0, 0.0), phase: float = 0.0):
        """A pulse of peak intensity ``I`` in W/cm^2 and photon energy in eV.

        ``kappa0 = E0 / omega``, ``E0`` from :func:`field_amplitude`. The peak
        is put three widths in when not given, where the envelope is 1e-4 of its
        maximum.
        """
        omega = float(photon_ev) * EV_TO_HA
        fwhm = float(fwhm_fs) * FS_TO_AU
        peak = 3.0 * fwhm if peak_fs is None else float(peak_fs) * FS_TO_AU
        return cls(amplitude=field_amplitude(intensity_w_cm2) / omega,
                   omega=omega, fwhm=fwhm, peak=peak,
                   polarization=tuple(polarization), phase=phase)

    @property
    def natural_duration(self) -> float:
        return 2.0 * self.peak

    def kappa_at(self, t):
        sigma = self.fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        shifted = t - self.peak
        envelope = jnp.exp(-0.5 * (shifted / sigma) ** 2)
        carrier = jnp.sin(self.omega * shifted + self.phase * math.pi / 180.0
                          + 0.5 * self.chirp * t**2)
        return envelope * carrier * self.amplitude * jnp.asarray(_unit(self.polarization))


@dataclass(frozen=True)
class Sin2(Pulse):
    """``kappa0 sin^2(pi t / T) cos(w (t - T/2) + phase)`` on ``[0, T]``, zero outside.

    The envelope of most published high-harmonic calculations in solids. ``T``
    is ``cycles`` periods of the carrier; ``kappa`` and ``E`` both vanish at
    both ends, so the pulse leaves no static vector potential behind and the
    excitation at the end is a population of the ground state's own bands.
    """

    amplitude: float
    omega: float
    cycles: float
    polarization: tuple = (1.0, 0.0, 0.0)
    phase: float = 0.0

    @classmethod
    def from_intensity(cls, intensity_w_cm2: float, photon_ev: float, cycles: float,
                       polarization=(1.0, 0.0, 0.0), phase: float = 0.0):
        omega = float(photon_ev) * EV_TO_HA
        return cls(amplitude=field_amplitude(intensity_w_cm2) / omega, omega=omega,
                   cycles=float(cycles), polarization=tuple(polarization), phase=phase)

    @property
    def natural_duration(self) -> float:
        return 2.0 * math.pi * self.cycles / self.omega

    def kappa_at(self, t):
        length = self.natural_duration
        inside = (t >= 0.0) & (t <= length)
        envelope = jnp.where(inside, jnp.sin(math.pi * t / length) ** 2, 0.0)
        carrier = jnp.cos(self.omega * (t - 0.5 * length) + self.phase * math.pi / 180.0)
        return envelope * carrier * self.amplitude * jnp.asarray(_unit(self.polarization))


@dataclass(frozen=True)
class Ramp(Pulse):
    """Elk's ramp: ``kappa0 (t - t0) (c1 + (t - t0)(c2 + (t - t0)(c3 + (t - t0) c4)))``.

    Zero before ``start``. ``amplitude`` multiplies ``direction``, as Elk's
    vector amplitude does, and the four coefficients are Elk's in Hartree a.u.
    A linear ramp of ``A`` is a constant field switched on at ``start``.
    """

    amplitude: float
    coefficients: tuple = (1.0, 0.0, 0.0, 0.0)
    direction: tuple = (1.0, 0.0, 0.0)
    start_time: float = 0.0

    def kappa_at(self, t):
        c1, c2, c3, c4 = (tuple(self.coefficients) + (0.0, 0.0, 0.0))[:4]
        s = t - self.start_time
        value = jnp.where(s > 0.0, s * (c1 + s * (c2 + s * (c3 + s * c4))), 0.0)
        return value * self.amplitude * jnp.asarray(_unit(self.direction))


@dataclass(frozen=True)
class Adiabatic(Pulse):
    """``kappa0 exp(eta t) cos(w t)`` on ``[-T, 0]``, switched on from ``-T``.

    The response of order ``n`` to it is ``exp(n eta t)`` times a periodic
    function, whose Fourier coefficients are the response functions at the
    complex frequencies ``m w + i n eta``, so ``eta`` is a broadening per photon
    and a unitary evolution needs no invented dephasing. ``T`` is
    ``eta_t / eta``, and ``exp(-eta T)`` sets the relative size of the start
    transient, which does not decay (``HARMONICS-NEXT.md``, "The claim the plan
    rests on").
    """

    amplitude: float
    omega: float
    eta: float
    eta_t: float = 14.0
    polarization: tuple = (1.0, 0.0, 0.0)

    @property
    def start(self) -> float:
        return -self.eta_t / self.eta

    @property
    def natural_duration(self) -> float:
        return self.eta_t / self.eta

    def kappa_at(self, t):
        return (jnp.exp(self.eta * t) * jnp.cos(self.omega * t) * self.amplitude
                * jnp.asarray(_unit(self.polarization)))


@dataclass(frozen=True)
class Sum(Pulse):
    """Several pulses at once; ``a + b`` builds one."""

    pulses: tuple = field(default_factory=tuple)

    @property
    def start(self) -> float:
        return min((p.start for p in self.pulses), default=0.0)

    @property
    def natural_duration(self) -> float | None:
        ends = [p.start + p.natural_duration for p in self.pulses
                if p.natural_duration is not None]
        return None if not ends else max(ends) - self.start

    def kappa_at(self, t):
        return sum(p.kappa_at(t) for p in self.pulses)

    def __add__(self, other: Pulse) -> "Sum":
        return Sum(self.pulses + (other,))


_PULSES: dict[str, type] = {}


def register_pulse(name: str, cls: type) -> None:
    """Make ``cls`` reachable as ``get_pulse(name, ...)``."""
    _PULSES[name.lower()] = cls


def get_pulse(name: str, **parameters) -> Pulse:
    """The pulse called ``name``, built from ``parameters``."""
    key = name.lower()
    if key not in _PULSES:
        raise ValueError(f"unknown pulse {name!r}; available: {pulse_names()}")
    return _PULSES[key](**parameters)


def pulse_names() -> tuple[str, ...]:
    return tuple(sorted(_PULSES))


for _name, _cls in (("kick", Kick), ("gaussian", Gaussian), ("sin2", Sin2),
                    ("ramp", Ramp), ("adiabatic", Adiabatic)):
    register_pulse(_name, _cls)
