"""Density mixing: turning a fixed-point iteration into a convergent one.

Feeding the output density straight back in (``rho_in = rho_out``) diverges for
anything but the smallest systems -- the charge sloshes between regions of the
cell, amplified each iteration by the Hartree term. Mixing damps that.

Two schemes, behind a registry (rule R4):

* **linear**: ``rho_in + beta (rho_out - rho_in)``. Robust, slow, and the right
  thing to check a new system with.
* **anderson**: extrapolate from the history of residuals to the density whose
  residual has the smallest norm in the span. This is Pulay/DIIS, and is what
  makes convergence take ten iterations instead of a hundred.

QE's default is Broyden mixing (``mix_rho.f90``), which is closely related;
Anderson is chosen here because it is the same idea with far less bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Mixer", "LinearMixer", "AndersonMixer", "get_mixer", "MIXERS"]


class Mixer:
    """Interface: given the input and output density, propose the next input."""

    def mix(self, rho_in: np.ndarray, rho_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        pass


@dataclass
class LinearMixer(Mixer):
    beta: float = 0.7

    def mix(self, rho_in, rho_out):
        return rho_in + self.beta * (rho_out - rho_in)


@dataclass
class AndersonMixer(Mixer):
    """Anderson/Pulay mixing with a bounded history."""

    beta: float = 0.7
    history: int = 8
    _densities: list = field(default_factory=list, repr=False)
    _residuals: list = field(default_factory=list, repr=False)

    def reset(self):
        self._densities.clear()
        self._residuals.clear()

    def mix(self, rho_in, rho_out):
        rho_in = np.asarray(rho_in).ravel()
        residual = np.asarray(rho_out).ravel() - rho_in

        self._densities.append(rho_in)
        self._residuals.append(residual)
        if len(self._densities) > self.history:
            self._densities.pop(0)
            self._residuals.pop(0)

        n = len(self._residuals)
        if n == 1:
            return (rho_in + self.beta * residual).reshape(np.asarray(rho_out).shape)

        # Minimise |sum_i c_i r_i| subject to sum_i c_i = 1, by solving the
        # constrained least-squares problem in the residual basis.
        overlap = np.empty((n + 1, n + 1))
        overlap[:n, :n] = [[float(a @ b) for b in self._residuals] for a in self._residuals]
        overlap[:n, n] = 1.0
        overlap[n, :n] = 1.0
        overlap[n, n] = 0.0

        rhs = np.zeros(n + 1)
        rhs[n] = 1.0
        try:
            coefficients = np.linalg.solve(overlap, rhs)[:n]
        except np.linalg.LinAlgError:
            # A degenerate history means the residuals are linearly dependent;
            # drop it and take a plain linear step rather than failing.
            self.reset()
            return (rho_in + self.beta * residual).reshape(np.asarray(rho_out).shape)

        mixed = sum(c * (d + self.beta * r) for c, d, r in
                    zip(coefficients, self._densities, self._residuals))
        return np.asarray(mixed).reshape(np.asarray(rho_out).shape)


#: Name -> mixer, as written in an input file's ``mixing_mode``.
MIXERS = {
    "linear": LinearMixer,
    "plain": AndersonMixer,  # QE's 'plain' is Broyden; Anderson is the stand-in
    "anderson": AndersonMixer,
    "broyden": AndersonMixer,
}


def get_mixer(name: str, **kwargs) -> Mixer:
    try:
        return MIXERS[name.lower()](**kwargs)
    except KeyError as error:
        raise ValueError(f"unknown mixing mode {name!r}; expected one of {sorted(MIXERS)}") from error
