"""Mixing the induced potential of a self-consistent response.

``solve_linter``'s loop is a fixed-point iteration on ``dV_scf`` exactly as the
ground-state SCF is one on ``rho``, and it had exactly one thing the SCF has
not: a **mixer**. All three response loops here -- the electric field, the
displacement and the strain -- advanced with one line of linear mixing,

    dvscf <- dvscf + alpha_mix (induced - dvscf)

where QE mixes with ``LR_Modules/mix_pot.f90``, a modified Broyden over the four
previous iterations. The difference is not a matter of speed. Linear mixing of a
map whose Jacobian has an eigenvalue below ``-1`` is **unstable**, and the
induced Hartree potential is ``4 pi e^2/G^2`` against the induced charge, so such
an eigenvalue is what a cell with a small smallest-``G`` has. Measured twice:

* **bilayer graphene**, 14 bohr of vacuum. ``alpha_mix = 0.7`` diverges from the
  first iteration, ``|ddv_scf|^2`` growing 1.34x per pass; 0.3 converges at 0.5x
  and needs 68 iterations.
* **rhombohedral BN**, a dense bulk crystal with no vacuum at all -- which is
  why it looked safe. ``alpha_mix = 0.7`` converges at 0.625x for *sixty-one*
  iterations, down to ``3.9e-7``, and then turns around and grows at 1.30x. A
  subdominant eigenmode with an amplification above one is invisible while the
  dominant one still dominates and inevitable once it dies, so the trace looks
  like a healthy calculation for an hour before it does not.

The second is the one that argues for this module rather than for a smaller
``alpha_mix``: there is no value of the mixing parameter a caller can be told to
use, because whether their system needs one cannot be seen until the run is most
of the way through. **A mixer that builds the inverse Jacobian from the residuals
it has already seen does not need to be told.**

**The pieces are mixed together, not separately**, which is why this wraps
:mod:`defumat.scf.mixing` rather than calling it three times. Anderson's step is
a least-squares problem over the *history of one vector*; splitting a coupled
state into two vectors and mixing each with its own history solves a different
problem, and the electric-field loop has two (``dV_scf`` and, for PAW, the
one-centre potential from ``dbecsum``). ``mix_pot`` concatenates them into one
array for the same reason.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from defumat.scf.mixing import get_mixer

__all__ = ["ResponseMixer", "DEFAULT_RESPONSE_MIXING", "ddv_scf"]


def _squared_norm(array) -> float:
    """``sum |x|^2`` over every entry, real or complex, numpy or JAX."""
    xp = np if isinstance(array, np.ndarray) else jnp
    return float(xp.real(xp.vdot(array, array)))


def ddv_scf(changes, onecentre_changes=None, *, joint: bool) -> float:
    """``ph.x``'s ``|ddv_scf|^2`` for one pass of a self-consistent response.

    ``LR_Modules/mix_pot.f90:77-83`` is

        dr2 = (|| vout - vin || / ndimtot)^2,

    the summed square of the change over the whole mixed vector divided by the
    *square* of its length in reals, with the complex potential counted as two
    reals an entry (``dfpt_kernels.f90:226``, ``:438``). ``dfpt_kernels`` tests
    ``dr2 < npert tr2_ph / npol`` and prints ``dr2 / npert`` (``:519-523``), so
    what this returns is the printed number, and the loops stop when it falls
    below ``tr2 / npol``. Two consequences of the formula, both QE's: ``dr2`` is
    ``N`` squared residuals over ``N^2``, so a fixed ``tr2`` admits an RMS
    residual of about ``sqrt(N tr2)`` per entry and loosens as the grid grows;
    and ``nspin_mag = 2`` doubles ``N`` while each channel's potential equals the
    one-channel one, so an unpolarized cell run at ``nspin = 2`` reads half.

    **The direction convention differs from ``ph.x``'s, and it is left so.**
    ``dvpsi_e.f90:83`` perturbs along ``at(:, ipol)``, a lattice vector in units
    of ``alat`` with no normalisation, where this code perturbs along Cartesian
    unit vectors. On an fcc cell each ``at`` has length ``1/sqrt(2)``, so
    ``ph.x``'s field values are half of these (measured on ``si-epsilon``: 2.00,
    2.00, 2.00, 1.98 over the first four passes); on a non-cubic cell the two
    are not related by a scalar at all, which is why no factor is applied.

    Args:
        changes: one entry per perturbation, each ``induced - input`` of that
            perturbation's grid potential, any shape.
        onecentre_changes: the same for PAW's one-centre block, which is mixed
            beside the grid potential and so belongs to the test as ``dbecsum``
            does to ``ph.x``'s (``dfpt_kernels.f90:433-436``), or ``None``.
        joint: ``True`` for ``ph.x``'s test over all the perturbations at once,
            divided by their number -- the field, whose three directions are
            ``solve_e``'s ``npert``. ``False`` for the largest single-perturbation
            value, which is ``ph.x``'s test for a one-dimensional irreducible
            representation and never looser than its test for a larger one:
            the phonon, the phonon at ``q`` and the strain, which mix every mode
            together where ``ph.x`` mixes one representation at a time.
    """
    changes = list(changes)
    others = (list(onecentre_changes) if onecentre_changes is not None
              else [None] * len(changes))
    squares, counts = [], []
    for grid, onecentre in zip(changes, others):
        square, count = _squared_norm(grid), int(np.size(grid))
        if onecentre is not None:
            square += _squared_norm(onecentre)
            count += int(np.size(onecentre))
        squares.append(square)
        counts.append(count)
    if joint:
        return sum(squares) / (2.0 * sum(counts)) ** 2 / len(changes)
    return max(s / (2.0 * n) ** 2 for s, n in zip(squares, counts))

#: What a response loop mixes with unless told otherwise. QE's ``mix_pot`` is a
#: modified Broyden; :mod:`defumat.scf.mixing` offers Anderson under that name,
#: which is the same quasi-Newton idea with a different history update.
DEFAULT_RESPONSE_MIXING = "anderson"


class ResponseMixer:
    """One mixing step over however many arrays the loop carries.

    The arrays are flattened and concatenated into the single vector the mixer's
    history is built from, then unpacked -- so a loop with two coupled pieces
    gets one Anderson problem rather than two, and a loop with one is unaffected.
    """

    def __init__(self, name: str = DEFAULT_RESPONSE_MIXING, beta: float = 0.7):
        self.mixer = get_mixer(name, beta=beta)
        self.name = name

    def mix(self, current, proposed, host: bool = False):
        """The next input, given this iteration's input and output.

        Args:
            current: one array, or a sequence of them, as the loop holds them.
            proposed: what the loop's evaluation produced from ``current``.
            host: return numpy arrays rather than device ones, for a loop that
                keeps its fields in host memory (the history already is).

        Returns the same structure, as JAX arrays unless ``host``.
        """
        one = not isinstance(current, (list, tuple))
        current = [current] if one else list(current)
        proposed = [proposed] if one else list(proposed)

        shapes = [np.asarray(a).shape for a in current]
        flat_in = np.concatenate([np.asarray(a, dtype=float).ravel() for a in current])
        flat_out = np.concatenate([np.asarray(a, dtype=float).ravel() for a in proposed])

        mixed = np.asarray(self.mixer.mix(flat_in, flat_out)).ravel()

        out, start = [], 0
        for shape in shapes:
            size = int(np.prod(shape)) if shape else 1
            piece = mixed[start:start + size].reshape(shape)
            out.append(piece if host else jnp.asarray(piece))
            start += size
        return out[0] if one else out

    def reset(self) -> None:
        self.mixer.reset()
