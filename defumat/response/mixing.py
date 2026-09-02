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

__all__ = ["ResponseMixer", "DEFAULT_RESPONSE_MIXING"]

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

    def mix(self, current, proposed):
        """The next input, given this iteration's input and output.

        Args:
            current: one array, or a sequence of them, as the loop holds them.
            proposed: what the loop's evaluation produced from ``current``.

        Returns the same structure, as JAX arrays.
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
            out.append(jnp.asarray(mixed[start:start + size].reshape(shape)))
            start += size
        return out[0] if one else out

    def reset(self) -> None:
        self.mixer.reset()
