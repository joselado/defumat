"""Kerker on the ultracell box, which is where the long wavelengths are.

``PLAN.md`` P88. Charge sloshing is a long-wavelength instability, and an
ultracell is a cell whose whole purpose is to have long wavelengths in it: the
smallest non-zero ``|G+Q|`` it carries is ``N`` times smaller than the unit
cell's, so the Hartree kernel's ``1/|G+Q|^2`` is ``N^2`` times larger there.
Plain linear mixing at any useful ``beta`` diverges on this, which is why Elk's
own Cr example runs at ``beta0 = 0.001`` for up to two thousand iterations.

Kerker is the standard answer and it transfers to the box unchanged, because
the box's reciprocal grid *is* the ``G + Q`` set:

    beta |G+Q|^2 / (|G+Q|^2 + q_TF^2),

which is ``beta`` at short wavelength and falls to zero at long, so exactly the
components an ultracell adds are the ones it damps. ``q_TF`` is the unit cell's
own Thomas-Fermi wavevector -- a property of the electron *density*, and the
ultracell has ``N`` times the electrons in ``N`` times the volume, so it is the
same number.

**The ``G + Q = 0`` component is annihilated**, and here that matters more than
it does in a unit cell: it is what keeps the ultracell's electron count fixed
while every other Fourier component of the envelope moves.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from defumat.scf.mixing import thomas_fermi_screening
from defumat.system.cell import Cell
from defumat.ultracell.grid import Ultracell

__all__ = ["box_kerker"]


def box_kerker(
    ultracell: Ultracell,
    cell: Cell,
    nelec: float,
    shape,
    beta: float = 0.7,
    screening: float | None = None,
):
    """A :attr:`~defumat.scf.mixing.Mixer.precondition` for the ultracell box.

    ``shape`` is the density's own ``(nspin, *box)``; the returned callable
    takes a flat residual and the input density, as the mixer protocol asks,
    and ignores the second -- Kerker's screening length is a property of the
    cell rather than of ``rho(r)``.
    """
    if screening is None:
        screening = thomas_fermi_screening(float(cell.volume), float(nelec))
    grid = ultracell.grid
    g2 = jnp.asarray(ultracell.g2(cell), dtype=ultracell.precision.real)
    factor = beta * g2 / (g2 + screening)
    nspin = int(shape[0])

    @jax.jit
    def apply(vector):
        head = vector.reshape(shape)

        def screened(channel):
            box = jnp.fft.fftn(channel.reshape(grid))
            return jnp.real(jnp.fft.ifftn(box * factor)).reshape(-1)

        if nspin == 1:
            out = [screened(head[0])]
        elif nspin == 2:
            # Only the *charge* is screened, exactly as ``approx_screening``
            # does it: the Thomas-Fermi divergence belongs to the charge
            # response and the magnetization has none, so screening both
            # channels would damp the direction a magnetic run has to move in.
            charge, moment = head[0] + head[1], head[0] - head[1]
            charge, moment = screened(charge), beta * moment.reshape(-1)
            out = [0.5 * (charge + moment), 0.5 * (charge - moment)]
        else:
            out = [screened(head[0])] + [
                beta * head[c].reshape(-1) for c in range(1, nspin)
            ]
        return jnp.concatenate(out)

    def preconditioner(residual, density=None):
        return np.asarray(apply(jnp.asarray(np.asarray(residual).ravel())))

    return preconditioner
