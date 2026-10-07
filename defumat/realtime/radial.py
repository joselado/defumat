"""The projectors at ``k + kappa(t)`` from a table of ``g_l(q^2)``, for the time loop.

A real-time step rebuilds ``vkb`` at ``k + kappa(t + dt/2)``, and the current
rebuilds it again at ``k + kappa(t + dt)``, so a run of 30000 steps evaluates
the projectors 60000 times. The ground-state code evaluates each radial form
factor by a direct Simpson transform per ``|k+G|``
(:func:`~defumat.pseudo.formfactors.projector_form_factors`), which is right for
a calculation that builds its projectors once and is differentiated rarely, and
it is what a step pays for: measured on two-atom silicon at ``ecutwfc = 12`` on
one core, the rebuild is 1.9 ms against 0.39 ms for one Hamiltonian application
on the four occupied bands (``tools/realtime/step_cost.py``).

The table is the function the transform computes, ``f_l(q) = q^l g_l(q^2)``,
with ``g_l`` an entire function of ``s = q^2``, expanded once per dataset in
Chebyshev polynomials of ``s`` on ``[0, s_max]``. A column is then

    Y_lm(qhat) f_l(|q|) = S_lm(q) g_l(|q|^2),

the regular solid harmonic (a polynomial in the components of ``q``,
:func:`~defumat.pseudo.harmonics.real_solid_harmonics`) times a polynomial in
``|q|^2``, which has no square root, no direction and no guard anywhere, so it
is differentiable to every order at ``k + G + kappa = 0`` as everywhere else
(the row ``HARMONICS-NEXT.md`` has a section on). The coefficients are read off
the direct transform at Chebyshev nodes, and the number of them is chosen until
the series has converged to round-off; the table is checked against the
transform it replaces at points the nodes do not contain
(``tests/unit/test_realtime_radial.py``).

**It is the real-time route's projector and nothing else's.** The ground state
keeps the transform, whose numbers every comparison with ``pw.x`` was taken
against (``PLAN.md`` D1 and D2 record the same trade the other way round: where
QE interpolates a table, the code integrates, for differentiability). Here the
table *is* differentiable, and the price is a Hamiltonian at ``kappa = 0`` that
agrees with the ground state's to the table's error rather than bit for bit,
which is what :func:`max_error` reports.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from defumat.pseudo.formfactors import projector_form_factors
from defumat.pseudo.harmonics import real_solid_harmonics
from defumat.pseudo.projectors import _projector_dataset_key, projector_channels

__all__ = ["RadialTable", "radial_table", "column_layout", "max_error"]


def column_layout(pseudos):
    """``(datasets, beta_of, lm_of, l_of)`` exactly as ``build_projector_core`` orders them.

    One column per channel of each distinct dataset, in the order the
    datasets are first declared, ``beta_of`` indexing the concatenation of the
    datasets' radial channels. A table built on this layout gives columns that
    sit where the projector core's do (``ProjectorCore.column_of_channel``
    selects from them unchanged).
    """
    datasets, seen = [], set()
    for pseudo in pseudos:
        key = _projector_dataset_key(pseudo)
        if key not in seen:
            seen.add(key)
            datasets.append(pseudo)
    beta_of, lm_of, l_of = [], [], []
    offset = 0
    for pseudo in datasets:
        for nb, l, lm in projector_channels(pseudo):
            beta_of.append(offset + nb)
            lm_of.append(lm)
            l_of.append(l)
        offset += len(pseudo.projectors)
    return tuple(datasets), beta_of, lm_of, l_of


class RadialTable(eqx.Module):
    """``g_l(s)`` for every radial channel as a Chebyshev series on ``[0, s_max]``.

    With the ``4 pi / sqrt(Omega)`` of the transform folded in, so
    :meth:`columns` returns what ``ProjectorCore.columns`` holds.
    """

    #: ``(nbeta, nterms)``, the Chebyshev coefficients in ``x = 2 s / s_max - 1``.
    coefficients: jnp.ndarray
    #: ``(ncols,)``: which radial channel and which harmonic each column takes.
    beta_of: jnp.ndarray
    lm_of: jnp.ndarray
    #: The end of the range, bohr^-2, a scalar array rather than a static field:
    #: the range follows the largest shift a pulse reaches, and a static one put
    #: it in the program, so a second run of another amplitude recompiled the
    #: whole step (four programs on two-atom silicon, found by counting).
    s_max: jnp.ndarray
    lmax: int = eqx.field(static=True)

    def radial(self, s):
        """``g(s)`` for every radial channel, ``(..., nbeta)``, by Clenshaw's recurrence.

        Outside ``[0, s_max]`` a Chebyshev series grows without bound, so the
        argument is held at the edge there, the value frozen and its derivative
        zero. The driver builds its table to ``(max |k+G| + kappa_max + 0.5)^2``,
        so no live plane wave reaches the edge in a run; a caller that builds a
        smaller table is clamped without a word.
        """
        s = jnp.where(s <= self.s_max, s, self.s_max)
        x = (2.0 * s / self.s_max - 1.0)[..., None]
        c = self.coefficients
        b1 = jnp.zeros(x.shape[:-1] + (c.shape[0],), dtype=c.dtype)
        b2 = b1
        for k in range(c.shape[1] - 1, 0, -1):
            b1, b2 = 2.0 * x * b1 - b2 + c[:, k], b1
        return x * b1 - b2 + c[:, 0]

    def columns(self, kg):
        """``S_lm(q) g_l(|q|^2)`` for every column, ``(..., ncols)``, real."""
        s = jnp.sum(kg * kg, axis=-1)
        solid = jnp.take(real_solid_harmonics(kg, self.lmax), self.lm_of, axis=-1)
        return solid * jnp.take(self.radial(s), self.beta_of, axis=-1)


def _chebyshev(values):
    """Coefficients of the interpolant through first-kind Chebyshev nodes, by a DCT."""
    n = values.shape[-1]
    j = np.arange(n)
    k = np.arange(n)[:, None]
    coefficients = (2.0 / n) * values @ np.cos(np.pi * k * (j[None, :] + 0.5) / n).T
    coefficients[..., 0] *= 0.5
    return coefficients


def _radial_values(datasets, s, volume):
    """``g_l(s) = f_l(sqrt s)/sqrt(s)^l`` from the direct transform, ``(nbeta, ns)``."""
    q = np.sqrt(s)
    rows = []
    for pseudo in datasets:
        table = np.asarray(projector_form_factors(pseudo, jnp.asarray(q), volume))
        for nb, projector in enumerate(pseudo.projectors):
            rows.append(table[nb] / q**projector.l)
    return np.asarray(rows)


def radial_table(pseudos, volume: float, s_max: float, *, tolerance: float = 1e-13,
                 start: int = 32, limit: int = 512) -> RadialTable:
    """The table for these datasets on ``[0, s_max]``, ``s_max`` in bohr^-2.

    The number of terms doubles from ``start`` until the last eighth of the
    coefficients is below ``tolerance`` times the largest, which for an entire
    function is the series having converged, or until a tail below 1e-11 has
    stopped falling, which is the transform's round-off floor; ``limit``
    refuses a dataset whose transform will not converge on the range, by name
    rather than with a table that is quietly wrong. The coefficients fall to the transform's own
    round-off and stop there, about 1e-14 of the largest by the twentieth term
    on ``Si.pz-vbc`` at 12 Ry, so ``tolerance`` sits above that floor, and every
    term is kept: the ones below the floor cost a multiply each.
    """
    datasets, beta_of, lm_of, l_of = column_layout(pseudos)
    lmax = max([0] + [p.lmax for p in datasets])
    if not beta_of:
        return RadialTable(coefficients=jnp.zeros((0, 1)), beta_of=jnp.zeros((0,), int),
                           lm_of=jnp.zeros((0,), int), s_max=jnp.asarray(float(s_max)),
                           lmax=lmax)
    n = start
    previous = None
    while True:
        nodes = np.cos(np.pi * (np.arange(n) + 0.5) / n)
        s = 0.5 * float(s_max) * (nodes + 1.0)
        coefficients = _chebyshev(_radial_values(datasets, s, volume))
        scale = np.abs(coefficients).max(axis=-1, keepdims=True)
        tail = np.abs(coefficients[:, -max(1, n // 8):]).max(axis=-1, keepdims=True)
        relative = float(np.max(tail / np.maximum(scale, 1e-300)))
        if relative <= tolerance:
            break
        # **A floor is convergence too.** The coefficients fall to the
        # transform's own round-off and stop, and that floor is the dataset's:
        # 1e-14 on ``Si.pz-vbc``, 4e-13 of the largest on the psl ultrasoft Al
        # and As at 12.9 bohr^-2, where doubling the terms from 256 to 512 left
        # the tail where it was. A tail that has stopped falling and is below
        # 1e-11 is that floor, and more terms buy nothing.
        if previous is not None and relative < 1e-11 and relative > 0.25 * previous:
            break
        previous = relative
        if n >= limit:
            raise ValueError(
                f"the radial table of g_l(q^2) did not converge in {limit} Chebyshev "
                f"terms on [0, {s_max:.4g}] bohr^-2 (tail {float(np.max(tail / scale)):.2e} "
                f"of the largest coefficient)")
        n *= 2
    return RadialTable(coefficients=jnp.asarray(coefficients),
                       beta_of=jnp.asarray(beta_of), lm_of=jnp.asarray(lm_of),
                       s_max=jnp.asarray(float(s_max)), lmax=lmax)


def max_error(table: RadialTable, pseudos, volume: float, samples: int = 257) -> float:
    """``max |table - transform| / max |transform|`` on a grid the nodes do not contain."""
    datasets, _, _, _ = column_layout(pseudos)
    s = np.linspace(1e-6, table.s_max, samples)
    exact = _radial_values(datasets, s, volume)
    tabulated = np.asarray(jax.vmap(table.radial)(jnp.asarray(s))).T
    return float(np.max(np.abs(tabulated - exact)) / np.max(np.abs(exact)))
