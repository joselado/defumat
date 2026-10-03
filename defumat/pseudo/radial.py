"""Radial-grid utilities: Simpson integration and spherical Bessel functions.

Pseudopotentials are tabulated on a logarithmic radial mesh, and every quantity
the plane-wave code needs is a radial integral of the form

    f(q) = int_0^inf dr r^2 f(r) j_l(qr)

Two details of QE's implementation are reproduced exactly because they change
the sixth decimal of a total energy:

* **Simpson's rule with QE's weights**, integrating against ``rab = dr/di``, and
  QE's particular handling of an even-length mesh.
* **The mesh is truncated at 10 bohr** and forced to an odd length
  (``msh`` in ``Modules/read_pseudo.f90``). Integrating the full tabulated mesh
  instead -- often out to 60 bohr, where the tabulated tail is numerical noise --
  gives a slightly different answer.

Everything here is written in JAX and stays differentiable in ``q``, which is
what lets ``vkb(k)`` be differentiated with respect to ``k`` later (rule D2).
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["simpson", "mesh_cutoff_index", "spherical_bessel",
           "spherical_bessel_derivative", "simpson_weights"]

#: QE truncates the radial mesh here before integrating (``rcut`` in read_pseudo).
RCUT = 10.0


def simpson_weights(rab) -> jnp.ndarray:
    """Simpson weights for QE's ``simpson``, already multiplied by ``rab``.

    Returning weights rather than performing the sum means an integral becomes a
    single dot product -- so a whole table of ``q`` values is one matrix
    multiplication, which is what the GPU wants.

    The weights of a *tabulated* mesh are host constants -- ``rab`` comes from
    the UPF file and nothing differentiates with respect to it -- so when the
    caller passes host data the product is done on the host and never dispatched
    to XLA. A traced ``rab`` still goes the JAX way, so the function remains
    usable inside a differentiated path.
    """
    mesh = np.shape(rab)[-1]

    coefficients = np.empty(mesh)
    coefficients[0] = 1.0 / 3.0
    # 4/3 at even positions (1-based even -> 0-based odd), 2/3 at odd ones.
    coefficients[1 : mesh - 1] = np.where(np.arange(2, mesh) % 2 == 0, 4.0, 2.0) / 3.0

    if mesh % 2 == 1:
        coefficients[mesh - 1] = 1.0 / 3.0
    else:
        # QE's even-mesh closure: ... + 2/3 f(n-3) + 15/12 f(n-2) + f(n-1) + 5/12 f(n).
        # ``simpsn.f90`` runs its loop to ``mesh-1``, so ``f(n-1)`` already carries
        # its 2/3 from the loop and the closure *adds* a whole term on top of it,
        # reaching 3/3 -- the closure's three lines are corrections to the running
        # sum rather than replacements, which is why two of them are ``+=``.
        coefficients[mesh - 3] -= 0.25 / 3.0
        coefficients[mesh - 2] += 1.0 / 3.0
        coefficients[mesh - 1] = 1.25 / 3.0

    if isinstance(rab, (np.ndarray, list, tuple)):
        return jnp.asarray(coefficients * np.asarray(rab))
    return jnp.asarray(coefficients) * rab


def simpson(func, rab) -> jnp.ndarray:
    """``int f(r) dr`` on a logarithmic mesh, by QE's Simpson rule."""
    return jnp.sum(jnp.asarray(func) * simpson_weights(rab), axis=-1)


def mesh_cutoff_index(r) -> int:
    """QE's ``msh``: the mesh truncated at 10 bohr, with an odd number of points.

    Points beyond ``RCUT`` carry no information -- the pseudopotential has long
    since reached its asymptotic form -- and including them makes the integral
    depend on how far the tabulation happens to extend.

    The rounding is QE's and has an off-by-one worth spelling out, because
    getting it wrong is invisible on some files and worth 1e-6 Ry on others.
    ``upflib``'s loop stops at the **first point beyond** ``RCUT`` and takes
    *that* index, not the last one inside::

        DO ir = 1, mesh
           IF ( r(ir) > rcut ) THEN
              msh = ir            ! one past the last point inside
              GOTO 5
           ENDIF
        ENDDO
        msh = mesh
      5 msh = 2*( (msh + 1)/2 ) - 1

    so the mesh QE integrates over includes one point past 10 bohr, and the
    odd-rounding is applied to that. Taking the last point inside instead gives
    a mesh **two points shorter** whenever the count is even -- which it is for
    most files. On a pseudopotential whose local potential has genuinely reached
    ``-2Z/r`` by 10 bohr (``Si.pz-vbc``) the two answers agree to 1e-11 and
    nothing shows; on the ``psl`` and ``rrkj`` sets they differ in the eighth
    decimal of ``V_loc(G=0)``, which is a constant shift of every eigenvalue and
    a ~1e-6 Ry error in the total energy that no cutoff makes smaller.
    """
    r = np.asarray(r)
    inside = int(np.searchsorted(r, RCUT, side="right"))
    first_beyond = inside + 1 if inside < len(r) else len(r)
    odd = 2 * ((first_beyond + 1) // 2) - 1
    return max(min(odd, len(r)), 3)


def spherical_bessel(l: int, x: jnp.ndarray) -> jnp.ndarray:
    """Spherical Bessel function ``j_l(x)`` for l = 0 .. 4.

    ``l = 4`` is needed by the augmentation charge of any dataset with ``d``
    projectors -- ``Q^L_ij`` runs to ``L = 2 lmax`` -- which is every transition
    metal, nickel included.

    The closed forms lose all their significant digits as ``x -> 0`` (``j_1``
    computes ``sin(x)/x^2 - cos(x)/x``, a difference of two large numbers), so
    below a threshold the leading Taylor series is used instead. The switch is a
    ``where`` on a *sanitised* argument: evaluating the closed form at x = 0 and
    discarding the result would still poison the gradient with NaN.
    """
    x = jnp.asarray(x)
    small = x < 0.05
    small4 = x < 1.0
    safe = jnp.where(small, 1.0, x)  # keeps the unused branch finite

    if l == 0:
        series = 1.0 - x**2 / 6.0 * (1.0 - x**2 / 20.0)
        exact = jnp.sin(safe) / safe
    elif l == 1:
        series = x / 3.0 * (1.0 - x**2 / 10.0 * (1.0 - x**2 / 28.0))
        exact = (jnp.sin(safe) / safe - jnp.cos(safe)) / safe
    elif l == 2:
        series = x**2 / 15.0 * (1.0 - x**2 / 14.0 * (1.0 - x**2 / 36.0))
        exact = ((3.0 / safe**2 - 1.0) * jnp.sin(safe) - 3.0 * jnp.cos(safe) / safe) / safe
    elif l == 3:
        series = x**3 / 105.0 * (1.0 - x**2 / 18.0 * (1.0 - x**2 / 44.0))
        exact = (
            (15.0 / safe**3 - 6.0 / safe) * jnp.sin(safe)
            - (15.0 / safe**2 - 1.0) * jnp.cos(safe)
        ) / safe
    elif l == 4:
        # Five correction terms and a threshold of 1, not the three-and-0.05 the
        # lower orders use: the closed form divides by ``x^5`` and its numerator
        # cancels to ``x^4/945``, so at x = 0.05 it has lost every significant
        # digit. The series in turn needs more terms to stay accurate out to
        # where the closed form becomes safe. Both branches are good to ~1e-12
        # relative at the crossover, which is where the two meet.
        series = x**4 / 945.0 * (
            1.0 - x**2 / 22.0 * (
                1.0 - x**2 / 52.0 * (
                    1.0 - x**2 / 90.0 * (
                        1.0 - x**2 / 136.0 * (1.0 - x**2 / 190.0)
                    )
                )
            )
        )
        big = jnp.where(small4, 1.0, x)
        exact = (
            jnp.sin(big) * (105.0 - 45.0 * big**2 + big**4)
            + jnp.cos(big) * (10.0 * big**3 - 105.0 * big)
        ) / big**5
        return jnp.where(small4, series, exact)
    else:
        raise NotImplementedError(f"spherical_bessel is implemented for l <= 4, got l = {l}")

    return jnp.where(small, series, exact)


#: Below this argument :func:`spherical_bessel_derivative` takes the Taylor
#: series, above it the closed form. At ``x = 2`` the closed form's cancellation
#: costs a factor of about 200 over the rounding for ``l = 4``, the worst order
#: (its terms go as ``105 / x^5`` against a value of ``x^4 / 945``); a switch at
#: 1 cost 1e5 and left 4.5e-12 relative in ``j_4'``.
DERIVATIVE_SERIES_BELOW = 2.0

#: Terms of that series. The sixteenth is below ``1e-35`` at ``x = 2`` for every
#: ``l``, so the sum and its first four derivatives are exact to the rounding
#: there.
DERIVATIVE_SERIES_TERMS = 16


def _bessel_series(l: int, x: jnp.ndarray) -> jnp.ndarray:
    """``j_l(x) = x^l sum_k (-x^2/2)^k / (k! (2l+2k+1)!!)``, to :data:`DERIVATIVE_SERIES_TERMS` terms."""
    coefficients = []
    for k in range(DERIVATIVE_SERIES_TERMS):
        double_factorial = float(np.prod(np.arange(2 * l + 2 * k + 1, 0, -2)))
        coefficients.append((-0.5) ** k / (math.factorial(k) * double_factorial))
    x2 = x * x
    total = coefficients[-1]
    for c in reversed(coefficients[:-1]):
        total = total * x2 + c
    return x**l * total


def _nth_derivative(f, n: int):
    for _ in range(n):
        f = (lambda g: lambda y: jax.jvp(g, (y,), (jnp.ones_like(y),))[1])(f)
    return f


def spherical_bessel_derivative(l: int, n: int, x: jnp.ndarray) -> jnp.ndarray:
    """``d^n j_l(x) / dx^n`` for ``n >= 1``, accurate where :func:`spherical_bessel`'s derivative is not.

    What a radial transform's derivative in ``q`` is made of
    (:func:`~defumat.pseudo.formfactors.bessel_transform`). **Not** the
    derivative of :func:`spherical_bessel` as written, which is QE's ``sph_bes``
    and switches from a series cut at ``x^4`` to the closed form at
    ``x = 0.05``: below the switch the cut series has no ``x^6`` term to
    differentiate, and above it the closed form of ``l = 2`` and 3 has lost
    digits to the cancellation of its ``x^-(l+1)`` terms, which differentiating
    amplifies. Measured against ``mpmath`` at 50 digits for ``x`` from 1e-4 to
    25, the first to fourth derivatives JAX takes of :func:`spherical_bessel`
    are off by up to 3.5e-10 absolute in ``j_0'`` and 3.5e-8 in ``j_0''`` (both
    just below 0.05), 1.2e-8 in ``j_3''`` and 4.8e-6 in ``j_3'''``, where these
    are off by at most **5.5e-14** for every ``l`` and ``n`` up to 4.

    The values of the transforms themselves stay :func:`spherical_bessel`'s, so
    they remain QE's; only their derivatives come from here. Each derivative is
    taken by nested forward-mode differentiation of the two forms, which is
    exact arithmetic on a polynomial below the switch and on the closed form
    above it, and the switch is a ``where`` on a sanitised argument as in
    :func:`spherical_bessel`.
    """
    if n < 1:
        raise ValueError(f"spherical_bessel_derivative wants n >= 1, got {n}")
    x = jnp.asarray(x)
    small = x < DERIVATIVE_SERIES_BELOW
    series = _nth_derivative(partial(_bessel_series, l), n)(
        jnp.where(small, x, 0.0))
    exact = _nth_derivative(partial(spherical_bessel, l), n)(
        jnp.where(small, DERIVATIVE_SERIES_BELOW, x))
    return jnp.where(small, series, exact)
