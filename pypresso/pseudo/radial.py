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

import jax.numpy as jnp
import numpy as np

__all__ = ["simpson", "mesh_cutoff_index", "spherical_bessel", "simpson_weights"]

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
        # QE's even-mesh closure: ... + 2/3 f(n-3) + 15/12 f(n-2) + f(n-1) + 5/12 f(n)
        coefficients[mesh - 3] -= 0.25 / 3.0
        coefficients[mesh - 2] = 1.0 / 3.0
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
