"""Slater integrals computed from the radial functions, screened or not.

Everywhere else in :mod:`defumat.hubbard` the interaction is *given*: a ``U``
and a ``J`` are read off the input card and the Slater integrals follow from
them by fixed ratios. This module computes them instead,

    F^k = int dr int dr'  [r phi(r)]^2 [r' phi(r')]^2  g_k(r_<, r_>)

from the manifold's own radial function, with two kernels:

    g_k(r, r')      = r_<^k / r_>^(k+1)                      (bare)
    g_k(r, r'; lam) = (2k+1) lam i_k(lam r_<) ktilde_k(lam r_>)   (Yukawa)

``i_k`` is the modified spherical Bessel function of the first kind and
``ktilde_k = (2/pi) k_k`` the second, normalised so that **the second kernel
becomes the first as ``lam -> 0``** -- ``i_k(x) -> x^k/(2k+1)!!`` and
``ktilde_k(x) -> (2k-1)!!/x^(k+1)``, and ``(2k+1)/(2k+1)!! * (2k-1)!! = 1``.
That limit is the module's own consistency check rather than a remark: it is
what :func:`slater_from_radial` is tested against at small ``lam``, and it is
why the normalisation is written this way instead of being copied.

Elk's ``fyukawa.f90`` and ``fyukawa0.f90`` (``inpdftu = 4``), with
``findlambda.f90`` (``inpdftu = 5``) solving ``F^0(lam) = U`` for the screening
length so that a *chosen* ``U`` fixes the whole interaction. ``pw.x`` has
neither: no Slater integral is ever computed there, only parameterised.

**The screening length is what makes this a model rather than a first-principles
``U``**, and Elk says so by making it an input. What it buys over reading ``U``
and ``J`` off a card is that the *ratios* ``F^2 : F^4 : F^6`` are then the
orbital's own rather than an atomic table's, and that one number fixes the whole
matrix.

**The radial function is the pseudised ``chi`` unless the dataset is PAW.** Elk
integrates over an all-electron muffin-tin function; a norm-conserving UPF has
only the pseudo-orbital, which is smooth inside ``r_c`` precisely where the
Coulomb integral is largest. So a ``U`` computed from one is a *pseudo-orbital*
``U`` and is not Elk's number -- the same pseudisation P32 measured on the mBJ
average and P61 on an allowed structure factor. A PAW dataset carries
``PP_AEWFC``, the all-electron partial wave, and :func:`manifold_radial` uses it
when it is there, which is the P32 pattern. Two further things keep even that
from being like-for-like and are reported rather than assumed: a partial wave is
per *projector* and need not be normalised on its own
(:func:`manifold_radial` returns the norm it measured), and Elk integrates only
out to the muffin-tin radius where a UPF's mesh runs much further, so
``cutoff`` says where the integral was stopped.

Units are Rydberg throughout: the kernels above are the Hartree-atomic-unit
Coulomb interaction, so every integral is multiplied by ``e2 = 2``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import spherical_in, spherical_kn

from defumat.pseudo.radial import simpson_weights
from defumat.units import E2

__all__ = [
    "RadialManifold",
    "manifold_radial",
    "screening_length",
    "slater_from_poisson",
    "slater_from_radial",
    "slater_set",
]

#: ``findlambda``'s bracket. Below ``lambdamin`` Elk switches to the unscreened
#: integral; above ``lambdamax`` the screened ``F^0`` is ~1e-3 and ``U`` is
#: effectively zero, so there is nothing to solve for.
LAMBDA_MIN = 1.0e-2
LAMBDA_MAX = 50.0


class RadialManifold:
    """One Hubbard manifold's radial function, with what it came from.

    ``chi`` is ``r phi(r)`` on ``r`` with integration weights ``weights``, on
    the dataset's **whole** mesh and **zero beyond** ``cutoff`` -- truncating
    the arrays instead would leave the radial Poisson solve of
    :func:`slater_from_poisson` with a source that does not vanish at its outer
    boundary, where it imposes the ``r^-(l+1)`` decay. The three attributes
    beside them are what makes the resulting ``U`` reportable rather than merely
    computed: which function was used, where it was cut, and what the function's
    own norm is inside that radius.
    """

    __slots__ = ("chi", "r", "weights", "kind", "cutoff", "norm")

    def __init__(self, chi, r, weights, kind, cutoff, norm):
        self.chi = chi
        self.r = r
        self.weights = weights
        self.kind = kind
        self.cutoff = cutoff
        self.norm = norm

    def __repr__(self) -> str:
        return (
            f"RadialManifold(kind={self.kind!r}, cutoff={self.cutoff:.3f} bohr, "
            f"norm={self.norm:.6f})"
        )


def manifold_radial(pseudo, n: int, l: int, cutoff: float | None = None) -> RadialManifold:
    """The radial function of the named manifold, all-electron where there is one.

    ``PP_AEWFC`` is preferred and the pseudo ``chi`` is the fallback. The
    all-electron partial waves are indexed by *projector*, so the one taken is
    the projector of the right ``l`` whose norm inside ``cutoff`` is largest --
    a manifold with two projectors of the same ``l`` has one that carries the
    bound state and one that does not, and picking by ``l`` alone can take
    either.

    **``cutoff`` is not optional for the all-electron route and the default is
    the dataset's own augmentation radius.** A PAW partial wave is a solution of
    the atomic problem *inside* the sphere and is continued outside it by
    whatever the generator left there, which diverges: integrated over the whole
    mesh, nickel's ``3d`` has a norm of 1.3e17 and an ``F^0`` of 1e34 Ry. Cut at
    ``PP_PAW``'s own ``cutoff_index`` -- 1.81 bohr here -- the same function has
    a norm of 0.972 and an ``F^0`` of 25.3 eV, which is the textbook bare Slater
    integral of a 3d transition metal. So the radius is read from the dataset,
    ``cutoff`` overrides it, and the norm that comes back says whether the
    manifold is actually bound inside it.
    """
    from defumat.hubbard.manifold import manifold_label, normalize_label

    r = np.asarray(pseudo.r, dtype=float)
    weights = np.asarray(simpson_weights(pseudo.rab), dtype=float)
    paw = getattr(pseudo, "paw", None)
    if cutoff is None and paw is not None:
        cutoff = float(r[paw.cutoff_index])
    keep = slice(None) if cutoff is None else slice(0, int(np.searchsorted(r, cutoff)) + 1)

    label = manifold_label(n, l)
    chi = None
    kind = "pseudo"
    ae = getattr(paw, "ae_wfc", None) if paw is not None else None
    if ae is not None and len(np.asarray(ae)):
        ae = np.asarray(ae, dtype=float)
        best, best_norm = None, -1.0
        for index, projector in enumerate(pseudo.projectors):
            if projector.l != l:
                continue
            norm = float(np.sum(ae[index][keep] ** 2 * weights[keep]))
            if norm > best_norm:
                best, best_norm = ae[index], norm
        if best is not None:
            chi, kind = best, "all-electron"
    if chi is None:
        for orbital in pseudo.orbitals:
            if normalize_label(orbital.label) == label:
                chi = np.asarray(orbital.chi, dtype=float)
                break
    if chi is None:
        raise ValueError(
            f"{pseudo.element}: no {label} orbital and no all-electron partial "
            f"wave with l = {l} to compute Slater integrals from"
        )
    truncated = np.zeros_like(r)
    truncated[keep] = chi[keep]
    norm = float(np.sum(truncated**2 * weights))
    return RadialManifold(
        chi=truncated, r=r, weights=weights, kind=kind,
        cutoff=float(r[keep][-1]), norm=norm,
    )


def _scaled_kernel(k: int, x: np.ndarray):
    """``(i_s, k_s)`` with ``i_k(x) = i_s e^x`` and ``ktilde_k(x) = k_s e^-x``.

    The exponentially scaled modified spherical Bessel pair. Computing ``i_k``
    and ``ktilde_k`` themselves and multiplying is arithmetically identical and
    numerically hopeless: at ``lambda = 50`` on a mesh reaching 59 bohr,
    ``i_k`` overflows to infinity exactly where ``ktilde_k`` underflows to zero
    and the product -- which is bounded by ``e^-lambda|r-r'|`` and therefore
    tiny -- comes out ``NaN``. ``findlambda`` searches up to that ``lambda``, so
    this is the difference between a root find and a crash.
    """
    from scipy.special import ive, kve

    root = np.sqrt(np.pi / (2.0 * x))
    return root * ive(k + 0.5, x), (2.0 / np.pi) * root * kve(k + 0.5, x)


def _accumulate(values: np.ndarray, decay: np.ndarray, reverse: bool) -> np.ndarray:
    """A cumulative sum with an exponential factored in.

    Forward: ``out[i] = values[i] + decay[i] out[i-1]``, with
    ``decay[i] = exp(-(x_i - x_{i-1}))``.
    Reverse: ``out[i] = values[i] + decay[i] out[i+1]``, with
    ``decay[i] = exp(-(x_{i+1} - x_i))``.

    **Both index ``decay`` at ``i``, and the two ``decay`` arrays are
    different** -- the gap behind a point and the gap ahead of it. Using one
    array for both, or shifting the index, evaluates the kernel at a screening
    length off by one mesh step: on a logarithmic grid consecutive gaps differ
    by ``e^dx`` (about one per cent here), and the resulting ``F^k`` is
    0.1 to 0.3 per cent wrong, growing with ``lambda``. Nothing but a direct
    ``O(N^2)`` double sum at finite ``lambda`` sees it -- the ``lam -> 0`` check
    has ``decay = 1``, the round trip is self-consistent, and the Poisson route
    only runs at ``lam = 0``.

    This is what it is: a recursion. The mesh is a thousand points and it runs
    once per manifold.
    """
    out = np.array(values, dtype=float)
    if reverse:
        for i in range(out.size - 2, -1, -1):
            out[i] += out[i + 1] * decay[i]
    else:
        for i in range(1, out.size):
            out[i] += out[i - 1] * decay[i]
    return out


def slater_from_radial(manifold: RadialManifold, k: int, lam: float = 0.0) -> float:
    """``F^k`` in **Ry**, by the double radial integral.

    ``lam = 0`` is the bare Coulomb kernel; a positive ``lam`` is the inverse
    Yukawa screening length, in inverse bohr.

    The inner integrals are cumulative sums of the same Simpson weights the
    outer one uses, so the two ends of the ``lam -> 0`` check share a
    discretisation and what it compares is the kernel rather than the
    quadrature. :func:`slater_from_poisson` is the *independent* discretisation.
    """
    r = manifold.r
    density = manifold.chi**2
    integrand = density * manifold.weights

    if lam <= 0.0:
        with np.errstate(divide="ignore", over="ignore"):
            a, b = r**k, 1.0 / r ** (k + 1)
        below = np.cumsum(integrand * a)
        above = np.cumsum((integrand * b)[::-1])[::-1]
    else:
        x = lam * r
        a, b = _scaled_kernel(k, x)
        a = (2 * k + 1) * lam * a
        # The gap *behind* each point and the gap *ahead* of it: two different
        # arrays on a logarithmic mesh, and :func:`_accumulate` indexes both at
        # ``i``.
        behind = np.empty_like(r)
        behind[0] = 0.0
        behind[1:] = np.exp(-np.diff(x))
        ahead = np.empty_like(r)
        ahead[-1] = 0.0
        ahead[:-1] = np.exp(-np.diff(x))
        below = _accumulate(integrand * a, behind, reverse=False)
        above = _accumulate(integrand * b, ahead, reverse=True)

    # ``below`` and ``above`` both include the point ``r`` itself, so the
    # diagonal is counted twice and one copy comes back off. It is second order
    # in the mesh weight and looks negligible; measured against the Poisson
    # route it is worth **1 to 4 per cent** and is systematic in the same
    # direction for every ``k`` and every dataset, which is what identified it.
    inner = b * below + a * above - a * b * integrand
    return float(E2 * np.sum(integrand * inner))


def slater_from_poisson(manifold: RadialManifold, k: int) -> float:
    """``F^k`` in Ry through the radial Poisson solve, for the bare kernel only.

    A second discretisation of the same number: :mod:`defumat.paw.hartree`
    integrates the Poisson equation with the Numerov scheme QE uses, where
    :func:`slater_from_radial` accumulates a quadrature. They agree to the
    accuracy of the mesh, which is the check that neither is wrong in a way the
    other shares.
    """
    from defumat.paw.hartree import radial_hartree

    r = manifold.r
    density = manifold.chi**2
    # ``radial_hartree`` is linear in its source and returns the potential "in
    # whatever units f carried", so the ``4 pi/(2l+1)`` QE folds into its own
    # call is not wanted here: the Slater kernel ``r_<^k/r_>^(k+1)`` carries no
    # solid-angle factor. What is wanted is the Rydberg ``e2``, and nothing
    # else -- calibrated against :func:`slater_from_radial` on a *smooth*
    # density, where the two discretisations agree to 1e-5 relative.
    prefactor = E2
    dx = float(np.log(r[-1] / r[0]) / (r.size - 1))
    potential = np.asarray(
        radial_hartree(
            prefactor * density, r, r**2, np.sqrt(r), dx, k, 2 * k + 2
        )
    )
    return float(np.sum(density * potential * manifold.weights))


def screening_length(manifold: RadialManifold, u: float) -> float:
    """``findlambda``: the ``lam`` whose screened ``F^0`` is the requested ``U``.

    ``u`` in Ry. ``F^0`` falls monotonically from its bare value to zero as
    ``lam`` grows, so this is a bracketed root find rather than Elk's
    half-interval-then-secant pair -- the same root, found the way a
    one-dimensional monotone root is found. A ``U`` above the bare ``F^0``
    cannot be screened *into* existence and is an error rather than a clamp.
    """
    bare = slater_from_radial(manifold, 0, 0.0)
    if u <= 0.0:
        raise ValueError(f"a screening length needs a positive U, not {u}")
    if u >= bare:
        raise ValueError(
            f"U = {u:.4f} Ry is at or above this manifold's unscreened F^0 = "
            f"{bare:.4f} Ry ({manifold.kind} radial function, cut at "
            f"{manifold.cutoff:.2f} bohr); screening only lowers it"
        )
    if slater_from_radial(manifold, 0, LAMBDA_MAX) > u:
        raise ValueError(
            f"U = {u:.4f} Ry is below the screened F^0 at lambda = {LAMBDA_MAX}; "
            "there is no screening length this weak"
        )
    return float(
        brentq(
            lambda lam: slater_from_radial(manifold, 0, lam) - u,
            LAMBDA_MIN, LAMBDA_MAX, xtol=1e-10, rtol=1e-12,
        )
    )


def slater_set(manifold: RadialManifold, l: int, lam: float = 0.0) -> np.ndarray:
    """``F[0:7]`` in Ry for one manifold: the even ``k`` up to ``2l``, rest zero."""
    f = np.zeros(7)
    for k in range(0, 2 * l + 1, 2):
        f[k] = slater_from_radial(manifold, k, lam)
    return f
