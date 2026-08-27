"""The exchange-correlation kernels of the Dyson equation, behind a name.

``PLAN.md`` P37. Every kernel here returns the same object -- the **symmetrised**
kernel

    F = v^{-1/2} f_xc v^{-1/2},

dimensionless, ``(nw, nm, nm)``, on :class:`~pypresso.tddft.chi0.ChiZero`'s
index set (three head directions then the body). Symmetrised because
:mod:`pypresso.tddft.chi0` stores ``v^{1/2} chi_0 v^{1/2}`` and the two have to
meet: ``genvfxc.f90`` builds exactly this array and calls it
``v^-1/2 f_xc v^-1/2``.

``rpa``
    ``f_xc = 0``. Not a kernel so much as the absence of one, and the reference
    every other entry here is read against -- a spectrum with an excitonic peak
    means nothing until the same code with this kernel is shown not to have one.
``alda``
    the adiabatic local density approximation: ``f_xc(r, r') = dv_xc/drho
    delta(r - r')``, which in reciprocal space is a matrix depending on
    ``G - G'`` alone. **Its head and wings are zero** -- ``f_xc`` is finite at
    ``q = 0`` where ``v`` diverges, so ``F = f_xc / v`` vanishes there, which is
    Elk's ``vfxc(1:3,...) = 0`` and is the reason ALDA cannot bind an exciton
    however large it is. It is one ``jvp`` of the exchange-correlation potential
    this code already writes down (rule D1), not a transcription of
    ``setup_dmuxc``.
``lrc``
    the long-range correction of Reining, Olevano, Rubio and Onida
    (PRL **88**, 066404 (2002)) and Botti *et al.* (PRB **69**, 155112 (2004)):
    ``f_xc = -alpha / q^2``, so ``F = -alpha / 4 pi`` on the diagonal and
    nothing else. Two lines, one empirical parameter, and the kernel the
    bootstrap exists to produce without one.
``bootstrap``
    Sharma, Dewhurst, Sanna and Gross (PRL **107**, 186401 (2011)); Elk's
    ``fxctype = 210``:

        f_xc^BS = - eps^-1(q, 0) v(q) / (eps_0(q, 0) - 1),

    which symmetrised is ``F = eps^-1(omega = 0) / X_00(omega = 0)``, with
    ``eps^-1`` taken from the Dyson equation this kernel is then fed back into.
    It is **self-consistent** and **static**: one matrix, reused at every
    frequency.
``bootstrap-1``
    Elk's ``fxctype = 211``: the same expression evaluated once, from the RPA
    ``eps^-1``, without iterating. Byun and Ullrich call this family's members
    the "0-bootstrap" and "RPA-bootstrap" kernels and find they differ from the
    self-consistent one by about 10% in ``alpha``.

**The head convention.** In the optical limit ``eps_0``'s head is a 3x3 block,
and the scalar the bootstrap divides by is its **trace over three** --
``genvfxc.f90``'s ``z1 = (eps0(1,1) + eps0(2,2) + eps0(3,3))/3``. On a cubic
crystal that is the same as any diagonal entry; on a uniaxial one it is not, and
it is what makes the kernel a scalar rather than a tensor.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from pypresso.units import FPI

__all__ = [
    "get_kernel",
    "kernel_names",
    "register_kernel",
    "XCKernel",
    "DEFAULT_KERNEL",
]

DEFAULT_KERNEL = "bootstrap"

_KERNELS: dict = {}


class XCKernel:
    """A kernel of the Dyson equation: how to build ``F``, and whether to iterate.

    ``build(chi, epsi, context)`` returns ``(nw, nm, nm)``. ``epsi`` is the
    inverse dielectric matrix from the previous pass, or ``None`` on the first
    one, and only a self-consistent kernel reads it.
    """

    def __init__(self, name, build, *, self_consistent=False, iterations=None,
                 description=""):
        self.name = name
        self.build = build
        #: Whether the kernel has to be rebuilt from its own answer until it
        #: stops moving. ``genvfxc``'s 210 against 211.
        self.self_consistent = self_consistent
        #: A fixed number of passes for a kernel that is neither -- 211 is one.
        self.iterations = iterations
        self.description = description

    def __repr__(self) -> str:  # pragma: no cover -- diagnostics
        return f"XCKernel({self.name!r}, self_consistent={self.self_consistent})"


def register_kernel(kernel: XCKernel) -> None:
    _KERNELS[kernel.name.lower()] = kernel


def get_kernel(name: str | None) -> XCKernel:
    key = (name or DEFAULT_KERNEL).lower()
    if key not in _KERNELS:
        raise ValueError(
            f"unknown exchange-correlation kernel {name!r}; available: "
            f"{sorted(_KERNELS)}"
        )
    return _KERNELS[key]


def kernel_names() -> tuple[str, ...]:
    return tuple(sorted(_KERNELS))


# --- rpa ---------------------------------------------------------------------

def _rpa(chi, epsi, context):
    return jnp.zeros_like(chi.x)


register_kernel(XCKernel(
    "rpa", _rpa,
    description="f_xc = 0: the random phase approximation",
))


# --- lrc ---------------------------------------------------------------------

def _lrc(chi, epsi, context):
    """``F = -alpha / 4 pi`` on the diagonal, head included.

    ``genvfxc.f90``'s ``fxctype = 200`` without its ``beta omega^2`` term, which
    is a dynamic extension nothing here asks for. ``alpha`` is dimensionless in
    the sense that matters: ``F`` is ``f_xc / v`` and so carries no units, which
    is why the literature's values may be used as they stand whether the code
    writes ``v = 4 pi / q^2`` or ``8 pi / q^2``.
    """
    alpha = context.get("alpha")
    if alpha is None:
        raise ValueError(
            "the 'lrc' kernel needs its parameter: pass alpha = ... . It is "
            "material-dependent and empirical, which is the whole reason the "
            "bootstrap kernel exists"
        )
    nw, nm, _ = chi.x.shape
    diagonal = jnp.full((nm,), -float(alpha) / FPI, dtype=chi.x.dtype)
    return jnp.broadcast_to(jnp.diag(diagonal), (nw, nm, nm))


register_kernel(XCKernel(
    "lrc", _lrc,
    description="f_xc = -alpha/q^2, the empirical long-range correction",
))


# --- alda --------------------------------------------------------------------

def _alda(chi, epsi, context):
    """``F`` for the adiabatic LDA: a body matrix in ``G - G'``, no head.

    The kernel itself comes from one ``jvp`` of
    :func:`~pypresso.scf.potential.exchange_correlation` with a tangent of ones.
    That is exact where ``v_xc`` is a pointwise function of the density -- which
    is what "adiabatic **local** density approximation" means, and why a
    gradient-corrected functional is refused rather than silently truncated to
    its local part.
    """
    matrix = context["alda_matrix"]  # (nbody, nbody), built once
    nw, nm, _ = chi.x.shape
    full = jnp.zeros((nm, nm), dtype=chi.x.dtype).at[3:, 3:].set(matrix)
    return jnp.broadcast_to(full, (nw, nm, nm))


register_kernel(XCKernel(
    "alda", _alda,
    description="the adiabatic local density approximation",
))


def alda_matrix(calculation, density, sphere) -> jnp.ndarray:
    """``f_xc(G, G') / sqrt(v_G v_G')`` over the response sphere's body.

    Built once per run and handed to :func:`_alda` through the context, because
    it depends on the ground state and not on the frequency or on the iteration.
    """
    from pypresso.scf.potential import exchange_correlation

    functional = calculation.functional
    if functional.is_gradient or functional.is_meta:
        raise NotImplementedError(
            f"the 'alda' kernel with the {functional.name} functional is not "
            "implemented: ALDA is the adiabatic *local* density approximation "
            "and its kernel is pointwise in the density, which a "
            "gradient-corrected or meta functional's is not. Truncating one to "
            "its local part is an approximation with no name, so it is refused "
            "rather than taken"
        )

    density = jnp.asarray(density)
    cell = calculation.system.cell

    def potential(rho):
        return exchange_correlation(rho, cell, calculation.rho_core, functional)[0]

    # A pointwise kernel's ``jvp`` along a tangent of ones *is* the kernel:
    # ``dv_xc/drho`` at every grid point, in one pass rather than one per point.
    _, kernel_r = jax.jvp(potential, (density,), (jnp.ones_like(density),))

    # **On the dense box, not the smooth one.** The density and the potential
    # live on the dense grid, so that is the box this field is sampled on, while
    # the response sphere was *selected* on the smooth one -- which is why it
    # carries signed Miller indices rather than a flat index into one particular
    # box. This is :func:`~pypresso.basis.fft.r_to_g` without its gather,
    # because the index below is a difference ``G - G'`` and reaches beyond
    # either sphere.
    grid = np.asarray(calculation.basis.dense.grid)
    kernel_g = jnp.fft.fftn(kernel_r[0]) / float(np.prod(grid))

    # ``f_xc(G, G') = ktilde(G - G')``: the Miller difference, folded into the
    # box the way every other G index here is.
    body = np.asarray(sphere.miller)
    index = (body[:, None, :] - body[None, :, :]) % grid
    matrix = kernel_g[index[..., 0], index[..., 1], index[..., 2]]

    coulomb = sphere.sqrt_coulomb
    return matrix / (coulomb[:, None] * coulomb[None, :])


# --- bootstrap ---------------------------------------------------------------

def _bootstrap(chi, epsi, context):
    """``F = eps^-1(omega = 0) / X_00(omega = 0)``, the same matrix at every omega.

    ``genvfxc.f90``'s 210/211 branch. Written there as
    ``z1 = -1/(eps_0(1,1,1) - 1)`` with ``eps_0 = 1 - X``, so ``z1 = 1/X_00``;
    and the head, in the optical limit, is the trace over the three directions.

    On the first pass ``epsi`` is the initialisation ``tddftlr.f90`` uses --
    ``1 + X`` -- rather than the converged inverse of anything. That is the
    first-order expansion of ``eps^-1``, and it is where the paper's algorithm
    starts ("start by setting ``f_xc = 0``").
    """
    static = context["static_index"]
    x = chi.x[static]
    if epsi is None:
        epsi = jnp.eye(x.shape[-1], dtype=x.dtype) + x
    else:
        epsi = epsi[static]
    head = jnp.trace(x[:3, :3]) / 3.0
    kernel = epsi / head
    return jnp.broadcast_to(kernel, chi.x.shape)


register_kernel(XCKernel(
    "bootstrap", _bootstrap, self_consistent=True,
    description="the bootstrap kernel, iterated to self-consistency (Elk 210)",
))
#: **Two passes, not one.** ``tddftlr.f90``'s 211 branch increments its counter
#: *after* the first Dyson solve and rounds the loop again while ``it <= 1``, so
#: the kernel it ends on was built from the first pass's ``eps^-1`` rather than
#: from the initialisation. "Single iteration" counts the update, not the pass.
#: Byun and Ullrich's "0-bootstrap" is a third thing again -- one pass from the
#: *true* RPA ``eps^-1`` -- and is not this.
register_kernel(XCKernel(
    "bootstrap-1", _bootstrap, iterations=2,
    description="the bootstrap kernel, a single update (Elk 211)",
))
