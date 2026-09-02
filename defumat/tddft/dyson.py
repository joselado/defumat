"""The Dyson equation, and the bootstrap kernel's fixed point.

``PLAN.md`` P37, and ``tddftlr.f90`` transcribed. Given
``X = v^{1/2} chi_0 v^{1/2}`` and a symmetrised kernel ``F``, the response
obeys

    eps^-1 = 1 + X (1 - X - F X)^-1,

which is Eq. (1) of PRL **107**, 186401 written in the symmetric basis. What an
experiment reads is not that matrix but its **macroscopic part**, and the
difference between the two is the local-field effect:

    eps_M(omega) = [ (eps^-1)_head ]^-1,

the inverse of the **3x3 head block alone**. Inverting the whole matrix and
taking its head gives something else -- the microscopic ``eps``, whose head in
RPA is exactly ``1 - X_head``, the *no-local-field* dielectric function. Elk
writes both, from the same array, thirty lines apart (``EPSILON_TDDFT_ij.OUT``
against ``EPSM_TDDFT_ij.OUT``), and taking the wrong one is invisible: it is
smooth, positive, has the right peaks and is 9% too large on silicon.

**The bootstrap is a fixed point of this equation and its own definition.**
``f_xc`` is built from ``eps^-1``, which is built from ``f_xc``. The paper's
algorithm is to start from ``f_xc = 0``, solve, rebuild, and repeat; Elk starts
one step earlier, from ``eps^-1 = 1 + X``, which is the same expansion to first
order. The loop is a plain Python one, as the SCF's is and for the same reason:
its exit test is data-dependent. **Failure to converge is an error**, not a
warning -- ``tddftlr.f90`` stops at ``maxit = 500`` and so does this.

**The kernel is static and the spectrum is not.** ``F`` is built once, from the
``omega = 0`` slice, and used at every frequency -- so the iteration costs one
matrix solve per frequency per pass, and the physics costs one ``chi_0``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from defumat.tddft.kernels import get_kernel
from defumat.units import FPI

__all__ = ["DysonSolution", "solve_dyson", "macroscopic_tensor"]

#: ``tddftlr.f90``'s ``maxit``.
MAX_ITERATIONS = 500

#: ``tddftlr.f90``'s convergence test on the head of ``F X``.
TOLERANCE = 1.0e-8


class DysonSolution(eqx.Module):
    """``eps^-1`` at every frequency, and the kernel that produced it."""

    #: ``(nw, nm, nm)`` -- the inverse dielectric matrix.
    epsilon_inverse: jnp.ndarray
    #: ``(nw, nm, nm)`` -- the symmetrised kernel ``v^-1/2 f_xc v^-1/2``.
    fxc: jnp.ndarray
    #: ``(nw, 3, 3)`` -- the macroscopic tensor, local fields included.
    epsilon: jnp.ndarray
    #: ``(nw, 3, 3)`` -- the same without local fields, ``1 - X_head``. Carried
    #: because the gap between the two *is* the local-field effect, and a
    #: reader who wants to know how large it is should not have to rerun.
    epsilon_no_local_fields: jnp.ndarray
    iterations: int = eqx.field(static=True)
    converged: bool = eqx.field(static=True)
    kernel: str = eqx.field(static=True)
    #: The long-range-correction parameter this kernel's head is equivalent to,
    #: ``alpha = -4 pi F_00``, so that any kernel here can be read on the same
    #: scale as ``lrc``'s one number. **Not the quantity Elk prints** under that
    #: name -- see :attr:`alpha_elk`.
    alpha: float = eqx.field(static=True)
    #: ``-4 pi (F X)_00``, which is what ``tddftlr.f90`` prints beside "``
    #: multiplied by -4 pi gives alpha``". By the time it reads ``vfxc``,
    #: ``genvfxc`` has already right-multiplied by ``vchi0``, so the array holds
    #: ``F X`` and not ``F``; the routine's own comment on that line is stale.
    #: Carried separately so a comparison with Elk compares like with like.
    alpha_elk: float = eqx.field(static=True)
    #: What the convergence test saw on each pass, for the same reason the SCF
    #: keeps its history.
    history: tuple = eqx.field(static=True, default=())


def solve_dyson(
    chi,
    kernel: str = "bootstrap",
    context: dict | None = None,
    *,
    static_index: int = 0,
    tolerance: float = TOLERANCE,
    max_iterations: int = MAX_ITERATIONS,
    verbose: bool = False,
) -> DysonSolution:
    """Solve ``eps^-1 = 1 + X (1 - X - F X)^-1`` with the named kernel.

    Args:
        chi: the :class:`~defumat.tddft.chi0.ChiZero` to screen.
        kernel: a name from :func:`~defumat.tddft.kernels.kernel_names`.
        context: extra ingredients a kernel needs -- ``alpha`` for ``lrc``,
            ``alda_matrix`` for ``alda``. ``static_index`` is added here.
        static_index: which frequency is ``omega = 0``. The bootstrap kernel is
            built from that slice alone, and ``init3.f90`` puts the point there
            deliberately -- carrying ``i eta``, not zero.
    """
    rule = get_kernel(kernel)
    context = dict(context or {})
    context["static_index"] = static_index

    x = chi.x
    identity = jnp.eye(x.shape[-1], dtype=x.dtype)
    eps0 = identity[None] - x

    epsi = None
    fxc = None
    previous = None
    history: list[float] = []
    converged = not rule.self_consistent

    passes = max_iterations if rule.self_consistent else (rule.iterations or 1)
    iteration = 0
    for iteration in range(1, passes + 1):
        fxc = rule.build(chi, epsi, context)
        # ``genvfxc`` returns ``F X``, not ``F``: it right-multiplies by
        # ``vchi0`` before returning, and every consumer expects that.
        fxc_x = fxc @ x
        epsi = x @ jnp.linalg.inv(eps0 - fxc_x) + identity[None]

        if not rule.self_consistent:
            continue
        # ``tddftlr.f90``'s test, transcribed including its shape: a difference
        # of *moduli* of one complex entry, the head of ``F X`` at omega = 0.
        current = complex(fxc_x[static_index, 0, 0])
        change = float("inf") if previous is None else abs(
            abs(previous) - abs(current)
        )
        previous = current
        history.append(change)
        if verbose:
            print(f"  bootstrap iter {iteration}: |d head(F X)| = {change:.3e}")
        if change <= tolerance:
            converged = True
            break

    if rule.self_consistent and not converged:
        raise RuntimeError(
            f"the {kernel!r} kernel did not converge in {max_iterations} "
            f"iterations: the head of F X last moved by {history[-1]:.3e}, "
            f"against a tolerance of {tolerance:.1e}. tddftlr.f90 stops here "
            "too. A spectrum from an unconverged kernel is not the spectrum of "
            "any functional, so it is refused rather than returned"
        )

    head = jnp.real(fxc[static_index, 0, 0])
    head_x = jnp.real((fxc @ x)[static_index, 0, 0])
    return DysonSolution(
        epsilon_inverse=epsi,
        fxc=fxc,
        epsilon=macroscopic_tensor(epsi),
        epsilon_no_local_fields=eps0[:, :3, :3],
        iterations=iteration,
        converged=bool(converged),
        kernel=rule.name,
        alpha=float(-FPI * head),
        alpha_elk=float(-FPI * head_x),
        history=tuple(history),
    )


def macroscopic_tensor(epsilon_inverse: jnp.ndarray) -> jnp.ndarray:
    """``eps_M``: the inverse of the **3x3 head** of ``eps^-1``, per frequency.

    ``tddftlr.f90``'s "find the macroscopic part of eps by inverting the 3x3
    head only". The whole content of the local-field effect is that this is not
    the head of the inverse of the whole matrix.
    """
    return jnp.linalg.inv(jnp.asarray(epsilon_inverse)[:, :3, :3])
