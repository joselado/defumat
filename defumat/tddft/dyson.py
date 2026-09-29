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
``omega = 0`` slice, and used at every frequency -- and the bootstrap reads
``eps^-1`` at that slice alone, and so does the convergence test. So for a
static kernel (every one registered here, :attr:`~defumat.tddft.kernels.
XCKernel.static`) the fixed point is iterated on **the one frequency it
depends on**, and the whole axis is screened once with the kernel it
converged to: the same ``F``, the same history and the same ``eps^-1`` to
round-off, for one matrix solve per pass instead of ``nw``, and without ``nw``
copies of ``F``. The final screening walks the frequencies ``w_batch`` at a
time (:func:`~defumat.batching.resolve_w_batch`), so only one chunk's
temporaries are in flight (``GPU-MEMORY-NEXT.md`` item 12, ``MEMORY-AUDIT.md``
A10); the physics still costs one ``chi_0``.
"""

from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from defumat.batching import resolve_w_batch
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
    #: ``(nw, nm, nm)`` -- the symmetrised kernel ``v^-1/2 f_xc v^-1/2``; for
    #: a static kernel (every one registered here) ``(1, nm, nm)``, the one
    #: matrix that is used at every frequency, which broadcasts against them.
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
    w_batch: int | None | str = "default",
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
        w_batch: how many frequencies are screened at once
            (:func:`~defumat.batching.resolve_w_batch`): the whole axis on a
            CPU, a budgeted chunk on an accelerator. Each frequency is its own
            matrix solve, so it moves nothing beyond round-off.
    """
    rule = get_kernel(kernel)
    context = dict(context or {})
    context["static_index"] = static_index

    x = chi.x
    nw, nm = int(x.shape[0]), int(x.shape[-1])
    identity = jnp.eye(nm, dtype=x.dtype)
    batch = resolve_w_batch(
        w_batch, nw=nw,
        # One frequency's temporaries: ``X``, ``F X``, the matrix inverted and
        # its inverse.
        block_bytes=4 * nm * nm * np.dtype(x.dtype).itemsize)

    def screen(response, f):
        """``1 + X (1 - X - F X)^-1`` at every frequency of ``response``.

        ``f`` is ``(1, nm, nm)`` for a static kernel, broadcast against every
        frequency, or one matrix per frequency. A chunked walk is a host loop
        writing each chunk into one donated output, so what is in flight is
        the output, ``X`` and one chunk's temporaries -- a compiled loop over
        the same chunks kept a second copy of its output carry (measured:
        60.0 MB against 40.4 for the whole axis, at 200 frequencies of
        ``nm = 115`` in chunks of 32).
        """
        n = response.shape[0]
        if batch is None or batch >= n:
            return _screen(response, f, identity)
        out = jnp.zeros_like(response)
        for start in range(0, n, batch):
            stop = min(start + batch, n)
            part = _screen(response[start:stop],
                           f if f.shape[0] == 1 else f[start:stop], identity)
            out = _write(out, part, start)
        return out

    # A static kernel depends on the static slice alone, and so does the
    # convergence test: iterate there, then screen the whole axis once.
    if rule.static:
        looped = eqx.tree_at(lambda c: c.x, chi,
                             x[static_index:static_index + 1])
        loop_context = {**context, "static_index": 0}
        index = 0
    else:
        looped, loop_context, index = chi, context, static_index

    epsi = None
    fxc = None
    previous = None
    history: list[float] = []
    converged = not rule.self_consistent

    passes = max_iterations if rule.self_consistent else (rule.iterations or 1)
    iteration = 0
    for iteration in range(1, passes + 1):
        fxc = rule.build(looped, epsi, loop_context)
        if rule.static:
            fxc = fxc[:1]  # one matrix, used at every frequency
        # ``genvfxc`` returns ``F X``, not ``F``: it right-multiplies by
        # ``vchi0`` before returning, and every consumer expects that.
        epsi = screen(looped.x, fxc)

        if not rule.self_consistent:
            continue
        # ``tddftlr.f90``'s test, transcribed including its shape: a difference
        # of *moduli* of one complex entry, the head of ``F X`` at omega = 0.
        current = complex((fxc[index] @ looped.x[index])[0, 0])
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

    if rule.static:
        epsi = screen(x, fxc)
    at_static = fxc[index]
    head = jnp.real(at_static[0, 0])
    head_x = jnp.real((at_static @ x[static_index])[0, 0])
    return DysonSolution(
        epsilon_inverse=epsi,
        fxc=fxc,
        epsilon=macroscopic_tensor(epsi),
        epsilon_no_local_fields=identity[None, :3, :3] - x[:, :3, :3],
        iterations=iteration,
        converged=bool(converged),
        kernel=rule.name,
        alpha=float(-FPI * head),
        alpha_elk=float(-FPI * head_x),
        history=tuple(history),
    )


@jax.jit
def _screen(response, f, identity):
    """``X (1 - X - F X)^-1 + 1`` for a block of frequencies; ``F`` broadcasts."""
    return response @ jnp.linalg.inv(identity - response - f @ response) + identity


@partial(jax.jit, donate_argnums=0)
def _write(out, part, start):
    """``out[start:start + len(part)] = part``, in the donated buffer."""
    return jax.lax.dynamic_update_slice_in_dim(out, part, start, 0)


def macroscopic_tensor(epsilon_inverse: jnp.ndarray) -> jnp.ndarray:
    """``eps_M``: the inverse of the **3x3 head** of ``eps^-1``, per frequency.

    ``tddftlr.f90``'s "find the macroscopic part of eps by inverting the 3x3
    head only". The whole content of the local-field effect is that this is not
    the head of the inverse of the whole matrix.
    """
    return jnp.linalg.inv(jnp.asarray(epsilon_inverse)[:, :3, :3])
