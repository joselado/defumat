"""Solving ``r(rho) = 0`` with the Jacobian, rather than iterating ``F``.

``scf/residual.py`` writes self-consistency as a residual; this module solves it.
The default remains mixing -- ``run_scf``'s loop, unchanged -- and this is the
other entry in the registry (rule R4).

**Newton-Krylov.** The step solves ``J d = -r`` for the Newton direction, with
``J`` never formed: GMRES asks only for ``J v``, which is one forward-mode
differentiation of one SCF iteration. Forming ``J`` densely is not an option and
is worth saying explicitly -- it is ``(nspin * nr)^2``, which on the smallest
slab here is a 90000 x 90000 matrix and on anything real is astronomically
worse. pyqula measured the same wall (``scftk/densitydensity_jax.py``: 32
orbitals did not finish three dense-Jacobian Newton steps in 280 s) on a problem
thousands of times smaller.

**Two things pyqula established that this does not repeat** (``PLAN.md`` P22):

* minimising the *physical energy* off the self-consistency surface fails --
  the SCF solution is a saddle of that off-shell functional, not a minimum, so
  a descent method walks away from it with a correct gradient;
* minimising ``||r||^2`` with a scalar-loss optimiser (L-BFGS) stalls, because
  it sees only ``J^T r`` and discards the residual *vector*. A Krylov method
  uses that vector directly, which is why this is a root-finder and not a
  minimiser.

**Two JVP backends, and finite differences won.** ``autodiff`` is ``jax.jvp``
through the step, Davidson's ``lax.while_loop`` included; ``finite-difference``
is a central difference of ``r``, which is classic Jacobian-free Newton-Krylov
and shares no machinery with the first. Measured on the aluminium slab
(``tests/regression/test_scf_solvers.py`` pins both numbers):

* from a **cold** start the two disagree by **109%** -- differentiating
  Davidson's trajectory from the pseudo-atomic orbitals is the derivative of a
  different map, one that merely lands in the same place;
* from a **warm** start they agree to **0.8%**, which is a Jacobian good enough
  to give a Krylov solver a direction and is not the response operator;
* and the finite difference is **4-7x faster**, because forward mode through the
  ``while_loop`` costs more than the two extra primal solves do when both start
  from a converged guess.

So ``finite-difference`` is the default. This is the empirical case for the
Sternheimer rule of ``PLAN.md`` P22c: the response has to be *written down*, not
taken from the eigensolver's tape.

**Cost, stated up front because the wall-clock case is not the case for this.**
One outer iteration is one residual evaluation, plus one line-search
evaluation per backtrack, plus **two** per GMRES iteration -- the default
Jacobian action is a central difference, so it costs two evaluations of ``F``,
and each evaluation is a diagonalisation, 84% of an SCF step
(``PERFORMANCE.md``). Anderson's whole iteration is one such diagonalisation and
a negligible least-squares solve. So Newton-Krylov is ahead only when it cuts
the *outer* count by more than the inner Krylov work adds -- and measured over a
sweep of increasingly ill-conditioned aluminium slabs it never is, because the
outer count stays flat at the cost of GMRES iterations that grow to replace it.
The numbers are in ``PERFORMANCE.md``; the summary is 19 to 139 evaluations
against Kerker-preconditioned Anderson's 14 to 36.

**So this is not the default and should not become one until the Jacobian action
is cheap** -- which means the Sternheimer rule of ``PLAN.md`` P22c, a projected
CG solve per occupied band rather than another diagonalisation. What the solver
buys today is a *capability*: Newton is stability-blind and converges on
whichever root it starts nearest, so it reaches **unstable** SCF solutions that
a damped mixing dynamics cannot hold.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.linalg import LinearOperator, lgmres

__all__ = ["NewtonKrylovResult", "newton_krylov", "SCF_SOLVERS", "get_scf_solver"]


@dataclass
class NewtonKrylovResult:
    """The solution and, as importantly, what it cost."""

    x: np.ndarray
    psi: object
    converged: bool
    accuracy: float
    iterations: int
    #: Evaluations of ``F``. This is the currency: each is one diagonalisation,
    #: and it is the only fair thing to compare against a mixer's iteration
    #: count.
    steps: int
    jvps: int
    history: list = field(default_factory=list)


def newton_krylov(
    residual,
    x0,
    psi0,
    accuracy_of,
    conv_thr: float = 1.0e-8,
    max_iterations: int = 30,
    gmres_maxiter: int = 20,
    forcing: float = 0.1,
    precondition=None,
    jvp: str = "finite-difference",
    steps_already_taken: int = 0,
    verbose: bool = False,
) -> NewtonKrylovResult:
    """Drive ``r(x)`` to zero by inexact Newton with a GMRES inner solve.

    ``accuracy_of(r)`` is the *same* measure the mixing loop converges on --
    QE's ``dr2``, the Hartree energy of the residual -- so ``conv_thr`` means
    here what it means in a ``pw.x`` input and the two solvers can be compared
    at all.

    ``forcing`` is the inexact-Newton forcing term: the inner solve is only
    asked for a relative residual of ``forcing``, because an exactly solved
    Newton direction is wasted while ``x`` is still far from the root. It is the
    single most important knob for the cost, since GMRES iterations *are* the
    cost.

    ``precondition`` is applied to the Krylov system, and Kerker is the natural
    choice: it is an approximation to ``-J^-1``, which is precisely what a
    preconditioner should be. **This is the honest comparison** -- a
    preconditioner that helps GMRES helps the mixer too, so both sides of any
    measurement get it.
    """
    if jvp not in ("autodiff", "finite-difference"):
        raise ValueError(f"unknown jvp backend {jvp!r}")

    x = np.asarray(x0, dtype=float).copy()
    psi = psi0
    # Warm-up evaluations of ``F`` taken by the caller before the solve started
    # count here too: ``steps`` is what a mixing run's iteration count is
    # compared against, and a currency that quietly omits part of the bill is
    # worse than no currency.
    counters = {"steps": int(steps_already_taken), "jvps": 0}
    history = []

    def evaluate(y, warm):
        counters["steps"] += 1
        r, psi_new = residual.residual(y, warm)
        return np.asarray(r, dtype=float), psi_new

    r, psi = evaluate(x, psi)
    accuracy = float(accuracy_of(r))
    converged = accuracy < conv_thr
    # The starting point is recorded so that a convergence history plotted
    # against ``steps`` begins where the mixer's does, and so that a solve that
    # converges before its first outer iteration is still visible as something
    # rather than as an empty history.
    history.append(
        {"iteration": 0, "accuracy": accuracy, "step": 0.0, "gmres": 0,
         "steps": counters["steps"], "jvps": 0, "seconds": 0.0}
    )

    for iteration in range(1, max_iterations + 1):
        if converged:
            break
        started = time.perf_counter()

        # ``psi`` is the converged eigenvector at the current ``x``, so it is
        # the right warm start for every JVP taken here: ``x`` does not move
        # during the inner solve.
        warm = psi

        def matvec(v):
            counters["jvps"] += 1
            if jvp == "autodiff":
                out, _ = residual.jvp(x, v, warm)
                return np.asarray(out, dtype=float)
            counters["steps"] += 2
            return np.asarray(
                residual.jvp_finite_difference(x, v, warm), dtype=float
            )

        if not np.all(np.isfinite(r)):
            # The *residual* rather than the direction, which the guard below
            # catches. What puts a NaN here is a step that left the region where
            # ``F`` is a function of the density at all, and the reproducible way
            # to do that is fixed occupations whose filled/empty boundary cuts a
            # **degenerate multiplet**: which member of the multiplet the
            # eigensolver hands back is arbitrary, so two evaluations at the same
            # density give different densities back and there is no map to solve.
            # An oxygen atom at ``tot_magnetization = 2`` is exactly that -- one
            # electron in a threefold 2p shell -- and it dies here at the first
            # line search, where the same atom with a smearing converges in four
            # Newton steps and a *gapped* fixed LSDA cell agrees with the mixer
            # to 5e-12. The mixer damps its way past it; a Newton solve cannot.
            raise FloatingPointError(
                "the SCF residual is not finite, so there is no Newton step to "
                "take. The reproducible cause is an occupation rule that splits "
                "a degenerate multiplet -- occupations='fixed' whose highest "
                "occupied and lowest unoccupied levels coincide -- which makes "
                "the density map multivalued rather than merely stiff. Check "
                "homo against lumo, and use a smearing or scf_solver='mixing' "
                "if they are equal"
            )
        operator = LinearOperator((x.size, x.size), matvec=matvec, dtype=float)
        # ``-P`` rather than ``P``: the Jacobian of ``r = F - x`` is close to
        # ``-1`` where the iteration is well behaved, so the approximate inverse
        # is the *negative* of the mixer's preconditioner. Getting this sign
        # wrong does not fail -- GMRES simply makes no progress.
        inverse = (
            None if precondition is None
            else LinearOperator((x.size, x.size),
                                matvec=lambda v: -np.asarray(precondition(v), dtype=float),
                                dtype=float)
        )
        direction, info = lgmres(
            operator, -r, rtol=forcing, atol=0.0, maxiter=gmres_maxiter, M=inverse,
        )
        if not np.all(np.isfinite(direction)):
            raise FloatingPointError(
                "the Newton direction is not finite; the Jacobian action has "
                "diverged. This is what an unconverged inner eigensolver looks "
                "like from here -- tighten ethr"
            )

        # Backtracking on the residual norm. A full Newton step is right near
        # the root and can be wild far from it, and the density is a quantity
        # with a sign: an overshoot that drives it negative makes the next
        # exchange-correlation evaluation meaningless rather than merely worse.
        norm = float(np.linalg.norm(r))
        length = 1.0
        for _ in range(4):
            trial = x + length * direction
            r_trial, psi_trial = evaluate(trial, warm)
            if float(np.linalg.norm(r_trial)) < (1.0 - 1.0e-4 * length) * norm:
                break
            length *= 0.5
        else:
            # Four halvings and still no decrease. The shortest step is taken
            # anyway rather than raising: the residual norm is not the quantity
            # being converged (``accuracy_of`` is), and a Newton direction from
            # an inexact inner solve can fail this test while still being an
            # improvement in the measure that decides convergence. What must not
            # happen is silence, so it is reported in ``history`` as
            # ``step`` = 0.0625 and the caller can see the solver was crawling.
            pass
        x, r, psi = trial, r_trial, psi_trial

        accuracy = float(accuracy_of(r))
        converged = accuracy < conv_thr
        history.append(
            {"iteration": iteration, "accuracy": accuracy, "step": length,
             "gmres": info, "steps": counters["steps"], "jvps": counters["jvps"],
             "seconds": time.perf_counter() - started}
        )
        if verbose:
            print(f"  newton {iteration:3d}   accuracy = {accuracy:.2e}   "
                  f"lambda = {length:.3f}   F evaluations = {counters['steps']:4d}   "
                  f"jvps = {counters['jvps']:4d}   {history[-1]['seconds']:.1f}s",
                  flush=True)

    return NewtonKrylovResult(
        x=x, psi=psi, converged=converged, accuracy=accuracy,
        iterations=len(history) - 1, steps=counters["steps"], jvps=counters["jvps"],
        history=history,
    )


#: Name -> solver of the SCF fixed point, as written in ``scf_solver``.
#: ``"mixing"`` is the loop in ``driver.py`` and is the default; it is in the
#: registry as a name rather than a function because it is not written as a
#: solver of a residual at all.
SCF_SOLVERS = {
    "mixing": None,
    "newton-krylov": newton_krylov,
    "newton_krylov": newton_krylov,
}


def get_scf_solver(name: str):
    try:
        return SCF_SOLVERS[name.lower()]
    except KeyError as error:
        raise ValueError(
            f"unknown scf_solver {name!r}; expected one of {sorted(SCF_SOLVERS)}"
        ) from error
