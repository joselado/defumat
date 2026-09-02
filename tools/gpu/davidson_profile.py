"""Inside a Davidson step: what the subspace algebra actually costs.

`GPU.md` Phase 1 profiled the SCF by *stage* -- ``h_psi`` 6.8-17.9x, Davidson
3.9-13.5x, ``v_of_rho`` 0.74-1.3x -- and named what it could not resolve at that
granularity:

> the small dense ``eigh`` and matmuls inside a ``lax.while_loop`` ... small
> dense operations inside a device loop are a classic accelerator pathology, and
> this one is already known to be large.

The measurement that made it the first thing worth attacking is the ``ethr``
tax: tightening ``conv_thr`` from 1e-8 to 1e-10 costs the CPU **nothing** per
iteration and costs the GPU **13x** at sixteen atoms, because QE's adaptive
schedule turns a tighter threshold into more Davidson steps and a Davidson step
is where the small dense algebra lives. But "the subspace algebra" is four
different operations with four different fixes, and Phase 1's stage timings
cannot say which one to attack. This script resolves it, at the *real* shapes of
a named case, on either platform -- CPU here today, GPU in the same job that
runs `phase0.py`.

**Three candidate changes, and this script exists to arbitrate between them
rather than to justify one.** All three are things ``cegterg`` does that static
shapes gave up:

* **(a) size the solve by the live basis.** ``davidson.py`` always diagonalises
  ``nvecx = david * nbnd``, masking the inactive directions to ``shift * I``,
  where ``cegterg`` sizes its ZGEMMs and its ``cdiaghg`` by ``nbase``. The
  first steps after a collapse therefore solve a problem four times wider than
  the basis actually spans, and the two Ritz rotations -- ``O(nvecx nbnd npw)``
  -- pay the same factor. ``DAVID_NDIM``'s own docstring records this cost and
  the fix it did not take.
* **(b) size the expansion by ``notcnv``.** The correction block is ``nbnd``
  rows wide whatever the number of unconverged roots, so at the endgame -- which
  is exactly the ``ethr`` regime that costs 13x -- ``h_psi`` is applied to a
  block that is mostly zeros. The sort compaction already puts the live rows
  first, so a narrower static width is a ``dynamic_slice`` away.
* **(c) stop paying for both branches of ``generalised_eigh``'s ``cond``.**
  The 64-atom ``NaN`` fix (``a351005``) put a ``lax.cond`` between the Cholesky
  route and canonical orthogonalisation. **Under ``vmap`` a ``cond`` with a
  batched predicate lowers to ``select_n``, which evaluates both branches** --
  and ``k_batch=None``, the GPU default since ``e562427``, is exactly a ``vmap``
  over k. So every multi-k GPU subspace solve pays the fallback's two extra
  ``eigh``s on every step, on top of the Cholesky it actually uses. The physics
  sweep (``c5dc7d4``) is downstream of that commit and its seven- and ten-k
  cases paid it; the single-k cells where the 13x was measured did not, because
  ``map_k`` does not batch a single k-point.

(c) is a lowering fact rather than a hardware one, so **it is measurable here**:
the same ``vmap`` costs the same extra work on a CPU, in a different ratio.

What is reported, per operation and at the case's own shapes, is a time and the
width it was measured at, so that a change is chosen by the size of its target
rather than by which of the three is easiest to write.

    python3 tools/gpu/davidson_profile.py si16-1k-ecut30 --json out.json
    python3 tools/gpu/davidson_profile.py si10-nc --trajectory   # multi-k: (c)

**The step trajectory needs no instrumentation of the solver**, which matters
because instrumenting a ``lax.while_loop`` would change what is being measured.
Running the solver with ``max_iterations = m`` for successive ``m`` returns the
eigenvalues *at* step ``m``; the solver's own convergence test is
``|E_m - E_{m-1}| < ethr`` per root, so the same subtraction outside it
reconstructs ``notcnv`` step by step. That is (b)'s premise, measured rather
than assumed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase0 import (  # noqa: E402
    PSEUDO_DIR,
    _dial,
    conv_thr,
    memory_stats,
    provenance,
    resolve_case,
)


def timed(fn, *arrays, repeats: int = 20) -> float:
    """Warm seconds per call: compile once, then the minimum over ``repeats``.

    The minimum rather than the mean, because what is being compared is the cost
    of an operation and not the machine's variance around it -- and because a
    profile whose entries carry ±20% cannot rank two candidates that differ by
    30%, which is the whole purpose here.
    """
    import jax

    jax.block_until_ready(fn(*arrays))
    best = float("inf")
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        out = fn(*arrays)
        jax.block_until_ready(out)
        best = min(best, time.perf_counter() - start)
    return best


# ------------------------------------------------------- the converged context

def context(case: Path, threshold: float, max_iterations: int):
    """A converged cell, and the Hamiltonian its last Davidson call ran on.

    Rebuilt from the result the way ``sternheimer.py`` and ``nscf.py`` rebuild
    it, so the operator profiled here is the one the SCF's own solver sees --
    including ``D_ij``, the augmentation charge and any Hubbard term -- rather
    than a fresh one at the starting density.
    """
    import jax
    from defumat.io.pwin import read_pw_input
    from defumat.pseudo import read_upf
    from defumat.scf.driver import Calculation, run_scf
    from defumat.system import build_system

    system = build_system(read_pw_input(case))
    pseudos = tuple(read_upf(PSEUDO_DIR / s.pseudo_file) for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=threshold,
                     max_iterations=max_iterations)
    jax.block_until_ready(result.density)

    potential = calculation.potential(result.density)
    _, ddd_paw = calculation.onecenter(result.becsum)
    hubbard = None
    if calculation.is_hubbard and result.ns is not None:
        _, _, hubbard = calculation.hubbard_terms(result.ns)
    hamiltonians = calculation.hamiltonian(potential.v_scf, ddd_paw, hubbard)
    return system, calculation, result, hamiltonians[0]


# --------------------------------------------------------- synthetic operands

def _subspace_pair(size: int, dtype, seed: int = 0):
    """A projected ``(H, S)`` pair of the shape Davidson solves at width ``size``.

    ``S`` is built near the identity because that is what it is: the subspace
    vectors are S-orthonormal Ritz vectors plus normalised corrections. It is
    deliberately *well* conditioned, so ``generalised_eigh`` takes its Cholesky
    branch -- which is the branch a real solve takes, and the one whose cost the
    ``cond`` comparison below has to be measured against.
    """
    import jax.numpy as jnp
    import numpy as np

    rng = np.random.default_rng(seed)
    a = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
    h = jnp.asarray(0.5 * (a + a.conj().T), dtype)
    b = 0.05 * (rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size)))
    s = jnp.asarray(np.eye(size) + 0.5 * (b + b.conj().T), dtype)
    return h, s


# ------------------------------------------------------------- the components

def profile_subspace(hamiltonian, nbnd: int, david: int, nk: int, repeats: int) -> list[dict]:
    """(a) and (c): the projected eigenproblem, by width and by batching."""
    import jax
    from defumat.solvers.subspace import (
        _canonical_route, _cholesky_route, generalised_eigh,
    )

    dtype = hamiltonian.dtype
    rows = []
    for multiple in range(1, david + 1):
        size = multiple * nbnd
        h, s = _subspace_pair(size, dtype)
        rows.append({
            "op": "generalised_eigh", "width": size, "multiple_of_nbnd": multiple,
            "batched_over_k": 1,
            "seconds": timed(jax.jit(generalised_eigh), h, s, repeats=repeats),
        })
        rows.append({
            "op": "_cholesky_route", "width": size, "multiple_of_nbnd": multiple,
            "batched_over_k": 1,
            "seconds": timed(jax.jit(_cholesky_route), h, s, repeats=repeats),
        })
    size = david * nbnd
    h, s = _subspace_pair(size, dtype)
    rows.append({
        "op": "_canonical_route", "width": size, "multiple_of_nbnd": david,
        "batched_over_k": 1,
        "seconds": timed(jax.jit(_canonical_route), h, s, repeats=repeats),
    })

    # (c): the same two operations under a k batch. ``vmap`` of a ``cond`` whose
    # predicate is batched is a ``select_n`` over both branches, so this pair is
    # the cost of the NaN fix in the mode the GPU default puts every multi-k run
    # into. One k-point is not a batch (``batching.py``), so it is skipped.
    if nk > 1:
        import jax.numpy as jnp
        stack_h = jnp.broadcast_to(h, (nk, size, size))
        stack_s = jnp.broadcast_to(s, (nk, size, size))
        rows.append({
            "op": "generalised_eigh", "width": size, "multiple_of_nbnd": david,
            "batched_over_k": nk,
            "seconds": timed(jax.jit(jax.vmap(generalised_eigh)), stack_h, stack_s,
                             repeats=repeats),
        })
        rows.append({
            "op": "_cholesky_route", "width": size, "multiple_of_nbnd": david,
            "batched_over_k": nk,
            "seconds": timed(jax.jit(jax.vmap(_cholesky_route)), stack_h, stack_s,
                             repeats=repeats),
        })
    return rows


def profile_rotations(hamiltonian, nbnd: int, david: int, repeats: int) -> list[dict]:
    """(a) again, on the half of it that is not the eigensolve.

    ``evc = coefficients.T @ psi`` and its ``hpsi`` twin are ``O(nvecx nbnd
    npw)`` each and are done on every step. They are sized by ``nvecx`` for the
    same reason the solve is, and they are the larger term whenever ``npw`` is.
    """
    import jax
    import jax.numpy as jnp

    dtype = hamiltonian.dtype
    ndim = hamiltonian.ndim
    rows = []
    for multiple in range(1, david + 1):
        size = multiple * nbnd
        coefficients = jnp.zeros((size, nbnd), dtype) + 1.0
        psi = jnp.zeros((size, ndim), dtype) + 1.0
        rows.append({
            "op": "ritz_rotation", "width": size, "multiple_of_nbnd": multiple,
            "ndim": ndim,
            "seconds": timed(jax.jit(lambda c, p: c.T @ p), coefficients, psi,
                             repeats=repeats),
        })
    return rows


def profile_expansion(hamiltonian, nbnd: int, david: int, repeats: int) -> list[dict]:
    """(b): what a step's ``h_psi`` costs as a function of the block's width.

    If this is linear in the width, narrowing the expansion to ``notcnv`` is
    worth exactly the fraction of roots already settled; if it is flat -- which
    is what a launch-bound device does to a small block -- then (b) is worth
    nothing and the trajectory below need not be read.
    """
    import jax
    import jax.numpy as jnp

    dtype = hamiltonian.dtype
    ndim = hamiltonian.ndim
    rows = []
    widths = sorted({1, max(1, nbnd // 4), max(1, nbnd // 2), nbnd})
    for width in widths:
        block = jnp.zeros((width, ndim), dtype) + 1.0e-3
        rows.append({
            "op": "h_psi", "width": width, "fraction_of_nbnd": width / nbnd,
            "seconds": timed(jax.jit(lambda b: hamiltonian.apply(b, 0)), block,
                             repeats=repeats),
        })

    from defumat.solvers.davidson import _extend_projection

    nvecx = david * nbnd
    empty = jnp.zeros((nvecx, nvecx), dtype)
    psi = jnp.zeros((nvecx, ndim), dtype) + 1.0
    becp = jnp.zeros((nvecx, 0), dtype)
    rows.append({
        "op": "_extend_projection", "width": nbnd, "nvecx": nvecx,
        "seconds": timed(
            jax.jit(lambda a, b, c: _extend_projection(a, a, b, b, c, c, 0, nbnd)),
            empty, psi, becp, repeats=repeats),
    })
    return rows


# ------------------------------------------------------------- the trajectory

def trajectory(hamiltonian, nbnd: int, ethr: float, steps: int, psi0=None) -> dict:
    """``notcnv`` step by step, from the eigenvalues alone.

    The solver's convergence test is ``|E_m - E_{m-1}| < ethr`` per root
    (``davidson.py``'s ``settled``), so running it capped at ``m`` steps for
    successive ``m`` and differencing the results outside it reconstructs the
    same flags without touching the ``while_loop``. Each ``m`` is a separate
    compilation, so this is a diagnostic and never a timing.
    """
    import jax
    import numpy as np
    from defumat.solvers.davidson import davidson_eigensolver

    history = []
    for cap in range(1, steps + 1):
        energies, _ = jax.jit(
            lambda h, start: davidson_eigensolver(h, 0, nbnd, start, ethr=ethr,
                                                  max_iterations=cap)
        )(hamiltonian, psi0)
        history.append(np.asarray(jax.device_get(energies), dtype=float))

    rows = []
    for index in range(1, len(history)):
        change = np.abs(history[index] - history[index - 1])
        rows.append({
            "step": index + 1,
            "notcnv": int(np.sum(change >= ethr)),
            "fraction_live": float(np.mean(change >= ethr)),
            "max_change_ry": float(change.max()),
        })
    settled = next((row["step"] for row in rows if row["notcnv"] == 0), None)
    return {"ethr": ethr, "nbnd": nbnd, "steps_probed": steps,
            "seeded": psi0 is not None,
            "converged_at_step": settled, "per_step": rows}


def whole_solve(hamiltonian, nbnd: int, thresholds, repeats: int,
                psi0=None) -> list[dict]:
    """The whole solve at several ``ethr``, which is the ``ethr`` tax itself.

    `PERFORMANCE.md` measures it end to end through ``conv_thr``; here it is the
    solver alone, so the tax is separated from the extra SCF iterations a
    tighter threshold also buys.
    """
    import jax
    from defumat.solvers.davidson import davidson_eigensolver

    rows = []
    for threshold in thresholds:
        solve = jax.jit(lambda h, start: davidson_eigensolver(h, 0, nbnd, start,
                                                              ethr=threshold))
        rows.append({"ethr": threshold, "seeded": psi0 is not None,
                     "seconds": timed(solve, hamiltonian, psi0, repeats=repeats)})
    if rows:
        base = rows[0]["seconds"]
        for row in rows:
            row["vs_loosest"] = row["seconds"] / base if base else None
    return rows


# ------------------------------------------------------------------ reporting

def _seed_of(result):
    """One k-point's converged wavefunctions, as ``psi0`` for a seeded solve.

    ``SCFResult`` drops the spin axis when ``nspin = 1`` and keeps it otherwise
    (`CLAUDE.md`'s "the spin channel is the leading axis, and it is squeezed on
    the way out"), so the rank is what says which layout arrived.
    """
    import jax.numpy as jnp

    psi = jnp.asarray(result.wavefunctions)
    while psi.ndim > 2:
        psi = psi[0]
    return psi


def report(record: dict) -> None:
    shapes = record["shapes"]
    print(f"\n{record['case']}: {shapes['nat']} atoms, {shapes['nk']} k, "
          f"nbnd {shapes['nbnd']}, ndim {shapes['ndim']}, nvecx {shapes['nvecx']}")

    print("\n  the projected eigenproblem  (a) width, (c) the cond's second branch")
    print(f"  {'op':<20} {'width':>6} {'k':>4} {'ms':>10}  {'vs cholesky':>12}")
    cholesky = {(row["width"], row["batched_over_k"]): row["seconds"]
                for row in record["subspace"] if row["op"] == "_cholesky_route"}
    for row in record["subspace"]:
        base = cholesky.get((row["width"], row["batched_over_k"]))
        ratio = f"{row['seconds'] / base:.2f}x" if base else "-"
        print(f"  {row['op']:<20} {row['width']:>6} {row['batched_over_k']:>4} "
              f"{row['seconds'] * 1e3:>10.3f}  {ratio:>12}")

    print("\n  the Ritz rotations  (a), the half that scales with npw")
    for row in record["rotations"]:
        print(f"  {'x' + str(row['multiple_of_nbnd']) + ' nbnd':<20} "
              f"{row['width']:>6} {'':>4} {row['seconds'] * 1e3:>10.3f}")

    print("\n  the expansion  (b): h_psi against the block's width")
    for row in record["expansion"]:
        fraction = row.get("fraction_of_nbnd")
        tag = f"{fraction:.2f} nbnd" if fraction else ""
        print(f"  {row['op']:<20} {row['width']:>6} {'':>4} "
              f"{row['seconds'] * 1e3:>10.3f}  {tag}")

    for key, label in (("whole_solve", "cold (a band-structure solve)"),
                       ("whole_solve_seeded", "seeded (what the SCF runs)")):
        if record.get(key):
            print(f"\n  the whole solve, by ethr -- {label}")
            for row in record[key]:
                print(f"  ethr {row['ethr']:<12g} {row['seconds'] * 1e3:>10.1f} ms"
                      f"   {row['vs_loosest']:.2f}x")

    for key, label in (("trajectory", "cold"), ("trajectory_seeded", "seeded")):
        if record.get(key):
            path = record[key]
            print(f"\n  notcnv step by step at ethr {path['ethr']:g}, {label} "
                  f"(converged at step {path['converged_at_step']})")
            for row in path["per_step"]:
                bar = "#" * int(round(40 * row["fraction_live"]))
                print(f"  step {row['step']:>3}  notcnv {row['notcnv']:>4} "
                      f"/{path['nbnd']:<4} {bar}")


# ----------------------------------------------------------------------- main

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", help="a name in benchmarks/ or tests/data/qe/, or a path")
    parser.add_argument("--k-batch", default=None, help="k-points in flight")
    parser.add_argument("--band-batch", default=None, help="bands transformed at once")
    parser.add_argument("--conv-thr", type=float, default=None,
                        help="SCF threshold for the context; the profile is of the "
                             "operator, so a loose one is enough")
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=20,
                        help="timing repeats; the minimum is reported")
    parser.add_argument("--trajectory", action="store_true",
                        help="reconstruct notcnv step by step (recompiles per step)")
    parser.add_argument("--trajectory-steps", type=int, default=14)
    parser.add_argument("--trajectory-ethr", type=float, default=1.0e-13,
                        help="the endgame threshold, which is where the tax is")
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args(argv)

    dials = {}
    for name, value in (("DEFUMAT_K_BATCH", arguments.k_batch),
                        ("DEFUMAT_BAND_BATCH", arguments.band_batch)):
        if value is not None:
            os.environ[name] = _dial(value)
            dials[name.lower().removeprefix("defumat_")] = os.environ[name]

    case = resolve_case(arguments.case)
    import defumat  # noqa: F401  -- x64 before any array exists
    from defumat import batching
    from defumat.solvers.davidson import DAVID_NDIM

    record = {
        "case": case.name,
        "provenance": provenance(),
        "dials": {"k_batch": str(batching.DEFAULT_K_BATCH),
                  "band_batch": str(batching.DEFAULT_BAND_BATCH),
                  "given": dials or None},
    }
    print(f"# defumat GPU.md phase 1 -- inside a Davidson step, {case.name}")
    print(f"# {record['provenance']['device_kind']} "
          f"({record['provenance']['platform']}), jax {record['provenance']['jax']}")

    threshold = arguments.conv_thr if arguments.conv_thr is not None else conv_thr(case)
    system, calculation, result, hamiltonian = context(case, threshold,
                                                       arguments.max_iterations)
    nbnd = int(result.eigenvalues.shape[-1])
    david = DAVID_NDIM
    record["shapes"] = {
        "nat": int(system.structure.nat),
        "nk": int(system.kpoints.nk),
        "nbnd": nbnd,
        "npwx": int(calculation.basis.npwx),
        "ndim": int(hamiltonian.ndim),
        "nvecx": david * nbnd,
        "david": david,
        "ecutwfc": float(system.ecutwfc),
        "scf_iterations": int(result.iterations),
        "scf_conv_thr": threshold,
    }

    record["subspace"] = profile_subspace(hamiltonian, nbnd, david,
                                          int(system.kpoints.nk), arguments.repeats)
    record["rotations"] = profile_rotations(hamiltonian, nbnd, david, arguments.repeats)
    record["expansion"] = profile_expansion(hamiltonian, nbnd, david, arguments.repeats)
    # The SCF re-seeds this solver from the previous iteration's wavefunctions
    # and re-runs it on a tightening ``ethr``, so a *cold* solve is the
    # band-structure regime and not the one the 13x tax was measured in. Both
    # are reported and the seeded one is the SCF's.
    seed = _seed_of(result)
    thresholds = (1.0e-6, 1.0e-10, 1.0e-13)
    repeats = max(3, arguments.repeats // 4)
    record["whole_solve"] = whole_solve(hamiltonian, nbnd, thresholds, repeats)
    record["whole_solve_seeded"] = whole_solve(hamiltonian, nbnd, thresholds, repeats,
                                               psi0=seed)
    if arguments.trajectory:
        record["trajectory"] = trajectory(hamiltonian, nbnd, arguments.trajectory_ethr,
                                          arguments.trajectory_steps)
        record["trajectory_seeded"] = trajectory(
            hamiltonian, nbnd, arguments.trajectory_ethr,
            arguments.trajectory_steps, psi0=seed)
    record["memory"] = memory_stats()

    report(record)
    if arguments.json:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(record, indent=2, default=str))
        print(f"\nwritten: {arguments.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
