"""``E(q)`` two ways: the SCF energies, and the integral of ``dE/dq``.

The point of computing both is that they are independent estimates of the same
curve and they carry k-sampling error differently -- the energy is a Brillouin
zone integral over every occupied state, the gradient a Fermi-surface-weighted
one -- so where they agree the mesh is converged, and where they do not the gap
is a measurement rather than a bug in either.

Two things make the comparison worth reading rather than a formality:

* ``pw.x`` has no spin spiral, so there is no reference output to check either
  curve against. These two, and the circuit closing on itself at ``Gamma``, are
  the checks there are.
* the gradient is taken at a **frozen plane-wave sphere**, so it does not see
  the jump the energy takes wherever a plane wave crosses the cutoff. The
  integrated curve is therefore smooth by construction and the direct one is
  not, and the difference between them at a fixed ``ecutwfc`` is the size of
  that Pulay error.

The circuit is not a straight line, so ``dE/dq`` is integrated against the
cartesian arc length with the gradient projected on each segment's own tangent.
At a corner the tangent is genuinely discontinuous and the point belongs to both
segments, which is handled by integrating segment by segment rather than by
differencing across the corner.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

RY_TO_MEV = 13605.693122994


def _runner():
    spec = importlib.util.spec_from_file_location(
        "run_spiral", Path(__file__).with_name("run_spiral.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--path", default="GKMG")
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args()

    files = sorted(args.results.glob(f"{args.path}-q*.json"))
    rec = sorted((json.loads(f.read_text()) for f in files), key=lambda r: r["q_index"])
    if not rec:
        raise SystemExit(f"no {args.path}-q*.json under {args.results}")

    runner = _runner()
    npoints = rec[0]["npoints"]
    qs = runner.q_path(args.path, npoints)
    s_all = runner.path_abscissa(qs)

    index = np.array([r["q_index"] for r in rec])
    if len(index) != npoints:
        print(f"!! {len(index)} of {npoints} points present: "
              f"missing {sorted(set(range(npoints)) - set(index.tolist()))}")

    q = np.array([r["q"] for r in rec])
    s = s_all[index]
    energy = np.array([r["total_energy"] for r in rec])
    grad = np.array([r["gradient"] if r["gradient"] else [np.nan] * 3 for r in rec])
    moment = np.linalg.norm([r["magnetization_vector"] for r in rec], axis=1)
    converged = np.array([r["converged"] for r in rec])

    # dE/ds, and the integral of it, **per interval rather than per point**.
    #
    # Parameterise a segment by cartesian arc length, q(s) = a + (b - a)(s -
    # s_a)/L with L the cartesian length; then dq_i/ds = (b - a)_i / L and the
    # chain rule gives
    #
    #     dE/ds = sum_i (dE/dq_i) (b - a)_i / L,
    #
    # a plain contraction of a covector with a vector, which carries no metric of
    # its own -- the metric is already spent in L. **The trap is to "normalise"
    # the tangent in the hexagonal metric** and treat the result as an orthonormal
    # direction: dE/dq is a derivative with respect to *crystal* coordinates, so
    # dotting it with a metric-unit tangent is off by tangent.tangent, which is
    # 2/3 on Gamma->K and 1 on M->Gamma -- wrong on two legs and right on the
    # third, which is the shape of error that survives a spot check.
    #
    # **And the tangent belongs to the interval, not to the point.** dE/ds is
    # genuinely discontinuous at a corner: the gradient there is one vector, but
    # it is projected on the Gamma->K tangent for the interval arriving and on
    # the K->M tangent for the interval leaving, and the two differ. Assigning
    # one tangent per point instead gets every interval after a corner wrong by
    # that difference -- and a synthetic test cannot see it if the fixture builds
    # its gradients through the same per-point assignment, because the error then
    # cancels exactly. So the segment is found from each interval's *midpoint*,
    # which lies on exactly one leg and is unambiguous.
    letters = list(args.path)
    nodes = [np.array(runner.CORNERS[c], float) for c in letters]
    segments = list(zip(nodes, nodes[1:]))

    def tangent_at(point):
        """The unit-arc-length tangent of the leg ``point`` lies on, or None."""
        for a, b in segments:
            if _fraction_along(a, b, point) is not None:
                return (b - a) / runner._hex_length(b - a)
        return None

    interval_tangent, crossings = [], []
    for j in range(len(rec) - 1):
        tang = tangent_at(0.5 * (q[j] + q[j + 1]))
        if tang is None:
            # Only reachable when a point is missing and the surviving interval
            # spans a corner. The chord is the best available and the interval is
            # named, because a silently chorded corner is a wrong number.
            chord = q[j + 1] - q[j]
            tang = chord / runner._hex_length(chord)
            crossings.append((index[j], index[j + 1]))
        interval_tangent.append(tang)

    # Displayed per point: the leg it is entering (the leg it leaves, for the
    # last point). At a corner this is one of the two one-sided values and the
    # other is its partner across the discontinuity.
    dEds = np.array([
        float(grad[j] @ (interval_tangent[j] if j < len(interval_tangent)
                         else interval_tangent[-1]))
        for j in range(len(rec))])

    # Each interval's own tangent is used on *both* its endpoints, so a corner
    # gradient is consumed twice, once per leg -- which is the point.
    left = np.array([float(grad[j] @ interval_tangent[j])
                     for j in range(len(rec) - 1)])
    right = np.array([float(grad[j + 1] @ interval_tangent[j])
                      for j in range(len(rec) - 1)])
    ds = np.diff(s)
    trapezoid = 0.5 * (left + right) * ds

    # The trapezoid is second order, and on twenty points of a curve this shape
    # that is 4% of the span -- measured, by refining a fixture whose gradients
    # come from an analytic function: 4.26 -> 1.09 -> 0.27 -> 0.068 meV as the
    # sampling doubles, a clean h^2. That error would be the whole content of the
    # comparison rather than the physics, so the integral is done instead with a
    # cubic spline through dE/ds -- **built per leg**, since dE/ds is genuinely
    # discontinuous at a corner and a spline across one would smear the kink over
    # its neighbours. The trapezoid is kept and reported beside it: where the two
    # quadratures differ by more than the two *curves* do, the sampling is too
    # coarse to say anything and the comparison is about the rule, not the code.
    contributions = _spline_integral(s, left, right, interval_tangent, trapezoid)

    direct = (energy - energy[0]) * RY_TO_MEV
    integrated = np.concatenate([[0.0], np.cumsum(contributions)]) * RY_TO_MEV
    by_trapezoid = np.concatenate([[0.0], np.cumsum(trapezoid)]) * RY_TO_MEV

    if crossings:
        print("!! these intervals span a corner because a point is missing, and "
              "are integrated on the chord: " + ", ".join(map(str, crossings)))

    label = _labels(runner, args.path, q)
    print(f"{'i':>3} {'q (lattice)':>24} {'':>4} {'E-E0 [meV]':>11} "
          f"{'int dE/dq':>11} {'diff':>8} {'dE/ds [Ry]':>11} "
          f"{'|m| [muB]':>10} {'it':>4} {'ok':>3}")
    for j in range(len(rec)):
        print(f"{index[j]:3d} {str(np.round(q[j], 4)):>24} {label[j]:>4} "
              f"{direct[j]:11.3f} {integrated[j]:11.3f} "
              f"{direct[j]-integrated[j]:8.3f} {dEds[j]:11.5f} {moment[j]:10.4f} "
              f"{rec[j]['iterations']:4d} {'y' if converged[j] else 'N':>3}")

    if not converged.all():
        print("\n!! the rows marked N did not converge and are not results")

    quadrature = np.nanmax(np.abs(integrated - by_trapezoid))
    disagreement = np.nanmax(np.abs(direct - integrated))
    print(f"\nagreement of the two curves: max |direct - integrated| = "
          f"{disagreement:.3f} meV over a span of "
          f"{np.nanmax(direct) - np.nanmin(direct):.3f} meV")
    # The gap between the two quadratures is an *upper* bound on the spline's
    # own error and a loose one, because it is dominated by the trapezoid's
    # second-order error rather than the spline's fourth-order one. Measured on
    # a fixture whose gradients come from an analytic function, at these twenty
    # points: spline 0.85 meV from the truth, trapezoid 4.26, and the gap
    # between them 5.07. So read this as "no worse than", and read the curve
    # comparison above as meaningful when it is well inside it.
    print(f"quadrature bound (spline vs trapezoid on the same gradients): "
          f"{quadrature:.3f} meV -- an upper bound on the integration error, "
          f"loose by roughly 5x on a fixture of this shape"
          + ("\n  the two curves differ by less than that bound, so their "
             "disagreement is not yet resolved from the integration rule; "
             "refining the circuit is what would separate them"
             if quadrature > disagreement else
             "\n  the two curves differ by more than that bound, so their "
             "disagreement is about the physics and not the integration"))

    # The circuit returns to Gamma, so the last point repeats the first: the
    # direct energies must agree to the SCF's own reproducibility and the
    # integral must come back to zero.
    if len(rec) == npoints and np.allclose(q[-1], q[0], atol=1e-12):
        print(f"circuit closure: direct {direct[-1]:+.4f} meV, "
              f"integrated {integrated[-1]:+.4f} meV -- both are zero for an "
              f"exact calculation, and the first is also this run's "
              f"reproducibility across two independent jobs")

    resid = [r["frozen_energy_residual"] for r in rec
             if r.get("frozen_energy_residual") is not None]
    if resid:
        print(f"frozen-state identity max |E_frozen - E_scf| = "
              f"{max(abs(x) for x in resid):.2e} Ry -- the check that the "
              f"gradient differentiates the energy the SCF converged")

    imin = int(np.nanargmin(direct))
    # The energies are referenced to the *first point of the path*, which is
    # only Gamma when the path starts there. Naming it Gamma regardless is
    # wrong by E(Gamma) - E(start) -- 27 meV of 58 on the M-K-G-M circuit --
    # and it is the same hardcoded assumption the title and the y-axis label
    # carried, so all three are derived from ``letters`` now.
    reference = _greek(letters[0])
    print(f"\nminimum of the direct curve at q = {np.round(q[imin], 4)} "
          f"({label[imin] or 'off-symmetry'}), {direct[imin]:.3f} meV below "
          f"{reference}" + ("  -- Gamma itself: a ferromagnet"
                            if label[imin] == "G" else
                            "  -- a spiral ground state"))
    print(f"moment across the circuit: {moment.min():.3f} to {moment.max():.3f} "
          f"muB (a point where it collapses is a failed seed, not a physical "
          f"result)")

    if args.plot:
        _plot(args, s, direct, integrated, dEds, moment, s_all, runner, letters, nodes)


def _spline_integral(s, left, right, tangents, fallback):
    """Integrate ``dE/ds`` leg by leg with a cubic spline, trapezoid if it cannot.

    A leg is a maximal run of intervals sharing a tangent. Inside one, ``dE/ds``
    is smooth and single valued -- ``left[j]`` and ``right[j-1]`` are the same
    number -- so a spline through the interval endpoints integrates it to fourth
    order. Fewer than four points on a leg, or no SciPy, and that leg falls back
    to the trapezoid, which is right and merely coarser.
    """
    try:
        from scipy.interpolate import CubicSpline
    except ImportError:
        return fallback

    out = np.array(fallback, dtype=float)
    start = 0
    for j in range(1, len(tangents) + 1):
        if j < len(tangents) and np.allclose(tangents[j], tangents[start]):
            continue
        # intervals [start, j) share a tangent, so points [start, j] do
        nodes = np.arange(start, j + 1)
        if len(nodes) >= 4:
            values = np.concatenate([left[start:j], [right[j - 1]]])
            spline = CubicSpline(s[nodes], values)
            anti = spline.antiderivative()
            out[start:j] = np.diff(anti(s[nodes]))
        start = j
    return out


def _fraction_along(a, b, q, tol=1e-9):
    d = b - a
    denom = float(d @ d)
    t = float((q - a) @ d) / denom if denom else 0.0
    if -tol <= t <= 1.0 + tol and np.allclose(a + t * d, q, atol=1e-7):
        return t
    return None


def _greek(letter):
    """``G`` is Gamma. One place, because it was spelled out in three."""
    return "Gamma" if letter == "G" else letter


def _labels(runner, path, q):
    out = []
    for qi in q:
        hit = ""
        for name, corner in runner.CORNERS.items():
            if np.allclose(qi, corner, atol=1e-9):
                hit = "G" if name == "G" else name
        out.append(hit)
    return out


def _plot(args, s, direct, integrated, dEds, moment, s_all, runner, letters, nodes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ticks, names = [0.0], [letters[0].replace("G", r"$\Gamma$")]
    walk = 0.0
    for a, b, c in zip(nodes, nodes[1:], letters[1:]):
        walk += runner._hex_length(b - a)
        ticks.append(walk)
        names.append(c.replace("G", r"$\Gamma$"))

    fig, axes = plt.subplots(3, 1, figsize=(6.4, 7.6), sharex=True,
                             height_ratios=[2.2, 1, 1])
    ax, bx, cx = axes
    ax.plot(s, direct, "o-", label="SCF energies")
    ax.plot(s, integrated, "s--", label=r"$\int \mathrm{d}E/\mathrm{d}q\,\mathrm{d}s$")
    # Both of these were hardcoded to the Gamma-K-M-Gamma circuit this script
    # no longer runs by default, so they went stale the moment the sweep was
    # reversed while the tick labels -- which come from the data -- stayed
    # right. Derived from ``letters`` now, like the ticks.
    def maths(letter):
        return r"\Gamma" if letter == "G" else letter

    ax.set_ylabel(rf"$E(q) - E({maths(letters[0])})$  [meV]")
    ax.legend(frameon=False)
    ax.set_title(r"NiI$_2$ monolayer: spin-spiral energy along $"
                 + r"\!-\!".join(maths(x) for x in letters) + r"$")
    bx.plot(s, dEds, "o-", color="C2")
    bx.axhline(0.0, lw=0.6, color="k")
    bx.set_ylabel(r"$\mathrm{d}E/\mathrm{d}s$  [Ry]")
    cx.plot(s, moment, "o-", color="C3")
    cx.set_ylabel(r"$|m|$  [$\mu_B$]")
    cx.set_xlabel("spiral wavevector")
    for a in axes:
        a.grid(alpha=0.3)
        for t in ticks[1:-1]:
            a.axvline(t, lw=0.6, color="0.5")
    cx.set_xticks(ticks)
    cx.set_xticklabels(names)
    fig.tight_layout()
    fig.savefig(args.plot, dpi=150)
    print(f"\nwrote {args.plot}")


if __name__ == "__main__":
    main()
