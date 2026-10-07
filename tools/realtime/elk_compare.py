"""The real-time current against Elk's two silicon examples, Si-dielectric and Si-ramp.

Elk's ``examples/TDDFT-time-evolution/Si-dielectric`` ships ``EPSILON_TDRT_11.OUT``
(task 481 after a kick of ``A = 0.1``) and ``Si-ramp`` ships ``JTOT_TD.OUT`` (the
current under ``A = 0.001 t^2``). The inputs at matched settings are in
``tests/data/elk/si_rt/`` with a README saying what is and is not matched.

At a frozen potential the k-points are independent and the current is a
weighted sum over them, so a run can be split across processes exactly: each
process propagates every ``nparts``-th point of the field's little group's
wedge with its own weights, and ``combine`` adds the shares and averages the
sum over the group as a polar vector, which is what
:func:`defumat.workflows.realtime.run_realtime` does in one process.

    python3 tools/realtime/elk_compare.py scf INPUT density.npz
    python3 tools/realtime/elk_compare.py run INPUT density.npz CASE PART NPARTS share.npz \
        [--dt 0.2] [--duration 4400]
    python3 tools/realtime/elk_compare.py combine current.npz share0.npz share1.npz ...
    python3 tools/realtime/elk_compare.py dielectric current.npz EPSILON_TDRT_11.OUT \
        [EPSILON_11.OUT ...] > dielectric.json
    python3 tools/realtime/elk_compare.py ramp current.npz JTOT_TD.OUT > ramp.json
    python3 tools/realtime/elk_compare.py check INPUT CASE current.npz

``CASE`` is ``kick`` (a step of ``kappa = 0.1/c`` along x, the Si-dielectric
pulse; Elk's Gaussian of FWHM 10000 lets ``A`` fall by 1.8 per cent over its 800
units, which is a quasi-static field of 4e-8 a.u. and is left out) or ``ramp``
(``kappa = 0.001 t^2 / c`` along x, Elk's ``ramp`` card with the same
coefficients). ``check`` runs ``Calculator.get_realtime`` in one process on
the same input and compares it with a combined current, which is the test that
the split is the public method.

**Sign and units.** Elk's ``JTOT_TD.OUT`` is ``sum_k w_k occ <p> - (A/c)(N - N_s)``
(``jtotk.f90``, ``timestep.f90``), the velocity summed over the cell with the
coupling ``p - A/c`` (``genhmlt.f90`` adds ``-(1/c) A.p``), so its electrons
couple as a charge of +1, while here the coupling is ``p + kappa`` and the
current is ``-(1/Omega) sum w <v>``, a charge of -1. With the same ``A`` on both
sides the two Hamiltonians are those of opposite fields, and for a crystal with
an inversion centre ``J[-A] = -J[A]`` exactly on an unshifted grid, so the two
signs cancel: ``J_here(t) = JTOT_TD(t) / Omega``, in ``e E_h/(hbar a_0^2)``. That
is the same statement as Elk's ``dielectric_tdrt.f90`` dividing ``JTOT`` by
``omega`` and calling it ``J``. ``N_s`` is Elk's static charge (``rhostatic.f90``),
which makes a static ``A`` give no current in its truncated band basis: on the
Si-dielectric input it is 21.46 of 28 electrons, so Elk's diamagnetic term
counts 6.54 electrons where the plane-wave current counts all eight through
``d^2 H / dk^2``.

**Elk's time labels.** ``timestep.f90`` builds ``H`` with ``A(t_i)``, propagates
the states to ``t_(i+1)``, and then computes the paramagnetic current from the
new states and the diamagnetic one from ``A(t_i)``; ``writetddft.f90`` writes the
sum at ``t_i``. So Elk's current at a listed time is the paramagnetic current one
step later. For a kick ``A`` is constant afterwards and the whole current is
shifted by one step; ``dielectric`` reproduces task 481 with that shift and
without it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PSEUDO = REPO / "tests" / "data" / "pseudo"
HARTREE_EV = 27.211386245988
#: Elk's speed of light, ``solsc`` in ``modmain.f90``.
ELK_C = 137.035999084


def _pulse(case: str):
    from defumat.realtime.pulse import Kick, Ramp
    from defumat.units import C_AU

    if case == "kick":
        return Kick(strength=0.1 / C_AU, direction=(1.0, 0.0, 0.0))
    if case == "ramp":
        return Ramp(amplitude=1.0 / C_AU, coefficients=(0.0, 0.001, 0.0, 0.0),
                    direction=(1.0, 0.0, 0.0))
    raise ValueError(f"unknown case {case!r}")


def _calculator(path):
    from defumat import Calculator

    return Calculator.from_file(path, pseudo_dir=PSEUDO, announce=False)


def cmd_scf(args):
    calculator = _calculator(args.input)
    start = time.perf_counter()
    result = calculator.get_scf()
    np.savez(args.out, density=np.asarray(result.density),
             energy=float(result.total_energy), seconds=time.perf_counter() - start)
    print(json.dumps({"energy_ry": float(result.total_energy),
                      "seconds": time.perf_counter() - start}), flush=True)


def cmd_run(args):
    import equinox as eqx

    from defumat.realtime.propagate import propagate
    from defumat.workflows.realtime import _kset, _occupied_states

    calculator = _calculator(args.input)
    system, pseudos = calculator.system, calculator.pseudos
    density = np.load(args.density)["density"]
    pulse = _pulse(args.case)
    grid = tuple(int(n) for n in system.kpoints.grid)
    kset, rotations = _kset(system, pulse, None, grid, True)
    rows = np.arange(kset.nk)[args.part::args.nparts]
    coords = np.asarray(kset.coords)[rows]
    weights = np.asarray(kset.weights)[rows]
    share = eqx.tree_at(lambda k: (k.coords, k.weights), kset,
                        (kset.precision.as_real(coords), kset.precision.as_real(weights)))
    start = time.perf_counter()
    calc, states, wg, v_scf = _occupied_states(
        system, pseudos, density, share, nbnd=None, conv_thr=1e-10, k_batch="default",
        calculation=None)
    prepared = time.perf_counter() - start
    print(json.dumps({"part": args.part, "nk": int(len(rows)), "of": int(kset.nk),
                      "nbnd": int(states.shape[1]), "npwx": int(states.shape[2]),
                      "prepare_seconds": prepared}), flush=True)
    start = time.perf_counter()
    result = propagate(calc, states, wg, v_scf, pulse, dt=args.dt, duration=args.duration,
                       start=0.0, block_steps=args.block_steps, symmetrise=None)
    seconds = time.perf_counter() - start
    np.savez(args.out, times=result.times, kappa=result.kappa, efield=result.efield,
             current=result.current, energy_times=result.energy_times,
             energy=result.energy, norm_drift=result.norm_drift, excited=result.excited,
             volume=result.volume, nelec=result.nelec, dt=result.dt,
             effective_cutoff=result.effective_cutoff,
             spectrum=np.asarray(result.spectrum), step_radius=result.step_radius,
             rotations=np.zeros((0, 3, 3)) if rotations is None else np.asarray(rotations),
             nk=len(rows), nk_wedge=kset.nk, seconds=seconds)
    print(json.dumps({"part": args.part, "propagate_seconds": seconds,
                      "steps": int(len(result.times) - 1),
                      "ms_per_step_per_k": 1e3 * seconds / (len(result.times) - 1) / len(rows),
                      "norm_drift": result.norm_drift,
                      "effective_cutoff_ry": result.effective_cutoff,
                      "spectrum_ry": list(map(float, result.spectrum)),
                      "step_radius": float(result.step_radius)}), flush=True)


def cmd_combine(args):
    shares = [np.load(p) for p in args.shares]
    first = shares[0]
    current = sum(s["current"] for s in shares)
    energy = sum(s["energy"] for s in shares)
    rotations = first["rotations"]
    if len(rotations):
        current = np.einsum("sab,tb->ta", rotations, current) / len(rotations)
    nk = int(sum(int(s["nk"]) for s in shares))
    if nk != int(first["nk_wedge"]):
        raise ValueError(f"the shares hold {nk} of the wedge's {int(first['nk_wedge'])} points")
    np.savez(args.out, times=first["times"], kappa=first["kappa"], efield=first["efield"],
             current=current, energy_times=first["energy_times"], energy=energy,
             norm_drift=max(float(s["norm_drift"]) for s in shares),
             excited=sum(float(s["excited"]) for s in shares), volume=first["volume"],
             nelec=sum(float(s["nelec"]) for s in shares), dt=first["dt"],
             effective_cutoff=first["effective_cutoff"], nk=nk,
             symmetry_operations=len(rotations),
             seconds=sum(float(s["seconds"]) for s in shares))
    work = -float(first["volume"]) * np.trapezoid(
        np.sum(current * first["efield"], axis=-1), first["times"])
    print(json.dumps({"nk": nk, "operations": int(len(rotations)),
                      "nelec": sum(float(s["nelec"]) for s in shares),
                      "norm_drift": max(float(s["norm_drift"]) for s in shares),
                      "excited": sum(float(s["excited"]) for s in shares),
                      "energy_gained_ha": float(energy[-1] - energy[0]),
                      "work_ha": work,
                      "seconds_summed": sum(float(s["seconds"]) for s in shares)}), flush=True)


# --- Elk's task 481, transcribed (dielectric_tdrt.f90, zftft.f90, zlrzncnv.f90) ---

def _simpson_weights(n, dt):
    weights = np.ones(n)
    if n % 2 == 1:
        weights[1:-1:2], weights[2:-1:2] = 4.0, 2.0
        return weights * dt / 3.0
    head = _simpson_weights(n - 1, dt)
    return np.concatenate([head, [0.0]]) + np.concatenate([np.zeros(n - 2), [0.5, 0.5]]) * dt


def lorentzian_filter(w, values, width):
    """``zlrzncnv``: the convolution with a Lorentzian of half width ``width``, Elk's rectangle rule."""
    dw = np.diff(w)
    kernel = dw[None, :] / ((w[None, :-1] - w[:, None]) ** 2 + width**2)
    return (kernel @ values[:-1]) * width / math.pi


def elk_task_481(times, current, kappa0, w, swidth, jtconst0=True):
    """``eps_xx(w)`` from ``J_x(t)`` on ``times`` after a kick of ``kappa0``, as Elk computes it.

    The spline weights of ``wsplint`` are replaced by Simpson's on the uniform
    grid; both are exact to fourth order in the step.
    """
    times = np.asarray(times, dtype=float)
    j = np.asarray(current, dtype=float).copy()
    wt = _simpson_weights(len(times), times[1] - times[0])
    if jtconst0:
        j = j - np.dot(wt, j) / times[-1]
    jw = (np.exp(1j * np.outer(w, times)) * wt[None, :]) @ j
    jw = lorentzian_filter(w, jw, swidth)
    ew = -kappa0
    z1 = jw / ew
    eps = 1.0 + 4.0 * math.pi * 1j * z1 / (w + 1j * swidth)
    return lorentzian_filter(w, eps, 2.0 * swidth)


def read_elk_epsilon(path):
    """``(w in Ha, eps)`` from an Elk ``EPSILON_*.OUT``: the real part, a blank line, the imaginary part."""
    blocks, block = [], []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            if block:
                blocks.append(np.array(block))
                block = []
            continue
        block.append([float(x) for x in line.split()])
    if block:
        blocks.append(np.array(block))
    real, imag = blocks[0], blocks[1]
    return real[:, 0], real[:, 1] + 1j * imag[:, 1]


def peaks(w_ev, values, lo=2.0, hi=6.0, prominence=0.05):
    """The local maxima of ``values`` between ``lo`` and ``hi`` eV, largest first, with their heights."""
    inside = (w_ev >= lo) & (w_ev <= hi)
    idx = np.flatnonzero(inside)
    found = []
    for i in idx[1:-1]:
        if values[i] > values[i - 1] and values[i] >= values[i + 1]:
            found.append((float(w_ev[i]), float(values[i])))
    top = max((v for _, v in found), default=0.0)
    return [p for p in found if p[1] >= prominence * top]


def _summary(w_ev, eps, label):
    below_gap = (w_ev >= 0.5) & (w_ev <= 1.5)
    return {"label": label,
            "eps1_mean_0.5_to_1.5_eV": float(np.mean(eps.real[below_gap])),
            "eps1_at_1.0_eV": float(np.interp(1.0, w_ev, eps.real)),
            "eps2_peaks_eV": peaks(w_ev, eps.imag)[:6],
            "eps2_max_eV": float(w_ev[(w_ev > 2) & (w_ev < 6)][np.argmax(eps.imag[(w_ev > 2) & (w_ev < 6)])]),
            "eps2_max": float(np.max(eps.imag[(w_ev > 2) & (w_ev < 6)]))}


def cmd_dielectric(args):
    from defumat.realtime.spectra import conductivity_from_kick
    from defumat.units import C_AU

    data = np.load(args.current)
    times, jx = data["times"], data["current"][:, 0]
    kappa0 = 0.1 / C_AU
    out = {"kappa0": kappa0, "dt": float(data["dt"]), "T_run": float(times[-1]),
           "nk": int(data["nk"]), "norm_drift": float(data["norm_drift"])}
    w_elk, eps_elk = read_elk_epsilon(args.tdrt)
    swidth = args.swidth
    T = 800.0
    nt = int(round(T / float(data["dt"]))) + 1
    # Elk's grid: ntimes = tstime/dtimes + 1 points, J read for the first
    # ntimes - 1 and the last repeated (readjtot.f90)
    rows = {}
    for label, shift in (("elk_481_replica_elk_labels", 1), ("elk_481_replica_true_times", 0)):
        j = np.empty(nt)
        j[:nt - 1] = jx[shift:shift + nt - 1]
        j[nt - 1] = j[nt - 2]
        eps = elk_task_481(times[:nt], j, kappa0, w_elk, swidth)
        rows[label] = _summary(w_elk * HARTREE_EV, eps, label)
        rows[label]["max_abs_minus_elk_tdrt"] = float(np.max(np.abs(eps - eps_elk)))
        np.savetxt(Path(args.current).with_suffix(f".{label}.dat"),
                   np.column_stack([w_elk, eps.real, eps.imag]))
    rows["elk_tdrt_shipped"] = _summary(w_elk * HARTREE_EV, eps_elk, "elk_tdrt_shipped")
    for extra in args.reference:
        w_ref, eps_ref = read_elk_epsilon(extra)
        rows[Path(extra).name] = _summary(w_ref * HARTREE_EV, eps_ref, extra)
    # the Lorentzian window of the code's own route, exp(-eta t), over the whole run
    w = np.linspace(0.0, 0.5, 401)
    for eta in args.eta:
        if eta * times[-1] < 15.0:
            continue
        for static in (False, True):
            response = conductivity_from_kick(times, data["current"], kappa0, (1.0, 0.0, 0.0),
                                              w, eta, subtract_static=static)
            label = f"window_eta_{eta}_subtract_{static}"
            eps = response.epsilon[:, 0]
            rows[label] = _summary(w * HARTREE_EV, eps, label)
            rows[label]["eps1_at_0"] = float(eps[0].real)
            np.savetxt(Path(args.current).with_suffix(f".{label}.dat"),
                       np.column_stack([w, eps.real, eps.imag]))
    # the mesh's band curvature, from the current's long-time average (spectra.py)
    tail = times > 0.5 * times[-1]
    out["mean_J_second_half"] = float(np.mean(jx[tail]))
    out["D_over_Omega_from_mean"] = float(-np.mean(jx[tail]) / kappa0)
    out["rows"] = rows
    print(json.dumps(out, indent=1))


def read_elk_jtot(path):
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1:4]


def cmd_ramp(args):
    data = np.load(args.current)
    times, current = data["times"], data["current"]
    volume = float(data["volume"])
    t_elk, j_elk = read_elk_jtot(args.jtot)
    j_elk = j_elk / volume
    here = np.interp(t_elk, times, current[:, 0])
    # Elk's paramagnetic current at a listed time is one step later (see the docstring)
    dt_elk = float(t_elk[1] - t_elk[0])
    here_next = np.interp(t_elk + dt_elk, times, current[:, 0])
    scale = float(np.max(np.abs(j_elk[:, 0])))
    rows = []
    for t in (25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 475, 499.5):
        i = int(np.argmin(np.abs(t_elk - t)))
        rows.append({"t": float(t_elk[i]),
                     "kappa": 0.001 * t_elk[i] ** 2 / ELK_C,
                     "E_au": -0.002 * t_elk[i] / ELK_C,
                     "J_here": float(here[i]), "J_here_next_step": float(here_next[i]),
                     "J_elk_over_Omega": float(j_elk[i, 0])})
    peak_here = float(times[np.argmax(current[:, 0])])
    peak_elk = float(t_elk[np.argmax(j_elk[:, 0])])
    zero_here = float(times[np.flatnonzero((current[:-1, 0] > 0) & (current[1:, 0] <= 0)
                                           & (times[:-1] > 100))[0]]) \
        if np.any((current[:-1, 0] > 0) & (current[1:, 0] <= 0) & (times[:-1] > 100)) else None
    zero_elk = float(t_elk[np.flatnonzero((j_elk[:-1, 0] > 0) & (j_elk[1:, 0] <= 0)
                                          & (t_elk[:-1] > 100))[0]]) \
        if np.any((j_elk[:-1, 0] > 0) & (j_elk[1:, 0] <= 0) & (t_elk[:-1] > 100)) else None
    windows = {}
    for lo, hi in ((0, 100), (100, 200), (200, 300), (300, 400), (400, 500)):
        sel = (t_elk >= lo) & (t_elk < hi)
        windows[f"{lo}-{hi}"] = {
            "max_abs_diff_over_scale": float(np.max(np.abs(here[sel] - j_elk[sel, 0])) / scale),
            "rms_ratio_here_over_elk": float(np.sqrt(np.mean(here[sel] ** 2)
                                                     / np.mean(j_elk[sel, 0] ** 2)))}
    print(json.dumps({"volume": volume, "scale_elk_over_Omega": scale,
                      "elk_J_at_t0_over_Omega": float(j_elk[0, 0]),
                      "peak_time_here": peak_here, "peak_time_elk": peak_elk,
                      "peak_here": float(np.max(current[:, 0])),
                      "peak_elk_over_Omega": float(np.max(j_elk[:, 0])),
                      "zero_crossing_here": zero_here, "zero_crossing_elk": zero_elk,
                      "transverse_max_here": float(np.max(np.abs(current[:, 1:]))),
                      "norm_drift": float(data["norm_drift"]),
                      "excited": float(data["excited"]),
                      "effective_cutoff_ry": float(data["effective_cutoff"]),
                      "windows": windows, "rows": rows}, indent=1))
    np.savetxt(Path(args.current).with_suffix(".vs_elk.dat"),
               np.column_stack([t_elk, here, here_next, j_elk[:, 0]]))


def cmd_check(args):
    """``Calculator.get_realtime`` in one process against a combined current."""
    calculator = _calculator(args.input)
    calculator.get_scf()
    data = np.load(args.current)
    pulse = _pulse(args.case)
    result = calculator.get_realtime(pulse, dt=float(data["dt"]),
                                     duration=float(data["times"][-1]), start=0.0)
    diff = np.max(np.abs(result.current - data["current"]))
    print(json.dumps({"max_abs_diff": float(diff),
                      "scale": float(np.max(np.abs(data["current"]))),
                      "operations": int(result.symmetry_operations)}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("scf")
    p.add_argument("input")
    p.add_argument("out")
    p.set_defaults(func=cmd_scf)
    p = sub.add_parser("run")
    p.add_argument("input")
    p.add_argument("density")
    p.add_argument("case", choices=("kick", "ramp"))
    p.add_argument("part", type=int)
    p.add_argument("nparts", type=int)
    p.add_argument("out")
    p.add_argument("--dt", type=float, default=0.2)
    p.add_argument("--duration", type=float, default=4400.0)
    p.add_argument("--block-steps", type=int, default=500)
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("combine")
    p.add_argument("out")
    p.add_argument("shares", nargs="+")
    p.set_defaults(func=cmd_combine)
    p = sub.add_parser("dielectric")
    p.add_argument("current")
    p.add_argument("tdrt")
    p.add_argument("reference", nargs="*")
    p.add_argument("--swidth", type=float, default=0.001)
    p.add_argument("--eta", type=float, nargs="*", default=[0.005, 0.01])
    p.set_defaults(func=cmd_dielectric)
    p = sub.add_parser("ramp")
    p.add_argument("current")
    p.add_argument("jtot")
    p.set_defaults(func=cmd_ramp)
    p = sub.add_parser("check")
    p.add_argument("input")
    p.add_argument("case", choices=("kick", "ramp"))
    p.add_argument("current")
    p.set_defaults(func=cmd_check)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
