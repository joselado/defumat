"""1H-NbSe2: the Fermi surface, and which of it an electron can tunnel out of.

The physics in one sentence: NbSe2's Fermi surface is a small pocket at the zone
centre and two large ones at the zone corners, the corner pockets carry most of
the density of states, and **an electron tunnelling vertically out of the layer
comes almost entirely from the zone centre anyway** -- because a state at large
in-plane momentum has a faster-decaying tail in the vacuum,

    psi(z) ~ exp( -sqrt(kappa_0^2 + |k_par|^2) z ),

and 3.5 A of vacuum turns that into an order of magnitude.

Three columns come out of one calculation, and the physics is in their ratios:

* ``bare``            -- the plain Fermi surface, ``sum_n delta(E_F - e_nk)``;
* ``tersoff_hamann``  -- the same states weighted by how much of each sits on
                         the tip plane, which is what an STM would see;
* ``weight``          -- the transmission all the way through, tip plane to
                         substrate plane, which is what a vertical junction
                         measures.

The reference is Elk, through elkpy's task 9007, on the *same* structure and the
same functional. Nothing about the two implementations is shared: Elk is
all-electron LAPW and this is a plane-wave pseudopotential code, and neither
``pw.x`` nor stock Elk computes this quantity at all. **Absolute total energies
are not comparable** and are not compared; the Fermi-surface weights, their
pocket shares, and the vacuum decay constants are.

Usage:  python3 calculations/nbse2-fermi-surface/tunnelling.py [grid]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from defumat import Calculator
from defumat.scf.driver import SCFResult
from defumat.transport.momentum import (
    decay_constants,
    decay_identity,
    pocket_mask,
)
from defumat.workflows.transport import run_momentum_transport

HERE = Path(__file__).parent
PSEUDO = HERE.parents[1] / "tests" / "data" / "pseudo"
STATE = HERE / "state"

BOHR = 0.529177210903          # A
HARTREE = 2.0                  # Ry
A_LATTICE = 3.442              # A, the in-plane lattice constant
C_HEIGHT = 18.0                # A, the cell height
SE_OFFSET = 1.68036            # A, the Se planes above and below the Nb

#: Elk's ``swidth`` for the ground state was 0.001 Ha; the *junction* broadening
#: is a separate number and elkpy's headline runs use 0.003 Ha. A Gaussian is
#: used on both sides by agreement: at the same width a Fermi-Dirac delta is
#: 2.1x wider in full width and still carries 1.3e-3 of its peak eight widths
#: out, which is where elkpy's export window ends -- so the two codes would
#: differ there for a truncation reason rather than a physical one.
ETA_RY = 0.003 * HARTREE

#: elkpy's numbers, so the table this prints has its reference beside it.
ELK = {
    "fermidos_states_per_ha": 64.393,        # 12x12x1, Fermi-Dirac at 0.001 Ha
    "bare_share_K": 0.698,                   # 54x54x1, eta = 0.003 Ha, 3.5 A
    "weight_share_K": 0.084,
    "reweighting_35": 25.32,
    "reweighting_25": 6.08,
    "kappa_gamma_per_a": 0.9551,             # 12x12x1, eta = 0.004 Ha, tip only
    "kappa_k_per_a": 1.5403,
}


def heights_in_crystal(distances_angstrom):
    """Tip-plane crystal coordinates ``distances`` A above the outer Se."""
    top = 0.5 + SE_OFFSET / C_HEIGHT
    return np.asarray([top + d / C_HEIGHT for d in distances_angstrom])


def exit_in_crystal(distance_angstrom):
    """The substrate plane's crystal coordinate, below the lower Se."""
    return 0.5 - SE_OFFSET / C_HEIGHT - distance_angstrom / C_HEIGHT


def main(grid=(24, 24, 1)) -> None:
    calculator = Calculator.from_file(HERE / "nbse2.in", pseudo_dir=PSEUDO)
    system, pseudos = calculator.system, calculator.pseudos

    # The ground state comes off disk rather than being recomputed: it is half
    # an hour, and every number below is post-processing of the *same* density
    # and the same wavefunctions. ``Calculator`` is the front door everywhere a
    # calculation starts from an input file; a state that is already converged
    # is what the functional entry point is for.
    checkpoint = STATE / "nbse2.state"
    if checkpoint.exists():
        scf = SCFResult.load(checkpoint, system=system)
        print(f"resumed the ground state from {checkpoint}")
    else:
        scf = calculator.get_scf()
    print(f"E_F = {scf.fermi_energy:.8f} Ry, {scf.iterations} iterations")

    # 1.5 A is the closest elkpy sampled and 4.5 A the furthest that is clean
    # here: past that the tip plane is closer to the *next* cell's slab than the
    # exponential it is meant to be measuring, which is this basis's analogue of
    # the plane-wave floor elkpy reports at the same place.
    distances = np.array([1.5, 2.5, 3.5, 4.5])
    tips = heights_in_crystal(distances)
    start = time.time()
    run = run_momentum_transport(
        system, pseudos, scf,
        exit_height=exit_in_crystal(3.5),
        height=tips,
        grid=tuple(grid),
        broadening=ETA_RY,
        smearing="gaussian",
        # **The delta is centred on the SCF's own Fermi level, not the dense
        # grid's.** Elk's task 9007 has no `occupy` call: it takes `efermi`
        # from `readstate` and evaluates its window once before the k-loop, so
        # its eigenvalues are dense and its chemical potential is the coarse
        # one. Letting the dense grid re-find its own would centre the two
        # codes' deltas at different energies and move every share.
        energies=float(scf.fermi_energy),
    )
    print(f"{np.prod(grid)} k-points, {len(distances)} heights, "
          f"{time.time() - start:.0f} s")

    print(f"  E_F on the {grid[0]}x{grid[1]} grid: {run.fermi_energy:.8f} Ry, "
          f"{(run.fermi_energy - scf.fermi_energy) * 1e3:+.3f} mRy from the "
          "SCF's -- which is the error freezing the level leaves in any "
          "absolute total,\n  and is why the shares and the reweighting are "
          "what this reports")

    corners = pocket_mask(run.kcartesian, system.cell)
    at = 2  # the 3.5 A row, which is elkpy's headline height

    # -- the columns, and their pocket shares ---------------------------------
    shares = run.shares(corners, index=at)
    record = {
        "grid": list(map(int, grid)),
        "eta_ry": ETA_RY,
        "smearing": run.smearing,
        "fermi_energy_ry": float(run.fermi_energy),
        "distances_angstrom": distances.tolist(),
        "corner_share_of_the_zone": float(corners.mean()),
        "shares_at_3.5A": shares,
        "reweighting_3.5A": run.reweighting(corners, index=at),
        "reweighting_2.5A": run.reweighting(corners, index=1),
        "least_eigenvalue": run.least_eigenvalue,
        "hermiticity": run.hermiticity,
    }

    # -- the vacuum decay, at Gamma and at K, and the identity ----------------
    #
    # Elk's sweep moves the tip plane only, leaving the substrate a whole cell
    # (``exit_region="cell"``), which is exactly the ``tersoff_hamann`` column
    # here -- so that is the column fitted, and the exponent is 2 kappa.
    crystal = np.round(run.kpoints, 8) % 1.0
    gamma = int(np.argmin(np.linalg.norm(crystal, axis=1)))
    k_corner = int(np.argmin(
        np.linalg.norm(crystal - np.array([1 / 3, 1 / 3, 0.0]), axis=1)))
    tersoff = run.sweep("tersoff_hamann")
    in_bohr = distances / BOHR
    # **Fitted over 1.5-3.5 A, which is elkpy's range and not the whole sweep.**
    # Their successive log-ratios (1.793, 1.996, 1.987 per A) say the first
    # interval is not yet asymptotic -- at 1.5 A the wavefunction still has
    # atomic structure -- so a fit that includes 4.5 A here and not there would
    # not be like for like. The ratios are reported so the same judgement can be
    # made of these numbers rather than taken on trust.
    window = slice(0, 3)
    kappa = {
        "gamma": float(decay_constants(in_bohr[window], tersoff[window, gamma])),
        "k": float(decay_constants(in_bohr[window], tersoff[window, k_corner])),
    }
    record["kappa_per_angstrom"] = {
        name: value / BOHR for name, value in kappa.items()}
    record["kappa_fit_range_angstrom"] = distances[window].tolist()
    # One kappa per interval, which is how a fit is told from an artefact: a
    # tail that is a tail gives the same number twice, a floor does not.
    step = np.diff(distances)
    record["log_ratios_per_angstrom"] = {
        name: (-np.diff(np.log(tersoff[:, i])) / step / 2.0).tolist()
        for name, i in (("gamma", gamma), ("k", k_corner))}
    record["interference_share_at_3.5A"] = float(
        np.abs(run.column("weight", at) - run.column("incoherent", at)).sum()
        / run.column("weight", at).sum())
    record["identity"] = decay_identity(
        kappa["k"], kappa["gamma"],
        run.kcartesian[k_corner], run.kcartesian[gamma])
    record["tersoff_at_gamma"] = tersoff[:, gamma].tolist()
    record["tersoff_at_k"] = tersoff[:, k_corner].tolist()
    record["weight_at_gamma"] = run.sweep("weight")[:, gamma].tolist()
    record["weight_at_k"] = run.sweep("weight")[:, k_corner].tolist()

    out = HERE / f"tunnelling-{grid[0]}x{grid[1]}.json"
    out.write_text(json.dumps(record, indent=1))
    np.savez_compressed(
        HERE / f"tunnelling-{grid[0]}x{grid[1]}.npz",
        kpoints=run.kpoints, kcartesian=run.kcartesian,
        kweights=run.kweights, distances=distances, corners=corners,
        **{name: getattr(run, name) for name in
           ("weight", "tersoff_hamann", "bare", "incoherent")},
    )
    _report(record, run, corners)
    print(f"  -> {out}")


def _report(record, run, corners) -> None:
    print()
    print(f"the zone-corner region is {record['corner_share_of_the_zone']:.3f} "
          f"of the zone by count (exactly 2/3 by area on a hexagon)")
    print()
    print("  share of each column carried by the K pockets, tip 3.5 A out")
    print(f"    bare            {record['shares_at_3.5A']['bare']:7.1%}"
          f"     (Elk {ELK['bare_share_K']:.1%})")
    print(f"    tersoff-hamann  {record['shares_at_3.5A']['tersoff_hamann']:7.1%}")
    print(f"    tunnelling      {record['shares_at_3.5A']['weight']:7.1%}"
          f"     (Elk {ELK['weight_share_K']:.1%})")
    print(f"    suppression     {record['shares_at_3.5A']['suppression']:7.2f}x")
    print()
    print(f"  reweighting (W_G/W_K)/(D_G/D_K)")
    print(f"    3.5 A   {record['reweighting_3.5A']:7.2f}"
          f"     (Elk {ELK['reweighting_35']:.2f})")
    print(f"    2.5 A   {record['reweighting_2.5A']:7.2f}"
          f"     (Elk {ELK['reweighting_25']:.2f})")
    print()
    kappa = record["kappa_per_angstrom"]
    print("  vacuum decay of the tip-plane weight, 1/A "
          f"(fitted over {record['kappa_fit_range_angstrom']} A)")
    for name, ratios in record["log_ratios_per_angstrom"].items():
        print(f"    per-interval kappa({name}): "
              + "  ".join(f"{r / BOHR:.3f}" for r in ratios))
    print(f"    kappa(Gamma)  {kappa['gamma']:.4f}   (Elk {ELK['kappa_gamma_per_a']:.4f})")
    print(f"    kappa(K)      {kappa['k']:.4f}   (Elk {ELK['kappa_k_per_a']:.4f})")
    identity = record["identity"]
    scale = 1.0 / BOHR**2
    print(f"    kappa_K^2 - kappa_G^2 = {identity['measured'] * scale:.4f} 1/A^2 "
          f"against |K|^2 = {identity['expected'] * scale:.4f}, "
          f"{identity['relative_residual']:.1%} low")
    print(f"    (a zone-centre pocket of radius "
          f"{identity['pocket_radius_that_would_close_it'] / BOHR:.3f} 1/A would "
          "close it -- a hypothetical, not a measurement: refitting over "
          "different\n     ranges moves the residual across zero, so read the "
          "per-interval kappas above before reading anything into the sign)")
    print()
    print(f"  interference (coherent minus incoherent) is "
          f"{record['interference_share_at_3.5A']:.2%} of the tunnelling column")
    print(f"  diagnostics: least Gram eigenvalue {run.least_eigenvalue:.3e} "
          f"(must not be negative), hermiticity {run.hermiticity:.1e}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    main((n, n, 1))
