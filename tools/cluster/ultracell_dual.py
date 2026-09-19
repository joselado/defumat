"""An augmented ultracell at the dual its dataset was generated for.

``PLAN.md`` P88 stage 9. The double grid was refused until 2026-09-20, so every
augmented number the phase carries was taken at ``ecutrho = 4 ecutwfc``, where
the augmentation charge sits on the wavefunction grid. The refusal is lifted and
what it protected turned out to be unreachable, so what is left is the
measurement nobody could take: the same ladder against a real supercell at 8 and
12 times ``ecutwfc``, in both spin regimes, on both silicon datasets and on the
one fully relativistic cell this package has.

One case per array task, one JSON per case. Each case runs

* the **null** -- nothing applied, so the ultracell must reproduce the unit
  cell's own SCF total, which at a dual exercises the box, the displaced tables
  and ``becsum`` on a dense grid the wavefunctions do not live on;
* the **ladder** -- a ``0.05 cos(2 pi x_1 / N)`` Ry modulation against a real
  ``N``-cell supercell run through this package's own SCF, at three band counts,
  reporting ``E - E_super/N`` (which must be positive and falling, the bases
  being nested), the induced density's error, and ``augmentation_residual``.

Run as ``python3 tools/cluster/ultracell_dual.py <case> <output directory>``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from defumat import Calculator
from defumat.basis.builder import build_basis
from defumat.scf.driver import Calculation, run_scf
from defumat.ultracell import run_ultracell, with_external_potential

AMPLITUDE = 0.05

#: Set by the probe arm below; see :func:`main`.
HALF_CUTOFF = False

SILICON = """&control
 calculation='scf'
/
&system
 ibrav=2, celldm(1)=10.20, nat=2, ntyp=1,
 ecutwfc={ecutwfc:.1f}, ecutrho={ecutrho:.1f},
 nosym=.true., noinv=.true.,{spin}
/
&electrons
 conv_thr=1.0d-12
/
ATOMIC_SPECIES
 Si 28.086 {upf}
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 {k0} {k1} {k2} 0 0 0
"""

#: The spinor cell carries a moment, so that the four Pauli components are all
#: alive rather than reducing to the charge one -- which is the difference
#: between this and the platinum case below.
SPINOR_BLOCK = """
 noncolin=.true., starting_magnetization(1)=0.2, angle1(1)=0.0, angle2(1)=0.0,
 occupations='smearing', smearing='gaussian', degauss=0.02,"""

PLATINUM = """&control
 calculation='scf'
/
&system
 ibrav=2, celldm(1)=7.42, nat=1, ntyp=1,
 ecutwfc={ecutwfc:.1f}, ecutrho={ecutrho:.1f},
 lspinorb=.true., noncolin=.true., starting_magnetization=0.0,
 occupations='smearing', smearing='mp', degauss=0.02,
 nosym=.true., noinv=.true.
/
&electrons
 conv_thr=1.0d-10
/
ATOMIC_SPECIES
 Pt 195.08 Pt.rel-pz-n-rrkjus.UPF
ATOMIC_POSITIONS alat
 Pt 0.0 0.0 0.0
K_POINTS automatic
 {k0} {k1} {k2} 0 0 0
"""

#: ``(label, template, dataset, ecutwfc, dual, spinor, nbnd rungs, reference nbnd,
#: kgrid, conv_thr)``. The last two are per case because platinum is a **metal**
#: and silicon is not: an fcc metal on eight k-points with Methfessel-Paxton
#: smearing does not reach 1e-12 at all (it stalled at 1.9e-3 in 100 iterations,
#: job `20350372` tasks 7 and 8), and the committed cell's own pair -- the
#: ``kgrid`` the guide's snippet runs and ``conv_thr = 1e-10`` from the input --
#: is what converges it.
#: The spinor rungs are twice the collinear ones because a spinor band holds one
#: electron where a collinear band holds two -- comparing the two regimes at the
#: same ``nbnd`` reads as a factor of two of missing convergence and looks
#: exactly like a missing term (``PLAN.md`` P88 stage 6).
CASES = [
    # **The control, and it is first.** The same code path at the pair every
    # committed augmented number was taken at, so that a rung at the dual is
    # read against this rather than against a table in another file: the phase
    # record has +1.07e-4, +4.46e-6 and +4.82e-7 Ry here at nbnd = 12, 24, 48.
    ("si-us-collinear-dual4", "si", "Si.pz-n-rrkjus_psl.0.1.UPF", 16.0, 4.0,
     False, (12, 24, 48, 96), 12, (1, 2, 2), 1.0e-12),
    ("si-paw-collinear-dual4", "si", "Si.pz-n-kjpaw_psl.0.1.UPF", 16.0, 4.0,
     False, (12, 24, 48, 96), 12, (1, 2, 2), 1.0e-12),
    ("si-us-collinear-dual8", "si", "Si.pz-n-rrkjus_psl.0.1.UPF", 16.0, 8.0,
     False, (12, 24, 48, 96), 12, (1, 2, 2), 1.0e-12),
    ("si-paw-collinear-dual8", "si", "Si.pz-n-kjpaw_psl.0.1.UPF", 16.0, 8.0,
     False, (12, 24, 48, 96), 12, (1, 2, 2), 1.0e-12),
    ("si-us-spinor-dual8", "si", "Si.pz-n-rrkjus_psl.0.1.UPF", 16.0, 8.0,
     True, (24, 48, 96, 192), 24, (1, 2, 2), 1.0e-12),
    ("si-paw-spinor-dual8", "si", "Si.pz-n-kjpaw_psl.0.1.UPF", 16.0, 8.0,
     True, (24, 48, 96, 192), 24, (1, 2, 2), 1.0e-12),
    ("si-paw-collinear-dual12", "si", "Si.pz-n-kjpaw_psl.0.1.UPF", 16.0, 12.0,
     False, (12, 24, 48, 96), 12, (1, 2, 2), 1.0e-12),
    ("pt-soc-dual8", "pt", "Pt.rel-pz-n-rrkjus.UPF", 30.0, 8.0,
     True, (16, 24, 40), 28, (2, 2, 2), 1.0e-10),
    ("pt-soc-dual4", "pt", "Pt.rel-pz-n-rrkjus.UPF", 30.0, 4.0,
     True, (16, 24, 40), 28, (2, 2, 2), 1.0e-10),
]


def unit_cell(work: Path, label, template, upf, ecutwfc, dual, spinor, kgrid,
              pseudo_dir) -> Calculator:
    path = work / f"{label}-cell.in"
    if template == "si":
        text = SILICON.format(
            upf=upf, ecutwfc=ecutwfc, ecutrho=dual * ecutwfc,
            spin=SPINOR_BLOCK if spinor else "",
            k0=kgrid[0], k1=kgrid[1], k2=kgrid[2],
        )
    else:
        text = PLATINUM.format(
            ecutwfc=ecutwfc, ecutrho=dual * ecutwfc,
            k0=kgrid[0], k1=kgrid[1], k2=kgrid[2],
        )
    path.write_text(text)
    return Calculator.from_file(path, pseudo_dir=pseudo_dir)


def supercell(work: Path, label, calculator, template, upf, ecutwfc, dual,
              spinor, shape, kgrid, pseudo_dir) -> Calculator:
    """The same crystal as a real ``shape`` supercell, atoms and all."""
    cell = calculator.system.cell
    vectors = np.asarray(cell.at_alat) * np.asarray(shape)[:, None]
    tau = np.asarray(calculator.system.structure.positions_alat(cell))
    shifts = np.stack(
        np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    positions = np.concatenate(
        [tau + shift @ np.asarray(cell.at_alat) for shift in shifts]
    )
    species = "Si" if template == "si" else "Pt"
    mass = "28.086" if template == "si" else "195.08"
    rows = "\n".join(f" {v[0]:.12f} {v[1]:.12f} {v[2]:.12f}" for v in vectors)
    atoms = "\n".join(
        f" {species} {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}" for p in positions
    )
    if template == "si":
        spin = SPINOR_BLOCK if spinor else ""
        electrons = "1.0d-12"
    else:
        spin = ("\n lspinorb=.true., noncolin=.true., "
                "starting_magnetization=0.0,\n"
                " occupations='smearing', smearing='mp', degauss=0.02,")
        electrons = "1.0d-10"
    path = work / f"{label}-supercell.in"
    path.write_text(f"""&control
 calculation='scf'
/
&system
 ibrav=0, celldm(1)={float(cell.alat):.10f}, nat={len(positions)}, ntyp=1,
 ecutwfc={ecutwfc:.1f}, ecutrho={dual * ecutwfc:.1f}, nosym=.true., noinv=.true.,{spin}
/
&electrons
 conv_thr={electrons}
/
CELL_PARAMETERS alat
{rows}
ATOMIC_SPECIES
 {species} {mass} {upf}
ATOMIC_POSITIONS alat
{atoms}
K_POINTS automatic
 {kgrid[0]} {kgrid[1]} {kgrid[2]} 0 0 0
""")
    return Calculator.from_file(path, pseudo_dir=pseudo_dir)


def fourier(field, grid, miller):
    """A real field read at a given list of Miller indices, on its own box."""
    box = np.asarray(grid)
    J = np.asarray(miller) % box
    flat = J[:, 0] * (box[1] * box[2]) + J[:, 1] * box[2] + J[:, 2]
    spectrum = np.fft.fftn(np.asarray(field)) / np.asarray(field).size
    return spectrum.reshape(-1)[flat]


def main(index: int, out: Path, pseudo_dir: Path) -> None:
    (label, template, upf, ecutwfc, dual, spinor, rungs, reference_nbnd,
     kgrid, conv_thr) = CASES[index]
    shape = (2, 1, 1)
    # A loose ground state is refused by ``run_ultracell`` and the loop's own
    # tolerances have to sit above it, so all three follow the case's cell.
    reference_thr = max(conv_thr * 10, 1.0e-11)
    loop_thr = max(conv_thr * 100, 1.0e-10)
    folded = tuple(n * m for n, m in zip(shape, kgrid))
    work = out / label
    work.mkdir(parents=True, exist_ok=True)
    record = {
        "case": label, "ecutwfc": ecutwfc, "ecutrho": dual * ecutwfc,
        "dual": dual, "spinor": spinor, "dataset": upf,
        "shape": list(shape), "kgrid": list(kgrid), "conv_thr": conv_thr,
    }
    started = time.time()

    calculator = unit_cell(work, label, template, upf, ecutwfc, dual, spinor,
                           folded, pseudo_dir)
    basis = build_basis(calculator.system)
    record["doublegrid"] = bool(basis.doublegrid)
    record["dense_grid"] = list(basis.dense.grid)
    record["smooth_grid"] = list(basis.smooth.grid)
    record["ngm"] = int(basis.dense.ngm)
    record["ngms"] = int(basis.smooth.ngm)
    print(f"[{label}] doublegrid={basis.doublegrid} dense={basis.dense.grid} "
          f"smooth={basis.smooth.grid}", flush=True)

    scf = calculator.get_scf(conv_thr=conv_thr,
                             nbnd=max(8, reference_nbnd // 2))
    record["unit_cell_energy"] = float(scf.total_energy)
    record["unit_cell_converged"] = bool(scf.converged)
    print(f"[{label}] unit cell {float(scf.total_energy):.10f} Ry", flush=True)

    # -- the null -----------------------------------------------------------
    null = run_ultracell(calculator.system, calculator.pseudos, scf, shape,
                         kgrid, nbnd=rungs[1], conv_thr=loop_thr)
    record["null_energy"] = float(null.total_energy)
    record["null_gap"] = float(null.total_energy) - float(scf.total_energy)
    record["null_residual"] = float(null.augmentation_residual)
    record["null_converged"] = bool(null.converged)
    print(f"[{label}] null gap {record['null_gap']:+.3e} Ry "
          f"residual {record['null_residual']:.3e}", flush=True)

    # -- the supercell reference --------------------------------------------
    reference_cell = supercell(work, label, calculator, template, upf, ecutwfc,
                               dual, spinor, shape, kgrid, pseudo_dir)
    reference_basis = build_basis(reference_cell.system)
    grid = reference_basis.dense.grid
    coordinates = np.stack(
        np.meshgrid(*[np.arange(m) / m for m in grid], indexing="ij"), axis=-1
    )
    calculation = with_external_potential(
        Calculation(reference_cell.system, reference_cell.pseudos),
        jnp.asarray(AMPLITUDE * np.cos(2 * np.pi * coordinates[..., 0])),
    )
    reference = run_scf(reference_cell.system, reference_cell.pseudos,
                        calculation=calculation, conv_thr=reference_thr,
                        nbnd=reference_nbnd)
    per_cell = float(reference.total_energy) / int(np.prod(shape))
    # **The supercell's own null, and it is the first thing a supercell
    # comparison should run** (``OPEN.md`` Part X item 2, which says it of a
    # metal that breaks its own symmetry and is just as true of a grid). With
    # nothing applied the supercell must reproduce its unit cell; at
    # ``ecutrho = 4 ecutwfc`` it does, to 5e-13 Ry, because its FFT box is
    # exactly the tiled one. At a dual it is not -- ``(54, 25, 25)`` against a
    # tiled ``(50, 25, 25)`` on silicon -- and the supercell then sits 2.2e-6 Ry
    # per cell *above* its own unit cell for the same state. Every raw gap
    # below carries that offset, which is why the modulation energy is reported
    # beside it: the offset cancels in a difference of differences.
    reference_null = run_scf(reference_cell.system, reference_cell.pseudos,
                             conv_thr=reference_thr, nbnd=reference_nbnd)
    null_per_cell = float(reference_null.total_energy) / int(np.prod(shape))
    record["supercell_null_per_cell"] = null_per_cell
    record["supercell_null_offset"] = null_per_cell - float(scf.total_energy)
    record["supercell_null_converged"] = bool(reference_null.converged)
    record["supercell_modulation_energy"] = per_cell - null_per_cell
    print(f"[{label}] supercell null offset "
          f"{record['supercell_null_offset']:+.4e} Ry per cell", flush=True)
    record["supercell_energy_per_cell"] = per_cell
    record["supercell_converged"] = bool(reference.converged)
    # **The two sides do not discretise on the same set and that is the first
    # thing to read when a gap comes out negative.** The ultracell keeps the
    # unit cell's dense sphere at *every* Q (Elk's convention) while the
    # supercell keeps a genuine sphere of its own, so the two truncations agree
    # only where the box does.
    record["supercell_grid"] = list(grid)
    record["supercell_ngm"] = int(reference_basis.dense.ngm)
    record["box_grid"] = list(null.ultracell.grid)
    record["box_is_tiled_cell"] = [
        int(n * m) for n, m in zip(shape, basis.dense.grid)
    ] == list(grid)
    print(f"[{label}] supercell/N {per_cell:.10f} Ry "
          f"converged={reference.converged}", flush=True)

    miller = np.asarray(reference_basis.dense.miller)
    exact = fourier(np.asarray(reference.density)[0], grid, miller)

    # -- the ladder, in both cut-off conventions -----------------------------
    #
    # **Why there are two.** The supercell keeps a genuine sphere of radius
    # ``sqrt(ecutrho)`` in its own reciprocal lattice; the ultracell keeps the
    # unit cell's dense sphere at *every* ``Q``, which is Elk's choice and what
    # makes the ``N = 1`` limit reduce to the unit cell exactly. Neither set
    # contains the other -- ``{G + Q}`` holds points with ``|G + Q|`` past the
    # cut-off and misses points inside it -- so the two sides do not discretise
    # the same functional and the nested-basis bound, which is what the sign of
    # a gap is read against, does not apply between them. The second arm masks
    # the ultracell's own ``|G + Q|`` sphere instead, where the containment is
    # exact, and the difference between the two arms is what the convention is
    # worth (``OPEN.md`` Part X item 1, which asks the same question of the
    # Hartree term alone).
    modulation = lambda r: AMPLITUDE * np.cos(2 * np.pi * r[..., 0] / shape[0])
    gcut = float(dual * ecutwfc)

    def sphere_convention(self, cell_mask, _cell=calculator.system.cell,
                          _gcut=gcut):
        return self.g2(_cell) <= (0.25 * _gcut if HALF_CUTOFF else _gcut)

    def ladder(arm, steps=None):
        from defumat.ultracell.grid import Ultracell

        original = Ultracell.reciprocal_mask
        if arm in ("sphere", "half"):
            Ultracell.reciprocal_mask = sphere_convention
        rows = {}
        try:
            for nbnd in (steps or rungs):
                result = run_ultracell(
                    calculator.system, calculator.pseudos, scf, shape, kgrid,
                    nbnd=nbnd, external=modulation, conv_thr=loop_thr,
                    states_conv_thr=loop_thr,
                )
                box = result.ultracell.grid
                ours = fourier(np.asarray(result.density)[0], box, miller)
                flat = fourier(
                    np.asarray(
                        result.ultracell.tile(jnp.asarray(scf.density))
                    )[0], box, miller,
                )
                induced, induced_exact = ours - flat, exact - flat
                rows[str(nbnd)] = {
                    "energy": float(result.total_energy),
                    "gap": float(result.total_energy) - per_cell,
                    # The quantity both sides measure the same way: each cell's
                    # energy relative to its *own* unmodulated state, so the
                    # grid offset above cancels and what is left is the physics.
                    "modulation_gap": (
                        (float(result.total_energy) - float(null.total_energy))
                        - (per_cell - null_per_cell)
                    ),
                    "density_error": float(
                        np.abs(induced - induced_exact).max()
                        / np.abs(induced_exact).max()
                    ),
                    "residual": float(result.augmentation_residual),
                    "converged": bool(result.converged),
                    "iterations": int(result.iterations),
                }
                print(f"[{label}] {arm:6s} nbnd={nbnd:3d} "
                      f"gap={rows[str(nbnd)]['gap']:+.4e} Ry "
                      f"modulation={rows[str(nbnd)]['modulation_gap']:+.4e} "
                      f"error={rows[str(nbnd)]['density_error']:.3e} "
                      f"residual={rows[str(nbnd)]['residual']:.3e}", flush=True)
        finally:
            Ultracell.reciprocal_mask = original
        return rows

    kept = int(np.sum(np.asarray(sphere_convention(null.ultracell, None))))
    record["sphere_kept"] = kept
    record["tiled_kept"] = int(np.prod(shape)) * int(basis.dense.ngm)
    print(f"[{label}] kept: sphere {kept}, tiled {record['tiled_kept']}, "
          f"supercell {record['supercell_ngm']}", flush=True)

    record["ladder"] = ladder("elk")
    # The convention arm on silicon only: on platinum the ladder is the
    # expensive half of the case and the convention is the cheap question.
    if template == "si":
        record["ladder_sphere"] = ladder("sphere")
    # **The arm whose job is to move.** If the two above agree, that is either
    # the convention being worth nothing or the patch point being dead, and the
    # two read identically. A quarter of the cut-off -- below ``4 ecutwfc``,
    # where the density is genuinely truncated -- has to move the total.
    global HALF_CUTOFF
    HALF_CUTOFF = True
    try:
        record["ladder_half"] = ladder("half", rungs[:1])
    finally:
        HALF_CUTOFF = False
    record["seconds"] = time.time() - started

    path = out / f"{label}.json"
    path.write_text(json.dumps(record, indent=2))
    print(f"[{label}] written to {path} in {record['seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    case = int(sys.argv[1])
    output = Path(sys.argv[2])
    pseudos = Path(sys.argv[3]) if len(sys.argv) > 3 else (
        Path(__file__).resolve().parents[2] / "tests" / "data" / "pseudo"
    )
    output.mkdir(parents=True, exist_ok=True)
    main(case, output, pseudos)
