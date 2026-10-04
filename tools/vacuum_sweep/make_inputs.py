"""Write the vacuum-sweep inputs: each cell at several vacuum sizes, positions in bohr.

The layer's internal geometry is held fixed (positions in bohr relative to the
layer's centre) and only the cell length along z changes, so the sweep changes
the vacuum and nothing else. ecut, k-mesh and smearing are each cell's own.
"""
import math
import sys
from pathlib import Path

out = Path(sys.argv[1])
pseudo = sys.argv[2]
out.mkdir(parents=True, exist_ok=True)

HEX = lambda a: [(a, 0.0), (-0.5 * a, 0.5 * math.sqrt(3) * a)]


def write(name, a_vectors, c, species, atoms, system, kpts, electrons):
    """atoms: (symbol, f1, f2, z) with f1, f2 in-plane crystal and z in bohr from the centre."""
    (a1, a2) = a_vectors
    lines = ["&control", "  calculation = 'scf'", f"  pseudo_dir = '{pseudo}'", "/",
             "&system", "  ibrav = 0", f"  nat = {len(atoms)}", f"  ntyp = {len(species)}"]
    lines += [f"  {k} = {v}" for k, v in system.items()]
    lines += ["/", "&electrons"] + [f"  {k} = {v}" for k, v in electrons.items()] + ["/"]
    lines += ["ATOMIC_SPECIES"] + [f"  {s} {m} {f}" for s, m, f in species]
    lines += ["CELL_PARAMETERS bohr",
              f"  {a1[0]:.10f} {a1[1]:.10f} 0.0", f"  {a2[0]:.10f} {a2[1]:.10f} 0.0",
              f"  0.0 0.0 {c:.10f}"]
    lines += ["ATOMIC_POSITIONS bohr"]
    for s, f1, f2, z in atoms:
        x = f1 * a1[0] + f2 * a2[0]
        y = f1 * a1[1] + f2 * a2[1]
        lines.append(f"  {s} {x:.10f} {y:.10f} {0.5 * c + z:.10f}")
    lines += ["K_POINTS automatic", f"  {kpts}"]
    (out / f"{name}.in").write_text("\n".join(lines) + "\n")


# Al(100), five layers, 15 bohr thick: benchmarks/al-slab.in with the vacuum varied.
a = 5.303301
al_z = [(f - 0.5) * 31.0 for f in (0.258065, 0.379032, 0.5, 0.620968, 0.741935)]
al_xy = [(0.0, 0.0), (0.5, 0.5), (0.0, 0.0), (0.5, 0.5), (0.0, 0.0)]
for vac in (16, 32, 48, 64):
    c = 15.0 + vac
    write(f"al-v{vac}", [(a, 0.0), (0.0, a)], c, [("Al", 26.98, "Al.pz-vbc.UPF")],
          [("Al", f1, f2, z) for (f1, f2), z in zip(al_xy, al_z)],
          {"ecutwfc": 12.0, "occupations": "'smearing'", "smearing": "'gaussian'",
           "degauss": 0.05},
          "2 2 1 0 0 0",
          {"conv_thr": "1.0d-8", "electron_maxstep": 200})

# NbSe2 monolayer (tests/data/qe/nbse2-monolayer.in): a 2D metal, 6.35 bohr Se to Se.
a = 6.5044373210
c0 = 5.2295177223 * a
dz = (0.5933533333 - 0.5) * c0
for vac in (16, 28, 48, 64):
    c = 2 * dz + vac
    write(f"nbse2-v{vac}", HEX(a), c,
          [("Nb", 92.906, "Nb.pbe-nc-sg15.UPF"), ("Se", 78.971, "Se.pbe-nc-sg15.UPF")],
          [("Nb", 0.0, 0.0, 0.0), ("Se", 1 / 3, 2 / 3, dz), ("Se", 1 / 3, 2 / 3, -dz)],
          {"ecutwfc": 40.0, "occupations": "'smearing'", "smearing": "'fermi-dirac'",
           "degauss": 0.002, "nbnd": 20},
          "9 9 1 0 0 0",
          {"conv_thr": "1.0d-8", "electron_maxstep": 200})

# Graphene (tests/data/qe/graphene-monolayer.in): a semimetal, one atomic plane.
a = 4.6511
for vac in (20, 40, 60):
    write(f"graphene-v{vac}", HEX(a), float(vac), [("C", 12.011, "C.pbe-hgh.UPF")],
          [("C", 0.0, 0.0, 0.0), ("C", 1 / 3, 2 / 3, 0.0)],
          {"ecutwfc": 40.0, "occupations": "'smearing'", "smearing": "'gaussian'",
           "degauss": 0.02, "nbnd": 12},
          "12 12 1 0 0 0",
          {"conv_thr": "1.0d-10", "mixing_beta": 0.5, "electron_maxstep": 200})

# hBN monolayer: a 2D insulator with a gap of several eV, fixed occupations.
# ecutwfc = 40 is low for HGH nitrogen; it is enough for the screening, which is
# what an iteration count measures, and it is the same at every vacuum.
a = 4.7419
for vac in (20, 40, 60):
    write(f"hbn-v{vac}", HEX(a), float(vac),
          [("B", 10.811, "B.pbe-hgh.UPF"), ("N", 14.007, "N.pbe-hgh.UPF")],
          [("B", 0.0, 0.0, 0.0), ("N", 1 / 3, 2 / 3, 0.0)],
          {"ecutwfc": 40.0, "occupations": "'fixed'"},
          "6 6 1 0 0 0",
          {"conv_thr": "1.0d-10", "electron_maxstep": 200})

print("\n".join(sorted(p.name for p in out.glob("*.in"))))
