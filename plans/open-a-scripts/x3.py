"""OPEN.md Part X item 3: the ultrasoft spinor ladder at ecutrho = 12 ecutwfc, where the supercell's box is the tiled one."""
import sys
from pathlib import Path
sys.path.insert(0, "/l/ladovj1/w-a/tools/cluster")
import ultracell_dual as u

u.CASES.append(("si-us-spinor-dual12", "si", "Si.pz-n-rrkjus_psl.0.1.UPF", 16.0, 12.0,
                True, (24, 48, 96, 192), 24, (1, 2, 2), 1.0e-12))
out = Path("/l/ladovj1/review/open-a/x3"); out.mkdir(parents=True, exist_ok=True)
u.main(len(u.CASES) - 1, out, Path("/l/ladovj1/w-a/tests/data/pseudo"))
print("X3-DONE", flush=True)
