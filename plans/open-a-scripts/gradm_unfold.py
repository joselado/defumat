"""noncollinear_gga = 'grad_m' on the cobalt helix against its unfolded supercell, at one density.

    gradm_unfold.py --root W [--ecutrho 160] [--iterations 20]

The spiral's potential and exchange-correlation energy at a density (twenty
iterations of the cell), and the four-cell supercell's at the unfolded density:
the supercell's ``etxc`` is four times the spiral's and its potential is the
unfolded one, exactly up to the two discretisations, if the laboratory gradient
of the transverse pair is right. The 'kubler' pair is printed as the control.
"""
import argparse, dataclasses, sys, warnings
from pathlib import Path
import numpy as np
import jax.numpy as jnp

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--ecutrho", type=float, default=160.0)
p.add_argument("--iterations", type=int, default=20)
a = p.parse_args()
ROOT = Path(a.root)
sys.path.insert(0, str(ROOT))
from tests.regression.test_spin_spirals_augmented import GENERATED, _cobalt, _pseudos
from defumat.io.pwin import parse_pw_input
from defumat.system import build_system
from defumat.scf.driver import Calculation, run_scf
from defumat.workflows.spiral import unfold_spiral_density

PSEUDO = ROOT / "tests" / "data" / "pseudo"
spiral = build_system(parse_pw_input(_cobalt(a.ecutrho)))
pseudos = _pseudos(spiral, PSEUDO)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    density = np.asarray(run_scf(spiral, pseudos, conv_thr=1e-11,
                                 max_iterations=a.iterations).density)
text = (GENERATED / "co-helix4-nosoc.in").read_text().replace(
    "ecutrho = 160.0", f"ecutrho = {a.ecutrho}")
supercell = build_system(parse_pw_input(text))
super_pseudos = _pseudos(supercell, PSEUDO)
for gga in ("kubler", "grad_m"):
    sc = Calculation(dataclasses.replace(spiral, noncollinear_gga=gga), pseudos)
    cc = Calculation(dataclasses.replace(supercell, noncollinear_gga=gga), super_pseudos)
    cc.quantization_axis = None
    shape = tuple(cc.basis.dense.grid)
    unfolded = unfold_spiral_density(density, spiral.spiral_q, (1, 1, 4), shape)
    ps = sc.potential(jnp.asarray(density))
    pc = cc.potential(jnp.asarray(unfolded))
    v_expected = unfold_spiral_density(np.asarray(ps.v_scf), spiral.spiral_q, (1, 1, 4), shape)
    v_super = np.asarray(pc.v_scf)
    print(gga, "etxc spiral x4 %.12f supercell %.12f diff %.3e" % (
        4 * float(ps.etxc), float(pc.etxc), 4 * float(ps.etxc) - float(pc.etxc)),
        "| v charge %.3e  B %.3e  (max |B| %.3e)" % (
        np.abs(v_super[0] - v_expected[0]).max(), np.abs(v_super[1:] - v_expected[1:]).max(),
        np.abs(v_expected[1:]).max()), flush=True)
