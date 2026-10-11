"""'grad_m' on PAW and on the meta-GGA at the SCF level (plans/noncollinear-gga-grad-m-2.md).

    gradm_paw_meta.py --root W chain|chain-kubler|meta|meta-ni

``chain``: the oxygen chain at q = b3/4 against its four-cell 90-degree
supercell on ``O.pbe-kjpaw.UPF`` under PBE, the electronic energy per cell
(the PAW spheres' covariant gradient is in play on the spiral side only).
``meta``: the oxygen chain at q = 0 on the PBE PAW dataset under TB09, one
moment along x, 'grad_m' against the signed
'kubler' branch, which are one functional on a collinear state.
"""
import sys, time, warnings
from pathlib import Path

ROOT = Path(sys.argv[sys.argv.index("--root") + 1]); sys.path.insert(0, str(ROOT))
mode = sys.argv[-1]
from defumat.io.pwin import parse_pw_input
from defumat.pseudo import read_upf
from defumat.scf import run_scf
from defumat.system import build_system
from tests.conftest import GENERATED

PSEUDO = ROOT / "tests" / "data" / "pseudo"


def run(text, **kw):
    system = build_system(parse_pw_input(text))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    t = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = run_scf(system, pseudos, **kw)
    print(f"  nspin_mag {system.nspin_mag} spiral {system.spiral} it {r.iterations} conv {r.converged} "
          f"acc {r.accuracy:.2e} E {r.total_energy:.12f} elec {r.total_energy - r.energy_terms['ewald']:.12f} "
          f"{time.time() - t:.0f} s", flush=True)
    return r


if mode.startswith("chain"):
    extra = "noncollinear_gga = 'kubler', " if mode == "chain-kubler" else ""
    def pbe(text):
        return text.replace("O.pz-kjpaw.UPF", "O.pbe-kjpaw.UPF").replace(
            "noncolin = .true.", f"{extra}noncolin = .true.")
    spiral = run(pbe((GENERATED / "o-chain-spiral-paw.in").read_text()),
                 conv_thr=1e-11, mixing_beta=0.3, max_iterations=300)
    supercell = run(pbe((GENERATED / "o-chain-90deg-paw.in").read_text()),
                    conv_thr=1e-11, mixing_beta=0.3, max_iterations=300)
    e1 = spiral.total_energy - spiral.energy_terms["ewald"]
    e4 = (supercell.total_energy - supercell.energy_terms["ewald"]) / 4.0
    print(f"{mode}: spiral - supercell/4 = {e1 - e4:.3e} Ry", flush=True)
elif mode == "meta-ni":
    base = (GENERATED / "ni-soc-nosym.in").read_text().replace(
        "occupations = 'smearing'", "input_dft = 'TB09', occupations = 'smearing'")
    out = {}
    for choice in ("grad_m", "kubler"):
        print(choice, flush=True)
        out[choice] = run(base.replace("noncolin = .true.",
                                       f"noncollinear_gga = '{choice}', noncolin = .true."),
                          conv_thr=1e-10, mixing_beta=0.3, max_iterations=150)
    print(f"meta-ni: grad_m - kubler = {out['grad_m'].total_energy - out['kubler'].total_energy:.3e} Ry", flush=True)
else:
    base = (GENERATED / "o-chain-spiral-paw.in").read_text().replace(
        "O.pz-kjpaw.UPF", "O.pbe-kjpaw.UPF").replace(
        "spiral_q(1) = 0.0, spiral_q(2) = 0.0, spiral_q(3) = 0.25", "").replace(
        "occupations = 'smearing'", "input_dft = 'TB09', occupations = 'smearing'")
    out = {}
    for choice in ("grad_m", "kubler"):
        print(choice, flush=True)
        out[choice] = run(base.replace("noncolin = .true.",
                                       f"noncollinear_gga = '{choice}', noncolin = .true."),
                          conv_thr=1e-11, mixing_beta=0.3, max_iterations=300)
    print(f"meta: grad_m - kubler = {out['grad_m'].total_energy - out['kubler'].total_energy:.3e} Ry", flush=True)
