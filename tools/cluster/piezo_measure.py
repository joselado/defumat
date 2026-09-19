"""``e_14`` two ways, so that the piezoelectric tensor's dataset refusal can go
with a number rather than with an argument.

The refusal (``response/piezo.py``) says an ultrasoft or PAW dataset is missing
one term in the strain leg, and names ``response/strain.py`` as refusing the
same datasets for the same reason. That second half is stale -- P41 wired the
``Q_ij(r)`` strain term and ``test_electrostriction.py`` pins it at 4.6e-4 for
ultrasoft and 4.7e-4 for PAW -- but "the stated blocker is gone" is not a
measurement, and the composite has never been measured on an augmented dataset.
This script measures it.

**The two routes share no machinery.** One is this package's: a Sternheimer
field response, then one ``jvp`` of the stress along it, which is
``clamped_ion_piezoelectric``. The other is Elk's (``src/piezoelt.f90``): one
*converged ground state per strain* and a finite difference of the Berry-phase
polarization, which touches no response solver at all. The Voigt convention here
puts the factor of two on the strain (``eps_4 = 2 eps_23``), so with a strain
tensor carrying ``E[1,2] = E[2,1] = s`` the finite difference is
``dP_x / d(2 s)`` and that is ``e_14`` directly.

**The norm-conserving cell is the calibration and is not optional.** Run alone,
a disagreement on the ultrasoft cell says nothing about which side is wrong. The
same pair on a norm-conserving zincblende cell, where the response route is the
validated one, says what the finite-difference harness itself is worth -- the
strain step, the polarization mesh, the branch of the quantum -- and the
ultrasoft comparison is then read against that floor.

**What it found, 2026-09-19, and why the script now has a mesh ladder.** The
calibration disagreed: the response route reads -0.763786 C/m^2 on the
norm-conserving cell and the finite difference -0.661386, **13.4 per cent** low,
against 15.7 per cent on the ultrasoft one. Doubling the strain moves the
difference by 0.3 per cent and away from the response route, so the step is not
it. The untested difference between the two routes is how they sample ``k`` --
the response integrates the SCF's ``4 4 4 0 0 0``, 64 points, and the Berry
phase runs ``nppstr x transverse`` strings, 396 on that cell -- so ``--kmesh``
moves the response's mesh and ``--nppstr``/``--transverse`` the difference's,
and whichever number moves toward the other is the unconverged one.

**Two traps, both checked, so that neither is chased again.** The two committed
AlAs cells are **enantiomorphs** and their ``e_14`` therefore have opposite
signs: ``alas-raman.in`` writes ``ATOMIC_POSITIONS (alat)`` and puts As at
``a(1/4, 1/4, 1/4)``, ``alas-piezo.in`` writes ``crystal``, and for
``ibrav = 2`` that triple is ``0.25 (a1 + a2 + a3) = a(-1/4, 1/4, 1/4)``, which
differs from the first by ``a(1/2, 0, 0)`` and is not a lattice vector. So the
two cells are compared against *their own* response route and never against each
other. And the contraction in :func:`finite_difference` uses the ``+s`` cell's
lattice vectors and volume for both strains, which is exact for this component
rather than a dropped term: for a pure ``y``-``z`` shear ``(S a_g)_x = 0`` for
all three vectors, so the ``x`` components of ``a_g(+s)`` and ``a_g(-s)`` are
identical and the volumes are equal at ``1 - s^2``. The same statement is why
``e_14`` carries no proper-against-improper correction and no dependence on the
polarization branch, both corrections pairing two different Cartesian labels.

Run one case per invocation (the cells cost very different amounts):

    python3 tools/cluster/piezo_measure.py nc   --out results-nc.json
    python3 tools/cluster/piezo_measure.py us   --out results-us.json

and one rung of the ladder per invocation too:

    python3 tools/cluster/piezo_measure.py nc --kmesh 6 --skip-difference
    python3 tools/cluster/piezo_measure.py nc --nppstr 15 --transverse 6 6 \
            --shears 0.005 --skip-response
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

#: The two cells, both zincblende (``-43m``), where ``e_14`` is the single
#: independent component. The norm-conserving one is the calibration.
CASES = {
    "nc": {
        "input": "tests/data/qe/alas-raman.in",
        "what": "AlAs, norm-conserving LDA (Al.pz-vbc, As.pz-bhs), ecutwfc 10",
        # Denser than the ultrasoft case's because this cell is cheap and it is
        # the one whose job is to say what the harness itself is worth.
        "nppstr": 11,
        "transverse": (6, 6),
    },
    "us": {
        "input": "tests/data/qe/alas-piezo.in",
        "what": "AlAs, ultrasoft PBE (Al/As.pbe-n-rrkjus_psl.1.0.0), ecutwfc 25",
        "nppstr": 7,
        "transverse": (4, 4),
    },
}


def shear(magnitude: float) -> np.ndarray:
    """``E[1,2] = E[2,1] = s``: the strain whose Voigt form is ``eps_4 = 2 s``."""
    strain = np.zeros((3, 3))
    strain[1, 2] = strain[2, 1] = magnitude
    return strain


def _without_symmetry(calculator):
    """The same crystal with ``nosym`` set, so that ``--kmesh`` may ladder it.

    ``alas-piezo.in`` keeps its symmetry and ``alas-raman.in`` does not, which is
    why the first ladder could move the calibration cell's k-mesh and not the
    ultrasoft one: :func:`_with_full_grid` refuses a reduced cell, and rightly,
    because the tensor would then be symmetrised on a group the response was not
    integrated over. Dropping the group removes the objection instead of
    ignoring it, and it makes the two cells comparable in the one way that
    matters here, since the calibration cell has no group either.

    ``System`` is an ``eqx.Module`` and ``nosym`` is a *static* field, so
    ``dataclasses.replace`` is the way in and ``eqx.tree_at`` is not: a static
    field is not a leaf. Nothing downstream is stale, because the rotations are
    read through ``None if self.nosym else ...`` every time they are asked for
    (``system/builder.py:539`` and ``:583``) rather than cached at build time.
    """
    import dataclasses

    from defumat.calculator import Calculator

    if calculator.system.nosym:
        return calculator
    return Calculator(dataclasses.replace(calculator.system, nosym=True),
                      calculator.pseudos, announce=False)


def _with_full_grid(calculator, mesh):
    """The **whole** ``mesh x mesh x mesh`` grid, unshifted, no symmetry.

    The calibration cell is ``nosym``/``noinv`` for P24's reason: a response on a
    reduced set is a polar vector field and must be symmetrised as one, and a
    *shifted* Monkhorst-Pack grid is not closed under the point group at all. So
    this builds the complete grid, and refuses a cell whose own set is reduced
    rather than quietly changing what is being compared -- unless the group has
    been dropped first with :func:`_without_symmetry`, which is what ``--nosym``
    does and is the only way the ultrasoft cell can be laddered at all.
    """
    from defumat.system.kpoints import KPoints

    if calculator.system.kpoints.reduced and not calculator.system.nosym:
        raise SystemExit("--kmesh refuses a symmetry-reduced cell: the response "
                         "would be compared on a different k-set from the one "
                         "the tensor was symmetrised on. Pass --nosym to drop "
                         "the group first, which is what the calibration cell "
                         "already does")
    grid = (int(mesh),) * 3
    return calculator.with_kpoints(
        KPoints.automatic(grid, (0, 0, 0), calculator.system.cell)
    )


def _polarization_child(payload, queue):
    """One strained geometry, in a process of its own. See :func:`polarization_vector`."""
    import numpy as np

    from defumat.calculator import Calculator

    extra = ({} if payload.get("k_batch") is None
             else {"k_batch": payload["k_batch"]})
    calculator = Calculator.from_file(payload["input"],
                                      pseudo_dir=payload["pseudo_dir"],
                                      announce=False, **extra)
    if payload.get("nosym"):
        calculator = _without_symmetry(calculator)
    if payload.get("kmesh"):
        calculator = _with_full_grid(calculator, payload["kmesh"])
    at = np.asarray(calculator.system.cell.at)
    strain = np.asarray(payload["strain"])
    strained = calculator.with_cell(at @ (np.eye(3) + strain).T)
    scf = strained.get_scf(conv_thr=payload["conv_thr"])
    cell = strained.system.cell
    # **``Cell.at`` is in bohr**, not in units of ``alat``: ``Cell.volume`` is
    # documented as bohr^3 of exactly this array and ``from_vectors`` takes
    # bohr. Multiplying by ``alat`` here, which is what a QE ``at`` would need,
    # put a factor of 10.575 on the first run of this script and made the
    # finite difference read -6.998 C/m^2 against the response route's -0.764.
    vectors = np.asarray(cell.at)  # bohr; row i is a_i
    phases = []
    for gdir in range(3):
        polarization = strained.get_polarization(
            gdir=gdir, nppstr=payload["nppstr"],
            transverse=tuple(payload["transverse"]),
        )
        phases.append(float(polarization.total_phase))
    queue.put({
        "phases": phases,
        "quantum": float(polarization.quantum),
        "energy": float(scf.total_energy),
        "volume": float(cell.volume),
        "vectors": vectors.tolist(),
    })


def polarization_vector(payload):
    """``P`` in e/bohr^2, cartesian, at clamped ions in the deformed cell.

    Clamped ions is what the tensor is: the atoms keep their **crystal**
    coordinates, which is what ``with_cell`` does when it is given no positions,
    so they follow the cell affinely and nothing relaxes.

    **Each geometry runs in its own process**, and that is not tidiness. XLA's
    CPU backend gives every jitted function its own ORC dylib and mmaps its
    sections; ``vm.max_map_count`` is the ordinary 65530, so a script that
    converges half a dozen distinct cells and runs three Berry-phase meshes on
    each exhausts **mappings** rather than bytes and dies with
    ``Failed to materialize symbols`` or ``LLVM compilation error: Cannot
    allocate memory`` -- the latter on a 118-byte request, with 120 GB
    allocated and three resident. ``jax.clear_caches()`` does not help: it drops
    JAX's own caches and not the loaded modules. A child process does, by
    exiting.
    """
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    child = context.Process(target=_polarization_child, args=(payload, queue))
    child.start()
    result = queue.get()
    child.join()
    if child.exitcode != 0:
        raise RuntimeError(f"the polarization child exited {child.exitcode}")
    result["vectors"] = np.asarray(result["vectors"])
    result["polarization"] = sum(
        result["phases"][gdir] * result["vectors"][gdir] for gdir in range(3)
    ) / result["volume"]
    return result


def wrapped(difference: float) -> float:
    """A phase difference folded onto ``[-1/2, 1/2)``, since ``P`` is modulo one.

    The two strains are small and the branch should not move, but reading the
    difference through the wrap costs nothing and turns a branch jump from a
    wrong number into a visible one: the raw phases are reported beside it.
    """
    return float(np.angle(np.exp(2j * np.pi * difference)) / (2.0 * np.pi))


def finite_difference(base, magnitude, nppstr, transverse, conv_thr):
    """``e_14`` by Elk's route: one ground state per strain, differenced."""
    from defumat.units import BOHR_RADIUS_SI, ELECTRON_SI

    factor = ELECTRON_SI / BOHR_RADIUS_SI**2  # e/bohr^2 -> C/m^2

    def payload(sign):
        return {**base, "strain": shear(sign * magnitude).tolist(),
                "nppstr": nppstr, "transverse": list(transverse),
                "conv_thr": conv_thr}


    plus = polarization_vector(payload(+1))
    minus = polarization_vector(payload(-1))

    # Through the phases rather than through the assembled vectors, so that the
    # wrap above applies before anything is multiplied by a lattice vector.
    derivative = np.zeros(3)
    for gdir in range(3):
        delta = wrapped(plus["phases"][gdir] - minus["phases"][gdir])
        derivative += delta * plus["vectors"][gdir]
    derivative = derivative / plus["volume"] / (2.0 * 2.0 * magnitude) * factor

    return {
        "e14": float(derivative[0]),
        "dP_deps4": [float(one) for one in derivative],
        "shear": magnitude,
        "phases_plus": plus["phases"],
        "phases_minus": minus["phases"],
        "quantum": plus["quantum"],
        "energies": [plus["energy"], minus["energy"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("--out", default=None, help="write the numbers here as JSON")
    parser.add_argument("--shears", type=float, nargs="+", default=[0.0025, 0.005],
                        help="E[1,2] = E[2,1]; the Voigt shear is twice each")
    parser.add_argument("--nppstr", type=int, default=None)
    # A *pair*: ``string_mesh`` takes how many strings run along each of the
    # two crystal directions that are not ``gdir``.
    parser.add_argument("--transverse", type=int, nargs=2, default=None)
    parser.add_argument("--conv-thr", type=float, default=1.0e-10)
    # The response route's own convergence parameter, and the one thing that
    # differs between the two routes that has never been varied.
    parser.add_argument("--kmesh", type=int, default=None,
                        help="run on the whole N x N x N unshifted grid instead "
                             "of the input's; refuses a reduced cell unless "
                             "--nosym comes with it")
    # `piezo.py`'s docstring says the tape "does not move with ``k_batch``,
    # because what the tape holds is not the k axis", measured at 64 k on the
    # small cell. The ladder's peaks are close to affine in ``nk`` (13.8 GiB at
    # 216 points against 28.3 at 512), so the statement has never been checked
    # where it would bite, and it decides how much memory the ultrasoft rungs
    # need. This dial is how to check it.
    parser.add_argument("--k-batch", type=int, default=None,
                        help="how many k-points are in flight at once; 1 is "
                             "QE's own loop and the smallest working set")
    parser.add_argument("--nosym", action="store_true",
                        help="drop the crystal's point group, which is what the "
                             "calibration cell already does and what lets the "
                             "ultrasoft cell be laddered in k")
    # `clamped_ion_piezoelectric` holds a forward-over-reverse tape and peaks at
    # 139.6 GiB on the ultrasoft cell at 64 k-points, 1.6 GiB a point, which puts
    # a `6 6 6` rung on a whole node. `zstar_eu` is the same number contracted
    # rather than taped and carries no tape at all, so it is how this cell gets
    # laddered -- and the two have now been shown to agree on an augmented
    # dataset, 2.6e-09 C/m^2 on `alas-piezo-tiny.in`, so the ladder is a
    # workstation job rather than a node.
    parser.add_argument("--method", default="autodiff",
                        choices=("autodiff", "zstar_eu"),
                        help="which route assembles the tensor above the shared "
                             "field response")
    parser.add_argument("--skip-response", action="store_true",
                        help="the finite difference alone")
    parser.add_argument("--skip-difference", action="store_true",
                        help="the response route alone, for the k-mesh ladder")
    parser.add_argument("--pseudo-dir", default=str(ROOT / "tests" / "data" / "pseudo"))
    arguments = parser.parse_args()

    import defumat.response.piezo as piezo
    from defumat.calculator import Calculator

    case = CASES[arguments.case]
    nppstr = arguments.nppstr or case["nppstr"]
    transverse = tuple(arguments.transverse or case["transverse"])
    print(f"=== {arguments.case}: {case['what']}", flush=True)
    print(f"    {case['input']}, strings of {nppstr} over a "
          f"{transverse[0]}x{transverse[1]} transverse mesh", flush=True)

    # The dataset refusal is the thing being measured; every other guard stays.
    # It is patched here rather than removed in the package, so that the
    # repository still refuses until there is a number to lift it with.
    measured = arguments.case == "us"
    if measured:
        piezo.require_a_measured_dataset = lambda calculation: None
        print("    the dataset refusal is lifted for this run only", flush=True)
        if arguments.method == "zstar_eu":
            # A *second* refusal and lifted separately, which is the whole
            # point of its being separate: the transcribed route was 1.8 per
            # cent out on this cell until the multipliers' own response was
            # added to it, and this run is what says whether that term is right.
            piezo.require_a_norm_conserving_transcription = lambda c: None
            print("    the transcription refusal is lifted too, which is what "
                  "this run measures", flush=True)

    extra = {} if arguments.k_batch is None else {"k_batch": arguments.k_batch}
    if extra:
        print(f"    k_batch = {arguments.k_batch}", flush=True)
    calculator = Calculator.from_file(ROOT / case["input"],
                                      pseudo_dir=arguments.pseudo_dir,
                                      announce=False, **extra)
    if arguments.nosym:
        calculator = _without_symmetry(calculator)
        print("    the crystal's point group is dropped for this run", flush=True)
    if arguments.kmesh:
        calculator = _with_full_grid(calculator, arguments.kmesh)
        print(f"    the whole {arguments.kmesh}^3 grid, "
              f"{calculator.system.kpoints.nk} k-points", flush=True)
    results = {"case": arguments.case, "input": case["input"],
               "what": case["what"], "nppstr": nppstr,
               "transverse": list(transverse), "kmesh": arguments.kmesh,
               "nosym": bool(arguments.nosym), "k_batch": arguments.k_batch,
               "method": arguments.method,
               "nk": int(calculator.system.kpoints.nk)}

    start = time.time()
    scf = calculator.get_scf(conv_thr=arguments.conv_thr)
    results["scf"] = {"energy": float(scf.total_energy),
                      "iterations": int(scf.iterations),
                      "seconds": time.time() - start}
    print(f"    SCF {scf.total_energy:.10f} Ry in {scf.iterations} iterations, "
          f"{results['scf']['seconds']:.1f} s", flush=True)

    reference = None
    if not arguments.skip_response:
        start = time.time()
        tensor = calculator.get_piezoelectric_tensor(method=arguments.method)
        reference = float(tensor.e14)
        results["response"] = {
            "e14": reference,
            "voigt": np.asarray(tensor.voigt).tolist(),
            "converged": bool(tensor.converged),
            "seconds": time.time() - start,
        }
        print(f"    response route ({arguments.method}): "
              f"e_14 = {reference: .6f} C/m^2 "
              f"(converged {tensor.converged}, "
              f"{results['response']['seconds']:.1f} s)", flush=True)

    results["finite_difference"] = []
    for magnitude in ([] if arguments.skip_difference else arguments.shears):
        start = time.time()
        one = finite_difference(
            {"input": str(ROOT / case["input"]), "pseudo_dir": arguments.pseudo_dir,
             "kmesh": arguments.kmesh, "nosym": bool(arguments.nosym),
             "k_batch": arguments.k_batch},
            magnitude, nppstr, transverse, arguments.conv_thr,
        )
        one["seconds"] = time.time() - start
        results["finite_difference"].append(one)
        against = ""
        if reference is not None:
            gap = one["e14"] - reference
            against = (f", difference {gap: .2e} "
                       f"({abs(gap / reference) * 100:.2f} per cent)")
        print(f"    Berry phase at eps_4 = {2 * magnitude:.4f}: "
              f"e_14 = {one['e14']: .6f} C/m^2{against}, "
              f"{one['seconds']:.1f} s", flush=True)
        print(f"      phases +/-: {np.array2string(np.asarray(one['phases_plus']), precision=6)}"
              f" {np.array2string(np.asarray(one['phases_minus']), precision=6)}"
              f", quantum {one['quantum']:.4f}", flush=True)

    if arguments.out:
        Path(arguments.out).write_text(json.dumps(results, indent=1))
        print(f"    wrote {arguments.out}", flush=True)


if __name__ == "__main__":
    main()
