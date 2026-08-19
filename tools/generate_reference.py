"""Run reference inputs through Quantum ESPRESSO and commit the output.

Most of pypresso's validation uses the outputs QE already ships in its
``test-suite/``. Ultrasoft and PAW silicon is the case where it does not: no
committed benchmark covers the pseudopotentials used here, so the reference has
to be generated once with the vendored ``pw.x`` and stored next to the input --
which is what `CLAUDE.md` asks for, so that the Fortran build is needed to
*create* a reference and never to *use* one.

    python3 tools/generate_reference.py                 # everything missing
    python3 tools/generate_reference.py --force si2-us  # regenerate one case

It also re-runs a short list of inputs that *do* have a committed benchmark, for
a narrower reason: those benchmarks were produced with QE 6.0 (the suite records
``REFERENCE_VERSION 6.0``) and QE has since changed the FFT grid it chooses for a
non-symmorphic crystal -- the dimensions must now be a multiple of the
denominators of the fractional translations, so diamond silicon's 15^3 grid
became 16^3. The exchange-correlation energy is evaluated pointwise on that
grid, so the total energy moved in the sixth decimal. Comparing against the
committed number would hold this code to a version of QE that is not the one
vendored here; comparing against a regenerated one holds it to the code it is a
reimplementation of. Verified directly: running the vendored ``pw.x`` on
``pw_scf/scf.in`` prints a 16^3 grid where the committed benchmark prints 15^3.

The stored file is QE's stdout with the run's absolute paths and timings left in
place: they are noise, but editing a reference output is a worse habit than
carrying a few irreproducible lines in it.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CASES = REPO / "tests" / "data" / "qe"
PSEUDO = REPO / "tests" / "data" / "pseudo"
QE_ROOT = REPO / "quantum_espresso" / "qe-7.5-ReleasePack" / "qe-7.5"
PW_X = Path(os.environ.get("PW_X", QE_ROOT / "bin" / "pw.x"))

#: Test-suite inputs whose committed benchmark predates QE's fractional-
#: translation constraint on the FFT dimensions, and is therefore stale in the
#: sixth decimal of the total energy. Regenerated here and stored as
#: ``reference.out.<directory>-<stem>``.
#: The restamped cases are re-run at this threshold rather than the 1e-6 their
#: inputs ask for; see ``run_case``.
RESTAMPED_CONV_THR = 1.0e-10

RESTAMPED = [
    "pw_scf/scf.in",
    "pw_scf/scf-kauto.in",
    "pw_scf/scf-kcrys.in",
    "pw_scf/scf-k0.in",
    "pw_scf/scf-occ.in",
]


def reference_path(case: Path) -> Path:
    return case.with_name(f"reference.out.{case.stem}")


def restamped_path(relative: str) -> Path:
    """Where a regenerated test-suite reference is stored."""
    directory, name = relative.split("/")
    return CASES / f"reference.out.{directory}-{Path(name).stem}"


def prerequisite(case: Path) -> Path | None:
    """The scf run a case needs to have happened first, by naming convention.

    ``<stem>-bands.in`` reads the density ``<stem>.in`` converged, so the two
    have to share an outdir. That is the only dependency between cases here, and
    encoding it in the name keeps the inputs plain ``pw.x`` inputs -- which they
    have to stay, since pypresso reads the same files.
    """
    if not case.stem.endswith("-bands"):
        return None
    parent = case.with_name(f"{case.stem[: -len('-bands')]}.in")
    if not parent.is_file():
        raise FileNotFoundError(f"{case.name} needs {parent.name}, which is missing")
    return parent


def run_case(case: Path, conv_thr: float | None = None) -> str:
    """Run one input and return QE's stdout.

    ``conv_thr`` overrides the input's own. It is used for the restamped
    test-suite cases, whose inputs ask for 1e-6: QE then stops with a density
    still wrong in the seventh decimal, and its printed energy *terms* -- which
    are first-order sensitive to the density where the total is second-order --
    are only good to about 1e-4. Comparing a converged pypresso against that
    measures QE's stopping point, not the physics. The dedicated inputs under
    ``tests/data/qe`` and ``benchmarks`` already ask for 1e-10 for the same
    reason; this brings the borrowed ones to the same footing.
    """
    with tempfile.TemporaryDirectory() as tmp:
        before = prerequisite(case)
        if before is not None:
            _invoke(before, tmp, RESTAMPED_CONV_THR)
        return _invoke(case, tmp, conv_thr)


def _invoke(case: Path, tmp: str, conv_thr: float | None) -> str:
    """One ``pw.x`` run in an existing directory."""
    # pseudo_dir and outdir are injected rather than written into the
    # committed input: the input has to stay a plain pw.x input that
    # pypresso reads unchanged, and neither path is a property of the case.
    text = case.read_text()
    text = re.sub(
        r"(&control\b)",
        f"\\1\n    pseudo_dir = '{PSEUDO}'\n    outdir = '{tmp}'",
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    if conv_thr is not None:
        text = re.sub(r"^\s*conv_thr\s*=.*$", "", text, flags=re.IGNORECASE | re.M)
        text = re.sub(
            r"(&electrons\b)",
            f"\\1\n    conv_thr = {conv_thr:.1e}",
            text,
            count=1,
            flags=re.IGNORECASE,
        )
    stdin = Path(tmp) / "pw.in"
    stdin.write_text(text)
    result = subprocess.run(
        [str(PW_X)],
        stdin=stdin.open(),
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        timeout=3600,
    )
    if "JOB DONE" not in result.stdout:
        raise RuntimeError(f"{case.name}: pw.x did not finish\n{result.stdout[-2000:]}")
    return result.stdout


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", help="case stems, e.g. si2-us (default: all)")
    parser.add_argument("--force", action="store_true", help="regenerate existing references")
    args = parser.parse_args(argv)

    if not PW_X.is_file():
        print(f"pw.x not found at {PW_X}; build it or set $PW_X", file=sys.stderr)
        return 1

    wanted = set(args.cases)
    inputs = sorted(c for c in CASES.glob("*.in") if not wanted or c.stem in wanted)
    if wanted and len(inputs) != len(wanted):
        missing = wanted - {c.stem for c in inputs}
        print(f"no such case: {', '.join(sorted(missing))}", file=sys.stderr)
        return 1

    todo = [(case, reference_path(case), None) for case in inputs]
    if not wanted:
        todo += [
            (QE_ROOT / "test-suite" / rel, restamped_path(rel), RESTAMPED_CONV_THR)
            for rel in RESTAMPED
        ]

    for case, out, conv_thr in todo:
        if not case.is_file():
            print(f"  {case.name}: input not present (QE tree absent?), skipped")
            continue
        if out.is_file() and not args.force:
            print(f"  {case.stem}: already generated")
            continue
        print(f"  {case.stem}: running pw.x ...", flush=True)
        out.write_text(run_case(case, conv_thr))
        energy = re.search(r"^!\s+total energy\s+=\s+(\S+)", out.read_text(), re.M)
        print(f"  {case.stem}: total energy {energy.group(1) if energy else '?'} Ry -> {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
