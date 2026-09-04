"""Run reference inputs through Quantum ESPRESSO and commit the output.

Most of defumat's validation uses the outputs QE already ships in its
``test-suite/``. Ultrasoft and PAW silicon is the case where it does not: no
committed benchmark covers the pseudopotentials used here, so the reference has
to be generated once with the vendored ``pw.x`` and stored next to the input --
which is what `CLAUDE.md` asks for, so that the Fortran build is needed to
*create* a reference and never to *use* one.

    python3 tools/generate_reference.py                 # everything missing
    python3 tools/generate_reference.py --force si2-us  # regenerate one case

The projected density of states is the second case of the same thing and a
sharper one: QE's test-suite has no ``projwfc`` case at all, so *every* reference
for it is generated here (see ``PROJWFC`` below). Those need ``projwfc.x``, which
is not part of a ``make pw`` build -- ``make pp`` inside the vendored tree builds
it, and without it the pdos cases are skipped rather than failed.

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
    # The spin and isolated-atom cases (P9). These have a second reason to be
    # restamped on top of the FFT-grid one: their committed benchmarks are QE
    # 6.1 runs stopped at conv_thr = 1e-6, where the printed energy *terms* are
    # only good to ~1e-4 Ry, and the pw_lsda ones additionally print a total
    # magnetization that is worth comparing at the same threshold as the energy.
    "pw_atom/atom.in",
    "pw_atom/atom-lsda.in",
    "pw_atom/atom-sigmapbe.in",
    "pw_lsda/lsda.in",
    "pw_lsda/lsda-tot_magnetization.in",
    "pw_lsda/lsda-nelup+neldw.in",
    "pw_pawatom/paw-atom_lda.in",
    "pw_pawatom/paw-atom_spin_lda.in",
    "pw_pawatom/paw-atom_spin.in",
    # Spin-orbit coupling (P14). Platinum with a fully-relativistic ultrasoft,
    # PBE-ultrasoft and PAW dataset: the three pseudopotential kinds a spin-orbit
    # run can use, all nonmagnetic, all with the same cell so that what differs
    # between them is only the dataset.
    "pw_spinorbit/spinorbit.in",
    "pw_spinorbit/spinorbit-pbe.in",
    "pw_spinorbit/spinorbit-paw.in",
    # Structural relaxation (P15). QE's own two-atom relax: a CO molecule with
    # the oxygen frozen by ``if_pos``, ultrasoft, at Gamma. Its committed
    # benchmark is a QE 6.1 run, and a relaxation's *final geometry* is what is
    # compared, so it is regenerated with the vendored pw.x like the rest.
    "pw_relax/relax.in",
    # Noncollinear magnetism and constrained moments (P17, P18). bcc iron with
    # its moment tilted, which is the only magnetic noncollinear case QE's
    # suite ships. The constrained ones have a second reason to be regenerated
    # and it is not the FFT grid: ``noncolin-constrain_atomic.in`` carries a
    # commented-out ``lambda = 1`` above the ``lambda = 0.005`` it actually
    # sets, and its committed benchmark prints a constraint energy of 8.022 Ry
    # at the starting density -- which is the *unscaled* sum of squares, i.e.
    # what lambda = 1 gives. The committed output does not belong to the
    # committed input, so comparing against it would be comparing against a run
    # nobody can reproduce.
    "pw_noncolin/noncolin.in",
    "pw_noncolin/noncolin-pbe.in",
    "pw_noncolin/noncolin-constrain_atomic.in",
    "pw_noncolin/noncolin-constrain_angle.in",
    "pw_noncolin/noncolin-constrain_total.in",
    # DFT+U (P20). The four cases of QE's ``pw_lda+U/`` whose pseudopotentials
    # are committed here: antiferromagnetic FeO with a U on both iron
    # sublattices, the same cell with U set to 1e-8 (which must reproduce the
    # plain LSDA run and is the null test), the same with
    # ``starting_ns_eigenvalue`` steering the occupation matrix, and the one
    # that prints forces. Their committed benchmarks stop at 1e-6 to 1e-9, where
    # the printed *terms* -- Hubbard energy included -- are worth about 1e-4 Ry.
    "pw_lda+U/lda+U.in",
    "pw_lda+U/lda+U-noU.in",
    "pw_lda+U/lda+U-user_ns.in",
    "pw_lda+U/lda+U_force.in",
    # The full (Liechtenstein) functional, ``lda_plus_u_kind = 1`` (P62a), on
    # the same FeO cell -- which QE's own case runs at ``J = 1e-12``, so it
    # tests the four-index assembly exactly at the point where it must reduce to
    # the simplified functional and tests ``F^2``/``F^4`` not at all. The
    # ``kind1-J`` case beside it is the same input with a real ``J`` and is
    # generated here because QE's suite has no collinear case that carries one.
    "pw_lda+U/lda+U_kind1_collin.in",
    # Variable-cell relaxation (P29). The four cases of ``pw_vc-relax/`` that
    # use BFGS: rhombohedral arsenic at zero pressure and at 500 kbar, the same
    # at ``nspin = 2``, and the same again with ``treinit_gvecs``, which rebuilds
    # the grids on every accepted step instead of once. ``vc-relax1`` and
    # ``vc-relax2`` are ``cell_dynamics = 'damp-w'`` -- Wentzcovitch damped
    # dynamics, a different optimizer -- and are refused by name rather than run.
    #
    # A relaxation's reference is its **final** geometry and the energy of the
    # final SCF, so it is regenerated with the vendored pw.x for the same reason
    # ``pw_relax/relax.in`` is.
    "pw_vc-relax/vc-relax3.in",
    "pw_vc-relax/vc-relax4.in",
    "pw_vc-relax/vc-relax5.in",
    "pw_vc-relax/vc-relax6.in",
]


def reference_path(case: Path) -> Path:
    return case.with_name(f"reference.out.{case.stem}")


def restamped_path(relative: str) -> Path:
    """Where a regenerated test-suite reference is stored."""
    directory, name = relative.split("/")
    return CASES / f"reference.out.{directory}-{Path(name).stem}"


#: Suffixes marking a case that reads a converged density rather than producing
#: one: ``<stem>-bands.in`` is a band structure on ``<stem>.in``'s density, and
#: ``<stem>-orbm.in`` is the orbital magnetization on it (``lorbm``, a
#: non-self-consistent run over a uniform grid). Both need their parent to have
#: run first *in the same outdir*.
FOLLOWS: tuple[str, ...] = ("-bands", "-orbm")


def prerequisite(case: Path) -> Path | None:
    """The scf run a case needs to have happened first, by naming convention.

    ``<stem>-bands.in`` reads the density ``<stem>.in`` converged, so the two
    have to share an outdir. That is the only kind of dependency between cases
    here, and encoding it in the name keeps the inputs plain ``pw.x`` inputs --
    which they have to stay, since defumat reads the same files.
    """
    for suffix in FOLLOWS:
        if case.stem.endswith(suffix):
            parent = case.with_name(f"{case.stem[: -len(suffix)]}.in")
            if not parent.is_file():
                raise FileNotFoundError(
                    f"{case.name} needs {parent.name}, which is missing"
                )
            return parent
    return None


def run_case(case: Path, conv_thr: float | None = None) -> str:
    """Run one input and return QE's stdout.

    ``conv_thr`` overrides the input's own. It is used for the restamped
    test-suite cases, whose inputs ask for 1e-6: QE then stops with a density
    still wrong in the seventh decimal, and its printed energy *terms* -- which
    are first-order sensitive to the density where the total is second-order --
    are only good to about 1e-4. Comparing a converged defumat against that
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
    # defumat reads unchanged, and neither path is a property of the case.
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
        # In the run directory, not the caller's. Reading its input from
        # standard input, ``pw.x`` copies it to a scratch file named
        # ``input_tmp.in`` **in the working directory**
        # (``Modules/open_close_input_file.f90``), so two runs sharing a
        # directory overwrite each other's input and one of them silently
        # computes the other's system. That is not hypothetical: generating
        # five of these references concurrently produced a hydrogen chain whose
        # stdout was silicon's, and the failure only surfaced at all because
        # the crossed input had a different ``ATOMIC_SPECIES`` count.
        cwd=tmp,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        timeout=3600,
    )
    if "JOB DONE" not in result.stdout:
        raise RuntimeError(f"{case.name}: pw.x did not finish\n{result.stdout[-2000:]}")
    return result.stdout



# --------------------------------------------------------------------------
# projwfc.x
# --------------------------------------------------------------------------
#
# The projected density of states is the one place where QE's test-suite has
# nothing at all: there is no ``pp_projwfc`` case and no committed ``filpdos``
# file anywhere in the tree. So its references are generated here, exactly as
# the ultrasoft and PAW ones are, and stored beside the inputs -- the Fortran
# build is needed to *create* them and never to use them.
#
# Each case is a ``pw.x`` run followed by a ``projwfc.x`` run **in the same
# outdir**, because that is how the tool works: it reads the wavefunctions the
# SCF left behind rather than solving anything. What is stored is the standard
# output (the state table, the per-band projections, the Löwdin charges and the
# spilling parameter) and every ``filpdos`` file it wrote.

#: ``DeltaE`` in eV, ``projwfc.x``'s own units for it. 0.05 rather than its
#: default 0.01 keeps the committed files to a few hundred lines apiece; with no
#: ``degauss`` given, the default broadening *is* ``DeltaE``, so this also sets
#: the width the comparison runs at.
PROJWFC_DELTA_E = 0.05

#: ``<stem> -> input``: a path relative to ``test-suite/``, or a bare stem for
#: one of the dedicated inputs in ``tests/data/qe``. Between them they cover
#: every path through the projection -- norm-conserving with fixed occupations,
#: ultrasoft and PAW (where ``S`` is not the identity and even the "atomic"
#: projectors are ``S phi``), two spin channels with a smearing, and both
#: tetrahedron families: the optimised one, which the projection uses as it
#: stands, and Bloechl's, which ``do_projwfc`` silently replaces by the linear
#: method.
PROJWFC = {
    "pw_scf-scf": "pw_scf/scf.in",
    "si2-us": "si2-us",
    "si2-us-dense": "si2-us-dense",
    "si2-paw": "si2-paw",
    "pw_lsda-lsda": "pw_lsda/lsda.in",
    "pw_metal-metal-tetrahedra": "pw_metal/metal-tetrahedra.in",
    "al-tetrahedra": "al-tetrahedra",
    # Ten sites, where the projection has ten atoms to resolve rather than two
    # and ``fill_nlmchi``'s ordering is the thing being compared as much as the
    # numbers are.
    "si10-nc": "si10-nc",
    # Spin-orbit coupling, where the projection is onto the spin-angle functions
    # |l j m_j> rather than onto real harmonics, and ``natomwfc`` is not a
    # doubling. Ultrasoft and PAW, because ``S`` carries ``qq_so`` in both and a
    # norm-conserving case cannot see it.
    "pt-soc": "pt-soc-nosym",
    "pt-soc-paw": "pt-soc-paw-nosym",
}

#: Cases whose projection must **not** be symmetrised, because defumat refuses a
#: symmetrised spinor projection (``sym_proj_so`` needs the SU(2) representation
#: of each operation) and offers ``nosym`` as the way out.
#:
#: **``nosym`` in the ``pw.x`` input is not enough and this was measured.**
#: ``projwfc.x`` re-derives the point group rather than inheriting the one the
#: SCF ran with, so ``lsym = .true.`` symmetrises even where ``pw.x`` printed
#: "No symmetry found": on ``pt-soc`` it averages the two Kramers partners of the
#: ``6S`` shell to 0.495 each where the unsymmetrised projection is 0.870/0.120.
#: The Loewdin charges and the spilling are identical either way, so nothing that
#: is summed over a shell can catch it -- only the per-column projection can.
PROJWFC_NOSYM = {"pt-soc", "pt-soc-paw"}

#: Cases that ask ``projwfc.x`` for a broadening of their own, in **Ry** --
#: which is the unit its ``degauss`` is in, unlike its ``DeltaE``. Without one
#: the default is ``DeltaE`` itself, which on a 29-point wedge resolves every
#: eigenvalue into its own spike; 0.0147 Ry is 0.2 eV, which is what a density
#: of states is usually plotted with. Notebook 15 is the consumer.
PROJWFC_DEGAUSS = {"si2-us-dense": 0.0147}

PROJWFC_X = Path(os.environ.get("PROJWFC_X", QE_ROOT / "bin" / "projwfc.x"))


def projwfc_reference(stem: str) -> Path:
    """Where one case's ``projwfc.x`` standard output is stored."""
    return CASES / f"reference.projwfc.{stem}"


def projwfc_case(source: str) -> Path:
    """The ``pw.x`` input a projwfc case runs first."""
    if "/" in source:
        return QE_ROOT / "test-suite" / source
    return CASES / f"{source}.in"


def run_projwfc(case: Path, conv_thr: float | None, degauss: float | None = None,
                lsym: bool = True) -> tuple[str, dict]:
    """``pw.x`` then ``projwfc.x`` in one directory; stdout and the pdos files."""
    with tempfile.TemporaryDirectory() as tmp:
        _invoke(case, tmp, conv_thr)
        namelist = Path(tmp) / "projwfc.in"
        namelist.write_text(
            "&projwfc\n"
            f"    outdir = '{tmp}'\n"
            "    prefix = 'pwscf'\n"
            "    filpdos = 'pdos'\n"
            f"    DeltaE = {PROJWFC_DELTA_E}\n"
            + (f"    degauss = {degauss}\n" if degauss else "")
            + ("" if lsym else
               # ``kresolveddos`` because ``projwfc.f90:229`` writes the partial
               # densities of states only ``IF ( lsym .OR. kresolveddos )``, and
               # ``filproj`` because the projections in the standard output are
               # rounded to three decimals, where the file carries ten -- which
               # is what a column-by-column comparison needs.
               "    lsym = .false.\n"
               "    kresolveddos = .true.\n"
               "    filproj = 'proj'\n")
            + "/\n"
        )
        result = subprocess.run(
            [str(PROJWFC_X)],
            stdin=namelist.open(),
            capture_output=True,
            text=True,
            cwd=tmp,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
            timeout=3600,
        )
        if "JOB DONE" not in result.stdout:
            raise RuntimeError(
                f"{case.name}: projwfc.x did not finish\n{result.stdout[-2000:]}"
            )
        files = {
            path.name: path.read_text()
            for path in sorted(Path(tmp).glob("pdos.pdos_*"))
        }
        files.update({
            path.name: path.read_text()
            for path in sorted(Path(tmp).glob("proj.projwfc_*"))
        })
        return result.stdout, files


def generate_projwfc(wanted: set, force: bool) -> None:
    """Run every projwfc case whose reference is missing."""
    if not PROJWFC_X.is_file():
        print(f"  projwfc.x not found at {PROJWFC_X}; skipping the pdos cases"
              " (build it with 'make pp' inside the vendored tree)")
        return
    for stem, source in PROJWFC.items():
        if wanted and stem not in wanted:
            continue
        case = projwfc_case(source)
        out = projwfc_reference(stem)
        if not case.is_file():
            print(f"  {stem}: input not present (QE tree absent?), skipped")
            continue
        if out.is_file() and not force:
            print(f"  {stem}: already generated")
            continue
        print(f"  {stem}: running pw.x + projwfc.x ...", flush=True)
        stdout, files = run_projwfc(
            case, RESTAMPED_CONV_THR, PROJWFC_DEGAUSS.get(stem),
            lsym=stem not in PROJWFC_NOSYM,
        )
        out.write_text(stdout)
        for name, text in files.items():
            # ``pdos.pdos_atm#1(Si)_wfc#1(s)`` keeps its own name behind the
            # reference prefix, so what each file holds stays readable.
            (CASES / f"reference.{stem}.{name[len('pdos.'):]}").write_text(text)
        spilling = re.search(r"Spilling Parameter:\s*(\S+)", stdout)
        print(f"  {stem}: spilling {spilling.group(1) if spilling else '?'}"
              f" -> {out.name} + {len(files)} pdos files")


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
    known = ({c.stem for c in inputs} | {Path(rel).stem for rel in RESTAMPED}
             | set(PROJWFC))
    if wanted - known:
        print(f"no such case: {', '.join(sorted(wanted - known))}", file=sys.stderr)
        return 1

    todo = [(case, reference_path(case), None) for case in inputs]
    todo += [
        (QE_ROOT / "test-suite" / rel, restamped_path(rel), RESTAMPED_CONV_THR)
        for rel in RESTAMPED
        if not wanted or Path(rel).stem in wanted
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

    generate_projwfc(wanted, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
