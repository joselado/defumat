#!/usr/bin/env python3
"""Time pypresso against Quantum ESPRESSO on a set of inputs and write a PDF.

Run it by hand whenever a number might have moved::

    python3 performance/run_performance.py                 # the default set
    python3 performance/run_performance.py --all           # everything in benchmarks/
    python3 performance/run_performance.py --cases si-1k si8-1k
    python3 performance/run_performance.py --repeats 3     # best of three, slower

It produces ``performance/performance.tex``, compiles it to
``performance/performance.pdf``, and leaves the raw numbers in
``performance/performance.json`` so two runs can be diffed.

**What it measures.** The project's primary metric: single-core pypresso against
single-core Quantum ESPRESSO, same machine, same input, same ``conv_thr``. The
QE side is QE's own timing report (``init_run``, ``electrons``) rather than a
stopwatch around the process, and both codes are pinned to one CPU by affinity
-- XLA sizes its thread pool from the affinity mask, so without that the
comparison flatters this code by the core count. The measurement code is
``tools/compare_qe.py``; this script drives it over several inputs and typesets
the result.

**Why each case is a subprocess.** It bounds the case two ways at once: a hard
per-case timeout (``--max-seconds``, 120 s by default -- a case that cannot
finish inside it is reported as over budget rather than silently making the run
long), and a fresh process, so one case's compilation cache and memory
high-water mark cannot colour the next. The peak RSS of each case is reported
too, because memory is part of the design here and not an afterthought.

**Cold, warm, and why both are shown.** ``setup`` and the first SCF include XLA
compiling every kernel; the warm SCF re-runs the loop with them compiled. QE has
no equivalent -- there is nothing to compile -- so the honest comparison of the
*arithmetic* is the warm one, and the cold one is what a user meets on a short
run. Both are in the report.

``pw.x`` is expected at ``quantum_espresso/.../bin/pw.x`` (override with
``$PW_X``); build it once with ``./configure --disable-parallel
--disable-openmp && make -j pw``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
BENCHMARKS = ROOT / "benchmarks"

#: The default set: one case per feature, each inside the per-case budget on the
#: machine this was written on. Ordered as the report tabulates them -- growing
#: system size first, then the features that change what an iteration contains.
DEFAULT_CASES = [
    ("si-1k", "Si, 2 atoms, LDA, norm-conserving"),
    ("si-1k-ecut40", "the same cell at a production cutoff"),
    ("si8-1k", "Si, 8 atoms -- where the cost is physics, not overhead"),
    ("si2-us-1k", "ultrasoft: augmentation charge, two grids"),
    ("si2-paw-1k", "PAW: the one-centre terms on top"),
    ("si8-pbe-1k", "a gradient-corrected functional"),
    ("si8-smeared-1k", "8 atoms, smeared -- the collinear half of the spinor pair"),
    ("si8-nc-1k", "the same run as spinors: what npol = 2 costs"),
    ("pt-so-1k", "spin-orbit: j-resolved projectors, complex 2x2 D_ij"),
]



# --------------------------------------------------------------- one case, once


def measure(input_path: Path, repeats: int) -> dict:
    """Run both codes on one input. Called in a subprocess by :func:`collect`.

    Imports ``tools.compare_qe``, which re-executes the interpreter pinned to a
    single core before JAX is imported -- that has to happen at import time,
    which is why this is a mode of the script rather than a function call.
    """
    sys.path.insert(0, str(ROOT))
    from tools import compare_qe

    pw_x = Path(os.environ.get("PW_X", compare_qe.DEFAULT_PW_X))
    conv_thr = compare_qe._conv_thr(input_path)

    started = time.perf_counter()
    qe = compare_qe.run_qe(input_path, pw_x, repeats)
    ours = compare_qe.run_pypresso(input_path, repeats, conv_thr)
    elapsed = time.perf_counter() - started

    ngm, npwx, grid = ours["basis"]
    return {
        "case": input_path.stem,
        "nat": int(ours["nat"]),
        "nk": int(ours["nk"]),
        "ecutwfc": float(ours["ecutwfc"]),
        "ngm": int(ngm),
        "npwx": int(npwx),
        "grid": list(grid),
        "conv_thr": conv_thr,
        "qe_init": qe["init"],
        "qe_scf": qe["scf"],
        "qe_total": qe["total"],
        "qe_iterations": int(qe["iterations"]),
        "qe_energy": qe["energy"],
        "setup": ours["setup"],
        "scf_cold": ours["scf_cold"],
        "scf_warm": ours["scf_warm"],
        "iterations": int(ours["iterations"]),
        "energy": float(ours["energy"]),
        "wall": elapsed,
        "peak_rss_gb": peak_rss_gb(),
    }


def peak_rss_gb() -> float:
    """This process's high-water mark, in GB (``VmHWM``)."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM"):
                return round(int(line.split()[1]) / 1048576.0, 2)
    except OSError:  # pragma: no cover - not Linux
        pass
    return float("nan")


# ------------------------------------------------------------------ the harness


def collect(cases: list[tuple[str, str]], repeats: int, max_seconds: float) -> list[dict]:
    """Each case in its own process, with a hard timeout."""
    results = []
    for name, description in cases:
        input_path = BENCHMARKS / f"{name}.in"
        if not input_path.is_file():
            print(f"  {name}: no such benchmark -- skipped", flush=True)
            continue
        print(f"  {name}: measuring (budget {max_seconds:.0f}s) ...", end=" ", flush=True)
        command = [sys.executable, str(Path(__file__).resolve()),
                   "--measure", str(input_path), "--repeats", str(repeats)]
        try:
            completed = subprocess.run(command, capture_output=True, text=True,
                                       timeout=max_seconds, cwd=ROOT)
        except subprocess.TimeoutExpired:
            print("over budget", flush=True)
            results.append({"case": name, "description": description, "over_budget": True})
            continue
        if completed.returncode != 0:
            print("failed", flush=True)
            print(completed.stdout[-2000:] or completed.stderr[-2000:], file=sys.stderr)
            results.append({"case": name, "description": description, "failed": True})
            continue
        record = json.loads(completed.stdout.strip().splitlines()[-1])
        record["description"] = description
        ratio = record["scf_warm"] / record["qe_scf"] if record["qe_scf"] else float("nan")
        print(f"{record['wall']:.0f}s wall, {ratio:.1f}x QE", flush=True)
        results.append(record)
    return results


def environment() -> dict:
    """What the reader needs in order to know what the numbers describe."""
    model = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:  # pragma: no cover - not Linux
        pass
    sys.path.insert(0, str(ROOT))
    import jax

    from pypresso import __version__ as pypresso_version

    return {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "host": platform.node(),
        "cpu": model or platform.processor(),
        "cores": os.cpu_count(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "pypresso": pypresso_version,
        "qe": qe_version(),
        "commit": git_commit(),
    }


def qe_version() -> str:
    pw_x = Path(os.environ.get("PW_X", ROOT / "quantum_espresso" / "qe-7.5-ReleasePack"
                               / "qe-7.5" / "bin" / "pw.x"))
    if not pw_x.exists():
        return "not built"
    # In a scratch directory, because pw.x fed an empty input writes ``CRASH``
    # and ``input_tmp.in`` into the working directory before it stops.
    try:
        with tempfile.TemporaryDirectory() as scratch:
            out = subprocess.run([str(pw_x)], input="", capture_output=True, text=True,
                                 timeout=30, cwd=scratch).stdout
    except (OSError, subprocess.TimeoutExpired):  # pragma: no cover
        return "unknown"
    match = re.search(r"Program PWSCF\s+(v\.\S+)", out)
    return match.group(1) if match else "unknown"


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):  # pragma: no cover
        return "unknown"


# ----------------------------------------------------------------------- LaTeX


def tex_escape(text: str) -> str:
    for character, replacement in (("\\", r"\textbackslash{}"), ("_", r"\_"),
                                   ("&", r"\&"), ("%", r"\%"), ("#", r"\#")):
        text = text.replace(character, replacement)
    return text


def _ratio(ours: float, theirs: float) -> str:
    if not theirs or theirs != theirs:
        return "--"
    return f"{ours / theirs:.1f}"


def _per_iteration(record: dict) -> tuple[float, float]:
    qe = record["qe_scf"] / max(record["qe_iterations"], 1)
    ours = record["scf_warm"] / max(record["iterations"], 1)
    return qe, ours


def write_tex(results: list[dict], info: dict, path: Path,
              skipped: list[str] | None = None) -> Path:
    rows, cold_rows, chart, notes = [], [], [], []
    for record in results:
        name = tex_escape(record["case"])
        if record.get("over_budget") or record.get("failed"):
            why = "over budget" if record.get("over_budget") else "failed"
            notes.append(f"\\texttt{{{name}}} -- {why}, so it is not in the tables.")
            continue
        qe_iter, our_iter = _per_iteration(record)
        rows.append(
            f"\\texttt{{{name}}} & {record['nat']} & {record['npwx']} & "
            f"{record['qe_scf']:.3f} & {record['scf_warm']:.3f} & "
            f"\\textbf{{{_ratio(record['scf_warm'], record['qe_scf'])}}} & "
            f"{qe_iter:.3f} & {our_iter:.3f} & "
            f"\\textbf{{{_ratio(our_iter, qe_iter)}}} & "
            f"{abs(record['qe_energy'] - record['energy']):.1e} \\\\"
        )
        cold_rows.append(
            f"\\texttt{{{name}}} & {record['qe_init']:.3f} & {record['setup']:.3f} & "
            f"{record['scf_cold']:.3f} & {record['scf_warm']:.3f} & "
            f"{record['qe_iterations']} & {record['iterations']} & "
            f"{record['peak_rss_gb']:.2f} \\\\"
        )
        chart.append((record["case"], _ratio(our_iter, qe_iter)))

    body = TEMPLATE.format(
        date=tex_escape(info["date"]),
        host=tex_escape(info["host"]),
        cpu=tex_escape(info["cpu"]),
        cores=info["cores"],
        python=info["python"],
        jax=info["jax"],
        pypresso=tex_escape(info["pypresso"]),
        qe=tex_escape(info["qe"]),
        commit=tex_escape(info["commit"]),
        rows="\n".join(rows) or "\\multicolumn{10}{c}{no case completed} \\\\",
        cold_rows="\n".join(cold_rows) or "\\multicolumn{8}{c}{no case completed} \\\\",
        chart=" ".join(f"({ratio},{{{case}}})" for case, ratio in reversed(chart)),
        ycoords=",".join(case for case, _ in reversed(chart)),
        chart_height=f"{max(4.0, 0.9 * len(chart) + 1.5):.1f}cm",
        notes=_notes(notes, skipped),
    )
    path.write_text(body)
    return path


def _notes(notes: list[str], skipped: list[str] | None) -> str:
    """What is *not* in the tables, so the report cannot read as full coverage."""
    if not notes:
        notes = ["Every case in the set completed inside its budget."]
    if skipped:
        listed = ", ".join(f"\\texttt{{{tex_escape(name)}}}" for name in sorted(skipped))
        notes.append(f"Benchmarks not run: {listed}. Run them all with "
                     "\\texttt{--all}, or name them with \\texttt{--cases}.")
    return "\n\n".join(notes)


TEMPLATE = r"""\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=2.2cm]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{xcolor}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.17}}
\usepackage{{hyperref}}
\hypersetup{{colorlinks=true,linkcolor=blue!60!black,urlcolor=blue!60!black}}

\title{{pypresso against Quantum ESPRESSO\\[2pt]
\large single core, same machine, same input}}
\author{{generated by \texttt{{performance/run\_performance.py}}}}
\date{{{date}}}

\begin{{document}}
\maketitle

\section*{{What was measured}}

Both codes ran the same \texttt{{pw.x}} input to the same \texttt{{conv\_thr}},
pinned to one CPU core by affinity mask. The Quantum ESPRESSO numbers are QE's
own timing report -- \texttt{{init\_run}} and \texttt{{electrons}} -- not a
stopwatch around the process. The pypresso SCF time is the \emph{{warm}} one:
the loop re-run with every XLA kernel already compiled, which is what predicts a
long run or a larger system. The cold time, which includes compilation, is in
Table~\ref{{tab:cold}} beside it.

\begin{{center}}
\begin{{tabular}}{{ll}}
\toprule
host & \texttt{{{host}}} ({cores} cores, one used) \\
CPU & {cpu} \\
Quantum ESPRESSO & {qe}, serial build \\
pypresso & {pypresso}, commit \texttt{{{commit}}} \\
JAX / Python & {jax} / {python} \\
\bottomrule
\end{{tabular}}
\end{{center}}

\section*{{The comparison}}

\begin{{table}}[h]
\centering
\small
\begin{{tabular}}{{lrr rrc rrc r}}
\toprule
& & & \multicolumn{{3}}{{c}}{{SCF loop (s)}} & \multicolumn{{3}}{{c}}{{per iteration (s)}} & \\
\cmidrule(lr){{4-6}} \cmidrule(lr){{7-9}}
case & atoms & PWs & QE & pypresso & ratio & QE & pypresso & ratio & $\Delta E$ (Ry) \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\caption{{Warm SCF against QE's \texttt{{electrons}}. $\Delta E$ is the
difference in total energy between the two codes -- a correctness check riding
along with the timing, not a performance number.}}
\end{{table}}

\begin{{figure}}[h]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
  xbar, width=0.72\textwidth, height={chart_height},
  xlabel={{pypresso / QE, per SCF iteration}},
  symbolic y coords/.expanded={{{ycoords}}},
  ytick=data, nodes near coords, nodes near coords align=horizontal,
  xmin=0, enlarge y limits=0.12, tick label style={{font=\small}},
  bar width=10pt,
  extra x ticks={{1}}, extra x tick labels={{}},
  extra x tick style={{grid=major, grid style={{gray,dashed}}}},
]
\addplot coordinates {{{chart}}};
\end{{axis}}
\end{{tikzpicture}}
\caption{{Lower is better; 1.0 would be parity with the Fortran.}}
\end{{figure}}

\begin{{table}}[h]
\centering
\small
\begin{{tabular}}{{lrrrr rr r}}
\toprule
& \multicolumn{{2}}{{c}}{{setup (s)}} & \multicolumn{{2}}{{c}}{{SCF (s)}} & \multicolumn{{2}}{{c}}{{iterations}} & \\
\cmidrule(lr){{2-3}} \cmidrule(lr){{4-5}} \cmidrule(lr){{6-7}}
case & QE \texttt{{init\_run}} & pypresso & cold & warm & QE & pypresso & peak RSS (GB) \\
\midrule
{cold_rows}
\bottomrule
\end{{tabular}}
\caption{{\label{{tab:cold}}What compilation costs, and what each case needs in
memory. The cold SCF is the first run of the loop in a fresh process, so it
carries the compilation of every kernel; the warm one is the same loop
afterwards. QE has nothing to compile, so it has no cold/warm distinction.}}
\end{{table}}

\section*{{Notes}}

{notes}

\medskip
\noindent Regenerate with \texttt{{python3 performance/run\_performance.py}}.
The running commentary on where the time goes, and what each change was worth,
is in \texttt{{PERFORMANCE.md}}.

\end{{document}}
"""


def compile_pdf(tex_path: Path) -> Path | None:
    """Two ``pdflatex`` passes, quietly, in the file's own directory."""
    if shutil.which("pdflatex") is None:
        print("  pdflatex not found -- wrote the .tex only", flush=True)
        return None
    for _ in range(2):
        completed = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent, capture_output=True, text=True,
        )
    if completed.returncode != 0:
        print(completed.stdout[-3000:], file=sys.stderr)
        raise SystemExit(f"pdflatex failed on {tex_path}")
    for suffix in (".aux", ".log", ".out"):
        tex_path.with_suffix(suffix).unlink(missing_ok=True)
    return tex_path.with_suffix(".pdf")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--measure", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--cases", nargs="+", metavar="NAME",
                        help="benchmark stems to run (default: the standard set)")
    parser.add_argument("--all", action="store_true",
                        help="every input in benchmarks/, including the slow ones")
    parser.add_argument("--repeats", type=int, default=1,
                        help="repeats per case, best taken (default 1)")
    parser.add_argument("--max-seconds", type=float, default=120.0,
                        help="per-case budget; over it the case is reported, not waited on")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--no-pdf", action="store_true", help="write the .tex and stop")
    parser.add_argument("--from-json", action="store_true",
                        help="re-typeset the last run's numbers without measuring again")
    args = parser.parse_args(argv)

    if args.measure is not None:
        print(json.dumps(measure(args.measure, args.repeats)))
        return 0

    if args.cases:
        cases = [(name, "") for name in args.cases]
    elif args.all:
        known = dict(DEFAULT_CASES)
        cases = [(path.stem, known.get(path.stem, ""))
                 for path in sorted(BENCHMARKS.glob("*.in"))]
    else:
        cases = DEFAULT_CASES

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.from_json:
        stored = json.loads((args.output_dir / "performance.json").read_text())
        results, info = stored["results"], stored["environment"]
        print(f"re-typesetting {len(results)} case(s) measured {info['date']}")
    else:
        print(f"pypresso vs QE, {len(cases)} case(s), single core each")
        results = collect(cases, args.repeats, args.max_seconds)
        info = environment()
        (args.output_dir / "performance.json").write_text(
            json.dumps({"environment": info, "results": results}, indent=2)
        )
    measured = {record["case"] for record in results}
    skipped = sorted(path.stem for path in BENCHMARKS.glob("*.in")
                     if path.stem not in measured)
    tex = write_tex(results, info, args.output_dir / "performance.tex", skipped)
    print(f"  wrote {tex.relative_to(ROOT)}")
    if not args.no_pdf:
        pdf = compile_pdf(tex)
        if pdf is not None:
            print(f"  wrote {pdf.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
