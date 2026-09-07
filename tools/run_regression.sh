#!/usr/bin/env bash
# Run the regression suite one file at a time, each inside a memory-capped
# cgroup, appending a durable summary line per file. A kill loses at most the
# file in flight, not the whole run -- rerun and it resumes, because a file
# already named in the summary is skipped. A file the cap *killed* is the one
# exception and is retried, since the reason to resume after one is that
# something changed.
#
# It exists because the suite is 1477 tests and hours long, so it was not being
# run: three separate phases' claims had drifted unnoticed by the time P38
# looked (P29's refusal list, P36's wedge number, and a `relax_spiral_q` that
# raised before taking a step). `python3 -m pytest -m regression` in one go is
# the same tests and is the right thing when you can wait for it.
#
# It runs the **slow** set by default, because that is the one that needs this
# treatment: the fast group is `tools/test-fast.sh` and finishes in one go.
#
#   tools/run_regression.sh [output-directory] [marker-expression] [file-glob]
#
# **The cap is the point of the wrapper, not a detail of it.** A file that goes
# over the machine's memory used to take the runner, the terminal and every
# uncommitted thing in it -- three times on this machine (P28b, P46, and a
# session on 2026-09-07 that left no record of what it had been running). The
# per-file process boundary only bounds the *cost* of a kill if something else
# survives to write down that it happened, so each file runs in a transient
# scope with `MemoryMax`: the kernel kills that scope, this loop notices exit
# 137, writes `killed` as the file's durable line, and goes on to the next one.
# Override the cap with `DEFUMAT_TEST_MEM_MAX` (systemd syntax, e.g. `20G`);
# `DEFUMAT_TEST_MEM_MAX=off` runs uncapped, which is what a machine without
# cgroup delegation gets anyway, with a warning.
#
# Two things the cap is not. It is not `ulimit -v`, which bounds *virtual*
# address space and fails tests needing a couple of GB because XLA reserves
# arenas far larger than it resides in (`PERFORMANCE.md`). And it is not
# `XLA_PYTHON_CLIENT_MEM_FRACTION`, which sizes the PJRT *device* pool and
# bounds nothing on the CPU backend. What a resident cap costs is nothing: the
# eight `test_scf.py` energy comparisons pass under `MemoryMax=4G` at 1.0 GB
# peak RSS.
cd "$(dirname "$0")/.."
OUT=${1:-regression-results}
MARK=${2:-slow}
GLOB=${3:-tests/regression/test_*.py}
MEMMAX=${DEFUMAT_TEST_MEM_MAX:-12G}
mkdir -p "$OUT"
SUMMARY=$OUT/summary.txt
INFLIGHT=$OUT/in-flight.log
# The end marker is rewritten rather than appended, so a resumed run does not
# leave a trail of them through the file.
[ -f "$SUMMARY" ] && { grep -v "^ALL FILES DONE$" "$SUMMARY" > "$SUMMARY.tmp"; mv "$SUMMARY.tmp" "$SUMMARY"; }

# Is a memory-capped scope available? Delegation of the memory controller to the
# user slice is what decides it, and it is absent on plenty of machines, so this
# is a probe rather than an assumption. `true` allocates nothing; what is being
# tested is whether systemd accepts the property at all.
CAPPED=1
if [ "$MEMMAX" = "off" ]; then
    CAPPED=0
    echo "run_regression: uncapped by request (DEFUMAT_TEST_MEM_MAX=off)" >&2
elif ! systemd-run --user --quiet -p MemoryMax=64M --scope true >/dev/null 2>&1; then
    CAPPED=0
    echo "run_regression: no cgroup memory cap available -- an out-of-memory" >&2
    echo "  kill will take this runner with it. Run fewer files at a time." >&2
fi

for f in $GLOB; do
    name=$(basename "$f" .py)
    if grep -q "^$name " "$SUMMARY" 2>/dev/null; then
        # A *killed* file is retried rather than skipped: the reason to resume
        # after one is that something changed -- a raised cap, one run instead
        # of two. If nothing did it will be killed again, which is what the
        # line already said. Drop the old line so the new one is the record.
        grep -q "^$name .*killed (SIGKILL" "$SUMMARY" || continue
        grep -v "^$name " "$SUMMARY" > "$SUMMARY.tmp" && mv "$SUMMARY.tmp" "$SUMMARY"
    fi
    # Say what is running *before* running it. A kill that gets past the cap --
    # the whole terminal going, an OOM in this loop rather than in the child --
    # otherwise leaves no trace of which file it was in, which is exactly what
    # made the 2026-09-07 session unrecoverable.
    echo "$name started $(date -Is)" >> "$INFLIGHT"
    if [ "$CAPPED" -eq 1 ]; then
        # The scope holds systemd-run and its descendants, not this loop, so the
        # kernel's kill lands there and the loop lives to record it. The unit
        # name carries this run's PID because a killed scope stays *loaded* --
        # reusing the name on a retry fails to start with "already loaded" and
        # looks like a test failure. `reset-failed` clears it afterwards; the
        # name itself is in the file's log, on systemd-run's own first line.
        unit="defumat-reg-$name-$$"
        systemd-run --user --unit="$unit" \
            -p MemoryMax="$MEMMAX" -p MemorySwapMax=0 --scope \
            python3 -m pytest "$f" -q -m "$MARK" --tb=line > "$OUT/$name.log" 2>&1
        status=$?
        systemctl --user reset-failed "$unit.scope" >/dev/null 2>&1
        (exit $status)
    else
        python3 -m pytest "$f" -q -m "$MARK" --tb=line > "$OUT/$name.log" 2>&1
    fi
    status=$?
    # pytest exits 5 when a file holds nothing matching the marker, which is a
    # normal outcome here and not a failure.
    [ "$status" -eq 5 ] && status=0
    line=$(tail -3 "$OUT/$name.log" | grep -E "passed|failed|error|no tests|deselected" | tail -1)
    # 137 is SIGKILL. Under a cap that is the cap, near enough to act on -- but
    # the journal is unreadable without the `systemd-journal` group here, so the
    # cause is not confirmed and the line does not claim it is.
    if [ "$status" -eq 137 ]; then
        line="killed (SIGKILL, cap=$MEMMAX) -- rerun this file alone or raise DEFUMAT_TEST_MEM_MAX"
    fi
    echo "$name exit=$status | $line" >> "$SUMMARY"
    echo "$name finished $(date -Is) exit=$status" >> "$INFLIGHT"
done
echo "ALL FILES DONE" >> "$SUMMARY"
