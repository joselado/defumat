# GPU.md Phase 5: the raw records

One JSON per (side, case, property), written by `tools/gpu/phase5.py`. The
tables built from them are in `PERFORMANCE.md` under "What each response
property's working set is" (the CPU tape, measured on the development
workstation) and "The response path on a GPU" (this pair of cluster jobs).

* `gpu-phase5-19951850-*.json` — one NVIDIA H200, `gpu63`, driver 580.173.02,
  jax 0.11.1, four host cores;
* `cpu-phase5-19951851-*.json` — four EPYC Milan cores, `milan1`, same jax and
  same commit, which is §2.3's baseline pinned to a stated core count.

Both jobs ran at commit `e562427` and passed **no** batching dials, so the
`dials` block in each record is also the production test of the per-platform
default: `None/None` on the device, `1/1` on the CPU node.

`alas-raman.raman` has no CPU record. That row died with a `MemoryError` on the
CPU node and the failure is unexplained — see `PERFORMANCE.md`.
