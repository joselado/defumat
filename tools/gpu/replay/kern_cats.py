"""Group an ``nsys stats -r cuda_gpu_kern_sum -f csv`` table into FFT, matrix products, dense solve, elementwise.

    python3 tools/gpu/replay/kern_cats.py <kern_sum.csv>
"""
import csv, sys, collections, re
def load(path):
    lines = [l for l in open(path).read().splitlines() if l.strip()]
    idx = next(i for i, l in enumerate(lines) if l.startswith('"Time (%)"') or l.startswith('Time (%)'))
    return list(csv.DictReader(lines[idx:]))
def cat(name):
    n = name.lower()
    if re.search(r"fft|cufft|regular_fft|vector_fft|bluestein", n): return "FFT"
    if re.search(r"sytrd|ormtr|syevd|stedc|steqr|heevd|hetrd|unmtr|potrf|trsm|trtri|getrf|getrs|orgtr|laswp|larf|syr|lauum|potrs|ungtr|zlarf", n): return "dense eigensolve / factor"
    if re.search(r"gemm|cutlass|ampere|gemv|cublas|dot|axpy|scal_kernel|nrm2|gerc|herk|syrk", n): return "matrix product / BLAS"
    if re.search(r"fusion|loop|reduce|scatter|gather|copy|transpose|concat|select|dynamic|slice", n): return "elementwise / data movement"
    return "other"
rows = load(sys.argv[1])
tk = [k for k in rows[0] if k.startswith("Total Time")][0]
ck = [k for k in rows[0] if k.startswith("Instances") or k.startswith("Count")][0]
nk = [k for k in rows[0] if k == "Name"][0]
tot = collections.defaultdict(float); cnt = collections.defaultdict(int); allt = 0.0
for r in rows:
    t = float(r[tk].replace(",", "")) / 1e6; c = int(r[ck].replace(",", ""))
    k = cat(r[nk]); tot[k] += t; cnt[k] += c; allt += t
print("kernel time %.0f ms over %d launches" % (allt, sum(cnt.values())))
for k, t in sorted(tot.items(), key=lambda kv: -kv[1]):
    print("  %-30s %9.1f ms  %5.1f%%   %7d launches  %.1f us each" % (k, t, 100 * t / allt, cnt[k], 1e3 * t / max(1, cnt[k])))
print("top kernels:")
for r in sorted(rows, key=lambda r: -float(r[tk].replace(",", "")))[:8]:
    print("   %9.1f ms  n=%-7s %s" % (float(r[tk].replace(",", "")) / 1e6, r[ck], r[nk][:100]))
