# Open items, carried forward

Two things came out of the 2026-09-10 regression run that are real and are not
this-session work. Neither is a wrong physical answer; one is a test asserting a
bound the solver never promised, the other is a memory peak sitting one bad
allocation away from an out-of-memory kill.

**The evidence is not repeated here.** Both are recorded with their numbers in
`PLAN.md` §3, in the P73 section, under "Three test failures were seen while
validating this phase". This file is only what to do about them, what it costs,
and how to know it worked. `GAPS.md` is not the place for either: that file is a
list of what a user can ask for and not get, and both of these are internal.

---

## 1. The Kramers bound asserts round-off where the solver promises `empty_ethr`

`tests/regression/test_spinorbit.py::test_kramers_degeneracy_survives_spin_orbit`
fails on `spinorbit.in` (6.5e-5 eV) and `spinorbit-pbe.in` (7.7e-5 eV) against a
bound of 1e-6 eV over every band. **The code is right and the test is wrong.**
Every band carrying weight is degenerate to 5e-12 eV; the whole failure is in
three bands whose occupation is 1e-87, more than 10 eV above `E_F`, which
`cegterg.f90:129` converges only to `empty_ethr = MAX(5 ethr, 1e-5)` Ry. With
`diago_full_acc = .true.` the same case gives 2.6e-11 eV.

**What to write.** Not "assert over the occupied bands only" — that gives up the
guard exactly where a non-Hermitian `D` or a mispaired spin block is least likely
to be noticed, which is the whole reason the test exists. Two bounds, matching
what the solver actually promises:

- pairs with `wg/wk >= 0.01` (QE's own `btype` rule, `sum_band.f90:118-128`):
  degenerate to round-off, and 1e-6 eV is the right number — measured 5e-12;
- every other pair: degenerate to `empty_band_threshold(ethr)` converted to eV,
  which at the floor is 1e-5 Ry = 1.36e-4 eV — measured 6.5e-5.

Take the occupations from the `SCFResult` the test already has rather than
re-deriving which bands are empty.

**Why it is not a one-liner.** It changes a P14 claim, so the commit that makes
the change says so and P14's entry in `PLAN.md` gains the two numbers. Worth
checking in the same pass whether any *other* test asserts an eigenvalue
degeneracy or difference over all bands without splitting occupied from empty --
the P10 continuation of 2026-09-08 wrote down that empty eigenvalues can now move
by `max(5 ethr, 1e-5)` Ry, and this is the first test found to have been written
against the old behaviour. It is a **slow** test, so the push gate could not have
caught it, and neither could anything else written the same way.

**Cost.** Minutes to write; about 2 minutes to verify, since only the two
non-PAW cases are involved:

```bash
python3 -m pytest tests/regression/test_spinorbit.py \
    -k test_kramers_degeneracy_survives_spin_orbit -q
```

---

## 2. `test_spinorbit.py` peaks at 11,088 MB against a 12 GB cap

92% of the cap on a 30 GB machine, so it is the next out-of-memory kill whether or
not it has happened yet. The watchdog named
`test_spin_orbit_total_energy[spinorbit-paw.in]` at 10,818 MB; **its assertions
passed** -- the memory is the finding, not a failure. For scale, the next file
down in the same run is `test_stress.py` at 6,317 MB.

**The one lead, and what it is not.** The stderr shows XLA constant-folding and
transposing an `f64[25,1277,34,34]` inside `jvp(jit(_paw_onecenter))`, twice,
each fold over 2 s. That shape is `PawSpecies.density_ae`/`density_ps`,
`(nh, nh, nlm, mesh)` at `nh = 34` for a fully-relativistic platinum dataset --
295 MB each. They are ordinary pytree fields, so inside `_paw_onecenter` they are
arguments; appearing as XLA *constants* means the enclosing `jit(<lambda>)`
closes over the object holding them, which would give every compiled variant its
own copy in a cache that never shrinks. **But 590 MB is under 6% of an 11 GB
peak.** The stderr says where the compile time goes. It does not say where the
resident set goes, and starting from "the constant is the memory" is chasing 6%.

**So the first step is a measurement, not a fix.** Two that cost little:

- `jit(f).lower(*ShapeDtypeStructs).compile().memory_analysis()` on the PAW
  spinor gradient at that case's shapes -- it runs the compiler and allocates
  nothing, the same route P73 used for the slab it could not run;
- the same test alone with the watchdog's own sampling, and the scope's
  `memory.current` beside `psutil`'s RSS, since the cgroup charges page cache and
  child processes and this process's anonymous pages are not the same number.

**Then the two candidates worth separating**, because the fixes are different: a
single large allocation inside one backward pass (the augmentation table
`Q_ij(G)` at `nh^2 x ngm` per atom is the known one, and `nh` is 34 here), versus
accumulation of XLA executables across the file's 28 tests. `jax.clear_caches()`
in an autouse fixture distinguishes them in one run -- if the peak drops, it is
accumulation.

**Cost.** The measurement is one 8-minute file at 11 GB. Run it **alone** --
nothing else on the machine, nothing else being timed -- and through
`tools/run_regression.sh` so a kill costs the file rather than the session.

---

## Neither of these is P73

Both reproduce on a worktree at `e22aa7d`, the commit before the P73 augmentation
work started, and item 1 reproduces there to four significant figures. The stress
`DID NOT WARN` failure from the same run is likewise pre-existing and is settled
in `PLAN.md`; it is not repeated here because it needs no decision, only a fix to
P11's refusal.
