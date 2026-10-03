# What is left to do about GPU memory, sized

A to-do list, written 2026-09-28 on branch `gpu-memory-modes`, right after `memory_mode`
landed (`batching.py`, `scf/streaming.py`; `PERFORMANCE.md` "Two memory modes"). It says
what still costs device memory, what each item scales with, what the fix is, and how to
measure the win on the GTX 1060 in this workstation. `MEMORY-AUDIT.md` is the older,
broader audit; where an item here is already recorded there, its id is given, and nothing
already fixed is listed.

**Where it came from.** A read-only survey: six agents, one per subsystem (SCF core,
derivatives, response, post-SCF workflows, setup and resident arrays, GPU runtime), then
two skeptical verifiers that opened every cited line, checked every size, marked
duplicates and dropped what did not survive. 45 candidates went in; 39 survived and
merge into the 26 items below. Nothing was executed during the survey.

**How to read the numbers.** *Measured* means taken on the GTX 1060 (jax 0.11.1) or with
`memory_analysis()`, and says so. Everything else is **arithmetic on shapes read from the
code**, on a named cell. The standing cells are: eight-atom Si at 20 Ry with `nosym` (the
memory-mode scan), `bn-ldau-noncol.in` (the stress that dies), `pt-soc-paw-nosym.in` and
`bismuthene-soc-small.in` (peaks set by k-independent objects), nbse2 (`nk = 43`,
`npwx = 9804`, `nbnd = 24`, many k) and the 45-atom NiBr2 slab (`nk = 6`, `nbnd = 403`,
`npol = 2`, `ndim = 312692`, smooth box 200x240x54, `natomwfc = 510`).

**Rule for anyone picking one up**: `CLAUDE.md`'s *an identity that closes is not
evidence; an A/B is*. Several fixes below are "remat this" or "chunk that", and
`MEMORY-AUDIT.md` A4 and A11 are two prescribed fixes that measurement refused. Measure
the peak before and after, one run per fresh process (`peak_bytes_in_use` has no reset),
with the compile cache warm (a miss costs *more* device memory on this card).

## Where the next session starts (written 2026-10-03 evening, master at `d227efe`; items 1 to 3 updated later that evening on branch `bessel-jvp`)

**Item 2 is done**: the field response, the Born charges, both phonons, the piezoelectric tensor, the
strain response, the elastic constants, the two third derivatives (electrostriction and Raman) and the
vibrational spectrum walk the k axis with their stores in host memory, and the radial transforms' chunk is
sized from the mesh, which took the strain derivatives' global step off the card; the numbers are the
"Done since" entries of 2026-10-03 and the matching `PERFORMANCE.md` entries. The regressions are
`tests/regression/test_streamed_{response,phonons,piezo,strain,third}.py` (slow set). Every piece was
planned in writing, reviewed by Fable against the code before it was written, and validated as a route
identity against the whole-k route on the same states.

**What to pick up, in order:**

1. **The two chunks under a strain's derivatives: both halves are done** (2026-10-03, later the same
   evening, branch `bessel-jvp`). Every radial transform's derivative in `|q|` is the same transform one
   order up, given to JAX as a `custom_jvp` (`formfactors.bessel_transform`), so no derivative holds the
   radial kernel and the radial chunk no longer trades memory against time below its default: on the card
   the ultrasoft AlAs piezoelectric tensor 198.5 -> 169.3 MB (its SCF's), the eight-atom silicon elastic
   constants 127.7 -> 45.0 (the strain response's), spin-orbit PAW platinum's stress 516.4 -> 274.8 (its
   SCF's; the old record had put that peak on the one-centre terms, wrongly), every result the same to the
   printed digits; `PERFORMANCE.md`, "The radial transforms' derivatives, as transforms". **The
   augmentation half is done too** (the same evening): the scanned table no longer forms `Q_ij(G)`;
   `becsum` meets the angular coefficients in the radial basis once, and a block of G holds the radial
   table and the harmonics (`augmentation._tabulated_charge`, `_tabulated_integrals`, `_beta_basis`;
   reviewed by Fable, who ran it against the block form on four kinds of dataset). The second derivative
   of AlAs's augmented density compiles to 67.7 MB where it took 176.1, and on the card the ultrasoft
   eight-atom strain response peaks at 140.7 MB where it read 180.8 (74 MB over its SCF where it was
   114); `PERFORMANCE.md`, "The scanned augmentation charge in the radial basis". **Left**: the
   augmentation chunk is still sized from `nh` for the block it no longer forms, so it is conservative,
   and re-keying it on `nbeta^2 nl` is a time trade not measured (the user's). **The strain response's
   remaining 74 MB is located**: its `finish_moved` pass (82.9 MB of temporaries on the card; the `bare`
   walk 59.9 is next), and in it the augmentation charge's forward derivative along the strain, 71.4 of
   86.1 MB compile only (D22's CPU, `review/bessel/moved_terms.py`). With eight atoms of one species the
   factored body's `(nat_t, 2L+1, chunk)` intermediate is about the old block's size (72 against 64
   complex values a G vector), so what it bought there (180.8 -> 140.7 MB) is fewer arrays alive at once,
   not smaller ones, and the 16384-vector chunk sized for the old block sets what is left: 71.4 / 42.8 / 71.5 MB at 16384 / 4096 / 1024 vectors, not monotonic, the same U the
   stress showed. Contracting one atom at a time inside the block was tried and is a null (74.1 and 43.7
   MB against 71.4 and 42.8), so the arrays that make up the rest are not located; re-keying the chunk is
   the lever left, a time trade.
2. **The per-mode grids with symmetry on** stay on the card, because
   `symmetrize_atom_displacement` acts on the whole `(3 nat, ...)` stack; the average's `nsym`-fold
   transient over them is gone (`90ed76a`). What is left, measured by a stage probe on eight-atom silicon
   with 24 operations (`review/piezo/phonon_stages.py` on D22, the warm second run): 39.0 MB after the
   SCF, 53.1 after the first iteration's solves (the 24 response densities), 72.9 after the first
   average (the stack in G and its transforms), then a creep of 1 to 5 MB an iteration to 84.7 at the
   end, which is not located; the probe places it in the `symmetrize_atom_displacement` calls (each
   iteration's `respond` adds nothing), so the eager `r_to_g`/`g_to_r` around the walked average are the
   first place to look. Per orbit of equivalent atoms, or on the host, is the lever for a large
   symmetric cell; the creep is the first thing to look at. **The creep is located, and it is not a
   leak** (2026-10-03 evening, D22, `review/bessel/phonon_inuse.py`, the card's `bytes_in_use` beside its
   peak after each stage): the bytes in use swing between 35 and 50 MB from iteration to iteration and
   end at 34.6 where the first iteration had 34.9, while the peak climbs 72.1 -> 77.5 -> 82.2 -> 84.4 ->
   87.1, each step at an average whose ~40 MB transient landed on a higher baseline (41.9 MB before the
   one that set 87.1). The baseline varies with when Python frees the previous iteration's arrays. On the
   CPU the live arrays are the same 35.8 -> 40.7 MB at every average and no program is compiled inside
   the iterations after the first. So the peak is a running maximum of a fluctuating sum; releasing the
   previous iteration's grids before the average would take the few MB off, and the average's own
   transient (per orbit, or on the host) is the lever that matters.
3. **Time**: the ultrasoft bare walks rebuild `newd` and its tangent once per chunk and perturbation --
   **sized 2026-10-03 evening as a null**: 36.7 ms a call on ultrasoft eight-atom silicon at 8 k-points,
   0.88 s of a 384 s walked phonon of one atom on the CPU, 0.2 per cent. The walked third derivative's
   electrostriction being 6 per cent slower than the whole route at 27 k-points is not separated.
4. **The radial chunk's ceiling on a production card**: the budget (8 MB an integrand) was chosen on the
   A2000 and the CPU; on a float64 card the smaller transforms' launch count may cost more than the 5 to 8
   per cent measured here, and `DEFUMAT_RADIAL_CHUNK` is the dial.

**Loose ends of the later evening (branch `bessel-jvp`):** the branch holds the radial derivative rule,
the factored augmentation charge and three test tolerances read off one rounding pattern (`CLAUDE.md`'s
`conv_thr` trap); whether it is merged and pushed is in the session's last commit message. On D22 the
worktrees `/l/ladovj1/defumat-bessel` and `/l/ladovj1/defumat-factored` and the runs in
`/l/ladovj1/review/bessel/` (`CLAUDE.local.md`).

**Loose ends, none blocking:** nothing is pushed (master `d227efe`, 27 commits past `5e1e6f6`); the
merged branch `streamed-piezo` can be deleted. On D22: the worktrees `/l/ladovj1/defumat-piezo` (master's
code as of `d227efe`, copied file by file over `32bfa0f`) and `/l/ladovj1/defumat-chunk` (the state before
the symmetry walk), the plain copy `/l/ladovj1/bisect`, and `/l/ladovj1/review/piezo/`, which holds every
run of the session **and the probe scripts, which are not in the repository**: `pass_memory_*.py` (each
compiled pass's `memory_analysis()`, by wrapping `jax.jit` in the module measured), `*_stages.py` (the card
peak after each stage of one call; run twice and read the second), `global_terms.py` and
`first_vs_second.py` (one energy term at a time, compile only, CPU), `sym_probe.py` and `walk_card.py`
(a symmetriser walked against batched), `aug_time.py` (SCF and stress time per chunk), and the two
plans Fable reviewed. `CLAUDE.local.md` has the D22 details. The morning's loose ends stand: the D22
worktrees `/l/ladovj1/defumat-old` and `defumat-hf`, and the pre-existing failure
`test_spinor_response.py::test_the_spinor_density_weights_are_the_ground_state_s` (a tolerance below the
rounding).

## Fixed while this list was written

Three survey findings were small enough to fix in the same branch, and are not repeated
below:

* **A streamed resume kept the whole loaded wavefunction set alive for the run**
  (`driver.py`, the release of the source span sat in the non-streamed branch only; it
  reopened `MEMORY-AUDIT.md` A18 on the stream path). Fixed in `ae99566`, with a test that
  holds a weakref to the loaded *array* and fails on the unfixed driver.
* **`at_kpoints` and `at_spiral_q(rebuild_basis=True)` dropped `projectors='rebuild'`**, so
  every band path, derived mesh, spiral scan step and `relax_spiral_q` step stored the
  whole-k `vkb` in memory mode. Both host-side movers now keep the dial.
* **The DFT+U overlap read `projectors.vkb[ik]`**, which on a lazy set builds the whole-k
  array to slice one row, once per k-point (`O(nk^2)` work, a whole-k transient). It reads
  `at_k(ik)` now; the projectors are bit-identical (`test_projector_storage.py`).

## Done since

* **Item 21** (the relaxation loops' retention): `run_vc_relax`, `run_spiral_scan` and
  `relax_spiral_q` drop the previous step's result and `Calculation` (`MEMORY-AUDIT.md`
  A2, extended). No number moves.
* **Item 7, first half**: `local_perturbation` reads `projectors.at_k(ik)` inside its
  guard; a norm-conserving perturbation no longer builds a whole-k `vkb` it never reads.
  The dozen other response sites are still open.
* **Items 13 and 14(b)**: the four radial transforms walk their chunks in a rematted scan
  (`formfactors._scan_rows`). BN's compiled stress temporary on the card **5.574 -> 1.609
  GiB**, and on the CPU 5.74 / 8.52 / 17.81 -> 1.88 / 1.42 / 1.93 GiB at 1 / 9 / 36
  k-points, so the linear growth in nk is gone as well. BN's SCF and stress now run on the
  GTX 1060 at 2052.9 MB peak, the stress within 2.1e-7 Ry/bohr^3 of `pw.x`. Cost: 7.6 per
  cent on `si2-us-1k`'s warm stress. `PERFORMANCE.md`, "The stress tape".
* **Item 9** (the band dial in memory mode): a `Calculation` resolves its band batch once
  and its Hamiltonians and density kernels carry it; in memory mode on a card the value is
  the largest batch whose estimate fits (`sizing.choose_band_batch`), the whole block
  whenever that fits. On `h40-chain-lsda.in` the peak runs 4127 / 2402 / 1551 / 1327 / 1230
  MB at all / 28 / 14 / 8 / 1 bands for 23.0 / 22.9 / 22.7 / 22.8 / 25.6 s; in a pool cut
  to 2225 MB the default chose 14 bands and ran at 1551 MB where `all` died.
  `PERFORMANCE.md`, "The band dial, budgeted from the card". **Not measured here**: the
  NiBr2 slab this was sized on, which needs a larger card.
* **Item 24, two of its lines**: the estimate now takes `wfc_store` (a streamed store is one
  chunk on the device) and carries the start as a stage (`start_buffer`, D10), and a
  non-dividing band batch pays its tail. The core build, the `qgm` accumulator, `wfcU`, the
  symmetry maps and the index arrays are still absent.
* **Item 15** (the augmentation table on the strain tape): the tabulated chunk sized for
  the backward pass (16 MB target: BN's tabulated stress 7.47 -> 0.47 GiB), and in memory
  mode `at_strain` rebuilds a stored table scanned with the exact radial integral (BN's
  stress tape 1.61 -> 0.42 GiB, bismuthene's 4.15 -> 0.46 GiB and its stress now fits the
  card), for 2.5x BN's stress time. Speed mode keeps the stored table.
* **Item 18** (the PAW one-centre tensors): factored into radial pair tables and Gaunt
  coefficients; one-atom spin-orbit PAW platinum 1309.6 -> 411.3 MB on the card.
* **Item 20, first part** (bands, NSCF and DOS build a second `Calculation`): `run_bands`,
  `run_nscf` and `run_dos` take `calculation=` and move it to their k-set with `at_kpoints`
  (`workflows/nscf.threaded_calculation`), and `Calculator.get_bands`/`get_nscf`/`get_dos`
  pass their own, so the augmentation and PAW tables are shared and the calculator's memory
  mode is followed. A spiral still builds its own (`at_kpoints` refuses one). **Not yet
  measured on the card** -- the check is bismuthene-soc-small's `get_scf` + `get_bands`
  peak, before against after; the fast tests pass. PDOS, STM, STS, transport, `sfac` and the
  spiral scan still build their own.
* **Item 6, second bullet** (the core built in k-chunks, memory mode): the 216-k-point
  Si8 peak 71.1 -> 61.8 MB (setup 40.4), the 800-point band path 163.6 -> 78.5 MB.
* **Item 6, first bullet** (real projector columns): bit-identical `vkb`; the 216-k-point
  eight-atom Si peak 93.6 -> 71.1 MB.
* **Item 8** (a streamed resume): a loaded checkpoint's wavefunctions stay in host memory;
  the same resume's peak 220.8 -> 71.1 MB, the setup's own.
* **Item 12, third bullet**: the third-derivative drivers stack each field block once and
  drop the dead displacement block; lifetimes only, not measured on this card.
* **Item 4** (post-SCF consumers of a streamed store, 2026-09-29): `Calculation.density` and
  `Calculation.becsum` walk a numpy store a chunk at a time (`streaming.stream_densities`,
  `stream_becsum`), which covers the STM image, the windowed structure factor and a
  relaxation's density extrapolation without touching them; `atomic_projections`, the site
  moments' density matrix, the force theorem's projected decomposition, its first-order
  spin-orbit energy and the spiral's `spiral_expectation` walk the chunks themselves (the
  last two also read `projectors_at(rows)`, not the whole-k `vkb`); the three sum-over-states
  workflows slice their bands before the upload. `test_streaming.py` holds the host and
  device routes to 1e-12 on ultrasoft silicon with a short last chunk. **Not done**: the
  in-loop orientation diagnostics, the magnetic torque's derivatives (they are item 3's
  shape), the response stack (item 2) and the ultracell. **Not measured on the card.**
* **Item 1, second half**: a fixed-density solve that keeps its states streams them into a
  numpy store where the store streams (`streaming.stream_states`), so a PDOS or STM on a
  denser grid holds one chunk on the device; the force theorem without `projected` now asks
  for energies only. Same test file, round-off and the same span. **On the card**, `si8-1k`
  SCF + PDOS on an 8x8x8 grid: 56.9 -> 45.7 MB (the whole-k atomic projectors remain).
* **Item 20, second part**: `run_pdos`, `run_stm`, `run_sts`, `run_vertical_transport`,
  `run_momentum_transport` and `run_structure_factors` take `calculation=` (the SCF's own,
  on its k-set or moved with `at_kpoints`), and the `Calculator` passes its own. Left: the
  spiral scan and `DFTSource._base`. Not measured on the card.
* **Item 7, second half** (2026-09-29): `becsum` and `spinor_becsum` take the `Projectors`
  and walk a lazy set per k (`density.walk_projectors`, `at_k` inside the sum, a spiral's
  rows `ik` and `ik + nk`), so the SCF's whole-set `becsum` on a CPU in memory mode, the
  Sternheimer solver's raw `becsum`, the ultrasoft position operator (`efield.
  ultrasoft_position`), the transverse susceptibility's `state_projections` and the topology
  states' `becp` no longer stack the whole-k `vkb`. `test_projector_storage.py` makes the
  whole-k property fail for a lazy set during the call. AlAs ultrasoft `epsilon` identical
  and `Z*` within 1e-14 between the two modes; the ultrasoft magnon and the ultrasoft
  topology checks pass with `DEFUMAT_PROJECTORS=rebuild`. Still whole-k: the traced movers
  (`at_strain`, `at_kcart`, `at_spiral_q`) and what reads them -- the velocity operator's
  projector derivative, the strain and phonon responses, `born.py`'s ultrasoft tail. Not
  measured on the card.
* **Item 11** (2026-09-29): the streamed Davidson call donates its starting block
  (`_every_k_donating`), and the robust retry is handed a fresh copy from the host store
  (`psi0_again`). `test_solvers.py` forces the retry and checks both halves. Not measured on
  the card.
* **Item 12, first bullet** (2026-09-29): `chi_0`'s pair axis has its own default
  (`batching.resolve_pair_batch`): the band dial's value on a CPU and wherever
  `DEFUMAT_BAND_BATCH` or an argument says, and on an accelerator the largest chunk whose
  pair boxes fit `PAIR_BUDGET_BYTES = 256 MB` rather than every pair. The budget is a guess
  until the 1/8/32 sweep on the card is taken; nothing changes on a CPU.
* **Item 23** (2026-09-29): importing the package sets `XLA_CLIENT_MEM_FRACTION = 0.9` unless
  either spelling is already set or `DEFUMAT_MEM_FRACTION` says otherwise (`off`, or a number).
  On the GTX 1060 the pool is **5452 MB against 4765 MB** (`bytes_limit`, measured).
* **Item 17** (2026-09-29): `HOISTED_FIELDS` adds the projector core, `|k+G|^2`, the FFT
  indices, a meta-GGA's `k + G`, `vltot` and the core charge. Ultrasoft silicon at 64
  k-points: the force gradient's constants **2.85 -> 0.29 MB**, the stress's 0.68 -> 0.58 MB
  (`make_jaxpr(...).consts`); forces and stress bit-identical to the old tuple on ultrasoft,
  PAW, noncollinear, DFT+U and LSDA cells. **On the card** (2026-09-29, a 256 MB array): a
  closed-over device array is *not* copied a second time -- the same 256 MB in use as an
  argument -- but it stays alive for as long as the compiled function does, where an
  argument is freed when dropped. So the hoisting changes retention, not peak.
* **Item 3** (2026-09-29): the force and the stress walk the k axis where the state is a
  host store or memory mode's chunk is smaller than the k-set (`forces/chunked.py`): a forward
  walk for the raw `becsum`, smooth density and `ns`, one `value_and_grad` of the global terms
  at the whole sums, and a second walk pulling each chunk back with cotangent
  `(1, g_b, g_rho, g_ns)` -- `forces/spiral.py`'s split, generalised to the positions and the
  strain. Round-off against the single pass (<= 3e-14 on force and stress) on ultrasoft, PAW,
  DFT+U, noncollinear, LSDA, spin-orbit, norm-conserving and gamma-only cells with a short last
  chunk; `test_forces.py` and `test_stress.py` pass in memory mode against `pw.x` (51 passed,
  7 skipped). Speed mode keeps the single pass. Each chunk's passes run on that chunk's
  row-subset calculation (`Calculation.at_rows`, passed as traced leaves so every chunk
  shares one compilation); without it every chunk's backward pass rebuilt every k-point's
  projector core. **Measured on the GTX 1060**, eight-atom Si at 20 Ry, `nosym`, memory mode,
  warm, one process per point: force peak **73.9 -> 35.0 MB** at 27 k-points and **167.8 ->
  39.9 MB** at 64 (the SCF's own peak both times, so the force adds nothing); stress **94.1 ->
  57.5 MB** and **220.8 -> 62.6 MB**. Time 0.26 -> 1.5 s (force) and 1.0 -> 2.2 s (stress) at
  27, 0.32 -> 2.9 s and 1.3 -> 4.2 s at 64: one chunk per k-point, two walks.
* **Item 10** (2026-09-29): Davidson carries the `(nvecx, nbnd)` Ritz coefficients and forms
  `evc`/`hevc` at the top of each step (`ritz`), at the width the solve used. **Bit-identical**
  eigenvalues, states and step counts on `si2-us`, `pt-soc-paw-nosym`, `si8-1k` and a
  gamma-only cell, both routes, cold and seeded. Compiled temporary, `si16-1k-ecut30`,
  `david = 4`: on the GTX 1060 **117.0 -> 111.2 MB** at 64 bands one band in flight (one
  band block), 129.43 MB unchanged at 32 bands all in flight (the FFT boxes set it there); on
  the CPU 46.0 -> 44.5 and 126.2 -> 123.4 MB.
* **Item 19** (2026-09-29): the stored `Q_ij(G)` is real, `R_ij(G)`, with
  `AugmentationCharge.pair_phase = (-i)^(l_i+l_j)`; the charge, the `D_ij` integrals, the
  displaced table's complex integrals and `addusforce` put the phase on the small side and
  contract the real table twice. Bismuthene's relativistic table **1067.9 -> 534.0 MB**, Pt's
  120.9 -> 60.5 MB (nbytes, CPU). Against the complex construction: table 8.6e-17, charge
  7.7e-16, integrals 4.5e-15 relative, `qq` exact; the wrong-parity part is exactly zero (the
  radial transforms store only the allowed `L`). Energies move by at most 1.1e-13 Ry. The
  storage gate still sizes the complex table, so no cell changes route. **On the card**
  (against this morning's code, memory mode): bismuthene's SCF + stress peak **3457.4 ->
  2393.5 MB**, same energy, stress and time.
* **Item 24, three more lines** (2026-09-29): the estimate counts the per-k basis tables
  (`|k+G|^2`, the FFT index, the mask, the gamma trick's `-(k+G)` index), the symmetry maps
  (`nsym ngm` permutations and phases) and DFT+U's `wfcU`, each held exactly against the
  arrays a built `Calculation` allocates (`test_sizing.py`). On two-atom PAW silicon the
  symmetry maps (6.9 MB, 48 operations) are now the largest line after the PAW tensors,
  above the real `Q_ij(G)` (3.1 MB). Still absent: the core build's transient and the `qgm`
  accumulator.
* **Item 5** (2026-09-29): `chern_number(stream=True)` walks the plaquette plane a column
  at a time, with links across columns taken between two state sets (`overlaps(...,
  other=)`, `q_ij(b)` included) and only link phases kept, so at most three columns are
  resident; `streamed_orbital_magnetization_sums` walks the volume mesh a plane at a time
  through the whole-mesh route's own per-k function, holding at most five planes. Both are
  the default where the store streams (`DFTSource.streams`) and off on a CPU. Models agree
  to 1e-12; AlAs flux to 9.6e-9 (NC) and 2.6e-8 (US), shrinking with `conv_thr` (each column
  is its own solve); iodine `M_LC`/`M_IC` to 3.5e-9/1.2e-9. Bismuthene 12x12: solve output
  446 -> 37 MB, occupied states 372 -> 93 MB (shapes). 0.96 s against 0.74 s on one CPU
  core (NC AlAs 6x6). **On the card**, bismuthene SCF + Berry curvature 8x8: the old code
  died at the 0.75 pool and peaked at 4578.1 MB at 0.9; now 2393.5 MB, the SCF's own.
* **Item 20, third part** (2026-09-29): `run_spiral_scan`, `spiral_spin_orbit_energy` and,
  through `DFTSource.calculation`, `run_berry_curvature`, `run_z2`, `run_z2_3d` and
  `run_orbital_magnetization` take `calculation=`, and the `Calculator` passes its own. With
  the same options the results are identical to a fresh setup's (AlAs Berry flux, the
  hydrogen-chain spiral scan: differences of 0.0). *That sentence then said nothing else
  built a second `Calculation` except the force theorem, which was wrong*: the six
  sum-over-states spectra did (next bullet).
* **Item 20, fourth part** (2026-09-29): `run_absorption`, `run_conductivity`, `run_shg`,
  `run_shift_current`, `run_spin_susceptibility` and `run_magnon_dispersion` take
  `calculation=` and the `Calculator` passes its own
  (`test_one_calculation_per_workflow.py`, which fails on all six without it). They are
  norm-conserving only, so the second setup cost little memory; what it dropped was the
  calculator's `memory_mode`, so a speed-mode calculator's spectrum ran in the platform's
  default mode without saying so. Now only the force theorem builds its own.
* **Item 25** (2026-09-29): recorded in `GPU.md` Phase 4 -- k-sharding divides time, not
  per-device memory; distributing the plane waves by sticks is the memory lever.
* **Item 12, second bullet** (2026-09-29): the TDDFT frequency axis has a dial, `w_batch`
  (`batching.resolve_w_batch`: an argument, then `DEFUMAT_W_BATCH`, then the whole axis on a
  CPU and a 256 MB budget on a card). `chi_0`'s whole-axis assembly held a `(2 npairs, nm, nm)`
  block; chunks hold `n` blocks of `(2 npairs, nm)`, written into one output
  (`map_windows`, so a chunk that does not divide `nw` concatenates no tail). Silicon at
  `nm = 115`, 60 bands, 200 frequencies: **130.8 -> 32.4 / 8.7 MB** per k-point at 32 / 8
  (compiled, CPU). Every registered kernel is static, so `solve_dyson` iterates the fixed
  point on the one frequency it depends on and screens the axis once in chunks into a donated
  output, keeping one `(1, nm, nm)` `fxc`. Bit-identical to the old loop on synthetic data; on
  silicon `eps_M` agrees to 8e-19 and `chi_0` to 7e-15; the bootstrap at `nw = 200` goes from
  56.6 to 0.94 s warm (machine loaded). **On the card**, `get_absorption` on the silicon
  cell (120 frequencies, bootstrap, scissor, static residual): 229.8 -> 226.5 MB in memory
  mode, 234.1 -> 230.8 MB in speed mode -- the frequency block is not the peak on this cell.
* **The row-subset `Calculation`** (2026-09-29), the prerequisite items 2 and 6 name:
  `Calculation.at_rows(rows)` slices every array with a k index (spheres selected, FFT and
  stick indices, `|k+G|^2`, `ProjectorCore.rows`, `wfcU`) and shares the rest; `npwx` and the
  stick count stay the whole set's and the k-points keep their weights. On ultrasoft silicon
  the subset's eigenvalues, `H|psi>` and `becsum` are bit-identical to the whole set's at
  those points and the velocity operator agrees to 5e-16 (`test_row_subset.py`). The chunked force and
  stress use it; the response stack does not yet.
* **Item 1** (the fixed-density solve): an eigenvalue-only solve streams where the store
  does and keeps no states. On eight-atom Si the band path's peak is 102.8 -> 44.8 MB at 200
  points and 383.4 -> 163.6 MB at 800, for 2-3 per cent in time. The 0.20 MB per k-point
  left then is since **0.072 MB**: the rest was the core build's transient, gone with item
  6's chunked build (`PERFORMANCE.md`, "the 0.11 MB per k-point left unattributed ... was
  this build's transient"). What remains is the band path's own resident per-k tables,
  which is item 6's third bullet.

* **The dials are sized for the SCF's bands** (2026-09-29, found locating item 12's 2.6 GB):
  the memory mode and the band batch were resolved once, for the SCF's band count, and a
  fixed-density solve at many more bands ran at them. `Calculation.for_bands(nbnd)`,
  called by `fixed_density_states`, re-resolves both from the same requests: speed mode
  falls back for a solve that would not fit, memory mode re-chooses its batch. `h40-chain-
  lsda.in` in a 2122 MB pool, 84 bands: batch 8 -> 1, peak 1456.1 -> 1371.5 MB, band
  energies to 6e-14 Ry; at 168 bands nothing fits and the run now says so before starting.
  `test_band_budget.py`.
* **`jax.device_put`, not `jnp.asarray`, for a large host array** (2026-09-29): on the card
  `jnp.asarray` of a host array peaks at twice its size on the device, `device_put` at
  once. The assembled projector core and the projections' chunks now cross with
  `device_put` (the streamed store already did): eight-atom Si, 216 k-points, setup peak
  38.5 -> 35.4 MB. Other `jnp.asarray` uploads of large host arrays have not been audited.

* **The day's items on the card** (2026-09-29, `PERFORMANCE.md`, "The projected DOS, the
  pairs and the rest, on the card"): item 4's windowed structure factor 160.2 -> 59.0 MB
  (the SCF's own); item 20's band path on the calculator's setup 2986.4 -> 2393.5 MB on
  bismuthene; the pair budget 895.8 -> 342.2 MB where the pairs are the peak (80 bands on
  eight-atom Si). **Two nulls**: item 7 on AlAs's wedge (326.5 / 325.4 MB, too few k-points
  for the whole-k set to matter) and item 11 on bismuthene (3457.4 both; one block under an
  augmentation-table peak) and on the hydrogen chain, whose peak is the eigensolver's
  (3936.1 both).
* **The projected DOS on the SCF's own k-points** (`ed1ef29`, found measuring item 4): the
  atomic projectors were built whole-k and the smearing DOS formed its `(nE, nk, nbnd)`
  intermediate six times over. Blocks of k-points on `at_rows` for the first, 16 MB energy
  blocks in one compiled kernel for the second: eight-atom Si, `nosym`, 216 k-points,
  **645.6 -> 96.2 MB** against an 86 MB store, the PDOS identical.
* **Item 18, the meta-GGA half** (`246f355`): the PAW kinetic maps are held as their factors,
  10.5 -> 0.30 MB on silicon PAW under `tb09`, `ddd` to 4.4e-16.

* **Item 6's third bullet, for a band path** (`19d4aa4`, see item 6): NbSe2's path flat at
  206 MB from 91 to 181 points, where it was 301.6 MB at 91 and growing 1.65 MB a point.
* **Verdicts** (2026-09-29): item 7's traced movers need nothing outside item 2, and item
  16's forward-mode stress is not needed on any cell here (see each item). Item 26's first
  blocker is named: setup arrays built from the radial tables ignore the precision policy.

* **Item 2, the field half** (2026-10-02, `0a9f317`, `bf821e7`, `6485073`): where the store
  streams, or memory mode's chunk is smaller than the mesh (`forces.chunked.walks_chunks`, the
  force's own rule), `dielectric_tensor` walks `k_chunks` with the states, the three bare
  perturbations and the three `dpsi` in host memory (`defumat/response/chunked.py`). Every index is
  the chunk's own, on `at_rows`; the whole set's Hamiltonians are sliced rather than rebuilt
  (`Calculation.restricted_hamiltonians`, bit-identical), `int3` is made once per direction and
  iteration, and the level shift `alpha_pv` is the whole set's. The Born charges are item 3's split
  one `jvp` up: a forward walk for the raw sums and their tangents, the `jvp` of the global terms'
  gradient with the full-zone shifts as tangent only, and a pull-back walk whose `jvp` carries the
  states, the matrix multipliers and the global cotangent. Against the whole-k route
  (`tests/regression/test_streamed_response.py`): ultrasoft AlAs 5e-14 in `epsilon` and 3.5e-13 in
  `Z*`, where dropping the `g_b . db_c/dx` cross term moves `Z*` by 39; norm-conserving and PAW
  silicon wedges at 1e-14 or below. A second call compiles nothing. **On the card** (RTX A2000,
  eight-atom Si at 20 Ry, `nosym`, memory mode at `k_batch = 1`): the default call's peak 225.9 /
  510.6 / 976.0 / 1695.2 -> 35.1 / 39.6 / 47.5 / 58.8 MB at 27 / 64 / 125 / 216 k-points,
  the SCF's own, for 12 to 14 per cent more time (`PERFORMANCE.md`, "The dielectric tensor and the
  Born charges a k-chunk at a time"). The regression is `tests/regression/test_streamed_response.py`
  (slow set, 6 min). Taken whole still: the phonons, the strain response, the third derivatives and
  anything asking for `keep_internals`, and a strained calculation (`_kcart`). **The sizing gap this
  first left is closed** (`d4be84c`): where `k_batch = 'fit'` took the whole mesh the response took
  the whole-k route unsized, and on ultrasoft eight-atom silicon at 27 k-points its Born charges read
  6825.4 MB where the SCF read 925.2. In memory mode on a card the response now always walks chunks,
  the whole mesh as one included, and the two Born passes that differentiated in the positions with
  `jacfwd` (3289 and 2185 MB of temporaries at that chunk) walk the tangents one at a time: the run
  reads 925.2 MB, the SCF's own, in 58.4 s against the whole-k route's 72.4. **Not covered**: speed
  mode, whose check against the card sizes the SCF and not the response; and at one k-point a chunk
  the ultrasoft cell's Born charges add a fixed 153 to 155 MB to a 67 to 75 MB SCF, at 27 and 64
  k-points, which is the frozen polarization's pass (166 MB of temporaries). **What inside it is the
  projectors' radial transforms at fewer than `CHUNK = 4096` values**, which `_scan_rows` takes in one
  piece, where XLA keeps every radial function's `(nq, kkbeta)` integrand live at once: 41.5 MB for
  one k-point's 1614 values of this dataset's four functions, against 26.9 MB for 8192 values walked in
  two pieces. The frozen pass holds four such evaluations (its value and three directions), the bare
  walk two (`PERFORMANCE.md`, the same entry). It is item 14's transform, and (a) there, one transform
  per `(dataset, l)`, is one of the ways to it.
* **Item 2, the Gamma phonon** (2026-10-03, `af246f1`, `defumat/response/chunked_phonon.py`):
  `dynamical_matrix` takes the field response's route under the field response's rule
  (`efield._streams`). The bare perturbations and `dpsi` per perturbation are host stores; `ort` is
  rebuilt per chunk from the `(nocc, nocc)` overlap derivative rather than stored; the solves reuse the
  field's compiled `respond` pass; a metal's `ldos` is walked once; and each column is the chunked
  force's split differentiated once more with the coordinate moving (for ultrasoft and PAW, a forward
  walk, the global terms' `jvp`, a pull-back walk; for norm-conserving, the global terms' `jvp` and one
  walk, since the density is an independent array there). The self-consistent loop is one function for
  both routes (`phonon.screening_loop`) and the whole route is bit-identical to before. **Before this a
  streamed store did not run at all** (`TracerArrayConversionError`: the solver keeps the store in host
  memory and the whole-k helpers index it with a traced k). Against the whole route
  (`tests/regression/test_streamed_phonons.py`): 5.1e-15 / 4.3e-15 / 1.3e-14 / 1.9e-14 / 1.0e-13 /
  2.2e-16 on norm-conserving silicon's wedge, two-atom aluminium (a metal), ultrasoft and PAW silicon's
  wedges, ultrasoft AlAs and half-sphere silicon. **On the card** (eight-atom Si at 20 Ry, `nosym`): one
  atom at one k-point a chunk now runs at the SCF's own 35.1 / 39.6 MB (27 / 64 k-points) and at 116.5
  against 66.9 MB ultrasoft; at `'fit'` all eight atoms read the SCF's 872.9 MB where they read 1119.0,
  and ultrasoft 925.2 MB where it read 1780.0, in 626 s against 762 (`PERFORMANCE.md`, "The Gamma phonon a k-chunk at a time"). The padding
  found a bug the Born charges could not: the constraint is weighted by the multipliers, not the
  occupations, so a padded row's `dLambda` has to be zeroed explicitly. **Found on the way and fixed
  separately** (`54654f5`): a PAW wedge's assembly counted the becsum symmetrisation twice, 4.0e-4
  Ry/bohr^2 against the same sample whole. **Not changed by this entry**: the `3 nat` dense-grid
  fields the loop carries, which are the next entry; and the bare walk recomputes `newd` and its
  tangent once per chunk and perturbation where the whole route did it once per perturbation, which
  costs time on an ultrasoft cell and was not separated.
* **Item 2, the phonon at `q`, and the per-mode grids** (2026-10-03, `61c71d4`, `7740ab7`,
  `3601e96`). `dynamical_matrix_at_q` takes the same route: the `k + q` states go into a host store a
  chunk at a time (`stream_states`), each chunk has two row-subset calculations built from the
  calculation's own fields (so a second `q` reuses the passes), the response density's sum over k is
  added over chunks and finished once, the electronic half of `D(q)` is one Gram product of the two host
  stores, and the frozen half is the Gamma passes with zero tangents. Before it the phonon at `q` put a
  streamed store back on the card whole. **And on a run without symmetry the loop's per-mode grids**
  (`dvscf`, the response, the induced potential, `drhous`, the core term: six to seven dense grids per
  perturbation at an iteration's peak) **are in host memory**, one mode at a time on the card, since the
  displacement average is the identity there; the phonon at `q` always runs so. Against the whole route:
  2.1e-14 (`q`, end to end at one k-point a chunk), 1.9e-14 (`q`, a chunk of 3 with one `k + q` store),
  and the host grids 2.2e-16 to 5.2e-14 on four cells. **On the card** (eight-atom Si at 20 Ry,
  `nosym`, one k-point a chunk): the phonon at `q`, all 24 perturbations, 953.7 MB -> 119.1 MB with the
  grids on the card and 103.0 MB with them on the host, against the SCF's 35.1; the Gamma phonon of all
  eight atoms 65.8 -> 35.1 MB (norm-conserving) and 203.9 -> 109.8 MB (ultrasoft, SCF 66.9), for under 1
  per cent of time (`PERFORMANCE.md`, "The phonon at q a k-chunk at a time, and the per-mode grids off
  the card"). **Found on the way and fixed separately** (`12fd7ea`): on two FFT grids the phonon at `q`
  dropped the imaginary part of its response (every mode imaginary at `ecutrho = 8 ecutwfc`). **Still on
  the card**: with symmetry, the per-mode grids, because `symmetrize_atom_displacement` acts on the stack;
  the `k + q` Hamiltonians' whole-k tables beside the `k` ones; the ultrasoft global step's 43 to 50
  MB of temporaries. **The phonon at `q`'s 68 MB above the SCF was the Ewald swap**, `jax.hessian` of
  the Gamma Ewald sum with all 24 tangents at once; a column at a time (`51e257e`) the call reads 49.5 MB
  against 103.0.
* **Item 2, the piezoelectric tensor** (2026-10-03, branch `streamed-piezo`). The differentiated route
  asked the field response for its internals, which forced the field's whole-k route, and took each
  column's `jvp` of the stress with the whole k axis on one tape. Where the field response walks
  (`efield._streams`), the re-diagonalisation now writes into a host store
  (`refined_states(stream=True)`), the field hands back its `StreamedField` with the stores in host
  memory (`dielectric_tensor(streamed_internals=True)`; the whole-k keys are then absent, so a consumer
  written for device arrays fails by name), and each column is the streamed Born charges' split with
  `at_strain` (`StreamedField.piezoelectric`, `_born_passes(kind="strain")`), no frozen-polarization
  term, and the sandwich of `add_for_charges` per chunk. **On the card**, two-atom ultrasoft AlAs at
  25/200 Ry, `nosym`, one k-point a chunk: 1208.4 / 1256.6 / 1349.6 -> 759.7 / 763.2 / 768.8 MB at 8 /
  27 / 64 k-points against an SCF of 169 to 178, and faster (28.2 -> 14.8 s at 8); eight-atom AlAs at 8
  k-points 1660.1 -> 986.3 MB against 669.6. What is left above the SCF is the global step's 705 MB of
  temporaries on the 200 Ry dense grid, k-independent and the same in both routes (`PERFORMANCE.md`,
  "The piezoelectric tensor a k-chunk at a time"). Against the whole route
  (`tests/regression/test_streamed_piezo.py`): 1.4e-15 e/bohr^2 or better on tensors of 1e-2, on
  norm-conserving, ultrasoft and PAW AlAs, closed grids and wedges; a falsifier there zeroes the walk's
  full-zone shift on the ultrasoft wedge and must move the tensor. The transcribed route
  (`method="zstar_eu"`) still takes the whole route.
* **Item 2, the strain response and the elastic constants** (2026-10-03, branch `streamed-piezo`).
  The strain response held six bare perturbations and six first-order states (and for ultrasoft or PAW
  the occupied block and `S'`) whole-k on the card, and **on a streamed store it did not run at all**
  (the whole-k helpers index the host store with a traced k, as the Gamma phonon's did before
  `af246f1`). Where the field response walks it now takes the Gamma phonon's three stages with the six
  strains as its modes (`defumat/response/chunked_strain.py`, `StreamedStrains`): the bare walk with
  the potential's strain tangent taken once per strain, `S'` and the frozen-state response's two halves
  finished from summed tangents, the field's `respond` pass, and the eigenvalue response walked; the
  loop is one function for both routes and the whole route is bit-identical to master on ultrasoft and
  PAW silicon. The elastic constants are the walked stress differentiated once more
  (`forces/chunked.chunked_gradient_tangent`). Against the whole route
  (`tests/regression/test_streamed_strain.py`): every field of the response to 1e-11 relative or better
  (the density 1.0e-14 on 0.10, `dpsi` 4.7e-13) on norm-conserving silicon (closed grid and wedge),
  ultrasoft and PAW silicon, and the elastic constants to 1.0e-11 GPa on `C_11` = 209.4. **On the
  card**, eight-atom silicon at 20 Ry, `nosym`, one k-point a chunk: the strain response runs at 45.7 /
  50.1 MB at 27 / 64 k-points against SCFs of 35.1 / 39.6, where it died before, and the elastic
  constants on top read 126.5 MB at 27 (`PERFORMANCE.md`, "The strain response and the elastic
  constants a k-chunk at a time").
* **Item 2, the third derivatives** (2026-10-03, `b29091e`). Electrostriction's `d(eps)/d(strain)` and
  the Raman tensors read the field's stores whole, so the field took its whole route under them. Both are
  a `jvp` of the variational second-order energy per tangent, and that energy is a sum over k of per-k
  terms reading two whole-cell objects (the moved potential, PAW's one-centre coefficients) plus terms of
  the whole raw response sums, so the derivative is a forward walk, one global step and a chunk walk
  that also solves `db` on the chunk (`defumat/response/chunked_third.py`); Raman's displacement
  response is the walked Gamma phonon's stages without the assembly. **On the card**, two-atom
  ultrasoft AlAs at one k-point a chunk: electrostriction 784.9 / 882.8 -> 263.5 / 264.9 MB at 8 / 27
  k-points, Raman 299.0 / 368.5 -> 199.3 / 203.1 MB, against an SCF of 169 to 172 (`PERFORMANCE.md`,
  "The third derivatives a k-chunk at a time"). Against the whole route
  (`tests/regression/test_streamed_third.py`): 1.3e-14 and 1.9e-15 relative on the same states and
  tangents, about 1e-13 relative end to end. **The vibrational spectrum walks too** (2026-10-03, after
  this entry): the walked Raman route hands back its walked displacement response
  (`DisplacementResponse.walked`) and `dynamical_matrix` assembles from its stores, with nothing
  re-diagonalised or put on the device. **Found on the way and fixed separately** (`da8f6cc`): on an
  ultrasoft or PAW dataset the vibrational spectrum had never run (no `becsum` for the dynamical matrix,
  an `IndexError`), and with that fixed it gave wrong phonons silently -- ultrasoft silicon's optical mode
  at 630.8 cm^-1 against 590.6 solved directly -- because the response the Raman tensors handed over
  lacked the bare perturbations and extras the ultrasoft assembly rebuilds the multipliers from; the
  existing reuse test ran on a norm-conserving cell, where both are zero.
* **The radial transforms' chunk, sized from the mesh** (2026-10-03, `a71b8a3`). The Bessel transforms
  took 4096 values of `|q|` at a time; the strain's derivatives hold many `(chunk, mesh)` integrands at
  once, so the chunk is now sized to 8 MB an integrand (`formfactors.radial_chunk`, capped at 4096,
  `DEFUMAT_RADIAL_CHUNK` overrides it). **On the card**, ultrasoft AlAs at 200 Ry and 8 k-points: the
  stress 652.2 -> 169.3 MB (the SCF's own) and the piezoelectric tensor 759.5 -> 198.4 MB, for 5 to 8 per
  cent of those calls' time; on the CPU the energy is bit-identical, the stress moves by 3e-15 relative,
  and the warm stress is no slower (`PERFORMANCE.md`, "The radial transforms' chunk, sized from the
  mesh"). This is the global step the piezoelectric tensor's entry located, and it was the stress's own
  memory as much as the third derivatives'.
* **The group averages walk the operations** (2026-10-03, `90ed76a`). Every symmetrisation gathered the
  field for all `nsym` operations at once, `nsym` copies of it, and a two-rotation one (strain, spin
  vector) an intermediate three times that. Past 64 MB of that working set each walks the operations in a
  compiled scan (`symmetry.GATHER_BUDGET_BYTES`); below it the batched gather stays to the last bit. **On
  the card**, eight-atom silicon at 20 Ry with its 24 operations: the Gamma phonon of all atoms 503.0 ->
  89.4 MB and the strain response 219.5 -> 50.7 MB against an SCF of 39.0 (`PERFORMANCE.md`, "The group
  averages walk the operations past a budget"). This was most of what "the per-mode grids with symmetry
  stay on the card" cost: the grids are `3 nat` fields, the average was `nsym` copies of them.
* **The augmentation chunk under a strain** (2026-10-03): measured, committed (`83ac598`) and reverted
  (`b164421`) -- see the handoff's first item.
* **The radial transforms' derivatives, as transforms** (2026-10-03 evening, branch `bessel-jvp`): the
  structural fix the first item asked for, on its radial half. A `custom_jvp` whose `q`-derivative is the
  transform one order up (`formfactors.bessel_transform`, value and slope from one walk in
  `_transform_pair`), more accurate derivatives of `j_l` (`radial.spherical_bessel_derivative`, 5.5e-14
  against `mpmath` where JAX's derivative of the `sph_bes` form was off by up to 3.5e-8 in `j_0''`), and on
  the card the AlAs piezoelectric tensor, the Si8 elastic constants and the Pt stress at their SCF's or
  strain response's peak (`PERFORMANCE.md`, "The radial transforms' derivatives, as transforms"). It also
  found two piezoelectric tests whose bounds rested on a rounding pattern: master fails them too with
  `DEFUMAT_K_BATCH=2`.

## Suggested order

Cheap and certain first, then the two that decide whether the large cells run in the
default mode:

1. ~~**Retention in the relaxation loops** (item 21)~~ -- done.
2. ~~**One misplaced line in the Sternheimer local perturbation** (item 7, first half)~~ --
   done.
3. ~~**Remat the radial transforms on the derivative path** (item 13)~~ -- done; it was the
   cause of the BN stress death (A/B above).
4. ~~**A budget for the band dial in memory mode** (item 9)~~ -- done, measured on
   `h40-chain-lsda.in`; the NiBr2 slab itself is not measured.
5. ~~**Stream the NSCF / band-structure solve** (item 1)~~ -- done for the eigenvalue-only
   callers; the band path's own `Calculation` is what grows now (items 6, 20).
6. ~~**Factor the PAW one-centre tensors** (item 18)~~ -- done, 1309.6 -> 411.3 MB.

Then the rest by priority. **As of the evening of 2026-09-29** items 3, 4, 5, 7, 10, 11, 12,
17, 18, 19, 20, 23, 24 and 25 are done or partly done (see "Done since"), and all of them
but 24 and 25 have been measured on the card, two as nulls. What is left, in order:

1. ~~**Measure the day's changes on the card**~~ -- done, including the nulls.
2. ~~**A `Calculation` restricted to a row subset of k**~~ -- done (`at_rows`), and the
   chunked force and stress already run on it.
3. **Stream the linear-response stack** (item 2) on top of it. **The dielectric tensor and
   the Born charges are done** (2026-10-02, see "Done since"): the Born charges were item 3's
   split one derivative up, as forecast. **The `Gamma` phonon and the phonon at `q` are done
   too** (2026-10-03, see "Done since"), and on a run without symmetry their per-mode grids
   are in host memory. **The piezoelectric tensor, the strain response, the elastic constants
   and the two third derivatives are done** (2026-10-03, see "Done since"), and so is the
   vibrational spectrum; the item is closed.
4. The small tail: a dense NSCF/DOS/PDOS mesh a block at a time (item 6's third bullet;
   the band path is done), item 14's per-`l` transform (time only), and the float32 tier's
   setup cast (item 26, its first blocker named). Items 7, 16 and 22 are closed by verdict.

---

## A. What still grows with the k-mesh

Memory mode bounds the SCF. These are the places outside it -- and the resident
bookkeeping inside it -- that still scale with `nk`.

### 1. Stream the fixed-density (NSCF, bands, DOS) solve -- priority 1, medium

**Done 2026-09-28 for the callers that want energies only** (`fixed_density_bands`). A
caller that keeps the states (PDOS, STM, the sum-over-states workflows) still gets them
stacked on the device; that is item 4's shape of problem.

`workflows/nscf.py` `fixed_density_states` calls `calculation.diagonalize(...)` and gets
the stacked `(nspin, nk, nbnd, npol*npwx)` set on the device (`nscf.py:278-281`), in
memory mode too; `batching.py`'s own docstring records that bands and NSCF "hold a full-k
store of their own" and "what they would want instead is to never stack the set at all".
It scales with the mesh or the path length, and a non-finite retry holds it three times.
nbse2 at 24x24: about 2.2 GB. `anisotropy.py:577` calls it where only energies are read.

**Fix**, QE's `c_bands_nscf`: when `wfc_store` resolves to `stream`, walk `k_chunks` and
call `eigensolver(h, nbnd, None, ethr, indices=rows)` -- `davidson_eigensolver_all`
already accepts `psi0=None` with `indices`. Keep eigenvalues on the host; either discard
the chunk's states (bands, DOS, nesting, the unprojected force theorem) or write them into
a host numpy store, the same object a streamed `SCFResult.wavefunctions` is. **Risk**:
round-off. **Measure**: `get_scf()` alone against `get_scf()` + `get_bands()` on a 200 and
an 800-point path; the difference should go from ~0.6 MB per k-point to flat.

### 2. Linear response holds its state whole-k and cannot stream -- priority 1, large

**The dielectric tensor and the Born charges are done (2026-10-02, `0a9f317`, `bf821e7`), and
the `Gamma` phonon and the phonon at `q` (2026-10-03, `af246f1`, `61c71d4`, with the per-mode grids
in host memory on a run without symmetry, `7740ab7`, `3601e96`; see "Done since"), and the
piezoelectric tensor, the strain response, the elastic constants and the two third derivatives,
Raman and electrostriction, and the vibrational spectrum (2026-10-03, branch `streamed-piezo`). The
item is closed.**
The text below is the item as it stood, and its per-k-point estimate was low: measured on the
card, the field response grew 5.3 MB per k-point and with the Born charges 7.6, against the
3.7 forecast below.

Its prerequisite (`Calculation.at_rows`) landed 2026-09-29. The solve itself chunks cleanly, but the
bare perturbation does not: `VelocityOperator` takes one `jvp` of `at_kcart` over the
*whole* k axis, so a chunked `bare` would rebuild every k-point's core once per chunk
(`nk / k_batch` full rebuilds). What it needs first is a `Calculation` restricted to a
row subset of k -- plane waves *selected* (`_planewaves_rows`), per-k tables sliced, core
rows, `wfcU` rows -- which is item 6's third bullet. Build that, then this.

`SternheimerSolver` does `self.psi = jnp.asarray(psi)[:, :, :keep]` (`sternheimer.py:317`)
and stacks every block of `dpsi` on the device; `efield.py:296` and `phonon.py:383/385/
430/440` upload the whole store, and a streamed (numpy) store is re-uploaded per call. It
is `(1 + P) nspin nk nocc npwx npol x 16 B` with `P` = 6 for the field (+3 for US Born
charges), `2 x 3nat` for phonons, 12 for the strain response. On eight-atom Si at 216
k-points that is about 3.7 MB per k-point for the field response against the SCF's 0.19.
Recorded as structural in `PERFORMANCE.md` (P24, and the Phase 5 GPU half); no audit id.

**Fix**: keep `psi`, `bare` and `dpsi` as host arrays when streaming; `solve` walks
`k_chunks` like `stream_diagonalize` (the Hamiltonian is indexed by k already); the
response density and `becsum` become per-chunk `jvp`s of `density_at`, accumulated raw and
finished once -- `stream_densities`' structure -- which is exact because both are linear
in the k-sum; build bare perturbations per chunk; turn the assemblies into sums over
chunks. **Risk**: round-off; never hand a whole host array to a `jit` (XLA computes on
the host, silently). **Measure**: a `tools/gpu/response_memory.py`, one property per
process, Si8 at 27/64/125/216 k: SCF only, + dielectric, + one atom's phonon column.

### 3. Forces and stress put the whole k axis on one tape -- priority 2, large

**Done 2026-09-29** (see "Done since"; `forces/chunked.py`).

`energy_at` reads `moved.projectors.vkb` and `state.wavefunctions` whole, and the force
and stress are single `jit(grad)` calls (`forces/autodiff.py`, `stress/autodiff.py`).
Force tape: `nk (npwx nkb + npwx nat + nbnd npwx npol) x 16 B`; the stress adds
`nk npwx (16 ncs + 8 kkbeta)`. `MEMORY-AUDIT.md` §8 records the mechanism (the k dial is
inert under reverse mode) and declined it when `nk = 1` was the case that bit; memory mode
targets many-k runs, which reopens it.

**Fix**: generalise `forces/spiral.py`'s `_split_energy_and_gradient`: a forward walk for
raw `becsum`, `ns` and the smooth density; one `value_and_grad` of the global terms
(local, Hartree, XC with core, augmentation, one-centre, Hubbard, Ewald, dispersion) at
the whole sums; a second walk pulling each chunk's `(e_c, b_c, ns_c)` back with cotangent
`(1, g_b, g_ns)`. The spiral route is the working template and its test is the model.
**Risk**: round-off, if every k-coupled term goes through the global function.

### 4. Post-SCF consumers move a streamed store to the device whole -- priority 2, medium

**Done 2026-09-29** for the consumers named below (see "Done since"), and the PAW
orientation torque of the second paragraph (`_streamed_onecenter_torque`: the pairing's
gradient in the turned `becsum` once, contracted with each chunk's forward derivative;
1.6e-16 against the whole-set `grad` on a torque of 0.04), and the orientation stepper, which
now turns a streamed store on the host and hands it to the chunked `becsum` and density
(spin-orbit PAW nickel, `rotate_moments`, five iterations: 6e-10 Ry between the stores).

`Calculation.density` has no stream branch (`driver.py:4098`) and `stm.py:205`,
`sfac.py:197` call it with `result.wavefunctions`; `projwfc/projections.py:341-343`,
`angular_momentum.py:301`, `relax.py:520` (every ultrasoft step), `anisotropy.py:780` do
the same. On the NiBr2 slab the streamed SCF keeps 12.10 GB in host RAM and one k-point's
2.0 GB on the card; any of these puts the 12.10 GB back. **Fix**: one helper that routes a
numpy store through `stream_densities` (which returns `becsum`, `rho`, `tau`, `ns`
finished), used at those sites; per-chunk loops for the projections and `<L>`.

The in-loop diagnostics do the same (priority 3, narrow): the PAW orientation torque
differentiates a whole-k `becsum(spin_turned(psi))` (`driver.py:5294-5300`) and the
orientation stepper turns the whole store on the device. Fix: forward mode per chunk
against a `d(pairing)/d(becsum)` computed once.

### 5. The Chern number and the orbital magnetization hold the whole mesh -- priority 2, medium

**Done 2026-09-29** (see "Done since"); on by default where the store streams.

`chern_number` diagonalises the whole plane at once (`invariants.py:132`,
`topology.py:271-294`) and `build_plane_wave_states` builds the whole-k `vkb` for `becp`
on US/PAW (`states.py:752-775`); the orbital magnetization takes the whole volume mesh
with the Hamiltonian kept. Bismuthene-soc-small at 12x12: 449 MB of solve output, 374 MB
of occupied copies, 424 MB more; about 5.0 GB at 24x24. Wilson-loop Z2 already streams.
**Fix**: diagonalise column by column (`indices=rows`) on one shared sphere and keep two
columns on the device for the overlaps.

### 6. The per-k basis tables stay whole on the card -- priority 2, large

What memory mode leaves resident per k-point (`driver.py:1919-1955`, `:2440`):
`npwx x (16 ncs [core columns] + 24 [kg] + 8 [kinetic] + index arrays + 1 [mask]
+ 16 npol nwfcU [DFT+U wfcU] + 24 [kplusg, meta-GGA])` bytes. Measured on Si8: 0.19 MB per
k-point (22.5 MB of columns and 8.4 MB of `kg` at 216 k). On nbse2 (`ncs = 36`) it is
6.1 MB per k-point, *more* than the 3.9 MB store that streaming moved off the card; with
DFT+U, `wfcU` alone is 18.5 MB against a 44.3 MB store on `ni10-ldau`, and it enters
every Davidson call whole through `HubbardTerm`. Three pieces, in order of cost:

* **Store the projector columns as real** -- priority 2, small. **Done 2026-09-28** for the
  projectors (bit-identical, 93.6 -> 71.1 MB on eight-atom Si at 216 k-points); the atomic
  orbitals' `i^l` is not done. `_species_columns` is
  `ylm (real) x radial (real) x (-i)^l` (`projectors.py:403-407`, `:482-488`), so each
  column is a real array times a fixed phase, stored complex128. Store float64 plus a
  static per-column phase, applied in `_apply_phases`; same for the atomic orbitals'
  `i^l`. Halves the largest per-k resident array (22.5 -> 11.2 MB on Si8 at 216 k,
  242.8 -> 121.4 MB on nbse2). Round-off only.
* **Build the core in k-chunks at setup** -- priority 3, small. **Done 2026-09-28** in memory
  mode (setup, `at_kpoints`, a spiral's rebuilt basis): 71.1 -> 40.4 MB at setup. `build_projector_core`
  runs `_angular_part`, the form factors and the radial table over the whole k set with
  every intermediate live (`projectors.py:359-415`); at 216 k on Si8 that setup transient
  (90.8 MB, measured) is the run's peak. Chunk it through `_planewaves_rows` /
  `_kpoints_rows`, which already exist for the streamed start.
* **A chunk-local Hamiltonian** -- large, and the prerequisite item 2 names too (a
  `Calculation` restricted to a row subset of k). Open. `rows(rows)` methods on `Hamiltonian`,
  `SpinorHamiltonian`, `Projectors`, `HubbardTerm`, `Sticks`, with the per-k leaves in
  host numpy and `device_put` per chunk like the store. `npw` must stay the global static
  tuple (it sets the Davidson subspace). This is what makes memory mode truly flat in
  `nk`.
  **Done 2026-09-29 for a band path, which is where it grows** (`19d4aa4`): the SCF's own
  mesh is modest and read every iteration, and QE keeps `igk_k` for every k-point too, so
  its tables stay; a band path is long and read once. `run_bands` in memory mode, where the
  store streams and the path's tables would pass `KSET_BLOCK_BYTES` (64 MB), builds each
  block of points as its own `at_kpoints`, padded to the whole path's `(npwx, nsticks)`
  (`basis.planewaves.sphere_widths`) so every block shares one compilation -- a per-block
  width would recompile per block, the Berry-string trap. NbSe2 monolayer (SCF 143 MB):
  a 91-point path **301.6 -> 206.1 MB**, 181 points **205.9 MB**, flat where it grew 1.65 MB
  a point; the bands identical to every printed digit. The Hamiltonian's static per-k
  plane-wave counts differ block to block and recompiled the solve per block until each
  block carried the whole path's minimum (`Calculation.hamiltonian_npw`, which is the cap
  the unblocked path has); since, a 240-point silicon path streamed on the CPU takes 3.34 s
  in four blocks against 3.28 s whole. **Not done**: a dense NSCF, DOS or
  PDOS mesh moved onto the same way -- its occupations need the Fermi level over the whole
  set, and the tetrahedron scheme the whole grid's corners, so it is a second design.

### 7. `projectors='rebuild'` does not reach the traced movers or the response -- priority 2

**The traced movers: nothing left to do outside item 2 (verdict, 2026-09-29).** In memory
mode each one now runs on a chunk's own k-points: the chunked force and stress move each
chunk's row-subset calculation (`at_rows`, then `at_strain`), so the core rebuilt under the
strain is that chunk's (measured flat in nk on the card, item 3), and the spiral's `dE/dq`
builds its projectors on the chunk's rows only (`forces/spiral.q_dependent_energy`). The
global pass's `at_spiral_q` still builds a whole-k set that nothing it returns reads;
whether XLA drops it from that compiled pass is not measured. The
one traced mover still taken whole-k is `at_kcart` under the velocity operator's `jvp`,
which is the response stack, item 2. Speed mode keeps the single pass by design.

**First half done 2026-09-28** (the misplaced line); **the `becsum` and response sites done
2026-09-29** (see "Done since"); the traced movers are still open.

`at_strain` (`driver.py:3050`), `at_kcart` (`:3299`) and `at_spiral_q(rebuild_basis=False)`
(`:3412`) still build a stored set; they run under the stress `grad`, the velocity `jvp`
and `dE/dq`, so check with `memory_analysis()` whether a lazy set on the tape saves
anything before switching (`MEMORY-AUDIT.md` C5). The response stack reads the whole-k
`vkb` at about a dozen sites; **the first one is a single misplaced line**:
`local_perturbation` runs `vkb = calculation.projectors.vkb` unconditionally
(`sternheimer.py:880`), outside `apply`, while its only reads sit under
`if coefficients is not None` -- so on a norm-conserving dataset a whole-k `vkb` is built
on every perturbation of every iteration and never read. On Si8 at 216 k that is 179 MB,
8x the core the dial keeps. Move it inside the guard; then pass `Projectors` and use
`at_k(ik)` in `becsum_of`, `_raw_becsum`, `mixed_becsum` and the velocity operator.

### 8. A streamed resume still lands the source set on the device once -- priority 2, small

**Done 2026-09-28** (220.8 -> 71.1 MB on the resume measured). `mmap_mode` was not needed:
the host copy is what a streamed run holds anyway.

Both halves are fixed. What the survey found, for the record: `load_state` did
`jnp.asarray` on every array including the wavefunctions (`checkpoint.py:89-90`,
`:293-295`) and `promote_wavefunctions` did `jnp.asarray(psi)`, concatenating with zeros
for a 2 -> 4 promotion (`continuation.py:903`, `:960-968`). So a streamed resume or
continuation put the whole set (12.10 GB on NiBr2, doubled by a promotion) on the device
at iteration 1. Now `load_state` keeps the wavefunctions, eigenvalues and occupations as
numpy and the promotion concatenates in numpy when its source is a host array
(`continuation.py:907`).

---

## B. Per k-point and per band: the next lever on a large cell

### 9. Memory mode never budgets the band dial -- priority 1, medium

**Done 2026-09-28** (see "Done since"). The response stack's `map_bands` sites
(Sternheimer, the q-phonon) are left at the platform default: they hold the whole k axis
anyway (item 2).

Both presets keep `band_batch = all` on a card (`batching.py:482`), a decision measured on
Si8, where a band loop costs 4.3x and buys nothing. On a large cell it is the largest per-k
term: the eigensolver's FFT line is `2 band_batch npol N_smooth x 16 B`
(`sizing.py:801-806`, a measured coefficient), which on the NiBr2 slab is **66.9 GB at
`all` against 2.65 GB at 16**, so one k-point in memory mode needs about 92.9 GB against
28.7 GB at 16 (the cluster runs set `DEFUMAT_BAND_BATCH=16` by hand). The start is worse:
`natomwfc` vectors, and D10's dump names `c128[510,2,200,240,54]` = 42.32 GB. The `nspin =
2` density stage doubles on top (A15, open). Nothing checks memory mode against the card.

**Fix**: in memory mode on an accelerator, resolve `band_batch` from the card -- the
largest `b` (preferring a divisor of `nbnd` and `natomwfc`) whose estimate times the
measured 1.6x fits -- and size the start with `natomwfc` (D10) and the `nspin = 2`
density (A15's three lines). The value needs a carrier, because `map_bands`' `"default"`
never sees the mode: a static field on the Hamiltonian, or a value resolved in the
`Calculation` and passed to `map_bands`/`sum_bands`. **Risk**: none to numbers
(`map_bands` is bit-identical across chunks); time. **Measure**: `h40-chain-lsda.in`
(one k-point, 56 bands, 40x40x640 box) in memory mode at `DEFUMAT_BAND_BATCH` = all, 28,
14, 8, 1: peak and warm time per point.

### 10. Davidson carries two band blocks it could rebuild -- priority 2, medium

**Done 2026-09-29**, bit-identical (see "Done since").

The `while_loop` state carries `evc` and `hevc` (`davidson.py:606-614`), so both are live
through the correction block's `h_psi`, which is the step's FFT peak: `2 k_in_flight
nbnd npol npwx x 16 B` -- 4.03 GB of the NiBr2 slab's 28.7 GB per-k stage at
`band_batch = 16` (14 per cent), 2 x 5.8 GiB on the 157-atom slab. `cegterg` carries the
Ritz coefficients `vc` instead. **Fix**: carry the `(nvecx, nbnd)` coefficients and the
live width, form `evc`/`hevc` at the top of the step at the same width (so the GEMM shapes,
hence the bits, are unchanged), and once more after the loop. **Risk**: must be
re-validated bit for bit on the reference cells.

### 11. Donate the streamed Davidson input -- priority 2, small

**Done 2026-09-29** (see "Done since").

`davidson.py:977-999` names the robustness retry's reuse of `psi0` as the only thing
blocking `donate_argnums`. On the streamed path `psi0` is a fresh `device_put` of a host
slice that the host still holds, so a retry can re-fetch it. `MEMORY-AUDIT.md` A16
measured that donating one input saves exactly one buffer (`alias_size_in_bytes`) and
removes an alternating large-block allocation it ties to fragmentation. **Fix**: a
donating variant of `_every_k` used only when the caller passes a way to rebuild `psi0`
(`psi0_again=lambda: _to_device(store[spin, rows])`).

### 12. The response's pair and frequency axes -- priority 2

* **Done 2026-09-29 (a budget; the card sweep was not discriminating).** Swept on the GTX
  1060 on the TDDFT silicon cell (`si-epsilon-unshifted-nosym`, 60 bands, `ecut_response =
  8`, 224 pairs): the peak is 2613.9 MB at every pair batch -- all, 1, 8, 32 and the budget --
  and `chi_0` identical, because 224 pair boxes on this grid are about 29 MB and the peak is
  set elsewhere -- **located 2026-09-29**: that script ran in speed mode, and 1867 MB of it
  is the 60-band solve with all 64 k-points in flight, the rest `chi_0`'s real-space states
  for all 64 at once. In memory mode the same chain is 110.2 MB (`PERFORMANCE.md`, "The
  2.6 GB silicon spectrum was speed mode"). The budget is unmeasured on a cell where the
  pair axis is the peak.
  **`chi_0`'s pair axis
  defaults to the band dial**, which is `all` on any card in both
  modes (`tddft/chi0.py:468`), so every pair's box is in flight -- the module's own 26 GB
  case comes back on a GPU (`D1` was closed on a CPU, where the band default is 1). Give the
  pair axis its own finite default and measure 1/8/32 on the card. Small.
* **The TDDFT frequency axis has no dial** (`MEMORY-AUDIT.md` A10, open, and
  `PERFORMANCE.md`'s backlog): the `chi_0` assembly holds `(nw, 2 npairs, nm)` per chunk
  and the Dyson loop iterates every frequency with `nw` copies of `f_xc`. Chunk `w`. Medium.
  **Done 2026-09-29** (`w_batch`, see "Done since"); on the card the silicon spectrum's
  whole chain reads 229.8 -> 226.5 MB, because the frequency block is not the peak there.
* **Done 2026-09-28, not measured on this card.** **Duplicate stacks and dead blocks in the third-derivative drivers**: `b = jnp.stack(
  internals['bare'])` and `u = ...['dpsi']` copy while the lists are still held
  (`response/nonlinear.py:469-470`), the commutator stack copies even where it aliases
  `bare` (`:506`), and `RamanTensors(field=field)` returns the internals regardless of
  `keep_internals` (`:519`). On `PERFORMANCE.md`'s P25 yardstick: 1.38 GB of duplicates and
  7.4 GB of a dead phonon `bare` at the assembly. Small and bit-identical.

---

## C. Derivative tapes: the stress, and the global part of `dE/dq`

### 13. The radial transforms are taped whole under the strain gradient -- priority 1, small

**Done 2026-09-28** (see "Done since"); the text below is the survey's, kept for the record.

`formfactors._chunked` (`formfactors.py:57-63`) is a Python loop of jitted kernels with no
remat, so every chunk's `(chunk, msh)` residuals stay on the reverse tape. `at_strain`
rebuilds `V_loc` and the core charge against the strained `|G|` inside the gradient
(`driver.py:3063-3078`). On BN both datasets carry a core correction and `ngm = 43903`, so
each float64 `(ngm, msh)` array is about 0.6 GB per transform. **This is the leading
candidate for BN's 5.57 GiB stress request** -- consistent with the arithmetic, not yet
proven by an A/B. `OPEN.md` S4 rematted `_qrad_kernel` only; `PERFORMANCE.md` backlog item
11 is the open record.

**Fix**, step 1 (S4's pattern): replace the loop with a `lax.scan` over padded `q` blocks,
body under `jax.checkpoint`, padding with `q = 0` (both kernels are finite there), the
`4 pi / Omega` prefactor outside the scan. The tape then holds the `q` vector and the
backward pass recomputes one block at a time: about 30 MB instead of about 1.2 GB per
array on BN. S4 was bit-identical on `si2-us-1k`. Step 2 (backlog item 11): a
`custom_jvp` returning `F(|G|)` and `dF/d|G|` from the closed forms already transcribed in
`stress/analytic.py`. **Measure**: compile only -- lower `jit(grad(strained_energy))`
with the state as `ShapeDtypeStruct`s at BN's shapes and read
`memory_analysis().temp_size_in_bytes` before and after; then run BN's stress on the card.

### 14. The projector and atomic-orbital transforms are taped per |k+G| -- priority 1, small

**(b) done 2026-09-28** with item 13, and it removed the growth in nk (measured). (a)
and (c) remain, and (a) is now a saving in time rather than in memory.

Under a strain, `build_projector_core` calls `projector_form_factors` on the flattened
`nk x npwx` `|k+G|` (`projectors.py:382-388`), one `_chunked` call per projector, each
with a `(chunk, kkbeta)` intermediate (`formfactors.py:331-334`). So **the stress tape
grows linearly with nk** -- which contradicts `PERFORMANCE.md:1482-1483` ("independent of
nk"), a sentence to correct. **Fix**: (a) one Bessel transform per `(dataset, l)` rather
than per projector (stack the rows of each `l`, as `_qrad_kernel` does -- 3x fewer on BN);
(b) item 13's rematted scan; (c) optionally QE's `tab_beta` on `dq = 0.01` with the
four-point stencil the augmentation already has -- this moves every nonlocal quantity by
the interpolation error, so it stays opt-in. **Measure**: `memory_analysis()` of the
stress on Si8 at 8/27/64/216 k before and after; report MB per k-point.

### 15. The stored augmentation table is assembled on the strain tape -- priority 2, small

**Done 2026-09-28** (see "Done since"; `PERFORMANCE.md`, "The augmentation table on the
stress tape"). **The spin spiral's displaced table `Q_ij(G - q)` was tried the same way on
2026-09-29 and measured worse**: routing `at_spiral_q`'s traced rebuild through the scanned
class (with the table's `|b|` margin skipped for the exact integral, so a traced shift
passes) gave the same `dE/dq` to 4e-17 on the ultrasoft oxygen chain, and made the global
`value_and_grad`'s compiled temporary **larger** -- 218.6 -> 225.9 MB on the oxygen chain,
**261.7 -> 834.4 MB** on bcc iron (`Fe.pz-nd-rrkjus`, `nh = 18`, `ngm = 6963`, `spiral_q =
0.1`, CPU). The stored displaced table is small on every augmented spiral cell in the tree
and the scanned route's per-chunk radial integral outweighs it; the change was reverted.
It may win on a spiral with a far larger `ngm`, which has not been sized.

**Measured 2026-09-28, and the prescription below does not yet hold.** After items 13/14
this is the largest thing left on BN's stress tape: its 1.42 GiB (CPU) is headed by
`c128[196,43903]` and a dozen `f64[196,1,1,1,43903]`, `nh^2 x ngm` pair blocks. But with
`DEFUMAT_AUG_MAX_BYTES=0`, which sends the table through the tabulated scan named below,
the same executable is **7.47 GiB**, fifteen whole-`ngm` complex pair tables. **Located the
same day**: the HLO places them in the backward pass of the tabulated route's own rematted
scan (`transpose(jvp(_addusdens))/while/body/checkpoint`), and each is one *chunk* --
`_aug_chunk` sizes the chunk for about 256 MB of forward block, which at `nh = 14` is more
than the whole G set, so BN runs one chunk of all 43903, and one chunk's backward holds
about fifteen blocks of it. With the chunk set small (`DEFUMAT_AUG_CHUNK`), same route,
same compile: **0.466 GiB at 4096, 0.450 GiB at 1024**, against 1.42 for the stored table.
So the prescription below holds **with a chunk sized for the backward pass** (a sixteenth
of the forward budget, or a separate budget under a derivative); the exact builder in place
of the interpolation is still what keeps the stored route's numbers.

`at_strain` calls `build_augmentation` from scratch (`driver.py:3054-3057`); below
`AUG_MAX_BYTES` the stored route runs `_assemble_qgm` over strain-dependent `ylm` and
radial blocks (`augmentation.py:1136-1157`), so the tape holds per-`L` `(nh, nh, ngm)`
blocks and `qgm` itself: on bismuthene-soc-small (`nh = 34`, `ngm = 60543`, `qgm` 1.12 GB,
stored) between 69 and 275 MB per `L` plus `qgm`, arithmetic. **Fix**: under a strain,
route stored datasets through the rematted G-chunk scan the tabulated ones use
(`_tabulated_charge`), with an exact builder (S4's `_qrad_block`) instead of the
interpolation.

### 16. Forward-mode stress, and forward-over-forward for piezo and elastic -- priority 2/3

**Not needed for the stress on any cell here (verdict, 2026-09-29).** Measured on the
card: in memory mode the chunked stress adds about 22 MB to the SCF's peak whatever the
mesh (eight-atom Si, 57.5 MB at 27 k-points and 62.6 at 64), and bismuthene's SCF plus
stress peaks at the SCF's own 2393.5 MB; BN's stress, the one that died, runs at 2052.9 MB.
The reverse tape is one chunk's, which is what a forward method would have bought, at 9x
the work. The piezoelectric and elastic assemblies are the response stack's and go with
item 2.

A `forward` stress method -- nine `jax.jvp` with unit strains, compiled once -- holds the
forward working set plus one tangent instead of the reverse tape, for about 9x the work;
`MEMORY-AUDIT.md` C1 is "one measurement from decidable". Select it in memory mode when
the reverse tape would not fit. The piezoelectric and elastic assemblies are
`jvp`-of-`grad` through `at_strain` and carry the whole reverse tape plus its tangent; a
nested-`jvp` mode removes the reverse pass (`MEMORY-AUDIT.md` C1, `OPEN.md` H7). Both
matter less once items 13-15 land -- measure first.

### 17. Compiled gradients still embed k-sized constants -- priority 2, small

**Done 2026-09-29** by extending `HOISTED_FIELDS` rather than partitioning the whole
`Calculation` (see "Done since").

`HOISTED_FIELDS = ('paw', 'augmentation')` (`forces/energy.py:109`); the force and stress
`jit`s close over the `Calculation`, so the projector core, `wfcU` and the `(nsym, ngm)`
symmetry maps become executable constants. Hoisting the first two took one-atom Pt's
stress constants from 1062 MB to 10.1 MB (`PERFORMANCE.md`, "A gigabyte of constants");
the rest is 253 MB of core on nbse2. **Fix**: split the `Calculation`'s array leaves with
`eqx.partition` and pass them as an argument, so no large array can become a constant.
Whether XLA:GPU keeps a second resident copy of a constant was unmeasured and decided the
size of the win: **it does not** (measured 2026-09-29, see "Done since"), so the win is in
how long a dropped table stays on the card, not in the peak.

---

## D. k-independent resident objects

### 18. The PAW one-centre tensors are rank-1 products kept whole -- priority 1, medium

**Done 2026-09-28**: Pt's peak 1309.6 -> 411.3 MB, energy moved by 1e-13 Ry
(`PERFORMANCE.md`, "The PAW one-centre tensors, factored"). The meta-GGA kinetic tensors are
still products (they are built by quadrature, not as one coefficient times one radial
function) and are what is left of this item.

`density_ae`, `density_ps` and, since P117, `density_rel` are `einsum('lij,ijr->ijlr')`
outer products put on the device (`paw/onecenter.py:884-925`). For
`Pt.rel-pbe-n-kjpaw_psl.0.1` (`nh = 34`, `nlm = 25`, `mesh = 1277`) each is 295.2 MB:
**885.7 MB of the ~1.3 GB measured Pt spin-orbit PAW peak**, the same in both modes.
`MEMORY-AUDIT.md` A12 is half done (hoisted out of the executables, not factored);
`density_rel` postdates it. **Fix**, QE's `PAW_rho_lm`: store the factors and contract
`becsum` with the coefficients over `(i, j)` first, then with `pfunc` over `(beta_i,
beta_j)`, inside the chunked one-centre kernel; no `(nh, nh, nlm, mesh)` array is formed.
**Risk**: round-off (reassociation); check the PAW stress residue and the `pw.x`
comparisons for spin-orbit PAW Pt. **Measure**: `pt-soc-paw-nosym.in`, peak in both modes,
expected to fall by about 0.88 GB with the energy identical to 1e-12 Ry.

### 19. `Q_ij(G)` is a real table times a known phase -- priority 2, medium

**Done 2026-09-29** (see "Done since").

`_assemble_qgm` sums `(-i)^l x` real Gaunt x real `Y_LM` x real radial terms, and real
Gaunt coefficients vanish unless `l_i + l_j + L` is even, so `Q_ij(G) = (-i)^(l_i+l_j)`
times a real number, stored complex128: 1.12 GB for bismuthene's relativistic dataset, and
the assembly holds about 1.9 GB of transient beside it. **Fix**: store the real table with
a static per-pair parity and fold the phase into `becsum`; the charge becomes two real
contractions. The displaced spiral table has the same structure. **Risk**: the forbidden
entries are ~1e-16 rather than 0 (the Gaunt table comes from a matrix inverse), so the
check is round-off, not equality.

### 20. Post-SCF workflows build a second `Calculation` -- priority 2, medium

**Done 2026-09-29** (see "Done since").

`get_bands`/`get_nscf`/`get_dos` pass no calculation, so `nscf.py:196-197` builds another
beside the facade's own, which stays alive; `pdos`, `stm`, `sts`, `transport`, `sfac` (with
`kpoints=None`), `DFTSource._base` and `run_spiral_scan` do the same, and the `Q_ij(G)`
dedup is local to one build, so each copy has its own table (1.12 GB on bismuthene). These
setups also do not see the calculator's `memory_mode` or `projectors`
(`docs/features.tex` says so in the amber box; `OPEN.md` Part XVII item 4 mentions
`at_kpoints` in passing). **Fix**: give the workflows a `calculation=` argument
(`fixed_density_states` has one) and pass `self.calculation` where the k-set is the SCF's,
`self.calculation.at_kpoints(k)` elsewhere -- `at_kpoints` now keeps the projector dial.

### 21. The relaxation loops keep the previous step alive -- priority 2, small

**Done 2026-09-28.**

`vc_relax`'s `previous = current` (`:345`) stays bound through the next iteration's SCF,
force and stress; `run_spiral_scan` (`spiral.py:304-316`) and `relax_spiral_q` (`:591`)
keep the previous result through the next SCF. `MEMORY-AUDIT.md` A2's fix was applied to
`run_relax` only. Under the stress, `at_cell` goes through `at_strain`, so `base`,
`previous` and `current` each hold their own augmentation table and core -- at least one
extra 1.12 GB `Q_ij(G)` on bismuthene. **Fix**: A2's lines -- `result = None` at the top of
each loop body, `del previous` after `_advance`. No numbers move.

### 22. In speed mode the cached ground state stays on the device -- priority 3, small

**Measured, and not worth changing what speed mode means (2026-09-29).** On
`bismuthene-soc-small` in speed mode the cached store is **21.8 MB** and the SCF + 13-point
band path peaks at 3203.1 MB (3203.9 on a second run), so parking the store on the host
would save at most 0.7 per cent of the peak while making every later consumer stream. Left
as it is unless a cell turns up whose store is a real fraction of its peak.

`Calculator._scf` and a derived calculator's seed keep a device store under every later
`get_*` in speed mode (`calculator.py:767`, `:2243`). Park it to host after the SCF (35 ms
per 102 MB each way on this card) once item 4 makes every consumer accept a host store.
The seed half is `MEMORY-AUDIT.md` A6(iii)/B2, open.

---

## E. Runtime and infrastructure

### 23. The device pool is left at JAX's 75 per cent -- priority 2, small

**Done 2026-09-29** (see "Done since"). The allocator alternatives (`cuda_async`, `vmm`)
are still untried.

`XLA_CLIENT_MEM_FRACTION` (and its deprecated `XLA_PYTHON_CLIENT_MEM_FRACTION`; setting
both raises) is untouched by the package; on this card the pool is 4.76 GB of 6. About 0.9
gives roughly +0.95 GB here, and the H200 entry in `PERFORMANCE.md` died 5 GiB over the
default limit. **Fix**: in `defumat/__init__.py`, before the backend initialises, set it
to ~0.9 when neither name is set and the mode is not `speed`. The installed jaxlib also
accepts `cuda_async`, `vmm` and `address` allocators, which are the untried half of A16's
fragmentation question. A capacity lever rather than a reduction.

### 24. `sizing.py` misses the stages that set the peak -- priority 3, an enabler

**Partly done 2026-09-28**: `wfc_store` and the start (D10) are in; **2026-09-29** the
per-k tables, the symmetry maps and `wfcU` (see "Done since"). The core build's transient
and the `qgm` accumulator are still absent.

`peak_bytes` is `resident + max(eigensolver buffer, augmentation Bessel transient)`
(`sizing.py:320-335`): the start (D10), the core build, the `qgm` accumulator, `wfcU`, the
symmetry maps and the index arrays are absent, and there is no `wfc_store` argument, so the
1.6x error on this card is covered by `SPEED_HEADROOM = 0.6`. Adding those lines is what
items 9 and 16 need to choose a band batch or a stress method from the card.

### 25. k-axis sharding is a throughput lever, not a memory one -- priority 3, large

`GPU.md` Phase 4 plans sharding the k axis. Once the SCF streams, that buys `nk / ndev`
time and no per-device memory: every device replicates the k-independent set, which is
what sets the bismuthene and Pt peaks in both modes, and a single k-point's solve is not
divided. QE's memory lever is distributing the plane waves (sticks): shard `npwx` and
`ngm` with `jax.sharding`, using the stick layout `basis/sticks.py` already builds, with
one all-to-all between the z and xy passes and a `psum` for the projections. Strategic, to
record in `GPU.md` rather than to start.

### 26. A float32 tier would halve every device array -- priority 3, large

**Done for the band side, 2026-10-01** (`PLAN.md` P126): `band_precision = 'single'` halves the
band-sized arrays and with them the 64-atom silicon peak on the card (1.65 against 3.15 GiB); the grid
side stays double by design, and ultrasoft, PAW, spinors and DFT+U are refused. The paragraphs below are
the item as it stood.

The arithmetic is exact (`zc` 16 -> 8), but no float32 SCF has ever run, so what blocks one
is unknown (`GPU.md` §4 item 2 and Phase 3; `AUDIT-2026-09-18.md` names a Sternheimer carry
mismatch and dtype-less constructions). Build the tier on this card -- eight-atom Si,
`precision=SINGLE`, memory mode -- keeping the density accumulation and the Davidson
overlaps in float64 and `jax_default_matmul_precision='highest'`, before exposing it.

**The first blocker, named (2026-09-29, time-boxed).** `benchmarks/si8-1k.in` built with
`precision=SINGLE` fails in its first Davidson step, 1.5 s in: `lax.dynamic_update_slice
requires arguments to have the same dtypes, got complex64, complex128` (`davidson.py`, the
expansion written into the basis). The states are complex64 and `H|psi>` comes back
complex128, because **the setup arrays built from the pseudopotentials' radial tables do
not read the precision policy**: `vltot` is float64, the starting density is float64 and
with it `v_scf`, and the bare `D_ij` is float64 (the projector core's columns are float64
too, but `vkb` comes out complex64). Casting `vltot` alone does not get past the same line.
The fix is one cast of every real and complex setup leaf to the policy's dtypes where
`Calculation` is built and wherever an `at_*` mover rebuilds one, with the float64
exceptions the paragraph above names kept deliberately -- a design decision rather than a
line, so it stops here. `sizing.py` already says `_build_species` does not read the policy.
