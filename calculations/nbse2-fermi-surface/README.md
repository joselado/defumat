# 1H-NbSe₂: the Fermi surface, and which of it an electron can tunnel out of

The Elk calculation behind elkpy's momentum-resolved tunnelling runs, done here
instead, and compared with it quantity by quantity. **The point of doing it
twice is that there is no third code**: neither `pw.x` nor stock Elk computes a
momentum-resolved tunnelling weight, so until now every check on it closed
inside one implementation.

## The physics

NbSe₂'s Fermi surface is a small pocket at the zone centre and two large ones at
the zone corners, and the corner pockets carry most of the density of states. An
electron tunnelling *vertically* out of the layer comes almost entirely from the
zone centre anyway, because a state at in-plane momentum `k_par` has a
faster-decaying tail in the vacuum,

```
psi(z) ~ exp( -sqrt(kappa_0^2 + |k_par|^2) z ),
```

and a few ångström of gap turns that into an order of magnitude. So the
tunnelling picture of this metal and its density-of-states picture are different
pictures, and the size of the difference is what both codes compute.

## Files

| file | what it is |
|---|---|
| `nbse2.in` | the calculation, in `pw.x` format, with the Elk mapping written out term by term in its header |
| `nbse2-24.in` | the same at 24×24×1, which is elkpy's Triton ground state |
| `smoke.in` | a reduced-cost copy used to shake the path out locally |
| `ground_state.py` | one SCF, then `D(E_F)` on the same mesh and with the same delta Elk uses |
| `tunnelling.py` | the momentum-resolved weight, the pocket shares, the height sweep and the decay identity |
| `plot.py` | the figure: the two Brillouin-zone maps and the sweep |

`state/` holds the converged wavefunctions and is gitignored: 43 k-points of 24
bands on a 9804-plane-wave sphere is 167 MB, which is a scratch file and not a
record. Delete it and `ground_state.py` rebuilds it in about twelve minutes.

## The Elk input, term by term

| Elk | here | note |
|---|---|---|
| `avec` | `ibrav = 4`, `celldm(1) = 6.5044373210`, `celldm(3) = 5.2295177223` | Elk's matrix *is* QE's hexagonal convention; `a2 = (-1.721, 2.980859, 0)` Å both ways |
| `atoms` / `atposl` | `ATOMIC_POSITIONS (crystal)` | the Se in-plane pair is the CIF's **truncated** 0.3333/0.6667, not exact thirds — see below |
| `xctype 20` | PBE | which is what the pseudopotential headers say, so no `input_dft` |
| `stype 3`, `swidth 0.001` | `smearing = 'fermi-dirac'`, `degauss = 0.002` | Elk's `swidth` is `k_B T` in **Hartree**; QE's `degauss` is `k_B T` in Ry. The two deltas are the same function: Elk's `sdelta_fd` is `e^-x/(1+e^-x)^2` and QE's `w0gauss` at `ngauss = -99` is `f(1-f)` |
| `ngridk 12 12 1`, `vkloff 0 0 0` | `K_POINTS automatic 12 12 1 0 0 0` | Γ-centred both ways |
| `spinpol .false.` | nothing | a **physics** choice: DFT sits on a Stoner instability for this monolayer, and a spurious moment would destroy the Fermi surface the calculation is about |
| `spinorb .false.` | the scalar-relativistic datasets | the `.rel-` pair is committed beside them for the Ising splitting, which is a different calculation |
| `tshift .false.` | nothing | defumat never moves the origin. It is mandatory in Elk here: it otherwise shifts the origin onto a symmetry centre while the plane heights stay in the input frame |
| `epspot 1e-7` | `conv_thr = 1.0d-10` | tightened, because the tunnelling weights are post-processed off this density |
| `rgkmax 7`, `gmaxvr 12`, `lmaxapw`, `lmaxo` | — | LAPW basis knobs with no plane-wave counterpart. `ecutwfc = 60 Ry` is where the total energy is settled to 0.5 mRy and `E_F` to 0.13 mRy over a 50–80 Ry sweep |
| 12 empty states | `nbnd = 24` | against 13 occupied: 25 valence electrons, Nb 13 and Se 6 each |
| `mixtype 3` | the default Broyden mixer | |

### Why the Se coordinates are 0.3333 and not 1/3

They are the CIF's, truncated, and the 3.3e-5 offset is above **both** codes'
symmetry tolerance. So the three-fold is broken and the crystal has 4 operations
out of the lattice's 24 rather than 12. Exact thirds are the same physics on a
different irreducible k-set, and would not be this input.

That both codes find **4** without being told is the first thing the two setups
agreed on, and it was not arranged. Where they differ is what happens next:
defumat reduces 144 k-points to **43** where Elk keeps **78**, which is time
reversal on top of a mirror that does nothing to a k-grid at `k_z = 0`.
Everything k-resolved here runs the whole grid anyway.

## What is comparable, and what is not

**Not comparable: the total energy.** Elk's −8681.0278332 Ha is all-electron
LAPW; a pseudopotential code has nothing to put beside it. It is not compared.

**Comparable and compared**: `D(E_F)`, the Fermi-surface pocket shares, the
tunnelling suppression, the reweighting, and the vacuum decay constants. Two of
those are ratios and are free of the tunnelling prefactors, which are unfixed on
both sides.

**The pseudopotential side is settled separately, against `pw.x`**, on the
identical input: −152.24534332 Ry here against −152.24534322 Ry, 1.0e-7 apart,
at 1.5× per SCF iteration single-core. That is the check an all-electron
comparison cannot give, and it is done first for that reason.

## Conventions that have to travel with every number

- **The delta's name and width.** A Gaussian and a Fermi-Dirac delta of the same
  width differ by a factor 2.1 in full width. The ground state uses
  Fermi-Dirac at Elk's `swidth`; the junction runs use a **Gaussian** at
  `eta = 0.003 Ha`, agreed with elkpy because their export window ends eight
  widths out and a Fermi-Dirac delta still carries 1.3e-3 of its peak there.
- **`occmax`.** Elk's `FERMIDOS.OUT` is in states/Hartree/cell with the spin
  degeneracy already in it (`occupy.f90:93`, `fermidos = fermidos*occmax*t0`),
  so 64.393 states/Ha is 32.197 states/Ry. That was read out of the source
  rather than assumed; a factor of two here would look exactly like a physics
  disagreement.
- **Which Fermi level.** The delta is centred on the **SCF's** level, not the
  dense grid's. Elk's task 9007 has no `occupy` call at all: it takes `efermi`
  from `readstate` (the binary `STATE.OUT`, written by `writestate.f90`) and
  evaluates its energy window once before the k-loop, so its eigenvalues are on
  the dense mesh and its chemical potential is the coarse one, frozen. Both
  codes therefore have an absolute total that is *not* a converged
  transmission — one more reason the reweighting and the per-k weights are the
  objects to lead with. The dense-grid level is reported anyway, because its
  shift *is* the error that freezing leaves.
- **The zone partition is 2/3 before any physics.** Closer to a corner than to
  Γ gives the corners exactly two-thirds of a hexagonal zone by area. A bare
  share near 70% at `K` is therefore close to saying the Fermi-level density of
  states is near-uniform per unit area — it is not "`K` dominates". The
  tunnelling share, and the reweighting, are where the physics is.
- **The mesh.** Every share is quoted with the mesh beside it. This is a laptop:
  24×24×1 is reachable and elkpy's 54×54 and 162×162 are not.

## The floor each code has, and they are different

Neither result means anything at a single tip height, because both codes stop
resolving the vacuum tail somewhere and a contrast quoted at one height cannot
be told from that.

- **Elk's** is the plane-wave floor: the interstitial sum's reach is set by
  `rgkmax`, the large-`|k|` pocket hits it first, and past ~3.5 Å the `K` weight
  flattens at ~1e-6 and the apparent contrast *falls*.
- **defumat's** is not a basis floor at all — a norm-conserving plane-wave
  sphere resolves the tail as far as the tail goes. It is the **periodic
  image**: in an 18 Å cell the vacuum is 14.6 Å, so a tip 6.5 Å above the top Se
  is 8.1 Å from the image slab below, a ~4% contamination at `kappa ~ 1 1/Å`,
  against 2e-7 at 3.5 Å.

So the sweep here runs 1.5–4.5 Å and `kappa` is fitted over **1.5–3.5 Å**, which
is elkpy's range, with a `kappa` per interval reported beside it so the
asymptote can be read rather than trusted.

## The check that shares no machinery with either code

Two pockets, two decay constants, and

```
kappa_K^2 - kappa_Gamma^2 = |K|^2,      |K| = 4 pi / (3 a) = 1.2172 1/A.
```

Nothing in either implementation enters it. `kappa` is the **amplitude** decay
and the weight is quadratic in the wavefunction on each plane, so a sweep of the
tip alone has `d ln W/dz = -2 kappa`; halving on one side only makes the identity
come out four times too large, which is what it is there to catch.

**What it does not measure.** It is tempting to read the residual's sign: `k_G`
is a pocket *centre*, so a zone-centre pocket of radius `k_0` would raise
`kappa_Gamma` and pull the difference low by about `k_0^2`. That reading does not
survive refitting. On the Elk sweep the residual is 1.4% low fitting Γ over
1.5–6.5 Å and 0.8% **high** once the non-asymptotic first interval is dropped,
and the scatter between fit ranges is four times the deficit any plausible pocket
would explain. The identity holds at the **1% level** and that is what it
establishes — the exponent, and that `kappa` is the amplitude decay on both
sides. No physical residual can be read out of it at this precision, and an
explanation that merely *fits* the number is not thereby established.

## The two codes' artefacts are different, which is why agreeing means something

Both codes stop resolving the vacuum tail somewhere, and the two failures have
the same signature — a decay that stops decaying — from opposite directions:

- **Elk's is pocket-selective and hits the pocket the result is about.** `K`'s
  true tail falls below what `rgkmax` can represent and flattens onto ~1e-6
  while Γ stays straight.
- **defumat's is not pocket-selective.** The tip approaches the periodic image
  slab from the other side, contaminating everything equally.

So a Γ/K contrast that agrees between the two codes over 1.5–3.5 Å is not two
codes sharing an artefact. Neither implementation can make that statement alone.

## Running it

```bash
python3 calculations/nbse2-fermi-surface/ground_state.py nbse2.in   # ~12 min
python3 calculations/nbse2-fermi-surface/tunnelling.py 24           # the run
python3 calculations/nbse2-fermi-surface/plot.py 24                 # the figure
```

The second is the expensive one and it is a whole 24×24×1 grid — 576
diagonalisations, no symmetry, because `W(k)` *is* a function of `k` and there is
nothing for a symmetry sum to do to it. The height sweep on top is free: the
bands do not know where the tip is.
