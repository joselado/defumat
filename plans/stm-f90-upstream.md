# Draft: `first_band` is not reset per k-point in `PP/src/stm.f90`

*A draft for the user to file with Quantum ESPRESSO, written 2026-10-11 (branch `round3-k`). It
is not sent. Everything below is meant to stand on its own with `pw.x`, `pp.x` and the inputs
quoted in it; the numbers are the ones recorded in `OPEN.md` Part XXIV item 1, measured with the
7.5 release.*

---

**Title.** `stm.f90`: `first_band` keeps the previous k-point's value when no band lies below the
window, so a state inside the window is left out of the STM image

**Version.** QE 7.5, `PP/src/stm.f90`, serial build.

**What the code does.** `first_band` and `last_band` are declared at line 52 and are never
initialised. In the loop over k-points they are set only by a band that lies below `down1` and
`up1` respectively (lines 146 to 149):

```fortran
  DO ik = 1, nks
     DO ibnd = 1, nbnd
        IF (et(ibnd,ik) < down1) first_band= ibnd+1
        IF (et(ibnd,ik) < up1)   last_band = ibnd
     ENDDO
     istates = istates +  (last_band - first_band + 1)
```

So at a k-point whose lowest band is already above `down1 = down - 3 degauss`, nothing assigns
`first_band`, and the band sum that follows starts from whatever the previous k-point left there,
or from an undefined value if that happens at the first k-point. The declaration's own comment
calls `first_band` the "first band close enough to the specified energy range", so the intent is
every state within `3 degauss` of the window, and a state inside it is dropped instead. The same
variables bound the `gamma_only` branch, so both branches are affected.

**Reproduction, with `pw.x` and `pp.x` only.** fcc aluminium, one atom, on the whole unshifted
4x4x4 grid with no symmetry, Marzari-Vanderbilt smearing at 0.05 Ry, the default six bands:

```
 &control
    calculation='scf', prefix='al', outdir='./out'
 /
 &system
    ibrav=2, celldm(1)=7.50, nat=1, ntyp=1,
    ecutwfc=15.0, nosym=.true., noinv=.true.,
    occupations='smearing', smearing='marzari-vanderbilt', degauss=0.05
 /
 &electrons
    conv_thr = 1.0d-12
 /
ATOMIC_SPECIES
 Al  26.98 Al.pz-vbc.UPF
ATOMIC_POSITIONS (alat)
 Al  0.00  0.00  0.00
K_POINTS automatic
 4 4 4 0 0 0
```

then the zero-bias STM image, and every single-state density in one run:

```
 &inputpp
 prefix='al', outdir='./out', filplot='al.stm.b20', plot_num=5, sample_bias=0.0
 /
```

```
 &inputpp
 prefix='al', outdir='./out', filplot='psi', plot_num=7,
 kpoint(1)=1, kpoint(2)=64, kband(1)=1, kband(2)=6, lsign=.false.
 /
```

The Fermi level is 7.9097 eV (0.5813526 Ry). At the six `W` points of the grid, k-points 28, 31,
40, 46, 55 and 58 in `pw.x`'s order, `(-1/2, -1, 0) 2pi/a` and its images, the lowest band lies
0.0360 Ry below the Fermi level, which is inside the window of `3 degauss = 0.15` Ry, so no band at
those k-points is below `down1` and `first_band` keeps the value 2 from the k-point before. Band 1
is therefore left out at each of the six, each with a weight `wk * w0gauss` of 0.0173.

**How large it is, and how it was attributed.** The `plot_num = 7` dumps give `|psi_nk|^2 / omega`
for one state at a time, so the `plot_num = 5` image can be rebuilt from `pp.x`'s own states, with
the eigenvalues, weights and Fermi level read from `data-file-schema.xml`, and with the window
rule written two ways:

| image rebuilt from the single-state dumps | largest difference from `al.stm.b20` |
|---|---|
| the rule as written in lines 146 to 149 (`first_band` not reset per k) | 1.2e-12 |
| `first_band` reset at every k-point, which is every state within `3 degauss` of the window | 2.602e-3 |

on an image whose peak is 8.66e-3, so the six dropped states account for all of the difference and
nothing else in the sum does. On this cell a shifted 4x4x4 grid has no k-point whose lowest band is
within `3 degauss` of the Fermi level, so `first_band` is assigned at every k-point and the two
rules select the same states there; the defect shows only where a grid holds such a point.

**Suggested fix**, two lines at the top of the k loop:

```fortran
  DO ik = 1, nks
     first_band = 1
     last_band = 0
     DO ibnd = 1, nbnd
        IF (et(ibnd,ik) < down1) first_band= ibnd+1
        IF (et(ibnd,ik) < up1)   last_band = ibnd
     ENDDO
```

With `last_band = 0` a k-point with no band below `up1` contributes no states and adds zero to
`istates`, where today it would reuse the previous k-point's `last_band` as well.
