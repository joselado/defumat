# The transverse block of an augmented spiral on the laboratory sphere (`OPEN.md` Part XXI item 1)

*Written 2026-10-09 on branch `open-a`, for a fable review before any code. Nothing below is
implemented. The scripts are in `plans/open-a-scripts/` (run on D22 through `run.sh`, which sets
the environment and the 6 GB cap); their output is in `/l/ladovj1/review/open-a/` on D22.*

## What was measured, and what it says the cause is

The entry's identity, reproduced on D22-0161 (`review/open-a/xxi1.py`, four performance cores,
one process): `co-helix4-spiral.in` converged, unfolded onto `co-helix4-nosoc.in`
(`unfold_spiral_density`), diagonalised once in the supercell at that density, and read
three ways. In every row the supercell's box is the spiral's repeated (no warning), so the
grid-repeat trap of P123 is not in play.

| `ecutrho` (`ecutwfc = 20`) | 80 | 160 | 160, LDA (`pz`) | 320 |
|---|---|---|---|---|
| worst eigenvalue, supercell against the union of the four spiral k-points | 3.93e-3 Ry | 7.27e-5 | 1.77e-5 | 2.5e-8 |
| the same, bands below `E_F + 0.1` Ry | 7.3e-5 | 3.4e-6 | 2.5e-6 | 8.4e-9 |
| output magnetization against input, supercell | 2.25e-2 | 4.20e-3 | 4.39e-3 | 1.36e-4 |
| output magnetization against input, **the spiral against itself** | 7.3e-6 | 1.25e-5 | 2.1e-6 | 1.35e-4 |
| share of the transverse difference `Delta m_+(G')` on the shell `{G' : |G'| < G_rho, |G' + q| > G_rho}` | 0.915 | **0.995** | **0.9998** | 5e-4 |
| that shell's share of the sphere's G-vectors | 9.1 % | 7.0 % | 7.0 % | 4.3 % |

(The 320 Ry spiral did not reach `conv_thr = 1e-11` in 150 iterations, 8.8e-9, and its own
one-shot is then 1.35e-4 off; the supercell matches it to that level, so at 320 the identity
holds to the spiral's own self-consistency and the eigenvalues to 2.5e-8.)

**The cause is the one the entry ended on, and it is a truncation, not a missing term.** The
supercell keeps the augmentation charge on its own sphere, `|G'| < G_rho`; the spiral keeps
its transverse block in the rotating frame on `|G| < G_rho`, which is `|G' + q| < G_rho` in the
supercell's labels, so the two discretise the transverse augmentation charge on two spheres
displaced by `q`. The difference the supercell's one-shot puts back sits 99.5 per cent (PBE)
and 99.98 per cent (LDA) on exactly the G-vectors one sphere has and the other does not, while
the charge and `m_z` differences have 2e-5 there (they are not displaced). It is the dense
sphere's edge and not the smooth one's: at 160 Ry the smooth sphere has radius 8.94 1/bohr and
the shell is at 12.64. It falls with `ecutrho` as the tail of cobalt's `Q_ij(G)` at `G_rho`,
by 2900 between 160 and 320. GGA is excluded by the LDA column. **P95's oxygen chain has the
same shell**, measured with the same script on `o-chain-spiral-us.in` against
`o-chain-90deg-us.in` at their 200 Ry: worst eigenvalue 3.4e-8 Ry, magnetization 6.7e-5 off
against the spiral's own 1.2e-6, and 99.8 per cent of the transverse difference on the shell
displaced the other way (its `q` is `+b3/4`). P95's 1.65e-9 Ry was a total energy, and the shell
is there on its cell too, so the entry's contradiction with P95 dissolves.

## What the change would be

Keep the transverse pair on the **laboratory** sphere in the rotating frame:
`{G : |G - q| < G_rho}` (sign to be fixed by the shell measurement above: the missing set is
`|G' + q| > G_rho`), where the charge and `m_z` stay on the resident `|G| < G_rho`. Then the
spiral and the supercell discretise the same functional and the identity becomes exact at any
cutoff, which is the property P95 and P122 assumed.

What it touches, from reading the code (`scf/driver.py`, `pseudo/augmentation.py`):

1. **A second dense G set for the transverse pair.** `build_augmentation(shift=-q)` already
   evaluates `Q_ij(G - q)`; it is given the resident `dense` set (`driver.py:2435-2439`). It
   would be given a displaced `GVectors` instead, the same Miller indices that satisfy
   `|G - q| < G_rho`, on the same box.
2. **The box.** The displaced sphere reaches `G_rho + |q|` along `q`. The spiral's dense box at
   160 Ry is `(20, 20, 25)`; whether the extra `|q| = 0.263` 1/bohr still fits under the box's
   Nyquist on every axis has to be checked per cell, and where it does not the box grows by one
   good FFT size along that axis. That is the only memory cost, and it is the dense box only.
3. **Every consumer of the transverse pair in G.** `_addusdens_spiral` (`cross.charge`),
   `_newd_noncollinear_integrals` (`transverse_g = r_to_g(potential[1] + i potential[2],
   dense.fft_index)`, `driver.py:4666-4671`), and anything that gathers the density on the
   dense sphere before mixing or for the energy (the Hartree term takes the charge only, so it
   is untouched; the exchange-correlation term is evaluated in real space and is untouched).
   The dense-to-smooth interpolation of the transverse potential has the same displacement at
   the smooth sphere's edge; it measured at most 0.5 per cent of the difference here and is
   left out of the first change, stated as such.
4. **`dE/dq` (P96).** The displaced sphere's membership is a step function of `q`, like
   `|k + q/2 + G|^2 <= ecutwfc`, so the gradient is taken at a frozen sphere, which is what
   `at_spiral_q` already does for the wavefunction spheres; the displaced dense set has to be
   rebuilt there too, and the compiled gradient dropped with it.
5. **PAW.** The one-centre terms do not see the grid and are unchanged; the grid half of a PAW
   spiral takes the same displaced set.

## How to know it worked

- The table above at 80 and 160 Ry: the worst eigenvalue falls from 3.9e-3 and 7.3e-5 Ry to
  the spiral's own self-consistency, and the shell share of `Delta m_+` stops being the
  dominant part. The **control** is the same run with the displaced set swapped back for the
  resident one, which must reproduce the 7.27e-5.
- `q = 0` still reduces to the ordinary noncollinear run (the displaced set is then the
  resident one, so this is a bit-identity, `test_a_zero_spiral_is_an_ordinary_noncollinear_run`).
- P95's oxygen numbers re-run; they are expected to move in the last digits only, and a move
  larger than that is a finding.
- P122's helix source rebuilt at a cutoff where the identity holds (320 Ry measured here), and
  the helix's plane and its curvature re-read from it, because P122 states that sign as owed
  to this entry.

## What is the user's to decide, and is not decided here

Whether to change the spiral's discretisation at all, against documenting it as a convergence
in `ecutrho` (2.5e-8 at 320 Ry on this cell) and adding a diagnostic that reports the shell's
share of the transverse augmentation charge on every augmented spiral run. The change moves
P95's and P96's validated numbers in their last digits, and the default discretisation of a
spiral is a choice this project has so far left to the user.
