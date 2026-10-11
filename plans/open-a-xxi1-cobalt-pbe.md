# The PBE cobalt helix does not converge on the laboratory sphere (`OPEN.md` Part XXI item 1, the open tail)

*Written 2026-10-11 from a reading of the code and of the item, before any run. Nothing
below has been measured here; every number quoted is the item's, and every prediction is
written so that the run that follows can only agree with one hypothesis. The runs go to
D22-0161 through `plans/open-a-scripts/run.sh` (six cores, a 7 GB scope), and the main
session runs them.*

## What is known, in one table

The cell is `tests/data/qe/co-helix4-spiral.in`: tetragonal cobalt, one atom, `a = 4.60`
bohr, `c/a = 1.30`, `Co.pbe-nd-rrkjus.UPF` (ultrasoft, PBE from the header),
`ecutwfc = 20`, `ecutrho = 160`, `q = -b3/4`, the moment started along `x`
(`angle1 = 90, angle2 = 0`), `3 3 4` unshifted, `nosym`, Marzari-Vanderbilt smearing
at 0.02 Ry, `conv_thr = 1e-10`. The "laboratory sphere" is the transverse pair of the
density and of the potential kept on `{G : |G - q| < G_rho}` and its smooth prefix
(`defumat/basis/gvectors.py:205-300`, `defumat/scf/driver.py:2475-2484`,
`lift_density` at `:5285-5300`, the cut of `W = V_x + i V_y` at `:5409-5414`, the
transverse `newd` gather at `:4732`); the "resident sphere" is the code before, which the
tests reproduce by handing back `basis.dense, basis.smooth`
(`tests/regression/test_spin_spirals_augmented.py:353-367`).

| run (`run_scf` defaults, 150 iterations) | laboratory sphere | resident sphere |
|---|---|---|
| PBE, 160 Ry | stops at 2.6e-10, energy wandering 5e-5 Ry, `becsum` magnetic residual 1.7e-3 | 22 iterations to 4.0e-12, `becsum` 2.4e-4 |
| PBE, 80 Ry | near 2e-7, energy wandering 2e-3 Ry | 36 iterations to 6.7e-12 |
| PBE, 320 Ry | not run | 8.8e-9 in 150 |
| LDA (`pz`), 80 and 160 Ry | 14 and 13 iterations | 14 and 13 iterations |

`mixing_beta = 0.3` and `mixing_mode = 'adaptive'` do no better on the laboratory
sphere. The identity that motivated the change holds there at round-off: one
diagonalisation of the unfolded supercell at the spiral's density reproduces the spiral's
eigenvalues to 2.98e-14 Ry and its output magnetization to 1.2e-9 relative, against
7.266e-5 and 4.20e-3 with the resident sets (`test_the_spiral_is_its_unfolded_supercell_on_the_laboratory_sphere`).

## Three readings to correct before any run

They cost nothing and each removes a hypothesis or a false alarm from the list.

**The energy wandering at an `accuracy` of 1e-10 is first order by construction, not a
sign of a non-variational map.** At an iteration that has not converged the reported
total is `eband - int V_in rho_out + E_H[rho_in] + E_xc[rho_in] + ...`
(`driver.py:1437-1454` for `deband`, the convention stated at `:8095-8110` and
`:8148-8150`), and no `descf` is added. Against the Harris-Foulkes functional
`E_HF[rho_in] = eband - int V_in rho_in + E_Hxc[rho_in]` it differs by
`-int V_in (rho_out - rho_in)`, which is linear in the residual. `accuracy` is QE's
`dr2`, the Hartree energy of the charge residual plus a flat-weighted magnetization term
(`defumat/scf/potential.py:160-215`), so it is quadratic: 2.6e-10 is a residual of about
2e-5 in the flat norm of the Fourier coefficients, and with a cell of 126.5 bohr^3 and an
exchange field of order 0.1 Ry the linear term is bounded by about 2.5e-4 Ry. 5e-5 Ry
sits inside that bound. So the sentence in the item, "a state whose energy is not
stationary to 1e-10", conflates a reporting convention with a property of the map; it is
measurement M1 below that says which it is.

**The `becsum` residual is the density residual in other units, not a blind spot.** For
an ultrasoft dataset the mixed `becsum` feeds nothing but this diagnostic
(`driver.py:601-760`: it is packed, stepped with the density's Anderson coefficients and
excluded from the fit; the potential is built from the mixed grid density, which already
holds `Q_ij becsum`), and `becsum_residual` is a Euclidean norm in electrons
(`defumat/scf/residual_split.py:165-221`). The item's two pairs are consistent with one
quadratic `dr2`: 1.7e-3 / 2.4e-4 = 7.1 against `sqrt(2.6e-10 / 4.0e-12) = 8.1`, and the
resident arm reports 2.4e-4 at a converged 4.0e-12. So "`becsum` not mixed" is not a
hypothesis for this cell.

**No member of the laboratory sphere sits at a wrapped index on this cell, so there is
no Nyquist or aliasing component to look for.** With `c = 5.98` bohr, `b3 = 1.0507`
1/bohr and `|q| = b3/4 = 0.2627` 1/bohr: at 160 Ry `G_rho = 12.649` 1/bohr is 12.04
reciprocal vectors along `c`, so the resident sphere spans Miller indices -12 to 12 on
the dense box of 25, and the displaced one `|m3 + 1/4| <= 12.04` spans -12 to 11; at 80
Ry (and the smooth sphere at 160 Ry, which has the same radius) 8.51 reciprocal vectors
on a box of 18, resident -8 to 8, displaced -8 to 8; along `a` and `b` the displacement
is zero. The item says the same ("no cell measured reaches it"). M0 confirms it in one
line and then this hypothesis is closed.

## The hypotheses

**H1, the leading one: the gradient correction's unsigned `|m|` on a magnetization with
nodes.** `gradcorr`'s noncollinear branch resolves the density as
`(n +- s |m|)/2`, runs the collinear GGA and rotates back
(`potential.py:488-546`). With a fixed axis `s = sign(m . u)` and "up stays up" across a
node; without one `s = 1`, and `|m|` has a kink on every surface where `m` passes
through zero. A spiral never gets an axis: `driver.py:2664` and `:3951` read
`fixed_quantization_axis(...) if self.nspin_mag == 4 and not self.spiral else None`. The
energy of a kink is small (the two resolutions have the same energy density pointwise
away from the node, so they differ only by the grid's Gibbs error at the node), but the
potential is not: `v = v1 - div h` with `h` jumping across the node is a band-limited
delta sheet, of amplitude set by the grid spacing, and its position moves with the
nodal surface. The SCF map then has a part no Anderson history can fit, which is a
plateau rather than a divergence. Cobalt has such nodes in the interstitial, where the
`sp` polarization is opposite to the `d` moment, and in the rotating frame they are
lattice-periodic surfaces of the one-atom cell.

What H1 accounts for: LDA converges on both arms, because the LDA field
`v_s(n, |m|) m / |m|` is smooth at a node (the energy is analytic in `|m|^2`); the
resident PBE arm is already slow (22 and 36 iterations against LDA's 13 and 14) and fails
at 320 Ry, where the sheet is sharper; and `pw.x` itself limit-cycles on the unsigned
branch, at 5e-6 Ry on four hydrogen moments 90 degrees apart (100 iterations at
`mixing_beta = 0.3`, 300 at 0.1) and at 5e-8 on two, where the same cells converge to
1e-11 under LDA (`tests/data/qe/fe-noncolin-pbe-stress.in:13-25`, `PLAN.md:15745-15750`).
What it does not account for on its own is why the laboratory sphere is worse than the
resident one at both 80 and 160 Ry; under H1 that is a shift of the nodal surfaces
against the grid by the 4e-3 the shell moves in the magnetization, which is a matter
of degree and is what M4 measures. The supercell's limit cycle is not evidence either
way: `co-helix4-spiral.in:12-20` records it in LDA as well, as an unstable mode the
one-atom cell forbids by symmetry.

**H2: the laboratory sphere makes the map non-variational, the potential's set differing
from the density's.** The chain is: the pair comes up through `pair_to_dense` on the
displaced smooth sphere (`defumat/basis/interpolate.py:89-110`), is augmented on the
displaced dense sphere (`_addusdens_spiral`), is mixed in real space on the whole box
(`DEFAULT_MIXING_SPACE = "r"`, `mixing.py:1645`; `'g'` is refused on a magnetic spiral),
goes into `v_of_rho` as four real fields on the box, and `W` comes back down through
`pair_to_smooth` and the transverse `newd` gather on the same displaced sets. Each gather
is the adjoint of the scatter that fed it, and the gradient correction takes its
gradient on the resident set of real fields `(n +- |m|)/2` (`potential.py:533`), whose
divergence is the adjoint operator on the same set (`:594-651`), so by reading the code
the map is the exact grid derivative of the grid functional. Two things keep H2 on the
list. The identity test cannot see a defect in the exchange-correlation code, because
the supercell side runs the same `_noncollinear_gradient_correction` with the same
`axis = None`; and the unsigned branch's derivative has never been measured against
anything (`fe-noncolin-pbe-stress.in` validates the signed one through the stress). H2
predicts a first-order remainder in M1 and no gain from M2 or M3.

**H3, `becsum` not mixed, and H4, a Nyquist component: closed by the arithmetic above.**
M0 is the one-line confirmation of H4.

**H5, a defect of the laboratory sphere's conditioning rather than of the functional.**
The transverse pair's real components now have support on the displaced sphere and on
its mirror, the flat Anderson fit runs over the whole box, and `accuracy` reads the
resident sphere (99.9 per cent of the residual, by the item's measurement). Nothing in
the code reading points at it, so it is the fallback: it is the hypothesis left if M2
converges on both arms and M3 does not help the laboratory arm.

## The measurements, by cost

All on the one-atom cell, `3 3 4`, 36 k-points, the dense box 20 x 20 x 25 at 160 Ry:
well under 1 GB, and from the test file's twenty-iteration helper about one second an
iteration on six cores, so a 150-iteration run is a few minutes and the whole list is
under 20 minutes. The script skeleton is `plans/open-a-scripts/xxi1.py` (`Calculator.
from_text`, `get_scf`, `fixed_density_states`, `calc.occupations`, `calc.density`); the
resident arm is the test's monkeypatch of `defumat.scf.driver._spiral_gvectors`
(`test_spin_spirals_augmented.py:353-367`); a `Calculation` built by hand and handed to
`run_scf(..., calculation=calc)` is where an axis is overridden, since
`quantization_axis` is a plain attribute read at each call (`driver.py:4533-4553`).

**M0, seconds, host only: the wrapped-index count.** `max(abs(spiral_dense.miller))`
per axis against `(n - 1) // 2` of the dense box, and the same for `spiral_smooth`, at
80 and 160 Ry. Prediction, all hypotheses: zero members past the half range on every
axis (the arithmetic above). A nonzero count is a finding and reopens H4; it is not
expected.

**M1, under a minute: is the potential the derivative of the energy, on this density?**
Take the plateau density: `get_scf(max_iterations = 60)` on the laboratory arm at 160 Ry,
PBE, and its `density`. Two numbers.

- `jax.grad` of `calc.potential(rho).ehart + calc.potential(rho).etxc` with respect
  to the dense `rho` (shape `(4, 20, 20, 25)`), against `v_scf * volume / N`. Prediction
  under H2: a mismatch of 1e-6 relative or more in channels 1 and 2 (the transverse pair),
  everywhere; under H1 and H5: round-off (below 1e-12 relative) except at grid points
  where `|m| < 1e-6 max |m|`, where the `where` masks at `potential.py:522-544` and
  `defumat/xc/functional.py:921-945` may differ between primal and tangent. Report the
  maximum mismatch and the `|m|` at the point it occurs.
- The second-order remainder `R = E_Hxc[rho_out] - E_Hxc[rho_in] - int V_in (rho_out -
  rho_in)`, with `rho_out` from one `fixed_density_states` at `rho_in`, against `dr2 =
  scf_accuracy_split(rho_out - rho_in, ...)[0]`. `R` is exactly `E_KS[rho_out] -
  E_HF[rho_in]`, the Kohn-Sham energy at the output against the Harris-Foulkes energy at
  the input (`eband` cancels between them), so it is the disagreement of the two energies
  "at the end", and it needs two `potential` calls and no eigenvalue. The assembly at
  `driver.py:8387-8404` is the sum of one-electron, Hartree, exchange-correlation, Ewald
  and smearing terms with no `descf`, which is what makes the reported total first order
  and `R` the number that is not. Three states: the PBE laboratory
  plateau, the PBE resident arm stopped at `max_iterations = 8` (a `dr2` near 1e-6, to
  have a comparable residual on a run that converges), and LDA stopped where its `dr2`
  is nearest the plateau's. Prediction under a smooth variational map (LDA, and H5):
  `R / dr2` of order 1 to 10 on every state. Under H1: `R / dr2` on the PBE states is
  ten to a thousand times LDA's, and the integrand of `R` (its pointwise density, read
  from the energy density of the two `potential` calls) sits on the points with small
  `|m|`. Under H2: `R` scales as `sqrt(dr2)`, which is read by repeating at
  `max_iterations = 40` and `60` on the laboratory arm (two plateau points with
  different `dr2`) and fitting the exponent: 1/2 is H2, 1 is a smooth map, and a kinked
  one gives 1 with a large prefactor.

**M2, minutes: the same cell at `q = 0`, with and without the sign axis.** Strip the
`spiral_q` line from `co-helix4-spiral.in` (the ordinary noncollinear run, as
`test_a_zero_spiral_is_an_ordinary_noncollinear_run` does at `:191`); one atom with its
moment along `x` gives `fixed_quantization_axis = (1, 0, 0)` by `potential.py:448-485`.
Arm A: `run_scf` with that axis. Arm B: `calc = Calculation(system, pseudos);
calc.quantization_axis = None; run_scf(system, pseudos, calculation=calc)`, the
functional every spiral runs. Both at 160 Ry, PBE, defaults, `conv_thr = 1e-10`, 150
iterations; then LDA (`input_dft = 'pz'`) for arm B only. No displaced sphere is in play
at `q = 0`, so this is the functional alone. Prediction under H1: arm A converges in 15
to 25 iterations and arm B plateaus above 1e-9 or takes more than 60 iterations, while
LDA arm B converges in about 14. Under H2 and H5: arms A and B converge alike (within a
few iterations of each other), because nothing in the spiral's sets is exercised. Under
H1 the two converged energies of arms A and B (if B does converge, late) differ by the
grid's Gibbs energy at the nodes, expected at 1e-6 Ry or below; record it, since it is
the price of the fix.

**M3, minutes: the spiral with a sign axis in the rotating frame.** The rotating-frame
starting moment is the input's, `(1, 0, 0)`. Build `Calculation(system, pseudos)` on the
spiral input, set `calc.quantization_axis = (1.0, 0.0, 0.0)`, run on the laboratory arm
and on the resident arm at 80 and 160 Ry, PBE, defaults, 150 iterations; four runs.
Prediction under H1: all four converge in 25 iterations or fewer to 1e-10, and the
160 Ry laboratory energy is within 1e-5 Ry of the resident arm's converged unsigned
energy of the item's 22-iteration run. Under H5: the resident arm converges and the
laboratory arm still plateaus. Under H2: no change on either arm. One caveat to read
with the result: a signed projection jumps by `2 |m'_y|` on the surface `m'_x = 0`
wherever `m'_y` is not zero there, and in the rotating frame `m'_y` is odd under
inversion through the atom and not zero away from it. If M3 converges but the signed
energy differs from the unsigned one by more than 1e-5 Ry, that jump is being paid for
and the fix is F2 rather than F1.

**M4, one one-shot at the plateau, seconds: where the residual lives.** From the M1
state, `|Delta m'(r)|^2` summed over the points with `|m'(r)| < 0.05 max |m'|`, as a
share of the total, against the share of grid points there; the same for
`|W(r)|^2`, the transverse potential. Prediction under H1: the residual's share is at
least five times the point share, and the ratio grows from 80 to 160 Ry. Under H2 and
H5: the two shares are proportional. This is the number that turns H1 from a reading
into a finding, and it is independent of M2 and M3.

**M5, minutes: plain linear mixing.** `mixing_mode = 'linear'` (`mixing.py:833-851`),
`mixing_beta = 0.1`, the laboratory arm at 160 Ry, PBE, 150 iterations. A smooth
contractive map converges geometrically, slowly and monotonically; an unstable
stationary point departs (the moment or the energy drifts in one direction); a map with
a delta sheet plateaus at a `dr2` set by the sheet, as under Anderson. Under H1 and H2
the prediction is a plateau; under H5 a slow monotone convergence, since H5 is a
property of the fit and not of the map. It separates H5 from the other two, which is
why it is here and not earlier.

**M6, only if H5 is left standing: one set at a time.** The laboratory sphere enters at
five places (the item's list). Hand back the resident sets to one consumer at a time
(the density lift and the `W` cut as a pair, the augmentation table and its `newd`
gather as a pair, `at_positions` alone) and run the 160 Ry PBE SCF for each; the pair
whose resident version restores the 22 iterations is the one to read. The identity test
will fail on every such arm, which is expected and is not the question.

## What each outcome leads to

**H1 confirmed (M2 arm B stalls and arm A converges; M3 converges; M4 concentrated).**

- F1: a spiral takes the sign axis from its starting moments in the rotating frame,
  which is QE's `lsign` rule applied in the frame where the texture is lattice-periodic:
  drop `and not self.spiral` at `driver.py:2664` and `:3951`. The identity test must
  then hold `quantization_axis = None` on both sides explicitly, because the supercell's
  moments turn in the laboratory frame and it has no axis; that needs an override
  argument rather than the attribute set by hand. P122's helix source and its curvatures
  are re-read with the axis, and the signed-against-unsigned energy difference from M2
  goes into `PLAN.md` as the measured price.
- F2, for a spiral cell whose starting moments are not parallel, or where M3's caveat
  bites: the limitation is `pw.x`'s own and is documented as such with the hydrogen
  numbers. A smoothed modulus `sqrt(|m|^2 + eps^2)` in the gradient correction is a
  different functional and is not the fix; if it is ever wanted as a named option it is
  refused by default and its energy shift is measured. The gradient-of-`m` form of the
  noncollinear GGA (the functional built from `grad m . grad m` rather than
  `(grad |m|)^2`, which is smooth at a node and rotationally invariant) is the proper
  cure, is a new functional, and gets its arXiv check before any code.
- F3: the item's sentence about the energy not being stationary is replaced by the
  first-order reading above, with M1's numbers, and the `becsum` paragraph by the
  ratio 7.1 against 8.1.

**H2 confirmed (M1's `jax.grad` mismatch is in the transverse channels, or `R` scales as
`sqrt(dr2)`).** The defect is in code the supercell shares, so it is fixed where M1
locates it (the back-rotation at `potential.py:537-544`, or the `where` masks), the
identity test is rerun as a regression and not as the proof, and M1 becomes a committed
test, since nothing else sees this class of defect.

**H5 left (M2 alike on both arms, M3 does not help the laboratory arm, M5 converges
monotonically).** M6 names the pair; the fix is in that consumer, and the item's
convergence table is re-measured on both arms.

**M2 arm B converges and M3 does not, with M4 proportional.** Then neither the
functional nor the sets is the cause on its own, and what is left is their combination:
the shell's 4e-3 of the magnetization moving a nodal surface onto the grid. The test is
M4 repeated on the resident arm at 320 Ry, where that arm fails too; if its residual is
concentrated there as the laboratory arm's is at 160, H1 stands with the laboratory
sphere as the trigger rather than the cause, and F1 is still the fix.

## The first step

M0 and M1 in one script, laboratory arm, 160 Ry, PBE, `max_iterations = 60` and then
`40`: the wrapped-index count, the `jax.grad` mismatch with its location in `|m|`, the
remainder `R` and its `dr2` at both plateau points, and M4's two shares from the same
one-shot. Under 5 minutes on six cores. It settles H2 and H4 before any functional is
changed, and its M4 share is the first number for H1. M2 follows in the same session,
since it needs no spiral code at all.
