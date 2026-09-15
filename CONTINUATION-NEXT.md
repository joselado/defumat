# What is left to do about continuing one run from another

`PLAN.md` P23 is the phase, `defumat/scf/continuation.py` is the code, and this file is
the part of it that is not done. The audit behind it was a sweep of the whole
`starting_from` path against the question a user actually asks, which is whether a
converged run can be carried into *any* other combination of features: magnetic or not,
collinear or spinor, with spin-orbit coupling or without, with a Hubbard `U` or without.

The state machinery itself is in good shape on the spin axis. What was closed in the same
pass as this file is `PLAN.md` P23b: the spin spiral's rotated frame, `magnetization=`
reaching the front door, the electron count that was never checked, a deformed cell refused
by name so the count cannot blame a dataset for it, and a held magnetization crossing into a
run that holds nothing. What is below is everything the sweep found and that pass did not
take.

Sizes are the author's estimate of the work, not a measurement.

## The notebook P23b does not have

`notebooks/43_magnetic_textures.md` already uses `with_moments`, and the result that belongs
in it is the one that made `with_moments` change its default: on the four-site hydrogen
chain, the *same* converged charge lands on two different magnetic states depending on
whether the moment is carried or seeded, 8.90 mRy apart, and **both runs report
convergence**. That is physics rather than implementation, which is the test a notebook cell
has to pass here, and it is the concrete form of the sentence the mixer section of the user
guide already carries: fewer iterations is satisfied by a run that found something else.

One cell, one comparison table, no new SCF beyond the two runs. The numbers are in
`PLAN.md` P23b.

## A seeded calculator never reads its own checkpoint

`run_scf` loads a checkpoint only when `starting_from is None`, and `Calculator.get_scf`
always sets `starting_from` to the seed on a derived calculator. So
`calc.with_spin(4).get_scf(checkpoint_dir=X)`, killed at its wall clock and resubmitted,
restarts from the seed every time and never reads the checkpoint it has been writing.

It is pre-existing and it is a stated design choice read one layer too narrowly: "an
explicit `starting_from` wins" is right about an argument the *caller* passed and wrong
about one the calculator inserted on their behalf, which is not something anybody chose per
run. The checkpoint is strictly later state than the seed in both cases, which is the
argument the driver's own comment already makes for preferring it over a caller's seed.

The fix is one condition and the question is where it belongs: `get_scf` could decline to
insert the seed when a `checkpoint_dir` holds a checkpoint, which keeps `run_scf`'s rule
exactly as written and is the smaller change.

## The missing `with_*` constructors

`Calculator` carries `with_positions`, `with_cell`, `with_kpoints`, `with_spin` and
`with_moments`, and every feature flag that is not one of those lives on `System` as a
static field with no constructor to reach it. None of this is blocked, since the functional
entry point takes whatever system and pseudopotentials it is handed and the state promotion
does not care how the target was built. What is missing is the front door, and the hazard
is that the obvious substitute is `dataclasses.replace`, which is exactly the thing
`System.with_moments` exists to stop people doing: it leaves the k-point weights carrying
`degspin` and leaves the set reduced with the wrong group, both silently.

- **`with_pseudos`.** The largest of the gap and the smallest to write. Switching
  spin-orbit coupling on means a *different dataset*, because QE refuses to `j`-average an
  ultrasoft or PAW pseudopotential (`PW/src/average_pp.f90`) and this code refuses the same,
  and `Calculator._derived` forwards `self.pseudos` unconditionally. So the one route P23
  measured, scalar PAW to fully-relativistic PAW on platinum at 7 iterations against 13, has
  no `Calculator` form. It is reachable in two lines today and the two lines are worth
  writing down, since a reader of the feature table would not guess them:

  ```python
  fr = (read_upf("Pt.rel-pbe-n-kjpaw_psl.0.1.UPF"),)
  spinor = calc.system.with_spin(4, lspinorb=True).calculator(pseudos=fr)
  result = spinor.get_scf(starting_from=calc.get_scf())
  ```

  What a real `with_pseudos` adds beyond that is the seed being carried automatically and
  one place to say what an electron-count change means, which the continuation now refuses
  by name.
- **`with_hubbard`.** Turning `U` on from a converged non-`U` run and off again. The state
  promotion already handles both directions correctly, so this is the constructor and the
  `HUBBARD` card built from Python arguments, in eV at the boundary as the card is.
- **`with_functional`.** `input_dft`, for the converge-in-LDA-then-refine-in-PBE workflow.
  Nothing in the continuation needs to change; a converged LDA density is a good PBE guess
  and that is the point.
- **`with_spiral`.** `spiral_q` as a constructor, with the `nosym` and the refusals
  `Calculation` already makes. `at_spiral_q` exists on `Calculation` for the relaxation's
  gradient and is not the same thing: it is an internal geometry step, not a calculator.
- **`with_field`** and **`with_constraint`.** The hold-then-release workflow, which is now
  warned about when it happens and still has no way to ask for it.

## `ns` is left at the atomic guess when `U` is switched on

`promote_ns` returns `None` when the source carried no `ns`, so a run that switches `U` on
starts from `init_ns` while its density is converged. The two guesses then disagree about
the shell on the first iteration, which is the same objection `_becsum_split` states for the
atomic start and which this module's own `_SpinTransfer` exists to prevent between the
density and `becsum`. The consistent seed is `new_ns` evaluated on the wavefunctions the
continuation is already carrying, which are the converged states of the non-`U` run.

An enhancement rather than a defect: the run converges either way, and what is at stake is
a few iterations on the quantity a Hubbard run is slowest in.

## `tau` is dropped by every change of spin regime

`run_scf` carries a source `tau` only when its shape matches the target's density exactly,
so any promotion between regimes falls back to the Thomas-Fermi guess. That is deliberate,
and it is the right trade: there is no counterpart to `spin_components` for a kinetic energy
density and a reshaped guess that is wrong is worse than a crude one that cannot be. It
belongs in the user guide rather than in the code, because what a user sees is a meta-GGA
continuation that saves fewer iterations than they expected.

## The fresh-run sibling of the spiral guard

The continuation now refuses to carry a magnetization onto a spiral's own rotation axis,
because a moment along `z` is invariant under the spiral's spin rotation and is therefore a
stationary point at every `q`: the run converges, reports a moment, and has computed the
ferromagnet.

**A run started from scratch reaches the same state by the same route and nothing says so.**
`starting_magnetization` with every `angle1` at zero builds exactly that seed, and the
spiral SCF has no more ability to break the symmetry on its own than the collinear one has
to become magnetic. The guard belongs at the door, in `Calculation`, beside the refusals a
spiral already makes for symmetry and for ultrasoft datasets: a spiral whose starting
moments are all on the rotation axis should be refused, naming `angle1` as the way through.

It is worth checking what the existing spiral tests and `workflows/spiral.py` do about this
before adding the refusal, since a `q` sweep that quietly includes a collapsed arm would
start failing rather than start being wrong, which is the intended outcome but is a change
to a committed number.

## Changing the cutoff or the grid is refused, and interpolating is the workflow

`_check_grid` demands that the source density's grid match the target's, so a continuation
cannot cross a change of `ecutwfc` or `ecutrho`. The refusal is honest and the message is
clear. What it rules out is the workflow people actually want on a large cell, which is to
converge cheaply and then refine: a density is a set of Fourier coefficients and moving it
between two grids is padding or truncating in `G` space, not interpolation in the numerical
sense, so it is exact in one direction and a projection in the other.

Not started, and it is a phase rather than an item: the wavefunctions cannot cross with the
density, since a change of `ecutwfc` is a different plane-wave sphere, so what is carried is
the charge and the run pays a fresh Rayleigh-Ritz. That is still most of the saving.

## The combinations with no number

This is the part that matters most against the repository's own rule, which is that a claim
is a number. `tests/regression/test_continuation.py` has six rows: five on the `nspin` axis
and one dataset swap for spin-orbit coupling. The check that means something is the
identity, that a continued run reaches the *same* self-consistent solution as a fresh one,
and it has never been run for any of these:

- `U` switched on from a converged non-`U` run, and switched off again.
- `U` crossing into `nspin = 4` together with `lspinorb`. P79 opened that promotion and
  measured the `ns` shapes; the round trip as an identity is not in the table.
- A spiral continued into a spiral at a neighbouring `q`, which is the one spiral case the
  new guards allow and the one a `q` sweep depends on.
- A collinear source promoted into a planar spiral, which is the other allowed case.
- A constrained or field-converged run released into a free one.
- Gamma-only storage continued into a full k-set. The dense grid is unchanged so the density
  crosses and the wavefunctions are dropped on the shape check; it should work and nobody
  has run it.
- A functional change, LDA to PBE, at fixed everything else.

The unit file covers the shapes and every refusal, including the new ones, and a shape test
cannot tell you where a run lands. Each row above is one pair of SCF runs on a small cell,
so the whole table is an afternoon of machine time and belongs in the slow set beside the
six that exist.
