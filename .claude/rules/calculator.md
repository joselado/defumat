---
paths:
  - "defumat/calculator.py"
  - "defumat/workflows/**"
  - "defumat/response/**"
  - "defumat/system/**"
  - "tests/unit/test_calculator*"
---

*Moved out of `CLAUDE.md` verbatim on 2026-10-08; it loads only when files matching the paths above are read. The text is binding exactly as it was.*

## The front door is `Calculator`

`defumat/calculator.py` (P38). A `Calculator` is a `System` together with its
pseudopotentials, and every workflow, force, stress, response and invariant is a method
on it — `Calculator.from_file("scf.in")`, then `get_scf()`, `get_bands()`,
`get_dielectric_tensor()`. It is what the README, the user guide and new notebooks use,
and `from defumat import Calculator` is the one import a script needs.

**It is a facade and nothing else.** No physics lives there: every method is a one-line
delegation to the functional entry point, which is unchanged and still the way anything
managing its own state is driven. A `get_*` that grew a computation of its own would be
a second implementation of something already validated against QE, and there is a test
asserting that none has.

Three things about it bind anything added to it:

- **State cannot move onto `System`.** `System` is an `eqx.Module` crossing `jit`/`grad`,
  so a `pseudos` field would change the pytree every compiled path sees and a cached
  result cannot live on a frozen module at all — and `System` does not *have* the
  pseudopotentials, only the file names. `System.calculator()` is a constructor, not a
  place to hang calculations.
- **Nothing mutates.** `with_positions`/`with_cell`/`with_spin` return a *new* calculator
  with an empty cache; the converged state crosses as a `starting_from` seed
  (`starting_state`), never as an answer. A cached result under a moved atom is the
  `test_geometry_invalidation` defect one layer up.
- **The refusals pass through untouched**, and the implicit SCF announces itself. The
  cache is one slot keyed by the options that filled it, because it holds the
  wavefunctions.

A new feature adds a `get_*` method in the same pass that adds its entry point. Shared
options go in `SHARED_OPTIONS` and are forwarded **by named parameter only** — a
`**kwargs` in a signature is not permission to pass everything, since several response
entry points forward theirs to solvers that would raise on `nbnd`.
