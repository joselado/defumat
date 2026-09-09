---
name: trap-check
description: Audit a change against defumat's recurring traps - the ones that produce a plausible wrong answer rather than an error. Covers abs at a forced zero, rule D4 band diagonals, a caller-built k-set that misses for_spin, an unsymmetrised wedge sum, np.asarray inside a differentiated path, hardcoded dtypes, a hand-rolled vmap over k, and a stencil containing its own centre. Use before committing anything that touches the SCF, a response, a force, a stress, an invariant, or any differentiated path; also when a number looks right but is not trusted, and when reviewing a subagent's physics code. Triggers - "check for traps", "review this diff", "is this safe to commit", "why is this number slightly wrong".
---

# Trap check

Every item here has been hit in more than one phase of this project, and every one
**produces a plausible wrong answer rather than an error**: the result is real,
finite, correctly symmetric, often passes a sum rule, and is wrong. No test in the
suite catches them automatically, which is why this is a checklist rather than a
pytest file.

`PLAN.md` holds the phase that found each. `CLAUDE.md`'s "The traps that recur"
is the short form; this is how to actually look.

## How to run it

1. Get the diff: `git diff` for uncommitted work, `git diff master...HEAD` for a branch,
   or take the file list the user names.
2. Walk the checklist below **in order**. Each item says how to find candidates, what
   separates a real hit from a false positive, and what it costs when missed.
3. Report only real hits, most severe first, with `file:line`. Say plainly when nothing
   was found - a clean report is the useful outcome here, and inventing findings to
   look thorough is worse than silence.

Do not fix anything unless asked. This is a review.

---

## 1. `abs` is not differentiable at zero, and the zero is often forced

```bash
git diff -U0 | grep -nE '^\+.*(jnp\.abs|np\.abs|\babs\(|\*\*2|\|\|)'
```

**Real hit** when the quantity can reach exactly zero and the code is inside anything
`jax.grad`, `jvp` or `linearize` will traverse. Four known shapes:

- `|psi|^2` written as `abs(psi)**2`. Must be `jnp.real(jnp.conj(psi) * psi)`.
- `abs(rho)**2` in a reciprocal-space Ewald sum. A structure factor vanishes *exactly*
  wherever symmetry arranges it, which a supercell does routinely, giving `0/0`.
- `|m|` in a gradient correction, which differentiates through its own nodes.
- A spin-channel density reaching zero, which any cell with vacuum guarantees. This is
  the same trap one derivative further out; clipping inside the response does **not**
  fix it. The fix that works is masking the *argument the derivative is taken at* -
  clipping the density leaves the primal singular and the tangent `0 * inf`.

**False positive** when the value is provably bounded away from zero, or the site is
never differentiated. Say which, do not just wave it away.

## 2. Rule D4 - a diagonal is not invariant under a degenerate rotation

```bash
git diff -U0 | grep -nE '^\+.*(einsum.*(knn|nn->|ii->)|\.diagonal\(|np\.diag|diag_part)'
```

Anything built from `<psi_n|A|psi_n>` **band by band** - a Drude weight, a band-velocity
difference, an incoherent channel sum, a transport diagonal - is not a physical quantity
as written. A degenerate eigensolver is free to return any unitary mixture inside a
multiplet, and a diagonal is not invariant under it. A quadratic form is.

**The fix is the multiplet block average**, or diagonalising the operator within each
block first. `defumat/response/spectra.py:degenerate_manifolds` is the shared helper;
`defumat/transport/green.py:channel_basis` is the in-place version.

**Cost when missed:** four orders of magnitude on silicon's second-order susceptibility;
69x against 26x on a bilayer's transport contrast. **No symmetry check sees it**, and
the wrong answer is smooth and plausible.

## 3. A caller-built k-set is a `for_spin` boundary

```bash
git diff -U0 | grep -nE '^\+.*(KPoints\(|kpoint_grid|from_list|monkhorst|denser_grid|whole_grid)'
```

Every `KPoints` constructor applies the unpolarized `degspin` unconditionally. A spinor
band holds **one** electron, not two. Any entry point that builds its own k-set must
pass it through `defumat.system.kpoints.for_spin(kpoints, nspin)`.

**Cost when missed:** a plasma frequency of 13.11 eV instead of 0.60; a factor of four
in a nesting function. The wrong numbers are the plausible ones.

## 4. A response on a reduced k-set is a vector field

Any new quantity summed over a symmetry-reduced wedge must be symmetrised as the
**polar or axial** object it is - and the magnetization is axial, so its rotation carries
`det(R)` and a further sign for an operation that is only a symmetry with time reversal.

Two things that are **not** escapes:

- Running the whole grid instead of the wedge. A **shifted** Monkhorst-Pack grid is not
  closed under the point group, so this is unsound and is refused by name.
- Assuming the wedge sum completes. It completes only for a quantity **linear** in a
  covariant per-k object. Where a functional is quadratic in one, the *value* inside it
  must be the full-zone object while its *derivative* stays the raw wedge sum. Getting
  this backwards is worth 2.5%, is worse than doing nothing, and only a sum rule catches it.

Also check the group: a quantity unfolded with `grid_equivalence` must be unfolded with
the group that `denser_grid` reduced it with (`defumat/workflows/nscf.py:grid_symmetry`).

## 5. `np.asarray` inside a differentiated path

```bash
git diff -U0 | grep -nE '^\+.*(np\.asarray|np\.array|numpy\.|float\(|\.item\(\))'
```

Kills the gradient **silently** - a term that should be there simply vanishes, and the
remaining terms are still a sensible-looking number. Real hit whenever the argument can
be a tracer. Host-side setup (G-vector enumeration, symmetry search, radial tables) is
fine and is where NumPy belongs.

## 6. Never hardcode a dtype

```bash
git diff -U0 | grep -nE '^\+.*(complex128|complex64|float64|float32|1\.0j|1j|0j)'
```

Single precision must stay viable for GPU. Real and complex dtypes come from the
precision policy - `defumat/config.py`'s `Precision`, reached as `DEFAULT_PRECISION` or
the object already threaded through the call site (`prec.as_complex`, `prec.zeros`,
`prec.real`). A literal `1.0j` is a hit; so is `dtype=jnp.complex128`.

## 7. A hand-rolled walk over the k axis

```bash
git diff -U0 | grep -nE '^\+.*(jax\.vmap|jnp\.vmap|for ik in|for k in range)'
```

Anything new that walks the k axis goes through `map_k`/`sum_k` in
`defumat/batching.py`, not its own `vmap`. That is what keeps the batching dial, the
chunked `lax.map`/`lax.scan` form, and the eventual sharding available. The chunk size
must never be visible in a result beyond round-off.

Also: `k` must stay the **leading independent axis** of every wavefunction-shaped array.

## 8. A stencil must not contain its own centre

Any finite difference over **k** rebuilds the plane-wave sphere at every point, and a
high-symmetry point is exactly where a shell sits on the cutoff - `Gamma` holds fewer
plane waves than every displaced point, so its eigenvalue is variationally *high* against
theirs and the second difference inherits an error that **grows as the stencil shrinks**.

## 9. Index order in a transposed pair reads as a sign

`f(n, m)` against `e(m, n)`; a Green's function conjugating `psi` in the *exit* variable
rather than the source one. Both give results that are real, non-negative, correctly
symmetric and wrong. **Only a literal check against the written definition catches this** -
re-derive the contraction from the equation in the docstring, index by index, and say so
in the report. If there is no equation in the docstring, that is itself the finding.

## 10. Neighbouring k-points do not share a G-sphere

Coefficients are aligned by **Miller index**, and the wrap at the zone edge is a *shift*
of that index. Without it a Chern number comes out smooth and non-integer.

## 11. The energy can be right while its derivative is wrong

Being stationary hides an error in the gradient. A total right to 3e-12 Ry sat beside a
force wrong by 0.4 Ry/bohr on a force of 0.06.

**So: an energy agreement is not evidence for a force.** And the identities that are sums
over *atoms* - the acoustic sum rule, the rigid-translation test - are **blind to a
transfer between atoms**. If a new derivative is validated only by an atom-sum identity,
that is a finding: the first real check is a per-mode response density against a finite
difference.

## 12. Inherit a refusal only after checking which machine it belongs to

A refusal written for the Sternheimer stack does not apply to a sum over states. Taking
one as read left a whole quantity marked impossible for a phase. If the diff copies a
`NotImplementedError` message from elsewhere, check the reason still holds here.

---

## Report format

Group by severity. For each: `file:line`, which trap, the concrete failure (inputs or
state that make it bite), and the fix. End with the items you checked and found clean,
in one line - that is what makes the report trustworthy next time.
