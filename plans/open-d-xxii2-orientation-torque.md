# Plan: the orientation torque without the reverse tape (`OPEN.md` Part XXII item 2)

Written 2026-10-09 by stream D of the OPEN.md split, branch `open-d`. Not implemented: it goes
to a fable review first, and which route to take is the user's decision (the entry says so).

## The entry, reproduced at a size D22 holds

`forces/torque.py:orientation_torque` differentiates `sum_n w_n <psi_n|H(R)|psi_n>` at frozen
states in reverse mode, through the Hamiltonian built from the turned density and the turned
`becsum`. On the fifteen-cell NiBr2 helix (45 atoms, 432 spinor bands) it asked an H200 for
33.37 GiB more at one k-point. The same call on D22's A2000, on
`plans/open-d-inputs/nii2-3cell-paw.in` at one k-point (nine atoms, fully-relativistic PAW,
PBE datasets, 116 spinor bands, `npwx = 27921`, smooth grid 90x30x192, a pool of
6.96 GiB at fraction 0.6), **runs out of the card**: in-use bytes went from 0.36 to 6.95 GiB
and it stopped with `RESOURCE_EXHAUSTED` while compiling the backward pass of
`_newd_noncollinear_integrals`. So the tape is more than 6.6 GiB here, and the entry is
reproduced on nine atoms.

## The three routes the NiBr2 session proposed, measured on the same states

`tools/gpu/torque_tape.py` (committed on this branch), one process a route, the same random
orthonormal states and occupations, `soc_scale = 1`. The torque is a derivative at frozen
states, so every route must give the same three numbers on any states, eigenstates or not,
and they do:

| route | what it differentiates | card peak rise (GiB) | compiled temp (GB) | time (s) | torque, Ry/rad |
|---|---|---|---|---|---|
| production, `jax.grad` (as on master) | the band energy, reverse mode | **> 6.6, out of the card** | | 95 to failure | |
| 1, the closed form | nothing in the band energy: `integral m_out x B_xc[R rho]` + the one-centre pairing | 1.58 | | **25** | (8.091603890453703e-06, -1.0546118492919592e-03, -1.4680276726897628e-03) |
| 2, forward mode | three `jax.jvp` of the band energy | **1.26** | 1.61 | 126 | agrees with route 1 to 1.9e-15 of its norm |
| 3, rematerialisation | `jax.grad` over blocks of 16 bands, each block `jax.checkpoint`-ed | 1.47 | 1.83 | 141 | agrees with route 1 to 2.2e-15 of its norm |

What the numbers say. Every route fits where the production call does not, by a factor of at
least four on this cell. Route 1 is five times faster than either derivative of the band
energy, because it costs one density and one one-centre derivative, an SCF iteration's work,
and its peak is that density's band loop (which the memory mode's band dial already sizes);
routes 2 and 3 each apply the Hamiltonian to every band three times or more. Route 1's
one-centre part is now itself in forward mode (`_onecenter_torque`, Part XXII item 1 on this
branch), so no route left keeps a reverse tape of the one-centre calculation either.

## Why route 1 is exact, and where it is not

The Hamiltonian of a one-shot depends on the rotation only through the turned input density
(its exchange-correlation field) and, on PAW, the turned input `becsum` (its one-centre
coefficients), so `-dF/dw = sum w <psi|dH/dw|psi>` is the grid integral of the output
magnetization against the turned input's exchange field, plus the derivative of
`ddd_paw[becsum_in(R)] . becsum_out`. That identity is already held against the reverse-mode
torque by `test_orientation_paw.py::test_the_paw_torque_is_the_grid_s_plus_the_one_centre_field_s`
(1.2e-11 on PAW nickel in P122; 1e-9 asserted, and it passes through the forward-mode
one-centre on this branch), and the table above repeats it on nine atoms of NiI2 at 2e-15.

The condition is how `becsum_in` turns. Route 1 turns it as a vector, which is exact when the
source was converged without the coupling, because without it the three magnetization
components of `becsum` are a cartesian vector (2.8e-16, P122). `relax_orientation` and
`run_orientation_torque` (Route A) take their source at `soc_scale = 0`, so route 1 is exact
there. It is **not** exact for a source converged with the coupling, where the
fully-relativistic `becsum` keeps only the blocks diagonal in `j` and a spin rotation mixes
them (4.6e-2); Route C's in-loop torque handles that case by turning through the states, and
`orientation_torque` would have to refuse a coupled `becsum` on route 1 or take it through
the states the same way (a k-chunk at a time in forward mode, as `_onecenter_torque` now does).

## The change, for whichever route is chosen

- **Route 1**: `orientation_torque` builds the output density and `becsum` of the given states
  (streamed when the store is), the potential of the turned texture, and returns
  `exchange_torque(...).total + _onecenter_torque(calculation, turned_becsum, becsum_out)`;
  `_chunked_value_and_grad` stays for `torque_at_angle`, which is a different quantity only
  in its parametrisation and can move the same way afterwards. A source whose `becsum` was
  converged with the coupling is refused by name unless the states route is added.
- **Route 2**: `_chunked_value_and_grad` becomes `_chunked_value_and_jvp`, three `jvp` per
  k-chunk, the k-chunks unchanged; nothing else moves.
- **Route 3**: the band loop inside `_band_energy` and the chunked form becomes a `lax.scan`
  over band blocks with a checkpointed body; the block size is the band dial's.

Tests, any route: the three torques against the production one on a cell where the
production one fits (the PAW nickel of P122 and the norm-conserving and ultrasoft cells of
`test_anisotropy.py`, to 1e-10 relative), and `torque_tape.py`'s peak on the NiI2 cell above
held below a stated bound on the card. The measurement to take after it lands is the
fifteen-cell helix at `1 1 1` and `1 3 1` on an H200, the run that failed.

## What it does not settle

The production route has no number of its own here: it did not fit, so 6.6 GiB is a lower
bound on its tape, and a size would need the compile-only `memory_analysis()` of the
production program, which the eager path through `defumat.eager` does not expose. The
numbers are one cell on one card. The ratio of the tape to the band loop grows with the
bands and the grid together, so on the fifteen-cell helix the gap between the production call
and the routes is larger than here, not smaller, but how much larger is not measured.
