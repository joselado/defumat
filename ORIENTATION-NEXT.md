# Relaxing the orientation of a magnetic texture under spin-orbit coupling

A plan recorded on 2026-09-26 for a later session, not started. We will now see what the
coordinate is, why a spin rotation is not an atomic position and what that changes for a
calculation that already carries the coupling, the two routes to build (one from a state
without the coupling, one inside a self-consistent run with it) and a third that was
considered and dropped, what in this code each builds on, the traps that are known before
any code is written, and the order in which to build and measure it. Statements that are an
expectation rather than a measurement are said to be so.

## The idea

A structural relaxation moves the atoms until the forces vanish. We want the same thing
with a different coordinate: a rigid rotation `R` of every spin in the cell, applied to a
magnetic texture `m0(r)` as `m_R(r) = R m0(r)`, with the charge density untouched. Without
spin-orbit coupling the energy does not depend on `R` at all, since spin space is decoupled
from real space; the coupling ties the spin to the lattice, and `E(R)` varies by the
magnetocrystalline anisotropy, from microrydberg in a 3d metal to millirydberg with a heavy
ligand. What a relaxation in `R` returns is:

- for a ferromagnet or a collinear antiferromagnet, the easy axis, with two coordinates
  live, since a rotation about the moment itself does nothing;
- for a coplanar texture, a commensurate spiral in a supercell for instance, the plane it
  lies in (two coordinates) and its phase within that plane (the third), which for a
  multiferroic decides the mechanism: the spin-current polarization is nonzero for a
  cycloid and zero for a proper screw;
- for a noncoplanar texture, all three.

The gradient with respect to `R` is the net torque on the texture, `dE/dw` with `w` the
rotation vector, in Ry per radian. It is the magnetic counterpart of the force.

## Why a spin rotation is not an atomic position

An atomic position is external to the electronic problem. The SCF cannot move an atom, so
an SCF at fixed positions is well defined and the Hellmann-Feynman force is its gradient.
The orientation of the magnetization is an electronic degree of freedom: a run with
spin-orbit coupling is free to turn the moments itself, and at its converged point the
energy is stationary under every variation of the density, a rigid rotation included, so
the torque of a converged unconstrained run is zero by construction. What keeps this from
being the end of the story is that the rotation is a soft mode, the anisotropy being a
millionth of the energy scale that sets `dr2`, so how far the moments have turned when a
run reaches `conv_thr` is partial and depends on the path. P87 measured a cardinal
direction of tetragonal cobalt drifting 0.000 degrees, which is symmetry, and
`RELAXED_DRIFT_TOL` exists because an oblique direction generally drifts (its comment puts
a genuine collapse at about 30 degrees), with no guarantee of having arrived anywhere. A
converged run with the coupling is therefore stationary in orientation only as far as its
threshold resolves a microrydberg mode, which is the "tolerance its own `conv_thr` does not
deliver" trap of `CLAUDE.md` in a new place.

There are two ways round it for a run with the coupling. One is to hold `R` with a
constraint during every SCF, which makes `R` a coordinate exactly like an atomic position,
with the holding field as its gradient; that is Route B, considered and dropped. The other
is to leave the SCF unconstrained and give its soft mode the step it cannot take on its
own, which is Route C. Where the source state has no coupling nothing needs holding,
because the density is frozen and turned by hand, which is Route A.

## Route A: from a state without the coupling (the force theorem, as a gradient)

Converge the texture without spin-orbit coupling, turn its magnetization by `R`,
diagonalise once with the coupling, and take the band free energy
`F(R) = sum w eps - TS`. This is P58's force theorem with a general rotation in place of a
direction, and P60's torque with three generators in place of one plane. At frozen density
every other term of the total energy is invariant under `R` (the Hartree term sees only the
charge, the exchange-correlation energy only `|m|` and a quantization axis that turns with
the texture, the Ewald sum neither), so differences of `F(R)` are total-energy differences
to first order in the density, and the gradient at frozen states is exact for `F` by the
envelope argument P60 checked to six digits. The gradient is `jax.grad` of `F` at frozen
states, the way this code computes forces, and a BFGS loop on `R` around it is the
relaxation: one diagonalisation and one gradient per step.

What the gradient is physically is worth writing down, since it gives a check that shares
no automatic differentiation with the assembly, and since Route C is built on it. Only the
exchange field `B` carries `R`, and `B[R rho0](r) = R B[rho0](r)`, so at the current
orientation `dB/dw_a = e_a x B` and

```
dF/dw = integral of B(r) x m_out(r) dr        (sign to be fixed by a central difference)
```

meaning that the torque is the frozen exchange field acting on the magnetization of the
coupled states. In the local spin-density approximation `B` is parallel to the frozen `m0`
at every point, so the integrand is the lean of the coupled states' magnetization off the
field they were solved in, which is exactly what the coupling produces.
`scf/spin_torque.py:exchange_torque` already integrates `m x B` for a density and a
potential handed to it separately, so the check is one call. It holds as written for a
norm-conserving dataset and for an ultrasoft one with the augmentation part in `m_out`; a
PAW dataset adds a one-centre torque it does not contain.

Route A turns the texture rigidly: the angles between the moments stay at the source's
values, and whatever canting the coupling would add on each site (single-ion anisotropy, a
Dzyaloshinskii-Moriya term where inversion is broken) is not in it.

The cost is one fixed-density diagonalisation and one backward pass per step, which is
P60's bill: 11.2 s for the torque on one-atom tetragonal cobalt, and a 6.42 GiB peak on the
cobalt slab, chunked over k.

## Route C: turning the moments inside the SCF loop

The route for a calculation that already has the coupling, decided on 2026-09-26 in place
of Route B. It is built after Route A, whose rotation and torque it reuses, and it runs
without Route A having been run. One SCF with the coupling in which every
iteration also takes a rotation step on the input density, so that the run converges
directly to the orientation of lowest energy.

**Why it is cheap.** At iteration `n` the SCF has diagonalised `H[rho_in]` with the
coupling and built `rho_out` from the result, which is exactly Route A's calculation with
`rho_in` as the frozen density. The gradient of the Harris-Foulkes functional with respect
to a rigid rotation of `rho_in` is therefore Route A's closed form,

```
G_n = integral of B[rho_in](r) x m_out(r) dr
```

since the double-counting terms of that functional are invariant under the rotation. It
costs one integral per iteration and no backward pass. **Without the coupling it is zero at
every iteration, converged or not**, because `H[R rho] = U H[rho] U^dagger` makes the output
turn with the input; so it is the coupling's torque alone, with nothing from the internal
exchange torques of a texture that has not converged yet, and the scheme does nothing to a
run without the coupling. Near self-consistency the Harris-Foulkes functional agrees with
the Kohn-Sham energy to second order in the residual, so `G_n` tends to the true
orientation gradient.

**What it is in the language of mixing.** A plain SCF moves the orientation too, since
mixing takes `rho_in` towards `rho_out` and the lean of `m_out` off `B` is a rotation; but
the lean is the anisotropy over the exchange, of order 1e-5 rad, and the mixer takes a
fraction `beta` of it, which is why the rotation is the slowest mode of the run. `G_n` is
the rotational component of the residual, and the step `w = -H^-1 G_n` divides it by the
orientation's curvature where the mixer multiplies it by `beta`, so it is a preconditioner
on three modes, as Kerker's is on the long-wavelength charge. Those three modes are exactly
the global spin rotations, which are zero modes without the coupling and are gapped only by
it; that is why they are the slow ones and why nothing else in the density needs this
treatment. The angles inside a texture are held by exchange and converge at the SCF's
usual rate, so unlike Route A the canting the coupling adds on each site is in the answer.

**What has to be solved.**

- **The curvature, and a start that does not need Route A.** A run with the coupling that
  the user already has must be able to use this on its own, so the first step carries no
  curvature at all: it is a step of a fixed trust angle along `-G_n`, as
  `_first_step_scale` does for the spiral wavevector, and the 3x3 secant builds from the
  torques that follow. Route A's relaxation, where it has been run, is an optional warm
  start, supplying both the starting orientation and a curvature. Early `G_n` is the
  force-theorem torque of an unconverged density, so every step is bounded by the trust
  angle and the steps are switched on once `dr2` falls below a threshold, in the way
  `ethr` is scheduled against it.
- **The mixer.** Anderson's history (`_densities`, `_residuals`, `_fits` in
  `scf/mixing.py`) is in the frame of earlier iterations, and a density turned under an
  unturned history is pulled back towards the old frame by the next extrapolation. So the
  history is turned with the density, which is consistent as long as a global spin
  rotation is an isometry of `rho_ddot`'s metric (true if the three magnetization
  components enter it alike, to check). Taking the rotational component out of what the
  mixer fits is **not** in the first version: it needs a metric to define that component,
  `G_n` weights the residual by `B` where the fit uses `rho_ddot`, and a projection in the
  wrong one leaves a rotational residual behind and fails in a way that looks like a mixer
  defect. The mixer's own step along the rotation is `beta` times a lean of 1e-5 rad,
  harmless beside the preconditioned step, so the order is to turn the history, measure
  whether the extrapolation still pulls back towards the old frame, and add the projection
  only if it does. P106's separate step for the magnetization (`beta_mag`) is the
  precedent for splitting what the mixer sees, if it comes to that.
- **Everything the SCF carries between iterations turns together**: the density, `becsum`,
  the mixer history above, and the wavefunctions handed to the next Davidson call as its
  start, which after a step are eigenstates in the old frame. The last is one spinor
  rotation `U(dw)`, and without it every step pays extra Davidson iterations, which would
  read as the method being slow when it is the start being cold. On PAW the closed form
  lacks the one-centre torque, which needs either that term written or a backward pass
  every few iterations.
- **The quantization axis does not turn at every iteration, and must not.** The GGA's
  axis is a static argument of the compiled SCF body (`Calculation.quantization_axis`, a
  tuple, `driver.py:2048`), so turning it per iteration would recompile per iteration and
  fill the kernel cache with one executable per angle, the same class of trap as P21's
  compiled gradient closing over its sphere. It does not need to turn: for a collinear
  texture `m(r) = s(r) |m(r)| n`, the sign `sign(m . u) = s(r) sign(n . u)` is the correct
  signed projection, up to a global sign the functional is even under, for any axis `u`
  not perpendicular to `n`, and the 36.8 meV trap was a moment turned exactly 90 degrees
  away from its axis. So the axis stays put and is reset, with one recompilation, only when
  the orientation has moved past a threshold angle from it (60 degrees, say); a
  noncollinear texture has no axis at all (`lsign = .FALSE.`) and none of this applies.
- **Noise, and why "does nothing without the coupling" has to be enforced rather than
  relied on.** `G_n = 0` without the coupling is exact for exact eigenstates; at a finite
  diagonalisation threshold it sits at the solver's noise, and dividing noise by a
  curvature that is itself zero makes steps out of nothing. So the rotation is off by
  construction when there is no coupling, and `G_n` measured at `soc_scale = 0` is the
  noise floor, which sets the smallest torque the scheme acts on and the gradient
  threshold. That measurement is paired with the same run at `soc_scale = 1`, so that the
  null is seen to be a null and not a silence. The floor depends on `dr2` as well, since
  the part of the residual that is not a rotation enters `G_n` as noise, so it is quoted
  at a stated `dr2`.
- **A second convergence condition**, `|G_n|` below a threshold, beside `dr2`, and met
  only below a matching `dr2` for the reason just given. On its own this fixes the reading
  of a plain SCF as relaxed when it is not, since `dr2` cannot see the orientation mode.

**What it converges to**, and how that is checked. The fixed point has `G = 0` (the
local-density torque `integral of B x m` vanishes pointwise at self-consistency), so it is
an unconstrained stationary state with the coupling, the state a plain SCF reaches given
enough iterations. The result is therefore the same physics reached faster, and every check
is one on an existing, unconstrained machine: a plain SCF started from the converged
density does not drift; along a direction where P87's `run_relaxed_direction` does not
drift either (a cardinal axis), the two give the same total energy to P87's floor; the
iteration count against a plain SCF from the same seed is the measurement of what it buys;
several starting orientations reach the same end point, or more than one minimum is
reported; and since stationary is not minimal, the start is away from every symmetry
element (where `G = 0` by symmetry and the run would converge without turning). What tells
a minimum from a saddle is the drift pair: plain SCFs restarted from the converged density
turned a few degrees off it, each of which has to turn back towards it. The curvature the
secant has built by the end is also reported against Route A's, but it is a weak check,
since by then it is built from differences of torques that sit at the noise floor.

**How A and C relate.** On tetragonal cobalt P87 measured the anisotropy at 0.447 meV
self-consistently against 0.552 meV for the force theorem's free energy, and the 0.105 meV
between them is the density relaxing in the coupling, ten times the self-consistent floor
of 0.011 meV. So we expect Route C to land on Route A's easy axis where the anisotropy is
simple, with a curvature about 20 per cent different, and Route A to be Route C's starting
orientation as a cheap force field is a starting geometry for a DFT relaxation.

**A first deliverable that takes no step at all**: `G_n` printed at every iteration of every
SCF with the coupling. It costs one integral and it says, for any existing run, whether its
orientation has relaxed.

## Route B, considered and dropped

Each step would have been a self-consistent run with the site directions held at `R t_i`
by P108's per-atom feedback field (`atomic fsm direction`, Elk's `fsmtype = -2`), with the
gradient read off the holding fields as `dE*/dw = sum_i m_i x B_i`, free once the run has
converged: the literal counterpart of an ionic relaxation, at one SCF per step. Dropped on
2026-09-26 in favour of Route C. What goes with it: the energy surface `E*(R)` at held
orientations away from the minimum, a gradient per site (the counterpart of relaxing every
atom separately), and a self-consistent curvature at the minimum independent of Route C's
own secant. Two things found while planning it are worth keeping if it is revived: the
identity needs `sphere_moments` and the field's potential to integrate with the same
`LocalRegions` weight, and its floor is the constraint's convergence, `2 K1 dtheta`, about
3e-9 Ry per radian at P108's 0.002 degrees on cobalt, where the field to find near the
minimum is about 1e-5 Ry, three orders below P108's and under the secant's
`FSM_MIN_STEP = 1e-6`.

## The coordinate

The orientation is stored as a rotation matrix, and every step, in either route, is a
rotation vector `w` about the current orientation, `R <- exp([w]x) R`. The gradient is only
ever taken at `w = 0`, where `exp([w]x) = 1 + [w]x` to first order, so no norm of `w`
appears inside a differentiated function. Writing Rodrigues' formula there instead puts
`sqrt(sum w^2)` at the origin, whose gradient is `0/0` at the start of every relaxation,
which is the `abs` trap in a new place. The exact rotation of a finite step is applied on
the host.

Euler angles are the readout, and are accepted as the starting orientation, but they are
not what the optimizer moves in: at `beta = 0` the angles `alpha` and `gamma` are the same
rotation, so the map from `w` to `(alpha, beta, gamma)` is singular there, and `beta = 0` is
an easy axis along `z`, which is where a uniaxial answer usually sits.

Which generators are live is decided by the texture, from the moment tensor
`M_ab = integral of m_a m_b`, whose leading eigenvector `_collinear_axis` already reads.
Rank one is collinear: the rotation about the axis is exactly null and is dropped. Rank two
is coplanar: the smallest eigenvector is the plane normal, and the rotation about it is the
phase, frozen by default because its curvature is orders below the tilt's and an optimizer
lets noise push a flat coordinate (its component is reported either way, so its size is
seen). Rank three frees all three. A `free=` argument overrides this, as in
`relax_spiral_q`.

Route A's optimizer is `relax/bfgs.py` on a three-vector coordinate, as P21 uses it for the
spiral wavevector, with `hessian_scale` set from the first step: P21 found the initial
inverse Hessian out by two orders on a millirydberg surface, and this one is microrydberg.
The loop hands BFGS the free energy and not `sum w eps`, because the gradient is the free
energy's, and P60 measured the entropy at 55 per cent of the band energy's slope at
`degauss = 0.02`, so a line search on the band energy would see an energy inconsistent with
its gradient.

A symmetric orientation is stationary by symmetry, a hard axis as much as an easy one, so a
relaxation started on one reports convergence without moving, a null that reads as a pass.
The start is therefore taken off every symmetry element, a start whose gradient is already
below threshold is reported as such rather than as converged, and Route A's final point gets
its curvature from a central difference of the gradient, three more gradient evaluations,
so that a minimum is told from a saddle.

## What in this code it builds on

- **P58 and P97**, the force theorem and its PAW form on one file, with `soc_scale = 0` on
  the self-consistent leg, which is Route A's handoff.
- **P60**, `forces/torque.py`, the one-plane torque at frozen states with its k-chunked
  backward pass. Its `rotated_density` reads a collinear axis off the density and refuses a
  genuinely noncollinear one by name; Route A replaces it with `rotate_texture(field, R)`,
  which is `R m` on the three magnetization components and reads no axis, applied to the
  density and to each species' `becsum` alike. `ultracell/kramers.py:time_reversed` does
  the same operation with `-1` in place of `R`.
- **`workflows/anisotropy.py:_with_quantization_axis`**, which already builds a rotation
  matrix and turns every species' `angle1`/`angle2` and the `STARTING_MOMENTS` card rigidly
  with it; it is factored into `_with_rotation(system, R)`. It carries the GGA quantization
  axis with the moment, which cost 36.8 meV between `x` and `z` on a cubic cell when it was
  left behind. Route C keeps its axis fixed and resets it only past a threshold angle, for
  the reason given in its section.
- **P87**, `run_relaxed_direction` and its drift, which is Route C's reference along a
  cardinal axis and its closing check.
- **P106 and `scf/mixing.py`**, the separate magnetization step and the Anderson history
  that Route C has to turn and split.
- **P21**, `relax_spiral_q`, the shape of Route A's loop and the BFGS settings for a
  coordinate that is not a position.
- **P92**, `scf/spin_torque.py:exchange_torque`, for Route A's closed-form check and Route
  C's per-iteration torque.

New code, as far as it can be listed before writing it. For Route A: `rotate_texture`, a
band free energy at a rotation and its three-component torque (the chunked form carries
over from P60 with a vector in place of a scalar), `run_orientation_torque` and
`relax_orientation` in `workflows/`, and `Calculator.get_orientation_torque` and
`get_relaxed_orientation`, forwarding the shared options by name. For Route C: an option of
`run_scf` (name to choose) carrying the per-iteration torque, the trust angle and the 3x3
secant, the turning of the density, `becsum`, the quantization axis and the mixer history,
and the second convergence condition, reached through `Calculator.get_scf`. There is no
`pw.x` variable for any of it, so the entry point is Python, as it is for `relax_spiral_q`.
One piece of physics is new in Route A: PAW's `ddd_paw` of a turned `becsum` inside the
traced energy, since `band_energy_at_angle` builds its potential with no `becsum` and so
reaches no PAW dataset today. `fixed_density_states` takes no starting wavefunctions; the
previous step's states turned by the spinor rotation `U(dR)` are a cheap start for the next
diagonalisation, and whether they are worth the argument is a timing.

**Refused by name in Route A**: a Hubbard U, whose `ns` has a spin structure that would
need turning; a potential-only meta-GGA, whose `tau` is not in the handoff and which has no
energy; a spin spiral in the generalized Bloch representation, which cannot carry the
coupling (a supercell can, and the unit-cell route is step 6 below); and symmetry, since
the magnetic group, and with it the k-set, moves with `R`, so `nosym` is required. A source
converged under a constraining field is refused at first, and the lift is to turn the field
with the texture, which `fixed_density_states(field=)` can carry and which keeps the frozen
state stationary and the field's energy invariant.

**Refused by name in Route C**: a potential-only meta-GGA, which has no energy for `G_n` to
be the gradient of; a Hubbard U at first, since its `ns` is mixed with the density and
would have to be turned with it; an applied field or a constraint, since the fixed point
would belong to a different functional and a held direction would fight the step; a spin
spiral in the generalized Bloch representation; and symmetry, since the orientation, and
with it the magnetic group, moves during the run.

## The order to build it in, and the number each step has to produce

1. **Route A on a collinear source.** `rotate_texture`, `_with_rotation`, the
   three-component torque. Numbers: the torque of P60's k-dial check on tetragonal cobalt
   (`tools/cluster/torque_batching.py`: 45 degrees from `z` towards `x`, the two-file
   route, `conv_thr = 1e-10`), `-4.059378978382e-05` Ry per radian, which is `-K1` at
   0.552 meV, reproduced as one component to twelve digits; the component along the moment
   zero to round-off, beside a nonzero one at the same oblique start so that the zero is
   seen to mean something; `F(w = 0)` at frozen states against `sum w eps - TS` to 1e-10;
   the Hartree, exchange-correlation and Ewald terms equal between two rotations to 1e-12.
2. **Route A on a noncollinear source.** Numbers: each component against a central
   difference of `F` from two separate diagonalisations (P60's standard is six digits); the
   closed form `integral of B x m_out` on a norm-conserving dataset; every component zero
   at `soc_scale = 0`, beside a nonzero one at 1; and the rotation identity. Turning the
   spins by `R` in a fixed lattice is the same calculation as turning the lattice and the
   positions by the inverse rotation under fixed spins, with the same crystal coordinates
   and the same grid, so `F` and its gradient must agree to round-off through
   `System.with_cell`, which shares no spin-rotation code with the route it checks. It is
   also how Elk changes the orientation (see below). The cell is the four-cell cobalt helix
   decided below.
3. **The relaxation for Route A.** Numbers: tetragonal cobalt from an oblique start
   converges on `c`, the easy axis of both P60 and P87, and the curvature there is `2 K1`
   against P60's `K1` (0.531 meV at `degauss = 0.002`, 0.552 at 0.02); the number of steps
   it takes, which is the measurement of the Hessian scale.
4. **Route C.** First the torque printed per iteration, with no step: its noise floor at
   `soc_scale = 0` beside its value at 1, and on an existing plain run of cobalt started
   oblique, the torque still nonzero when `dr2` has reached `conv_thr`, which is the
   measurement that the plain SCF had not relaxed. Then the step. Numbers: on cobalt from
   an oblique start, the angle to `c` against iteration beside a plain SCF from the same
   seed; the total energy at the end against P87's `run_relaxed_direction` along `c`, to
   P87's floor of 0.011 meV; a plain SCF restarted from the converged density drifting by
   less than `RELAXED_DRIFT_TOL`, and plain SCFs restarted a few degrees off it turning
   back; the same end point from three starting orientations, one of them with no Route A
   warm start at all; the number of recompilations over the run, which has to be the
   number of axis resets and nothing more; and on the noncollinear cell of step 2, the
   same end point from Route A's minimum and from a start 30 degrees away from it.
5. **The spiral plane on a production cell**: a commensurate approximant of the NiI2 or
   NiBr2 spiral in a supercell, Route A to find the plane and Route C to converge it with
   the coupling self-consistent. Size it before choosing it: bulk NiI2's in-plane
   wavevector of about 0.14 reciprocal lattice units (from memory, to check) puts the
   approximant at about seven cells, 21 atoms with the coupling, and what bounds Route A is
   the backward pass, which holds `nbnd x 2 x N_smooth` complex numbers per k-point in
   flight (`SpinorHamiltonian._local_block`, the block P60 sized at 33 GB for one k-point
   of the P74 slab), beside the states at `nk x nbnd x 2 npwx`; Route C needs no backward
   pass on a norm-conserving or ultrasoft dataset. The ultracell is the tempting shortcut
   and has a trap of its own. P121's truncation error on the hydrogen helix is 1.28e-6 to
   4e-7 Ry per cell on the Kramers-closed basis, a few per cent of cobalt's anisotropy per
   atom and not small against a weak one, and nothing guarantees it is the same at every
   orientation once the coupling is inside the frozen states (an expectation, to measure).
   The control cannot see it: without the coupling the Kramers-closed span is exactly
   invariant under a global spin rotation, so the `soc_scale = 0` torque is zero whatever
   the basis. In an ultracell the torque is therefore converged in `nbnd` against a
   supercell before it is believed.
6. **The unit-cell route to the spiral plane**, which is `SPIRAL-SOC-NEXT.md` step 2 with
   `R` as the coordinate. For a spiral the Dzyaloshinskii-Moriya energy is first order in
   the coupling and linear in the plane normal, so it is the leading orientation
   dependence, where for a collinear magnet P120 measured the first-order term at 0.3 per
   cent of the whole. The generalized Bloch theorem wants the spiral axis fixed, so there
   the rotation is applied to the coupling's spin frame instead of the texture, which is
   the same calculation, since `U(R)^dagger (L . sigma) U(R) = L . (R sigma)`. A
   dependency, not a first deliverable; at a commensurate `q` and weak coupling it has to
   agree with steps 2 and 5 up to the second-order remainder.

## What the reference codes have

Neither code has this. QE's `lforcet` (`PW/src/potinit.f90:343-376`,
`nc_magnetization_from_lsda`) turns a collinear LSDA density onto the one direction of
`angle1(1)`/`angle2(1)` and nothing more: no noncollinear source, no gradient, and no
magnetic torque anywhere in `PW/src`. Its constraints print a field only for the
total-moment scheme, and not as a torque. Elk's tasks 28 and 29 (`mae.f90`) scan a set of
directions with one full SCF each, and they rotate the **lattice**, not the spins
(`axangrot` applied to `avec`, lines 66-73), with `cmagz = .true.` holding the
magnetization collinear along `z` at every point and `fsmtype = 0`. So Elk covers collinear
cells only, at held directions, which is P87's quantity rather than Route C's: its energies
compare with P87's along the directions where P87 does not drift, and its lowest direction
with Route C's easy axis, with the difference that `cmagz` forbids the canting Route C lets
in written beside the number. Elk's task 160 (`torque.f90`) is the `m x B_xc` diagnostic
P92 transcribed, not an orientation gradient. For a noncollinear texture there is therefore
no external number, and the identities of steps 2 and 4 are the number. The external
numbers live in the collinear limit: P58's force theorem against `pw.x`, and Elk's task 28
against P87 and Route C. The `mae.f90` line numbers above are read off Elk 11.0.2, while
Triton's module is 10.2.4, and `CLAUDE.local.md` records the two versions disagreeing on
the default mixer, so a comparison says which version ran and reads its settings off that
run's `INFO.OUT`.

## What the finished phase leaves behind

The five things `CLAUDE.md` asks of a phase, as far as they can be fixed now.

- **The number**: the identities of steps 1 to 4, and for cobalt the easy axis, `K1`
  against P60 and the relaxed energy against P87.
- **One README row**, a quantity rather than a routine: the orientation of a magnetic
  texture that minimises its energy under spin-orbit coupling. `QE (✓)`, with a note that
  `lforcet` gives the force-theorem energy of a collinear density along one direction per
  run, with no gradient and no relaxation; `Elk (✓)`, with a note that task 28 scans
  directions self-consistently for a collinear cell by rotating the lattice. Neither
  relaxes the orientation, and neither takes a noncollinear texture.
- **A `docs/features.tex` entry**: Route A's gradient and Route C's per-iteration torque as
  equations, the entry points, a snippet that has been run, and the refusals above in the
  amber box.
- **A notebook**: one-atom tetragonal cobalt from an oblique start, relaxed onto `c` by
  Route A and converged there by Route C, with the angle to `c` against iteration, plain
  SCF beside Route C, as the figure. It fits the ten minutes only if both take few steps,
  so the cell is fixed after steps 3 and 4 have counted them.
- **The timing pairs**: Route A's collinear leg against one `pw.x` `lforcet` NSCF per
  direction, and Route C's SCF against one `pw.x` SCF with the coupling from the same seed
  to the same `conv_thr`, one core each, with both iteration counts and the angle each run
  ends at, since the plain run reaching `conv_thr` without reaching the easy axis is the
  point of the comparison.

## What to decide at the start of that session

- ~~Route A first or Route B first.~~ **Decided 2026-09-26: Route A first**, because it
  tests the rotation, the loop and the coordinate cheaply.
- ~~Route B, and whether it holds the texture rigid, only its frame, or every site.~~
  **Decided 2026-09-26: Route C instead of Route B.** The question of rigid against free
  internal angles goes with it, since Route A is rigid by construction and Route C lets the
  internal angles relax.
- ~~The validation cell for the noncollinear steps.~~ **Decided 2026-09-26: the four-cell
  commensurate helix of tetragonal cobalt along `c`**, 90 degrees per cell, which is
  stationary by symmetry without the coupling, has all three generators live, and serves
  both routes. The alternatives were tetragonal cobalt doubled along `c` with the two
  moments canted and held (zero-cant limit P60's `K1`, but a constraint, so Route A only),
  and the existing canted iron pair `fe2-canted-soc.in` (a `pw.x` reference with the
  coupling, but cubic, so its anisotropy is quartic in the direction cosines and the torque
  small). The helix is built for this, and its cost on the workstation is measured before
  it is committed as a test.
- ~~Whether a source converged under a constraint is refused in Route A or taken with its
  field turned along with the texture.~~ **Decided 2026-09-26: refused at first**, so Route
  A takes only textures that are stationary on their own (a ferromagnet, an
  antiferromagnet, a commensurate helix), which is all the chosen cell needs. Turning the
  field rigidly with the density stays recorded above as the lift.

Every decision this plan listed is now taken; the next session starts at step 1.
