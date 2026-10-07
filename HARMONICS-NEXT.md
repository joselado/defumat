# High harmonics and the third harmonic: what to build, in what order, and what each piece costs

A plan written on 2026-10-06 for the session that implements it.

> **Status, 2026-10-06 evening.** The first two phases are implemented: the repair of the
> row at k + G = 0, the propagator at a frozen potential, the linear response, the
> high-harmonic workflow and the little group of the field are `PLAN.md` P134, and the
> perturbative orders with the third harmonic by the real-time route are P135. Two
> departures from the text below, each measured: the projectors at k + kappa come from a
> Chebyshev table of g_l(q^2) rather than from `at_kcart`, since the rebuild by the radial
> transform was three quarters of a step; and the linear identity against the Kubo sum
> carries the mesh sum of band curvature, `sigma_RT = sigma_Kubo + i D/(Omega z)`, which on
> two k-points is the size of the answer. The stage "Updating the potential in time" is
> `PLAN.md` P136 (2026-10-06 night), with one departure, measured: the potential is
> `v_scf + U[rho(t)] - U[rho0]` rather than `U[rho(t)]`, because the propagated states'
> density is the ground state's only on its own mesh (1.0e-2 Ry apart on a 6x6x6 mesh of
> silicon converged on 4x4x4). The third phase, the frequency-domain hierarchy, is P137
> (2026-10-07): the computed bands held exactly by a projector and the rest of the sphere by
> a right-preconditioned BiCGStab, both chosen by a review that measured the candidates
> (QE's GMRES(4) of `solve_e_fpol.f90` took about a thousand products above the gap where
> this takes 40 to 50). The self-consistent hierarchy at first order is P138, `eps_M(w)` with
> local fields band-complete, the static shift equal to the Sternheimer stack's to 8e-6; above
> first order it is refused, since the second-order density artefact is 7 per cent of `rho` on a
> 2x2x2 mesh. What is left is the list "What is left for later". The rest of this file is the
> plan as written.

When written, nothing below was implemented: the package had no real-time propagation, no
$\chi^{(3)}$ and no entry point for either, and `README.md` listed real-time propagation
under "Not yet". `PLAN.md` stays
the record of what is done, and this file is about what is to be done, why in this order,
and what was and was not verified while planning it. The next free phase number on the
day of writing was P134.

We want two quantities. The first is the high-harmonic spectrum of a solid, the emission
of a crystal driven by an intense pulse, which is a non-perturbative response and can only
come from propagating the Kohn-Sham states in time. The second is the third-harmonic
susceptibility $\chi^{(3)}(-3\omega;\omega,\omega,\omega)$, which is the weak-field limit
of the lowest odd harmonic above the fundamental and is a perturbative quantity. The two
share one engine, and the plan is built around that: the propagator is written once, the
high harmonics are its output at a strong field, and the perturbative orders are its
derivatives with respect to the field amplitude at zero field, taken by forward-mode
differentiation through the propagation rather than by fitting runs at several
amplitudes.

One thing was measured while planning and is the reason the plan has this shape. On a
one-dimensional plane-wave model with a k-dependent separable projector
(`tools/realtime/toy_orders.py`), the first, second and third orders of the current from
nested `jax.jvp` through a real-time run agree with an independent frequency-domain
calculation to between 1.2e-6 and 1.7e-6 on all five components, the third harmonic
included. The section "The claim the plan rests on" has the numbers and what they do not
cover.

The plan was then reviewed against the code by a second session, and the review found
one defect in code that exists today and that this work would have inherited silently:
the second and higher derivatives of $H(\mathbf k)$ are wrong on the row
$\mathbf k+\mathbf G=0$. It is the first thing to repair and has its own section, "The
row at k + G = 0". The other corrections of that review are folded into the text below.

## The physics, and the one variable that carries the field

We take a uniform electric field $\mathbf E(t)$ and describe it in the velocity gauge by a
vector potential $\mathbf A(t)$ with $\mathbf E=-(1/c)\,\partial_t\mathbf A$. What the
electrons see is a shift of the crystal momentum, so the simplest way to carry the field
is as a wavevector,

$$\boldsymbol\kappa(t)=\frac{e}{\hbar c}\mathbf A(t),\qquad
\hbar\,\partial_t\boldsymbol\kappa=-e\,\mathbf E(t),$$

in units of 1/bohr, which is the same number in Hartree and in Rydberg atomic units. The
time-dependent Kohn-Sham equation for the periodic part of each occupied state is then

$$i\,\partial_t\,|u_{n\mathbf k}(t)\rangle
 = H\big(\mathbf k+\boldsymbol\kappa(t)\big)\big[\rho(t)\big]\,|u_{n\mathbf k}(t)\rangle ,$$

meaning that the Hamiltonian at time $t$ is the ordinary Bloch Hamiltonian evaluated at a
displaced k-point, on the plane-wave sphere that was built for $\mathbf k$ and is never
rebuilt. For a local potential this is the minimal coupling
$(\mathbf p+\hbar\boldsymbol\kappa)^2/2m$. For a nonlocal pseudopotential the gauge
transformation turns $V_{NL}$ into $e^{-i\boldsymbol\kappa\cdot\mathbf r}V_{NL}\,e^{i\boldsymbol\kappa\cdot\mathbf r'}$,
and in plane waves that is the projector evaluated at
$\mathbf k+\mathbf G+\boldsymbol\kappa$, since the phase $e^{-i\boldsymbol\kappa\cdot\mathbf R}$
cancels between the two projectors of one atom. So for a norm-conserving dataset the
whole coupling to the field is exactly $H(\mathbf k+\boldsymbol\kappa)$ on the frozen
sphere, with no expansion in the field and no empty states. In the code this holds as it
stands away from $\mathbf k+\mathbf G+\boldsymbol\kappa=0$: `at_kcart` rebuilds only the
kinetic term and the projector core, the structure factor is applied afterwards and
carries no $\boldsymbol\kappa$, and the core correction lives in the potential. The
exception is the origin row of the next section.

The quantity that is measured is the macroscopic current,

$$\mathbf J(t)=-\frac{e}{\Omega}\sum_{n\mathbf k} w_{n\mathbf k}\,
\Big\langle u_{n\mathbf k}(t)\Big|\,\frac{1}{\hbar}\frac{\partial H}{\partial\mathbf k}\Big|_{\mathbf k+\boldsymbol\kappa(t)}\Big|u_{n\mathbf k}(t)\Big\rangle ,$$

which is the derivative of the band energy with respect to $\boldsymbol\kappa$ at frozen
wavefunctions. The emitted spectrum is $|\mathrm{FT}[\partial_t\mathbf J]|^2$, equivalently
$\omega^2|\mathbf J(\omega)|^2$, and the linear dielectric function is
$\epsilon(\omega)=1+4\pi i\,\sigma(\omega)/\omega$ with $\sigma=J(\omega)/E(\omega)$.

Two properties of this expression should be stated before any code is written. The
derivative of $|\mathbf k+\mathbf G+\boldsymbol\kappa|^2$ already contains the
diamagnetic current, the term Elk adds by hand as $-(1/c)\mathbf A\,N$ in `timestep.f90`,
and the derivative of the projectors already contains the commutator $[\mathbf r,V_{NL}]$
that real-space codes add to the momentum, so nothing is added separately and a session
that transcribes Elk's line would count the diamagnetic term twice. And only the kinetic
and the nonlocal terms carry $\boldsymbol\kappa$, so the current needs no Fourier
transform at all.

The perturbative orders are defined by scaling the field,
$\boldsymbol\kappa(t)=\lambda\,\mathbf a(t)$, and expanding the current,
$\mathbf J(t)=\sum_n\lambda^n\mathbf J^{(n)}(t)$ with
$\mathbf J^{(n)}=(1/n!)\,\partial^n_\lambda\mathbf J|_{\lambda=0}$. This is a derivative
of code that exists once the propagator does, so it is taken with `jax.jvp`, and the
orders come out separated exactly rather than up to the next order in the field. To read
a susceptibility off $\mathbf J^{(n)}(t)$ we switch the field on adiabatically,
$a(t)=e^{\eta t}\cos\omega t$ from $t=-T$ to $t=0$. The reason for this choice is that
with an exponentially growing envelope the response of order $n$ is exactly
$e^{n\eta t}$ times a periodic function, whose Fourier coefficients are the response
functions at the complex frequencies $m\omega+in\eta$, so that $\eta$ is a Lorentzian
broadening per photon, the same parameter a sum over states calls its broadening, and no
dephasing has to be invented for a unitary evolution. The third harmonic is the
$m=3$ coefficient of $e^{-3\eta t}J^{(3)}(t)$.

## The row at k + G = 0

A projector column is $Y_{lm}(\hat{\mathbf q})\,f_l(|\mathbf q|)$ with
$\mathbf q=\mathbf k+\mathbf G$, and both factors guard the origin: `gvectors.modulus`
returns a constant zero inside $|\mathbf q|^2\le10^{-8}$ (`ORIGIN_TOL`,
`basis/gvectors.py:107`), since the square root has no derivative there, and the
harmonics return zero for a vector with no direction. `_origin_tangent_rule`
(`pseudo/projectors.py:568`) puts back the one first-order tangent this loses, the
$l=1$ column, which is linear in $\mathbf q$. Nothing puts back the second order: the
curvature of the $l=0$ form factor at $q=0$, the $q_aq_b$ growth of $l=2$ and the cubic
part of $l=1$ are all differentiated as zero.

Measured on `si2-nosym.in` (4x4x4, `ecutwfc = 12`), `second_matrix_elements` $xx$
against a central difference of `matrix_elements` at $\mathbf k\pm h\hat x$, both arms
outside the guard: at $\Gamma$ the two differ by **3.46e-2 Ry bohr$^2$** in one element
(on a block of norm 5.15), the same at $h=10^{-3}$ and at $3\times10^{-4}$, while at two
other k-points they differ by 2e-7 and 2e-8, which is the $h^2$ of the stencil. So the
finite difference is right and the nested `jvp` is not. The review measured it and the
planning session reproduced it with the same script.

What it means here, in the order it would bite:

- every order above the first at zero field is wrong at $\Gamma$: $h_2$, $h_3$, $h_4$
  and the nested derivative of the propagation. $h_2$ at $\Gamma$ is the diamagnetic
  part of the linear response, so even the first order gains a spurious tail;
- at a finite field the primal itself is guarded: while $|\boldsymbol\kappa|<10^{-4}$
  at $\Gamma$ the $l=1$ column is zero, so a linear-response kick below that amplitude
  loses the coupling for the whole run, and a pulse passes through the guard once per
  zero crossing of $\boldsymbol\kappa$;
- the full unshifted mesh that every comparison below asks for always contains
  $\Gamma$, and the model of the previous section cannot see any of this, since its mesh
  is offset and its form factor is analytic;
- `VelocityOperator.apply_second` has the defect today, and the shift current is its
  consumer (`response/photocurrent.py:647`).

The repair is to write the column so that it needs no guard: a regular solid harmonic
$|\mathbf q|^l\,Y_{lm}(\hat{\mathbf q})$, which is a polynomial in the components of
$\mathbf q$, times $g_l(q^2)=f_l(q)/q^l$, which is an analytic function of $q^2$. This
is differentiable to every order at the origin, where a second custom rule would cover
order two and leave three and four. Its test is the measurement above brought to the
stencil's own error at $\Gamma$, repeated for the third derivative, and the first-order
number `_origin_tangent_rule` was written against (the $\Gamma_1$ by $\Gamma_{15}$ block
at 0.3695 of its value without it). Until it is done, a first version can run on a
uniform mesh that avoids $\Gamma$ with a kick of at least $10^{-3}$ bohr$^{-1}$, which
is a workaround and makes the comparisons against the unshifted-mesh references
impossible, so we do not recommend it. Whether the shift current's recorded numbers move
is a separate question for `OPEN.md`.

## Why plane waves and occupied states, and what Elk is the reference for

Elk is the one established code here with real-time propagation, and we read it first
(`vendor/elk/src/tddft.f90`, `timestep.f90`, `genhmlt.f90`, `jtotk.f90`,
`genafieldt.f90`, `tdinit.f90`, `dielectric_tdrt.f90`, in the elkpy checkout). Its
algorithm is to expand the time-dependent states in the ground-state Kohn-Sham states of
each k-point, `nstsv` of them, to build
$H(t)=H_{KS}[\rho(t)]-(1/c)\,\mathbf A(t)\cdot\mathbf p$ in that basis from stored
kinetic and momentum matrices, to diagonalise it at every step and to multiply each
instantaneous eigenvector by $e^{-i\epsilon\,\Delta t}$.

We do not transcribe that algorithm, and the reason is one property of the Hamiltonian.
Elk can add $\mathbf A\cdot\mathbf p$ linearly because an LAPW potential is local, so the
field enters to first order exactly and $A^2$ is a phase. A pseudopotential projector at
$\mathbf k+\mathbf G+\boldsymbol\kappa$ is not linear in $\boldsymbol\kappa$, so the same
construction in a band basis would be an approximation in the field on top of a
truncation in the bands, with nothing to check it against. The truncation alone is
severe in the velocity gauge: `NONLINEAR.md` measured the shift current's intermediate
sum at 0.52 of its value at 120 bands of 158, and a truncated velocity-gauge response is
known to diverge at low frequency (arXiv:1710.01300). Propagating the occupied states on
the full sphere has neither problem, it reuses the Hamiltonian application that every
other part of the code is built on, and it is what the plane-wave and real-space codes
that computed high harmonics in silicon and diamond do (arXiv:1609.09298,
arXiv:1705.10707, arXiv:1810.06500).

Elk remains the reference for the protocol and for numbers:

- `genafieldt.f90`: the pulse as a sum of Gaussian-enveloped sines with a phase and a
  chirp, plus polynomial ramps and steps. We take the three shapes, since they make an
  Elk input reproducible here, and add a $\sin^2$ envelope and the adiabatic exponential.
- `tdinit.f90`: the k-set is reduced only with the symmetries that leave $\mathbf A(t)$
  invariant at all times, and `timestep.f90` symmetrises the total current as a vector
  (`symvec`). This is the rule for the stage "The little group of the field".
- `timestep.f90`: `ntsorth`, a re-orthonormalisation every 1000 steps, and `ntsbackup`,
  a checkpoint of the states. We keep both as options and report the norm drift.
- `dielectric_tdrt.f90` (tasks 480 and 481): $\epsilon(\omega)$ from the Fourier
  transform of $\mathbf J(t)$, with `jtconst0` removing the constant part and a
  Lorentzian filter of width `swidth`. Task 481 assumes a step in $\mathbf A$ at $t=0$,
  a delta kick in the field, and is the numerically stable one.
- `examples/TDDFT-time-evolution/`: `Si-dielectric` ships its output
  `EPSILON_TDRT_11.OUT`, `Si-ramp` ships `JTOT_TD.OUT` for a strong linearly growing
  field, and `GaAs-HHG` is Elk's own harmonic example (a 0.544 eV pulse of 48 fs FWHM at
  $7.5\times10^{8}$ W/cm$^2$, 40000 steps of 0.1 a.u., a 12x12x12 mesh, 32 empty states).
  Elk has no task that extracts harmonics, so its spectrum is the Fourier transform of
  `JTOT_TD.OUT`, and it has no $\chi^{(3)}$ at all: task 125 is its only nonlinear one.

A comparison against Elk is never like-for-like and the plan says where: Elk's basis is
LAPW with a truncated set of states, its GaAs example uses a scissor that a full
plane-wave propagation cannot apply (see "What is refused"), and the built Elk is 11.0.2
on the workstation and 10.2.4 on Triton. For a comparison that shares the functional,
run Elk with `scissor 0` and the LDA.

## Units, which is where a day is most easily lost

The internal units are Rydberg atomic units and the equation is
$i\,\partial_t\psi=H\psi$ with $H$ in Ry, meaning that the internal time unit is
$\hbar/\mathrm{Ry}=48.378$ as, twice the Hartree unit of 24.189 as that Elk's `dtimes`
and `tstime` and every paper quoted here use. `units.py`'s `AU_SEC` is the Hartree one.
The rule we propose is one conversion at the input boundary and none inside:

- the field is carried as $\boldsymbol\kappa(t)$ in 1/bohr, with
  $\boldsymbol\kappa=\mathbf A_{Ha}/137.036$ for an Elk vector potential;
- times are given by the user in Hartree atomic units or in femtoseconds and converted
  once, so `dt = 0.1` means what it means in Elk;
- a pulse is specified by a photon energy and a peak intensity, with
  $E_0[\mathrm{a.u.}]=\sqrt{I/3.509\times10^{16}\ \mathrm{W/cm^2}}$ and
  $\kappa_0=E_0/\omega$ in Hartree units;
- $\mathbf J$ and $\mathbf E$ are reported in Hartree atomic units, the susceptibilities
  in SI ($\chi^{(2)}$ in pm/V as `get_shg` does, $\chi^{(3)}$ in m$^2$/V$^2$).

Elk's `bfieldc` factor of 274 that P86 found is the sibling of this trap. The scale of
the result is pinned by two checks together rather than by reading the conversions: a
wrong factor on $\mathbf J$ shows as the same factor at first and second order, a wrong
factor on $\mathbf E$ shows once at first order and twice at second, so the linear and
the second-harmonic comparisons below are two equations for the two unknowns, and the
third order then has no freedom left.

The size of $\kappa$ matters and is not small. For the published silicon run (0.43 eV,
$10^{11}$ W/cm$^2$) $\kappa_0=0.11$ bohr$^{-1}$, for MgO (0.93 eV,
$3\times10^{12}$) 0.27, for diamond at 800 nm and $2\times10^{13}$ 0.42, against a
$\Gamma$-X distance of 0.62 in silicon. See "The frozen sphere under a large shift".

## The claim the plan rests on

The claim is that nested forward-mode derivatives through the propagation give the
perturbative orders of the current, and that the adiabatic envelope turns each order into
the response function at a complex frequency. `tools/realtime/toy_orders.py` tests it on
a model that has every structural feature of the real problem: nine plane waves on a
frozen sphere, a local potential without an inversion centre, a separable nonlocal term
whose form factor depends on $k+G$, a fourth-order Taylor step at the midpoint, and the
current as the gradient of $\langle\psi|H(k+\kappa)|\psi\rangle$ with respect to
$\kappa$. The reference is the steady-state hierarchy solved by dense linear algebra at
the frequencies $m\omega+in\eta$, which shares the model Hamiltonian with the real-time
route and nothing else. Relative differences, for the components $(n,m)$ = (1,1), (2,2),
(2,0), (3,3), (3,1):

| steps per period | $\eta T$ | (1,1) | (2,2) | (2,0) | (3,3) | (3,1) |
|---|---|---|---|---|---|---|
| 400 | 14 | 2.0e-5 | 2.9e-5 | 5.1e-5 | 1.7e-5 | 2.6e-5 |
| 1600 | 14 | 1.2e-6 | 8.4e-6 | 3.2e-5 | 9.1e-6 | 5.2e-7 |
| 1600 | 20 | 1.3e-6 | 1.4e-6 | 1.2e-6 | 1.6e-6 | 1.7e-6 |

The first-order error falls by 16 when the step is divided by 4, which is the second
order of the midpoint rule, and the plateau of the other components at $\eta T=14$ goes
away at $\eta T=20$, so it is the transient left by starting the envelope at a finite
time, of relative size $e^{-\eta T}$ times a resonance factor, and it does not decay
because the evolution is unitary. The model run takes three seconds.

What this does not cover: the model has one occupied band and no self-consistent
potential, so it says nothing about the tangent of the density through `v_of_rho`, and it
does not test the conversion from $J^{(3)}$ to $\chi^{(3)}$ in SI units, which is the
job of the validation ladder.

## What exists to build on

All of this was read from the code on 2026-10-06 and should be checked again before use.

- `Calculation.at_kcart(kcart)` (`scf/driver.py:4112`) returns the calculation with
  $|\mathbf k+\mathbf G|^2$ and `vkb` rebuilt at a traced `kcart` on the frozen sphere.
  It is the whole of $H(\mathbf k+\boldsymbol\kappa)$, and
  `calc.at_kcart(k0 + kappa).hamiltonian_from(terms)` is the pattern
  `VelocityOperator._operator` already uses (`response/velocity.py:304`), which shows
  that it traces. It refuses a spiral and gamma-only storage. Two properties make it
  less than "a few lines" for a propagation: it takes the whole mesh, `(nk, 3)`, and
  rebuilds `vkb` for every k-point (`driver.py:4201`), so a k-chunk in flight needs a
  row-subset calculation first (`at_rows`, `driver.py:4020`, and the `projector_rows`
  machinery of `tests/unit/test_row_subset.py`); and the form factors are a direct
  Simpson radial transform per $|\mathbf q|$ (`pseudo/formfactors.py:431`), about
  $n_\beta\,n_{pwx}\,n_r$ Bessel evaluations per k-point, $1.5\times10^{5}$ for silicon
  at 12 Ry against sixteen transforms of a $15^3$ box, so the rebuild is expected to cost
  as much as the four Hamiltonian applications of a step or more. Not timed.
- `Calculation.local_terms(v_scf)` and `hamiltonian_from(terms)` (`driver.py:5306`,
  `:5349`) split off the half that does not depend on k, so a frozen potential is built
  once for the whole run.
- `Hamiltonian.apply(psi, ik)` (`hamiltonian/operator.py:367`) and
  `SpinorHamiltonian` take an integer `ik`; k enters through the stored `kinetic` and
  `projectors.vkb` only.
- `VelocityOperator` gives $\partial H/\partial\mathbf k$ applied to states and
  `apply_second` the second derivative. There is no occupation-weighted total current
  anywhere, so $\mathbf J$ is new: the $\boldsymbol\kappa$ derivative of the kinetic and
  nonlocal expectation values on top of the row-subset `at_kcart`.
- `Calculation.density(wavefunctions, weights)`, `potential(rho)` and `v_of_rho` are the
  SCF driver's own, and `SCFResult.occupations` is `wg`, occupation times k-weight.
- `fixed_density_states` (`workflows/nscf.py:83`) gives the ground states on a caller's
  k-set, `is_reduced` (`system/kpoints.py:536`) is the one predicate for a wedge, and
  `for_spin` is the boundary a caller-built k-set must cross.
- `map_k` and `sum_k` (`batching.py`), `defumat.eager.compiled` and
  `compiled_function`, `sizing.estimate_size` and the dtype policy `cell.precision` are
  what a new walk over k, a new closure, a new budget and a new array go through.
  `compiled` retraces at each call to find its key, so a Python loop over chunks uses
  `compiled_function` with the chunk's arrays as arguments.
- The comparisons: `run_conductivity` (`optical_conductivity`, interband and Drude,
  ultrasoft allowed, collinear `nspin = 2` refused), `run_absorption` (`chi_0` plus the Dyson equation, kernels `rpa`,
  `alda`, `lrc`, `bootstrap`, norm-conserving and unpolarized only), and `run_shg`
  ($\chi^{(2)}(-2\omega;\omega,\omega)$ by a sum over states, validated against Elk's
  task 125). `tests/data/qe/alas-shg.in` is norm-conserving zincblende AlAs
  (`Al.pz-vbc`, `As.pz-bhs`) and is the cell for anything of even order.

## The stages

Each stage ends with a number, and the order is chosen so that each number is available
before the next stage needs to trust it. The first phase is the first four together
with the little group of the field, which is written here after the update of the
potential and is done before the notebook (see "Decisions"); the second phase is the
perturbative orders by the real-time route, and the third the hierarchy.

### Measure before building

The origin row is repaired first, with the test its section gives. Then three
measurements decide the defaults, each takes an hour, and none needs new physics.
The cost of one Taylor step per band and per k-point on silicon at `ecutwfc = 12` and 30
on one core and on the D22 card, split into the four Hamiltonian applications and the
rebuild of `vkb` at $\mathbf k+\boldsymbol\kappa$, which happens at every step and is the
cost a ground-state code never pays; if the radial transform dominates, as its operation
count suggests, the answer is a one-dimensional table of $g_l(q^2)$ per dataset
($n_\beta\times n_q$ numbers, QE's `interp_beta` in a form smooth enough to be
differentiated four times), which the repair of the origin row already introduces as a
function and which the entry on the radial derivative rule in `PLAN.md` should be read
against. Whether `at_kcart` traces cleanly inside a
`lax.scan` body with `kappa` as the scanned argument, checked the way the eager-closure
trap is checked, by a second chunk that compiles nothing. And the error of the frozen
sphere, as the eigenvalues of $H(\mathbf k+\boldsymbol\kappa)$ on the sphere of
$\mathbf k$ against those of a sphere rebuilt at $\mathbf k+\boldsymbol\kappa$, at
$\kappa$ = 0.1, 0.3 and 0.5 bohr$^{-1}$.

### The propagator at a frozen potential

`defumat/realtime/` with three modules: `pulse.py` (the pulse shapes behind a registry,
returning $\boldsymbol\kappa$ on the time grid and at the midpoints), `propagators.py`
(the step behind a registry) and `propagate.py` (the driver, the current, the
diagnostics). The entry point is `workflows/realtime.py:run_realtime`, with
`Calculator.get_realtime(pulse, ...)` as its one-line delegation, returning a
`RealTimeResult` with the time grid, $\boldsymbol\kappa(t)$, $\mathbf E(t)$,
$\mathbf J(t)$, the energy, the norm drift and the number of excited electrons at the
end.

The first mode holds the Hartree and exchange-correlation potential at its ground-state
value, which is the independent-particle response and is what the published silicon
calculation found sufficient for its conditions (arXiv:1609.09298: the spectrum "does not
change" between the full evolution and the static ground-state potential, for one
material, the adiabatic LDA and one weak mid-infrared pulse, which is the scope of that
statement). It is also the mode in which the k-points are independent, which decides the
loop order in "Speed".

The step is a fourth-order Taylor expansion of
$\exp[-iH(t+\Delta t/2)\,\Delta t]$, four Hamiltonian applications, with the field at the
midpoint known in closed form. It is stable for $\Delta t\,\rho<2\sqrt2$, with $\rho$ the
spectral radius of $H$ after subtracting the centre of its spectrum, and the shift by
the centre roughly doubles the admissible step since the spectrum is one-sided. At
`ecutwfc = 30` this is about 0.36 Hartree units centred and 0.18 uncentred, so for the
soft norm-conserving datasets here the step of 0.08 to 0.1 that the published silicon
runs use is set by accuracy and not by stability. The driver computes the bound from
$(\sqrt{E_{cut}}+\kappa_{max})^2$ and the range of the local potential, and refuses a
step above it by name, since a Taylor instability is an exponential that looks like
physics for the first thousand steps.

Checks, in the order they become available: with no field the states are stationary,
$\mathbf J$ and the energy are constant to round-off, and the norm drift per step is
recorded as a function of $\Delta t$. With a field the work done equals the energy
gained, $\Delta E=-\Omega\int\mathbf J\cdot\mathbf E\,dt$ up to the sign convention,
which checks that the current is the $\boldsymbol\kappa$ derivative of the Hamiltonian the
step applies and that the step is unitary, and is blind to the time unit and to every
constant (see "The traps this work will meet"). And on silicon the even
harmonics vanish by inversion, which is a null result and is paired with AlAs, where
they must not: the guard has to be seen to fire.

### The linear response, against the code's own two routes

A step in $\boldsymbol\kappa$ at $t=0$ small enough to be linear, the Fourier transform
of $\mathbf J(t)$ with the window $e^{-\eta t}$, and $\epsilon(\omega)$ as Elk's task
481 computes it. The pairing of mode and reference matters and a wrong pairing reads as a
defect: the frozen potential is the independent-particle response and is compared with
the interband `optical_conductivity`, or with `run_absorption`'s
`epsilon_no_local_fields`, at the same $\eta$ and on the same full unshifted mesh, while
the run that updates the Hartree potential (a later stage) is compared with the `rpa`
kernel with local fields, and the one that also updates $v_{xc}$ with `alda`. Before
calling a difference a defect, read how each reference uses its broadening: the
real-time window is exactly the Kubo sum at the complex frequency $\omega+i\eta$
everywhere, prefactor included, and a reference that broadens a delta function or keeps
a real $\omega$ in the prefactor agrees only to order $\eta/\omega$. The review read
both: `optical_conductivity` (`response/conductivity.py:589`, `:871`) and
`independent_response` (`tddft/chi0.py:493`, `:645`) are analytic in
$z=\omega+i\eta$ with no real $\omega$ anywhere, so either is a target for an identity
and not only for an agreement, on one condition: both carry the truncated sum
$\Pi_N(z)-\Pi_N(0)$ while the real-time route has the exact diamagnetic term, so the
reference is run with every band on the sphere, `nbnd` equal to the number of plane
waves, on a cutoff chosen small for that purpose. A route identity is testable at any
cutoff. For a metal the Drude term matches only with `relaxation` set to $\eta$ and at
convergence in k. The external number is Elk's committed `Si-dielectric` spectrum.

### The high-harmonic workflow

`Calculator.get_hhg(pulse, ...)`, a delegation to `workflows/realtime.py:run_hhg`, which
runs the propagation and returns the spectrum $\omega^2|\mathbf J(\omega)|^2$ with a
stated window, the harmonic orders and their intensities, and the cutoff. The physics to
show in the notebook is the plateau and its cutoff against the field strength on
silicon, with odd harmonics only, next to AlAs with both parities. The comparison is
qualitative against arXiv:1609.09298 (a 28x28x28 mesh shifted four times there, far
beyond a notebook) and quantitative against Elk's `Si-ramp` current and the GaAs or AlAs
harmonic run with the scissor off, with the basis difference stated.

Two properties of a solid's spectrum should be written in the guide rather than
discovered. The spectrum of a single cell without dephasing is noisy above the gap and
converges slowly in k, and published work either adds a dephasing time of a few
femtoseconds or propagates the pulse through the sample (arXiv:1705.10707); we offer the
window and the k-mesh and say so. And a velocity-gauge propagation on a fixed mesh has no
structure-gauge problem, since each k-point evolves alone and no k-derivative of a Bloch
state appears.

### The perturbative orders, and the third harmonic by the real-time route

`workflows/realtime.py:run_harmonic_orders`, with `Calculator.get_third_harmonic` on top
of it: the adiabatic envelope at one frequency and one polarization, the propagation
wrapped in nested `jax.jvp` with respect to the amplitude, and the Fourier projection of
$e^{-n\eta t}J^{(n)}(t)$ over the last period. The tensor is rebuilt from a few
polarizations (two independent components, $xxxx$ and $xxyy$, in a cubic crystal, from
fields along [100] and [110]) and symmetrised with the rank-4 form of `symtensor3`,
which P36 wrote at any rank.

The validation is a ladder and the order is the point of it:

1. $J^{(1)}$ against the complex-frequency Kubo sum of the previous stage, which should
   agree to the time-step error, as (1,1) does in the model.
2. $J^{(2)}$ against `get_shg` on `alas-shg.in`. This cannot be done on silicon, where
   the second order is zero by inversion and agreement would be about a residue. It pins
   the factors of $1/2$ and $1/4$ of the cosine, $J=\partial_tP$, and the normalisation
   of $\chi^{(2)}$, which `NONLINEAR.md` records as ambiguous by a factor of two between
   papers. Three things keep this from being an identity and each is stated in the
   test. `second_harmonic` evaluates at $z=\omega-i\eta$ in both denominators
   (`response/shg.py:493`), which is the per-photon convention of the adiabatic
   envelope in the conjugate time convention, so the $(2,2)$ coefficient is compared
   after conjugation. It drops pairs closer than the smearing width (`shg.py:56`,
   `:289`). And it is a truncated sum over states where the real-time route has no band
   count, so the reference has to be converged in `nbnd`: the step from 22 to 23 bands
   in `alas-shg.in`'s header, -78.3 to -78.5 pm/V, is 0.26 per cent and is not a
   plateau. The tolerance is about 1 per cent, which pins a factor of two and nothing
   finer.
3. The digits at orders two and three come from the model's second route on the real
   Hamiltonian: at a small cutoff, the dense $H(\mathbf k)$ of `Hamiltonian.matrix(ik)`
   and its derivative matrices along the field, the hierarchy solved by dense linear
   algebra with every band, against the nested `jvp` of the propagation on the same
   cell. This is the check of the model carried to the code, and it is what would show
   the origin row if its repair were incomplete.
4. $J^{(3)}$ on silicon at 1.55 eV against arXiv:1810.06500, whose own TDDFT values
   (Table V) are $|\chi^{(3)}_{1111}(\omega)|=2.2\times10^{-18}$ and
   $|\chi^{(3)}_{1111}(3\omega)|=1.3\times10^{-18}$ m$^2$/V$^2$ in the LDA
   ($8.6\times10^{-19}$ and $1.4\times10^{-18}$ with TB-mBJ). *Corrected 2026-10-06
   (P135): this item first quoted 2.5e-18 and 3.0e-18, which are the table's
   optical-polarizability column and its $3\chi_{1122}(\omega)$ row.* That calculation updates
   the Hartree and exchange-correlation potentials and extracts from pulses, so it is a
   check of scale and not of digits, and at this frequency $3\omega$ is above the LDA
   gap, so the value depends on $\eta$. Their $\chi^{(3)}(\omega)$ is the $(3,1)$
   coefficient, the response at $\omega+3i\eta$ to two photons at $\omega+i\eta$ and one
   at $-\omega+i\eta$, and $\chi^{(3)}(3\omega)$ is the $(3,3)$ one.
5. The same order by a finite difference in the amplitude of four full runs, which
   shares the propagator and checks only the differentiation.

The cost of this route has to be said plainly, since it decides what comes next. One run
serves one frequency. Its length is $T=\ln(C/\varepsilon)/\eta$ for a transient of
relative size $\varepsilon$, where the model gives $C$ of about 7 for the $(3,3)$
component and 70 for $(2,0)$ (at $\eta T=10$ the errors are 2.6e-4 and 3.2e-3 against
$e^{-10}=4.5\times10^{-5}$), so 29 fs (12000 steps of 0.1) at $\eta=0.2$ eV and
$\varepsilon=10^{-3}$, and 58 fs at 0.1 eV. A smooth start does not shorten it: a
$\sin^2$ ramp over four periods made the $(3,3)$ error ten times larger below the gap
and three times smaller above it, and left $(2,0)$ unchanged. Third-order nested forward mode carries eight
copies of the state and about eight times the Hamiltonian applications, unless
`jax.experimental.jet` or a hand-written hierarchy brings that to four, which is a
measurement to take with the nested `jvp` as its test. On silicon with a 12x12x12 mesh
reduced by the little group of a [100] field this is $3\times10^{8}$ Fourier transform
pairs per frequency, affordable; a converged spectrum of thirty
frequencies on 16x16x16 is not, on a CPU. The real-time route is therefore the one that
needs no new solver and the one every other route is checked against, and it is not the
one that produces a spectrum.

### The third harmonic as a spectrum: the frequency-domain hierarchy

The model's second route is the production one. At order $n$ and harmonic $m$ the
steady-state component $c^{(n)}_m$ of each occupied state solves

$$\big(\epsilon_{n\mathbf k}+m\omega+in\eta-H_0\big)\,c^{(n)}_m
=\sum_{p\ge1}\frac{1}{p!\,2^p}\sum_s\binom{p}{s}\,h_p\,c^{(n-p)}_{m-(2s-p)},
\qquad h_p=\frac{\partial^pH}{\partial k^p}\ \text{along the field},$$

and $J^{(3)}_{m=3}$ is a sum of matrix elements $c^\dagger h_{p+1}c$ over the components
of total order three, with the same weights $\binom ps/(p!\,2^p)$ and the selection
$M=-m_1+(2s-p)+m_2$; `freq_components` in `tools/realtime/toy_orders.py` is the
statement of both and the review checked the equation above against it. Per frequency
the third harmonic takes six solves per band and per k-point ($c^{(1)}_{\pm1}$,
$c^{(2)}_{\pm2}$, $c^{(3)}_{\pm3}$; $c_{-m}$ is not the conjugate of $c_m$), and the
$(3,1)$ component, the intensity-dependent index, takes nine, with $c^{(2)}_0$ and
$c^{(3)}_{\pm1}$ added. The real-time run costs about $4\times10^{5}$ applications per
band and k-point, so the hierarchy is cheaper by the ratio of that to six or nine times
the iteration count of the solver, which is not known: an indefinite shifted solve at
$\eta=0.1$ eV on a spectrum 400 eV wide has not been timed here. The operators $h_3$ and $h_4$ are purely
nonlocal, since the kinetic energy is quadratic, and are two more `jvp` of the function
`apply_second` already differentiates twice.

What it needs is the solver `NONLINEAR.md` has named as the missing machine since P35: a
linear solve with $H-\epsilon-z$ for complex $z$, which is indefinite and not Hermitian,
where the projected conjugate gradients of the Sternheimer stack do not apply. With
$\eta>0$ the operator is normal and its smallest singular value is $n\eta$, so a shifted
Krylov method with the kinetic preconditioner is the candidate (COCG or BiCGStab; QE's `solve_e_fpol.f90` is
the nearest reference in the vendored tree and should be read first). Building it also
opens the three items that file lists behind the same solver, the truncation-free shift
current, $\alpha(\omega)$ and resonant Raman, which is an argument for doing it as its
own phase. Its check is the real-time route at three frequencies, below, at and above
the two-photon and three-photon resonances.

Two things are ill-conditioned here that the real-time trace does not suffer from, and
both are to be designed for and not discovered. $c^{(2)}_0$ always has a component
$1/(2i\eta)$ along the state itself, the secular term of the Stark shift, which is the
worst component in the model. And with several occupied bands each band's solve carries
denominators $\epsilon_n+m\omega-\epsilon_{n'}$ between occupied states that cancel only
in the sum over bands, so the single-band right-hand sides are large where the answer
is not; projecting the occupied subspace out of the right-hand side, as the Sternheimer
stack does, is the usual cure and needs its own derivation at second and third order.

### Updating the potential in time

The Hartree and exchange-correlation potentials rebuilt from $\rho(t)$ at every step with
the driver's own `density` and `potential`, and a predictor-corrector for the midpoint
(extrapolate the potential, step, rebuild, step again), which doubles the cost of a step.
The checks are the two remaining pairings of the linear stage, `rpa` with local fields
and `alda`, and energy conservation with the field off after a kick, which the frozen
mode satisfies trivially and this one does not. The perturbative orders need no new code
here, since the tangent of the density goes through `v_of_rho` as it does in the
Sternheimer stack. For bulk silicon the published result is that this changes little, so
it is where local fields matter, a layered or a low-dimensional cell, that it earns its
cost.

### The little group of the field

Reduce the k-set with the operations that leave the polarization invariant, with time
reversal off, and symmetrise $\mathbf J$ as a polar vector and $\rho(t)$ with the same
subgroup. A field along [100] in silicon keeps eight operations, which is the difference
between a notebook that fits in ten minutes and one that does not, so this may have to
move ahead of the notebook. The traps are the two the linear-response work already
found: a shifted Monkhorst-Pack mesh is not closed under the group and is refused, and a
wedge is detected with `is_reduced` and not from the weights.

### What is left for later, each with what it needs first

- Ultrasoft and PAW, collinear `nspin = 2` and spinors: **done at a frozen potential**
  (`PLAN.md` P139 to P141, 2026-10-07), and with the potential updated for `nspin = 2` and
  spinors. The term the moving overlap adds is the one derived below, `P_kappa = E.X`, now
  checked rather than only derived: against the Kubo sum with P99's generalised velocity at
  first order, and the hierarchy against the propagation above it. What is left of the
  item is the potential updated for an ultrasoft or PAW dataset (the augmentation charge of
  `rho(t)` from the projections at `k + kappa(t)`, `newd`'s `D(t)` every step, PAW's
  one-centre `D(t)`), and laser-driven spin dynamics against Elk's `Ni-laser-pulse`, for
  which the spinor route with the potential updated is the machinery.
- Re-centring the sphere when $\boldsymbol\kappa$ crosses a reciprocal lattice vector's
  half, which is a shift of Miller index of the kind P16 does at the zone edge, for
  mid-infrared pulses whose $\kappa_0$ approaches the zone size.
- A split-operator step, see "Speed".
- The self-consistent hierarchy in the frequency domain, which adds the density response
  at $2\omega$ and $3\omega$.
- A sum over states for $\chi^{(3)}$. We do not recommend it: it needs the same $h_3$
  and $h_4$, it inherits the truncation the shift current measured, and the degeneracy
  handling of rule D4 enters three times.

## What is refused, by name, in the first version

*Updated 2026-10-07 (P139 to P141): the first two items below are no longer refused at a
frozen potential. The derivation is kept because it is the one the code follows, with the
signs checked: `X = sum |b_i>[d_ij <b_j| + i q_ij <db_j/dk|]` is `i` times P99's connection,
`X - X^dag = i dS/dk` holds to 5e-11 against a finite difference of `S`, and the current
`<dH/dk> - 2 Im <H S^-1 X>` is P99's generalised velocity between eigenstates.*

- Ultrasoft and PAW datasets. With the projectors at
  $\mathbf k+\boldsymbol\kappa(t)$ the overlap $S$ depends on time, and the equation of
  motion gains a term,
  $i\,S_\kappa\partial_t\psi=(H_\kappa+P_\kappa)\psi$ with
  $P_\kappa=\mathbf E(t)\cdot\sum_{ij}|\beta_i\rangle\,\big[\mathbf d_{ij}\langle\beta_j|-q_{ij}\langle\beta_j|(\mathbf r-\mathbf R)\big]$,
  where $\mathbf d_{ij}$ is the dipole of the augmentation function about its atom and
  $(\mathbf r-\mathbf R)$ stands to the right of the bra and acts on the state, so that
  $\langle\beta_j|(\mathbf r-\mathbf R)$ is $i\,\partial_{\boldsymbol\kappa}\langle\beta_j|$.
  The overall signs were not checked when this was written; they are now (above). The
  published form is Qian, Li, Lin and Yip's moving-ion term, PRB 73, 035408 (2006),
  arXiv:cond-mat/0510643, Eqs. 19 to 22, `P = -i T^dag dT/dt`, with the field's
  `kappa` for the ions' coordinates; GPAW's (arXiv:1109.6157, Eqs. 49 to 51) is the same
  construction. Abinit's real-time PAW (arXiv:2507.08578, and its `src/80_rttddft` read on
  2026-10-07) rebuilds the projectors and `S^-1` at `k + A` but carries no `dS/dt` term.
- A symmetry-reduced or shifted k-set, until the little-group stage.
- A spin spiral and gamma-only storage, which `at_kcart` already refuses.
- DFT+U with the potential updated in time. At a frozen potential and a fixed
  occupation matrix the machinery exists, since `at_kcart` rebuilds `wfcU` at the traced
  k-point (`driver.py:4208`) and the velocity operator carries the Hubbard term, so that
  mode is refused in the first version only until its linear check is run.
- Collinear `nspin = 2` and spinors: no longer refused (P139, P140).
- `occupations = 'fixed'` cutting a degenerate multiplet, by the diagnosis that already
  exists: the weights then differ inside a multiplet and $\mathbf J$ depends on the
  rotation the eigensolver returned.
- A potential-only meta-GGA with the potential updated in time, since the kinetic-energy
  density is not gauge invariant under $\boldsymbol\kappa$ and needs the current
  correction $\tau-|\mathbf j|^2/\rho$. With the potential frozen, `tb09` is allowed and
  is the way to a correct gap, where published work finds that TB-mBJ lowers silicon's
  $\chi^{(3)}$ by a factor of 2 to 3 through the gap alone.
- A scissor. A rigid shift of the empty states is an operator built from the projector on
  the occupied states at $\mathbf k+\boldsymbol\kappa(t)$, which a propagation on the
  full sphere does not have. `tb09` at a frozen potential takes its place.
- A magnetic field or a constrained moment, for the reason the response stack gives.
- `DEFUMAT_POOLS` with the potential updated in time, in the first version. The frozen
  mode needs no communication between k-points, which is an argument that pools are
  simple there and has not been checked against `parallel.py`.
- A metal is allowed in the frozen mode with its ground-state occupations held fixed,
  and the guide says that the intraband response then converges slowly in k.

## The frozen sphere under a large shift

The sphere contains the plane waves with $|\mathbf k+\mathbf G|^2\le E_{cut}$ while the
state at time $t$ has kinetic energy $|\mathbf k+\mathbf G+\boldsymbol\kappa|^2$, so the
basis is a sphere displaced from where the state lives and its effective cutoff along
the field is $(\sqrt{E_{cut}}-\kappa_{max})^2$: 11.2 Ry instead of 12 for the published
silicon pulse, 25.6 instead of 30 for the diamond one. This is the frozen-sphere
variational error of P48, reached here by the field instead of by a stencil. The driver
prints the effective cutoff before it starts and the user converges `ecutwfc` at
$\kappa_{max}$, and the measurement of the first stage says how large the error is. The
structural fix is the re-centring listed above.

## Memory

Nothing is differentiated in reverse through time, so there is no tape: the current is
one small gradient with respect to $\boldsymbol\kappa$ per step, and the perturbative
orders are forward mode. The working set is therefore the state and a few copies of what
is in flight:

$$\text{bytes}\approx16\,n_s\,n_k\,n_{occ}\,n_{pol}\,n_{pwx}\times(1\ \text{or}\ 8)
\;+\;\text{in flight}\times\big[\,3\times16\,n_{occ}\,n_{pol}\,n_{pwx}+16\,n_{pwx}n_{kb}+\text{FFT boxes}\,\big],$$

where the factor 8 is third-order nested forward mode, the three copies are the Taylor
temporaries (the review counts about five), and $n_{pwx}n_{kb}$ is `vkb` at the
displaced k-point, rebuilt per step and never stored for the whole mesh. The direct
radial transform adds an intermediate of $8\,n_\beta\,n_{pwx}\,n_r$ bytes per k-point in
flight, times 8 at third order, which the table of $g_l(q^2)$ removes. And the current
is kept as the weighted sum over each chunk: $\mathbf J_{\mathbf k}(t)$ per k-point on
16x16x16 over 28500 steps is 2.8 GB of host memory for nothing. Only occupied bands are carried, which is where the
route is cheaper than Elk's `nstsv` squared per k-point.

| cell | mesh | `ecutwfc` | $n_{pwx}$ | $n_{occ}$ | store | at third order |
|---|---|---|---|---|---|---|
| Si, 2 atoms | 16x16x16 | 12 | 186 | 4 | 49 MB | 0.4 GB |
| Si, 2 atoms | 16x16x16 | 30 | 736 | 4 | 0.19 GB | 1.5 GB |
| Si, 2 atoms | 28x28x28 | 30 | 736 | 4 | 1.0 GB | 8.3 GB |
| Si, 2 atoms | 28x28x28, four shifts | 30 | 736 | 4 | 4.1 GB | 33 GB |
| Si, 16 atoms | 8x8x8 | 30 | 5890 | 32 | 1.5 GB | 12 GB |
| MoS$_2$ monolayer, spinor | 36x36 | 40 | 3740 | 26 | 4.0 GB | 32 GB |
| MoS$_2$ monolayer, spinor | 72x72 | 40 | 3740 | 26 | 16 GB | 129 GB |

The plane-wave counts are $\Omega k_c^3/6\pi^2$ estimates and `sizing.estimate_size`
should replace them; the meshes are full, before any reduction by the little group.

Two design consequences. In the frozen mode the k-points never talk to each other, so the
loop is k outside and time inside: a chunk of k-points is propagated through the whole
pulse and only its weighted contribution to $\mathbf J(t)$ is kept, the peak is one chunk whatever the
mesh, and the ground states can come from a streamed store. With the potential updated
the loop is time outside and k inside, the whole mesh must be resident on the device
(streaming the store through a card at every one of 30000 steps is not an option), and
`sizing` has to refuse in advance when it is not. And a checkpoint of the states every
so many steps is part of the design and not an afterthought, since a 60 fs pulse on a
production mesh outlives a queue limit; P67's checkpointing is the precedent, and in the
frozen mode the natural unit of restart is a finished k-chunk.

## Speed

The count that sets the cost is Fourier transform pairs: one per band per Hamiltonian
application, four applications per Taylor step. A 25 fs pulse with a $\sin^2$ envelope
spans about 69 fs, 28500 steps of 0.1, so silicon on 16x16x16 is $6.6\times10^{4}$ pairs
per step and $1.9\times10^{9}$ for the run, and on 28x28x28 it is $10^{10}$. What a pair
costs on this code is the first stage's measurement and not assumed here.

In order of what we expect each to be worth:

- The time loop is a `lax.scan` over a chunk of steps with $\boldsymbol\kappa$ at the
  steps and the midpoints as the scanned arrays, so one executable serves every step and
  every pulse, and $\mathbf J$ comes back once per chunk. A Python loop over a compiled
  step with a host read per step pays the dispatch 30000 times. The chunk function is
  jitted with its shapes static and its arrays as arguments, and the test is a second
  chunk that compiles nothing (the eager-closure trap; on Triton the cap on mappings
  makes it fatal rather than slow). The structure is a Python loop over k-chunks, inside
  it a Python loop over chunks of steps, and one `compiled_function` whose scan body
  maps the k-chunk. `map_k` is not the outer loop: with a scan inside the mapped
  function it would stack `(nk, nt, 3)` and run the whole mesh and the whole pulse as
  one executable, which leaves no place for a checkpoint.
- `band_batch` and `band_precision` are the existing dials for how many bands go through
  the box at once and in what precision, and the step goes through them and through the
  stick-based transform as every Hamiltonian application does.
- In the frozen mode, k outside and time inside keeps a core's working set at a few
  bands of one k-point, which is the regime "One calculation on many cores" found a CPU
  to be fast in, and `DEFUMAT_POOLS` then needs no communication until the final sum. On
  a card the k-chunk is `'fit'`.
- The little group divides everything by up to eight.
- `vkb` at $\mathbf k+\boldsymbol\kappa(t)$ is rebuilt at every step. For a linearly
  polarized field $\boldsymbol\kappa$ moves on a line, so the projectors could be
  tabulated on that line per k-point and interpolated, at $n_k n_{pwx}n_{kb}$ times the
  number of nodes in memory. The one-dimensional table of $g_l(q^2)$ per dataset does
  the same job for any polarization at a negligible size and is the one to try first.
- A split-operator step. The kinetic term is diagonal in $\mathbf G$ at any
  $\boldsymbol\kappa$, the local potential is diagonal in real space, and the exponential
  of the separable nonlocal term is a small matrix per atom, so a symmetric
  Suzuki-Trotter step costs one transform pair per band instead of four and is exactly
  unitary. Its error is a commutator of the kinetic and the nonlocal terms and has to be
  measured against the Taylor step on $\mathbf J(t)$ before it can be a default; it does
  not extend to an overlap operator. It goes in the propagator registry as a candidate.
- Implicit steps with a parallel-transport gauge take steps of 10 to 100 as at 5 to 13
  applications each (arXiv:1805.10575), measured on molecules at cutoffs of 20 to 30
  Hartree where an explicit step is below 1 as. At the cutoffs of the datasets here the
  explicit step is already 2 as, so the gain is at most a few and this is not planned.
- Single precision. The propagation in `complex64` halves the memory and suits a card
  such as D22's, which runs double precision at 1/70 of single. A harmonic spectrum
  spans ten orders of magnitude in intensity and the phase error accumulates over 30000
  steps, so this is a measurement (the spectrum in both precisions on one cell, to which
  harmonic order they agree) and never the mode a number is claimed in.
- The first call compiles and, on a card, autotunes; the timing rules of `CLAUDE.md`
  apply, and a card job points `DEFUMAT_CACHE_DIR` at a directory that outlives it.

## The traps this work will meet

From the list in `CLAUDE.md`, the ones with a known site here:

- `abs` at a forced zero: the norm and the density inside the step are
  `Re(conj(psi) psi)`, and the tangent of the adiabatic run passes through $\lambda=0$,
  where every first-order quantity vanishes exactly.
- A quantity symmetry forces to zero is a residue: the second order on silicon, and any
  component of $\chi^{(3)}$ a cubic crystal forbids. The even-order checks are on AlAs.
- A caller-built k-set is a `for_spin` boundary, and the full mesh here is caller-built.
- Rule D4 does not apply to the propagation, which never resolves a band inside a
  multiplet: the occupied subspace is propagated as a whole and $\mathbf J$ is a trace
  over it. It does apply to any diagnostic that projects on individual ground-state
  bands, such as a band-resolved excitation.
- A test asserting a tolerance its step does not deliver: every assertion on $\mathbf J$
  states its $\Delta t$, its $\eta T$ and the ground state's `conv_thr`, since the
  ground state's residual is a perturbation that oscillates at interband frequencies for
  the whole run.
- A check that cannot see what it is cited for: the energy balance
  $dE/dt=\sum w\langle\psi|\partial_{\boldsymbol\kappa}H|\psi\rangle\cdot\dot{\boldsymbol\kappa}$
  holds for any unitary evolution under a Hamiltonian that commutes with the one in the
  energy, so it is blind to a wrong time unit (a step under $2H$ satisfies it) and to
  any constant on $\mathbf J$ or $\mathbf E$. What it does see is a current that is not
  the $\boldsymbol\kappa$ derivative of the Hamiltonian in the step, a dropped nonlocal
  term for instance, and a step that is not unitary. The time unit and the constants are
  pinned by the linear and second-order comparisons and by nothing else.
- The energy can be right while its derivative is wrong: with the potential updated the
  Hamiltonian applied in the step must be the functional derivative of the energy that
  is monitored, which a potential-only meta-GGA breaks by construction.

## What a finished phase leaves behind

For each of the two phases, the five things `CLAUDE.md` asks for: the numbers of the
ladder in `PLAN.md` §3; a README row for the real-time response and the high-harmonic
spectrum (Elk ticked for tasks 460 and 480, QE to be checked in the vendored tree, where
`TDDFPT/` is Liouville-Lanczos and not a propagation) and one for the third harmonic
(expected blank in both, to be confirmed by a search of both sources); the entries in
`docs/features.tex` with an executed snippet and the refusals above in the amber box;
a notebook on silicon with the harmonic spectrum as its figure, under the ten-minute
ceiling, which the frozen mode and the little group are sized for; and a timing against
Elk's task 460 on the `Si-dielectric` input, one core each, from a converged ground
state on both sides, with the statement that Elk's step diagonalises a matrix in a band
basis and ours applies the Hamiltonian in plane waves, and that the comparison is per
femtosecond at each code's converged step.

Edits elsewhere that go with it: `NONLINEAR.md`'s statement that $\chi^{(3)}$ needs a
second-order wavefunction is a statement about the Sternheimer stack and gets a dated
correction of the kind that file already carries for the shift current;
`ELK-FEATURES.md` has no rows for tasks 450 to 481; and the README's "Not yet" loses
real-time propagation. Before any module is written the plan goes to a review by a
subagent with the decisive files named, as every phase-sized feature here does.

## Decisions

Taken by the user on 2026-10-06: the third harmonic is done by both routes, as two
phases. The real-time route comes first, since it needs no new solver and is the check
of the second, and the frequency-domain hierarchy with its complex shifted solver
follows as its own phase.

The two smaller ones were left to the plan's own recommendation the same day:

1. The first phase ships at a frozen potential. Every check of the ladder is available
   there, and the update of the potential in time follows as its own stage with its own
   two pairings (`rpa` with local fields, `alda`).
2. The little group of the field goes ahead of the notebook, since the full mesh is at
   the edge of the ten-minute ceiling by the count of transform pairs. The one case
   that reverses this is a first-stage timing that puts the full-mesh notebook under
   five minutes, in which case the notebook comes first and the reduction after it.

## What was read, and what could not be verified

Read in full or in an ar5iv rendering: arXiv:1609.09298 and arXiv:1706.02890 (high
harmonics in Si and MgO, Octopus), arXiv:1705.10707 (diamond), arXiv:1810.06500
($\chi^{(3)}$ of Si, diamond and quartz from real-time pulses), arXiv:1710.08573
(velocity gauge with a nonlocal pseudopotential in an atomic-orbital basis),
arXiv:1309.4012 (real-time second and third harmonics in the length gauge with a
dephasing and a Fourier fit over the last period), arXiv:cond-mat/0510643 (real-time
ultrasoft), arXiv:1805.10575 (parallel transport), arXiv:1412.0996 (Elk's propagation),
the real-time paragraph of arXiv:2507.08578, and Elk's source and examples.

Not verified, and not to be leaned on:

- the velocity-gauge equation for an overlap operator, which is a derivation and has no
  source;
- any reference value for the static $\chi^{(3)}$ of silicon;
- the propagator and the step of the Octopus runs, which their papers do not give;
- the stability bound $2\sqrt2$, which is the imaginary-axis interval of a fourth-order
  explicit scheme and was not taken from a paper;
- the practice of single precision on cards in other codes;
- whether Elk has been used for a published high-harmonic calculation in a solid: one
  search found none.
- the split of a step's cost between the Hamiltonian applications and the projector
  rebuild, which is an operation count and was not timed;
- the iteration count of a complex shifted solve, and with it the ratio between the two
  routes to the third harmonic;
- whether the frozen mode runs under `DEFUMAT_POOLS` as it stands;
- whether Elk's task 125 assembles the complete $\chi^{(2)}$ as a function of the
  complex frequency, which matters only if it is used beside `get_shg` in the ladder.

Cited above from their abstracts only: arXiv:1710.01300, arXiv:1712.04924,
arXiv:1703.07796, arXiv:1601.01201 (a sum over states for silicon's third harmonic in
LAPW, the nearest published perturbative reference).
