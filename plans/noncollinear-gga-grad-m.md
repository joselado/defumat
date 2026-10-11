# A noncollinear GGA built from `grad m` (the cure for `OPEN.md` Part XXI item 1's open tail)

*Written 2026-10-11, after the diagnosis in `OPEN.md` Part XXI item 1 ("Diagnosed
2026-10-11") and a literature check. Nothing here is measured yet. The calculations go to
D22-0161; the code is developed on the workstation.*

## Why

`_noncollinear_gradient_correction` (`defumat/scf/potential.py`) is `gradcorr`'s
`nspin = 4` branch: it builds `rho_up/dw = (n +- s|m|)/2` and runs the collinear GGA on
that, with `s = sign(m . u)` along a fixed axis when every starting moment is parallel
(`fixed_quantization_axis`, QE's `lsign`) and `s = 1` otherwise. A spiral never has an
axis, so it takes `|m|`, whose gradient has a kink at every node of `m`. Measured: the
PBE ferromagnet at `q = 0` stalls at 3.1e-8 without the axis and converges in 26 with it;
the spiral converges to 1e-10 and no further; the potential is the exact derivative of
the energy (3.8e-15), so the defect is the functional, not the code.

## What the literature has (subagent report, 2026-10-11)

* **Scalmani-Frisch** (JCTC 8, 2193 (2012), quoted second-hand from arXiv:2606.31482):
  invariants from `grad m` -- `g = |grad n|^2`, `Gamma = sum_k |grad m_k|^2`,
  `a_k = grad n . grad m_k`, `A = |a|`, `t = a . m`, `s = |m|` -- with
  `sigma_up/dw = (g + Gamma +- 2 f A)/4`, `f = sgn(t)`. Exact collinear limit, invariant
  under a joint spin rotation, but `f` jumps where `t` changes sign at `s, A != 0`, so the
  energy is discontinuous off the collinear limit.
* **Multicollinear** (Pu et al., arXiv:2110.09897; NCXC, arXiv:2504.15024): smooth by
  construction and exact in the collinear limit, but one collinear evaluation per
  direction on a Lebedev sphere (about 1000), which breaks rotational invariance at about
  1e-12 Ha, the scale we chase. Never run in a plane-wave code.
* **Elk** (`potxcmt.f90:131-137`, `potxcir.f90:89-94`): `sqrt(m.m + dncgga)` with
  `dncgga = 1e-8`, gradients of the channel densities, so the same `grad |m|` regularised;
  `ncgga` is obsolete. **QE 7.5** has only `compute_rho`. VASP's `GGA_COMPAT` is unrelated.
* **Peralta-Scuseria-Frisch** (PRB 75, 125119 (2007)), Eich-Gross (PRL 111, 156401):
  singular at `m = 0` or a different functional; not candidates.

**A fact that reframes the goal.** A collinear spin GGA is unchanged under
`(m_z, grad m_z) -> -(m_z, grad m_z)` (the swap of up and down), so the unsigned `|m|`
already gives the collinear *energy* exactly, nodes included. What it breaks is the
transverse direction: near a node `|m| ~ |m_par| + m_perp^2/(2|m_par|)`, so the second
derivative in a transverse direction diverges there, which is what the SCF linearises.
And on the grid, `grad |m|` is the FFT gradient of a field that is not band-limited,
while `grad m_k` from `m(G)` is.

## The construction

Scalmani-Frisch's invariants with the sign replaced by a cosine weight (derived by the
subagent and checked here; not found published):

    c = t / (s A)  in [-1, 1]
    E_x = 1/4 sum_{sigma, tau = +-1} (1 + sigma tau c) e_x(n + sigma s, g + 2 tau A + Gamma)

with `e_x(rho, sigma)` the unpolarized exchange energy density (spin scaling, as
`_spin_exchange_energy` does it). Correlation is unchanged: it depends on `n`,
`zeta = s/n` and `|grad n|^2` only, and is even and analytic in `zeta`.

* **Collinear limit, nodes included.** When `a` is parallel to `m`, `c = +-1`, two of the
  four terms drop and the sum is `1/2 sum_sigma e_x(n + sigma m_z, g + 2 sigma b + Gamma)`
  with `b = grad n . grad m_z`, which is the collinear exchange exactly (checked by hand:
  `sigma s c A = sigma sign(m_z) b` and the channel relabelling).
* **Smooth.** Write `F(s, A) = 1/2 sum_sigma e_x(n + sigma s, g + 2 sigma A + Gamma)`; then
  `E_x = F_even + c F_odd` with `F_even, F_odd` its even and odd parts in `A`. `F` is
  invariant under `(s, A) -> (-s, -A)`, so `F_odd` is odd in `s` and in `A`, meaning that
  `F_odd = s A H(s^2, A^2)` and `c F_odd = t H`. Everything is a smooth function of
  `s^2`, `A^2` and `t`, which are polynomials in `m`, `grad n` and `grad m`.
* **Axis-free.** `g, Gamma, A, t, s` are invariant under a joint rotation of `m` and
  `grad m` in spin space, so no axis, no global torque.
* **Admissible gradients.** `A <= |grad n| sqrt(Gamma)`, so `g - 2A + Gamma >= 0`.

**What it is not.** It is not `pw.x`'s functional off the collinear limit; a spiral's
energy moves by the difference between `grad |m|` and this, which is a new number, not a
reproduction. So it is an option, not a replacement.

## Implementation

1. `Functional.noncollinear_exchange_energy(n, s, g, Gamma, A, t)` (or the sigma pair and
   the weights) in `defumat/xc/functional.py`: `_spin_exchange_energy`'s gates and spin
   scaling on a per-channel `sigma` given directly, evaluated on the two `tau` branches
   and mixed with `(1 +- c)/2`. `c` from `jnp.where(s*A > eta, t / (s*A), 0)`, with `A`
   and `s` through `safe_modulus` so that no derivative is `0/0`.
2. `_noncollinear_gradient_correction_grad_m` in `defumat/scf/potential.py`: transforms
   `n + core` and the three `m_k` (4 forward), takes `grad n` and `grad m_k` (12 back),
   builds the invariants, and gets **the potential as `jax.value_and_grad` of the
   integrated energy with respect to `rho_r`**, times `N/Omega`. Every term (the
   divergence included) then comes from the same expression, as M1 checked for the
   present branch, and second order for the response stack is autodiff.
3. Selection: a static `Calculation`/`System` field, `noncollinear_gga`, with
   `'kubler'` (today's, `pw.x`'s, the default) and `'grad_m'`; an input key in
   `&system` read by `builder.py`, a keyword through `run_scf`/`Calculator`, as
   `nosource` is plumbed. Under `'grad_m'` the quantization axis is ignored (it is
   irrelevant: the collinear limit is exact without it).
4. Refused by name under `'grad_m'` until checked: the meta-GGA branch
   (`_noncollinear_meta_exchange`), the PAW one-centre GGA (`defumat/paw/gradient.py`,
   which has its own `|m|` rotation), `nosource`, and the ultracell. A PAW spiral with
   `'grad_m'` would otherwise mix two functionals.

## Validation, each against something that shares no machinery with it

* **V1, collinear with a node.** A collinear field `m = m_z(r) e` with sign changes, `e`
  along `z` and along a tilted direction: the `'grad_m'` energy equals the `nspin = 2`
  `gradient_correction` energy on `((n+m_z)/2, (n-m_z)/2)` to round-off, and its
  potential's longitudinal part equals the collinear `(v_up - v_dw)/2`.
* **V2, rotation.** A random noncollinear field and a random SO(3) rotation applied to
  `m`: energy equal to round-off, potential rotated with it.
* **V3, smoothness.** Energy along `m(lambda) = m0 + lambda dm` through a node and through
  a point where `t` changes sign: second finite difference bounded, where the
  `'kubler'` unsigned branch's diverges.
* **V4, the potential.** Finite difference of the energy against `v` on two random
  directions, and the guard test: a field that makes `s A < eta` somewhere, with a
  non-`NaN` gradient and Hessian-vector product.
* **V5, SCF.** On D22: the `q = 0` cobalt ferromagnet without an axis converges (where
  `'kubler'` stalls at 3.1e-8); its energy equals the signed-axis `'kubler'` energy to
  the convergence threshold, since the converged state is collinear; and the spiral on
  the laboratory sphere reaches `conv_thr = 1e-11`. Iteration counts against LDA's 12.
* **V6, reference.** Optionally a multicollinear energy (Lebedev, a few hundred
  directions) on one frozen noncollinear density, as an independent number for the
  off-collinear part. Not a gate.

## Deliverables

A phase in `PLAN.md` §3 with the numbers above, a row in the README (QE no, Elk no), an
entry in `docs/features.tex` with its amber box, `OPEN.md` Part XXI item 1 closed or
reworded, and the notebook rule: extend the spin-spiral notebook with the PBE helix
converging under `'grad_m'`. The timing pair against `pw.x` is the cost of the potential
call (13 transforms against 8), not an SCF against `pw.x` (which does not have it).

## Open for the reviewer

* The default: `'kubler'` everywhere (matches `pw.x`), or `'grad_m'` for a spiral only
  (where `'kubler'` has no axis and `pw.x` has the same stall).
* The guard `eta` and whether a Taylor form below it is needed.
* Whether V6 is worth it.
