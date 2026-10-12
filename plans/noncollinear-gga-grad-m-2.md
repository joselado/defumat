# `noncollinear_gga = 'grad_m'` as the default, on PAW, on meta-GGAs and on every spiral route

*Written 2026-10-11, after P142 (`plans/noncollinear-gga-grad-m.md`). The user asked for: the
`grad m` functional as the default; it working for PAW and for the (potential-only) meta-GGAs;
it working for spirals; and the tests redone under the new default. Nothing below is measured
yet. Calculations on D22-0161.*

## A. The default, and `dE/dq` (being implemented now, not for review unless something is wrong)

`System.noncollinear_gga` defaults to `None`, resolved in `Calculation`: `'grad_m'` wherever
it applies (an `nspin_mag = 4` run with a gradient functional), `'kubler'` otherwise; an
explicit choice is checked as today. Once B and C land, nothing falls back.

`dE/dq` (`forces/spiral.py`): under `'grad_m'` the exchange depends on `q` through the
laboratory gradient `i(G - q)`. `at_spiral_q(rebuild_basis=False)` will carry a traced
`spiral_q_crystal` that `Calculation._spiral_q_argument` reads (the laboratory sphere frozen
with the basis), so the single-pass route and the augmented split route (`q_global_energy`)
differentiate it; the norm-conserving chunked route, which treats Hartree + XC as constant,
gets one extra `grad` of `q_global_energy` in `q` at the frozen density. Check: `dE/dq` against
a central difference of the frozen-state energy, and against the SCF energy scan.

## B. PAW one-centre (`defumat/paw/gradient.py:_noncollinear_gradient`)

Today: rotate onto `(n +- s|m|)/2` per quadrature direction and radius, re-project the channels
onto multipoles, collinear `spin_gradient_terms_and_energy`, `v1 - div h`, rotate back.

Proposal: the sphere carries `rho_lm` for all four components, and `m_k` is **linear** in them,
so `grad n` and `grad m_k` come from `_gradient(rho_lm[c], density[c], paw)` per component with
no re-projection (the kink is never formed). The physical spherical components `(r, phi, theta)`
are orthonormal, so `g`, `Gamma`, `a_k`, `A`, `t` are the same dot products. A new
`Functional.noncollinear_gradient_terms_and_energy(n, m, grad_n, grad_m)` returns
`(v1 (4,...), h (4,3,...), e)` by `value_and_grad` of the pointwise energy density (exchange by
`noncollinear_gradient_exchange_energy`, correlation by `spin_gradient_correlation_energy`);
then `h *= r2`, the theta component divided by `sin_theta`, `v_lm = v1` projected and
`v_lm - div h_lm` per component, exactly the collinear branch's assembly on four components.
The grid branch could use the same method (an explicit `v1 - div h` in place of the autodiff of
the whole integral); not required.

`with_small` (fully relativistic PAW, `add_small_mag`): the radial vector potential for the
small component's `ddd` share is `v_m` on the radial grid, `ylm . (v1 - div h)[1:]`.

**PAW on a spiral.** The sphere's magnetization is also in the rotating frame. The laboratory
field near atom `a` is `R(-q.(tau_a + r)) m'`, so the covariant gradient is
`grad m'_x + q m'_y`, `grad m'_y - q m'_x`, with `q` resolved on each direction's local
`(r-hat, phi-hat, theta-hat)` (`paw.angular.directions` gives `r-hat`; `phi-hat`, `theta-hat`
from it). The constant `tau_a` phase is a global rotation and drops out of every invariant.
`q` (1/bohr) has to reach `onecenter`; it is not there today.

Validation: (i) a collinear PAW density with nodes, tilted, against the `nspin = 2` sphere
branch, energy and `ddd`; (ii) a random spin rotation; (iii) finite difference of the one-centre
energy in `becsum`; (iv) PAW SCF `fe-noncolin` or nickel against the signed `'kubler'` (collinear
ground state, equal to the convergence threshold); (v) a PAW spiral against its unfolded
supercell at one density (does a PAW spiral/supercell pair exist? `co-helix4` is ultrasoft).

## C. Meta-GGA, potential-only (TB09, BJ06), `_noncollinear_meta_exchange`

This is the open design question. There is no energy, so "the derivative of a smooth
functional" is not available: the potential is the object, built per channel from
`(rho_s, |grad rho_s|^2, lap rho_s, tau_s)`. Today the channels are
`(n +- s|m|)/2` with gradient and Laplacian by FFT of the kinked field and
`tau_s = (tau_0 +- s tau.m-hat)/2`, and `B = s (v_up - v_dw)/2 m-hat`.

The collinear identity we want: for `m = m_z u`, the channel ingredients are
`(n +- m_z)/2`, `(g + Gamma +- 2b)/4`, `(lap n +- lap m_z)/2`, `(tau_0 +- tau_z)/2`, and
`B = (v_+ - v_-)/2 u`.

Candidates:

* **C1, cosine-mixed branches.** Two branches `f = +-1` weighted `(1 + f c)/2`, `c = t/(sA)`;
  branch `f`: `rho_s = (n +- s)/2`, `sigma_s = (g + Gamma +- 2 f A)/4`,
  `lap_s = (lap n +- m-hat . lap m)/2`, `tau_s = (tau_0 +- m-hat . tau)/2`;
  `v_0 = sum_f w_f (v_+ + v_-)/2`, `B = sum_f w_f (v_+ - v_-)/2 m-hat`. Exact collinear limit.
  But near a point node every odd ingredient is `X . m-hat`, so `B ~ (X . m-hat) m-hat`, which
  depends on the direction of approach: discontinuous at the node, as `'kubler'`'s GGA field is.
  Cheap.
* **C2, multicollinear (Pu et al., arXiv:2110.09897) applied to the potential.** Average the
  collinear potential over directions `e` on a Lebedev sphere with ingredients linear in `m`
  (`(n +- m.e)/2`, `grad (n +- m.e)`, `lap (n +- m.e)`, `(tau_0 +- tau.e)/2`): smooth, axis-free,
  but the collinear limit is exact only with an `E_eff`-type correction, which for an energy is
  `E + d/dx E(x m)` at `x = 1` and for a potential would need deriving; cost `N_dir` (~100 to
  1000) pointwise evaluations of TB09 on the grid.
* **C3, a smooth local reference.** Signed projections along a direction field that is smooth
  through the node (e.g. the density-weighted average of `m` over a neighbourhood). Ad hoc.
* **C4, matrix-valued.** Evaluate TB09 on the 2x2 spin-density matrices
  `rho = (n + m.sigma)/2`, `tau = (tau_0 + tau.sigma)/2`, `grad`, `lap` likewise, as matrix
  functions. Exact for commuting (collinear) arguments; the non-commuting case needs a symmetrised
  definition (`sqrt(tau rho^-1)`, Becke-Roussel's `x(Q/rho^(5/3))`), not unique.

Note for the reviewer: TB09 is paired with LDA correlation in this code (check), so under C the
correlation side is the LDA's, which is already smooth.

Validation for whichever is chosen: collinear limit with nodes, tilted, against the `nspin = 2`
meta branch; spin rotation; the SCF of a magnetic meta cell (which exists? `si2-tb09.in` is
nonmagnetic); a spiral against its supercell.

## D. The tests under the new default

Fast gate on D22 (with `DEFUMAT_QE_ROOT`), then every slow file that runs a noncollinear GGA
(spirals, anisotropy, torques, noncollinear GGA, `nosource`, Elk pairs) through
`tools/run_regression.sh`; each failure is either a functional change on a canted GGA cell
(re-pin to the explicit `'kubler'` where the reference is `pw.x` or Elk) or a defect.

## For the reviewer

1. B: is the proposal right, including the spiral's covariant term in spherical components and
   `with_small`? Anything in `onecenter` that rotates with `|m|` besides this function?
2. C: pick one of C1 to C4 (or something better) for a potential-only meta-GGA, and say what the
   collinear-limit and node behaviour of the pick are. If C2, derive the potential's collinear
   correction.
3. A: anything wrong in the `dE/dq` plan?
4. D: which test files are most likely pinned to `'kubler'` numbers on a canted GGA cell?

## Outcome, 2026-10-12

All of A to C landed (`PLAN.md` P143). One change from the plan: **the PAW spheres take no
spiral term.** With B's covariant gradient the PBE PAW quarter turn missed its 90-degree
supercell by 7.7e-5 Ry (6.4e-5 with the sign flipped), and without it by 6.7e-11, since
`becsum` carries the spiral's phase only at the atom. The meta-GGA spiral stays refused, as it
was under `'kubler'`.
