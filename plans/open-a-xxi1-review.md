# Fable review of `open-a-xxi1.md` (2026-10-10)

The user chose the laboratory sphere on 2026-10-10. This is the review the plan received before
any code, read against the code, with nothing run. Follow it where it corrects the plan.

**Verdict.** The mechanism is right and the sign matches the code's convention, but the central
sentence overclaims: moving the transverse pair's *dense* set to `{G : |G - q| < G_rho}` removes
the 99.5 per cent the shell carries and leaves the identity at the two smooth-sphere cuts the
plan defers, so it is "exact at any cutoff" only once the smooth sphere is moved the same way.
The energy stays variational and `dE/dq` stays the gradient of the computed functional provided
every one of five call sites takes the same displaced set; the change costs no memory on either
validated cell (neither box grows); and the proposed validation can fail, but the P95-cell
checks the plan leans on cannot, because the shell on the oxygen chain is below the energy's
resolution.

1. **The central claim, and the second places.** `build_augmentation(shift=-q)` evaluates
   `Q_ij(G - q)` with phases `e^{-i(G - q).tau}` (`pseudo/augmentation.py:78-85`, `:1273-1274`;
   `scf/driver.py:2434-2437`), and `unfold_spiral_density` has `m_+ = m'_+ e^{-iq.r}`
   (`workflows/spiral.py:780-786`), so the laboratory sphere is `{G : |G - q| < G_rho}`. In the
   diagnostic that is `shell_split`'s `"minus"` key for cobalt and `"plus"` for the oxygen chain;
   record which key held the 0.995. With the dense set moved, the remaining resident-sphere cuts
   of the transverse pair are: (a) `to_dense(rho_smooth)` in `finish_density`
   (`driver.py:5219-5220`, `basis/interpolate.py:62-66`); (b) `to_smooth` of the transverse
   potential in `local_terms` (`driver.py:5330-5332`); (c) the `SphereLayout` under
   `mixing_space = 'g'` (`driver.py:7342`, `mixing.py:1689-1760`), which drops the displaced
   shell at every step, so an augmented spiral under `'g'` is refused or given a complex block on
   `S_q`; (d) `scf_accuracy` on the resident sphere (`potential.py:199`), harmless and `pw.x`'s
   own behaviour. Not displaced: the GGA, `symmetrize` under `nosym`, Hartree, the wavefunction
   spheres, PAW one-centre terms, forces and stress (refused for a spiral), the P122 torque, P123.
   At `ecutrho = 80` (dual 4) (a) and (b) are identities, so there the dense change alone should
   reach the solver's floor; at 160 the remainder is bounded by `1 - 0.995`. Move the smooth
   sphere in the same pass (`{G : |G - q| < G_s}` as the prefix of the displaced list), or state
   the number (a) and (b) leave at 80 and 160.
2. **Variational, and the derivatives.** `_addusdens_spiral` (`driver.py:924-929`) and the
   `cross_integrals` of `_newd_noncollinear_integrals` (`driver.py:838-843`,
   `augmentation.py:237-263`) must contract on the same `S_q`, or it is the P68 shape (energy
   right, `D_ij` wrong). `at_spiral_q(rebuild_basis=False)` (`driver.py:4330-4333`) must take the
   *frozen* displaced set built at the SCF's `q`; the `rebuild_basis=True` branch (`:4360-4363`)
   rebuilds it. Not named by the plan: `at_positions` hands the cross table the resident
   `dense.cartesian(cell)` (`driver.py:3360-3362`), which would rebuild the phases on the wrong
   set; `GEOMETRY_FIELDS` carries `cross_augmentation` (`forces/energy.py:171`).
3. **Shapes and cost.** The displaced set has `ngm` members to within the shell's asymmetry;
   cobalt at 80/160/320 and oxygen at 200 keep their boxes (`floor((G_rho + |q|)/|b_i|)` against
   `floor(G_rho/|b_i|)` is the general check). `_addusdens_spiral` stacks three channels into one
   `(3, ngm)` `g_to_r` (`driver.py:927-929`), which needs its own transform with the displaced
   `fft_index`. The displaced list keeps `generate_gvectors`' order with `G = 0` first
   (`basis/gvectors.py:190-194`), since the tabulated builder reads `gcart[:1]` as the origin
   (`augmentation.py:1117-1131`).
4. **What moves, and whether the validation can fail.** P95 (1.65e-9, 3.26e-7, 3.61e-7,
   2.9e-12), P96 (3.8e-10, 2.4e-7, 8.5e-8, the q = 1/2 rows) and P122's helix sign move; no
   committed tolerance is at risk. `test_a_zero_spiral_is_an_ordinary_noncollinear_run` is a
   1e-10 energy tolerance, not a bit identity. The cobalt check can fail (a wrong sign roughly
   doubles 7.27e-5); keep the control at 7.27e-5 as a test. The oxygen check cannot, and neither
   can `test_a_scan_moves_the_displaced_table_with_q` (1e-10 on the energy). Assert on a quantity
   linear in the density (the supercell's output-minus-input magnetization, or the worst
   eigenvalue on the cobalt pair at 160 Ry) and re-base the scan test the same way.
5. **Missed.** Correct P95's sentence "the truncation shell, if it is there at all, is below
   that" in `PLAN.md`; the plan's 0.5 per cent for the smooth sphere is the remainder, not a
   measurement, and the LDA column (0.9998) puts most of it in the GGA path at the potential
   interpolation; `unfold_spiral_density`'s alias comment (`spiral.py:825-833`) describes a case
   that no longer arises.
