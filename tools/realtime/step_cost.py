"""What one real-time step costs, split into its parts, and the frozen sphere's error.

The first stage of ``HARMONICS-NEXT.md`` ("Measure before building"): on a
silicon cell at one k-point, warm medians of

* the projector rebuild at ``k + kappa`` (``Calculation.at_kcart`` on a
  one-row calculation, which rebuilds ``|k+G+kappa|^2`` and ``vkb``),
* one Hamiltonian application on the occupied bands,
* the current, the ``kappa`` gradient of the kinetic and nonlocal band energy,
* one fourth-order Taylor step (one rebuild and four applications),

each compiled once and timed over repeats; and the eigenvalues of
``H(k + kappa)`` on the sphere of ``k`` against a sphere rebuilt at
``k + kappa`` at ``kappa`` = 0.1, 0.3 and 0.5 1/bohr.

    JAX_PLATFORMS=cpu python3 tools/realtime/step_cost.py benchmarks/si-1k.in [repeats]

Prints one JSON line per measurement.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import defumat  # noqa: F401  (x64)
from defumat import Calculator
from defumat.realtime.propagators import taylor4
from defumat.system.kpoints import KPoints
from defumat.workflows.nscf import fixed_density_states


def _median(fn, repeats):
    jax.block_until_ready(fn())
    samples = []
    for _ in range(repeats):
        t = time.perf_counter()
        jax.block_until_ready(fn())
        samples.append(time.perf_counter() - t)
    return float(np.median(samples)), samples


def main(path, repeats=15):
    repo = Path(__file__).resolve().parents[2]
    calculator = Calculator.from_file(path, pseudo_dir=repo / "tests" / "data" / "pseudo",
                                      announce=False)
    scf = calculator.get_scf()
    system = calculator.system
    nocc = int(round(calculator.calculation.nelec / 2))
    k = np.array([[0.1, 0.2, 0.3]])
    calculation, _, eigenvalues, psi = fixed_density_states(
        system, calculator.pseudos, scf.density,
        kpoints=KPoints(coords=k, weights=np.ones(1)), nbnd=nocc + 4, conv_thr=1e-10)
    psi = jnp.asarray(psi)[0, 0, :nocc]  # (nocc, npwx)
    v_scf = calculation.potential(jnp.asarray(scf.density)).v_scf
    terms = calculation.local_terms(v_scf)
    row = calculation.at_rows([0])
    k0 = jnp.asarray(calculation.system.kpoints.cartesian(calculation.system.cell))
    kappa = jnp.asarray([0.05, 0.0, 0.0])
    dij = calculation.projectors.dij
    npwx = calculation.basis.planewaves.npwx

    @jax.jit
    def rebuild(kappa):
        moved = row.at_kcart(k0 + kappa)
        return moved.kinetic, moved.projectors.vkb

    @jax.jit
    def apply(kappa, psi):
        ham = row.at_kcart(k0 + kappa).hamiltonian_from(terms)[0]
        return ham.apply(psi, 0)

    hams = row.at_kcart(k0 + kappa).hamiltonian_from(terms)

    @jax.jit
    def apply_only(psi):
        return hams[0].apply(psi, 0)

    @jax.jit
    def current(kappa, psi):
        def band_energy(x):
            moved = row.at_kcart(k0 + x)
            kin = jnp.sum(moved.kinetic[0] * jnp.real(jnp.conj(psi) * psi))
            becp = jnp.conj(moved.projectors.vkb[0]).T @ psi.T  # (nkb, nocc)
            nl = jnp.real(jnp.einsum("in,ij,jn->", jnp.conj(becp), dij, becp))
            return kin + nl
        return jax.grad(band_energy)(kappa)

    @jax.jit
    def step(kappa, psi):
        ham = row.at_kcart(k0 + kappa).hamiltonian_from(terms)[0]
        return taylor4(lambda x: ham.apply(x, 0), psi, 0.05, 0.0)

    # The same three operations through the table of g_l(q^2)
    # (:mod:`defumat.realtime.radial`) instead of the direct transform.
    import dataclasses

    import equinox as eqx

    from defumat.realtime.radial import radial_table

    ecut = float(calculation.system.ecutwfc) if hasattr(calculation.system, "ecutwfc") else None
    kmax = float(np.max(np.linalg.norm(np.asarray(row.projector_core.kg[0]), axis=-1)))
    table = radial_table(calculator.pseudos, calculation.system.cell.volume, (kmax + 1.0) ** 2)
    core = row.projector_core
    gcart = core.kg - k0[:, None, :]
    mask = row.basis.planewaves.mask
    positions = jnp.asarray(calculation.system.structure.positions)
    template = row.hamiltonian_from(terms)[0]

    def tabulated(kappa):
        kg = gcart + (k0 + kappa)[:, None, :]
        kinetic = jnp.where(mask, jnp.sum(kg * kg, axis=-1), 0.0).astype(template.kinetic.dtype)
        moved = eqx.tree_at(lambda c: (c.columns, c.kg), core,
                            (table.columns(kg).astype(core.columns.dtype), kg))
        return kinetic, moved.at_positions(positions)

    @jax.jit
    def table_rebuild(kappa):
        kinetic, projectors = tabulated(kappa)
        return kinetic, projectors.vkb

    @jax.jit
    def table_current(kappa, psi):
        def band_energy(x):
            kinetic, projectors = tabulated(x)
            kin = jnp.sum(kinetic[0] * jnp.real(jnp.conj(psi) * psi))
            becp = jnp.conj(projectors.vkb[0]).T @ psi.T
            return kin + jnp.real(jnp.einsum("in,ij,jn->", jnp.conj(becp), dij, becp))
        return jax.grad(band_energy)(kappa)

    @jax.jit
    def table_step(kappa, psi):
        kinetic, projectors = tabulated(kappa)
        ham = dataclasses.replace(template, kinetic=kinetic, projectors=projectors)
        return taylor4(lambda x: ham.apply(x, 0), psi, 0.05, 0.0)

    # the table against the transform at kappa: the projectors and the current
    vkb_t = np.asarray(table_rebuild(kappa)[1]); vkb_d = np.asarray(rebuild(kappa)[1])
    j_t = np.asarray(table_current(kappa, psi)); j_d = np.asarray(current(kappa, psi))

    out = {"input": str(path), "npwx": int(npwx), "nocc": nocc,
           "nkb": int(calculation.projectors.nkb),
           "grid": list(calculation.basis.smooth.grid),
           "table_terms": int(table.coefficients.shape[1]),
           "table_vkb_max_diff": float(np.abs(vkb_t - vkb_d).max()),
           "table_current_max_diff": float(np.abs(j_t - j_d).max()),
           "current_scale": float(np.abs(j_d).max())}
    for name, fn in (("rebuild", lambda: rebuild(kappa)),
                     ("apply_occupied", lambda: apply_only(psi)),
                     ("rebuild_and_apply", lambda: apply(kappa, psi)),
                     ("current", lambda: current(kappa, psi)),
                     ("taylor4_step", lambda: step(kappa, psi)),
                     ("table_rebuild", lambda: table_rebuild(kappa)),
                     ("table_current", lambda: table_current(kappa, psi)),
                     ("table_taylor4_step", lambda: table_step(kappa, psi))):
        median, samples = _median(fn, repeats)
        out[name + "_ms"] = 1e3 * median
    print(json.dumps(out), flush=True)

    # The frozen sphere's error: H(k + kappa) on the sphere of k against the
    # sphere of k + kappa, lowest nocc + 4 eigenvalues, kappa along x.
    from defumat.units import RY_TO_EV

    cell = calculation.system.cell
    for size in (0.1, 0.3, 0.5):
        shift = np.array([[size, 0.0, 0.0]])
        moved_cart = np.asarray(k0) + shift
        frozen = fixed_density_states(
            system, calculator.pseudos, scf.density,
            kpoints=KPoints(coords=k, weights=np.ones(1)), nbnd=nocc + 4,
            conv_thr=1e-10, kcart=moved_cart)[2]
        coords = moved_cart / (2 * np.pi / cell.alat)
        rebuilt = fixed_density_states(
            system, calculator.pseudos, scf.density,
            kpoints=KPoints(coords=coords, weights=np.ones(1)), nbnd=nocc + 4,
            conv_thr=1e-10)[2]
        difference = (np.asarray(frozen) - np.asarray(rebuilt)).reshape(-1) * RY_TO_EV
        print(json.dumps({"kappa": size, "frozen_minus_rebuilt_ev_max": float(np.abs(difference).max()),
                          "frozen_minus_rebuilt_ev_occupied": difference[:nocc].tolist()}),
              flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 15)
