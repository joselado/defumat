"""What the orientation torque costs on the card, by route (``OPEN.md`` Part XXII item 2).

The torque ``-dF/dw`` of a rigid turn of the texture, on one k-point of a spinor
PAW cell, by several routes, one process each:

* ``whole``: :func:`defumat.forces.torque.orientation_torque` as it is, the
  production call (three ``jax.jvp`` since 2026-10-10, a reverse pass before);
* ``reverse``: ``jax.grad`` of the public energy ``band_energy_at_rotation``,
  which is the route ``whole`` took until 2026-10-10 (what asked 33.37 GiB
  more on the fifteen-cell NiBr2 helix at ``1 1 1``);
* ``vmap``: the three ``jvp`` of ``whole`` taken together, a ``vmap`` over the
  tangents, so the primal is evaluated once and the tangent three wide;
* ``remat``: the same energy summed over blocks of ``--block`` bands in a scan
  whose body is ``jax.checkpoint``-ed, so the backward pass holds one block's
  residuals;
* ``forward``: three ``jax.jvp``, one per generator, over the same blocks;
* ``closed``: no derivative of the band energy at all, the force theorem's closed
  form, ``integral m_out x B_xc[R rho]`` plus the one-centre pairing with the
  input ``becsum`` turned as a vector (exact without the coupling on the source,
  which is Route A's, P122).

Each prints the torque (so the routes can be compared on the same states), the
calculation's resolved ``band_batch`` and ``k_batch`` (the forward routes' peak
is set by the first), and the device's ``peak_bytes_in_use`` before and after;
the single-program routes also print the compiled ``memory_analysis()``. With
``--repeat 2`` the call is made twice and the second is timed as well, since
the first includes the compilation. The states are random and orthonormal,
which is enough for the memory: the torque is a derivative at frozen states, so
every route gives the same three numbers on any states (2e-15 of the norm on
nine atoms of NiI2, 2026-10-09; ``plans/open-d-xxii2-orientation-torque.md``).

    python3 tools/gpu/torque_tape.py INPUT --pseudo-dir DIR --mode whole [--block 16]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

GIB = 1024.0 ** 3


def _stats():
    import jax

    stats = jax.devices()[0].memory_stats() or {}
    return int(stats.get("bytes_in_use", 0)), int(stats.get("peak_bytes_in_use", 0))


def _analysis(compiled):
    try:
        ma = compiled.memory_analysis()
        return {"temp": int(ma.temp_size_in_bytes),
                "argument": int(ma.argument_size_in_bytes),
                "output": int(ma.output_size_in_bytes)}
    except Exception as error:  # a backend without the analysis
        return {"error": str(error)[:200]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--pseudo-dir", required=True)
    parser.add_argument("--mode", required=True,
                        choices=("whole", "reverse", "vmap", "remat", "forward",
                                 "closed"))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--block", type=int, default=16)
    parser.add_argument("--soc-scale", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np

    from defumat import Calculator
    from defumat.forces.torque import (
        _turned_becsum, orientation_torque,
        rotate_texture, rotation_near,
    )

    calculator = Calculator.from_file(arguments.input, pseudo_dir=arguments.pseudo_dir,
                                      announce=False)
    system = calculator.system
    if arguments.soc_scale != 1.0:
        system = system.with_soc_scale(arguments.soc_scale)
    from defumat.scf.driver import Calculation

    calculation = Calculation(system, calculator.pseudos, k_batch=1)
    texture = jnp.asarray(calculation.starting_density())
    becsum = tuple(calculation.starting_becsum())
    nk = calculation.system.kpoints.nk
    nbnd = int(calculator.estimate().nbnd)
    ndim = int(calculation.basis.npwx * calculation.npol)
    rng = np.random.default_rng(11)
    raw = rng.standard_normal((nk, ndim, nbnd)) + 1j * rng.standard_normal((nk, ndim, nbnd))
    mask = np.asarray(calculation.basis.planewaves.mask).reshape(nk, -1)
    raw = raw * np.tile(mask, (1, calculation.npol))[:, :, None]
    states = np.stack([np.linalg.qr(raw[k])[0].T for k in range(nk)])[None]
    nocc = int(round(float(calculation.nelec)))
    weights = np.zeros((1, nk, nbnd))
    weights[:, :, :nocc] = 1.0 / nk
    states = jnp.asarray(states, dtype=calculation.system.cell.precision.complex)
    weights = jnp.asarray(weights)
    base = jnp.eye(3)
    origin = jnp.zeros(3)
    jax.block_until_ready((states, texture))
    in_use0, peak0 = _stats()
    record = {"mode": arguments.mode, "nbnd": nbnd, "ndim": ndim, "nk": nk,
              "band_batch": calculation.band_batch, "k_batch": calculation.k_batch,
              "smooth_grid": list(calculation.basis.smooth.grid),
              "in_use_before": in_use0, "peak_before": peak0}
    print(json.dumps(record), flush=True)

    clock = time.time()
    try:
        if arguments.mode in ("whole", "reverse", "vmap"):
            # The production call, as ``run_orientation_torque`` makes it at one
            # k-point, kept compiled by ``defumat.eager``; the other two through
            # the same keeper. Not wrapped in a ``jit`` here, since a ``jit``
            # closing over the states folds them in as constants and the compile
            # alone then exceeded 8 GB of host memory.
            from defumat.eager import compiled_function
            from defumat.forces.torque import _hamiltonian_of, band_energy_at_rotation

            def energy(value):
                rotation = rotation_near(value, base)
                hamiltonian = _hamiltonian_of(
                    calculation, rotate_texture(texture, rotation),
                    _turned_becsum(becsum, rotation))
                total = 0.0
                for ik in range(nk):
                    applied = hamiltonian.apply(states[0, ik], ik)
                    total = total + jnp.sum(weights[0, ik] * jnp.real(
                        jnp.sum(jnp.conj(states[0, ik]) * applied, axis=-1)))
                return total

            def call():
                if arguments.mode == "whole":
                    return orientation_torque(calculation, states, weights, texture,
                                              base, k_batch=None, becsum=becsum)
                if arguments.mode == "reverse":
                    gradient = compiled_function(jax.grad(
                        lambda w: band_energy_at_rotation(
                            calculation, states, weights, texture, base, w, becsum)),
                        origin)
                    return -np.asarray(gradient(origin))
                program = compiled_function(
                    lambda v, t: jax.vmap(lambda one: jax.jvp(energy, (v,), (one,))[1])(t),
                    origin, jnp.eye(3, dtype=origin.dtype))
                return -np.asarray(program(origin, jnp.eye(3, dtype=origin.dtype)))

            torque = call()
            jax.block_until_ready(jnp.asarray(torque))
            record["first_seconds"] = time.time() - clock
            for _ in range(arguments.repeat - 1):
                again = time.time()
                torque = call()
                jax.block_until_ready(jnp.asarray(torque))
                record["second_seconds"] = time.time() - again
        elif arguments.mode in ("remat", "forward"):
            from defumat.forces.torque import _hamiltonian_of

            block = arguments.block
            nblocks = -(-nbnd // block)
            pad = nblocks * block - nbnd
            psi = jnp.pad(states[0], ((0, 0), (0, pad), (0, 0)))
            occupation = jnp.pad(weights[0], ((0, 0), (0, pad)))

            # The states, their weights and the texture are arguments, not
            # constants of the program: closed over, a 100 MB block of states is
            # folded into the executable and the compile ran out of 8 GB of host.
            def energy(value, psi, occupation, texture):
                rotation = rotation_near(value, base)
                hamiltonian = _hamiltonian_of(calculation, rotate_texture(texture, rotation),
                                              _turned_becsum(becsum, rotation))
                total = 0.0
                for ik in range(nk):
                    blocks = psi[ik].reshape(nblocks, block, -1)
                    occ = occupation[ik].reshape(nblocks, block)

                    def body(carry, item, ik=ik):
                        vectors, w = item
                        applied = hamiltonian.apply(vectors, ik)
                        bands = jnp.real(jnp.sum(jnp.conj(vectors) * applied, axis=-1))
                        return carry + jnp.sum(w * bands), None

                    if arguments.mode == "remat":
                        body = jax.checkpoint(body)
                    part, _ = jax.lax.scan(body, jnp.zeros((), psi.real.dtype), (blocks, occ))
                    total = total + part
                return total

            data = (psi, occupation, texture)
            if arguments.mode == "remat":
                program = jax.jit(jax.grad(energy))
                compiled = program.lower(origin, *data).compile()
                record["compiled"] = _analysis(compiled)
                torque = -np.asarray(compiled(origin, *data))
            else:
                def tangent(axis, psi, occupation, texture):
                    return jax.jvp(lambda v: energy(v, psi, occupation, texture),
                                   (origin,), (axis,))[1]

                program = jax.jit(tangent)
                compiled = program.lower(jnp.eye(3)[0], *data).compile()
                record["compiled"] = _analysis(compiled)
                torque = -np.array([float(compiled(axis, *data)) for axis in jnp.eye(3)])
        else:
            from defumat.scf.driver import _onecenter_torque
            from defumat.scf.spin_torque import exchange_torque

            becsum_out = calculation.becsum(states, weights)
            output = calculation.density(states, weights, becsum_out)
            potential = calculation.potential(rotate_texture(texture, base), 1.0, None)
            grid = np.asarray(exchange_torque(output, potential.v_scf,
                                              calculation.system.cell).total)
            one_centre = (_onecenter_torque(calculation, becsum, becsum_out)
                          if calculation.is_paw else 0.0)
            torque = grid + one_centre
        jax.block_until_ready(jnp.asarray(torque))
    except Exception as error:  # an allocation the card refuses is the measurement
        in_use1, peak1 = _stats()
        record.update({"error": str(error)[:2000], "in_use_after": in_use1,
                       "peak_after": peak1, "seconds": time.time() - clock})
        print(json.dumps(record), flush=True)
        if arguments.out is not None:
            arguments.out.write_text(json.dumps(record, indent=1))
        raise
    in_use1, peak1 = _stats()
    record.update({"seconds": time.time() - clock, "torque": list(map(float, torque)),
                   "in_use_after": in_use1, "peak_after": peak1,
                   "peak_rise": peak1 - peak0})
    print(json.dumps(record), flush=True)
    print(f"[torque] {arguments.mode}: peak {peak0 / GIB:.3f} -> {peak1 / GIB:.3f} GiB "
          f"(+{(peak1 - peak0) / GIB:.3f}); compiled {record.get('compiled')}", flush=True)
    if arguments.out is not None:
        arguments.out.write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
