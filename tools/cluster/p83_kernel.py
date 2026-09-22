"""Is the exchange-correlation kernel of a textured spinor right? Two routes
that share only the ground state.

**The problem.** P83's dielectric tensor of a magnetic spin-orbit insulator
agrees with ``ph.x`` to 4.6e-4 across the moment and is **5.3 per cent** away
along it, on a cell where the two codes agree on the ground state to the printed
digit. That is 40 per cent of the whole exchange-correlation contribution, the
same solve in RPA giving 1.377 against the screened 1.574. The quantity is
refused by name until the disagreement is located.

**What is already excluded, so that none of it is tested a fourth time.** The
*solve* is clear: ``chi_0`` under a potential probe, which carries ``dvan_so``
and no kernel at all, was central-differenced against the density on this cell
in P83 and falls as ``h^2`` on every probe with a transverse component, with all
four density channels live. And ``dmxc_nc`` differs from a ``jvp`` of
``v_of_rho`` in exactly three places, every one a threshold, and each fires at
**zero** of this cell's 157464 grid points. So the disagreement is downstream of
``chi_0`` and is not a convention at an edge, which leaves the kernel, the
electric-field source term, and ``ph.x`` itself.

**What this script does, and why it separates them.** It measures one number,
the longitudinal spin susceptibility ``dm_z/dB_z`` at ``q = 0``, two ways:

* **the response route** solves the same screened fixed point the dielectric
  constant is built on, ``drho = chi_0(dv_bare + K drho)``, with ``K`` the same
  ``dv_of_drho`` and ``chi_0`` the same spinor Sternheimer solve;
* **the finite-difference route** converges the SCF under a small uniform Zeeman
  field at ``+h`` and ``-h`` and differences the converged moment, which touches
  no response solver, no kernel object and no perturbation expression at all.

They share the ground state and nothing above it. If they agree, the kernel and
its self-consistency are right together, and what is left of the 5.3 per cent is
the electric-field source term or ``ph.x``. If they disagree, the kernel is
located without any reference to another code, which is the better outcome
because it needs no argument about whose convention is whose.

**Why a magnetic probe rather than an electric one.** The electric-field
perturbation is spin-independent, and the *nonmagnetic* spinor case already
reproduces ``ph.x`` to 4.3e-5, so the source term is largely cleared and the
kernel's magnetic blocks are what has never been exercised on its own. A uniform
Zeeman field reaches exactly those blocks: it is blind to Hartree, which does not
see the magnetization at all, so what screens it **is** ``f_xc``.

**Longitudinal and transverse are both taken, and the transverse one is not a
throwaway.** In noncollinear LSDA the kernel splits into
``(dB_xc/d|m|) m m`` along the moment and ``(B_xc/|m|) (1 - m m)`` across it, and
the second is a different expression rather than a component of the first -- it
is the term that rotates the exchange field with the magnetization. At ``q = 0``
a transverse response would be the Goldstone mode and would diverge, except that
this cell carries spin-orbit coupling, which gaps it, so the number is finite and
is the only check either branch has ever had.

Usage:
    python3 tools/cluster/p83_kernel.py --case i-atom-soc --steps 2e-4,1e-4
"""
import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.response.efield import _screening_kernel
from defumat.response.sternheimer import make_sternheimer
from defumat.scf import Calculation, run_scf
from defumat.scf.fields import MagneticField
from defumat.system import build_system

CASES = Path("tests/data/qe")
PSEUDO = Path("tests/data/pseudo")

#: The three Cartesian directions a uniform field can point along, and the two
#: that matter: ``z`` is the moment's own axis on this cell and is the component
#: ``ph.x`` disagrees along, ``x`` is across it.
DIRECTIONS = {"z": (0.0, 0.0, 1.0), "x": (1.0, 0.0, 0.0)}


def ground_state(case: str, conv_thr: float, max_iterations: int):
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(
        read_upf(PSEUDO / species.pseudo_file)
        for species in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    result = run_scf(
        system, pseudos, calculation=calculation,
        conv_thr=conv_thr, max_iterations=max_iterations,
    )
    if not result.converged:
        raise SystemExit(
            f"the zero-field ground state did not converge: {result.accuracy}"
        )
    return system, pseudos, calculation, result


def moment_of(density, cell):
    """The cell's total moment vector, from the three magnetization channels."""
    weight = cell.volume / density[0].size
    return np.asarray([float(jnp.sum(density[i]) * weight) for i in (1, 2, 3)])


def field_potential(calculation, density, direction, amplitude):
    """The bare perturbation a uniform field of this size puts in the potential.

    Taken from :meth:`MagneticField.potential` rather than written out here, so
    that the two routes cannot differ by a sign or a factor of two: the finite
    difference converges the SCF under *this* object, and the response route
    differentiates *this* potential.
    """
    field = MagneticField(
        regions=None,
        uniform=jnp.asarray([amplitude * component for component in direction]),
        atomic=None, targets=None, penalty=0.0,
    )
    potential, _, _ = field.potential(density, calculation.system.cell)
    return field, potential


def finite_difference(system, pseudos, calculation, reference, direction, step,
                      conv_thr, max_iterations):
    """``dm/dB`` from two converged SCF runs under plus and minus the field."""
    moments, records = {}, {}
    for sign in (+1, -1):
        field, _ = field_potential(
            calculation, reference.density, direction, sign * step
        )
        started = time.time()
        result = run_scf(
            system, pseudos, calculation=calculation,
            conv_thr=conv_thr, max_iterations=max_iterations,
            magnetic_field=field, starting_from=reference,
        )
        moments[sign] = moment_of(result.density, calculation.system.cell)
        records[sign] = {
            "converged": bool(result.converged),
            "iterations": int(result.iterations),
            "accuracy": None if result.accuracy is None else float(result.accuracy),
            "moment": moments[sign].tolist(),
            "seconds": time.time() - started,
        }
        if not result.converged:
            raise SystemExit(
                f"the field leg at {sign * step} did not converge: "
                f"{result.accuracy}"
            )
    susceptibility = (moments[+1] - moments[-1]) / (2.0 * step)
    return susceptibility, records


def screened_response(calculation, reference, direction, iterations, threshold):
    """``dm/dB`` from the same screened response the dielectric constant uses.

    **The fixed point is solved as a linear system rather than iterated**, and
    that is not a convenience. What is being measured is

        (1 - chi_0 K) drho = chi_0 dv_bare,

    and the whole physics of a magnet is that the operator on the left is nearly
    singular: the interacting susceptibility is ``chi_0/(1 - I chi_0)``, so a
    Stoner-enhanced cell has an eigenvalue of ``chi_0 K`` approaching one from
    below. Simple mixing on such a system converges as ``|1 - beta(1 - J)|`` per
    step, which for ``J`` near one is arbitrarily slow -- and the transverse
    direction is worse still, since at ``q = 0`` a rotation of the moment is a
    Goldstone mode gapped only by the spin-orbit anisotropy, milli-electronvolt
    scale against an exchange field of electronvolts. A hundred iterations there
    would return a residual rather than a number, and a residual is not a
    measurement.

    GMRES treats the enhancement exactly: the number of matrix applications is
    set by the spectrum's *spread* rather than by its proximity to one, so the
    near-singular direction costs a few extra Krylov vectors instead of an
    unbounded number of sweeps. Each application is one ``chi_0`` solve and one
    kernel evaluation, which is the same unit of work simple mixing spends per
    sweep.
    """
    from scipy.sparse.linalg import LinearOperator, gmres

    solver = make_sternheimer(calculation, reference, noncollinear=True)
    screen = _screening_kernel(calculation, reference.density, "full")
    _, bare = field_potential(calculation, reference.density, direction, 1.0)

    shape = np.asarray(reference.density).shape
    applications = []

    def apply(vector):
        density = jnp.asarray(vector.reshape(shape))
        result = density - solver.chi0(screen(density))
        applications.append(1)
        return np.asarray(result).ravel()

    operator = LinearOperator(
        (int(np.prod(shape)), int(np.prod(shape))), matvec=apply, dtype=float
    )
    right_hand_side = np.asarray(solver.chi0(bare)).ravel()

    residuals = []
    solution, info = gmres(
        operator, right_hand_side, rtol=threshold, maxiter=iterations,
        callback=lambda value: residuals.append(float(value)),
        callback_type="pr_norm",
    )
    drho = jnp.asarray(solution.reshape(shape))
    # The residual is recomputed rather than read off the solver, because what
    # matters is the residual of the system that was meant to be solved and not
    # the one GMRES restarted on.
    residual = float(
        np.linalg.norm(apply(solution) - right_hand_side)
        / max(np.linalg.norm(right_hand_side), 1e-300)
    )
    return (
        moment_of(drho, calculation.system.cell),
        {
            "applications": len(applications),
            "info": int(info),
            "converged": bool(info == 0),
            "residual": residual,
            "trace": residuals,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="i-atom-soc")
    parser.add_argument("--steps", default="2e-4,1e-4",
                        help="field amplitudes in Ry for the central difference")
    parser.add_argument("--directions", default="z,x")
    parser.add_argument("--conv-thr", type=float, default=1e-12)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--response-iterations", type=int, default=120)
    parser.add_argument("--response-threshold", type=float, default=1e-8)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    steps = [float(value) for value in arguments.steps.split(",")]
    directions = [name.strip() for name in arguments.directions.split(",")]

    system, pseudos, calculation, reference = ground_state(
        arguments.case, arguments.conv_thr, arguments.max_iterations
    )
    record = {
        "case": arguments.case,
        "total_energy": float(reference.total_energy),
        "moment": moment_of(reference.density, calculation.system.cell).tolist(),
        "directions": {},
    }
    print(f"ground state   {reference.total_energy:.9f} Ry, "
          f"moment {record['moment']}")

    for name in directions:
        direction = DIRECTIONS[name]
        started = time.time()
        response, response_record = screened_response(
            calculation, reference, direction,
            arguments.response_iterations, arguments.response_threshold,
        )
        response_record["seconds"] = time.time() - started
        if not response_record["converged"]:
            # **Stop rather than spend the finite differences on it.** An
            # unconverged solve returns a number, and that number compared
            # against a converged finite difference reads as a disagreement
            # about the kernel when it is a disagreement about the solve. The
            # directions are ordered with the moment's own axis first for this
            # reason: it is the one ``ph.x`` disagrees along, and the transverse
            # one is the near-Goldstone direction that is expected to be harder.
            print(f"[{name}] the screened solve did not converge "
                  f"(residual {response_record['residual']:.3e}); "
                  f"stopping before the finite difference")
            record["directions"][name] = {
                "response": response.tolist(),
                "response_detail": response_record,
                "finite_difference": {},
                "abandoned": "the screened solve did not converge",
            }
            break
        entry = {
            "response": response.tolist(),
            "response_detail": response_record,
            "finite_difference": {},
        }
        print(f"[{name}] response      dm/dB = {response} "
              f"({response_record['iterations']} iterations, "
              f"residual {response_record['residual']:.3e})")

        for step in steps:
            difference, legs = finite_difference(
                system, pseudos, calculation, reference, direction, step,
                arguments.conv_thr, arguments.max_iterations,
            )
            # The component along the field is the one the two routes are
            # compared on; the other two are reported because a nonzero
            # off-axis response is a statement about spin-orbit coupling rather
            # than noise, and reading only the diagonal would hide it.
            axis = [index for index, value in enumerate(direction) if value][0]
            relative = abs(
                (difference[axis] - response[axis]) / response[axis]
            ) if response[axis] else None
            entry["finite_difference"][f"{step:g}"] = {
                "value": difference.tolist(),
                "legs": legs,
                "relative_difference_on_axis": relative,
            }
            print(f"[{name}] step {step:g}   dm/dB = {difference}  "
                  f"relative on axis {relative}")
        record["directions"][name] = entry

    Path(arguments.out).write_text(json.dumps(record, indent=2, default=str))
    print(f"written to {arguments.out}")


if __name__ == "__main__":
    main()
