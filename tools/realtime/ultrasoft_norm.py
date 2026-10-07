"""What shifting the projectors alone does to an ultrasoft norm: <psi|S(k + kappa)|psi> - 1.

The refusal of ultrasoft and PAW datasets in real time rests on a derived term,
``P_kappa``, that the equation of motion gains when the overlap moves with the
field (``HARMONICS-NEXT.md``, "What is refused"). This measures the part that
needs no derivation: at fixed states, ``<psi|S(k + kappa)|psi>`` moves away from
one, to first order by ``kappa . <psi|dS/dk|psi>``, which is
``VelocityOperator.apply_s``. Printed for each occupied band of ultrasoft
silicon at Gamma and at a general point, and the exact value at the peak
``kappa`` of the published silicon pulse (0.11 1/bohr, 0.43 eV at 1e11 W/cm^2).

    JAX_PLATFORMS=cpu python3 tools/realtime/ultrasoft_norm.py
"""
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import defumat  # noqa: F401
from defumat import Calculator
from defumat.response import VelocityOperator
from defumat.system.kpoints import KPoints
from defumat.workflows.nscf import fixed_density_states

repo = Path(__file__).resolve().parents[2]
calculator = Calculator.from_file(repo / "tests/data/qe/si2-us.in",
                                  pseudo_dir=repo / "tests/data/pseudo", announce=False)
scf = calculator.get_scf(conv_thr=1e-10)
kp = KPoints(coords=np.array([[0.0, 0.0, 0.0], [0.25, 0.1, -0.05]]), weights=np.array([0.5, 0.5]))
calculation, _, eigenvalues, psi = fixed_density_states(
    calculator.system, calculator.pseudos, scf.density, kpoints=kp, nbnd=4, conv_thr=1e-11)
psi = jnp.asarray(psi)
v_scf = calculation.potential(jnp.asarray(scf.density)).v_scf
velocity = VelocityOperator(calculation, v_scf)
ds = np.asarray(velocity.apply_s(psi, jnp.asarray([1.0, 0.0, 0.0])))
slope = np.real(np.einsum("skng,skng->skn", np.conj(np.asarray(psi)), ds))[0]
kappa = 0.11
kc = np.asarray(velocity.kcart)
moved = calculation.at_kcart(jnp.asarray(kc + np.array([kappa, 0.0, 0.0])[None, :]))
ham = moved.hamiltonian(v_scf)[0]
exact = []
for ik in range(psi.shape[1]):
    spsi = np.asarray(ham.apply_s(psi[0, ik], ik))
    exact.append(np.real(np.einsum("ng,ng->n", np.conj(np.asarray(psi[0, ik])), spsi)) - 1.0)
print(json.dumps({"dS_dkx_per_band": slope.tolist(), "norm_minus_one_at_kappa_0.11": np.asarray(exact).tolist(),
                  "first_order_estimate": (kappa * slope).tolist()}), flush=True)
