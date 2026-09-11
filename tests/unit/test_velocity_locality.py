"""What of ``v_scf`` reaches ``dH/dk``, and therefore which field a velocity is.

``OPEN.md`` A2: the response stack rebuilt the Zeeman term from the **input**
magnetic field rather than the one the ground state converged under. Two of the
call sites are outside the Sternheimer stack --
:func:`defumat.response.velocity.band_velocities` and
:func:`defumat.workflows.run_conductivity` -- and both now thread
``SCFResult.magnetic_field`` and ``.field_scale`` through. This file measures
what that is worth, and the answer has two halves.

A magnetic field and a constraining field both enter ``v_scf`` as a *local*
multiplicative potential (:mod:`defumat.scf.fields`), while
:class:`~defumat.response.velocity.VelocityOperator` differentiates ``H(k)``
with respect to ``kcart`` at a frozen sphere. The obvious conclusion -- that a
local potential is a constant of that derivative, so nothing moves -- **is true
for a norm-conserving dataset and false for an ultrasoft or PAW one**, and the
second half is the reason the fix is a fix rather than a tidy-up.

The route is ``newd``. :meth:`~defumat.scf.driver.Calculation.hamiltonian`
builds ``deeq`` from ``v_scf + vltot`` on every call, and ``deeq`` contracts
with the projectors ``vkb(k)``, which carry ``k``. So on a soft dataset the
local potential reaches the velocity through the nonlocal term, and using the
input field instead of the converged one changes every band velocity, every
optical conductivity and every anomalous Hall number built from them -- by 0.37
Ry bohr out of 398 for the arbitrary bump below, with no error and nothing in
the output to say which field was used.
"""

from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.response import VelocityOperator
from defumat.scf import Calculation
from defumat.system import build_system

pytestmark = pytest.mark.unit

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"


@lru_cache(maxsize=2)
def _calculation(case: str, pseudo_dir: Path):
    """No SCF: the claim is about the *operator*, not about a ground state."""
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    return Calculation(system, pseudos)


def _velocity_shift(case: str, pseudo_dir: Path) -> tuple[float, float]:
    """``(how much dH/dk moves, what it moves against)`` for a local bump.

    The bump is random rather than constant on purpose -- a constant shifts
    every eigenvalue and would leave a norm-conserving ``dH/dk`` unchanged for
    a reason weaker than the one being asserted.
    """
    calculation = _calculation(case, pseudo_dir)
    rng = np.random.default_rng(0)

    v_scf = calculation.potential(
        jnp.zeros((calculation.nspin_mag,) + tuple(calculation.basis.dense.grid))
    ).v_scf
    nk = len(calculation.system.kpoints.weights)
    width = calculation.basis.planewaves.npwx * calculation.npol
    psi = jnp.asarray(
        rng.normal(size=(calculation.nspin, nk, 4, width))
        + 1j * rng.normal(size=(calculation.nspin, nk, 4, width))
    )

    plain = VelocityOperator(calculation, v_scf, None).matrix_elements(psi)
    bumped = VelocityOperator(
        calculation, v_scf + jnp.asarray(rng.normal(size=v_scf.shape)), None
    ).matrix_elements(psi)
    return float(jnp.abs(bumped - plain).max()), float(jnp.abs(plain).max())


def test_a_norm_conserving_velocity_cannot_see_a_local_potential(pseudo_dir):
    """Exactly zero, not "small": the term is absent rather than cancelling."""
    moved, scale = _velocity_shift("si2-nc-force", pseudo_dir)
    assert moved == 0.0
    # The scale it is zero against. A null that cannot be told from an operator
    # returning zeros is the trap ``CLAUDE.md`` names.
    assert scale > 1.0


def test_an_ultrasoft_velocity_does_see_one_through_newd(pseudo_dir):
    """And this is what makes threading the converged field a correctness fix.

    ``deeq`` is rebuilt from ``v_scf`` inside ``hamiltonian`` and multiplies
    ``vkb(k)``, so the local potential arrives at ``dH/dk`` through the
    nonlocal term. Feeding it the input field where the density belongs to a
    reduced one is a wrong velocity with no symptom.
    """
    moved, scale = _velocity_shift("si2-us", pseudo_dir)
    # 0.366 against 398, which is 9.2e-4 of it: small only because the bump is
    # arbitrary, and far above any round-off the norm-conserving half sits at.
    assert moved > 1.0e-4 * scale
