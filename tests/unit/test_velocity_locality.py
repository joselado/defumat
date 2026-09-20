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


# --- the l = 1 tangent at k + G = 0 ------------------------------------------
#
# `basis/gvectors.py:117`'s `modulus` and `harmonics.py`'s origin guards are each
# right on their own factor and wrong on the product. `f_l(|q|)` and `Y_lm(qhat)`
# are both guarded by zeroing at `q = 0`, so the chain rule returns
# `Y df + dY f = 0`. For `l = 1` the *product* is smooth there and its derivative
# is not zero: `f_1(q) -> c q` and `Y_1m(qhat) = sqrt(3/4pi) q_alpha/q`, so the
# product is `sqrt(3/4pi) c q_alpha`, a linear function of the vector `q`.
# `l = 0` is genuinely flat (`f_0` is even) and `l >= 2` genuinely vanishes
# (the product goes as `q^l`), so `l = 1` is the only channel affected -- and it
# is in almost every dataset.


def _gamma_states(pseudo_dir, case="si2-nosym", nbnd=10):
    """Converged silicon, then `nbnd` states at Gamma on that density."""
    from defumat import Calculator
    from defumat.system.kpoints import KPoints
    from defumat.workflows.nscf import fixed_density_states

    calculator = Calculator.from_file(
        CASES / f"{case}.in", pseudo_dir=pseudo_dir, announce=False)
    scf = calculator.get_scf(conv_thr=1e-9)
    gamma = KPoints(coords=np.zeros((1, 3)), weights=np.ones(1))
    return fixed_density_states(
        calculator.system, calculator.pseudos, scf.density, kpoints=gamma,
        nbnd=nbnd, conv_thr=1e-11) + (scf.density,)


def _jvp_against_difference(calculation, psi, density, coords, h=5.0e-4):
    """``max|dH/dk by jvp - dH/dk by a frozen-sphere difference|``, and its scale.

    The sphere is frozen on both sides -- ``at_kcart`` replaces the k-point's
    coordinates and keeps the plane waves it was selected with -- so what is
    compared is the same operator twice and not two bases.
    """
    from defumat.system.kpoints import KPoints

    v = VelocityOperator(calculation, calculation.potential(density).v_scf)
    k0 = jnp.asarray(v.kcart)
    code = np.asarray(v.matrix_elements(psi))
    rows = []
    for axis in range(3):
        step = jnp.zeros_like(k0).at[:, axis].set(h)
        moved = (v._operator(psi, k0 + step, overlap=False)
                 - v._operator(psi, k0 - step, overlap=False)) / (2.0 * h)
        rows.append(np.asarray(jnp.einsum("skmg,skng->skmn", psi.conj(), moved)))
    difference = np.stack(rows)
    return code, difference


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason=(
    "gvectors.py:117 -- the l = 1 tangent at k + G = 0 is zeroed, so dH/dk at "
    "Gamma is wrong on every matrix element pairing a state with weight at "
    "G = 0 with one an l = 1 projector sees. Measured on si2-nosym: the "
    "Gamma_1-to-Gamma_15 block comes out at 0.3695 of its value, and the worst "
    "element is 0.13245 Ry bohr out of 1.0775. When this passes the defect is "
    "fixed and PLAN/OPEN must be updated"))
def test_the_velocity_at_gamma_matches_a_frozen_sphere_difference(pseudo_dir):
    """The defect, pinned by the number rather than described.

    **``nbnd`` must reach the conduction bands or this is a null**, and that is
    not a detail: in diamond the occupied manifold at Gamma is ``Gamma_1`` plus
    ``Gamma_25'``, and every matrix element of a *vector* operator among those
    four vanishes by symmetry. Run at the default ``nbnd = 4`` and the whole
    matrix is 1e-8 and so is the difference -- a clean pass that says nothing.
    The term needs one partner with weight at ``G = 0`` (the ``s``-like
    ``Gamma_1``) and one an ``l = 1`` projector sees (the ``p``-like
    ``Gamma_15``), which is a transition across the gap. Measured at
    ``nbnd = 4``: ``max|difference| = 2.65e-7`` and the discrepancy 3.10e-7
    beside it, against 1.0775 and 0.13245 at ``nbnd = 10``.

    The quantity quoted in the mark is the Frobenius norm of the
    ``Gamma_1``-by-``Gamma_15`` block over the three axes, not one entry:
    ``Gamma_15`` is a degenerate triplet and the individual entries are the
    eigensolver's arbitrary basis inside it (rule D4). The block is **0.16957
    against 0.45892**, and the second ``Gamma_1`` state's block is 1.97044
    against 1.88369 -- 4.6 per cent *high* where the first is 63 per cent low,
    because the sign follows the relative phase of ``psi(G = 0)`` and
    ``<beta|psi>``. Silicon's own optical transition, ``Gamma_25'`` to
    ``Gamma_15``, is untouched at **2.33892 either way**: neither partner of it
    has weight at ``G = 0``.
    """
    calculation, _, _, psi, density = _gamma_states(pseudo_dir)
    code, difference = _jvp_against_difference(
        calculation, jnp.asarray(psi), jnp.asarray(density), None)
    assert np.abs(difference).max() > 1.0, "the scale the comparison is against"
    np.testing.assert_allclose(code, difference, atol=1e-6)


@pytest.mark.slow
def test_the_velocity_off_the_reciprocal_lattice_is_right(pseudo_dir):
    """The control that says it is the ``k + G = 0`` row and nothing else.

    ``_TINY`` is on ``|k+G|^2`` at 1e-8, so the guard is the *exact* zero row
    and not a neighbourhood: one k-point away from the reciprocal lattice and
    the same comparison is pure truncation, falling as ``h^2``. Measured
    1.86e-7 at ``h = 2e-3`` against 1.16e-8 at ``5e-4`` on a matrix whose
    largest element is 1.025, where Gamma's 0.15460 does not move with ``h``
    at all.
    """
    from defumat.system.kpoints import KPoints
    from defumat.workflows.nscf import fixed_density_states
    from defumat import Calculator

    calculator = Calculator.from_file(
        CASES / "si2-nosym.in", pseudo_dir=pseudo_dir, announce=False)
    scf = calculator.get_scf(conv_thr=1e-9)
    offset = KPoints(coords=np.array([[0.1, 0.0, 0.0]]), weights=np.ones(1))
    calculation, _, _, psi = fixed_density_states(
        calculator.system, calculator.pseudos, scf.density, kpoints=offset,
        nbnd=10, conv_thr=1e-11)
    code, difference = _jvp_against_difference(
        calculation, jnp.asarray(psi), jnp.asarray(scf.density), None)
    assert np.abs(difference).max() > 1.0
    np.testing.assert_allclose(code, difference, atol=1e-7)
