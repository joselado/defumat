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


@lru_cache(maxsize=4)
def _calculation(case: str, pseudo_dir: Path, origin_tangent: bool = True):
    """No SCF: the claim is about the *operator*, not about a ground state."""
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    return Calculation(system, pseudos, origin_tangent=origin_tangent)


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
def test_the_velocity_at_gamma_matches_a_frozen_sphere_difference(pseudo_dir):
    """The ``l = 1`` tangent at ``k + G = 0``, which used to be zero.

    Both origin guards are right about their own factor and the product is what
    carries the derivative, so the chain rule returned ``Y df + dY f = 0`` where
    the truth is ``sqrt(3/4pi) f_1'(0)``. Before the repair the
    ``Gamma_1``-by-``Gamma_15`` block came out at **0.3695** of its value,
    0.16957 against 0.45892, with the worst entry 0.13245 Ry bohr out of 1.0775
    and **no dependence on the step size**, which is what separates a missing
    term from truncation. After it the comparison behaves like any other
    k-point: 2.11e-7 at ``h = 2e-3`` falling to 1.32e-8 at ``5e-4``.

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
    np.testing.assert_allclose(code, difference, atol=1e-7)


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



def test_the_origin_slope_is_the_transform_of_the_same_table(pseudo_dir):
    """``lim f_l(q)/q^l`` in closed form against the transform it is the limit of.

    The correction's one number is ``f_1'(0)``, and taking it analytically
    rather than by evaluating the transform at some small ``q`` is what keeps it
    from being a second convention: same ``kkbeta`` range, same Simpson weights,
    same prefactor, so the two agree by construction. Checked on the three kinds
    of dataset and for ``l`` up to 2, because ``(2l+1)!!`` and the extra power
    of ``r`` are the two places this can be written down wrong -- and it was
    written down wrong once, with ``r^l`` where ``_beta_kernel`` carries
    ``r^(l+1)``, which reads 0.2465 against 0.2291.
    """
    from defumat.pseudo.formfactors import (
        projector_form_factors, projector_origin_slopes)

    q = np.array([1.0e-5, 2.0e-5, 4.0e-5])
    for name in ("Si.pz-vbc.UPF", "C.pz-rrkjus.UPF", "Pt.pbe-n-kjpaw_psl.0.1.UPF"):
        pseudo = read_upf(pseudo_dir / name)
        volume = 265.302
        closed = np.asarray(projector_origin_slopes(pseudo, volume))
        table = np.asarray(projector_form_factors(pseudo, q, volume))
        for nb, projector in enumerate(pseudo.projectors):
            l = projector.l
            fitted = float(np.polyfit(q, table[nb] / q**l, 1)[1])
            assert closed[nb] == pytest.approx(fitted, abs=2e-9, rel=1e-7), (
                f"{name} channel {nb} (l = {l})")
        assert len(closed) == len(pseudo.projectors)


@pytest.mark.slow
def test_the_correction_adds_exactly_zero_to_every_primal(pseudo_dir):
    """It owns a ``jvp`` rule and nothing else, so no value may move.

    ``_with_origin_tangent``'s primal is ``return columns``; the whole repair
    is its custom ``jvp`` rule. Measured across a norm-conserving, a mixed
    PAW-and-norm-conserving and an ultrasoft cell, the total energy, the
    eigenvalues, the forces and the stress are **bit-identical** before and
    after. The stress is the one that had to be checked rather than argued: it
    differentiates through ``modulus`` with respect to the *cell*, and it stays
    exact because ``k + G = 0`` scales to zero under any strain, so the tangent
    the rule fires on is itself zero there.

    This test is the cheap standing version of that: the primal is compared
    **byte for byte** with what was handed to it, which is the one comparison
    that also separates ``-0.0`` from ``0.0`` and so would catch an earlier
    draft of this that added an array of zeros instead. Beside it, the fact
    that the rows the rule is about exist in this cell at all -- a zero that is
    a statement about an empty set is the trap ``CLAUDE.md`` names.
    """
    from defumat import Calculator
    from defumat.pseudo.projectors import (
        _origin_slopes, _with_origin_tangent, projector_channels)

    calculator = Calculator.from_file(
        CASES / "si2-nosym.in", pseudo_dir=pseudo_dir, announce=False)
    calculation = calculator.calculation
    core = calculation.projector_core
    axes, slopes = _origin_slopes(
        calculator.pseudos,
        [projector_channels(p) for p in calculator.pseudos],
        calculation.system.cell.volume,
    )
    out = np.asarray(_with_origin_tangent(core.columns, core.kg, slopes, axes))
    before = np.asarray(core.columns)
    assert out.shape[-1] == len(axes)
    # **Byte for byte, not ``allclose``.** The primal is ``return columns``,
    # so the claim is identity rather than agreement, and comparing the raw
    # bytes is the one comparison that also separates ``-0.0`` from ``0.0`` --
    # which is exactly what adding a zeros array would have changed.
    assert out.tobytes() == before.tobytes()
    # ...and the rows it is *about* exist in this cell, or the zero above is a
    # statement about an empty set.
    at_origin = np.asarray(np.sum(core.kg * core.kg, axis=-1)) <= 1.0e-8
    assert at_origin.any(), "si2-nosym's unshifted grid has a k + G = 0 row"
    assert any(s != 0 for s in np.asarray(slopes)), "and an l = 1 channel"


def _velocity(case: str, pseudo_dir: Path, origin_tangent: bool):
    """``<psi|dH/dk|psi>`` on a fixed random block, one leg of the switch."""
    calculation = _calculation(case, pseudo_dir, origin_tangent)
    rng = np.random.default_rng(3)
    v_scf = calculation.potential(
        jnp.zeros((calculation.nspin_mag,) + tuple(calculation.basis.dense.grid))
    ).v_scf
    nk = len(calculation.system.kpoints.weights)
    width = calculation.basis.planewaves.npwx * calculation.npol
    psi = jnp.asarray(
        rng.normal(size=(calculation.nspin, nk, 4, width))
        + 1j * rng.normal(size=(calculation.nspin, nk, 4, width))
    )
    return np.asarray(VelocityOperator(calculation, v_scf, None).matrix_elements(psi))


def test_qes_origin_convention_is_reachable_and_changes_only_gamma(pseudo_dir):
    """``origin_tangent=False`` is QE's zero, and it is inert off Gamma.

    QE drops the ``l = 1`` tangent of ``<k+G|beta>`` at ``k + G = 0`` twice
    over -- ``PW/src/commutator_Hx_psi.f90:113-118`` sets ``gk_vpol = 0`` where
    ``g2k < 1.0d-10``, and ``upflib/dylmr2.f90:88-92`` sets ``dg = 0`` where
    ``gg <= eps``, so ``dylm`` goes with it -- while the product
    ``f_1(q) Y_1m(qhat) -> c sqrt(3/4pi) q_m`` is linear in the vector and has
    the gradient ``c sqrt(3/4pi) delta_m,alpha`` there. This code carries the
    term and the flag puts QE's zero back, which is what a ``ph.x`` comparison
    on a Gamma-containing mesh is held to (``o2-fixed-lsda`` and
    ``si10-epsilon``; `PLAN.md` P24 has the numbers).

    **Both halves are asserted, because a switch that did nothing would pass
    the half that matters less.** On ``si2-nosym``, whose mesh holds Gamma, the
    two legs must *differ*; on ``si-epsilon``, whose explicit k-list is shifted
    and holds no ``k + G = 0`` at all, they must be **bit-identical**, which is
    the statement that the flag reaches exactly the row it claims to and no
    other.
    """
    at_gamma = [_velocity("si2-nosym", pseudo_dir, flag) for flag in (True, False)]
    moved = np.abs(at_gamma[0] - at_gamma[1]).max()
    scale = np.abs(at_gamma[0]).max()
    assert moved > 1.0e-3 * scale, (
        f"the flag did nothing on a Gamma-containing mesh: {moved} on {scale}")

    shifted = [_velocity("si-epsilon", pseudo_dir, flag) for flag in (True, False)]
    assert shifted[0].tobytes() == shifted[1].tobytes(), (
        "the flag moved a mesh that has no k + G = 0 in it")
