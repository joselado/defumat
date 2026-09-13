"""``epsilon_infinity`` and the Born charges of a spinor run.

P81 opened the Sternheimer **solve** for ``noncolin = .true.`` and stopped
there: every assembly above it still went through
:func:`~defumat.response.sternheimer.require_a_sternheimer_regime` without the
opt-in, so no user-facing quantity changed. This is the first assembly to ask
for it, and asking found three places that were collinear and had never been
reached:

* :func:`defumat.response.born._raw_mixed_state`'s own density builder, which
  is a *local* copy of the SCF's and was the collinear ``sum_band``. A
  ``2 npwx``-long spinor does not give a wrong number there, it fails to
  broadcast -- which is how the site was found;
* :func:`defumat.forces.energy.frozen_energy`'s refusal of the matrix
  orthonormality multipliers for a spinor, which was a statement about the
  **metric** (``qq_so`` against the scalar ``qq``) and not about the spinor. A
  norm-conserving spinor has ``S = 1`` and never reaches that term;
* the same function's ``spinors`` opt-in, which
  :func:`defumat.response.born.born_effective_charges` now asks for
  deliberately, as :func:`defumat.forces.energy.reject_spinors` requires.

**The anchors are two identities and one external number.**

1. The same silicon run as a scalar and as a spinor with no magnetization is the
   same physics on a doubled space -- four bands of two electrons against eight
   of one -- so ``epsilon`` and ``Z*`` must be the same numbers. That is the
   check on the bookkeeping every ``KPoints`` constructor's unconditional
   ``degspin`` reaches: the k-point weights, the occupied-band count and the
   density.
2. The symmetrised wedge against the closed grid, as a spinor. Nothing is
   shared between the two routes except the solve, so agreeing is the check on
   :meth:`~defumat.scf.driver.Calculation.symmetrize_directional` reached from
   this assembly.
3. ``ph.x``'s own 13.806689470, which the spinor route has to reproduce as well
   as the scalar one does.

**What the identity is measured at, and why not tighter.** At
``conv_thr = 1e-8`` and ``1e-10`` the two routes agree to **2.1e-14** and
**5.0e-14**; at ``1e-12`` they sit **1.35e-7** apart and at ``1e-14``
**9.8e-9**, which is not a trend and not a route difference. Three controls say
so, and together they are what stops "the two ground states differ a little"
from being an explanation accepted because it fits:

* with four empty bands present the two routes agree at ``conv_thr = 1e-12`` to
  **every digit printed** -- 13.806634668362 on both sides;
* the *scalar* run alone moves by **2.7e-7** under a change of ``k_batch``,
  which is a summation order and not physics, so the scatter exists without a
  spinor anywhere near it;
* ``epsilon`` itself wanders by **3e-5** between ``conv_thr = 1e-10`` and
  ``1e-14``, which is the same order as this cell's whole disagreement with
  ``ph.x`` (4.3e-5). The identity is satisfied far below the level the quantity
  is determined at, at every threshold.

The nbnd sensitivity (1.1e-5 between no empty bands and four) is the **scalar**
code's and predates this work; it is recorded here because it is what sets the
floor the identity is read against, not because P83 introduced it.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.response.efield import dielectric_tensor
from defumat.scf import run_scf
from defumat.system import build_system

pytestmark = [pytest.mark.regression]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: What the vendored ``ph.x`` prints for norm-conserving silicon, which
#: ``test_response.py`` takes as the scalar reference. The spinor route has to
#: reproduce it as well as the scalar one does, and to the same tolerance: what
#: is left on both sides is QE's ``dq = 0.01`` radial form-factor table against
#: a direct integration here.
QE_DIELECTRIC = 13.806689470
QE_BORN = -0.07571
EPSILON_TOLERANCE = 5e-4
BORN_TOLERANCE = 1e-4

#: How far the scalar and spinor routes may sit apart at ``conv_thr = 1e-10``.
#: Measured 5.0e-14 in ``epsilon`` and 3.0e-15 in ``Z*``; this is two orders of
#: room on the first. **Not asserted at 1e-12**, where the same identity would
#: need 1e-6 and would stop discriminating anything (see the module docstring).
IDENTITY = 1e-12

#: The wedge against the closed grid, as a spinor: measured 7.4e-13, against the
#: scalar pair's own 4.1e-13 on the same cell.
WEDGE = 5e-12


@lru_cache(maxsize=2)
def _silicon(noncolin: bool, conv_thr: float, nbnd=None):
    """``si-epsilon`` with the field response solved on top.

    The spinor run is the *same input file* with one line added rather than a
    second committed input, so the identity cannot be weakened by the two sides
    drifting apart. ``Si.pz-vbc.UPF`` is not relativistic and nothing seeds a
    moment, so ``nspin_mag`` stays 1.

    ``maxsize = 2`` and never ``None``: the response holds the wavefunctions and
    three first-order responses beside them, and an unbounded cache of those is
    the accumulation that has killed test files on this machine.
    """
    from defumat.scf import Calculation

    parsed = read_pw_input(CASES / "si-epsilon.in")
    if noncolin:
        parsed.namelists["system"]["noncolin"] = True
    if nbnd is not None:
        parsed.namelists["system"]["nbnd"] = nbnd
    system = build_system(parsed)
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    assert calculation.noncolin == noncolin
    result = run_scf(system, pseudos, calculation=calculation,
                     conv_thr=conv_thr, max_iterations=80)
    assert result.converged
    response = dielectric_tensor(
        calculation, result.wavefunctions, result.eigenvalues, result.density,
        result.becsum, born_charges=True,
    )
    assert response.converged
    return calculation, result, response


def test_a_spinor_with_no_magnetization_gives_the_scalar_dielectric_tensor():
    """The identity that catches a factor of two in the spin sum.

    Every ``KPoints`` constructor applies the unpolarized ``degspin``
    unconditionally and a spinor band holds **one** electron, so the bookkeeping
    reaches the k-point weights, the occupied-band count and the density -- and
    a factor of two anywhere along it is 100 per cent in ``epsilon``.

    ``Z*`` is the sharper half. Silicon's is zero by symmetry, so what is
    compared is a **residue** of about 4 against an electronic part near 4.076,
    and it is the only quantity here that goes through
    :func:`defumat.response.born.born_effective_charges` -- the whole
    ``jax.grad`` of the frozen energy through moving atoms, with the matrix
    multipliers that were refused for a spinor until this phase.

    Measured at ``conv_thr = 1e-10``: **5.0e-14** in ``epsilon`` and
    **3.0e-15** in ``Z*``, with the two total energies 1.8e-15 Ry apart.
    """
    _, scalar_scf, scalar = _silicon(False, 1e-10)
    _, spinor_scf, spinor = _silicon(True, 1e-10)

    assert float(spinor_scf.total_energy) == pytest.approx(
        float(scalar_scf.total_energy), abs=1e-12
    )
    assert float(spinor.isotropic) == pytest.approx(
        float(scalar.isotropic), abs=IDENTITY
    )
    assert np.abs(
        np.asarray(spinor.epsilon) - np.asarray(scalar.epsilon)
    ).max() < IDENTITY
    assert np.abs(
        np.asarray(spinor.born_charges) - np.asarray(scalar.born_charges)
    ).max() < IDENTITY


@pytest.mark.slow
def test_the_spinor_dielectric_constant_matches_quantum_espresso():
    """The external number, at the threshold the scalar test uses.

    A nonmagnetic spinor silicon is the same physics as the scalar run, so this
    is not an independent measurement of the physics -- it is the statement that
    the spinor *route* reaches ``ph.x`` as well as the scalar route does, which
    is what a user asking for ``get_dielectric_tensor()`` on a spin-orbit run
    is promised. Measured 13.806645970 against 13.806689470.
    """
    _, _, spinor = _silicon(True, 1e-12)
    assert float(spinor.isotropic) == pytest.approx(
        QE_DIELECTRIC, abs=EPSILON_TOLERANCE
    )
    assert float(np.asarray(spinor.born_charges)[0, 0, 0]) == pytest.approx(
        QE_BORN, abs=BORN_TOLERANCE
    )


@pytest.mark.slow
def test_the_symmetrised_wedge_and_the_closed_grid_agree_for_a_spinor():
    """``symmetrize_directional`` reached from this assembly, as a spinor.

    ``si-epsilon-unshifted`` is 8 k-points reduced from an **unshifted** 4x4x4
    grid with 48 operations, and ``si-epsilon-unshifted-nosym`` is the whole 64
    of them with no symmetry at all. An unshifted grid *is* closed under the
    point group -- a shifted one is not, which is what
    :func:`~defumat.response.efield.require_a_symmetrisable_response` refuses --
    so the two routes must give one tensor, and they share nothing but the
    solve.

    ``nbnd`` is set above the occupied count on both sides deliberately: with no
    empty band the two land 1e-7 apart for a reason that has nothing to do with
    symmetry (see the module docstring), and that would be read here as a
    symmetriser error.

    Measured **7.4e-13**, against the scalar pair's own 4.1e-13 on the same cell.
    """
    from defumat.scf import Calculation

    tensors = []
    for case in ("si-epsilon-unshifted", "si-epsilon-unshifted-nosym"):
        parsed = read_pw_input(CASES / f"{case}.in")
        parsed.namelists["system"]["noncolin"] = True
        parsed.namelists["system"]["nbnd"] = 16
        system = build_system(parsed)
        pseudos = tuple(
            read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
        )
        calculation = Calculation(system, pseudos)
        result = run_scf(system, pseudos, calculation=calculation,
                         conv_thr=1e-12, max_iterations=80)
        assert result.converged
        response = dielectric_tensor(
            calculation, result.wavefunctions, result.eigenvalues,
            result.density, result.becsum, born_charges=False,
        )
        assert response.converged
        tensors.append(np.asarray(response.epsilon))

    wedge, closed = tensors
    assert np.abs(wedge - closed).max() < WEDGE
