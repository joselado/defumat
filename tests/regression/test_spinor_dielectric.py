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
3. ``ph.x``'s own number, and it has to be ``ph.x``'s *spinor* one. QE run as a
   spinor on this cell gives **13.806615123** against **13.806689470** as a
   scalar, so its own scalar-against-spinor identity is **7.4e-5** where this
   code's is 5.0e-14. Against the like-for-like number this code sits 3.1e-5
   away, tighter than the 4.3e-5 the two *scalar* runs sit at. The bound below
   is loose enough to hold either pairing, which is deliberate: it is the
   ``dq = 0.01`` radial-table floor and not a statement about the spin axis.

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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.response.efield import dielectric_tensor
from defumat.scf import run_scf
from defumat.system import build_system


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """``jax.clear_caches()`` between tests, for ``CLAUDE.md``'s reason.

    Seventy of this file's failures on a cluster node were
    ``INTERNAL: Failed to materialize symbols``, the second-worst count in the
    whole ``slow`` set, and a spinor dielectric constant compiles the response
    stack on a doubled space for every cell it visits.

    The results stay cached and only the compiled executables are dropped, which
    trades recompilation against both the resident set and the process's count
    of virtual-memory mappings -- ``vm.max_map_count`` is 65530 on an ordinary
    node and XLA maps every compiled executable anonymously. Measured on a
    Berry-phase loop, one ``jax.clear_caches()`` released 8924 mappings where
    ``gc.collect()`` released none, so this does reach them (``OPEN.md``
    Part XIII item 2).

    **What it cannot do is help a single test that exhausts them on its own**,
    because a fixture with a ``yield`` fires between tests. Where one test is
    the offender the clear has to go inside its loop, which is what
    ``run_polarization``'s ``clear_caches`` does, or the test has to run in a
    process of its own -- ``tools/run_regression.sh`` takes node IDs for that.
    """
    yield
    jax.clear_caches()


pytestmark = [pytest.mark.regression]

#: The three refusal tests are the only ones in the gate. The whole file is
#: **3m43s and 3966 M peak** (measured), of which the three self-consistent
#: responses are 215 s -- a 21 per cent surcharge on a 7-minute gate for one
#: test, which is what `slow` is the line for. Cost decides it, not importance:
#: the identity below is the central claim of the phase and is still `slow`.

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


@pytest.mark.slow
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
    is promised. Measured 13.806645970, against ``ph.x``'s own spinor run's
    13.806615123 (3.1e-5) and its scalar run's 13.806689470 (4.3e-5).
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


def test_a_textured_spinor_is_refused_by_name():
    """``nspin_mag = 4`` runs and is not trusted, so it is refused. P83.

    The refusal is unusual here in that nothing is *missing*. The assembly
    produces a tensor that passes two internal checks on ``i-atom-soc.in`` --
    uniaxial along the moment with nothing imposing it, and the distinct axis
    following the moment to nine digits when the moment is turned -- and then
    disagrees with ``ph.x`` by **5.3 per cent** in the component along the
    moment, on a ground state the two codes agree on to the printed digit.

    The three thresholds ``dmxc_nc`` has that a ``jvp`` of ``v_of_rho`` does not
    were each measured on that cell's converged density and each fires at
    **zero** of 157464 grid points, so it is not a convention at an edge the way
    P70's ``|zeta| >= 1`` turned out to be. It is unlocated, and a number that
    looks like a working calculation and is 5 per cent out is exactly what a
    refusal is for.
    """
    from defumat.scf import Calculation

    system = build_system(read_pw_input(CASES / "i-atom-soc.in"))
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    # The cell has to *be* the refused regime, or the test passes for the wrong
    # reason: a guard that cannot fire reads the same as one that did not need to.
    assert calculation.noncolin and calculation.nspin_mag == 4
    assert system.lspinorb and not calculation.is_ultrasoft

    with pytest.raises(NotImplementedError, match="textured"):
        dielectric_tensor(
            calculation, jnp.zeros((1, 1, 1, 1)), jnp.zeros((1, 1, 1)),
            jnp.zeros((4, 1, 1, 1)),
        )


#: What the vendored ``ph.x`` prints for the fully-relativistic ultrasoft AlAs
#: (``alas-epsilon-us-soc.ph.in``, committed as
#: ``reference.out.ph-alas-epsilon-us-soc``). The scalar-relativistic cell's own
#: number is 9.520257751, so spin-orbit coupling is worth **8.6e-3** here --
#: 245 times the residual below, which is what makes the agreement a statement
#: about the coupling rather than about the crystal.
QE_RELATIVISTIC_EPSILON = 9.528846009
QE_RELATIVISTIC_BORN = (2.10114, -2.16587)


@lru_cache(maxsize=2)
def _augmented(case: str, noncolin: bool):
    """One of the augmented cases, converged and solved.

    ``maxsize = 2`` for the reason :func:`_silicon` gives: the response holds
    the wavefunctions and three first-order responses beside them.
    """
    from defumat.scf import Calculation

    parsed = read_pw_input(CASES / f"{case}.in")
    if noncolin:
        parsed.namelists["system"]["noncolin"] = True
    system = build_system(parsed)
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    assert calculation.noncolin == (noncolin or system.noncolin)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-12,
                     max_iterations=100)
    assert result.converged
    response = dielectric_tensor(
        calculation, result.wavefunctions, result.eigenvalues, result.density,
        result.becsum, born_charges=False,
    )
    assert response.converged
    return calculation, result, response


@pytest.mark.slow
@pytest.mark.parametrize("case", ["si-epsilon-us", "si-epsilon-paw"])
def test_an_augmented_spinor_gives_the_scalar_run_s_dielectric_tensor(case):
    """The identity, one dataset at a time: ultrasoft and then PAW.

    The same file with ``noncolin = .true.`` added, on a dataset that is not
    relativistic and a cell that seeds no moment, so ``nspin_mag`` stays 1 and
    the two runs are the same physics on a doubled space. Every term this phase
    wrote is on the path -- the spinor ``int3`` contraction, the spinor position
    operator and the dipole's spin blocks -- so a shape error or a dropped
    component shows here at 100 per cent.

    Measured: **9.2e-14** on ultrasoft and **1.0e-13** on PAW, with the total
    energies 0 and 1.4e-14 Ry apart.

    **What this cannot see, and it is the reason
    :func:`test_the_relativistic_ultrasoft_dielectric_constant_matches_quantum_espresso`
    exists.** A scalar-relativistic dataset has ``fcoef = 1``, so ``qq_so`` is
    block diagonal, ``dpqq_so`` is the scalar dipole on both spin blocks and the
    recombination inside ``int3`` collapses to the collinear one. The identity
    therefore exercises all three terms and *distinguishes* none of them: it
    would pass with the ``fcoef`` sandwich deleted from every one.
    """
    scalar = _augmented(case, False)[2].isotropic
    spinor = _augmented(case, True)[2].isotropic
    print(f"\n{case}: scalar {scalar:.12f}  spinor {spinor:.12f}")
    assert abs(scalar - spinor) < 1e-12


@pytest.mark.slow
def test_the_relativistic_ultrasoft_dielectric_constant_matches_quantum_espresso():
    """``epsilon_infinity`` of an augmented **spinor**, against the vendored ``ph.x``.

    ``alas-epsilon-us-soc`` is ``alas-epsilon-us`` with the two
    fully-relativistic files in place of the scalar ones and nothing else
    changed, so ``fcoef`` is not the identity and none of the three spin-space
    objects collapses: the overlap's ``qq_so``, the augmentation dipole's
    ``dpqq_so`` and the recombination inside ``int3``.

    Measured: **9.528810788** against ``ph.x``'s **9.528846009**, 3.5e-5, beside
    the ground-state total energy at -25.564414818 Ry against -25.56441482. That
    residual is the same ``dq = 0.01`` radial-table floor the scalar cases sit
    at (4.3e-5, 5.2e-5, 3.4e-5, 1.2e-4).
    """
    calculation, result, response = _augmented("alas-epsilon-us-soc", False)
    assert calculation.is_ultrasoft and any(p.has_so for p in calculation.pseudos)
    print(f"\nrelativistic AlAs: {response.isotropic:.12f} against "
          f"{QE_RELATIVISTIC_EPSILON}")
    assert response.isotropic == pytest.approx(
        QE_RELATIVISTIC_EPSILON, abs=5e-4
    )


def test_an_augmented_spinor_is_no_longer_refused():
    """The dataset half of the edge, which is gone (P98).

    ``set_int3_nc`` was named as the missing object and it is not written here
    either: the ``jvp`` of :meth:`~defumat.scf.driver.Calculation.coefficients`
    passes through the noncollinear recombination, whose ``fcoef`` sandwich is
    linear, so the tangent comes out already dressed. All three datasets pass
    the guard now, and what still refuses is the *texture*, one line below.

    The cells are insulators on purpose. A spinor ultrasoft **metal** -- the
    platinum cells this test used to refuse -- is still refused, by the guard
    about metals rather than the one about datasets, which is the edge that
    moved rather than the one that went.
    """
    from defumat.scf import Calculation
    from defumat.response.sternheimer import require_a_sternheimer_regime

    for case in ("si-epsilon", "si-epsilon-us", "si-epsilon-paw",
                 "alas-epsilon-us-soc"):
        parsed = read_pw_input(CASES / f"{case}.in")
        if not case.endswith("-soc"):
            parsed.namelists["system"]["noncolin"] = True
        system = build_system(parsed)
        pseudos = tuple(
            read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
        )
        calculation = Calculation(system, pseudos)
        assert calculation.noncolin
        require_a_sternheimer_regime(
            calculation, spin_polarized=True, noncollinear=True
        )


def test_the_opt_in_is_what_lifts_the_refusal():
    """Every *other* assembly still refuses a spinor, and must.

    P81's refusal is opt-in for the reason this project has paid for twice: a
    refusal belongs to a machine, and a caller that has not been measured in a
    regime must not inherit permission from one that has. The phonons and the
    Raman tensor go through the same guard without the flag.
    """
    from defumat.scf import Calculation
    from defumat.response.sternheimer import require_a_sternheimer_regime

    parsed = read_pw_input(CASES / "si-epsilon.in")
    parsed.namelists["system"]["noncolin"] = True
    system = build_system(parsed)
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    with pytest.raises(NotImplementedError, match="noncollinear or spin-orbit"):
        require_a_sternheimer_regime(calculation, spin_polarized=True)
