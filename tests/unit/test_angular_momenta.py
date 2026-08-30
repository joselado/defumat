"""P48b: site-resolved ``<L>``, ``<S>`` and ``<J>`` (Elk tasks 15/16).

Three layers, and the first is where a mistake would actually be:

* the **operators**, which are ``L_x``, ``L_y``, ``L_z`` conjugated out of the
  complex spherical harmonics into the real ones ``ylmr2`` builds. They must be
  Hermitian, **purely imaginary** (a consequence of ``L`` being Hermitian and
  the real harmonics being real, so it is a test and not a convention), obey
  ``[L_x, L_y] = i L_z``, and have ``L^2 = l(l+1)``;
* the **quenching**, which is the headline: ``<L>`` vanishes identically
  without spin-orbit coupling, because nothing in a scalar-relativistic
  Hamiltonian locks the orbital moment to the lattice. Measured, not bounded --
  1.7e-16 on silicon;
* the **refusals**, which are the promise that a run which starts is a run
  whose physics is there.

The physics number -- nickel's orbital moment, and that ``<L>`` rotates with
the magnetization -- is expensive enough to be marked slow.
"""

import numpy as np
import pytest

from pypresso.calculator import Calculator
from pypresso.projwfc.angular_momentum import (
    PAULI,
    angular_momenta,
    orbital_matrices,
)

pytestmark = pytest.mark.unit


_SILICON = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.2, nat = 2, ntyp = 1, ecutwfc = 12.0
  {extra}
/
&electrons
  conv_thr = 1e-10
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS crystal
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 4 4 4 0 0 0
"""


# --- the operators ----------------------------------------------------------


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_the_angular_momentum_operators_obey_their_own_algebra(l):
    matrices = orbital_matrices(l)
    size = 2 * l + 1
    assert matrices.shape == (3, size, size)
    for axis in range(3):
        assert np.allclose(matrices[axis], matrices[axis].conj().T, atol=1e-14)
    # Purely imaginary: <y_i|L|y_j> with real harmonics on both sides.
    assert np.abs(matrices.real).max() < 1.0e-14
    lx, ly, lz = matrices
    assert np.allclose(lx @ ly - ly @ lx, 1j * lz, atol=1e-13)
    assert np.allclose(ly @ lz - lz @ ly, 1j * lx, atol=1e-13)
    assert np.allclose(lz @ lx - lx @ lz, 1j * ly, atol=1e-13)
    casimir = sum(matrices[a] @ matrices[a] for a in range(3))
    assert np.allclose(casimir, l * (l + 1) * np.eye(size), atol=1e-13)


def test_the_pauli_matrices_are_the_ones_dmatls_uses():
    """``xs`` in ``dmatls.f90``, written out as the three sigma matrices."""
    sx, sy, sz = PAULI
    assert np.allclose(sx @ sy - sy @ sx, 2j * sz)
    for sigma in PAULI:
        assert np.allclose(sigma @ sigma, np.eye(2))


def test_an_l_beyond_the_table_is_refused():
    with pytest.raises(ValueError, match="l must be in"):
        orbital_matrices(4)


# --- the quenching ----------------------------------------------------------


@pytest.fixture(scope="module")
def silicon(pseudo_dir):
    text = _SILICON.format(extra=", nosym = .true.")
    calculator = Calculator.from_text(text, pseudo_dir, announce=False)
    return calculator, calculator.get_scf()


def test_the_orbital_moment_is_quenched_without_spin_orbit(silicon):
    """The headline: ``<L>`` is zero, and it is zero to machine precision.

    Nothing in a scalar-relativistic Hamiltonian locks the orbital moment to
    the lattice, so this is an identity rather than a tolerance -- the
    "vanishes pointwise" style of P47's silicon curvature. It is also the check
    that would catch an ``L`` matrix built in the wrong basis, because a wrong
    unitary gives a small non-zero answer and not an obviously wrong one.
    """
    calculator, scf = silicon
    momenta = angular_momenta(calculator.calculation, scf)
    assert np.abs(momenta.orbital).max() < 1.0e-12
    assert np.abs(momenta.spin).max() == 0.0  # nspin = 1 has no spin structure
    assert np.abs(momenta.total).max() < 1.0e-12


def test_the_projected_charge_is_the_valence_count(silicon):
    """Four electrons per silicon, up to what the projector set misses."""
    calculator, scf = silicon
    momenta = angular_momenta(calculator.calculation, scf)
    assert len(momenta.atoms) == 2
    for atom in momenta.atoms:
        assert 3.5 < atom.charge < 4.2
    assert momenta.table().startswith("Angular momenta on the ortho-atomic")


def test_j_is_l_plus_s(silicon):
    calculator, scf = silicon
    momenta = angular_momenta(calculator.calculation, scf)
    for atom in momenta.atoms:
        assert np.allclose(atom.j, atom.l + atom.s)


# --- the refusals -----------------------------------------------------------


def test_a_symmetry_reduced_kset_is_refused(pseudo_dir):
    """``<L>`` is an axial vector and a wedge sum is a wedge sum.

    The escape is the whole grid, unshifted, which is closed under the point
    group -- the same one ``dielectric_tensor`` documents. Refusing is what
    keeps a plausible, smooth, wrong vector from being returned.
    """
    calculator = Calculator.from_text(
        _SILICON.format(extra=""), pseudo_dir, announce=False
    )
    scf = calculator.get_scf()
    with pytest.raises(NotImplementedError, match="symmetry-reduced k-set"):
        angular_momenta(calculator.calculation, scf)


def test_an_unknown_projector_set_is_refused(silicon):
    calculator, scf = silicon
    with pytest.raises(ValueError, match="unknown projector set"):
        angular_momenta(calculator.calculation, scf, kind="muffin-tin")


def test_the_calculator_reaches_it():
    """P38's rule: a new entry point gets a facade method in the same pass."""
    import ast
    import inspect
    import textwrap

    assert hasattr(Calculator, "get_angular_momenta")
    source = textwrap.dedent(inspect.getsource(Calculator.get_angular_momenta))
    body = ast.parse(source).body[0].body
    if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    code = "\n".join(ast.unparse(node) for node in body)
    assert "angular_momenta(" in code
    assert "np." not in code and "einsum" not in code


# --- the physics, which costs a spinor SCF ----------------------------------


_NICKEL = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 6.65, nat = 1, ntyp = 1, ecutwfc = 60.0,
  occupations = 'smearing', smearing = 'mv', degauss = 0.02, nosym = .true.,
  noncolin = .true., lspinorb = .true., starting_magnetization(1) = 0.3,
  angle1(1) = {angle1}, angle2(1) = {angle2}
/
&electrons
  conv_thr = 1e-8, mixing_beta = 0.3
/
ATOMIC_SPECIES
 Ni 58.69 Ni.rel-pbe-nc-dojo.UPF
ATOMIC_POSITIONS crystal
 Ni 0.00 0.00 0.00
K_POINTS automatic
 4 4 4 0 0 0
"""


def _nickel(pseudo_dir, angle1, angle2=0.0):
    text = _NICKEL.format(angle1=angle1, angle2=angle2)
    calculator = Calculator.from_text(text, pseudo_dir, announce=False)
    scf = calculator.get_scf(conv_thr=1e-10)
    return angular_momenta(calculator.calculation, scf), scf


@pytest.mark.slow
def test_nickel_has_an_orbital_moment_when_spin_orbit_is_on(pseudo_dir):
    """The physics this feature exists for.

    ``<L_z>`` is **0.0365** hbar on a fully-relativistic norm-conserving nickel,
    against a measured orbital moment of about 0.05 mu_B -- the underestimate
    GGA is known for. What makes it a check rather than a number is the ratio:
    ``|L|/|S| = 0.11665`` against an experimental ``m_L/m_S`` of about 0.1, and
    the same quantity is identically zero without the coupling.
    """
    momenta, scf = _nickel(pseudo_dir, angle1=0.0)
    orbital, spin = momenta.orbital[0], momenta.spin[0]
    assert 0.02 < orbital[2] < 0.06
    assert abs(orbital[0]) < 1.0e-4 and abs(orbital[1]) < 1.0e-4
    assert 0.05 < np.linalg.norm(orbital) / np.linalg.norm(spin) < 0.20
    # <L> is parallel to <S>, which is the sign of the spin-orbit coupling in a
    # more-than-half-filled shell (Hund's third rule).
    assert orbital @ spin > 0.0


@pytest.mark.slow
def test_the_orbital_moment_rotates_with_the_magnetization(pseudo_dir):
    """``|<L>|`` is a scalar and nothing here imposes that it behaves like one.

    The moment is driven along ``z``, ``x`` and ``y`` in turn; the magnitude
    must be the same in all three -- the three cubic axes are equivalent by
    symmetry -- and ``<L>`` must follow ``<S>``. Measured: **0.0364767** in each,
    a spread of 7.3e-11 over the three axes, with ``L.S/|L||S| = 1.00000000``.

    The threshold matters and is why this converges to 1e-10: at 1e-8 the same
    spread is 1e-5, which is the SCF's own scatter and not the symmetry.
    """
    magnitudes = []
    for angle1, angle2 in ((0.0, 0.0), (90.0, 0.0), (90.0, 90.0)):
        momenta, _ = _nickel(pseudo_dir, angle1, angle2)
        orbital, spin = momenta.orbital[0], momenta.spin[0]
        cosine = orbital @ spin / (np.linalg.norm(orbital) * np.linalg.norm(spin))
        assert abs(cosine - 1.0) < 1.0e-6, (angle1, angle2, cosine)
        magnitudes.append(np.linalg.norm(orbital))
    assert np.ptp(magnitudes) < 1.0e-9, magnitudes


@pytest.mark.slow
def test_a_negative_occupation_enters_with_its_sign(pseudo_dir):
    """``wg`` is a weight, not a magnitude, and Methfessel-Paxton makes it negative.

    The site matrix was once accumulated with ``sqrt(w_a w_b)``, which is
    ``|w|``: identical until a smearing pushes an occupation below zero, and
    then that band enters ``rho`` with the **wrong sign**. Silent, small, and
    invisible to the rotation check above, because it is systematic across
    orientations. Methfessel-Paxton on this nickel puts **486 of 1664**
    occupations below zero, which is what makes it the case that catches it.

    The assertion is the inequality a projection cannot violate: an orthonormal
    orbital set can only *lose* weight relative to the electron count, never
    gain it, so ``sum_a Tr rho_a <= nelec``. With ``|w|`` the projected charge
    is inflated by twice the negative weight and the inequality fails.
    """
    text = _NICKEL.format(angle1=0.0, angle2=0.0).replace(
        "smearing = 'mv'", "smearing = 'mp'"
    )
    calculator = Calculator.from_text(text, pseudo_dir, announce=False)
    scf = calculator.get_scf(conv_thr=1e-8)

    occupations = np.asarray(scf.occupations)
    assert (occupations < 0.0).sum() > 100, "this case is meant to have them"

    momenta = angular_momenta(calculator.calculation, scf)
    nelec = calculator.calculation.nelec
    charge = sum(atom.charge for atom in momenta.atoms)
    assert charge <= nelec + 1.0e-8, (charge, nelec)
    # ...and the spilling is small, so this is not passing by losing everything.
    assert charge > 0.95 * nelec


@pytest.mark.slow
def test_a_relativistic_ultrasoft_dataset_is_refused(pseudo_dir):
    """``qq_so``: the spinor overlap's off-diagonal spin blocks are not applied.

    The projection here applies the *scalar* ``S`` to each spinor component,
    which is ``projwfc.x``'s validated path in every other regime and is missing
    a term in this one -- so it is named rather than silently run. Platinum
    because its relativistic ultrasoft dataset is already committed; the nickel
    one is 3.4 MB and nothing else needs it.
    """
    platinum = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 7.42, nat = 1, ntyp = 1, ecutwfc = 30.0,
  ecutrho = 250.0, occupations = 'smearing', smearing = 'mv', degauss = 0.02,
  nosym = .true., noncolin = .true., lspinorb = .true.
/
&electrons
  conv_thr = 1e-6
/
ATOMIC_SPECIES
 Pt 195.08 Pt.rel-pz-n-rrkjus.UPF
ATOMIC_POSITIONS crystal
 Pt 0.00 0.00 0.00
K_POINTS automatic
 2 2 2 0 0 0
"""
    calculator = Calculator.from_text(platinum, pseudo_dir, announce=False)
    scf = calculator.get_scf()
    with pytest.raises(NotImplementedError, match="qq_so"):
        angular_momenta(calculator.calculation, scf)
