"""P62b, P62c and P62d: the DFT+U axes beyond a scalar `U` on a collinear cell.

Neither has a reference to compare against -- QE computes no around-mean-field
double counting and no Slater integral at all, and an all-electron code's
numbers are not comparable to a pseudopotential's. So these are **whole-SCF
property tests**: each asserts something about the converged run that would be
false if the flavour were quietly not in force, and the sharp checks on the
functionals themselves live in ``tests/unit/test_hubbard_full.py`` and
``tests/unit/test_hubbard_slater.py``.

* ``feo-amf`` -- antiferromagnetic FeO at ``U = 4.3``, ``J = 1.0`` eV with
  ``hubbard_double_counting = 'amf'``, against the committed ``feo-kind1-J``
  run of the same cell under the fully-localised limit. AMF's whole point is
  that it does not correct a uniformly filled shell, so its occupation matrix
  must end up **closer to uniform** and its Hubbard energy far smaller. Both are
  statements about the physics rather than about a number.
* ``bn-ldau-noncol`` -- relativistic BN with ``noncolin`` and ``lspinorb``, the
  one case here that *does* have a reference: the occupation matrix as a 2x2
  matrix in spin space, against a ``pw.x`` run generated for it.
* ``pt-yukawa`` -- fcc platinum with ``hubbard_slater = 'yukawa'``, where the
  input gives a ``U`` and the screening length, ``F^2``, ``F^4`` and ``J`` are
  all computed from the 5d all-electron partial wave. The assertion is that
  ``F^0`` comes back as the ``U`` that was asked for, which is the round trip
  through the whole input path rather than through the solver alone.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from defumat.io import read_qe_output
from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.scf import run_scf
from defumat.scf.driver import Calculation
from defumat.system import build_system
from defumat.units import RY_TO_EV

pytestmark = [pytest.mark.regression, pytest.mark.slow]

GENERATED = Path(__file__).resolve().parents[1] / "data" / "qe"


@lru_cache(maxsize=3)
def _converged(name, pseudo_dir):
    pwin = read_pw_input(GENERATED / name)
    system = build_system(pwin)
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    result = run_scf(
        system, pseudos, calculation=calculation, conv_thr=1e-9,
        max_iterations=250,
        mixing_beta=float(pwin.get("electrons", "mixing_beta", 0.7)),
    )
    return calculation, result


@pytest.fixture(autouse=True)
def _bound_compilation_cache():
    """Three distinct cells in one process; see `CLAUDE.md` on memory."""
    yield
    import jax

    jax.clear_caches()


def test_around_mean_field_converges_and_barely_corrects(pseudo_dir):
    _, amf = _converged("feo-amf.in", pseudo_dir)
    assert amf.converged
    # FLL on the same cell gives +0.232 Ry; AMF is two orders smaller, which is
    # the whole difference between the two double countings.
    assert abs(amf.energy_terms["hubbard"]) < 0.02


def test_around_mean_field_leaves_the_shell_more_uniform(pseudo_dir):
    """The physical statement, against the fully-localised limit on the same cell.

    FLL drives the occupations toward 0 and 1 -- that is what notebook 13 shows
    and what makes it a gap opener. AMF corrects only the *deviation* from the
    mean, so the same cell keeps a shell closer to uniform: its spread of
    eigenvalues about their mean must be smaller.
    """
    _, amf = _converged("feo-amf.in", pseudo_dir)
    _, fll = _converged("feo-kind1-J.in", pseudo_dir)

    def spread(result):
        ns = np.asarray(result.ns)
        return float(
            np.mean([
                np.std(np.linalg.eigvalsh(block))
                for channel in ns for block in channel
            ])
        )

    assert spread(amf) < spread(fll)


def test_the_force_under_the_full_functional_is_the_energy_gradient(pseudo_dir):
    """Rule D5 for ``kind = 1``, and ``pw.x`` cannot do this at all.

    ``force_hub.f90`` stops on *"forces in the DFT+U+J scheme are not
    implemented"*, so there is no reference and a central difference of the SCF
    energy is the whole check. The force here is ``jax.grad`` of the energy
    through projectors that move with the atoms, so nothing about ``kind = 1``
    had to be written for it -- which is exactly why it needs testing.

    **The two displaced runs must be seeded from the undisplaced one.** With a
    full interaction matrix the DFT+U landscape has more than one solution, and
    a cold-started difference silently samples two of them: on this cell the
    displaced runs converge to the *symmetric* state, whose two nickel atoms
    have identical occupations where the undisplaced ground state's differ
    (4.915/4.344 against 4.935/4.174). The resulting "force" is -0.245, -0.005
    and +0.054 Ry/bohr at three step sizes -- not noise around an answer, three
    different answers. Seeded, the same three steps give -0.00794, -0.00749 and
    -0.00749 against an analytic -0.00736.
    """
    system, pseudos, calculation, result = _force_case(pseudo_dir)
    from defumat.forces import compute_forces

    forces = np.asarray(compute_forces(calculation, result).forces)

    def energy_at(positions):
        moved = calculation.at_positions(np.asarray(positions))
        return run_scf(
            moved.system, pseudos, calculation=moved, conv_thr=1e-12,
            mixing_beta=0.3, max_iterations=300, starting_from=result,
        ).total_energy

    step = 2.0e-3
    origin = np.asarray(system.structure.positions)
    plus, minus = origin.copy(), origin.copy()
    plus[0, 0] += step
    minus[0, 0] -= step
    difference = -(energy_at(plus) - energy_at(minus)) / (2.0 * step)
    assert difference == pytest.approx(forces[0, 0], abs=3e-4)


@lru_cache(maxsize=1)
def _force_case(pseudo_dir):
    pwin = read_pw_input(GENERATED / "ni-kind1-force.in")
    system = build_system(pwin)
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    result = run_scf(
        system, pseudos, calculation=calculation, conv_thr=1e-12,
        mixing_beta=0.3, max_iterations=300,
    )
    assert result.converged
    return system, pseudos, calculation, result


def test_a_spinor_occupation_matrix_matches_pw_x(pseudo_dir):
    """P62b: DFT+U on a two-component spinor, with spin-orbit coupling.

    Relativistic BN with ``U`` on the nitrogen ``2p``, ``noncolin`` and
    ``lspinorb``, against a ``pw.x`` run generated for it. The occupation matrix
    is one 2x2 matrix in spin space rather than two real matrices, and it is
    measured on projector columns that are themselves spinors.

    **It runs with ``nosym``**, which the input carries, because the group
    average of a spinor occupation matrix needs the ``SU(2)`` representation of
    each operation beside the rotation of the ``m`` indices; that is refused by
    name. The grid is unshifted 3x3x1, so the two codes sample the same
    k-points either way.
    """
    calculation, result = _converged("bn-ldau-noncol.in", pseudo_dir)
    reference = read_qe_output(GENERATED / "reference.out.bn-ldau-noncol")
    assert result.converged
    assert np.asarray(result.ns).shape[0] == 4
    assert np.iscomplexobj(np.asarray(result.ns))
    assert result.total_energy == pytest.approx(reference.total_energy, abs=1e-6)
    assert result.energy_terms["hubbard"] == pytest.approx(
        reference.energy_terms["hubbard"], abs=1e-6
    )
    # ``Tr[ns]`` per diagonal spin block, against the five decimals ``write_ns``
    # prints: 2.14903 / 2.14903.
    (up, down, total), = result.hubbard_occupations.values()
    assert (up, down) == pytest.approx((2.14903, 2.14903), abs=1e-4)


def test_a_spinor_occupation_matrix_is_hermitian(pseudo_dir):
    """The matrix a spinor run measures is an operator, not two numbers."""
    _, result = _converged("bn-ldau-noncol.in", pseudo_dir)
    ns = np.asarray(result.ns).reshape(2, 2, -1, 3, 3)
    assert ns == pytest.approx(np.conj(np.einsum("stnab->tsnba", ns)), abs=1e-12)


def test_the_full_functional_on_a_spinor_is_refused(pseudo_dir):
    """The composition of two validated axes that does not close.

    Measured rather than assumed, which is why it is a refusal and not a gap:
    on QE's own ``lda+U_kind1_noncollin`` case the two codes are 4.2e-4 Ry apart
    with both converged and both on the same solution.
    """
    from defumat.scf.driver import Calculation

    pwin = read_pw_input(GENERATED / "fe-kind1-noncol.in")
    system = build_system(pwin)
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    with pytest.raises(NotImplementedError, match="noncolin"):
        Calculation(system, pseudos)


def test_the_computed_interaction_reproduces_the_requested_u(pseudo_dir):
    """The whole input path: a ``U`` in, a screening length and ``F^k`` out.

    ``F^0`` is what the root find matched, so it must come back at the card's
    value exactly; ``F^2``, ``F^4`` and ``J`` are *results* and are only
    asserted to be positive and ordered, since nothing outside the package has
    a number for platinum's.
    """
    calculation, result = _converged("pt-yukawa.in", pseudo_dir)
    assert result.converged
    setup = calculation.hubbard
    species = setup.species[setup.types[0]]
    kind, cutoff, norm, lam = species.provenance
    assert kind == "all-electron"
    assert 0.7 < norm < 1.0
    assert 0.0 < cutoff < 5.0
    assert lam > 0.0
    f = species.slater
    assert f[0] * RY_TO_EV == pytest.approx(4.0, abs=1e-9)
    assert f[2] > f[4] > 0.0
    assert species.j > 0.0
