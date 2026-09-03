"""Half-sphere storage, against the full sphere it replaces.

``K_POINTS gamma`` stores one plane wave of each ``(G, -G)`` pair, because at
``k = 0`` a state can be chosen real. It halves ``npwx`` and with it every array
a band lives in -- the wavefunctions, ``vkb``, the Davidson subspace -- which on
a 157-atom slab is 189 GB against 96.

**The reference needs no other code.** The same cell run with an explicit
``k = 0`` on the whole sphere is the same physics exactly, so the two must agree
to round-off rather than to a tolerance. They do: the totals to ~1e-14 Ry, the
eigenvalues to ~3e-15 and the forces to ~3e-16 Ry/bohr.

**Only the wavefunction sphere halves.** The dense G set stays whole, which is a
deliberate departure from ``pw.x`` -- see ``build_basis``. The memory is in the
plane-wave-sized arrays; halving the dense set too would save 0.2 per cent more
and put a conjugate fill inside every consumer of a real field.
"""

import warnings

import numpy as np
import pytest

from defumat.calculator import Calculator
from defumat.forces import compute_forces

pytestmark = pytest.mark.regression

_TEMPLATE = """
&control
  calculation = 'scf', tprnfor = .true.
/
&system
  ibrav = 2, celldm(1) = 10.20, nat = 2, ntyp = 1, ecutwfc = 12.0,
  nosym = .true.{extra}
/
&electrons
  conv_thr = 1.0d-12
/
ATOMIC_SPECIES
 Si 28.086 {pseudo}
ATOMIC_POSITIONS alat
 Si 0.01 0.00 0.00
 Si 0.26 0.24 0.25
K_POINTS {kpoints}
"""

_MAGNETIC = (", nspin = 2, starting_magnetization(1) = 0.1,\n"
             "  occupations = 'smearing', smearing = 'mv', degauss = 0.02")

#: The four regimes the FePc slab this was built for actually needs: a GGA, two
#: spin channels and a smearing. Each is a separate path through the density and
#: the potential, and each is checked.
REGIMES = {
    "lda": "",
    "pbe": ", input_dft = 'PBE'",
    "lsda": _MAGNETIC,
    "pbe-lsda": ", input_dft = 'PBE'" + _MAGNETIC,
}


def _run(pseudo_dir, extra, kpoints, pseudo="Si.pz-vbc.UPF"):
    text = _TEMPLATE.format(extra=extra, kpoints=kpoints, pseudo=pseudo)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_text(text, pseudo_dir, announce=False)
        result = calculator.get_scf()
        forces = compute_forces(calculator.calculation, result)
    return calculator, result, forces


@pytest.mark.slow
@pytest.mark.parametrize("regime", sorted(REGIMES), ids=sorted(REGIMES))
def test_gamma_reproduces_the_full_sphere(pseudo_dir, regime):
    """The whole claim, in one comparison per regime.

    Not a tolerance: the two runs are the same calculation in two storages, so
    anything above round-off is a dropped ``G = 0`` term or a missing factor of
    two. Both have happened here -- see the force test below.
    """
    extra = REGIMES[regime]
    whole, full, _ = _run(pseudo_dir, extra, "automatic\n 1 1 1 0 0 0")
    half, gamma, _ = _run(pseudo_dir, extra, "gamma")

    assert whole.calculation.gamma_only is False
    assert half.calculation.gamma_only is True
    assert full.converged and gamma.converged

    assert gamma.total_energy == pytest.approx(full.total_energy, abs=1e-12)
    np.testing.assert_allclose(
        np.asarray(gamma.eigenvalues), np.asarray(full.eigenvalues), atol=1e-12
    )


@pytest.mark.slow
def test_the_force_agrees_to_round_off(pseudo_dir):
    """The force is where a dropped ``G = 0`` term shows and the energy is not.

    Measured while building this: with the correction missing from
    ``forces/energy.py``'s plane-wave sums the total energy was right to
    **3e-12 Ry** and the force was wrong by **0.4 Ry/bohr** on a force of 0.06.
    The frozen energy is the functional the force is the gradient of, so being
    stationary at the ground state hides an error in its derivative.
    """
    _, _, full = _run(pseudo_dir, "", "automatic\n 1 1 1 0 0 0")
    _, _, gamma = _run(pseudo_dir, "", "gamma")

    scale = float(np.abs(np.asarray(full.forces)).max())
    assert scale > 1e-3, "the test geometry must carry a real force"
    np.testing.assert_allclose(
        np.asarray(gamma.forces), np.asarray(full.forces), atol=1e-10
    )


@pytest.mark.slow
def test_the_plane_waves_halve_and_the_dense_set_does_not(pseudo_dir):
    """Where the memory comes from, and where it deliberately does not.

    ``npwx`` halves; ``ngm`` is untouched. Halving the dense set as well is what
    ``pw.x`` does and would save 0.2 per cent more here -- at the cost of a
    conjugate fill in ``to_dense``, ``v_of_rho``, the GGA gradient, the
    augmentation charge and the symmetriser, each a place the ``G = 0`` term can
    go missing quietly.
    """
    whole, _, _ = _run(pseudo_dir, "", "automatic\n 1 1 1 0 0 0")
    half, _, _ = _run(pseudo_dir, "", "gamma")

    npwx_full = whole.calculation.basis.npwx
    npwx_half = half.calculation.basis.npwx
    assert npwx_half == (npwx_full + 1) // 2
    assert half.calculation.basis.dense.ngm == whole.calculation.basis.dense.ngm


@pytest.mark.slow
def test_the_estimate_mirrors_what_the_run_builds(pseudo_dir):
    """The size tool has to make the same consumption decision as the driver.

    Sizing the substitution where the run consumes the half sphere overstates
    every band-sized array by two, which is the failure the tool exists to
    prevent.
    """
    half, _, _ = _run(pseudo_dir, "", "gamma")
    estimate = half.estimate()
    assert estimate.gamma_only is True
    assert estimate.npwx == half.calculation.basis.npwx
    assert estimate.ngm == half.calculation.basis.dense.ngm


# --- what it refuses --------------------------------------------------------


def _substitutes(pseudo_dir, extra, pseudo="Si.pz-vbc.UPF"):
    """Whether a gamma input is substituted away rather than consumed."""
    text = _TEMPLATE.format(extra=extra, kpoints="gamma", pseudo=pseudo)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_text(text, pseudo_dir, announce=False)
    return not calculator.calculation.gamma_only


def test_symmetry_falls_back_to_the_full_sphere(pseudo_dir):
    """``symmetry_maps`` permutes the G list and a rotation leaves the half.

    An operation carries a stored ``G`` onto an unstored ``-G'``, so the
    permutation would have to carry a conjugation. Substituted with a warning
    rather than refused, because the run is still perfectly valid -- it just
    costs twice the plane waves, and ``nosym = .true.`` is the way out.
    """
    # The template carries ``nosym``, so that case consumes the storage ...
    assert not _substitutes(pseudo_dir, "")
    # ... and the same input without it falls back.
    text = _TEMPLATE.format(extra="", kpoints="gamma", pseudo="Si.pz-vbc.UPF")
    text = text.replace("  nosym = .true.", "  nosym = .false.")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calculator = Calculator.from_text(text, pseudo_dir, announce=False)
        assert not calculator.calculation.gamma_only
    assert any("full G sphere" in str(w.message) for w in caught)


def test_ultrasoft_falls_back_to_the_full_sphere(pseudo_dir):
    """``addusdens`` and ``newd`` need their own ``fact = 2``, and have none.

    A gap rather than a missing term -- nothing in the assembly is
    norm-conserving -- but an unchecked one, so the storage is not taken.
    """
    assert _substitutes(pseudo_dir, ", ecutrho = 120.0",
                        pseudo="Si.pz-n-kjpaw_psl.0.1.UPF")


@pytest.mark.slow
@pytest.mark.parametrize("mixing", ["local-TF", "TF"], ids=["local-tf", "tf"])
def test_a_screened_mixer_is_unaffected_by_the_storage(pseudo_dir, mixing):
    """The mixer's preconditioner solves on the **dense** set, which stays whole.

    Worth a test rather than an argument, and worth one for a specific reason:
    a wrong factor inside a preconditioner changes the *convergence* and not the
    fixed point, so it does not show up in the energy comparison above -- the
    run reaches the same answer and takes a different number of iterations to do
    it. On a slab that is the difference between converging and not
    (``local-TF`` is what QE's Co(0001) film needs at ``mixing_beta = 0.7``).

    It passes structurally: ``local_tf_preconditioner`` and
    ``kerker_preconditioner`` are handed ``gvectors``, which is the dense set,
    and only the wavefunction sphere halves. The test pins that rather than
    trusting it, and would catch anyone later deciding to halve the dense set
    after all.
    """
    extra_electrons = f",\n  mixing_mode = '{mixing}', mixing_beta = 0.3"

    def run(kpoints):
        text = _TEMPLATE.format(extra="", kpoints=kpoints,
                                pseudo="Si.pz-vbc.UPF")
        text = text.replace("conv_thr = 1.0d-12",
                            "conv_thr = 1.0d-12" + extra_electrons)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calculator = Calculator.from_text(text, pseudo_dir, announce=False)
            return calculator, calculator.get_scf()

    whole, full = run("automatic\n 1 1 1 0 0 0")
    half, gamma = run("gamma")

    assert half.calculation.gamma_only and not whole.calculation.gamma_only
    assert full.converged and gamma.converged
    assert gamma.total_energy == pytest.approx(full.total_energy, abs=1e-12)
    # The iteration count is the quantity a preconditioner bug moves, so it is
    # the one asserted -- not equal, but not drifting either.
    assert abs(gamma.iterations - full.iterations) <= 2
