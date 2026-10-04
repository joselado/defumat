"""The sum-over-states assemblies walk the k axis and give the whole axis's answer.

``OPEN.md`` Part XXIII item 24, second half. The optical conductivity, the
second harmonic, the shift current, the TDDFT ``chi_0`` and the transverse spin
``chi_0`` are each a sum over k of per-k terms, and they now walk
:func:`~defumat.batching.k_chunks` (:mod:`defumat.response.walk`): one chunk's
states on the device, its matrix elements built on its own rows, its share added
to the sum. What is checked here is the identity that makes that legitimate --
a chunk of three on eight k-points, so the last chunk is padded with rows of zero
weight, against the whole axis in one chunk, to round-off -- and that a host
store crosses a chunk at a time rather than whole.

The states are the starting ones and the eigenvalues are made up with a gap at
the occupied count: the walk is an identity whatever the states are, and no SCF
is needed to test it (``tests/unit/test_streaming.py`` does the same).
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest

from defumat.calculator import Calculator

pytestmark = pytest.mark.unit

PSEUDO = "tests/data/pseudo"

#: Two-atom silicon on the whole unshifted 2x2x2 grid: eight k-points, so a
#: chunk of three leaves a last chunk with one padded row.
SILICON = """
 &control
    calculation = 'scf'
 /
 &system
    ibrav = 2, celldm(1) = 10.2, nat = 2, ntyp = 1,
    ecutwfc = 10.0, nosym = .true.{smearing}
 /
 &electrons
    conv_thr = 1.0d-10
 /
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS crystal
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS (automatic)
 2 2 2 0 0 0
"""

#: fcc hydrogen, spin-polarized, on the same grid: the transverse response's cell.
HYDROGEN = """
 &control
    calculation = 'scf'
 /
 &system
    ibrav = 2, celldm(1) = 6.5, nat = 1, ntyp = 1,
    ecutwfc = 16.0,
    occupations = 'smearing', smearing = 'gaussian', degauss = 0.02,
    nspin = 2, nosym = .true., noinv = .true.,
    starting_magnetization(1) = 0.9
 /
 &electrons
    conv_thr = 1.0d-10
 /
ATOMIC_SPECIES
 H 1.008 H.pz-vbc.UPF
ATOMIC_POSITIONS crystal
 H 0.0 0.0 0.0
K_POINTS (automatic)
 2 2 2 0 0 0
"""


def _states(text, nbnd, nocc, k_batch=3):
    """A calculation, a potential, starting states and made-up eigenvalues."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculation = Calculator.from_text(text, PSEUDO, k_batch=k_batch,
                                           announce=False).calculation
    potential = calculation.potential(calculation.starting_density())
    states = calculation.starting_wavefunctions(
        calculation.hamiltonian(potential.v_scf), nbnd)
    nspin, nk = states.shape[:2]
    rng = np.random.default_rng(7)
    below = np.sort(rng.uniform(-0.6, 0.1, (nspin, nk, nocc)), axis=-1)
    above = np.sort(rng.uniform(0.5, 1.6, (nspin, nk, nbnd - nocc)), axis=-1)
    return calculation, potential.v_scf, states, np.concatenate([below, above], -1)


def _close(walked, whole):
    walked, whole = np.asarray(walked), np.asarray(whole)
    scale = float(np.max(np.abs(whole)))
    assert scale > 0.0
    assert float(np.max(np.abs(walked - whole))) <= 1e-12 * scale


def _uploads(monkeypatch):
    """Record the size of every ``device_put``, so a test can say none was the store."""
    sizes = []
    real = jax.device_put

    def spy(value, *args, **kwargs):
        sizes.append(int(getattr(value, "nbytes", 0)))
        return real(value, *args, **kwargs)

    monkeypatch.setattr(jax, "device_put", spy)
    return sizes


def test_the_conductivity_walks_to_the_whole_axis(monkeypatch):
    from defumat.response.conductivity import optical_conductivity

    calculation, v_scf, states, eigenvalues = _states(
        SILICON.format(smearing=",\n    occupations = 'smearing', "
                       "smearing = 'gaussian', degauss = 0.05"), 8, 4)
    host = np.array(states)
    # A scissors shift, so that the renormalised matrix elements walk too.
    options = dict(fermi_energy=0.3, nw=24, window=1.0, scissor=0.05)

    whole = optical_conductivity(calculation, host, eigenvalues, v_scf,
                                 k_batch=None, **options)
    sizes = _uploads(monkeypatch)
    walked = optical_conductivity(calculation, host, eigenvalues, v_scf,
                                  k_batch=3, **options)
    monkeypatch.undo()
    on_device = optical_conductivity(calculation, states, eigenvalues, v_scf,
                                     k_batch=3, **options)

    assert max(sizes) < host.nbytes / 2
    for result in (walked, on_device):
        _close(result.interband, whole.interband)
        _close(result.intraband, whole.intraband)
        assert result.degenerate_pairs == whole.degenerate_pairs


#: The same silicon on an ultrasoft dataset, whose generalised velocity carries
#: the augmentation dipole's connection: a stage of the walk of its own.
ULTRASOFT = SILICON.replace("ecutwfc = 10.0,", "ecutwfc = 12.0, ecutrho = 96.0,").replace(
    "Si.pz-vbc.UPF", "Si.pz-n-rrkjus_psl.0.1.UPF")


def _insulator(text=None):
    """Silicon with fixed occupations, as a strided band slice of a host store."""
    calculation, v_scf, states, eigenvalues = _states(
        text or SILICON.format(smearing=""), 8, 4)
    # A strided band slice, as the workflows hand it on.
    return calculation, v_scf, np.array(states)[..., :7, :], eigenvalues[..., :7]


@pytest.mark.slow
@pytest.mark.parametrize("dataset", ["norm-conserving", "ultrasoft"])
def test_the_second_harmonic_walks_to_the_whole_axis(dataset):
    from defumat.response.shg import second_harmonic

    calculation, v_scf, host, eigenvalues = _insulator(
        ULTRASOFT.format(smearing="") if dataset == "ultrasoft" else None)
    ddd_paw = calculation.onecenter(())[1] if calculation.is_paw else None
    with warnings.catch_warnings():
        # The 15^3 grid does not hold diamond's fractional translation; that
        # is a statement about the tensor, not about the walk.
        warnings.simplefilter("ignore")
        shg = [second_harmonic(calculation, host, eigenvalues, v_scf, nw=16,
                               ddd_paw=ddd_paw, scissor=0.05, k_batch=batch)
               for batch in (3, None)]
    _close(shg[0].chi, shg[1].chi)
    _close(shg[0].sigma_ii, shg[1].sigma_ii)
    assert shg[0].truncation == pytest.approx(shg[1].truncation, rel=1e-10)


def test_the_shift_current_walks_to_the_whole_axis():
    from defumat.response.photocurrent import shift_current

    calculation, v_scf, host, eigenvalues = _insulator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shift = [shift_current(calculation, host, eigenvalues, v_scf, nw=16,
                               k_batch=batch) for batch in (3, None)]
    _close(shift[0].sigma, shift[1].sigma)
    assert shift[0].truncation == pytest.approx(shift[1].truncation, rel=1e-10)


def test_the_tddft_chi0_walks_to_the_whole_axis():
    from defumat.tddft.chi0 import independent_response

    calculation, v_scf, states, eigenvalues = _states(SILICON.format(smearing=""), 8, 4)
    host = np.array(states)
    frequencies = np.array([0.0, 0.2]) + 0.01j
    chi = [independent_response(calculation, host, eigenvalues, v_scf, frequencies,
                                ecut_response=2.0, broadening=0.0, k_batch=batch)
           for batch in (3, None)]
    _close(chi[0].x, chi[1].x)


def test_the_transverse_response_walks_to_the_whole_axis():
    from defumat.tddft.spinchi0 import transverse_response

    calculation, _, states, eigenvalues = _states(HYDROGEN, 4, 1)
    host = np.array(states)
    frequencies = np.array([0.0, 0.05]) + 0.01j
    for q in ((0.0, 0.0, 0.0), (0.0, 0.0, 0.5)):
        chi = [transverse_response(calculation, host, eigenvalues, q, frequencies,
                                   ecut_response=4.0, broadening=0.0, k_batch=batch)
               for batch in (3, None)]
        _close(chi[0].x, chi[1].x)
