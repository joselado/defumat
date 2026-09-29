"""The streamed wavefunction store, and the two memory modes in front of it.

``wfc_store = 'stream'`` keeps the store in host memory for the whole SCF and
moves one k-chunk at a time through every pass that reads it
(:mod:`defumat.scf.streaming`). It is a regrouping of the same sums, so the
standard is round-off against the whole-set store, regime by regime -- every
regime has its own per-k arrays to slice, and a slice taken from the wrong one
(the spiral's doubled basis list, the gamma trick's ``-(k+G)`` index, the
Hubbard projectors) gives a density that is plausible and wrong.

Everything here runs on a CPU: streaming is asked for by name, which is what
lets the gate exercise it without a card. What a card adds -- the peak it
removes -- is measured in ``PERFORMANCE.md`` and cannot be asserted here,
since the CPU client reports no memory statistics.
"""

from __future__ import annotations

import types
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from defumat import batching
from defumat.batching import k_chunks, memory_preset, resolve_memory_mode
from defumat.calculator import Calculator

pytestmark = pytest.mark.unit

PSEUDO = "tests/data/pseudo"

#: Two-atom, eight-k-point silicon: the reference cell on the whole
#: 2x2x2 grid, so there are several chunks and a short last one at
#: ``k_batch = 3``.
SILICON_8K = """
 &control
    calculation = 'scf'
 /
 &system
    ibrav = 2, celldm(1) = 10.2, nat = 2, ntyp = 1,
    ecutwfc = 12.0, nosym = .true.
 /
 &electrons
    conv_thr = 1.0d-12
 /
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS crystal
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS (automatic)
 2 2 2 0 0 0
"""


def _pair(make, k_batch=1, **options):
    """The same SCF with the store streamed and with it on the device."""
    runs = {}
    for where in ("stream", "device"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calculator = make(k_batch=k_batch, announce=False)
            runs[where] = calculator.get_scf(wfc_store=where, **options)
    return runs["stream"], runs["device"]


def _assert_same(streamed, whole, energy=1e-10):
    assert streamed.converged and whole.converged
    assert streamed.iterations == whole.iterations
    assert streamed.total_energy == pytest.approx(whole.total_energy, abs=energy)
    np.testing.assert_allclose(np.asarray(streamed.eigenvalues),
                               np.asarray(whole.eigenvalues), atol=1e-9)
    np.testing.assert_allclose(np.asarray(streamed.density),
                               np.asarray(whole.density), atol=1e-9)
    # The store stays where it was kept: a streamed result is a host array,
    # and anything that reads it whole brings it across itself.
    assert isinstance(streamed.wavefunctions, np.ndarray)


# ---------------------------------------------------------------------------
# the chunks


@pytest.mark.parametrize("nk, batch", [(8, 1), (8, 3), (8, 8), (8, None), (1, 4)])
def test_the_chunks_cover_every_k_point_once_and_share_one_shape(nk, batch):
    seen, shapes = [], set()
    for rows, live in k_chunks(nk, batch):
        shapes.add(rows.shape)
        seen.extend(rows[:live].tolist())
        # The padding repeats a row of the same chunk, never one of another.
        assert set(rows[live:].tolist()) <= set(rows[:live].tolist())
    assert seen == list(range(nk))
    assert len(shapes) == 1


# ---------------------------------------------------------------------------
# the streamed SCF against the whole-set one


@pytest.mark.parametrize("k_batch", [
    3, pytest.param(1, marks=pytest.mark.slow)])
def test_a_streamed_scf_is_the_whole_set_scf(k_batch):
    """Norm-conserving silicon on eight k-points, and a short last chunk at 3.

    ``k_batch = 3`` stays in the gate although it takes about ten seconds,
    most of it compilation: streaming is the accelerator default and this is
    the only test the gate runs through it, padding included.
    """
    streamed, whole = _pair(
        lambda **o: Calculator.from_text(SILICON_8K, PSEUDO, **o), k_batch)
    _assert_same(streamed, whole)


@pytest.mark.slow
def test_a_streamed_ultrasoft_scf_is_the_whole_set_scf():
    """``becsum`` from each chunk's own projectors, symmetrised once."""
    streamed, whole = _pair(lambda **o: Calculator.from_file(
        "tests/data/qe/si2-us.in", pseudo_dir=PSEUDO, **o))
    _assert_same(streamed, whole)


@pytest.mark.slow
def test_a_streamed_spiral_is_the_whole_set_spiral():
    """The spiral's basis list is doubled: a chunk reads rows ``ik`` and ``ik + nk``."""
    streamed, whole = _pair(lambda **o: Calculator.from_file(
        "tests/data/qe/h-chain-spiral.in", pseudo_dir=PSEUDO, **o))
    _assert_same(streamed, whole)


def test_a_streamed_start_is_the_whole_set_start():
    """``wfcinit`` per chunk spans what the whole-set start spans.

    Eight atomic orbitals and two random vectors per k-point, rotated among
    themselves: ``nbnd`` equals the span's size, so the start *is* the span
    and the chunked start must span the same space at every k-point. That is
    what pins the random top-up -- one fixed key at every k-point -- which a
    key split over the batch would break from the first iteration on.

    **The span and not the vectors**, for the reason rule D4 gives: silicon's
    levels are degenerate here, and a rotation within a multiplet is free for
    an ``eigh`` compiled over a different batch shape to choose differently.
    Measured: the vectors themselves differ by up to 0.92 inside multiplets,
    while every singular value of the overlap below is 1 to round-off.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculation = Calculator.from_text(
            SILICON_8K, PSEUDO, k_batch=1, announce=False).calculation
    rho = calculation.starting_density()
    potential = calculation.potential(rho)
    hamiltonians = calculation.hamiltonian(potential.v_scf)
    nbnd = 10  # eight atomic orbitals and two random vectors
    whole = np.asarray(calculation.starting_wavefunctions(hamiltonians, nbnd))
    for rows, live in k_chunks(calculation.system.kpoints.nk, 3):
        chunk = np.asarray(calculation.starting_wavefunctions(
            hamiltonians, nbnd, rows=rows))
        for position, ik in enumerate(rows[:live]):
            overlap = chunk[0, position] @ whole[0, ik].conj().T
            np.testing.assert_allclose(np.linalg.svd(overlap, compute_uv=False),
                                       1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# the modes


def test_the_presets(monkeypatch):
    """What each mode sets, on a CPU and on a card."""
    monkeypatch.setattr(batching, "_backend", lambda: "cpu")
    assert memory_preset("speed") == {"k_batch": 1, "band_batch": 1,
                                      "projectors": "store", "wfc_store": "device"}
    assert memory_preset("memory")["projectors"] == "rebuild"
    monkeypatch.setattr(batching, "_backend", lambda: "gpu")
    assert memory_preset("speed") == {"k_batch": None, "band_batch": None,
                                      "projectors": "store", "wfc_store": "device"}
    assert memory_preset("memory") == {"k_batch": 1, "band_batch": None,
                                       "projectors": "rebuild",
                                       "wfc_store": "stream"}


def test_memory_is_the_default_on_a_card_and_speed_on_a_cpu(monkeypatch):
    monkeypatch.delenv("DEFUMAT_MEMORY_MODE", raising=False)
    monkeypatch.setattr(batching, "_backend", lambda: "cpu")
    assert resolve_memory_mode() == "speed"
    monkeypatch.setattr(batching, "_backend", lambda: "gpu")
    assert resolve_memory_mode() == "memory"
    monkeypatch.setenv("DEFUMAT_MEMORY_MODE", "speed")
    assert resolve_memory_mode() == "speed", "the environment beats the platform"
    assert resolve_memory_mode("memory") == "memory", "an argument beats both"
    with pytest.raises(ValueError, match="memory_mode must be one of"):
        resolve_memory_mode("fast")


def test_speed_that_would_not_fit_falls_back_to_memory(monkeypatch):
    """The fall-back fires, and it fires only when the preset is what would run.

    Tested by making the check say no rather than by finding a cell too large
    for a card -- a guard is tested by feeding it the case that must trip it.
    """
    from defumat import sizing
    from defumat.scf import driver

    monkeypatch.setattr(batching, "_backend", lambda: "gpu")
    monkeypatch.delenv("DEFUMAT_K_BATCH", raising=False)
    monkeypatch.setattr(sizing, "speed_mode_fits", lambda *a, **k: sizing.SpeedCheck(
        fits=False, estimate=10 * 2**30, available=4 * 2**30))
    calculator = Calculator.from_text(SILICON_8K, PSEUDO, announce=False)
    system, pseudos = calculator.system, calculator.pseudos

    with pytest.warns(RuntimeWarning, match="does not fit this device"):
        mode = driver._resolve_memory_mode_for("speed", system, pseudos,
                                               "default", "default", None)
    assert mode == "memory"

    # An explicit k_batch means the preset is not what runs, so nothing is
    # sized and nothing is overridden.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert driver._resolve_memory_mode_for(
            "speed", system, pseudos, 4, "default", None) == "speed"
        assert driver._resolve_memory_mode_for(
            "memory", system, pseudos, "default", "default", None) == "memory"


def test_the_calculator_takes_the_mode_and_the_setup_follows_it():
    """``memory_mode`` builds the ``Calculation`` and is not forwarded past it."""
    calculator = Calculator.from_text(SILICON_8K, PSEUDO, memory_mode="memory",
                                      announce=False)
    calculation = calculator.calculation
    assert calculation.memory_mode == "memory"
    assert calculation.projector_storage == "rebuild"
    assert calculation.projectors.is_lazy
    # Re-asking with the same mode keeps the setup; another mode rebuilds it.
    calculator._adopt({"memory_mode": "memory"})
    assert calculator._calculation is calculation
    calculator._adopt({"memory_mode": "speed"})
    assert calculator.calculation.memory_mode == "speed"


def test_a_lazy_chunk_projector_is_the_stored_one():
    """``projectors_at`` builds a chunk's ``vkb`` from the core, not by slicing."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stored = Calculator.from_text(SILICON_8K, PSEUDO, projectors="store",
                                      announce=False).calculation
        lazy = Calculator.from_text(SILICON_8K, PSEUDO, projectors="rebuild",
                                    announce=False).calculation
    rows = np.array([5, 2, 7])
    np.testing.assert_allclose(np.asarray(lazy.projectors_at(rows)),
                               np.asarray(stored.projectors.vkb)[rows], atol=1e-13)
    assert jnp.shape(stored.projectors_at(rows)) == (3,) + stored.projectors.vkb.shape[1:]


@pytest.mark.slow  # 27 s on the CPU: an SCF and three path solves
def test_an_eigenvalue_only_solve_streams_and_keeps_no_states(k_batch=3):
    """``fixed_density_bands`` walks the chunks and drops each one's states.

    A band path or an NSCF grid asks for energies, and the streamed solve is
    ``c_bands_nscf``: every chunk from scratch, its states discarded as it
    returns. Round-off against the whole-set solve, a short last chunk at
    ``k_batch = 3``, and no wavefunctions handed back.
    """
    from defumat.system.kpoints import KPoints
    from defumat.workflows.nscf import fixed_density_bands, fixed_density_states

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_text(SILICON_8K, PSEUDO, k_batch=k_batch,
                                          announce=False)
        result = calculator.get_scf()
    system = calculator.system
    path = KPoints.from_cartesian(
        np.linspace([0.0, 0.0, 0.0], [0.5, 0.5, 0.5], 7), np.full(7, 1.0 / 7))
    common = dict(nbnd=8, conv_thr=1e-10, k_batch=k_batch)
    _, _, streamed = fixed_density_bands(system, calculator.pseudos,
                                         result.density, path,
                                         wfc_store="stream", **common)
    _, _, whole, states = fixed_density_states(system, calculator.pseudos,
                                               result.density, path, **common)
    np.testing.assert_allclose(streamed, np.asarray(whole), atol=1e-9)
    assert states is not None
    _, _, _, dropped = fixed_density_states(
        system, calculator.pseudos, result.density, path, wfc_store="stream",
        keep_states=False, **common)
    assert dropped is None


# ---------------------------------------------------------------------------
# consumers of a streamed store

#: The eight-k-point cell on an ultrasoft dataset, so ``becsum`` and the
#: augmentation charge are read through each chunk's own projectors.
SILICON_8K_US = SILICON_8K.replace(
    "ecutwfc = 12.0,", "ecutwfc = 12.0, ecutrho = 96.0,").replace(
    "conv_thr = 1.0d-12", "conv_thr = 1.0d-8").replace(
    "Si.pz-vbc.UPF", "Si.pz-n-rrkjus_psl.0.1.UPF")


def test_a_host_store_is_read_a_chunk_at_a_time():
    """A numpy store gives the whole-set density, ``becsum`` and projections.

    What a streamed SCF hands back is a host array, and the consumers after
    it -- an STM image, a windowed structure factor, a relaxation's density
    extrapolation, a projected DOS, the site moments -- walk it a chunk at a
    time instead of moving it to the device whole (``GPU-MEMORY-NEXT.md``
    item 4). The same states in both forms, a short last chunk at
    ``k_batch = 3``: round-off.
    """
    from defumat.hubbard.projectors import build_atomic_projectors
    from defumat.projwfc.angular_momentum import _site_density_matrix
    from defumat.projwfc.channels import projection_channels
    from defumat.projwfc.projections import atomic_projections

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculation = Calculator.from_text(SILICON_8K_US, PSEUDO, k_batch=3,
                                           announce=False).calculation
    # The starting states rather than converged ones: the two routes are
    # compared at the same states, and no SCF is needed for that.
    potential = calculation.potential(calculation.starting_density())
    device = calculation.starting_wavefunctions(
        calculation.hamiltonian(potential.v_scf), 4)
    host = np.array(device)
    kweights = np.asarray(calculation.system.kpoints.weights)
    weights = jnp.asarray(np.broadcast_to(kweights[None, :, None], (1, 8, 4)))
    result = types.SimpleNamespace(wavefunctions=device, occupations=weights[0])

    for whole, chunked in zip(calculation.becsum(device, weights),
                              calculation.becsum(host, weights)):
        np.testing.assert_allclose(np.asarray(chunked), np.asarray(whole),
                                   atol=1e-13)
    np.testing.assert_allclose(np.asarray(calculation.density(host, weights)),
                               np.asarray(calculation.density(device, weights)),
                               atol=1e-12)
    np.testing.assert_allclose(atomic_projections(calculation, host),
                               atomic_projections(calculation, device),
                               atol=1e-12)
    # The site moments' density matrix rather than the moments: silicon's
    # ``<L>`` and ``<S>`` are zero by symmetry, which both routes would pass.
    system = calculation.system
    projectors = np.asarray(build_atomic_projectors(
        calculation.pseudos, system.structure, system.cell,
        calculation.basis.smooth, calculation.basis.planewaves,
        calculation.basis_kpoints, calculation._overlap))
    channels = projection_channels(calculation.pseudos, system.structure)
    whole = _site_density_matrix(calculation, result, projectors, channels)
    streamed = _site_density_matrix(
        calculation, types.SimpleNamespace(wavefunctions=host,
                                           occupations=result.occupations),
        projectors, channels)
    assert np.max(np.abs(whole)) > 0.1
    np.testing.assert_allclose(streamed, whole, atol=1e-12)


@pytest.mark.slow  # 7.7 s in the gate: an SCF and two path solves
def test_a_streamed_solve_that_keeps_its_states_keeps_them_on_the_host(k_batch=3):
    """``fixed_density_states`` in a streamed store: the whole-set solve, in host memory."""
    from defumat.system.kpoints import KPoints
    from defumat.workflows.nscf import fixed_density_states

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_text(SILICON_8K, PSEUDO, k_batch=k_batch,
                                          announce=False)
        result = calculator.get_scf()
    system = calculator.system
    path = KPoints.from_cartesian(
        np.linspace([0.0, 0.0, 0.0], [0.5, 0.5, 0.5], 7), np.full(7, 1.0 / 7))
    common = dict(nbnd=8, conv_thr=1e-10, k_batch=k_batch)
    calculation, _, streamed, host = fixed_density_states(
        system, calculator.pseudos, result.density, path, wfc_store="stream",
        **common)
    _, _, whole, device = fixed_density_states(
        system, calculator.pseudos, result.density, path, **common)
    np.testing.assert_allclose(streamed, np.asarray(whole), atol=1e-9)
    assert isinstance(host, np.ndarray) and host.shape == device.shape
    # The same states up to a phase and, inside a multiplet, a rotation (rule
    # D4): each k-point's two sets span the same space.
    for ik in range(path.nk):
        overlap = host[0, ik] @ np.asarray(device[0, ik]).conj().T
        occupied = np.linalg.svd(overlap[:4, :4], compute_uv=False)
        np.testing.assert_allclose(occupied, 1.0, atol=1e-6)


@pytest.mark.slow  # 15 s on the CPU: a PAW spinor setup and two torques
def test_the_paw_orientation_torque_walks_a_host_store():
    """The one-centre torque through the output states, streamed.

    ``_onecenter_torque`` differentiates the turned ``becsum`` in the three
    generators; a streamed store takes each chunk's forward derivative instead
    of moving the set to the device. The input ``becsum`` is the output turned
    by a finite rotation so the torque is not zero by alignment -- a zero here
    would pass whatever the regrouping did.
    """
    from defumat.forces.torque import rotate_texture
    from defumat.scf.driver import _onecenter_torque
    from defumat.workflows.anisotropy import rotation_from_euler

    text = open("tests/data/qe/o2-paw-texture.in").read()
    text = text.replace("celldm(1) = 14.0", "celldm(1) = 9.0").replace(
        "ecutwfc = 30, ecutrho = 240", "ecutwfc = 25, ecutrho = 200").replace(
        "K_POINTS {gamma}", "K_POINTS {automatic}\n 3 1 1 0 0 0")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculation = Calculator.from_text(text, PSEUDO, announce=False,
                                           k_batch=2).calculation
    potential = calculation.potential(calculation.starting_density())
    psi = calculation.starting_wavefunctions(
        calculation.hamiltonian(potential.v_scf), 8)
    kweights = np.asarray(calculation.system.kpoints.weights)
    weights = jnp.asarray(np.broadcast_to(kweights[None, :, None],
                                          psi.shape[:3]).astype(float))
    becsum_out = calculation.becsum(psi, weights)
    rotation = jnp.asarray(rotation_from_euler(0.4, 0.9, -0.3))
    becsum_in = tuple(None if b is None else rotate_texture(b, rotation)
                      for b in becsum_out)
    whole = _onecenter_torque(calculation, becsum_in, becsum_out,
                              wavefunctions=psi, weights=weights)
    streamed = _onecenter_torque(calculation, becsum_in, becsum_out,
                                 wavefunctions=np.array(psi), weights=weights)
    assert np.linalg.norm(whole) > 1e-3
    np.testing.assert_allclose(streamed, whole, atol=1e-13)
