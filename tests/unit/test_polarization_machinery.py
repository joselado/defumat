"""The pieces a Berry-phase polarization is assembled from.

Nothing here runs an SCF. What is checked is the geometry of a string mesh, the
branch bookkeeping that turns a set of string phases into one channel phase, the
ionic phase and its quantum, and -- the one statement with an analytic answer --
the Zak phase of the SSH model, which is pinned to ``0`` or ``pi`` by symmetry
and is therefore exact on *any* mesh.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from pypresso.topology.mesh import string_mesh
from pypresso.topology.polarization import (
    combine_string_phases,
    ionic_phase,
    polarization_quantum,
    string_phase,
)
from pypresso.topology.states import ModelStates
from pypresso.workflows.polarization import _refuse_ungapped, run_polarization

pytestmark = pytest.mark.unit


# --- the mesh ---------------------------------------------------------------

def test_a_string_mesh_is_open_across_the_plane_and_closed_along_the_string():
    mesh = string_mesh((2, 3), 5, gdir=2)
    assert mesh.shape == (6, 5)
    assert mesh.closed == (False, True)
    assert np.array_equal(mesh.span2, [0, 0, 1])
    # stepping off the end of a string comes back to its start plus b_gdir
    assert mesh.neighbour(0, 4, 1) == (0, pytest.approx(np.array([0, 0, 1])))


def test_a_string_spans_the_zone_once_without_repeating_its_endpoint():
    """``npoints`` distinct points at ``j / npoints``, not ``j / (npoints - 1)``.

    QE's ``kp_strings`` lays out ``nppstr`` points spanning the whole reciprocal
    vector, so its last one is the first one's image and is diagonalised twice.
    Dropping the repeat leaves the same links, which is what the wrap is for.
    """
    mesh = string_mesh((1, 1), 5, gdir=2)
    assert np.allclose(mesh.points[0, :, 2], [0.0, 0.2, 0.4, 0.6, 0.8])


def test_the_transverse_directions_are_the_two_that_are_not_gdir():
    for gdir in (0, 1, 2):
        mesh = string_mesh((2, 2), 3, gdir=gdir)
        along = mesh.points[..., gdir]
        # every string is a distinct transverse point, and only ``gdir`` varies
        # within one string
        assert len({tuple(np.delete(p, gdir)) for p in mesh.points[:, 0]}) == 4
        assert np.allclose(along[:, 0], along[0, 0])


def test_a_string_of_one_point_has_no_phase():
    with pytest.raises(ValueError, match="at least two k-points"):
        string_mesh((2, 2), 1)


# --- the branch bookkeeping -------------------------------------------------

def test_the_channel_phase_is_the_weighted_average():
    phases = np.array([0.10, 0.12, 0.11, 0.09])
    weights = np.full(4, 0.25)
    combined = combine_string_phases(phases, weights, nspin=1)
    # nspin = 1 doubles: one string set, two electrons per band
    assert combined.total == pytest.approx(2.0 * phases.mean() / (2 * np.pi), rel=1e-12)


def test_the_average_is_blind_to_which_branch_a_string_arrives_on():
    """Adding a whole turn to one string's raw phase must change nothing.

    This is the reason ``bp_c_phase.f90`` averages unit complex numbers rather
    than angles: a string that happens to land on the far side of the cut is the
    same physical phase, and an arithmetic mean of the angles is not.
    """
    phases = np.array([3.0, -3.0, 3.05, -3.05])
    weights = np.full(4, 0.25)
    plain = combine_string_phases(phases, weights)
    shifted = combine_string_phases(phases + np.array([0, 2, 0, -2]) * np.pi,
                                    weights)
    assert shifted.total == pytest.approx(plain.total, abs=1e-12)
    assert np.allclose(np.sort(shifted.phases), np.sort(plain.phases), atol=1e-12)


def test_the_channel_is_folded_before_it_is_doubled():
    """QE's order, and the two orders differ by *half* an all-even cell's quantum.

    ``bp_c_phase.f90`` reduces the single-channel phase onto ``[-1/2, 1/2)``
    (``pdl_elec_up - nint(pdl_elec_up)``) and only then adds the two channels.
    Doubling first and reducing the sum gives a different number whenever the
    doubled phase leaves that interval -- and since an all-even cell's quantum
    is 2, the two answers are half a quantum apart rather than a whole one.

    Neither committed reference can see this: AlAs has an odd valence, so its
    quantum is 1 and the difference is a full quantum there; silicon's
    electronic phase is exactly zero. So the convention is pinned here instead.
    """
    for single, expected in [(0.3, 0.6), (0.6, -0.8), (-0.45, -0.9)]:
        phases = np.full(4, single * 2 * np.pi)
        combined = combine_string_phases(phases, np.full(4, 0.25), nspin=1)
        assert combined.total == pytest.approx(expected, abs=1e-12), (
            f"single-channel {single} should double to {expected}"
        )


def test_a_spin_resolved_channel_is_not_doubled():
    phases = np.array([0.2, 0.2])
    weights = np.full(2, 0.5)
    one = combine_string_phases(phases, weights, nspin=1)
    two = combine_string_phases(phases, weights, nspin=2)
    assert one.total == pytest.approx(2.0 * two.total, rel=1e-12)


def test_the_weights_have_to_match_the_strings():
    with pytest.raises(ValueError, match="weights"):
        combine_string_phases(np.zeros(4), np.zeros(3))


# --- the ions ---------------------------------------------------------------

def test_the_ionic_phase_of_zincblende():
    """``Z_v`` times the crystal coordinate, reduced by each atom's own quantum.

    AlAs along the third reciprocal vector: Al sits at the origin and As at
    ``-1/4`` in that coordinate with five valence electrons, so its raw phase is
    ``-1.25`` and its quantum is 1 because five is odd. ``pw.x`` prints exactly
    this table (``tests/data/qe/reference.out.alas-berry``).
    """
    positions = np.array([[0.0, 0.0, 0.0], [-0.25, -0.25, -0.25]])
    per_atom, total, quantum = ionic_phase(positions, [3.0, 5.0], gdir=2)
    assert per_atom == pytest.approx([0.0, -0.25])
    assert total == pytest.approx(-0.25)
    assert quantum == 1.0


def test_an_all_even_cell_keeps_the_larger_quantum():
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.25]])
    _, _, quantum = ionic_phase(positions, [4.0, 6.0], gdir=2)
    assert quantum == 2.0
    assert polarization_quantum([4.0, 6.0], nspin=1) == 2.0
    assert polarization_quantum([4.0, 5.0], nspin=1) == 1.0
    # resolving the two channels halves it whatever the valences are
    assert polarization_quantum([4.0, 6.0], nspin=2) == 1.0


# --- the one analytic answer ------------------------------------------------

def _ssh(v: float, w: float):
    """Su-Schrieffer-Heeger in the periodic gauge, where ``H(k + 1) = H(k)``."""
    def hamiltonian(k):
        f = v + w * jnp.exp(2j * jnp.pi * k[0])
        return jnp.array([[0.0 + 0j, jnp.conj(f)], [f, 0.0 + 0j]])
    return hamiltonian


@pytest.mark.parametrize("npoints", [8, 21, 40])
@pytest.mark.parametrize(
    "v, w, expected", [(1.0, 0.4, 0.0), (0.4, 1.0, 1.0)]
)
def test_the_ssh_zak_phase_is_quantised_on_any_mesh(npoints, v, w, expected):
    """``0`` in the trivial phase and ``pi`` in the topological one, exactly.

    Chiral symmetry quantises the Zak phase, and the string product is a
    determinant of overlaps, so the answer is an exact 0 or pi on a mesh of any
    size -- not a Riemann sum converging onto one. That is the same property
    that makes a Fukui-Hatsugai-Suzuki Chern number an exact integer, and it is
    what pins this module's conventions without running a DFT calculation.
    """
    points = np.zeros((npoints, 3))
    points[:, 0] = np.arange(npoints) / npoints
    states = ModelStates.solve(_ssh(v, w), points, nocc=1)
    phase = string_phase(states, k_batch=1)
    assert abs(phase / np.pi) == pytest.approx(expected, abs=1e-10)


def test_a_string_phase_needs_more_than_one_point():
    points = np.zeros((1, 3))
    states = ModelStates.solve(_ssh(1.0, 0.4), points, nocc=1)
    with pytest.raises(ValueError, match="at least two k-points"):
        string_phase(states, k_batch=1)


# --- the refusals -----------------------------------------------------------

class _System:
    """The three fields the refusals read, and nothing else."""

    def __init__(self, occupations="fixed", nspin=1, spiral_q=None):
        self.occupations = occupations
        self.nspin = nspin
        self.spiral_q = spiral_q


def test_a_metal_is_refused_by_name():
    """A Berry phase is a property of a gapped manifold and a metal has none."""
    with pytest.raises(NotImplementedError, match="metal"):
        _refuse_ungapped(_System(occupations="smearing"))
    with pytest.raises(NotImplementedError, match="metal"):
        _refuse_ungapped(_System(occupations="tetrahedra"))
    _refuse_ungapped(_System(occupations="fixed"))
    _refuse_ungapped(_System(occupations="from_input"))


def test_two_spin_channels_are_refused_by_name():
    with pytest.raises(NotImplementedError, match="nspin = 2"):
        run_polarization(_System(nspin=2), (), None)


def test_a_spin_spiral_is_refused_by_name():
    with pytest.raises(NotImplementedError, match="spiral"):
        run_polarization(_System(spiral_q=(0.0, 0.0, 0.5)), (), None)
