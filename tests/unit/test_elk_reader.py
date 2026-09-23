"""The Elk state reader: the binary layout, the conventions, and the refusals.

``PLAN.md`` P72. Everything here runs off the committed fixture in
``tests/data/elk/h_sc`` or off a synthetic file written on the spot, so no Elk
binary is needed.

The checks are chosen for the things that are *silently* wrong rather than
loudly so. A reader that takes Elk's one-record-two-arrays framing as two
records desyncs at the first density record and looks like a byte-order
problem; a Hartree-to-Rydberg factor applied to the density instead of the
potential is a factor of two on a number nothing else here prints; and the
species-outer atom index only shows itself on a cell with two species, which
hydrogen cannot reach at all.

Two fixtures, and the second exists only for the traps the first cannot fail:
``h_sc`` is simple-cubic hydrogen, and ``sic_zb`` is zincblende SiC, whose two
species have **different radial mesh lengths** and therefore expose both the
``natmtot`` sum and the uninitialised padding past ``nrmt(is)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from defumat.io.elk import (
    ElkGeometry,
    ElkState,
    read_elk_geometry,
    read_elk_state,
)
from defumat.io.elk_density import (
    poly4,
    real_spherical_harmonics,
    spline_weights,
)

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "elk" / "h_sc"
TWO_SPECIES = Path(__file__).resolve().parents[1] / "data" / "elk" / "sic_zb"

pytestmark = pytest.mark.skipif(
    not (FIXTURE / "STATE.OUT").is_file(),
    reason=f"no Elk fixture at {FIXTURE}",
)

needs_sic = pytest.mark.skipif(
    not (TWO_SPECIES / "STATE.OUT").is_file(),
    reason=f"no Elk fixture at {TWO_SPECIES}",
)


@pytest.fixture(scope="module")
def state() -> ElkState:
    return read_elk_state(FIXTURE)


@pytest.fixture(scope="module")
def sic() -> ElkState:
    return read_elk_state(TWO_SPECIES)


def read_rho3d(path):
    """``RHO3D.OUT``: a grid-size header line, then ``x y z rho`` per point."""
    rows = []
    with open(path) as handle:
        handle.readline()
        for line in handle:
            values = line.split()
            if len(values) == 4:
                rows.append([float(x) for x in values])
    data = np.array(rows)
    return data[:, :3], data[:, 3]


# --- what the file says about itself ------------------------------------------

def test_the_header_is_read(state):
    """Every scalar the density reconstruction is indexed by.

    ``rmt`` is the post-``autormt`` radius, which is ``rsp(nrmt)`` and not the
    species file's own ``rmt``: the input asked for 1.4 bohr and Elk kept it
    here, but on a denser cell it would not have.
    """
    assert state.version == (11, 0, 2)
    assert state.ngridg == (12, 12, 12)
    assert state.ngvec == 751
    assert state.nrmt == (197,)
    assert state.lmmaxo == 49
    assert state.lmaxo == 6
    assert state.geometry.species == ("H.in",)
    assert state.geometry.natoms == (1,)
    assert state.natmtot == 1
    assert state.rmt == pytest.approx([1.4], abs=1e-12)
    assert state.omega == pytest.approx(27.0, abs=1e-12)


def test_the_hartree_factor_is_on_the_potentials_and_not_the_density(state):
    """Elk is Hartree and defumat is Rydberg, and only some fields carry it.

    ``efermi`` is 0.0739705280 Ha in the file. A density is e/bohr^3 in both
    codes and takes no factor at all -- which the pointwise agreement against
    ``RHO3D.OUT`` below would catch, but only as a factor of two in a place
    where a factor of two is easy to blame on something else.
    """
    assert state.efermi == pytest.approx(0.14794105596912, rel=1e-12)


def test_the_muffin_tin_block_has_the_shape_the_record_length_asserts(state):
    """``(lmmaxo, nrmtmax, natmtot)``, Fortran-ordered, one record per pair."""
    assert state.rhomt.shape == (49, 197, 1)
    assert state.rhoir.shape == (12, 12, 12)
    for field in (state.vclmt, state.vxcmt, state.vsmt):
        assert field.shape == state.rhomt.shape
    for field in (state.vclir, state.vxcir, state.vsir):
        assert field.shape == state.rhoir.shape


def test_the_inner_region_boundary_is_recovered_from_the_zeroed_tail(state):
    """``nrmti`` is never written, and ``rfmtpack`` leaves it recoverable.

    Elk zeroes the ``lm > lmmaxi`` entries for ``ir <= nri`` on the way out, so
    the boundary is the first radius at which the high harmonics stop being
    exactly zero. It matters only for the three-point interpolation window that
    straddles it, at ``r = 0.014`` bohr against ``rmt = 1.4``.
    """
    assert list(state.nri) == [129]
    assert state.rsp[0][128] == pytest.approx(0.0131292, abs=1e-6)
    assert state.rsp[0][128] < 0.01 * state.rmt[0]


def test_the_origin_is_the_one_value_that_is_not_an_interpolation(state):
    """``rho(0) = f_00(0) Y_00`` exactly, and it is the reader's own check.

    Everywhere else ``rfpts`` interpolates, so a residual there mixes the
    reader with the interpolation. At the origin the window is clamped to the
    first mesh point and the harmonic sum is the ``l = 0`` term alone, so the
    file's own first printed value is a direct assertion about ``rhomt``.
    """
    points, values = read_rho3d(FIXTURE / "RHO3D.OUT")
    y00 = 1.0 / np.sqrt(4.0 * np.pi)
    assert np.allclose(points[0], 0.0)
    assert state.rhomt[0, 0, 0] * y00 == pytest.approx(values[0], abs=1e-10)


# --- the pieces of ``rfpts`` --------------------------------------------------

def test_real_spherical_harmonics_against_the_closed_forms():
    """``genrlmv``'s convention, ``j = l(l+1) + m + 1``, one-based in Fortran.

    **This is not the textbook real harmonic**, and that is the whole reason
    for the test. Elk takes ``sqrt(2) Re Y_{lm}`` for ``m > 0`` and
    ``sqrt(2) Im Y_{lm}`` for ``m < 0`` with the Condon-Shortley phase left
    inside ``Y``, so ``p_x``, ``p_y``, ``d_xy``, ``d_yz`` and ``d_xz`` all come
    out with the opposite sign to the usual definition while ``d_z2`` and
    ``d_x2-y2`` do not. Nothing else here can catch it: the sign cancels in any
    norm, and the committed fixture is one atom on a cubic site, where every
    channel this affects is zero. Getting it wrong would show first on a real
    crystal, as a density mirrored in a plane.
    """
    rng = np.random.default_rng(0)
    v = rng.normal(size=(20, 3))
    unit = v / np.linalg.norm(v, axis=1)[:, None]
    x, y, z = unit.T

    rlm = real_spherical_harmonics(2, v)
    assert rlm.shape == (20, 9)

    c = np.sqrt(3.0 / (4.0 * np.pi))
    d = np.sqrt(15.0 / (4.0 * np.pi))
    expected = {
        0: np.full(20, np.sqrt(1.0 / (4.0 * np.pi))),
        1: -c * y,
        2: c * z,
        3: -c * x,
        4: -d * x * y,
        5: -d * y * z,
        6: np.sqrt(5.0 / (16.0 * np.pi)) * (3.0 * z ** 2 - 1.0),
        7: -d * x * z,
        8: 0.5 * d * (x ** 2 - y ** 2),
    }
    for index, value in expected.items():
        assert rlm[:, index] == pytest.approx(value, abs=1e-12)


def test_real_spherical_harmonics_are_orthonormal_on_the_sphere():
    """A Lebedev-free check: the Gram matrix of a dense angular sample."""
    n = 200
    theta = np.arccos(np.linspace(-1.0, 1.0, n, endpoint=False) + 1.0 / n)
    phi = (np.arange(n) + 0.5) * 2.0 * np.pi / n
    t, p = np.meshgrid(theta, phi, indexing="ij")
    v = np.stack(
        [np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], axis=-1
    ).reshape(-1, 3)
    rlm = real_spherical_harmonics(3, v)
    gram = rlm.T @ rlm * (4.0 * np.pi / len(v))
    assert gram == pytest.approx(np.eye(16), abs=2e-3)


def test_poly4_is_exact_on_a_cubic():
    """A 4-point Lagrange polynomial reproduces any cubic through its nodes."""
    xa = np.array([[0.5, 0.9, 1.4, 2.2]])
    coefficients = np.array([0.3, -1.1, 2.0, 0.7])
    ya = np.polyval(coefficients, xa)
    x = np.array([[0.6, 1.0, 1.8, 2.0]])
    assert poly4(xa, ya, x) == pytest.approx(np.polyval(coefficients, x), abs=1e-12)


def test_spline_weights_integrate_a_cubic_exactly(state):
    """``wsplint`` on the fixture's own log mesh, against the analytic integral.

    Simpson's rule on this mesh is 9e-6 away from Elk's ``chgmt``, which is
    large enough to read as a reader bug, so the weights are transcribed rather
    than substituted -- and this is what says the transcription is right.
    """
    r = state.rsp[0][:state.nrmt[0]]
    w = spline_weights(r)
    for power in range(4):
        exact = (r[-1] ** (power + 1) - r[0] ** (power + 1)) / (power + 1)
        assert float(w @ r ** power) == pytest.approx(exact, rel=1e-10)


# --- the atom index, which no committed fixture can reach ---------------------

def test_the_atom_index_runs_species_outer_atom_inner():
    """``ias`` is Elk's ``idxas``, and both fixtures here have one species.

    ``init0.f90`` numbers atoms species-outer, so a two-species cell puts every
    atom of species 1 before the first atom of species 2. Getting it backwards
    reads one atom's muffin-tin block for another, which on a cell of identical
    species is invisible -- hence a synthetic geometry rather than a file.
    """
    geometry = ElkGeometry(
        avec=np.eye(3) * 5.0,
        species=("B", "N"),
        natoms=(2, 3),
        positions=np.zeros((5, 3)),
        magnetic_fields=np.zeros((5, 3)),
    )
    assert geometry.natmtot == 5
    assert list(geometry.species_of()) == [0, 0, 1, 1, 1]


def test_the_geometry_carries_the_scale_factors(tmp_path):
    """``scale`` multiplies all three vectors and ``scale1..3`` one each.

    Elk applies them in ``readinput`` before anything else sees ``avec``, so a
    ``GEOMETRY.OUT`` written with a scale describes a cell twice the size of the
    numbers printed under ``avec``.
    """
    path = tmp_path / "GEOMETRY.OUT"
    path.write_text(
        "scale\n  2.0\n\nscale2\n  3.0\n\n"
        "avec\n  1.0 0.0 0.0\n  0.0 1.0 0.0\n  0.0 0.0 1.0\n\n"
        "atoms\n  1 : nspecies\n'H.in'\n  1 : natoms\n"
        "  0.0 0.0 0.0  0.0 0.0 0.0\n"
    )
    geometry = read_elk_geometry(path)
    assert geometry.avec == pytest.approx(np.diag([2.0, 6.0, 2.0]))
    assert geometry.omega == pytest.approx(24.0)


def test_the_committed_geometry_is_the_cell_the_defumat_input_uses(state):
    """3.0 bohr simple cubic, the origin unshifted for one atom at a corner."""
    assert state.geometry.avec == pytest.approx(np.eye(3) * 3.0, abs=1e-12)
    assert state.geometry.positions == pytest.approx(np.zeros((1, 3)), abs=1e-12)


# --- the refusals -------------------------------------------------------------

def write_header(path, *, spinpol=False, dftu=0, ftmtype=0, version=(11, 0, 2)):
    """A synthetic ``STATE.OUT`` that stops after the header.

    Every refusal fires before the first density record, so a file with no
    density records at all is enough to test them -- and a file that is *only*
    a header is also the sharpest test that the header's record order is what
    the reader thinks it is.
    """
    from scipy.io import FortranFile

    with FortranFile(path, "w") as f:
        f.write_record(np.array(version, dtype=np.int32))
        f.write_record(np.array([int(spinpol)], dtype=np.int32))
        f.write_record(np.array([1], dtype=np.int32))          # nspecies
        f.write_record(np.array([49], dtype=np.int32))         # lmmaxo
        f.write_record(np.array([12], dtype=np.int32))         # nrmtmax
        f.write_record(np.array([6], dtype=np.int32))          # nrcmtmax
        f.write_record(np.array([1], dtype=np.int32))          # natoms(1)
        f.write_record(np.array([12], dtype=np.int32))         # nrmt(1)
        f.write_record(np.linspace(1e-6, 1.4, 12))             # rsp(:, 1)
        f.write_record(np.array([6], dtype=np.int32))          # nrcmt(1)
        f.write_record(np.linspace(1e-6, 1.4, 6))              # rcmt(:, 1)
        f.write_record(np.array([4, 4, 4], dtype=np.int32))    # ngridg
        f.write_record(np.array([13], dtype=np.int32))         # ngvec
        f.write_record(np.array([0], dtype=np.int32))          # ndmag
        f.write_record(np.array([1], dtype=np.int32))          # nspinor
        f.write_record(np.array([0], dtype=np.int32))          # fsmtype
        f.write_record(np.array([ftmtype], dtype=np.int32))
        f.write_record(np.array([dftu], dtype=np.int32))
        f.write_record(np.array([0], dtype=np.int32))          # lmmaxdm
        f.write_record(np.array([0], dtype=np.int32))          # xcgrad
        f.write_record(np.array([0.1]))                        # efermi
        f.write_record(np.array([0.0]))                        # dlefe
    return path


def write_geometry(path):
    path.write_text(
        "avec\n  3.0 0.0 0.0\n  0.0 3.0 0.0\n  0.0 0.0 3.0\n\n"
        "atoms\n  1 : nspecies\n'H.in'\n  1 : natoms\n"
        "  0.0 0.0 0.0  0.0 0.0 0.0\n"
    )
    return path


@pytest.mark.parametrize(
    "kwargs, wanted",
    [
        ({"spinpol": True}, "spin"),
        ({"dftu": 1}, "dftu"),
        ({"ftmtype": 1}, "tensor moment"),
        ({"version": (1, 4, 18)}, "2.0.0"),
    ],
)
def test_a_state_carrying_physics_this_cannot_transfer_is_refused(
    tmp_path, kwargs, wanted
):
    """Each of these is a further field, not a format detail.

    They are refused at *read* time rather than at use time: a state whose
    moment or whose occupation matrices are silently dropped is a wrong answer
    that starts, and the place to stop it is before anything has been built
    from it.
    """
    write_header(tmp_path / "STATE.OUT", **kwargs)
    write_geometry(tmp_path / "GEOMETRY.OUT")
    with pytest.raises(NotImplementedError, match=wanted):
        read_elk_state(tmp_path)


def test_a_run_directory_without_a_geometry_is_refused(tmp_path):
    """``STATE.OUT`` carries neither the cell nor the positions.

    And ``elk.in`` is not a substitute: Elk's default ``tshift`` moves the
    origin onto the inversion centre, and only ``GEOMETRY.OUT`` is written in
    that shifted frame.
    """
    write_header(tmp_path / "STATE.OUT")
    with pytest.raises(FileNotFoundError, match="GEOMETRY.OUT"):
        read_elk_state(tmp_path)


def test_a_directory_without_a_state_is_refused(tmp_path):
    write_geometry(tmp_path / "GEOMETRY.OUT")
    with pytest.raises(FileNotFoundError, match="STATE.OUT"):
        read_elk_state(tmp_path)


def test_a_geometry_from_a_different_run_is_refused(tmp_path):
    """The species count is the one cross-check the two files allow.

    It is worth making because the muffin-tin record is dimensioned by
    ``natmtot``, which comes from the geometry: a mismatched pair reads the
    wrong number of bytes rather than the wrong numbers.
    """
    write_header(tmp_path / "STATE.OUT")
    (tmp_path / "GEOMETRY.OUT").write_text(
        "avec\n  3.0 0.0 0.0\n  0.0 3.0 0.0\n  0.0 0.0 3.0\n\n"
        "atoms\n  2 : nspecies\n'H.in'\n  1 : natoms\n"
        "  0.0 0.0 0.0  0.0 0.0 0.0\n"
        "'He.in'\n  1 : natoms\n  0.5 0.5 0.5  0.0 0.0 0.0\n"
    )
    with pytest.raises(ValueError, match="not the same run"):
        read_elk_state(tmp_path)


def test_the_two_arrays_in_one_record_framing_is_asserted(tmp_path):
    """The framing trap, made to fire.

    ``write(100) rhomt, rhoir`` is a *single* record with both arrays
    concatenated. A file whose density record is the wrong length is what a
    reader that expects one array per record would produce, and the error says
    so rather than reporting a shape.
    """
    from scipy.io import FortranFile

    path = write_header(tmp_path / "STATE.OUT")
    write_geometry(tmp_path / "GEOMETRY.OUT")
    with open(path, "ab") as handle:
        with FortranFile(handle, "w") as f:
            f.write_record(np.zeros(49 * 12 * 1))   # rhomt alone, no rhoir
    with pytest.raises(ValueError, match="ONE record"):
        read_elk_state(tmp_path)


# --- the two traps only a second species can reach ----------------------------
#
# Both of these are handled by construction in the reader and were untested by
# anything until zincblende SiC was committed beside hydrogen. Neither produces
# an error when it is got wrong: the first reads half the muffin-tin block and
# then desyncs, the second integrates leftover buffer as if it were density.
#
# It has to be SiC rather than BN. Elk's species files set ``nrmt`` by
# **periodic-table row** -- 300 for row 2, 400 for row 3, rounded at
# ``init0.f90:363`` -- and ``checkmt``/``autormt`` move ``rmt`` without ever
# touching ``nrmt``. Boron and nitrogen are both row 2, so a BN fixture would
# have ``nrmt(1) = nrmt(2) = nrmtmax`` and the padding would stay exactly as
# invisible as it is on hydrogen.

@needs_sic
def test_a_second_species_is_read_at_all(sic):
    """``natmtot`` is ``sum(natoms)``, and the record length says so.

    ``nspecies = 2`` with ``natoms = 1 1``, so a reader that wrote ``natoms(1)``
    would dimension the muffin-tin block to one atom, take the second atom's
    rows as interstitial density, and desync. The record-length equality is what
    turns that into a message instead of a plausible answer.
    """
    assert sic.geometry.species == ("C.in", "Si.in")
    assert sic.geometry.natoms == (1, 1)
    assert sic.natmtot == 2
    assert sic.natmtot != sic.geometry.natoms[0]
    assert list(sic.geometry.species_of()) == [0, 1]
    assert sic.rhomt.shape == (49, 397, 2)
    assert sic.rhoir.shape == (24, 24, 24)


@needs_sic
def test_the_padding_past_the_short_mesh_is_leftover_buffer(sic):
    """The rows past ``nrmt(is)`` are uninitialised, and here they are visibly so.

    Carbon is row 2 and silicon row 3, so their meshes are 297 and 397 points
    and ``nrmtmax`` is 397. Carbon's rows 298-397 in the file hold leftovers of
    order 1e-3 -- sixteen orders above the 1e-19 floor of a channel that is zero
    by symmetry, and a couple of per cent of the real ``l > 0`` density at the
    sphere boundary. Nothing about them looks like round-off, which is the
    point: a reader that integrates to ``nrmtmax`` gets a wrong charge, not an
    error.
    """
    assert sic.nrmt == (297, 397)
    assert sic.nrmtmax == 397

    carbon_padding = np.abs(sic.rhomt[:, 297:, 0])
    assert carbon_padding.max() > 1e-4

    # Silicon's mesh is the long one, so it has no padding to hold anything.
    assert sic.rhomt[:, :, 1].shape[1] == sic.nrmt[1] == sic.nrmtmax


@needs_sic
def test_the_charges_are_taken_over_each_species_own_mesh(sic):
    """``chgmt`` per atom against ``INFO.OUT``, which is the padding's test.

    Integrating carbon to ``nrmtmax`` instead of to ``nrmt(1)`` would add the
    leftovers above to its charge. These are the printed values, so the slicing
    is checked by the physics rather than by an assertion about array bounds.
    """
    charges = sic.muffin_tin_charges()
    assert charges[0] == pytest.approx(4.832618957, abs=1e-8)
    assert charges[1] == pytest.approx(11.73695755, abs=1e-7)
    assert float(charges.sum()) + sic.interstitial_charge() == pytest.approx(
        20.0, abs=1e-8
    )


@needs_sic
def test_the_radial_meshes_and_radii_are_per_species(sic):
    """``rmt(is) = rsp(nrmt(is), is)``, which differs between the two species.

    Silicon's sphere is 22 per cent larger than carbon's, and neither is the
    radius the species file asked for -- ``autormt`` sets them from the bond
    length. A reader that took one radius for the cell would put interstitial
    points inside a sphere.
    """
    assert sic.rmt[0] == pytest.approx(1.582809072, abs=1e-9)
    assert sic.rmt[1] == pytest.approx(1.934544422, abs=1e-9)
    assert len(sic.rsp[0]) >= sic.nrmt[0]
    assert len(sic.rsp[1]) >= sic.nrmt[1]
    assert list(sic.nri) == [201, 273]


@needs_sic
def test_a_gradient_functional_state_is_read(sic, state):
    """``xcgrad`` is 1 here and 0 on hydrogen, and it is a versioned field.

    It only exists in the header from Elk 5.1, so a reader that assumed it was
    always there -- or never -- reads every field after it at the wrong offset.
    Hydrogen alone cannot tell the two mistakes apart, because its value is the
    same as the fallback.
    """
    assert sic.xcgrad == 1
    assert state.xcgrad == 0
    assert sic.efermi == pytest.approx(2.0 * 0.33148292208740043, rel=1e-12)
