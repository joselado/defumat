"""P17: a noncollinear run that actually carries a magnetization.

P14 built the spinor machinery and validated it where the density has **one**
component -- a spin-orbit run on a nonmagnetic crystal. This is the other case,
``nspin_mag = 4``, where the density is ``(n, m_x, m_y, m_z)`` and three things
that were deferred become necessary: the symmetry group is smaller than the
crystal's, the operations that survive may need time reversal, and symmetrising
the magnetization means *rotating* it rather than averaging three scalars.

Three kinds of check, in increasing order of how much they involve QE:

**Identities**, which involve QE not at all. Without spin-orbit coupling nothing
in the Hamiltonian knows which way spin points, so (a) a noncollinear run with
every moment along ``z`` must reproduce the collinear LSDA run, and (b) the
total energy must not depend on the direction the moments point in. The second
fails on any error in the rotation of the magnetization or in which operations
were kept, and it is sharp: rotation invariance is exact, not approximate.

**Group counts**, which compare against what QE *prints*: bcc iron with its
moment along ``x`` has 16 symmetry operations of the 48 the lattice has, and
QE's own header says 16.

**The benchmark**, ``pw_noncolin/noncolin.in``: total energy, moment, and the
local moment inside the sphere ``make_pointlists`` builds.

*A caveat that is QE's and not this code's.* The identity (a) is exact only
where the density is non-negative. ``v_xc``'s ``nspin = 4`` branch integrates
``e_xc`` against ``ABS(rho(ir,1))``, where the ``nspin = 1`` and ``2`` branches
use the signed density; in a cell with vacuum, where a truncated plane-wave
density rings slightly negative, the two conventions differ -- 2.6e-4 Ry on a
PAW oxygen atom whose density is negative on a fifth of the grid. So the
identity is checked on a strictly positive density (hydrogen, and iron's dense
crystal) and the *rotation* invariance, which does not depend on the
convention, is what the open-shell PAW case is asked for.
"""

from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from pypresso.io import read_qe_output
from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.scf import run_scf
from pypresso.scf.driver import Calculation
from pypresso.scf.locals import build_local_regions, default_radii, get_locals
from pypresso.system import build_system
from pypresso.system.builder import local_moments
from pypresso.system.symmetry import find_symmetries, magnetic_symmetries
from tests.conftest import GENERATED
from tests.tolerances import TOTAL_ENERGY_RY

pytestmark = [pytest.mark.regression]

#: The identity between two runs of *this* code is a round-off comparison, not a
#: tolerance borrowed from a QE benchmark: both numbers come from the same
#: machinery and any difference between them is a bug.
IDENTITY_RY = 1e-9


def _pseudos(system, pseudo_dir: Path):
    return tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)


@lru_cache(maxsize=None)
def _run(path: Path, pseudo_dir: Path):
    system = build_system(read_pw_input(path))
    return system, run_scf(system, _pseudos(system, pseudo_dir), conv_thr=1e-10,
                           max_iterations=100)


def _variant(text: str, replacements, directory: Path, name: str) -> Path:
    for old, new in replacements:
        assert old in text, old
        text = text.replace(old, new)
    path = directory / name
    path.write_text(text)
    return path


def _noncollinear(text: str, theta: float, phi: float) -> str:
    """The collinear input with the moment turned to ``(theta, phi)`` degrees."""
    return text.replace(
        "    nspin = 2",
        f"    noncolin = .true.\n    angle1(1) = {theta}, angle2(1) = {phi}",
    )


def test_magnetic_symmetry_group_matches_qe(qe_testsuite):
    """bcc iron with its moment along x keeps 16 of the lattice's 48 operations.

    Half of those 16 are symmetries only together with time reversal -- an
    operation that sends the moment to minus its image is one, because the
    magnetization is odd under time reversal and the crystal is not. QE prints
    "16 Sym. Ops., with inversion, found" for exactly this input.
    """
    path = qe_testsuite / "pw_noncolin" / "noncolin.in"
    system = build_system(read_pw_input(path))
    assert system.nspin_mag == 4

    full = find_symmetries(system.cell, system.structure)
    moments = local_moments(system.structure, 4, system.starting_magnetization,
                            system.angle1, system.angle2)
    magnetic = magnetic_symmetries(system.cell, system.structure, full, moments)

    assert full.nsym == 48
    assert magnetic.nsym == 16
    assert sum(magnetic.time_reversed) == 8
    assert system.symmetry_group().nsym == 16


def test_explicit_kpoints_are_completed(qe_testsuite, benchmark):
    """The 11-point input list becomes the 22 points QE runs.

    ``irreducible_BZ`` treats an explicit list as the wedge of the *lattice's*
    point group and completes it for the crystal's -- which does nothing at all
    until the crystal's group is smaller, and pointing a moment along ``x`` in a
    cubic crystal is exactly when it is.
    """
    system = build_system(read_pw_input(qe_testsuite / "pw_noncolin" / "noncolin.in"))
    reference = read_qe_output(benchmark("pw_noncolin", "noncolin.in"))

    assert system.kpoints.nk == len(reference.kpoints) == 22
    ours = np.sort(np.asarray(system.kpoints.weights))
    theirs = np.sort(np.asarray(reference.weights))
    assert ours == pytest.approx(theirs, abs=1e-6)


@pytest.mark.slow
def test_noncollinear_reproduces_lsda(pseudo_dir, tmp_path_factory):
    """Moments along z, and nothing else changed: the same answer as LSDA.

    Term by term, because the total energy is a sum that could hide two
    cancelling errors.
    """
    text = (GENERATED / "h-atom-lsda.in").read_text()
    directory = tmp_path_factory.mktemp("noncollinear-identity")
    collinear = _variant(text, [], directory, "lsda.in")
    along_z = _variant(_noncollinear(text, 0.0, 0.0), [], directory, "nc-z.in")

    reference_system, reference = _run(collinear, pseudo_dir)
    system, result = _run(along_z, pseudo_dir)

    assert reference_system.nspin == 2 and system.nspin_mag == 4
    assert reference.converged and result.converged
    assert result.total_energy == pytest.approx(reference.total_energy, abs=IDENTITY_RY)
    for term, value in reference.energy_terms.items():
        assert result.energy_terms[term] == pytest.approx(value, abs=1e-6), term

    # The moment comes out along z and nowhere else.
    m = np.asarray(result.magnetization_vector)
    assert m[2] == pytest.approx(reference.magnetization, abs=1e-6)
    assert np.abs(m[:2]).max() < 1e-12


@pytest.mark.slow
@pytest.mark.parametrize(
    ("theta", "phi"), [(90.0, 0.0), (90.0, 90.0), (54.7356103172, 45.0)]
)
def test_total_energy_does_not_depend_on_the_direction(
    theta, phi, pseudo_dir, tmp_path_factory
):
    """Rotating every moment rigidly changes nothing. It is a symmetry, so it is exact.

    This is the check the whole phase rests on: it fails if the magnetization is
    symmetrised as three scalars, if the axial sign is dropped, if a
    time-reversed operation is treated as an ordinary one, or if the magnetic
    filter kept an operation that reverses the moment.
    """
    text = (GENERATED / "h-atom-lsda.in").read_text()
    directory = tmp_path_factory.mktemp(f"rotation-{int(theta)}-{int(phi)}")
    along_z = _variant(_noncollinear(text, 0.0, 0.0), [], directory, "nc-z.in")
    turned = _variant(_noncollinear(text, theta, phi), [], directory, "nc-turned.in")

    _, reference = _run(along_z, pseudo_dir)
    system, result = _run(turned, pseudo_dir)

    assert result.converged
    assert result.total_energy == pytest.approx(reference.total_energy, abs=IDENTITY_RY)

    # ... and the moment points where it was asked to.
    direction = np.array([
        np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi)),
        np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi)),
        np.cos(np.deg2rad(theta)),
    ])
    m = np.asarray(result.magnetization_vector)
    assert m == pytest.approx(np.linalg.norm(m) * direction, abs=1e-8)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("theta", "phi"), [(90.0, 0.0), (54.7356103172, 45.0)]
)
def test_paw_one_centre_terms_are_rotation_invariant(
    theta, phi, pseudo_dir, tmp_path_factory
):
    """The same invariance with PAW's one-centre terms in the sum.

    PAW adds a second place where the magnetization has to be handled as a
    vector -- the on-sphere exchange-correlation, which resolves it onto a local
    axis exactly as the plane-wave one does, and ``becsum``'s symmetrisation,
    where the three components rotate into each other. Both are exercised here
    and neither is by the norm-conserving case above.
    """
    text = (GENERATED / "o-atom-lsda.in").read_text()
    directory = tmp_path_factory.mktemp(f"paw-rotation-{int(theta)}-{int(phi)}")
    along_z = _variant(_noncollinear(text, 0.0, 0.0), [], directory, "nc-z.in")
    turned = _variant(_noncollinear(text, theta, phi), [], directory, "nc-turned.in")

    _, reference = _run(along_z, pseudo_dir)
    system, result = _run(turned, pseudo_dir)

    assert result.converged
    assert result.total_energy == pytest.approx(reference.total_energy, abs=IDENTITY_RY)
    assert result.energy_terms["one_center_paw"] == pytest.approx(
        reference.energy_terms["one_center_paw"], abs=1e-8
    )


@pytest.mark.slow
def test_iron_total_energy_and_moment(qe_testsuite, pseudo_dir, benchmark):
    """``pw_noncolin/noncolin.in``: bcc iron, ultrasoft, LDA, moment along x."""
    reference = read_qe_output(benchmark("pw_noncolin", "noncolin.in"))
    system, result = _run(qe_testsuite / "pw_noncolin" / "noncolin.in", pseudo_dir)

    assert system.nspin_mag == 4
    assert result.converged
    assert result.total_energy == pytest.approx(reference.total_energy, abs=TOTAL_ENERGY_RY)

    # QE prints two decimals, and the moment is a bulk quantity that both codes
    # locate through their own Fermi level, so this is a physics check rather
    # than a numerical one.
    m = np.asarray(result.magnetization_vector)
    assert m == pytest.approx(np.asarray(reference.magnetization_vector), abs=5e-3)
    assert result.absolute_magnetization == pytest.approx(
        reference.absolute_magnetization, abs=5e-3
    )


@pytest.mark.slow
def test_local_moment_matches_qe(qe_testsuite, pseudo_dir, benchmark):
    """``make_pointlists`` and ``get_locals``, against what QE prints.

    Two independent things: the radius QE derives when the input does not give
    one (half the nearest-neighbour distance, over the 1.2 taper, less 1%), and
    the charge and moment integrated in that sphere with the linear taper.
    """
    reference = read_qe_output(benchmark("pw_noncolin", "noncolin.in"))
    if reference.r_m is None or reference.local_charges is None:
        pytest.skip("this reference does not print the local moments")
    system, result = _run(qe_testsuite / "pw_noncolin" / "noncolin.in", pseudo_dir)

    radii = default_radii(system.cell, system.structure)
    assert radii == pytest.approx(np.asarray(reference.r_m), abs=1e-4)

    calculation = Calculation(system, _pseudos(system, pseudo_dir))
    regions = build_local_regions(
        system.cell, system.structure, calculation.basis.dense.grid
    )
    charge, moment = get_locals(jnp.asarray(result.density), regions, system.cell)

    assert np.asarray(charge) == pytest.approx(reference.local_charges, abs=1e-4)
    assert np.asarray(moment) == pytest.approx(reference.local_moments, abs=1e-3)
