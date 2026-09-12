"""``||rho - sym(rho)|| / ||rho||`` on the density a run starts from.

The group a run symmetrises with is fixed before iteration 1, so a seed that
group does not leave alone is a seed whose non-invariant part the first
iteration averages away -- after which the run converges cleanly and nothing in
its output says what was removed. That is P75's failure: a four-atom cycloid
handed in under a group built for a ferromagnet went collinear between
iterations 3 and 6 and then converged for another seventeen, and the only
surviving evidence was the per-site moments.

**The check has to be on the seed and not on an output density.** An output
density is a sum over an irreducible wedge, which is not invariant by
construction -- putting the rest of the zone back is what ``sym_rho`` is for --
so the same number computed there is large exactly when symmetry is working and
cannot be told from the failure. On a seed there is no such ambiguity, and the
measurement below says how little: **6e-16 against 1.0**, with nothing in
between, which is what makes :data:`~defumat.scf.driver.SYMMETRY_SEED_RESIDUAL`
a threshold on a discriminator rather than a physical tolerance.

**The charge cannot see it.** In the failing case the charge residual is 5.7e-16
while the magnetization's is 1.0 -- the group leaves the charge exactly alone and
destroys the whole texture -- so a residual computed on the density as one
object, or on its charge, reads as a clean pass. That is why
:meth:`~defumat.scf.driver.Calculation.symmetry_residual` returns the two apart.
"""

from pathlib import Path

import numpy as np
import pytest

from defumat.io.pwin import parse_pw_input
from defumat.pseudo import read_upf
from defumat.scf import Calculation
from defumat.scf.driver import SYMMETRY_SEED_RESIDUAL
from defumat.system.builder import build_system

pytestmark = pytest.mark.unit

PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: Four hydrogens along ``z`` in a cube, one species, noncollinear. Simple
#: tetragonal so the lattice group is large and the magnetic filter has
#: something to cut: 16 operations for a ferromagnet along ``z``, 4 for the
#: cycloid below.
HEAD = """\
 &control
    calculation = 'scf'
 /
 &system
    ibrav = 6, celldm(1) = 12.0, celldm(3) = 1.0,
    nat = 4, ntyp = 1,
    ecutwfc = 20.0,
    occupations = 'smearing', smearing = 'gaussian', degauss = 0.10
    noncolin = .true.
    starting_magnetization(1) = 0.6
 /
 &electrons
 /
ATOMIC_SPECIES
 H  1.008  H.pz-vbc.UPF
ATOMIC_POSITIONS (crystal)
 H 0.0 0.0 0.00
 H 0.0 0.0 0.25
 H 0.0 0.0 0.50
 H 0.0 0.0 0.75
K_POINTS {automatic}
 1 1 4 0 0 0
"""

#: A 90-degree-per-site cycloid in the xy plane, at the same length the
#: ``starting_magnetization`` above asks for.
CYCLOID = 0.6 * np.array(
    [(np.cos(t), np.sin(t), 0.0) for t in np.deg2rad([0, 90, 180, 270])]
)

CARD = "STARTING_MOMENTS\n" + "\n".join(
    f" {x:.10f} {y:.10f} {z:.10f}" for x, y, z in CYCLOID
) + "\n"


def _calculation(card=""):
    system = build_system(parse_pw_input(HEAD + card))
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    return system, Calculation(system, pseudos)


@pytest.fixture(scope="module")
def pair():
    """The same cell twice: a ferromagnet along ``z``, and the stated cycloid."""
    return _calculation(), _calculation(CARD)


def test_a_seed_from_a_card_is_invariant_under_its_own_group(pair):
    """The filter's whole purpose, measured as a residual rather than an nsym.

    ``sgam_at_mag`` cuts the group to the operations that preserve the stated
    texture, so a seed built from the card is invariant *exactly* -- 6e-16 here.
    That is the half of the discriminator that must read zero, and it reads zero
    for the large group as well as the small one, which is what says the number
    is measuring invariance and not group size.
    """
    (_, ferro), (_, cycloid) = pair
    assert ferro.symmetries.nsym == 16
    assert cycloid.symmetries.nsym == 4, "the magnetic filter did not cut the group"

    for calculation in (ferro, cycloid):
        charge, moment = calculation.symmetry_residual(calculation.starting_density())
        assert charge < 1e-12
        assert moment < 1e-12


def test_a_texture_handed_in_under_the_wrong_group_is_wholly_averaged_away(pair):
    """The case that must trip the guard, and the size of what it catches.

    The cycloid's own density, handed to the *ferromagnet's* group. The moments
    do not shrink and they do not rotate: they are removed, residual **1.0**,
    because averaging four directions 90 degrees apart over a group that
    permutes the four sites gives zero. The run that follows is a nonmagnetic
    one wearing a magnetic input file.
    """
    (_, ferro), (_, cycloid) = pair
    rho = cycloid.starting_density()

    charge, moment = ferro.symmetry_residual(rho)
    assert charge < 1e-12, "the charge is invariant; only the texture is not"
    assert moment == pytest.approx(1.0, abs=1e-6)
    assert moment > SYMMETRY_SEED_RESIDUAL

    # And what that means per site: 0.27 Bohr magnetons before, nothing after.
    _, before = cycloid.site_moments(rho)
    _, after = ferro.site_moments(ferro.symmetrize(rho))
    assert np.linalg.norm(np.asarray(before), axis=1).min() > 0.2
    assert np.linalg.norm(np.asarray(after), axis=1).max() < 1e-10


def test_run_scf_warns_before_the_first_iteration(pair):
    """The guard fires where it is useful: before any SCF work is done.

    ``max_iterations = 1`` is enough -- the warning is raised on the seed, not on
    a result -- and it is what keeps this test cheap. The message has to name
    the handed-in case rather than the card, because which of the two it is
    decides whether the fix is ``nosym`` or a bug report.
    """
    from defumat.scf import run_scf

    (ferro_system, ferro), (_, cycloid) = pair
    rho = cycloid.starting_density()
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in ferro_system.structure.species
    )
    with pytest.warns(RuntimeWarning, match="not invariant under the 16 symmetry"):
        run_scf(
            ferro_system, pseudos, calculation=ferro, starting_density=rho,
            max_iterations=1, verbose=False,
        )


def test_a_nosym_run_is_not_warned_about(pair):
    """Nothing is averaged where nothing is symmetrised.

    The escape the warning recommends must itself be silent, or the advice is
    noise. ``use_symmetry`` is the one gate, and it covers both ``nosym`` and a
    group that came out trivial.
    """
    (_, ferro), (_, cycloid) = pair
    system = build_system(parse_pw_input(
        HEAD.replace("noncolin = .true.", "noncolin = .true.\n    nosym = .true.")
        + CARD
    ))
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    free = Calculation(system, pseudos)
    assert not free.use_symmetry
    assert free.symmetry_residual(cycloid.starting_density()) == (0.0, 0.0)
