"""Does a **vector** magnetic texture survive an SCF, and does symmetry decide?

The collinear half of this question was closed earlier: a two-site
antiferromagnet stated on one species survives with symmetry on, where before the
filter existed it converged cleanly to the nonmagnetic state 6.6 meV up
(``tests/unit/test_collinear_symmetry.py``). A collinear texture is a scalar with
a sign, though, and the symmetriser that averages it is the *scalar* one. A
cycloid is a genuine vector field: the magnetization rotates from site to site,
an operation of the group rotates it **and** permutes the sites, and nothing
smaller than the magnetic group leaves it alone.

**It survives, and the two runs agree.** Four hydrogens along z with each moment
turned 90 degrees from the one before, converged twice at ``conv_thr = 1e-8`` --
once with the magnetic group (``nsym = 4`` out of the lattice's 16) and once with
``nosym``:

                         nosym          symmetry on
    iterations              4                4
    total energy (Ry)   -3.8187830180    -3.8187830260
    site moment (mu_B)   0.4542           0.4543
    neighbour angles     90.00 x4         90.00 x4
    singular values     0.64240, 0.64239, 8.1e-6    ..., 1.7e-21

8.0e-9 Ry apart, and site by site to 8e-5 mu_B. The third singular value is the
one place they differ, and it differs the right way round: the four surviving
operations *forbid* an out-of-plane component, so the symmetrised run is planar
to 1.7e-21 where the free one finds planarity only to 8e-6. Symmetry buys
exactness here rather than costing physics.

**The discriminator is a pair, and this file exists partly to say why.** "Two
nonzero singular values of the ``(nat, 3)`` site-moment matrix means the texture
held" is scale-free, and on the same cell at **3 bohr** spacing instead of 5 it
gives ``[3.235e-4, 3.231e-4, 1.4e-23]`` with neighbour angles 90.06/89.89/89.99/
90.05 and the two runs agreeing to 1.1e-8 Ry. A perfect cycloid by that reading,
and nothing at all: the moments had fallen from 0.1465 at iteration 1 by a factor
of **450**, because a hydrogen chain is not magnetic at 3 bohr. So both halves are
asserted below -- ``sigma_2/sigma_1`` for the shape, and ``sigma_1`` against its
own value at iteration 1 for whether there is anything left to have a shape.
"""

from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
import pytest

from defumat import Calculator

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"

#: How far the symmetrised run may sit from the free one. Both converge to
#: ``conv_thr = 1e-8`` and land 8.0e-9 Ry apart, which is that threshold.
ENERGY_TOLERANCE = 5.0e-8

#: ...and site by site, in Bohr magnetons. Measured at 8e-5.
MOMENT_TOLERANCE = 5.0e-4


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """Two cells that share no shape compile the SCF stack twice over, and XLA
    keeps every executable for the life of the process. The results stay cached
    below; only the compiled code is dropped."""
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _converged(stem, pseudo_dir):
    calculator = Calculator.from_text(
        (CASES / f"{stem}.in").read_text(), pseudo_dir, announce=False
    )
    scf = calculator.get_scf(max_iterations=200)
    assert scf.converged
    return scf


def _shape(moments):
    """``(planarity, sigma_1)``: how flat the texture is, and how big it is.

    ``planarity`` is ``sigma_2 / sigma_1``, which is 1 for a texture whose
    moments span a plane evenly and 0 for a collinear one. ``sigma_1`` carries
    the scale the ratio has thrown away.
    """
    sv = np.linalg.svd(np.asarray(moments), compute_uv=False)
    return sv[1] / sv[0], sv[0], sv[2] / sv[0]


def test_a_ninety_degree_cycloid_survives_and_stays_planar(pseudo_dir):
    """Both halves of the discriminator, on the free run.

    ``nosym``, so nothing is averaged and what is left is what the energy surface
    keeps. The moment must still be *there* -- which the singular-value ratio
    alone cannot say, and is the half the queue's own criterion left out.
    """
    scf = _converged("h4-cycloid-90-nosym", pseudo_dir)
    moments = np.asarray(scf.site_moments)
    assert moments.shape == (4, 3)

    planarity, sigma1, out_of_plane = _shape(moments)
    first = _shape(np.asarray(scf.history[0]["site_moments"]))[1]

    # The shape: four directions evenly spanning a plane.
    assert planarity > 0.99
    assert out_of_plane < 1e-4
    # And the scale, which is the half that catches a collapse to nothing. The
    # moment grows very slightly here rather than shrinking.
    assert sigma1 / first > 0.9
    assert np.linalg.norm(moments, axis=1).min() > 0.4

    # The angles, which is what "cycloid" means read directly.
    hats = moments / np.linalg.norm(moments, axis=1)[:, None]
    angles = np.degrees(np.arccos(np.clip(
        [hats[i] @ hats[(i + 1) % 4] for i in range(4)], -1.0, 1.0
    )))
    assert angles == pytest.approx(90.0, abs=0.1)


def test_the_magnetic_group_keeps_the_texture_the_free_run_keeps(pseudo_dir):
    """P77's filter, on a texture that is a vector field rather than a sign.

    The comparison is the whole point: a group too large for the state does not
    converge to a worse answer, it converges to a *different* one -- the same
    cycloid handed to this cell's ferromagnetic group has its magnetization
    removed entirely (P80, ``tests/unit/test_seed_symmetry.py``). So agreement
    site by site is the statement that the four operations the filter kept are
    the texture's own.
    """
    free = _converged("h4-cycloid-90-nosym", pseudo_dir)
    symmetric = _converged("h4-cycloid-90", pseudo_dir)

    assert symmetric.total_energy == pytest.approx(
        free.total_energy, abs=ENERGY_TOLERANCE
    )
    assert np.asarray(symmetric.site_moments) == pytest.approx(
        np.asarray(free.site_moments), abs=MOMENT_TOLERANCE
    )

    # Symmetry buys exactness on the component the group forbids: the
    # symmetrised run is planar to round-off where the free one is planar to
    # 1e-5. Both are "planar"; only one is planar by construction.
    assert _shape(symmetric.site_moments)[2] < 1e-15
    assert _shape(free.site_moments)[2] > 1e-7
