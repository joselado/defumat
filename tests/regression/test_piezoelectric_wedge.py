"""The wedge half of the augmented piezoelectric pair, in a file of its own.

``test_piezoelectric_augmented.py`` says the two assemblies are the same mixed
second derivative, on the whole unshifted grid with no symmetry. This says the
**wedge sum** has been completed, which on an augmented dataset is a separate
statement: one term of this derivative is quadratic in a per-k tangent, so no
average of the finished rank-3 tensor can repair it and one factor has to be
made whole before it is contracted.

**It is a separate file for memory and the reason is worth reading once.** Each
of the two cells peaks at about 10 GiB of ``tools/run_regression.sh``'s 12 GiB
cap, and a file boundary is a process boundary under that runner. The two tests
*did* pass together once, at 5m52, and then failed with the same code the next
time they were run -- because the first run compiled the second cell's kernels
and the second read them back off ``~/.cache/defumat/jax``. That is
``CLAUDE.md``'s cache rule with its sign reversed: **a cache miss is cheaper in
memory and dearer in time**, so a two-cell file measured cold is not a two-cell
file that runs warm, and the passing measurement was of a cache state that no
longer existed by the time it was believed.
"""

import numpy as np
import pytest

from tests.regression.test_piezoelectric_augmented import (  # noqa: F401
    _bounded_compilation, _both_routes, _field,
)

pytestmark = [pytest.mark.regression, pytest.mark.slow]

#: The same cell and the same unshifted grid as
#: ``test_piezoelectric_augmented.py``'s with the point group kept, so that
#: eight k-points reduce to three.
WEDGE = "alas-piezo-tiny-wedge"

#: ``e_14`` on the closed grid, C/m^2, which is what
#: ``test_the_two_routes_agree_on_an_augmented_dataset`` asserts on the cell
#: beside this one -- quoted rather than recomputed, because recomputing it is
#: the second ground state this file exists to avoid holding.
CLOSED_GRID_E14 = 1.474377366


def test_the_wedge_completes_and_the_taped_route_is_where_it_did_not():
    """The pair that says a wedge sum of a *product* has been completed.

    On an augmented dataset the density moves with the strain at frozen states,
    so the mixed second derivative carries ``int (drho/d(eps)) K (drho/dE)`` --
    a product of two per-k tangents. A wedge sum of a product is not the product
    of the full-zone objects, so averaging the finished rank-3 tensor cannot
    repair it: one factor has to be made whole *before* it is contracted, which
    is :func:`~defumat.response.born._full_zone_field_response` and is P36's
    rule. The two routes need it in different amounts and that is the point of
    asserting both: the contracted one's screening factor is the field's
    converged ``dvscf``, which ``dielectric_tensor`` mixes from the
    **symmetrised** density response and which is therefore full-zone already,
    while the taped one builds its own from unsymmetrised builders.

    Measured against the code of this morning, on this cell: the taped route
    gave 1.475427270 C/m^2 on the wedge against 1.474377366 on the closed grid,
    **1.05e-03** apart, where the contracted one split by 4.8e-06 -- a factor of
    219, which is what says the defect was the taped route's and not the
    symmetriser's. With the completion in, the two agree **on the wedge** to
    3.4e-08 and both sit 4.8e-06 from the closed grid -- which is **not** this
    assembly's and is why the closed-grid tolerance below is 1e-4 rather than
    1e-6: the dielectric constant the two routes share splits between the same
    two cells by 1.573e-05 relative, five times larger, so nothing is left for
    the strain leg to be blamed for. ``OPEN.md`` carries that one.

    The closed-grid value is quoted rather than recomputed so this costs one
    ground state; it is the number
    :func:`test_the_two_routes_agree_on_an_augmented_dataset` asserts on the
    cell beside this one.
    """
    taped, contracted = _both_routes(WEDGE)
    # The identity again, and on three k-points it is the sharper of the two
    # statements: this is what was 1.05e-03 before the completion went in.
    assert abs(taped - contracted) < 1e-6, (
        f"the two routes disagree on the wedge by {taped - contracted:.3e} C/m^2"
    )
    # ... and the wedge is the closed grid, which is what the pair exists for.
    assert abs(taped - CLOSED_GRID_E14) < 1e-4
    assert abs(contracted - CLOSED_GRID_E14) < 1e-4
    # Three points, not eight: if the reduction stopped happening the test above
    # would still pass and would be comparing the cell with itself.
    calculation, *_ = _field(WEDGE)
    assert calculation.system.kpoints.nk == 3
