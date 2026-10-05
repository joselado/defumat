"""Under half-sphere storage the Davidson's plane-wave products are real, as ``regterg``'s are.

``OPEN.md`` Part XXIII item 19. ``regterg``, ``calbec_gamma`` and
``add_vuspsi_gamma`` read a gamma-point state as a real array of length
``2 npw`` and contract it with DGEMM. The gamma Davidson here made its projected
matrices real but formed them, both Ritz rotations, ``calbec`` and the nonlocal
unproject as complex products, four real multiply-adds an element where two
suffice, with the imaginary half of the result either discarded or exactly zero.
It now carries the states as real planes for the whole solve, so the compiled
solve holds no complex product at all; the transform inside ``h_psi`` is the
only place a complex block is rebuilt, one band chunk at a time.

The structural check reads the compiled program, which is what was wrong: the
numbers were right before and are right now, to round-off. Run against the
complex route, it fails with every dot of the solve complex (37 of 37 on
``si16-gamma-ecut30``). The energy check pins the complex route's own number on
the same cell.
"""

import re
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat.basis.fft import gamma_inner
from defumat.calculator import Calculator
from defumat.solvers import davidson

pytestmark = pytest.mark.unit

#: The smallest cell that consumes the half sphere: ``tests/regression/
#: test_gamma_only.py``'s two displaced silicon atoms, with ``nosym`` so that the
#: storage is not substituted away.
GAMMA = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.20, nat = 2, ntyp = 1, ecutwfc = 12.0,
  nosym = .true.
/
&electrons
  conv_thr = 1.0d-12
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS alat
 Si 0.01 0.00 0.00
 Si 0.26 0.24 0.25
K_POINTS gamma
"""

#: ``GAMMA``'s total energy on the complex route this replaced (``fbbf7ce``), in
#: Ry. The real route gave -14.515602274962971, 5e-15 away, with the same step
#: count at every SCF iteration and the same 27 steps from a cold start.
COMPLEX_ROUTE_ENERGY = -14.515602274962966

NBND = 4

#: The HLO name of each real dtype a dot may carry.
HLO_REAL = {np.dtype(np.float64): "f64", np.dtype(np.float32): "f32"}

DOT = re.compile(r"=\s*(\w+)\[([\d,]*)\]\S*\s+dot\(")


@pytest.fixture(scope="module")
def calculator(pseudo_dir):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_text(GAMMA, pseudo_dir, announce=False)
    # Otherwise the run was substituted to the whole sphere and every check
    # below would pass vacuously on the k-point path.
    assert calculator.calculation.gamma_only
    return calculator


@pytest.fixture(scope="module")
def hamiltonian(calculator):
    calculation = calculator.calculation
    potential = jnp.zeros((calculation.nspin_mag,) + tuple(calculation.basis.dense.grid),
                          calculation.vltot.dtype)
    hamiltonian = calculation.hamiltonian(potential)[0]
    assert hamiltonian.gamma_only
    return hamiltonian


def test_the_compiled_gamma_solve_holds_no_complex_product(calculator, hamiltonian):
    """Every dot of the compiled solve is real, the rotations and rows among them.

    Compile only: nothing runs. The Ritz rotation is the real ``(nbnd, 2 npwx)``
    product and the projection rows the ``(nbnd, m)`` ones; on the complex route
    the same program has the same dots with a complex type.

    **One complex dot is left out by name**: the preconditioner's diagonal,
    ``usnldiag``'s ``gi,ij,gj->g`` of ``vkb`` and ``D``, built once per call
    before the loop, ``(npwx, nkb)``. It passed this test while Part III M5's
    block form wrote it as a broadcast and a sum, and came back with M5's
    revert (``dacdd8b``: the broadcast was not fused on a ten-atom spinor cell,
    3.6 against 12.7 GB).
    """
    psi0 = jnp.zeros((hamiltonian.nk, NBND, hamiltonian.ndim), hamiltonian.dtype)
    ethr = jnp.full((hamiltonian.nk, NBND), 1e-10, hamiltonian.kinetic.dtype)
    text = davidson._every_k.lower(
        hamiltonian, NBND, psi0, ethr, None, davidson.DAVID_NDIM,
        davidson.MAX_ITERATIONS, calculator.calculation.k_batch, robust=False,
        return_steps=True, return_finite=True,
    ).compile().as_text()

    dots = [found for line in text.splitlines() if " dot(" in line
            and "gi,ij,gj->g" not in line for found in DOT.findall(line)]
    real = HLO_REAL[np.dtype(jnp.finfo(hamiltonian.dtype).dtype)]
    kinds = sorted({kind for kind, _ in dots})
    assert dots, "no dot in the compiled solve: the pattern no longer reads the HLO"
    assert kinds == [real], f"complex products in the gamma solve: {kinds}"
    shapes = {shape for _, shape in dots}
    # A card's compiler may lay a dot out transposed (``170,4`` for ``4,170``),
    # so a shape is looked for in either order.
    shapes |= {",".join(reversed(shape.split(","))) for shape in shapes}
    # the Ritz rotation, over both planes ...
    assert f"{NBND},{2 * hamiltonian.npwx}" in shapes
    # ... and the projected rows at the first width, which only the CPU's
    # compiler keeps as a dot of its own: a card's folds it into another.
    if jax.default_backend() == "cpu":
        assert f"{NBND},{NBND}" in shapes


def test_the_planes_operator_is_the_complex_one(hamiltonian):
    """``apply_projected_planes`` and the planes product against the complex forms.

    The same arithmetic in another layout, so round-off apart: ``H`` of a random
    real-``G = 0`` block, and the gamma overlap matrix, the doubling and the
    ``G = 0`` term included.
    """
    # here rather than at the top, so that the two checks above and below can be
    # run against the complex route, which has none of these
    from defumat.hamiltonian.operator import from_planes, planes_inner, to_planes

    rng = np.random.default_rng(0)
    shape = (NBND, hamiltonian.npwx)
    block = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    block[:, 0] = block[:, 0].real
    block = jnp.where(hamiltonian.mask[0], jnp.asarray(block, hamiltonian.dtype), 0.0)

    applied = hamiltonian.apply(block, 0)
    planes, becp, becq = hamiltonian.apply_projected_planes(to_planes(block), 0)
    assert planes.dtype == jnp.finfo(hamiltonian.dtype).dtype
    assert becp.shape == becq.shape == (NBND, 0)   # norm-conserving: no S
    scale = float(jnp.max(jnp.abs(applied)))
    np.testing.assert_allclose(np.asarray(from_planes(planes)), np.asarray(applied),
                               atol=1e-13 * scale)

    overlap = gamma_inner(block[:, None, :], block[None, :, :], True)
    np.testing.assert_allclose(np.asarray(planes_inner(to_planes(block), to_planes(block))),
                               np.asarray(overlap), atol=1e-12)


def test_the_gamma_scf_energy_is_the_complex_routes(calculator):
    """The total energy on the real route, against the complex route's own."""
    result = calculator.get_scf()
    assert result.converged
    assert float(result.total_energy) == pytest.approx(COMPLEX_ROUTE_ENERGY, abs=1e-12)
