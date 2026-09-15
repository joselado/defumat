"""The ultracell energy assembly, without an SCF in it.

``PLAN.md`` P88 stage 4. Four conventions live in
:mod:`defumat.ultracell.energy` and each of them is the kind that goes wrong
silently: a term normalised per ultracell where it should be per unit cell is a
factor of ``N`` in a number that still looks like an energy; a field paired into
``deband`` with the wrong sign is an energy that moves the right way and by
twice the right amount; and a term left out of the sum is invisible on any cell
that does not carry it. So each is asserted here against arithmetic, in the
push gate, rather than against a converged run.

The physics numbers -- the ``N = 1`` null, the variational ladder against a
supercell -- are in ``tests/regression/test_ultracell.py``, where they need an
SCF.
"""

from pathlib import Path

import numpy as np
import pytest

import jax.numpy as jnp

from defumat.basis.builder import build_basis
from defumat.io.pwin import read_pw_input
from defumat.system import build_system
from defumat.ultracell.energy import (
    total_of,
    ultracell_energy,
    ultracell_entropy,
)
from defumat.ultracell.grid import Ultracell

ROOT = Path(__file__).resolve().parents[2]


class _Potential:
    """What :func:`ultracell_energy` reads off an ``UltracellPotential``."""

    def __init__(self, v_scf, ehart=0.0, etxc=0.0):
        self.v_scf = v_scf
        self.ehart = ehart
        self.etxc = etxc


@pytest.fixture(scope="module")
def silicon():
    system = build_system(read_pw_input(ROOT / "benchmarks" / "si-1k.in"))
    return system, build_basis(system)


def _pieces(shape, basis, cell, nspin_mag=1, seed=0):
    ultracell = Ultracell.build(shape, basis.dense.grid)
    rng = np.random.default_rng(seed)
    box = (nspin_mag,) + tuple(ultracell.grid)
    density = jnp.asarray(rng.random(box))
    v = jnp.asarray(rng.standard_normal(box))
    return ultracell, density, v


def test_the_total_is_the_terms_that_are_not_underscored(silicon):
    """``total_of`` sums what is in the total, which is not every entry.

    ``_field_energy`` and ``_deband`` are on the dict so that a caller can read
    them -- the first is the convention made visible, the second is the term
    whose pairing is the whole accuracy of the assembly -- and adding either to
    the total would be a double count.
    """
    system, basis = silicon
    ultracell, density, v = _pieces((2, 1, 1), basis, system.cell)
    terms = ultracell_energy(
        density, _Potential(v), _Potential(v, ehart=3.0, etxc=-1.0),
        band_energy=2.0, ultracell=ultracell, cell=system.cell, ewald=-7.0,
    )
    assert set(terms) >= {"one-electron", "hartree", "xc", "ewald"}
    assert total_of(terms) == pytest.approx(
        sum(value for key, value in terms.items() if not key.startswith("_"))
    )
    assert "_field_energy" in terms and "_deband" in terms


def test_every_term_is_per_unit_cell(silicon):
    """A tiled state gives the same energy whatever ``N`` is.

    This is the normalisation that a converged run cannot check cheaply and
    that a factor of ``N`` hides in: ``deband`` is an integral over the whole
    ultracell and the Hartree and exchange-correlation energies come in as
    ultracell totals, so all three carry a ``/N`` and **omitting it on one of
    them scales with the cell** rather than showing up as an error at ``N = 1``.
    """
    system, basis = silicon
    reference = None
    for shape in [(1, 1, 1), (2, 1, 1), (2, 2, 1), (3, 1, 1)]:
        ultracell = Ultracell.build(shape, basis.dense.grid)
        cells = ultracell.cells
        rng = np.random.default_rng(1)
        one = rng.random((1,) + tuple(basis.dense.grid))
        v_one = rng.standard_normal((1,) + tuple(basis.dense.grid))
        # The same lattice-periodic density and potential, tiled: the physical
        # state is identical in every cell, so the energy per cell must be too.
        density = ultracell.tile(jnp.asarray(one))
        v = ultracell.tile(jnp.asarray(v_one))
        terms = ultracell_energy(
            density, _Potential(v),
            _Potential(v, ehart=3.0 * cells, etxc=-1.0 * cells),
            band_energy=2.0, ultracell=ultracell, cell=system.cell, ewald=-7.0,
        )
        total = total_of(terms)
        if reference is None:
            reference = total
        assert total == pytest.approx(reference, rel=1e-12), shape


def test_the_field_leaves_the_total_and_is_reported_beside_it(silicon):
    """``int B . m`` is removed from the total, exactly, and returned.

    QE's and Elk's shared convention, and this package's: a field put in by
    hand has its Zeeman energy carried beside the total rather than inside it.
    The field is inside ``dV`` and therefore inside every eigenvalue, so the
    thing that removes it is pairing it into ``deband`` -- and the test that
    this is what happened is that the two totals differ by the *reported*
    field energy and by nothing else.
    """
    system, basis = silicon
    ultracell, density, v = _pieces((2, 1, 1), basis, system.cell, nspin_mag=2,
                                    seed=3)
    rng = np.random.default_rng(4)
    b = jnp.asarray(rng.standard_normal(tuple(ultracell.grid)))
    # ``add_bfield``'s two lines: ``v_up -= B``, ``v_dw += B``.
    zeeman = jnp.stack([-b, b])

    plain = ultracell_energy(
        density, _Potential(v), _Potential(v, ehart=3.0, etxc=-1.0),
        band_energy=2.0, ultracell=ultracell, cell=system.cell, ewald=-7.0,
    )
    fielded = ultracell_energy(
        density, _Potential(v), _Potential(v, ehart=3.0, etxc=-1.0),
        band_energy=2.0, ultracell=ultracell, cell=system.cell, ewald=-7.0,
        magnetic_potential=zeeman,
    )
    field_energy = fielded["_field_energy"]
    assert field_energy is not None and field_energy != 0.0
    # ... and without a field the entry is ``None`` rather than zero, because a
    # field whose Zeeman energy vanishes -- one perpendicular to every moment,
    # which a turning field passes through -- is not the same thing as no field.
    assert plain["_field_energy"] is None
    assert total_of(fielded) == pytest.approx(
        total_of(plain) - field_energy, rel=1e-12
    )

    # ... and what was removed is ``-int B . m`` on the same grid, computed
    # here from the magnetization rather than from the potential, so that a
    # sign living in ``add_bfield``'s two lines cannot cancel itself.
    element = ultracell.volume(system.cell) / density[0].size
    magnetization = np.asarray(density[0] - density[1])
    expected = -element * float(np.sum(np.asarray(b) * magnetization)) / ultracell.cells
    assert field_energy == pytest.approx(expected, rel=1e-12)


def test_the_spinor_field_needs_no_sign_of_its_own(silicon):
    """The same contraction reads ``-int B . m`` in both magnetic regimes.

    Collinear it is ``(v_up, v_dw) = (-B, +B)`` against ``(rho_up, rho_dw)``;
    spinor it is ``v(1:4) = -B`` against ``(n, m_x, m_y, m_z)``. Written as one
    sum over components the two are the same expression, which is why
    :func:`ultracell_energy` has no branch on ``nspin`` -- and this is the test
    that says the absent branch is absent because it is not needed.
    """
    system, basis = silicon
    ultracell = Ultracell.build((2, 1, 1), basis.dense.grid)
    rng = np.random.default_rng(5)
    grid = tuple(ultracell.grid)
    charge = rng.random(grid)
    m = rng.standard_normal((3,) + grid)
    b = rng.standard_normal((3,) + grid)

    density = jnp.asarray(np.concatenate([charge[None], m]))
    zeeman = jnp.asarray(np.concatenate([np.zeros_like(charge)[None], -b]))
    v = jnp.zeros_like(density)
    terms = ultracell_energy(
        density, _Potential(v), _Potential(v), band_energy=0.0,
        ultracell=ultracell, cell=system.cell, ewald=0.0,
        magnetic_potential=zeeman,
    )
    element = ultracell.volume(system.cell) / charge.size
    expected = -element * float(np.sum(b * m)) / ultracell.cells
    assert terms["_field_energy"] == pytest.approx(expected, rel=1e-12)


def test_the_dispersion_term_is_the_unit_cells_own(silicon):
    """Grimme's D2 enters per cell, for the Ewald term's reason exactly.

    It is a pair sum over the nuclei outside ``v_of_rho`` and the nuclei do not
    move, so an ``N``-cell supercell's is ``N`` times the unit cell's and the
    ultracell carries the unit cell's unchanged. **Nothing refuses a van der
    Waals correction here**, so a D2 dataset reaches this assembly, and the
    term was missing from the first draft of it -- which would have been a real
    energy quietly absent rather than a convention.
    """
    system, basis = silicon
    ultracell, density, v = _pieces((2, 1, 1), basis, system.cell, seed=6)
    without = ultracell_energy(
        density, _Potential(v), _Potential(v), band_energy=0.0,
        ultracell=ultracell, cell=system.cell, ewald=0.0,
    )
    with_it = ultracell_energy(
        density, _Potential(v), _Potential(v), band_energy=0.0,
        ultracell=ultracell, cell=system.cell, ewald=0.0, dispersion=-0.25,
    )
    assert "dispersion" not in without
    assert with_it["dispersion"] == pytest.approx(-0.25)
    assert total_of(with_it) == pytest.approx(total_of(without) - 0.25)


def test_deband_pairs_the_potential_it_is_handed(silicon):
    """``deband`` reads ``potential_in`` and the energies read ``potential_out``.

    The two are different objects at every iteration and the same one only at
    exact self-consistency, and pairing ``deband`` with the wrong one is worth
    a factor of 10^7 in the ``N = 1`` null. Here the two potentials are made
    deliberately different and the term is required to follow the first.
    """
    system, basis = silicon
    ultracell, density, v_in = _pieces((2, 1, 1), basis, system.cell, seed=7)
    v_out = v_in * 2.0
    terms = ultracell_energy(
        density, _Potential(v_in), _Potential(v_out, ehart=1.0, etxc=2.0),
        band_energy=0.0, ultracell=ultracell, cell=system.cell, ewald=0.0,
    )
    element = ultracell.volume(system.cell) / density[0].size
    expected = -element * float(jnp.sum(density * v_in)) / ultracell.cells
    assert terms["_deband"] == pytest.approx(expected, rel=1e-12)
    # and the Hartree and exchange-correlation halves came from the other one,
    # divided by the cell count the same way.
    assert terms["hartree"] == pytest.approx(1.0 / ultracell.cells)
    assert terms["xc"] == pytest.approx(2.0 / ultracell.cells)


def test_the_entropy_is_rescaled_to_one_unit_cell(silicon):
    """``demet`` is found for the ultracell and divided by ``N``, like the rest.

    The Fermi search runs with the ``k0`` weights as they are and ``N`` times
    the electrons (``driver._occupy``), so the entropy that comes back belongs
    to the ultracell too. Getting the rescaling backwards is a factor of ``N``
    in a term small enough to hide one, which is why it is asserted against the
    same quantity computed on the unit cell's own states.
    """
    system, basis = silicon

    class _System:
        degauss = 0.02
        smearing = "gaussian"

    eigenvalues = jnp.asarray(np.linspace(-0.5, 0.5, 12).reshape(1, 2, 6))
    weights = jnp.asarray([1.0, 1.0])
    one = Ultracell.build((1, 1, 1), basis.dense.grid)
    four = Ultracell.build((2, 2, 1), basis.dense.grid)
    single = ultracell_entropy(eigenvalues, weights, 0.0, one, _System())
    many = ultracell_entropy(eigenvalues, weights, 0.0, four, _System())
    assert single != 0.0
    assert many == pytest.approx(single / four.cells, rel=1e-12)
