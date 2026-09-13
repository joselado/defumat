"""The becsum and ``ns`` group averages must be projectors.

A group average ``P = (1/N) sum_S M_S`` is idempotent if and only if ``{M_S}`` is
a representation of the group -- ``M_S M_T = M_{ST}``. That is exactly the
property destroyed by pairing the harmonic rotation of ``S`` with the wrong atom
permutation, and it is a property of the *result* rather than of an index order,
so the check needs no opinion about whose convention (QE's ``d_matrix`` or this
project's) is being followed.

Both symmetrisers contract the rotation on the target harmonic index where QE's
``PAW_symmetrize``/``new_ns`` contract it on the source one. The two are the same
sum with ``S`` relabelled ``S^-1``, so the atom has to be relabelled with it --
and ``irt(S^-1, a)`` is the inverse permutation, not ``irt(S, a)``.

**The cell matters.** Every cell validated against QE so far has an atom orbit on
which every operation's permutation is its own inverse, where the two orderings
coincide and nothing can tell them apart. Three atoms on the cube's face centres
have a three-fold axis through ``(111)`` that cycles them, and 16 of the 48
operations then permute them non-involutively -- which is the whole point of this
geometry and why it is not one of the physics benchmarks.
"""

import pathlib

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.io.pwin import parse_pw_input
from defumat.pseudo import read_upf
from defumat.scf.driver import Calculation
from defumat.system.builder import build_system

pytestmark = pytest.mark.unit

#: Three atoms on the faces of a simple cubic cell: one orbit, cycled by the
#: three-fold axes, so the forward and the inverse atom permutation differ.
_THREE_FOLD_ORBIT = """
&control
  calculation = 'scf'
/
&system
  ibrav = 1, celldm(1) = 10.0, nat = 3, ntyp = 1,
  ecutwfc = {ecutwfc}, ecutrho = {ecutrho}{extra}
/
&electrons
/
ATOMIC_SPECIES
 {name} {mass} {pseudo}
ATOMIC_POSITIONS crystal
 {name} 0.50 0.00 0.00
 {name} 0.00 0.50 0.00
 {name} 0.00 0.00 0.50
K_POINTS automatic
 2 2 2 0 0 0
{cards}"""


def _calculation(pseudo_dir, **kwargs):
    text = _THREE_FOLD_ORBIT.format(**kwargs)
    system = build_system(parse_pw_input(text))
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    return Calculation(system, pseudos)


def _non_involutive(permutations: np.ndarray) -> int:
    """How many operations permute this orbit's atoms non-involutively."""
    identity = np.arange(permutations.shape[1])
    return sum(
        not np.array_equal(p[p], identity) for p in np.asarray(permutations)
    )


def _symmetric(rng, shape) -> jnp.ndarray:
    """A random array symmetric in its last two indices, as an occupation is."""
    x = rng.standard_normal(shape)
    return jnp.asarray(x + np.swapaxes(x, -1, -2))


def test_becsum_symmetrisation_is_a_projector(pseudo_dir):
    """``PAW_symmetrize``'s average, applied twice, must not move."""
    calculation = _calculation(
        pseudo_dir, ecutwfc=12.0, ecutrho=96.0, extra="",
        name="Si", mass=28.086, pseudo="Si.pz-n-kjpaw_psl.0.1.UPF", cards="",
    )
    symmetry = calculation._becsum_symmetry
    assert symmetry is not None
    assert _non_involutive(symmetry.mapping[0]) > 0, "this cell proves nothing"

    rng = np.random.default_rng(0)
    natoms = np.asarray(symmetry.mapping[0]).shape[1]
    becsum = tuple(
        None if operator is None
        else _symmetric(rng, (1, natoms, operator.shape[1], operator.shape[2]))
        for operator in symmetry.operators
    )

    once = symmetry.apply(becsum)
    twice = symmetry.apply(once)
    for a, b in zip(once, twice):
        if a is not None:
            assert np.allclose(a, b, atol=1e-12), "not idempotent"


def test_ns_symmetrisation_is_a_projector(pseudo_dir):
    """``new_ns``'s average, applied twice, must not move."""
    calculation = _calculation(
        pseudo_dir, ecutwfc=25.0, ecutrho=200.0,
        extra=(", nspin = 2, starting_magnetization(1) = 0.3,\n"
               "  occupations = 'smearing', smearing = 'gaussian', degauss = 0.02"),
        name="Ni", mass=58.69, pseudo="Ni.pz-nd-rrkjus.UPF",
        cards="HUBBARD (atomic)\n U Ni-3d 4.0\n",
    )
    symmetry = calculation.hubbard_symmetry
    assert symmetry is not None
    assert _non_involutive(symmetry.sources[0]) > 0, "this cell proves nothing"

    natoms, ldim = symmetry.shape
    ns = _symmetric(np.random.default_rng(0), (2, natoms, ldim, ldim))

    once = symmetry.apply(ns)
    assert np.allclose(once, symmetry.apply(once), atol=1e-12), "not idempotent"


def test_the_becsum_operator_is_the_factor_and_not_its_outer_product(pseudo_dir):
    """The pair operator is never built, and contracting twice is the same sum.

    ``becsum`` is indexed by two projector channels, so the object the group
    average applies is ``D_ik D_jl``. That was stored, as
    ``(nsym, nh, nh, nh, nh)`` -- fine at the ``nh`` of a silicon dataset and
    **489 MB** at the ``nh = 34`` of a fully-relativistic platinum one, against
    444 kB for the ``(nsym, nh, nh)`` factor it is an outer product of. It cost
    twice, because a compiled force or stress gradient closes over it and an
    array reached from a closure is a *constant* embedded in the executable
    (``PERFORMANCE.md``, "A gigabyte of constants").

    This asserts both halves: that what is stored is the factor, and that
    applying it twice reproduces the contraction with the product it replaced,
    which is the algebra the six call sites rest on.
    """
    calculation = _calculation(
        pseudo_dir, ecutwfc=12.0, ecutrho=96.0, extra="",
        name="Si", mass=28.086, pseudo="Si.pz-n-kjpaw_psl.0.1.UPF", cards="",
    )
    symmetry = calculation._becsum_symmetry
    assert symmetry is not None

    operator = next(o for o in symmetry.operators if o is not None)
    assert operator.ndim == 3, (
        "the stored operator must be the (nsym, nh, nh) factor: storing its "
        "outer product is 489 MB on a relativistic platinum dataset"
    )
    assert operator.shape[0] == symmetry.nsym

    rng = np.random.default_rng(1)
    natoms = np.asarray(symmetry.mapping[0]).shape[1]
    nh = operator.shape[1]
    becsum = tuple(
        None if o is None else _symmetric(rng, (1, natoms, nh, nh))
        for o in symmetry.operators
    )
    got = symmetry.apply(becsum)

    # The expression this replaced, written out with the product materialised.
    pair = np.einsum("sik,sjl->sijkl", np.asarray(operator), np.asarray(operator))
    values = np.asarray(becsum[0])
    gathered = values[:, np.asarray(symmetry.mapping[0])]
    want = np.einsum("sijkl,zsnkl->znij", pair, gathered) / symmetry.nsym
    assert np.allclose(np.asarray(got[0]), want, rtol=0, atol=1e-13)
