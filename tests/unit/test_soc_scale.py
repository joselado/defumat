"""``soc_scale``: switching spin-orbit coupling off without changing the dataset.

Elk's ``socscf`` (manual 5.118, ``gensocfr.f90``), which exists there for one
reason -- "to enhance the effect of spin-orbit coupling in order to accurately
determine the magnetic anisotropy energy". ``pw.x`` has no counterpart.

A pseudopotential has no additive ``xi L.S`` operator to scale, so what the knob
interpolates along has to be chosen. **It is one rule, the spin trace of the
``fcoef`` sandwich, applied to every spin-independent matrix the sandwich
dresses**, and that is the content of this file:

* ``dvan_so`` and ``qq_so`` are sandwiches of spin-*independent* radial data, so
  everything spin-dependent in them is the coupling and a spin trace removes
  exactly it;
* ``newd_so`` sandwiches the potential's integrals, which carry the exchange
  field, so the trace is taken **one Pauli component at a time**: tracing the
  whole block would switch off the magnet, and collapsing the sandwich to
  ``fcoef = identity`` instead (what this code did until `PLAN.md` P115) gives
  the potential a different dressing from the overlap;
* ``becsum`` takes the transpose of that map, which is what makes the density
  the one whose energy the reduced Hamiltonian is the derivative of.
"""

import numpy as np
import pytest

from defumat.pseudo import read_upf
from defumat.pseudo.spinorbit import SpinOrbitCoupling, spin_trace
from tests.conftest import GENERATED

pytestmark = [pytest.mark.unit]

RELATIVISTIC = "Co.rel-pbe-nd-rrkjus.UPF"
SCALAR = "Co.pbe-nd-rrkjus.UPF"


def _pseudo(name):
    return read_upf(GENERATED.parent / "pseudo" / name)


def test_zero_scale_makes_dvan_spin_diagonal_and_spin_independent():
    """The property that makes an anisotropy impossible, whatever else is true.

    A nonlocal potential that is diagonal in spin *and* the same in both
    channels cannot couple the spin direction to the lattice, so the band
    energy cannot depend on where the moment points. It is asserted rather than
    argued because it is the premise the whole ``soc_scale = 0`` control rests
    on.
    """
    coupling = SpinOrbitCoupling(_pseudo(RELATIVISTIC), 0.0)
    dvan = coupling.dvan_so
    assert np.abs(dvan[:, :, 0, 1]).max() == 0.0
    assert np.abs(dvan[:, :, 1, 0]).max() == 0.0
    np.testing.assert_array_equal(dvan[:, :, 0, 0], dvan[:, :, 1, 1])

    # ... and the full operator is emphatically not like that.
    full = SpinOrbitCoupling(_pseudo(RELATIVISTIC), 1.0).dvan_so
    assert np.abs(full[:, :, 0, 1]).max() > 1.0


def test_the_knob_has_two_ends_and_nothing_in_between():
    pseudo = _pseudo(RELATIVISTIC)
    zero = SpinOrbitCoupling(pseudo, 0.0)
    one = SpinOrbitCoupling(pseudo, 1.0)
    np.testing.assert_allclose(zero.dvan_so, one.dvan_scalar, atol=1e-15)
    with pytest.raises(ValueError, match="only 0 and 1"):
        SpinOrbitCoupling(pseudo, 0.5)
    # ``soc_scale = 1`` must be bit-for-bit the unscaled operator: it is the
    # branch every ordinary run takes, and a rounding-level change there would
    # move validated numbers.
    # ``soc_scale = 1`` takes the branch that skips the blend entirely, so it
    # is bit-for-bit the unscaled operator -- a rounding-level change there
    # would move every validated spin-orbit number in the package.
    assert one.dvan_so is not None
    assert np.abs(one.dvan_so).max() > 0.0


def test_a_scalar_relativistic_species_ignores_the_knob():
    """It has no coupling to scale, so the knob must be a no-op, not a scaling."""
    pseudo = _pseudo(SCALAR)
    assert not pseudo.has_so
    for scale in (0.0, 1.0):
        np.testing.assert_array_equal(
            SpinOrbitCoupling(pseudo, scale).dvan_so,
            SpinOrbitCoupling(pseudo, 1.0).dvan_so,
        )


def test_the_overlap_is_scaled_the_same_way_and_stays_spin_free_at_zero():
    """``qq_so`` has to be scaled too, and by the same rule as ``dvan_so``.

    A fully-relativistic ultrasoft dataset's *overlap* carries the coupling as
    well -- ``transform_qq_so`` gives it off-diagonal spin blocks -- so
    scaling only the potential would leave the metric coupled and
    ``soc_scale = 0`` would not be coupling-free. The scalar end has to be
    ``spin_trace`` and not the undressed ``qq``: the ``fcoef = identity``
    limit *looks* like the scalar-relativistic overlap, and using it gives
    169 meV of anisotropy at ``soc_scale = 0`` where the answer is exactly
    zero.
    """
    pseudo = _pseudo(RELATIVISTIC)
    rng = np.random.default_rng(0)
    values = rng.random((pseudo.nh, pseudo.nh))
    qq = 0.5 * (values + values.T) + pseudo.nh * np.eye(pseudo.nh)

    for scale in (0.0, 1.0):
        blocks = SpinOrbitCoupling(pseudo, scale).qq_so(qq)
        matrix = blocks.transpose(0, 2, 1, 3).reshape(2 * pseudo.nh, 2 * pseudo.nh)
        matrix = 0.5 * (matrix + matrix.conj().T)
        assert np.linalg.eigvalsh(matrix).min() > -1.0e-8, (
            f"the overlap stopped being positive semidefinite at {scale}"
        )

    # At zero it is also spin-diagonal and spin-independent, which is what
    # makes the anisotropy vanish identically there.
    zero = SpinOrbitCoupling(pseudo, 0.0).qq_so(qq)
    assert np.abs(zero[:, :, 0, 1]).max() == 0.0
    np.testing.assert_array_equal(zero[:, :, 0, 0], zero[:, :, 1, 1])


def test_spin_trace_is_idempotent_and_kills_the_traceless_part():
    rng = np.random.default_rng(1)
    blocks = rng.random((4, 4, 2, 2)) + 1j * rng.random((4, 4, 2, 2))
    traced = spin_trace(blocks)
    np.testing.assert_allclose(spin_trace(traced), traced, atol=1e-15)
    np.testing.assert_allclose(traced[..., 0, 1], 0.0, atol=1e-15)
    np.testing.assert_allclose(traced[..., 0, 0], traced[..., 1, 1], atol=1e-15)


def test_the_system_carries_it_without_moving_the_k_points():
    """Two ``soc_scale`` must be sampled identically or the law is unmeasurable."""
    from defumat.system.builder import system_from_file

    system = system_from_file(GENERATED / "co-slab-forcetheorem-par.in")
    assert system.soc_scale == 1.0
    scaled = system.with_soc_scale(0.0)
    assert scaled.soc_scale == 0.0
    assert scaled.kpoints is system.kpoints
    for bad in (-1.0, 0.5, 2.0):
        with pytest.raises(ValueError, match="only 0 and 1"):
            system.with_soc_scale(bad)


# The reduction of the **density** and of ``newd_so`` (`PLAN.md` P115). Before
# it, ``soc_scale = 0`` spin-traced ``dvan_so`` and ``qq_so``, built ``becsum``
# with the full ``fcoef`` sandwich, and collapsed ``newd_so`` to the plain
# recombination: three dressings, so the Hamiltonian was not the derivative of
# the energy it reported. Each identity below failed on that code.


def _spin_density(nh, seed):
    """``B[k, s, l, t] = sum_n <psi_n|beta_k s><beta_l t|psi_n>`` for random states."""
    rng = np.random.default_rng(seed)
    amplitudes = rng.normal(size=(2 * nh, 7)) + 1j * rng.normal(size=(2 * nh, 7))
    return (amplitudes.conj() @ amplitudes.T).reshape(nh, 2, nh, 2)


def _symmetric(nh, seed, count=None):
    rng = np.random.default_rng(seed)
    shape = (nh, nh) if count is None else (count, nh, nh)
    values = rng.normal(size=shape)
    return 0.5 * (values + np.swapaxes(values, -1, -2))


def _becsum(coupling, spin_density, nspin_mag, scale):
    import jax.numpy as jnp

    from defumat.pseudo.spinorbit import becsum_transform

    return np.asarray(becsum_transform(
        jnp.asarray(coupling.fcoef), jnp.asarray(spin_density)[None], nspin_mag,
        soc_scale=scale,
    ))[:, 0]


@pytest.mark.parametrize("scale", [0.0, 1.0])
def test_the_augmentation_charge_is_the_overlap_the_eigenproblem_normalises_by(scale):
    """``sum_ij qq_ij becsum_ij = sum <psi|beta> qq_so <beta|psi>``, at both ends.

    The left side is the charge the density carries; the right is what the
    generalised eigenproblem sets to one per state. At ``soc_scale = 0`` the
    old ``becsum`` kept the full sandwich while ``qq_so`` was spin-traced, and
    the cobalt cell's density integrated to 8.99999859 of 9.
    """
    pseudo = _pseudo(RELATIVISTIC)
    coupling = SpinOrbitCoupling(pseudo, scale)
    qq = _symmetric(pseudo.nh, seed=11)
    spin_density = _spin_density(pseudo.nh, seed=12)

    carried = float(np.sum(qq * _becsum(coupling, spin_density, 1, scale)[0]))
    overlap = np.einsum("klst,kslt->", coupling.qq_so(qq), spin_density)
    assert carried == pytest.approx(float(np.real(overlap)), rel=1e-12)


@pytest.mark.parametrize("scale", [0.0, 1.0])
def test_newd_is_the_derivative_of_the_augmentation_energy(scale):
    """``sum_c int V_c rho_aug,c`` differentiated by the occupations is ``newd_so``.

    ``E = sum_c sum_ij deeq_c,ij becsum_c,ij(B)`` is linear in ``B``, so its
    derivative is the matrix ``D`` with ``E = Re sum D[s,t,k,l] B[k,s,l,t]``, and
    that has to be what :func:`_newd_noncollinear` returns (less ``dvan_so``).
    A Hamiltonian that is not this derivative converges to a state its own
    total is first order in, which is what the reduced cobalt cell did.
    """
    import jax.numpy as jnp

    from defumat.scf.driver import _newd_noncollinear, _spin_block_diagonal

    pseudo = _pseudo(RELATIVISTIC)
    coupling = SpinOrbitCoupling(pseudo, scale)
    deeq = _symmetric(pseudo.nh, seed=21, count=4)
    spin_density = _spin_density(pseudo.nh, seed=22)

    energy = float(np.sum(deeq * _becsum(coupling, spin_density, 4, scale)))
    fcoef = jnp.asarray(_spin_block_diagonal([coupling.fcoef]))
    zero = jnp.zeros_like(fcoef)
    potential = np.asarray(_newd_noncollinear(jnp.asarray(deeq), zero, fcoef, scale))
    paired = float(np.real(np.einsum("stkl,kslt->", potential, spin_density)))
    assert energy == pytest.approx(paired, rel=1e-12)


def _rotated(spin_density, angle, axis):
    """``B`` of the same states with every spinor turned by one SU(2) rotation."""
    sigma = {
        "x": np.array([[0, 1], [1, 0]], dtype=complex),
        "y": np.array([[0, -1j], [1j, 0]]),
    }[axis]
    u = np.cos(angle / 2) * np.eye(2) - 1j * np.sin(angle / 2) * sigma
    return np.einsum("sa,kalb,tb->kslt", u.conj(), spin_density, u)


def test_the_reduced_density_does_not_know_where_the_spin_points():
    """At ``soc_scale = 0`` a global spin rotation leaves the charge and ``|m|`` alone.

    That is the identity behind the directional degeneracy: the charge
    component of ``becsum`` does not move and the three magnetization
    components rotate as a vector. The full sandwich fails it, because
    ``fcoef`` ties the spin to the orbital index, which is asserted too so that
    the check is shown to fire.
    """
    pseudo = _pseudo(RELATIVISTIC)
    spin_density = _spin_density(pseudo.nh, seed=31)
    turned = _rotated(spin_density, 1.1, "y")

    zero = SpinOrbitCoupling(pseudo, 0.0)
    before, after = _becsum(zero, spin_density, 4, 0.0), _becsum(zero, turned, 4, 0.0)
    np.testing.assert_allclose(after[0], before[0], atol=1e-12 * np.abs(before[0]).max())
    np.testing.assert_allclose(
        np.einsum("cij,ckl->ijkl", after[1:], after[1:]),
        np.einsum("cij,ckl->ijkl", before[1:], before[1:]),
        atol=1e-11 * np.abs(before).max() ** 2,
    )

    one = SpinOrbitCoupling(pseudo, 1.0)
    moved = _becsum(one, turned, 4, 1.0)[0] - _becsum(one, spin_density, 4, 1.0)[0]
    assert np.abs(moved).max() > 1e-3 * np.abs(before[0]).max()


def test_the_spin_traced_sandwich_is_the_spin_trace_of_the_overlap():
    """:func:`spin_traced_sandwich` on ``qq`` is :func:`spin_trace` of ``transform_qq_so``.

    One map for the overlap, the bare ``D``, ``newd_so`` and (transposed) the
    density, so this ties the new helper to the rule ``qq_so`` already followed.
    """
    import jax.numpy as jnp

    from defumat.pseudo.spinorbit import spin_traced_sandwich
    from defumat.scf.driver import _spin_block_diagonal

    pseudo = _pseudo(RELATIVISTIC)
    coupling = SpinOrbitCoupling(pseudo, 0.0)
    qq = _symmetric(pseudo.nh, seed=41)
    fcoef = jnp.asarray(_spin_block_diagonal([coupling.fcoef]))
    np.testing.assert_allclose(
        np.asarray(spin_traced_sandwich(fcoef, jnp.asarray(qq, dtype=fcoef.dtype))),
        coupling.qq_so(qq)[:, :, 0, 0], atol=1e-13,
    )


PAW_RELATIVISTIC = "Ni.rel-pbe-spn-kjpaw_psl.1.0.0.UPF"


@pytest.mark.slow
def test_the_paw_sphere_does_not_know_where_the_spin_points_at_zero():
    """The small component's magnetization is coupling, so ``soc_scale = 0`` removes it.

    A fully-relativistic PAW dataset's small component carries magnetization
    along the radial direction, ``-2 (m . r) r`` on the sphere, which ties the
    spin to the lattice. With it left on, a coupling-free run has a direction
    dependence of its own; the check here is that the one-centre energy of a
    non-spherical collinear ``becsum`` does not move when the moment is turned,
    and that at ``soc_scale = 1`` it does, so that the check is shown to fire.

    Slow, and not for the reason a test usually is: one ``nh = 34`` species is
    2.6 GB and 16 s built on its own, and in the gate's process it raised the
    peak to 7.5 GB (2026-09-25), so the two species are built one at a time.
    """
    from defumat.paw.onecenter import _build_species, onecenter_species
    from defumat.pseudo.projectors import projector_channels
    from defumat.xc.functional import resolve_functional

    pseudo = _pseudo(PAW_RELATIVISTIC)
    functional = resolve_functional([pseudo.functional])
    nh = len(projector_channels(pseudo))
    rng = np.random.default_rng(7)
    a = rng.normal(size=(nh, nh)) * 0.15
    charge = a @ a.T + np.diag(rng.uniform(0.2, 1.0, nh))

    def becsum(direction):
        return np.concatenate([charge[None], 0.4 * np.multiply.outer(direction, charge)])

    def turned(paw):
        z, x = (float(onecenter_species(paw, becsum(u), axis=None)[0])
                for u in (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])))
        return x - z

    zero = _build_species(pseudo, functional, 0.0)
    assert zero.nh == nh and zero.density_rel is None
    assert abs(turned(zero)) < 1e-11
    del zero

    one = _build_species(pseudo, functional, 1.0)
    assert one.density_rel is not None
    assert abs(turned(one)) > 1e-6  # 7.5e-6 Ry, 0.10 meV
