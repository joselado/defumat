"""``soc_scale``: switching spin-orbit coupling off without changing the dataset.

Elk's ``socscf`` (manual 5.118, ``gensocfr.f90``), which exists there for one
reason -- "to enhance the effect of spin-orbit coupling in order to accurately
determine the magnetic anisotropy energy". ``pw.x`` has no counterpart.

A pseudopotential has no additive ``xi L.S`` operator to scale, so what the knob
interpolates along has to be chosen, and **the choice is different for the two
kinds of object it has to act on**. That is the whole content of this file:

* ``dvan_so`` and ``qq_so`` are built from spin-*independent* radial data, so
  everything spin-dependent in ``dvan_so`` is the coupling and a spin trace
  removes exactly it;
* ``deeq`` and ``qq_so`` are ``fcoef`` **sandwiches**, and for those the
  coupling is the dressing -- their scalar limit is ``fcoef = identity``. For
  ``deeq`` a spin trace would remove the *exchange field* instead, switching
  off the magnet rather than the coupling; for ``qq_so`` it would leave a
  metric that is not the overlap of anything.
"""

import numpy as np
import pytest

from pypresso.pseudo import read_upf
from pypresso.pseudo.spinorbit import SpinOrbitCoupling, spin_trace
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
    from pypresso.system.builder import system_from_file

    system = system_from_file(GENERATED / "co-slab-forcetheorem-par.in")
    assert system.soc_scale == 1.0
    scaled = system.with_soc_scale(0.0)
    assert scaled.soc_scale == 0.0
    assert scaled.kpoints is system.kpoints
    for bad in (-1.0, 0.5, 2.0):
        with pytest.raises(ValueError, match="only 0 and 1"):
            system.with_soc_scale(bad)
