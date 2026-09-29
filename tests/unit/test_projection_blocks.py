"""The atomic projections built a block of k-points at a time are the whole build.

In memory mode :func:`~defumat.projwfc.projections.atomic_projections` builds
the atomic projector set on each block's row-subset calculation instead of for
every k-point at once, which on eight-atom silicon at 216 k-points took the
card from 645.6 MB to the SCF's own peak. Each k-point's set is its own
arithmetic, so the blocks must reproduce the one-shot build to round-off --
with a short, padded last block, and for a device array and a host store
alike. No SCF is needed: the projection of any vector set is a fixed linear
map, so random states on each k-point's sphere test it as well as converged
ones would.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from defumat import Calculator
from defumat.projwfc import projections
from defumat.projwfc.projections import atomic_projections

pytestmark = pytest.mark.unit


_ULTRASOFT_SILICON = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.20, nat = 2, ntyp = 1,
  ecutwfc = 16.0, ecutrho = 128.0, nosym = .true., noinv = .true.
/
&electrons
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-n-rrkjus_psl.0.1.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""

_SPIN_ORBIT_PLATINUM = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 7.42, nat = 1, ntyp = 1,
  ecutwfc = 16.0, ecutrho = 128.0, nosym = .true., noinv = .true.,
  noncolin = .true., lspinorb = .true.,
  occupations = 'smearing', smearing = 'mv', degauss = 0.02
/
&electrons
/
ATOMIC_SPECIES
 Pt 195.08 Pt.rel-pz-n-rrkjus.UPF
ATOMIC_POSITIONS alat
 Pt 0.00 0.00 0.00
K_POINTS automatic
 2 2 2 0 0 0
"""


def _random_states(calculation, nbnd=6, seed=0):
    """``(1, nk, nbnd, npol npwx)`` random vectors on each k-point's sphere."""
    mask = np.asarray(calculation.basis.planewaves.mask, dtype=bool)
    npol = 2 if calculation.system.noncolin else 1
    mask = np.concatenate([mask] * npol, axis=-1)
    rng = np.random.default_rng(seed)
    shape = (1,) + mask.shape[:1] + (nbnd,) + mask.shape[1:]
    psi = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return psi * mask[None, :, None, :]


@pytest.mark.parametrize("text", [
    pytest.param(_ULTRASOFT_SILICON, id="ultrasoft-silicon"),
    # Six seconds, most of it the relativistic dataset's setup.
    pytest.param(_SPIN_ORBIT_PLATINUM, id="spin-orbit-platinum",
                 marks=pytest.mark.slow),
])
def test_blocks_reproduce_the_whole_build(text, pseudo_dir, monkeypatch):
    calculation = Calculator.from_text(text, pseudo_dir, announce=False,
                                       memory_mode="memory").calculation
    assert calculation.memory_mode == "memory"
    psi = _random_states(calculation)
    nk = psi.shape[1]

    monkeypatch.setattr(projections, "PROJECTOR_BLOCK_BYTES", 2**40)
    whole = atomic_projections(calculation, psi, symmetrize=False)
    per_k = psi.shape[-1] * whole.shape[2] * 16
    scale = np.max(np.abs(whole))
    assert scale > 0
    # Three k-points a block on eight: the last block is padded with repeats.
    monkeypatch.setattr(projections, "PROJECTOR_BLOCK_BYTES", 3 * per_k)
    assert nk % 3
    for states in (jnp.asarray(psi), psi):
        blocks = atomic_projections(calculation, states, symmetrize=False)
        assert blocks.shape == whole.shape
        np.testing.assert_allclose(blocks, whole, rtol=0, atol=1e-12 * scale)
