"""An unpolarized run is projected once for the site density matrix, not twice.

``OPEN.md`` Part XXIII item 17. At ``nspin = 1`` the site density matrix has two
diagonal spin blocks and both come from the one channel there is, each with half
of ``wg``. The two blocks were filled by projecting that channel twice, the same
``(nk, nbnd, natomwfc)`` product computed a second time; the second component now
reuses the first. Checked on stand-in arrays rather than on a converged run: the
projection count is the claim, and the matrix against the two-projection form is
the proof that reusing it changed no bit.
"""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat.projwfc.angular_momentum import _site_density_matrix

pytestmark = pytest.mark.unit

NK, NBND, NPWX, NATOMWFC = 3, 4, 11, 5


def _inputs(host: bool):
    rng = np.random.default_rng(1)
    shape = (1, NK, NBND, NPWX)
    psi = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    projectors = (rng.normal(size=(NK, NPWX, NATOMWFC))
                  + 1j * rng.normal(size=(NK, NPWX, NATOMWFC)))
    weights = rng.uniform(size=(NK, NBND))
    result = types.SimpleNamespace(
        wavefunctions=psi if host else jnp.asarray(psi), occupations=weights)
    return types.SimpleNamespace(nspin=1), result, projectors


@pytest.mark.parametrize("host", [False, True], ids=["device", "host-store"])
def test_an_unpolarized_channel_is_projected_once(host, monkeypatch):
    calculation, result, projectors = _inputs(host)
    calls = []
    real = jax.numpy.einsum

    def counting(subscripts, *operands, **kwargs):
        if subscripts in ("kgi,kbg->kbi", "gi,bg->bi"):
            calls.append(subscripts)
        return real(subscripts, *operands, **kwargs)

    monkeypatch.setattr(jax.numpy, "einsum", counting)
    density = _site_density_matrix(calculation, result, projectors, channels=None)
    monkeypatch.undo()

    # One pass over the k-points: one call on the device, one per k-point when
    # the store streams. Twice that is the channel projected once per spin
    # component.
    assert len(calls) == (NK if host else 1), calls

    # The projection by the same route the function takes, done here once.
    psi = np.asarray(result.wavefunctions)[0]
    if host:
        coefficients = np.stack([
            np.asarray(jnp.einsum("gi,bg->bi", projectors[ik].conj(),
                                  jnp.asarray(psi[ik])))
            for ik in range(NK)])
    else:
        coefficients = np.asarray(
            jnp.einsum("kgi,kbg->kbi", projectors.conj(), jnp.asarray(psi)))
    block = np.einsum("kb,kbi,kbj->ij", 0.5 * result.occupations,
                      coefficients, coefficients.conj())
    # Exactly the block either projection gave, in both diagonal positions, and
    # nothing between them: two collinear components have no coherence.
    assert np.array_equal(density[:, 0, :, 0], block)
    assert np.array_equal(density[:, 1, :, 1], block)
    assert not density[:, 0, :, 1].any() and not density[:, 1, :, 0].any()
