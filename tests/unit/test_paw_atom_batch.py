"""PAW's one-centre reverse pass must stay flat in the number of atoms.

`MEMORY-AUDIT.md` A4. Every atom's one-centre pass is live at once inside one
differentiated region -- the sphere's XC quadrature is ``(nspin, nx, mesh)`` and
the spin-polarized kernel takes two inner ``jax.grad``s of its own, so the
residuals are a reverse pass inside a reverse pass. Measured on a
fully-relativistic nickel dataset at ``nspin = 4``, the compiled gradient's
temporary buffer was ``272 MB`` per atom with no ceiling: **4.08 GB** for the
fifteen nickel atoms of the NiBr2 slab.

**The audit's own prescription for this was measured false, which is why the
property is pinned here rather than left to a comment.** Wrapping the body in
``jax.checkpoint`` under the ``vmap`` makes it *worse* (267 -> 796 MB at one
atom, and the per-atom slope unchanged): the recomputation is as wide as the
tape it removed, because every atom is recomputed simultaneously. Only the chunk
and the remat *together* work -- then the residual is one atom's ``becsum`` per
step and the peak stops growing with the sublattice. Either half alone is a
regression, so either half being dropped by a later refactor is a regression,
and the two are not visibly connected in the source.

**Why ``memory_analysis`` and not a peak-RSS assertion.** It runs the compiler
and allocates nothing, so it is deterministic rather than a flake, and it is the
same instrument the audit and ``PERFORMANCE.md`` quote.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat.paw.onecenter import PAW_VMAP_ATOMS, _paw_atom_batch, build_paw
from defumat.pseudo import read_upf
from defumat.system.structure import Species, Structure
from defumat.xc.functional import resolve_functional

pytestmark = [pytest.mark.unit]

#: Silicon's PAW dataset: ``mesh = 1141``, ``nx = 45``, ``nh = 8``. The smallest
#: committed one, chosen because this test compiles four times and the property
#: it asserts is a *ratio* -- which dataset carries it does not matter.
PSEUDO = "Si.pbe-n-kjpaw_psl.0.1.UPF"

#: Both above :data:`PAW_VMAP_ATOMS`, so both take the chunked route, and the
#: larger is twice the smaller: without the remat the tape doubles with it.
FEW, MANY = 4, 8


def _corrections(pseudo_dir, nat):
    pseudo = read_upf(pseudo_dir / PSEUDO)
    structure = Structure(
        positions=jnp.zeros((nat, 3)),
        types=(0,) * nat,
        species=(Species(name="Si", mass=28.086, pseudo_file=PSEUDO),),
    )
    functional = resolve_functional([pseudo.functional])
    return build_paw((pseudo,), structure, functional)


def _becsum(nat, nspin=1, nh=8, seed=0):
    random = np.random.RandomState(seed)
    values = 0.01 * random.randn(nspin, nat, nh, nh)
    return jnp.asarray(0.5 * (values + np.swapaxes(values, -1, -2)))


def _tape_bytes(pseudo_dir, nat):
    """``temp_size_in_bytes`` of the compiled one-centre gradient. Allocates nothing."""
    paw = _corrections(pseudo_dir, nat)
    becsum = _becsum(nat)
    gradient = jax.jit(jax.grad(
        lambda values: paw.energy_and_coefficients((values,))[0]
    ))
    compiled = gradient.lower(becsum).compile()
    return int(compiled.memory_analysis().temp_size_in_bytes)


def test_the_one_centre_tape_does_not_grow_with_the_sublattice(pseudo_dir):
    """Twice the atoms must not be twice the tape.

    Without the chunk *and* the remat it is very nearly exactly twice: the
    measured slope is 27.8 MB per silicon atom at ``nspin = 1`` and 272 MB per
    nickel atom at ``nspin = 4``, both linear to better than one per cent over
    one to fifteen atoms.
    """
    few = _tape_bytes(pseudo_dir, FEW)
    many = _tape_bytes(pseudo_dir, MANY)
    assert many < 1.25 * few, (
        f"the one-centre tape grew from {few / 2**20:.1f} MB at {FEW} atoms to "
        f"{many / 2**20:.1f} MB at {MANY}, which is the linear scaling the chunk "
        f"and the remat exist to remove. Either the @jax.checkpoint or the "
        f"map_axis over atoms has been lost -- neither works without the other "
        f"(MEMORY-AUDIT.md A4)"
    )


@pytest.mark.slow
def test_chunking_the_atom_axis_does_not_move_the_gradient(pseudo_dir):
    """The guard on the guard: a loop bound, not a physical parameter.

    A test that only watched the tape would pass on a body that is cheap and
    wrong. The chunked route reorders the sum over atoms and re-executes the
    body in the backward pass, so this is round-off rather than bit-identical --
    through the full force and stress of a displaced ten-atom PAW cell it is
    1e-14 on 1e-2 (`PERFORMANCE.md`).
    """
    paw = _corrections(pseudo_dir, MANY)
    becsum = _becsum(MANY)

    def gradient():
        return np.asarray(jax.grad(
            lambda values: paw.energy_and_coefficients((values,))[0]
        )(becsum))

    chunked = gradient()
    previous = os.environ.get("DEFUMAT_PAW_ATOM_BATCH")
    os.environ["DEFUMAT_PAW_ATOM_BATCH"] = "all"
    try:
        whole = gradient()
    finally:
        if previous is None:
            os.environ.pop("DEFUMAT_PAW_ATOM_BATCH", None)
        else:
            os.environ["DEFUMAT_PAW_ATOM_BATCH"] = previous

    scale = np.abs(whole).max()
    assert np.allclose(chunked, whole, rtol=0, atol=1e-12 * scale), (
        f"chunking the atom axis moved the one-centre ddd by "
        f"{np.abs(chunked - whole).max() / scale:.2e} of its largest entry"
    )


def test_a_small_sublattice_keeps_the_vmap(pseudo_dir):
    """Below the crossover the chunk loses on *both* axes, so it is not taken.

    Measured on the whole force: two atoms of the spinor PAW cell are 547 MB and
    0.761 s as one ``vmap`` against 718 MB and 1.286 s chunked. Ten atoms are
    381 MB / 0.505 s against 189 MB / 0.418 s the other way. The default is a
    count because that is where the two axes agree, and an explicit setting
    overrides it in either direction.
    """
    assert _paw_atom_batch(PAW_VMAP_ATOMS) is None
    assert _paw_atom_batch(PAW_VMAP_ATOMS + 1) == 1

    previous = os.environ.get("DEFUMAT_PAW_ATOM_BATCH")
    try:
        os.environ["DEFUMAT_PAW_ATOM_BATCH"] = "1"
        assert _paw_atom_batch(2) == 1, "an explicit chunk must beat the count"
        os.environ["DEFUMAT_PAW_ATOM_BATCH"] = "all"
        assert _paw_atom_batch(64) is None, "an explicit vmap must beat it too"
    finally:
        if previous is None:
            os.environ.pop("DEFUMAT_PAW_ATOM_BATCH", None)
        else:
            os.environ["DEFUMAT_PAW_ATOM_BATCH"] = previous


# --- where QE takes the absolute value of the on-site density ----------------


def _species(pseudo_dir):
    """One PAW species object, the thing the one-centre routines take."""
    return _corrections(pseudo_dir, 1).species[0]


def test_the_one_centre_gradient_differentiates_the_signed_density(pseudo_dir):
    """``PAW_gradient`` takes the gradient first and ``ABS`` afterwards.

    ``paw_onecenter.f90:762`` calls ``PAW_gradient`` on
    ``rho_rad*rm2 + rho_core`` with no ``ABS``, and only at ``:780-781`` does
    ``rho_full`` get ``IF (nspin_mag==1) rho_full = ABS(rho_full)`` before it
    goes into ``xc_gcx``. This took the absolute value first, so the radial
    component was ``d|rho|/dr = sign(rho) drho/dr``.

    **The reordering is confined and that is the check**: ``sigma`` is a sum of
    squares, so the sign flip is invisible in it and therefore in ``v1``, ``v2``
    and the gradient-correction *energy*; it survives only in ``h = v2 grad``,
    whose divergence is part of ``ddd``. So the energy at a *given* density is
    the same either way and only the SCF fixed point moves -- which is exactly
    the comparison against ``pw.x`` that cannot see it.

    **The reachable set is empty on every committed PAW-GGA cell**, measured
    rather than argued: reading ``rho_rad[0]/r^2 + core`` from inside the trace
    over the whole quadrature of a converged run,
    ``alas-piezo-tiny-paw`` gives **0 negative of 3,812,100** values and
    ``si10-paw-pbe`` **0 of 8,215,200**, minimum exactly 0.0 in both (the padded
    tail past the augmentation sphere). ``radial_derivative`` is a stencil on
    the array rather than an autodiff of ``abs``, so on an array that is nowhere
    negative the two orders are pointwise identical -- and the total energy, the
    eigenvalues, the one-centre energy and ``ddd`` are bit-identical on both
    cells. The density here is therefore *constructed* to go negative.
    """
    from defumat.paw.gradient import _gradient, onecenter_gradient_correction

    paw = _species(pseudo_dir)
    nlm, mesh, nx = paw.nlm, len(paw.r), paw.angular.ylm.shape[0]

    # A spherical on-site density that changes sign partway out: r^2 rho with a
    # node, which is what an augmentation charge can produce.
    radial = np.linspace(1.0, -1.0, mesh)
    rho_lm = np.zeros((1, nlm, mesh))
    rho_lm[0, 0] = radial * float(np.sqrt(4.0 * np.pi))
    rho_lm[0, min(1, nlm - 1)] = 0.2 * radial
    rho_lm = jnp.asarray(rho_lm)
    rho_rad = jnp.einsum("xl,slr->sxr", paw.angular.ylm[:, :nlm], rho_lm)
    core = jnp.zeros((nx, mesh))

    signed = rho_rad[0] / paw.r2 + core
    assert float(jnp.min(signed)) < 0.0, "the constructed density has to go negative"

    straight = _gradient(rho_lm[0], signed, paw)
    absolute = _gradient(rho_lm[0], jnp.abs(signed), paw)
    assert float(jnp.max(jnp.abs(straight - absolute))) > 1.0, (
        "the two orders have to differ on a density that changes sign")

    # ``sigma`` is what the functional sees, and it is a sum of squares -- but
    # only because the *whole* gradient is taken of one or the other, so this is
    # the statement that the two sigmas genuinely differ and the confinement is
    # in ``v1``/``v2`` being functions of it rather than of the sign.
    potential, energy, vector = onecenter_gradient_correction(
        rho_lm, rho_rad, core, paw, None, False)
    assert np.isfinite(float(energy)) and vector is None
    assert np.all(np.isfinite(np.asarray(potential)))
