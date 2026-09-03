"""The transverse Kohn-Sham spin susceptibility ``X_0^{+-}(q, omega)_{GG'}``.

``PLAN.md`` P63, and the collinear corner of Elk's ``genspchi0.f90``. This is
the object whose pole, once the exchange-correlation kernel has enhanced it, is
the **magnon**.

**For a collinear ground state the 4x4 spin-density response block-diagonalises
and the transverse block decouples**, which is what makes this a plain matrix
in ``(G, G')`` rather than the ``(4 ngrf)^2`` object Elk carries for the general
case. ``tfm2213`` says so: with the local axis along ``z``, every charge-spin
and off-diagonal spin-spin entry of its kernel vanishes and the ``xx`` and
``yy`` entries are equal. Three consequences, each of which *removes* machinery
that :mod:`defumat.tddft.chi0` needs:

* **No Coulomb.** The Hartree term belongs to the charge channel alone -- Elk
  adds the regularised interaction to the ``(1,1)`` element only -- so nothing
  is symmetrised with ``sqrt(v)``, there is no ``1/q^2``, and ``G = 0`` is an
  ordinary entry of the matrix rather than a direction-resolved 3x3 head.
* **No velocity operator.** The optical limit is what needed ``dH/dk``; a
  transverse susceptibility is finite and nonzero at ``q = 0``, that being the
  Goldstone mode.
* **No new k-set.** ``q`` is restricted to a difference of two k-points of the
  grid, so ``k + q`` is already in it and the down states the sum needs are the
  ones already computed, read through a permutation. What is left of the
  ``k + q`` problem is an **umklapp shift** of the gather index, which is the
  zone-edge shift ``u_{k+b}(G) = u_k(G+b)`` that :mod:`defumat.topology` already
  lives on.

**What is stored is ``X_0 = delta m^- / delta B^-``**, the response of the
magnetization to a transverse field in the pairing ``(v, B)`` conjugate to
``(n, m)`` -- so ``E`` contains ``int (v n + B m)`` and ``v_up = v + B``. In
terms of the bare Adler-Wiser sum over the off-diagonal block of the density
matrix that is a **factor of two**, because ``rho_{up,dn} = m^-/2`` and
``V_{up,dn} = B^-``. Getting it wrong moves the magnon by a factor of two and
nothing about the spectrum looks wrong; the Goldstone identity below is what
catches it.

    X_0(G, G', omega) = (2/Omega) sum_k sum_{nm}
        (f_{nk,up} - f_{m k+q,dn}) conj(M_G) M_G'
        / (omega + eps_{nk,up} - eps_{m k+q,dn} + i eta)

    M_G = <psi_{nk,up}| e^{-i(q+G).r} |psi_{m k+q,dn}>

**Both members of every band pair are summed, and only one ordering.** Unlike
the charge channel, ``+-`` is not symmetric under exchanging the pair: the
reversed pair belongs to ``X^{-+}``, which is a different function with its pole
at the opposite sign of ``omega``. So the pole of a given majority direction is
found at one sign of ``omega`` only, and looking at the other and finding
nothing is not a bug (:func:`transverse_response` takes ``flip`` for that).

**The pairs are all ``nbnd x nbnd``, not occupied times empty**, which is also
why the metal refusal of the charge channel does not transfer. There the
argument is that ``f_i - f_j`` kills the ``i = j`` term and the intraband
response is lost; here the two members of a pair are in *different spin
channels*, ``f_{n,up} - f_{n,dn}`` does not vanish, and that term is the Stoner
continuum. A ferromagnetic metal is the point of this feature, not an
exclusion.

**The degenerate pair is a ``0/0`` and is not left to the broadening.** Where
``eps_{n,up} = eps_{m,dn}`` the weight is the derivative of the occupation with
respect to energy, which is the smearing function itself -- zero to machine
precision for a gapped system and the intraband weight for a metal. Relying on
a finite ``eta`` to hide the division silently drops that term from the
spectrum instead of computing it.

**Cost.** One transform per band pair per k-point, and the matrix is
``nw nm^2`` complex. With the pair axis walked one *up* band at a time the
working set is ``nbnd`` grid-sized fields rather than ``nbnd^2`` of them, which
is what makes an eighteen-band transition metal fit.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from defumat.basis.fft import g_to_r, r_to_g
from defumat.batching import resolve_k_batch, sum_k
from defumat.scf.occupations import smearing_order, w0gauss

__all__ = [
    "SpinSphere",
    "SpinChiZero",
    "spin_response_sphere",
    "transverse_response",
    "require_a_transverse_regime",
    "commensurate_shift",
    "occupation_slope",
]

#: How close two crystal coordinates must be to be the same k-point.
_GRID_TOLERANCE = 1.0e-6


class SpinSphere(eqx.Module):
    """The reciprocal lattice vectors the transverse response is a matrix over.

    ``G = 0`` **is** in this set, which is the whole difference from
    :class:`~defumat.tddft.chi0.ResponseSphere`: there is no Coulomb
    interaction in this channel, so the ``G = 0`` entry is finite and is in fact
    the one the magnon is read off.
    """

    #: ``(nm,)`` flat indices into the *smooth* FFT box.
    fft_index: jnp.ndarray
    #: ``(nm, 3)`` signed Miller indices. Carried rather than recovered, because
    #: the kernel is assembled on the **dense** box from differences of these.
    miller: jnp.ndarray
    ecut: float = eqx.field(static=True)

    @property
    def nm(self) -> int:
        return self.fft_index.shape[0]


def spin_response_sphere(calculation, ecut: float) -> SpinSphere:
    """The G-vectors inside ``ecut`` (Ry), ``G = 0`` first."""
    gvectors = calculation.basis.smooth
    g2 = np.asarray(gvectors.kinetic(calculation.system.cell))
    if ecut < 0.0:
        raise ValueError(
            f"the response cutoff cannot be negative, got {ecut}: it selects "
            "the G-vectors the transverse susceptibility is a matrix over "
            "(Elk's gmaxrf), and zero means G = 0 alone -- the local-field-free "
            "magnon"
        )
    inside = np.flatnonzero(g2 <= ecut)
    return SpinSphere(
        fft_index=jnp.asarray(np.asarray(gvectors.fft_index)[inside]),
        miller=jnp.asarray(np.asarray(gvectors.miller)[inside]),
        ecut=float(ecut),
    )


class SpinChiZero(eqx.Module):
    """``X_0^{+-}`` at every requested frequency, and what built it."""

    #: ``(nw, nm, nm)`` complex.
    x: jnp.ndarray
    #: ``(nw,)`` complex: ``omega + i eta``, Ry.
    frequencies: jnp.ndarray
    sphere: SpinSphere
    #: ``(3,)`` the wavevector in crystal coordinates, as asked for.
    q: tuple = eqx.field(static=True)
    nbnd: int = eqx.field(static=True)
    #: Whether the two spin channels were exchanged, which turns ``X^{+-}``
    #: into ``X^{-+}`` and moves the pole to the other sign of ``omega``.
    flip: bool = eqx.field(static=True, default=False)

    @property
    def nm(self) -> int:
        return self.x.shape[-1]


def occupation_slope(calculation, eigenvalues, levels) -> jnp.ndarray:
    """``w_k df/deps`` at every eigenvalue, the weight a degenerate pair takes.

    The occupation is ``w_k wgauss((ef - eps)/degauss)``, so the derivative is
    ``-w_k w0gauss(...)/degauss`` -- QE's smeared delta function, obtained from
    ``wgauss`` by differentiation rather than transcribed (``w0gauss`` here is
    a ``jvp`` already). The k-point weight is carried because the occupations
    this is compared against carry it too.

    Zero for a fixed-occupation or tetrahedron run: a step function has no
    derivative to speak of and its degenerate pairs contribute nothing but at
    ``omega = 0``, where the transverse response of a gapped system has no
    intraband weight in any case.
    """
    eigenvalues = jnp.asarray(eigenvalues)
    scheme = str(calculation.system.occupations)
    if not scheme.startswith(("smearing", "gaussian", "mp", "mv", "fd")):
        return jnp.zeros_like(eigenvalues)
    degauss = float(calculation.system.degauss)
    ngauss = smearing_order(calculation.system.smearing)
    if "fermi_energy_up" in levels:
        fermi = jnp.asarray(
            [levels["fermi_energy_up"], levels["fermi_energy_down"]]
        )[:, None, None]
    else:
        fermi = jnp.asarray(levels["fermi_energy"])
    weights = jnp.asarray(calculation.system.kpoints.weights)[None, :, None]
    return -weights * w0gauss((fermi - eigenvalues) / degauss, ngauss) / degauss


def require_a_transverse_regime(calculation) -> None:
    """What the transverse susceptibility refuses, on the calculation alone.

    ``nspin != 2``
        the ground state has to be collinear and polarized for the transverse
        block to decouple. A noncollinear one needs Elk's full 4x4
        ``genspchi0``, which is a different object rather than a spin axis on
        this one; an unpolarized one has ``m = 0`` and no kernel at all.
    ultrasoft and PAW
        ``<u_i|e^{-i(q+G).r}|u_j>`` is not the plane-wave overlap when the
        charge is not all in ``|psi|^2``. This is the same missing
        ``Q_ij(q+G)`` that ``PLAN.md`` P40 measured and did not close in the
        charge channel.
    a spin spiral
        the two spinor components live on spheres centred at ``k +- q/2``, and
        the generalized Bloch theorem's rotating frame is not the frame this
        kernel was derived in.
    a Hubbard ``U``
        the Hubbard term responds to the magnetization too, so leaving it out
        of the enhancement is a silent approximation.
    an applied magnetic field
        the rotation argument the kernel rests on assumes the energy is
        invariant under a global spin rotation, and an applied field is exactly
        what breaks that -- a magnon in a field is gapped by the Zeeman energy,
        which this construction would not produce.
    a potential-only meta-GGA
        there is no ``E_xc`` to be rotationally invariant, and ``v_of_rho``
        would want ``tau``.
    a reduced k-set
        ``X_0(G, G')`` is a matrix in two G indices and symmetrising it rotates
        both at once, which nothing here implements -- and the ``k + q`` fold
        needs the grid to be closed anyway. Run with ``nosym`` and ``noinv``.
    """
    system = calculation.system
    if calculation.is_ultrasoft or calculation.is_paw:
        raise NotImplementedError(
            "a transverse spin susceptibility with an ultrasoft or PAW "
            "pseudopotential is not implemented: every matrix element gains "
            "the augmentation charge Q_ij(q+G), and without it the magnon is "
            "wrong by the whole augmentation and still looks like a magnon. "
            "Use a norm-conserving dataset"
        )
    if system.noncolin:
        raise NotImplementedError(
            "a transverse spin susceptibility of a noncollinear ground state "
            "is not implemented: the 4x4 spin-density response no longer "
            "block-diagonalises, so the transverse channel is not a matrix in "
            "(G, G') on its own (Elk's genspchi0 carries all sixteen blocks)"
        )
    if calculation.nspin != 2:
        raise NotImplementedError(
            f"the transverse spin susceptibility needs nspin = 2 and this run "
            f"has nspin = {calculation.nspin}: without a collinear "
            "magnetization there is no transverse channel to separate and the "
            "kernel B_xc/m does not exist"
        )
    if calculation.spiral:
        raise NotImplementedError(
            "a transverse spin susceptibility on a spin spiral is not "
            "implemented: the spinor components live on spheres centred at "
            "k +- q/2, so a plane-wave matrix element is not a single gather"
        )
    if getattr(calculation, "is_hubbard", False):
        raise NotImplementedError(
            "a transverse spin susceptibility with a Hubbard U is not "
            "implemented: the Hubbard term responds to the magnetization as "
            "well, so omitting it from the enhancement is a silent "
            "approximation rather than a missing feature"
        )
    if getattr(calculation, "magnetic_field", None) is not None:
        raise NotImplementedError(
            "a transverse spin susceptibility with an applied magnetic field "
            "or a constrained moment is not implemented: the kernel B_xc/m "
            "comes from the energy being invariant under a global spin "
            "rotation, and an applied field breaks exactly that -- the magnon "
            "acquires a Zeeman gap this construction does not carry"
        )
    if calculation.functional.is_meta:
        raise NotImplementedError(
            "a transverse spin susceptibility with a potential-only meta-GGA "
            "is not implemented: there is no exchange-correlation energy for "
            "the rotation argument to be about"
        )
    weights = np.asarray(calculation.system.kpoints.weights)
    if weights.size > 1 and np.ptp(weights) > 1e-8 * np.abs(weights).max():
        raise NotImplementedError(
            "the transverse spin susceptibility needs the full k-grid, not a "
            "symmetry-reduced wedge: X_0(G, G') is a matrix in two G indices, "
            "and the k + q fold needs the set to be closed under translation "
            "by q. Run with nosym = .true. and noinv = .true."
        )


def commensurate_shift(calculation, q):
    """``(index, umklapp)``: where ``k + q`` sits in the same k-set.

    Returns ``index`` of shape ``(nk,)`` -- the position of the grid point
    ``k + q`` lands on -- and ``umklapp`` of shape ``(nk, 3)``, the integer
    ``G0 = k + q - k'`` that brings it back. ``q`` is in **crystal**
    coordinates, as ``spiral_q`` is.

    A k-grid is closed under translation by any difference of two of its
    points, so the wavevectors this admits are exactly the ones the grid can
    represent -- Elk's requirement that ``vecql`` be commensurate with the
    mesh, reached here by construction rather than by a check on ``ngridq``.
    """
    cell = calculation.system.cell
    points = np.asarray(calculation.system.kpoints.crystal(cell), dtype=float)
    q = np.asarray(q, dtype=float).reshape(3)

    # A dictionary on the rounded fractional part. Rounding to 1e-6 is what
    # makes two points built by different routes -- one from the grid, one from
    # a sum -- compare equal.
    def key(point):
        folded = point - np.floor(point + 0.5 + _GRID_TOLERANCE)
        return tuple(np.round(folded / _GRID_TOLERANCE).astype(np.int64))

    where = {key(point): n for n, point in enumerate(points)}
    index = np.empty(len(points), dtype=np.int32)
    umklapp = np.empty((len(points), 3), dtype=np.int64)
    for n, point in enumerate(points):
        target = point + q
        found = where.get(key(target))
        if found is None:
            raise NotImplementedError(
                f"q = {tuple(q)} in crystal coordinates does not map the "
                f"k-set onto itself: k = {tuple(point)} goes to "
                f"{tuple(target)}, which is not a k-point of this run. The "
                "transverse susceptibility needs the states at k + q, and it "
                "gets them from the same grid rather than from a second "
                "diagonalisation -- so q must be a difference of two k-points"
            )
        index[n] = found
        umklapp[n] = np.rint(target - points[found]).astype(np.int64)
        if np.max(np.abs(target - points[found] - umklapp[n])) > _GRID_TOLERANCE:
            raise RuntimeError(  # pragma: no cover -- the key matched
                "k + q and its image differ by a non-integer vector"
            )
    return index, umklapp


def transverse_response(
    calculation,
    wavefunctions,
    eigenvalues,
    q,
    frequencies,
    *,
    ecut_response: float,
    broadening: float,
    flip: bool = False,
    k_batch: int | None | str = "default",
) -> SpinChiZero:
    """``X_0^{+-}(q, omega)`` over a response sphere and a frequency grid.

    Args:
        calculation: the :class:`~defumat.scf.driver.Calculation` the states
            belong to, on the **whole** k-grid.
        wavefunctions: ``(2, nk, nbnd, npwx)`` from a fixed-density run.
        eigenvalues: ``(2, nk, nbnd)``, Ry.
        q: the wavevector in crystal coordinates. It must be a difference of
            two k-points of the grid -- see :func:`commensurate_shift`.
        frequencies: ``(nw,)`` Ry, real or complex; ``broadening`` is added.
        ecut_response: the cutoff (Ry) selecting the matrix's G-vectors.
        broadening: ``eta``, Ry. Unlike the charge channel this is **not**
            added to a static point behind the caller's back: the Goldstone
            identity is exact only at ``eta = 0`` and an ``O(eta)`` error in it
            reads as an assembly bug.
        flip: exchange the two spin channels, giving ``X^{-+}``. Which of the
            two carries the magnon at positive ``omega`` depends on the
            majority direction, and the cheap way to find out is to look.
    """
    require_a_transverse_regime(calculation)

    wavefunctions = jnp.asarray(wavefunctions)
    eigenvalues = jnp.asarray(eigenvalues)
    if wavefunctions.ndim != 4 or wavefunctions.shape[0] != 2:
        raise ValueError(
            "the transverse susceptibility needs both spin channels' "
            f"wavefunctions, (2, nk, nbnd, npwx); got {wavefunctions.shape}"
        )
    majority, minority = (1, 0) if flip else (0, 1)

    sphere = spin_response_sphere(calculation, ecut_response)
    precision = calculation.system.cell.precision
    index, umklapp = commensurate_shift(calculation, q)

    weights, info = calculation.occupations(eigenvalues)
    weights = jnp.asarray(weights)

    grid = calculation.basis.smooth.grid
    volume = calculation.system.cell.volume
    mask = jnp.asarray(calculation.basis.planewaves.mask)
    fft_index = jnp.asarray(calculation.fft_index)

    # The gather index of the *shifted* sphere, one per k-point: the matrix
    # element reads the pair density at ``G + G0`` rather than at ``G``, and
    # ``G0`` is which reciprocal lattice vector brought ``k + q`` back into the
    # grid. This is the zone-edge shift of the Berry-phase machinery, and
    # without it a magnon dispersion is smooth, positive and wrong wherever
    # ``k + q`` leaves the first zone -- which is most of the grid.
    miller = np.asarray(sphere.miller)
    box = np.asarray(grid)
    shifted = (miller[None, :, :] + umklapp[:, None, :]) % box
    flat = jnp.asarray(
        (shifted[..., 0] * box[1] + shifted[..., 1]) * box[2] + shifted[..., 2]
    )

    zomega = jnp.asarray(frequencies) + 1j * precision.as_real(broadening)
    zomega = zomega.astype(precision.complex)
    slope = occupation_slope(calculation, eigenvalues, info)

    def one_k(arrays):
        (psi_up, index_up, mask_up, eig_up, occ_up, slope_up,
         psi_dn, index_dn, mask_dn, eig_dn, occ_dn, gather) = arrays
        return _one_k_terms(
            psi_up, index_up, mask_up, eig_up, occ_up, slope_up,
            psi_dn, index_dn, mask_dn, eig_dn, occ_dn,
            gather, grid, volume, zomega,
        )

    x = sum_k(
        one_k,
        (
            wavefunctions[majority], fft_index, mask,
            eigenvalues[majority], weights[majority], slope[majority],
            wavefunctions[minority][index], fft_index[index], mask[index],
            eigenvalues[minority][index], weights[minority][index],
            flat,
        ),
        batch=resolve_k_batch(k_batch),
    )
    return SpinChiZero(
        x=2.0 * x,
        frequencies=zomega,
        sphere=sphere,
        q=tuple(float(component) for component in np.asarray(q).reshape(3)),
        nbnd=int(eigenvalues.shape[-1]),
        flip=bool(flip),
    )


def _one_k_terms(psi_up, index_up, mask_up, eig_up, occ_up, slope_up,
                 psi_dn, index_dn, mask_dn, eig_dn, occ_dn,
                 gather, grid, volume, zomega):
    """One k-point's contribution, walked one majority band at a time.

    The pair axis is ``nbnd^2`` and a pair density is a whole FFT box, so
    forming them all at once is ``nbnd^2`` grid-sized fields. Walking the
    majority band with :func:`jax.lax.map` makes the working set ``nbnd`` of
    them instead, at no cost in flops -- the transforms are the same ones.
    """
    # The padding must be zeroed before the scatter: every padding entry
    # shares the flat index of ``G = 0``, so an unmasked coefficient lands on
    # top of a real one. **Both spheres are masked with their own mask**, the
    # down states living at ``k + q`` and therefore on a different sphere.
    fields_up = g_to_r(psi_up * mask_up[None, :], index_up, grid)
    fields_dn = g_to_r(psi_dn * mask_dn[None, :], index_dn, grid)

    def one_band(carry):
        field, energy, occupation, slope = carry
        # ``M_G = <psi_up| e^{-i(q+G).r} |psi_dn>`` for every minority band, at
        # the shifted gather.
        matrix = r_to_g(jnp.conj(field)[None] * fields_dn, gather)  # (nbnd, nm)

        difference = occupation - occ_dn  # f_up - f_dn, (nbnd,)
        denominator = zomega[:, None] + (energy - eig_dn)[None, :]

        # **The pair that is degenerate at omega = 0 is a 0/0 and takes the
        # occupation's own derivative.** As the two channels' eigenvalues come
        # together, ``f_up - f_dn -> (df/deps) (eps_up - eps_dn)``, so the whole
        # scalar tends to ``df/deps`` -- the smeared delta function, which is
        # the intraband weight of the Stoner continuum and is zero to machine
        # precision for a gapped system. It happens only where ``omega`` and the
        # gap vanish together, which for an antiferromagnet is *every* diagonal
        # pair at *every* k-point, by translation-plus-spin-flip. Leaving it to
        # a finite broadening drops the term silently instead of computing it.
        singular = jnp.abs(denominator) < _DEGENERACY_TOLERANCE
        safe = jnp.where(singular, 1.0, denominator)
        scalars = jnp.where(
            singular, jnp.broadcast_to(slope, difference.shape)[None, :],
            difference[None, :] / safe,
        ) / volume
        return jnp.einsum("wp,pa,pb->wab", scalars, jnp.conj(matrix), matrix)

    total = jax.lax.map(one_band, (fields_up, eig_up, occ_up, slope_up))
    return jnp.sum(total, axis=0)


#: Two eigenvalues closer than this (Ry) are treated as degenerate, and their
#: pair takes the smearing derivative instead of the ratio.
_DEGENERACY_TOLERANCE = 1.0e-8
