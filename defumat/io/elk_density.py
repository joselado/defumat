"""Reconstructing Elk's density at a point, and turning it into a defumat seed.

Elk stores a density in two pieces that meet at a sharp sphere: inside each
muffin tin it is a real-spherical-harmonic expansion on a logarithmic radial
mesh, :math:`\\rho(\\mathbf{r}) = \\sum_{lm} f_{lm}(r) R_{lm}(\\hat{\\mathbf{r}})`,
and everywhere else it is a periodic function on a uniform grid. Evaluating it
at an arbitrary point therefore means *first* deciding which region the point is
in, and only then doing an interpolation or a Fourier sum.

**This is a transcription of Elk's own ``rfpts.f90``, deliberately literal**,
because the check that says the transcription is right is a pointwise comparison
against Elk's ``RHO3D.OUT`` -- which Elk produces by calling exactly this routine
(``rhoplot.f90`` task 33 -> ``plot3d.f90`` -> ``rfpts``). Substituting a spline
for Elk's 4-point Lagrange interpolation, or a nearest-neighbour test for its
27-image sphere search, would leave a residual at interpolation level that could
not be read as either agreement or disagreement. Four pieces are copied rather
than improved:

* ``findmtpt``: fold the point into ``[0, 1)`` with ``r3frac``, then test all 27
  periodic images of every atom, **first match wins**. Skipping the images puts
  a point near a cell face in the interstitial when it belongs to a sphere.
* ``poly4``: a 4-point Lagrange polynomial on the window ``ir0 = ir - 2``,
  clamped at both ends of the mesh, with ``r`` clamped up to ``rsp(1)`` at the
  nucleus.
* ``genrlmv``: real spherical harmonics indexed ``j = l(l+1)+m+1``, defined from
  the *complex* ones as :math:`\\sqrt2\\,{\\rm Re}\\,Y_{lm}` for ``m > 0``,
  :math:`\\sqrt2\\,{\\rm Im}\\,Y_{lm}` for ``m < 0`` and :math:`{\\rm Re}\\,Y_{l0}`
  for ``m = 0``, Condon-Shortley inside :math:`Y_{lm}`.
* ``wsplint``/``splint``: the spline quadrature weights the radial integral uses.
  A Simpson rule on the same log mesh is 9e-6 away from Elk's own ``chgmt``,
  which is large enough to be mistaken for a reader bug.

One place this departs from ``rfpts``, knowingly. Elk truncates the sum to
``lmmaxi`` terms whenever the interpolation window starts inside the inner
region (``ir0 <= nri``); this always sums all ``lmmaxo``. The two agree
identically everywhere except a three-point window straddling ``nri`` -- the
file's ``lm > lmmaxi`` entries are exactly zero for ``ir <= nri``, so inside the
inner region the extra terms contribute nothing at all. On the hydrogen fixture
``nri = 129`` sits at ``r = 0.0141`` bohr against ``rmt = 1.4``, so the band is
0.4% of the sphere radius, and neither ``nrmti`` nor ``lmmaxi`` is written to
``STATE.OUT`` for the truncation to be reproduced from.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "real_spherical_harmonics",
    "poly4",
    "spline_weights",
    "evaluate_at",
    "muffin_tin_charges",
    "characteristic_function",
    "interstitial_charge",
    "interstitial_coefficients",
    "density_on",
    "SeedReport",
]

#: ``epslat``: Elk's tolerance for calling a fractional coordinate zero.
EPSLAT = 1.0e-6

#: :math:`Y_{00} = 1/\sqrt{4\pi}`.
Y00 = 0.5 / np.sqrt(np.pi)


# --- the pieces copied from Elk ----------------------------------------------

def real_spherical_harmonics(lmax: int, v: np.ndarray) -> np.ndarray:
    """``genrlmv``: real spherical harmonics at directions ``v``, packed by ``lm``.

    ``v`` is ``(..., 3)`` and need not be normalised. The result is
    ``(..., (lmax+1)**2)`` indexed ``j = l(l+1) + m`` (zero-based; Elk's
    one-based ``j = l(l+1)+m+1``), with ``m`` ascending from ``-l`` to ``+l``
    and the ``l`` blocks contiguous.
    """
    from scipy.special import sph_harm_y

    v = np.asarray(v, dtype=float)
    r = np.linalg.norm(v, axis=-1)
    safe = np.where(r > 0.0, r, 1.0)
    theta = np.where(r > 0.0, np.arccos(np.clip(v[..., 2] / safe, -1.0, 1.0)), 0.0)
    phi = np.arctan2(v[..., 1], v[..., 0])

    out = np.empty(v.shape[:-1] + ((lmax + 1) ** 2,), dtype=float)
    root2 = np.sqrt(2.0)
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            y = sph_harm_y(l, abs(m), theta, phi)
            j = l * (l + 1) + m
            if m > 0:
                # Y_{l,-m} = (-1)^m conj(Y_{lm}), so Im Y_{l,-m} = -(-1)^m Im Y_lm
                out[..., j] = root2 * np.real(y)
            elif m < 0:
                sign = -1.0 if (m % 2) else 1.0
                out[..., j] = root2 * sign * (-np.imag(y))
            else:
                out[..., j] = np.real(y)
    return out


def poly4(xa: np.ndarray, ya: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Elk's ``poly4``: the cubic through four points, evaluated at ``x``.

    Vectorised over a leading batch axis: ``xa`` is ``(..., 4)``, ``ya`` is
    ``(..., 4)`` broadcast against it, ``x`` is ``(...)``. The arithmetic is
    Elk's, term for term, so that a comparison against ``RHO3D.OUT`` is a
    comparison of the *data* rather than of two interpolation schemes.
    """
    x0 = xa[..., 0]
    x1 = xa[..., 1] - x0
    x2 = xa[..., 2] - x0
    x3 = xa[..., 3] - x0
    y0 = ya[..., 0]
    y1 = ya[..., 1] - y0
    y2 = ya[..., 2] - y0
    y3 = ya[..., 3] - y0
    t4 = x1 - x2
    t5 = x1 - x3
    t6 = x2 - x3
    t1 = x1 * x2 * y3
    t2 = x2 * x3 * y1
    t3 = x1 * x3
    t0 = 1.0 / (x2 * t3 * t4 * t5 * t6)
    t3 = t3 * y2
    c3 = t1 * t4 + t2 * t6 - t3 * t5
    a4 = x1 ** 2
    a5 = x2 ** 2
    a6 = x3 ** 2
    c2 = t1 * (a5 - a4) + t2 * (a6 - a5) + t3 * (a4 - a6)
    c1 = t1 * (x2 * a4 - x1 * a5) + t2 * (x3 * a5 - x2 * a6) + t3 * (x1 * a6 - x3 * a4)
    t = x - x0
    return y0 + t0 * t * (c1 + t * (c2 + c3 * t))


def _splint(x: np.ndarray, f: np.ndarray) -> float:
    """Elk's ``splint``: the integral of the piecewise cubic through ``(x, f)``.

    Only the ``n >= 5`` branch is transcribed. Every muffin-tin mesh Elk writes
    is a couple of hundred points, and the short branches go through ``polynm``,
    which nothing here needs.
    """
    n = len(x)
    if n < 5:
        raise ValueError(f"splint needs at least 5 points, got {n}")

    def piece(x0, x1, x2, x3, y0, y1, y2, y3, step):
        t4 = x1 - x2
        t5 = x1 - x3
        t6 = x2 - x3
        t3 = x1 * x3
        t1 = x1 * x2 * y3
        t2 = x2 * x3 * y1
        t0 = 0.5 / (t3 * t4 * t5 * t6)
        t3 = t3 * y2
        t7 = t1 * t4 + t2 * t6 - t3 * t5
        a4 = x1 ** 2
        a5 = x2 ** 2
        a6 = x3 ** 2
        u1 = t3 * a6 - t1 * a5
        u2 = t1 * a4 - t2 * a6
        u3 = t2 * a5 - t3 * a4
        s1 = x1 * u1 + x2 * u2 + x3 * u3
        s2 = u1 + u2 + u3
        return step * (y0 + t0 * (s1 + step * (0.5 * t7 * step - (2.0 / 3.0) * s2)))

    x0 = x[0]
    total = piece(
        x0, x[1] - x0, x[2] - x0, x[3] - x0,
        f[0], f[1] - f[0], f[2] - f[0], f[3] - f[0], x[2] - x0,
    )
    for i in range(2, n - 3):
        x0 = x[i]
        total += piece(
            x0, x[i - 1] - x0, x[i + 1] - x0, x[i + 2] - x0,
            f[i], f[i - 1] - f[i], f[i + 1] - f[i], f[i + 2] - f[i], x[i + 1] - x0,
        )
    # The last panel differs: it integrates out to ``x(n)`` rather than to the
    # next mesh point, so the step is ``x3`` and the ordering of the four points
    # around ``x(n-2)`` is the one ``splint`` uses at the end of the mesh.
    x0 = x[n - 3]
    x1 = x[n - 4] - x0
    x2 = x[n - 2] - x0
    x3 = x[n - 1] - x0
    y0 = f[n - 3]
    y1 = f[n - 4] - y0
    y2 = f[n - 2] - y0
    y3 = f[n - 1] - y0
    t4 = x1 - x2
    t5 = x1 - x3
    t6 = x2 - x3
    t1 = x1 * x2
    t2 = x2 * x3 * y1
    t3 = x1 * x3 * y2
    t0 = 0.5 / (t1 * t4 * t5 * t6)
    t1 = t1 * y3
    t7 = t1 * t4 + t2 * t6 - t3 * t5
    a4 = x1 ** 2
    a5 = x2 ** 2
    a6 = x3 ** 2
    u1 = t3 * a6 - t1 * a5
    u2 = t1 * a4 - t2 * a6
    u3 = t2 * a5 - t3 * a4
    s1 = x1 * u1 + x2 * u2 + x3 * u3
    s2 = u1 + u2 + u3
    total += x3 * (y0 + t0 * (s1 + x3 * (0.5 * t7 * x3 - (2.0 / 3.0) * s2)))
    return float(total)


def spline_weights(x: np.ndarray) -> np.ndarray:
    """Elk's ``wsplint``: quadrature weights ``w`` with ``int f dx = sum w*f``.

    ``splint`` is linear in ``f``, so the weights are what it returns on the
    unit vectors -- taken on sliding 9-point windows in the interior, which is
    what makes the rule local.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= 9:
        raise ValueError(
            f"the muffin-tin mesh has {n} points; Elk's short-mesh branch of "
            "wsplint is not transcribed and no real dataset reaches it"
        )
    w = np.empty(n)
    for i in range(4):
        f = np.zeros(9)
        f[i] = 1.0
        w[i] = _splint(x[:9], f)
    f = np.zeros(9)
    f[4] = 1.0
    for i in range(4, n - 4):
        w[i] = _splint(x[i - 4:i + 5], f)
    for i in range(4):
        f = np.zeros(9)
        f[i + 5] = 1.0
        w[n - 4 + i] = _splint(x[n - 9:], f)
    return w


# --- the sphere test ----------------------------------------------------------

def _find_muffin_tin(state, points_lattice: np.ndarray):
    """``findmtpt``, vectorised: which sphere each point is in, and where.

    Returns ``(atom, radius, direction)``. ``atom`` is the ``ias`` index or
    ``-1`` for the interstitial, ``radius`` the distance to that atom's centre
    and ``direction`` the cartesian offset from it (the argument the real
    harmonics are evaluated at).

    The 27 images matter: a point at a cell face is inside the sphere of an atom
    that sits across the boundary, and a search that only measured the distance
    to the atoms in the home cell would put it in the interstitial.
    """
    avec = state.geometry.avec                       # rows are lattice vectors
    v = np.asarray(points_lattice, dtype=float)
    frac = v - np.floor(v)
    frac = np.where((frac < EPSLAT) | (frac > 1.0 - EPSLAT), 0.0, frac)
    cart = frac @ avec                               # (np, 3)

    shifts = np.array(
        [i1 * avec[0] + i2 * avec[1] + i3 * avec[2]
         for i1 in (-1, 0, 1) for i2 in (-1, 0, 1) for i3 in (-1, 0, 1)]
    )                                                # (27, 3)

    atom = np.full(len(v), -1, dtype=int)
    radius = np.zeros(len(v))
    direction = np.zeros((len(v), 3))

    species_of = state.geometry.species_of()
    centres = state.geometry.positions @ avec        # (natmtot, 3), cartesian
    rmt = state.rmt

    open_ = np.ones(len(v), dtype=bool)
    # Elk's loop order is species-outer, atom-inner, image-innermost, and the
    # *first* match wins. Spheres do not overlap, so the order only decides what
    # happens on a tie at the boundary; keeping it costs nothing.
    for ias in range(state.natmtot):
        if not open_.any():
            break
        rmt2 = rmt[species_of[ias]] ** 2
        offsets = (cart[open_][:, None, :] - centres[ias]) + shifts[None, :, :]
        r2 = np.einsum("pic,pic->pi", offsets, offsets)
        which = np.argmin(r2, axis=1)
        best = r2[np.arange(len(which)), which]
        hit = best < rmt2
        if not hit.any():
            continue
        index = np.nonzero(open_)[0][hit]
        atom[index] = ias
        radius[index] = np.sqrt(best[hit])
        direction[index] = offsets[hit, which[hit]]
        open_[index] = False

    return atom, radius, direction


def _radial_index(state, species: int, r: np.ndarray) -> np.ndarray:
    """``findmtpt``'s mesh index: ``nint(t1 log(r/rmin)) + 1``, clamped.

    Fortran's ``nint`` rounds half away from zero; ``np.rint`` rounds half to
    even, and the two differ on exactly the points where a different four-point
    window would be picked.
    """
    nr = state.nrmt[species]
    rsp = state.rsp[species]
    rmin = rsp[0]
    t1 = (nr - 1) / np.log(state.rmt[species] / rmin)
    ir = np.where(
        r > rmin,
        np.floor(t1 * np.log(np.maximum(r, rmin) / rmin) + 0.5).astype(int) + 1,
        1,
    )
    return np.minimum(ir, nr)


# --- the interstitial ---------------------------------------------------------

def interstitial_coefficients(state, gmax2: float | None = None):
    """``rhoir`` in G-space: the Miller indices Elk keeps, and their coefficients.

    Elk's interstitial sum runs over ``ngvec``, **not** ``ngtot``: the FFT box's
    corners lie outside the ``|G| < gmaxvr`` sphere and ``rfpts`` drops them.
    ``ngvec`` is in ``STATE.OUT``, and since Elk sorts the whole box by ``|G|``
    and cuts at the first vector past ``gmaxvr``, the first ``ngvec`` sorted
    vectors are exactly the sphere -- no shell is ever split.

    The Miller range is Elk's ``intgv``, ``n/2 - n + 1 .. n/2``, which is not the
    same as ``fftfreq``'s.

    ``gmax2`` truncates further, to ``|G|^2 <= gmax2`` in 1/bohr^2. That is what
    a *seed* wants: defumat's dense sphere is usually smaller than Elk's, and
    coefficients past it would alias rather than be dropped.
    """
    ngridg = state.ngridg
    avec = state.geometry.avec
    bvec = 2.0 * np.pi * np.linalg.inv(avec)         # columns: b_i, rows index xyz
    bvec = bvec.T                                    # rows are b_i, to match ``at``

    ranges = [np.arange(n // 2 - n + 1, n // 2 + 1) for n in ngridg]
    grids = np.meshgrid(*ranges, indexing="ij")
    miller = np.stack([g.ravel() for g in grids], axis=1)
    gcart = miller @ bvec
    glen = np.linalg.norm(gcart, axis=1)

    order = np.argsort(glen, kind="stable")
    keep = order[:state.ngvec]
    miller = miller[keep]
    gcart = gcart[keep]

    if gmax2 is not None:
        inside = np.sum(gcart ** 2, axis=1) <= gmax2
        miller = miller[inside]
        gcart = gcart[inside]

    # ``zfftifc(3, ngridg, -1, z)`` is FFTW's forward transform divided by the
    # number of points, i.e. numpy's ``fftn`` normalised -- so the coefficients
    # are ``c(G)`` with ``f(r) = sum_G c(G) exp(+i G.r)``.
    box = np.fft.fftn(state.rhoir) / state.rhoir.size
    coefficients = box[
        miller[:, 0] % ngridg[0], miller[:, 1] % ngridg[1], miller[:, 2] % ngridg[2]
    ]
    return miller, gcart, coefficients


def characteristic_function(state) -> np.ndarray:
    """Elk's ``cfunir``: 0 inside the muffin tins, 1 outside, Fourier truncated.

    From ``gencfun.f90``'s own closed form, which needs nothing but the radii,
    the positions and the cell:

    .. math::

        \\tilde\\Theta_i(G) = \\frac{4\\pi R_i^3}{3\\Omega}\\ (G = 0), \\quad
        \\frac{4\\pi R_i^3}{\\Omega}\\frac{j_1(GR_i)}{GR_i}\\ (0 < G \\le G_{\\max}),
        \\quad 0\\ (G > G_{\\max})

    and :math:`\\tilde\\Theta(\\mathbf{G}) = \\delta_{G,0} - \\sum_i
    e^{-i\\mathbf{G}\\cdot\\mathbf{r}_i}\\tilde\\Theta_i(G)`.

    Nothing in the *reconstruction* needs this -- ``rfpts`` tests a sharp sphere
    and never touches ``cfunir``. It is here so that Elk's own ``chgir`` can be
    reproduced the way Elk computes it, which is the only way to see how far a
    truncated step is from a sharp one.
    """
    ngridg = state.ngridg
    miller, gcart, _ = interstitial_coefficients(state)
    glen = np.linalg.norm(gcart, axis=1)
    omega = state.omega
    centres = state.geometry.positions @ state.geometry.avec
    species_of = state.geometry.species_of()

    box = np.zeros(ngridg, dtype=complex)
    box[0, 0, 0] = 1.0
    for ias in range(state.natmtot):
        radius = state.rmt[species_of[ias]]
        volume = 4.0 * np.pi * radius ** 3
        gr = glen * radius
        # j1(x)/x -> 1/3 as x -> 0, which is the G = 0 entry.
        with np.errstate(invalid="ignore", divide="ignore"):
            shape = np.where(gr > 0.0, (np.sin(gr) - gr * np.cos(gr)) / gr ** 3, 1.0 / 3.0)
        form = (volume / omega) * shape
        phase = np.exp(-1j * (gcart @ centres[ias]))
        np.add.at(
            box,
            (miller[:, 0] % ngridg[0], miller[:, 1] % ngridg[1], miller[:, 2] % ngridg[2]),
            -form * phase,
        )
    return np.real(np.fft.ifftn(box) * box.size)


def interstitial_charge(state) -> float:
    """``chgir``: ``(Omega/ngtot) sum_r rhoir * cfunir``, Elk's own expression."""
    cfunir = characteristic_function(state)
    return float(state.omega / cfunir.size * np.sum(state.rhoir * cfunir))


def sharp_interstitial_charge(state) -> float:
    """The same charge with a **sharp** sphere test instead of a truncated step.

    This is not ``chgir`` and must not be compared with it as though it were.
    ``cfunir`` is a step function truncated at ``gmaxvr``, so it rings; the
    reconstruction here tests ``r < rmt`` exactly. The gap between the two is
    that ringing plus the staircasing of a sphere on a coarse grid, and it is
    reported as a measured number rather than chased as a bug.
    """
    ngridg = state.ngridg
    fractional = np.stack(
        np.meshgrid(*[np.arange(n) / n for n in ngridg], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    atom, _, _ = _find_muffin_tin(state, fractional)
    outside = (atom < 0).reshape(ngridg)
    return float(state.omega / state.rhoir.size * np.sum(state.rhoir * outside))


# --- the two things a caller wants --------------------------------------------

def muffin_tin_charges(state) -> np.ndarray:
    """``chgmt`` per atom: :math:`4\\pi Y_{00}\\int f_{00}(r)\\,r^2\\,dr`.

    A clean check against ``INFO.OUT``: a pure radial integral of the ``l = 0``
    channel, with the characteristic function nowhere in it. The :math:`r^2` is
    in the weight and not in the stored function -- ``rhomt`` is
    :math:`f_{lm}(r)`, not :math:`r^2 f_{lm}(r)`.
    """
    species_of = state.geometry.species_of()
    out = np.empty(state.natmtot)
    weights = {}
    for ias in range(state.natmtot):
        species = species_of[ias]
        if species not in weights:
            nr = state.nrmt[species]
            r = state.rsp[species][:nr]
            weights[species] = spline_weights(r) * r ** 2
        nr = state.nrmt[species]
        f00 = state.rhomt[0, :nr, ias]
        out[ias] = 4.0 * np.pi * Y00 * float(weights[species] @ f00)
    return out


def evaluate_at(state, points, coordinates: str = "lattice") -> np.ndarray:
    """The density at arbitrary points, in e/bohr^3.

    ``points`` is ``(np, 3)`` in lattice (fractional) coordinates by default, or
    cartesian bohr with ``coordinates="cartesian"``. This is ``rfpts``: a sharp
    in-or-out sphere test, then a real-harmonic sum with 4-point radial
    interpolation inside and the ``ngvec`` Fourier sum outside.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if points.shape[-1] != 3:
        raise ValueError(f"points must be (np, 3), got {points.shape}")
    if coordinates == "cartesian":
        points = points @ np.linalg.inv(state.geometry.avec)
    elif coordinates != "lattice":
        raise ValueError(f"coordinates must be 'lattice' or 'cartesian', got {coordinates!r}")

    atom, radius, direction = _find_muffin_tin(state, points)
    out = np.zeros(len(points))

    # -- inside a sphere
    species_of = state.geometry.species_of()
    lmaxo = state.lmaxo
    for ias in np.unique(atom[atom >= 0]):
        index = np.nonzero(atom == ias)[0]
        species = species_of[ias]
        nr = state.nrmt[species]
        rsp = state.rsp[species][:nr]
        r = radius[index]
        ir = _radial_index(state, species, r)
        ir0 = np.where(ir <= 3, 1, np.where(ir > nr - 2, nr - 3, ir - 2))
        r = np.maximum(r, rsp[0])
        window = ir0[:, None] - 1 + np.arange(4)        # (n, 4), zero-based
        xa = rsp[window]                                # (n, 4)
        ya = state.rhomt[:, :nr, ias][:, window]        # (lmmaxo, n, 4)
        values = poly4(xa[None], ya, r[None])           # (lmmaxo, n)
        rlm = real_spherical_harmonics(lmaxo, direction[index])  # (n, lmmaxo)
        out[index] = np.einsum("ln,nl->n", values, rlm)

    # -- the interstitial
    index = np.nonzero(atom < 0)[0]
    if index.size:
        _, gcart, coefficients = interstitial_coefficients(state)
        cart = points[index] @ state.geometry.avec
        # Chunked: the full outer product is ``npoints x ngvec`` complex, which
        # is a gigabyte on a fine grid and nothing at all in pieces.
        chunk = max(1, int(2_000_000 // max(len(gcart), 1)))
        for start in range(0, len(index), chunk):
            piece = cart[start:start + chunk]
            phase = piece @ gcart.T
            out[index[start:start + chunk]] = (
                np.cos(phase) @ coefficients.real - np.sin(phase) @ coefficients.imag
            )
    return out


class SeedReport:
    """What ``density_on`` measured on the way, so the caller can print it.

    ``integral`` is what the reconstruction integrates to on the target grid
    *before* renormalisation, ``nelec`` what defumat's valence count is, and
    ``outside_fraction`` how much of the density's own norm the Elk state puts
    past defumat's dense sphere -- which is where the nuclear cusp lives and is
    the term that explains a residual concentrated at small ``r``.
    """

    def __init__(self, integral, nelec, muffin_tin, outside_fraction, ngvec_kept, ngvec):
        self.integral = float(integral)
        self.nelec = float(nelec)
        self.muffin_tin = float(muffin_tin)
        self.outside_fraction = float(outside_fraction)
        self.ngvec_kept = int(ngvec_kept)
        self.ngvec = int(ngvec)

    def __repr__(self) -> str:
        return (
            f"SeedReport(integral={self.integral:.6f}, nelec={self.nelec:g}, "
            f"muffin_tin={self.muffin_tin:.6f}, "
            f"outside_fraction={self.outside_fraction:.3e}, "
            f"ngvec_kept={self.ngvec_kept}/{self.ngvec})"
        )


def density_on(state, calculation, renormalise: bool = True, report=None):
    """Reconstruct Elk's density on a defumat calculation's dense grid.

    Returns ``(nspin_mag, n1, n2, n3)`` in e/bohr^3, ready to hand to
    :func:`~defumat.scf.driver.run_scf` as ``starting_density``.

    **This is a seed and it is not a substitute for defumat's own SCF.** Elk's
    density is all-electron: it carries the core and the nuclear cusp, where
    defumat's is a smooth pseudo valence density. On hydrogen those are the same
    function -- there is no core at all -- and everywhere else they are not.
    ``run_scf`` trusts ``starting_density`` as given, neither renormalising it
    nor checking its shape, so both of those happen here.

    Two truncations, and both are deliberate:

    * the interstitial sum is cut to defumat's **dense sphere**
      (``ecutrho``). Elk's ``gmaxvr`` is typically 12 bohr^-1, a density cutoff
      of 144 Ry, which is at or above what a plane-wave run uses; the
      coefficients past it would alias onto the target grid rather than be
      dropped, so they are dropped;
    * the density is renormalised to ``nelec``, defumat's **valence** count.
      Elk's integrates to the all-electron total.

    A run whose ``nelec`` is far from what the Elk state integrates to is a run
    with a core in it, and that is refused by name rather than scaled away.
    """
    import jax.numpy as jnp

    # Refused before any work, the way the read-time refusals are: a run that
    # cannot take this seed should say so in the first line rather than after a
    # reconstruction it will throw away.
    nspin_mag = calculation.nspin_mag
    if nspin_mag != 1:
        raise NotImplementedError(
            f"this calculation has nspin_mag = {nspin_mag}, and only an "
            "unpolarized seed is implemented. An Elk magnetization is a second "
            "field with a transfer rule of its own, and splitting a charge "
            "density evenly between two channels would seed a magnetic run "
            "from a non-magnetic guess without saying so."
        )

    system = calculation.system
    cell = np.asarray(system.cell.at, dtype=float)
    if not np.allclose(cell, state.geometry.avec, atol=1e-6, rtol=1e-6):
        raise ValueError(
            "the Elk state and this calculation are not the same cell:\n"
            f"  Elk      {state.geometry.avec.tolist()}\n"
            f"  defumat  {cell.tolist()}\n"
            "A density is only transferable between identical lattices -- there "
            "is no interpolation here that would make it mean anything else."
        )

    dense = calculation.basis.dense
    grid = tuple(int(n) for n in dense.grid)
    miller_target = np.asarray(dense.miller)
    gcut = float(np.asarray(dense.kinetic(system.cell)).max())

    # -- the interstitial, straight onto the target grid.
    # For a *regular* grid the Fourier sum is an inverse FFT rather than a
    # quadratic point-by-G contraction, which is both exact and orders of
    # magnitude cheaper. Only the coefficients inside defumat's own sphere go in.
    miller, gcart, coefficients = interstitial_coefficients(state, gmax2=gcut)
    reachable = np.all(np.abs(miller) <= np.array(grid) // 2, axis=1)
    miller = miller[reachable]
    coefficients = coefficients[reachable]
    box = np.zeros(grid, dtype=complex)
    np.add.at(
        box,
        (miller[:, 0] % grid[0], miller[:, 1] % grid[1], miller[:, 2] % grid[2]),
        coefficients,
    )
    rho = np.real(np.fft.ifftn(box) * box.size)

    # -- the muffin tins, pointwise, overwriting the interstitial there.
    fractional = np.stack(
        np.meshgrid(*[np.arange(n) / n for n in grid], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    atom, radius, direction = _find_muffin_tin(state, fractional)
    inside = np.nonzero(atom >= 0)[0]
    if inside.size:
        values = evaluate_at(state, fractional[inside])
        flat = rho.reshape(-1)
        flat[inside] = values
        rho = flat.reshape(grid)

    omega = state.omega
    integral = float(rho.sum()) * omega / rho.size
    nelec = float(calculation.nelec)

    # How much of the state's own norm sits past defumat's dense sphere. The
    # answer cannot be read off the target grid, where it has already aliased
    # in, so it is measured on Elk's own G-set.
    _, gcart_full, coefficients_full = interstitial_coefficients(state)
    weight = np.abs(coefficients_full) ** 2
    outside = np.sum(gcart_full ** 2, axis=1) > gcut
    outside_fraction = float(weight[outside].sum() / max(weight.sum(), 1e-300))

    measured = SeedReport(
        integral=integral,
        nelec=nelec,
        muffin_tin=float(muffin_tin_charges(state).sum()),
        outside_fraction=outside_fraction,
        ngvec_kept=len(miller),
        ngvec=state.ngvec,
    )

    if abs(integral - nelec) > 0.5 * max(nelec, 1.0):
        raise NotImplementedError(
            f"the Elk state integrates to {integral:.4f} electrons and this "
            f"calculation has {nelec:g} valence electrons. Elk's density is "
            "all-electron, so a gap this size is the frozen core, and there is "
            "no way to subtract it from STATE.OUT alone -- the core is inside "
            "rhomt with nothing to separate it from the valence charge. "
            "Transferring a density for an element with a core needs a "
            "*difference* against a one-iteration Elk state, which is not "
            "implemented; hydrogen, with no core at all, is what this path is "
            "for."
        )

    if renormalise:
        rho = rho * (nelec / integral)

    if report is not None:
        report.append(measured)

    return jnp.asarray(rho)[None]
