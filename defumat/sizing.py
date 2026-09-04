"""What a run will cost before anything is allocated.

``Calculation.__init__`` builds the G-vectors, the plane-wave basis and the
projectors on the device, so the first thing a too-large input does is die in
setup -- with no report of *what* was too large. That is the wrong order for a
calculation whose whole feasibility question is a working set: a 157-atom slab
at ``ecutwfc = 60`` either fits on the card or it does not, and the answer is
arithmetic on the cutoffs and the datasets rather than something to be
discovered by running out of memory.

**Nothing here touches the device.** Every count is host-side ``numpy``, and
the G-vector enumeration -- the one step whose intermediate is larger than its
result -- runs in slabs of the first Miller index so that peak host memory stays
a few tens of megabytes whatever the box is. That is what lets this be called on
a laptop for a calculation destined for a GPU.

The counts are **exact**, not extrapolated: ``ngm``, ``ngms`` and ``npwx`` come
from the same predicates :mod:`defumat.basis.gvectors` and
:mod:`defumat.basis.planewaves` select with, ``nbnd`` from
:func:`~defumat.scf.driver.default_nbnd`, and ``nkb`` from
:func:`~defumat.pseudo.projectors.projector_channels`. A count that disagreed
with what the setup then built would be worse than no count at all, so
``tests/unit/test_sizing.py`` asserts each against a real ``Calculation`` on
cells small enough to build.

**The byte figures are a floor and say so.** They cover the arrays whose size is
a function of the basis -- the wavefunctions, the projectors, the eigensolver's
subspace, the fields on the two grids -- which is what decides whether a run
starts. They do not cover XLA's own scratch, the temporaries of a fused kernel,
or the autodiff tape of a derivative that has not been asked for; a reverse-mode
force carries intermediates this cannot see. Read the total as "at least this",
which is the direction that makes it useful.

**One of those omissions is large enough to decide feasibility on its own, so it
is estimated separately.** A compiled executable's temporaries are one
contiguous buffer, and for ``davidson_eigensolver_all`` at a 157-atom slab's
shapes that buffer is **110 GiB** where this module's Davidson lines add to 69
GB -- so a run whose floor fits on the card dies asking for a single allocation
larger than the card. :attr:`SizeEstimate.eigensolver_buffer` reports it and
:attr:`SizeEstimate.peak_bytes` is the number that decides. The fit is

    2.18 nvecx npwx zc  +  4.20 nbnd npwx zc  +  2.00 band_batch N_smooth zc

-- the subspace ``psi``/``hpsi`` pair, about five more ``(nbnd, npwx)`` blocks
live inside the Davidson subspace solve, and roughly two FFT boxes per band in
flight. It comes from compiling the real function over ``nbnd``, ``david`` and
``DEFUMAT_BAND_BATCH`` on a small cell and reading
``memory_analysis().temp_size_in_bytes``. It is fitted **through the origin**:
a constant term improves the fit on the cell it was measured on (1.6 per cent
against 6.4) and means nothing three orders of magnitude out, which is where it
is used.

**The coefficients are measured on the CPU backend and were checked against
fourteen H200 measurements at the slab's own shapes**, over ``david`` in
{2, 3, 4}, ``nbnd`` in {900, 1020} and ``band_batch`` in {16, 32, 64, 128}. On
**twelve** of the fourteen the formula was within **3.1 per cent** -- a stronger
transfer than a CPU fit had any right to, given that the two backends schedule
buffers differently. The two it missed are both ``david = 2`` with
``band_batch = 64``, where the card asked for **30 per cent less** than the
formula said (64.5 GiB against 84.3, and 59.7 against 77.1): the buffer is *not
monotonic* in the band batch there -- 66.1, 74.9, 64.5 GiB at 16, 32, 64 -- so
at a small subspace XLA finds a schedule the third term does not describe. It
errs in the safe direction, and it errs.

**Those GPU points were taken before the second coefficient fell from 6.20 to
4.20**, which is :func:`~defumat.solvers.davidson.davidson_eigensolver` no
longer carrying the expansion block across its loop and no longer holding two
copies of that loop for the robustness retry -- two ``(nbnd, npwx)`` blocks
exactly, measured as 2.00 by a refit and 11.6 GiB at the slab's shapes. Add
``2 nbnd npwx zc`` back to compare against them. Where the answer is close to
the card, **measure it** rather than adding anything back:
``tools/gpu/davidson_memory.py`` returns the real backend's buffer at any
(``david``, ``nbnd``, ``band_batch``) without executing anything, and
``tools/gpu/force_memory.py`` does the same for the reverse-mode force, whose
tape this module cannot see either.

**``band_batch`` is therefore an assumption of the estimate the way ``k_batch``
and ``davidson_basis`` already were**, and it is reported beside them: it is the
third term above, worth 22 GiB at 64 and 44 at 128 on that slab.

References for the conventions rather than the code: ``PW/src/setup.f90`` for
``nbnd``, ``Modules/recvec_subs.f90`` (``ggen``) for the sphere, and
``PW/src/n_plane_waves.f90`` for ``npwx``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from defumat.basis.fftgrid import fft_grid_dimensions, gcut_from_ecut
from defumat.system.builder import System

__all__ = ["SizeEstimate", "estimate_size"]

#: Bytes in one complex number at the precision a run will use.
_COMPLEX_BYTES = {"double": 16, "single": 8}
_REAL_BYTES = {"double": 8, "single": 4}

#: Coefficients of the eigensolver's XLA temp buffer, in units of
#: ``nvecx npwx zc``, ``nbnd npwx zc`` and ``band_batch N_smooth zc``. Fitted to
#: ``memory_analysis().temp_size_in_bytes`` of the compiled
#: :func:`~defumat.solvers.davidson.davidson_eigensolver_all` over ``nbnd`` in
#: {32, 48, 64, 96}, ``david`` in {2, 3, 4, 6} and ``DEFUMAT_BAND_BATCH`` in
#: {4, 8, 32} on ``benchmarks/si16-1k-ecut30.in``, **through the origin**;
#: residuals <= 8 per cent there. See the module docstring for the H200 check
#: and for what changed under it. Measured on the CPU backend: replace them
#: with a GPU fit rather than tuning them, and say which backend they came from
#: when you do.
#:
#: The band-batch sweep goes down to **4** rather than starting at 8, and that
#: is what makes the second coefficient meaningful: what separates the three
#: terms is the ratio ``nbnd npwx / (band_batch N_smooth)``, which is 0.5 on
#: the 157-atom slab and 0.06 at ``si16`` with ``band_batch = 32``. Fitted only
#: in the batch-dominated corner, the ``nbnd`` term is barely identified at all
#: -- and a change worth two whole blocks reads as 1 per cent there and as 10
#: per cent at the ratio a production run actually has.
_SUBSPACE_COEFFICIENT = 2.18
_RITZ_COEFFICIENT = 4.20
_FFT_COEFFICIENT = 2.00

#: Miller-index slabs are counted this many rows at a time. The intermediate is
#: ``rows * n2 * n3`` triples, so this bounds the host peak at tens of MB for
#: any box a plane-wave code would use.
_SLAB = 8


# ``ggen``'s representative of each ``(G, -G)`` pair, taken from the module that
# selects with it rather than restated here: the count below is only meaningful
# if the predicate is the same one.
from defumat.basis.gvectors import _half_sphere


def _walk_sphere(grid, bg, gcut, gamma_only):
    """Yield ``(miller, g2)`` for the G-vectors of one slab at a time.

    The enumeration range and the ``<=`` are
    :func:`~defumat.basis.gvectors.generate_gvectors`'s, so the count this
    produces is the count that function would return.
    """
    ranges = [np.arange(-((n - 1) // 2), (n - 1) // 2 + 1) for n in grid]
    j, k = np.meshgrid(ranges[1], ranges[2], indexing="ij")
    j, k = j.ravel(), k.ravel()
    for start in range(0, len(ranges[0]), _SLAB):
        rows = ranges[0][start : start + _SLAB]
        miller = np.stack(
            [
                np.repeat(rows, len(j)),
                np.tile(j, len(rows)),
                np.tile(k, len(rows)),
            ],
            axis=1,
        )
        g2 = np.sum((miller @ bg) ** 2, axis=1)
        inside = g2 <= gcut
        if gamma_only:
            inside &= _half_sphere(miller)
        if inside.any():
            yield miller[inside], g2[inside]


def _count_sphere(grid, bg, gcut, gamma_only) -> int:
    return sum(len(m) for m, _ in _walk_sphere(grid, bg, gcut, gamma_only))


def _plane_wave_counts(grid, bg, gcut_rho, gcut_smooth, gcut_wfc, gamma_only, kcoords):
    """``npw`` per k-point, counted against the *smooth* set.

    The plane waves are selected from the smooth G-vectors
    (:func:`~defumat.basis.builder.build_basis` passes ``smooth``), so the
    predicate is ``|k+G|^2 <= gcutw`` over the G with ``|G|^2 <= gcut_smooth``.
    Both are applied slab by slab, which keeps the whole thing out of memory:
    only the ``nk`` running counts survive a slab.
    """
    counts = np.zeros(len(kcoords), dtype=np.int64)
    for miller, g2 in _walk_sphere(grid, bg, gcut_rho, gamma_only):
        smooth = miller[g2 <= gcut_smooth]
        if not len(smooth):
            continue
        g = smooth @ bg
        for ik, k in enumerate(kcoords):
            kg2 = np.sum((k + g) ** 2, axis=1)
            counts[ik] += int(np.count_nonzero(kg2 <= gcut_wfc))
    return counts


@dataclass(frozen=True)
class SizeEstimate:
    """The shapes a run will allocate, and a floor on the bytes they cost."""

    #: Structure and electrons.
    nat: int
    nsp: int
    nelec: float
    nbnd: int
    #: Spin, kept as QE's three separate numbers (``CLAUDE.md``).
    nspin: int
    npol: int
    nspin_mag: int
    nk: int
    #: Basis.
    ngm: int
    ngms: int
    npwx: int
    npw: tuple
    nkb: int
    dense_grid: tuple
    smooth_grid: tuple
    #: Whether the *input* asked for ``K_POINTS gamma``.
    gamma_requested: bool
    #: Whether the half-sphere storage is what will actually be allocated. It
    #: is not, today: :func:`~defumat.scf.driver._without_gamma_storage`
    #: substitutes an explicit ``k = 0`` on the full sphere, because the gamma
    #: trick's storage is generated and not consumed. Reporting the halved
    #: counts would be reporting a run that does not happen.
    gamma_only: bool
    doublegrid: bool
    #: Precision policy the cell carries.
    precision: str
    #: The Davidson subspace multiple and the k-points in flight this was
    #: sized for. Reported because they are *assumptions* rather than
    #: properties of the input, and a reader comparing two estimates has to be
    #: able to see which one moved.
    davidson_basis: int
    k_batch: int | None
    #: Bands in flight inside ``h_psi`` -- :mod:`defumat.batching`'s other
    #: dial, and an assumption of the estimate for the same reason the two
    #: above it are: it sizes the third term of the eigensolver's buffer and
    #: appears nowhere else.
    band_batch: int | None
    #: ``name -> bytes`` for each array whose size the basis fixes.
    arrays: dict = field(default_factory=dict)
    #: The compiled eigensolver's single contiguous XLA temp buffer, estimated
    #: from the fit in the module docstring. It is **not** a member of
    #: ``arrays``: it supersedes the two Davidson entries there rather than
    #: adding to them, which is what :attr:`peak_bytes` does with it.
    eigensolver_buffer: int = 0

    #: The ``arrays`` entries the eigensolver's buffer stands in for.
    _SUPERSEDED = ("Davidson subspace psi+hpsi", "Davidson Ritz block")

    @property
    def total_bytes(self) -> int:
        return int(sum(self.arrays.values()))

    @property
    def peak_bytes(self) -> int:
        """The number that decides whether the run starts.

        The persistent arrays, with the two Davidson lines replaced by the one
        buffer XLA actually asks the allocator for. Those two are a floor on
        what the subspace *contains*; the buffer is what a single allocation
        has to be, and on a large cell it is the larger by about 60 per cent.
        """
        resident = sum(
            size for name, size in self.arrays.items()
            if name not in self._SUPERSEDED
        )
        return int(resident + self.eigensolver_buffer)

    @property
    def dense_points(self) -> int:
        return int(np.prod(self.dense_grid))

    @property
    def smooth_points(self) -> int:
        return int(np.prod(self.smooth_grid))

    def report(self) -> str:
        """A human-readable table -- what the CLI prints."""
        def gb(n):
            return f"{n / 2**30:9.2f} GB"

        lines = [
            "Sizes",
            f"  atoms                {self.nat} of {self.nsp} species",
            f"  valence electrons    {self.nelec:g}",
            f"  bands (nbnd)         {self.nbnd}",
            f"  k-points (nk)        {self.nk}",
            f"  nspin/npol/nspin_mag {self.nspin}/{self.npol}/{self.nspin_mag}",
            f"  dense G (ngm)        {self.ngm}",
            f"  smooth G (ngms)      {self.ngms}"
            + ("" if self.doublegrid else "   [same grid: dual <= 4]"),
            f"  plane waves (npwx)   {self.npwx}"
            + (f"   (min {min(self.npw)})" if len(set(self.npw)) > 1 else ""),
            f"  projectors (nkb)     {self.nkb}",
            f"  dense FFT grid       {'x'.join(str(n) for n in self.dense_grid)}"
            f"  ({self.dense_points} points)",
            f"  smooth FFT grid      {'x'.join(str(n) for n in self.smooth_grid)}"
            f"  ({self.smooth_points} points)",
            "",
            f"Memory floor ({self.precision} precision, "
            f"diago_david_ndim = {self.davidson_basis}, "
            f"k in flight = {self.k_batch if self.k_batch is not None else 'all'}, "
            f"bands in flight = "
            f"{self.band_batch if self.band_batch is not None else 'all'})",
        ]
        if self.gamma_requested and not self.gamma_only:
            lines[1:1] = [
                "  NOTE: K_POINTS gamma was requested and cannot be consumed by",
                "        this run (it needs norm-conserving and nosym), so it is",
                "        an explicit k = 0 on the FULL sphere. Everything below",
                "        is sized for that -- about twice what gamma would cost.",
            ]
        elif self.gamma_only:
            lines[1:1] = [
                "  NOTE: gamma-only storage IS in use: one plane wave of each",
                "        (G, -G) pair, so npwx and every array a band lives in",
                "        is halved. The dense G set is whole -- only the",
                "        wavefunction sphere halves.",
            ]
        for name, size in sorted(self.arrays.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {name:<34s}{gb(size)}")
        lines.append(f"  {'TOTAL (floor, see module docstring)':<34s}{gb(self.total_bytes)}")
        lines += [
            "",
            "What the allocator is actually asked for",
            f"  {'eigensolver XLA temp buffer':<34s}{gb(self.eigensolver_buffer)}",
            "        one contiguous allocation, and it stands in for the two",
            "        Davidson lines above rather than adding to them. Estimated",
            "        from a fit measured on the CPU backend and checked on an",
            "        H200: within 3.1% on 12 of 14 points, 30% HIGH on the two",
            "        at david 2 / band_batch 64. Close to the card? Measure it",
            "        with tools/gpu/davidson_memory.py.",
            f"  {'PEAK (resident + that buffer)':<34s}{gb(self.peak_bytes)}",
        ]
        return "\n".join(lines)


def estimate_size(
    system: System,
    pseudos,
    nbnd: int | None = None,
    k_batch: int | None = None,
    davidson_basis: int | None = None,
    band_batch: int | None | str = "default",
) -> SizeEstimate:
    """Size a run from its input alone, allocating nothing on the device.

    Args:
        system: the :class:`~defumat.system.builder.System` an SCF would run.
        pseudos: the loaded pseudopotentials, in species order -- the same
            tuple :class:`~defumat.scf.driver.Calculation` takes. They are read
            for ``z_valence`` and for their projector channels only; nothing is
            transformed.
        nbnd: override the band count, as an input's ``nbnd`` would.
        k_batch: how many k-points the eigensolver holds in flight at once --
            :mod:`defumat.batching`'s dial. ``None`` means the whole axis, which
            is what an accelerator defaults to; ``1`` is QE's own loop and is
            the CPU default.
        davidson_basis: the subspace multiple ``nvecx/nbnd`` --
            ``diago_david_ndim``. ``None`` takes
            :data:`~defumat.solvers.davidson.DAVID_NDIM`, which is what a run
            with nothing set uses; it is read from the constant rather than
            written as a literal so the two cannot drift apart.

            **A caller who has a** :class:`~defumat.calculator.Calculator`
            **should not be passing this by hand**:
            :meth:`~defumat.calculator.Calculator.estimate` fills it from the
            same defaults the run would use, which is the whole point of a size
            estimate. Sizing at 4 an input that says ``diago_david_ndim = 2``
            reports a run that does not happen, and by 35 GB on the cell this
            module was written for -- the same mistake as sizing
            ``K_POINTS gamma`` as the request rather than the substitution, one
            option along.
        band_batch: how many bands ``h_psi`` transforms at once --
            :mod:`defumat.batching`'s other dial, and the third term of the
            eigensolver's buffer. ``"default"`` resolves it the way a run
            would, from ``DEFUMAT_BAND_BATCH`` and the platform; ``None`` means
            every band at once, which is what an accelerator defaults to.

    Returns:
        a :class:`SizeEstimate`. Its counts are exact; its bytes are a floor.
    """
    # What the SCF will actually run, which is not always what was asked for.
    # ``K_POINTS gamma`` is *consumed* where the run can consume it and
    # substituted for an explicit k = 0 on the full sphere where it cannot, and
    # this has to mirror that decision exactly -- sizing the request rather than
    # the run understates every array by two, and sizing the substitution where
    # the run consumes the half sphere overstates them by two.
    from defumat.scf.driver import _without_gamma_storage, gamma_storage_is_consumable

    gamma_requested = bool(system.kpoints.gamma_only)
    gamma_only = gamma_storage_is_consumable(system, pseudos)
    if not gamma_only:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = _without_gamma_storage(system)

    cell = system.cell
    structure = system.structure
    bg = np.asarray(cell.bg_2pi_alat)
    at = np.asarray(cell.at_alat)

    # The FFT box carries the fractional translations' divisibility, exactly as
    # ``build_basis`` sets it -- and ``nosym`` removes that constraint, which
    # changes the box and so the G-count.
    if system.nosym:
        factors = (1, 1, 1)
    else:
        from defumat.system.symmetry import find_symmetries

        factors = find_symmetries(cell, structure).fft_factors()

    gcut_rho = gcut_from_ecut(system.ecutrho, cell.alat)
    dense_grid = fft_grid_dimensions(at, bg, gcut_rho, factors)

    dual = system.ecutrho / system.ecutwfc
    doublegrid = dual > 4.0 + 1.0e-8
    if doublegrid:
        gcut_smooth = gcut_from_ecut(4.0 * system.ecutwfc, cell.alat)
        smooth_grid = fft_grid_dimensions(at, bg, gcut_smooth, factors)
    else:
        gcut_smooth, smooth_grid = gcut_rho, dense_grid

    # The **dense** set is whole whatever the storage is: only the plane-wave
    # sphere halves (``build_basis``). That is why ``ngm`` below carries no
    # ``gamma_only`` and ``npwx`` does.
    ngm = _count_sphere(dense_grid, bg, gcut_rho, False)
    ngms = ngm if not doublegrid else _count_sphere(
        dense_grid, bg, gcut_smooth, False
    )

    gcut_wfc = gcut_from_ecut(system.ecutwfc, cell.alat)
    npw = _plane_wave_counts(
        dense_grid, bg, gcut_rho, gcut_smooth, gcut_wfc, gamma_only,
        np.asarray(system.kpoints.coords),
    )
    npwx = int(npw.max())

    # Electrons and bands, by ``Calculation``'s own rules.
    from defumat.scf.driver import default_nbnd
    from defumat.pseudo.projectors import projector_channels

    nelec = float(sum(pseudos[t].z_valence for t in structure.types))
    nspin, npol, nspin_mag = system.nspin, system.npol, system.nspin_mag
    if nbnd is None:
        nbnd = system.nbnd
    if nbnd is None:
        nelup = neldw = None
        if system.tot_magnetization is not None and nspin == 2:
            nelup = 0.5 * (nelec + system.tot_magnetization)
            neldw = 0.5 * (nelec - system.tot_magnetization)
        nbnd = default_nbnd(
            nelec, system.occupations, nelup=nelup, neldw=neldw,
            noncolin=(nspin == 4),
        )
    nkb = sum(len(projector_channels(pseudos[t])) for t in structure.types)

    from defumat.solvers.davidson import DAVID_NDIM

    if davidson_basis is None:
        davidson_basis = DAVID_NDIM
    if isinstance(band_batch, str):
        from defumat.batching import resolve_band_batch

        band_batch = resolve_band_batch(band_batch)
    name = getattr(cell.precision, "name", "double")
    zc, zr = _COMPLEX_BYTES.get(name, 16), _REAL_BYTES.get(name, 8)
    nk = len(npw)
    ndim = npwx * npol  # a spinor is one vector of length 2 npwx

    # ``nspin`` is the wavefunctions' leading axis for a collinear run and is 1
    # for a spinor one, where the two components are inside ``ndim``.
    wf_spin = 2 if nspin == 2 else 1
    # **The eigensolver is not doubled by spin.** ``Calculation.diagonalize``
    # loops over the channels and solves them one after another, so the Davidson
    # workspace is one channel's whatever ``nspin`` is -- where the
    # *wavefunctions* it writes into are held for both. Doubling it was worth
    # 90 GB of phantom on the cell this module was written for.
    k_live = nk if k_batch is None else min(k_batch, nk)
    nvecx = davidson_basis * nbnd

    arrays = {
        "wavefunctions (nspin,nk,nbnd,ndim)": wf_spin * nk * nbnd * ndim * zc,
        "projectors vkb (nk,npwx,nkb)": nk * npwx * nkb * zc,
        # ``psi`` and ``hpsi``, both ``(nvecx, ndim)`` -- the subspace and H
        # applied to it. ``S|psi>`` is deliberately not stored (the Ritz
        # vector's projections are a rotation of ``becq``), which is why this
        # is two and not three.
        "Davidson subspace psi+hpsi": 2 * k_live * nvecx * ndim * zc,
        # ``evc``, ``hevc``, ``sevc`` and ``residual``, each ``(nbnd, ndim)``,
        # live at once inside ``solve``.
        "Davidson Ritz block": 4 * k_live * nbnd * ndim * zc,
        "density+potential (nspin_mag,ngm)": 2 * nspin_mag * ngm * zc,
        "fields on dense grid": 3 * nspin_mag * int(np.prod(dense_grid)) * zr,
    }
    if nkb:
        arrays["Davidson becp+becq (nvecx,nkb)"] = 2 * k_live * nvecx * nkb * zc
    if doublegrid:
        arrays["fields on smooth grid"] = (
            2 * nspin_mag * int(np.prod(smooth_grid)) * zr
        )

    # A GGA carries the density's gradient on the dense grid -- three components
    # per spin channel, plus the vector field ``gradcorr`` builds back before
    # taking its divergence. An LDA carries neither, so this is asked of the
    # functional the run will actually use rather than assumed.
    from defumat.xc.functional import resolve_functional

    functional = resolve_functional(
        [p.functional for p in pseudos], system.input_dft
    )
    if functional.is_gradient:
        arrays["GGA gradient temporaries"] = (
            6 * nspin_mag * int(np.prod(dense_grid)) * zr
        )

    # The eigensolver's own XLA temp buffer -- see the module docstring. The
    # FFT term is on the **smooth** grid, which is the box ``h_psi`` transforms
    # a band in; ``band_batch = None`` is every band at once.
    bands_in_flight = nbnd if band_batch is None else min(band_batch, nbnd)
    eigensolver_buffer = int(
        _SUBSPACE_COEFFICIENT * nvecx * ndim * zc
        + _RITZ_COEFFICIENT * nbnd * ndim * zc
        + _FFT_COEFFICIENT * bands_in_flight * int(np.prod(smooth_grid)) * zc
    )

    return SizeEstimate(
        nat=len(structure.types), nsp=len(pseudos), nelec=nelec, nbnd=int(nbnd),
        nspin=nspin, npol=npol, nspin_mag=nspin_mag, nk=nk,
        ngm=ngm, ngms=ngms, npwx=npwx, npw=tuple(int(n) for n in npw), nkb=nkb,
        dense_grid=tuple(int(n) for n in dense_grid),
        smooth_grid=tuple(int(n) for n in smooth_grid),
        gamma_requested=gamma_requested, gamma_only=gamma_only,
        doublegrid=doublegrid, precision=name,
        davidson_basis=int(davidson_basis), k_batch=k_live,
        band_batch=None if band_batch is None else int(band_batch),
        arrays=arrays, eigensolver_buffer=eigensolver_buffer,
    )
