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
subspace, the fields on the two grids -- and the two setup products the run
keeps, the augmentation charge ``Q_ij(G)`` and, on a PAW dataset, the
one-centre tensors, which is what decides whether a run starts. The second is a
function of the datasets alone, ``nh^2 nlm mesh`` per distinct dataset whatever
the cell, so it can be the largest line on a two-atom cell and is small beside
``Q_ij(G)``, which grows with ``ngm``, on a slab. Both are counted per distinct
*dataset* rather than per species label, which is what the setup holds (P73,
P110). They do not cover XLA's own scratch, the temporaries of a fused kernel,
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

    k_batch (2.18 nvecx npwx zc + 4.20 nbnd npwx zc
             + 2.00 band_batch npol N_smooth zc)

-- the subspace ``psi``/``hpsi`` pair, about five more ``(nbnd, npwx)`` blocks
live inside the Davidson subspace solve, and roughly two FFT boxes per band in
flight. It comes from compiling the real function over ``nbnd``, ``david`` and
``DEFUMAT_BAND_BATCH`` on a small cell and reading
``memory_analysis().temp_size_in_bytes``, **one k-point at a time**: the
parenthesis is one chunk, and ``davidson_eigensolver_all`` holds ``k_batch`` of
them at once, which is why the whole of it carries that factor. It is fitted **through the origin**:
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
at a small subspace XLA finds a schedule the third term does not describe.

**At ``david = 2`` the error goes both ways, and the low direction is the one
that kills a run.** A second sweep at a *different* cell and a different card --
the 45-atom NiBr2 spinor PAW slab, ``nbnd = 403``, on an A100-80GB (job
**20252132**, commit ``ad89fd9``) -- measured **23.18 GiB** at
``david 2 / band_batch 16`` where this formula said **18.55**: **4.63 GiB, 20
per cent LOW**. That figure has **two independent routes** and is the one number
from that cell that does: :mod:`tools.gpu.davidson_memory` compiled it, and the
run that died asked the allocator for exactly **24,889,513,216 B** from
``jit__every_k``, which is the same 23.18 GiB. So the fourteen points above do not describe this corner; they
were taken at ``nbnd`` 900/1020 on an H200, and the low reading is at 403 on an
A100. Both errors are at the smallest subspace, which is where the schedule is
least like the fit, and ``david = 2`` is also what a memory-constrained cell is
forced into -- so the corner that errs is the corner that gets used. **Where the
answer is close to the card at ``david = 2``, the formula is not evidence.**

The non-monotonicity moved corners with the backend rather than staying with
the formula: on the H200 it was at ``david 2`` (66.1, 74.9, 64.5 at 16, 32, 64)
and on the A100 that row rises cleanly (23.18, 23.94, 25.25) while ``david 3``
is the ragged one (29.30, 27.74, 29.07). That is XLA's scheduler, not this
expression, and it is the reason the sentence above says *measure it*.

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

**A short calibration run measures the regime the calculation leaves, not the
one it fails in.** This is the counterpart of "never time a first call", for
memory and for a whole SCF rather than one executable. On a 45-atom NiBr2 spinor
PAW slab, iterations 1 to 12 held a flat 77.63 GB at about 21 s each and gave no
warning at all; at iteration 10 ``ethr`` reached 2.30e-6 and the Davidson average
went from **2.0 inner steps to 73.5**, the iteration cost from 21 s to 390 s, and
three iterations later the allocator's arena had no hole large enough left --
25.7 GB free in total, largest hole 14.9 GB, and a 23.0 GB request. So a
two-iteration calibration, and even a twelve-iteration one, says nothing about
the peak a converging run reaches: the loose starting threshold is a **different
calculation** from the tight one it schedules towards. Where a card is chosen on
this estimate, choose it for the tight end.

**A gap against a measured peak is not automatically a model error, and on the
one large cell where that was tested it mostly was not.** The same 45-atom
spinor PAW slab reported 49.52 GB against a 77.63 GB measured peak, which looked
like a 60 per cent shortfall in this module. It was not: **21.40 GB of the 28.11
GB gap was one allocation**, the eigensolver's finiteness guard, which sat
*outside* the executable this module sizes and so appeared in no line of the
report at all. Folding that guard into the solver removed it. What is left is
**4.78 GiB, 6.6 per cent of the peak**, inside the eigensolver fit's own error
bar. ``OPEN.md`` Part VII item 1 has the subtraction and the caveat on it.

The lesson for anyone extending this module is the one that generalises: **the
first thing to look for in a gap is a whole term that lives outside the sized
unit**, not a coefficient that is ten per cent off. An allocation this module
does not model is invisible; a coefficient that is wrong is merely inaccurate.

References for the conventions rather than the code: ``PW/src/setup.f90`` for
``nbnd``, ``Modules/recvec_subs.f90`` (``ggen``) for the sphere, and
``PW/src/n_plane_waves.f90`` for ``npwx``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace

import numpy as np

from defumat.basis.fftgrid import fft_grid_dimensions, gcut_from_ecut
from defumat.system.builder import System

__all__ = ["SizeEstimate", "estimate_size", "SpeedCheck", "speed_mode_fits",
           "SPEED_HEADROOM", "KBatchChoice", "choose_k_batch"]

#: Bytes in one complex number at the precision a run will use.
_COMPLEX_BYTES = {"double": 16, "single": 8}
_REAL_BYTES = {"double": 8, "single": 4}

#: Coefficients of the eigensolver's XLA temp buffer, in units of
#: ``nvecx npwx zc``, ``nbnd npwx zc`` and ``band_batch npol N_smooth zc``.
#: Fitted to ``memory_analysis().temp_size_in_bytes`` of the compiled
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
    #: Which projector storage this was sized for: ``store`` holds
    #: ``(nk, npwx, nkb)``, ``rebuild`` holds the core and forms each
    #: k-point's on demand. It changes the largest single line in
    #: :attr:`arrays` on a many-k run, so the report names it.
    projectors: str = "store"
    #: ``name -> bytes`` for each array whose size the basis fixes.
    arrays: dict = field(default_factory=dict)
    #: The compiled eigensolver's single contiguous XLA temp buffer, estimated
    #: from the fit in the module docstring, times ``k_batch``. It is **not** a
    #: member of ``arrays``: it supersedes the two Davidson entries there rather
    #: than adding to them, which is what :attr:`peak_bytes` does with it -- so
    #: it must scale with the k-batch exactly as those two do, or the peak falls
    #: when the batch grows.
    eigensolver_buffer: int = 0
    #: The largest single transient ``Calculation.__init__`` allocates before
    #: the SCF starts -- the augmentation charge's ``(ngm, kkbeta)`` Bessel
    #: intermediate. It is gone by the time the eigensolver runs, so it
    #: **bounds** the peak against the eigensolver's buffer rather than adding
    #: to it; on a vacuum-padded slab it is the larger of the two.
    setup_transient: int = 0
    #: Where the wavefunction set lives between the points that read it --
    #: :func:`~defumat.batching.resolve_wfc_store`'s answer. ``stream`` holds it
    #: in host RAM and puts one chunk on the device per call, so the
    #: ``wavefunctions`` line is that chunk rather than the whole set.
    wfc_store: str = "device"
    #: **The start** -- ``wfcinit``: ``max(natomwfc, nbnd)`` vectors, ``H`` and
    #: ``S`` applied to them, and their band loop through the grid, at
    #: :attr:`band_batch`. It runs once, before the first Davidson call, and is
    #: a moment rather than a standing cost, so it enters :attr:`peak_bytes`
    #: beside the eigensolver's buffer (`MEMORY-AUDIT.md` D10, where it set the
    #: peak of the one cell measured).
    start_buffer: int = 0
    #: How many vectors the start rotates: ``max(natomwfc, nbnd)``, with
    #: ``natomwfc`` counted the way ``wfcinit`` counts it for this regime.
    start_vectors: int = 0
    #: The band-independent parts of the two band-looped stages, and what one
    #: band in flight adds to either (``_FFT_COEFFICIENT npol N_smooth zc``,
    #: times the k-points in flight). Kept so that :meth:`at_band_batch` can
    #: re-evaluate the estimate at another band batch without rebuilding the
    #: basis, which is what :func:`choose_band_batch` walks.
    eigensolver_fixed: float = 0.0
    start_fixed: float = 0.0
    band_box_bytes: float = 0.0

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
        # Setup's transient, the start and the eigensolver's buffer never
        # coexist -- each is freed before the next is asked for -- so the peak
        # takes the largest of them, not their sum.
        return int(resident + max(self.eigensolver_buffer, self.setup_transient,
                                  self.start_buffer))

    def at_band_batch(self, band_batch: int | None) -> "SizeEstimate":
        """The same estimate with ``band_batch`` bands in flight instead.

        Only the two band-looped stages move -- the eigensolver's grid line and
        the start's -- and both are ``fixed + boxes x band_box_bytes``, so this
        is arithmetic on stored numbers and builds nothing.
        """
        return replace(
            self,
            band_batch=None if band_batch is None else int(band_batch),
            eigensolver_buffer=int(self.eigensolver_fixed + _boxes_in_flight(
                band_batch, self.nbnd) * self.band_box_bytes),
            start_buffer=int(self.start_fixed + _boxes_in_flight(
                band_batch, self.start_vectors) * self.band_box_bytes)
            if self.start_vectors else 0,
        )

    def at_k_batch(self, k_batch: int | None) -> "SizeEstimate":
        """The same estimate with ``k_batch`` k-points in flight instead.

        ``None`` is the whole mesh. The lines that move are the ones
        :func:`_k_live_arrays` names, the eigensolver's two parts and the
        start's, each linear in the k-points in flight and each evaluated by
        the expression :func:`estimate_size` evaluates, so the result equals a
        fresh estimate at that chunk field by field. Nothing is counted again:
        this is what lets :func:`choose_k_batch` bisect on one estimate.
        """
        k_live = self.nk if k_batch is None else min(k_batch, self.nk)
        nvecx = self.davidson_basis * self.nbnd
        ndim = self.npwx * self.npol
        zc = _COMPLEX_BYTES.get(self.precision, 16)
        in_flight = _k_live_arrays(k_live, self.nbnd, nvecx, ndim, self.npwx,
                                   self.nkb, zc)
        band_box_bytes, eigensolver_fixed = _k_live_buffers(
            k_live, self.nbnd, nvecx, ndim, self.npol, self.smooth_points, zc)
        start_fixed = _k_live_start(k_live, self.nk, self.start_vectors, ndim, zc,
                                    self.wfc_store)
        return replace(
            self,
            k_batch=k_live,
            # By name, so the lines this estimate's dials left out stay out and
            # the order the report breaks ties in is kept.
            arrays={name: in_flight.get(name, size) for name, size in self.arrays.items()},
            eigensolver_fixed=float(eigensolver_fixed),
            start_fixed=float(start_fixed),
            band_box_bytes=float(band_box_bytes),
        ).at_band_batch(self.band_batch)

    @property
    def dense_points(self) -> int:
        return int(np.prod(self.dense_grid))

    @property
    def smooth_points(self) -> int:
        return int(np.prod(self.smooth_grid))

    def report(self) -> str:
        """A human-readable table -- what the CLI prints."""
        def gb(n):
            # **GiB, and it says so.** This divides by 2^30 and said "GB",
            # which is the unit a card's own specification is *not* in: an
            # H200 is 143.8 GB and 133.9 GiB, and a report read against the
            # wrong one of those is out by 7.4 per cent in the direction that
            # says a run fits. `peak_bytes` is bytes; only this formatter
            # chose a unit.
            return f"{n / 2**30:9.2f} GiB"

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
            f"{self.band_batch if self.band_batch is not None else 'all'}, "
            f"projectors = {self.projectors})",
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
            f"  {'setup transient (augmentation)':<34s}{gb(self.setup_transient)}",
            "        the (ngm, kkbeta) Bessel intermediate of the augmentation",
            "        charge, freed before the SCF starts. It bounds the peak",
            "        against the line below rather than adding to it.",
            f"  {'start (wfcinit)':<34s}{gb(self.start_buffer)}",
            f"        {self.start_vectors} vectors -- max(natomwfc, nbnd) -- with",
            "        H and S applied and their grid loop at the band batch. Runs",
            "        once, before the first Davidson call.",
            f"  {'eigensolver XLA temp buffer':<34s}{gb(self.eigensolver_buffer)}",
            "        one contiguous allocation, and it stands in for the two",
            "        Davidson lines above rather than adding to them. Estimated",
            "        from a fit measured on the CPU backend and checked on an",
            "        H200: within 3.1% on 12 of 14 points, 30% HIGH on the two",
            "        at david 2 / band_batch 64 -- but 20% LOW at david 2 /",
            "        band_batch 16 on a 45-atom spinor PAW slab on an A100.",
            "        At david 2 the error goes BOTH ways and the low direction",
            "        is the one that kills a run. Close to the card? Measure it",
            "        with tools/gpu/davidson_memory.py.",
            f"  {'PEAK (resident + the larger)':<34s}{gb(self.peak_bytes)}",
        ]
        return "\n".join(lines)


def estimate_size(
    system: System,
    pseudos,
    nbnd: int | None = None,
    k_batch: int | None = None,
    davidson_basis: int | None = None,
    band_batch: int | None | str = "default",
    projectors: str | None = "default",
    wfc_store: str | None = "default",
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
        wfc_store: where the wavefunction set lives, as
            :func:`~defumat.batching.resolve_wfc_store` resolves it. ``stream``
            (memory mode on an accelerator) keeps it in host RAM, so the device
            holds one chunk of it; sizing the whole set there would count a
            store that is not on the card.

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
    # ``ncs`` is ``nkb``'s per-*dataset* counterpart: what ``ProjectorCore``
    # holds, and what the ``rebuild`` route pays instead of ``vkb``. **A dataset
    # and not a species label**: two labels naming one UPF file -- an
    # antiferromagnet's ``Fe1``/``Fe2``, or a noncollinear texture written one
    # species per site -- share one block of columns (P110), and summed over
    # labels this line reported that block once per label for a run that holds
    # it once. The key is ``build_projector_core``'s own function rather than a
    # restatement of it, so the two cannot disagree about what "the same
    # dataset" is. Like the build, it runs over every species declared, whether
    # or not an atom carries it, and is empty when no atom has a projector
    # (the build's ``nkb == 0`` branch).
    from defumat.pseudo.projectors import _projector_dataset_key

    ncs, projector_datasets = 0, set()
    for pseudo in (pseudos if nkb else ()):
        key = _projector_dataset_key(pseudo)
        if key not in projector_datasets:
            projector_datasets.add(key)
            ncs += len(projector_channels(pseudo))

    from defumat.solvers.davidson import DAVID_NDIM

    if davidson_basis is None:
        davidson_basis = DAVID_NDIM
    if isinstance(band_batch, str):
        from defumat.batching import resolve_band_batch

        band_batch = resolve_band_batch(band_batch)
    # Resolved here rather than taken as given, for the reason ``D1`` gives
    # about the other dials: the estimate has to describe the run that will
    # happen, and this one is resolved from the environment when it is not
    # named.
    from defumat.batching import resolve_projectors

    projectors = resolve_projectors(projectors)
    from defumat.batching import resolve_wfc_store

    wfc_store = resolve_wfc_store(wfc_store)
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
    # Every line that holds ``k_live`` k-points' worth, written once for this
    # function and :meth:`SizeEstimate.at_k_batch`.
    in_flight = _k_live_arrays(k_live, nbnd, nvecx, ndim, npwx, nkb, zc)

    arrays = {
        # **Streamed, the set is not on the device**: one chunk of it is, put
        # there per Davidson call from the host store (``scf/streaming.py``),
        # one channel at a time.
        **({"wavefunctions (nspin,nk,nbnd,ndim)": wf_spin * nk * nbnd * ndim * zc}
           if wfc_store != "stream" else
           {_STREAMED_CHUNK: in_flight[_STREAMED_CHUNK]}),
        # **The core is resident under BOTH routes**, and putting it inside the
        # conditional below was this model's second wrong turn about the same
        # object. ``Calculation.__init__`` assigns ``projector_core``
        # unconditionally, *before* it resolves the dial with
        # ``resolve_projectors``, so ``columns`` and ``kg``
        # are a cost ``store`` pays too. Counting them as a cost of ``rebuild``
        # alone understates the stored floor and makes the modelled saving
        # ``vkb - columns - kg - chunk`` where the measured resident saving is
        # ``vkb`` flat. The slab's A/B settles it: the delta is 13.96 GB at six
        # brackets with 0.00 residual, and ``columns + kg`` there is 0.6-0.7 GB,
        # which would have shown. (A13 made the mirror-image error first, by
        # counting the core as a *new* cost of the rebuilt route.)
        # Real since ``GPU-MEMORY-NEXT.md`` item 6: the ``(-i)^l`` is kept per
        # column and applied on use.
        "projector core columns (nk,npwx,ncs)": nk * npwx * ncs * zr,
        "projector core kg (nk,npwx,3)": nk * npwx * 3 * zr,
        # **What the dial actually chooses**: the whole-k array, or one chunk
        # rebuilt from the core above and freed again. Nothing else moves.
        **({"projectors vkb (nk,npwx,nkb)": nk * npwx * nkb * zc}
           if projectors == "store" else
           {_REBUILT_CHUNK: in_flight[_REBUILT_CHUNK]}),
        # The subspace and the Ritz block, :func:`_k_live_arrays` says what each is.
        _SUBSPACE: in_flight[_SUBSPACE],
        _RITZ: in_flight[_RITZ],
        "density+potential (nspin_mag,ngm)": 2 * nspin_mag * ngm * zc,
        "fields on dense grid": 3 * nspin_mag * int(np.prod(dense_grid)) * zr,
    }
    if nkb:
        arrays[_BEC] = in_flight[_BEC]
    # **The per-k basis bookkeeping every Hamiltonian reads** (``GPU-MEMORY-NEXT.md``
    # item 24): ``|k+G|^2`` (real), the FFT index (int32) and the sphere's mask,
    # and the gamma trick's ``-(k+G)`` index where the half sphere is consumed.
    # Resident for the run in both memory modes, and one of the things that
    # still grows with the mesh in memory mode.
    arrays["per-k basis tables (nk,npwx)"] = (
        nk * npwx * (zr + 4 + 1 + (4 if gamma_only else 0)))
    # The density's symmetrisation: a permutation of the dense G set (int64)
    # and its translation phases, per operation of the run's group.
    if not system.nosym:
        nsym = system.symmetry_group().nsym
        if nsym > 1:
            arrays["symmetry maps (nsym,ngm)"] = nsym * ngm * (8 + zc)
    # DFT+U's projectors ``S|phi>``, the same layout as ``vkb`` with the
    # Hubbard manifold's columns: resident, and read by every ``h_psi``.
    from defumat.hubbard.manifold import build_hubbard_setup

    hubbard = build_hubbard_setup(system.hubbard, structure, pseudos,
                                  noncolin=bool(system.noncolin))
    if hubbard is not None:
        arrays["Hubbard projectors wfcU (nk,ndim,nwfcU)"] = (
            nk * ndim * hubbard.nwfcU * zc)
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

    # **Setup, which is where the three largest allocations on a slab live.**
    # ``Calculation.__init__`` builds the augmentation charge before the SCF
    # starts, and on a vacuum-padded cell it dwarfs everything below: this
    # module once reported 34.78 GB for a 45-atom NiBr2 slab whose measured
    # peak was 117.55 GB, and each of the terms here was larger than that
    # total. Counted per distinct **dataset** rather than per species, which is
    # what ``build_augmentation`` builds (a noncollinear texture is one species
    # per magnetic site, all of them naming one file).
    #
    # **Which of the two storage schemes is sized is decided the same way the
    # run decides it**, by calling the same budget on the same number. Sizing
    # the stored array for a run that will tabulate is this module's own error
    # inverted: a red light for a calculation that fits.
    setup_transient = 0
    from defumat.pseudo.augmentation import (
        AUG_CELL_FACTOR, AUG_DQ, _aug_chunk, _aug_max_bytes, _dataset_key,
    )

    ultrasoft = [pseudos[t] for t in sorted(set(structure.types))]
    ultrasoft = [p for p in ultrasoft if p.is_ultrasoft and projector_channels(p)]
    if ultrasoft:
        nl_all = 2 * max(p.lmax for p in pseudos) + 1
        seen, datasets = set(), []
        for pseudo in ultrasoft:
            nqlc = pseudo.augmentation.nqlc if pseudo.augmentation else nl_all
            nl_species = min(nl_all, nqlc)
            key = _dataset_key(pseudo, nl_species)
            if key in seen:
                continue
            seen.add(key)
            datasets.append((pseudo, nl_species, len(projector_channels(pseudo))))

        nh_max = max(nh for _, _, nh in datasets)
        # The route is decided on the complex table's size, as
        # ``build_augmentation`` decides it; what a stored route then holds is
        # the real ``R_ij(G)`` (``AugmentationCharge``), half of that.
        gate_bytes = sum(nh * nh * ngm * zc for _, _, nh in datasets)
        qgm_bytes = sum(nh * nh * ngm * zr for _, _, nh in datasets)
        if gate_bytes <= _aug_max_bytes():
            arrays["augmentation Q_ij(G) (nh,nh,ngm)"] = qgm_bytes
            arrays["augmentation phases (nat,ngm)"] = len(structure.types) * ngm * zc
            # ``_species_charge``'s intermediate, one species at a time, so the
            # peak is the largest species' atom count rather than ``nat``. It
            # used to be a second ``(nh, nh, ngm)`` -- the same size as the
            # stored table -- which is why this line did not exist and the
            # estimate was half the truth on an ultrasoft run.
            arrays["augmentation contraction (nat_t,ngm)"] = ngm * zc * max(
                sum(1 for t in structure.types if t == index)
                for index in sorted(set(structure.types))
                if pseudos[index].is_ultrasoft and projector_channels(pseudos[index])
            )
            # ``_qrad_kernel``'s ``(ngm, kkbeta)`` Bessel intermediate: built one
            # L at a time, so the peak is one dataset's rather than their sum.
            # Transient -- gone before the eigensolver runs, which is why it
            # bounds the peak rather than adding to it.
            setup_transient = max(
                ngm * pseudo.kkbeta * zr for pseudo, _, _ in datasets
            )
        else:
            # The tabulated scheme. ``nqx`` is QE's sizing with this code's
            # ``cell_factor``, and the Bessel intermediate shrinks with it --
            # the transform runs on the knots instead of on the G sphere, which
            # is the whole difference.
            nqx = int(AUG_CELL_FACTOR * np.sqrt(system.ecutrho) / AUG_DQ + 4)
            chunk = _aug_chunk(nh_max, ngm)
            npad = -(-ngm // chunk) * chunk
            arrays["augmentation table (nbeta,nbeta,nl,nqx)"] = sum(
                pseudo.nbeta**2 * nl * nqx * zr for pseudo, nl, _ in datasets
            )
            arrays["augmentation phases (nat,npad)"] = len(structure.types) * npad * zc
            arrays["augmentation G set (npad,3)"] = npad * 3 * zr + npad * zr
            # What one block of the rebuild holds. Since the scan contracts in
            # the radial basis (``augmentation._tabulated_charge``) it forms no
            # ``(nh, nh, chunk)`` block: what it holds is the radial table
            # ``(nbeta, nbeta, nl, chunk)`` and the harmonics. ``_aug_chunk`` is
            # still sized from ``nh``, so this is smaller than the dial assumes.
            arrays["augmentation rebuild block"] = chunk * max(
                pseudo.nbeta**2 * nl * zr + nl * nl * zr for pseudo, nl, _ in datasets
            )
            setup_transient = max(
                nqx * pseudo.kkbeta * zr for pseudo, _, _ in datasets
            )

    # **The PAW one-centre tables**, the ``becsum -> r^2 rho_lm`` maps of
    # :class:`~defumat.paw.onecenter.PawSpecies`. ``Calculation.__init__``
    # keeps them as ``self.paw`` and ``_paw_onecenter`` takes them as an
    # argument at every SCF iteration, so they are **resident** for the life of
    # the run and belong in the floor, not in the setup transient. What one
    # dataset holds is read off ``_build_species``. The density maps are
    # factored -- the Gaunt coefficients ``(nlm, nh, nh)``, the channel map
    # ``(nh, nbeta)`` and three radial pair tables, ``pfunc`` and ``pfunc_rel``
    # ``(nbeta, nbeta, mesh)`` and the pseudo one with its augmentation charge
    # ``(nbeta, nbeta, nlm, mesh)`` -- so
    #
    #     [nlm nh^2 + nh nbeta + nbeta^2 mesh (1 + nlm + [ae_wfc_rel])] zr
    #
    # and a meta-GGA adds ``kinetic_ae``/``kinetic_ps``, factored the same way
    # since 2026-09-29: one shared angular table ``(nlm, nh, nh)`` and two
    # ``(nh, mesh)`` radial factors per sphere, ``(nlm nh^2 + 4 nh mesh) zr``
    # where the formed maps were ``2 nh^2 nlm mesh zr``. ``nlm = (l_max_rho
    # + 1)^2`` from the header and ``mesh`` the whole radial mesh rather than
    # ``kkbeta``, because the energies are integrated to the end of it. Until
    # ``GPU-MEMORY-NEXT.md`` item 18 the density maps were products of the two
    # factors, ``nh^2 nlm mesh`` each and 295.2 MB apiece on platinum. Counted
    # per distinct dataset with ``build_paw``'s own key, for the reason ``ncs``
    # is (P110).
    #
    # Left out: each dataset's seven ``(mesh,)`` radial vectors, its angular
    # quadrature's tables, and the sphere workspace inside ``_paw_onecenter``,
    # an XLA temporary this module cannot see for the same reason it cannot
    # see the eigensolver's. The bytes are ``zr``, as on every real line here,
    # but ``_build_species`` builds through NumPy and does not read the
    # precision policy, so under ``single`` the tables it allocates are still
    # float64 and this line is half of them.
    from defumat.paw.onecenter import _lmax_rho, _paw_dataset_key

    onecentre_bytes, paw_datasets = 0, set()
    for pseudo in pseudos:
        if not pseudo.is_paw:
            continue
        key = _paw_dataset_key(pseudo)
        if key in paw_datasets:
            continue
        paw_datasets.add(key)
        nh = len(projector_channels(pseudo))
        nbeta = len(pseudo.projectors)
        # ``pfunc_rel``'s table is not built at ``soc_scale = 0`` (P117).
        relativistic = (
            pseudo.paw is not None and pseudo.paw.ae_wfc_rel is not None
            and system.soc_scale != 0.0
        )
        nlm = (_lmax_rho(pseudo) + 1) ** 2
        onecentre_bytes += (
            nlm * nh * nh + nh * nbeta
            + nbeta * nbeta * pseudo.mesh * (1 + nlm + int(relativistic))
            + int(functional.is_meta) * (nlm * nh * nh + 4 * nh * pseudo.mesh)
        ) * zr
    if onecentre_bytes:
        arrays["PAW one-centre tables"] = onecentre_bytes

    # The eigensolver's own XLA temp buffer -- see the module docstring. The
    # FFT term is on the **smooth** grid, which is the box ``h_psi`` transforms
    # a band in; ``band_batch = None`` is every band at once.
    #
    # **The FFT term carries ``npol``, and that factor is measured.** A spinor
    # band is two fields on that grid -- ``vloc_psi_nc`` transforms each
    # component -- so a band in flight is ``npol`` boxes, not one. The
    # coefficients were fitted on ``si16`` and checked on a gamma-storage slab,
    # both ``npol = 1``, so nothing here moves where they were measured. On
    # ``tests/data/qe/h-chain-90deg.in`` (``npol = 2``, ``nbnd = 24``, a
    # 40x40x64 smooth grid) ``memory_analysis().temp_size_in_bytes`` rises by
    # **6,553,600 bytes per band in flight**, flat over ``band_batch``
    # 1 -> 2 -> 4 -> 8, against a box of 40*40*64*16 = 1,638,400 -- so a spinor
    # band is **exactly 4.00 boxes**, which is this term's 2.00 times ``npol``.
    # The remainder at ``band_batch = 1``, 39.84 MB, is the two ``ndim`` terms'
    # 40.2 MB to one per cent, so the whole form transfers. (``band_batch = 16``
    # is off that ladder and is not a counter-example: 24 bands at 16 compile a
    # 16-block *and* an 8-tail, and the executable holds both.)
    #
    # **And it carries ``k_live``, for the same reason the two lines it
    # supersedes do.** The fit was taken one k-point at a time --
    # ``tools/gpu/davidson_memory.py`` defaults ``--k-batch`` to 1 -- so every
    # term above describes *one chunk*, and ``davidson_eigensolver_all``
    # ``vmap``s that chunk over ``k_live`` k-points at once. Without the factor
    # the buffer superseded two ``arrays`` lines that do carry it, so
    # :attr:`SizeEstimate.peak_bytes` *fell* as the batch grew: at ``nk = 8``,
    # ``k_batch = None``, ``nbnd = 64``, ``ndim = 10000``, ``nvecx = 256`` and a
    # 45^3 smooth grid it removed 983 MB of floor and put 319 MB back, reporting
    # 664 MB **below** the floor it discarded and a smaller peak at the end of
    # the dial that actually holds ``nk`` subspaces. That is this module's own
    # stated error inverted -- a green light for the larger calculation.
    #
    # **A band batch that does not divide the band count costs its tail too**:
    # ``map_bands`` compiles the full blocks and the remainder separately, and
    # the executable holds both (the ``band_batch = 16`` point above, 24 bands
    # as a 16-block and an 8-tail) -- so the boxes in flight are ``b + n % b``,
    # :func:`_boxes_in_flight`.
    #
    # The terms that carry ``k_live`` are :func:`_k_live_buffers`'s, so that
    # :meth:`SizeEstimate.at_k_batch` evaluates the same expressions.
    band_box_bytes, eigensolver_fixed = _k_live_buffers(
        k_live, nbnd, nvecx, ndim, npol, int(np.prod(smooth_grid)), zc)
    eigensolver_buffer = int(
        eigensolver_fixed + _boxes_in_flight(band_batch, nbnd) * band_box_bytes
    )

    # **The start** (`MEMORY-AUDIT.md` D10). ``starting_wavefunctions`` builds
    # the atomic span -- ``natomwfc`` vectors, topped up with random ones to
    # ``nbnd`` -- and ``rayleigh_ritz`` applies ``H`` and ``S`` to it, so three
    # blocks of ``max(natomwfc, nbnd)`` vectors, the span itself whole-k unless
    # the store streams (``stream_start`` builds it chunk by chunk). Three is
    # D10's count and is within 5 per cent of the one measured rise it names
    # (45.93 GB of blocks, plus the grid line at the band batch that run set,
    # against 51.05). The grid line is ``h_psi``'s, so it takes the
    # eigensolver's coefficient; that transfer is an assumption, not a fit.
    from defumat.pseudo.atomic import (
        count_atomic_wavefunctions, count_spinor_wavefunctions,
    )

    if (npol == 2 and system.lspinorb and not system.spiral
            and any(p.has_so for p in pseudos)):
        natomwfc = count_spinor_wavefunctions(pseudos, structure, lspinorb=True)
    else:
        natomwfc = npol * count_atomic_wavefunctions(pseudos, structure)
    start_vectors = max(int(natomwfc), int(nbnd))
    start_fixed = _k_live_start(k_live, nk, start_vectors, ndim, zc, wfc_store)
    start_buffer = int(
        start_fixed + _boxes_in_flight(band_batch, start_vectors) * band_box_bytes
    )

    return SizeEstimate(
        nat=len(structure.types), nsp=len(pseudos), nelec=nelec, nbnd=int(nbnd),
        nspin=nspin, npol=npol, nspin_mag=nspin_mag, nk=nk,
        ngm=ngm, ngms=ngms, npwx=npwx, npw=tuple(int(n) for n in npw), nkb=nkb,
        dense_grid=tuple(int(n) for n in dense_grid),
        smooth_grid=tuple(int(n) for n in smooth_grid),
        gamma_requested=gamma_requested, gamma_only=gamma_only,
        doublegrid=doublegrid, precision=name, projectors=projectors,
        davidson_basis=int(davidson_basis), k_batch=k_live,
        band_batch=None if band_batch is None else int(band_batch),
        arrays=arrays, eigensolver_buffer=eigensolver_buffer,
        setup_transient=setup_transient, wfc_store=wfc_store,
        start_buffer=start_buffer, start_vectors=start_vectors,
        eigensolver_fixed=float(eigensolver_fixed),
        start_fixed=float(start_fixed), band_box_bytes=float(band_box_bytes),
    )


#: The names of the ``arrays`` lines that hold ``k_live`` k-points' worth.
_STREAMED_CHUNK = "wavefunctions, one streamed chunk (k,nbnd,ndim)"
_REBUILT_CHUNK = "projectors rebuilt, one chunk (npwx,nkb)"
_SUBSPACE = "Davidson subspace psi+hpsi"
_RITZ = "Davidson Ritz block"
_BEC = "Davidson becp+becq (nvecx,nkb)"


def _k_live_arrays(k_live, nbnd, nvecx, ndim, npwx, nkb, zc) -> dict:
    """Every ``arrays`` line that holds ``k_live`` k-points' worth, by name.

    The caller keeps the ones its dials put in the estimate: the streamed chunk
    only when the store streams, the rebuilt projectors only when they are
    rebuilt, the two ``bec`` blocks only when there are projectors.
    """
    return {
        _STREAMED_CHUNK: k_live * nbnd * ndim * zc,
        _REBUILT_CHUNK: k_live * npwx * nkb * zc,
        # ``psi`` and ``hpsi``, both ``(nvecx, ndim)`` -- the subspace and H
        # applied to it. ``S|psi>`` is deliberately not stored (the Ritz
        # vector's projections are a rotation of ``becq``), which is why this
        # is two and not three.
        _SUBSPACE: 2 * k_live * nvecx * ndim * zc,
        # ``evc``, ``hevc``, ``sevc`` and ``residual``, each ``(nbnd, ndim)``,
        # live at once inside ``solve``.
        _RITZ: 4 * k_live * nbnd * ndim * zc,
        _BEC: 2 * k_live * nvecx * nkb * zc,
    }


def _k_live_buffers(k_live, nbnd, nvecx, ndim, npol, smooth_points, zc):
    """``(band_box_bytes, eigensolver_fixed)`` at ``k_live`` k-points in flight.

    The eigensolver buffer's two parts, :func:`estimate_size`'s comment above
    their use says where each comes from.
    """
    band_box_bytes = k_live * _FFT_COEFFICIENT * npol * smooth_points * zc
    eigensolver_fixed = k_live * (
        _SUBSPACE_COEFFICIENT * nvecx * ndim * zc
        + _RITZ_COEFFICIENT * nbnd * ndim * zc
    )
    return band_box_bytes, eigensolver_fixed


def _k_live_start(k_live, nk, start_vectors, ndim, zc, wfc_store):
    """The start's band-independent part: the span whole-k unless the store streams."""
    span_k = k_live if wfc_store == "stream" else nk
    return (span_k + 2 * k_live) * start_vectors * ndim * zc


def _boxes_in_flight(band_batch: int | None, n: int) -> int:
    """How many band boxes a loop over ``n`` bands in blocks of ``band_batch`` holds.

    ``None`` (or a batch of ``n`` or more) is every band at once. Otherwise the
    full blocks and the remainder are compiled separately and the executable
    holds both, so a batch that does not divide ``n`` pays its tail as well.
    """
    if band_batch is None or band_batch >= n:
        return int(n)
    band_batch = max(1, int(band_batch))
    return band_batch + n % band_batch


#: The fraction of the device's free memory a ``speed``-mode estimate may fill
#: before :func:`speed_mode_fits` says it does not fit.
#:
#: **It is a calibration and it is this module's floor showing.**
#: :attr:`SizeEstimate.peak_bytes` is a floor by construction, and measured
#: against ``peak_bytes_in_use`` on a GTX 1060 (jax 0.11.1, eight-atom silicon
#: at 20 Ry, ``nosym``, the whole k-set in flight) the card used **1.57-1.61x**
#: the estimate at 8, 27 and 64 k-points -- 261.3 MB against 166.4, 876.8
#: against 557.5, 2133.3 against 1323.8. ``1 / 1.61 = 0.62``, so a threshold at
#: 0.6 falls back when the *measured* peak would pass the whole free pool. The
#: module docstring's H200 and A100 sweeps put the eigensolver-buffer fit
#: within 3 per cent on most shapes and 20 per cent low on one, so on those
#: cards 0.6 is conservative -- which is the direction that matters, since the
#: cost of a wrong ``fits`` is a run that dies in its first Davidson call and
#: the cost of a wrong ``does not fit`` is a run that is about 20 per cent
#: slower.
SPEED_HEADROOM = 0.6


@dataclass(frozen=True)
class SpeedCheck:
    """Whether a ``speed``-mode run fits the device, and the two numbers behind it."""

    fits: bool
    #: :attr:`SizeEstimate.peak_bytes` for the whole k-set in flight, bytes.
    estimate: int
    #: ``bytes_limit - bytes_in_use`` on the device, bytes; ``None`` where the
    #: client reports no statistics (the CPU client returns ``None``), in
    #: which case there is nothing to fit and :attr:`fits` is ``True``.
    available: int | None
    headroom: float = SPEED_HEADROOM

    def describe(self) -> str:
        if self.available is None:
            return "no device memory statistics on this backend"
        return (f"estimated peak {self.estimate / 2**30:.2f} GiB against "
                f"{self.headroom:.0%} of {self.available / 2**30:.2f} GiB free")


def speed_mode_fits(system, pseudos, nbnd: int | None = None,
                    davidson_basis: int | None = None,
                    headroom: float = SPEED_HEADROOM) -> SpeedCheck:
    """Would ``memory_mode = 'speed'`` fit on the default device?

    Sizes the run with every k-point and every band in flight and the
    projectors stored -- :func:`~defumat.batching.memory_preset`'s ``speed`` on
    an accelerator -- and compares :attr:`SizeEstimate.peak_bytes` with
    ``headroom`` times what the allocator has left. Nothing is allocated; the
    only device call is ``memory_stats()``, whose keys are indexed rather than
    ``.get`` so that an unpopulated one raises instead of reading as a pass.
    """
    import jax

    stats = jax.local_devices()[0].memory_stats()
    if not stats:
        return SpeedCheck(fits=True, estimate=0, available=None, headroom=headroom)
    available = int(stats["bytes_limit"]) - int(stats["bytes_in_use"])
    estimate = estimate_size(
        system, pseudos, nbnd=nbnd, k_batch=None, davidson_basis=davidson_basis,
        band_batch=None, projectors="store", wfc_store="device",
    ).peak_bytes
    return SpeedCheck(fits=estimate <= headroom * available, estimate=int(estimate),
                      available=available, headroom=headroom)


@dataclass(frozen=True)
class BandBatchChoice:
    """The band batch memory mode runs at on this device, and why."""

    #: ``None`` for the whole block, which is what memory mode runs whenever
    #: it fits -- a band loop on a card is slower (4.3x at one band on the
    #: eight-atom silicon cell on a GTX 1060, measured while the stick fill was
    #: still a loop on a card; 71.0 against 64.0 ms an iteration on sixteen
    #: atoms on an RTX A2000 since), so the dial moves only when it has to.
    band_batch: int | None
    #: Whether the estimate at :attr:`band_batch` fits. ``False`` only when not
    #: even one band at a time does, in which case :attr:`band_batch` is 1 and
    #: the run is expected to die in its first large allocation.
    fits: bool
    #: :attr:`SizeEstimate.peak_bytes` at :attr:`band_batch`, bytes.
    estimate: int
    #: ``bytes_limit - bytes_in_use``, bytes; ``None`` without device
    #: statistics (the CPU client), where nothing is chosen.
    available: int | None
    headroom: float = SPEED_HEADROOM

    def describe(self) -> str:
        if self.available is None:
            return "no device memory statistics on this backend"
        return (f"estimated peak {self.estimate / 2**30:.2f} GiB against "
                f"{self.headroom:.0%} of {self.available / 2**30:.2f} GiB free")


@dataclass(frozen=True)
class KBatchChoice:
    """The k-chunk a ``k_batch = 'fit'`` run takes on this device, and why."""

    #: ``None`` for the whole mesh in one call, else the chunk size; 1 when no
    #: larger chunk fits.
    k_batch: int | None
    #: Whether the estimate at :attr:`k_batch` fits. ``False`` only when not even
    #: one k-point a call does.
    fits: bool
    #: :attr:`SizeEstimate.peak_bytes` at :attr:`k_batch`, bytes.
    estimate: int
    #: ``bytes_limit - bytes_in_use``, bytes; ``None`` without device statistics
    #: (the CPU client), where nothing is chosen.
    available: int | None
    headroom: float = SPEED_HEADROOM

    def describe(self) -> str:
        if self.available is None:
            return "no device memory statistics on this backend"
        return (f"estimated peak {self.estimate / 2**30:.2f} GiB against "
                f"{self.headroom:.0%} of {self.available / 2**30:.2f} GiB free")


def choose_k_batch(system, pseudos, nbnd: int | None = None,
                   davidson_basis: int | None = None,
                   band_batch: int | None = None, projectors: str = "rebuild",
                   wfc_store: str = "stream",
                   headroom: float = SPEED_HEADROOM,
                   available: int | None = None) -> KBatchChoice:
    """The largest k-chunk whose estimated peak fits the device.

    **Why a chunk and not one k-point a call.** One k-point per call is QE's
    ``k_loop`` and it is what memory mode does; on a card it pays a fixed cost
    per call -- one dispatch per kernel of a Davidson solve sized for one small
    k-point -- that batching over k amortises. Measured on an RTX A2000 on
    eight-atom silicon at 20 Ry with 27 k-points, Davidson steps equal: 559 ms
    per SCF iteration at one k-point a call, 414 at 8 and 323 at the whole mesh,
    against 308 in speed mode; neither the width ladder's host round trips nor
    XLA's command buffers were the cost (``PERFORMANCE.md``, "Memory mode on a
    k-mesh"). So the chunk is as large as the card allows, sized from
    :func:`estimate_size` at the run's other dials -- which matched the measured
    peaks to 3 per cent at one, 8 and 27 k-points a call -- against
    :data:`SPEED_HEADROOM` of what the allocator has left, as
    :func:`choose_band_batch` sizes the band block.

    **The whole mesh wins whenever it fits.** Otherwise the choice minimises the
    number of calls and then the padding: :func:`~defumat.batching.k_chunks`
    pads the last chunk with repeats that are solved and discarded, so 22 is
    cheaper than 25 on a 64-point mesh (three calls each, 2 against 11 padded
    solves). A chunk larger than one is a ``vmap`` over k, under which the
    Davidson loop runs until its slowest k-point has converged; that is a pure
    speed lever only while the steps are equal across k-points, which they are
    on every cell measured since the subspace solve stopped setting its own
    error (``PERFORMANCE.md``, "The endgame on a card is a stall").

    ``band_batch`` is the band dial the chunk is sized at: the whole block
    (``None``) is the only one a chunk is grown for, since a run whose block does
    not fit at one k-point has nothing to spare. ``available`` replaces the
    device query, for a test or a planned run on a card that is not this one.
    """
    if available is None:
        import jax

        stats = jax.local_devices()[0].memory_stats()
        if not stats:
            return KBatchChoice(k_batch=1, fits=True, estimate=0,
                                available=None, headroom=headroom)
        available = int(stats["bytes_limit"]) - int(stats["bytes_in_use"])
    budget = headroom * available
    # One estimate, re-evaluated at every chunk the bisection tries
    # (:meth:`SizeEstimate.at_k_batch`, arithmetic only): it used to be a whole
    # :func:`estimate_size` per step, each counting both spheres and ``npw`` at
    # every k-point again for terms that are linear in the chunk.
    base = estimate_size(
        system, pseudos, nbnd=nbnd, k_batch=None, davidson_basis=davidson_basis,
        band_batch=band_batch, projectors=projectors, wfc_store=wfc_store,
    )

    def peak(chunk):
        return base.at_k_batch(chunk).peak_bytes

    whole = base.peak_bytes
    if whole <= budget:
        return KBatchChoice(k_batch=None, fits=True, estimate=int(whole),
                            available=int(available), headroom=headroom)
    nk = base.nk
    # The peak grows with the chunk, so the largest that fits is found by
    # bisection, and among the chunks no larger that need the same number of
    # calls the one with the least padding is ceil(nk / calls).
    low, high = 1, max(1, nk - 1)
    if peak(low) > budget:
        return KBatchChoice(k_batch=1, fits=False, estimate=int(peak(1)),
                            available=int(available), headroom=headroom)
    while low < high:
        middle = (low + high + 1) // 2
        if peak(middle) <= budget:
            low = middle
        else:
            high = middle - 1
    calls = -(-nk // low)
    chunk = -(-nk // calls)
    return KBatchChoice(k_batch=chunk, fits=True, estimate=int(peak(chunk)),
                        available=int(available), headroom=headroom)


def choose_band_batch(system, pseudos, nbnd: int | None = None,
                      davidson_basis: int | None = None,
                      k_batch: int | None = 1, projectors: str = "rebuild",
                      wfc_store: str = "stream",
                      headroom: float = SPEED_HEADROOM,
                      available: int | None = None) -> BandBatchChoice:
    """The largest band batch whose estimated peak fits the device.

    Memory mode bounds what grows with the k-mesh; on a large cell what is left
    is one k-point's working set, and the largest term of that is the band
    loop through the grid -- ``2 b npol N_smooth`` complex numbers at ``b``
    bands in flight, 66.9 GB at every band against 2.65 GB at 16 on the 45-atom
    NiBr2 slab (``GPU-MEMORY-NEXT.md`` item 9). This sizes the run once at the
    memory preset's other dials and re-evaluates it at each ``b``
    (:meth:`SizeEstimate.at_band_batch`, arithmetic only).

    **The whole block wins whenever it fits**, so a cell that ran before runs
    exactly as before. Otherwise the choice minimises the number of blocks --
    the time -- and, among batches with the same number, the boxes in flight,
    which prefers a batch that divides the band count (a remainder costs its
    own block, :func:`_boxes_in_flight`). The threshold is
    :data:`SPEED_HEADROOM` of what the allocator has left, the calibration
    :func:`speed_mode_fits` uses, and for the same reason.

    ``available`` replaces the device query, for a test or a planned run on a
    card that is not this one.
    """
    if available is None:
        import jax

        stats = jax.local_devices()[0].memory_stats()
        if not stats:
            return BandBatchChoice(band_batch=None, fits=True, estimate=0,
                                   available=None, headroom=headroom)
        available = int(stats["bytes_limit"]) - int(stats["bytes_in_use"])
    base = estimate_size(
        system, pseudos, nbnd=nbnd, k_batch=k_batch,
        davidson_basis=davidson_basis, band_batch=None, projectors=projectors,
        wfc_store=wfc_store,
    )
    budget = headroom * available
    if base.peak_bytes <= budget:
        return BandBatchChoice(band_batch=None, fits=True,
                               estimate=int(base.peak_bytes),
                               available=int(available), headroom=headroom)
    best = None
    for b in range(base.nbnd, 0, -1):
        estimate = base.at_band_batch(b)
        if estimate.peak_bytes > budget:
            continue
        key = (-(-base.nbnd // b), _boxes_in_flight(b, base.nbnd))
        if best is None or key < best[0]:
            best = (key, b, estimate)
    if best is None:
        return BandBatchChoice(band_batch=1, fits=False,
                               estimate=int(base.at_band_batch(1).peak_bytes),
                               available=int(available), headroom=headroom)
    return BandBatchChoice(band_batch=best[1], fits=True,
                           estimate=int(best[2].peak_bytes),
                           available=int(available), headroom=headroom)
