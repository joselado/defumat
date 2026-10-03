"""Density mixing: turning a fixed-point iteration into a convergent one.

Feeding the output density straight back in (``rho_in = rho_out``) diverges for
anything but the smallest systems -- the charge sloshes between regions of the
cell, amplified each iteration by the Hartree term. Mixing damps that.

Two schemes, behind a registry (rule R4):

* **linear**: ``rho_in + beta (rho_out - rho_in)``. Robust, slow, and the right
  thing to check a new system with.
* **anderson**: extrapolate from the history of residuals to the density whose
  residual has the smallest norm in the span. This is Pulay/DIIS, and is what
  makes convergence take ten iterations instead of a hundred.

QE's default is Broyden mixing (``mix_rho.f90``), which is closely related;
Anderson is chosen here because it is the same idea with far less bookkeeping.

**Both of them are quasi-Newton methods on the SCF residual**, and saying so is
the calibration for everything built on top: Anderson fits a secant Jacobian to
the residual history and takes the Newton step inside its span. What decides how
well that works is the *conditioning* of the true Jacobian, and that is what the
third scheme addresses:

* **kerker**: Anderson, with the scalar ``beta`` replaced by the
  Thomas-Fermi-screened operator ``beta |G|^2 / (|G|^2 + q_TF^2)`` acting on the
  residual in G-space. This is QE's ``mixing_mode = 'TF'``
  (``mix_rho.f90``'s ``approx_screening``) and Kerker's 1981 preconditioner, and
  it is an *approximation to the inverse Jacobian*: in a metal the dielectric
  function diverges as ``q^-2`` at long wavelength, so the unpreconditioned
  iteration amplifies long-wavelength charge transfer. Dividing that out costs
  one FFT per iteration and is worth 24 iterations to 14 on the aluminium slab
  of ``benchmarks/al-slab.in``.

  **Where it is worth reaching for is narrower than the textbook story**, and
  the measurement is in ``PERFORMANCE.md``: a *homogeneous* metal in a long cell
  is not the problem it is usually said to be here -- a sixteen-atom aluminium
  cell 30 bohr long converges in five Anderson iterations, because an eight-deep
  residual history already spans the few badly-conditioned directions. What does
  hurt is **inhomogeneous** screening, a metal beside vacuum, and there Kerker's
  assumption that one ``q_TF`` describes the whole cell is only partly right: it
  wins by 24-to-14 with 16 bohr of vacuum, by 34-to-20 with 32, and has lost its
  advantage by 64 (36 against 35). QE's answer to that is ``local-TF``
  (``approx_screening2``, :func:`local_tf_preconditioner`), which makes the
  screening length a function of ``rho(r)`` so that the metal and the vacuum get
  different values of it in the same cell; ``scf/solvers.py`` is the other route,
  an exact Jacobian that makes no such assumption and costs far more.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from defumat.basis.fft import g_to_r_gamma
from defumat.basis.gvectors import GVectors, _half_sphere
from defumat.units import E2, FPI

__all__ = ["Mixer", "LinearMixer", "AndersonMixer", "AdaptiveMixer", "get_mixer",
           "MIXERS", "PRECONDITIONED", "kerker_preconditioner",
           "local_tf_preconditioner", "thomas_fermi_screening", "SphereLayout",
           "MIXING_SPACES", "DEFAULT_MIXING_SPACE", "resolve_mixing_space",
           "kerker_preconditioner_g", "local_tf_preconditioner_g"]


class Mixer:
    """Interface: given the input and output density, propose the next input."""

    #: Applied to the residual in place of multiplying by ``beta``. ``None`` is
    #: the plain scalar. The driver installs one for a preconditioned mixer,
    #: because building it needs the G-vectors and the mixer does not have them.
    precondition = None

    #: Whether :attr:`precondition` means anything for this mixer. False for a
    #: scheme whose step is not one scalar times the residual, so that an
    #: install site can refuse **at setup** rather than at the first mix -- a
    #: run that cannot work should not find out three hours in.
    accepts_precondition = True

    #: A separate step length for the **magnetization**, VASP's ``AMIX_MAG``.
    #: ``None`` is ``pw.x``'s own rule, one ``alphamix`` for every component of
    #: ``mix_type``, and is the default. See :meth:`magnetic_step`.
    beta_mag = None

    #: ``(nspin_mag, n1, n2, n3)``, needed only when :attr:`beta_mag` is set,
    #: because the residual reaches :meth:`step` as a flat vector and which part
    #: of it is the magnetization is not recoverable from the vector alone. The
    #: driver installs it beside the preconditioner.
    shape = None

    #: ``(drho, dns, dtau) -> F`` with ``F(a) . F(b)`` the ``rho_ddot`` of two
    #: residuals, installed by the driver beside the preconditioner because it
    #: needs the G-vectors (:func:`~defumat.scf.driver._rho_ddot_metric`).
    #: ``None`` keeps the flat inner product over the packed vector. Only a mixer
    #: that fits coefficients reads it; the driver's ``_mix`` evaluates it and
    #: hands the vector to :meth:`mix` as ``fit``.
    metric = None

    #: Where the density part of the packed vector lives: ``None`` is the whole
    #: dense box in real space, and a :class:`SphereLayout` is ``pw.x``'s
    #: ``mix_type``, the smooth sphere in G with the shell above it mixed
    #: linearly and never stored. Installed by the driver beside the
    #: preconditioner, for the same reason: it needs the G-vectors. The driver's
    #: ``_mix`` reads it; the mixer itself only ever sees a flat vector.
    layout = None

    #: Whether :attr:`layout` may be a :class:`SphereLayout`. False for a scheme
    #: that is pointwise in real space by construction, where the components of
    #: a vector of Fourier coefficients are not the components it was designed
    #: to adapt.
    accepts_layout = True

    def mix(self, rho_in: np.ndarray, rho_out: np.ndarray, exclude: slice | None = None,
            fit: np.ndarray | None = None) -> np.ndarray:
        """``exclude`` is a part of the packed vector that is mixed but not *fitted*.

        ``fit`` is this residual in the inner product the fit should use
        (:attr:`metric`), in place of the packed vector itself. Both matter only
        to a mixer that fits coefficients to its history
        (:class:`AndersonMixer`); the others take them and ignore them, so the
        driver can pass them without knowing which mixer it holds.
        """
        raise NotImplementedError

    def step(self, residual: np.ndarray, density: np.ndarray | None = None) -> np.ndarray:
        """``beta * r``, or the preconditioner's version of it.

        ``density`` is the input density this residual belongs to. Kerker
        ignores it -- its screening length is a property of the *cell* -- and
        ``local-TF`` does not: its screening is a function of ``rho(r)``, so it
        is rebuilt at every iteration. That is the whole difference between the
        two (``approx_screening`` against ``approx_screening2``), and it is why
        the argument is here rather than closed over when the mixer is built.
        """
        if self.precondition is None:
            stepped = self.beta * residual
        else:
            stepped = self.precondition(residual, density)
        return self.magnetic_step(stepped)

    def magnetic_step(self, stepped: np.ndarray) -> np.ndarray:
        """Rescale the magnetization's part of an already-taken step.

        **Why this is one place rather than four.** Every path here gives the
        magnetization a plain ``beta * r``: the unpreconditioned step does it by
        construction, and both preconditioners do it deliberately -- Kerker
        screens the *charge* alone, because the Thomas-Fermi ``q^-2`` divergence
        is a property of the charge response and the magnetization has none, so
        its own branches read ``beta * head[c]``. So multiplying the magnetic
        components of the *output* by ``beta_mag/beta`` turns every one of them
        into ``beta_mag * r`` and nothing else moves.

        **What it is for**, and it is a departure from ``pw.x`` rather than a
        correction to it. QE uses one ``alphamix`` for every component of
        ``mix_type``; VASP exposes ``AMIX_MAG`` and Elk gives the magnetic
        channel its own control, because one scalar is known not to serve both.
        The reason is the spectrum rather than taste: the charge's slow direction
        is the long-wavelength Hartree one, which a preconditioner compresses,
        and the magnetization's is the **Stoner** enhancement
        ``chi_0/(1 - I chi_0)``, which is local, has no ``1/q^2`` for Kerker to
        divide out, and amplifies a uniform change at every wavelength equally.
        ``PLAN.md`` P102 is what says that matters here: on
        ``fe-noncolin-pbe-stress.in`` **28 of 43 iterations are magnetism**,
        against 4 of 25 on the collinear benchmark, and the longitudinal residual
        plateaus while the charge keeps falling.

        **The rotation is not optional at ``nspin = 2``.** A collinear density is
        carried as ``(up, down)`` and not as ``(charge, magnetization)``, so
        scaling channel 1 would scale *down* rather than the moment, which is a
        different operator that also changes the charge. The pair is rotated,
        scaled and rotated back, exactly as Kerker does around its screening. At
        ``nspin_mag = 4`` channel 0 already is the charge and no rotation is
        needed.

        The tail beyond the density -- ``becsum``, ``ns``, ``tau`` -- is left at
        ``beta``. Giving it the magnetic step would be a second departure with no
        measurement behind it, and ``becsum``'s own residual is now reported
        (:func:`~defumat.scf.residual_split.becsum_residual`) so that whether it
        needs one can be asked with a number.
        """
        if self.beta_mag is None or self.shape is None:
            return stepped
        shape = tuple(self.shape)
        nspin = shape[0]
        if nspin == 1:
            return stepped
        ratio = float(self.beta_mag) / float(self.beta)
        stepped = np.array(stepped, copy=True)
        size = int(np.prod(shape))
        head = stepped[:size].reshape(shape)
        _scale_magnetization(head, ratio)
        stepped[:size] = head.reshape(-1)
        return stepped

    def rotate_history(self, transform, residual_transform=None) -> None:
        """Apply ``transform`` to every stored density and residual, or forget them.

        ``residual_transform`` is the map for the residuals when it differs from
        the densities', which it does for an affine map ``x -> R x + c``: a
        residual is a difference of two densities and sees ``R`` alone.

        The in-loop rotation of ``ORIENTATION-NEXT.md`` Route C turns the input
        density between iterations, and a history left in the old frame is
        extrapolated back towards it. A mixer that can carry its history across
        a common linear map overrides this; the default is to reset, which costs
        iterations and nothing else.
        """
        self.reset()

    def reset(self) -> None:
        pass


def _scale_magnetization(head: np.ndarray, ratio: float) -> None:
    """Multiply the magnetization of ``head`` by ``ratio`` in place, the charge kept.

    ``head`` is ``(nspin, ...)``: at ``nspin = 2`` the ``(up, down)`` pair is
    turned into ``(charge, moment)``, scaled and turned back, and at four the
    channels after the first are the moment already. Linear and channel by
    channel, so it is the same operator on a real-space field and on its
    Fourier coefficients, which is why the step of the shell
    (:meth:`SphereLayout.shell_step`) shares it with :meth:`Mixer.magnetic_step`.
    """
    if head.shape[0] == 2:
        charge, moment = head[0] + head[1], head[0] - head[1]
        moment = ratio * moment
        head[0], head[1] = 0.5 * (charge + moment), 0.5 * (charge - moment)
    else:
        head[1:] *= ratio


@dataclass
class LinearMixer(Mixer):
    beta: float = 0.7

    def mix(self, rho_in, rho_out, exclude=None, fit=None):
        return rho_in + self.step(rho_out - rho_in, rho_in)


def _contiguous_part(exclude, size: int) -> slice | None:
    """The fitted part of a packed vector as one slice, or ``None`` if it is two.

    The same entries ``fitted[exclude] = False`` leaves set, read off the slice
    rather than off the mask: nothing excluded, or an excluded block at either
    end, leaves one contiguous stretch, and an empty fit falls back to the whole
    vector as :meth:`AndersonMixer.mix` does. A block in the middle, a strided
    slice or anything that is not a slice gives ``None``, and the caller keeps
    the masked copy.
    """
    if exclude is None:
        return slice(0, size)
    if not isinstance(exclude, slice):
        return None
    start, stop, step = exclude.indices(size)
    if step != 1:
        return None
    if stop <= start or (start == 0 and stop >= size):
        return slice(0, size)
    if stop >= size:
        return slice(0, start)
    if start == 0:
        return slice(stop, size)
    return None


def _same_fit(cached, fitted) -> bool:
    """Whether a cached Gram matrix was built in the inner product now asked for."""
    if isinstance(cached, str) or isinstance(fitted, str):
        return isinstance(cached, str) and isinstance(fitted, str) and cached == fitted
    return np.array_equal(cached, fitted)


@dataclass
class AndersonMixer(Mixer):
    """Anderson/Pulay mixing with a bounded history."""

    beta: float = 0.7
    history: int = 8
    #: Above this, the newest-first history is trimmed rather than solved. Four
    #: orders above the worst conditioning measured on any cell here (1.7e8, on
    #: sixty-four atoms at ``ecutwfc = 30``), so it does not fire on a run that
    #: was already working.
    condition_limit: float = 1.0e12
    _densities: list = field(default_factory=list, repr=False)
    _residuals: list = field(default_factory=list, repr=False)
    #: ``r_i . r_j`` and ``|r_i|`` over the fitted part of every entry of
    #: ``_residuals``, in the same order: a cache, extended by one row and
    #: column per call rather than rebuilt (:meth:`_extend_gram`). Arrays, so a
    #: checkpoint carries them beside the history they describe.
    _gram: np.ndarray | None = field(default=None, repr=False)
    _norms: np.ndarray | None = field(default=None, repr=False)
    #: Which entries of the packed vector the cache was built over. A different
    #: ``exclude`` is a different inner product, and it rebuilds the cache. The
    #: string ``"metric"`` when it was built over :attr:`_fits` instead.
    _fit_mask: np.ndarray | str | None = field(default=None, repr=False)
    #: Each entry of ``_residuals`` in the fit's inner product (``mix``'s
    #: ``fit``), rolled and reset with it. Empty when no metric is installed.
    #: **This is the memory the metric costs**: a fit vector holds the dense
    #: sphere's complex coefficients as reals, about ``pi/6`` of the box twice
    #: over, so it is the size of a residual to within five per cent and the
    #: history's resident set doubles (``OPEN.md`` S2 sizes it at 3.18 GB on the
    #: 157-atom slab without them). Kept rather than recomputed because each
    #: one is an FFT of a whole residual, and the new row of the Gram matrix
    #: needs all of them every call; ``history`` is the dial.
    _fits: list = field(default_factory=list, repr=False)

    #: Private, and only for the test that pins the cache to what it replaced:
    #: ``False`` rebuilds the whole Gram matrix on every call, as before
    #: ``OPEN.md`` M3. A class attribute rather than a field, so ``get_mixer``
    #: does not take it and a checkpoint does not carry it (unless it is set on
    #: an instance, which then writes it like any other bool in ``vars``).
    _cache_gram = True

    def reset(self):
        self._densities.clear()
        self._residuals.clear()
        self._fits.clear()
        self._drop_gram()

    def rotate_history(self, transform, residual_transform=None) -> None:
        """Turn the stored densities and residuals with the input density.

        ``residual_transform`` maps the residuals when the densities' map is
        affine rather than linear (Anderson's step is equivariant under a common
        shift of every stored density, since its coefficients sum to one).

        ``transform`` is a spin rotation of every magnetization in a packed
        vector, one map for the whole history. On the flat fit that rotation is
        an isometry, ``|R m|`` being ``|m|`` point by point, so the cached Gram
        matrix of the residuals is unchanged and stays valid. A metric's fit
        vectors are in that metric's own layout, which this does not unpack, so
        a history carrying them is dropped instead.
        """
        if self._fits:
            self.reset()
            return
        residual_transform = residual_transform or transform
        self._densities[:] = [transform(v) for v in self._densities]
        self._residuals[:] = [residual_transform(v) for v in self._residuals]

    def _drop_gram(self):
        self._gram = self._norms = self._fit_mask = None

    def mix(self, rho_in, rho_out, exclude=None, fit=None):
        """One Anderson step; ``exclude`` is left out of the fit and still mixed.

        **With ``fit``, the fit is in that inner product and ``exclude`` is
        moot.** ``fit`` is this residual as :attr:`Mixer.metric` maps it, whose
        dot products are ``rho_ddot``'s: the Hartree energy of the charge, the
        flat magnetization and ``tau`` terms, ``U/2`` on ``ns`` and nothing on
        ``becsum``. That is ``pw.x``'s inner product (``mix_rho.f90:403-425``),
        and the fit differs from ``pw.x``'s in three ways that are stated rather
        than hidden: without a :class:`SphereLayout` it runs over the whole
        dense set, where ``pw.x`` fits only ``G < ngms`` and mixes the rest
        linearly (``mix_rho.f90:132``, ``scf_mod.f90:549-552``), which the
        layout reproduces; at ``nspin = 2`` the ``tau`` weight is four
        times ``tauk_ddot``'s, deliberately (:func:`~defumat.scf.potential.tau_accuracy`);
        and the combination is Anderson's rather than modified Broyden's. The
        paragraphs below describe the flat path, which is what runs when no
        metric is installed.

        **What ``exclude`` is for.** The driver passes the ``becsum`` block, and
        that is ``pw.x``'s rule rather than a choice: ``rho_ddot``
        (``scf_mod.f90:718``) is the only inner product ``mix_rho`` fits its
        ``betamix`` in, and it reads the density, ``ns`` and ``tau`` but never
        ``becsum``. For an ultrasoft run QE does not carry ``becsum`` in
        ``mix_type`` at all, and for PAW it carries it and fits without it,
        ``paw_ddot`` being commented out because it is not positive definite.

        **Why it is not cosmetic.** The Gram matrix here is flat, and
        ``becsum``'s entries are not in the density's units, so nothing fixes
        how much say each block gets. Measured on ``benchmarks/fe-mag-1k.in``
        (ultrasoft iron): ``becsum`` is **98 to 99.99 per cent** of the squared
        residual at every iteration, so the fit was to the projector
        occupations and the density rode along. Against ``pw.x``'s own optimum
        over the same history, in ``pw.x``'s metric, those coefficients leave a
        residual **5.2 times** larger at the median iteration and 32 times at
        the worst; the density block alone, fitted flat, is within 1.3 of it.
        That is the flat-against-``1/G^2`` difference item F in
        ``MAGNETISM-NEXT.md`` suspected, measured to be the smaller of the two
        by a factor of four at the median. See ``PLAN.md`` P107.

        The excluded block is still combined with the same coefficients, which
        is what keeps it consistent with the density it belongs to (the
        driver's ``_mix`` docstring gives the reason).
        """
        rho_in = np.asarray(rho_in).ravel()
        residual = np.asarray(rho_out).ravel() - rho_in
        segment = None
        if fit is not None:
            if len(self._fits) != len(self._residuals):
                # A history written without fit vectors -- a checkpoint from
                # before the metric, or a mixer driven both ways -- cannot be
                # fitted in this inner product, so it restarts here.
                self.reset()
            fitted = "metric"
        else:
            fitted = np.ones(residual.size, dtype=bool)
            if exclude is not None:
                fitted[exclude] = False
                if not fitted.any():
                    fitted[:] = True
            # The mask is still built, because it is what the cached Gram
            # matrix is keyed on (``_fit_mask``, and a checkpoint carries it);
            # what is no longer made from it is a copy of every entry.
            segment = _contiguous_part(exclude, residual.size)

        self._densities.append(rho_in)
        self._residuals.append(residual)
        if fit is not None:
            self._fits.append(np.asarray(fit).ravel())
        if len(self._densities) > self.history:
            self._densities.pop(0)
            self._residuals.pop(0)
            if self._fits:
                self._fits.pop(0)
            if self._gram is not None and self._norms is not None:
                # The cache rolls with the history it describes: the entry that
                # left is the oldest, so its row and column are the leading ones.
                self._gram, self._norms = self._gram[1:, 1:], self._norms[1:]

        # Every entry's fitted part is read on every call, because the new row
        # of the Gram matrix needs all of them; what is no longer recomputed is
        # the rest of the matrix. Extended before the ``n == 1`` return, which
        # costs one dot there and keeps the cache the size of the history after
        # every call. With a metric the fit vectors are stored.
        #
        # **A view where the fitted part is one block, a copy only where it is
        # two.** Nothing excluded, or ``becsum`` as the tail of the packed
        # vector (every run without ``ns`` or ``tau``), leaves one contiguous
        # stretch, and slicing it is free where the mask made a whole copy of
        # every entry on every call: eight vectors at the default depth, each
        # the size of the dense-grid state. The dot of a view and of a copy of
        # the same contiguous numbers is one call to the same BLAS routine over
        # the same values, so no number moves (measured: MKL's ``ddot`` gives
        # the same bits at every 8-byte offset of either operand). ``becsum``
        # between the density and ``ns`` or ``tau`` keeps the masked copy,
        # since joining two stretches would split the dot into two sums.
        if fit is not None:
            fit = self._fits
        elif segment is not None:
            fit = [r[segment] for r in self._residuals]
        else:
            fit = [r[fitted] for r in self._residuals]
        gram, norms = self._extend_gram(fit, fitted)

        n = len(self._residuals)
        if n == 1:
            return (rho_in + self.step(residual, rho_in)).reshape(
                np.asarray(rho_out).shape
            )

        # Minimise |sum_i c_i r_i| subject to sum_i c_i = 1, by solving the
        # constrained least-squares problem in the residual basis -- **in the
        # basis of unit-norm residuals**, which is the whole of what keeps this
        # solvable near convergence.
        #
        # The Gram matrix ``r_i . r_j`` built from the raw residuals spans the
        # square of their magnitudes, and over a converging history those
        # magnitudes cover many orders: on the 16-atom cell at ``ecutwfc = 30``
        # they run 5e-1 down to 5e-5 by the eighth iteration, so the bordered
        # system's condition number reaches **1.1e11 and grows about two orders
        # per iteration**. ``np.linalg.solve`` does not raise on that -- it
        # raises only on an exactly singular matrix -- so past about 1e16 it
        # returns coefficients that are silently garbage, the mixed density
        # explodes, and the run ends in ``NaN`` having reported nothing wrong.
        # That is not hypothetical: it is what 64 atoms at ``ecutwfc = 30`` did
        # on a GPU, converging happily to ``conv_thr = 1e-8`` and dying on the
        # way to 1e-10 (`PERFORMANCE.md`).
        #
        # Writing ``c_i = e_i / |r_i|`` makes the Gram matrix unit-diagonal, so
        # its conditioning reflects only how *aligned* the residuals are and no
        # longer how their sizes differ. Measured on the same histories:
        # **1.1e11 -> 2.7e4**, with coefficients identical to every digit. The
        # substitution is exact, so this changes no converged result; it changes
        # which ones are reachable.
        #
        # The check is over every norm and not only the new one: at ``n == 1``
        # nothing is checked, so a zero first entry is caught here, at ``n = 2``.
        if not np.all(norms > 0.0):
            self.reset()
            return (rho_in + self.step(residual, rho_in)).reshape(
                np.asarray(rho_out).shape
            )

        # Normalising removes the conditioning that came from the residuals'
        # *spread*; it cannot remove what comes from their *alignment*, and that
        # grows with the cell -- the same measurement gives 2.7e4 on sixteen
        # atoms and 1.7e8 on sixty-four. So the oldest entries are dropped until
        # what is left is solvable, which is standard DIIS practice and is a
        # bound rather than a hope. The cap sits four orders above the worst
        # value ever measured here, so it never fires on anything already
        # working, and it keeps the coefficients' relative error near 1e-4 in
        # the regime where it does. The trimming is per solve: the history, and
        # the cached Gram matrix beside it, keep every entry.
        keep = n
        while keep > 1:
            trimmed = self._build_overlap(gram, norms, keep)
            if np.linalg.cond(trimmed) < self.condition_limit:
                break
            keep -= 1
        overlap = self._build_overlap(gram, norms, keep)
        used = slice(n - keep, n)

        rhs = np.zeros(keep + 1)
        rhs[keep] = 1.0
        try:
            coefficients = np.linalg.solve(overlap, rhs)[:keep] / norms[used]
        except np.linalg.LinAlgError:
            # A degenerate history means the residuals are linearly dependent;
            # drop it and take a plain linear step rather than failing.
            self.reset()
            return (rho_in + self.step(residual, rho_in)).reshape(
                np.asarray(rho_out).shape
            )

        # The belt to the braces above. Normalising fixes the conditioning that
        # comes from the *spread* of the residuals; it cannot fix a history whose
        # members have become genuinely parallel, and a solve that survived
        # ``LinAlgError`` can still return non-finite coefficients. Falling back
        # to a linear step costs one slow iteration; not checking costs the run.
        if not np.all(np.isfinite(coefficients)):
            self.reset()
            return (rho_in + self.step(residual, rho_in)).reshape(
                np.asarray(rho_out).shape
            )

        # **Combine first, precondition the combination once.** That is
        # ``mix_rho.f90``'s order -- its comment reads "preconditioning the new
        # search direction", and ``approx_screening``/``approx_screening2`` are
        # applied to the *mixed* residual with the *mixed* density, after the
        # Broyden combination and before ``alphamix`` scales the step.
        #
        # For a linear preconditioner the two orders are identical: ``P`` comes
        # out of the sum. For ``local-TF`` they are not, because ``P`` depends
        # on the density it is built at -- and preconditioning each history
        # entry separately would also run its Krylov solve once per entry
        # instead of once per iteration, which is up to eight times the cost.
        #
        # **Accumulated in place, in the order the sums always had.** Written
        # as ``total = total + c * d`` the loop made a whole vector per term
        # per sum even with numpy's temporary elision, 16 at the default depth,
        # each a fresh mapping the size of the dense-grid state. Here there are
        # three buffers for the whole call, and every element is the same
        # floating-point operation as before: each product ``c * d`` in full,
        # then added to the running total, term by term in history order. The
        # first term is ``0.0 + c * d`` as it always was, the product and then
        # an add of zero, which is not a no-op on a negative zero. The buffers
        # take the dtype ``c * d`` promotes to, float64 for a float32 density
        # too (NEP 50: the coefficient is a float64 scalar), and they are
        # allocated per call rather than kept, because the returned array is
        # handed to ``jnp.asarray``, which on a CPU may share its memory.
        densities, residuals = self._densities[used], self._residuals[used]
        dtype = np.result_type(coefficients.dtype, *(d.dtype for d in densities),
                               *(r.dtype for r in residuals))
        mixed_density = np.empty(rho_in.size, dtype=dtype)
        mixed_residual = np.empty(rho_in.size, dtype=dtype)
        term = np.empty(rho_in.size, dtype=dtype) if keep > 1 else None
        for index, (c, d, r) in enumerate(zip(coefficients, densities, residuals)):
            if index == 0:
                np.add(np.multiply(c, d, out=mixed_density), 0.0, out=mixed_density)
                np.add(np.multiply(c, r, out=mixed_residual), 0.0, out=mixed_residual)
            else:
                np.add(mixed_density, np.multiply(c, d, out=term), out=mixed_density)
                np.add(mixed_residual, np.multiply(c, r, out=term), out=mixed_residual)
        del term
        mixed = np.add(mixed_density, self.step(mixed_residual, mixed_density),
                       out=mixed_density)
        return mixed.reshape(np.asarray(rho_out).shape)

    def _extend_gram(self, fit, fitted):
        """``(gram, norms)`` over ``fit``, from the cache and the one entry that is new.

        **What is kept, and why.** ``r_i . r_j`` between two entries of the
        history does not change while both are in it, and each call adds one
        entry and drops at most one, so rebuilding the matrix is ``n^2 + n``
        host dots per SCF iteration of which ``n`` are new: 72 against 8 at the
        default depth of 8, each a pass over ``nspin x n_dense`` doubles. On the
        NiBr2 grid ``PERFORMANCE.md`` sizes P74 against, one residual is 83 MB,
        so the matrix streamed 10.6 GB of host memory per iteration and its one
        new row streams 1.3 GB (``OPEN.md`` M3, arithmetic rather than a
        timing). The fitted parts are views where they are one contiguous block
        and copies, one per entry per call, only where ``exclude`` sits in the
        middle of the packed vector (:meth:`mix`). ``pw.x`` rebuilds too, over the upper
        triangle (``mix_rho.f90:403-425``, ``betamix(i,j) = rho_ddot(df(j),
        df(i))`` and then ``betamix(j,i) = betamix(i,j)``), so this departs from
        it in cost and not in arithmetic.

        **Why no number moves.** Every entry is a value the rebuild computed:
        ``float(fit[i] @ fit[j])`` with ``i`` the older entry, which is the old
        upper triangle, and the lower triangle filled by copying it rather than
        by computing ``fit[j] @ fit[i]``. The diagonal is one dot, and the norm
        is the square root of that same dot, as the old ``float(np.sqrt(r @ r))``
        was. The norm is kept beside the matrix rather than read off its
        diagonal, because under a float32 policy ``np.sqrt`` of the float32 dot
        and of its float64 copy differ in the last bit. What the cache does
        change is *when* a pair is computed, once, on the fitted part read in
        the call it arrived in; that the old code's recomputation on a fresh
        copy gave the same bits assumes the BLAS returns one dot of two vectors
        the same wherever they sit in memory, which is also what makes a view
        and a copy interchangeable here, and which was measured for MKL's
        ``ddot`` at every 8-byte offset of either operand.

        **When it is rebuilt from scratch**, which is exactly when it would
        otherwise describe something other than ``fit``: after :meth:`reset`,
        which the three restart-on-failure paths in :meth:`mix` go through; when
        the fitted mask differs from the one it was built over, since that is a
        different inner product; and when its size is not one short of the
        history, which is what a checkpoint written before the cache existed
        restores (a full ``_residuals`` beside ``_gram = None``) and what a
        partly failed restore leaves. :meth:`_build_overlap`'s trimming needs
        nothing, because it drops entries from one solve and never from the
        history.
        """
        n = len(fit)
        if not self._cache_gram:
            self._drop_gram()
            norms = np.array([float(np.sqrt(r @ r)) for r in fit])
            gram = np.array([[float(a @ b) for b in fit] for a in fit])
            return gram, norms

        cached = 0
        if (self._gram is not None and self._norms is not None
                and self._fit_mask is not None
                and np.shape(self._gram) == (n - 1, n - 1)
                and np.shape(self._norms) == (n - 1,)
                and _same_fit(self._fit_mask, fitted)):
            cached = n - 1
        gram = np.empty((n, n))
        norms = np.empty(n)
        if cached:
            gram[:cached, :cached] = self._gram
            norms[:cached] = self._norms
        for j in range(cached, n):
            for i in range(j):
                gram[i, j] = gram[j, i] = float(fit[i] @ fit[j])
            square = fit[j] @ fit[j]
            gram[j, j] = float(square)
            norms[j] = float(np.sqrt(square))
        self._gram, self._norms, self._fit_mask = gram, norms, fitted
        return gram, norms

    @staticmethod
    def _build_overlap(gram, norms, keep):
        """The bordered system over the newest ``keep`` residuals, normalised.

        ``c_i = e_i / |r_i|`` makes the Gram block unit-diagonal; the constraint
        ``sum_i c_i = 1`` becomes ``sum_i e_i / |r_i| = 1``, which is the border.
        """
        gram = gram[-keep:, -keep:]
        norms = norms[-keep:]
        overlap = np.empty((keep + 1, keep + 1))
        overlap[:keep, :keep] = gram / np.outer(norms, norms)
        overlap[:keep, keep] = 1.0 / norms
        overlap[keep, :keep] = 1.0 / norms
        overlap[keep, keep] = 0.0
        return overlap


@dataclass
class AdaptiveMixer(Mixer):
    """Elk's ``mixadapt``: one ``beta`` per component, steered by the residual's sign.

    ``src/mixadapt.f90``, transcribed. Every other mixer here answers the badly
    conditioned directions of the SCF map by *modelling* them -- Anderson fits a
    secant Jacobian to the residual history, Kerker divides out the ``1/q^2`` the
    Hartree kernel puts in. This one models nothing and watches instead: for each
    component ``j`` of the mixed vector it keeps its own step length, and

        beta_j <- min(beta_j + beta_0, beta_max)     if r_j did not change sign
        beta_j <- (beta_j + beta_0) / 2              if it did

    with the step then taken as ``rho_j + beta_j r_j``. A component that keeps
    being pushed the same way is one the iteration is crawling along, so its step
    grows until it is moving; a component that overshoots and comes back is one
    the step was too long for, so it is cut. **That makes it a stall detector by
    construction rather than by a threshold**, which is why it is worth having
    beside Anderson rather than instead of it: the directions it is for are the
    ones where the residual is *small and persistent*, and a secant fit built from
    small residuals has nothing to extrapolate from.

    The magnetic directions of a noncollinear cell are exactly of that shape.
    Turning every moment in the cell together costs no energy without spin-orbit
    coupling, so the residual has no component along that rotation at all and the
    manifold has to be traversed rather than descended; twisting them slowly costs
    only the spin-wave energy ``D q^2``, which vanishes as the cell grows.
    ``MAGNETISM-NEXT.md`` item F2 is the full account of which directions those
    are and what else is on offer for them.

    **The whole packed vector adapts, one step length per entry**, so ``becsum``
    and ``ns`` get their own adaptation rather than the density's. That is worth
    saying because it arrives for free and closes a real blind spot: ``becsum`` is
    mixed at the plain ``beta`` by every other mixer here and appears in no
    convergence measure at all (``OPEN.md`` Y2), and on a PAW magnet the moment
    lives in the d-shell ``becsum``.

    Three differences from the Fortran, all of them deliberate:

    * **Elk mixes the potential and this mixes the density.** The rule is
      pointwise and scale-free, so it transfers, but the two are not the same
      iteration and no Elk iteration count carries over.
    * **Elk's ``mu`` is not kept.** It is Elk's own record of the previous input,
      and this driver hands the input in, which is the same array whenever nothing
      touched the density between iterations and the right one when something did.
    * **``d`` is not returned.** Elk's RMS of the residual is not what this code
      measures self-consistency with; that is ``rho_ddot`` (:func:`scf_accuracy`).

    ``beta`` is Elk's ``beta0`` and **is not a step length**: it is the increment,
    the starting value and the floor the halving relaxes towards, all three, so
    Elk's default of 0.05 is small on purpose and ``mixing_beta = 0.7`` would
    saturate at ``beta_max`` in one iteration and leave plain linear mixing at 0.9
    wearing this mixer's name. That is what the warning below is for, and it is
    why the entry points default ``mixing_beta`` to ``None`` and let each mixer
    supply its own.

    Not compatible with a preconditioner, and it raises rather than ignoring one:
    the step is pointwise in real space with its own factor per point, and a
    G-space operator carrying a single scalar does not compose with that.
    """

    accepts_precondition = False
    #: Pointwise in real space, as Elk's ``mixadapt`` is: a step length per grid
    #: point, steered by the sign of that point's residual. The sign of the real
    #: or imaginary part of a Fourier coefficient is not that, so this mixer
    #: keeps the density on the box and refuses ``mixing_space = 'g'``.
    accepts_layout = False

    #: Elk's ``beta0``: the increment, the initial value and the floor at once.
    beta: float = 0.05
    #: Elk's ``betamax``, and Elk's own bound on it.
    beta_max: float = 1.0
    _betas: np.ndarray | None = field(default=None, repr=False)
    #: Elk's ``f``: the previous residual, whose sign against this one is the rule.
    _previous: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        # ``readinput.f90:699-708``'s two checks, with Elk's own messages behind
        # them. A negative increment would shrink the step every iteration and a
        # ``beta_max`` above one would take a step past the output density.
        if self.beta < 0.0:
            raise ValueError(
                f"beta = {self.beta} is negative; for mixing_mode = 'adaptive' it "
                f"is Elk's beta0, an increment added to every component's step "
                f"each iteration, and Elk requires it to be at least zero"
            )
        if not 0.0 <= self.beta_max <= 1.0:
            raise ValueError(
                f"beta_max = {self.beta_max} is outside [0, 1], which is Elk's "
                f"own bound (readinput.f90:705): a step longer than the residual "
                f"overshoots the output density it is aiming at"
            )
        if self.beta > 0.5 * self.beta_max:
            warnings.warn(
                f"mixing_beta = {self.beta:g} under mixing_mode = 'adaptive' is "
                f"Elk's beta0, which is an increment and a floor rather than a "
                f"step length, so this saturates at beta_max = "
                f"{self.beta_max:g} within two iterations and leaves plain linear "
                f"mixing under an adaptive name. Elk's default is 0.05, and "
                f"leaving mixing_beta unset gives it",
                RuntimeWarning, stacklevel=3,
            )

    def reset(self):
        self._betas = None
        self._previous = None

    def mix(self, rho_in, rho_out, exclude=None, fit=None):
        if self.precondition is not None:
            raise ValueError(
                "mixing_mode = 'adaptive' cannot take a preconditioner: it holds "
                "one step length per component of the packed vector and applies "
                "it pointwise in real space, where a Kerker or local-TF operator "
                "is one scalar acting in G-space. Choose one or the other"
            )
        shape = np.asarray(rho_out).shape
        # **The packed vector's own dtype is kept**, not forced to float64. It
        # comes from ``config.dtypes`` through the density, and a mixer that
        # upcast it would hand the next iteration a float64 density under a
        # float32 policy, which is the hardcoded-dtype rule. The test is
        # ``test_a_mixer_does_not_promote_the_densitys_precision``, and it is on
        # the **density** block deliberately: ``_mix`` casts the ``ns`` block
        # back to its own real type, so an assertion there passes whatever the
        # mixer did.
        rho_in = np.asarray(rho_in).ravel()
        rho_out = np.asarray(rho_out).ravel()
        if np.iscomplexobj(rho_in) or np.iscomplexobj(rho_out):
            # **Refused, because numpy would not refuse it.** The rule below is
            # ``residual * previous >= 0``, and numpy compares complex numbers
            # with ``>=`` rather than raising -- it orders them on the real part
            # and breaks ties on the imaginary one -- so a complex vector would
            # adapt every step length on the real part alone and report nothing.
            # ``_mix`` packs a spinor ``ns`` as a real view precisely so that
            # this never happens from inside the driver.
            raise TypeError(
                "the adaptive mixer's state is one real vector: its step length "
                "per component is chosen by the sign of that component's "
                "residual, and a complex number has no sign. Pack a complex "
                "block as a real view first, which is what scf/driver.py's _mix "
                "does for a spinor ns"
            )
        residual = rho_out - rho_in
        if self._betas is None or self._betas.shape != residual.shape:
            # ``iscl < 1``: Elk seeds every component at ``beta0`` with a zero
            # previous residual, and a zero compares ``>= 0`` against anything, so
            # the first mixing step already increments and runs at ``2 beta0``.
            # Transcribed rather than smoothed -- it is why the scheme leaves the
            # ground at all from a start as small as 0.05.
            self._betas = np.full(residual.shape, self.beta, dtype=residual.dtype)
            self._previous = np.zeros_like(residual)
        self._betas = np.where(
            residual * self._previous >= 0.0,
            np.minimum(self._betas + self.beta, self.beta_max),
            0.5 * (self._betas + self.beta),
        )
        self._previous = residual
        # ``beta nu + (1 - beta) mu`` and not the algebraically equal
        # ``mu + beta (nu - mu)``: the two differ in the last bit, and writing
        # the Fortran's form is what lets the test against it assert equality
        # rather than a tolerance. A tolerance would pass on a transcription
        # that had drifted, which is the whole thing that test is for.
        return (self._betas * rho_out + (1.0 - self._betas) * rho_in).reshape(shape)


#: Name -> mixer, as written in an input file's ``mixing_mode``.
MIXERS = {
    "linear": LinearMixer,
    "plain": AndersonMixer,  # QE's 'plain' is Broyden; Anderson is the stand-in
    "anderson": AndersonMixer,
    "broyden": AndersonMixer,
    # QE's names. 'kerker' and 'tf' screen with one length for the whole cell
    # (``approx_screening``); 'local-TF' makes it a function of rho(r)
    # (``approx_screening2``), which is what a slab needs.
    "kerker": AndersonMixer,
    "tf": AndersonMixer,
    "local-tf": AndersonMixer,
    # QE's own default name, so an unedited pw.x input reaches a mixer here.
    "default": AndersonMixer,
    # **Elk's, and deliberately not aliased to any QE name.** ``pw.x`` has no
    # adaptive mode, so giving this one of QE's names would make a pw.x input
    # mean something pw.x does not do.
    "adaptive": AdaptiveMixer,
}

#: Mixing modes whose ``beta`` is an operator, so the driver has to build it.
PRECONDITIONED = {"kerker", "tf", "local-tf"}

#: Of those, the ones whose operator depends on the density and is therefore
#: rebuilt at every iteration rather than once per run.
DENSITY_DEPENDENT = {"local-tf"}

def thomas_fermi_screening(volume: float, nelec: float) -> float:
    """``q_TF^2`` in 1/bohr^2, from ``mix_rho.f90``'s ``approx_screening``.

    **QE derives this from the system and does not fix it**, and copying that is
    not pedantry -- a hand-picked screening length is wrong by a factor of two
    on a cell of a different density, and over-screening is worse than not
    preconditioning at all. The Fortran is

        rs   = (3 omega / 4 pi / nelec)^(1/3)
        agg0 = (12/pi)^(2/3) / tpiba2 / rs

    with ``gg`` in units of ``tpiba2``; multiplying through by ``tpiba2`` leaves
    ``q_TF^2 = (12/pi)^(2/3) / rs`` in 1/bohr^2, which is what is returned here
    because this code carries ``|G|^2`` in 1/bohr^2 rather than in QE's units.

    ``rs`` is the mean valence-electron spacing over the **whole** cell, so a
    slab's vacuum enters it: more vacuum means a larger ``rs``, a smaller
    ``q_TF``, and less screening. That is the right direction and not enough --
    the screening is still uniform, and the metal and the vacuum want different
    values of it in the same cell. QE's answer to that is ``local-TF``
    (:func:`local_tf_preconditioner`, ``approx_screening2``), which makes the
    screening length a function of ``rho(r)``.
    """
    rs = (3.0 * volume / (4.0 * np.pi * nelec)) ** (1.0 / 3.0)
    return (12.0 / np.pi) ** (2.0 / 3.0) / rs


def kerker_preconditioner(gvectors, cell, shape, beta=0.7, screening=None, nelec=None):
    """``beta |G|^2 / (|G|^2 + q_TF^2)`` on the density part of a packed state.

    ``screening`` is ``q_TF^2`` in 1/bohr^2 and defaults to
    :func:`thomas_fermi_screening` of the cell, which is QE's choice.

    ``shape`` is the density's own shape; anything past it in the flat vector is
    ``becsum`` (for DFT+U, ``ns``; for a meta-GGA, ``tau``) and gets the plain
    scalar ``beta``. Mixing them with two different factors is consistent
    because the preconditioner is an approximate inverse Jacobian, not a step
    length -- the parts of the state whose Jacobian block is already well
    conditioned want no preconditioning. ``becsum`` and ``ns`` live on the atoms
    rather than on the grid, so Kerker has nothing to say about them; ``tau``
    does live on the grid, but the ``q^-2`` divergence Kerker cancels is a
    property of the *charge* response and ``tau`` has no such divergence, so it
    is treated as the magnetization is.

    **The G = 0 component is annihilated**, which is what preserves the electron
    count: ``|G|^2/(|G|^2 + q_TF^2)`` is zero there, so a preconditioned step can
    never change the total charge. That is a property worth having and not an
    accident of the formula.
    """
    import jax.numpy as jnp

    if screening is None:
        if nelec is None:
            raise ValueError("kerker_preconditioner needs either screening or nelec")
        screening = thomas_fermi_screening(float(cell.volume), float(nelec))
    grid = gvectors.grid
    size = int(np.prod(shape))
    factor = jnp.asarray(gvectors.kinetic(cell))
    factor = beta * factor / (factor + screening)
    index = gvectors.fft_index
    nspin = shape[0]

    def screened(channel):
        box = jnp.fft.fftn(channel.reshape(grid))
        coefficients = box.reshape(-1)[index] * factor
        box = jnp.zeros(box.size, dtype=box.dtype).at[index].set(coefficients)
        return jnp.real(jnp.fft.ifftn(box.reshape(grid))).reshape(-1)

    @jax.jit
    def apply(vector):
        head = vector[:size].reshape(shape)
        if nspin == 1:
            out = [screened(head[0])]
        elif nspin == 2:
            # **Only the total charge is screened.** ``approx_screening`` acts
            # on ``drho%of_g(:ngm0,1)`` alone, and index 1 of QE's density is
            # the *charge*, not a spin channel -- the Thomas-Fermi q^-2
            # divergence is a property of the charge response and the
            # magnetization has no such divergence. Densities are carried here
            # as ``(up, down)`` (see ``_magnetization``), so they are rotated
            # into ``(charge, magnetization)`` and back around the screening.
            # Screening both channels instead would damp the magnetization by
            # the charge's factor, which on a magnetic metal suppresses exactly
            # the direction the SCF has to move in.
            charge, moment = head[0] + head[1], head[0] - head[1]
            charge, moment = screened(charge), beta * moment.reshape(-1)
            out = [0.5 * (charge + moment), 0.5 * (charge - moment)]
        else:
            # ``nspin_mag = 4``: channel 0 already *is* the charge and 1..3 are
            # the magnetization as a cartesian vector, so no rotation is needed.
            out = [screened(head[0])] + [beta * head[c].reshape(-1) for c in range(1, nspin)]
        return jnp.concatenate([jnp.concatenate(out), beta * vector[size:]])

    def preconditioner(residual, density=None):
        # ``density`` is accepted and ignored: Kerker's screening length is a
        # property of the *cell* (``approx_screening``), not of ``rho(r)``.
        # ``local_tf_preconditioner`` is the one that reads it.
        return np.asarray(apply(jnp.asarray(np.asarray(residual).ravel())))

    return preconditioner


#: ``mmx`` in ``approx_screening2``: the Krylov space's width before it is
#: restarted, and how many times it may be restarted before giving up.
LOCAL_TF_MMX, LOCAL_TF_REFRESHES = 12, 4

#: ``eps32`` in ``approx_screening2``: below this the local density has no
#: Wigner-Seitz radius worth taking, and the point is left out of the average.
LOCAL_TF_EPS = 1.0e-32


def local_tf_preconditioner(gvectors, cell, shape, beta=0.7):
    """``approx_screening2``: Thomas-Fermi screening with a *local* length.

    Kerker and ``approx_screening`` screen with one number for the whole cell.
    That is exactly wrong for a **slab**, where the metal wants strong screening
    and the vacuum wants none, and a single compromise value over-screens one
    and under-screens the other -- which is charge sloshing between the surfaces
    and is what makes an unpreconditioned or uniformly preconditioned slab SCF
    diverge. ``local-TF`` makes the screening a function of ``rho(r)``.

    The screened residual is the solution ``v`` of

        4 pi e2 v(G) + |G|^2 (alpha v)(G) = |G|^2 (alpha drho)(G),

    where ``(alpha f)`` means multiplying by ``alpha(r)`` in *real* space, so
    the operator is not diagonal in ``G`` and has to be inverted iteratively --
    which is the whole reason this is a hundred lines where Kerker is three.
    ``alpha(r) = 3 (2 pi / 3)^(5/3) r_s(r)`` with ``r_s(r) = (3 / 4 pi
    |rho(r)|)^(1/3)`` the local Wigner-Seitz radius: dense regions get a small
    ``alpha`` and are screened hard, vacuum gets a large one and is left alone.

    QE solves it by a least-squares Krylov method in the **Coulomb metric**
    ``<a, b> = 4 pi e2 (Omega/2) sum_{G != 0} Re(conj(a) b) / |G|^2``, which is
    the same inner product ``rho_ddot`` measures self-consistency in, restarting
    every ``mmx = 12`` directions. That is transcribed rather than replaced by a
    library solve: the metric, the restart and the stopping rule
    ``max(1e-12, 1e-6 dr2)`` are all part of how it behaves.

    ``beta`` multiplies the result, as it does for Kerker, and for the same
    reason -- at large ``|G|`` the operator tends to the identity, so the
    convention matches. The **G = 0 component is annihilated** (the metric skips
    it and the first direction vanishes there), so a preconditioned step cannot
    change the electron count.

    Only the *charge* is screened; the magnetization, ``becsum``, ``ns`` and
    ``tau`` take the plain ``beta``, exactly as in
    :func:`kerker_preconditioner` and for the argument given there.
    """
    import jax.numpy as jnp

    grid = gvectors.grid
    size = int(np.prod(shape))
    nspin = shape[0]
    index = gvectors.fft_index
    g2 = np.asarray(gvectors.kinetic(cell))
    volume = float(cell.volume)
    points = int(np.prod(grid))
    # ``e2 = 2`` in Rydberg atomic units, so ``fpi * e2 = 8 pi``.
    fpi_e2 = 4.0 * np.pi * 2.0
    # ``gstart``: the metric and the operator both skip G = 0.
    nonzero = g2 > 1.0e-12
    weight = np.zeros_like(g2)
    weight[nonzero] = 1.0 / g2[nonzero]

    @jax.jit
    def _to_sphere(field):
        return jnp.fft.fftn(field.reshape(grid)).reshape(-1)[index] / points

    @jax.jit
    def _to_grid(coefficients):
        box = jnp.zeros(points, dtype=coefficients.dtype).at[index].set(coefficients)
        return jnp.real(jnp.fft.ifftn(box.reshape(grid))).reshape(-1) * points

    def _alpha(charge):
        """``alpha(r)`` and ``agg0``, the cell-averaged screening it falls back on."""
        magnitude = np.abs(np.asarray(charge).reshape(-1))
        dense = magnitude > LOCAL_TF_EPS
        radius = np.zeros_like(magnitude)
        radius[dense] = (3.0 / (4.0 * np.pi * magnitude[dense])) ** (1.0 / 3.0)
        # ``avg_rsm1`` is the *harmonic* mean of r_s over the grid: QE sums
        # 1/r_s and divides the point count by it, so a vacuum point -- with a
        # huge r_s and a negligible 1/r_s -- pulls the average almost not at
        # all. A plain mean would let the vacuum dominate the fallback.
        inverse = np.sum(1.0 / radius[dense]) if np.any(dense) else 0.0
        average = points / inverse if inverse > 0.0 else np.inf
        agg0 = (12.0 / np.pi) ** (2.0 / 3.0) / average
        alpha = 3.0 * (2.0 * np.pi / 3.0) ** (5.0 / 3.0) * radius
        return jnp.asarray(alpha), float(agg0)

    def _screen(residual_charge, density_charge):
        alpha, agg0 = _alpha(density_charge)

        def operator(v):
            """``4 pi e2 v + |G|^2 (alpha v)``, the system's left-hand side."""
            return fpi_e2 * v + g2 * _to_sphere(alpha * _to_grid(v))

        drho = _to_sphere(jnp.asarray(np.asarray(residual_charge).reshape(-1)))
        dv = g2 * _to_sphere(alpha * _to_grid(drho))
        dv = dv.at[~nonzero].set(0.0)

        def dot(a, b):
            return float(
                fpi_e2 * 0.5 * volume * jnp.sum(weight * jnp.real(jnp.conj(a) * b))
            )

        directions = [dv / (g2 + agg0)]
        applied, aa, bb = [], [], []
        target, best, refreshes = 0.0, None, 0
        while True:
            applied.append(operator(directions[-1]))
            m = len(applied)
            aa = np.pad(np.asarray(aa).reshape(m - 1, m - 1), ((0, 1), (0, 1))) \
                if m > 1 else np.zeros((1, 1))
            for i in range(m):
                aa[i, m - 1] = aa[m - 1, i] = dot(applied[i], applied[m - 1])
            bb = np.append(np.asarray(bb), dot(applied[m - 1], dv))
            try:
                vec = np.linalg.solve(aa, bb)
            except np.linalg.LinAlgError:
                # A dependent direction: keep the best estimate so far rather
                # than failing the whole SCF iteration for a preconditioner.
                break
            if not np.all(np.isfinite(vec)):
                break
            best = sum(c * v for c, v in zip(vec, directions))
            residue = dv - sum(c * w for c, w in zip(vec, applied))
            error = dot(residue, residue)
            if target == 0.0:
                target = max(1.0e-12, 1.0e-6 * error)
            if error < target:
                break
            if m >= LOCAL_TF_MMX:
                if refreshes >= LOCAL_TF_REFRESHES:
                    break
                # Restart from the best estimate, which is what keeps the
                # Krylov space bounded without throwing the answer away.
                refreshes += 1
                directions, applied, aa, bb = [best], [], [], []
                continue
            directions.append(residue / (g2 + agg0))
        if best is None:
            best = directions[0]
        return _to_grid(best.at[~nonzero].set(0.0))

    def preconditioner(residual, density=None):
        if density is None:
            raise ValueError(
                "local-TF is a density-dependent preconditioner and was called "
                "without one; Mixer.step passes it"
            )
        flat = jnp.asarray(np.asarray(residual).ravel())
        rho = np.asarray(density).ravel()[:size].reshape(shape)
        head = flat[:size].reshape(shape)
        if nspin == 1:
            out = [_screen(head[0], rho[0])]
        elif nspin == 2:
            charge, moment = head[0] + head[1], head[0] - head[1]
            charge = _screen(charge, rho[0] + rho[1])
            moment = beta * moment.reshape(-1)
            out = [0.5 * (beta * charge + moment), 0.5 * (beta * charge - moment)]
            return np.asarray(
                jnp.concatenate([jnp.concatenate(out), beta * flat[size:]])
            )
        else:
            out = [beta * _screen(head[0], rho[0])] + [
                beta * head[c].reshape(-1) for c in range(1, nspin)
            ]
            return np.asarray(
                jnp.concatenate([jnp.concatenate(out), beta * flat[size:]])
            )
        return np.asarray(
            jnp.concatenate([beta * out[0], beta * flat[size:]])
        )

    return preconditioner


#: ``run_scf``'s ``mixing_space``, spelled either way, to the layout it selects:
#: ``'g'`` is :class:`SphereLayout`, ``'r'`` the whole dense box in real space.
MIXING_SPACES = {"g": "g", "reciprocal": "g", "r": "r", "real": "r"}

#: What an unset ``mixing_space`` resolves to, for every mixer that accepts a
#: layout (:attr:`Mixer.accepts_layout`); the adaptive mixer always gets ``'r'``.
DEFAULT_MIXING_SPACE = "r"


def resolve_mixing_space(space, mixer) -> str:
    """``'g'`` or ``'r'`` for ``mixer``, from ``run_scf``'s ``mixing_space``.

    ``None`` is the default for a mixer that takes a layout and real space for
    one that does not; asking a real-space mixer for ``'g'`` is refused rather
    than ignored, since the run would otherwise not be the one asked for.
    """
    if space is None:
        return DEFAULT_MIXING_SPACE if mixer.accepts_layout else "r"
    try:
        resolved = MIXING_SPACES[str(space).lower()]
    except KeyError as error:
        raise ValueError(
            f"mixing_space = {space!r} is not a place to mix the density in; it "
            f"takes 'g' (the smooth sphere in reciprocal space, as pw.x's "
            f"mix_type does) or 'r' (the whole dense grid in real space)"
        ) from error
    if resolved == "g" and not mixer.accepts_layout:
        raise ValueError(
            f"mixing_space = {space!r} is not defined for {type(mixer).__name__}: "
            "it keeps one step length per grid point and steers each by the sign "
            "of that point's residual, which a Fourier coefficient does not have. "
            "Leave mixing_space unset, or set it to 'r'"
        )
    return resolved


@partial(jax.jit, static_argnames=("grid",))
def _to_half_sphere(field, index, grid):
    """A real field on the box -> its coefficients on the listed half sphere (QE's fwfft)."""
    points = grid[0] * grid[1] * grid[2]
    box = jnp.fft.fftn(field, axes=(-3, -2, -1)) / points
    return box.reshape(field.shape[:-3] + (points,))[..., index]


@partial(jax.jit, static_argnames=("grid",))
def _from_half_sphere(coefficients, index, index_minus, grid):
    """Half-sphere coefficients -> the real field, ``c(-G) = conj(c(G))`` filled in."""
    return g_to_r_gamma(coefficients, index, index_minus, grid)


class SphereLayout:
    """The density in the mixer's state as ``pw.x`` keeps it: the smooth sphere, in G.

    **What ``pw.x`` stores and what this code stored.** ``mix_rho`` works on
    ``mix_type`` objects whose density is ``of_g(1:ngms, nspin)``
    (``scf_mod.f90:216``, ``:316``): the coefficients inside the **smooth**
    cutoff ``4 ecutwfc`` and no others. It fits its Broyden coefficients there
    (``ngm0 = ngms``, ``mix_rho.f90:132``) and mixes the shell between ``ngms``
    and ``ngm`` linearly with the same ``alphamix``, without ever storing it
    (``high_frequency_mixing``, ``scf_mod.f90:549-553``, called at
    ``mix_rho.f90:548``). This code stored the whole dense box in real space,
    ``n1 n2 n3`` reals a channel, so above dual 4 each history entry was larger
    than ``pw.x``'s by about ``(dual/4)^1.5`` (``OPEN.md`` Part XXIII item 22).

    **Half of the smooth sphere, and why that is exact.** Every channel of the
    density is a real field -- the charge, and each component of the
    magnetization -- so ``c(-G) = conj(c(G))`` and one G of each pair carries all
    of it (:func:`~defumat.basis.gvectors._half_sphere`, ``ggen``'s
    ``gamma_only`` selection). The mixer combines entries with real
    coefficients, which preserves that symmetry exactly, so the stored half is a
    real field before and after every step, and the other half is put back by
    conjugation only on the way out (:func:`~defumat.basis.fft.g_to_r_gamma`).
    Nothing is lost on the way in either: the density this driver builds is
    band-limited to the dense sphere (``|psi|^2`` lies inside ``2 sqrt(ecutwfc)``,
    and the augmentation charge and ``sym_rho`` are put in G), measured at
    4.5e-35 of its norm outside it on ``benchmarks/si8-us-1k.in``. A pair
    ``(G, -G)`` has one ``|G|^2``, so it never straddles ``ngms``. So a channel
    is ``(ngms + 1)/2`` complex numbers, stored as ``ngms`` reals (``Im c(0)``
    is zero and is not kept): half of ``pw.x``'s complex ``of_g(ngms)``.

    **The stored reals are scaled so that their flat dot is the real-space one.**
    With ``c = fwfft(f)``, ``sqrt(N)`` on ``Re c(0)`` and ``sqrt(2N)`` on the real
    and imaginary parts of every other stored G make ``u . v`` equal
    ``sum_r f(r) g(r)`` over the box for the smooth parts of two fields
    (Parseval, the pair counted twice). Two things rest on that. The flat
    Anderson fit keeps weighing the density against ``ns`` and ``tau`` exactly as
    before, so moving the history into G does not also change how much say
    each block has. And at dual 4, where the shell is empty, the fit is the
    old real-space one to round-off, which is what tests the packing.

    **The shell is mixed and not stored** (:meth:`shell_step`), with the plain
    ``beta`` and no preconditioner, which is ``high_frequency_mixing``: Kerker's
    ``approx_screening`` and ``approx_screening2`` act on ``of_g(:ngm0)`` only.

    The transforms run where the density lives, compiled once per grid
    (:func:`_to_half_sphere`, :func:`_from_half_sphere`), and everything after the
    gather is host NumPy: one forward transform of each of the two densities and
    one inverse of the mixed one, per channel per iteration, and the host copy
    is the half dense sphere rather than the box.
    """

    #: The tag the mixer's history is written under (:mod:`~defumat.scf.checkpoint`).
    name = "g"

    def __init__(self, dense: GVectors, ngms: int, cell, shape):
        shape = tuple(int(n) for n in shape)
        if len(shape) != 4 or shape[1:] != tuple(dense.grid):
            raise ValueError(
                f"the density's shape {shape} is not (nspin,) + the dense grid "
                f"{tuple(dense.grid)}"
            )
        miller = np.asarray(dense.miller)
        ngm = int(miller.shape[0])
        ngms = int(ngms)
        if not 0 < ngms <= ngm:
            raise ValueError(f"ngms = {ngms} is not inside the dense set of {ngm}")
        if np.any(miller[0] != 0):
            raise AssertionError("G = 0 must be the first dense G-vector")
        half = _half_sphere(miller)
        smooth = np.flatnonzero(half[:ngms])
        shell = ngms + np.flatnonzero(half[ngms:])
        if 2 * smooth.size - 1 != ngms or 2 * shell.size != ngm - ngms:
            # A pair split by the cutoff, or a dense list that is already a half.
            raise AssertionError(
                f"the smooth sphere ({ngms}) and the shell ({ngm - ngms}) do not "
                f"each hold whole (G, -G) pairs"
            )
        order = np.concatenate([smooth, shell])
        self.shape = shape
        self.nspin = shape[0]
        self.grid = tuple(int(n) for n in dense.grid)
        self.points = int(np.prod(self.grid))
        self.ngm, self.ngms = ngm, ngms
        #: Complex coefficients a channel keeps (G = 0 first) and the shell's.
        self.nsmooth, self.nshell = int(smooth.size), int(shell.size)
        #: ``(nspin, ngms)``: the density block of the packed vector, and what
        #: :attr:`Mixer.shape` is set to so :meth:`Mixer.magnetic_step` finds it.
        self.stored_shape = (self.nspin, ngms)
        self._index = jnp.asarray(np.asarray(dense.fft_index)[order])
        self._index_minus = jnp.asarray(np.asarray(dense.fft_index_minus)[order])
        g2 = np.asarray(dense.kinetic(cell))[smooth]
        #: ``|G|^2`` in 1/bohr^2 for each stored real, in the stored order.
        self.kinetic = np.concatenate([g2[:1], g2[1:], g2[1:]])
        self._volume = float(cell.volume)
        self._w0 = float(np.sqrt(self.points))
        self._w = float(np.sqrt(2.0 * self.points))

    def history_bytes(self, depth: int, itemsize: int = 8) -> int:
        """Resident bytes of a full history's density blocks: densities and residuals."""
        return 2 * int(depth) * self.nspin * self.ngms * int(itemsize)

    def forward(self, field):
        """A real field ``(nspin, n1, n2, n3)`` -> ``(nspin, nsmooth + nshell)`` complex."""
        return _to_half_sphere(jnp.asarray(field), self._index, self.grid)

    def pack(self, coefficients):
        """Half-sphere coefficients -> ``(stored, shell)``, host arrays.

        ``stored`` is ``(nspin, ngms)`` real: ``Re c(0)``, then the real and then
        the imaginary parts of the rest of the smooth half, scaled as the class
        docstring says. ``shell`` is the shell's half, complex and unscaled.
        """
        c = np.asarray(coefficients)
        n = self.nsmooth
        stored = np.concatenate(
            [c[:, :1].real * self._w0, c[:, 1:n].real * self._w,
             c[:, 1:n].imag * self._w], axis=1)
        return stored, c[:, n:]

    def unpack(self, stored):
        """``(nspin, ngms)`` stored reals -> ``(nspin, nsmooth)`` complex coefficients."""
        stored = np.asarray(stored).reshape(self.stored_shape)
        n = self.nsmooth
        real = np.concatenate([stored[:, :1] / self._w0, stored[:, 1:n] / self._w],
                              axis=1)
        imag = np.concatenate([np.zeros_like(stored[:, :1]), stored[:, n:] / self._w],
                              axis=1)
        return real + 1j * imag

    def field(self, stored, shell=None):
        """The real field of ``stored`` and ``shell`` (zero if ``None``), on the device."""
        smooth = self.unpack(stored)
        if shell is None:
            shell = np.zeros((self.nspin, self.nshell), dtype=smooth.dtype)
        coefficients = np.concatenate([smooth, np.asarray(shell, dtype=smooth.dtype)],
                                      axis=1)
        return _from_half_sphere(jnp.asarray(coefficients), self._index,
                                 self._index_minus, self.grid)

    def stored_of(self, field) -> np.ndarray:
        """The stored reals of a real-space field's smooth part, ``(nspin, ngms)``."""
        return self.pack(jax.device_get(self.forward(field)))[0]

    def shell_step(self, residual, beta: float, beta_mag: float | None = None):
        """``beta`` times the shell's residual, the magnetization at ``beta_mag``.

        ``high_frequency_mixing``: ``alphamix`` on every component and nothing
        else, so the shell converges as plain linear mixing does. ``beta_mag``
        is :attr:`Mixer.beta_mag`, given the shell too so that it means one step
        for the magnetization at every wavelength, which is what it meant when
        the whole box went through :meth:`Mixer.magnetic_step`.
        """
        stepped = beta * np.asarray(residual)
        if beta_mag is None or self.nspin == 1:
            return stepped
        stepped = np.array(stepped, copy=True)
        _scale_magnetization(stepped, float(beta_mag) / float(beta))
        return stepped

    def rho_ddot_vector(self, stored_residual) -> np.ndarray:
        """``F`` with ``F(a) . F(b) = rho_ddot(a, b)`` over the smooth sphere, ``ngm0 = ngms``.

        :func:`~defumat.scf.potential.rho_ddot_vector`'s weights on the stored
        reals: ``0.5 Omega e2 4 pi / G^2`` on the charge with ``G = 0`` dropped
        and ``0.5 Omega e2 4 pi / (2 pi)^2`` on the magnetization with it kept,
        each divided by the ``N`` the stored scaling put in (whose doubling of
        the pair is already the full sphere's sum). Over ``ngms`` and not the
        dense set, which is ``mix_rho.f90:403-425``'s ``rho_ddot(..., ngm0)``.
        """
        head = np.asarray(stored_residual).reshape(self.stored_shape)
        g2 = self.kinetic
        inverse = np.where(g2 > 1e-12, 1.0 / np.where(g2 > 1e-12, g2, 1.0), 0.0)
        charge = head[0] if self.nspin == 4 else np.sum(head, axis=0)
        parts = [np.sqrt(0.5 * self._volume * E2 * FPI * inverse / self.points) * charge]
        if self.nspin > 1:
            moment = head[1:] if self.nspin == 4 else (head[0] - head[1])[None]
            weight = 0.5 * self._volume * E2 * FPI / (2.0 * np.pi) ** 2 / self.points
            parts.append(np.sqrt(weight) * moment)
        return np.concatenate([np.ravel(p) for p in parts])


def kerker_preconditioner_g(layout: SphereLayout, cell, beta=0.7, screening=None,
                            nelec=None):
    """:func:`kerker_preconditioner` on a :class:`SphereLayout`: a multiplication.

    ``approx_screening`` as ``pw.x`` applies it, on ``drho%of_g(:ngm0, 1)``: the
    charge on the smooth sphere times ``beta |G|^2 / (|G|^2 + q_TF^2)``, the
    magnetization and everything after the density at the plain ``beta``. No
    transform, because the coefficients are what is stored; the real-space form
    above is still what the residual solver's Krylov preconditioner and its
    warm-up mixer use, on a packed vector that is not in this layout. ``G = 0``
    has ``|G|^2 = 0`` and is annihilated, so a step never moves the electron
    count. The shell takes ``beta`` from :meth:`SphereLayout.shell_step`.
    """
    if screening is None:
        if nelec is None:
            raise ValueError("kerker_preconditioner_g needs either screening or nelec")
        screening = thomas_fermi_screening(float(cell.volume), float(nelec))
    factor = beta * layout.kinetic / (layout.kinetic + screening)
    shape = layout.stored_shape
    size = int(np.prod(shape))
    nspin = layout.nspin

    def preconditioner(residual, density=None):
        # ``density`` is ignored, as in the real-space form: Kerker's screening
        # length is a property of the cell.
        residual = np.asarray(residual).ravel()
        scale = factor.astype(residual.dtype, copy=False)
        out = beta * residual
        head = residual[:size].reshape(shape)
        if nspin == 1:
            out[:size] = scale * head[0]
        elif nspin == 2:
            # Only the charge is screened; the pair is turned into
            # ``(charge, moment)`` and back, as in the real-space form.
            charge, moment = head[0] + head[1], head[0] - head[1]
            charge, moment = scale * charge, beta * moment
            out[:size] = np.concatenate([0.5 * (charge + moment),
                                         0.5 * (charge - moment)])
        else:
            out[:shape[1]] = scale * head[0]
        return out

    return preconditioner


def local_tf_preconditioner_g(layout: SphereLayout, dense: GVectors, cell, beta=0.7):
    """:func:`local_tf_preconditioner` on a :class:`SphereLayout`, on ``pw.x``'s sphere.

    ``approx_screening2(drho, rhobest)`` works on ``mix_type`` objects, so both
    the residual it screens and the density it reads ``r_s(r)`` from are their
    ``of_g(:ngm0)``: the smooth sphere, the shell included in neither. The
    real-space solver is reused unchanged with its own transforms, built on the
    smooth sphere's G-vectors placed in the dense box, so its Krylov space and
    its Coulomb metric run over ``ngm0`` as ``pw.x``'s do; the residual and the
    density go to the box from their stored smooth halves, and the screened
    charge comes back the same way. Everything after the density takes ``beta``.
    """
    smooth = GVectors(miller=dense.miller[:layout.ngms], grid=dense.grid,
                      ecut=dense.ecut, gamma_only=dense.gamma_only)
    screen = local_tf_preconditioner(smooth, cell, layout.shape, beta)
    size = int(np.prod(layout.stored_shape))

    def preconditioner(residual, density=None):
        if density is None:
            raise ValueError(
                "local-TF is a density-dependent preconditioner and was called "
                "without one; Mixer.step passes it"
            )
        residual = np.asarray(residual).ravel()
        density = np.asarray(density).ravel()
        head = np.asarray(layout.field(residual[:size])).ravel()
        rho = np.asarray(layout.field(density[:size])).ravel()
        screened = np.asarray(screen(head, rho)).reshape(layout.shape)
        out = beta * residual
        out[:size] = layout.stored_of(screened).ravel()
        return out

    return preconditioner


def get_mixer(name: str, **kwargs) -> Mixer:
    """The mixer registered under ``name``, built from the keywords it takes.

    A keyword whose value is ``None`` is dropped rather than passed, so a caller
    can forward an unset input variable without a branch of its own; one the
    mixer has no field for is a named error rather than a ``TypeError``, because
    the caller is usually an input file and the fix is in the input file.
    """
    try:
        mixer = MIXERS[name.lower()]
    except KeyError as error:
        raise ValueError(f"unknown mixing mode {name!r}; expected one of {sorted(MIXERS)}") from error
    supplied = {key: value for key, value in kwargs.items() if value is not None}
    accepted = {
        entry.name for entry in dataclasses.fields(mixer)
        if not entry.name.startswith("_")
    }
    unknown = sorted(set(supplied) - accepted)
    if unknown:
        raise ValueError(
            f"mixing mode {name!r} has no {', '.join(unknown)}; it takes "
            f"{sorted(accepted)}"
        )
    return mixer(**supplied)
