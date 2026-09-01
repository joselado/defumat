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

from dataclasses import dataclass, field

import jax
import numpy as np

__all__ = ["Mixer", "LinearMixer", "AndersonMixer", "get_mixer", "MIXERS",
           "PRECONDITIONED", "kerker_preconditioner", "local_tf_preconditioner",
           "thomas_fermi_screening"]


class Mixer:
    """Interface: given the input and output density, propose the next input."""

    #: Applied to the residual in place of multiplying by ``beta``. ``None`` is
    #: the plain scalar. The driver installs one for a preconditioned mixer,
    #: because building it needs the G-vectors and the mixer does not have them.
    precondition = None

    def mix(self, rho_in: np.ndarray, rho_out: np.ndarray) -> np.ndarray:
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
            return self.beta * residual
        return self.precondition(residual, density)

    def reset(self) -> None:
        pass


@dataclass
class LinearMixer(Mixer):
    beta: float = 0.7

    def mix(self, rho_in, rho_out):
        return rho_in + self.step(rho_out - rho_in, rho_in)


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

    def reset(self):
        self._densities.clear()
        self._residuals.clear()

    def mix(self, rho_in, rho_out):
        rho_in = np.asarray(rho_in).ravel()
        residual = np.asarray(rho_out).ravel() - rho_in

        self._densities.append(rho_in)
        self._residuals.append(residual)
        if len(self._densities) > self.history:
            self._densities.pop(0)
            self._residuals.pop(0)

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
        norms = np.array([float(np.sqrt(r @ r)) for r in self._residuals])
        if not np.all(norms > 0.0):
            self.reset()
            return (rho_in + self.step(residual, rho_in)).reshape(
                np.asarray(rho_out).shape
            )

        gram = np.array([[float(a @ b) for b in self._residuals] for a in self._residuals])

        # Normalising removes the conditioning that came from the residuals'
        # *spread*; it cannot remove what comes from their *alignment*, and that
        # grows with the cell -- the same measurement gives 2.7e4 on sixteen
        # atoms and 1.7e8 on sixty-four. So the oldest entries are dropped until
        # what is left is solvable, which is standard DIIS practice and is a
        # bound rather than a hope. The cap sits four orders above the worst
        # value ever measured here, so it never fires on anything already
        # working, and it keeps the coefficients' relative error near 1e-4 in
        # the regime where it does.
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
        combination = zip(coefficients, self._densities[used], self._residuals[used])
        mixed_density, mixed_residual = 0.0, 0.0
        for c, d, r in combination:
            mixed_density = mixed_density + c * d
            mixed_residual = mixed_residual + c * r
        mixed = mixed_density + self.step(mixed_residual, mixed_density)
        return np.asarray(mixed).reshape(np.asarray(rho_out).shape)

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


def get_mixer(name: str, **kwargs) -> Mixer:
    try:
        return MIXERS[name.lower()](**kwargs)
    except KeyError as error:
        raise ValueError(f"unknown mixing mode {name!r}; expected one of {sorted(MIXERS)}") from error
