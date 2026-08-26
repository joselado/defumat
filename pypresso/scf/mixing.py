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
  (``approx_screening2``), a space-dependent screening length, which is not
  implemented here and is refused by name; ``scf/solvers.py`` is the other route,
  an exact Jacobian that makes no such assumption and costs far more.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import numpy as np

__all__ = ["Mixer", "LinearMixer", "AndersonMixer", "get_mixer", "MIXERS",
           "PRECONDITIONED", "kerker_preconditioner", "thomas_fermi_screening"]


class Mixer:
    """Interface: given the input and output density, propose the next input."""

    #: Applied to the residual in place of multiplying by ``beta``. ``None`` is
    #: the plain scalar. The driver installs one for a preconditioned mixer,
    #: because building it needs the G-vectors and the mixer does not have them.
    precondition = None

    def mix(self, rho_in: np.ndarray, rho_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def step(self, residual: np.ndarray) -> np.ndarray:
        """``beta * r``, or the preconditioner's version of it."""
        if self.precondition is None:
            return self.beta * residual
        return self.precondition(residual)

    def reset(self) -> None:
        pass


@dataclass
class LinearMixer(Mixer):
    beta: float = 0.7

    def mix(self, rho_in, rho_out):
        return rho_in + self.step(rho_out - rho_in)


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
            return (rho_in + self.step(residual)).reshape(np.asarray(rho_out).shape)

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
            return (rho_in + self.step(residual)).reshape(np.asarray(rho_out).shape)

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
            return (rho_in + self.step(residual)).reshape(np.asarray(rho_out).shape)

        # The belt to the braces above. Normalising fixes the conditioning that
        # comes from the *spread* of the residuals; it cannot fix a history whose
        # members have become genuinely parallel, and a solve that survived
        # ``LinAlgError`` can still return non-finite coefficients. Falling back
        # to a linear step costs one slow iteration; not checking costs the run.
        if not np.all(np.isfinite(coefficients)):
            self.reset()
            return (rho_in + self.step(residual)).reshape(np.asarray(rho_out).shape)

        mixed = sum(c * (d + self.step(r)) for c, d, r in
                    zip(coefficients, self._densities[used], self._residuals[used]))
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
    # QE's names for the same thing. 'local-TF' is a *space-dependent* screening
    # length and is not implemented; it is refused rather than silently given
    # the uniform one, which is the whole distinction that matters here.
    "kerker": AndersonMixer,
    "tf": AndersonMixer,
    # QE's own default name, so an unedited pw.x input reaches a mixer here.
    "default": AndersonMixer,
}

#: Mixing modes whose ``beta`` is an operator, so the driver has to build it.
PRECONDITIONED = {"kerker", "tf"}

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
    (``approx_screening2``), a space-dependent screening length, which is not
    implemented here and is refused by name rather than substituted.
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

    def preconditioner(residual):
        return np.asarray(apply(jnp.asarray(np.asarray(residual).ravel())))

    return preconditioner


def get_mixer(name: str, **kwargs) -> Mixer:
    if name.lower() == "local-tf":
        raise NotImplementedError(
            "mixing_mode = 'local-TF' is QE's approx_screening2, a "
            "*space-dependent* Thomas-Fermi screening length. It is not "
            "implemented, and it is refused rather than substituted by the "
            "uniform 'TF' -- the difference between them is the whole point "
            "on the inhomogeneous systems either is used for (PLAN.md P22)"
        )
    try:
        return MIXERS[name.lower()](**kwargs)
    except KeyError as error:
        raise ValueError(f"unknown mixing mode {name!r}; expected one of {sorted(MIXERS)}") from error
