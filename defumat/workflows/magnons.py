"""Magnons: the pole of the transverse spin susceptibility.

``PLAN.md`` P63, and Elk's tasks 330/331 (``tddftsplr.f90``) for a collinear
ground state. Two entry points -- :func:`run_spin_susceptibility` at one
wavevector, and :func:`run_magnon_dispersion` along a path of them.

**What a magnon is, in one line of algebra.** The Kohn-Sham transverse response
``X_0`` describes independent spin-flip excitations, whose weight is spread over
the Stoner continuum ``eps_dn - eps_up``. Turning on the exchange-correlation
kernel,

    X = [1 - X_0 F]^{-1} X_0,     F(r) = B_xc(r)/m(r),

pulls a **collective** mode out from below that continuum: a pole of ``X``
where ``1 - X_0 F`` is singular. That mode is the spin wave. At ``q = 0`` it
sits at exactly zero energy, because rotating every spin together costs
nothing, and how far the computed pole is from zero is the calculation's own
error bar (:attr:`SpinSusceptibility.goldstone`).

**There is no Coulomb term in this Dyson equation** and that is not an
omission: the Hartree interaction lives in the charge channel, and Elk adds its
regularised form to the ``(1,1)`` element of the 4x4 kernel only. A transverse
spin fluctuation moves no charge.

**The spectral function is read at ``G = G' = 0``.** ``-Im X_{00}(q, omega)/pi``
is what a neutron scattering experiment measures, up to the magnetic form
factor: the response of the cell-averaged transverse magnetization. The
local-field structure of the matrix is what shifts the pole, not what is read
off it.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from defumat.tddft.spinchi0 import (
    require_a_transverse_regime,
    transverse_response,
)
from defumat.tddft.spinkernel import (
    goldstone_residual,
    transverse_kernel_matrix,
)
from defumat.units import RY_TO_EV
from defumat.workflows.nscf import fixed_density_states

__all__ = [
    "SpinSusceptibility",
    "MagnonDispersion",
    "run_spin_susceptibility",
    "run_magnon_dispersion",
]

#: 1 Ry in cm^-1, the unit a spin-wave energy is usually quoted in -- and the
#: one the phonon side of this package already prints.
RY_TO_MEV = RY_TO_EV * 1000.0


class SpinSusceptibility(eqx.Module):
    """``X^{+-}(q, omega)`` at one wavevector, and what it took."""

    #: ``(3,)`` crystal coordinates.
    q: tuple = eqx.field(static=True)
    #: ``(nw,)`` Ry, as asked for.
    frequencies: np.ndarray
    #: ``(nw,)`` the enhanced response at ``G = G' = 0``.
    chi: np.ndarray
    #: ``(nw,)`` the same element of the Kohn-Sham response, for the Stoner
    #: continuum the collective mode sits below.
    chi0: np.ndarray
    #: How far ``X_0 B_xc = m`` is from holding, relative to ``|m|``. Zero for
    #: an exact calculation at ``q = 0``; a double truncation in ``nbnd`` and
    #: ``ecut_response`` otherwise. ``None`` when ``q != 0``, where the
    #: identity is not a statement.
    goldstone: float | None = eqx.field(static=True, default=None)
    #: The magnon energy (Ry): the frequency at which the spectral function
    #: peaks, refined by a parabola through its three highest points. ``None``
    #: when the maximum is at the edge of the frequency grid, which means the
    #: grid does not contain the pole.
    magnon: float | None = eqx.field(static=True, default=None)
    #: How the magnon energy was found: ``"crossing"`` (the root of
    #: ``lambda_max(omega) = 1``, which is exact and needs no broadening) or
    #: ``"peak"`` (the maximum of the spectral function, which is what is left
    #: once the mode is inside the Stoner continuum and Landau-damped).
    method: str = eqx.field(static=True, default="crossing")
    #: ``lambda_max(X_0 F)`` at ``omega = 0``. One is the Goldstone value at
    #: ``q = 0``; **above one at any q it is an instability** -- the transverse
    #: susceptibility has changed sign, so the collinear state is not a
    #: minimum and the "magnon" would be at negative energy.
    enhancement: float = eqx.field(static=True, default=0.0)
    #: The factor the kernel was scaled by, one unless a Goldstone correction
    #: was applied.
    kernel_scale: float = eqx.field(static=True, default=1.0)
    nbnd: int = eqx.field(static=True, default=0)
    nm: int = eqx.field(static=True, default=0)
    broadening: float = eqx.field(static=True, default=0.0)

    @property
    def unstable(self) -> bool:
        """Whether the collinear ground state is unstable at this ``q``."""
        return self.enhancement > 1.0

    @property
    def spectral_function(self) -> np.ndarray:
        """``-Im X_{00}/pi``, which is what a neutron sees."""
        return -np.imag(np.asarray(self.chi)) / np.pi

    @property
    def kohn_sham_spectral_function(self) -> np.ndarray:
        return -np.imag(np.asarray(self.chi0)) / np.pi

    @property
    def magnon_mev(self) -> float | None:
        return None if self.magnon is None else self.magnon * RY_TO_MEV

    def plot(self, ax=None, **kwargs):
        """The spectral function against energy, with the Stoner one beneath."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        energy = np.asarray(self.frequencies) * RY_TO_MEV
        kwargs.setdefault("label", "interacting")
        ax.plot(energy, self.spectral_function, **kwargs)
        ax.plot(energy, self.kohn_sham_spectral_function, ls="--", color="0.6",
                label="Kohn-Sham (Stoner)")
        ax.set_xlabel("Energy (meV)")
        ax.set_ylabel(r"$-\mathrm{Im}\,\chi^{+-}_{00}/\pi$")
        ax.legend()
        return ax


class MagnonDispersion(eqx.Module):
    """``omega(q)`` along a path, and the susceptibilities it was read from."""

    #: ``(nq, 3)`` crystal coordinates.
    qpoints: np.ndarray
    #: ``(nq,)`` Ry, with ``nan`` where no peak was found inside the grid.
    energies: np.ndarray
    #: ``(nq, nw)`` the spectral function, so the continuum can be drawn too.
    spectral: np.ndarray
    frequencies: np.ndarray
    susceptibilities: tuple = ()
    goldstone: float | None = eqx.field(static=True, default=None)
    #: The factor the kernel was scaled by -- one unless a Goldstone
    #: correction was applied.
    kernel_scale: float = eqx.field(static=True, default=1.0)
    #: ``(nq,)`` ``lambda_max(X_0 F)`` at ``omega = 0``. Anything above one is
    #: an instability of the collinear state at that wavevector, which is a
    #: result rather than a failure -- it says where the spiral energy ``E(q)``
    #: has its minimum.
    enhancements: np.ndarray | None = None

    @property
    def unstable(self) -> np.ndarray:
        """``(nq,)`` where the collinear ground state is unstable."""
        return np.asarray(self.enhancements) > 1.0

    @property
    def energies_mev(self) -> np.ndarray:
        return np.asarray(self.energies) * RY_TO_MEV

    def plot(self, ax=None, **kwargs):
        """The dispersion over a colour map of the spectral function."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        distance = _path_distance(np.asarray(self.qpoints))
        energy = np.asarray(self.frequencies) * RY_TO_MEV
        ax.pcolormesh(distance, energy, np.asarray(self.spectral).T,
                      shading="nearest", cmap="magma")
        kwargs.setdefault("color", "white")
        kwargs.setdefault("marker", "o")
        ax.plot(distance, self.energies_mev, **kwargs)
        ax.set_xlabel("q along the path")
        ax.set_ylabel("Energy (meV)")
        return ax


def run_spin_susceptibility(
    system,
    pseudos,
    density,
    q,
    frequencies,
    *,
    kpoints=None,
    nbnd: int | None = None,
    broadening: float = 0.002,
    ecut_response: float = 8.0,
    flip: bool = False,
    interacting: bool = True,
    kernel_scale: float = 1.0,
    crossing_points: int = 17,
    conv_thr: float = 1.0e-8,
    k_batch: int | None | str = "default",
    goldstone: bool | None = None,
    states=None,
) -> SpinSusceptibility:
    """``X^{+-}(q, omega)`` for a converged collinear magnet.

    Args:
        system: the converged :class:`~defumat.system.builder.System`. It must
            carry the **whole** k-grid, ``nosym`` and ``noinv``: the states at
            ``k + q`` are read out of the same grid, which needs it closed
            under translation by ``q``.
        density: ``SCFResult.density``, ``(2, ...)``.
        q: the wavevector in **crystal** coordinates. It has to be a difference
            of two k-points of the grid, which is Elk's requirement that
            ``vecql`` be commensurate with the mesh reached by construction.
        frequencies: ``(nw,)`` Ry. A magnon is at a few tens or hundreds of
            meV, which is ``1e-3`` to ``1e-2`` Ry -- a far finer grid than an
            optical spectrum wants. **The magnon energy comes from the
            eigenvalue crossing rather than from this grid**, so a dozen points
            are enough unless the spectral function itself is wanted.
        nbnd: how many bands the fixed-density run computes. Both channels need
            empty states -- the minority ones are where the majority electrons
            go.
        broadening: ``eta``, Ry. It sets the width of the spectral function's
            peak and nothing else; the pole is found without it.
        ecut_response: the cutoff (Ry) of the G-set the response is a matrix
            over -- Elk's ``gmaxrf``. **On a transition metal this, and not the
            band count, is what the Goldstone residual measures**: ``B_xc`` of
            a 3d shell has structure on the orbital's own scale, and a sphere
            that cannot hold it truncates the identity's own left-hand side.
        flip: exchange the spin channels, which gives ``X^{-+}``. The magnon of
            a majority-up ferromagnet is at positive ``omega`` in ``X^{+-}``,
            which is the default; a run whose majority channel is the second
            one needs this.
        interacting: solve the Dyson equation. ``False`` returns the Kohn-Sham
            response alone, which is the Stoner continuum with no collective
            mode -- which is how one shows the mode is the kernel's doing.
        kernel_scale: a factor on ``F``. One is the calculation as it stands;
            :func:`run_magnon_dispersion`'s ``goldstone_correction`` sets it to
            ``1/lambda_max(q = 0)`` so that the uniform mode sits at exactly
            zero. See that function for what the correction is and is not.
        crossing_points: how many frequencies the eigenvalue ``lambda_max(omega)``
            is evaluated at when locating the pole. A dense eigendecomposition
            is ``O(nm^3)``, so this is subsampled from ``frequencies`` rather
            than run at every one of them.
        goldstone: whether to measure the ``q = 0`` identity. The default is to
            do it exactly when ``q`` is zero, where it is a statement; it costs
            one extra frequency at ``eta = 0``.
        states: ``(calculation, system, eigenvalues, wavefunctions)`` from an
            earlier :func:`~defumat.workflows.nscf.fixed_density_states`, to be
            reused. A dispersion is many wavevectors on **one** set of states,
            which is the whole reason ``q`` is restricted to the grid.
    """
    from defumat.scf.driver import Calculation

    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
    q = np.asarray(q, dtype=float).reshape(3)

    if states is None:
        if kpoints is not None:
            system = eqx.tree_at(lambda s: s.kpoints, system, kpoints)
        # The refusals are checked before the fixed-density run rather than
        # inside it: they are statements about the calculation, and a caller
        # asking for something this cannot do should not first pay for the
        # bands.
        require_a_transverse_regime(Calculation(system, pseudos, k_batch=k_batch))
        states = fixed_density_states(
            system, pseudos, density,
            nbnd=nbnd or _default_nbnd(system, pseudos),
            conv_thr=conv_thr, k_batch=k_batch,
        )
    calculation, system, eigenvalues, wavefunctions = states

    if goldstone is None:
        goldstone = bool(np.all(np.abs(q) < 1.0e-10))

    # **Index 0 is the static point and it carries no broadening.** Both things
    # read there are exact statements rather than spectra -- the Goldstone
    # identity, and ``lambda_max`` at ``omega = 0``, which is the enhancement --
    # and an ``eta`` puts an ``O(eta)`` error into each that reads as an
    # assembly bug.
    grid = np.concatenate([[0.0], frequencies])
    imaginary = np.concatenate([[0.0], np.full(len(frequencies), broadening)])

    chi0 = transverse_response(
        calculation, wavefunctions, eigenvalues, q, grid + 1j * imaginary,
        ecut_response=ecut_response, broadening=0.0, flip=flip,
        k_batch=k_batch,
    )

    residual = None
    if goldstone:
        residual = goldstone_residual(chi0, calculation, jnp.asarray(density))

    x0 = np.asarray(chi0.x)
    kernel = None
    if interacting:
        kernel = float(kernel_scale) * np.asarray(
            transverse_kernel_matrix(calculation, jnp.asarray(density), chi0.sphere)
        )
        identity = np.eye(chi0.nm)
        x = np.linalg.solve(identity[None] - x0 @ kernel[None], x0)
    else:
        x = x0

    enhancement = 0.0 if kernel is None else float(
        np.linalg.eigvals(x0[0] @ kernel).real.max()
    )
    energy, method = _locate(
        frequencies, x0[1:], kernel, x[1:, 0, 0], enhancement, crossing_points,
    )
    return SpinSusceptibility(
        q=tuple(float(component) for component in q),
        frequencies=frequencies,
        chi=x[1:, 0, 0],
        chi0=x0[1:, 0, 0],
        goldstone=residual,
        magnon=energy,
        method=method,
        enhancement=enhancement,
        kernel_scale=float(kernel_scale),
        nbnd=int(eigenvalues.shape[-1]),
        nm=chi0.nm,
        broadening=float(broadening),
    )


def _locate(frequencies, x0, kernel, head, enhancement, points):
    """The magnon energy, by the eigenvalue crossing where that is possible.

    **The pole is a root of ``lambda_max(omega) = 1``, not a maximum of the
    spectral function**, and finding it that way is both cheaper and more
    honest. Below the Stoner onset ``X_0`` is Hermitian and ``lambda_max`` is
    real and rises monotonically with ``omega``, so the crossing is a
    one-dimensional root on a handful of frequencies -- no broadening enters
    it, and it says "below zero" where the state is unstable instead of
    reporting the grid's own edge as a magnon.

    Inside the continuum the mode is Landau-damped: ``lambda_max`` picks up an
    imaginary part, there is no root, and what is left is a broadened peak.
    That is when this falls back to :func:`_peak`, and the returned ``method``
    says which happened -- a dispersion whose outer points are ``"peak"`` is
    reporting a damped mode and should be read as one.
    """
    if kernel is None:
        return _peak(frequencies, -np.imag(head)), "peak"
    if enhancement > 1.0 + _SOFT_TOLERANCE:
        # The uniform-rotation mode has already gone soft: the crossing is at
        # negative frequency, so there is no magnon to report at this q.
        return None, "unstable"
    if enhancement > 1.0 - _SOFT_TOLERANCE:
        # Sitting on the crossing. This is where ``q = 0`` lands once a
        # Goldstone correction has been applied -- by construction, since that
        # is what the correction sets -- and reporting it as an instability
        # because the eigenvalue came back 1 + 1e-12 would be an arithmetic
        # accident rather than physics.
        return 0.0, "crossing"

    stride = max(1, len(frequencies) // max(int(points) - 1, 1))
    sample = np.arange(0, len(frequencies), stride)
    values = np.array([
        np.linalg.eigvals(x0[i] @ kernel).real.max() for i in sample
    ])
    energies = frequencies[sample]
    above = np.flatnonzero(values >= 1.0)
    if above.size and above[0] > 0:
        i = int(above[0])
        left, right = values[i - 1], values[i]
        weight = (1.0 - left) / (right - left)
        return float(energies[i - 1] + weight * (energies[i] - energies[i - 1])), "crossing"
    return _peak(frequencies, -np.imag(head)), "peak"


#: How close to one ``lambda_max`` may be at ``omega = 0`` and still count as
#: sitting on the crossing rather than past it. A magnon energy is linear in
#: this near the crossing, and the frequency grids involved are coarser than it
#: by orders of magnitude.
_SOFT_TOLERANCE = 1.0e-6


def run_magnon_dispersion(
    system,
    pseudos,
    density,
    qpoints,
    frequencies,
    *,
    kpoints=None,
    nbnd: int | None = None,
    conv_thr: float = 1.0e-8,
    k_batch: int | None | str = "default",
    goldstone_correction: bool = False,
    **kwargs,
) -> MagnonDispersion:
    """``omega(q)`` along a list of wavevectors, on **one** set of states.

    Every ``q`` is a difference of two k-points of the grid, so the states at
    ``k + q`` are the states already computed -- the fixed-density run happens
    once and each wavevector costs one sum over states and one matrix solve.

    ``goldstone_correction`` scales the kernel by ``1/lambda_max(q = 0)``, the
    **same factor at every q**, so that the uniform mode sits at exactly zero.
    It is what the magnon literature does (a kernel scaling in Sasioglu,
    Buczek, Rousseau and Skovhus-Olsen; a shift of the Stoner spectrum in
    Muller-Friedrich-Blugel), and the reason it is a kernel scaling rather than
    a shift is the one measured here: what limits the identity on a transition
    metal is the **response sphere**, which truncates the kernel's own matrix.

    **It is opt-in and it must be shown to work rather than assumed to.** The
    correction is honest only if the error it absorbs is q-independent, and the
    way to know is to run the dispersion at two ``(nbnd, ecut_response)``
    settings: corrected, they must agree better than uncorrected. Where they do
    not, the residual error is q-dependent and the correction is cosmetic. The
    factor is reported on every point (:attr:`SpinSusceptibility.kernel_scale`)
    so a reader can see how large it was.
    """
    from defumat.scf.driver import Calculation

    if kpoints is not None:
        system = eqx.tree_at(lambda s: s.kpoints, system, kpoints)
    require_a_transverse_regime(Calculation(system, pseudos, k_batch=k_batch))
    states = fixed_density_states(
        system, pseudos, density, nbnd=nbnd or _default_nbnd(system, pseudos),
        conv_thr=conv_thr, k_batch=k_batch,
    )

    qpoints = np.atleast_2d(np.asarray(qpoints, dtype=float))
    scale = kwargs.pop("kernel_scale", 1.0)
    if goldstone_correction:
        uniform = run_spin_susceptibility(
            system, pseudos, density, np.zeros(3), frequencies,
            states=states, k_batch=k_batch, **kwargs,
        )
        if uniform.enhancement <= 0.0:
            raise RuntimeError(
                "the Goldstone correction needs the interacting response: "
                "lambda_max at q = 0 came back non-positive, which happens "
                "when interacting = False"
            )
        scale = 1.0 / uniform.enhancement

    results = tuple(
        run_spin_susceptibility(
            system, pseudos, density, q, frequencies,
            states=states, k_batch=k_batch, kernel_scale=scale, **kwargs,
        )
        for q in qpoints
    )
    return MagnonDispersion(
        qpoints=qpoints,
        energies=np.array([
            np.nan if r.magnon is None else r.magnon for r in results
        ]),
        spectral=np.array([r.spectral_function for r in results]),
        frequencies=np.atleast_1d(np.asarray(frequencies, dtype=float)),
        susceptibilities=results,
        goldstone=next((r.goldstone for r in results if r.goldstone is not None),
                       None),
        kernel_scale=float(scale),
        enhancements=np.array([r.enhancement for r in results]),
    )


def _default_nbnd(system, pseudos) -> int:
    """Enough empty states for the spin-flip sum, which is not the optical rule.

    Both channels need empty bands here -- the minority ones are where the
    majority electrons go -- so the count is taken off the *larger* of the two
    channel fillings rather than off half the electrons.
    """
    from defumat.scf.driver import Calculation

    calculation = Calculation(system, pseudos)
    # ``nelup``/``neldw`` are ``None`` unless the magnetization is constrained,
    # so the electron count is the fallback rather than the exception.
    counts = [calculation.nelec / 2.0]
    for count in (getattr(calculation, "nelup", None),
                  getattr(calculation, "neldw", None)):
        if count is not None:
            counts.append(float(count))
    filled = max(counts)
    return max(int(4 * filled), int(filled) + 8)


def _peak(frequencies, weight):
    """The maximum of a sampled spectral function, refined by a parabola.

    ``None`` when the maximum is at an end of the grid: the grid does not
    contain the pole then, and reporting its edge as a magnon energy is how a
    dispersion acquires a flat branch that is nothing but the window.
    """
    weight = np.asarray(weight)
    if weight.size < 3:
        return None
    best = int(np.argmax(weight))
    if best in (0, weight.size - 1):
        return None
    left, middle, right = weight[best - 1], weight[best], weight[best + 1]
    denominator = left - 2.0 * middle + right
    step = 0.0 if abs(denominator) < 1e-300 else 0.5 * (left - right) / denominator
    spacing = frequencies[best + 1] - frequencies[best - 1]
    return float(frequencies[best] + 0.5 * step * spacing)


def _path_distance(qpoints):
    """Cumulative distance along a q-path, in crystal coordinates."""
    steps = np.linalg.norm(np.diff(qpoints, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])
