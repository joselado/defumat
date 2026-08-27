"""The optical absorption spectrum of an insulator, with an excitonic kernel.

``PLAN.md`` P37. One entry point, :func:`run_absorption`, which does what
Elk's ``task = 320`` (``tddftlr.f90``) does: a fixed-density run with empty
states, the Adler-Wiser ``chi_0`` built from them, and the Dyson equation solved
at every frequency with a named exchange-correlation kernel.

**What comes out is ``eps_M(omega)``**, the macroscopic dielectric function with
local-field effects, whose imaginary part is the absorption spectrum. With
``kernel = 'rpa'`` it is the independent-particle spectrum screened only by
Hartree; with ``'bootstrap'`` it carries the bound electron-hole pair that no
adiabatic local kernel can produce, because the bootstrap's ``f_xc`` diverges as
``1/q^2`` and ALDA's does not.

**Two knobs are part of the physics rather than of the numerics**, and both are
easy to leave at a wrong default:

``nbnd``
    how many empty states the sum runs over. There is no refusal that can catch
    this being too small -- an undersized spectrum is smooth, positive and
    plausible -- so :attr:`OpticalSpectrum.static_residual` measures it against
    an independent, band-complete route and is computed by default.
``scissor``
    the rigid shift of the empty states. PRL 107, 186401 Eq. (3) makes it part
    of the method: the kernel is built from a ``chi_0`` whose gap is right, and
    every published bootstrap spectrum uses one. Left at zero the spectrum
    inherits the Kohn-Sham gap and sits too low.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from pypresso.tddft.chi0 import (
    independent_response,
    require_a_sum_over_states_regime,
)
from pypresso.tddft.dyson import MAX_ITERATIONS, TOLERANCE, solve_dyson
from pypresso.tddft.kernels import DEFAULT_KERNEL, alda_matrix, get_kernel
from pypresso.units import RY_TO_EV
from pypresso.workflows.nscf import fixed_density_states

__all__ = ["OpticalSpectrum", "run_absorption"]


class OpticalSpectrum(eqx.Module):
    """``eps_M(omega)`` and what it took to get there."""

    #: ``(nw,)`` Ry, as asked for -- the internal ``omega = 0`` points are not
    #: in here.
    frequencies: np.ndarray
    #: ``(nw, 3, 3)`` complex: the macroscopic tensor, local fields included.
    epsilon: np.ndarray
    #: ``(nw, 3, 3)`` complex: the same with the local-field effect removed,
    #: which is ``1 - X_head``. The gap between the two is worth about 10% on
    #: silicon and is the reason the Dyson equation is a matrix equation.
    epsilon_no_local_fields: np.ndarray
    kernel: str = eqx.field(static=True)
    #: The long-range-correction parameter the kernel's head is equivalent to,
    #: ``-4 pi F_00``. One number that says how strongly this kernel binds.
    alpha: float = eqx.field(static=True)
    #: ``-4 pi (F X)_00``, the quantity ``tddftlr.f90`` prints under the name
    #: alpha. Different from :attr:`alpha`; carried so that a comparison with
    #: Elk compares like with like.
    alpha_elk: float = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    converged: bool = eqx.field(static=True)
    nbnd: int = eqx.field(static=True)
    nocc: int = eqx.field(static=True)
    nm: int = eqx.field(static=True)
    scissor: float = eqx.field(static=True)
    #: ``eps_M(0)`` from this sum over states, in RPA and at **no** scissors
    #: shift, minus the same quantity from the Sternheimer solve -- which is
    #: band-complete, so the difference is the **band truncation of this run**
    #: and nothing else. Both matchings are deliberate: differencing a bootstrap
    #: spectrum against a Sternheimer number that contains ``f_xc`` measures the
    #: kernel, and differencing a scissor-shifted one measures the shift (worth
    #: -3.46 against +0.013 on silicon at 0.05 Ry). ``None`` when it was not
    #: asked for.
    static_residual: float | None = eqx.field(static=True, default=None)
    #: ``eps_M(0)`` in RPA from this run, the left half of that difference.
    static_rpa: float | None = eqx.field(static=True, default=None)

    @property
    def frequencies_ev(self) -> np.ndarray:
        return np.asarray(self.frequencies) * RY_TO_EV

    @property
    def absorption(self) -> np.ndarray:
        """``Im eps_M`` averaged over the three diagonal directions, ``(nw,)``.

        The isotropic average, which is what an unpolarised measurement on a
        cubic crystal sees and what every published spectrum here is compared
        against. The full tensor is on :attr:`epsilon` for anything anisotropic.
        """
        return np.imag(np.trace(np.asarray(self.epsilon), axis1=1, axis2=2)) / 3.0

    def plot(self, ax=None, part: str = "imaginary", ev: bool = True, **kwargs):
        """Draw the spectrum, and return the axes.

        ``part`` selects ``"imaginary"`` (absorption, the usual one),
        ``"real"``, or ``"both"``. The curve is the *macroscopic* dielectric
        function -- the inverse of the head of ``eps^-1``, not the head of the
        inverse -- so what is drawn already carries the local fields.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        x = self.frequencies_ev if ev else np.asarray(self.frequencies)
        # The isotropic average, as :attr:`absorption` takes it: ``epsilon`` is
        # the full ``(nw, 3, 3)`` tensor and a curve is one number per
        # frequency. Anything anisotropic is plotted from that tensor by hand.
        trace = np.trace(np.asarray(self.epsilon), axis1=1, axis2=2) / 3.0
        kwargs.setdefault("label", self.kernel)
        if part in ("imaginary", "both"):
            ax.plot(x, trace.imag, **kwargs)
        if part in ("real", "both"):
            style = dict(kwargs)
            if part == "both":
                style["label"] = f"{kwargs['label']} (real)"
                style["ls"] = "--"
            ax.plot(x, trace.real, **style)
        if part == "both":
            ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_xlabel("Energy (eV)" if ev else "Energy (Ry)")
        ax.set_ylabel({"real": r"Re $\epsilon_M$",
                       "imaginary": r"Im $\epsilon_M$"}.get(part, r"$\epsilon_M$"))
        ax.legend()
        return ax

    @property
    def refraction(self) -> np.ndarray:
        """``Re eps_M``, averaged the same way."""
        return np.real(np.trace(np.asarray(self.epsilon), axis1=1, axis2=2)) / 3.0


def run_absorption(
    system,
    pseudos,
    density,
    frequencies,
    *,
    kernel: str = DEFAULT_KERNEL,
    kpoints=None,
    nbnd: int | None = None,
    broadening: float = 0.01,
    ecut_response: float = 4.0,
    scissor: float = 0.0,
    alpha: float | None = None,
    becsum: tuple = (),
    ns=None,
    conv_thr: float = 1.0e-10,
    k_batch: int | None | str = "default",
    static_residual: bool = True,
    tolerance: float = TOLERANCE,
    max_iterations: int = MAX_ITERATIONS,
    verbose: bool = False,
) -> OpticalSpectrum:
    """``eps_M(omega)`` for a converged insulator, on the k-grid it converged on.

    Args:
        system: the converged :class:`~pypresso.system.builder.System`.
        density: the converged density (``SCFResult.density``).
        kpoints: the k-set the spectrum is computed on, if it is not the one
            the density was converged on. It must be the **whole** grid, not a
            symmetry-reduced wedge -- see
            :func:`~pypresso.tddft.chi0.require_a_sum_over_states_regime` -- and
            that is exactly why this argument exists: a density converges
            perfectly well on a wedge, and an optical spectrum needs both a
            closed grid and a denser one than any ground state does. An
            **unshifted** grid is what makes the whole-grid route sound (P24,
            P28b); a shifted one works for ``chi_0`` itself, which symmetrises
            nothing, but the Sternheimer solve behind ``static_residual``
            refuses it, so pass ``static_residual = False`` with one.
        frequencies: ``(nw,)`` Ry, real. Two more are added internally, both at
            ``omega = 0``: one carrying ``i broadening``, which is where the
            kernel is built (``init3.f90`` puts the bootstrap's static point
            there rather than at zero), and one at ``eta = 0`` for the static
            residual. **Elk clobbers the caller's first grid point to make
            room; this appends instead**, which is the one deliberate deviation
            from ``tddftlr.f90`` here.
        kernel: ``rpa``, ``alda``, ``lrc``, ``bootstrap`` or ``bootstrap-1``.
        nbnd: how many bands the fixed-density run computes. The default is
            four times the occupied count, which is enough for a converged
            static value on silicon and is not a substitute for looking at
            :attr:`OpticalSpectrum.static_residual`.
        broadening: ``eta``, Ry -- Elk's ``swidth``. It sets the width of every
            peak, so a spectrum is only as resolved as this is small and only
            as smooth as the k-grid lets it be.
        ecut_response: the cutoff (Ry) of the G-set ``chi_0`` is a matrix over,
            Elk's ``gmaxrf``. Its own convergence parameter: too small drops the
            local-field effect, and the cost is quadratic in it.
        scissor: rigid shift (Ry) of the empty states, applied to the
            eigenvalues *and* to the velocity matrix elements.
        alpha: the ``lrc`` kernel's parameter; ignored by every other kernel.
        static_residual: whether to measure this run's band truncation against
            the Sternheimer solve. Costs one self-consistent field response.
    """
    from pypresso.scf.driver import Calculation

    rule = get_kernel(kernel)
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))

    # The refusals are checked **before** the fixed-density run, not inside
    # ``independent_response`` where they would also be checked: they are
    # statements about the calculation, and a caller asking for something this
    # cannot do should not first pay for sixty bands at every k-point.
    if kpoints is not None:
        import equinox as eqx

        system = eqx.tree_at(lambda s: s.kpoints, system, kpoints)
    require_a_sum_over_states_regime(Calculation(system, pseudos, k_batch=k_batch))

    calculation, system, eigenvalues, wavefunctions = fixed_density_states(
        system, pseudos, density, nbnd=nbnd or _default_nbnd(system, pseudos),
        conv_thr=conv_thr, k_batch=k_batch, ns=ns, becsum=becsum,
    )
    nocc = int(round(calculation.nelec / 2))
    nbnd = int(eigenvalues.shape[-1])
    potential = calculation.potential(jnp.asarray(density))

    # Two extra points, both at omega = 0: index 0 carries no broadening and is
    # the residual's, index 1 carries ``i eta`` and is the kernel's.
    grid = np.concatenate([[0.0, 0.0], frequencies])
    imaginary = np.concatenate([[0.0], np.full(len(frequencies) + 1, broadening)])
    chi = independent_response(
        calculation, wavefunctions, eigenvalues, potential.v_scf,
        grid + 1j * imaginary, ecut_response=ecut_response, broadening=0.0,
        scissor=scissor, k_batch=k_batch,
    )

    context = {}
    if rule.name == "alda":
        context["alda_matrix"] = alda_matrix(calculation, jnp.asarray(density),
                                             chi.sphere)
    if alpha is not None:
        context["alpha"] = alpha

    solution = solve_dyson(
        chi, kernel, context, static_index=1, tolerance=tolerance,
        max_iterations=max_iterations, verbose=verbose,
    )

    residual, rpa_static = None, None
    if static_residual:
        rpa_static, residual = _static_residual(
            calculation, wavefunctions, eigenvalues, density, chi,
            scissor=scissor, ecut_response=ecut_response, k_batch=k_batch,
            v_scf=potential.v_scf,
        )

    epsilon = np.asarray(solution.epsilon)[2:]
    return OpticalSpectrum(
        frequencies=frequencies,
        epsilon=epsilon,
        epsilon_no_local_fields=np.asarray(solution.epsilon_no_local_fields)[2:],
        kernel=solution.kernel,
        alpha=solution.alpha,
        alpha_elk=solution.alpha_elk,
        iterations=solution.iterations,
        converged=solution.converged,
        nbnd=nbnd,
        nocc=nocc,
        nm=chi.nm,
        scissor=float(scissor),
        static_residual=residual,
        static_rpa=rpa_static,
    )


def _default_nbnd(system, pseudos) -> int:
    """Four times the occupied count, which is a starting point and not a choice.

    There is no way to pick this correctly without looking at the answer, which
    is why :attr:`OpticalSpectrum.static_residual` exists. Four times is enough
    for a static value good to a few parts in a thousand on silicon and is not
    enough for every system.
    """
    from pypresso.scf.driver import Calculation

    nocc = int(round(Calculation(system, pseudos).nelec / 2))
    return max(4 * nocc, nocc + 8)


def _static_residual(calculation, wavefunctions, eigenvalues, density, chi, *,
                     scissor, ecut_response, k_batch, v_scf):
    """``eps_M(0)`` here in RPA, against the Sternheimer solve's RPA value.

    **Everything about this comparison has to match except the truncation**,
    which is the only thing it is meant to measure, and there are two ways to
    break that -- both silent, both giving a large residual that looks like a
    badly converged band count.

    *The kernel.* The Sternheimer dielectric constant is screened by
    ``dv_of_drho``, Hartree **plus** ``f_xc``, so differencing a bootstrap or
    ALDA number against it measures the kernel. Both sides are therefore taken
    in RPA, where the two routes are an identity of each other.

    *The Hamiltonian.* A scissors shift moves the empty states of ``chi_0`` and
    the Sternheimer solve knows nothing about it -- measured: a 0.05 Ry shift
    turns a residual of ``+0.013`` into ``-3.46``, which is the shift's own
    effect on ``eps_M`` and not a truncation at all. So a shifted run builds one
    more ``chi_0``, at a single frequency and no shift, rather than reporting a
    number that means something else. It costs a second pass over the pairs, and
    the diagnostic is worth it: this is the phase's one unrefusable error.
    """
    from pypresso.response.efield import dielectric_tensor

    nocc = int(round(calculation.nelec / 2))
    reference = dielectric_tensor(
        calculation, wavefunctions[:, :, :nocc], eigenvalues[:, :, :nocc],
        jnp.asarray(density), screening="hartree", born_charges=False,
    )
    sternheimer = float(np.diag(np.asarray(reference.epsilon)).mean())

    index = 0
    if scissor:
        chi = independent_response(
            calculation, wavefunctions, eigenvalues, v_scf, np.array([0.0]),
            ecut_response=ecut_response, broadening=0.0, scissor=0.0,
            k_batch=k_batch,
        )
    # Index 0 is the unbroadened static point; the Sternheimer solve has no
    # broadening either, so the two are comparable without a limit being taken.
    rpa = solve_dyson(chi, "rpa", {}, static_index=index)
    here = float(np.real(np.diag(np.asarray(rpa.epsilon)[index])).mean())
    return here, here - sternheimer
