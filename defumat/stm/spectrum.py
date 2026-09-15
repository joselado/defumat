"""The tunnelling spectrum: dI/dV at a point, over an energy axis.

``PLAN.md`` P90. :mod:`defumat.stm.image` gives the local density of states at
*one* tip energy, which is one picture at one bias; what an experiment takes
beside it is the other section of the same function, the curve at one place over
many biases, and on a modulated crystal it is the section that carries the
physics -- a charge density wave is a gap that opens in antiphase with the
charge maxima, and that is an energy axis at every position.

    dI/dV(r, V) = sum_{k,n} w_k delta(E_F + V - e_kn) |psi_kn(r)|^2

which is :func:`~defumat.stm.image.tunnelling_weights` at every energy of an
axis and nothing else. Two things follow, and the second is why this module
exists rather than a loop over :func:`~defumat.workflows.stm.run_stm`.

**The energy axis is nearly free, and only one of the two routes makes it so.**
An image rebuilds the whole density from the states, so an axis of ``nE``
energies costs ``nE`` images. Here the states are sampled at the tip points
**once**, giving ``a_n(r) = psi_n(r)``, and the axis is then the matrix product
``W[nE, n] |a|^2[n, r]`` -- the same shape of saving ``PERFORMANCE.md`` already
measures on the transport side, where eighty-one energies cost 8 per cent more
than one. Measured on eight cells of silicon with a 96x12 map and 41 energies,
the amplitude route is a tenth of the density route's work, and on the spectrum
a tip actually takes -- one point per cell -- a thousandth of it.

**What the two routes cost is not what they give.** The density route returns
the tunnelling density on the whole FFT box, so it can be read anywhere and can
be inverted for a constant-current height; the amplitude route knows only the
points it was handed. That is the whole of the difference between an image and a
spectrum here, and it is why the two are separate entry points rather than an
argument.

**A degenerate multiplet is safe, and it is worth saying why**, because rule D4
(``CLAUDE.md``) refuses most quantities built band by band: the weight here
depends on the state only through its eigenvalue, so inside a degenerate block
every member carries the same delta and the sum is ``delta(E - e)`` times the
trace of the block's projector, which no rotation of the block can change. A
spin projection is still a trace and is safe for the same reason. What would not
be safe is a weight that told two members of a multiplet apart.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from defumat.stm.image import tunnelling_weights

__all__ = ["STMSpectrum", "spectrum_weights", "state_densities",
           "accumulate_spectrum"]


def spectrum_weights(eigenvalues, kweights, energies, width: float,
                     smearing: str = "gaussian", bias: float | None = None,
                     band_cutoff: float | None = None):
    """``(nE, nspin, nk, nbnd)``: the image's own weights, at every energy.

    :func:`~defumat.stm.image.tunnelling_weights` called once per energy and
    stacked, so the delta, the window, the k-weights and ``band_cutoff`` have
    one implementation and not two. The axis is small -- tens of energies
    against ``nk nbnd`` states -- so nothing is gained by vectorising it, and
    what would be lost is the guarantee that a spectrum at one energy is the
    image at that energy to the last bit.
    """
    energies = np.atleast_1d(np.asarray(energies, dtype=float))
    if energies.ndim != 1:
        raise ValueError(f"the energies are an axis, got shape {energies.shape}")
    if energies.size == 0:
        raise ValueError("the energy axis is empty")
    return np.stack([
        tunnelling_weights(eigenvalues, kweights, energy=float(energy),
                           width=width, smearing=smearing, bias=bias,
                           band_cutoff=band_cutoff)
        for energy in energies
    ])


def state_densities(amplitudes, nspin_mag: int = 1):
    """``(nspin_mag, nbnd, npoints)``: what each state puts at each point.

    Args:
        amplitudes: ``(nbnd, npol, npoints)`` complex, ``psi_n`` at the tip
            points -- :func:`~defumat.basis.sample.sample_wavefunctions` on one
            k-point's block.
        nspin_mag: 1 for a charge alone, 4 for ``(n, m_x, m_y, m_z)`` from a
            spinor. A collinear run is two independent channels and passes 1
            for each of them.

    The Pauli convention is :func:`~defumat.scf.density.spinor_band_density`'s,
    taken from the one place with a ``pw.x`` number behind it rather than
    written again: ``m_x = 2 Re(conj(u) d)``, ``m_y = 2 Im(conj(u) d)``,
    ``m_z = |u|^2 - |d|^2``.
    """
    amplitudes = np.asarray(amplitudes)
    if amplitudes.ndim != 3:
        raise ValueError(
            f"amplitudes are (nbnd, npol, npoints), got {amplitudes.shape}")
    npol = amplitudes.shape[1]
    # ``Re(conj(z) z)`` and never ``abs(z)**2``: the trap CLAUDE.md lists, and
    # the same one :mod:`defumat.ultracell.density` names twice.
    if npol == 1:
        single = amplitudes[:, 0]
        return np.real(np.conj(single) * single)[None]
    if npol != 2:
        raise ValueError(f"npol is 1 or 2, got {npol}")

    up, down = amplitudes[:, 0], amplitudes[:, 1]
    up_density = np.real(np.conj(up) * up)
    down_density = np.real(np.conj(down) * down)
    if int(nspin_mag) == 1:
        # A spin-orbit run carrying no magnetization: the charge is the only
        # component, exactly as every routine above this one treats it.
        return (up_density + down_density)[None]
    cross = np.conj(up) * down
    return np.stack([up_density + down_density,
                     2.0 * np.real(cross),
                     2.0 * np.imag(cross),
                     up_density - down_density])


def accumulate_spectrum(total, weights, amplitudes, channel: int | None = None,
                        nspin_mag: int = 1):
    """``total[c] += sum_n W[E, n] rho_c[n, r]`` for one k-point, in place.

    ``channel`` is the collinear block this k-point belongs to, or ``None`` when
    the state carries every component itself -- which is the one structural
    difference between a spinor run and a collinear one and is the same split
    :func:`~defumat.ultracell.states.ultracell_band_density` makes.
    """
    densities = state_densities(amplitudes, nspin_mag=nspin_mag)
    contribution = np.einsum("en,cnp->cep", np.asarray(weights, dtype=float),
                             densities, optimize=True)
    if channel is None:
        total += contribution
    else:
        total[int(channel)] += contribution[0]
    return total


@dataclass
class STMSpectrum:
    """dI/dV over an energy axis, at a set of tip positions."""

    #: ``(nE, npoints)`` or ``(nE, n1, n2)`` -- the tunnelling density of states
    #: at each energy and each place, in 1/(bohr^3 Ry). With ``bias`` set it is
    #: the window's electron count instead, in electrons/bohr^3.
    values: np.ndarray
    #: ``(nE,)`` the tip energies in Ry, ``E_F + V``.
    energies: np.ndarray
    #: ``(npoints, 3)`` the tip positions in **crystal** coordinates, always
    #: there; ``plane`` is the plane they came from, or ``None`` when they were
    #: given explicitly.
    points: np.ndarray
    plane: object = None
    #: ``(nspin_mag, nE, npoints)`` the channels behind :attr:`values` --
    #: ``(up, down)`` for a collinear run and ``(n, m_x, m_y, m_z)`` for a
    #: spinor one -- or ``None`` when there is only a charge.
    values_by_spin: np.ndarray | None = None
    #: ``(nE,)`` the spectrum integrated over the cell at each energy: ``D(E)``
    #: per unit cell, in states/Ry. The sum rule the assembly is checked with.
    integral: np.ndarray | None = None
    #: The tip's moment direction and polarization, or ``None``.
    spin: object = None
    polarization: float = 1.0
    width: float = 0.0
    smearing: str = "gaussian"
    bias: float | None = None
    fermi_energy: float | None = None
    #: ``(n1, n2, n3)`` how many unit cells the spectrum spans, for an
    #: ultracell, and ``None`` for an ordinary run.
    supercell: tuple[int, int, int] | None = None
    notes: dict = field(default_factory=dict)

    @property
    def bias_axis(self) -> np.ndarray:
        """``(nE,)`` the sample bias in Ry, the axis an experiment plots on.

        The energies measured from the Fermi level, which is what ``V`` means;
        it is the energies themselves when the run has no Fermi level to
        measure from.
        """
        if self.fermi_energy is None:
            return np.asarray(self.energies, dtype=float)
        return np.asarray(self.energies, dtype=float) - float(self.fermi_energy)

    @property
    def current(self) -> np.ndarray:
        """``I(V)``: the spectrum integrated from the Fermi level outwards.

        In the Tersoff-Hamann limit the current is the local density of states
        integrated over the window the bias opens, so within this
        approximation -- a tip whose own density of states is flat and a
        transmission that does not depend on energy -- it is the cumulative
        integral of what :attr:`values` holds, with the sign of ``V``:
        positive bias counts the empty states above ``E_F`` and negative bias
        the filled ones below it, which is the convention an experiment uses.

        The units are the window's, electrons/bohr^3, the same
        ``bias=`` counts. What it is **not** is ``bias=``'s own number to the
        last digit: ``tunnelling_weights`` damps a state outside the window by
        the delta's *value* undivided by the width, which is ``stm.f90``'s
        expression transcribed rather than corrected, where integrating the
        delta gives its cumulative one -- 0.564 against 0.5 for a Gaussian level
        exactly on an edge. The two agree once every level is a few widths clear
        of both edges: measured on a modulated two-cell silicon ultracell at
        **8.5e-2, 3.1e-3 and 4.0e-9** as the nearest level moves 0.4, 1.6 and
        4.0 widths away from the edge. The integral is the physical one of the
        two.

        **It is the trapezoid of the axis that was asked for**, so it is only as
        good as that axis is dense, and it is a statement about this
        approximation rather than about a measured current: neither the tip's
        own spectrum nor the bias dependence of the barrier is in it. The decay
        of each state into the vacuum *is* in it, since that is what
        ``|Psi(r)|^2`` carries at the tip, and the prefactor is not fixed --
        this is a shape, as every tunnelling quantity here is.
        """
        if self.bias is not None:
            raise ValueError(
                "this spectrum is already a window count rather than a "
                "dI/dV, so integrating it again is not a current: run it "
                "without bias= to get the conductance whose integral this is"
            )
        if self.fermi_energy is None:
            raise ValueError(
                "this spectrum has no zero of bias -- the run has neither a "
                "Fermi level nor a gap to put one in -- and a current is an "
                "integral outwards from V = 0, so there is nothing to start "
                "it at. Integrate the values against the energies yourself, "
                "from wherever the tip is referenced"
            )
        values = np.asarray(self.values, dtype=float)
        axis = self.bias_axis
        # **A trapezoid cannot integrate a delta it does not resolve**, and the
        # failure is silent and enormous: one smeared level of width ``w`` on an
        # axis of step ``h`` integrates to 1.000000 at ``h = w/2``, 1.00004 at
        # ``h = w``, 1.14 at ``2w``, 0.10 at ``4w`` and exactly zero by ``250w``
        # -- which is what a 41-point axis over 0.1 Ry does against the 1e-5
        # width a fixed-occupation run gets by default.
        step = float(np.max(np.diff(axis))) if axis.size > 1 else 0.0
        # One point per width is the boundary and it passes: a level read at
        # exactly that step integrates to 1.00004. The tolerance is there so
        # that an axis built by ``linspace`` to land on it does not fail on the
        # last bit of the division.
        if step > float(self.width) * (1.0 + 1.0e-8):
            raise ValueError(
                f"the energy axis steps {step:.3e} Ry where the smeared delta "
                f"is {self.width:.3e} Ry wide, so the trapezoid steps over the "
                "levels rather than integrating them and the current would be "
                "wrong by orders rather than by per cent. Use at least one "
                "point per width -- or a wider width, which is what a spectrum "
                "meant to be integrated wants"
            )
        if axis.min() > 0.0 or axis.max() < 0.0:
            warnings.warn(
                "the energy axis does not include V = 0, so this current is "
                "integrated from the end of the axis nearest the Fermi level "
                f"({min(abs(axis.min()), abs(axis.max())):.4f} Ry away) rather "
                "than from zero bias: the states in between are missing from "
                "every value of it",
                stacklevel=2,
            )
        below, above = axis <= 0.0, axis >= 0.0
        out = np.zeros_like(values)
        # Integrate outwards from V = 0 in both directions, so that I(0) = 0
        # exactly and the filled side comes back negative.
        if above.sum() > 1:
            out[above] = _cumulative(axis[above], values[above])
        if below.sum() > 1:
            flipped = _cumulative(-axis[below][::-1], values[below][::-1])
            out[below] = -flipped[::-1]
        return out

    @property
    def spectrum(self) -> np.ndarray:
        """``(nE, npoints)`` whatever the shape of the sampling was."""
        values = np.asarray(self.values)
        return values.reshape(values.shape[0], -1)


def _cumulative(x, y):
    """Cumulative trapezoid of ``y`` along the first axis, starting at zero."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    steps = np.diff(x)
    shape = (1,) * (y.ndim - 1)
    areas = 0.5 * steps.reshape((-1,) + shape) * (y[1:] + y[:-1])
    return np.concatenate([np.zeros((1,) + y.shape[1:]), np.cumsum(areas, axis=0)])
