"""The tunnelling density and the image made from it -- Elk's task 162.

Tersoff-Hamann: the current an s-wave tip draws at ``r`` is proportional to the
sample's local density of states there, at the energy the bias selects. So an
STM image is a *density built from different occupations*, and Elk says so in
the plainest possible way -- ``wfplot.f90`` overwrites ``occsv`` with a
normalised delta at the Fermi level and calls ``rhomagv`` again:

    rho_STM(r) = sum_kn w_k delta(E - e_kn) |psi_kn(r)|^2

with ``delta`` the same smeared delta the smearing density of states is built
from. Nothing else about the calculation changes, which is why there is no
second band sum here either: the weights go into
:meth:`~defumat.scf.driver.Calculation.density`, and the symmetrisation and the
augmentation charge come along with them.

**Two energy selections, and Elk implements only the first.**

* ``bias = None`` -- Elk's, a delta at ``E``. This is the zero-bias limit and
  the quantity is a *density of states per unit volume*, in 1/(bohr^3 Ry). It
  images a metal; on an insulator a delta in the gap is identically zero, which
  is a true statement about zero bias and a useless image.
* ``bias = V`` -- QE's ``PP/src/stm.f90``, every state between ``E`` and
  ``E + V`` counted with weight one and the two edges softened by the same
  smearing function. The quantity is a density, in electrons/bohr^3, and a
  negative ``V`` images the filled states below ``E`` -- which is the sign
  convention of a real experiment and the reason the window mode is here at all,
  a semiconductor surface being the common case.

**The delta defaults to a Gaussian** whatever smearing the run used, for
``PLAN.md`` P52's reason: Methfessel-Paxton and cold smearing go negative on
their wings, and a negative tunnelling current is not a small error but a
meaningless one. Elk's own default is Fermi-Dirac (``stype = 3``) and QE's
``stm.f90`` inherits the run's ``ngauss``; both are reachable through
``smearing``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from defumat.scf.occupations import smearing_order, w0gauss

__all__ = ["STMImage", "tunnelling_weights", "constant_current_height",
           "project_spin", "smeared_delta"]


def smeared_delta(x, smearing: str = "gaussian"):
    """The normalised delta this module images with, by name.

    Everything QE's ``w0gauss`` offers, plus a **Lorentzian**, which QE
    has no counterpart for because it is not an occupation scheme: there
    is no ``wgauss`` to go with it and it would not fill a band. It is a
    delta and only a delta, and it is here because a broadening that comes
    from a *lifetime* rather than from a numerical smearing is Lorentzian
    -- which is what the leads impose in
    :mod:`defumat.transport.green`, and what makes the Tersoff-Hamann
    limit of a vertical transmission comparable with an image from here.
    """
    x = np.asarray(x, dtype=float)
    if smearing.strip().lower() in ("lorentzian", "lorentz"):
        return 1.0 / (np.pi * (1.0 + x ** 2))
    return np.asarray(w0gauss(x, smearing_order(smearing)))


def tunnelling_weights(eigenvalues, kweights, energy: float, width: float,
                       smearing: str = "gaussian", bias: float | None = None,
                       band_cutoff: float | None = None):
    """The occupations that turn a density into a tunnelling density of states.

    Args:
        eigenvalues: ``(nspin, nk, nbnd)`` in Ry, the channel axis always there.
        kweights: ``(nk,)``, the run's own -- their sum carries the spin
            degeneracy exactly as it does everywhere else here (2 unpolarized,
            1 per polarized channel), so no factor is written down.
        energy: the energy the tip is tuned to, in Ry. The Fermi level, usually.
        width: the smearing width in Ry -- Elk's ``swidth``, QE's ``degauss``.
        smearing: which smeared delta, by the name :func:`smeared_delta`
            knows -- everything ``w0gauss`` has, plus ``'lorentzian'``.
        bias: ``None`` for the delta, or a sample bias in Ry for the window.
        band_cutoff: drop every state further than this many widths outside the
            window, QE's ``down1``/``up1``, which ``stm.f90`` fixes at 3. It is
            an approximation and not a small one for a smearing with wings --
            on QE's own fcc aluminium at ``degauss = 0.05`` the truncation is
            worth 0.4 per cent of the image -- so it is off by default and is
            here to reproduce ``pp.x`` exactly.

    Returns ``(nspin, nk, nbnd)``. Summed, it is ``D(E)`` in the delta case and
    the number of electrons in the window in the other -- which is the identity
    the tests check the whole assembly with.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if eigenvalues.ndim != 3:
        raise ValueError(f"eigenvalues must be (nspin, nk, nbnd), got {eigenvalues.shape}")
    kweights = np.asarray(kweights, dtype=float)[None, :, None]
    if width <= 0.0:
        raise ValueError(f"the smearing width must be positive, got {width}")
    smeared_delta(0.0, smearing)  # validate the name before any work

    if bias is None:
        # Elk: occsv = occmax * wkpt * sdelta((E_F - e)/swidth) / swidth. Elk's
        # ``occmax * wkpt`` is this code's ``kweights``, and the 1/width is what
        # makes it a delta rather than a step -- QE's stm.f90 leaves it out, so
        # its image is this one times ``degauss``.
        delta = smeared_delta((energy - eigenvalues) / width, smearing)
        return kweights * delta / width * _band_mask(
            eigenvalues, energy, energy, width, band_cutoff)

    low, high = (energy, energy + bias) if bias > 0 else (energy + bias, energy)
    inside = (eigenvalues > low) & (eigenvalues < high)
    # ``stm.f90``'s edges: a state outside the window is damped by the *delta*
    # rather than by the step function, undivided by the width. It is QE's
    # expression and is transcribed rather than corrected -- with a width far
    # below the window it is a soft edge either way.
    below = smeared_delta((low - eigenvalues) / width, smearing)
    above = smeared_delta((high - eigenvalues) / width, smearing)
    tails = np.where(eigenvalues <= low, below, above)
    return kweights * np.where(inside, 1.0, tails) * _band_mask(
        eigenvalues, low, high, width, band_cutoff)


def _band_mask(eigenvalues, low, high, width, band_cutoff):
    """``stm.f90``'s ``first_band``/``last_band``, or no truncation at all."""
    if band_cutoff is None:
        return 1.0
    reach = float(band_cutoff) * width
    return ((eigenvalues >= low - reach) & (eigenvalues <= high + reach)).astype(float)


def project_spin(density, direction, polarization: float = 1.0):
    """The tunnelling density a tip magnetized along ``direction`` measures.

    Spin-polarized STM: a magnetic tip does not count every state at the tip
    energy equally, it counts the ones whose spin is along its own moment. The
    tunnelling density of states is a 2x2 matrix in spin space, carried here as
    a charge and a magnetization exactly as an ordinary density is, and its
    projection on a direction ``n`` is the diagonal entry in the frame where
    ``n`` is up:

        rho(r; n) = [ rho(r) + P n.m(r) ] / 2

    ``P = 1`` is a fully polarized tip and makes the result a genuine spin
    channel -- the local density of states of the states with spin along ``n``.
    ``P = 0`` is a nonmagnetic tip and gives half the charge, which is the
    average of the two channels and is the same image the charge gives.
    ``P = -1`` is the same tip reversed, and the pair adds back to the charge.

    Args:
        density: ``(nspin_mag, ...)``. Two channels is a collinear run, whose
            magnetization has only a ``z`` component; four is ``(n, mx, my,
            mz)``.
        direction: a cartesian 3-vector, normalised here, or ``"up"``/
            ``"down"`` for a collinear run.
        polarization: the tip's spin polarization, in ``[-1, 1]``.

    A collinear run carries no transverse magnetization, so a direction with a
    component off the ``z`` axis is **refused** rather than silently projected
    onto it: ``m_x`` and ``m_y`` are not zero there, they are absent, and the
    two statements give different images.
    """
    density = np.asarray(density)
    nspin_mag = density.shape[0]
    if not -1.0 <= float(polarization) <= 1.0:
        raise ValueError(
            f"the tip polarization must be in [-1, 1], got {polarization}"
        )
    if nspin_mag == 1:
        raise NotImplementedError(
            "a spin-polarized image of a run with no magnetization is refused: "
            "there is nothing for the tip's moment to couple to, and every "
            "direction would return half the charge"
        )

    if nspin_mag == 2:
        axis = _collinear_axis(direction)
        charge = density[0] + density[1]
        magnetization = density[0] - density[1]
        return 0.5 * (charge + float(polarization) * axis * magnetization)
    if nspin_mag == 4:
        unit = _unit_vector(direction)
        charge = density[0]
        projected = np.tensordot(unit, density[1:], axes=(0, 0))
        return 0.5 * (charge + float(polarization) * projected)
    raise ValueError(f"a density with {nspin_mag} channels is not a spin density")


def _collinear_axis(direction) -> float:
    """``+1`` or ``-1``: which channel of a collinear run the tip selects."""
    if isinstance(direction, str):
        name = direction.strip().lower()
        if name in ("up", "+z", "z", "majority"):
            return 1.0
        if name in ("down", "dn", "-z", "minority"):
            return -1.0
        raise ValueError(
            f"unknown spin direction {direction!r} for a collinear run: use "
            "'up', 'down', or a vector along z"
        )
    unit = _unit_vector(direction)
    if abs(unit[2]) < 1.0 - 1.0e-8:
        raise NotImplementedError(
            f"a collinear run carries only m_z, and {tuple(unit)} has a "
            "transverse component: the transverse magnetization is absent "
            "rather than zero, so projecting on this direction would be a "
            "statement the calculation cannot make. Run noncollinear, or ask "
            "for 'up' or 'down'"
        )
    return float(np.sign(unit[2]))


def _unit_vector(direction) -> np.ndarray:
    if isinstance(direction, str):
        named = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1),
                 "up": (0, 0, 1), "down": (0, 0, -1), "dn": (0, 0, -1),
                 "-x": (-1, 0, 0), "-y": (0, -1, 0), "-z": (0, 0, -1)}
        key = direction.strip().lower()
        if key not in named:
            raise ValueError(
                f"unknown spin direction {direction!r}: use a 3-vector or one "
                f"of {sorted(named)}"
            )
        direction = named[key]
    unit = np.asarray(direction, dtype=float).reshape(-1)
    if unit.shape != (3,):
        raise ValueError(f"a spin direction is a 3-vector, got {direction!r}")
    norm = float(np.linalg.norm(unit))
    if norm < 1.0e-12:
        raise ValueError("the spin direction has zero length")
    return unit / norm


def constant_current_height(heights, values, current: float):
    """Where along a scan the tunnelling density reaches ``current``.

    Args:
        heights: ``(nz,)`` increasing, the scan coordinate -- crystal
            coordinates along the surface normal, or bohr; whatever they are,
            that is what comes back.
        values: ``(nz, n1, n2)`` the tunnelling density on the stack of planes.
        current: the set-point, in the units ``values`` carries.

    Returns ``(n1, n2)``, ``nan`` where the set-point is not crossed inside the
    scan -- above every point of the range, or below all of it.

    The interpolation is **linear in the logarithm**, because that is what the
    quantity does: the density decays as ``exp(-2 kappa z)`` into the vacuum, so
    a log-linear interpolant is nearly exact over one step of a scan and a
    linear one is not.
    """
    heights = np.asarray(heights, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.shape[0] != heights.shape[0]:
        raise ValueError(
            f"the scan has {heights.shape[0]} heights and {values.shape[0]} "
            "planes of values"
        )
    if np.any(np.diff(heights) <= 0.0):
        raise ValueError("the scan heights must be strictly increasing")
    if current <= 0.0:
        raise ValueError(f"the set-point must be positive, got {current}")

    above = values >= current
    # The *outermost* crossing: the tip is withdrawn until the current falls
    # to the set-point, so what is wanted is the largest height still above it,
    # not the first one encountered from below.
    nz = heights.shape[0]
    last = np.where(above.any(axis=0), nz - 1 - above[::-1].argmax(axis=0), -1)

    out = np.full(values.shape[1:], np.nan)
    usable = (last >= 0) & (last < nz - 1)
    if not usable.any():
        return out
    i = np.where(usable, last, 0)
    rows, cols = np.indices(values.shape[1:])
    lo, hi = values[i, rows, cols], values[i + 1, rows, cols]
    z0, z1 = heights[i], heights[i + 1]
    # log-linear: z = z0 + (z1 - z0) (ln I - ln lo)/(ln hi - ln lo)
    safe = usable & (lo > 0.0) & (hi > 0.0) & (hi < lo)
    fraction = np.where(
        safe,
        (np.log(np.where(safe, lo, 1.0)) - np.log(current))
        / (np.log(np.where(safe, lo, 1.0)) - np.log(np.where(safe, hi, np.e))),
        np.nan,
    )
    # A step that is not monotonically decaying (inside the slab, or noise at
    # the far end of the scan) falls back to a linear interpolation, which is
    # always defined where the set-point is bracketed.
    linear = np.where(usable & (lo != hi), (lo - current) / (lo - hi), np.nan)
    fraction = np.where(safe, fraction, linear)
    return np.where(usable, z0 + (z1 - z0) * fraction, np.nan)


@dataclass
class STMImage:
    """A scanning-tunnelling image and the tunnelling density behind it."""

    #: ``(n1, n2)`` in constant-height mode: the tunnelling density on the
    #: plane, in 1/(bohr^3 Ry) at zero bias and electrons/bohr^3 in a window.
    #: In constant-current mode it is the set-point, repeated -- the image is
    #: then :attr:`heights`.
    values: np.ndarray
    #: The plane the image was sampled on.
    plane: object
    #: ``(n1, n2)`` the corrugation, in the scan's own coordinate, or ``None``
    #: in constant-height mode. ``nan`` where the set-point was not reached.
    heights: np.ndarray | None = None
    #: ``(n1, n2)`` the same corrugation in bohr along the surface normal.
    heights_bohr: np.ndarray | None = None
    #: The tunnelling density on the FFT grid, channel axis squeezed when there
    #: is one -- what a comparison against ``pp.x``'s ``plot_num = 5`` is with.
    density: np.ndarray | None = None
    #: ``(nspin_mag, n1, n2)`` the density's own channels sampled on the plane
    #: -- ``(up, down)`` for a collinear run and ``(n, mx, my, mz)`` for a
    #: noncollinear one -- or ``None`` when there is only a charge.
    values_by_spin: np.ndarray | None = None
    #: The tip's moment direction, or ``None`` for an ordinary charge image,
    #: and its spin polarization.
    spin: object = None
    polarization: float = 1.0
    mode: str = "constant-height"
    #: The energy the tip was tuned to, in Ry, and the bias window if any.
    energy: float = 0.0
    bias: float | None = None
    width: float = 0.0
    smearing: str = "gaussian"
    #: ``int rho_STM d3r``: ``D(E)`` in states/Ry at zero bias, and the number
    #: of electrons in the window otherwise. The sum rule the assembly is
    #: checked against.
    integral: float = 0.0
    #: The k-set the tunnelling density was built on, ``None`` when it was the
    #: SCF's own.
    grid: tuple[int, int, int] | None = None

    @property
    def coordinates(self) -> np.ndarray:
        """``(n1, n2, 2)`` in-plane cartesian coordinates in bohr."""
        return self.plane.coordinates

    @property
    def image(self) -> np.ndarray:
        """What to plot: the density at constant height, the height otherwise."""
        return self.values if self.heights is None else self.heights_bohr

    def extent(self) -> tuple[float, float, float, float]:
        """``(x0, x1, y0, y1)`` in bohr, for ``imshow``."""
        x, y = self.coordinates[..., 0], self.coordinates[..., 1]
        return (float(x.min()), float(x.max()), float(y.min()), float(y.max()))
