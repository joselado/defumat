"""Berry curvature and the Chern number.

Two implementations, behind the registry in :mod:`pypresso.topology.registry`,
and they answer two different questions.

``fhs`` -- the default, and the only one an *invariant* may use.
    Fukui, Hatsugai and Suzuki's lattice field strength (J. Phys. Soc. Jpn. 74,
    1674 (2005)). Each plaquette of the mesh carries the phase of a product of
    four link variables, taken on the principal branch, and the Chern number is
    their sum over the zone divided by ``2 pi``. Two properties make it the
    right tool: it is built from determinants of overlaps, so it is invariant
    under any unitary mixing inside the occupied manifold and needs no gauge
    fixing and no band-by-band identification; and the sum is an **exact
    integer on any mesh**, because every link phase appears once with each sign
    and what is left is a winding number. A coarse mesh gives the wrong integer,
    never a non-integer.

``kubo`` -- for a smooth curvature, and never for an integer.
    The sum-over-states expression

        Omega(k) = -2 Im sum_{n occ} sum_{m unocc}
                   <n|v_1|m><m|v_2|n> / (E_n - E_m)^2

    with the velocity operator ``v_mu = dH/dk_mu`` from ``jax.jacfwd`` of the
    Hamiltonian -- PLAN.md D2's stated intent, and the only thing here that
    differentiates anything. It is pointwise, so it resolves a curvature
    hotspot that a coarse plaquette would average over, and it is what a
    publication-quality ``Omega(k)`` map wants. It is also degenerate-band
    singular (D4) and its Brillouin-zone sum is an ordinary Riemann sum, which
    converges to an integer without ever being one.

**Measured, not assumed** (``tests/unit/test_topology_curvature.py``): on the
Haldane model at ``t2 = 0.2, phi = pi/2, mass = 0``, the link construction
returns ``C = -1`` to ``2e-16`` on a 6x6 mesh and on every finer one. The Kubo
sum is off by 8.6e-3 at 6x6, 1.7e-5 at 12x12 and 4.8e-11 at 24x24 -- it
converges *spectrally*, because a smooth periodic integrand on a uniform mesh
is what a trapezoidal rule is best at, and on a gapped model with no curvature
hotspot it is an excellent approximation. It is still an approximation, and
what sets its error is the sharpness of ``Omega``: near a gap closing the
curvature concentrates and the convergence collapses to algebraic, exactly
where an invariant matters most. The lattice construction does not have an
error to converge.

**``kubo`` runs on a plane-wave calculation too**, as of P47 --
:mod:`pypresso.topology.kubo`. What it once refused for ("``v = dH/dk`` needs
``d vkb/dk``, and the sphere gains and loses members with ``k``") is what P24
wrote: :class:`~pypresso.response.velocity.VelocityOperator` is one ``jvp`` of
``H(k)`` at a **frozen** sphere, which is exact on each piece of a membership
that is piecewise constant in ``k``. The sum is written as band matrix elements
between the states an NSCF already produced -- ``H`` is never formed as a
matrix, which for a plane-wave basis would be the ``npw^2`` a dense solve is a
test fixture for. It carries the ``e_n dS/dk`` term of the generalised
eigenproblem, and an ultrasoft or PAW dataset is nonetheless **refused by
name**, because that term is identically zero for a norm-conserving one and so
no norm-conserving validation can see it. The truncation of the sum over empty
states is reported rather than tuned away.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from pypresso.topology.links import berry_phase, link_phase
from pypresso.topology.mesh import PlaneMesh
from pypresso.topology.registry import register_curvature_method

__all__ = [
    "BerryCurvature",
    "berry_curvature",
    "fhs_curvature",
    "kubo_curvature",
    "link_variables",
    "plaquette_flux",
]


@dataclass
class BerryCurvature:
    """The curvature on a plane mesh, and its Brillouin-zone integral."""

    mesh: PlaneMesh
    #: ``(n1, n2)`` in units of (crystal k-area)^-1, so that the mean times
    #: ``2 pi`` is the Chern number. Multiply by the reciprocal cell's area to
    #: put it in bohr^2, which is what a plot usually wants.
    curvature: np.ndarray
    #: ``(n1, n2)`` the plaquette phases themselves, for the ``fhs`` method:
    #: the curvature times the plaquette area, each in ``(-pi, pi]``.
    flux: np.ndarray | None = None
    method: str = "fhs"
    #: ``(n1, n2, nocc)`` the curvature band by band, for the ``kubo`` method.
    #: **Gauge invariant only for a non-degenerate band**: inside a degenerate
    #: multiplet the eigensolver's arbitrary rotation moves the members'
    #: values and only their sum is defined. :attr:`curvature` never has that
    #: problem -- it is built from occupied/empty pairs alone.
    curvature_by_band: np.ndarray | None = None
    #: How many bands the ``kubo`` sum over empty states ran over.
    nbnd: int | None = None
    #: How many of them are occupied.
    nocc: int | None = None
    #: The ``kubo`` sum's truncation, as P37 reports ``static_residual``: the
    #: largest shift in ``Omega(k)`` when the **highest** empty band is dropped
    #: from the sum, divided by ``max |Omega|``. A number to read, not a knob:
    #: the sum stops where the eigensolver stopped, and this says what that
    #: cost. ``None`` for ``fhs``, which has no sum over states. Where
    #: symmetry forces ``Omega`` to vanish the ratio is noise over noise and
    #: means nothing -- read :attr:`truncation_abs` there instead (measured on
    #: silicon: ``truncation = 0.98`` beside a ``truncation_abs`` of 3.4e-5).
    truncation: float | None = None
    #: The same shift, unnormalised, in the units of :attr:`curvature`.
    truncation_abs: float | None = None

    def plot(self, ax=None, cmap: str = "RdBu_r", colorbar: bool = True,
             **kwargs):
        """Draw ``Omega(k)`` over the mesh plane, and return the axes.

        The map is what a curvature is looked at for: where the hot spots are,
        whether they are signed, and whether the mesh resolves them. It is
        drawn on the plane's two fractional coordinates, symmetric about zero
        so that a sign change reads as a colour change rather than as a
        brightness one.

        matplotlib is imported here rather than at module scope: it is not a
        dependency of any calculation, and a headless run should not need it.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        field = np.asarray(self.curvature)
        limit = float(np.max(np.abs(field)))
        kwargs.setdefault("vmin", -limit)
        kwargs.setdefault("vmax", limit)
        image = ax.imshow(field.T, origin="lower", extent=(0.0, 1.0, 0.0, 1.0),
                          cmap=cmap, aspect="auto", **kwargs)
        ax.set_xlabel("$k_1$   [fraction of the mesh plane]")
        ax.set_ylabel("$k_2$   [fraction of the mesh plane]")
        if colorbar:
            ax.figure.colorbar(image, ax=ax, label=r"$\Omega(k)$")
        return ax

    @property
    def chern_number(self) -> float:
        """``(1 / 2 pi) int Omega d^2k`` -- an exact integer for ``fhs``."""
        n1, n2 = self.mesh.shape
        return float(np.sum(self.curvature) / (n1 * n2 * 2.0 * np.pi))

    @property
    def max_flux(self) -> float:
        """The largest plaquette phase.

        FHS's own admissibility condition: the construction is exact only while
        every plaquette phase stays inside ``(-pi, pi)``, and one approaching
        ``pi`` means the mesh is too coarse to resolve a curvature hotspot. An
        integer computed past that point is a wrong integer, not a noisy one.
        """
        if self.flux is None:
            n1, n2 = self.mesh.shape
            return float(np.max(np.abs(self.curvature)) / (n1 * n2))
        return float(np.max(np.abs(self.flux)))


def link_variables(states, mesh: PlaneMesh, k_batch="default"):
    """The two families of link phases of a closed plane mesh.

    Returns ``(U1, U2)``, each ``(n1, n2)`` complex of unit modulus:
    ``U_d[i, j] = det M / |det M|`` for the overlap between mesh point
    ``(i, j)`` and its neighbour along direction ``d``.

    The pairs are grouped by direction and handed to
    :meth:`~pypresso.topology.states.StateSet.overlaps` a direction at a time,
    which is what lets the ultrasoft ``q_ij(b)`` be built once per direction
    (the step is the same at every point of a uniform mesh, wrap included).
    """
    if states.nk != mesh.nk:
        raise ValueError(
            f"the state set has {states.nk} k-points and the mesh {mesh.nk}; "
            "they must be the same points in the same order"
        )
    n1, n2 = mesh.shape
    links = []
    for direction in (0, 1):
        pairs = []
        for i in range(n1):
            for j in range(n2):
                target, shift = mesh.neighbour(i, j, direction)
                pairs.append((int(mesh.index(i, j)), target, shift))
        matrices = states.overlaps(pairs, k_batch=k_batch)
        links.append(link_phase(matrices).reshape(n1, n2))
    return links[0], links[1]


def fhs_curvature(states, mesh: PlaneMesh, k_batch="default") -> BerryCurvature:
    """Berry curvature by the lattice field strength of FHS.

    The plaquette anchored at ``(i, j)`` is

        w = U_1(i, j) U_2(i+1, j) / [ U_1(i, j+1) U_2(i, j) ]

    and its flux is ``-arg(w)`` -- the sign fixed once, in
    :func:`pypresso.topology.links.berry_phase`. Both neighbour indices wrap,
    which is what makes the ``n1 * n2`` plaquettes tile the zone exactly.
    """
    if not all(mesh.closed):
        raise ValueError(
            "the Chern number is an integral over a closed surface; this mesh "
            "is open in at least one direction"
        )
    u1, u2 = link_variables(states, mesh, k_batch=k_batch)
    n1, n2 = mesh.shape
    flux = plaquette_flux(u1, u2)
    return BerryCurvature(
        mesh=mesh, curvature=flux * n1 * n2, flux=flux, method="fhs"
    )


def plaquette_flux(u1: jnp.ndarray, u2: jnp.ndarray) -> np.ndarray:
    """The FHS lattice field strength from the two families of link phases.

        w(i, j) = U_1(i, j) U_2(i+1, j) / [ U_1(i, j+1) U_2(i, j) ],
        F(i, j) = -arg w(i, j).

    Written separately from :func:`fhs_curvature` because it is where the
    orientation and the sign live, and both are pinned by tests that feed it
    link phases directly rather than states: a phase placed on the numerator
    link ``U_1(0, 0)`` must come back as ``-theta``, and the same phase on the
    denominator link ``U_1(0, 1)`` as ``+theta``. Gauge invariance cannot catch
    a sign error -- the conjugate construction is equally gauge invariant -- so
    it has to be pinned directly.
    """
    product = u1 * jnp.roll(u2, -1, axis=0) / (jnp.roll(u1, -1, axis=1) * u2)
    return np.asarray(berry_phase(product))


def kubo_curvature(states, mesh: PlaneMesh, nocc: int | None = None, **_) -> BerryCurvature:
    """Berry curvature from the velocity operator, band by band.

    Two routes, one expression. A :class:`~pypresso.topology.states.ModelStates`
    carries ``H(k)`` as a differentiable dense matrix, so the whole thing is
    ``jacfwd`` plus an ``eigh``. A :class:`~pypresso.topology.states.
    PlaneWaveStates` carries a :class:`~pypresso.response.velocity.
    VelocityOperator` instead and nothing dense is formed --
    :mod:`pypresso.topology.kubo`.

    ``k_batch`` and the rest of the caller's keywords are swallowed by ``**_``
    on the model path, which has no k-axis walk of its own.
    """
    axes = _plane_directions(mesh)
    hamiltonian = getattr(states, "hamiltonian", None)
    if hamiltonian is None:
        from pypresso.topology.kubo import plane_wave_kubo
        from pypresso.topology.states import PlaneWaveStates

        if isinstance(states, PlaneWaveStates):
            # No dense ``H(k)`` to differentiate, and none is built:
            # :mod:`pypresso.topology.kubo` contracts one ``jvp`` of the
            # Hamiltonian against the states instead.
            return plane_wave_kubo(states, mesh, axes, nocc=nocc, **_)
        raise NotImplementedError(
            "the Kubo route needs either H(k) as a differentiable function of "
            "k (ModelStates) or the velocity operator of a plane-wave "
            "calculation (PlaneWaveStates); a bare coefficient array is "
            "neither, since nothing in it says how the states depend on k. "
            "Use the 'fhs' method, which needs only the overlaps"
        )
    nocc = states.nbnd if nocc is None else int(nocc)
    points = jnp.asarray(mesh.points.reshape(-1, 3))
    values = jax.vmap(lambda k: _kubo_point(hamiltonian, k, nocc, axes))(points)
    curvature = np.asarray(values).reshape(mesh.shape)
    return BerryCurvature(mesh=mesh, curvature=curvature, flux=None, method="kubo")


def _plane_directions(mesh: PlaneMesh) -> tuple[int, int]:
    """Which crystal directions the mesh's two spans point along."""
    d1 = int(np.argmax(np.abs(mesh.span1)))
    d2 = int(np.argmax(np.abs(mesh.span2)))
    return d1, d2


def _kubo_point(hamiltonian, k, nocc: int, axes):
    """``Omega_12(k)`` summed over the lowest ``nocc`` bands."""
    d1, d2 = axes
    jacobian = jax.jacfwd(hamiltonian)(k)  # (dim, dim, 3)
    v1, v2 = jacobian[..., d1], jacobian[..., d2]
    energies, vectors = jnp.linalg.eigh(hamiltonian(k))
    a1 = vectors.conj().T @ v1 @ vectors
    a2 = vectors.conj().T @ v2 @ vectors
    gap = energies[:, None] - energies[None, :]
    # The n = m terms cancel in the antisymmetrised product; masking them is
    # what keeps the 1/gap^2 finite. Degeneracies *between* an occupied and an
    # empty band are a genuine singularity and are left to blow up (D4): a
    # Kubo curvature there has no value to return.
    weight = jnp.where(jnp.abs(gap) > 1.0e-12, 1.0 / jnp.where(gap == 0, 1.0, gap) ** 2, 0.0)
    occupied = jnp.arange(energies.shape[0]) < nocc
    mask = occupied[:, None] & ~occupied[None, :]
    terms = jnp.where(mask, a1 * a2.T * weight, 0.0)
    return -2.0 * jnp.imag(jnp.sum(terms))


def berry_curvature(
    states, mesh: PlaneMesh, method: str | None = None, **kwargs
) -> BerryCurvature:
    """Berry curvature on a plane mesh, by the named method."""
    from pypresso.topology.registry import get_curvature_method

    return get_curvature_method(method)(states, mesh, **kwargs)


register_curvature_method("fhs", fhs_curvature)
register_curvature_method("kubo", kubo_curvature)
