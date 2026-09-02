"""Grimme's D2 dispersion correction (``vdw_corr = 'grimme-d2'``).

A semilocal functional has no van der Waals attraction: the correlation energy it
writes down is a functional of the density *where the orbitals are*, and the
London force comes from the correlated fluctuations of two densities that do not
overlap at all. Grimme's D2 (`S. Grimme, J. Comp. Chem. 27, 1787 (2006)
<https://doi.org/10.1002/jcc.20495>`_, as QE implements it after `V. Barone et
al., J. Comp. Chem. 30, 934 (2009) <https://doi.org/10.1002/jcc.21112>`_) adds it
back as a pair potential over the nuclei,

    E_disp = -(s6/2) sum_{a b R} C6_ab / |tau_a - tau_b + R|^6 f_damp(d_ab)
    f_damp(d) = 1 / (1 + exp(-beta (d / (R_a + R_b) - 1)))     beta = 20

with ``C6_ab = sqrt(C6_a C6_b)`` and both ``C6`` and the van der Waals radii
tabulated per **element** (:data:`D2_COEFFICIENTS`, Z = 1..86). The damping
switches the ``1/r^6`` off inside the sum of the two radii, where the functional
already has the correlation and where an undamped ``1/r^6`` diverges.

**It is a function of the nuclei and of nothing else.** That is the whole of its
place in the code and it is worth saying explicitly, because it is what makes
every derived quantity free: the dispersion energy does not enter ``v_of_rho``,
so the density, the potential, the eigenvalues and every response are *bit for
bit* what they would be without it (``PW/src/electrons.f90`` adds ``elondon`` to
``etot`` after the SCF loop has produced them, and ``force_london`` and
``stres_london`` are likewise added at the end). What it does change is the total
energy, the force and the stress -- and therefore a relaxation, a cell, and the
elastic constants that the electrostriction tensors are built from.

**This is the Ewald sum's twin and is written as one** (:mod:`defumat.scf.ewald`).
Both are a pair sum over the nuclei and their periodic images; both fix a
neighbour list on the host, once, so that the sum is a pure JAX function of the
positions and the force is ``jax.grad`` of it rather than a second expression;
both deform that list with the cell under a strain so the stress is ``jax.grad``
in the other coordinate. QE's ``force_london`` and ``stres_london`` are
transcribed as *cross-checks* (:mod:`defumat.forces.analytic`,
:mod:`defumat.stress.analytic`) and not as the implementation.

**``rgen``'s fold is kept, and it is not cosmetic.** QE reduces each pair's
separation into the cell at the origin before building images around it; here
one list of lattice translations serves every pair, and the separation is folded
the same way before the list is added to it. The fold is by a lattice vector, so
it permutes which translation supplies which image and changes no separation --
the sum is the same number either way, *provided the list is long enough*. What
it buys is that the list can be built to ``rcut + fold_radius(at)``, a bound that
depends on the cell alone, so the same object stays exact however far the atoms
move and wherever outside the cell they are written. Without it the list has to
reach ``rcut`` plus the largest separation the *current* geometry has, which is
:class:`~defumat.scf.ewald.EwaldSum`'s choice and is why that one consults the
positions it was built with. ``jnp.round`` has zero derivative, so the fold is
invisible to ``grad``: the integer is frozen and the lattice vector it multiplies
deforms with the cell exactly as the translations do.

**Memory, and why the cutoff is as large as it is.** The kernel broadcasts to
``(nat, nat, ntrans, 3)``, and ``ntrans`` grows as ``(4 pi/3)(rcut +
fold_radius)^3 / Omega`` -- on QE's graphite test cell, 1.6e5 translations at the
default ``london_rcut = 200`` bohr, which is 63 MB of separations for four atoms,
51 ms for the energy and 153 ms for its gradient. That looks extravagant for a ``1/r^6`` potential and is not:
the number of pairs in a shell grows as ``r^2``, so the *truncation error* falls
only as ``1/rcut^3``, and on graphite the sum is -0.039133 Ry at 30 bohr,
-0.039945 at 60, -0.039975 at 200 and -0.039975 at 300. Reaching QE's printed
1e-8 takes ~150 bohr. The default here is QE's so that an input reproduces
``pw.x`` term for term; a cell with a large vacuum has a large ``Omega`` and
therefore *fewer* translations at the same radius, so the expensive case is a
small dense cell.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from defumat.system.cell import Cell, fold_radius, lattice_translations
from defumat.system.elements import atomic_number
from defumat.system.structure import Structure
from defumat.vdw.registry import register_vdw

__all__ = ["D2_COEFFICIENTS", "GrimmeD2", "build_grimme_d2", "D2_BETA"]

#: ``(C6, R_vdw)`` per element, ``D2_COEFFICIENTS[Z - 1]``, for Z = 1..86.
#: Transcribed from ``Modules/mm_dispersion.f90``'s ``vdw_coeffs``, which is the
#: DFT-D2 section of Grimme's own ``dftd3.f`` with ``C6`` already converted from
#: J nm^6/mol to **Ry bohr^6** (a factor of 34.69) and the radii from Angstrom to
#: **bohr**. Z = 55-86 were contributed to QE by M. Andersson (2011).
D2_COEFFICIENTS: tuple[tuple[float, float], ...] = (
    (    4.857, 1.892), (    2.775, 1.912), (   55.853, 1.559), (   55.853, 2.661),  # Z = 1-4
    (  108.584, 2.806), (   60.710, 2.744), (   42.670, 2.640), (   24.284, 2.536),  # Z = 5-8
    (   26.018, 2.432), (   21.855, 2.349), (  198.087, 2.162), (  198.087, 2.578),  # Z = 9-12
    (  374.319, 3.097), (  320.200, 3.243), (  271.980, 3.222), (  193.230, 3.180),  # Z = 13-16
    (  175.885, 3.097), (  159.927, 3.014), (  374.666, 2.806), (  374.666, 2.785),  # Z = 17-20
    (  374.666, 2.952), (  374.666, 2.952), (  374.666, 2.952), (  374.666, 2.952),  # Z = 21-24
    (  374.666, 2.952), (  374.666, 2.952), (  374.666, 2.952), (  374.666, 2.952),  # Z = 25-28
    (  374.666, 2.952), (  374.666, 2.952), (  589.405, 3.118), (  593.221, 3.264),  # Z = 29-32
    (  567.896, 3.326), (  438.498, 3.347), (  432.600, 3.305), (  416.642, 3.264),  # Z = 33-36
    (  855.833, 3.076), (  855.833, 3.035), (  855.833, 3.097), (  855.833, 3.097),  # Z = 37-40
    (  855.833, 3.097), (  855.833, 3.097), (  855.833, 3.097), (  855.833, 3.097),  # Z = 41-44
    (  855.833, 3.097), (  855.833, 3.097), (  855.833, 3.097), (  855.833, 3.097),  # Z = 45-48
    ( 1294.678, 3.160), ( 1342.899, 3.409), ( 1333.532, 3.555), ( 1101.101, 3.575),  # Z = 49-52
    ( 1092.775, 3.575), ( 1040.391, 3.555), (10937.246, 3.405), ( 7874.678, 3.330),  # Z = 53-56
    ( 6114.381, 3.251), ( 4880.348, 3.313), ( 4880.348, 3.313), ( 4880.348, 3.313),  # Z = 57-60
    ( 4880.348, 3.313), ( 4880.348, 3.313), ( 4880.348, 3.313), ( 4880.348, 3.313),  # Z = 61-64
    ( 4880.348, 3.313), ( 4880.348, 3.313), ( 4880.348, 3.313), ( 4880.348, 3.313),  # Z = 65-68
    ( 4880.348, 3.313), ( 4880.348, 3.313), ( 4880.348, 3.313), ( 3646.454, 3.378),  # Z = 69-72
    ( 2818.308, 3.349), ( 2818.308, 3.349), ( 2818.308, 3.349), ( 2818.308, 3.349),  # Z = 73-76
    ( 2818.308, 3.349), ( 2818.308, 3.349), ( 2818.308, 3.349), ( 1990.022, 3.322),  # Z = 77-80
    ( 1986.206, 3.752), ( 2191.161, 3.673), ( 2204.274, 3.586), ( 1917.830, 3.789),  # Z = 81-84
    ( 1983.327, 3.762), ( 1964.906, 3.636),  # Z = 85-86
)

#: The damping function's steepness. Not an input in QE either -- it is fixed at
#: 20 in ``mm_dispersion.f90`` and in Grimme's paper.
D2_BETA = 20.0

#: ``london_s6``'s default. D2 scales the whole sum by one number per functional
#: -- 0.75 for PBE, which is what QE defaults to for every functional rather than
#: looking it up (the per-functional table is D3's, not D2's).
D2_DEFAULT_S6 = 0.75

#: ``london_rcut``'s default, in bohr, as ``input_parameters.f90`` declares it.
D2_DEFAULT_RCUT = 200.0


class GrimmeD2(eqx.Module):
    """The D2 pair sum with its neighbour list and coefficients already fixed.

    ``c6`` and ``r_sum`` are ``(nat, nat)`` rather than ``(ntyp, ntyp)``: the
    expansion costs nothing at these sizes and it removes the type index from
    the kernel, which would otherwise be a gather inside a differentiated
    function. ``translations`` is the *traced* leaf -- it is what a strain
    deforms -- and everything else is static.
    """

    #: ``(nat, nat)``, ``sqrt(C6_a C6_b)`` in Ry bohr^6.
    c6: jnp.ndarray
    #: ``(nat, nat)``, ``R_a + R_b`` in bohr.
    r_sum: jnp.ndarray
    #: ``(ntrans, 3)`` cartesian lattice translations in bohr.
    translations: jnp.ndarray
    #: The cell's lattice vectors as *rows* and their reciprocal (``inv(at)``,
    #: no ``2 pi``), which the fold needs. Traced, like the translations: they
    #: are what a strain deforms.
    lattice: jnp.ndarray
    reciprocal: jnp.ndarray
    s6: float = eqx.field(static=True, default=D2_DEFAULT_S6)
    rcut: float = eqx.field(static=True, default=D2_DEFAULT_RCUT)
    beta: float = eqx.field(static=True, default=D2_BETA)
    #: Per-*species* ``C6`` and ``R_vdw``, for the report QE prints in
    #: ``print_london``. Carried as static metadata; nothing computes with them.
    species_c6: tuple[float, ...] = eqx.field(static=True, default=())
    species_rvdw: tuple[float, ...] = eqx.field(static=True, default=())

    #: The name this correction is registered under, so a result can say which
    #: one it used without the driver keeping a parallel label.
    name: str = eqx.field(static=True, default="grimme-d2")

    def energy(self, positions: jnp.ndarray) -> jnp.ndarray:
        """The dispersion energy at ``positions`` (cartesian bohr), in Ry."""
        return _dispersion_kernel(
            positions, self.c6, self.r_sum, self.translations,
            self.lattice, self.reciprocal, self.s6, self.rcut, self.beta,
        )

    def at_cell(self, deformation: jnp.ndarray) -> "GrimmeD2":
        """The same sum in a cell deformed by ``a_i -> D a_i``.

        The neighbour list is a set of *lattice translations*, so it deforms
        exactly as the lattice vectors do and no image is gained or lost. What a
        strain does change is which of them the ``rcut`` mask keeps, and at 200
        bohr the terms there are ``C6/r^6 ~ 1e-12`` Ry.
        """
        lattice = self.lattice @ deformation.T
        return eqx.tree_at(
            lambda d: (d.translations, d.lattice, d.reciprocal),
            self,
            (self.translations @ deformation.T, lattice, jnp.linalg.inv(lattice)),
        )

    def report(self, labels) -> str:
        """QE's ``print_london`` table, for a run that wants to echo its input."""
        rows = "\n".join(
            f"        {label:<3s}      {r:7.3f}      {c:9.3f}"
            for label, c, r in zip(labels, self.species_c6, self.species_rvdw)
        )
        return (
            "Parameters for Dispersion (Grimme-D2) Correction:\n"
            "  atom      VdW radius       C_6\n" + rows
        )


def _folded(tau, reciprocal, lattice):
    """``tau_a - tau_b`` reduced into the cell centred on the origin.

    ``rgen``'s ``ds - anint(ds)``. ``jnp.round`` is a step function of the
    positions and of the cell, so its derivative is zero everywhere it is
    defined and the integer it produces is a *constant* to ``grad`` -- which is
    right: the fold subtracts a lattice vector, and a lattice vector's own strain
    derivative is carried by the ``lattice`` it multiplies.
    """
    separations = tau[:, None, :] - tau[None, :, :]
    fractional = separations @ reciprocal
    return separations - jnp.round(fractional) @ lattice


@jax.jit
def _dispersion_kernel(tau, c6, r_sum, translations, lattice, reciprocal, s6, rcut, beta):
    """``-(s6/2) sum_{a b R} C6_ab f_damp / d^6`` over every pair and image.

    :func:`defumat.scf.ewald._real_kernel`'s shape and its trap, verbatim: the
    self term ``R = 0, a = b`` and the images past the cutoff are dropped **by
    weight, not by indexing**, so the shape stays static -- and the *squared*
    distance is what is sanitised, before the square root rather than after it,
    because masking the result of ``sqrt(0)`` still leaves an infinite
    derivative to multiply by zero and ``0 * inf`` is NaN. That NaN appears only
    in the gradient, so it would survive until the day the forces were wanted.

    The damping is written as a logistic rather than as QE's guarded
    ``exp``. ``f_damp = sigmoid(beta (d/R_sum - 1))`` is the same function, and
    the argument it exponentiates is bounded below by ``-beta = -20`` for any
    non-negative distance, so the overflow ``mm_dispersion.f90`` guards against
    by testing ``dist6 < 40`` cannot occur on this side.
    """
    # (nat, nat, ntrans, 3): every pair, every image.
    separations = (
        _folded(tau, reciprocal, lattice)[:, :, None, :] + translations[None, None, :, :]
    )
    square = jnp.sum(separations**2, axis=-1)
    keep = (square > 1.0e-16) & (square <= rcut**2)
    distances = jnp.sqrt(jnp.where(keep, square, 1.0))

    damping = jax.nn.sigmoid(beta * (distances / r_sum[:, :, None] - 1.0))
    terms = jnp.where(keep, c6[:, :, None] * damping / distances**6, 0.0)
    return -0.5 * s6 * jnp.sum(terms)


@register_vdw("grimme-d2")
def build_grimme_d2(
    cell: Cell,
    structure: Structure,
    *,
    s6: float = D2_DEFAULT_S6,
    rcut: float = D2_DEFAULT_RCUT,
    c6=None,
    rvdw=None,
) -> GrimmeD2:
    """Fix the coefficients and the neighbour list for this cell and geometry.

    Args:
        cell: the unit cell, whose lattice vectors the image list is built from.
        structure: the atoms; only their *species labels* are read here, the
            positions being the argument :meth:`GrimmeD2.energy` takes.
        s6: ``london_s6``, the global scaling factor.
        rcut: ``london_rcut`` in bohr.
        c6: ``london_c6`` per species in Ry bohr^6, ``None`` or a negative
            entry meaning "take the tabulated value". QE's sentinel is -1 and it
            is honoured here, so an input's array can be passed straight
            through.
        rvdw: ``london_rvdw`` per species in bohr, same sentinel.

    The parameters are per **species**, as QE's inputs are, and are expanded to
    per-atom pairs here.
    """
    labels = [species.name for species in structure.species]
    ntyp = len(labels)
    c6_species = np.full(ntyp, -1.0) if c6 is None else np.asarray(c6, dtype=float)
    rvdw_species = np.full(ntyp, -1.0) if rvdw is None else np.asarray(rvdw, dtype=float)
    if len(c6_species) != ntyp or len(rvdw_species) != ntyp:
        raise ValueError(
            f"london_c6/london_rvdw must have one entry per species ({ntyp})"
        )

    tabulated_c6 = np.empty(ntyp)
    tabulated_r = np.empty(ntyp)
    for t, label in enumerate(labels):
        z = atomic_number(label)
        if z > len(D2_COEFFICIENTS):
            raise NotImplementedError(
                f"no Grimme-D2 coefficients for {label} (Z = {z}): the table stops "
                f"at Z = {len(D2_COEFFICIENTS)}, as QE's does. Supply london_c6 and "
                "london_rvdw for this species, or use a correction that has them"
            )
        tabulated_c6[t], tabulated_r[t] = D2_COEFFICIENTS[z - 1]

    # QE's sentinels differ between the two, and the difference is deliberate on
    # its side: ``in_C6 > -eps16`` accepts a *zero* C6 (a species excluded from
    # the correction) where ``in_rvdw > 0`` does not, a zero radius being a
    # division by zero in the damping.
    per_species_c6 = np.where(c6_species > -1.0e-16, c6_species, tabulated_c6)
    per_species_r = np.where(rvdw_species > 0.0, rvdw_species, tabulated_r)

    types = np.asarray(structure.types, dtype=int)
    atom_c6 = per_species_c6[types]
    atom_r = per_species_r[types]

    at = np.asarray(cell.at)
    # ``rcut`` plus the furthest a *folded* separation can reach, which depends
    # on the cell and not on the geometry -- see the module docstring.
    radius = rcut + fold_radius(at)
    return GrimmeD2(
        c6=jnp.asarray(np.sqrt(atom_c6[:, None] * atom_c6[None, :])),
        r_sum=jnp.asarray(atom_r[:, None] + atom_r[None, :]),
        translations=jnp.asarray(lattice_translations(at, radius)),
        lattice=jnp.asarray(at),
        reciprocal=jnp.asarray(np.linalg.inv(at)),
        s6=float(s6),
        rcut=float(rcut),
        beta=D2_BETA,
        species_c6=tuple(float(x) for x in per_species_c6),
        species_rvdw=tuple(float(x) for x in per_species_r),
    )
