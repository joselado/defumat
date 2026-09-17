"""The total energy of an ultracell, which neither reference code computes.

``PLAN.md`` P88 stage 4. Elk's ``energyulr.f90`` is four lines and computes the
occupied eigenvalue sum alone; ``writeengyu.f90`` prints that and the Fermi
energy. ``pw.x`` has no ultracell at all. So an ultracell run could say what a
modulation *does* and not whether the modulation is the ground state, and the
second question is the one a spin density wave, a domain wall or a charge
density wave is actually about.

**The assembly is the unit cell's, term for term, and the reason is that the
ultracell matrix is an exact Rayleigh-Ritz.** ``H_cell`` is lattice periodic, so
it conserves crystal momentum modulo a *unit-cell* ``G`` and cannot couple two
different ``Q``: the diagonal block is exactly ``eps_{k+Q,n}``, and the frozen
eigenvalue carries the whole unit-cell potential. ``dV`` then corrects that to
the ultracell's, so

    eps_j = <Psi_j| T + v_loc + v_NL + v_H[rho] + v_xc[rho] + v_ext |Psi_j>,

and the band sum double-counts the Hartree and exchange-correlation energies
exactly the way an ordinary SCF's does. Hence

    E = (eband + deband) + E_H + E_xc + E_Ewald + demet,

per unit cell, with ``deband = -int rho v_scf`` over the ultracell.

**Which potential ``deband`` pairs with is the whole accuracy of the term, and
it is the input one.** ``eband`` carries ``v[rho_in]`` inside every eigenvalue,
because that is the potential the matrix was built from, so ``eband + deband``
is the one-electron term only if the same potential appears in both -- and then
it cancels identically, leaving ``T + int rho v_loc + E_NL``, which is what the
Hartree and exchange-correlation energies at ``rho_out`` are then added to.
Measured on silicon at ``N = 1``, where the loop converges in one iteration and
the two densities differ by the two Davidson solves: paired with ``v[rho_in]``
the total reproduces the unit cell's own SCF to **2.3e-14 Ry**, and paired with
``v[rho_out]`` it is **2.9e-7 Ry** out. A factor of 10^7, which is why this is
assembled inside the loop rather than from the result afterwards.

**The two applied fields get opposite treatment.** A scalar ``external``
potential stays in the total, once, through ``eband``: that is ``vltot``'s
convention, QE's total includes ``int rho v_loc``, and it is what makes the
comparison against a supercell run through ``with_external_potential`` --
which puts the modulation in ``vltot`` -- like for like. A ``magnetic_field``
does not: this package's convention, which is QE's and Elk's, is that a field's
Zeeman energy is carried beside the total rather than inside it
(``scf/fields.py``, ``SCFResult.field_energy``). The field is inside ``dV`` and
therefore inside every eigenvalue, so what keeps it out is pairing it into
``deband`` as well -- which is exactly what ``run_scf`` does, ``v_field`` being
part of the ``v_scf`` its own ``deband`` integrates against
(``scf/driver.py:2812``).

**And that convention takes the variational bound with it, which is the price of
it and is not obvious.** What the calculation minimises is the *full* energy,
the Zeeman term included; the reported total is that minus ``int B . m``, so it
is a bound on nothing, and it was measured going the wrong way: on a
partly-polarized hydrogen cell under a uniform 0.02 Ry field, ``total_energy``
sits 7.4e-07 Ry above an ordinary ``run_scf`` at ``nbnd = 12`` and **9.7e-07 and
1.6e-06 Ry below it** at 24 and 40 -- monotone in the wrong direction. Add
``field_energy`` back and the bound returns: +4.15e-06, +2.47e-06, +5.15e-07,
above at every rung and falling. So the quantity to compare across ``nbnd``,
or between two states, under a field is ``total_energy + field_energy``.

**The Ewald energy is the unit cell's, unchanged**, because the atoms do not
move: an ``N``-cell supercell's is ``N`` times it. Measured, and it holds to
7.1e-15 Ry where the two cells pick the same screening parameter and to 7.2e-9
where they do not -- which is ``ewald_alpha``'s own tolerance, QE's 1e-7 on
``upperbound`` stepped in 0.1 (``scf/ewald.py:44``), rather than anything about
the claim.

**What the energy buys that no other quantity in this phase does: a sign.** The
density ladder and the eigenvalue ladder carry none, and ``PLAN.md`` P88 says
so -- this Hamiltonian moves with its own truncated density, so an ultracell
eigenvalue is not an upper bound on the supercell's. The energy is different.
The Kohn-Sham free energy minimised over a family of subspaces that are *nested*
in ``nbnd`` is a monotone non-increasing upper bound on the supercell's own, and
it was measured to be: two-cell silicon under an applied modulation reads
+8.15e-05, +3.52e-06, +3.62e-07 and +1.43e-07 Ry above a real four-atom
supercell at ``nbnd = 12, 24, 48, 64``. Under a **magnetic field** the bounded
quantity is ``total_energy + field_energy`` rather than the total, for the
reason above.

That claim has two preconditions and both are real. The two calculations must
discretise the **same** functional, so their FFT boxes must be the same box: on
the unmatched pair the silicon ladder goes *below* the supercell at the third
rung, by a tenth of the 1.08e-6 Ry the box difference is worth. And the frozen
states must be **eigenstates**, since the matrix is ``delta eps + <psi|dV|psi>``
and that is the Rayleigh-Ritz matrix only if they are. Davidson returns Ritz
vectors by construction -- ``evc`` is a rotation of the trial set
(``solvers/davidson.py:422``) -- so ``diag(eps)`` is the exact projected
``H_cell`` at any ``states_conv_thr``, and a loose one only makes the span
slightly worse, which the argument tolerates at second order.

**Everything here syncs to the host, and that is deliberate rather than
careless.** Every term is returned as a Python ``float``, which would kill a
gradient silently if anything differentiated it -- ``CLAUDE.md``'s
``np.asarray`` trap. Nothing does: the ultracell loop is Python and
``grid.py``'s own docstring records that nothing in the subpackage crosses a
``jit`` or a ``grad``, because the atoms do not move and there is no force to
take. A future derivative of this energy -- a `dE/dq` for a modulation
wavevector, say -- would have to take the terms as arrays first, and it would
find the sync rather than a wrong answer, because a traced value cannot be
turned into a ``float`` at all.

There is a third precondition that is not a tolerance either: the free energy
has to be *minimised* at the self-consistent point, and **Methfessel-Paxton
smearing breaks that** -- its occupation function is non-monotone, so the
generalised entropy is not concave in the occupations. Gaussian, Fermi-Dirac
and Marzari-Vanderbilt are safe. The energy is computed either way; what is
lost is the sign.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from defumat.scf.occupations import smearing_entropy
from defumat.ultracell.grid import Ultracell

__all__ = ["ultracell_energy", "ultracell_entropy", "total_of"]


def ultracell_entropy(eigenvalues, weights, fermi, ultracell: Ultracell,
                      system) -> float:
    """QE's ``demet``, the ``-TS`` a smeared ultracell carries, per unit cell.

    The search that found ``fermi`` was done for the **ultracell's** electron
    count with the ``k0`` weights as they are (``driver._occupy``), so the
    entropy is the ultracell's too and is divided by ``N`` afterwards -- the
    same rescaling, for the same reason, and getting it the other way round is
    a factor of ``N`` in a term that is small enough to hide one.

    It is what makes a smeared total energy variational: the quantity a
    smeared calculation minimises is a free energy, not the energy at
    fictitious occupations, and the monotone bound above is a statement about
    that free energy.
    """
    entropy = smearing_entropy(
        jnp.asarray(eigenvalues), jnp.asarray(weights), fermi,
        float(system.degauss), system.smearing or "gaussian",
    )
    return float(entropy) / ultracell.cells


def ultracell_energy(
    density_out: jnp.ndarray,
    potential_in,
    potential_out,
    band_energy: float,
    ultracell: Ultracell,
    cell,
    ewald: float,
    magnetic_potential=None,
    entropy: float = 0.0,
    dispersion: float = 0.0,
    onecentre: float = 0.0,
    paw_deband: float = 0.0,
) -> dict:
    """The energy terms of one ultracell iteration, per unit cell, in Ry.

    Args:
        density_out: ``(nspin_mag, *box)``, the density the states just built,
            normalised per unit cell.
        potential_in: the :class:`~defumat.ultracell.potential.UltracellPotential`
            the matrix was built from -- the one ``band_energy`` carries.
        potential_out: the same object rebuilt at ``density_out``, which is
            where ``E_H`` and ``E_xc`` are taken. Evaluating them here rather
            than at the mixed density is what makes QE's ``descf`` unnecessary:
            the number this returns is the Kohn-Sham energy of the state the
            iteration actually produced, at every iteration and not only at the
            last.
        band_energy: ``sum_j w_j eps_j``, per unit cell, Elk's ``evalsum``.
        ewald: the **unit cell's** Ewald energy.
        magnetic_potential: the applied field's contribution to ``dV``, shaped
            like the density, or ``None``. Its energy is removed from the total
            and returned as ``field_energy``.
        entropy: ``demet``, per unit cell, from :func:`ultracell_entropy`.
        dispersion: Grimme's D2 pair sum, the **unit cell's**, for the Ewald
            term's reason exactly: it is a sum over the nuclei outside
            ``v_of_rho`` and the nuclei do not move, so an ``N``-cell
            supercell's is ``N`` times it. The ultracell refuses no van der
            Waals correction -- ``require_an_ultracell_regime`` has no check for
            one -- so a D2 unit cell reaches here, and dropping the term would
            leave its total short by a real energy rather than by a convention.
        onecentre: PAW's one-centre energy, **per unit cell** -- the sum over
            the ``N`` copies divided by ``N``, at the output ``becsum``. It
            enters the total whole, exactly as ``one_center_paw`` does in the
            unit cell, and what removes the part already inside every eigenvalue
            is ``paw_deband`` rather than any subtraction here.
        paw_deband: ``-(1/N) sum_R sum_ij ddd^R_ij becsum^R_ij``, the one-centre
            half of ``delta_e``. It is folded into the one-electron term beside
            the grid's ``deband`` because that is what it is: the one-centre
            potential is inside every eigenvalue through ``D_ij``, so ``eband``
            double-counts it exactly the way it double-counts ``int rho v``.

    Returns a dict whose non-underscored entries sum to the total energy, in
    the shape ``SCFResult.energy_terms`` uses, plus ``_field_energy``, which is
    deliberately *not* summed.
    """
    density_out = jnp.asarray(density_out)
    cells = ultracell.cells
    # The element of volume is the same on the box as on the unit cell's own
    # grid -- ``Omega_u / N_box = Omega_cell / N_cell`` -- so this is the unit
    # cell's ``volume / size`` written on the ultracell, and the ``/ cells``
    # below is what turns an integral over the ultracell into one per cell.
    element = ultracell.volume(cell) / density_out[0].size

    v = potential_in.v_scf
    field_energy = None
    if magnetic_potential is not None:
        # ``int rho . v_field`` over the components *is* ``-int B . m``,
        # collinear (``v_up = -B``, ``v_dw = +B``) and spinor (``v(2:4) = -B``)
        # alike, so the contraction is written once and neither regime needs a
        # sign of its own. Pairing it into ``deband`` removes it from the total,
        # which is the convention; reporting it separately is what makes the
        # convention visible rather than merely applied.
        v = v + jnp.asarray(magnetic_potential)
        field_energy = element * float(
            jnp.sum(density_out * jnp.asarray(magnetic_potential))
        ) / cells
    deband = -element * float(jnp.sum(density_out * v)) / cells + float(paw_deband)

    terms = {
        "one-electron": float(band_energy) + deband,
        "hartree": float(potential_out.ehart) / cells,
        "xc": float(potential_out.etxc) / cells,
        "ewald": float(ewald),
    }
    if onecentre:
        terms["one_center_paw"] = float(onecentre)
    if dispersion:
        terms["dispersion"] = float(dispersion)
    if entropy:
        terms["smearing"] = float(entropy)
    terms["_field_energy"] = field_energy
    terms["_deband"] = deband
    return terms


def total_of(terms: dict) -> float:
    """The sum of the terms that are in the total, which is not all of them."""
    return float(sum(v for k, v in terms.items() if not k.startswith("_")))
