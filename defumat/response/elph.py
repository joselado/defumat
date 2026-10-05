"""Electron-phonon coupling at one wavevector: ``g``, the linewidths and ``lambda``.

``ph.x``'s ``electron_phonon = 'simple'`` (``PHonon/PH/elphon.f90``, ``elphel``
and ``elphsum_simple``), on top of the phonon at ``q`` of
:mod:`defumat.response.phononq`.

**The matrix element** is the first-order change of the self-consistent
potential under a displacement, between a state at ``k`` and one at ``k + q``:

    g^(a i)_(mn)(k, q) = <psi_(m, k+q)| dV_bare^(a i) + dV_scf^(a i) |psi_(n k)>

in Ry/bohr, one per atom ``a`` and cartesian direction ``i``. Nothing in it is
new: ``dV_bare |psi>`` is the vector the Sternheimer right-hand side was built
from (P71's one ``jvp`` through the moved atoms, ``bare_displacements_at_q``),
and ``dV_scf`` is the converged input of the screening loop, applied by the
same :func:`~defumat.response.phononq.induced_perturbation_at_q` the solver
used. That is ``elphel``'s ``dvpsi`` read back from ``iubar`` plus ``dvscfins``
applied to ``evc`` (``elphon.f90:492``, ``:523-525``), and the contraction is
its one ``zgemm`` over every band at both spheres (``:571-574``).

**The sum** is ``elphsum_simple`` (``elphon.f90:1191-1392``), transcribed. For
each broadening ``sigma`` the Fermi level and the density of states are
recomputed from the ``k`` eigenvalues at that width with the smearing
``el_ph_ngauss`` (Methfessel-Paxton by default), while the two deltas are plain
Gaussians:

    S^(mu nu) = sum_k w_k sum_(mn) delta(e_nk - E_F) delta(e_(m,k+q) - E_F)
                conj(g^mu_mn) g^nu_mn
    gamma_nu  = (pi / 2) z_nu^dagger S z_nu
    lambda_nu = gamma_nu / (pi N(E_F) omega_nu^2)

with ``z_nu = e_nu / sqrt(M)`` the displacement of mode ``nu`` (``dyndia``'s
``dyn``), ``w_k`` summing to 2 for one spin channel and ``N(E_F)`` per spin.
``gamma_nu`` is the phonon linewidth, half-width in energy, which is what
inelastic neutron or x-ray scattering resolves; ``lambda_nu`` the mode's
dimensionless coupling, whose Brillouin-zone average is the ``lambda`` of
McMillan's ``T_c``. The ``1/2`` is the ``sqrt(hbar / 2 M omega)`` of the
quantised displacement, and ``lambda`` is zero for a mode below 20 cm^-1, where
``ph.x`` sets it so.

**Two things that are this code's and not** ``ph.x``'s, both about what is
invariant:

* ``S`` is a trace over band pairs with weights that depend only on
  eigenvalues, so it is unchanged by the rotation a degenerate eigensolver is
  free in at ``k`` or at ``k + q`` (rule D4). A per-band ``|g_mn|^2`` is not,
  and is not exposed.
* inside a degenerate set of *modes* ``gamma_nu`` depends on which basis of the
  set the diagonalisation returned, unless ``S`` restricted to it is a multiple
  of the identity -- which symmetry guarantees and a full-grid sum delivers
  only to the solves' residue. ``ph.x`` symmetrises ``S`` over the small group
  of ``q`` (``symdyn_munu_new``); here the mode set's average is reported, which
  is basis-free and is the number ``ph.x`` prints for each member.

**What is refused.** Everything :func:`~defumat.response.phononq.
dynamical_matrix_at_q` refuses, so ultrasoft and PAW (whose ``dvpsi`` carries
``adddvscf``'s ``int3``, P97's second term), spin, spinors, a reduced k-set and
a metal at ``q = 0`` (``ef_shift``, and ``def`` subtracted from ``dvscf``);
and an insulator, for which the double delta at ``E_F`` is empty and the
coupling to the Fermi surface is not defined. The ``'interpolated'`` route
(dense-grid eigenvalues, ``alpha^2 F``, ``T_c``) is not here.

References: Allen, *Phys. Rev. B* **6**, 2577 (1972), for ``gamma`` and
``lambda``; Baroni, de Gironcoli, Dal Corso and Giannozzi, *Rev. Mod. Phys.*
**73**, 515 (2001), §VI.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from defumat.eager import compiled
from defumat.units import AMU_TO_RY, RY_TO_CMM1, RY_TO_THZ

__all__ = ["ElectronPhonon", "electron_phonon_at_q", "elphsum_simple",
           "RY_TO_GHZ"]

#: ``constants.f90``'s ``RY_TO_GHZ``: ``RY_TO_THZ * 1000``.
RY_TO_GHZ = RY_TO_THZ * 1000.0

#: ``elphsum_simple``'s ``eps``: below 20 cm^-1 ``lambda`` is set to zero.
SOFT_MODE_CMM1 = 20.0


@dataclass
class ElectronPhonon:
    """The coupling of the phonons at one ``q`` to the Fermi surface."""

    #: ``(3,)`` in cartesian units of ``2 pi / alat``, as ``ph.x`` prints ``xq``.
    q: np.ndarray
    #: The phonon at ``q`` the coupling was computed with.
    phonons: object
    #: ``(nsigma,)`` Gaussian broadenings of the double delta, in Ry.
    sigmas: np.ndarray
    #: ``(nsigma,)`` the Fermi level recomputed at each broadening, in Ry.
    fermi_energies: np.ndarray
    #: ``(nsigma,)`` ``N(E_F)`` in states per spin per Ry per cell.
    dos: np.ndarray
    #: ``(nsigma,)`` ``sum_k w_k sum_mn delta delta``, ``ph.x``'s "double delta at Ef".
    phase_space: np.ndarray
    #: ``(nsigma, 3 nat, 3 nat)`` ``S^(mu nu)`` in cartesian components, Ry^2/bohr^2.
    el_ph_sum: np.ndarray
    #: ``(nsigma, 3 nat)`` linewidths in Ry, averaged over degenerate modes.
    gamma: np.ndarray
    #: ``(nsigma, 3 nat)`` dimensionless ``lambda_(q nu)``, averaged likewise.
    lambdas: np.ndarray

    @property
    def frequencies(self) -> np.ndarray:
        """``(3 nat,)`` in cm^-1, those of :attr:`phonons`."""
        return np.asarray(self.phonons.frequencies)

    @property
    def gamma_ghz(self) -> np.ndarray:
        """:attr:`gamma` in GHz, the unit ``ph.x`` prints it in."""
        return self.gamma * RY_TO_GHZ


def matrix_elements(internals, nat: int):
    """``g^(a i)_(mn)(k, q)``: ``(nat, 3, nspin, nk, nbnd_kq, nbnd_k)`` complex, Ry/bohr.

    ``m`` runs over the states at ``k + q`` and ``n`` over those at ``k``.
    Built per mode as one compiled walk over k (:func:`~defumat.batching.map_k`):
    the bare vector plus the converged ``dV_scf`` applied to ``psi_k``, then
    contracted with ``psi_(k+q)`` on the second sphere. Allocates the result
    only; the operands are the phonon's.
    """
    from defumat.batching import map_k
    from defumat.response.phononq import induced_perturbation_at_q

    solver = internals["solver"]
    calculation, kq = solver.calculation, internals["calculation_kq"]
    dvscf = internals["dvscf"]
    nspin, nk = solver.psi.shape[:2]
    out = []
    for atom in range(nat):
        for cart in range(3):
            induced = induced_perturbation_at_q(calculation, kq, dvscf[atom, cart])
            bare = internals["bare"][atom, cart]

            def walk(psi, psi_kq, bare, induced=induced):
                channels = []
                for spin in range(nspin):
                    def one_k(item, spin=spin):
                        ik, states, states_kq, applied = item
                        dvpsi = applied + induced(states, ik, spin)
                        return jnp.einsum("mg,ng->mn", jnp.conj(states_kq), dvpsi)

                    channels.append(map_k(
                        one_k,
                        (jnp.arange(nk), psi[spin], psi_kq[spin], bare[spin]),
                        batch=calculation.k_batch,
                    ))
                return jnp.stack(channels)

            out.append(compiled(walk, internals["psi"], solver.psi_kq, bare))
    return np.asarray(jnp.stack(out)).reshape((nat, 3) + out[0].shape)


def elphsum_simple(g, eigenvalues, eigenvalues_kq, kpoint_weights, nelec,
                   omega2, displacements, sigmas, ngauss: int = 1,
                   degeneracy_cmm1: float = 0.05):
    """``elphsum_simple``: the double-delta sum over the k-grid, per broadening.

    ``g`` is :func:`matrix_elements`' array, ``eigenvalues`` and
    ``eigenvalues_kq`` ``(nspin, nk, nbnd)`` in Ry, ``kpoint_weights`` the run's
    own (``(nk,)``, summing to 2 for one channel), ``omega2`` the ``(3 nat,)``
    squared frequencies in Ry^2 and ``displacements`` the ``(3 nat, 3 nat)``
    mode displacements ``e_nu / sqrt(M)`` as columns, mass in Rydberg units.

    Returns ``(fermi_energies, dos, phase_space, el_ph_sum, gamma, lambdas)``.
    """
    from defumat.scf.occupations import bisect_fermi, w0gauss

    nat = g.shape[0]
    nmodes = 3 * nat
    flat = g.reshape((nmodes,) + g.shape[2:])          # (mu, s, k, m, n)
    eps = np.asarray(eigenvalues)
    eps_kq = np.asarray(eigenvalues_kq)
    wk = np.asarray(kpoint_weights, dtype=float)
    weights = np.broadcast_to(wk[None, :, None], eps.shape)
    frequencies = np.sign(omega2) * np.sqrt(np.abs(omega2)) * RY_TO_CMM1
    groups = _degenerate_sets(frequencies, degeneracy_cmm1)

    out = {name: [] for name in ("ef", "dos", "phase", "sum", "gamma", "lambda")}
    for sigma in np.asarray(sigmas, dtype=float):
        # ``efermig`` with the k+q weights zero (``elphon.f90:1279-1280``), at
        # this width and ``el_ph_ngauss``; ``dos_ef`` halved to per spin.
        # ``bisect_fermi`` takes one weight per row and broadcasts it over the
        # bands; the spin channels are rows of their own.
        ef = float(bisect_fermi(jnp.asarray(eps.reshape(-1, eps.shape[-1])),
                                jnp.asarray(np.tile(wk, eps.shape[0])),
                                float(nelec), sigma, ngauss))
        dos = float(np.sum(weights * np.asarray(
            w0gauss(jnp.asarray((ef - eps) / sigma), ngauss)) / sigma)) / 2.0
        # The two deltas are Gaussians whatever ``el_ph_ngauss`` is (``ngauss1 = 0``).
        delta = np.asarray(w0gauss(jnp.asarray((ef - eps) / sigma), 0)) / sigma
        delta_kq = np.asarray(w0gauss(jnp.asarray((ef - eps_kq) / sigma), 0)) / sigma
        # weight_(s k m n) = w_k delta(e_nk) delta(e_(m k+q))
        weight = wk[None, :, None, None] * delta_kq[:, :, :, None] * delta[:, :, None, :]
        summed = np.einsum("skmn,askmn,bskmn->ab", weight, np.conj(flat), flat)
        phase = float(np.sum(weight))

        projected = np.real(np.einsum(
            "an,ab,bn->n", np.conj(displacements), summed, displacements))
        gamma = np.pi * projected / 2.0
        gamma = _average_over(groups, gamma)
        soft = np.abs(frequencies) <= SOFT_MODE_CMM1
        coupling = np.where(
            soft, 0.0, gamma / np.pi / np.where(soft, 1.0, omega2) / dos)

        for name, value in (("ef", ef), ("dos", dos), ("phase", phase),
                            ("sum", summed), ("gamma", gamma),
                            ("lambda", coupling)):
            out[name].append(value)
    return tuple(np.asarray(out[name]) for name in
                 ("ef", "dos", "phase", "sum", "gamma", "lambda"))


def _degenerate_sets(frequencies, tolerance):
    """Consecutive modes within ``tolerance`` cm^-1 of the previous one, as index lists."""
    groups, current = [], [0]
    for index in range(1, len(frequencies)):
        if abs(frequencies[index] - frequencies[index - 1]) <= tolerance:
            current.append(index)
        else:
            groups.append(current)
            current = [index]
    groups.append(current)
    return groups


def _average_over(groups, values):
    out = np.array(values, dtype=float)
    for group in groups:
        out[group] = np.mean(out[group])
    return out


def electron_phonon_at_q(
    calculation, wavefunctions, eigenvalues, density, becsum=(),
    q=(0.0, 0.0, 0.0), q_cartesian: bool = False,
    sigmas=None, el_ph_sigma: float = 0.02, el_ph_nsigma: int = 10,
    el_ph_ngauss: int = 1, degeneracy_cmm1: float = 0.05,
    nbnd: int | None = None, threshold: float | None = None,
    alpha_mix: float = 0.7, tr2: float = 1.0e-14, max_iterations: int = 100,
    verbose: bool = False,
) -> ElectronPhonon:
    """The phonon at ``q``, its coupling matrix elements, and ``elphsum_simple``.

    ``sigmas`` defaults to ``ph.x``'s ``el_ph_sigma * (1, ..., el_ph_nsigma)``,
    0.02 to 0.20 Ry; ``el_ph_ngauss`` is the smearing the Fermi level and
    ``N(E_F)`` are recomputed with at each of them (1, Methfessel-Paxton, as in
    ``ph.x``). ``degeneracy_cmm1`` is how close two frequencies must be to be
    averaged as one degenerate set. ``nbnd``, ``threshold``, ``alpha_mix``,
    ``tr2``, ``max_iterations`` and ``verbose`` are
    :func:`~defumat.response.phononq.dynamical_matrix_at_q`'s.
    """
    from defumat.response.phononq import dynamical_matrix_at_q

    if calculation.system.occupations == "fixed":
        raise NotImplementedError(
            "electron-phonon coupling to the Fermi surface needs a metal: with "
            "fixed occupations there are no states at E_F, the double delta is "
            "empty and gamma and lambda are not defined. Run the ground state "
            "with occupations = 'smearing'"
        )
    phonons = dynamical_matrix_at_q(
        calculation, wavefunctions, eigenvalues, density, becsum, q=q,
        q_cartesian=q_cartesian, nbnd=nbnd, threshold=threshold,
        alpha_mix=alpha_mix, tr2=tr2, max_iterations=max_iterations,
        verbose=verbose, keep_internals=True)
    internals = phonons.internals
    nat = calculation.system.structure.nat
    g = matrix_elements(internals, nat)

    solver = internals["solver"]
    masses = np.repeat(np.asarray(calculation.system.structure.masses), 3) * AMU_TO_RY
    displacements = np.asarray(phonons.eigenvectors) / np.sqrt(masses)[:, None]
    if sigmas is None:
        sigmas = el_ph_sigma * np.arange(1, el_ph_nsigma + 1)
    ef, dos, phase, summed, gamma, coupling = elphsum_simple(
        g, internals["eigenvalues"], solver.eigenvalues_kq,
        calculation.system.kpoints.weights, calculation.nelec,
        phonons.omega2, displacements, sigmas, ngauss=el_ph_ngauss,
        degeneracy_cmm1=degeneracy_cmm1,
    )
    # The internals hold the whole response; the result keeps the phonon alone.
    phonons.internals = None
    return ElectronPhonon(
        q=np.asarray(internals["q_cart"]) / calculation.system.cell.tpiba,
        phonons=phonons, sigmas=np.asarray(sigmas, dtype=float),
        fermi_energies=ef, dos=dos, phase_space=phase, el_ph_sum=summed,
        gamma=gamma, lambdas=coupling,
    )
