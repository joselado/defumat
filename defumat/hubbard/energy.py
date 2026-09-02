"""The Hubbard energy, and the potential as its derivative.

The simplified rotationally-invariant functional of Dudarev *et al.*,
Phys. Rev. B **57**, 1505 (1998), extended with the ``J0`` and ``beta`` terms of
Himmetoglu *et al.*, Phys. Rev. B **84**, 115108 (2011) -- QE's
``lda_plus_u_kind = 0``, ``v_hubbard`` in ``PW/src/v_of_rho.f90``:

    E_U = sum_{I,s} [ (alpha_I + Ueff_I/2) Tr n^{Is}
                      - (Ueff_I/2) Tr (n^{Is} n^{Is}) ]
        + sum_{I,s} [ sgn(s) beta_I Tr n^{Is}
                      + (J0_I/2) Tr (n^{Is} n^{I,-s}) ]

with ``Ueff = U - J0``. **Only the energy is written down**; the potential
``v^{Is}_{m1 m2} = dE_U / dn^{Is}_{m2 m1}`` is ``jax.grad`` of it, which is this
project's rule (`PLAN.md` D1) and the same arrangement
:mod:`defumat.scf.fields` uses for the magnetic field. QE's hand-derived
``v_hubbard`` is transcribed below as :func:`qe_hubbard_potential` and is a
*test*, not a second implementation.

**The full (Liechtenstein) functional is here too** -- QE's
``lda_plus_u_kind = 1``, ``v_hubbard_full`` -- selected by the presence of a
``J`` in the ``HUBBARD`` card, exactly as ``read_cards.f90:3240`` selects it:

    E_U = 1/2 sum_{I,s} sum_{m1..m4} [ (vee(m1,m2,m3,m4) - vee(m1,m2,m4,m3))
                                        n^{Is}_{m1 m3} n^{Is}_{m2 m4}
                                     + vee(m1,m2,m3,m4)
                                        n^{Is}_{m1 m3} n^{I,-s}_{m2 m4} ]
        - sum_I 1/2 [ U_I N_I (N_I - 1) - J_I N_I (N_I/2 - 1)
                      - J_I M_I^2 / 2 ]

with ``vee`` from :mod:`defumat.hubbard.interaction`, ``N`` the shell's total
occupation and ``M`` its moment. The second line is the fully-localised-limit
double counting, and it is the *only* double counting QE offers.

**Around-mean-field is the second double counting and it is not a second
functional** -- Elk's ``dftu = 2``, ``vmatmtdu.f90``, which ``pw.x`` does not
have at all (verified by grep over ``PW/src``, ``Modules`` and ``upflib``). FLL
assumes the occupations are integers, which is right for a localised magnetic
insulator and wrong for a metal, where the correction it applies to a uniformly
filled shell should be nothing and is not. AMF subtracts the shell's *mean*
occupation instead and then applies the **same** ``vee`` contraction with no
double-counting term at all:

    n^s -> n^s - (Tr n^s / (2l+1)) I,     E = E_u[n - nbar],   E_dc = 0

so a full, an empty and a uniformly fractional shell all give exactly zero. The
shift is written down and the potential is still ``jax.grad``, which
differentiates through ``nbar(n)`` where Elk's hand-derived expression does not.
**The two still agree**, and the reason is worth recording rather than
measuring alone: the extra term is ``-delta_{ab} Tr(V)/(2l+1)``, and ``Tr(V)``
vanishes identically because ``sum_a vee(a,c,a,d)`` and ``sum_a vee(a,c,d,a)``
are rotationally invariant rank-two tensors and therefore proportional to
``delta_{cd}``, which they contract with a **traceless** shifted matrix.
:func:`qe_hubbard_full_potential` is the cross-check either way.

**The same "differentiate the per-channel energy" rule carries across, and the
double-counting term is where it bites.** For ``nspin = 1`` the reported energy
is ``2 E_u - E_dc`` and the potential is the derivative of ``E_u - E_dc/2``, so
what is differentiated is again "the energy of one channel" and the reported
value is again twice it. Halving ``E_dc`` inside the differentiated function is
what makes those two statements the same one; differentiating ``2 E_u - E_dc``
gives a double-counting potential twice too large, which is a converged run with
the wrong total.

**The factor of two for an unpolarized run is not in the potential.** QE halves
``ns`` in ``new_ns`` when ``nspin = 1`` -- so ``ns`` always means the occupation
*of one spin channel* -- and then doubles ``eth`` and the ``deband`` correction
at the end, because there are two identical channels. It does **not** double
``v_hub``, which acts on one channel at a time. So the function differentiated
here is the per-channel sum, and the doubling is applied to the reported energy
afterwards (:func:`hubbard_energy`). Differentiating the doubled energy instead
gives a potential twice too large, an SCF that converges, and a total energy
wrong in the second decimal.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "coefficients_from_setup",
    "hubbard_energy",
    "hubbard_potential",
    "elk_amf_potential",
    "ns_ddot",
    "qe_hubbard_full_potential",
    "qe_hubbard_potential",
]


def coefficients_from_setup(setup) -> dict:
    """The per-slot coefficients of the four terms, already branched.

    QE's ``v_hubbard`` guards each pair of terms with a test on the *input*
    parameters -- the ``U``/``alpha`` pair runs only when one of them is
    nonzero, the ``J0``/``beta`` pair likewise -- and ``Ueff = U - J0`` only
    when ``J0`` is nonzero. Those are host-side branches on static numbers, so
    they are resolved here, once, into arrays; nothing downstream branches.

    A plain ``dict`` of arrays, which is a pytree, so it crosses ``jit`` and
    ``grad`` boundaries as the constant it is.
    """
    u = setup.parameter("u")
    if getattr(setup, "kind", 0) == 1:
        return {
            "kind": 1,
            "double_counting": getattr(setup, "double_counting", "fll"),
            "vee": jnp.asarray(_vee_table(setup)),
            "ldims": jnp.asarray(np.asarray(setup.ldims, dtype=float)),
            "u": jnp.asarray(u),
            "j": jnp.asarray(setup.parameter("j")),
            # ``ns_ddot`` takes the bare ``U`` for every kind but 2
            # (``scf_mod.f90:962``), so the mixing metric is unchanged.
            "u_metric": jnp.asarray(0.5 * u),
        }
    j0 = setup.parameter("j0")
    alpha = setup.parameter("alpha")
    beta = setup.parameter("beta")

    effective = np.where(j0 != 0.0, u - j0, u)
    first = (u != 0.0) | (alpha != 0.0)
    second = (j0 != 0.0) | (beta != 0.0)
    return {
        "kind": 0,
        "linear": jnp.asarray(np.where(first, alpha + 0.5 * effective, 0.0)),
        "quadratic": jnp.asarray(np.where(first, 0.5 * effective, 0.0)),
        "beta": jnp.asarray(np.where(second, beta, 0.0)),
        "exchange": jnp.asarray(np.where(second, 0.5 * j0, 0.0)),
        # ``ns_ddot`` uses the bare U, not ``Ueff``.
        "u_metric": jnp.asarray(0.5 * u),
    }


def _vee_table(setup) -> np.ndarray:
    """``(nslot, ldmx, ldmx, ldmx, ldmx)``: each slot's Coulomb matrix, padded.

    Built once per calculation on the host. The padding is zero, so a slot whose
    manifold is shorter than ``ldmx`` contributes nothing through its padded
    rows -- the same arrangement ``ns`` itself uses.
    """
    from defumat.hubbard.interaction import coulomb_matrix

    ldmx = setup.ldmx
    table = np.zeros((setup.nslot, ldmx, ldmx, ldmx, ldmx))
    cache: dict[int, np.ndarray] = {}
    for slot, kind in enumerate(setup.types):
        item = setup.species[kind]
        if kind not in cache:
            cache[kind] = coulomb_matrix(item.l, item.slater)
        block = cache[kind]
        width = block.shape[0]
        table[slot, :width, :width, :width, :width] = block
    return table


def _full_energy(ns: jnp.ndarray, coefficients) -> jnp.ndarray:
    """``eth_u - eth_dc`` of ``v_hubbard_full``, **per channel**.

    ``eth_dc`` is halved for ``nspin = 1`` so that the whole expression is the
    one whose derivative is ``v_hub`` and whose double is ``eth``. See the
    module docstring: that halving is the one thing about the full functional
    that is not a transcription of the Fortran's own arithmetic.
    """
    nspin = ns.shape[0]
    vee = coefficients["vee"]

    if coefficients["double_counting"] == "amf":
        # ``vmatmtdu.f90``'s ``dftu = 2``: subtract the shell's mean occupation
        # from each channel and run the same contraction with no double
        # counting. Elk writes the shift as ``n0 +- mg0``, which for a collinear
        # occupation matrix is ``Tr n^s / (2l+1)`` in either spin regime -- its
        # ``nspinor`` and its halving of ``dm`` cancel between the two branches.
        # Padded rows have ``ldim`` from the setup rather than ``ldmx``, so a
        # short manifold is not averaged over its own padding.
        mean = jnp.einsum("snmm->sn", ns) / coefficients["ldims"]
        identity = jnp.eye(ns.shape[-1], dtype=ns.dtype)
        shifted = ns - mean[:, :, None, None] * identity * _slot_mask(coefficients)
        return _full_interaction(shifted, vee)

    trace = jnp.einsum("snmm->sn", ns)  # (nspin, nslot)
    total = jnp.sum(trace, axis=0)
    if nspin == 1:
        total = 2.0 * total
        moment = jnp.zeros_like(total)
    else:
        moment = trace[0] - trace[1]

    double_counting = 0.5 * (
        coefficients["u"] * total * (total - 1.0)
        - coefficients["j"] * total * (0.5 * total - 1.0)
        - 0.5 * coefficients["j"] * moment**2
    )
    if nspin == 1:
        double_counting = 0.5 * double_counting

    return _full_interaction(ns, vee) - jnp.sum(double_counting)


def _slot_mask(coefficients) -> jnp.ndarray:
    """``(nslot, ldmx, ldmx)``: ones on the real diagonal of each block.

    A manifold shorter than ``ldmx`` is padded with zeros, and the mean-field
    shift must not be written into that padding -- a padded row is not an
    orbital and a shifted one would contribute to every trace below.
    """
    ldmx = coefficients["vee"].shape[-1]
    index = jnp.arange(ldmx)
    return (index[None, :] < coefficients["ldims"][:, None])[:, :, None] * jnp.eye(ldmx)


def _full_interaction(ns: jnp.ndarray, vee: jnp.ndarray) -> jnp.ndarray:
    """``E_u``: the four-index contraction, shared by both double countings.

    ``0.5 sum_{s,s'} [ vee_abcd n^s_ac n^s'_bd - delta_ss' vee_abdc n^s_ac n^s'_bd ]``
    -- ``v_hubbard_full``'s ``eth_u`` with the two spin sums written out. The
    exchange term is diagonal in spin because a collinear occupation matrix has
    no coherence between the channels; the noncollinear form of the same
    expression carries it (``v_hubbard_full_nc``).
    """
    nspin = ns.shape[0]
    opposite = ns[::-1] if nspin == 2 else ns
    direct = jnp.einsum("nabcd,snac,snbd->", vee, ns, ns)
    exchange = jnp.einsum("nabdc,snac,snbd->", vee, ns, ns)
    cross = jnp.einsum("nabcd,snac,snbd->", vee, ns, opposite)
    return 0.5 * (direct - exchange + cross)


def _spin_blocks(ns: jnp.ndarray) -> jnp.ndarray:
    """``(2, 2, nslot, ldmx, ldmx)`` from the packed ``(4, ...)`` spinor ``ns``."""
    return ns.reshape((2, 2) + ns.shape[1:])


def _noncollinear_energy(ns: jnp.ndarray, coefficients) -> jnp.ndarray:
    """``v_hubbard_nc`` and ``v_hubbard_full_nc``, written once.

    The simplified functional is the collinear one on the **combined**
    ``(m, spin)`` index -- ``(alpha + U/2) Tr N - (U/2) Tr(N N)`` with ``N`` the
    whole ``2(2l+1)`` Hermitian matrix -- which is exactly what
    ``v_hubbard_nc``'s spin-flip and non-spin-flip branches add up to. The full
    one is the same four-index contraction with both spin sums written out:

        E_u = 1/2 sum_{s,s'} [ vee_abcd N^{ss}_ac N^{s's'}_bd
                             - vee_abdc N^{ss'}_ac N^{s's}_bd ]

    where the exchange term carries the **off-diagonal** spin blocks a collinear
    occupation matrix does not have. ``J0`` and ``beta`` are absent because
    ``card_hubbard`` refuses them with ``noncolin``, and ``v_hubbard_nc`` has no
    branch for them either.
    """
    blocks = _spin_blocks(ns)
    diagonal = jnp.einsum("ssnab->nab", blocks)          # N^uu + N^dd
    trace = jnp.einsum("stnmm->stn", blocks)             # (2, 2, nslot)

    if coefficients["kind"] == 0:
        linear = jnp.sum(coefficients["linear"] * jnp.einsum("nmm->n", diagonal))
        square = jnp.einsum("stnab,tsnba->n", blocks, blocks)
        return jnp.real(linear - jnp.sum(coefficients["quadratic"] * square))

    vee = coefficients["vee"]
    if coefficients["double_counting"] == "amf":
        mean = trace / coefficients["ldims"]
        identity = jnp.eye(ns.shape[-1], dtype=ns.dtype)
        shift = mean[:, :, :, None, None] * identity * _slot_mask(coefficients)
        blocks = blocks - shift
        diagonal = jnp.einsum("ssnab->nab", blocks)
        return jnp.real(_noncollinear_interaction(blocks, diagonal, vee))

    # ``v_hubbard_full_nc``'s double counting: the shell's charge and the
    # magnitude of its moment, the second of which is what the collinear
    # expression writes as ``mag^2`` with one component.
    total = jnp.real(trace[0, 0] + trace[1, 1])
    moment = jnp.stack([
        jnp.real(trace[0, 1] + trace[1, 0]),
        jnp.imag(trace[0, 1] - trace[1, 0]),
        jnp.real(trace[0, 0] - trace[1, 1]),
    ])
    double_counting = 0.5 * (
        coefficients["u"] * total * (total - 1.0)
        - coefficients["j"] * total * (0.5 * total - 1.0)
        - 0.5 * coefficients["j"] * jnp.sum(moment**2, axis=0)
    )
    return jnp.real(
        _noncollinear_interaction(blocks, diagonal, vee)
    ) - jnp.sum(double_counting)


def _noncollinear_interaction(blocks, diagonal, vee) -> jnp.ndarray:
    """``E_u`` with both spin sums written out; see :func:`_noncollinear_energy`."""
    direct = jnp.einsum("nabcd,nac,nbd->", vee, diagonal, diagonal)
    exchange = jnp.einsum("nabdc,stnac,tsnbd->", vee, blocks, blocks)
    return 0.5 * (direct - exchange)


def _per_channel_energy(ns: jnp.ndarray, coefficients) -> jnp.ndarray:
    """The bracketed sum above, **without** the ``nspin = 1`` doubling.

    ``ns`` is ``(nspin, nslot, ldmx, ldmx)``; padded rows of a manifold shorter
    than ``ldmx`` are zero and contribute nothing to either trace.

    ``coefficients["kind"]`` is a plain Python integer -- the coefficients are
    built once on the host and closed over, never passed as a traced argument,
    so the branch resolves at trace time and neither functional is compiled into
    a run that does not use it.
    """
    if ns.shape[0] == 4:
        return _noncollinear_energy(ns, coefficients)
    if coefficients["kind"] == 1:
        return _full_energy(ns, coefficients)
    nspin = ns.shape[0]
    trace = jnp.einsum("snmm->sn", ns)
    square = jnp.einsum("snab,snba->sn", ns, ns)

    energy = jnp.sum(coefficients["linear"] * trace)
    energy = energy - jnp.sum(coefficients["quadratic"] * square)

    sign = jnp.asarray([1.0, -1.0])[:nspin] if nspin == 2 else jnp.asarray([1.0])
    energy = energy + jnp.sum(sign[:, None] * coefficients["beta"] * trace)
    # ``isop``: the opposite channel when there are two, the same one when there
    # is one. QE writes it as an index; reversing the spin axis is the same map.
    opposite = ns[::-1] if nspin == 2 else ns
    energy = energy + jnp.sum(
        coefficients["exchange"] * jnp.einsum("snab,snba->sn", ns, opposite)
    )
    return energy


def hubbard_energy(ns: jnp.ndarray, coefficients) -> jnp.ndarray:
    """``eth``: the Hubbard energy in Ry, as ``electrons.f90`` adds it to ``etot``.

    Doubled for ``nspin = 1``, because ``ns`` is then one channel of two.
    """
    energy = _per_channel_energy(ns, coefficients)
    return 2.0 * energy if ns.shape[0] == 1 else energy


def hubbard_potential(ns: jnp.ndarray, coefficients) -> jnp.ndarray:
    """``v_hub``: ``(nspin, nslot, ldmx, ldmx)``, the derivative of the energy.

    Not of :func:`hubbard_energy` -- of the per-channel sum. See the module
    docstring.

    For a spinor ``ns`` is complex and the energy is real, so the derivative is
    taken with ``holomorphic = False`` and **conjugated**: ``jax.grad`` of a
    real function of a complex argument returns the conjugate of the Wirtinger
    derivative (it is built for gradient descent), and the potential is the
    derivative itself -- ``v_hub`` is Hermitian, and the unconjugated array is
    its transpose, which is the one bug this branch invites and the one a
    symmetric test matrix cannot see.
    """
    if ns.shape[0] == 4:
        return jnp.conj(jax.grad(_per_channel_energy, holomorphic=False)(ns, coefficients))
    return jax.grad(_per_channel_energy)(ns, coefficients)


def qe_hubbard_potential(ns: np.ndarray, setup) -> np.ndarray:
    """``v_hubbard`` transcribed literally from ``PW/src/v_of_rho.f90``.

    The cross-check for :func:`hubbard_potential`, written in the Fortran's own
    loop structure so that the two implementations share nothing but the input.
    """
    ns = np.asarray(ns)
    nspin, nslot, ldmx, _ = ns.shape
    v_hub = np.zeros_like(ns)
    sgn = (1.0, -1.0)

    for slot in range(nslot):
        item = setup.species[setup.types[slot]]
        ldim = setup.ldims[slot]
        if item.u != 0.0 or item.alpha != 0.0:
            eff = item.u - item.j0 if item.j0 != 0.0 else item.u
            for spin in range(nspin):
                for m1 in range(ldim):
                    v_hub[spin, slot, m1, m1] += item.alpha + 0.5 * eff
                    for m2 in range(ldim):
                        v_hub[spin, slot, m1, m2] -= eff * ns[spin, slot, m2, m1]
        if item.j0 != 0.0 or item.beta != 0.0:
            for spin in range(nspin):
                other = 1 - spin if nspin == 2 else 0
                for m1 in range(ldim):
                    v_hub[spin, slot, m1, m1] += sgn[spin] * item.beta
                    for m2 in range(ldim):
                        v_hub[spin, slot, m1, m2] += (
                            item.j0 * ns[other, slot, m2, m1]
                        )
    return v_hub


def qe_hubbard_full_potential(ns: np.ndarray, setup) -> tuple[np.ndarray, float]:
    """``v_hubbard_full`` transcribed literally from ``PW/src/v_of_rho.f90``.

    The cross-check for :func:`hubbard_potential` under ``kind = 1``, written in
    the Fortran's own loop structure -- the quadruple ``m`` loop, the
    ``MOD(nspin,2)+1`` factor, the ``nspin+1-is`` opposite channel -- so that
    the two implementations share nothing but ``vee`` and the input.

    Returns ``(v_hub, eth)`` with ``eth`` the *reported* energy, already doubled
    for ``nspin = 1``, which is what ``electrons.f90`` adds to ``etot``.
    """
    from defumat.hubbard.interaction import coulomb_matrix

    ns = np.asarray(ns)
    nspin, nslot, ldmx, _ = ns.shape
    v_hub = np.zeros_like(ns)
    energy_u = 0.0
    energy_dc = 0.0
    factor = (nspin % 2) + 1

    for slot in range(nslot):
        item = setup.species[setup.types[slot]]
        ldim = setup.ldims[slot]
        u_matrix = coulomb_matrix(item.l, item.slater)

        total = 0.0
        for spin in range(nspin):
            for m1 in range(ldim):
                total += ns[spin, slot, m1, m1]
        if nspin == 1:
            total = 2.0 * total
        moment = 0.0
        if nspin == 2:
            for m1 in range(ldim):
                moment += ns[0, slot, m1, m1] - ns[1, slot, m1, m1]
        moment = moment**2

        energy_dc += 0.5 * (
            item.u * total * (total - 1.0)
            - item.j * total * (0.5 * total - 1.0)
            - 0.5 * item.j * moment
        )

        for spin in range(nspin):
            channel = 0.0
            for m1 in range(ldim):
                channel += ns[spin, slot, m1, m1]
            other = nspin - 1 - spin
            for m1 in range(ldim):
                v_hub[spin, slot, m1, m1] += (
                    item.j * channel + 0.5 * (item.u - item.j) - item.u * total
                )
                for m2 in range(ldim):
                    for m3 in range(ldim):
                        for m4 in range(ldim):
                            for is1 in range(nspin):
                                v_hub[spin, slot, m1, m2] += (
                                    factor
                                    * u_matrix[m1, m3, m2, m4]
                                    * ns[is1, slot, m3, m4]
                                )
                            v_hub[spin, slot, m1, m2] -= (
                                u_matrix[m1, m3, m4, m2] * ns[spin, slot, m3, m4]
                            )
                            energy_u += 0.5 * (
                                (u_matrix[m1, m2, m3, m4] - u_matrix[m1, m2, m4, m3])
                                * ns[spin, slot, m1, m3]
                                * ns[spin, slot, m2, m4]
                                + u_matrix[m1, m2, m3, m4]
                                * ns[spin, slot, m1, m3]
                                * ns[other, slot, m2, m4]
                            )
    if nspin == 1:
        energy_u = 2.0 * energy_u
    return v_hub, energy_u - energy_dc


def elk_amf_potential(ns: np.ndarray, setup) -> tuple[np.ndarray, float]:
    """``vmatmtdu.f90``'s ``dftu = 2`` branch, transcribed as the AMF check.

    Elk's loop structure, its shifted density matrix and its hand-derived
    potential -- which does **not** differentiate through the shift, where
    :func:`hubbard_potential` does. The two agree anyway, for the reason in the
    module docstring, and this is where that is measured rather than argued.

    Returns ``(v_hub, eth)``, ``eth`` already doubled for ``nspin = 1``.
    """
    from defumat.hubbard.interaction import coulomb_matrix

    ns = np.asarray(ns)
    nspin, nslot, ldmx, _ = ns.shape
    v_hub = np.zeros_like(ns)
    energy = 0.0

    for slot in range(nslot):
        item = setup.species[setup.types[slot]]
        ldim = setup.ldims[slot]
        vee = coulomb_matrix(item.l, item.slater)

        # ``dm(lm1,ispn,lm1,ispn) -= n0 +- mg0``: for a collinear occupation
        # matrix that is the channel's own mean, in either spin regime.
        dm = np.array(ns[:, slot, :ldim, :ldim])
        for spin in range(nspin):
            dm[spin] -= np.eye(ldim) * (np.trace(dm[spin]) / ldim)

        for spin in range(nspin):
            other_channels = range(nspin)
            for m1 in range(ldim):
                for m2 in range(ldim):
                    total = 0.0
                    for m3 in range(ldim):
                        for m4 in range(ldim):
                            for other in other_channels:
                                total += vee[m1, m3, m2, m4] * dm[other, m3, m4]
                            total -= vee[m1, m3, m4, m2] * dm[spin, m3, m4]
                    v_hub[spin, slot, m1, m2] = total
        if nspin == 1:
            # One channel stands for two: the direct term counts both, exactly
            # as ``MOD(nspin,2)+1`` does in ``v_hubbard_full``.
            for m1 in range(ldim):
                for m2 in range(ldim):
                    extra = 0.0
                    for m3 in range(ldim):
                        for m4 in range(ldim):
                            extra += vee[m1, m3, m2, m4] * dm[0, m3, m4]
                    v_hub[0, slot, m1, m2] += extra

        opposite = dm[::-1] if nspin == 2 else dm
        direct = np.einsum("abcd,sac,sbd->", vee, dm, dm)
        exchange = np.einsum("abdc,sac,sbd->", vee, dm, dm)
        cross = np.einsum("abcd,sac,sbd->", vee, dm, opposite)
        energy += 0.5 * (direct - exchange + cross)

    if nspin == 1:
        energy = 2.0 * energy
    return v_hub, energy


def ns_ddot(residual: jnp.ndarray, coefficients) -> jnp.ndarray:
    """``ns_ddot`` (``PW/src/scf_mod.f90``): the ``ns`` part of the mixing metric.

    ``U/2 sum |dns|^2``, doubled for ``nspin = 1``, added to ``rho_ddot`` to
    make ``dr2`` an estimate of the self-consistency error in the *whole*
    functional rather than in its plane-wave part alone. Leaving it out does not
    change the converged answer; it changes ``dr2``, and with it the ``ethr``
    schedule and the iteration count, so QE's numbers would no longer be
    reproducible step for step.
    """
    # ``CONJG`` for the spinor branch, where the residual is complex and
    # ``ns_ddot`` is still a norm (``scf_mod.f90``'s ``nspin == 4`` case).
    square = jnp.real(jnp.conj(residual) * residual)
    value = jnp.sum(coefficients["u_metric"] * jnp.sum(square, axis=(2, 3)))
    return 2.0 * value if residual.shape[0] == 1 else value
