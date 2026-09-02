"""The independent-particle response ``chi_0(G, G', omega)`` at ``q -> 0``.

``PLAN.md`` P37. This is the one object in the TDDFT phase that costs anything,
and it is the one thing the rest of :mod:`defumat.tddft` cannot get from code
that already exists: everywhere else in ``defumat/response/`` the
susceptibility appears as an *operator*, ``drho = chi_0 dV`` from a Sternheimer
solve, and a Dyson equation needs it as a **matrix** over reciprocal lattice
vectors, at every frequency.

**Sum over states, and the tempting shortcut is a trap.** The kernel of P37 is
static, so ``chi_0`` is needed at ``omega = 0`` for it and at every ``omega``
only for the spectrum -- which suggests taking the kernel from the Sternheimer
stack, where ``omega = 0`` already lives, and the spectrum from a sum over
states. ``tddftlr.f90`` says why not: it hands ``genvfxc`` the ``omega = 0``
*slice of the same array* it then inverts at every frequency, so the kernel and
the spectrum carry one band truncation, one broadening and one response cutoff.
Two builders would put a band-complete, unbroadened kernel inside a truncated,
broadened spectrum, and the bootstrap's self-consistency would then couple them.
Nothing in the answer shows that. So there is one builder, it is Adler-Wiser
(``genvchi0.f90``), and the Sternheimer stack is the independent referee
instead (``tests/regression/test_tddft.py``).

**What is stored is the symmetrised ``v^1/2 chi_0 v^1/2``**, not ``chi_0``.
That is what keeps the optical limit finite: ``chi_0`` itself vanishes as
``q^2`` and ``v`` diverges as ``1/q^2``, and only the product has a limit. In
that limit the ``G = 0`` row and column become **direction-dependent**, so they
are a 3x3 block rather than one number and the matrix has

    nm = 3 + (ngrf - 1)

rows: three head directions, then every ``G != 0`` inside the response cutoff.
This is Elk's ``t3hw`` layout (``nm = ngrf + 2``) with its index 1 doing double
duty removed.

**The whole matrix is a sum of rank-one terms, which is what makes it cheap.**
For one pair of states,

    r_a    = -sqrt(8 pi) <u_i| dH/dk_a |u_j> / (eps_i - eps_j)      (head)
    r_G    =  sqrt(8 pi / |G|^2) <u_i| e^{-iG.r} |u_j>              (body)

and the pair contributes ``cw(omega) r r^dagger`` with the scalar

    cw(omega) = (wg_i - wg_j) / Omega / (eps_i - eps_j + omega + i eta).

So the pair vectors are built once, per k-point, and the frequency axis costs
one matrix product each -- ``(pairs, nm)`` scaled by a frequency-dependent
weight and contracted against itself.

**The head is the one line of Elk that must not be transcribed.**
``genvchi0.f90`` reads momentum matrix elements off a file, which is exactly
right in an all-electron code and silently wrong in a pseudopotential one:
``[H, r] != p`` when the pseudopotential is nonlocal, and the difference is not
small. What is wanted is ``dH/dk``, and that is
:class:`~defumat.response.velocity.VelocityOperator` -- one ``jvp`` of ``H(k)``
at a frozen sphere, rule D2. This module is otherwise a transcription; this is
the place where the autodiff the project is built on is load-bearing rather
than decorative.

**Both orderings of every pair are summed**, as ``genvchi0`` does -- the
resonant term and the antiresonant one. Keeping only ``v -> c`` and doubling
gets the real part roughly right and the imaginary part wrong. The reversed
pair needs no second transform, because

    <u_j| e^{-iG.r} |u_i> = conj(<u_i| e^{+iG.r} |u_j>),

so its vector is the first one reflected through the origin and conjugated. The
head reflects with a **sign**, since ``rho_0`` is linear in ``q`` and the
reflection is ``q -> -q``.

**Scissors is part of the method**, not a convenience: PRL 107, 186401's Eq. (3)
replaces ``chi_0`` by a gap-corrected model response, and every published
bootstrap spectrum is computed that way. The shift is applied to the empty
states' eigenvalues, and the velocity matrix elements are renormalised by
``e_ij / (e_ij -+ Delta)`` with it -- Elk's ``getpmat.f90``, which is the
Del Sole-Girlanda factor and is what keeps ``<u|r|u'>`` right when the energies
it was divided by have moved.

**What it costs, and where.** Two working sets, and they scale differently:

* the **stored matrix**, ``nw nm^2`` complex, which is what the caller keeps.
  With ``nm = 115`` (silicon at ``ecut_response = 8``) and 150 frequencies that
  is 25 MB, and it is quadratic in the response cutoff.
* the **assembly**, ``nw (2 npairs) nm`` complex in flight per k-chunk, which is
  the ``einsum`` below and is the larger of the two: 100 MB at the same sizes.
  It is a stated trade rather than an accident -- the alternative is one matrix
  product per frequency, which halves the flop rate on CPU -- and the frequency
  axis is the dial that would fix it if it ever mattered. ``PERFORMANCE.md``
  carries the measurement.

The transforms are ``nk npairs`` of them, one per occupied-empty pair, and they
are the *time* rather than the memory: the pair densities go through ``map_k``
so only one k-chunk's are ever alive.

Refused by name: finite ``q``, ultrasoft and PAW, metals, ``nspin != 1``,
noncollinear magnetism and spin-orbit coupling, and a **reduced k-set** -- see
:func:`require_a_sum_over_states_regime`.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from defumat.basis.fft import g_to_r, r_to_g
from defumat.batching import resolve_k_batch, sum_k
from defumat.response.velocity import VelocityOperator
from defumat.units import E2, FPI

__all__ = [
    "ResponseSphere",
    "ChiZero",
    "independent_response",
    "require_a_sum_over_states_regime",
    "response_sphere",
]

#: Below this the occupation difference of a pair is treated as zero and the
#: pair is dropped. ``genvchi0.f90``'s own ``1.d-8``.
_OCCUPATION_TOLERANCE = 1.0e-8


class ResponseSphere(eqx.Module):
    """The reciprocal lattice vectors ``chi_0`` is built on -- Elk's ``ngrf``.

    A cutoff of its own, and a small one: the body of ``chi_0`` is what carries
    the local-field effects, so it cannot be dropped, but it converges long
    before ``ecutrho`` and inheriting that cutoff makes ``nm`` -- and the cost,
    which is quadratic in it -- explode for nothing.

    ``G = 0`` is **not** in the body. In the optical limit it is the head, and
    the head is a 3x3 block of directions rather than a single entry.

    **An empty body is allowed and is a named approximation**, not a degenerate
    case: ``ecut = 0`` leaves the 3x3 head alone, which is the **head-only**
    kernel of the long-range-correction literature and is what Elk's own
    ``LiF-bootstrap`` example asks for with ``gmaxrf = 0.0``. Byun and Ullrich
    use it throughout and report that it differs little from the full matrix for
    the bootstrap family. It removes the local-field effect entirely, so
    ``eps_M`` then equals ``1 - X_head``; with the ``alda`` kernel, whose head
    and wings are identically zero, it is therefore *exactly* RPA.
    """

    #: ``(nbody,)`` flat indices into the smooth FFT box, for the gather that
    #: reads pair densities out of a transform.
    fft_index: jnp.ndarray
    #: ``(nbody,)`` ``sqrt(v(G)) = sqrt(8 pi / |G|^2)`` in Rydberg units, the
    #: factor that symmetrises ``chi_0``.
    sqrt_coulomb: jnp.ndarray
    #: ``(nbody,)`` the position of ``-G`` in this same list. A G-sphere is
    #: closed under inversion whatever the crystal's symmetry, so this always
    #: exists; it is what gives the antiresonant term for free.
    reflection: jnp.ndarray
    #: ``(nbody, 3)`` the signed Miller indices, carried rather than recovered
    #: from :attr:`fft_index`. A flat index is a statement about *one* FFT box,
    #: and the exchange-correlation kernel is built on the **dense** one where
    #: these vectors were selected on the smooth one -- so anything that has to
    #: index a second box needs the integers, not the wrapping.
    miller: jnp.ndarray
    ecut: float = eqx.field(static=True)

    @property
    def nbody(self) -> int:
        return self.fft_index.shape[0]

    @property
    def nm(self) -> int:
        """The full matrix dimension: three head directions plus the body."""
        return 3 + self.nbody


def response_sphere(calculation, ecut: float) -> ResponseSphere:
    """The G-vectors inside ``ecut`` (Ry), taken from the smooth set.

    The **smooth** set and not the dense one, because a pair density is formed
    on the smooth grid -- that is where the wavefunctions live -- and a
    component the smooth box cannot hold is an aliased one. The smooth cutoff is
    ``4 ecutwfc``, which is exactly the largest ``|G|`` a product of two
    wavefunctions has, so nothing is lost by the restriction.
    """
    gvectors = calculation.basis.smooth
    cell = calculation.system.cell
    g2 = np.asarray(gvectors.kinetic(cell))  # |G|^2 in 1/bohr^2 == Ry
    if ecut < 0.0:
        raise ValueError(
            f"the response cutoff cannot be negative, got {ecut}: it selects "
            "the G-vectors chi_0 is a matrix over (Elk's gmaxrf), and zero "
            "means the head alone"
        )
    inside = np.flatnonzero(g2 <= ecut)
    body = inside[inside != 0]  # G = 0 is the head, not a body entry

    miller = np.asarray(gvectors.miller)
    where = {tuple(m): n for n, m in enumerate(miller[body])}
    try:
        reflection = np.array([where[tuple(-m)] for m in miller[body]], dtype=np.int32)
    except KeyError as exc:  # pragma: no cover -- a G-sphere is always closed
        raise RuntimeError(
            "the response G-set is not closed under G -> -G, which a sphere "
            f"sorted by |G|^2 always is; missing {exc}"
        ) from exc

    precision = cell.precision
    return ResponseSphere(
        fft_index=jnp.asarray(np.asarray(gvectors.fft_index)[body]),
        sqrt_coulomb=precision.as_real(np.sqrt(E2 * FPI / g2[body])),
        reflection=jnp.asarray(reflection),
        miller=jnp.asarray(miller[body]),
        ecut=float(ecut),
    )


class ChiZero(eqx.Module):
    """``v^1/2 chi_0 v^1/2`` at every requested frequency, and what built it.

    ``x`` is ``(nw, nm, nm)``. The leading 3x3 block of each frequency's matrix
    is the head -- the optical limit, resolved by direction -- and the rest is
    the body over ``sphere``'s G-vectors.
    """

    x: jnp.ndarray
    frequencies: jnp.ndarray  # (nw,) complex: omega + i eta
    sphere: ResponseSphere
    #: How many valence-conduction pairs went into it, per k-point. Reported
    #: because band truncation is this phase's one unrefusable error.
    npairs: int = eqx.field(static=True)
    nocc: int = eqx.field(static=True)
    nbnd: int = eqx.field(static=True)

    @property
    def nm(self) -> int:
        return self.x.shape[-1]

    @property
    def head(self) -> jnp.ndarray:
        """``(nw, 3, 3)``: the optical block, which is what a spectrum reads."""
        return self.x[:, :3, :3]


def require_a_sum_over_states_regime(calculation) -> None:
    """What P37's ``chi_0`` refuses, checked on the calculation alone.

    Each of these is a missing term rather than a missing convenience, and each
    would otherwise produce a plausible spectrum:

    ``q != 0``
        not an argument here at all yet. The perturbed states would live at
        ``k + q``, which is the two-sphere machinery P19 built for spin spirals.
    ultrasoft and PAW
        ``<u_i| e^{-iG.r} |u_j>`` is not the plane-wave overlap when the charge
        is not all in ``|psi|^2``: the augmentation charge ``Q_ij(q+G)`` enters
        every matrix element. **It has been tried and it does not close**
        (``PLAN.md`` P40): with ``Q_ij(G)`` in the body -- gathered from the
        dense table by Miller index, and multiplied by the cell volume, which
        is a factor of 265 and was the first thing wrong -- and the head's
        ``q``-linear ``dpqq`` term beside it, ultrasoft silicon's ``eps_M(0)``
        is still **2.1%** from the Sternheimer solve where a norm-conserving
        control on the same machinery is at **0.06%**. So a third term is
        missing and the refusal stands until it is found; what is *excluded* is
        worth as much as what is not.
    metals
        the ``f_i - f_j`` weight kills the ``i = j`` term, so the intraband
        (Drude) response is missing entirely at ``q = 0``. Elk's ``tddftlr``
        does not add it either.
    ``nspin != 1``, noncollinear, spin-orbit
        the kernel acquires spin components (Elk's ``nscfxc``, and a separate
        routine ``tddftsplr.f90``).
    a reduced k-set
        ``genvchi0`` sums the **full** non-reduced grid, and so does this.
        Symmetrising ``chi_0(G, G')`` on a wedge is a rotation in *two* G
        indices at once, which the rank-N symmetriser P36 wrote is not -- it is
        Cartesian. The compensation is that the shifted-grid refusal of P24
        does not apply here, because nothing is being symmetrised.
    """
    system = calculation.system
    if calculation.is_ultrasoft or calculation.is_paw:
        raise NotImplementedError(
            "a sum-over-states chi_0 with an ultrasoft or PAW pseudopotential "
            "is not implemented: <u_i|e^{-iG.r}|u_j> gains the augmentation "
            "charge Q_ij(G) in every matrix element, and without it the matrix "
            "is wrong by the whole augmentation and still looks like a "
            "dielectric function. Use a norm-conserving dataset"
        )
    if calculation.nspin != 1:
        raise NotImplementedError(
            f"chi_0 here is restricted to nspin = 1 and this run has "
            f"nspin = {calculation.nspin}: with spin the kernel becomes a "
            "tensor in the spin components (Elk reaches it through a separate "
            "routine, tddftsplr.f90)"
        )
    if system.noncolin:
        raise NotImplementedError(
            "a noncollinear or spin-orbit chi_0 is not implemented here: the "
            "states are two-component spinors and the matrix element carries "
            "the spin trace"
        )
    if calculation.spiral:
        raise NotImplementedError(
            "chi_0 on a spin spiral is not implemented: the two spinor "
            "components live on spheres centred at k +- q/2, so a plane-wave "
            "matrix element is not a single gather"
        )
    if getattr(calculation, "is_hubbard", False):
        raise NotImplementedError(
            "chi_0 with a Hubbard U is not implemented: the Hubbard term "
            "responds to the density too (QE's adddvhubscf), and leaving it "
            "out of the screening is a silent approximation rather than a "
            "missing feature"
        )
    if not _is_an_insulator(system):
        raise NotImplementedError(
            "a sum-over-states chi_0 for a metal is not implemented: the "
            "occupation difference f_i - f_j vanishes for i = j, so the "
            "intraband (Drude) term is absent from this expression entirely "
            "and the spectrum would be an interband one wearing a metal's name"
        )
    if _kpoints_are_reduced(calculation):
        raise NotImplementedError(
            "chi_0 needs the full k-grid, not a symmetry-reduced wedge: "
            "chi_0(G, G') is a matrix in two G indices and symmetrising it "
            "under the point group rotates both at once, which nothing here "
            "implements. Run with nosym = .true. and noinv = .true. (an "
            "unshifted grid is closed under the point group, which is what "
            "makes that sound -- see PLAN.md P24 and P28b)"
        )


def _is_an_insulator(system) -> bool:
    occupations = getattr(system, "occupations", "fixed")
    return str(occupations).lower() in ("fixed", "from_input")


def _kpoints_are_reduced(calculation) -> bool:
    """Whether the k-set is a wedge rather than the whole grid.

    Read off the weights: an unreduced grid gives every k-point the same weight,
    and a reduction is exactly what makes them differ. That is a weaker test
    than comparing against the grid the input asked for, and it is the one that
    survives an explicit ``K_POINTS`` list -- which is how the closed-grid cases
    in ``tests/data/qe`` are written.
    """
    weights = np.asarray(calculation.system.kpoints.weights)
    if weights.size <= 1:
        return False
    return bool(np.ptp(weights) > 1e-8 * np.abs(weights).max())


def independent_response(
    calculation,
    wavefunctions,
    eigenvalues,
    v_scf,
    frequencies,
    *,
    ecut_response: float,
    broadening: float,
    scissor: float = 0.0,
    k_batch: int | None | str = "default",
) -> ChiZero:
    """``v^1/2 chi_0 v^1/2`` over a response sphere and a frequency grid.

    Args:
        calculation: the :class:`~defumat.scf.driver.Calculation` the states
            belong to. It must be on the **whole** k-grid; see
            :func:`require_a_sum_over_states_regime`.
        wavefunctions: ``(nspin, nk, nbnd, npwx)``, from a fixed-density run
            with empty states -- :func:`~defumat.workflows.nscf.
            fixed_density_states`. ``nbnd`` is the convergence parameter of this
            whole phase and there is no refusal that can catch it being too
            small, which is why :attr:`ChiZero.nbnd` is carried out.
        eigenvalues: ``(nspin, nk, nbnd)`` or the squeezed ``(nk, nbnd)``, Ry.
        v_scf: the converged potential, which ``dH/dk`` is built at.
        frequencies: ``(nw,)`` Ry. Real is the usual case and ``broadening``
            supplies the imaginary part; a **complex** array is accepted too and
            has ``broadening`` added to it, which is how a caller asks for one
            point at ``eta = 0`` beside a broadened grid. Elk's ``wrf`` is
            complex for the same reason.
        ecut_response: the cutoff (Ry) selecting the G-vectors of the matrix.
        broadening: ``eta``, Ry -- Elk's ``swidth``. Every frequency carries it,
            **including omega = 0**: ``init3.f90`` sets the bootstrap's static
            point to ``0 + i swidth`` rather than to zero, and the kernel is
            built there.
        scissor: a rigid shift (Ry) of the empty states, PRL 107, 186401's
            Eq. (3). The velocity matrix elements are renormalised with it.

    Returns:
        A :class:`ChiZero`. Nothing is symmetrised: on the full grid there is
        nothing to put back.
    """
    require_a_sum_over_states_regime(calculation)

    wavefunctions = jnp.asarray(wavefunctions)
    if wavefunctions.ndim == 3:
        wavefunctions = wavefunctions[None]
    eigenvalues = jnp.asarray(eigenvalues)
    if eigenvalues.ndim == 2:
        eigenvalues = eigenvalues[None]

    nbnd = int(eigenvalues.shape[-1])
    nocc = _occupied_bands(calculation, eigenvalues, nbnd)
    sphere = response_sphere(calculation, ecut_response)
    precision = calculation.system.cell.precision

    # The scissors shift moves the empty states and nothing else. It is applied
    # here rather than by the caller so that the matrix elements below can be
    # renormalised consistently -- the two are one approximation, not two.
    shifted = eigenvalues.at[..., nocc:].add(precision.as_real(scissor))

    # ``dH/dk``, the head's ingredient and the reason this is not a pure
    # transcription. One jvp per cartesian direction over the whole k axis.
    velocity = VelocityOperator(calculation, v_scf)
    elements = velocity.matrix_elements(wavefunctions)  # (3, nspin, nk, nb, nb)
    elements = jnp.moveaxis(elements[:, 0], 0, 1)  # (nk, 3, nbnd, nbnd)

    weights, _ = calculation.occupations(eigenvalues)
    weights = jnp.asarray(weights)[0]  # (nk, nbnd), summing to nelec

    rows, columns = _pairs(nocc, nbnd)
    zomega = jnp.asarray(frequencies) + 1j * precision.as_real(broadening)
    zomega = zomega.astype(precision.complex)

    grid = calculation.basis.smooth.grid
    volume = calculation.system.cell.volume
    mask = jnp.asarray(calculation.basis.planewaves.mask)
    batch = resolve_k_batch(k_batch)

    def one_k(arrays):
        psi, fft_index, band_mask, eig, occupation, element = arrays
        vectors, scalars = _pair_terms(
            psi, fft_index, band_mask, eig, occupation, element,
            rows, columns, sphere, grid, volume, zomega, scissor, precision,
        )
        # ``(nw, nm, nm)``: one matrix product per frequency, the pair axis
        # contracted away. This is the whole frequency cost of the phase.
        return jnp.einsum("wp,pa,pb->wab", scalars, vectors, jnp.conj(vectors))

    x = sum_k(
        one_k,
        (wavefunctions[0], calculation.fft_index, mask,
         shifted[0], weights, elements),
        batch=batch,
    )
    return ChiZero(
        x=x,
        frequencies=zomega,
        sphere=sphere,
        npairs=int(rows.size),
        nocc=nocc,
        nbnd=nbnd,
    )


def _pairs(nocc: int, nbnd: int):
    """The ordered ``(i, j)`` pairs with ``i`` occupied and ``j`` empty.

    Only that half is enumerated. The reversed pairs -- which
    ``genvchi0.f90``'s loop runs explicitly, and which are the antiresonant
    term -- come from reflecting these, in :func:`_pair_terms`, without a
    second transform.
    """
    rows, columns = np.meshgrid(
        np.arange(nocc), np.arange(nocc, nbnd), indexing="ij"
    )
    return rows.reshape(-1), columns.reshape(-1)


def _pair_terms(psi, fft_index, band_mask, eig, occupation, element,
                rows, columns, sphere, grid, volume, zomega, scissor,
                precision):
    """One k-point's pair vectors ``r`` and their frequency weights.

    Returns ``(vectors, scalars)`` of shapes ``(2 npairs, nm)`` and
    ``(nw, 2 npairs)``: the resonant pairs followed by the antiresonant ones,
    which are the same states read the other way round.
    """
    if sphere.nbody:
        # The states in real space, once. Padding must be zeroed before the
        # scatter, since padding entries share the index of G = 0.
        fields = g_to_r(psi * band_mask[None, :], fft_index, grid)

        # ``<u_i| e^{-iG.r} |u_j>`` for every pair, from one transform each. The
        # product is formed on the smooth grid, which holds every G a product of
        # two wavefunctions has.
        products = jnp.conj(fields[rows]) * fields[columns]
        body = r_to_g(products, sphere.fft_index) * sphere.sqrt_coulomb
    else:
        # **The head-only kernel needs no transform at all**, and the saving is
        # the whole of the phase's cost rather than a corner of it: with an
        # empty body there is no plane-wave matrix element to form, so the
        # states never go to real space and ``chi_0`` is built from the velocity
        # matrix elements alone. Writing the gather over an empty index would
        # still transform every pair and then keep none of it.
        body = jnp.zeros(rows.shape + (0,), dtype=psi.dtype)

    # The head. ``<u_i|du_j/dk_a> = <u_i|dH/dk_a|u_j> / (eps_j - eps_i)``, and
    # the Coulomb factor sqrt(8 pi / q^2) cancels the q the numerator is linear
    # in, which is what leaves a finite, direction-resolved 3x3 block.
    difference = eig[rows] - eig[columns]  # eps_i - eps_j, negative
    matrix = element[:, rows, columns]  # (3, npairs)
    if scissor:
        # ``getpmat.f90``: the eigenvalues have already moved, so the matrix
        # element has to move with them or <u|r|u'> silently changes. Here
        # ``i`` is occupied and ``j`` empty, which is Elk's second branch.
        matrix = matrix * (difference / (difference + scissor))[None, :]
    head = -jnp.sqrt(precision.as_real(E2 * FPI)) * matrix / difference[None, :]
    vectors = jnp.concatenate([head.T, body], axis=-1)  # (npairs, nm)

    # The reversed pair. ``<u_j|e^{-iG.r}|u_i> = conj(<u_i|e^{+iG.r}|u_j>)``,
    # so the body is this body reflected through the origin and conjugated;
    # the head reflects with a sign, because ``rho_0`` is linear in ``q``.
    reflected = jnp.concatenate(
        [-jnp.conj(vectors[:, :3]),
         jnp.conj(vectors[:, 3:][:, sphere.reflection])],
        axis=-1,
    )

    weight = (occupation[rows] - occupation[columns]) / volume
    resonant = weight[None, :] / (difference[None, :] + zomega[:, None])
    antiresonant = -weight[None, :] / (-difference[None, :] + zomega[:, None])
    return (
        jnp.concatenate([vectors, reflected], axis=0),
        jnp.concatenate([resonant, antiresonant], axis=-1),
    )


def _occupied_bands(calculation, eigenvalues, nbnd: int) -> int:
    """How many bands are filled, and a check that the question is well posed.

    An insulator has the same number at every k-point, which is what makes the
    valence-conduction split a *static* shape and the pair list a compile-time
    object. A run where it is not is a metal, and metals are refused above --
    but the refusal reads the input's ``occupations``, and a fixed-occupation
    input can still be handed a state that crosses. This catches that.
    """
    nocc = int(round(calculation.nelec / 2))
    if nocc < 1 or nocc >= nbnd:
        raise ValueError(
            f"a sum-over-states chi_0 needs empty states: this run has "
            f"{nocc} occupied bands out of {nbnd}. Run a fixed-density "
            "calculation with more bands (nbnd), which is the convergence "
            "parameter of the whole spectrum"
        )
    weights, _ = calculation.occupations(jnp.asarray(eigenvalues))
    weights = np.asarray(weights)[0]
    empty = np.abs(weights[:, nocc:]).max()
    if empty > _OCCUPATION_TOLERANCE:
        raise NotImplementedError(
            f"band {nocc} carries occupation {empty:.3e} at some k-point, so "
            "this state has no clean valence-conduction split -- it is a metal "
            "or a zero-gap system. The sum-over-states chi_0 here is the "
            "interband expression and has no intraband term"
        )
    return nocc
