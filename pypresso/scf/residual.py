"""The SCF as a root-find: the residual whose zero is self-consistency.

``scf/driver.py`` solves ``rho = F(rho)`` by iterating ``F`` and damping the
iteration with a mixer. This module writes the same fixed point as a *residual*

    r(x) = F(x) - x,   x = the packed (rho, becsum) state,

so that a solver which knows the derivative of ``r`` can drive it to zero
instead. ``F`` is one SCF iteration with every host-side decision frozen -- one
diagonalisation at a fixed ``ethr``, no retry, no ``ethr`` schedule, no field
feedback -- which is what makes it a *function* rather than a step of a loop.

**Calibration, because this is easy to oversell.** Anderson mixing already is a
quasi-Newton method on this residual: ``mixing.py`` builds a secant
approximation to ``dr/dx`` from the residual history and takes the Newton step
inside its span, and QE's Broyden (``mix_rho.f90``) is the same idea. Nothing
here replaces iteration with a closed form; what changes is that the Jacobian is
*exact* rather than fitted, which matters only where the fit is bad.

**Why the Jacobian comes from ``jax.jvp`` and not from a tape.** ``r`` is built
from ``Calculation``'s own methods, so it is one pure JAX function from the
density to the density, Davidson's ``lax.while_loop`` included. Forward-mode
differentiation of a ``while_loop`` is supported (reverse-mode is not, and its
tape would be ``n_davidson_steps * nvecx * nbnd * npwx`` complex numbers -- see
``PLAN.md`` P22), and forward mode is all a Krylov solver needs: it asks only for
``J v``.

Two things had to be written down rather than differentiated:

* **The Fermi level.** ``bisect_fermi`` is a bisection, and a bisection's
  tangent is zero or garbage. It now carries a ``custom_jvp`` with the implicit
  derivative of ``N(E_F) = nelec`` (``scf/occupations.py``), which is the
  Fermi-shift term of metallic linear response.
* **Nothing else, yet.** Davidson is differentiated as implemented, so the
  Jacobian is the Jacobian of the *approximate* eigensolver and is accurate to
  about ``ethr``. That is enough for a Krylov solver, which only needs a
  direction; it is not the Sternheimer response, which would be exact and is
  what ``PLAN.md`` P22c plans.

**DFT+U joins the state rather than being refused.** ``ns`` is not a function of
the density -- the Hubbard potential is built from it before the Hamiltonian
exists -- so ``mix_rho.f90`` carries it inside ``mix_type`` and the mixing loop
mixes it alongside ``rho`` and ``becsum``. A root-finder solves for it on the
same footing, for the same reason, and nothing about the Jacobian action changes:
``v_hubbard`` is already ``jax.grad`` of the Hubbard energy (P20) and the
projectors ``wfcU`` are fixed while the atoms are.

**Refused rather than approximated:** external and constrained magnetic fields
(the field is driven by a secant *outside* the density, so ``F`` would not be a
function of ``x`` alone), tetrahedron and ``from_input`` occupations (built on
the host, and the second is not a function of the eigenvalues at all), and spin
spirals. Each raises by name.

Memory: the packed state is ``nspin * dense_grid`` floats plus ``becsum`` and
``ns``, and a Krylov subspace holds a few tens of those -- tens of MB,
negligible beside the wavefunctions. One ``jvp`` doubles the wavefunction
working set for the duration of the call, because the tangent ``psi`` has the
same shape as ``psi``; that is the number to watch, and it is why the JVP goes
through the same ``k_batch`` dial the primal does.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from pypresso.scf.occupations import fixed_occupations, smeared_occupations

__all__ = ["ScfResidual", "make_residual"]


@dataclass(frozen=True)
class ScfResidual:
    """``F`` and ``r = F - x`` as flat real vectors, plus the packing.

    ``psi0`` is a warm start for Davidson carried *outside* the differentiated
    function: it changes how many iterations the solver takes and not what it
    converges to, so it is a constant of ``F`` at every point where ``F`` is
    evaluated, and updating it between outer iterations is legitimate. It is
    also the reason ``step`` returns the wavefunctions: the caller feeds them
    back.
    """

    calculation: object
    nbnd: int
    ethr: float
    shapes: tuple
    size: int
    #: Shape of the DFT+U occupation matrix, or ``None`` without a ``U``.
    #: ``ns`` is part of the *mixed state* rather than a function of the
    #: density (``mix_rho.f90`` carries it in ``mix_type`` for that reason), so
    #: it is part of the state a root-finder solves for on exactly the same
    #: footing.
    ns_shape: tuple | None = None

    # ---- packing -------------------------------------------------------
    def pack(self, rho, becsum_, ns=None) -> np.ndarray:
        """The mixed state as one flat real vector, exactly ``_mix``'s packing."""
        flat = [np.asarray(rho, dtype=float).ravel()]
        for part in becsum_:
            if part is not None:
                flat.append(np.asarray(part, dtype=float).ravel())
        if self.ns_shape is not None:
            if ns is None:
                raise ValueError("this calculation has a Hubbard U; ns is part of the state")
            flat.append(np.asarray(ns, dtype=float).ravel())
        return np.concatenate(flat)

    def unpack(self, x):
        """The inverse of :meth:`pack`. Returns ``(rho, becsum, ns)``."""
        x = jnp.asarray(x)
        rho_shape = self.shapes[0]
        n = int(np.prod(rho_shape))
        rho = x[:n].reshape(rho_shape)
        offset = n
        becsum_ = []
        for shape in self.shapes[1:]:
            if shape is None:
                becsum_.append(None)
                continue
            n = int(np.prod(shape))
            becsum_.append(x[offset : offset + n].reshape(shape))
            offset += n
        ns = None
        if self.ns_shape is not None:
            ns = x[offset : offset + int(np.prod(self.ns_shape))].reshape(self.ns_shape)
        return rho, tuple(becsum_), ns

    # ---- the map -------------------------------------------------------
    def step(self, x, psi0):
        """One SCF iteration as a pure function: ``x -> F(x)``, and the ``psi``."""
        calculation = self.calculation
        rho, becsum_, ns = self.unpack(x)
        potential = calculation.potential(rho)
        _, ddd_paw = calculation.onecenter(becsum_)
        hubbard_terms = None
        if ns is not None:
            # ``v_hubbard`` is ``jax.grad`` of the Hubbard energy (P20), so this
            # whole branch is differentiable without anything being added for it.
            _, _, hubbard_terms = calculation.hubbard_terms(ns)
        hamiltonians = calculation.hamiltonian(potential.v_scf, ddd_paw, hubbard_terms)
        eigenvalues, psi = calculation.diagonalize(
            hamiltonians, self.nbnd, psi0, self.ethr
        )
        wg = _weights(calculation, eigenvalues)
        becsum_out = calculation.becsum(psi, wg)
        rho_out = calculation.density(psi, wg, becsum_out)
        ns_out = None if ns is None else calculation.occupation_matrix(psi, wg)
        return self.flatten(rho_out, becsum_out, ns_out), psi

    def flatten(self, rho, becsum_, ns=None):
        """:meth:`pack`, inside a trace -- ``jnp`` rather than ``np``."""
        flat = [jnp.reshape(jnp.real(rho), (-1,))]
        for part in becsum_:
            if part is not None:
                flat.append(jnp.reshape(jnp.real(part), (-1,)))
        if ns is not None:
            flat.append(jnp.reshape(jnp.real(ns), (-1,)))
        return jnp.concatenate(flat)

    def residual(self, x, psi0):
        """``r(x) = F(x) - x``, and the wavefunctions ``F`` produced."""
        fx, psi = self.step(x, psi0)
        return fx - jnp.asarray(x), psi

    def jvp(self, x, v, psi0):
        """``J v = d r / d x . v``, by forward-mode differentiation of ``F``.

        ``psi0`` is passed as a closed-over constant, so no tangent flows
        through the warm start -- which is right: it is not part of the map
        being differentiated, only of how fast it is evaluated.
        """
        def f(y):
            return self.step(y, psi0)[0]

        primal, tangent = jax.jvp(f, (jnp.asarray(x),), (jnp.asarray(v),))
        return np.asarray(tangent - jnp.asarray(v)), np.asarray(primal)

    def jvp_finite_difference(self, x, v, psi0, epsilon=None):
        """The same ``J v``, by a central difference of ``r``.

        This is the *test* of :meth:`jvp`, in the same spirit as QE's
        hand-derived forces standing as the test of the autodiff ones: the two
        share no machinery, so agreeing means both are right. It is also the
        fallback for a regime where the autodiff route is refused, and it costs
        two evaluations of ``F`` against one ``jvp``.
        """
        x = np.asarray(x)
        v = np.asarray(v)
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            return np.zeros_like(x)
        if epsilon is None:
            # Nocedal-Wright's forward-difference step, scaled by the state's
            # own magnitude so that a nearly converged density is not perturbed
            # below the eigensolver's own noise floor.
            epsilon = 1.0e-6 * max(1.0, float(np.linalg.norm(x))) / norm
        plus, _ = self.residual(x + epsilon * v, psi0)
        minus, _ = self.residual(x - epsilon * v, psi0)
        return np.asarray((plus - minus) / (2.0 * epsilon))


def _weights(calculation, eigenvalues):
    """``wg`` alone, and traceable.

    ``Calculation.occupations`` cannot be used here: it converts the Fermi level
    and the entropy to Python floats for the iteration report, which is a host
    sync and a trace error. This is the same dispatch with only the weights kept
    -- the two must agree, and ``test_weights_agree_with_the_driver`` checks
    that they do rather than leaving it to inspection.
    """
    system = calculation.system
    scheme = system.occupations
    if scheme.startswith("tetrahedra") or scheme == "from_input":
        raise NotImplementedError(
            f"occupations={scheme!r} has no differentiable form here: the "
            "tetrahedron weights are built on the host from the k-grid and the "
            "symmetry, and 'from_input' is not a function of the eigenvalues at "
            "all. Use occupations='smearing' or 'fixed' with a residual solver"
        )
    degeneracy = 1 if calculation.noncolin else 2
    weights = system.kpoints.weights

    if scheme == "fixed":
        return fixed_occupations(eigenvalues, weights, calculation.nelec, degeneracy)[0]
    counts = (calculation.nelup, calculation.neldw) if calculation.two_fermi_energies else None
    wg, _ = smeared_occupations(
        eigenvalues, weights, calculation.nelec, system.degauss,
        system.smearing, counts=counts,
    )
    return wg


def make_residual(calculation, nbnd: int, ethr: float) -> ScfResidual:
    """Build the residual for a calculation, refusing the regimes it cannot hold."""
    system = calculation.system
    if calculation.magnetic_field is not None:
        raise NotImplementedError(
            "an external or constraining magnetic field has no residual form "
            "here: the field is driven by a secant *outside* the density "
            "(reducebf, the fixed-spin-moment feedback), so F changes between "
            "evaluations and is not a function of the state alone"
        )
    if getattr(system, "spiral_q", None) is not None:
        raise NotImplementedError("spin spirals are not supported by a residual solver")

    rho = calculation.starting_density()
    becsum_ = calculation.starting_becsum()
    shapes = (tuple(np.shape(rho)),) + tuple(
        None if part is None else tuple(np.shape(part)) for part in becsum_
    )
    ns_shape = (
        tuple(np.shape(calculation.starting_ns())) if calculation.is_hubbard else None
    )
    size = int(np.prod(shapes[0])) + sum(
        int(np.prod(s)) for s in shapes[1:] if s is not None
    )
    if ns_shape is not None:
        size += int(np.prod(ns_shape))
    return ScfResidual(calculation, nbnd, ethr, shapes, size, ns_shape)
