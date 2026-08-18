"""Eigensolvers, behind a name registry.

Adding a solver is a new file plus one registration, not an edit to a growing
branch in the driver (rule R4). Every entry has the same signature,

    solver(hamiltonian, nbnd, psi0=None) -> (eigenvalues, wavefunctions)

with shapes ``(nk, nbnd)`` and ``(nk, nbnd, npwx)``: the k-point axis is the
solver's business, because how to batch over it is part of the algorithm --
``vmap`` for both of the solvers here, but not necessarily for one that needs to
distribute k-points over devices.

``psi0`` is the previous estimate of the wavefunctions. An iterative solver uses
it and converges in a step or two; a direct solver ignores it. Callers pass it
whenever they have one and never have to ask which kind of solver they hold.

* ``dense`` -- form the matrix and diagonalise it. ``O(npw^3)``, exact, and the
  ground truth every other solver is checked against.
* ``davidson`` -- QE's default. Iterative, touches only the lowest ``nbnd``
  states, and the only choice that scales to a real system.
"""

from __future__ import annotations

from pypresso.solvers.davidson import davidson_eigensolver, davidson_eigensolver_all
from pypresso.solvers.dense import dense_eigensolver, dense_eigensolver_all

__all__ = [
    "EIGENSOLVERS",
    "get_eigensolver",
    "dense_eigensolver",
    "dense_eigensolver_all",
    "davidson_eigensolver",
    "davidson_eigensolver_all",
]

#: Name -> solver, as written in an input file's ``diagonalization``.
EIGENSOLVERS = {
    "davidson": davidson_eigensolver_all,
    "david": davidson_eigensolver_all,
    "dense": dense_eigensolver_all,
    "exact": dense_eigensolver_all,
}

#: What a calculation uses unless it asks for something else.
DEFAULT_EIGENSOLVER = "davidson"


def get_eigensolver(name: str | None = None):
    """Look up a solver by the name an input file would use."""
    name = (name or DEFAULT_EIGENSOLVER).lower()
    try:
        return EIGENSOLVERS[name]
    except KeyError as error:
        raise ValueError(
            f"unknown diagonalization {name!r}; expected one of {sorted(EIGENSOLVERS)}"
        ) from error
