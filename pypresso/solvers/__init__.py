"""Eigensolvers, behind a name registry.

Adding a solver is a new file plus one registration, not an edit to a growing
branch in the driver (rule R4). Every entry has the same signature,

    solver(hamiltonian, nbnd, psi0=None) -> (eigenvalues, wavefunctions)

with shapes ``(nk, nbnd)`` and ``(nk, nbnd, npwx)``: the k-point axis is the
solver's business, because how to batch over it is part of the algorithm -- how
many k-points are in flight is :mod:`pypresso.batching`'s dial here, but a
solver that distributed them over devices would decide differently.

``psi0`` is the previous estimate of the wavefunctions. An iterative solver uses
it and converges in a step or two; a direct solver would ignore it. Callers pass
it whenever they have one and never have to ask which kind of solver they hold.

* ``davidson`` -- QE's default, and the only entry. Iterative, touches only the
  lowest ``nbnd`` states, and the only kind of solver that scales to a real
  system.

There is deliberately no dense solver. Forming ``H`` costs ``O(npw^2)`` memory
and diagonalising it ``O(npw^3)`` time -- the largest allocation a plane-wave
code can make, and the thing an iterative solver exists to avoid -- so offering
it as a ``diagonalization`` name invites it to be chosen. Correctness comes from
Quantum ESPRESSO, input for input; where a fast test wants an exact answer on a
two-hundred-plane-wave cell it forms the matrix itself, in
``tests/exact_reference.py``.
"""

from __future__ import annotations

from pypresso.solvers.davidson import davidson_eigensolver, davidson_eigensolver_all

__all__ = [
    "EIGENSOLVERS",
    "get_eigensolver",
    "davidson_eigensolver",
    "davidson_eigensolver_all",
]

#: Name -> solver, as written in an input file's ``diagonalization``.
EIGENSOLVERS = {
    "davidson": davidson_eigensolver_all,
    "david": davidson_eigensolver_all,
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
