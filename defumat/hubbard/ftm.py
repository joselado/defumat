"""Fixing a tensor moment, and letting the SCF find the state that has it.

The decomposition of :mod:`defumat.hubbard.tensormoments` says what is in an
occupation matrix. This *chooses* what is in it: name one moment
``w^{kpr}_t``, give it a value, and the self-consistent field converges to a
state carrying it. That selects an orbital or multipolar ordering the SCF would
not find on its own -- there is usually more than one solution and the one a run
lands in is decided by where it starts -- and the energy against the
unconstrained state is what the constraint is for.

Elk's ``ftmtype``/``tm3fix`` (``vmatmtftm.f90``). ``pw.x`` has nothing of the
kind, and the *quantity* it constrains does not exist there either.

**The penalty is written down and the potential is ``jax.grad`` of it**, which
is P18's rule for the constrained moments and is the one deviation from Elk
here: ``vmatmtftm`` accumulates a potential by proportional feedback --
measure the moment, subtract the target, scale by ``tauftm``, add to a running
matrix -- which is the same scheme ``bfieldfsm.f90`` uses for a fixed spin
moment and is a controller rather than a functional. Written as

    E_ftm = (lambda / 2) sum_i ( w_i - w_i^target )^2

the potential is a derivative of something, the constraint has a stationary
point rather than a fixed point of an iteration, and the two are the same
answer wherever the controller converges.

**The penalty's energy is not part of the reported total**, and that is P18's
convention as well: a constrained run is a run of a different functional, and
adding the penalty to the total would make the two incomparable. It is carried
separately, and at convergence it is what says how hard the constraint had to
push -- a penalty energy near zero means the state was there anyway.
"""

from __future__ import annotations

import numpy as np

__all__ = ["TensorMomentConstraint", "build_constraints", "constraint_energy"]


class TensorMomentConstraint:
    """The constraints of a run, resolved onto slots and moment matrices.

    ``gamma`` is ``(nfix, 4, ldmx, ldmx)`` complex -- the moment matrix of each
    constrained moment, padded to the largest manifold -- ``slot`` says which
    correlated atom each belongs to, and ``target`` what it is fixed at.
    """

    __slots__ = ("gamma", "slot", "target", "penalty", "labels", "scale")

    def __init__(self, gamma, slot, target, penalty, labels, scale):
        self.gamma = gamma
        self.slot = slot
        self.target = target
        self.penalty = penalty
        self.labels = labels
        #: ``sqrt(2(2l+1))`` per entry: ``vmatmtftm``'s "conventional
        #: normalisation", which is what makes a full shell's ``w^{000}_0`` the
        #: number of electrons rather than that over ``sqrt(2(2l+1))``. The
        #: targets are stored *divided* by it and reported multiplied, so the
        #: input and the output speak the same units and only this module sees
        #: the basis's own.
        self.scale = scale

    def __len__(self) -> int:
        return len(self.target)


def build_constraints(setup, entries, penalty: float, names):
    """Resolve ``(species, n, l, k, p, r, t, value)`` entries against the setup.

    ``names`` is the ``ATOMIC_SPECIES`` list, which is what the card's labels
    refer to -- the two magnetic sublattices of an antiferromagnet are different
    species precisely so that they can be constrained differently. ``None`` comes
    back when there is nothing to fix, which is what makes every call site's test
    a single ``is None``.
    """
    from defumat.hubbard.manifold import manifold_label
    from defumat.hubbard.tensormoments import MomentLabel, moment_labels, moment_matrices

    if not entries:
        return None
    if not setup.noncolin:
        raise NotImplementedError(
            "a fixed tensor moment needs a spinor occupation matrix "
            "(noncolin = .true.): for a collinear run the spin-rank moments "
            "keep only their z component, so the constraint degenerates into "
            "the fixed spin moment constrained_magnetization already is"
        )

    gammas, slots, targets, labels, scales = [], [], [], [], []
    for species, n, l, k, p, r, t, value in entries:
        if species not in names:
            raise ValueError(
                f"TENSOR_MOMENTS names {species!r}, which is not in ATOMIC_SPECIES"
            )
        kind = names.index(species)
        found = [slot for slot, other in enumerate(setup.types) if other == kind]
        item = setup.species[kind]
        if not found or item is None or (item.n, item.l) != (n, l):
            raise ValueError(
                f"{species} has no Hubbard {manifold_label(n, l)} manifold to fix "
                "a tensor moment of"
            )
        label = MomentLabel((k, p, r, t))
        catalogue = moment_labels(l)
        if label not in catalogue:
            raise ValueError(
                f"{label} is not a moment of an l = {l} shell: k runs 0..{2 * l}, "
                "p is 0 or 1, r runs |k-p|..k+p and t runs -r..r"
            )
        basis = moment_matrices(l)[catalogue.index(label)]   # (2, 2, ldim, ldim)
        width = basis.shape[-1]
        padded = np.zeros((4, setup.ldmx, setup.ldmx), dtype=complex)
        padded[:, :width, :width] = basis.reshape(4, width, width)
        scale = np.sqrt(2.0 * (2 * l + 1))
        for slot in found:
            gammas.append(padded)
            slots.append(slot)
            targets.append(value / scale)
            scales.append(scale)
            labels.append((setup.atoms[slot], label))

    return TensorMomentConstraint(
        gamma=np.stack(gammas),
        slot=np.asarray(slots, dtype=int),
        target=np.asarray(targets, dtype=float),
        penalty=float(penalty),
        labels=tuple(labels),
        scale=np.asarray(scales, dtype=float),
    )


def constraint_energy(ns, constraint):
    """``(lambda/2) sum (w - w_target)^2`` in Ry, differentiable in ``ns``."""
    import jax.numpy as jnp

    if constraint is None:
        return jnp.asarray(0.0)
    blocks = ns[:, constraint.slot]                    # (4, nfix, ldmx, ldmx)
    measured = jnp.real(
        jnp.einsum("isab,siab->i", jnp.conj(constraint.gamma), blocks)
    )
    return 0.5 * constraint.penalty * jnp.sum(
        (measured - constraint.target) ** 2
    )


def measured_moments(ns, constraint) -> np.ndarray:
    """The constrained moments of an occupation matrix, in Elk's normalisation."""
    if constraint is None:
        return np.zeros(0)
    blocks = np.asarray(ns)[:, constraint.slot]
    measured = np.real(
        np.einsum("isab,siab->i", np.conj(constraint.gamma), blocks)
    )
    return measured * constraint.scale
