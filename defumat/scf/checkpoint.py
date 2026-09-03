"""A converged state on disk, so a wall clock does not cost a calculation.

Nothing here was serialisable before: ``run_scf(starting_from=result)`` needs
the object in memory, so a job that hits its time limit loses the run, and two
processes cannot share a ground state at all. On a cell whose SCF is measured in
hours that is the difference between a calculation and a calculation that has to
be restarted from the beginning.

**What is saved is the state, not the system.** A :class:`System` carries
pseudopotential file names, a precision policy and a symmetry group, and writing
those to a file makes the file a second, weaker input parser. The resume path a
cluster job actually takes already has the input: it rebuilds the
:class:`~defumat.calculator.Calculator` from ``scf.in`` and hands the system in.
So what is stored beside the arrays is a **fingerprint** -- the shapes and the
counts a continuation would have to agree on anyway -- and loading against a
system that does not match is refused rather than discovered three iterations
later as a wrong answer.

**Every field of** :class:`~defumat.scf.driver.SCFResult` **is accounted for**,
and that is enforced rather than intended: :data:`_ARRAYS`, :data:`_SCALARS` and
:data:`_REFUSED` must together cover ``dataclasses.fields(SCFResult)``, and
``tests/unit/test_checkpoint.py`` asserts it. A field added to the result and
not to one of those three lists fails the test instead of being silently
dropped -- which is the failure mode that matters here, since a checkpoint that
quietly loses ``becsum`` reloads as a *different* state that converges to
something plausible.

Two things are refused by name rather than half-saved, both because losing them
is silent. A converged **magnetic field** is not the input field wherever
``reducebf`` or the fixed-spin-moment scheme changed it, and reloading a state
without it applies a rigid Zeeman shift that a later invariant still returns an
integer for (``PLAN.md`` P56 is the record of that bug found the hard way). And
a **Hubbard setup** is what says which atom each slot of ``ns`` belongs to, so
``ns`` without it is an array of numbers about nothing.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np

__all__ = ["save_state", "load_state", "state_fingerprint"]

#: Array-valued fields, stored as their own entries in the ``.npz``. ``None`` is
#: representable: the key is simply absent.
_ARRAYS = (
    "eigenvalues", "occupations", "wavefunctions", "density", "potential",
    "potential_change", "ns",
    # ``tau`` is *state*, not a diagnostic: under a meta-GGA the potential is
    # rebuilt from the density **and** this, and a fixed density alone does not
    # determine it. It was the field the coverage check above was written for --
    # it was missing from the first draft of this list, and a reloaded meta-GGA
    # run would have silently rebuilt a different potential.
    "tau",
)

#: Everything JSON can hold. ``energy_terms`` is a dict of floats and
#: ``magnetization_vector`` a tuple of them.
_SCALARS = (
    "converged", "iterations", "total_energy", "energy_terms", "fermi_energy",
    "homo", "lumo", "accuracy", "nspin", "magnetization",
    "absolute_magnetization", "fermi_energy_up", "fermi_energy_down",
    "magnetization_vector", "field_energy", "constraint_energy", "field_scale",
    "nspin_mag", "meta_c",
)

#: Handled specially: a tuple of arrays, one per ultrasoft/PAW species.
_TUPLES = ("becsum",)

#: Not saved, and refused rather than dropped. See the module docstring.
_REFUSED = ("magnetic_field", "hubbard_setup")

#: Saved by nothing and dropped on purpose: these are what the run *reported*,
#: not what it converged to. A checkpoint is a state to continue from -- the
#: stress is recomputed from the state in one strain gradient, the solver record
#: and the iteration history describe a run that is over. A reloaded result is
#: therefore a state and not a report, which :func:`load_state` says in its
#: docstring rather than leaving to be noticed.
_DROPPED = ("stress", "solver", "history")

#: Reconstructed from the ``system`` the caller supplies on load.
_FROM_CALLER = ("system",)

#: Bumped when the layout changes in a way an older file cannot be read as.
FORMAT_VERSION = 1


def state_fingerprint(result) -> dict:
    """The shapes and counts a continuation has to agree on.

    Not a hash of the input file: two runs that differ in ``conv_thr`` or in
    their mixing share a state perfectly well, and refusing that would make
    checkpoints useless for exactly the restart they exist for. What must match
    is the basis, the k-set, the spin regime and the electron count -- which is
    what :func:`~defumat.scf.continuation._check_grid` checks, one layer down.
    """
    density = np.shape(result.density)
    wavefunctions = np.shape(result.wavefunctions)
    system = getattr(result, "system", None)
    return {
        "density_shape": list(density),
        "wavefunction_shape": list(wavefunctions),
        "nspin": int(result.nspin),
        "nspin_mag": int(result.nspin_mag),
        "nbecsum": len(result.becsum or ()),
        "natoms": (None if system is None
                   else len(np.asarray(system.structure.types))),
        "types": (None if system is None
                  else [int(t) for t in np.asarray(system.structure.types)]),
    }


def _describe(fingerprint) -> str:
    return (f"density {tuple(fingerprint['density_shape'])}, "
            f"wavefunctions {tuple(fingerprint['wavefunction_shape'])}, "
            f"nspin {fingerprint['nspin']}")


def save_state(result, path) -> Path:
    """Write a converged :class:`SCFResult` to ``path`` as a ``.npz``.

    The wavefunctions dominate the file -- ``nspin nk nbnd npwx`` complex, which
    is tens of gigabytes on a large cell -- so this is a scratch-directory
    operation, not something to do every iteration without meaning to.
    """
    path = Path(path)
    for name in _REFUSED:
        if getattr(result, name, None) is not None:
            raise NotImplementedError(
                f"this result carries {name!r}, which a checkpoint cannot "
                "represent, and dropping it would be silent: a converged "
                "magnetic field is not the input field wherever reducebf or a "
                "fixed spin moment changed it, and a Hubbard setup is what says "
                "which atom each slot of ns belongs to. Saving is refused "
                "rather than lossy"
            )

    payload, meta = {}, {"format": FORMAT_VERSION}
    for name in _ARRAYS:
        value = getattr(result, name, None)
        if value is not None:
            payload[name] = np.asarray(value)
    for name in _SCALARS:
        meta[name] = getattr(result, name, None)
    becsum = result.becsum or ()
    meta["nbecsum"] = len(becsum)
    for index, block in enumerate(becsum):
        payload[f"becsum_{index}"] = np.asarray(block)
    meta["fingerprint"] = state_fingerprint(result)

    payload["__meta__"] = np.frombuffer(
        json.dumps(meta).encode("utf-8"), dtype=np.uint8
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written beside the target and moved into place, so an interrupted write
    # never leaves a half file that a resume would read as a state.
    scratch = path.with_suffix(path.suffix + ".partial")
    np.savez(scratch, **payload)
    written = scratch.with_suffix(scratch.suffix + ".npz") if not scratch.suffix.endswith(".npz") else scratch
    written.replace(path)
    return path


def load_state(path, system=None, calculation=None, strict: bool = True):
    """Read back what :func:`save_state` wrote.

    Args:
        path: the ``.npz``.
        system: the :class:`System` the state belongs to. Supplied rather than
            stored -- see the module docstring -- and checked against the
            fingerprint.
        calculation: optional; when given, the basis is checked too, which
            catches a cutoff change that the fingerprint alone would not.
        strict: whether a fingerprint mismatch raises. ``False`` warns instead,
            which is for inspecting a file rather than for running from one.

    **What comes back is a state, not a report.** ``stress``, ``solver`` and
    ``history`` describe the run that produced the file and are not stored, so
    they are at their defaults here. Everything a continuation consumes --
    the density, ``becsum``, ``ns``, ``tau`` and the wavefunctions -- is.
    """
    from defumat.scf.driver import SCFResult

    path = Path(path)
    with np.load(path, allow_pickle=False) as handle:
        meta = json.loads(bytes(handle["__meta__"]).decode("utf-8"))
        if meta.get("format") != FORMAT_VERSION:
            raise ValueError(
                f"{path} is checkpoint format {meta.get('format')}, and this is "
                f"version {FORMAT_VERSION}"
            )
        fields = {name: meta[name] for name in _SCALARS if name in meta}
        if fields.get("magnetization_vector") is not None:
            fields["magnetization_vector"] = tuple(fields["magnetization_vector"])
        arrays = {
            name: jnp.asarray(handle[name]) for name in _ARRAYS if name in handle
        }
        # ``eigenvalues`` and ``occupations`` are numpy on the result, not JAX:
        # they are indexed and sliced host-side throughout.
        for name in ("eigenvalues", "occupations"):
            if name in arrays:
                arrays[name] = np.asarray(handle[name])
        becsum = tuple(
            jnp.asarray(handle[f"becsum_{index}"])
            for index in range(int(meta.get("nbecsum", 0)))
        )

    result = SCFResult(
        **fields, **arrays, becsum=becsum, system=system,
    )
    if system is not None:
        _check_fingerprint(meta["fingerprint"], state_fingerprint(result),
                           path, strict)
    if calculation is not None:
        grid = tuple(calculation.basis.dense.grid)
        shape = tuple(np.shape(result.density))[1:]
        if shape != grid:
            raise ValueError(
                f"{path} holds a density on a {shape} grid and this calculation "
                f"uses {grid}: the cell or a cutoff has changed"
            )
    return result


def _check_fingerprint(stored, rebuilt, path, strict) -> None:
    """The stored state and the supplied system must be the same calculation."""
    differences = [
        key for key in ("density_shape", "wavefunction_shape", "nspin",
                        "nspin_mag", "nbecsum", "types")
        # ``types`` is ``None`` in the rebuilt copy when no system was given,
        # and in the stored one when the result that was saved had none.
        if stored.get(key) is not None and rebuilt.get(key) is not None
        and stored[key] != rebuilt[key]
    ]
    if not differences:
        return
    message = (
        f"{path} does not describe this system: {', '.join(differences)} "
        f"differ. Stored {_describe(stored)}; this run {_describe(rebuilt)}"
    )
    if strict:
        raise ValueError(message)
    warnings.warn(message, stacklevel=3)


def _covered_fields() -> set:
    """Every field name the lists above claim to handle."""
    return (set(_ARRAYS) | set(_SCALARS) | set(_TUPLES) | set(_REFUSED)
            | set(_FROM_CALLER) | set(_DROPPED))


def unhandled_fields() -> set:
    """Fields of ``SCFResult`` no list mentions -- empty, and tested to be.

    A checkpoint that silently omits a field reloads as a different state, so
    the coverage is asserted rather than maintained by hand.
    """
    from defumat.scf.driver import SCFResult

    return {f.name for f in dataclasses.fields(SCFResult)} - _covered_fields()


# --- the optimizer's own state ---------------------------------------------
#
# A relaxation that resumes from positions alone throws away the inverse Hessian
# and the trust radius, so it takes its first step as if it were step one -- and
# BFGS earns its convergence rate entirely from that history. Restarting a
# half-finished relaxation without it is close to restarting it.

#: Rebuilt by constructing a fresh :class:`~defumat.relax.bfgs.BFGS` with the
#: same cell and settings, so they are not stored. ``at``, the metric and the
#: volume are functions of the cell; ``settings`` and the thresholds come from
#: the caller; ``h`` and the metric blocks are derived in ``_setup``.
_OPTIMIZER_DERIVED = frozenset({
    "at", "h", "metric", "inverse_metric", "metric_blocks",
    "inverse_metric_blocks", "omega", "settings", "energy_thr", "grad_thr",
    "cell_thr", "variable_cell", "pressure", "cell_mask",
})


def save_optimizer(optimizer, path) -> Path:
    """Write a :class:`~defumat.relax.bfgs.BFGS`'s evolving state.

    Everything in its ``__dict__`` that is not derivable from the cell and the
    settings -- the inverse Hessian above all, but also the trust radius and the
    previous step, which together decide the next one. What is *not* stored is
    listed in :data:`_OPTIMIZER_DERIVED` and is reconstructed by building the
    optimizer normally before the state is poured back in.
    """
    path = Path(path)
    payload, meta = {}, {"format": FORMAT_VERSION}
    for name, value in vars(optimizer).items():
        if name in _OPTIMIZER_DERIVED:
            continue
        if isinstance(value, np.ndarray):
            payload[f"array_{name}"] = value
        elif value is None or isinstance(value, (bool, int, float, str)):
            # ``np.inf`` survives JSON as ``Infinity``, which ``json`` reads
            # back; the errors start at infinity and a resume before the first
            # step has to see that rather than a zero.
            meta[name] = value
        else:
            raise NotImplementedError(
                f"the optimizer holds {name!r} of type {type(value).__name__}, "
                "which this checkpoint cannot represent. Add it to "
                "_OPTIMIZER_DERIVED if it is rebuilt from the cell and the "
                "settings, or extend this function"
            )
    payload["__meta__"] = np.frombuffer(
        json.dumps(meta).encode("utf-8"), dtype=np.uint8
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(path.suffix + ".partial")
    np.savez(scratch, **payload)
    written = (scratch if scratch.suffix.endswith(".npz")
               else scratch.with_suffix(scratch.suffix + ".npz"))
    written.replace(path)
    return path


def load_optimizer(optimizer, path):
    """Pour a saved state back into a freshly constructed optimizer.

    ``optimizer`` is built by the caller with the cell and settings the run
    uses, which is what supplies everything :data:`_OPTIMIZER_DERIVED` names;
    this then restores the history on top of it. Returns the same object.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as handle:
        meta = json.loads(bytes(handle["__meta__"]).decode("utf-8"))
        if meta.get("format") != FORMAT_VERSION:
            raise ValueError(
                f"{path} is checkpoint format {meta.get('format')}, and this is "
                f"version {FORMAT_VERSION}"
            )
        for key in handle.files:
            if key.startswith("array_"):
                setattr(optimizer, key[len("array_"):], np.asarray(handle[key]))
    for name, value in meta.items():
        if name != "format":
            setattr(optimizer, name, value)
    return optimizer


def unhandled_optimizer_fields(optimizer) -> set:
    """Attributes neither stored nor declared derived -- empty, and tested."""
    stored = set()
    for name, value in vars(optimizer).items():
        if name in _OPTIMIZER_DERIVED:
            continue
        if isinstance(value, np.ndarray) or value is None or isinstance(
            value, (bool, int, float, str)
        ):
            stored.add(name)
    return set(vars(optimizer)) - stored - set(_OPTIMIZER_DERIVED)
