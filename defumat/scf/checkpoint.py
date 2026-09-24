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

**One thing is refused by name** rather than half-saved, because losing it is
silent: a **magnetic field the run drove away from the input's** -- the
fixed-spin-moment schemes, Elk's ``fsmtype``, where the field *is* the
controller's state and is replaced after every iteration -- cannot be rebuilt
from the input file, and reloading without it applies a rigid Zeeman shift that
a later invariant still returns an integer for (``PLAN.md`` P56 is the record of
that bug found the hard way).

**A Hubbard setup used to be the second, and it was the same mistake as the
blanket field refusal one paragraph down.** "``ns`` without its setup is an
array of numbers about nothing" is true of the *file* and the file is not what
a resume reads it against: a state is loaded against a ``system``, and
``run_scf`` rebuilds the manifold from the ``HUBBARD`` card through
``build_hubbard_setup`` before it looks at ``ns`` at all. Nothing in the loop
can make the two disagree -- ``Calculation.hubbard`` is assigned in exactly one
place (``driver.py``'s ``__init__``), the loop mixes ``ns`` and never the setup,
and the only attribute written on a ``HubbardSetup`` anywhere is
``constraints``, inside ``build_hubbard_setup`` before it returns. ``ns_adj``
does not reach a resume either: it is gated on ``iteration == 1``
(``PW/src/init_ns.f90``'s ``IF (first .AND. starting_pot == 'atomic')``) and a
resume re-enters at ``resumed_at + 1``. The mid-SCF path was always the
evidence -- ``_InProgressState`` has never carried a setup, so every DFT+U run
with ``checkpoint_dir`` has been writing and reloading one of these correctly.
So the setup is :data:`_FROM_CALLER` now, beside ``system``, and what a load
gives back without a ``calculation`` is exactly what a mid-SCF checkpoint has
always given back (``OPEN.md`` Part VII item 3).

**A field is not refused for being a field**, and the difference is worth
stating because the blanket version of this refusal made checkpointing useless
for exactly the runs it exists for -- long, magnetic, and unable to restart. An
applied ``LOCAL_MAGNETIC_FIELDS`` card, and a penalty constraint of any flavour,
leave :class:`~defumat.scf.fields.MagneticField` exactly as the input built it:
the resume rebuilds the calculator from ``scf.in`` and gets the same object
back. Elk's ``reducebf`` does not change it either -- it multiplies a *scalar*,
``field_scale``, which is saved here and restored by the driver on resume. So
what has to be refused is the feedback set (:data:`~defumat.scf.fields.FEEDBACK`)
and nothing wider.

**The scale and the refusal are one mechanism, not two.** ``field_scale``
riding in :data:`_SCALARS` while the loop reset it to 1.0 on resume was a
``reducebf`` run silently coming back at full field, which is the same silent
loss this refusal exists to prevent, taken one level up.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from defumat.scf.fields import FEEDBACK

__all__ = ["save_state", "load_state", "state_fingerprint",
           "save_mixer", "load_mixer"]

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
    # ``ethr`` is loop state rather than a report: ``save_in_electrons.f90``
    # writes ``iter, dr2, ethr`` because QE's schedule is indexed on the
    # iteration number, and a resume that re-enters at 1 without it takes the
    # reset and converges on a different schedule than the one it left.
    "homo", "lumo", "accuracy", "ethr", "nspin", "magnetization",
    "absolute_magnetization", "fermi_energy_up", "fermi_energy_down",
    "magnetization_vector", "field_energy", "constraint_energy", "field_scale",
    "nspin_mag", "meta_c",
)

#: Handled specially: a tuple of arrays, one per ultrasoft/PAW species.
_TUPLES = ("becsum",)

#: Not saved, and refused rather than dropped. See the module docstring.
#: ``magnetic_field`` is here because the coverage check
#: (``tests/unit/test_checkpoint.py``) has to see every field of ``SCFResult``
#: accounted for -- but it is *conditionally* refused, by :func:`_refusal`,
#: which is the only entry left in this tuple.
_REFUSED = ("magnetic_field",)

#: Saved by nothing and dropped on purpose: these are what the run *reported*,
#: not what it converged to. A checkpoint is a state to continue from -- the
#: stress is recomputed from the state in one strain gradient, the solver record
#: and the iteration history describe a run that is over. A reloaded result is
#: therefore a state and not a report, which :func:`load_state` says in its
#: docstring rather than leaving to be noticed.
#:
#: The site charges and moments are reports of the same kind and are dropped for
#: a second reason as well: they are a contraction of the *density* against the
#: integration spheres, costing about 2 ms, and the spheres are a function of the
#: geometry. Carrying them would let a pair computed for one geometry ride along
#: beside a density reloaded into another -- the ``at_cell`` defect one layer up.
#: :meth:`Calculation.site_moments` recomputes them from the loaded density.
#: ``constraint_residual`` is dropped for the same pair of reasons as the site
#: moments: it is a report rather than state, and it is a contraction of the
#: *density* against the constraint's target, so
#: ``MagneticField.cell_residual`` recomputes it from the loaded density. The
#: target itself rides on ``magnetic_field``, which is carried.
_DROPPED = ("stress", "solver", "history", "site_charges", "site_moments",
             "site_residuals", "constraint_residual")

#: Reconstructed from what the caller supplies on load. ``system`` comes back
#: directly; ``hubbard_setup`` is ``Calculation.hubbard``, which
#: :func:`load_state` takes off a ``calculation`` when one is given and leaves
#: ``None`` when it is not -- ``build_hubbard_setup`` needs the datasets as well
#: as the card, and a bare system does not carry them.
_FROM_CALLER = ("system", "hubbard_setup")

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


def _refusal(result) -> str | None:
    """Why this state cannot be written, or ``None`` if it can.

    One place rather than two: the mid-SCF checkpoint and the converged one go
    through the same test, which is the half that was missing. The in-progress
    state used to carry no field at all, so a fixed-spin-moment run wrote a
    checkpoint every cadence with its driven field silently dropped, while a
    plain applied field -- which a resume rebuilds perfectly from the input --
    was refused at the end of the run. Both are the same question and it is
    asked here.
    """
    field = getattr(result, "magnetic_field", None)
    if field is not None and getattr(field, "constraint", "none") in FEEDBACK:
        return (
            f"this run's magnetic field is driven by the "
            f"{field.constraint!r} fixed-spin-moment scheme, so the field is "
            "the controller's state rather than the input's: it is replaced "
            "after every iteration and no input file determines what it "
            "reached. A checkpoint without it resumes at the input field and "
            "converges somewhere else without saying so, which is why saving "
            "is refused rather than lossy. An applied field, and a penalty "
            "constraint, are saved normally -- they are the input's to the end"
        )
    return None


def save_state(result, path) -> Path:
    """Write a converged :class:`SCFResult` to ``path`` as a ``.npz``.

    The wavefunctions dominate the file -- ``nspin nk nbnd npwx`` complex, which
    is tens of gigabytes on a large cell -- so this is a scratch-directory
    operation, not something to do every iteration without meaning to.
    """
    path = Path(path)
    refused = _refusal(result)
    if refused is not None:
        raise NotImplementedError(refused)

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

    ``hubbard_setup`` is the one field rebuilt rather than read: it is
    ``Calculation.hubbard``, so it comes back when a ``calculation`` is given
    and is ``None`` otherwise. A DFT+U state loaded without one carries ``ns``
    and nothing that says which atom each slot belongs to, which is what
    ``SCFResult.hubbard_occupations`` needs; the *resume* does not care, since
    ``run_scf`` rebuilds the manifold from the ``HUBBARD`` card either way.
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
        # Rebuilt from the caller, like ``system``. ``build_hubbard_setup``
        # resolves the ``HUBBARD`` card against the structure *and* the
        # datasets, so a bare system cannot do it and the field stays ``None``
        # -- which is what ``_InProgressState`` has always written, and what
        # every mid-SCF DFT+U resume has always reloaded.
        hubbard_setup=None if calculation is None else calculation.hubbard,
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


#: Rebuilt by ``get_mixer`` from ``mixing_mode``/``mixing_beta``, or installed by
#: the driver because building it needs the G-vectors the mixer does not have.
#: Everything else on a mixer is evolving state and is written.
_MIXER_DERIVED = frozenset({
    # ``beta_max`` is here for the same reason ``beta`` is: it is a setting
    # ``get_mixer`` rebuilds from the input, and storing it would mean a resume
    # silently ignored a caller who changed it. What the adaptive mixer *does*
    # carry across a resume is its evolved per-component state (``_betas``,
    # ``_previous``), which is stored like any other array.
    "beta", "beta_max", "history", "condition_limit", "precondition", "metric",
})


def save_mixer(mixer, path) -> Path:
    """Write a :class:`~defumat.scf.mixing.Mixer`'s evolving history.

    **This is the half of a restart that is easy to leave out and expensive to
    get wrong.** A resume that restores only the density hands Anderson an empty
    history, so the first iterations back are plain mixing and the run pays back
    the iterations the checkpoint saved -- the same trap P67 solved on the BFGS
    side, where the assertion is "2 + 4 steps, not 2 + 6".

    ``_densities`` and ``_residuals`` are lists of equal-length one-dimensional
    arrays, so they stack; a :class:`~defumat.scf.mixing.LinearMixer` has no
    state at all and writes a file with nothing in it but the format, which is
    correct rather than a special case. What is *not* written is
    :data:`_MIXER_DERIVED` -- the settings, which ``get_mixer`` rebuilds, and
    the preconditioner, which the driver installs because it needs the
    G-vectors.
    """
    path = Path(path)
    payload, meta = {}, {"format": FORMAT_VERSION}
    for name, value in vars(mixer).items():
        if name in _MIXER_DERIVED:
            continue
        if isinstance(value, np.ndarray):
            payload[f"array_{name}"] = value
        elif isinstance(value, list):
            # The history. Stacked with its length in the metadata, because an
            # empty list and a list of empty arrays are different states and
            # ``np.stack`` cannot tell them apart on the way back.
            meta[f"len_{name}"] = len(value)
            for index, entry in enumerate(value):
                payload[f"list_{name}_{index}"] = np.asarray(entry)
        elif value is None or isinstance(value, (bool, int, float, str)):
            meta[name] = value
        else:
            raise NotImplementedError(
                f"the mixer holds {name!r} of type {type(value).__name__}, "
                "which this checkpoint cannot represent. Add it to "
                "_MIXER_DERIVED if ``get_mixer`` rebuilds it, or extend this "
                "function -- dropping it would restart the history silently"
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


def load_mixer(mixer, path):
    """Pour a saved history back into a freshly built mixer. Returns it."""
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
                setattr(mixer, key[len("array_"):], np.asarray(handle[key]))
        for key, count in meta.items():
            if not key.startswith("len_"):
                continue
            name = key[len("len_"):]
            setattr(mixer, name, [
                np.asarray(handle[f"list_{name}_{index}"])
                for index in range(int(count))
            ])
    for name, value in meta.items():
        if name != "format" and not name.startswith("len_"):
            setattr(mixer, name, value)
    return mixer


def unhandled_mixer_fields(mixer) -> set:
    """Attributes neither stored nor declared derived -- empty, and tested."""
    stored = set()
    for name, value in vars(mixer).items():
        if name in _MIXER_DERIVED:
            continue
        if isinstance(value, (np.ndarray, list)) or value is None or isinstance(
            value, (bool, int, float, str)
        ):
            stored.add(name)
    return set(vars(mixer)) - stored - set(_MIXER_DERIVED)


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
