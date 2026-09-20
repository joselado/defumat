"""One object with bound methods, so that a script is what it computes.

The functional API underneath -- ``run_scf(system, pseudos, ...)``,
``dielectric_tensor(calculation, wavefunctions, eigenvalues, density, becsum)``
-- mirrors ``pw.x``'s own variable structure, and it stays. What it costs the
reader is that the *state* linking one calculation to the next is threaded by
hand:

.. code-block:: python

    pwin = read_pw_input("scf.in")
    system = build_system(pwin)
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file)
                    for s in system.structure.species)
    result = run_scf(system, pseudos)
    calculation = Calculation(system, pseudos)
    bands = run_bands(system, pseudos, result.density, kpoints=path,
                      becsum=result.becsum)
    eps = dielectric_tensor(calculation, result.wavefunctions,
                            result.eigenvalues, result.density, result.becsum)

against

.. code-block:: python

    calc = Calculator.from_file("scf.in")
    bands = calc.get_bands(kpoints=path)
    eps = calc.get_dielectric_tensor()

**Brevity is the smaller half of the reason.** The state being threaded is not
one array but four -- ``density``, ``becsum``, ``ns`` and ``tau`` -- and which
of them are *load-bearing* depends on the run: ``ns`` under a Hubbard ``U``,
``tau`` under a meta-GGA, ``becsum`` for a PAW dataset. None can be rebuilt
from the density, so "passed the density, forgot the ``becsum``" is a thing a
hand-threaded call can be.

It is worth being exact about what that costs, because the package is already
careful here: ``fixed_density_bands`` and ``dielectric_tensor`` **refuse** a
PAW run whose ``becsum`` is missing, and ``run_nscf`` refuses a Hubbard run
without ``ns`` and a meta-GGA without ``tau``. So the failure is a stopped run
and a puzzle rather than a wrong number -- the refusals hold, as they are meant
to. What the facade removes is the puzzle: every response entry point takes the
same ``(calculation, wavefunctions, eigenvalues, density, becsum)`` prefix,
which is precisely the tuple :class:`~defumat.scf.driver.SCFResult` already
carries, so unpacking the cached result is something no caller should have to
get right twice.

Three rules keep it honest, and they are the design:

* **Nothing mutates.** :meth:`Calculator.with_cell`, :meth:`with_positions` and
  :meth:`with_spin` return a *new* calculator with an empty cache, seeded
  through P23's continuation where that applies. pyqula's ``h.add_swave(...)``
  is safe there because nothing expensive is cached on ``h``; here a mutated
  cell under a cached ``SCFResult`` would hand back the previous geometry's
  dielectric tensor.
* **The implicit SCF says so.** A method that needs a ground state and finds no
  cache runs one and prints a line to stderr first. :meth:`get_scf` is the
  explicit path and is the same code, so there is one behaviour rather than
  two, and :attr:`scf_result` looks at the cache without triggering anything.
* **Refusals pass through untouched.** Nothing here catches an exception. The
  package's contract is that a run which starts is a run whose physics is
  there, and the refusals are what make that promise legible -- a facade that
  swallowed them would be a facade that broke it.

The state lives here rather than on :class:`~defumat.system.builder.System`
for two reasons, the second decisive. ``System`` is an ``equinox.Module``
crossing ``jit``/``grad`` boundaries, so a ``pseudos`` field would change the
pytree every compiled path sees and a cached result cannot live on a frozen
module at all. And ``System`` does not *have* the pseudopotentials: a
``system.get_bands()`` would need them as an argument, which is the API this
module exists to shorten. The unit that can compute is ``system + pseudos``,
which is exactly what :class:`~defumat.scf.driver.Calculation` already takes.
"""

from __future__ import annotations

import dataclasses
import inspect
import sys
import warnings
from pathlib import Path

import numpy as np

from defumat.pseudo.upf import Pseudopotential, read_upf
from defumat.scf.driver import (Calculation, SCF_CHECKPOINT, SCFResult,
                                run_scf)
from defumat.system.builder import System, build_system
from defumat.system.kpoints import for_spin

__all__ = ["Calculator", "SHARED_OPTIONS", "SCF_ONLY_OPTIONS",
           "electrons_defaults"]


#: Options a :class:`Calculator` accepts once, at construction, and forwards to
#: every method whose entry point names them. They are the knobs that describe
#: *this calculator* rather than one quantity computed with it: how many bands,
#: how tightly to converge, how many k-points in flight, which eigensolver.
#:
#: The filtering is by *named parameter*, never by a ``**kwargs`` catch-all --
#: :func:`~defumat.response.electrostriction.electrostriction` has one, and it
#: forwards to the Sternheimer solvers, which have no ``nbnd``.
SHARED_OPTIONS = frozenset({
    "nbnd",
    "conv_thr",
    "k_batch",
    # Where the wavefunction store lives between the points that read it, the
    # placement counterpart of ``k_batch``'s flight dial. Only ``run_scf`` names
    # it, so the forwarding reaches that and nothing else -- which is the whole
    # reason the filtering is by named parameter.
    "wfc_store",
    "diagonalization",
    "david",
    "diago_full_acc",
    "mixing_mode",
    "mixing_beta",
    "mixing_ndim",
    "mixing_fixed_ns",
    "max_iterations",
    "scf_solver",
    "scf_solver_options",
    "checkpoint_dir",
    "checkpoint_every",
    "mixing_from",
    "verbose",
})

#: The options a :class:`Calculator` accepts at construction and adopts from a
#: call, but which are **not** :data:`SHARED_OPTIONS` and are never forwarded to
#: an entry point: they configure the :class:`~defumat.scf.driver.Calculation`
#: itself, which :attr:`Calculator.calculation` builds from them.
#:
#: There is one, and the reason it is not simply shared is the same collision
#: :data:`SCF_ONLY_OPTIONS` documents, one step worse. ``projectors`` is this
#: code's **memory** dial -- ``'default'`` keeps the projectors resident,
#: ``'rebuild'`` recomputes them per k-point -- and
#: :func:`~defumat.workflows.pdos.run_pdos` has a parameter of the same name
#: meaning the **Hubbard projector scheme**, ``'ortho-atomic'`` against
#: ``'atomic'``. Sharing the name would forward a memory setting into a physics
#: one, so it is accepted, adopted and consumed here, and stops here.
SETUP_ONLY_OPTIONS = frozenset({"projectors"})

#: The subset of :data:`SHARED_OPTIONS` that describes the **SCF loop** and
#: nothing else, and is therefore *not* forwarded past it.
#:
#: The name is the whole problem: ``max_iterations`` is the SCF's iteration
#: count in :func:`~defumat.scf.driver.run_scf`, the **self-consistent
#: response's** in :func:`~defumat.response.efield.dielectric_tensor`
#: (whose default is that module's own ``MAX_ITERATIONS = 40``; what comes from
#: :mod:`defumat.response.mixing` is the ``mixing_mode`` two lines below it in
#: the same signature, which is how the misattribution read as checked) and the
#: **Dyson fixed point's** in
#: :func:`~defumat.workflows.run_absorption` (``tddft/dyson.py``'s 500). Three
#: loops, one word. ``mixing_mode``/``mixing_beta`` collide the same way between
#: the density mixer and the response mixer.
#:
#: So this was wrong in both directions at once. Nine response methods dropped
#: the shared options entirely -- a calculator built with ``verbose=True`` got a
#: silent dielectric solve, against a docstring promising otherwise -- and the
#: one method that did forward, ``get_absorption``, was silently capping a Dyson
#: iteration with a number chosen for the SCF. Forwarding *everything* would
#: have spread the second bug to the other nine rather than fixing the first.
#:
#: What stays shared is what means the same thing wherever it is named: how many
#: bands, how tightly to diagonalise, how many k-points in flight, which
#: eigensolver, and whether to print. An SCF-only option is still reachable per
#: call -- ``get_absorption(max_iterations=...)`` is unambiguous *at the call
#: site*, which is exactly what a constructor default is not.
SCF_ONLY_OPTIONS = frozenset({
    # ``diago_full_acc`` is SCF-only because the thing it switches -- the
    # looser threshold on an empty band -- exists only where there are
    # occupations to call a band empty. A band-structure or response run
    # diagonalises every band at ``ethr`` regardless, so forwarding the flag
    # past ``run_scf`` would name an option that does nothing.
    "diago_full_acc",
    "mixing_mode",
    "mixing_beta",
    "mixing_ndim",
    "mixing_fixed_ns",
    "max_iterations",
    "scf_solver",
    "scf_solver_options",
    # The SCF's own iteration state reaches disk on a cadence and comes back
    # through it. A response's inner loop is not restartable and has no such
    # directory, so forwarding these past ``run_scf`` would name an option that
    # does nothing -- which is the failure the comment above this set describes.
    "checkpoint_dir",
    "checkpoint_every",
    "mixing_from",
})


#: The shared options that belong to the **second** system a method runs on,
#: and so must not cross into it from this calculator's defaults.
#:
#: The force theorem is two calculators: ``self`` is the scalar-relativistic
#: collinear leg and ``spinor`` is the fully-relativistic noncollinear one,
#: built from a different input file and a different set of pseudopotentials.
#: A band count is a property of the system whose bands are being counted, and
#: the two systems disagree about it, because a spinor band holds one electron
#: where a collinear one holds two. Forwarding it put the collinear leg's
#: number into the spinor run, where :func:`~defumat.workflows.nscf.run_nscf`'s
#: ``nbnd or system.nbnd or default_nbnd(...)`` lets a stated value win over
#: the spinor input's own, so the second leg ran with roughly half the bands it
#: needs. Withholding it is what lets the spinor input's ``&system nbnd`` reach
#: the run, which is the chain :meth:`Calculator.get_anisotropy` describes, and
#: :func:`_spinor_leg` carries the spinor calculator's own value across in its
#: place so that setting it there is not silently lost either.
_SPINOR_LEG_OPTIONS = frozenset({"nbnd"})


#: The shared options that are a **bound on an error**, where the two values in
#: play are ordered and one of them is strictly better.
#:
#: There is one, and it is what keeps :func:`_callee_chose` from being wrong in
#: the other direction. A threshold is a bound, so asking for a smaller one is
#: always honourable and asking for a larger one silently defeats a choice the
#: workflow made: a calculator default may therefore **tighten** a value the
#: callee chose and never loosen it. Without the distinction an input stating
#: ``conv_thr = 1e-10`` would have been withheld from
#: :func:`~defumat.ultracell.driver.run_ultracell`, whose own 1e-8 is a second
#: driver's default for the *same* quantity rather than a tightening -- that
#: module says so outright, "``conv_thr`` means the same thing in an ultracell
#: run as in every other run in this package" -- and the run would have come
#: back looser than the file asked for. Nothing else here is ordered:
#: ``max_iterations`` of 40 against 100 and ``k_batch`` of 1 against the whole
#: axis are not better or worse, they are what the workflow wants.
_TOLERANCE_OPTIONS = frozenset({"conv_thr"})


#: :func:`~defumat.scf.driver.run_scf`'s own defaults, the reference an entry
#: point's are read against in :func:`_callee_chose`.
_RUN_SCF_DEFAULTS = {name: parameter.default for name, parameter
                     in inspect.signature(run_scf).parameters.items()}


def _callee_chose(parameters, name) -> bool:
    """Has this entry point stated a value of its own for ``name``?

    A shared option means the same thing everywhere it is named -- that is what
    put it in :data:`SHARED_OPTIONS` -- but meaning the same thing is not
    wanting the same value. **Twenty** entry points declare a ``conv_thr``
    between 1e-8 and 1e-12 and say in their own docstrings why: an anisotropy
    is a difference of band-energy sums in the fifth decimal of an eV, an
    effective mass is a second difference of eigenvalues, a Berry phase is a
    product of overlaps around a closed loop. A calculator built by
    :meth:`Calculator.from_file` carries the input's ``&electrons conv_thr``,
    and ``pw.x``'s own 1e-6 is the usual value there, so the calculator's
    *default* was silently replacing every one of those choices with a
    threshold four orders looser -- on both legs of a force theorem at once,
    which is why the workflow's own ``drifts`` could not see it. Six more state
    a ``max_iterations`` and one a ``k_batch``, twenty-seven pairs in all.

    The rule is therefore that a calculator default yields to a value the
    callee chose, and ``run_scf``'s signature is what "chose" is measured
    against: an entry point repeating 1e-6 is repeating the SCF's default and
    has no opinion, one that writes 1e-10 does. ``None`` is not a stated value
    either -- :func:`~defumat.workflows.spiral.relax_spiral_q` writes it for
    ``max_iterations`` precisely so the SCF's own number comes through -- and
    neither is a parameter with no default at all, which the calculator must
    supply or the call fails.

    Nor is it symmetric, and :data:`_TOLERANCE_OPTIONS` is where that is
    written down: a calculator ``conv_thr`` *tighter* than the callee's still
    goes through, because a threshold is a bound on an error and a smaller one
    is never the wrong thing to hand a workflow. What is withheld is a looser
    one.

    What this is not is a refusal either. Naming the option at the call site
    still reaches it, ``get_anisotropy(soc, conv_thr=1e-6)`` included, which is
    the difference between a default and a refusal and is the same line
    :meth:`_shared_scf_options` draws with ``withheld``.

    It does **not** replace that ``withheld``, and the two are not
    interchangeable. ``withheld`` guards the ``**kwargs`` route, where there is
    no signature to read: :func:`~defumat.workflows.anisotropy
    .run_relaxed_anisotropy` has no ``conv_thr`` parameter at all and the
    tightening happens inside it, so nothing here can see the choice and
    :meth:`get_relaxed_anisotropy` must keep naming it. This guards the
    named-parameter route, which is where the twenty-seven are. The second
    guard written by hand, :meth:`get_magnetoelectric_tensor`'s
    ``exclude={"conv_thr"}``, is vestigial for a better reason: that signature
    now spells the two thresholds apart as ``scf_conv_thr`` and
    ``polarization_conv_thr``, which is the repair rather than the guard -- and
    its ``max_iterations = 120`` is one of the twenty-seven and was reached.
    """
    default = parameters[name].default
    if default is inspect.Parameter.empty or default is None:
        return False
    return name in _RUN_SCF_DEFAULTS and default != _RUN_SCF_DEFAULTS[name]


def _yields_to_callee(parameters, name, value) -> bool:
    """Should this calculator default stand aside for the entry point's own?

    :func:`_callee_chose` says whether there is a choice to stand aside for;
    this says whether standing aside is the right thing to do with *this*
    value. For a :data:`_TOLERANCE_OPTIONS` member it is not, when the value is
    the stricter of the two: a bound on an error can always be made smaller.
    """
    if not _callee_chose(parameters, name):
        return False
    if name in _TOLERANCE_OPTIONS and isinstance(value, (int, float)):
        return not value < parameters[name].default
    return True


#: Parameter name -> the :class:`~defumat.scf.driver.SCFResult` attribute that
#: fills it. All five are properties of the converged *state* that cannot be
#: rebuilt from the density, so an entry point that names one is supplied it
#: rather than left to refuse. ``field``/``field_scale`` are the pair the SCF
#: ended with, which is not the input's whenever ``reducebf`` or the
#: fixed-spin-moment scheme was in use -- hence the mapping rather than a plain
#: attribute lookup.
_STATE_ARGUMENTS = {
    "ns": "ns",
    "tau": "tau",
    "becsum": "becsum",
    "field": "magnetic_field",
    "field_scale": "field_scale",
}


#: ``&electrons`` variable -> the :data:`SHARED_OPTIONS` name it fills.
#:
#: ``pw.x`` states how to converge a run in the input file; nothing here read
#: that namelist, though ``system/builder.py`` has always read ``&control``'s
#: ``etot_conv_thr``, ``forc_conv_thr`` and ``nstep``. The asymmetry was worth
#: real boilerplate: 28 of the 29 notebooks passed a ``conv_thr`` by hand, and
#: eleven defined a local ``load()`` helper, four of which existed only to
#: re-parse the input and hand these values straight back (P49).
#:
#: ``electron_maxstep`` is renamed on the way in because ``max_iterations``
#: means three different loops here -- see :data:`SCF_ONLY_OPTIONS` -- and the
#: input file's number is unambiguously the SCF's.
_ELECTRONS_OPTIONS = {
    "conv_thr": "conv_thr",
    "mixing_beta": "mixing_beta",
    "mixing_mode": "mixing_mode",
    # ``mixing_ndim``: how many previous iterations the extrapolation spans.
    # Raising it is the first thing a pw.x user does to a magnetic cell that
    # will not converge, and it was parsed and dropped on the floor until P78.
    # 8 is pw.x's own default and this code's, so an input that does not set it
    # behaves exactly as it did.
    "mixing_ndim": "mixing_ndim",
    "mixing_fixed_ns": "mixing_fixed_ns",
    "electron_maxstep": "max_iterations",
    # ``diago_david_ndim``: adopted where ``diagonalization`` beside it is not,
    # and the difference is that this one cannot fail. It is an integer that
    # always means the same thing to the one solver here, where a solver *name*
    # this package does not have would turn a valid pw.x input into a
    # ValueError. It is a setup option all the same -- it belongs to the
    # Calculation rather than to a run -- so it is in SETUP_OPTIONS too.
    "diago_david_ndim": "david",
    # ``diago_full_acc``: adopted for the same reason ``diago_david_ndim`` is.
    # It is a logical that means exactly what it means in ``pw.x`` -- hold the
    # empty states to ``ethr`` as well -- and it changes no shape and no
    # ``Calculation``, so it is a run default rather than a setup option.
    "diago_full_acc": "diago_full_acc",
}

#: Read but deliberately **not** adopted: ``diagonalization``.
#:
#: It is a :data:`Calculator.SETUP_OPTIONS` member rather than a run default, so
#: it decides which ``Calculation`` exists; and this package offers one
#: eigensolver, so an input saying ``diagonalization = 'cg'`` -- valid ``pw.x``
#: input, and common -- would stop being a run that works and start being a
#: ``ValueError`` from ``get_eigensolver``. Silently mapping it onto Davidson is
#: the other half of that trade and is worse: it is exactly the substitution
#: this package refuses elsewhere. So the variable is left alone, and a caller
#: who wants a solver names it at construction.
_ELECTRONS_NOT_ADOPTED = ("diagonalization",)


def electrons_defaults(pwin) -> dict:
    """The ``&electrons`` namelist as :data:`SHARED_OPTIONS` keyword arguments.

    Absent variables are absent from the result rather than given a default, so
    that whatever :func:`~defumat.scf.driver.run_scf` already defaults to keeps
    deciding. See :data:`_ELECTRONS_OPTIONS` for what is mapped and
    :data:`_ELECTRONS_NOT_ADOPTED` for the one that is not.
    """
    adopted = {}
    for name, option in _ELECTRONS_OPTIONS.items():
        value = pwin.get("electrons", name)
        if value is None:
            continue
        if option in ("conv_thr", "mixing_beta"):
            value = float(value)
        elif option in ("mixing_fixed_ns", "max_iterations", "david",
                        "mixing_ndim"):
            value = int(value)
        elif option == "diago_full_acc":
            # A Fortran logical, which ``_convert`` has already turned into a
            # Python ``bool``. Without this arm it would fall through to the
            # string branch below and arrive as ``"True"`` -- truthy either
            # way, and the wrong type.
            value = bool(value)
        else:
            value = str(value).strip().strip("'\"")
        adopted[option] = value
    return adopted


class Calculator:
    """A system, its pseudopotentials, and every calculation they support.

    Args:
        system: the :class:`~defumat.system.builder.System` to compute on.
        pseudos: its pseudopotentials, in species order. ``None`` -- the usual
            case -- loads them from the names the input file gave, resolved
            against ``pseudo_dir``.
        pseudo_dir: where to find those files. Defaults to the directory the
            input file came from, matching what the CLI already does.
        basis: a prebuilt :class:`~defumat.basis.builder.Basis`, for the rare
            caller that has one. Normally left alone; the basis is built on
            first use, so constructing a ``Calculator`` compiles nothing.
        announce: print a line to stderr when a method runs an SCF that the
            caller did not ask for by name.
        **defaults: any of :data:`SHARED_OPTIONS`, applied to every method that
            names them -- **except** :data:`SCF_ONLY_OPTIONS`, which stop at the
            SCF, because past it the same spelling means a different loop
            (``max_iterations`` is the SCF's, the self-consistent response's and
            the Dyson fixed point's, in three different callees). A per-call
            keyword overrides both for that call, and is the way to reach an
            SCF-only option elsewhere -- at the call site the name is
            unambiguous. The exception to *that* is :data:`SETUP_OPTIONS`
            (``diagonalization``, ``k_batch``, ``david``), which do not
            describe a *run*
            but the ``Calculation`` every run goes through, so giving one per
            call rebuilds it and it stays.

    A minimal script::

        from defumat import Calculator

        calc = Calculator.from_file("scf.in")
        print(calc.get_scf().total_energy)
        calc.get_bands(kpoints=path).plot()
    """

    def __init__(
        self,
        system: System,
        pseudos=None,
        *,
        pseudo_dir=None,
        basis=None,
        announce: bool = True,
        **defaults,
    ):
        unknown = set(defaults) - SHARED_OPTIONS - SETUP_ONLY_OPTIONS
        if unknown:
            raise TypeError(
                f"unknown calculator option(s) {sorted(unknown)}. A keyword given "
                "here applies to every method that takes it, so only the run-wide "
                f"ones are accepted: {sorted(SHARED_OPTIONS | SETUP_ONLY_OPTIONS)}. "
                "Anything else belongs on the method it configures"
            )

        self.system = system
        self.pseudos = _resolve_pseudos(system, pseudos, pseudo_dir)
        self.defaults = dict(defaults)
        self.announce = bool(announce)

        self._basis = basis
        self._calculation: Calculation | None = None
        self._scf: SCFResult | None = None
        self._scf_options: dict | None = None
        self._relax = None
        self._relax_options = None
        self._relax_variable_cell = False
        #: The last ultracell, and the options that filled it. One slot, for
        #: :meth:`get_scf`'s reason: what it holds is the frozen states.
        self._ultracell = None
        self._ultracell_options: dict | None = None
        self._strain_response = None
        self._strain_response_options = None
        #: A converged state from another calculator, handed to the first SCF
        #: as ``starting_from``. Not a cache -- a starting point (P23).
        self._seed = None
        #: How :attr:`_seed`'s magnetization crosses, ``run_scf``'s own
        #: ``magnetization``. It belongs to the seed rather than to the
        #: calculator, which is why it is not a shared option: a default set on
        #: the constructor would reach runs that have no seed for it to act on,
        #: and ``run_scf`` refuses that combination by name.
        self._seed_magnetization = "auto"
        #: The NSCF states the last :meth:`get_dos` or :meth:`get_pdos` ran on.
        #: Those entry points return a pair; the second half is kept here so
        #: that the method can return the quantity that was asked for.
        self.dos_states = None
        self.pdos_states = None

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def estimate(self, **options):
        """What this run will cost, without allocating anything on the device.

        A :class:`~defumat.sizing.SizeEstimate`: the exact ``ngm``, ``npwx``,
        ``nbnd`` and ``nkb`` the setup would build, the two FFT grids, and a
        floor on the bytes. Unlike every other method here it does **not** touch
        :attr:`calculation`, which is the whole point -- it answers "will this
        fit" for an input too large to build.

        ``options`` are :func:`~defumat.sizing.estimate_size`'s: ``nbnd``,
        ``k_batch``, ``davidson_basis`` and ``band_batch``.

        **What is not given is taken from this calculator's own defaults**, not
        from the library's, because the question this answers is "will *this
        run* fit" and any other reading makes the answer describe a run that
        does not happen. An input saying ``diago_david_ndim = 2`` is sized at 2,
        and ``k_batch`` is resolved the way the SCF would resolve it -- which is
        QE's one-k-point loop on a CPU, not the whole axis. Sizing the first of
        those at the library default was worth 35 GB on the cell this was
        written for; it is the same mistake as sizing ``K_POINTS gamma`` as the
        request rather than as the substitution, one option along.
        """
        from defumat.batching import resolve_k_batch
        from defumat.sizing import estimate_size

        options.setdefault("davidson_basis", self.defaults.get("david"))
        options.setdefault("nbnd", self.defaults.get("nbnd"))
        # The band dial has no input-file variable, so a caller's value or the
        # environment's is the whole of it -- but it still has to be resolved
        # here rather than inside, for the same reason ``k_batch`` is: the
        # answer describes *this run*.
        if options.get("band_batch") is None:
            options["band_batch"] = "default"
        # The same argument as ``k_batch`` below: this calculator's own answer,
        # not the library's, because the question is "will *this run* fit" and
        # the dial changes the largest line in the table.
        options.setdefault("projectors",
                           self.defaults.get("projectors", "default"))
        if options.get("k_batch") is None:
            options["k_batch"] = resolve_k_batch(
                self.defaults.get("k_batch", "default")
            )
        return estimate_size(self.system, self.pseudos, **options)

    @classmethod
    def from_file(cls, path, pseudo_dir=None, **defaults) -> "Calculator":
        """Read a ``pw.x`` input file and load the pseudopotentials it names.

        The one line that replaces four. ``pseudo_dir`` defaults to the input
        file's own directory.

        The input's ``&electrons`` namelist is adopted as this calculator's
        defaults -- ``conv_thr``, ``mixing_beta``, ``mixing_mode``,
        ``mixing_fixed_ns`` and ``electron_maxstep`` -- so a ``pw.x`` input that
        states how to converge itself converges the same way here. A keyword
        argument given to this call still wins over the file. See
        :func:`electrons_defaults`.
        """
        from defumat.io.pwin import read_pw_input

        path = Path(path)
        pwin = read_pw_input(path)
        if pseudo_dir is None:
            # The input's own ``&control pseudo_dir`` first, resolved relative
            # to the input file as ``pw.x`` resolves a relative one, and this
            # code's own default -- the input file's directory -- behind it.
            # An explicit argument still wins over both.
            named = pwin.get("control", "pseudo_dir")
            pseudo_dir = (
                (path.parent / str(named).strip(), path.parent)
                if named else path.parent
            )
        return cls(build_system(pwin),
                   pseudo_dir=pseudo_dir,
                   **{**electrons_defaults(pwin), **defaults})

    @classmethod
    def from_text(cls, text: str, pseudo_dir, **defaults) -> "Calculator":
        """The same, from the text of an input file rather than a path.

        ``pseudo_dir`` is required here: there is no file to take it from.
        ``&electrons`` is adopted exactly as in :meth:`from_file`.
        """
        from defumat.io.pwin import parse_pw_input

        pwin = parse_pw_input(text)
        return cls(build_system(pwin), pseudo_dir=pseudo_dir,
                   **{**electrons_defaults(pwin), **defaults})

    # ------------------------------------------------------------------
    # the fixed setup, built once and on demand
    # ------------------------------------------------------------------

    @property
    def calculation(self) -> Calculation:
        """The :class:`~defumat.scf.driver.Calculation` this calculator uses.

        Built on first use rather than in ``__init__``, so that constructing a
        calculator is free and a refusal (gamma-only storage, a relativistic
        dataset without ``lspinorb``) is raised when a run starts rather than
        when the object is named.
        """
        if self._calculation is None:
            self._calculation = Calculation(
                self.system,
                self.pseudos,
                basis=self._basis,
                diagonalization=self.defaults.get("diagonalization"),
                k_batch=self.defaults.get("k_batch", "default"),
                david=self.defaults.get("david"),
                projectors=self.defaults.get("projectors", "default"),
            )
        return self._calculation

    #: The options that define a :class:`~defumat.scf.driver.Calculation`
    #: rather than one run over it. Given per call, they have to rebuild it.
    SETUP_OPTIONS = ("diagonalization", "k_batch", "david", "projectors")

    def _adopt(self, options) -> None:
        """Take a per-call setup option as this calculator's own.

        Rebuilding is the correct semantics rather than a workaround: these two
        decide *which* ``Calculation`` exists, so a call that changes one is
        asking for a different setup, and everything cached under the old one
        is about to be replaced anyway.
        """
        changed = {name: options[name] for name in self.SETUP_OPTIONS
                   if name in options and options[name] != self.defaults.get(
                       name, "default"
                       if name in ("k_batch", "projectors") else None)}
        if changed:
            self.defaults.update(changed)
            self._calculation = None
            self._strain_response = self._strain_response_options = None

    # ------------------------------------------------------------------
    # the ground state, and the cache in front of it
    # ------------------------------------------------------------------

    @property
    def scf_result(self) -> SCFResult | None:
        """The cached ground state, or ``None`` -- **without** computing one.

        The property to test against. ``calc.get_scf()`` in an ``if`` would run
        the SCF in order to answer the question.
        """
        return self._scf

    @property
    def converged(self) -> bool:
        """Whether a converged ground state is cached. Computes nothing."""
        return self._scf is not None and bool(self._scf.converged)

    @property
    def starting_state(self):
        """A converged state from another calculator, or ``None``.

        **Not a cache and not an answer** -- a starting point. A calculator
        derived by :meth:`with_positions` or :meth:`with_spin` inherits the
        parent's converged state here rather than in :attr:`scf_result`, and
        the first SCF is handed it as ``starting_from`` (P23): the density of
        the previous geometry is a far better guess than the atomic one, and a
        spin promotion whose magnetization only has to be rotated converges in
        one iteration instead of twenty-five.
        """
        return self._seed

    def get_scf(self, **options) -> SCFResult:
        """Run the self-consistent field loop, or return the cached result.

        Called again with the same options this returns the cache; called with
        different ones it reruns and *replaces* the cache. The cache is a single
        slot rather than a dictionary keyed by option sets, because what it
        holds is the wavefunctions -- the largest arrays in the process -- and a
        keyed cache would quietly hold several sets of them.

        Every SCF keyword lives here or on the constructor, never on
        ``get_bands`` and its relatives: ``conv_thr`` means the SCF's own
        threshold in this method and "the accuracy the density was converged
        to" in those, and a shared passthrough would silently conflate them.

        **The old state is dropped before the new run starts, not after.**
        Rebinding a name does not release what it pointed at until the
        right-hand side has returned, so ``self._scf = run_scf(...)`` keeps the
        previous run's wavefunctions resident for the whole of the new SCF,
        Davidson peak included -- and a strain response built on them is six
        ``(nk, nocc, npwx)`` blocks more. The price is that a run which raises
        leaves no cache behind, which is the honest state anyway: what was
        cached is no longer what this calculator is set up for.

        **A checkpoint in ``checkpoint_dir`` beats the inherited seed**, which
        is the same rule :func:`~defumat.scf.driver.run_scf` states for a seed
        the caller passed and for the same reason: the checkpoint is strictly
        later state, and the recovery a checkpoint advertises is "resubmit the
        same command line". A derived calculator inserts its parent's state as
        ``starting_from`` on the caller's behalf, and ``run_scf`` reads its
        directory only when nothing was passed, so a
        ``calc.with_spin(4).get_scf(checkpoint_dir=...)`` killed at its wall
        clock and resubmitted used to start from the seed every time and never
        read the checkpoint it had been writing. The seed and the
        ``magnetization`` beside it are withheld **together**, since a resume
        refuses a ``magnetization`` argument -- there is nothing left for it to
        decide -- and ``with_moments`` sets one by default.
        """
        # A :data:`SETUP_ONLY_OPTIONS` member is **not part of the key**, and
        # leaving it in made a plain ``get_scf()`` after a
        # ``get_scf(projectors='rebuild')`` miss its own cache and run the
        # whole SCF a second time -- the first call's ``options`` carried it
        # and the second's did not, while ``_defaults_for`` no longer supplies
        # it from ``defaults``. It belongs out rather than back in: it says
        # which ``Calculation`` exists, not which run was made over it, and
        # :meth:`_adopt` below already drops the cache when one changes.
        merged = {name: value for name, value in
                  {**self._defaults_for(run_scf), **options}.items()
                  if name not in SETUP_ONLY_OPTIONS}
        if self._scf is not None and _same_options(merged, self._scf_options):
            return self._scf
        # ``diagonalization`` and ``k_batch`` are *not* arguments of the SCF:
        # they are what a ``Calculation`` is built with, and ``run_scf``
        # documents that it ignores them when handed one. Passing them per call
        # would otherwise be a silent no-op that still counted as a cache miss
        # -- the same run, again, under a different name. They rebuild it.
        self._adopt(options)
        # Before the call rather than after it -- see the docstring. An SCF
        # makes every response built on the previous one stale in any case.
        self._scf = self._strain_response = self._strain_response_options = None
        self._ultracell = self._ultracell_options = None
        self._scf = run_scf(self.system, self.pseudos,
                            calculation=self.calculation, **self._seeded(merged))
        # The key is the options, without the seed -- see ``_seeded``.
        self._scf_options = merged
        return self._scf

    def _seeded(self, merged: dict) -> dict:
        """``merged`` with the inherited seed in it, unless a checkpoint wins.

        Both keys go in or neither does. Dropping only ``starting_from`` would
        let the ``magnetization`` beside it reach a resume, which refuses one
        by name -- so a ``with_moments`` calculator, whose default is
        ``'seed'``, would go from restarting silently to raising.

        **This is why the cache key is the options and not what is passed.** The
        seed is a property of the calculator rather than of the call, so keying
        on it would add nothing -- except that whether it goes in depends on a
        file on disk, and a converged run does not delete its last checkpoint.
        The second ``get_scf`` would then miss its own cache and rerun the whole
        SCF, from a mid-run state at that.
        """
        if self._seed is None or _has_checkpoint(merged.get("checkpoint_dir")):
            return merged
        seeded = dict(merged)
        seeded.setdefault("starting_from", self._seed)
        seeded.setdefault("magnetization", self._seed_magnetization)
        return seeded

    def get_elk_seed(self, directory, renormalise: bool = True, report=None):
        """Elk's converged density on this run's grid, as a starting guess.

        Hand the result to :meth:`get_scf` as ``starting_density`` to continue
        an Elk ground state here:

        .. code-block:: python

            seed = calc.get_elk_seed("elk_run/")
            result = calc.get_scf(starting_density=seed)

        ``directory`` is an Elk *run directory* -- ``STATE.OUT`` beside
        ``GEOMETRY.OUT``, because ``STATE.OUT`` carries neither the cell nor the
        atomic positions.

        **It is a seed, not an answer.** Elk is all-electron and this is a
        pseudopotential code, so the two converged densities are different
        functions wherever there is a core; defumat's own SCF still runs on top.
        Passing an Elk density as a *fixed* density is refused
        (:func:`defumat.io.elk_density.density_on`), as are a spin-polarized Elk
        state, a spin spiral and a DFT+U one.
        """
        from defumat.io.elk import ElkState

        return ElkState.read(directory).density_on(
            self.calculation, renormalise=renormalise, report=report
        )

    def _ground_state(self, quantity: str) -> SCFResult:
        """The cached ground state, running one first if there is none."""
        if self._scf is None:
            if self.announce:
                conv = self.defaults.get("conv_thr", 1.0e-6)
                print(
                    f"[defumat] {quantity}: no ground state cached, running the "
                    f"SCF first (conv_thr = {conv:g}). Call get_scf() to do this "
                    "explicitly.",
                    file=sys.stderr,
                )
            self.get_scf()
        # One implementation, on the result rather than here, so that a caller
        # who takes ``scf.density`` to a functional entry point can make the
        # same refusal: see :meth:`~defumat.scf.driver.SCFResult.require_converged`.
        return self._scf.require_converged(quantity)

    # ------------------------------------------------------------------
    # band structure, densities of states
    # ------------------------------------------------------------------

    def get_bands(self, kpoints=None, **options):
        """Diagonalise on a k-path at the converged density.

        The Fermi level and the HOMO come from the cached ground state unless
        given, so the band plot has its zero without being told.
        """
        from defumat.workflows.bands import run_bands

        result = self._ground_state("a band structure")
        kw = self._call_options(run_bands, result, options)
        kw.setdefault("fermi_energy", result.fermi_energy)
        kw.setdefault("homo", result.homo)
        return run_bands(self.system, self.pseudos, result.density,
                         kpoints=kpoints, **kw)

    def get_nscf(self, kpoints=None, **options):
        """Diagonalise on a k-grid at the converged density, and occupy it."""
        from defumat.workflows.nscf import run_nscf

        result = self._ground_state("an NSCF run")
        return run_nscf(self.system, self.pseudos, result.density,
                        kpoints=kpoints,
                        **self._call_options(run_nscf, result, options))

    def get_dos(self, grid=None, **options):
        """The density of states, on a denser grid than the SCF's if asked.

        Returns the :class:`~defumat.workflows.dos.DensityOfStates`; the NSCF
        states it was integrated from are left on :attr:`dos_states`.
        """
        from defumat.workflows.dos import run_dos

        result = self._ground_state("a density of states")
        dos, states = run_dos(self.system, self.pseudos, result.density,
                              grid=grid,
                              **self._call_options(run_dos, result, options))
        self.dos_states = states
        return dos

    def get_pdos(self, grid=None, **options):
        """The projected density of states, with Löwdin charges and spilling.

        Returns the :class:`~defumat.workflows.pdos.ProjectedDOS`; its NSCF
        states are left on :attr:`pdos_states`.
        """
        from defumat.workflows.pdos import run_pdos

        result = self._ground_state("a projected density of states")
        pdos, states = run_pdos(self.system, self.pseudos, result, grid=grid,
                                **self._defaults_for(run_pdos, options))
        self.pdos_states = states
        return pdos

    # ------------------------------------------------------------------
    # derivatives of the energy
    # ------------------------------------------------------------------

    def get_forces(self, method=None):
        """The forces on the atoms, by ``jax.grad`` of the energy by default."""
        from defumat.forces import compute_forces

        result = self._ground_state("the forces")
        return compute_forces(self.calculation, result, method=method)

    def get_stress(self, method=None, terms: bool = False):
        """The stress tensor.

        An SCF run with ``tstress`` has already computed it, and that result is
        returned rather than differentiated a second time.
        """
        from defumat.stress import compute_stress

        result = self._ground_state("the stress")
        if result.stress is not None and method is None and not terms:
            return result.stress
        return compute_stress(self.calculation, result, method=method,
                              terms=terms)

    def get_relax(self, variable_cell: bool = False, **options):
        """Relax the geometry, and return the whole optimisation.

        ``variable_cell`` relaxes the cell with the atoms at an applied
        pressure -- QE's ``vc-relax``, whose result carries the Pulay error of
        the frozen basis beside the relaxed structure.

        This does **not** move *this* calculator: its cached ground state still
        belongs to the geometry it was built for. :meth:`relaxed` gives the
        calculator at the endpoint.

        The previous relaxation is dropped before the new one starts, for the
        reason :meth:`get_scf` gives: it holds every ionic step's converged
        wavefunctions, and a rebind releases nothing until the call returns.
        """
        run = _relax_entry_point(variable_cell)
        merged = self._defaults_for(run, options)
        self._relax = self._relax_options = None
        self._relax = run(self.system, self.pseudos,
                          calculation=self.calculation, **merged)
        self._relax_variable_cell = variable_cell
        # The key, for :meth:`relaxed`'s cache test. ``get_relax`` itself always
        # reruns, so this is the only thing that reads it.
        self._relax_options = merged
        return self._relax

    def relaxed(self, variable_cell: bool = False, **options) -> "Calculator":
        """A new calculator at the relaxed geometry, with its SCF cached.

        The endpoint of :meth:`get_relax` as an object one can go on computing
        with::

            phonons = calc.relaxed().get_phonons()

        A relaxation already converged an SCF at its final geometry, so that
        result is carried across rather than recomputed -- **when it is a
        result in the relaxed geometry's own basis**, which is not the same
        thing and was not checked.

        A variable-cell relaxation with ``final_scf=False`` and the default
        ``treinit_gvectors=False`` leaves ``VCRelaxResult.scf`` as the last SCF
        *of the relaxation*, and that ran through ``Calculation.at_cell``,
        which freezes the FFT grid and the sphere's Miller indices at the
        starting cell, ``scale_h.f90`` fashion. Carrying it here put those
        wavefunctions in front of a ``Calculation`` enumerated on the relaxed
        cell: a shape error where ``npwx`` differs, and a silently wrong force,
        stress, phonon or dielectric tensor where the two counts happen to
        coincide, since the Miller indices still differ. With
        ``treinit_gvectors=True`` there is no such gap -- every ionic step
        built its own ``Calculation`` at its own cell -- which is why the test
        is :attr:`~defumat.workflows.vc_relax.VCRelaxResult.scf_in_relaxed_basis`
        and not ``final_scf`` alone.

        Where it does not hold the derived calculator simply starts with an
        empty cache and converges its own SCF, which is what it would have done
        had the relaxation never run. Nothing is lost but the reuse, and the
        alternative was a wrong number.
        """
        # **Keyed by the options, like every other slot on this facade.** A
        # second ``relaxed(forc_conv_thr=1e-5, nstep=100)`` after a plain
        # ``relaxed()`` used to return the calculator built on the *loose*
        # relaxation and never pass either option to ``run_relax``, with nothing
        # on the result saying which relaxation it came from -- and what is
        # computed next, a dynamical matrix above all, then sits at a geometry
        # carrying the default relaxation's residual forces, where the acoustic
        # sum rule is an atom-sum identity and blind to exactly that.
        merged = self._defaults_for(_relax_entry_point(variable_cell), options)
        if (self._relax is None
                or self._relax_variable_cell != variable_cell
                or not _same_options(merged, self._relax_options)):
            self.get_relax(variable_cell=variable_cell, **options)
        result = self._relax
        if not result.converged:
            raise ValueError(
                "the relaxation did not converge, so there is no relaxed "
                "geometry to build a calculator on. Read get_relax()'s own "
                "result to see how far it got"
            )
        in_basis = getattr(result, "scf_in_relaxed_basis", True)
        return self._derived(result.system,
                             scf=result.scf if in_basis else None)

    # ------------------------------------------------------------------
    # linear response
    # ------------------------------------------------------------------

    def get_band_velocities(self, kpoints=None, **options):
        """``d(eps)/dk`` for every band, from one ``jvp`` of ``H(k)``.

        The velocity operator is P24's first layer and the thing a Fermi
        velocity or an effective mass is read off. ``kpoints`` computes them
        somewhere other than the ground state's own grid -- a band path,
        typically -- which is an NSCF diagonalisation followed by the same
        operator.

        The overlap carries a velocity too, so what is returned is
        ``<psi|dH/dk - eps dS/dk|psi>`` and not the bare ``dH/dk``.
        """
        from defumat.response.velocity import band_velocities

        result = self._ground_state("the band velocities")
        return band_velocities(
            self.calculation, result, kpoints=kpoints,
            **self._defaults_for(band_velocities, options,
                                 exclude=SCF_ONLY_OPTIONS),
        )

    def get_effective_mass(self, kpoint, **options):
        """``(1/m*)_ab = (1/2) d^2 eps_n/dk_a dk_b`` at one k-point, in 1/m_e.

        ``kpoint`` is in crystal coordinates and the tensor that comes back is
        cartesian. The first derivative is the velocity operator's ``jvp`` and
        the second is one central difference of it, so this costs an NSCF over
        a thirteen-point stencil and nothing else. Bands inside a degenerate
        multiplet are refused individually and reported as the multiplet's
        invariant sum.
        """
        from defumat.response.effmass import effective_mass

        result = self._ground_state("the effective mass")
        return effective_mass(
            self.calculation, result, kpoint,
            **self._defaults_for(effective_mass, options,
                                 exclude=SCF_ONLY_OPTIONS),
        )

    def get_angular_momenta(self, **options):
        """``<L>``, ``<S>`` and ``<J>`` on every atom, in units of ``hbar``.

        The site decomposition ``pw.x`` has no counterpart for -- ``lorbm``
        gives the *cell's* orbital magnetization and nothing per atom. ``<L>``
        is quenched to zero without spin-orbit coupling, so a nonzero one is a
        statement about the coupling rather than about the projector set.
        """
        from defumat.projwfc.angular_momentum import angular_momenta

        result = self._ground_state("the site angular momenta")
        return angular_momenta(
            self.calculation, result,
            **self._defaults_for(angular_momenta, options,
                                 exclude=SCF_ONLY_OPTIONS),
        )

    def get_dielectric_tensor(self, **options):
        """``epsilon_infinity`` by the Sternheimer route -- no empty states."""
        from defumat.response.efield import dielectric_tensor

        result = self._ground_state("the dielectric tensor")
        return dielectric_tensor(
            self.calculation, result.wavefunctions, result.eigenvalues,
            result.density, result.becsum,
            **self._defaults_for(dielectric_tensor, options,
                                 exclude=SCF_ONLY_OPTIONS),
        )

    def get_born_charges(self, **options):
        """The Born effective charges ``Z* = dF/dE``, as a ``(nat, 3, 3)``.

        They come with the dielectric tensor -- both are the response to the
        same field -- so this solves once and returns the charges.
        """
        options.setdefault("born_charges", True)
        tensor = self.get_dielectric_tensor(**options)
        if tensor.born_charges is None:
            raise ValueError(
                "the dielectric solve was asked for no Born charges "
                "(born_charges=False), so there are none to return"
            )
        return tensor.born_charges

    def get_phonons(self, **options):
        """The dynamical matrix at ``Gamma``, and the modes it diagonalises to."""
        from defumat.response.phonon import dynamical_matrix

        result = self._ground_state("the dynamical matrix")
        return dynamical_matrix(
            self.calculation, result.wavefunctions, result.eigenvalues,
            result.density, result.becsum,
            **self._defaults_for(dynamical_matrix, options,
                                 exclude=SCF_ONLY_OPTIONS),
        )

    def get_phonons_at_q(self, q=(0.0, 0.0, 0.0), **options):
        """The dynamical matrix at one wavevector ``q``, and its frequencies.

        ``q`` is in crystal coordinates of the reciprocal lattice unless
        ``q_cartesian = True``, which reads it in the ``2 pi / alat`` units
        ``ph.x`` prints. ``q = 0`` is the same physics as :meth:`get_phonons`
        through a second plane-wave sphere, so it is a regression rather than a
        second way of asking; away from the zone centre it is the only route.
        """
        from defumat.response.phononq import dynamical_matrix_at_q

        result = self._ground_state("the dynamical matrix at q")
        return dynamical_matrix_at_q(
            self.calculation, result.wavefunctions, result.eigenvalues,
            result.density, result.becsum, q=q,
            **self._defaults_for(dynamical_matrix_at_q, options,
                                 exclude=SCF_ONLY_OPTIONS),
        )

    def get_raman_tensors(self, **options):
        """``d(epsilon)/d(tau)``: the Raman tensor of each atom."""
        from defumat.response.nonlinear import raman_tensors

        result = self._ground_state("the Raman tensors")
        return raman_tensors(
            self.calculation, result,
            **self._defaults_for(raman_tensors, options,
                                 exclude=SCF_ONLY_OPTIONS),
        )

    def get_vibrational_spectrum(self, **options):
        """Per-mode Raman and infrared activities -- what a spectrum plots."""
        from defumat.response.spectra import vibrational_spectrum

        result = self._ground_state("a vibrational spectrum")
        return vibrational_spectrum(
            self.calculation, result,
            **self._defaults_for(vibrational_spectrum, options,
                                 exclude=SCF_ONLY_OPTIONS),
        )

    def get_strain_response(self, **options):
        """The first-order response to a homogeneous strain.

        Cached, because both the elastic constants and the electrostriction are
        built from it, and it is the expensive half of either.

        ``options`` recomputes, and the old response is released first: it is
        six distinct ``(nk, nocc, npwx)`` blocks -- the ``(3, 3)`` array of
        ``dpsi`` after symmetrisation -- so holding it while its replacement is
        built doubles the largest thing this calculator owns.

        **The slot is keyed by the options that filled it**, which is the
        facade's rule for its one cache and what ``_scf`` and ``_ultracell``
        already do. Testing ``or options`` instead let a *later* call with no
        options reuse whatever an earlier call with them had produced: a slot
        filled at ``tr2 = 1e-6`` came straight back out of
        :meth:`get_elastic_constants`, which asks for the response with no
        options at all, and nothing on the result said which tolerance it had
        been built at.
        """
        from defumat.response.strain import strain_response

        result = self._ground_state("the strain response")
        merged = self._defaults_for(strain_response, options,
                                    exclude=SCF_ONLY_OPTIONS)
        if self._strain_response is not None and _same_options(
            merged, self._strain_response_options
        ):
            return self._strain_response
        self._strain_response = self._strain_response_options = None
        self._strain_response = strain_response(
            self.calculation, result.wavefunctions, result.eigenvalues,
            result.density, result.becsum, **merged,
        )
        self._strain_response_options = merged
        return self._strain_response

    def get_elastic_constants(self, **options):
        """``C_ijkl``, the stress differentiated along the strain response."""
        from defumat.response.elastic import elastic_constants

        result = self._ground_state("the elastic constants")
        response = self.get_strain_response()
        return elastic_constants(
            self.calculation, result.wavefunctions, result.eigenvalues,
            result.density, response,
            **self._defaults_for(elastic_constants, options,
                                 exclude=SCF_ONLY_OPTIONS),
        )

    def get_piezoelectric_tensor(self, **options):
        """``e_(k)ij``: the clamped-ion piezoelectric tensor, in C/m^2."""
        from defumat.response.piezo import piezoelectric_tensor

        result = self._ground_state("the piezoelectric tensor")
        return piezoelectric_tensor(
            self.calculation, result,
            **self._defaults_for(piezoelectric_tensor, options,
                                 exclude=SCF_ONLY_OPTIONS),
        )

    def get_piezoelectric_kmesh_ladder(self, **options):
        """``e_(k)ij`` at a ladder of k-meshes, which is the check it has none of.

        Every statement the piezoelectric tensor makes about itself is blind to
        the Brillouin-zone sum -- the routes share one field response and the
        ``Z*`` anchor is the same assembly in another coordinate -- so a
        committed-quality mesh returned a number thirteen per cent out and said
        nothing. This runs a fresh ground state and response per mesh and
        reports what the last step moved. It is the whole calculation over again
        per rung, so reach for ``method='zstar_eu'`` on an augmented dataset.
        """
        from defumat.workflows.piezo_ladder import piezoelectric_kmesh_ladder

        return piezoelectric_kmesh_ladder(
            self.system, self.pseudos,
            **self._defaults_for(piezoelectric_kmesh_ladder, options,
                                 exclude=SCF_ONLY_OPTIONS),
        )

    def get_electrostriction(self, **options):
        """``d(chi)/d(strain)`` and the four electrostriction tensors."""
        from defumat.response.electrostriction import electrostriction

        result = self._ground_state("the electrostriction tensors")
        options.setdefault("strain", self._strain_response)
        return electrostriction(
            self.calculation, result,
            **self._defaults_for(electrostriction, options,
                                 exclude=SCF_ONLY_OPTIONS),
        )

    def get_absorption(self, frequencies, **options):
        """An optical absorption spectrum, by a sum over states plus a Dyson
        solve with an exchange-correlation kernel from the registry."""
        from defumat.workflows.tddft import run_absorption

        result = self._ground_state("an absorption spectrum")
        return run_absorption(
            self.system, self.pseudos, result.density, frequencies,
            **self._call_options(run_absorption, result, options,
                                 exclude=SCF_ONLY_OPTIONS)
        )

    def get_optical_conductivity(self, **options):
        """``sigma_ab(omega)``, the Kerr angle and the anomalous Hall conductivity.

        The whole complex tensor, interband plus Drude. Its **antisymmetric**
        part is what needs magnetism and spin-orbit coupling at the same time,
        and is what a magneto-optical Kerr measurement reads; the result
        carries ``.kerr`` in degrees and ``.hall_conductivity`` in S/cm.
        """
        from defumat.workflows.conductivity import run_conductivity

        result = self._ground_state("the optical conductivity")
        return run_conductivity(
            self.system, self.pseudos, result.density,
            **self._call_options(run_conductivity, result, options,
                                 exclude=SCF_ONLY_OPTIONS)
        )

    def get_structure_factors(self, hmax: float = 6.0, **options):
        """``F(H)``, the X-ray and magnetic structure factors of the density.

        The Fourier coefficients of the converged density and magnetization on
        the reflections a diffraction experiment measures, in electrons and in
        Bohr magnetons per cell. ``hmax`` is the cutoff on ``|H|`` in 1/bohr
        and cannot exceed ``sqrt(ecutrho)``; ``window`` is Elk's ``wsfac``,
        which rebuilds the density from a chosen energy range of states and is
        what makes the quantity a probe of bonding.

        The density is **valence-only**, so these are not the experimental
        structure factors -- except in a forbidden reflection, where the
        spherical part of every atom cancels and what is left is the bonding
        charge the pseudopotential keeps.
        """
        from defumat.workflows.sfac import run_structure_factors

        result = self._ground_state("structure factors")
        return run_structure_factors(
            self.system, self.pseudos, result, hmax=hmax,
            **self._call_options(run_structure_factors, result, options,
                                 exclude=SCF_ONLY_OPTIONS)
        )

    def get_stm(self, height=None, **options):
        """A Tersoff-Hamann scanning-tunnelling image of a surface.

        The tunnelling current an s-wave tip draws is the sample's local
        density of states at the tip, so the image is the density rebuilt from
        the states the bias selects -- a delta at the Fermi level with no
        ``bias`` (Elk's task 162) and the window ``[E_F, E_F + V]`` with one
        (``PP/src/stm.f90``). ``height`` is the crystal coordinate of the tip
        plane above the slab; ``mode="constant-current"`` with a ``current``
        set-point returns the corrugation in bohr instead.

        A delta at the Fermi level wants a **denser k-grid** than the SCF's,
        which ``grid`` re-solves the bands on.
        """
        from defumat.workflows.stm import run_stm

        result = self._ground_state("an STM image")
        self._say_if_an_ultracell_is_waiting("an STM image", "get_ultracell_stm()")
        if height is not None:
            options = {**options, "height": height}
        return run_stm(
            self.system, self.pseudos, result,
            **self._call_options(run_stm, result, options,
                                 exclude=SCF_ONLY_OPTIONS)
        )

    def get_sts(self, energies=None, **options):
        """``dI/dV(r, V)``: a tunnelling spectrum, the curve beside the image.

        :meth:`get_stm` gives the local density of states at one tip energy,
        which is one picture at one bias; this is the other section of the same
        function, the curve at one place over many biases. It resolves a gap, a
        band edge or a state in the gap, none of which an image at a single
        energy shows.

        ``energies`` is the axis in Ry and has no default -- it *is* the
        measurement. ``tip`` gives the positions explicitly, one point being a
        spectrum and a row of them a line cut; ``height`` gives a whole plane
        and a map at every energy.

        Two things a spectrum refuses where an image does not, both because
        ``psi(r)`` is sampled and squared rather than summed into a density: a
        **symmetry-reduced** k-set, which nothing here symmetrises, and a tip
        inside an **augmentation sphere**, where the pseudo-wavefunction is not
        the true one.
        """
        from defumat.workflows.stm import run_sts

        result = self._ground_state("a tunnelling spectrum")
        self._say_if_an_ultracell_is_waiting("a tunnelling spectrum",
                                             "get_ultracell_sts()")
        if energies is not None:
            options = {**options, "energies": energies}
        return run_sts(
            self.system, self.pseudos, result,
            **self._call_options(run_sts, result, options,
                                 exclude=SCF_ONLY_OPTIONS)
        )

    def get_vertical_transport(self, exit_height=None, height=None, **options):
        """Tunnelling *through* a two-dimensional material, tip to substrate.

        An electron enters at a point above the material and leaves into an
        infinite plane below it, so what decides the current is the nonlocal
        Green's function between the two rather than the local density of
        states at the tip. On a material with one band to tunnel through the
        map is the Tersoff-Hamann image of :meth:`get_stm`; on a stack the
        bands interfere on the way through and it is not, which
        :attr:`~defumat.transport.green.VerticalTransport.interference`
        reports.

        ``exit_height`` is the substrate plane's crystal coordinate and
        ``height`` the tip plane's, with the material between them. The k-set
        must be the **whole** grid, which ``grid`` builds.

        **Both leads can be magnetic, and the names are asymmetric.** ``spin``
        and ``polarization`` describe the **substrate**, which is the opposite
        of :meth:`get_stm`, where they describe the tip; the tip's own moment is
        ``tip_spin`` / ``tip_polarization``. Given both, the map depends on the
        angle between the two moments, which is a tunnelling-magnetoresistance
        image.
        """
        from defumat.workflows.transport import run_vertical_transport

        result = self._ground_state("a vertical transmission")
        self._say_if_an_ultracell_is_waiting("a vertical transmission", "get_ultracell_transport()")
        if exit_height is not None:
            options = {**options, "exit_height": exit_height}
        if height is not None:
            options = {**options, "height": height}
        return run_vertical_transport(
            self.system, self.pseudos, result,
            **self._call_options(run_vertical_transport, result, options,
                                 exclude=SCF_ONLY_OPTIONS)
        )

    def get_momentum_transport(self, exit_height=None, height=None, **options):
        """Which k-points the tunnelling current comes out of.

        The conjugate of :meth:`get_vertical_transport`: a **plane** tip in
        place of a point one, so the real-space map collapses and what is left
        is one weight per k-point -- which pocket of the Fermi surface an
        electron actually leaves through. It is the same object, and
        integrating the map over the tip plane gives the sum over k of this.

        Three columns come back together because they come from the same two
        Gram matrices, and the physics is in their ratio: the transmission, its
        Tersoff-Hamann limit (the substrate made structureless), and the plain
        Fermi surface. The k-set must be the **whole** grid, which ``grid``
        builds.

        A whole grid of a real slab is an hour, most of it in a band solve whose
        k loop is compiled and cannot print from inside itself, so pass
        ``report=print`` on anything that size: it says the shape of the work
        before the silence rather than after it.
        """
        from defumat.workflows.transport import run_momentum_transport

        result = self._ground_state("a momentum-resolved transmission")
        self._say_if_an_ultracell_is_waiting("a momentum-resolved transmission", "get_ultracell_transport(), which maps it in real space instead")
        if exit_height is not None:
            options = {**options, "exit_height": exit_height}
        if height is not None:
            options = {**options, "height": height}
        return run_momentum_transport(
            self.system, self.pseudos, result,
            **self._call_options(run_momentum_transport, result, options,
                                 exclude=SCF_ONLY_OPTIONS)
        )

    def get_ultracell(self, supercell, kgrid=(1, 1, 1), **options):
        """A density or potential modulated over many unit cells at once.

        The ultra long-range method (``PLAN.md`` P88, Elk's task 700). A spin
        density wave, a screened impurity or a domain wall is a slow **envelope**
        on a crystal that is still atomically periodic, and a supercell pays the
        same price for the envelope as for the atoms. Here the unit cell's own
        Kohn-Sham states at the ``N`` folded k-points are computed **once** and
        then used as the basis for an ultracell of ``supercell = (n1, n2, n3)``
        cells, so the self-consistency runs on the envelope alone.

        ``kgrid`` samples the *ultracell's* Brillouin zone, which is ``N`` times
        smaller than the unit cell's. ``nbnd`` is the one knob the accuracy
        depends on -- it is the size of the variational basis per folded
        k-point, and the answer converges to the real ``N``-cell supercell as it
        grows -- so pass it. ``external`` is an applied potential in Ry over the
        ultracell and ``magnetic_field`` an applied ``B(r)``. Both spin regimes
        are in: ``nspin = 2`` takes a scalar field and gives a modulation of the
        moment's *length*, and ``nspin = 4`` takes a vector one and gives a
        modulation of its *direction* -- a helix or a cycloid -- with spin-orbit
        coupling along for free, since that lives entirely in the frozen states.

        **A magnetic modulation comes from one of two places and they are
        different quantities**, because nothing in an SCF breaks spin symmetry
        on its own and the tiled state is an exact fixed point. A
        ``magnetic_field`` *drives* one, and what comes back is the Q-resolved
        susceptibility -- a response. ``seed_magnetization`` hands the loop the
        texture as its *initial condition* and the loop keeps it, which is the
        ordered state itself: a factor ``s(r)`` on the converged cell's own
        moment, scalar for ``nspin = 2`` and a vector that turns it for
        ``nspin = 4``, with ``|s| <= 1``.

        The atoms do not move and the local band structure cannot relax: this
        computes what a modulation does to a fixed crystal, not a different
        crystal.
        """
        from defumat.ultracell.driver import run_ultracell

        result = self._ground_state("an ultracell calculation")
        merged = self._call_options(run_ultracell, result, options,
                                    exclude=SCF_ONLY_OPTIONS)
        key = {**merged, "supercell": tuple(int(n) for n in supercell),
               "kgrid": tuple(int(m) for m in kgrid)}
        # One slot keyed by the options that filled it, for :meth:`get_scf`'s
        # reason and more so: what it holds is the frozen states at the ``N``
        # folded k-points, which is the largest array the method makes and the
        # one an image or a transmission needs afterwards.
        if self._ultracell is not None and _same_options(key, self._ultracell_options):
            return self._ultracell
        self._ultracell = self._ultracell_options = None
        self._ultracell = run_ultracell(
            self.system, self.pseudos, result, supercell, kgrid, **merged
        )
        self._ultracell_options = key
        return self._ultracell

    def _say_if_an_ultracell_is_waiting(self, quantity: str, instead: str):
        """Say which crystal is about to be imaged, when there are two.

        An ultracell result does not become the calculator's ground state --
        it is an expansion *around* one -- so a ``get_*`` that takes the ground
        state still images the **unmodulated** unit cell, correctly and
        silently. That is the right answer to the question asked and the wrong
        answer to the one usually meant, and the two are indistinguishable in a
        plot, so the alternative is named rather than left to be noticed.
        """
        if self._ultracell is None:
            return
        cells = "x".join(str(n) for n in self._ultracell.ultracell.shape)
        warnings.warn(
            f"{quantity} is of the unmodulated unit cell, and a {cells} "
            f"ultracell is cached: its modulation is not in this. Call "
            f"{instead} for the modulated one",
            stacklevel=3,
        )

    def _ultracell_state(self, quantity: str):
        """The cached ultracell, or the refusal that says how to make one.

        Unlike :meth:`_ground_state` this does **not** run one: an SCF has
        defaults for everything and an ultracell has no default ``supercell``,
        so there is nothing to guess and a guess would be an expensive wrong
        answer rather than a convenience.
        """
        if self._ultracell is None:
            raise ValueError(
                f"{quantity} needs an ultracell and none is cached: call "
                "get_ultracell(supercell, kgrid, nbnd=...) first. There is no "
                "default supercell to run one with -- how many cells the "
                "modulation spans is the question being asked"
            )
        return self._ultracell

    def get_ultracell_stm(self, height=None, **options):
        """A Tersoff-Hamann image of the modulation :meth:`get_ultracell` found.

        What an experiment sees above a charge or spin density wave, a screened
        impurity or a domain wall: the tunnelling density of states of the
        modulated crystal, on a plane spanning the whole ultracell. ``height``
        and the plane are in **unit-cell** crystal coordinates running over
        ``[0, n_i)``, the convention ``external=`` already uses, so the number
        means what it means in a unit-cell run.

        A **magnetic tip** (``spin``, ``polarization``) is what a spin density
        wave needs: it is flat in the charge and modulated in the
        magnetization, so a nonmagnetic tip sees almost nothing of it.
        """
        from defumat.workflows.ultracell import run_ultracell_stm

        result = self._ultracell_state("an ultracell STM image")
        if height is not None:
            options = {**options, "height": height}
        return run_ultracell_stm(
            self.system, self.pseudos, result,
            **self._call_options(run_ultracell_stm, result, options,
                                 exclude=SCF_ONLY_OPTIONS)
        )

    def get_ultracell_sts(self, energies=None, **options):
        """``dI/dV`` across the modulation :meth:`get_ultracell` found.

        :meth:`get_ultracell_stm` at every energy of an axis, which is the
        measurement a modulated crystal is actually read with: a charge density
        wave is a gap that opens in antiphase with the charge maxima, a spin
        density wave moves the two channels in opposite directions from cell to
        cell, and a domain wall carries a state that lives nowhere else. None of
        the three shows in an image at one energy.

        The states are sampled at the tip points once and the axis is a matrix
        product after that, so a spectrum costs about what one image costs
        rather than ``nE`` of them. ``tip`` and ``height`` are in **unit-cell**
        crystal coordinates over ``[0, n_i)``, :meth:`get_ultracell_stm`'s
        convention.
        """
        from defumat.workflows.ultracell import run_ultracell_sts

        result = self._ultracell_state("an ultracell tunnelling spectrum")
        if energies is not None:
            options = {**options, "energies": energies}
        return run_ultracell_sts(
            self.system, self.pseudos, result,
            **self._call_options(run_ultracell_sts, result, options,
                                 exclude=SCF_ONLY_OPTIONS)
        )

    def get_ultracell_transport(self, exit_height=None, height=None, **options):
        """Vertical tunnelling through the modulation, tip to substrate.

        :meth:`get_vertical_transport` on an ultracell: the electron enters at a
        point above the sheet and leaves into an infinite plane below it, so the
        map is the nonlocal Green's function of the *modulated* material, and
        where several bands are degenerate at the tip energy it departs from
        :meth:`get_ultracell_stm` by their interference.

        The ultracell must be one cell deep along the stacking axis -- a
        modulation there would put the exit plane between two stacked slabs --
        and its ``kgrid`` must have one division along it.
        """
        from defumat.workflows.ultracell import run_ultracell_transport

        result = self._ultracell_state("an ultracell transmission")
        if exit_height is not None:
            options = {**options, "exit_height": exit_height}
        if height is not None:
            options = {**options, "height": height}
        return run_ultracell_transport(
            self.system, self.pseudos, result,
            **self._call_options(run_ultracell_transport, result, options,
                                 exclude=SCF_ONLY_OPTIONS)
        )

    def get_nesting(self, grid=None, **options):
        """``N(q)``, the Fermi-surface nesting function, on a dense grid.

        How much of the Fermi surface maps onto itself when translated by
        ``q`` -- the geometric half of a charge- or spin-density-wave
        instability, and what says where a phonon will soften or a spin spiral
        will find its pitch. ``grid`` is the convergence parameter of the whole
        quantity and wants to be much denser than the density needed.

        ``q = 0`` is the maximum on every crystal by Cauchy-Schwarz, so the
        result's ``.peak()`` reports the largest ``N(q)`` away from it.
        """
        from defumat.workflows.nesting import run_nesting

        result = self._ground_state("the nesting function")
        if grid is not None:
            options = {**options, "grid": grid}
        return run_nesting(
            self.system, self.pseudos, result.density,
            **self._call_options(run_nesting, result, options,
                                 exclude=SCF_ONLY_OPTIONS)
        )

    def get_spin_susceptibility(self, q, frequencies, **options):
        """``chi^{+-}(q, omega)``, the transverse spin susceptibility.

        Its pole is the **magnon**: a collective precession of the whole
        magnetization, pulled out from under the Stoner continuum of
        independent spin flips by the exchange-correlation kernel
        ``B_xc/m``. ``q`` is in crystal coordinates and must be a difference
        of two k-points of the run's own grid, which is where the states at
        ``k + q`` come from.

        The result carries ``.goldstone`` -- how far ``X_0 B_xc = m`` is from
        holding, which is the calculation's own error bar and is worth reading
        before its magnon energy.
        """
        from defumat.workflows.magnons import run_spin_susceptibility

        result = self._ground_state("a spin susceptibility")
        return run_spin_susceptibility(
            self.system, self.pseudos, result.density, q, frequencies,
            **self._call_options(run_spin_susceptibility, result, options,
                                 exclude=SCF_ONLY_OPTIONS)
        )

    def get_magnon_dispersion(self, qpoints, frequencies, **options):
        """``omega(q)`` along a path of wavevectors: the spin-wave dispersion.

        One fixed-density run serves every ``q``, because each is a difference
        of two k-points of the grid. Points where the result's
        ``.enhancements`` exceed one are **instabilities** rather than
        magnons: the collinear state is not a minimum there, and that is where
        a spin spiral would lower the energy.
        """
        from defumat.workflows.magnons import run_magnon_dispersion

        result = self._ground_state("a magnon dispersion")
        return run_magnon_dispersion(
            self.system, self.pseudos, result.density, qpoints, frequencies,
            **self._call_options(run_magnon_dispersion, result, options,
                                 exclude=SCF_ONLY_OPTIONS)
        )

    def get_shift_current(self, kpoints=None, nbnd=None, **options):
        """``sigma^abc(0; w, -w)``, the bulk photovoltaic effect, in A/V^2.

        The direct current a non-centrosymmetric crystal carries under
        illumination, with no junction and no built-in field: the photoexcited
        electron is born displaced, and the shift between the valence and
        conduction Wannier centres is what the current counts. Zero by symmetry
        in any crystal with an inversion centre.

        ``nbnd`` is required and is the convergence parameter of the whole
        quantity -- more so than for an absorption spectrum, because the
        intermediate sum of the generalised derivative runs over the same bands.
        Read ``.truncation`` before believing a number.
        """
        from defumat.workflows.photocurrent import run_shift_current

        result = self._ground_state("the shift current")
        if kpoints is not None:
            options = {**options, "kpoints": kpoints}
        if nbnd is not None:
            options = {**options, "nbnd": nbnd}
        return run_shift_current(
            self.system, self.pseudos, result.density,
            **self._call_options(run_shift_current, result, options,
                                 exclude=SCF_ONLY_OPTIONS)
        )

    def get_shg(self, kpoints=None, nbnd=None, **options):
        """``chi^(2)(-2w; w, w)``, the second-harmonic tensor, in pm/V.

        How much of the light shone on a crystal comes back out at twice the
        frequency. A polar rank-3 tensor, symmetric in its two field labels,
        and identically zero in any crystal with an inversion centre.

        ``nbnd`` is required and is the convergence parameter of the whole
        quantity, for the same reason it is for ``get_shift_current``: the sum
        over the intermediate state is an identity only over a complete basis.
        Read ``.truncation`` before believing a number, and note that the
        literature usually quotes ``d = chi/2`` -- ``.d_coefficient`` gives it.
        """
        from defumat.workflows.shg import run_shg

        result = self._ground_state("second-harmonic generation")
        if kpoints is not None:
            options = {**options, "kpoints": kpoints}
        if nbnd is not None:
            options = {**options, "nbnd": nbnd}
        return run_shg(
            self.system, self.pseudos, result.density,
            **self._call_options(run_shg, result, options,
                                 exclude=SCF_ONLY_OPTIONS)
        )

    # ------------------------------------------------------------------
    # topology
    # ------------------------------------------------------------------

    def get_berry_curvature(self, **options):
        """The Berry curvature on a plane mesh; its integral is the Chern
        number, which the result carries as ``.chern_number``."""
        from defumat.workflows.topology import run_berry_curvature

        result = self._ground_state("the Berry curvature")
        return run_berry_curvature(self.system, self.pseudos, result.density,
                                   **self._call_options(run_berry_curvature,
                                                        result, options))

    def get_orbital_magnetization(self, **options):
        """``M_orb`` in Bohr magnetons per cell, by the modern theory.

        The circulating part of a magnet's moment -- what ``pw.x`` reaches with
        ``lorbm`` and what no integral over the cell can give. It needs a
        magnetic spinor calculation with spin-orbit coupling and a gapped
        manifold; ``divisions`` is the uniform grid it is assembled on and
        defaults to the one the k-points came from.
        """
        from defumat.workflows.orbital_magnetization import (
            run_orbital_magnetization,
        )

        result = self._ground_state("the orbital magnetization")
        return run_orbital_magnetization(
            self.system, self.pseudos, result.density,
            **self._call_options(run_orbital_magnetization, result, options),
        )

    def get_polarization(self, **options):
        """The Berry-phase polarization along one reciprocal lattice vector.

        ``pw.x``'s ``lberry`` run: strings of k-points along ``gdir``, the
        occupied manifold's Berry phase along each, and the ions' phase on top.
        The value is defined **modulo a quantum**, which the result carries --
        what is physical is a *difference* between two geometries, not the
        number on its own.
        """
        from defumat.workflows.polarization import run_polarization

        berry = getattr(self.system, "berry", None)
        if berry is not None:
            # `lberry`, `gdir` and `nppstr` off the input, so that a pw.x
            # polarization run transfers unchanged. An explicit argument still
            # wins, which is what makes a convergence sweep possible without
            # editing the input file.
            options.setdefault("gdir", berry[0])
            options.setdefault("nppstr", berry[1])
        result = self._ground_state("the Berry-phase polarization")
        return run_polarization(self.system, self.pseudos, result.density,
                                **self._call_options(run_polarization,
                                                     result, options))

    def get_magnetoelectric_tensor(self, **options):
        """``alpha_ij = dP_i/dB_j``: the polarization a magnetic field induces.

        Six self-consistent runs and a central difference, which is Elk's task
        390 and the only route available here -- the cheap one needs a
        noncollinear Sternheimer solve. Needs spin-orbit coupling, a gap, a
        crystal without an inversion centre, and time reversal already broken
        (usually by the applied field itself).
        """
        from defumat.response.magnetoelectric import magnetoelectric_tensor

        # ``conv_thr`` is excluded deliberately. The input file's number is the
        # *SCF*'s, and this function's ``conv_thr`` would land in the
        # polarization's slot instead -- which leaves the six ground states on
        # their defaults and is worth a factor of 300 in the spin-orbit null.
        # The two thresholds are named apart in the signature for that reason.
        return magnetoelectric_tensor(
            self.system, self.pseudos,
            **self._defaults_for(magnetoelectric_tensor, options,
                                 exclude=frozenset({"conv_thr"})),
        )

    def get_chern(self, **options) -> float:
        """The Chern number of one plane -- an exact integer on any mesh."""
        return self.get_berry_curvature(**options).chern_number

    def get_anisotropy(self, spinor, directions=None, **options):
        """The magnetocrystalline anisotropy, by the force theorem.

        ``self`` is the **scalar-relativistic collinear** leg -- the run whose
        density this is -- and ``spinor`` is the fully-relativistic
        noncollinear one, a second :class:`Calculator` built from the ``nscf``
        input. Two calculators rather than one because the two legs are two
        different *pseudopotential files*, which is how ``pw.x`` does it as
        well: only the density crosses between them.

            sr  = Calculator.from_file("sr.in")
            soc = Calculator.from_file("par.in")
            mae = sr.get_anisotropy(soc, directions="xyz")

        See :mod:`defumat.workflows.anisotropy`.
        """
        from defumat.workflows.anisotropy import becsum_for_leg, run_anisotropy

        result = self._ground_state("the magnetic anisotropy")
        if result.nspin == 1:
            raise ValueError(
                "the force theorem's first leg is a spin-polarized run and "
                "this one has nspin = 1: there is no magnetization to rotate, "
                "and every direction would come out equal"
            )
        system, pseudos, leg = _spinor_leg(spinor)
        forwarded = self._defaults_for(run_anisotropy, options,
                                       exclude=_SPINOR_LEG_OPTIONS)
        for name, value in leg.items():
            forwarded.setdefault(name, value)
        forwarded.setdefault(
            "becsum", becsum_for_leg(result.becsum, self.pseudos, pseudos)
        )
        return run_anisotropy(
            system, pseudos, result.density, directions=directions, **forwarded,
        )

    def get_relaxed_anisotropy(self, directions=None, **options):
        """The magnetocrystalline anisotropy from **total** energies, relaxed.

        One full self-consistent noncollinear run per direction, differenced.
        ``self`` is the fully-relativistic noncollinear calculator itself and
        there is no second leg, because nothing is handed between two runs --
        which is exactly why this route reaches what the force theorem cannot:

            soc = Calculator.from_file("ni-soc.in")     # PAW is fine here
            mae = soc.get_relaxed_anisotropy(directions="xz")

        Against :meth:`get_anisotropy`, which freezes the density converged
        without spin-orbit coupling and diagonalises once per direction, this
        lets the density respond in each direction and pays for it in two SCF
        runs. The extra energy is variational and the anisotropy is the
        difference of two such gains. **PAW and a Hubbard ``U`` are allowed
        here and refused there**, because the theorem's handoff is a density
        and ``ddd_paw``/``ns`` are properties of wavefunctions.

        Read ``.converged`` and ``.drifts`` before the number: nothing holds
        the moment, so a direction that is not stationary by symmetry can end
        up somewhere other than where it was started.
        """
        from defumat.workflows.anisotropy import run_relaxed_anisotropy

        return run_relaxed_anisotropy(
            self.system, self.pseudos, directions=directions,
            **{**self._shared_scf_options(withheld=("conv_thr",)),
               **self._defaults_for(run_relaxed_anisotropy, options),
               **options},
        )

    def get_exchange_torque(self):
        """``int m x B_xc``: how far the texture is from stationary.

        Elk's task 160. **It is identically zero unless the run set
        ``nosource``**, because a functional of ``|m|`` alone builds a field
        parallel to ``m`` at every point -- the result carries
        ``parallel_fraction`` so that a zero can be told from that silence. See
        :mod:`defumat.scf.spin_torque`.
        """
        from defumat.scf.spin_torque import torque_of_result

        result = self._ground_state("the exchange-correlation spin torque")
        return torque_of_result(self.calculation, result)

    def get_torque(self, spinor, angle=None, **options):
        """The magnetic torque, and the anisotropy constant from one angle.

        The derivative route to what :meth:`get_anisotropy` takes as a
        difference -- see :func:`defumat.workflows.anisotropy.run_torque`.
        """
        from defumat.workflows.anisotropy import run_torque

        result = self._ground_state("the magnetic torque")
        system, pseudos, leg = _spinor_leg(spinor)
        merged = self._defaults_for(run_torque, options,
                                    exclude=_SPINOR_LEG_OPTIONS)
        for name, value in leg.items():
            merged.setdefault(name, value)
        if angle is not None:
            merged["angle"] = angle
        return run_torque(system, pseudos, result.density, **merged)

    def get_first_order_soc(self, spinor, direction=None, **options):
        """The spin-orbit term's expectation value at coupling-free states.

        The calculation :meth:`get_anisotropy` is often assumed to be, and it
        returns essentially zero -- see
        :func:`defumat.workflows.anisotropy.frozen_expectation`.
        """
        from defumat.workflows.anisotropy import frozen_expectation

        result = self._ground_state("the first-order spin-orbit term")
        system, pseudos, leg = _spinor_leg(spinor)
        merged = self._defaults_for(frozen_expectation, options,
                                    exclude=_SPINOR_LEG_OPTIONS)
        for name, value in leg.items():
            merged.setdefault(name, value)
        return frozen_expectation(
            system, pseudos, result.density, direction=direction, **merged,
        )

    def get_force_theorem(self, spinor, direction=None, **options):
        """One direction of :meth:`get_anisotropy`: its band energy alone."""
        from defumat.workflows.anisotropy import run_force_theorem

        result = self._ground_state("a force-theorem band energy")
        system, pseudos, leg = _spinor_leg(spinor)
        merged = self._defaults_for(run_force_theorem, options,
                                    exclude=_SPINOR_LEG_OPTIONS)
        for name, value in leg.items():
            merged.setdefault(name, value)
        return run_force_theorem(
            system, pseudos, result.density, direction=direction, **merged,
        )

    def get_z2(self, **options):
        """The 2D Z2 invariant of one plane, by Wilson loops or Fu-Kane parity."""
        from defumat.workflows.topology import run_z2

        result = self._ground_state("the Z2 invariant")
        return run_z2(self.system, self.pseudos, result.density,
                      **self._call_options(run_z2, result, options))

    def get_z2_3d(self, **options):
        """The four Z2 indices of a three-dimensional crystal."""
        from defumat.workflows.topology import run_z2_3d

        result = self._ground_state("the 3D Z2 invariants")
        return run_z2_3d(self.system, self.pseudos, result.density,
                         **self._call_options(run_z2_3d, result, options))

    # ------------------------------------------------------------------
    # spin spirals
    # ------------------------------------------------------------------

    def get_spiral_scan(self, wavevectors, **options):
        """One SCF per spiral wavevector, sharing what does not depend on ``q``."""
        from defumat.workflows.spiral import run_spiral_scan

        return run_spiral_scan(self.system, self.pseudos, wavevectors,
                               **{**self._shared_scf_options(), **options})

    def get_spiral_relaxation(self, **options):
        """Relax the spiral wavevector itself: ``dE/dq`` downhill by BFGS."""
        from defumat.workflows.spiral import relax_spiral_q

        return relax_spiral_q(self.system, self.pseudos,
                              calculation=self.calculation,
                              **self._defaults_for(relax_spiral_q, options))

    # ------------------------------------------------------------------
    # deriving one calculator from another
    # ------------------------------------------------------------------

    def with_positions(self, positions) -> "Calculator":
        """A calculator with the atoms moved, and an empty cache.

        The cached ground state belongs to the geometry it converged for, so it
        does not cross. The density does: it is handed to the new calculator as
        a starting guess, which is what ``update_pot.f90`` does between the
        steps of a relaxation and is worth several SCF iterations.
        """
        system = self.system.with_cell(self.system.cell.at, positions)
        return self._derived(system, seed=True, basis=self._frozen_basis(system))

    def _frozen_basis(self, system: System):
        """This calculator's plane-wave setup, to be reused at a new geometry.

        **The FFT grid is a function of the symmetry, not only of the cutoffs**,
        and moving an atom changes the symmetry. ``symm_base.f90``'s rule that
        the grid dimensions be a multiple of the fractional translations'
        denominators is implemented here as ``fft_factors``, so on the
        canonical silicon cell a displacement of **0.02 bohr** takes ``nsym``
        from 48 to 4, ``fft_factors`` from ``(4, 4, 4)`` to ``(1, 1, 1)``, and
        the dense grid from ``(16, 16, 16)`` to ``(15, 15, 15)``.

        Rebuilding, which is what this used to do, was wrong in both of its
        branches. With a converged parent the seed this method promises to
        carry was then refused by ``_check_grid`` -- *"the source density is on
        a (16, 16, 16) grid and this run uses (15, 15, 15)"*, for a displacement
        that changed neither the cell nor either cutoff. Without one it ran
        silently at the displaced grid, so a finite difference built through the
        front door differenced two different grids.

        Freezing is not a workaround but the same thing the rest of the package
        does: :meth:`~defumat.scf.driver.Calculation.at_positions` freezes the
        grid for exactly this reason, ``run_relax`` goes through it, and this
        method's own docstring calls itself the ``update_pot.f90`` step between
        relaxation steps. What it costs is that a *large* displacement keeps a
        grid chosen for the original symmetry, which is denser than the new
        geometry needs rather than sparser -- the safe direction. A genuinely
        fresh setup at a new geometry is ``Calculator.from_file`` with the new
        positions, which is what it always was.

        **Freezing is conditional, and the condition is the rule that made the
        grid move in the first place.** ``symm_base.f90`` requires the grid
        dimensions to be a multiple of the fractional translations'
        denominators, and this package's answer is to choose the grid to fit
        the translations rather than filter the translations to fit the grid.
        So a frozen grid is only usable where the *target's* group is still
        commensurate with it -- and it is not, in the one direction this
        method's own purpose does not cover: moving an atom **onto** a more
        symmetric site. Measured, a calculator built directly on a displaced
        silicon (``nsym = 4``, grid ``(15, 15, 15)``) moved to the ideal site
        (``nsym = 48``, ``fft_factors = (4, 4, 4)``): 15 is not a multiple of 4,
        so freezing there would hand ``sym_rho`` operations whose fractional
        translations the grid cannot represent. That case rebuilds, and the
        seed check then raises honestly, because the grid really did change.
        """
        basis = self.calculation.basis
        grid = basis.dense.grid
        factors = system.symmetry_group(system.nosym).fft_factors()
        if any(n % int(f) for n, f in zip(grid, factors)):
            return None
        return basis

    def with_cell(self, at, positions=None) -> "Calculator":
        """A calculator on a deformed cell, with an empty cache.

        Nothing is carried across as a starting guess here, unlike
        :meth:`with_positions`: the density is on a grid the cell defines, so a
        cell that has moved is a different grid.
        """
        return self._derived(self.system.with_cell(at, positions))

    def with_kpoints(self, kpoints) -> "Calculator":
        """A calculator on the same crystal, sampled differently.

        A k-set is part of the ``System``, so a different one is a different
        system and gets its own calculator rather than an argument. The density
        crosses as a starting guess -- the crystal has not moved, so the
        previous sampling's density is a good one -- but not as an answer,
        since it is a *different* integral over the zone.

        The unreduced grid beside its irreducible wedge is the usual reason to
        want this, and comparing the two is the check on the symmetry
        reduction: same energy, fewer k-points.

        **The weights are normalised on the way in**
        (:func:`defumat.system.kpoints.for_spin`), because every ``KPoints``
        constructor applies the spin degeneracy unconditionally and a polarized
        run wants it halved -- ``setup.f90`` applies ``degspin`` only in its LDA
        branch. Substituting a raw ``KPoints.automatic`` here used to skip that
        step, which counts every electron twice and does not fail: the Fermi
        level moves and the run integrates to the right electron count at the
        wrong energy, on exactly the comparison the paragraph above recommends.
        A k-set that has already been normalised -- one from
        :func:`~defumat.workflows.nscf.denser_grid`, say -- is left alone, the
        division being idempotent through ``KPoints.spin_normalized``.
        """
        return self._derived(
            dataclasses.replace(
                self.system, kpoints=for_spin(kpoints, self.system.nspin)
            ),
            seed=True,
        )

    def with_spin(self, nspin=None, magnetization="auto", **options) -> "Calculator":
        """A calculator in another spin regime, warm-started from this one.

        The three regimes are three ways of writing the same ``(n, m)``, so a
        converged state promotes into the target's variables rather than being
        thrown away (P23): a collinear result promoted into a noncollinear run
        whose magnetization only has to be rotated converges in one iteration
        instead of twenty-five.

        ``magnetization`` is how that state's moment crosses, and it is
        ``run_scf``'s argument of the same name reaching the front door.
        ``"auto"`` is the default and is what a promotion wants: carry the
        source's moment where it has one, seed this system's
        ``starting_magnetization`` where it does not. Pass ``"seed"`` to keep
        the converged **charge** and take the moment from the seed regardless,
        which is how a *different* magnetic state is reached from the same
        charge, and ``"none"`` to start unpolarized.
        """
        system = self.system.with_spin(nspin, **options)
        return self._derived(system, seed=True, magnetization=magnetization)

    def with_moments(self, per_atom, magnetization="seed") -> "Calculator":
        """A calculator with a different moment on each atom, warm-started.

        The Python route to a ``STARTING_MOMENTS`` card: ``(nat, 3)`` Bohr
        magnetons in ``ATOMIC_POSITIONS`` order, or ``None`` to drop the card.
        It is what a sweep over magnetic configurations needs -- a 120-degree
        Neel state against a collinear one, a cone at five angles -- and the
        alternative was writing an input file per configuration.

        The k-points are rebuilt with the new moments' magnetic group, which is
        the whole reason this is a method: see
        :meth:`~defumat.system.builder.System.with_moments` for what
        ``dataclasses.replace`` gets silently wrong.

        The converged state comes across as a seed, not as an answer, exactly
        as it does through :meth:`with_spin` -- a different texture is a
        different calculation, so the cache is empty and the next ``get_scf``
        reruns from this density.

        **``magnetization`` defaults to ``"seed"`` here where
        :meth:`with_spin` leaves it at ``"auto"``, and the difference is the
        whole method.** A new texture is a new ``STARTING_MOMENTS`` card, and
        ``"auto"`` on a source that already has a moment resolves to *carry* --
        which for a noncollinear source is the old texture copied over with no
        rotation, so the card this method exists to set is never read and the
        run starts, and very often finishes, on the configuration it was meant
        to move away from. Seeding keeps the converged charge, which is what the
        iterations went into, and takes the moments from the new card. Pass
        ``magnetization="carry"`` to override, which is what a sweep wants when
        the texture is being nudged rather than replaced.
        """
        return self._derived(self.system.with_moments(per_atom), seed=True,
                             magnetization=magnetization)

    def _derived(self, system: System, *, seed: bool = False, scf=None,
                 magnetization: str = "auto", basis=None) -> "Calculator":
        """A calculator on ``system``, sharing this one's pseudos and options.

        ``basis`` freezes the plane-wave setup at the parent's rather than
        letting the child enumerate its own. Only :meth:`with_positions` passes
        it, and :meth:`_frozen_basis` is where the reason is.
        """
        derived = Calculator(system, self.pseudos, announce=self.announce,
                             basis=basis, **self.defaults)
        if scf is not None:
            derived._scf = scf
            # **Deliberately no options**, so that any explicit ``get_scf``
            # reruns. This state came out of the relaxation's own loop, at
            # whatever settings *it* used; claiming the parent's options
            # produced it would let a later ``get_scf(conv_thr=...)`` cache-hit
            # on a state converged to something else.
            derived._scf_options = None
        elif seed and self._scf is not None and self._scf.converged:
            # Not a cache: a starting point. ``run_scf`` promotes it into the
            # target's variables, seeding the magnetization where the source has
            # none, and converges from there.
            derived._seed = self._scf
            derived._seed_magnetization = magnetization
        return derived

    # ------------------------------------------------------------------
    # plumbing
    # ------------------------------------------------------------------

    def _shared_scf_options(self, withheld: tuple[str, ...] = ()) -> dict:
        """The shared options, for an entry point that hides them in ``**kwargs``.

        :meth:`_defaults_for` matches against a signature, and a ``**kwargs``
        is deliberately not read as permission to forward everything. Two
        workflows nonetheless take their SCF options that way and pass them
        straight to :func:`~defumat.scf.driver.run_scf`, so for those the match
        is against ``run_scf``'s own signature instead:
        :meth:`get_spiral_scan` and :meth:`get_relaxed_anisotropy`. Without it
        the forwarding was not strict but empty -- an input whose
        ``&electrons`` sets ``mixing_beta = 0.2`` because the magnetic cell
        diverges at 0.7 ran its noncollinear SCFs at the library default, and a
        ``k_batch = 1`` set to keep a slab inside RAM was dropped the same way.

        ``withheld`` is for an option the workflow sets deliberately:
        ``run_relaxed_direction`` tightens ``conv_thr`` to 1e-10 because an
        anisotropy is a difference of total energies at the micro-Rydberg
        level, so a calculator's 1e-6 must not undo it. Naming it in the call
        still reaches it, which is the difference between a default and a
        refusal.
        """
        return {name: value
                for name, value in self._defaults_for(run_scf).items()
                if name not in withheld}

    def _defaults_for(self, func, options=None, exclude=frozenset()) -> dict:
        """This calculator's shared options that ``func`` actually names.

        Filtering is strictly by named parameter. A ``**kwargs`` in the
        signature is *not* taken as permission to pass everything: several
        entry points forward theirs to the Sternheimer solvers, which have no
        ``nbnd`` and would raise on one.

        ``exclude`` drops names whose *meaning* differs in the callee even
        though the spelling matches -- :data:`SCF_ONLY_OPTIONS`, which is where
        the reasoning lives. It applies to the calculator's defaults only:
        anything the caller passed in ``options`` was written at the call site
        and is honoured.

        A name whose meaning is the same and whose *value* the callee chose for
        itself is dropped too, and :func:`_callee_chose` is where that reasoning
        lives. The two rules are separate because they fail separately: a
        meaning collision has to be listed by hand, where a value the callee
        chose is on its own signature and can be read off it.
        """
        parameters = inspect.signature(func).parameters
        shared = {name: value for name, value in self.defaults.items()
                  if name in parameters and name not in exclude
                  and name not in SETUP_ONLY_OPTIONS
                  and not _yields_to_callee(parameters, name, value)}
        return {**shared, **(options or {})}

    def _call_options(self, func, result: SCFResult, options,
                      exclude=frozenset()) -> dict:
        """``_defaults_for`` plus the pieces of the mixed state ``func`` needs.

        This is the method that makes the error class go away: ``ns`` under a
        Hubbard ``U``, ``tau`` under a meta-GGA and ``becsum`` for a PAW
        dataset are properties of the *states*, cannot be rebuilt from the
        density, and are supplied together or not at all.
        """
        parameters = inspect.signature(func).parameters
        merged = self._defaults_for(func, options, exclude=exclude)
        for name, attribute in _STATE_ARGUMENTS.items():
            if name in parameters and name not in merged:
                value = getattr(result, attribute, None)
                if value is not None and not (name == "becsum" and not value):
                    merged[name] = value
        return merged

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        structure = self.system.structure
        formula = "".join(
            f"{name}{count}" if count > 1 else name
            for name, count in _formula(structure)
        )
        parts = [
            f"{formula} ({structure.nat} atoms)",
            f"nspin={self.system.nspin}",
            f"ecutwfc={self.system.ecutwfc:g} Ry",
            f"{self.system.kpoints.nk} k-points",
        ]
        if self._scf is None:
            parts.append("no SCF yet")
        elif self._scf.converged:
            parts.append(f"E = {self._scf.total_energy:.8f} Ry")
        else:
            parts.append("SCF not converged")
        return f"<Calculator: {', '.join(parts)}>"


def _formula(structure) -> list[tuple[str, int]]:
    """The species in order of first appearance, with their counts."""
    counts: dict[str, int] = {}
    for index in structure.types:
        name = structure.species[int(index)].name
        counts[name] = counts.get(name, 0) + 1
    return list(counts.items())


def _resolve_pseudos(system: System, pseudos, pseudo_dir) -> tuple:
    """Explicit pseudopotentials, or the files the input file named.

    The resolution order is the CLI's: what was passed, else ``pseudo_dir``,
    and a missing file is reported with the species that asked for it rather
    than as a bare path.
    """
    if pseudos is not None:
        pseudos = tuple(pseudos)
        if len(pseudos) != system.structure.ntyp:
            raise ValueError(
                f"{len(pseudos)} pseudopotentials for "
                f"{system.structure.ntyp} species; they are matched by "
                "position, in the order of the ATOMIC_SPECIES card"
            )
        for pseudo in pseudos:
            if not isinstance(pseudo, Pseudopotential):
                raise TypeError(
                    "pseudos must be Pseudopotential objects (defumat.pseudo."
                    f"read_upf), not {type(pseudo).__name__}"
                )
        return pseudos

    if pseudo_dir is None:
        raise ValueError(
            "no pseudopotentials given and no pseudo_dir to find them in. "
            "Pass pseudo_dir=..., or use Calculator.from_file(), which takes "
            "it from the input file's own directory"
        )

    # **A sequence of directories, tried in order, because the input file names
    # one too.** ``pw.x`` reads ``&control``'s ``pseudo_dir`` and nothing here
    # did: an input whose pseudopotentials live only there -- which is how QE's
    # own test-suite is arranged, ``pseudo_dir = '../../pseudo'`` -- raised on a
    # file ``pw.x`` runs, and the message pointed at this function's keyword
    # rather than at the input variable that already answers it. The resolution
    # is per *file* rather than per directory, so a ``pseudo_dir`` that exists
    # but does not hold this species still falls back rather than failing, and
    # the error names every place that was looked in.
    directories = [Path(one) for one in (
        (pseudo_dir,) if isinstance(pseudo_dir, (str, Path)) else tuple(pseudo_dir)
    )]
    loaded = []
    for species in system.structure.species:
        for directory in directories:
            path = directory / species.pseudo_file
            if path.is_file():
                loaded.append(read_upf(path))
                break
        else:
            looked = ", ".join(str(one / species.pseudo_file) for one in directories)
            raise FileNotFoundError(
                f"{species.name}: {looked} does not exist. The ATOMIC_SPECIES "
                f"card names {species.pseudo_file!r}; point pseudo_dir at the "
                "directory holding it"
            )
    return tuple(loaded)


def _has_checkpoint(checkpoint_dir) -> bool:
    """Whether :func:`~defumat.scf.driver.run_scf` would resume from here.

    The same test the driver makes, spelled the same way, so that the two
    cannot drift apart: what decides a resume is the state file, and the mixer
    history beside it is picked up only if the state was.
    """
    return (checkpoint_dir is not None
            and (Path(checkpoint_dir) / SCF_CHECKPOINT).exists())


def _relax_entry_point(variable_cell: bool):
    """``run_vc_relax`` or ``run_relax``, which decides the option signature."""
    if variable_cell:
        from defumat.workflows.vc_relax import run_vc_relax as run
    else:
        from defumat.workflows.relax import run_relax as run
    return run


def _same_options(new: dict, old: dict | None) -> bool:
    """Whether two SCF option sets are the same run.

    Options can hold arrays -- ``starting_density`` above all -- whose ``==``
    is elementwise and whose truth value raises. A comparison that cannot be
    made is treated as a mismatch, so the failure mode is recomputing rather
    than returning the wrong cache.
    """
    if old is None or set(new) != set(old):
        return False
    for key, value in new.items():
        other = old[key]
        if value is other:
            continue
        try:
            if isinstance(value, np.ndarray) or hasattr(value, "shape"):
                return False
            if value != other:
                return False
        except Exception:  # pragma: no cover - defensive, see the docstring
            return False
    return True


def _spinor_leg(spinor):
    """``(system, pseudos, defaults)`` of the force theorem's one-shot leg.

    Accepts a :class:`Calculator` -- the ordinary way, built from the ``nscf``
    input -- or the pair directly, for a caller already holding both.

    ``defaults`` is the part of the second leg's own options that describes the
    second leg's *system* rather than the first's, which is
    :data:`_SPINOR_LEG_OPTIONS` and is a band count. The collinear calculator
    withholds its own, so without this a ``nbnd`` set on the spinor calculator
    at construction would reach nothing at all: it is not on
    ``spinor.system``, and only the system and the pseudopotentials cross.
    """
    if isinstance(spinor, Calculator):
        return spinor.system, spinor.pseudos, {
            name: value for name, value in spinor.defaults.items()
            if name in _SPINOR_LEG_OPTIONS
        }
    system, pseudos = spinor
    return system, pseudos, {}
