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
which is precisely the tuple :class:`~pypresso.scf.driver.SCFResult` already
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

The state lives here rather than on :class:`~pypresso.system.builder.System`
for two reasons, the second decisive. ``System`` is an ``equinox.Module``
crossing ``jit``/``grad`` boundaries, so a ``pseudos`` field would change the
pytree every compiled path sees and a cached result cannot live on a frozen
module at all. And ``System`` does not *have* the pseudopotentials: a
``system.get_bands()`` would need them as an argument, which is the API this
module exists to shorten. The unit that can compute is ``system + pseudos``,
which is exactly what :class:`~pypresso.scf.driver.Calculation` already takes.
"""

from __future__ import annotations

import dataclasses
import inspect
import sys
from pathlib import Path

import numpy as np

from pypresso.pseudo.upf import Pseudopotential, read_upf
from pypresso.scf.driver import Calculation, SCFResult, run_scf
from pypresso.system.builder import System, build_system, system_from_file
from pypresso.system.kpoints import for_spin

__all__ = ["Calculator", "SHARED_OPTIONS", "SCF_ONLY_OPTIONS"]


#: Options a :class:`Calculator` accepts once, at construction, and forwards to
#: every method whose entry point names them. They are the knobs that describe
#: *this calculator* rather than one quantity computed with it: how many bands,
#: how tightly to converge, how many k-points in flight, which eigensolver.
#:
#: The filtering is by *named parameter*, never by a ``**kwargs`` catch-all --
#: :func:`~pypresso.response.electrostriction.electrostriction` has one, and it
#: forwards to the Sternheimer solvers, which have no ``nbnd``.
SHARED_OPTIONS = frozenset({
    "nbnd",
    "conv_thr",
    "k_batch",
    "diagonalization",
    "mixing_mode",
    "mixing_beta",
    "mixing_fixed_ns",
    "max_iterations",
    "scf_solver",
    "scf_solver_options",
    "verbose",
})

#: The subset of :data:`SHARED_OPTIONS` that describes the **SCF loop** and
#: nothing else, and is therefore *not* forwarded past it.
#:
#: The name is the whole problem: ``max_iterations`` is the SCF's iteration
#: count in :func:`~pypresso.scf.driver.run_scf`, the **self-consistent
#: response's** in :func:`~pypresso.response.efield.dielectric_tensor`
#: (``response/efield.py:322``, whose default comes from
#: :mod:`pypresso.response.mixing`) and the **Dyson fixed point's** in
#: :func:`~pypresso.workflows.run_absorption` (``tddft/dyson.py``'s 500). Three
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
    "mixing_mode",
    "mixing_beta",
    "mixing_fixed_ns",
    "max_iterations",
    "scf_solver",
    "scf_solver_options",
})


#: Parameter name -> the :class:`~pypresso.scf.driver.SCFResult` attribute that
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


class Calculator:
    """A system, its pseudopotentials, and every calculation they support.

    Args:
        system: the :class:`~pypresso.system.builder.System` to compute on.
        pseudos: its pseudopotentials, in species order. ``None`` -- the usual
            case -- loads them from the names the input file gave, resolved
            against ``pseudo_dir``.
        pseudo_dir: where to find those files. Defaults to the directory the
            input file came from, matching what the CLI already does.
        basis: a prebuilt :class:`~pypresso.basis.builder.Basis`, for the rare
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
            (``diagonalization``, ``k_batch``), which do not describe a *run*
            but the ``Calculation`` every run goes through, so giving one per
            call rebuilds it and it stays.

    A minimal script::

        from pypresso import Calculator

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
        unknown = set(defaults) - SHARED_OPTIONS
        if unknown:
            raise TypeError(
                f"unknown calculator option(s) {sorted(unknown)}. A keyword given "
                "here applies to every method that takes it, so only the run-wide "
                f"ones are accepted: {sorted(SHARED_OPTIONS)}. Anything else "
                "belongs on the method it configures"
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
        self._relax_variable_cell = False
        self._strain_response = None
        #: A converged state from another calculator, handed to the first SCF
        #: as ``starting_from``. Not a cache -- a starting point (P23).
        self._seed = None
        #: The NSCF states the last :meth:`get_dos` or :meth:`get_pdos` ran on.
        #: Those entry points return a pair; the second half is kept here so
        #: that the method can return the quantity that was asked for.
        self.dos_states = None
        self.pdos_states = None

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path, pseudo_dir=None, **defaults) -> "Calculator":
        """Read a ``pw.x`` input file and load the pseudopotentials it names.

        The one line that replaces four. ``pseudo_dir`` defaults to the input
        file's own directory.
        """
        path = Path(path)
        system = system_from_file(path)
        if pseudo_dir is None:
            pseudo_dir = path.parent
        return cls(system, pseudo_dir=pseudo_dir, **defaults)

    @classmethod
    def from_text(cls, text: str, pseudo_dir, **defaults) -> "Calculator":
        """The same, from the text of an input file rather than a path.

        ``pseudo_dir`` is required here: there is no file to take it from.
        """
        from pypresso.io.pwin import parse_pw_input

        return cls(build_system(parse_pw_input(text)), pseudo_dir=pseudo_dir,
                   **defaults)

    # ------------------------------------------------------------------
    # the fixed setup, built once and on demand
    # ------------------------------------------------------------------

    @property
    def calculation(self) -> Calculation:
        """The :class:`~pypresso.scf.driver.Calculation` this calculator uses.

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
            )
        return self._calculation

    #: The options that define a :class:`~pypresso.scf.driver.Calculation`
    #: rather than one run over it. Given per call, they have to rebuild it.
    SETUP_OPTIONS = ("diagonalization", "k_batch")

    def _adopt(self, options) -> None:
        """Take a per-call setup option as this calculator's own.

        Rebuilding is the correct semantics rather than a workaround: these two
        decide *which* ``Calculation`` exists, so a call that changes one is
        asking for a different setup, and everything cached under the old one
        is about to be replaced anyway.
        """
        changed = {name: options[name] for name in self.SETUP_OPTIONS
                   if name in options and options[name] != self.defaults.get(
                       name, "default" if name == "k_batch" else None)}
        if changed:
            self.defaults.update(changed)
            self._calculation = None
            self._strain_response = None

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
        """
        merged = {**self._defaults_for(run_scf), **options}
        if self._seed is not None:
            merged.setdefault("starting_from", self._seed)
        if self._scf is not None and _same_options(merged, self._scf_options):
            return self._scf
        # ``diagonalization`` and ``k_batch`` are *not* arguments of the SCF:
        # they are what a ``Calculation`` is built with, and ``run_scf``
        # documents that it ignores them when handed one. Passing them per call
        # would otherwise be a silent no-op that still counted as a cache miss
        # -- the same run, again, under a different name. They rebuild it.
        self._adopt(options)
        self._scf = run_scf(self.system, self.pseudos,
                            calculation=self.calculation, **merged)
        self._scf_options = merged
        # An SCF makes every response built on the previous one stale.
        self._strain_response = None
        return self._scf

    def _ground_state(self, quantity: str) -> SCFResult:
        """The cached ground state, running one first if there is none."""
        if self._scf is None:
            if self.announce:
                conv = self.defaults.get("conv_thr", 1.0e-6)
                print(
                    f"[pypresso] {quantity}: no ground state cached, running the "
                    f"SCF first (conv_thr = {conv:g}). Call get_scf() to do this "
                    "explicitly.",
                    file=sys.stderr,
                )
            self.get_scf()
        if not self._scf.converged:
            raise ValueError(
                f"{quantity} needs a converged ground state and the SCF stopped "
                f"at an accuracy of {self._scf.accuracy:g} Ry after "
                f"{self._scf.iterations} iterations. Rerun get_scf() with a "
                "looser conv_thr, more max_iterations or a different mixing "
                "before reading a derived quantity off it"
            )
        return self._scf

    # ------------------------------------------------------------------
    # band structure, densities of states
    # ------------------------------------------------------------------

    def get_bands(self, kpoints=None, **options):
        """Diagonalise on a k-path at the converged density.

        The Fermi level and the HOMO come from the cached ground state unless
        given, so the band plot has its zero without being told.
        """
        from pypresso.workflows.bands import run_bands

        result = self._ground_state("a band structure")
        kw = self._call_options(run_bands, result, options)
        kw.setdefault("fermi_energy", result.fermi_energy)
        kw.setdefault("homo", result.homo)
        return run_bands(self.system, self.pseudos, result.density,
                         kpoints=kpoints, **kw)

    def get_nscf(self, kpoints=None, **options):
        """Diagonalise on a k-grid at the converged density, and occupy it."""
        from pypresso.workflows.nscf import run_nscf

        result = self._ground_state("an NSCF run")
        return run_nscf(self.system, self.pseudos, result.density,
                        kpoints=kpoints,
                        **self._call_options(run_nscf, result, options))

    def get_dos(self, grid=None, **options):
        """The density of states, on a denser grid than the SCF's if asked.

        Returns the :class:`~pypresso.workflows.dos.DensityOfStates`; the NSCF
        states it was integrated from are left on :attr:`dos_states`.
        """
        from pypresso.workflows.dos import run_dos

        result = self._ground_state("a density of states")
        dos, states = run_dos(self.system, self.pseudos, result.density,
                              grid=grid,
                              **self._call_options(run_dos, result, options))
        self.dos_states = states
        return dos

    def get_pdos(self, grid=None, **options):
        """The projected density of states, with Löwdin charges and spilling.

        Returns the :class:`~pypresso.workflows.pdos.ProjectedDOS`; its NSCF
        states are left on :attr:`pdos_states`.
        """
        from pypresso.workflows.pdos import run_pdos

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
        from pypresso.forces import compute_forces

        result = self._ground_state("the forces")
        return compute_forces(self.calculation, result, method=method)

    def get_stress(self, method=None, terms: bool = False):
        """The stress tensor.

        An SCF run with ``tstress`` has already computed it, and that result is
        returned rather than differentiated a second time.
        """
        from pypresso.stress import compute_stress

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
        """
        if variable_cell:
            from pypresso.workflows.vc_relax import run_vc_relax as run
        else:
            from pypresso.workflows.relax import run_relax as run

        self._relax = run(self.system, self.pseudos,
                          calculation=self.calculation,
                          **self._defaults_for(run, options))
        self._relax_variable_cell = variable_cell
        return self._relax

    def relaxed(self, variable_cell: bool = False, **options) -> "Calculator":
        """A new calculator at the relaxed geometry, with its SCF cached.

        The endpoint of :meth:`get_relax` as an object one can go on computing
        with::

            phonons = calc.relaxed().get_phonons()

        A relaxation already converged an SCF at its final geometry, so that
        result is carried across rather than recomputed.
        """
        if self._relax is None or self._relax_variable_cell != variable_cell:
            self.get_relax(variable_cell=variable_cell, **options)
        result = self._relax
        if not result.converged:
            raise ValueError(
                "the relaxation did not converge, so there is no relaxed "
                "geometry to build a calculator on. Read get_relax()'s own "
                "result to see how far it got"
            )
        return self._derived(result.system, scf=result.scf)

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
        from pypresso.response.velocity import band_velocities

        result = self._ground_state("the band velocities")
        return band_velocities(
            self.calculation, result, kpoints=kpoints,
            **self._defaults_for(band_velocities, options,
                                 exclude=SCF_ONLY_OPTIONS),
        )

    def get_dielectric_tensor(self, **options):
        """``epsilon_infinity`` by the Sternheimer route -- no empty states."""
        from pypresso.response.efield import dielectric_tensor

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
        from pypresso.response.phonon import dynamical_matrix

        result = self._ground_state("the dynamical matrix")
        return dynamical_matrix(
            self.calculation, result.wavefunctions, result.eigenvalues,
            result.density, result.becsum,
            **self._defaults_for(dynamical_matrix, options,
                                 exclude=SCF_ONLY_OPTIONS),
        )

    def get_raman_tensors(self, **options):
        """``d(epsilon)/d(tau)``: the Raman tensor of each atom."""
        from pypresso.response.nonlinear import raman_tensors

        result = self._ground_state("the Raman tensors")
        return raman_tensors(
            self.calculation, result,
            **self._defaults_for(raman_tensors, options,
                                 exclude=SCF_ONLY_OPTIONS),
        )

    def get_vibrational_spectrum(self, **options):
        """Per-mode Raman and infrared activities -- what a spectrum plots."""
        from pypresso.response.spectra import vibrational_spectrum

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
        """
        from pypresso.response.strain import strain_response

        result = self._ground_state("the strain response")
        if self._strain_response is None or options:
            self._strain_response = strain_response(
                self.calculation, result.wavefunctions, result.eigenvalues,
                result.density, result.becsum,
                **self._defaults_for(strain_response, options,
                                     exclude=SCF_ONLY_OPTIONS),
            )
        return self._strain_response

    def get_elastic_constants(self, **options):
        """``C_ijkl``, the stress differentiated along the strain response."""
        from pypresso.response.elastic import elastic_constants

        result = self._ground_state("the elastic constants")
        response = self.get_strain_response()
        return elastic_constants(
            self.calculation, result.wavefunctions, result.eigenvalues,
            result.density, response,
            **self._defaults_for(elastic_constants, options,
                                 exclude=SCF_ONLY_OPTIONS),
        )

    def get_electrostriction(self, **options):
        """``d(chi)/d(strain)`` and the four electrostriction tensors."""
        from pypresso.response.electrostriction import electrostriction

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
        from pypresso.workflows.tddft import run_absorption

        result = self._ground_state("an absorption spectrum")
        return run_absorption(
            self.system, self.pseudos, result.density, frequencies,
            **self._call_options(run_absorption, result, options,
                                 exclude=SCF_ONLY_OPTIONS)
        )

    # ------------------------------------------------------------------
    # topology
    # ------------------------------------------------------------------

    def get_berry_curvature(self, **options):
        """The Berry curvature on a plane mesh; its integral is the Chern
        number, which the result carries as ``.chern_number``."""
        from pypresso.workflows.topology import run_berry_curvature

        result = self._ground_state("the Berry curvature")
        return run_berry_curvature(self.system, self.pseudos, result.density,
                                   **self._call_options(run_berry_curvature,
                                                        result, options))

    def get_chern(self, **options) -> float:
        """The Chern number of one plane -- an exact integer on any mesh."""
        return self.get_berry_curvature(**options).chern_number

    def get_z2(self, **options):
        """The 2D Z2 invariant of one plane, by Wilson loops or Fu-Kane parity."""
        from pypresso.workflows.topology import run_z2

        result = self._ground_state("the Z2 invariant")
        return run_z2(self.system, self.pseudos, result.density,
                      **self._call_options(run_z2, result, options))

    def get_z2_3d(self, **options):
        """The four Z2 indices of a three-dimensional crystal."""
        from pypresso.workflows.topology import run_z2_3d

        result = self._ground_state("the 3D Z2 invariants")
        return run_z2_3d(self.system, self.pseudos, result.density,
                         **self._call_options(run_z2_3d, result, options))

    # ------------------------------------------------------------------
    # spin spirals
    # ------------------------------------------------------------------

    def get_spiral_scan(self, wavevectors, **options):
        """One SCF per spiral wavevector, sharing what does not depend on ``q``."""
        from pypresso.workflows.spiral import run_spiral_scan

        # ``run_spiral_scan`` takes its SCF options through a ``**kwargs`` that
        # forwards to ``run_scf``; the strict filter cannot see that, so the
        # shared ones are matched against ``run_scf``'s own signature instead.
        return run_spiral_scan(self.system, self.pseudos, wavevectors,
                               **{**self._defaults_for(run_scf), **options})

    def get_spiral_relaxation(self, **options):
        """Relax the spiral wavevector itself: ``dE/dq`` downhill by BFGS."""
        from pypresso.workflows.spiral import relax_spiral_q

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
        return self._derived(system, seed=True)

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
        (:func:`pypresso.system.kpoints.for_spin`), because every ``KPoints``
        constructor applies the spin degeneracy unconditionally and a polarized
        run wants it halved -- ``setup.f90`` applies ``degspin`` only in its LDA
        branch. Substituting a raw ``KPoints.automatic`` here used to skip that
        step, which counts every electron twice and does not fail: the Fermi
        level moves and the run integrates to the right electron count at the
        wrong energy, on exactly the comparison the paragraph above recommends.
        A k-set that has already been normalised -- one from
        :func:`~pypresso.workflows.nscf.denser_grid`, say -- is left alone, the
        division being idempotent through ``KPoints.spin_normalized``.
        """
        return self._derived(
            dataclasses.replace(
                self.system, kpoints=for_spin(kpoints, self.system.nspin)
            ),
            seed=True,
        )

    def with_spin(self, nspin=None, **options) -> "Calculator":
        """A calculator in another spin regime, warm-started from this one.

        The three regimes are three ways of writing the same ``(n, m)``, so a
        converged state promotes into the target's variables rather than being
        thrown away (P23): a collinear result promoted into a noncollinear run
        whose magnetization only has to be rotated converges in one iteration
        instead of twenty-five.
        """
        system = self.system.with_spin(nspin, **options)
        return self._derived(system, seed=True)

    def _derived(self, system: System, *, seed: bool = False, scf=None) -> "Calculator":
        """A calculator on ``system``, sharing this one's pseudos and options."""
        derived = Calculator(system, self.pseudos, announce=self.announce,
                             **self.defaults)
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
        return derived

    # ------------------------------------------------------------------
    # plumbing
    # ------------------------------------------------------------------

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
        """
        parameters = inspect.signature(func).parameters
        shared = {name: value for name, value in self.defaults.items()
                  if name in parameters and name not in exclude}
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
                    "pseudos must be Pseudopotential objects (pypresso.pseudo."
                    f"read_upf), not {type(pseudo).__name__}"
                )
        return pseudos

    if pseudo_dir is None:
        raise ValueError(
            "no pseudopotentials given and no pseudo_dir to find them in. "
            "Pass pseudo_dir=..., or use Calculator.from_file(), which takes "
            "it from the input file's own directory"
        )

    directory = Path(pseudo_dir)
    loaded = []
    for species in system.structure.species:
        path = directory / species.pseudo_file
        if not path.is_file():
            raise FileNotFoundError(
                f"{species.name}: {path} does not exist. The ATOMIC_SPECIES "
                f"card names {species.pseudo_file!r}; point pseudo_dir at the "
                "directory holding it"
            )
        loaded.append(read_upf(path))
    return tuple(loaded)


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
