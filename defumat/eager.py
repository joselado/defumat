"""One compiled program per *structure* for a closure that is called eagerly.

A loop over k written as ``map_k(lambda ik: ..., indices)`` and called at the
top level, outside any ``jit``, is an eager ``lax.map``: JAX traces the closure,
binds a ``scan`` with the traced body, and compiles that ``scan`` on first use.
The compiled program is cached on the body's jaxpr *object*, and a closure built
afresh at every call traces to a fresh object, so the same program is compiled
again at every call -- and with the persistent cache on, every one of those is a
cache hit that maps a new executable into the process. Measured on zincblende
AlAs (``tests/data/qe/alas-berry.in``) a second ``get_dielectric_tensor()``
compiled 94 programs, one Sternheimer loop and one response density per field
direction per iteration, and took 23.2 s with the cache on and 67.8 s with it
off (``PERFORMANCE.md``, "The response stack compiled its k loops again at
every iteration").

Where the closure is built from arrays alone the cure is a module-level ``jit``
with the arrays as arguments and the layout static, which is what
``ultracell_matrix`` and ``topology.states._pair_overlaps`` do. The response
stack's closures carry a *callable* -- a perturbation, a bare-plus-induced
potential -- so there is no static argument to key on, and :func:`compiled` keys
on the traced structure instead: the closure is traced to a jaxpr with every
array it closes over hoisted to an argument, the jaxpr's printed form is hashed,
and one ``jit`` of that jaxpr is kept per hash and handed the hoisted arrays.
Two calls whose closures differ only in the arrays they hold share one program.
:func:`compiled_function` is the same kept program handed back as a callable,
for a loop that calls one closure many times and should trace it once
(the torque's chunked derivative).

**When it engages, and why each refusal is there.**

* Only at the top level (``trace_state_clean``, a private JAX predicate). Under
  an outer ``jit`` the caller's compile already covers the loop, and under
  ``jvp``, ``grad`` or ``vmap`` the stored program would be *differentiated*,
  which reads ``custom_jvp`` rules the printed form names but does not show. A
  caller that wants a derivative cached wraps the whole ``jax.jvp`` instead, so
  the stored jaxpr is already the derivative and is evaluated primal only
  (``Sternheimer.response_density``).
* A nested jaxpr's own constants are *hashed into the key*. A ``jit`` that
  closes over an array, or builds one from a static argument
  (``pseudo.projectors._with_origin_tangent``'s index array), keeps it inside its
  ``ClosedJaxpr`` rather than hoisting it, and the printed form shows its type
  and not its value, so two calls with different arrays there would print
  alike. Their bytes go into the key, and past ``NESTED_LIMIT`` of them the call
  is evaluated as it was before.
* Not when a callback primitive is present, whose host function is named in the
  printed form and not shown.
* Not when ``fn`` needs the *value* of something that depends on its arguments
  (``float`` of it, ``np.asarray`` of it, a branch on it): an eager call allowed
  that, ``map_axis`` with a single entry calling its body on concrete values, and
  a trace cannot, so such a call is ``fn(*args)`` as before. Where only
  arithmetic on constants needs a value, a second trace evaluates it
  (``jax.ensure_compile_time_eval``), as it was evaluated eagerly.

The cache is module-level mutable state that outlives a call, and unlike the
package's memos of host-side tables it holds executables. It holds the open
jaxpr and its compiled program, never the arrays, and it is bounded
(``CACHE_SIZE``) because every entry pins an executable.
"""

from __future__ import annotations

import collections
import hashlib
import warnings

import jax
import numpy as np
from jax._src import core as _core  # trace_state_clean has no public spelling

#: Programs kept; the least recently used is dropped past it, with one warning,
#: because a process cycling through more structures than this would compile at
#: every call again with nothing else to say so. An entry is an executable handle.
CACHE_SIZE = 128

#: Bytes of nested constants hashed into a key before the call is refused.
NESTED_LIMIT = 1 << 24

_PROGRAMS: collections.OrderedDict = collections.OrderedDict()

_NEEDS_VALUES = (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError,
                 jax.errors.TracerBoolConversionError, jax.errors.TracerIntegerConversionError)


def compiled(fn, *args):
    """``fn(*args)``, compiled once for every call that traces to the same jaxpr.

    ``args`` is a pytree of arrays and ``fn`` returns one. At the top level the
    result is the same computation as ``fn(*args)`` run through ``jit``; under a
    trace, or where a guard in the module docstring refuses, it is ``fn(*args)``
    itself, evaluated as it was before this function existed.
    """
    if not _core.trace_state_clean():
        return fn(*args)
    traced = _trace(fn, args)
    if traced is None:
        # ``fn`` reads a value that depends on its arguments, which an eager
        # call allows and a trace does not: ``map_axis`` with one entry calls
        # its body on concrete values.
        return fn(*args)
    closed, shape = traced
    flat, _ = jax.tree_util.tree_flatten(args)
    out_tree = jax.tree_util.tree_structure(shape)
    jaxpr, consts = closed.jaxpr, list(closed.consts)
    nested = _nested_digest(jaxpr)
    if nested is None:
        # The traced jaxpr evaluated eagerly is what calling ``fn`` would have
        # done, and it does not run ``fn``'s Python a second time.
        return jax.tree_util.tree_unflatten(
            out_tree, jax.core.eval_jaxpr(jaxpr, consts, *flat))
    program = _kept(jaxpr, nested, flat, consts)
    return jax.tree_util.tree_unflatten(out_tree, program(consts, flat))


def compiled_function(fn, *args):
    """``fn`` as a callable, traced once here at ``args`` and kept as :func:`compiled` keeps it.

    For a Python loop that calls one closure many times with arguments of one
    structure, a derivative taken a k-chunk at a time being the case it was
    written for (``forces.torque``): :func:`compiled` would trace ``fn`` at
    every call, which for the ``value_and_grad`` of a Hamiltonian's build is a
    large part of the call, while this traces it once and returns ``run`` with
    ``run(*args)`` equal to ``fn(*args)``. The program behind ``run`` is the one
    :func:`compiled` keeps under the same key, so a later loop over another
    closure of the same structure, such as one built around a new
    ``Calculation`` with arrays of the same shapes, compiles nothing.

    A call of ``run`` whose arguments differ from ``args`` in structure, shape
    or dtype goes through :func:`compiled`. Where a guard of the module
    docstring refuses, the callable is ``jax.jit(fn)``, the program such a loop
    compiled at every call before this existed.
    """
    if not _core.trace_state_clean():
        return jax.jit(fn)
    traced = _trace(fn, args)
    if traced is None:
        return jax.jit(fn)
    closed, shape = traced
    flat, in_tree = jax.tree_util.tree_flatten(args)
    jaxpr, consts = closed.jaxpr, list(closed.consts)
    nested = _nested_digest(jaxpr)
    if nested is None:
        return jax.jit(fn)
    program = _kept(jaxpr, nested, flat, consts)
    out_tree = jax.tree_util.tree_structure(shape)
    signature = (in_tree, tuple(_aval(x) for x in flat))

    def run(*call_args):
        call_flat, call_tree = jax.tree_util.tree_flatten(call_args)
        if (call_tree, tuple(_aval(x) for x in call_flat)) != signature:
            return compiled(fn, *call_args)
        return jax.tree_util.tree_unflatten(out_tree, program(consts, call_flat))

    return run


def _trace(fn, args):
    """``(closed jaxpr, output shape)`` of ``fn(*args)``, or ``None`` where it needs values."""
    try:
        return jax.make_jaxpr(fn, return_shape=True)(*args)
    except _NEEDS_VALUES:
        try:
            # A setup step such as ``augmentation_dipole``'s
            # ``np.asarray(simpson_weights(...))`` needs the value of arithmetic
            # on constants, which ``make_jaxpr`` stages; evaluated while tracing it
            # is hoisted, as it was when ``fn`` ran eagerly. Not the first try,
            # because it also evaluates a heavy constant subcomputation eagerly
            # (a density from frozen states under a ``jvp`` in ``becsum``), and
            # that compiles its own loops again at every call.
            with jax.ensure_compile_time_eval():
                return jax.make_jaxpr(fn, return_shape=True)(*args)
        except _NEEDS_VALUES:
            return None


def _kept(jaxpr, nested, flat, consts):
    """The program kept for this jaxpr and these argument types, compiled on first use."""
    key = (hashlib.sha256(str(jaxpr).encode()).hexdigest(), nested,
           tuple(_aval(x) for x in flat), tuple(_aval(c) for c in consts))
    program = _PROGRAMS.get(key)
    if program is None:
        program = _program(jaxpr)
        _PROGRAMS[key] = program
        while len(_PROGRAMS) > CACHE_SIZE:
            _PROGRAMS.popitem(last=False)
            _warn_evicted()
    else:
        _PROGRAMS.move_to_end(key)
    return program


def compiled_jvp(fun, primals, tangents):
    """``jax.jvp(fun, primals, tangents)`` through :func:`compiled`.

    The whole derivative is traced and kept, so what is stored is already the
    tangent program and is only ever evaluated. Under a trace it is ``jax.jvp``
    itself. The response stack's top-level ``jvp`` calls (a displacement, a
    strain, a field direction) each build a new closure, which is what this is
    for.
    """
    return compiled(lambda p, t: jax.jvp(fun, p, t), tuple(primals), tuple(tangents))


_EVICTED = []


def _warn_evicted() -> None:
    if not _EVICTED:
        _EVICTED.append(True)
        warnings.warn(
            f"defumat.eager kept {CACHE_SIZE} compiled programs and dropped the "
            "oldest: a closure whose program was dropped compiles again on its "
            "next call, so a loop over more structures than this pays a compile "
            "at every call (raise defumat.eager.CACHE_SIZE)",
            RuntimeWarning, stacklevel=3)


def _program(jaxpr):
    @jax.jit
    def run(consts, flat):
        return jax.core.eval_jaxpr(jaxpr, consts, *flat)
    return run


def _aval(x):
    aval = jax.typeof(x)
    return (tuple(aval.shape), str(aval.dtype), bool(getattr(aval, "weak_type", False)))


def _nested_digest(jaxpr):
    """A hash of every nested constant, or ``None`` where the call cannot be keyed."""
    digest, total = hashlib.sha256(), 0
    for const in _nested_consts(jaxpr):
        if const is None:
            return None
        try:
            value = np.asarray(const)
        except (TypeError, ValueError):  # a key or token array has no bytes to hash
            return None
        total += value.nbytes
        if total > NESTED_LIMIT:
            return None
        digest.update(f"{value.dtype}{value.shape}".encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _nested_consts(jaxpr):
    """The constants nested jaxprs hold, in order; ``None`` marks a callback."""
    for eqn in jaxpr.eqns:
        if "callback" in eqn.primitive.name:
            yield None
            return
        for value in eqn.params.values():
            for sub in _jaxprs(value):
                if isinstance(sub, _core.ClosedJaxpr):
                    yield from sub.consts
                    sub = sub.jaxpr
                yield from _nested_consts(sub)


def _jaxprs(value):
    if isinstance(value, (_core.ClosedJaxpr, _core.Jaxpr)):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _jaxprs(item)


def clear() -> None:
    """Drop every kept program (a test's isolation; ``jax.clear_caches`` frees them)."""
    _PROGRAMS.clear()
    _EVICTED.clear()
