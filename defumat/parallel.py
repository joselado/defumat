"""k-point pools: one calculation split over processes by k-point, ``pw.x -nk``.

``PW/src/c_bands.f90`` and ``sum_band.f90`` walk ``DO ik = 1, nks`` where
``nks`` is the pool's own share of the k-points (``divide_et_impera.f90``),
and the pools meet twice per iteration: ``mp_sum(rho%of_r, inter_pool_comm)``
at ``sum_band.f90:221``, and the broadcast of the mixed density, ``dr2`` and
``conv_elec`` from the root pool after ``mix_rho`` (``electrons.f90:877-879``),
which is what keeps every pool's control flow identical. This module is the
same decomposition for this code, and the SCF driver uses it at exactly those
two places plus the gather of the eigenvalues the occupations need.

**Why processes, and why the streamed store.** Measured on six identical cores
(``PERFORMANCE.md``, "On identical cores the conclusion holds"), a k-point
gains at most 1.33x from four threads at sixteen atoms and 2.16x from six at
sixty-four, so a core spent on another k-point is worth more than a core spent
on a thread, and the pools want to be narrow. A pool is a process pinned to its
own cores; within it, the k-point walk is :mod:`defumat.scf.streaming`'s, which
already visits the store a chunk of rows at a time and accumulates the raw sums
over k before finishing them once (``becsum`` and ``ns`` symmetrised, the
density lifted, augmented and symmetrised). A pool walks only its own rows, the
raw sums are all-reduced, and every pool finishes the same total. Nothing
jitted changes: the Hamiltonians, the eigensolver and the density kernels are
the whole calculation's, called with ``indices``/``rows``, so every pool
compiles the executables a serial run compiles and the Davidson cap is the
whole set's.

**What crosses, per SCF iteration.** An all-reduce of the raw ``becsum``,
smooth-grid density, ``tau`` and ``ns``; a gather of the eigenvalues (and of
the Davidson step counts, for the printed average), since the Fermi level and
the tetrahedra read every k-point and the occupations are then evaluated on the
whole set by every pool; and broadcasts from rank 0 of every value a decision
is taken on (``accuracy``, the final ``converged``, the deadline) and of the
mixed state, so that pools with unequal thread counts, whose floating-point
reductions can differ in the last bit, cannot drift apart. An insulator's
occupations need no eigenvalues from other pools; the gather is kept for both
cases because it is ``nk x nbnd`` numbers and it keeps one code path.

**A pool's store cannot be read as a whole-set store.** :class:`PoolStore`
holds the pool's rows and the global row index of each, and it is deliberately
not an ``ndarray``: every whole-set consumer (``Calculation.becsum``, a force,
a band structure) walks position ``i`` as global row ``i``, which on any pool
but the first would read the wrong k-point's tables and weights and return a
plausible wrong number. Handed a :class:`PoolStore`, those consumers fail
instead; the pool-aware streaming functions are the only ones that unwrap it.

**The transport** is a registry (:func:`register_communicator`): ``serial``,
the identity, and ``jax``, ``jax.distributed`` with gloo collectives on a CPU
(NCCL on a card). ``mpi4py`` would be one more entry; it is not installed on
Triton, and JAX's own collectives carry the GPU case with the same call.
``jax.distributed`` has to be initialised before any JAX computation, so
:func:`start_from_environment` runs at ``import defumat`` and is gated on
``DEFUMAT_POOLS`` alone: gating on ``SLURM_NTASKS`` would make every
multi-task job that imports the package for independent work wait at start-up
for peers that never come.

Environment: ``DEFUMAT_POOLS`` (the pool count), ``DEFUMAT_POOL_RANK`` (else
``SLURM_PROCID``), ``DEFUMAT_COORDINATOR`` (``host:port`` of rank 0; under
Slurm it may be left unset and ``jax.distributed`` finds it),
``DEFUMAT_POOL_TIMEOUT`` (seconds to wait for the peers, 300),
``DEFUMAT_POOL_HEARTBEAT`` (seconds without a heartbeat before a peer counts as
dead, 100), ``DEFUMAT_POOL_INTERFACE`` (the network interface gloo binds to,
``ib0`` on Triton; unset uses the address the hostname resolves to), and
``DEFUMAT_THREADS``, which every pool should set to its width.

**A pool that dies takes the others with it, and does not hang them.**
Measured 2026-09-30, one rank of three leaving with ``os._exit`` before an
all-reduce (``tools/parallel/comm_bench.py --kill-after``): on one machine the
survivors block in the collective until the coordination service misses the
dead rank's heartbeat and aborts them, exit 134 after the 100 s default; across
two Triton nodes under ``srun`` their all-reduce fails in 0.2 s with
``Connection closed by peer`` and ``srun`` cancels the step. Either way the job
ends rather than holding its allocation, and ``DEFUMAT_POOL_HEARTBEAT`` shortens
the first case.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

__all__ = [
    "PoolStore", "Pools", "current_pools", "register_communicator",
    "start_from_environment", "pool_count_from_environment", "poll_stop",
    "stop_latched",
]

_COMMUNICATORS: dict[str, type] = {}
_STARTED = False


def register_communicator(name: str):
    """Register a transport under ``name``; a new one is a class and this decorator."""
    def decorate(cls):
        _COMMUNICATORS[name] = cls
        return cls
    return decorate


def pool_count_from_environment() -> int:
    """``DEFUMAT_POOLS`` as an integer, 1 when it is unset or empty."""
    setting = (os.environ.get("DEFUMAT_POOLS") or "").strip()
    if not setting:
        return 1
    try:
        count = int(setting)
    except ValueError as error:
        raise ValueError(f"DEFUMAT_POOLS={setting!r} is not an integer") from error
    if count < 1:
        raise ValueError(f"DEFUMAT_POOLS={count} must be at least 1")
    return count


def pool_rank_from_environment() -> int:
    """This process's rank: ``DEFUMAT_POOL_RANK``, else ``SLURM_PROCID``, else 0."""
    for name in ("DEFUMAT_POOL_RANK", "SLURM_PROCID"):
        setting = (os.environ.get(name) or "").strip()
        if setting:
            return int(setting)
    return 0


def start_from_environment() -> None:
    """``jax.distributed.initialize`` when ``DEFUMAT_POOLS`` asks for more than one pool.

    Called once from ``defumat/__init__.py``, before anything creates an array.
    A no-op for a single pool, so an ordinary run never touches the
    distributed runtime.
    """
    global _STARTED
    size = pool_count_from_environment()
    if size <= 1 or _STARTED:
        return
    import jax

    rank = pool_rank_from_environment()
    coordinator = (os.environ.get("DEFUMAT_COORDINATOR") or "").strip() or None
    if coordinator is None and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            f"DEFUMAT_POOLS={size} asks for pools, but neither DEFUMAT_COORDINATOR "
            "(host:port of rank 0) nor a Slurm job is there to say where the "
            "peers meet"
        )
    timeout = int((os.environ.get("DEFUMAT_POOL_TIMEOUT") or "300").strip())
    heartbeat = int((os.environ.get("DEFUMAT_POOL_HEARTBEAT") or "100").strip())
    interface = (os.environ.get("DEFUMAT_POOL_INTERFACE") or "").strip()
    if interface:
        _route_gloo_through(interface)
    jax.config.update("jax_cpu_collectives_implementation", "gloo")
    jax.distributed.initialize(
        coordinator_address=coordinator, num_processes=size, process_id=rank,
        initialization_timeout=timeout, heartbeat_timeout_seconds=heartbeat,
    )
    _STARTED = True


def _route_gloo_through(interface: str) -> None:
    """Make gloo's TCP transport bind to ``interface`` rather than the hostname's address.

    ``jax._src.xla_bridge`` builds the CPU collectives with
    ``make_gloo_tcp_collectives(distributed_client=...)`` and passes neither of
    the two keywords that function takes, ``hostname`` and ``interface``, so the
    transport uses whatever the node's name resolves to. On Triton that is the
    Ethernet address (``milan3`` is 10.30.250.3, ``eth0``) while the nodes also
    have ``ib0``: measured 2026-09-30, an 83 MB all-reduce between two nodes took
    868 ms, about 100 MB/s. The keyword is added by wrapping the constructor
    before the CPU client exists, which is the one place it can be given; a name
    the node does not have fails at start-up (``Unable to find address for``),
    not silently.
    """
    import functools

    from jax._src.lib import _jax

    original = _jax.make_gloo_tcp_collectives
    if getattr(original, "_defumat_interface", None) == interface:
        return

    @functools.wraps(original)
    def make(*args, **kwargs):
        kwargs.setdefault("interface", interface)
        return original(*args, **kwargs)

    make._defumat_interface = interface
    _jax.make_gloo_tcp_collectives = make


class PoolStore:
    """One pool's rows of the ``(nspin, nk, nbnd, ndim)`` wavefunction store.

    ``array`` is ``(nspin, len(rows), nbnd, ndim)`` in host memory and
    ``rows`` the global k index of each of its rows; ``nk`` is the whole set's
    count. Not an ``ndarray``, and :meth:`__array__` raises, so that no
    consumer written for the whole set can read it as one (module docstring).
    """

    __slots__ = ("array", "rows", "nk")

    def __init__(self, array: np.ndarray, rows, nk: int):
        self.array = array
        self.rows = np.asarray(rows)
        self.nk = int(nk)
        if self.array.shape[1] != len(self.rows):
            raise ValueError(
                f"a pool store of {self.array.shape[1]} rows was given "
                f"{len(self.rows)} row indices")

    def __array__(self, *args, **kwargs):
        raise TypeError(
            f"this wavefunction store holds k-points {self.rows[0]} to "
            f"{self.rows[-1]} of {self.nk} (one k-point pool's share), so it "
            "cannot be read as the whole set; only the pool-aware streaming "
            "passes in defumat.scf.streaming unwrap it"
        )

    def __repr__(self) -> str:
        return (f"PoolStore(rows {self.rows[0]}..{self.rows[-1]} of {self.nk}, "
                f"array {self.array.shape} {self.array.dtype})")


@register_communicator("serial")
class SerialCommunicator:
    """One process: every collective is the identity."""

    size = 1
    rank = 0

    def allreduce_sum(self, tree):
        return tree

    def allgather(self, array) -> np.ndarray:
        return np.asarray(array)[None]

    def broadcast(self, tree):
        return tree


@register_communicator("jax")
class JaxCommunicator:
    """``jax.distributed`` collectives, one device per process on a mesh axis ``pool``.

    The all-reduce is a jitted sum over a global ``(size, ...)`` array whose
    rows are the processes' partial sums, which XLA lowers to one all-reduce
    (gloo on a CPU). Complex leaves travel as their real view, since the CPU
    collectives are not asked to know about complex dtypes. The gather and the
    broadcast are ``jax.experimental.multihost_utils``'.
    """

    def __init__(self):
        import jax
        from jax.sharding import Mesh, NamedSharding, PartitionSpec

        self.size = jax.process_count()
        self.rank = jax.process_index()
        first = {}
        for device in jax.devices():
            first.setdefault(device.process_index, device)
        mesh = Mesh(np.array([first[i] for i in range(self.size)]), ("pool",))
        self._split = NamedSharding(mesh, PartitionSpec("pool"))
        whole = NamedSharding(mesh, PartitionSpec())
        self._sum = jax.jit(lambda a: a.sum(axis=0), out_shardings=whole)

    def _reduce_leaf(self, leaf):
        import jax
        import jax.numpy as jnp

        local = np.ascontiguousarray(np.asarray(leaf))
        dtype = local.dtype
        if np.iscomplexobj(local):
            local = local.reshape(local.shape + (1,)).view(local.real.dtype)
        stacked = jax.make_array_from_process_local_data(
            self._split, local[None], (self.size,) + local.shape)
        total = np.asarray(self._sum(stacked).addressable_data(0))
        if np.iscomplexobj(np.empty(0, dtype)):
            total = np.ascontiguousarray(total).view(dtype).reshape(total.shape[:-1])
        return jnp.asarray(total)

    def allreduce_sum(self, tree):
        import jax
        return jax.tree_util.tree_map(self._reduce_leaf, tree)

    def allgather(self, array) -> np.ndarray:
        from jax.experimental import multihost_utils
        return np.asarray(multihost_utils.process_allgather(np.asarray(array), tiled=False))

    def broadcast(self, tree):
        import jax
        from jax.experimental import multihost_utils
        host = jax.tree_util.tree_map(np.asarray, tree)
        return jax.tree_util.tree_map(
            np.asarray, multihost_utils.broadcast_one_to_all(host))


@dataclass(frozen=True)
class Pools:
    """The pool layout of this process: how many, which one, and the transport."""

    communicator: object

    @property
    def size(self) -> int:
        return self.communicator.size

    @property
    def rank(self) -> int:
        return self.communicator.rank

    def counts(self, nk: int) -> list[int]:
        """k-points per pool, contiguous blocks, the remainder to the first pools.

        ``divide_et_impera.f90``'s split, with ``kunit = 1``.
        """
        base, extra = divmod(nk, self.size)
        return [base + (1 if pool < extra else 0) for pool in range(self.size)]

    def rows(self, nk: int) -> np.ndarray:
        """The global k indices this pool owns."""
        counts = self.counts(nk)
        start = sum(counts[: self.rank])
        return np.arange(start, start + counts[self.rank])

    def gather_k(self, local, nk: int, axis: int = 1) -> np.ndarray:
        """Every pool's ``local`` rows along ``axis``, in global k order.

        Uneven splits are padded to the largest share for the transport and
        trimmed after it.
        """
        local = np.asarray(local)
        counts = self.counts(nk)
        if self.size == 1:
            return local
        width = max(counts)
        pad = [(0, 0)] * local.ndim
        pad[axis] = (0, width - local.shape[axis])
        gathered = self.communicator.allgather(np.pad(local, pad))
        pieces = [np.take(gathered[pool], np.arange(counts[pool]), axis=axis)
                  for pool in range(self.size)]
        return np.concatenate(pieces, axis=axis)

    def allreduce_sum(self, tree):
        return self.communicator.allreduce_sum(tree)

    def broadcast(self, tree):
        return self.communicator.broadcast(tree)

    def broadcast_scalar(self, value) -> float:
        return float(np.asarray(self.broadcast(np.asarray(value, dtype=np.float64))))

    def broadcast_flag(self, value) -> bool:
        return bool(np.asarray(self.broadcast(np.asarray(bool(value), dtype=np.int32))))


#: The preemption protocol's step id and its answer, process-wide. See
#: :func:`poll_stop`.
_STOP_STEP = 0
_STOP_LATCHED = False


def poll_stop() -> bool:
    """Whether a SIGTERM has reached any pool, answered identically on every pool.

    **A pooled process does not die of SIGTERM, and this is how it stops.**
    ``jax.distributed.initialize`` starts JAX's preemption service, whose
    handler (C++, invisible to :func:`signal.getsignal`) catches the signal,
    tells every other task, and agrees with them on a step at which all of them
    stop: ``multihost_utils.reached_preemption_sync_point(step)`` is True at that
    step on every pool and nowhere else. Measured 2026-09-30 with two pools in
    lockstep through an all-reduce: SIGTERM to either rank, both answered True at
    the same step and exited cleanly; a call costs under a microsecond and never
    blocks, since the protocol runs on the service's own threads.

    Three rules the protocol imposes, all kept here rather than at call sites.
    The step id must pass through every integer, so it is one process-wide
    counter advanced by exactly one call per SCF iteration (``run_scf``'s loop
    top), never reset across the SCF runs of a relaxation. The True is given
    once, so it is **latched**, and every later call returns it too. And a
    single process has no service and no handler, so it answers False and a
    SIGTERM ends it as it always has.

    A job that wants the stop before Slurm's kill asks for the warning early:
    ``#SBATCH --signal=TERM@<seconds>``, with the lead at least two SCF
    iterations plus a checkpoint write, since the agreed step can be the one
    after next.
    """
    global _STOP_STEP, _STOP_LATCHED
    if not _STARTED:
        return False
    if not _STOP_LATCHED:
        from jax.experimental import multihost_utils
        _STOP_LATCHED = bool(multihost_utils.reached_preemption_sync_point(_STOP_STEP))
        _STOP_STEP += 1
    return _STOP_LATCHED


def stop_latched() -> bool:
    """Whether :func:`poll_stop` has answered True; reads the latch without polling."""
    return _STOP_LATCHED


_CURRENT: Pools | None = None


def current_pools() -> Pools:
    """This process's pools: the ``jax`` transport once started, ``serial`` otherwise."""
    global _CURRENT
    if _CURRENT is None:
        name = "jax" if _STARTED else "serial"
        _CURRENT = Pools(_COMMUNICATORS[name]())
    return _CURRENT
