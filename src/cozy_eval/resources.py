"""One compute budget for the whole library, resolved once at import.

An eval run is a batch job that shares its machine — a laptop, a shared build
box, a pod that is also serving. Unbounded is the wrong default everywhere: left
alone, numpy/BLAS, OpenMP, torch, OpenCV and ffmpeg each size their own pool
from the visible core count, so one ``score`` of a handful of clips can take a
32-core box to ~200 threads. Nothing here is faster for it — the work is
embarrassingly parallel across CLIPS, and the per-clip kernels stop scaling
long before the core count.

So the budget is a first-class value, not a local workaround:

* ONE knob, ``COZY_EVAL_THREADS``, a positive integer.
* Default ``min(4, cpu_count)`` — modest, everywhere, including big pods. A
  dedicated pod raises it explicitly (``COZY_EVAL_THREADS=32``); the library
  never takes the machine because it happened to find it.
* An externally-set ``OMP_NUM_THREADS`` is the caller's choice and wins over the
  default (we never overwrite it).
* Everything downstream is derived from that one number: BLAS/OpenMP pool
  sizes, torch intra- and inter-op threads, ``cv2.setNumThreads``, the worker
  count of every process pool, and the ``-threads`` / ``n_threads`` of every
  ffmpeg and libvmaf subprocess. A child process or subprocess that did not
  inherit the budget is the bug this module exists to prevent.

Order matters: the pool-size environment variables must be set BEFORE numpy or
torch is imported, because those pools are created on first use and never
resized. :mod:`cozy_eval` therefore calls :func:`configure` as its first
statement.

The resolved budget is logged once, at INFO on the ``cozy_eval`` logger
(``cozy-eval: 4 compute threads, 2 workers``), so a run always says what it took.
"""

from __future__ import annotations

import logging
import os
import sys

import msgspec

#: The default cap: modest on any machine, on purpose.
DEFAULT_MAX_THREADS = 4

#: The one knob.
THREADS_ENV = "COZY_EVAL_THREADS"

#: Pool sizes read by the native libraries under numpy/scipy/torch, at their
#: FIRST use. Set only if the caller has not set them.
POOL_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_log = logging.getLogger("cozy_eval")


class ComputeBudget(msgspec.Struct, frozen=True, kw_only=True):
    """How much of the machine this process may use, and where that came from."""

    threads: int
    workers: int
    source: str

    @property
    def worker_threads(self) -> int:
        """Threads for ONE pool worker, so ``workers * worker_threads <= threads``."""
        return max(1, self.threads // max(1, self.workers))

    def describe(self) -> str:
        return (f"cozy-eval: {self.threads} compute thread{'s' * (self.threads != 1)}, "
                f"{self.workers} worker{'s' * (self.workers != 1)} ({self.source})")


def cpu_count() -> int:
    """Cores this process may actually run on (affinity/cgroup aware)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:                                      # pragma: no cover
        return max(1, os.cpu_count() or 1)


def _positive_int(raw: str | None) -> int | None:
    try:
        n = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def resolve_budget(threads: int | None = None) -> ComputeBudget:
    """The budget for this process: explicit > ``COZY_EVAL_THREADS`` > caller's
    ``OMP_NUM_THREADS`` > ``min(4, cpu_count)``."""
    if (n := _positive_int(threads)) is not None:
        source = "explicit"
    elif (n := _positive_int(os.environ.get(THREADS_ENV))) is not None:
        source = THREADS_ENV
    elif (n := _positive_int(os.environ.get("OMP_NUM_THREADS"))) is not None:
        source = "OMP_NUM_THREADS"        # the caller's choice; never overridden
    else:
        n, source = min(DEFAULT_MAX_THREADS, cpu_count()), "default"
    n = min(n, cpu_count())
    return ComputeBudget(threads=n, workers=max(1, n // 2), source=source)


_active: ComputeBudget | None = None
_capped: set[str] = set()


def configure(threads: int | None = None, *, force: bool = False,
              announce: bool = True) -> ComputeBudget:
    """Resolve the budget and apply it. Idempotent; the first call wins.

    ``force`` re-resolves and re-applies — for a pool worker, which must be
    pinned to its own share of the parent's budget rather than inherit the whole
    thing.
    """
    global _active                      # noqa: PLW0603 — process-wide by nature
    if _active is not None and not force:
        return _active
    budget = resolve_budget(threads)
    # An explicit budget owns every pool variable, including ones the caller
    # set. A default one only fills in the blanks — the caller's own
    # OMP_NUM_THREADS is a decision, and an empty value is not one.
    owns_env = force or budget.source in ("explicit", THREADS_ENV)
    for var in POOL_ENV:
        if owns_env or _positive_int(os.environ.get(var)) is None:
            os.environ[var] = str(budget.threads)
    os.environ[THREADS_ENV] = str(budget.threads)
    _capped.clear()
    _active = budget
    apply_runtime_caps()
    _log.log(logging.INFO if announce else logging.DEBUG, "%s", budget.describe())
    return budget


def active() -> ComputeBudget:
    """The budget in force, configuring it on first ask."""
    return _active or configure()


def apply_runtime_caps() -> None:
    """Cap the in-process libraries that ignore the environment once imported.

    Cheap and idempotent. Called at configure time and again from
    :func:`cozy_eval.device.resolve_device` and the OpenCV entry points, because
    torch and cv2 are imported lazily — long after the environment was set.
    """
    n = active().threads
    if (torch := sys.modules.get("torch")) is not None and "torch" not in _capped:
        _capped.add("torch")
        if torch.get_num_threads() > n:
            torch.set_num_threads(n)
        try:
            if torch.get_num_interop_threads() > n:
                torch.set_num_interop_threads(n)
        except RuntimeError:
            pass          # inter-op pool already started: intra-op cap still holds
    if (cv2 := sys.modules.get("cv2")) is not None and "cv2" not in _capped:
        _capped.add("cv2")
        cv2.setNumThreads(n)


def opencv():
    """``cv2``, capped. Import OpenCV through here and nowhere else: a bare
    import leaves it sized to the host core count, and OpenCV's pool ignores
    ``OMP_NUM_THREADS`` on most wheels."""
    import cv2

    apply_runtime_caps()
    return cv2


def ffmpeg_thread_args() -> list[str]:
    """``-threads N`` for any ffmpeg/ffprobe invocation. An ffmpeg started
    without it sizes itself from the host core count and blows the budget."""
    return ["-threads", str(active().threads)]


def worker_count(requested: int | None = None) -> int:
    """Pool size: the caller's explicit ask, else the budget's."""
    return _positive_int(requested) or active().workers


def pool_worker_init(threads: int) -> None:
    """Process-pool initializer: pin one worker to its share of the budget.

    Quiet: the parent already announced the run's budget, and N workers
    repeating their slice of it is noise, not information (it is at DEBUG).
    """
    configure(threads, force=True, announce=False)


__all__ = [
    "DEFAULT_MAX_THREADS",
    "POOL_ENV",
    "THREADS_ENV",
    "ComputeBudget",
    "active",
    "apply_runtime_caps",
    "configure",
    "cpu_count",
    "ffmpeg_thread_args",
    "opencv",
    "pool_worker_init",
    "resolve_budget",
    "worker_count",
]
