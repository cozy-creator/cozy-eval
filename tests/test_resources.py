"""The compute budget: bounded by default, honoured everywhere, invisible to scores.

Two promises are under test. First, that a plain ``import cozy_eval`` caps the
machine — environment pools, torch, OpenCV, process pools and every ffmpeg
subprocess — and that an explicit caller choice survives. Second, and the reason
the cap is allowed to exist at all: THE THREAD COUNT DOES NOT MOVE A NUMBER. A
budget that changed a metric would be a silent verdict change.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import pytest

from cozy_eval import resources
from cozy_eval.resources import POOL_ENV, THREADS_ENV, resolve_budget

SRC = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src")


def _run(code: str, **env: str) -> dict:
    """Run ``code`` in a fresh interpreter; it prints one JSON object.

    An env value of ``""`` UNSETS the variable, so a test can ask for a machine
    that carries no thread settings at all.
    """
    child = {**os.environ, "PYTHONPATH": SRC, **env}
    child = {k: v for k, v in child.items() if v != ""}
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, check=True, env=child,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


_PROBE = """
import json, os, sys
import cozy_eval
from cozy_eval import resources
b = resources.active()
import torch, cv2
resources.apply_runtime_caps()
print(json.dumps({
    "threads": b.threads, "workers": b.workers, "source": b.source,
    "env": {k: os.environ.get(k) for k in resources.POOL_ENV},
    "torch": torch.get_num_threads(),
    "torch_interop": torch.get_num_interop_threads(),
    "cv2": cv2.getNumThreads(),
    "ffmpeg": resources.ffmpeg_thread_args(),
}))
"""


# ---------------------------------------------------------------------------
# the default is modest, and it reaches every pool
# ---------------------------------------------------------------------------

def test_a_bare_import_caps_the_machine() -> None:
    """PIN — the incident: a score run spawned 195 threads (~10 cores) on a
    shared 32-core box because nothing bounded numpy/BLAS/torch/OpenCV."""
    got = _run(_PROBE, **{k: "" for k in (THREADS_ENV, *POOL_ENV)})
    cap = min(4, resources.cpu_count())
    assert got["threads"] == cap
    assert got["source"] == "default"
    assert got["env"] == dict.fromkeys(POOL_ENV, str(cap))
    # The cap only LOWERS torch; its own default (physical cores) may sit below
    # the budget on small hosts, so the promise is a bound, not equality.
    assert 1 <= got["torch"] <= cap and got["torch_interop"] <= cap
    assert got["cv2"] == cap
    assert got["ffmpeg"] == ["-threads", str(cap)]


def test_the_caller_s_own_omp_num_threads_is_never_overridden() -> None:
    got = _run(_PROBE, OMP_NUM_THREADS="3", **{THREADS_ENV: ""})
    assert (got["threads"], got["source"]) == (3, "OMP_NUM_THREADS")
    assert got["env"]["OMP_NUM_THREADS"] == "3"
    assert 1 <= got["torch"] <= 3


def test_the_knob_raises_the_budget_on_a_dedicated_machine() -> None:
    """A pod that owns its cores says so; the library never assumes it."""
    n = min(8, resources.cpu_count())
    got = _run(_PROBE, OMP_NUM_THREADS="", **{THREADS_ENV: str(n)})
    assert (got["threads"], got["source"]) == (n, THREADS_ENV)
    assert got["env"]["OPENBLAS_NUM_THREADS"] == str(n)
    assert got["cv2"] == n


def test_the_knob_outranks_an_inherited_omp() -> None:
    got = _run(_PROBE, OMP_NUM_THREADS="7", **{THREADS_ENV: "2"})
    assert (got["threads"], got["source"]) == (2, THREADS_ENV)
    assert got["env"]["OMP_NUM_THREADS"] == "2"


@pytest.mark.parametrize("raw", ["", "0", "-4", "many"])
def test_a_junk_budget_falls_back_to_the_default(raw: str) -> None:
    os.environ[THREADS_ENV] = raw
    try:
        assert resolve_budget().threads == min(4, resources.cpu_count())
    finally:
        os.environ.pop(THREADS_ENV, None)


# ---------------------------------------------------------------------------
# pools SPLIT the budget; they do not multiply it
# ---------------------------------------------------------------------------

def test_a_pool_splits_the_budget_instead_of_multiplying_it() -> None:
    budget = resolve_budget(4)
    assert budget.workers == 2
    assert budget.workers * budget.worker_threads <= budget.threads


def _child_env() -> tuple[str | None, int]:
    from cozy_eval import resources as r

    return os.environ.get("OMP_NUM_THREADS"), r.active().threads


def test_a_pool_worker_is_pinned_to_its_share() -> None:
    with ProcessPoolExecutor(max_workers=2, initializer=resources.pool_worker_init,
                             initargs=(2,)) as ex:
        got = list(ex.map(_noop, range(2)))
    assert got == [("2", 2), ("2", 2)]


def _noop(_: int) -> tuple[str | None, int]:
    return _child_env()


# ---------------------------------------------------------------------------
# subprocesses inherit it — an unbudgeted ffmpeg is the whole bug class
# ---------------------------------------------------------------------------

def test_the_video_decoder_hands_ffmpeg_the_budget(monkeypatch) -> None:
    from cozy_eval import frames

    seen: list[list[str]] = []

    class _Boom(Exception):
        pass

    def fake_popen(cmd, **_kw):
        seen.append(cmd)
        raise _Boom

    monkeypatch.setattr(frames, "probe", lambda p: (16, 16, 24.0, 2))
    monkeypatch.setattr(frames.subprocess, "Popen", fake_popen)
    with pytest.raises(_Boom):
        next(frames.iter_video("clip.mp4"))
    n = str(resources.active().threads)
    assert seen[0][:1] == ["ffmpeg"]
    assert seen[0][seen[0].index("-threads") + 1] == n


def test_vmaf_budgets_ffmpeg_and_libvmaf_separately(monkeypatch) -> None:
    """libvmaf runs its OWN pool on top of ffmpeg's threads; both are capped."""
    from cozy_eval.metrics import reference

    seen: list[list[str]] = []

    class _Proc:
        returncode = 1
        stderr = "nope"

    def fake_run(cmd, **_kw):
        seen.append(cmd)
        return _Proc()

    monkeypatch.setattr(reference, "_probe_video", lambda p: (64, 64, "24/1"))
    monkeypatch.setattr(reference.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError):
        reference.vmaf("ref.mp4", "cand.mp4")
    n = str(resources.active().threads)
    cmd = seen[0]
    assert cmd[cmd.index("-threads") + 1] == n
    assert cmd[cmd.index("-filter_threads") + 1] == n
    assert f"libvmaf=n_threads={n}:" in cmd[cmd.index("-lavfi") + 1]


def test_the_audio_decoder_hands_ffmpeg_the_budget(monkeypatch) -> None:
    from cozy_eval import audio

    seen: list[list[str]] = []

    class _Out:
        stdout = b""

    def fake_run(cmd, **_kw):
        seen.append(cmd)
        return _Out()

    monkeypatch.setattr(audio, "probe_audio", lambda p: (48000, 2, 1.0))
    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    with pytest.raises(Exception):  # noqa: B017 — decodes nothing; we want the argv
        audio.read_audio("clip.mp4")
    n = str(resources.active().threads)
    assert seen[0][seen[0].index("-threads") + 1] == n


# ---------------------------------------------------------------------------
# and none of it moves a number
# ---------------------------------------------------------------------------

_SCORE = """
import json, numpy as np
from cozy_eval.metrics import signal, similarity, temporal

rng = np.random.default_rng(7)
base = rng.random((12, 48, 64, 3), dtype=np.float32)
ramp = np.linspace(0, 1, 12, dtype=np.float32)[:, None, None, None]
ref = np.clip(base * 0.6 + 0.2 * ramp, 0, 1)
cand = np.clip(ref + rng.normal(0, 0.01, ref.shape).astype(np.float32), 0, 1)

out = {
    "signal": signal.score(ref).metrics(),
    "integrity": {k: v for k, v in temporal.integrity_stats(ref).items()
                  if not isinstance(v, list)},
    "flow": temporal.temporal_fidelity(ref, cand),
    "dframe_psnr": temporal.dframe_psnr_series(ref, cand),
    "ssim": similarity.ssim((ref[0] * 255).astype("uint8"), (cand[0] * 255).astype("uint8")),
}
print(json.dumps(out, sort_keys=True))
"""


def test_the_budget_is_invisible_to_every_score() -> None:
    """Speed may never move a verdict: one thread and four must agree BIT for
    BIT across the numpy, OpenCV-flow and torch paths."""
    one = _run(_SCORE, **{THREADS_ENV: "1"})
    four = _run(_SCORE, **{THREADS_ENV: "4"})
    assert one == four
