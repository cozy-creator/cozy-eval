"""Track stability against the OWNER-LABELED corpus, not synthetic warble.

`tests/test_tracks.py` proves the instrument responds to a warble it constructed.
This proves it responds to the one that actually happened, and — the harder half
— that it stays QUIET on the arms the owner looked at and called identical.

Two layers:

* the banked calibration (``calibration/track-stability.json``, committed
  evidence produced by ``calibration/run_tracks.py``) is asserted ALWAYS. It is
  what fixes ``STABILITY_RATIO_FLOOR``, so a change that moves the separation
  has to move this file, in the diff, on purpose.
* re-scoring the clips end to end runs only when the corpus is on disk. They are
  multi-GB internal renders, not a repo fixture.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cozy_eval import tracks
from cozy_eval.metrics import tracks as T

BANKED = Path(__file__).resolve().parents[1] / "calibration" / "track-stability.json"
CORPUS = Path.home() / "cozy/samples"


@pytest.fixture(scope="module")
def banked() -> dict:
    if not BANKED.exists():                                     # pragma: no cover
        pytest.skip(f"calibration evidence missing at {BANKED}")
    return json.loads(BANKED.read_text())


# ---------------------------------------------------------------------------
# the acceptance bar, as banked
# ---------------------------------------------------------------------------

def test_the_rejected_arms_all_fail(banked: dict) -> None:
    """Every owner-rejected sparse-attention pair the family could measure is a
    REJECT — no pass, and none of them squeaking in at the floor."""
    rows = banked["pairs"]["rejected"]
    measured = [r for r in rows if r["ratio"] is not None]
    assert len(measured) >= 25
    assert all(r["verdict"] == "reject" for r in measured)
    assert max(r["ratio"] for r in measured) < T.STABILITY_RATIO_FLOOR


def test_the_arms_the_owner_called_identical_all_pass(banked: dict) -> None:
    """THE HARD HALF. A detector that fires on the fp8-attention arms — or on a
    same-arm re-render across a pod change — would be unusable."""
    rows = banked["pairs"]["identical"]
    measured = [r for r in rows if r["ratio"] is not None]
    assert len(measured) >= 10
    assert all(r["verdict"] == "pass" for r in measured)
    assert min(r["ratio"] for r in measured) > T.STABILITY_RATIO_FLOOR


def test_the_two_populations_have_an_empty_middle_and_the_floor_is_in_it(
    banked: dict,
) -> None:
    sep = banked["separation"]
    assert sep["empty_middle"] is True
    assert sep["rejected_max"] < T.STABILITY_RATIO_FLOOR < sep["identical_min"]
    # a margin, not a hairline: at least 3 points of daylight on each side
    assert T.STABILITY_RATIO_FLOOR - sep["rejected_max"] > 0.03
    assert sep["identical_min"] - T.STABILITY_RATIO_FLOOR > 0.02


def test_bit_identical_pairs_score_exactly_one(banked: dict) -> None:
    rows = banked["bit_exact"]
    assert rows
    for row in rows:
        assert row["ratio"] == 1.0
        assert row["exactly_one"] is True
        assert row["all_fields_identical"] is True


def test_the_independent_negative_controls_also_fire(banked: dict) -> None:
    """Arms a different instrument already called broken (untrained and grouped
    selectors) must not be the ones this family lets through."""
    controls = {r["pair"]: r for r in banked["pairs"]["controls"]}
    for name in ("negative:busker jl-k32", "negative:busker g8-k32"):
        assert controls[name]["verdict"] == "reject"
        assert controls[name]["ratio"] < 0.1


def test_the_trackability_floor_is_what_stops_a_false_reject(banked: dict) -> None:
    """The floor's own justification: on the untrackable loom cell the family
    WOULD have called a pair the owner judged identical a catastrophic reject.
    That is why an untrackable reference is UNMEASURED and not a verdict."""
    suppressed = [
        r for r in banked["pairs"]["identical"]
        if r["ratio"] is None and r.get("suppressed_ratio") is not None
    ]
    assert suppressed, "the corpus must contain an untrackable clean pair"
    assert min(r["suppressed_ratio"] for r in suppressed) < 0.5
    assert all(r["ref_survival"] < T.TRACKABILITY_FLOOR for r in suppressed)


def test_decimation_does_not_change_a_single_verdict(banked: dict) -> None:
    """The compute-budget pin (ce#12): the shipped four-window budget reads the
    labeled set exactly as tracking EVERY frame of every clip does."""
    assert banked["decimation_equivalent"] is True
    pin = banked["decimation_pin"]
    assert len(pin) >= 40
    assert all(row["same"] for row in pin)


def test_the_cost_stays_inside_the_budget_the_library_promises(banked: dict) -> None:
    cost = banked["cost_seconds"]
    # Seconds are load-dependent (the banked run shared a box at load 21, which
    # tripled it against an idle 1.2 s), so this pins the ORDER OF MAGNITUDE and
    # the SHAPE of the ladder, not a wall clock.
    assert cost["median_per_clip"] < 8.0
    assert cost["max_per_clip"] < 15.0
    # the undecimated arm is the reason the budget exists
    assert cost["median_per_clip_undecimated"] > 1.3 * cost["median_per_clip"]
    assert cost["median_per_clip_halved"] < cost["median_per_clip"]


def test_the_banked_settings_are_the_ones_the_library_ships(banked: dict) -> None:
    s = banked["settings"]
    assert s["windows"] == T.TRACK_WINDOWS and s["window"] == T.TRACK_WINDOW
    assert s["target_h"] == T.TRACK_TARGET_H and s["points"] == T.TRACK_POINTS
    assert s["jitter_knee"] == T.JITTER_KNEE
    assert s["rigidity_knee"] == T.RIGIDITY_KNEE
    assert s["trackability_floor"] == T.TRACKABILITY_FLOOR
    assert s["stability_ratio_floor"] == T.STABILITY_RATIO_FLOOR


# ---------------------------------------------------------------------------
# end to end on the real clips, when they are on disk
# ---------------------------------------------------------------------------

def _decode(path: Path) -> np.ndarray:
    wh = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    w, h = (int(x) for x in wh.split("x"))
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-threads", "4", "-i", str(path),
         "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def _pair(banked: dict, name: str, block: str) -> dict:
    for row in banked["pairs"][block]:
        if row["pair"] == name:
            return row
    raise AssertionError(f"{name} not in the banked {block} block")


@pytest.mark.corpus
@pytest.mark.parametrize(
    ("block", "name", "expected"),
    [("rejected", "freesel:busker/s20260808/mp-k16", tracks.REJECT),
     ("identical", "r2:buskerC", tracks.PASS)],
)
def test_the_verdict_reproduces_from_the_clips_themselves(
    banked: dict, block: str, name: str, expected: str,
) -> None:
    """The banked JSON is only evidence if the shipped code still produces it."""
    row = _pair(banked, name, block)
    ref, cand = CORPUS / row["reference"], CORPUS / row["candidate"]
    if not ref.exists() or not cand.exists():
        pytest.skip(f"labeled corpus not on disk at {CORPUS}")
    verdict = tracks.track_verdict(_decode(cand), _decode(ref))
    assert verdict.verdict == expected
    assert verdict.measured["track_stability_ratio"] == pytest.approx(
        row["ratio"], rel=1e-6
    )
