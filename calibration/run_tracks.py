"""Regenerate ``track-stability.json`` — the evidence that fixes BOTH bounds.

Scores the LABELED H3 corpus through the shipped library path
(:func:`cozy_eval.metrics.tracks.track_stats`) and reports the separation the
family has to clear:

* **rejected** — the sparse-attention k16/k32 arms the owner rejected for object
  warble, each against its own same-cell, same-seed dense control.
* **identical** — the SageAttention-2 fp8 arms the owner reviewed as identical
  to their FA3-exact control, plus same-arm re-renders across a pod change and a
  torch-line change (the re-render noise floor).
* **oversmoothed** (@9) — the arm that scored 2.169 and PASSED the floor-only
  gate while dropping the content it was asked for. The ceiling's evidence.
* **bit-exact** — a clip against itself. Must be exactly 1.0, not approximately.

Two empty middles, and both bounds sit in one: rejected_max < FLOOR <
identical_min, and identical_max < CEILING < oversmoothed_min.

Also measures the DECIMATION EQUIVALENCE pin: the cheaper window budget must not
move a single verdict on the labeled set.

The clips are internal renders and are not redistributed; point --samples-root
at them to reproduce. The committed JSON is the evidence.

    python calibration/run_tracks.py --samples-root ~/cozy/samples
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cozy_eval import resources
from cozy_eval.metrics import tracks as T

FREESEL = "h3-freesel-20260811/out/idx"
FLEET = "h3-indexer-20260810/out/idx-fleet"
IDX = "h3-indexer-20260810/out/idx"
R2 = "pgw1081-r2confirm-20260811/evidence"
COV15 = "h3-15s-coverage-20260811/out/idx"

#: (label, reference clip, candidate clip). The owner REJECTED every candidate
#: here for objects that warble and reshape under camera motion.
REJECTED: list[tuple[str, str, str]] = [
    *[
        (f"freesel:{cell}/{seed}/{arm}",
         f"{FREESEL}/{cell}-5s__dense-{seed}.mp4",
         f"{FREESEL}/{cell}-5s__{arm}-{seed}.mp4")
        for cell in ("busker", "carpet", "glassblower", "whale")
        for seed in ("s20260808", "s20260811")
        for arm in ("mn-k16", "mp-k16", "ph-k16")
    ],
    *[
        (f"fleet:{cell}/{arm}",
         f"{FLEET}/{cell}-5s__dense-r1.mp4",
         f"{FLEET}/{cell}-5s__{arm}-r1.mp4")
        for cell in ("busker", "glassblower", "whale")
        for arm in ("ph-k16", "ph-k32", "mp-k16", "mp-k32")
    ],
]

#: The owner reviewed these as IDENTICAL. A detector that fires here is useless.
IDENTICAL: list[tuple[str, str, str]] = [
    *[
        (f"r2:{cell}{suf}", f"{R2}/{cell}{suf}-30__exactB.mp4",
         f"{R2}/{cell}{suf}-30__r2.mp4")
        for cell in ("busker", "loom", "samurai")
        for suf in ("", "B", "C")
    ],
    ("r2w:samurai", f"{R2}/samurai-30__exactB.mp4", f"{R2}/samurai-30__r2w.mp4"),
    ("r2w:samuraiB", f"{R2}/samuraiB-30__exactB.mp4", f"{R2}/samuraiB-30__r2w.mp4"),
    # same ARM, re-rendered on another pod / another torch line: the pure
    # re-render noise floor, which any real effect has to clear.
    ("floor:busker exact(pod+line)", f"{R2}/busker-30__exactB.mp4",
     f"{R2}/busker-30__exact.mp4"),
    ("floor:samurai exact(pod+line)", f"{R2}/samurai-30__exactB.mp4",
     f"{R2}/samurai-30__exact.mp4"),
    ("floor:loom exact(pod+line)", f"{R2}/loom-30__exactB.mp4",
     f"{R2}/loom-30__exact.mp4"),
    ("floor:loom exactB2(pod)", f"{R2}/loom-30__exactB.mp4",
     f"{R2}/loom-30__exactB2.mp4"),
]

#: THE OTHER SIDE (@9). An arm that re-rolled into a simpler, slower take: it
#: holds MORE coherent tracks than its control and PASSED the floor-only gate at
#: 2.169 while dropping the cargo bike and the parcel the prompt asked for
#: (pgw#1145, 15 s courier cell). The ceiling exists for this class.
OVERSMOOTHED: list[tuple[str, str, str]] = [
    ("cov15:courier/mn-c10", f"{COV15}/courier-15s__dense-r1.mp4",
     f"{COV15}/courier-15s__mn-c10-r1.mp4"),
]

#: Arms an independent instrument already called bad, and the oracle top-k arm.
#: Not part of the acceptance bar; reported so the family's reading of a known
#: answer is visible.
CONTROLS: list[tuple[str, str, str]] = [
    ("negative:busker jl-k32", f"{FLEET}/busker-5s__dense-r1.mp4",
     f"{IDX}/busker-5s__jl-k32-r1.mp4"),
    ("negative:busker g8-k32", f"{FLEET}/busker-5s__dense-r1.mp4",
     f"{IDX}/busker-5s__g8-k32-r1.mp4"),
    ("oracle:busker oracle-k16", f"{FLEET}/busker-5s__dense-r1.mp4",
     f"{FLEET}/busker-5s__oracle-k16-r1.mp4"),
]

BITEXACT = [("bitexact:busker dense", f"{FLEET}/busker-5s__dense-r1.mp4",
             f"{FLEET}/busker-5s__dense-r1.mp4"),
            ("bitexact:samurai exactB", f"{R2}/samurai-30__exactB.mp4",
             f"{R2}/samurai-30__exactB.mp4")]


def decode(path: Path) -> np.ndarray:
    """Whole clip as (T, H, W, 3) uint8 — the pre-encode array a producer holds."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    w, h = (int(x) for x in probe.split("x"))
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", *resources.ffmpeg_thread_args(), "-i", str(path),
         "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def score_all(root: Path, clips: list[str], windows: int) -> dict[str, dict]:
    """``windows=0`` means UNDECIMATED: enough windows to cover the whole clip."""
    out: dict[str, dict] = {}
    for rel in clips:
        path = root / rel
        if not path.exists():
            print(f"  MISSING {rel}", file=sys.stderr)
            continue
        frames = decode(path)
        n = windows or -(-int(frames.shape[0]) // T.TRACK_WINDOW)
        t0 = time.monotonic()
        stats = T.track_stats(frames, windows=n)
        seconds = time.monotonic() - t0
        out[rel] = {
            "track_stability": stats.track_stability,
            "track_survival": stats.track_survival,
            "track_jitter": stats.track_jitter,
            "track_rigidity_error": stats.track_rigidity_error,
            "motion_magnitude": stats.motion_magnitude,
            "frames": int(frames.shape[0]),
            "height": int(frames.shape[1]), "width": int(frames.shape[2]),
            "seconds": round(seconds, 3),
        }
        print(f"  {rel:64s} stability={stats.track_stability:.4f} "
              f"survival={stats.track_survival:.3f} {seconds:.2f}s", flush=True)
        del frames
    return out


def pairs_table(scores: dict[str, dict], pairs: list[tuple[str, str, str]]) -> list[dict]:
    rows = []
    for label, ref, cand in pairs:
        a, b = scores.get(ref), scores.get(cand)
        if a is None or b is None:
            continue
        row = {"pair": label, "reference": ref, "candidate": cand,
               "ref_stability": a["track_stability"],
               "cand_stability": b["track_stability"],
               "ref_survival": a["track_survival"],
               "cand_survival": b["track_survival"]}
        raw = (b["track_stability"] / a["track_stability"]
               if a["track_stability"] > 1e-9 else None)
        if a["track_survival"] < T.TRACKABILITY_FLOOR or a["track_stability"] <= 1e-9:
            row["ratio"] = None
            row["verdict"] = "unmeasured"
            row["why"] = "untrackable reference"
            # What the family WOULD have said. Banked because it is the floor's
            # own justification: on the loom cell it would call a pair the owner
            # judged identical a catastrophic reject.
            row["suppressed_ratio"] = raw
        else:
            row["ratio"] = raw
            if raw < T.STABILITY_RATIO_FLOOR:
                row["verdict"], row["why"] = "reject", "object warble (below floor)"
            elif raw > T.STABILITY_RATIO_CEILING:
                # The ceiling needs a higher trackability bar than the floor:
                # a small denominator inflates the UPWARD tail (loomB reads
                # 2.766 undecimated on content the owner called identical).
                if a["track_survival"] < T.CEILING_TRACKABILITY_FLOOR:
                    row["verdict"] = "pass"
                    row["why"] = "ceiling unmeasured: control too marginal to track"
                else:
                    row["verdict"], row["why"] = "reject", "over-smoothing (above ceiling)"
            else:
                row["verdict"] = "pass"
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    vals = [r["ratio"] for r in rows if r["ratio"] is not None]
    if not vals:
        return {"n": 0, "unmeasured": len(rows)}
    return {
        "n": len(vals), "unmeasured": sum(1 for r in rows if r["ratio"] is None),
        "median": float(np.median(vals)), "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "verdicts": {v: sum(1 for r in rows if r["verdict"] == v)
                     for v in ("pass", "reject", "unmeasured")},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-root", type=Path, default=Path.home() / "cozy/samples")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "track-stability.json")
    ap.add_argument("--reuse-scores", type=Path, default=None, help=(
        "re-derive the verdict tables from a previous run's clip scores instead "
        "of re-scoring 76 clips at three budgets. For a BOUND change only — the "
        "scores are the expensive, unchanged half"))
    args = ap.parse_args()
    resources.configure()

    every = sorted({c for _, a, b in REJECTED + IDENTICAL + CONTROLS + OVERSMOOTHED
                    for c in (a, b)})
    if args.reuse_scores:
        prev = json.loads(args.reuse_scores.read_text())
        shipped = prev["clip_scores"]
        full = prev["clip_scores_undecimated"]
        halved = prev["clip_scores_halved_budget"]
        missing = [c for c in every if c not in shipped]
        if missing:
            print(f"--reuse-scores is missing {len(missing)} clips: {missing[:3]}",
                  file=sys.stderr)
            return 1
        print(f"reusing {len(shipped)} clip scores from {args.reuse_scores} — "
              "re-deriving verdicts only")
    else:
        print(f"scoring {len(every)} clips at the shipped budget "
              f"(windows={T.TRACK_WINDOWS}, window={T.TRACK_WINDOW}, "
              f"target_h={T.TRACK_TARGET_H})")
        shipped = score_all(args.samples_root, every, T.TRACK_WINDOWS)
        print(f"scoring {len(every)} clips UNDECIMATED (windows cover the whole clip) "
              "— the equivalence pin")
        full = score_all(args.samples_root, every, 0)
        print(f"scoring {len(every)} clips at HALF the shipped budget (windows=2) — "
              "the bottom of the ladder")
        halved = score_all(args.samples_root, every, 2)

    blocks = {}
    for name, pairs in (("rejected", REJECTED), ("identical", IDENTICAL),
                        ("oversmoothed", OVERSMOOTHED), ("controls", CONTROLS)):
        blocks[name] = pairs_table(shipped, pairs)
    # BIT-EXACT: two INDEPENDENT decodes of the same file, scored by two
    # independent calls. Dividing one stored number by itself would prove
    # nothing; this proves the tracker is deterministic, which is the only way a
    # zero-change arm can score exactly zero change.
    bitexact = json.loads(args.reuse_scores.read_text())["bit_exact"] \
        if args.reuse_scores else []
    for label, rel, _ in ([] if args.reuse_scores else BITEXACT):
        path = args.samples_root / rel
        if not path.exists():
            continue
        a = T.track_stats(decode(path))
        b = T.track_stats(decode(path))
        bitexact.append({
            "pair": label, "clip": rel,
            "a": a.track_stability, "b": b.track_stability,
            "ratio": b.track_stability / a.track_stability,
            "exactly_one": b.track_stability / a.track_stability == 1.0,
            "all_fields_identical": (a.track_survival == b.track_survival
                                     and a.track_jitter == b.track_jitter
                                     and a.track_rigidity_error == b.track_rigidity_error),
        })

    rej = [r["ratio"] for r in blocks["rejected"] if r["ratio"] is not None]
    ident = [r["ratio"] for r in blocks["identical"] if r["ratio"] is not None]
    over = [r["ratio"] for r in blocks["oversmoothed"] if r["ratio"] is not None]

    def equivalence(other: dict[str, dict], label: str) -> list[dict]:
        rows = []
        for name, pairs in (("rejected", REJECTED), ("identical", IDENTICAL),
                            ("oversmoothed", OVERSMOOTHED)):
            for a, b in zip(pairs_table(shipped, pairs), pairs_table(other, pairs),
                            strict=True):
                rows.append({
                    "set": name, "pair": a["pair"], "shipped": a["verdict"],
                    label: b["verdict"], "shipped_ratio": a["ratio"],
                    f"{label}_ratio": b["ratio"], "same": a["verdict"] == b["verdict"],
                })
        return rows

    pin = equivalence(full, "undecimated")
    ladder = equivalence(halved, "halved")

    doc = {
        "library": "cozy-eval",
        "metric_set": "cozy-eval/metrics@9",
        "what": ("track-stability calibration: the owner-labeled separation that "
                 "fixes BOTH bounds — STABILITY_RATIO_FLOOR against warble, and "
                 "(@9) STABILITY_RATIO_CEILING against over-smoothing"),
        "settings": {"windows": T.TRACK_WINDOWS, "window": T.TRACK_WINDOW,
                     "target_h": T.TRACK_TARGET_H, "points": T.TRACK_POINTS,
                     "jitter_knee": T.JITTER_KNEE, "rigidity_knee": T.RIGIDITY_KNEE,
                     "trackability_floor": T.TRACKABILITY_FLOOR,
                     "stability_ratio_floor": T.STABILITY_RATIO_FLOOR,
                     "stability_ratio_ceiling": T.STABILITY_RATIO_CEILING,
                     "ceiling_trackability_floor": T.CEILING_TRACKABILITY_FLOOR,
                     "threads": resources.active().threads},
        "clip_scores": shipped,
        "clip_scores_undecimated": full,
        "clip_scores_halved_budget": halved,
        "pairs": blocks,
        "bit_exact": bitexact,
        "summary": {k: summarize(v) for k, v in blocks.items()},
        "separation": {
            "rejected_max": max(rej) if rej else None,
            "identical_min": min(ident) if ident else None,
            "empty_middle": (max(rej) < min(ident)) if rej and ident else None,
            "floor": T.STABILITY_RATIO_FLOOR,
            "identical_max": max(ident) if ident else None,
            "oversmoothed_min": min(over) if over else None,
            "empty_middle_upper": (max(ident) < min(over)) if ident and over else None,
            "ceiling": T.STABILITY_RATIO_CEILING,
        },
        "decimation_pin": pin,
        "decimation_equivalent": all(e["same"] for e in pin),
        "decimation_ladder_halved": ladder,
        "halved_equivalent": all(e["same"] for e in ladder),
        "cost_seconds": {
            "median_per_clip": float(np.median([c["seconds"] for c in shipped.values()])),
            "max_per_clip": float(np.max([c["seconds"] for c in shipped.values()])),
            "median_per_clip_undecimated": float(
                np.median([c["seconds"] for c in full.values()])),
            "median_per_clip_halved": float(
                np.median([c["seconds"] for c in halved.values()])),
        },
    }
    args.out.write_text(json.dumps(doc, indent=1) + "\n")

    print()
    for name in ("rejected", "identical", "oversmoothed", "controls"):
        print(f"{name:10s} {json.dumps(doc['summary'][name])}")
    print(f"separation {json.dumps(doc['separation'])}")
    print(f"bit-exact  {json.dumps(bitexact)}")
    print(f"decimation pin (shipped vs undecimated) equivalent: "
          f"{doc['decimation_equivalent']}")
    for e in pin:
        if not e["same"]:
            print(f"  DIFFERS {e}")
    print(f"half budget equivalent: {doc['halved_equivalent']}")
    for e in ladder:
        if not e["same"]:
            print(f"  DIFFERS {e}")
    print(f"cost {json.dumps(doc['cost_seconds'])}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
