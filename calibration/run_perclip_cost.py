"""Regenerate ``perclip-cost.json`` — what one clip of the CPU tier costs.

The portfolio question is coverage per second, so the bill has to be measured
rather than assumed. Times each family of the per-clip video tier on the same
labeled 15 s pair the bounds were fixed on (362 frames, 1344x768), and times the
``signal`` family BOTH ways: the full per-frame feature pass (``signal.score``,
the population lane's entry point) and the luma-only pass the per-clip path uses
from @9 (``signal.temporal_score``).

The cut it justifies: the full pass computes six per-frame feature families —
an FFT, a Laplacian, a 64-bin histogram, a box filter, saturation and luma sigma
at full resolution — to return two luma scalars, and its feature matrix has
exactly one consumer, ``benchmarks.imaging()``, which is the POPULATION lane.

    python calibration/run_perclip_cost.py --samples-root ~/cozy/samples
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

from cozy_eval import integrity, registry, resources
from cozy_eval.metrics import signal as S
from cozy_eval.metrics import temporal as TP
from cozy_eval.metrics import tracks as T

COV15 = "h3-15s-coverage-20260811/out/idx"
REFERENCE = f"{COV15}/courier-15s__dense-r1.mp4"
CANDIDATE = f"{COV15}/courier-15s__mn-c10-r1.mp4"


def decode(path: Path) -> np.ndarray:
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


def timed(fn):
    t0 = time.monotonic()
    out = fn()
    return round(time.monotonic() - t0, 3), out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-root", type=Path, default=Path.home() / "cozy/samples")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "perclip-cost.json")
    args = ap.parse_args()
    resources.configure()

    ref_path, cand_path = args.samples_root / REFERENCE, args.samples_root / CANDIDATE
    if not ref_path.exists() or not cand_path.exists():
        print("clips not present; point --samples-root at them", file=sys.stderr)
        return 1

    decode_ref, ref = timed(lambda: decode(ref_path))
    decode_cand, cand = timed(lambda: decode(cand_path))

    cost: dict[str, float] = {}
    cost["decode_per_clip"] = round((decode_ref + decode_cand) / 2, 3)
    cost["integrity"], _ = timed(lambda: integrity.output_integrity(cand))
    cost["signal_full_feature_pass"], full = timed(lambda: S.score(cand))
    cost["signal_luma_only"], lean = timed(lambda: S.temporal_score(cand))
    cost["warp_error_per_arm"], _ = timed(lambda: TP.warp_error(cand))
    cost["temporal_fidelity_pair"], fid = timed(lambda: TP.temporal_fidelity(ref, cand))
    cost["track_fidelity_pair"], trk = timed(lambda: T.track_fidelity(ref, cand))

    exact = (lean["flicker"] == full.flicker
             and lean["jerk_ratio"] == full.jerk_ratio)
    # The per-clip gate = everything a single paired clip pays, both ways round.
    families = ("integrity", "warp_error_per_arm", "temporal_fidelity_pair",
                "track_fidelity_pair")
    base = cost["decode_per_clip"] * 2 + sum(cost[k] for k in families)
    doc = {
        "library": "cozy-eval",
        "metric_set": registry.METRIC_SET_VERSION,
        "what": ("per-clip CPU cost of the video tier, measured on the labeled 15 s "
                 "pair; and the signal cut's exactness proof"),
        "clip": {"reference": REFERENCE, "candidate": CANDIDATE,
                 "frames": int(cand.shape[0]), "height": int(cand.shape[1]),
                 "width": int(cand.shape[2])},
        "threads": resources.active().threads,
        "seconds": cost,
        "gate_seconds": {
            "before_at8": round(base + cost["signal_full_feature_pass"], 1),
            "after_at9": round(base + cost["signal_luma_only"], 1),
            "saved": round(cost["signal_full_feature_pass"] - cost["signal_luma_only"], 1),
        },
        "signal_cut_is_exact": exact,
        "signal_values": {
            "luma_flicker_full": full.flicker, "luma_flicker_lean": lean["flicker"],
            "jerk_ratio_full": full.jerk_ratio, "jerk_ratio_lean": lean["jerk_ratio"],
        },
        "measured": {"track_stability_ratio": trk["track_stability_ratio"],
                     "motion_mag_ratio": fid["motion_mag_ratio"],
                     "warp_error_delta": fid["warp_error_delta"]},
    }
    args.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: doc[k] for k in
                      ("seconds", "gate_seconds", "signal_cut_is_exact", "measured")},
                     indent=1))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
