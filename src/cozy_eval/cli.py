"""``cozy-eval`` command line.

Three subcommands:

* ``score``  — no-reference statistics for one or more clips/images.
* ``gate``   — run the full population gate on a paired prompt set.
* ``compare``— reference metrics for a same-trajectory change (refuses otherwise).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import __version__
from .backends.signal import score
from .gate import run_population_gate, run_reference_gate
from .protocol import ChangeKind, Protocol, TrajectoryPerturbingError


def _protocol(a, n_prompts: int) -> Protocol:
    return Protocol(
        family=a.family,
        reference_arm=a.reference_arm,
        candidate_arm=a.candidate_arm,
        changes=tuple(ChangeKind(c) for c in a.change),
        width=a.width, height=a.height, frames=a.frames, steps=a.steps,
        seeds=tuple(a.seed) or (0,),
        prompts=tuple(a.prompt) if a.prompt else tuple(str(i) for i in range(n_prompts)),
        execution_lane=a.execution_lane,
        hardware=a.hardware,
        same_pod=not a.cross_pod,
    )


def _add_protocol_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--family", required=True)
    p.add_argument("--reference-arm", required=True)
    p.add_argument("--candidate-arm", required=True)
    p.add_argument("--change", action="append", required=True,
                   choices=[str(c) for c in ChangeKind],
                   help="what the candidate changed; repeatable. Decides the lane.")
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--frames", type=int, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--seed", type=int, action="append", default=[])
    p.add_argument("--prompt", action="append", default=[])
    p.add_argument("--execution-lane", choices=("compiled", "eager"), required=True)
    p.add_argument("--hardware", required=True)
    p.add_argument("--cross-pod", action="store_true",
                   help="declare that the arms were NOT rendered on the same pod")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser("cozy-eval", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="no-reference statistics for clips/images")
    s.add_argument("paths", nargs="+")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--json", type=Path)

    g = sub.add_parser("gate", help="population gate over a paired prompt set")
    g.add_argument("--reference", action="append", required=True)
    g.add_argument("--candidate", action="append", required=True)
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--json", type=Path)
    _add_protocol_args(g)

    c = sub.add_parser("compare", help="reference metrics (same-trajectory lane only)")
    c.add_argument("--reference", action="append", required=True)
    c.add_argument("--candidate", action="append", required=True)
    c.add_argument("--vmaf", action="store_true")
    c.add_argument("--lpips", action="store_true")
    c.add_argument("--json", type=Path)
    _add_protocol_args(c)

    a = ap.parse_args(argv)

    if a.cmd == "score":
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            scores = dict(zip(a.paths, ex.map(score, a.paths), strict=True))
        out = {p: s.metrics() for p, s in scores.items()}
        for p, d in out.items():
            print(f"{p}\n  " + "  ".join(f"{k}={v:.5g}" for k, v in d.items()))
        if a.json:      # lossless: ClipScore.from_dict rebuilds these exactly
            a.json.write_text(json.dumps({p: s.to_dict() for p, s in scores.items()}, indent=1))
        return 0

    if a.cmd == "gate":
        if len(a.reference) != len(a.candidate):
            ap.error("--reference and --candidate must be given the same number of times")
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            scores = list(ex.map(score, a.reference + a.candidate))
        k = len(a.reference)
        report = run_population_gate(list(zip(scores[:k], scores[k:], strict=True)),
                                     _protocol(a, k))
        print(report.summary())
        if a.json:
            a.json.write_text(json.dumps(report.to_dict(), indent=1))
        return 0 if report.verdict == "pass" else 1

    try:
        report = run_reference_gate(a.reference, a.candidate, _protocol(a, len(a.reference)),
                                    with_lpips=a.lpips, with_vmaf=a.vmaf)
    except TrajectoryPerturbingError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(report.summary())
    if a.json:
        a.json.write_text(json.dumps(report.to_dict(), indent=1))
    return 0 if report.verdict == "pass" else 1


if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(main())
