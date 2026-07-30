"""Gate orchestration: run the right lane, stamp the protocol, return a verdict."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import benchmarks as B
from .control import NullControl
from .metrics import reference as R
from .metrics.signal import ClipScore, score
from .protocol import (
    Lane,
    Protocol,
    ProtocolError,
    TrajectoryPerturbingError,
    Verdict,
    require_reference_lane,
)


@dataclass(slots=True)
class GateReport:
    verdict: Verdict
    lane: Lane
    protocol: Protocol
    benchmarks: dict[str, B.BenchmarkResult] = field(default_factory=dict)
    reference_metrics: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    clip_scores: dict[str, list[dict[str, float]]] = field(default_factory=dict)
    #: The zero-change arm that decided which budgets were trustworthy, if one
    #: was supplied. Without it, every budget is believed as calibrated.
    null_control: NullControl | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": str(self.verdict),
            "lane": str(self.lane),
            "protocol": self.protocol.to_dict(),
            "benchmarks": {k: v.to_dict() for k, v in self.benchmarks.items()},
            "reference_metrics": self.reference_metrics,
            "reasons": self.reasons,
            "clip_scores": self.clip_scores,
            "null_control": self.null_control.to_dict() if self.null_control else None,
            "library": "cozy-eval",
        }

    def summary(self) -> str:
        lines = [f"{self.verdict.upper()}  [{self.lane}]", self.protocol.line()]
        if self.null_control is not None:
            lines.append("  " + self.null_control.line())
        for name, r in self.benchmarks.items():
            mark = {True: "PASS", False: "FAIL", None: "n/a "}[r.passed]
            lines.append(f"  {mark}  {name}")
            for b in r.budgets:
                flag = "---" if b.get("disregarded") else ("ok " if b["passed"] else "BAD")
                lines.append(f"        {flag} "
                             f"{b['budget']}   measured {b['value']:.4f}")
            for n in r.notes:
                lines.append(f"        note: {n}")
        if self.reference_metrics:
            lines.append("  reference metrics: " + ", ".join(
                f"{k}={v:.4f}" for k, v in self.reference_metrics.items()))
        for r in self.reasons:
            lines.append(f"  ! {r}")
        return "\n".join(lines)


def _degenerate_report(pairs: list[tuple[ClipScore, ClipScore]],
                       protocol: Protocol) -> GateReport | None:
    """NO SIGNAL: refuse to score an arm made of constant frames.

    A candidate of uniformly black (or white, or flat) frames scores an imaging
    index of 0.0 and an unbounded Frechet distance against any real reference.
    Those numbers are arithmetic, not evidence, and reporting them as a FAIL
    invites ranking degenerate arms against each other. There is nothing to
    measure; say so.
    """
    cand = [i for i, (_, c) in enumerate(pairs) if c.is_degenerate]
    ref = [i for i, (r, _) in enumerate(pairs) if r.is_degenerate]
    if not cand and not ref:
        return None
    reasons = []
    if cand:
        reasons.append(
            f"candidate carries NO SIGNAL on prompts {cand}: every frame is a constant "
            "fill (black/white/flat). This is not a quality FAIL — a FAIL is a measured "
            "comparison and there is nothing here to compare. Do not rank this arm; "
            "look at the render."
        )
    if ref:
        reasons.append(
            f"reference carries NO SIGNAL on prompts {ref}: every ratio in this gate is "
            "taken against the reference arm, so a constant reference makes the whole "
            "comparison undefined."
        )
    diag = B.BenchmarkResult(
        "signal", None,
        {"degenerate_candidates": float(len(cand)),
         "degenerate_references": float(len(ref)),
         "n": float(len(pairs))},
        [], ["no benchmark was run: the inputs do not carry a measurement."],
    )
    return GateReport(Verdict.DEGENERATE, protocol.lane, protocol, {"signal": diag},
                      {}, reasons)


def _control_faults(control: NullControl | None, protocol: Protocol) -> list[str]:
    """Reasons a supplied null control cannot fully validate this run's budgets."""
    if control is None:
        return []
    reasons = []
    mism = control.mismatches(protocol)
    if mism:
        reasons.append(
            "the null control was not rendered under this run's conditions (" +
            "; ".join(mism) + "): a control only validates budgets for the render "
            "conditions it shares."
        )
    if control.untrusted:
        reasons.append(
            "budgets disregarded on the null control's evidence — "
            + "; ".join(f"{k} (control measured {control.values[k]:.4f})"
                        for k in sorted(control.untrusted))
            + ". A zero-change arm trips them, so they do not transfer to this family "
            "and what they measure is take spread, "
            "not damage. The candidate cannot PASS on axes this family has no working "
            "budget for; the best available verdict is INDETERMINATE."
        )
    if control.n < B.MIN_PROMPTS:
        reasons.append(
            f"the null control ran at n={control.n} < {B.MIN_PROMPTS}: it is itself "
            "under-powered, so its rulings on which budgets transfer are weak."
        )
    return reasons


def _protocol_faults(protocol: Protocol, n_pairs: int) -> list[str]:
    faults = []
    if n_pairs != protocol.n:
        faults.append(f"protocol names {protocol.n} prompts but {n_pairs} pairs were given")
    if not protocol.same_pod:
        faults.append(
            "arms were rendered on different pods: host and SKU move these metrics "
            "on their own, so a cross-pod comparison cannot attribute a difference "
            "to the change under test"
        )
    return faults


def run_population_gate(
    pairs: list[tuple[ClipScore, ClipScore]],
    protocol: Protocol,
    *,
    budgets=B.VIDEO_BUDGETS,
    null_control: NullControl | None = None,
) -> GateReport:
    """The trajectory-perturbing lane: three no-reference benchmarks over n prompts.

    ``pairs`` is ``[(reference_score, candidate_score), ...]``, one entry per
    prompt, in the protocol's prompt order.

    ``null_control`` is a :class:`~cozy_eval.control.NullControl` for
    this family — a zero-change arm that decides which budgets are trustworthy
    here. Budgets it trips are reported but never decisive, and a run with any
    disregarded budget cannot rise above INDETERMINATE. Strongly recommended
    outside the calibrated envelope; see :mod:`cozy_eval.control`.
    """
    if protocol.lane is not Lane.POPULATION:
        raise ProtocolError(
            f"{protocol.changes} is a post-latent change: run run_reference_gate, "
            "which is both valid and far more sensitive for it"
        )
    degenerate = _degenerate_report(pairs, protocol)
    if degenerate is not None:
        return degenerate
    untrusted = null_control.untrusted if null_control else None
    reasons = _protocol_faults(protocol, len(pairs)) + _control_faults(null_control, protocol)
    results = {
        "imaging": B.imaging(pairs, budgets, untrusted),
        "temporal": B.temporal(pairs, budgets, untrusted),
        "distributional": B.distributional(pairs, budgets, untrusted),
    }
    if len(pairs) < B.MIN_PROMPTS:
        reasons.append(
            f"n={len(pairs)} is below the protocol minimum of {B.MIN_PROMPTS} paired "
            "prompts: at n=1 the shipped-clean LTX w8a8 recipe scores an imaging "
            "index of 0.934 on one prompt and 1.005 over eight. A single prompt "
            "measures the take, not the recipe."
        )

    failed = [k for k, v in results.items() if v.passed is False]
    if failed:
        verdict = Verdict.FAIL
        reasons.insert(0, "failed benchmarks: " + ", ".join(failed))
    else:
        verdict = Verdict.INDETERMINATE if reasons else Verdict.PASS
    return GateReport(verdict, protocol.lane, protocol, results, {}, reasons,
                      null_control=null_control)


def run_reference_gate(
    reference_sources,
    candidate_sources,
    protocol: Protocol,
    *,
    budgets=B.REFERENCE_BUDGETS,
    with_lpips: bool = False,
    with_vmaf: bool = False,
) -> GateReport:
    """The same-trajectory lane: PSNR/SSIM (+VMAF/LPIPS) between matching takes.

    Raises :class:`TrajectoryPerturbingError` if the protocol's changes perturb
    the sampling trajectory. That refusal is the point of this library.
    """
    require_reference_lane(*protocol.changes)
    refs, cands = list(reference_sources), list(candidate_sources)
    if len(refs) != len(cands):
        raise ProtocolError(f"{len(refs)} reference sources vs {len(cands)} candidates")
    reasons = _protocol_faults(protocol, len(refs))
    metrics: dict[str, list[float]] = {}
    for ref, cand in zip(refs, cands, strict=True):
        row = R.compare(ref, cand, with_lpips=with_lpips)
        if with_vmaf:
            row.update(R.vmaf(ref, cand))
        for k, v in row.items():
            metrics.setdefault(k, []).append(v)
    agg: dict[str, float] = {}
    for k, vs in metrics.items():
        agg[k] = min(vs) if ("worst" in k or "min" in k) else sum(vs) / len(vs)

    values = dict(agg)
    ok, rows = B._budgets_for({b.name for b in budgets}, budgets, values)
    result = B.BenchmarkResult("reference", ok, values, rows)
    verdict = Verdict.FAIL if not ok else (
        Verdict.INDETERMINATE if reasons else Verdict.PASS)
    return GateReport(verdict, protocol.lane, protocol, {"reference": result}, agg, reasons)


def score_pairs(reference_sources, candidate_sources) -> list[tuple[ClipScore, ClipScore]]:
    """Convenience: score two aligned lists of frame sources into gate input."""
    refs = list(reference_sources)
    cands = list(candidate_sources)
    if len(refs) != len(cands):
        raise ProtocolError(f"{len(refs)} reference sources vs {len(cands)} candidates")
    return [(score(r), score(c)) for r, c in zip(refs, cands, strict=True)]


__all__ = [
    "GateReport",
    "TrajectoryPerturbingError",
    "Verdict",
    "run_population_gate",
    "run_reference_gate",
    "score_pairs",
]
