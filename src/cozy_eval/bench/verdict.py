"""The tri-state verdict: two questions, not one.

A paired comparison of a candidate against a reference answers two genuinely
different questions, and collapsing them into one pass/fail throws away the
distinction that matters:

  faithfulness   does the candidate approximate the reference as closely as
                 possible? This is the SIMILARITY dimension. It is the gold
                 standard — a faithful candidate is a free win.
  capability     does the candidate do the job as well as the reference? This
    parity       is the ADHERENCE and PREFERENCE dimensions, as deltas against
                 the reference's own scores. It is the fallback.

  free_win            faithfulness holds. Ships, no human needed.
  conditional_parity  not faithful, but parity holds. The candidate did the job
                      differently and equally well. Ships past a REVIEWER, with
                      the divergence shown.
  reject              neither holds. No review needed to ship that verdict.

TWO ENCODINGS HERE ARE SAFETY-CRITICAL AND DELIBERATE.

1. **Parity rescues a divergent candidate ONLY where the metric was measured
   AND a threshold exists to judge it against.** "Parity was never checked" is
   not "parity holds", and a measured number with no threshold is not a check
   either. Without this rule, `reject` becomes unreachable the moment parity is
   merely *measured*, silently un-gating every run that has no judge attached.

2. **faithful-but-parity-degraded is not a case the policy enumerates**, and is
   near self-contradictory — a render that close to the reference should score
   like it. The verdict follows the rule literally (faithfulness passing means
   `free_win`) but the result carries an explicit ``anomaly`` flag rather than
   resolving it silently in either direction.

Thresholds are NOT shipped with the library. A budget is calibrated against the
population you are gating; a number that is a sensible tail limit for one model
family is meaningless for another. See ``Budget``.
"""

from __future__ import annotations

import msgspec

from . import registry

FREE_WIN = "free_win"
CONDITIONAL_PARITY = "conditional_parity"
REJECT = "reject"

# Which dimensions answer which question.
FAITHFULNESS_DIMENSIONS: tuple[str, ...] = (registry.SIMILARITY,)
PARITY_DIMENSIONS: tuple[str, ...] = (registry.ADHERENCE, registry.PREFERENCE)


class Threshold(msgspec.Struct, frozen=True, kw_only=True):
    """A budget for one metric.

    ``mean_limit`` bounds the population mean, ``tail_limit`` the worst single
    sample. Both are expressed in the metric's own direction: the registry's
    ``higher_is_better`` decides whether a limit is a floor or a ceiling, so a
    caller never has to remember which way LPIPS runs.

    A tail limit is the point of this structure. A mean hides one catastrophic
    sample inside seven good ones; the worst row is what a user actually hits.
    """

    metric: str
    mean_limit: float | None = None
    tail_limit: float | None = None


Budget = tuple[Threshold, ...]


class Measurement(msgspec.Struct, frozen=True, kw_only=True):
    """What a run actually observed for one metric.

    ``mean``/``worst`` are None when the metric was NOT MEASURED — which is a
    different fact from scoring zero, and the whole verdict logic turns on
    keeping them distinct.
    """

    metric: str
    mean: float | None = None
    worst: float | None = None
    worst_row: str = ""


class Verdict(msgspec.Struct, kw_only=True):
    verdict: str
    faithfulness_passed: bool
    # None = parity was NOT MEASURED (no judge / no preference model / no budget).
    parity_passed: bool | None = None
    faithfulness_reasons: list[str] = msgspec.field(default_factory=list)
    parity_reasons: list[str] = msgspec.field(default_factory=list)
    needs_human_review: bool = False
    anomaly: str = ""
    note: str = ""


def _breaches(value: float, limit: float, higher_is_better: bool) -> bool:
    return value < limit if higher_is_better else value > limit


def _reasons(
    measurements: dict[str, Measurement],
    budget: dict[str, Threshold],
    dimensions: tuple[str, ...],
) -> list[str]:
    """Human-readable breach lines for the gated metrics of these dimensions."""
    out: list[str] = []
    for spec in registry.REGISTRY:
        if spec.dimension not in dimensions or not spec.gated:
            continue
        m, t = measurements.get(spec.name), budget.get(spec.name)
        if m is None or t is None:
            continue
        hib = spec.higher_is_better
        if (m.mean is not None and t.mean_limit is not None
                and _breaches(m.mean, t.mean_limit, hib)):
            out.append(f"{spec.name}_mean {m.mean:.4g} vs budget {t.mean_limit:.4g}")
        if (m.worst is not None and t.tail_limit is not None
                and _breaches(m.worst, t.tail_limit, hib)):
            row = f" on {m.worst_row}" if m.worst_row else ""
            out.append(
                f"{spec.name}_worst {m.worst:.4g} vs tail budget {t.tail_limit:.4g}{row}"
            )
    return out


def _parity_measured(
    measurements: dict[str, Measurement], budget: dict[str, Threshold]
) -> bool:
    """True only where some parity metric was measured AND has a threshold."""
    for spec in registry.REGISTRY:
        if spec.dimension not in PARITY_DIMENSIONS or not spec.gated:
            continue
        m, t = measurements.get(spec.name), budget.get(spec.name)
        if m is None or t is None:
            continue
        has_value = m.mean is not None or m.worst is not None
        has_limit = t.mean_limit is not None or t.tail_limit is not None
        if has_value and has_limit:
            return True
    return False


def evaluate(measurements: list[Measurement] | dict[str, Measurement], budget: Budget) -> Verdict:
    """Decide the tri-state from measurements and a calibrated budget."""
    obs = (
        {m.metric: m for m in measurements}
        if isinstance(measurements, list)
        else dict(measurements)
    )
    bud = {t.metric: t for t in budget}

    faith = _reasons(obs, bud, FAITHFULNESS_DIMENSIONS)
    parity = _reasons(obs, bud, PARITY_DIMENSIONS)
    measured = _parity_measured(obs, bud)

    if not faith:
        anomaly, note = "", "faithfulness holds: the candidate approximates the reference"
        if measured and parity:
            anomaly = "faithful_but_parity_degraded"
            note = (
                "ANOMALY: faithfulness holds while a measured capability "
                "dimension degraded — " + "; ".join(parity)
            )
        return Verdict(
            verdict=FREE_WIN,
            faithfulness_passed=True,
            parity_passed=(not parity) if measured else None,
            parity_reasons=parity,
            anomaly=anomaly,
            note=note,
        )

    if measured and not parity:
        return Verdict(
            verdict=CONDITIONAL_PARITY,
            faithfulness_passed=False,
            parity_passed=True,
            faithfulness_reasons=faith,
            needs_human_review=True,
            note=(
                "diverges from the reference but holds capability parity — "
                "a human decides, with the divergence shown"
            ),
        )

    return Verdict(
        verdict=REJECT,
        faithfulness_passed=False,
        parity_passed=False if measured else None,
        faithfulness_reasons=faith,
        parity_reasons=parity,
        note=(
            "diverges from the reference AND degrades on capability"
            if measured
            else "diverges from the reference, and capability parity was NOT "
            "MEASURED, so nothing rescues it"
        ),
    )


__all__ = [
    "CONDITIONAL_PARITY",
    "FAITHFULNESS_DIMENSIONS",
    "FREE_WIN",
    "PARITY_DIMENSIONS",
    "REJECT",
    "Budget",
    "Measurement",
    "Threshold",
    "Verdict",
    "evaluate",
]
