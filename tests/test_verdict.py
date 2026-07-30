"""The tri-state verdict policy.

The truth table below is the whole point of these tests: the first cut of this
logic made REJECT unreachable the moment parity was merely *measured*, which
was only caught by writing the cases out explicitly. They stay explicit — one
named row per rule — and `test_every_verdict_is_reachable` still guards that
bug directly.
"""

from __future__ import annotations

import pytest

from cozy_eval.bench import registry
from cozy_eval.bench.verdict import (
    CONDITIONAL_PARITY,
    FREE_WIN,
    REJECT,
    Measurement,
    Threshold,
    evaluate,
)

# LPIPS is lower-is-better; element_recall/pref deltas are higher-is-better.
FAITHFUL = Measurement(metric="lpips", mean=0.20, worst=0.45)
DIVERGENT = Measurement(metric="lpips", mean=0.68, worst=0.85, worst_row="t07")
TAIL_ONLY = Measurement(metric="lpips", mean=0.20, worst=0.72, worst_row="t11")
LPIPS_BUDGET = Threshold(metric="lpips", mean_limit=0.35, tail_limit=0.60)

PARITY_OK = Measurement(metric="pref_delta", mean=0.01, worst=-0.02)
PARITY_BAD = Measurement(metric="pref_delta", mean=-0.40, worst=-0.90, worst_row="t03")
PREF_BUDGET = Threshold(metric="pref_delta", mean_limit=-0.05, tail_limit=-0.30)

# SSIM is report-only: measured and reported, but it cannot fail a candidate.
SSIM_DISASTER = Measurement(metric="ssim", mean=0.01, worst=0.0)
SSIM_BUDGET = Threshold(metric="ssim", mean_limit=0.9)

BOTH_BUDGETS = (LPIPS_BUDGET, PREF_BUDGET)

# (measurements, budgets, verdict, faithfulness_passed, parity_passed,
#  needs_human_review, anomaly, substrings that must appear in the reasons)
CASES = [
    pytest.param(
        [FAITHFUL, PARITY_OK], BOTH_BUDGETS,
        FREE_WIN, True, True, False, "", (),
        id="faithful_is_a_free_win_needing_no_human",
    ),
    pytest.param(
        [DIVERGENT, PARITY_OK], BOTH_BUDGETS,
        CONDITIONAL_PARITY, False, True, True, "", ("lpips_worst", "t07"),
        id="divergent_but_parity_holds_is_conditional_and_needs_a_human",
    ),
    pytest.param(
        [DIVERGENT, PARITY_BAD], BOTH_BUDGETS,
        REJECT, False, False, False, "", ("t07", "t03"),
        id="divergent_and_degraded_is_rejected_without_review",
    ),
    pytest.param(
        [TAIL_ONLY, PARITY_OK], BOTH_BUDGETS,
        CONDITIONAL_PARITY, False, True, True, "", ("t11",),
        id="a_tail_breach_alone_is_enough_to_diverge",
    ),
    pytest.param(
        [FAITHFUL, PARITY_BAD], BOTH_BUDGETS,
        FREE_WIN, True, False, False, "faithful_but_parity_degraded", ("t03",),
        id="faithful_but_parity_degraded_is_flagged_not_silently_resolved",
    ),
    pytest.param(
        [FAITHFUL, SSIM_DISASTER], (LPIPS_BUDGET, SSIM_BUDGET),
        FREE_WIN, True, None, False, "", (),
        id="a_report_only_metric_never_gates",
    ),
]


@pytest.mark.parametrize(
    "measurements,budgets,verdict,faithful,parity,needs_human,anomaly,reasons", CASES,
)
def test_the_verdict_truth_table(
    measurements: list, budgets: tuple, verdict: str,
    faithful: bool, parity: bool | None, needs_human: bool,
    anomaly: str, reasons: tuple[str, ...],
) -> None:
    """One row per rule. `faithful` separates "the render matched" from
    `parity`, "it fulfilled the request equally well" — the second can rescue
    the first into a human-review item, but only on the terms below."""
    v = evaluate(measurements, budgets)
    assert v.verdict == verdict
    assert v.faithfulness_passed is faithful
    assert v.parity_passed is parity
    assert v.needs_human_review is needs_human
    assert v.anomaly == anomaly
    if anomaly:
        assert "ANOMALY" in v.note
    all_reasons = " ".join(v.faithfulness_reasons + v.parity_reasons)
    for fragment in reasons:
        assert fragment in all_reasons, fragment
    if faithful:
        assert not v.faithfulness_reasons


def test_every_verdict_is_reachable() -> None:
    """Guards the bug this module was written around: a policy where one of the
    three states can never occur is a two-state policy with extra words. The
    table above must actually produce all three."""
    reached = {case.values[2] for case in CASES}
    assert {FREE_WIN, CONDITIONAL_PARITY, REJECT} <= reached


def test_unmeasured_parity_cannot_rescue_a_divergent_candidate() -> None:
    """SAFETY-CRITICAL. 'Parity was never checked' is not 'parity holds'.

    Without this rule every run that has no judge attached would silently
    upgrade from reject to a human-review item.
    """
    v = evaluate([DIVERGENT], BOTH_BUDGETS)
    assert v.verdict == REJECT
    assert v.parity_passed is None  # unmeasured, NOT False
    assert "NOT MEASURED" in v.note


def test_a_measured_parity_metric_with_no_threshold_is_not_a_check() -> None:
    """SAFETY-CRITICAL. A number with no budget to judge it against is not
    evidence — otherwise merely computing pref_delta un-gates the candidate."""
    v = evaluate([DIVERGENT, PARITY_OK], (LPIPS_BUDGET,))  # no pref budget
    assert v.verdict == REJECT
    assert v.parity_passed is None


def test_direction_follows_the_registry_not_the_caller() -> None:
    """higher_is_better decides floor-vs-ceiling, so a caller never has to
    remember which way LPIPS runs."""
    assert registry.BY_NAME["lpips"].higher_is_better is False
    assert registry.BY_NAME["pref_delta"].higher_is_better is True
    # A high LPIPS breaches a ceiling; a low pref_delta breaches a floor.
    assert evaluate([DIVERGENT], (LPIPS_BUDGET,)).faithfulness_reasons
    assert evaluate([FAITHFUL, PARITY_BAD], BOTH_BUDGETS).parity_reasons


def test_measurements_accept_a_dict_as_well_as_a_list() -> None:
    assert evaluate({"lpips": FAITHFUL}, (LPIPS_BUDGET,)).verdict == FREE_WIN
