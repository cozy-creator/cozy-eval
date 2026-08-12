"""Track-stability mode: ONE verdict on whether objects hold together in 3D.

STABILITY: locked core (v0.x) for :func:`track_verdict`'s call shape and the
:class:`TrackVerdict` it returns.

WHY THIS EXISTS. The owner rejected sparse-attention k16 renders that every
family in this library had passed: "each static frame looks roughly correct, but
objects lose their coherence across frames... it warbles and reshapes itself."
The fine-detail detectors read one frame at a time. The temporal-fidelity family
reads whole-frame flow statistics over sampled pairs — a summary in which one
object's surface reshaping under an otherwise-correct global motion averages
away. The VLM reads an ordered strip of stills. None of the three follows a
point on an object through time, so none of them could have caught this, and the
band they agreed on was not evidence of quality. :mod:`cozy_eval.metrics.tracks`
is the instrument that does follow the point.

THE GATE IS THE PAIRED RATIO, deliberately. ``track_stability`` reference-free
is content-set — steam and dense weave are untrackable by any sparse tracker, so
a low absolute number means "hard scene" at least as often as "bad render". The
number that separates arms is ``track_stability_ratio``, candidate over a
same-seed control, and it is valid across a re-rolled take because both sides
are computed on their OWN clip and never compared pixel to pixel.

AND THE GATE IS TWO-SIDED (@9). Shipping it floor-only was a hole the very next
labeled arm walked through: the 15 s courier rung scored **2.169** — 2.17x the
control's track stability — and PASSED, while it had dropped the cargo bike and
the parcel the prompt asked for. Scoring far above the control is not better
coherence, it is a simpler take being easier to track, and one-sidedness is
endemic to this whole metric class (CD-FVD measured that motion-free video
LOWERS FVD by 31.6-54.8%; four of VBench's six custom dimensions are maximized
by a frozen clip). Both bounds come from the same labeled sets — see
:data:`cozy_eval.metrics.tracks.STABILITY_RATIO_FLOOR` and
:data:`~cozy_eval.metrics.tracks.STABILITY_RATIO_CEILING`.

Tri-state, like every verdict here: ``pass`` / ``reject`` / ``unmeasured``, and
UNMEASURED is never silently a pass. An untrackable REFERENCE is the commonest
unmeasured case and it is the honest one — a ratio of two numbers each computed
on nine surviving points is noise wearing a verdict's clothes.
"""

from __future__ import annotations

import time
from typing import Any

import msgspec

from .metrics import tracks as track_metrics

PASS = "pass"
REJECT = "reject"
UNMEASURED = "unmeasured"

#: Appended to every PASS. A green track-stability line says objects held their
#: shape through motion; it says nothing about what those objects LOOK like.
SCOPE_NOTE = (
    "SCOPE: track stability catches object warble and 3D-inconsistency under "
    "motion — points on an object that jitter, reshape, or stop being "
    "themselves as the camera moves. It is blind to per-frame damage (melted "
    "faces, pseudo-glyphs, halos: the detail detectors), to content adherence "
    "(the checklist + VLM), and to whole-clip shimmer (warp_error). A clean "
    "ratio here is one axis, not a quality verdict."
)

#: Appended to an OVER-SMOOTHING reject (@9). Scoring far ABOVE the control is
#: not a quality finding on its own axis — it says the arm's take is easier to
#: track than the control's, which is what a motion-collapsed or re-rolled
#: simpler render looks like. What it CANNOT say is what the arm dropped.
OVERSMOOTHING_NOTE = (
    "OVER-SMOOTHING is measured as excess trackability, not as motion. The "
    "companion instrument you would expect — motion_mag_ratio, the flow "
    "magnitude the candidate holds against the control — was MEASURED not to "
    "separate this class at any pooling (@9 calibration: 1.006 top-5%-pooled, "
    "0.920 whole-frame, both inside the owner-identical band 0.80-1.05). What "
    "an over-smoothed arm loses is CONTENT, and the instrument for that is the "
    "windowed element_recall, gated on its worst window."
)

#: The reference-free number cannot gate, and a report must say why rather than
#: leave a reader to assume a low value is a defect.
CONTENT_NOTE = (
    "track_stability is CONTENT-DEPENDENT reference-free: steam, water, molten "
    "glass and dense repetitive weave hold few tracks however good the render "
    "is (measured: 0.02-0.04 on the glassblower cell's own clean control, "
    "0.33-0.49 on busker/carpet). Only the paired ratio against a same-seed "
    "control gates."
)


class TrackVerdict(msgspec.Struct, kw_only=True):
    """One clip's track-stability answer and the numbers behind it."""

    verdict: str = UNMEASURED
    defects: list[str] = msgspec.field(default_factory=list)
    measured: dict[str, float] = msgspec.field(default_factory=dict)
    unmeasured: dict[str, str] = msgspec.field(default_factory=dict)
    notes: list[str] = msgspec.field(default_factory=list)
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        """True only on an explicit PASS — UNMEASURED is not a pass."""
        return self.verdict == PASS

    def summary(self) -> str:
        head = f"track stability {self.verdict.upper()}"
        if self.defects:
            head += " — " + "; ".join(self.defects)
        ratio = self.measured.get("track_stability_ratio")
        tail = f", ratio {ratio:.3f}" if ratio is not None else ""
        return (f"{head} (track_stability "
                f"{self.measured.get('track_stability', float('nan')):.3f}{tail})")


def track_verdict(
    frames: Any,
    reference: Any = None,
    *,
    ratio_floor: float = track_metrics.STABILITY_RATIO_FLOOR,
    ratio_ceiling: float = track_metrics.STABILITY_RATIO_CEILING,
    trackability_floor: float = track_metrics.TRACKABILITY_FLOOR,
    ceiling_trackability_floor: float = track_metrics.CEILING_TRACKABILITY_FLOOR,
    windows: int = track_metrics.TRACK_WINDOWS,
    window: int = track_metrics.TRACK_WINDOW,
    target_h: int = track_metrics.TRACK_TARGET_H,
    points: int = track_metrics.TRACK_POINTS,
) -> TrackVerdict:
    """Does this clip's content hold together as a 3D scene through motion?

    ``frames`` / ``reference`` are clips in any form
    :func:`cozy_eval.metrics.temporal.as_frames` accepts. They need NOT be
    frame-aligned or the same size — each arm is scored on its own trajectories.
    Without ``reference`` the numbers are reported and the verdict is
    UNMEASURED, because the absolute level is content-set.
    """
    t0 = time.monotonic()
    result = TrackVerdict()
    kw = {"windows": windows, "window": window, "target_h": target_h, "points": points}

    try:
        cand = track_metrics.track_stats(frames, **kw)
    except ImportError as exc:
        result.unmeasured["track_stability"] = (
            f"opencv not installed (pip install 'cozy-eval[video]'): {exc}"
        )
        result.seconds = round(time.monotonic() - t0, 3)
        return result
    except Exception as exc:
        result.unmeasured["track_stability"] = f"{type(exc).__name__}: {exc}"
        result.seconds = round(time.monotonic() - t0, 3)
        return result

    result.measured.update(cand.as_dict())

    if reference is None:
        result.unmeasured["track_stability_ratio"] = (
            "no reference clip: the gate is the paired ratio against a same-seed "
            "control. Render one to gate this axis."
        )
        result.notes.append(CONTENT_NOTE)
        result.seconds = round(time.monotonic() - t0, 3)
        return result

    try:
        ref = track_metrics.track_stats(reference, **kw)
    except Exception as exc:
        result.unmeasured["track_stability_ratio"] = (
            f"the reference clip could not be tracked: {type(exc).__name__}: {exc}"
        )
        result.seconds = round(time.monotonic() - t0, 3)
        return result

    result.measured["track_stability_ref"] = ref.track_stability
    result.measured["track_survival_ref"] = ref.track_survival

    if ref.track_survival < trackability_floor or ref.track_stability <= 1e-9:
        result.unmeasured["track_stability_ratio"] = (
            f"UNTRACKABLE REFERENCE: the control clip itself holds only "
            f"{ref.track_survival:.1%} of its tracks (floor "
            f"{trackability_floor:.0%}) — steam, water, molten glass and dense "
            f"weave defeat sparse tracking, and a ratio between two numbers "
            f"computed on a handful of survivors is noise, not a verdict"
        )
        result.seconds = round(time.monotonic() - t0, 3)
        return result

    ratio = cand.track_stability / ref.track_stability
    result.measured["track_stability_ratio"] = ratio
    if ref.track_survival > 1e-9:
        result.measured["track_survival_ratio"] = cand.track_survival / ref.track_survival

    if ratio < ratio_floor:
        parts = []
        if cand.track_survival < ref.track_survival * 0.9:
            parts.append(
                f"tracks die: survival {cand.track_survival:.2f} vs the control's "
                f"{ref.track_survival:.2f}"
            )
        if cand.track_jitter == cand.track_jitter and ref.track_jitter == ref.track_jitter:
            if cand.track_jitter > ref.track_jitter:
                parts.append(
                    f"trajectories jitter: {cand.track_jitter:.3f} vs "
                    f"{ref.track_jitter:.3f}"
                )
            if cand.track_rigidity_error > ref.track_rigidity_error:
                parts.append(
                    f"neighbours disagree: rigidity error "
                    f"{cand.track_rigidity_error:.3f} vs "
                    f"{ref.track_rigidity_error:.3f}"
                )
        result.defects.append(
            f"OBJECT WARBLE: track_stability_ratio {ratio:.3f} < floor "
            f"{ratio_floor} — the candidate retains {ratio:.0%} of the control's "
            f"coherent tracks"
            + (" (" + "; ".join(parts) + ")" if parts else "")
        )
        result.verdict = REJECT
    elif ratio > ratio_ceiling and ref.track_survival < ceiling_trackability_floor:
        result.unmeasured["track_stability_ratio_ceiling"] = (
            f"OVER-SMOOTHING UNMEASURED: ratio {ratio:.3f} is above the ceiling "
            f"{ratio_ceiling}, but the control holds only "
            f"{ref.track_survival:.1%} of its tracks (ceiling needs "
            f"{ceiling_trackability_floor:.0%}) — a small denominator inflates "
            f"the upward tail, and a labeled owner-IDENTICAL pair on content "
            f"this marginal reads 2.766. The FLOOR still gated this pair"
        )
        result.verdict = PASS
        result.notes.append(SCOPE_NOTE)
    elif ratio > ratio_ceiling:
        parts = []
        if cand.motion_magnitude < ref.motion_magnitude:
            parts.append(
                f"and moves less: motion_magnitude {cand.motion_magnitude:.3f} vs "
                f"the control's {ref.motion_magnitude:.3f}"
            )
        if (cand.track_jitter == cand.track_jitter          # NaN test
                and ref.track_jitter == ref.track_jitter
                and cand.track_jitter < ref.track_jitter):
            parts.append(
                f"trajectories are smoother than the control's: "
                f"{cand.track_jitter:.3f} vs {ref.track_jitter:.3f}"
            )
        result.defects.append(
            f"OVER-SMOOTHING: track_stability_ratio {ratio:.3f} > ceiling "
            f"{ratio_ceiling} — the candidate holds {ratio:.0%} of the control's "
            f"coherent tracks, which is not better coherence but a simpler, "
            f"slower take"
            + (" (" + "; ".join(parts) + ")" if parts else "")
        )
        result.notes.append(OVERSMOOTHING_NOTE)
        result.verdict = REJECT
    else:
        result.verdict = PASS
        result.notes.append(SCOPE_NOTE)
    result.seconds = round(time.monotonic() - t0, 3)
    return result


__all__ = [
    "CONTENT_NOTE",
    "OVERSMOOTHING_NOTE",
    "PASS",
    "REJECT",
    "SCOPE_NOTE",
    "UNMEASURED",
    "TrackVerdict",
    "track_verdict",
]
