"""Image lane — the degenerate single-frame case of the same machinery.

An image gate is a video gate with one frame: the imaging benchmark and the
distributional benchmark apply unchanged, the temporal benchmark does not exist.
The two-lane rule is identical and matters just as much — a same-seed LPIPS
against a bf16 render is valid for a VAE-decode change and invalid for a weight
quantization, on images exactly as on video. It is only *less obviously* invalid
on images, because a single 1024² frame diverges less visibly than 121 frames do.

This surface exists so an existing image degradation gate can migrate onto the
library incrementally: score images, get the same ``GateReport``, keep whatever
reference-metric budgets it already validated for its own post-latent lane.
"""

from __future__ import annotations

from .benchmarks import VIDEO_BUDGETS, Budget, distributional, imaging
from .control import NullControl
from .gate import GateReport, Verdict, run_reference_gate  # noqa: F401
from .metrics.signal import ClipScore
from .metrics.signal import score as _score
from .protocol import Lane, Protocol, ProtocolError

#: Image budgets reuse the video imaging/distributional calibration. They are
#: PROVISIONAL for images, and *measurably* so: on two image families, zero-change
#: null controls read ``population_frechet`` 0.4688 and 1.3275 against a 0.20
#: budget, and one read ``imaging_worst_prompt`` 0.6102 against 0.85. Run a
#: :func:`~cozy_eval.control.measure_null_control` arm and pass it to
#: :func:`run_image_population_gate` — on images that is not optional advice,
#: it is the difference between a verdict and a number.
IMAGE_BUDGETS: tuple[Budget, ...] = tuple(
    b for b in VIDEO_BUDGETS if b.name not in ("jerk_excess", "flicker_ratio")
)


def score_image(source) -> ClipScore:
    """Score a single image (or anything :func:`~cozy_eval.frames.iter_frames` accepts)."""
    s = _score(source)
    if s.n_frames != 1:
        raise ValueError(f"score_image got {s.n_frames} frames; use the video surface")
    return s


def run_image_population_gate(
    pairs: list[tuple[ClipScore, ClipScore]],
    protocol: Protocol,
    *,
    budgets=IMAGE_BUDGETS,
    null_control: NullControl | None = None,
) -> GateReport:
    """Population gate without the temporal benchmark.

    Pass a ``null_control`` (see :mod:`cozy_eval.control`). The image
    budgets are inherited from video calibration and two of them are measured
    not to transfer; without a control this gate can return a confident FAIL
    that a zero-change arm would also have earned.
    """
    if protocol.lane is not Lane.POPULATION:
        raise ProtocolError("post-latent change: use run_reference_gate")
    from .gate import _control_faults, _degenerate_report, _protocol_faults
    degenerate = _degenerate_report(pairs, protocol)
    if degenerate is not None:
        return degenerate
    untrusted = null_control.untrusted if null_control else None
    reasons = _protocol_faults(protocol, len(pairs)) + _control_faults(null_control, protocol)
    results = {
        "imaging": imaging(pairs, budgets, untrusted),
        "distributional": distributional(pairs, budgets, untrusted),
    }
    failed = [k for k, v in results.items() if v.passed is False]
    if failed:
        reasons.insert(0, "failed benchmarks: " + ", ".join(failed))
        verdict = Verdict.FAIL
    else:
        verdict = Verdict.INDETERMINATE if reasons else Verdict.PASS
    return GateReport(verdict, protocol.lane, protocol, results, {}, reasons,
                      null_control=null_control)


__all__ = ["IMAGE_BUDGETS", "run_image_population_gate", "score_image"]
