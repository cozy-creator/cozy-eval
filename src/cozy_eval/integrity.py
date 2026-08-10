"""Output integrity: the NOISE / BLANK floor every render must clear.

STABILITY: locked core (v0.x) for :func:`output_integrity`'s call shape and the
:class:`OutputIntegrity` it returns.

WHY THIS EXISTS. Production minimax-h3 0.3.8 served pure VAE-decoded NOISE on
billed, settled requests (ie#634). Nothing caught it: the "proof" banked ffprobe
metadata and a billing row, and nobody looked at pixels. Metadata is not pixels.
This is the cheapest possible instrument that would have caught it — median
adjacent-frame grey correlation over a handful of pairs spread across the clip,
plus a per-frame contrast floor for blank output. numpy only, no reference, no
model. MEASURED cost on a 121-frame 1344x768 uint8 clip (0.37 GB of pixels):
**8.3 ms**, because only ~10 frames are touched and each is decimated before the
float conversion. It is meant to run on EVERY render, including the serve path
before upload — at that price there is no run it is not worth doing.

THE FLOOR IS MEASURED, not guessed: VAE-decoded noise sits at 0.29, real renders
at 0.92-0.99. ``NOISE_CORR_FLOOR`` = 0.6 sits in that empty middle. The MEDIAN
over a spread of pairs is what makes a hard CUT safe — a cut drives one pair's
correlation to ~0 while the rest stay high.

SCOPE HONESTY — READ THIS BEFORE QUOTING A PASS. This gate catches NOISE and
BLANK. It is NOT a quality gate, and it is specifically blind to the fp8-melt
class: smearing REMOVES high-frequency temporal variation, so a melted render
scores HIGHER than a clean one (measured: melted arm 0.956 vs clean 0.916; a
blurred control here reproduces the same inversion). Fine-detail damage belongs
to :mod:`cozy_eval.detail`'s detectors and the VLM rubric; motion damage belongs
to the temporal-fidelity family. Three axes, no one of them sufficient.

Tri-state, like every verdict here: ``pass`` / ``reject`` / ``unmeasured``, and
UNMEASURED is never silently a pass.
"""

from __future__ import annotations

import time
from typing import Any

import msgspec

from .metrics import temporal

PASS = "pass"
REJECT = "reject"
UNMEASURED = "unmeasured"

#: Appended to every PASS so a green integrity line can never be read as a
#: quality pass. The melt class scores HIGHER here than a clean render.
SCOPE_NOTE = (
    "SCOPE: the integrity floor catches NOISE and BLANK output only. It is not "
    "a quality gate — a melted/over-smoothed render scores HIGHER on "
    "adjacent_frame_corr than a clean one, so fine-detail damage needs the "
    "detail detectors + VLM rubric and motion damage needs the "
    "temporal-fidelity family."
)


class OutputIntegrity(msgspec.Struct, kw_only=True):
    """One clip's integrity answer, and the two numbers behind it."""

    verdict: str = UNMEASURED
    defects: list[str] = msgspec.field(default_factory=list)
    adjacent_frame_corr: float = float("nan")
    frame_std_min: float = float("nan")
    corr_series: list[float] = msgspec.field(default_factory=list)
    notes: list[str] = msgspec.field(default_factory=list)
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        """True only on an explicit PASS — UNMEASURED is not a pass."""
        return self.verdict == PASS

    def summary(self) -> str:
        head = f"integrity {self.verdict.upper()}"
        if self.defects:
            head += " — " + "; ".join(self.defects)
        return (f"{head} (adjacent_frame_corr {self.adjacent_frame_corr:.3f}, "
                f"frame_std_min {self.frame_std_min:.4f})")


def output_integrity(
    frames: Any,
    *,
    corr_floor: float = temporal.NOISE_CORR_FLOOR,
    std_floor: float = temporal.BLANK_STD_FLOOR,
    pairs: int = temporal.INTEGRITY_PAIRS,
) -> OutputIntegrity:
    """Reject a clip that is noise or blank. Milliseconds, no reference.

    ``frames`` is any clip form :func:`cozy_eval.metrics.temporal.as_frames`
    accepts, but only the ~``2 * pairs`` sampled frames are ever touched, so the
    cost does not grow with clip length. A clip too short to have an adjacent
    pair comes back UNMEASURED, never a pass.
    """
    t0 = time.monotonic()
    result = OutputIntegrity()
    try:
        stats = temporal.integrity_stats(frames, pairs=pairs)
    except Exception as exc:  # an unscoreable clip is UNMEASURED, never a pass
        result.notes.append(f"INTEGRITY UNMEASURED: {exc}")
        result.seconds = round(time.monotonic() - t0, 4)
        return result

    result.corr_series = [float(v) for v in stats["corr_series"]]
    result.adjacent_frame_corr = float(stats["adjacent_frame_corr"])
    result.frame_std_min = float(stats["frame_std_min"])

    if result.frame_std_min < std_floor:
        result.defects.append(
            f"BLANK: a sampled frame carries no spatial signal "
            f"(std {result.frame_std_min:.5f} < {std_floor})"
        )
    if result.adjacent_frame_corr < corr_floor:
        result.defects.append(
            f"NOISE: median adjacent-frame correlation "
            f"{result.adjacent_frame_corr:.3f} < floor {corr_floor} — consecutive "
            f"frames are unrelated, which is what VAE-decoded noise looks like"
        )

    result.verdict = REJECT if result.defects else PASS
    if result.verdict == PASS:
        result.notes.append(SCOPE_NOTE)
    result.seconds = round(time.monotonic() - t0, 4)
    return result


__all__ = [
    "PASS",
    "REJECT",
    "SCOPE_NOTE",
    "UNMEASURED",
    "OutputIntegrity",
    "output_integrity",
]
