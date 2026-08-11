"""Track stability: do OBJECTS hold together in 3D as the camera moves?

STABILITY: experimental (v0.x) for these function signatures; the metric NAMES
(``track_stability``, ``track_survival``, ``track_jitter``,
``track_rigidity_error``, ``track_stability_ratio``) are locked by the registry.

WHY THIS EXISTS. The owner rejected a sparse-attention arm whose every frame
looked right: "objects lose their coherence across frames... suppose object-A is
at position X,Y and we move the camera slightly; the object should move on the
frame correctly as you would expect when moving through 3-dimensional space.
Instead it warbles and reshapes itself." Three metric families PASSED those
clips — the fine-detail detectors (@5, per FRAME), the coarse temporal family
(@6, whole-frame flow statistics over decimated frame pairs) and the VLM strip
read (an ordered strip of stills). None of them follows a POINT ON AN OBJECT
through time, which is the only place that failure lives.

THE INSTRUMENT. Seed corner features, track them frame to frame with pyramidal
Lucas-Kanade, forward-backward validated, and ask three questions of each
TRAJECTORY:

* **temporal smoothness** — a point on a rigid object moving through a 3D scene
  traces a SMOOTH curve on the image plane, whatever the camera does. Warble is
  high-frequency: it shows up as second-derivative (acceleration) energy that
  genuine 3D motion does not have. Normalized by each track's own speed, so a
  fast pan is not penalized for moving fast.
* **survival** — a point that reshapes stops matching itself. Tracks on a
  warbling object die or teleport; the forward-backward check catches both.
* **local rigidity** — neighbouring points on one surface keep their relative
  geometry (up to a smooth scale/perspective change). Warble breaks that
  agreement, and it breaks it JERKILY: the second derivative of the normalized
  neighbour distance is the number, so genuine parallax and perspective — which
  are smooth — do not fire.

WHY LUCAS-KANADE AND NOT A LEARNED TRACKER. CoTracker is the obvious candidate
and it is **CC-BY-NC-4.0** — the same non-commercial bar that got pyiqa dropped
at 0.1.16 and keeps DOVER/FAST-VQA unwrapped. It is not installable here at any
quality. Of the permissive learned trackers, TAP-Net/TAPIR is Apache-2.0 but
adds a JAX/torch checkpoint download and seconds-per-clip of GPU inference to a
library whose whole temporal tier is currently 24 Farneback flow fields on CPU.
Pyramidal LK is BSD, already installed (it is the same ``opencv`` the flow
family uses — this family adds NO dependency), deterministic, and costs
~1 second per clip on four idle threads. It is a weaker tracker than CoTracker on
occlusion and large displacement; that weakness is bounded by the trackability
floor below, and the labeled separation it achieves is measured, not asserted.

THE TRACKABILITY FLOOR IS THE HONEST PART. Steam, molten glass, water and dense
repetitive weave are untrackable by ANY sparse tracker: the loom cell's own
clean control holds 13-30% of its tracks and its numbers swing by a factor of
five between two renders of the SAME arm. When the reference clip's own survival
is below :data:`TRACKABILITY_FLOOR`, this family returns UNMEASURED. A number computed on
nine surviving points is noise, and reporting it as a verdict is how a detector
starts lying.

FRAME CONVENTION: ``(T, H, W, 3)`` RGB, uint8 or float [0, 1], or a list of PIL
frames — whatever :func:`cozy_eval.metrics.temporal.as_frames` accepts.
"""

from __future__ import annotations

from typing import Any

import msgspec

from ..errors import ConfigError
from .temporal import _is_255_scale, _work_size, normalize_frame, stacked_frames

TRACK_LIBRARY = "cozy-eval:tracks"

#: Working HEIGHT the tracking runs at. Width follows the aspect ratio, even.
#: The same 384 the flow family uses, for one reason: both instruments then see
#: the same scale of motion and their numbers read on one scale.
TRACK_TARGET_H = 384

#: How many CONSECUTIVE-frame windows are tracked, spread over the clip. This is
#: the decimation knob — cost is linear in ``windows * window``. Tracking cannot
#: skip frames the way the flow family samples pairs: a chained tracker needs
#: consecutive frames, so the clip is sampled in WINDOWS rather than in frames.
TRACK_WINDOWS = 4

#: Frames per window. 24 = one second at 24 fps: long enough for warble to
#: accumulate into the second derivative, short enough that a clean track
#: survives it.
TRACK_WINDOW = 24

#: Corner features seeded per window.
TRACK_POINTS = 400

#: Forward-backward re-projection tolerance, working-resolution px. A track that
#: does not come back to where it started is a teleport, not a track (Kalal's
#: FB-error criterion).
TRACK_FB_TOL = 1.0

#: Neighbours per track for the local-rigidity test.
TRACK_NEIGHBOURS = 6

#: Speed below this (working px/frame) is treated as this, so a nearly-static
#: point cannot divide its jitter by ~0.
SPEED_FLOOR = 0.25

#: Soft knee for per-track normalized acceleration: a track here scores 0.5 on
#: the smoothness half. SOFT on purpose — a hard threshold makes the family a
#: step function of two constants and small real degradations then move nothing.
#: MEASURED on the trackable labeled corpus: clean controls median 0.156,
#: rejected sparse arms median 0.236. NOT knife-edge in this constant: a
#: development sweep over J0 in {0.06, 0.08, 0.10, 0.15} x R0 in {0.5, 0.8} left
#: the rejected and identical populations disjoint at every setting, and the
#: separation only starts to close at J0 >= 0.20.
JITTER_KNEE = 0.10

#: Soft knee for per-track neighbour-rigidity jerk (x100). MEASURED on the same
#: corpus: clean controls median 0.366, rejected sparse arms median 0.571.
RIGIDITY_KNEE = 0.8

#: A clip whose own tracks do not survive is UNTRACKABLE CONTENT, not an
#: unstable render, and this family says UNMEASURED rather than guessing.
#: MEASURED: the loom (dense weave) and glassblower (steam/molten) cells hold
#: 13-30% and 18-32% of their tracks on their CLEAN controls, while the
#: trackable cells hold 83-96%. The floor is fixed by the FALSE POSITIVE it
#: prevents, not by taste: at 0.15 the loom seed-A pair — which the owner
#: reviewed as identical — reads 0.10-0.55 across its three clean arms, two of
#: which would be called catastrophic rejects.
TRACKABILITY_FLOOR = 0.25

#: THE GATE, paired. ``track_stability_ratio`` = candidate / reference. MEASURED
#: separation on the owner-labeled H3 corpus, not a guess: 29 measurable
#: owner-REJECTED sparse-attention pairs score 0.029-0.846 (median 0.366) while
#: 12 measurable pairs the owner judged IDENTICAL (SageAttention-2 fp8 vs
#: FA3-exact, plus same-arm re-renders across a pod change and a torch-line
#: change) score 0.930-1.251 (median 1.050). 0.90 sits in an 8-point empty
#: middle. See ``calibration/track-stability.json``.
STABILITY_RATIO_FLOOR = 0.90

_LK = {
    "winSize": (21, 21),
    "maxLevel": 3,
    "criteria": (3, 30, 0.01),  # TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT
}


class TrackStats(msgspec.Struct, frozen=True, kw_only=True):
    """One clip's track-stability numbers.

    ``track_jitter`` / ``track_rigidity_error`` are NaN when no window kept
    enough tracks to compute a trajectory shape — the clip is untrackable, and
    :data:`TRACKABILITY_FLOOR` is the test a caller should apply before quoting
    ``track_stability``.
    """

    track_stability: float
    track_survival: float
    track_jitter: float
    track_rigidity_error: float
    motion_magnitude: float
    n_seeded: int
    n_windows: int

    @property
    def trackable(self) -> bool:
        return self.track_survival >= TRACKABILITY_FLOOR

    def as_dict(self) -> dict[str, float]:
        """The registry-named numbers, dropping the ones that are NaN."""
        out = {
            "track_stability": self.track_stability,
            "track_survival": self.track_survival,
            "track_jitter": self.track_jitter,
            "track_rigidity_error": self.track_rigidity_error,
        }
        return {k: v for k, v in out.items() if v == v}


# ---------------------------------------------------------------------------
# the tracker
# ---------------------------------------------------------------------------

def window_starts(total: int, windows: int, window: int) -> list[int]:
    """Start indices of ``windows`` consecutive-frame windows spread over the clip."""
    if total < 3:
        raise ConfigError(
            f"track stability needs at least 3 frames, got {total} — a second "
            "derivative is undefined on two"
        )
    span = min(window, total)
    last = total - span
    if last <= 0 or windows <= 1:
        return [0]
    step = last / (windows - 1)
    return sorted({round(i * step) for i in range(windows)})


def gray_ladder(frames: Any, indices: list[int], *,
                target_h: int = TRACK_TARGET_H) -> dict[int, Any]:
    """Working-resolution uint8 grey planes for JUST ``indices``.

    A uint8 clip is resized straight from uint8 — the float round-trip
    :func:`~cozy_eval.metrics.temporal.as_frames` would do costs two full copies
    of every touched frame and buys nothing for a corner tracker.
    """
    import numpy as np

    from ..resources import opencv

    cv2 = opencv()
    stacked = stacked_frames(frames)
    h, w = stacked.shape[1:3]
    size = _work_size(h, w, target_h)
    scale_255 = _is_255_scale(stacked)
    out: dict[int, Any] = {}
    for i in indices:
        frame = stacked[i]
        if getattr(frame, "dtype", None) != np.uint8:
            frame = np.clip(
                normalize_frame(frame, scale_255=scale_255) * 255.0, 0, 255
            ).astype(np.uint8)
        small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        out[i] = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    return out


def track_window(gray: dict[int, Any], start: int, span: int, *,
                 points: int = TRACK_POINTS, fb_tol: float = TRACK_FB_TOL):
    """Chase corner features through ``span`` consecutive frames from ``start``.

    Returns ``(trajectories, alive)``: ``trajectories`` is ``(span, n, 2)`` with
    NaN after a track dies, ``alive`` is the per-track survived-to-the-end mask.
    ``None`` when the first frame has too few corners to say anything.
    """
    import numpy as np

    from ..resources import opencv

    cv2 = opencv()
    first = gray[start]
    seeds = cv2.goodFeaturesToTrack(first, maxCorners=points, qualityLevel=0.01,
                                    minDistance=8, blockSize=7)
    if seeds is None or len(seeds) < 8:
        return None
    n = len(seeds)
    traj = np.full((span, n, 2), np.nan, np.float32)
    traj[0] = seeds[:, 0, :]
    alive = np.ones(n, bool)
    cur = seeds.astype(np.float32)
    h, w = first.shape[:2]
    for i in range(1, span):
        a, b = gray[start + i - 1], gray[start + i]
        nxt, fwd_ok, _ = cv2.calcOpticalFlowPyrLK(a, b, cur, None, **_LK)
        back, bwd_ok, _ = cv2.calcOpticalFlowPyrLK(b, a, nxt, None, **_LK)
        fb = np.linalg.norm((back - cur)[:, 0, :], axis=1)
        p = nxt[:, 0, :]
        inside = (p[:, 0] >= 0) & (p[:, 0] < w - 1) & (p[:, 1] >= 0) & (p[:, 1] < h - 1)
        alive &= (fwd_ok[:, 0] == 1) & (bwd_ok[:, 0] == 1) & (fb < fb_tol) & inside
        traj[i] = np.where(alive[:, None], p, np.nan)
        cur = nxt
    return traj, alive


def similarity_fit(a: Any, b: Any, iters: int = 3):
    """Robust similarity transform ``b ~ M @ a + t``, by IRLS. DETERMINISTIC.

    Deliberately not ``cv2.estimateAffinePartial2D``: that draws RANSAC samples
    from OpenCV's process-global RNG, so scoring the SAME clip twice in one
    process can return two different numbers — and a paired ratio that is not
    exactly 1.0 on a bit-identical pair is a broken instrument.
    """
    import numpy as np

    wt = np.ones(len(a), np.float64)
    M, tvec = np.eye(2), np.zeros(2)
    for _ in range(iters):
        total = wt.sum()
        if total < 1e-9:
            break
        ca = (a * wt[:, None]).sum(0) / total
        cb = (b * wt[:, None]).sum(0) / total
        pa, pb = a - ca, b - cb
        dot = float((wt * (pa[:, 0] * pb[:, 0] + pa[:, 1] * pb[:, 1])).sum())
        cross = float((wt * (pa[:, 0] * pb[:, 1] - pa[:, 1] * pb[:, 0])).sum())
        den = float((wt * (pa ** 2).sum(1)).sum())
        if den < 1e-9:
            break
        M = np.array([[dot / den, -cross / den], [cross / den, dot / den]])
        tvec = cb - M @ ca
        res = np.linalg.norm(b - (a @ M.T + tvec), axis=1)
        sigma = max(float(np.median(res)), 0.05)
        wt = 1.0 / (1.0 + (res / (2.0 * sigma)) ** 2)
    return M, tvec


def _camera_removed(P: Any) -> Any:
    """Trajectories re-expressed in frame-0 coordinates, camera motion removed.

    Chains a per-step robust similarity fit over the whole surviving track set
    (that IS the camera, up to the scene's own consensus) and applies its
    inverse. What remains is each point's motion RELATIVE to the scene, which is
    where object warble lives — a camera that itself judders is caught by the
    raw-trajectory number instead.
    """
    import numpy as np

    Q = np.empty_like(P)
    Q[0] = P[0]
    m_cum, t_cum = np.eye(2), np.zeros(2)
    for i in range(1, len(P)):
        M, t = similarity_fit(P[i - 1], P[i])
        try:
            m_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:      # degenerate step: carry the frame through
            Q[i] = P[i] @ m_cum.T + t_cum
            continue
        t_inv = -m_inv @ t
        m_cum, t_cum = m_cum @ m_inv, m_cum @ t_inv + t_cum
        Q[i] = P[i] @ m_cum.T + t_cum
    return Q


def _neighbour_rigidity(P: Any, neighbours: int) -> Any:
    """Per-track jerk of its neighbours' distances, ``(m,)``, scale-normalized.

    ``d_ij(t) / d_ij(0)``, divided by the frame's median over all pairs (which
    absorbs a zoom), then second-differenced: a surface deforming SMOOTHLY under
    perspective reads ~0, a surface that reshapes frame to frame does not.
    """
    import numpy as np

    m = P.shape[1]
    k = min(neighbours, m - 1)
    d0 = np.linalg.norm(P[0][:, None, :] - P[0][None, :, :], axis=2)
    np.fill_diagonal(d0, np.inf)
    nb = np.argsort(d0, axis=1)[:, :k]
    ii = np.repeat(np.arange(m), k)
    jj = nb.ravel()
    dist = np.linalg.norm(P[:, ii, :] - P[:, jj, :], axis=2)
    ratio = dist / np.maximum(dist[0], 2.0)
    ratio = ratio / np.maximum(np.median(ratio, axis=1, keepdims=True), 1e-6)
    jerk = np.abs(ratio[2:] - 2 * ratio[1:-1] + ratio[:-2]).mean(0)
    return jerk.reshape(m, k).mean(1) * 100.0


def window_stats(traj: Any, alive: Any, *, neighbours: int = TRACK_NEIGHBOURS,
                 jitter_knee: float = JITTER_KNEE,
                 rigidity_knee: float = RIGIDITY_KNEE) -> dict[str, float]:
    """One window's numbers, from its trajectories.

    ``coherent`` is the SOFT count: each surviving track contributes a value in
    (0, 1] that falls off with its normalized jitter and its neighbour-rigidity
    jerk. Soft on purpose — a hard threshold makes the whole family a step
    function of two constants, and small real degradations then move nothing.
    """
    import numpy as np

    n = traj.shape[1]
    out = {"n_seeded": float(n), "n_alive": float(alive.sum()), "coherent": 0.0}
    if alive.sum() < 8:
        return out
    P = traj[:, alive, :].astype(np.float64)
    step = np.linalg.norm(P[1:] - P[:-1], axis=2)          # (span-1, m)
    out["motion"] = float(np.median(step.mean(1)))
    speed = np.maximum(np.median(step, axis=0), SPEED_FLOOR)
    Q = _camera_removed(P)
    accel = np.linalg.norm(Q[2:] - 2 * Q[1:-1] + Q[:-2], axis=2).mean(0)
    jitter = accel / speed
    rigidity = _neighbour_rigidity(P, neighbours)
    coherent = (1.0 / (1.0 + (jitter / jitter_knee) ** 2)
                * 1.0 / (1.0 + (rigidity / rigidity_knee) ** 2))
    out["coherent"] = float(coherent.sum())
    out["jitter"] = float(np.median(jitter))
    out["rigidity"] = float(np.median(rigidity))
    return out


def track_stats(frames: Any, *, windows: int = TRACK_WINDOWS,
                window: int = TRACK_WINDOW, target_h: int = TRACK_TARGET_H,
                points: int = TRACK_POINTS) -> TrackStats:
    """Reference-free track stability for one clip.

    ``track_stability`` is the fraction of seeded points that both SURVIVE their
    window and move like a point on a real object while they do — one number in
    [0, 1], where a dead track contributes 0 and a perfectly smooth, rigid one
    contributes 1. It is CONTENT-DEPENDENT (steam scores low because steam is
    untrackable, not because the render is bad), so it is a per-clip diagnostic
    reference-free and a GATE only as the paired ratio in
    :func:`track_fidelity`.
    """
    import numpy as np

    stacked = stacked_frames(frames)
    total = int(stacked.shape[0])
    starts = window_starts(total, windows, window)
    needed = sorted({i for k in starts for i in range(k, min(k + window, total))})
    gray = gray_ladder(stacked, needed, target_h=target_h)

    seeded = coherent = alive = 0.0
    jitters: list[float] = []
    rigidities: list[float] = []
    motions: list[float] = []
    scored = 0
    for k in starts:
        span = min(window, total - k)
        if span < 3:
            continue
        tracked = track_window(gray, k, span, points=points)
        if tracked is None:
            continue
        scored += 1
        w = window_stats(*tracked)
        seeded += w["n_seeded"]
        alive += w["n_alive"]
        coherent += w["coherent"]
        for key, sink in (("jitter", jitters), ("rigidity", rigidities),
                          ("motion", motions)):
            if key in w:
                sink.append(w[key])
    if not seeded:
        raise ConfigError(
            "track stability: no window had enough corner features to track — "
            "the clip carries no trackable structure at all"
        )
    med = lambda xs: float(np.median(xs)) if xs else float("nan")  # noqa: E731
    return TrackStats(
        track_stability=coherent / seeded,
        track_survival=alive / seeded,
        track_jitter=med(jitters),
        track_rigidity_error=med(rigidities),
        motion_magnitude=med(motions),
        n_seeded=int(seeded),
        n_windows=scored,
    )


def track_fidelity(reference: Any, candidate: Any, *, windows: int = TRACK_WINDOWS,
                   window: int = TRACK_WINDOW, target_h: int = TRACK_TARGET_H,
                   points: int = TRACK_POINTS) -> dict[str, float]:
    """The paired block: how much track stability the candidate arm LOST.

    ``track_stability_ratio`` = candidate / reference. A ratio and not a
    difference because the absolute level is content-set: 0.03 on a steam cell
    and 0.49 on a carpet cell are both clean, and only the fraction retained is
    comparable across cells.

    VALID FOR THE TRAJECTORY-PERTURBING LANE, which is the point. Both sides are
    computed on their OWN clip; nothing is compared pixel to pixel, so a
    same-seed arm that re-rolled the take is measured on whether its objects
    hold together, not on how far its take drifted. Exactly the property
    ``warp_error_delta`` has and ``lpips`` does not.

    The two clips do NOT have to be frame-aligned or the same size.
    """
    ref = track_stats(reference, windows=windows, window=window,
                      target_h=target_h, points=points)
    cand = track_stats(candidate, windows=windows, window=window,
                       target_h=target_h, points=points)
    out = {
        "track_stability": cand.track_stability,
        "track_stability_ref": ref.track_stability,
        "track_stability_ratio": (cand.track_stability / ref.track_stability
                                  if ref.track_stability > 1e-9 else float("nan")),
        "track_survival": cand.track_survival,
        "track_survival_ref": ref.track_survival,
        "track_survival_ratio": (cand.track_survival / ref.track_survival
                                 if ref.track_survival > 1e-9 else float("nan")),
    }
    for name, value in (("track_jitter", cand.track_jitter),
                        ("track_rigidity_error", cand.track_rigidity_error)):
        if value == value:
            out[name] = value
    return {k: v for k, v in out.items() if v == v}


__all__ = [
    "JITTER_KNEE",
    "RIGIDITY_KNEE",
    "SPEED_FLOOR",
    "STABILITY_RATIO_FLOOR",
    "TRACKABILITY_FLOOR",
    "TRACK_FB_TOL",
    "TRACK_LIBRARY",
    "TRACK_NEIGHBOURS",
    "TRACK_POINTS",
    "TRACK_TARGET_H",
    "TRACK_WINDOW",
    "TRACK_WINDOWS",
    "TrackStats",
    "gray_ladder",
    "similarity_fit",
    "track_fidelity",
    "track_stats",
    "track_window",
    "window_starts",
    "window_stats",
]
