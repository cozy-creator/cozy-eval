"""The always-available no-reference backend: elementary signal statistics.

One streaming pass over a clip produces every number the three benchmarks need:
six per-frame *imaging* features, four *temporal* statistics, and the per-frame
feature matrix the distributional test consumes. Nothing here loads a model, so
it runs on a CPU worker in seconds and is deterministic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

from ..frames import LUMA, iter_frames

#: Per-frame feature order. Log-scaled entries are stored already-logged so the
#: paired test and the feature covariance are on a sane scale.
FEATURES = ("log_sharpness", "log_hf_ratio", "detail_entropy",
            "contrast", "saturation", "log_local_contrast")

#: Imaging metrics, split into the two groups the gate weights differently.
DETAIL_METRICS = ("sharpness", "hf_ratio", "local_contrast")
TONAL_METRICS = ("contrast", "saturation", "detail_entropy")
IMAGING_METRICS = DETAIL_METRICS + TONAL_METRICS
TEMPORAL_METRICS = ("jerk_ratio", "shimmer", "flicker", "motion_energy")

#: A frame whose luma standard deviation is below this carries no spatial signal:
#: it is a constant fill (black, white, flat grey, a solid colour). Every imaging
#: statistic on such a frame is pinned at its numeric floor, so a ratio against a
#: real reference is arithmetic rather than evidence.
FLAT_FRAME_TOLERANCE = 1e-4

#: Fraction of flat frames at which the whole clip is called degenerate.
DEGENERATE_FRAME_FRACTION = 0.5


def box3(x: np.ndarray) -> np.ndarray:
    """3x3 box blur with edge replication."""
    p = np.pad(x, 1, mode="edge")
    s = p[:-2] + p[1:-1] + p[2:]
    return (s[:, :-2] + s[:, 1:-1] + s[:, 2:]) / 9.0


def laplacian_variance(lum: np.ndarray) -> float:
    """Focus measure. Falls when a model softens; rises when it injects noise."""
    k = (-4.0 * lum[1:-1, 1:-1] + lum[:-2, 1:-1] + lum[2:, 1:-1]
         + lum[1:-1, :-2] + lum[1:-1, 2:])
    return float(np.var(k * 255.0))


def hf_energy_ratio(lum: np.ndarray, cutoff: float = 0.25) -> float:
    """Fraction of spectral energy above ``cutoff`` cycles/pixel — detail retention."""
    spec = np.abs(np.fft.rfft2(lum - lum.mean())) ** 2
    yy = np.abs(np.fft.fftfreq(lum.shape[0]))[:, None]
    xx = np.fft.rfftfreq(lum.shape[1])[None, :]
    radius = np.sqrt(yy ** 2 + xx ** 2)
    return float(spec[radius > cutoff].sum() / max(float(spec.sum()), 1e-12))


def histogram_entropy(lum: np.ndarray, bins: int = 64) -> float:
    """Tonal richness in bits. Collapses when output goes flat or grey."""
    hist, _ = np.histogram(lum, bins=bins, range=(0.0, 1.0))
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


@dataclass(slots=True, eq=False)
class ClipScore:  # noqa: PLW1641 — mutable and array-valued: equality yes, hashing no
    """One clip's measured quality. Every field is no-reference."""

    n_frames: int
    width: int
    height: int
    sharpness: float
    sharpness_p10: float
    hf_ratio: float
    detail_entropy: float
    contrast: float
    saturation: float
    local_contrast: float
    brightness: float
    flicker: float
    jerk_ratio: float
    shimmer: float
    motion_energy: float
    #: How many frames carry no spatial signal at all (constant fill).
    flat_frames: int = 0
    features: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 6)))

    @property
    def is_degenerate(self) -> bool:
        """True when the clip carries no signal to measure.

        A degenerate clip is not a bad clip — it is the absence of a measurement.
        Scoring one produces confident-looking numbers (an imaging index of 0.0
        against a real reference) that mean nothing and must not be ranked.

        Two signatures: most frames flat, or — the form that survives any
        serialization, since ``contrast`` is the mean of the per-frame luma
        standard deviations — a clip-wide contrast of zero.
        """
        return self.n_frames > 0 and (
            self.contrast < FLAT_FRAME_TOLERANCE
            or self.flat_frames / self.n_frames >= DEGENERATE_FRAME_FRACTION)

    def imaging(self) -> dict[str, float]:
        return {m: getattr(self, m) for m in IMAGING_METRICS}

    def temporal(self) -> dict[str, float]:
        return {m: getattr(self, m) for m in TEMPORAL_METRICS}

    def metrics(self) -> dict[str, float]:
        """The flat scalar view — what you print in a table or a report row.

        Lossy by design (no per-frame ``features``). To persist a score, use
        :meth:`to_dict` / :meth:`from_dict`, which round-trip exactly.
        """
        d = {"n_frames": self.n_frames, "width": self.width, "height": self.height,
             "brightness": self.brightness, "sharpness_p10": self.sharpness_p10}
        d.update(self.imaging())
        d.update(self.temporal())
        return d

    # --- serialization ------------------------------------------------------ #
    # `features` is a (frames, 6) float64 array, and a naive JSON round-trip
    # restores it as a list of lists, which then fails deep inside the
    # distributional benchmark. These two methods are the supported path: they
    # round-trip every field exactly, through plain JSON types.

    def to_dict(self) -> dict[str, Any]:
        """Lossless, JSON-safe. ``from_dict(s.to_dict()) == s``."""
        d: dict[str, Any] = {f.name: getattr(self, f.name) for f in fields(self)
                             if f.name != "features"}
        d["features"] = np.asarray(self.features, np.float64).tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ClipScore:
        """Rebuild a score persisted by :meth:`to_dict` (or a ``.npz``-style mapping)."""
        names = {f.name for f in fields(cls)}
        missing = names - set(d) - {"flat_frames", "features"}
        if missing:
            raise ValueError(f"not a ClipScore mapping: missing {sorted(missing)}")
        kw = {k: v for k, v in d.items() if k in names and k != "features"}
        for k in ("n_frames", "width", "height", "flat_frames"):
            if k in kw:
                kw[k] = int(kw[k])
        for k, v in kw.items():
            if k not in ("n_frames", "width", "height", "flat_frames"):
                kw[k] = float(v)
        feats = np.asarray(d.get("features", np.zeros((0, 6))), np.float64)
        return cls(**kw, features=feats.reshape(-1, len(FEATURES)))

    def save(self, path: str | Path) -> None:
        """Write one score as JSON. Lists of scores: ``[s.to_dict() for s in scores]``."""
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: str | Path) -> ClipScore:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClipScore):
            return NotImplemented
        return all(
            np.array_equal(getattr(self, f.name), getattr(other, f.name))
            if f.name == "features" else getattr(self, f.name) == getattr(other, f.name)
            for f in fields(self)
        )


def temporal_score(source) -> dict[str, float]:
    """``flicker`` and ``jerk_ratio`` ONLY, from per-frame luma — the per-clip path.

    :func:`score` computes six full-resolution feature families per frame (an
    FFT, a Laplacian, a 64-bin histogram, a 3x3 box filter, saturation, luma
    sigma) to build the feature matrix the POPULATION lane needs. The per-clip
    video report consumes two of its scalars and pays for all of it: MEASURED at
    **82.6 s of a ~140 s CPU tier** on a 362-frame 1344x768 clip, the single
    biggest line in the bill (ce#14).

    Both scalars are functions of the luma channel alone, so this pass drops
    everything else and reproduces them EXACTLY — bit-for-bit, same accumulation
    order, pinned by a test. It is not an approximation of the old numbers.
    ``score()`` is unchanged and stays the population lane's entry point.
    """
    means: list[float] = []
    d1: list[float] = []
    d2: list[float] = []
    prev = prev2 = None
    n = 0

    for rgb in iter_frames(source):
        n += 1
        lum = rgb @ LUMA
        means.append(float(lum.mean()))
        if prev is not None:
            d1.append(float(np.abs(lum - prev).mean()))
        if prev2 is not None:
            d2.append(float(np.abs(lum - 2.0 * prev + prev2).mean()))
        prev2, prev = prev, lum

    if n == 0:
        raise ValueError("no frames decoded")

    m = np.asarray(means, np.float64)
    return {
        "flicker": float(m.std() / max(m.mean(), 1e-6) * 100.0) if n > 1 else 0.0,
        "jerk_ratio": float(np.mean(d2) / max(np.mean(d1), 1e-9)) if d2 else 0.0,
    }


def score(source) -> ClipScore:
    """Score any frame source (video path, image dir, array, iterable of frames).

    A single image is the degenerate case: temporal statistics come back 0.
    """
    rows: list[list[float]] = []
    sharp: list[float] = []
    means: list[float] = []
    d1: list[float] = []
    d2: list[float] = []
    hp_d1: list[float] = []
    hp_mag: list[float] = []
    prev = prev2 = prev_hp = None
    n = 0
    flat = 0
    width = height = 0

    for rgb in iter_frames(source):
        n += 1
        height, width = rgb.shape[0], rgb.shape[1]
        lum = rgb @ LUMA
        if float(lum.std()) < FLAT_FRAME_TOLERANCE:
            flat += 1
        hp = lum - box3(lum)
        lv = laplacian_variance(lum)
        sharp.append(lv)
        means.append(float(lum.mean()))
        rows.append([
            float(np.log(max(lv, 1e-6))),
            float(np.log(max(hf_energy_ratio(lum), 1e-12))),
            histogram_entropy(lum),
            float(lum.std()),
            float((rgb.max(axis=2) - rgb.min(axis=2)).mean()),
            float(np.log(max(float(np.abs(hp).mean()), 1e-9))),
        ])
        if prev is not None:
            d1.append(float(np.abs(lum - prev).mean()))
            hp_d1.append(float(np.abs(hp - prev_hp).mean()))
            hp_mag.append(float(np.abs(hp).mean()))
        if prev2 is not None:
            d2.append(float(np.abs(lum - 2.0 * prev + prev2).mean()))
        prev2, prev, prev_hp = prev, lum, hp

    if n == 0:
        raise ValueError("no frames decoded")

    f = np.asarray(rows, np.float64)
    m = np.asarray(means, np.float64)
    return ClipScore(
        n_frames=n, width=width, height=height,
        sharpness=float(np.mean(sharp)),
        sharpness_p10=float(np.percentile(sharp, 10)),
        hf_ratio=float(np.exp(f[:, 1]).mean()),
        detail_entropy=float(f[:, 2].mean()),
        contrast=float(f[:, 3].mean()),
        saturation=float(f[:, 4].mean()),
        local_contrast=float(np.exp(f[:, 5]).mean()),
        brightness=float(m.mean()),
        # exposure flicker: frame-mean luma wobble as a percentage of the mean
        flicker=float(m.std() / max(m.mean(), 1e-6) * 100.0) if n > 1 else 0.0,
        # jitter: second temporal difference over the first. Smooth motion keeps
        # this low whatever the motion magnitude; per-frame instability raises it.
        jerk_ratio=float(np.mean(d2) / max(np.mean(d1), 1e-9)) if d2 else 0.0,
        # shimmer: how much of the fine-detail layer is replaced each frame
        shimmer=float(np.mean(hp_d1) / max(np.mean(hp_mag), 1e-9)) if hp_d1 else 0.0,
        motion_energy=float(np.mean(d1) * 255.0) if d1 else 0.0,
        flat_frames=flat,
        features=f,
    )
