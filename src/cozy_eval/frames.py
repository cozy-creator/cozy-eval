"""Frame access. One streaming decoder, so a 241-frame 1080p clip costs O(1) RAM.

Decode backend order: ``torchcodec`` / ``av`` / ``imageio`` if the caller already
has one installed, else the ``ffmpeg`` CLI. We do not vendor a decoder — but we
do not require one either, because the CLI is present wherever these renders are
produced and the base install must stay numpy-only.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from .errors import DecodeError
from .resources import ffmpeg_thread_args

LUMA = np.array([0.299, 0.587, 0.114], np.float32)

Frame = np.ndarray          # (H, W, 3) float32 in [0, 1]
FrameSource = "str | Path | Iterable[Frame] | np.ndarray"


def probe(path: str | Path) -> tuple[int, int, float, int]:
    """(width, height, fps, nb_frames) via ffprobe."""
    try:
        raw = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DecodeError(f"ffprobe failed on {path}: {exc}") from exc
    streams = [s for s in json.loads(raw)["streams"] if s["codec_type"] == "video"]
    if not streams:
        raise DecodeError(f"no video stream in {path}")
    s = streams[0]
    num, _, den = str(s.get("r_frame_rate", "0/1")).partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    return int(s["width"]), int(s["height"]), fps, int(s.get("nb_frames") or 0)


def iter_video(path: str | Path) -> Iterator[Frame]:
    """Yield every frame of a video as float32 RGB in [0, 1]."""
    w, h, _, _ = probe(path)
    stride = w * h * 3
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "quiet", *ffmpeg_thread_args(),
         "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
    )
    try:
        while True:
            buf = proc.stdout.read(stride)
            if len(buf) < stride:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3).astype(np.float32) / 255.0
    finally:
        proc.stdout.close()
        proc.wait()


def iter_frames(source) -> Iterator[Frame]:
    """Normalize any accepted frame source into a stream of float RGB frames.

    Accepts a path to a video, a path to a directory of images, an ``(T,H,W,3)``
    array, or any iterable of frames. Producers that already hold decoded frames
    in memory (the conversion worker does, pre-encode) should pass them directly
    and never touch a codec.
    """
    if isinstance(source, np.ndarray):
        arr = source
        if arr.ndim == 3:
            arr = arr[None]
        for f in arr:
            yield _as_rgb01(f)
        return
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_dir():
            for img in sorted(p.iterdir()):
                if img.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                    yield _load_image(img)
            return
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            yield _load_image(p)
            return
        yield from iter_video(p)
        return
    if isinstance(source, Iterable):
        for f in source:
            yield _as_rgb01(np.asarray(f))
        return
    raise TypeError(f"unsupported frame source: {type(source)!r}")


def _as_rgb01(f: np.ndarray) -> Frame:
    f = np.asarray(f)
    if f.dtype == np.uint8:
        f = f.astype(np.float32) / 255.0
    else:
        f = f.astype(np.float32)
        if f.max(initial=0.0) > 1.5:      # 0-255 floats
            f = f / 255.0
    if f.ndim == 2:
        f = np.repeat(f[..., None], 3, axis=2)
    if f.shape[-1] == 4:
        f = f[..., :3]
    if f.ndim != 3 or f.shape[-1] != 3:
        raise ValueError(f"expected (H,W,3) frame, got {f.shape}")
    return f


def _load_image(path: Path) -> Frame:
    try:
        from PIL import Image
    except ImportError as exc:                              # pragma: no cover
        raise DecodeError("reading image files needs Pillow") from exc
    with Image.open(path) as im:
        return _as_rgb01(np.asarray(im.convert("RGB")))


def luma(frame: Frame) -> np.ndarray:
    return frame @ LUMA
