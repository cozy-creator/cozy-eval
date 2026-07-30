"""MUSIQ — multi-scale image quality transformer, PyTorch port.

STABILITY: experimental (v0.x). The metric NAME ``musiq`` is locked by the
registry; ``preprocess`` / ``convert_npz`` / ``load_model`` are internals of the
port and may change.

ATTRIBUTION: adapted from google-research/musiq, which is Apache-2.0 licensed
(Copyright 2021 Google LLC). This is a derivative work under Apache-2.0 section
4; the original licence and copyright notice travel with it. Changes: reimplemented
in PyTorch from the flax reference, plus an original npz->state_dict converter.
See PROVENANCE.md.

Ke et al., "MUSIQ: Multi-scale Image Quality Transformer" (ICCV 2021,
arXiv:2108.05997). ADAPTED with attribution from the Apache-2.0 reference
(google-research/musiq, flax); the released Apache checkpoints load directly
via :func:`convert_npz` — no permissive PyTorch path existed before this port.

Why it matters: MUSIQ scores the image at native resolution plus 224/384
aspect-ratio-preserving scales in ONE variable-length sequence (32px patches,
hash-based 10x10 spatial embedding, per-scale embedding), so nothing is
squashed to a square crop first — the property that made it the go-to learned
NR metric.

Faithfulness notes, measured against the NC oracle in tests/fixtures:
  - TF `GAUSSIAN` resize is reimplemented (sigma 0.5, radius 1.5, half-pixel
    centers, no antialias widening — TF's defaults).
  - `extract_patches` SAME padding and the v1 nearest-neighbour hash-index
    rule are reproduced exactly.

Checkpoint: koniq (default; MOS scale ~0-100). ~163 MB fp32, CPU-fine.
"""

from __future__ import annotations

import math
import os
import urllib.request
from pathlib import Path
from typing import Any

from ..device import AUTO, resolve_device
from ..errors import BackendError, ConfigError

PATCH = 32
GRID = 10                    # hash-based spatial embedding grid
SCALES = (224, 384)          # sorted longer-side lengths; native res is last id
HIDDEN = 384
LAYERS = 14
HEADS = 6
MLP_DIM = 1152

CHECKPOINT_URLS = {
    "koniq": "https://storage.googleapis.com/gresearch/musiq/koniq_ckpt.npz",
    "spaq": "https://storage.googleapis.com/gresearch/musiq/spaq_ckpt.npz",
    "paq2piq": "https://storage.googleapis.com/gresearch/musiq/paq2piq_ckpt.npz",
}

_MODEL_CACHE: dict[str, Any] = {}


def free_models() -> None:
    _MODEL_CACHE.clear()


# ---------------------------------------------------------------------------
# preprocessing — multi-scale patches + position annotations
# ---------------------------------------------------------------------------

def _resize_kernel(in_size: int, out_size: int) -> Any:
    """TF GAUSSIAN resize weights for one axis: sigma 0.5, radius 1.5,
    half-pixel centres, antialias off (TF resize defaults)."""
    import numpy as np

    scale = in_size / out_size
    src = (np.arange(out_size) + 0.5) * scale - 0.5
    offsets = np.arange(-2, 3)  # radius 1.5 -> at most 4 taps; 5 is safe
    idx = np.floor(src)[:, None] + offsets[None, :]
    x = idx - src[:, None]
    weights = np.where(np.abs(x) <= 1.5, np.exp(-(x * x) / (2 * 0.5**2)), 0.0)
    weights /= weights.sum(axis=1, keepdims=True)
    idx = np.clip(idx, 0, in_size - 1).astype(np.int64)
    return idx, weights


def _gaussian_resize(image: Any, out_h: int, out_w: int) -> Any:
    """Separable gaussian resample of [H, W, C] float."""
    import numpy as np

    idx, w = _resize_kernel(image.shape[0], out_h)
    image = (image[idx] * w[:, :, None, None]).sum(axis=1)
    idx, w = _resize_kernel(image.shape[1], out_w)
    image = (image[:, idx] * w[None, :, :, None]).sum(axis=2)
    return np.ascontiguousarray(image)


def _hash_ids(count: int, grid: int = GRID) -> Any:
    """TF v1 nearest-neighbour semantics: floor(i * grid / count), clipped."""
    import numpy as np

    return np.minimum((np.arange(count) * grid) // count, grid - 1)


def _extract_scale(image: Any, scale_id: int, max_len: int | None) -> tuple[Any, ...]:
    """(patches [L, 3072], spatial [L], scale [L], mask [L]) for one scale.
    ``max_len`` pads/cuts to a fixed length (fixed scales); None keeps all
    (native resolution)."""
    import numpy as np

    h, w = image.shape[:2]
    count_h, count_w = -(-h // PATCH), -(-w // PATCH)
    pad_h, pad_w = count_h * PATCH - h, count_w * PATCH - w
    image = np.pad(
        image,
        ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)),
    )
    # Row-major 32x32 blocks, flattened in (ph, pw, c) order like TF.
    patches = image.reshape(count_h, PATCH, count_w, PATCH, 3)
    patches = patches.transpose(0, 2, 1, 3, 4).reshape(count_h * count_w, -1)

    hh, ww = _hash_ids(count_h), _hash_ids(count_w)
    spatial = (hh[:, None] * GRID + ww[None, :]).reshape(-1)
    n = patches.shape[0]
    scale = np.full(n, scale_id, dtype=np.int64)
    mask = np.ones(n, dtype=np.int64)
    if max_len is not None:
        if n < max_len:
            pad = max_len - n
            patches = np.pad(patches, ((0, pad), (0, 0)))
            spatial = np.pad(spatial, (0, pad))
            scale = np.pad(scale, (0, pad))
            mask = np.pad(mask, (0, pad))
        else:
            patches, spatial = patches[:max_len], spatial[:max_len]
            scale, mask = scale[:max_len], mask[:max_len]
    return patches, spatial, scale, mask


def preprocess(image: Any) -> tuple[Any, Any, Any, Any]:
    """Image -> the model's variable-length multi-scale sequence."""
    import numpy as np

    arr = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
    arr = arr.astype(np.float64) / 255.0 * 2.0 - 1.0  # [-1, 1]
    h, w = arr.shape[:2]

    parts = []
    for scale_id, longer in enumerate(SCALES):
        ratio = longer / max(h, w)
        rh, rw = round(h * ratio), round(w * ratio)
        resized = _gaussian_resize(arr, rh, rw)
        max_len = math.ceil(longer / PATCH) ** 2
        parts.append(_extract_scale(resized, scale_id, max_len))
    parts.append(_extract_scale(arr, len(SCALES), None))

    patches, spatial, scale, mask = (np.concatenate(x) for x in zip(*parts, strict=True))
    return patches.astype(np.float32), spatial, scale, mask


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def _make_model_class() -> type:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class StdConv(nn.Conv2d):
        """Weight-standardized conv (per out-channel, population std, eps 1e-5)."""

        def forward(self, x):
            w = self.weight
            w = w - w.mean(dim=(1, 2, 3), keepdim=True)
            w = w / (w.std(dim=(1, 2, 3), keepdim=True, unbiased=False) + 1e-5)
            return self._conv_forward(x, w, self.bias)

    class Bottleneck(nn.Module):
        """The single resnet_emb unit: 64 -> 256 with projection, GN eps 1e-4."""

        def __init__(self):
            super().__init__()
            self.conv1 = StdConv(64, 64, 1, bias=False)
            self.gn1 = nn.GroupNorm(32, 64, eps=1e-4)
            self.conv2 = StdConv(64, 64, 3, padding=1, bias=False)
            self.gn2 = nn.GroupNorm(32, 64, eps=1e-4)
            self.conv3 = StdConv(64, 256, 1, bias=False)
            self.gn3 = nn.GroupNorm(32, 256, eps=1e-4)
            self.conv_proj = StdConv(64, 256, 1, bias=False)
            self.gn_proj = nn.GroupNorm(32, 256, eps=1e-4)

        def forward(self, x):
            residual = self.gn_proj(self.conv_proj(x))
            x = F.relu(self.gn1(self.conv1(x)))
            x = F.relu(self.gn2(self.conv2(x)))
            x = self.gn3(self.conv3(x))
            return F.relu(residual + x)

    class Attention(nn.Module):
        """flax MultiHeadDotProductAttention with key padding."""

        def __init__(self):
            super().__init__()
            self.query = nn.Linear(HIDDEN, HIDDEN)
            self.key = nn.Linear(HIDDEN, HIDDEN)
            self.value = nn.Linear(HIDDEN, HIDDEN)
            self.out = nn.Linear(HIDDEN, HIDDEN)

        def forward(self, x, mask):
            b, seq, _ = x.shape
            dh = HIDDEN // HEADS

            def split(t):
                return t.view(b, seq, HEADS, dh).transpose(1, 2)  # [B, H, L, dh]

            q, k, v = split(self.query(x)), split(self.key(x)), split(self.value(x))
            scores = q @ k.transpose(-2, -1) / math.sqrt(dh)
            scores = scores.masked_fill(~mask[:, None, None, :], -1e10)
            attn = scores.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(b, seq, HIDDEN)
            return self.out(out)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(HIDDEN, eps=1e-6)
            self.attn = Attention()
            self.ln2 = nn.LayerNorm(HIDDEN, eps=1e-6)
            self.mlp = nn.Sequential(
                nn.Linear(HIDDEN, MLP_DIM), nn.GELU(), nn.Linear(MLP_DIM, HIDDEN),
            )

        def forward(self, x, mask):
            x = x + self.attn(self.ln1(x), mask)
            return x + self.mlp(self.ln2(x))

    class Musiq(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_root = StdConv(3, 64, 7, stride=2, bias=False)
            self.gn_root = nn.GroupNorm(32, 64, eps=1e-6)
            self.block1 = Bottleneck()
            self.embedding = nn.Linear(8 * 8 * 256, HIDDEN)
            self.pos_embedding = nn.Parameter(torch.zeros(GRID * GRID, HIDDEN))
            self.scale_embedding = nn.Parameter(torch.zeros(len(SCALES) + 1, HIDDEN))
            self.cls = nn.Parameter(torch.zeros(HIDDEN))
            self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
            self.encoder_norm = nn.LayerNorm(HIDDEN, eps=1e-6)
            self.head = nn.Linear(HIDDEN, 1)

        def forward(self, patches, spatial, scale, mask):
            """patches [B, L, 3072]; spatial/scale/mask [B, L]."""
            import torch.nn.functional as F

            b, seq, _ = patches.shape
            x = patches.view(b * seq, PATCH, PATCH, 3).permute(0, 3, 1, 2)
            x = F.pad(x, (2, 3, 2, 3))            # SAME for k7 s2 on 32px
            x = F.relu(self.gn_root(self.conv_root(x)))
            x = F.max_pool2d(F.pad(x, (0, 1, 0, 1), value=-torch.inf), 3, stride=2)
            x = self.block1(x)                     # [B*L, 256, 8, 8]
            x = x.permute(0, 2, 3, 1).reshape(b, seq, -1)  # flax (h, w, c) order
            x = self.embedding(x)
            x = x + self.pos_embedding[spatial] + self.scale_embedding[scale]
            cls = self.cls.expand(b, 1, HIDDEN)
            x = torch.cat([cls, x], dim=1)
            key_mask = torch.cat(
                [torch.ones(b, 1, dtype=torch.bool, device=mask.device), mask.bool()],
                dim=1,
            )
            for block in self.blocks:
                x = block(x, key_mask)
            x = self.encoder_norm(x)
            return self.head(x[:, 0]).squeeze(-1)

    return Musiq


# ---------------------------------------------------------------------------
# checkpoint conversion (npz -> torch state dict)
# ---------------------------------------------------------------------------

def convert_npz(path: str | Path) -> dict[str, Any]:
    """Google's released flax .npz -> this module's state dict. Pure numpy."""
    import numpy as np
    import torch

    raw = np.load(path)
    params = {
        k[len("opt/target/"):]: raw[k] for k in raw.files if k.startswith("opt/target/")
    }

    def t(key):  # dense kernel (in, out) -> torch (out, in)
        return torch.from_numpy(np.ascontiguousarray(params[key].T))

    def conv(key):  # (H, W, in, out) -> (out, in, H, W)
        return torch.from_numpy(np.ascontiguousarray(params[key].transpose(3, 2, 0, 1)))

    def gn(key):  # (1, 1, 1, C) -> (C,)
        return torch.from_numpy(params[key].reshape(-1))

    def vec(key):
        return torch.from_numpy(np.ascontiguousarray(params[key]))

    out: dict[str, Any] = {
        "conv_root.weight": conv("conv_root/kernel"),
        "gn_root.weight": gn("gn_root/scale"),
        "gn_root.bias": gn("gn_root/bias"),
        "embedding.weight": t("embedding/kernel"),
        "embedding.bias": vec("embedding/bias"),
        "pos_embedding": vec("Transformer/posembed_input/pos_embedding")[0],
        "scale_embedding": vec("Transformer/scaleembed_input/scale_embedding")[0],
        "cls": vec("Transformer/cls").reshape(-1),
        "encoder_norm.weight": vec("Transformer/encoder_norm/scale"),
        "encoder_norm.bias": vec("Transformer/encoder_norm/bias"),
        "head.weight": t("head/kernel"),
        "head.bias": vec("head/bias"),
    }
    unit = "block1/unit1"
    for name in ("conv1", "conv2", "conv3", "conv_proj"):
        out[f"block1.{name}.weight"] = conv(f"{unit}/{name}/kernel")
    for name in ("gn1", "gn2", "gn3", "gn_proj"):
        out[f"block1.{name}.weight"] = gn(f"{unit}/{name}/scale")
        out[f"block1.{name}.bias"] = gn(f"{unit}/{name}/bias")

    for i in range(LAYERS):
        src = f"Transformer/encoderblock_{i}"
        dst = f"blocks.{i}"
        out[f"{dst}.ln1.weight"] = vec(f"{src}/LayerNorm_0/scale")
        out[f"{dst}.ln1.bias"] = vec(f"{src}/LayerNorm_0/bias")
        out[f"{dst}.ln2.weight"] = vec(f"{src}/LayerNorm_2/scale")
        out[f"{dst}.ln2.bias"] = vec(f"{src}/LayerNorm_2/bias")
        attn = f"{src}/MultiHeadDotProductAttention_1"
        for proj in ("query", "key", "value"):
            kernel = params[f"{attn}/{proj}/kernel"]      # (384, H, dh)
            bias = params[f"{attn}/{proj}/bias"]          # (H, dh)
            out[f"{dst}.attn.{proj}.weight"] = torch.from_numpy(
                np.ascontiguousarray(kernel.reshape(HIDDEN, HIDDEN).T)
            )
            out[f"{dst}.attn.{proj}.bias"] = torch.from_numpy(bias.reshape(-1))
        kernel = params[f"{attn}/out/kernel"]             # (H, dh, 384)
        out[f"{dst}.attn.out.weight"] = torch.from_numpy(
            np.ascontiguousarray(kernel.reshape(HIDDEN, HIDDEN).T)
        )
        out[f"{dst}.attn.out.bias"] = vec(f"{attn}/out/bias")
        mlp = f"{src}/MlpBlock_3"
        out[f"{dst}.mlp.0.weight"] = t(f"{mlp}/Dense_0/kernel")
        out[f"{dst}.mlp.0.bias"] = vec(f"{mlp}/Dense_0/bias")
        out[f"{dst}.mlp.2.weight"] = t(f"{mlp}/Dense_1/kernel")
        out[f"{dst}.mlp.2.bias"] = vec(f"{mlp}/Dense_1/bias")
    return out


#: Where downloaded checkpoints land. Override with ``COZY_EVAL_CACHE``, e.g.
#: to point an offline pod at a pre-baked directory.
CACHE_ENV = "COZY_EVAL_CACHE"


def cache_dir() -> Path:
    return Path(os.environ.get(CACHE_ENV) or (Path.home() / ".cache" / "cozy-eval"))


def _checkpoint_path(variant: str) -> Path:
    """The local npz for ``variant``, downloading Google's Apache-2.0 checkpoint
    on first use. Pre-place the file (or set ``COZY_EVAL_CACHE``) to run
    offline; nothing is fetched when it already exists."""
    if variant not in CHECKPOINT_URLS:
        raise ConfigError(
            f"unknown MUSIQ variant {variant!r}; known: {sorted(CHECKPOINT_URLS)}"
        )
    cache = cache_dir()
    path = cache / f"musiq_{variant}.npz"
    if path.exists():
        return path
    cache.mkdir(parents=True, exist_ok=True)
    url = CHECKPOINT_URLS[variant]
    try:
        urllib.request.urlretrieve(url, path)
    except OSError as exc:
        path.unlink(missing_ok=True)  # never leave a truncated checkpoint behind
        raise BackendError(
            f"could not download the MUSIQ {variant} checkpoint from {url}: {exc}. "
            f"Place it at {path} manually, or set {CACHE_ENV} to a directory that has it."
        ) from exc
    return path


def load_model(variant: str = "koniq", *, checkpoint: str | Path | None = None) -> Any:
    """Build the model and load the (converted) Apache checkpoint."""
    model = _make_model_class()()
    state = convert_npz(checkpoint if checkpoint is not None else _checkpoint_path(variant))
    model.load_state_dict(state, strict=True)
    return model.eval()


def musiq(image: Any, *, variant: str = "koniq", device: str = AUTO) -> float:
    """MUSIQ score for one image (koniq: MOS-scale, roughly 0-100, higher
    is better). The sequence length is content-dependent, so batching is
    per-image."""
    import torch

    device = resolve_device(device)
    key = f"musiq:{variant}:{device}"
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_model(variant).to(device)
    model = _MODEL_CACHE[key]
    patches, spatial, scale, mask = preprocess(image)
    to = lambda a, dt: torch.from_numpy(a).to(device=device, dtype=dt).unsqueeze(0)  # noqa: E731
    with torch.no_grad():
        score = model(
            to(patches, torch.float32), to(spatial, torch.long),
            to(scale, torch.long), to(mask, torch.long),
        )
    return float(score[0])


__all__ = [
    "CACHE_ENV",
    "CHECKPOINT_URLS",
    "GRID",
    "PATCH",
    "SCALES",
    "cache_dir",
    "convert_npz",
    "free_models",
    "load_model",
    "musiq",
    "preprocess",
]
