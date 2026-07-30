"""Dimension 3: human preference.

STABILITY: experimental (v0.x) for these function signatures; the metric NAMES
(``pref_delta``, ``pref_score``) are locked by the registry.

**Primary is PickScore; the accuracy leader is not installable here.**

HPSv3 (``MizzenAI/HPSv3``) is the only released preference model that still
discriminates modern renders — an independent 2026 reproduction (DiT-Reward,
arXiv 2606.23626) puts it at 76.9% on HPDv3 where ImageReward/PickScore/
HPSv2/MPS collapse to 58-66%, i.e. near chance on current-generation images.
Its LICENCE is clean (MIT code, Apache-2.0 weights); what blocks it is a
dependency pin — the ``hpsv3`` package hard-pins ``transformers==4.45.2``.
``hpsv2`` is worse: ``protobuf<4``, and it declares pytest as a RUNTIME
dependency. Re-hosting HPSv3's weights against modern transformers is a real
option rather than a dead end — the backbone is Qwen2-VL, which current
transformers supports.

**PickScore** (``yuvalkirstain/PickScore_v1``) is the default, chosen on merit
for THIS measurement. Its HF card carries no license field — an absence of a
grant rather than a permissive one — and the project owner accepted that risk
explicitly (2026-07-27): *"assume it's fully open source for our purposes;
I'll take on the risk; we'll do something different if they complain."*
Recorded on every row that uses it, and in PROVENANCE.md.

Why it wins here, given HPSv3 scores higher on paper (65.6 vs 76.9 on HPDv3):

* **Delta resolution.** Our metric is a candidate-minus-reference delta on the
  SAME prompt, and those deltas are small. PickScore is a CLIP-H contrastive
  head emitting a continuous scalar. A generative reward model that states
  "7/10" quantizes to integers, so most quant pairs would score EXACTLY equal
  and the dimension would report zeros — measuring nothing. A coarse ruler is
  the wrong instrument for a small difference, whatever its ranking accuracy.
* **Cost.** One CLIP-H forward pass (~2 GB, milliseconds) against a 2B
  autoregressive generation per image. The constraint is that metrics stay
  fast.
* **Dependencies.** Zero new ones — it loads through the ``transformers``
  CLIPModel path this suite already uses for ``clip_delta``.

**UnifiedReward-2.0-qwen3vl-2b** (MIT code AND MIT weights, 4.26 GB) stays as
the SECONDARY: a second opinion on a genuinely different backbone family and a
different training objective. Disagreement between the two is the signal that a
preference verdict should not be trusted — the role HPSv2 would have played
before its dependency pins disqualified it.

``PRIMARY_MODEL`` is a single knob: point it at a re-hosted HPSv3 and nothing
else in the suite moves.
"""

from __future__ import annotations

import re
from typing import Any

import msgspec

from ..device import AUTO, resolve_device
from ..errors import ConfigError

# CLIP-H contrastive head, continuous scalar, ~2 GB, zero new dependencies.
PRIMARY_MODEL = "yuvalkirstain/PickScore_v1"
PRIMARY_PROCESSOR = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
# Recorded on every row scored by PickScore.
PRIMARY_LICENSE_NOTE = (
    "yuvalkirstain/PickScore_v1 publishes no license field; the project owner accepted the "
    "risk 2026-07-27: 'assume it's fully open source for our purposes; I'll "
    "take on the risk; we'll do something different if they complain.'"
)
# Second opinion on a different backbone family and training objective.
SECONDARY_MODEL = "CodeGoat24/UnifiedReward-2.0-qwen3vl-2b"
# Accuracy leader; BLOCKED by its transformers==4.45.2 pin. Kept named so the
# follow-up is unambiguous about what it is meant to restore.
PREFERRED_BLOCKED_MODEL = "MizzenAI/HPSv3"

# Keyed by (role, model, device). Keying on the role alone silently hands back
# a DIFFERENT model's scores when a caller switches model_ref mid-process.
_CACHE: dict[str, Any] = {}

_SCORE_PROMPT = (
    "Rate the overall quality of this image as a response to the prompt below, "
    "considering how well it follows the prompt, its aesthetics, and its "
    "realism. Reply with ONLY a single number from 1 to 10.\n\nPrompt: {prompt}"
)
_SCORE_RE = re.compile(r"(\d+(?:\.\d+)?)")


class PreferenceScores(msgspec.Struct, kw_only=True):
    measured: bool = False
    model_ref: str = ""
    scores: list[float] = msgspec.field(default_factory=list)
    uncertainty: list[float] = msgspec.field(default_factory=list)
    load_seconds: float = 0.0
    vram_bytes: int = 0
    note: str = ""


def free_models() -> None:
    _CACHE.clear()


def _scorer(model_ref: str, device: str) -> tuple[Any, Any]:
    import time

    key = f"pref:{model_ref}:{device}"
    if key not in _CACHE:
        started = time.monotonic()
        from transformers import AutoModelForImageTextToText, AutoProcessor

        _CACHE[key] = (
            AutoModelForImageTextToText.from_pretrained(
                model_ref, dtype="auto", device_map=device,
            ).eval(),
            AutoProcessor.from_pretrained(model_ref),
        )
        _CACHE[f"load_seconds:{key}"] = time.monotonic() - started
    return _CACHE[key]


def _parse_score(reply: str) -> float | None:
    match = _SCORE_RE.search(str(reply))
    if not match:
        return None
    value = float(match.group(1))
    return value if 0.0 <= value <= 10.0 else None


def _pickscore(device: str) -> tuple[Any, Any]:
    key = f"pick:{PRIMARY_MODEL}:{device}"
    if key not in _CACHE:
        import time

        from transformers import AutoModel, AutoProcessor

        started = time.monotonic()
        _CACHE[key] = (
            AutoModel.from_pretrained(PRIMARY_MODEL).to(device).eval(),
            AutoProcessor.from_pretrained(PRIMARY_PROCESSOR),
        )
        _CACHE[f"load_seconds:{key}"] = time.monotonic() - started
    return _CACHE[key]


def pickscore_scores(
    images: list[Any], prompts: list[str], *, device: str = AUTO,
) -> PreferenceScores:
    """PickScore: logit_scale * cos(image_emb, text_emb). Continuous scalar."""
    from .adherence import resident_vram

    device = resolve_device(device)
    try:
        model, processor = _pickscore(device)
    except Exception as exc:
        return PreferenceScores(
            measured=False, model_ref=PRIMARY_MODEL,
            note=f"PickScore unavailable: {type(exc).__name__}: {exc}",
        )
    import torch

    scores: list[float] = []
    with torch.no_grad():
        for image, prompt in zip(images, prompts, strict=True):
            img_in = processor(
                images=[image], return_tensors="pt",
            ).to(device)
            txt_in = processor(
                text=[prompt[:300]], padding=True, truncation=True,
                max_length=77, return_tensors="pt",
            ).to(device)
            img_e = model.get_image_features(**img_in)
            img_e = img_e / img_e.norm(dim=-1, keepdim=True)
            txt_e = model.get_text_features(**txt_in)
            txt_e = txt_e / txt_e.norm(dim=-1, keepdim=True)
            scores.append(float(
                model.logit_scale.exp() * (txt_e @ img_e.T)[0][0]
            ))
    return PreferenceScores(
        measured=True, model_ref=PRIMARY_MODEL, scores=scores,
        load_seconds=float(_CACHE.get(f"load_seconds:pick:{PRIMARY_MODEL}:{device}", 0.0)),
        vram_bytes=resident_vram(), note=PRIMARY_LICENSE_NOTE,
    )


def primary_scores(
    images: list[Any], prompts: list[str], *, device: str = AUTO,
    model_ref: str = PRIMARY_MODEL,
) -> PreferenceScores:
    """Per-image preference score. Absolute values are only comparable WITHIN
    a prompt — same-prompt deltas are the meaningful quantity."""
    if len(images) != len(prompts):
        raise ConfigError(
            f"preference: got {len(images)} images but {len(prompts)} prompts; "
            "each image is scored against its own prompt, so the lists must match"
        )
    device = resolve_device(device)
    if model_ref == PRIMARY_MODEL:
        return pickscore_scores(images, prompts, device=device)
    try:
        model, processor = _scorer(model_ref, device)
    except Exception as exc:  # package or weights absent on this image
        return PreferenceScores(
            measured=False, model_ref=model_ref,
            note=f"preference model unavailable: {type(exc).__name__}: {exc}",
        )
    import torch

    scores: list[float] = []
    for image, prompt in zip(images, prompts, strict=True):
        text = processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": _SCORE_PROMPT.format(prompt=prompt[:600])},
            ]}],
            add_generation_prompt=True, tokenize=False,
        )
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(
            model.device
        )
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        reply = processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )[0]
        value = _parse_score(reply)
        if value is None:
            return PreferenceScores(
                measured=False, model_ref=model_ref,
                note=f"preference model returned an unparseable score: {reply!r}",
            )
        scores.append(value)
    from .adherence import resident_vram

    return PreferenceScores(
        measured=True, model_ref=model_ref, scores=scores,
        load_seconds=float(_CACHE.get(f"load_seconds:pref:{model_ref}:{device}", 0.0)),
        vram_bytes=resident_vram(),
    )


def deltas(candidate: PreferenceScores, reference: PreferenceScores) -> list[float]:
    """Same-prompt candidate-minus-reference; [] when either arm is unmeasured."""
    if not (candidate.measured and reference.measured):
        return []
    if len(candidate.scores) != len(reference.scores):
        raise ConfigError(
            f"preference: the candidate arm has {len(candidate.scores)} scores and the "
            f"reference arm {len(reference.scores)}; a delta needs one pair per prompt"
        )
    return [c - r for c, r in zip(candidate.scores, reference.scores, strict=True)]


__all__ = [
    "PREFERRED_BLOCKED_MODEL",
    "PRIMARY_LICENSE_NOTE",
    "PRIMARY_MODEL",
    "PRIMARY_PROCESSOR",
    "SECONDARY_MODEL",
    "PreferenceScores",
    "deltas",
    "free_models",
    "pickscore_scores",
    "primary_scores",
]
