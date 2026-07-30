"""Soft VQA alignment: p("yes") instead of a parsed yes/no.

STABILITY: experimental (v0.x). Metric NAMES (``vqascore``, ``atom_score``) are
locked by the registry; ``VlmSoftJudge`` and the aggregation signatures are not.

Two methods on one primitive:

  vqascore   ONE number for prompt-image alignment: the probability a
             generative VLM answers "yes" to `Does this figure show "{text}"?`
             (Lin et al. 2024, t2v_metrics — Apache-2.0). No decomposition.
  soft_tifa  per-atom p("yes") over a template-decomposed checklist,
             arithmetic-mean atom score + geometric-mean prompt score —
             GenEval2's Soft-TIFA aggregation (arXiv:2512.16853; method only,
             their CC-BY-NC code/data untouched). Atoms come from
             ``cozy_eval.decompose`` and are versioned data, not eval-time
             LLM output.

Soft scores resolve small deltas a hard parse rounds away, at a cost: one
prefill per atom instead of the hard checklist lane's ONE call per image.
Both lanes stay: hard for cheap gates, soft for leaderboard resolution.

The default judge is the same Apache-2.0 checkpoint the adherence lane uses;
p("yes") is normalized against p("no") over the first answer token, summing
capitalization/whitespace variants.
"""

from __future__ import annotations

import math
import time
from typing import Any

import msgspec

from ..device import AUTO, resolve_device
from ..errors import BackendError
from ..judge import SoftJudge
from .adherence import DEFAULT_JUDGE, KIND_VQA, ChecklistItem

VQASCORE_TEMPLATE = 'Does this figure show "{text}"? Please answer yes or no.'

_YES = ("yes", "Yes", " yes", " Yes", "YES")
_NO = ("no", "No", " no", " No", "NO")


class AtomScore(msgspec.Struct, kw_only=True):
    id: str
    question: str
    p: float                  # p(constraint satisfied); absent items inverted


class SoftTifaScore(msgspec.Struct, kw_only=True):
    measured: bool = False
    atom_mean: float = 0.0    # arithmetic mean — the per-image atom score
    prompt_score: float = 0.0  # geometric mean — punishes any near-zero atom
    atoms: list[AtomScore] = msgspec.field(default_factory=list)


def aggregate(atoms: list[AtomScore]) -> SoftTifaScore:
    """Soft-TIFA aggregation, pure. Geometric mean goes to ~0 when any single
    constraint is clearly violated — one wrong object cannot be averaged away."""
    if not atoms:
        return SoftTifaScore()
    mean = sum(a.p for a in atoms) / len(atoms)
    geo = math.exp(sum(math.log(max(a.p, 1e-6)) for a in atoms) / len(atoms))
    return SoftTifaScore(measured=True, atom_mean=mean, prompt_score=geo, atoms=atoms)


class VlmSoftJudge:
    """A resident local VLM read for one-token answer probabilities rather than
    generation — the reference :class:`cozy_eval.judge.SoftJudge`."""

    def __init__(self, model_ref: str = DEFAULT_JUDGE, *, device: str = AUTO):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_ref = model_ref
        self.device = resolve_device(device)
        device = self.device
        self.load_seconds = 0.0
        self.infer_seconds = 0.0
        self.call_count = 0
        started = time.monotonic()
        self.processor = AutoProcessor.from_pretrained(model_ref)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_ref, dtype="auto", device_map=device,
        ).eval()
        self.load_seconds = time.monotonic() - started
        tok = self.processor.tokenizer
        self.yes_ids = _single_token_ids(tok, _YES)
        self.no_ids = _single_token_ids(tok, _NO)
        if not self.yes_ids or not self.no_ids:
            raise BackendError(
                f"{model_ref}: tokenizer encodes no single-token 'yes'/'no' variant, so "
                "p(yes) cannot be read at one position; use a different judge model"
            )

    def p_yes(self, images: list[Any], question: str) -> float:
        """p(yes) / (p(yes) + p(no)) at the first answer position."""
        import torch

        started = time.monotonic()
        content: list[dict[str, Any]] = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": question})
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=False,
        )
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(
            self.model.device
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        yes = float(sum(probs[i] for i in self.yes_ids))
        no = float(sum(probs[i] for i in self.no_ids))
        self.infer_seconds += time.monotonic() - started
        self.call_count += 1
        if yes + no <= 0.0:
            return 0.0
        return yes / (yes + no)


def _single_token_ids(tokenizer: Any, variants: tuple[str, ...]) -> tuple[int, ...]:
    ids = []
    for v in variants:
        toks = tokenizer.encode(v, add_special_tokens=False)
        if len(toks) == 1:
            ids.append(toks[0])
    return tuple(dict.fromkeys(ids))


def vqascore(judge: SoftJudge, image: Any, text: str) -> float:
    """Single-number prompt-image alignment (Lin et al. 2024)."""
    return judge.p_yes([image], VQASCORE_TEMPLATE.format(text=text))


def soft_tifa(
    judge: SoftJudge, image: Any, items: tuple[ChecklistItem, ...],
) -> SoftTifaScore:
    """Score a template-decomposed atom checklist; one prefill per vqa atom.
    OCR items are skipped here — text fidelity has its own reader."""
    scored: list[AtomScore] = []
    for item in items:
        if item.kind != KIND_VQA:
            continue
        p = judge.p_yes([image], item.question)
        scored.append(AtomScore(
            id=item.id, question=item.question,
            p=(1.0 - p) if item.absent else p,
        ))
    return aggregate(scored)


__all__ = [
    "VQASCORE_TEMPLATE",
    "AtomScore",
    "SoftTifaScore",
    "VlmSoftJudge",
    "aggregate",
    "soft_tifa",
    "vqascore",
]
