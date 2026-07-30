"""The judge interface: what a vision-language model has to do to grade for us.

STABILITY: locked core (v0.x).

There are TWO judge protocols, because there are two genuinely different reads
of the same model, and conflating them was the original sin this module fixes:

  Judge      HARD. One structured, multi-question call per image; the reply is
             text that :func:`cozy_eval.bench.metrics.adherence.parse_judge_answers`
             turns into yes/no. Cheap — a 10-item checklist costs ONE call — and
             it is what the authored-checklist lane and the paired gate use.

  SoftJudge  SOFT. One question at a time, answered with the PROBABILITY of
             "yes" rather than a parsed token. Resolves small differences a hard
             parse rounds away, at one prefill per question. What the VQAScore /
             Soft-TIFA lane uses.

Both are :class:`typing.Protocol` — structural, not inherited. Anything with the
right method is a judge, so a test stub, an HTTP client for a hosted model, or a
resident local VLM are interchangeable with no registration step. The concrete
local implementations are
:class:`cozy_eval.bench.metrics.adherence.VlmJudge` and
:class:`cozy_eval.bench.metrics.vqascore.VlmSoftJudge`.

The optional accounting attributes are read by the suite to fill in the report's
cost fields. They are declared here so an implementer knows what is looked at;
all are optional and default sensibly when absent.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Judge(Protocol):
    """A VLM that answers a batch of yes/no questions about images as text."""

    #: HF id or endpoint of the model, recorded in the report's ``models``.
    model_ref: str

    def ask(self, images: list[Any], prompt: str) -> str:
        """Answer one multi-question prompt about ``images``.

        ``prompt`` is built by
        :func:`cozy_eval.bench.metrics.adherence.build_judge_prompt` and asks for a
        JSON array of ``{"n": <question number>, "answer": "yes"|"no"}``. Return
        the model's raw reply — parsing is tolerant and lives on our side. In
        edit mode ``images`` is ``[before, after]``, in that order.
        """
        ...


@runtime_checkable
class SoftJudge(Protocol):
    """A VLM read for the probability of "yes" rather than a generated token."""

    model_ref: str

    def p_yes(self, images: list[Any], question: str) -> float:
        """p(yes) / (p(yes) + p(no)) for one question, in ``[0, 1]``."""
        ...


#: Accounting attributes the suite reads off a judge when present. Absent ones
#: are simply not reported — implementing them is optional.
ACCOUNTING_ATTRS = ("load_seconds", "infer_seconds", "call_count", "vram_bytes")


__all__ = ["ACCOUNTING_ATTRS", "Judge", "SoftJudge"]
