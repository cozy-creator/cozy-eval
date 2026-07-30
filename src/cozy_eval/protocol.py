"""Protocol stamping and the two-lane validity rule.

This module is the reason the library exists. Everything else is measurement;
this is the part that stops you measuring the wrong thing.

**The rule.** A quality comparison between two model arms is only meaningful if
you know what the change did to the *sampling trajectory*.

* A change that leaves the latent trajectory bit-identical and only alters what
  happens after it (VAE decode dtype, decoder tiling, colour conversion, the
  video encoder) produces a candidate that is pixel-comparable to the reference.
  Reference metrics — PSNR, SSIM, LPIPS, VMAF — are correct here, at n=1.
* A change that perturbs the trajectory (weight or activation quantization,
  ``torch.compile``, an attention-backend swap, a scheduler/step-count change, a
  LoRA attach) produces a *different take* of the same prompt. The reference
  render is no longer the same video; distance to it measures divergence, not
  damage.

The second case is not a subtlety. Measured on banked LTX-2.3 clips at 1920x1088
(``~/cozy/samples/w8a8-audit/``): a **compile-only control** — zero quantization
change — scores LPIPS 0.196-0.249 against a fleet fp8 budget of 0.25, i.e. the
no-op consumes the entire budget. The distance is already 0.29-0.41 at frame 0
and flat-to-falling across the clip, which is the signature of a different take
rather than accumulating error. Worst of all the **ranking inverts**: an
fp8-storage-cast arm and an unscaled-w8a8 arm carry identical weight bytes, but
the cast arm computes its GEMMs in bf16 and is therefore strictly the more
accurate path — and it scores *worse* (0.3875 vs 0.3052). A metric that ranks a
strictly-better arm below a strictly-worse one cannot be used to choose between
arms.

So this library refuses. :func:`require_reference_lane` raises on a
trajectory-perturbing change instead of returning a number that looks fine.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from .errors import ProtocolError, TrajectoryPerturbingError


class Lane(enum.StrEnum):
    """Which family of metrics is admissible for a comparison."""

    #: Latent trajectory identical; reference metrics (PSNR/SSIM/LPIPS/VMAF) valid at n=1.
    SAME_TRAJECTORY = "same-trajectory"
    #: Trajectory perturbed; only no-reference, population-level statistics are valid.
    POPULATION = "population"


class Verdict(enum.StrEnum):
    """What a gate concluded. Only :attr:`PASS` permits promotion."""

    PASS = "pass"
    FAIL = "fail"
    #: Ran, but the protocol was not satisfied (too few prompts, mismatched
    #: execution lanes, cross-pod arms) or a budget the verdict would rest on is
    #: not trustworthy for this family. Never promote on this.
    INDETERMINATE = "indeterminate"
    #: An arm carries no signal — constant/zero-variance frames. Not a FAIL: a
    #: FAIL is a measured comparison, and there is nothing here to compare. The
    #: arm cannot be scored, ranked, or promoted; look at the render.
    DEGENERATE = "degenerate"


class ChangeKind(enum.StrEnum):
    """What the candidate arm changed, relative to the reference arm.

    The lane follows from this and nothing else. If a change is not listed,
    :func:`classify_change` raises rather than guessing — an unclassified change
    silently defaulting to the reference lane is exactly the failure this
    library exists to prevent.
    """

    # --- trajectory-perturbing: POPULATION lane -------------------------------
    WEIGHT_QUANTIZATION = "weight-quantization"
    ACTIVATION_QUANTIZATION = "activation-quantization"
    WEIGHT_STORAGE_CAST = "weight-storage-cast"
    COMPILE = "compile"
    ATTENTION_BACKEND = "attention-backend"
    SCHEDULER = "scheduler"
    STEP_COUNT = "step-count"
    GUIDANCE = "guidance"
    LORA_ATTACH = "lora-attach"
    LORA_FUSE = "lora-fuse"
    DENOISER_DTYPE = "denoiser-dtype"
    OFFLOAD_PLACEMENT = "offload-placement"
    HARDWARE_SKU = "hardware-sku"
    MODEL_WEIGHTS = "model-weights"
    #: Nothing changed but the seed. This is the NULL CONTROL arm (see
    #: :mod:`cozy_eval.control`): identical weights, different draw. It
    #: is trajectory-perturbing — a different seed is a different take — so it
    #: correctly lands on the population lane, and it is the only ChangeKind a
    #: null control may declare.
    SEED = "seed"

    # --- post-latent: SAME_TRAJECTORY lane ------------------------------------
    VAE_DECODE_DTYPE = "vae-decode-dtype"
    VAE_DECODE_TILING = "vae-decode-tiling"
    VAE_DECODE_BACKEND = "vae-decode-backend"
    COLOR_CONVERSION = "color-conversion"
    OUTPUT_RESIZE = "output-resize"
    VIDEO_ENCODER = "video-encoder"
    CONTAINER_MUX = "container-mux"


_POST_LATENT = frozenset({
    ChangeKind.VAE_DECODE_DTYPE,
    ChangeKind.VAE_DECODE_TILING,
    ChangeKind.VAE_DECODE_BACKEND,
    ChangeKind.COLOR_CONVERSION,
    ChangeKind.OUTPUT_RESIZE,
    ChangeKind.VIDEO_ENCODER,
    ChangeKind.CONTAINER_MUX,
})


def _reject_packed(changes: tuple[Any, ...], caller: str) -> None:
    """Give the varargs mistake its own error instead of 'unclassified change'."""
    if len(changes) == 1 and not isinstance(changes[0], str) and isinstance(changes[0], Iterable):
        raise TypeError(
            f"{caller}() takes each ChangeKind as a separate argument, not a sequence — "
            f"call {caller}(*protocol.changes), not {caller}(protocol.changes)"
        )


def classify_change(*changes: ChangeKind | str) -> Lane:
    """Return the admissible lane for a set of simultaneous changes.

    **Varargs**: ``classify_change(ChangeKind.COMPILE, ChangeKind.SCHEDULER)``, and
    ``classify_change(*protocol.changes)`` for a protocol's tuple.

    A comparison is on the same-trajectory lane only if EVERY change is
    post-latent. One trajectory-perturbing change contaminates the whole
    comparison.
    """
    _reject_packed(changes, "classify_change")
    if not changes:
        raise ProtocolError("a comparison must declare at least one ChangeKind")
    kinds = []
    for c in changes:
        try:
            kinds.append(ChangeKind(c))
        except ValueError as exc:
            raise ProtocolError(
                f"unclassified change {c!r}: add it to ChangeKind and decide its lane "
                "deliberately — an unclassified change must never default to the "
                "reference lane"
            ) from exc
    return Lane.SAME_TRAJECTORY if all(k in _POST_LATENT for k in kinds) else Lane.POPULATION


def require_reference_lane(*changes: ChangeKind | str) -> None:
    """Guard reference metrics. Raises unless every change is post-latent.

    Varargs, like :func:`classify_change`: ``require_reference_lane(*protocol.changes)``.
    """
    _reject_packed(changes, "require_reference_lane")
    if classify_change(*changes) is not Lane.SAME_TRAJECTORY:
        perturbing = [str(ChangeKind(c)) for c in changes if ChangeKind(c) not in _POST_LATENT]
        raise TrajectoryPerturbingError(
            f"reference metrics (PSNR/SSIM/LPIPS/VMAF) are invalid for {perturbing}: "
            "these change the sampling trajectory, so the reference render is a "
            "different take of the prompt and distance to it measures divergence, "
            "not damage. Measured on banked LTX clips, a compile-only control scores "
            "LPIPS 0.196-0.249 against a 0.25 budget and the arm ranking inverts. "
            "Use the population lane (no-reference statistics over n>=8 paired "
            "prompts) instead."
        )


@dataclass(frozen=True, slots=True)
class Protocol:
    """The pinning stamp every result carries.

    A quality number without its protocol is not a result. Two numbers taken at
    different resolutions, step counts or execution lanes are not comparable,
    and the most common way to ship a wrong verdict is to compare them anyway.
    """

    family: str                     # e.g. "ltx-2.3-distilled"
    reference_arm: str              # e.g. "bf16 compiled"
    candidate_arm: str              # e.g. "w8a8-pcs compiled"
    changes: tuple[ChangeKind, ...]
    width: int
    height: int
    frames: int
    steps: int
    seeds: tuple[int, ...]
    prompts: tuple[str, ...]
    execution_lane: str             # "compiled" | "eager" — MUST match across arms
    hardware: str                   # e.g. "NVIDIA H100 80GB HBM3"
    fps: float = 0.0
    guidance: float = 0.0
    same_pod: bool = True
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompts:
            raise ProtocolError("protocol must name the prompt set it was measured on")
        if len(self.seeds) not in (1, len(self.prompts)):
            raise ProtocolError(
                f"seeds ({len(self.seeds)}) must be one shared seed or one per prompt "
                f"({len(self.prompts)})"
            )
        if self.execution_lane not in ("compiled", "eager"):
            raise ProtocolError("execution_lane must be 'compiled' or 'eager'")
        if min(self.width, self.height, self.frames, self.steps) <= 0:
            raise ProtocolError("width/height/frames/steps must all be positive")

    @property
    def n(self) -> int:
        return len(self.prompts)

    @property
    def lane(self) -> Lane:
        return classify_change(*self.changes)

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"

    def line(self) -> str:
        """One-line human stamp — put this next to every number you publish."""
        return (
            f"{self.family} | {self.candidate_arm} vs {self.reference_arm} | "
            f"{self.resolution} x{self.frames}f | {self.steps} steps | n={self.n} prompts | "
            f"{self.execution_lane} both arms | {self.hardware} | "
            f"{'same pod' if self.same_pod else 'CROSS-POD'} | lane={self.lane}"
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["changes"] = [str(c) for c in self.changes]
        d["lane"] = str(self.lane)
        d["n"] = self.n
        d["stamp"] = self.line()
        return d
