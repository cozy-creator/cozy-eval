"""OCR reader for the adherence dimension's quoted-string items.

A DETECTOR+RECOGNIZER, deliberately not a VLM: generative OCR emits plain text
with no per-line confidence and no boxes, which is exactly what a strict "was
this exact string rendered" check needs.

**rapidocr** (Apache-2.0 code AND weights) runs the same PP-OCR models exported
to ONNX, and is the whole reason this is cheap: rapidocr + onnxruntime + the
Latin pipeline weights is **~35 MB**, pre-bakeable for an offline pod. The
obvious alternative, `paddleocr`, looks light and then does not work — the
194 MB `paddlepaddle` framework is NOT a pip dependency of it, its GPU wheels
are not on PyPI at all, and it hard-pins the non-headless opencv build, which
drags GUI libraries into a headless pod. `surya-ocr` is disqualified outright:
Apache-2.0 code but revenue-capped OpenRAIL weights.

Absent a reader entirely, OCR items report as UNMEASURED — never as a silent
zero, which would read as "the model failed to render text".
"""

from __future__ import annotations

from typing import Any

import msgspec

ENGINE_RAPID = "rapidocr:PP-OCRv5-latin"

_CACHE: dict[str, Any] = {}


class OcrPage(msgspec.Struct, kw_only=True):
    measured: bool = False
    engine: str = ""
    text: str = ""                # all recognized lines, newline joined
    lines: list[str] = msgspec.field(default_factory=list)
    confidences: list[float] = msgspec.field(default_factory=list)
    note: str = ""


def free_models() -> None:
    _CACHE.clear()


def _rapid() -> Any:
    if "rapid" not in _CACHE:
        from rapidocr import RapidOCR

        _CACHE["rapid"] = RapidOCR()
    return _CACHE["rapid"]


def _doctr() -> Any:
    if "doctr" not in _CACHE:
        from doctr.models import ocr_predictor

        _CACHE["doctr"] = ocr_predictor(pretrained=True)
    return _CACHE["doctr"]


def _as_array(image: Any) -> Any:
    import numpy as np

    return np.asarray(image.convert("RGB")) if hasattr(image, "convert") else np.asarray(image)


def _read_rapid(image: Any) -> OcrPage:
    result = _rapid()(_as_array(image))
    lines = [str(t) for t in (getattr(result, "txts", None) or ())]
    confs = [float(s) for s in (getattr(result, "scores", None) or ())]
    return OcrPage(
        measured=True, engine=ENGINE_RAPID,
        text="\n".join(lines), lines=lines, confidences=confs,
    )


def _read_doctr(image: Any) -> OcrPage:
    import io as _io

    from doctr.io import DocumentFile

    buf = _io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    doc = _doctr()(DocumentFile.from_images([buf.getvalue()]))
    lines: list[str] = []
    confs: list[float] = []
    for page in doc.pages:
        for block in page.blocks:
            for line in block.lines:
                lines.append(" ".join(w.value for w in line.words))
                confs += [float(w.confidence) for w in line.words]
    return OcrPage(
        measured=True, engine="doctr",
        text="\n".join(lines), lines=lines, confidences=confs,
    )


def read(image: Any) -> OcrPage:
    """Recognized text for one image, trying rapidocr then docTR."""
    notes = []
    for name, fn in (("rapidocr", _read_rapid), ("doctr", _read_doctr)):
        try:
            return fn(image)
        except Exception as exc:
            notes.append(f"{name}: {type(exc).__name__}: {exc}")
    return OcrPage(measured=False, note="; ".join(notes))


def available() -> bool:
    import importlib.util

    return any(
        importlib.util.find_spec(m) is not None for m in ("rapidocr", "doctr")
    )


__all__ = ["ENGINE_RAPID", "OcrPage", "available", "free_models", "read"]
