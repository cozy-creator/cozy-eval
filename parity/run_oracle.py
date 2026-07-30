"""Bank the quality metrics against the NC reference implementation.

`pyiqa` is a DEV-ONLY oracle: run it in a throwaway venv, commit the NUMBERS
(facts are not code), and CI compares against the banked JSON with no NC install
anywhere in the dependency tree. 0.1.15 is the last Apache-2.0 release, so even
the throwaway venv stays on the permissive line.

    # oracle numbers ("values") — throwaway venv, never a dependency
    uv venv /tmp/oracle-venv -p 3.12
    uv pip install -p /tmp/oracle-venv/bin/python "pyiqa==0.1.15" "setuptools<81"
    /tmp/oracle-venv/bin/python parity/run_oracle.py oracle

    # our numbers ("ours") — the package's own venv
    python parity/run_oracle.py ours

`match` says what the banked comparison asserts:

  exact     our implementation reproduces the oracle inside `tolerance_rel`,
            once driven with the oracle's own configuration (`oracle_config`).
  diverged  we deliberately compute something else; `divergence` says what and
            why, and the test guards `ours` against regression instead.
"""

import json
import sys
from datetime import date
from pathlib import Path

FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures"
NAMES = ("scene", "scene_blur", "scene_noise", "whitenoise")

#: pyiqa's five hand-written CLIP-IQA antonym pairs. The paper's canonical probe
#: is one pair ("Good photo."/"Bad photo."), which is our default.
PYIQA_CLIPIQA_PROMPTS = (
    ("Good image", "bad image"),
    ("Sharp image", "blurry image"),
    ("sharp edges", "blurry edges"),
    ("High resolution image", "low resolution image"),
    ("Noise-free image", "noisy image"),
)

METRICS = {
    "niqe": {
        "pyiqa": "niqe",
        "tolerance_rel": 0.05,
        "match": "exact",
        "note": (
            "OUR implementation of Mittal et al. 2013. Residual vs the Matlab-convention "
            "oracle comes from resize and filter-border choices (PIL bicubic vs imresize, "
            "nearest-edge correlate); measured max 2.4% on these fixtures."
        ),
    },
    "musiq": {
        "pyiqa": "musiq",
        "tolerance_rel": 0.06,
        "match": "exact",
        "note": (
            "OUR PyTorch port of the Apache reference. Rankings identical; residuals (max "
            "4.5%, on the aliasing-sensitive noise fixture) come from GAUSSIAN-resize "
            "approximations — the oracle's own preprocessing also deviates from TF. "
            "whitenoise scoring ~45 is the model's real out-of-distribution behavior, "
            "reproduced by both implementations."
        ),
    },
    "clip_iqa": {
        "pyiqa": "clipiqa",
        "tolerance_rel": 1e-4,
        "match": "exact",
        "oracle_config": {"prompts": [list(p) for p in PYIQA_CLIPIQA_PROMPTS]},
        "note": (
            "NOT our implementation: torchmetrics (Apache-2.0) over the same piq-hosted "
            "CLIP-IQA RN50 weights. Bit-identical to the oracle once given the oracle's "
            "prompt set, so the DEFAULT difference is the prompt choice and nothing else: "
            "we use the paper's single canonical pair, pyiqa averages five hand-written "
            "pairs that also probe sharpness, resolution and noise."
        ),
    },
    "arniqa": {
        "pyiqa": "arniqa",
        "tolerance_rel": 1e-4,
        "match": "diverged",
        "divergence": (
            "half-scale branch. We resize with antialiasing (torchvision `transforms."
            "Resize`, via torchmetrics); pyiqa decimates with `F.interpolate(mode="
            "'bilinear')`, which aliases exactly the high-frequency energy an NR quality "
            "metric reads — its whitenoise fixture scores 0.376 against our 0.576. Same "
            "encoder and regressor otherwise: swapping in the non-antialiased resize "
            "reproduces the oracle to 1e-4, so the resize is the whole difference."
        ),
        "note": (
            "NOT our implementation: torchmetrics (Apache-2.0), koniq10k regressor. "
            "`ours` is the regression guard; `values` is what pyiqa returns."
        ),
    },
}


def _path(name: str) -> Path:
    return FIXTURES / f"oracle_{name}.json"


def _load(name: str) -> dict:
    path = _path(name)
    return json.loads(path.read_text()) if path.exists() else {}


def _write(name: str, doc: dict) -> None:
    _path(name).write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print("wrote", _path(name), doc.get("values"), doc.get("ours"))


def bank_oracle() -> None:
    import pyiqa

    for ours, cfg in METRICS.items():
        model = pyiqa.create_metric(cfg["pyiqa"], device="cpu")
        doc = _load(ours)
        doc.update(
            {k: v for k, v in cfg.items() if k != "pyiqa"},
            metric=ours,
            oracle=(f"pyiqa=={pyiqa.__version__} ({cfg['pyiqa']}), CPU, dev-only venv — "
                    "never a dependency of this package"),
            generated=str(date.today()),
            regenerate="parity/run_oracle.py",
            values={n: float(model(str(FIXTURES / f"{n}.png"))) for n in NAMES},
        )
        _write(ours, doc)


def bank_ours() -> None:
    """Only for metrics whose banked assertion is against `ours` (match=diverged)."""
    from PIL import Image

    from cozy_eval.metrics import quality as iqa

    for name, cfg in METRICS.items():
        if cfg["match"] != "diverged":
            continue
        fn = getattr(iqa, name)
        doc = _load(name)
        doc["ours"] = {n: fn(Image.open(FIXTURES / f"{n}.png"), device="cpu") for n in NAMES}
        _write(name, doc)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "oracle"
    {"oracle": bank_oracle, "ours": bank_ours}[mode]()
