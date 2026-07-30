"""Regenerate banked oracle numbers from the NC reference implementations.

The NC packages are a DEV-ONLY oracle: run this in a throwaway venv, commit the
NUMBERS (facts are not code), and CI compares against the banked JSON with no
NC install anywhere in the dependency tree.

    uv venv /tmp/oracle-venv -p 3.12
    uv pip install -p /tmp/oracle-venv/bin/python pyiqa
    /tmp/oracle-venv/bin/python parity/run_oracle.py
"""

import json
from datetime import date
from pathlib import Path

FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures"
NAMES = ("scene", "scene_blur", "scene_noise", "whitenoise")


def main() -> None:
    import pyiqa

    for metric, tol in (("niqe", 0.05), ("musiq", 0.06)):
        model = pyiqa.create_metric(metric, device="cpu")
        values = {n: float(model(str(FIXTURES / f"{n}.png"))) for n in NAMES}
        out = {
            "metric": metric,
            "oracle": (f"pyiqa=={pyiqa.__version__}, CPU, dev-only venv — "
                       "never a dependency of this package"),
            "generated": str(date.today()),
            "regenerate": "parity/run_oracle.py",
            "tolerance_rel": tol,
            "values": values,
        }
        path = FIXTURES / f"oracle_{metric}.json"
        path.write_text(json.dumps(out, indent=1) + "\n")
        print("wrote", path, values)


if __name__ == "__main__":
    main()
