"""Bank our ITU-R BS.1770-4 loudness against pyloudnorm.

Same discipline as `run_oracle.py`: the oracle goes in a THROWAWAY VENV, only
the NUMBERS are committed, and nothing in the dependency tree ever gains a
second loudness implementation.

pyloudnorm is MIT, so this one is not a licence quarantine — it is a correctness
quarantine. Two independent implementations of a published standard should agree
to a fraction of a dB, and if they ever stop agreeing we want to find out from a
banked number rather than from a wrong verdict.

    # oracle
    uv venv /tmp/loudness-oracle -p 3.12
    uv pip install -p /tmp/loudness-oracle/bin/python pyloudnorm soundfile numpy
    /tmp/loudness-oracle/bin/python parity/run_loudness_oracle.py oracle

    # ours
    python parity/run_loudness_oracle.py ours

The signals are SYNTHESISED here rather than committed as wav fixtures: they are
deterministic from a fixed seed, they exercise the gating logic (a silent
passage, a quiet passage below the relative gate, a loud passage), and a repo
does not need binary audio blobs to prove a filter is right.
"""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).parent.parent / "tests" / "fixtures" / "oracle_loudness.json"

#: 32 kHz is the rate that matters — MiniMax-H3's output, and the rate BS.1770-4
#: does NOT tabulate coefficients for, so it is where a wrong bilinear transform
#: would show up. 48 kHz is included because it is the rate the standard's own
#: coefficients are given at, so agreement there isolates the resampling of the
#: filter design from the rest of the algorithm.
RATES = (32000, 48000)

#: Anything within this of the oracle is agreement. Two correct BS.1770-4
#: implementations differ only by float accumulation order.
TOLERANCE_LU = 0.05


def signals(rate: int) -> dict[str, np.ndarray]:
    """Deterministic stereo test signals, (N, 2) float32 in [-1, 1]."""
    rng = np.random.default_rng(1770)
    n = 10 * rate
    t = np.arange(n) / rate
    out: dict[str, np.ndarray] = {}

    tone = 0.5 * np.sin(2 * np.pi * 997 * t)              # the classic 997 Hz probe
    out["tone_997"] = np.stack([tone, tone], axis=1)

    pink = rng.normal(0, 0.1, (n, 2))
    out["noise"] = pink

    quiet = pink * 0.01
    out["quiet_noise"] = quiet

    # Gating exercise: 4 s loud, 3 s of digital silence, 3 s at -30 dB. The
    # relative gate must discard the tail, the absolute gate the silence.
    gated = np.zeros((n, 2))
    gated[: 4 * rate] = pink[: 4 * rate]
    gated[7 * rate:] = pink[7 * rate:] * 0.03
    out["gated"] = gated

    # Stereo with real channel difference — the weighted channel sum matters.
    wide = np.stack([tone, 0.5 * np.sin(2 * np.pi * 1003 * t)], axis=1)
    out["wide_stereo"] = wide

    return {k: np.clip(v, -1.0, 1.0).astype(np.float32) for k, v in out.items()}


def run_oracle() -> dict:
    import pyloudnorm

    values = {}
    for rate in RATES:
        # DeMan, NOT pyloudnorm's default "K-weighting". Its default drives the
        # RBJ cookbook forms at fc=1500/38 Hz, which is a coarser approximation
        # of BS.1770 and disagrees with the recommendation's own tabulated
        # coefficients; "DeMan" is the accurate rate-generalized K-weighting and
        # is what we implement. The gap between the two is ~0.2 LU on tones.
        meter = pyloudnorm.Meter(rate)
        meter.filter_class = "DeMan"
        for name, data in signals(rate).items():
            values[f"{name}@{rate}"] = float(meter.integrated_loudness(data))
    return {
        "generated": str(date.today()),
        "oracle": f"pyloudnorm {pyloudnorm.__version__ if hasattr(pyloudnorm, '__version__') else ''}".strip(),
        "standard": "ITU-R BS.1770-4",
        "tolerance_lu": TOLERANCE_LU,
        "values": values,
    }


def run_ours() -> dict:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from cozy_eval.metrics.audio import as_audio, loudness_lufs

    values = {}
    for rate in RATES:
        for name, data in signals(rate).items():
            values[f"{name}@{rate}"] = float(loudness_lufs(as_audio(data, rate)))
    return values


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "ours"
    if mode == "oracle":
        banked = run_oracle()
        FIXTURE.write_text(json.dumps(banked, indent=1) + "\n")
        print(f"wrote {FIXTURE}")
        return

    ours = run_ours()
    if not FIXTURE.exists():
        print(json.dumps(ours, indent=1))
        print("\nno banked oracle yet — run the oracle mode in a throwaway venv first")
        return
    banked = json.loads(FIXTURE.read_text())
    worst = 0.0
    for key, expected in banked["values"].items():
        got = ours[key]
        delta = abs(got - expected)
        worst = max(worst, delta)
        flag = "ok " if delta <= banked["tolerance_lu"] else "FAIL"
        print(f"{flag} {key:24s} ours {got:9.3f}  oracle {expected:9.3f}  Δ {delta:.4f} LU")
    print(f"\nworst Δ {worst:.4f} LU against a tolerance of {banked['tolerance_lu']} LU")


if __name__ == "__main__":
    main()
