"""Local analysis of the pod's results.json: delta resolution + agreement."""

import json
import statistics as st
import sys


def deltas(rows, key):
    return [r[f"{key}_cand"] - r[f"{key}_ref"] for r in rows]


def spread(values):
    values = sorted(values)
    n = len(values)
    return {
        "mean": st.mean(values), "sd": st.pstdev(values),
        "iqr": values[3 * n // 4] - values[n // 4],
        "min": values[0], "max": values[-1],
        "ties": sum(1 for v in values if v == 0.0),
        "distinct": len({round(v, 6) for v in values}),
    }


def main(path):
    data = json.loads(open(path).read())
    rows = data["rows"]
    real = [r for r in rows if not r["population"].startswith("control")]
    ctrl = [r for r in rows if r["population"].startswith("control")]

    print(f"device={data['device']} seconds={ {k: round(v,1) for k,v in data['seconds'].items()} }")
    print(f"pairs: {len(real)} real + {len(ctrl)} control\n")

    for name, key in (("pickscore", "pick"), ("hpsv3", "hps")):
        dr, dc = deltas(real, key), deltas(ctrl, key)
        sr, sc = spread(dr), spread(dc)
        # separation: how far a typical real |delta| sits above control noise
        noise = st.pstdev(dc) or 1e-9
        z = st.mean([abs(d) for d in dr]) / noise
        print(f"[{name}] real:    mean={sr['mean']:+.4f} sd={sr['sd']:.4f} "
              f"iqr={sr['iqr']:.4f} range=({sr['min']:+.4f},{sr['max']:+.4f}) "
              f"ties={sr['ties']}/{len(dr)} distinct={sr['distinct']}")
        print(f"[{name}] control: mean={sc['mean']:+.4f} sd={sc['sd']:.4f} iqr={sc['iqr']:.4f} "
              f"ties={sc['ties']}/{len(dc)} distinct={sc['distinct']}")
        print(f"[{name}] |real-delta| / control-noise z = {z:.2f}\n")

    dp, dh = deltas(real, "pick"), deltas(real, "hps")
    same_sign = sum(
        1 for a, b in zip(dp, dh, strict=True) if (a > 0) == (b > 0) and a != 0 and b != 0
    )
    n = len(dp)
    mp, mh = st.mean(dp), st.mean(dh)
    sp = st.pstdev(dp) or 1e-9
    sh = st.pstdev(dh) or 1e-9
    pearson = st.mean([(a - mp) * (b - mh) for a, b in zip(dp, dh, strict=True)]) / (sp * sh)
    rank = lambda xs: {i: r for r, i in enumerate(sorted(range(len(xs)), key=lambda i: xs[i]))}  # noqa: E731
    rp, rh = rank(dp), rank(dh)
    d2 = sum((rp[i] - rh[i]) ** 2 for i in range(n))
    spearman = 1 - 6 * d2 / (n * (n * n - 1))
    print(f"[agreement on real pairs] delta sign agreement {same_sign}/{n} "
          f"pearson={pearson:+.3f} spearman={spearman:+.3f}")

    print("\nper-population mean deltas (cand - ref; negative = quant arm worse):")
    for pop in sorted({r["population"] for r in rows}):
        sub = [r for r in rows if r["population"] == pop]
        print(f"  {pop:16s} pick={st.mean(deltas(sub, 'pick')):+.4f}  "
              f"hps={st.mean(deltas(sub, 'hps')):+.4f}  n={len(sub)}")


if __name__ == "__main__":
    main(sys.argv[1])
