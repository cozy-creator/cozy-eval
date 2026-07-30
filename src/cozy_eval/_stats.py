"""The two statistics the population gate needs, with no SciPy in the core.

A paired t-test and Holm-Bonferroni. SciPy would do both, but pulling a 40 MB
compiled dependency into the base install for one continued fraction is not a
trade worth making — the base install has to be importable inside a conversion
worker image that already carries torch and diffusers.
"""

from __future__ import annotations

import math


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + b * math.log1p(-x) + a * math.log(x)
    ) * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    """Two-sided p-value of a t statistic."""
    if df <= 0:
        return 1.0
    if not math.isfinite(t):
        return 0.0
    return float(betainc(df / 2.0, 0.5, df / (df + t * t)))


def paired_t(diffs) -> tuple[float, float]:
    """(t, two-sided p) for a paired sample of differences."""
    n = len(diffs)
    if n < 2:
        return 0.0, 1.0
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    if se <= 0.0:
        return (0.0, 1.0) if mean == 0.0 else (math.inf, 0.0)
    t = mean / se
    return t, t_two_sided_p(t, n - 1)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    order = sorted(pvalues, key=lambda k: pvalues[k])
    k = len(order)
    out: dict[str, float] = {}
    running = 0.0
    for i, name in enumerate(order):
        running = max(running, min(1.0, pvalues[name] * (k - i)))
        out[name] = running
    return out
