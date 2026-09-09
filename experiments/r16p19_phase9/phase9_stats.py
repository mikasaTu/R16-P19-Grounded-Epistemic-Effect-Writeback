"""Pure count intervals; no evidence I/O."""
import math
Z95=1.959963984540054

def cp_upper(k: int, n: int, confidence: float = 0.95) -> float | None:
    if n <= 0:
        return None
    if k >= n:
        return 1.0
    if k <= 0:
        return float(1.0 - (1.0 - confidence) ** (1.0 / n))
    tail = 1.0 - confidence

    def logsumexp(values: list[float]) -> float:
        m = max(values)
        return m + math.log(sum(math.exp(v - m) for v in values))

    def cdf(p: float) -> float:
        lp = math.log(p)
        lq = math.log1p(-p)
        terms = [
            math.lgamma(n + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            + i * lp
            + (n - i) * lq
            for i in range(k + 1)
        ]
        return math.exp(min(0.0, logsumexp(terms)))

    lo, hi = float(k) / n, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if cdf(mid) > tail:
            lo = mid
        else:
            hi = mid
    return float(hi)

def wilson(k: int, n: int, z: float = Z95) -> list[float | None]:
    if n <= 0:
        return [None, None]
    p = float(k) / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    w = z * math.sqrt(max(0.0, p * (1.0 - p) / n + z * z / (4.0 * n * n))) / d
    return [float(c - w), float(c + w)]

def binom_lower_tail(k: int, n: int, p: float = 0.5) -> float | None:
    if n <= 0:
        return None
    if k >= n:
        return 1.0
    lp = math.log(p)
    lq = math.log1p(-p)
    values = [
        math.lgamma(n + 1)
        - math.lgamma(i + 1)
        - math.lgamma(n - i + 1)
        + i * lp
        + (n - i) * lq
        for i in range(k + 1)
    ]
    m = max(values)
    return float(math.exp(m) * sum(math.exp(v - m) for v in values))
