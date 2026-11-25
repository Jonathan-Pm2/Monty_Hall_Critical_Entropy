#!/usr/bin/env python3
"""Independent probes for Theorem 8.14(2) and Theorem 8.16 (Section 8.1, case A2, c = 2/3).
Python standard library only. Fixed parameters, no random numbers.
Exit code 0 if every assertion passes, 1 otherwise.

  STURM  For every p/q in (1/3, 1/2) in lowest terms with q <= 40, the endpoint
         polynomials P^-_{p/q} and P^+_{p/q} have exactly one root in (0,1), and it
         is simple (Sturm sequence and gcd(P, P') in exact rational arithmetic).
         A sanity test checks the Sturm count on (x-1/4)(x-1/2)(x-3/4)(x-2).
  BC     For the plateaus 2/5, 3/7, 5/12, 3/8, 7/17 and 9/20, the endpoints are
         computed from the polynomials by exact bisection, and the map
         f(x) = (1-lambda) x + lambda 1{x < c} is iterated in 80-digit decimals.
         At lambda = lambda^- + 10^-k and lambda^+ - 10^-k (k in 3, 5, 7, 9, keeping
         only the k with 10^-k below half the plateau width), the
         orbit closes into a cycle of period q with p stays, and the distance from
         the cycle to c decreases with k (border collision, Theorem 8.16(3)).
         Just outside each plateau the stay frequency differs from p/q.
         The side from which the cycle approaches c is printed as numerical
         information only; it is not asserted and not claimed as proved.

Run: python3 t13_probes.py
"""
import math
import sys
from decimal import Decimal as D, getcontext
from fractions import Fraction as Fr

FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ((" :: " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------- exact polynomials
# Coefficient lists, lowest degree first.
def trim(P):
    P = list(P)
    while len(P) > 1 and P[-1] == 0:
        P.pop()
    return P


def rem(A, B):
    A = trim(A)
    B = trim(B)
    while len(A) >= len(B) and any(A):
        f = A[-1] / B[-1]
        s = len(A) - len(B)
        for k in range(len(B)):
            A[k + s] -= f * B[k]
        A = trim(A[:-1]) if len(A) > 1 else [Fr(0)]
    return trim(A)


def der(P):
    return trim([k * P[k] for k in range(1, len(P))] or [Fr(0)])


def ev(P, x):
    r = Fr(0)
    for c in reversed(P):
        r = r * x + c
    return r


def sturm(P):
    seq = [trim(P), der(P)]
    while len(seq[-1]) > 1:
        r = rem(seq[-2], seq[-1])
        if not any(r):
            break
        seq.append([-c for c in r])
    return seq


def sign_changes(seq, x):
    v = [ev(p, x) for p in seq]
    v = [t for t in v if t != 0]
    return sum(1 for i in range(len(v) - 1) if (v[i] > 0) != (v[i + 1] > 0))


def count_roots(P, a, b):
    """Number of distinct real roots in (a, b], for P(a), P(b) != 0."""
    s = sturm(P)
    return sign_changes(s, a) - sign_changes(s, b)


def gcdpoly(A, B):
    A = trim(A)
    B = trim(B)
    while any(B):
        A, B = B, rem(A, B)
    return A


def Pm(p, q):
    """P^-_{p/q}(a) = Q_q(a) - 3 sum_{i=1}^p a^{n_i - 1}, n_i = floor(i q / p)."""
    P = [Fr(1)] * q
    for i in range(1, p + 1):
        P[(i * q) // p - 1] -= 3
    return P


def Pp(p, q):
    """P^+_{p/q}(a) = Q_q(a) - 3 (sum_{i=1}^{p-1} a^{n_i - 1} + a^{q-2})."""
    P = [Fr(1)] * q
    for i in range(1, p):
        P[(i * q) // p - 1] -= 3
    P[q - 2] -= 3
    return P


def group_sturm(qmax=40):
    P = [Fr(1)]
    for r in (Fr(1, 4), Fr(1, 2), Fr(3, 4), Fr(2)):
        P = [(P[k - 1] if k > 0 else 0) - r * (P[k] if k < len(P) else 0) for k in range(len(P) + 1)]
    check("STURM sanity: (x-1/4)(x-1/2)(x-3/4)(x-2) has 3 roots in (0,1)", count_roots(P, Fr(0), Fr(1)) == 3)
    n = 0
    bad = []
    nonsqf = []
    for q in range(3, qmax + 1):
        for p in range(1, q):
            if math.gcd(p, q) != 1 or not (Fr(1, 3) < Fr(p, q) < Fr(1, 2)):
                continue
            for name, P in (("-", Pm(p, q)), ("+", Pp(p, q))):
                n += 1
                if ev(P, Fr(0)) == 0 or ev(P, Fr(1)) == 0 or count_roots(P, Fr(0), Fr(1)) != 1:
                    bad.append((p, q, name))
                if len(gcdpoly(P, der(P))) > 1:
                    nonsqf.append((p, q, name))
    check("STURM exactly one root in (0,1) for every endpoint polynomial, p/q in (1/3,1/2), q<=%d" % qmax,
          not bad, "%d polynomials, failures %s" % (n, bad[:5]))
    check("STURM every endpoint polynomial is squarefree, q<=%d" % qmax, not nonsqf, str(nonsqf[:5]))


# ---------------------------------------------------------------- border collisions
getcontext().prec = 80
C = D(2) / D(3)


def root01(P, it=200):
    lo, hi = Fr(0), Fr(1)
    slo = ev(P, lo) > 0
    for _ in range(it):
        m = (lo + hi) / 2
        if (ev(P, m) > 0) == slo:
            lo = m
        else:
            hi = m
    return lo


def to_dec(x):
    return D(x.numerator) / D(x.denominator)


def step(x, lam):
    return (1 - lam) * x + (lam if x < C else 0)


def cycle(lam, q, burn=6000):
    """After burn-in, return (stays in q steps, closure error, signed distance to c)."""
    x = D(1) / D(2)
    for _ in range(burn):
        x = step(x, lam)
    pts = []
    stays = 0
    for _ in range(q):
        pts.append(x)
        stays += x >= C
        x = step(x, lam)
    d = min(pts, key=lambda y: abs(y - C))
    return stays, abs(x - pts[0]), d - C


def freq(lam, burn=2000, n=38000):
    x = D(1) / D(2)
    st = 0
    for i in range(burn + n):
        if i >= burn:
            st += x >= C
        x = step(x, lam)
    return Fr(st, n)


def group_bc(plateaus=((2, 5), (3, 7), (5, 12), (3, 8), (7, 17), (9, 20)), ks_all=(3, 5, 7, 9)):
    for p, q in plateaus:
        lm = 1 - to_dec(root01(Pm(p, q)))
        lp = 1 - to_dec(root01(Pp(p, q)))
        ok = D(0) < lm < lp < D(1) / D(2)
        # use only offsets smaller than half the plateau width, so every lambda is inside
        ks = tuple(k for k in ks_all if D(10) ** -k < (lp - lm) / 2)
        ok &= len(ks) >= 2
        left = [cycle(lm + D(10) ** -k, q) for k in ks]
        right = [cycle(lp - D(10) ** -k, q) for k in ks]
        for side in (left, right):
            ok &= all(s == p and cl < D(10) ** -40 for s, cl, _ in side)
            dist = [abs(d) for _, _, d in side]
            ok &= all(u > v for u, v in zip(dist, dist[1:]))
        out_m = freq(lm - D(10) ** -6)
        out_p = freq(lp + D(10) ** -6)
        ok &= out_m != Fr(p, q) and out_p != Fr(p, q)
        check("BC %d/%d: lambda in [%.10f, %.10f], period-%d cycle with %d stays near both endpoints, "
              "distance to c decreasing as lambda -> endpoint, frequency != p/q just outside"
              % (p, q, float(lm), float(lp), q, p), ok)
        print("     info (numerical, not asserted): distance to c at 10^-k inside, k=%s" % (ks,))
        print("       left : %s  side %s" % (" ".join("%.2e" % float(abs(d)) for _, _, d in left),
                                           "above" if left[-1][2] > 0 else "below"))
        print("       right: %s  side %s" % (" ".join("%.2e" % float(abs(d)) for _, _, d in right),
                                           "above" if right[-1][2] > 0 else "below"))
        print("       frequency 1e-6 outside: left %.5f, right %.5f, p/q %.5f"
              % (float(out_m), float(out_p), p / q))


if __name__ == "__main__":
    group_sturm()
    group_bc()
    print("TOTAL: %s" % ("all checks passed" if not FAILS else "%d check(s) failed" % len(FAILS)))
    sys.exit(1 if FAILS else 0)
