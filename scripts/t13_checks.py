"""Checks for the results added in revision T13 (GSTA-2026-0432, R1).

Every value asserted here is quoted in monty_hall_paper_en_R1.tex. Groups:
  L8.13  Lemma 8.13: the plateau inequalities (23) with delta = 1 - a c hold
         iff cbar_lo(a) <= c <= cbar_hi(a) (exact, random rationals), and
         cbar_lo_{1/3}(a) - 2/3 = (1-a)^2 / (3 S_3(a)) > 0.
  T8.14  Theorem 8.14: Psi_lambda(w) increasing in lambda (Step 3, grid),
         sigma non-decreasing on a 400-point lambda grid (tolerance 2e-4),
         1/3 < sigma <= 1/2, sigma = 1/2 for lambda >= 1/2,
         sigma(0.4) = 2/5 and sigma(0.48) = 3/7 (exact plateau test and
         simulation).
  P8.15  Proposition 8.15: the endpoint polynomials equal P^-_{p/q} and
         P^+_{p/q} computed from their definitions, each has exactly one
         root in (0,1) (Sturm, exact), and each stated enclosure of an
         endpoint contains an exact sign change (Fractions); closed form
         of the right endpoint of 2/5; the cubic in t = a + 1/a for 3/7.
  C7.7   Corollary 7.7: the restriction of h to the cone is not concave,
         d^2F/dv^2 > 0 at (u, v) = (1/20, 7/10), certified with the
         interval arithmetic of dirichlet_certify.py.

Python standard library only. Exit code 0 if every assertion holds,
1 otherwise. Run: python3 t13_checks.py (about one second).
"""
import os
import random
import sys
from fractions import Fraction as Fr
from math import floor, gcd, log, sqrt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t10_checks import sturm_roots01  # noqa: E402  (exact Sturm count on (0,1])
import dirichlet_certify as dc  # noqa: E402

RESULTS = []


def check(group, name, ok, detail=""):
    RESULTS.append((group, name, bool(ok), detail))


# ----------------------------------------------------------------- helpers
def n_i(i, p, q):
    return (i * q) // p


def S(q, a):
    return sum(a**j for j in range(q))


def cbar_hi(a, p, q):
    return 1 - sum(a ** (n_i(i, p, q) - 1) for i in range(1, p + 1)) / S(q, a)


def cbar_lo(a, p, q):
    return cbar_hi(a, p, q) - (1 - a) * a ** (q - 2) / S(q, a)


def ln_ineq(a, d, p, q):
    """Inequality (23) of the paper (Laurent-Nogueira 2018, Thm 3(ii))."""
    cc = 1 + sum((((k + 1) * p) // q - (k * p) // q) * a**k for k in range(1, q - 1))
    fac = (1 - a) / (1 - a**q)
    return fac * cc <= d <= fac * (cc + a ** (q - 1) - a**q)


def poly_P(p, q, sign):
    """Coefficients (lowest degree first) of P^-_{p/q} (sign=-1) or P^+_{p/q} (sign=+1)."""
    c = [Fr(1)] * q
    if sign < 0:
        exps = [n_i(i, p, q) - 1 for i in range(1, p + 1)]
    else:
        exps = [n_i(i, p, q) - 1 for i in range(1, p)] + [q - 2]
    for e in exps:
        c[e] -= 3
    while len(c) > 1 and c[-1] == 0:
        c.pop()
    return c


def ev(poly, x):
    return sum(cf * x**i for i, cf in enumerate(poly))


def sigma(lam, n=20000):
    """Rotation number of g(z) = a z + delta mod 1 via the lift, |error| <= 1/n."""
    a, d = 1 - lam, 1 - (1 - lam) * 2 / 3
    z = 0.0
    for _ in range(n):
        fz = floor(z)
        z = a * (z - fz) + d + fz
    return z / n


# ------------------------------------------------------------------ groups
def group_l813():
    g = "L8.13 plateaus as bounds on the threshold"
    rng = random.Random(813)
    ok = True
    tested = 0
    for _ in range(4000):
        q = rng.randint(2, 9)
        p = rng.randint(1, q - 1)
        if gcd(p, q) != 1:
            continue
        a = Fr(rng.randint(1, 199), 200)
        c = Fr(rng.randint(1, 999), 1000)
        ok &= ln_ineq(a, 1 - a * c, p, q) == (cbar_lo(a, p, q) <= c <= cbar_hi(a, p, q))
        tested += 1
    check(g, "(23) with delta = 1 - a c  <=>  cbar_lo(a) <= c <= cbar_hi(a) (exact, seed 813)", ok,
          "%d cases" % tested)
    ok = all(cbar_lo(Fr(k, 100), 1, 3) - Fr(2, 3) == (1 - Fr(k, 100)) ** 2 / (3 * S(3, Fr(k, 100)))
             for k in range(1, 100))
    check(g, "cbar_lo_{1/3}(a) - 2/3 = (1-a)^2/(3 S_3(a)) > 0 at a = 0.01..0.99 (exact)", ok)


def group_t814():
    g = "T8.14 devil's staircase in the memory rate"
    # Step 3: Psi_lambda(w) = log_a(3 lambda + a^(w+1)) increasing in lambda.
    def psi(lam, w):
        a = 1 - lam
        return log(3 * lam + a ** (w + 1)) / log(a)
    ok = all(psi(k / 200, w / 20) < psi((k + 1) / 200, w / 20)
             for k in range(1, 199) for w in range(20))
    check(g, "Step 3: Psi_lambda(w) strictly increasing in lambda (grid 199 x 20)", ok)

    lams = [k / 401 for k in range(1, 401)]
    sig = [sigma(l) for l in lams]
    drops = [i for i in range(len(sig) - 1) if sig[i + 1] < sig[i] - 2e-4]
    check(g, "sigma non-decreasing on a 400-point grid of (0,1) within 2e-4", not drops,
          "violations: %d" % len(drops))
    ok = all(1 / 3 < s <= 0.5 + 1e-4 for s in sig)
    ok &= all(abs(s - 0.5) <= 1e-4 for l, s in zip(lams, sig) if l >= 0.5)
    ok &= all(s < 0.5 - 1e-3 for l, s in zip(lams, sig) if l < 0.49)
    check(g, "1/3 < sigma <= 1/2 on the grid; sigma = 1/2 for lambda >= 1/2", ok,
          "min %.5f at lambda=%.4f" % (min(sig), lams[sig.index(min(sig))]))
    for lam, (p, q) in ((Fr(2, 5), (2, 5)), (Fr(12, 25), (3, 7))):
        a = 1 - lam
        exact = ln_ineq(a, 1 - a * Fr(2, 3), p, q) and cbar_lo(a, p, q) <= Fr(2, 3) <= cbar_hi(a, p, q)
        s = sigma(float(lam), 40000)
        check(g, "sigma(%s) = %d/%d (exact plateau test; simulation within 1/n)" % (float(lam), p, q),
              exact and abs(s - p / q) <= 1 / 40000, "simulated %.6f" % s)


def group_p815():
    g = "P8.15 exact plateau endpoints"
    stated = {
        (1, 2, -1): [1, -2], (1, 2, +1): [-2, 1],
        (2, 5, -1): [1, -2, 1, 1, -2], (2, 5, +1): [1, -2, 1, -2, 1],
        (3, 7, -1): [1, -2, 1, -2, 1, 1, -2], (3, 7, +1): [1, -2, 1, -2, 1, -2, 1],
    }
    ok = all(poly_P(p, q, s) == [Fr(v) for v in coeffs] for (p, q, s), coeffs in stated.items())
    check(g, "stated polynomials equal P^-_{p/q}, P^+_{p/q} from the definition (1/2, 2/5, 3/7)", ok)
    rows = []
    ok = True
    for (p, q, s), coeffs in stated.items():
        if (p, q, s) == (1, 2, +1):
            ok &= all(ev([Fr(v) for v in coeffs], Fr(k, 100)) < 0 for k in range(101))
            rows.append("P+_{1/2} < 0 on [0,1]")
            continue
        nr = sturm_roots01([Fr(v) for v in coeffs])
        ok &= nr == 1
        rows.append("%s%d/%d: %d" % ("-" if s < 0 else "+", p, q, nr))
    check(g, "each endpoint polynomial has exactly one root in (0,1) (Sturm, exact)", ok, "; ".join(rows))

    encl = {
        (1, 2, -1): ("0.5", "0.5"),
        (2, 5, -1): ("0.3261584037872", "0.3261584037873"),
        (2, 5, +1): ("0.4689899435403", "0.4689899435405"),
        (3, 7, -1): ("0.4702144981998", "0.4702144982000"),
        (3, 7, +1): ("0.4936244042056", "0.4936244042057"),
    }
    ok = True
    rows = []
    for (p, q, s), (lo, hi) in encl.items():
        P = poly_P(p, q, s)
        vlo, vhi = ev(P, 1 - Fr(lo)), ev(P, 1 - Fr(hi))
        if lo == hi:
            good = vlo == 0
        else:
            good = (vlo > 0) != (vhi > 0) and vlo != 0 and vhi != 0
        ok &= good
        rows.append("%s%d/%d %s" % ("-" if s < 0 else "+", p, q, "ok" if good else "FAIL"))
    check(g, "exact sign change of P(1-lambda) inside each stated enclosure", ok, "; ".join(rows))

    lam_plus = (1 - sqrt(2) + sqrt(2 * sqrt(2) - 1)) / 2
    a = 1 - lam_plus
    ok = abs(lam_plus - 0.4689899435404) < 5e-13 and abs(1 - 2 * a + a**2 - 2 * a**3 + a**4) < 1e-12
    check(g, "lambda^+_{2/5} = (1 - sqrt2 + sqrt(2 sqrt2 - 1))/2 = 0.4689899435404", ok, "%.13f" % lam_plus)
    # 3/7: t = a + 1/a, largest root of t^3 - 2t^2 - 2t + 2, and a^3 * cubic(t) = P^+_{3/7}(a).
    rng = random.Random(37)
    ok = True
    for _ in range(50):
        x = Fr(rng.randint(1, 999), 1000)
        t = x + 1 / x
        ok &= x**3 * (t**3 - 2 * t**2 - 2 * t + 2) == ev(poly_P(3, 7, +1), x)
    lo, hi = Fr(2), Fr(4)
    cub = lambda t: t**3 - 2 * t**2 - 2 * t + 2
    for _ in range(80):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if cub(m) < 0 else (lo, m)
    t = float(lo)
    a = (t - sqrt(t * t - 4)) / 2
    ok &= cub(Fr(2)) < 0 < cub(Fr(4)) and 0.4936244042056 <= 1 - a <= 0.4936244042057 + 1e-13
    check(g, "3/7 right endpoint: a^3 (t^3-2t^2-2t+2) = P^+_{3/7}(a), largest root gives 0.4936244042056",
          ok, "lambda = %.13f" % (1 - a))


def group_c77():
    g = "C7.7 Dirichlet non-concavity"
    H = dc.hess(dc.cv(Fr(1, 20)), dc.cv(Fr(7, 10)))
    check(g, "d^2F/dv^2 at (u,v) = (1/20, 7/10) > 0 (certified interval), so h is not concave on the cone",
          H[2].lo > 0, "[%s, %s]" % (str(H[2].lo)[:14], str(H[2].hi)[:14]))


def main():
    for fn in (group_l813, group_t814, group_p815, group_c77):
        fn()
    failed = 0
    groups = []
    for grp, _, _, _ in RESULTS:
        if grp not in groups:
            groups.append(grp)
    for grp in groups:
        items = [r for r in RESULTS if r[0] == grp]
        n_ok = sum(r[2] for r in items)
        print("[%s] %d/%d passed" % (grp, n_ok, len(items)))
        for _, name, ok, detail in items:
            print("  %s %s%s" % ("PASS" if ok else "FAIL", name, ("  (" + str(detail) + ")") if detail else ""))
        failed += len(items) - n_ok
    print("TOTAL: %s" % ("all assertions passed" if failed == 0 else "%d assertion(s) failed" % failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
