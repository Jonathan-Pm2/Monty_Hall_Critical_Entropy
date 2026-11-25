#!/usr/bin/env python3
"""Checks for Lemma plateau-c, Theorem staircase and Theorem border-collisions
(Section 8.1, case A2, c = 2/3).  Python standard library only.
Exit code 0 if every check passes, 1 otherwise.

  CF   exact closed forms: (1-lo)/a = B_{p/q}(a), (1-hi)/a = A_{p/q}(a) against the
       Laurent-Nogueira bounds of (eq:plateau), and the Sturmian series forms
       1-B = (1-a) sum_i a^{floor(iq/p)-1}, 1-A = (1-a) sum_i a^{ceil(iq/p)-2}.
  STURM every endpoint polynomial P^-_{p/q}, P^+_{p/q}, 1/3 < p/q <= 1/2, q <= 40,
       has exactly one root in (0,1), and it is simple (Sturm, exact rationals).
  ORDER plateaus are nondegenerate and ordered like p/q (exact bisection).
  SIM  plateau test vs simulation on lambda = 0.07..0.98; sigma non-decreasing on a fine grid.
  PSI  Step 2-3 of the proof: G_lambda non-decreasing, Psi_lambda increasing in lambda (grid),
       also for K = 1/(1-c) in [3, 6] and K slightly below 3 on compact lambda-ranges.
  BC   border collisions: distance from the attracting cycle to c tends to 0 at both
       endpoints of each plateau with q <= 13; side of approach recorded.
  IRR  irrational value omega = (3 - sqrt 5)/2: root of B_omega(a) = 2/3 and simulated sigma.

Deterministic: no random numbers are used. Prints a summary per group.
Run: python3 staircase_check.py
"""
import math, sys
from fractions import Fraction as Fr

C = Fr(2, 3)
FAILS = []
RESULTS = []
def check(name, ok, detail=""):
    RESULTS.append(name)
    print(("PASS " if ok else "FAIL ") + name + ((" :: " + detail) if detail else ""))
    if not ok: FAILS.append(name)

def rationals(qmax, lo=Fr(1, 3), hi=Fr(1, 2)):
    out = set()
    for q in range(2, qmax + 1):
        for p in range(1, q):
            if math.gcd(p, q) == 1 and lo < Fr(p, q) <= hi: out.add((p, q))
    return sorted(out, key=lambda t: Fr(*t))

# ------------------------------------------------------------- polynomials (coeff lists, low first)
def n_i(i, p, q): return (i * q) // p
def Pminus(p, q):   # Q_q(a) - 3 sum_{i=1}^p a^{n_i-1}
    P = [Fr(1)] * q
    for i in range(1, p + 1): P[n_i(i, p, q) - 1] -= 3
    return P
def Pplus(p, q):    # Q_q(a) - 3 (sum_{i<p} a^{n_i-1} + a^{q-2})
    P = [Fr(1)] * q
    for i in range(1, p): P[n_i(i, p, q) - 1] -= 3
    P[q - 2] -= 3
    return P
def peval(P, x):
    r = Fr(0)
    for c in reversed(P): r = r * x + c
    return r
def trim(P):
    P = list(P)
    while len(P) > 1 and P[-1] == 0: P.pop()
    return P
def deriv(P): return trim([k * P[k] for k in range(1, len(P))] or [Fr(0)])
def prem(A, B):
    A = trim(A); B = trim(B)
    while len(A) >= len(B) and any(A):
        f = A[-1] / B[-1]; s = len(A) - len(B)
        for k in range(len(B)): A[k + s] -= f * B[k]
        A = trim(A[:-1]) if len(A) > 1 else [Fr(0)]
    return A
def sturm_count(P, lo=Fr(0), hi=Fr(1)):
    seq = [trim(P), deriv(P)]
    while any(seq[-1]) and len(seq[-1]) > 1:
        r = prem(seq[-2], seq[-1]); seq.append([-c for c in r])
        if not any(seq[-1]): seq.pop(); break
    def V(x):
        vals = [peval(s, x) for s in seq if any(s)]
        vals = [v for v in vals if v != 0]
        return sum(1 for u, v in zip(vals, vals[1:]) if (u > 0) != (v > 0))
    sturm_count.last_degree = len(trim(seq[-1])) - 1   # 0 iff P is squarefree
    return V(lo) - V(hi)   # distinct roots in (lo, hi]

# ------------------------------------------------------------- CF
def ln_bounds(a, p, q):
    cpq = 1 + sum(((k + 1) * p // q - k * p // q) * a ** k for k in range(1, q - 1))
    f = (1 - a) / (1 - a ** q)
    return f * cpq, f * (cpq + a ** (q - 1) - a ** q)
def B_(a, p, q):
    Q = sum(a ** j for j in range(q))
    return 1 - sum(a ** (n_i(i, p, q) - 1) for i in range(1, p + 1)) / Q
def A_(a, p, q):
    Q = sum(a ** j for j in range(q))
    return B_(a, p, q) - (1 - a) * a ** (q - 2) / Q

def group_cf():
    ok = True; cnt = 0; ok2 = True
    for (p, q) in rationals(25, Fr(0), Fr(1)):
        for a in (Fr(1, 7), Fr(1, 2), Fr(3, 5), Fr(9, 10), Fr(97, 100)):
            lo, hi = ln_bounds(a, p, q)
            ok &= (1 - lo) / a == B_(a, p, q) and (1 - hi) / a == A_(a, p, q); cnt += 1
            # series forms (geometric tail summed exactly over periods)
            per_B = sum(a ** (n_i(i, p, q) - 1) for i in range(1, p + 1))
            per_A = sum(a ** (-(-i * q // p) - 2) for i in range(1, p + 1))
            ok2 &= 1 - B_(a, p, q) == (1 - a) * per_B / (1 - a ** q)
            ok2 &= 1 - A_(a, p, q) == (1 - a) * per_A / (1 - a ** q)
    check("CF (1-lo)/a = B, (1-hi)/a = A, exact, all coprime p/q in (0,1), q<=25", ok, "%d cases" % cnt)
    check("CF Sturmian series forms of 1-B and 1-A", ok2)
    check("CF P^-_{1/2} = 1-2a, P^+_{1/2} = a-2", trim(Pminus(1, 2)) == [1, -2] and trim(Pplus(1, 2)) == [-2, 1])
    check("CF P^-_{2/5}, P^+_{2/5} as in the paper",
          Pminus(2, 5) == [1, -2, 1, 1, -2] and Pplus(2, 5) == [1, -2, 1, -2, 1])
    a = Fr(1, 3)
    check("CF A_{1/3}(a) - 2/3 = (1-a)^2/(3 Q_3(a)) (a=1/3)", A_(a, 1, 3) - C == (1 - a) ** 2 / (3 * (1 + a + a * a)))

# ------------------------------------------------------------- STURM, ORDER
def bisect_root(P, it=80):
    lo, hi = Fr(0), Fr(1); slo = peval(P, lo) > 0
    for _ in range(it):
        m = (lo + hi) / 2
        if (peval(P, m) > 0) == slo: lo = m
        else: hi = m
    return lo, hi

def group_sturm():
    R40 = rationals(40); bad = []; nonsimple = []
    for p, q in R40:
        want = (1, 0) if (p, q) == (1, 2) else (1, 1)   # P^+_{1/2} = a-2 has no root: plateau [1/2,1)
        for P, w in zip((Pminus(p, q), Pplus(p, q)), want):
            if sturm_count(P) != w or peval(P, Fr(1)) == 0: bad.append((p, q))
            if sturm_count.last_degree != 0: nonsimple.append((p, q))
    check("STURM P^- and P^+ have exactly one root in (0,1) (P^+_{1/2}: none), all %d p/q in (1/3,1/2], q<=40" % len(R40),
          not bad, str(bad[:5]))
    check("STURM every endpoint polynomial is squarefree (all roots simple), q<=40", not nonsimple, str(nonsimple[:5]))
    R = rationals(25)
    # endpoints lambda^-_{p/q} = 1 - root(P^-), lambda^+_{p/q} = 1 - root(P^+)
    ends = []
    for p, q in R:
        am = bisect_root(Pminus(p, q), 60); ap = bisect_root(Pplus(p, q), 60)
        lm = (1 - am[1], 1 - am[0]); lp = (1 - ap[1], 1 - ap[0])
        ends.append(((p, q), lm, lp))
    ok = True; detail = ""
    for (pq, lm, lp) in ends:
        if pq == (1, 2):
            ok &= lm[0] <= Fr(1, 2) <= lm[1]
        else:
            ok &= lm[1] < lp[0] and lp[1] < Fr(1, 2)
    for (pq1, lm1, lp1), (pq2, lm2, lp2) in zip(ends, ends[1:]):
        if not lp1[1] < lm2[0]: ok = False; detail = "%s/%s vs %s/%s" % (pq1 + pq2)
    check("ORDER 0<lambda^-<lambda^+<1/2 and lambda^+_{p/q} < lambda^-_{p'/q'} for p/q<p'/q' (q<=25)", ok, detail)
    e = {pq: (float(lm[0]), float(lp[0])) for pq, lm, lp in ends}
    check("ORDER plateau 2/5 = [0.3261584038, 0.4689899435]",
          abs(e[(2, 5)][0] - 0.3261584038) < 1e-9 and abs(e[(2, 5)][1] - 0.4689899435) < 1e-9,
          "%.10f %.10f" % e[(2, 5)])
    check("ORDER closed form lambda^+_{2/5} = (1-sqrt2+sqrt(2sqrt2-1))/2",
          abs(e[(2, 5)][1] - 0.5 * (1 - math.sqrt(2) + math.sqrt(2 * math.sqrt(2) - 1))) < 1e-12)
    tab = {(3, 8): (0.2501, 0.3177), (2, 5): (0.3262, 0.4690), (5, 12): (0.4690, 0.4702),
           (3, 7): (0.4702, 0.4936), (4, 9): (0.4937, 0.4985)}
    check("ORDER paper table (4 decimals)",
          all(round(e[k][0], 4) == v[0] and round(e[k][1], 4) == v[1] for k, v in tab.items()),
          str({k: (round(e[k][0], 5), round(e[k][1], 5)) for k in tab}))
    return e

# ------------------------------------------------------------- simulation
def sim(lam, c=2 / 3, burn=20000, n=200000):
    a = 1 - lam; x = 0.3; st = 0
    for _ in range(burn): x = a * x + (lam if x < c else 0.0)
    for _ in range(n):
        if x >= c: st += 1
        x = a * x + (lam if x < c else 0.0)
    return st / n

def plateau_q(lam, qmax=60):
    a = 1 - lam
    for q in range(2, qmax + 1):
        for p in range(1, q):
            if math.gcd(p, q) != 1: continue
            Q = sum(a ** j for j in range(q))
            B = 1 - sum(a ** ((i * q) // p - 1) for i in range(1, p + 1)) / Q
            A = B - (1 - a) * a ** (q - 2) / Q
            if A <= 2 / 3 <= B: return p, q
    return None

def group_sim():
    agree = 0; tot = 0; bad = []
    for k in range(7, 99):
        lam = k / 100; pq = plateau_q(lam)
        if pq is None: continue
        tot += 1; s = sim(lam, n=60000)
        if abs(s - pq[0] / pq[1]) < 2e-4: agree += 1
        else: bad.append((lam, pq, s))
    check("SIM plateau test (q<=60) vs simulated stay frequency, lambda=0.07..0.98", agree == tot and tot >= 90,
          "%d/%d %s" % (agree, tot, bad[:3]))
    prev = 0; mono = True; lo = 1
    for k in range(1, 1000):
        s = sim(k / 1000, burn=3000, n=40000)
        if s < prev - 2e-4: mono = False
        prev = max(prev, s); lo = min(lo, s)
    check("SIM sigma non-decreasing on lambda=0.001..0.999 (tol 2e-4), sigma>1/3", mono and lo > 1 / 3 - 1e-3)

# ------------------------------------------------------------- PSI (steps 2-3)
def Psi(lam, w, K=3.0):
    a = 1 - lam
    return math.log(lam * K + a ** (w + 1)) / math.log(a)

def group_psi():
    ok = True; okG = True
    ws = [j / 200 for j in range(200)]
    for K in (3.0, 3.5, 4.0, 6.0):
        lams = [k / 400 for k in range(1, 400)]
        for w in ws[::5]:
            vals = [Psi(l, w, K) for l in lams]
            ok &= all(u < v for u, v in zip(vals, vals[1:])) and all(v < 0 for v in vals)
        for l in lams[::7]:
            vals = [Psi(l, w, K) for w in ws]
            okG &= all(u < v for u, v in zip(vals, vals[1:])) and Psi(l, 0.999999, K) - Psi(l, 0, K) < 1
    check("PSI Psi_lambda(w) < 0, strictly increasing in lambda, K in {3,3.5,4,6}", ok)
    check("PSI G_lambda non-decreasing (Psi increasing in w, Psi(1)-Psi(0)<1)", okG)
    # K slightly below 3 on compact lambda ranges [0.2, 0.5]
    ok = True
    for K in (2.99, 2.999):
        lams = [0.2 + k * 0.3 / 300 for k in range(301)]
        for w in ws[::5]:
            vals = [Psi(l, w, K) for l in lams]; ok &= all(u < v for u, v in zip(vals, vals[1:]))
    check("PSI monotone in lambda on [0.2,0.5] for K = 2.99, 2.999 (local perturbation used in Thm border)", ok)
    # failure for small lambda when K < 3 (shows the compactness restriction is needed)
    fails = any(Psi(0.001, 0.99, 2.5) > Psi(0.002, 0.99, 2.5) for _ in [0])
    check("PSI for K=2.5 monotonicity fails near lambda=0 (restriction needed)", fails)

# ------------------------------------------------------------- BC
def cycle_dist(lam, q, c=2 / 3):
    a = 1 - lam; x = 0.3
    for _ in range(int(60 / lam) + 4000): x = a * x + (lam if x < c else 0.0)
    pts = []
    for _ in range(q):
        pts.append(x); x = a * x + (lam if x < c else 0.0)
    d = min(pts, key=lambda y: abs(y - c))
    return abs(d - c), (d >= c)

def group_bc(e):
    ok = True; info = []
    for (p, q), (lm, lp) in e.items():
        if q > 13 or (p, q) == (1, 2): continue
        ds = [cycle_dist(lm + 10 ** -k, q)[0] for k in (3, 4, 5)]
        side_m = cycle_dist(lm + 1e-5, q)[1]
        dp = [cycle_dist(lp - 10 ** -k, q)[0] for k in (3, 4, 5)]
        side_p = cycle_dist(lp - 1e-5, q)[1]
        ok &= ds[0] > ds[1] > ds[2] and ds[2] < 1e-4 and dp[0] > dp[1] > dp[2] and dp[2] < 1e-4
        info.append("%d/%d:left %s,right %s" % (p, q, "above" if side_m else "below", "above" if side_p else "below"))
    check("BC cycle point -> c at both endpoints of every plateau with q<=13", ok, "; ".join(info))
    # at the endpoint itself sigma = p/q (closed plateaus), just outside it differs
    ok = True
    for (p, q), (lm, lp) in e.items():
        if q > 13 or (p, q) == (1, 2): continue
        ok &= abs(sim(lm - 2e-4, n=40000) - p / q) > 1e-3 and abs(sim(lp + 2e-4, n=40000) - p / q) > 1e-3
    check("BC sigma != p/q just outside each plateau (q<=13)", ok)

# ------------------------------------------------------------- IRR
def group_irr():
    om = (3 - math.sqrt(5)) / 2
    def Bw(a):
        s = 0.0; i = 1
        while True:
            e = math.floor(i / om) - 1
            if a ** e < 1e-18: break
            s += a ** e; i += 1
        return 1 - (1 - a) * s
    lo, hi = 0.01, 0.99
    for _ in range(60):
        m = (lo + hi) / 2
        if Bw(m) > 2 / 3: lo = m
        else: hi = m
    lam = 1 - lo; s = sim(lam, burn=50000, n=400000)
    check("IRR omega=(3-sqrt5)/2: B_omega(a)=2/3 at lambda=%.10f, simulated sigma=%.5f" % (lam, s), abs(s - om) < 1e-3)

if __name__ == "__main__":
    group_cf(); e = group_sturm(); group_sim(); group_psi(); group_bc(e); group_irr()
    print()
    for g in ("CF", "STURM", "ORDER", "SIM", "PSI", "BC", "IRR"):
        n = sum(1 for nm in RESULTS if nm.startswith(g + " "))
        f = sum(1 for nm in FAILS if nm.startswith(g + " "))
        print("[%s] %d/%d passed" % (g, n - f, n))
    print("TOTAL: %s" % ("all checks passed" if not FAILS else "%d check(s) failed" % len(FAILS)))
    sys.exit(1 if FAILS else 0)
