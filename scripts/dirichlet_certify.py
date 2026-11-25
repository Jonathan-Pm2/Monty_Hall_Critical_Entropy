#!/usr/bin/env python3
"""Computer-assisted certification for Corollary 7.7 (Dirichlet priors on the
indifference cone).  Python standard library only.

Rigour.  All real numbers are enclosed in intervals [lo, hi] of Decimals with
40 significant digits.  +, -, *, / are rounded outward (ROUND_FLOOR for lo,
ROUND_CEILING for hi; the decimal module rounds these operations correctly).
Decimal.ln and Decimal.sqrt are correctly rounded (Python docs), and their
results are widened by two units in the last place.  The special functions
are enclosed with the recurrences
    psi'(x) = sum_{k<n} (x+k)^-2 + psi'(x+n),  psi''(x) = -2 sum_{k<n} (x+k)^-3 + psi''(x+n),
    psi(x)  = psi(x+n) - sum_{k<n} 1/(x+k),   lnGamma(x) = lnGamma(x+n) - ln prod_{k<n}(x+k),
with x+n >= 20 and explicit two-sided bounds at y = x+n:
    T1(y) < psi'(y) < T1(y) + 1/(42 y^7),         T1 = 1/y + 1/(2y^2) + 1/(6y^3) - 1/(30y^5)
    T2(y) - 1/(6y^8) < psi''(y) < T2(y),           T2 = -1/y^2 - 1/y^3 - 1/(2y^4) + 1/(6y^6)
    A(y) - 1/(252 y^6) <= psi(y) <= A(y),          A  = ln y - 1/(2y) - 1/(12y^2) + 1/(120y^4)
    B(y) <= lnGamma(y) <= B(y) + 1/(1260 y^5),     B  = (y-1/2)ln y - y + ln(2pi)/2 + 1/(12y) - 1/(360y^3)
The first two are proved in Part 0 (telescoping + exact polynomial sign check);
the last two follow by integrating them from y to infinity, using
psi(y) - ln y -> 0 and Stirling's formula.
"""
import math, sys, time
from fractions import Fraction as Fr
from decimal import Decimal as D, Context, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_EVEN

PREC = 40
DN = Context(prec=PREC, rounding=ROUND_FLOOR)
UP = Context(prec=PREC, rounding=ROUND_CEILING)
HE = Context(prec=PREC, rounding=ROUND_HALF_EVEN)
NSHIFT = 20

def fail(msg):
    print("CERTIFICATION FAILED:", msg); sys.exit(1)

# ---------------------------------------------------------------- intervals
class I:
    __slots__ = ("lo", "hi")
    def __init__(s, lo, hi=None):
        s.lo = lo; s.hi = lo if hi is None else hi
        if not s.lo <= s.hi: raise ValueError("bad interval")
    def __add__(a, b): b = cv(b); return I(DN.add(a.lo, b.lo), UP.add(a.hi, b.hi))
    __radd__ = __add__
    def __sub__(a, b): b = cv(b); return I(DN.subtract(a.lo, b.hi), UP.subtract(a.hi, b.lo))
    def __rsub__(a, b): return cv(b) - a
    def __neg__(a): return I(-a.hi, -a.lo)
    def __mul__(a, b):
        b = cv(b); p = [(x, y) for x in (a.lo, a.hi) for y in (b.lo, b.hi)]
        return I(min(DN.multiply(x, y) for x, y in p), max(UP.multiply(x, y) for x, y in p))
    __rmul__ = __mul__
    def __truediv__(a, b): return a * recip(cv(b))
    def __rtruediv__(a, b): return cv(b) * recip(a)
    def __repr__(s): return "[%s, %s]" % (s.lo, s.hi)

def cv(x):
    if isinstance(x, I): return x
    if isinstance(x, int): return I(D(x))
    if isinstance(x, D): return I(x)
    if isinstance(x, Fr):
        n, d = D(x.numerator), D(x.denominator)
        return I(DN.divide(n, d), UP.divide(n, d))
    raise TypeError(type(x))

def recip(b):
    if not (b.lo > 0 or b.hi < 0): raise ZeroDivisionError("interval contains 0")
    return I(DN.divide(1, b.hi), UP.divide(1, b.lo))

def ipow(x, k):
    r = x
    for _ in range(k - 1): r = r * x
    return r

def widen(v, ctx_lo, ctx_hi):
    return I(DN.next_minus(DN.next_minus(v)), UP.next_plus(UP.next_plus(v)))

def ln(x):
    x = cv(x)
    if x.lo <= 0: raise ValueError("ln of nonpositive")
    return I(widen(HE.ln(x.lo), DN, UP).lo, widen(HE.ln(x.hi), DN, UP).hi)

def sqrt_hi(v):  # upper bound for sqrt(v), v >= 0 Decimal
    return widen(HE.sqrt(v), DN, UP).hi

def hull(*xs): return I(min(x.lo for x in xs), max(x.hi for x in xs))

# ------------------------------------------------- Part 0: polynomial lemmas
def padd(p, q):
    n = max(len(p), len(q)); return [(p[i] if i < len(p) else 0) + (q[i] if i < len(q) else 0) for i in range(n)]
def pmul(p, q):
    r = [Fr(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q): r[i + j] += a * b
    return r
def ppow(p, k):
    r = [Fr(1)]
    for _ in range(k): r = pmul(r, p)
    return r
def pshift1(p):  # p(1+y)
    r = [Fr(0)]
    for c in reversed(p): r = padd(pmul(r, [Fr(1), Fr(1)]), [c])
    return r

def remainder_numerator(phi, R):
    """E(x) = phi(x) - (R(x) - R(x+1)), phi and R given as {p: coeff} meaning
    sum coeff/x^p.  Returns N with E(x) = N(x) / (x^P (x+1)^P)."""
    P = max(list(phi) + list(R))
    X, X1 = [Fr(0), Fr(1)], [Fr(1), Fr(1)]
    N = [Fr(0)]
    for p, c in phi.items(): N = padd(N, [c * t for t in pmul(ppow(X, P - p), ppow(X1, P))])
    for p, c in R.items():
        N = padd(N, [-c * t for t in pmul(ppow(X, P - p), ppow(X1, P))])
        N = padd(N, [c * t for t in pmul(ppow(X, P), ppow(X1, P - p))])
    return N

def sign_on(N, from_one):
    q = pshift1(N) if from_one else N
    q = [c for c in q if c != 0]
    if q and all(c > 0 for c in q): return +1
    if q and all(c < 0 for c in q): return -1
    return 0

def part0():
    print("Part 0. Tail bounds by telescoping: f(x) - R(x) = sum_k E(x+k), so sign(E) on [1,oo) gives sign(f - R) on [1,oo).")
    tri, tet = {2: Fr(1)}, {3: Fr(-2)}
    L  = {1: Fr(1), 2: Fr(1, 2)}
    U  = {1: Fr(1), 2: Fr(1, 2), 3: Fr(1, 6)}
    T1 = {1: Fr(1), 2: Fr(1, 2), 3: Fr(1, 6), 5: Fr(-1, 30)}
    T1u = dict(T1); T1u[7] = Fr(1, 42)
    T2 = {2: Fr(-1), 3: Fr(-1), 4: Fr(-1, 2), 6: Fr(1, 6)}
    T2l = dict(T2); T2l[8] = Fr(-1, 6)
    cases = [("psi' - (1/x+1/(2x^2))            ", tri, L, +1, False),
             ("psi' - (1/x+1/(2x^2)+1/(6x^3))   ", tri, U, -1, False),
             ("psi' - T1                         ", tri, T1, +1, True),
             ("psi' - T1 - 1/(42x^7)             ", tri, T1u, -1, True),
             ("psi'' - T2                        ", tet, T2, -1, True),
             ("psi'' - T2 + 1/(6x^8)             ", tet, T2l, +1, True)]
    for name, phi, R, want, one in cases:
        s = sign_on(remainder_numerator(phi, R), one)
        dom = "[1,oo)" if one else "(0,oo)"
        print("  %s sign on %s: %+d  %s" % (name, dom, s, "OK" if s == want else "FAIL"))
        if s != want: fail("Part 0 " + name)

# ------------------------------------------------ constants and functions
def atan_inv(m, terms):  # enclosure of atan(1/m), alternating series
    s = Fr(0)
    for k in range(terms): s += Fr((-1) ** k, (2 * k + 1) * m ** (2 * k + 1))
    t = Fr(1, (2 * terms + 1) * m ** (2 * terms + 1))
    return (s - t, s + t)
a5, a239 = atan_inv(5, 40), atan_inv(239, 40)
PI = I(cv(16 * a5[0] - 4 * a239[1]).lo, cv(16 * a5[1] - 4 * a239[0]).hi)
LN2PI = ln(PI * 2)
HALF = cv(Fr(1, 2))

_cache = {}
def _shift(x):
    n = max(0, math.ceil(NSHIFT - float(x)))
    return n, I(x) + n

def tri_pt(x):
    k_ = ("t", x)
    if k_ in _cache: return _cache[k_]
    n, y = _shift(x); X = I(x); s = I(D(0))
    for k in range(n): t = X + k; s = s + recip(t * t)
    iy = recip(y)
    base = iy + HALF * iy * iy + cv(Fr(1, 6)) * ipow(iy, 3) - cv(Fr(1, 30)) * ipow(iy, 5)
    r = s + base + I(D(0), (cv(Fr(1, 42)) * ipow(iy, 7)).hi)
    _cache[k_] = r; return r

def tet_pt(x):
    k_ = ("q", x)
    if k_ in _cache: return _cache[k_]
    n, y = _shift(x); X = I(x); s = I(D(0))
    for k in range(n): s = s - 2 * recip(ipow(X + k, 3))
    iy = recip(y)
    base = -(iy * iy) - ipow(iy, 3) - HALF * ipow(iy, 4) + cv(Fr(1, 6)) * ipow(iy, 6)
    r = s + base - I(D(0), (cv(Fr(1, 6)) * ipow(iy, 8)).hi)
    _cache[k_] = r; return r

def dig_pt(x):
    k_ = ("d", x)
    if k_ in _cache: return _cache[k_]
    n, y = _shift(x); X = I(x); s = I(D(0))
    for k in range(n): s = s + recip(X + k)
    iy = recip(y)
    A = ln(y) - HALF * iy - cv(Fr(1, 12)) * iy * iy + cv(Fr(1, 120)) * ipow(iy, 4)
    r = A - I(D(0), (cv(Fr(1, 252)) * ipow(iy, 6)).hi) - s
    _cache[k_] = r; return r

def lgam_pt(x):
    k_ = ("g", x)
    if k_ in _cache: return _cache[k_]
    n, y = _shift(x); X = I(x); p = I(D(1))
    for k in range(n): p = p * (X + k)
    iy = recip(y)
    B = (y - HALF) * ln(y) - y + HALF * LN2PI + cv(Fr(1, 12)) * iy - cv(Fr(1, 360)) * ipow(iy, 3)
    r = B + I(D(0), (cv(Fr(1, 1260)) * ipow(iy, 5)).hi) - (ln(p) if n else I(D(0)))
    _cache[k_] = r; return r

# interval extensions (x.lo > 0): psi' decreasing, psi'' increasing, psi increasing, lnGamma convex
def tri(x): return I(tri_pt(x.hi).lo, tri_pt(x.lo).hi)
def tet(x): return I(tet_pt(x.lo).lo, tet_pt(x.hi).hi)
def dig(x): return I(dig_pt(x.lo).lo, dig_pt(x.hi).hi)
def lgam(x):
    if x.lo == x.hi: return lgam_pt(x.lo)
    hi = max(lgam_pt(x.lo).hi, lgam_pt(x.hi).hi)
    t1 = lgam_pt(x.lo) + dig_pt(x.lo) * (x - x.lo)   # tangent lines lie below
    t2 = lgam_pt(x.hi) + dig_pt(x.hi) * (x - x.hi)
    return I(max(t1.lo, t2.lo), hi)

# ------------------------------------------------ h on the cone, (u,v)=(b2,b3)
def betas(U, V): return (2 * U + V, U, V), 3 * U + 2 * V

def h_nat(U, V):
    Bs, B0 = betas(U, V)
    r = (B0 - 3) * dig(B0) - lgam(B0)
    for B in Bs: r = r + lgam(B) - (B - 1) * dig(B)
    return r

def grad(U, V):
    Bs, B0 = betas(U, V); c = (B0 - 3) * tri(B0)
    h1, h2, h3 = [c - (B - 1) * tri(B) for B in Bs]
    return 2 * h1 + h2, h1 + h3

def hess(U, V):
    Bs, B0 = betas(U, V)
    s = tri(B0) + (B0 - 3) * tet(B0)
    d1, d2, d3 = [tri(B) + (B - 1) * tet(B) for B in Bs]
    return 9 * s - 4 * d1 - d2, 6 * s - 2 * d1, 4 * s - d1 - d3

def lam_max_hi(H):  # upper bound of the largest eigenvalue of [[a,b],[b,c]]
    a, b, c = H
    amc = a - c; m = max(abs(amc.lo), abs(amc.hi)); bb = max(abs(b.lo), abs(b.hi))
    rad = sqrt_hi(UP.add(UP.multiply(UP.divide(m, 2), UP.divide(m, 2)), UP.multiply(bb, bb)))
    return UP.add(UP.divide(UP.add(a.hi, c.hi), 2), rad)

GAUSS = LN2PI + 1 - HALF * ln(cv(27))   # h <= GAUSS - ln(1+beta0)

def sup_phi(B, w):  # sup of -w/x - ln x on B (B.lo >= 0, B.hi > 0); max at x = w
    X = I(B.hi) if B.hi <= w else (I(B.lo) if B.lo >= w else cv(w))
    return (-(cv(w) / X) - ln(X)).hi

def ub_box(u0, u1, v0, v1):
    U, V = I(cv(u0).lo, cv(u1).hi), I(cv(v0).lo, cv(v1).hi)
    Bs, B0 = betas(U, V)
    best = (GAUSS - ln(B0 + 1)).hi
    # singular split h = S + R (from Gamma(x+1)=x Gamma(x), psi(x+1)=psi(x)+1/x)
    R = (B0 - 3) * dig(B0 + 1) - lgam(B0 + 1)
    for B in Bs: R = R + lgam(B + 1) - (B - 1) * dig(B + 1)
    S2 = UP.add(2, ln(I(B0.hi)).hi)
    for B in Bs: S2 = UP.add(S2, sup_phi(B, Fr(2, 3)))       # uses 3/b0 <= (1/3) sum 1/b_i
    S = S2
    if B0.lo > 0:
        S1 = UP.add(2, (3 / I(B0.lo) + ln(I(B0.hi))).hi)
        for B in Bs: S1 = UP.add(S1, sup_phi(B, 1))
        S = min(S1, S2)
    best = min(best, UP.add(S, R.hi))
    if u0 > 0 and v0 > 0:
        best = min(best, h_nat(U, V).hi)
        cu, cv_ = (u0 + u1) / 2, (v0 + v1) / 2; ru, rv = (u1 - u0) / 2, (v1 - v0) / 2
        Cu, Cv = cv(cu), cv(cv_)
        f0 = h_nat(Cu, Cv); gu, gv = grad(Cu, Cv); a, b, c = hess(U, V)
        mag = lambda x: max(abs(x.lo), abs(x.hi))
        Ru, Rv = cv(ru), cv(rv)
        t = (f0 + mag(gu) * Ru + mag(gv) * Rv
             + HALF * (max(a.hi, D(0)) * Ru * Ru + 2 * mag(b) * Ru * Rv + max(c.hi, D(0)) * Rv * Rv))
        best = min(best, t.hi)
    return best

# ---------------------------------------------------------- Part 1
def g_sym(C):  # d/dc h(3c,c,c)
    return 5 * (5 * C - 3) * tri(5 * C) - 3 * (3 * C - 1) * tri(3 * C) - 2 * (C - 1) * tri(C)
def gp_sym(C):
    a, b, c = hess(C, C); return a + 2 * b + c

def part1():
    print("\nPart 1. Symmetric family beta=(3c,c,c); g(c)=d/dc h(3c,c,c).")
    pi_sq_hi = (PI * PI).hi
    lb = cv(Fr(4, 3)) - I(pi_sq_hi) / 10
    print("  (0,1/5]: c^2 g(c) >= 26/15-2c-(5pi^2/6)(3-5c)c^2 >= 4/3 - pi^2/10 >= %s > 0" % HE.create_decimal(lb.lo).quantize(D('1e-6'), rounding=ROUND_FLOOR))
    if not lb.lo > 0: fail("small-c bound")
    print("  [1,oo):  g(c) <= -1/c + 9/(10c^2) - 1/(50c^3) < 0 (uses Part 0 bounds L < psi' < U)")
    # middle range
    labels = []
    def classify(a, b, depth):
        C = I(cv(a).lo, cv(b).hi); G = g_sym(C)
        if G.lo > 0: labels.append((a, b, "P")); return
        if G.hi < 0: labels.append((a, b, "N")); return
        if gp_sym(C).hi < 0: labels.append((a, b, "D")); return
        if depth > 40: fail("middle range not resolved near %s" % float(a))
        m = (a + b) / 2; classify(a, m, depth + 1); classify(m, b, depth + 1)
    classify(Fr(1, 5), Fr(1), 0)
    labels.sort()
    # a D-interval (g strictly decreasing) with g>0 at its right end has g>0 on it,
    # and one with g<0 at its left end has g<0 on it
    for i, (a, b, l) in enumerate(labels):
        if l == "D":
            if g_sym(cv(b)).lo > 0: labels[i] = (a, b, "P")
            elif g_sym(cv(a)).hi < 0: labels[i] = (a, b, "N")
    pat = "".join(l for _, _, l in labels)
    import re
    if not re.fullmatch(r"P*D+N*", pat): fail("label pattern " + pat)
    Ds = [(a, b) for a, b, l in labels if l == "D"]
    a, b = Ds[0][0], Ds[-1][1]
    print("  [1/5,1]: %d subintervals, pattern %s...; g>0 left of, g'<0 on, g<0 right of [%.6f, %.6f]"
          % (len(labels), pat[:3], float(a), float(b)))
    # g(a) > 0 and g(b) < 0 hold (endpoints of P/N intervals or analytic ranges); bisection
    while b - a > Fr(1, 2 ** 21):
        m = (a + b) / 2; G = g_sym(cv(m))
        if G.lo > 0: a = m
        elif G.hi < 0: b = m
        else: fail("sign of g undecided at %s" % m)
    fa, fb = h_nat(cv(a), cv(a)), h_nat(cv(b), cv(b))
    Gab = g_sym(I(cv(a).lo, cv(b).hi))
    hlo = max(fa.lo, fb.lo); hhi = (fa + max(Gab.hi, D(0)) * cv(b - a)).hi
    print("  unique root c* in [%.9f, %.9f] (width %.2e)" % (float(a), float(b), float(b - a)))
    print("  beta* in [%.7f,%.7f] x [%.7f,%.7f]^2" % (3 * float(a), 3 * float(b), float(a), float(b)))
    print("  h(3c*,c*,c*) in [%s, %s]" % (str(hlo)[:13], str(hhi)[:13]))
    if hhi - hlo > D("1e-6"): fail("h(c*) width")
    return (a, b), (hlo, hhi)

# ---------------------------------------------------------- Part 2
def part2():
    print("\nPart 2. Cone beta=(2u+v,u,v), u,v>0; F(u,v)=h(beta).")
    # approximate maximiser (floating point Newton; only a guess, then certified)
    u, v = 0.8, 1.0
    for _ in range(30):
        gu, gv = grad(cv(Fr(u)), cv(Fr(v))); a, b, c = hess(cv(Fr(u)), cv(Fr(v)))
        gu, gv, a, b, c = [float(x.lo) for x in (gu, gv, a, b, c)]
        det = a * c - b * b; u -= (c * gu - b * gv) / det; v -= (-b * gu + a * gv) / det
    xu, xv = Fr(round(u * 2 ** 40), 2 ** 40), Fr(round(v * 2 ** 40), 2 ** 40)
    Flo = h_nat(cv(xu), cv(xv)).lo
    print("  F(xhat) >= %s at xhat=(%.10f, %.10f)" % (str(Flo)[:14], float(xu), float(xv)))
    print("  beta0 >= 12: F <= ln(2 pi e) - ln(27)/2 - ln(13) <= %s < F(xhat)  (Gaussian max-entropy bound)"
          % str((GAUSS - ln(cv(13))).hi)[:9])
    if not (GAUSS - ln(cv(13))).hi < Flo: fail("gaussian")
    wmin = Fr(1, 2 ** 10); stack = [(Fr(0), Fr(4), Fr(0), Fr(6))]; keep = []; nbox = 0
    while stack:
        u0, u1, v0, v1 = stack.pop(); nbox += 1
        if 3 * u0 + 2 * v0 >= 12: continue          # whole box has beta0 >= 12
        if ub_box(u0, u1, v0, v1) < Flo: continue
        if max(u1 - u0, v1 - v0) <= wmin: keep.append((u0, u1, v0, v1)); continue
        if u1 - u0 >= v1 - v0: m = (u0 + u1) / 2; stack += [(u0, m, v0, v1), (m, u1, v0, v1)]
        else: m = (v0 + v1) / 2; stack += [(u0, u1, v0, m), (u0, u1, m, v1)]
    U0, U1 = min(b[0] for b in keep), max(b[1] for b in keep)
    V0, V1 = min(b[2] for b in keep), max(b[3] for b in keep)
    print("  branch and bound: %d boxes examined, %d kept; every maximiser lies in R=[%.6f,%.6f]x[%.6f,%.6f]"
          % (nbox, len(keep), float(U0), float(U1), float(V0), float(V1)))
    if not (U0 > 0 and V0 > 0): fail("R touches the boundary")
    # strict concavity on R
    k = 8; lam = None
    for i in range(k):
        for j in range(k):
            a0, a1 = U0 + (U1 - U0) * i / k, U0 + (U1 - U0) * (i + 1) / k
            b0, b1 = V0 + (V1 - V0) * j / k, V0 + (V1 - V0) * (j + 1) / k
            l = lam_max_hi(hess(I(cv(a0).lo, cv(a1).hi), I(cv(b0).lo, cv(b1).hi)))
            lam = l if lam is None else max(lam, l)
    m = -lam
    print("  Hessian on R: largest eigenvalue <= %s < 0, so F is strictly concave on R" % str(lam)[:9])
    if not m > 0: fail("concavity")
    if not (U0 <= xu <= U1 and V0 <= xv <= V1): fail("xhat not in R")
    gu, gv = grad(cv(xu), cv(xv))
    gn = sqrt_hi(UP.add(UP.multiply(max(abs(gu.lo), abs(gu.hi)), max(abs(gu.lo), abs(gu.hi))),
                        UP.multiply(max(abs(gv.lo), abs(gv.hi)), max(abs(gv.lo), abs(gv.hi)))))
    rho = UP.divide(gn, m); dF = UP.divide(UP.multiply(gn, gn), UP.multiply(2, m))
    Fhi = UP.add(h_nat(cv(xu), cv(xv)).hi, dF)
    print("  |grad F(xhat)| <= %.2e, so |x* - xhat| <= %.2e and F(x*) - F(xhat) <= %.2e" % (gn, rho, dF))
    print("  maximiser (b2,b3)* = (%.10f, %.10f) +- %.1e" % (float(xu), float(xv), float(rho)))
    print("  beta* = (%.10f, %.10f, %.10f), each coordinate +- %.1e" % (2 * float(xu) + float(xv), float(xu), float(xv), 3 * float(rho)))
    print("  max h on cone in [%s, %s]" % (str(Flo)[:14], str(Fhi)[:14]))
    if rho > D("1e-6") or Fhi - Flo > D("1e-6"): fail("precision")
    print("  R exact (dyadic): [%s, %s] x [%s, %s]" % (U0, U1, V0, V1))
    # non-concavity of F on the cone: d^2F/dv^2 > 0 at (u,v) = (1/20, 7/10)
    Hn = hess(cv(Fr(1, 20)), cv(Fr(7, 10)))
    print("  non-concavity: d^2F/dv^2 at (u,v)=(1/20,7/10) in [%s, %s] > 0" % (str(Hn[2].lo)[:14], str(Hn[2].hi)[:14]))
    assert Hn[2].lo > 0, "non-concavity not certified"
    if not Hn[2].lo > 0: fail("non-concavity")

if __name__ == "__main__":
    t0 = time.time()
    part0(); part1(); part2()
    print("\nALL CLAIMS CERTIFIED (%.1f s)" % (time.time() - t0))
