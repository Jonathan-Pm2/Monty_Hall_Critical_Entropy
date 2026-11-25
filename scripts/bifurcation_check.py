"""Numerical checks for the adaptive host across episodes (Section 8).

Case A2: x' = (1-lam) x + lam * 1{x < 2/3}   (belief on the A1-A2 edge, x = mu_1)
Case A3: x' = (1-lam) x + lam * 1{x < 1/2}   (belief on the A1-A3 edge, x = mu_1)
Conjugacy (case A2) to the contracted rotation g(z) = a z + delta mod 1,
a = 1 - lam, delta = 1 - (1 - lam) c, c = 2/3; long-run 'stay' frequency = rotation number.
Plateaus from Laurent & Nogueira (2018), Theorem 3(ii).
Run: python3 bifurcation_check.py
"""
from fractions import Fraction
from math import floor, gcd


def simulate(lam, c, n=20000, burn=15000, x=0.3):
    seq = []
    for k in range(n):
        sw = x < c
        x = (1 - lam) * x + (lam if sw else 0.0)
        if k >= burn:
            seq.append(1 if sw else 0)
    s = "".join(map(str, seq[-600:]))
    per = next((p for p in range(1, 300) if s[p:] == s[:-p]), None)
    return 1 - sum(seq) / len(seq), per  # stay frequency, period


def ln_plateau(a, p, q):
    """Laurent-Nogueira Thm 3(ii): rho = p/q iff lo <= delta <= hi (0<p<q coprime)."""
    cc = 1.0 + sum((floor((k + 1) * p / q) - floor(k * p / q)) * a**k for k in range(1, q - 1))
    f = (1 - a) / (1 - a**q)
    return f * cc, f * (cc + a ** (q - 1) - a**q)


def ln_rho(lam, c=2 / 3, qmax=60):
    a, d = 1 - lam, 1 - (1 - lam) * c
    for q in range(2, qmax + 1):
        for p in range(1, q):
            if gcd(p, q) == 1:
                lo, hi = ln_plateau(a, p, q)
                if lo - 1e-12 <= d <= hi + 1e-12:
                    return Fraction(p, q)
    return None


print("Case A2 (threshold 2/3): lam, stay freq (sim), period (sim), rho from L-N Thm 3(ii)")
mismatch = []
grid = [k / 100 for k in range(2, 99)]
for lam in grid:
    st, per = simulate(lam, 2 / 3)
    r = ln_rho(lam)
    ok = r is not None and abs(float(r) - st) < 2e-3 and (per is None or per == r.denominator)
    if not ok:
        mismatch.append((lam, round(st, 4), per, str(r)))
    if lam in (0.02, 0.1, 0.2, 0.3, 0.33, 0.34, 0.4, 0.47, 0.48, 0.49, 0.5, 0.51, 0.7, 0.98):
        print("  lam=%.2f stay=%.4f switch=%.4f period=%s rho_LN=%s" % (lam, st, 1 - st, per, r))
print("Case A2: grid points where simulation and L-N disagree: %d of %d" % (len(mismatch), len(grid)))
print("  (lam, stay freq, simulated period, L-N rho with q <= 60):", mismatch)
print("  None means the period exceeds the search bound q <= 60 or the 600-step window.")

# Period-2 border collision: rho = 1/2 iff lam >= 1/2
def is_half(lam):
    a, d = 1 - lam, 1 - (1 - lam) * 2 / 3
    lo, hi = ln_plateau(a, 1, 2)
    return lo <= d <= hi
lo_, hi_ = 0.3, 0.7
for _ in range(80):
    m = (lo_ + hi_) / 2
    lo_, hi_ = (lo_, m) if is_half(m) else (m, hi_)
print("Case A2: rho=1/2 plateau starts at lam = %.12f (expected 0.5); holds at lam=0.999: %s" % (hi_, is_half(0.999)))

# Plateau intervals in lam for small q (from the L-N inequalities)
print("Case A2: plateaus in lam for q <= 7")
for q in range(2, 8):
    for p in range(1, q):
        if gcd(p, q) != 1:
            continue
        inside = [lam for lam in [k / 20000 for k in range(1, 20000)]
                  if (lambda a, d: ln_plateau(a, p, q)[0] <= d <= ln_plateau(a, p, q)[1])(1 - lam, 1 - (1 - lam) * 2 / 3)]
        if inside:
            print("  rho=%d/%d (switch %d/%d): lam in [%.4f, %.4f]" % (p, q, q - p, q, inside[0], inside[-1]))

print("Case A3 (threshold 1/2): stay frequency and period on the grid")
bad = [lam for lam in grid if simulate(lam, 0.5)[1] != 2]
print("  grid points without period 2:", bad)
