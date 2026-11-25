"""Numerical checks for Section 7 (revised indifference set 2*m2 + m3 = m1).

Plain Python, no scipy. Run: python3 section7_check.py
"""
from math import lgamma, log, exp


def trigamma(x):
    acc = 0.0
    while x < 6.0:
        acc += 1.0 / (x * x)
        x += 1.0
    x2 = 1.0 / (x * x)
    return acc + 1.0 / x + x2 / 2.0 + (1.0 / x**3) * (1.0 / 6 - x2 * (1.0 / 30 - x2 * (1.0 / 42 - x2 / 30)))


def digamma(x):
    acc = 0.0
    while x < 6.0:
        acc -= 1.0 / x
        x += 1.0
    x2 = 1.0 / (x * x)
    return acc + log(x) - 0.5 / x - x2 * (1.0 / 12 - x2 * (1.0 / 120 - x2 * (1.0 / 252 - x2 / 240)))


def dir_entropy(a):
    a0 = sum(a)
    k = len(a)
    return (sum(lgamma(x) for x in a) - lgamma(a0) + (a0 - k) * digamma(a0)
            - sum((x - 1) * digamma(x) for x in a))


def bisect(fun, lo, hi, n=200):
    flo = fun(lo)
    for _ in range(n):
        mid = (lo + hi) / 2
        if (fun(mid) > 0) == (flo > 0):
            lo, flo = mid, fun(mid)
        else:
            hi = mid
    return (lo + hi) / 2


# 1. Discrete maximum-entropy prior on 2 m2 + m3 = m1: m proportional to (1/t, t^2, t), 2t^3 + t^2 = 1.
t = bisect(lambda t: 2 * t**3 + t**2 - 1, 0.0, 1.0)
p = [1 / t, t * t, t]
s = sum(p)
p = [x / s for x in p]
H = -sum(x * log(x) for x in p)
print("discrete: t=%.10f p*=(%.6f, %.6f, %.6f)" % (t, *p))
print("discrete: H*=%.6f nats = %.6f bits, H*/log3=%.4f" % (H, H / log(2), H / log(3)))
print("check constraint 2p2+p3-p1 = %.2e, s(p*) = %.6f" % (2 * p[1] + p[2] - p[0], p[1] + 2 * p[2] / 3))

# 2. Dirichlet, symmetric subfamily alpha = (3c, c, c) on the cone alpha1 = 2 alpha2 + alpha3.
def dH_sym(c):
    a0 = 5 * c
    return 5 * (a0 - 3) * trigamma(a0) - 3 * (3 * c - 1) * trigamma(3 * c) - 2 * (c - 1) * trigamma(c)

grid = [0.05 * k for k in range(1, 200)]
signs = [(c, dH_sym(c)) for c in grid]
roots = [bisect(dH_sym, signs[i][0], signs[i + 1][0]) for i in range(len(signs) - 1)
         if (signs[i][1] > 0) != (signs[i + 1][1] > 0)]
print("symmetric: sign changes of dH/dc at", ["%.6f" % r for r in roots])
for c in roots:
    a = (3 * c, c, c)
    print("symmetric: c*=%.6f alpha*=(%.6f, %.6f, %.6f) H=%.6f nats" % (c, *a, dir_entropy(a)))
# numeric derivative check of the analytic first-order condition
c = roots[0]
h = 1e-6
num = (dir_entropy((3 * (c + h), c + h, c + h)) - dir_entropy((3 * (c - h), c - h, c - h))) / (2 * h)
print("symmetric: numeric dH/dc at c* = %.2e" % num)

# 3. Full maximisation on the cone alpha1 = 2a + b, alpha2 = a, alpha3 = b (coarse grid + local refine).
best = None
for i in range(1, 120):
    for j in range(1, 120):
        a, b = 0.025 * i, 0.025 * j
        v = dir_entropy((2 * a + b, a, b))
        if best is None or v > best[0]:
            best = (v, a, b)
v, a, b = best
step = 0.0125
while step > 1e-9:
    improved = False
    for da, db in ((step, 0), (-step, 0), (0, step), (0, -step)):
        na, nb = a + da, b + db
        if na > 0 and nb > 0:
            nv = dir_entropy((2 * na + nb, na, nb))
            if nv > v:
                v, a, b, improved = nv, na, nb, True
    if not improved:
        step /= 2
print("cone max: alpha=(%.6f, %.6f, %.6f) H=%.6f nats" % (2 * a + b, a, b, v))
print("uniform Dir(1,1,1): H=%.6f (=-log 2 = %.6f), s=5/9" % (dir_entropy((1, 1, 1)), -log(2)))
