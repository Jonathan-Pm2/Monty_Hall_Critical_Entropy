"""Regression probes for the revised manuscript (GSTA-2026-0432, R1).

Three groups of assertions, each printed with a pass/fail summary:
  S6    game model: exact enumeration of the extensive form, including the
        plays in which the host makes no offer (Section 5, Example 6.5).
  S7    entropy and Dirichlet results (Theorem 7.4, Example 7.5,
        Theorem 7.6, Corollary 7.7; Dirichlet parameter beta as in the paper).
  S8.1  adaptive host across episodes: conjugacy to a contracted rotation
        (Lemma 8.8), Laurent-Nogueira plateaus and the two tie-break cases
        (Theorem 8.9).

Conventions (as in the paper): the car C and the contestant's pick X are
uniform; host type A1 offers the switch only when X = C, A2 only when
X != C, A3 always; when the host opens a door it chooses uniformly among
the goat doors distinct from X. B1 = always switch when offered,
B2 = never switch. Without an offer the contestant keeps X.

Python standard library only. Exit code 0 if every assertion holds, 1
otherwise. Run: python3 probes_check.py
"""
import random
import sys
from fractions import Fraction as Fr
from math import floor, gcd, lgamma, log

RESULTS = {}


def check(group, name, ok, detail=""):
    RESULTS.setdefault(group, []).append((name, bool(ok), detail))


# ---------------------------------------------------------------- S6: game
DOORS = (1, 2, 3)
TYPES = ("A1", "A2", "A3")


def offers(host, x, c):
    return {"A1": x == c, "A2": x != c, "A3": True}[host]


def plays(host):
    """Yield (prob, x, c, offer, h) for every play; h = 0 means no door opened."""
    for c in DOORS:
        for x in DOORS:
            p = Fr(1, 9)
            if not offers(host, x, c):
                yield p, x, c, 0, 0
                continue
            goats = [d for d in DOORS if d != x and d != c]
            for h in goats:
                yield p / len(goats), x, c, 1, h


def win_prob(host, gamma):
    """Unconditional winning probability when switching with prob. gamma at every offer."""
    w = Fr(0)
    for p, x, c, o, _ in plays(host):
        if o:
            w += p * (gamma * (x != c) + (1 - gamma) * (x == c))
        else:
            w += p * (x == c)
    return w


def offer_prob(host):
    return sum(p for p, _, _, o, _ in plays(host) if o)


def info_set_posterior(mu, x, h):
    """Posterior over types at the information set {X=x, O=1, H=h}."""
    joint = [m * sum(p for p, xx, _, o, hh in plays(t) if o and xx == x and hh == h)
             for m, t in zip(mu, TYPES)]
    z = sum(joint)
    return tuple(j / z for j in joint)


def p_switch_wins_given_offer(mu):
    num = sum(m * sum(p for p, x, c, o, _ in plays(t) if o and x != c) for m, t in zip(mu, TYPES))
    den = sum(m * offer_prob(t) for m, t in zip(mu, TYPES))
    return num / den


def group_s6():
    g = "S6 game"
    table = {t: (win_prob(t, 1), win_prob(t, 0)) for t in TYPES}
    check(g, "payoff matrix (B1,B2): A1 (0,1/3), A2 (1,1/3), A3 (2/3,1/3)",
          table == {"A1": (0, Fr(1, 3)), "A2": (1, Fr(1, 3)), "A3": (Fr(2, 3), Fr(1, 3))}, table)
    rates = tuple(offer_prob(t) for t in TYPES)
    check(g, "offer probabilities 1/3, 2/3, 1", rates == (Fr(1, 3), Fr(2, 3), 1), rates)

    # Unique maximin: min over hosts of the winning probability is maximised only at gamma = 0.
    gammas = [Fr(k, 40) for k in range(41)]
    worst = {b: min(win_prob(t, b) for t in TYPES) for b in gammas}
    best = max(worst.values())
    check(g, "B2 (gamma=0) is the unique maximin strategy, value 1/3",
          best == Fr(1, 3) and [b for b in gammas if worst[b] == best] == [0])

    # Equilibria of the zero-sum game: exactly gamma = 0 and 2 a2 + a3 <= a1.
    bad = 0
    n = 24
    for i in range(n + 1):
        for j in range(n + 1 - i):
            a = (Fr(i, n), Fr(j, n), Fr(n - i - j, n))
            for b in [Fr(k, 10) for k in range(11)]:
                u = lambda bb, aa=a: sum(w * win_prob(t, bb) for w, t in zip(aa, TYPES))
                contestant_br = u(b) >= max(u(0), u(1))  # payoff linear in gamma
                host_br = u(b) <= min(win_prob(t, b) for t in TYPES)
                is_eq = contestant_br and host_br
                predicted = b == 0 and 2 * a[1] + a[2] <= a[0]
                bad += is_eq != predicted
    check(g, "equilibria exactly (alpha, B2) with 2 alpha2 + alpha3 <= alpha1", bad == 0,
          "%d mismatches" % bad)

    # Bayes versus minimax on random rational priors.
    rng = random.Random(20260432)
    bad = 0
    for _ in range(400):
        k = [rng.randint(0, 30) for _ in range(3)]
        if sum(k) == 0:
            continue
        mu = tuple(Fr(v, sum(k)) for v in k)
        s = sum(m * win_prob(t, 1) for m, t in zip(mu, TYPES))
        stay = sum(m * win_prob(t, 0) for m, t in zip(mu, TYPES))
        m_adv = 2 * mu[1] + mu[2] - mu[0]
        pe = p_switch_wins_given_offer(mu)
        posts = {info_set_posterior(mu, x, h)
                 for x in DOORS for h in DOORS if h != x}
        rates = [offer_prob(t) for t in TYPES]
        expected_post = tuple(m * r / sum(mm * rr for mm, rr in zip(mu, rates))
                              for m, r in zip(mu, rates))
        ok = (stay == Fr(1, 3) and s - Fr(1, 3) == m_adv / 3
              and (s > Fr(1, 3)) == (m_adv > 0) == (pe > Fr(1, 2))
              and (s == Fr(1, 3)) == (pe == Fr(1, 2))
              and posts == {expected_post})
        bad += not ok
    check(g, "Prop 5.6: switch iff 2mu2+mu3>mu1, unconditional and offer-conditional agree, "
             "same posterior at every offer information set", bad == 0, "%d mismatches" % bad)

    # Example 6.5, uniform prior.
    u = (Fr(1, 3),) * 3
    s = sum(m * win_prob(t, 1) for m, t in zip(u, TYPES))
    po = sum(m * offer_prob(t) for m, t in zip(u, TYPES))
    post = info_set_posterior(u, 1, 2)
    pe = p_switch_wins_given_offer(u)
    check(g, "Example 6.5: s=5/9, P(O=1)=2/3, posterior (1/6,1/3,1/2), P(E|O=1)=2/3",
          (s, po, post, pe) == (Fr(5, 9), Fr(2, 3), (Fr(1, 6), Fr(1, 3), Fr(1, 2)), Fr(2, 3)),
          (s, po, post, pe))
    q_a1 = sum(p for p, x, c, o, _ in plays("A1") if o and x != c)
    check(g, "Q_A1(E)=0 given an offer", q_a1 == 0)


# ------------------------------------------------------- S7: entropy, Dirichlet
def digamma(x):
    acc = 0.0
    while x < 6.0:
        acc -= 1.0 / x
        x += 1.0
    x2 = 1.0 / (x * x)
    return acc + log(x) - 0.5 / x - x2 * (1.0 / 12 - x2 * (1.0 / 120 - x2 * (1.0 / 252 - x2 / 240)))


def trigamma(x):
    acc = 0.0
    while x < 6.0:
        acc += 1.0 / (x * x)
        x += 1.0
    x2 = 1.0 / (x * x)
    return acc + 1.0 / x + x2 / 2.0 + (1.0 / x**3) * (1.0 / 6 - x2 * (1.0 / 30 - x2 * (1.0 / 42 - x2 / 30)))


def dir_entropy(a):
    a0 = sum(a)
    return (sum(lgamma(x) for x in a) - lgamma(a0) + (a0 - len(a)) * digamma(a0)
            - sum((x - 1) * digamma(x) for x in a))


def dir_entropy_mc(a, n, rng):
    """-E[log f] with f the density w.r.t. Lebesgue measure on the chart (p1, p2)."""
    lognorm = lgamma(sum(a)) - sum(lgamma(x) for x in a)
    acc = 0.0
    for _ in range(n):
        g = [rng.gammavariate(x, 1.0) for x in a]
        z = sum(g)
        acc -= lognorm + sum((x - 1) * log(v / z) for x, v in zip(a, g))
    return acc / n


def shannon(p):
    return -sum(x * log(x) for x in p if x > 0)


def bisect(fun, lo, hi, n=200):
    flo = fun(lo)
    for _ in range(n):
        mid = (lo + hi) / 2
        fm = fun(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return (lo + hi) / 2


def group_s7():
    g = "S7 entropy/Dirichlet"
    t = bisect(lambda t: 2 * t**3 + t**2 - 1, 0.0, 1.0)
    w = (1 / t, t * t, t)
    mu = tuple(v / sum(w) for v in w)
    h_star = shannon(mu)
    check(g, "mu* from 2t^3+t^2=1 lies on 2mu2+mu3=mu1", abs(2 * mu[1] + mu[2] - mu[0]) < 1e-12)
    check(g, "H* = 0.959626 nats, mu* = (0.582744, 0.165487, 0.251769)",
          abs(h_star - 0.959626) < 1e-6
          and max(abs(a - b) for a, b in zip(mu, (0.582744, 0.165487, 0.251769))) < 1e-6,
          "H*=%.6f mu*=(%.6f, %.6f, %.6f)" % ((h_star,) + mu))

    # Brute force: max of H over {M <= 0}, M = 2mu2 + mu3 - mu1 (staying optimal).
    n = 900
    best = (-1.0, None)
    for i in range(n + 1):
        for j in range(n + 1 - i):
            p = (i / n, j / n, (n - i - j) / n)
            if 2 * p[1] + p[2] - p[0] <= 1e-15:
                v = shannon(p)
                if v > best[0]:
                    best = (v, p)
    v, p = best
    check(g, "brute-force max of H over {M<=0} matches H* and mu*",
          abs(v - h_star) < 1e-4 and max(abs(a - b) for a, b in zip(p, mu)) < 3e-3,
          "grid max %.6f at (%.4f, %.4f, %.4f)" % ((v,) + p))
    u = (1 / 3,) * 3
    check(g, "uniform prior has M>0 and H(uniform) > H*",
          2 * u[1] + u[2] - u[0] > 0 and shannon(u) > h_star)

    # Dirichlet: symmetric subfamily (3c, c, c), first-order condition.
    def dh_sym(c):
        a0 = 5 * c
        return 5 * (a0 - 3) * trigamma(a0) - 3 * (3 * c - 1) * trigamma(3 * c) - 2 * (c - 1) * trigamma(c)

    grid = [0.01 * k for k in range(1, 1001)]
    vals = [dh_sym(c) for c in grid]
    roots = [bisect(dh_sym, grid[i], grid[i + 1]) for i in range(len(grid) - 1)
             if (vals[i] > 0) != (vals[i + 1] > 0)]
    ok = len(roots) == 1 and abs(roots[0] - 0.881442) < 1e-5
    hs = dir_entropy((3 * roots[0], roots[0], roots[0])) if roots else float("nan")
    check(g, "symmetric family: single root c* = 0.8814, h = -1.308083",
          ok and abs(hs + 1.308083) < 1e-5, "roots=%s h=%.6f" % (["%.6f" % r for r in roots], hs))

    # Dirichlet: maximum over the cone beta1 = 2 beta2 + beta3 (grid + pattern search).
    best = None
    for i in range(1, 80):
        for j in range(1, 80):
            a, b = 0.04 * i, 0.04 * j
            v = dir_entropy((2 * a + b, a, b))
            if best is None or v > best[0]:
                best = (v, a, b)
    v, a, b = best
    step = 0.02
    while step > 1e-9:
        moved = False
        for da, db in ((step, 0), (-step, 0), (0, step), (0, -step)):
            na, nb = a + da, b + db
            if na > 0 and nb > 0:
                nv = dir_entropy((2 * na + nb, na, nb))
                if nv > v:
                    v, a, b, moved = nv, na, nb, True
        if not moved:
            step /= 2
    beta_ = (2 * a + b, a, b)
    check(g, "cone maximum h = -1.291594 at (2.5897, 0.8072, 0.9752)",
          abs(v + 1.291594) < 1e-5
          and max(abs(x - y) for x, y in zip(beta_, (2.5897, 0.8072, 0.9752))) < 2e-3,
          "h=%.6f at (%.4f, %.4f, %.4f)" % ((v,) + beta_))

    # Closed form against Monte Carlo (fixes the base measure: chart Lebesgue, mass 1/2).
    rng = random.Random(7)
    mc = dir_entropy_mc((2.5897, 0.8072, 0.9752), 120000, rng)
    check(g, "Monte Carlo entropy agrees with the closed form at the cone maximum",
          abs(mc - dir_entropy((2.5897, 0.8072, 0.9752))) < 5e-3, "MC=%.5f" % mc)
    check(g, "Dir(1,1,1) has h = -log 2", abs(dir_entropy((1.0, 1.0, 1.0)) + log(2)) < 1e-12)


# ------------------------------------------------ S8.1: episodes, rotations
def f_map(x, lam, c):
    return (1 - lam) * x + (lam if x < c else 0.0)


def ln_plateau(a, p, q):
    """Laurent-Nogueira (2018) Thm 3(ii): rotation number p/q iff lo <= delta <= hi."""
    cc = 1.0 + sum((floor((k + 1) * p / q) - floor(k * p / q)) * a**k for k in range(1, q - 1))
    fac = (1 - a) / (1 - a**q)
    return fac * cc, fac * (cc + a ** (q - 1) - a**q)


def rotation_number(a, d, n=6000):
    z = 0.0
    for _ in range(n):
        z = a * (z - floor(z)) + d + floor(z)
    return z / n


def stay_freq(lam, c, x=0.3, n=6000, burn=4000):
    stays = 0
    for k in range(n):
        if k >= burn:
            stays += x >= c
        x = f_map(x, lam, c)
    return stays / (n - burn)


def alternates(lam, c, x, n=3000, tail=200):
    seq = []
    for _ in range(n):
        seq.append(x < c)
        x = f_map(x, lam, c)
    t = seq[-tail:]
    return all(t[i] != t[i + 1] for i in range(len(t) - 1)), x


def group_s81():
    g = "S8.1 conjugacy/plateaus/A3"
    rng = random.Random(81)

    # Lemma 8.8: absorbing interval J, conjugacy to g(z) = a z + delta mod 1.
    err, escapes, not_entered = 0.0, 0, 0
    for _ in range(20000):
        lam, c = rng.uniform(0.01, 0.99), rng.uniform(0.05, 0.95)
        a, d = 1 - lam, 1 - (1 - lam) * c
        lo, hi = a * c, a * c + lam
        x = rng.uniform(lo, hi)
        fx = f_map(x, lam, c)
        if not lo <= fx < hi + 1e-15:
            escapes += 1
            continue
        z = (x - lo) / lam
        gz = (a * z + d) % 1.0
        err = max(err, abs((fx - lo) / lam - gz))
        y = rng.random()
        for _ in range(2000):
            if lo <= y < hi:
                break
            y = f_map(y, lam, c)
        else:
            not_entered += 1
    check(g, "Lemma 8.8: f(J) in J, every orbit enters J, conjugacy error < 1e-12",
          escapes == 0 and not_entered == 0 and err < 1e-12,
          "escapes=%d not_entered=%d max_err=%.1e" % (escapes, not_entered, err))

    # Lift F(z) = a(z - floor z) + delta + floor z: degree one and non-decreasing.
    bad = 0
    for _ in range(5000):
        a, d = rng.uniform(0.01, 0.99), rng.uniform(0.0, 1.0)
        F = lambda z: a * (z - floor(z)) + d + floor(z)
        z = rng.uniform(-3, 3)
        bad += abs(F(z + 1) - F(z) - 1) > 1e-12 or abs((F(z) - a * (z % 1.0) - d + 0.5) % 1.0 - 0.5) > 1e-9
        bad += F(z) > F(z + rng.uniform(0, 1)) + 1e-12
    check(g, "lift F is a degree-one non-decreasing lift of g", bad == 0, "%d failures" % bad)

    # Plateau formula against simulated rotation numbers.
    tested = mism = 0
    for _ in range(3000):
        a, d = rng.uniform(0.05, 0.95), rng.uniform(0.0, 1.0)
        hit = None
        for q in range(2, 41):
            for p in range(1, q):
                if gcd(p, q) == 1:
                    lo, hi = ln_plateau(a, p, q)
                    if lo + 1e-9 < d < hi - 1e-9:
                        hit = Fr(p, q)
                        break
            if hit is not None:
                break
        if hit is None:
            continue
        tested += 1
        mism += abs(rotation_number(a, d) - float(hit)) > 1e-3
    check(g, "Laurent-Nogueira plateaus agree with simulated rotation numbers",
          tested >= 1000 and mism == 0, "%d plateau cases, %d mismatches" % (tested, mism))

    # Case A3 (threshold 1/2): decisions alternate for every lambda; 2-cycle at x_a, x_b.
    bad = 0
    for _ in range(2000):
        lam, x0 = rng.uniform(0.01, 0.99), rng.random()
        alt, _ = alternates(lam, 0.5, x0)
        xa, xb = (1 - lam) / (2 - lam), 1 / (2 - lam)
        x = x0
        for _ in range(3000):
            x = f_map(x, lam, 0.5)
        near = min(abs(x - xa), abs(x - xb)) < 1e-9
        bad += not (alt and near and xa < 0.5 < xb)
    check(g, "case A3: 2-cycle x_a=(1-l)/(2-l), x_b=1/(2-l) for every lambda", bad == 0,
          "%d failures" % bad)

    # Case A2 (threshold 2/3): rho = 1/2 iff lambda >= 1/2; alternation for lambda > 1/2.
    def is_half(lam):
        lo, hi = ln_plateau(1 - lam, 1, 2)
        return lo <= 1 - (1 - lam) * 2 / 3 <= hi

    lo_, hi_ = 0.3, 0.7
    for _ in range(80):
        m = (lo_ + hi_) / 2
        lo_, hi_ = (lo_, m) if is_half(m) else (m, hi_)
    check(g, "case A2: the rho=1/2 plateau starts at lambda = 1/2",
          abs(hi_ - 0.5) < 1e-9 and is_half(0.999), "start=%.12f" % hi_)
    above = all(alternates(rng.uniform(0.501, 0.99), 2 / 3, rng.random())[0] for _ in range(300))
    below = all(abs(stay_freq(lam, 2 / 3) - 0.5) > 1e-3
                for lam in [rng.uniform(0.05, 0.499) for _ in range(150)])
    check(g, "case A2: alternation for lambda > 1/2, stay frequency != 1/2 below",
          above and below, "above=%s below=%s" % (above, below))

    # Plateau edges in lambda quoted in the paper.
    def first_lambda(p, q, start, stop, step=1e-5):
        lam = start
        while lam < stop:
            lo, hi = ln_plateau(1 - lam, p, q)
            if lo <= 1 - (1 - lam) * 2 / 3 <= hi:
                return lam
            lam += step
        return None

    e25, e37 = first_lambda(2, 5, 0.30, 0.40), first_lambda(3, 7, 0.46, 0.48)
    check(g, "plateau edges: stay 2/5 from lambda~0.3262, stay 3/7 from ~0.4702",
          e25 is not None and e37 is not None and abs(e25 - 0.3262) < 2e-4 and abs(e37 - 0.4702) < 2e-4,
          "2/5 at %s, 3/7 at %s" % (e25, e37))
    sim = stay_freq(0.40, 2 / 3)
    check(g, "simulation at lambda=0.40 sits on the 2/5 plateau", abs(sim - 0.4) < 1e-3,
          "stay=%.4f" % sim)


def main():
    for fn in (group_s6, group_s7, group_s81):
        fn()
    failed = 0
    for group, items in RESULTS.items():
        n_ok = sum(ok for _, ok, _ in items)
        print("[%s] %d/%d passed" % (group, n_ok, len(items)))
        for name, ok, detail in items:
            print("  %s %s%s" % ("PASS" if ok else "FAIL", name, ("  (" + str(detail) + ")") if detail else ""))
        failed += len(items) - n_ok
    print("TOTAL: %s" % ("all assertions passed" if failed == 0 else "%d assertion(s) failed" % failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
