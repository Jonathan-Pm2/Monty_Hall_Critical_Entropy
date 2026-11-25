"""Checks for the numbers added in revision T10 (GSTA-2026-0432, R1).

Every value asserted here is quoted in monty_hall_paper_en_R1.tex. Groups:
  T3.2   Theorem 3.2 and Corollary 3.3: the win probability of always
         switching is 2/3 for every tie-break r = pi(h|x,x), the
         conditional probability is q = 1/(1+r), coverage of a wrong
         model's interval tends to 0, cell frequencies recover q.
  T7     Theorem 7.4 endpoint entropies and the Dirichlet entropy of
         Theorem 7.6(3): E[log mu_i] = psi(beta_i) - psi(beta_0), the
         derived formula equals the displayed one, Monte Carlo agrees,
         Dir(1,1,1) has h = -log 2 (nu has mass 1/2).
  T8.8   Lemma 8.8: discounted Dirichlet counts give the memory rule (20).
  T8.10  Theorem 8.10: case A2 at lambda = 1/2 in exact arithmetic,
         the bound sigma <= (M+1)/(2M+3) below 1/2, case A3 distance
         lambda/(2(2-lambda)) from 1/2.
  P8.12  Proposition 8.12: switching-frequency identity and bounds on a
         lambda grid.
  PLAT   Plateau endpoints as roots of explicit polynomials (table of
         Section 8.1, and the plateaus n/(2n+1)), cycles at 19 interior
         points, border collisions at the endpoints of 2/5 and 3/7, the
         92-of-97 grid comparison and monotonicity of sigma on a fine grid.
  L8.11  Lemma 8.11 (a genuine cycle inside a plateau): Sturm count of
         the endpoint polynomials (one root in (0,1) each), the
         inequality Phi_delta >= Phi_delta' + (delta - delta') of step
         (ii), and the zero y* of Phi_delta built as in step (iv), in
         exact rational arithmetic, giving a cycle of g of period q with
         p wraps that avoids 0.
  T9.1   Theorem 9.1 (n doors, neutral host), Table 1, Corollary 9.2.

Python standard library only, fixed seeds. Exit code 0 if every
assertion holds, 1 otherwise. Run: python3 t10_checks.py (a few seconds).
"""
import random
import sys
from fractions import Fraction as Fr
from itertools import combinations
from math import comb, exp, floor, gcd, lgamma, log, sqrt

RESULTS = {}


def check(group, name, ok, detail=""):
    RESULTS.setdefault(group, []).append((name, bool(ok), detail))


# ------------------------------------------------------------------ helpers
def digamma(x):
    acc = 0.0
    while x < 6.0:
        acc -= 1.0 / x
        x += 1.0
    x2 = 1.0 / (x * x)
    return acc + log(x) - 0.5 / x - x2 * (1.0 / 12 - x2 * (1.0 / 120 - x2 * (1.0 / 252 - x2 / 240)))


def shannon(p):
    return -sum(v * log(v) for v in p if v > 0)


def f_map(x, lam, c):
    """Episode map, equation (21): x' = (1-lam) x + lam 1{x < c}."""
    return (1 - lam) * x + (lam if x < c else 0)


def ln_plateau(a, p, q):
    """Laurent-Nogueira (2018) Thm 3(ii): rho = p/q iff lo <= delta <= hi."""
    cc = 1.0 + sum((((k + 1) * p) // q - (k * p) // q) * a**k for k in range(1, q - 1))
    fac = (1 - a) / (1 - a**q)
    return fac * cc, fac * (cc + a ** (q - 1) - a**q)


def on_plateau(lam, p, q, c=2 / 3):
    lo, hi = ln_plateau(1 - lam, p, q)
    return lo <= 1 - (1 - lam) * c <= hi


# ------------------------------------------------- T3.2: Theorem 3.2, Cor 3.3
def plays_tiebreak(r):
    """Yield (prob, x, c, h) under uniform C, X; when X=C the host opens the
    lower-index other door with probability r (this is pi(h1|x,x) = r)."""
    for c in (1, 2, 3):
        for x in (1, 2, 3):
            p = Fr(1, 9)
            others = [d for d in (1, 2, 3) if d != x]
            if x == c:
                yield p * r, x, c, others[0]
                yield p * (1 - r), x, c, others[1]
            else:
                h = next(d for d in others if d != c)
                yield p, x, c, h


def group_t32():
    g = "T3.2 what the win frequency identifies"
    rs = [Fr(k, 20) for k in range(21)]
    ok_p = ok_q = True
    cell_min = Fr(1)
    for r in rs:
        pl = list(plays_tiebreak(r))
        p_win = sum(pr for pr, x, c, h in pl if c == 6 - x - h)
        ok_p &= p_win == Fr(2, 3)
        for x in (1, 2, 3):
            for h in (1, 2, 3):
                if h == x:
                    continue
                cell = sum(pr for pr, xx, c, hh in pl if xx == x and hh == h)
                win = sum(pr for pr, xx, c, hh in pl if xx == x and hh == h and c == 6 - x - h)
                cell_min = min(cell_min, cell)
                pi_h = r if h == min(d for d in (1, 2, 3) if d != x) else 1 - r
                ok_q &= win / cell == 1 / (1 + pi_h)
    check(g, "p_theta = 2/3 for every tie-break r in {0, 1/20, ..., 1}", ok_p)
    check(g, "q_theta(x,h) = 1/(1 + pi(h|x,x)) at every (x,h), every r", ok_q)
    check(g, "q = 2/3 under MH (r=1/2); q in {1/2, 1} under MC (r in {0,1}); range [1/2,1]",
          1 / (1 + Fr(1, 2)) == Fr(2, 3) and {1 / (1 + Fr(0)), 1 / (1 + Fr(1))} == {1, Fr(1, 2)})
    check(g, "Cor 3.3: P(X=x, H=h) >= 1/9 for every cell and every r", cell_min == Fr(1, 9),
          "min cell prob = %s" % cell_min)

    # Thm 3.2(3): analyst assumes MH, T_n = V_n/n -> 2/3; truth MC with pi(h|x,x)=1, q0 = 1/2.
    # Coverage of [T_n - w_n, T_n + w_n], w_n = n^(-1/3), of q0 = 1/2 under Binom(n, 2/3), exact.
    def coverage(n):
        w = n ** (-1 / 3)
        tot = 0.0
        for v in range(n + 1):
            if abs(v / n - 0.5) <= w:
                tot += exp(lgamma(n + 1) - lgamma(v + 1) - lgamma(n - v + 1)
                           + v * log(2 / 3) + (n - v) * log(1 / 3))
        return tot
    covs = [(n, coverage(n)) for n in (10**2, 10**3, 10**4, 10**5)]
    check(g, "Thm 3.2(3): coverage of q0=1/2 by the MH-centred interval tends to 0",
          covs[-1][1] < 1e-6 and covs[2][1] < covs[1][1],
          ", ".join("n=%d: %.3g" % c for c in covs))

    # Cor 3.3 second part: cell win proportion converges to q0 without a host model.
    rng = random.Random(32)
    r = 0.3
    n = 200000
    cnt = wins = 0
    for _ in range(n):
        c, x = rng.randint(1, 3), rng.randint(1, 3)
        others = [d for d in (1, 2, 3) if d != x]
        if x == c:
            h = others[0] if rng.random() < r else others[1]
        else:
            h = next(d for d in others if d != c)
        if x == 1 and h == 2:
            cnt += 1
            wins += c == 3
    q0 = 1 / (1 + r)
    se = sqrt(q0 * (1 - q0) / cnt)
    check(g, "Cor 3.3: win proportion in the cell (x,h)=(1,2) -> 1/(1+r) (r=0.3, seed 32)",
          abs(wins / cnt - q0) < 4 * se, "%.4f vs %.4f (se %.4f)" % (wins / cnt, q0, se))


# ------------------------------------------- T7: endpoints, Dirichlet entropy
def dir_entropy_displayed(b):
    b0 = sum(b)
    return (sum(lgamma(x) for x in b) - lgamma(b0) + (b0 - 3) * digamma(b0)
            - sum((x - 1) * digamma(x) for x in b))


def dir_entropy_derived(b):
    """h = log B(beta) - sum (beta_i - 1)(psi(beta_i) - psi(beta_0))."""
    b0 = sum(b)
    log_b = sum(lgamma(x) for x in b) - lgamma(b0)
    return log_b - sum((x - 1) * (digamma(x) - digamma(b0)) for x in b)


def group_t7():
    g = "T7 entropy endpoints and Dirichlet derivation"
    check(g, "Thm 7.4 endpoints: H(1/2,0,1/2) = log 2 = 0.693, H(2/3,1/3,0) = log 3 - (2/3) log 2 = 0.637",
          abs(shannon((0.5, 0, 0.5)) - log(2)) < 1e-15
          and abs(shannon((2 / 3, 1 / 3, 0)) - (log(3) - 2 / 3 * log(2))) < 1e-15
          and round(log(2), 3) == 0.693 and round(log(3) - 2 / 3 * log(2), 3) == 0.637)

    rng = random.Random(76)
    worst = 0.0
    for _ in range(2000):
        b = [rng.uniform(0.05, 8.0) for _ in range(3)]
        worst = max(worst, abs(dir_entropy_derived(b) - dir_entropy_displayed(b)))
    check(g, "derived h = log B - sum(beta_i-1)(psi_i - psi_0) equals the displayed formula",
          worst < 1e-10, "max diff %.1e on 2000 random beta" % worst)

    # d/dbeta_i log B(beta) = psi(beta_i) - psi(beta_0) (differentiation under the integral).
    worst = 0.0
    for _ in range(500):
        b = [rng.uniform(0.1, 6.0) for _ in range(3)]
        for i in range(3):
            e = 1e-5
            bp, bm = list(b), list(b)
            bp[i] += e
            bm[i] -= e
            lb = lambda v: sum(lgamma(x) for x in v) - lgamma(sum(v))
            worst = max(worst, abs((lb(bp) - lb(bm)) / (2 * e) - (digamma(b[i]) - digamma(sum(b)))))
    check(g, "d log B / d beta_i = psi(beta_i) - psi(beta_0) (finite differences)", worst < 1e-6,
          "max diff %.1e" % worst)

    # Monte Carlo: E[log mu_i] and h = -E[log rho] with rho the density w.r.t. nu.
    rng = random.Random(7)
    details = []
    ok = True
    for b in ((1.0, 1.0, 1.0), (2.644, 0.881, 0.881), (2.590, 0.807, 0.975), (5.0, 2.0, 0.5)):
        n = 100000
        b0 = sum(b)
        log_b = sum(lgamma(x) for x in b) - lgamma(b0)
        s_log = [0.0, 0.0, 0.0]
        s_log2 = [0.0, 0.0, 0.0]
        s_h = s_h2 = 0.0
        for _ in range(n):
            gs = [rng.gammavariate(x, 1.0) for x in b]
            z = sum(gs)
            lm = [log(v / z) for v in gs]
            for i in range(3):
                s_log[i] += lm[i]
                s_log2[i] += lm[i] ** 2
            hv = log_b - sum((x - 1) * l for x, l in zip(b, lm))  # -log rho
            s_h += hv
            s_h2 += hv * hv
        for i in range(3):
            m = s_log[i] / n
            se = sqrt(max(s_log2[i] / n - m * m, 0) / n)
            ok &= abs(m - (digamma(b[i]) - digamma(b0))) < 5 * se + 1e-12
        mh = s_h / n
        se_h = sqrt(max(s_h2 / n - mh * mh, 0) / n)
        ok &= abs(mh - dir_entropy_displayed(b)) < 5 * se_h + 1e-12
        details.append("beta=%s h=%.4f MC=%.4f" % (b, dir_entropy_displayed(b), mh))
    check(g, "Monte Carlo (seed 7, 1e5 draws): E[log mu_i] and h agree with the closed forms", ok,
          "; ".join(details))
    check(g, "Dir(1,1,1): h = -log 2 = -0.693 (nu has mass 1/2)",
          abs(dir_entropy_displayed((1.0, 1.0, 1.0)) + log(2)) < 1e-12)
    check(g, "symmetric point beta* = (3c*, c*, c*), c* = 0.8814: h = -1.308",
          round(dir_entropy_displayed((3 * 0.881442, 0.881442, 0.881442)), 3) == -1.308)


# ------------------------------------------------------- T8.8: Lemma 8.8
def group_t88():
    g = "T8.8 discounted Dirichlet counts"
    rng = random.Random(88)
    ok = True
    for _ in range(200):
        lam = Fr(1, rng.randint(2, 20))
        w = [rng.randint(1, 50) for _ in range(3)]
        n0 = [Fr(v, sum(w)) / lam for v in w]  # total 1/lam
        mu = [lam * v for v in n0]
        cnt = list(n0)
        for _ in range(40):
            t = rng.randrange(3)
            cnt = [(1 - lam) * v + (1 if i == t else 0) for i, v in enumerate(cnt)]
            mu = [(1 - lam) * v + lam * (1 if i == t else 0) for i, v in enumerate(mu)]
            ok &= sum(cnt) == 1 / lam and [lam * v for v in cnt] == mu
            # Dir(N) mean is N / sum N.
            ok &= [v / sum(cnt) for v in cnt] == mu
            # Weight of the new observation in the mean stays lambda.
    check(g, "totals stay 1/lambda and lambda*N_n follows the memory rule (exact)", ok)

    # Without discounting the n-th observation has weight 1/(sum N_0 + n) in the mean.
    s0 = Fr(7, 2)
    cnt = [Fr(1), Fr(3, 2), Fr(1)]
    ok = True
    for n in range(1, 30):
        before = [v / (s0 + n - 1) for v in cnt]
        cnt[0] += 1
        after = [v / (s0 + n) for v in cnt]
        wgt = Fr(1, 1) / (s0 + n)
        ok &= after == [(1 - wgt) * b + wgt * (1 if i == 0 else 0) for i, b in enumerate(before)]
    check(g, "without discounting the n-th observation has weight 1/(sum N_0 + n) -> 0", ok)

    # Bayesian update: Dir(N) prior, theta ~ mu, posterior Dir(N + e_theta) (Monte Carlo means).
    rng = random.Random(881)
    big_n = (1.5, 0.7, 2.3)
    sums = {t: [0.0, 0.0, 0.0] for t in range(3)}
    cnts = {t: 0 for t in range(3)}
    for _ in range(300000):
        gs = [rng.gammavariate(x, 1.0) for x in big_n]
        z = sum(gs)
        mu = [v / z for v in gs]
        u = rng.random()
        t = 0 if u < mu[0] else (1 if u < mu[0] + mu[1] else 2)
        cnts[t] += 1
        for i in range(3):
            sums[t][i] += mu[i]
    worst = 0.0
    for t in range(3):
        post = [v + (1 if i == t else 0) for i, v in enumerate(big_n)]
        for i in range(3):
            worst = max(worst, abs(sums[t][i] / cnts[t] - post[i] / sum(post)))
    check(g, "Monte Carlo (seed 881): posterior mean after a draw t is that of Dir(N + e_t)",
          worst < 5e-3, "max |diff| = %.4f" % worst)


# ----------------------------------------------- T8.10: Theorem 8.10 numbers
def group_t810():
    g = "T8.10 period-2 plateau, sigma bound, case A3"
    # Exact arithmetic at lambda = 1/2, c = 2/3.
    lam, c = Fr(1, 2), Fr(2, 3)
    ok = True
    for x0 in (Fr(0), Fr(1, 3), Fr(3, 10), Fr(1, 2), Fr(7, 10), Fr(99, 100), Fr(1)):
        x = x0
        seq = []
        for _ in range(80):
            seq.append(x)
            x = f_map(x, lam, c)
        j_lo, j_hi = (1 - lam) * c, (1 - lam) * c + lam
        k0 = next(i for i, v in enumerate(seq) if j_lo <= v < j_hi)
        tail = seq[k0:]
        stays = [v >= c for v in tail]
        ok &= all(stays[i] != stays[i + 1] for i in range(len(stays) - 1))
        sp = [v for v in tail if v >= c]
        wp = [v for v in tail if v < c]
        ok &= all(sp[i] >= sp[i + 1] for i in range(len(sp) - 1)) and min(sp) >= c
        ok &= all(wp[i] >= wp[i + 1] for i in range(len(wp) - 1)) and min(wp) >= Fr(1, 3)
        ok &= sp[-1] - c < Fr(1, 10**20) and wp[-1] - Fr(1, 3) < Fr(1, 10**20)
    ok &= f_map(Fr(1, 3), lam, c) == c and 1 / (2 - lam) == c
    check(g, "lambda=1/2 (exact): decisions alternate from entry into J, stays decrease to x_b=2/3, "
             "switches to x_a=1/3, f(1/3)=2/3 is a stay", ok)

    # Bounds used in the proof of (3): stay -> switch for all lambda; switch -> stay iff lambda >= 1/2.
    ok = True
    for k in range(1, 100):
        lam = Fr(k, 100)
        a = 1 - lam
        ok &= a * (Fr(2, 3) * a + lam) < Fr(2, 3)
        ok &= ((a * a * Fr(2, 3) + lam - Fr(2, 3)) == lam / 3 * (2 * lam - 1))
    check(g, "proof of (3): (1-l)(2/3(1-l)+l) < 2/3 and (1-l)^2 2/3 + l - 2/3 = l(2l-1)/3", ok)

    # sigma <= (M+1)/(2M+3) for lambda < 1/2: every window of 2M+3 episodes after entry into J
    # has at most M+1 stays, and no two consecutive stays.
    rows = []
    ok = True
    rng = random.Random(810)
    lams = [k / 100 for k in range(1, 50)] + [rng.uniform(0.01, 0.4999) for _ in range(30)]
    for lam in lams:
        c = 2 / 3
        xb = 1 / (2 - lam)
        big_m = floor(log(2 / 3 - xb) / (2 * log(1 - lam)))
        bound = (big_m + 1) / (2 * big_m + 3)
        x = rng.random()
        lo, hi = (1 - lam) * c, (1 - lam) * c + lam
        while not lo <= x < hi:
            x = f_map(x, lam, c)
        st = []
        for _ in range(30000):
            st.append(1 if x >= c else 0)
            x = f_map(x, lam, c)
        w = 2 * big_m + 3
        run = sum(st[:w])
        mx = run
        for i in range(w, len(st)):
            run += st[i] - st[i - w]
            mx = max(mx, run)
        no_double = all(not (st[i] and st[i + 1]) for i in range(len(st) - 1))
        sig = sum(st[10000:]) / 20000
        ok &= mx <= big_m + 1 and no_double and bound < 0.5 and sig <= bound + 1e-9
        if lam in (0.1, 0.2, 0.3, 0.4, 0.45, 0.49):
            rows.append("l=%.2f M=%d bound=%.4f sigma=%.4f" % (lam, big_m, bound, sig))
    check(g, "lambda<1/2: <= M+1 stays per 2M+3 episodes, no consecutive stays, sigma <= (M+1)/(2M+3) < 1/2",
          ok, "; ".join(rows))

    # Case A3: x_a and x_b at distance lambda/(2(2-lambda)) from 1/2 (no border collision).
    ok = True
    for k in range(1, 100):
        lam = Fr(k, 100)
        xa, xb = (1 - lam) / (2 - lam), 1 / (2 - lam)
        d = lam / (2 * (2 - lam))
        ok &= xb - Fr(1, 2) == d and Fr(1, 2) - xa == d
        ok &= (1 - lam) * xa + lam == xb and (1 - lam) * xb == xa
    check(g, "case A3: x_b - 1/2 = 1/2 - x_a = lambda/(2(2-lambda)) > 0 (exact, lambda = k/100)", ok)
    check(g, "Rem 8.13: long-run payoff sigma/3 <= 1/6 < 1/3", Fr(1, 2) / 3 == Fr(1, 6))


# --------------------------------------------- P8.12: Proposition 8.12 bounds
def group_t811():
    g = "P8.12 switching-frequency bounds"
    rng = random.Random(811)
    ok_id = ok_j = ok_a2 = True
    rows = []
    cases = [(k / 100, 2 / 3) for k in range(1, 100)] + [(rng.uniform(0.01, 0.99), rng.uniform(0.05, 0.95))
                                                        for _ in range(40)]
    for lam, c in cases:
        x = rng.random()
        lo, hi = (1 - lam) * c, (1 - lam) * c + lam
        while not lo <= x < hi:
            x = f_map(x, lam, c)
        n = 60000
        x_start = x
        sw = 0
        sx = 0.0
        for _ in range(n):
            sw += x < c
            sx += x
            x = f_map(x, lam, c)
        freq, mean = sw / n, sx / n
        ok_id &= abs(freq - (mean + (x - x_start) / (lam * n))) < 1e-9
        ok_j &= lo - 1e-12 <= mean <= hi + 1e-12
        if c == 2 / 3:
            tol = 1 / (lam * n)
            ok_a2 &= max(0.5, 2 / 3 - 2 * lam / 3) - tol <= freq <= 2 / 3 + lam / 3 + tol
            if lam in (0.01, 0.1, 0.3, 0.5, 0.9):
                rows.append("l=%.2f switch=%.4f in [%.4f, %.4f]" % (lam, freq, max(0.5, 2 / 3 - 2 * lam / 3),
                                                                    2 / 3 + lam / 3))
    check(g, "identity #switches/n = mean x + (x_n - x_0)/(lambda n) (exact up to rounding)", ok_id)
    check(g, "mean of x_k lies in closure(J) = [(1-l)c, (1-l)c + l] (random c and lambda)", ok_j)
    check(g, "case A2: max{1/2, 2/3 - 2l/3} <= 1 - sigma <= 2/3 + l/3 on lambda = 0.01..0.99", ok_a2,
          "; ".join(rows))
    lam, c = 0.001, 2 / 3
    x = 0.5
    n = 1000000
    sw = 0
    for _ in range(n):
        sw += x < c
        x = f_map(x, lam, c)
    check(g, "lambda=0.001: switching frequency within 2l/3 + 1/(l n) of 2/3 (limit 2/3)",
          abs(sw / n - 2 / 3) <= 2 * lam / 3 + 1 / (lam * n), "%.5f" % (sw / n))


# -------------------------------------------------- PLAT: plateau polynomials
def pmul(p, q):
    r = [Fr(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            r[i + j] += a * b
    return r


def padd(p, q):
    n = max(len(p), len(q))
    return [(p[i] if i < len(p) else 0) + (q[i] if i < len(q) else 0) for i in range(n)]


def ptrim(p):
    p = list(p)
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    return p


def pdiv_linear(p, root):
    """Divide by (a - root), assert zero remainder."""
    p = ptrim(p)
    out = [Fr(0)] * (len(p) - 1)
    acc = Fr(0)
    for i in range(len(p) - 1, 0, -1):
        acc = p[i] + acc * root
        out[i - 1] = acc
    assert p[0] + acc * root == 0
    return out


def peval(p, a):
    return sum(float(cf) * a**i for i, cf in enumerate(p))


def reduce_poly(p):
    """Remove the factors a and (1 - a), which are roots of every plateau polynomial."""
    p = ptrim(p)
    while p[0] == 0:
        p = p[1:]
    while sum(p) == 0:
        p = pdiv_linear(p, Fr(1))
    if p[0] < 0:
        p = [-v for v in p]
    return p


def plateau_polys(p, q):
    """3(1-a^q)(delta - lo) >= 0 and 3(1-a^q)(hi - delta) >= 0, delta = 1 - 2a/3."""
    cc = [Fr(1)] + [Fr(((k + 1) * p) // q - (k * p) // q) for k in range(1, q - 1)]
    one_m_a = [Fr(1), Fr(-1)]
    one_m_aq = [Fr(1)] + [Fr(0)] * (q - 1) + [Fr(-1)]
    three_d = [Fr(3), Fr(-2)]
    lower = padd(pmul(three_d, one_m_aq), [-3 * v for v in pmul(one_m_a, cc)])
    cc_hi = padd(cc, [Fr(0)] * (q - 1) + [Fr(1), Fr(-1)])
    upper = padd([3 * v for v in pmul(one_m_a, cc_hi)], [-v for v in pmul(three_d, one_m_aq)])
    return reduce_poly(lower), reduce_poly(upper)


def root01(poly):
    grid = [i / 20000 for i in range(1, 20000)]
    vals = [peval(poly, a) for a in grid]
    idx = [i for i in range(len(grid) - 1) if (vals[i] > 0) != (vals[i + 1] > 0)]
    if len(idx) != 1:
        return None, len(idx)
    lo, hi = grid[idx[0]], grid[idx[0] + 1]
    flo = peval(poly, lo)
    for _ in range(200):
        mid = (lo + hi) / 2
        fm = peval(poly, mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return (lo + hi) / 2, 1


def plateau_lambda(p, q):
    lp, up = plateau_polys(p, q)
    r1, n1 = root01(lp)
    r2, n2 = root01(up)
    if r1 is None or r2 is None:
        return None, (n1, n2)
    lams = sorted((1 - r1, 1 - r2))
    return lams, (n1, n2)


def cycle(lam, p, q, c=2 / 3, steps=4000):
    """Attracting cycle of f_c on the plateau p/q: simulate, read q symbols,
    solve the periodic orbit exactly for that itinerary, verify consistency."""
    x = 0.3
    for _ in range(steps):
        x = f_map(x, lam, c)
    sym = []
    y = x
    for _ in range(q):
        sym.append(y < c)
        y = f_map(y, lam, c)
    a = 1 - lam
    y0 = sum(a ** (q - 1 - j) * lam * s for j, s in enumerate(sym)) / (1 - a**q)
    pts = [y0]
    for j in range(q - 1):
        pts.append(a * pts[-1] + lam * sym[j])
    consistent = all((pt < c) == s for pt, s in zip(pts, sym))
    closes = abs(a * pts[-1] + lam * sym[-1] - y0) < 1e-12
    stays = sum(not s for s in sym)
    minimal = all(sym[r:] + sym[:r] != sym for r in range(1, q))
    above = min((pt - c for pt in pts if pt >= c), default=None)
    below = min((c - pt for pt in pts if pt < c), default=None)
    return dict(ok=consistent and closes and stays == p and minimal and abs(x - min(
        pts, key=lambda v: abs(v - x))) < 1e-9, above=above, below=below)


def group_plat():
    g = "PLAT plateaus, border collisions, grid"
    # The 2/5 polynomials quoted in the paper.
    lp, up = plateau_polys(2, 5)
    paper_lo = [Fr(v) for v in (1, -2, 1, 1, -2)]
    paper_hi = [Fr(v) for v in (1, -2, 1, -2, 1)]
    check(g, "sigma=2/5 endpoints: roots of 1-2a+a^2+a^3-2a^4 and 1-2a+a^2-2a^3+a^4",
          sorted([lp, up]) == sorted([paper_lo, paper_hi]), "polys %s / %s" % (
              [int(v) for v in lp], [int(v) for v in up]))

    table = {(3, 8): (0.2501, 0.3177), (2, 5): (0.3262, 0.4690), (5, 12): (0.4690, 0.4702),
             (3, 7): (0.4702, 0.4936), (4, 9): (0.4937, 0.4985)}
    got = {}
    ok = True
    rows = []
    for (p, q), (lo_t, hi_t) in table.items():
        lams, ns = plateau_lambda(p, q)
        ok &= lams is not None and ns == (1, 1)
        if lams is None:
            rows.append("%d/%d: sign changes %s" % (p, q, ns))
            continue
        got[(p, q)] = lams
        ok &= round(lams[0], 4) == lo_t and round(lams[1], 4) == hi_t
        rows.append("%d/%d: [%.5f, %.5f]" % (p, q, lams[0], lams[1]))
    check(g, "plateau table 3/8, 2/5, 5/12, 3/7, 4/9 (single sign change, 4 decimals)", ok,
          "; ".join(rows))
    check(g, "plateau 5/12 starts at 0.46904, after the end 0.46899 of 2/5",
          (2, 5) in got and (5, 12) in got and round(got[(5, 12)][0], 5) == 0.46904
          and round(got[(2, 5)][1], 5) == 0.46899 and got[(5, 12)][0] > got[(2, 5)][1])

    # Plateaus n/(2n+1), n = 2..6: below 1/2 and increasing; 6/13 = [0.49963, 0.49991].
    seq = []
    ok = True
    for n in range(2, 7):
        lams, ns = plateau_lambda(n, 2 * n + 1)
        ok &= lams is not None and ns == (1, 1)
        if lams:
            seq.append(lams)
    ok &= len(seq) == 5 and all(seq[i][1] < seq[i + 1][0] for i in range(4)) and seq[-1][1] < 0.5
    ok &= round(seq[-1][0], 5) == 0.49963 and round(seq[-1][1], 5) == 0.49991
    check(g, "plateaus n/(2n+1), n=2..6, increase towards 1/2; 6/13 = [0.49963, 0.49991]", ok,
          "; ".join("%d/%d: [%.5f, %.5f]" % (n + 2, 2 * n + 5, l[0], l[1]) for n, l in enumerate(seq)))

    # Polynomial plateaus agree with the Laurent-Nogueira test of (23) inside and just outside.
    ok = True
    for (p, q), (lo, hi) in got.items():
        mid = (lo + hi) / 2
        ok &= on_plateau(mid, p, q) and not on_plateau(lo - 1e-6, p, q) and not on_plateau(hi + 1e-6, p, q)
    check(g, "polynomial endpoints agree with the plateau inequalities (23)", ok)

    # 19 interior points of each plateau: cycle of period q, p stays, not containing c.
    ok = True
    for (p, q), (lo, hi) in got.items():
        for i in range(1, 20):
            cy = cycle(lo + (hi - lo) * i / 20, p, q)
            ok &= cy["ok"] and cy["above"] > 1e-9 and cy["below"] > 1e-9
    check(g, "19 interior points per plateau: cycle of period q with p stays, avoiding c", ok)

    # Border collisions at the endpoints of 2/5 and 3/7.
    ok = True
    rows = []
    for (p, q) in ((2, 5), (3, 7)):
        lo, hi = got[(p, q)]
        ab = [cycle(lo + e, p, q)["above"] for e in (1e-3, 1e-4, 1e-5)]
        be = [cycle(hi - e, p, q)["below"] for e in (1e-3, 1e-4, 1e-5)]
        ok &= ab[0] > ab[1] > ab[2] and ab[2] < 1e-4 and be[0] > be[1] > be[2] and be[2] < 1e-4
        ok &= not on_plateau(lo - 1e-6, p, q) and not on_plateau(hi + 1e-6, p, q)
        rows.append("%d/%d: left %s, right %s" % (p, q, ", ".join("%.1e" % v for v in ab),
                                                 ", ".join("%.1e" % v for v in be)))
    check(g, "endpoints of 2/5 and 3/7: a cycle point tends to c from above (left), from below (right)",
          ok, "distances at eps=1e-3,1e-4,1e-5: " + "; ".join(rows))
    xb = [1 / (2 - (0.5 + e)) for e in (1e-2, 1e-3, 1e-4)]
    check(g, "lambda = 1/2: x_b decreases to 2/3 as lambda decreases to 1/2",
          xb[0] > xb[1] > xb[2] > 2 / 3 and xb[2] - 2 / 3 < 1e-4)

    # Grid lambda = 0.02..0.98: plateau test with q <= 60 at 92 points, agreement with simulation.
    def ln_rho(lam):
        for q in range(2, 61):
            for p in range(1, q):
                if gcd(p, q) == 1:
                    lo, hi = ln_plateau(1 - lam, p, q)
                    if lo - 1e-12 <= 1 - (1 - lam) * 2 / 3 <= hi + 1e-12:
                        return Fr(p, q)
        return None

    grid = [k / 100 for k in range(2, 99)]
    found = agree = avoid = 0
    none_small = True
    for lam in grid:
        r = ln_rho(lam)
        if r is None:
            none_small &= lam <= 0.06
            continue
        found += 1
        x = 0.3
        seq, xs = [], []
        for k in range(20000):
            if k >= 15000:
                seq.append(1 if x >= 2 / 3 else 0)
                xs.append(x)
            x = f_map(x, lam, 2 / 3)
        s = "".join(map(str, seq[-600:]))
        per = next((pp for pp in range(1, 300) if s[pp:] == s[:-pp]), None)
        agree += abs(sum(seq) / len(seq) - float(r)) < 2e-3 and per == r.denominator
        avoid += lam == 0.5 or min(abs(v - 2 / 3) for v in xs[-600:]) > 1e-9
    check(g, "grid 0.02..0.98 (97 points): test finds p/q (q<=60) at 92 points (lambda>=0.07), "
             "simulation agrees at all 92, cycles avoid c except at 1/2",
          found == 92 and agree == 92 and avoid == 92 and none_small and len(grid) == 97,
          "found=%d agree=%d avoid=%d" % (found, agree, avoid))

    # Monotonicity of sigma on lambda = 0.001..0.999 via the lift: |F^n(z)/n - rho| <= 1/n.
    n = 8000
    sig = []
    for k in range(1, 1000):
        lam = k / 1000
        a, d = 1 - lam, 1 - (1 - lam) * 2 / 3
        z = 0.0
        for _ in range(n):
            fz = floor(z)
            z = a * (z - fz) + d + fz
        sig.append(z / n)
    drops = [i for i in range(len(sig) - 1) if sig[i + 1] < sig[i] - 2.0 / n]
    check(g, "sigma(lambda) non-decreasing on lambda = 0.001..0.999 (up to the error 2/n, n=%d)" % n,
          not drops, "violations: %d" % len(drops))


# ------------------------------------- L8.11: genuine cycle inside a plateau
def prem(p, d):
    """Remainder of p divided by d (coefficient lists, lowest degree first)."""
    p = ptrim(p)
    d = ptrim(d)
    while len(p) >= len(d) and any(p):
        coef = p[-1] / d[-1]
        shift = len(p) - len(d)
        p = ptrim([v - (coef * d[i - shift] if 0 <= i - shift < len(d) else 0) for i, v in enumerate(p)])
        if len(p) < len(d) or (len(p) == 1 and p[0] == 0):
            break
        if p[-1] == 0:
            p = ptrim(p)
    return p


def sturm_roots01(poly):
    """Number of distinct real roots in (0, 1] by Sturm's theorem (exact)."""
    seq = [ptrim(poly), ptrim([i * v for i, v in enumerate(poly)][1:])]
    while len(seq[-1]) > 1 or seq[-1][0] != 0:
        r = prem(seq[-2], seq[-1])
        if len(r) == 1 and r[0] == 0:
            break
        seq.append([-v for v in r])

    def changes(x):
        vals = [sum(cf * x**i for i, cf in enumerate(s)) for s in seq]
        vals = [v for v in vals if v != 0]
        return sum(1 for u, v in zip(vals, vals[1:]) if (u > 0) != (v > 0))
    return changes(Fr(0)) - changes(Fr(1))


def lift(y, a, d):
    m = floor(y)
    return a * (y - m) + d + m


def phi(y, a, d, p, q):
    x = y
    for _ in range(q):
        x = lift(x, a, d)
    return x - y - p


def next_break(y, a, d, q):
    """Smallest b > y at which some F^j, j < q, reaches an integer."""
    best = None
    x = y
    for j in range(q):
        b = y + (floor(x) + 1 - x) / a**j
        best = b if best is None or b < best else best
        x = lift(x, a, d)
    return best


def interior_cycle(lam, p, q):
    """Proof of Lemma 8.11 in exact arithmetic at a = 1 - lam, delta = 1 - 2a/3."""
    a = 1 - lam
    d = 1 - Fr(2, 3) * a
    cc = 1 + sum((((k + 1) * p) // q - (k * p) // q) * a**k for k in range(1, q - 1))
    fac = (1 - a) / (1 - a**q)
    if not fac * cc < d < fac * (cc + a ** (q - 1) - a**q):
        return None
    ys = [Fr(k, 400) for k in range(400)]
    vals = [phi(y, a, d, p, q) for y in ys]
    y1 = next(y for y, v in zip(ys, vals) if v > 0)
    y2 = next(y + (1 if y < y1 else 0) for y, v in zip(ys, vals) if v < 0)
    # y* = inf{y in [y1, y2]: Phi(y) <= 0}, walking the affine pieces of Phi (slope a^q - 1).
    y = y1
    ystar = None
    while y < y2:
        v = phi(y, a, d, p, q)
        if v <= 0:
            return None
        b = min(next_break(y, a, d, q), y2)
        z = y + v / (1 - a**q)
        if z < b:
            ystar = z
            break
        y = b
    if ystar is None or phi(ystar, a, d, p, q) != 0:
        return None
    x = ystar
    for _ in range(q):
        if x.denominator == 1:
            return None
        x = lift(x, a, d)
    z = ystar - floor(ystar)
    orbit, wraps, w = [], 0, z
    for _ in range(q):
        orbit.append(w)
        w = a * w + d
        if w >= 1:
            wraps += 1
            w -= 1
    if w != z or wraps != p or len(set(orbit)) != q or 0 in orbit:
        return None
    return min(orbit)


def group_l811():
    g = "L8.11 genuine cycle inside a plateau"
    plateaus = ((3, 8), (2, 5), (5, 12), (3, 7), (4, 9))
    ok = True
    rows = []
    for p, q in plateaus:
        lp, up = plateau_polys(p, q)
        n1, n2 = sturm_roots01(lp), sturm_roots01(up)
        ok &= n1 == 1 and n2 == 1
        rows.append("%d/%d: %d, %d" % (p, q, n1, n2))
    check(g, "Sturm: each endpoint polynomial of the table has exactly one root in (0,1) (exact)", ok,
          "; ".join(rows))

    rng = random.Random(1811)
    ok = True
    for _ in range(300):
        a = Fr(rng.randint(1, 99), 100)
        d2 = Fr(rng.randint(0, 999), 1000)
        d1 = d2 - Fr(rng.randint(1, 200), 1000)
        y = Fr(rng.randint(-2000, 2000), 1000)
        q = rng.randint(1, 9)
        ok &= phi(y, a, d2, 0, q) >= phi(y, a, d1, 0, q) + (d2 - d1)
    check(g, "step (ii): Phi_delta >= Phi_delta' + (delta - delta') (exact, 300 random cases, seed 1811)", ok)

    ok = True
    rows = []
    for p, q in plateaus:
        lams, _ = plateau_lambda(p, q)
        if lams is None:
            ok = False
            continue
        lo, hi = lams
        worst = 1.0
        for i in range(1, 20):
            lam = Fr(round((lo + (hi - lo) * i / 20) * 10**6), 10**6)
            m = interior_cycle(lam, p, q)
            ok &= m is not None
            if m is not None:
                worst = min(worst, float(m))
        rows.append("%d/%d: min z %.1e" % (p, q, worst))
    check(g, "19 rational interior lambda per plateau: Phi(y*) = 0 exactly at the infimum, "
             "g^q(z*) = z*, p wraps, minimal period q, orbit avoids 0", ok, "; ".join(rows))


# ----------------------------------------------------------- T9.1: n doors
def group_t91():
    g = "T9.1 n doors, neutral host"
    ok = True
    for n in range(3, 8):
        for k in range(1, n - 1):
            x = 0
            joint = {}
            for c in range(n):
                goats = [d for d in range(n) if d != x and d != c]
                subsets = list(combinations(goats, k))
                for s in subsets:
                    joint.setdefault(s, {})
                    joint[s][c] = joint[s].get(c, 0) + Fr(1, n) / len(subsets)
            for s, post in joint.items():
                tot = sum(post.values())
                ok &= post.get(x, 0) / tot == Fr(1, n)
                for j in range(n):
                    if j != x and j not in s:
                        ok &= post.get(j, 0) / tot == Fr(n - 1, n * (n - k - 1))
                ok &= all(post.get(j, 0) == 0 for j in s)
            ok &= Fr(comb(n - 2, k), comb(n - 1, k)) == Fr(n - 1 - k, n - 1)
    check(g, "exact enumeration n=3..7, all k: P(C=x|S)=1/n, remaining door (n-1)/(n(n-k-1))", ok)
    formula = lambda n, k: Fr(n - 1, n * (n - k - 1))
    tab = {(3, 1): Fr(2, 3), (4, 1): Fr(3, 8), (4, 2): Fr(3, 4), (1000, 998): Fr(999, 1000)}
    ok = all(formula(n, k) == v for (n, k), v in tab.items())
    ok &= abs(float(formula(1000, 1)) * 999 - 1) < 1e-3
    check(g, "Table 1: 2/3, 3/8, 3/4, ~1/999, 999/1000", ok,
          "P(1000,1) = %s = %.7f" % (formula(1000, 1), float(formula(1000, 1))))
    ok = all(max(range(1, n - 1), key=lambda k: formula(n, k)) == n - 2
             and formula(n, n - 2) == Fr(n - 1, n) for n in range(3, 60))
    check(g, "Cor 9.2: the per-door maximum is at k = n-2, value (n-1)/n", ok)


def main():
    for fn in (group_t32, group_t7, group_t88, group_t810, group_t811, group_plat, group_l811, group_t91):
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
