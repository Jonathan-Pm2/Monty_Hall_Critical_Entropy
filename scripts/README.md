# Numerical checks for the revised manuscript (GSTA-2026-0432, R1)

These scripts reproduce every number in Section 7 and in the subsection
on the adaptive host across episodes (Section 8) of
`monty_hall_paper_en_R1.tex`, check the main claims of Sections 5 and 6,
and reproduce every number added in the second round of revisions
(Theorem 3.2, the Dirichlet entropy derivation, Subsection 8.1 and Section 9). The third round adds a
computer-assisted proof for Corollary 7.7 (`dirichlet_certify.py`) and checks for
Lemma 8.13, Theorem 8.14 and Proposition 8.15 (`t13_checks.py`). The fourth round adds checks for
Theorem 8.14(2) and Theorem 8.16 (border collisions at every plateau endpoint):
`staircase_check.py` and `t13_probes.py`. They need only Python 3 and its standard
library (no NumPy or SciPy).

| Script | What it checks | Paper reference |
|---|---|---|
| `section7_check.py` | The maximum-entropy prior on the indifference set `2*mu2 + mu3 = mu1` (`t`, `mu*`, `H*` in nats and bits, ratio to `log 3`). The Dirichlet first-order condition along the symmetric family `(3c, c, c)` (root `c*`, entropy at `c*`, and a numeric derivative that confirms the analytic condition). The maximum over the whole indifference cone. | Theorem "Critical entropy level", Example "Numerical values", Theorem "Dirichlet priors", Corollary "Symmetric subfamily" |
| `bifurcation_check.py` | Simulates the audience's belief under the adaptive host for both tie-breaks. With A2 (threshold 2/3), it compares the simulated stay frequency and period with the rotation number from Laurent and Nogueira (2018, Theorem 3(ii)), finds where the rho = 1/2 plateau begins (lambda = 1/2), and lists the plateaus in lambda for q <= 7. With A3 (threshold 1/2), it checks that the dynamics has period 2 on the whole grid. | Subsection "An adaptive host across episodes" |
| `probes_check.py` | Assertion-based regression check of the main claims, in three groups with fixed seeds: the game of Section 5 (payoff matrix, offer rates, unique maximin, exact equilibrium set, Proposition 5.6, Example 6.5); Section 7 (maximum-entropy prior, H*, Dirichlet critical point, cone maximum, Monte Carlo check of the closed-form entropy); Subsection 8.1 (conjugacy, plateau formula against simulated rotation numbers, case A3 two-cycle, the rho = 1/2 plateau starting at lambda = 1/2). Exits with code 1 if any assertion fails. | Section 5, Example 6.5, Section 7, Subsection 8.1 |
| `t10_checks.py` | Assertion-based check of the numbers added in the revision, fixed seeds, exit code 1 on failure. Theorem 3.2 and Corollary 3.3: win probability 2/3 for every tie-break r = pi(h\|x,x), conditional probability q = 1/(1+r), coverage of the wrong-model interval tending to 0, cell frequencies recovering q, cell probability at least 1/9. Theorem 7.4 endpoint entropies (0.693, 0.637). Theorem 7.6(3): E[log mu_i] = psi(beta_i) - psi(beta_0), derived formula equal to the displayed one, Monte Carlo agreement, Dir(1,1,1) with h = -log 2. Lemma 8.8: discounted Dirichlet counts reproduce the memory rule (19), exactly. Theorem 8.10: lambda = 1/2 in exact arithmetic, sigma <= (M+1)/(2M+3) < 1/2 below 1/2 (window count), case A3 distance lambda/(2(2-lambda)). Proposition 8.12: switching-frequency identity and bounds on a lambda grid, limit 2/3. Plateaus: endpoints as roots of explicit polynomials (3/8, 2/5, 5/12, 3/7, 4/9 and n/(2n+1) up to 6/13), cycles at 19 interior points of each plateau, border collisions at the endpoints of 2/5 and 3/7, the 92-of-97 grid comparison, monotonicity of sigma on lambda = 0.001..0.999 up to the rotation-number error. Lemma 8.11 (a genuine cycle inside a plateau): each endpoint polynomial of the plateau table has exactly one root in (0,1) by Sturm's theorem in exact arithmetic, the inequality of step (ii) of the proof, and the zero of Phi built as in step (iv) of the proof, in exact rational arithmetic at 19 interior lambda of each plateau, giving a cycle of g of period q with p wraps that avoids 0. Theorem 9.1, Table 1 and Corollary 9.2 by exact enumeration. | Theorem 3.2, Corollary 3.3, Theorems 7.4 and 7.6, Lemma 8.8, Theorem 8.10, Lemma 8.11, Proposition 8.12, plateau table of Subsection 8.1, Section 9 |
| `dirichlet_certify.py` | Computer-assisted proof of Corollary 7.7 in interval arithmetic (40-digit decimals, outward rounding). Part 0: tail bounds for the trigamma and tetragamma functions by telescoping and exact polynomial sign checks. Part 1: the symmetric family `(3c, c, c)` has exactly one critical point, `c*` in [0.881441879, 0.881442261], with `h` in [-1.3080834208, -1.3080834198]. Part 2: branch and bound on the cone with the Gaussian maximum-entropy bound for `beta0 >= 12`; every maximiser lies in the dyadic rectangle [825/1024, 207/256] x [3987/4096, 3999/4096] in `(beta2, beta3)`, where `h` is strictly concave (largest Hessian eigenvalue <= -0.78), so the maximiser is unique: `beta# = (2.5896675021, 0.8072419342, 0.9751836336)` +- 5.2e-10, max `h` in [-1.29159391972, -1.29159391871]; non-concavity on the cone, `d^2F/dv^2 > 0` at `(u, v) = (1/20, 7/10)`. Prints `ALL CLAIMS CERTIFIED` and exits 0, or prints `CERTIFICATION FAILED` and exits 1. | Corollary 7.7 |
| `t13_checks.py` | Assertion-based check, exit code 1 on failure. Lemma 8.13: the plateau inequalities (23) with `delta = 1 - a c` are equivalent to bounds on the threshold `c` (exact, random rationals). Theorem 8.14: `Psi_lambda` increasing in `lambda` (Step 3), `sigma` non-decreasing on a 400-point grid within 2e-4, `1/3 < sigma <= 1/2`, `sigma = 1/2` for `lambda >= 1/2`, `sigma(0.4) = 2/5` and `sigma(0.48) = 3/7` (exact plateau test and simulation). Proposition 8.15: the endpoint polynomials equal their definitions, have one root each in (0,1) by Sturm's theorem, and change sign exactly inside each stated enclosure; closed forms of the right endpoints of 2/5 and 3/7. Corollary 7.7: non-concavity, using the interval arithmetic of `dirichlet_certify.py`. Imports `t10_checks.py` and `dirichlet_certify.py` from the same folder. | Lemma 8.13, Theorem 8.14, Proposition 8.15, Corollary 7.7 |
| `staircase_check.py` | Assertion-based check, deterministic, exit code 1 on failure; prints PASS/FAIL per check and a summary per group. CF: exact closed forms of the plateau bounds of (23) and their Sturmian series forms. STURM: every endpoint polynomial `P^-_{p/q}`, `P^+_{p/q}` with 1/3 < p/q <= 1/2 and q <= 40 has exactly one root in (0,1) (`P^+_{1/2}` none), and it is simple (exact rationals). ORDER: plateaus nondegenerate and ordered like p/q (q <= 25), the plateau 2/5 and the closed form of its right endpoint, the plateau table to 4 decimals. SIM: plateau test against simulated stay frequency on lambda = 0.07..0.98, sigma non-decreasing on lambda = 0.001..0.999. PSI: Steps 2-3 of the proof of Theorem 8.14 on grids, also for other thresholds. BC: numerically, for every plateau with q <= 13, the distance from the attracting cycle to c tends to 0 at both endpoints and sigma differs from p/q just outside; the side of approach is printed, not asserted. IRR: the irrational value (3 - sqrt 5)/2. | Lemma 8.13, Theorem 8.14, Proposition 8.15, Theorem 8.16 |
| `t13_probes.py` | Assertion-based check, fixed parameters, exit code 1 on failure. STURM: a separate implementation of the Sturm count in exact rational arithmetic, with a sanity test on a polynomial with known roots, confirms that every endpoint polynomial with p/q in (1/3,1/2) and q <= 40 (160 polynomials) has exactly one root in (0,1) and is squarefree. BC: for the plateaus 3/8, 2/5, 7/17, 5/12, 3/7 and 9/20, endpoints by exact bisection, then 80-digit decimal iteration: near both endpoints the orbit closes into a cycle of period q with p stays, its distance to c decreases as lambda tends to the endpoint, and the stay frequency differs from p/q just outside the plateau. The side from which the cycle approaches c is printed as numerical information only; it is not asserted and is not a proved claim. | Theorem 8.14(2), Theorem 8.16 |

## How to run

Use `python3 -B` to avoid writing `__pycache__` folders (the import in
`t13_checks.py` would otherwise create one).

```bash
python3 section7_check.py
python3 -u bifurcation_check.py   # about one minute
python3 probes_check.py           # about 5 seconds; exit code 0 means all assertions passed
python3 t10_checks.py             # a few seconds; exit code 0 means all assertions passed
python3 dirichlet_certify.py      # about one second; last line ALL CLAIMS CERTIFIED, exit code 0
python3 t13_checks.py             # about one second; exit code 0 means all assertions passed
python3 staircase_check.py        # a few seconds; last line TOTAL: all checks passed, exit code 0
python3 t13_probes.py             # about one second; last line TOTAL: all checks passed, exit code 0
```

Equation numbers in `t10_checks.py` were written for the previous
compilation of the paper ((19) memory rule, (20) episode map,
(22) plateau inequalities). After Corollary 7.7 gained a displayed
equation they are (20) memory rule, (21) episode map and (23) plateau
inequalities in the current `monty_hall_paper_en_R1.pdf`; theorem
numbers in Section 8 up to Proposition 8.12 are unchanged.

`bifurcation_check.py` can report a few grid points where the
simulation and the plateau formula disagree. At those points the
period is longer than the search bound (q <= 60) or the 600-step
window used to detect it, so they are not counterexamples.
