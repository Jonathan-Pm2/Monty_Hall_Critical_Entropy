# Monty Hall: critical entropy and a bifurcation of decisions across episodes

Supplementary code for the manuscript

> G. Pazmiño Moreira and J. J. Proaño Morales, *Beliefs About the Host in the
> Monty Hall Problem: A Critical Entropy Level and a Bifurcation of Decisions
> Across Episodes* (under review).

The scripts reproduce or certify every numerical value in the paper. They use
only the Python 3 standard library (no NumPy or SciPy).

## Contents

All scripts are in [`scripts/`](scripts/); see [`scripts/README.md`](scripts/README.md)
for what each one checks and which result of the paper it refers to.

| Script | Purpose |
|---|---|
| `section7_check.py` | Maximum-entropy prior on the indifference set, critical entropy level, Dirichlet values |
| `bifurcation_check.py` | Adaptive host across episodes: simulation against the plateau formula |
| `probes_check.py` | Assertion checks of the game, entropy and bifurcation results |
| `t10_checks.py` | Assertion checks of the remaining numbers (plateau endpoints, frequency bounds, n doors, ...) |
| `t13_checks.py` | Assertion checks of the devil's staircase in the memory rate |
| `staircase_check.py` | Exact checks of the plateau polynomials (Sturm sequences), ordering and border collisions |
| `t13_probes.py` | Exact Sturm counts for all plateaus with q <= 40 and border-collision probes |
| `dirichlet_certify.py` | Computer-assisted proof (interval arithmetic) for the Dirichlet entropy maximiser |

## How to run

```bash
cd scripts
for s in *.py; do python3 -B "$s" || echo "FAILED: $s"; done
```

Every script exits with code 0 when all its checks pass and with code 1
otherwise. The full run takes a few minutes; `bifurcation_check.py` is the
slowest (about one minute).
