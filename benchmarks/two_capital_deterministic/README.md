# Deterministic Two-Capital Benchmark: Finite Differences vs Neural Network

A self-contained validation benchmark that solves the **deterministic two-capital
adjustment-cost model** (no random capital shocks; `adjustment_cost_no_random.tex`)
two independent ways and compares the optimal investment-rate ratio `i^g / i^d`
across the green-capital share `Z`:

1. **Finite differences** — upwind, semi-implicit false-transient iteration for the
   1-D HJB in `Z` (the exact reference solution).
2. **Neural network** — Deep-Galerkin (DGM-PIA style) network approximating `v(Z)`,
   trained on the HJB residual, mirroring the parent project's method.

The point is a clean, tractable cross-check of the NN method against an exact
solver on a sub-model of the full climate economy.

## Model

State reduction `V(logK, Z) = logK + v(Z)` leaves a 1-D problem in
`Z = K^g/(K^d+K^g)`. Capital evolves deterministically with log adjustment costs
`phi_j(i) = alpha_j + Gamma_j log(1 + theta_j i)`; utility is `delta log c`. The
reduced HJB is the first-order nonlinear ODE

```
0 = max_{i^d,i^g} { delta*log c - delta*v(Z)
                    + (1-Z) phi_d(i^d) + Z phi_g(i^g)
                    + Z(1-Z)[phi_g(i^g) - phi_d(i^d)] v'(Z) }
```

Under the common adjustment technology used here (`Gamma_d=Gamma_g`, `theta_d=theta_g`)
consumption and the controls are closed form given the slope `p = v'(Z)`:

```
q_d = 1 - Z p,   q_g = 1 + (1-Z) p
c(Z) = delta(1 + theta*Abar(Z)) / (theta(delta+Gamma))         # independent of p
i^d  = Gamma c q_d/delta - 1/theta,   i^g = Gamma c q_g/delta - 1/theta
```

so only `v'(Z)` is unknown — exactly what each method solves for.

## Parameters

Loaded from the parent project's `models/params.py` (so this stays in lock-step
with the rest of the codebase). Per the task, **green productivity uses the POST
tech-jump value `A_g = A_g'' = A_g_prime_prime = 0.1567`** (params.py), with
`A_d = 0.1303`, `delta = 0.01`, `alpha_d=alpha_g=-0.035`, `Gamma_d=Gamma_g=0.060`,
`theta_d=theta_g=16.7`. (Pass `--a-g A_g_prime` or `--a-g A_g` to use the
intermediate or pre-jump productivity instead.)

## Files

| File | Role |
|---|---|
| `two_capital_model.py` | Calibration loader + closed-form algebra (`c`, controls, `phi`, residual, boundaries, perturbation). |
| `fd_solver.py` | Upwind semi-implicit false-transient FD solver, `solve_fd(p)`. |
| `nn_solver.py` | DGM neural-network solver, `solve_nn(p)` (TensorFlow). |
| `compare.py` | Driver: runs both, plots `i^g/i^d` vs `Z`, writes `outputs/`. |
| `run.sbatch` | Optional Slurm wrapper for the NN training run. |

## Run

```bash
module load cuda/11.2 python/anaconda-2021.05
export PYTHONNOUSERSITE=1

# Finite differences only (instant):
python fd_solver.py

# Full comparison + figure (FD + NN):
python compare.py --a-g A_g_prime_prime --nn-iters 40000
# FD-only figure: add --no-nn
```

Outputs land in `outputs/`:
- `ig_id_ratio_<A_g_choice>.png` — 3 panels: `i^g/i^d` vs `Z` (FD vs NN), the
  investment levels `i^d,i^g`, and the value slope `v'(Z)` (with the perturbation
  overlay).
- `compare_<A_g_choice>.csv` — `Z, v', i_d, i_g, ratio` for both methods.

## Validation notes

- **Symmetric benchmark** (`A_g=A_d`): the solver returns `v'(Z)=0` to ~1e-12 and
  `i^g/i^d = 1` to machine precision — exact check of the machinery.
- **Interior HJB residual** of the FD solution is `~1e-5` over `Z∈[0.1,0.9]`
  (the larger value at the very ends is the degenerate one-sided-slope boundary,
  where `mu_Z→0`).
- The first-order **perturbation** slope `v'(Z) ≈ (A_g-A_d)/c(Z)` is only accurate
  for small heterogeneity; with the ~19% post-jump productivity gap it is a rough
  guide, not the exact solution — the FD/NN agreement is the real check.
