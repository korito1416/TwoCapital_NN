# Task Log — Deterministic Two-Capital: FD vs NN benchmark (2026-06-24)

**Assigned task.** Solve the deterministic two-capital adjustment-cost model
(`adjustment_cost_no_random.tex`) two ways — finite differences and the project's
neural-network (DGM) method — and compare, plotting `i^g/i^d` vs `Z`. Parameters
from the parent `models/params.py`; **green productivity uses `A_g = A_g''` =
`A_g_prime_prime` = 0.1567** (post tech jump). `A_d = 0.1303`, `delta=0.01`,
`alpha_d=alpha_g=-0.035`, `Gamma_d=Gamma_g=0.060`, `theta_d=theta_g=16.7`.
Later asks: add **aggregate productivity Ā(Z)** and **consumption/output ratio
C/Y** to the figure; FD loss should reach ~1e-6; **the NN must use the project's
3-loss structure** (value + two FOC), trained via sbatch (login node is weak).

## Files
- `two_capital_model.py` — calibration loader (from models/params.py) + closed-form
  algebra (c, controls, phi, HJB residual, one-capital boundaries, perturbation).
  **Verified CORRECT** by a subagent (all functions match the tex).
- `fd_solver.py` — upwind semi-implicit false-transient FD. Converges (interior
  residual ~6e-5) and passes the symmetric benchmark exactly (A_g=A_d ⇒ v'=0,
  ratio=1, residual ~1e-17). **BUT carries O(dZ) numerical diffusion → its slope
  v'(Z) is biased** (steeper than the true solution near Z→0).
- `nn_solver.py` — DGM net for v(Z) with **closed-form controls** and **only the
  HJB residual loss**. Verified bug-free by a subagent, but plateaus at
  pde_rmse ~5e-3 (Adam-PINN plateau) with a too-flat v' → `i^g/i^d` disagrees 2–5×
  with FD. **This is the simplified version that must be replaced (see Open #2).**
- `reference_solver.py` — high-accuracy references: central-difference Newton via
  `scipy.optimize.root` (added, not yet run to completion) + a `solve_bvp` attempt.
- `compare.py` — driver; writes `outputs/ig_id_ratio_*.png` and `outputs/compare_*.csv`.
- `run.sbatch` — Slurm wrapper (logs to `logging/`).

## Status / what we learned
- **Params confirmed** to come from parent `models/params.py` (A_d=0.1303,
  A_g=A_g_prime_prime=0.1567).
- **The problem is weakly conditioned in v':** with delta=0.01 and a small Z-drift
  mu_Z, the HJB residual is nearly insensitive to the slope v'(Z). Multiple
  slopes give a "low-looking" residual, so residual minimization alone does NOT
  pin v' well. This explains (a) the NN plateau, and (b) why my upwind FD and a
  central solve disagree on v'(Z) while both report small residuals.
- **Ground truth not yet nailed.** Upwind FD: v'(0.1)≈1.18→v'(0.9)≈0.39. User's
  central-Newton: v'(0.1)≈1.0 (res ~2e-5). My damped central iteration DIVERGED
  (ill-conditioned without upwinding); solve_bvp failed. A proper central-difference
  Newton (scipy.optimize.root, in reference_solver.py) is the right tool — needs to
  be run to settle the true v'(Z) (target residual ~1e-6).
- **i^g/i^d (current, from upwind FD):** ≈ 3.3 at Z=0.1 down to ≈ 2.2 at Z=0.9
  (green invests more, since A_g>A_d post-jump). NN currently off by 2–5×.

## Open items / next steps
1. **Settle the FD ground truth** with the central-difference Newton
   (`reference_solver.solve_newton`, scipy root) to ~1e-6, and adopt it as the
   reference v'(Z) (replace/augment the diffusion-biased upwind).
2. **Rebuild the NN as faithful DGM-PIA with the project's 3 losses** (see
   `models/PostDamagePostTech.py:objective_fn`):
   - separate `v`, `i_d`, `i_g` networks (NOT closed-form controls);
   - value loss = √mean((rhs−pv)²) + √mean(FOC_d²) + √mean(FOC_g²) [+ sign penalty];
   - control loss = −mean(rhs−pv) + √mean(FOC_d²) + √mean(FOC_g²);
   - two-step train_step (value, then controls). The FOC residual directly pins
     the controls given the value derivatives — this is the remedy for the weak
     identification that the HJB-only NN suffered.
   - **Run training via sbatch** (`run.sbatch`), not the login node.
3. **Figure additions:** add panels/columns for Ā(Z)=(1−Z)A_d+Z·A_g and
   C/Y = c/Ā(Z), alongside i^g/i^d, levels, and v'.
4. After both methods agree on i^g/i^d: **convexity analysis of the FOC loss
   function** (the user's final ask) — analytically and numerically characterize
   the shape/convexity of the control objective.

## How the "FD loss" is computed
The FD has no training loss; its accuracy is the **HJB residual**
`R(Z) = delta·log c − delta·v + (1−Z)phi_d + Z·phi_g + mu_Z·v'` evaluated at the
converged grid solution (with an accurate central derivative for v'), reported in
L2 / max norm over the interior grid — the **same residual object the NN minimizes**,
so the two methods are compared on an apples-to-apples accuracy scale.
