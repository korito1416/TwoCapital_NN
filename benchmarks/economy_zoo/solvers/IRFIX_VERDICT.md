# JONES i_r-damping decisive test — verdict (2026-07-20)

**Job:** `zoo_jones_irfix2` (52406971). **Cell:** JONES PostDamagePreTech, l3idx=2 (λ3=0.167),
**ξ=148.4 (NEUTRAL)** — so this is not a ξ-mechanism test; it isolates the Howard i_r update.
Coarse grid (31,21,25), T=1200, dt=2.5, howard_max=300, **relax i_r=0.02** (heavy damping) vs
i_d=i_g=0.3, exp_clip=35, step_exact, no warm start. 5187 s (~1.5 h).
Data: `solvers/outputs/jones_irfix/jones_PostDamagePreTech_l32_xi148p4.npz` (+ PROVENANCE.json).

## Verdict: i_r-damping BREAKS the limit cycle — diagnosis confirmed
The interior policy update converges **monotonically**, no oscillation:

| Howard iter | 0 | 10 | 30 | 100 | 175 | 295 |
|---|---|---|---|---|---|---|
| di_int | 0.423 | 0.0276 | 0.00766 | 0.00198 | 0.00102 | 0.000655 |
| i_r | 0.0032 | 0.0106 | 0.0139 | 0.0172 | 0.0182 | 0.0187 |

`di_int_final = 6.47e-4`, `mutual_consistency_gate_int = 6.50e-4`. Monotonicity clean
(`frac_W_increasing_in_Y = 0`, `frac_W_decreasing_in_s = 2.6e-4`, `min_dW_ds = -7.7e-5`).

→ **The JONES tech-channel non-convergence was the Howard i_r↔W_S update cycling**
(contraction factor J/(δ+J) ≈ 0.9989), NOT the ξ-robustness mechanism and NOT an unsolvable economy.
Heavy i_r relaxation (0.02) tames it: the interior settles smoothly.

## Caveat: solvable but budget- and boundary-limited
- **Budget-limited.** It hit `howard_max=300` before `tol_pocket=1e-6`; di_int was still falling
  (~1e-5 per 5 iters) and **i_r had not fully settled** (still creeping 0.01865→0.01868 at iter 295).
  A converged JONES cell needs a larger Howard budget than the other zoo economies.
- **Boundary-limited residual.** The full-grid metric `di_full` spikes late (3.8e-2 at iter 180,
  2.7e-1 at 210, 6.8e-1 at 290) — the **s-top boundary layer**, not the interior. `true_residual_max
  = 0.0222`, `true_residual_rms = 0.0061`. The interior is fine; the top-s slice is the limiter.

## Implication for the JONES/PHYSRISK go/no-go
JONES is **not a dead end** — mechanism healthy, converges under damping — so it *can* be a zoo member
via the map-level ladder (option A). But it is the **slowest/most expensive** zoo economy and its
residual is boundary-limited; a design-grid (61,41,61) solve needs heavy i_r damping **and** a raised
Howard budget **and** s-top boundary handling. Decision (keep via ladder vs cut) is the user's.
