# Two-capital model with shocks AND model uncertainty (robustness)

Hansen–Sargent robust-control extension of the two-capital-with-shocks benchmark. A
robust planner distrusts the dirty/green Brownian capital drifts and plays a max–min
game against drift distortions `(h_d, h_g)` penalised by relative entropy with
multiplier `xi`. Derivation is referee-passed; FD and a DGM neural net (with `log(xi)`
as a pseudo-state) both solve it.

## The result (derivation.tex / .pdf)

Robustness collapses to a single nonpositive **robustness-drag** term added to the
shock HJB:

```
- 1/(2 xi) [ (1-Z)^2 sigma_d^2 q_d^2  +  Z^2 sigma_g^2 q_g^2 ]
```

with worst-case drifts `h_d* = -(1/xi)(1-Z) sigma_d q_d`, `h_g* = -(1/xi) Z sigma_g q_g`,
`q_d = 1 - Z v'`, `q_g = 1 + (1-Z) v'`. Control FOCs are **unchanged** (the drag depends
on `i` only through `q_d, q_g`). Boundaries gain an extra `-sigma_j^2/(2 xi delta)`.
Limits: `xi -> inf` recovers the shock model, `sigma -> 0` the deterministic model. The
effect scales as `sigma^2 / xi`. A 5-referee adversarial panel found **0 errors**
(verified to ~1e-16; matches `models/PostDamagePostTech.py` lines 192-193/240/242).

## Files

| File | What |
|------|------|
| `derivation.tex` / `.pdf` | full derivation + referee report & rebuttals |
| `compare_uncertainty_fd.py` | FD sweep over `xi`, plots i_d/i_g/q_d/v' and worst-case drifts |
| `nn_dgm_uncertainty.py` | DGM-PIA solver, 2-D input `(Z, log xi)`, drag term, xi-dependent boundary |
| `compare_uncertainty_nn.py` | builds the FD reference over a log-xi grid, trains the DGM, plots NN vs FD |
| `run_uncertainty.sbatch` | Slurm driver (env: SIGMA, FD_SUPERVISE, PRECOND, NN_ITERS, NUM_NEURONS, NUM_LAYERS, LABEL) |

The robustness lives in the **shared** shock model (`../two_capital_shock/`,
`P["xi"]`, default `inf` = off), alongside the correlation parameter `P["rho"]`.

## Numerics

`xi in {0.05 (strong), 0.1 (moderate), 148.4 (~ no robustness)}`; `log xi` is sampled
stratified on `[-3, 5]` and fed to the network as a pseudo-state, exactly as in
`models/`. The FD solver solves each `xi` separately as the reference.

### outputs/
- `uncertainty_FD_sigma0.01.png`, `uncertainty_FD_sigma0.2.png` — FD `xi`-sweep
  (controls, marginal values, value slope, worst-case drifts `h_d*, h_g*`).
- `uncertainty_FDvsNN_sigma0.01_*.png` — **DGM(log xi) vs FD at the calibration**:
  matches across all three `xi` to ~1e-4 in controls, ~3e-3 in slope. HJB residual 3.3e-6.
- `uncertainty_FDvsNN_sigma0.2_*.png` — stress test at large `sigma`. The strong-
  robustness region (`xi=0.05`) develops very sharp features (`v'` to ~-2.5) and the FD
  reference itself does not fully converge there (residual ~0.2-0.3); the DGM is
  correspondingly less accurate. The calibration `sigma=0.01` is the reliable case.

## Findings

- At the **calibration `sigma=0.01`** the robustness effect on behaviour is small
  (scales `sigma^2/xi`), but the worst-case drift distortions are non-trivial
  (`|h*|` up to ~0.2 at `xi=0.05`): the planner entertains meaningful misspecification
  even when actions barely move.
- At large `sigma`, strong robustness amplifies precaution: `v'` flips sign and the
  marginal value of dirty capital rises sharply (a worst-case-diversification motive).
