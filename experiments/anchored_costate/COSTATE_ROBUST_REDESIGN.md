# Costate-First Robust Architecture Redesign

This note is a redesign proposal for the next solver generation.  It takes the
user's two priorities as primitive:

1. The inner robustness/minimizer should itself be represented by a neural model,
   conditioned on `xi`.
2. The outer value problem should be trained primarily through costates, with
   the value level recovered by an EGM-style backward/integration step.

The objective is not just lower residuals.  The architecture must preserve the
robust-control economics: the learned inner minimizer must be checked against
the Hansen-Sargent closed form or a duality gap.

## Current Implementation Status

Implemented in this sandbox:

- `costate_robust_root.py`: terminal/root costate field with line-integral value
  recovery and an `xi`-conditioned Brownian drift `RobustMinimizerNet`.
- `train_two_regime_learned_jump.py`: anchored two-regime trainer with an
  `xi`-conditioned damage-jump `JumpRobustNet` for `log_g_l`.

The preferred minimizer form is:

```text
h_hat     = h_closed + scale_h * tanh(R_h(...))
log_g_hat = log_g_closed + scale_g * tanh(R_g(...)).
```

This is the GAN/minimax idea, but with the closed-form robust-control solution
used as the baseline and as a duality check.  Pure direct learning is useful only
as a negative control: in the jump problem it can create explosive `g` and a huge
pre-regime HJB before it learns the correct scale.

---

## 1. Main Representation

For each regime `r`, do not make the scalar value `v_r(x)` the primitive output.
Instead train a costate field

```text
C_r(x, zeta_r) -> p_r(x) = (p_logK, p_Z, p_Y[, p_logR])
```

where `x` are true states and `zeta_r` collects pseudo/regime parameters:

```text
zeta_r = (logxi, lambda3 if post-damage, tech-state/productivity, regime id)
```

Then recover the transformed value by a line integral:

```text
v_r(x, zeta_r)
  = a_r(zeta_r) + ∫_{anchor_r(zeta_r)}^x p_r(s, zeta_r) · ds.
```

This is the EGM/backward idea: solve the marginal object first, then recover the
level from one anchor.  The level is no longer weakly identified through
`-delta*v`; it is pinned by `a_r`.

### Recommended costate parameterization

Use a direct costate network with an integrability penalty:

```text
p_r = C_r(x, zeta_r)
L_curl = ||Jac_x C_r - Jac_x C_r' ||_antisym
```

Why not force `p = grad Phi` by construction?  That is conservative, but training
the differentiated HJB or second-order HJB terms then tends to require higher
autodiff order and is slower.  Direct `p` plus curl penalty is cheaper and worked
in the root benchmark.

Use:

- normalized true states;
- `logxi`, `exp(-logxi)`, and possibly clipped `1/xi` as separate conditioning
  features;
- regime embedding or separate heads;
- smooth activations: `tanh` or `silu`, no hard ReLU kinks;
- output scaling/prior:
  - `p_logK` initialized near `1`;
  - `p_Z`, `p_Y`, `p_logR` initialized near small values;
  - optional residual form `p = p_prior + p_residual`.

---

## 2. Value Recovery / Anchors

Each regime has one scalar anchor function:

```text
a_r(zeta_r) = v_r(anchor_r(zeta_r), zeta_r).
```

The anchor should come from the jump tree:

- terminal regime: FD / trusted surrogate / calibrated scalar anchor;
- pre-damage post-tech: anchor at damage boundary to post-damage value average;
- pre-tech regimes: anchor to already solved post-tech neighbor at technology
  jump boundary;
- full joint solve: anchors are live, but regularized by boundary matching.

For two adjacent regimes `r -> q`, enforce:

```text
v_r(boundary) ≈ E_q[v_q(boundary, realized state)]
```

This is the cross-regime level communication that fixed the previous weak-id
problem.

---

## 3. Inner Robustness Network

The robust minimizer is not just a nuisance formula.  Treat it as an amortized
inner model:

```text
R_omega(x, p_r, jump_gaps, logxi, regime_id)
  -> h_hat, log_g_hat
```

where

```text
h_hat = (h_d, h_g, h_y[, h_r])
log_g_hat = log distortions for active jump channels.
```

### Parameterization

Do not let the robustness net learn from scratch.  Use the closed-form minimizer
as the base and let the network learn a residual:

```text
h_hat      = h_closed(x, p, xi) + scale_h * Δh_omega(...)
log_g_hat  = log_g_closed(Delta v, xi) + scale_g * Δlog_g_omega(...)
g_hat      = exp(clamp(log_g_hat, -Gmax, Gmax)).
```

Closed forms:

```text
h_closed = -(1/xi) sigma' V_x
log_g_closed = -(V_post - V_pre) / xi.
```

This gives the user-desired neural inner robustness, but keeps the economics
honest.  If the NN residual is useful, it will reduce inner/outer loss.  If not,
regularization pushes it back to the exact minimizer.

### Xi conditioning

The robustness net should be explicitly conditioned on risk aversion:

```text
features_xi = [logxi, exp(-logxi), clipped_exp(-logxi)]
```

and ideally use FiLM modulation:

```text
hidden_l = gamma_l(logxi) * hidden_l + beta_l(logxi).
```

Reason: the minimizer changes mainly through `1/xi`; making the network infer
that scaling from raw `logxi` alone is inefficient.

### Inner loss

For fixed `p`, `v`, controls, train `R_omega` by minimizing the robust inner
Hamiltonian:

```text
L_inner =
    H_drift(p, h_hat)
  + xi/2 * ||h_hat||^2
  + sum_j J_j * [ g_hat_j * DeltaV_j
      + xi * (1 - g_hat_j + g_hat_j log g_hat_j) ].
```

Add a duality/closed-form consistency diagnostic:

```text
L_duality =
    ||h_hat - h_closed||^2
  + ||log_g_hat - log_g_closed||^2.
```

During early training, use `L_duality` as a strong regularizer.  Later, relax it
only if the learned minimizer demonstrably lowers the true inner objective.

This is academically important: otherwise a learned `h/g` can artificially lower
the PDE residual by violating the robust preference problem.

---

## 4. Control Network

Controls should be costate-aware:

```text
A_phi(x, p_r, logxi, regime_id) -> (i_d, i_g[, i_r])
```

This is better than state-only controls because the FOCs depend directly on
costates:

```text
delta / c = phi_d'(i_d) * (p_logK - Z p_Z)
delta / c = phi_g'(i_g) * (p_logK + (1-Z) p_Z)
delta / c = psi_r'(i_r) * p_logR
```

Use bounded activations for feasibility and a low-investment initialization.
Keep FOC loss as a hard diagnostic; do not trade it away for a lower HJB residual.

---

## 5. Training Losses

For each regime:

```text
L_HJB      = RMS(HJB_r[v_from_p, p, Jac p, controls, robustness])
L_FOC      = RMS(FOC_d) + RMS(FOC_g) [+ RMS(FOC_r)]
L_curl     = RMS(antisymmetric Jacobian of p)
L_anchor   = boundary / jump-tree anchor matching
L_inner    = robust minimizer inner objective
L_duality  = distance to closed-form robust minimizer
L_feasible = consumption/control feasibility penalties
```

Recommended composite:

```text
L_costate =
    w_hjb * L_HJB
  + w_curl * L_curl
  + w_anchor * L_anchor
  + w_inner * L_inner
  + w_duality * L_duality

L_control =
    w_foc * L_FOC + w_feasible * L_feasible
```

The architecture should use alternating updates:

1. update robustness net `R_omega` on `L_inner + L_duality`;
2. update controls on `L_control`;
3. update costates on `L_costate`;
4. update anchors/boundary coupling across regimes.

---

## 6. Federated / Cross-Regime Structure

Use multiple regime networks, but let them communicate through live value gaps:

```text
DeltaV_{r->q}(x) = v_q(jump_state(x)) - v_r(x)
```

This gap is fed to:

- the jump term in the outer HJB;
- the robustness net for `log_g`;
- the boundary anchor loss.

For stability:

- start with teacher-style detached post-regime values;
- then allow gradients through the gap (`federated_grad`);
- finally use a cyclic schedule: root-balanced phase, pre-focused phase, polish.

The overnight runs showed that heavy pre weighting can push pre HJB to
`~1.2e-4`, but root HJB loosens.  A cyclic schedule is the likely next step:

```text
Phase A: balanced root/pre, keep both HJBs around 1e-3
Phase B: pre-focused, push pre HJB down
Phase C: root repair, freeze or reduce pre learning rate
Phase D: low-LR joint polish
```

---

## 7. Proposed Network Modules

### CostateFieldNet

```text
input:  normalized state x, regime features, xi features
output: p = (p_logK, p_Z, p_Y[, p_logR])
trunk:  shared FiLM-conditioned MLP or DGM block
heads:  regime-specific linear heads
```

### RobustMinimizerNet

```text
input:  x, p, DeltaV list, intensity list, xi features, regime id
output: Δh, Δlog_g
base:   closed-form h/log_g
final:  h_hat, g_hat
```

### CostateAwareControlNet

```text
input:  x, p, xi features, regime id
output: feasible controls i_d, i_g[, i_r]
loss:   FOC + feasibility
```

### AnchorNet / AnchorTable

```text
input:  pseudo-state zeta_r
output: scalar anchor a_r(zeta_r)
source: FD/surrogate/jump-boundary/live neighbor
```

For early experiments, use deterministic anchor functions rather than a learned
anchor net.  Only learn anchors after boundary matching is stable.

---

## 8. Staged Experiment Plan

### Stage 0: terminal root costate + closed-form robustness

Replicate root costate benchmark with fixed validation and larger batch.

Success:

- HJB `<= 1e-3`;
- FOC `<= 1e-3` eventually;
- curl small;
- level spread small across seeds.

### Stage 1: terminal root costate + robustness net

Add `RobustMinimizerNet`, but keep closed-form regularization strong.

Success:

- inner duality gap near zero;
- no degradation versus closed-form robustness.

### Stage 2: two-regime costate federated solve

Use `PostDamagePostTech -> PreDamagePostTech`.

Success:

- boundary gap `~1e-8`;
- both root/pre HJB decrease, not just one;
- learned `log_g` matches closed-form unless it improves true inner objective.

### Stage 3: cyclic schedule

Use the best overnight lesson:

- balanced phase for root/pre;
- pre-focused phase;
- root-repair phase;
- low-LR joint polish.

Success:

- root and pre both below `1e-4`;
- then attempt `1e-5`.

### Stage 4: extend to 4-state regimes

Add `p_logR`, R&D FOC, and tech jumps.

Do not jump to full six-regime before two-regime costate is stable.

---

## 9. Key Recommendation

The next serious architecture should be:

```text
CostateFieldNet
  + line-integral EGM value recovery
  + hard jump-tree anchors
  + RobustMinimizerNet conditioned on xi
  + costate-aware controls
  + federated cross-regime gap communication
  + cyclic balanced/pre-focused/root-repair training
```

This keeps the user's costate intuition as the main engine, while still letting
inner robustness be learned as a `xi`-parameterized neural response.

The important academic guardrail is that learned robustness must be judged by
its inner minimization objective / duality gap, not merely by a lower outer HJB
residual.
