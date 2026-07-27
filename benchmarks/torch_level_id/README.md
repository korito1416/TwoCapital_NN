# torch_level_id — value-LEVEL identifiability study (PyTorch port)

**Date:** 2026-06-30 · **Branch:** TwoTechJump · **Status:** sandbox research (NOT production; loop in Joe & Mike before any production change)

## Why this exists

The production failure behind the low-ξ work: the worst-case **damage-jump** density
`g = exp(-(1/ξ)(V_post - V_pre))` collapses below admissibility for **ξ < ~0.025**. Diagnosis (see
the daily work): this is **weak-identification of the value LEVEL**, not float overflow. The HJB
residual constrains the value's *derivatives* (costates) but pins its absolute *level* only through
the `-δV` term with **δ = 0.01** — so the level is identified only to `~residual/δ ≈ 0.1–1`, and the
factor `1/ξ` amplifies any inter-regime level mismatch into the jump distortion.

These tests run on the **faithful PyTorch port** of the TF DGM-PIA solver (`models_torch/`,
`models_torch_train/`) and quantify the weak-id and which architecture / parameterization changes
**pin the level**.

## Port fidelity (precondition)

`models_torch_train/phase1_validate_surrogate.py` — loads the SAME TF checkpoint into TF and torch,
compares `pde_rhs` on a shared sample:

| quantity | torch vs TF (max abs) |
|---|---|
| HJB residual `rhs-pv` | **2.6e-8** (float32 round-off) |
| controls `1+θ·i_{d,g}` | ~3e-6 |
| value loss | torch **1.71895e-3** = TF **1.71895e-3** |

→ The torch port reproduces the TF solution to float32 round-off. Safe to use as the research platform.

---

## Test 1 — warm-start ablation (`scripts/ablation_root.py`)

From the TF surrogate (warm start), change ONE knob, train 500 steps (root regime
PostDamagePostTech), measure: residual floor, controls (FOC), and **value-LEVEL drift**
`dV = Δ(pv/δ)` on a fixed 256-pt probe.

| config | loss_v 0→1 | FOC_d | FOC_g | **dV_mean** | dV_absmax | sec |
|---|---|---|---|---|---|---|
| A base (f32, Adam) | 1.62e-3 → 7.20e-4 | 1.2e-2 | 1.7e-2 | **+0.747** | 1.49 | 69 |
| B float64 | 1.62e-3 → 7.34e-4 | 1.1e-2 | 1.7e-2 | +0.749 | 1.49 | **1259** |
| C L-BFGS (value) | 1.62e-3 → 4.81e-4 | 1.9e-1 | 1.2e-1 | +0.791 | 1.79 | 14 |
| D value-only | 1.62e-3 → 9.22e-4 | 1.7e-1 | 1.1e-1 | +0.736 | 1.62 | 8 |
| E longer (2000) | 1.62e-3 → **2.90e-4** | 8.9e-3 | 9.3e-3 | +0.823 | 1.74 | 62 |
| F concentrate-Y | 1.62e-3 → 7.75e-4 | 9.0e-3 | 2.0e-2 | +0.810 | 1.60 | 16 |
| G high-LR 1e-3 | 1.62e-3 → 4.12e-4 | 1.1e-2 | 1.1e-2 | +0.853 | 1.71 | 16 |

**Findings**
1. **Residual floor is soft** — continued training drops loss_v from 1.7e-3 to ~3e-4 (longer/high-LR/L-BFGS). The TF checkpoint was not at the floor.
2. **Naive re-training degrades controls** — FOC rises from ~4e-4 (warm start) to ~1e-2 (worse for value-only / L-BFGS). The warm start's value↔control consistency is fragile.
3. **float64 ≡ float32** — identical residual (7.34e-4 vs 7.20e-4) and level drift (0.749 vs 0.747), 18× slower. **Precision is NOT the bottleneck.**
4. **🎯 The value LEVEL drifts ~+0.75 (absmax 1.5–1.8) under ALL configs** while the residual stays ~1e-3. Residual "stiffness" in the level direction ≈ `1e-3/0.75 ≈ δ`. **Minimizing the residual does not pin the level** — the quantified root of the inter-regime weak-id.

---

## Test 2 — single-point method trials (`scripts/method_trial.py`, `scripts/costate_trial.py`)

From scratch, 3 seeds, 3000 steps. Metric: **level_spread** = std over seeds of the probe-mean V
(low = level pinned), with loss_v / FOC as the cost guardrail. Raw lines:
`results/method_trial_RESULTS.txt`.

| variant | loss_v | FOC_d / FOC_g | **level_spread** | controls trained? |
|---|---|---|---|---|
| baseline (padded+softplus) | 0.014 | ~4e5 / ~4e5 | 1.092 | ❌ stuck (c→clamp) |
| no_BN | 0.014 | ~4e5 | 1.092 | == baseline (frozen BN is a no-op in the port) |
| linear_output | 0.0024 | ~1 / 0.5 | 1.182 | ✅ (= clean weak-id reference) |
| no_padding | 0.0043 | 0.13 / 0.11 | **0.224** | ✅ |
| structural_xi `V_∞ − W/ξ` | 0.0034 | 0.21 / 0.15 | **0.552** | ✅ |
| valuenet_anchor | 0.035 | ~6e5 | 0.057 | ❌ (anchor pins level but arch stalls controls) |
| **🏆 costate_egm (ODE level-recovery)** | **0.0030** | **0.021 / 0.0095** | **0.0077** | ✅ (best FOC; curl 1.2e-4) |

**Findings** (reference = `linear_output`, the working-control baseline, level_spread ≈ 1.18)
- **🏆 costate/EGM level-recovery wins decisively: level_spread 1.18 → 0.0077 (~150×)** with the *best* residual AND controls, and a near-conservative field (curl 1.2e-4). 3-seed levels `[3.688, 3.705, 3.704]` are essentially identical and anchored near the surrogate truth (~3.89). **Not degenerate** (loss_v 0.003 rules out collapse-to-constant). Confirms the theory: **level freedom = one constant; pin it once.**
- **no_padding ~5×**, **structural_xi ~2×** — both real, clean, stackable wins.
- **valuenet_anchor** pins the level (0.057) but the padded-softplus arch stalls its controls — the anchor *principle* works; it needs a trainable arch.
- Side: **softplus is not binding in the root regime** (`frac_v<0 = 0`, v>0 always) but **hurts from-scratch trainability** (baseline controls stall; linear_output's don't). **no_BN is a no-op** in this port (frozen-affine BN has no trainable params).

---

## Test 4 — cross-regime boundary value-matching (proof-of-concept, ξ=0.01)

Pre-damage `PreDamagePostTech` trained from scratch (3 seeds, 3000 steps) with the post-damage value
FROZEN to the `PostDamagePostTech` surrogate; damage jump `g_l = exp(-(1/ξ)(v_post − v_pre))`.
**FREE** (no anchor) vs **ANCHORED** (value-matching `v_pre(Y=y_lower) ≈ v_post_surrogate` at jump onset).
Job 51316855. Trustworthy admissibility metric = fraction of probe with g_avg≥1 (the aggregate `g_avg`
mean is exp-tail junk at ξ=0.01 — do not quote it).

| metric @ ξ=0.01 | FREE | ANCHORED |
|---|---|---|
| gap_spread across seeds | 1.197 | **0.014** (~85×) |
| v_pre level spread | 1.197 | 0.0099 (~120×) |
| **frac probe admissible (g_avg≥1)** | **0.02** | **1.0** |
| loss_v | 0.042 | 0.022 (not hurt) |

**FREE reproduces the production collapse** (per-seed g_avg `{0.017, 171, 0}` — inconsistent, 2% admissible).
**ANCHORED restores admissibility** (gap→0, 100% admissible, every seed), confirming the analytical
prediction: the inter-regime gap needs ONE shared anchor.

**Honest caveats (load-bearing):** (1) post-damage is FROZEN — this shows the anchor *target* is right,
not that a JOINT pre+post solve converges; (2) matched transformed `v=V−logN`, not continuous `V`
(logN differs off-boundary via λ3) — a cleaner anchor adds the logN offset; (3) under-converged, one
degenerate FREE seed inflates its spread (conclusion holds without it).

## Conclusions

- The inter-regime damage-jump collapse is **weak-identification of the value LEVEL**, quantified: the level drifts ~0.75–1.1 freely under residual-minimization (Test 1.4) and across seeds (Test 2 baseline).
- **You cannot fix it by training harder / float64 / better optimizer** — those move the level *more* (Test 1). The level must be **constrained externally**.
- **The costate / EGM "predict differentials → integrate the ODE from one anchor" approach pins the level ~150× without cost** (Test 2).
- **Test 5 (Occam): the RE-CENTERED value net `v_rec = φ(s) − φ(x0) + v0` BEATS the costate/EGM** — level_spread **0.0051** (vs costate 0.0077), loss_v 2.1e-3, FOC comparable, with NONE of the costate machinery (no costate heads, no line integral, no curl penalty, no 2nd-deriv inconsistency). Clean-arch plain (no re-center) = 0.119; re-centering does the work (24×). **Adopt the re-centered value net.** The costate idea was the right *insight* (pin the one constant, per the analytical level=1 result); the re-centered net is its minimal, exactly-conservative realization. Scripts: `scripts/occam_trial.py`.

## Analytical result (Test 3 — derivation, sympy-checked)

Answers "can level-recovery reduce to an ODE by eliminating irrelevant variables?":
- **Level recovery = a 1-D ODE (quadrature).** Given the costate field, `V(s) = V(anchor) + ∫p·dl`; axis-aligned paths make it a chain of `dV/dt = p_i`. **Level freedom = exactly 1 scalar per regime**, fixed by one anchor. Requires the field conservative (symmetric Hessian: 3 conditions root / 6 in 4-state).
- **Why costate/EGM pins the level (rigorous):** differentiate the HJB w.r.t. each state → the **coefficient of the bare level is exactly 0** (sympy-verified); `−δV` becomes `−δ·p_i`. So the **costates satisfy a closed, level-free first-order system**; the level is a downstream quadrature. Raw-`V` DGM has only the weak `−δV` (δ=0.01) anchor → level wanders; costate parameterization does not.
- **No coordinate is analytically eliminable.** `V` is NOT additively separable in `logK` — obstructed by (a) the `+δ logK` flow (`∂_logK = δ ≠ 0`) and (b) climate coupling through `exp(logK)` (residual is quadratic in `exp(logK)`, `O(u)`–`O(u²)`). The costate PDE stays full-dimensional; only the final level-recovery is 1-D. **But** climate coupling is mild → `V ≈ logK + W(Z,Y)`, `V_logK ≈ 1` — a strong normalization/warm-start structure (a candidate "structural-logK" parameterization, analogous to structural-ξ).
- **Inter-regime payoff:** the gap `V_post − V_pre` needs only **one shared anchor** (fix `c_post − c_pre`), not two absolute levels — boundary value-matching fixes the offset → the whole gap field → `g` everywhere.

## Next (planned, sandbox)

1. **JOINT pre+post solve** (removes the load-bearing frozen-surrogate limitation of Test 4): co-anchor both regimes' levels to one global reference (e.g. the terminal regime) instead of freezing a known V_post, and confirm g stays admissible at ξ=0.01 → 0.005 under the joint solve.
2. **logN-corrected anchor** (match continuous `V`, not transformed `v`).
3. **Structural-logK** parameterization `V = logK + W(Z,Y,…)` (build in `V_logK ≈ 1`).
4. Push to the **full 4-regime** one-jump system; re-run the production admissibility test to pin the new ξ*.
5. Loop in **Joe & Mike** before any production change.

## How to run

```bash
module load python/anaconda-2021.05
cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal
# Test 1 (warm-start ablation): 7 configs
sbatch benchmarks/torch_level_id/scripts/ablation_root.sbatch     # or: python -u .../ablation_root.py
# Test 2 (method trials): one variant at a time
python -u benchmarks/torch_level_id/scripts/method_trial.py --method costate_egm --steps 3000 --seeds 0,1,2
python -u benchmarks/torch_level_id/scripts/costate_trial.py --steps 3000 --seeds 0,1,2
```
Scripts import the validated port from `models_torch/` and `models_torch_train/` (absolute paths);
they write working outputs to the session scratchpad. CPU, torch 1.12.
