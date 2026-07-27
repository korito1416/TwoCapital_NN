## Final Review — TwoStageTechJump_FOCIr (one-tech-jump, pi=1)

Scope: model math + NN-HJB training method. Severities are the *corrected* (post-adversarial) ones. Refuted findings are dropped. Every claim is cited to verified `file:line`.

### Bottom line
The MODEL MATH is essentially correct. The two "high" math findings (logR Itô sign; post-damage `h_y` slope) are both **real but numerically immaterial** (downgraded to low by adversarial verification) — fix them for correctness/consistency hygiene, but they do NOT explain the control-identification symptom. The real lever is **training-side**: a genuine ill-conditioning problem (κ(JᵀJ)≈7e6, weak v′ identification) plus a dead/misleading BatchNorm with no input normalization. The deliverable preconditioner is well-grounded but its scale assumptions needed re-checking (see "Important scale correction" below).

---

## THEME 1 — Model math correctness

**1.1 [LOW, confirmed] logR drift carries +½σ_κ² instead of −½σ_κ² in all pre-/interm-tech HJB regimes; PDE disagrees with its own deterministic simulator.**
- Evidence: `models/PostDamagePreTech.py:255`, `models/PreDamagePreTech.py:258` — `v_logR_term = -ζ + ψ0*exp(...) + 0.5*σ_κ**2 + σ_κ*h_r`, multiplied by `dv_dlogR` at `:284`/`:297`. Spec `PAPER_HJB_REFERENCE.md:37-40` requires −½σ_r² and explicitly warns "NOT +½σ_r²". `SimulationDeterministic.py:542` uses −0.5σ_κ² (correct); `SimulationRandom*.py` uses +0.5σ_κ² (wrong). The diffusion term `v_logRlogR_term = 0.5*σ_κ²` (`:256/:259`) is carried *separately* on `d2v_dlogR2` — so this is a genuine extra wrong-sign drift, not a rearrangement.
- Impact: σ_κ=0.0078 (`params.py:50`) ⇒ defect = +σ_κ² ≈ 6.1e-5/yr, ~0.1% of the competing logR drift terms. Will not move policies; the symptom downgrade to LOW is justified.
- Fix: flip `+ 0.5*σ_κ**2` → `- 0.5*σ_κ**2` in `v_logR_term` in `PostDamagePreTech.py:255`, `PreDamagePreTech.py:258` (and the two Interm files), and in `SimulationRandom*.py`. Do it in `models_precond/` too.

**1.2 [LOW, confirmed] `h_y` uses the pre-damage slope (λ1+λ2·Y) while `v_logN_term` uses the post-damage slope in the two POST-damage regimes — worst-case substitution is internally inconsistent.**
- Evidence: `PostDamagePostTech.py:197` and `PostDamagePreTech.py:213` set `h_y = -(1/ξ)(dv_dY - (λ1+λ2*Y))*η*A_d*(1-Z)*K*ϛ`, but `v_logN_term` at `:235`/`:253` uses `(λ1+λ2*Y+λ3*(Y-y_upper))`. The closed-form drag identity only holds when the (logN)_y inside `h_y` equals the (logN)_y multiplying `v_y_term`. Pre-damage regimes are already consistent.
- Impact: error term is `(ϛ²Eₜ²/2ξ)·(λ3(Y−ŷ))²`; ϛ≈2.23e-3 (`params.py:58`) makes the whole climate-distortion block O(ϛ²)≈5e-6. Worst-corner absolute rhs error ~7e-6 (≈0.5% of the climate drift term, ≈0.5% of the 1.5e-3 plateau). Real but negligible; unrelated to control identification. LOW.
- Fix: in `PostDamagePostTech.py:197` and `PostDamagePreTech.py:213` replace `(λ1 + λ2*Y)` with `(λ1 + λ2*Y + λ3*(Y - y_upper))`.

**1.3 [MEDIUM, confirmed] Unbounded `g = exp(-(1/ξ)(Vℓ−V))` overflows float32 for small ξ; nonfinite mini-batches are only caught post-hoc (abort/garbage gradient).**
- Evidence: `PreDamagePreTech.py:268,273,283`; `PostDamagePreTech.py:265`; `PreDamagePostTech.py:251` compute `g = tf.exp(-1/ξ*(V_post - v))` with NO clip; `logξ_min=-3` (`params.py:34`) ⇒ ξ_min≈0.05, so |Vℓ−V|~few units overflows exp(>88). The only guard is the post-hoc nonfinite break (`PostDamagePostTech.py:427`).
- Fix (the established "jump-exp clamp"): `expo = tf.clip_by_value(-(1/ξ)*(V_post - v), -50., 50.); g = tf.exp(expo)` and compute `g*log(g) = g*expo` directly. Apply to every `g_l*`. Does not change the converged solution; removes the nonfinite aborts. Apply in `models_precond/`.

**1.4 [LOW, confirmed-convention] Damage-jump post-value is evaluated at fixed Y=y_upper, not the running Y.**
- Evidence: `PreDamagePostTech.py:250`, `PreDamagePreTech.py:282` evaluate the post-damage net at `tf.ones(...)*y_upper`. This is plausibly the intended BBHHH anchoring at ŷ=ȳ=2.5 (which makes `λ2(y−ŷ+ȳ)=λ2·y` exact and the slope formula correct), but it is not derivable from the generic jump operator.
- Fix: confirm against the BBHHH appendix; if anchoring is intended, add a one-line comment documenting that ŷ=ȳ=2.5 is load-bearing for the `v_logN_term` slope. No code change unless the appendix says "running Y".

**1.5 [LOW, doc-only] Three regime docstrings say `v = V − logN` while all four implement `v = V + logN`.**
- Evidence: `PostDamagePostTech.py:4` correct ("v = V + log N"); `PreDamagePostTech.py:4`, `PostDamagePreTech.py:4,211`, `PreDamagePreTech.py:4` wrong. The math (`pv=δv`, `flow=δ(log c+logK)`, `rhs += -v_logN_term`) implements `v=V+logN` everywhere. Fix the comments only.

**What is FINE (math):** the closed-form robust inner min is correct and verified (`PostDamagePostTech.py:192-197`, drag `+0.5ξΣh²` at `:242`). h is never a trainable variable, so there is NO live min-max — no extragradient/OGDA needed. Do not add an inner h iteration.

---

## THEME 2 — Optimization / conditioning (the real problem)

**2.1 [MEDIUM, confirmed] Weak v′ identification + global ill-conditioning (κ(JᵀJ)≈7.16e6); `models/` has NO residual preconditioning.**
- Evidence: `PostDamagePostTech.py:305-308` sums `sqrt(mean((rhs-pv)²)) + sqrt(mean(FOC_g²)) + sqrt(mean(FOC_d²)) + dvdY` with UNIT weights. Conditioning study (re-run, verified): κ(JᵀJ)=7.16e6; #1 HJB-residual weight=591x; **#3 block balancing=1855x (best single)**; naive #1*#3=0.3x (BACKFIRES); principled #1→#3→#2col=2209x (`precond_conditioning_study.py:163-197`). The local FOC/HJB sensitivity ratio is median ≈32, max ≈223 (NOT "hundreds/thousands" as originally claimed).
- Correction to prior framing: this is a real conditioning issue, but **block balancing (#3), not the pointwise weight (#1), is the strongest lever**. The symptom (V agrees 2-5%, controls disagree 50-360%) is more tied to control-Hessian / block conditioning than to v′-residual weight alone. Severity MEDIUM (not "dominant high"). See the preconditioner spec for the safe first-run choice.

**2.2 [LOW, uncertain mechanism] Control loss mixes a signed `-mean(rhs-pv)` with two RMS FOC terms.**
- Evidence: `PostDamagePostTech.py:294-296` (and `:336/:348/:317` in the other regimes). The signed mean has different units/scale from the RMS FOCs and no explicit coefficient.
- Adversarial correction: the envelope identity `∂rhs/∂i_d=(1-Z)·FOC_d`, `∂rhs/∂i_g=Z·FOC_g` (verified from `pde_rhs`: i enters only via `flow` and `log(1+θ_j i_j)`) means the Hamiltonian-term gradient w.r.t. the controls IS a Z-reweighted copy of the same FOC residuals. So the term is **redundant, not antagonistic** — its claimed "inconsistent pull at the optimum" is FALSE (both gradients vanish at FOC≡0). It is not a material driver of control disagreement. LOW.
- Optional cleanup (not required for first run): drop the bare signed mean and train controls on FOC-only with per-equation Newton weights `w_j=(1+θ_j i_j)/(Γ_j θ_j)` (diagonal #2 scaling, which drives cond(−H_ctrl)→~1, `precond_conditioning_study.py:149-152`). Do NOT touch this on the first preconditioned run — keep the control loss RAW so the comparison is clean.

**What is FINE (optimization):** two-step Adam value-then-control is a valid damped policy iteration; keep it.

---

## THEME 3 — ML-systems / stability / efficiency

**3.1 [HIGH→corrected MEDIUM, confirmed-with-correction] BatchNorm is dead and there is NO input normalization.**
- Evidence: `feedforward_subnet.py:46` `call(x, training=False)`; every net call (`PostDamagePostTech.py:164-166`) and `pde_rhs` (`:110`, no training arg) runs BN in inference mode; grep confirms no call site passes `training=True`. moving_mean/var stay at Keras defaults 0/1, so BN reduces to `γ·x+β`.
- Adversarial correction (important): γ,β ARE trainable and ARE in `v_nn.trainable_variables` (`:337`), updated every Adam step. So BN is a **trainable redundant diagonal affine**, NOT a "frozen random map." Its Jacobian is the constant diagonal γ — smooth, Lipschitz; it does NOT "corrupt" the 1st/2nd PINN derivatives. The "critical / training-killing" framing is refuted. The genuine defects: (a) BN provides *zero* normalization (dead code), and (b) raw heterogeneous inputs (logK∈[4,7], logξ∈[-3,5]) are never normalized — a real conditioning issue given κ≈7e6, and (c) γ/β initializers (`:14-15`) are unseeded, a plausible cross-run-variance source. MEDIUM.
- Fix (in `models_precond/`): remove BN; add a fixed input-normalization-to-[-1,1] layer using the known sampling bounds (`logK_min/max`, etc., already in params) as the first op; plain Dense+swish (value)/Dense+tanh+custom (controls). The capacity test fit a tiny tanh MLP with NO BN to ~1e-4 — representation is not the bottleneck.

**3.2 [LOW, confirmed-but-headline-refuted] 1,000,000 iterations is oversized; no early stopping.**
- Evidence: `sensitivity_stage.sbatch:28` NUM_ITERATIONS=1e6; loop runs full budget (`PostDamagePostTech.py:412`), only break is the nonfinite guard (`:429`); best-weights tracked (`:430-435`) but no patience counter.
- Adversarial correction: the original "<5% loss change / 7-20x overkill" is REFUTED — it only inspected `loss_v` (which is flat after ~50k), but `FOC_d/FOC_g` keep dropping ~15-18x through the cosine tail (700k-990k), and the validation *score* (loss_v + 5·ΣFOC) falls ~3.4x over that range. So the controls — the thing we care about — keep refining late, partly because the cosine LR→0 only near the end. Cutting blindly to 150k would forfeit control refinement. LOW.
- Fix: keep the directionally-correct kernel — add a score-based early stop (patience ~30 logging intervals on `validation_score`, restore best weights) + log initial vs final min_loss. Choose iteration budget per THEME 5 (with the LR schedule re-scaled so it actually anneals).

**3.3 [MEDIUM, confirmed] WarmupCosine `total_steps=num_iterations`, so any early stop leaves LR un-annealed; and there are TWO base LRs.**
- Evidence: `feedforward_subnet.py:290-295` `WarmupCosine(total_steps=num_iterations)`, min_lr=0, warmup=n//100. At 50-150k the cosine factor is ~0.95-0.99 of base ⇒ most of training runs at near-peak LR. `LEARNING_RATES=10e-6,10e-4`=[1e-5,1e-3]; optimizer[0]@1e-5 trains the VALUE net (`:350`), optimizer[1]@1e-3 trains the CONTROL nets (`:354`). So the near-peak LR is specifically the ill-identified control net.
- Fix: when you set the iteration budget, `total_steps` auto-follows so it anneals. Additionally anneal the control LR faster / lower it; note "10e-4"=1e-3 in a comment. MEDIUM.

**3.4 [MEDIUM, confirmed] No controllable random seed in the training path.**
- Evidence: `set_seed` exists only in `SimulationStochasticJumps.py:538-539`; training regimes have none. Init uses seed=0 but warm-start overwrites weights (`PostDamagePostTech.py:393-396`); the live stochastic source is the unseeded minibatch stream (`feedforward_subnet.py:67,70`). With 50-360% cross-run control spread, a single run per arm cannot separate preconditioner effect from sampling noise.
- Fix: read `MODEL_SEED` env in each `__main__`, apply `tf.random.set_seed`+`np.random.seed` BEFORE net build and the loop; thread through `sensitivity_stage.sbatch` like the `MODEL_GAMMA_*` overrides. Treat seed as the unit of replication; "B beats A" must hold across ≥3 seeds. This is a prerequisite for trusting any preconditioner verdict.

**3.5 [MEDIUM, confirmed] Cross-arm metric comparability: the preconditioned arm reweights the training residual, so its in-training loss and `validation_score` are NOT comparable to vanilla.**
- Evidence: `validation_score` (`feedforward_subnet.py:109-119`) and best-checkpoint selection (`PostDamagePostTech.py:402-405,423-431`) minimize the value-loss residual. If the value loss divides by w, the score lives on a different scale.
- Fix: keep the `training=False` eval branch (`:318`) RAW (unweighted `sqrt(mean((rhs-pv)²))`) so reported pde_rmse and best-checkpoint selection stay apples-to-apples; recompute the audit metrics RAW for both arms. NEVER compare the weighted training loss across arms.

**3.6 [MEDIUM, confirmed] `if control_constraints > 0` inside `@tf.function` traces a data-dependent branch that replaces the whole batch objective with a hinge penalty.**
- Evidence: `objective_fn` is `@tf.function` (`:258`) with `if control_constraints > 0:` (`:284`) returning a constraint-penalty branch (`:290`) vs the HJB-loss branch. When any point is infeasible the HJB/FOC gradient is discarded for that step.
- Fix: replace with an always-on smooth additive penalty: `loss += λ_pen*(softplus(-(1+θ_d i_d)) + softplus(-(1+θ_g i_g)) + softplus(-c))`. Keep the `maximum(...,1e-8)` clamps for log-domain safety. Pairs well with using the bounded `investment_rate_activation` (already 'custom' for i_g,i_d at `OUTPUT_LAYER_ACTIVATIONS`, `sbatch:33`).

**3.7 [LOW, hygiene] eps literal inconsistency (`10e-8`=1e-7 vs `1e-8` clamps) at `objective_fn:267,270,302,316`. Standardize to a named constant. No functional change.**

**What is FINE (systems):** investment activation + log-domain clamps are safe (`params.py:101`, `PostDamagePostTech.py:206,214-215`); CPU/caslake is the right device for a 4k-param net (GPU only wins if batch pushed to ~16k+); the tensorboard recompute is gated off in production (`sbatch`).

---

## THEME 4 — PINN / methodology

**4.1 [MEDIUM, confirmed] No boundary / jump-continuity residuals; only a one-sided `dv_dY>0` penalty above y_upper.**
- Evidence: `PostDamagePostTech.py:301-308` penalizes `dv_dY>0` ONLY where `Y>y_upper`; below y_upper v(Y) monotonicity is unconstrained (weak ID). Pre-tech/pre-damage jump terms read neighbor nets at shifted states (`PreDamagePreTech.py:282`), so value-continuity is the de-facto coupling with NO explicit continuity residual — correctness depends entirely on upstream nets being converged (the "stale-checkpoint gotcha").
- Fix: (a) extend the monotonicity penalty to the full Y range where the damage slope implies v_Y<0; (b) verify each upstream regime is converged before training downstream. Optional: explicit jump-matching residual at y_upper.

**4.2 [LOW, methodology] Collocation: per-axis stratification is FINE (the "diagonal sweep" claim was REFUTED — `stratified_uniform` shuffles each column independently at `feedforward_subnet.py:70`, giving a valid product/LHS sample; fresh batch each of 1e6 steps ⇒ ~128M points, not a fixed 128-pt grid; logξ coverage is uniform, the [-3,0] band gets 3/8 not 1/8).**
- The only surviving point is a *difficulty* (not coverage) issue: residual stiffness rises at small ξ (h*=−exposure/ξ). Optional importance/curriculum weighting on small logξ, or residual-adaptive resampling (RAR), if the small-ξ corner remains under-resolved after the preconditioner. Not required for the first run. Raising batch_size to ~1024-2048 (cheap, reduces SGD variance) is the higher-ROI move.

**4.3 [LOW, confirmed] Input X wastes columns: `PostDamagePostTech.py:160` X concatenates a CONSTANT `A_g_prime_prime` column and DUPLICATES logξ; logR is sampled but unused in this regime.**
- Fix: harmless under frozen-BN but once BN is removed the constant column is a dead feature. When removing BN, prefer the 5 genuine states (logK, Z, Y, λ3, logξ) and set n_inputs accordingly — BUT verify pretrained checkpoint width (trained at 7) before changing; if reusing checkpoints keep width 7 and just de-duplicate.

**Do NOT add Fourier features / PirateNet first** — the capacity test shows representation is not the bottleneck; that would add params without addressing the optimization floor. Reserve a larger net only as a fallback arm.

---

## THEME 5 — Experiment design

- Use the seed as the unit of replication (THEME 3.4): ≥3 seeds per arm, report min/median/max of RAW metrics.
- Success criterion: control cross-run spread `i_d/i_g relative_delta` (baseline ~0.5-3.6, `network_training_audit_all_regimes_4096.csv`) must tighten to <~10% while RAW pde_rmse and V_relative_delta (2-5%) do NOT regress. Declare success only on that, measured with the RAW eval metric.
- Order of changes (do not stack): (1) BN removal + input normalization + seed control FIRST (architecture must be clean before preconditioner tuning is identifiable); (2) jump-exp clamp + smooth constraint penalty; (3) #1 pointwise residual weight ONLY; (4) escalate to safe #1→#3 frozen-β block balancing only if #1 alone doesn't tighten controls.

---

## IMPORTANT scale correction to the preconditioner design
The adversarial verdict claimed `v_y_term ~ O(hundreds)` dominates w, making `eps=5e-3` "meaningless." I verified numerically with the actual calibration (η=0.291, A_d=0.1303, θ̄=1.86e-3, ϛ=2.23e-3, `params.py:39,56,57,58`): `v_y_term` ranges 0.0019 (logK=4) → 0.0086 (logK=5.5) → **0.039 (logK=7)**, while `|v_logK_term|`≈0.035. They are the SAME ORDER, not hundreds-vs-0.01. So eps=5e-3 IS meaningful (~1/8 of the dominant terms) and the realized 1/w dynamic range across a batch is only ~2x, not 100x. The K=exp(logK) factor is real but bounded because logK≤7. This makes the finalized weight below safe and the eps choice defensible.