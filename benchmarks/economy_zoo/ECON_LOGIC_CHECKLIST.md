# Master Economic-Logic Checklist — Barnett–Brock–Hansen–Hu–Huang climate-innovation model

**Merged from 5 verified lenses** (foc-statics, value-costates, robustness, jump-composition, climate-block), deduplicated 74 → **46 checks**. Repo: `/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal` (branch `TwoTechJump`). Authoritative math: the NBER tex + `AGENTS.md` + `PAPER_HJB_REFERENCE.md`.

**How to read.** Each check states: the signed claim, WHERE the sign comes from (FOC differentiation / envelope / value-function monotonicity / intensity algebra / comparison principle), the honest ambiguity (offsetting GE forces) and which experiment resolves it, and the exact protocol. Status tags: **[ANCHORED]** validated by an existing artifact; **[FD-NEW]** needs a new cheap terminal-regime FD solve (exact `--set` given); **[SIM]** simulation readout; **[PROBE]** trained-net checkpoint probe; **[RETRAIN]** needs a variant NN solve. Ordering: severity tiers first (FATAL → STRONG → INFORMATIVE), blocks within each tier.

---

## Global validity guards (apply to every check)

- **G1 (ξ frontier).** Any object dividing a cross-regime value difference by ξ (g's, π̃, λ3 weights, jump densities) is reliable only for **ξ ≥ 0.05**; the documented level-identification frontier is ξ* ≈ 0.025 (δ=0.01 level anchor × 1/ξ amplification). A failure below 0.05 indicts the level anchor, not the economics. Report the ξ* where belief-orderings break as the solution's belief-validity frontier.
- **G2 (entry-slice rule).** Post-damage regimes are economically visited only at Y ≥ ŷ = 2.5. Evaluate them at the entry slice Y = 2.5 — **except** the h_y×λ3 leg (RX-3), which is identically zero there and must be probed at Y = 3 (see CR-3).
- **G3 (visited box ≠ training box).** Fatal verdicts on state-space claims apply to the visited box (5th–95th pct of simulated states at t = 20/40/60). Corner violations (Z→0.99, extreme logK, Y=4 boundary ±2 cells) are informative only.
- **G4 (level slack).** Cross-net value DIFFERENCES carry ~0.075 additive slack (residual RMS 7.5e-4 / δ). Ordering violations within slack = solver-health log; beyond slack = fatal. Level-immune objects (same-net λ3 slices, derivatives, dS/dY, FOC identities) are exempt.
- **G5 (quantity dictionary).** R&D readout = `RD.txt` (= I_r/Output), NOT `i_r.txt` (= I_r/K); green/dirty investment = `I_g.txt`/`I_d.txt` LEVELS, NOT `GreenInvestment`/`DirtyInvestment` (= I/Y ratios). The level≈ratio-at-t=0 trap has caused 3 false alarms.
- **G6 (transform correctness).** Any V_Y-based object (h_y, SCC, V_YY) must be built as true V_Y = v_y − (logN)_y with the REGIME-CORRECT slope: pre-damage λ1+λ2y; post-damage λ1+λ2(y−ŷ+ȳ)+λ3(y−ŷ). This is the fixed h_y-λ3 bug class — replicate the model code's formula, not the paper's typo-flagged constants.
- **G7 (warm-start confound).** Cross-run comparisons of independently-trained checkpoints are contaminated by warm-start basin effects (seed-control null; box-family attractor). Any V-level comparison across parameter arms requires a COMMON warm-start base, identical schedule, fixed MODEL_SEED.

---

## Conflict resolutions (explicit, between-lens)

- **CR-1 — δ-slope bookkeeping** (foc-statics vs value-costates/robustness). Foc carried a "+dlog c₀/dδ ≈ +1/δ consumption-level offset" against −g/δ²; costates/robustness show the level term integrates out EXACTLY: with flow δ·logC̃ and ∫δe^{−δt}dt ≡ 1, the envelope derivative holding the policy fixed is dV/dδ = −g_u/δ² with the u₀ term contributing exactly 0. **Resolution: envelope form wins as the clean statement.** The foc lens's consumption-level gain and the extra growth loss it trades against are policy-mediated terms that cancel pairwise by the envelope — both lenses net to −g_u/δ² (predicted ≈ −290 at g_u ≈ 0.029; FD anchor −320). Not a sign conflict; magnitude gate = −g_u/δ² with g_u read from each solve's own realized growth.
- **CR-2 — damage-side distortion strength** (foc `damage_jump_accel` "ḡ>1 fatal, clearly earlier density" vs robustness `damage_density_early` correction vs jump `g_asymmetry`). **Refined version wins**: fatal only for g^ℓ non-increasing in ℓ, g^L ≤ 1, ḡ significantly < 1, or a tech-sized damage-side distortion. ḡ ≥ 1 weakly with MODEST magnitude is the paper-consistent expectation ("almost no adjustment to damage function probability"); the sign of g¹−1 is recorded, not gated (the ℓ=1 jump reveals the benign model AND extinguishes the hazard — good news, so V¹ > V is admissible). The foc lens itself flagged its Jensen route as not airtight; this is refinement, not contradiction.
- **CR-3 — h_y×λ3 probe location** (foc: probe at Y = ŷ = 2.5 and 3.0 vs robustness: vacuous at 2.5). **Robustness wins by algebra**: the λ3 term in (logN)_y is λ3·(y−ŷ), identically zero at ŷ. The λ3-monotonicity leg MUST be probed strictly above ŷ (Y = 3, where the λ3 contribution ≈ 0.083 dominates λ1+λ2 ≈ 0.013), avoiding Z ≈ 1 (E → 0 degeneracy). Positivity and 1/ξ-scaling legs may use both slices.
- **CR-4 — dV′/dπ** (jump lens's original "V′ π-independent" vs its own correction). **Corrected version wins**: the intermediate regime carries breakthrough intensity π·R (paper appendix; `models/PreDamageIntermTech.py:255`, `PostDamageIntermTech.py:253`), so dV′/dπ = FK[ξ(R/ϱ)(1−g″)] > 0. The TRUE π-invariance in the tree is post-tech: dV″/dπ ≡ 0 (no jump terms). Convention-detecting: grep the IntermTech jump construction before applying the sign; under a total-intensity-R refactor the sign reverts to exactly 0. The correction STRENGTHENS dV/dπ > 0 (RD-2).
- **CR-5 — i_r vs aversion, dual tex citations** (final text L1157/L792: R&D UP under aversion; commented L1188: "significantly lower R&D"). **Resolution: genuinely ambiguous in theory; the DELIVERED full-channel resolution is UP** (final text + published figures supersede the comment). L1188 is an earlier-vintage / tech-only-channel record, repurposed as the mechanism experiment: a tech-only-uncertainty variant must FLIP the ordering (XE-1). `race_composition_in_r1`'s premise (which took L1188 as current) is updated; its mechanical-dominance conclusion survives because even the delivered UP response is moderate (JC-9).
- **CR-6 — η leverage magnitude** (foc's self-correction of "~70" to Γc/δ ≈ 0.66). Corrected RELATIVE gate adopted: |Δlog i_d|/|Δlog v| ≥ O(50) — the absolute q_d shift and Δv are the same order; the leverage is baseline i_d being tiny (CD-4).
- **CR-7 — i_r vs jump-intensity scale** (foc: globally nonmonotone, rising-at-calibration a hypothesis vs jump: di_r/dϱ < 0 at calibration, V-leg fatal). **Unified**: the V-leg (V pointwise increasing in scale) is the fatal, unconditional envelope gate; the i_r-leg is a strong calibration-prior gated only on scale ∈ {0.5, 1, 2}, with scale = 4 as turning-point exploration (informative). An i_r monotonicity break with a clean V-leg = economics finding (surplus-compression branch), not auto-fail; a V-leg break = solver bug (RD-3).
- **CR-8 — π̃ neutral limit** (exact π = 0.04 vs paper's printed 0.0369). 0.0369 is a year-40 path-weighted simulated readout, not the pointwise limit. Require the neutral limit within ~10% of π and, at ξ = 0.05 (printed 0.0000), test the LOG-ODDS identity against the measured V″−V′ instead of ratios (JC-3).
- **CR-9 — A_g″ pre-tech anticipatory gap** (costates asserts i_g↑/i_d↓ generally; foc flags the wait-option + R&D-diversion offset at t=0 pre-tech). Post-tech legs are fatal (FD-anchored). The pre-tech t=0 gap is demoted to informative — deferral (build green after the jump) and i_r diversion can locally reverse it in a CORRECT solution; the pre-tech gates are the emissions-path ordering and i_r↑ (RD-1).
- **CR-10 — V_logK < 1 scope** (climate lens's correction of the blanket claim). Post-tech (no-R&D) regimes only; R&D-active regimes carry the opposing +V_logR·∂ψ_r/∂logK > 0 source (bigger economy accumulates knowledge/hazard faster), sign numerical there. Consistent with the S1 memory (v_logK → 1 exactly at the Z=1 BGP corner) (PT-5).
- **CR-11 — protocol capability** (what each harness can test — verified against the code this session). The terminal FD calibration dict `fd_pdpt_v5.P` = {delta, A_d, A_gpp, a_d/a_g, G_d/G_g, t_d/t_g, s_d/s_g, thbar, eta, vars, l1, l2, y_up}: NO hazard keys (r1, r2, y̲) and no ϱ/π — the post-damage terminal regime has no jump term, so r1-family checks are FD-inert there. `_ENVIRONMENT_OVERRIDES` = {MODEL_SIGMA_D/G, MODEL_GAMMA_D/G, MODEL_THETA_D/G, MODEL_PSI0} only — A_g″, δ, π, r1, r2, y̲ need params edits / argv / one-line override additions (MODEL_SIGMA_D pattern). π = argv[14], intensity scale = argv[13] of the pre-tech solvers. ŷ = y_up stays pinned at 2.5 when varying y̲ (moving y_up confounds the post-damage damage function with hazard timing).

---

# TIER 1 — FATAL-IF-VIOLATED

## Block A: Preferences / technology

### PT-1. dV/dδ < 0, magnitude −g_u/δ² (≈ −300 at δ = 0.01; anchored slope −320)
- **Sign source:** Envelope + delta-normalized weighting. V = E∫δe^{−δt}(logC̃+logK)dt; holding policy fixed (saddle-point envelope kills control AND distortion terms), the level term contributes exactly 0 (∫δe^{−δt}dt ≡ 1) and the trend term gives dV/dδ = −g_u/δ², g_u ≈ 0.029 (balanced growth). See CR-1 for the bookkeeping merge.
- **Ambiguity:** none in sign. δ is exactly the level-identification channel (∂residual/∂level = −δ): a δ-insensitive V, or one moving O(0.1) instead of O(1), has a level/anchor pathology.
- **Protocol:** [ANCHORED] `fd_dlt{0080,0090,0110,0125}.npz`: v(ref) 7.240 → 5.802 over δ 0.008 → 0.0125. Gate: slope within ~15% of −g_u/δ² using each solve's OWN realized logK+log(C/K) growth ([SIM] leg for g_u). Candidates: rerun `run_solver.py --method pibys --set delta=X`; NN leg via `ab_delta_override/` family [RETRAIN]. Gotcha: deep-δ arms need the jump-exp clip ±35 (fd_dlt0080 NaN history).
- *Merged from:* foc `dV_ddelta`, costates `dV_ddelta_level`, robustness `dV_ddelta_growth_envelope`.

### PT-2. A_g″ terminal statics: dV″/dA_g″ > 0, di_g/dA_g″ > 0, di_d/dA_g″ < 0
- **Sign source:** Envelope in the absorbing regime: dV″/dA_g″ = E∫δe^{−δt}Z_t/(C/K)_t dt > 0. Cross-partial of δlog c in (Z, A_g″) > 0 raises V_Z ⇒ Q_g = V_logK+(1−Z)V_Z up (i_g↑ by FOC), Q_d = V_logK−ZV_Z down (i_d↓).
- **Ambiguity:** none for the terminal legs. (Pre-tech t=0 gap: demoted per CR-9 → RD-1/informative.)
- **Protocol:** [ANCHORED] `fd_agpp{0150,0165}.npz`: v 6.206 → 6.670, i_g 0.157 → 0.172, i_d 0.0126 → 0.0080. Candidates: `--set A_gpp={0.150,0.165}`, all three legs at the reference slice.
- *Merged from:* costates `dAgpp_block` (terminal legs), robustness `Agpp_statics_and_g_cross` (level/policy legs), foc `dgapE_dAgpp` (post-tech part).

### PT-3. FOC-ratio identity + V_Z > 0 post-breakthrough
- **Sign source:** FOC division with identical (Γ,θ) across sectors (params-verified): (1+θi_g)/(1+θi_d) = [V_logK+(1−Z)V_Z]/[V_logK−ZV_Z] EXACTLY — couples control nets to the value costate. Post-breakthrough V_Z > 0 by Feynman–Kac: swapping dirty→green at fixed K gains flow A_g″−A_d = 0.0264 > 0 AND the avoided-emissions damage annuity — both components positive, no offset; diffusion/robustness contributions are O(σ²), too small to flip. Hence i_g > i_d there.
- **Ambiguity:** scope restricted to the visited box (G3); pre-tech V_Z's sign is an OUTPUT, not asserted (A_g < A_d), though anchors imply positive at calibration.
- **Protocol:** [PROBE] (i) identity residual |(1+θi_g)Q_d − (1+θi_d)Q_g| from i-nets vs autodiff Q's over the visited box — catches value/policy drift-apart; (ii) V_Z sign scan on PostDamagePostTech + PreDamagePostTech: any interior V_Z < 0 fatal. [ANCHORED] FD npz grids give the same scan NN-free.
- *Merged from:* costates `VZ_sign_focratio`.

## Block B: R&D / innovation

### RD-1. di_r/dA_g″ > 0 in pre-tech regimes; 60y emissions path LOWER for larger A_g″
- **Sign source:** R&D FOC δ/c = ψ_r′(i_r)V_logR with ψ_r′ strictly decreasing. J_g ∝ R makes the robust jump bracket ξJ[π(1−g″)+(1−π)(1−g′)] the V_logR flow itself; d/dA_g″ = J·π·g″·d(V″−V)/dA_g″ > 0 since the surplus widens (V″ rises by more than pre-jump V — bracket slope e^{−S/ξ} < 1). Author-intended (tex ~L1192: "bigger tech shock amplifies investment decisions … noticeably larger reduction in emissions"). Robust caveat: the response carries factor g″ ⇒ ATTENUATES at deep aversion, never flips (test at ξ=∞ first).
- **Ambiguity:** i_r/emissions legs are GE (i_r crowds C/K, damping all margins) — sign preserved, magnitude may be small at ξ = 0.05. Pre-tech t=0 (i_g−i_d) gap informative-only per CR-9; the pre-tech gates are emissions-path ordering and i_r↑.
- **Protocol:** [RETRAIN] A_g″ NOT env-overridable (verified) — edit `PARAMS["A_g_prime_prime"]` ∈ {0.150, 0.1567, 0.165}, re-solve full chains post→pre from a common base; read `RD.txt` at t=0 (G5) + probe V_logR at the initial state + emissions series. Flat/falling i_r in A_g″, or a HIGHER 60y emissions path, contradicts the paper's own static. Test ξ ∈ {∞, 0.1} for attenuation-not-flip.
- *Merged from:* foc `dir_dAgpp`, `dgapE_dAgpp` (emissions leg), costates `dAgpp_block` (i_r/emissions legs).

### RD-2. dV/dπ > 0 and di_r/dπ > 0
- **Sign source:** Intensity algebra + comparison principle. Direct: ∂/∂π of the robust tech term = ξ(R/ϱ)(g′−g″) > 0 since V″ > V′ ⇒ g″ < g′ (JC-1). Indirect reinforcement: dV′/dπ > 0 (CR-4) and the jump bracket is monotone in the post-jump value (∂/∂V′ = J·g′ > 0); dV″/dπ = 0, so NO offsetting channel — sign unambiguous. Same bracket = V_logR flow ⇒ i_r↑ via the FOC. (The foc sub-claim "tilts mix toward i_r vs i_g" is DROPPED — not derivable.)
- **Ambiguity:** measurement only: (i) both channels vanish as ξ → 0 (XE-2) so the V-readout at ξ=0.05 is below level noise — test V at large ξ, use level-free i_r at all ξ; (ii) existing π=0.04 vs OneJump π=1 checkpoints have different warm-start histories (G7). Second-order: V rises with π, compressing surpluses — flattening near π=1, no flip.
- **Protocol:** [SIM] primary: OneJump vs baseline families, `RD.txt` t=0–20 higher under π=1; report v with the level-drift caveat. [RETRAIN] controlled: π ∈ {0.04, 0.5, 1.0} via argv[14], common base + fixed MODEL_SEED; require V pointwise increasing at ξ=148.4 and i_r(t=0) + 60y RD path increasing at every ξ; report π=0.5 midpoint for monotonicity.
- *Merged from:* foc `dir_dpi`, costates `dV_dpi_onejump`, jump `dV_dpi_positive`.

### RD-3. dV/dϱ < 0 (V pointwise increasing in tech_jump_intensity_scale) — fatal leg; di_r increasing in scale at calibration — strong leg
- **Sign source:** Envelope: dV/dϱ = −FK[(1/ϱ)ξJ{(1−π)(1−g′)+π(1−g″)}] < 0 unconditionally (g′,g″ < 1 by JC-1 — the tech jump is unambiguously good news; no net-bad condition needed). i_r: the V_logR jump component is the bracket itself ∝ 1/ϱ ⇒ V_logR up with scale, i_r up via decreasing ψ_r′.
- **Ambiguity (CR-7):** i_r-leg only — globally NONMONOTONE: as scale → ∞ arrival is imminent regardless of R&D, J·(surplus) saturates and convex R&D cost forces i_r down. Which branch calibration (30–40y expected arrival) sits on is NOT anchored. Resolution: sweep with i_r-monotonicity gated only on {0.5, 1, 2}; scale = 4 = turning-point exploration (informative). i_r break + clean V-leg = economics finding; V-leg break = bug.
- **Protocol:** [RETRAIN] scale ∈ {0.5, 1.0, 2.0, 4.0} via argv[13] (TwoJump2 = 2.0 exists), common base; V pointwise increasing at large ξ (fatal; large ξ so signal > 0.075 slack), i_r(t=0) + `RD.txt` path (strong), V_logR probe along logR.
- *Merged from:* jump `dvarrho_value_and_rd`, foc `dir_dintensity`.

## Block C: Climate / damages

### CD-1. dV/dθ̄ < 0; di_d/dθ̄ < 0; cumulative emissions decreasing in θ̄
- **Sign source:** θ̄ enters only via V_yθ̄E. Envelope: dV/dθ̄ = Ẽ[∫e^{−δt}V_yE_t dt] < 0 (V_y < 0, E > 0). Policy: ∂/∂θ̄ of the dirty-q source = V_yηA_dK < 0 (no extra θ̄ factor — corrected) ⇒ Q_d↓ ⇒ i_d↓; substitution AND income effects on i_d point the SAME way, so i_d↓ is unambiguous. E-path: i_d↓ ⇒ Ż↑, K̇↓, both lowering E. (i_g is NOT part of this check — CD-11.)
- **Ambiguity:** none.
- **Protocol:** [FD-NEW] `--set thbar=0.0015 / 0.0022` (baseline 0.00186), read v, i_d at ref slice (log 880, 0.7, 3.0), strictly decreasing. Quantitative: Δv/Δθ̄ vs ∫e^{−δt}V_yE dt along the FD-policy path (vY saved in npz). [PROBE] optional: repeat under the distorted measure at ξ=0.05.
- *Merged from:* climate `dV_dthetabar`.

### CD-2. Emission-intensity vs productivity asymmetry: di_d/dη < 0 but di_d/dA_d > 0 (E symmetric in η, A_d); emissions elasticity(η) < 1 < elasticity(A_d)
- **Sign source:** FOC differentiation + incidence. η enters ONLY the E-terms (pure climate cost, zero output side) ⇒ full burden on the dirty margin, i_d↓, with policy feedback OFFSETTING the direct emission effect ⇒ elasticity < 1. A_d also enters consumption: ∂flow/∂A_d ⊃ δ(1−Z)/(C/K) > 0, first-order, ⇒ i_d↑, feedback AMPLIFIES ⇒ elasticity > 1.
- **Ambiguity:** sign asymmetry is double-anchored (fatal); the elasticity ordering is quantitative — an A_d-elasticity < 1 means the damage channel is over-weighted.
- **Protocol:** [ANCHORED] η arms (i_d 0.0158 → 0.0058, v flat) + the existing A_d FD sweep (i_d increasing in A_d, zero-crossing near A_d ≈ 0.073). [FD-NEW] `--set A_d=0.11 / 0.15` (hold A_gpp = 0.1567 — do NOT re-impose the λⁿ identity); [SIM] integrate E along deterministic FD-policy 60y paths per variant, assert dlog(∫E)/dlogη < 1 < dlog(∫E)/dlogA_d.
- *Merged from:* climate `eta_vs_Ad_asymmetry`.

### CD-3. V_Y < 0, V_YY < 0 on the visited box; SCC proxy −V_Y/V_logK > 0, increasing in Y on [1.1, 2.4]; increasing as ξ falls; d(−V_Y)/dλ2 > 0 with dV/dλ2 < 0
- **Sign source:** Feynman–Kac with Y permanent (dY_t/dY_0 = 1): V_Y = −E∫δe^{−δt}(logN)′(Y_t)dt − hazard term (J_n′ > 0 × negative g-weighted jump surplus) < 0. V_YY: flow term −∫δe^{−δt}λ2 dt < 0 + convex-hazard curvature — NOT pure envelope (policy feedback enters second derivatives) but direct terms dominate at calibration. d(−V_Y)/dλ2 = +E∫δe^{−δt}Y_t dt > 0. ξ-leg: distorted measure tilts toward high λ3 AND h_y > 0 raises the distorted Y-path, both raising the damage-slope expectation. V_logK near Y-flat ⇒ ratio inherits |V_Y|.
- **Ambiguity:** visited-box claims only (G3). Ratio could flip if V_logK rose faster than |V_Y| in 1/ξ — rule out by reporting numerator and denominator SEPARATELY. λ2×ξ cross (amplification at small ξ) is a prior with an opposing lower-equilibrium-Y-path channel — deferred to a λ2-override retrain sweep.
- **Protocol:** [PROBE] autodiff per regime on a Y-grid at path-median states, TRUE V_Y per G6; fatal: V_Y < 0, V_YY < 0, ratio > 0 & increasing on [1.1, 2.4]. ξ-leg [PROBE]: ratio across ξ ∈ {148, 0.1, 0.05}, monotone increase as ξ falls. λ2-leg [FD-NEW]: `--set l2=0.0035 / 0.0055` (optionally 0.0066): −V_Y up pointwise (finite-difference saved grids), v down. Cross [RETRAIN, deferred].
- *Merged from:* costates `VY_VYY_SCC`, robustness `scc_ratio_xi_lambda2`.

### CD-4. Dirty-q emission-load identity: the direct marginal-emission cost loads ENTIRELY on q_d and cancels EXACTLY in q_g; consequences: i_d↓ in η (large, −63%) with v flat; dV_Z/dη > 0; i_d and E decreasing in aversion with widening 60y gap
- **Sign source:** Differentiate the HJB in logK and Z: dE/dlogK = E, dE/dZ = −E/(1−Z) ⇒ q_d = V_logK−ZV_Z receives source V_yθ̄E[1+Z/(1−Z)] = V_yθ̄ηA_dK < 0; q_g = V_logK+(1−Z)V_Z receives V_yθ̄E − (1−Z)V_yθ̄E/(1−Z) = **0 exactly** (source-term identity, not full-equilibrium). The i_d FOC has NO direct η term — the entire response routes through the costate, which is why the check discriminates: i_d insensitive to η = dead V_Z climate channel. Aversion amplifies |V_y| (λ3 tilt + h_y > 0) ⇒ q_d↓ ⇒ i_d decreasing in aversion.
- **Ambiguity:** the exact-zero is a DIRECT-source statement — the coupled PDE propagates indirect load into q_g (expected small; verify). dV_Z/dη has a second-order path/MU offset — anchor says direct proportionality dominates. Relative gate per CR-6: |Δlog i_d|/|Δlog v| ≥ O(50), NOT absolute units.
- **Protocol:** [ANCHORED] `fd_eta{0250,0330}.npz`: i_d 0.0158 → 0.0058 (−63%), v 6.426 → 6.409 (−0.26%); NEW readout from the same files: V_Z at ref slice (finite-differenced) increasing in η. [SIM] aversion leg: `I_d.txt` LEVELS + E paths for ξ ∈ {148, 0.1, 0.05}, ordered decreasing in aversion, widening gap. [PROBE] η×ξ steepening: autodiff the q_d V_y-contribution at logξ = log(0.1) vs log(0.05), larger at 0.05; full η×ξ cross [RETRAIN, deferred] (no MODEL_ETA override exists).
- *Merged from:* robustness `dirty_q_emission_load_id_eta`, foc `did_deta`, costates `eta_incidence`.

### CD-5. Hazard localization: damage-jump mass is EXACTLY zero while Y < y̲ = 1.5
- **Sign source:** Intensity algebra: J_n = r1(e^{(r2/2)(y−y̲)²}−1)·1{y≥y̲} ≡ 0 below y̲, and the distorted intensity J_n·g inherits the zero — no distortion can create early hazard. Deterministic time-to-threshold from Y0=1.1 ≈ 15–21y. Corrected: EXACTLY zero only for the deterministic-path pipeline (the paper's Fig construction); fully stochastic paths have σ_Y = ςE ≈ 0.022/√yr ⇒ early mass near-zero (< ~1e-3 by year 10) but not identically zero.
- **Ambiguity:** none for the localization leg.
- **Protocol:** [SIM] `stochastic_density_plot_stage.sbatch`: damage first-jump density exactly zero at every t with Y_t < 1.5 (pathwise consistency); deterministic variant identically zero for t ≲ 15y. ANY hazard mass at Y < 1.5 = broken indicator or y̲. (dV/dr1 envelope leg → JC-6; dV/dy̲ → JC-8.)
- *Merged from:* climate `r1_localization_zero_mass` (fatal leg).

## Block D: Robustness-ξ

### RX-1. dV/dξ ≥ 0 everywhere, every regime; strict where distortions active
- **Sign source:** Value-function monotonicity, airtight: for FIXED (h,g) the objective is nondecreasing in ξ (relative-entropy integrand ≥ 0); the min of a family increasing in ξ is increasing; the outer max preserves it. Danskin at the saddle: dV/dξ = Ẽ[discounted worst-case entropy] ≥ 0; backward induction through the jump tree adds only nonnegative inherited sources J·g·W^ℓ. No GE offset exists.
- **Ambiguity:** none in economics. This is exactly the check the level-anchor defect breaks first — sub-0.075 wiggles are level noise (G4); super-tolerance inversions = wrong saddle.
- **Protocol:** [PROBE] sweep the logξ input over [log 0.05, log 148.4] on all six regime v_nn's, identical value to every duplicated ξ slot; pointwise monotone up to slack. Report the ξ* where ordering fails (belief-validity frontier; known ≈ 0.025 on current canonicals). [ANCHORED] FD ξ arms corroborate strictness.
- *Merged from:* costates `V_monotone_xi`, robustness `dV_dxi_entropy_envelope`.

### RX-2. Neutral-limit rates: h = O(1/ξ), (g−1) = O(1/ξ), entropy cost = O(1/ξ); weights → 1/L, π̃ → π, densities → baseline
- **Sign source:** Taylor algebra, exact: h = −(1/ξ)σ′V_x; g = 1 − (Vℓ−V)/ξ + O(ξ⁻²); ξ(1−g+g log g) = (Vℓ−V)²/(2ξ); jump bracket → J(Vℓ−V), recovering the linear neutral HJB.
- **Ambiguity:** none. Tolerance per CR-8: neutral breakthrough share within ~10% of π (paper's 0.0369 is a path readout, not the limit).
- **Protocol:** [PROBE] at logξ = log 148.4: max|h_j|, max|gℓ−1| ≲ 1e-3; verify the 1/ξ rate by regressing log|h_y| on logξ over the LARGE-ξ half only (slope −1 ± 0.1 — full-range regression is contaminated by V_x's own ~10% drift). [SIM] rerun the distorted-density pipeline at ξ=148: tech/damage first-jump densities overlay baseline to plotting accuracy.
- *Merged from:* robustness `neutral_limit_rates`.

### RX-3. h_y master block: h_y > 0 everywhere (fatal); ξ·h_y ≈ ξ-invariant; post-damage dh_y/dλ3 > 0 at Y = 3; ∂h_y/∂Y|_{Z,K} > 0; distorted sensitivity θ̃ = θ̄ + ς²E|V_y|/ξ increasing in E (hence in K0, η, 1−Z)
- **Sign source:** h_y = −(1/ξ)V_yEς with V_y < 0 (temperature has NO benefit channel: (logN)_y > 0 in both branches; pre-damage adds J_n′ > 0 × negative jump bracket) ⇒ h_y > 0 — the worst case always tilts warming UP. λ3-leg: post-damage (logN)_y contains +λ3(y−ŷ), raising |V_y| in ℓ for Y > ŷ (regression test for the fixed h_y-λ3 bug). Partial ∂h_y/∂Y|_{Z,K} = −(1/ξ)ςE·V_YY > 0 (dE/dY = 0). θ̃-leg: ∂(θ̃−θ̄)/∂E = ς²|V_y|/ξ + ς²E·∂|V_y|/∂E, both positive (|V_y| grows with the future emission path — belief distortion is endogenously state-dependent, not a fixed TCRE reweighting).
- **Ambiguity:** (i) dh_y/dλ3 is a CROSS-partial — envelope does NOT kill the policy feedback (higher λ3 → harder cuts → lower future Y−ŷ); direct effect plausibly dominates at fixed Y=3, resolved by the ℓ-probe. (ii) ξ·h_y invariance holds only to first order in V_y's ξ-sensitivity (~2× at 0.1→0.05, not exact). (iii) The PATH claim "h_y rises while Y rises" is NOT implied — E_t falls as Z_t → 1, h_y can peak and decline; only the fixed-(Z,K) partial is signed. (iv) Boundary slack ±2 cells at Y=4 (G3).
- **Protocol:** [PROBE] per CR-3: post-damage checkpoints at **Y = 3** (NOT 2.5 — vacuous), ℓ = 1..5, ξ ∈ {0.05, 0.1}, avoid Z ≈ 1; compute h_y exactly as the model code does (G6). Fatal: any interior V_y ≥ 0 / h_y ≤ 0 (cross-check terminal vY field in FD npz, NN-free). Strong: monotone in ℓ, ~2× fall at ξ 0.05→0.1, partial-∂Y leg via −(1/ξ)ςE·V_YY on a Y-grid. [SIM] θ̃-leg: tilt orderings at K0 ∈ {880, 1760}, Z0 ∈ {0.5, 0.7}; distortion-histogram right-tail overweight grows with dirtier initialization; distorted temperature drift (θ̄+ςh_y)E exceeds baseline.
- *Merged from:* foc `hy_lambda3`, robustness `hy_sign_and_lambda3_slope`, climate `hy_sign_state_dependence`, costates `distortion_orientation` (h_y legs).

## Block E: Jump composition

### JC-1. Value ordering V ≤ V′ ≤ V″ pointwise (⇒ g″ ≤ g′ ≤ 1); equality only near the absorbing Z = 0 edge
- **Sign source:** Comparison/maximum principle, robust to the nonlinear jump operator (it is monotone in the post-jump value: ∂/∂V^post = J·g > 0). V vs V′: at an interior touching point the extra catch-up channel contributes 0, breakthrough channels coincide, and the flow strictly favors interm (A_g′ = A_d > A_g for Z > 0). V′ vs V″: flow gain A_g″ > A_g′, i_r a lower-bound cost only, interm jump pulls TOWARD V″. Premise of JC-2, JC-3, RD-2, RD-3.
- **Ambiguity:** violations within the 0.075 level slack are the DOCUMENTED under-identification artifact — log as solver-health, don't adjudicate as economics (G4).
- **Protocol:** [PROBE] 4096 shared points × ξ ∈ {0.05, 0.1, 148.4}, three pre-damage nets (and three post-damage at each λ3); report min(V′−V), min(V″−V′) + violation locations (expect near Z ≈ 0). Fatal only beyond slack; optionally FD-leveled post-damage side.
- *Merged from:* jump `value_ordering_tech_tree`.

### JC-2. Distortion asymmetry — pessimism DELAYS the good jump and (modestly) HASTENS the bad one. Tech side (strict, fatal): g′, g″ < 1, hazard multiplier m = (1−π)g′+πg″ < 1 decreasing in aversion, distorted first-tech-jump CDF strictly below baseline at every horizon. Damage side (per CR-2): gℓ increasing in ℓ, g^L > 1; mean ḡ ≥ 1 weakly and MODEST, increasing in aversion; g¹−1 recorded, not gated
- **Sign source:** Tech: both realizations raise value (JC-1) ⇒ surpluses individually signed ⇒ g = e^{−S/ξ} < 1 — no Jensen gap needed. Damage: gℓ = e^{(V−Vℓ)/ξ}; high-ℓ Vℓ < V (extra λ3 convexity) ⇒ g^L > 1; ḡ > 1 needs mean-ℓ(Vℓ−V) ≤ 0 (FK-net-bad), strict by Jensen convexity given spread — but ℓ=1 reveals the benign model AND extinguishes the hazard, so g¹ < 1 admissible (anchor ω¹ = 0.10 < 1/L consistent). A wiring sign error flips tech and damage sides JOINTLY — the asymmetry pair is the diagnostic. Equilibrium feedback (averse planner raises i_r ⇒ R-path up vs m < 1 down): paper's resolution = distortion dominates; check m < 1 POINTWISE so an R_t artifact can't fake a pass.
- **Ambiguity:** damage side per CR-2 — fatal only for gℓ non-increasing, g^L ≤ 1, ḡ significantly < 1, or a LARGE damage distortion either way ("almost no adjustment" is the paper's language). Resolve ḡ's condition by measuring mean-ℓ(Vℓ−V) at the entry slice along the path.
- **Protocol:** [PROBE] g′, g″, gℓ pointwise on the 60y deterministic path, ξ ∈ {0.05, 0.1, 148.4} (G1, G2; FD-leveled post-damage Vℓ from `fd_pdpt_v5_stable_lam3_*` to suppress level noise; read monotonicity-in-1/ξ rather than ḡ's exact level). [SIM] density pipeline: tech CDF strictly below baseline, gap growing 0.1 → 0.05, converging at ξ=∞ (fatal); damage CDF weakly above and CLOSE (modest), sign reported at years 20/40/60. J_n = 0 until Y crosses 1.5 (~year 25) — damage readouts bind only on the later path. Consistency: the averse arms must ALSO show higher i_r (XE-1) — delay + lower i_r = full-channel resolution wrong.
- *Merged from:* foc `damage_jump_accel` + `tech_jump_delay`, costates `distortion_orientation` (jump legs), robustness `damage_density_early` + `tech_density_late`, jump `g_asymmetry_tech_down_damage_up`.

### JC-3. Breakthrough composition: π̃ = πg″/(πg″+(1−π)g′) < π wherever V″ > V′; log-odds shifted by EXACTLY −(V″−V′)/ξ; dπ̃/dξ > 0 with π̃ → π as ξ → ∞; drastic collapse at solved aversions (paper table: 0.0369 / 0.0005 / 0.0000 at ξ = ∞ / 0.1 / 0.05)
- **Sign source:** Intensity algebra, closed form: the pre-jump V CANCELS in the ratio ⇒ π̃/(1−π̃) = [π/(1−π)]e^{−(V″−V′)/ξ}; V″ > V′ (JC-1: higher productivity now, absorbing, no second wait) ⇒ strictly below π. Exponent shrinks in ξ ⇒ increasing to π. Depends on ONE cross-net difference only. Back-out: 0.0005 at ξ=0.1 ⇒ V″−V′ ≈ 0.43 — the paper's numbers imply a LARGE gap; the test has teeth.
- **Ambiguity:** monotonicity through the ENDOGENOUS (V″−V′)(ξ) channel is second-order; paper's 3-point table is monotone ⇒ treat non-monotonicity as solver defect. Per CR-8: neutral limit gated at π ± 10%; at ξ=0.05 test the LOG-ODDS identity against the measured V″−V′ (the printed 0.0000 makes ratio tests meaningless). G1: ξ ≥ 0.05; degenerate in the OneJump π=1 runtime.
- **Protocol:** [PROBE] zero retraining: v_nn of PreDamageIntermTech + PreDamagePostTech along the year-40 deterministic path (matching the table's construction); require π̃(0.05) < π̃(0.1) < π̃(∞) ≈ π, order-of-magnitude match to the table at ξ=0.1, log-odds consistency at 0.05. Any π̃ ≥ π or ξ-non-monotonicity = fatal (instant sign-wiring detector).
- *Merged from:* foc `breakthrough_tilt`, costates `breakthrough_composition_xi`, robustness `breakthrough_share_downweight`, jump `pitilde_breakthrough_lt_pi_incr_xi`.

### JC-4. λ3-belief block: V^ℓ strictly decreasing in λ3; distorted weights ω^ℓ strictly increasing in ℓ; EXACT identity log(ω_L/ω_1) = (V¹−V^L)/ξ (anchor: e^{0.054/0.05} = 2.94 ≈ 0.29/0.10); tilt decreasing in ξ, → uniform 1/L at neutrality
- **Sign source:** Envelope (λ3 is a flow parameter): dV^ℓ/dλ3 = −Ẽ∫δe^{−δt}(Y_t−ŷ)²/2 dt < 0 (post-damage paths at Y ≥ ŷ with E > 0). All L channels share intensity J_n/L ⇒ ω^ℓ ∝ g^ℓ is a softmax of −V^ℓ/ξ; pre-jump V cancels ⇒ log-weights LINEAR in V^ℓ with slope −1/ξ. LEVEL-IMMUNE: V^ℓ are slices of one λ3-pseudo-state net — cross-net constants cannot fake it. Tilt-in-ξ by inspection of the exponent.
- **Ambiguity:** none for these legs (the quantitative identity, not mere monotonicity, is the gate — it catches g-exponent magnitude errors). G1 below ξ = 0.05. (1/ξ-LINEARITY across ξ — a joint test — is JC-5; A_g″ cross is XE-6.)
- **Protocol:** [ANCHORED] five-λ3 FD family `fd_pdpt_v5_stable_lam3_{0000..0333}_xi148.npz` (spread 0.054) + the standing λ3-belief screen ([0.10, 0.16, 0.21, 0.25, 0.29] at ξ=0.05). [PROBE] weights at entry slice (log 880, 0.7, 2.5) at ξ ∈ {0.05, 0.1}: strictly increasing in ℓ; log-ratio identity vs the independently measured spread to ~10%; ≈ [0.2×5] at ξ=148. [SIM] year-40 weights: increasing in ℓ; tilt strictly smaller at ξ=0.1 than 0.05 (the 0.19 anchor is an FD entry-slice reference, NOT a 0.01-tolerance target for the year-40 path object). Pre-damage leg is a grid-perturbation statement (no per-ℓ input in pre-damage nets).
- *Merged from:* foc `dweights_dl`, robustness `V_decreasing_lambda3` + `lambda3_weights_tilt_and_xi`, climate `lambda3_weight_tilt`, costates `lambda3_weights_monotone_x_Agpp` (monotone leg).

---

# TIER 2 — STRONG

## Block A: Preferences / technology

### PT-4. δ-reallocation signature: d(i_g−i_d)/dδ < 0 with i_g↓ AND i_d↑ (tilt toward dirty, not a uniform cut)
- **Sign source:** FOC identity: i_g − i_d = (Γc/δ)(Q_g−Q_d) = (Γc/δ)V_Z exactly; c/δ ≈ invariant (log-AK) ⇒ sign = dV_Z/dδ. Duration argument: V_logK is near duration-neutral (≈1 on BGP) while V_Z's climate component (avoided-emissions annuity) is long-duration ⇒ impatience cuts V_Z proportionally more; green q also embeds the deferred tech-option payoff (e^{−δτ}-discounted) vs front-loaded A_d.
- **Ambiguity:** duration HEURISTIC, not theorem — V_Z also has a front-loaded delta-robust component (A_g″−A_d flow gain), and with δ-normalized weights a PERMANENT flow advantage would be δ-invariant; the sign needs back-loadedness. **Resolved empirically at this calibration**: FD anchor shows both legs. Treat as calibration-verified, not universal.
- **Protocol:** [ANCHORED] fd_dlt arms: i_g 0.172 → 0.154 AND i_d 0.0087 → 0.0123. Candidates: check BOTH component signs (a mis-specified solver cuts both together) + recompute V_Z/V_logK falling by finite-differencing the saved v-grids; [PROBE] FOC-ratio identity pointwise on NN δ-variants. Pre-tech extension: report, don't gate.
- *Merged from:* foc `dgap_ddelta`, costates `dinvspread_ddelta`, robustness `dV_ddelta_growth_envelope` (composition leg).

### PT-5. Scale break by emissions (post-tech ONLY): 1 − V_logK > 0 wherever future emissions positive; gap increasing in η and θ̄, → 0 as Z → 1; EXACT V_logK = 1 at η = 0; ∂i_d/∂logK < 0
- **Sign source:** η = 0 + no-R&D ⇒ logK enters only via additive δlogK ⇒ exact BGP V = logK + W(Z,Y) (sympy-verified). With emissions, ∂E/∂logK = E adds the negative source V_yθ̄E + ς²E²V_yy − (ς²/ξ)V_y²E² to the FK equation for V_logK ⇒ gap = PV of marginal-capital emission damages, linear in θ̄, increasing in η; λ2 convexity makes the gap grow in logK ⇒ i_d decreasing in logK.
- **Ambiguity (CR-10):** R&D-active regimes have the opposing +V_logR·∂ψ_r/∂logK > 0 source — sign numerical there, NOT gated. Mid-box FD is a concave transient (v_logK ≈ 0.37–0.95): test the clean limits and monotonicity, not levels; allow the documented ~2.5% corner overshoot.
- **Protocol:** [FD-NEW] (a) `--set eta=0`: V_logK = 1 ± 0.03 everywhere; (b) `--set eta=0.15 / 0.4` (+ thbar arms): 1−V_logK monotone; (c) vlK(Z=0.98) > vlK(Z=0.5) from saved fields; (d) [PROBE] ∂i_d/∂logK < 0 on post-tech nets at (Z=0.7, Y=1.1) — logK-independent controls are scale-invariant-wrong; (e) informative: pre-tech V_logK vs the post-tech gap to measure the knowledge-scale offset.
- *Merged from:* climate `VlogK_scale_break_emissions`.

## Block B: R&D / innovation

### RD-4. π-convention pair: dV′/dπ > 0 (interm carries π·R) and dV″/dπ ≡ 0 EXACTLY (post-tech has no jump terms)
- **Sign source:** CR-4. Envelope on the interm HJB: minimized tech term = ξ·scale·(πR/ϱ)(1−g″) ⇒ dV′/dπ = FK[ξ·scale·(R/ϱ)(1−g″)] > 0 since g″ < 1. Convention-detecting: the paper's "jump goes to ℓ″ w.p. 1" refers to state-changing jumps under Poisson thinning; the appendix equations + code settle it. Side finding: J_g_prime computed-but-unused in IntermTech files = dead code, not a bug.
- **Ambiguity:** none under the repo convention; a total-intensity-R refactor reverts the sign to exactly 0 — grep first.
- **Protocol:** Static: grep π in `models/*IntermTech.py` jump construction (lines 255/253 confirmed). [RETRAIN] PreDamageIntermTech at π ∈ {0.04, 1.0} (argv[14], same seed/base): V′ and i_r′ increase. Invariance leg: post-tech solves bit-identical across π arms at fixed MODEL_SEED — tolerance is machine-level (π never enters their graphs), far sharper than the 0.2% seed-noise bound; any drift = genuine π leakage.
- *Merged from:* jump `dVinterm_dpi_zero` (corrected).

### RD-5. ψ0 elasticity: dlog i_r/dlog ψ0 > 0 with PARTIAL elasticity exactly 1/(1−ψ1) = 2; executable object = the FOC identity (1−ψ1)Δlog i_r = Δlog ψ0 + Δlog(V_logR·c)
- **Sign source:** Log-differentiate δ/c = ψ0ψ1 i_r^{ψ1−1}e^{ψ1(logK−logR)}V_logR. Feedbacks dampening: i_r crowds c (Δlog c < 0), faster knowledge compresses the surplus (Δlog V_logR < 0 expected) ⇒ total in (0, 2) as prior.
- **Ambiguity:** the (0,2) bracket is not a theorem (dlog V_logR/dlog ψ0 unsigned — arrival-proximity can push up, echoing RD-3). GATE ON THE IDENTITY (all four pieces measured); identity failure = mis-implemented FOC (e.g. wrong ψ1 exponent); out-of-bracket with clean identity = interesting GE, not a bug.
- **Protocol:** [RETRAIN] MODEL_PSI0 (override EXISTS, verified) at 0.8× / 1.2× of 0.10583, shared base; compute both identity sides at t=0 + total elasticity.
- *Merged from:* foc `dir_dpsi0`.

### RD-6. logR FOC-slope decomposition: at ψ1 = 0.5, fixed logK: dlog i_r/dlogR − 2·dlog(V_logR·c)/dlogR = −1 POINTWISE (knowledge congestion −ψ1/(1−ψ1)); +1 in the logK direction
- **Sign source:** Pure FOC algebra in the state direction (the e^{ψ1(logK−logR)} congestion term) — an identity of any FOC-consistent solution; does NOT sign the total slope.
- **Ambiguity:** total dlog i_r/dlogR genuinely ambiguous (arrival nearer vs surplus shrinks) — the IDENTITY RESIDUAL is the check, localizing errors to i_I_nn vs the value costate. Caveat: this repo trains controls with FOC supervision ⇒ identity holding is necessary wiring evidence, NOT independent HJB evidence.
- **Protocol:** [PROBE] no re-solve: finite-difference i_I_nn and v_nn along logR ∈ [1, 6] at the initial (logK, Z, Y) slice, c from all three control nets; residual small vs 1 (report actual training tolerance) + report the total slope. Instantly catches i_r nets ignoring the logR input.
- *Merged from:* foc `dlogir_dlogR`.

### RD-7. Scale break by knowledge: joint (K0, R0) → ×s EXACTLY multiplies the t=0 tech hazard by s (J_g linear in the LEVEL of R, not R/K); median breakthrough strictly earlier; fixed-R0 K0-doubling scales knowledge accumulation by 2^{ψ1} = √2
- **Sign source:** Intensity algebra: total tech intensity = R/ϱ, mechanical, policy-free at t=0. ψ_r depends on logK−logR (unchanged under joint scaling) ⇒ hazard-path ratio starts exactly at s; corrected: does NOT persist exactly (emissions break scale-invariance — the bigger economy warms ∝ K faster).
- **Ambiguity:** fixed-R0 i_r response is policy-feedback ambiguous (resolve by simulating); Y(60) proportionality only loose (±20% or pre-breakthrough only) — earlier breakthrough partially offsets faster warming.
- **Protocol:** [SIM] no retrain: initial conditions {(880, 11.2), (1760, 22.4), (1760, 11.2)}; note logK(1760) ≈ 7.47 exceeds the [4,7] box — prefer the DOWNSCALED arm (440, 5.6), in-box, predicting hazard ratio exactly 0.5. Assert exact t=0 ratio, median-arrival monotone in scale, pre-breakthrough Y-drift ratio ≈ K0 ratio.
- *Merged from:* climate `K0_scale_break_tech_hazard`.

## Block C: Climate / damages

### CD-6. dV/dλ1 envelope IDENTITY: dV/dλ1 = −Ẽ∫e^{−δt}δY_t dt < 0; controls move < 1%
- **Sign source:** λ1 multiplies y in BOTH branches ⇒ ∂flow/∂λ1 = −δY uniformly; λ1 enters no intensity/dynamics (h_y only via V_y, enveloped) ⇒ identity exact. Magnitude = discounted mean temperature path — at the FD ref slice Y=3.0 this is ≈ −[3.0 + PV of remaining warming] ∈ [−4.5, −3.0] (NOT the pre-damage ≈ −3 figure assuming Y0=1.1); compute from the FD-policy path, respecting the Y=4 cap and declining warming as Z → 1.
- **Ambiguity:** none.
- **Protocol:** [FD-NEW] `--set l1=0.01` vs baseline 0.00017675: Δv/Δλ1 vs the path integral within ~10%; simultaneously |Δi_d|, |Δi_g| < 1%. λ2-layer optional per the IMPLEMENTED post-damage formula only (paper's matching constants are typo-flagged); pre-jump λ2 identity needs a pre-damage solve — not in the FD protocol.
- *Merged from:* climate `dV_dlambda1_envelope`.

### CD-7. V_ZY > 0 on the visited box; steepening in λ2
- **Sign source:** Decompose V_Z at fixed logK: only the climate component depends on Y — dE/dZ = −ηA_dK < 0 gives the avoided-emissions annuity +ηA_dK·∫e^{−δt}[(logN)′+hazard]dt, whose Y-derivative loads λ2 + J_n″ terms > 0. Transform-clean: v = V ± logN(Y) shifts by a Z-independent function ⇒ v_ZY = V_ZY, no correction needed.
- **Ambiguity:** policy-feedback in cross-partials not enveloped; visited-box only. The r2-hazard-convexity leg is DROPPED as untestable at reasonable cost (terminal FD has no hazard; would need pre-damage retrain) — future-only.
- **Protocol:** [PROBE] mixed autodiff ∂²v/∂Z∂Y on PreDamagePreTech + PreDamagePostTech over Z ∈ [0.6, 0.9] × Y ∈ [1.1, 2.4]: positive, larger at higher Y. [FD-NEW] `--set l2={0.0044, 0.0066}`: V_Z(Y) profile steepens (finite-difference saved grids).
- *Merged from:* costates `VZY_cross`.

### CD-8. Policy–temperature slope + separability kill: ∂i_d/∂Y < 0 pre-damage, steepening near y̲ = 1.5; with λ2 = λ3 = 0 (terminal, jump-free) the Y-slope collapses to O(λ1θ̄E) ≈ 1e-4
- **Sign source:** Log utility + multiplicative damages ⇒ −δlogN is additively separable, no direct investment distortion. Y reaches controls through exactly three channels: (i) |V_y| increasing via λ2 (λ3 post) in V_yθ̄E; (ii) jump proximity ∂J_n/∂y > 0 × negative bracket, switching on convexly at y̲ (the slope kink is the channel-(ii) diagnostic); (iii) finite-ξ h_y amplification. Killing the curvatures leaves only the λ1 constant.
- **Ambiguity:** kill cleanly executable only in the terminal harness (no hazard there — the kill is λ2 = λ3 = 0); the r1 = 0 pre-damage kill needs a retrain (optional). Steepening = average slopes on [1.5, 2.4] vs [0.8, 1.4], not pointwise convexity.
- **Protocol:** [PROBE] i_d_nn(Y) at (logK 5.5, Z 0.7, logR 2.4, ξ 0.05), Y ∈ [0.8, 2.4] on PreDamagePreTech: decreasing + steeper above 1.5. [FD-NEW] `--lam3 0 --set l2=0`: i_d Y-slope ≈ 0. A solver with a strong i_d–Y gradient under the kill has smuggled damages into marginal utility.
- *Merged from:* climate `policy_Y_slope_separability`.

### CD-9. η-vs-θ̄ wedge: ∂V/∂logη − ∂V/∂logθ̄ ≤ 0 — ≈ 0 (order 1e-2) at ξ=∞, strictly negative at finite ξ, finite-ξ part scaling ς²/ξ
- **Sign source:** Substituted climate block C = V_yθ̄E + ½ς²E²V_yy − (ς²E²/2ξ)V_y²; E ∝ η doubles the E² terms ⇒ wedge = discounted ς²E²(V_yy − V_y²/ξ). η scales drift AND noise loading, θ̄ drift only — the wedge isolates the noise channel; the −V_y²/ξ piece is strictly negative.
- **Ambiguity:** V_yy < 0 expected but not pinned pointwise ⇒ ξ=∞ wedge only "small", not sign-certain; resolution = the integrand probe region by region.
- **Protocol:** [FD-NEW] (a) paired `--set thbar=0.002046` vs `--set eta=0.3201` (+10% each) at ξ=148.4: |ΔV(η) − ΔV(θ̄)| ≤ order 1e-2. (b) ξ=0.05 arm: the FD harness DROPS the worst-case drift (fd_pdpt_v5.py header) — needs the h-extension or NN retrains [RETRAIN, deferred]. (c) [PROBE] integrate ς²E²(V_yy − V_y²/ξ) along simulated paths at ξ ∈ {0.05, 0.1, ∞}: negative, ~2× from 0.1 → 0.05, ~4× under doubled ς in the integrand.
- *Merged from:* climate `eta_vs_thetabar_wedge`.

### CD-10. A_g″ emissions decomposition: post-breakthrough E-path decreasing in A_g″ even though the capital path RISES — in Ė/E = K̇/K − Ż/(1−Z) the composition term must dominate the scale term
- **Sign source:** FOC pair moves oppositely (i_g↑, i_d↓, FD-anchored) ⇒ Ż = Z(1−Z)[φ_g−φ_d+…] gains BOTH legs while K̇/K gains only the Z-weighted average — the difference beats the average when legs move oppositely.
- **Ambiguity:** quantitative race, not sign-forced term by term (wealth channel raises K̇ too). E increasing in A_g″ ⇒ the wealth channel was mis-weighted (i_d not falling).
- **Protocol:** [SIM] executable NOW, no NN: integrate deterministic paths under the existing `fd_agpp0150` / baseline / `fd_agpp0165` policies; log-decompose the E-path into dlogK and dlog(1−Z); composition dominates, E-paths order decreasing in A_g″. Pre-jump i_r layer → RD-1.
- *Merged from:* climate `emissions_decomposition_Agpp`.

## Block D: Robustness-ξ

### RX-4. Each |h_d|, |h_g|, |h_r|, |h_y| decreasing in ξ at fixed state
- **Sign source:** h_j = −(1/ξ)σ_j′V_x: direct term −|σ′V_x|/ξ² strictly negative; flips only if dlog|σ′V_x|/dlogξ > 1 — marginal values move a few percent while 1/ξ moves 3× between 0.1 and 0.05.
- **Ambiguity:** formally ambiguous through the equilibrium V_x(ξ) term; resolution = the probe; report dlog|h_j|/dlogξ ≈ −1 channel-by-channel — any channel with positive slope identifies WHICH marginal value misbehaves.
- **Protocol:** [PROBE] autodiff the four costates along year 0–60 baseline-path states; sweep logξ ∈ [log 0.05, log 148]; monotone decreasing + log-log slope report.
- *Merged from:* robustness `abs_h_monotone_in_xi`.

### RX-5. ς-quadratic tilt: worst-case-minus-baseline temperature drift scales as ς² (dlog tilt/dlog ς = 2.000 at fixed solution, ≈ 2 with re-solve); baseline mean path ς-invariant at ξ=∞; dV/dς ≈ 0 at ξ=∞, < 0 first-order at ξ=0.05
- **Sign source:** Tilt = ςh_yE = ς²E²|V_y|/ξ — exactly quadratic at fixed (V_y, E). Envelope: ∂HJB/∂ς = ςE²(V_yy − V_y²/ξ): at ξ=∞ only the level-neutral risk term (σ-anchor |Δv| ≤ 0.014); at 0.05 the robust term dominates. ς is a pure-risk parameter activated almost entirely by robustness — unlike θ̄.
- **Ambiguity:** re-solve exponent near 2, not exact. **Slope ≈ 1 is the diagnostic failure** (h_y implemented linear-in-ς = one missing ς factor); the exact-2 layer must FIX the solution (vary ς only in the simulator).
- **Protocol:** [SIM] layer (a): hold nets fixed, scale ς in SimulationStochasticJumps at {0.5, 1, 2}×(1.2·1.86/1000); regress log(year-60 distorted-minus-baseline Y gap) on logς: slope 2.000. [FD-NEW] ξ=148.4: `--set vars=` ×{0.5, 2}: distorted gap ≡ 0, |Δv| ≤ ~0.02. Layer (b) GE at ξ=0.05: needs the FD h-extension or retrains [RETRAIN, deferred], slope ≈ 2 ± 15%.
- *Merged from:* climate `varsigma_quadratic_tilt`.

## Block E: Jump composition

### JC-5. λ3 log-odds ≈ linear in 1/ξ: log(w̃_{ℓ+1}/w̃_ℓ) = (V^ℓ(ξ) − V^{ℓ+1}(ξ))/ξ > 0 per adjacent pair; exactly linear ONLY if the gaps are ξ-invariant (they are not — each post-damage solve is robust at its own ξ)
- **Sign source:** Softmax algebra (pre-jump V cancels; common intensity) + FD-verified gap ordering. Linearity broken only by the ξ-drift of the GAPS — a second-order difference of envelope terms (each dV^ℓ/dξ ≥ 0), expected small.
- **Ambiguity:** halve-ξ-doubles-log-odds is a JOINT test of (i) the softmax formula, (ii) gap ξ-invariance, (iii) cross-λ3 level consistency; deviations don't self-identify. Disambiguate by measuring FD gaps at each ξ separately; residual deviation after gap-correction = the known level defect ⇒ doubles as solver-health.
- **Protocol:** [PROBE] w̃_ℓ at the entry slice at ξ = 0.1, 0.05; regress adjacent log-odds on 1/ξ. [FD-NEW] rerun the five-λ3 family at `--xi 0.1` and `--xi 0.05` (flag: fd_pdpt_v5 drops the worst-case drift h — penalty-consistent only after the h-extension; treat pre-extension runs as approximations). Positivity of every adjacent log-odds = fatal direction (subsumed by JC-4); slope agreement within measured gap-drift + level tolerance = this check.
- *Merged from:* jump `lambda3_logodds_linear_in_invxi`.

### JC-6. dV/dr1 < 0 pre-damage (conditional on the FK-net-bad jump, ḡ > 1 on the visited path); |dV/dr1| localized to Y ∈ [1.5, 2.5] path segments, ≈ 0 at low-Y states; magnitude MODEST
- **Sign source:** J_n linear in r1; envelope over controls and minimizing g's: dV/dr1 = −Ẽ∫e^{−δt}ξ(J_n/r1)(ḡ−1)dt — negative wherever J_n > 0 AND ḡ > 1, zero below y̲.
- **Ambiguity:** inherits JC-2's net-bad condition (flips only if mean-ℓ(Vℓ−V) > 0 — learning-option dominance; five-λ3 anchor + paper narrative say no). "Almost no adjustment" ⇒ small negative is the PREDICTION, not a red flag.
- **Protocol (original was vacuous — r1 is FD-inert in the terminal harness, CR-11):** (i) [PROBE/SIM] FK-integrand accumulation of e^{−δt}ξ(J_n/r1)(ḡ−1) along simulated 60y paths (FD-leveled post-damage Vℓ) — sign + Y-localization read off directly, executable NOW; (ii) [RETRAIN] add MODEL_R1 to `_ENVIRONMENT_OVERRIDES` (one line, MODEL_SIGMA_D pattern — currently absent, verified), PreDamage* chains at r1 ∈ {0.75, 1.5, 3.0}, frozen post-damage nets, common base: v ordering (level caveat) + level-free policy shifts.
- *Merged from:* jump `dV_dr1_negative`, climate `r1_localization_zero_mass` (envelope leg).

### JC-7. r2 timing SHAPE: early first-damage-jump density (Y near y̲) unchanged (dJ/dr2 vanishes quadratically at y̲), mid-horizon density up, late tail down, median arrival decreasing in r2 — distinguishes r2 from a uniform r1 rescale
- **Sign source:** Intensity algebra: dJ_n/dr2 = r1e^{(r2/2)(y−y̲)²}(y−y̲)²/2 = O((y−y̲)²) at threshold, growing in y; survival unperturbed at crossing, falls faster later ⇒ mass shifts mid at the tail's expense. Exact for the frozen-policy arm.
- **Ambiguity:** re-optimization offset — higher r2 lowers i_d, slows Y, DELAYS crossing, and even the "invariant early window" shifts via continuation-value policy responses. Run the invariance readout on the FROZEN arm (exact); the frozen-vs-reoptimized gap QUANTIFIES the behavioral delay. Mechanical channel expected to dominate (η anchor: big policy moves, tiny v moves).
- **Protocol:** MODEL_R2 does NOT exist (verified) — add it (one line). [SIM] frozen arm: 10k paths, baseline nets, r2 ∈ {0.18, 0.36, 0.72} overridden at sim time (J recomputed at simulation, no retrain): early-window (~15y post-crossing) invariance, mid-gain, tail-loss, median ordering. [RETRAIN] re-optimized arm: PreDamage* per r2; report the median gap.
- *Merged from:* jump `dr2_timing_shape`.

### JC-8. y̲ statics: dV/dy̲ > 0 (conditional on net-bad); SCC ratio −V_Y/V_logK at (Y0 = 1.1) decreasing in y̲; first-damage-jump density shifts right
- **Sign source:** dJ_n/dy̲ ≤ 0 pointwise + support shrinkage ⇒ envelope dV/dy̲ = −FK[ξ(∂J_n/∂y̲)(ḡ−1)] > 0 given ḡ > 1. V_Y below the threshold is pure discounted anticipation; V_logK production-dominated ⇒ ratio falls (two-derivative claim, numerical confirmation needed).
- **Ambiguity:** (i) net-bad conditionality as JC-6; (ii) V_logK quasi-invariance — verify, don't assume; (iii) **protocol-bug fix**: vary y̲ ALONE with y_up = 2.5 FIXED (CR-11 — moving y_up confounds the post-damage damage function and the pinned entry slice with hazard timing; the window-width change IS part of the y̲ static).
- **Protocol:** [RETRAIN] add MODEL_YLOWER; PreDamage* chains at y̲ ∈ {1.25, 1.5, 1.75}, common base (no pre-damage FD exists — the original "FD variant" was not executable). Read: v ordering (level caveat); [PROBE] autodiff −V_Y/V_logK at (log 880, 0.7, 1.1, log 11.2) decreasing in y̲; [SIM] frozen-policy densities (y̲ override at sim time) shifting right.
- *Merged from:* jump `dylower_fear_recession`.

## Block F: Cross-effects

### XE-1. i_r × aversion: theoretically AMBIGUOUS; the paper's DELIVERED resolution is UP — RD(0.05) > RD(0.1) > RD(∞) with ≈ +75/+50 bp gaps; the tech-only channel ALONE goes DOWN (mechanism experiment)
- **Sign source:** Down-force: the certainty-equivalent per arrival ξ(1−e^{−S/ξ}) is increasing in ξ (1 − e^{−u}(1+u) > 0) ⇒ aversion shrinks the option value of knowledge. Up-forces: worst-case damage tilt (JC-4) + accelerated distorted damage jump (JC-2) widen the green-escape surplus V″−V, raising V_logR; the i_d cut frees resources (C/K up). No algebraic ranking. **CR-5**: final text (L1157/792) = UP; commented L1188 ("significantly lower R&D", i_d↓/i_g↑ instead) = the tech-only channel / earlier vintage — the sign flipped between vintages, proving the ambiguity is real.
- **Ambiguity:** genuinely two-forced; delivered resolution is calibration-specific. A solution ordering RD oppositely contradicts the published figures — flag loudly as paper-inconsistency, not economic impossibility.
- **Protocol:** (1) [SIM] full-channel gate: `RD.txt` (G5) across logξ slices — monotone, averse highest, no crossing, tex-anchored ≈ 75/50 bp magnitudes. (2) [RETRAIN] mechanism gate: tech-only variant (hard-set damage g^ℓ := 1 in `models/PreDamage*.py` — code edit + retrain from the shared base) must FLIP the ordering; passing (1) but failing (2) = right answer by accident. (3) [PROBE] attribution: decompose V_logR into arrival piece ξ[(1−π)(1−g′)+π(1−g″)]R/ϱ vs continuation piece at years 0/40 per ξ — records which channel wins and localizes failures.
- *Merged from:* foc `dir_daversion_channels`, costates + robustness `ir_vs_xi_ambiguous`.

### XE-2. π × ξ: sensitivity to π INCREASES in ξ and VANISHES under deep aversion — D(ξ) = V(π↑) − V(π baseline) > 0, increasing, with D(0.05)/D(148) ≪ 1; same pattern for the i_r response
- **Sign source:** Direct channel ξ[e^{−b/ξ} − e^{−(a+b)/ξ}] (b = V′−V, a = V″−V′ > 0) → 0 as ξ → 0, → a as ξ → ∞; the indirect channel (through dV′/dπ, CR-4) obeys the SAME limits. Economic content: the deep pessimist prices the breakthrough at ~0 (table: 0.0005 at ξ=0.1) ⇒ π is economically dead at low ξ.
- **Ambiguity:** monotonicity BETWEEN the limits not proven pointwise — verify on the grid. **Protocol threat**: D(0.05) predicted ≈ 0 < level slack 0.075 ⇒ V-readout noise-dominated at small ξ; D(0.05) below noise counts as CONSISTENT-with-vanishing, NOT a violation. Level-free i_r is the primary small-ξ readout.
- **Protocol:** [RETRAIN] V(π = 0.08) vs V(0.04) per ξ ∈ {0.05, 0.1, 148.4} (argv[14]; common base, identical schedule ⇒ level drift common-mode); D > 0, increasing; i_r(0.08) − i_r(0.04) at t=0 same pattern at all ξ. **Deep-aversion arms responding MORE to π than neutral = robustness channel wired backwards, fatal.**
- *Merged from:* jump `cross_pi_xi_sensitivity_vanishes`.

### XE-3. r1 × aversion amplification: direct cross ∂²V/(∂r1 ∂(1/ξ)) < 0 UNCONDITIONALLY (per-intensity jump cost deepens in 1/ξ: φ_ε = [e^{−u}(1+u)−1]/ε² ≤ 0 for ALL real u); total DiD expected negative but NOT unambiguous
- **Sign source:** Pointwise partial sign-definite regardless of individual gap signs ((1+u)e^{−u} ≤ 1). The original "NO ambiguity" overclaimed only about the TOTAL.
- **Ambiguity:** two ε-feedbacks: (i) aversion lowers i_d ⇒ slower Y ⇒ LESS hazard-region occupancy (offsetting); (ii) gaps Δ_ℓ move with ε. Expected: direct dominates (occupancy shifts over 60y modest per the η anchor). Isolate by computing the FK-integrand DiD twice: frozen common baseline path (direct only) vs ε-specific re-optimized paths (total).
- **Protocol:** [RETRAIN] 2×2 {r1 = 1.5, 3.0} × {ξ = 0.1, 0.05} via MODEL_R1 (after the one-line addition), common base: DiD [V(3, .05) − V(1.5, .05)] − [V(3, .1) − V(1.5, .1)] < 0 at the reference state; [PROBE] frozen-vs-reoptimized FK-integrand DiD from checkpoints separates channels.
- *Merged from:* jump `cross_r1_aversion_amplification`.

### XE-4. A_g″ × ξ: dlog g″/dA_g″ = −(1/ξ)d(V″−V)/dA_g″ < 0 with magnitude scaling 1/ξ — a bigger tech shock is MORE heavily down-weighted by the averse planner
- **Sign source:** Envelope: d(V″−V)/dA_g″ > 0 (V gains only the arrival-weighted π-scaled echo) ⇒ log g″ falls, amplified by 1/ξ. Phrase in LOG g″ (Δlog g″ doubles from ξ 0.1 → 0.05); Δg″ itself does not double (g″ near floor: ΔV″ ≈ 0.46 ⇒ Δlog g″ ≈ −9 at 0.05, consistent with the table's 0.0000).
- **Ambiguity:** none in sign; executability per CR-11.
- **Protocol:** (a) [ANCHORED] cheap bound: since |dV/dA_g″| ≪ dV″/dA_g″, use the FD ΔV″ (fd_agpp arms) alone to bound Δlog g″ ≈ −ΔV″/ξ and check the NN's g″ response is of that order; (b) [RETRAIN] full: A_g″ params-edit chains, g″ at year-40 states for ξ ∈ {0.1, 0.05}: Δlog g″ ≈ 2× larger at 0.05. G1 applies.
- *Merged from:* robustness `Agpp_statics_and_g_cross` (cross leg).

### XE-5. Warming × knowledge: V_logR,Y > 0 — the tech surplus S = V″ − V is increasing in Y (probed as dS/dY > 0)
- **Sign source:** J affine in R ⇒ sign(V_logR,Y) tracks sign(dS/dY) = V″_Y − V_Y; the pre-tech economy stays dirty longer ⇒ larger discounted damage-slope + hazard exposure per initial degree ⇒ |V_Y| > |V″_Y|. **Key measurement property: differencing in Y kills the additive cross-net level mismatch — this cross-regime probe is level-immune**, unlike π̃/g objects.
- **Ambiguity:** the |V_Y| ordering is path-dominance heuristic (curvature could reverse locally at high Y where both are hazard-dominated); no FD resolution exists (needs logR). Resolution = the direct S(Y) probe; the i_r-path corollary is explicitly NOT the check (MU moves too).
- **Protocol:** [PROBE] S(Y) = v_PreDamagePostTech − v_PreDamagePreTech on Y ∈ [1.1, 2.4] at path-median (logK, Z, logR) per logξ slice (transform terms cancel within the same damage state): increasing. Complement: mixed autodiff ∂²v/∂logR∂Y. Discriminating power: V_logR carries the largest cross-run divergence (RUNA/RUNB finding).
- *Merged from:* costates `VlRY_cross`.

---

# TIER 3 — INFORMATIVE

### PT-6. σ near-neutrality MAGNITUDE bound: |ΔV| ≤ O(σ²/δ) ≈ 1e-2 under σ_d, σ_g ×2 or ÷2 (at large ξ); tolerance widens ~1/ξ at small ξ (robust drag σ²/2ξ)
- **Sign source:** every σ-term is O(σ²) ≈ 1e-4 in flow units, capitalized by 1/δ. Sign genuinely near-degenerate (Itô drag vs second-derivative/Z-diffusion terms net within solver precision) — the CHECK is the bound, never the sign. A solution moving V by O(0.1) under σ×2 is wrong.
- **Protocol:** [ANCHORED] `fd_sig{0005,0020}.npz`: |Δv| ≤ 0.014 at ξ=148.4. [RETRAIN cheap] NN leg fully executable — MODEL_SIGMA_D/G EXIST: short warm-started terminal retrains, |ΔV| ≤ 5e-2; repeat at ξ = 0.05 with widened tolerance to probe the drag interaction.
- *Merged from:* costates `sigma_neutrality`.

### RD-8. ϱ-timing feedback: median first-tech-jump delay from raising ϱ is AMPLIFIED by re-optimization (T_med re-optimized > frozen), and frozen-arm dlogT/dlogϱ < 1 (R_s rising ⇒ F convex)
- **Sign source:** threshold algebra ∫R_s ds = ϱ log2/scale; the frozen-vs-reoptimized gap's sign IS the sign of the i_r response — a behavioral-feedback detector.
- **Ambiguity:** gap second-order small; training noise can contaminate — warm-start the scale-0.5 arm from baseline nets with a short gentle schedule; compute T_med on DETERMINISTIC paths under the BASELINE measure (specify the measure — density-pipeline defaults can be distorted).
- **Protocol:** [SIM + RETRAIN] A/B at scale 0.5: retrained arm vs baseline nets with sim-time-only scale override.
- *Merged from:* jump `varrho_timing_feedback_amplification`.

### CD-11. i_g vs θ̄: AMBIGUOUS — substitution (V_Z↑, emission cost loads on the (1−Z) share only) vs income (V_logK↓)
- **Sign source:** FOC δ/c = φ_g′(i_g)[V_logK + (1−Z)V_Z] with offsetting costate moves. Log-utility income effects on investment RATES are weak, and the η anchor (resources reallocated, not destroyed: i_d collapse, v flat) suggests substitution dominates ⇒ expected i_g↑, shrinking as Z → 1.
- **Resolution experiment:** [FD-NEW] the CD-1 θ̄ grid, reading i_g at Z ∈ {0.5, 0.7, 0.9}; plus a genuinely NEW readout of Δi_g from the EXISTING `fd_eta{0250,0330}.npz` (the provenance tables report only i_d). No retrain.
- *Merged from:* climate `ig_thetabar_ambiguous`.

### RX-6. Worst-case vs baseline expected emissions under the SAME robust policy: sign AMBIGUOUS, gap predictably TINY (drift shifts σ_jh_j ≈ q·σ²/ξ ≈ 3e-4/yr at σ=0.01, ξ=0.05)
- **Sign source:** h_d, h_g < 0 shrink K (E↓); composition dZ-shift = Z(1−Z)[σ_gh_g − σ_dh_d] is state-dependent (can push E↑). Distinct from the SIGNED "averse-POLICY E < neutral-policy E" (CD-4).
- **Resolution experiment:** [SIM + small script extension] twin simulation at ξ=0.05, identical policy nets: baseline drifts vs drifts + σh (the h-shifted capital drift is NOT a stock switch in SimulationStochasticJumps — budget the extension). Gate: |gap| consistent with the ~3e-4/yr bound — a LARGE gap either way = h or q bug; report the signed gap + composition term.
- *Merged from:* robustness `worstcase_emissions_vs_baseline`.

### JC-9. Race composition: P(first jump is tech | a jump by year 60) decreasing in r1 — robust under the DISTORTED measure (mechanical scaling + both g-channels reinforce); BASELINE-measure sign ambiguous in principle
- **Sign source:** intensity algebra holding paths fixed; offsets = endogenous (i_r, i_d) responses. Premise updated per CR-5: the aversion→proactive-R&D offset is real but MODERATE (delivered UP), so mechanical dominance remains the expected resolution.
- **Resolution experiment:** [SIM] frozen arm (baseline nets + MODEL_R1 at sim time, after the one-line addition) isolates the mechanical sign (must be negative); [RETRAIN] re-optimized arm per r1 reports the behavioral offset. 10k paths, record which jump fires first, tabulate under BOTH measures — the distorted-measure sign must be negative or something is miswired.
- *Merged from:* jump `race_composition_in_r1`.

### XE-6. A_g″ × damage concern (author-intended, tex ~L1190-92): SMALLER tech shock ⇒ BIGGER distorted-λ3 tilt (year-40 ω⁵−ω¹ larger at A_g″ = 0.150); tilt-difference itself shrinking in ξ
- **Sign source:** PATH-mediated, genuinely two-forced: (widens) the green-escape option is worth more in high-λ3 states — cutting A_g″ lowers V^(L) more than V^(1), and lower A_g″ slows greening, keeping Y higher longer, multiplying the differential (Y−ŷ)² exposure; (narrows) poorer severe-damage planners buy less escape. Common level shifts cancel in the ratio — only the SPREAD survives. The post-damage V^ℓ contain their own tech option ⇒ partially offsetting entries. Author asserts the widening force wins; not resolvable by algebra.
- **Resolution experiment:** [FD-NEW] 15-solve cross `--lam3 {0, 1/12, 1/6, 1/4, 1/3}` × `--set A_gpp={0.150, 0.1567, 0.165}`, weights at the entry slice at ξ = 0.05, 0.1 (post-damage POST-tech branch = a direction-valid proxy for the post-damage PRE-tech V^ℓ the planner actually weights). [RETRAIN] escalation if the FD proxy is flat: pre-damage chains per A_g″ (check first whether the PostTech nets SAMPLED A_gpp as an input — if so, that side is probe-only), year-40 histograms. Record the result against the author's prior EITHER WAY; disagreement = flag for review, not auto-fail.
- *Merged from:* robustness `smaller_techshock_bigger_damage_tilt`, jump `dAgpp_cross_damage_concern`, costates `lambda3_weights_monotone_x_Agpp` (cross leg).

---

# TEST-EXECUTION PLAN

## Bucket A — Already validated by existing artifacts (regrade candidates against them; no new reference compute)
| Artifact | Checks anchored |
|---|---|
| `outputs_variants/fd_dlt{0080,0090,0110,0125}.npz` | PT-1 (sign + slope −320), PT-4 (both legs) |
| `outputs_variants/fd_agpp{0150,0165}.npz` | PT-2 (all three legs), XE-4 bound (ΔV″ side), CD-10 policies |
| `outputs_variants/fd_eta{0250,0330}.npz` | CD-4 η legs (i_d −63%, v flat) + NEW same-file readouts: V_Z (CD-4), i_g (CD-11) |
| `outputs_variants/fd_sig{0005,0020}.npz` | PT-6 bound at ξ=148.4 |
| `post_damage_post_tech/outputs/fd_pdpt_v5_stable_lam3_{0000..0333}_xi148.npz` | JC-4 (V^ℓ decreasing, spread 0.054), FD-leveled Vℓ for JC-2/JC-6 |
| Existing A_d FD sweep (ad-sensitivity study) | CD-2 sign asymmetry (i_d↑ in A_d, crossing ≈ 0.073) |
| Standing λ3-belief screen (report discipline) | JC-4 monotone-weights leg at ξ ∈ {0.05, 0.1} |

These fix the reference VALUES; every candidate solution must still be re-graded (rerun the variant per candidate, or probe candidate nets).

## Bucket B — NEW cheap FD variant solves
Harness (verified): `python benchmarks/two_capital_selection_criterion/run_solver.py --method pibys [--lam3 L] [--xi X] --set KEY=VAL --save ... --grade`. Valid `--set` keys (verified in `fd_pdpt_v5.P`): `delta, A_d, A_gpp, a_d, a_g, G_d, G_g, t_d, t_g, s_d, s_g, thbar, eta, vars, l1, l2, y_up`. Gotcha: deep-δ / small-ξ arms need the jump-exp clip ±35 (fd_dlt0080 NaN history). Submit via sbatch, not the login node.
1. `--set thbar=0.0015` and `--set thbar=0.0022` → CD-1 (v, i_d ↓); read i_g at Z ∈ {0.5, 0.7, 0.9} → CD-11.
2. `--set A_d=0.11` and `--set A_d=0.15` (A_gpp untouched) → CD-2 elasticity layer.
3. `--set l2=0.0035` and `--set l2=0.0055` (opt `l2=0.0066`) → CD-3 λ2-leg, CD-7 steepening.
4. `--set l1=0.01` → CD-6 envelope identity (±10% vs path integral).
5. `--lam3 0 --set l2=0` → CD-8 separability kill (i_d Y-slope ≈ 0).
6. `--set eta=0` (V_logK = 1 ± 0.03), `--set eta=0.15`, `--set eta=0.4` → PT-5.
7. Paired `--set thbar=0.002046` vs `--set eta=0.3201` (+10%) at `--xi 148.4` → CD-9(a).
8. `--set vars=0.001116` and `--set vars=0.004464` (×0.5/×2) at ξ=148.4 → RX-5 neutral-invariance leg.
9. Five-λ3 grid at `--xi 0.1` and `--xi 0.05` → JC-5 per-ξ gaps (flag: harness drops worst-case drift h — approximation until the h-extension lands).
10. 15-solve cross `--lam3 {0, 0.0833, 0.1667, 0.25, 0.3333}` × `--set A_gpp={0.150, 0.1567, 0.165}` → XE-6 FD side.

## Bucket C — Simulation readouts (existing pipelines; no retrain unless noted)
- `SimulationDeterministic.sh` / `Simulation.ipynb` (G5 dictionary!): `RD.txt` across logξ → XE-1 gate; `I_d.txt` levels + E paths → CD-4 aversion leg; realized growth g_u → PT-1 magnitude.
- `SimulationStochasticJumps.py` + `stochastic_density_plot_stage.sbatch` at ξ ∈ {0.05, 0.1, 148.4}: tech CDF strictly below baseline / damage CDF weakly above & modest → JC-2; neutral overlay at ξ=148 → RX-2; zero damage mass below Y=1.5 → CD-5; year-40 λ3 weights → JC-4; race tabulation both measures → JC-9.
- Sim-time-only overrides (no retrain): ς rescale → RX-5(a) slope = 2.000; r2 / r1 / y̲ frozen-policy arms (after the one-line `_ENVIRONMENT_OVERRIDES` additions) → JC-7, JC-9, JC-8 densities; initial-condition rescale (use ×0.5 arm to stay in the logK box) → RD-7.
- Small script extensions: σh-shifted drifts twin sim → RX-6; FK-integrand accumulators → JC-6(i), XE-3 direct arm, CD-9(c).
- FD-policy deterministic path integration (no NN at all): emissions decomposition under fd_agpp policies → CD-10; discounted-Y integral → CD-6; per-variant ∫E dt → CD-2 elasticities.

## Bucket D — Trained-net probes (checkpoints + autodiff only; guards G1–G6 bind)
Cheapest and highest bug-yield — run FIRST on any candidate: JC-3 (π̃ + log-odds), JC-4 (identity vs spread), RX-1 (logξ monotonicity + ξ* frontier), RX-3 (h_y block at Y=3), PT-3 (FOC-ratio residual + V_Z scan). Then: JC-1 (ordering, 4096 pts), JC-2 pointwise g's (FD-leveled post side), RX-2 (rates, large-ξ half regression), RX-4 (|h| slopes), CD-3 (V_Y/V_YY/SCC, numerator & denominator separately), CD-7 (∂²v/∂Z∂Y), CD-8(a) (i_d(Y) slope), XE-5 (S(Y), level-immune), RD-6 (logR identity), RD-5 identity pieces, XE-4(a) bound, JC-5 regression, XE-1(3) attribution decomposition. Remember: controls are FOC-supervised, so FOC identities are wiring checks, not independent HJB evidence.

## Bucket E — Variant NN retrains (budget as studies: common warm-start base, fixed MODEL_SEED, `output_<study>_<YYYYMMDD>/` + MANIFEST.txt, sbatch chains post→pre)
1. **A_g″ chains** {0.150, 0.1567, 0.165} — params edit (NOT env-overridable, verified); check first whether PostTech nets sampled A_gpp as an input (then that side is probe-only) → RD-1, CD-10 pre-jump, XE-4(b), XE-6 NN side.
2. **π arms** {0.04, 0.08, 0.5, 1.0} via argv[14] → RD-2, RD-4 (post-tech machine-tolerance invariance), XE-2.
3. **Intensity-scale arms** {0.5, 1, 2, 4} via argv[13] → RD-3, RD-8.
4. **MODEL_PSI0** ∈ {0.8×, 1.2×}·0.10583 (override exists) → RD-5.
5. **Tech-only-uncertainty variant** (hard-set damage g^ℓ := 1 in `models/PreDamage*.py`; code edit) → XE-1 mechanism flip — the highest-value single retrain (adjudicates the flagship ambiguity).
6. **Hazard overrides** — add MODEL_R1, MODEL_R2, MODEL_YLOWER (one line each, MODEL_SIGMA_D pattern): r1 ∈ {0.75, 1.5, 3.0} → JC-6(ii), XE-3 2×2, JC-9; r2 arms → JC-7; y̲ ∈ {1.25, 1.5, 1.75} (y_up pinned 2.5) → JC-8.
7. **δ NN leg** via the `ab_delta_override/` family → PT-1/PT-4 NN legs.
8. **σ arms** ×{0.5, 2} short warm-started terminal retrains (overrides exist) at the ξ=0.05 slice → PT-6 drag interaction.
9. **Deferred** (declare, don't pretend they're probes): λ2×ξ cross (CD-3), robust-FD h-extension enabling CD-9(b)/RX-5(b)/JC-5-exact, full η×ξ cross (CD-4).

**Execution priority:** A (free regrades) → D (cheap, instant wiring-bug detection) → B (hours of FD) → C (sims) → E (retrains, led by #5 tech-only and #2 π arms, which gate paper-critical claims).
