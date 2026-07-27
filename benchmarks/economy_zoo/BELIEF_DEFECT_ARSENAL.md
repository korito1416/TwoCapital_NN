# BELIEF-DEFECT ARSENAL — Reference Solution (OneJump π=1, 4 regimes)

Durable debugging memory (2026-07-20). Built by the belief-defect-investigation workflow (4 channels
characterized, each adversarially verified — all defect/health verdicts CONFIRMED; skeptics corrected
several overstated magnitudes, folded in below). Companion to `REFERENCE_SCORECARD.md`.

Reference = inherited warm-start fossil at `output_001/OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000/`.
Sim beliefs pre-written in `output_xiprofile_ref_20260714/SimulationDeterministic/SimulationOutputs_ξ_<xi>/`.
FD benchmark = `benchmarks/post_damage_post_tech/outputs/fd_pdpt_v5_stable_lam3_{0000,0083,0167,0250,0333}_xi148.npz`.

---

## 1. ONE-LINE VERDICT PER CHANNEL (most-severe first)

1. **λ3 damage-curvature belief — FATALLY OVER-TILTED (JC-4).** Confirmed ~7× value-spread defect: net
   V₁−V_L = 0.383 (PostDamagePreTech, the belief-driving regime) / 0.398 (PostDamagePostTech, the FD-having
   sister) vs FD 0.0542 at the in-box anchor (logK=6.78, Z=0.7, Y=2.5). Anchor-to-anchor odds over-tilt at
   ξ=0.05 ≈ **718× (PDPreTech) / 961× (PDPostTech)**. Delivered weights ~[0.0003,0.003,0.026,0.17,0.80] vs
   FD-implied ~[0.10,0.16,0.21,0.25,0.29]. Level-immune, ξ-invariant (spread drifts +2.6% over ξ
   0.05→148.6), exceeds the benchmark-free envelope cap 0.375. **Corrected:** the earlier "928×/1000×"
   headline was apples-to-oranges (delivered green-heavy state odds ÷ anchor FD odds); honest anchor figure
   is 718–961×. Delivered-state (Z=0.872, logK≈7.15) numbers are EXTRAPOLATED beyond box/grid max 7.0 — but
   honest in-box numbers make the defect LARGER, not smaller.

2. **Damage-JUMP belief — MIXED (per-model catastrophic / aggregate accidentally near-correct).** Confirmed.
   Per-model attribution (`lambda3_weights_distorted.txt`) inherits JC-4 → ~2000× odds error at ξ=0.05 (~45×
   at ξ=0.1). Aggregate intensity is only ACCIDENTALLY healthy: ~10× Jensen inflation (J=9.958) cancelled by
   ~10× cross-regime level suppression (level=0.102, d̄=+0.114 at ξ=0.05) → ḡ_end=1.0145. **Corrected:**
   over-tilt is ~22× (range ~18–24×), NOT "22–29×". "The damage timing looks fine" is a coincidence, not a
   validation.

3. **Tech-JUMP belief — HEALTHY (direction+wiring), magnitude NO-GROUND-TRUTH.** Genuine, correctly-signed,
   ξ-invariant value gap: net V_post−V_pre reproduces sim −ξ·ln(g_post) to 5 decimals at all 8 ξ (gap
   0.0967→0.1014, +4.8%, SMALLEST at smallest ξ — opposite of a level-anchor blowup). Direction guaranteed
   by comparison principle (A_g''=0.1567 > A_g=0.1085 + R&D cost removed). Does NOT inherit JC-4 (λ3 never
   enters g_post inputs) and immune to post-tech policy sign defects (reads LEVELS not policies).
   **Corrected:** "ALL delivered tech-density/CDF clean" is too strong — the delivered competing-risk
   density/CDF carry a JC-4-inflated damage-survival leg worth ~3–6% of the 60y tech-CDF gap. The BELIEF
   g_post is clean; the downstream densities are ~94–97% tech-belief. Magnitude 0.10 unverifiable.

4. **Climate-sensitivity belief (h_y/θ over 144 models) — HEALTHY, economically immaterial.** Sign-correct
   (V_y<0, h_y>0 all ξ), ξ-invariant (backed-out V_y drifts 5.15%), level-immune (h_y=f(dv_dY) only).
   Structurally tiny, BENCHMARK-INDEPENDENT: ς²=4.98e-6, delivered θ-shift at ξ=0.05 = 0.43% of θ_bar; even
   7× |V_y| → 3.0%, 20× → 9%. **Corrected:** do NOT say "matches FD within 7%" — that over-credits a
   cross-regime/cross-Y/cross-ξ PROXY (post-damage-post-tech terminal vs the pre-damage-pre-tech V_y probed;
   "entry-slice exact −0.0132 vs −0.0132" is a cross-Y coincidence, same-Y ratio is 1.142). Correct basis
   for "healthy" = the ς²/ξ structural smallness.

---

## 2. THE COMMON ROOT — one disease

**The *belief* problem is essentially ONE defect: the ~7× over-sized post-damage λ3 value-spread (~0.40 vs
FD 0.054, a within-net SLOPE error).**
- **Shared by channels 1 & 2**: λ3 enters both post-damage regimes through the IDENTICAL damage function
  N(y;ℓ); PDPreTech (drives the belief, no FD) and PDPostTech (has FD, 7.3× over-tilted) share it, net
  spreads agree ~4%. It is a SLOPE defect **distinct from the δ-level-anchor artifact**: level-immune
  (within-net difference) and ξ-invariant (2.6% drift), whereas the level-anchor artifact is created by
  small ξ and gave the low-ξ density its ξ*≈0.025 fragility. Channel 2 = channel 1 fed through a
  competing-risk survival + a cross-regime level factor d̄ that roughly cancels the Jensen inflation.
- **Channels 3 & 4 are structurally immune**: they read λ3-agnostic pre-damage value nets (input
  logK,Z,Y,logR,logξ — no λ3 slice). Leakage is bounded and small (climate ~14% top-of-path via
  J_n(y)(V^ℓ−V), J_n≤0.11/yr; tech ~3–6% of CDF gap via the competing-risk damage leg) — downstream-density
  contamination, NOT belief contamination.
- **Second, separate cluster (post-tech POLICIES)** barely touches beliefs: dead-Y input in PreDamagePostTech
  and sign-inverted PostDamagePostTech policies corrupt POLICIES; g_post reads value LEVELS, so the tech
  belief survives (dead-Y inflates the gap ~3% within Y≤2.11, negligible on-path).

---

## 3. DEBUGGING PLAYBOOK (per channel: SYMPTOM / DETECTION / ROOT CAUSE / FIX HANDLE)

### Channel 1 — λ3 damage-curvature belief (JC-4)
- **SYMPTOM:** `lambda3_weights_distorted.txt` collapses onto the severest model (~[0.0003,…,0.80]); odds
  w_L/w_1 ~2740 at ξ=0.05; weights DON'T collapse toward uniform as green share rises.
- **DETECTION:** (a) reconstruct `softmax(−V_PostDamagePreTech(state)/ξ)` = delivered weights to 1e-7 (not
  stale; the (1/L)J_n and V_predamage cancel in normalization → no competing-risk confound); (b) net
  λ3-spread = ξ·log(w_L/w_1) = 0.383; (c) load 5 FD npz, spread at anchor = 0.0542 → 7.07×; (d) **DECISIVE
  Z-collapse:** sweep Z in FD — spread collapses 0.0542→0.0289→0.0100→0.0036 (Z=0.70→0.95) while net stays
  flat 0.383→0.380 (no 1/ξ amplification, no extrapolation, Z≤0.95<0.98, logK in-box); (e) envelope: net
  0.40 > cap 0.375=(1/3)/2·1.5².
- **ROOT CAUSE:** post-damage value net's λ3-dependence is ~7× too steep — FD says ~0.054 and should collapse
  with green share; net learned a fixed, unphysically large spread.
- **FIX HANDLE:** anchor the post-damage regimes' λ3-spread to FD (the **λ3-DIFFERENCE anchor**, validated on
  the terminal regime — holds +0.044 while HJB residual converges 2× control; see [[rival-model-program]]).
  Amplified by 1/ξ ⇒ MUST be pinned at the value level, not the belief level.

### Channel 2 — damage-JUMP belief
- **SYMPTOM:** per-model damage weights catastrophically tilted (inherits ch.1); aggregate ḡ_end≈1.01–1.06
  looks fine; delivered damage CDF rises 0.145→0.409 across ξ.
- **DETECTION:** factorize from delivered artifacts ONLY: ḡ_end = λ_d/J_n(Y_end) from `dmg_jump_intensity.txt`;
  Jensen J=(1/L)exp(−mean ln w)=9.958 from `lambda3_weights_distorted.txt`; level=ḡ/J=0.102,
  d̄=−ξ·ln(level)=+0.114. The ~10× Jensen × ~10× level cancellation is the tell. **Competing-risk check:**
  pure damage-only first-passage CDF is INERT (0.827→0.819) while delivered moves 2.8× — the ξ-motion is a
  tech-delay confound. Code: `SimulationDeterministic.py` `_damage_weights` L471–491, `grouped_first_jump_statistics` L197–283.
- **ROOT CAUSE:** JC-4 (in-net Jensen inflation) × an unbenchmarkable cross-regime level factor d̄ that
  roughly offsets it.
- **FIX HANDLE:** fixing ch.1 fixes the per-model half; the aggregate/level half needs a PreTech FD (does not
  exist) to certify d̄ is real vs a level-anchor artifact.

### Channel 3 — tech-JUMP belief
- **SYMPTOM:** would show as g_post drifting the WRONG way with ξ (blowing up at small ξ), gap sign-flip, or
  not reproducing from the nets.
- **DETECTION:** TF probe both frozen value nets → V_post−V_pre must equal sim −ξ·ln(g_post) (5 dec, all ξ);
  ξ-drift sign: gap SMALLEST at smallest ξ (0.0967→0.1014) = NOT a level-anchor artifact;
  `dist_int=g_post·(R/varrho)` to 5e-9; inputs L441/L445 confirm λ3 absent.
- **ROOT CAUSE (of the small contamination):** delivered density/CDF (`SimulationDeterministic.py` L648–649)
  build a competing-risk survival whose damage leg carries JC-4-inflated ḡ (up to 1.22–1.52). Belief g_post
  itself clean.
- **FIX HANDLE:** none for the belief; downstream density cleans up once ch.1 fixed. Magnitude 0.10
  unverifiable without a pre-damage cross-regime FD.

### Channel 4 — climate-sensitivity belief
- **SYMPTOM:** would show as h_y wrong sign, θ-shift a material fraction of θ_bar, or ξ·h_y not ξ-invariant.
- **DETECTION:** back out V_y = v_transformed_Y − (λ1+λ2·Y+λ3(Y−ŷ)) (`SimulationDeterministic.py` L614 =
  `fd_pdpt_v5.py simulate_v` L124–133); structural bound θ-shift = −V_y·E·ς²/ξ, ς²=4.98e-6 → 0.43% of θ_bar
  at ξ=0.05, 3% even at 7× |V_y|; rigid uniform-1/144 mean-shift (L852, L860).
- **ROOT CAUSE:** none — immaterial by the ς²/ξ argument. The ~14% top-of-path excess over FD(lam3=0) is real
  damage-jump anticipation via J_n(y)(V^ℓ−V), bounded by J_n≤0.11/yr.
- **FIX HANDLE:** none warranted. Do NOT cite the FD proxy as validation — cite the ς² structural smallness.

---

## 4. HONESTY BOUNDARY — what we CAN and CANNOT say to a PI

**CAN say (proven, benchmarked, skeptic-CONFIRMED):**
- The λ3 damage-curvature belief is **over-tilted ~7× at the value-spread level** (net 0.383/0.398 vs FD
  0.0542, in-box anchor) → **~700–960× odds over-tilt at ξ=0.05**. A genuine SLOPE defect, level-immune and
  ξ-invariant — NOT the δ-level-anchor artifact, NOT a low-ξ/extrapolation artifact.
- **Cleanest single number for a PI:** the FD λ3-spread COLLAPSES with green share (0.054→0.004 as Z:0.70→0.95,
  economically obvious via E=ηA_d(1−Z)K) and the net does not — extrapolation-free, amplification-free.
- The per-model damage-JUMP belief is **catastrophically wrong (~2000× odds error at ξ=0.05)**; its aggregate
  intensity is only ACCIDENTALLY near-correct (Jensen×level cancellation).
- Tech-jump and climate beliefs are **direction-correct and free of JC-4/level-anchor artifacts**; climate is
  **economically immaterial at every ξ** by a benchmark-independent ς²/ξ bound.

**CANNOT say (no ground truth — bound only):**
- The EXACT over-tilt multiple for the belief-DRIVING regime (PostDamagePreTech has no FD; benchmarked
  against its post-tech sister, which is itself 7.3× over-tilted, agree ~4%).
- That the odds multiples hold "exactly" down to ξ=0.05 (assumes FD value-spread ξ-invariant; justified,
  never measured at low ξ).
- The tech-jump gap MAGNITUDE (~0.10) is correct — theory limits only BOUND direction. "Healthy" = "passes
  every available check," not "magnitude verified."
- That the damage-jump level factor d̄ (=0.114) is a level-anchor artifact vs a real economic gap — no FD for
  V_PreDamagePreTech.
- Do NOT claim the climate belief "matches FD within 7%" — cross-regime/cross-Y proxy coincidence.

---

## 5. OPEN GAPS (need a new solve to close)

1. **PostDamagePreTech FD** — the regime that ACTUALLY drives the λ3 belief has no ground truth. HIGHEST-value
   new solve: certifies the exact JC-4 multiple + confirms the Z-collapse transfers (should — emissions
   depend on A_d and Z, not A_g).
2. **PreDamagePreTech (+ PreDamagePostTech) FD** — no benchmark for (a) the cross-regime level factor d̄
   (damage-jump), (b) the tech-jump gap magnitude ~0.10, (c) the pre-damage-pre-tech V_y (climate proxy).
3. **Low-ξ FD** — all odds multiples assume ξ-invariance of the FD value-spread down to ξ=0.05.
4. **Extreme-Z FD resolution** — FD spread ~0.0036 at Z=0.95 nears the numerical floor (anchor 0.054 at Z=0.7
   is solid; the 7× defect is safe, the >100× extreme-Z ratio is noisier).
5. **Stochastic-jump sim contamination** — the JC-4 leakage into tech density (3–6%) and climate V_y (14%
   top-of-path) is measured only on the deterministic path (Y≤2.11); a stochastic sim entering post-tech at
   high Y would compound both, unquantified.
6. **Post-tech policy corruption → belief** — g_post (levels) confirmed immune, but the pre-damage HJB consumes
   post-tech VALUES via the tech-jump term; whether the post-tech-cluster value corruption leaks into
   V_PreDamagePreTech was not directly checked (pre-tech v_nn is a distinct network, so expected small).
