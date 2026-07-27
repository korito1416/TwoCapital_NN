# Full-model test plan — joint co-anchored level-pinning across the jump tree

**Goal:** turn the single-regime level-pin (costate/EGM ~150×) and the frozen-surrogate cross-regime
POC (ξ=0.01 admissibility 2%→100%) into a **real multi-regime solve** with NO frozen surrogate, and
**push ξ\* from 0.025 down toward 0.005**. Sandbox; Joe & Mike before any production change.

> **Grid: CONFIRMED = 2 damage × 3 tech = 6 regimes** (the canonical full model, WITH the
> intermediate-tech column — PreDamage/PostDamage × Pre/Interm/PostTech). Damage = {pre, post}
> (post carries the λ3 curvature pseudo-state); tech = {pre, interm (catch-up A_g'=A_d),
> post (breakthrough)}. The one-jump 4-regime (no interm) is the S1/S2 stepping stone.
>
> **Convergence:** the prototypes ran 3000 steps from scratch (loss_v ~2e-3, under-converged — fine
> for the FREE-vs-CO-ANCHORED *direction*). For the real stages, train to convergence
> (**target loss_v ~1e-3**, ~15k–30k+ steps from scratch, or derivative-supervised warm-start) and
> confirm the level-pin / admissibility SURVIVE convergence. All on the torch port.

---

## 1. Regime grid & jump tree (one-jump, 2 damage × 2 tech = 4)

```
                T0 = pre-tech                 T1 = post-tech (breakthrough)
 D0 = pre-dmg   PreDamagePreTech  --tech-->   PreDamagePostTech
                     |   \                          |
                  damage  \  (both jumps)        damage
                     v     \                        v
 D1 = post-dmg  PostDamagePreTech --tech-->   PostDamagePostTech   (terminal, absorbing)
```

- **Tech jump** T0→T1: `g_tech = exp(-(1/ξ)(V_postTech − V_preTech))`, intensity ∝ R.
- **Damage jump** D0→D1: `g_dmg^ℓ = exp(-(1/ξ)(V_postDmg(λ3_ℓ) − V_preDmg))`, summed over λ3 realizations, intensity `J_d(Y)` on `Y>y_lower`.
- **Solve order (backward):** PostDamagePostTech → {PostDamagePreTech, PreDamagePostTech} → PreDamagePreTech.
- **Full model (6):** insert an intermediate-tech column T½ (PreDamageIntermTech, PostDamageIntermTech); tech jumps T0→T½→T1 with prob (1−π)/π split. Same plan, one more column.

State dims: 3-state regimes = post-tech (logK,Z,Y); 4-state = R&D-active (logK,Z,Y,logR).

---

## 2. What is being tested (the mechanism)

Per regime, parameterize the value with a **level-pinned** form (the winner of the Occam test —
re-centered value net `v_rec = φ(s) − φ(x0) + v0`, OR costate/EGM if it proves better). The level of
each regime is then **one constant**, tied to its already-solved post-jump neighbor by **boundary
value-matching**:

- **Terminal** PostDamagePostTech: pinned by ONE global reference (normalization or analytic anchor).
- **Each upstream regime**: anchor its level to its post-jump neighbor's value at the **jump boundary**
  (tech jump at the tech-transition slice; damage jump at `Y=y_lower`, per λ3 realization), so the
  inter-regime gap that drives `g` is identified, not seed-noise.
- This propagates **one** global level constant through the entire tree → every jump gap is pinned.

Analytical basis (Test 3): each regime's costates satisfy a closed level-free system; the level is one
quadrature; the inter-regime gap needs **one shared anchor**, not two absolute levels.

---

## 3. Anchoring strategy (the key design choice)

Three options, in increasing fidelity — test in this order:

| # | anchor | what it removes | use |
|---|---|---|---|
| A | **boundary value-match to a FROZEN neighbor** (the POC) | per-regime level drift, given a correct neighbor | warm-up sanity only |
| B | **JOINT solve, co-anchored chain**: solve backward, each regime value-matched to the *just-solved* neighbor at the jump boundary; terminal normalized once | the inter-regime offsets, WITHOUT a frozen oracle | **primary test** |
| C | B **+ logN-corrected anchor** (match continuous V, not transformed v: add the `(λ3/2)(Y−ŷ)²` offset across the damage boundary) | the transform mismatch | correctness pass |

Option **B is the real fix**: no frozen surrogate; the terminal's single normalization is the only
free constant; everything else is tied by value-matching. The damage anchor must be applied
**per λ3 realization** at `Y=y_lower` (the gap is λ3-dependent).

---

## 4. Build / solve protocol

1. **Terminal** PostDamagePostTech (3-state): solve with the level-pinned parameterization; fix its
   one global constant by a normalization (e.g. `v=v_ref` at a reference state, or the analytic
   `V≈logK+W` normalization with `V_logK≈1`). → `V_T`.
2. **Post-damage / pre-tech** PostDamagePreTech (4-state, R&D): solve; tech-jump term reads `V_T`;
   anchor its level to `V_T` at the tech-jump slice. → `V_{D1T0}`.
3. **Pre-damage / post-tech** PreDamagePostTech (3-state): solve; damage-jump term reads `V_T`
   (per λ3); anchor to `V_T` at `Y=y_lower`. → `V_{D0T1}`.
4. **Pre-damage / pre-tech** PreDamagePreTech (4-state): solve; reads `V_{D1T0}` (damage) and
   `V_{D0T1}` (tech); anchor to the already-solved neighbors at both boundaries. → `V_{D0T0}`.
5. (**Full 6**: insert intermediate-tech regimes between steps, same pattern.)

Each step: from scratch or warm-started by **derivative-supervision** (pre-fit the net to the
surrogate's costates) to reach loss_v ~1e-3, then continue on the HJB residual. Separate Adam for
value/costate vs controls; grad-clip; standard DGM-PIA.

---

## 5. Metrics & success criterion

**Per regime** (3 seeds): `level_spread` (cross-seed std of probe-mean V), `loss_v`, `FOC`.
**Per jump boundary**: `gap_spread` across seeds; and the **admissibility test** over the ξ sweep.

**Admissibility (the headline)** — at each ξ in `{0.05, 0.025, 0.015, 0.01, 0.005}`, on a probe over
the jump region:
- damage: `g_dmg^ℓ` and the worst-case cumulative damage-jump probability `P_worst`; require
  `P_worst ≥ P_undistorted` (Jensen) and **frac of probe with g_avg≥1 ≈ 1.0** (NOT the exp-tail mean,
  which is junk at small ξ);
- monotonicity: `P_worst` rises as ξ falls (no collapse/turnover).

**Success = ξ\* drops from 0.025 to ≤ 0.01 (target 0.005):** the smallest ξ at which all jumps stay
admissible AND monotone. Cross-check against the production worst-case density test
(`dmg_jump_prob[end] ≥ baseline`, the table in [[lowxi-worstcase-density-limit]]): the new solver
should keep it admissible+monotone below 0.025.

---

## 6. Staged execution

| Stage | scope | gate |
|---|---|---|
| **S0** | Occam test (re-centered vs costate, single regime) — *running* | pick the parameterization |
| **S1** | **4-regime one-jump**, anchor option **B** (joint, co-anchored) | does ξ\* drop below 0.025? frac-admissible at 0.01? |
| **S2** | S1 + logN-corrected anchor (option C) + derivative-supervised warm-start to 1e-3 | converged ξ\* (target 0.005); controls match FD/surrogate |
| **S3** | **full 6-regime** (add intermediate tech) | full-model admissibility + the production economics unchanged |
| **S4** | structural-logK / structural-ξ stacked (optional precision) | residual / level further tightened |

Each stage: agents build the per-regime scripts + the chain driver; **run via sbatch** (compute
survives agent death); read RESULT lines directly; validate with the admissibility test. Joe & Mike
review S1 results before S3/production.

---

## 7. "3 damage × 2 tech" concrete instantiation

If the discrete 3×2 grid is wanted: damage = {D0 pre, D1 post(λ3=λ_a), D2 post(λ3=λ_b)}, tech =
{T0 pre, T1 post}. 6 regimes; damage jump D0→{D1,D2} with the L-weighting; tech jump T0→T1. Same
solve order (terminal D{1,2}T1 first) and same co-anchor protocol. This is a clean finite testbed for
S1 before the continuous-λ3 production model.

---

## 8. Honest risks / watch-items

- **Does the JOINT solve converge?** The frozen-surrogate POC does not prove it; the co-anchored chain
  could still drift if the terminal normalization is off or the value-matching is too weak/strong.
  Watch gap_spread at every boundary, every seed.
- **Conservativeness** of the costate field is only approximate (curl≠0) → the re-centered value net
  (exactly conservative) is preferred if the Occam test says it matches.
- **High-Z tiny loading** `(1−Z)σ_d ≈ 0.001` made a prior costate/BSDE attempt fail (the costate was
  unidentified there). Probe the high-Z corner explicitly.
- **logN transform** across the damage boundary (post carries the λ3 curvature) — use option C for
  correctness, not just internal consistency.
- **Control trainability** from scratch (consumption clamp → FOC blow-up) — derivative-supervised
  warm-start (S2) is the mitigation.

---

*Status: S0 (Occam) running (job in scratchpad/occam_*.out). S1 is the first real multi-regime test.*
