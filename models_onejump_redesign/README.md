# One-jump HJB loss / representation redesign (exploratory)

**Scope:** OneJump (π=1), 4 regimes. **Production `models/` is never touched.** Every switch in
`config.py` defaults to production behaviour, so with no environment variables set this variant is
production. Each experiment flips **one** switch (Mike's one-variable-at-a-time rule).

## What was measured first (before building anything)

| finding | number | consequence |
|:--|:--|:--|
| FOC is scale-homogeneous | all terms 0-order | nothing to fix |
| HJB is **not** homogeneous | 0-order + log(K) + climate ∝K + ∝K² + jump ∝R | inhomogeneity **is** the climate externality — intentional economics, not a defect |
| natural HJB scale at X₀ | ≈0.05–0.07 (δ·logK=0.068, −δV=−0.042, δlog(C/K)=−0.031) | residual 2.2e-3 ≈ **3% relative** |
| climate term at X₀ | −1.5e-4 | ~300× below the δ-block: the scale-breaking term is numerically tiny here |
| **scale elasticity** `a=V_logK+V_logR` | **0.77 → 0.45 across logK**; spread only 0.0012 in Z, 0.0021 in Y | **kills constant-`a` detrend**, **proves additive separability** |
| value loss composition | RMS(HJB)≈2e-3 vs RMS(FOC)≈1e-4, summed | scaling the HJB also **re-weights** it vs the FOCs |

### The two results that drove the design

**1. `V = a·logK + W` is mis-specified — do not build it.** A constant `a` forces the scale
elasticity to be constant; measured, it falls 0.7735→0.4532 over logK∈[4.5,7]. A constant-`a` fit
injects `δ·logK·Δa ≈ 2.2e-2` ≈ **10× the residual it is meant to help**.

**2. But `a` varies in logK *alone*** (0.32 vs ~0.002 in Z,Y) — the signature of additive
separability. So the detrend is **generalized**, with `A` a learned 1-D function:

```
V(logK, Z, Y, u) = A(logK) + W(Z, Y, u),      u = logR − logK
    ⟹  V_logK + V_logR = A'(logK)        a function of logK alone  ✓ matches the data
```
This absorbs the `δ·logK` flow term, reduces 4-D → (1-D + 3-D), and leaves **exactly one** additive
constant for the anchor. `A'(logK)` is economically meaningful: the **marginal value of scale**,
≡1 in a climate-free CRS economy; its decline **is** the climate externality.

## Modules

| file | purpose |
|:--|:--|
| `config.py` | all switches; defaults == production; `summary()` for MANIFESTs |
| `uncertainty.py` | θ=1/ξ math; the telescoped jump term `J(1−e^{−θΔv})/θ` (verified to 1e-8) |
| `value_net.py` | `plain` / `recenter` / `separable` value parameterizations |
| `state_layout.py` | unified 6-slot input + **warm-start first-layer remap** |
| `hjb_scaling.py` | non-dimensionalization + the evenness diagnostic |

## Honest limits (verified, not assumed)

- **Non-dimensionalization cannot fix the level.** Gauss–Newton: any weight `w(x)` gives
  `H_level = δ²Σw` and `H_shape = O(1)Σw`, so the ratio `δ²:1 ≈ 1e-4` is **invariant to the divisor**.
  It can only even out fit quality. (Same algebra explains why preconditioning was refuted before.)
- **θ does NOT fix the deep-ξ overflow.** The blow-up is in the product `θ·Δv`: at ξ=0.005, Δv=−1 the
  exponent is 200 and both forms overflow (θ-form measured → `-inf`). A clip is still needed. What θ
  *does* fix is the neutral limit (θ=0 exact) and the 0/0 cancellation.
- **Uniform-θ sampling overcorrects.** ξ<0.1 goes 8.6%→79.9% (good) but near-neutral ξ>10 collapses
  33.8%→**0.2%** (bad — the neutral solution anchors every ξ-comparison). Use a mixture/stratified
  scheme and always report both ends.
- **Unified inputs break warm-start.** First-layer width/order changes; this project's good solution
  is *inherited*. `remap_first_layer_kernel` is the low-risk path (sums the duplicated logξ rows,
  drops the dead `A_g''` column, applies the chain-rule correction for logR→u).

## Anchor: is it economically defensible?

**Within a regime — yes, and it imposes nothing.** The FOCs use only *derivatives* of V, so the level
is pure **gauge**; anchoring selects a representative of the equivalence class. The target is
externally validated: an independent FD solve gives `V(ξ=0.05)=3.669` vs the re-centered net's ≈3.64.
**Across regimes — the level difference is economically real** (it enters `g=exp(−(V^ℓ−V)/ξ)`), so
regimes must be **co-anchored** (value-matched at the jump boundary), never anchored independently.
**Caveat:** pinned ≠ pinned-*correct* — earlier runs pinned the gap consistently but some seeds landed
on the *inadmissible* side, so admissibility is reported **min-over-seeds, never mean**.

**Costate is deliberately NOT used** as the parameterization: it parameterizes the gradient as an
unconstrained vector field (admits non-physical curl → path-dependent level), and lost to re-centering
in this project's own tests (level_spread 0.0077 vs 0.0051; true residual 2.98e-3 vs 2.05e-3).
Re-centering keeps the costate *insight* (level freedom = 1 scalar) with an exactly-conservative
realization.

## Staged plan (one variable per stage; gates are quantitative)

Proving ground for stages 0–2 is **PostDamagePostTech** (3-state terminal, FD ground truth exists,
cheapest). Metrics everywhere: true unclamped HJB residual, FOC residuals, **cross-seed level spread
(min over seeds)**, per-region relative residual, and — for jump stages — frac_admissible (min over seeds).

| # | change (switch) | gate | falsifier |
|:--|:--|:--|:--|
| 0 | baseline replication (no switches) | matches `models/` to ~1e-6 | any drift ⇒ variant is not faithful |
| 1 | non-dim (`REDESIGN_HJB_SCALE=natural`) | region-relative-residual spread ↓ ≥2× | **level spread unchanged** — predicted, and *confirms* the GN argument |
| 2 | anchor (`REDESIGN_ANCHOR=recenter`) | cross-seed level spread ↓ ≥10× at no residual cost | residual worsens >1.2× ⇒ not free here |
| 3 | separable (`REDESIGN_DETREND=on`) | residual ≤ baseline **and** `A'(logK)` reproduces 0.77→0.45 | `A'` flat ⇒ separability wrong |
| 4 | θ (`REDESIGN_XI_PARAM=theta`) | deep-ξ residual ↓ **and neutral-ξ residual not worse** | neutral degrades ⇒ fix sampling first |
| 5 | unified inputs (`REDESIGN_INPUTS=unified`) | remap reproduces legacy outputs to ~1e-5 before any training | mismatch ⇒ remap algebra wrong |

## Questions to ask Lars/Mike before going further

1. The scale elasticity `A'(logK)` falling 0.77→0.45 is the climate externality as a *number*. Is that
   a known/reported object, or is it new and worth a table of its own?
2. Was the full `(logK,Z,Y,logR)` value (rather than a scale-reduced form) chosen deliberately
   *because* the climate externality breaks exact homogeneity?
3. For the unified inputs: is the logξ duplication load-bearing in any way we don't know about, or is
   it purely legacy checkpoint-width padding (our reading)?
4. θ=1/ξ ∈ [0,50] is Lars's suggestion; uniform-θ nearly abandons the neutral end. Is a mixture
   (half uniform-θ, half uniform-logξ) acceptable, or is there a reason to want uniform-θ exactly?
