---
title: "A reporting standard for our neural-network solutions"
author: "TwoCapital team"
date: 2026-07-20
---

**Change log.** This report establishes a reporting standard for our neural-network solutions. Every solution is reported on its delivered outcomes — the no-shock investment, emissions, and consumption paths — and its distorted worst-case beliefs, across the grid $\xi \in \{0.05, 0.1, 0.3, 148.6\}$ (ξ = 148.6 is the uncertainty-neutral case). We demonstrate the standard on two solutions, each a row in every figure: the reference solution we have been using, and a second solution — warm start B — trained from the same base.

## The re-simulation reproduces the earlier results exactly

The reference solution is the smart-guess (warm-start) solution we have been using, trained earlier by Haoyang. A fresh re-simulation of it matches the stored results. Over the full 60-year path, across every quantity and every ξ, consumption, with the investment, emissions, and temperature paths are bit-identical.

The first two rows of every figure are the reference — Haoyang's stored results and this note's re-simulation; they are visually indistinguishable at every ξ ({numref}`fig-align`), so each figure doubles as a replication check. The third row is a second solution, warm start B, shown in the same format; it is a different solution, not a re-simulation of the reference, so its row does not coincide with the first two.

:::{figure} figures/alignment_paths.png
:width: 100%
:name: fig-align
Top two rows: Haoyang's stored results and this note's re-simulation of the reference, each with the
full ξ overlay — indistinguishable at every ξ (largest absolute difference across every quantity, every
ξ, and the full 60-year path is $3.8 \times 10^{-6}$). Bottom row: warm start B, a second trained
solution, in the same format.
:::

Within each solution the delivered outcomes barely move with ξ: more uncertainty aversion (smaller ξ) lowers green and dirty investment and emissions and raises the consumption share, but the spread across ξ is narrow. The two solutions, by contrast, differ materially — warm start B runs higher green and dirty investment, higher emissions that peak before falling, and a lower consumption share: a larger, later-decarbonizing economy than the reference.

## The aversion concentrates in the worst-case beliefs

The first-jump densities move substantially with ξ ({numref}`fig-density`): under more uncertainty aversion the worst-case belief pushes the technology breakthrough later and pulls the damage jump earlier and much more likely. These densities also track each solution's own path: warm start B, the higher-emission economy, carries an earlier and more likely worst-case damage jump than the reference.

:::{figure} figures/jump_densities.png
:width: 100%
:name: fig-density
Distorted first-jump densities over 60 years across ξ — reference (Haoyang's stored results, then this note's re-simulation) and warm start B, technology at left and damage at right. Smaller ξ moves the technology breakthrough later and the damage jump earlier and more likely; warm start B's higher-emission path gives it an earlier, more likely damage jump.
:::

The reweighting over the two model families is more nearly common across the solutions. Baseline (equal-weight prior) against the distorted worst case, at ξ = 0.3, 0.1, 0.05 (the neutral case has no distortion and is omitted): over the carbon–temperature model pairs, **smaller ξ shifts belief mass toward higher climate sensitivity** ({numref}`fig-climate`), and over the damage-curvature realizations, toward the more severe (higher-$\lambda_3$) models ({numref}`fig-dmg`). Both solutions tilt the same way and by close to the same amount, though at these ξ both shifts are modest relative to the jump-timing distortions above.

:::{figure} figures/climate_dist_row.png
:width: 100%
:name: fig-climate
Distorted probability of the climate-sensitivity models (carbon–temperature model pairs), baseline vs. worst-case, at ξ = 0.3, 0.1, 0.05 — reference (Haoyang's stored results, then this note's re-simulation) and warm start B. Smaller ξ shifts belief mass toward higher climate sensitivity; the reweighting is close across all three rows.
:::

:::{figure} figures/dmg_dist_row.png
:width: 100%
:name: fig-dmg
Distorted probability of the damage-curvature models over $\lambda_3$, baseline vs. worst-case, at ξ = 0.3, 0.1, 0.05 — reference (Haoyang's stored results, then this note's re-simulation) and warm start B. Smaller ξ shifts belief mass toward the more severe (higher-$\lambda_3$) damage realizations; both solutions load on the severe realizations, warm start B slightly more.
:::

Taken together, both solutions' investment and consumption paths are nearly invariant to ξ, while their worst-case beliefs carry most of the response to uncertainty aversion. Across the two solutions the pattern sharpens: their delivered outcomes differ materially, their path-driven jump-timing beliefs follow suit, but their model-family reweightings — which climate model, which damage curvature — nearly coincide. That is why the standard reports both: the outcomes are what training pins down for each economy, while a solution's response to uncertainty lives in its distorted beliefs.
