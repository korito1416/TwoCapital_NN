---
title: "Warm and cold starts, and what they select"
author: "TwoCapital team"
date: 2026-07-16
---

## How the trial is designed

The training algorithm, networks, and calibration are exactly those of our previous note (solution_identification_2026-07-11). This note is a controlled trial over the remaining free choices, holding everything else fixed:

- **varied**: the initialization (warm vs. cold start);
- **matched**: every setting uses the same Latin-hypercube **sampling seed**, so runs see the same batch sequences across settings and their training losses are directly comparable. A warm start's initial weights are deterministic;


| setting | description |
|---|-----|
| warm start A | warm start from weights of an earlier model |
| warm start B | warm start + small noise |
| cold start | random start |
| warm start A with wider ξ | as A, with the ξ sampling floor lowered from 0.05 to 0.005 |

## Which settings converge

Every run shown reaches the reference's error tier. The **reference** is the solution we have been using so far. Final training loss (`loss_v`, root-mean-square error in the HJB equation; replicate means; columns = damage state / technology state):

| setting | post / post | post / pre | pre / post | pre / pre |
|-----|---|---|---|---|
| reference | $1.5 \times 10^{-3}$ | $2.4 \times 10^{-3}$ | $2.2 \times 10^{-3}$ | $2.4 \times 10^{-3}$ |
| warm start A | $1.6 \times 10^{-3}$ | $2.3 \times 10^{-3}$ | $1.8 \times 10^{-3}$ | $2.6 \times 10^{-3}$ |
| warm start B | $1.4 \times 10^{-3}$ | $2.0 \times 10^{-3}$ | $1.8 \times 10^{-3}$ | $2.2 \times 10^{-3}$ |
| cold start | $3.2 \times 10^{-3}$ | $8.1 \times 10^{-3}$ | $1.9 \times 10^{-3}$ | $3.8 \times 10^{-3}$ |
| warm start A with wider ξ | $2.2 \times 10^{-3}$ | $3.3 \times 10^{-3}$ | $2.2 \times 10^{-3}$ | $3.4 \times 10^{-3}$ |

The cold start's error is visibly worse in three of the four jump states; we keep its row above for the record and drop it from all figures below. The composition of the objective ({numref}`fig-compl`) reads as in the previous note: the error in the HJB equation dominates everywhere (70–98% of the total).

:::{figure} figures/losscomp_levels.png
:width: 100%
:name: fig-compl
Loss terms in levels, last 100k training steps, common scale across all panels; rows: reference, warm start A, warm start B, warm start A at the wider ξ range. These are the training objectives themselves — aggregated over the entire sampled ξ range, not evaluated at a single ξ; with matched sampling seeds the levels are comparable across rows.
:::

## The initialization selects the solution; the equation errors do not

The comparison logic is: fix ξ, take the decision rules and value function each run delivers, and measure how well they solve the model along the economy's own 60-year trajectory — every term re-evaluated at the states visited along the simulated path and averaged across the years, with the post-damage jump states entered at $Y=\hat y=2.5$.

The error in the HJB equation does not order the accepted solutions in any jump state.

:::{figure} figures/cs1_hjb_by_state.png
:width: 85%
:name: fig-cs1
Error in the HJB equation by jump state across ξ, averaged (root mean square) over the deterministic 60-year trajectory; one curve per setting, averaged across replicates. Triangles: warm start A retrained at the wider ξ range, shown down to ξ = 0.005 together with the reference.
:::

Yet the delivered economies differ far beyond this accuracy spread ({numref}`fig-fan`): at year 60 and ξ = 0.05, green investment averages 67.7 in the reference, 81.9 for warm start B, and 132.8 for warm start A. 

:::{figure} figures/policy_fan_xi0p05_final.png
:width: 100%
:name: fig-fan
Investment paths at ξ = 0.05 along the deterministic trajectory (Brownian shocks shut down, the convention of the paper's path figures); one line per setting, averaged across replicates; gray dotted = the reference.
:::

Under Brownian shocks — with the jumps still shut down, the same no-jump scenario — the picture is unchanged ({numref}`fig-fan-stoch`): the 10–90% bands are narrow relative to the gaps across settings, so the ordering of the delivered economies is not an artifact of suppressing the shocks.

:::{figure} figures/policy_fan_stochastic_xi0p05.png
:width: 100%
:name: fig-fan-stoch
Stochastic counterpart of {numref}`fig-fan`: investment paths at ξ = 0.05 under Brownian shocks (jumps off), 2,000 paths with common random numbers across settings; lines = means, shaded = 10–90% bands, replicate paths pooled.
:::

The sensitivity of the delivered investments to ξ is mild within every solution, and the differences across starts dominate at every ξ ({numref}`fig-invxi`).

:::{figure} figures/investments_vs_xi.png
:width: 100%
:name: fig-invxi
Year-60 green, dirty, and R&D investment against ξ (log scale), deterministic trajectories, replicate means. Within each solution the investments move only mildly with ξ; the gaps across starts dominate at every ξ.
:::

## The selected solutions differ through the welfare level and the marginal values

The previous note measured a nearly parallel welfare-level offset of 1.2–1.8% between two trainings and identified the mechanism: the objective pins the welfare level V only through the discount rate, so a level shift of Δ moves the error in the HJB equation by only δ·Δ. 

Across the accepted settings the welfare level at the initial state runs from 4.21 (reference) to 4.87 (warm start A), all replicate means — and the marginal values move with it.

:::{figure} figures/four_states_final_xi0p05.png
:width: 100%
:name: fig-four
Welfare $V$ and its marginal values by jump state along the deterministic trajectory (Brownian shocks shut down) at ξ = 0.05; one curve per setting (including warm start A at the wider ξ range), averaged across replicates. Post-damage jump states at their entry temperature $Y=\hat y = 2.5$.
:::

The same holds under Brownian shocks ({numref}`fig-four-stoch`): the welfare-level offset and the marginal-value gaps are unchanged, with narrow bands.

:::{figure} figures/four_states_stochastic_xi0p05.png
:width: 100%
:name: fig-four-stoch
Stochastic counterpart of {numref}`fig-four`: welfare $V$ and its marginal values by jump state along stochastic paths (Brownian shocks on, jumps off) at ξ = 0.05; lines = means, shaded = 10–90% bands, replicate paths pooled. Post-damage jump states at their entry temperature $Y=\hat y=2.5$.
:::

**One direction: a selection criterion.** The trial leaves us with several solutions that all pass the accuracy-based acceptance criteria, and the equation errors cannot rank them. What is missing is a criterion, beyond the training loss and the equation errors, for selecting among accepted solutions. Since the differences concentrate in the welfare level — the direction the equation errors are least sensitive to — a useful criterion must measure the level directly; that is the design requirement for whatever check we adopt next.
