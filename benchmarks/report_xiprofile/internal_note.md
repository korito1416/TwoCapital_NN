---
title: "INTERNAL — the decarbonizing solutions, compared on Mike's set"
author: "TwoCapital team (internal working note — not for circulation)"
date: 2026-07-19
---

*Internal only. REPORT DISCIPLINE: every decarbonization-successful solution is compared on Mike's
figure set, one row per solution, Haoyang's formats and normalizations — (1) no-shock 60-year paths,
(2) distorted technology and damage first-jump densities, (3) distorted probability of the climate
models, (4) distorted probability of the damage models over $\lambda_3$. Nothing else. (The
four-channels chart is not part of Mike's set; it stays an internal diagnostic.)*

## The qualifying solutions and where their initializations come from

**Eight solutions decarbonize (emissions fall over the 60 years); every one of them was initialized
from the solution of an economy, and no solution initialized any other way qualifies.** The rows of
every figure below, with the initializing economy in one line:

| solution | initializing economy (the Process behind the map) | λ3-belief screen |
|---|---|---|
| reference (inherited) | the FOUNDING single-jump model (full Process recovered below) | pass (over-tilted vs FD) |
| warm start B | the reference solution + small weight noise (stays in its basin) | pass (over-tilted vs FD) |
| FD anchor (base) | the PAPER's terminal economy solved by verified finite differences (PIBYS forward integral; Richardson + analytic-corner certified) | **FAIL (direction flipped)** |
| FD anchor ($\delta$=0.008) | the same FD economy at a lower discount rate (higher-level decarbonizing planner) | **FAIL (direction flipped)** |
| FD anchor ($\delta$=0.0125) | the same at a higher discount rate | **FAIL (direction flipped)** |
| ABSORB | Nelson–Phelps catch-up: Cobb-Douglas complement trees, deterministic embodied ladder $A_g(s)$; closed form + 1-D solve; structured robustness — ξ moves the LEVEL only (log-separability theorem) | pass (weak; closest to FD) |
| RACE | Aghion–Howitt patent race: breakthrough hazard bought by R&D flow, no knowledge stock; fully analytic in ξ ($i_r^*(\xi)$ from 0.0063 to 0.054) | **FAIL (direction flipped)** |
| GHKM | fossil-flow emissions, productivity damages on the dirty sector, Romer knowledge in production; closed form + robust $f(Y;\xi)$ ODE family | **FAIL (direction flipped)** |

Failed for the record (not shown): warm start A and its wider-ξ retrain (emissions rise to 13–14),
level-shift/refit-weights arms (dirty attractor, emissions 25+), cold and analytic (degenerate). The
regularity: **maps obtained by solving an economy decarbonize; maps obtained by operating on network
weights do not.**

## The inherited warm start is itself a solved economy — its Process, recovered

**Every solution we have ever delivered inherits its initialization from one specific economy, and we
have recovered exactly which one.** The production warm start (the NBER-legacy checkpoints behind the
reference and warm start A) is the trained solution of the FOUNDING single-jump model — a different
economy from the paper's, on ten counted axes. Its Process, reconstructed from the recovered
provenance records:

- **States**: $\log K \in [4,7]$, green share $Z \in [0.01,0.99]$, temperature $Y \in [0,3]$,
  log-knowledge $\log R \in [1,6]$ — same block structure as today (two-capital AK with log
  adjustment costs, knowledge stock, temperature driven by dirty-capital emissions).
- **Pseudo-states**: TWO uncertainty-aversion parameters ($\log\xi$, $\log\xi_{baseline}$, each on
  $[-3,5]$ — dual-ξ robustness, later collapsed to today's single ξ), a post-jump productivity GRID
  $A_g' \in [0.12, 0.16]$ (ten points — the tech jump lands on a sampled grid, not the two-stage
  catch-up/breakthrough tree), and damage curvature $\lambda_3 \in [0, 1/3]$ (five points).
- **Jumps**: a SINGLE technology jump $A_g \to A_g'$ and the damage jump at threshold
  $\hat y = 2.0$ (today 2.5), intensity $r_1(e^{r_2(Y-\underline y)^2/2}-1)$ with $(r_1, r_2) = (1.5, 2.5)$
  (today $r_2 = 0.36$).
- **Calibration** (founding vs today): $\delta$ 0.025 vs 0.01; $A_d$ 0.12 vs 0.1303; $A_g$ 0.10 vs
  0.1085; $\sigma_d=\sigma_g$ **0.15 vs 0.01**; $(\alpha, \Gamma, \theta)$ (−0.0236, 0.025, 100) vs
  (−0.035, 0.060, 16.7); $\sigma_r$ 0.016 vs 0.0078; $\varrho$ 448 vs 746.67; $\eta$ 0.17 vs 0.291.
- **Origin**: the founding networks were SUPERVISED-FIT to `model_results.json` — a finite-difference
  grid solution on 129-point-per-state grids — then trained 2M iterations (piecewise-constant LR
  $10^{-5}$). Those checkpoints are `pretrained/nber_legacy`; every later generation warm-started from
  them through recalibrations, and the FD file itself is lost.

**The inherited map is therefore an economy-zoo member avant la lettre**: a specific, differently
calibrated, differently structured economy whose solved value and policies seeded everything since —
implicitly, and with its own anchor now unrecoverable. The zoo below does explicitly and verifiably
what the pipeline has been doing implicitly for five generations.


## 1. No-shock 60-year paths

**All eight rows decarbonize, and no two rows are the same economy** ({numref}`fig-d1`): year-60
green investment spans 67.7 (reference) to 151.2 (RACE), dirty investment 4.8 down to 0.68, emissions
6.1 down to 2.77 — the depth ordering follows the initializing economy's innovation theory (patent
race deepest, knowledge-in-production next, catch-up mildest of the zoo; the δ-family brackets the FD
anchor; warm start B shadows the reference).

:::{figure} figures_internal/decarb_paths.png
:width: 100%
:name: fig-d1
No-shock 60-year paths, one row per decarbonizing solution: green $I_g$, dirty $I_d$, R&D share of
output, emissions, consumption share; ξ overlaid (reference and warm start B carry the full 4-ξ grid;
the newer arms ξ = 0.05 and neutral).
:::

## 2. Distorted technology and damage first-jump densities

**The worst-case jump beliefs order the solutions more sharply than the paths do** ({numref}`fig-d2`):
under aversion every row pushes the technology jump later and the damage jump earlier and more likely,
but the magnitudes differ by row — the deeper-decarbonizing rows carry visibly different
damage-density peaks at ξ = 0.05, consistent with their initializing economies' different R&D
responses to feared damage.

:::{figure} figures_internal/decarb_densities.png
:width: 100%
:name: fig-d2
Distorted first-jump densities over the 60 years, one row per solution, ξ overlaid; left technology,
right damage.
:::

## 3. Distorted probability of the climate models

**The climate-sensitivity reweighting is nearly common across rows** ({numref}`fig-d3`): at ξ = 0.05
every solution tilts belief modestly toward the higher-sensitivity carbon–temperature pairs, and the
tilt is close to identical — this belief margin does not distinguish the solutions.

:::{figure} figures_internal/decarb_climate.png
:width: 100%
:name: fig-d3
Distorted probability of the climate models (baseline vs. worst case at ξ = 0.05), one panel per
solution, recomputed from each solution's stored $h_y$ path.
:::

## 4. Distorted probability of the damage models over λ3

**Unlike the climate margin, the damage-curvature reweighting SPLITS the solutions** ({numref}`fig-d4`):
only reference, warm start B, and (weakly) ABSORB move mass toward the severe realizations; the other
five rows tilt the WRONG way — the screen developed in §5. The two model-family margins therefore
behave oppositely: climate beliefs are common (§3), damage-curvature beliefs are diagnostic.

:::{figure} figures_internal/decarb_lambda3.png
:width: 100%
:name: fig-d4
Distorted probability of the damage models over $\lambda_3$ (baseline vs. worst case at ξ = 0.05),
one panel per solution.
:::

## 5. The λ3-belief screen: figure 4 is a solution test, and most retrained solutions fail it

**The worst-case damage-model belief must load on the SEVERE realizations (weights increasing in
$\lambda_3$); five of the six newly trained solutions get the DIRECTION wrong** (red tags in
{numref}`fig-d4`). Probing the trained nets in-box confirms this is not a readout artifact: their
post-damage value INCREASES in $\lambda_3$ — training flipped the (correct) slope their maps carried.
Against the verified FD benchmark (five-$\lambda_3$ family: mild tilt, weights
$[0.10, 0.16, 0.21, 0.25, 0.29]$ at ξ = 0.05), the screen is three-way: direction-flipped (five
arms, fail), over-tilted ~7× (reference and warm start B, direction right), and FD-consistent
(ABSORB, weakly). The $\lambda_3$-slope is a further weakly-identified axis that naked training
scrambles — and figure 4 of Mike's own set is precisely the instrument that catches it.

Taken together: eight solutions decarbonize and pass the loss/FOC acceptance checks, yet only
three survive the $\lambda_3$-belief screen — and none matches the verified FD on every axis — the set an eventual selection criterion has to choose
from, and the demonstration that the choice of initializing economy is the choice that matters.
