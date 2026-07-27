---
title: "The welfare cost of robustness: where it comes from, how the economy responds"
author: "TwoCapital team"
date: 2026-07-20
---

The planner's aversion to model misspecification carries a welfare cost. By the envelope theorem that cost is the marginal value of relaxing the penalty, and it equals a discounted relative entropy — the statistical distance between the worst-case model the planner acts on and the baseline model — accumulated along the path:
$$
\frac{\partial V}{\partial \xi} \;=\; \mathbb{E}\!\int_0^\infty e^{-\delta t}\Big[\underbrace{\tfrac12\lVert h_t\rVert^2}_{\text{diffusion}} \;+\; \underbrace{\textstyle\sum_\ell \mathcal{J}^\ell_t\big(1 - g^\ell_t + g^\ell_t\log g^\ell_t\big)}_{\text{jumps}}\Big]\,dt .
$$
The integrand is a **sum over uncertainty channels**, so the cost decomposes additively: each Brownian channel contributes $\tfrac12 h^2$ (with $h=-\tfrac1\xi\sigma V_x$), each jump channel contributes $\mathcal{J}(1-g+g\log g)$ (with $g=e^{-(V^\ell-V)/\xi}$). We read this decomposition off the reference's own simulated paths, then ask how the economy actually *responds* to the shocks that dominate it.

## Where the cost comes from, and when

{numref}`fig-paths` traces the welfare cost as a path: how the discounted ($\delta=0.01$) per-channel entropy accumulates over the reference's 60 years (left, to the total $\partial V/\partial\xi = 1.64$ at $\xi=0.05$) and when it is incurred (right, the flow).

:::{figure} figures/welfare_paths.png
:width: 100%
:name: fig-paths
Welfare cost of robustness along the path, $\xi=0.05$. Left: cumulative discounted entropy, stacked by channel, accruing to $1.64$. Right: the flow (integrand) — where along the path each channel's cost is incurred.
:::

The path shows a structure the totals hide:

- **The cost is almost entirely the two jumps, and they are timed oppositely.** The technology-jump cost is *front-loaded* — its flow is highest at the start ($\sim\!0.01$/yr) and decays, because the breakthrough fear is largest while R&D is still building and resolves as the breakthrough becomes likely. The damage-jump cost is *back-loaded* — its flow is zero until $\sim$year 18, then climbs steeply, switching on only as temperature approaches the damage threshold. They cross near year 30. Over the horizon the damage jump accrues $1.06$ and the technology jump $0.48$ — together 94% of the cost.
- **The diffusion channels never leave the floor.** Capital contributes $0.10$; knowledge and climate essentially nothing ($<10^{-3}$). The climate-sensitivity channel (the 144 carbon–temperature models) is welfare-dead: it enters only through $h_y\propto\varsigma V_y$, its cost scales *exactly* as $\varsigma^2$, and $\varsigma=2.2\times10^{-3}$ is the smallest loading in the model — the channel sits $\sim\!1300\times$ below the technology jump. As a solution-held-fixed sensitivity, $\varsigma$ would have to be scaled $\sim\!37\times$ before it even reached parity. If climate-model uncertainty is meant to bind, that is the explicit knob.
- **The damage-jump share is contingent on the belief.** Its $1.06$ rides on the reference's worst-case damage-curvature belief, which our audit finds over-tilted; with a corrected belief its entropy drops about an order of magnitude and the **technology jump becomes dominant**. We would state this rather than lead with the raw split. (The total scales strongly with aversion: $1.64$ at $\xi=0.05$, $0.80$ at $\xi=0.1$, $0.13$ at $\xi=0.3$.)

## How the economy responds: the technology-breakthrough impulse response

The largest single shock in the model is the green-technology breakthrough — green productivity jumps $A_g:0.1085\to0.1567$ and overtakes dirty ($A_d=0.1303$). {numref}`fig-irf` is its impulse response: we take the reference's post-breakthrough policy and integrate the economy forward 60 years from the initial state, for each $\xi$ (solid), against the no-breakthrough counterfactual (dashed).

> *Method.* Deterministic no-shock dynamics; the worst-case drift correction is $\varsigma h_y \sim 10^{-5}$, negligible against $\bar\theta$, so baseline and worst-case paths coincide. The same integrator reproduces the reference's own `SimulationDeterministic` path to $<0.1\%$ ($Y_{60}$ 2.110 vs 2.111, $I_{d,60}$ 4.82 vs 4.83, $C/Y_{60}$ 47.57 vs 47.56%), which fixes the dynamics and control conventions.

:::{figure} figures/irf_tech.png
:width: 100%
:name: fig-irf
Technology-breakthrough impulse response by $\xi$. Solid: post-breakthrough path; dashed: no-breakthrough counterfactual, same $\xi$/colour. $I_g,I_d$ are investment levels, $\mathcal{E}$ emissions, $C/Y$ the consumption share.
:::

**The response is large but nearly uncertainty-invariant.** The breakthrough more than doubles green investment (year-60 $I_g$: $\sim\!63\to155$–$168$). Yet across the entire aversion range — from $\xi=0.05$ to the neutral $\xi=148.6$, a factor of $3000$ — the four response curves nearly coincide: year-60 green investment moves only $155\to168$ ($\sim\!8\%$), emissions $9.6\to9.1$ ($\sim\!6\%$), the consumption share $49.6\to48.3\%$ ($\sim\!1.3$pp). This is the behavioural counterpart of the decomposition's headline: the welfare *cost* of robustness is large, but the *action* it induces — even in response to the model's biggest shock — is almost independent of $\xi$. Weak levers (tiny $\sigma$) and log utility keep policy flat while the fears behind it swing.

*One caveat we would footnote rather than feature: in the reference solution the breakthrough does not decarbonize — dirty investment stays high (dashed $\to5$ absent the shock, solid $\sim\!11$–$12$ with it) and year-60 emissions end above the no-breakthrough path, because the post-tech dirty-investment policy rises with capital and temperature where the finite-difference benchmark falls. That is a property of that policy, flagged separately by the derivative audit; it leaves the near-$\xi$-invariance — the welfare message here — untouched.*
