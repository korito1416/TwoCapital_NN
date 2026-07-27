---
title: "One-Jump Climate–Economy Model: Adjustment-Cost Sensitivity"
date: 2026-06-27
exports:
  - format: pdf
    template: plain_latex
    output: one_jump_adjcost_report.pdf
---

## 1. What changed

This note reports the effect of **halving the capital adjustment cost** in the
one-technology-jump ($\pi=1$) climate–economy model. *Only the adjustment-cost
block is modified; every other parameter and equation is unchanged.*

The model has two reproducible capital stocks — dirty $K^d$ and green $K^g$, with
green share $Z=K^g/(K^d+K^g)$ — whose use generates emissions that accumulate into
a temperature anomaly and a Poisson **damage jump**; R&D drives a Poisson
**technology jump** (a one-time green-productivity breakthrough); and the planner is
averse to model misspecification with aversion parameter $\xi$ (smaller $\xi$ = more
robustness, $\xi=\infty$ = rational expectations). Each capital accumulates with the
logarithmic adjustment technology

$$
\phi_j(i^j)=\alpha_j+\Gamma_j\log\!\big(1+\theta_j\,i^j\big),
\qquad j\in\{d,g\},
$$

where $i^j=I^j/K^j$ is the investment rate, subject to $1+\theta_j i^j>0$ (capital
cannot go negative, but $i^j<0$ — de-investment — is admissible). Scaled
consumption is

$$
c=(1-Z)\big(A_d-i^d\big)+Z\big(A_g-i^g\big)-i^r .
$$

The parameter $\theta_j$ sets the friction: a **larger** $\theta_j$ is a cheaper,
more flexible adjustment technology (the marginal penalty $\partial\phi_j/\partial
i^j=\Gamma_j\theta_j/(1+\theta_j i^j)$ stays positive but the curvature falls). The
experiment halves $\theta$ and doubles $\Gamma$,

$$
\big(\Gamma_d,\Gamma_g\big):\ 0.06\ \longrightarrow\ 0.12,
\qquad
\big(\theta_d,\theta_g\big):\ 16.7\ \longrightarrow\ 8.35,
$$

with all other parameters at baseline. The value function is solved with the
project's neural-network DGM-PIA solver, warm-started from the converged baseline
($\theta=16.7$) and validated against it (per-regime HJB residual
$1.5\text{–}2.4\times10^{-3}$). Every path below overlays
$\xi\in\{0.05,0.1,0.3,\infty\}$.

---

## 2. Optimal paths under the half-adjustment-cost calibration

The planner starts in the pre-damage pre-tech state; the deterministic path is
integrated for 60 years. Colours: $\xi=0.05$ (orange), $0.1$ (blue), $\infty$ (red).

```{figure} figures/half_E.png
:width: 70%
:name: fig-E
**Emissions.** Peak near year 6, then a decline to $\sim$6 by year 60.
```

```{figure} figures/half_RD.png
:width: 70%
:name: fig-RD
**R&D investment, $I_r/Y$.** Small and roughly flat across $\xi$.
```

```{figure} figures/half_DirtyInvestment.png
:width: 70%
:name: fig-Id
**Dirty investment, $I_d/Y$ (%).** Falls from $\sim$11% to $\sim$1%, but stays
positive — **no de-investment** (at this calibration $A_g$ is too small to make
de-investing dirty capital optimal).
```

```{figure} figures/half_I_d.png
:width: 70%
:name: fig-Id-level
**Dirty investment level, $I_d$.** Same monotone draw-down as the output share.
```

```{figure} figures/half_GreenInvestment.png
:width: 70%
:name: fig-Ig
**Green investment, $I_g/Y$ (%).** Dominant use of output, rising $\sim$52%$\to$57%
— the transition is a *reallocation toward green*.
```

```{figure} figures/half_I_g.png
:width: 70%
:name: fig-Ig-level
**Green investment level, $I_g$.** Same rising profile as the output share.
```

```{figure} figures/half_ConsumptionOutputRatio.png
:width: 70%
:name: fig-C
**Consumption, $C/Y$ (%).** Rises $\sim$32%$\to$41% as investment needs ease.
```

```{figure} figures/half_tech_jump_prob.png
:width: 70%
:name: fig-techprob
**Technology first-jump cumulative incidence (distorted).**
```

```{figure} figures/half_dmg_jump_prob.png
:width: 70%
:name: fig-dmgprob
**Damage first-jump cumulative incidence (distorted).**
```

```{figure} figures/half_Dmg_Dist_xi148600.png
:width: 60%
:name: fig-dmgdist-inf
**Damage-model probabilities, $\xi=\infty$** (distorted = baseline).
```

```{figure} figures/half_Dmg_Dist_xi0050.png
:width: 60%
:name: fig-dmgdist-rob
**Damage-model probabilities, $\xi=0.05$** (mass moved onto the worst $\lambda_3$).
```

```{figure} figures/half_Climate_Dist_xi148600.png
:width: 60%
:name: fig-climdist-inf
**Climate-model distribution, $\xi=\infty$** (distorted = baseline).
```

```{figure} figures/half_Climate_Dist_xi0050.png
:width: 60%
:name: fig-climdist-rob
**Climate-model distribution, $\xi=0.05$** (mass moved toward the worst case).
```

The four $\xi$ curves enter every panel; uncertainty aversion shifts the *controls*
by under a percentage point but distorts the *jump beliefs* strongly — the damage
jump up ($0.15\to0.42$ by year 60) and the technology jump down ($0.84\to0.57$),
with the damage- and climate-model distributions ({numref}`fig-dmgdist-rob`,
{numref}`fig-climdist-rob`) relocated onto the worst cases. That belief channel is a
property of $\xi$, not of the adjustment cost, and is not pursued further here.

---

## 3. Adjustment-cost sensitivity: baseline vs half

We compare the baseline ($\theta=16.7,\ \Gamma=0.06$) and half
($\theta=8.35,\ \Gamma=0.12$) calibrations along the $\xi=\infty$ path, so the
contrast isolates the adjustment-cost friction.

```{figure} figures/adjcost_comparison.png
:width: 100%
:name: fig-adjcost
**Baseline (solid) vs half adjustment cost (dashed), $\xi=\infty$.** Left: dirty
investment $I_d/Y$. Middle: green investment $I_g/Y$. Right: consumption $C/Y$.
```

Halving the adjustment cost makes capital reallocation cheaper, and the planner
front-loads the green transition:

- **Green investment $\sim$8–9 pp higher** throughout (52$\to$57% vs 45$\to$48%).
- **Dirty investment drawn down faster** — slightly higher at first, then much
  lower (1.3% vs 3.6% at year 60), crossing the baseline near year 6.
- **Consumption $\sim$7–8 pp lower** (32$\to$40% vs 40$\to$47%): the extra output
  funds the faster green build-out.

---

## 4. Summary

Halving the capital adjustment cost ($\theta:16.7\to8.35$, $\Gamma:0.06\to0.12$)
buys a faster, more capital-intensive green transition: green investment rises
$\sim$8–9 pp, dirty capital is wound down faster (though never de-invested), and
near-term consumption falls $\sim$7–8 pp. All other model behaviour — emissions
profile, jump dynamics, and the robustness belief-distortion — is unchanged by the
adjustment-cost block.
