---
title: "One-Jump Climate–Economy Model: Adjustment-Cost Sensitivity"
date: 2026-06-27
exports:
  - format: pdf
    template: plain_latex
    output: one_jump_adjcost_report.pdf
---

## 1. What changed

We halve the capital adjustment cost in the one-technology-jump ($\pi=1$)
climate–economy model — **the project's full neural-network (DGM-PIA) model, not the
simplified two-capital benchmark**. Each capital $j\in\{d,g\}$ accumulates with the
logarithmic adjustment technology

$$
\phi_j(i^j)=\alpha_j+\Gamma_j\log\!\big(1+\theta_j\,i^j\big),
\qquad i^j=I^j/K^j,\quad 1+\theta_j i^j>0,
$$

in which a larger $\theta_j$ is a cheaper, more flexible adjustment technology. The
experiment doubles $\Gamma$ and halves $\theta$,

$$
\big(\Gamma_d,\Gamma_g\big):\ 0.06\ \longrightarrow\ 0.12,
\qquad
\big(\theta_d,\theta_g\big):\ 16.7\ \longrightarrow\ 8.35,
$$

with every other parameter unchanged. Both calibrations are solved with the
project's neural-network solver (the half-cost value function warm-started from, and
validated against, the baseline).

## 2. Adjustment-cost sensitivity: baseline vs half

The two calibrations are compared along the certainty-equivalent path (no
model-uncertainty distortion, $\xi=\infty$), so the contrast isolates the
adjustment-cost friction.

```{figure} figures/adjcost_comparison.png
:width: 100%
:name: fig-adjcost
**Baseline (solid, $\theta=16.7$) vs half adjustment cost (dashed, $\theta=8.35$).**
Left: dirty investment $I_d/Y$. Middle: green investment $I_g/Y$. Right: consumption
$C/Y$. All as a percent of output, over 60 years.
```

Halving the adjustment cost makes capital reallocation cheaper, and the planner
front-loads the green transition:

- **Green investment is $\sim$8–9 pp higher** throughout (52$\to$57% vs 45$\to$48%).
- **Dirty investment is drawn down faster** — slightly higher at first, then much
  lower (1.3% vs 3.6% at year 60), crossing the baseline near year 6; it stays
  positive throughout (no de-investment).
- **Consumption is $\sim$7–8 pp lower** (32$\to$40% vs 40$\to$47%): the extra output
  funds the faster green build-out.

The dirty stock is run down but never actively scrapped. With adjustment costs,
dirty capital still productive, and the de-investment threshold for green
productivity ($A_g\approx2\text{–}4\times A_d$ in the simple model versus $\sim1.2\times$
here) far out of reach, the planner optimally lets dirty capital depreciate rather
than de-investing it — and even strong robustness does not change this.

In one line: cheaper adjustment buys a faster, more capital-intensive green
transition at the cost of near-term consumption.

## 3. Consequences: a deeper transition, but not lower emissions

Do those investment shifts actually decarbonize faster? The two outcomes that
matter are the green capital share $Z=K^g/(K^d+K^g)$ and emissions $E$.

```{figure} figures/adjcost_consequences.png
:width: 100%
:name: fig-consequences
**Consequences, baseline (solid) vs half adjustment cost (dashed), $\xi=\infty$.**
Left: emissions $E$. Right: green capital share $Z$ (%).
```

- **The capital transition is deeper.** The green share reaches **93.6%** by year 60
  under half adjustment cost, versus **87%** at baseline ($+6.6$ pp): cheaper
  reallocation lets the planner tilt the capital stock further toward green, sooner.
- **Emissions are not lower — they are slightly higher through the transition.**
  Half-cost emissions sit *above* baseline for years $\sim$5–50 (peaking near 10.1
  rather than declining monotonically) and only converge by year 60. Cheaper
  adjustment accumulates more *total* capital — dirty as well as green early on — so
  the larger economy emits more even as its green share climbs faster.

The adjustment cost is therefore a lever on the **speed and depth of the capital
transition**, not on emissions: halving it buys a greener capital stock but is
climate-neutral (mildly emissions-raising) along the path.

## 4. How much does uncertainty aversion ($\xi$) matter?

Overlaying $\xi\in\{0.05,0.1,0.3,\infty\}$ isolates the robustness effect (smaller
$\xi$ = more uncertainty aversion; $\xi=\infty$ = rational expectations).

```{figure} figures/adjcost_xi.png
:width: 100%
:name: fig-xi
**Effect of uncertainty aversion $\xi$.** Top: the dirty and green investment rates
$i_d$, $i_g$. Bottom: the distorted technology- and damage-jump cumulative
probabilities.
```

- **On quantities, a small precautionary tilt away from dirty.** The robust planner
  invests a little less overall but cuts the *dirty* rate $i_d$ much more than the
  green rate $i_g$ ($\sim$7% vs $\sim$2% by year 60), so dirty capital grows more
  slowly and the green share ends slightly *higher* ($93.9\%$ vs $93.6\%$) — a tilt
  toward green by cutting dirty, not by adding green.
- **On beliefs, a great deal.** The robust planner distorts the jump probabilities
  strongly and pessimistically: by year 60 the *feared* damage jump rises from
  $0.16$ at $\xi=\infty$ to $0.42$ at $\xi=0.05$, while the *hoped-for* technology
  jump falls from $0.84$ to $0.57$.

So uncertainty aversion acts mainly through **worst-case belief distortion**, with a
secondary precautionary tilt away from the damaging dirty capital.

## 5. Next steps

1. **De-investment threshold in the full model** — sweep $A_g$ (and/or damage
   severity) to locate where active de-investment of dirty capital first becomes
   optimal, pinning down on the full model what the simple model only brackets.
2. **Finer $\xi$ grid** — any $\xi\in[0.05,148.6]$ is a quick re-simulation (trained
   input), reportable on demand.
3. **Technology-jump intensity ($1\times$ vs $2\times$)** — breakthrough speed's
   effect on the transition and emissions.

---

*Model source: full one-jump climate–economy model (NN DGM-PIA solver) — baseline
$\theta=16.7$ vs half-cost $\theta=8.35$; comparisons at $\xi=\infty$ and, in §4, the
$\xi$ grid. The two-capital FD benchmark is reported separately.*
