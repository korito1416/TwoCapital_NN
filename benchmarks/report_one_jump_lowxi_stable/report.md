---
title: "One-Technology-Jump Climate-Economy Model — Robustness (xi) Sensitivity"
---

<!-- topic: One-tech-jump (pi=1) robust climate-economy solution — xi sensitivity (full diagnostic set) | date: 2026-07-01 -->


# Scope and model


All results come from the **one-technology-jump ($\pi=1$) climate–economy model**, solved with the
Deep-Galerkin / policy-improvement neural method (DGM-PIA) in double precision, for **both
adjustment-cost calibrations** (full and half). The calibration is the current paper calibration
(Barnett–Brock–Hansen–Hu–Huang, *TwoCapital_RandD_NBER_ClimFin_2025*): $\delta=0.01$, $Y_0=1.2$,
five equi-spaced damage curvatures $\lambda_3(\ell)\in\{0,\tfrac1{12},\tfrac16,\tfrac14,\tfrac13\}$.


This round changes **only** the robustness penalty $\xi$ — the cost the planner attaches to model
misspecification. Smaller $\xi$ means more aversion; $\xi\to\infty$ is the uncertainty-neutral planner.


We report the full diagnostic set (investment pathways, worst-case distorted probabilities per
channel, and per-$\xi$ numerical accuracy) and flag **only the objects we can certify as numerically
correct** at the $\xi$ shown:

- The **economics** (investment pathways and emissions) and the **continuous-channel worst-case
  temperature distortion** are certified across the full set
  $\xi\in\{\infty,\,0.040,\,0.030,\,0.025,\,0.020,\,0.010,\,0.005\}$.
- The **worst-case damage-jump distribution** is certified **only for $\xi\geq0.025$**. Below that
  it is limited by a cross-regime value-level identification effect (§6–§7), not an arithmetic one.

Year-5 quantities are read along the simulated deterministic path at $t=5$ years; pathway plots run
to 60 years.


&nbsp;


# Robust (uncertainty-averse) terms


A single penalty $\xi$ governs every uncertainty channel. It enters the Hamilton–Jacobi–Bellman
equation through two distortion families, both increasing in $1/\xi$.


### Diffusion (drift) distortion $h$


The worst-case Brownian drift in each independent channel solves

$$
\min_h\; V_x'(\mu+\sigma h)+\tfrac12\mathrm{tr}[\sigma'V_{xx}\sigma]+\tfrac{\xi}{2}h'h
\quad\Rightarrow\quad
h^{*}=-\tfrac1\xi\,\sigma'V_x ,
$$

which, with independent shocks, separates into

$$
h_d=-\tfrac{1}{\xi}\big(V_{\log K}-Z\,V_Z\big)(1-Z)\,\sigma_d,\qquad
h_g=-\tfrac{1}{\xi}\big(V_{\log K}+(1-Z)\,V_Z\big)Z\,\sigma_g,
$$

$$
h_r=-\tfrac{1}{\xi}\,V_{\log R}\,\sigma_r,\qquad
h_y=-\tfrac{1}{\xi}\,V_Y\,\mathcal{E}_t\,\varsigma,
\qquad
\mathcal{E}_t=\eta\,A_d\,(1-Z)\,K ,
$$

where $\mathcal{E}_t$ is emissions and $V_Y$ is the (transform-correct) marginal value of the
temperature anomaly. After substitution the diffusion-uncertainty contribution collapses to the
**quadratic drag**

$$
-\frac{1}{2\xi}\,V_x'\,\sigma\sigma'\,V_x .
$$


### Jump distortion $g$


For each Poisson jump channel with intensity $\mathcal{J}^{\ell}$ and post-jump value $V^{\ell}$,
the worst-case intensity re-weighting solves

$$
\min_{g^{\ell}\ge 0}\ \sum_\ell \mathcal{J}^{\ell}\,g^{\ell}\,(V^{\ell}-V)
+\xi\sum_\ell \mathcal{J}^{\ell}\big(1-g^{\ell}+g^{\ell}\log g^{\ell}\big)
\quad\Rightarrow\quad
g^{\ell *}=\exp\!\Big(-\tfrac{1}{\xi}\big(V^{\ell}-V\big)\Big),
$$

contributing to the HJB

$$
\xi\sum_{\ell}\mathcal{J}^{\ell}(x)\Big[\,1-\exp\!\Big(-\tfrac{1}{\xi}\big(V^{\ell}-V\big)\Big)\Big].
$$

Here $\ell$ ranges over the damage-curvature realizations, with intensity
$\mathcal{J}^{\ell}_n(y)=\tfrac{1}{L}\mathcal{J}_n(y)$ and
$\mathcal{J}_n(y)=r_1\big(\exp(\tfrac{r_2}{2}(y-\underline y)^2)-1\big)\mathbf 1_{\{y\ge\underline y\}}$,
together with the technology jump. The worst-case damage-curvature *weights* are the softmax
$w_\ell=\exp(-V^{\ell}/\xi)/\sum_k\exp(-V^{k}/\xi)$.


### Neutral limit


As $\xi\to\infty$ both distortions vanish: the drag $-\tfrac{1}{2\xi}V_x'\sigma\sigma'V_x\to 0$ and
the jump bracket $\to\sum_\ell\mathcal{J}^{\ell}(V^{\ell}-V)$, recovering the uncertainty-neutral
planner. The $\xi=\infty$ column is this neutral baseline: along the path $g\equiv1$ and $h_y=0$.


&nbsp;


# Investment pathways and emissions vs $\xi$


These controls are reliable at **every** $\xi$ in the set. The continuous robustness channel — the
drift distortion $h_y$ acting on dirty capital, the source of uncertain climate damage — dominates
the control response; the damage-jump distortion does not materially move $i_d$, $i_g$, or
$\mathcal{E}$.


```{figure} mike_diagnostics/figA_investment_paths.png
:label: figpaths
:width: 100%

Simulated pathways of green ($i^g$), dirty ($i^d$), and R&D ($i^r$) investment rates over the
60-year horizon, one curve per $\xi$, both adjustment-cost calibrations (top: full; bottom: half).
The pathways are smooth and monotone in $\xi$ across the entire range including $\xi=0.005$.
```


```{figure} figures/fig1_investment_emissions_vs_xi.png
:label: fig1
:width: 100%

Year-5 dirty investment rate $i_d$, green investment rate $i_g$, and emissions $\mathcal{E}$,
versus $\xi$ (more robustness aversion to the right; $\xi=\infty$ at the left). Both
adjustment-cost calibrations.
```


**Year-5 dirty investment rate $i_d$** (and change from the uncertainty-neutral $\xi=\infty$):

| $\xi$ | Full | (vs. $\infty$) | Half | (vs. $\infty$) |
|:---:|:---:|:---:|:---:|:---:|
| $\infty$ | 10.274\% | — | 10.615\% | — |
| 0.040 | 9.508\% | $-7.5\%$ | 10.221\% | $-3.7\%$ |
| 0.030 | 9.422\% | $-8.3\%$ | 10.186\% | $-4.0\%$ |
| 0.025 | 9.367\% | $-8.8\%$ | 10.164\% | $-4.2\%$ |
| 0.020 | 9.299\% | $-9.5\%$ | 10.138\% | $-4.5\%$ |
| 0.010 | 9.088\% | $-11.5\%$ | 10.063\% | $-5.2\%$ |
| 0.005 | 8.891\% | $-13.5\%$ | 10.003\% | $-5.8\%$ |


**Year-5 emissions $\mathcal{E}$:**

| $\xi$ | Full | (vs. $\infty$) | Half | (vs. $\infty$) |
|:---:|:---:|:---:|:---:|:---:|
| $\infty$ | 9.8605 | — | 10.1845 | — |
| 0.040 | 9.7749 | $-0.9\%$ | 10.1301 | $-0.5\%$ |
| 0.030 | 9.7651 | $-1.0\%$ | 10.1254 | $-0.6\%$ |
| 0.025 | 9.7588 | $-1.0\%$ | 10.1226 | $-0.6\%$ |
| 0.020 | 9.7511 | $-1.1\%$ | 10.1192 | $-0.6\%$ |
| 0.010 | 9.7270 | $-1.4\%$ | 10.1096 | $-0.7\%$ |
| 0.005 | 9.7044 | $-1.6\%$ | 10.1019 | $-0.8\%$ |


Green investment $i_g$ stays within about $0.3$ percentage points across the whole range
(full $44.49$–$44.75\%$; half $50.55$–$50.79\%$) — essentially flat, with a shallow dip near
$\xi=0.04$ before a slight rise.


**Year-5 R&D investment rate $i^r$** ($=I^r/(K^d+K^g)$):

```{figure} figures/fig4_rd_investment_vs_xi.png
:label: fig4
:width: 75%

Year-5 R&D investment rate $i^r$ versus $\xi$, both adjustment-cost calibrations.
```

| $\xi$ | Full | (vs. $\infty$) | Half | (vs. $\infty$) |
|:---:|:---:|:---:|:---:|:---:|
| $\infty$ | 0.542\% | — | 0.479\% | — |
| 0.040 | 0.549\% | $+1.3\%$ | 0.479\% | $-0.1\%$ |
| 0.030 | 0.544\% | $+0.4\%$ | 0.475\% | $-0.9\%$ |
| 0.025 | 0.541\% | $-0.2\%$ | 0.473\% | $-1.4\%$ |
| 0.020 | 0.537\% | $-0.9\%$ | 0.469\% | $-2.2\%$ |
| 0.010 | 0.524\% | $-3.2\%$ | 0.457\% | $-4.7\%$ |
| 0.005 | 0.510\% | $-5.8\%$ | 0.442\% | $-7.8\%$ |


**Readings.**

- **Lower $\xi$ pulls back dirty investment, monotonically**, at every $\xi$ down to $0.005$. The
  robust planner treats dirty capital as less attractive. The effect is **much larger under the full
  adjustment cost** ($-13.5\%$ at $\xi=0.005$) than the half ($-5.8\%$).
- **Green investment is essentially unchanged across $\xi$**: robustness aversion acts on the
  *dirty* side; the green-transition path is set by technology and adjustment economics, not $\xi$.
- **R&D investment $i^r$ is nearly flat**, with a shallow peak near $\xi=0.04$ (Full $+1.3\%$) before
  edging down toward $\xi=0.005$ — small, and in the same direction as the dirty pullback in the
  small-$\xi$ tail. ($i^r\sim0.5\%$, an order of magnitude below $i_d$.)
- **Emissions edge down** with more aversion — a small effect driven by the lower dirty investment.
- **No de-investment.** $i_d$ stays positive at every $\xi$; the robust pullback is a reduction in
  dirty investment, not disinvestment.


&nbsp;


# Worst-case distorted probabilities — climate and damage-curvature channels


We now show the endogenous worst-case probability distortions the robust planner uses, baseline vs
distorted, for the diffusion (climate/capital) and the damage-curvature channels.


```{figure} mike_diagnostics/figD_drift_distortions.png
:label: figdrift
:width: 100%

Worst-case Brownian drift distortions per $\xi$: the climate/TCRE channel $h_y>0$ (worst case shifts
warming up) and the capital channels $h_d,h_g,h_r<0$. All scale with $1/\xi$; the climate channel is
the economically dominant one.
```


```{figure} mike_diagnostics/figB_curvature_distortion.png
:label: figcurv
:width: 100%

Worst-case damage-curvature ($\lambda_3$) distribution: baseline uniform $1/L$ (grey) vs distorted
weights (coloured), per calibration; right panel, weight on the most-severe curvature vs $1/\xi$.
The concentration onto the most-severe damage is **smooth and monotone down to $\xi=0.005$** — this
channel is a *ratio* $w_\ell=g^\ell/\sum_k g^k$, in which the common value-level error cancels, so it
stays reliable where the jump *probability* (§6) does not.
```


The damage-curvature distortion concentrates on the worst (highest-$\lambda_3$) realization as $\xi$
falls — the economically expected direction — reaching essentially a point mass by $\xi\le0.01$. This
is the *tilting toward the worst-case damage model* the planner performs, and it is far more
pronounced than the climate-sensitivity channel.


&nbsp;


# Worst-case temperature drift distortion vs $\xi$ — continuous channel


The **continuous (drift-distortion $h_y$) channel** is certified across the full set including
$\xi=0.01$ and $0.005$. The worst-case temperature drift is
$\bar\theta\,\mathcal{E}_t+\varsigma\,h_y\,\mathcal{E}_t$, against the baseline
$\bar\theta\,\mathcal{E}_t$ (same path emissions, so the only difference is $h_y$). Because $h_y>0$
the worst case is **hotter** at every horizon, and the shift grows linearly in $1/\xi$.


```{figure} figures/fig2_worstcase_temperature_drift.png
:label: fig2
:width: 100%

Isolated worst-case temperature shift $\Delta Y(t)=Y_{\rm worst}-Y_{\rm base}$ over the 60-year
horizon, driven by $h_y$, for the full $\xi$ set. Monotone in time and ordered by $1/\xi$. Left:
full adjustment cost; right: half.
```


**Worst-case minus baseline temperature at $t=60$ yr** ($^\circ$C hotter):

| $\xi$ | Full $\Delta Y_{60}$ | Half $\Delta Y_{60}$ |
|:---:|:---:|:---:|
| 0.040 | $+0.0057$ | $+0.0082$ |
| 0.030 | $+0.0076$ | $+0.0109$ |
| 0.025 | $+0.0091$ | $+0.0131$ |
| 0.020 | $+0.0113$ | $+0.0163$ |
| 0.010 | $+0.0223$ | $+0.0326$ |
| 0.005 | $+0.0438$ | $+0.0651$ |


The shift roughly doubles each time $\xi$ halves — the signature of the $1/\xi$ scaling of $h_y$.


&nbsp;


# Worst-case damage-jump distribution and its stability limit


This is the worst-case **cumulative damage-jump probability** along the temperature path — the
channel the jump distortion $g^{\ell}$ re-weights. Unlike the curvature *weights* (§4), the jump
*probability* is proportional to $g$ itself, so it is exposed to the **absolute** value gap
$V^{\ell}-V$ and is the fragile object at small $\xi$.


```{figure} mike_diagnostics/figC_jump_stability.png
:label: figstab
:width: 100%

**The stability limit, shown.** Left: cumulative worst-case damage-jump probability path per $\xi$.
Middle: its year-60 value vs $1/\xi$ — it rises to a peak at $\xi^*\approx0.025$ and then
**collapses** ($0.63\to0.085\to0.000$ at $\xi=0.025,0.01,0.005$), i.e. more aversion spuriously
gives a *lower* worst-case probability. Right: the tech-jump probability, for contrast.
```


```{figure} figures/fig3_worstcase_jump_cdf_validated.png
:label: fig3
:width: 100%

Certified range $\xi\geq0.025$: worst-case cumulative damage-jump probability $P(\text{jump by }Y)$
vs the temperature anomaly $Y$. Dashed black: undistorted baseline ($g=1$). Coloured: worst-case for
$\xi=0.04,0.03,0.025$. Left: full; right: half. This is a **competing-risk cumulative incidence**
(probability the damage jump fires *first* along the 60-year path), so it tops out well below $1$
(at $\xi=0.025$, $\approx0.63$ full): the green-technology jump competes and a small survival
probability remains at 60 years.
```


**Terminal worst-case damage-jump probability** (cumulative over the 60-year path):

| $\xi$ | Full | Half |
|:---:|:---:|:---:|
| $\infty$ (baseline, $g=1$) | 0.1380 | 0.1602 |
| 0.040 | 0.5193 | 0.5117 |
| 0.030 | 0.6011 | 0.5696 |
| 0.025 | 0.6305 | 0.5836 |
| 0.020 *(past peak)* | 0.6151 | 0.5539 |
| 0.010 *(collapsed)* | 0.0846 | 0.0725 |
| 0.005 *(collapsed)* | 0.0000 | 0.0000 |


**Why the validated range stops at $\xi\approx0.025$.** The jump distortion
$g^{\ell}=\exp\!\big(-\tfrac{1}{\xi}(V^{\ell}-V)\big)$ depends on the *level difference* between the
post-damage value $V^{\ell}$ and the pre-damage value $V$. Each regime is solved as a separate
network, and the HJB pins each one's *absolute level* only weakly — through the $-\delta V$ term with
$\delta=0.01$ — so the two regimes' levels are mutually consistent only to about
$\text{residual}/\delta\approx0.1$. The factor $1/\xi$ amplifies this cross-regime inconsistency; for
$\xi$ below $\approx0.025$ it overwhelms the true value gap and the worst-case jump probability
becomes unreliable (it can violate the admissibility bound $g_{\rm avg}\geq1$ and collapse toward
zero). This is an **identification** limit, not an arithmetic one — double precision does not remove
it (§7 shows the HJB residual itself does *not* degrade at low $\xi$). We therefore certify the
damage-jump *distribution* only for $\xi\geq0.025$.


**Readings.**

- In the certified range the worst-case damage-jump probability **rises monotonically as $\xi$
  falls** and stays admissibly **above** the undistorted baseline: more robustness aversion makes the
  planner act as if the feared damage jump is more likely.
- The $\xi=\infty$ run reproduces the undistorted baseline exactly ($g\equiv1$).
- Below $\xi^*\approx0.025$ the probability collapses — the mapped reliability frontier of this
  object, not a property of the economics (which stay smooth, §3).


&nbsp;


# Numerical accuracy per $\xi$


To document that the low-$\xi$ fragility is confined to the jump *distortion* and is **not** a failure
of the underlying solve, we evaluate the production model's own HJB residual (the trained `loss_v`
quantity, raw and un-rescaled) and FOC error with the $\log\xi$ pseudo-state pinned to each $\xi$, on
a fresh 4096-point batch (float64; the jump-exp clamp never fires, so these are the true unclamped
residuals).


```{figure} mike_diagnostics/figF_per_xi_residual.png
:label: figresid
:width: 100%

Left: raw HJB residual and FOC error vs $1/\xi$ — both hover at the paper's $10^{-3}$ accuracy across
the entire range; the terminal 3-state regime stays clean at every $\xi$. Right: the mechanism — the
value-level gap $\lvert V^\ell-V\rvert$ is essentially flat in $\xi$ (RMS $0.12\to0.23$), but the jump
exponent $\tfrac1\xi\lvert V^\ell-V\rvert$ scales linearly with $1/\xi$, reaching $\approx94$ at
$\xi=0.005$ (past the float32 overflow at $\approx88.7$).
```


| $\xi$ | $1/\xi$ | HJB resid. (initial) | FOC (max) | HJB resid. (terminal) | $g$-exponent |
|---:|---:|---:|---:|---:|---:|
| $\infty$ | 0.007 | 2.03e-3 | 2.0e-4 | 1.52e-3 | 0.002 |
| 0.10 | 10 | 2.63e-3 | 2.0e-4 | 1.67e-3 | — |
| 0.05 | 20 | 5.13e-3 | 2.1e-4 | 1.85e-3 | 6.6 |
| 0.04 | 25 | 5.04e-3 | 2.2e-4 | 1.94e-3 | — |
| 0.03 | 33 | 3.77e-3 | 2.7e-4 | 2.11e-3 | — |
| 0.025 | 40 | 3.23e-3 | 3.0e-4 | 2.24e-3 | — |
| 0.02 | 50 | 4.78e-3 | 3.2e-4 | 2.45e-3 | — |
| 0.01 | 100 | 8.86e-3 | 2.8e-4 | 3.63e-3 | 43 |
| 0.005 | 200 | 6.54e-3 | 7.7e-4 | 6.22e-3 | 94 |

*(initial = PreDamagePreTech, carrying all jump terms; terminal = PostDamagePostTech, 3-state;*
*$g$-exponent $=\max\lvert\tfrac1\xi(V^{\ell}-V)\rvert$, the argument of the jump distortion.)*


**Reading.** The raw HJB residual stays of order $10^{-3}$ at every $\xi$ (worst $\approx9\times
10^{-3}$ at $\xi=0.01$, a $\sim$3–4$\times$ rise from neutral — not a blow-up), and the FOC error is
$\sim10^{-4}$ throughout, an order below the residual. By the paper's own accuracy metric **every
$\xi$ down to $0.005$ converges to $\sim10^{-3}$.** The low-$\xi$ fragility of the damage-jump
distribution (§6) is therefore not a failure of the HJB solve; it is entirely the $1/\xi$
amplification of the value-level gap inside $g$ (right panel above). A converged HJB residual across
$\xi$ does **not** by itself certify the low-$\xi$ worst-case *jump distribution* — which is exactly
why §6 certifies that object only for $\xi\geq0.025$ while the economics (§3) and continuous channel
(§5) are certified to $0.005$.
