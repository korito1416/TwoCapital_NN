---
title: One-jump robustness ($\xi$) sensitivity
---

# What changed: the robustness parameter $\xi$

**Model source.** Figures come from the **full one-jump ($\pi=1$) climate neural-network model**
(Deep-Galerkin / policy-improvement, DGM-PIA; four jump regimes), **not** a finite-difference
benchmark; both adjustment-cost calibrations are solved.

This round changes **only $\xi$** (the penalty on model misspecification). A single $\xi$ governs
every uncertainty channel, entering the Hamilton–Jacobi–Bellman equation through three terms,
**all increasing in $1/\xi$** (smaller $\xi$ = more aversion):

1. **Diffusion (drift) distortion** — the worst-case Brownian drift in each channel,
$$
h_d=-\tfrac{1}{\xi}\big(V_{\log K}-Z\,V_Z\big)(1-Z)\,\sigma_d,\qquad
h_g=-\tfrac{1}{\xi}\big(V_{\log K}+(1-Z)\,V_Z\big)Z\,\sigma_g,
$$
$$
h_r=-\tfrac{1}{\xi}\,V_{\log R}\,\sigma_r,\qquad
h_y=-\tfrac{1}{\xi}\,V_Y\,\eta\,A_d\,(1-Z)\,K\,\varsigma,
$$
(where $\eta\,A_d\,(1-Z)\,K=\mathcal{E}_t$ is emissions, and $V_Y$ is the marginal value of the
temperature anomaly),
which after substitution leave the **quadratic drag**
$$
-\frac{1}{2\xi}\,V_x'\,\sigma\sigma'\,V_x .
$$

2. **Jump distortion** — the worst-case re-weighting of each Poisson jump (damage jump and the
$\pi=1$ technology jump),
$$
g^{\ell}=\exp\!\Big(-\tfrac{1}{\xi}\big(V^{\ell}-V\big)\Big),
\qquad\text{contributing}\qquad
\xi\sum_{\ell}\mathcal{J}^{\ell}(x)\Big[1-\exp\!\Big(-\tfrac{1}{\xi}\big(V^{\ell}-V\big)\Big)\Big].
$$

3. As $\xi\to\infty$ both vanish ($-\tfrac{1}{2\xi}\,V_x'\sigma\sigma'V_x\to 0$ and the jump
bracket $\to\sum_\ell \mathcal{J}^{\ell}(V^{\ell}-V)$): the uncertainty-neutral planner.

**$\xi$ values swept:** $\{\infty,\,0.10,\,0.05,\,0.04,\,0.03,\,0.02,\,0.01\}$
($\infty$ implemented numerically as $\xi=148.6$). The four smallest were the new request.

# $\xi$ sensitivity: investment and emissions

```{figure} figures/fig1_paths_by_xi.png
:label: fig1
:width: 100%

Deterministic 60-year paths of the dirty investment rate $i_d$, green investment rate $i_g$,
and emissions $\mathcal{E}$, for each $\xi$, under the full (left) and half (right)
adjustment-cost calibrations. Darker/redder = more robustness aversion (smaller $\xi$).
```

```{figure} figures/fig2_pullback_vs_xi.png
:label: fig2
:width: 100%

The robustness pullback at year 5: dirty investment rate $i_d$ (left) and green investment
rate $i_g$ (right) versus $\xi$ (log axis, more aversion to the right), both calibrations.
```

**Dirty investment rate $i_d$ at year 5** (and change from the uncertainty-neutral $\xi=\infty$):

| $\xi$ | Full | (vs. $\infty$) | Half | (vs. $\infty$) |
|:---:|:---:|:---:|:---:|:---:|
| $\infty$ | 10.27\% | — | 10.47\% | — |
| 0.10 | 9.66\% | $-5.9\%$ | 10.12\% | $-3.3\%$ |
| 0.05 | 9.45\% | $-8.0\%$ | 10.04\% | $-4.1\%$ |
| 0.04 | 9.38\% | $-8.7\%$ | 10.02\% | $-4.3\%$ |
| 0.03 | 9.29\% | $-9.5\%$ | 9.99\% | $-4.6\%$ |
| 0.02 | 9.16\% | $-10.8\%$ | 9.96\% | $-4.9\%$ |
| 0.01 | 8.95\% | $-12.9\%$ | 9.91\% | $-5.3\%$ |

Green investment rate $i_g$ moves by under $0.3$ percentage points across the whole range
($44.5\%\to44.6\%$ full; $51.1\%\to51.0\%$ half); emissions $\mathcal{E}$ at year 5 fall
$9.86\to9.71$ (full, $-1.5\%$) and $10.17\to10.09$ (half, $-0.8\%$).

**Readings.**

- **Lower $\xi$ pulls back dirty investment, monotonically.** The robust planner treats dirty
  capital — the source of uncertain climate damage — as less attractive, so $i_d$ falls as
  $\xi$ drops. The effect is **much larger under the full adjustment cost** than the half.
- **Green investment is essentially unchanged across $\xi$.** Robustness aversion acts on the
  *dirty* side; the green-transition path is set by the technology/adjustment economics, not by
  $\xi$.
- **Emissions edge down** with more aversion (a small effect, driven by the lower dirty
  investment), consistent with the adjustment-cost report's finding that the green transition,
  not robustness, is the dominant emissions lever.
- **No de-investment.** $i_d$ stays positive at every $\xi$ down to $0.01$; the robust pullback
  is a reduction in dirty investment, not disinvestment.
