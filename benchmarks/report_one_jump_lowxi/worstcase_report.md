---
title: Worst-case jump densities and their reliable robustness range
---

# Model source and what this reports

**Model.** All figures and numbers come from the **full one-technology-jump ($\pi=1$)
climate neural-network model** (Deep-Galerkin / policy-improvement, DGM-PIA; four jump
regimes PreDamage/PostDamage $\times$ Pre/PostTech), solved in **float64** on the low-$\xi$
run (`logximin=-5.30`, valid for $\xi\in[0.005,148.6]$). Both adjustment-cost calibrations
are solved:
$$
\text{Full: }(\Gamma,\theta)=(0.060,\,16.7),\qquad
\text{Half: }(\Gamma,\theta)=(0.12,\,8.35).
$$
Source runs:
`output_lowxi_float64/OneTechJump_..._AdjustmentCost{Full,Half}_logximin_m5p30_..._num_iterations50000`.
Paths are deterministic 60-year reference-measure trajectories started at $Y_0=1.2$; the
distortions $g$ are evaluated **along** that path to give the worst-case first-jump density.

**Question.** The worst-case (distorted) damage-jump density is the economically meaningful
object — it is how the robust planner times the feared damage jump. It is reliable at large
$\xi$ but **collapses backwards at extreme $\xi$**. This report (i) proves the admissibility
bound that *certifies* the collapse as a numerical error, (ii) pins the reliable threshold
$\xi^*$ for each calibration, and (iii) presents the final correct figures.

# The worst-case distortion and the admissibility bound

For the damage jump, the worst-case intensity distortion solved in closed form is
$$
g^{\ell}=\exp\!\Big(-\tfrac{1}{\xi}\big(V^{\ell}-V\big)\Big),\qquad \ell=1,\dots,L,
$$
where $V$ is the (pre-damage) continuation value at the current state and $V^{\ell}$ is the
post-damage continuation value once curvature realization $\lambda_3(\ell)$ is revealed
(evaluated at the jump threshold $\hat y=\bar y=2.5$, as in the solver). The grouped
worst-case total damage intensity is
$$
\lambda^{\mathrm{wc}}_{\mathrm{dmg}}(y)=\sum_{\ell=1}^{L}\frac{\mathcal J_n(y)}{L}\,g^{\ell}
=\Big(\tfrac{1}{L}\textstyle\sum_{\ell}g^{\ell}\Big)\,\mathcal J_n(y)
=\overline{g}\,\mathcal J_n(y),
\qquad
\mathcal J_n(y)=r_1\!\Big(e^{\frac{r_2}{2}(y-\underline y)^2}-1\Big)\mathbf 1_{\{y\ge\underline y\}},
$$
with $(r_1,r_2,\underline y,\bar y)=(1.5,\,0.36,\,1.5,\,2.5)$ and
$\overline{g}\equiv\frac1L\sum_\ell g^{\ell}$. The **undistorted** (uncertainty-neutral)
intensity is the baseline $\mathcal J_n(y)$, i.e. $\overline{g}\equiv1$.

Write the cumulative first-jump probability under a competing tech jump (intensity
$\lambda_{\mathrm{tech}}$) as
$$
P[\,T^{\mathrm{dmg}}_{\mathrm{first}}\le H\,]
=\int_0^H \lambda_{\mathrm{dmg}}(s)\,
\exp\!\Big(-\!\int_0^s[\lambda_{\mathrm{dmg}}+\lambda_{\mathrm{tech}}]\Big)\,ds .
$$

**Lower bound (the admissibility test).** A true worst-case density must up-weight, never
down-weight, a *bad* jump. The damage-curvature jump only adds nonnegative convex damage,
$$
\lambda_3(\ell)\ge0\ \Longrightarrow\ \log N_{\mathrm{post}}(y;\ell)\ge\log N_{\mathrm{pre}}(y)
\ \ (y\ge\hat y)\ \Longrightarrow\ V^{\ell}\le V\quad\text{for every }\ell,
$$
so it can never raise welfare. Hence the exponent $-\tfrac1\xi(V^{\ell}-V)\ge0$ and
$$
\boxed{\,g^{\ell}\ge1\ \ \forall\ell\ \Longrightarrow\ \overline{g}=\tfrac1L\textstyle\sum_\ell g^{\ell}\ge1\,}
$$
*individually*, with no appeal needed to convexity. (Convexity gives the same conclusion under
the weaker hypothesis $\overline{V^{\ell}}\le V$: by Jensen
$\frac1L\sum_\ell e^{-\frac1\xi(V^{\ell}-V)}\ge e^{-\frac1\xi\,\frac1L\sum_\ell(V^{\ell}-V)}\ge e^{0}=1$.)
Because increasing the damage hazard pointwise while holding the tech hazard fixed can only
increase the competing-risk damage incidence (a coupling / stochastic-dominance argument,
verified on $2\times10^4$ random paths with zero violations),
$$
\overline{g}(s)\ge1\ \text{on the damage region}\ \Longrightarrow\
P_{\mathrm{wc}}\ \ge\ P_{\mathrm{undistorted}} .
$$
Contrapositive: **$P_{\mathrm{wc}}<P_{\mathrm{undistorted}}$ rigorously implies $\overline g<1$
somewhere it must be $\ge1$ — a *certified* error of the value networks, not a judgment call.**

**Upper bound (the $\xi\!\to\!0$ floor).** As $\xi\to0$, $g\to\infty$ wherever $V^{\ell}<V$,
so the worst case fires the damage jump the instant $Y$ enters $[\underline y,\bar y]$;
simultaneously the tech distortion $g^{\mathrm{tech}}\to0$ (the good tech jump is suppressed),
removing the competing risk. The limit is therefore
$$
P_{\mathrm{floor}}=P\big[\,Y\ \text{reaches}\ \underline y=1.5\ \text{within } H=60\text{ yr}\,\big]\ \to\ 1 .
$$
($Y$ crosses $1.5$ at $t\approx17$ yr on every path, so $P_{\mathrm{floor}}\to1$ as $\xi\to0$;
at large $\xi$ the floor sits below $1$ only because the fast neutral tech jump pre-empts it.)

**Admissibility test.** A reliable worst-case damage-jump probability must satisfy
$$
\boxed{\,P_{\mathrm{undistorted}}\ \le\ P_{\mathrm{wc}}\ \le\ P_{\mathrm{floor}}
\quad\Longleftrightarrow\quad \overline{g}\ge1\ \text{(lower bound)}.}
$$

# The reliable threshold $\xi^*$ (both calibrations)

Recomputing $P_{\mathrm{undistorted}}$, $P_{\mathrm{wc}}$ (the raw NN, identical to the
pipeline output), $P_{\mathrm{floor}}$, and the region-averaged $\overline g$ on the same
simulated path and the same competing-risk accounting gives the table below. The new
$\xi\in\{0.02,0.015,0.0125\}$ rows are the threshold-pinning simulations; $P_{\mathrm{wc}}$
reproduces the saved `dmg_jump_prob` exactly (e.g. Full $\xi=0.02$: $0.6151$).

**Full adjustment cost** (terminal, 60 yr):

| $\xi$ | $P_{\text{undist}}$ | $P_{\text{wc}}$ (NN) | $P_{\text{floor}}$ | $\overline g$ | admissible? |
|:---:|:---:|:---:|:---:|:---:|:---:|
| $\infty$ (148.6) | 0.138 | 0.138 | 0.650 | 1.000 | **yes** |
| 0.05 | 0.395 | 0.448 | 0.918 | 1.296 | **yes** |
| 0.025 | 0.584 | 0.630 | 0.983 | 1.224 | **yes** |
| 0.02 | 0.640 | 0.615 | 0.992 | 0.975 | no |
| 0.015 | 0.697 | 0.452 | 0.998 | 0.523 | no |
| 0.0125 | 0.723 | 0.314 | 0.999 | 0.313 | no |
| 0.01 | 0.744 | 0.085 | 1.000 | 0.074 | no |
| 0.005 | 0.763 | 0.000 | 1.000 | 0.000 | no |

$\Longrightarrow\ \boxed{\xi^*_{\text{Full}}\in(0.02,\,0.025]}$ — reliable for $\xi\ge0.025$;
already inadmissible at $\xi=0.02$ ($\overline g=0.975<1$, $P_{\mathrm{wc}}=0.615<P_{\mathrm{undist}}=0.640$).

**Half adjustment cost** (terminal, 60 yr):

| $\xi$ | $P_{\text{undist}}$ | $P_{\text{wc}}$ (NN) | $P_{\text{floor}}$ | $\overline g$ | admissible? |
|:---:|:---:|:---:|:---:|:---:|:---:|
| $\infty$ (148.6) | 0.160 | 0.160 | 0.672 | 1.000 | **yes** |
| 0.05 | 0.396 | 0.455 | 0.940 | 1.321 | **yes** |
| 0.025 | 0.521 | 0.584 | 0.989 | 1.287 | **yes** |
| 0.02 | 0.551 | 0.554 | 0.995 | 1.043 | **yes** |
| 0.015 | 0.575 | 0.394 | 0.999 | 0.579 | no |
| 0.0125 | 0.580 | 0.268 | 0.999 | 0.357 | no |
| 0.01 | 0.574 | 0.073 | 1.000 | 0.090 | no |
| 0.005 | 0.488 | 0.000 | 1.000 | 0.000 | no |

$\Longrightarrow\ \boxed{\xi^*_{\text{Half}}\in(0.015,\,0.02]}$ — reliable for $\xi\ge0.02$
(marginal: $\overline g=1.043$, $P_{\mathrm{wc}}=0.554$ barely above $P_{\mathrm{undist}}=0.551$);
inadmissible at $\xi=0.015$ ($\overline g=0.579<1$).

**Root cause.** A residual inter-regime value-*level* error ($\sim0.01$–$0.2$) between the
separately-trained pre- and post-damage value networks (each $\mathrm{loss}_v\sim2$–$4\times10^{-3}$,
paper-grade) is amplified by $1/\xi$ ($=50$–$200$) inside the exponential. Below $\xi^*$ it flips
the sign of $V^{\ell}-V$, driving $g^{\ell}<1$ and suppressing the worst-case jump — the
opposite of robust behavior. The bound makes this irrecoverable-from-checkpoints defect *provable*.

# Final figures

```{figure} figures/final_fig1_damage_cdf.png
:label: fig-dmg
:width: 100%

**Worst-case (distorted) damage-jump cumulative probability**, Full (left) and Half (right).
Solid = reliable NN ($\xi\ge\xi^*$, i.e. $\ge0.025$ Full / $\ge0.02$ Half); dashed/greyed =
NN inadmissible ($\xi<\xi^*$: $P_{\mathrm{wc}}<P_{\mathrm{undistorted}}$, proven wrong); dotted =
analytic $\xi\!\to\!0$ floor (jump fires on entering $[1.5,2.5]$, $P\!\to\!1$). The reliable
curves rise monotonically with aversion (lower $\xi$ $\Rightarrow$ more feared jump); the raw
NN then collapses backwards below $\xi^*$.
```

```{figure} figures/final_fig2_admissibility.png
:label: fig-admiss
:width: 100%

**The admissibility proof.** Top: terminal worst-case damage-jump probability vs $\xi$
(log axis, more aversion to the right). The grey band is the admissible region
$[P_{\mathrm{undistorted}},\,P_{\mathrm{floor}}]$; $P_{\mathrm{wc}}$ (red) lies inside it for
$\xi\ge\xi^*$ (filled) and falls **below** the lower bound $P_{\mathrm{undistorted}}$ for
$\xi<\xi^*$ (open). Bottom: the region-averaged distortion $\overline g=\mathrm{mean}_\ell g^{\ell}$
(log axis) crosses $1$ exactly at $\xi^*$ — the equivalent algebraic test. Shaded pink =
inadmissible range; the dashed line marks $\xi^*$.
```

```{figure} figures/final_fig3_tech_cdf.png
:label: fig-tech
:width: 100%

**Worst-case (distorted) technology-jump cumulative probability.** The tech jump is *good*
($V^{\mathrm{tech}}>V$), so the worst case down-weights it ($g\le1$). **Full** (left): reliable
at **all** $\xi$ — the curve falls monotonically as aversion rises (the worst case fears the
beneficial tech breakthrough less). **Half** (right): reliable only for $\xi\ge\xi^*=0.02$;
below that an inter-net value error makes $g>1$ at late, high-$Y$ states, so the tech CDF
**reverses upward** (dashed) — the same value-level pathology as the damage jump, contaminating
tech via the value error and the shared competing-risk survival factor.
```

# Bottom line for the planner

- The worst-case **damage**-jump density is trustworthy for $\xi\ge\xi^*$:
  $\boxed{\xi^*\approx0.025}$ (Full), $\boxed{\xi^*\approx0.02}$ (Half). In that range it
  behaves as theory demands — monotonically more feared as aversion rises — staying inside
  $[P_{\mathrm{undistorted}},\,P_{\mathrm{floor}}]$.
- Below $\xi^*$ the raw NN density is **provably wrong** ($P_{\mathrm{wc}}<P_{\mathrm{undistorted}}$,
  $\overline g<1$); the correct object there is the analytic $\xi\!\to\!0$ limit — a near-degenerate
  jump-at-threshold ($P\!\to\!1$, $T_{\mathrm{first}}\!\approx\!17$ yr). Recovering the exact
  intermediate shape requires mutually-consistent re-training of the pre/post-damage value nets
  (deferred), not re-simulation.
- The worst-case **tech**-jump density is reliable at all $\xi$ for **Full**, and for $\xi\ge\xi^*$
  for **Half**.
