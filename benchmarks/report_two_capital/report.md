---
title: "Two-Capital Adjustment-Cost Models: Settings and Numerical Experiments"
subtitle: "Deterministic, capital-shock, and model-uncertainty variants — a benchmark report"
authors:
  - name: Climate–Innovation–Uncertainty project
date: 2026-06-25
exports:
  - format: pdf
    template: plain_latex
    output: two_capital_report.pdf
---

## 0. Purpose

This note documents three nested versions of the two-capital adjustment-cost model
used as validation benchmarks for the project's neural-HJB (DGM-PIA) solver, and
reports the numerical experiments run for each. The three models share one
state-reduction and one set of first-order conditions; they differ only in what
randomness/robustness is layered on top:

1. **Deterministic** — no capital risk (a first-order HJB ODE).
2. **Capital shocks** — independent Brownian shocks to each capital stock (a
   second-order HJB ODE).
3. **Model uncertainty** — Hansen–Sargent robustness on top of the shocks (the
   second-order HJB plus a robustness "drag").

Each model is solved two ways — a finite-difference (FD) reference and the
project's neural-network (NN) DGM-PIA solver — so the NN can be validated against
a controlled ground truth. The newest experiments (Sections 2.3–2.4) sweep the
two productivities $A_d$ and $A_g$ to map where the planner switches between
investing and **de-investing** dirty capital.

---

## 1. Common framework

Two reproducible capital stocks, dirty $K^d_t$ and green $K^g_t$. The single
endogenous state is the green capital share

$$
Z_t = \frac{K^g_t}{K^d_t+K^g_t}\in(0,1), \qquad \log K_t = \log\!\big(K^d_t+K^g_t\big).
$$

The problem is homogeneous of degree one in total capital, so the value function
separates,

$$
V(\log K, Z) = \log K + v(Z),
$$

reducing it to a one-dimensional object $v(Z)$. Capital accumulates with
logarithmic adjustment costs

$$
\frac{\mathrm{d}K^j_t}{K^j_t} = \phi_j(i^j_t)\,\mathrm{d}t + (\text{risk, model-specific}),
\qquad
\phi_j(i) = \alpha_j + \Gamma_j\log(1+\theta_j i), \quad j\in\{d,g\},
$$

with investment rates $i^j = I^j/K^j$ and the domain restriction $i^j>-1/\theta_j$
(equivalently the capital-growth factor $1+\theta_j i^j>0$, so capital cannot go
negative even though **investment may be negative — de-investment is allowed**).
Scaled consumption (output net of investment, per unit capital) is

$$
c = \frac{C}{K} = (1-Z)(A_d - i^d) + Z(A_g - i^g).
$$

Writing the marginal values $q_d(Z) = 1 - Z\,v'(Z)$ and $q_g(Z) = 1 + (1-Z)\,v'(Z)$
(so $(1-Z)q_d + Z q_g = 1$), the first-order conditions — **identical in all three
models** — are

$$
\frac{\delta}{c} = q_d\,\frac{\Gamma_d\theta_d}{1+\theta_d i^d}
                 = q_g\,\frac{\Gamma_g\theta_g}{1+\theta_g i^g}
\qquad\Longrightarrow\qquad
i^j = \frac{\Gamma_j\,c\,q_j(Z)}{\delta} - \frac{1}{\theta_j}.
$$

Risk and robustness enter only through the equilibrium slope $v'(Z)$ (hence
through $q_d,q_g$): the control map given $v'$ never changes.

### Calibration (from `models/params.py`)

| Parameter | Value | Note |
|---|---|---|
| $\delta$ | $0.01$ | subjective discount rate |
| $(\alpha_d,\Gamma_d,\theta_d)$ | $(-0.035,\ 0.060,\ 16.7)$ | dirty adjustment cost |
| $(\alpha_g,\Gamma_g,\theta_g)$ | $(-0.035,\ 0.060,\ 16.7)$ | green adjustment cost (symmetric) |
| $A_d$ | $0.1303$ | dirty productivity |
| $A_g$ | $0.1567$ | $=A_g''$, post-tech (breakthrough) |
| $\sigma_d,\sigma_g$ | $0.01$ | capital volatility (shock & uncertainty models; $0.2$ for visibility) |
| $\xi$ | $\{148.4,\ 0.1,\ 0.05\}$ | robustness multiplier (uncertainty model) |
| $Z$ range | $[0.01,\ 0.99]$ | interior grid |

Note $\Gamma_j\theta_j = 0.060\times16.7 = 1.002\approx 1$ (the calibrated Tobin's-$q$
normalization), and $\alpha_d=\alpha_g<0$.

---

## 2. Model 1 — Deterministic (no capital risk)

### 2.1 Setting

With no randomness, Itô gives deterministic laws of motion

$$
\frac{\mathrm{d}\log K_t}{\mathrm{d}t} = (1-Z)\phi_d(i^d) + Z\phi_g(i^g),
\qquad
\frac{\mathrm{d}Z_t}{\mathrm{d}t} = Z(1-Z)\big[\phi_g(i^g) - \phi_d(i^d)\big],
$$

and the HJB collapses to the **first-order** ODE

$$
0 = \max_{i^d,i^g}\Big\{\delta\log c - \delta v(Z)
+ (1-Z)\phi_d(i^d) + Z\phi_g(i^g)
+ Z(1-Z)\big[\phi_g(i^g) - \phi_d(i^d)\big]\,v'(Z)\Big\}.
$$

At $Z\in\{0,1\}$ the $Z$-drift vanishes and $v$ equals the one-capital value
$v_j = \log c_j + \tfrac{\alpha_j}{\delta} + \tfrac{\Gamma_j}{\delta}\log\!\big(\tfrac{\Gamma_j\theta_j c_j}{\delta}\big)$,
$c_j = \tfrac{\delta(1+\theta_j A_j)}{\theta_j(\delta+\Gamma_j)}$.

### 2.2 FD vs NN benchmark

The first-order ODE is weakly conditioned in $v'$ (the $Z$-drift $Z(1-Z)\Delta\phi$
is small at $\delta=0.01$), so a residual-only NN under-identifies the slope. The
remedy is the project's three-loss DGM-PIA (value loss = HJB residual + FOC$_d$ +
FOC$_g$; control loss = Hamiltonian + FOC$_d$ + FOC$_g$), validated against an
upwind-Newton FD reference.

```{figure} figures/det_fd_vs_nn.png
:label: fig-det-fdnn
:width: 95%

Deterministic FD (blue) vs NN (red): $i^d(Z)$, $i^g(Z)$, slope $v'(Z)$, aggregate
productivity $\bar A(Z)=(1-Z)A_d+ZA_g$, and consumption/output $C/Y$.
```

### 2.3 Adjustment-cost curvature sweep ($\theta$)

Holding $\theta_j\Gamma_j=1$ (so $\phi_j'(0)=1$ fixed) and varying the curvature
$\theta$ from near-linear to stiff. At small $\theta$ (cheap to adjust) the planner
de-invests dirty ($i^d<0$) to grow green; $i^d$ crosses zero near $\theta\approx2$
and peaks around $\theta\approx30$. The capital-growth factor $1+\theta i^j$ stays
positive throughout.

```{figure} figures/det_theta_Z07.png
:label: fig-det-theta
:width: 95%

$\theta$ sweep at $Z=0.7$: investment rates $i^d,i^g$ (left) and the capital-growth
factor $1+\theta i^j\ge0$ (right). Investment can be negative; capital cannot.
```

### 2.4 Dirty-productivity sweep ($A_d$) — *new*

**Experiment.** Keep the original adjustment cost, fix green at $A_g=0.1567$, and
sweep dirty productivity $A_d$ from $0.01$ to $0.13$. Read the optimal dirty
investment $i^d$ at three fixed shares $Z=0.6,0.7,0.8$.

**Result.** $i^d$ rises monotonically with $A_d$ and crosses zero at
$A_d^\star\approx0.073$ (almost independent of $Z$). Below the crossing dirty
capital is unproductive enough that the planner **de-invests** it
($i^d<0$, down to $\approx-0.038$ at $A_d=0.01$); above it $i^d>0$ (up to
$\approx+0.064$ at $A_d=0.13$). Green $i^g$ stays $\approx0.14$–$0.15$.

```{figure} figures/det_ad_combined.png
:label: fig-det-ad
:width: 80%

Dirty investment $i^d$ vs $A_d$ at $Z=0.6,0.7,0.8$ ($A_g=0.1567$ fixed). Zero
crossing at $A_d\approx0.073$; the three shares nearly coincide.
```

The per-$Z$ panels (with the green reference curve) are below.

```{figure} figures/det_ad_Z06.png
:label: fig-det-ad06
:width: 70%

$A_d$ sweep, $Z=0.6$.
```
```{figure} figures/det_ad_Z07.png
:label: fig-det-ad07
:width: 70%

$A_d$ sweep, $Z=0.7$.
```
```{figure} figures/det_ad_Z08.png
:label: fig-det-ad08
:width: 70%

$A_d$ sweep, $Z=0.8$.
```

### 2.5 Green-productivity sweep ($A_g$) — *new, the symmetry question*

**Experiment (Lars's follow-up).** Hold dirty fixed at $A_d=0.1303$ and instead
sweep the **green** productivity $A_g$ — i.e. ask how large a tech jump in $A_g$ is
needed before the planner **de-invests dirty** ($i^d<0$). The natural conjecture,
read off Section 2.4, is symmetry: $i^d$ should turn negative once
$A_g\approx 2A_d$.

**Result — the response is *not* symmetric.** $i^d$ falls with $A_g$ but only
*slowly*, and does not cross zero until

| $Z$ | $i^d=0$ at $A_g$ | as multiple of $A_d$ | ($i^g=0$ at $A_g$) |
|---|---|---|---|
| $0.6$ | $0.542$ | $4.16\times A_d$ | $0.059$ |
| $0.7$ | $0.515$ | $3.95\times A_d$ | $0.056$ |
| $0.8$ | $0.494$ | $3.79\times A_d$ | $0.051$ |

So green productivity must reach roughly **four times** dirty productivity
($A_g\approx0.5$), not two times, to drive dirty investment negative.

```{figure} figures/det_ag_combined.png
:label: fig-det-ag
:width: 80%

Dirty investment $i^d$ vs $A_g$ at $Z=0.6,0.7,0.8$ ($A_d=0.1303$ fixed). $i^d$ does
not cross zero until $A_g\approx0.5\approx 4A_d$ (red line). The vertical dashed line
marks the symmetric point $A_g=A_d$.
```

**Why the asymmetry.** From the FOC, $i^d = \Gamma c\,q_d/\delta - 1/\theta$, and
since $\Gamma\theta\approx1$, $i^d<0$ requires $c\,q_d \le \delta/(\Gamma\theta)\approx\delta$.
Lowering $A_d$ directly cuts *both* output $c$ *and* dirty's own attractiveness, so
$i^d$ collapses quickly (Section 2.4). Raising $A_g$ instead *raises* output $c$
(which pushes $i^d$ up) while only gradually lowering $q_d=1-Z v'(Z)$ as green
becomes more valuable — the two effects nearly cancel. Economically: the planner
abandons dirty capital readily when dirty is *intrinsically* unproductive, but is
reluctant to abandon a still-productive dirty stock merely because green has become
attractive — especially since the extra green output makes maintaining the dirty
stock easy to afford. The per-$Z$ panels (showing $i^g$ rising as $i^d$ falls) are
below.

```{figure} figures/det_ag_Z06.png
:label: fig-det-ag06
:width: 70%

$A_g$ sweep, $Z=0.6$.
```
```{figure} figures/det_ag_Z07.png
:label: fig-det-ag07
:width: 70%

$A_g$ sweep, $Z=0.7$.
```
```{figure} figures/det_ag_Z08.png
:label: fig-det-ag08
:width: 70%

$A_g$ sweep, $Z=0.8$.
```

---

## 3. Model 2 — Capital shocks

### 3.1 Setting

Each stock carries an independent Brownian shock,

$$
\frac{\mathrm{d}K^j_t}{K^j_t} = \phi_j(i^j)\,\mathrm{d}t + \sigma_j\,\mathrm{d}W^j_t,
\qquad W^d\perp W^g.
$$

Itô reduction to $(\log K, Z)$ yields the **second-order** HJB

$$
\begin{aligned}
0 = \max_{i^d,i^g}\Big\{ &\,\delta\log c - \delta v(Z) + (1-Z)\phi_d + Z\phi_g
- \tfrac12\big(\sigma_d^2(1-Z)^2 + \sigma_g^2 Z^2\big) \\
&+ \big[\phi_g - \phi_d + (1-Z)\sigma_d^2 - Z\sigma_g^2\big]Z(1-Z)\,v'(Z)
+ \tfrac12 Z^2(1-Z)^2(\sigma_d^2+\sigma_g^2)\,v''(Z)\Big\}.
\end{aligned}
$$

The active $v''$ term makes the problem elliptic and **better conditioned** than
the deterministic case. Boundary values pick up an Itô variance drag
$-\sigma_j^2/(2\delta)$. A correlation variant replaces $W^d\perp W^g$ with
$\mathrm{d}\langle W^d,W^g\rangle = \rho\,\mathrm{d}t$ (tested at $\rho=0$ vs $0.9$);
$\rho=0$ reproduces the independent-shock model exactly.

### 3.2 Experiments

```{figure} figures/shock_multisigma.png
:label: fig-shock-sigma
:width: 95%

FD solutions at $\sigma\in\{0.01,0.2\}$: investment, marginal values $q_d,q_g$,
slope $v'(Z)$, and $C/Y$. At the calibrated $\sigma=0.01$ the diffusion correction
is tiny; $\sigma=0.2$ makes the precautionary effect visible.
```

```{figure} figures/shock_correlation.png
:label: fig-shock-corr
:width: 95%

Correlated shocks ($\rho=0$ vs $0.9$) at $\sigma=0.2$: positive correlation shrinks
the $Z$-diffusion $\propto(\sigma_g\mathrm{d}W^g-\sigma_d\mathrm{d}W^d)$, flattening
the precautionary response.
```

```{figure} figures/shock_fd_vs_nn.png
:label: fig-shock-fdnn
:width: 95%

FD vs NN (DGM-PIA with the $v''$ term) at $\sigma=0.01$, forward-net architecture
with FD warm-start and HJB preconditioning.
```

---

## 4. Model 3 — Model uncertainty (robust control)

### 4.1 Setting

A Hansen–Sargent planner distrusts the capital drifts. Under an alternative model
indexed by distortions $h=(h_d,h_g)$, $\mathrm{d}K^j/K^j = [\phi_j(i^j)+\sigma_j h_j]\mathrm{d}t + \sigma_j\,\mathrm{d}\tilde W^j$,
with relative-entropy cost $\tfrac12(h_d^2+h_g^2)$ and robustness multiplier $\xi>0$
(small $\xi$ = strong uncertainty aversion; $\xi\to\infty$ = no robustness). The
inner minimization over $h$ is closed-form, giving worst-case drifts

$$
h_d^\star = -\tfrac1\xi(1-Z)\sigma_d q_d, \qquad
h_g^\star = -\tfrac1\xi Z\sigma_g q_g,
$$

and adds a **robustness drag** to the second-order HJB:

$$
0 = \max_{i^d,i^g}\Big\{ \underbrace{(\text{shock-model bracket})}_{\text{Section 3.1}}
\;-\; \frac{1}{2\xi}\big[(1-Z)^2\sigma_d^2 q_d^2 + Z^2\sigma_g^2 q_g^2\big]\Big\}.
$$

The drag is $\le0$ and scales as $\sigma^2/\xi$. Limits: $\xi\to\infty$ recovers the
shock model; $\sigma\to0$ recovers the deterministic model. Controls are affected
only indirectly, through the equilibrium $v'(Z)$.

### 4.2 Experiments

FD reference solved independently at $\xi\in\{148.4,\,0.1,\,0.05\}$, with the
worst-case drifts $h_d^\star,h_g^\star$ recovered ex post. At the calibrated
$\sigma=0.01$ the drag $\sigma^2/(2\xi)$ is $\sim10^{-3}$ even at the most
uncertainty-averse $\xi=0.05$, so $\sigma=0.2$ is included to make the mechanism
visible.

```{figure} figures/unc_fd_sigma001.png
:label: fig-unc-001
:width: 95%

Robust FD across $\xi\in\{148.4,0.1,0.05\}$ at $\sigma=0.01$ (calibration):
$i^d,i^g$, $q_d$, $v'(Z)$, and worst-case drifts $h_d^\star,h_g^\star$.
```

```{figure} figures/unc_fd_sigma02.png
:label: fig-unc-02
:width: 95%

Same at $\sigma=0.2$: the robustness drag is now visible — stronger uncertainty
aversion (smaller $\xi$) tilts investment toward the lower-exposure stock.
```

---

## 5. Takeaways

- **One framework, three layers.** The state reduction and FOCs are common; risk
  ($v''$ term) and robustness (the $-\tfrac{1}{2\xi}$ drag) enter only through the
  equilibrium slope $v'(Z)$. This makes the deterministic model the clean base
  case for validating the NN solver before adding $\sigma$ and $\xi$.

- **De-investment is governed by *own* productivity.** Dirty investment turns
  negative at $A_d\approx0.073$ when dirty is made unproductive directly, but the
  same switch requires $A_g\approx0.5\approx4A_d$ when triggered by raising green
  productivity. The model is therefore **far from symmetric** in $(A_d,A_g)$: a
  large green tech jump shifts *new* investment toward green and lowers $i^d$, but
  does not by itself drive the planner to liquidate a still-productive dirty stock.

- **NN validation.** In all three models the three-loss DGM-PIA NN matches the FD
  reference once the FOC residuals are included (curing the weak $v'$ identification
  of a residual-only loss).
