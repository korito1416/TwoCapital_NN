---
title: "Robust Climate Model — Worst-Case Distortions and Stability across xi"
subtitle: "One-Jump Climate-Innovation Model (DGM-PIA): pathways, distorted probabilities, stability, accuracy"
date: 2026-07-01
authors:
  - name: TwoCapital solver team
---

# Scope

Full one-jump ($\pi=1$) climate–innovation NN model (DGM-PIA), double precision, both adjustment-cost
calibrations, current paper calibration ($\delta=0.01$, $Y_0=1.2$). Robustness aversion $\xi$ swept over
$\{\infty,0.05,0.04,0.03,0.025,0.02,0.01,0.005\}$; smaller $\xi$ = more averse. Each section below shows one
of the requested diagnostics for **all** $\xi$.

**Headline.** We pushed $\xi$ well below the range the paper uses ($\xi\ge0.05$). As $\xi$ falls below
$\approx0.025$, the worst-case damage-jump distribution **breaks down** (§5): its density
collapses and the worst-case jump intensity falls below the neutral baseline, which is impossible. The
solver's own error stays $\sim10^{-3}$ at every $\xi$ (§6), so this is not a convergence failure. The cause
(§7) is a lack of cross-regime consistency: the value function is determined only up to an additive
constant, and the before- and after-jump regimes are solved separately, so their value functions are not
comparable. The worst-case belief depends on the difference between the two, so it inherits the mismatch,
which robustness ($1/\xi$) amplifies until it dominates below $\xi\approx0.025$. This mismatch is present at every $\xi$ and
becomes a *provable* failure of the worst-case belief only below $\xi^*\approx0.025$; above it the belief
is bounded but never independently certified. The decision variables (investment, emissions) depend on the
derivatives of the value function rather than its constant, so they are the least exposed to this error —
but they are not independently validated either (there is no finite-difference ground truth for the
four-state regime), and below $\xi^*$ the broken belief feeds back into them.

---

# Investment pathways

Simulated green, dirty, and R&D investment for every $\xi$ — the planner's policy response to
robustness aversion.

```{figure} figA_investment_paths.png
:label: fig-invest
:width: 100%
Green, dirty, and R&D investment, each as a percent of output, one curve per $\xi$, both calibrations.
```

**Conclusion.** More aversion smoothly lowers dirty investment; green investment is nearly unchanged;
R&D is roughly flat — a shallow peak near $\xi=0.04$ then a slight decline. All smooth in $\xi$ down to
$0.005$. (Smoothness is necessary, not sufficient — see the accuracy caveat in §6.)

---

# Distorted probabilities: damage curvature

The worst-case belief over the five damage-curvature models $\lambda_3$, baseline (red) vs distorted
(blue), for every $\xi$ — the overlaid histograms.

```{figure} figCURV_curvature_histograms.png
:label: fig-curv
:width: 100%
Damage-curvature ($\lambda_3$) distribution per $\xi$: baseline uniform (red) vs worst-case distorted (blue).
```

**Conclusion.** Aversion tilts the belief toward the most-severe damage; by $\xi\approx0.01$ it is a
degenerate point mass (all weight on the worst curvature). Being a *ratio*, it stays normalized and
keeps the correct ordering even at low $\xi$; but below $\xi\approx0.01$ the underlying distortions have
collapsed (§5), so the point mass is a ratio of near-zero quantities — the *direction* is trustworthy,
the exact low-$\xi$ weights are not.

---

# Distorted probabilities: climate and capital channels

The continuous (Brownian) worst-case distortions — climate/TCRE $h_y$ and the capital channels — for
every $\xi$.

```{figure} figD_drift_distortions.png
:label: fig-drift
:width: 100%
Worst-case drift distortions $h_y$ (climate), $h_d,h_g,h_r$ (capital), one curve per $\xi$, over 60 years.
```

**Conclusion.** The climate distortion $h_y>0$ (worst case is hotter) and the capital distortions scale
smoothly with $1/\xi$ across the whole range; the climate channel is modest relative to the damage one.

---

# Jump probabilities and the stability issue

The worst-case damage-jump probability and its density — the object whose "stability" was in question.

```{figure} figDENS_density.png
:label: fig-dens
:width: 90%
Worst-case damage-jump density vs temperature $Y$, every $\xi$. It rises with aversion to
$\xi\approx0.025$, then collapses; at $\xi=0.005$ it is flat at zero.
```

```{figure} figC_jump_stability.png
:label: fig-stab
:width: 100%
Left: cumulative damage-jump probability path per $\xi$. Middle: its year-60 value vs $1/\xi$ — rises to
a peak at $\xi^*\approx0.025$ then collapses (both calibrations). Right: tech-jump probability, for contrast.
```

The break is a *provable* error. Because the damage jump lowers welfare ($V^\ell\le V$), every
distortion $g^\ell=\exp(-\tfrac1\xi(V^\ell-V))\ge1$, so their mean — the intensity multiplier
$\bar g=\tfrac1L\sum_\ell g^\ell$ — is $\ge1$ as well: the worst-case intensity can never fall below baseline.
Computed along the path, $\bar g$ obeys this down to $\xi\approx0.025$ (values $1.30,1.35,1.33,1.22$ at
$\xi=0.05,0.04,0.03,0.025$) and then **violates it** — $\bar g=0.97$ at $\xi=0.02$, $0.07$ at $0.01$, $0$ at
$0.005$.

```{figure} figJENSEN_admissibility.png
:label: fig-jensen
:width: 85%
Worst-case damage-jump intensity multiplier $\bar g=\lambda_{\rm distorted}/\lambda_{\rm baseline}$ vs
$1/\xi$. Admissibility requires $\bar g\ge1$ (dashed line); the numerical value falls into the
forbidden region (grey) at $\xi=0.02$ and collapses to zero below — the explicit error.
```

**Conclusion.** This is the stability issue: the worst-case damage-jump distribution should keep rising
with aversion, but it peaks at $\xi^*\approx0.025$ and then collapses, violating the admissibility bound
$\bar g\ge1$ at $\xi\approx0.02$ and below. The object clears this necessary check only for $\xi\ge0.025$;
below $\xi^*$ it is provably wrong.

---

# Numerical accuracy per xi

Is the FOC/HJB error drastically different at low $\xi$? We evaluate the model's own residual with
$\log\xi$ pinned to each $\xi$, for every $\xi$.

```{figure} figACC_accuracy.png
:label: fig-acc
:width: 100%
Left: HJB residual (initial and terminal regimes) and FOC error vs $1/\xi$ — all near the $10^{-3}$
reference at every $\xi$. Right: the value gap $|V^\ell-V|$ is flat, but the jump exponent
$\tfrac1\xi|V^\ell-V|$ grows with $1/\xi$, reaching the float32-overflow scale ($\approx88$) near
$\xi\approx0.005$ — harmless in the double precision used here.
```

**Conclusion.** No — the HJB residual stays $\sim10^{-3}$ and the FOC error $\sim10^{-4}$ at **every**
$\xi$ down to $0.005$; the solve converges everywhere. The collapse in §5 is not a failure of the solve:
it is the $1/\xi$ amplification of a $\sim0.1$ mismatch between the two regimes' additive constants, inside
the jump distortion $g=\exp(-\tfrac1\xi(V^\ell-V))$. A converged residual is necessary but not sufficient:
it certifies neither the low-$\xi$ belief nor the value functions themselves — it bounds the PDE residual,
not the pointwise solution.

---

# Root cause and bottom line

Below $\xi^*\approx0.025$ the worst-case damage-jump distribution collapses (§5): the density falls to
zero and $\bar g$ drops below the admissibility floor of $1$. Pushing $\xi$ this far — well past the
range the paper uses ($\xi\ge0.05$) — exposes a **defect in the solver's design**: it never makes the before- and after-jump value functions
comparable — each is determined only up to an additive constant, and the two are never tied together.

The jump term is the change in the value function when the damage jump hits, $V^\ell-V$, and that
difference is meaningful only if the two value functions are comparable — carrying the same additive
constant. The solver
trains backward (after-jump regime first, then before-jump against it), so the jump term does couple the
two constants, but loosely: the constant enters the equation only through $\delta=0.01$, the coupling
acts only where the jump can fire, and the after-jump value function already carries its own undetermined
constant. Solved one after another, the two constants end up matched only to about $\sim0.1$. That
mismatch cancels in the policies (which use the derivatives of the value function, not its constant) and
in the curvature belief (a ratio); the worst-case jump belief is the difference $V^\ell-V$ itself, so it
carries the mismatch, amplified by $1/\xi$.

At $\xi=0.05$ a mismatch $\epsilon$ between the two constants shifts the worst-case jump intensity by
$\exp(\epsilon/\xi)$:

| constant mismatch $\epsilon$ | error in worst-case intensity at $\xi=0.05$ |
|---|---|
| $0.005$ | $+11\%$ |
| $0.01$ | $+22\%$ |
| $0.02$ | $+49\%$ |

With the $\sim0.05$–$0.1$ agreement the solver achieves between the two constants, this is a
tens-of-percent effect on the pointwise belief at $\xi=0.05$; aggregates average over it and are more
robust, but the belief carries the mismatch at every $\xi$ and is not certified anywhere. The same
mismatch times a larger $1/\xi$ swamps the real difference below $\xi^*\approx0.025$, where the belief
collapses.

```{figure} figMONEY_dominance.png
:label: fig-money
:width: 100%
The tiny mismatch between the two regimes' additive constants ($\sim0.1$), once amplified by $1/\xi$ (red,
right axis), grows to dominate: as it crosses the scale of the real difference near $\xi\approx0.025$, the
worst-case intensity $\bar g$ (blue, left axis) falls through its floor of $1$ into the forbidden region.
Below $\xi^*$ the mismatch between the constants, not the economics, sets the answer.
```

The collapse is not a convergence failure (the solver's error stays $\sim10^{-3}$ at every $\xi$, §6),
not overflow, and not threshold resolution. More computing does not remove it: nothing in the loss asks
the two constants to agree. Removing it is a question of cross-regime consistency — how to pin down the
difference $V^\ell-V$ between the separately-solved regimes — not a matter of more compute or a different
solve order; how best to do that is open.

**Bottom line.** The value function is determined only up to an additive constant, and as the solver
currently produces them the before- and after-jump regimes are not comparable. The worst-case damage-jump
belief depends on their difference, so it is never independently certified, and becomes a *provable*
failure below $\xi^*\approx0.025$. The decision variables, which depend on the derivatives of the value
function rather than its constant, are the least exposed to this error, but they are not independently
validated either. What we can state is one-sided: we prove the belief wrong below $\xi^*$, but we certify
nothing correct. Establishing cross-regime consistency is an open question we are pursuing.

