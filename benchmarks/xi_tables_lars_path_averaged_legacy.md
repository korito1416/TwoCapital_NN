# Archived path-averaged analysis: uncertainty aversion (ξ)

## Setup

Objects are read from the trained neural solution (initial regime, *pre‑damage / pre‑tech*) and
**averaged over the economically‑visited states** of the model's own simulated economy — 256 paths
started at the calibrated initial state (K = 880, Z = 0.70, Y = 1.2, R = 11.2), run 60 years under the
robust (ξ‑distorted) drift, sampled yearly. Output is $Y = [A_d(1-Z) + A_g Z]\,K$; at the initial
state $Y = 101.2$ (capital units). ξ enters the network as the $\log\xi$ pseudo‑state, so one solution
delivers every column: **ξ = 0.05** (most averse) · 0.1 · 0.3 · 1.0 · **148.6** (≈ uncertainty‑neutral).

*Why the path average rather than the single initial state:* at Y = 1.2 the economy sits **below the
damage‑jump threshold** $\underline y = 1.5$, where the damage‑belief term is identically zero — so an
initial‑state cut omits that channel and understates ξ. Temperature rises through 1.5 by ≈ year 20
(reaching ≈ 2.1 by year 60), so the path average captures the damage channel once it is active.

---

## Table 1 — Investment and consumption relative to output  (path‑averaged)

| share of output | ξ = 0.05 | ξ = 0.1 | ξ = 0.3 | ξ = 1.0 | ξ = 148.6 | Δ (0.05 − neutral) |
|:---|---:|---:|---:|---:|---:|---:|
| Consumption  $C/Y$ | 0.4481 | 0.4450 | 0.4415 | 0.4389 | 0.4367 | **+0.0114** |
| Dirty investment  $I_d/Y$ | 0.0630 | 0.0640 | 0.0647 | 0.0658 | 0.0662 | −0.0032 |
| Green investment  $I_g/Y$ | 0.4671 | 0.4688 | 0.4716 | 0.4731 | 0.4748 | −0.0077 |
| R&D investment  $I_r/Y$ | 0.0217 | 0.0221 | 0.0222 | 0.0222 | 0.0223 | −0.0006 |
| **Total (market clears)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | — |

Monotone in ξ throughout: more robustness (smaller ξ) shifts ≈ 1 % of output from investment — *all three
categories, including R&D* — into consumption.

## Table 2 — Marginal values  $\partial V/\partial(\text{state})$  (path‑averaged)

| marginal value | ξ = 0.05 | ξ = 0.1 | ξ = 0.3 | ξ = 1.0 | ξ = 148.6 | Δ (0.05 − neutral) |
|:---|---:|---:|---:|---:|---:|---:|
| Capital  $V_{\log K}$ | 0.3950 | 0.3984 | 0.4026 | 0.4062 | 0.4090 | −0.0141 |
| Green share  $V_Z$ | 0.1098 | 0.1093 | 0.1089 | 0.1091 | 0.1103 | −0.0006 |
| Temperature  $V_Y$ | −0.00343 | −0.00332 | −0.00310 | −0.00292 | −0.00284 | −0.00059 |
| Knowledge  $V_{\log R}$ | 0.0319 | 0.0324 | 0.0328 | 0.0330 | 0.0332 | −0.0014 |

---

## Table 3 — Where the ξ‑effect concentrates: initial vs. late

The ξ = 0.05‑minus‑neutral gap, at the initial state (Y = 1.2, damage channel *off*) vs. late on the
path (Y ≈ 2.1, damage channel *on*). The **climate‑relevant** responses roughly double once temperature
crosses the damage threshold; the rest are flat.

| gap (0.05 − neutral) | at initial  Y = 1.2 | at late  Y ≈ 2.1 |
|:---|---:|---:|
| Consumption  $C/Y$ | +0.0101 | +0.0124 |
| Dirty investment  $I_d/Y$ | −0.0037 | −0.0023 |
| **Green investment  $I_g/Y$** | **−0.0049** | **−0.0098** |
| R&D investment  $I_r/Y$ | −0.0015 | −0.0004 |
| Capital  $V_{\log K}$ | −0.0145 | −0.0141 |
| **Temperature  $V_Y$** | **−0.00041** | **−0.00071** |
| Knowledge  $V_{\log R}$ | −0.0014 | −0.0015 |

## Table 4 — Initial‑state levels and ratios (the level twin)

At the calibrated start (output $Y = 101.2$, capital units), each quantity in both forms.

| quantity | level (0.05) | level (0.1) | level (neutral) | share (0.05) | share (0.1) | share (neutral) |
|:---|---:|---:|---:|---:|---:|---:|
| Consumption  $C$ | 41.65 | 41.37 | 40.63 | 0.4114 | 0.4087 | 0.4013 |
| Dirty investment  $I_d$ | 10.35 | 10.44 | 10.73 | 0.1023 | 0.1031 | 0.1060 |
| Green investment  $I_g$ | 44.86 | 44.98 | 45.35 | 0.4431 | 0.4443 | 0.4480 |
| R&D investment  $I_r$ | 4.37 | 4.45 | 4.52 | 0.0432 | 0.0439 | 0.0447 |
| Output  $Y$ | 101.2 | 101.2 | 101.2 | 1.0000 | 1.0000 | 1.0000 |

---

## Reading

- **Every direction is economically sensible.** More robustness (smaller ξ) → consume more, invest less
  in every category; in particular **less R&D** — the robust planner discounts the *good* breakthrough
  ($g^{\ell''}=\exp(-(V^{\ell''}-V)/\xi)<1$) — and a **more negative marginal value of temperature**
  (climate fear).
- **The economy decarbonizes over the horizon** (ξ = 0.05): $I_d/Y$ falls 0.102 → 0.035, $I_g/Y$ rises
  0.443 → 0.480, R&D tapers 0.043 → 0.011 as knowledge accumulates, and consumption rises 0.411 → 0.474.
  The averages in Tables 1–2 summarize this whole transition.
- **The ξ‑response is present, correctly signed, and state‑dependent, but modest** — ≈ 1–3 % on the
  allocation and a few percent on the marginal values. It strengthens exactly where it should: the
  green‑investment and temperature responses roughly double once temperature crosses the damage
  threshold (Table 3). So the channel is *live*, not dead; the open question is whether its magnitude
  should be larger — the finite‑difference cross‑check in the three‑state regimes is meant to settle that.

## Method notes

- **Market clearing** is the resource constraint $C/Y + I_d/Y + I_g/Y + I_r/Y = 1$; consumption is the
  residual $C/K=(A_d-i_d)(1-Z)+(A_g-i_g)Z-i_r$, positive throughout — the rows summing to 1 confirm the
  accounting, and $C>0$ confirms feasibility.
- **Marginal values are level‑robust:** the welfare *level* is only weakly pinned, but these are
  *derivatives*, so the ξ‑differences reported here are identified (the additive level constant cancels).
- Controls are $i_d=I_d/K^d$, $i_g=I_g/K^g$, $i_r=I_r/K$ from the policy networks; the shares above convert
  to output units.
- **What is and isn't coupled.** These tables query **only the pre‑damage / pre‑tech network**. That
  network was *trained* against an HJB that references the other three regimes' value functions, so its
  policies and marginal values already embed the belief‑weighted damage‑ and tech‑jump valuation (this is
  why the marginal values respond as $Y$ nears the damage threshold). But the simulated path uses the
  **no‑jump drift — it does not fire jumps or switch regimes**, and the other three networks are not
  evaluated here. A fully coupled simulation that lets the economy jump across all four regimes (the next
  IRF step) is a separate build and would shift the visited‑state distribution.
