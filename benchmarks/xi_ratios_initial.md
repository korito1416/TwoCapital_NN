# Investment and consumption relative to output, at the initial condition, by ξ

*Reported at the calibrated initial state* **X₀ = (K=880, Z=0.70, Y=1.2, R=11.2)**, with
`Output = [A_d(1−Z) + A_g Z]·K = 101.2` (capital/output units). *ξ = 0.01† · 0.05 (most averse trained) · 0.1 · 0.3 ·
1.0 · 148.6 (≈ uncertainty-neutral).*

**These are a direct evaluation of the trained control networks at X₀** — `i_d, i_g, i_r` are NN outputs and
consumption is the residual `c = (A_d−i_d)(1−Z) + (A_g−i_g)Z − i_r`. No simulation, no jumps, no impulse
response is involved, so none of the path/first-variation questions bear on these numbers.

**† ξ = 0.01 is an extrapolation** — the network was trained on `logξ ∈ [−3, 5]` (**ξ ∈ [0.05, 148.4]**), and
ξ=0.01 (`logξ = −4.6`) is below that range. It is a direct evaluation as requested (no retraining), but should be
read as extrapolated, not validated. (A retrained, in-range small-ξ solution is not yet available.)

## Table 1 — shares of output

| share of output | ξ = 0.01† | ξ = 0.05 | ξ = 0.1 | ξ = 0.3 | ξ = 1.0 | ξ = 148.6 |
|:---|---:|---:|---:|---:|---:|---:|
| Consumption  C/Y | 0.4181 | 0.4114 | 0.4087 | 0.4050 | 0.4029 | 0.4013 |
| Dirty investment  I_d/Y | 0.1012 | 0.1023 | 0.1031 | 0.1046 | 0.1059 | 0.1060 |
| Green investment  I_g/Y | 0.4415 | 0.4431 | 0.4443 | 0.4459 | 0.4467 | 0.4480 |
| R&D investment  I_r/Y | 0.0393 | 0.0432 | 0.0439 | 0.0445 | 0.0445 | 0.0447 |
| **Total** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Table 2 — the same quantities as levels (output units, Output = 101.2)

| level | ξ = 0.01† | ξ = 0.05 | ξ = 0.1 | ξ = 0.3 | ξ = 1.0 | ξ = 148.6 |
|:---|---:|---:|---:|---:|---:|---:|
| Consumption  C | 42.33 | 41.65 | 41.37 | 41.01 | 40.79 | 40.63 |
| Dirty investment  I_d | 10.24 | 10.35 | 10.44 | 10.59 | 10.72 | 10.73 |
| Green investment  I_g | 44.69 | 44.86 | 44.98 | 45.14 | 45.23 | 45.35 |
| R&D investment  I_r | 3.97 | 4.37 | 4.45 | 4.50 | 4.50 | 4.52 |

## Reading

Monotone in ξ across the trained range: more uncertainty aversion (smaller ξ) → slightly **more consumption**
and slightly **less investment** in every category, including green. The effect is modest — about 1 percentage
point of output shifts from investment to consumption between ξ=148.6 and ξ=0.05 (C/Y 0.4013 → 0.4114). The
extrapolated ξ=0.01 column continues the same direction (C/Y 0.4181; the sharpest move is R&D, I_r/Y 0.0447 →
0.0393).

**Note on green investment vs. `V_Z`.** The green-investment *share* falls as ξ falls (I_g/Y 0.4480 → 0.4415),
even though the marginal value of the green *share* `V_Z` rises. These are not in conflict: green investment is
governed by `V_logK^g/Z = V_logK + (1−Z)V_Z`, together with marginal utility
and the adjustment-cost slope, not by `V_Z` alone. See the companion note
[green-share FOC transform](xi_green_share_foc.md).
