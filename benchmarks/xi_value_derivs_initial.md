# Sectoral log-capital marginal values at the initial condition, by ξ

*Reported at the calibrated initial state* **X₀ = (K=880, Z=0.70, Y=1.2, R=11.2)**.
*ξ = 0.01† · 0.05 (most averse trained) · 0.1 · 0.3 · 1.0 · 148.6 (≈ neutral).*

**† ξ = 0.01 is an extrapolation.** The network was trained on `logξ ∈ [−3, 5]`, i.e. **ξ ∈ [0.05, 148.4]**;
ξ=0.01 (`logξ = −4.6`) is below that range, so that column is the trained network's extrapolation — a direct
evaluation, read for direction only, not validated. (A retrained, in-range small-ξ solution is not yet available;
see "On reliability at smaller ξ".)

## Coordinate transform (chain rule): from `(logK, Z)` to `(logK^g, logK^d)`

The network's endogenous capital inputs are `logK = log(K^d+K^g)` and the green **share** `Z = K^g/(K^d+K^g)`.
Because `Z` is a ratio, `∂V/∂Z` is the marginal value of the *share* (reallocating dirty→green at fixed total
capital) — not directly interpretable. We re-express in the individual capital stocks
`logK^g = logK + log Z`, `logK^d = logK + log(1−Z)`. The chain rule (Jacobian of the inverse map, using
`∂logK/∂logK^g = Z`, `∂Z/∂logK^g = Z(1−Z)`, `∂logK/∂logK^d = 1−Z`, `∂Z/∂logK^d = −Z(1−Z)`) gives

$$
V_{\log K^g} = Z\,V_{\log K} + Z(1-Z)\,V_Z = Z\big[V_{\log K} + (1-Z)V_Z\big],\qquad
V_{\log K^d} = (1-Z)\big[V_{\log K} - Z\,V_Z\big],
$$

with the identity **`V_logK = V_logK^g + V_logK^d`**. These sectoral
log-capital derivatives are the primary reporting objects because they measure the
value of a one-percent change and do not depend on the units used for capital.
The investment FOCs use
`V_logK^g/Z = V_logK + (1−Z)V_Z` and
`V_logK^d/(1−Z) = V_logK − ZV_Z`, not `V_Z` alone (companion note
[xi_green_share_foc.md](xi_green_share_foc.md)).

## Table 1 — complete marginal-value report at X₀

| marginal value ∂V/∂· | ξ = 0.01† | ξ = 0.05 | ξ = 0.1 | ξ = 0.3 | ξ = 1.0 | ξ = 148.6 |
|:---|---:|---:|---:|---:|---:|---:|
| **Total productive capital** $V_{\log K}$ | 0.42425 | 0.43136 | 0.43487 | 0.43997 | 0.44344 | 0.44581 |
| **Green capital** $V_{\log K^g}$ | 0.32311 | 0.32689 | 0.32924 | 0.33280 | 0.33519 | 0.33711 |
| **Dirty capital** $V_{\log K^d}$ | 0.10114 | 0.10447 | 0.10563 | 0.10718 | 0.10825 | 0.10870 |
| **Green capital share** $V_Z$ | 0.12445 | 0.11873 | 0.11823 | 0.11817 | 0.11801 | 0.11925 |
| **Temperature** $V_Y$ | −0.00865 | −0.00869 | −0.00862 | −0.00845 | −0.00829 | −0.00828 |
| **R&D/knowledge capital** $V_{\log K^r}\equiv V_{\log R}$ | 0.03003 | 0.03163 | 0.03216 | 0.03266 | 0.03283 | 0.03303 |

### Fixed calibration and evaluation inputs

<span style="color:#B45309"><strong>Amber entries are fixed inputs, not value-function outputs:</strong></span>
<span style="color:#B45309"><strong>K=880, Z=0.70, K^g=616, K^d=264, Y=1.2, K^r=R=11.2,
A_d=0.1303, A_g=0.1085, δ=0.01, Γ_d=Γ_g=0.060, and θ_d=θ_g=16.7.</strong></span>

Here $K^r\equiv R$, so the network input $\log R$ already gives
$V_{\log K^r}$ directly; no additional scaling or coordinate transformation is
needed for that row. The level derivatives $V_{K^g}$ and $V_{K^d}$ can be recovered if needed, but are
not reported because their numerical magnitudes depend on the units used for
capital.

Because the network parameterizes $v=V+\log N(Y)$, the temperature row reports
the economic derivative $V_Y=v_Y-(\log N)_Y$. The other derivatives are
unchanged because $\log N$ depends only on temperature.

## Value level V(X₀), retained as a diagnostic

| | ξ = 0.01† | ξ = 0.05 | ξ = 0.1 | ξ = 0.3 | ξ = 1.0 | ξ = 148.6 |
|:---|---:|---:|---:|---:|---:|---:|
| Value level  V(X₀) | 3.9337 | 4.0643 | 4.1072 | 4.1543 | 4.1802 | 4.2061 |
| V(X₀) − neutral | −0.2724 | −0.1418 | −0.0989 | −0.0517 | −0.0259 | 0.0000 |

*The **absolute** level is only weakly identified (the HJB pins it through the `δ=0.01` term alone), so read the
`V(X₀) − neutral` row, not the level itself. That ξ‑difference is the welfare cost of robustness at X₀; it grows
as ξ falls (the `1/ξ` amplification), which is exactly the channel that becomes unreliable at small ξ (see below).*

## Reading

- **Signs** are as expected:
  $V_{\log K^g},V_{\log K^d},V_{\log K^r}>0$ and $V_Y<0$.
- **As ξ falls (more aversion)** both sectoral log-capital values fall. At this state their
  associated policies have lower `I_g/Y` and `I_d/Y`, while the green share of capital investment rises
  modestly. The two FOCs and the resource constraint determine that joint movement; `V_Z` by itself does not.
- Effects are modest at X₀ (Y=1.2 is below the damage threshold ŷ=2.5, so that channel is not yet active).

## On reliability at smaller ξ

Within the trained range (ξ ≥ 0.05) these marginal values are well identified — better than the value *level*,
whose weak-identification (pinned only through `δ=0.01`) a derivative differences away. The concern that grows as
ξ falls acts on the *level* and, through it, on the *worst-case belief densities*, which scale the level gap by
`1/ξ` and become unreliable below **ξ*≈0.025**. The ξ=0.01 column above is both below that boundary and outside
the trained range, so read it for direction only.

**A validated (in-range) model solution for ξ < 0.05 is not yet available.** We have an exploratory
extended-range run (trained to ξ=0.005) whose pointwise HJB residual stays of order $10^{-3}$ with no numerical
overflow, but its ξ‑response is not yet monotone, so its sub-0.05 values are not reported. A trustworthy small-ξ
solution is a retrain-and-audit task, not a direct evaluation of the current network.
