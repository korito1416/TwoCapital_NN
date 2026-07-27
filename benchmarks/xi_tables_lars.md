# Uncertainty aversion (ξ): results at the calibrated initial state

## Setup

Every entry below is a **direct network evaluation at the same initial state**
$X_0=(K,Z,Y,R)=(880,0.70,1.2,11.2)$. There is no path simulation or path averaging in
these tables. Holding $X_0$ fixed isolates the solution's response to $\xi$.

<span style="color:#B45309"><strong>Amber entries are fixed calibration/evaluation inputs, not network outputs:</strong></span>
<span style="color:#B45309"><strong>$K=880$, $Z=0.70$, $Y=1.2$, $K^r=R=11.2$,
$A_d=0.1303$, $A_g=0.1085$, $\delta=0.01$,
$\Gamma_d=\Gamma_g=0.060$, and $\theta_d=\theta_g=16.7$.</strong></span>

At this state, output is

$$
\mathrm{Output}_0
=K\big[A_d(1-Z)+A_gZ\big]
=101.2.
$$

The $\xi=0.01$ column is marked † because it is a direct extrapolation outside the
network's training range, $\log\xi\in[-3,5]$ (approximately
$\xi\in[0.05,148.4]$). It should be read as a stress test, not as an equally
validated solution.

## Table 1 — Investment and consumption relative to output at $X_0$

| share of output | ξ = 0.01† | ξ = 0.05 | ξ = 0.1 | ξ = 0.3 | ξ = 1.0 | ξ = 148.6 | Δ (0.05 − neutral) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Consumption $C/\mathrm{Output}$ | 0.4181 | 0.4114 | 0.4087 | 0.4050 | 0.4029 | 0.4013 | **+0.0101** |
| Dirty investment $I_d/\mathrm{Output}$ | 0.1012 | 0.1023 | 0.1031 | 0.1046 | 0.1059 | 0.1060 | −0.0037 |
| Green investment $I_g/\mathrm{Output}$ | 0.4415 | 0.4431 | 0.4443 | 0.4459 | 0.4467 | 0.4480 | −0.0049 |
| R&D investment $I_r/\mathrm{Output}$ | 0.0393 | 0.0432 | 0.0439 | 0.0445 | 0.0445 | 0.0447 | −0.0015 |
| **Total** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | — |

At the common initial state, greater robustness (smaller $\xi$) shifts output from
all three investment categories into consumption.

## Table 2 — Economically interpretable marginal values at $X_0$

This table retains all of the network's economically relevant state derivatives
and adds the sectoral capital derivatives obtained from the $(\log K,Z)$
transformation. Log-capital derivatives measure the value of a proportional
change in the corresponding stock and are invariant to the units in which capital
is measured.

| marginal value | ξ = 0.01† | ξ = 0.05 | ξ = 0.1 | ξ = 0.3 | ξ = 1.0 | ξ = 148.6 | Δ (0.05 − neutral) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Total productive capital $V_{\log K}$ | 0.42425 | 0.43136 | 0.43487 | 0.43997 | 0.44344 | 0.44581 | −0.01445 |
| Green capital $V_{\log K^g}$ | 0.32311 | 0.32689 | 0.32924 | 0.33280 | 0.33519 | 0.33711 | −0.01022 |
| Dirty capital $V_{\log K^d}$ | 0.10114 | 0.10447 | 0.10563 | 0.10718 | 0.10825 | 0.10870 | −0.00423 |
| Green capital share $V_Z$ | 0.12445 | 0.11873 | 0.11823 | 0.11817 | 0.11801 | 0.11925 | −0.00052 |
| Temperature $V_Y$ | −0.00865 | −0.00869 | −0.00862 | −0.00845 | −0.00829 | −0.00828 | −0.00041 |
| R&D/knowledge capital $V_{\log K^r}\equiv V_{\log R}$ | 0.03003 | 0.03163 | 0.03216 | 0.03266 | 0.03283 | 0.03303 | −0.00140 |

The underlying state coordinates are $(\log K,Z,Y,\log R)$. The green and dirty
capital rows are the chain-rule transformation of $(V_{\log K},V_Z)$ and satisfy
$V_{\log K}=V_{\log K^g}+V_{\log K^d}$ in every column. Because the network
parameterizes the transformed value $v=V+\log N(Y)$, the reported temperature
entry is the economic derivative $V_Y=v_Y-(\log N)_Y$; the capital, share, and
knowledge derivatives are unaffected by this correction.

## How these values enter the investment FOCs

Because the network uses $\log K=\log(K^d+K^g)$ and
$Z=K^g/(K^d+K^g)$,

$$
V_{\log K^g}
=Z\big[V_{\log K}+(1-Z)V_Z\big],
\qquad
V_{\log K^d}
=(1-Z)\big[V_{\log K}-ZV_Z\big].
$$

Writing the knowledge stock as $K^r\equiv R$,

$$
V_{\log K^r}=V_{\log R}.
$$

The green and dirty entries therefore require a coordinate transformation,
whereas the R&D/knowledge entry is already a derivative with respect to its
sectoral log stock in the network.

Therefore the green and dirty investment FOCs can be written as

$$
\frac{\delta}{C/K}
=\phi_g'(i^g)\frac{V_{\log K^g}}{Z},
\qquad
\frac{\delta}{C/K}
=\phi_d'(i^d)\frac{V_{\log K^d}}{1-Z},
$$

where

$$
\phi_j'(i^j)=\frac{\Gamma_j\theta_j}{1+\theta_j i^j}.
$$

The R&D FOC is

$$
\frac{\delta}{C/K}
=\psi_r'(i^r)V_{\log K^r}
=\psi_r'(i^r)V_{\log R}.
$$

Strictly speaking, there is no separate “FOC for $Z$”: $Z$ is a state, not a
control. The relevant objects are the dirty and green investment FOCs above.

## Answer to the apparent $V_Z$ puzzle

$V_Z$ values a **composition change** from dirty to green capital while holding
total capital fixed. It is related to the two sectoral marginal values by

$$
V_Z
=\frac{V_{\log K^g}}{Z}
-\frac{V_{\log K^d}}{1-Z}.
$$

It does not determine green investment on its own. Green investment is determined
by $V_{\log K^g}/Z$ jointly with marginal utility of consumption and the
adjustment-cost slope in the green FOC.

At $X_0$, stronger robustness lowers both sectoral accumulation values. The dirty
value falls more in percentage terms, so the **composition** of investment becomes
slightly greener; nevertheless, the level of green investment relative to output
falls because the planner reduces total accumulation and consumes more. Thus a
higher composition value $V_Z$ is fully consistent with a lower
$I_g/\mathrm{Output}$.

## Method note

The table is produced by evaluating the trained pre-damage/pre-tech value and
policy networks once at $X_0$ for each displayed $\xi$. The rows in Table 1 sum to
one by the resource constraint; this accounting identity is distinct from a
pointwise check of the investment FOC residuals.
