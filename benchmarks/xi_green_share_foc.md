# Green share $Z$ and the investment first-order conditions

## Interpretation of $Z$

The network state

$$
Z=\frac{K^g}{K^d+K^g}
$$

is the green share of total capital. Therefore $V_Z$ is the value of changing the
green/dirty **composition** while holding total capital fixed. It is neither the
marginal value of green capital alone nor a sufficient statistic for green
investment.

Strictly speaking, $Z$ has no separate first-order condition because it is a state,
not a control. The relevant conditions are the FOCs for dirty and green
investment, expressed below in the network's $(\log K,Z)$ coordinates.

## Coordinate transformation

With $K=K^d+K^g$,

$$
V_{\log K^g}
=Z\big[V_{\log K}+(1-Z)V_Z\big],
\qquad
V_{\log K^d}
=(1-Z)\big[V_{\log K}-ZV_Z\big].
$$

Equivalently,

$$
V_Z
=\frac{V_{\log K^g}}{Z}
-\frac{V_{\log K^d}}{1-Z},
\qquad
V_{\log K}=V_{\log K^g}+V_{\log K^d}.
$$

$V_{\log K^g}$ and $V_{\log K^d}$ are the economically useful reporting
objects: each measures the value of a one-percent increase in the corresponding
capital stock.

## Investment FOCs

Let

$$
\frac{C}{K}
=(A_d-i^d)(1-Z)+(A_g-i^g)Z-i^r
$$

and

$$
\phi_j'(i^j)=\frac{\Gamma_j\theta_j}{1+\theta_j i^j}.
$$

Then

$$
\boxed{
\frac{\delta}{C/K}
=\phi_g'(i^g)\frac{V_{\log K^g}}{Z}
=\frac{\Gamma_g\theta_g}{1+\theta_g i^g}
\frac{V_{\log K^g}}{Z}
}
\qquad\text{(green)}
$$

and

$$
\boxed{
\frac{\delta}{C/K}
=\phi_d'(i^d)\frac{V_{\log K^d}}{1-Z}
=\frac{\Gamma_d\theta_d}{1+\theta_d i^d}
\frac{V_{\log K^d}}{1-Z}
}
\qquad\text{(dirty)}.
$$

For R&D, writing the knowledge stock as $K^r\equiv R$,

$$
\boxed{
\frac{\delta}{C/K}
=\psi_r'(i^r)V_{\log K^r}
=\psi_r'(i^r)V_{\log R}
}.
$$

## Why a larger $V_Z$ need not imply larger $I_g/\mathrm{Output}$

$V_Z$ compares green with dirty capital at fixed total capital. By contrast, the
green investment FOC depends on the absolute green accumulation value
$V_{\log K^g}/Z$, marginal utility $\delta/(C/K)$, and the adjustment-cost slope.

At the calibrated initial state, stronger robustness lowers both
$V_{\log K^g}$ and $V_{\log K^d}$. The dirty value falls more in percentage terms,
which raises the relative appeal of green capital and makes the investment mix
slightly greener. But total accumulation falls and consumption rises, so
$I_g/\mathrm{Output}$ still declines. There is no contradiction: $V_Z$ is a
composition wedge, not a green-investment demand schedule.

The corresponding initial-state tables are in
[xi_tables_lars.md](xi_tables_lars.md).
