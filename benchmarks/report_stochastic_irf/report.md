# Stochastic impulse responses on the neural solution

*TwoCapital team — 2026-07-22. Method + results for Mike's point 2 ("stochastic impulse
responses as in the mitigation paper"). Everything is reconstructed from the trained neural
(DGM) solution's own outputs — no finite-difference solution and no external construct.*

---

## 1. Setup and notation

**States.** The model has four endogenous states

$$\textcolor{#1f6feb}{X}=(\textcolor{#1f6feb}{\log K},\ \textcolor{#1f6feb}{Z},\ \textcolor{#1f6feb}{Y},\ \textcolor{#1f6feb}{\log R})$$

— log total capital, green-capital share, temperature anomaly, and log knowledge (R&D) stock. The
social planner chooses investments $(\textcolor{#2da44e}{i^d},\textcolor{#2da44e}{i^g},\textcolor{#2da44e}{i^r})$; being averse to model misspecification, it
evaluates outcomes not under the baseline probabilities but under a **worst-case ("robust")
probability measure** $\tilde{\mathbb P}$, indexed by the uncertainty-aversion parameter $\xi$ (small
$\xi$ = more averse; $\xi\to\infty$ = neutral).

**Shock.** A *shock* is a marginal perturbation of the **initial** state in a chosen unit direction

$$\lambda=e_k\in\{e_{\textcolor{#1f6feb}{\log K}},\,e_{\textcolor{#1f6feb}{Z}},\,e_{\textcolor{#1f6feb}{Y}},\,e_{\textcolor{#1f6feb}{\log R}}\},$$

i.e. a unit push to log-capital, the green share, temperature, or log-R&D at $t=0$.

**What we compute (two objects).**

- **(A) the state / economic impulse response** — the response *process* $\textcolor{#2da44e}{\Lambda_t}$ that tracks how
  that initial shock has propagated into every state by date $t$ (§3). The deterministic
  forward-integration IRF we had before is its $\textcolor{#1f6feb}{\sigma}\!\to\!0$ degenerate case.
- **(B) the marginal value** of the shock, $\textcolor{#2da44e}{\partial_x V}(\textcolor{#1f6feb}{X_0})\!\cdot\!\lambda$ (its shadow price — how
  much welfare $\textcolor{#2da44e}{V}$ moves per unit of it), §4.

Both are *reconstructed from the four neural outputs*, as §2 makes precise.

---

## 2. Reconstructing the model from the four neural outputs

This is the organizing principle of the whole analysis: **every object below is a functional of the
neural network's own outputs, obtained by automatic differentiation — nothing is imposed from
outside.** Per regime the solver delivers four functions of the state $\textcolor{#1f6feb}{x}$ and aversion $\xi$,

$$\textcolor{#2da44e}{V}(\textcolor{#1f6feb}{x};\xi),\qquad \textcolor{#2da44e}{i^d}(\textcolor{#1f6feb}{x};\xi),\qquad \textcolor{#2da44e}{i^g}(\textcolor{#1f6feb}{x};\xi),\qquad \textcolor{#2da44e}{i^r}(\textcolor{#1f6feb}{x};\xi)$$

— the value function and the three investment policies. Autodiff of these networks gives, *exactly*,

$$\text{first order: }\ \textcolor{#2da44e}{\partial_x V},\ \textcolor{#2da44e}{\partial_x i^d},\ \textcolor{#2da44e}{\partial_x i^g},\ \textcolor{#2da44e}{\partial_x i^r};
\qquad\text{second order: }\ \textcolor{#2da44e}{\partial^2_x V}\ \text{(the Hessian / cross derivatives)} .$$

**Three kinds of symbol** (the same colour convention is used throughout, including Appendix A):

- **states** $\textcolor{#1f6feb}{\log K},\ \textcolor{#1f6feb}{Z},\ \textcolor{#1f6feb}{Y},\ \textcolor{#1f6feb}{\log R}$ — vary along the path;
- **neural outputs** $\textcolor{#2da44e}{V},\ \textcolor{#2da44e}{i^d},\ \textcolor{#2da44e}{i^g},\ \textcolor{#2da44e}{i^r}$ — functions of the state and $\xi$, delivered by the network;
- **calibrated constants** — fixed numbers, shown in $\text{amber}$:
  $\delta=0.01$;
  $(\alpha_j,\Gamma_j,\theta_j,\textcolor{#1f6feb}{\sigma_j})=(-0.035,\,0.060,\,16.7,\,0.01)$ for $j\in\{d,g\}$;
  $A_d=0.1303$, $A_g=0.1085$;
  $\eta=0.291$, $\bar\theta=1.86\times10^{-3}$, $\varsigma=1.2\,\bar\theta$;
  $\zeta=0$, $\psi_0=0.10583$, $\psi_1=0.5$, $\textcolor{#1f6feb}{\sigma_\kappa}=0.0078$;
  $\varrho=746.67$, $\pi=0.04$;
  and the damage/jump constants $\lambda_1,\lambda_2,\lambda_3(\ell),\,r_1=1.5,\,r_2=0.36,\,\underline y=1.5,\,\hat y=2.5$.

So in every formula below, only the (blue) states and (green) outputs vary; every $\text{amber}$
coefficient — $\alpha_j,\Gamma_j,\theta_j,\textcolor{#1f6feb}{\sigma_j},\dots$ — is a fixed calibrated number.

The **functional forms are fixed by the model too**, not learned: the capital-drift map $\textcolor{#2da44e}{\phi_j}(i)=\alpha_j+\Gamma_j\log(1+\theta_j i)$, the R&D-drift map $\textcolor{#2da44e}{\psi_r}$, the damage function $\textcolor{#1f6feb}{N}(\textcolor{#1f6feb}{Y})$, and the jump intensities $J^\ell$ are all determined once the amber constants are set. The **only** things the network supplies are the four outputs $\textcolor{#2da44e}{V},\textcolor{#2da44e}{i^d},\textcolor{#2da44e}{i^g},\textcolor{#2da44e}{i^r}$; everything else in the formulas is either a (blue) state or a fixed model object built from the amber constants — including the composite recipes $\textcolor{#2da44e}{\mu},\textcolor{#1f6feb}{\sigma},\textcolor{#2da44e}{c},\textcolor{#1f6feb}{\mathcal E}$ (left uncoloured because they are *assembled* from the primitives rather than being primitives themselves).

From these — and the model's algebraic laws of motion — we rebuild everything:

- **(i) Static quantities** (algebraic in $\textcolor{#1f6feb}{x}$ and the controls): the capital-drift map
  $\textcolor{#2da44e}{\phi_j}(\textcolor{#2da44e}{i^j})=\alpha_j+\Gamma_j\log(1+\theta_j\,\textcolor{#2da44e}{i^j})$;
  consumption $\textcolor{#2da44e}{c}=(A_d-\textcolor{#2da44e}{i^d})(1-\textcolor{#1f6feb}{Z})+(A_g-\textcolor{#2da44e}{i^g})\textcolor{#1f6feb}{Z}-\textcolor{#2da44e}{i^r}$;
  emissions $\textcolor{#1f6feb}{\mathcal E}=\eta\,A_d(1-\textcolor{#1f6feb}{Z})e^{\textcolor{#1f6feb}{\log K}}$;
  damages $\textcolor{#1f6feb}{N}(\textcolor{#1f6feb}{Y})$.
  *(Green $\textcolor{#2da44e}{\phi_j},\textcolor{#2da44e}{c}$ — they contain an output; blue $\textcolor{#1f6feb}{\mathcal E},\textcolor{#1f6feb}{N}$ — states only.)*

- **(ii) State drift $\textcolor{#2da44e}{\mu}(\textcolor{#1f6feb}{x})$ and diffusion $\textcolor{#1f6feb}{\sigma}(\textcolor{#1f6feb}{x})$** — the model's laws of motion, functions of
  the controls and the state, e.g.
  $\textcolor{#2da44e}{\mu_{\log K}}=(1-\textcolor{#1f6feb}{Z})\textcolor{#2da44e}{\phi_d}+\textcolor{#1f6feb}{Z}\textcolor{#2da44e}{\phi_g}-\tfrac12\!\big(\textcolor{#1f6feb}{\sigma_d^2}(1-\textcolor{#1f6feb}{Z})^2+\textcolor{#1f6feb}{\sigma_g^2}\textcolor{#1f6feb}{Z}^2\big)$, with diffusion
  loadings $\textcolor{#1f6feb}{\sigma_{\log K}}=\big(\textcolor{#1f6feb}{\sigma_d}(1-\textcolor{#1f6feb}{Z}),\,\textcolor{#1f6feb}{\sigma_g} \textcolor{#1f6feb}{Z},\,0,\,0\big)$, $\textcolor{#1f6feb}{\sigma_Y}=(0,0,\varsigma\textcolor{#1f6feb}{\mathcal E},0)$,
  $\textcolor{#1f6feb}{\sigma_{\log R}}=(0,0,0,\textcolor{#1f6feb}{\sigma_\kappa})$ (rows = states, columns = the four Brownian channels).

- **(iii) Robustness** — the closed-form worst-case drift distortion
  $\textcolor{#2da44e}{h^{\ast}}=-\tfrac1\xi\textcolor{#1f6feb}{\sigma'}\textcolor{#2da44e}{\partial_x V}$ (needs the value gradient $\textcolor{#2da44e}{\partial_x V}$), hence the distorted
  drift $\textcolor{#2da44e}{\tilde\mu}=\textcolor{#2da44e}{\mu}+\textcolor{#1f6feb}{\sigma} \textcolor{#2da44e}{h^{\ast}}$.

- **(iv) The variational coefficients** $D\textcolor{#2da44e}{\tilde\mu},\ D\textcolor{#1f6feb}{\sigma}$ that drive the response process (§3):

$$
D\textcolor{#1f6feb}{\sigma}=\partial_x\textcolor{#1f6feb}{\sigma}\ \ (\text{state only}),\qquad
D\textcolor{#2da44e}{\tilde\mu}=\underbrace{\partial_x\textcolor{#2da44e}{\mu}\big|_\alpha+\partial_\alpha\textcolor{#2da44e}{\mu}\,\partial_x\alpha}_{\text{policy Jacobians }\textcolor{#2da44e}{\partial_x i^j}}
\;+\;\underbrace{\partial_x\!\big(\textcolor{#1f6feb}{\sigma} \textcolor{#2da44e}{h^{\ast}}\big)}_{\text{Hessian }\textcolor{#2da44e}{\partial^2_x V}\ (\text{via }\textcolor{#2da44e}{h^{\ast}})} .
$$

  So the tangent's drift Jacobian is exactly where the **policy Jacobians** $\textcolor{#2da44e}{\partial_x i^j}$ and the
  **cross second derivatives** $\textcolor{#2da44e}{\partial^2_x V}$ enter — both delivered by autodiff.

- **(v) The marginal value** of a shock is the gradient contracted with the direction,
  $\textcolor{#2da44e}{\partial_x V}(\textcolor{#1f6feb}{X_0})\!\cdot\!\lambda$ — a single autodiff evaluation of the value network (§4).

The **only** objects that reach outside the current regime's four outputs are the jump terms — the
post-jump continuation values $\textcolor{#2da44e}{V}^\ell$ and their gradients $\textcolor{#2da44e}{\nabla V^\ell}$ (other regimes' networks),
which enter the *valuation*. That cross-regime coupling is the single place the analysis depends on the
mutual consistency of separately-trained networks, and (§8) it is exactly where it breaks.

The message: the stochastic impulse response is **not** a construct laid on top of the solution — it
*is* the four neural outputs, differentiated. **Appendix A** writes out every drift, diffusion,
distortion and tangent-Jacobian entry explicitly in terms of the outputs and their derivatives.

---

## 3. The stochastic impulse response (variational process)

**The robust measure and the state dynamics.** Robustness (Hansen–Sargent multiplier preferences)
makes the planner act *as if* a worst-case forecaster tilts the Brownian innovations by the drift
distortion $\textcolor{#2da44e}{h^{\ast}}$ of §2(iii). Under that worst-case measure $\tilde{\mathbb P}$ the state follows

$$
d\textcolor{#1f6feb}{X_t}=\textcolor{#2da44e}{\tilde\mu}(\textcolor{#1f6feb}{X_t})\,dt+\sum_{\ell}\textcolor{#1f6feb}{\sigma_\ell}(\textcolor{#1f6feb}{X_t})\,dW^\ell_t,
$$

where $W=(W^d,W^g,W^y,W^r)$ is a $\tilde{\mathbb P}$-Brownian motion and $\ell\in\{d,g,y,r\}$ indexes
the four **independent** Brownian channels — the **dirty-capital** ($d$), **green-capital** ($g$),
**temperature** ($y$), and **knowledge/R&D** ($r$) shocks ("$d$" is *dirty*, "$g$" is *green* — the two
capital sectors, not "capital" as a whole). ($\textcolor{#2da44e}{h^{\ast}}$ vanishes as $\xi\to\infty$, where
$\tilde{\mathbb P}$ is the baseline measure; here it is numerically tiny, so the response is essentially
the NN policy's own forward dynamics.)

**How the response process $\textcolor{#2da44e}{\Lambda_t}$ arises.** Push the initial state, $\textcolor{#1f6feb}{X_0}\mapsto \textcolor{#1f6feb}{X_0}+\varepsilon\lambda$,
and ask how the *whole* future path moves. To first order the date-$t$ state moves by $\varepsilon\textcolor{#2da44e}{\Lambda_t}$,
where

$$\textcolor{#2da44e}{\Lambda_t}:=\frac{\partial \textcolor{#1f6feb}{X_t}}{\partial \textcolor{#1f6feb}{X_0}}\,\lambda$$

is the directional derivative of the date-$t$ state with respect to its own initial condition, in the
shock direction $\lambda$. Differentiating the state SDE in $\textcolor{#1f6feb}{X_0}$ gives its law of motion: with the
variational coefficients $D\textcolor{#2da44e}{\tilde\mu},\,D\textcolor{#1f6feb}{\sigma}$ reconstructed in §2(iv), the **stochastic response
process** solves

$$
\boxed{\;d\textcolor{#2da44e}{\Lambda_t}=D\textcolor{#2da44e}{\tilde\mu}(\textcolor{#1f6feb}{X_t})\,\textcolor{#2da44e}{\Lambda_t}\,dt+\sum_\ell D\textcolor{#1f6feb}{\sigma_\ell}(\textcolor{#1f6feb}{X_t})\,\textcolor{#2da44e}{\Lambda_t}\,dW^\ell_t,
\qquad \textcolor{#2da44e}{\Lambda_0}=\lambda.\;}
$$

$\textcolor{#2da44e}{\Lambda_t}$ is **random** because the Jacobians are evaluated along the stochastic path $\textcolor{#1f6feb}{X_t}$ (the
dynamics are nonlinear). The impulse response of the state is $\tilde{\mathbb E}[\textcolor{#2da44e}{\Lambda_t}]$; the
response of any derived quantity $g$ (emissions, consumption, investment) is $\nabla g(\textcolor{#1f6feb}{X_t})\!\cdot\!\textcolor{#2da44e}{\Lambda_t}$.

**Relation to the deterministic IRF.** Dropping the $\sum_\ell D\textcolor{#1f6feb}{\sigma_\ell}\textcolor{#2da44e}{\Lambda_t}\,dW^\ell$ term
(set $\textcolor{#1f6feb}{\sigma}=0$) collapses $\textcolor{#2da44e}{\Lambda_t}$ to the ODE-flow Jacobian along a single mean path — the earlier
deterministic forward-integration IRF. The stochastic object adds propagation through the
*state-dependent volatility* and the resulting path dispersion (the bands in the figures).

---

## 4. The marginal value: direct gradient, and the Feynman–Kac decomposition

**The direct object (what the NN says).** The marginal value of a shock is, by definition, the value
gradient of §2(v),

$$\textcolor{#2da44e}{\partial_x V}(\textcolor{#1f6feb}{X_0})\!\cdot\!\lambda\;=\;\lim_{\varepsilon\to0}\frac{\textcolor{#2da44e}{V}(\textcolor{#1f6feb}{X_0}+\varepsilon\lambda)-\textcolor{#2da44e}{V}(\textcolor{#1f6feb}{X_0})}{\varepsilon},$$

read off in one autodiff evaluation of the value network. This is the NN's *own* shadow price of the
shock, and it is what we report (§6.2).

**The Feynman–Kac decomposition (an alternative representation).** Differentiating the robust HJB the
value solves,

$$
0=-\delta \textcolor{#2da44e}{V}+\sup_{\alpha}\Big\{\textcolor{#2da44e}{U}+\textcolor{#2da44e}{V_x'}\textcolor{#2da44e}{\mu}+\tfrac12\mathrm{tr}(\textcolor{#1f6feb}{\sigma'}\textcolor{#2da44e}{V_{xx}}\textcolor{#1f6feb}{\sigma})
-\tfrac{1}{2\xi}\textcolor{#2da44e}{V_x'}\textcolor{#1f6feb}{\sigma}\textcolor{#1f6feb}{\sigma'}\textcolor{#2da44e}{V_x}\Big\}
+\xi\sum_\ell J^\ell\Big[1-e^{-(\textcolor{#2da44e}{V}^\ell-\textcolor{#2da44e}{V})/\xi}\Big],
$$

in $\textcolor{#1f6feb}{x}$ and contracting with $\lambda$ gives a **linear Feynman–Kac PDE** for the same marginal value,
with stochastic representation (Hansen–Souganidis 2025; Barnett–Brock–Hansen–Zhang mitigation §5)

$$
\textcolor{#2da44e}{\partial_x V}(\textcolor{#1f6feb}{X_0})\!\cdot\!\lambda=\tilde{\mathbb E}\!\Big[\int_0^\infty
\textcolor{#2da44e}{\mathrm{Dis}_t}\,\big(\textcolor{#2da44e}{\Lambda_t}\!\cdot\!\textcolor{#1f6feb}{\mathcal S_t}\big)\,dt\Big],\qquad
\textcolor{#2da44e}{\mathrm{Dis}_t}=\exp\!\Big(-\!\int_0^t\!\big[\delta+\textstyle\sum_\ell \textcolor{#2da44e}{g^{\ell}_s} J^\ell_s\big]ds\Big),
\ \textcolor{#2da44e}{g^\ell}=e^{-(\textcolor{#2da44e}{V}^\ell-\textcolor{#2da44e}{V})/\xi},
$$

with the **source** $\textcolor{#1f6feb}{\mathcal S_t}$ the sum of three flows,

$$
\textcolor{#1f6feb}{\mathcal S_t}=\underbrace{\delta\,\nabla\!\big[\log \textcolor{#2da44e}{c}-\log \textcolor{#1f6feb}{N}(\textcolor{#1f6feb}{Y})\big]}_{\text{flow i: direct utility}}
+\underbrace{\sum_\ell \xi J^\ell(1-\textcolor{#2da44e}{g^\ell})\,\partial_x\!\log J^\ell}_{\text{flow ii: jump-intensity}}
+\underbrace{\sum_\ell \textcolor{#2da44e}{g^\ell} J^\ell\,\textcolor{#2da44e}{\nabla V^\ell}}_{\text{flow iii: post-jump value}} ,
$$

where $\delta$ is the discount rate, $\textcolor{#2da44e}{U}$ flow utility, $J^\ell$ the jump intensities (damage-curvature
and technology jumps), $\textcolor{#2da44e}{V}^\ell$ the post-jump continuation values, $\textcolor{#2da44e}{c}$ consumption, and $\textcolor{#1f6feb}{N}(\textcolor{#1f6feb}{Y})$ the
damage function. **Were the solution exact, this would equal the direct gradient.** Its only appeal
over the direct gradient is that it *decomposes* the value into economic flows. But it is also the one
place (via flow iii's $\textcolor{#2da44e}{\nabla V^\ell}$) that stitches together *separately-trained* regime networks — and
on this solution the two do not agree (§8). We therefore report the direct gradient as the marginal
value and treat the Feynman–Kac decomposition as a **diagnostic**.

---

## 5. Algorithm: implementation on the neural solution

We reuse the machinery in `analysis/compute_svrd_decomposition.py` and extend it in
`analysis/stochastic_irf.py`. The response process $\textcolor{#2da44e}{\Lambda_t}$ is propagated by the **exact
neural-network derivative** — a forward-mode automatic-differentiation Jacobian-vector product
(`tf.autodiff.ForwardAccumulator`) of the *same* one-step map that advances the baseline path. Concretely,
let $\Phi_{dt}(\textcolor{#1f6feb}{x})$ be the discretized update $\textcolor{#1f6feb}{x}\mapsto \textcolor{#1f6feb}{x}+\textcolor{#2da44e}{\tilde\mu}(\textcolor{#1f6feb}{x})\,dt+\sum_\ell \textcolor{#1f6feb}{\sigma_\ell}(\textcolor{#1f6feb}{x})\,dW^\ell$
(the production `nojump_step`, which itself re-evaluates the four networks and $\textcolor{#2da44e}{h^{\ast}}$ at $\textcolor{#1f6feb}{x}$). Then

$$
\textcolor{#1f6feb}{X_{t+dt}}=\Phi_{dt}(\textcolor{#1f6feb}{X_t}),\qquad
\textcolor{#2da44e}{\Lambda_{t+dt}}=\partial_{\textcolor{#1f6feb}{x}}\Phi_{dt}(\textcolor{#1f6feb}{X_t})\,\textcolor{#2da44e}{\Lambda_t}
\;=\;\textcolor{#2da44e}{\Lambda_t}+D\textcolor{#2da44e}{\tilde\mu}\,\textcolor{#2da44e}{\Lambda_t}\,dt+\sum_\ell D\textcolor{#1f6feb}{\sigma_\ell}\,\textcolor{#2da44e}{\Lambda_t}\,dW^\ell,
$$

the second equality being exactly the §3 tangent SDE. Because we differentiate $\Phi_{dt}$ itself, the
**policy Jacobians** $\textcolor{#2da44e}{\iota^m_k}$ (through $\partial_{\textcolor{#1f6feb}{x}}i^m$) and the **value Hessian** $\textcolor{#2da44e}{V_{jk}}$ (through
$\partial_{\textcolor{#1f6feb}{x}}h^{\ast}$, a forward-over-reverse Hessian-vector product) of §2(iv) enter *analytically* — no
step size $\varepsilon$, no interpolation, and none of the $1/\varepsilon$ rounding-noise floor that a finite difference
carries. Derived-quantity responses $\big(\textcolor{#2da44e}{\mathcal E},\textcolor{#2da44e}{C},\textcolor{#2da44e}{i^d},\textcolor{#2da44e}{i^g},\textcolor{#2da44e}{i^r}\big)$ are the same JVP applied to those
outputs, $\nabla(\cdot)(\textcolor{#1f6feb}{X_t})\!\cdot\!\textcolor{#2da44e}{\Lambda_t}$; marginal values are the direct autodiff gradient (§4). One
baseline path carries four tangents (one per shock coordinate) and yields all four state IRFs in one pass;
run per $\xi$ (the NN carries $\log\xi$ as a pseudo-state), $16384$ paths, $60$ years, $dt=1/12$
(`--analytic`).

*Equivalent finite-difference realization (fallback).* Setting $\textcolor{#2da44e}{\Lambda_t}=(\textcolor{#1f6feb}{X}^\varepsilon_t-\textcolor{#1f6feb}{X_t})/\varepsilon$ from a
common-random-number perturbed path $\textcolor{#1f6feb}{X}^\varepsilon_t$ (start $\textcolor{#1f6feb}{X_0}+\varepsilon e_k$, same $dW$) satisfies the same SDE
to $O(\varepsilon)$ and agrees with the analytic tangent in the mean to $\lesssim0.25\%$; but its band inherits a
$\propto 1/\varepsilon$ float32 noise floor that dominates the *weak-signal* Temperature/Technology responses (where
$\partial/\partial Y,\ \partial/\partial\log R$ are small). The analytic JVP removes that floor at the source — band
roughness on those responses drops $\sim\!5\text{–}20\times$ while the genuine damage-threshold dispersion in $Y$ is
retained — which is why it is the default here.

---

## 6. Results

$\xi\in\{0.05,\ 0.1,\ 148.6\}$ (more averse, less averse, ~neutral). Solid lines are the path-mean
response $\tilde{\mathbb E}[\textcolor{#2da44e}{\Lambda_t}]$; the shaded band is the 10–90th percentile across paths (the
stochastic dispersion) on the most-averse $\xi$.

### 6.1 State / economic impulse responses (the NN policies, forward)

These are the four neural policies propagated forward — the faithful "what the NN does" object.

**Technology (log-R&D) shock.** A positive knowledge shock decarbonizes: emissions fall (~$-0.29$ by
year 60), temperature falls (~$-0.022$), consumption rises then decays, investment tilts green early
then reallocates; the own response $\textcolor{#2da44e}{\Lambda_{\log R}}$ decays $1\to0.18$.

![Technology shock IRF](figures/stoch_irf_Technology.png)

**Temperature ($\textcolor{#1f6feb}{Y}$) shock.** *Permanent* ($\textcolor{#2da44e}{\Lambda_Y}\approx1$ throughout — $\textcolor{#1f6feb}{Y}$ has no mean reversion):
it triggers more abatement (emissions and dirty investment fall) and more R&D, consumption eventually
lower.

![Temperature shock IRF](figures/stoch_irf_Temperature.png)

**Capital (log-$K$) and green-share ($\textcolor{#1f6feb}{Z}$) shocks.**

![Capital shock IRF](figures/stoch_irf_Capital.png)

![Green-share shock IRF](figures/stoch_irf_GreenShare.png)

Across the whole aversion range the response curves nearly coincide — the behavioural response to a
marginal shock is almost independent of $\xi$ (the *muting* also seen in the welfare work).

### 6.2 Marginal values: the NN's own gradient

The shock's marginal value is the direct value gradient $\textcolor{#2da44e}{\partial_x V}(\textcolor{#1f6feb}{X_0})\!\cdot\!e_k$ (§4), in
consumption-equivalent units ($/\textcolor{#2da44e}{\mathrm{MU}_0}$, $\textcolor{#2da44e}{\mathrm{MU}_0}=\delta \textcolor{#1f6feb}{N_0}/C_0$ the initial marginal
utility):

| shock | $\xi=0.05$ | $\xi=0.1$ | $\xi=148.6$ |
|---|---|---|---|
| Capital (log-$K$) | $+1791$ | $+1793$ | $+1805$ |
| green share $\textcolor{#1f6feb}{Z}$ | $+493$ | $+488$ | $+483$ |
| temperature $\textcolor{#1f6feb}{Y}$ | $-13.4$ | $-13.0$ | $-11.4$ |
| Technology (log-$R$) | $+131$ | $+133$ | $+134$ |

Economically sensible and, again, nearly $\xi$-invariant: capital is most valuable, a greener capital
mix and more knowledge are valuable, a hotter climate is costly.

The Feynman–Kac **decomposition** of these values (below) is included as a **diagnostic only**: on this
solution it does *not* reproduce the direct gradient — e.g. at $\xi=0.05$ it returns green share
$-131$ (wrong sign vs $+493$), temperature $-35$ (vs $-13$), capital $+2274$ (vs $+1791$). Read it as
evidence about the solution (§8), not as settled magnitudes.

![Feynman–Kac decomposition (diagnostic)](figures/stoch_irf_priced.png)

---

## 7. Findings

1. **Direct marginal values are sensible and $\xi$-muted.** The NN's own $\textcolor{#2da44e}{\partial_x V}$ (§6.2) ranks
   capital ≫ green share ≫ R&D in value with a costly temperature, and barely moves across a $3000\times$
   range of $\xi$ — the marginal-valuation counterpart of the muting in the state IRFs.
2. **Decarbonization channels are visible in the state IRFs.** An R&D shock lowers emissions and
   temperature and tilts investment green; a temperature shock pulls forward abatement and R&D. These
   come straight from the NN policies and are robust.
3. **Muting.** For every shock the state responses are nearly $\xi$-invariant — the economy's response
   to a marginal shock barely depends on uncertainty aversion.
4. **A solution-consistency defect (diagnostic).** The Feynman–Kac reconstruction of the marginal value
   — which stitches the separately-trained pre- and post-jump networks through flow iii — does not
   reproduce the direct gradient, and even flips the sign of the green-share value. Since the two are
   equal for an exact HJB solution, the gap is a fresh, independent symptom of the known cross-regime
   inconsistency / weak identification of the neural solution. (Confirmed structural: unchanged under
   $dt\!:\!1/12\to1/48$ and horizon $60\to150$ yr, so neither Euler bias, truncation, nor MC noise.)

---

## 8. Method notes and honest caveats

- **Marginal value comes from the direct gradient $\textcolor{#2da44e}{\partial_x V}$, not the Feynman–Kac integral.** The
  two disagree on this solution (§6.2, §7.4); the FK integral is kept only as the cross-regime
  consistency diagnostic. Everything else is reconstructed from the four NN outputs (§2).
- The state response $\textcolor{#2da44e}{\Lambda_t}$ is under the **pre-first-jump** diffusion measure (jumps would enter a
  full-horizon extension through a realized regime switch); the state IRFs use only the current regime's
  four outputs, so they are not exposed to the cross-regime inconsistency above.
- Bands on the near-zero **investment** responses are the delicate quantity (the per-path response
  straddles zero). Propagating them by the exact NN-derivative JVP (§5) rather than a finite difference
  removes the $1/\varepsilon$ rounding floor at the source; $16384$ paths then resolve the true 10–90 band
  without post-hoc smoothing. The residual width on the Temperature investment response is *genuine*
  dispersion — paths that cross the damage threshold $\underline y=1.5$ respond differently — not noise.
- Two bugs were fixed in the reused port: a stale models path after the move to `analysis/`, and the
  $\textcolor{#1f6feb}{\log R}$ Itô sign ($+\tfrac12\textcolor{#1f6feb}{\sigma_\kappa^2}\to-\tfrac12\textcolor{#1f6feb}{\sigma_\kappa^2}$). Production `models/` is untouched.

**Reproduce.** `bash submit/submit_stochastic_irf.sh` (one Slurm job per $\xi$; set `N_PATHS`, `OUT_DIR`),
then `python3 analysis/plot_stochastic_irf.py --data-dir <OUT_DIR>`.

---

## Appendix A. Explicit reconstruction of the simulated processes from the four outputs

*Every object we simulate, written as an explicit functional of the four neural outputs
$\textcolor{#2da44e}{V},\textcolor{#2da44e}{i^d},\textcolor{#2da44e}{i^g},\textcolor{#2da44e}{i^r}$ and their autodiff derivatives. Nothing else enters.* Write the state as
$\textcolor{#1f6feb}{x}=(\textcolor{#1f6feb}{x_1},\textcolor{#1f6feb}{x_2},\textcolor{#1f6feb}{x_3},\textcolor{#1f6feb}{x_4})=(\textcolor{#1f6feb}{\log K},\ \textcolor{#1f6feb}{Z},\ \textcolor{#1f6feb}{Y},\ \textcolor{#1f6feb}{\log R})$ and the four Brownian channels as $\ell\in\{d,g,y,r\}$
(dirty capital, green capital, temperature, knowledge).

### A.0 The two processes we reconstruct

**(P1) the state process** $\textcolor{#1f6feb}{X_t}$ — to draw paths:

$$d\textcolor{#1f6feb}{x_j}=\textcolor{#2da44e}{\tilde\mu_j}(\textcolor{#1f6feb}{X_t})\,dt+\sum_{\ell} \textcolor{#1f6feb}{\sigma_{j\ell}}(\textcolor{#1f6feb}{X_t})\,dW^\ell_t,\qquad j=1,\dots,4 .$$

**(P2) the tangent / response process** $\textcolor{#2da44e}{\Lambda_t}$ — the impulse response itself:

$$d\textcolor{#2da44e}{\Lambda_t}=D\textcolor{#2da44e}{\tilde\mu}(\textcolor{#1f6feb}{X_t})\,\textcolor{#2da44e}{\Lambda_t}\,dt+\sum_\ell D\textcolor{#1f6feb}{\sigma_\ell}(\textcolor{#1f6feb}{X_t})\,\textcolor{#2da44e}{\Lambda_t}\,dW^\ell_t,\qquad \textcolor{#2da44e}{\Lambda_0}=\lambda,$$

where $D\textcolor{#2da44e}{\tilde\mu}=\big[\partial\textcolor{#2da44e}{\tilde\mu_j}/\partial \textcolor{#1f6feb}{x_k}\big]$ and $D\textcolor{#1f6feb}{\sigma_\ell}=\big[\partial\textcolor{#1f6feb}{\sigma_{j\ell}}/\partial \textcolor{#1f6feb}{x_k}\big]$ are $4\times4$.

### A.1 The four outputs and the derivatives autodiff gives

$$\textcolor{#2da44e}{V}(\textcolor{#1f6feb}{x};\xi),\quad \textcolor{#2da44e}{i^d}(\textcolor{#1f6feb}{x};\xi),\quad \textcolor{#2da44e}{i^g}(\textcolor{#1f6feb}{x};\xi),\quad \textcolor{#2da44e}{i^r}(\textcolor{#1f6feb}{x};\xi);$$
$$
\textcolor{#2da44e}{V_j}:=\frac{\partial \textcolor{#2da44e}{V}}{\partial \textcolor{#1f6feb}{x_j}},\qquad
\textcolor{#2da44e}{V_{jk}}:=\frac{\partial^2 \textcolor{#2da44e}{V}}{\partial \textcolor{#1f6feb}{x_j}\partial \textcolor{#1f6feb}{x_k}}\ (\text{Hessian}),\qquad
\textcolor{#2da44e}{\iota^{m}_k}:=\frac{\partial \textcolor{#2da44e}{i^m}}{\partial \textcolor{#1f6feb}{x_k}}\ (\text{policy Jacobian}),\ \ m\in\{d,g,r\}.
$$

### A.2 Static objects (algebraic in $\textcolor{#1f6feb}{x}$ and the controls)

$$
\textcolor{#2da44e}{\phi_j}(i)=\alpha_j+\Gamma_j\log(1+\theta_j i),\quad \textcolor{#2da44e}{\phi_j'}(i)=\frac{\Gamma_j\theta_j}{1+\theta_j i}\ \ (j\in\{d,g\});
$$
$$
\textcolor{#2da44e}{c}=(A_d-\textcolor{#2da44e}{i^d})(1-\textcolor{#1f6feb}{Z})+(A_g-\textcolor{#2da44e}{i^g})\textcolor{#1f6feb}{Z}-\textcolor{#2da44e}{i^r},\qquad
\textcolor{#1f6feb}{\mathcal E}=\eta A_d(1-\textcolor{#1f6feb}{Z})e^{\textcolor{#1f6feb}{\log K}}.
$$

### A.3 Baseline drift $\textcolor{#2da44e}{\mu}$ (before robustness)

$$
\begin{aligned}
\textcolor{#2da44e}{\mu_1}=\textcolor{#2da44e}{\mu_{\log K}}&=(1-\textcolor{#1f6feb}{Z})\textcolor{#2da44e}{\phi_d}(\textcolor{#2da44e}{i^d})+\textcolor{#1f6feb}{Z}\textcolor{#2da44e}{\phi_g}(\textcolor{#2da44e}{i^g})-\tfrac12\big[\textcolor{#1f6feb}{\sigma_d^2}(1-\textcolor{#1f6feb}{Z})^2+\textcolor{#1f6feb}{\sigma_g^2}\textcolor{#1f6feb}{Z}^2\big],\\
\textcolor{#2da44e}{\mu_2}=\textcolor{#2da44e}{\mu_{Z}}&=\textcolor{#1f6feb}{Z}(1-\textcolor{#1f6feb}{Z})\big[\textcolor{#2da44e}{\phi_g}(\textcolor{#2da44e}{i^g})-\textcolor{#2da44e}{\phi_d}(\textcolor{#2da44e}{i^d})+(1-\textcolor{#1f6feb}{Z})\textcolor{#1f6feb}{\sigma_d^2}-\textcolor{#1f6feb}{Z}\textcolor{#1f6feb}{\sigma_g^2}\big],\\
\textcolor{#1f6feb}{\mu_3}=\textcolor{#1f6feb}{\mu_{Y}}&=\bar\theta\,\textcolor{#1f6feb}{\mathcal E},\\
\textcolor{#2da44e}{\mu_4}=\textcolor{#2da44e}{\mu_{\log R}}&=-\zeta+\psi_0\,(\textcolor{#2da44e}{i^r})^{\psi_1}e^{\psi_1(\textcolor{#1f6feb}{\log K}-\textcolor{#1f6feb}{\log R})}-\tfrac12\textcolor{#1f6feb}{\sigma_\kappa^2} .
\end{aligned}
$$

### A.4 Diffusion matrix $\textcolor{#1f6feb}{\sigma}(\textcolor{#1f6feb}{x})$ (rows = states, columns = channels $d,g,y,r$)

$$
\textcolor{#1f6feb}{\sigma}(\textcolor{#1f6feb}{x})=
\begin{pmatrix}
\textcolor{#1f6feb}{\sigma_d}(1-\textcolor{#1f6feb}{Z}) & \textcolor{#1f6feb}{\sigma_g} \textcolor{#1f6feb}{Z} & 0 & 0\\[2pt]
-\textcolor{#1f6feb}{\sigma_d} \textcolor{#1f6feb}{Z}(1-\textcolor{#1f6feb}{Z}) & \textcolor{#1f6feb}{\sigma_g} \textcolor{#1f6feb}{Z}(1-\textcolor{#1f6feb}{Z}) & 0 & 0\\[2pt]
0 & 0 & \varsigma\,\textcolor{#1f6feb}{\mathcal E} & 0\\[2pt]
0 & 0 & 0 & \textcolor{#1f6feb}{\sigma_\kappa}
\end{pmatrix}.
$$

Only $\textcolor{#1f6feb}{Z}$ and $\textcolor{#1f6feb}{\mathcal E}$ (hence $\textcolor{#1f6feb}{\log K},\textcolor{#1f6feb}{Z}$) enter $\textcolor{#1f6feb}{\sigma}$ — **no control and no $\textcolor{#2da44e}{V}$** — so $D\textcolor{#1f6feb}{\sigma}$ needs only
elementary state-partials (A.7).

### A.5 Robust distortion $\textcolor{#2da44e}{h^{\ast}}$ and distorted drift $\textcolor{#2da44e}{\tilde\mu}$ — *where $\textcolor{#2da44e}{V_j}$ plugs in*

$\textcolor{#2da44e}{h^{\ast}}=-\tfrac1\xi\textcolor{#1f6feb}{\sigma'}\textcolor{#2da44e}{\partial_xV}$, i.e. $\textcolor{#2da44e}{h^{\ast}}_\ell=-\tfrac1\xi\sum_j\textcolor{#1f6feb}{\sigma_{j\ell}}\textcolor{#2da44e}{V_j}$:

$$
\textcolor{#2da44e}{h^{\ast}_d}=-\tfrac1\xi\,\textcolor{#1f6feb}{\sigma_d}(1-\textcolor{#1f6feb}{Z})\big[\textcolor{#2da44e}{V_1}-\textcolor{#1f6feb}{Z}\textcolor{#2da44e}{V_2}\big],\quad
\textcolor{#2da44e}{h^{\ast}_g}=-\tfrac1\xi\,\textcolor{#1f6feb}{\sigma_g} \textcolor{#1f6feb}{Z}\big[\textcolor{#2da44e}{V_1}+(1-\textcolor{#1f6feb}{Z})\textcolor{#2da44e}{V_2}\big],\quad
\textcolor{#2da44e}{h^{\ast}_y}=-\tfrac1\xi\,\varsigma\textcolor{#1f6feb}{\mathcal E}\,\textcolor{#2da44e}{V_3},\quad
\textcolor{#2da44e}{h^{\ast}_r}=-\tfrac1\xi\,\textcolor{#1f6feb}{\sigma_\kappa} \textcolor{#2da44e}{V_4} .
$$

Then $\textcolor{#2da44e}{\tilde\mu}=\textcolor{#2da44e}{\mu}+\textcolor{#1f6feb}{\sigma} \textcolor{#2da44e}{h^{\ast}}$, i.e. $\textcolor{#2da44e}{\tilde\mu_j}=\textcolor{#2da44e}{\mu_j}+\sum_\ell\textcolor{#1f6feb}{\sigma_{j\ell}}\textcolor{#2da44e}{h^{\ast}}_\ell$:

$$
\begin{aligned}
\textcolor{#2da44e}{\tilde\mu_1}&=\textcolor{#2da44e}{\mu_1}+\textcolor{#1f6feb}{\sigma_d}(1-\textcolor{#1f6feb}{Z})\,\textcolor{#2da44e}{h^{\ast}_d}+\textcolor{#1f6feb}{\sigma_g} \textcolor{#1f6feb}{Z}\,\textcolor{#2da44e}{h^{\ast}_g},\qquad
\textcolor{#2da44e}{\tilde\mu_2}=\textcolor{#2da44e}{\mu_2}-\textcolor{#1f6feb}{\sigma_d} \textcolor{#1f6feb}{Z}(1-\textcolor{#1f6feb}{Z})\,\textcolor{#2da44e}{h^{\ast}_d}+\textcolor{#1f6feb}{\sigma_g} \textcolor{#1f6feb}{Z}(1-\textcolor{#1f6feb}{Z})\,\textcolor{#2da44e}{h^{\ast}_g},\\
\textcolor{#2da44e}{\tilde\mu_3}&=\textcolor{#1f6feb}{\mu_3}+\varsigma\textcolor{#1f6feb}{\mathcal E}\,\textcolor{#2da44e}{h^{\ast}_y}=\bar\theta\textcolor{#1f6feb}{\mathcal E}-\tfrac1\xi\varsigma^2\textcolor{#1f6feb}{\mathcal E}^2 \textcolor{#2da44e}{V_3},\qquad
\textcolor{#2da44e}{\tilde\mu_4}=\textcolor{#2da44e}{\mu_4}+\textcolor{#1f6feb}{\sigma_\kappa} \textcolor{#2da44e}{h^{\ast}_r}=\textcolor{#2da44e}{\mu_4}-\tfrac1\xi\textcolor{#1f6feb}{\sigma_\kappa^2} \textcolor{#2da44e}{V_4} .
\end{aligned}
$$

**This is the only place the value function enters the state process — through its gradient $\textcolor{#2da44e}{V_j}$.**

### A.6 The state SDE, assembled

$$\boxed{\,d\textcolor{#1f6feb}{x_j}=\Big(\textcolor{#2da44e}{\mu_j}+\textstyle\sum_\ell\textcolor{#1f6feb}{\sigma_{j\ell}}\textcolor{#2da44e}{h^{\ast}}_\ell\Big)dt+\sum_\ell\textcolor{#1f6feb}{\sigma_{j\ell}}\,dW^\ell\,,\quad
\textcolor{#2da44e}{h^{\ast}}_\ell=-\tfrac1\xi\textstyle\sum_i\textcolor{#1f6feb}{\sigma_{i\ell}}\textcolor{#2da44e}{V_i}\,}$$

— closed in $\big(\textcolor{#2da44e}{i^d},\textcolor{#2da44e}{i^g},\textcolor{#2da44e}{i^r},\;\textcolor{#2da44e}{V_1},\dots,\textcolor{#2da44e}{V_4}\big)$: the controls and the value gradient, nothing else.

### A.7 The tangent coefficients — *where $\textcolor{#2da44e}{\iota^m_k}$ and $\textcolor{#2da44e}{V_{jk}}$ plug in*

$D\textcolor{#1f6feb}{\sigma_\ell}=[\partial\textcolor{#1f6feb}{\sigma_{j\ell}}/\partial \textcolor{#1f6feb}{x_k}]$ is elementary (only $\textcolor{#1f6feb}{Z},\textcolor{#1f6feb}{\mathcal E}$ vary), e.g.
$\partial\textcolor{#1f6feb}{\sigma_{1d}}/\partial \textcolor{#1f6feb}{Z}=-\textcolor{#1f6feb}{\sigma_d}$, $\partial\textcolor{#1f6feb}{\sigma_{2d}}/\partial \textcolor{#1f6feb}{Z}=-\textcolor{#1f6feb}{\sigma_d}(1-2\textcolor{#1f6feb}{Z})$,
$\partial\textcolor{#1f6feb}{\sigma_{3y}}/\partial \textcolor{#1f6feb}{x_k}=\varsigma\,\partial\textcolor{#1f6feb}{\mathcal E}/\partial \textcolor{#1f6feb}{x_k}$ with
$\partial\textcolor{#1f6feb}{\mathcal E}/\partial\textcolor{#1f6feb}{\log K}=\textcolor{#1f6feb}{\mathcal E}$, $\partial\textcolor{#1f6feb}{\mathcal E}/\partial \textcolor{#1f6feb}{Z}=-\textcolor{#1f6feb}{\mathcal E}/(1-\textcolor{#1f6feb}{Z})$.

$D\textcolor{#2da44e}{\tilde\mu_{jk}}$ has **three chain-rule channels** — direct, through the controls, and through $\textcolor{#2da44e}{V}$ inside $\textcolor{#2da44e}{h^{\ast}}$:

$$
\boxed{\;
\frac{\partial\textcolor{#2da44e}{\tilde\mu_j}}{\partial \textcolor{#1f6feb}{x_k}}
=\underbrace{\frac{\partial\textcolor{#2da44e}{\tilde\mu_j}}{\partial \textcolor{#1f6feb}{x_k}}\Big|_{i,\,\textcolor{#2da44e}{V}\ \text{fixed}}}_{\text{direct (state)}}
+\underbrace{\sum_{m\in\{d,g,r\}}\frac{\partial\textcolor{#2da44e}{\tilde\mu_j}}{\partial \textcolor{#2da44e}{i^m}}\,\textcolor{#2da44e}{\iota^{m}_k}}_{\text{policy Jacobian}}
+\underbrace{\sum_{i}\frac{\partial\textcolor{#2da44e}{\tilde\mu_j}}{\partial \textcolor{#2da44e}{V_i}}\,\textcolor{#2da44e}{V_{ik}}}_{\text{Hessian of }\textcolor{#2da44e}{V}}\;}
$$

- **control channel** (controls enter only $\textcolor{#2da44e}{\mu}$):
  $\partial\textcolor{#2da44e}{\mu_1}/\partial \textcolor{#2da44e}{i^d}=(1-\textcolor{#1f6feb}{Z})\textcolor{#2da44e}{\phi_d'}(\textcolor{#2da44e}{i^d})$, $\partial\textcolor{#2da44e}{\mu_1}/\partial \textcolor{#2da44e}{i^g}=\textcolor{#1f6feb}{Z}\textcolor{#2da44e}{\phi_g'}(\textcolor{#2da44e}{i^g})$,
  $\partial\textcolor{#2da44e}{\mu_2}/\partial \textcolor{#2da44e}{i^d}=-\textcolor{#1f6feb}{Z}(1-\textcolor{#1f6feb}{Z})\textcolor{#2da44e}{\phi_d'}(\textcolor{#2da44e}{i^d})$, $\partial\textcolor{#2da44e}{\mu_2}/\partial \textcolor{#2da44e}{i^g}=\textcolor{#1f6feb}{Z}(1-\textcolor{#1f6feb}{Z})\textcolor{#2da44e}{\phi_g'}(\textcolor{#2da44e}{i^g})$,
  $\partial\textcolor{#2da44e}{\mu_4}/\partial \textcolor{#2da44e}{i^r}=\psi_0\psi_1(\textcolor{#2da44e}{i^r})^{\psi_1-1}e^{\psi_1(\textcolor{#1f6feb}{\log K}-\textcolor{#1f6feb}{\log R})}$ ($\textcolor{#1f6feb}{\mu_3}$ has no control);
  these multiply the **policy Jacobian** $\textcolor{#2da44e}{\iota^m_k}$.
- **value channel**: $\partial\textcolor{#2da44e}{\tilde\mu_j}/\partial \textcolor{#2da44e}{V_i}=-\tfrac1\xi\sum_\ell\textcolor{#1f6feb}{\sigma_{j\ell}}\textcolor{#1f6feb}{\sigma_{i\ell}}$
  (e.g. $\partial\textcolor{#2da44e}{\tilde\mu_3}/\partial \textcolor{#2da44e}{V_3}=-\tfrac1\xi\varsigma^2\textcolor{#1f6feb}{\mathcal E}^2$,
  $\partial\textcolor{#2da44e}{\tilde\mu_4}/\partial \textcolor{#2da44e}{V_4}=-\tfrac1\xi\textcolor{#1f6feb}{\sigma_\kappa^2}$); these multiply the **Hessian** $\textcolor{#2da44e}{V_{ik}}$.

So the **policy Jacobians** enter only the drift-through-controls, and the **value Hessian** enters only the
robustness term — exactly the two second-order pieces autodiff provides.

### A.8 Derived-quantity responses

For any $g(\textcolor{#1f6feb}{x})$ (emissions, consumption, a control): $\ \dfrac{\partial g(\textcolor{#1f6feb}{X_t})}{\partial \textcolor{#1f6feb}{X_0}}\!\cdot\!\lambda=\nabla g(\textcolor{#1f6feb}{X_t})\!\cdot\!\textcolor{#2da44e}{\Lambda_t}$,
with e.g. $\nabla\textcolor{#1f6feb}{\mathcal E}=(\textcolor{#1f6feb}{\mathcal E},\,-\textcolor{#1f6feb}{\mathcal E}/(1-\textcolor{#1f6feb}{Z}),\,0,\,0)$ and $\nabla \textcolor{#2da44e}{i^m}=(\textcolor{#2da44e}{\iota^m_1},\dots,\textcolor{#2da44e}{\iota^m_4})$.

### A.9 Plug-in summary

| NN quantity | enters | via |
|---|---|---|
| $\textcolor{#2da44e}{i^d},\textcolor{#2da44e}{i^g},\textcolor{#2da44e}{i^r}$ (levels) | $\textcolor{#2da44e}{\mu}$, $\textcolor{#2da44e}{c}$, $\textcolor{#1f6feb}{\mathcal E}$ | A.2–A.3 |
| $\textcolor{#2da44e}{V_j}=\partial \textcolor{#2da44e}{V}/\partial \textcolor{#1f6feb}{x_j}$ | $\textcolor{#2da44e}{h^{\ast}}$ → $\textcolor{#2da44e}{\tilde\mu}$; **and** the marginal value $\textcolor{#2da44e}{\partial_xV}\!\cdot\!\lambda$ | A.5 |
| $\textcolor{#2da44e}{\iota^m_k}=\partial \textcolor{#2da44e}{i^m}/\partial \textcolor{#1f6feb}{x_k}$ | $D\textcolor{#2da44e}{\tilde\mu}$, control channel | A.7 |
| $\textcolor{#2da44e}{V_{jk}}=\partial^2\textcolor{#2da44e}{V}/\partial \textcolor{#1f6feb}{x_j}\partial \textcolor{#1f6feb}{x_k}$ | $D\textcolor{#2da44e}{\tilde\mu}$, value/robustness channel | A.7 |
| (state only, no NN) | $\textcolor{#1f6feb}{\sigma}$, $D\textcolor{#1f6feb}{\sigma}$ | A.4, A.7 |
| $\textcolor{#2da44e}{V}^\ell,\textcolor{#2da44e}{\nabla V^\ell}$ (**other regimes**) | jump terms / FK decomposition **only** | §4, §8 |

### A.10 What the code does

`analysis/stochastic_irf.py` (`--analytic`) advances the tangent $\textcolor{#2da44e}{\Lambda_t}$ by the forward-mode
autodiff Jacobian-vector product of the one-step map $\Phi_{dt}$ (`tf.autodiff.ForwardAccumulator` over the
production `nojump_step`): $\textcolor{#2da44e}{\Lambda_{t+dt}}=\partial_{\textcolor{#1f6feb}{x}}\Phi_{dt}(\textcolor{#1f6feb}{X_t})\,\textcolor{#2da44e}{\Lambda_t}$. Because $\Phi_{dt}$ re-evaluates the four
networks and $\textcolor{#2da44e}{h^{\ast}}$ at $\textcolor{#1f6feb}{X_t}$, differentiating it *is* A.7 — the policy Jacobians $\textcolor{#2da44e}{\iota^m_k}$ and the value
Hessian $\textcolor{#2da44e}{V_{jk}}$ (a forward-over-reverse Hessian-vector product) enter exactly, with no $\varepsilon$ and no
finite-difference noise floor. Derived responses are the same JVP applied to $\big(\textcolor{#2da44e}{\mathcal E},\textcolor{#2da44e}{C},\textcolor{#2da44e}{i^d},\textcolor{#2da44e}{i^g},\textcolor{#2da44e}{i^r}\big)$
(A.8). The equivalent common-random-number finite difference $\textcolor{#2da44e}{\Lambda_t}=(\textcolor{#1f6feb}{X}^\varepsilon_t-\textcolor{#1f6feb}{X_t})/\varepsilon$ (default off)
reproduces A.7 to $O(\varepsilon)$ and agrees in the mean, but reintroduces the $1/\varepsilon$ floor on the weak-signal
responses. The state process and the marginal value use only A.2–A.6, i.e. the current regime's four outputs.
