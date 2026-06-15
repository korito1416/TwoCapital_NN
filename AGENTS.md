# Model Specification: Climate Change, Innovation, and Uncertainty

> Reference for solving the social planner's HJB system numerically.
> Source: Barnett, Brock, Hansen, Hu, Huang — *A Deep Learning Analysis of Climate Change, Innovation, and Uncertainty*.
> This document prioritizes the **mathematical specification** needed to code the solver. Economic narrative is compressed to the minimum needed to interpret terms.

---

## 1. Overview

A continuous-time, infinite-horizon social planner chooses investment in three capital types under model-uncertainty aversion. The solution is a recursive Markovian equilibrium characterized by a coupled system of Hamilton–Jacobi–Bellman (HJB) PDEs, one per *jump regime*. Two jump processes (climate-damage curvature, green-technology productivity) connect the regimes; aversion to misspecification adds endogenous nonlinear terms (drift distortion `h`, jump distortions `g`).

**Endogenous state variables** (true states with dynamics):

| Symbol | Meaning |
|---|---|
| `logK = log(K^d + K^g)` | log total capital |
| `Z = K^g / (K^d + K^g)` | green capital share, `Z ∈ (0,1)` |
| `Y` | global mean temperature anomaly (°C above preindustrial) |
| `logR` | log knowledge (R&D) capital stock |

**Controls:** `i^d` (dirty investment rate), `i^g` (green investment rate), `i^r` (R&D investment-to-total-capital rate). Misspecification minimizers `h` (Brownian drift distortion), `g^ℓ, g^{ℓ'}, g^{ℓ''}` (jump intensity distortions) are solved in closed form (see §7).

**Pseudo-states** (sampled, no dynamics — used as NN inputs to enforce monotonicity / share structure across the parameterized PDE family):
- `ξ` — model-uncertainty aversion parameter.
- `ℓ` — damage-curvature realization index.

---

## 2. Climate Dynamics

Temperature anomaly (linear-in-cumulative-emissions / TCRE approximation, stochastic variant):

$$
dY_t = \mathcal{E}_t\left(\bar\theta\, dt + \varsigma\, dW^y_t\right)
$$

Emissions come **only** from dirty capital:

$$
\mathcal{E}_t = \eta\, A_d\, K^d_t = \eta\, A_d\, (1-Z_t)\, K_t
$$

- `\bar\theta` = mean TCRE (climate sensitivity) across 144 carbon–temperature model pairs.
- `\varsigma` = temperature volatility loading.
- Carbon emissions have a **permanent** effect on `Y` (no mean reversion).

---

## 3. Climate Damages

Damages `N` scale capital/output/consumption proportionally and enter utility as `-δ log N(Y)`. Piecewise log-quadratic with a **curvature jump** at threshold `ŷ` (the temperature when the damage jump fires):

$$
\log N(y) =
\begin{cases}
\lambda_1 y + \tfrac{\lambda_2}{2} y^2, & 0 \le y \le \hat y \\[4pt]
\lambda_1 y + \tfrac{\lambda_2}{2}\hat y^2 + \tfrac{\lambda_3(\ell)}{2}(y-\hat y)^2 + (\text{matching terms}), & y \ge \hat y
\end{cases}
$$

- Before the damage jump: each curvature value `λ_3(ℓ)`, `ℓ = 1,…,L`, has equal prior weight `1/L`.
- The jump reveals the true `λ_3(ℓ)`; **earlier jump (smaller ŷ) ⇒ more severe damages**.
- The added piece for `y ≥ ŷ` is constructed for value/level matching at `ŷ`; conceptually it superimposes `+ (λ_3(ℓ)/2)(y−ŷ)^2` extra convexity on the pre-jump quadratic. (The paper's printed matching constants contain typos; implement value+slope continuity at `ŷ`.)

**Damage jump intensity** (increasing in `y`, localized to `[y̲, ȳ]`):

$$
\mathcal{J}^{\ell}_n(y) = \frac{1}{L}\,\mathcal{J}_n(y), \qquad
\mathcal{J}_n(y) = r_1\!\left(\exp\!\left(\tfrac{r_2}{2}(y-\underline y)^2\right) - 1\right)\mathbf{1}_{\{y \ge \underline y\}}
$$

---

## 4. Production and Capital

Two sectors, perfectly substitutable output, AK technology:

$$
F_j(K^j_t) = A_j K^j_t, \qquad j \in \{d, g\}
$$

Each sector's capital evolves with log adjustment costs and Brownian shocks:

$$
dK^j_t = K^j_t\big[\alpha_j + \Gamma_j \log(1 + \theta_j i^j_t)\big]\,dt + \sigma_j K^j_t\, dW^j_t
$$

Define the **per-sector capital drift** (used throughout the HJBs):

$$
\phi_j(i^j) \equiv \alpha_j + \Gamma_j \log(1 + \theta_j i^j), \qquad
\phi_j'(i^j) = \frac{\Gamma_j \theta_j}{1 + \theta_j i^j}
$$

### Transformed states (the ones you actually solve in)

With `logK = log(K^d + K^g)` and `Z = K^g/(K^d+K^g)`, Itô gives:

$$
\begin{aligned}
d\log K_t =\;& \Big[(1-Z)\phi_d(i^d) + Z\,\phi_g(i^g) - \tfrac12\big(\sigma_d^2(1-Z)^2 + \sigma_g^2 Z^2\big)\Big]dt \\
& + (1-Z)\sigma_d\, dW^d_t + Z\sigma_g\, dW^g_t
\end{aligned}
$$

$$
\begin{aligned}
dZ_t =\;& Z(1-Z)\Big[\phi_g(i^g) - \phi_d(i^d) + (1-Z)\sigma_d^2 - Z\sigma_g^2\Big]dt \\
& + Z(1-Z)\sigma_g\, dW^g_t - Z(1-Z)\sigma_d\, dW^d_t
\end{aligned}
$$

**Key asymmetries:**
- Only sector `d` emits (`\mathcal{E}_t = \eta A_d K^d`).
- Initially `A_d > A_g` (dirty more productive). Emissions are "sticky": reducing them requires shrinking `K^d`, which depreciates slowly and is costly to disinvest.

---

## 5. Knowledge Capital and Green-Technology Jump

R&D builds knowledge capital `R`:

$$
dR_t = -\zeta R_t\, dt + \psi_0 (I^r_t)^{\psi_1}(R_t)^{1-\psi_1}\, dt + R_t \sigma_r\, dW^r_t
$$

with `i^r = I^r/(K^d + K^g)`. In `logR` the drift term is denoted `ψ_r(i^r)`; from the appendix it takes the form

$$
\psi_r(i^r) = -\zeta + \psi_0 (i^r)^{\psi_1}\exp\!\big(\psi_1(\log K - \log R)\big) - \tfrac12 \sigma_r^2 \quad(\text{plus the } +\tfrac12\sigma_r^2 \text{ Itô term handled explicitly}),
$$

so that the HJB carries `ψ_r(i^r) V_{logR} + (σ_r^2/2) V_{logR,logR}`. Treat `ψ_r'(i^r) = ∂(logR drift)/∂i^r` in the investment FOC.

### Technology jump structure (three productivity states)

- `A_g` — pre-jump green productivity (`A_g < A_d`).
- `A_g(ℓ')` — **intermediate "catch-up"**, set equal to `A_d` (transient state; can still jump again).
- `A_g(ℓ'')` — **"breakthrough"**, absorbing, with `A_g(ℓ'') > A_g(ℓ')`.

Conditional on a tech jump from the pre-jump state: prob `π` to breakthrough `ℓ''`, prob `1−π` to intermediate `ℓ'`. From the intermediate state, a jump goes to `ℓ''` with prob 1.

**Tech jump intensities** (affine in knowledge stock, `R` in arrival-rate units):

$$
\mathcal{J}^{\ell'}_g(R) = (1-\pi)\,R, \qquad \mathcal{J}^{\ell''}_g(R) = \pi\,R
$$

---

## 6. Preferences and Resource Constraint

Flow utility (log, damaged consumption):

$$
U(\tilde C) = \delta \log C - \delta \log N(Y)
$$

Market-clearing output constraint (consumption = output net of all three investments):

$$
C = A_d K^d - i^d K^d + A_g K^g - i^g K^g - i^r(K^g + K^d)
$$

In the `(logK, Z)` representation (and dividing through by `K`), the consumption argument inside `δ log(·)` is:

$$
C/K = (A_d - i^d)(1-Z) + (A_g - i^g)Z - i^r
$$

(`δ log K` appears as a separate additive term; the `−i^r` term is present only in regimes where R&D is still active, i.e. pre-breakthrough.)

`δ` = subjective discount rate.

---

## 7. Uncertainty Aversion (the source of nonlinearity)

Robust (multiplier) preferences à la Hansen–Sargent. A common penalty parameter `ξ` governs all channels. Smaller `ξ` ⇒ more aversion; `ξ = ∞` ⇒ uncertainty-neutral.

### 7.1 Brownian / diffusion misspecification (drift distortion `h`)

Adds to the HJB:

$$
\min_h\; V_x'\big[\mu + \sigma h\big] + \tfrac12\mathrm{tr}\!\big[\sigma' V_{xx}\sigma\big] + \tfrac{\xi}{2}h'h
\quad\Rightarrow\quad
h^* = -\tfrac{1}{\xi}\,\sigma' V_x
$$

With independent Brownians, `h` separates by channel:

$$
\begin{aligned}
h_d &= -\tfrac{1}{\xi}\{V_{\log K} - V_Z Z\}(1-Z)\,\sigma_d \\
h_g &= -\tfrac{1}{\xi}\{V_{\log K} + V_Z (1-Z)\}Z\,\sigma_g \\
h_r &= -\tfrac{1}{\xi}\,V_{\log R}\,\sigma_r \\
h_y &= -\tfrac{1}{\xi}\,V_y\,\mathcal{E}_t\,\varsigma
\end{aligned}
$$

After substitution the diffusion-uncertainty contribution becomes the quadratic penalty term `−(1/2ξ) V_x σ σ' V_x` (see augmented HJB below).

### 7.2 Jump misspecification (intensity distortion `g`)

For each jump channel with intensity `J^ℓ`, post-jump value `V^ℓ`:

$$
\min_{g^\ell \ge 0}\; \sum_\ell J^\ell\, g^\ell\,[V^\ell - V] + \xi\sum_\ell J^\ell\big[1 - g^\ell + g^\ell \log g^\ell\big]
\quad\Rightarrow\quad
g^{\ell*} = \exp\!\left(-\tfrac{1}{\xi}\big[V^\ell - V\big]\right)
$$

Substituting gives the jump contribution:

$$
\xi \sum_\ell J^\ell(x)\left[\,1 - \exp\!\left(-\tfrac{1}{\xi}\big(V^\ell - V\big)\right)\right]
$$

### 7.3 Augmented generic HJB (all channels)

$$
\begin{aligned}
0 = -\delta V(x) + \sup_{\alpha}\Big\{\, & U(x,\alpha) + V_x'\mu(x,\alpha) + \tfrac12\mathrm{tr}\!\big[\sigma' V_{xx}\sigma\big] \\
& - \tfrac{1}{2\xi}\,V_x'\,\sigma\sigma'\,V_x \Big\}
+ \xi \sum_{\ell} J^\ell(x)\Big[1 - \exp\!\big(-\tfrac{1}{\xi}(V^\ell - V)\big)\Big]
\end{aligned}
$$

`ξ → ∞` recovers the standard (uncertainty-neutral) HJB: the `−1/(2ξ)` term vanishes and the jump bracket → `∑ J^ℓ (V^ℓ − V)`.

---

## 8. Investment First-Order Conditions

Let `C/K` be the consumption argument from §6 and `MU = δ/(C/K)` (marginal utility of consumption). FOCs equate marginal benefit of capital to `MU`:

$$
\begin{aligned}
\frac{\delta}{C/K} &= \phi_d'(i^d)\,\big[V_{\log K} - Z\,V_Z\big] \\
\frac{\delta}{C/K} &= \phi_g'(i^g)\,\big[V_{\log K} + (1-Z)\,V_Z\big] \\
\frac{\delta}{C/K} &= \psi_r'(i^r)\,V_{\log R} \qquad\text{(only in R\&D-active regimes)}
\end{aligned}
$$

These are solved jointly with the consumption constraint at each grid/sample point (the policy-improvement step parameterizes `i^d, i^g, i^r` as networks; FOCs guide the control loss).

---

## 9. The Coupled HJB System (six regimes)

Solve **backward through the jump tree**: from the absorbing terminal regime to the initial pre-pre regime. Index regimes by (tech state, damage state).

| Regime | Value fn | States | R&D active? | Jump terms present |
|---|---|---|---|---|
| Post-damage, Post-tech (terminal) | `V^{(ℓ,ℓ'')}` | `logK, Z, Y` | no | none |
| Post-damage, Interm-tech | `V^{(ℓ,ℓ')}` | `logK, Z, Y, logR` | yes | tech jump `ℓ'→ℓ''` |
| Post-damage, Pre-tech | `V^{(ℓ)}` | `logK, Z, Y, logR` | yes | tech jumps `→ℓ', →ℓ''` |
| Pre-damage, Post-tech | `V^{(ℓ'')}` | `logK, Z, Y` | no | damage jump (sum over ℓ) |
| Pre-damage, Interm-tech | `V^{(ℓ')}` | `logK, Z, Y, logR` | yes | tech `→ℓ''` + damage jump |
| Pre-damage, Pre-tech (initial) | `V` | `logK, Z, Y, logR` | yes | tech `→ℓ',→ℓ''` + damage jump |

**Diffusion block** (common to all regimes; substitute the regime's value function):

$$
\begin{aligned}
& \Big[(1-Z)\phi_d(i^d) + Z\phi_g(i^g) - \tfrac{\sigma_d^2(1-Z)^2 + \sigma_g^2 Z^2}{2}\Big]V_{\log K}
+ \tfrac{\sigma_d^2(1-Z)^2 + \sigma_g^2 Z^2}{2}V_{\log K,\log K} \\
&+ \Big[\phi_g(i^g) - Z\sigma_g^2 - \phi_d(i^d) + (1-Z)\sigma_d^2\Big]Z(1-Z)\,V_Z
+ \tfrac12 Z^2(1-Z)^2(\sigma_g^2 + \sigma_d^2)\,V_{ZZ} \\
&+ \big[-Z(1-Z)^2\sigma_d^2 + Z^2(1-Z)\sigma_g^2\big]V_{\log K, Z}
+ V_y\,(\bar\theta + \varsigma\!\cdot\! h)\,\mathcal{E}_t + \tfrac{\varsigma^2 \mathcal{E}_t^2}{2}V_{yy} \\
&+ \Big(\psi_r(i^r) - \tfrac12\sigma_r^2 + \sigma_r\!\cdot\! h\Big)V_{\log R}
+ \tfrac{\sigma_r^2}{2}V_{\log R,\log R}
+ \xi\,\tfrac{|h|^2}{2}
\end{aligned}
$$

(`logR` terms and the `−i^r` consumption term drop in post-breakthrough regimes.)

**Full pre-damage, pre-tech HJB (the initial-state PDE, 4 states + pseudo-states):**

$$
\begin{aligned}
0 = \max_{i^d,i^g,i^r}\ \min_{g^\ell,g^{\ell'},g^{\ell''},h}\ &
\delta\log\!\big([A_d-i^d](1-Z) + [A_g-i^g]Z - i^r\big) + \delta\log K - \delta\log N(y) - \delta V \\
& + \text{[diffusion block above]} \\
& + \mathcal{J}^{\ell'}_g(R)\,g^{\ell'}\,(V^{(\ell')}-V) + \mathcal{J}^{\ell''}_g(R)\,g^{\ell''}\,(V^{(\ell'')}-V) + \sum_{\ell=1}^{L}\mathcal{J}^{\ell}_n(y)\,g^{\ell}\,(V^{(\ell)}-V) \\
& + \xi\Big[\mathcal{J}^{\ell'}_g(R)(1 - g^{\ell'} + g^{\ell'}\log g^{\ell'}) + \mathcal{J}^{\ell''}_g(R)(1 - g^{\ell''} + g^{\ell''}\log g^{\ell''}) \\
& \qquad\quad + \sum_{\ell=1}^{L}\mathcal{J}^{\ell}_n(y)(1 - g^{\ell} + g^{\ell}\log g^{\ell})\Big]
\end{aligned}
$$

with closed-form minimizers
`g^{ℓ} = exp(−(1/ξ)(V^{(ℓ)}−V))`, `g^{ℓ'} = exp(−(1/ξ)(V^{(ℓ')}−V))`, `g^{ℓ''} = exp(−(1/ξ)(V^{(ℓ'')}−V))`, and `h` as in §7.1.

The other five regime HJBs are obtained by dropping/keeping the relevant jump-expectation and `logR` terms per the table, replacing `A_g` by the realized productivity, and fixing `λ_3(ℓ)` in `N(y;ℓ)`.

---

## 10. Functional-Form Assumptions

- Independent Brownian shocks: all cross-products `σ_d'σ_g = σ_d'ς = σ_d'σ_r = σ_g'ς = σ_g'σ_r = ς'σ_r = 0`. ⇒ `h` separates cleanly (§7.1) and `σσ'` is diagonal.
- AK production in both sectors.
- Log utility, log adjustment costs.
- Damage and tech jumps modeled as Poisson-type arrivals with state-dependent intensities.

---

## 11. Parameter Values

### State-variable initial values and ranges

| State | Initial | Range |
|---|---|---|
| `K_0` | 880 | — |
| `Z_0` | 0.7 | `[0.01, 0.99]` |
| `Y_0` | 1.1 | `[0, 4]` |
| `R_0` | 11.2 | — |
| `logK` | — | `[4, 7]` |
| `logR` | — | `[1, 6]` |

### Economic parameters

| Parameter | Value |
|---|---|
| `δ` | 0.01 |
| `(α_d, Γ_d, θ_d, σ_d)` | `(−0.035, 0.060, 16.7, 0.01)` |
| `(α_g, Γ_g, θ_g, σ_g)` | `(−0.035, 0.060, 16.7, 0.01)` |
| `A_d` | 0.1303 |
| `{A_g, A_g(ℓ'), A_g(ℓ'')}` | `{0.1085, 0.1303, 0.1567}` |
| `(λ, n)` (Acemoglu step size, gap) | `(1.063, 3)` |
| `(ζ, ψ_0, ψ_1, σ_r)` | `(0, 0.10583, 0.5, 0.0078)` |
| `ϱ` (R&D scaling) | 746.67 |
| `π` (breakthrough prob \| tech jump) | 0.04 |

Calibration identities used to pin `A_d, A_g`:
`A_d = λ^n · A_g`,  `ᾱ = Z_0·A_g + (1−Z_0)·A_d`, with target average productivity `ᾱ = 0.115`.
Catch-up: `A_g(ℓ') = A_d`. Breakthrough: `A_g(ℓ'') = λ^n · A_d`.

### Climate / damage parameters

| Parameter | Value |
|---|---|
| `\bar θ` | `1.86 / 1000` |
| `η` | 0.291 |
| `ς` | `1.2 × 1.86 / 1000` |
| `(λ_1, λ_2)` | `(0.00017675, 0.0044)` |
| `λ_3(ℓ)` | `(1/3)·(ℓ−1)/(L−1)`, `ℓ = 1,…,L` |
| `(r_1, r_2)` | `(1.5, 0.36)` |
| `(y̲, ȳ)` | `(1.5, 2.5)` |

### Uncertainty / solution settings (results)

- `ξ ∈ {0.05, 0.1, ∞}` (more averse, less averse, neutral).
- Damage realizations: `L = 5`, equi-spaced `λ_3(ℓ) ∈ {0, 1/12, 1/6, 1/4, 1/3}`.
- Tech: intermediate `A_g(ℓ') = 0.1303`, breakthrough `A_g(ℓ'') = 0.1567`.

---

## 12. Numerical Algorithm (DGM-PIA + pseudo-states)

**Method:** Deep Galerkin Method – Policy Improvement Algorithm. Parameterize value function `V^θ` and controls `α^φ` as separate networks; alternate two gradient steps per epoch.

Generic HJB target:
$$
-\delta V(x) + \sup_{\alpha\in\mathcal{A}}\{\mathcal{L}^{\alpha}V(x) + f(x,\alpha)\} = 0
$$

- **Step 1 (value):** minimize PDE residual for fixed policy
  `L_V(θ) = (1/M) Σ_m [ −δV(x_m;θ) + 𝓛^{α(x_m;φ)}V(x_m;θ) + f(x_m, α(x_m;φ)) ]²`
- **Step 2 (policy):** maximize the Hamiltonian for fixed value
  `α^{φ_{n+1}} ∈ argmax_α { 𝓛^α V^{θ_n}(x) + f(x,α) }` via gradient on the control loss.

**Pseudo-states (the key extension):** include `ξ` and the damage realization index `ℓ` as *network inputs*; sample them alongside the true states in the SGD mini-batch. This lets a single network learn the solution across the parameter family, exploiting implicit regularization / smoothness and (empirically) preserving monotonicity of `V` in `ξ` and `ℓ`. Adding `p` pseudo-states adds only `p·m` weights to the first layer (width `m`) — negligible cost.

**Network / training hyperparameters:**
- Feedforward, 4 hidden layers, width 32.
- Hidden activations: `tanh` and `softplus` for controls; `swish` for the value function.
- Output layer: customized `tanh` + `softplus`.
- Epochs: 2,000,000; batch size 128.
- Learning rates `lr_V = lr_α = 1e-5`; optimizer ADAM.
- ~2.3 h on a single Colab GPU.

**Solve order:** terminal regime first (3 states, no jump terms), then each intermediate regime using already-solved post-jump value functions as inputs to the jump-expectation terms, ending with the pre-damage/pre-tech regime.

---

## 13. Validation (sanity checks to reproduce)

- 3-state post-jump regimes: compare NN solution to a finite-difference (false-transient, conjugate-gradient) solution — should match across the state space.
- Training loss (`√` mean-squared PDE residual) target: order `1e-3` per regime.
- Check residuals over economically relevant regions: 5th/95th percentiles of states at `t = 20, 40, 60` from 10,000 simulated paths; residuals should stay ~`1e-3` and not concentrate in visited regions.

---

## 14. Implementation Checklist

1. Encode states `(logK, Z, Y, logR)` + pseudo-states `(ξ, ℓ)`; box-bound per §11.
2. Implement `φ_d, φ_g, φ_d', φ_g', ψ_r, ψ_r'`, `𝓔 = η A_d (1−Z) K`, and `N(y;ℓ)`.
3. Implement closed-form distortions `h_{d,g,r,y}` and `g^{ℓ,ℓ',ℓ''}` (§7).
4. Build the diffusion block + regime-specific jump terms (§9 table).
5. Investment FOCs (§8) for the policy loss / control network.
6. Solve regimes backward; feed post-jump `V` into pre-jump jump-expectation terms.
7. Sweep `ξ ∈ {∞, 0.1, 0.05}` and `ℓ = 1..5` via pseudo-state sampling.
8. Validate against finite differences (3-state regimes) and residual diagnostics.