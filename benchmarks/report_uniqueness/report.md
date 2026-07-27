---
title: "Does HJB/FOC residual minimization determine a unique numerical solution? A uniqueness analysis"
author: "TwoCapital team"
date: 2026-07-09
---

## The problem

**Headline. Our current solver — least-squares minimization of the HJB residual plus the control (FOC)
residuals — does not determine a unique, stable numerical solution.** Independently trained networks that all
reach the same paper-grade equation error encode a materially different marginal value of capital and, therefore,
different decisions and densities.

*One sentence:* the residual-minimization objective is missing the conditions that make the HJB solution unique,
so its minimizers form a family, and different trainings land on different members.

**Both trainings converge.** Two networks were trained with identical calibration but a different sampled range of
the model-uncertainty aversion parameter $\xi$ (the economically relevant axis; one training sees deeper $\xi$,
i.e. stronger worst-case scenarios, than the other). In every regime the HJB residual settles onto the same
$\sim 10^{-3}$ floor — by the standard diagnostic, both are converged:

:::{figure} figures/L_loss_history.png
:width: 100%
:name: fig-loss
HJB residual (loss$_v$), last $50{,}000$ training steps, four regimes, two trainings. Both sit at the same
$\sim 10^{-3}$ floor everywhere.
:::

**Yet the solutions differ.** Simulated from the same initial state, the two converged solutions disagree on
every reported quantity — the decisions *and* the jump densities — even at $\xi=\infty$, where there is no
robustness distortion at all. Each panel below is one quantity; red is training 1, blue training 2.

:::{figure} figures/S1_green_investment.png
:width: 85%
:name: fig-ig
Green investment $I_g$.
:::

:::{figure} figures/S2_dirty_investment.png
:width: 85%
:name: fig-id
Dirty investment $I_d$.
:::

:::{figure} figures/S3_rd.png
:width: 85%
:name: fig-rd
R\&D-to-output ratio.
:::

:::{figure} figures/S4_emissions.png
:width: 85%
:name: fig-e
Emissions.
:::

:::{figure} figures/S5_consumption.png
:width: 85%
:name: fig-c
Consumption-to-output ratio.
:::

:::{figure} figures/S6_damage_density.png
:width: 85%
:name: fig-dmg
Damage first-jump density.
:::

:::{figure} figures/S7_tech_density.png
:width: 85%
:name: fig-tech
Technology first-jump density.
:::

Same calibration, same diagnostics, different results. This note asks the underlying question: is that
non-uniqueness a property of the *PDE*, or of the *objective we minimize*?

---

## The model and the solver

### The HJB and the loss

**States and controls.** Endogenous states $x=(\log K, Z, Y, \log R)$: log total capital, green capital share
$Z=K^g/K$, temperature anomaly $Y$, log knowledge stock. Controls: dirty/green investment rates $i^d,i^g$ and
R\&D rate $i^r$. Pseudo-states: robustness parameter $\xi$ and damage index $\ell$. Per-sector capital drift
$\phi_j(i)=\alpha_j+\Gamma_j\log(1+\theta_j i)$, $\ \phi_j'(i)=\Gamma_j\theta_j/(1+\theta_j i)$.

**The augmented robust HJB** (one per regime; worst-case distortions already substituted in closed form):

$$
0 = -\delta V + \sup_{i^d,i^g,i^r}\Big\{\, \delta\big[\log(C/K)+\log K-\log N(Y;\ell)\big]
+ \mathcal{L}V - \tfrac{1}{2\xi}\,\nabla V^\top \sigma\sigma^\top \nabla V \,\Big\}
+ \xi\sum_{\ell}\mathcal{J}^{\ell}\Big[1-e^{-(V^{\ell}-V)/\xi}\Big],
$$

with $C/K=(A_d-i^d)(1-Z)+(A_g-i^g)Z-i^r$ and the diffusion block

$$
\begin{aligned}
\mathcal{L}V =\;& \Big[(1-Z)\phi_d+Z\phi_g-\tfrac{\sigma_d^2(1-Z)^2+\sigma_g^2Z^2}{2}\Big]V_{\log K}
+\tfrac{\sigma_d^2(1-Z)^2+\sigma_g^2Z^2}{2}V_{\log K\log K}\\
&+\big[\phi_g-Z\sigma_g^2-\phi_d+(1-Z)\sigma_d^2\big]Z(1-Z)\,V_Z
+\tfrac{1}{2}Z^2(1-Z)^2(\sigma_g^2+\sigma_d^2)\,V_{ZZ}\\
&+\big[-Z(1-Z)^2\sigma_d^2+Z^2(1-Z)\sigma_g^2\big]V_{\log K,Z}
+ V_Y(\bar\theta+\varsigma h_y)\mathcal{E}+\tfrac{\varsigma^2\mathcal{E}^2}{2}V_{YY}\\
&+\big(\psi_r(i^r)-\tfrac12\sigma_r^2\big)V_{\log R}+\tfrac{\sigma_r^2}{2}V_{\log R,\log R},
\qquad \mathcal{E}=\eta A_d(1-Z)K .
\end{aligned}
$$

**The FOCs.** With marginal utility $\delta/(C/K)$,

$$
\frac{\delta}{C/K}=\phi_d'(i^d)\big[V_{\log K}-Z\,V_Z\big],\qquad
\frac{\delta}{C/K}=\phi_g'(i^g)\big[V_{\log K}+(1-Z)\,V_Z\big],\qquad
\frac{\delta}{C/K}=\psi_r'(i^r)\,V_{\log R}.
$$

**The loss (transcribed from the solver).** The state ranges $\log K\in[4,7],\,Z\in[0.01,0.99],\,Y\in[0,4],\,
\log R\in[1,6]$ are sampled stratified-uniform each step (mesh-free). The value network minimizes an
equal-weight sum of root-mean-square terms — exactly as coded, with no tunable coefficients:

$$
\begin{aligned}
\mathcal{L}(\theta)=\;&\underbrace{\sqrt{\tfrac1M\textstyle\sum_m\big(\mathrm{rhs}_m-\delta V_m\big)^2}}_{\text{HJB residual}}
+\sqrt{\tfrac1M\textstyle\sum_m(\mathrm{FOC}^d_m)^2}+\sqrt{\tfrac1M\textstyle\sum_m(\mathrm{FOC}^g_m)^2}
+\sqrt{\tfrac1M\textstyle\sum_m(\mathrm{FOC}^r_m)^2}\\[2pt]
&+\underbrace{\sqrt{\tfrac1M\textstyle\sum_m\big[(V_Y)_+\,\mathbf{1}_{\{Y>\bar y\}}\big]^2}
+\sqrt{\tfrac1M\textstyle\sum_m\big[(V_{\log R})_-\big]^2}}_{\text{monotonicity penalties}} .
\end{aligned}
$$

Here the pointwise residual is $\mathrm{rhs}-\delta V$, where $\mathrm{rhs}$ is the full right-hand side of the
HJB other than the $-\delta V$ term,

$$
\mathrm{rhs}=\delta\big[\log(C/K)+\log K-\log N\big]+\mathcal{L}V-\frac{1}{2\xi}\,\big|\sigma^\top\nabla V\big|^2
+\xi\sum_{\ell}\mathcal{J}^{\ell}\big[1-e^{-(V^{\ell}-V)/\xi}\big],
$$

evaluated at the current controls (at the optimum it realizes the $\sup$ in (1)). The three control residuals,
one per investment margin, are

$\mathrm{FOC}^d=-\dfrac{\delta}{C/K}+\dfrac{\Gamma_d\theta_d}{1+\theta_d\,i^d}\big(V_{\log K}-Z\,V_Z\big)$,

$\mathrm{FOC}^g=-\dfrac{\delta}{C/K}+\dfrac{\Gamma_g\theta_g}{1+\theta_g\,i^g}\big(V_{\log K}+(1-Z)\,V_Z\big)$,

$\mathrm{FOC}^r=-\dfrac{\delta}{C/K}+\psi_0\,\psi_1\,e^{\psi_1(\log i^r+\log K-\log R)}\,\dfrac{V_{\log R}}{i^r}$,

each of which vanishes at the optimal control. The two monotonicity penalties push welfare *down* in temperature
above $\bar y$ (the positive part $(V_Y)_+$) and *up* in knowledge (the negative part $(V_{\log R})_-$). The policy
step instead maximizes the Hamiltonian $\tfrac1M\sum_m(\mathrm{rhs}_m-\delta V_m)$ under the same FOC terms.

Two facts used below: the *overall level of welfare* enters only through $-\delta V$; and the green/dirty split is
set entirely by $V_Z$, the marginal value of the capital mix (opposite signs in the $i^d$ and $i^g$ FOCs). No
anchor or normalization is imposed.

### The training procedure

Deep-Galerkin policy improvement: alternate a value step (minimize the residual, controls fixed) and a policy
step (satisfy the FOCs, value fixed); regimes are solved backward through the jump tree with downstream value
networks frozen inside the jump term.

```{raw} latex
\begin{center}
\begin{tikzpicture}[font=\small, node distance=1.1cm]
\node[draw, rounded corners, fill=gray!10, minimum width=3.5cm, minimum height=0.9cm] (box) {sample $x$ from the state ranges (mesh-free)};
\node[draw, rounded corners, fill=blue!8, minimum height=1cm, minimum width=2.6cm, below left=1.2cm and 0.2cm of box] (v) {value net $V_\theta$};
\node[draw, rounded corners, fill=green!8, minimum height=1cm, minimum width=2.6cm, below right=1.2cm and 0.2cm of box] (a) {control nets $\alpha_\varphi$};
\draw[-{Stealth}, thick] (box) -- (v); \draw[-{Stealth}, thick] (box) -- (a);
\draw[-{Stealth}, thick] (v.north east) to[bend left=18] node[above, align=center, font=\footnotesize]{Step 2: fix $V$, hit the FOCs} (a.north west);
\draw[-{Stealth}, thick] (a.south west) to[bend left=18] node[below, align=center, font=\footnotesize]{Step 1: fix $\alpha$, minimize HJB residual} (v.south east);
\end{tikzpicture}
\end{center}
```

---

## What makes the solution unique

```{raw} latex
\medskip\noindent\textit{Setup.} Fix the pseudo-states $(\xi,\ell)$. On the open domain
$\Omega=(4,7)\times(0.01,0.99)\times(0,4)\times(1,6)\subset\mathbb{R}^4$ with $x=(\log K,Z,Y,\log R)$, the regime's
value function solves $F(x,V,DV,D^2V)=0$ in the viscosity sense, where, after the worst-case distortions are
substituted in closed form,
\[
F(x,r,p,M)=\delta r-\tfrac12\operatorname{tr}\!\big(A(x)\,M\big)-H(x,p)-\mathcal{J}[x,r],
\]
with diffusion matrix $A=\sigma\sigma^\top\succeq0$; first-order Hamiltonian
$H(x,p)=\sup_{(i^d,i^g,i^r)\in\mathcal{A}}\{\,b(x,i)\cdot p+\delta\log(C/K)\,\}-\tfrac{1}{2\xi}\,|\sigma^\top p|^2$;
and jump term $\mathcal{J}[x,r]=\xi\sum_{\ell}\mathcal{J}^\ell(x)\big(1-e^{-(V^\ell(x)-r)/\xi}\big)$, the downstream
values $V^\ell$ entering as frozen data.

\begin{defn}[Proper operator]
$F$ is \emph{degenerate elliptic} if $M\preceq M'\Rightarrow F(x,r,p,M)\ge F(x,r,p,M')$, and \emph{monotone} if
$r\le s\Rightarrow F(x,r,p,M)\le F(x,s,p,M)$; it is \emph{proper} if both hold.
\end{defn}

\begin{assu}\label{a:main}
\emph{(i)}~$\delta=0.01>0$;\quad \emph{(ii)}~$A=\sigma\sigma^\top\succeq0$ is Lipschitz on $\overline\Omega$;\quad
\emph{(iii)}~the downstream data $\{V^\ell\}$ are fixed and Lipschitz (acyclic backward coupling), so
$\mathcal{J}[x,\cdot]$ depends on the unknown only through the local value $r$;\quad \emph{(iv)}~the controlled
state is confined to $\overline\Omega$ over the $\delta^{-1}\approx100$-year horizon, so a state-constraint
boundary condition applies. All four hold; (iv) is confirmed by the simulated paths.
\end{assu}

\begin{lem}[$F$ is proper, with modulus $\delta$]\label{l:proper}
For every $(x,r,p)$ and $A(x)\succeq0$,
\[
\frac{\partial F}{\partial M}=-\tfrac12A(x)\preceq0,
\qquad
\frac{\partial F}{\partial r}=\delta+\sum_{\ell}\mathcal{J}^\ell(x)\,e^{-(V^\ell(x)-r)/\xi}
=\delta+\sum_{\ell}\mathcal{J}^\ell(x)\,g^\ell\ \ge\ \delta>0 .
\]
Hence $F$ is degenerate elliptic and strictly monotone with modulus $\delta$.
\end{lem}
\begin{proof}
Differentiate term by term; the worst-case belief $g^\ell:=e^{-(V^\ell-r)/\xi}\ge0$ and the intensities
$\mathcal{J}^\ell\ge0$.
\end{proof}

\begin{thm}[Comparison principle; Crandall--Ishii--Lions 1992]\label{t:cil}
On a bounded domain a continuous, proper operator obeys a comparison principle: a bounded subsolution lies below a
bounded supersolution (given the boundary ordering, or Soner's state-constraint condition on the
absorbing/characteristic faces). In particular the solution is unique. See CIL 1992 for the statement and its
doubling-of-variables proof.
\end{thm}

\begin{prop}[The HJB is uniquely solvable]\label{p:uniq}
Under Assumption~\ref{a:main} the equation $F=0$ has at most one bounded (constrained) viscosity solution.
\end{prop}
\begin{proof}[Proof sketch.]
\emph{The discount is the comparison constant.} At a maximum of $u-v$, the sub- and supersolution inequalities
together with Lemma~\ref{l:proper} give $\delta\,\sup_{\overline\Omega}(u-v)\le0$; since $\delta>0$ this forces
$u\le v$, and exchanging $u$ and $v$ yields uniqueness. (The maximum is made rigorous by the standard
doubling-of-variables argument, CIL 1992.)
\end{proof}
\noindent\textit{Economic content.} Discounting makes the value a strict contraction of the future; the proof
collapses to $\delta M\le0$, so $\delta$ is exactly the constant multiplying the gap between two candidate
solutions. At $\delta=0$ the problem is ergodic and the welfare level floats freely.

\begin{prop}[The overall level of welfare is pinned only weakly, at rate $\delta$]\label{p:level}
\emph{(i)}~The uniform shift $V\mapsto V+c$ changes the residual by exactly $-\delta c$
$\big(\partial F[V+c]/\partial c=-\delta$; all other terms invariant$\big)$.
\emph{(ii)}~If a network $V^\theta$ satisfies $F[V^\theta]=R$ pointwise with $\|R\|_\infty\le\varepsilon$, then
$\|V^\theta-V^\star\|_\infty\le\varepsilon/\delta$ for the exact solution $V^\star$. The welfare level is thus
pinned $\delta^{-1}=100$ times more weakly than the shape of the value function.
\end{prop}
\begin{proof}
\emph{(i)}~$-\delta(V+c)=-\delta V-\delta c$, while $H$, $\mathcal{J}$ and $\operatorname{tr}(AD^2V)$ are
shift-invariant. \emph{(ii)}~$V^\theta$ is a classical sub- and supersolution of $F=\pm\varepsilon$; the estimate
of Proposition~\ref{p:uniq} with inhomogeneity $R$ gives $\delta\,\sup|V^\theta-V^\star|\le\|R\|_\infty$.
\end{proof}
\noindent\textit{Economic content.} Welfare has an absolute zero, pinned by $\delta$; but the solver holds it with
a spring $100\times$ weaker than it holds the shape of $V$, so independent trainings agree on the welfare level
only to $\sim1\%$.

\begin{prop}[The marginal value of the green/dirty capital mix ($V_Z$) is not pinned]\label{p:vz}
$V_Z$ enters $F$ only through the first-order coefficient
$\mu_Z(x)=Z(1-Z)\big[\phi_g-\phi_d+(1-Z)\sigma_d^2-Z\sigma_g^2\big]=O(10^{-3})$ and the second-order coefficient
$A_{ZZ}=Z^2(1-Z)^2(\sigma_g^2+\sigma_d^2)=O(\sigma^2)=O(10^{-4})$, so
$\partial F/\partial V_Z=-\mu_Z+O(\sigma^2)=O(10^{-3})$. At residual floor $\varepsilon$ the admissible spread is
$\|V_Z^\theta-V_Z^\star\|=O(\varepsilon/10^{-3})=O(1)$; thus $V_Z$ is undetermined and may differ in sign across
trainings. Moreover $Z\in\{0,1\}$ are characteristic $(A_{ZZ}\to0)$ and absorbing, so no boundary value for $V_Z$
can be posted there.
\end{prop}
\noindent\textit{Economic content.} The capital mix carries almost no intrinsic risk, so nothing in the dynamics
pins how much welfare responds to $Z$; and as absorbing corners $Z=0,1$ admit no boundary datum --- unlike an
ordinary boundary. Yet through the FOCs $V_Z$ fixes the entire green/dirty split.

\begin{coro}\label{c:main}
The non-uniqueness seen across trainings is numerical under-identification of the unique solution of
Proposition~\ref{p:uniq}, concentrated exactly where Propositions~\ref{p:level}--\ref{p:vz} place it: the overall
welfare level (rate $\delta$) and the marginal value of the capital mix $V_Z$ (rate $O(\sigma^2)$). The training
objective supplies the hypotheses of Proposition~\ref{p:uniq} and the confinement of Assumption~\ref{a:main}; it
omits an anchor for the welfare level and a pinning of the capital-mix value $V_Z$ --- the two rows in bold below.
\end{coro}
```

```{raw} latex
\begin{center}
\makebox[0pt]{%
\renewcommand{\arraystretch}{1.45}
\begin{tabular}{p{4.7cm} p{8.6cm} p{5.4cm}}
\hline
\textbf{Condition} & \textbf{What it fixes} & \textbf{In our objective?} \\
\hline
$\delta>0$ (Prop.\,1) & discounting makes the value a contraction & \textbf{Yes} --- $\delta=0.01$ \\
$\sigma\sigma^\top$ smooth (Prop.\,1) & a well-behaved diffusion & \textbf{Yes} \\
acyclic backward coupling & uniqueness propagates down the jump tree & \textbf{Yes} \\
confinement to the domain (Assum.\ iv) & the economy stays inside over the horizon & \textbf{Yes} --- validated \\
\textbf{anchor for the welfare level (Prop.\,2)} & pins welfare's absolute zero; removes the $100\times$ weakness & \textbf{No} --- only the weak $\delta V$ term \\
\textbf{pinning of the capital-mix value $V_Z$ (Prop.\,3)} & pins the green/dirty split, hence the policies & \textbf{No} --- $\sigma^2=10^{-4}$; no supervision \\
\hline
\end{tabular}}
\end{center}
```

The route to a unique, stable numerical solution is to add the two missing rows directly — an explicit anchor for
the welfare level, and direct supervision of the capital-mix value $V_Z$ — not more residual minimization, which
is stationary in precisely these directions.
