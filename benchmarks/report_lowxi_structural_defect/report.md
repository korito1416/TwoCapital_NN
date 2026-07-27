---
title: "Same calibration, divergent trainings: which loss term carries the difference"
author: "TwoCapital team"
date: 2026-07-11
---

## How the solutions are trained

We solve the one-technology-jump climate model ($\pi=1$: four jump states, pre-post damage $\times$ pre-post
technology) with the paper's DGM-PIA algorithm. Each jump state has its own networks: the value function and
each investment rule ($i_g$, $i_d$, $i_r$) are separate feedforward networks, four hidden
layers of width 32. Each of the 1 million training iterations:

- **draws a fresh batch of 128 states by Latin-hypercube sampling** over the training ranges of the state
  variables ($\log K$, $Z$, $Y$, $\log R$) and the pseudo-states (damage curvature $\lambda_3$,
  uncertainty-aversion $\log\xi$);
- evaluates the objective on that batch: six equally weighted root-mean-square terms — the error in the HJB
  equation, the FOC errors, and two monotonicity penalties;
- takes alternating gradient steps on the value and the investment networks.

The four jump states are trained in sequence (post-jump states first), the trained value functions
feeding the next jump state's jump expectations.

```{raw} latex
\begin{center}
\begin{tikzpicture}[font=\small,
  box/.style={draw, rounded corners, align=center, minimum height=0.9cm, inner sep=5pt}]
\node[box, fill=gray!10, minimum width=9.2cm] (s)
  {Latin-hypercube draw (measure $\nu$), batch of 128:\\
   states $(\log K,\, Z,\, Y,\, \log R)$ \ $+$ \ pseudo-states $(\lambda_3,\, \log\xi)$};
\node[box, fill=blue!8, below left=1.0cm and -2.6cm of s] (v) {value network $V$\\ (4 hidden layers $\times$ 32)};
\node[box, fill=green!8, below right=1.0cm and -2.6cm of s] (a) {investment networks\\ $i_g,\ i_d,\ i_r$};
\node[box, fill=red!8, below=3.3cm of s, minimum width=9.2cm] (L)
  {objective: error in the HJB equation $+$ FOC errors $+$ monotonicity penalties};
\draw[-{Stealth}, thick] (s) -- (v); \draw[-{Stealth}, thick] (s) -- (a);
\draw[-{Stealth}, thick] (v) -- (L); \draw[-{Stealth}, thick] (a) -- (L);
\draw[-{Stealth}, thick] (L.west) to[bend left=30] node[left, font=\footnotesize, align=center]{gradient step\\(value)} (v.west);
\draw[-{Stealth}, thick] (L.east) to[bend right=30] node[right, font=\footnotesize, align=center]{gradient step\\(investments)} (a.east);
\end{tikzpicture}

\vspace{0.5cm}

\begin{tikzpicture}[font=\small,
  st/.style={draw, rounded corners, align=center, inner sep=5pt, minimum height=0.9cm, fill=gray!8}]
\node[st] (pp) {post-dmg\\ post-tech};
\node[st, above right=0.25cm and 1.6cm of pp] (prepost) {pre-dmg\\ post-tech};
\node[st, below right=0.25cm and 1.6cm of pp] (postpre) {post-dmg\\ pre-tech};
\node[st, right=5.6cm of pp] (prepre) {pre-dmg\\ pre-tech};
\draw[-{Stealth}, thick] (pp) -- (prepost); \draw[-{Stealth}, thick] (pp) -- (postpre);
\draw[-{Stealth}, thick] (prepost) -- (prepre); \draw[-{Stealth}, thick] (postpre) -- (prepre);
\node[below=0.35cm of pp, font=\footnotesize, align=center]{trained first};
\node[below=0.35cm of prepre, font=\footnotesize, align=center]{trained last};
\node[below=0.55cm of postpre, font=\footnotesize, align=center]
  {trained value functions feed the next jump state's jump expectations};
\end{tikzpicture}
\end{center}
```

We train twice from the same calibration and initial state; the two trainings share all of the above.

## Both solutions pass the standard diagnostics

Both trainings drive the error in the HJB equation to the same paper-grade floor — order $10^{-3}$ in every
jump state ({numref}`fig-loss`) — with investment optimality errors of order $10^{-4}$.

:::{figure} figures/L_loss_history.png
:width: 100%
:name: fig-loss
HJB-equation error (loss$_v$), last 50,000 training steps, four jump states, two trainings. Both sit at the
same $\sim10^{-3}$ floor in every jump state.
:::

Calibration (common): $\delta=0.01$; $(\alpha,\Gamma,\theta,\sigma)_d=(\alpha,\Gamma,\theta,\sigma)_g=
(-0.035,0.06,16.7,0.01)$; $A_d=0.1303$, $A_g=0.1085$, $A_g''=0.1567$; initial state $Y_0=1.2$, $K_0=880$,
$Z_0=0.7$, $R_0=11.2$.

## The training losses are not comparable across the two trainings

Over the final 100,000 iterations every jump state's objective is dominated by the HJB-equation error
({numref}`fig-K5`): 78–98% of the total, highest in the post-technology jump states. In levels
({numref}`fig-K5b`), the two trainings are close in the post-technology jump states, while training 2's total
is higher in the two R&D-active ones ($3.2$ vs $2.5$ and $3.3$ vs $2.4\times10^{-3}$), where its R&D term
bursts episodically. 

**Neither fact ranks the solutions**: each objective averages over its own sampled ranges of the states and
pseudo-states. The training loss measures each run's own convergence, not a comparison
between the runs.

:::{figure} figures/K5_loss_composition.png
:width: 100%
:name: fig-K5
Loss composition, last 100,000 training steps, all four jump states (100% stacked; top row training 1, bottom
training 2). HJB-equation error (blue) 78–98% of each objective; optimality terms (orange/green/red) largest
in the R&D-active jump states; monotonicity (gray) negligible.
:::

:::{figure} figures/K5b_loss_magnitude.png
:width: 100%
:name: fig-K5b
Same as {numref}`fig-K5` in levels, common scale across all panels. Each objective is evaluated on its own
Latin-hypercube draw over its own sampled ranges, so levels are not comparable across the trainings; training
2's excess concentrates in the R&D-active jump states, where its R&D term bursts (red) on batches landing
where that term is largest.
:::

## The delivered solutions can be compared, jump state by jump state

Each training ultimately delivers, per jump state, a value function and the investment rules (the functions as
of its last iteration). The comparison logic: fix $\xi$, take the decision rules and value function each
training generates, and measure how well they solve the model along the economy's own 60-year trajectory —
every term of the objective re-evaluated at the states visited along the simulated path and reported as the
**average over the 60 years** (root mean square across years). 

Each solution carries its own downstream value
functions in the jump expectations; the post-damage jump states, entered at the damage threshold, are
evaluated at $Y=\hat y=2.5$ with the other state variables following the path.

**The HJB-equation error does not order the two solutions in any jump state** ({numref}`fig-CS1`): the two
stay within $2.4\times$ of each other, and which is lower flips with $\xi$ and with the jump state — at
$\xi=148.6$ training 2 is lower in all four jump states, at $\xi=0.05$ training 1 is lower in the two
R&D-active ones.

:::{figure} figures/CS1_hjb_by_state.png
:width: 100%
:name: fig-CS1
Error in the HJB equation by jump state: for each $\xi$, both solutions' decision rules and value functions
are re-evaluated along the simulated 60-year path and the error is averaged (root mean square) across the
years. Trajectories are simulated at $\xi=0.05$ and $\xi=148.6$; each grid $\xi$ uses the nearer trajectory's
states. Post-damage jump states at entry temperature $Y=2.5$.
:::

**The optimality errors are smaller for training 2 nearly everywhere at the robust end and mixed at the
neutral end** ({numref}`fig-CS2`): at $\xi=0.05$ training 2 is lower on every term in every jump state (up to
$17\times$ for green investment in pre-damage/post-technology); at $\xi=148.6$ the orderings are mixed across
jump states and terms.

:::{figure} figures/CS2_foc_by_state.png
:width: 100%
:name: fig-CS2
Investment optimality (FOC) errors by jump state, averaged (root mean square) across the 60 years of the
simulated path at each $\xi$; log scales, rows share a scale. Training 2 lower on every term in every jump
state at $\xi=0.05$; mixed orderings at $\xi=148.6$. R&D row: R&D-active jump states only.
:::

The difference between the solutions lives in the value function's **marginal values**, not its level, and the
pattern repeats across the jump states ({numref}`fig-K6a`, $\xi=148.6$; {numref}`fig-K6b`, $\xi=0.05$): the
welfare level $V$ nearly coincides in every jump state ($\le0.13\%$ at the initial state at $\xi=148.6$; a
nearly parallel offset of $1.2$–$1.8\%$ at $\xi=0.05$), while the marginal value of capital ($V_{\log K}$) is
higher in training 1 in **every** jump state (5–15%) and the marginal value of knowledge ($V_{\log R}$)
carries gaps of 6–20% in the R&D-active ones. A systematic, same-signed difference across all four jump states
is a coherent alternative solution, not training noise.

:::{figure} figures/K6a_four_states_xi148.png
:width: 100%
:name: fig-K6a
Welfare $V$ and its marginal values by jump state (columns) along the simulated path, $\xi=148.6$; each row
shares a scale. Post-damage jump states are evaluated at their entry temperature $Y=\hat y=2.5$; the state at
each year is otherwise the one visited at that year, no averaging. Levels nearly coincide; the marginal value
of capital is higher in training 1 in every jump state.
:::

:::{figure} figures/K6b_four_states_xi0p05.png
:width: 100%
:name: fig-K6b
Same as {numref}`fig-K6a` at $\xi=0.05$. The welfare level shows a nearly parallel offset ($1.2$–$1.8\%$) in
every jump state — the additive constant the objective pins only through the discount rate — while the
marginal-value gaps widen (knowledge: 18–20% in the R&D-active jump states).
:::

Because the value function depends on the state variables, we also compare it one state variable at a time,
holding the others at their initial values ({numref}`fig-K7`; pre-damage/pre-technology jump state). At
$\xi=0.05$ the two solutions differ by a nearly parallel vertical offset ($0.07$, about $1.8\%$); at
$\xi=148.6$ the levels nearly coincide (offset $2\times10^{-4}$) while slope differences remain (against
$\log R$, 19% and 8%).

:::{figure} figures/K7_value_slices.png
:width: 100%
:name: fig-K7
Welfare $V$, one state variable at a time, others at their initial values ($K=880$, $Z=0.7$, $Y=1.2$,
$R=11.2$); pre-damage/pre-technology jump state; top row $\xi=0.05$, bottom $\xi=148.6$.
:::

The weighting across states matters for these orderings: re-evaluating the same terms under a uniform
Latin-hypercube weighting over the whole training ranges instead of the path, the $\xi=148.6$ ordering in the
initial jump state reverses — training 1 lower ($2.1$ vs $2.5\times10^{-3}$), against training 2 lower along
the path ($2.3$ vs $1.9\times10^{-3}$).

**One direction for pinning down the value function within the same algorithm: anchor the training to the
terminal jump state**. The post-damage/post-technology value function is the best-determined object in the
system — three state variables, no jump expectations, and verifiable against finite differences — and it
already enters every other jump state's objective through the jump expectations, a channel too weak to tie the
welfare levels. Adding to each subsequent jump state's objective a term that penalizes deviation of its value
function from the one the terminal jump state implies would supply the anchor that the HJB-equation error
alone leaves weakly determined, with no change to the rest of the algorithm.
