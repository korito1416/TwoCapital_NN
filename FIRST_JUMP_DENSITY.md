# Grouped First-Jump Density

Along a state-variable history \(X_{[0,t]}\), define each distorted jump
intensity as

\[
\widetilde{\lambda}^{\ell}_t
= G_t^{*\ell}\mathcal{J}^{\ell}(X_t).
\]

The grouped damage intensity sums over all damage realizations:

\[
\widetilde{\lambda}_{d,t}
= \sum_{\ell=1}^{L}G_t^{*\ell}\mathcal{J}^{\ell}_d(X_t)
= \frac{\mathcal{J}_d(Y_t)}{L}\sum_{\ell=1}^{L}G_t^{*\ell}.
\]

For the pre-technology regime, the grouped technology intensity sums over
the intermediate and post-technology destinations:

\[
\widetilde{\lambda}_{g,t}
= G_t^{*\prime}(1-\pi)s_g\frac{R_t}{\varrho}
+ G_t^{*\prime\prime}\pi s_g\frac{R_t}{\varrho},
\]

where \(s_g\) is `tech_jump_intensity_scale`.

The survival probability for no damage or technology jump before \(t\) is

\[
S_t
= \exp\left[-\int_0^t
\left(\widetilde{\lambda}_{d,\tau}
+\widetilde{\lambda}_{g,\tau}\right)d\tau\right].
\]

Therefore, the grouped path-conditional first-jump subdensities are

\[
q_{d,t}=\widetilde{\lambda}_{d,t}S_t,
\qquad
q_{g,t}=\widetilde{\lambda}_{g,t}S_t.
\]

Their cumulative incidences satisfy

\[
F_{d,t}=\int_0^t q_{d,\tau}d\tau,
\qquad
F_{g,t}=\int_0^t q_{g,\tau}d\tau,
\qquad
F_{d,t}+F_{g,t}=1-S_t.
\]

The density plots condition on a first jump occurring by the deterministic
simulation horizon \(H\):

\[
f_{d,t}\mid(T_{\mathrm{first}}\leq H)
=\frac{q_{d,t}}{1-S_H},
\qquad
f_{g,t}\mid(T_{\mathrm{first}}\leq H)
=\frac{q_{g,t}}{1-S_H}.
\]

The damage and technology conditional densities therefore integrate jointly
to one over the simulation horizon. The unnormalized curves remain available
as `dmg_jump_subdensity` and `tech_jump_subdensity`.

Conditioning instead on the time of the first jump gives the probability that
the first jump is damage or technology:

\[
\Pr(D\mid T_{\mathrm{first}}=t,X)
=\frac{q_{d,t}}{q_{d,t}+q_{g,t}}
=\frac{\widetilde{\lambda}_{d,t}}
{\widetilde{\lambda}_{d,t}+\widetilde{\lambda}_{g,t}},
\]

\[
\Pr(G\mid T_{\mathrm{first}}=t,X)
=\frac{q_{g,t}}{q_{d,t}+q_{g,t}}
=\frac{\widetilde{\lambda}_{g,t}}
{\widetilde{\lambda}_{d,t}+\widetilde{\lambda}_{g,t}}.
\]

The cumulative probabilities conditional on a first jump by \(H\) are

\[
\Pr(D,\ T_{\mathrm{first}}\leq t\mid T_{\mathrm{first}}\leq H)
=\frac{F_{d,t}}{1-S_H},
\qquad
\Pr(G,\ T_{\mathrm{first}}\leq t\mid T_{\mathrm{first}}\leq H)
=\frac{F_{g,t}}{1-S_H}.
\]

These are saved as `conditional_dmg_jump_prob` and
`conditional_tech_jump_prob`. Their terminal values sum to one.

The exact-time jump-type probabilities are saved separately as
`first_jump_type_dmg_prob` and `first_jump_type_tech_prob`. They sum to one
whenever the total jump intensity is positive.

The deterministic simulation computes these quantities conditional on its
deterministic state path. An ex ante density under stochastic state dynamics
would additionally average the conditional densities across simulated state
paths.

The default deterministic workflow evaluates and compares
\(\xi \in \{148.6, 0.3, 0.1\}\), with \(148.6\) used as the numerical
approximation to the uncertainty-neutral case.

For the submitted four-regime `OneTechJump_Pi_0p0` experiments, the initial
model is `PreDamageIntermTech`. Its current HJB uses only
\(\pi s_g R_t/\varrho\) for the intermediate-to-post technology jump.
Consequently, setting \(\pi=0\) makes its technology first-jump density
identically zero.
