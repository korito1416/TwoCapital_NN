---
title: "The damage belief prices a temperature the economy never reaches"
author: "TwoCapital team"
date: 2026-07-20
---

The reference solution's worst-case belief over the damage-curvature models is set by one marginal value, $\Delta = V(\lambda_3{=}0) - V(\lambda_3{=}1/3)$. By the envelope theorem $\Delta = \tfrac16\langle(Y-\hat y)^2\rangle$, the discounted-average squared temperature overshoot, so the belief is in effect pricing a single temperature: $Y_{\mathrm{imp}} = \hat y + \sqrt{6\Delta}$.

We can then check the belief against the model itself. We **predict** the temperature the economy reaches by integrating the model's own law of motion $dY = \mathcal{E}\,\bar\theta\,dt$ along the reference's simulated emission path, and confront it with the belief's $Y_{\mathrm{imp}}$, year by year ({numref}`fig`). The belief prices $4.04^\circ$C for the whole horizon. But on the reference's own no-shock path the temperature never crosses the $2.5^\circ$C damage threshold — it reaches $2.1^\circ$C at year 60 — and *even if the damage jump fired*, the economy's own emissions carry temperature only to $3.4^\circ$C over the following 60 years. The belief prices a warming the model's dynamics never produce, and one above the $4^\circ$C bound the model places on temperature at all.

:::{figure} figures/byyear.png
:width: 92%
:name: fig
Temperature by year, $\xi = 0.05$. Solid orange: the temperature the reference's damage belief prices, $Y_{\mathrm{imp}} = \hat y + \sqrt{6\Delta} = 4.04^\circ$C. Solid blue: the temperature the economy reaches if the damage jump fires, integrating $dY = \mathcal{E}\bar\theta\,dt$ from the threshold along the reference's own emissions — $3.4^\circ$C after 60 years. Dashed blue: the actual no-shock temperature, which never reaches the $2.5^\circ$C threshold. Dashed grey: the model's $4^\circ$C bound.
:::

The belief is detached from the temperature the solution itself generates: it prices catastrophic-warming damage in an economy whose own dynamics keep temperature far below — a contradiction with the model it solves, not a matter of calibration.
