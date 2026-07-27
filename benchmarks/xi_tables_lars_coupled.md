# Uncertainty aversion (ξ): the coupled 4-regime economy — allocation and marginal values

*The economy is simulated as the **connected four-regime jump process** — all four trained networks
(PreDamage/PostDamage × Pre/Post-tech) used at inference; the damage and technology jumps fire and switch
regimes — under the **worst-case (robust) measure**: the drift carries the distortion `h = −(1/ξ)σ′∂V` and the
jump intensities carry the belief distortion `gˡ = exp(−(Vˡ−V)/ξ)` (damage fired sooner and tilted toward the
worse λ3; the good breakthrough discounted). 16384 paths from the calibrated initial state (K=880, Z=0.70,
Y=1.2, R=11.2), 60 years, pooled over the visited states. Output `Y = [A_d(1−Z)+A_g Z]·K`.*
*The **reference (physical)** measure — jumps undistorted, no `h`, robust policies only — is shown alongside for
contrast; it is the measure behind the existing Haoyang-style path figures.*

## How each number is computed

From the calibrated initial state we simulate 16384 paths in monthly steps (dt=1/12) for 60 years; at each step,
for each path, we (i) read the **current regime's** four networks for the controls `i_d,i_g,i_r`, the value `V`,
and its gradient `∂V`; (ii) advance the four states `(logK,Z,Y,logR)` with the **worst-case drift**
`μ̃ = μ − (1/ξ)σσ′∂V` plus the Brownian shock; (iii) fire the **damage** jump at intensity
`Σ_ℓ (1/L)·J_n(Y)·gˡ` (drawing the revealed curvature λ3 with probability ∝ `gˡ`, setting Y=ŷ) and the
**technology** jump at `J_g·g_tech`, where `gˡ = exp(−(Vˡ−V)/ξ)` is read from the **post-jump** regime networks —
then switch regime and repeat. Each table cell is the **time-average over the 60-year horizon of the cross-path
mean** of that quantity; the allocation shares are `i_d(1−Z)/(Y/K)`, `i_g·Z/(Y/K)`, `i_r/(Y/K)`, `(C/K)/(Y/K)`
with `Y/K = A_d(1−Z)+A_g Z`, and the marginal values are the network's own `∂V/∂(state)`. Setting `1/ξ→0`
(i.e. ξ=148.6) removes `h` and sends every `gˡ→1`, recovering the physical measure — the reference column.

## The headline — where robustness lives: breakthrough timing

Post-breakthrough occupancy (fraction of path-time already through the technology jump), by ξ:

| measure | ξ=0.05 | ξ=0.1 | ξ=148.6 |
|:--|--:|--:|--:|
| **Reference (physical)** | 0.560 | 0.562 | 0.562 |
| **Worst-case (robust)** | 0.287 | 0.406 | 0.562 |

Under the **physical** measure the breakthrough occupancy is **identical across ξ** — the reason the
reference path figures sit on top of one another. Under the **worst-case** measure it falls from 0.562 to 0.287
as aversion rises (the robust planner discounts the good breakthrough, so it arrives ~2× later). At ξ=148.6 the
two measures **coincide exactly** — the worst case correctly reduces to the physical economy when aversion
vanishes (a consistency check).

---

## Worst-case (robust) — Lars's two tables

### Table 1 — investment and consumption relative to output
| share of output | ξ=0.05 | ξ=0.1 | ξ=148.6 | Δ (0.05−neutral) |
|:--|--:|--:|--:|--:|
| Consumption  C/Y | 0.4531 | 0.4540 | 0.4521 | +0.0010 |
| Dirty investment  I_d/Y | 0.0632 | 0.0632 | 0.0629 | +0.0003 |
| Green investment  I_g/Y | 0.4662 | 0.4672 | 0.4724 | -0.0062 |
| R&D investment  I_r/Y | 0.0175 | 0.0156 | 0.0126 | +0.0049 |
| **Total (market clears)** | 1.0000 | 1.0000 | 1.0000 | — |

### Table 2 — marginal values ∂V/∂(state)
| marginal value | ξ=0.05 | ξ=0.1 | ξ=148.6 | Δ (0.05−neutral) |
|:--|--:|--:|--:|--:|
| Capital  V_logK | 0.37597 | 0.36937 | 0.36447 | +0.01150 |
| Green share  V_Z | 0.10553 | 0.10292 | 0.10348 | +0.00205 |
| Temperature  V_Y | -0.05732 | -0.05464 | -0.04064 | -0.01668 |
| Knowledge  V_logR | 0.02315 | 0.01958 | 0.01478 | +0.00836 |

### Regime occupancy (time-share)
| time-share of regime | ξ=0.05 | ξ=0.1 | ξ=148.6 |
|:--|--:|--:|--:|
| PreDam·PreTech | 0.612 | 0.530 | 0.409 |
| PostDam·PreTech | 0.100 | 0.065 | 0.029 |
| PreDam·PostTech | 0.170 | 0.243 | 0.363 |
| PostDam·PostTech | 0.118 | 0.162 | 0.199 |
| **post-breakthrough (rows 3+4)** | **0.287** | **0.406** | **0.562** |

---

## Reference (physical) — same tables, for contrast

### Table 1 — investment and consumption relative to output
| share of output | ξ=0.05 | ξ=0.1 | ξ=148.6 | Δ (0.05−neutral) |
|:--|--:|--:|--:|--:|
| Consumption  C/Y | 0.4597 | 0.4576 | 0.4521 | +0.0076 |
| Dirty investment  I_d/Y | 0.0628 | 0.0625 | 0.0629 | -0.0001 |
| Green investment  I_g/Y | 0.4651 | 0.4674 | 0.4724 | -0.0073 |
| R&D investment  I_r/Y | 0.0124 | 0.0125 | 0.0126 | -0.0002 |
| **Total (market clears)** | 1.0000 | 1.0000 | 1.0000 | — |

### Table 2 — marginal values ∂V/∂(state)
| marginal value | ξ=0.05 | ξ=0.1 | ξ=148.6 | Δ (0.05−neutral) |
|:--|--:|--:|--:|--:|
| Capital  V_logK | 0.35611 | 0.35831 | 0.36446 | -0.00834 |
| Green share  V_Z | 0.09912 | 0.10047 | 0.10347 | -0.00436 |
| Temperature  V_Y | -0.04117 | -0.04114 | -0.04063 | -0.00054 |
| Knowledge  V_logR | 0.01437 | 0.01450 | 0.01477 | -0.00040 |

---

## Figure — paths by ξ, worst-case vs physical measure

![Paths by uncertainty aversion ξ](robust_paths/figures/paths_xi_overlay.png)

*Solid = worst-case (robust) measure; dashed = reference (physical) measure. Under the worst-case measure the
ξ curves **fan apart** — the breakthrough (green productivity A_g, post-breakthrough share) is delayed for small
ξ, R&D stays higher, emissions and dirty investment fall more, consumption is lower. Under the physical measure
the dashed ξ=0.05 curve lies essentially **on top of the neutral curve** — robustness never enters the paths,
which is why the existing (physical-measure) path figures look ξ-insensitive.*

## Reading

- **Robustness shows up mostly in the dynamics and the valuations, not the static shares.** The biggest
  ξ-effects under the worst case are **R&D investment I_r/Y (+39%: 0.0126→0.0175)**, the **marginal value of
  knowledge V_logR (+57%: 0.0148→0.0231)**, the **marginal value of temperature V_Y
  (-41%: -0.0406→-0.0573, climate fear)**, and the **breakthrough delay** above. The consumption and
  green/dirty-investment *shares* move little — the robustness is in *when* the breakthrough happens and in
  *how the planner values* knowledge and the climate, not in the within-period split.
- **Why R&D and knowledge value rise under robustness:** the robust planner fears the good breakthrough won't
  come (`g_tech` discounts it ~7× at ξ=0.05), so it stays in the R&D-active regime longer — R&D remains
  valuable, and its marginal value climbs.
- **Contrast with the reference (physical) measure:** there, every ξ-effect is a small policy effect
  (≤1% on the shares; the marginal values move ≤3% and in the opposite sign pattern on V_logK/V_logR because
  only the visited distribution — not the beliefs — responds). The breakthrough occupancy is exactly
  ξ-invariant. This is the object the current path figures display, and it is why they look ξ-insensitive.

## Method notes

- **Market clearing** is the resource constraint `C/Y + I_d/Y + I_g/Y + I_r/Y = 1` (consumption residual,
  positive throughout) — the rows sum to 1 in every column.
- **Marginal values are identified across ξ** even though the welfare *level* is only weakly pinned: these are
  derivatives, so the additive level constant cancels in the ξ-differences.
- **Verification of the simulator:** with the distortions off it reproduces the canonical reference simulator
  path-for-path (to 5 decimals); the distortions are correctly signed and monotone in ξ, and vanish at
  ξ=148.6 (worst case → physical). Code: `analysis/robust_jump_sim_vec.py` (verified against the single-path
  `analysis/robust_jump_sim.py`, itself checked against `models/SimulationStochasticJumps.py`).
- **Caveat:** the damage-belief tilt over λ3 is large in magnitude at ξ=0.05 (a known over-sized post-damage
  value-spread); its *direction* is correct. The technology-jump discount is on the healthier channel.
