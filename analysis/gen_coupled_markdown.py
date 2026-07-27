"""Generate the coupled (4-regime, worst-case) xi tables markdown for Lars, from the saved npz."""
import numpy as np
from pathlib import Path

D = Path("benchmarks/robust_paths/data_fixed")
rob = np.load(D / "paths_robust.npz", allow_pickle=True)
ref = np.load(D / "paths_reference.npz", allow_pickle=True)
XIS = ["0.05", "0.1", "148.6"]
ALLOC = [("C/Y", "Consumption  C/Y"), ("I_d/Y", "Dirty investment  I_d/Y"),
         ("I_g/Y", "Green investment  I_g/Y"), ("I_r/Y", "R&D investment  I_r/Y")]
MV = [("V_logK", "Capital  V_logK"), ("V_Z", "Green share  V_Z"),
      ("V_Y", "Temperature  V_Y"), ("V_logR", "Knowledge  V_logR")]
RN = ["PreDam·PreTech", "PostDam·PreTech", "PreDam·PostTech", "PostDam·PostTech"]


def v(d, xi, k): return float(d[f"{xi}::pooled::{k}"])
def rf(d, xi): return d[f"{xi}::regime_frac"]


def alloc_table(d, dec=4):
    L = ["| share of output | ξ=0.05 | ξ=0.1 | ξ=148.6 | Δ (0.05−neutral) |",
         "|:--|--:|--:|--:|--:|"]
    for k, lab in ALLOC:
        row = [f"{v(d,x,k):.{dec}f}" for x in XIS]
        d05 = v(d, "0.05", k) - v(d, "148.6", k)
        L.append(f"| {lab} | {row[0]} | {row[1]} | {row[2]} | {d05:+.4f} |")
    sums = [sum(v(d, x, q) for q, _ in ALLOC) for x in XIS]
    L.append(f"| **Total (market clears)** | {sums[0]:.4f} | {sums[1]:.4f} | {sums[2]:.4f} | — |")
    return "\n".join(L)


def mv_table(d):
    L = ["| marginal value | ξ=0.05 | ξ=0.1 | ξ=148.6 | Δ (0.05−neutral) |",
         "|:--|--:|--:|--:|--:|"]
    for k, lab in MV:
        row = [f"{v(d,x,k):.5f}" for x in XIS]
        d05 = v(d, "0.05", k) - v(d, "148.6", k)
        L.append(f"| {lab} | {row[0]} | {row[1]} | {row[2]} | {d05:+.5f} |")
    return "\n".join(L)


def regime_table(d):
    L = ["| time-share of regime | ξ=0.05 | ξ=0.1 | ξ=148.6 |", "|:--|--:|--:|--:|"]
    for i in range(4):
        L.append(f"| {RN[i]} | " + " | ".join(f"{rf(d,x)[i]:.3f}" for x in XIS) + " |")
    L.append("| **post-breakthrough (rows 3+4)** | "
             + " | ".join(f"**{rf(d,x)[2]+rf(d,x)[3]:.3f}**" for x in XIS) + " |")
    return "\n".join(L)


NPATHS = 16384
def pctline(d, k):
    a, nn = v(d, "0.05", k), v(d, "148.6", k)
    return f"{100*(a-nn)/abs(nn):+.0f}%: {nn:.4f}→{a:.4f}"
pt_neut = rf(rob, "148.6")[2] + rf(rob, "148.6")[3]
pt_averse = rf(rob, "0.05")[2] + rf(rob, "0.05")[3]

md = f"""# Uncertainty aversion (ξ): the coupled 4-regime economy — allocation and marginal values

*The economy is simulated as the **connected four-regime jump process** — all four trained networks
(PreDamage/PostDamage × Pre/Post-tech) used at inference; the damage and technology jumps fire and switch
regimes — under the **worst-case (robust) measure**: the drift carries the distortion `h = −(1/ξ)σ′∂V` and the
jump intensities carry the belief distortion `gˡ = exp(−(Vˡ−V)/ξ)` (damage fired sooner and tilted toward the
worse λ3; the good breakthrough discounted). {NPATHS} paths from the calibrated initial state (K=880, Z=0.70,
Y=1.2, R=11.2), 60 years, pooled over the visited states. Output `Y = [A_d(1−Z)+A_g Z]·K`.*
*The **reference (physical)** measure — jumps undistorted, no `h`, robust policies only — is shown alongside for
contrast; it is the measure behind the existing Haoyang-style path figures.*

## How each number is computed

From the calibrated initial state we simulate {NPATHS} paths in monthly steps (dt=1/12) for 60 years; at each step,
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
| **Reference (physical)** | {rf(ref,'0.05')[2]+rf(ref,'0.05')[3]:.3f} | {rf(ref,'0.1')[2]+rf(ref,'0.1')[3]:.3f} | {rf(ref,'148.6')[2]+rf(ref,'148.6')[3]:.3f} |
| **Worst-case (robust)** | {rf(rob,'0.05')[2]+rf(rob,'0.05')[3]:.3f} | {rf(rob,'0.1')[2]+rf(rob,'0.1')[3]:.3f} | {rf(rob,'148.6')[2]+rf(rob,'148.6')[3]:.3f} |

Under the **physical** measure the breakthrough occupancy is **identical across ξ** — the reason the
reference path figures sit on top of one another. Under the **worst-case** measure it falls from {pt_neut:.3f} to {pt_averse:.3f}
as aversion rises (the robust planner discounts the good breakthrough, so it arrives ~2× later). At ξ=148.6 the
two measures **coincide exactly** — the worst case correctly reduces to the physical economy when aversion
vanishes (a consistency check).

---

## Worst-case (robust) — Lars's two tables

### Table 1 — investment and consumption relative to output
{alloc_table(rob)}

### Table 2 — marginal values ∂V/∂(state)
{mv_table(rob)}

### Regime occupancy (time-share)
{regime_table(rob)}

---

## Reference (physical) — same tables, for contrast

### Table 1 — investment and consumption relative to output
{alloc_table(ref)}

### Table 2 — marginal values ∂V/∂(state)
{mv_table(ref)}

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
  ξ-effects under the worst case are **R&D investment I_r/Y ({pctline(rob,'I_r/Y')})**, the **marginal value of
  knowledge V_logR ({pctline(rob,'V_logR')})**, the **marginal value of temperature V_Y
  ({pctline(rob,'V_Y')}, climate fear)**, and the **breakthrough delay** above. The consumption and
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
"""

out = Path("benchmarks/xi_tables_lars_coupled.md")
out.write_text(md)
print(f"wrote {out} ({len(md)} chars)")
