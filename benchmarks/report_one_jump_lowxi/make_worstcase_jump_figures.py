"""Worst-case (distorted) jump density / cumulative-probability figures for Lars.

One-jump (pi=1) climate NN model, BOTH adjustment-cost calibrations, float64 low-xi run
(logximin = -5.30, covers xi in [0.005, 148.6]).  Damage jump AND technology jump.

CORRECTNESS NOTE (Phase 2/3 diagnosis, verified):
  The raw NN worst-case damage distortion g_l = exp(-(1/xi)(V^l - V)) is RELIABLE only for
  xi >= xi* ~ 0.025.  Below that, a residual ~0.01-0.05 inter-regime value-level error between
  the separately-trained pre- and post-damage value nets (each loss_v ~ 2-4e-3, paper-grade) is
  amplified by 1/xi (=100-200), flipping the worst-realization gap from negative to positive, so
  every g_l < 1 and the worst-case damage jump is SUPPRESSED instead of amplified.  Result: the raw
  NN cumulative damage-jump probability COLLAPSES (0.682 at xi=0.025 -> 0.367 at 0.01 -> 0.0002 at
  0.005), which is economically backwards (more aversion must make the worst case FEAR the jump MORE).

  This is NOT a transform/sign bug and NOT a sim-vs-training inconsistency (both verified).  It is
  irrecoverable from the saved checkpoints without mutually-consistent retraining (deferred).

  We therefore present:
    * xi >= 0.025  -> RELIABLE: the NN worst-case densities/CDFs (solid lines).
    * xi <  0.025  -> UNRELIABLE NN (shown dashed/greyed for transparency) PLUS the analytic
      xi->0 worst-case limit: as xi->0 the planner up-weights g->inf inside the damage region
      [y_lower, y_upper], so the damage jump fires ~immediately on entering -> the cumulative
      worst-case damage-jump probability -> ~1 (a near-degenerate "jump-at-threshold" CDF).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 18, "axes.labelsize": 20, "axes.titlesize": 21,
    "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 15,
    "lines.linewidth": 2.6, "figure.dpi": 130, "savefig.bbox": "tight",
})

ROOT = "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_lowxi_float64"
FULL = ROOT + "/OneTechJump_Pi_1p0_TechIntensityScale_1p0_AdjustmentCostFull_logximin_m5p30_LR_warmup_cosine_10e-6,40e-5_128_neurons_32_#HiddenLayer_4_num_iterations50000/SimulationDeterministic"
HALF = ROOT + "/OneTechJump_Pi_1p0_TechIntensityScale_1p0_AdjustmentCostHalf_Gamma0p12_Theta8p35_logximin_m5p30_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations50000/SimulationDeterministic"
OUT = "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/report_one_jump_lowxi/figures"
os.makedirs(OUT, exist_ok=True)

# damage-jump primitives (params.py): J_n(y) = r1(exp((r2/2)(y-y_lower)^2)-1) 1{y>=y_lower}
R1, R2, Y_LOWER, Y_UPPER = 1.5, 0.36, 1.5, 2.5
XISTAR = 0.025  # reliable floor

# xi label  ,  reliable?
XIS = [
    ("148.600", r"$\xi=\infty$ (neutral)", True),
    ("0.050",   r"$\xi=0.05$",            True),
    ("0.025",   r"$\xi=0.025$ (reliable floor)", True),
    ("0.010",   r"$\xi=0.01$ (NN unreliable)",   False),
    ("0.005",   r"$\xi=0.005$ (NN unreliable)",  False),
]
# neutral grey -> deep red as aversion rises
COLORS = ["0.45", "#3b6fb6", "#2c9c8f", "#e08a1e", "#c0392b"]


def load(base, xi, name):
    p = os.path.join(base, f"SimulationOutputs_ξ_{xi}", name + ".txt")
    return np.loadtxt(p) if os.path.exists(p) else None


def years(base, xi):
    t = load(base, xi, "t")
    return t


def analytic_worstcase_damage_cdf(base, xi, tech_int=None):
    """xi->0 worst-case limit: g -> inf inside [y_lower, y_upper] so the damage hazard saturates.
    Competing-risk cumulative damage-jump probability with a huge in-region damage intensity, using
    the SAME survival accounting as the production pipeline (lambda = lam_dmg + lam_tech)."""
    Y = load(base, xi, "Y")
    t = load(base, xi, "t")
    dt = float(t[1] - t[0])
    if tech_int is None:
        tech_int = load(base, xi, "tech_jump_intensity")
        if tech_int is None:
            tech_int = np.zeros_like(Y)
    BIG = 1e6
    lam_d = np.where(Y > Y_LOWER, BIG, 0.0)
    lam = lam_d + tech_int
    surv = np.exp(-np.cumsum(lam) * dt)          # survival up to and including step
    surv = np.concatenate([[1.0], surv[:-1]])    # survival entering each step
    dmg_sub = lam_d * surv
    cum = np.cumsum(dmg_sub) * dt
    return t, cum


# ============================================================================
# FIGURE 1 — Worst-case DAMAGE-jump cumulative probability (CDF), both calibrations
# ============================================================================
fig, ax = plt.subplots(1, 2, figsize=(17, 7.5), sharey=True)
for c, (base, ctitle) in enumerate([
        (FULL, "Full adjustment cost  ($\\Gamma=0.060,\\ \\theta=16.7$)"),
        (HALF, "Half adjustment cost  ($\\Gamma=0.12,\\ \\theta=8.35$)")]):
    a = ax[c]
    for (xi, lab, ok), col in zip(XIS, COLORS):
        cum = load(base, xi, "dmg_jump_prob")          # unconditional cumulative P(damage jump by t)
        if cum is None:
            continue
        t = years(base, xi)
        st = "-" if ok else "--"
        a.plot(t, cum, st, color=col, label=lab, alpha=1.0 if ok else 0.55)
    # analytic xi->0 worst-case limit (jump fires at threshold) -- one representative path
    t, cum_lim = analytic_worstcase_damage_cdf(base, "0.005")
    a.plot(t, cum_lim, ":", color="k", linewidth=3.0,
           label=r"analytic $\xi\!\to\!0$ limit ($P\!\to\!1$)")
    a.set_title(ctitle, fontsize=18, pad=10)
    a.set_xlabel("Year")
    if c == 0:
        a.set_ylabel("Cumulative worst-case\ndamage-jump probability")
    a.set_ylim(-0.02, 1.02)
    a.grid(alpha=0.25)
    a.legend(loc="upper left", framealpha=0.92)
fig.suptitle("Worst-case (distorted) damage-jump cumulative probability — one-jump ($\\pi=1$) climate model\n"
             "solid = reliable NN ($\\xi\\geq0.025$);  dashed = NN unreliable ($\\xi<0.025$, value-error amplified by $1/\\xi$);  dotted = analytic $\\xi\\!\\to\\!0$ limit",
             fontsize=16, y=1.005)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(OUT, "wc_fig1_damage_cdf.png"))
print("saved wc_fig1_damage_cdf.png")

# ============================================================================
# FIGURE 2 — Worst-case DAMAGE-jump density (sub-density), both calibrations
# ============================================================================
fig, ax = plt.subplots(1, 2, figsize=(17, 7.5), sharey=False)
for c, (base, ctitle) in enumerate([
        (FULL, "Full adjustment cost"),
        (HALF, "Half adjustment cost")]):
    a = ax[c]
    for (xi, lab, ok), col in zip(XIS, COLORS):
        dens = load(base, xi, "dmg_jump_subdensity")
        if dens is None:
            continue
        t = years(base, xi)
        st = "-" if ok else "--"
        a.plot(t, dens, st, color=col, label=lab, alpha=1.0 if ok else 0.55)
    a.axvspan(0, 0, alpha=0)  # placeholder
    a.set_title(ctitle, fontsize=18, pad=10)
    a.set_xlabel("Year")
    if c == 0:
        a.set_ylabel("Worst-case damage-jump\nsub-density  (first-jump)")
    a.grid(alpha=0.25)
    a.legend(loc="upper right", framealpha=0.92)
fig.suptitle("Worst-case (distorted) damage-jump density — one-jump ($\\pi=1$) climate model\n"
             "(as $\\xi\\!\\to\\!0$ the true density degenerates toward a spike at the threshold-crossing $\\sim$yr 17; see CDF figure)",
             fontsize=16, y=1.005)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(OUT, "wc_fig2_damage_density.png"))
print("saved wc_fig2_damage_density.png")

# ============================================================================
# FIGURE 3 — Worst-case TECHNOLOGY-jump cumulative probability + density, both calibrations
# ============================================================================
fig, ax = plt.subplots(2, 2, figsize=(17, 13))
for c, (base, ctitle) in enumerate([
        (FULL, "Full adjustment cost"),
        (HALF, "Half adjustment cost")]):
    for r, (q, ylab, loc) in enumerate([
            ("tech_jump_prob", "Cumulative worst-case\ntech-jump probability", "lower right"),
            ("tech_jump_subdensity", "Worst-case tech-jump\nsub-density  (first-jump)", "upper right")]):
        a = ax[r][c]
        for (xi, lab, ok), col in zip(XIS, COLORS):
            y = load(base, xi, q)
            if y is None:
                continue
            t = years(base, xi)
            # Tech jump g = exp(-(1/xi)(V_breakthrough - V)); V_breakthrough > V (tech is GOOD), so
            # the worst case SHOULD down-weight tech.  But for xi<0.025 the tech CDF is ALSO
            # unreliable: (i) the collapsed damage hazard inflates tech's competing-risk first-jump
            # probability via the shared survival factor (Full), and (ii) the Half-calibration tech
            # intensity itself blows up (value-error up-weighting tech, e.g. 0.12->39.5 at xi=0.005)
            # -- both economically backwards.  Mark xi<0.025 unreliable for tech too.
            st = "-" if ok else "--"
            a.plot(t, y, st, color=col, label=lab, alpha=1.0 if ok else 0.55)
        if r == 0:
            a.set_title(ctitle, fontsize=18, pad=10)
            a.set_ylim(-0.02, 1.02)
        if c == 0:
            a.set_ylabel(ylab)
        if r == 1:
            a.set_xlabel("Year")
        a.grid(alpha=0.25)
        a.legend(loc=loc, framealpha=0.92)
fig.suptitle("Worst-case (distorted) technology-jump probability & density — one-jump ($\\pi=1$) model\n"
             "solid = reliable NN ($\\xi\\geq0.025$);  dashed = unreliable ($\\xi<0.025$): damage-collapse inflates tech's competing-risk CDF (Full) / Half tech intensity blows up",
             fontsize=14, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(OUT, "wc_fig3_tech_prob_density.png"))
print("saved wc_fig3_tech_prob_density.png")

# ============================================================================
# FIGURE 4 — Summary: terminal worst-case jump probability vs xi (the headline)
# ============================================================================
xi_vals = [148.6, 0.050, 0.025, 0.010, 0.005]
xi_keys = ["148.600", "0.050", "0.025", "0.010", "0.005"]
fig, ax = plt.subplots(1, 2, figsize=(17, 7))
for k, (jump, fname, ylab) in enumerate([
        ("damage", "dmg_jump_prob", "Cumulative worst-case\ndamage-jump prob by 60 yr"),
        ("technology", "tech_jump_prob", "Cumulative worst-case\ntech-jump prob by 60 yr")]):
    a = ax[k]
    for base, lab, col, mk in [(FULL, "Full adj. cost", "#c0392b", "o"),
                               (HALF, "Half adj. cost", "#3b6fb6", "s")]:
        P = []
        for xi in xi_keys:
            cum = load(base, xi, fname)
            P.append(cum[-1] if cum is not None else np.nan)
        P = np.array(P)
        rel = np.array([True, True, True, False, False])  # xi>=0.025 reliable for BOTH jump types
        # reliable points solid, unreliable hollow
        a.plot(xi_vals, P, "-", color=col, alpha=0.5, zorder=1)
        a.scatter(np.array(xi_vals)[rel], P[rel], marker=mk, s=110, color=col,
                  label=lab + " (reliable)", zorder=3)
        if (~rel).any():
            a.scatter(np.array(xi_vals)[~rel], P[~rel], marker=mk, s=110,
                      facecolors="none", edgecolors=col, linewidths=2.2,
                      label=lab + " (NN unreliable)", zorder=3)
    if jump == "damage":
        # analytic worst-case floor for small xi -> ~1
        a.axhline(1.0, ls=":", color="k", lw=2.2, label=r"analytic $\xi\!\to\!0$ limit $P\!\to\!1$")
        a.axvspan(min(xi_vals), XISTAR, color="0.85", alpha=0.5, zorder=0)
        a.text(0.0085, 0.5, "NN\nunreliable\n($\\xi<0.025$)", ha="center", va="center",
               fontsize=14, color="0.35")
        if jump == "technology":
            a.axvspan(min(xi_vals), XISTAR, color="0.85", alpha=0.5, zorder=0)
    a.set_xscale("log")
    a.invert_xaxis()
    a.set_xlabel(r"$\xi$  (smaller $\rightarrow$ more robustness aversion)")
    a.set_ylabel(ylab)
    a.set_ylim(-0.02, 1.05)
    a.grid(alpha=0.25)
    a.legend(loc="center left" if jump == "damage" else "upper left", framealpha=0.92, fontsize=13)
fig.suptitle("Worst-case cumulative jump probability by 60 yr vs robustness aversion $\\xi$  (one-jump $\\pi=1$)\n"
             "Reliable for $\\xi\\geq0.025$ (filled, shaded=unreliable).  DAMAGE: monotone-up to 0.63; raw NN collapses below — true limit $\\to1$.  TECH: collapse-contaminated below 0.025",
             fontsize=14, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT, "wc_fig4_terminal_prob_vs_xi.png"))
print("saved wc_fig4_terminal_prob_vs_xi.png")

# ============================================================================
# print the numbers table
# ============================================================================
print("\n=== terminal (60yr) cumulative worst-case jump probabilities ===")
for base, tag in [(FULL, "FULL"), (HALF, "HALF")]:
    print(f"-- {tag} --")
    print(f"{'xi':>9} {'dmg(uncond)':>12} {'dmg(cond)':>11} {'tech(uncond)':>13} {'horizon':>9} reliable")
    for xi, _, ok in XIS:
        d = load(base, xi, "dmg_jump_prob")
        dc = load(base, xi, "conditional_dmg_jump_prob")
        tt = load(base, xi, "tech_jump_prob")
        acc = os.path.join(base, f"SimulationOutputs_ξ_{xi}", "first_jump_density_accounting.txt")
        hor = np.nan
        if os.path.exists(acc):
            for ln in open(acc):
                if "horizon_first_jump_probability" in ln:
                    hor = float(ln.split(":")[1])
        dd = d[-1] if d is not None else np.nan
        dcc = dc[-1] if dc is not None else np.nan
        ttt = tt[-1] if tt is not None else np.nan
        print(f"{xi:>9} {dd:12.4f} {dcc:11.4f} {ttt:13.4f} {hor:9.4f}  {'YES' if ok else 'NO (use analytic limit P->1)'}")
print("\nDone.")
